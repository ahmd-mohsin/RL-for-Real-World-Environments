"""G-world_synth for the toy pack: a variant helpdesk world with five tasks, forged by the swarm and accepted by M9.

    python -m crucible_pack_toy.forge.world_synth [--seed N] [--llm MODEL --budget USD] [--out DIR]

1. **Miner** (deterministic): reads the toy ticket traces (M3 store), extracts signals with the toy mapper (M4), and
   picks the approval trap's signal: a supervisor's ``violation`` flag ``refund_over_50_needs_approval``.
2. The orchestrator runs the roles as a DAG (``crucible.forge.orchestrator``):
   ``architect -> env -> (oracle || checks) -> negatives -> audit``. The Test writer is staged *without* the oracle,
   the oracle carries a canary, and any output that holds material its node could not see fails the plan.
3. The forged pack root is written (world yaml, seed data, decoy policy, tools) and every task is assembled:
   ``crucible/task.yaml`` + ``card.yaml`` (forge record with each role's visible inputs), ``checks.json``,
   ``negatives/*.json``, ``solution/actions.jsonl``, ``instruction.md``.
4. Each task goes through the M9 subset (A1 A2 A3 A4-lite A5 A7 A8) in fresh Worlds.

Scripted roles run with 0 paid calls (CI). ``--llm`` swaps in the LLM Architect (narrative) and LLM Test writer and
reports their acceptance outcome; LLM spend is capped by the forge ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from crucible.accept.contract import accept, summarize
from crucible.connectors.trace_store import TraceStore
from crucible.core.schema.task import CrucibleTask, Difficulty, ForgeRecord, Grounding, Negative, Splits, TaskCard
from crucible.core.store import Store
from crucible.forge.budget import Ledger
from crucible.forge.orchestrator import Journal, Node, Orchestrator, Plan
from crucible.forge.schema import Architecture, AuditReport, CheckSets, EnvBuild, Finding, NegativeSet, NegativeSpec, OracleSet
from crucible.signals.extract import extract
from crucible_pack_toy import PACK_ROOT
from crucible_pack_toy.forge.env import EnvBuilder, env_validator, write_pack
from crucible_pack_toy.forge.roles import ScriptedAdversary, ScriptedOracleBuilder, ScriptedTestWriter, audit, checks_validator
from crucible_pack_toy.forge.scenario import LLMNarrativeArchitect, ScriptedArchitect, architect_validator
from crucible_pack_toy.forge.task import ToyForgedTask

CRUCIBLE = PACK_ROOT.parent.parent
M3 = CRUCIBLE / "data" / "m3" / "store"
REPORT = CRUCIBLE / "docs" / "modules" / "reports"


def mine_signals() -> tuple[dict[str, str], set[str]]:
    """The Miner: signal ids the Architect may build traps on, plus every resolvable toy signal id (for A8)."""
    ts = TraceStore(Store(M3))
    approval, known = [], set()
    for tid in sorted(i for i in ts.ids() if i.startswith("toy/support_ticket:")):
        for s in extract(ts.get(tid)).signals:
            known.add(s.id)
            if s.type == "violation" and s.meta.get("rule") == "refund_over_50_needs_approval":
                approval.append(s.id)
    if not approval:
        raise SystemExit("no refund-approval violation signal in the toy traces (run the M3 toy ingest first)")
    return {"approval": min(approval)}, known


class AuditorRole:
    name, version = "auditor", "toy-1"

    async def run(self, staged: dict[str, Any], *, feedback: list[str], knobs: dict[str, Any]) -> AuditReport:
        f = audit(Architecture.model_validate(staged["architecture"]), EnvBuild.model_validate(staged["env"]), CheckSets.model_validate(staged["checks"]),
                  NegativeSet.model_validate(staged["negatives"]))
        return AuditReport(findings=[Finding(**x) for x in f])


def build_plan(seed: int, signals: dict[str, str], *, world_id: str) -> Plan:
    sig = "=" + json.dumps(signals, sort_keys=True)
    return Plan(task_id=f"{world_id}@seed{seed}", generator="world_synth", knobs={"seed": seed, "world_id": world_id, "distractor_rows": 6, "decoy_docs": 1},
                nodes=[
                    Node("architect", "architect", {"signals": sig}, validators=[architect_validator], estimate_usd=0.01),
                    Node("env", "env_builder", {"architecture": "@architect"}, validators=[env_validator]),
                    Node("oracle", "oracle_builder", {"architecture": "@architect", "env": "@env"}, canary_out=True),
                    # the Test writer is staged without the oracle: it cannot read what it must not
                    Node("checks", "test_writer", {"architecture": "@architect", "env": "@env", "signals": sig}, validators=[checks_validator], estimate_usd=0.02),
                    Node("negatives", "adversary", {"architecture": "@architect", "oracles": "@oracle", "checks": "@checks", "signals": sig}),
                    Node("audit", "auditor", {"architecture": "@architect", "env": "@env", "checks": "@checks", "negatives": "@negatives"}),
                ])


def assemble(out_root: Path, arch: Architecture, env: EnvBuild, oracles: OracleSet, checks: CheckSets, negs: NegativeSet, journal: Journal, plan: Plan) -> list[Path]:
    write_pack(env, out_root)
    world_file = f"worlds/{arch.world_id.split('/')[-1]}.yaml"
    roles = [{"role": e["role"], "node": e["node"], "visible": e.get("visible"), "input_keys": e.get("input_keys"), "cost_usd": e.get("cost_usd")}
             for e in (json.loads(x) for x in journal.path.read_text().splitlines()) if e.get("event") == "node" and e.get("status") == "done"]
    dirs = []
    by_o = {o.task_id: o for o in oracles.oracles}
    by_c = {c.task_id: c for c in checks.sets}
    for t in arch.tasks:
        key = t.task_id.split("/")[-1]
        d = out_root / "tasks" / key
        if d.exists():
            shutil.rmtree(d)
        (d / "crucible" / "negatives").mkdir(parents=True)
        (d / "solution").mkdir()
        (d / "instruction.md").write_text(t.instruction + "\n")
        (d / "solution" / "actions.jsonl").write_text("".join(json.dumps(a) + "\n" for a in by_o[t.task_id].actions))
        (d / "crucible" / "checks.json").write_text(by_c[t.task_id].model_dump_json(indent=1) + "\n")
        mine = [n for n in negs.negatives if n.task_id == t.task_id]
        for n in mine:
            (d / "crucible" / "negatives" / f"{n.id.split('/')[-1]}.json").write_text(n.model_dump_json(indent=1) + "\n")
        task = CrucibleTask(id=f"toy/helpdesk/{key}@1", family="toy/helpdesk", horizon=t.horizon, world=f"../../{world_file}", milestones=None,
                            oracle="solution/actions.jsonl", oracle_kind="actions",
                            negatives=[Negative(id=n.id.split("/")[-1], path=f"crucible/negatives/{n.id.split('/')[-1]}.json", expected_gate=n.expected_gate,
                                                kind=n.kind, expected=n.expected, provenance=n.provenance, description=n.description) for n in mine],
                            derived_from=f"{arch.world_id}@seed{arch.scenario['seed']}")
        card = TaskCard(id=task.id, family=task.family, horizon=t.horizon, title=t.title,
                        grounding=Grounding(traces=[p for p in t.provenance if p.startswith("sig_")], oracle_source="builder_oracle",
                                            checks_with_provenance=f"{sum(1 for c in by_c[t.task_id].checks if c.provenance)}/{len(by_c[t.task_id].checks)}"),
                        splits=Splits(train=False), forge=ForgeRecord(generator=plan.generator, knobs=plan.knobs, seed_ref=plan.task_id, roles=roles, state="constructed"),
                        difficulty=Difficulty(**{k: v for k, v in t.difficulty.items() if k in Difficulty.model_fields}))
        (d / "crucible" / "task.yaml").write_text(yaml.safe_dump(task.model_dump(mode="json", exclude_none=True), sort_keys=False))
        (d / "crucible" / "card.yaml").write_text(yaml.safe_dump(card.model_dump(mode="json", exclude_none=True), sort_keys=False))
        dirs.append(d)
    return dirs


async def forge(seed: int = 7, *, llm_model: str | None = None, budget: float = 0.05, out: Path | None = None, world_id: str = "toy/helpdesk_v2",
                collect: list | None = None) -> dict:
    signals, known = mine_signals()
    out = Path(out or CRUCIBLE / "data" / "forge" / f"{world_id.replace('/', '_')}_s{seed}{'_llm' if llm_model else ''}")
    out.mkdir(parents=True, exist_ok=True)
    store = Store(out / "store")
    llms = []
    roles: dict[str, Any] = {"architect": ScriptedArchitect(), "env_builder": EnvBuilder(), "oracle_builder": ScriptedOracleBuilder(),
                             "test_writer": ScriptedTestWriter(), "adversary": ScriptedAdversary(), "auditor": AuditorRole()}
    if llm_model:
        from crucible.actors.llm import LLMClient
        from crucible_pack_toy.forge.roles import LLMTestWriter

        a, tw = LLMClient(llm_model, budget_usd=budget), LLMClient(llm_model, budget_usd=budget)
        llms = [a, tw]
        roles["architect"], roles["test_writer"] = LLMNarrativeArchitect(a), LLMTestWriter(tw)
    ledger = Ledger(cap_usd=budget, observe=lambda: sum(c.spend.usd for c in llms))
    orch = Orchestrator(store, Journal(out / "journal.jsonl"), ledger, roles)
    plan = build_plan(seed, signals, world_id=world_id)
    res = await orch.run_plan(plan)
    rep: dict[str, Any] = {"plan": plan.task_id, "state": res.state, "reason": res.reason, "llm": llm_model,
                           "nodes": {k: {"status": v.status, "attempts": v.attempts, "reused": v.reused, "findings": v.findings[:5], "cost_usd": round(v.cost_usd, 6)}
                                     for k, v in res.nodes.items()}, "signals": signals, "spend_usd": round(sum(c.spend.usd for c in llms), 6)}
    if res.state != "constructed":
        return rep
    o = res.outputs
    arch, env = Architecture.model_validate(o["architect"]), EnvBuild.model_validate(o["env"])
    oracles = OracleSet.model_validate({k: v for k, v in o["oracle"].items() if k != "_canary"})
    checks, negs = CheckSets.model_validate(o["checks"]), NegativeSet.model_validate(o["negatives"])
    aud = AuditReport.model_validate(o["audit"])
    rep["audit"] = {"sound": aud.sound, "findings": [f.model_dump() for f in aud.findings]}
    dirs = assemble(out, arch, env, oracles, checks, negs, orch.journal, plan)
    rep["tasks_dir"] = str(out / "tasks")
    world_file = f"worlds/{arch.world_id.split('/')[-1]}.yaml"
    reports = []
    by_c = {c.task_id: c for c in checks.sets}
    by_o = {x.task_id: x for x in oracles.oracles}
    for t in arch.tasks:
        mine = [NegativeSpec.model_validate(n.model_dump()) for n in negs.negatives if n.task_id == t.task_id]
        task = ToyForgedTask(t, by_c[t.task_id], by_o[t.task_id].actions, mine, pack_root=out, world_file=world_file, scenario=arch.scenario,
                             canary=o["oracle"].get("_canary"))
        if collect is not None:  # M9: the constructed tasks themselves, for the full contract
            collect.append((task, lambda s: s in known))
            continue
        r = await accept(task, resolve=lambda s: s in known)
        reports.append(r)
        state = "accepted" if r.ok and aud.sound else "rejected"
        orch.set_state(t.task_id, state, reason=None if state == "accepted" else str([c.id for c in r.checks if c.status == "fail"]), task_hash=task.task_hash)
    rep["accept"] = {r.task_id: r.to_dict() for r in reports}
    rep["summary"] = summarize(reports)
    rep["pass"] = aud.sound and rep["summary"]["accepted"] == len(arch.tasks) == 5
    rep["spend_usd"] = round(sum(c.spend.usd for c in llms), 6)
    rep["task_dirs"] = [str(d) for d in dirs]
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--llm")
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rep = asyncio.run(forge(a.seed, llm_model=a.llm, budget=a.budget, out=a.out))
    name = f"m8_toy_world_synth{'_llm' if a.llm else ''}.json"
    (REPORT / name).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"pass": rep.get("pass"), "state": rep["state"], "audit": rep.get("audit", {}).get("sound"), "summary": rep.get("summary"),
                      "nodes": {k: (v["status"], v["attempts"]) for k, v in rep["nodes"].items()}, "spend_usd": rep.get("spend_usd")}, indent=1))


if __name__ == "__main__":
    main()
