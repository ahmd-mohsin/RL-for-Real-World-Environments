"""End-to-end smoke test of the built modules with the cheapest tool-calling model (``smoke-cheap`` in crucible.toml):
does every layer still work together on a real LLM, for cents?

  M1 world      start the helpdesk World and audit it
  M2 actors     an LLM agent works ticket 2 against record-grounded customers and a supervisor bot
  M3 connectors redact + store the episode trace (idempotent)
  M4 signals    extract signals from a stored coding PR trace (the coding mapper; no LLM)
  M5 workflow   segment the episode; Φ and stage over helpdesk SQL milestones from the verifier view
  M6 credit     the gated reward and R-properties on the episode end state; one LLM-judge pass on its prefix
  harness       BashLoopPolicy (native harness) solves a small shell task in a workspace World
  M7 verify     the helpdesk task's hidden checks (kit invariants + predicates) grade the LLM episode; reward files written
  M8 forge      world_synth forges a 5-task variant world (LLM Architect narrative on the cheap model) and M9 accepts it

``python -m crucible_pack_toy.smoke [budget_usd]`` writes ``docs/modules/reports/smoke_cheap.json``. Correctness of the
cheap model's *answers* is not the point (it is weak); every layer running and producing well-formed output is.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from crucible.actors.grounded import GroundedResponder
from crucible.actors.harness import BashLoopPolicy
from crucible.actors.llm import LLMClient
from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
from crucible.actors.records import RecordStore
from crucible.actors.scripted import RelayActor, Rule
from crucible.connectors.redact import RedactionPolicy, redact
from crucible.connectors.trace_store import TraceStore
from crucible.core.schema.milestone import Expectation, Milestone, MilestoneCheck
from crucible.core.schema.state import Query
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.credit.compose import compose
from crucible.credit.judge import judge, render_prefix
from crucible.workflow.milestones.evaluate import evaluate
from crucible.workflow.potential import potential
from crucible.workflow.steps import segment
from crucible.world.audit import audit_world
from crucible.world.world import World
from crucible_pack_toy import INSTRUCTION, PACK_ROOT, helpdesk_spec

REPORT = PACK_ROOT.parent.parent / "docs" / "modules" / "reports" / "smoke_cheap.json"
FOCUS = (" Work on ticket 2 only: confirm the duplicate charge with the customer by calling send_message (to: customer_lee), "
         "refund the duplicate per the refund policy, add a note and resolve the ticket.")

MILESTONES = [  # helpdesk SQL milestones for ticket 2 (the expert end state of the M2 toy gate)
    Milestone(id="sql/ticket2_note", check=MilestoneCheck(app="tickets", query=Query(op="scalar", args={"sql": "SELECT COUNT(*) FROM notes WHERE ticket_id = 2"}),
                                                          expect=Expectation(kind="between", lo=1))),
    Milestone(id="sql/ticket2_refund", check=MilestoneCheck(app="tickets", query=Query(op="scalar", args={"sql": "SELECT COUNT(*) FROM refunds WHERE ticket_id = 2"}),
                                                            expect=Expectation(kind="between", lo=1))),
    Milestone(id="sql/ticket2_resolved", requires=["sql/ticket2_refund"],
              check=MilestoneCheck(app="tickets", query=Query(op="scalar", args={"sql": "SELECT status FROM tickets WHERE id = 2"}), expect=Expectation(kind="equals", value="resolved"))),
]


async def step(name: str, report: dict[str, Any], fn) -> Any:
    t0 = time.time()
    try:
        out = await fn()
        report[name] = {"ok": True, "seconds": round(time.time() - t0, 2), **(out if isinstance(out, dict) else {"result": out})}
        return out
    except Exception as e:  # noqa: BLE001 -- a smoke test reports every failure and keeps going
        report[name] = {"ok": False, "seconds": round(time.time() - t0, 2), "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-1500:]}
        return None


async def run(budget: float = 0.10) -> dict[str, Any]:
    llm = LLMClient("smoke-cheap", budget_usd=budget, cache_mode="off")
    rep: dict[str, Any] = {"model": llm.model}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        w = World(helpdesk_spec(), instructions={"agent": INSTRUCTION + FOCUS}, store=Store(d / "s"), workdir=d / "w", pack_root=PACK_ROOT)
        records = RecordStore.from_jsonl(PACK_ROOT / "data" / "customers.jsonl")
        w.attach(GroundedResponder("customer_lee", records, mode="verbatim", roles={"customer_lee"}, min_score=0.3))
        w.attach(GroundedResponder("customer_dana", records, mode="verbatim", roles={"customer_dana"}, min_score=0.5))
        w.attach(RelayActor("supervisor", [Rule(r"approv\w*.*?(\d+(?:\.\d+)?)", reply="Approved refund of {0} USD. - Sam Lee")], default="Please state the ticket, order and amount."))

        async def m1():
            await w.start()
            a = await audit_world(w)
            return {"audit_ok": a.ok, "checks": {c.id: c.status for c in a.checks}}

        await step("M1_world", rep, m1)
        state: dict[str, Any] = {}

        async def m2():
            agent = LLMToolPolicy("agent", llm, temperature=0.2)
            res = await run_episode(w, {"agent": agent}, limits=EpisodeLimits(max_steps=20))
            state["res"] = res
            calls = [e.data.get("function") for e in res.trace.events if e.actor == "agent" and e.type == "tool_call"]
            return {"steps": res.steps, "stop": res.stop_condition, "masked": res.masked, "tool_calls": calls[:20], "events": len(res.trace.events)}

        await step("M2_actors_llm_episode", rep, m2)

        async def m3():
            tr = redact(state["res"].trace, RedactionPolicy())
            ts = TraceStore(Store(d / "traces"))
            k1, k2 = ts.put(tr), ts.put(tr)
            return {"stored": k1 == k2, "events": len(tr.events)}

        await step("M3_connectors_ingest", rep, m3)

        async def m4():
            from crucible.signals.extract import extract  # the coding mapper resolves through the registry

            store = TraceStore(Store(PACK_ROOT.parent.parent / "data" / "m4"))
            ids = store.ids()
            if not ids:
                return {"skipped": "no stored PR traces (data/m4)"}
            t = extract(store.get(ids[0]))
            return {"trace": ids[0], "signals": len(t.signals), "types": sorted({s.type for s in t.signals})}

        await step("M4_signals", rep, m4)

        async def m5():
            st = segment(state["res"].trace)
            phis = await evaluate(MILESTONES, w.verifier_view())
            return {"steps": len(st), "kinds": sorted({s.kind for s in st}), "phis": phis, "phi": round(potential(MILESTONES, phis)[0], 3)}

        out5 = await step("M5_workflow", rep, m5)

        async def m6():
            phis = (out5 or {}).get("phis") or {}
            v = w.verifier_view()
            from crucible.workflow.milestones.evaluate import decode

            refunds = decode((await v.query("tickets", {"op": "sql", "args": {"sql": "SELECT amount_usd, approved_by FROM refunds WHERE ticket_id = 2"}})).value)["rows"]
            unapproved = any(float(r[0] if isinstance(r, list | tuple) else r["amount_usd"]) > 50 and not (r[1] if isinstance(r, list | tuple) else r["approved_by"])
                             for r in refunds)
            rj = compose(gates={"refund_policy": not unapproved}, outcome=phis.get("sql/ticket2_resolved") == 1.0 and phis.get("sql/ticket2_refund") == 1.0,
                         phi=potential(MILESTONES, phis)[0])
            msgs = [{"role": "user", "content": INSTRUCTION + FOCUS}] + [{"role": "assistant", "content": e.text or str(e.data.get("function"))}
                                                                          for e in state["res"].trace.events if e.actor == "agent"][:10]
            j = await judge(llm, render_prefix(msgs), passes=1, max_tokens=600, key="smoke")
            return {"reward": rj.reward, "gates": rj.gates, "outcome": rj.outcome, "progress": rj.progress, "judge_p_success": j.p_success, "judge_error": j.error}

        await step("M6_credit", rep, m6)

        async def m7():
            from crucible.verify.grader import grade_episode
            from crucible.verify.runtime.plan import Context
            from crucible.verify.runtime.trusted import Trusted
            from crucible_pack_toy.verify import plan as toy_plan

            out = await grade_episode(toy_plan(), Context(Trusted(d / "grade"), params={"view": w.verifier_view()}), out_dir=d / "verifier",
                                      milestones=MILESTONES, series=[(out5 or {}).get("phis") or {}])
            flat = json.loads((d / "verifier" / "reward.json").read_text())
            return {"reward": out.reward.reward, "gates": out.reward.gates, "failing": out.report.failing_gates(), "flat_numeric": all(isinstance(v, float) for v in flat.values()),
                    "reward_txt": (d / "verifier" / "reward.txt").read_text().strip()}

        await step("M7_verify", rep, m7)
        await w.close()

        async def harness():
            spec = WorldSpec.model_validate({"world": "smoke/harness", "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "ws"}},
                                                                                 "params": {"shell": True, "allow_unsafe_local_shell": True}}],
                                             "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}]})
            hw = World(spec, store=Store(d / "hs"), workdir=d / "hw",
                       instructions={"agent": "Create a file fib.py that prints the first 10 Fibonacci numbers (0 1 1 2 3 5 8 13 21 34) one per line, run it to check, then submit."})
            await hw.start()
            try:
                pol = BashLoopPolicy("agent", llm, temperature=0.2, max_tokens=800)
                res = await run_episode(hw, {"agent": pol}, limits=EpisodeLimits(max_steps=12))
                out = await hw.verifier_view().query("ws", Query(op="run", args={"cmd": "python3 fib.py 2>&1 | tr '\\n' ' '"}))
                return {"steps": res.steps, "stop": pol.stop or res.stop_condition, "format_errors": pol.format_errors,
                        "fib_output": (out.stdout or "").strip()[:120], "correct": (out.stdout or "").split() == ["0", "1", "1", "2", "3", "5", "8", "13", "21", "34"]}
            finally:
                await hw.close()

        await step("harness_bash_loop", rep, harness)

        async def m8():
            from crucible_pack_toy.forge.world_synth import forge

            r = await forge(seed=11, llm_model=None, out=d / "forge")
            # one LLM role on the cheap model: the Architect's narrative, validated deterministically (literals and graded sentences verbatim)
            from crucible.forge.schema import Architecture
            from crucible_pack_toy.forge.scenario import LLMNarrativeArchitect, architect_validator

            arch = await LLMNarrativeArchitect(llm).run({"signals": r["signals"]}, feedback=[], knobs={"seed": 11, "world_id": "toy/helpdesk_v2"})
            findings = architect_validator(Architecture.model_validate(arch.model_dump()), {})
            return {"state": r["state"], "accepted": r.get("summary", {}).get("accepted"), "tasks": r.get("summary", {}).get("total"),
                    "audit_sound": r.get("audit", {}).get("sound"), "llm_narrative_findings": findings[:3], "sample_instruction": arch.tasks[1].instruction[:220]}

        await step("M8_forge", rep, m8)
    rep["spend_usd"] = round(llm.spend.usd, 6)
    rep["calls"] = llm.spend.calls
    rep["all_layers_ran"] = all(v.get("ok") for k, v in rep.items() if isinstance(v, dict) and "ok" in v)
    return rep


def main() -> None:
    rep = asyncio.run(run(float(sys.argv[1]) if len(sys.argv) > 1 else 0.10))
    REPORT.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ("traceback", "checks")} if isinstance(v, dict) else v) for k, v in rep.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
