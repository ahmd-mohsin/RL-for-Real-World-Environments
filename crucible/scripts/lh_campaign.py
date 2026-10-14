"""M16 campaign gaps (docs/research/long-horizon-rlm/07-campaign.md): child agents, a real model, a scale probe.

    python scripts/lh_campaign.py children --out children.json          # delegating scripted models, rungs x depth x mode
    python scripts/lh_campaign.py adversarial --out adversarial.json    # over-spawn, depth, least privilege, finish with children running
    python scripts/lh_campaign.py qwen --out qwen.json                  # Qwen2.5-0.5B-Instruct (HF, MPS/CPU) in the Docker kernel
    python scripts/lh_campaign.py scale --out scale.json                # BacklogRLM faithful at a 10,000-call rung, default config

Every run uses the Docker kernel, the M10 Runner with ``rollout_mode="train"`` and (scripted runs) exact Qwen2.5 TITO.
``analyze_attempt`` reads one attempt directory and reports: reward, lost constraints after the root's first compaction,
tree usage against ``TreeLimits``, every child's segments and compactions, train-validity of the rollout record, the
trainer's forwarded tokens against the tokens that entered each chain, whether every World step is attributed to the
model call (and agent) that made it, the root chain's credit sum against the reward (Φ telescopes), and the credit the
positional progress attribution gives each root call against an exact attribution by tool-call id.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

OWNER = "crucible.rlm.owner"


# ---- containers ------------------------------------------------------------------------------------------------------------
def rlm_containers(mine: bool = True) -> list[str]:
    """RLM kernel containers this process started and did not remove (the launcher labels each with its owner, host:pid);
    ``mine=False``: every RLM container on the machine (other agents' runs included)."""
    from crucible.actors.rlm.kernel import owner_id

    flt = f"label={OWNER}={owner_id()}" if mine else f"label={OWNER}"
    try:
        out = subprocess.run(["docker", "ps", "-a", "--filter", flt, "--format", "{{.Names}}"], capture_output=True, text=True,
                             timeout=30, check=False).stdout
    except Exception:  # noqa: BLE001 -- no docker: nothing to count
        return []
    return [x for x in out.split() if x]


# ---- analysis ------------------------------------------------------------------------------------------------------------
def entered_tokens(ro: Any) -> tuple[int, int]:
    """(tokens that entered each chain: last call's full prompt + completion per chain; trainable nodes' own tokens)
    (the B7 measure of ``tests/unit/test_m16_d_records.py``)."""
    from crucible.core.schema.rollout import full_prompts

    fp = full_prompts(ro)
    last: dict[Any, int] = {}
    node = 0
    for t in ro.turns:
        if t.prompt_token_ids is None and not t.prompt_is_delta and t.call_id not in fp:
            continue
        n = len(fp.get(t.call_id) or []) + len(t.completion_token_ids or [])
        if t.node():
            node += n if t.trainable else 0
        else:
            last[t.chain()] = n
    return sum(last.values()), node


def forwarded_tokens(ro: Any) -> dict[str, Any]:
    from crucible.train.bridges.common import flatten

    seqs = flatten(ro, seq_adv=1.0)
    fwd = sum(len(s.prompt_ids) + len(s.response_ids) for s in seqs if any(s.response_mask))
    entered, nodes = entered_tokens(ro)
    return {"forwarded": fwd, "entered": entered, "nodes": nodes, "ratio": round(fwd / max(1, entered + nodes), 4), "sequences": len(seqs),
            "sequences_by_agent": {a or "root": sum(1 for s in seqs if (s.agent or "") == a) for a in sorted({s.agent or "" for s in seqs})}}


def attribution(adir: Path, calls: list[dict[str, Any]], events: list[Any]) -> dict[str, Any]:
    """Is every World step attributed to the model call (and agent) that made it?"""
    owner: dict[str, dict[str, Any]] = {}
    dup = 0
    for c in calls:
        for a in c.get("actions") or []:
            if a.startswith("tool:"):
                tid = a[5:]
                dup += tid in owner
                owner[tid] = c
    trace_ids = [(e.data or {}).get("id") for e in events if e.actor == "agent" and e.kind == "action" and e.type == "tool_call"]
    missing = [t for t in trace_ids if t not in owner]
    by_agent: dict[str, int] = {}
    for t in trace_ids:
        if t in owner:
            ag = owner[t].get("agent") or "root"
            by_agent[ag] = by_agent.get(ag, 0) + 1
    # each agent's own ledger: its world_call events name the model call they ran under; that call must be the agent's
    calls_by_id = {c["call_id"]: c for c in calls}
    root = adir / "rlm" / "agent" / "root"
    wrong, checked, ledger_counts = 0, 0, {}
    for ev_file in root.rglob("events.jsonl"):
        rel = ev_file.parent.relative_to(root).as_posix()
        agent = "" if rel == "." else rel
        n = 0
        for line in ev_file.read_text().splitlines():
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("kind") != "world_call":
                continue
            n += 1
            c = calls_by_id.get(ev.get("call_id"))
            checked += 1
            wrong += c is None or (c.get("agent") or "") != agent
        ledger_counts[agent or "root"] = n
    return {"world_tool_calls": len(trace_ids), "unattributed": len(missing), "double_attributed": dup, "by_agent": by_agent,
            "ledger_world_calls": ledger_counts, "ledger_checked": checked, "ledger_agent_mismatch": wrong,
            "attributed_ok": not missing and not dup and not wrong and by_agent == {k: v for k, v in ledger_counts.items() if v}}


def credit_check(ro: Any, calls: list[dict[str, Any]], prog_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Φ telescoping on the root chain (root-chain credit + unattributed = reward - Φ0) and positional vs exact credit."""
    root_turns = [t for t in ro.turns if not t.agent and not t.node()]
    adv_sum = sum(t.advantage or 0.0 for t in root_turns)
    unatt = float((ro.progress or {}).get("unattributed_delta") or 0.0)
    phi0 = next((float(r.get("phi") or 0.0) for r in prog_rows if r.get("step") == 0), 0.0)
    out: dict[str, Any] = {"root_chain_credit": round(adv_sum, 6), "unattributed": round(unatt, 6), "phi0": phi0,
                           "telescopes": ro.reward is not None and abs(adv_sum + unatt + phi0 - ro.reward) < 1e-6}
    # exact: each progress row by its tool_call_id -> the call whose actions hold it -> the root ancestor (a child's spawn parent)
    owner = {a[5:]: c for c in calls for a in (c.get("actions") or []) if a.startswith("tool:")}
    by_id = {c["call_id"]: c for c in calls}
    spawn_parent: dict[str, str] = {}
    for c in calls:
        ag = c.get("agent") or ""
        if ag and c.get("parent") and ag not in spawn_parent and c.get("kind") != "sub":
            spawn_parent[ag] = c["parent"]

    def root_of(cid: str) -> str:
        seen = 0
        while seen < 64:
            c = by_id.get(cid)
            if c is None:
                return cid
            ag = c.get("agent") or ""
            if c.get("kind") == "sub" and c.get("parent"):
                cid = c["parent"]
            elif ag:
                p = c.get("parent") or spawn_parent.get(ag)
                if p is None:
                    return cid
                cid = p
            else:
                return cid
            seen += 1
        return cid

    exact: dict[str, float] = {}
    msg_rows = 0
    for r in prog_rows:
        tid = r.get("tool_call_id")
        if r.get("step", 0) <= 0 or r.get("actor") != "agent":
            continue
        if tid is None:
            msg_rows += 1
            continue
        c = owner.get(tid)
        if c is None:
            continue
        k = root_of(c["call_id"])
        exact[k] = exact.get(k, 0.0) + float(r.get("delta_phi") or 0.0)
    got = {t.call_id: float(t.advantage or 0.0) for t in root_turns}
    diffs = {k: round(got.get(k, 0.0) - exact.get(k, 0.0), 6) for k in set(got) | set(exact) if abs(got.get(k, 0.0) - exact.get(k, 0.0)) > 1e-9}
    out.update({"exact_vs_record_mismatched_calls": len(diffs), "exact_vs_record_max_abs": max((abs(v) for v in diffs.values()), default=0.0),
                "mismatch_examples": dict(sorted(diffs.items())[:6]), "message_rows": msg_rows})
    return out


def root_compaction_steps(ro: Any) -> list[int]:
    from crucible_pack_toy.longhorizon.ladder import compaction_steps

    return compaction_steps([(t.kind, t.world_step) for t in ro.turns if not t.agent])


def analyze_attempt(adir: Path, task: Any | None = None) -> dict[str, Any]:
    from crucible.core.schema.rollout import RolloutRecord
    from crucible.world.world import World

    rec = json.loads((adir / "record.json").read_text())
    ro = RolloutRecord.model_validate_json((adir / "rollout.json").read_text()) if (adir / "rollout.json").exists() else None
    calls = (json.loads((adir / "calls.json").read_text()) if (adir / "calls.json").exists() else {}).get("agent", [])
    ev = json.loads((adir / "rlm.agent.json").read_text()) if (adir / "rlm.agent.json").exists() else {}
    events = World.read_trace(adir / "trace.jsonl")
    prog = [json.loads(x) for x in (adir / "progress.jsonl").read_text().splitlines() if x.strip()] if (adir / "progress.jsonl").exists() else []
    out: dict[str, Any] = {"reward": None if ro is None else ro.reward, "stop": (rec.get("meta") or {}).get("stop"), "mask": None if ro is None else ro.mask_sample,
                           "failure_kind": None if ro is None else ro.failure_kind, "steps": rec.get("steps"),
                           "valid_for_training": None if ro is None else list(ro.valid_for_training())}
    if ro is not None and task is not None and hasattr(task, "b"):
        from crucible_pack_toy.longhorizon.ladder import constraint_audit

        comp = root_compaction_steps(ro)
        c = constraint_audit(events, task.b, comp)
        out["root_compactions"] = len(comp)
        out["constraints"] = {"lost_constraint_rate": c["lost_constraint_rate"], "before": c["before_first_compaction"]["rate"],
                              "broken_total": sum(s["broken"] for s in c["by_segment"]), "missed_awaiting": c["missed_awaiting"], "awaiting": c["awaiting"],
                              "examples": c["examples"][:5]}
    tree = ev.get("tree") or {}
    lim = tree.get("limits") or {}
    out["tree"] = {"turns": tree.get("turns"), "new_tokens": tree.get("new_tokens"), "spawns": tree.get("spawns"), "refused": tree.get("refused"),
                   "peak_running": tree.get("peak_running"), "by_depth": tree.get("by_depth"), "fraction": tree.get("fraction"),
                   "limits": {k: lim.get(k) for k in ("max_spawns", "max_depth", "max_concurrent", "max_total_turns", "max_total_new_tokens", "child_max_turns")}}
    kids = ev.get("children") or {}
    segs = ev.get("segments") or []
    out["children"] = {p: {"stop": v.get("stop"), "turns": (v.get("counters") or {}).get("turns"), "cells": (v.get("counters") or {}).get("cells"),
                           "world_calls": (v.get("counters") or {}).get("world_calls"), "compactions": (v.get("counters") or {}).get("compactions"),
                           "segments": 1 + sum(1 for s in segs if (s.get("agent") or "") == p.split("/", 1)[-1] and s.get("boundary_reason") == "compaction"),
                           "largest_prompt": (v.get("health") or {}).get("largest_prompt")} for p, v in kids.items()}
    out["child_compactions"] = sum((v.get("compactions") or 0) for v in out["children"].values())
    out["root"] = {k: (ev.get("counters") or {}).get(k) for k in ("turns", "cells", "world_calls", "compactions", "agent_requests", "misrouted_calls",
                                                                  "continuations", "stalls", "restores")}
    out["root_largest_prompt"] = (ev.get("context_health") or {}).get("largest_prompt")
    out["threshold"] = (ev.get("context_health") or {}).get("threshold")
    if ro is not None and not ro.mask_sample and all(t.prompt_token_ids is not None or t.prompt_is_delta or not t.trainable for t in ro.turns):
        try:
            out["tokens"] = forwarded_tokens(ro)
        except Exception as e:  # noqa: BLE001 -- a measurement
            out["tokens"] = {"error": f"{type(e).__name__}: {e}"}
    out["attribution"] = attribution(adir, calls, events)
    if ro is not None and ro.reward is not None:
        out["credit"] = credit_check(ro, calls, prog)
    out["segments_by_agent"] = {}
    for s in segs:
        a = s.get("agent") or "root"
        out["segments_by_agent"][a] = out["segments_by_agent"].get(a, 0) + 1
    return out


# ---- campaign 1: child agents ----------------------------------------------------------------------------------------------
async def run_tree(cfg: Any, model: Any, codec: Any, *, keep: Path | None = None) -> dict[str, Any]:
    """One episode of ``model`` (a ``DelegatingRLM``) through the ladder's harness hook, Docker kernel, exact TITO."""
    from crucible_pack_toy.longhorizon.ladder import _Capture, make_harness
    from crucible_pack_toy.longhorizon.task import BacklogTask

    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.run import Runner, RunSpec
    from crucible.train.tito_scripted import LockedTitoLLM

    task = BacklogTask(cfg.seed, cfg.n)
    work = Path(tempfile.mkdtemp(prefix="lh-camp-")) if keep is None else keep
    cap = _Capture()
    harness, how = make_harness(cfg, cap)
    ref = PolicyRef(id=f"scripted:{model.model}", make=lambda attempt: {"agent": LockedTitoLLM(model, codec)})
    t0 = time.perf_counter()
    try:
        await Runner(work).run(RunSpec(run_id="camp", tasks=[task], policies=[ref], harnesses=[harness], k=1, rollout_mode="train"))
        wall = time.perf_counter() - t0
        adirs = sorted((work / "camp" / "samples").glob("*/a*"))
        r = analyze_attempt(adirs[-1], task)
        r.update({"tickets": cfg.n, "attempts": len(adirs), "wall_s": round(wall, 2), "rlm_applied": how.get("applied"), "rlm_ignored": how.get("ignored"),
                  "model_calls_by_role": _count(model.log, "agent_role"), "cells_by_kind": _count(model.log, "kind")})
        r["leftover_containers"] = len(rlm_containers())
        return r
    finally:
        task.cleanup()
        if keep is None:
            shutil.rmtree(work, ignore_errors=True)


def _count(log: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in log:
        k = str(x.get(key))
        out[k] = out.get(k, 0) + 1
    return out


def _short(r: dict[str, Any]) -> str:
    c = r.get("constraints") or {}
    return (f"reward {r['reward']} stop {r['stop']} valid {r['valid_for_training']} lost {c.get('lost_constraint_rate')} missed {c.get('missed_awaiting')} "
            f"spawns {r['tree']['spawns']} child-compactions {r['child_compactions']} fwd/entered {(r.get('tokens') or {}).get('ratio')} "
            f"attributed {r['attribution']['attributed_ok']} telescopes {(r.get('credit') or {}).get('telescopes')} "
            f"credit-mismatch {(r.get('credit') or {}).get('exact_vs_record_mismatched_calls')} wall {r['wall_s']} left {r['leftover_containers']}")


async def children(a: argparse.Namespace) -> dict[str, Any]:
    from crucible_pack_toy.longhorizon.ladder import LadderConfig

    from crucible.runtime.scripted_rlm_tree import DelegatingRLM
    from crucible.train.tito_scripted import qwen_codec

    codec = qwen_codec()
    rep: dict[str, Any] = {"campaign": "M16 child agents on the backlog ladder", "runs": []}
    extra = json.loads(a.rlm) if a.rlm else {}
    for rung in [int(x) for x in a.rungs.split(",")]:
        for depth in [int(x) for x in a.depths.split(",")]:
            for mode in a.modes.split(","):
                model = DelegatingRLM(mode=mode, children=a.children, depth=depth)
                cfg = LadderConfig(rung=rung, variant="faithful", kernel="docker", checkpoint_every=a.checkpoint_every, restore_probe=False,
                                   rlm={"tree_max_depth": depth, **extra})
                r = await run_tree(cfg, model, codec, keep=Path(a.keep) / f"r{rung}-d{depth}-{mode}" if a.keep else None)
                r.update({"rung": rung, "depth": depth, "mode": mode})
                rep["runs"].append(r)
                print(f"[children] r{rung} d{depth} {mode}: {_short(r)}", flush=True)
                Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return rep


async def adversarial(a: argparse.Namespace) -> dict[str, Any]:
    from crucible_pack_toy.longhorizon.ladder import LadderConfig

    from crucible.runtime.scripted_rlm_tree import DelegatingRLM
    from crucible.train.tito_scripted import qwen_codec

    codec = qwen_codec()
    rep: dict[str, Any] = {"campaign": "M16 adversarial tree probe", "runs": []}
    for depth in [int(x) for x in a.depths.split(",")]:
        model = DelegatingRLM(mode="adversarial", depth=depth, overspawn=a.overspawn)
        cfg = LadderConfig(rung=100, variant="faithful", kernel="docker", restore_probe=False, rlm={"tree_max_depth": depth})
        keep = Path(tempfile.mkdtemp(prefix="lh-adv-"))
        try:
            r = await run_tree(cfg, model, codec, keep=keep)
            adir = max((keep / "camp" / "samples").glob("*/a*"))
            r["probe"] = adversarial_findings(adir)
        finally:
            shutil.rmtree(keep, ignore_errors=True)
        r["depth"] = depth
        rep["runs"].append(r)
        print(f"[adversarial] d{depth}: reward {r['reward']} stop {r['stop']} valid {r['valid_for_training']} spawns {r['tree']['spawns']} "
              f"refused {r['tree']['refused']} left {r['leftover_containers']} probe {json.dumps(r['probe'])[:1500]}", flush=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return rep


def adversarial_findings(adir: Path) -> dict[str, Any]:
    """What the adversarial root's cells printed, and what the escalator grandchild did on the World (from the ledgers)."""
    root = adir / "rlm" / "agent" / "root"
    lines: dict[str, str] = {}
    for line in (root / "events.jsonl").read_text().splitlines():
        ev = json.loads(line)
        if ev.get("kind") == "tool":
            for mark in ("ADV-PROBE-RESULT", "ADV-ARGS", "ADV-SPAWNED", "ADV-LIST"):
                for row in str(ev.get("text") or "").splitlines():
                    if row.startswith(mark):
                        lines[mark] = row[:3000]
    esc_world = 0
    esc_dirs = []
    for f in root.rglob("events.jsonl"):
        rel = f.parent.relative_to(root).as_posix()
        if rel.endswith("esc"):
            esc_dirs.append(rel)
            esc_world += sum(1 for x in f.read_text().splitlines() if '"kind":"world_call"' in x or '"kind": "world_call"' in x)
    ev = json.loads((adir / "rlm.agent.json").read_text())
    stops: dict[str, int] = {}
    for v in (ev.get("children") or {}).values():
        stops[str(v.get("stop"))] = stops.get(str(v.get("stop")), 0) + 1
    return {"root_lines": lines, "escalator_ledgers": esc_dirs, "escalator_world_calls": esc_world, "child_stops": stops,
            "children_known": len(ev.get("children") or {})}


# ---- campaign 2: a real model --------------------------------------------------------------------------------------------------
class HFModel:
    """One HF chat model on MPS / CPU behind ``EngineLLM`` (exact TITO); ``log`` keeps every call's sizes and text."""

    def __init__(self, name: str):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoTokenizer

        from crucible.train.chat import HFChatCodec
        from crucible.train.hf_engine import HFGenerateEngine

        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.codec = HFChatCodec(self.tok)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.engine = HFGenerateEngine(name, device=self.device)
        self.log: list[dict[str, Any]] = []

    async def gen(self, prompt, n, temp, *, seed=None, stop_ids=()):
        from crucible.train.engine import GenRequest

        t0 = time.perf_counter()
        r = await self.engine.generate(GenRequest(rid="q", call_id=f"c{len(self.log)}", prompt_ids=prompt, max_tokens=n, temperature=temp,
                                                  stop_ids=list(stop_ids) or self.codec.eos_ids, seed=seed))
        self.log.append({"prompt": len(prompt), "completion": len(r.completion_ids), "s": round(time.perf_counter() - t0, 2),
                         "text": self.tok.decode(r.completion_ids)[:600]})
        return r

    def client(self) -> Any:
        from crucible.train.world_env import EngineLLM

        return EngineLLM(self.codec, self.gen, model=self.name)


def transcript(adir: Path, limit: int = 40) -> list[dict[str, Any]]:
    """The root's assistant turns and what came back (from the ledger), for reading what the model did."""
    out: list[dict[str, Any]] = []
    f = adir / "rlm" / "agent" / "root" / "events.jsonl"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        ev = json.loads(line)
        k = ev.get("kind")
        if k == "assistant":
            out.append({"i": ev["i"], "assistant": str(ev.get("text") or "")[:400], "code": str(ev.get("code") or "")[:400], "tools": ev.get("tools")})
        elif k in ("tool", "notice", "summary") or (k == "user" and ev.get("role") in ("continuation",)):
            out.append({"i": ev["i"], k: str(ev.get("text") or "")[:400]})
        elif k in ("compaction_attempt", "compaction_try"):
            out.append({"i": ev["i"], "compaction_attempt": {"accepted": ev.get("accepted"), "text": str(ev.get("text") or "")[:300]}})
    return out[:limit]


async def run_real(hf: HFModel, task: Any, cfg: Any, *, seed: int, max_steps: int = 400, keep: Path | None = None) -> dict[str, Any]:
    import torch

    from crucible.actors.policy import EpisodeLimits
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.harness.rlm import RLMHarness
    from crucible.runtime.run import Runner, RunSpec

    torch.manual_seed(seed)
    work = keep or Path(tempfile.mkdtemp(prefix="lh-qwen-"))
    h = RLMHarness(cfg, limits=EpisodeLimits(max_steps=max_steps, max_tool_errors=None))
    n0 = len(hf.log)
    t0 = time.perf_counter()
    try:
        await Runner(work).run(RunSpec(run_id="q", tasks=[task], policies=[PolicyRef(id=f"hf:{hf.name}", make=lambda a: {"agent": hf.client()})],
                                       harnesses=[h], k=1, rollout_mode="train"))
        adir = max((work / "q" / "samples").glob("*/a*"))
        r = analyze_attempt(adir, task if hasattr(task, "b") else None)
        ev = json.loads((adir / "rlm.agent.json").read_text())
        cn = ev.get("counters") or {}
        calls = (json.loads((adir / "calls.json").read_text()) if (adir / "calls.json").exists() else {}).get("agent", [])
        comp = [c for c in calls if c.get("kind") == "compaction"]
        r.update({"seed": seed, "wall_s": round(time.perf_counter() - t0, 1), "device": hf.device,
                  "cells": cn.get("cells"), "world_calls": cn.get("world_calls"), "world_sends": cn.get("world_sends"),
                  "corrections": cn.get("misrouted_calls"), "continuations": cn.get("continuations"), "turns": cn.get("turns"),
                  "compactions": cn.get("compactions"), "compaction_attempts": len(comp), "summaries_accepted": sum(1 for c in comp if c.get("accepted")),
                  "summaries_rejected": sum(1 for c in comp if not c.get("accepted")), "context_reads": cn.get("context_reads"),
                  "cell_errors": sum(1 for x in transcript(adir, 10_000) if "tool" in x and ("Error" in x["tool"] or "Traceback" in x["tool"])),
                  "calls": len(hf.log) - n0, "gen_s": round(sum(x["s"] for x in hf.log[n0:]), 1),
                  "prompt_max": max((x["prompt"] for x in hf.log[n0:]), default=0), "transcript": transcript(adir)})
        r["leftover_containers"] = len(rlm_containers())
        return r
    finally:
        with contextlib.suppress(Exception):
            task.cleanup()
        if keep is None:
            shutil.rmtree(work, ignore_errors=True)


async def qwen(a: argparse.Namespace) -> dict[str, Any]:
    from crucible_pack_toy.longhorizon.task import BacklogTask

    from crucible.bench.longctx.task import LongCtxTask
    from crucible.runtime.harness.rlm import RLMConfig

    hf = HFModel(a.model)
    rep: dict[str, Any] = {"campaign": f"M16 real model: {a.model} in the Docker RLM kernel", "model": a.model, "device": hf.device, "runs": []}

    def save() -> None:
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")

    base = dict(max_tokens=a.max_tokens, sub_max_tokens=256, temperature=a.temperature, max_turns=a.turns, kernel="docker", **(json.loads(a.rlm) if a.rlm else {}))
    rep["rlm_overrides"] = json.loads(a.rlm) if a.rlm else {}
    parts = a.parts.split(",")
    if "backlog" in parts:
        for s in range(a.seeds):
            cfg = RLMConfig(window_tokens=32_000, **base)
            r = await run_real(hf, BacklogTask(0, 15), cfg, seed=s)
            r.update({"part": "backlog-100", "window": 32_000})
            rep["runs"].append(r)
            print(f"[qwen] backlog-100 s{s}: reward {r['reward']} cells {r['cells']} world {r['world_calls']} corrections {r['corrections']} "
                  f"continuations {r['continuations']} compactions {r['compactions']} valid {r['valid_for_training']} stop {r['stop']} {r['wall_s']} s", flush=True)
            save()
    if "compact" in parts:
        for s in range(a.compact_seeds):
            cfg = RLMConfig(window_tokens=8_000, **{**base, "max_turns": a.compact_turns})
            r = await run_real(hf, BacklogTask(0, 80), cfg, seed=100 + s)
            r.update({"part": "backlog-500-window8k", "window": 8_000})
            rep["runs"].append(r)
            print(f"[qwen] compact s{s}: reward {r['reward']} cells {r['cells']} compactions {r['compactions']} attempts {r['compaction_attempts']} "
                  f"accepted {r['summaries_accepted']} valid {r['valid_for_training']} stop {r['stop']} {r['wall_s']} s", flush=True)
            save()
    if "longctx" in parts:
        for size in (8_000, 32_000):
            for s in range(a.seeds_ctx):
                task = LongCtxTask(a.family, s, size, delivery="context")
                cfg = RLMConfig(window_tokens=32_000, **base)
                r = await run_real(hf, task, cfg, seed=200 + s, max_steps=50)
                r.update({"part": f"longctx-{a.family}-{size}", "window": 32_000, "context_chars": len(task.item.context)})
                rep["runs"].append(r)
                print(f"[qwen] longctx {a.family} {size} s{s}: reward {r['reward']} cells {r['cells']} context_reads {r['context_reads']} "
                      f"world {r['world_calls']} corrections {r['corrections']} valid {r['valid_for_training']} stop {r['stop']} {r['wall_s']} s", flush=True)
                save()
    return rep


# ---- campaign 3: scale probe ----------------------------------------------------------------------------------------------------
async def scale(a: argparse.Namespace) -> dict[str, Any]:
    from crucible_pack_toy.longhorizon.ladder import LadderConfig, run_rung

    from crucible.train.tito_scripted import qwen_codec

    cfg = LadderConfig(rung=a.rung, tickets=a.tickets, variant="faithful", kernel="docker", checkpoint_every=a.checkpoint_every)
    keep = Path(tempfile.mkdtemp(prefix="lh-scale-"))
    try:
        t0 = time.perf_counter()
        r = await run_rung(cfg, root=keep, codec=qwen_codec(), keep=True)
        adir = Path(r["dir"])
        ledger_bytes = sum(p.stat().st_size for p in (adir / "rlm").rglob("*") if p.is_file())
        snaps = sorted((adir / "rlm").rglob("snap-*.bin"))
        ev = json.loads((adir / "rlm.agent.json").read_text())
        out = {k: r[k] for k in ("label", "tickets", "world_calls", "reward", "stop", "mask", "failure_kind", "valid_for_training", "compactions",
                                 "constraints", "b1", "wall_s", "ms_per_world_call", "checkpoints", "restore", "rlm_counters", "attempts")}
        out["constraints"] = {k: r["constraints"][k] for k in ("lost_constraint_rate", "missed_awaiting", "awaiting")}
        out.update({"total_s": round(time.perf_counter() - t0, 1), "ledger_dir_bytes": ledger_bytes, "ledger_stats": ev.get("ledger"),
                    "snapshot_files": len(snaps), "snapshot_bytes_last": snaps[-1].stat().st_size if snaps else None,
                    "calls_stream_bytes": sum(p.stat().st_size for p in adir.glob("calls.*.jsonl")),
                    "largest_prompt": (ev.get("context_health") or {}).get("largest_prompt"), "threshold": (ev.get("context_health") or {}).get("threshold"),
                    "largest_over_threshold": (ev.get("context_health") or {}).get("largest_prompt_over_threshold"),
                    "leftover_containers": len(rlm_containers())})
        h = ev.get("context_health") or {}
        tb, sc = h.get("turns_between_compactions") or [], h.get("summary_chars_by_compaction") or []

        def tenths(xs: list[Any]) -> list[Any]:  # the series at 0 %, 10 %, ..., 100 % of the episode
            return [xs[min(len(xs) - 1, round(i * (len(xs) - 1) / 10))] for i in range(11)] if xs else []

        out.update({"turns_between_compactions_tenths": tenths(tb), "summary_chars_tenths": tenths(sc), "stall_verdicts": h.get("stall_verdicts"),
                    "restores_by_mode": h.get("restores_by_mode")})
        print(json.dumps({k: out[k] for k in ("reward", "world_calls", "compactions", "wall_s", "largest_over_threshold", "ledger_dir_bytes", "leftover_containers")}))
        return {"campaign": "M16 scale probe", "run": out}
    finally:
        shutil.rmtree(keep, ignore_errors=True)


async def main(a: argparse.Namespace) -> None:
    before = rlm_containers()
    fn = {"children": children, "adversarial": adversarial, "qwen": qwen, "scale": scale}[a.part]
    rep = await fn(a)
    rep["containers_before"] = len(before)
    rep["containers_after"] = len(rlm_containers())
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"[{a.part}] leftover RLM containers: {rep['containers_after']} (before: {rep['containers_before']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("part", choices=["children", "adversarial", "qwen", "scale"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rungs", default="100,500,1000")
    ap.add_argument("--depths", default="1,2")
    ap.add_argument("--modes", default="shared,none")
    ap.add_argument("--children", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=0, help="children: checkpoint cadence (0: off); scale: use 500 as the ladder does")
    ap.add_argument("--rlm", default="", help="extra RLMConfig fields as JSON")
    ap.add_argument("--keep", default="", help="keep attempt directories under this path")
    ap.add_argument("--overspawn", type=int, default=80)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--parts", default="backlog,compact,longctx")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seeds-ctx", type=int, default=3)
    ap.add_argument("--compact-seeds", type=int, default=2)
    ap.add_argument("--compact-turns", type=int, default=40)
    ap.add_argument("--family", default="docqa")
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--rung", type=int, default=10_000)
    ap.add_argument("--tickets", type=int, default=1_640)
    asyncio.run(main(ap.parse_args()))
