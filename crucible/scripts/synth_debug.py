#!/usr/bin/env python
"""crucible.synth M5 on the debug lane's training pool (part2-paper-core §14 M5). CPU and local Docker only; no model call.

    python scripts/synth_debug.py migrate  --pool P --out D [--tasks a,b]     # stored gate verdicts recomputed through the adapter
    python scripts/synth_debug.py certify  --pool P --out D [--tasks a,b] [--max-hunks 16] [--no-visible] [--attack --cap USD [--attack-rounds 3]]
    python scripts/synth_debug.py realdefects --out D [--records R] [--synth-data S] [--debug-store T] [--execute]

``--pool`` is a debug root of training windows (``data/debug-pool``); ``--out`` is where reports go (the main checkout's
``crucible/data/synth/debug``: never a temporary directory). Tasks default to the manifest's ``ready`` list
(``docs/research/swarm-pipeline/07-debug-lane/training-pool/pool.json``). Docker containers are ``crucible-lh-d-synth-*``,
at most ``--parallel`` (default 4) at once, removed by name at the end.

``certify --attack`` (Part 1 V2) is the one paid mode: the role table's attackers (``SynthWire``, never the student's
family) attack each task for up to ``--attack-rounds`` rounds with the failed checks fed back, the panel decides each pass,
and ``--cap`` bounds the spend (``certificates/spend.json``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MANIFEST = ROOT / "docs" / "research" / "swarm-pipeline" / "07-debug-lane" / "training-pool" / "pool.json"


def ready(tasks: str) -> list[str]:
    m = json.loads(MANIFEST.read_text())
    ids = [t for t, r in m["tasks"].items() if r.get("status") == "ready"]
    if tasks:
        want = [t for t in tasks.split(",") if t]
        bad = [t for t in want if t not in ids]
        if bad:
            raise SystemExit(f"not ready training tasks: {bad}")
        return want
    return sorted(ids)


def write(out: Path, name: str, obj: Any) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")


async def migrate(args: argparse.Namespace) -> int:
    from crucible.factory.synth_patch_wire import lane_runner, open_pool, patch_env, remove_containers
    from crucible.synth.patch import migrate as recompute

    pool = open_pool(Path(args.pool), parallel=args.parallel)
    out = Path(args.out) / "migration"
    summary: dict[str, Any] = {}
    try:
        for wid in ready(args.tasks):
            task = pool.task(wid)
            env = patch_env(task)
            runner = lane_runner(task, parallel=args.parallel)
            try:
                rep = await recompute(env, runner, pool.gates(wid))
            finally:
                runner.close()
            write(out, f"{wid}.json", {**rep.model_dump(mode="json"), "equal": rep.equal, "runner_log": runner.log})
            summary[wid] = {"equal": rep.equal, "rows": len(rep.rows), "differences": [d.model_dump(mode="json") for d in rep.differences()],
                            "errors": rep.errors, "docker_runs": len(runner.log), "seconds": rep.seconds}
            print(f"{wid}: equal={rep.equal} rows={len(rep.rows)} diffs={len(rep.differences())} errors={rep.errors} runs={len(runner.log)} "
                  f"({rep.seconds}s)", flush=True)
    finally:
        left = remove_containers()
        if left:
            print(f"removed leftover containers: {left}")
    prev = json.loads((out / "summary.json").read_text()).get("tasks", {}) if (out / "summary.json").is_file() else {}
    merged = {**prev, **summary}  # a run over some tasks updates their rows and keeps the others'
    write(out, "summary.json", {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tasks": merged, "all_equal": all(v["equal"] for v in merged.values())})
    return 0 if all(v["equal"] for v in summary.values()) else 1


async def certify(args: argparse.Namespace) -> int:
    from crucible.factory.synth_patch_wire import lane_runner, open_pool, patch_env, prior_of, remove_containers
    from crucible.synth.certificate import summary as cert_summary
    from crucible.synth.falsify.patch import applicability, certify_patch, findings_summary

    pool = open_pool(Path(args.pool), parallel=args.parallel)
    out = Path(args.out) / "certificates"
    table: dict[str, Any] = {}
    wire = roles = None
    if args.attack:
        from crucible.factory.synth_wire import SynthWire

        if args.cap <= 0:
            raise SystemExit("--attack calls paid models: give --cap USD")
        wire = SynthWire(out, cap_usd=args.cap)
        roles = wire.run_roles()
    try:
        for wid in ready(args.tasks):
            t0 = time.monotonic()
            task = pool.task(wid)
            env = patch_env(task)
            runner = lane_runner(task, parallel=args.parallel)
            kw: dict[str, Any] = {}
            if roles is not None:
                from crucible.synth.falsify.agents import AgentBudget, Panel, PanelMember, llm_atk

                kw = {"attackers": [llm_atk(n, c, budget=AgentBudget(max_calls=args.atk_calls, max_turns=args.atk_calls + 2), tag=f"ATK:{wid}") for n, c in roles.atk],
                      "exclude_families": {"qwen"}, "attack_rounds": args.attack_rounds,
                      "panel": Panel(members=[PanelMember(n, c) for n, c in roles.panel], lam=0.7, queue=out / "human_queue.jsonl", tag=f"adjudicate:{wid}")
                      if len(roles.panel) >= 2 else None}
            try:
                cert, results = await certify_patch(env, runner, prior=prior_of(pool.gates(wid)), max_hunks=args.max_hunks, visible=not args.no_visible, **kw)
            finally:
                runner.close()
            fs = findings_summary(results)
            row = {"status": cert.status, "reasons": cert.reasons, "provisional_for": cert.provisional_for, "classes": cert_summary(cert),
                   "applicability": applicability(results), "findings": fs, "docker_runs": len(runner.log), "seconds": round(time.monotonic() - t0, 1)}
            write(out, f"{wid}.json", {"certificate": cert.model_dump(mode="json"), **row})
            table[wid] = row
            print(f"{wid}: {cert.status} {cert.reasons} confirmed={[(c['op'], c['defect']) for c in fs['confirmed']]} "
                  f"suspects={len(fs['suspects'])} report_only={len(fs['report_only'])} runs={len(runner.log)} ({row['seconds']}s)", flush=True)
    finally:
        left = remove_containers()
        if left:
            print(f"removed leftover containers: {left}")
    prev = json.loads((out / "summary.json").read_text()).get("tasks", {}) if (out / "summary.json").is_file() else {}
    write(out, "summary.json", {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tasks": {**prev, **table}})
    if wire is not None:
        write(out, "spend.json", wire.cost())
    return 0


def realdefects(args: argparse.Namespace) -> int:
    from crucible.synth import realdefects as rd

    docs = ROOT
    records = Path(args.records) if args.records else ROOT / "docs" / "research" / "env-synthesis" / "runs" / "m5" / "debug_lane_defects.json"
    rs = rd.build_set(debug_records=records, docs_root=docs, synth_data=Path(args.synth_data) if args.synth_data else None,
                      debug_store=Path(args.debug_store) if args.debug_store else None)
    out = Path(args.out)
    if args.execute:
        from crucible.factory.synth_patch_wire import detect_debug

        debug_root = Path(args.debug_store).parents[2] if args.debug_store else None  # <root>/gen/store/train -> <root>

        async def det(d: Any) -> Any:
            return await detect_debug(d, root=debug_root)  # type: ignore[arg-type]

        rs = asyncio.run(rd.measure(rs, detect_debug=det if debug_root else None, synth_data=Path(args.synth_data) if args.synth_data else None,
                                    work=out / "replay_work"))
    write(out, "realdefects.json", rs.model_dump(mode="json"))
    write(out, "realdefects_summary.json", rd.summary(rs))
    (out / "realdefects_table.md").write_text(rd.counts_table(rs) + "\n")
    print(json.dumps(rd.summary(rs), indent=1))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("migrate", "certify"):
        p = sub.add_parser(name)
        p.add_argument("--pool", default=os.environ.get("CRUCIBLE_DEBUG_POOL") or str(ROOT / "data" / "debug-pool"))
        p.add_argument("--out", required=True)
        p.add_argument("--tasks", default="")
        p.add_argument("--parallel", type=int, default=4)
        if name == "certify":
            p.add_argument("--max-hunks", type=int, default=16)
            p.add_argument("--no-visible", action="store_true")
            p.add_argument("--attack", action="store_true", help="Part 1 V2: run the role table's attackers (paid)")
            p.add_argument("--attack-rounds", type=int, default=3)
            p.add_argument("--atk-calls", type=int, default=24, help="tool calls per attack round")
            p.add_argument("--cap", type=float, default=0.0, help="USD cap for --attack")
    p = sub.add_parser("realdefects")
    p.add_argument("--out", required=True)
    p.add_argument("--records", default="")
    p.add_argument("--synth-data", default="")
    p.add_argument("--debug-store", default="")
    p.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if getattr(args, "parallel", 1) > 4:
        raise SystemExit("at most 4 containers at once")
    if args.cmd == "migrate":
        sys.exit(asyncio.run(migrate(args)))
    if args.cmd == "certify":
        sys.exit(asyncio.run(certify(args)))
    sys.exit(realdefects(args))


if __name__ == "__main__":
    main()
