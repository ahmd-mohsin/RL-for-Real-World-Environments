#!/usr/bin/env python3
"""crucible.synth on the Tinker swarm (part2-paper-core M4): World synthesis, task synthesis, certification, reports.

    synth.py worldgen --keyword "IT helpdesk ticket triage" --name it_helpdesk --cap 8
    synth.py propose  --domain finance_ap --n 20 --cap 2              # proposals only (no oracles, no falsifiers)
    synth.py certify  --domain finance_ap --n 20 --cap 25             # propose -> oracle -> contract -> falsify/repair -> certify
    synth.py certify  --domain it_helpdesk --world <run dir of worldgen>/world --n 20 --cap 25
    synth.py report   --domain finance_ap [--run-id R]                # a run's report.md (default: the latest run)
    synth.py budget                                                   # the milestone's Tinker spend so far
    synth.py smoke --cap 0.5                                          # one JSON call per model of the assignment
    (plant / recall / evolve: M3, the planter and the regularised loop)

Artifacts: ``<data-root>/<domain>/<run-id>/`` (default data root: ``$CRUCIBLE_DATA_DIR/synth``, else the config's data
dir + ``/synth``), the counterexample archive ``<data-root>/<domain>/archive.jsonl``. **Spend:** each run has one hard
cap (``--cap``, booked in its ``spend.jsonl``), and the cap is clipped to what the milestone budget
(``MILESTONE_CAP_USD`` = 60) has left over every ``spend.jsonl`` under the data root; a run with nothing left is refused.
The key comes from the environment only: ``set -a; . <repo>/.env; set +a`` in the shell that starts this script.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE / "src") not in sys.path:
    sys.path.insert(0, str(HERE / "src"))


def data_root(a: argparse.Namespace) -> Path:
    if a.data_root:
        return Path(a.data_root)
    if os.environ.get("CRUCIBLE_DATA_DIR"):
        return Path(os.environ["CRUCIBLE_DATA_DIR"]) / "synth"
    from crucible.core.config import get_config

    return get_config().data_path() / "synth"


def run_cap(a: argparse.Namespace, root: Path) -> float:
    from crucible.factory.synth_wire import MILESTONE_CAP_USD, milestone_spent

    spent = milestone_spent(root)
    left = round(MILESTONE_CAP_USD - spent, 4)
    cap = min(float(a.cap), left)
    print(f"milestone spend so far {spent:.4f} of {MILESTONE_CAP_USD}; this run's cap {cap:.4f}", flush=True)
    if cap < 0.05:
        raise SystemExit("the milestone's Tinker budget is spent: refusing to start")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment (set -a; . <repo>/.env; set +a)")
    return cap


def new_run(root: Path, domain: str, tag: str) -> tuple[str, Path]:
    rid = time.strftime("%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
    d = root / domain / rid
    d.mkdir(parents=True, exist_ok=False)
    return rid, d


def cmd_worldgen(a: argparse.Namespace) -> None:
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.worldgen import worldgen

    root = data_root(a)
    cap = run_cap(a, root)
    _rid, d = new_run(root, a.name, a.tag)
    wire = SynthWire(d, cap_usd=cap)
    rep = asyncio.run(worldgen(a.keyword, d / "world", wire.worldgen_clients(), guidance=a.guidance, max_debug_rounds=a.rounds))
    out = {"run": str(d), "report": rep.model_dump(mode="json"), "cost": wire.cost()}
    (d / "worldgen_cost.json").write_text(json.dumps(out["cost"], indent=1) + "\n")
    print(json.dumps(out, indent=1, default=str))


def _env(a: argparse.Namespace, d: Path):
    from crucible.factory.synth_wire import finance_ap_env, generated_env

    if a.domain == "finance_ap":
        return finance_ap_env(d / "work", seeds=tuple(int(x) for x in a.seeds.split(",")))
    if not a.world:
        raise SystemExit("a generated domain needs --world <worldgen output dir>")
    w = Path(a.world)
    rep = json.loads((w / "worldgen_report.json").read_text())
    if rep.get("status") != "accepted":
        raise SystemExit(f"the World at {w} was not accepted: {rep.get('status')}")
    design = json.loads((w / "design.json").read_text())
    return generated_env(w, d / "work", name=a.domain, description=f"{rep['keyword']}: {design.get('description', '')}")


def _runner(a: argparse.Namespace, *, stage_only: bool = False):
    from crucible.factory.synth_wire import SynthWire
    from crucible.synth.falsify.agents import AgentBudget
    from crucible.synth.pipeline import RunConfig, Runner

    root = data_root(a)
    cap = run_cap(a, root)
    rid, d = new_run(root, a.domain, a.tag)
    wire = SynthWire(d, cap_usd=cap)
    cfg = RunConfig(n_tasks=a.n, batch=a.batch, max_rounds=a.rounds, concurrency=a.concurrency, calibrate=not a.no_calibrate,
                    alt_budget=AgentBudget(max_calls=a.alt_calls, max_turns=a.alt_calls + 4),
                    atk_budget=AgentBudget(max_calls=a.atk_calls, max_turns=a.atk_calls + 2), show_reasons=not a.blind_panel)
    env = _env(a, d)
    r = Runner(env, wire.run_roles(), cfg, d, root, run_id=rid, cost=wire.cost, families=wire.families())
    return r, d, wire


def cmd_propose(a: argparse.Namespace) -> None:
    r, d, wire = _runner(a)

    async def go() -> None:
        await r.survey()
        recs = await r.propose()
        r.report.tasks = recs
        r.save()

    asyncio.run(go())
    print(json.dumps({"run": str(d), "proposed": len(r.report.tasks), "cost": wire.cost()}, indent=1))


def cmd_certify(a: argparse.Namespace) -> None:
    r, d, wire = _runner(a)
    rep = asyncio.run(r.run())
    print((d / "report.md").read_text())
    print(json.dumps({"run": str(d), "funnel": rep.funnel, "cost": wire.cost(), "stopped": rep.stopped}, indent=1, default=str))


def cmd_report(a: argparse.Namespace) -> None:
    root = data_root(a) / a.domain
    runs = sorted(p for p in root.iterdir() if (p / "report.md").exists()) if root.exists() else []
    if not runs:
        raise SystemExit(f"no runs under {root}")
    d = root / a.run_id if a.run_id else runs[-1]
    print((d / "report.md").read_text())


def cmd_budget(a: argparse.Namespace) -> None:
    from crucible.factory.synth_wire import MILESTONE_CAP_USD, milestone_spent

    root = data_root(a)
    rows = {}
    for p in sorted(root.rglob("spend.jsonl")):
        tot = sum(float(json.loads(x).get("usd") or 0) for x in p.read_text().splitlines() if x.strip())
        rows[str(p.parent.relative_to(root))] = round(tot, 4)
    print(json.dumps({"runs": rows, "spent": milestone_spent(root), "cap": MILESTONE_CAP_USD}, indent=1))


def cmd_smoke(a: argparse.Namespace) -> None:
    """One small JSON call per distinct model of the assignment: the key, the tokenizers, the meter and the parser."""
    from crucible.factory.synth_wire import SynthWire

    root = data_root(a)
    cap = run_cap(a, root)
    _rid, d = new_run(root, "_smoke", a.tag)
    wire = SynthWire(d, cap_usd=cap)
    seen: dict[str, tuple[str, int]] = {}
    for role, ms in wire.table.items():
        for i, m in enumerate(ms):
            seen.setdefault(m, (role, i))

    async def go() -> dict[str, object]:
        out: dict[str, object] = {}
        for m, (role, i) in seen.items():
            c = wire.client(role, i)
            t0 = time.time()
            try:
                doc = await c.complete_json([{"role": "user", "content": 'Reply with the JSON object {"ok": true, "sum": <the value of 2+3>}.'}],
                                            schema=["ok"], max_tokens=1024, tag="smoke:models")
                out[m] = {"ok": doc.get("ok"), "sum": doc.get("sum"), "s": round(time.time() - t0, 1)}
            except Exception as e:  # noqa: BLE001 - the smoke reports every model
                out[m] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        return out

    res = asyncio.run(go())
    print(json.dumps({"run": str(d), "models": res, "cost": wire.cost()}, indent=1, default=str))


def cmd_m3(a: argparse.Namespace) -> None:
    raise SystemExit(f"'{a.cmd}' belongs to M3 (the planter, recall matrix and regularised loop): not part of this CLI yet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("worldgen")
    w.add_argument("--keyword", required=True)
    w.add_argument("--name", required=True)
    w.add_argument("--guidance", default="")
    w.add_argument("--rounds", type=int, default=3)
    w.add_argument("--cap", type=float, default=8.0)
    w.add_argument("--tag", default="")
    w.set_defaults(fn=cmd_worldgen)

    for name, fn in (("propose", cmd_propose), ("certify", cmd_certify)):
        p = sub.add_parser(name)
        p.add_argument("--domain", required=True)
        p.add_argument("--world", default=None, help="a worldgen output dir (generated domains)")
        p.add_argument("--seeds", default="0,1,2,3", help="finance_ap start-state seeds")
        p.add_argument("--n", type=int, default=20)
        p.add_argument("--batch", type=int, default=5)
        p.add_argument("--rounds", type=int, default=3)
        p.add_argument("--concurrency", type=int, default=6)
        p.add_argument("--alt-calls", type=int, default=24)
        p.add_argument("--atk-calls", type=int, default=14)
        p.add_argument("--no-calibrate", action="store_true")
        p.add_argument("--blind-panel", action="store_true", help="the panel never sees the grader's reason codes")
        p.add_argument("--cap", type=float, default=25.0)
        p.add_argument("--tag", default="")
        p.set_defaults(fn=fn)

    r = sub.add_parser("report")
    r.add_argument("--domain", required=True)
    r.add_argument("--run-id", default=None)
    r.set_defaults(fn=cmd_report)

    s = sub.add_parser("smoke")
    s.add_argument("--cap", type=float, default=0.5)
    s.add_argument("--tag", default="")
    s.set_defaults(fn=cmd_smoke)

    b = sub.add_parser("budget")
    b.set_defaults(fn=cmd_budget)
    for name in ("plant", "recall", "evolve"):
        m = sub.add_parser(name, help="M3")
        m.set_defaults(fn=cmd_m3)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
