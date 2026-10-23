"""Difficulty tools for the swarm (swarm-pipeline/12): the cheap predictor's fit and leave-one-task-out report, and
checkpoint-conditioned difficulty per task, from the measured S15 targeting data. No model calls, no spend.

    uv run python scripts/lh_difficulty.py predict --out docs/research/swarm-pipeline/runs/difficulty_predictor.json
    uv run python scripts/lh_difficulty.py conditional --arms data/targeting/debug-300/arms/student --out ...

Training tasks only: held-out records are skipped (their traces never shape training tasks, V9).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crucible.factory.difficulty import backfill_budget_stops, conditional, predictor_report

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "docs" / "research" / "swarm-pipeline" / "runs"


def _windows(store: Path) -> dict[str, list[dict]]:
    out = {}
    for split in ("train",):
        for w in sorted((store / split / "tasks").glob("*/window.json")):
            out[w.parent.name] = json.loads(w.read_text()).get("milestones") or []
    return out


def cmd_predict(a: argparse.Namespace) -> None:
    arms = list(a.arms) + [None] * (len(a.runs) - len(a.arms))
    runs = [backfill_budget_stops(json.loads((RUNS / f).read_text()), Path(d) if d else None) for f, d in zip(a.runs, arms, strict=False)]
    cands = {c["id"]: c for c in json.loads(Path(a.candidates).read_text())}
    before = predictor_report([json.loads((RUNS / f).read_text()) for f in a.runs], cands)
    rep = predictor_report(runs, cands)
    rep["before_backfill"] = {k: before.get(k) for k in ("rows", "tasks", "mae", "mae_baseline") if k in before}
    rep["inputs"] = {"runs": a.runs, "arms": a.arms, "candidates": str(a.candidates), "split": "train only",
                     "budget_stops": {f: sorted({t.get("budget_stops_source") for t in r.get("tasks") or [] if t.get("budget_stops_source")}) for f, r in zip(a.runs, runs, strict=False)}}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("predictions", "fit")}, default=str))
    for p in rep["predictions"]:
        print(p)


def cmd_conditional(a: argparse.Namespace) -> None:
    wins = _windows(Path(a.store))
    out = {}
    for d in sorted(Path(a.arms).iterdir()):
        ep = d / "episodes.jsonl"
        if d.name not in wins or not ep.exists():
            continue  # held-out (not in the train store) or not run
        rows = [json.loads(x) for x in ep.read_text().splitlines() if x.strip()]
        out[d.name] = conditional(rows, wins[d.name])
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    for t, c in out.items():
        print(t, "p_full", c["p_full"], "phi", None if c["phi_mean"] is None else round(c["phi_mean"], 3), "stall", c["stall_level"],
              {m: (v["p"], v["p_given_prereqs"]) for m, v in c["milestones"].items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("predict")
    p.add_argument("--runs", nargs="+", default=["s15_debug_targeting.json", "s15_debug_targeting_t300.json"])
    p.add_argument("--candidates", default=str(ROOT / "data" / "debug" / "candidates.json"))
    p.add_argument("--arms", nargs="*", default=[str(ROOT / "data" / "targeting" / "debug" / "arms"), str(ROOT / "data" / "targeting" / "debug-300" / "arms")],
                   help="each run's arms directory (same order as --runs): budget stops backfilled from its episodes")
    p.add_argument("--out", default=str(RUNS / "difficulty_predictor.json"))
    p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("conditional")
    p.add_argument("--arms", default=str(ROOT / "data" / "targeting" / "debug-300" / "arms" / "student"))
    p.add_argument("--store", default=str(ROOT / "data" / "debug" / "gen" / "store"))
    p.add_argument("--out", default=str(RUNS / "difficulty_conditional.json"))
    p.set_defaults(fn=cmd_conditional)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
