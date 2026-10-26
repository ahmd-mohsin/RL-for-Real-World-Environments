"""The training monitor (swarm-pipeline/11 §4.4; ``factory.train_monitor``, ``observe.train_monitor``).

    python scripts/train_monitor.py report --roots DIR [DIR ...] [--label DIR=cohort ...] [--out OUT] [--repath-store STORE]
        [--judge N --judge-cap USD --env-file ../.env] [--wandb]
    python scripts/train_monitor.py watch --run RUN_DIR [--every 120] [--out OUT] [--wandb]
    python scripts/train_monitor.py health (--run RUN_DIR | --roots DIR ...) [--repath-store STORE] [--k 8] [--out OUT] [--wandb]
    python scripts/train_monitor.py fix --run RUN_DIR --defect fd-... --component C --version V --commit SHA [--after-step N]

``health``: the framework side (``factory.framework_health``): ranked framework defects with their component, impact and
reproduction, the tuning proposals and the versions per step; exit 3 while a CRITICAL defect is open (pause before the
next step). ``fix``: record a fix in the run's ``framework_fixes.jsonl`` (the health report then marks the defect fixed, or
recurring if it shows up after the fix's step).

``report``: general vs task-specific failures, trends, alerts and proposals over stored episodes (targeting arms, a
training run's ``rollouts/``, benchmark runs). ``watch``: the same over a live training run, re-read every ``--every``
seconds, each new step logged to W&B (project ``crucible-train``). Outputs under ``--out`` (default
``data/observe/train_monitor/<name>``, gitignored): ``report.json``, ``report.md``, ``report.html``, ``proposals.jsonl``.
The judge pass is capped (``--judge-cap``), reads only training-split failures, and never runs on a Qwen judge.
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

HERE = Path(__file__).resolve().parents[1]


def load_env_file(path: str) -> list[str]:
    """``KEY=VALUE`` lines into the environment, keeping variables already set; returns the names set, never a value."""
    names = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, _, v = ln.removeprefix("export ").partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.isidentifier() and k not in os.environ:
            os.environ[k] = v
            names.append(k)
    return names


def _write(out: Path, report: dict[str, Any], title: str) -> None:
    from crucible.factory.train_monitor import journal_proposals
    from crucible.observe.train_monitor import html, markdown

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "report.md").write_text(markdown(report, title=title))
    (out / "report.html").write_text(html(report, title=title))
    journal_proposals(report, out / "proposals.jsonl")


def _run(name: str, a: argparse.Namespace) -> Any:
    from crucible.observe.naming import project
    from crucible.observe.run import ObserveConfig, ObsRun

    return ObsRun(project("train"), name, job_type="monitor", config={"roots": getattr(a, "roots", None), "run": getattr(a, "run", None)},
                  cfg=ObserveConfig.from_env(use_wandb=bool(a.wandb)))


def cmd_report(a: argparse.Namespace) -> None:
    from crucible.factory.train_monitor import build_report, ingest, judge, repath_now, step_health

    labels = dict(x.split("=", 1) for x in a.label or [])
    eps, bundles = ingest(a.roots, labels=labels, tool_limit=a.tool_limit, keep_bundles=bool(a.judge))
    if a.repath_store:
        repath_now(eps, a.repath_store)
    jr: dict[str, Any] = {}
    if a.judge:
        if a.env_file:
            load_env_file(a.env_file)
        if not os.environ.get("TINKER_API_KEY"):
            raise SystemExit("TINKER_API_KEY is not in the environment (--env-file ../.env)")
        from crucible.factory.models import ModelBook, PromptJsonClient
        from crucible.train.bridges.tinker.sampler import ensure_cap, shared_meter

        ensure_cap(a.judge_cap)
        book = ModelBook(preset="swarm")
        spec = book.spec(a.judge_role)
        start = shared_meter().totals.usd
        jr = asyncio.run(judge([e for e in eps if not a.policy or a.policy in e.policy], bundles, PromptJsonClient(book.client(a.judge_role)), judge_model=spec.model, n=a.judge, cap_usd=a.judge_cap,
                               spent=lambda: shared_meter().totals.usd - start))
        jr["usd"] = round(shared_meter().totals.usd - start, 4)
        jr["model"] = spec.model
    steps = [r for root in a.roots for r in step_health(root)]
    report = build_report(eps, steps=steps, cohort_order=a.order or (), min_tasks=a.min_tasks, judge_result=jr, focus=a.policy or None)
    out = Path(a.out) if a.out else HERE / "data" / "observe" / "train_monitor" / (a.name or time.strftime("report-%Y%m%d-%H%M%S"))
    _write(out, report, a.name or "Training monitor")
    if a.wandb:
        from crucible.observe.train_monitor import log_report

        with _run(a.name or out.name, a) as run:
            log_report(run, report)
    g = report["general"]
    print(json.dumps({"out": str(out), "episodes": report["episodes"], "general": g["general"], "alerts": [x["kind"] for x in report["alerts"]],
                      "judge": jr}, default=str))


def cmd_watch(a: argparse.Namespace) -> None:
    from crucible.factory.train_monitor import build_report, ingest, step_health
    from crucible.observe.train_monitor import log_report, log_step

    run_dir = Path(a.run)
    out = Path(a.out) if a.out else HERE / "data" / "observe" / "train_monitor" / f"watch-{run_dir.name}"
    logged: set[int] = set()
    obs = _run(f"monitor-{run_dir.name}", a) if a.wandb else None
    try:
        while True:
            eps, _ = ingest([run_dir], tool_limit=a.tool_limit)
            steps = step_health(run_dir)
            report = build_report(eps, steps=steps)
            _write(out, report, f"Training monitor: {run_dir.name}")
            for row in report["trends"]["rows"]:
                s = int(row["order"])
                if s not in logged and row["at"].startswith("step"):
                    logged.add(s)
                    if obs is not None:
                        log_step(obs, row, s)
            for al in report["alerts"]:
                print(json.dumps({"alert": al}), flush=True)
            from crucible.factory import framework_health as fh
            from crucible.observe.train_monitor import health_markdown

            hv = fh.load_views([run_dir], tool_limit=a.tool_limit)
            hr = fh.health_report(hv, roots=[run_dir], run_dirs=[run_dir], journal=fh.read_journal([run_dir]))
            (out / "health.json").write_text(json.dumps(hr, indent=1, default=str))
            (out / "health.md").write_text(health_markdown(hr, title=f"Framework health: {run_dir.name}"))
            if hr["pause"]:
                print(json.dumps({"alert": {"kind": "framework_critical", "severity": "critical", "detail": "PAUSE before the next step: "
                                            + "; ".join(hr["pause_reason"][:3])}}), flush=True)
            if obs is not None:
                log_report(obs, report)
            if a.once or (run_dir / "done.json").exists():
                break
            time.sleep(a.every)
    finally:
        if obs is not None:
            obs.finish()
    print(json.dumps({"out": str(out), "steps_logged": sorted(logged)}))


def cmd_health(a: argparse.Namespace) -> int:
    """The framework-health report. Exit 3 when a CRITICAL defect is open: the operator pauses before the next step."""
    from crucible.factory import framework_health as fh
    from crucible.factory.train_monitor import build_report
    from crucible.observe.train_monitor import health_markdown, log_health

    roots = [a.run] if a.run else list(a.roots or [])
    if not roots:
        raise SystemExit("--run or --roots")
    labels = dict(x.split("=", 1) for x in a.label or [])
    views = fh.load_views(roots, labels=labels, tool_limit=a.tool_limit)
    run_dirs = [a.run] if a.run else [r for r in roots if (Path(r) / "steps.jsonl").exists()]
    mon = build_report([v.ep for v in views], focus=a.policy or None)
    modes = {r["mode"]: r["p"] for r in (mon.get("general") or {}).get("modes") or [] if r.get("p") is not None}
    h = fh.health_report(views, roots=roots, run_dirs=run_dirs, store=a.repath_store or None, journal=fh.read_journal(run_dirs), monitor_modes=modes, k=a.k)
    out = Path(a.out) if a.out else HERE / "data" / "observe" / "train_monitor" / (a.name or f"health-{Path(roots[0]).name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "health.json").write_text(json.dumps(h, indent=1, default=str))
    (out / "health.md").write_text(health_markdown(h, title=a.name or "Framework health"))
    if a.wandb:
        with _run(f"health-{out.name}", a) as run:
            log_health(run, h)
    print(json.dumps({"out": str(out), "episodes": h["episodes"], "by_severity": h["by_severity"], "by_component": h["by_component"], "pause": h["pause"],
                      "pause_reason": h["pause_reason"][:5]}, default=str))
    if h["pause"]:
        print("PAUSE before the next step: open CRITICAL framework defects (see health.md)", file=sys.stderr)
        return 3
    return 0


def cmd_fix(a: argparse.Namespace) -> int:
    from crucible.factory.framework_health import record_fix

    row = record_fix(a.run, defect=a.defect, component=a.component, version=a.version, commit=a.commit, after_step=a.after_step, note=a.note,
                     regraded=a.regraded or ())
    print(json.dumps(row))
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(HERE / "src"))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("report")
    s.add_argument("--roots", nargs="+", required=True)
    s.add_argument("--label", nargs="*", help="ROOT=cohort (default: the arm name, or the run id)")
    s.add_argument("--order", nargs="*", help="cohort order for trends (e.g. before after)")
    s.add_argument("--name", default="")
    s.add_argument("--out", default="")
    s.add_argument("--min-tasks", type=int, default=3)
    s.add_argument("--policy", default="", help="analyse only this policy (a substring of its id, e.g. Qwen3.6-35B-A3B); every model is still compared")
    s.add_argument("--tool-limit", type=int, default=300)
    s.add_argument("--repath-store", default="", help="a debug-lane window store: re-path the episodes under the current path rules")
    s.add_argument("--judge", type=int, default=0, help="judge-label this many training failures (0: facts only)")
    s.add_argument("--judge-cap", type=float, default=3.0)
    s.add_argument("--judge-role", default="judge")
    s.add_argument("--env-file", default="")
    s.add_argument("--wandb", action="store_true")
    w = sub.add_parser("watch")
    w.add_argument("--run", required=True)
    w.add_argument("--every", type=float, default=120.0)
    w.add_argument("--out", default="")
    w.add_argument("--tool-limit", type=int, default=300)
    w.add_argument("--once", action="store_true")
    w.add_argument("--wandb", action="store_true")
    h = sub.add_parser("health")
    h.add_argument("--run", default="", help="a training run directory (its steps.jsonl and framework_fixes.jsonl are read)")
    h.add_argument("--roots", nargs="*", help="stored episode roots instead of a run")
    h.add_argument("--label", nargs="*")
    h.add_argument("--name", default="")
    h.add_argument("--out", default="")
    h.add_argument("--policy", default="")
    h.add_argument("--tool-limit", type=int, default=300)
    h.add_argument("--k", type=int, default=8, help="the run's group size (for the group-size proposal)")
    h.add_argument("--repath-store", default="", help="a debug-lane window store: flag stored path verdicts the current rules flip")
    h.add_argument("--wandb", action="store_true")
    f = sub.add_parser("fix")
    f.add_argument("--run", required=True)
    f.add_argument("--defect", required=True)
    f.add_argument("--component", required=True)
    f.add_argument("--version", required=True, help="the new path-rules or verifier version the fix makes")
    f.add_argument("--commit", required=True)
    f.add_argument("--after-step", type=int, default=None)
    f.add_argument("--note", default="")
    f.add_argument("--regraded", nargs="*")
    a = ap.parse_args(argv)
    rc = {"report": cmd_report, "watch": cmd_watch, "health": cmd_health, "fix": cmd_fix}[a.cmd](a)
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
