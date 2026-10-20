#!/usr/bin/env python
"""Crucible observability: offline W&B runs and Weave traces, synced when a key exists (swarm-pipeline/11).

Everything is written under ``--root`` (default ``CRUCIBLE_OBSERVE_DIR`` or ``data/observe``). Commands:

- ``status``: the mode, whether ``WANDB_API_KEY`` is set (presence only), runs waiting for ``wandb sync``, spans waiting
  for the Weave upload;
- ``backfill PATH ...`` (``--force``): existing results (``docs/research/swarm-pipeline/runs/*.json``, Tinker run
  directories under ``data/tinker``, benchmark / compare directories, ``funnel.jsonl`` logs) into offline runs and traces;
- ``sync`` (``--entity``, ``--wandb-only`` / ``--weave-only``): ``wandb sync`` every offline run not yet synced, then the
  Weave spans (same call ids, so the links already in W&B tables open). Needs ``WANDB_API_KEY`` in the environment:
  ``set -a; . ../.env; set +a`` (the script never reads ``.env`` itself);
- ``trace EPISODE | ATTEMPT_DIR`` (``--project``, ``--html OUT``): one episode as a tree (turns, tool calls, Φ, the
  auditor's verdict) from the local spans, or straight from an attempt directory;
- ``episodes`` (``--project``, ``--failing``): the exported episodes with their links;
- ``links``: the Weave URL of every exported episode (after a sync with ``--entity`` or ``WANDB_ENTITY`` set);
- ``report`` (``--entity``, ``--group``, ``--push``): the W&B Report template (``report_template.json``); ``--push``
  saves it (needs the key);
- ``replay RUN_DIR``: a mirrored run (logged where ``wandb`` was not installed) into a W&B offline run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _cfg(args):
    from crucible.observe import ObserveConfig

    return ObserveConfig.from_env(root=args.root) if args.root else ObserveConfig.from_env()


def cmd_status(args) -> None:
    from crucible.observe.run import has_key, mirror_runs

    cfg = _cfg(args)
    synced = _synced(cfg.root)
    offline = [str(p) for p in sorted(cfg.root.glob("runs/*/*/wandb/offline-run-*"))]
    weave = {}
    for d in sorted((cfg.root / "weave").glob("*")):
        sp = d / "spans.jsonl"
        n = sum(1 for _ in sp.open()) if sp.exists() else 0
        done = len(json.loads((d / "synced.json").read_text())) if (d / "synced.json").exists() else 0
        weave[d.name] = {"spans": n, "uploaded": done, "episodes": sum(1 for x in (d / "index.jsonl").open() if '"kind": "episode"' in x)
                         if (d / "index.jsonl").exists() else 0}
    print(json.dumps({"root": str(cfg.root), "mode": cfg.mode, "WANDB_API_KEY_set": has_key(), "entity": cfg.entity, "mirrored_runs": len(mirror_runs(cfg.root)),
                      "offline_runs": len(offline), "offline_runs_not_synced": len([p for p in offline if p not in synced]), "weave": weave}, indent=1))


def cmd_backfill(args) -> None:
    from crucible.observe.backfill import backfill

    rows = backfill([Path(p) for p in args.paths], cfg=_cfg(args), force=args.force, kinds=args.kind or None)
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if k in ("kind", "source", "run", "views", "traces", "skipped", "error", "project")}, default=str))
    errs = [r for r in rows if r.get("error")]
    print(f"{len(rows)} sources: {sum(1 for r in rows if 'run' in r and not r.get('skipped'))} logged, "
          f"{sum(1 for r in rows if r.get('skipped'))} already backfilled, {len(errs)} failed", file=sys.stderr)


def _synced(root: Path) -> set[str]:
    p = root / "wandb_synced.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


def cmd_sync(args) -> None:
    from crucible.observe.run import has_key
    from crucible.observe.weave_sync import sync_all

    if not has_key():
        sys.exit("WANDB_API_KEY is not set (source the repo .env into this shell: set -a; . ../.env; set +a). Nothing was uploaded.")
    cfg = _cfg(args)
    if not args.weave_only:
        done = _synced(cfg.root)
        todo = [p for p in sorted(cfg.root.glob("runs/*/*/wandb/offline-run-*")) if str(p) not in done]
        for p in todo:
            cmd = [sys.executable, "-m", "wandb", "sync", str(p)]
            if args.entity:
                cmd += ["--entity", args.entity]
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            print(f"wandb sync {p.name}: {'ok' if r.returncode == 0 else 'FAILED ' + r.stderr[-300:]}")
            if r.returncode == 0:
                done.add(str(p))
                (cfg.root / "wandb_synced.json").write_text(json.dumps(sorted(done), indent=0))
    if not args.wandb_only:
        for row in sync_all(cfg.root, ent=args.entity):
            print(json.dumps(row))


def cmd_trace(args) -> None:
    from crucible.observe.traces import Span, TraceStore, episode_spans, render_html, render_text

    target = Path(args.episode)
    if target.is_dir() and (target / "record.json").exists():
        spans, _, _ = episode_spans(target, project=args.project or "local", max_chars=args.chars)
    else:
        cfg = _cfg(args)
        projects = [args.project] if args.project else [p.name for p in sorted((cfg.root / "weave").glob("*"))]
        spans: list[Span] = []
        for proj in projects:
            st = TraceStore(cfg.root, proj)
            spans = st.spans(args.episode) or [s for s in st.spans() if args.episode in s.episode]
            if spans:
                eps = sorted({s.episode for s in spans})
                if len(eps) > 1:
                    sys.exit("several episodes match:\n" + "\n".join(eps))
                break
        if not spans:
            sys.exit(f"no exported episode matches {args.episode!r} (scripts/observe.py episodes lists them)")
    print(render_text(spans, width=args.width))
    if args.html:
        Path(args.html).write_text(render_html(spans))
        print(f"wrote {args.html}", file=sys.stderr)


def cmd_episodes(args) -> None:
    from crucible.observe.traces import TraceStore

    cfg = _cfg(args)
    for d in sorted((cfg.root / "weave").glob("*")):
        if args.project and d.name != args.project:
            continue
        for key, r in TraceStore(cfg.root, d.name).index().items():
            if args.failing and (r.get("passed") or r.get("masked")):
                continue
            a = r.get("audit") or {}
            print(f"{d.name}\t{key}\t{r.get('arm')}\treward={r.get('reward')}\tpassed={r.get('passed')}\tmasked={r.get('masked')}\t"
                  f"cause={a.get('cause')}\t{r.get('url')}")


def cmd_links(args) -> None:
    from crucible.observe.naming import call_url
    from crucible.observe.traces import TraceStore

    cfg = _cfg(args)
    ent = args.entity or cfg.entity
    for d in sorted((cfg.root / "weave").glob("*")):
        st = TraceStore(cfg.root, d.name)
        moved = json.loads((st.dir / "links.json").read_text()) if (st.dir / "links.json").exists() else {}
        for key, r in st.index().items():
            cid = (moved.get(r["root"]) or {}).get("id", r["root"])
            print(f"{key}\t{call_url(d.name, cid, ent)}")


def cmd_report(args) -> None:
    from crucible.observe.report import build, push, write_template

    out = Path(args.out)
    write_template(out, group=args.group, model=args.model)
    print(f"wrote {out}")
    if args.entity:
        build(args.entity, group=args.group, model=args.model)  # offline: the spec builds
        print("report spec builds (wandb_workspaces)")
    if args.push:
        if not args.entity:
            sys.exit("--push needs --entity")
        print(push(args.entity, group=args.group, model=args.model))


def cmd_replay(args) -> None:
    from crucible.observe.run import replay

    r = replay(Path(args.run_dir), cfg=_cfg(args) if args.root else None)
    print(json.dumps({"run": str(r.dir), "sync": [str(p) for p in r.sync_dirs]}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="observe root (default CRUCIBLE_OBSERVE_DIR or data/observe)")
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("status")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("backfill")
    p.add_argument("paths", nargs="+")
    p.add_argument("--force", action="store_true")
    p.add_argument("--kind", action="append", choices=["train", "bench", "compare", "funnel", "report"])
    p.set_defaults(fn=cmd_backfill)
    p = sub.add_parser("sync")
    p.add_argument("--entity")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--wandb-only", action="store_true")
    g.add_argument("--weave-only", action="store_true")
    p.set_defaults(fn=cmd_sync)
    p = sub.add_parser("trace")
    p.add_argument("episode", help="episode key (<run>/<sample>/<attempt>, or a unique part of it) or an attempt directory")
    p.add_argument("--project")
    p.add_argument("--html")
    p.add_argument("--chars", type=int, default=2000)
    p.add_argument("--width", type=int, default=200)
    p.set_defaults(fn=cmd_trace)
    p = sub.add_parser("episodes")
    p.add_argument("--project")
    p.add_argument("--failing", action="store_true")
    p.set_defaults(fn=cmd_episodes)
    p = sub.add_parser("links")
    p.add_argument("--entity")
    p.set_defaults(fn=cmd_links)
    p = sub.add_parser("report")
    p.add_argument("--entity")
    p.add_argument("--group", help="the comparison's W&B group (tasks-<manifest hash>)")
    p.add_argument("--model")
    p.add_argument("--out", default="report_template.json")
    p.add_argument("--push", action="store_true")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("replay")
    p.add_argument("run_dir")
    p.set_defaults(fn=cmd_replay)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
