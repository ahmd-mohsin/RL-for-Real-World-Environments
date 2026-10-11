"""M15 studio commands (loaded lazily by ``crucible.cli.main``). Each command wraps a library function. Reporting
commands print a short human summary, or the full JSON with ``--json``, and finish with the next commands to run."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import click


def _emit(obj: dict[str, Any], as_json: bool, lines: list[str]) -> None:
    if as_json:
        click.echo(json.dumps(obj, indent=1, sort_keys=True, default=str))
        return
    for line in lines:
        click.echo(line)
    for n in obj.get("next") or []:
        click.echo(f"  next: {n}")


# ---- pack ----------------------------------------------------------------------------------------------------------------
@click.group("pack")
def pack() -> None:
    """Create and validate domain packs."""


@pack.command("new")
@click.argument("name")
@click.option("--dir", "dest", type=click.Path(path_type=Path), default=None, help="where to create it (default: packs/<name>)")
@click.option("--json", "as_json", is_flag=True)
def pack_new(name: str, dest: Path | None, as_json: bool) -> None:
    """Scaffold a pack from packs/_template (conflict-safe: identical files are kept, any conflict writes nothing)."""
    from crucible.studio.pack import PACKS_ROOT, new

    r = new(name, dest or PACKS_ROOT / name)
    _emit(r, as_json, [f"created {len(r['created'])} file(s), kept {len(r['existing'])} identical: {r['path']}"])


@pack.command("check")
@click.argument("ref", required=False)
@click.option("--level", type=click.Choice(["static", "fixture", "semantic"]), default="static", show_default=True)
@click.option("--all", "all_", is_flag=True, help="every pack under packs/")
@click.option("--allow", default="", help="requirements this machine satisfies, e.g. gpu,paid (docker is detected)")
@click.option("--json", "as_json", is_flag=True)
def pack_check(ref: str | None, level: str, all_: bool, allow: str, as_json: bool) -> None:
    """Validate a pack. static: manifest, placeholders, entry point, hooks, ingest; fixture: designed cases grade exactly;
    semantic: gold passes and no-op fails. Exit 0 = pass or warn, 1 = a check failed."""
    from crucible.studio.pack import all_packs, check

    refs = [str(p) for p in all_packs()] if all_ else [ref or "."]
    allowed = tuple(x for x in allow.split(",") if x)
    reports = [asyncio.run(check(r, level=level, allow=allowed)) for r in refs]
    if as_json:
        click.echo(json.dumps([r.to_json() for r in reports] if all_ else reports[0].to_json(), indent=1, sort_keys=True))
    else:
        for r in reports:
            click.echo(f"{r.pack} ({r.level}): {'ok' if r.ok else 'FAILED'}")
            for c in r.results:
                click.echo(f"  {c.status:5s} {c.id:22s} {c.detail}")
    if not all(r.ok for r in reports):
        raise SystemExit(1)


# ---- label / build ---------------------------------------------------------------------------------------------------------
@click.command("label")
@click.option("--pack", "pack_ref", default=None, help="label a pack's workflow logs (M3 -> M4 -> M5/M6)")
@click.option("--run", "run_dir", type=click.Path(path_type=Path, exists=True), default=None, help="label an M10 run directory")
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/label"), show_default=True)
@click.option("--filter", "expr", default=None, help='per-task filter, e.g. "0 < avg < 1" (names: avg best worst n n_correct solved)')
@click.option("--select", type=click.Choice(["correct", "best", "all"]), default="correct", show_default=True, help="which episodes become SFT rows")
@click.option("--max-per-task", type=int, default=None)
@click.option("--threshold", type=float, default=None, help="reward that counts as solved (default: the pack's)")
@click.option("--margin", type=float, default=0.25, show_default=True, help="minimum reward gap for a best-vs-worst DPO pair")
@click.option("--mask-negative", is_flag=True, help="exclude assistant turns with negative credit from the SFT loss")
@click.option("--dry-run", is_flag=True)
@click.option("--json", "as_json", is_flag=True)
def label(pack_ref, run_dir, out, expr, select, max_per_task, threshold, margin, mask_negative, dry_run, as_json) -> None:
    """Workflows -> traces with dense rewards, plus SFT / DPO / KTO datasets (TRL formats)."""
    from crucible.studio.label import label as run

    s = asyncio.run(run(out=out, pack=pack_ref, run=run_dir, filter=expr, select=select, max_per_task=max_per_task, threshold=threshold, margin=margin,
                        mask_negative=mask_negative, dry_run=dry_run))
    if dry_run:
        return _emit(s, as_json, [json.dumps(s, indent=1, default=str)])
    d = s["datasets"]
    lines = [f"labelled {s.get('traces', s.get('episodes'))} {'traces' if 'traces' in s else 'episodes'} -> {out}",
             f"  sft {d['sft']}  dpo {d['dpo']} ({', '.join(f'{k} {v}' for k, v in d['dpo_kinds'].items())})  kto {d['kto']}"]
    if "outcomes" in s:
        lines.append(f"  outcomes {s['outcomes']}  signals {sum(s['signals'].values())}  credited {s['credited']}")
    _emit(s, as_json, lines)


@click.command("build")
@click.argument("pack_ref")
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/build"), show_default=True)
@click.option("--fresh", "n_fresh", type=int, default=2, show_default=True, help="fresh benchmark items from the pack's generator")
@click.option("--k", type=int, default=1, show_default=True, help="samples per task and policy")
@click.option("--dry-run", is_flag=True)
@click.option("--json", "as_json", is_flag=True)
def build(pack_ref, out, n_fresh, k, dry_run, as_json) -> None:
    """Workflows -> RL tasks and a benchmark: ingest, forge, accept, export, bench."""
    from crucible.studio.build import build as run

    s = asyncio.run(run(pack_ref, out=out, n_fresh=n_fresh, k=k, dry_run=dry_run))
    if dry_run:
        return _emit(s, as_json, [json.dumps(s, indent=1, default=str)])
    _emit(s, as_json, [f"ingested {s['ingest']['traces']} traces; forged {s['forge']['tasks']} tasks; accepted {s['accept']['accepted']}/{s['accept']['total']}",
                       f"  exported {s['export']['canonical']} (canonical reward.json + steps.jsonl) -> {s['export']['dir']}",
                       f"  bench {s['bench']['items']}: " + ", ".join(f"{p} {r}" for p, r in s["bench"]["models"].items()),
                       f"  report {s['bench']['report']}"])


# ---- the module steps --------------------------------------------------------------------------------------------------
@click.command("ingest")
@click.argument("pack_ref")
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/m3/store"), show_default=True)
@click.option("--json", "as_json", is_flag=True)
def ingest(pack_ref, out, as_json) -> None:
    """M3: fetch, normalize, redact and store the pack's workflow sources (idempotent)."""
    from crucible.studio.label import ingest as run
    from crucible.studio.pack import find_pack, load_manifest

    d = find_pack(pack_ref)
    ts = run(d, load_manifest(d), out)
    _emit({"traces": len(ts), "store": str(out)}, as_json, [f"{len(ts)} traces in {out}"])


@click.command("signals")
@click.argument("pack_ref")
@click.option("--json", "as_json", is_flag=True)
def signals(pack_ref, as_json) -> None:
    """M4: expert signals, outcome labels and pivots per ingested trace."""
    import tempfile

    from crucible.signals.extract import extract, mapper_for
    from crucible.studio.label import ingest as run
    from crucible.studio.pack import find_pack, load_manifest

    d = find_pack(pack_ref)
    with tempfile.TemporaryDirectory() as tmp:
        ts = run(d, load_manifest(d), Path(tmp))
    counts: dict[str, int] = {}
    for t in ts:
        if mapper_for(t.family) is not None:
            for s in extract(t).signals:
                counts[s.type] = counts.get(s.type, 0) + 1
    _emit({"traces": len(ts), "signals": dict(sorted(counts.items()))}, as_json, [f"{len(ts)} traces; signals {dict(sorted(counts.items()))}"])


@click.command("forge")
@click.argument("pack_ref")
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/forge"), show_default=True)
@click.option("--json", "as_json", is_flag=True)
def forge(pack_ref, out, as_json) -> None:
    """M8: construct tasks from the pack's ingested workflows (the pack's forge hook)."""
    from crucible.studio.label import ingest as run
    from crucible.studio.pack import find_pack, load_manifest, resolve

    d = find_pack(pack_ref)
    m = load_manifest(d)
    if not m.hooks.forge:
        raise ValueError(f"pack {m.name} has no hooks.forge")
    run(d, m, Path(out) / "store")
    col = asyncio.run(resolve(d, m.hooks.forge)(Path(out) / "forge", Path(out) / "store"))
    _emit({"tasks": [t.task_id for t, _ in col], "dir": str(Path(out) / "forge" / "tasks")}, as_json, [f"forged {len(col)} tasks in {Path(out) / 'forge' / 'tasks'}"])


@click.command("accept")
@click.argument("pack_ref")
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/forge"), show_default=True)
@click.option("--json", "as_json", is_flag=True)
def accept(pack_ref, out, as_json) -> None:
    """M9 (model-free checks A1, A2, A3, A4-lite, A5, A7, A8) on the pack's forged tasks. The full contract needs Docker and
    paid calls: python -m crucible_pack_toy.accept_gate."""
    from crucible.accept.contract import accept as run_accept
    from crucible.studio.label import ingest as run
    from crucible.studio.pack import find_pack, load_manifest, resolve

    d = find_pack(pack_ref)
    m = load_manifest(d)
    run(d, m, Path(out) / "store")
    col = asyncio.run(resolve(d, m.hooks.forge)(Path(out) / "forge", Path(out) / "store"))
    reps = {t.task_id: asyncio.run(run_accept(t, resolve=r)) for t, r in col}
    res = {k: {"ok": v.ok, "checks": {c.id: c.status for c in v.checks}} for k, v in reps.items()}
    _emit({"accepted": sum(v.ok for v in reps.values()), "total": len(reps), "tasks": res}, as_json,
          [f"{k}: {'accepted' if v['ok'] else 'REJECTED'} {v['checks']}" for k, v in res.items()])
    if not all(v.ok for v in reps.values()):
        raise SystemExit(1)


@click.command("train")
@click.argument("job", type=click.Path(path_type=Path, exists=True))
@click.option("--work", type=click.Path(path_type=Path), default=Path("runs/launch"), show_default=True)
def train(job, work) -> None:
    """M12: run a training job spec (TOML; see crucible.train.launch)."""
    import tomllib

    from crucible.train.launch import run_job

    out = asyncio.run(run_job(tomllib.loads(Path(job).read_text()), Path(work)))
    click.echo(json.dumps(out, indent=1, default=str))


# ---- bench / run -------------------------------------------------------------------------------------------------------
@click.group("bench")
def bench() -> None:
    """M13: run, report and compare benchmarks."""


async def _bench_pack(pack_ref: str, out: Path, n_fresh: int, k: int, key: bytes, policies: str | None) -> dict[str, Any]:
    from crucible.actors.policy import EpisodeLimits
    from crucible.bench import dynamic
    from crucible.bench.run import run_bench
    from crucible.runtime.harness.native import NativeHarness
    from crucible.studio.pack import find_pack, load_manifest, resolve

    d = find_pack(pack_ref)
    m = load_manifest(d)
    if not m.hooks.tasks and not m.hooks.generator:
        raise ValueError(f"pack {m.name} declares neither hooks.tasks nor hooks.generator: nothing to benchmark")
    anchors = [(getattr(t, "id", str(i)), "anchor", t) for i, (t, _) in enumerate(resolve(d, m.hooks.tasks)())] if m.hooks.tasks else []
    gens = {"generated": resolve(d, m.hooks.generator)} if m.hooks.generator and n_fresh else {}
    bm = await dynamic.build(f"{m.name}-bench", "v1", key=key, anchors=anchors, generators=gens, n_fresh=n_fresh if gens else 0)
    pols = resolve(d, policies or m.hooks.policies)() if (policies or m.hooks.policies) else None
    if not pols:
        raise ValueError(f"pack {m.name} has no hooks.policies; pass --policies module:attr returning PolicyRefs")
    try:
        res = await run_bench(f"{m.name}-bench", root=out, tasks=[i.task for i in bm.items], policies=pols, harnesses=[NativeHarness(limits=EpisodeLimits(max_steps=1000))],
                              k=k, stages_of=resolve(d, m.hooks.stages) if m.hooks.stages else None, extras={"manifest": bm.manifest()})
    finally:
        for i in bm.items:
            if callable(getattr(i.task, "cleanup", None)):
                i.task.cleanup()
    return res


@bench.command("run")
@click.argument("pack_ref")
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/runs"), show_default=True)
@click.option("--fresh", "n_fresh", type=int, default=2, show_default=True)
@click.option("--k", type=int, default=2, show_default=True)
@click.option("--key", default="crucible-bench-v1", show_default=True, help="the item-seed key (keep it secret for a published version)")
@click.option("--policies", default=None, help="module:attr returning PolicyRefs (default: the pack's hooks.policies)")
@click.option("--json", "as_json", is_flag=True)
def bench_run(pack_ref, out, n_fresh, k, key, policies, as_json) -> None:
    """A benchmark on a pack: its tasks as anchors + fresh generated items, every step metric, an HTML report."""
    res = asyncio.run(_bench_pack(pack_ref, out, n_fresh, k, key.encode(), policies))
    lines = [f"{m['policy']}: mean {m['overall']['deterministic'].get('mean_reward')} pass@1 {m['overall']['deterministic'].get('pass@1')} "
             f"AUC {m['overall']['deterministic'].get('progress_auc')}" for m in res["summary"]["models"]]
    _emit({"summary": res["summary"], "paths": res["paths"], "next": [f"open {res['paths']['report']}"]}, as_json, lines + [f"report {res['paths']['report']}"])


@bench.command("report")
@click.argument("run_dir", type=click.Path(path_type=Path, exists=True))
@click.option("--name", default=None)
def bench_report(run_dir, name) -> None:
    """(Re)compute every metric and the HTML report for an existing M10 run directory."""
    from crucible.bench.run import report_run

    res = report_run(name or Path(run_dir).name, Path(run_dir))
    click.echo(res["paths"]["report"])


@bench.command("compare")
@click.argument("base_dir", type=click.Path(path_type=Path, exists=True))
@click.argument("cand_dir", type=click.Path(path_type=Path, exists=True))
@click.option("--base-policy", default=None)
@click.option("--cand-policy", default=None)
@click.option("--threshold", type=float, default=1.0, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def bench_compare(base_dir, cand_dir, base_policy, cand_policy, threshold, as_json) -> None:
    """Paired comparison of two runs (or two policies in one run): flips, exact McNemar, paired bootstrap CI."""
    from crucible.bench.compare import compare, to_markdown
    from crucible.bench.metrics import load_run

    b = [e for e in load_run(base_dir) if base_policy in (None, e.policy)]
    c = [e for e in load_run(cand_dir) if cand_policy in (None, e.policy)]
    r = compare(b, c, threshold=threshold)
    click.echo(json.dumps(r, indent=1, sort_keys=True) if as_json else to_markdown(base_policy or str(base_dir), cand_policy or str(cand_dir), r))


@click.command("model")
@click.argument("pack_ref")
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/model"), show_default=True)
@click.option("--json", "as_json", is_flag=True)
def model(pack_ref, out, as_json) -> None:
    """M5: the workflow model (milestones, stages) mined from a pack's ingested workflow logs."""
    import tempfile

    from crucible.studio.label import ingest as run
    from crucible.studio.pack import find_pack, load_manifest, resolve

    d = find_pack(pack_ref)
    m = load_manifest(d)
    if not m.hooks.model:
        raise ValueError(f"pack {m.name} has no hooks.model")
    with tempfile.TemporaryDirectory() as tmp:
        traces = run(d, m, Path(tmp), labelled_only=True)
    wm = asyncio.run(resolve(d, m.hooks.model)(traces))
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "workflow.json").write_text(json.dumps(wm, indent=1, sort_keys=True, default=str) + "\n")
    _emit({"milestones": len(wm.get("milestones", [])), "stages": len(wm.get("stages", [])), "file": str(Path(out) / "workflow.json")}, as_json,
          [f"{len(wm.get('milestones', []))} milestones, {len(wm.get('stages', []))} stages -> {Path(out) / 'workflow.json'}"])


@click.command("export")
@click.option("--to", "target", type=click.Choice(["canonical", "atif", "nemo-gym", "openenv"]), required=True)
@click.option("--pack", "pack_ref", default=None)
@click.option("--run", "run_dir", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/export"), show_default=True)
@click.option("--json", "as_json", is_flag=True)
def export(target, pack_ref, run_dir, out, as_json) -> None:
    """M11: export a pack's tasks (canonical replay, NeMo-Gym server, OpenEnv environment) or a run's episodes (ATIF)."""
    from crucible.studio.export import export as run

    r = asyncio.run(run(target, out=out, pack=pack_ref, run=run_dir))
    _emit(r, as_json, [json.dumps(r, indent=1, default=str)])
