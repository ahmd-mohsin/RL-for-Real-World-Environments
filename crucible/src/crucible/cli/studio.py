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
@click.option("--level", type=click.Choice(["static", "fixture", "semantic", "accept"]), default="static", show_default=True)
@click.option("--all", "all_", is_flag=True, help="every pack under packs/, each checked in its own process")
@click.option("--allow", default="", help="requirements this machine satisfies, e.g. gpu,paid (docker and gpu are detected)")
@click.option("--allow-skip", is_flag=True, help="exit 0 although a requested level was skipped for a missing requirement")
@click.option("--publish", is_flag=True, help="publication gate: template text, no authors, no README and a wheel without pack.yaml fail")
@click.option("--json", "as_json", is_flag=True)
def pack_check(ref: str | None, level: str, all_: bool, allow: str, allow_skip: bool, publish: bool, as_json: bool) -> None:
    """Validate a pack. static: manifest, placeholders, licence, pyproject, entry point, hooks, ingest; fixture: designed
    cases grade exactly, in range, deterministically; semantic: gold reaches the top, no-op the floor, generated items have
    an oracle, adversaries stay bounded; accept: the M9 contract. Exit 0 = pass or warn, 1 = a check failed, 2 = a level
    was skipped (missing requirement, without --allow-skip), 3 = internal error."""
    from crucible.studio.pack import EXIT_INTERNAL, all_packs, check, check_in_subprocess

    allowed = tuple(x for x in allow.split(",") if x)
    kw = {"level": level, "allow": allowed, "allow_skip": allow_skip, "publish": publish}
    if all_:  # one process per pack (M14 audit V7 / G2 pick 5): every report, whatever another pack does
        reports = [check_in_subprocess(p, **kw) for p in all_packs()]
    else:
        try:
            reports = [asyncio.run(check(ref or ".", **kw))]
        except (ValueError, LookupError, FileNotFoundError, PermissionError):
            raise  # a user error (no such pack): one line, exit 1
        except Exception as e:  # noqa: BLE001 - not the pack's code (its hooks are caught as check failures): crucible's
            click.echo(f"internal error: {type(e).__name__}: {e}", err=True)
            raise SystemExit(EXIT_INTERNAL) from None
    if as_json:
        click.echo(json.dumps([r.to_json() for r in reports] if all_ else reports[0].to_json(), indent=1, sort_keys=True, default=str))
    else:
        for r in reports:
            head = "ok" if r.ok else ("FAILED" if any(c.status == "fail" for c in r.results) else "INCOMPLETE")
            click.echo(f"{r.pack} ({r.level}): {head}" + "".join(f"; {lv}: SKIPPED ({why})" for lv, why in r.skipped))
            for c in r.results:
                click.echo(f"  {c.status:5s} {c.id:22s} {c.detail}")
    code = max((r.exit_code for r in reports), key=lambda c: (c == EXIT_INTERNAL, c == 1, c == 2), default=0)
    if code:
        raise SystemExit(code)


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
@click.option("--docker", is_flag=True, help="Docker is available: run the acceptance checks that need it")
@click.option("--paid", is_flag=True, help="allow the acceptance checks that make paid model calls")
@click.option("--json", "as_json", is_flag=True)
def build(pack_ref, out, n_fresh, k, dry_run, docker, paid, as_json) -> None:
    """Workflows -> RL tasks and a benchmark: ingest, forge, accept (full contract), export, bench."""
    from crucible.accept.model import Need
    from crucible.studio.build import build as run

    needs = tuple(n for n, on in ((Need.DOCKER, docker), (Need.PAID, paid)) if on)
    s = asyncio.run(run(pack_ref, out=out, n_fresh=n_fresh, k=k, dry_run=dry_run, needs=needs))
    if dry_run:
        return _emit(s, as_json, [json.dumps(s, indent=1, default=str)])
    a = s["accept"]
    head = (f"ingested {s['ingest']['traces']} traces; forged {s['forge']['tasks']} tasks; accepted {a['accepted']}/{a['total']}, "
            f"provisional {a['provisional']} (checks pending on needs this run lacks), rejected {a['rejected']}")
    _emit(s, as_json, [head,
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
@click.option("--docker", is_flag=True, help="Docker is available: run the checks that need it")
@click.option("--paid", is_flag=True, help="allow the checks that make paid model calls")
@click.option("--json", "as_json", is_flag=True)
def accept(pack_ref, out, docker, paid, as_json) -> None:
    """M9: the full Acceptance Contract on the pack's forged tasks (a pack without hooks.forge: its fixture tasks, anchors
    and a generated item). Each task is accepted, provisional (nothing failed; checks that need Docker or paid calls are
    pending without --docker/--paid) or rejected. Exit code 1 if any is rejected."""
    from crucible.accept.model import Need
    from crucible.accept.run import accept_task, verdict
    from crucible.studio.label import ingest as run
    from crucible.studio.pack import find_pack, load_manifest, resolve

    d = find_pack(pack_ref)
    m = load_manifest(d)
    needs = [n for n, on in ((Need.DOCKER, docker), (Need.PAID, paid)) if on]
    import contextlib

    from crucible.studio.pack import isolated

    res = {}
    with isolated(d, m.package) if not m.hooks.forge else contextlib.nullcontext():
        if m.hooks.forge:
            run(d, m, Path(out) / "store")
            col = asyncio.run(resolve(d, m.hooks.forge)(Path(out) / "forge", Path(out) / "store"))
        else:  # M14 audit G2 D3: a pack without a forge hook is accepted on its own tasks (fixtures, anchors, generated items)
            from crucible.studio.accept_adapter import pack_accept_tasks

            col = [(t, lambda s: False) for t in asyncio.run(pack_accept_tasks(d, m, n_fresh=1))]
        try:
            for t, r in col:
                rep = asyncio.run(accept_task(t, needs=needs, cache_root=Path(out) / "accept", resolve=r))
                res[t.task_id] = {**verdict(rep), "checks": {c: str(x.status) for c, x in rep.results.items()}}
        finally:
            for t, _ in col if not m.hooks.forge else []:
                t.cleanup()
    states = [v["state"] for v in res.values()]
    _emit({"accepted": states.count("accepted"), "provisional": states.count("provisional"), "rejected": states.count("rejected"),
           "total": len(res), "tasks": res}, as_json, [f"{k}: {v['state'].upper()} {v['blocking']}" for k, v in res.items()])
    if "rejected" in states:
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


def _ks(text: str | None) -> list[int] | None:
    return [int(x) for x in text.split(",") if x.strip()] if text else None


def _model_lines(summary: dict[str, Any]) -> list[str]:
    """One line per model: the verifiers dual headline with its denominators (G1 §4), then every warning (H1 V5)."""
    from crucible.bench.report import headline, warnings

    lines = []
    for m in summary["models"]:
        d = m["overall"]["deterministic"]
        ks = summary.get("ks") or [1]
        lines.append(f"{m['policy']}@{m['harness']}{' (control)' if m.get('control') else ''}: mean {headline(d)}; "
                     + ", ".join(f"pass@{k} {d.get(f'pass@{k}')}" for k in ks) + f"; AUC {d.get('progress_auc')}")
    return lines + [f"  warning: {w}" for w in (summary.get("warnings") if summary.get("warnings") is not None else warnings(summary))]


def _import(ref: str) -> Any:
    import importlib

    mod, attr = ref.split(":", 1)
    obj: Any = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _harbor_tasks(dirs: tuple[Path, ...]) -> list[Any]:
    """Harbor task dirs in place (campaign B17). Two dirs with the same task name get their parent's name as a prefix
    (B7: Lego-X and PI-Lego ``task_00000`` collided); the oracle is the task's own ``solution/`` replayed under /tmp."""
    from crucible.core.schema.actions import ToolCall
    from crucible.interop.harbor.inplace import HarborInPlaceTask

    tasks = [HarborInPlaceTask(d) for d in dirs]
    ids = [t.id for t in tasks]
    for t in tasks:
        if ids.count(t.id) > 1:
            t.id, t.world_id = f"harbor/{t.dir.parent.name}/{t.id.split('/', 1)[1]}", f"harbor:{t.dir.parent.name}/{t.world_id.split(':', 1)[1]}"

        def oracle(t=t) -> list[Any] | None:
            sol = t.dir / "solution"
            if not (sol / "solve.sh").exists():
                return None
            acts = [ToolCall(tool_call_id=f"o{i}", function_name="term.write_file",
                             arguments={"path": f"/tmp/crucible_oracle/{f.relative_to(sol).as_posix()}", "content": f.read_text()})
                    for i, f in enumerate(sorted(p for p in sol.rglob("*") if p.is_file()))]
            return [*acts, ToolCall(tool_call_id="orun", function_name="term.exec",
                                    arguments={"command": "bash /tmp/crucible_oracle/solve.sh > /tmp/crucible_oracle/oracle.txt 2>&1; echo rc=$?", "timeout_s": 1800})]

        t.oracle_actions = oracle
    return tasks


async def _awaited(x: Any) -> Any:
    """A hook's result, awaited when the hook is a coroutine function (M14 campaign B6: coding's ``tasks`` is async;
    ``pack check`` awaited it and ``bench run`` iterated the coroutine)."""
    import inspect

    return await x if inspect.isawaitable(x) else x


async def _bench_pack(pack_ref: str | None, out: Path, n_fresh: int, k: int, key_file: Path | None, policies: str | None, ks: list[int] | None,
                      *, controls: bool = True, anchors_path: Path | None = None, harbor: tuple[Path, ...] = (), tasks_ref: str | None = None,
                      allow: tuple[str, ...] = (), notes: list[str] | None = None) -> dict[str, Any]:
    """A pack (anchors + fresh validated items) or plain tasks (``--harbor`` dirs, ``--tasks module:attr``), with the
    oracle / no-op controls first (campaign B4/B5) and the anchor parameters saved for the next version (V12)."""
    from crucible.actors.policy import EpisodeLimits
    from crucible.bench import dynamic, irt
    from crucible.bench.run import bench_key, gold_and_noop_check, run_bench
    from crucible.runtime.harness.native import NativeHarness
    from crucible.studio.pack import find_pack, load_manifest, resolve

    anchors = irt.load_anchors(anchors_path) if anchors_path else None
    harness = NativeHarness(limits=EpisodeLimits(max_steps=1000))
    if pack_ref is None:  # plain tasks: no generator, no fresh items
        if not policies:
            raise ValueError("plain tasks need --policies module:attr returning PolicyRefs")
        tasks = _harbor_tasks(harbor) if harbor else [x[0] if isinstance(x, tuple) else x for x in await _awaited(_import(tasks_ref)())]  # type: ignore[arg-type]
        name = "harbor" if harbor else tasks_ref.split(":")[0].rsplit(".", 1)[-1]  # type: ignore[union-attr]
        return await run_bench(name, root=out, tasks=tasks, policies=list(await _awaited(_import(policies)())), harnesses=[harness], k=k, ks=ks,
                               controls=controls, anchors=anchors)
    from crucible.studio.pack import missing_requirements, policy_refs

    d = find_pack(pack_ref)
    m = load_manifest(d)
    if miss := missing_requirements(m, allow):  # M14 campaign B6: the bench ran a pack's live tasks whatever it requires
        raise ValueError(f"pack {m.name} requires {', '.join(miss)}, which this machine does not provide: pass --allow {','.join(miss)} "
                         "where it does (its tasks may not run, or may not mean what they claim, without it)")
    if not m.hooks.tasks and not m.hooks.generator:
        raise ValueError(f"pack {m.name} declares neither hooks.tasks nor hooks.generator: nothing to benchmark")
    if not m.reward.higher_is_better:  # M14 audit V12: every bench metric counts reward >= threshold as solved
        raise ValueError(f"pack {m.name} declares reward.higher_is_better: false; bench metrics (pass@k, IRT, controls) assume higher is "
                         "better, so a lower-is-better pack is refused rather than reported upside down")
    lo, hi = m.reward.range
    thr = m.label.threshold  # V12: the pack's solved threshold and top of range, not 1.0
    anchor_items = []
    pol_hook = policies or m.hooks.policies
    for i, x in enumerate(await _awaited(resolve(d, m.hooks.tasks)()) if m.hooks.tasks else []):
        t, gold = x if isinstance(x, tuple) else (x, None)
        if gold and not hasattr(t, "bench_gold"):
            t.bench_gold = list(gold)  # the oracle control replays the pack's gold (hooks.tasks pairs a task with it)
        anchor_items.append((getattr(t, "id", str(i)), "anchor", t))
    gens = {"generated": resolve(d, m.hooks.generator)} if m.hooks.generator and n_fresh else {}  # called with item_id= (dynamic V3)
    rdir = Path(out) / f"bench-{m.name}-bench"
    key = bench_key(rdir, key_file=key_file)
    try:
        bm = await dynamic.build(f"{m.name}-bench", "v1", key=key, anchors=anchor_items, generators=gens, n_fresh=n_fresh if gens else 0,
                                 checks=[gold_and_noop_check(rdir / "validate", best=hi, threshold=thr)] if gens else (), registry=Path(out) / "bench.keys.jsonl")
    except dynamic.ShortBenchmark as e:
        man = e.benchmark.manifest()
        raise ValueError(f"only {man['counts']['fresh']} of {n_fresh} fresh items passed validation; yields {man['yields']}; "
                         f"rejected {[r.get('reason') or r for r in man['rejected']][:5]}") from None
    if pol_hook:
        pols = policy_refs(await _awaited(resolve(d, pol_hook)()))
        if not pols:
            raise ValueError(f"{pol_hook} returned no policies; pass --policies module:attr returning PolicyRefs")
    else:  # B6: a pack without hooks.policies benches its own gold and a no-op (model-free), and says so
        from crucible.actors.policy import ScriptedPolicy
        from crucible.bench.run import _oracle_actions
        from crucible.runtime.harness.native import PolicyRef

        pols = [PolicyRef(id="pack:gold", make=lambda a: {"agent": ScriptedPolicy("agent", _oracle_actions(a.task) or [])}),
                PolicyRef(id="pack:noop", make=lambda a: {"agent": ScriptedPolicy("agent", [])})]
        if notes is not None:
            notes.append(f"pack {m.name} has no hooks.policies: benchmarked its gold and a no-op; pass --policies module:attr for models")
    try:
        res = await run_bench(f"{m.name}-bench", root=out, tasks=[i.task for i in bm.items], policies=pols, harnesses=[harness],
                              k=k, ks=ks, threshold=thr, stages_of=resolve(d, m.hooks.stages) if m.hooks.stages else None,
                              extras={"manifest": bm.manifest(), "reward_range": [lo, hi]}, controls=controls, anchors=anchors)
    finally:
        for i in bm.items:
            if callable(getattr(i.task, "cleanup", None)):
                i.task.cleanup()
    fit = res.get("irt_fit")
    ids = [i.id for i in bm.items if i.kind == "anchor"]
    if fit is not None and all(i in fit.items for i in ids) and ids:
        res["paths"]["anchors"] = str(irt.save_anchors(rdir / "anchors.json", {i: fit.items[i] for i in ids}, meta={"bench": f"{m.name}-bench", "version": "v1"}))
    return res


@bench.command("run")
@click.argument("pack_ref", required=False)
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/runs"), show_default=True)
@click.option("--fresh", "n_fresh", type=int, default=2, show_default=True)
@click.option("--k", type=int, default=2, show_default=True)
@click.option("--ks", default=None, help="pass@k columns, comma-separated (default: 1, powers of two up to k, and k)")
@click.option("--key-file", type=click.Path(path_type=Path, exists=True), default=None,
              help="the item-seed key (else $CRUCIBLE_BENCH_KEY / $CRUCIBLE_BENCH_KEY_FILE, else a new key kept in the run's bench.key)")
@click.option("--policies", default=None, help="module:attr returning PolicyRefs (default: the pack's hooks.policies)")
@click.option("--controls/--no-controls", default=True, show_default=True,
              help="run the oracle and no-op controls first; tasks whose oracle fails or whose no-op passes are excluded")
@click.option("--anchors", "anchors_path", type=click.Path(path_type=Path, exists=True), default=None, help="anchors.json of an earlier version (IRT linking)")
@click.option("--harbor", type=click.Path(path_type=Path, exists=True, file_okay=False), multiple=True, help="a Harbor task dir, run in place (repeatable)")
@click.option("--tasks", "tasks_ref", default=None, help="module:attr returning tasks (NeMo-Gym rows, EOG, OpenEnv, ... wrapped as RunTasks)")
@click.option("--allow", default="", help="requirements of the pack this machine satisfies, e.g. gpu,paid (docker and gpu are detected)")
@click.option("--json", "as_json", is_flag=True)
def bench_run(pack_ref, out, n_fresh, k, ks, key_file, policies, controls, anchors_path, harbor, tasks_ref, allow, as_json) -> None:
    """A benchmark on a pack (its tasks as anchors + fresh validated items) or on plain tasks (--harbor / --tasks):
    controls first, every step metric, an HTML report. A pack whose ``requires`` this machine lacks is refused unless
    --allow vouches for it."""
    if sum(x is not None and x != () for x in (pack_ref, harbor or None, tasks_ref)) != 1:
        raise click.UsageError("give exactly one of PACK_REF, --harbor or --tasks")
    notes: list[str] = []
    res = asyncio.run(_bench_pack(pack_ref, out, n_fresh, k, key_file, policies, _ks(ks), controls=controls, anchors_path=anchors_path,
                                  harbor=tuple(harbor), tasks_ref=tasks_ref, allow=tuple(x for x in allow.split(",") if x), notes=notes))
    _emit({"summary": res["summary"], "paths": res["paths"], "notes": notes, "next": [f"open {res['paths']['report']}"]}, as_json,
          [f"note: {n}" for n in notes] + _model_lines(res["summary"]) + [f"report {res['paths']['report']}"])


@bench.command("report")
@click.argument("run_dir", type=click.Path(path_type=Path, exists=True))
@click.option("--out", type=click.Path(path_type=Path), default=None, help="default: <run>/bench-report (the run's own report is kept)")
@click.option("--force", is_flag=True, help="overwrite the run's own report in <run>/bench")
@click.option("--ks", default=None, help="pass@k columns, comma-separated (default: the saved report's, else from the run's k)")
@click.option("--threshold", type=float, default=None)
@click.option("--json", "as_json", is_flag=True)
def bench_report(run_dir, out, force, ks, threshold, as_json) -> None:
    """(Re)compute every metric and the HTML report for an existing M10 run directory, from what its report saved
    (title, stage maps, validity, extras), into a new directory unless --force (campaign B10)."""
    from crucible.bench.run import report_saved

    res = report_saved(Path(run_dir), out_dir=out, ks=_ks(ks), threshold=threshold, force=force)
    _emit({"summary": res["summary"], "paths": res["paths"]}, as_json, _model_lines(res["summary"]) + [res["paths"]["report"]])


@bench.command("compare")
@click.argument("base_dir", type=click.Path(path_type=Path, exists=True))
@click.argument("cand_dir", type=click.Path(path_type=Path, exists=True))
@click.option("--base-policy", default=None)
@click.option("--cand-policy", default=None)
@click.option("--base-harness", default=None)
@click.option("--cand-harness", default=None)
@click.option("--threshold", type=float, default=1.0, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def bench_compare(base_dir, cand_dir, base_policy, cand_policy, base_harness, cand_harness, threshold, as_json) -> None:
    """Task-level comparison of one model per side (two runs, or two policies in one run): task flips, exact McNemar
    over tasks, a t interval and a sign-flip test on per-task deltas. A side holding several models is refused, and
    tasks a validity manifest marks invalid are left out."""
    from crucible.bench.compare import compare, to_markdown
    from crucible.bench.metrics import load_run
    from crucible.bench.run import load_validity

    sides = []
    for side, d, pol, har in (("base", base_dir, base_policy, base_harness), ("candidate", cand_dir, cand_policy, cand_harness)):
        if not (Path(d) / "results.jsonl").exists() and not (Path(d) / "run.json").exists():
            raise ValueError(f"{d} holds no run (no run.json or results.jsonl)")
        eps = load_run(d)
        got = [e for e in eps if pol in (None, e.policy) and har in (None, e.harness)]
        if not got:  # campaign B14: a misspelled policy gave pairs 0, p = 1.0, exit 0
            raise ValueError(f"no {side} episodes for policy {pol!r} harness {har!r} in {d}; it holds {sorted({f'{e.policy}@{e.harness}' for e in eps})}")
        sides.append(got)
    invalid = {t for d in (base_dir, cand_dir) for t, v in (load_validity(Path(d)) or {}).items() if v.get("valid") is False}
    r = compare(*sides, threshold=threshold, exclude_tasks=invalid)
    click.echo(json.dumps(r, indent=1, sort_keys=True) if as_json else to_markdown(r["base"] or str(base_dir), r["candidate"] or str(cand_dir), r))


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
