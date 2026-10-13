"""``crucible build``: workflows → RL tasks and a benchmark (ARCHITECTURE §4 M15: M3 → M8 → M9 → M11 → M13).

Steps, each recorded in ``summary.json``:
1. **Ingest** the pack's workflow sources into ``out/store`` (M3; idempotent).
2. **Forge** tasks from them with the pack's ``forge`` hook (M8; the toy pack uses scripted roles and makes no paid
   calls). Tasks are written as Harbor-style directories with the Crucible overlay.
3. **Accept** each task with the full M9 contract (``accept_task``). With the needs this run has (``needs``: Docker, paid
   calls, ...), a task is ``accepted``, ``provisional`` (nothing failed; only checks that need what this run lacks are
   pending) or ``rejected`` (M9 audit V2: build used to run the model-free subset and call it accepted).
4. **Export** (M11): each accepted or provisional task's oracle (provisional ones labelled so) is replayed into canonical
   ``reward.json`` + ``steps.jsonl``, the outputs every runtime adapter (OpenEnv, NeMo-Gym, verifiers, Harbor through
   ATIF replay) must reproduce. A World task is not a Harbor-native directory (it has no Dockerfile or ``tests/``);
   Harbor reaches it through ATIF replay.
5. **Splits and contamination** (M13 V13 / L12): the anchors and ``n_fresh`` generated items are split (lineage-closed,
   sealed) into ``train`` and ``bench``; the bench items are scanned against the training corpus (the train split and the
   ingested workflow traces ``crucible label`` trains on); a contaminated bench item is invalid.
6. **Bench** (M13): only the ``bench`` split runs, with the pack's scripted policies; the seal rides in the run's report
   and is verified after the run.

M15 audit (H2 V1 V4-V9, G2 D6-D8 D12): the identity of a re-run is compared before anything is written (``use a new
--out``); every output (``accept.json``, ``splits.json``, ``export/``, ``anchors.json``, ``contamination.json``,
``summary.json``) is written to a staging dir and swapped in only when the build completes, so a failed rebuild leaves
the previous outputs whole; the bench preflight (``requires``, ``higher_is_better``) is the one ``bench run`` uses; an
anchor whose oracle did not replay to the top of ``reward.range`` is invalid; a build with no trace, no forged task or no
accepted task raises ``BuildFailed`` (stage named), and one with rejected tasks, export errors or owed samples has
``summary["ok"] is False`` (the CLI exits 1) unless the matching allow flag is given.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

OUTPUTS = ("accept.json", "splits.json", "export", "anchors.json", "contamination.json", "summary.json")


class BuildFailed(RuntimeError):
    """A build stage that leaves nothing to benchmark (no traces, no forged task, no accepted task, ...)."""

    def __init__(self, stage: str, reason: str, summary: dict[str, Any] | None = None):
        super().__init__(f"build failed at {stage}: {reason}")
        self.stage, self.reason, self.summary = stage, reason, dict(summary or {})


def bench_preflight(m: Any, allow: tuple[str, ...] = ()) -> None:
    """The refusals ``bench run`` applies (M14 campaign B6, audit V12), shared with ``build`` (H2 V5)."""
    from crucible.studio.pack import missing_requirements

    if miss := missing_requirements(m, allow):
        raise ValueError(f"pack {m.name} requires {', '.join(miss)}, which this machine does not provide: pass --allow {','.join(miss)} "
                         "where it does (its tasks may not run, or may not mean what they claim, without it)")
    if not m.reward.higher_is_better:
        raise ValueError(f"pack {m.name} declares reward.higher_is_better: false; bench metrics (pass@k, IRT, controls) assume higher is "
                         "better, so a lower-is-better pack is refused rather than reported upside down")


def oracle_reasons(reward: Any, best: float) -> list[str]:
    """Why an anchor's replayed oracle does not certify it (H2 V1/V6): not replayed, non-finite, or short of the top."""
    if reward is None:
        return ["oracle_not_replayed"]
    if not isinstance(reward, (int, float)) or isinstance(reward, bool) or not math.isfinite(reward):
        return ["broken_oracle"]
    return [] if abs(reward - best) <= 1e-9 else ["broken_oracle"]


def _text_of(task: Any) -> str:
    return str(getattr(task, "instruction", "") or "")


def contamination_step(bench: dict[str, str], train: dict[str, str], traces: list[Any], bench_name: str,
                       generated: set[str] | frozenset[str] = frozenset()) -> dict[str, Any]:
    """The M13 contamination scan in build (L12): the bench items' instructions against the training corpus (the train-split
    items and the ingested workflow traces), with the canary registry and template subtraction.

    The template is learnt over the whole benchmark (bench and train items: M13's "the bench's own tasks"), as the scan
    is meant to: the toy's five anchors share a family preamble and its generated items one generator template, so a
    bench-only template (1-2 items) flagged every bench item as contaminated by the train items of its own family. A bench
    text that equals a corpus text (normalized) is a hit whatever the template (``exact``): two identical items would
    otherwise teach the template that the whole text is boilerplate. Two ``generated`` items (fresh items: distinct seeds,
    validated by the gold-and-no-op chain) may share an instruction word for word, since the generator varies the World,
    not the text; the exact check skips those pairs."""
    from crucible.bench import contamination as c

    corpus = [(f"train:{k}", v) for k, v in sorted(train.items()) if v]
    for t in traces:
        try:
            corpus.append((f"trace:{getattr(t, 'id', '?')}", json.dumps(t.model_dump(mode="json"), sort_keys=True, default=str)))
        except Exception:  # noqa: BLE001 - a trace that does not serialize is scanned as its repr
            corpus.append((f"trace:{getattr(t, 'id', '?')}", repr(t)))
    texts = {k: v for k, v in bench.items() if v}
    if not corpus:
        return {"corpus": None, "note": "no training corpus (no train-split item and no ingested trace): nothing to scan against",
                "scanned": len(texts), "hits": {}, "contaminated": []}
    reg = c.CanaryRegistry.for_bench(bench_name, ["v1"])
    every = sorted({**{k: v for k, v in train.items() if v}, **texts}.items())
    fps = {f.id: f for f in c.fingerprints([(k, v, "") for k, v in every], registry=reg)} if texts else {}
    hits: dict[str, list[dict[str, Any]]] = {}
    want = {"canary", "13-gram", "code-substring", "near-duplicate", "containment"}
    for h in c.scan([fps[k] for k in texts if k in fps], corpus, registry=reg):
        if h.get("kind") in want and h.get("task") in texts:
            hits.setdefault(h["task"], []).append({x: h[x] for x in ("kind", "source", "overlap", "score") if x in h})
    norm = {src: " ".join(c._norm(txt).split()) for src, txt in corpus}
    for k, v in texts.items():
        nv = " ".join(c._norm(v).split())
        hits.setdefault(k, []).extend({"kind": "exact", "source": src} for src, t in norm.items()
                                      if t == nv and not (k in generated and src.removeprefix("train:") in generated))
    hits = {k: v for k, v in hits.items() if v}
    return {"corpus": {"train_items": len(train), "traces": len(traces)}, "scanned": len(texts),
            "hits": {k: v[:5] for k, v in sorted(hits.items())}, "contaminated": sorted(hits)}


def assign_splits(items: list[Any], key: bytes, bench_frac: float) -> dict[str, Any]:
    """The build's sealed split. A fractional draw over a handful of lineages can leave the bench empty (0.7^5 = 17% for the
    toy's five anchors at 0.3); then the lineage ranked lowest under the key is sealed to bench, so the choice stays secret
    and is reproduced by the same key (the seal of a re-run matches)."""
    from crucible.bench import splits

    salt = key + b"|splits"
    sp = splits.assign(items, salt=salt, bench_frac=bench_frac)
    if len({r for it in items for r in it.roots}) > 1 and (not sp["counts"].get("bench") or not sp["counts"].get("train")):
        # either side empty (0.7^5 = 17% no bench, 0.3^5 = 0.24% no train for the toy's five anchors): the lineage ranked
        # lowest under the key goes to bench and the one ranked highest to train, both secret and reproduced by the key
        ranked = sorted(items, key=lambda it: splits._u(salt, min(it.roots)))
        sealed = {ranked[0].id: "bench"}
        if set(ranked[-1].roots).isdisjoint(ranked[0].roots):
            sealed[ranked[-1].id] = "train"
        sp = splits.assign(items, salt=salt, bench_frac=bench_frac, sealed=sealed)
    return sp


def _swap(stage: Path, out: Path) -> None:
    """Move every staged output into ``out``; an existing one is moved aside first and removed after (H2 V7)."""
    trash = Path(tempfile.mkdtemp(prefix=".build-old-", dir=out))
    try:
        for name in OUTPUTS[:-1]:
            new, cur = stage / name, out / name
            if cur.exists() or cur.is_symlink():
                os.replace(cur, trash / name)
            if new.exists():
                os.replace(new, cur)
        os.replace(stage / "summary.json", out / "summary.json")  # last: a reader never sees a summary ahead of its outputs
    finally:
        shutil.rmtree(trash, ignore_errors=True)


async def build(pack: str | Path, *, out: Path, n_fresh: int = 2, k: int = 1, key: bytes | None = None, dry_run: bool = False,
                needs: tuple = (), allow: tuple[str, ...] = (), allow_rejected: bool = False, allow_empty: bool = False,
                bench_frac: float = 0.3, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """``key``: the HMAC item key; None reads ``$CRUCIBLE_BENCH_KEY`` / ``$CRUCIBLE_BENCH_KEY_FILE`` or keeps a new random
    key in ``out/bench.key`` (the public default ``crucible-build-v1`` let anyone regenerate the held-out items: M13 G2-4).
    Fresh items pass the gold-and-no-op floor before they count. ``allow``: requirements this machine satisfies (as
    ``bench run --allow``); ``allow_rejected``: rejected tasks and export errors do not fail the build; ``allow_empty``: a
    build with no trace, forged or accepted task goes on to a fresh-items-only bench."""
    from crucible.bench import dynamic
    from crucible.bench.run import bench_key
    from crucible.studio import runs
    from crucible.studio.pack import find_pack, load_manifest

    if k < 1 or n_fresh < 0:
        raise ValueError(f"build needs k >= 1 and fresh >= 0 (got k={k}, fresh={n_fresh})")
    pack_dir = find_pack(pack)
    m = load_manifest(pack_dir)
    missing = [h for h in ("forge", "policies") if getattr(m.hooks, h) is None]
    if missing:
        raise ValueError(f"pack {m.name} has no {', '.join('hooks.' + h for h in missing)}: build needs them"
                         + (f" (benchmark its tasks with: crucible bench run {pack_dir})" if m.hooks.tasks or m.hooks.generator else ""))
    bench_preflight(m, allow)  # H2 V5: before anything runs or is written
    if dry_run:
        return {"command": "build", "dry_run": True, "pack": m.name, "ingest": [s.glob for s in m.ingest], "hooks": m.hooks.model_dump(),
                "needs": sorted(str(n) for n in needs), "k": k, "n_fresh": n_fresh}
    out = Path(out)
    notes: list[str] = []
    key = key if key is not None else bench_key(out, notes=notes)
    commitment = dynamic.commitment(key, dynamic.commit_salt(f"{m.name}-build", "v1"))
    identity = {"pack": m.name, "n_fresh": n_fresh, "k": k, "needs": sorted(str(n) for n in needs), "key_commitment": commitment, "bench_frac": bench_frac}
    runs.record(out, "build", {"pack_dir": str(pack_dir), "allow": list(allow), "allow_rejected": allow_rejected, "allow_empty": allow_empty, **identity},
                identity=identity, pack_dir=pack_dir, manifest=m)  # G2 D6: compared before any step writes
    stage = Path(tempfile.mkdtemp(prefix=".build-stage-", dir=out))
    try:
        s = await _build(pack_dir, m, out, stage, notes=notes, key=key, n_fresh=n_fresh, k=k, needs=needs, allow_rejected=allow_rejected,
                            allow_empty=allow_empty, bench_frac=bench_frac, say=progress or (lambda s: None))
        return s
    finally:
        shutil.rmtree(stage, ignore_errors=True)


async def _build(pack_dir: Path, m: Any, out: Path, stage: Path, *, notes: list[str], key: bytes, n_fresh: int, k: int, needs: tuple, allow_rejected: bool,
                 allow_empty: bool, bench_frac: float, say: Callable[[str], None]) -> dict[str, Any]:
    from crucible.accept.run import accept_task, verdict
    from crucible.interop.replay import Action, replay
    from crucible.studio import runs
    from crucible.studio.export import task_dirs
    from crucible.studio.label import ingest
    from crucible.studio.pack import resolve

    summary: dict[str, Any] = {"command": "build", "pack": m.name, "provenance": runs.provenance(pack_dir, m), "notes": notes}
    failures: list[dict[str, str]] = []

    say("ingest: reading the pack's workflow sources")
    traces = ingest(pack_dir, m, out / "store")
    summary["ingest"] = {"traces": len(traces)}
    if not traces and not allow_empty:
        raise BuildFailed("ingest", f"the ingest sources of {m.name} gave no trace (pass --allow-empty to bench fresh items only)", summary)
    say(f"forge: from {len(traces)} traces")
    forged = await resolve(pack_dir, m.hooks.forge)(out / "forge", out / "store")
    summary["forge"] = {"tasks": len(forged), "dir": str((out / "forge" / "tasks").resolve())}
    if not forged and not allow_empty:
        raise BuildFailed("forge", f"the forge hook of {m.name} made no task (pass --allow-empty to bench fresh items only)", summary)
    reports = {}
    accepted = []
    for n, (t, res) in enumerate(forged, 1):
        say(f"accept: {n}/{len(forged)} {t.task_id}")
        rep = await accept_task(t, needs=list(needs), cache_root=out / "accept", resolve=res)
        v = verdict(rep)
        reports[t.task_id] = {"state": v["state"], "blocking": v["blocking"], "checks": {c: str(r.status) for c, r in rep.results.items()}}
        if v["state"] in ("accepted", "provisional"):
            accepted.append(t)
    (stage / "accept.json").write_text(json.dumps(reports, indent=1, sort_keys=True) + "\n")
    states = [r["state"] for r in reports.values()]
    summary["accept"] = {"accepted": states.count("accepted"), "provisional": states.count("provisional"), "rejected": states.count("rejected"),
                         "total": len(forged), "needs": sorted(str(n) for n in needs),
                         "blocking": {k: v["blocking"] for k, v in reports.items() if v["blocking"]}}
    if forged and not accepted and not allow_empty:
        raise BuildFailed("accept", f"M9 rejected all {len(forged)} forged tasks: {dict(list(summary['accept']['blocking'].items())[:3])}", summary)
    if summary["accept"]["rejected"]:
        failures.append({"stage": "accept", "reason": f"{summary['accept']['rejected']} of {len(forged)} tasks rejected by M9"})

    exported, export_errors = {}, {}
    dirs = task_dirs([t.task_id for t in accepted], stage / "export")  # H2 V9: safe, distinct, inside
    for t in accepted:  # M11: the canonical outputs every runtime adapter (OpenEnv, NeMo-Gym, verifiers, Harbor via ATIF) must reproduce
        say(f"export: {t.task_id}")
        d = dirs[t.task_id]
        try:
            acts = [Action(a["function_name"], dict(a.get("arguments") or {})) if a.get("kind", "tool_call") == "tool_call"
                    else Action(None, {}, to=a.get("to"), text=a.get("text")) for a in t.oracle]
            r = await replay(t.runtime_task(), acts, d, episode_key=f"{t.task_id}|oracle", plan=t.plan)
            exported[t.task_id] = {"reward": r["reward"], "steps": r["steps"], "dir": str((out / "export" / d.name).resolve()),
                                   "acceptance": reports[t.task_id]["state"]}
        except Exception as e:  # noqa: BLE001 - reported per task, and the task is invalid in the benchmark (V1)
            export_errors[t.task_id] = f"{type(e).__name__}: {e}"[:300]
    summary["export"] = {"canonical": len(exported), "oracle_rewards": {k: v["reward"] for k, v in sorted(exported.items())}, "errors": export_errors,
                         "dir": str((out / "export").resolve())}
    if export_errors:
        failures.append({"stage": "export", "reason": f"{len(export_errors)} oracle replay(s) failed: {dict(list(export_errors.items())[:3])}"})
    anchors = []
    for t in accepted:
        rt = t.runtime_task()
        rt.oracle = t.oracle  # the scripted policies replay it
        anchors.append((t.task_id, "forged", rt))
    made: list[Any] = []
    gens: dict[str, Any] = {}
    if m.hooks.generator and n_fresh:
        gen = resolve(pack_dir, m.hooks.generator)

        def generator(seed: int, **kw: Any) -> Any:  # forwards item_id= (dynamic V3: the published id never spells the seed)
            made.append(gen(seed, **kw))
            return made[-1]

        gens = {"generated": generator}
    try:
        await _bench(pack_dir, m, out, stage, summary, failures, reports, exported, anchors, gens, key=key, n_fresh=n_fresh, k=k, traces=traces,
                     bench_frac=bench_frac, say=say)
    finally:
        for t in made:
            if callable(getattr(t, "cleanup", None)):
                t.cleanup()
    summary["failures"] = failures
    summary["ok"] = not [f for f in failures if not (allow_rejected and f["stage"] in ("accept", "export"))]
    summary["next"] = [f"open {summary['bench']['report']}", f"crucible label --pack {pack_dir}"]
    (stage / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    _swap(stage, out)
    return summary


async def _bench(pack_dir: Path, m: Any, out: Path, stage: Path, summary: dict[str, Any], failures: list[dict[str, str]], reports: dict[str, Any],
                 exported: dict[str, Any], anchors: list[Any], gens: dict[str, Any], *, key: bytes, n_fresh: int, k: int, traces: list[Any],
                 bench_frac: float, say: Callable[[str], None]) -> None:
    from crucible.actors.policy import EpisodeLimits
    from crucible.bench import dynamic, irt, splits
    from crucible.bench.run import gold_and_noop_check, run_bench
    from crucible.runtime.harness.native import NativeHarness
    from crucible.studio.pack import policy_refs, resolve

    lo, hi = m.reward.range
    say(f"bench: {len(anchors)} anchors, {n_fresh if gens else 0} fresh items to validate")
    try:
        bm = await dynamic.build(f"{m.name}-build", "v1", key=key, anchors=anchors, generators=gens, n_fresh=n_fresh if gens else 0,
                                 checks=[gold_and_noop_check(out / "validate", best=hi, threshold=m.label.threshold)] if gens else (), registry=out / "bench.keys.jsonl")
    except dynamic.ShortBenchmark as e:
        man = e.benchmark.manifest()
        raise BuildFailed("bench", f"only {man['counts']['fresh']} of {n_fresh} fresh items passed validation; yields {man['yields']}; "
                                   f"rejected {man['rejected'][:5]}", summary) from None
    # validity (M13 campaign B4/B5): M9 acceptance for the forged anchors and their replayed oracle reward at the top of the
    # pack's range (H2 V1/V6), the gold-and-no-op chain for the fresh items; nothing enters the benchmark without evidence
    validity: dict[str, dict[str, Any]] = {}
    for i in bm.items:
        if i.kind == "anchor":
            orc = exported.get(i.id, {}).get("reward")
            bad = oracle_reasons(orc, hi)
            validity[i.id] = {"valid": not bad, "reasons": bad, "source": f"M9 accept ({reports[i.id]['state']})", "oracle": {"replayed_reward": orc}}
        else:
            validity[i.id] = {"valid": True, "reasons": [], "source": "dynamic validation chain (gold_and_noop)", "checks": i.checks}
    # splits (M13 V13): sealed before any run; only the bench split is benchmarked (H2 V8)
    items = [splits.SplitItem(i.id, lineage=(str(getattr(i.task, "seed_ref", None) or i.id),), axes={"family": i.family}, source_kind="synthetic") for i in bm.items]
    sp = assign_splits(items, key, bench_frac)
    prev = out / "splits.json"
    if prev.is_file():
        old = json.loads(prev.read_text()).get("seal")
        if old != sp["seal"]:  # H2 V7: a sealed split is never overwritten by another
            raise BuildFailed("splits", f"{prev} is sealed {str(old)[:12]} and this build seals {sp['seal'][:12]} (the item set changed): use a new --out", summary)
    splits.write_splits(stage / "splits.json", sp)
    bench_items = [i for i in bm.items if sp["splits"].get(i.id) == "bench"]
    train_items = [i for i in bm.items if sp["splits"].get(i.id) == "train"]
    say(f"splits: {len(train_items)} train, {len(bench_items)} bench (seal {sp['seal'][:12]})")
    con = contamination_step({i.id: _text_of(i.task) for i in bench_items}, {i.id: _text_of(i.task) for i in train_items}, traces, f"{m.name}-build",
                             generated={i.id for i in bm.items if i.kind != "anchor"})
    (stage / "contamination.json").write_text(json.dumps(con, indent=1, sort_keys=True, default=str) + "\n")
    for tid in con["contaminated"]:
        validity[tid] = {**validity[tid], "valid": False, "reasons": [*validity[tid]["reasons"], "contaminated"]}
    summary["contamination"] = {x: con[x] for x in ("corpus", "scanned", "contaminated", "note") if x in con}
    if not bench_items:
        raise BuildFailed("splits", f"the bench split is empty ({len(bm.items)} items, bench_frac {bench_frac}): nothing to benchmark", summary)
    split_doc = {"seal": sp["seal"], "counts": sp["counts"], "benched": "bench", "items": len(bench_items)}
    say(f"bench: running {len(bench_items)} bench-split items, k={k}")
    res = await run_bench(f"{m.name}-build", root=out / "runs", tasks=[i.task for i in bench_items], policies=policy_refs(resolve(pack_dir, m.hooks.policies)()),
                          threshold=m.label.threshold, harnesses=[NativeHarness(limits=EpisodeLimits(max_steps=1000))], k=k,
                          stages_of=resolve(pack_dir, m.hooks.stages) if m.hooks.stages else None,
                          extras={"manifest": bm.manifest(), "reward_range": [lo, hi], "splits": split_doc, "provenance": summary["provenance"]}, validity={i.id: validity[i.id] for i in bench_items})
    saved = json.loads(Path(res["paths"]["summary"]).read_text())
    seal_ok = (saved.get("splits") or {}).get("seal") == sp["seal"] and splits.verify_seal(sp["splits"], key + b"|splits", sp["seal"], sp["params"])
    if not seal_ok:
        raise BuildFailed("bench", "the run's report does not carry this build's split seal", summary)
    fit = res.get("irt_fit")
    aids = [i.id for i in bench_items if i.kind == "anchor"]
    anchors_file = None
    if fit is not None and aids and all(i in fit.items for i in aids):
        irt.save_anchors(stage / "anchors.json", {i: fit.items[i] for i in aids}, meta={"bench": f"{m.name}-build", "version": "v1"})
        anchors_file = str((out / "anchors.json").resolve())
    rs = res["summary"]
    summary["bench"] = {"items": dict(bm.manifest()["counts"]),
                        "benched": {"anchor": sum(i.kind == "anchor" for i in bench_items), "fresh": sum(i.kind != "anchor" for i in bench_items)},
                        "report": str(Path(res["paths"]["report"]).resolve()), "digest": saved["digest"],
                        "models": {mm["policy"]: mm["overall"]["deterministic"].get("mean_reward") for mm in rs["models"]},  # mean of task means
                        "tasks": {mm["policy"]: mm["overall"]["deterministic"].get("mean_reward_tasks") for mm in rs["models"]},
                        "run": {x: rs["run"].get(x) for x in ("planned", "measured", "masked", "owed", "stopped")}, "warnings": rs.get("warnings") or [],
                        "validity": rs.get("validity"), "invalid_tasks": sorted(rs.get("invalid_tasks") or {}), "anchors": anchors_file,
                        "splits": {"seal": sp["seal"], "counts": sp["counts"], "file": str((out / "splits.json").resolve()), "verified": seal_ok},
                        "yields": bm.manifest().get("yields"), "commitment": bm.manifest().get("commitment")}
    if (rs["run"].get("owed") or 0) > 0:
        failures.append({"stage": "bench", "reason": f"{rs['run'].get('owed')} planned samples have no result"})
    if not any(validity[i.id]["valid"] for i in bench_items):  # every bench item invalid: no metric means anything
        failures.append({"stage": "bench", "reason": f"none of the {len(bench_items)} bench items is valid: "
                                                     + "; ".join(f"{i.id} ({'/'.join(validity[i.id]['reasons'])})" for i in bench_items[:5])})
