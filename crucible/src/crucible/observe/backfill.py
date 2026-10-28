"""Backfill: existing local results into W&B runs and Weave traces, offline (swarm-pipeline/11 §7).

``discover(paths)`` classifies what it finds:

| source | recognized by | becomes |
|---|---|---|
| a Tinker training run | ``config.json`` + ``steps.jsonl`` | a ``crucible-train`` run: per-step metrics from ``steps.jsonl`` and ``spend.jsonl``, drops, the trained episodes as traces, artifacts (``train_views.finish_train``) |
| a benchmark arm | ``metrics.json`` + ``episodes.jsonl`` | a ``crucible-bench`` run (``bench.log_benchmark``) with its episodes as traces |
| a comparison | ``compare.json`` | ``bench.log_compare`` |
| a funnel log | ``funnel.jsonl`` (``generators.base.Funnel`` rows) | a ``crucible-factory`` funnel view |
| a run report (``*.json``) | its keys | ``crucible-factory`` (or ``crucible-train`` for the Tinker smoke) views: funnels (``heldout_funnel``, ``summary.funnel``, swarm funnel), verifier cards (``families``), targeting (``target`` / ``windows`` with bands), the trace audit (``facts`` + ``first_wrong_step``), spend (``spend_by_role``), the benchmark manifest, and the file itself as an artifact |

A source already backfilled (the ledger ``<root>/backfill.json``, keyed by path and content hash) is skipped unless
``force``. Everything is offline: ``scripts/observe.py sync`` uploads it later.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.observe import factory_views as fv
from crucible.observe.bench import log_benchmark, log_compare
from crucible.observe.naming import factory_name, project, run_name, slug, tags
from crucible.observe.run import ObserveConfig, ObsRun, scalars
from crucible.observe.train_views import finish_train, train_run


def _read(p: Path) -> Any:
    return json.loads(p.read_text())


def _jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _hash(p: Path) -> str:
    h = hashlib.sha256()
    files = [p] if p.is_file() else sorted(x for x in p.iterdir() if x.is_file())
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def kind_of(p: Path) -> str | None:
    if p.is_dir():
        if (p / "config.json").exists() and (p / "steps.jsonl").exists():
            return "train"
        if (p / "compare.json").exists():
            return "compare"
        if (p / "metrics.json").exists() and (p / "episodes.jsonl").exists():
            return "bench"
        return None
    if p.name == "funnel.jsonl":
        return "funnel"
    if p.suffix == ".json" and p.name not in RUN_FILES:
        return "report"
    return None


# the files of a run directory (a training run, a benchmark arm, the runtime's own): parts of a source, never a report
RUN_FILES = {"config.json", "state.json", "summary.json", "receipt.json", "versions.json", "metrics.json", "compare.json", "run.json", "lock.json",
             "owner.json", "record.json", "rollout.json", "trace.json", "calls.json", "trajectory.agent.json", "teardown.json"}


def _is_smoke(p: Path) -> bool:
    """A Tinker smoke or probe report (``data/tinker/...``, ``runs/tinker_smoke.json``): logged to ``crucible-train``."""
    return "tinker" in p.parts or p.stem.startswith(("tinker", "probe", "smoke", "openai_check")) or "tinker" in p.stem


def discover(paths: Iterable[Path | str], *, depth: int = 3) -> list[tuple[str, Path]]:
    """``(kind, path)`` for every source under ``paths`` (a recognized directory is not searched further; rollout
    directories are never sources)."""
    out: list[tuple[str, Path]] = []

    def walk(p: Path, d: int) -> None:
        if p.name in ("rollouts", "adapter", "work", "wandb", "observe", "__pycache__") or p.name.startswith("."):
            return
        k = kind_of(p)
        if k is not None:
            out.append((k, p))
            if k != "report" and k != "funnel":
                return
        if p.is_dir() and d > 0:
            for c in sorted(p.iterdir()):
                walk(c, d - 1)

    for p in paths:
        walk(Path(p), depth)
    seen: set[Path] = set()
    uniq = []
    for k, p in out:  # a benchmark arm inside a comparison is logged by the comparison's own arms (and once more as an arm: fine)
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append((k, p))
    return uniq


# ---- adapters --------------------------------------------------------------------------------------------------------------
def backfill_train(d: Path, cfg: ObserveConfig) -> dict[str, Any]:
    doc = _read(d / "config.json")
    conf = doc.get("config") or {}
    summary = _read(d / "summary.json") if (d / "summary.json").exists() else {}
    tasks = sorted((doc.get("objective") or {}).keys())
    run = train_run(conf, run_dir=d, objective_id=str(doc.get("objective_id")), tasks=tasks, cfg=cfg, extra={"backfill": str(d), "model_row": doc.get("model")})
    spend = _jsonl(d / "spend.jsonl")
    drops_rows = _jsonl(d / "drops.jsonl")
    prev_usd = 0.0
    for r in _jsonl(d / "steps.jsonl"):
        step = int(r["step"])
        rs = [x for g in r.get("rewards") or [] for x in g]
        m: dict[str, Any] = {"reward/mean": statistics.fmean(rs) if rs else None, "reward/std": statistics.pstdev(rs) if len(rs) > 1 else 0.0,
                             "reward/min": min(rs) if rs else None, "reward/max": max(rs) if rs else None,
                             "pass_rate": sum(x >= 1.0 - 1e-9 for x in rs) / len(rs) if rs else None, "train/loss": r.get("loss"),
                             "kl/sample_vs_train": r.get("kl_sample_vs_train"), "tokens/train": r.get("train_tokens"), "tokens/trained": r.get("trained_tokens"),
                             "usd/cum": r.get("usd_cum"), "usd/step": (r.get("usd_cum") or 0.0) - prev_usd, "usd/cap": summary.get("spend", {}).get("cap_usd"),
                             "sampler/version": step, "episodes": r.get("episodes"), "datums": r.get("datums"), "time/train_s": r.get("train_s"),
                             **{f"train/{k}": v for k, v in (r.get("fb_metrics") or {}).items()}, **{f"drops/{k}": v for k, v in (r.get("drops") or {}).items()}}
        vs = [v for v in r.get("min_versions") or [] if v is not None]
        if vs:
            m["staleness/max"] = float(step - 1 - min(vs))  # the step's weights were version step-1 when it sampled
        smp = [x for x in spend if x.get("kind") == "sample" and x.get("version") == step - 1]
        if smp:
            m["tokens/sampled"] = sum(int(x.get("sample") or 0) for x in smp)
            m["tokens/prefill_uncached"] = sum(int(x.get("prefill") or 0) for x in smp)
            m["tokens/prefill_cached"] = sum(int(x.get("prefill_cached") or 0) for x in smp)
        prev_usd = r.get("usd_cum") or prev_usd
        run.log(scalars(m), step)
    for x in drops_rows:
        run.record("drop", x, int(x.get("step") or 0))
    if not summary:
        summary = {"model": conf.get("model"), "objective_id": doc.get("objective_id"), "stop": "unknown"}
    out = finish_train(run, summary, d, status="ok" if summary.get("stop") in ("done", "unknown") else str(summary.get("stop")))
    return {"kind": "train", **out}


def backfill_bench(d: Path, cfg: ObserveConfig) -> dict[str, Any]:
    rep = _read(d / "metrics.json")
    return {"kind": "bench", **log_benchmark(rep, d, cfg=cfg, extra_config={"backfill": str(d)})}


def backfill_compare(d: Path, cfg: ObserveConfig) -> dict[str, Any]:
    cmp = _read(d / "compare.json")
    for arm in ("control", "trained"):
        if (d / arm / "metrics.json").exists():
            log_benchmark(_read(d / arm / "metrics.json"), d / arm, cfg=cfg, extra_config={"backfill": str(d / arm)})
    return {"kind": "compare", **log_compare(cmp, d, cfg=cfg)}


def funnel_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """``Funnel.summary`` from its rows (the last row of a (candidate, stage) wins), stages in first-seen order."""
    last: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for r in rows:
        last[(str(r.get("generator")), str(r.get("candidate")), str(r.get("stage")))] = r
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (g, _c, st), r in last.items():
        s = out.setdefault(g, {}).setdefault(st, {"in": 0, "pass": 0, "reasons": {}})
        s["in"] += 1
        if r.get("ok"):
            s["pass"] += 1
        else:
            k = str(r.get("reason") or "fail").split(":")[0].split("(")[0].strip()[:70] or "fail"
            s["reasons"][k] = s["reasons"].get(k, 0) + 1
    return out


def backfill_funnel(p: Path, cfg: ObserveConfig) -> dict[str, Any]:
    lanes = funnel_summary(_jsonl(p))
    with ObsRun(project("factory"), factory_name("funnel", p.parent.name, _hash(p)), group="funnel", job_type="funnel", cfg=cfg,
                config={"source": str(p), "backfill": True}, tags=tags(phase="funnel", lane=list(lanes))) as run:
        fv.funnel(run, lanes)
    return {"kind": "funnel", "run": str(run.dir), "lanes": list(lanes)}


def _report_views(run: ObsRun, doc: Mapping[str, Any], name: str) -> list[str]:
    """Every view whose keys the report has; returns the views logged."""
    done: list[str] = []
    lanes: dict[str, Mapping[str, Any]] = {}
    if isinstance(doc.get("heldout_funnel"), Mapping):
        lanes["window-heldout"] = doc["heldout_funnel"]
    fun = (doc.get("summary") or {}).get("funnel") if isinstance(doc.get("summary"), Mapping) else None
    if isinstance(fun, Mapping):
        for split, v in fun.items():
            if isinstance(v, Mapping) and isinstance(v.get("total"), Mapping):
                lanes[f"window-{split}"] = v["total"]
    sw = doc.get("swarm") if isinstance(doc.get("swarm"), Mapping) else None
    if sw and isinstance(sw.get("funnel"), Mapping):
        lanes["swarm"] = sw["funnel"]
    if lanes:
        fv.funnel(run, lanes)
        done.append("funnel")
    fams = doc.get("families")
    if isinstance(fams, Mapping) and fams and all(isinstance(v, Mapping) and ("card" in v or "fp" in v) for v in fams.values()):
        fv.verifier_cards(run, fams)
        done.append("verifier_cards")
    for key in ("target", "windows"):
        t = doc.get(key)
        if isinstance(t, Mapping) and t and any(isinstance(v, Mapping) and "band" in v for v in t.values()):
            fv.targeting(run, {k: v for k, v in t.items() if isinstance(v, Mapping) and "band" in v}, model=(doc.get("pilot") or {}).get("model")
                         if isinstance(doc.get("pilot"), Mapping) else None)
            done.append("targeting")
            break
    rep = doc if ("facts" in doc and "first_wrong_step" in doc) else (doc.get("real") if isinstance(doc.get("real"), Mapping) and "first_wrong_step" in doc["real"]
                                                                      else None)
    if rep is None and isinstance(doc.get("audit"), Mapping) and isinstance(doc["audit"].get("report"), Mapping):
        rep = doc["audit"]["report"]
    if rep is not None:
        fv.audit_summary(run, rep)
        props = (rep.get("proposals") or {}).get("rows") if isinstance(rep.get("proposals"), Mapping) else None
        if isinstance(props, list):
            run.table("audit/proposals", ["id", "kind", "task", "status", "traces", "why"],
                      [[p.get("id"), p.get("kind"), p.get("task"), p.get("status"), p.get("traces"), p.get("why")] for p in props if isinstance(p, Mapping)])
        done.append("audit")
    sp = doc.get("spend_by_role")
    if sp is None and isinstance(doc.get("summary"), Mapping):
        sp = doc["summary"].get("spend_usd")
    if isinstance(sp, Mapping):
        fv.spend(run, sp)
        done.append("spend")
    if isinstance(doc.get("tasks"), list) and doc.get("benchmark"):
        run.table("bench_manifest/tasks", ["id", "category", "generator", "intent", "repo", "horizon_class", "milestones", "in_split", "prompt_status"],
                  [[t.get("id"), t.get("category"), t.get("generator"), t.get("intent"), t.get("repo"), t.get("horizon_class"), t.get("milestones"),
                    t.get("in_split"), t.get("prompt_status")] for t in doc["tasks"] if isinstance(t, Mapping)])
        done.append("bench_manifest")
    top = {k: v for k, v in doc.items() if isinstance(v, int | float | bool) or (isinstance(v, str) and len(v) <= 200)}
    run.summary({f"report/{k}": v for k, v in top.items()})
    run.log(scalars({"report": {k: v for k, v in doc.items() if isinstance(v, int | float | Mapping) and k not in ("families", "target", "windows")}}), step=0)
    return done


def backfill_report(p: Path, cfg: ObserveConfig) -> dict[str, Any]:
    doc = _read(p)
    if not isinstance(doc, Mapping):
        doc = {"rows": doc}
    stem = p.stem
    is_bench = bool(doc.get("benchmark"))
    is_tinker = _is_smoke(p) or ("probes" in doc and "rl" in doc)
    proj = project("bench" if is_bench else "train" if is_tinker else "factory")
    phase = str(doc.get("phase") or stem)
    name = run_name("bench-manifest", stem, _hash(p)[:8]) if is_bench else run_name("train-smoke", stem, _hash(p)[:8]) if is_tinker else \
        factory_name(slug(stem.split("_")[0], 16), phase, _hash(p))
    with ObsRun(proj, name, group="manifests" if is_bench else ("smokes" if is_tinker else f"report-{slug(stem.split('_')[0], 16)}"),
                job_type="manifest" if is_bench else ("smoke" if is_tinker else "report"), cfg=cfg, config={"source": str(p), "backfill": True},
                tags=tags(phase=phase, source=stem)) as run:
        views = _report_views(run, doc, stem)
        if is_tinker and any(k in doc for k in ("rl", "probes", "tools")):
            rl = doc.get("rl") or {}
            run.table("smoke/rl", ["run", "model", "complete", "calls", "usd", "usd_per_episode", "steps_done", "sample_tokens_per_s_per_call", "adapter"],
                      [[k, v.get("model"), v.get("complete"), v.get("calls"), v.get("usd"), v.get("usd_per_episode"), v.get("steps_done"),
                        v.get("sample_tokens_per_s_per_call"), (v.get("adapter") or {}).get("path") if isinstance(v.get("adapter"), Mapping) else None]
                       for k, v in rl.items() if isinstance(v, Mapping)])
            pr = doc.get("probes") or {}
            run.table("smoke/probes", ["probe", "n", "ok", "usd", "wall_s", "latency"], [[k, v.get("n"), v.get("ok"), v.get("usd"), v.get("wall_s"),
                                                                                           json.dumps(v.get("latency_s"))] for k, v in pr.items() if isinstance(v, Mapping)])
            tl = doc.get("tools") or {}
            if isinstance(tl, Mapping):
                run.table("smoke/tools", ["model", "summary"], [[k, json.dumps(v, default=str)[:2000]] for k, v in tl.items()])
            views.append("tinker_smoke")
        run.artifact(run_name("report", stem), "report", paths={p.name: p}, metadata={"source": str(p), "views": views})
    return {"kind": "report", "run": str(run.dir), "views": views, "project": proj}


ADAPTERS = {"train": backfill_train, "bench": backfill_bench, "compare": backfill_compare, "funnel": backfill_funnel, "report": backfill_report}


def backfill(paths: Iterable[Path | str], *, cfg: ObserveConfig | None = None, force: bool = False, kinds: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Every source under ``paths`` into offline runs and traces (module doc); returns one row per source."""
    cfg = cfg or ObserveConfig.from_env()
    cfg.root.mkdir(parents=True, exist_ok=True)
    led_p = cfg.root / "backfill.json"
    ledger: dict[str, Any] = json.loads(led_p.read_text()) if led_p.exists() else {}
    out = []
    for kind, p in discover(paths):
        if kinds and kind not in kinds:
            continue
        key = f"{p.resolve()}#{_hash(p)}"
        if key in ledger and not force:
            out.append({"kind": kind, "source": str(p), "skipped": "already backfilled", "run": ledger[key].get("run")})
            continue
        try:
            row = ADAPTERS[kind](p, cfg)
        except Exception as e:  # noqa: BLE001 -- one malformed source must not stop the backfill of the rest
            out.append({"kind": kind, "source": str(p), "error": f"{type(e).__name__}: {str(e)[:300]}"})
            continue
        row["source"] = str(p)
        ledger[key] = {**{k: v for k, v in row.items() if isinstance(v, str | int | float | list)}, "at": round(time.time(), 1)}
        led_p.write_text(json.dumps(ledger, indent=1, sort_keys=True, default=str))
        out.append(row)
    return out
