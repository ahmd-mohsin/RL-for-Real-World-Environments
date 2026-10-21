"""Benchmark reports in W&B (swarm-pipeline/11 §4): one run per arm, one run per comparison, every episode a trace.

``log_benchmark(report, run_dir)`` (one arm: ``factory.tinker.benchmark``'s report and its run directory):

- config: the arm's settings (``arm_settings``: model, context, k, temperature, max tokens, thinking, tool limit, timeouts,
  system prompt, harness, codec), checkpoint, purpose, seed, the task manifest hash;
- scalars: every number of ``metrics`` (pass@1, pass^k, mean final Φ, CIs, calibration, give-up, hack rate, USD per
  passing episode); a ``pass@k`` and a ``pass^k`` curve;
- tables: ``bench/per_task`` (c of n with a Wilson interval), ``bench/by_language``, ``bench/by_family``, ``bench/by_skill``,
  ``bench/episodes`` (one row per episode with its Weave trace link and, once audited, the cause and the quote);
- artifacts: the benchmark manifest (task list, settings, hashes), the report files, the episodes' full text.

``log_compare(compare, run_dir)`` (``factory.tinker.compare``: control and trained, plus the fresh baseline and the
selection arms when given) adds the headline: pass@1 per arm with its task-clustered CI (an HTML bar chart with CIs and
a native bar chart), the per-task chart, the paired trained − control difference with its CI and MDE, pass@k and pass^k
curves per arm, cost per passing episode, the per-language / family / skill breakdowns per arm, and ``check_protocol``'s
result (``protocol/ok`` and the problems). Every arm of one task list shares the W&B group ``tasks-<manifest hash>``.

**Power** (``bench.power``, swarm-pipeline/11 §3): an arm's ``power`` block becomes ``bench/power/<metric>/*`` scalars
(MDE, target, N, k, powered) and the tag ``power:<verdict>``; a comparison's ``power`` stamp becomes ``compare/power/*``,
the ``compare/power`` table, the tag ``power:<verdict>`` and the summary's ``headline``, and an ``underpowered`` stamp
leads the headline chart's subtitle.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.observe import charts
from crucible.observe.naming import bench_name, digest, manifest_hash, project, run_name, tags
from crucible.observe.run import ObserveConfig, ObsRun, scalars
from crucible.observe.traces import EPISODE_COLUMNS, TraceStore, episode_rows

ARM_ORDER = ("selection", "baseline", "control", "trained")
DIMS = ("language", "family", "skill")


def wilson(c: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = c / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def read_rows(run_dir: Path | str) -> list[dict[str, Any]]:
    p = Path(run_dir) / "episodes.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def per_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """task -> scored n, passes c, pass@1 with its Wilson interval, mean final Φ, USD, USD per passing episode, masked."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        t = out.setdefault(str(r.get("task")), {"n": 0, "c": 0, "masked": 0, "phi": [], "usd": 0.0})
        t["usd"] += float(r.get("usd") or 0.0)
        if r.get("masked"):
            t["masked"] += 1
            continue
        t["n"] += 1
        t["c"] += 1 if r.get("passed") else 0
        if r.get("final_phi") is not None:
            t["phi"].append(float(r["final_phi"]))
    for t in out.values():
        lo, hi = wilson(t["c"], t["n"])
        phis = t.pop("phi")
        t.update(pass1=t["c"] / t["n"] if t["n"] else None, lo=lo, hi=hi, mean_final_phi=sum(phis) / len(phis) if phis else None,
                 usd_per_pass=t["usd"] / t["c"] if t["c"] else None)
    return out


def breakdown(rows: Sequence[Mapping[str, Any]], task_meta: Mapping[str, Mapping[str, Any]] | None, dim: str) -> dict[str, dict[str, Any]]:
    """Per value of ``dim`` (language | family | skill): tasks, mean per-task pass@1 (tasks weighted equally, as pass@1 is),
    episodes, USD per passing episode."""
    pt = per_task(rows)
    groups: dict[str, list[str]] = {}
    for t in pt:
        v = str(((task_meta or {}).get(t) or {}).get(dim) or "unknown")
        groups.setdefault(v, []).append(t)
    out = {}
    for v, ts in sorted(groups.items()):
        p1 = [pt[t]["pass1"] for t in ts if pt[t]["pass1"] is not None]
        c = sum(pt[t]["c"] for t in ts)
        usd = sum(pt[t]["usd"] for t in ts)
        out[v] = {"tasks": len(ts), "pass1": sum(p1) / len(p1) if p1 else None, "episodes": sum(pt[t]["n"] for t in ts), "passes": c,
                  "usd_per_pass": usd / c if c else None}
    return out


def _ks(d: Mapping[str, Any] | None) -> tuple[list[float], list[float | None]]:
    items = sorted(((int(k), v) for k, v in (d or {}).items()), key=lambda x: x[0])
    return [float(k) for k, _ in items], [v for _, v in items]


def _per_task_table(run: ObsRun, key: str, pt: Mapping[str, Mapping[str, Any]], meta: Mapping[str, Mapping[str, Any]] | None, arm: str | None = None) -> None:
    cols = ["task", *(["arm"] if arm else []), "n", "c", "pass@1", "lo", "hi", "mean_final_phi", "masked", "usd", "usd_per_pass", *DIMS]
    rows = [[t, *([arm] if arm else []), v["n"], v["c"], v["pass1"], v["lo"], v["hi"], v["mean_final_phi"], v["masked"], round(v["usd"], 6), v["usd_per_pass"],
             *(((meta or {}).get(t) or {}).get(d) for d in DIMS)] for t, v in sorted(pt.items())]
    run.table(key, cols, rows)


def _breakdowns(run: ObsRun, arms_rows: Mapping[str, Sequence[Mapping[str, Any]]], meta: Mapping[str, Mapping[str, Any]] | None, prefix: str) -> None:
    for dim in DIMS:
        rows, vals = [], {}
        for arm, rs in arms_rows.items():
            for v, b in breakdown(rs, meta, dim).items():
                rows.append([v, arm, b["tasks"], b["episodes"], b["passes"], b["pass1"], b["usd_per_pass"]])
                vals[(v, arm)] = (b["pass1"], None, None)
        run.table(f"{prefix}/by_{dim}", [dim, "arm", "tasks", "episodes", "passes", "pass@1", "usd_per_pass"], rows)
        groups = sorted({r[0] for r in rows})
        if groups and len(arms_rows) > 1:
            run.html(f"{prefix}/by_{dim}_chart", charts.grouped_bars_ci(groups, list(arms_rows), vals, title=f"pass@1 by {dim}",
                                                                         subtitle="mean over tasks of per-task pass@1"))
        elif groups:
            arm = next(iter(arms_rows))
            run.bar(f"{prefix}/by_{dim}_bar", groups, [vals[(g, arm)][0] for g in groups], title=f"pass@1 by {dim}", label=dim, value="pass@1")


def _export_traces(cfg: ObserveConfig, proj: str, rollouts: Path, arm: str, run_name_: str, model: str | None,
                   attrs: Mapping[str, Any] | None = None) -> tuple[TraceStore, list[dict[str, Any]]]:
    store = TraceStore(cfg.root, proj)
    rows = store.export_all(rollouts, arm=arm, run=run_name_, model=model, attrs=attrs, max_chars=cfg.max_chars) if rollouts.exists() else []
    return store, rows


def log_benchmark(report: Mapping[str, Any], run_dir: Path | str, *, task_meta: Mapping[str, Mapping[str, Any]] | None = None,
                  cfg: ObserveConfig | None = None, traces: bool = True, extra_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One benchmark arm in W&B (module doc). ``task_meta``: task id -> {language, family, skill} for the breakdowns."""
    cfg = cfg or ObserveConfig.from_env()
    run_dir = Path(run_dir)
    proj = project("bench")
    purpose = str(report.get("purpose") or "baseline")
    tasks = [str(t) for t in report.get("tasks") or []]
    mh = manifest_hash(tasks)
    model = str(report.get("model") or "")
    settings = report.get("settings") or {}
    ident = digest([mh, purpose, report.get("seed"), report.get("checkpoint"), str(run_dir.resolve())])
    name = bench_name(purpose, model, report.get("seed"), ident)
    langs = sorted({str((task_meta or {}).get(t, {}).get("language")) for t in tasks if (task_meta or {}).get(t, {}).get("language")})
    config = {"kind": "benchmark", "purpose": purpose, "model": model, "checkpoint": report.get("checkpoint"), "seed": report.get("seed"), "k": report.get("k"),
              "settings": settings, "tasks_manifest_hash": mh, "n_tasks": len(tasks), "run_dir": str(run_dir), **dict(extra_config or {})}
    rows = read_rows(run_dir)
    store, trows = (_export_traces(cfg, proj, run_dir / "rollouts", purpose, name, model, {"purpose": purpose, "checkpoint": report.get("checkpoint")})
                    if traces else (None, []))
    with ObsRun(proj, name, group=f"tasks-{mh}", job_type=purpose, cfg=cfg, config=config,
                tags=tags(model=model, purpose=purpose, k=report.get("k"), thinking=settings.get("enable_thinking"), harness=settings.get("harness"),
                          lang=langs, checkpoint="adapter" if report.get("checkpoint") else "base",
                          power=power_verdict(report.get("power")))) as run:
        m = report.get("metrics") or {}
        run.log(scalars({"bench": {**{k: v for k, v in m.items() if k not in ("pass@k", "pass^k")}, "power": power_scalars(report.get("power"))}}), step=0)
        if report.get("power"):
            run.table("bench/power", ["metric", "verdict", "mde", "mde_aa", "target", "n_tasks", "k", "n_needed", "source", "headline"],
                      [_gate_row(m_, b) for m_, b in sorted(report["power"].items())])
        xs, pak = _ks(m.get("pass@k"))
        _, phk = _ks(m.get("pass^k"))
        if xs:
            run.lines("bench/pass_at_k", xs, [pak, phk], ["pass@k", "pass^k"], title="pass@k and pass^k", xname="k")
            run.html("bench/pass_at_k_chart", charts.lines(xs, {"pass@k": pak, "pass^k": phk}, title="pass@k and pass^k", subtitle=f"{purpose} | {model}"))
        pt = per_task(rows)
        _per_task_table(run, "bench/per_task", pt, task_meta)
        if pt:
            run.html("bench/per_task_chart", charts.grouped_bars_ci(sorted(pt), [purpose], {(t, purpose): (v["pass1"], v["lo"], v["hi"]) for t, v in pt.items()},
                                                                     title="pass@1 per task", subtitle=f"{purpose} | Wilson 95% intervals over k attempts"))
        _breakdowns(run, {purpose: rows}, task_meta, "bench")
        if trows:
            run.table("bench/episodes", EPISODE_COLUMNS, episode_rows([store.index()[r["episode"]] for r in trows]))  # type: ignore[union-attr]
            run.artifact(run_name("episodes", name), "episode-traces", paths=store.full_paths([r["episode"] for r in trows]),  # type: ignore[union-attr]
                         metadata={"project": proj, "episodes": len(trows), "weave_spans": str(store.spans_path)})  # type: ignore[union-attr]
        run.artifact(f"bench-manifest-{mh}", "benchmark-manifest", files={"manifest.json": {"tasks": tasks, "tasks_manifest_hash": mh, "settings": settings,
                                                                                               "task_meta": {t: (task_meta or {}).get(t) for t in tasks}}})
        run.artifact(run_name("bench-report", name), "benchmark-report",
                     paths={p.name: p for p in (run_dir / "metrics.json", run_dir / "episodes.jsonl", run_dir / "spend.jsonl") if p.exists()},
                     metadata={"checkpoint": report.get("checkpoint"), "model": model, "purpose": purpose})
        run.summary({"pass@1": m.get("pass@1"), "usd_per_passing_episode": m.get("usd_per_passing_episode"), "episodes": m.get("episodes"),
                     "masked": m.get("masked"), "tasks_manifest_hash": mh, "checkpoint": report.get("checkpoint"),
                     "power": (((report.get("power") or {}).get("pass") or {}).get("gate") or {}).get("headline")})
    return {"run": str(run.dir), "sync": [str(p) for p in run.sync_dirs], "traces": len(trows), "project": proj, "name": name}


def power_scalars(power: Mapping[str, Any] | None) -> dict[str, Any]:
    """An arm's ``power`` block (``bench.power.power_report``) as flat numbers per metric."""
    out: dict[str, Any] = {}
    for m, blk in (power or {}).items():
        g = (blk or {}).get("gate") or {}
        nz = (blk or {}).get("noise") or {}
        aa = nz.get("aa") or {}
        out[m] = {"mde": g.get("mde"), "mde_aa": g.get("mde_aa"), "target": g.get("target"), "n_tasks": g.get("n_tasks"), "k": g.get("k"),
                  "n_needed": g.get("n_needed"), "powered": g.get("verdict") == "powered", "sigma2": nz.get("sigma2"), "aa_diff": aa.get("diff"),
                  "aa_se": aa.get("se")}
    return out


def power_verdict(power: Mapping[str, Any] | None, metric: str = "pass") -> str | None:
    return (((power or {}).get(metric) or {}).get("gate") or {}).get("verdict")


def _gate_row(metric: str, blk: Mapping[str, Any]) -> list[Any]:
    g = blk.get("gate") or {}
    return [metric, *(g.get(c) for c in ("verdict", "mde", "mde_aa", "target", "n_tasks", "k", "n_needed", "source", "headline"))]


def _arm_ci(arm: Mapping[str, Any]) -> tuple[float | None, float | None, float | None]:
    m = arm.get("metrics") or {}
    ci = (m.get("ci") or {}).get("pass@1") or {}
    return m.get("pass@1"), ci.get("lo"), ci.get("hi")


def log_compare(cmp: Mapping[str, Any], run_dir: Path | str, *, task_meta: Mapping[str, Mapping[str, Any]] | None = None, cfg: ObserveConfig | None = None,
                arm_dirs: Mapping[str, Path | str] | None = None) -> dict[str, Any]:
    """The comparison in W&B (module doc). ``arm_dirs``: arm -> its benchmark run directory (default ``run_dir/<arm>``
    for control and trained); the baseline's and selection's give their per-task bars and breakdowns."""
    from crucible.observe.report import write_template

    cfg = cfg or ObserveConfig.from_env()
    run_dir = Path(run_dir)
    proj = project("bench")
    arms = {a: cmp["arms"][a] for a in ARM_ORDER if a in (cmp.get("arms") or {})}
    tasks = [str(t) for t in (arms.get("control") or next(iter(arms.values()), {})).get("tasks") or []]
    mh = manifest_hash(tasks)
    dirs = {**{a: run_dir / a for a in ("control", "trained")}, **{a: Path(p) for a, p in (arm_dirs or {}).items()}}
    arm_rows = {a: read_rows(dirs[a]) for a in arms if a in dirs and (Path(dirs[a]) / "episodes.jsonl").exists()}
    model = str(cmp.get("model") or "")
    ident = digest([mh, cmp.get("checkpoint"), cmp.get("seed"), str(run_dir.resolve())])
    name = run_name("bench", "compare", model.rsplit("/", 1)[-1], f"s{cmp.get('seed')}", ident[:8])
    problems = list(cmp.get("protocol_problems") or [])
    pw = cmp.get("power") or {}
    config = {"kind": "compare", "model": model, "checkpoint": cmp.get("checkpoint"), "k": cmp.get("k"), "seed": cmp.get("seed"), "settings": cmp.get("settings"),
              "tasks_manifest_hash": mh, "n_tasks": len(tasks), "arms": list(arms), "arm_seeds": {a: v.get("seed") for a, v in arms.items()},
              "run_dir": str(run_dir)}
    with ObsRun(proj, name, group=f"tasks-{mh}", job_type="compare", cfg=cfg, config=config,
                tags=tags(model=model, purpose="compare", k=cmp.get("k"), protocol="ok" if not problems else "problems", power=pw.get("verdict"))) as run:
        vals = {("pass@1", a): _arm_ci(v) for a, v in arms.items()}
        run.table("compare/arms", ["arm", "pass@1", "ci_lo", "ci_hi", "seed", "checkpoint", "episodes", "scored", "usd_per_passing_episode", "usd_total"],
                  [[a, *_arm_ci(v), v.get("seed"), v.get("checkpoint"), (v.get("metrics") or {}).get("episodes"), (v.get("metrics") or {}).get("scored"),
                    (v.get("metrics") or {}).get("usd_per_passing_episode"), (v.get("metrics") or {}).get("usd_total")] for a, v in arms.items()])
        run.bar("compare/pass@1", list(arms), [vals[("pass@1", a)][0] for a in arms], title="pass@1 per arm", label="arm", value="pass@1")
        d = cmp.get("paired_trained_minus_control") or {}
        ci = d.get("ci") or [None, None]
        sub = (f"k={cmp.get('k')} per task, {len(tasks)} tasks, task-clustered 95% CIs; trained - control = "
               f"{d.get('mean_diff') if d.get('mean_diff') is None else round(100 * d['mean_diff'], 1)} points "
               f"(CI {None if ci[0] is None else round(100 * ci[0], 1)} to {None if ci[1] is None else round(100 * ci[1], 1)}; MDE "
               f"{None if d.get('mde') is None else round(100 * d['mde'], 1)})" + ("" if not problems else f"; PROTOCOL: {'; '.join(problems)}"))
        if pw and pw.get("verdict") != "powered":
            sub = f"{pw.get('headline')} | {sub}"
        run.html("compare/headline", charts.grouped_bars_ci(["pass@1"], list(arms), vals, title=f"{model}: pass@1 before and after RL", subtitle=sub))
        run.log(scalars({"compare": {"paired": {k: d.get(k) for k in ("mean_diff", "sd", "mde", "mde_assumed", "significant", "n_tasks", "p_better")},
                                     "power": {"mde": pw.get("mde"), "target": pw.get("target"), "n_tasks": pw.get("n_tasks"), "k": pw.get("k"),
                                               "powered": pw.get("verdict") == "powered", "preregistered": pw.get("preregistered"),
                                               "by_metric": power_scalars(pw.get("metrics"))},
                                     "paired_ci_lo": ci[0], "paired_ci_hi": ci[1], "protocol_ok": not problems,
                                     "pass@1": {a: vals[("pass@1", a)][0] for a in arms}}}), step=0)
        run.table("compare/paired_per_task", ["task", "trained_minus_control"], [[t, v] for t, v in sorted((d.get("per_task") or {}).items())])
        run.table("compare/protocol", ["problem"], [[p] for p in problems] or [["none"]])
        run.table("compare/power", ["metric", "verdict", "mde", "mde_aa", "target", "n_tasks", "k", "n_needed", "source", "headline"],
                  [_gate_row(m_, b) for m_, b in sorted((pw.get("metrics") or {}).items())] or [["pass", "unmeasured", *([None] * 8)]])
        pts = {a: per_task(rs) for a, rs in arm_rows.items()}
        for a, pt in pts.items():
            _per_task_table(run, f"compare/per_task_{a}", pt, task_meta, arm=a)
        if pts:
            all_t = sorted({t for pt in pts.values() for t in pt})
            run.html("compare/per_task_chart", charts.grouped_bars_ci(all_t, list(pts), {(t, a): (pt[t]["pass1"], pt[t]["lo"], pt[t]["hi"]) for a, pt in pts.items()
                                                                                            for t in pt}, title="pass@1 per task and arm",
                                                                       subtitle="k attempts per task; Wilson 95% intervals"))
        xs: list[float] = []
        curves, hats = {}, {}
        for a, v in arms.items():
            m = v.get("metrics") or {}
            xa, ya = _ks(m.get("pass@k"))
            _, yh = _ks(m.get("pass^k"))
            if xa:
                xs = xa if len(xa) > len(xs) else xs
                curves[a], hats[a] = ya, yh
        if curves:
            run.lines("compare/pass_at_k", xs, list(curves.values()), list(curves), title="pass@k per arm", xname="k")
            run.lines("compare/pass_hat_k", xs, list(hats.values()), list(hats), title="pass^k per arm", xname="k")
            run.html("compare/pass_at_k_chart", charts.lines(xs, curves, title="pass@k per arm"))
            run.html("compare/pass_hat_k_chart", charts.lines(xs, hats, title="pass^k per arm"))
        run.bar("compare/usd_per_passing_episode", list(arms), [(v.get("metrics") or {}).get("usd_per_passing_episode") for v in arms.values()],
                title="USD per passing episode", label="arm", value="usd")
        if arm_rows:
            _breakdowns(run, arm_rows, task_meta, "compare")
        tpl = write_template(run.dir / "report_template.json", group=f"tasks-{mh}", model=model)
        run.artifact(run_name("compare-report", name), "benchmark-compare", files={"compare.json": dict(cmp)}, paths={"report_template.json": tpl},
                     metadata={"checkpoint": cmp.get("checkpoint"), "protocol_problems": problems})
        run.summary({"protocol_ok": not problems, "protocol_problems": "; ".join(problems) or "none", "paired_note": d.get("note"),
                     "headline": cmp.get("headline"), "power_verdict": pw.get("verdict"), "power_headline": pw.get("headline"),
                     "paired_mean_diff": d.get("mean_diff"), "checkpoint": cmp.get("checkpoint"), "tasks_manifest_hash": mh})
    return {"run": str(run.dir), "sync": [str(p) for p in run.sync_dirs], "project": proj, "name": name, "group": f"tasks-{mh}"}
