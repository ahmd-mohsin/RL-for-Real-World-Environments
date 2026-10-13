"""The benchmark report: ``summary.json`` (deterministic components + cost), ``episodes.json``, one self-contained
``report.html``, and two export files that Harbor and NeMo-Gym users can read.

The HTML follows rLLM's single-file viewer (inline CSS, no build step, archivable; ``eval/visualizer.py``) and borrows
Harbor's per-trial waffle and reward–cost Pareto views. It adds what no framework has: the per-step Φ heatmap (task ×
normalized-step bins, per model) and the gate-profile heat table. The page holds no timestamp, so two runs with equal
deterministic components render byte-identical HTML unless cost differs. Every value is escaped (``_fmt``).

M13 audit additions (``docs/audit/2026-10-13``):
- a warning banner: the runner's stop reason and owed samples (H1 V5), CIs on fewer than 10 tasks (G1 D5), refused
  pass@k columns (G1 D2), exposed episodes scored 0 (G2-1), common-support gaps;
- denominators in every cell (tasks for the mean and CI, measured / masked / owed for coverage), masks by kind (G1 D9),
  a per-task table, the run configuration (G2-18);
- the Pareto y axis spans the data (H1 V18: rewards above 1 or below 0 drew outside the plot);
- ``harbor_result.json`` (Harbor ``JobStats`` shape), ``crucible_aggregate_metrics.json`` (NeMo-Gym aggregate shape
  with ``coverage/*``) and the verifiers dual headline (``headline``), G1 §4.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from crucible.bench.metrics import CI_MIN_TASKS, Episode, deterministic_digest, heatmap

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#e5e5ea;--card:#f7f7f9;--warn:#fff4e5;--warnline:#f0b35a}
@media (prefers-color-scheme:dark){:root{--bg:#111214;--fg:#ececf0;--mut:#9a9aa3;--line:#2a2b30;--card:#1a1b1f;--warn:#2b2214;--warnline:#8a6424}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;margin:0;padding:24px 16px;max-width:1200px;margin:auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}.mut{color:var(--mut)}
table{border-collapse:collapse;width:100%;margin:6px 0;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}th{font-weight:600}
.scroll{overflow-x:auto}.cell{width:26px;height:18px;padding:0;border:1px solid var(--bg)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:8px 0}
.warn{background:var(--warn);border:1px solid var(--warnline);border-radius:8px;padding:10px 12px;margin:12px 0}
.waffle span{display:inline-block;width:12px;height:12px;margin:1px;border-radius:2px}
"""


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.3f}"
    if isinstance(x, dict) and "lo" in x and "hi" in x:  # a task-level CI
        return html.escape(f"[{x['lo']:.3f}, {x['hi']:.3f}] n={x.get('n_tasks')}" + (" approx." if x.get("approximate") else ""))
    return html.escape(str(x))


def _color(v: float | None) -> str:
    if v is None:
        return "background:transparent"
    v = max(0.0, min(1.0, float(v)))
    return f"background:hsl({int(10 + 120 * v)},65%,{int(62 - 18 * v)}%)"


def headline(det: dict[str, Any]) -> str:
    """verifiers' dual headline (``utils/format.py:57-72``): the measured mean, the raw mean (masked as 0) in parentheses
    when anything was masked, and the denominator."""
    m = det.get("mean_reward")
    if m is None:
        cov = det.get("coverage") or {}
        return f"— (nothing measured; {cov.get('masked', 0)} masked, {cov.get('owed', 0)} owed)"
    cov = det.get("coverage") or {}
    raw = det.get("mean_reward_raw")
    txt = f"{m:.3f}" + (f" ({raw:.3f})" if cov.get("masked") and raw is not None else "")
    return txt + f" over {det.get('mean_reward_tasks')} tasks, {cov.get('measured')} measured samples"


def warnings(summary: dict[str, Any]) -> list[str]:
    """Everything a reader must know before trusting a number (denominators that are not the whole benchmark)."""
    out: list[str] = []
    run = summary.get("run") or {}
    if run.get("stopped"):
        out.append(f"the runner stopped before the plan was done: {run['stopped']}")
    if run.get("owed"):
        out.append(f"{run['owed']} of {run.get('planned')} planned samples have no result: every number covers the measured samples only")
    v = summary.get("validity") or {}
    if v.get("status") == "unknown":
        out.append(f"validity unknown: no oracle or no-op evidence for these {len(v.get('unknown_tasks') or [])} tasks, so vacuous tasks (a no-op passes) "
                   "and broken ones (the oracle fails) may be counted; run with controls or pass a validity manifest")
    elif v.get("unknown_tasks"):
        out.append(f"validity unknown for {len(v['unknown_tasks'])} tasks (no oracle evidence): {', '.join(v['unknown_tasks'][:10])}")
    inv = summary.get("invalid_tasks") or {}
    if inv:
        out.append(f"{len(inv)} invalid tasks are excluded from every model's metrics, IRT and compare: "
                   + ", ".join(f"{t} ({'/'.join(x['reasons'])})" for t, x in list(inv.items())[:10]))
    for m in summary.get("models") or []:
        name = f"{m['policy']}@{m['harness']}"
        d = (m.get("overall") or {}).get("deterministic") or {}
        cov = d.get("coverage") or {}
        if cov.get("fully_masked_tasks"):
            out.append(f"{name}: {cov['fully_masked_tasks']} of {cov.get('tasks')} tasks have no measured sample and are absent from every quality metric")
        ci = d.get("mean_reward_ci")
        if isinstance(ci, dict) and ci.get("approximate"):
            out.append(f"{name}: the 95% interval rests on {ci['n_tasks']} tasks (fewer than {CI_MIN_TASKS}): approximate")
        for k in summary.get("ks") or []:
            if d.get(f"pass@{k}_excluded"):
                out.append(f"{name}: pass@{k}, pass^{k} and max@{k} refused: {d[f'pass@{k}_excluded']} tasks have fewer than {k} measured samples")
        ex = (d.get("exposure") or {}).get("exposed")
        if ex:
            out.append(f"{name}: {ex} exposed episodes (answer key reached) are scored 0")
        hk = d.get("hacks") or {}
        if hk.get("grader_tampered") or hk.get("flagged_passes"):
            out.append(f"{name}: {hk.get('grader_tampered', 0)} grader-tampering episodes (scored 0) and {hk.get('flagged_passes', 0)} passes with a rule flag "
                       f"{sorted(hk.get('flag_rules') or {})}")
    cs = summary.get("common_support")
    if cs and any(v["dropped_tasks"] for v in cs["models"].values()):
        out.append(f"models were measured on different task sets: compare them on the {cs['tasks']} common tasks (common support table)")
    return out


def _pareto_svg(points: list[tuple[str, float, float]]) -> str:
    if not points:
        return ""
    w, h, pad = 520, 220, 36
    xmax = max(p[1] for p in points) or 1.0
    ylo, yhi = min(0.0, *(p[2] for p in points)), max(1.0, *(p[2] for p in points))  # the data's range (H1 V18)
    marks = []
    for name, cost, reward in points:
        x = pad + (w - 2 * pad) * (cost / xmax)
        y = h - pad - (h - 2 * pad) * ((reward - ylo) / (yhi - ylo))
        marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="currentColor"/><text x="{x + 8:.1f}" y="{y + 4:.1f}" font-size="11">{html.escape(name)}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" role="img" aria-label="reward vs cost">'
            f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="currentColor" opacity=".3"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="currentColor" opacity=".3"/>'
            f'<text x="{w - pad}" y="{h - 8}" font-size="11" text-anchor="end">cost per planned sample</text>'
            f'<text x="8" y="{pad - 10}" font-size="11">mean reward ({ylo:g} to {yhi:g})</text>{"".join(marks)}</svg>')


def render(title: str, summary: dict[str, Any], eps: Sequence[Episode], *, extras: dict[str, Any] | None = None, bins: int = 10) -> str:
    extras = extras or {}
    hm = heatmap(eps, bins=bins)
    run = summary.get("run") or {}
    out = [(f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"),
           (f"<h1>{html.escape(title)}</h1><div class=mut>deterministic digest {deterministic_digest(summary)} · threshold {_fmt(summary.get('threshold'))}"
            f" · planned {_fmt(run.get('planned'))} samples, measured {_fmt(run.get('measured'))}, masked {_fmt(run.get('masked'))}, owed {_fmt(run.get('owed'))}"
            " · masked samples are excluded from every quality metric; exposed samples score 0</div>")]
    warn = summary.get("warnings") if summary.get("warnings") is not None else warnings(summary)
    if warn:
        out.append("<div class=warn role=alert><b>Read before using these numbers</b><ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in warn) + "</ul></div>")
    ks = summary.get("ks") or [1]
    out.append("<h2>Models</h2><div class=scroll><table><tr><th>policy</th><th>harness</th><th>measured / masked / owed</th>"
               "<th>mean reward of task means (raw)</th><th>tasks</th><th>95% CI on task means</th>"
               + "".join(f"<th>pass@{k}</th><th>pass^{k}</th>" for k in ks)
               + "<th>hacks (exposed / tampered / flagged passes)</th><th>progress AUC</th><th>stage reach (frac)</th><th>regressions</th><th>msgs / success</th><th>USD / success</th></tr>")
    points = []
    for m in summary["models"]:
        d, c = m["overall"]["deterministic"], m["overall"]["cost"]
        cov = d.get("coverage", {})
        hk = d.get("hacks") or {}
        out.append(f"<tr><td>{_fmt(m['policy'])}{' (control)' if m.get('control') else ''}</td><td>{_fmt(m['harness'])}</td>"
                   f"<td>{_fmt(cov.get('measured'))} / {_fmt(cov.get('masked'))} / {_fmt(cov.get('owed'))}</td>"
                   f"<td>{_fmt(d.get('mean_reward'))} ({_fmt(d.get('mean_reward_raw'))})</td><td>{_fmt(d.get('mean_reward_tasks'))}</td><td>{_fmt(d.get('mean_reward_ci'))}</td>"
                   + "".join(f"<td>{_fmt(d.get(f'pass@{k}'))}</td><td>{_fmt(d.get(f'pass^{k}'))}</td>" for k in ks)
                   + f"<td>{_fmt(hk.get('exposed'))} / {_fmt(hk.get('grader_tampered'))} / {_fmt(hk.get('flagged_passes'))}</td><td>{_fmt(d.get('progress_auc'))}</td><td>{_fmt(d.get('stage_reach_frac'))}</td>"
                   f"<td>{_fmt(d.get('regressions'))}</td><td>{_fmt(d.get('messages_per_success'))}</td><td>{_fmt((c.get('per_success') or {}).get('usd'))}</td></tr>")
        if d.get("mean_reward") is not None and not m.get("control"):
            points.append((m["policy"], c.get("usd", 0.0) / max(1, cov.get("planned", 1)), d["mean_reward"]))
    out.append("</table></div>")
    v = summary.get("validity") or {}
    inv = summary.get("invalid_tasks") or {}
    out.append(f"<h2>Validity</h2><div class=mut>status {_fmt(v.get('status'))}; valid {_fmt(v.get('valid_tasks'))}, invalid {len(inv)}, "
               f"unknown {len(v.get('unknown_tasks') or [])}; sources {_fmt(', '.join(v.get('sources') or []) or None)}</div>")
    if inv:
        out.append("<div class=scroll><table><tr><th>invalid task (excluded)</th><th>reasons</th><th>per model: mean reward (measured)</th></tr>")
        for t, x in inv.items():
            per = "; ".join(f"{mm}: {_fmt(r.get('mean_reward'))} ({r['coverage'].get('measured')})" for mm, r in x.get("models", {}).items())
            out.append(f"<tr><td>{_fmt(t)}</td><td>{_fmt(', '.join(x['reasons']))}</td><td>{html.escape(per)}</td></tr>")
        out.append("</table></div>")
    cs = summary.get("common_support")
    if cs:
        out.append(f"<h2>Common support</h2><div class=mut>every model on the {_fmt(cs['tasks'])} tasks measured for all models</div><div class=scroll><table>"
                   "<tr><th>model</th><th>mean reward</th><th>pass@1</th><th>measured tasks</th><th>dropped tasks</th></tr>")
        for name, v in cs["models"].items():
            out.append(f"<tr><td>{_fmt(name)}</td><td>{_fmt(v['mean_reward'])}</td><td>{_fmt(v['pass@1'])}</td><td>{_fmt(v['measured_tasks'])}</td>"
                       f"<td>{_fmt(', '.join(v['dropped_tasks']))}</td></tr>")
        out.append("</table></div>")
    kinds = sorted({k for m in summary["models"] for k in ((m["overall"]["deterministic"].get("coverage") or {}).get("masked_by_kind") or {})})
    if kinds:
        out.append("<h2>Masked samples by kind</h2><div class=scroll><table><tr><th>model</th>" + "".join(f"<th>{html.escape(k)}</th>" for k in kinds) + "<th>retries</th></tr>")
        for m in summary["models"]:
            mk = (m["overall"]["deterministic"].get("coverage") or {}).get("masked_by_kind") or {}
            out.append(f"<tr><td>{_fmt(m['policy'])}@{_fmt(m['harness'])}</td>" + "".join(f"<td>{_fmt(mk.get(k, 0))}</td>" for k in kinds)
                       + f"<td>{_fmt(m['overall']['cost'].get('retries'))}</td></tr>")
        out.append("</table></div>")
    gates = sorted({g for m in summary["models"] for g in (m["overall"]["deterministic"].get("gate_profile") or {}).get("gates", {})})
    if gates:
        out.append("<h2>Gate profile</h2><div class=scroll><table><tr><th>model</th>" + "".join(f"<th>{html.escape(g)}</th>" for g in gates)
                   + "<th>verifier-level</th></tr>")
        for m in summary["models"]:
            gp = m["overall"]["deterministic"].get("gate_profile") or {}
            cells = "".join(f"<td style='{_color((gp.get('gates', {}).get(g) or {}).get('pass_rate'))}'>{_fmt((gp.get('gates', {}).get(g) or {}).get('pass_rate'))}</td>" for g in gates)
            out.append(f"<tr><td>{_fmt(m['policy'])}</td>{cells}<td>{_fmt(gp.get('verifier_level_pass_rate'))}</td></tr>")
        out.append("</table></div>")
    out.append("<h2>Per task</h2>")
    for m in summary["models"]:
        out.append(f"<div class=card><b>{_fmt(m['policy'])}@{_fmt(m['harness'])}</b><div class=scroll><table><tr><th>task</th><th>measured / masked / owed</th>"
                   "<th>mean reward</th><th>pass@1</th><th>exposed</th></tr>")
        for t, g in (m.get("tasks") or {}).items():
            d = g["deterministic"]
            cov = d.get("coverage") or {}
            out.append(f"<tr><td>{_fmt(t)}</td><td>{_fmt(cov.get('measured'))} / {_fmt(cov.get('masked'))} / {_fmt(cov.get('owed'))}</td>"
                       f"<td>{_fmt(d.get('mean_reward'))}</td><td>{_fmt(d.get('pass@1'))}</td><td>{_fmt((d.get('exposure') or {}).get('exposed'))}</td></tr>")
        out.append("</table></div></div>")
    out.append(f"<h2>Progress Φ per normalized step</h2><div class=mut>mean Φ at the end of each of {bins} step bins; one row per task</div>")
    for model, tasks in hm.items():
        out.append(f"<div class=card><b>{html.escape(model)}</b><div class=scroll><table><tr><th>task</th>"
                   + "".join(f"<th class=mut>{int((b + 1) * 100 / bins)}%</th>" for b in range(bins)) + "</tr>")
        for task, cells in tasks.items():
            out.append(f"<tr><td>{html.escape(task)}</td>" + "".join(f"<td class=cell style='{_color(v)}' title='{_fmt(v)}'></td>" for v in cells) + "</tr>")
        out.append("</table></div></div>")
    out.append("<h2>Episodes</h2><div class=mut>one cell per planned sample: colour = reward; grey = masked; outlined = owed</div>")
    by_model: dict[str, list[Episode]] = {}
    for e in eps:
        by_model.setdefault(f"{e.policy}@{e.harness}", []).append(e)
    for model, es in sorted(by_model.items()):
        cells = "".join(f"<span title='{html.escape(e.task)}#{e.sample_idx}: {'owed' if e.owed else _fmt(e.value)}{' exposed' if e.exposed else ''}' "
                        f"style='{_color(e.value) if not e.masked else ('outline:1px solid #8888' if e.owed else 'background:#8884')}'></span>"
                        for e in sorted(es, key=lambda e: (e.task, e.sample_idx)))
        out.append(f"<div class='card waffle'><b>{html.escape(model)}</b><br>{cells}</div>")
    if points:
        out.append(f"<h2>Reward vs cost</h2>{_pareto_svg(points)}")
    if summary.get("config"):
        out.append(f"<h2>Configuration</h2><div class=card><pre style='white-space:pre-wrap;margin:0'>{html.escape(json.dumps(summary['config'], indent=1, sort_keys=True, default=str))}</pre></div>")
    for name, val in extras.items():
        out.append(f"<h2>{html.escape(name)}</h2><div class=card><pre style='white-space:pre-wrap;margin:0'>{html.escape(json.dumps(val, indent=1, sort_keys=True, default=str))}</pre></div>")
    out.append("</body></html>")
    return "\n".join(out)


def harbor_job_stats(summary: dict[str, Any], eps: Sequence[Episode], *, dataset: str = "crucible") -> dict[str, Any]:
    """Harbor's ``JobStats`` shape (``harbor/models/job/result.py``): evals keyed ``agent__model__dataset`` (here
    ``harness__policy__dataset``). Harbor scores errored trials 0; here masked trials are excluded and listed under
    ``exception_stats``, and the ``crucible`` block says so."""
    run = summary.get("run") or {}
    evals: dict[str, Any] = {}
    for m in summary["models"]:
        d, c = m["overall"]["deterministic"], m["overall"]["cost"]
        mine = [e for e in eps if e.policy == m["policy"] and e.harness == m["harness"]]
        rs: dict[str, dict[str, list[str]]] = {"reward": {}}
        exc: dict[str, list[str]] = {}
        for e in mine:
            name = e.attempt or f"{e.task}#{e.sample_idx}"
            if e.masked:
                exc.setdefault("owed" if e.owed else (e.failure_kind or "unknown"), []).append(name)
            else:
                rs["reward"].setdefault(str(e.value), []).append(name)
        evals[f"{m['harness']}__{m['policy']}__{dataset}"] = {
            "n_trials": (d.get("coverage") or {}).get("measured", 0), "n_errors": sum(len(v) for v in exc.values()),
            "metrics": [{"mean": d.get("mean_reward")}] if d.get("mean_reward") is not None else [],
            "pass_at_k": {str(k): d[f"pass@{k}"] for k in summary.get("ks") or [] if d.get(f"pass@{k}") is not None},
            "reward_stats": rs, "exception_stats": exc, "n_retries": c.get("retries", 0)}
    return {"n_completed_trials": run.get("terminal", 0), "n_errored_trials": run.get("masked", 0), "n_running_trials": 0,
            "n_pending_trials": run.get("owed", 0), "n_cancelled_trials": 0, "n_retries": sum(e.attempts - (0 if e.owed else 1) for e in eps if e.attempts),
            "evals": evals, "n_input_tokens": sum(e.tokens_in for e in eps), "n_cache_tokens": None, "n_output_tokens": sum(e.tokens_out for e in eps),
            "cost_usd": round(sum(e.cost_usd for e in eps), 6),
            "crucible": {"masked": "excluded from metrics, never scored 0 (DESIGN P16); listed under exception_stats",
                         "exposed": "scored 0", "mean": "mean of per-task means", "stopped": run.get("stopped"), "digest": deterministic_digest(summary)}}


def nemo_gym_aggregate(summary: dict[str, Any], eps: Sequence[Episode]) -> list[dict[str, Any]]:
    """NeMo-Gym's ``_aggregate_metrics.json`` shape (``rollout_collection.py:3049-3094``): per agent ``agent_metrics`` /
    ``key_metrics`` (pass@k ×100, ``coverage/*``), ``group_level_metrics`` per task, ``repeat_level_metrics`` per sample
    index, so ``gym eval compare`` reads a Crucible run."""
    out = []
    for m in summary["models"]:
        d = m["overall"]["deterministic"]
        cov = d.get("coverage") or {}
        km: dict[str, Any] = {"mean/reward": d.get("mean_reward"),
                              **{f"pass@{k}/accuracy": 100.0 * d[f"pass@{k}"] for k in summary.get("ks") or [] if d.get(f"pass@{k}") is not None}}
        am = {**km, "coverage/expected": cov.get("planned"), "coverage/scored": cov.get("measured"), "coverage/missing": (cov.get("masked") or 0) + (cov.get("owed") or 0),
              "coverage/measured_rollouts": cov.get("measured"), "coverage/masked_rollouts": cov.get("masked"), "coverage/owed_rollouts": cov.get("owed"),
              "coverage/measured_tasks": cov.get("measured_tasks"), "coverage/fully_masked_tasks": cov.get("fully_masked_tasks")}
        mine = [e for e in eps if e.policy == m["policy"] and e.harness == m["harness"]]
        groups = []
        for t, g in (m.get("tasks") or {}).items():
            gd = g["deterministic"]
            groups.append({"task": t, "mean/reward": gd.get("mean_reward"), "pass@1/accuracy": None if gd.get("pass@1") is None else 100.0 * gd["pass@1"],
                           "coverage/scored": (gd.get("coverage") or {}).get("measured")})
        reps: dict[int, list[float]] = {}
        for e in mine:
            if not e.masked:
                reps.setdefault(e.sample_idx, []).append(float(e.value or 0.0))
        out.append({"agent_ref": {"name": f"{m['policy']}@{m['harness']}"}, "agent_metrics": am, "key_metrics": km, "group_level_metrics": groups,
                    "repeat_level_metrics": [{"repeat": i, "mean/reward": round(sum(v) / len(v), 9), "n": len(v)} for i, v in sorted(reps.items())]})
    return out


def write(out_dir: Path, title: str, summary: dict[str, Any], eps: Sequence[Episode], *, extras: dict[str, Any] | None = None) -> dict[str, str]:
    from crucible.bench.metrics import episodes_json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary.get("warnings") is None:
        summary["warnings"] = warnings(summary)
    doc = {"title": title, "digest": deterministic_digest(summary), "summary": summary, **(extras or {})}
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    (out_dir / "episodes.json").write_text(json.dumps(episodes_json(eps), indent=1, sort_keys=True, default=str) + "\n")
    (out_dir / "report.html").write_text(render(title, summary, eps, extras=extras))
    (out_dir / "harbor_result.json").write_text(json.dumps(harbor_job_stats(summary, eps), indent=1, sort_keys=True, default=str) + "\n")
    (out_dir / "crucible_aggregate_metrics.json").write_text(json.dumps(nemo_gym_aggregate(summary, eps), indent=1, sort_keys=True, default=str) + "\n")
    return {"summary": str(out_dir / "summary.json"), "episodes": str(out_dir / "episodes.json"), "report": str(out_dir / "report.html"),
            "harbor": str(out_dir / "harbor_result.json"), "nemo_gym": str(out_dir / "crucible_aggregate_metrics.json")}
