"""The benchmark report: ``summary.json`` (deterministic components + cost) and one self-contained ``report.html``.

The HTML follows rLLM's single-file viewer (inline CSS, no build step, archivable; ``eval/visualizer.py``) and borrows
Harbor's per-trial waffle and reward–cost Pareto views. It adds what no framework has: the per-step Φ heatmap (task ×
normalized-step bins, per model) and the gate-profile heat table. The page holds no timestamp, so two runs with equal
deterministic components render byte-identical HTML unless cost differs.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from crucible.bench.metrics import Episode, deterministic_digest, heatmap

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#e5e5ea;--card:#f7f7f9}
@media (prefers-color-scheme:dark){:root{--bg:#111214;--fg:#ececf0;--mut:#9a9aa3;--line:#2a2b30;--card:#1a1b1f}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;margin:0;padding:24px 16px;max-width:1200px;margin:auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}.mut{color:var(--mut)}
table{border-collapse:collapse;width:100%;margin:6px 0;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}th{font-weight:600}
.scroll{overflow-x:auto}.cell{width:26px;height:18px;padding:0;border:1px solid var(--bg)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:8px 0}
.waffle span{display:inline-block;width:12px;height:12px;margin:1px;border-radius:2px}
"""


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return html.escape(str(x))


def _color(v: float | None) -> str:
    if v is None:
        return "background:transparent"
    v = max(0.0, min(1.0, float(v)))
    return f"background:hsl({int(10 + 120 * v)},65%,{int(62 - 18 * v)}%)"


def _pareto_svg(points: list[tuple[str, float, float]]) -> str:
    if not points:
        return ""
    w, h, pad = 520, 220, 36
    xmax = max(p[1] for p in points) or 1.0
    marks = []
    for name, cost, reward in points:
        x = pad + (w - 2 * pad) * (cost / xmax)
        y = h - pad - (h - 2 * pad) * reward
        marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="currentColor"/><text x="{x + 8:.1f}" y="{y + 4:.1f}" font-size="11">{html.escape(name)}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" role="img" aria-label="reward vs cost">'
            f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="currentColor" opacity=".3"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="currentColor" opacity=".3"/>'
            f'<text x="{w - pad}" y="{h - 8}" font-size="11" text-anchor="end">cost per episode</text>'
            f'<text x="8" y="{pad - 10}" font-size="11">mean reward</text>{"".join(marks)}</svg>')


def render(title: str, summary: dict[str, Any], eps: Sequence[Episode], *, extras: dict[str, Any] | None = None, bins: int = 10) -> str:
    extras = extras or {}
    hm = heatmap(eps, bins=bins)
    out = [(f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"),
           (f"<h1>{html.escape(title)}</h1><div class=mut>deterministic digest {deterministic_digest(summary)} · threshold {summary.get('threshold')}"
            " · masked samples are excluded from every quality metric</div>")]
    ks = summary.get("ks") or [1]
    out.append("<h2>Models</h2><div class=scroll><table><tr><th>policy</th><th>harness</th><th>measured / masked</th><th>mean reward (raw)</th><th>95% CI</th>"
               + "".join(f"<th>pass@{k}</th><th>pass^{k}</th>" for k in ks)
               + "<th>progress AUC</th><th>stage reach</th><th>regressions</th><th>msgs / success</th><th>USD / success</th></tr>")
    points = []
    for m in summary["models"]:
        d, c = m["overall"]["deterministic"], m["overall"]["cost"]
        cov = d.get("coverage", {})
        out.append(f"<tr><td>{_fmt(m['policy'])}</td><td>{_fmt(m['harness'])}</td><td>{cov.get('measured')} / {cov.get('masked')}</td>"
                   f"<td>{_fmt(d.get('mean_reward'))} ({_fmt(d.get('mean_reward_raw'))})</td><td>{_fmt(d.get('mean_reward_ci'))}</td>"
                   + "".join(f"<td>{_fmt(d.get(f'pass@{k}'))}</td><td>{_fmt(d.get(f'pass^{k}'))}</td>" for k in ks)
                   + f"<td>{_fmt(d.get('progress_auc'))}</td><td>{_fmt(d.get('stage_reach'))}</td><td>{_fmt(d.get('regressions'))}</td>"
                   f"<td>{_fmt(d.get('messages_per_success'))}</td><td>{_fmt((c.get('per_success') or {}).get('usd'))}</td></tr>")
        if d.get("mean_reward") is not None:
            points.append((m["policy"], c.get("usd", 0.0) / max(1, cov.get("episodes", 1)), d["mean_reward"]))
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
    out.append(f"<h2>Progress Φ per normalized step</h2><div class=mut>mean Φ at the end of each of {bins} step bins; one row per task</div>")
    for model, tasks in hm.items():
        out.append(f"<div class=card><b>{html.escape(model)}</b><div class=scroll><table><tr><th>task</th>"
                   + "".join(f"<th class=mut>{int((b + 1) * 100 / bins)}%</th>" for b in range(bins)) + "</tr>")
        for task, cells in tasks.items():
            out.append(f"<tr><td>{html.escape(task)}</td>" + "".join(f"<td class=cell style='{_color(v)}' title='{_fmt(v)}'></td>" for v in cells) + "</tr>")
        out.append("</table></div></div>")
    out.append("<h2>Episodes</h2><div class=mut>one cell per episode: colour = reward; grey = masked</div>")
    by_model: dict[str, list[Episode]] = {}
    for e in eps:
        by_model.setdefault(f"{e.policy}@{e.harness}", []).append(e)
    for model, es in sorted(by_model.items()):
        cells = "".join(f"<span title='{html.escape(e.task)}#{e.sample_idx}: {_fmt(e.reward)}' style='{_color(e.reward) if not e.masked else 'background:#8884'}'></span>"
                        for e in sorted(es, key=lambda e: (e.task, e.sample_idx)))
        out.append(f"<div class='card waffle'><b>{html.escape(model)}</b><br>{cells}</div>")
    if points:
        out.append(f"<h2>Reward vs cost</h2>{_pareto_svg(points)}")
    for name, val in extras.items():
        out.append(f"<h2>{html.escape(name)}</h2><div class=card><pre style='white-space:pre-wrap;margin:0'>{html.escape(json.dumps(val, indent=1, sort_keys=True, default=str))}</pre></div>")
    out.append("</body></html>")
    return "\n".join(out)


def write(out_dir: Path, title: str, summary: dict[str, Any], eps: Sequence[Episode], *, extras: dict[str, Any] | None = None) -> dict[str, str]:
    from crucible.bench.metrics import episodes_json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"title": title, "digest": deterministic_digest(summary), "summary": summary, **(extras or {})}
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    (out_dir / "episodes.json").write_text(json.dumps(episodes_json(eps), indent=1, sort_keys=True, default=str) + "\n")
    (out_dir / "report.html").write_text(render(title, summary, eps, extras=extras))
    return {"summary": str(out_dir / "summary.json"), "episodes": str(out_dir / "episodes.json"), "report": str(out_dir / "report.html")}
