"""The training monitor in W&B and on disk (swarm-pipeline/11 §4.4). Plain rows in (``factory.train_monitor.build_report``'s
dict), so this module needs nothing above ``observe``:

- ``log_step(run, row, step)``: one trend row (mode prevalence and health per step) as scalars, every training step;
- ``log_report(run, report)``: the general-vs-task-specific table, the trend lines, alerts and proposals as tables;
- ``markdown(report)`` / ``html(report)``: the local per-run report (written under ``crucible/data``, gitignored).

Held-out tasks never reach this module by name: the report carries them as placeholders and aggregates.
"""

from __future__ import annotations

import html as _html
import json
from collections.abc import Mapping, Sequence
from typing import Any

from crucible.observe.run import ObsRun


def _f(x: Any, d: int = 2) -> str:
    return "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def log_step(run: ObsRun, row: Mapping[str, Any], step: int, *, key: str = "monitor") -> None:
    run.log({f"{key}/{k}": v for k, v in row.items() if isinstance(v, int | float) and not isinstance(v, bool)}, step=step)


def log_report(run: ObsRun, report: Mapping[str, Any], *, key: str = "monitor", step: int | None = None) -> None:
    modes = (report.get("general") or {}).get("modes") or []
    run.table(f"{key}/modes", ["mode", "general", "k", "n", "p", "lo", "hi", "tasks", "langs", "lanes"],
              [[r["mode"], r["general"], r["k"], r["n"], r["p"], r["lo"], r["hi"], len(r["tasks"]), ",".join(r["langs"]), ",".join(r["lanes"])]
               for r in modes], step=step)
    rows = (report.get("trends") or {}).get("rows") or []
    if len(rows) >= 2:
        names = [r["mode"] for r in modes if r["general"]] or [r["mode"] for r in modes[:4]]
        run.lines(f"{key}/mode_lines", [float(r["order"]) for r in rows], [[r.get(f"mode/{m}") for r in rows] for m in names], names,
                  title="general failure modes over training", xname="step")
        run.lines(f"{key}/health_lines", [float(r["order"]) for r in rows], [[r.get(k) for r in rows] for k in ("reward", "full_pass", "phi", "hack")],
                  ["reward", "full_pass", "phi", "hack"], title="health over training", xname="step")
    run.table(f"{key}/alerts", ["kind", "severity", "detail"], [[a["kind"], a["severity"], a["detail"]] for a in report.get("alerts") or []], step=step)
    run.table(f"{key}/proposals", ["mode", "lever", "text", "p", "tasks", "trend"],
              [[p["mode"], p["lever"], p["text"], p["evidence"]["p"], p["evidence"]["tasks"], p.get("trend")] for p in report.get("proposals") or []],
              step=step)
    run.summary({f"{key}/general_modes": len((report.get("general") or {}).get("general") or []), f"{key}/alerts": len(report.get("alerts") or []),
                 f"{key}/episodes": (report.get("episodes") or {}).get("train")})


def markdown(report: Mapping[str, Any], *, title: str = "Training monitor") -> str:
    ep = report.get("episodes") or {}
    h = report.get("health") or {}
    g = report.get("general") or {}
    out = [f"# {title}", "", f"Episodes: {ep.get('train')} training ({ep.get('masked')} masked overall), {ep.get('heldout')} held out (aggregate only).",
           f"By lane {json.dumps(ep.get('by_lane'))}; by language {json.dumps(ep.get('by_lang'))}.", "",
           (f"Health: reward {_f(h.get('reward'))}, full pass {_f(h.get('full_pass'))}, Φ {_f(h.get('phi'))}, budget stops {_f(h.get('budget_stops'))}, "
           f"hacks {_f(h.get('hack'))}, path violations {_f(h.get('path_violation'))}, tool calls {_f(h.get('tool_calls'), 0)}, Brier {_f(h.get('brier'))}."),
           "", f"## General failures (>= {g.get('min_tasks')} tasks and >= 2 languages or lanes)", "",
           "| mode | episodes | prevalence (95% CI) | tasks | languages | trend |", "|---|---|---|---|---|---|"]
    verdicts = (report.get("trends") or {}).get("verdicts") or {}
    for r in g.get("modes") or []:
        if r["general"]:
            out.append(f"| {r['mode']} | {r['k']}/{r['n']} | {_f(r['p'])} ({_f(r['lo'])}–{_f(r['hi'])}) | {len(r['tasks'])} | {', '.join(r['langs'])} | "
                       f"{verdicts.get(r['mode'], {}).get('verdict', '–')} |")
    out += ["", "## Task-specific", "", "Modes seen in too few tasks or languages to be general:", ""]
    for r in g.get("modes") or []:
        if not r["general"] and r["k"]:
            out.append(f"- {r['mode']}: {r['k']}/{r['n']} in {len(r['tasks'])} task(s) ({', '.join(r['langs'])})")
    out += ["", "Milestones a task's failures keep missing:", ""]
    for t in g.get("task_specific") or []:
        if t["stuck_milestones"]:
            out.append(f"- {t['task']}: {', '.join(t['stuck_milestones'])} ({t['failed']} failed episodes; causes {json.dumps(t['causes'])})")
    pr = report.get("path_rules") or {}
    if pr.get("repathed"):
        out += ["", (f"Path rules: of {pr['repathed']} re-pathed episodes, path:install failed {pr['install_fail_stored']} times under the stored rules "
                f"and {pr['install_fail_now']} under the current ones.")]
    al = [f"- **{a['kind']}** ({a['severity']}): {a['detail']}" for a in report.get("alerts") or []]
    out += ["", "## Alerts", ""] + (al or ["- none"])
    out += ["", "## Proposals (never applied; each needs executed evidence)", ""]
    for p in report.get("proposals") or []:
        out.append(f"- [{p['lever']}] {p['mode']}: {p['text']} (p {_f(p['evidence']['p'])}, {p['evidence']['tasks']} tasks)")
    bm = report.get("by_model") or {}
    if bm:
        gm = (report.get("general") or {}).get("general") or []
        out += ["", "## By policy model (training split)", "", "| model | episodes | full pass | " + " | ".join(gm) + " |", "|---|---|---|" + "---|" * len(gm)]
        for m, v in bm.items():
            out.append(f"| {m} | {v['episodes']} | {_f(v['health'].get('full_pass'))} | " + " | ".join(_f(v["modes"].get(k)) for k in gm) + " |")
    ho = report.get("heldout") or {}
    if ho.get("episodes"):
        out += ["", "## Held out (aggregate)", "", f"{ho['episodes']} episodes over {len(ho['tasks'])} tasks."]
        for c, v in (ho.get("cohorts") or {}).items():
            top = sorted(((m, p) for m, p in v["modes"].items() if p), key=lambda x: -x[1])[:4]
            out.append(f"- {c}: full pass {_f(v['health'].get('full_pass'))}; top modes {', '.join(f'{m} {p:.2f}' for m, p in top)}")
    j = report.get("judge") or {}
    if j:
        out += ["", f"Judge pass: {json.dumps(j)}"]
    return "\n".join(out) + "\n"


def html(report: Mapping[str, Any], *, title: str = "Training monitor") -> str:
    body = _html.escape(markdown(report, title=title))
    return (f"<!doctype html><meta charset='utf-8'><title>{_html.escape(title)}</title>"
            f"<pre style='font:14px/1.5 ui-monospace,monospace;white-space:pre-wrap;max-width:1000px;margin:24px auto'>{body}</pre>")


def table_rows(report: Mapping[str, Any]) -> Sequence[Sequence[Any]]:
    return [[r["mode"], r["general"], r["p"], r["lo"], r["hi"]] for r in (report.get("general") or {}).get("modes") or []]
