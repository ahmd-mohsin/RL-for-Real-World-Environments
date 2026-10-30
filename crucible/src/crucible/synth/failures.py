"""Why proposals fail: mining the run records of the synthesis pipeline by stage and reason code (E6, yield work).

``mine_run(run_dir)`` reads one ``pipeline.Runner`` run directory (``report.json``, ``tasks/*/cegis.json``,
``panel.jsonl``, ``usage.jsonl``) and returns one row per task:

- **stage reached** and **final status** (``TaskRecord``);
- the **primary reason** of a failure: the first entry of ``PRIMARY_ORDER`` the task's reasons carry, so every
  rejected task is counted once (a confirmed deterministic counterexample outranks an ALT failure, which outranks an open
  suspect, which outranks the MUT bound);
- for each failed ALT attempt the panel judged, a **failure mode** from the panel's written reasons
  (``alt_mode``): ``no_action`` (an empty trace), ``no_write`` (reads only; the decision never applied), ``not_found``
  (the solver could not locate the entity or the document), ``skipped_rule`` (acted without checking a rule the task
  needs), ``wrong_decision`` (a different decision or reason code), ``incomplete`` (part of a multi-entity task),
  ``split`` (the panel members disagreed: an ambiguity signal), ``valid`` (the panel judged the attempt valid although
  the grader failed it: an over-strict contract, D6), ``other``;
- with an injected **truth** function (a domain with executed ground truth, e.g. the finance_ap pack's own policy
  decisions), whether the task's **oracle** is right: ``right``, ``wrong`` or ``unknown``. This separates "the solver
  misapplied the rule" from "the oracle misapplied it and the solver was right".

``summarise(rows)`` aggregates (funnel, primary reasons, ALT modes per solver family and per oracle truth, cost per
stage from ``usage.jsonl``); ``markdown`` renders it. Nothing here calls a model.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# a rejected task is counted under the first of these its reasons carry (``<head>`` = the text before ``=``)
PRIMARY_ORDER: tuple[str, ...] = (
    "oracle:dead", "error", "budget",
    "REP:confirmed", "RED:confirmed", "TAINT:confirmed", "NULL:confirmed", "MUT:confirmed", "TWIN:confirmed", "PERT:confirmed",
    "archive:reopened", "ALT:confirmed", "ATK:confirmed", "ALT:no_pass", "ATK:suspects", "ALT:suspects", "MUT:suspects", "TWIN:suspects",
    "MUT:upper",
)

ALT_MODES = ("no_action", "no_write", "not_found", "skipped_rule", "wrong_decision", "incomplete", "split", "valid", "other")

_MODE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("no_action", re.compile(r"trace (is|was) empty|no actions? (were |was )?(taken|performed)|performed no", re.IGNORECASE)),
    # a different write than the one judged right ("approved instead of rejecting") is a decision, not a missing write
    ("wrong_decision", re.compile(r"instead of|rather than|incorrectly (applied|approved|held|put|placed|used|routed|set|escalated)|"
                                  r"wrong reason|used the (incorrect|wrong)|with (the )?(reason|priority) '[^']+' (instead|but)", re.IGNORECASE)),
    ("not_found", re.compile(r"could not (find|locate)|failed to (find|locate)|returned no (matches|results)|found no(ne| results)|"
                             r"empty invoices|did not find|no results", re.IGNORECASE)),
    ("no_write", re.compile(r"never (makes?|made|performs?|performed|executes?|executed|calls?|called|applies|applied|writes?|"
                            r"records?)[^.]{0,40}(write|update|action|change|decision|call)|only (read|reads|shows|performed|lists|listed|"
                            r"retriev)|no (write|action|update) (call|was|to)|without (actually )?(calling|making|performing)|"
                            r"not (carried out|performed|executed|completed)|made no", re.IGNORECASE)),
    ("incomplete", re.compile(r"only (processed|escalated|put|handled|decided) |did not (escalate|process|decide|handle) |missed |"
                              r"the other |not .*every|incomplete", re.IGNORECASE)),
    ("skipped_rule", re.compile(r"without (first )?(checking|verifying|performing|searching|retrieving|comparing|reading|consulting)|"
                                r"failed to (check|verify|normalize|apply)|did not (check|verify|search)|no (duplicate|receipt|"
                                r"vendor)[^.]{0,20}check", re.IGNORECASE)),
    ("wrong_decision", re.compile(r"incorrect(ly)?|wrong|not the required|violat|contradict|should have|requires? "
                                  r"(rejecting|holding|approving|a hold|putting)", re.IGNORECASE)),
)


def head(reason: str) -> str:
    return reason.split("=", 1)[0].strip()


def primary_reason(status: str, reasons: Sequence[str]) -> str:
    """The one reason a failed task is counted under (module docstring); ``certified`` for an admitted task."""
    if status == "certified":
        return "certified"
    heads = [head(r) for r in reasons]
    for k in PRIMARY_ORDER:
        if k in heads:
            return k
    return heads[0] if heads else status


def alt_mode(row: Mapping[str, Any]) -> str:
    """The failure mode of one adjudicated failed ALT attempt (a ``panel.jsonl`` row; module docstring)."""
    votes = [v for v in row.get("votes") or [] if v.get("valid") is not None]
    if row.get("decision") == "valid":
        return "valid"
    if len({v["valid"] for v in votes}) > 1:
        return "split"
    text = " ".join(v.get("why") or "" for v in votes if v.get("valid") is False)
    for mode, pat in _MODE_RULES:
        if pat.search(text):
            return mode
    return "other"


def _jl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _site_family(site: str) -> str:
    # "agent:alt_solver-nemotron" (M4 naming) or "agent:alt_solver-nemotron-ultra-0" (unique names since the yield work)
    s = site.split(":", 1)[-1]
    return s.split("-", 1)[1] if "-" in s else s


def mine_run(run_dir: str | Path, *, truth: Callable[[Mapping[str, Any]], str | None] | None = None) -> list[dict[str, Any]]:
    """One row per task of a run directory (module docstring). ``truth(task_json)`` returns ``right``/``wrong``/None."""
    d = Path(run_dir)
    rep = json.loads((d / "report.json").read_text()) if (d / "report.json").exists() else {"tasks": []}
    panel = _jl(d / "panel.jsonl")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in panel:
        by_task[r.get("task", "")].append(r)
    rows = []
    for t in rep.get("tasks") or []:
        n = t.get("n")
        cj = d / "tasks" / f"t{n:02d}" / "cegis.json"
        task = json.loads(cj.read_text()).get("task") if cj.exists() else None
        alts = [r for r in by_task.get(t["id"], []) if r.get("cls") == "ALT"]
        modes = [{"family": _site_family(r.get("site", "")), "mode": alt_mode(r), "decision": r.get("decision")} for r in alts]
        alt = (t.get("classes") or {}).get("ALT") or {}
        row = {"run": d.name, "domain": rep.get("domain"), "task": t["id"], "status": t.get("status"), "stage": t.get("stage"),
               "reasons": list(t.get("reasons") or []), "primary": primary_reason(t.get("status", ""), t.get("reasons") or []),
               "first_round_confirmed": dict(t.get("first_round_confirmed") or {}), "rounds": t.get("rounds", 0),
               "repairs": t.get("repairs_accepted", 0), "alt_n": alt.get("n"), "alt_passes": alt.get("passes"), "alt_modes": modes,
               "oracle_steps": len((task or {}).get("oracle", {}).get("steps") or []), "oracle_writes": t.get("oracle_writes", 0),
               "oracle_truth": None}
        if truth is not None and task is not None:
            try:
                row["oracle_truth"] = truth(task)
            except Exception as e:  # noqa: BLE001 - an unreadable task is "unknown", never a guess
                row["oracle_truth"] = f"unknown: {type(e).__name__}"
        rows.append(row)
    return rows


def stage_cost(run_dirs: Iterable[str | Path]) -> dict[str, float]:
    out: Counter[str] = Counter()
    for d in run_dirs:
        for r in _jl(Path(d) / "usage.jsonl"):
            out[r.get("stage") or "?"] += float(r.get("usd") or 0.0)
    return {k: round(v, 5) for k, v in sorted(out.items())}


def summarise(rows: Sequence[Mapping[str, Any]], *, cost: Mapping[str, float] | None = None) -> dict[str, Any]:
    st = Counter(r["status"] for r in rows)
    order = ["proposed", "oracle", "contract", "refined"]
    reached: Counter[str] = Counter()
    for r in rows:
        stg = r.get("stage") or "proposed"
        for s in order[: order.index(stg) + 1 if stg in order else 1]:
            reached[s] += 1
    prim = Counter(r["primary"] for r in rows if r["status"] != "certified")
    modes: Counter[str] = Counter()
    by_fam: dict[str, Counter[str]] = defaultdict(Counter)
    by_truth: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        for m in r["alt_modes"]:
            modes[m["mode"]] += 1
            by_fam[m["family"]][m["mode"]] += 1
            by_truth[str(r.get("oracle_truth"))][m["mode"]] += 1
    truth = Counter((str(r.get("oracle_truth")), r["status"], bool(r.get("alt_passes"))) for r in rows if r.get("oracle_truth") is not None)
    alt_fail_tasks = [r for r in rows if r["primary"] == "ALT:no_pass"]
    return {"tasks": len(rows), "status": dict(st), "reached": dict(reached), "primary": dict(prim.most_common()),
            "first_round_confirmed_tasks": dict(Counter(k for r in rows for k, v in r["first_round_confirmed"].items() if v).most_common()),
            "alt_modes": dict(modes.most_common()), "alt_modes_by_family": {k: dict(v.most_common()) for k, v in sorted(by_fam.items())},
            "alt_modes_by_oracle_truth": {k: dict(v.most_common()) for k, v in sorted(by_truth.items())},
            "oracle_truth": {"/".join(map(str, k)): v for k, v in sorted(truth.items())},
            "alt_no_pass_tasks": len(alt_fail_tasks),
            "alt_no_pass_by_truth": dict(Counter(str(r.get("oracle_truth")) for r in alt_fail_tasks)),
            "cost": dict(cost or {})}


def markdown(s: Mapping[str, Any], title: str = "Failure analysis") -> str:
    lines = [f"## {title}", "", f"{s['tasks']} tasks. Status: " + ", ".join(f"{k} {v}" for k, v in sorted(s["status"].items())), "",
             "| primary reason (one per failed task) | tasks |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in s["primary"].items()]
    lines += ["", "| failed ALT attempt: mode | attempts |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s["alt_modes"].items()]
    if s.get("alt_modes_by_family"):
        fams = sorted(s["alt_modes_by_family"])
        lines += ["", "| mode | " + " | ".join(fams) + " |", "|---|" + "---|" * len(fams)]
        for m in ALT_MODES:
            if any(s["alt_modes_by_family"][f].get(m) for f in fams):
                lines.append(f"| {m} | " + " | ".join(str(s["alt_modes_by_family"][f].get(m, 0)) for f in fams) + " |")
    if s.get("oracle_truth"):
        lines += ["", "| oracle truth / status / ALT passed | tasks |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s["oracle_truth"].items()]
        bt = {k: v for k, v in (s.get("alt_modes_by_oracle_truth") or {}).items() if k != "None"}
        if bt:
            ks = sorted(bt)
            lines += ["", "| failed ALT attempt mode | " + " | ".join(f"oracle {k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
            for m in ALT_MODES:
                if any(bt[k].get(m) for k in ks):
                    lines.append(f"| {m} | " + " | ".join(str(bt[k].get(m, 0)) for k in ks) + " |")
    if s.get("cost"):
        lines += ["", "Cost by stage (USD): " + ", ".join(f"{k} {v:.3f}" for k, v in s["cost"].items())]
    return "\n".join(lines) + "\n"


__all__ = ["ALT_MODES", "PRIMARY_ORDER", "alt_mode", "head", "markdown", "mine_run", "primary_reason", "stage_cost", "summarise"]
