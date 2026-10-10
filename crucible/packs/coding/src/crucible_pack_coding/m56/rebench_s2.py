"""M6 gate S2 on SWE-rebench (held-out, later issues): does Φ at the midpoint predict ``resolved``?

Protocol (DESIGN §6.5, made concrete):
- **time split**: issues sorted by ``created_at``; milestone-type weights are fitted on the earliest 70% only, every
  number reported is on the latest 30%;
- **weights**: per milestone *type* (located, removed_call, added_call, ...), weight = max(0, 2·(AUC − 0.5)) of that
  type's mean φ at the midpoint on the fit split (the §6.5 rule); ``uniform`` keeps weight 1 for every type;
- **metrics**: pooled AUC over all held-out attempts, mean within-issue AUC over issues with mixed outcomes (controls
  for task difficulty), group-bootstrap 95% CI by issue; baselines that need no milestones (fewer tool calls; edited
  any file by the midpoint) and, when available, the LLM-judge baseline (``rebench_judge.py``), compared with a
  paired, issue-grouped bootstrap;
- attempts whose replay failed (quarantined, M5) are excluded and counted.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from crucible.credit.validity import auc, bootstrap_ci, paired_bootstrap, time_split

FRAC = "0.5"


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def type_scores(row: dict[str, Any], frac: str) -> dict[str, float]:
    phis = row["phi"].get(frac) or {}
    by: dict[str, list[float]] = defaultdict(list)
    for mid, v in phis.items():
        by[row["milestone_types"].get(mid, "other")].append(v or 0.0)
    return {t: statistics.fmean(v) for t, v in by.items()}


def phi(row: dict[str, Any], frac: str, w: dict[str, float] | None) -> float:
    ts = type_scores(row, frac)
    if not ts:
        return 0.0
    if w is None:
        phis = row["phi"].get(frac) or {}
        return statistics.fmean([v or 0.0 for v in phis.values()]) if phis else 0.0
    den = sum(w.get(t, 0.0) for t in ts)
    return sum(w.get(t, 0.0) * s for t, s in ts.items()) / den if den else 0.0


def within(rows: list[dict[str, Any]], score) -> tuple[float | None, int]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["instance_id"]].append(r)
    aucs = [a for rs in by.values() if (a := auc([score(r) for r in rs], [r["resolved"] for r in rs])) is not None]
    return (statistics.fmean(aucs) if aucs else None), len(aucs)


def analyse(rows: list[dict[str, Any]], judge: dict[str, float] | None = None, holdout: float = 0.3,
            judge_end: dict[str, float] | None = None) -> dict[str, Any]:
    usable = [r for r in rows if r["replay_agent_state"] and r["phi"]]
    quarantined = sum(1 for r in rows if not r["replay_agent_state"])
    no_milestones = sum(1 for r in rows if r["replay_agent_state"] and not r["phi"])
    created = {r["instance_id"]: r["created_at"] for r in usable}
    fit_inst, hold_inst = time_split(sorted(created), time=lambda i: created[i], holdout=holdout)
    fit = [r for r in usable if r["instance_id"] in set(fit_inst)]
    hold = [r for r in usable if r["instance_id"] in set(hold_inst)]
    types = sorted({t for r in fit for t in r["milestone_types"].values()})
    weights, type_auc = {}, {}
    for t in types:
        xs = [(type_scores(r, FRAC).get(t), r["resolved"]) for r in fit]
        xs = [(s, y) for s, y in xs if s is not None]
        a = auc([s for s, _ in xs], [y for _, y in xs]) if xs else None
        type_auc[t] = a
        weights[t] = max(0.0, 2 * (a - 0.5)) if a is not None else 0.0
    y = [r["resolved"] for r in hold]
    groups = [r["instance_id"] for r in hold]

    def report(name: str, scores: list[float], fn) -> dict[str, Any]:
        w_auc, n_w = within(hold, fn)
        lo, hi = bootstrap_ci(scores, y, groups=groups, n=500)
        return {"name": name, "auc": None if (a := auc(scores, y)) is None else round(a, 4), "ci": [lo and round(lo, 4), hi and round(hi, 4)],
                "within_issue_auc": w_auc and round(w_auc, 4), "issues_with_mixed_outcomes": n_w}

    candidates = {
        "phi_mid_validity_weighted": lambda r: phi(r, FRAC, weights),
        "phi_mid_uniform": lambda r: phi(r, FRAC, None),
        "phi_end_validity_weighted": lambda r: phi(r, "1.0", weights),
        "phi_q1_validity_weighted": lambda r: phi(r, "0.25", weights),
        "baseline_fewer_tool_calls": lambda r: -r["n_calls"],
        "baseline_any_milestone_by_mid": lambda r: float(any((v or 0) > 0 for v in (r["phi"].get(FRAC) or {}).values())),
    }
    out = {"rows": len(rows), "usable": len(usable), "quarantined_by_replay": quarantined, "no_milestones": no_milestones,
           "fit_issues": len(fit_inst), "holdout_issues": len(hold_inst), "holdout_attempts": len(hold), "holdout_resolved_rate": round(statistics.fmean(y), 4) if y else None,
           "type_weights": {t: round(w, 4) for t, w in weights.items()}, "type_auc_fit": {t: a and round(a, 4) for t, a in type_auc.items()},
           "metrics": {k: report(k, [fn(r) for r in hold], fn) for k, fn in candidates.items()}}
    if judge:
        jh = [r for r in hold if r["trajectory_id"] in judge]
        if jh:
            yj, gj = [r["resolved"] for r in jh], [r["instance_id"] for r in jh]
            pj = [phi(r, FRAC, weights) for r in jh]
            sj = [judge[r["trajectory_id"]] for r in jh]
            out["judge"] = {"attempts": len(jh), "judge_auc": auc(sj, yj), "phi_auc_same_attempts": auc(pj, yj),
                            "judge_within_issue_auc": within(jh, lambda r: judge[r["trajectory_id"]])[0],
                            "phi_minus_judge": paired_bootstrap(pj, sj, yj, groups=gj, n=1000)}
    if judge_end:
        jh = [r for r in hold if r["trajectory_id"] in judge_end]
        if jh:
            yj, gj = [r["resolved"] for r in jh], [r["instance_id"] for r in jh]
            pe = [phi(r, "1.0", None) for r in jh]
            se = [judge_end[r["trajectory_id"]] for r in jh]
            out["judge_end"] = {"attempts": len(jh), "judge_auc": auc(se, yj), "judge_within_issue_auc": within(jh, lambda r: judge_end[r["trajectory_id"]])[0],
                                "phi_end_auc_same_attempts": auc(pe, yj), "phi_end_within_issue_auc": within(jh, lambda r: phi(r, "1.0", None))[0],
                                "phi_end_minus_judge_end": paired_bootstrap(pe, se, yj, groups=gj, n=1000),
                                "combined_auc": auc([0.5 * a + 0.5 * b for a, b in zip(pe, se, strict=True)], yj),
                                "combined_within_issue_auc": within(jh, lambda r: 0.5 * phi(r, "1.0", None) + 0.5 * judge_end[r["trajectory_id"]])[0]}
    return out


if __name__ == "__main__":
    rows = load(Path(sys.argv[1]))
    judge = None
    if len(sys.argv) > 2 and Path(sys.argv[2]).exists():
        judge = {j["trajectory_id"]: j["p_success"] for j in load(Path(sys.argv[2])) if j.get("p_success") is not None}
    judge_end = None
    if len(sys.argv) > 3 and Path(sys.argv[3]).exists():
        judge_end = {j["trajectory_id"]: j["p_success"] for j in load(Path(sys.argv[3])) if j.get("p_success") is not None}
    print(json.dumps(analyse(rows, judge, judge_end=judge_end), indent=1))
