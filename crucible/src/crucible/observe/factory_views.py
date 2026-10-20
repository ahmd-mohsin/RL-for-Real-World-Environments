"""Task-factory views in W&B (swarm-pipeline/11 §4.4): the funnel per lane, targeting, intent and verifier cards, the trace
audit, failure findings and spend. Each function takes an ``ObsRun`` (project ``crucible-factory``) and plain dicts as the
factory writes them (``Funnel.summary``, ``windows.funnel``, ``targeting.band`` / ``cause_mix`` / ``target_verdict``,
``cards.IntentCard`` / ``failures.BuilderIntent``, ``cards.VerifierCard`` or P2's per-family report, ``FailureFinding``),
so the factory imports nothing new and a report on disk can be backfilled the same way.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from crucible.observe import charts
from crucible.observe.bench import wilson
from crucible.observe.run import ObsRun
from crucible.observe.traces import TraceStore


def _dump(x: Any) -> Any:
    return x.model_dump(mode="json") if hasattr(x, "model_dump") else (x.to_json() if hasattr(x, "to_json") else x)


def funnel(run: ObsRun, lanes: Mapping[str, Mapping[str, Any]], *, key: str = "funnel") -> list[list[Any]]:
    """``lanes``: lane -> stage -> a count (``windows.funnel()["total"]``: candidates reaching the stage) or ``{"in", "pass",
    "reasons"}`` (``Funnel.summary()``). Stages keep their order. One table for all lanes, a bar chart per lane."""
    rows: list[list[Any]] = []
    for lane, stages in lanes.items():
        labels, vals = [], []
        for i, (stage, v) in enumerate(stages.items()):
            if isinstance(v, Mapping):
                n_in, n_pass = int(v.get("in") or 0), int(v.get("pass") or 0)
                reasons = dict(v.get("reasons") or {})
            elif isinstance(v, int | float) and not isinstance(v, bool):
                n_in, n_pass, reasons = int(v), int(v), {}
            else:
                continue
            rows.append([lane, i, stage, n_in, n_pass, n_in - n_pass, json.dumps(reasons, sort_keys=True) if reasons else None])
            labels.append(f"{i:02d} {stage}")
            vals.append(float(n_pass))
        if labels:
            run.bar(f"{key}/{lane}", labels, vals, title=f"funnel: {lane}", label="stage", value="reached")
    run.table(f"{key}/table", ["lane", "order", "stage", "in", "passed", "dropped", "reasons"], rows)
    return rows


def targeting(run: ObsRun, records: Mapping[str, Mapping[str, Any]], *, model: str | None = None, key: str = "targeting") -> list[list[Any]]:
    """Per task: the target model's pass rate and band, the cause mix, the counterfactual, the prerequisite probe (the
    fairness solver), and the S15 verdict with its route."""
    rows, mix = [], []
    vals = {}
    for task, r in sorted(records.items()):
        b = r.get("band") or {}
        c = r.get("cause") or {}
        v = r.get("verdict") or {}
        pr = r.get("probe") or {}
        cf = r.get("counterfactual") or {}
        rng = b.get("reward_range") or [None, None]
        fair = r.get("fairness")
        rows.append([task, r.get("model") or model, b.get("n"), b.get("pass_rate"), b.get("in_band"), rng[0], rng[1], c.get("skill_gap_share"), c.get("n"),
                     c.get("ok"), cf.get("verdict") or v.get("counterfactual"), pr.get("ok"), json.dumps(fair, default=str) if fair is not None else None,
                     v.get("ok"), v.get("route"), v.get("owner"), "; ".join(v.get("findings") or [])[:500] or None, "; ".join(v.get("flags") or [])[:500] or None])
        for lab, n in (c.get("labels") or {}).items():
            mix.append([task, lab, int(n)])
        if b.get("pass_rate") is not None:
            vals[(task, "in band" if b.get("in_band") else "out of band")] = (float(b["pass_rate"]), None, None)
    run.table(f"{key}/table", ["task", "model", "n", "pass_rate", "in_band", "reward_min", "reward_max", "skill_gap_share", "labels", "cause_ok",
                               "counterfactual", "probe_ok", "fairness", "verdict_ok", "route", "owner", "findings", "flags"], rows)
    run.table(f"{key}/cause_mix", ["task", "label", "count"], mix)
    if vals:
        run.html(f"{key}/band_chart", charts.grouped_bars_ci(sorted(records), ["in band", "out of band"], vals, title=f"target pass rate per task ({model or 'model'})",
                                                              subtitle="in band: the group's rewards have spread (the task can teach)"))
    run.summary({f"{key}/tasks": len(records), f"{key}/in_band": sum(1 for r in records.values() if (r.get("band") or {}).get("in_band")),
                 f"{key}/verdict_ok": sum(1 for r in records.values() if (r.get("verdict") or {}).get("ok"))})
    return rows


def intent_cards(run: ObsRun, cards: Sequence[Any], *, key: str = "intents") -> list[list[Any]]:
    """``IntentCard`` rows (or ``BuilderIntent`` candidates: their ``intent_card`` fields plus student and mode)."""
    rows = []
    for c in cards:
        d = _dump(c)
        rows.append([d.get("id"), d.get("status", "candidate"), d.get("student"), d.get("mode"), d.get("behaviour"), d.get("failure_mode"),
                     len(d.get("evidence") or []), json.dumps(d.get("difficulty_axes") or d.get("axes") or []), json.dumps(d.get("coverage") or {}),
                     d.get("approved_by")])
    run.table(f"{key}/table", ["id", "status", "student", "mode", "behaviour", "failure_mode", "evidence", "difficulty_axes", "coverage", "approved_by"], rows)
    return rows


def _rate(pair: Any) -> tuple[int | None, int | None, float | None, float | None]:
    if isinstance(pair, list | tuple) and len(pair) == 2 and pair[1]:
        k, n = int(pair[0]), int(pair[1])
        return k, n, k / n, wilson(k, n)[1]
    return None, None, None, None


def verifier_cards(run: ObsRun, families: Mapping[str, Any], *, key: str = "verifier_cards") -> list[list[Any]]:
    """Per family: false-pass and false-fail (errors of n, rate, Wilson upper bound), mutant kill, alternatives passing,
    flake bound, κ, confirmed hacks. ``families[f]`` is a ``VerifierCard`` (``fp``/``fn`` pairs) or P2's per-family entry
    (``card`` plus measured ``bounds``)."""
    rows = []
    for fam, x in sorted(families.items()):
        d = _dump(x)
        card = d.get("card") or d
        b = d.get("bounds") or {}
        fp_b, fn_b = b.get("truly_failing") or {}, b.get("truly_passing") or {}
        fp = (fp_b.get("errors"), fp_b.get("n"), fp_b.get("rate"), fp_b.get("wilson_upper")) if fp_b else _rate(card.get("fp"))
        fn = (fn_b.get("errors"), fn_b.get("n"), fn_b.get("rate"), fn_b.get("wilson_upper")) if fn_b else _rate(card.get("fn"))
        mk, alt = _rate(card.get("mutant_kill")), _rate(card.get("alternatives_pass"))
        rows.append([fam, card.get("verifier_version"), *fp, *fn, mk[2], mk[1], alt[2], alt[1], card.get("flake_upper"), card.get("kappa"),
                     card.get("hacks_confirmed"), d.get("windows")])
    run.table(f"{key}/table", ["family", "verifier_version", "false_pass_errors", "false_pass_n", "false_pass_rate", "false_pass_upper", "false_fail_errors",
                               "false_fail_n", "false_fail_rate", "false_fail_upper", "mutant_kill", "mutants", "alternatives_pass", "alternatives",
                               "flake_upper", "kappa", "hacks_confirmed", "windows"], rows)
    if rows:
        fams = [r[0] for r in rows]
        vals = {**{(r[0], "false pass"): (r[4], None, r[5]) for r in rows if r[4] is not None},
                **{(r[0], "false fail"): (r[8], None, r[9]) for r in rows if r[8] is not None}}
        run.html(f"{key}/chart", charts.grouped_bars_ci(fams, ["false pass", "false fail"], {k: (v[0], v[0], v[2]) for k, v in vals.items()},
                                                        title="verifier error rates per family", subtitle="rate on the gold set; whisker to the Wilson 95% upper bound",
                                                        ymax=max([0.1, *(v[2] or 0 for v in vals.values())])))
    return rows


def spend(run: ObsRun, by_phase: Mapping[str, float], *, key: str = "spend") -> float:
    """USD per phase or role (``spend_by_role``); the total goes to the summary."""
    items = [(k, float(v)) for k, v in by_phase.items() if isinstance(v, int | float) and not isinstance(v, bool)]
    run.table(f"{key}/table", ["phase", "usd"], [[k, round(v, 6)] for k, v in items])
    if items:
        run.bar(f"{key}/bar", [k for k, _ in items], [v for _, v in items], title="spend per phase (USD)", label="phase", value="usd")
    total = sum(v for _, v in items)
    run.summary({f"{key}/total_usd": round(total, 6)})
    return total


def audit_summary(run: ObsRun, report: Mapping[str, Any], *, key: str = "audit") -> None:
    """The trace audit's report (``traceaudit.run_loop`` / P3 report): facts, first wrong step kinds, causes, the auditor."""
    run.table(f"{key}/facts", ["fact", "episodes"], [[k, v] for k, v in sorted((report.get("facts_episodes") or report.get("facts") or {}).items())])
    fws = report.get("first_wrong_step") or {}
    run.table(f"{key}/first_wrong_step", ["kind", "episodes"], [[k, v] for k, v in sorted((fws.get("kinds") or {}).items())])
    causes = report.get("causes") or {}
    run.table(f"{key}/causes", ["source", "cause", "count"], [[src, c, n] for src in ("deterministic", "labelled_trusted")
                                                               for c, n in sorted((causes.get(src) or {}).items())])
    a = report.get("auditor") or {}
    run.summary({f"{key}/bundles": report.get("bundles"), f"{key}/failing": report.get("failing"), f"{key}/passed": report.get("passed"),
                 f"{key}/masked": report.get("masked"), f"{key}/skill_gap_share": causes.get("skill_gap_share"), f"{key}/auditor": a.get("model"),
                 f"{key}/trusted_labels": a.get("trusted_labels"), f"{key}/auditor_usd": a.get("session_usd")})


FINDING_COLUMNS = ["id", "episode", "task", "student", "mode", "cause", "cause_source", "routed", "first_wrong_kind", "first_wrong_step", "first_wrong_turn",
                   "quote", "milestones_missed", "milestones_lost", "phi_final", "stop", "trace"]


def findings(run: ObsRun, items: Sequence[Any], *, store: TraceStore | None = None, key: str = "findings") -> list[list[Any]]:
    """Failure findings (``failures.FailureFinding``) as a table; with ``store`` each finding is also attached to its
    episode's Weave trace (``TraceStore.add_audit``) and the row carries the trace link."""
    rows = []
    for f in items:
        d = _dump(f)
        url = None
        if store is not None:
            store.add_audit(str(d.get("trace")), d)
            url = (store.index().get(str(d.get("trace"))) or {}).get("url")
        fws = d.get("first_wrong_step") or {}
        rows.append([d.get("id"), d.get("trace"), d.get("task"), d.get("student"), d.get("mode"), d.get("cause"), d.get("cause_source"), d.get("routed"),
                     fws.get("kind"), fws.get("step"), fws.get("turn"), d.get("quote"), json.dumps(d.get("milestones_missed") or []),
                     json.dumps(d.get("milestones_lost") or []), d.get("phi_final"), d.get("stop"), url])
    run.table(f"{key}/table", FINDING_COLUMNS, rows)
    return rows
