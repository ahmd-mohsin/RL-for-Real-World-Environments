"""The lawful-remedy procedure (DESIGN §7.7 item 24; runs/policy-1.md, the paper's §9.7).

``resolve(condition)`` returns the first lawful remedy for a blocking condition:

1. The rules that name the condition's kind (else ``R-default``, the principles alone) give their ladder steps; a step
   applies when its predicate holds on the condition's facts (a missing fact makes it inapplicable, so resolution falls
   toward later, safer rungs). Remedies proposed from elsewhere (``extra``: a model's suggestion, a script's) join the
   candidates.
2. Candidates are walked in rung order: evidence -> repair -> relabel -> variant -> defer -> escalate (a stable sort, so
   a rule's own order holds inside a rung). This is P-more-evidence-first.
3. Each applicable candidate is built as a ``Remedy`` (one that relaxes a validity parameter cannot be built: P-valid)
   and checked against every principle (``PRINCIPLE_CHECKS``). The first that passes is returned, with every candidate
   passed over and why (``considered``).
4. Nothing passes: HOLD, which every principle admits.

A person is asked only through an ``escalate`` step, and only on the four grounds of P-person, each of which must be
borne out by the facts (``grounds``). The escalation is a typed record whose options are ledger amendments: the
person's answer becomes a new policy version (``PolicyLedger.answer``), never a code change.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

from crucible.synth.policy.ledger import Policy, current
from crucible.synth.policy.model import (
    ACTIONS,
    AUTO_PARAMS,
    PERSON_EFFECTS,
    RUNG_ORDER,
    UNSAFE_EFFECTS,
    Condition,
    Escalation,
    ParamChange,
    PolicyViolation,
    Remedy,
    Rule,
    Step,
)

# the condition kinds the code raises, with the facts each one's rules read (a fact the caller does not measure is left
# out: the steps that need it do not apply)
CONDITIONS: dict[str, tuple[str, list[str]]] = {
    "confirmed_counterexample": ("a class confirmed a counterexample", ["cls", "repairer_available", "repair_rounds"]),
    "underpowered_certificate": ("c = 0 but fewer than n_min shown distinct mutants (U_MUT > 0.05)",
                                 ["shown_n", "heldout_n", "can_densify", "densify_tries"]),
    "heldout_survivor": ("a held-out mutant survived the certified version", ["survivors"]),
    "no_solver_pass": ("ALT has no pass from a non-builder family", ["truth_available", "oracle_checked", "rungs_tried", "budget_ok", "horizon"]),
    "band_too_hard": ("the student's band is too hard (decided on non-uncertain rollouts)",
                      ["truth_available", "oracle_checked", "unintended_with_evidence", "reskill_attempted", "deriver_available", "easier_knob"]),
    "band_too_easy": ("the student's band is too easy", ["truth_available", "oracle_checked", "unintended_with_evidence", "reskill_attempted",
                                                         "deriver_available", "harder_knob"]),
    "band_undecided": (("a band measurement does not place the task: it is unconfident, its 1 - alpha interval straddles a band edge, or "
                        "two measurements of the same task disagree (R-band v2, taskloop-2)"),
                       ["rollouts", "probe_budget_ok", "measurements", "disagree", "straddles", "lineage_rollouts"]),
    "lemma_candidate": ("an accepted repair's counterexample class recurs on other lineages: a candidate lemma for the gate (R-lemma-lift)",
                        ["recurs_tasks", "leak_free", "units_needed", "units_available", "budget_ok"]),
    "p3_unmeasured": ("P3 has no valid EVID, SWAP or TWIN trial on a candidate edit (R-p3)", ["edit", "executed_check_available"]),
    "mostly_uncertain_rollouts": ("most graded rollouts are uncertain", ["task_evidence_run"]),
    "student_probe_over_cap": ("the student probe cap is reached", []),
    "uncalibrated_panel": ("the panel's lambda is not calibrated", ["executed_check_available", "panel_calibrated", "more_anchors_available"]),
    "open_suspects": ("suspects are undecided (MUT, TWIN, LEAK, ALT, ATK)", ["cls", "executed_check_available", "panel_calibrated",
                                                                             "more_anchors_available"]),
    "wrong_oracle": ("ORACLE confirmed the reference decision wrong", ["repair_attempts"]),
    "disputed_oracle": ("independent runs dispute the reference decision", ["truth_available", "oracle_checked", "panel_calibrated"]),
    "dead_oracle": ("the oracle program does not run", ["retries"]),
    "density_gate_failing": ("a World instance fails the density gate", ["can_reseed", "reseed_tries"]),
    "world_gate_failing": ("a World instance fails another gate", ["gate", "restartable", "rebuild_tries"]),
    "over_budget": ("a plan exceeds its cap", ["stage", "planned_usd", "min_useful_usd", "stage_remaining_usd", "part2_remaining_usd", "arms",
                                               "arms_equal"]),
    "scale_shortfall": ("the pool is below the scale target", ["environments", "tasks", "families", "budget_ok"]),
    "same_family_check": ("a family would check its own work", ["role", "family", "builders", "alternative_family_available"]),
    "unequal_budget_comparison": ("an improvement claim lacks an equal-spend control", ["control_rerun_possible", "budget_ok"]),
    "underpowered_decision": ("an update test cannot reach its power", ["units_available", "units_needed"]),
    "delete_request": ("something would be deleted", ["what"]),
    "sealed_access": ("sealed or held-out data would be read outside its report channel", ["path", "purpose"]),
    "irreversible_action": ("an irreversible or external action", ["action"]),
    "principle_change": ("a principle (or a fixed validity parameter) would change", ["principle", "proposal"]),
}


# ---------------------------------------------------------------------------------------------------------- predicates

def _operand(pred: dict[str, Any], facts: dict[str, Any], params: dict[str, Any]) -> tuple[bool, Any]:
    if "ref" in pred:
        return (pred["ref"] in facts, facts.get(pred["ref"]))
    if "param" in pred:
        return (pred["param"] in params, params.get(pred["param"]))
    if "param_len" in pred:
        return (pred["param_len"] in params, len(params.get(pred["param_len"]) or []))
    return ("value" in pred, pred.get("value"))


def holds(pred: dict[str, Any] | None, facts: dict[str, Any], params: dict[str, Any] | None = None) -> bool:
    """Evaluate a step predicate. A missing fact (or operand) makes a comparison false: fail closed."""
    params = params or {}
    if pred is None:
        return True
    if "all" in pred:
        return all(holds(p, facts, params) for p in pred["all"])
    if "any" in pred:
        return any(holds(p, facts, params) for p in pred["any"])
    if "not" in pred:
        return not holds(pred["not"], facts, params)
    f, op = pred["fact"], pred["op"]
    if op == "absent":
        return f not in facts
    if f not in facts:
        return False
    x = facts[f]
    if op == "present":
        return True
    if op == "truthy":
        return bool(x)
    if op == "falsy":
        return not bool(x)
    ok, y = _operand(pred, facts, params)
    if not ok:
        return False
    try:
        if op == "eq":
            return bool(x == y)
        if op == "ne":
            return bool(x != y)
        if op == "lt":
            return bool(x < y)
        if op == "le":
            return bool(x <= y)
        if op == "gt":
            return bool(x > y)
        if op == "ge":
            return bool(x >= y)
        if op == "in":
            return x in (y or [])
    except TypeError:
        return False
    raise ValueError(f"unknown predicate op {op!r}")


# ---------------------------------------------------------------------------------------------------------- the principles' tests

Check = Callable[[Remedy, Condition, Policy], list[str]]


def _effects(r: Remedy, p: str) -> list[str]:
    return [f"declares {e}" for e in r.effects if UNSAFE_EFFECTS.get(e) == p]


def _p_valid(r: Remedy, c: Condition, pol: Policy) -> list[str]:
    out = _effects(r, "P-valid")
    from crucible.synth.policy.model import relaxes

    out += [f"{x.rule}.{x.param} relaxes" for x in r.changes if relaxes(x.rule, x.param, x.old, x.new)]
    return out


def _p_evidence(r: Remedy, c: Condition, pol: Policy) -> list[str]:
    out = _effects(r, "P-evidence")
    if r.action == "panel_decide" and c.facts.get("panel_calibrated") is not True:
        out.append("a panel decides only when calibrated")
    if r.disposition is None and r.rung == "defer":
        out.append("a deferral names its disposition")
    return out


def _p_intent(r: Remedy, c: Condition, pol: Policy) -> list[str]:
    out = _effects(r, "P-intent")
    if "relabels_intent" in r.effects and (r.action != "reskill" or r.evidence != "executed"):
        out.append("the skill changes only by a RESKILL derived from executed evidence")
    if r.action == "descend":
        knob = r.params.get("knob", c.facts.get("easier_knob"))
        if knob in (pol.param("R-too-hard", "forbidden_easier") if "R-too-hard" in pol.rules else ["derive_to_given"]):
            out.append(f"{knob} is never an easier move")
    return out


def _p_no_delete(r: Remedy, c: Condition, pol: Policy) -> list[str]:
    return _effects(r, "P-no-delete")


def _p_student(r: Remedy, c: Condition, pol: Policy) -> list[str]:
    out = _effects(r, "P-student")
    if "spends_student_rollouts" in r.effects and c.facts.get("probe_budget_ok") is not True:
        out.append("student rollouts only within the probe cap")
    if "spends_student_rollouts" in r.effects and c.kind == "band_undecided" and "R-band" in pol.rules:
        cap, n = pol.rules["R-band"].params.get("k_cap"), c.facts.get("rollouts")
        if cap is not None and not (isinstance(n, int | float) and not isinstance(n, bool) and n < cap):
            out.append(f"a band measurement draws at most k_cap = {cap} rollouts (R-band)")
    return out


def _p_more_evidence_first(r: Remedy, c: Condition, pol: Policy) -> list[str]:
    """An undecided measurement is answered by more evidence or a deferral (taskloop-2: a task the band cannot place is
    never edited, re-labelled or moved on that band)."""
    if c.kind == "band_undecided" and r.rung in ("repair", "relabel", "variant"):
        return [f"{r.action} on an undecided band: only more evidence or a deferral"]
    return []


def _p_equal_budget(r: Remedy, c: Condition, pol: Policy) -> list[str]:
    out = _effects(r, "P-equal-budget")
    if r.action == "reallocate_within_total" and int(c.facts.get("arms", 1) or 1) > 1 and c.facts.get("arms_equal") is not True:
        out.append("a budget change across compared arms must be equal")
    return out


def _p_family(r: Remedy, c: Condition, pol: Policy) -> list[str]:
    out = _effects(r, "P-family")
    fams = _names(r.params.get("families"))
    banned = _names(c.facts.get("builders"))
    if "R-family" in pol.rules:
        banned |= set(pol.param("R-family", "student_families"))
    if fams & banned:
        out.append(f"families {sorted(fams & banned)} would check their own (or the student's) work")
    return out


def _names(x: Any) -> set[str]:
    if isinstance(x, str):
        return {x}
    return {str(v) for v in x} if isinstance(x, (list, tuple, set, frozenset)) else set()


def _p_person(r: Remedy, c: Condition, pol: Policy) -> list[str]:
    out = []
    eff = [e for e in r.effects if e in PERSON_EFFECTS]
    if eff and r.rung != "escalate":
        out.append(f"{eff} needs a person (an escalation)")
    if r.rung == "escalate":
        kind = r.escalation.kind if r.escalation else None
        if kind is None or not grounds(kind, c, pol):
            out.append(f"no ground for a person: {kind!r} is not borne out by the condition {c.kind!r}")
    return out


def _p_ledger(r: Remedy, c: Condition, pol: Policy) -> list[str]:
    out = _effects(r, "P-ledger")
    for x in r.changes:
        if (x.rule, x.param) not in AUTO_PARAMS:
            out.append(f"{x.rule}.{x.param} changes only through the ledger (a person's answer)")
    return out


PRINCIPLE_CHECKS: dict[str, Check] = {
    "P-valid": _p_valid, "P-evidence": _p_evidence, "P-more-evidence-first": _p_more_evidence_first, "P-intent": _p_intent,
    "P-no-delete": _p_no_delete, "P-student": _p_student, "P-equal-budget": _p_equal_budget, "P-family": _p_family,
    "P-person": _p_person, "P-ledger": _p_ledger,
}


def check(r: Remedy, c: Condition, pol: Policy | None = None) -> list[str]:
    """Every principle's objections to a remedy, in principle order (empty: lawful)."""
    pol = pol or current()
    out: list[str] = []
    for p in pol.principles:
        out += [f"{p.id}: {w}" for w in PRINCIPLE_CHECKS.get(p.id, lambda *_: ["no test for this principle"])(r, c, pol)]
    return out


def grounds(kind: str, c: Condition, pol: Policy) -> bool:
    """Whether the facts bear out an escalation ground (P-person)."""
    f = c.facts
    if kind == "budget_beyond_total":
        return c.kind == "over_budget" and holds({"fact": "planned_usd", "op": "gt", "ref": "part2_remaining_usd"}, f)
    if kind == "sealed_access":
        return c.kind == "sealed_access" and f.get("purpose") != "report"
    if kind == "irreversible_action":
        return c.kind == "irreversible_action"
    if kind == "principle_change":
        return c.kind == "principle_change"
    return False


# ---------------------------------------------------------------------------------------------------------- escalations

def _esc_id(c: Condition, kind: str, pol: Policy) -> str:
    return "esc-" + hashlib.sha256(json.dumps([kind, c.model_dump(mode="json"), pol.sha], sort_keys=True, default=str).encode()).hexdigest()[:12]


def escalation_for(kind: str, c: Condition, pol: Policy) -> Escalation:
    """The typed question and its options; every option is a ledger amendment (or none), so the answer is a version."""
    f = c.facts
    if kind == "budget_beyond_total":
        need = float(f.get("planned_usd", 0.0)) - float(f.get("part2_remaining_usd", 0.0))
        total = float(pol.param("R-budget", "part2_total_usd"))
        q = f"Stage {f.get('stage', '?')} plans USD {f.get('planned_usd')} with USD {f.get('part2_remaining_usd')} left of the Part 2 total {total}."
        opts = [{"id": "raise_total", "says": f"raise the Part 2 total by USD {need:.2f}",
                 "amendment": {"kind": "amend_rule", "rule_id": "R-budget", "set": {"part2_total_usd": round(total + need, 2)}}},
                {"id": "decline", "says": "keep the total; the plan stays trimmed", "amendment": None}]
    elif kind == "sealed_access":
        q = f"Read sealed data {f.get('path', '?')} for {f.get('purpose', '?')!r} (outside its report channel)?"
        opts = [{"id": "grant_once", "says": "grant this access once, recorded",
                 "amendment": {"kind": "add_rule", "rule": _grant_rule(c, pol)}},
                {"id": "deny", "says": "deny", "amendment": None}]
    elif kind == "irreversible_action":
        q = f"Carry out the irreversible or external action {f.get('action', '?')!r}?"
        opts = [{"id": "approve_once", "says": "approve once, recorded", "amendment": None}, {"id": "deny", "says": "deny", "amendment": None}]
    else:
        q = f"Change principle {f.get('principle', '?')}: {f.get('proposal', '')}"
        opts = [{"id": "accept", "says": "accept the proposed change", "amendment": f.get("amendment")},
                {"id": "reject", "says": "keep the principle", "amendment": None}]
    return Escalation(id=_esc_id(c, kind, pol), kind=kind, condition=c, question=q, options=opts, policy_version=pol.tag)  # type: ignore[arg-type]


def _grant_rule(c: Condition, pol: Policy) -> dict[str, Any]:
    from crucible.synth.policy.model import Source

    rid = f"R-grant-{_esc_id(c, 'sealed_access', pol)[4:]}"
    return Rule(id=rid, title="A recorded sealed-data grant", statement=f"One access to {c.facts.get('path')} for {c.facts.get('purpose')!r}.",
                params={"path": c.facts.get("path"), "purpose": c.facts.get("purpose"), "uses": 1}, principles=["P-person"],
                source=Source(kind="person", ref="escalation answer")).model_dump(mode="json")


# ---------------------------------------------------------------------------------------------------------- resolve

def _build(step: Step, rule: Rule, c: Condition, pol: Policy) -> Remedy:
    spec = ACTIONS[step.action]
    params = dict(step.params)
    changes: list[ParamChange] = []
    if step.action == "reallocate_within_total":
        stage = str(c.facts.get("stage", ""))
        caps = dict(pol.param("R-budget", "stage_caps_usd"))
        need = float(c.facts.get("planned_usd", 0.0)) - float(c.facts.get("stage_remaining_usd", 0.0))
        new = {**caps, stage: round(float(caps.get(stage, 0.0)) + max(0.0, need), 2)}
        changes.append(ParamChange(rule="R-budget", param="stage_caps_usd", old=caps, new=new))
    if step.action == "escalate_solver_ladder":
        params.setdefault("rung", int(c.facts.get("rungs_tried", 0)))
        params.setdefault("rungs", rule.params.get("ladder_rungs"))
        params.setdefault("usd_per_config_task", rule.params.get("usd_per_config_task"))
    esc = escalation_for(step.escalation, c, pol) if step.escalation else None
    return Remedy(action=step.action, rung=spec.rung, rule=rule.id, disposition=spec.disposition, params=params, effects=list(spec.effects),
                  evidence=spec.evidence, changes=changes, principles=list(rule.principles), implemented_by=spec.implemented_by,
                  reason=step.note or rule.statement[:200], escalation=esc, policy_version=pol.tag)


def resolve(condition: Condition, policy: Policy | None = None, *, extra: Sequence[Remedy | dict[str, Any]] = ()) -> Remedy:
    """The first lawful remedy for ``condition`` (module docstring)."""
    pol = policy or current()
    cands: list[tuple[int, int, Any]] = []
    i = 0
    for rule in pol.rules_for(condition.kind):
        for step in rule.ladder:
            if step.kinds is not None and condition.kind not in step.kinds:
                continue
            cands.append((RUNG_ORDER.index(step.rung), i, (rule, step)))
            i += 1
    for x in extra:
        rung = (x.rung if isinstance(x, Remedy) else x.get("rung", "defer"))
        cands.append((RUNG_ORDER.index(rung) if rung in RUNG_ORDER else len(RUNG_ORDER), i, x))
        i += 1
    cands.sort(key=lambda t: (t[0], t[1]))
    considered: list[dict[str, Any]] = []
    for _rank, _i, cand in cands:
        if isinstance(cand, tuple):
            rule, step = cand
            name = f"{rule.id}:{step.action}"
            if not holds(step.when, condition.facts, rule.params):
                considered.append({"candidate": name, "passed_over": "not applicable on the facts"})
                continue
            try:
                rem = _build(step, rule, condition, pol)
            except PolicyViolation as e:
                considered.append({"candidate": name, "refused": [str(e)]})
                continue
            except (TypeError, ValueError) as e:  # a malformed fact (a number that is not one): this rung cannot be built
                considered.append({"candidate": name, "passed_over": f"malformed facts: {str(e)[:120]}"})
                continue
        else:
            name = f"proposed:{cand.action if isinstance(cand, Remedy) else cand.get('action')}"
            try:
                rem = cand if isinstance(cand, Remedy) else Remedy.model_validate(cand)
            except (PolicyViolation, ValueError) as e:
                considered.append({"candidate": name, "refused": [str(e)[:300]]})
                continue
            if rem.action not in ACTIONS:
                considered.append({"candidate": name, "refused": [f"unknown action {rem.action!r}: the action vocabulary is closed"]})
                continue
            spec = ACTIONS[rem.action]
            if rem.rung != spec.rung:
                considered.append({"candidate": name, "refused": [f"{rem.action} is a {spec.rung} action, not {rem.rung}"]})
                continue
            rem = rem.model_copy(update={"effects": sorted({*rem.effects, *spec.effects}), "policy_version": pol.tag,
                                         "disposition": rem.disposition or spec.disposition})
        why = check(rem, condition, pol)
        if why:
            considered.append({"candidate": name, "refused": why})
            continue
        return rem.model_copy(update={"considered": considered})
    hold = Remedy(action="hold", rung="defer", rule="", disposition="HOLD", reason="no lawful remedy applies: hold with the condition recorded",
                  principles=["P-no-delete", "P-person"], implemented_by=ACTIONS["hold"].implemented_by, policy_version=pol.tag,
                  considered=considered)
    assert not check(hold, condition, pol), "HOLD must always be lawful"
    return hold


__all__ = ["CONDITIONS", "PRINCIPLE_CHECKS", "check", "escalation_for", "grounds", "holds", "resolve"]
