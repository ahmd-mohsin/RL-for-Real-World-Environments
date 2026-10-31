"""The policy layer's types (DESIGN §7.7 item 24; runs/policy-1.md, the paper's §9.7).

A **principle** is a short, ordered, testable statement (``P-valid`` ... ``P-ledger``); its test is a function over a
candidate remedy (``synth.policy.resolve.PRINCIPLE_CHECKS``). A **rule** is the codified form of one decision already
taken: the conditions it applies to, its parameters, the remedy ladder for those conditions, the principles it derives
from, its source (the lead's decision and date, or an inferred rule with the code or run note it was inferred from) and
its version. Rules and principles are data: they live in a hash-chained ledger (``synth.policy.ledger``), and a policy
version is a ledger position.

Three vocabularies are closed and live in code, because code implements them:
- the **actions** a remedy can take (``ACTIONS``: each with its rung, its declared effects and the call site that
  carries it out);
- the **validity parameters** (``VALIDITY``: every rule parameter that guards admission, with the direction that is
  stricter): P-valid's data;
- the **escalation grounds** (``ESCALATION_KINDS``): the only four reasons a person is asked.

A remedy that would relax a validity parameter cannot be constructed (``Remedy`` raises ``PolicyViolation``): that is
what "refused by construction" means.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from crucible.core.schema.base import CrucibleModel

SCHEMA = "crucible.synth.policy/v1"

# ---------------------------------------------------------------------------------------------------------- rungs

Rung = Literal["evidence", "repair", "relabel", "variant", "defer", "escalate"]
RUNG_ORDER: tuple[str, ...] = ("evidence", "repair", "relabel", "variant", "defer", "escalate")

Disposition = Literal["HOLD", "RESERVE", "REJECT", "PROVISIONAL", "HUMAN_QUEUE", "NOT_USED", "TRIM", "DROPPED", "NO_CLAIM",
                      "NO_UPDATE", "SHORTFALL"]

EvidenceKind = Literal["executed", "calibrated", "judged", "none"]

EscalationKind = Literal["budget_beyond_total", "sealed_access", "irreversible_action", "principle_change"]
ESCALATION_KINDS: tuple[str, ...] = ("budget_beyond_total", "sealed_access", "irreversible_action", "principle_change")


class PolicyViolation(Exception):
    """A remedy, a ledger entry or an escalation that the principles forbid. Not a ``ValueError``, so pydantic validation
    never turns it into an ordinary validation error: constructing a relaxing remedy raises this, by name."""

    def __init__(self, principle: str, why: str) -> None:
        super().__init__(f"{principle}: {why}")
        self.principle = principle
        self.why = why


# ---------------------------------------------------------------------------------------------------------- actions

class ActionSpec(CrucibleModel):
    rung: Rung
    effects: list[str] = Field(default_factory=list)
    evidence: EvidenceKind = "none"
    disposition: Disposition | None = None
    implemented_by: str = ""  # the call site that carries the action out


def _a(rung: Rung, impl: str, *effects: str, evidence: EvidenceKind = "none", disposition: Disposition | None = None) -> ActionSpec:
    return ActionSpec(rung=rung, effects=list(effects), evidence=evidence, disposition=disposition, implemented_by=impl)


ACTIONS: dict[str, ActionSpec] = {
    # more evidence (P-more-evidence-first)
    "check_oracle": _a("evidence", "synth.falsify.oracle.ORACLE; meta.rsi.TaskLoop._oracle", "runs_executed_check", evidence="executed"),
    "escalate_solver_ladder": _a("evidence", "synth.solvers.escalate_alt", "runs_solvers", evidence="executed"),
    "densify_world": _a("evidence", "worldgen.reseed; families.make_instance(k_min=); synth_wire.finance_ap_world(invoices=)",
                        "new_world_instance", "recertify", evidence="executed"),
    "reseed_denser": _a("evidence", "worldgen.reseed; families.make_instance(k_min=)", "new_world_instance", evidence="executed"),
    "more_rollouts": _a("evidence", "learn.decide looks; meta.rsi.TaskLoop._placed_band (escalation looks to k_cap)", "spends_student_rollouts",
                        evidence="executed"),
    "gather_task_evidence": _a("evidence", "outcome.TaskEvidence", "runs_executed_check", evidence="executed"),
    "executed_check": _a("evidence", "falsify.agents.Adjudicated (re-execution); falsify.online.ReplayConfirmer", "runs_executed_check",
                         evidence="executed"),
    "add_calibration_anchors": _a("evidence", "pipeline.Runner.calibrate_panel; falsify.twinjudge.twin_anchors", "runs_executed_check",
                                  evidence="executed"),
    "panel_decide": _a("evidence", "falsify.agents.Panel.decide (calibrated λ)", "judged_calibrated", evidence="calibrated"),
    "rerun_control_equal_spend": _a("evidence", "synth.batch arms; frontier.compare; meta.lemmas.lemma_acceptance; meta.lift.LemmaLift (the lemma gate)", "spends_budget",
                                    evidence="executed"),
    "larger_fresh_view": _a("evidence", "meta.rsi.Sampler(strict=True); meta.update.UpdateEngine.decide", "spends_budget", evidence="executed"),
    "synthesize_more": _a("evidence", "synth.batch.Batch", "spends_budget", evidence="executed"),
    "build_more_families": _a("evidence", "synth.families (scripts/synth_worlds.py build)", "spends_budget", evidence="executed"),
    # repair with evidence
    "repair_with_counterexamples": _a("repair", "cegis.refine (repairer; monotone archive)", "edits_task", evidence="executed"),
    "repair_oracle": _a("repair", "meta.rsi.TaskLoop._repair (ORACLE counterexample)", "edits_task", evidence="executed"),
    "retry_oracle": _a("repair", "pipeline.Runner.build (oracle retries with errors)", "edits_task", evidence="executed"),
    "rebuild_stage": _a("repair", "synth.families restart_from / build --redo", "edits_world", evidence="executed"),
    "reassign_family": _a("repair", "roles.family_problems; factory.synth_wire.check_assignment", "reassigns_role"),
    "reallocate_within_total": _a("repair", "synth.policy.ledger (a ledger entry raising one stage cap inside the Part 2 total)",
                                  "changes_stage_cap"),
    # honest relabel
    "reskill": _a("relabel", "meta.reskill; meta.rsi.TaskLoop._try_reskill (RS1-RS6)", "relabels_intent", evidence="executed"),
    # a lawful difficulty move (a new variant; the original is kept)
    "descend": _a("variant", "meta.rsi.TaskLoop.step (ComposeKnobs: undo, fewer_distractors, unpaginate)", "new_variant", evidence="executed"),
    "ascend": _a("variant", "meta.rsi.TaskLoop.step (ComposeKnobs: distractors, paginate)", "new_variant", evidence="executed"),
    # defer (nothing is deleted)
    "hold": _a("defer", "the caller records HOLD", disposition="HOLD"),
    "reserve": _a("defer", "meta.rsi.TaskLoop (lineage status reserved)", disposition="RESERVE"),
    "reject": _a("defer", "certificate.certify (status rejected, reasons kept)", disposition="REJECT"),
    "provisional": _a("defer", "certificate.certify (status provisional, never counted certified)", disposition="PROVISIONAL"),
    "human_queue": _a("defer", "falsify.agents.Panel human_queue.jsonl; triage", disposition="HUMAN_QUEUE"),
    "not_used": _a("defer", "worldgates gates.json (instance not used, reasons kept)", disposition="NOT_USED"),
    "trim_plan": _a("defer", "the CLI clips the run's cap (work_spent / milestone_spent)", disposition="TRIM"),
    "drop": _a("defer", "pipeline (status dropped, the record kept)", disposition="DROPPED"),
    "no_claim": _a("defer", "the report states no improvement claim", disposition="NO_CLAIM"),
    "no_update": _a("defer", "meta.update.UpdateEngine (the incumbent stays); meta.lift.LemmaLift (the candidate is deferred, the bank stays)",
                    disposition="NO_UPDATE"),
    "report_shortfall": _a("defer", "the run report states the shortfall", disposition="SHORTFALL"),
    # escalate (a person; only ESCALATION_KINDS)
    "escalate": _a("escalate", "synth.policy.ledger.EscalationQueue"),
}

# effects no catalogue action has; a remedy proposed from elsewhere (a model, a script) may declare them, and a
# principle refuses it
UNSAFE_EFFECTS: dict[str, str] = {
    "counts_heldout": "P-valid", "segments_task": "P-valid", "admits_open_suspects": "P-valid", "relaxes_admission": "P-valid",
    "lowers_holdout": "P-valid", "admits_on_judgement": "P-evidence", "decides_uncalibrated": "P-evidence",
    "changes_skill": "P-intent", "derive_to_given": "P-intent", "deletes": "P-no-delete", "student_reward_in_J": "P-student",
    "repair_by_pass_rate": "P-student", "unequal_budget_claim": "P-equal-budget", "same_family_check": "P-family",
    "irreversible": "P-person", "external": "P-person", "reads_sealed": "P-person", "exceeds_part2_total": "P-person",
    "changes_principle": "P-person", "changes_rule": "P-ledger",
}
PERSON_EFFECTS = ("irreversible", "external", "reads_sealed", "exceeds_part2_total", "changes_principle")

# ---------------------------------------------------------------------------------------------------------- validity

Direction = Literal["lower_strict", "higher_strict", "true_strict", "false_strict", "superset_strict", "subset_strict", "fixed"]

# (rule, parameter) -> the direction that is stricter. P-valid's data: a change in the other direction is a relaxation.
VALIDITY: dict[tuple[str, str], Direction] = {
    ("R-admission-fixed", "u_max_mut"): "lower_strict",
    ("R-admission-fixed", "n_min"): "higher_strict",
    ("R-admission-fixed", "max_confirmed"): "lower_strict",
    ("R-admission-fixed", "atk_max_confirmed"): "lower_strict",
    ("R-admission-fixed", "zero_confirmed_classes"): "superset_strict",
    ("R-admission-fixed", "require_applicable"): "superset_strict",
    ("R-admission-fixed", "require_alt_pass"): "true_strict",
    ("R-admission-fixed", "alt_non_builder"): "true_strict",
    ("R-admission-fixed", "heldout_survivor_rejects"): "true_strict",
    ("R-admission-fixed", "allow_open_suspects"): "false_strict",
    ("R-admission-fixed", "provisional_counts_as_certified"): "false_strict",
    ("R-holdout", "h"): "fixed",
    ("R-holdout", "count_heldout_in_bound"): "false_strict",
    ("R-ALT-whole", "segmented"): "false_strict",
    ("R-ALT-whole", "change_rule_on_failure"): "false_strict",
    ("R-panel", "min_decided"): "higher_strict",
    ("R-panel", "require_both"): "true_strict",
    ("R-panel", "refuse_uncalibrated"): "true_strict",
    ("R-panel", "target"): "higher_strict",
    ("R-panel", "floor"): "higher_strict",
    ("R-panel", "panel_sufficient"): "false_strict",
    ("R-band", "band_lo"): "fixed",
    ("R-band", "band_hi"): "fixed",
    ("R-band", "alpha"): "lower_strict",
    ("R-band", "require_settled"): "true_strict",  # taskloop-2: a placement needs a confident result whose interval straddles no edge
    ("R-band", "place_on_second_alone"): "false_strict",  # a second measurement never places a task the first disagrees with
    ("R-band", "pool_diagnosis_into_acceptance"): "false_strict",  # sample splitting (G4) stays
    ("R-lemma-lift", "admit_only_through_gate"): "true_strict",
    ("R-lemma-lift", "alpha_t"): "lower_strict",
    ("R-p3", "unmeasured_counts_as_on_intent"): "false_strict",
    ("R-too-hard", "easier_knobs"): "subset_strict",
    ("R-too-hard", "forbidden_easier"): "superset_strict",
    ("R-too-hard", "reskill_cap"): "lower_strict",
    ("R-density", "k_min"): "higher_strict",
    ("R-density", "n_min"): "higher_strict",
    ("R-family", "student_families"): "superset_strict",
    ("R-family", "no_self_check"): "true_strict",
    ("R-student-probe", "student_reward_in_J"): "false_strict",
    ("R-equal-budget", "equal_budget_required"): "true_strict",
    ("R-equal-budget", "budget_tolerance"): "lower_strict",
    ("R-rsi-power", "alpha"): "lower_strict",
    ("R-rsi-power", "reuse_units"): "false_strict",
    ("R-rsi-power", "delta_moves_down"): "false_strict",
}

# rule parameters a remedy may change without a person: stage caps, inside the Part 2 total (R-budget). Every other
# parameter changes only through a ledger entry a person's answer creates.
AUTO_PARAMS: frozenset[tuple[str, str]] = frozenset({("R-budget", "stage_caps_usd")})
# parameters only a person's answer may raise or change (not validity, but outside any automatic remedy)
PERSON_PARAMS: frozenset[tuple[str, str]] = frozenset({("R-budget", "part2_total_usd")})


def relaxes(rule: str, param: str, old: Any, new: Any) -> bool:
    """True when changing ``rule.param`` from ``old`` to ``new`` loosens a validity guard (P-valid)."""
    d = VALIDITY.get((rule, param))
    if d is None or old == new:
        return False
    if d == "fixed":
        return True  # a fixed parameter changes only through a principle change (a person)
    if d == "lower_strict":
        return float(new) > float(old)
    if d == "higher_strict":
        return float(new) < float(old)
    if d == "true_strict":
        return bool(old) and not bool(new)
    if d == "false_strict":
        return (not bool(old)) and bool(new)
    if d == "superset_strict":
        return not set(_as_list(new)) >= set(_as_list(old))
    if d == "subset_strict":
        return not set(_as_list(new)) <= set(_as_list(old))
    return True


def _as_list(x: Any) -> list[Any]:
    return list(x) if isinstance(x, (list, tuple, set, frozenset)) else [x]


# ---------------------------------------------------------------------------------------------------------- principles and rules

class Source(CrucibleModel):
    kind: Literal["lead", "inferred", "person"]
    who: str = ""
    date: str = ""
    ref: str = ""  # the decision's record: a DESIGN item, a run note, a code site
    escalation: str | None = None  # a person's answer: the escalation it answers


class Principle(CrucibleModel):
    id: str
    rank: int  # earlier wins a conflict
    statement: str
    test: str  # what the check refuses, in words (the check itself is PRINCIPLE_CHECKS[id])


Pred = dict[str, Any]  # {"fact": f, "op": "eq|ne|lt|le|gt|ge|in|truthy|falsy|present", "value": v} | {"all": [...]} | {"any": [...]} | {"not": p}


class Step(CrucibleModel):
    """One rung of a rule's ladder: an action, when it applies (a predicate over the condition's facts; a missing fact
    makes it inapplicable: fail closed toward later, safer rungs) and which condition kinds it serves."""

    action: str
    when: Pred | None = None
    kinds: list[str] | None = None  # None: every kind the rule applies to
    params: dict[str, Any] = Field(default_factory=dict)
    escalation: EscalationKind | None = None
    note: str = ""

    @model_validator(mode="after")
    def _known(self) -> Step:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}: the action vocabulary is closed (policy.model.ACTIONS)")
        if (self.action == "escalate") != (self.escalation is not None):
            raise ValueError("an escalate step names its ground, and only an escalate step does")
        return self

    @property
    def rung(self) -> str:
        return ACTIONS[self.action].rung


class Rule(CrucibleModel):
    id: str
    title: str
    statement: str  # the decision, in words
    applies_to: list[str] = Field(default_factory=list)  # condition kinds ("*": any kind no other rule covers)
    params: dict[str, Any] = Field(default_factory=dict)
    ladder: list[Step] = Field(default_factory=list)
    principles: list[str] = Field(default_factory=list)
    source: Source
    version: int = 1
    enforced_at: list[str] = Field(default_factory=list)  # code sites that read or obey it

    @model_validator(mode="after")
    def _ladder_order(self) -> Rule:
        for kind in {*self.applies_to, None}:
            steps = [s for s in self.ladder if kind is None or s.kinds is None or kind in s.kinds]
            ranks = [RUNG_ORDER.index(s.rung) for s in steps]
            if ranks != sorted(ranks):
                raise ValueError(f"rule {self.id}: the ladder must run evidence -> repair -> relabel -> variant -> defer -> escalate "
                                 f"({[s.action for s in steps]})")
        return self


# ---------------------------------------------------------------------------------------------------------- conditions and remedies

class Condition(CrucibleModel):
    """A blocking condition: what blocks, on what subject, with the facts the caller measured. Facts are evidence
    availability and counts (``can_densify``, ``shown_n``, ``panel_calibrated``, ...), never a judgement."""

    kind: str
    subject: str = ""
    facts: dict[str, Any] = Field(default_factory=dict)
    raised_by: str = ""


class ParamChange(CrucibleModel):
    rule: str
    param: str
    old: Any = None
    new: Any = None


class Escalation(CrucibleModel):
    """A question for a person. Only the four grounds exist; the answer becomes a ledger entry (a new policy version)."""

    id: str
    kind: EscalationKind
    condition: Condition
    question: str
    options: list[dict[str, Any]] = Field(default_factory=list)  # {"id", "says", "amendment": ledger entry draft | None}
    policy_version: str = ""
    created: str = ""
    answered: str | None = None  # the ledger entry's sha once answered


class Remedy(CrucibleModel):
    """The lawful remedy ``resolve`` returns. Constructing one that relaxes a validity parameter raises."""

    action: str
    rung: Rung
    rule: str = ""  # the rule whose ladder produced it ("" for a remedy proposed from elsewhere)
    disposition: Disposition | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    effects: list[str] = Field(default_factory=list)
    evidence: EvidenceKind = "none"
    changes: list[ParamChange] = Field(default_factory=list)
    principles: list[str] = Field(default_factory=list)
    implemented_by: str = ""
    reason: str = ""
    escalation: Escalation | None = None
    policy_version: str = ""
    considered: list[dict[str, Any]] = Field(default_factory=list)  # every rung tried before it, with why it was passed over

    @model_validator(mode="after")
    def _p_valid(self) -> Remedy:
        for c in self.changes:
            if relaxes(c.rule, c.param, c.old, c.new):
                raise PolicyViolation("P-valid", f"{c.rule}.{c.param} {c.old!r} -> {c.new!r} relaxes a validity rule")
        bad = [e for e in self.effects if UNSAFE_EFFECTS.get(e) == "P-valid"]
        if bad:
            raise PolicyViolation("P-valid", f"declared effects {bad} relax admission")
        return self


__all__ = [
    "ACTIONS", "AUTO_PARAMS", "ESCALATION_KINDS", "PERSON_EFFECTS", "PERSON_PARAMS", "RUNG_ORDER", "SCHEMA", "UNSAFE_EFFECTS", "VALIDITY",
    "ActionSpec", "Condition", "Disposition", "Escalation", "EscalationKind", "EvidenceKind", "ParamChange", "PolicyViolation",
    "Principle", "Remedy", "Rule", "Rung", "Source", "Step", "relaxes",
]
