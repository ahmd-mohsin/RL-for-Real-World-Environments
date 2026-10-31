"""Ledger amendments after the genesis, as data (DESIGN §7.7 item 25; runs/taskloop-2.md).

Each batch is a list of ledger drafts; ``python -m crucible.synth.policy.ledger --amend <batch>`` appends the drafts of a
batch that the ledger does not hold yet (matched by ``batch`` and ``draft``), so applying a batch twice changes
nothing. The shipped ``ledger.jsonl`` holds them; ``test_synth_policy`` checks that it does, entry by entry. A batch is
never edited after it is applied: a later decision is a later batch.

**taskloop-2** (the lead's brief of 2026-10-30 on the findings of runs/taskloop-1.md):
1. ``R-band`` v2: ``band_undecided`` covers a forced (unconfident) final look, a 1 - alpha interval that straddles a band
   edge, and two measurements of one task that disagree. The remedy is more fresh evidence first (the looks to 16, then
   to ``k_cap`` = 32, within the per-lineage ``task_rollout_cap``), then HOLD. A second measurement never places a task
   alone and diagnosis rollouts are never pooled into acceptance.
2. ``R-student-probe``: ``k_max`` follows R-band's ``k_cap`` (32), so the probe cap names the same bound.
3. ``R-too-hard``: ``family_knobs``: the declared difficulty knobs that are lawful per World family. finance_ap has none in
   either direction (reasons in ``why``), so the loop stops trying them and RESERVE follows at once.
4. ``R-lemma-lift`` (new): an accepted repair's counterexample class that recurs on ``min_tasks`` lineages becomes a
   candidate lemma; it enters the lemma bank only through the gate (``meta.lemmas.lemma_acceptance`` on an equal-budget
   paired build at alpha_t); an underpowered gate defers.
5. ``R-p3`` (new): P3 unmeasured on a candidate edit is a named condition: run the executed P3 probes the World supports
   (``intent.p3`` with the SQL evidence channel), else HOLD the edit. Unmeasured never counts as on intent.
"""

from __future__ import annotations

from typing import Any

from crucible.synth.policy.genesis import _all, _p
from crucible.synth.policy.model import Rule, Source, Step

TL2_DATE = "2026-10-30"
TL2_SRC = Source(kind="lead", who="lead", date=TL2_DATE, ref="taskloop-2 brief (the findings of runs/taskloop-1.md §B.2, §B.4, §B.7); runs/taskloop-2.md")


def _src(ref: str = "") -> dict[str, Any]:
    return TL2_SRC.model_copy(update={"ref": f"{TL2_SRC.ref}{'; ' + ref if ref else ''}"}).model_dump(mode="json")


R_BAND_V2_LADDER = [
    Step(action="more_rollouts", when=_all(_p("rollouts", "lt", param="k_cap"), _p("probe_budget_ok", "truthy")),
         note="fresh rollouts on the same measurement (the next escalation look), never the other measurement's"),
    Step(action="hold", note="the band stays undecided: the task waits (never placed by one measurement against another)"),
]

FINANCE_AP_KNOBS = {
    "easier": [],
    "harder": [],
    "why": {
        "undo_harder_knob": "no finance_ap task carries a recorded harder knob (no harder knob is lawful here), so there is nothing to undo",
        "fewer_distractors": "needs a variant World with distractor clones; the finance_ap SQL World takes no clone perturbation",
        "unpaginate": "needs a paginated variant, which finance_ap never gets (paginate is not lawful here)",
        "derive_to_given": "never an easier move (R-too-hard forbidden_easier): it changes the skill",
        "distractors": "a clone of an AP row is an input of a policy rule (duplicate invoices, a vendor's invoice history, PO and receipt "
                       "matching), so a cloned distractor can change the decision the executed policy truth computes from the "
                       "generator: the variant's reference decision would no longer be certified. Unsound on this family",
        "paginate": "applies mechanically, but taskloop-1 measured it on 27 finance_ap lineages (52 decisions): it never moved the band "
                    "(20 still too easy on 16 fresh rollouts) and P3 was unmeasured on the paginated variants (28 G2 refusals). A "
                    "named-invoice decision reads the invoice by number, so a paged list does not touch the skill",
        "actor": "needs a writer and a second actor in the World; finance_ap has one policy actor",
        "derive": "needs a writer that moves a stated value out of the instruction; none is wired for finance_ap and the decision "
                  "values are already derived (not stated)",
        "densify_superset": "sound (the superset World keeps every base row and decision, synth.densify) but not a difficulty knob: "
                            "the student finds the named invoice by number, so more pending invoices on new vendors do not change "
                            "what it must do",
    },
}

R_LEMMA_LIFT = Rule(
    id="R-lemma-lift", title="Repairs lift to lemmas only through the gate",
    statement="An accepted repair is recorded as a (counterexample class, repair) pair. When one counterexample class (the defect "
              "of the repair's counterexamples) recurs on at least min_tasks lineages, its pairs become a candidate lemma: the "
              "abstractor (a non-builder family) writes the rule from the scrubbed cluster, and the deterministic leakage critic "
              "refuses a rule that names a task, entity or value of its evidence. The candidate enters the lemma bank only "
              "through the gate: meta.lemmas.lemma_acceptance on an equal-budget paired build (with and without the lemma) at "
              "alpha_t, and J rising under the update rule. A gate that cannot reach power at target_gain with the units the "
              "budget buys is not run: the candidate is deferred (no update), never admitted.",
    applies_to=["lemma_candidate"],
    params={"min_tasks": 2, "key": "defect", "alpha_t": 0.005, "target_gain": 0.2, "power": 0.8, "usd_per_cell": 0.08,
            "admit_only_through_gate": True},
    ladder=[Step(action="rerun_control_equal_spend",
                 when=_all(_p("leak_free", "truthy"), _p("budget_ok", "truthy"), _p("units_available", "ge", ref="units_needed")),
                 note="the gate: paired McNemar on the first-round confirmed-counterexample rate at equal spend, plus J under R1-R10"),
            Step(action="no_update", note="underpowered (or leaking, or no budget): the candidate is deferred; the bank is unchanged")],
    principles=["P-more-evidence-first", "P-equal-budget", "P-valid", "P-family"],
    source=TL2_SRC,
    enforced_at=["meta.lift.LemmaLift (record, recur, draft, gate)", "meta.rsi.TaskLoop._repair (lift_lemmas)", "meta.lemmas.lemma_acceptance",
                 "scripts/synth_taskloop.py lift"])

R_P3 = Rule(
    id="R-p3", title="An unmeasured P3 is a named condition, never on intent",
    statement="P3 (on intent) is measured on every candidate edit (repair, variant, RESKILL) by intent.p3: TAINT, TWIN, EVID and "
              "SWAP. When no valid EVID, SWAP or TWIN trial runs, the edit raises p3_unmeasured: run the executed P3 probes the "
              "World supports (a SQL-backed World edits its start state through the evidence channel, iface.SqlEvidenceWorld), "
              "else HOLD the edit with the condition recorded. Unmeasured never counts as on intent and is never a silent refusal.",
    applies_to=["p3_unmeasured"],
    params={"unmeasured_counts_as_on_intent": False},
    ladder=[Step(action="executed_check", when=_p("executed_check_available", "truthy"),
                 note="intent.p3 through the World's evidence channel (EVID on a SQL World)"),
            Step(action="hold", note="the edit is not accepted; the lineage keeps its state with p3_unmeasured recorded")],
    principles=["P-evidence", "P-intent", "P-more-evidence-first"],
    source=TL2_SRC,
    enforced_at=["meta.rsi.TaskLoop._p3 (repairs, variants, RESKILL)", "synth.intent.p3", "synth.iface.SqlEvidenceWorld",
                 "meta.reskill.p3_checker (evidence=True)"])

BATCHES: dict[str, list[dict[str, Any]]] = {
    "taskloop-2": [
        {"kind": "amend_rule", "rule_id": "R-band", "draft": "band_undecided",
         "set": {"k_cap": 32, "escalation_looks": [32], "task_rollout_cap": 96, "interval": "equal-tailed 1 - alpha posterior (Jeffreys)",
                 "require_settled": True, "place_on_second_alone": False, "pool_diagnosis_into_acceptance": False},
         "statement": "The keep band is p in [0.1, 0.6] on the student, measured predictor-first in looks of 4, 8, 12, 16 rollouts at alpha "
                      "0.1. A measurement places a task only when it is settled: confident (the class holds posterior mass >= 1 - "
                      "alpha) and its 1 - alpha interval straddles no band edge. A forced final look, a straddling interval, or two "
                      "measurements of one task that disagree (a RESKILL's fresh band against the diagnosis band) raise "
                      "band_undecided: more fresh rollouts on that measurement first (escalation looks to k_cap = 32, within "
                      "task_rollout_cap per lineage and the probe cap), then HOLD. A second measurement never places a task alone; "
                      "diagnosis rollouts are never pooled into acceptance.",
         "ladder": [s.model_dump(mode="json") for s in R_BAND_V2_LADDER],
         "enforced_at": ["synth.learn (decide, settled, credible_interval)", ("meta.rsi.TaskRules (band_lo, band_hi, band_alpha, looks, k_cap, "
                                                                              "escalation_looks, task_rollout_cap)"), "meta.rsi.TaskLoop._placed_band"],
         "source": _src("R-band v2"),
         "rationale": "taskloop-1 §B.2: 5 of 7 in-band tasks re-labelled by RESKILL were reserved on their 16 fresh rollouts although only the "
                      "label changed. 6 of the 7 diagnosis bands were forced at the final look (unconfident at 16), so the second "
                      "measurement placed them alone: the band test's own error, not the task. P-more-evidence-first: escalate, then hold."},
        {"kind": "amend_rule", "rule_id": "R-student-probe", "draft": "k_max", "set": {"k_max": 32}, "source": _src("R-band v2 k_cap"),
         "rationale": "the probe rule names the same per-measurement bound as R-band v2 (k_cap 32); the USD cap is unchanged"},
        {"kind": "amend_rule", "rule_id": "R-too-hard", "draft": "family_knobs", "set": {"family_knobs": {"finance_ap": FINANCE_AP_KNOBS}},
         "enforced_at": ["meta.rsi.TaskLoop.step (order; resolve; family_knobs)", "meta.rsi.TaskRules (reskill_cap, kappa, max_rounds)",
                         "meta.reskill.check_proposal"],
         "source": _src("family knobs"),
         "rationale": "taskloop-1 §B.1, §B.7: DESCEND had no lawful knob and ASCEND 0 of 52 on finance_ap; no skill-preserving knob is "
                      "sound and effective on that family (reasons per knob), so the moves are recorded inapplicable and the loop "
                      "reserves at once instead of re-probing"},
        {"kind": "add_rule", "draft": "R-lemma-lift", "rule": R_LEMMA_LIFT.model_dump(mode="json"), "source": _src("lemma lift"),
         "rationale": "taskloop-1 §B.4: 3 accepted repairs were lemma candidates but lift_lemmas was not wired; the lift is wired and "
                      "admission stays with the existing gate (rsi-stats §9.4''), which defers when underpowered"},
        {"kind": "add_rule", "draft": "R-p3", "rule": R_P3.model_dump(mode="json"), "source": _src("P3 on SQL Worlds"),
         "rationale": "taskloop-1 §B.7: G2 refused 28 ASCEND and 25 repair decisions silently on an unmeasured P3 (no EVID on a SQL "
                      "World). The SQL evidence channel measures it where the World has an evidence site; elsewhere it is a named "
                      "condition with a lawful remedy"},
    ],
}


def drafts(batch: str) -> list[dict[str, Any]]:
    """The batch's ledger drafts, each tagged with its batch (for idempotent application)."""
    return [{**d, "batch": batch} for d in BATCHES[batch]]


__all__ = ["BATCHES", "FINANCE_AP_KNOBS", "R_BAND_V2_LADDER", "R_LEMMA_LIFT", "R_P3", "drafts"]
