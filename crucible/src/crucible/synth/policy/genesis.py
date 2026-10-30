"""The genesis of the Part 2 policy: the principles and every decision taken before the policy layer existed, as data.

``ledger.jsonl`` beside this file is the ledger these entries start (``python -m crucible.synth.policy.ledger --genesis``
writes it once; ``test_synth_policy`` checks the shipped file replays to exactly these entries). Later changes are
ledger entries appended to that file (``PolicyLedger.append`` / ``answer``), never edits here.

Sources: "lead" is a decision the lead took (date and the record that holds it); "inferred" is a rule read from the
code or a run note that already behaved this way (the record names it). Facts named in predicates are the ones the
caller measures (``resolve.CONDITIONS`` lists them per condition kind).
"""

from __future__ import annotations

from typing import Any

from crucible.synth.policy.model import Principle, Rule, Source, Step

PRINCIPLES: list[Principle] = [
    Principle(id="P-valid", rank=1, statement="Never relax a validity or admission rule to raise yield.",
              test="refuses a remedy that changes a VALIDITY parameter in its loose direction, counts held-out trials, segments a "
                   "task the rule says is whole, or admits with open suspects; such a remedy cannot be constructed"),
    Principle(id="P-evidence", rank=2, statement="Prefer executed evidence. A panel verdict is necessary, never sufficient, and an "
                                                 "uncalibrated panel abstains.",
              test="refuses admission on a judgement alone, and a panel decision while the panel is not calibrated"),
    Principle(id="P-more-evidence-first", rank=3, statement="When a decision is underpowered, gather more evidence (probes, rollouts, "
                                                            "a denser World, a stronger checker) before any other remedy.",
              test="resolve walks rungs in order; a later rung is chosen only when every applicable evidence rung was refused or is "
                   "inapplicable on the measured facts"),
    Principle(id="P-intent", rank=4, statement="The skill and cell are frozen except through an evidence-derived RESKILL.",
              test="refuses a remedy that changes the skill outside RESKILL, a RESKILL not derived from executed evidence, and "
                   "derive -> given as an easier move"),
    Principle(id="P-no-delete", rank=5, statement="Nothing is deleted; defer to RESERVE, HOLD or REJECT with the reason kept.",
              test="refuses a remedy that deletes"),
    Principle(id="P-student", rank=6, statement="Student probe rollouts may inform task decisions within caps; student reward never "
                                                "enters J; RSI is a creation-time loop unrelated to training.",
              test="refuses student reward in the objective, a repair justified by a pass rate, and rollouts past the probe cap"),
    Principle(id="P-equal-budget", rank=7, statement="Every claimed improvement is compared at equal spend.",
              test="refuses an improvement claim without an equal-spend control, and a budget change that favours one arm"),
    Principle(id="P-family", rank=8, statement="No family checks its own work.",
              test="refuses a remedy that has a builder's (or the student's) family check, solve-as-reference, judge or attack "
                   "the work"),
    Principle(id="P-person", rank=9, statement="A person decides only budget beyond the Part 2 total, sealed or held-out data access, "
                                               "irreversible or external actions, and changes to the principles.",
              test="refuses an escalation on any other ground, and any of those four effects without an escalation"),
    Principle(id="P-ledger", rank=10, statement="Every rule change is a versioned ledger entry; every run records the policy version it "
                                                "used.",
              test="refuses a remedy that changes a rule parameter other than a stage cap inside the total; the ledger refuses an "
                   "entry that relaxes validity or edits a principle without a person's answer"),
]


def _p(fact: str, op: str, value: Any = None, **ref: str) -> dict[str, Any]:
    d: dict[str, Any] = {"fact": fact, "op": op}
    if ref:
        d.update(ref)
    elif value is not None:
        d["value"] = value
    return d


def _all(*ps: dict[str, Any]) -> dict[str, Any]:
    return {"all": list(ps)}


def _any(*ps: dict[str, Any]) -> dict[str, Any]:
    return {"any": list(ps)}


LEAD_1029 = "2026-10-29"
LEAD_1030 = "2026-10-30"

RULES: list[Rule] = [
    Rule(
        id="R-admission-fixed", title="The admission rule is fixed",
        statement="Admit only with U_MUT <= 0.05 at n >= 59 shown distinct mutants; an ALT pass from a non-builder family; zero "
                  "confirmed counterexamples in every class, ORACLE and LEAK included; no open suspects; a held-out survivor "
                  "rejects; provisional is never counted certified. A confirmed counterexample is repaired with evidence under "
                  "the monotone archive, else the task is rejected with its reasons kept.",
        applies_to=["confirmed_counterexample"],
        params={"u_max_mut": 0.05, "n_min": 59, "max_confirmed": 0, "atk_max_confirmed": 0,
                "zero_confirmed_classes": ["MUT", "NULL", "RED", "TAINT", "TWIN", "PERT", "REP", "ALT", "ORACLE", "LEAK"],
                "require_applicable": ["MUT", "NULL", "RED", "TAINT", "REP"], "require_alt_pass": True, "alt_non_builder": True,
                "heldout_survivor_rejects": True, "allow_open_suspects": False, "provisional_counts_as_certified": False,
                "max_repair_rounds": 3},
        ladder=[Step(action="repair_with_counterexamples", when=_all(_p("repairer_available", "truthy"),
                                                                        _p("repair_rounds", "lt", param="max_repair_rounds"))),
                Step(action="reject", note="the certificate keeps every reason; the archive keeps every counterexample")],
        principles=["P-valid", "P-evidence", "P-no-delete"],
        source=Source(kind="lead", who="lead", date=LEAD_1030,
                      ref="part2-paper-core §6; DESIGN §7.7 items 13, 18 (compose-2: ORACLE and LEAK count), 22; Part 1 V5 (held-out)"),
        enforced_at=["synth.certificate.AdmissionPolicy (defaults read the policy)", "synth.certificate.certify (policy_version, deviations)",
                     "synth.pipeline.admission_policy", "synth.perfect.perfect (a deviating certificate is not valid)", "synth.cegis.refine"]),
    Rule(
        id="R-holdout", title="The held-out mutant share is fixed",
        statement="h = 0.2 of each MUT neighbourhood is held out and never shown to a repairer. Held-out mutants never count toward "
                  "the bound, and h is never lowered. When fewer than 59 shown distinct mutants remain, densify the World "
                  "(re-instantiate with more siblings) and re-certify; until then the task is provisional. A held-out survivor "
                  "rejects the version.",
        applies_to=["underpowered_certificate", "heldout_survivor"],
        params={"h": 0.2, "count_heldout_in_bound": False, "max_densify": 2},
        ladder=[Step(action="densify_world", kinds=["underpowered_certificate"],
                     when=_all(_p("can_densify", "truthy"), _p("densify_tries", "lt", param="max_densify"))),
                Step(action="provisional", kinds=["underpowered_certificate"], note="admitted provisional with U reported; never certified"),
                Step(action="reject", kinds=["heldout_survivor"], note="a repair would show the held-out mutants: none is attempted")],
        principles=["P-valid", "P-more-evidence-first"],
        source=Source(kind="lead", who="lead", date=LEAD_1030, ref="runs/recert-1.md §4 (the holdout decision); DESIGN §7.7 item 22"),
        enforced_at=["synth.falsify.mut.HOLDOUT (reads the policy)", "synth.falsify.population", "synth.certificate.certify (MUT-HO has no U)"]),
    Rule(
        id="R-ALT-whole", title="ALT passes the whole task",
        statement="A long composite needs a non-builder family to pass the whole task from its instruction, never segmented. On "
                  "failure, escalate the solver ladder (the stronger configurations measured in compose-2) at equal spend with "
                  "independent seeds; then reject with the reason. The rule never changes on failure.",
        applies_to=["no_solver_pass"],
        params={"segmented": False, "change_rule_on_failure": False, "independent_seeds": True, "usd_per_config_task": 0.12,
                "max_attempts": 3,
                "ladder_rungs": [
                    {"name": "L0 certifying set", "configs": ["gpt-oss-120b:v3", "nemotron-3-super:v3", "inkling-small:v3"], "per_call": 1.5},
                    {"name": "L1 stronger models", "configs": ["nemotron-3-ultra:v2", "inkling:v3"], "per_call": 1.5},
                    {"name": "L2 long call budget", "configs": ["inkling:v3", "nemotron-3-ultra:v2"], "per_call": 3.0}]},
        ladder=[Step(action="check_oracle", when=_all(_p("truth_available", "truthy"), _p("oracle_checked", "falsy")),
                     note="a wrong oracle is its own condition (R-oracle), never a solver failure"),
                Step(action="escalate_solver_ladder", when=_all(_p("rungs_tried", "lt", param_len="ladder_rungs"), _p("budget_ok", "truthy"))),
                Step(action="reject", params={"reason": "ALT:no_pass after the solver ladder"})],
        principles=["P-valid", "P-more-evidence-first", "P-family", "P-equal-budget"],
        source=Source(kind="lead", who="lead", date=LEAD_1030, ref="DESIGN §7.7 item 18 (compose-2 decision); runs/compose-2.md §4"),
        enforced_at=["synth.solvers.escalate_alt", "synth.solvers.ladder_rungs", "synth.certificate.with_ladder"]),
    Rule(
        id="R-too-hard", title="The per-task move order",
        statement="ORACLE -> REPAIR -> RESKILL (RS1-RS6) -> DESCEND (skill-preserving knobs only; derive -> given is never a "
                  "DESCEND) or ASCEND -> KEEP or RESERVE. A task with no lawful move is reserved, never deleted.",
        applies_to=["band_too_hard", "band_too_easy"],
        params={"order": ["ORACLE", "REPAIR", "RESKILL", "DESCEND", "ASCEND", "KEEP", "RESERVE"],
                "easier_knobs": ["undo_harder_knob", "fewer_distractors", "unpaginate"], "forbidden_easier": ["derive_to_given"],
                "harder_knobs": ["distractors", "paginate", "actor", "derive"], "reskill_cap": 0.2, "reskill_attempts_per_lineage": 1,
                "kappa": 0.1, "max_rounds": 3},
        ladder=[Step(action="check_oracle", when=_all(_p("truth_available", "truthy"), _p("oracle_checked", "falsy"))),
                Step(action="repair_with_counterexamples", when=_p("unintended_with_evidence", "gt", 0),
                     note="T1-T7: only unintended outcomes with executed or calibrated evidence; never a pass rate"),
                Step(action="reskill", when=_all(_p("reskill_attempted", "falsy"), _p("deriver_available", "truthy"))),
                Step(action="descend", kinds=["band_too_hard"], when=_p("easier_knob", "in", param="easier_knobs")),
                Step(action="ascend", kinds=["band_too_easy"], when=_p("harder_knob", "in", param="harder_knobs")),
                Step(action="reserve", note="kept for stronger (or weaker) checkpoints under its honest label")],
        principles=["P-intent", "P-no-delete", "P-evidence", "P-student"],
        source=Source(kind="lead", who="lead", date=LEAD_1030, ref="runs/rsi-2-prereg.md amendment 1; runs/rsi-stats.md §9.5'''; DESIGN item 21"),
        enforced_at=["meta.rsi.TaskLoop.step (order; wiring TODO: call resolve)", "meta.rsi.TaskRules (reskill_cap, kappa, max_rounds)",
                     "meta.reskill.check_proposal"]),
    Rule(
        id="R-band", title="The learning band and its sequential test",
        statement="The keep band is p in [0.1, 0.6] on the student, measured predictor-first in looks of 4, 8, 12, 16 rollouts at "
                  "alpha 0.1. An undecided band gets more fresh rollouts up to k_max within the probe cap, else HOLD.",
        applies_to=["band_undecided"],
        params={"band_lo": 0.1, "band_hi": 0.6, "alpha": 0.1, "looks": [4, 8, 12, 16], "k_max": 16},
        ladder=[Step(action="more_rollouts", when=_all(_p("rollouts", "lt", param="k_max"), _p("probe_budget_ok", "truthy"))),
                Step(action="hold")],
        principles=["P-more-evidence-first", "P-student"],
        source=Source(kind="lead", who="lead", date=LEAD_1029, ref="part2-paper-core §10; DESIGN item 19 (band test), item 21 (looks to 16)"),
        enforced_at=["synth.learn (band test)", "meta.rsi.TaskRules (band_lo, band_hi, band_alpha, looks)"]),
    Rule(
        id="R-uncertain", title="Uncertain outcomes never decide",
        statement="A failure that is not a clean skill gap and has no executed evidence is uncertain; uncertain outcomes never "
                  "count toward too hard. Gather the task's executed evidence; if most graded rollouts stay uncertain, HOLD.",
        applies_to=["mostly_uncertain_rollouts"],
        params={"uncertain_hold": 0.5, "uncertain_counts_as_hard": False},
        ladder=[Step(action="gather_task_evidence", when=_p("task_evidence_run", "falsy")), Step(action="hold")],
        principles=["P-evidence", "P-student", "P-more-evidence-first"],
        source=Source(kind="inferred", ref="synth/outcome.py (the key rule); meta.rsi.TaskRules.uncertain_hold; runs/rsi-stats.md disagreement 4"),
        enforced_at=["synth.outcome", "meta.rsi.TaskLoop.step (uncertain_hold)"]),
    Rule(
        id="R-student-probe", title="The student may probe, never steer",
        statement="The student (Qwen3.6-35B-A3B) may run probe rollouts for band and outcome classification within the probe cap. "
                  "Its reward never enters J. RSI is a creation-time loop, unrelated to training. Past the cap, the decision "
                  "holds.",
        applies_to=["student_probe_over_cap"],
        params={"student": "Qwen3.6-35B-A3B", "uses": ["band", "outcome_classification"], "p2_cap_usd": 15.0, "k_max": 16,
                "student_reward_in_J": False, "rsi_is_creation_time": True},
        ladder=[Step(action="hold")],
        principles=["P-student"],
        source=Source(kind="lead", who="lead", date=LEAD_1029, ref="DESIGN item 21 (the per-task probe loop); factory.synth_student_wire.P2_CAP_USD"),
        enforced_at=["factory.synth_student_wire.P2_CAP_USD", "synth.learn.band_verdicts (no pass count crosses)", "meta.rsi.ProbeLoop T2"]),
    Rule(
        id="R-panel", title="An uncalibrated panel abstains",
        statement="A panel decides only at a lambda calibrated on executed-truth anchors (>= 6 decided, both truths). Uncalibrated, "
                  "it abstains: use executed evidence when it exists, else more anchors, else the human queue (the task stays on "
                  "hold). A panel verdict is necessary, never sufficient.",
        applies_to=["uncalibrated_panel", "open_suspects"],
        params={"min_decided": 6, "require_both": True, "refuse_uncalibrated": True, "target": 0.9, "floor": 0.7, "panel_sufficient": False},
        ladder=[Step(action="executed_check", when=_p("executed_check_available", "truthy")),
                Step(action="add_calibration_anchors", when=_all(_p("panel_calibrated", "falsy"), _p("more_anchors_available", "truthy"))),
                Step(action="panel_decide", kinds=["open_suspects"], when=_p("panel_calibrated", "truthy")),
                Step(action="human_queue", note="the suspect stays open; the task is on hold, not admitted")],
        principles=["P-evidence", "P-more-evidence-first", "P-family"],
        source=Source(kind="inferred", ref="DESIGN item 20 (agents.calibrate strict rule); runs/compose-2.md §6 (the LEAK split: the rule "
                                           "is not overridden); the lead's P-evidence"),
        enforced_at=["synth.falsify.agents.calibrate", "synth.policy.panel_calibration (the strict keyword set)",
                     "synth.pipeline.Runner.calibrate_panel (strict when RunConfig.panel_rule='policy')"]),
    Rule(
        id="R-oracle", title="Every task gets a correct reference decision",
        statement="A wrong oracle is repaired with its ORACLE counterexample, else rejected. A disputed oracle is checked against "
                  "the World's executed rules, else by a calibrated panel, else held for a person. A dead oracle is retried with "
                  "its errors, else dropped (recorded).",
        applies_to=["wrong_oracle", "disputed_oracle", "dead_oracle"],
        params={"max_oracle_repairs": 2, "max_oracle_retries": 3},
        ladder=[Step(action="check_oracle", kinds=["disputed_oracle"], when=_all(_p("truth_available", "truthy"), _p("oracle_checked", "falsy"))),
                Step(action="panel_decide", kinds=["disputed_oracle"], when=_p("panel_calibrated", "truthy")),
                Step(action="repair_oracle", kinds=["wrong_oracle"], when=_p("repair_attempts", "lt", param="max_oracle_repairs")),
                Step(action="retry_oracle", kinds=["dead_oracle"], when=_p("retries", "lt", param="max_oracle_retries")),
                Step(action="reject", kinds=["wrong_oracle"]),
                Step(action="drop", kinds=["dead_oracle"]),
                Step(action="human_queue", kinds=["disputed_oracle"])],
        principles=["P-evidence", "P-valid"],
        source=Source(kind="lead", who="lead", date=LEAD_1030, ref="DESIGN item 21 (the typed per-task loop: ORACLE precondition)"),
        enforced_at=["meta.rsi.TaskLoop._oracle / _repair", "synth.pipeline (oracle retries)"]),
    Rule(
        id="R-density", title="Worlds are dense enough, by execution",
        statement="Every target collection holds >= 40 entities and some reference oracle reaches n >= 59 distinct mutants. A "
                  "failing density gate re-seeds denser; another failing gate rebuilds its stage; else the instance is not used, "
                  "with its reasons. Gates are never lowered.",
        applies_to=["density_gate_failing", "world_gate_failing"],
        params={"k_min": 40, "n_min": 59, "max_reseed": 2, "max_rebuild": 2},
        ladder=[Step(action="reseed_denser", kinds=["density_gate_failing"],
                     when=_all(_p("can_reseed", "truthy"), _p("reseed_tries", "lt", param="max_reseed"))),
                Step(action="rebuild_stage", kinds=["world_gate_failing"],
                     when=_all(_p("restartable", "truthy"), _p("rebuild_tries", "lt", param="max_rebuild"))),
                Step(action="not_used")],
        principles=["P-valid", "P-more-evidence-first"],
        source=Source(kind="lead", who="lead", date=LEAD_1029, ref="DESIGN item 16 (gates; the density bar per collection is the task spec's)"),
        enforced_at=["synth.worldgates.N_MIN", "synth.families.FamilyConfig.k_min"]),
    Rule(
        id="R-budget", title="Per-run and per-stage caps",
        statement="Every run and stage has a hard cap. A plan over its stage cap is trimmed to what fits; an exhausted stage may "
                  "draw from the unallocated Part 2 total (a ledger entry, equal for every arm of a comparison); spend beyond the "
                  "Part 2 total escalates to a person.",
        applies_to=["over_budget"],
        params={"part2_total_usd": 150.0, "m4_milestone_usd": 60.0,
                "stage_caps_usd": {"worlds": 30.0, "rsi": 40.0, "yield": 35.0, "compose": 10.0, "compose2": 8.0, "perfect": 15.0,
                                   "falsifiers2": 15.0, "recert": 8.0, "rsi2_pilot": 30.0}},
        ladder=[Step(action="reallocate_within_total",
                     when=_all(_p("stage_remaining_usd", "lt", ref="min_useful_usd"), _p("part2_remaining_usd", "ge", ref="planned_usd"),
                               _any(_p("arms", "le", 1), _p("arms_equal", "truthy")))),
                Step(action="trim_plan", when=_p("stage_remaining_usd", "ge", ref="min_useful_usd")),
                Step(action="escalate", escalation="budget_beyond_total", when=_p("planned_usd", "gt", ref="part2_remaining_usd"))],
        principles=["P-person", "P-equal-budget", "P-ledger"],
        source=Source(kind="lead", who="lead", date=LEAD_1029, ref="part2-paper-core §14 (spend beyond $150 needs the lead); DESIGN items 13-22 caps"),
        enforced_at=["factory.synth_wire (WORLDS_CAP_USD, RSI_CAP_USD, YIELD_CAP_USD read the policy)", "factory.synth_student_wire.P2_CAP_USD",
                     "scripts/synth*.py stage caps"]),
    Rule(
        id="R-scale", title="The near-term target",
        statement="10 environments x 10 tasks (configurable), from at least 4 World families. A shortfall is closed by synthesizing "
                  "more within budget, never by relaxing admission; otherwise it is reported.",
        applies_to=["scale_shortfall"],
        params={"environments": 10, "tasks_per_environment": 10, "min_families": 4},
        ladder=[Step(action="build_more_families", when=_all(_p("families", "lt", param="min_families"), _p("budget_ok", "truthy"))),
                Step(action="synthesize_more", when=_p("budget_ok", "truthy")),
                Step(action="report_shortfall")],
        principles=["P-valid", "P-more-evidence-first"],
        source=Source(kind="lead", who="lead", date=LEAD_1029, ref="runs/rsi-stats.md (the 10 x 10 pool); runs/worlds-1.md (4 or 5 families)"),
        enforced_at=["synth.policy.scale_target"]),
    Rule(
        id="R-family", title="No family checks its own work",
        statement="Builders never solve-as-reference, attack, judge or derive on their own tasks; the student's family (Qwen) is in "
                  "no role. A same-family assignment is reassigned, else the step holds.",
        applies_to=["same_family_check"],
        params={"student_families": ["qwen"], "no_self_check": True,
                "checker_roles": ["ALT", "ATK", "panel", "leak_panel", "intent_deriver", "clarity_checker", "realism_rater", "blind_tester"]},
        ladder=[Step(action="reassign_family", when=_p("alternative_family_available", "truthy")), Step(action="hold")],
        principles=["P-family"],
        source=Source(kind="inferred", ref="synth/roles.py (differs_from, independent, family_problems); DomainEnv.built_by; the swarm-builds rule"),
        enforced_at=["synth.roles.family_problems", "synth.pipeline.Runner.refine_one (ALT/ATK exclude builders)", "synth.solvers.escalate_alt"]),
    Rule(
        id="R-equal-budget", title="Improvements are claimed at equal spend",
        statement="An improvement is claimed only against a control at the same spend (within the comparison's tolerance); "
                  "otherwise re-run the control at equal spend, else make no claim.",
        applies_to=["unequal_budget_comparison"],
        params={"equal_budget_required": True, "budget_tolerance": 0.35},
        ladder=[Step(action="rerun_control_equal_spend", when=_all(_p("control_rerun_possible", "truthy"), _p("budget_ok", "truthy"))),
                Step(action="no_claim")],
        principles=["P-equal-budget"],
        source=Source(kind="inferred", ref="part2-paper-core §7 and §9.4 R9; frontier.compare (tol 0.35); meta.lemmas (0.05)"),
        enforced_at=["synth.frontier.FrontierPolicy.budget_tol", "synth.batch (one cap per arm)", "synth.solvers.evaluate (equal per-task USD)"]),
    Rule(
        id="R-rsi-power", title="An underpowered update is not accepted",
        statement="An update is decided on fresh units at alpha_t against delta_AA. Underpowered: draw a larger fresh view if the "
                  "decision split has the units; else no update (the incumbent stays). Units are never reused, delta never moves "
                  "down.",
        applies_to=["underpowered_decision"],
        params={"alpha": 0.05, "reuse_units": False, "delta_moves_down": False},
        ladder=[Step(action="larger_fresh_view", when=_p("units_available", "ge", ref="units_needed")), Step(action="no_update")],
        principles=["P-more-evidence-first", "P-valid"],
        source=Source(kind="lead", who="lead", date=LEAD_1029, ref="runs/rsi-stats.md; runs/rsi-2-prereg.md (independent seeds, delta_AA)"),
        enforced_at=["meta.update.UpdateEngine (paired mode)", "meta.stats.AlphaSpender"]),
    Rule(
        id="R-reserve", title="RESERVE replaces RETIRE",
        statement="A request to delete a task, World or record is resolved as RESERVE: kept for stronger checkpoints, out of the pool.",
        applies_to=["delete_request"],
        ladder=[Step(action="reserve")],
        principles=["P-no-delete"],
        source=Source(kind="lead", who="lead", date=LEAD_1030, ref="runs/rsi-2-prereg.md amendment 1"),
        enforced_at=["meta.rsi.TaskLoop (lineage status reserved)"]),
    Rule(
        id="R-seed", title="Seeds by tag",
        statement="Every model call is seeded from its tag (salted per run and client), so repeated attempts are independent samples.",
        params={"seed_by_tag": True},
        principles=["P-evidence", "P-equal-budget"],
        source=Source(kind="lead", who="lead", date=LEAD_1029, ref="DESIGN items 19, 22"),
        enforced_at=["factory.synth_wire.WireClient / SynthWire (seed_by_tag=True default)"]),
    Rule(
        id="R-patch-scope", title="Repository tasks: the recorded exemption",
        statement="Repository (PatchTask) certificates run MUT, NULL, RED, TAINT, REP; ALT and ATK are paid and were not run in M5, "
                  "so the ALT-pass rule is not required there. This is the one recorded exemption from R-admission-fixed; it is "
                  "scoped to repository certificates and changes only through the ledger.",
        params={"scope": "repository", "zero_confirmed_classes": ["MUT", "NULL", "RED", "TAINT", "REP"], "require_alt_pass": False},
        principles=["P-valid", "P-ledger"],
        source=Source(kind="inferred", ref="DESIGN item 14 (PATCH_POLICY drops the ALT-pass rule, recorded); falsify/patch.py"),
        enforced_at=["synth.falsify.patch.PATCH_POLICY (scope='repository')"]),
    Rule(
        id="R-sealed", title="Sealed and held-out data",
        statement="Sealed data is read only through its report channels (purpose='report', Thresholdout). Any other access is a "
                  "person's decision.",
        applies_to=["sealed_access"],
        ladder=[Step(action="escalate", escalation="sealed_access")],
        principles=["P-person"],
        source=Source(kind="inferred", ref="factory.synth_patch_wire.Pool (data/sealed refused); meta.rsi locked set; meta.stats.Thresholdout"),
        enforced_at=["synth.perfect.load_pool (refuses sealed roots)", "meta.rsi (purpose='report')"]),
    Rule(
        id="R-irreversible", title="Irreversible and external actions",
        statement="An irreversible or external action (deleting data, pushing, publishing, contacting a third party) is a person's "
                  "decision.",
        applies_to=["irreversible_action"],
        ladder=[Step(action="escalate", escalation="irreversible_action")],
        principles=["P-person", "P-no-delete"],
        source=Source(kind="lead", who="lead", date=LEAD_1030, ref="the policy-layer brief (escalation grounds)")),
    Rule(
        id="R-principles", title="Principles change only by a person",
        statement="A change to a principle (or to a fixed validity parameter) is a person's decision, recorded as a ledger entry.",
        applies_to=["principle_change"],
        ladder=[Step(action="escalate", escalation="principle_change")],
        principles=["P-person", "P-ledger"],
        source=Source(kind="lead", who="lead", date=LEAD_1030, ref="the policy-layer brief (escalation grounds)")),
    Rule(
        id="R-default", title="A new blocking condition",
        statement="A blocking condition no rule names is resolved by the principles alone: run the executed check the facts say "
                  "exists, else HOLD with the condition recorded (it becomes a rule through the ledger).",
        applies_to=["*"],
        ladder=[Step(action="executed_check", when=_p("executed_check_available", "truthy")), Step(action="hold")],
        principles=["P-more-evidence-first", "P-no-delete", "P-person"],
        source=Source(kind="inferred", ref="the principles (P-more-evidence-first, P-no-delete, P-person)")),
]


def genesis_entries() -> list[dict[str, Any]]:
    """The ledger's opening entries (no chain fields: the ledger adds seq, parent and sha)."""
    out: list[dict[str, Any]] = [{"kind": "principles", "principles": [p.model_dump(mode="json") for p in PRINCIPLES],
                                  "source": Source(kind="lead", who="lead", date=LEAD_1030, ref="the policy-layer brief").model_dump(mode="json"),
                                  "rationale": "the principles every rule derives from and every remedy is checked against"}]
    for r in RULES:
        out.append({"kind": "add_rule", "rule": r.model_dump(mode="json"), "source": r.source.model_dump(mode="json"),
                    "rationale": f"codifies {r.source.ref}"})
    return out


__all__ = ["PRINCIPLES", "RULES", "genesis_entries"]
