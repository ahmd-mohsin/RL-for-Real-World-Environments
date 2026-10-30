# policy-1: policy as data, and the lawful-remedy procedure (2026-10-30)

Code: DESIGN §7.7 item 24. `synth/policy/` (`model.py`, `genesis.py`, `ledger.py`, `resolve.py`, `ledger.jsonl`), the
ALT solver ladder in `synth/solvers.py` (`escalate_alt`) with `certificate.with_ladder`, and the wiring listed in §4.
Tests: `tests/unit/test_synth_policy.py`. No model call and no Tinker spend.

**The lead's requirement.** "Whatever decisions you are asking me, we need them built into the Part 2 framework as its
mechanism." Until now each blocking condition (a task with too few mutants, a panel that cannot decide, a composite no
solver passes, a band that is too hard) became a question to the lead, answered one at a time and then coded by hand.
This note gives the proposed text of a new §9.7 of `part2-paper-core.md` (not edited here), then the policy as built,
the gaps the build found, and the wiring left for files other agents own.

---

## §9.7 Policy as data: the lawful-remedy procedure (proposed text)

A synthesis framework meets blocking conditions that its rules do not settle by themselves: a certificate whose bound
needs more probes than the World supports, an adjudication panel that is not calibrated, a long task no reference solver
passes, a task too hard for the student. A pipeline that settles them ad hoc moves its own goalposts: the easiest
remedy for a blocked task is usually to relax the rule that blocked it, and each relaxation raises yield. We make every
such decision data and give one procedure that resolves blocking conditions from stated principles.

**Principles.** An ordered list Π = (π₁, …, π₁₀). Each principle is a short statement and a test `test_π(m, c, P)` that
returns its objections to a candidate remedy m for a condition c under policy P (empty: no objection).

| # | Principle | Its test refuses |
|---|---|---|
| π₁ | **P-valid.** Never relax a validity or admission rule to raise yield. | a change to a validity parameter in its loose direction; counting held-out trials; segmenting a whole-task rule; admitting with open suspects |
| π₂ | **P-evidence.** Prefer executed evidence; a panel verdict is necessary, never sufficient; an uncalibrated panel abstains. | admission on a judgement alone; a panel decision while the panel is uncalibrated |
| π₃ | **P-more-evidence-first.** An underpowered decision first gathers more evidence. | (the order of the procedure) |
| π₄ | **P-intent.** The skill and cell are frozen except through an evidence-derived RESKILL. | a skill change outside RESKILL; a RESKILL not derived from executed evidence; derive → given as an easier move |
| π₅ | **P-no-delete.** Nothing is deleted. | deletion |
| π₆ | **P-student.** Student probes may inform task decisions within caps; student reward never enters J. | student reward in J; a repair justified by a pass rate; rollouts past the cap |
| π₇ | **P-equal-budget.** Every claimed improvement is compared at equal spend. | a claim without an equal-spend control; a budget change that favours one arm |
| π₈ | **P-family.** No family checks its own work. | a builder's or the student's family as solver, attacker, judge or deriver |
| π₉ | **P-person.** A person decides only budget beyond the total, sealed or held-out data access, irreversible or external actions, and changes to the principles. | an escalation on any other ground; any of those four effects without one |
| π₁₀ | **P-ledger.** Every rule change is a versioned ledger entry; every run records its policy version. | a remedy that changes a rule parameter (except a stage cap inside the total) |

**Rules.** A rule ρ = (id, K_ρ, θ_ρ, L_ρ, Π_ρ, source, version) is the codified form of one decision already taken:
the condition kinds K_ρ it applies to, its parameters θ_ρ (for example h = 0.2), its ladder L_ρ (an ordered list of
steps), the principles Π_ρ it derives from, and its source (the person who decided and the date, or the code and run
note an inferred rule was read from). A step is (a, φ, K′): an action a, a predicate φ over the condition's facts, and
optionally the subset K′ ⊆ K_ρ it serves.

**Closed vocabularies.** Three sets live in code because code implements them, and change only with the code:
- the actions A, each with a rung r(a) ∈ {evidence, repair, relabel, variant, defer, escalate}, its declared effects,
  and the call site that carries it out (34 actions);
- the validity parameters V ⊆ (rule × parameter), each with its strict direction (lower, higher, true, false, superset,
  subset, fixed): π₁'s data;
- the escalation grounds G = {budget beyond the total, sealed access, irreversible action, principle change}: π₉'s data.

**Conditions.** A condition c = (kind, subject, F) where F are facts the caller measured: availability and counts
(`can_densify`, `shown_n`, `panel_calibrated`, `rungs_tried`, `planned_usd`), never a judgement. A predicate on a
missing fact is false, so resolution falls toward later, safer rungs (fail closed).

**The procedure.** For a condition c, a policy P and optional proposed remedies X (a model's or a script's suggestion):

```text
resolve(c, P, X):
  R  ← {ρ ∈ P : kind(c) ∈ K_ρ}   or, if empty, {R-default}           # the principles alone
  C  ← [(ρ, s) : ρ ∈ R, s ∈ L_ρ, kind(c) ∈ K′_s] ++ X
  sort C stably by rung: evidence < repair < relabel < variant < defer < escalate
  for m in C:
      if m is a step and not φ_m(F): skip                              # inapplicable on the facts
      build m as a Remedy                                              # relaxing V cannot be built: π₁ by construction
      if ∃ π ∈ Π: test_π(m, c, P) ≠ ∅: record the objections; skip
      return m (with every skipped candidate and why)
  return HOLD                                                          # lawful under every π
```

**Properties.** Each holds by construction and is checked by a test (§5).
1. *No relaxation.* No returned remedy changes a validity parameter in its loose direction: a `Remedy` carrying such a
   change raises at construction, and π₁'s test refuses declared relaxing effects. The ledger (below) extends this across
   versions: no version relaxes V except through a person's answer to a principle-change escalation. So the admission
   rule a run applies is never looser than the one before it unless a person changed a principle, and the ledger names
   who and when.
2. *Evidence first.* A remedy past the evidence rung is returned only when every evidence step for c was inapplicable on
   F or refused by a principle.
3. *Totality.* `resolve` always returns: HOLD is lawful under every principle (asserted at run time).
4. *Bounded escalation.* `resolve` returns an escalation only when its ground in G is borne out by F (for example
   `planned_usd > part2_remaining_usd`); an escalation on any other condition is refused by π₉, and an `Escalation`
   outside G cannot be constructed.
5. *Auditability.* `resolve` is a pure function of (c, P, X). The remedy records the policy version, the rule, the
   action's call site, and every candidate passed over with its reason.
6. *Closed actions.* A proposed remedy whose action is not in A is refused; a rule whose ladder runs out of rung order or
   names an unknown action is not a rule.

**Escalation and versioning.** An escalation is a typed record (id, ground, condition, question, options, the policy
version it was asked under). Every option is a ledger amendment or none. The person's answer is one ledger entry of
kind `answer` holding the escalation, the choice and its amendment, so even a "no" is a new, recorded version. The
answer never becomes a code change.

**The ledger.** Rules and principles live in an append-only, hash-chained ledger; a policy version is a ledger position
(`policy-v<n>@<sha>`), and every certificate and run report records the version it ran under. The ledger refuses:
- an entry that relaxes a validity parameter or changes a principle, unless it is a person's answer to a principle-change
  escalation (this binds the lead too: a relaxation is a principle-level decision, made explicitly);
- a change to the Part 2 total, unless it answers a budget escalation;
- from an inferred source: anything but adding a rule, tightening a validity parameter, or raising a stage cap within
  the recorded remainder of the total; retiring a rule (rules are retired, never removed) is a person's decision.

**What the procedure does not do.** It does not measure: facts come from executed checks, and a weak check stays weak.
It does not run the remedy: the action's call site does, under the same certificate and archive rules as before. It
cannot invent a remedy: the action vocabulary is closed. A blocking condition no rule names (the R-default case) is held
with its condition recorded, which is the cue for a new rule entry.

**The rules at genesis** (22 rules; the ledger's first 23 entries; source and parameters in `synth/policy/genesis.py`).

| Rule | Source | Applies to | Ladder (first applicable, lawful) |
|---|---|---|---|
| R-admission-fixed | lead, 2026-10-30 (§6; compose-2; V5) | confirmed counterexample | repair with counterexamples (≤ 3 rounds, monotone archive) → reject |
| R-holdout | lead, 2026-10-30 | underpowered certificate; held-out survivor | densify the World and re-certify (≤ 2) → provisional; survivor → reject |
| R-ALT-whole | lead, 2026-10-30 | no solver pass | check the oracle → solver ladder L0 → L1 → L2 (equal spend, independent seeds) → reject "ALT:no_pass after the solver ladder" |
| R-too-hard | lead, 2026-10-30 | band too hard / too easy | ORACLE → REPAIR → RESKILL → DESCEND (skill-preserving knobs) or ASCEND → RESERVE |
| R-band | lead, 2026-10-29 | band undecided | more fresh rollouts to k_max = 16 within the probe cap → HOLD |
| R-uncertain | inferred (outcome.py) | mostly uncertain rollouts | gather the task's executed evidence → HOLD |
| R-student-probe | lead, 2026-10-29 | probe cap reached | HOLD |
| R-panel | inferred (item 20; compose-2 LEAK split) | uncalibrated panel; open suspects | executed check → more anchors → calibrated panel → human queue |
| R-oracle | lead, 2026-10-30 | wrong / disputed / dead oracle | executed rules → calibrated panel → repair / retry → reject / drop / human queue |
| R-density | lead, 2026-10-29 | density gate; other World gate | re-seed denser → rebuild the stage → not used |
| R-budget | lead, 2026-10-29 | over budget | reallocate inside the total (equal across arms) → trim → escalate beyond the total |
| R-scale | lead, 2026-10-29 | scale shortfall | more families → more synthesis → report the shortfall |
| R-family | inferred (roles.py) | same-family check | reassign → HOLD |
| R-equal-budget | inferred (§7, R9) | unequal comparison | equal-spend control → no claim |
| R-rsi-power | lead, 2026-10-29 | underpowered update | larger fresh view → no update |
| R-reserve | lead, 2026-10-30 | delete request | RESERVE |
| R-sealed, R-irreversible, R-principles | lead / inferred | the three person grounds | escalate |
| R-seed, R-patch-scope | lead / inferred | (constraints) | (parameters only) |
| R-default | the principles | any other kind | executed check → HOLD |

**The solver ladder (R-ALT-whole).** The rule stays: a non-builder family passes the whole task from its instruction.
On `ALT:no_pass`, the ladder runs the stronger configurations compose-2 measured, rung by rung, on the whole task:
L0 the certifying set (gpt-oss, Nemotron-3-Super, Inkling-Small; harness v3; 1.5 × horizon + 6 calls), L1 the stronger
models (Nemotron-3-Ultra v2, Inkling v3), L2 the long call budget (3 × horizon). Every configuration of a rung gets the
same per-task budget (USD 0.12), each attempt has its own tag `ALT:<task>#<config>@L<k>~<j>` (independent seeds), builder
and student families are skipped, and the ladder stops at the first rung with a pass. Failed attempts go to the
adjudicator: a valid failed attempt is a confirmed D6, an undecided one a suspect; `certificate.with_ladder` re-decides
the certificate under its own admission rule, so a ladder pass after an unjudged failure is not admitted. With no pass
the task is rejected with the reason.

**What the paper reports.** The policy version of every result; the ledger (entries, sources, answers); per run, the
count of each remedy by rule and action, the escalations raised and how they were answered; and the conformance tests
that pin each code constant to its rule.

---

## 1. As built

| Part | What it is |
|---|---|
| `policy/model.py` | `Principle`, `Rule`, `Step`, `Condition`, `Remedy`, `Escalation`, `ParamChange`, `Source`; `ACTIONS` (34, rung, effects, call site), `VALIDITY` (37 parameters with directions), `UNSAFE_EFFECTS`, `AUTO_PARAMS` (stage caps), `PERSON_PARAMS` (the Part 2 total), `relaxes`. `PolicyViolation` is not a `ValueError`, so pydantic never hides it |
| `policy/genesis.py` | the 10 principles and 22 rules as data, each with its source |
| `policy/ledger.py` | `PolicyLedger` (append-only JSONL, sha256 chain, `check_entry` guards, `answer`), `EscalationQueue` (asked once; answering appends a row), `replay`, `current()`; `python -m crucible.synth.policy.ledger --genesis` |
| `policy/ledger.jsonl` | the shipped ledger: 23 entries, `policy-v23@e2834a86285f` |
| `policy/resolve.py` | `resolve`, `check`, `grounds`, `holds` (the predicate language: eq ne lt le gt ge in truthy falsy present absent, `all`/`any`/`not`, operands `value`/`ref` (a fact)/`param`/`param_len`), `PRINCIPLE_CHECKS`, `CONDITIONS` (25 kinds with their facts), `escalation_for` |
| `policy/__init__.py` | accessors the code reads: `tag`, `param`, `admission_params(scope)`, `admission_deviations`, `panel_calibration`, `holdout`, `stage_cap`, `part2_total`, `scale_target` |
| `solvers.py` | `LadderRung`, `ladder_rungs()` (from the ledger), `escalate_alt`, `LadderOutcome`; `attempt(sink=)` |
| `certificate.py` | `AdmissionPolicy` defaults read R-admission-fixed; `scope`; `Certificate.policy_version`, `deviations`; `decide` (the rule over stats) and `with_ladder` |

## 2. Gaps the build found (each fixed, with a test)

1. **The pipeline's panel decided while uncalibrated.** `Runner.calibrate_panel` used the M4 rule: when no λ reached
   the target, λ became the grid's largest (0.95) and the panel still decided suspects. That breaks P-evidence. Now
   `RunConfig.panel_rule = "policy"` (the default) calibrates with R-panel's strict rule (≥ 6 decided anchors of both
   truths, else λ > 1: the panel abstains, suspects go to the human queue); `"m4"` keeps the old rule for measurement.
   One existing test (`test_per_part_repair_repairs_only_the_owner_part_and_recomposes`) calibrates a two-family fixture
   panel on too few anchors and now passes `panel_rule="m4"`, since it tests per-part repair, not calibration.
2. **The pipeline's held-out share defaulted to 0.** R-holdout fixes h = 0.2. `RunConfig.heldout_mutants` now defaults
   to the ledger's h; 0 remains available for a measurement and is recorded in the run's config.
3. **The certificate's default admission policy lacked ALT, ORACLE and LEAK.** Only `pipeline.admission_policy()` had
   them; a direct `certify(task, results)` admitted a confirmed ORACLE finding. The defaults are now the rule, and
   `admission_policy()` returns `AdmissionPolicy()`.
4. **Looser policies were silent.** Many measurements and tests pass `AdmissionPolicy(u_max={}, require_alt_pass=False)`.
   That stays possible, but the certificate now records `deviations` and `policy_version`, and `perfect.load_pool` keeps
   a deviating certificate out of the pool.
5. **The repository exemption had no name.** `PATCH_POLICY` drops the ALT-pass rule (DESIGN item 14, "recorded"). It is
   now R-patch-scope, an inferred rule scoped to `scope="repository"`: the one exemption, visible in the ledger, changed
   only through it. Whether repository certificates should run ALT once paid solvers exist is a principle-level choice
   the ledger will record when made.
6. **Budget constants were scattered.** Nine stage caps lived in `factory/synth_wire.py`, `factory/synth_student_wire.py`
   and five scripts. They now read R-budget (same values). The M4 milestone default (USD 60, `MILESTONE_CAP_USD`) and
   the Part 2 total (USD 150, from the environment) are both parameters of R-budget; the code's environment override is
   unchanged.

## 3. Unchanged on purpose

- No stored certificate is rewritten; old ones load with `policy_version = None`.
- The `R-ALT-whole` ladder is an escalation before rejection, not a new admission path: a ladder pass only adds ALT
  passes, and every failed ladder attempt must be adjudicated or it stays a suspect.
- `resolve` decides nothing about tasks by itself; a call site raises a condition with measured facts and carries out
  the returned action.

## 4. Wiring for files other agents own (to apply after they merge)

`meta/rsi.py` (TaskLoop):
- `TaskRules` defaults read the ledger: `band_lo = policy.param("R-band", "band_lo")`, `band_hi`, `band_alpha =
  param("R-band", "alpha")`, `looks = tuple(param("R-band", "looks"))`, `reskill_cap`, `kappa`, `max_rounds` from
  R-too-hard, `uncertain_hold` from R-uncertain (today equal; pinned by `test_r_too_hard_and_r_band_conform_to_taskloop`).
- `TaskLoop.step`: after the ORACLE precondition, raise `Condition("wrong_oracle" | "disputed_oracle", facts={truth_available,
  oracle_checked, panel_calibrated, repair_attempts})`; for the uncertain hold `Condition("mostly_uncertain_rollouts",
  facts={task_evidence_run})`; after the band, `Condition("band_undecided", facts={rollouts, probe_budget_ok})` or
  `Condition("band_too_hard" | "band_too_easy", facts={truth_available, oracle_checked, unintended_with_evidence,
  reskill_attempted: lin.reskilled, deriver_available: self.derive_intent is not None, easier_knob / harder_knob: the first
  knob's name})`, and dispatch on `resolve(c).action` (`repair_with_counterexamples` → `_repair`, `reskill` →
  `_try_reskill`, `descend`/`ascend` → the knob branch, `reserve`/`hold` → the status). Record `remedy.policy_version`
  and `remedy.rule` on `TaskDecision`, and save `resolve`'s `considered` in the ledger record. The skipped test
  `test_taskloop_resolves_through_the_policy` names this site.
- `archive_recheck(heldout=)` default: `policy.holdout()` instead of 0.0 when called from `TaskLoop`.
- `TaskLoop._refuse` (RESERVE after `max_refusals`) is R-too-hard's last rung; `rollback_reskill` stays (RS6).

`recert.py`:
- line 251: the phase-1 policy `admission_policy().model_copy(update={"require_alt_pass": False, "allow_open_suspects":
  True})` is a deliberate deviation for phase 1 only; certificates written from it now carry `deviations`. Add
  `scope="recert-phase1"` and a ledger rule if the lead wants it named like R-patch-scope.
- line 291: `calibrate(p, anchors, floor=0.7)` → `calibrate(p, anchors, **policy.panel_calibration())` (R-panel; today the
  M4 rule, so an uncalibrated panel decides at 0.95).
- `ManifestRow`: add `policy_version` (the tag the re-certification ran under).
- `det_verdict`/the statuses: an `underpowered` provisional row raises `Condition("underpowered_certificate", facts={shown_n,
  can_densify, densify_tries})` so the manifest carries R-holdout's remedy (densify) beside the status.

`families.py` / `worldgates.py`:
- `worldgates.N_MIN = policy.param("R-density", "n_min")`; `FamilyConfig.k_min` default `policy.param("R-density",
  "k_min")` (today equal; pinned by `test_r_scale_r_density_r_family_r_seed_conform`).
- The gate runner: a failing density gate raises `Condition("density_gate_failing", facts={can_reseed, reseed_tries})`, any
  other failing gate `Condition("world_gate_failing", facts={gate, restartable, rebuild_tries})`; record the remedy in
  `gates.json`.
- `worldgates.calibrate_panel` (the realism panel): use `policy.panel_calibration()`.

## 5. Tests (`tests/unit/test_synth_policy.py`: 65 pass, 37 of them the per-parameter P-valid cases; 1 skipped wiring TODO)

| Rule / property | Test |
|---|---|
| ledger | the shipped ledger is the genesis and its chain verifies; tampering breaks replay; every lead decision is a rule with its source; an out-of-order ladder or unknown action is no rule |
| P-valid by construction | a relaxing `Remedy` cannot be built, for each of the 37 validity parameters |
| ledger guards | a relaxation is refused even from the lead and accepted only as a person's principle-change answer; inferred entries may tighten, not loosen or retire; the total changes only by a budget answer |
| resolve, property | 400 random (condition, facts, proposed remedies incl. relaxing ones): no returned remedy relaxes validity, has an unsafe effect, fails a principle, changes a rule parameter other than a stage cap, or escalates outside the four grounds |
| P-more-evidence-first, property | 300 random (condition, facts): a remedy past the evidence rung only after every evidence step was passed over |
| R-default | an unknown condition: executed check, else HOLD |
| R-ALT-whole | resolve order and the reject reason; a segmenting remedy is refused; the ladder on scripted solvers: rung 0 fails, rung 1 passes, rung 2 never runs, builder skipped, tags carry the rung; `with_ladder` admits only with failures adjudicated (unjudged: suspect; valid: D6) |
| R-too-hard | ORACLE → REPAIR → RESKILL → DESCEND → RESERVE; derive → given never descends; ASCEND for too easy; a skill change refused; conformance with `TaskRules`/`TaskMove` |
| R-band, R-uncertain | more rollouts within cap, else HOLD; task evidence, else HOLD |
| R-holdout | h is the ledger's in MUT, MUTHeldOut, population; held-out mutants never count in U; survivors reject; densify → provisional |
| R-admission-fixed | the default policy is the rule; ORACLE, LEAK, ALT confirmed reject; suspects reject; ALT pass required; certificates record the version; a looser policy is a recorded deviation and leaves the pool; PATCH_POLICY is exempt only in its scope |
| R-panel | the strict calibration abstains on too few anchors; executed check → anchors → human queue; an uncalibrated panel decision is refused; the pipeline is strict by default |
| R-student-probe | the P2 cap is the ledger's; `BandVerdict` carries no pass count; reward-in-J and pass-rate repairs refused; no rollouts without the cap |
| R-budget | wire caps are the ledger's; trim / reallocate / escalate; equal arms; the answer raises the total as a new version; a stage cap rises only within the recorded remainder; a declined answer is a version too |
| R-scale, R-density, R-family, R-seed, R-equal-budget, R-rsi-power, R-oracle, R-reserve, R-sealed, R-irreversible | ladders and conformance with `worldgates.N_MIN`, `FamilyConfig.k_min`, the wire's `seed_by_tag`, `roles.family_problems` |
| run reports | `Runner` records the policy version |
