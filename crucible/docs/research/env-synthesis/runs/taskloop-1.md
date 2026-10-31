# taskloop-1: denser Worlds for the held-out-blocked tasks, and the first real run of the per-task loop (2026-10-30)

Code: DESIGN §7.7 item 23. `synth/densify.py`, the finance_ap pack's `world.draw(extra=)` and `studio.APTask(extra=)`,
`factory/synth_wire.py` (`finance_ap_world(extra_invoices=)`, `finance_ap_denser`, the truth keyed by density),
`synth/meta/loopkit.py` (`Recertifier`, `TaskKeyedContext`, `label_cell`, `DomainClassifier`), `synth/meta/rsi.py`
(TaskLoop: P1, the policy wiring of item 24, concurrent rounds, stored diagnosis, the pilot's fixes),
`synth/meta/reskill.py` (the deriver's written values), `synth/pipeline.py` (`make_repairer`), `synth/recert.py`
(`det_run`, `Item.task_file`, `ManifestRow.policy_version` and `remedy`, the named phase-1 deviation), CLI
`scripts/synth_taskloop.py` (`densify`, `manifest`, `loop`, `report`) and `scripts/synth_recert.py` (strict panel
calibration, `--densify-tries`). Tests: `tests/unit/test_synth_taskloop_pool.py`, additions to
`test_synth_taskloop.py` and `test_synth_policy.py` (the TaskLoop wiring test, un-skipped).

Data (gitignored, under `crucible/data/synth/`): `recert/20261030-153839-pool-2/` (densify, re-certification, the first
updated manifest), `recert/20261030-161708-pool-2b/` (the 7 re-measured renamed tasks; **the pool manifest TaskLoop ran
on: `manifest.pool-2b.pool.json`**), `recert/20261030-dense-logs/` (logs), `taskloop/20261030-162426-taskloop-pilot/`,
`taskloop/20261030-163717-taskloop-1/`. New certificates are written once beside the stored ones as
`tasks/tNN/certificate.pool-2.json` and `certificate.pool-2b.json` (never overwritten).

Models: ALT Nemotron-3-Super, gpt-oss-120b and Inkling-Small (v3); ATK Super + gpt-oss; panels Nemotron-3-Ultra +
gpt-oss; leak probe Inkling-Small; repairer DeepSeek (a builder); intent deriver gpt-oss (the abstractor's non-builder
family); the student Qwen3.6-35B-A3B on Tinker, only as the probe. No Qwen model in any other role. No sealed or
benchmark task was read; no task was edited by hand.

**Policy versions.** Densify and the pool-2 re-certification (phase 1 and phase 2) ran before item 24 merged: no policy
version (`policy_version = ""` in their manifest rows), with recert-1's panel calibration (λ floor 0.7, M4 rule). The
pool-2b re-certification, the pilot and taskloop-1 ran under `policy-v23@e2834a86285f` (strict R-panel calibration,
TaskLoop resolving through the policy).

## Part A: denser Worlds

### A.1 What "a denser World of the same family" has to be

The brief's route, "the same seed with more invoices (`--invoices N`)", does not keep the task: the pack's `draw(seed, n)`
plans all n invoices, then shuffles the plans, so `draw(0, 60)` is a different World from `draw(0, 30)`. The invoice a
task names (INV-90575, key 126) is another invoice with another decision there. So the re-instantiation needs a
**superset**: `draw(seed, n, extra=k)` generates the World of (seed, n) unchanged, then k more pending invoices from a
second random stream (`.../extra`), each with an ordinary invoice's base rates and only NEW vendors, POs, receipts and
earlier invoices. Tests pin it: the base Worlds' digests are unchanged (golden hashes taken before the change), every
base row and every base decision (`decide`) is unchanged in the superset, and no base vendor gains an invoice (so a
vendor's invoice list, which duplicate checks read, is the same). `synth.densify` is family-generic: a `Densifier` maps
a `WorldRef` and a level to a superset WorldRef; finance_ap's is `synth_wire.finance_ap_denser` (`extra_invoices`), and
the policy truth (ORACLE) draws the denser World by (seed, invoices, extra).

`densified(task, ref)` is the next version with only the World replaced; instruction, intent card, contract and
reference program are asserted equal. `check_dense` runs π* on both Worlds: it must pass its contract on the denser one,
with the same effect and answer, and every observation must be the same or only grown (a list holding the original
elements and more). A changed observation marks the task density-sensitive (a finding; nothing is edited).

### A.2 Measured, not guessed: the cap binds before density helps

On one held-out-blocked task (`yield-1/a0 … s0.k1-b0-2`, a single-invoice decision, 30 invoices), MUT's shown split at
h = 0.2:

| extra invoices | `max_mutants` 240: shown n | held-out n | all n | `max_mutants` 480: shown n | probes generated |
|---|---|---|---|---|---|
| 0 | 48 | 18 | 60 | 53 | 354 |
| 15 | 51 | 22 | 66 | — | — |
| 30 | 46 | 25 | 51 | 64 | 388 |
| 45 | 51 | 16 | 36 | — | — |
| 60 | — | — | — | 73 | 412 |

At the certificate's 240-mutant cap, density does not raise n (it even lowers the whole-neighbourhood n): the round-robin
over operator families fills the cap with sibling swaps of read steps, which mostly share one effect (the read finds
nothing, no write), so more siblings mean more duplicate effects inside a fixed budget. A single-invoice decision's whole
neighbourhood on these Worlds is 350–430 mutants, so the dense certificates use `max_mutants = 480` (recorded in each
certificate's config; never bound: `cap_bound` is false on every chosen level). The cap alone, without densifying,
lifts only 3 of the 60 tasks over 59 (median shown n 53 at level 0, cap 480). Both are needed.

### A.3 The levels chosen (levels 15, 30, 45, 60, 90 extra invoices; the smallest whose shown split reaches 59)

| chosen level | tasks |
|---|---|
| +15 | 47 |
| +30 | 8 |
| +45 | 2 |
| +60 | 1 |
| +90 | 1 |
| none reaches 59 | 1 |

Shown n at the chosen level: 59–79; held-out n median 21. About 21 s of CPU per level per task (8 workers: 8 minutes for
the 60). No task was density-sensitive: every observation was the same or grew, every effect and answer the same, and
the policy truth agreed with every pinned decision on the denser World.

**Findings (recorded, the tasks not edited):**
1. *A task density cannot help* (`yield-1/a1_lemmas … s3.k0-b0-1`): shown n 40 → 50 → 50 → 50 → 49 → 47 over levels
   0–90. Its reference program reaches the invoice through a ref (`s1.invoices.0.id` after `find_invoices(vendor, number)`),
   so SwapEntity only swaps the vendor (one shared effect), and ExtraWrite's distractors are a seeded sample of
   `MAX_SIBLINGS = 64` keys of the whole invoice collection (pending and historical): the neighbourhood is capped by the
   sampling constant, not by the World. It stays provisional. Raising `MAX_SIBLINGS` would change MUT for every task (and
   RSI-2's preregistered numbers), so it is not changed here.
2. *A false density alarm, fixed.* One task's π* reads `kb:ap_policy.md` (a wrong path): the error message names the
   World's own work directory, which differs per World, so `check_dense` called the step changed. Errors are now compared
   with filesystem paths masked (test).
3. *Renaming changes MUT's split.* The first pass measured levels under each task's old id, then gave 46 tasks a unique
   id (below). MUT's held-out bucket and its sibling sample are keyed by the task id, so 7 renamed tasks certified at
   shown n 56–58 (provisional) although measured at ≥ 59. `densify` now renames first; the 7 were re-measured under their
   final ids (levels +30 ×5, +60, +90: a second densify try, within R-holdout's `max_densify = 2`) and re-certified
   (`pool-2b`): all 7 certified.

**Unique ids.** 19 task ids are held by 46 tasks of different runs (yield-1/2/3, smoke-1, speed runs). TaskLoop keys
lineages, the archive's regressions and the student's seeds (`student:<id>@<v>:<i>`) by task id, so two tasks would
share an archive history and identical student samples. Every such task gets `<run>/<id>` (e.g.
`yield-2/a1/finance_ap.s0.k0-b0-0`), its archived counterexamples re-keyed into the new archive, and a full
re-certification under the new id (its MUT split and TWIN sample change with the id).

### A.4 Full re-certification of the densified versions

All classes, h = 0.2, the monotone archive replayed from the recorded traces (recert-1's per-source archives copied and
re-keyed), real seeds by tag. 76 items in pool-2 (59 densified, 17 renamed only), 7 in pool-2b.

| | the 60 held-out-blocked tasks |
|---|---|
| **certified** | **53** |
| provisional | 1 (finding 1: density cannot help) |
| on hold | 2 (an undecided ATK pass; an undecided LEAK quote) |
| rejected | 4: `TWIN:confirmed` 2 (D9; the twin is another invoice of the *base* World, e.g. INV-93933 → INV-16784: the new version's TWIN sample drew a sibling the old one had not, so the defect was there before), `ALT:no_pass` 2 (no pass from Super, gpt-oss or Inkling-Small on the re-sample) |

The renamed-only tasks (not in the 60): one stored-certified task falls to `TWIN:confirmed` under its new id (the same
mechanism: TWIN's sample is keyed by the id); the rest keep their status class. The pool manifest TaskLoop starts from
(`manifest.pool-2b.pool.json`, 195 tasks): **57 certified, 32 provisional, 10 on hold, 96 rejected** (recert-1: 4 / 90 /
10 / 91). Every provisional row now carries R-holdout's remedy (`underpowered_certificate` → `densify_world` while tries
remain, else `provisional`) when re-certified under the policy.

Spend for Part A: USD 2.87 (pool-2 phase 2 2.46 + 0.16 for the 5 rows its cap cut; pool-2b 0.25), all under the recert
stage cap (8; recert-1 had used 5.11).

## Part B: the first real run of the per-task loop

`synth_taskloop.py loop --manifest manifest.pool-2b.pool.json --cap 20 --student-cap 13 --rounds 3 --concurrency 8`
(`taskloop/20261030-163717-taskloop-1`), after a pilot on 8 tasks (`…-taskloop-pilot`, USD 1.11) that found three
framework problems (§B.6). All 195 training-side tasks (57 certified, 32 provisional, 10 on hold, 96 rejected), one
typed decision per active lineage per round, rounds 1–3. Order of moves as the policy resolves it: ORACLE → P1 (the
certificate) → REPAIR → (band) → RESKILL / DESCEND / ASCEND → KEEP or RESERVE; HOLD where the evidence does not decide.

Student probes: Qwen3.6-35B-A3B (JSON agent v2, 20 calls, T = 1.0, seeds by tag). Diagnosis: 8 rollouts; for 87 tasks
whose exact version (id, version, digest) was measured by the learnability run, those stored rollouts are the round-1
diagnosis (prereg §8.1: none re-bought). The band is the sequential test in looks of 4, 8, 12, 16 (R-band); every
acceptance draws fresh rollout indices (G4). The outcome classifier is `synth.outcome` per domain with the strictly
calibrated panel (finance_ap λ 0.7, TWIN λ 0.5, leak λ 0.7; all calibrated); uncertain never counts as a defect or as
too hard. Every edit is fully re-certified (142 re-certifications in the run).

### B.1 Moves

| move | decisions | accepted | final lineages |
|---|---|---|---|
| REPAIR | 74 | **3** (all P1: rejected tasks repaired from their shown counterexamples) | — |
| RESKILL | 25 accepted (+ 33 refused attempts inside other decisions) | **25** (2 kept, 23 reserved under the new label) | — |
| DESCEND | 0 | 0 | — |
| ASCEND | 52 (27 lineages) | **0** | — |
| KEEP | 17 | 17 | **kept 19** (17 + 2 re-labelled) |
| RESERVE | 23 | — | **reserved 71** |
| HOLD | 35 | — | **held 85**, **rejected 20** |

- *Repaired* 3: `a1/finance_ap.s3.k0-b0-0` (22 shown MUT D9 counterexamples: the repairer pinned the status update
  to invoice key 101, the invoice INV-47820 names), `a1_lemmas/finance_ap.s3.k2-b0-0` (TWIN D9: the contract pinned only
  the invoice number of the changed row; the repair pins the row the instruction names), `yield-1/a1/finance_ap.s2.k2-b0-2`
  (the densified task TWIN rejected in Part A). Each re-certified, then 4 fresh rollouts showed no recurrence (G4).
  71 repairs were refused (decisions; a decision can cite several rules): T5 (the candidate's re-certification fails)
  53, G2 (P3 on the candidate: a TWIN or TAINT finding, or P3 unmeasured) 25, the repairer declined 16, T4 (what justified
  it still passes) 7, T3 (an intended pass now fails) 2, T2 (an intended fail now passes) 1 (issue 11), G1 1. Refused or declined, an invalid task is **held**, never reserved or kept.
- *Re-labelled* 25 (RS1–RS6 each passed: a DomainSpec skill, evidence-only derivation, P3 and leak and full
  re-certification under the new card, novelty in the new cell, the trust region and the 20% cap, one attempt). From →
  to (cells): `unmapped → three_way_match` 7, `unmapped → vendor_risk` 4, `three_way_match → vendor_risk` 3,
  `policy_threshold → three_way_match` 2, `policy_threshold → vendor_risk` 2, `duplicate_check → three_way_match` 2,
  `three_way_match → ap_figure` 1, `unmapped → ap_figure` 1, `duplicate_check → vendor_risk` 1, `batch_review →
  vendor_risk` 1. E.g. `a0/finance_ap.s2.k1-b0-1` "AP invoice decision under policy" → `vendor_risk` (π* holds the
  invoice for bank verification); fresh band too easy: reserved under its honest label. Refused attempts: RS3 18
  (mostly a re-sampled ATK or ALT suspect in the candidate's re-certification), RS4 15 (not novel in the new cell).
- *Descended* 0: the 10 too-hard tasks are plain tasks; R-too-hard's easier knobs (undo a harder knob,
  fewer_distractors, unpaginate) need a variant World, so they were reserved ("no lawful move makes this task easier").
- *Ascended* 0: finance_ap's SQL World takes neither declared harder knob that needs a perturbable World (distractors);
  `paginate` applied on 27 lineages and never moved the band (52 decisions; a decision can cite several rules: G4 20,
  still too easy on 16 fresh rollouts; G2 28, P3 unmeasured on the paginated variant ("no valid EVID, SWAP or TWIN
  trial") or a TWIN finding; G3 14, its re-certification; G6 10, no independent solver passed it). 15 too-easy tasks had
  no applicable harder knob at all and were reserved.
- *Kept* 19: 11 certified and 6 provisional at KEEP, plus 2 re-labelled into the band.
- *Held* 85: invalid tasks whose repair was refused or declined (≈ 70), on-hold certificates (R-panel → human queue,
  10), disputed reference decisions on it_helpdesk (R-oracle → human queue, 3: e.g. 8 student runs set ticket T-1006's
  escalation reason to `vip_request`, the oracle `sla_breach`, and it_helpdesk has no executable rules), mostly uncertain
  rollouts (R-uncertain, 2). *Rejected* 20: rejected versions with no counterexample a repairer may see (only held-out
  mutants or `ALT:no_pass`): R-admission-fixed's last rung.

Per round: round 1 decided all 195 (41 min); round 2 the 29 active lineages (25 re-tried ASCEND, 1 RESKILL, 3 KEEP);
round 3 two. Every decision records the policy version (`policy-v23@e2834a86285f`), the rule and each condition raised
with its facts and the rungs passed over (`remedies`).

### B.2 The band, before and after

| | too hard | in band | too easy | not measured |
|---|---|---|---|---|
| before (the original version's diagnosis band, 90 valid tasks reached the band) | 10 | 22 | 58 | 105 (invalid, held or on hold before the band) |
| after (final lineages) | — | **19 kept** | — | 71 reserved (too easy or too hard under an honest label), 85 held, 20 rejected |

The learnability run (perfect-1) found 47 of 89 too hard; on the re-certified pool most of those are invalid (wrong
oracles, TAINT, TWIN): of the valid tasks only 10 are too hard and 58 too easy for this student.

**A RESKILL changes the band verdict without changing the task** (finding). 7 in-band tasks were re-labelled (the
instruction, World and contract unchanged): on RS4's 16 fresh rollouts 2 stayed in band and 5 came out too easy (2) or
too hard (3), so they were reserved. The relabel cannot change the student's pass rate; this is the band test's error at
k ≤ 16 (0.29–0.37 at the edges, item 19) acting on a second measurement. For the lead: under P-more-evidence-first a
disagreement between the diagnosis band and RS4's band could raise `band_undecided` (more rollouts, else HOLD) instead
of placing the task by the second measurement alone. Not changed in this run.

### B.3 The drift alarm and unexplained pass-rate changes

Re-probed repaired tasks: 3 (the alarm needs ≥ 5): unexplained changes 0.125, 0.0, 0.0. The alarm did not fire and is
not testable at this size. No pass-rate change reached a decision: repairs were accepted on T1–T7 and re-certification,
variants on the direction of a fresh band (logged as knob calibration: 27 ASCEND attempts, 0 moved).

### B.4 Lessons lifted to the system, and the R1–R10 gate

- **PROBE counterexamples**: 52, one natural cluster (`hidden_info`, should pass) on 7 tasks: the learnability run's
  unintended fails (an exact free-text note or JSON answer the instruction never states). All 7 tasks are already
  rejected by TAINT (D5) in their certificates, so the cluster is no headroom: an operator lifted from it would catch only
  what TAINT confirms. None was proposed.
- **World-builder fixes**: 3 `oracle_review` records (the it_helpdesk disputes above): it_helpdesk has no executable
  policy truth, so ORACLE cannot settle a dispute there (rsi-stats disagreement 5).
- **Lemma candidates**: the 3 accepted repairs (pin the entity the instruction names, not a field of it; 2 are TWIN D9
  fixes). `lift_lemmas` is not wired, so none entered the lemma bank.
- **The gate was not run, and could not accept anything at this size.** A lifted operator is tested on fresh decision
  units of other tasks at α_t = 0.005 (t_max 10, prereg §5). The exact McNemar minimum detectable recall gain at power
  0.8: n = 30 → 0.33, 52 → 0.20, 60 → 0.17, 120 → 0.085, 240 → 0.045. The only cluster has 52 members on 7 tasks
  (cluster-correlated: the effective n is nearer 7), and it is already covered by TAINT; so H1–H3 are not testable from
  this run's lessons.

### B.5 Cost and time

| item | USD |
|---|---|
| student rollouts (2,444 drawn by the loop's decisions; 696 diagnosis rollouts were stored, not bought) | 7.45 |
| swarm (ALT 2.39, panels 1.32, repairer 0.77, outcome residue 0.54, ATK 0.22, deriver 0.09, calibration 0.12, LEAK 0.01) | 4.84 |
| **taskloop-1** | **12.29** |
| pilot (8 tasks) and the two one-task re-runs after the fixes (§B.6) | 1.11 + 0.07 + 0.26 |
| Part A (pool-2, pool-2b re-certifications) | 2.87 |
| **this work** | **16.60** of the 28 cap |

**Perfect tasks** (kept, certified, on intent by P3, novel in their cell): **8**, all densified finance_ap tasks from
Part A. USD 1.54 per perfect task for the loop; USD 2.07 with Part A, the pilot and the re-runs. 15 kept tasks are
valid (certified or provisional), on intent and novel. Wall-clock: 51 minutes for the loop (round 1: 41 min for 195
tasks at 8 concurrent steps), plus about 25 minutes of CPU for Part A's measurement and phase 1.

### B.6 Framework problems the real runs found, and their fixes (each with a test)

| # | found by | problem | fix |
|---|---|---|---|
| 1 | Part A | `--invoices N` re-draws the World: the task's invoice is another invoice | superset density `draw(extra=)` (§A.1) |
| 2 | Part A | at MUT's 240-mutant cap density does not raise n | dense certificates use 480 (recorded; never bound) |
| 3 | Part A | renaming after measuring changed MUT's split (7 tasks certified at n 56–58) | `densify` renames first; the 7 re-measured |
| 4 | Part A | a false density alarm from World paths inside error messages | errors compared with paths masked |
| 5 | wiring | task ids repeat across runs; the loop's archive and student seeds are keyed by id | unique `<run>/<id>`, re-keyed archives, re-certified |
| 6 | wiring | concurrent steps on tasks of one World instance would share one World | `TaskKeyedContext`; G7/RS5 reservations under a lock |
| 7 | wiring | a rejected task would be banded and KEPT (the loop never read the certificate) | P1 precondition (`validity`) |
| 8 | pilot | a deriver whose answers all fail RS2 (it proposed field names: `invoice_id`) crashed the step | a refused RESKILL; the deriver sees `written_values`, two retries |
| 9 | pilot | 8 student runs agreeing on another decision held a task whose oracle the World's rules confirm | R-oracle: executed rules settle it (`overruled_dispute`) |
| 10 | pilot | an inapplicable harder knob counted as a refused edit and re-probed the task every round | the next declared knob, else RESERVE |
| 11 | run | a wrong-oracle repair was refused by T2: the student's "intended fail" was the policy's right decision, graded against the wrong oracle | labels decided under a wrong reference decision are void in its repair; re-run (`taskloop-1c`): the task is repaired, then reserved (too easy; no applicable knob) |
| 12 | run | a variant refused under G4 was re-tried on fresh rollouts next round (25 lineages, 352 rollouts, USD ~1.3): re-measuring until a pass | a refused knob is not re-tried on its lineage |
| 13 | re-run | `--only` calibrated the panels on the selected tasks only (one task: no anchors, every panel abstained, the repair held on suspects) | panels calibrate on the whole pool's valid tasks |

### B.7 What does not work yet (honest list)

- DESCEND and ASCEND do nothing useful on finance_ap: its World cannot be perturbed (no distractors), and pagination
  does not make these single-invoice decisions harder for this student. Ladders need knobs the World supports.
- P3 is often unmeasured on finance_ap variants and candidates (no valid EVID, SWAP or TWIN trial: EVID needs a
  perturbable World); G2 then refuses, fail closed (28 of 52 ASCEND decisions and 25 of 71 refused repairs cite G2).
- Full re-certification re-samples ALT and ATK: a label-only change (RESKILL) or a repair can fail on a fresh ATK or ALT
  suspect the panel does not decide (RS3 18, T5 on-hold cases).
- The drift alarm and the R1–R10 gate are underpowered at this size (§B.3, §B.4).
- One held-out-blocked task stays provisional because MUT's sibling sample (`MAX_SIBLINGS`), not the World, caps n.
