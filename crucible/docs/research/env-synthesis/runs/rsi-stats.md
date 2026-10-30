# RSI statistics: a rigorous update test, run-level error control, power, headroom, and the per-task probe loop

Date: 2026-10-29. Code: `crucible/src/crucible/synth/meta/{stats,update,rsi,headroom,ablate,par}.py`; tests
`tests/unit/test_synth_rsi_stats.py`, `tests/unit/test_synth_rsi_loop2.py`; real-data CLI `scripts/synth_rsi2.py`.
Data: `crucible/data/synth/rsi/rsi-2/stats/` and `rsi-2/rsi2-pilot-*` (gitignored). This file is the proposed text for
part2-paper-core §9 (marked **Proposed §9 text** below) plus the evidence behind it. The preregistration of RSI-2 is
`runs/rsi-2-prereg.md`.

## 0. Results in one table

| Item | Result |
|---|---|
| **δ_AA vs RSI-1's δ** | A/A, 5 replicates (seed pairs (1,2)…(9,10)) on the rsi-2 dev pool: replicate mean differences −0.016, 0, 0.008, 0, 0.008 on 125 units, **δ_AA = 0.0196**; on RSI-1's own 75-unit calibration view **δ_AA = 0.0327**. RSI-1's between-view δ was **0.0407**. Per-unit A/A discordance 0.64%: the seed changes 0–2 of 125 verdicts (MUT's sibling sample on it_helpdesk `eq_to_any` plants). |
| **RSI-1's decisions, re-tested (report only)** | 3 of RSI-1's 14 evaluated decisions can be reconstructed exactly (the views from the sampler's seeds, matched by hash; J matches the ledger to 4 decimals); the other 11 had no stored operator spec. The best (`mutate_written_value_other_seen`, the one that "missed R2 by 0.003"): 7 gains, 0 losses on 75 units, mean 0.093, bootstrap 95% [0.040, 0.160], cluster bootstrap [0.028, 0.159]. The paired betting test against δ_AA = 0.0327: e-value 4.2 (needs 60 at α/3, 20 even uncorrected); anytime LCB −0.004 at α = 0.05. At independent seeds: mean 0.067–0.093, e-values 3.4–4.2. **The paired machinery also refuses it, more firmly**: RSI-1's normal LCB (0.038) was an approximation that does not hold for 7 discrete gains in 75; the binding constraint was the sample, not δ. The two other candidates: 2 and 3 gains, e-value 1.0. |
| **Run-level error (simulation)** | 0.000 for T = 5–40 null candidates (§3), against 0.20–0.80 for the uncorrected per-test rule |
| **Power (per class, locked)** | 33 paired plants detect a 0.2 recall gain at power 0.8 (71 for 0.1); rsi-2 locked: D6 12 plants (MDE 0.50), D7 18 (0.36), D10 13 (0.47); D1, D3, D4, D8 have 1–2 plants (no detectable effect). Raising `max_per_op` to 6 adds 9 plants (81): the cap is not the constraint, the source tasks are. Full table in `rsi-2-prereg.md` §5. |
| **Headroom (dev)** | 7 of 125 dev plants are missed by the population (all `relax_value:eq_to_any`, D3, MUT suspects only); **all 7** are caught by some expressible operator with no anchor alarm (e.g. `mutate_arg(*, written_key values) expect fail confirm always`): **headroom 5.6% absolute, 100% relative**; 34,132 of 37,236 grammar operators raise no anchor alarm; 7,625 distinct programs executed. Unchanged by DESIGN item 20 (the new population's dev misses are the same 7). |
| **Headroom (locked, after item 20)** | the new population catches 72/72 rsi-2 locked plants: the old locked sets have no headroom and can no longer measure a gain. The four RSI-2 D3 variants (`held_out_planter_v2`) restore it: see §0.1. |

### 0.1 The locked set after DESIGN item 20

The item-20 population (MUT with read mutants, layered TAINT) catches every plant of the rsi-1 and rsi-2 locked sets.
A locked set at 100% recall can show a regression but never a gain. Five new held-out D3 variants were added to the
planter (`held_out_planter_v2`; `relax_value:eq_to_regex`, `eq_to_ne`, `approx_rel`, `where_in`, `any_last`, disjoint
from the dev variants). On the fixture tasks they plant witnessed defects the population misses (MUT reaches suspects
only: `test_the_v2_held_out_d3_variants_...`). On the 7 real rsi-2 locked sources they do not:

| locked set | planter | plants | population recall | headroom | post-hoc gain of RSI-1's 3 gate-passing operators |
|---|---|---|---|---|---|
| rsi-2 (old) | held_out_planter, cap 2 | 72 | 72/72 (item 20's run) | 0 | — |
| rsi-2c | held_out_planter, cap 6 | 81 (measured before item 20 merged) | — | — | — |
| rsi-2v2 | v2 (4 D3 variants), cap 6 | 83 | 83/83 | 0 | 0, 0, 0 |
| rsi-2v3 | v2 + `any_last`, cap 6 | 84 (D3 4) | 84/84 | 0 | 0, 0, 0 |

**Finding.** On today's sources the only planted defects the population misses are the dev set's 7
`relax_value:eq_to_any` plants on it_helpdesk tasks (MUT suspects that need the panel). No held-out variant tried (18
in all) reproduces that gap on the locked sources. **The locked set cannot measure a gain on the current pool; it
measures non-regression only (gate G).** What would change this: (1) the 10 × 10 pool's 30 locked tasks (more task
shapes, more chances for the panel-only D3 gap); (2) planters found by the P loop with the headroom reward, on dev
first; (3) with MUT's suspects adjudicated by the panel (the M4 `Adjudicated` wrapper), even the dev gap closes and
dev headroom is 0. RSI-2's preregistration therefore makes headroom a precondition (prereg §6).

## 5. The pilot on Tinker: what broke, and the fixes

All runs: `scripts/synth_rsi2.py`, the rsi-2 dev pool (125 plants, 10 dev sources; decision split 5 sources, 64 units),
the abstractor and planter proposer gpt-oss-120b (no Qwen; the student never called), receipts under
`data/synth/rsi/rsi-2/rsi2-pilot-*`.

| # | Run / step | Problem found | Fix |
|---|---|---|---|
| 1 | A/A on the dev pool (CPU) | TWIN confirms a D9 on the clean it_helpdesk source `t02` at seeds 3 and 6 (π*′ fails C′ for twin T-1007): an anchor that alarms at some seeds makes R3's FA ≤ 0 a lottery, and its plants were "caught" by the source's own alarm | per-seed anchor re-check before calibration (`Loop.clean_anchors`: the anchor is dropped and reported); **source-relative hits** (a class that alarms on a plant's clean source at the same seed is not credited on that plant) |
| 2 | A/A, serial | one base-population pass over 125 plants took 435–550 s | `meta/par.ShardPool` (sharded `spawn` workers with per-seed caches) |
| 3 | locked rebuild | the shared `--domains` default planted dev sources into a "locked" set (`rsi-2b`) | `--held-out-domains`, refused when it overlaps dev; the invalid set renamed `*.INVALID-overlaps-dev.json` |
| 4 | smoke1 (1 round, 4 candidates, $0.0027) | the A/A view was the 60%-sized calibration view: 37 units, δ_AA 0.045 (too noisy, not the decision size) | an `aa` view of `n_max` proposal units |
| 5 | smoke1 | 0/4 candidates passed the birth gate (none reproduced its cited sources), 0/3 planters accepted (2 not novel, 1 reward 0): no decision reached R2, no α spent | none needed: the rule worked; the pool simply has 4 proposal-split misses |
| 6 | smoke1 | the archive grew by 93 counterexamples from a parallel batch run while the loop ran | none: `refresh_natural` is channel (b) working; the abstractor sees them as natural clusters |
| 7 | p2 | two arms filled one headroom cache concurrently: a table was read while its programs were still running (KeyError) | a lock per cache |
| 8 | p2 | a ShardPool per arm measured every shared unit twice | one pool for both arms |
| 9 | p3 | two arms measured the locked set concurrently through one report context: a file-backed World was reset under the other (SnapshotMismatch) | report measurements serialised; each evolve arm gets its own Worlds |
| 10 | item 19 (main) | every sampled call used seed 0, so repeated calls were copies | the proposers' clients seed by tag, with a salt per run and per E10 seed. The A/A numbers are unaffected (the falsifier seeds are CPU-side, no model call); smoke1's single proposal call is unaffected |
| 11 | retro | a killed parent left its 10 shard workers computing | killed by pid; workers exit at their next pipe read |

<!-- PILOTRES -->

## 1. What was wrong with RSI-1's update test

| Problem | Consequence |
|---|---|
| δ was the spread of J across *different* views (between-view sampling noise) | The test statistic is a *paired* difference on one view, whose noise is far smaller. The bar measured the wrong noise. |
| "Fresh" views were overlapping 60% draws from one fixed pool of 125 plants | R1 held only for view hashes: every decision re-read most of the same plants, and the abstractor's miss clusters were drawn from those same plants |
| Each candidate was tested at a one-sided 95% normal LCB, with no correction | Candidates × rounds is a multiple-testing problem. With T looks at noise the family-wise error grows like 1 − 0.95^T. |
| The meta loop accepted A′ on gate pass rates of 2/2 vs 0/2 | The exact one-sided p-value of 2/2 vs 0/2 is 1/6. Those two acceptances were not evidence. |
| The locked set had 1 plant for D1, D3 and D4 | A per-class recall change was undetectable: the minimum detectable effect at n = 1 does not exist |

## 2. Proposed §9 text

### 9.4′ R1 and R2, replaced

**Proposed §9 text.** *Units and the null.* An update u: Θ → Θ′ is evaluated on n decision units (planted tasks). Unit i
gives D_i = Y_i(Θ′, s′) − Y_i(Θ, s) ∈ [−1, 1], with s and s′ two independent evaluator seeds (the falsifiers' random
choices: MUT's sibling sample, NULL's random calls, TWIN's twins, PERT's perturbations). The null of the test is

> H_u: E[D] ≤ δ_AA, the update is no better than the unchanged Θ is against itself.

*The noise floor, δ_AA.* Before the loop, the unchanged Θ is evaluated against itself on one fresh view with k ≥ 5
disjoint pairs of independent seeds (A/A replicates). Each replicate gives the mean paired difference d̄_r. The floor is
δ_AA = max(2·sd(d̄_r), max_r |d̄_r|). It is measured at the decision view's size, is never tuned, and only moves up (a
recalibration keeps the maximum). M3's between-view δ is still computed and reported (`delta_spread`), but it is not the
bar: the decision statistic is paired, so the reference noise must be the paired noise of an update that changes
nothing. The A/A distribution is exactly the null distribution of the decision statistic for Θ′ = Θ.

*The test.* R2 holds when a one-sided test by betting (Waudby-Smith & Ramdas, JRSS-B 2024) rejects H_u at level α_t.
On X_i = (D_i + 1)/2 and m = (δ_AA + 1)/2 the wealth K_n = Π_{i≤n} (1 + λ_i (X_i − m)), with predictable bets
λ_i ∈ [0, c/m] (aGRAPA, c = 3/4), is a non-negative supermartingale under H_u. Ville's inequality gives
P(sup_n K_n ≥ 1/α_t) ≤ α_t, so the test is valid at every stopping time. The evaluation therefore runs in chunks of fresh
units and stops as soon as the wealth crosses 1/α_t (accept R2), or when, after n_fut units, the mean gain is at or below
δ_AA (refuse: futility only lowers power). Units never evaluated are released and stay fresh. R2's anytime lower
confidence bound (the largest null mean the test still rejects) is reported with the paired-bootstrap 95% interval and
the cluster bootstrap by source task (plants of one task are correlated).

*R3–R7* compare Θ′ and Θ at the *same* seed (common random numbers), exactly as M3 did. *R9* is a second betting test,
of J′ − J_control, at the same α_t.

*R1, strict.* A share of each domain's dev source tasks is reserved as the **decision split**. Decision views draw their
units without replacement from the decision split's units that no earlier decision consumed. The decision split is never
shown to a proposer, the birth gate or the planter loop. Calibration, A/A and audit views draw from the proposal split.
A planter's new plants on decision sources join the decision split. Every α spend and every unit reservation is written
to the ledger (a `test` record) **before** the units are evaluated, so a crashed round never gets its α or its units back.

### 9.4″ Error control over a whole run

**Proposed §9 text.** Every decision that can accept something is a test t with a level α_t fixed before its data are
drawn, from one spender per loop run (per arm): online Bonferroni, α_t = α / t_max for t ≤ t_max (t_max preregistered),
0 afterwards; or, open-ended, α_t = α·6/(π² t²). **Theorem.** If each test t is valid conditional on everything before it
(P(reject_t | F_{t−1}) ≤ α_t whenever H_t is true), then

> P(some non-improving update is accepted in the run) ≤ Σ_t P(reject_t, H_t true) = Σ_t E[1{H_t} P(reject_t | F_{t−1})] ≤ Σ_t α_t ≤ α.

The union bound needs no assumption on the dependence between tests, and the candidate of test t may depend on
anything before it (the abstractor's proposals, the previous outcomes). Conditional validity comes from Ville's
inequality for the betting test, given that the units of test t are exchangeable draws that nothing before t has seen:
that is what R1-strict guarantees. Alpha-investing and online e-BH control the false discovery rate, not this
family-wise error, so they are not used.

**What the guarantee covers.** "Non-improving" means E[D] ≤ δ_AA on the decision-split population at the time of the test
(planted defects of the current and every past planter, on dev source tasks the proposers never saw). It does not by
itself cover generalisation to other domains, other World families or other operator variants. Those are the
generalisation gates (9.4‴).

**What else spends α.** R9 (same α_t as R2); a prune (a non-inferiority test, H: E[J′ − J] ≤ −δ_AA, on fresh units; a
prune buys cost, not J, and must show it costs no more than the floor); the meta loop A (a one-sided Fisher exact test of
the birth-gate pass rate of A′ against A, only with ≥ 4 candidates a side: 2 against 2 can never reach significance); the
lemma gate of S (its McNemar or Newcombe test at the α_t it is handed). **What does not.** An audit (it can only roll
back, never accept: tested at a fixed 0.05) and planter acceptance (witness validity, novelty, natural-likeness and
reward are executed checks, not an improvement claim).

### 9.4‴ Generalisation is the acceptance test

**Proposed §9 text.** A system-level update is accepted only if, beyond R1–R10:
- **R7** its gain appears on dev domains that produced none of its evidence;
- **G (locked set)** the locked set does not regress. Any decision that reads held-out data goes through a reusable
  holdout (Thresholdout, Dwork et al., Science 2015): the gate asks for the locked recall gain of Θ′ over Θ and receives
  the dev value unless the locked value differs by more than a noisy threshold, in which case it receives a noisy
  locked value and the holdout's budget drops by one. Every access is logged (`holdout_access.jsonl`: the query, the
  purpose, whether the holdout was revealed, the budget left, never the held-out value). Budget spent: the gate fails
  closed. G is off by default (no loop reads the locked set); RSI-2's preregistration turns it on.
- **The held-out World family** (a family no round's evidence came from) is measured every round and reported; no
  decision reads it. It is the headline measure of generalisation.

### 9.3′ The P/F game produces headroom

**Proposed §9 text.** *Headroom* is the share of dev plants the current population misses that some operator expressible
in the DSL catches with no false alarm on the dev anchors (`meta/headroom.py`; the grammar is the DSL on the random
control's constants: 6,206 bodies × 6 label rules = 37,236 operators). It bounds what one round of DSL operators can add to J. A planter is
rewarded for headroom, not for misses alone: it joins the population only if its novel plants on proposal sources contain
a miss some expressible operator could catch. A miss nothing expressible catches is a DSL blind spot, reported, but it
gives F nothing to learn. The game's trajectory (F's recall, the misses, headroom, the planter rewards, the pool's class
KL and MMD to the real-defect reference) is reported every round.

### 9.5′ The per-task probe loop (RSI at task-creation time)

**Proposed §9 text (replaces "the frontier loop" and every mention of RL training as an input).** RSI runs when tasks are
created, not after training:
1. a task is created and admitted by its certificate;
2. the student runs on it as **probe rollouts** (not training);
3. each outcome is classified: **intended pass** (the skill was used), **intended fail** (a genuine skill gap),
   **unintended pass** (a shortcut, a leak, a vacuous check, a frame hole, a hack path), **unintended fail** (an ambiguous
   instruction, hidden information, an over-strict check, a World or tool bug, contradictory docs, an unsolvable variant);
4. unintended outcomes with executed or calibrated evidence become counterexamples (an executed one is the student's own
   program with the verdict the task should have given, class PROBE in the archive) or panel-adjudicated suspects, and
   drive a repair of the task, its World instance or its contract through CEGIS under the monotone archive;
5. the lesson is lifted: the counterexample enters the archive and, through natural clusters, the falsifier loop; the
   repair becomes a candidate design lemma for the writers; a World or tool bug becomes a World-builder fix;
6. this repeats, round after round.

The regularisers of a per-task repair (all checked, non-compensatory):

| # | Rule | What it prevents |
|---|---|---|
| T1 | **Evidence only.** A repair needs ≥ 1 unintended outcome with executed or calibrated evidence. A judged-only or evidence-free "unintended" failure is demoted to an unattributed fail and treated as a skill gap; such a pass goes to the panel queue. A label that contradicts the trusted grader's verdict is void. | Repairs justified by opinion; tuning by pass rate |
| T2 | **Never ease a skill gap.** Every intended (and unattributed) fail's program is replayed on the repaired task; one that now passes refuses the repair. | Making tasks easier for the student |
| T3 | **Never over-constrain.** Every intended pass's program is replayed; one that now fails refuses the repair. | A repair that adds a D6 |
| T4 | **Close what justified it.** Every unintended pass must now fail; every executed unintended fail of a valid program must now pass. | Cosmetic repairs |
| T5 | **Monotone archive.** No archived counterexample re-opens (student-probe ones included); the deterministic classes stay clean. | Forgetting |
| T6 | **Caps.** ≤ 3 repairs per task; calibrated-only (judge) evidence stops after 2 repairs without new executed evidence. | Self-verification saturation (R8) |
| T7 | **Independent attribution.** A failure that claims the task cannot be solved as stated (ambiguous, hidden information, unsolvable, contradictory docs) is demoted when an ALT solver of a non-student, non-builder family solved the same task. | Calling a skill gap a task defect |

**J and every acceptance gate never use the raw student pass rate.** The repair decision is a function of the counted
outcomes and the replays of their programs; no aggregate pass rate reaches it, and a rollout carrying one is refused.
**The student-tuning drift monitor** reads pass rates for one purpose, an alarm: on re-probed repaired tasks,
d = p_after − (passes_before − unintended_passes)/n_before; a fixed unintended fail explains a rise of up to
unintended_fails/n_before; the unexplained part u = d − clip(d, 0, that bound) is averaged over tasks, and the alarm fires
when at least 5 tasks give a mean |u| above 0.1 with a 95% bootstrap interval that excludes 0. An alarm stops the loop's
task repairs for review; nothing is accepted because of it.

**System-level updates** lifted from the probe loop (falsifier operators, lemmas, World-builder fixes) are accepted only
under 9.4–9.4‴: a fresh-unit paired test at an α_t of the run's spender, R7 on held-out domains, the locked set through
the reusable holdout, the held-out World family reported.

**Interfaces consumed.** The student-probe runner is the learnability agent's `StudentProbe` (`rollouts(task, n, start)`
→ graded `Rollout`s with a replayable literal program). The outcome classifier (`OutcomeClassifier.classify(task,
rollouts)` → `ProbeOutcome`: index, the grader's verdict, kind, mechanism, defect, evidence ∈ executed / calibrated /
judged / none, `alt_solved` and its family, the program) is the learnability agent's `synth.outcome`, adapted by
`rsi.LearnClassifier`: its `uncertain` label is never a task defect (an uncertain fail becomes an unattributed fail, an
uncertain pass a suspect), `decided_by` = execution gives executed evidence and = panel calibrated evidence, and the
certification's ALT solvers are T7's independent solver. `meta/rsi.py` enforces T1–T7 whatever the classifier emits.
Measured with the real classifier on a planted D1 task and three scripted rollouts
(`test_the_probe_loop_consumes_the_learnability_classifier`): the full solution is an intended pass, the shortcut that
skips the customer email an unintended pass ("lax requirement", decided by execution), a wrong article an intended fail
("skill_gap: decision"); the shortcut is archived as a PROBE counterexample and the repair is accepted under T2–T5. The
online monitor during RL training is a separate safety net and is **not** an input to this loop.

### 9.5″ The typed decision per task (`rsi.TaskLoop`)

**Proposed §9 text.** "Works for all tasks" means every task gets a correct, typed decision, not that every task is
admitted. Per task and round:

| Decision | When | Action |
|---|---|---|
| (precondition) **ORACLE** | before any difficulty move | the reference decision is checked against the World's executed rules (`falsify.oracle.ORACLE`, a falsifier class: a registered policy truth says `wrong` → confirmed D6, op `policy`) and against independent runs (`decision_disagreement`: ≥ 2 runs agree on another value of a pinned decision field they actually changed) |
| **REPAIR** | the oracle is wrong; or an unintended fail or pass with executed or calibrated evidence | CEGIS repair of the oracle, contract, World instance or instruction; T1–T7 |
| **HOLD** | the reference decision is disputed (panel review queued); more than half the graded rollouts are uncertain; the band test does not decide at k_max | nothing is edited |
| **DESCEND** | intended fails, band too hard | one declared knob step easier → a new variant; the original is kept as the harder rung |
| **ASCEND** | intended passes, band too easy | one declared knob step harder (distractors, paginate, actor, derive, depth) → a new variant |
| **KEEP** | in band, valid | the lineage is kept |
| **RETIRE** | no declared knob in the needed direction; a move against the lineage's direction; 2 refused edits; the round cap | out of the band pool (the task and its archive stay) |

| # | Regulariser | As built |
|---|---|---|
| G1 | Evidence-gated edits | a REPAIR cites a counted unintended outcome (T1) or the ORACLE counterexample; a move cites a declared knob and its direction |
| G2 | Frozen intent | the variant or repair keeps the skill and the cell and passes the P3 probes (`p3_check`: `intent.p3`, EVID SWAP TAINT TWIN); else refused |
| G3 | Full re-certification under the monotone archive | `recheck` (archive regressions incl. PROBE and ORACLE counterexamples, the deterministic classes) after every edit |
| G4 | Sample splitting | the diagnosis's rollout indices are never reused: acceptance draws fresh indices (new seeds), sequentially in looks of 4, 8, 12, 16 (k_max) through `learn.decide`; a repair is accepted only if no unintended outcome of the same mechanism recurs on the fresh rollouts; a variant only if its fresh band moved in the declared direction |
| G5 | Bounded movement | one knob step per round; ≤ 3 rounds per lineage; a lineage that moved up is never moved down (and the reverse): RETIRE |
| G6 | Not tuned to the student | a variant must be solved by an independent ALT family (`alt_check`, non-student, non-builder); the drift monitor watches repairs; knob moves are logged as knob calibration (predicted vs realised band) |
| G7 | Pool regularisation | KL(the pool's cell histogram with the variant ‖ the pool at the start) ≤ κ, entropy ≥ H_min; originals are never retired by a variant |
| G8 | System-level lifts | archive counterexamples, lemma candidates and World-builder fixes go to the update rule (R1–R10, the generalisation gates) |
| G9 | Ledger and rollback | every decision is a ledger record; a repair is a new version and a variant a new task, the previous ones restorable |

**Where I disagree with the spec, with reasons.**
1. **DESCEND has almost no lawful move today.** Every knob of `compose` is declared "harder". A plain swarm task has
   no declared easier knob that keeps the skill: stating a derived value gives the decision away (D7), dropping a
   requirement is D1, shortening the program changes what is taught. The only lawful easier moves are structural
   inverses on tasks that were built harder (undo a recorded knob: fewer distractors, no pagination; drop the last part
   of a composition). `ComposeKnobs` implements exactly those. On the 89-task pool, most of the 47 too-hard tasks are
   plain tasks, so they will be **RETIRED** ("no declared knob makes this task easier"), not descended. Generating
   easier rungs needs new declared easier knobs (or building tasks as ladders from the start), which is a design
   decision for the lead, not something to improvise in the loop.
2. **"Frozen intent card" and "derive instead of given" conflict.** ASCEND's derive knob moves a value from the
   instruction to the intent card's derive list. I freeze the skill, the cell and the P3 result; the given/derive
   lists may move only by that declared knob.
3. **DESCEND and ASCEND are steered by the student's band.** That is the point of a curriculum, so the regularisers
   cannot remove it; they bound it (one declared step, a direction lock, an independent solver, fresh rollouts, the KL
   trust region). The drift monitor is the right alarm for repairs (where the pass rate should move only as the fixed
   defects explain); for knob moves the pass rate is supposed to move, so they are logged as knob calibration.
4. **Uncertain outcomes (67 of 676) must not count toward "too hard".** A routed failure (protocol break, budget) is not
   evidence of difficulty. The loop holds a task whose graded rollouts are mostly uncertain.
5. **A wrong oracle can hide behind agreement.** The 6 finance_ap tasks fooled the oracle and the ALT solvers alike.
   ORACLE's policy truth needs the World's executable rules; `decision_disagreement` only triggers review when the
   *student* and other runs agree on another value, and a shared mistake of all agents stays invisible without rules.
   For Worlds without executable rules the precondition is weaker than for finance_ap, and the report says so.

### 9.6′ What the paper reports about the loop (additions)

- δ_AA with its replicates, beside the between-view spread;
- per test: t, α_t, the e-value, the stopping point, the paired effect with its bootstrap and cluster-bootstrap intervals;
- the run's α spent and the decision split's units left;
- headroom and the P/F game's trajectory;
- E10 ablations at equal budget, ≥ 3 seeds, with paired differences to the full loop;
- the probe loop's outcome counts, demotions, repairs and refusals by rule, lifted lessons, and the drift monitor.

## 3. Proof sketch and simulation of the run-level guarantee

The guarantee is the union bound above; the only property each test needs is conditional validity, which the betting
supermartingale gives under exchangeable fresh units. Simulations (scripted fakes, `test_synth_rsi_stats.py`):

**Family-wise error of a run of T null candidates** (every candidate's true gain is 0; units: ties, with ±1 seed noise at
3% a side; 40 units per test; δ_AA from 5 A/A replicates per run; 2,000 runs per row):

| T tests | ours: betting at α/T against δ_AA | uncorrected: one-sided normal LCB₉₅ > 0 (M3's rule with a 0 floor) | uncorrected LCB₉₅ > δ_AA |
|---|---|---|---|
| 5 | 0.000 | 0.204 | 0.004 |
| 10 | 0.000 | 0.323 | 0.004 |
| 20 | 0.000 | 0.556 | 0.006 |
| 40 | 0.000 | 0.798 | 0.011 |

The machinery end to end (calibrate + decide in paired mode, 120 runs of 6 null updates each, scripted evaluator with
3% seed flips) accepts no null update (`test_the_whole_engine_keeps_the_run_level_error_under_alpha`); the betting test
holds its level at the null boundary (E[D] = δ, 1,500 draws: rejection rate within 0.05 + 3 se).

**The price is power.** Betting at α_t with the A/A margin is conservative: the observed family-wise error is 0, far
under 0.05. Power at α_t = 0.0025 for a true gain g on n units (same noise model): n = 40: 0.00 (g 0.1), 0.02 (0.2),
0.13 (0.3); n = 80: 0.01, 0.24, 0.72; n = 160: 0.08, 0.81, 0.99. With the real seed noise of the rsi-2 pool (0.64%
discordance) the minimum detectable gains are in the preregistration (§5): 0.094 at 240 units, 0.061 at 375.
Two levers stay within the rules: larger decision views (power grows with n, δ_AA shrinks as 1/√n), and fewer
preregistered tests (t_max). Common random numbers would remove the seed noise from D but not the A/A margin, and the
user's specification (independent seeds, δ_AA as the bar) is kept.

## 4. Open issues

| Issue | Detail |
|---|---|
| Power is the binding constraint | Run-level control plus the A/A margin needs ~240 fresh units for a 0.1 gain (α_t = 0.005). The 10 × 10 pool's decision split (~375 units) supports one or two full tests. A 0.056 gain (RSI-1's best) is not detectable in any feasible RSI-2 decision. |
| No locked headroom | After item 20, 18 held-out variants give 0 misses on the 7 real locked sources. The locked set measures non-regression until the larger pool or the P loop provides headroom. |
| Dev headroom is panel-only | The 7 dev misses are MUT suspects (D3 `eq_to_any`). With MUT adjudicated by the panel they close, and dev headroom is 0: the F loop then has nothing to learn from planted defects alone. |
| D1, D3, D4, D8 under-powered | ~1 witnessed plant per 7 source tasks; ~210 sources for 30 per class. Needs new held-out variants with ≥ 1 witnessed plant per task for these classes. |
| Super-population validity | The betting test is exact for exchangeable units of the decision population. Plants of one source task are correlated; the cluster bootstrap is reported beside every decision, but the guarantee itself treats plants as exchangeable. |
| The probe loop's interfaces | `StudentProbe` and `OutcomeClassifier` are the learnability agent's; `ProbeOutcome` is defined here as the record consumed. When their classifier lands, a thin adapter maps its output; T1–T7 hold whatever it emits. The probe loop has run only with a scripted student (the student was never called). |
| The between-view δ and R3 bands | R3's per-class bands are still between-view (M3), on proposal-split views; on a small proposal split the views repeat and the bands are the configured minimum (0.02). |
| Parallel workers | A killed parent leaves its shard workers computing until their current request ends (they exit at the next pipe read). |
| Two anchors with real leaks | item 20 lists the fixture `ap/hold-1003` and it_helpdesk `t02` as D7 candidates for a person to settle; `t02` is also the anchor TWIN alarms on at seeds 3 and 6 (dropped as an anchor in paired mode). |
