# 12 Difficulty: tasks the policy sometimes solves, as the policy changes

**The problem.** A GRPO group in which every attempt passes, or every attempt fails, has no outcome spread: its
advantage is zero and it teaches nothing. Our first 21 gated tasks show it plainly. The Qwen3.6-35B-A3B student is
bimodal: about 0.9 on 2-fault tasks and about 0 on 3–4-fault tasks, with mean Φ 0.34–0.50 on the hard ones. And
difficulty is not fixed: a task in band at step 0 leaves the band as the policy improves.

**The subsystem**, in four parts, all reusing the framework's own pieces:

| Part | When | Module | Reuses |
|---|---|---|---|
| 1. Online difficulty filtering | training | `forge/difficulty.py` (tracker), `train/curriculum.py` (sampler), `train/bridges/tinker/trainer.py` (`sampler="difficulty"`) | `forge.learnability` (4p(1-p), Wilson), `train.quarantine` (unchanged; the sampler forwards to it) |
| 2. Generator controller toward p ≈ ½ | factory | `factory/difficulty.py` `control`, `Journal` | `factory.calibration.KNOBS` (K1–K6) and `NEVER`, the targeting cause and mode labels |
| 3. Cheap difficulty predictor | factory | `factory/difficulty.py` `features`, `fit`, `loo`, `Predictor.predict`, `plan`; `scripts/lh_difficulty.py predict` | the S15 targeting records, `data/debug/candidates.json` |
| 4. Full-task vs checkpoint-conditioned difficulty, checkpoint starts | both | `factory/difficulty.py` `conditional`, `checkpoint_levels`, `checkpoint_start`, `check_start`, `gate_start`; `train/curriculum.py` reverse curriculum | `factory.checkpoint` `credited` / `potential` (C1), `GeneratedTask.simplify_units`, the gates, `credit.compose` turn credit |

Views: `observe/difficulty.py` (p̂ over time per task, band membership, actions, predictor error).

## 1. Online difficulty filtering (training time)

Pattern: *Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning* (Bae et al., 2025, arXiv
2504.03380). There, the current policy's rollouts estimate each prompt's pass rate during training, and only prompts of
intermediate difficulty are kept, because a balanced pass rate maximizes what one GRPO step can learn.

**Ours.** For long-horizon tasks one rollout costs about $1.7 and 25 minutes, so we do not re-sample to filter. The
estimate comes from the groups the trainer runs anyway.

- **The tracker** (`DifficultyTracker`) keeps discounted counts per task. Before an observation at step s, the old counts are multiplied by `decay^(s − last)` (default 0.9 per step), so evidence from an older policy fades. The estimate is p̂ = k_eff/n_eff with a Wilson interval on the effective counts.
- **Masked draws are never evidence.** They are infrastructure failures (`None`). A property test checks that the sampler's counts equal the counts with the masked draws removed.
- **A failed budget stop is not skill** (`max_steps` and similar). It stays out of p̂ and is counted apart. Its measured Φ still counts, because that progress is real.
- **Φ spread:** `signal = max(4p(1-p), 4·Var(Φ_final))`. A task that never passes but whose Φ varies still teaches through step credit, exactly as `train.quarantine` already rules (it never quarantines Φ spread).
- **The sampler** (`DifficultySampler.pick`, same contract as `TaskQuarantine.pick`):
  - It drops a task only when the evidence says it is outside the band, that is, when its interval does not overlap [0.1, 0.7]. At 8 draws, 8/8 still overlaps (Wilson lower bound 0.68), so a task leaves only on real evidence.
  - A task with Φ spread is always kept.
  - `explore` (default 0.1) of picks ignore the filter, so out-of-band tasks are re-measured as the policy moves.
  - Kept tasks are drawn with weight `max(floor, signal)`. Unmeasured tasks have signal 1, so every task is measured first.
  - The quarantine still blocks flat tasks for r steps.
  - Held-out tasks are refused at construction (V9).
  - The RNG state is saved, so a resume makes the same picks.
- **Trainer wiring:**
  - `TrainConfig(sampler="difficulty", band_lo, band_hi, explore, difficulty_decay)`.
  - CLI: `scripts/tinker_rl.py train --sampler difficulty --band-lo 0.1 --band-hi 0.7 --explore 0.1 --difficulty-decay 0.9`.
  - These settings may change on resume, because they change scheduling, never the objective.
  - The state is saved in `state.json["difficulty"]` and the events in `difficulty.jsonl`. The step metrics are `difficulty/measured`, `difficulty/in_band` and `difficulty/mean_signal`.
  - The sampler is stdlib only, so the DeltaAI (vLLM) trainer can reuse it unchanged.

## 2. The generator controller (factory time)

Pattern: *R-Zero: Self-Evolving Reasoning LLM from Zero Data* (Huang et al., 2025, arXiv 2508.05004). A Challenger
proposes questions and is rewarded by the Solver's uncertainty, `1 − 2|p̂ − ½|`; the Solver trains on the questions near
½. In our swarm, the generator (the builder roles) is the challenger, a *task variant* is the question, and the
35B student is the solver. Difficulty moves only through knobs, never through a vaguer prompt or a lie (V5).

`control(task, rate, causes=..., modes=..., conditional_rates=..., frontier_passes=..., applied=...)` returns one
`KnobAction`:

| Action | When | Knobs |
|---|---|---|
| `sample_more` | nothing measured, or (not final) the interval does not decide the band | — |
| `budget_bound` | most failures are budget stops | none: the budget stays at 300 calls (`calibration.NEVER`) |
| `fix_task` | most failures are a spec defect, a World bug or env flake; or under 70% of failures are skill gaps; or no frontier solver passes (unfair) | — |
| `harden` | p̂ above the band | H2 (take back the last easing knob), H1 (add a real interacting fault), H3 (a longer span) |
| `ease` | p̂ below the band | K1–K6 ordered by the failure modes (e.g. `claimed_done_without_progress/reproduce_from_report` → K4 a visible reproduction); **C1 checkpoint start first** when some later milestone is in band once its prerequisites hold |
| `keep` | inside the band | — |

- Every action carries its reason, the knob texts and the **generator reward** (R-Zero's uncertainty reward of the measured variant). The swarm can then prefer the knob moves that landed near ½.
- `Journal` appends every action to `difficulty_actions.jsonl`.
- Held-out tasks raise `HeldOutRefused`: they are fixed for defects only, never eased or hardened from their traces.

## 3. The cheap difficulty predictor

The idea is to predict p before the expensive measurement (k=8 at 300 calls costs about $14 per task for the student alone).

- **Features:** faults, executed edges, modules, log oracle lines, log milestone tests, tool limit/300, fairness flags, and the lane language.
- **Probes (optional):**
  - k passes of n low-budget student rollouts, fused as a Beta update (the static prediction counts as 4 pseudo-draws);
  - the frontier's pass rate (0 flags the task as possibly unfair).
- **Model:** ridge logistic regression with a binomial likelihood, stdlib only.
- **Data:** training tasks only, because held-out traces must not shape training tasks. Budget stops are taken out of n.
- **Use:** `plan` orders the measurement queue by |p − ½| and only *suggests* skips. **It never keeps or rejects a task (V6).**

**Leave-one-task-out on our data** (`runs/difficulty_predictor.json`; 10 training tasks, 17 rows from the 150- and
300-call runs):

| Metric | Value |
|---|---|
| Predictor MAE | **0.102** |
| Constant baseline MAE | 0.146 |

This is not a useful predictor yet, and we report it as such:
- **Almost no positive examples.** 15 of the 17 rows have p = 0, and the predictor wins mainly by predicting near 0 there.
- **It misses the one easy task.** For `db-go-echo-61eca05b` (2 faults, 7/8) it predicts 0.02 when that task is held out, because it is the only passing example.
- **The 150-call rows don't record budget stops**, so their failures include budget stops.

It becomes useful once the co-authors' easier tasks (2–3 faults, from TASKS_CREDIT_AND_EASING) add rows inside the
band. Refit with `scripts/lh_difficulty.py predict`.

## 4. Full-task vs checkpoint-conditioned difficulty, and checkpoint starts

**Two difficulties.** `conditional(episodes, milestones)` uses `factory.checkpoint.credited` and `potential`, the same
C1 rule and weights the step credit uses. It reports:
- p(full pass) and mean Φ;
- per milestone: p(m), credited, and **p(m | every prerequisite of m credited)**;
- the stall level, the deepest DAG level reached.

Masked draws and budget stops are left out. On the 300-call student runs (`runs/difficulty_conditional.json`):

| Task | p(full) | Milestones: p(m) / p(m given its prerequisites) |
|---|---|---|
| go-toml a02114e3 | 0 | m04: 0.50 / **0.67** given m03 |
| black ae5830b7 | 0 | m04: 0.50 / **0.50** given m03; m02: 0 / 0 |
| mvdan-sh 5e983709 | 0 | m01 0.83, m03 0.67, m02 0.33, m04 0 / 0 |
| black 90117fa0 | 0.33 | m03: 0.33 |

Some later milestones are in band once their prerequisites hold, while the full task is at 0. That is the case for a
**checkpoint start**.

**Checkpoint starts.**
- **Levels.** `checkpoint_levels` lists the pre-applied set of each start: the first L milestones in DAG order (by depth, then history order). Each set is prerequisite-closed and leaves at least one milestone.
- **Building a start.** `checkpoint_start(task, pre, level)` builds the variant with the task's own `simplify_units(units, kind="checkpoint")`, the machinery of the counterfactual control. The start tree is A′ with the real fixes of `pre` applied. The pre-applied milestones leave the task with their requirements, their prompt lines and their mutants.
- **Tagging.** The variant carries `curriculum = (parent, level)` and `checkpoint_of`.
- **Held-out tasks** get no variants.
- **Gates.** A variant is a new task, so every gate runs again:
  - `check_start` grades the *original* milestones on the variant's start. Each pre-applied milestone must be credited, every other must fail, and unmeasured is a problem.
  - `gate_start` then runs the gates (e.g. `deterministic_gates`: S5 oracle 1 and start 0, S8 floors and mutants).

**Credit stays valid.**
- **The pre-applied milestones are no longer milestones of the variant.** No state can credit them, and Φ over the variant's milestones is 0 at its start.
- **If a variant is graded on the original milestones,** the checkpoint sidecar grades the start as Φ0 (`grade_start`). `progress_rows` books only changes after it, and the ΔΦ telescopes to Φ_final − Φ0. `credit.compose.turn_advantages` gets `phi0 = Φ0`, so an episode that merely keeps the pre-applied fixes earns no ΔΦ and no turn credit.
- `test_a_checkpoint_starts_prerequisites_never_earn_credit` checks both paths.

**Reverse curriculum** (the sampler, `levels`):
- Per parent, only the current level is sampled, starting at the latest start.
- When the level's p̂ lower bound reaches 0.5 (at least 4 effective draws), the start moves one level earlier.
- When its upper bound falls below 0.1, it moves back.
- Level 0 is the full task. Each move is a `start_earlier` / `start_later` event.

**Regression guards (fixed 2026-10-23).** A pre-applied milestone's own tests now join the variant's pass-to-pass set
(`windows.guard_tests`, in both `WindowTask.simplify` and `GeneratedTask.simplify_units`; listed as
`simplified_from.guards`). They pass at the new start, so they gate the regression check `c_regress`: an agent that undoes
an earlier fix keeps its Φ (the guards are not milestones) but loses the full pass. Test:
`test_a_simplified_start_guards_the_pre_applied_fixes_in_the_regression_check`.

**Predictor data (2026-10-23).** The 150-call targeting rows had no `budget_stops`, so their step-limit failures were
read as skill. `backfill_budget_stops` fills them from the episodes (failed, unmasked `max_steps` stops; else the row's
stop counts, at most its failures). Leave-one-task-out after the backfill: MAE 0.204 against 0.257 for the constant
baseline over 12 rows of 9 tasks (before: 0.102 against 0.146 over 17 rows, flattered by budget zeros). It still
predicts the one easy task (echo, 7/8) at 0.07: advisory only, as before.

## 5. SPELL

We read *SPELL* (self-play RL for long-context language models: one model as questioner, responder and verifier, with
a curriculum that grows the context). We did not port it:
- our verifier is executed tests, never a model (V6, V12);
- our tasks come from real fixes, not from model-written questions (V1).

Its curriculum idea, moving along a difficulty axis as the policy improves, is what the reverse curriculum does with
checkpoint starts.

## 6. Tests

`tests/unit/test_difficulty.py` (27 tests) covers:
- **The tracker:** masked and budget handling, decay, Φ spread, conditional rates, round trip.
- **The sampler:** the held-out refusal, the band filter, explore, the quarantine interplay, the property test (never samples a held-out task, never weights a masked draw), the reverse curriculum, a resume that makes the same picks, `draws_of`.
- **The Tinker trainer** with `sampler="difficulty"`, on the fake service: after measuring, it stops sampling the solved task. Switching the sampler on resume is allowed.
- **The controller:** every action, the held-out refusal, the journal.
- **Conditional difficulty,** checkpoint levels, starts, start checks and gates, and the credit test.
- **The predictor:** monotone in faults, LOO beats the baseline on synthetic data, the advisory plan, held-out and budget rows skipped.
- **The views.**

## 7. The pilot (2026-10-23): the controller on the swarm, measured

**The wiring** (`factory/difficulty_bridge.py`, `factory/revise_debug.py`, `scripts/lh_difficulty_pilot.py`):
- `ControlController` makes `control()` a `difficulty_hooks.Controller`. One knob vocabulary serves both sides: K1-K6,
  C1, H1-H3.
- C1's level comes from `choose_checkpoint`: the checkpoint level whose remaining milestones' measured
  `p(m | prerequisites)` multiply closest to 1/2 inside the band. An unmeasured remaining milestone rules a level out.
- The swarm's `prompt_author` writes the variant's report. The lane's executed gates decide.

**What the pilot changed in the controller** (each with a test in `tests/unit/test_difficulty_bridge.py`):
- **Budget-bound tasks get a shorter horizon.** `budget_bound` now carries the horizon-shortening knobs C1 and K1.
  The 300-call budget never moves.
- **Root milestones carry no checkpoint signal.** A root milestone's rate (no prerequisites) is no longer read as a
  checkpoint signal. Every task used to get "checkpoint start first".
- **A conditional rate above the band is not in band.**
- **K1 drops only a fault that costs passes.** The fault's measured rate must be at most 0.7; otherwise the next knob
  is used. Before this, K1 picked a fault the student fixed 8/8.

**What the pilot changed in the reviser:**
- **A knob changes only what it targets.** C1, K1, K2, K3 and K4 keep every kept requirement word for word: K3 appends
  a user-visible area, K4 adds `repro`. K6 rewrites only its own milestone.
- **Why:** a free rewrite lost a fact the hidden tests need. go-toml's C1 report dropped "date and time values" from m03,
  and m03 went to 0/8. With the requirement restored, it was 8/8.

**Full task vs checkpoint start, measured.** The conditional rate from the student's own episodes did not predict the
real-fix checkpoint start on go-toml:
- **Prediction:** p(m04 | m03) = 0.67 in the parent.
- **Measured at the C1 start (m01 and m02 really fixed):** m04 was 0/8. One hidden test,
  `TestMarshalTextMarshalerStringKindOmitempty`, failed in every episode. It failed in 1 of 8 parent episodes.
- **What follows:** the agent's own partial fixes are not the real fixes, so a checkpoint start can be harder than its
  estimate. The estimate is a prior; the executed measurement decides, as everywhere else.
- **Black, the other way round:** its C1 start removed the budget stops (6/8 to 0/8), and the student passed 1/8.

The per-task results are in `13` §9.

