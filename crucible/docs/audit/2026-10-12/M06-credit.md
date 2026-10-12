# M6 credit: audit (2026-10-12)

Scope: `src/crucible/credit/` (operators potential, defect_to_check, branch_value, rubric_item, prm, infra_mask;
compose, steps, validity, properties, shaping, gar, hack_detector, judge, the new `signed`), `workflow/potential.py`,
the grader that composes rewards (`verify/grader.py`), the judged rubric items (`verify/judge_item.py`, `answers.py`
criteria), and every place credit leaves Crucible: the trainer (`train/trainer.py`, `train/credit.py`,
`train/group_ops.py`), the NeMo-Gym server and the agentenv capture.

Checked against ARCHITECTURE §M6, DESIGN §6 and `docs/modules/M06-credit.md`, with the M0 protocol. The framework
comparison is the shared M5-M7 sweep (`M05-M07-study-sweep.md`, M6 picks); the defect hunt is `M06-study-defects.md`
(K1-K20). 17 regression tests (`tests/unit/test_m6_audit.py`) all fail on the M5 commit (`103fd25`) and pass after
the fixes.

## 1. Requirement ledger

| # | Requirement (source) | Before | Evidence |
|---|---|---|---|
| L1-L3 | C1 Φ, carried values, telescoping shaping (§6.2, §6.4) | live, unguarded | raw lists unvalidated (weights [2, -1] gave Φ = 2.0; NaN gave NaN; K12); step 0 credited with Φ(s0) (K11) |
| L4 | R = G·[O(1+αp+βe) + κ(1−O)Φ], bounds (§6.4) | live | a masked composition reported `reward: 0.0` beside `infra/masked: 1` (K10) |
| L5 | A gate can never be traded for partial credit (§6.4) | **wrong** in the dense channel | a gate-failed finance_ap episode's turns earned +0.143 each (K1); the live trainer added raw ΔΦ per call, uncentred and ungated (sweep G4 D1: a group's net advantage +9.0) |
| L6 | Per-turn delta: centred, scaled, "never exceeds the outcome advantage" (§6.4 correction) | **wrong** | unbounded, growing with √turns: at 800 turns a failure's turn out-ranked every turn of the success (K2); group centring paid oscillation (K3) |
| L8 | Injection into trainers (§6.4) | partial | NeMo-Gym's `reward_components` broke NeMo-RL's sum contract (sweep G2); the C6 mask and rubric judges had no production caller |
| L13 | C4 lifecycle: fixtures before use, validated on held-out traces (§6.5) | **wrong** | an LLM item with no fixtures, n = 4, got weight 1.0; the bootstrap interval was never read (K7); a quoted PASS flipped FAIL; an unparseable verdict counted as unmet (K8) |
| L14 | C5 PRM | partial | an unfit model imputed 0.5; NaN scored 1.0 (K19) |
| L15 | C6: a failed measurement is never a zero (P16) | partial | unwired; a partly unmeasured terminal state scored its unmeasured milestones as 0 (K9) |
| L16-L17 | §6.5 validity gate; S2 | partial | `time_split` could return an empty training split (K14); NaN made AUC order-dependent (K15); S2 weighted every milestone type, including six with fit AUC 0.47-0.54, against §6.5's keep rule (K16) |
| L18 | R1-R6 pass on every coding and toy task | **vacuous** | Φ ≡ 0 still passed all six; R3 was skipped on task01 yet reported ok; R5 had no prerequisites to test; R6 tested a lambda (K4); DESIGN's R3 ("every negative gets 0") contradicted the code and the κ term (K5); R5 failed a legitimate fractional prerequisite (K13) |
| L19 | M4 signals feed credit (§5.2, §6.1) | **missing** | nothing in M6 reads `Signal.grades`, `polarity` or `reliability`; `StepRow.violations` is never filled |
| L22 | Two-level hack detector | false negatives | World writes are `file:<path>`; root-level `conftest.py`, `tests/helpers.py`, `solution/`, `.crucible/` went unflagged, as did Go/JS test files, `-k="not ..."` and skip decorators (K6, K20) |

## 2. Defects (reproduced) and fixes

| Id | Sev | Defect | Fix |
|---|---|---|---|
| K1, sweep G4-D1 | P1 (P0 on the trainer path) | dense credit ignores gates; the trainer added raw ΔΦ | `Episode.gated/hack`: no positive local term; a group with no outcome spread gets none; the trainer (`train/credit.turn_credit_map`) passes every call's ΔΦ through the same rule before any path adds it to tokens |
| K2 | P1 | unbounded local term | the episode's local vector is scaled down as a whole so no term exceeds its outcome advantage: the sign never flips, zero mass is kept |
| K3 | P1 | centring paid oscillation and length | centred within each episode: the local term only redistributes an episode's own outcome advantage |
| sweep (AReaL GVPO) | — | no step veto | `Episode.vetoed`: a vetoed turn is never positive; in a flat group it gets `-zeta` |
| sweep (Xiaomi verl `adv_signed`) | — | the format penalty subtracted mass (net −5.0 on a toy batch) | `credit/signed.py` (Apache-2.0 port, pure Python) and `TrainerConfig.format_penalty_mode="signed"` (default stays `reduction`) |
| K4 | P1 | R1-R6 could not fail | R1 also needs Φ(oracle) = 1; untested properties are reported (`not_tested`), and only by-construction ones (R5 without prerequisites, R6 on a state-only reward) count as holding; R3 without negatives is not a pass |
| K5 | P1 | R3 contradicted DESIGN | DESIGN and code now agree: a **crafted** negative must be caught by a gate (reward 0), an **honest** failure keeps κΦ below every success. The toy's crafted negatives became gates (`no_reversed_refund`, `answered_if_resolved`) |
| sweep (OpenEnv field evidence) | — | progress farmable | R7: a shallow mutant's κΦ ≤ 0.1 (coding: located/created/deleted milestones only; toy: a premature "resolved") |
| K6, K20 | P1 / P2 | hack rules missed real World writes and common skips | match on the path after `file:`; Go, Rust, JS/TS test files, `pytest.ini`, reward files, `-k=`, skip decorators |
| K7 | P1 | items validated without fixtures or data | an LLM item needs its good/bad fixtures judged right; at least 10 held-out traces per class; the interval's lower bound > 0.5 |
| K8, sweep (verifiers, MiMo, TRL) | P2 | verdict parsing; unmeasured counted as unmet | reason-first prompt with an adaptive fence; the verdict alone on its own line, the last one winning; `<think>` cut-offs are unmeasured; unmeasured items leave the share; non-negative weights and unique ids; `judge_items` takes votes with a majority, retries an unparseable reply, and masks the rubric channel when a critical item stays unmeasured; `answers.validate_criteria` refuses negative, all-zero or duplicate criteria |
| K9, K10 | P2 | C6 unwired; masked reward 0.0 | the grader computes Φ over measured milestones (`progress/unmeasured` reported) and masks through `mask_for` when nothing could be measured; `RewardJson.reward` is `None` exactly when masked |
| K11, K12 | P2 | Φ(s0); unvalidated potential | `trajectory(initial=...)`, `grade_episode(base_phis=...)`; `potential()` refuses bad weights, φ and prerequisites |
| K13 | P2 | R5 vs fractional prerequisites | a stub needs a prerequisite at 0 |
| K14, K15 | P2 | empty time split; NaN AUC | the cut moves to the nearest tie boundary and refuses a single-time set; non-finite scores raise |
| K16 | P2 | S2 off-spec | the §6.5 keep rule per type (`types_kept`); one midpoint helper (`fraction_index`) for the toy and future rebench rows |
| K17 | P2 | branch-value edge cases | a verifier `None` counts as masked; values outside [0, 1] raise (a composed reward is not an outcome); the World is restored to the branch point at the end |
| K18 | P2 | step rows misaligned | `rows_for` checks one call id and one violation list per step; `to_turn_shaping` sums rows per call and honours `gated` |
| K19 | P2 | PRM imputation | `score` returns `None` for an unfit model or a non-finite feature |
| sweep G2 (NeMo-RL) | P1 | `reward_components` did not sum to the reward (4.0 vs 1.033); NeMo-RL raises | one component equal to the reward; the channels move to `crucible_channels`; `/reverify_mode` returns NeMo-Gym's enum value |
| sweep G5-D1 | P2 | the trainer-side GAR passed any reward > 0 | passes at `reward >= 1` (a composed success) |
| sweep G4 (Xiaomi verl length penalty) | P2 | an over-budget success scored 0, below a failure's κΦ | passed-only, bounded penalty (max 0.2 < 1 − κ) |
| sweep G5-D5 | P2 | agentenv capture collapsed every repeated digest | `CaptureLogConnector`'s retry rule (marked retries or failed attempts only) |

## 3. Coverage and contracts

- `crucible.credit`: 74% before, **83%** after over the full suite (`rubric_item.py` 0% -> 93%, `properties.py` 100% -> 98%, the new `signed.py` 81%). Still thin: `judge.py` 0% (the paid LLM-judge baseline), `defect_to_check.py` 46%, `prm.py` 58%.
- Suite (`tests/unit`, `tests/toy_e2e`): 965 passed, 3 skipped (948 before, plus the 17 new tests).
- `lint-imports`: 4 kept. `ruff`: clean.

## 4. Comparison with the local frameworks

The picks are in `M05-M07-study-sweep.md` (M6 table). Built here: the gate-aware, bounded, per-episode-centred turn
credit with AReaL's veto; `adv_signed` (Xiaomi verl, PORT); NeMo-RL's components contract; C6 wired with NeMo-Gym's
judge boundary; verifiers' verdict and weight validation with MiMo's votes and retries; Xiaomi verl's bounded length
penalty; R7 from OpenEnv's field evidence. Queued: GiGPO anchor-state groups (ROLL), a prime-rl credit bridge, C5 from
C3 branch pairs (OpenRLHF), agentenv step-level mistake labels, C3 branches for outside harness agents.

Where Crucible is stronger: a gated composition with bounded partial credit inside the scalar (no framework gates
dense credit), a validity gate on reward items, explicit masking of unmeasured outcomes, and a property suite that
every task's reward must pass. AReaL's `process_weighted` was cited as sign-preserving; it flips negative advantages
to positive, and `docs/research/improvements/M06-M08.md` is corrected.

## 5. Gate re-run and resolution

| Gate | Before | After |
|---|---|---|
| R1-R6 (coding) | "pass" on 5 tasks | R1-R7 pass on task01v2, 02, 03, 04; **task01 cannot test R3** (no failing run) and is reported, so the gate does not pass (`m6_properties_coding.json`). R7 partial credit of shallow mutants: 0.027-0.078 |
| R1-R6 (toy) | "pass" (crafted negatives earned 0.2, 0.133) | R1-R7 pass; crafted negatives 0, the honest "reopened" 0.133; R5 and R6 hold by construction |
| S2 (coding) | Φ_mid 0.50 vs judge 0.75, every type weighted | the §6.5 rule keeps **no** milestone type at the midpoint, so validity-weighted Φ_mid is 0.50 (uninformative), the judge 0.75: S2 fails as before, now measured as specified |
| corr(C3, ΔΦ) on task04 | pending (GPU, ~$130) | still pending |

Open:
- M4 signals into credit (L19): `StepRow.violations` is still unfilled and no operator reads `Signal.grades` or
  `reliability`. The intended route is `adv_signed` marks with κ = 1 + λ·reliability from M4 negative step signals
  and the veto from violations; both operators are built, the producers are not.
- The queued picks above; behavioural milestones for S2; the C3 correlation run (needs approval).
- The trainer's format penalty defaults to `reduction`; switching the default to `signed` changes training runs and
  is left for an explicit decision.
- The cached S2 rows (`rebench_phi`) predate the M5 stale-state rule and the new midpoint helper.
