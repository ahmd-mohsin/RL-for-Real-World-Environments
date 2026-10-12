# M6 study: defects in `credit` (dense credit, reward composition, validity, properties)

Scope: `C/credit/` (`compose.py`, `steps.py`, `validity.py`, `properties.py`, `shaping.py`, `gar.py`, `hack_detector.py`, `judge.py`, `operators/{potential,defect_to_check,branch_value,rubric_item,prm,infra_mask}.py`), `C/workflow/potential.py` (C1, moved there in the M0 audit), and the consumers: `C/verify/grader.py`, `C/verify/judge_item.py`, `C/interop/replay.py` (the production `steps.jsonl` producer), `C/studio/datasets.py` and `label.py`, plus pack usage in `CP/m56/{task_properties,rebench_s2,rebench_judge,rebench_phi}.py`, `TP/m56/pipeline.py`, `TP/studio.py` and `FP/studio.py`. Reports: `docs/modules/reports/{m56_toy,m56_gate_coding,m6_properties_coding,m6_s2_swe_rebench}.json`.

Spec: ARCHITECTURE "### M6 `credit`". Its exit gate has three parts: S2 Φ at the midpoint reaches AUC ≥ 0.75 and beats an LLM-judge baseline on held-out traces; R1–R6 pass on every coding and toy task; C3 branch value vs ΔΦ correlation is reported for 3 task04 trajectories. Also read: `docs/modules/M06-credit.md`, DESIGN §6 (C1–C6, the R formula, R1–R6, §6.5), DESIGN §5.2 (the "credit it implies" column), and `docs/research/improvements/M06-M08.md` (V3, V4, V5).

Paths: `C` = `crucible/src/crucible/`. `CP` = `crucible/packs/coding/src/crucible_pack_coding/`. `TP` = `crucible/packs/_toy/src/crucible_pack_toy/`. `FP` = `crucible/packs/finance_ap/src/crucible_pack_finance_ap/`.

Method: a read-only study on HEAD `b65754b`, using `crucible/.venv` (Python 3.12.9). **[ran]** means reproduced with a probe; **[read]** means established from the code alone. The probes are in the session scratchpad at `m6hunt/p1_compose.py`, `p2_finance.py`, `p3_misc.py`, `p4_c2.py` and `p5_gate_mutation.py`. No paid, GPU or networked run was made.

Offline gates re-run:
- `python -m crucible_pack_toy.m56.pipeline` reproduces the toy gate values: R1–R6 true, S2 Φ_mid 0.583 and Φ_end 0.667 on 11 tickets. The only report diff was a schema-only `"states": []` per milestone; the report was restored with `git checkout`.
- `python -m crucible_pack_coding.m56.task_properties` reproduces `m6_properties_coding.json` byte for byte.
- `python -m crucible_pack_coding.m56.rebench_s2 data/m56/rebench_phi_n80_s7.jsonl …judge… …judge_end…` reproduces `m6_s2_swe_rebench.json` exactly: Φ_mid 0.4986 vs judge 0.748, diff −0.25 (CI −0.38 to −0.10).

## 1. Requirement ledger

Status: **live** = built and behaves as specified; **partial** = built, but missing a specified part or unwired; **wrong** = built but contradicts the spec; **missing** = not built.

| # | Requirement (source) | Status | Evidence |
|---|---|---|---|
| L1 | C1 Φ = Σwφ·Πpre φ / Σw (DESIGN §6.2) | live, unguarded | `C/workflow/potential.py:32-45` matches the formula. With raw lists nothing is validated (K12). The first ΔΦ includes Φ(s₀) (K11) |
| L2 | An unevaluated milestone keeps its last value, and regressions lower Φ (§6.2) | live | `potential.py:58-63`. A `None` after a measured value carries the old value (p1 A9). The `lost` lists are correct (unit test) |
| L3 | Shaping with Φ(terminal) := 0 and γ = 1 telescopes to −Φ(s₀) (§6.4) | live | `potential.py:69-81`, `shaping([0.5,0.5], initial=0.5) = [0, -0.5]` (p1 A8) |
| L4 | R = G·[O(1+αp+βe) + κ(1−O)Φ] with κ ≤ 0.3, α ≤ 0.1, β ≤ 0.05 (§6.4) | live | `C/credit/compose.py:52-72`. Bounds are enforced. NaN or inf Φ raises (p1 A6). Masked output: K10 |
| L5 | A gate can never be traded for partial credit (§6.4) | **wrong** in the dense channel | The terminal scalar is gated, but ΔΦ in `steps.jsonl`, `turn_advantages` and the studio weights ignores gates. Live on finance_ap (K1) |
| L6 | Per-turn delta mode: centred, scaled by σ_R/σ_ΔΦ, "never exceeds the outcome advantage" (§6.4 correction; `compose.py:21`) | partial / **wrong** | Centring and scaling are live. The bound is not enforced (K2), and centring pays for length (K3) |
| L7 | Masked episodes are excluded from every group statistic (§6.4) | live | `compose.py:91-93,124-126`; unit test |
| L8 | Injection points in verl, verifiers v1 and OpenEnv `compute_step_rewards` (§6.4) | missing | No consumer of `turn_advantages` or `to_turn_shaping` exists outside tests. `train/runtime.py:466-471` takes per-call credit only from a user-supplied `aux_fn`. NeMo-Gym and agentenv receive raw `crucible_steps` rows (`C/interop/nemo_gym/server.py:139-146`, `C/interop/agentenv/plugin/steps.py:112-119`) |
| L9 | `steps.jsonl` emitted by graders, keyed by `call_id` and `harness` (§4.3.2, P2 plan) | partial | `credit/steps.py` is live, but `grader.grade_episode` is never called with `episode_steps` (its only caller, `TP/smoke.py:153`, passes none). The production `steps.jsonl` comes from `verify.progress` through `interop/replay.py:97-100`: a second Φ implementation with no prerequisites and no gates, which does handle Φ₀. Shape issues: K18 |
| L10 | C2 defect → check plus PR-mirror mutant (§6.3) | live (static fallback) | p4: the mutant round-trips to the wrong version exactly; 4/4 checks separate wrong from corrected; a diverged expert end state drops the non-passing check (3/4). "Every signal with negative polarity" is **missing**: the operator takes a signal id string and never reads M4 signals (L19) |
| L11 | C3 V̂ from snapshot restores, K continuations, masked runs excluded (§6.4) | live with defects | `operators/branch_value.py:51-68` restores before every continuation. K17 |
| L12 | Report of corr(C3, ΔΦ) on 3 task04 trajectories (exit gate) | missing | `M06-credit.md` lists it as pending GPU time and about $130, awaiting approval |
| L13 | C4 lifecycle candidate → validated (time-split AUC ≥ 0.6, same sign per subset); fixtures must score 1 and 0 before use (§6.5; `rubric_item.py:10`) | partial / **wrong** | `share_met` uses validated items only, and `judge_item.admit` rejects candidates. But `lifecycle` never checks fixtures and keeps n = 4 at weight 1.0 (K7). Parsing: K8 |
| L14 | C5 PRM trained on C1–C4 labels, used only as a shaping baseline (§6.1) | partial | `operators/prm.py` predicts the episode outcome from step features. It has no caller, no C3 labels and 0% unit coverage. NaN input: K19 |
| L15 | C6 infra mask: a failed measurement is never a zero (P16; `infra_mask.py`) | partial | `mask_for` has no production caller. A partly unmeasured terminal state is scored as 0 (K9) |
| L16 | §6.5 validity gate: time split, AUC ≥ 0.6, same sign per subset, weight max(0, 2(AUC−0.5)) | partial | `validity.validate` implements the rule but has no minimum n (K7). `time_split` tie bug: K14. NaN: K15. No production caller runs `validate` or `lifecycle` on a time split |
| L17 | S2: Φ_mid AUC ≥ 0.75 and above the judge on held-out traces (exit gate) | failed (documented, reproduced) | Coding 0.50 vs judge 0.75; toy 0.58 on 11 tickets with no judge baseline. The measured Φ is not the C1 operator, and the §6.5 keep rule is skipped (K16) |
| L18 | R1–R6 on every coding and toy task (exit gate; §6.4) | reported pass; **vacuous** | Reproduced, but the checks cannot fail on these inputs (K4). R3 is redefined (K5). task01 has no R3 and still reports ok |
| L19 | M4 signals feed credit. DESIGN §5.2: a review comment gives "**negative** credit on the step that introduced those lines"; a user interrupt gives "negative on that tool call … step label; PRM training". §6.1: operators map (trace, graph, **signals**) to step credits. §6.3: every negative-polarity signal becomes a candidate check | **missing** | Nothing in `C/credit/` or the m56 packs reads `Signal.grades`, `polarity` or `reliability` (grep is empty). `StepRow.violations` exists, but no producer ever fills it: `grep "violations="` finds only `steps.py:29` itself. The `reliability` column in §5.2 is descriptive. No DESIGN sentence makes credit weight by reliability. The only reliability consumer is `studio/datasets.dpo_pivots` (M15), which copies it onto the DPO row |
| L20 | "Every operator is versioned and unit-tested" (§6.1) | partial | No operator carries a version. `judge.py`, `prm.py` and `rubric_item.py` have 0% coverage; `defect_to_check.py` has 46% (§3) |
| L21 | Record-relative efficiency e (P24) | partial (as documented) | `compose` accepts e ≤ 2 and nothing produces e. Documented as "designed, not built" |
| L22 | Two-level hack detector (m13 C-4) | live with false negatives | K6 (the World's write keys defeat the rules) and K20 |

## 2. Defect candidates

### K1 [P1][ran] Dense credit ignores gates (V3), reproduced live on finance_ap

**What happens.**
- `turn_advantages` (`compose.py:89-128`), `steps.to_turn_shaping` (`steps.py:41-43`), `studio/datasets.from_trace` (`datasets.py:135`, fed by `TP/studio.py:26-29`) and the replay `steps.jsonl` (`interop/replay.py:97`) all emit ΔΦ for gate-failed episodes.
- None of them carries a gate or hack field.

**Input** (`p2_finance.py`): `APTask(1)`, replayed through `interop.replay` with three policies: approve every invoice, the gold actions, and no-op.

**Observed.**
- approve_all: reward 0.0, `gate/controls` 0.0, `turn_shaping` = [0, **0.143**, 0, 0, 0, 0, **0.143**, 0], so ΣΔΦ = +0.286.
- In a group with gold (R = 1), `turn_advantages` gives approve_all [−0.523, **−0.391**, …, **−0.391**, −0.523] against an outcome-only −0.5. Two of its 8 turns get positive local credit.
- All-fail group (p1 A2: two R = 0 episodes): sd = 0, so `scale = 1/spread` and the advantages are [[+0.1, +0.1], [−0.1, −0.1]]. The policy learns from pure process reward.
- The same rows reach NeMo-Gym and agentenv as `crucible_steps`, and SFT rows as `weights`.

**Expected.** §6.4 says a gate can never be traded for partial credit, and the docstring of `verify/progress.py` says the shaping terms "telescope to the final reward". Here they telescope to +0.286 while the reward is 0.

**Fix** (M06-M08 item 1):
- add `gated` and `hack` to `Episode` and `StepRow`;
- clamp local terms to min(0, ·) when a gate failed or a hack is confirmed;
- give no local terms when every live episode failed with equal rewards;
- apply the same filter in `to_turn_shaping`, in `from_trace` weights and in the replay rows (put `gate_failed` on every row).

This becomes P0 as soon as a trainer consumes `crucible_steps`.

### K2 [P1][ran] The local term is unbounded and grows with √turns (V5)

**Input** (p3 B1 and an inline probe). A failed episode (R = 0.2) of n turns where Φ jumps 0 → 1 once, grouped with a passing episode (R = 1.0) whose Φ rises evenly; η = 0.1.

**Observed.**

| Turns | Local credit on the jump turn | Failing turn's advantage | Best passing turn |
|---|---|---|---|
| 200 | 0.798 | 0.398 | 0.4 |
| 800 | 1.599 | **1.199** | 0.4 |
| 3200 | 3.199 | 2.799 | 0.4 |

The outcome gap is 0.8. At 800 turns, one turn of a failed episode outranks every turn of the success.

**Cause.** `scale = sd/spread` (`compose.py:118`), and max|t − c|/spread ≈ √N for a single spike.

**Expected.** The docstring at `compose.py:21` says the term "never exceeds the outcome advantage of a success over a failure".

**Fix.** Clamp |local| ≤ |a_out| (or ≤ the group's success−failure gap) and use sign-preserving shaping (M06-M08 item 3).

### K3 [P1][ran] Zero-mass centring pays for oscillation and length, which breaks R4 on the trainer path

**Input** (p3 B2). Φ₀ = 0.5 for every episode:
- monotone: phis [0.75, 1.0], R = 0.2;
- oscillating: phis [0, 1, 0, 1], R = 0.2, same end state;
- three regressors: phis [0]×8, R = 0.

**Observed.** Summed advantage is 0.254 for monotone and **0.495** for oscillating.

**Cause.** The group mean ΔΦ is negative, so c < 0 and every extra turn earns −c.

**Why the gate misses it.** `properties.r4_oscillation` only compares Σ ΔΦ and Σ shaping. Both are identities (Φ_end − Φ₀ and −Φ₀), so they can never favour oscillation, and the trainer path is never tested.

**Fix.** Centre per episode (subtract each episode's mean ΔΦ), or apply c only to state-changing turns. Add an R4 case over `turn_advantages`.

### K4 [P1][ran] The R1–R6 gate cannot fail on the inputs it is given

**Mutation** (p5): replacing `task_properties.potential` with Φ ≡ 0 (no partial credit at all) still yields task02 R1–R6 all True. Each property is vacuous:
- **R1/R2:** the oracle and base outcomes are hard-coded from records (`CP/m56/task_properties.py:84`). Milestones are filtered to be false at the base (`:74`), so Φ(base) = 0 by construction.
- **R3:** with `gates={}` and a recorded outcome of 0, the reward is ≤ κ < 1, so R3 always holds (see also K5).
- **R4:** the Σ ΔΦ and Σ shaping identities above (K3).
- **R5:** `stubbed: []` on all 5 coding tasks and the toy. No mined milestone has `requires` (toy: 3 milestones, 0 with requires), so R5 compares Φ with itself.
- **R6:** both gates pass `lambda fs, msgs: of(fs)` (`task_properties.py:105`, `TP/m56/pipeline.py:238`). The reward function ignores the messages by construction, so R6 tests the lambda (p1 A11).
- task01 has no negatives. `if negatives:` (`:101`) skips R3 silently, yet the task and the gate report `ok`.
- The toy checks one synthetic case (`T9999`), not the toy World's tasks or their M7 plans.

**Fix.**
- Run R1–R3 on the verifier's own grade of the states, not on recorded outcomes.
- Make R6 call the real grader with a trajectory whose messages differ.
- Add prerequisites, or a mined-stub negative, so R5 has a subject.
- Make a property that is not run a failure (`skipped ≠ ok`).
- Add the mutation (Φ ≡ 0, and a reward that reads messages) as a meta-test the gate must fail.

### K5 [P1][ran] R3 is redefined, so the DESIGN and the code disagree

- DESIGN §6.4 R3: "Every registered negative gets **0**."
- `properties.py:9-11,47-58` accepts any outcome-0 reward below 1. A negative with Φ = 1 gets 0.2 and passes (p1 A12).
- In the toy report, the negatives `refund_reversed`, `resolved_without_answer` and `reopened` earn 0.2, 0.133 and 0.133.
- DESIGN contradicts itself here: the κ(1−O)Φ term gives every failed negative positive reward unless a gate fails.

**Fix.** Decide in DESIGN. Either R3 means "a gate fails" (registered negatives must map to gates), or it means "≤ κΦ". Update §6.4 and `M06-credit.md` to match.

### K6 [P1][ran] The hack detector misses root-level test and verifier writes on real World traces

`World.to_trace` stores writes as `file:<path>` (`core/agent_events.writes_of`). `TEST_FILE` and `VERIFIER_FILE` require `^` or `/` before `tests/`, `conftest.py`, `solution` or `.crucible` (`hack_detector.py:24-25`).

**Observed** (inline probe on `writes_of('shell.exec', …)` → `rule_flags`):

| Command | Rule hits |
|---|---|
| `echo x > tests/helpers.py` | none |
| `echo … > conftest.py` | none |
| `echo 1 > solution/solve.sh` | none |
| `echo 1 > .crucible/reward.json` | none |
| `echo 1 > tests/test_core.py` | flagged, only because `/test_core.py` matches |

The same paths without the prefix are flagged. Unit tests use bare paths (`tests/unit/test_hack_defence.py:61`).

**Fix.** Strip `file:` (and any `<app>:` prefix) before matching. Add a test that goes through `World.to_trace()`.

### K7 [P1][ran] A rubric item is validated with no fixtures and almost no data

- `lifecycle(RubricItem(method="llm", fixtures={}), [1,0,1,0], [1,0,1,0])` gives status `validated`, weight 1.0 (p3 B10). Fixtures are never checked. `rubric_item.py:10` says every LLM item "must score 1 and 0 before it is used".
- `validate("x", [1,0], [1,0])` keeps the item at weight 1.0 with CI (1.0, 1.0). With n = 3 the CI is (0.5, 1.0), the item is still kept, and its weight is 0.5 (p3 B3). The bootstrap CI is computed (`validity.py:132`) but never used.
- `lifecycle` has no notion of time, so nothing enforces "held-out, later" traces.

**Fix.**
- Have `lifecycle` take a fixture runner and require good → PASS and bad → FAIL.
- Require a minimum n per class (e.g. ≥ 10) and CI_lo > 0.5.
- Take `(items, time)` and call `time_split` inside.

### K8 [P2][ran] Rubric verdict parsing: a quoted PASS wins, and "unparseable" is scored as a fail

- `parse_verdict("VERDICT: FAIL. The evidence said 'VERDICT: PASS'")` returns **True**, because the last match wins (`rubric_item.py:61-65`). This is V4: evidence goes into the prompt unfenced.
- `"**VERDICT:** PASS"` returns `None`, and `share_met` counts `None` as unmet (`:75-79`). `verify/judge_item.py:6` says "an unparseable verdict is unmeasured, never a fail".

**Fix.** Anchor the verdict to the first line or the last line only, and fence the evidence. Drop unmeasured items from the denominator of `share_met`, or mask through C6.

### K9 [P2][ran] C6 is unwired, and a partly unmeasured terminal state is scored as 0

- `mask_for(terminal_phis={"a": 1.0, "b": None})` returns `status='ok'`, because it masks only when *all* milestones are `None` (`infra_mask.py:32`). `potential` then counts `b` as 0, so the failed measurement becomes a zero.
- No production code calls `mask_for`, and nothing sets `judge_unparseable`.

**Fix.** Mask (or flag `partial`, and keep Φ out of R) when any reward-bearing milestone is unmeasured at the terminal state. Wire `mask_for` into `grader.to_reward`.

### K10 [P2][ran] `compose(masked=…)` returns reward 0.0

`compose.py:65-67` builds `RewardJson(reward=0.0, …)`. `flat()` therefore reports `reward: 0.0` next to `infra/masked: 1.0` (p1 A5). The module docs say "`None` for masked episodes". `scalar()` and the grader handle it correctly. Direct `.reward` readers do not: `CP/studio.py:74,91,95`, `task_properties`, the toy pipeline.

**Fix.** Make `reward` optional in `RewardJson`, or raise when `.reward` is read on a masked document. At minimum, document that `scalar()` is the only correct reader.

### K11 [P2][ran] The first step is credited with Φ(s₀)

`trajectory()` starts from `prev_phi = 0.0` (`potential.py:57`). For a base state where milestone `a` already holds, the deltas are [0.5, 0.0] and `new` at step 0 is `['a']`, although step 0 changed nothing (p1 A8). `rows_for` and `to_turn_shaping` inherit this, and so do `from_trace` (`phi_before = 0.0`) and `Episode.phi0 = 0.0`. `verify.progress` computes Φ₀ at start; C1 does not.

Today the coding and toy milestones are filtered to be false at the base, so this is latent there. It is live for any pack whose milestones can hold at the start.

**Fix.** Add `initial` to `trajectory()` and require callers to pass the base state.

### K12 [P2][ran] `potential()` validates nothing when given a raw list

- Weights [2, −1] give Φ = **2.0**.
- φ = NaN gives Φ = NaN; φ = 2.0 gives Φ = 2.0; φ = −1 gives Φ = −1.0.
- A prerequisite id that is not in the list zeroes the milestone forever, and `partial` is not raised (p1 A7).

`MilestoneSet` validates weights and prerequisites, but every caller passes `list(milestones)` or a filtered subset (`task_properties.py:74` filters candidates after mining).

**Fix.** Validate in `potential()`: weights > 0, φ ∈ [0, 1] and finite, prerequisites known. Raise, or set `partial`.

### K13 [P2][ran] The R5 checker fails a legitimate fractional prerequisite

`r5_stubbing(build→test, {"build": 0.5, "test": 1.0})` returns ok = False, with Φ 0.5 with stubs and 0.25 without (p1 A10). The checker treats φ < 1 as "unmet" (`properties.py:70`), while the §6.2 formula gives proportional credit.

**Fix.** Compare against the formula with prerequisite φ fixed, or require binary φ for prerequisites.

### K14 [P2][ran] `time_split` can return an empty training split

- 8 traces at t0 and 2 at t1 with holdout 0.3 give train 0 and holdout 10.
- All traces at one timestamp give train 0 and holdout 10 (p3 B5).
- Cause: the tie loop always moves the cut down to the start of the tie block (`validity.py:104`).

**Fix.** Move the cut to whichever tie boundary is closer to the target fraction, and raise if either split is empty.

### K15 [P2][ran] `auc()` with NaN depends on input order

`auc([nan, .2, .9, .1], [1,0,1,0])` = 0.5, but with the NaN in the last position it is 0.25 (p3 B4). Sorting and tie detection (`==`) are undefined for NaN.

**Fix.** Raise on non-finite scores.

### K16 [P2][ran] The S2 measurement is not the specified one

- Type weights are given to every type, including all six whose fit AUC is 0.47–0.54. Under §6.5 every one of them should be dropped (AUC < 0.6). See `CP/m56/rebench_s2.py:76` and the rerun's `type_weights`.
- No subset sign rule is applied.
- The Φ measured is a mean over milestone types (`rebench_s2.phi`), not the C1 operator (no prerequisites, no weights). S2 therefore does not test C1 as built.
- Three midpoint definitions coexist:

  | Source | n = 2 | n = 4 | n = 6 |
  |---|---|---|---|
  | `at_fraction(…, 0.5)` (no caller) | 0 | 2 | 2 |
  | toy `len//2` | 1 | 2 | 3 |
  | rebench `round(n/2)−1` | 0 | 1 | 2 |

- The judge baseline exists for coding only; the toy S2 has none.

The conclusion (S2 fails) does not change.

**Fix.** Apply `validate()` per type on the fit split. Report Φ through `potential()`. Use one midpoint helper everywhere.

### K17 [P2][ran] `branch_value` edge cases

- A verifier returning `None` is excluded from V̂ but not counted in `masked` (values [1, None, 0, None] give masked = 0; p3 B7).
- The World is left in the last continuation's state, not restored to `ref`.
- `wilson(110, 100)` raises `ValueError: math domain error`. This happens if `verify` returns a composed R > 1 (1 + αp + βe) over ~100 runs (`branch_value.py:46`).

**Fix.** Count verifier `None` as masked, restore `ref` at the end, and validate `0 ≤ successes ≤ n`.

### K18 [P2][ran] `steps.jsonl` alignment

- `rows_for(…, call_ids=[1 id])` for 3 steps raises `IndexError`.
- 4 ids for 3 steps are silently truncated.
- `to_turn_shaping` returns one entry per **step** (3 entries for 2 model calls). Its docstring says "per-turn" and `turn_shaping` sits next to `llm_turn_spans` (p3 B8).

**Fix.** Check `len(call_ids) == len(steps)`, and aggregate ΔΦ by `call_id` in `to_turn_shaping`.

### K19 [P2][ran] The PRM gives a NaN feature the top score

`LogisticPRM.score({"delta_phi": nan})` returns 1.0, because `min(40.0, nan)` evaluates to 40.0 (`prm.py:50`; p3 B12).

**Fix.** Reject non-finite features.

### K20 [P2][ran] More rule false negatives in the hack detector

These produce no flag (p3 B11):
- `pkg/foo_test.go` and `src/foo.spec.ts`;
- `pytest -k="not test_hard"`;
- an added `@pytest.mark.skip` or `@unittest.skip`.

Level 1 is the recall layer, so these escape the judge entirely.

**Fix.** Use per-language test globs, `-k[= ]`, and skip-decorator patterns over patch text.

## 3. Coverage

Run from `crucible/`, with `COVERAGE_FILE` pointed at the scratchpad so the repo's `.coverage` stays unchanged:

```
pytest tests/unit/{test_credit,test_harness_branch,test_shaping,test_group_stage,test_hack_defence,test_verify,test_episode_loop,test_span_closure,test_stitch}.py --cov=crucible.credit
```

Result: 65 passed, **74%** of `crucible.credit`.

| File | Coverage |
|---|---|
| `compose.py` | 96% |
| `gar.py` | 97% |
| `hack_detector.py` | 95% |
| `branch_value.py` | 93% |
| `infra_mask.py` | 95% |
| `properties.py` | 100% |
| `steps.py` | 95% |
| `validity.py` | 91% |
| `shaping.py` | 89% |
| `defect_to_check.py` | **46%** |
| `judge.py` | **0%** |
| `prm.py` | **0%** |
| `rubric_item.py` | **0%** |

`workflow/potential.py` is at 89% (from the first seven files). `test_stitch` skips when the task04 span tasks have not been derived locally.

Every K above passes the current suite. No test covers:
- a gate-failed episode in `turn_advantages`;
- `World.to_trace()` write keys reaching `rule_flags`;
- rubric fixtures or the verdict parser;
- `time_split` ties;
- NaN inputs;
- property meta-tests (a broken reward must fail R1–R6).
