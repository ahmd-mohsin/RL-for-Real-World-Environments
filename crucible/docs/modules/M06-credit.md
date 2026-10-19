# M6 `credit` — dense credit and reward composition

**Status: built (2026-10-09); audited 2026-10-12 (`docs/audit/2026-10-12/M06-credit.md`). R1–R7 pass on the toy and on 4 of 5 coding tasks (task01 has no failing run to test R3). S2 fails as specified (finding below). The C3-vs-ΔΦ report on task04 is pending (GPU plus a paid model).**

- **Code:** `src/crucible/credit/`, `src/crucible/actors/harness.py` (native harness), coding pack `m56/`.
- **Tests:** `tests/unit/test_credit.py`, `tests/unit/test_harness_branch.py`, `tests/unit/test_m6_audit.py`.
- **Reports:** `reports/m6_s2_swe_rebench.json`, `reports/m6_properties_coding.json`, `reports/m56_toy.json`, `reports/m56_gate_coding.json`.

## What it is

| Piece | What | Built on |
|---|---|---|
| `operators/potential.py` (C1) | Φ = Σ wᵢ·φᵢ·Πⱼ∈pre(i) φⱼ / Σ wᵢ. An unmeasured value keeps its last value; a never-measured one is flagged `partial`. Shaping with Φ(terminal)=0 telescopes; `at_fraction` gives Φ at the midpoint | DESIGN §6.2 |
| `operators/defect_to_check.py` (C2) | A (wrong, corrected) pair becomes static checks that must separate the two and pass the expert's end state, plus a PR-mirror mutant (the correction reversed) as an R3 negative | SWE-smith PR-mirror; agentenv severity localization |
| `operators/branch_value.py` (C3) | Restore the `WorldRef`, then run K continuations, each with a fresh policy; V̂ comes with a Wilson CI and masked runs excluded. ΔV̂ is the advantage. `correlation()` compares ΔV̂ with ΔΦ, split into state-changing and investigation steps. Every estimate records `harness@version` | VinePPO; GitSwarm matched resumption; research 06 (quiescent branch points) |
| `operators/rubric_item.py` (C4) | Item schema: tier, method, question, `pass_anchor` (truth plus accept/reject equivalents), fixtures. Lifecycle: candidate → validated (time-split AUC ≥ 0.6, same direction on every subset) or retired. Strict binary judge prompt | MiMo `verifier_meta`; MiMo text judge; prime-environments' fail-closed parsers |
| `operators/prm.py` (C5) | L2 logistic regression over step features. Interface `fit` / `score`. A shaping baseline only | — |
| `operators/infra_mask.py` (C6) | Typed mask from failure kinds, `answer_exposed`, unmeasured gates or milestones, and an unparseable judge. A failed measurement is never a zero | NeMo-Gym `mask_sample`; verl `is_infra`; MiMo `testbed_corrupted` |
| `compose.py` | R = G·[O(1+αp+βe) + κ(1−O)Φ], with κ ≤ 0.3, α ≤ 0.1, β ≤ 0.05 enforced and `None` for masked episodes. Per-turn advantages in two modes (below), centred to zero net mass per group, masked episodes excluded from group statistics | verl fork (mass conservation, group-mean imputation); Dr.GRPO scaling |
| `steps.py` | `steps.jsonl` rows with `call_id` (the model call the credit attaches to) and `harness`; `to_turn_shaping` for a trainer's turn-span operator | verifiers v1 message graph; OpenEnv capture graph |
| `validity.py` | Tie-aware AUC, time split, subset sign rule, weight = max(0, 2(AUC−0.5)), group bootstrap and paired bootstrap | DESIGN §6.5 |
| `properties.py` | R1–R7 as reusable checks. Crafted negatives must hit a gate (0); honest failures keep κΦ; R7 bounds a shallow mutant's partial credit; untested properties are reported, by-construction ones say so | DESIGN §6.4; NeMo-Gym verifier fixtures; Repo2RLEnv control evidence; OpenEnv's quick-submit evidence |
| `signed.py` | `adv_signed`: a marked call's tokens go to 0 on a positive rollout and ×κ on a negative one, each sign's batch mass conserved (trainer: `format_penalty_mode="signed"`) | PORT of Xiaomi's verl fork `signed_rebalance.py` (Apache-2.0) |
| `judge.py` | The LLM-judge outcome baseline: task plus trajectory prefix → P(success), 3 sampled passes averaged with spread reported, anchored parse, masked if unparseable | GitSwarm judging; prime-environments `torch_ao`/`nextjs` parsers |
| `actors/harness.py` | `BashLoopPolicy`, a port of the mini-swe-agent v2 loop onto World shell tools: one bash block per turn, head/tail-truncated observations, explicit submit, format-error templates (3 strikes), cut-off replies treated as errors | mini-swe-agent (MIT) |

## Correction to DESIGN §6.4: shaping return-to-go is a baseline, not credit

With Φ(terminal) := 0 and γ = 1, the shaping return-to-go from turn t telescopes to −Φ(s_{t−1}). That is a state baseline: unbiased, and it reduces variance, but it is independent of the action taken at t. `compose.turn_advantages` therefore offers:
- `mode="baseline"`: DESIGN's formula, for variance reduction only;
- `mode="delta"`: η·ΔΦ_t, the local credit for the turn's own action. This is a process-reward term (biased), so it is kept small, centred to zero net mass, and scaled by the group's outcome spread.

Also corrected: DESIGN cited verl's `arvo_penalties.py` as a turn-level advantage rewriter. It only marks spans, applies a group length penalty and rebalances signs; its `apply_tool_penalty` path also raises `TypeError` as written. Crucible implements the shaping operator itself.

## Exit gate

| Requirement (ARCHITECTURE M6) | Result |
|---|---|
| S2: Φ at the midpoint predicts the outcome, AUC ≥ 0.75, above an LLM-judge baseline, on held-out traces | ❌ **Coding** (SWE-rebench: 913 OpenHands attempts on 80 issues, weights fitted on the earliest 70% of issues, 200 attempts on 23 later issues held out): Φ_mid AUC 0.50 (CI 0.42–0.58) vs the judge's 0.75. The judge (claude-haiku-5.5, 3 passes, $0.90) is better by 0.25 (CI 0.10–0.38). **Toy:** Φ_mid AUC 0.58 on 11 tickets |
| R1–R6 on every coding and toy task | ⚠️ (M6 audit: the earlier ✅ could not fail: Φ ≡ 0 passed all six). Now R1–R7 with untested properties reported. **Coding:** pass on 01v2, 02, 03, 04; **task01 has no failing run, so R3 is untested and the gate does not pass**. Failing frontier runs are honest failures and keep bounded partial credit; shallow mutants reach κΦ 0.027–0.078 (R7 ≤ 0.1). **Toy:** pass; the crafted negatives (unapproved refund, reversed refund, resolved without an answer) are gates and score 0 |
| corr(C3 branch values, ΔΦ) reported for 3 task04 trajectories | ⏸ Operator and harness are built and tested (V̂ from restored snapshots in a real World). The run needs K continuations per branch point of 3 task04 trajectories on the lab RTX 5090 with a paid model, about $130 estimated; not started without approval |

## Findings (the S2 result in detail)

1. **Φ cannot predict anything before the agent has changed the state.** At the midpoint of the tool calls, only 30% of failing and 36% of successful OpenHands attempts have satisfied any milestone. Edits happen between 50% and 75% of an attempt, and Φ at 75% equals Φ at the end. A state potential (P12) is blind during exploration; a judge reading the exploration is not. As specified, S2 asks Φ to do something it cannot do by construction on exploration-heavy harnesses.
2. **Where it matters for RL, Φ is the better signal.** At the end of attempts, held out:

   | Predictor | Pooled AUC | Within-issue AUC |
   |---|---|---|
   | Φ | 0.66 | 0.72 |
   | Judge | 0.75 | 0.67 |
   | Both combined | 0.77 | 0.77 |

   The judge's pooled edge comes from issue difficulty, since it reads the task. Within an issue, which is what a GRPO group compares, Φ discriminates better, and the two signals are complementary.
3. **Proposed revision of S2, pending a decision:**
   - (a) Measure Φ where state has started changing: at 75% of the attempt, or at the midpoint between the first and last state change.
   - (b) Report within-task AUC, the quantity group-relative advantages use.
   - (c) Credit for the exploration phase comes from C3 values or a judge-trained C5 PRM, not from Φ.
4. **Gold-token milestones are brittle across implementations.** Correct alternative solutions score low Φ: task01's passing Fable run 0.49, and the v1-passing task04 runs about 0.32 against the v3 oracle. "Located" milestones (the expert's block was changed) were the most predictive type. Behavioural milestones (the expert's fail-to-pass tests) are the expected fix; they need the issues' images.
5. **Reasoning models answer with an empty string when the token cap is small.** The judge's first smoke test returned empty text with `finish=length` at a 400-token cap, because hidden reasoning used it all (M2 saw the same). The judge's default cap is now 2,500.

## Per-turn credit: one rule, hard gates, present or refused (2026-10-19; `docs/research/improvements/2026-10-14/M06-credit.md` R1–R3)

Per-turn credit (`turn_weight > 0`, off by default) was wrong in three places; all three are fixed, each reproduced
before and after through the real code paths.

1. **Gates mean hard rules (R2).** A grade report already says which checks are hard rules and which are outcome or
   score checks (`meta["kinds"]`); the record writers dropped it, so every ordinary failure counted as rule-breaking and
   an honest attempt that made progress trained *below* a no-op (−1.113 vs −1.000 at η 0.1). Now:
   - `GradeReport.gate_kinds()`; every record writer (native, workspace, agentenv harnesses; the runner; `world_env`;
     the slime bridge) carries the kinds, and `RolloutRecord.gate_kinds` is new (`crucible.rollout/v1.2`, excluded when
     empty, so older records keep their bytes). `gates` itself is unchanged (the bench gate profile reads every check).
   - `credit.compose.hard_gate_failed(gates, kinds)` is the one test: a failed gate whose kind is `"gate"`, or
     undeclared (what every older record meant). `train.credit.turn_credit_map` and the slime bridge both use it.
   - A rule-breaking or hacked episode gets **no local term at all** (every call trains at the outcome advantage). The
     clamp it replaces, applied after centring, left the call that bought progress by breaking the rule the least
     punished of its episode (−0.5 against −0.75).
   - Hacks are seen whichever way they were flagged: `hack/confirmed`, `hack/zeroed` (`group_ops.zero_hacks`) or
     `gar/hack_zeroed` (D2).
   - finance_ap, the toy backlog and the coding dense task declare their outcome gates; a multi-step task's
     `step/<name>` gate is an outcome unless one of the step's own hard rules failed.
2. **One rule on every trainer path (R1).** `train.credit.credited_records` is the only place raw ΔΦ becomes trainable
   credit; the torch trainer and the rLLM bridge call it (the verl loop and slime already applied the rule). The rLLM
   bridge used to send raw `seq_adv + η·ΔΦ`, which rLLM's default (sync) tinker path trains verbatim. It now sends the
   group's outcome advantage (`train.credit.outcome_advantages`, R − mean over live rollouts, unless `seq_adv` is given)
   plus the rule's credit, sub-calls and child agents at their parent's credit. `backend="verl"` with
   `turn_weight > 0` and `trajectories(turn_weight > 0)` (one rollout cannot apply a group rule) are refused.
3. **Present or refused (R3).** Live M12 rollouts recorded no ΔΦ, so `turn_weight > 0` silently trained outcome-only.
   - `world_env.rollout` attaches a `ProgressTracker` for tasks with `subgoals`, as the M10 native harness does; the
     calls' ΔΦ sum to Φ_final − Φ_0 (the toy backlog: +0.25 per resolved ticket, 0.5 in total for 2 of 4).
   - The torch trainer and the rLLM bridge raise when `turn_weight > 0` and no live rollout carries `Turn.advantage`
     (`train.credit.require_turn_credit`; `None` is absent progress, 0.0 is progress that did not move).
   - The verl agent loop cannot raise (verl would mask every sample), so it marks every output of such a group
     `crucible_credit_fallback="no_turn_credit"` (`"legacy_worker"` on `use_v1: false`, `"timeout"` after the
     rendezvous) and warns once per process.
4. **The raw reward survives the verbosity penalty (D3).** `verbosity_group_fn` keeps `raw/reward` (now defined in
   `credit/shaping.py`); the runtime's pass-rate input reads it, so a success 0.5 % over budget is still a success for
   GAR and the pass-rate alarm.

Tests: `tests/unit/test_m6_turn_credit.py` (properties over random groups with masked, rule-breaking, outcome-failed
and hacked members, RLM sub-calls and child agents; live backlog rollouts across seeds); `test_m12_audit_bridges.py`
(the rLLM tinker contract against rLLM's own code; the verl loop's `no_turn_credit` marking). Each fix was checked by
re-introducing the defect: every one fails its tests.

Not changed (listed in the research report §7b): the bound uses the record reward's outcome advantage, not a GAR or
shaped one (D1); centring is per call, not per token (D4).

## Not done

- The C3 task04 correlation run (approval for GPU time and about $130).
- Behavioural milestones on SWE-rebench.
- A capture-proxy upgrade (Responses dialect, token ids, budgets), noted in research 01/02/06.
- Trainer adapters that consume `steps.jsonl` (M12).
