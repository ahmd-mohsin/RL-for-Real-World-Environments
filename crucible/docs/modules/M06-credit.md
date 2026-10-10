# M6 `credit` — dense credit and reward composition

**Status: built (2026-10-09). R1–R6 pass on every coding and toy task. S2 fails as specified (finding below). The C3-vs-ΔΦ report on task04 is pending (GPU plus a paid model).**

- **Code:** `src/crucible/credit/`, `src/crucible/actors/harness.py` (native harness), coding pack `m56/`.
- **Tests:** `tests/unit/test_credit.py`, `tests/unit/test_harness_branch.py`.
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
| `properties.py` | R1–R6 as reusable checks | DESIGN §6.4; NeMo-Gym verifier fixtures; Repo2RLEnv control evidence |
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
| R1–R6 on every coding and toy task | ✅ **Coding:** tasks 01, 01v2, 02, 03, 04, with recorded verifier outcomes and oracle-derived milestones (filtered). The oracle has Φ = 1, the base 0. Failing frontier runs earn bounded partial credit (Φ 0.27–0.61, reward 0.05–0.12). **Toy:** all six on the state-only helpdesk reward |
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

## Not done

- The C3 task04 correlation run (approval for GPU time and about $130).
- Behavioural milestones on SWE-rebench.
- A capture-proxy upgrade (Responses dialect, token ids, budgets), noted in research 01/02/06.
- Trainer adapters that consume `steps.jsonl` (M12).
