# Crucible Part 2: a domain-generic, self-improving framework for RL task and environment synthesis

**Date.** 2026-10-29. **Status.** Consolidated design, for the lead to build from and to send to collaborators.
**Companion.** Part 1 (the async RL trainer and the simulated World) is in `part1-framework-improvements.md` in this
folder. This document assumes it and only names the Part 1 items that Part 2 depends on.

**What Part 2 is.** Part 2 turns the World into environments and tasks with verifiers. It must reach 500–1000 tasks in
any domain, for the student Qwen3.6-35B-A3B now and a 397B student later. It must deliver:
- intent alignment: the verifier, the World and the prompt all exercise the intended skill;
- rigorous verifier testing;
- forced diversity;
- difficulty set per student;
- failure analysis from traces;
- a recursive self-improvement loop over the verifier, the World and the task;
- a route from enterprise traces to tasks.

**What exists today.** One live lane: multi-language debugging (`notes/00-crucible-map.md`). It mines real fixes,
executes every gate, and has kept 13 training tasks and a sealed 10-task benchmark at $30–130 of Tinker plus hours of human time
per kept task. Its contracts are patch-shaped. This document makes that lane one plug-in of a generic factory.

**How to read the labels.**

| Label | Meaning |
|---|---|
| (2602.06820) | An arXiv id. Every claim about a paper traces to the six extracts or the per-paper cards in `cards/`. |
| **[Proposal]** | Our own design choice or number. It is not a paper result. |
| [PARTIAL] | The card read only part of the paper (§1.2 lists them). |
| [uncertain] | The card or extract flags the number or claim as uncertain. |
| [contaminated] | The paper trained or synthesised on material from the benchmark it reports (§1.4). |
| [no-control] | The result has no matched or equal-budget control. |
| P0 / P1 / P2 | P0: build before the next batch of tasks. P1: the generic core. P2: later or at scale. |

**Stage and gate vocabulary.** Stages F0–F12 and gates G1–G14 keep the names of `framework-plan.md` §3. New gates are
G5b and G15–G24 (§3.7 has the full table, and §14.1 explains one numbering clash). Gates that test a harness, a judge or a
simulated person, rather than one task, get the prefixes H, J and U.

---

## 0. The framework on one page

### 0.1 The stage graph

```text
  enterprise traces / logs ------------------------------+
  (redact first, section 8)                              |
                                                         v
  brief --> F0 DomainSpec --> F1 App + tool synthesis --> F2 Seed, distractors, exact reset
              ^  (axes, skills,      (state API, blind frozen      (double-replay hash,
              |   knobs, families)    tool tests, conformance)      latents, people, clock)
              |                                                          |
              |       +--------------------------------------------------+
              |       v
              |  F3 Grounded proposal --> F4 Oracle by construction --> F5 Verifier + negative library
              |  (cell, World slice,      (executed action program,     (closed-world diff, answer checks,
              |   K nearest + failed,      tool-only, run twice)         milestones, mutants, twins, honeypots)
              |   ledger, Assignment)                                         |
              |       ^                                                       v
              |       |                                        F6 Gates: CPU first (G1-G24, H, J, U)
              |       |                                        <= 3 side-attributed repair rounds
              |       |                                                       |
              |       |                                                       v
              |       |                     F7 Behaviour signature --> F8 Dedup (text, embedding, behaviour)
              |       |                                                       |
              |       |                                                       v
              |       |   F9 Difficulty per (task, student, checkpoint, simulator, time mode)
              |       |   predictor rank -> frontier sample audit -> sequential k=4->8 -> IRT
              |       |                                                       |
              |       |                                                       v
              |       |        F10 Intent audit (ablations, fault sides) --> F11 Archive, pool, curriculum
              |       |                                                       |
              |       |                                                       v
              |       +----- task loop ------- F12 Feedback <------------ training rollouts
              |                                (p-hat, executed failure attribution,
              |                                 hack flags + quarantine, drift)
              |                                   |            |             |
              +------------ generator loop -------+   World loop -> F1/F2   verifier loop -> F5
                            (pipeline doctor)

  Guards on every loop: evaluator out of reach (proxy-enforced), frozen anchors + locked set, canaries,
  equal-budget controls, round caps for judge-only loops, never select on visible score, learned reward,
  student reward or chain of thought.
```

Two order rules matter. **Cheap before paid:** every deterministic gate (F6 CPU half, F7, F8) runs before the first
student rollout (F9). **Execution decides:** a model may write a World, an oracle, a check or a repair, but each is held to
executed gates and cross-checked by a different model family.

### 0.2 Design principles and their strongest evidence

| # | Principle | Strongest evidence |
|---|---|---|
| 1 | **The LLM proposes; execution decides.** No ground truth is a model's opinion. | Without execution feedback SWE-Factory's build rate fell 50.2% → 0.3% (2506.10954). Running the reference during the build took valid environments from 19.5% to 84.8% (GPT-5.6-sol) and 19.4% to 73.0% (DeepSeek-V4-Flash) (2610.14163). Execution-checked GUI RL gained +14.61 (DART, 2509.23866) against +2.5 for VLM-judged RL (ZeroGUI, 2505.23762), indicative only. |
| 2 | **Solution by construction; sampling is an audit.** Every task ships an oracle that ran in the World. | AWM had no oracle, and 23.7–28.2% of its training tasks were never solved once (2602.10090). Only 5.2% of SCA's self-made tasks survived its executed filters (2506.01716). |
| 3 | **The two-sided filter is not enough.** "Fails on start, passes on oracle" selects lenient verifiers; must-fail negatives are required. | In SCA, the solution filter alone *raised* false positives; adding ≥ 3 near-miss failure cases removed all of them (2506.01716). |
| 4 | **Grade the frame, not only the postcondition.** Pass iff expected ⊆ diff ⊆ expected ∪ allowed. | Unrequested changes survived verifiers: AppWorld "Extra" edits passed 39/48, 6 of them false positives (2610.09142). WorkBench side-effect rates reached 54% (2405.00823). |
| 5 | **Red on start.** Target checks fail on the untouched start state; preservation checks pass on it. | Dropping WildAWM's initial-unsatisfaction gate cut its ablation average 31.86 → 24.08 and in-domain 21.59 → 3.05 on Qwen3.6-35B-A3B data (2610.26828). |
| 6 | **Communicability is checked statically, before any rollout.** Every value, rule, format and location a check reads is in the instruction, the start state or the simulated user's known information. | Removing requirements and interface text cut GPT-5 on SWE-Bench Pro 25.9 → 8.4 (2509.16941). 25.7% of 34,285 tasks in 168 benchmarks had a major issue, mostly instruction defects (2605.26079). |
| 7 | **Judges never award a pass on write tasks, and never see the agent's own claims.** | No web judge exceeded about 70% precision (2504.08942). A rubric-free judge scored κ 0.018 (2509.26490). One token such as "Thought process:" fooled judges 60–90% of the time (2507.08794). |
| 8 | **Remove the affordance; do not ask the model to behave.** | Read-only tests cut shortcuts to 0.9% against 19.5% with editable tests (2610.03055). A validity prompt cut hacking by only 6.2 pp (2608.30724). Four hardening toggles cut exploits 6.5% → 0.8% at no measured cost to success (2605.02964). |
| 9 | **The grader lives outside the agent's reach and fails closed.** | Working exploits existed for 10 of 10 audited benchmarks (2605.12673). A separate verifier environment removed 82–95% of exposed vectors (2609.11028). A meta-agent pulled 591 dev labels out through grader tracebacks (2606.04455). |
| 10 | **A chain of thought may raise a flag, never clear one.** | Planted hacks were exploited on > 99% of prompts but verbalised on < 2% in 5 of 6 environments (2505.05410). CoT-monitor pressure in the reward taught obfuscation (2503.11926). |
| 11 | **A hack flag quarantines the task at once.** | Hacking learned on 10% of the data stayed near full strength (2508.17511). Above 2% hacked episodes, every misalignment evaluation rose (2511.18397). Re-exposure brought hacking back 7.5× faster than from scratch (2606.09711). |
| 12 | **A different model family checks every oracle, check and repair; gate code is hidden from builders.** | Without AEvo's protected evaluator, 2 of 3 runs hacked the reward (2605.13821). GLM-5.2 was the most frequent hacker on HVTB (glm-5.2: 44.9% in the ablation setting; 4.5% even under an explicit ban) (2608.22103). DeepSeek V4 Pro hacked *more* under a validity prompt (+8.3 pp) (2608.30724). |
| 13 | **Diversity comes from structure, not from prompts or model mix.** | Cross-model output similarity was 0.71–0.82 (2510.22954). Over 15 rounds RST's nearest-neighbour similarity rose 0.223 → 0.464 while its label entropy stayed flat (2608.05466). Removing GRADUS's coverage ledger dropped telecom 0.75 → 0.54, its largest ablation (2610.27781). |
| 14 | **More distinct environments beat more tasks per environment.** | At fixed ~35K records: 2× environments 56.0, 2× queries per environment 53.8, 2× solutions per query 53.9 (2609.04148). The nuance: at small budgets fewer Apps can win for SFT (2610.37648). |
| 15 | **Hardness comes from structure, and every knob change must move the behaviour signature and the measured p̂.** | SETA's operators moved Qwen3-8B in the declared direction only 77% and 60% of the time (2607.10891). Reasoning Gym's own "hard" configs moved parameters the easy way (2505.24760). All of OpenThoughts-Agent's task augmentations failed to beat untouched tasks at 10K (2606.24855). |
| 16 | **Difficulty belongs to (task, student, checkpoint, user simulator, time mode).** | Swapping only the user model moved success 67.0% → 75.9% and moved 11 of 18 tasks to another bin (2601.17087). GPT-5 (high) scored 0.0 on Gaia2's Time split and 34.4 in instant mode (2602.11964). The learner was the best evaluator of its own data (2409.00920). |
| 17 | **Bands exist for learnability, and hack rates rise at the hard edge, so hard variants are re-gated.** | Balanced bands beat one-sided cuts; (0.3, 0.7) gave 30.1 against plain GRPO's 26.3 (2504.03380). Shortcuts rose from 40 at levels 1–10 to 458 at levels 11–20 (2604.15149). |
| 18 | **Reward-bearing state lives in code. Simulated people and LLM-simulated tools narrate; they never decide reward.** | Swapping only the simulator moved agent averages 67.9% → 29.3% (2604.10866). Direct-prompted world models reached 97% in simulation but 3% replayed in the real environment (2610.06100). |
| 19 | **Traces become Worlds, never text.** | Re-solving recovered intents gave 52.1 against 36.7 for SFT on the source trajectories, which was below the 47.0 base (2609.04148). |
| 20 | **Loops never select on visible score, learned reward, student reward or chain of thought; claims come with power.** | An unguarded verifier collapsed to always-pass while downstream task scores stayed equal (2610.11464). The best-by-visible-score node was a 2,900-line lookup table (2605.21384). At 10 tasks the minimum detectable effect is about 35 pp (2411.00640). |

### 0.3 The P0 build list

Each item maps to an existing code area in `notes/00-crucible-map.md`. Details are in §13.

| # | P0 item | Where it lands | Why first |
|---|---|---|---|
| P0-1 | `TaskContract` with closed-world diff grading, answer checks, milestones, minefields and timing windows; a fail-closed grader that returns reason codes only; the `PatchTask` adapter for the debug lane | `factory/contract.py` (new), `factory/longtask.py`, `verify/state.py`, `verify/answers.py`, `world/custody.py` | Every other stage needs a non-patch contract |
| P0-2 | The negative library: oracle mutation operators (side effects first), idempotence replay, wrong-task twins, impossible twins, gold-sanity on every LLM-written check, kill report | `accept/mutation/actions.py` (extend), `factory/verifier2.py`, `factory/lanes/windows.py` | Verifier strength is the main hack defence |
| P0-3 | Deterministic pre-rollout gates: G1, G2 with double replay and reset fidelity, red-on-start G4, static G8, structural G9, G11 leak lint and VCS hygiene, G13, G15 gaming battery, G5b | `factory/generators/base.py` (`deterministic_gates`), `factory/benchaudit.py`, `factory/pathcheck.py` | Moves defect detection before the $14 measurement |
| P0-4 | Trust boundary: grading in a separate custody World, affordance removal at `_call_tool`, honeypot twins on every task, fail-closed fault injection (G18) | `world/world.py` `_call_tool`, `world/custody.py`, `world/verifier.py`, `guard/isolate` | Built but unused today |
| P0-5 | Provenance with the real author, family, simulator id and version, time mode, harness, endpoint pin; `DifficultyRecord` keyed by (task, student, checkpoint, user simulator, time mode) | `factory/provenance.py`, `factory/swarm_prompts.py`, `forge/difficulty.py` | "The swarm builds tasks" is unmeasurable without it |
| P0-6 | Frozen anchor sets (200–500 items per judge type), a locked set no loop reads, and the canaries (always-pass, always-fail, constant-tie, master keys, narrated completion) | `factory/goldset.py`, `factory/labels.py`, `factory/framework_health.py` | Every loop and judge depends on it |
| P0-7 | Hack flags and quarantine on training rollouts: L1 rules on the live `WorkspaceHarness`, a different-family trajectory monitor, clean-vs-raw pass rates | `credit/hack_detector.py`, `runtime/harness/workspace.py`, `train/quarantine.py` | L1 has no data on the live path today |
| P0-8 | Behaviour signatures, the archive meters and three dedup layers; wire the herding, ortho and dedup guards into the debug lane | `factory/archive.py` (new), `factory/ortho.py`, `forge/dedup.py`, `factory/root.py` | Kept tasks cluster on four repos |
| P0-9 | Executed failure attribution by snapshot replay, and the `Assignment` consumer of `BuilderIntent` | `factory/traceaudit.py`, `factory/failures.py`, `world/world.py` restore | The failure → task route has no consumer |
| P0-10 | Statistics: Wilson, clustered and paired SEs, MDE before any ablation, PPI for judge-labelled rates | `bench/power.py`, `bench/ppi.py` (new) | Without it no factory claim is defensible |
| P0-11 | Predictor-first, sequential measurement (k = 4 → 8), reusing training rollouts for p̂ | `factory/targeting.py`, `factory/difficulty.py`, `train/curriculum.py` | Measurement is the cost centre |

---

## 1. Scope and evidence base

### 1.1 What was read

The evidence base is 337 papers in 12 groups of reading cards (`cards/group-01.md` … `group-12.md`). Six extracts
(`cards/extract-a.md` … `extract-f.md`) condense the cards by mechanism into nine buckets:

| Bucket | Content |
|---|---|
| B1 | Domain spec and World / environment synthesis |
| B2 | Task proposal and task sources |
| B3 | Verifiers and gates |
| B4 | Intent, validity, communicability and leakage |
| B5 | Diversity and coverage |
| B6 | Difficulty and curriculum per student |
| B7 | Self-improvement loops, guards, statistics, judge calibration |
| B8 | Pipelines, costs, yields, scale; training results |
| B9 | Corrections and contradictions |

| Extract | Groups | Papers | Theme |
|---|---|---|---|
| A | 01–02 | 58 | Environment and task synthesis, tool-use data, SWE environments, verifier testing, loops |
| B | 03–04 | 58 | Proposer conditioning, open-endedness, judge hacking, self-play, frontier-lab reports |
| C | 05–06 | 58 | Difficulty filtering, curricula, test-based verifier weakness, World and benchmark design, privacy |
| D | 07–08 | 58 | Verifier testing, validity, leakage, judges, failure attribution, reward hacking |
| E | 09–10 | 58 | Loop guards, judge statistics, enterprise and web Worlds |
| F | 11–12 | 47 | CUA and MCP benchmarks, GUI RL, tool and user simulators, dense Worlds, traces to Worlds |

### 1.2 Read levels

| Level | Count | Meaning |
|---|---|---|
| FULL | 316 | Method, experiments and appendices read (some long appendices skimmed; some tables lost in HTML conversion and taken from text) |
| PARTIAL | 21 | Body read in full; parts skimmed |
| ABSTRACT-ONLY | 0 | — |

The 21 PARTIAL papers: Kimi K2 (2507.20534; post-training read in full, pre-training summarised); OfficeTown
(2610.23194); ProgAgent (2610.14774); MiniMax-M2 (2605.26494); DeepSeek-V3.2 (2512.02556); OpenThoughts (2506.04178);
Towards a Science of AI Agent Reliability (2602.16666); School of Reward Hacks (2508.17511); HackTrace (2610.03055);
VisualAgentBench (2408.06327); VisualWebArena (2401.13649); OSWorld (2404.07972); UI-TARS (2501.12326); OpenCUA
(2508.09123); Non-Collaborative User Simulators (2509.23124); LLMs Get Lost in Multi-Turn Conversation (2505.06120);
SimulatorArena (2510.05444); MirrorBench (2601.08118); UserBench (2507.22034); UserRL (2509.19736); the agentic
environment engineering survey (2606.12191).

Tables lost in conversion (numbers taken from text) are flagged in the cards for 2501.12599, 2506.08989, 2509.24726,
2510.18821, 2506.03136, 2406.13542, 2505.17063, 2606.04455, 2410.10934, 2404.12272, 2609.12439, 2602.16179, 2506.13131
and 2408.08435.

### 1.3 How the flags work

- **A flag travels with the number.** If a card marks a figure [uncertain], "figure only" or "n/r" (not reported), this
  document keeps the mark wherever the figure appears.
- **PARTIAL** marks any claim that rests on a part of the paper the card did not read in full. MiniMax-M2's Terminal-Gym,
  for example, reports no task counts or yields at all, and the flag stays.
- **[no-control]** marks a gain without a matched or equal-budget baseline (Agent-World's arena loop, EAU's meta-agent,
  AgentMercury's out-of-domain math and code gains, Corecraft's environment claims).
- **[contaminated]** marks a result where benchmark material entered synthesis, selection or training (§1.4). Such a
  result may still show that a mechanism *runs*; it never shows that the mechanism *transfers*.
- **Single run.** Many ablations are single runs or peak-checkpoint numbers (SearchEyes, TRACE main tables,
  Qwen-AgentWorld, FEE, Agent0). We say so where it matters.

### 1.4 The contaminated-evidence list

These papers fed benchmark material into synthesis, selection or training, or evaluated on a benchmark built by their own
pipeline. For Crucible this hardens one rule: **nothing from a benchmark task, a sealed task or a held-out family may
feed F0–F3, the failure clusterer, the skill selector or any proposer's references.**

| Paper | What leaked | Consequence for us |
|---|---|---|
| TRACE (2604.05336) | τ² discovery rollouts on the evaluated tasks | Its +15.3 τ² gain is an upper bound; the selector mechanism is still usable on our own rollouts |
| NoisyAgent (2605.27209) | Trains on the τ² and VitaBench domains it evaluates | Robustness gains in-domain only |
| Magnet (2503.07826) | BFCL-v3 implementations in its function pool | Graph operators usable; scores not |
| Learn-by-interact (2501.10893) | SWE-bench Verified files used as "documents" | Relabelling mechanism usable; scores not |
| SearchEyes (2607.05943) | Evaluates on a benchmark built by its own pipeline | Measures in-distribution skill |
| ZeroGUI (2505.23762) | Test-time RL on the benchmark's own test tasks; exemplars were test tasks | Reported score contaminated |
| ARPO (2505.16282), DART (2509.23866), ComputerRL (2508.14040) | Train on OSWorld tasks or its domain; ARPO's held-out transfer is about zero | Only indicative of executed-reward value |
| AgentScaler (2509.13311) | Reproduced τ-bench schemas "highly consistently" | Catalogue-first synthesis regenerates benchmarks |
| ACE-Router (2601.08276) | Seed tools include MCP-Universe's own | Mutation gains partly in-domain |
| APIGen-MT (2504.03601), Simia (2511.01824), LOGIGEN (2603.00540) | Train on τ-bench / τ² domains; Simia's prompts include `{testbed_text}` | In-domain |
| EnvScaler (2601.05808) | Seeds from API-Bank and ToolACE task sets | Seed-set leakage |
| GRADUS (2610.27781), Trajectory2Gym (2610.33568), AutoForge (2512.22857) | Train inside the evaluation environments or their domains | Large τ² gains are in-domain |
| OSWorld-MCP (2510.24563) | Tools derived from the evaluated tasks' solutions | Tool-to-task leakage |
| GTM (2512.04535) | The tool simulator generates outputs from the task's ground-truth answer | Answer-conditioned tools; never copy |
| Environment Tuning (2510.10197) | Stage and checkpoint choice used 100 samples of the held-in test set | Selection on test |
| DGM (2505.22954) | The diagnostician saw private test patches; selection and report on one benchmark | Leak by design |
| AEvo (2605.13821), ADAS (2408.08435) | Same benchmarks for search and report | Overfit risk |
| Data World (2610.21656) | Rubric judge during RL is the base model itself | Self-judging |
| Corecraft (2602.16179), EntWorld (2601.17722) | Random or in-distribution splits of one World | No held-out family |
| SWE-bench Verified generally (2506.12286, 2505.23419) | Models reproduce file paths and code verbatim | Use only as a memorisation reference |

Three papers carry a contamination *risk* the authors argue away: Qwen-AgentWorld's simulation instruction carries the
success state (2606.24597); DeepSeek-V3.2 fills sandbox DBs with internet data [PARTIAL] (2512.02556); Magpie may
regurgitate alignment-set prompts (2406.08464).

### 1.5 Evidence on our student family

The closest evidence to our setting. Where a result trains in the evaluation domain it is marked.

| Paper | Student, regime | Result |
|---|---|---|
| WildAWM (2610.26828) | Qwen3.6-35B-A3B, SFT on 100 tasks then GRPO on 208 synthesised CUA tasks | 8-benchmark OOD average 42.84 → 53.42; gains in the 2nd and 3rd difficulty quartiles, **none in the hardest quartile** (beyond the teacher). Every gate it ablated paid for itself. Same family (GLM-5.2) built Worlds, tasks, judges and the teacher. |
| CompoWorld (2609.33665) | Qwen3.6-35B-A3B, SFT 3K + GRPO 1K | Average +9.17 on eight benchmarks; RL added only 1.2 over SFT |
| VHD-Play (2609.27321) | Qwen3.6-35B-A3B, GRPO, 34 steps | BFCL V4 61.25 → 64.08; a matched EnvScaler-scenario control scored 58.56, below base |
| Data World (2610.21656) | Qwen3.6-35B-A3B RL-zero | DataWorld-Bench 37.64 → 44.70; self-judged |
| AutoEnvScaling (2610.14163) | Qwen3.5/3.6-35B-A3B | TB2.1 26.7 → 38.9; as self-proposer the 35B's valid-environment rate rose 32.3% → 86.8% after SFT |
| AgentMercury (2608.20634) | Qwen3.5-35B-A3B GRPO | Held-out EnterpriseOps-Gym 24.8 → 28.1: about 3 points, the closest published RL gain on a held-out enterprise benchmark; OOD math and code gains [no-control] |
| Interlock (2610.19686) | Qwen3.6-35B-A3B rejection-sampling SFT | AutomationBench 9.00 → 13.50 |
| TRACE (2604.05336) | Qwen3-30B-A3B | τ² 32.9 → 48.2 [contaminated] |
| GRADUS (2610.27781) | Qwen3-30B-A3B | τ² telecom 0.38 → 0.97 [contaminated] |

**Reading.** At our size, the RL results that held used executable state-based rewards and composed or targeted tasks
(TRACE, WEFT 2609.36887, WildAWM). Gains stop near the teacher's frontier (WildAWM). RL often adds little over SFT
(CompoWorld). Expect a few points, not tens, on held-out enterprise families.

---

## 2. Core objects and typed contracts

Every stage reads one typed artifact and writes one. Schemas live in `core/schema/taskgen.py` (new) and export as
`crucible.taskgen/v1` JSON schema, beside the existing `crucible.world/v1`. Every artifact carries `provenance` (§2.7).
The shapes below are **[Proposal]**; each field cites the paper that motivates it.

### 2.1 DomainSpec: domain content is data

A `DomainSpec` holds everything domain-specific. Stage code is generic. Coding is one `DomainSpec` among others (§12).

```text
DomainSpec
  domain_id, version, owner
  skill_tree: [SkillLeaf{id, definition, negative_definition,          # 2511.04703: say what the skill is NOT
                         confounders: [{name, probe}],                  #   e.g. "copying the fix from history is not debugging"
                         observable_in: [BehaviourSignature field],     # F7 must see it, else relabel
                         aliases, merged_from, support}]                # InsTag normalisation (2308.07074)
  axes: [Axis{name, values|bins, source: skill|scenario|world_facet|knob|regime|persona_stratum|instruction_form,
              target_share?}]                                           # coverage grid; Surveyor tag manifest (2610.08048)
  joint_slices: [[axis, axis]]                                          # template-locking alarm (2608.27260)
  knobs: [Knob{id, kind, class: difficulty|structural|stylistic,        # Reasoning Gym's three classes (2505.24760)
               range, monotone: measured|unknown, ladder_ref}]
      kind ∈ {count, linkage, distractors, horizon, tools, tool_exposure, observability, hints, start_offset,
              obfuscation, noise, fault_profile, pressure, latency, contention, user_behaviour, disclosure,
              time_mode, instruction_form, persona_stratum, policy_load, error_informativeness}
  task_regimes: [write, read_answer, multi_turn_user, deadline, infeasible, clarify, confidential, partial_recovery]
  regime_shares: {infeasible_share, no_action_share, confidential_share}   # WorkBench 18% no-action (2405.00823)
  concerns: [ConcernSpec{text, example_diffs}]                          # concern-targeted mutants (2501.12862)
  forbidden_actions: [PathConstraint]                                   # generalised pathcheck
  leak_rules: [LeakRule]                                                # generalised report lint
  master_keys: [str]                                                    # judge probes (2507.08794)
  ignore_fields: [table.col]                                            # auto ids, timestamps (2610.09142)
  policies: [PolicySpec{text, enforced_in: api|agent|trigger, visible_doc_ref}]   # τ-bench (2406.12045)
  families: [FamilySpec{id, world_template, skill_leaves, held_out: bool, atomic_groups}]
  sources: [mined | synthetic | trace | selfplay | mirror]
  answer_kinds: [lookup, aggregate, recommend, refuse, ask, infeasible]
```

**Rules for the spec.**
- **A tag must be able to be false.** A tag with a target share above about 0.5 is a baseline requirement, not a coverage
  axis (Daedalus, 2610.08048).
- **Write the policy before the tasks.** τ² wrote its policy from the tasks and solutions, so the policy may encode the
  fault list (2506.07982). Make some rules deliberately "ad hoc" so the policy is load-bearing: removing the policy cut
  τ-bench airline 33.2 → 10.8 but retail only 61.2 → 56.8 (2406.12045).
- **Instructions must not forbid what the verifier would reward.** Fixing a hackable reward by telling the model not to
  exploit it can raise misalignment (2511.18397). The authoring guide says so.
- **A family's size is its set of atomic groups, not its composed count.** τ²'s 2,285 combinations came from 15 groups
  over a 4-customer DB (2506.07982).

**Bootstrapping a DomainSpec** (F0, §3.1): an executed Surveyor (2610.08048), a GLAN-style skill tree pruned by the lead
(2402.13064), usage weights from real logs or labour statistics (2412.14161, 2510.04374), InsTag passes on any
LLM-made labels (2308.07074), and a lead review.

### 2.2 WorldSpec and the App contract

`crucible.world/v1` already declares Apps, actors, RBAC, the clock, events, isolation, turns and verifier custody
(`core/schema/world.py`). Part 2 adds a required **App contract** that every App template must meet before any task uses
it.

```text
AppSpec
  template, version, tier: real_oss | mock_rest | code_app | cache_tier | llm_text_only
  state_api: {load(envelope), reset(), export() -> envelope, health()}   # OSWorld 2.0 /api/state (2606.29537)
  tenancy_key                                                            # many task instances per container
  resources: [{name, class: agent_visible|verifier_only|forbidden|handoff,          # BenchShield (2609.11028)
               durability: authoritative|derived|ephemeral}]                       # CUA-Sandbox (2609.32750)
  tools: [ToolSpec]
  policies_as_code: [Trigger | PreWriteHook]                             # LOGIGEN BEFORE/AFTER triggers (2603.00540)
  invariants: [Invariant{trigger_service, target_service, check_sql, visible_doc_ref}]   # AgentMercury (2608.20634)
  column_policy: {table.col: exempt|exact|semantic}                       # ScaleEnv (2602.06820)
  config_flags: {latency_ms, error_rate, hide_labels, pricing_multiplier, ...}           # REAL (2504.11543)
  clock_injection: true                                                  # wall clock banned (2609.05576)
  settle()                                                               # outcome finality (2608.14940)
  events_out: [InternalEvent]                                            # verifier hooks (2506.07672)
  conformance: {golden_fixtures_ref, quirk_log_ref, bug_class_checklist} # ClawsBench (2604.05172)

ToolSpec
  app, name, description, input_schema, read_or_write, role_scope: agent|user|env   # Gaia2/ARE (2602.11964, 2509.17158)
  params: [{name, provenance: external|internal}]                       # G8: EnvFactory (2605.18703)
  duration, occupies_actor, locks: [resource]                           # TimeArena (2402.05733)
  error_envelope: {success, data | error{code, message, violated_rule?, hint?}}      # CompoWorld, LOGIGEN
  limits: {page_size, truncation_signalled: bool}                       # silent truncation is a knob (2602.16179)
  mutated_from?                                                         # distractor tools (2601.08276)
```

**The engineering rules** (EnvCraft Table 15, 2609.05576, plus the cards):

| Rule | Why |
|---|---|
| Dict returns with structured errors; one envelope for every tool | CompoWorld; correct `isError` mattered (38.3 → 34.5 when disabled, 2610.14774) |
| No mutable globals; immutable data dir during rollouts; atomic writes; bounded locks | Determinism and parallel rollouts |
| Timestamps from `base_time` + action index; wall clock banned | G2 holds by construction |
| Sessions bound by trainer-injected environment variables; trainer commands hidden from agent help | Trust boundary |
| State touched only through CRUD primitives, so every change is logged | SynthAPI (2610.37648) |
| Generated or stubbed tools memoised per input | RandomWorld (2506.11045) |
| Errors as observations, never loop ends; outputs over a bound are paged, not dropped | Toolathlon (2510.25726) |
| Infrastructure errors generic; domain errors name the entity, field, state or valid alternative | EnvCraft; error informativeness is a knob (§3.2) |
| No "task completed" pop-ups or redirects | GUI-Genesis (2602.14093) |
| State kept server-side; nothing decodable in client storage | REAL and WebForge leaks (2504.11543, 2604.10988) |
| The verifier never deserialises or executes the deliverable | BenchShield's pickle case (2609.11028) |

**The state API and reset.** Every App exposes one JSON state envelope with `load`, `reset` and `export`, a tenancy key
and a health probe (OSWorld 2.0, 2606.29537). Crucible's `StateAdapter` (`world/state/base.py`: fingerprint, snapshot,
restore, diff, query) is the implementation. Three resource classes decide isolation: authoritative state gets a private
copy-on-write delta over an immutable base, derived state is private and rebuildable, ephemeral state is recreated. A
resource with no fine-grained adapter goes private, never shared (CUA-Sandbox, 2609.32750). Reset, snapshot, clone and
fork drain in-flight work and publish generation g+1 atomically, and the evaluator reads the same generation as the last
action. Direct publication gave 100/100 mixed-generation events; transactional publication gave 0/100.

**Conformance tests per App** (before any task uses the App):
1. **Blind frozen tool tests** (§3.2, H-tool): four tiers, three layers, negative probes, expected digests frozen at write
   time.
2. **Golden-fixture conformance** for any App that imitates a real API: compare at every nesting level (key sets, value
   types, empty-collection shapes, mutation side effects). ClawsBench found 11 recurring mock bug classes, for example
   null vs absent, list reusing the detail serialiser, static values for computed fields (2604.05172).
3. **Error operators**: every tool must reject a wrong type, a missing required parameter, an extra parameter and a
   semantically invalid value with a non-crash error (GTM, 2512.04535).
4. **Soak test**: thousands of resets with the fingerprint checked after each, since long-running env servers leak
   memory (AgentGym-RL, 2509.08755) and a reset that drops the seed silently changes expected values (VeriEnv,
   2603.10505).
5. **Settle test** for any App with async jobs: `settle()` advances the clock until every pending operation is terminal.
   Snapshot and terminal labels disagreed in all 150 trials with a 25–250 ms write delay (2608.14940).

### 2.3 ActorSpec: simulated people

Today `persona`, `grounded_on`, `policy_doc` and `transport` are schema-only and every responder is Python wiring
(map §1.1). Part 2 makes people declarative data a generator can emit.

```text
ActorSpec
  id, role, org_position, permissions                       # OfficeTown roles with authority rules (2610.23194)
  traits, backstory, memories                               # Concordia persona generation (2312.03664)
  goal_hidden, goal_shards: [Shard{fact, kind}]             # goal sharding (2509.23124, 2505.06120)
  knowledge: {public: [fact], gated: [{fact, dimension}], internal: [fact]}   # EnterpriseBench (2609.37658)
  release_schedule: {per_turn_max: 1, release_on: targeted_question | stall_after(n)}   # UserBench (2507.22034)
  delivery_tracker: true                                    # undelivered shards must go out before END
  ending_verifier: true                                     # blocks a premature END
  behaviour_modules: [collaborative | unavailable_service | tangential | impatient | incomplete | mind_changer
                      | nitpicker | vague]                  # 2509.23124; SWE-chat personas (2604.20779)
  style: {attributes, length_range}                         # SimulatorArena profiles (2510.05444)
  user_tools: [ToolSpec role_scope=user]                    # τ² dual control (2506.07982)
  hidden_demos: [dialogue]                                  # ToolSandbox (2408.04682)
  relationships: [{other, kind, visibility}]                # SOTOPIA visibility table (2310.11667)
  backend: {model_id, model_version, prompt_hash, temperature, family}   # keys p-hat (2601.17087)
```

**Rules.**
- **State and reward live in code; the LLM is the voice.** Hidden state lives in a rule automaton, and the model only
  phrases replies from it (UserRL, 2509.19736). Any LLM-proposed state change is a typed effect validated on a copy;
  on rejection only the observation is committed (Trace2Env, 2610.06100).
- **The knowledge boundary is a hard field.** Adding a knowledge boundary and hidden demos cut user-simulator
  hallucination from 12.4% to 6.97% and instruction errors from 6.2% to 0.77% on 1,032 annotated trajectories
  (ToolSandbox, 2408.04682).
- **Behaviour is a module, not prompt text.** A prompt-only non-collaborative simulator barely lowered agent success; the
  modular one did, and humans preferred its realism about 70% of the time (2509.23124). A findings-informed persona prompt
  *lowered* fidelity (USI 70.9 → 64.6) (2603.11245).
- **User tools make the user an actor.** The user owns tools over its own state, so it can act but not plan; removing the
  user (No-User mode) moved pass^1 by 18–25 points (τ², 2506.07982).
- **Never a reward from the simulator's own judgement.** UserRL's Persuade and Intention rewards come from the
  simulator (2509.19736); a satisfied simulated user is not success: 57.5% of human-satisfied conversations failed
  (GAUGE, 2609.12191).

Section 4 gives the validation gates for people (U1–U4).

### 2.4 The clock and event model

Crucible has a clock (step, wall, sim), a scheduler, seeded Poisson triggers and events, used only in toy packs
(`world/clock.py`, `core/triggers.py`). Part 2 adopts the Gaia2 / ARE model (2602.11964, 2509.17158) and adds finality.

| Element | Specification | Source |
|---|---|---|
| Event queue | Every action is a timestamped event in a time-ordered queue with DAG dependencies | ARE |
| Event kinds | Agent, User, Env; **Conditional** (polls a predicate, e.g. "cancel only if booked"); **Validation** (a milestone or constraint that fails the run on timeout); **Oracle** (ground-truth actions for the verifier) | ARE |
| Notification policy | low (user messages only), medium (consequences of agent actions; default), high (all env events); an observability knob | Gaia2 |
| System App | `get_current_time`, `wait`, `wait_for_next_notification`; a wait jumps event to event | Gaia2 |
| Time modes | `real_latency` (generation time consumes sim time), `paused_latency` (clock paused during generation, then advanced by measured latency), `instant`; recorded in every `DifficultyRecord` | Gaia2: GPT-5 (high) 0.0 → 34.4 on Time in instant mode |
| Tool durations | `duration`, `occupies_actor`, `locks`; async outcomes land later than wake time, so status must be checked | TimeArena (2402.05733); Vending-Bench (2502.15840) |
| Settlement | `settle()` before grading; each verifier declares `outcome_time ∈ {endpoint, settled}` | 2608.14940 |
| Open effects | `OpenEffect{handle, status, cross_run_route}` from App wrappers; shared-state runs are grouped before pass^k | 2608.14940 |
| Reset of clock | Device clock reset to a fixed instant per task start | AndroidWorld (2405.14573) |
| Clock offsets in measurement | p̂ is sampled over seeds *and* clock offsets | OpenCUA: only the date changed, Pass@1 20.1 vs Pass@16 38.6 (2508.09123) |

**Time is both a skill and a confound.** Thinking longer missed deadlines (Gaia2 inverse scaling). So a deadline family
declares its time mode, F9 measures in the training time mode, and F10 runs a time-mode probe (§3.11).

### 2.5 TaskContract

Today `LongTask.grade_patch(patch)` takes a patch (`factory/longtask.py:72`). The replacement grades a `WorldRef`.

```text
class TaskContract(Protocol):
    task_id; family_id; domain_id; source; regime
    world: SeedBundleRef                  # WorldSpec + seed bundle + clock start + time mode
    instruction: str
    user: ActorSpec | None
    answer_contract: AnswerContract | None    # hidden decisions named, no values (ClawLoop 2610.14118)
    verifier: VerifierBundle              # custody: verifier World only (world/custody.py)
    oracle: Oracle
    knobs: KnobVector; cell: CellClaim
    async def start(self, sandbox) -> WorldRef
    async def grade(self, end: WorldRef, answer: Answer | None, trace: Trace) -> Grade
    def negatives(self) -> list[NegativeCase]
    def alternatives(self) -> list[Oracle]
    def simplify(self, knob, **kw) -> TaskContract     # ladders; shares the parent's VerifierBundle
```

**The grading rule for write tasks (AppWorld, 2407.18901).**

```text
D   = StateDelta(start, settle(end)) minus DomainSpec.ignore_fields     # typed: added, removed, updated
C_e = expected changes (from the oracle's executed end state)
C_a = allowed changes (declared, audited; G14 reviews additions)
pass_state  = C_e ⊆ D  and  D ⊆ C_e ∪ C_a                               # realisation AND frame
```

The left inclusion is *realisation*: everything that should change did, for every object in the declared population
(Interlock's (basis, population, applicability, outcome) form, 2610.19686). The right inclusion is the *frame*: nothing
else changed. A side effect is a change that differs from both the gold and the start state, so a failure that changed
nothing is not a side effect (WorkBench, 2405.00823). Multisets are compared as multisets: verifiers that matched records
by key into a dict, compared a queue as a set, or relied on a page-local flag let duplicates pass (2610.09142). The
allowed set is hand-written in AppWorld, which risks false fails, so additions to C_a go through a G14 audit.

**Answer checks for read-only goals.**
- Answer = a transform (count, sum, identity, argmax) over `relevant_state`; the oracle is free (AndroidWorld,
  2405.14573).
- `exclusion_conditions` are required: seeded noise must not change the answer (no noise event on the asked date).
- **Uniqueness by query:** after any obfuscation, query the World for every entity that satisfies the stated
  constraints, and accept only if exactly one remains or the full set is accepted (WebSailor 2507.02592; Search Self-play
  2510.18821).
- Negative checks: the answer must not contain wrong candidates, which blocks "emit all possible answers" (ABC,
  2507.02825) and shotgun guessing (UserBench: about +40% from listing options, 2507.22034).
- Answers go through a structured `submit` tool with a declared schema and tolerance; no LLM answer extractor
  (2504.08942).
- For free-text answers only, atomic claims derived from the oracle end state, one judge call per claim, numeric
  tolerance, pass at ≥ 0.75 coverage (MCP-Atlas, 2602.00933), with negatives from perturbing the gold along named axes
  (RaR, 2507.17746).

**Milestone DAGs, minefields and guardrails (ToolSandbox, 2408.04682).**
- A milestone is a target DB state or tool-trace pattern with per-column similarity (exact, ROUGE-L, AST). Row
  similarity is the geometric mean of column similarities, so one violated must-hold column zeroes it.
- The trajectory's milestone score is the best mapping of turns to milestones that respects a topological order of the
  DAG.
- A **minefield** has the same form, but any match sets the reward to 0 (for example, calling `timestamp_diff` when the
  current time is unknown, which means a hallucinated argument).
- A **guardrail** is a no-touch window: a given table must not change between two milestones.
- An **information-flow check** requires a value from an earlier tool output to flow into a later call. It is the formal
  form of the "skip a read" mutant.
- Milestones feed Φ only. Φ must lie strictly inside (0, 1) on partial states, rise monotonically to 1 along the oracle
  and fall on regression (Tev progress validity, 2610.18868). Milestones are order-gated and use per-instance nonces, so
  checkpoint farming does not pay (RHB step verification, 2605.02964).

**Timing windows (ARE verifier, 2509.17158).** Each oracle write with a relative delay over 1 s must be matched by an agent
write inside [Δt − 5 s, Δt + 25 s] of its matched parent, and every oracle parent must already be matched to an earlier
agent action. Reads are free. On 450 human-labelled trajectories this matcher reached agreement 0.98, precision 0.99,
recall 0.95, against 0.72 / 0.53 / 0.83 for an in-context LLM judge. Its blind spots are equivalent alternative writes and
harmless extra writes, which Crucible's state checks and G14 cover.

**The fail-closed grader.**

```text
Grade = state_pass AND answer_pass AND path_ok AND timing_ok AND integrity == Checked
reward_binary = Grade                    # the keep-rule metric
reward_shaped = 1[Grade] + λ·Φ(s_T)       # optional, λ = 0.25 (Tev PPRS); any pass outranks any failure
on {verifier crash, timeout, unparsable output, skipped check, unknown event}: Grade = FAIL, reason_code set
integrity ∈ {Checked, VectorExposed, AgentViolation, Inconclusive}         # BenchShield (2609.11028)
GraderResponse{verdict, reason_code}      # no free text leaves the verifier sandbox (2606.04455)
range assertion: every check returns a bounded value of its declared type   # STOP >1000% "accuracy" (2310.02304)
```

Tev's shaped objective can prefer a less successful policy only if the pass-rate gap is below λ(1 − p₁)/(1 + λ) ≤ 0.2
(2610.18868). At 27B, terminal-only reward could not complete one optimiser step, which is why the
shaped form exists. AgentViolation zeroes the reward and files a hack flag. Inconclusive is a coverage gap, never a pass.
An environment error is labelled `env_error`, gets reward 0 and is excluded from the student's statistics, never scored
as a student failure (AWM 2602.10090; HAL 2510.11977).

### 2.6 VerifierBundle

```text
VerifierBundle
  written_before_oracle_rollout: true                            # Terminal-Universe (2609.04148)
  state_checks, collateral, answer_checks, path_constraints, timing_checks, milestones, minefields, guardrails
  visibility: {visible: [check_id], hidden: [check_id], hidden_compositional: [check_id]}   # SpecBench (2605.21384)
  negatives: [NegativeCase{class, operator, step, artifact, expect: fail, decisive_step}]
       class ∈ {missing, wrong_value, partial, claim_only, extra, extra_field, side_effect, wrong_task,
                skip_read, reorder, late_write, constant_answer, stale_output, feature_isolated,
                tamper, bait, impossible_twin, attack}
  alternatives: [Oracle]                                         # must pass (G14)
  heldout_mutants_ref                                            # never visible to repair
  impossible_twin_ref                                            # no honest run passes it
  honeypots: [HoneypotSpec{path, source: oracle|tests|verifier|gold_state, watch: read|write|exec}]
  isomorph_ref                                                   # id-renamed twin (2604.15149)
  quality: {per_check_kills, distinguish_rate, toxicity_rate, mu, kill_report, precision, recall,
            judge_share_max}                                     # R2E-Gym, CURE, GateTruth
  judge_components?: [JudgeSpec{model, family, kappa, lambda_calibrated, judge_input_type}]
  rules_version, rubric_version
```

**Check quality, defined.** With a correct set (oracle + accepted alternatives) and an incorrect set (negatives + failed
student and attacker artifacts):
- *Distinguish(t)* = 1 if the best incorrect artifact and the best correct one differ on check t (R2E-Gym App. C.3,
  2504.07164). In R2E-Gym fewer than 20% of generated tests distinguished the top correct from the top incorrect patch for
  most problems.
- *Toxic(t)* = 1 if an incorrect artifact passes where a correct one fails. Toxicity must be 0.
- *μ* = p_u(1 − p_01) − (1 − p_u)·p_00, with p_u the share of correct checks, p_01 the chance a wrong artifact passes a
  correct check, p_00 that a wrong artifact passes a wrong check. Ranking by checks passed is reliable as the number of
  checks grows if and only if μ > 0, at rate about 1 − exp(−μ²m/8) (CURE, 2506.03136).
- *Kill report* = valid mutants killed / valid mutants, with stillborn mutants excluded, other-channel kills separated,
  timeouts counted as indeterminate, a byte-identical rerun, sequential certification and a baseline that the
  unmodified reference must pass first (GateTruth, 2608.12635).
- Every check must kill at least one negative; a check that kills none is dead weight.

### 2.7 Provenance

Today the `builder` field cannot separate human, Claude and swarm authorship, because all 17 co-author reports carry a
copied Claude stamp (map §3.2). Every artifact now carries:

```text
Provenance
  author: {kind: swarm|human|mined|script, model_id, family, role, prompt_hash, session_id}
  cost_usd, tokens_in, tokens_out
  generator_family, generator_recipe_version              # value per dollar per generator (F12)
  simulator: {model_id, version, prompt_hash, temperature, family}   # user / NPC
  time_mode, clock_offset, seed
  harness: {name, version, tool_order, retrieval, context_rule, output_token_cap, step_cap}
  endpoint_pin: {provider, model_revision, quantisation}  # infra drift (2510.11977)
  rules_version, rubric_version, verifier_version
  in_context_exemplar_ids                                 # pool drift and hack contagion (2610.12822)
  revisions: [{reason, before_metric, after_metric}]      # e.g. "revised to pass gate" (2608.12635)
```

A lint refuses a copied stamp. Endpoint pinning matters: one provider swapped weights under one model name, another mixed
FP4 and FP8 across calls, and silent rate-limit failures were scored as wrong answers (HAL, 2510.11977). The output-token
cap is part of the harness record because moving it from 4,096 to 16,384 moved one model from fifth to first
(2608.12635).

### 2.8 The difficulty record

```text
DifficultyRecord          # key = (task_id, student_id, checkpoint, user_sim_id@version, time_mode)
  harness_ref, budget: {tool_calls, output_tokens, step_cap}, context_rule, seeds, clock_offsets
  predictor: {score, version, features_hash}                      # never a training reward
  frontier_audit?: {model, family, k, passes}
  student: {k, passes, wilson_lo, wilson_hi, pass_at_k, pass_hat_k, consistency}   # C_out (2602.16666)
  noise_gap?: success_clean − success_noisy                        # NoisyAgent (2605.27209)
  regret: ref_pass − student_pass                                  # oracle gives the upper term (2012.02096)
  irt: {b, a, se}                                                  # 2PL over the pass matrix (2509.11106)
  decided_at_step / horizon                                        # EarlyEval (2609.02783)
  lucky_pass_share?, hacked_pass_share, clean_pass_rate             # AgentLens; HF-WR (2605.20744)
  halted_by_predictor: bool                                         # Wilson bounds can be recomputed
  band_verdict: below | in | above | unmeasured
  cost_usd
```

**Why each key part is there.**

| Key part | Evidence that it moves p̂ |
|---|---|
| student | A stronger evaluator drops samples it finds easy that the learner does not (ToolACE: learner as evaluator 59.22 vs a 7B other 57.61, 2409.00920) |
| checkpoint | Batches seen by a live checkpoint stay near 0.5 while fixed checkpoints see them harden (2504.03380) |
| user simulator | Swapping only the user model moved success about 9 points and reshuffled bins (2601.17087) |
| time mode | Gaia2 Time split 0.0 vs 34.4 for GPT-5 (high) (2602.11964) |
| harness (in the record) | Claude-Code-trained beat OpenClaw-trained under Claude Code (24.2 vs 16.1) and the reverse on WildClaw (2610.04906); scaffolds dominated ClawsBench (2604.05172) |
| seeds and clock offsets | Across-seed variance exceeded within-seed variance in AndroidWorld (27.6 / 26.3 / 33.2%) (2405.14573) |

---

## 3. The pipeline, stage by stage

Each stage below has the same parts: purpose, method, gates, evidence, failure modes, what to build in Crucible, and
priority. The code areas are those of `notes/00-crucible-map.md`. Every stage writes its yield, its cost and a reject
reason per candidate to the funnel (`data/<domain>/funnel.jsonl`), so the cost model of §11 can be fitted.

### 3.1 F0 — DomainSpec and coverage planning

**Purpose.** Turn a brief (a domain keyword, a customer's systems, a skill to train) into a `DomainSpec`: the skill tree,
the coverage axes, the knobs, the policies, the families and the held-out families. F0 decides *what* the factory must
cover before any World exists.

**Method.**

```text
F0(brief):
  1. survey = Surveyor(World or App catalogue, read-only)            # Daedalus (2610.08048)
       for each area: answer five probes from executed calls:
         record contents and cross-area keys; what lies past page 1; behaviour on wrong and stale ids;
         what a write does to related records (read back); what it refuses and with which error
       emit coverage goal (one axis, a usage-weighted distribution, pitfalls) + closed tag manifest
  2. tree = SkillTree(brief, survey)                                  # GLAN (2402.13064)
       LLM lists fields -> sub-fields -> skills from several prompts; the lead prunes the top levels only
  3. weights = usage weights from real logs, labour statistics or trace clusters   # §8, Clio (2412.13678)
  4. normalise any LLM-made labels: frequency floor, lexical canonicalisation,
       embedding clustering, association-rule merge; audit per-item precision and per-label consistency
       with planted wrong labels                                       # InsTag (2308.07074)
  5. axes += task regimes (clarify, refuse, infeasible, partial recovery)   # ACE survey (2608.27260)
     axes += capability splits (search, execution, adaptability, time, ambiguity, noise, collaboration)  # Gaia2
     axes += persona_stratum, instruction_form                         # GAUGE (2609.12191); 2510.08996
  6. families: choose 15-20% of families to hold out entirely          # AgentScaler card; ScaleEnv OOD setup
  7. policies written before tasks; mark which are enforced in the API and which are left to the agent
  8. lead review: axes make sense; at least one held-out family named; no benchmark material used
```

The Surveyor's tag manifest gives a live UNDER / OK / OVER tally that F3 reads as a quota signal. Adding the survey cut
Daedalus's total generation cost 48% and raised session acceptance from 86.7% to 97.8% on AppWorld (cumulative ablation).

**Gates.** F0 has no executed gate. Its gate is the lead review, plus two cheap checks: every `SkillLeaf` names at least
one `observable_in` field, and every axis value is reachable by at least one App tool (checked after F1 by G13).

**Evidence.**

| Mechanism | Paper | Number |
|---|---|---|
| Executed survey before the plan | Daedalus (2610.08048) | −48% generation cost; acceptance 86.7% → 97.8% |
| LLM hierarchy pruned by humans at the top | GLAN (2402.13064) | 126 disciplines kept; gains uneven across disciplines |
| Label normalisation in four passes | InsTag (2308.07074) | 100K+ raw tags → 6,587; per-item precision 96.1%, per-label consistency 86.6% |
| Usage weighting from logs | WebArena (2307.13854); SWE-chat (2604.20779) | SWE-chat: "understand" 18.2%, debug 12.1% of real intents, so a debug-only pool under-covers real use |
| Top-down domain choice from labour data | TheAgentCompany (2412.14161); GDPval (2510.04374) | GDPval: 44 occupations, $3T in wages |
| Persona strata as an axis | GAUGE (2609.12191) | Six strata cut mean reward 42% against one cooperative user |
| Instruction form as an axis | Saving SWE-Bench (2510.08996) | Real-query templates cut success 20–40% relative |

**Failure modes.**
- A structural taxonomy weights by tree shape, not by use (GLAN's uneven gains). Fix: usage weights in step 3.
- Seeding themes from benchmark task sets regenerates the benchmark (AgentScaler reproduced τ-bench schemas). Fix: seed
  from real specs, PRDs, logs and labour data, never from benchmark tasks.
- Over-broad tags ("uses a tool") cannot be false and give no coverage signal (Daedalus).

**Build in Crucible.** `DomainSpec` loader and schema in `core/schema/taskgen.py` (new). Move the debug lane's `SKILLS`,
`failure_skill`, K/C/H knob texts and report lint into a `debug` DomainSpec file, with tests that the old constants equal
the loaded data (`factory/generators/debug.py:592-615,717-755`, `factory/difficulty.py:44-62`). The Surveyor runs as an
RLM session through `factory/sessions.py`. Re-express `finance_ap` and the toy helpdesk as the first two non-coding
DomainSpecs (under the lead's no-detour rule, these are the cheapest tests of genericity).

**Priority.** P0 for the schema and the debug DomainSpec; P1 for the Surveyor and taxonomy tooling.

### 3.2 F1 — App and tool synthesis

**Purpose.** Produce an `AppBundle`: schema, tools, policies as code, invariants and column policies, with tools tested
blind against frozen expectations. F1 is where a World's correctness is decided. A bug here propagates to every task
built on the App.

**Five build orders.** All end in code over a typed state; they differ in what is fixed first (extract A B1.1). The
DomainSpec chooses one per family.

| Order | Steps | Use when | Weak point |
|---|---|---|---|
| Tool-first (ScaleEnv, 2602.06820) | Tool schemas with pre/post-conditions → DB schema reverse-engineered → per-column match policy → DB code + integrity tests → tool code tested by a separate agent → dependency graph | Tool-heavy domains | Its debugger may edit DB instances, so expectations are not frozen |
| Tasks-first (AWM 2602.10090; InfiniteWeb 2601.04126) | Scenarios → tasks → DB only for what tasks need → tools → code | Narrow domains | Off-path functions stay buggy; AWM's audit found 74–83% of envs with ≥ 1 bug |
| Catalogue-first (AgentScaler 2509.13311; EnvScaler 2601.05808; Agent-World 2604.18292; CompoWorld 2609.33665) | Real API or MCP catalogue → tool graph → communities → schema → code | Broad coverage | Regenerates benchmark schemas |
| Spec-DSL-first (AutoEnv, 2511.19304) | Theme → design doc → typed YAML DSL → schema check → code → ≤ 40 repair steps → smoke test | Games and rule systems | Human review of the spec was its largest lever (yield 60% → 80%) |
| Mechanism-first (Data World 2610.21656; Tev 2610.18868; VHD-Play 2609.27321; ClawLoop 2610.14118) | Fix the mechanism (causal DAG, latent world, solver family, seeded generator) → derive state → render observations; the answer is computed | Wherever an answer or optimum is computable | Generator authoring cost not reported |

**Method: contract-first synthesis with blind, frozen tests.**

```text
F1(DomainSpec, family):
  1. design_doc = Architect(DomainSpec, order)                 # reviewed before code (AutoEnv: +20 pp yield)
  2. spec = AppSpec(design_doc)                                # schema-validated; tools carry read/write, role, params provenance
  3. contracts = per-tool {arg -> state mapping, reads, writes, returns}      # WEFT (2609.36887)
  4. tests = BlindTester(spec, contracts)                      # different family, never sees code
       four tiers: simple, medium, complex, boundary           # EnvFactory (2605.18703)
       three layers: L1 load (blocking), L2 2-3 cases per tool, L3 save and diff
       negative probes: wrong types, missing ids, out of range, empty, extreme, special characters   # EnvScaler
       error operators: wrong type, missing required, extra param, semantically invalid           # GTM
       each case: {args, db_instance_ref, expect: success|anticipated_rejection, expected_state_digest}
       expected_state_digest is computed from the contract and FROZEN at write time
  5. code = Coder(spec, contracts)                             # sees spec, never the tests
  6. run tests outside the builder sandbox                     # AutoEnvScaling (2610.14163)
  7. repair loop (<= R rounds), routed by error class:          # WEFT
       structural -> schema; mapping -> contract; runtime/state access -> backend
       the Debugger may edit code only; any edit to a test, a DB instance or an expectation, and any
       "scenario problem" verdict, needs a different-family checker's approval
  8. accept the App only at 100% test pass; log repair_iterations, abandoned
  9. tool graph: edges {dataflow, precondition, shared_table, policy_permitted} with evidence
  10. G13 interaction completeness: dependency-closed BFS from the start state, EXECUTING each added tool
```

**Crucible's addition, stated plainly.** No published pipeline freezes the expected state at write time. EnvScaler's
frontend tester writes no expected outcome, and its backend checker judges code against the observed state diff with a
15% failure tolerance (191 of 266 kept). EnvFactory's reviser may declare a failure a "scenario problem" with no
independent check. Frozen digests are Crucible's rule, not a published practice.

**Run-the-reference repair.** Building with the reference executed in the loop is the largest measured lever on build
quality. In AutoEnvScaling (2610.14163), valid environments with reference execution off vs on were 19.5% → 84.8%
(GPT-5.6-sol), 19.4% → 73.0% (DeepSeek-V4-Flash) and 52.0% → 94.0% (Opus 5). The checks run host-side; the builder may
edit its memory and tools, never the checks. Cross-domain admission was 71.3% of 1,489 runs, lower for browser (26–81%)
and professional workflows (33–62%).

**Policies as code.** Rules compile into SQLite BEFORE triggers that reject with
`POLICY_VIOLATION{code, message, violated_rule, hint}` and AFTER triggers that apply mandated side effects; tools are
insert / query / update only (LOGIGEN, 2603.00540). Policies as Python unit tests over the executed trace catch
cross-action conflicts judges miss (APIGen-MT, 2504.03601). VitaBench expresses rules as tool pre/post-conditions so
domains compose (2509.26490). Invariants are *not* enforced by the transition function; they are checked after the
episode, so the agent must do the cross-service follow-up itself (AgentMercury, 2608.20634). The commonest authoring
failure there was collapsing a cross-service invariant into one service (GPT-5.4 on 10 of 30 briefs), so a gate checks
that a declared cross-service invariant really spans two services.

**Distractor tools.** Mutate real tools five ways (usage extension, function enhancement, workflow chain, helper,
parameter redesign): ACE-Router grew 627 to 2,005 tools, and without mutation MCP-Mark fell 60 → 44.00 (2601.08276)
[contaminated: its seeds include MCP-Universe's]. Distractors are real code that passes F1 tests and carries
`mutated_from`; G14 and F8 read it. Mutants were never checked for distinctness there; Crucible's F8 does.

**Error informativeness as a knob.** Simia's RL in an explanatory-error simulator beat RL in real OfficeBench (64.7 / 34.5
vs 60.8 / 28.6), attributed by case study to explanatory errors (2511.01824) [contaminated; one benchmark, no seeds].
Environment Tuning rewrites opaque errors into root-cause hints that a judge approves only if they avoid exact calls,
concrete values, full sequences and single-path prescriptions (2510.10197). In Crucible the knob is
`error_informativeness ∈ {terse, explanatory, hinting}`, a versioned World-side overlay, replayed with the knob on and off
to prove identical end states and grades.

**Gates.** H-tool (100% blind tests), H-conform (golden fixtures for mock real APIs), H-error (four error operators),
G13 (execute-reachable completeness), the cross-service invariant check, and the App soak test (§2.2).

**Evidence.**

| Mechanism | Paper | Number |
|---|---|---|
| Execution verification of tools | ScaleEnv (2602.06820) | 8B τ² avg@4 42.3 / 30.0 / 25.2 without vs 50.9 / 37.5 / 27.2 with; Airline without it is below base |
| Execution feedback in builds | SWE-Factory (2506.10954) | Build rate 50.2% → 0.3% without it |
| Deterministic recorder emits the artifact | Repo2Run (2502.13681) | No recorder (LLM writes Dockerfile from history): 19.5% / 13.8% vs 100% / 86.0% |
| Role split with failure routing and memory | SWE-Factory | No explorer 42.9%, no memory 45.0%, vs 50.2% |
| Per-step specs vs one growing session | InfiniteWeb vs Claude-Code sites (extract A B1.1) | Same 600-task GRPO: +0.7 / +0.7 / −1.3 on Claude-Code sites against clear gains on InfiniteWeb sites |
| Coding agent with vision for UI Apps | VeriEnv (2603.10505) | 149 sites, 26% failed; $3.6 per site; text-only coders could not fix front-end bugs |
| Trace-driven App per task | GUI-Genesis (2602.14093) | 1,118 Apps for $302.12; 82.65% pass within 5 tries; 17.35% unverified kept anyway (a defect) |
| Mock conformance | ClawsBench (2604.05172) | 169 golden fixtures; 11 recurring mock bug classes |

**Failure modes.**
- **Debugger edits the expectation** to make code pass (ScaleEnv, EnvFactory). Fixed by frozen digests and the
  different-family approval rule.
- **The builder that writes a test approves it.** The role that edits a test script never approves it (SWE-Factory roles).
- **One session accumulates the codebase** and quality falls (InfiniteWeb comparison). One session per stage, sharing
  only on-disk artifacts with a ledger (Data World).
- **Tests derived from the same workflow** (ComputerRL, 2508.14040). Tests come from a different family, from the
  spec only.
- **Live remote services** drift and cannot reset (MCPMark: Postgres 76.19 vs Notion 41.96 for gpt-5-medium,
  2509.24002). Prefer self-hosted or template-copy states; cache-first for read-only third-party services (§4.4).

**Build in Crucible.** `forge/world_synth/` (new) driving F1 as machine states (`factory/machine.py`) and RLM sessions
(`factory/sessions.py`). Apps minted with the agentenv server SDK (`@tool`, `@reset_data`, `@add_data`, `@get_data`) give
MCP tools plus snapshot and restore for free (map §1.7); SQLite Apps reuse `world/apps/sqlite_app.py` and the
`finance_ap` tool-module pattern (`world/packaging/mcp_tools.py`). The vendored AWM order (`_vendor/openenv_awm`) is the
tasks-first recipe. Tool tests live in `AppBundle.tool_tests` and run in the verifier sandbox.

**Priority.** P0 for frozen blind tests, run-the-reference repair and the error-class routing; P1 for the five build
orders as recipes, distractor tools, conformance and policies as code; P2 for the LLM-simulated tier.

### 3.3 F2 — Seeding, distractors, determinism and exact reset

**Purpose.** Fill the App with a start state that is consistent, dense enough to be realistic, seeded with deliberate
distractors and traps, deterministic under its seed, and exactly resettable.

**Method.**

```text
F2(AppBundle, cell slice, seed):
  1. mechanism first, observation second:                              # Data World (2610.21656); Tev (2610.18868)
       latent traits (hidden) -> canonical truth, layer by layer, one session per layer, fixed seed
       -> observation projection by a separate session with NAMED defects
          (id spellings, delays, duplicates, unit drift, schema drift, unlabelled foreign keys)
  2. latents CAUSE records, generated in FK-dependency order           # CRMArena (2411.02305)
       e.g. ShoppingHabit per customer, Skill per support agent; without the Skill latent no transfers exist,
       so the transfer task family cannot exist
  3. seed through the App's own tested APIs, backdated in time           # AppWorld (2407.18901)
       every write passes the App's validation and multi-table consistency
  4. people in dependency order: persona -> relationships -> contacts -> messages, emails, calendars   # Gaia2, ARE
  5. progressive grounding in three rounds: targets + minimal executable state;
       cross-tool relations and constraints; history and distractors      # WEFT (2609.36887)
  6. distractors:
       minimal core + D-1 staged distractor batches (told what came before, non-colliding ids,
         related to the main user, must not pre-solve or block)            # SynthAPI (2610.37648)
       near-misses: each violates exactly one constraint                   # VitaBench (2509.26490)
       boundary-adjacent: thresholds at N-1; states one action from a violation   # LOGIGEN (2603.00540)
       needles + fillers; traps (duplicate external ids across tenants, a policy-violating tool default,
         an expired default card, pagination that does not signal truncation)   # 2604.05172, 2603.13594, 2602.16179
  7. randomise per seed every ordering a heuristic could exploit:          # protocol validity (2607.22368)
       entity ids, row orders, timestamps, file orders; uncorrelated with the answer
  8. several start states per instruction (S > 1)                           # SynthAPI
  9. for dense Worlds: deterministic day deltas                             # WildAWM (2610.26828)
       delta(d) replayed hermetically over state(d-1) reproduces state(d): frozen timestamps,
       derived ids (MAX(id)+1), idempotent inserts, fixed salts; produces.json declares every write
  10. fingerprint = hash(export())
```

**Gates.**
- **G1 Integrity.** Schema, FK and constraint checks, plus the coherence checks Gaia2 says it does not handle:
  timestamps vs calendars, relationships across channels, cross-modal references (2602.11964).
- **G2 Determinism.** Seed twice → same fingerprint; replay the delta chain twice → identical content hash (WildAWM);
  reset → fingerprint equal to the seeded state (VeriEnv's top judge error was a reset that dropped the fill seed,
  2603.10505); render libraries pinned and content-addressed (a font change silently altered cues in 2608.03744).
  Expectation hashes are checked at reset, mid-run and validation, and configs are deep-copied between driver and
  verifier (a shared-config bug overwrote the verifier's expected value in 2610.09142).
- **G20 Sufficiency** (for rebuilt start states, §8): a read-only judge returns {label, present, missing_critical}
  (Terminal-Universe, 2609.04148).
- **H-fidelity** (only when seeding from customer data, §8.3): rare-value merge, DCR-CDF integral ≤ 0, shadow-model
  membership inference near AUC 0.5, canary audit.
- **G23 Generator regularity** (proposed by extract D): a trivial positional or lexical heuristic must not beat chance on
  the family (the faulty invoice must not always be the last row).

**Evidence.**

| Mechanism | Paper | Number |
|---|---|---|
| Apps, seeds and distractors set the ceiling; tasks per App set the rate | SynthAPI (2610.37648) | At fixed 32,768 tasks, (A = 64, T = 128, S = 4, D = 4) beat (A = 32, T = 1,024, S = 1, D = 2); AppWorld Test-C 4B 38.4 → 45.0 (SFT, not RL) |
| Best seeds per proposal | SynthAPI | S = 1 at 32–512 tasks, 2 at 512–4,096, 4 beyond; needs state-generic verifiers |
| Staged distractors | SynthAPI | D = 4 beat D = 1, 2 at high budgets |
| Latents make families possible | CRMArena (2411.02305) | 90% of 10 experts rated it realistic; CRMArena-Pro's denser World fell to 66.7% / 62.3% |
| Seed through the API | AppWorld (2407.18901) | 106 people, about 370K rows; reset under 0.5 s; diff under 0.6 s |
| Dense Worlds by day deltas | WildAWM (2610.26828) | 640 world states, 171.6K DB records, median 52.4K tokens of state vs 1.1K for injected states |
| Generator regularity shortcut | 2607.22368 | Restricting parents to lower-indexed variables raised a score 0.018 → 0.639 (G = 0.621) |
| Real resource pools | WorkForge (2610.04906) | Real workspaces had 5.4× more files and 279 vs 21.5 KB |

**Failure modes.**
- **Ground truth leans on latents the agent cannot see** (CRMArena reports no check). Fixed by the G8 variant: every
  latent-derived answer must also be reachable through policy-visible tools (§3.7).
- **Density is not realism.** Realism fell as CRMArena-Pro added objects. Track a realism meter (§6.1).
- **Empty "not worked in" records** were experts' main complaint. Seed history, not only entities.
- **Non-determinism from libraries and clocks** (VeriEnv, the clinical study). G2 extensions above.
- **Credential files left in the workspace** (WildAWM). Path constraints and G11 scan the start tree.

**Build in Crucible.** `SeedBundle` construction in `forge/world_synth/seed.py` (new) over `StateAdapter`
(`world/state/base.py`); per-row hashes and table change counters in the SQL adapter (`world/state/sql.py`); the clock
injected through `world/clock.py`. The `finance_ap` generator's seeded Python with near-miss invoices
(`packs/finance_ap/.../world.py`) is the existing precedent. The World snapshot and restore checks
(`world/world.py:1069-1159`) give G2's reset half.

**Priority.** P0 for G1, G2 (double replay, reset fidelity) and per-seed randomisation; P1 for latents, staged
distractors, traps and day deltas.

### 3.4 F3 — Task sources and grounded proposal

**Purpose.** Produce `TaskProposal`s that cover the grid, grounded in the World as it really is, aimed at a named cell and
student, and never copied from a benchmark.

#### 3.4.1 The task sources

| Source | Ground truth | Best evidence | Weak point | Priority |
|---|---|---|---|---|
| **Mined real fixes** (the debug lane) | Real fixes and their own fail-to-pass tests | Highest realism; F2P > 0 and P2F = 0 as the acceptance rule (DeepSeek-V3.2 [PARTIAL], 2512.02556) | About 1% of fix units yield a candidate; real links in about 3 of 45 screened repos (map §4) | live |
| **Solution-first World tasks** (`world_synth`) | An oracle program executed on a seeded start state | Walk-execute-freeze (AgentScaler, ScaleEnv, Agent-World); verifier-first-then-state-then-oracle (SynthAPI); (init, solution, assert) triples (τ², 2506.07982) | Construction models know public domains | P0 |
| **Mechanism families** | A solver computes the optimum u* and a default value u₀ before the environment exists | VHD-Play (2609.27321): a frozen Qwen3.6-35B-A3B setter admitted 70% once the mechanism was fixed | Needs a computable optimum | P1 |
| **Template + outcome function** | An outcome function computes the expected DB change from slots and data | WorkBench (2405.00823); AndroidWorld `TaskEval` (2405.14573); WorkArena setup / validate / cheat (2403.07718) | Instances differ only in values; dedup on templates | P1 |
| **Act-then-label** | The trace is produced first; the instruction is written from it; an executed predicate on the diff accepts it | NNetNav prefix pruning (2410.02907), BAGEL round trips (2403.08140), Go-Browse frontier (2506.03533), Explorer (2502.11357) | Every published version used an LLM judge; Crucible uses the diff | P0 (exploration mode) |
| **Composition of verified atoms** | Join of sub-oracles that each executed; the joined oracle re-runs | WEFT task graphs (2609.36887); AgentSynth chains (2506.14205); WorkArena++ (2407.05291) | Parts that do not interact add length, not skill | P1 |
| **Failure-targeted variants** | The normal F4–F10 gates on a variant aimed at a measured weakness | SwS (2506.08989); TRACE selector (2604.05336) [contaminated]; WebRL (2411.02337) | Many "new" WebRL tasks were clarifications, i.e. spec defects | P0 (consumer) |
| **Mirror** | Abstract a real incident into a pattern and plant it in an already-gated World; the oracle is the reverse of the planted change | SWE-Mirror (2509.08724): 46% overall yield; 12% inconsistent with the source on human audit | The statement saw `fix.patch` (leak) | P1 |
| **Traces to tasks** | A rebuilt World snapshot and the logged end state | Terminal-Universe (2609.04148); Trace2Env worldbook (2610.06100) | Privacy; coverage bounded by traces | §8 |
| **Self-play** | The proposer's planted ground truth, checked by execution | SSR (2512.18552); SPICE (2510.24684); Search Self-play (2510.18821) | Label drift (R-Zero 79% → 63%, 2508.05004); dominant strategies | P2 |

**Solution-first patterns** (extract A B2.1), each an F4 oracle route:

| Pattern | How the oracle arises | Paper |
|---|---|---|
| Walk, execute, freeze | Directed walk over the tool graph; arguments generated and executed step by step on a seeded DB; end state frozen | AgentScaler, ScaleEnv, Agent-World |
| Blueprint | Instruction + ground-truth write actions + expected outputs; reads left free; executed, recorded as a diff | APIGen-MT (2504.03601) |
| (init, solution, assert) | `init` sets the fault; `solution` uses agent or user tools; `assert` holds at the end; groups hold mutually exclusive subtasks | τ² (2506.07982) |
| Verifier first | `verify(db, answer)` from instruction and API docs; `setup_database()` fits it; `solve(db)` restricted to public APIs; keep iff v(Db₀) = 0 and v(Exec(π*, Db₀)) = 1 | SynthAPI (2610.37648) |
| DAG with reasoning nodes | Merge walks, add computation nodes, fill arguments in topological order, execute, rewrite the intent from (S, S*) | AutoForge (2512.22857) |
| Typed call skeleton | Sample types and a length; add tools whose inputs are subtypes of available variables; trim dead steps; writer sees variables, not values | RandomWorld (2506.11045) |
| Program as task | A solver script with loops, branches and aggregation; its output is the answer | Agent-World; EnvLunch (2610.33817) |
| Seeded generator | build(seed) → (s₀, answer computed by code, answer contract); the LLM never writes task data | ClawLoop (2610.14118); Tev; Reasoning Gym (2505.24760) |
| Client–Consultant episode | A Client agent with a goal and a Consultant agent with tools play an episode; the end DB is the target; a user-view projection writes the instruction | LOGIGEN (2603.00540); explorer success 99.3%, but every Consultant choice must be communicated (G8) |

#### 3.4.2 The proposal step

```text
F3(cell, student):
  context:
    World slice sampled for the cell (entities, tables, people, events)         # grounding
    one sampled real passage or document for surface                          # VHD-Play: label entropy 0.997
    persona (for realism, not diversity)                                       # Prismatic: personas ranked backwards
    K = 5 nearest KEPT tasks in the cell + 5 nearest FAILED or rejected ones,
      each with its measured signature, p-hat and F6/F9 failure reason       # OMNI-EPIC (2405.15568)
    the batch's own earlier proposals (in-context regeneration)                # NoveltyBench (2504.05228)
    the coverage ledger of covered (policy, outcome) pairs                      # GRADUS (2610.27781)
    the live tag tally (UNDER / OK / OVER)                                      # Daedalus
    an explicit band statement for the named student, and its failure-cluster summary for variants
    a wrong-pattern list (known invalid task shapes)                            # ADAS (2408.08435)
    Assignment{parent_ref, student_id, p_hat, failure_cluster, trace_refs} for targeted variants
  exploration first:
    the proposer calls the World's own tools (<= 15 steps) and may cite only ids it observed   # SCA (2506.01716)
  generation:
    Verbalized Sampling, multi-turn: k = 5 candidates with stated probabilities;
    take a mix of high- and low-probability candidates; tail thresholds when a cell stalls   # 2510.01171
    the measured F7 signature, not the stated probability, decides novelty
  doctrine (Daedalus):
    delete every id, amount or date the solver could find;
    get hardness from scope (superlatives, universals, relative windows, pagination, none or many matches,
    policy limits), not from more steps or more apps
  outputs: TaskProposal{cell_claim, knob_vector (declared difficulty vector delta), instruction,
           user_intent, user_known_info, answer_kind, privileged_notes, formal_graph,
           verbalized_p, proposal_batch_id, in_context_exemplar_ids}
```

**Hard rules.**
- **Only references that passed G5 and G12 are shown.** Showing a target k leaky-passing source submissions made a
  frontier model adopt the hack at 51.5 / 73.6 / 84.9% for k = 3 / 5 / 10 (2610.12822). A reference later flagged as
  hackable is withdrawn and its descendants re-gated. The same rule covers swarm memory (`swarm_journal`).
- **No benchmark task, sealed task or held-out family** in any proposer context (§1.4).
- **No tool or server names** in the question; target tools only in `privileged_notes` (TOUCAN, 2510.01179;
  MCP-Atlas, 2602.00933).
- **Draft and refine by different families.** WebForge drafted at T = 2.0 with one family and refined at T = 1.0 with
  another; without plan refinement its validated yield fell 74.1% → 59.5% (2604.10988).
- **Several candidates per prompt, grounded on a sampled start state.** ZeroGUI's single-candidate ablation fell 27.2 →
  24.1 (2505.23762) [contaminated: test-task exemplars].

**Composition operators** (§6.2 has the coverage view).

| Operator | Detail | Evidence and failure mode |
|---|---|---|
| Group composition | Concatenate init / solution / assert of ≤ 1 subtask per group | τ²: subtask count is a monotone knob; combinatorial volume is not behavioural diversity |
| Sequential and parallel-then-sequential | Add a task consuming the first output; or a parallel task then one needing both | BUTTON (2410.12952): bottom-up beat a monolithic "write a complex task" prompt (−15.0% relative success for the latter) |
| Chain and summarise | k of n verified subtasks merged into one instruction; k = 1…6 gives six levels | AgentSynth (2506.14205): o4-mini 18% → 4%; asking for long tasks directly gave easy survivors |
| Depth and width | Replace an index with a minimal unique superset plus a relation; merge independent subtasks | TaskCraft (2506.10055): width adds length, not interaction |
| Service walks | K services + K distractor services; walk length L ∈ [7, 12], L > K forces a revisit | CompoWorld (2609.33665): composed beat single-service at equal data |
| Add one App | Extend a verified task by one relevant App | OfficeBench (2407.19056): 64.5 → 60.0 → 21.4 for 1 / 2 / 3 Apps |
| f(g(x)) | Composition curriculum | AZR (2505.03335): **collapsed**, f learned to return g(x) |

A composition is accepted only with dataflow between its parts, and F7 rejects a composite whose signature is the union
of its parents with no new edges (the AZR warning).

**Infeasible, clarify and restraint tasks are first-class from day one.**

| Source | Device | Evidence |
|---|---|---|
| Shuffled tools that cannot solve the task | An oracle search over the shown tools must confirm unsolvability | TOUCAN (2510.01179): core data alone dropped irrelevance 68.46 → 46.88 |
| Invalid item fills of a prototype | Free generator | VisualAgentBench (2408.06327) [PARTIAL] |
| False presuppositions | Expected answer "None" | CRMArena (2411.02305): 30% of five tasks |
| Missing tool, policy violation, unavailable resource | With ~10 side-effect checks each | EnterpriseOps-Gym (2603.13594): refusal without side effects ≤ 53.9% |
| Withheld details served by an ask tool | `ask_user` events | HiL-Bench in AutoEnvScaling (2610.14163): 35B success 6.8% → 33.9% |
| Repository rules (refuse, disclose, verify, hand off) | Policy cards | RepoComplianceBench (2607.26819): native Refuse 0% and Handoff 0% for all agents |

Each infeasible task needs positive evidence (a refusal with the right reason *and* the policy lookup), never "state
unchanged" alone: an unchanged-state rule let a do-nothing agent score 38% on τ-bench airline (ABC, 2507.02825). G4
adds "always infeasible" and "always FAIL" null agents, and F9 reports the false-infeasible rate on feasible tasks: a
"you may say unachievable" hint made GPT-4 call 54.9% of feasible WebArena tasks impossible (2307.13854).

**Gates at F3.** Format and schema (an AndroidWorld-style defect checklist: missing or hallucinated fields, unused
parameters, non-specific prompts, 2405.14573); the claimed cell is reachable; the leak lint on the instruction (G11); the
text dedup layer against the K shown (F8 layer 1).

**Build in Crucible.** `forge/world_synth/propose.py` (new) as an RLM session; the `Assignment` object and its consumer
in `factory/failures.py:198-279` (today `BuilderIntent`s wait for a human and nothing consumes them); the coverage
ledger and tag tally in `factory/archive.py` (new). Prompt templates are recipe-versioned (`factory/versioning.py`).

**Priority.** P0 for the grounding context, gate-passed references, the `Assignment` consumer and infeasible types; P1
for composition operators, mirrors and templates; P2 for self-play.

### 3.5 F4 — The oracle by construction

**Purpose.** For every proposal, build and execute a solution that reaches the goal state from the start state through
policy-visible tools only. The executed oracle is the solvability proof; frontier sampling is an audit.

**Why construction, not sampling.** Sampling-as-proof admits lucky and trial-and-error tasks and still misses broken ones:
- AWM: no oracle; 23.7–28.2% of training tasks never solved once by the students (2602.10090).
- Endless Terminals kept o3 pass@16 > 0; about half the kept tasks were solved by all 16 attempts (2601.16443), and its
  tasks transferred little: 0.0 → 2.2 (Llama-3.2-3B), 2.2 → 3.4 (Qwen2.5-7B), 1.1 → 6.7 (Qwen3-8B).
- DeepSeek-V3.2 kept pass@100 > 0 under the policy (2512.02556) [PARTIAL].

**Method.**

```text
F4(proposal):
  1. oracle = OracleProgrammer(proposal, tools)            # action program over policy-visible tools
       may not call private helpers, raw CRUD, setattr, or the DB directly        # SynthAPI; G7
       may use only ids it read from earlier tool outputs or the instruction       # G8: v_t grounding (2610.33568)
       user-side steps only through user tools the simulated user owns            # τ²
  2. execute twice from a fresh start; same end-state fingerprint                    # G2
  3. trim: delete every call whose outputs do not feed the goal; resample until all calls are used   # RandomWorld
     minimality: delete each hunk or step while the goal still holds; a survivor is noise   # R2E-Gym (2504.07164)
  4. gold_end_state_digest, gold_answer, tools_used, privileged ids touched
  5. alternative oracle by a DIFFERENT family, written from the spec only           # G17 (Tev)
       must reach the same end-state class on every seed; disagreement = unresolved spec -> reject
  6. privileged-solver check: a solver given only the needed tools, in order, with signatures and examples,
     must reach the goal from the instruction; failure = under-specified, not infeasible   # RandomWorld
  7. horizon and budget: the oracle's call count and virtual time set the budget floor for F9
```

**Oracle kinds.** `action_program` (default), `end_state` computed from slots by an outcome function (template families,
WorkBench), `patch` (debug lane), and `solver` for mechanism families (u* from DP, Held-Karp, HiGHS or SLSQP before the
environment exists, VHD-Play). Replay agreement is a G3 variant for solver-backed families: replaying the solver's policy
must reproduce u* within tolerance; VHD-Play audited 80 held-out environments × (optimal, default, idle drivers) × 2 runs
with all 240 repeated outcomes agreeing and no realised utility above u* (2609.27321).

**Gates.** G2 (twice), G3 (oracle scores 1, Φ = 1, no path violation), G7 (tool-only), G8 static (§3.7), G17 (solver
agreement).

**Evidence.**

| Mechanism | Paper | Number |
|---|---|---|
| Reference replayed in the environment, drop on failure | AgentEvolver (2511.10395) | Synthetic tasks: AppWorld 7B 1.8 → 23.2; hybrid with human tasks 43.6, better than either |
| Execution filter vs LLM filter | BAGEL (2403.08140) | Removing demos that fail their instruction on execution: +7 points |
| Validation solution as a regression test | AppWorld (2407.18901) | Kept per scenario; cannot catch a wrong expectation that Setup shares |
| Independent solver audit | ProgAgent (2610.14774) [PARTIAL] | An independent agent writes its own solver at the hardest scale |
| Two independent oracles agree | Tev (2610.18868) | Disagreement = unresolved spec; adopted as G17 |
| Oracle reads privileged ids | SynthAPI (2610.37648) | A G8 gap even with its restrictions |

**Failure modes.**
- **Setup and oracle share a wrong assumption,** so both pass and the check is wrong (AppWorld; SETA's JSON literals,
  2607.10891). G17's independent oracle and G8's check-literal scan catch it.
- **The oracle reads ids the agent cannot discover** (SynthAPI). The v_t rule rejects it.
- **The Consultant's arbitrary choices enter the target** (LOGIGEN). Every choice must be communicated or the column
  becomes `semantic`.
- **A hardcoded oracle** that echoes the answer instead of computing it (Terminal-Bench's LLM check, 2601.11868). G7
  requires the oracle to compute through tools.

**Build in Crucible.** `Oracle` in `core/schema/taskgen.py`; execution through the World's `_call_tool` choke point so
the oracle's trace is a normal trace (`world/world.py:505-610`); `floors.py` and `accept/mutation/actions.py` already
support tool-call sequences. The debug lane keeps `Oracle(kind=patch)` from mining.

**Priority.** P0.

### 3.6 F5 — Verifier construction and the negative library

**Purpose.** Build a `VerifierBundle` from privileged ground truth and prove it can tell right from wrong *before* any
student sees the task. This is where most hackability is decided.

**Method.**

```text
F5(oracle, proposal):
  1. VerifierWriter (different family from proposer and oracle programmer; sees the oracle and gold end state,
     not the proposer's notes) writes checks BEFORE any non-oracle rollout:
       state checks over StateDelta under column policies (business-key lookup, never hard-coded ids)
       collateral: closed-world frame check (whole delta vs gold delta + allowed set)
       realisation: for every object in the population where applicability holds, the outcome holds
       answer checks with exclusion conditions and negative checks
       path constraints, minefields, guardrails, timing windows, information-flow checks
       milestones for Phi (order-gated, nonce-bound)
       hidden checks, including hidden COMPOSITIONAL checks that make subtasks interact   # SpecBench
       rejection checks: on an invalid or infeasible sub-request the correct behaviour is refusal
  2. several candidate implementations per check goal; keep the one with the best
     coverage of negatives vs false-failure rate on oracle + alternatives          # EvalGen (2404.12272)
  3. gold sanity on EVERY check: it passes on the oracle end state, on every alternative, and is
     red on the start state (target checks) or green on it (preservation checks)  # G4 strengthened
  4. NegativeBuilder (different family from the VerifierWriter) builds the negative library (below)
  5. score the bundle: kills per check, distinguish, toxicity (must be 0), mu > 0, kill report
  6. two-sided repair (<= 3 rounds):                                                 # SWE-ABS (2603.00520)
       equivalent mutant that FAILS  -> the check is over-strict -> relax it (it must still kill the rest)
       non-equivalent mutant that PASSES -> blind spot -> write a killing check
       every new check re-runs gold sanity and must be entailed by the instruction (G8)
```

**The negative library.** Every negative is an executed artifact with `expect: fail`. Most come from mutation operators
applied to the oracle program, which needs no task knowledge and is nearly free.

| Operator | Mutation | Source |
|---|---|---|
| Claim-only | No write; the agent claims success | 2610.09142; Kimi K2's hack check (2507.20534) |
| Omit | Drop the last write | 2610.09142 |
| Drop a step | Omit one action; G6 drops *each* step, not only prefixes | τ² wording (2506.07982); §3.3 of the plan |
| Partial | Keep only the first write of a repeated sequence | 2610.09142 |
| Partial propagation | The decision is right but some downstream records do not reflect it | Interlock (2610.19686) |
| Wrong value | Move one written value to another legal value | 2610.09142 |
| Wrong entity / decoy | Act on the near-miss row the seeder planted | VitaBench, EnvCraft near-misses |
| **Extra** | Repeat the last write | 2610.09142 |
| **Extra-NI** | Extra restricted to writes whose repeat changes per-table counters, found by an **idempotence replay** | 2610.09142 |
| **Extra-field** | Fill a field the task never names | 2610.09142 |
| **Touch a distractor** | Also edit a row orthogonal to the task (noise rows unchanged is the default collateral) | AndroidWorld (2405.14573) |
| Skip a read | Act without the read that justifies the action | Plan §3.3; ToolSandbox information flow |
| Reorder | Swap two causally ordered actions | Plan §3.3 |
| Late write | Shift a write outside its timing window | Plan §3.3 |
| Wrong task (G5b) | Grade task j's oracle run with task i's verifier, at four distances down to a one-parameter twin | 2610.09142: 0 of 2,689 valid off-diagonal runs accepted |
| Constant answer | An answer that does not change across perturbed Worlds | EnvLunch (2610.33817) |
| Stale output | A plausible wrong state from before an update | Tev counterfactuals |
| Feature-isolated | Passes each feature check, breaks composition | SpecBench (2605.21384) |
| Wrong answer (k plausible) | Verbalized Sampling for wrong end states or answers | VS-CoT: 0.89 incorrect rate, 0.57 coverage of distinct wrong answers (2510.01171) |
| Concern-targeted | A typical fault of a declared concern ("duplicate write", "PII to the wrong user") that survives the current verifier | ACH (2501.12862) |
| Deceptive outcome | The World returns a distinct, plausible failing outcome code per mistake class | WebForge (2604.10988) |
| Tamper | Edit tests, grader files or audit tables; write to the DB directly | RHB (2605.02964); ImpossibleBench (2510.20270) |
| Bait use | Use the planted shortcut (§5.2) | HVE (2605.20744); BaitBench (2608.30724) |
| Impossible twin | An unsatisfiable predicate set or one conflicting hidden test; no honest run passes | ImpossibleBench (2510.20270); tracer (2604.23488) |
| Revert shortcut (debug) | "Revert all changes since A′" must not count as a fix | BugPilot (2510.19898) |
| Master keys (judged parts) | Empty, ":", "Thought process:", an opener with no work, a success claim with no tool calls | 2507.08794 |

**The side-effect class is mandatory.** Missing or wrong work was always rejected in the side-effect study (AppWorld
168/168; WorkArena 84/93 with 9 harmless passes), and wrong-task runs never passed. Unrequested changes did: AppWorld Extra
passed 39 of 48 (33 harmless, 6 false positives), and the WorkArena form verifier passed 36 of 36 valid extra-field pairs.
All three common RL reward forms gave the duplicate run the correct run's maximum reward (2610.09142). The idempotence
replay runs first, so a harmless repeat is classified, not counted as a survivor. One count assertion per template
rejected 6 of 6 duplicates and passed every correct and natural run.

**Gold-sanity is non-negotiable.** An un-gated repair loop claimed 10 of 11 tasks fixed; a Docker re-check found 1 of 8
judge-endorsed blocking tests valid. With a gold gate, **65 of 105 decisive LLM-written augmentations failed on gold
(61.9%)**; gate with retry fixed 9 of 11, gate without retry 3 of 11 (2606.16062).

**Generated tests count only behind gates.** A gold-informed generated test counts only if it passes an alternative valid
fix and kills the revert and the generic mutants. Showing a candidate solution to the check writer lowered test quality
for every model, even when told to ignore it (Scoring Verifiers, 2502.13820). Prompting for failing tests raised attack
rate by 13.5 pp but cut output accuracy by 16.5 pp (UTGen, 2502.01619); Crucible computes every expected value from the
gold run instead of predicting it.

**Do not adopt.** EvolveCoder's rule "a test passed by < 10% of solutions is wrong" deletes the only discriminating
checks on hard tasks (2603.12698).

**Gates.** G3, G4 (red on start), G5 (every negative fails per class; the side-effect class present; kill floor ≥ 95%
of valid generic mutants with the corrected protocol; the held-out mutant set unused in repair), G5b (wrong-task grid),
G6 (undoing any single planted change flips some check), G14 (alternatives pass), G16 (reward causality).

**Evidence.**

| Mechanism | Paper | Number |
|---|---|---|
| Near-miss negatives remove lenient verifiers | SCA (2506.01716) | ≥ 3 near-miss cases removed all false positives on 200 audited trajectories |
| Mutation strengthening | SWE-ABS (2603.00520) | 19.78% of passing patches rejected; top agent 78.80 → 62.20; mutation was the lever (251 vs 198 strengthened) |
| Differentiating tests | PatchDiff (2503.15223) | 29.6% of plausible patches differ from the oracle; 82.7% of those missed by the full suite; $0.04–0.10 per patch |
| Verifier precision changes RL | HardTests (2505.24098) | Qwen3-4B pass@1/5/10: 39.4 / 57.9 / 64.8 with HardTests vs 37.0 / 51.0 / 57.1 with TACO tests |
| Multi-oracle test validity | Kimi k1.5 (2501.12599) | A test is valid if ≥ 7/10 references agree; a problem is admitted if ≥ 9/10 pass (32% admitted) |
| Kill floor is Goodhart-prone | GateTruth (2608.12635) | 10 testbenches had been revised to pass the gate; first claim 60/60, corrected 46/60 |
| Concern-targeted mutants | ACH (2501.12862) | 36% of sampled mutants were equivalent; with identity and comment-strip pre-filters the detector reached 0.95 precision / 0.96 recall |
| End-state perturbations | MCPMark (2509.24002); Gaia2 | Hand-edited near-miss end states; automated here |

**Failure modes.**
- **One agent edits solution or verifier until they agree** (DeepSeek-V3.2). The writer of a check is never the writer of
  the solution, and every hardening round re-runs the negatives.
- **Over-strict single-reference checks** (WEFT; SWE-ABS: 10.6% of instances over-specialised to gold after decoupling,
  fixed by hand). The two-sided rule and G14.
- **Model mutants that only add a comment** (ACH: 61% of equivalent ones). Strip comments; an F6 lint rejects planted
  faults that only add a comment, which are no-op tasks and leak risks.
- **Verifier revised "to pass the gate".** Log every such revision with its pre-revision kill rate in provenance.

**Build in Crucible.** Extend `accept/mutation/actions.py` (drop_step and drop_read exist) with the operators above over
tool-call programs, and `factory/lanes/windows.py` for the debug lane's side-effect negatives. Gold-sanity in
`factory/verifier2.py`. The kill runner as `MutantRun{op, site, status}` and `KillReport{pooled, per_task, by_op,
baseline_ok}`. `VerifierBundle.quality` written per task.

**Priority.** P0 for the operator library with the side-effect class, gold sanity, red-on-start and the kill report;
P1 for concern-targeted mutants, hidden compositional checks and two-sided repair.

### 3.7 F6 — The gate table

**Purpose.** Run every check that can kill a bad task before the student pays to measure it. Gates are ordered by cost:
deterministic CPU gates first, then frontier-model gates, and only then the paid student measurement (F9).

**Numbering.** G1–G14 keep their plan meaning. G5b and G15–G24 are new. Template-level gates (run once per App or harness)
are H-gates, judge gates are J-gates, and simulated-person gates are U-gates. All new gates are **[Proposal]**, each
backed by the cited evidence. §14.1 records how two extracts proposed clashing numbers.

#### 3.7.1 Template-level gates (per App, harness or World template)

| Gate | Check | Evidence |
|---|---|---|
| **H0 Trust boundary** | Recon of entry points and scorers, static scan (semgrep, bandit, hadolint, an AST trust mapper), then an exploit attempt through the official entry point with a minimal agent. Run once per template and on every harness change. Covers isolation, answers shipped with the test, code execution into the evaluator, judge prompt injection, weak string matching, validators never called, trusting agent-controllable output, excessive permissions. | BenchJack (2605.12673): exploits on 10/10 benchmarks; a nine-line `conftest.py` passed about all SWE-bench tasks; design flaws (a shared trust domain) cannot be patched in code |
| **H1 Parser gold sanity** | Parsed per-test or per-check outcomes on gold and known-failing runs must match the framework's machine-readable report | UTBoost (2506.09289): the log parser was wrong on 54.7% / 54.2% of instances |
| **H-tool** | 100% of blind, frozen tool tests pass (§3.2) | Crucible rule; ScaleEnv, EnvScaler, EnvFactory precedents |
| **H-conform** | Golden-fixture conformance for any App imitating a real API | ClawsBench (2604.05172) |
| **H-error** | Each tool rejects the four error operators with a non-crash error | GTM (2512.04535) |
| **H-soak** | Thousands of resets; fingerprint equal after each; memory bounded | AgentGym-RL (2509.08755); VeriEnv (2603.10505) |
| **H-settle** | `settle()` reaches terminal states; snapshot and settled grades agree | 2608.14940 |
| **H-parity** | An adapter (Harbor, MCP sidecar) gives the same grades as the native path on parity runs | Terminal-Bench (2601.11868) |
| **H-fidelity** | Customer-data seeds pass the privacy and fidelity gate (§8.3) | 2508.06647; 2210.14348 |

#### 3.7.2 Task-level gates, in run order

| Gate | Check | Cost | Evidence |
|---|---|---|---|
| **G1 Integrity** | Start state satisfies schema, FK and constraints; cross-app coherence | CPU | ScaleEnv; Gaia2's listed gaps |
| **G2 Determinism** | Oracle twice → same fingerprint; delta chain replayed twice → same hash; reset fidelity; expectation hashes stable; pinned render libraries | CPU | AZR (2505.03335) runs twice and bans `random`, `datetime`, `time`, `subprocess`; WildAWM; VeriEnv |
| **G7 Tool-only** | The oracle calls only policy-visible tools; a hardcoded oracle that echoes the answer fails | CPU | DeepSeek-V3.2 [PARTIAL]; Terminal-Bench |
| **G8 Communicable** (static) | (1) every oracle argument tagged `external` appears in the instruction or the user's known info; every `internal` one comes from an earlier oracle call; (2) every literal, rule, format, ordering, default, boundary and file location a check reads is stated, documented in a file the instruction points to, a named deliverable, or a pure anti-placeholder check; (3) every hidden rule is named in the `AnswerContract`; (4) every latent-derived answer is reachable through visible tools; (5) every user-side fact is a deliverable shard; (6) tested ⊆ described **and** described ⊆ tested | CPU | SWE-Bench Pro 25.9 → 8.4 (2509.16941); SETA's four JSON literals (2607.10891); RST rule D.1 (2608.05466); Terminal-Bench bidirectional check; 2609.08149: 92 of 102 refined tasks needed requirement edits |
| **G9 No shortcut** (structural) | No constant node attached directly to the target; no constant–constant edges; no pseudo-superset hop; the answer is not in the question; a tool that solves the task in one call is a shortcut | CPU | WebShaper (2507.15061); TaskCraft (2506.10055); SearchEyes: without anti-shortcut filters 61.8 → 54.1 (2607.05943) [contaminated, single runs] |
| **G11 Leak and contamination** | Domain leak rules; 13-gram, then embedding nearest-neighbour and a paraphrase judge against the sealed set and held-out families; tool-catalogue and schema overlap; VCS, network and metadata channels; compiled or reference artefacts; no tool output computed from the gold answer; date rule | CPU | AutoEnvScaling 13-gram + blocked benchmark pages; WebUniverse (2610.33149); 2609.08149 leak channels; GTM |
| **G13 Interaction completeness** | Dependency-closed BFS from the start state, executing each tool; every exposed tool returns a non-crash observation | CPU | ScaleEnv |
| **G23 Generator regularity** | A trivial positional or lexical heuristic does not beat chance on the family | CPU | 2607.22368: generator topological-order shortcut G = 0.621 |
| **G24 Planted-fault cue** | A classifier or diff heuristic reading one file, hunk or step at a time does not find the planted fault far above chance (style, comments, odd identifiers, "always the last-modified file") | CPU | 2610.17747: a one-step TF-IDF classifier found the faulty step in 311/311 cases |
| **G4 Red on start** | Target checks fail on the untouched start state with an empty answer; preservation checks pass on it; null agents score 0: empty, claim everything, dump all readable data, list all answers, always infeasible, always FAIL, always act, never act, act on the else-branch, default and idle drivers; a random or single-field change does not satisfy Φ | CPU | WildAWM: dropping it 31.86 → 24.08 (2610.26828); ABC: do-nothing 38% on τ-bench airline (2507.02825); WorkBench; OSWorld FAIL hack |
| **G3 Solvable** | Oracle scores 1, Φ = 1, no path violation; replay agreement for solver families | CPU | SCA; VHD-Play |
| **G5 Negatives fail** | Every negative fails, per class; side-effect class present; kill ≥ 95% of valid generic mutants under the corrected protocol; held-out mutants unused in repair | CPU | §3.6 |
| **G5b Wrong-task grid** | Each verifier rejects its one-parameter twin's oracle run and the sibling grid | CPU | 2610.09142 |
| **G6 Inverse mutation** | Undoing any single planted change flips some check; drop *each* oracle step, not only prefixes; restoring any single corrupted record flips some check | CPU | SSR (2512.18552); τ² prefix wording |
| **G14 Alternatives** | ≥ 2 valid alternatives, one from a different family, pass; their action traces differ; a format- or precision-different but semantically equal alternative passes under column policies; checker recall measured on a sample | CPU + tokens | Official web checkers: 55.9% recall (2504.08942); WebArena 18–24% wrong checks (2408.06327); Hivemind: "two alternatives" may be the same solution (2510.22954) |
| **G15 Gaming-probe battery** | Tev's 14 probes (empty output, minimal schema-valid output, copied input, single-seed hard-coding, path traversal, private lookup, …) plus "repeat the rewarded action" score 0 on the verifier and on Φ; any rollout above the oracle's reward or above 1 normalised triggers G12 review | CPU | Tev (2610.18868); AutoEnv's "verified" reward reached 138.84% of max (2511.19304) |
| **G16 Reward causality** | In a passing state, mutate each cause; the matching reward component must drop | CPU | Tev |
| **G17 Solver agreement** | Two oracles from different families reach the same end-state class on every seed | tokens + CPU | Tev |
| **G18 Fail-closed** | Inject verifier faults (error, timeout, unparsable output, skipped check, unknown event); each must yield FAIL with a reason code | CPU | BenchShield I5 (2609.11028): no isolation mechanism moved fail-closed handling |
| **G19 Clean-custody regrade** | Copy only the declared deliverable (staged diff, exported state) into a fresh image or World and regrade; on disagreement take two more and use the majority; restore every path the test patch touches; delete agent files that collide with hidden tests; clear caches and locks | CPU | SWE-Bench Pro V2 (2610.24206): a `python3` replaced by a PASSED printer scored 1 in-sandbox |
| **G21 Isomorphic regrade** | Replay the oracle and each negative on an id-renamed twin (`SeedBundle.isomorph(Φ)`); verdicts must not change | CPU | IPT (2604.15149): an extensional verifier let rewards diverge near step 250; the isomorphic one did not |
| **G22 Honeypot twin** | A watched copy of solution, tests or gold state; any read, write or exec by an oracle-derived bait negative scores 0 and is flagged | CPU | HVTB (2608.22103); HVE (2605.20744) |
| **G20 Sufficiency** | For rebuilt start states: a read-only judge reports nothing critical missing | tokens | Terminal-Universe (2609.04148) |
| **G10 Flake bound** | Run completeness first (no empty transcripts); then 60 oracle grades, sequential, Clopper–Pearson upper bound on false negatives ≤ 0.05 | CPU (slow) | `factory/flake.py`; 2610.27465; GateTruth sequential certification |
| **G8x Executable communicability** | A privileged frontier solver given only the needed tools, in order, reaches the goal; a recoverability session without hidden state re-derives each analytical finding | tokens | RandomWorld (2506.11045); Data World auditor |
| **G9x Shortcut probes** | Instruction-only, no-World, no-reasoning guessing (8 tries), search-only and read-only agents fail, on a frontier model *and* the student family | tokens | MCP-Universe admission (2508.14704); Kimi k1.5; Online-Mind2Web search-only solved 51% of WebVoyager (2504.01382) |
| **G12 Hackability** | A different-family attacker with K tries and 3 rounds finds no exploit (H(t) = 0 within that budget); deterministic RHB detectors run first; re-run on every variant that raises the band | tokens | 2606.16062; RHB (2605.02964); exploit rate jumps at the first step the agent cannot privately verify |

H(t) = 0 is a statement about the attacker's budget, not a proof. Unhackable rewards are non-trivial only on finite policy
sets (2209.13085), so the real defence is restricting the policy set (tool-only actions, path rules, typed actions) and
re-running G12 when the student starts to saturate a task.

#### 3.7.3 Judge and simulator gates (run on any task that uses them)

| Gate | Check | Evidence |
|---|---|---|
| **J1 Calibration** | The judge holds κ, precision and recall on its anchor set (§9.2), including recall on seeded hacks, near-miss acceptance and benign-perturbation acceptance; it operates above a calibrated confidence threshold λ | AgentSynth: near-miss states accepted 12% (2506.14205); Trust or Escalate (2407.18370) |
| **J2 Counterfactual probe** | A fresh session re-judges blind, then is told the opposite verdict; it passes only if it defends with cited evidence; caving by editing verifier scripts is detected by diff and reverted | WildAWM: dropping the probe 31.86 → 27.91 (2610.26828) |
| **J3 Master keys** | Any non-zero false-positive rate on the domain's master-key set disqualifies the judge | 2507.08794 |
| **J4 Style and sanity check** | Any soft judge used in a reward has a task-agnostic style / sanity check against content flooding | Gaia2 / ARE: templated pseudo-code overwhelmed the soft judge in RL (2509.17158) |
| **J5 Judge consistency** | During synthesis, executed verifier = LLM judge = agent judge on k frontier pilot rollouts; disagreement flags a verifier defect | WildAWM: dropping it 31.86 → 22.56, the largest effect; WEFT kept 1,954 of 5,000 by rubric–executor agreement |
| **J6 Prompt-change probe** | Every judge-prompt edit passes equivalent pairs, clear passes and fails, and moderate gaps; constant-Tie canary | 2609.12439: debiasing prompts drove a 98-point loss on moderate gaps |
| **J7 GV-gap** | Filtering by the judge raises precision against human labels; report the relative gap per judge version | Mind the Gap (2412.02674) |
| **U1 Incentive validity** | Vary the persona's stated goal; behaviour moves in the predicted direction; a scripted positive-control persona shows the metric detects the effect | 2609.01519: a stronger profit prompt gave *less* profit (33.8 vs 57.5) |
| **U2 Simulator fidelity** | Front-load ratio, identifiers per turn and politeness within a band of a human or customer-trace reference; ECE by difficulty bin on a sample | Sim2Real (2603.11245); Lost in Simulation (2601.17087) |
| **U3 Simulator swap** | p̂ stays in band when only the user model changes to another family; G8 holds under both | Lost in Simulation: 11 of 18 tasks changed bin after a one-line politeness tweak |
| **U4 Real-vs-sim parity** | For any LLM-simulated App or tool tier: a fixed agent's scores in the real and simulated tier match within noise; consistency ratio CR = replayed-in-real / sim successes ≥ about 0.9 as a starting bar | MirrorAPI (2503.20527); Trace2Env CR 0.032 vs 0.914 (2610.06100) |

#### 3.7.4 Repair rounds

```text
repair(task, failed_gate, evidence):        # at most 3 rounds per task; budget logged
  side = attribute(failed_gate, evidence)   # task | oracle | verifier | world | simulator | harness
     by replay: run reference and demonstration from the same start; agree on outcome + policy checks?  # GRADUS
     a model may label the side; execution certifies it
  route:
     world/harness  -> F1 backend or the App template (re-run H-gates and every task of the family)
     task/state     -> F2/F3 (start-state patch re-runs G1, G2, G8)
     verifier       -> F5 (gold sanity, negatives, G14 after every change)
     instruction    -> state the missing requirement; never describe how the oracle implements it
     policy failure -> leave the task unchanged
  rules:
     repair the instruction, never weaken a check to fit a wrong solution        # 2610.24206
     a check is relaxed ONLY when an accepted alternative fails it (over-strict, two-sided rule)   # SWE-ABS
     every instruction edit is validated by a blind re-implementation by a different family that saw
       neither the task nor the oracle                                             # 2610.24206
     repairs may not weaken semantic checks, remove shortcut protection or expose private requirements;
       edits limited to a file allowlist                                           # RST (2608.05466)
     a defect in tests, fixtures, the verifier or the image means quarantine or discard
     report yield with and without repair, and the difficulty mix of first-pass vs repaired accepts   # GRADUS
```

Repair raises yield and keeps hard tasks. APIGen-MT's feedback summariser raised blueprint yield 28% → 70%
(2504.03601); GRADUS's validity loop with task recovery raised telecom / disambiguation 0.65 / 0.40 → 0.75 / 0.60, and
recovered tasks were disproportionately hard (2610.27781). But repair must re-pass every affected gate: WildAWM caps fixes
at 2 rounds and re-runs every gate (2610.26828).

**Build in Crucible.** `factory/generators/base.py` `deterministic_gates` (S6 → S5 → S8 → S3 → S10 today) becomes the
generic gate runner over `GateReport{verdicts, repair_rounds, H_t, evidence_refs}`. Existing pieces: S5 null agent and
clean-World checks (`factory/longtask.py:262-298`), S8 floors and mutants (`factory/verifier2.py`), S10 flake
(`factory/flake.py`), the attacker (`forge/hacker.py`, `accept/checks/redteam.py`), differential and alternatives
(`factory/differential.py`, `alternatives_gate.py`), fairness (`factory/fairness.py`), contamination
(`factory/benchaudit.py`). New: G4 red-on-start, G5b, G15–G24, H0, J- and U-gates.

**Priority.** P0: G1, G2, G3, G4, G5, G5b, G6, G7, static G8, structural G9, G10, G11, G13, G15, G18, G19 (debug lane),
G22, H0, H1, J3. P1: G14 at full strength, G16, G17, G20, G21, G23, G24, G8x, G9x, G12 on every task, J1–J7, U1–U4.

### 3.8 F7 — Behaviour signatures

**Purpose.** Measure what a task actually makes the solver do, from the oracle's executed trace, so that cell membership,
dedup, diversity and intent are judged by behaviour rather than by the generator's label or the instruction text.

**Method.**

```text
BehaviourSignature(oracle_trace, StateDelta):        # through _call_tool and StateAdapter.diff
  apps, tools (multiset), tool_chain (ordered), interfaces_used
  tables_read, tables_written, rows_written, entities_touched, people_contacted
  events_crossed, virtual_time, calls, horizon, cross_app_flows (dataflow edges between apps)
  memory_span, app_switches                            # AgentSynth (2506.14205)
  elicitation_count, necessary_waits, unnecessary_waits   # UserBench; TimeArena
  efficient_path_available, efficient_path_taken        # OSWorld-MCP TIR (2510.24563)
  traversed_graph_edges vs TaskProposal.formal_graph     # WebShaper / SearchEyes
  measured_cell                                          # derived from the above
  debug lane: files and functions touched, milestone test ids, language, fault count, executed links, fault class
```

**Rules.**
- **A claimed cell that disagrees with the measured cell is relabelled or rejected.** Summariser hallucination ("claimed
  cell ≠ measured cell") is the failure it blocks (Explorer's summariser invented constraints, 2502.11357).
- **A claimed skill must be observable** in its `SkillLeaf.observable_in` fields, else the task is relabelled
  (ACE, 2608.27260).
- **A variant must change the signature on the knob it moved.** Stylistic and interaction variants keep the signature and
  should not move p̂; structural ones must change it (Reasoning Gym; GRADUS kernels; AutoEnv BaseEnv / ObsEnv / SkinEnv).
- **A composite whose signature is the union of its parents with no new edges is rejected** (the AZR collapse).
- **Normalise before signing:** drop dead steps (RandomWorld) and echo turns, so formatting does not create false
  novelty.

**Gates.** F7's gate is "measured cell = claimed cell, and the declared difficulty vector δ matches the measured
signature" (WebForge's declared vector, 2604.10988).

**Evidence.** Two tasks that touch the same tables with the same tool sequence are the same task to the learner. TOUCAN's
"same tools, same order, same outcome, new persona" variants are duplicates to the learner (2510.01179). AppWorld reports
86% of tasks with a unique API set over 52 App combinations, a usable signature meter (2407.18901). AgentMercury's task
text looked diverse (mean pairwise similarity 0.13) while its median nearest-neighbour similarity was 0.83 and 8.3% of
tasks had a near-twin, mostly across Worlds sharing templates (2608.20634).

**Build in Crucible.** `factory/signature.py` (new), reading the World's tool-call trace and `StateAdapter.diff`; the
debug lane's executed-edge data from `generators/rebreak.py` supplies its links. **Priority.** P0.

### 3.9 F8 — Dedup and the cheap diversity layers

**Purpose.** Drop duplicates before paid measurement, using three layers that mean different things.

| Layer | Method | Meaning | Action | Calibration |
|---|---|---|---|---|
| 1 Perceptual | MinHash on text; ROUGE-L < 0.7 to the pool (Self-Instruct) | Near-identical text | drop | — |
| 2 Semantic | Embedding cosine (all-mpnet-base-v2 at 0.9 took 75,084 problems to 30,437, 2603.12698); a reranker layer (AgentFrontier ≥ 0.7, a reranker score over questions, not a cosine, 2510.24695) | Same content, different surface | **hold for review** | On a small human-labelled set of task pairs (duplicate vs legitimate variant) |
| 3 Behavioural | Equal `BehaviourSignature`; ordered tool chain a non-contiguous subsequence of another's (WEFT); identical check pass vector (EvolveCoder) | Semantically redundant | drop | — |

Two tasks with near-identical instruction embeddings can differ in the one World fact that changes the solution, so the
embedding layer holds rather than drops (SemDeDup card critique, 2303.09540). Removing 37% of LAION by semantic dedup cost
nothing on zero-shot accuracy, and which duplicate is kept barely mattered (2303.09540).

**Also at F8.**
- A per-generator `distinct_k` meter: sample k proposals for one cell and count signature classes (NoveltyBench,
  2504.05228).
- The pool-level shortcut diagnostic (Level 1 of 2610.22243): can instruction length, App name, a template phrase or
  entity count predict pass, fail or band? If yes, set quotas per generator, template and source.
- Inspect the largest duplicate clusters: templated "one name swapped" items dominate them (SemDeDup).

**Failure modes.** Content-only rewrites ("search running shoes" → "search slippers", UI-Simulator-Grow, 2510.14969) must
be flagged as near-duplicates unless a knob changes. Instance counts overstate diversity: WorkArena's 19,912 instances
come from 33 templates (2403.07718).

**Build in Crucible.** `forge/dedup.py` (built, not wired into the debug lane) and `factory/ortho.py` (4-layer
orthogonality); `DedupDecision{layer, nearest_id, sim, signature_equal, action}`. **Priority.** P0 for wiring; P1 for
calibration of layer 2.

### 3.10 F9 — Per-student measurement

**Purpose.** Place each gated task in or out of the learnable band *for a named student*, at the lowest student spend.
Today this stage costs about $14 per k = 8 and runs after the kill (map §4.4, §5.1). Section 7 gives the band theory and
knobs; this section gives the measurement protocol.

**Method.**

```text
F9(task, student, checkpoint, user_sim, time_mode):
  0. prerequisites: G1-G24 passed; the run-completeness check is on; infra failures are labelled and dropped
  1. rank before paying (no student rollouts):
       predictor p_pred(task, student) from text + signature features + structural predictors
         (files touched, gold writes, horizon, apps, entities x attributes x operations)      # map: MAE 0.204 today
       optional student NLL on the oracle program (one forward pass), only after it is validated   # ToolACE card
       optional teacher-forcing loss band (25th-75th percentile)                                    # UI-Simulator
       frontier token spend as a hardness feature (about +3 pp when used as a filter)               # 2606.24855
  2. frontier audit on a SAMPLE (not every task): k = 2, two families, stratified to hard tasks    # bug finder
       frontier-fails-but-student-passes and predicted-easy-but-nobody-solves -> audit queue      # ABC T.10
  3. sequential student measurement:                                          # AutoEnvScaling (2610.14163)
       k = 4 rollouts over different seeds and clock offsets
       if mean within 0.15 of a band edge: expand to k = 8
       0/4 -> ladder (section 7.5) or zero-solve audit; never spend 4 more
       4/4 -> mutation queue (harder variant)
       keep iff Wilson band check passes AND std >= 0.1
  4. LILO-style allocation within a batch: 2 rollouts per task, then repeatedly
       give 2 more to the task with the highest p_hat(1 - p_hat)              # 2502.12272
  5. IRT: fit 2PL over the (task x student/checkpoint/frontier) pass matrix with priors, d = 1-2 per family;
       b_j relative to each student's theta is the band; tasks with near-zero or negative
       discrimination go to the defect queue                                  # Fluid (2509.11106)
  6. record DifficultyRecord with the full key, pass@k and pass^k, consistency, cost
  7. after training starts: refresh p_hat from training rollouts every step (zero extra cost)   # RODS (2606.19047)
```

**The keep rule.** The current rule stays: training admission at student full-pass rate in [0.1, 0.6] over 8 attempts,
benchmark ≤ 0.5, and the on-intent share ≥ 0.5 set by the lead on 2026-10-27 (map §3.1). Section 7.1 adds a symmetric
(0.2, 0.8) arm to the R2 measurement and a ranking utility inside the band. Section 3.11 changes how the on-intent share
is *computed* (fault-side accounting), not the bar.

**What the protocol may and may not do.**
- **A predictor halt is never a training reward.** EarlyEval's failure head halted 34.8% of SWE-bench Verified runs at
  95.0% accuracy with mean |ΔPass@1| 1.1 pp (2609.02783); it is a measurement filter only, and `halted_by_predictor` is
  recorded so Wilson bounds can be recomputed.
- **Flaky tasks look maximally learnable.** p(1 − p) and mean |Â| reward aleatoric noise as well as learnability, so G10
  runs before any learnability-based admission (2502.12272; 2505.14970).
- **Measure on the live student.** Fixed checkpoints see batches harden while the live one stays near 0.5
  (2504.03380).
- **Measure pass@k as well as pass@1 before admitting a family for the 397B.** ProRL found a strong negative correlation
  between base pass@128 and RL gain at pass@128; families where pass@k ≈ 1 even at mid pass@1 are "diminish" families
  (2505.24864).
- **Measure in the training time mode and with the training simulator.** Record the eval-time simulator separately
  (Lost in Simulation; UserRL: a cheap open user transferred to GPT-4o evaluation, 2509.19736).

**Evidence.**

| Mechanism | Paper | Number |
|---|---|---|
| Band + spread + adaptive k | AutoEnvScaling (2610.14163) | 3.1× more non-zero-variance groups than its T_max baseline |
| Zero-cost p̂ from training rollouts | RODS (2606.19047) | Boundary-zone variance 2.0–2.2× the others over 4,800 measurements |
| Learnability sampling | LILO (2502.12272) | 1.5×–3.3× speed-ups to baseline-best; it speeds fitting, not generalisation |
| Adaptive IRT selection | Fluid (2509.11106) | 50 adaptive items beat the full set on validity; tinyBenchmarks at 100 items was noisier than random (30.5 vs 19.8) |
| Small-k noise | GenEnv bound (2512.19682) | About 490 rollouts to separate two tasks 0.2 apart in band distance at 95%: per-task placement at k = 8 is noisy, so pool estimates per cell |
| All-fail groups | Tev | (1 − p)^G: 66% at p = 0.05, G = 8 |
| Seeds over repeats | AndroidWorld (2405.14573) | Across-seed variance exceeded within-seed variance |
| Runs per stochastic multi-turn task | VitaBench (2509.26490) | k = 4 cut the MSE of the mean by 77.5% vs k = 1; k = 8 added little |

**Failure modes.**
- Bands set on a frontier model, not the student (ProgAgent; EnvCraft at 27B). The evaluator must be the learner.
- Discarding frontier-trivial tasks biases the pool to frontier-hard tasks that may be unlearnable for the 35B
  (EnterpriseOps-Gym, 2603.13594). Frontier pre-runs are bug finders, not difficulty filters.
- Dynamic sampling that silently drops all-fail groups hides broken tasks (DAPO, 2503.14476). Every dropped all-fail group
  goes to the zero-solve audit; every all-pass group to the mutation queue.

**Build in Crucible.** Key `forge/difficulty.py`'s discounted counts by the full `DifficultyRecord` key; sequential k in
`factory/targeting.py` (the keep rule, `KeepRule`) and `scripts/lh_debug_target.py`; IRT from `bench/irt.py` (2PL exists,
used by the benchmark runner only); predictor in `factory/calibration.py` and `factory/difficulty.py`; training-rollout
reuse through `train/curriculum.py` (opt-in today). **Priority.** P0 for keying and sequential k; P1 for IRT, LILO
allocation and the validated NLL predictor.

### 3.11 F10 — Intent probes and the IntentCard

**Purpose.** Show that success *needs* the intended skill, and that failures *involve* it. Today's mechanisms
(`task_intent`, `failure_skill`, `disclosure_check`, human-approved intent cards) show only the second (map §3.5).

**The verdict vocabulary.** VALID, INVALID or INCONCLUSIVE (2609.01519). A known incentive or isolation violation is
INVALID. An underpowered or non-monotone probe is INCONCLUSIVE, never "no effect". Protocol validity defines VALID: a
score supports a capability claim only if the protocol keeps that capability necessary; a run is mislead-positive only
with an exposure, active use of it, and grader credit (2607.22368).

**The probes.**

| Probe | Run | INVALID or INCONCLUSIVE signal | Evidence |
|---|---|---|---|
| Null instruction | Empty or generic instruction; frontier k = 2, student k = 4 | Pass rate ≈ full-instruction rate | Plan §3.7 |
| Instruction only / no World | Answer or end state from text alone, 8 tries | High hit rate: memorisation or guessing | Kimi k1.5 (2501.12599); SWE-Bench Illusion file-path probe: 60–76% on Verified vs < 53% on outside repos (2506.12286) |
| Written-out vs informed vs agentic | The same draw written out, with parameters revealed, or with parameters acquired by probing | Most of the gap is stateful execution, not problem solving | VHD-Play: base 0.962 / 0.231 / 0.204 → trained 0.992 / 0.875 / 0.815 (2609.27321) |
| No-User and Oracle Plan | No-User gives the ticket and all tools; Oracle Plan gives the tool sequence | A near-zero Default vs No-User gap means the user part is decorative | τ²: 18–25 points (2506.07982) |
| Policy removal | Drop the policy document | A policy-heavy family must drop sharply | τ-bench airline 33.2 → 10.8 vs retail 61.2 → 56.8 (2406.12045) |
| Budget 2× and direct access | Twice the tool calls; or remove composed dependencies | A jump means the hardness is budget or access, not skill; the card says so | RES: removing composed dependencies → 100% pass from 0% on 15 tasks (2610.16397) |
| Single-subtask split + hidden compositional checks | Solve each subtask alone | Joint pass ≈ product of singles: the subtasks do not interact. Needs SpecBench-style composition checks to measure anything | Plan [inferred]; SpecBench (2605.21384) |
| Format and paraphrase perturbation | Format-only or paraphrase change | A collapse means the task teaches surface form | 2602.16666 [PARTIAL]: models broke on rephrasings, not on tool errors |
| Simulator swap and behaviour mode | Change only the user model; or the behaviour module | p̂ leaves the band: the task measures the simulator | Lost in Simulation (2601.17087); 2509.23124 |
| Time mode | Paused vs real latency vs instant | Large swing: the task measures latency handling | Gaia2 (2602.11964) |
| Oracle evidence given | All evidence in context | Pass with evidence, fail without: the task exercises search | BrowseComp-Plus: gpt-4.1 93.49% with all evidence (2508.06600) |
| Cue injection | A misleading cue in the start state (stale answer field, wrong note from a person, fake grader comment) | Switching to the cue: trusts the cue over the World | 2505.05410 |
| Elicitation | A short instruction that *asks* for the shortcut | Passing the verifier this way means the shortcut is reachable (a G9/G12 failure) | 2510.05024 |
| Counterfactual rerun (referee) | Re-run with a suspected shortcut element removed | A pass that disappears is shortcut-dependent | 2608.03744 |
| Capability control | Partial correlation of family pass rate with a general capability composite (frontier θ from IRT) | Fully explained by general capability: weak skill-specific evidence | 2607.28685 |
| Failure consistency | All trials fail the same checks | Instruction defect | Terminal-Bench (2601.11868) |
| Zero frontier solves | Frontier pass@k = 0 | A defect until ≥ 2 confirmed alternatives or a clean agentic audit; per-step model-pool mixing may prove solvability first | Zero-solve tasks hold more judge-flagged tasks (2606.26300); ComputerRL (2508.14040) |
| Lucky-pass rate (after training) | Merge ≥ 2 (ideally 5) passing traces into a prefix tree; tier each | Most passes Lucky: weak checks or trial and error | AgentLens: 10.7% Lucky overall (2605.12925) |
| Static audit | A different-family auditor on instruction, environment and evaluation axes, severity by thought experiment | A major flag routes to repair or INCONCLUSIVE, not INVALID (precision 58–73%) | ABA (2605.26079): $0.42–0.80 per task |

**Fault-side accounting for the on-intent share.** Each failure label is an edge between components (model, owner,
grader, third party, context, memory, tool, local or external environment) with a fault side and a mode, labelled at the
earliest unrecovered failure (2607.28802). A failure counts against the *task* when its side is owner, grader, tool or
environment: those route to repair, never to training. It counts as on-intent only when it is model-side and its mode
maps to the family's skill leaf. **[Proposal, reconciling extract D with the lead's 2026-10-27 rule]:** keep the lead's
0.5 bar, and compute the share over model-side failures only, with non-model failures excluded from the denominator and
sent to the fixing loop. This keeps the lenient bar and still fixes framework causes. Use three non-student judge
families with 3-of-3 agreement for automatic routing, else abstain: ≥ 3 of 4 judges gave 0.83 precision at 90%
coverage; unanimity 0.96 at 68% (2607.28802).

**The IntentCard.**

```text
IntentCard
  intended_skills, claim, negative_definition_checked
  ablations: {probe: {run_ids, delta, ci, verdict}}         # paired, clustered differences
  inflation_gap G = S_exploit - S_intended                  # only with a defensible comparison (2607.22368)
  on_intent_share (fault-side accounted), fault_sides, attributor_disagreement   # 2603.25001
  lucky_pass_rate?, hacked_pass_share, H_t
  verdict: VALID | INVALID | INCONCLUSIVE, reasons, plan_if_inconclusive
```

**Rule.** Hack and intent flags come from execution and counterfactual runs, never from the student's stated reasons:
drifters toward a hidden decoy rubric almost never said so (1 of 10 on text, 1 of 134 on imaging, 2608.03744), and about
32.7% of exploit reasoning recovered by direct probes never appeared in the chain of thought (2606.09711).

**Cost.** About 4 extra short runs per task; F10 reuses F9's rollouts for failure consistency, lucky passes and fault
sides.

**Build in Crucible.** `factory/targeting.py` (the counterfactual, paraphrase and prerequisite probes exist at
`:217-456` but are off in the debug lane), `factory/traceaudit.py` (`first_wrong_step`, typed proposals), `factory/cards.py`
(the closed cause taxonomy). **Priority.** P0 for fault-side accounting, failure consistency and the zero-solve rule;
P1 for the ablation set; P2 for capability control.

### 3.12 F11 — Archive, pool and curriculum

**Purpose.** Decide which gated, measured tasks enter the training pool, the held-out families and the benchmark, and in
what proportions, and sample from the pool during training.

**The archive.**

```text
Archive = MAP-Elites grid over DomainSpec axes, with islands per generator family     # AlphaEvolve (2506.13131)
  cell elite score = learnability(cell, student) x interestingness
     learnability: band membership, then Gaussian-variance utility exp(-(Var - 0.25)^2 / 0.02)   # SPICE (2510.24684)
     interestingness: a different-family judge votes "interestingly different" vs "near-copy"
       against the cell's elites and the student's mastered tasks; it only rejects near-copies,
       never ranks valid alternatives                                                    # OMNI (2306.01711); Hivemind
  stepping stones: novel non-elite tasks are kept, not deleted                            # DGM, ADAS
  admit into a sparse or empty measured cell first, then by learnability                 # Prismatic (2505.20161)
quotas:
  per cell, per family, per World, per template, per source, per generator family
  lineage cap per cell; <= 4 descendants per parent; caps relaxed only on a logged schedule   # RST (2608.05466)
  lineage gate: >= 80% recoverable ancestry, else the round aborts                        # RST
  per-family floor shares in the sampler                                                  # SEC; DUMP regressions
splits:
  train | heldout_family (15-20% of families, whole) | benchmark (sealed, IRT-anchored)
```

**The in-training sampler.** Prioritised Level Replay over the kept pool, keyed per (task, student, checkpoint):
P_replay = (1 − ρ)·P_S + ρ·P_C with rank prioritisation h = 1/rank, P_S ∝ h^(1/β), β = 0.1, staleness weight ρ = 0.1–0.3
(2010.03934). For GRPO with no value function, the score is p̂(1 − p̂) or mean |advantage|. Scores must fall as a task is
mastered, else replay collapses. RODS's lifecycle rules (2606.19047): variants enter at the next epoch at ≤ 20% of the
active pool; burn-in eviction if the first observation is below 0.20; drift eviction above 0.95 or below 0.20 after a
minimum number of observations; originals are never retired; append-only logs are replayed after a crash. Its ablations
on Qwen3-4B BFCL (56.00 full): random seed selection −4.75, no retirement −3.38, static pool −2.88.

**Composition of the pool.**
- **Uniform over a large, mostly easy pool with a minority of hard tasks** beat hard-biased sampling, which overfit after
  about 36K trajectories; "only easy" beat "only medium" (WebGym, 2601.02439). TRUSTEE found a fixed hardest setting hurt
  (2604.17739). Bands move; the mix is not hard-weighted.
- **Mixed, not staged.** Mixed single-stage beat two-stage in SFT (13.03 vs 10.39, Nemotron-Terminal, 2602.21193).
  Nemotron 3's claim that simultaneous training is less hack-prone has no numbers (2512.20856) and is cited only as
  qualitative. Mixing is not a hack guard: one hackable family contaminates a run even at 50% clean environments
  (2511.18397).
- **Hold out whole families.** Within-domain CV hides lookup shortcuts: an exact-tool lookup scored F1 0.901 under grouped
  CV and 0.495 under domain holdout (2610.22243).
- **Never re-admit a learned task.** Apply a minimal criterion against all past checkpoints (Enhanced POET,
  2003.08536).

**The factory's progress curve.** ANNECS-OMNI: the cumulative count of tasks that were novel by cell or signature, in band
at admission, later solved by some checkpoint, and judged interesting (2405.15568; 2003.08536). Original POET's ANNECS
plateaued; Enhanced POET's kept rising.

**Build in Crucible.** `factory/archive.py` (new); the 40% herding guard (`factory/root.py:29-31`), `factory/family.py`
(its 20-task, 3-surface rule is revisited, §6.3), `factory/sources/lease.py` (owner-disjoint held-out split),
`factory/benchseal.py` and `factory/sealguard.py` (sealing unchanged), `train/curriculum.py` (the PLR sampler).
**Priority.** P0 for quotas, held-out families and the herding guard in the debug lane; P1 for the archive, PLR and
RODS lifecycle.

### 3.13 F12 — Feedback and failure attribution by execution replay

**Purpose.** Turn training and measurement rollouts into signals the loops consume: p̂ updates, executed failure
attributions, failure clusters, hack flags, drifting cells, and value per dollar per generator.

**Executed failure attribution.** The decisive error in a failed trace is the earliest step whose correction alone turns
failure into success (2505.00212 definition; AgenTracer 2509.03312, whose Eq. 4 has 0 and 1 swapped while its Algorithm 1
is right). A snapshot World computes it by execution:

```text
decisive_step(failed_trace, T):
  candidates = cheap judge proposals or bisection over [0, T)
  for t in candidates (bisect: O(log T) replays):
     s = World.restore(snapshot_before(t))
     run Oracle.complete_from(s) twice                 # the correction comes from the oracle, never a model rewrite
     flipped(t) = both runs pass
  t* = earliest t with flipped(t)
  emit DecisiveError{rollout_id, t_star, method: replay, replays_used}
  emit (a) a late-start ladder rung at snapshot_before(t*)
       (b) an Assignment: "variant of task T aimed at the skill at t*"
  cluster on (skill leaf, decisive-step tool, operator class) at t*, not on all errors
```

Model-only localisation is weak. Best TRAIL joint accuracy was 0.183 (GAIA) and 0.050 (SWE) (2505.08638); Who&When's
best step accuracy was 14.2%, near 0% on 93–130-step logs (2505.00212); DeFA reached 17.2% exact step (2610.01256). If the
"correction" is too strong it leaks the answer and every early step looks decisive (AgenTracer); taking the corrected
action from the oracle program avoids this. Restarting from the critical step with guidance raised ALFWorld success 21 →
55 for GPT-4o-mini (AgentDebug, 2509.25370), which conflates localisation with a strong ladder effect.

**Fault injection on passing traces gives labelled failures for free.** Apply a mutation operator to the oracle at a
sampled step, re-run, and keep the result if the grade flips: the injected step is the decisive error by construction
(AgenTracer's D⁺). This is the same mechanism as F5 negatives, so each kept negative is also an anchor item with a known
answer for every judge and clusterer. Never train a labeller on injected-only labels: on AFTraj, a current-turn TF-IDF
method won on injected failures (65.7 vs 48.1) and lost on natural ones (20.0 vs 54.5) (2610.17747).

**Executed step deviations.** At oracle step t, restore the snapshot, let a student or another family propose a different
action, execute it and finish. A pass makes it a valid alternative (G14); a fail makes it an executed plausible-wrong
negative (G5) that no operator would invent (ToolPRMBench recipe made executable, 2601.12294).

**Contrastive skill selection.** Label each trajectory × capability NA, PRESENT or LACKING with a fixed dictionary; keep
capability c when the lacking-rate gap between failures and successes Δ(c) ≥ 0.20 and failure coverage Cov(c) ≥ 0.10,
stable over ≥ 3 relabel runs by a different-family labeller (TRACE, 2604.05336 [contaminated]). The SwS weakness rule
complements it: a task is a weakness iff its accuracy never exceeds 0.5 *and* its slope is negative over training
checkpoints (2506.08989), since base-model failures are often format issues RL fixes quickly.

**Failure taxonomy.** First level: implicit (finished but wrong) vs explicit (context overflow, turn limit, premature
stop, malformed calls); in MCPMark implicit failures were above 50% for every model (2509.24002). Second level: MCP-Atlas
modes with one primary mode, contributing modes and an `is_root_cause` flag (2602.00933). Fault sides add `user_sim` and
`world_bug` (MCPWorld: 48.42% of MCP-only failures were insufficient MCP coverage, a World defect, 2506.07672).
Cheap detectors: a meltdown detector (repeated escalation, no tool calls; Vending-Bench, 2502.15840); a ≥ 10-long repeated
base command ≈ 89% failure (SWE-smith, 2504.21798); command diversity after the first error 0.49 in successes vs 0.18 in
looping failures (Endless Terminals, 2601.16443). Validate any clusterer end to end on synthetic traces with known
categories first (Clio reconstructed 94% of 20 known categories, 2412.13678).

**Hack signals** (§5.6 gives the response): L1 rule flags, a different-family trajectory monitor with an evolving pattern
set, honeypot hits, impossible-twin passes, visible-minus-hidden gap Δ and its P90, r_proxy − r_invariant from isomorphic
regrades, train reward against a stronger oracle on a fixed sample per checkpoint, and clean-vs-raw pass rate per family.

**Build in Crucible.** `factory/traceaudit.py` (deterministic facts → `first_wrong_step` → `RLMAuditor` → typed
proposals) gains the replay method through `World.restore` (`world/world.py:1104-1159`); `factory/failures.py` emits
`Assignment`s instead of waiting `BuilderIntent`s; `factory/audit.py` keeps the quote rule; `observe/otel_export.py`
(planned in Part 1) exports traces. **Priority.** P0 for replay attribution and the `Assignment` consumer; P1 for
contrastive selection, step deviations and clustering at scale.

---

## 4. Simulated people and time (cross-cutting)

### 4.1 Why people are part of the World, and part of the difficulty key

The user moves scores as much as the agent. τ² Telecom was 76.3 with a GPT-4.1 user and 90.4 with GPT-5-thinking
(AutoForge, 2512.22857). A strong user with a weak assistant (44.3%) beat a weak user with a strong assistant (41.7%)
(GEM, 2510.01051). So a task's p̂ is meaningless without the simulator id, and the simulator is versioned like code.

### 4.2 Known defects of simulated users

| Defect | Number | Source |
|---|---|---|
| "Easy mode": most general simulators raise agent success above the human 63.6%, up to 77.8% | 451 humans vs 31 simulators on 165 τ-bench tasks | Sim2Real (2603.11245) |
| Front-load identifiers (4.8 vs 2.6 per turn), rarely short turns (1% vs 29%), too polite (49% vs 15%) | same | 2603.11245 |
| Too verbose: 89.5 words per message vs humans' 15.5; length control brings 21.4 | | SimulatorArena (2510.05444) [PARTIAL] |
| Shift blame to the agent: 48.9% agent-caused in simulation vs 24.5% with humans | | Lost in Simulation (2601.17087) |
| A single cooperative user inflates success: six persona strata cut reward 42% | | GAUGE (2609.12191) |
| Simulator error rates: airline 47% (13% critical), retail 40% (12%), telecom 16% (6%) | two annotators | τ² (2506.07982); its caption says no critical telecom errors while the table lists 3 |
| Disclosure leaks: facts_per_reply 1 → 3 disclosed the reference solution; a lower volunteer interval raised success while questions fell 40% → 26% | EAU run record | 2610.16523 |
| Simulator hacking: trained at simulator T = 0.1, the policy's training reward rose faster but its score against a GPT-4.1 simulator fell after 600–700 steps; T = 1.0 stayed robust | Qwen-32B | LOGIGEN (2603.00540) |
| The opposite direction exists: real users gave cooperative hints that raised the 8B's Telepathy score 0.56 → 0.78 | | UserRL (2509.19736) [PARTIAL] |

The direction of the sim-to-human gap depends on the domain and the simulator (Lost in Simulation vs UserRL), so it is
measured per family (U2), not assumed.

### 4.3 The design recipe

1. **Code holds state; the model voices it** (UserRL; Concordia's grounded variables, 2312.03664). Money, votes,
   inventory and approvals are App state; a Game-Master-style layer rejects actions that break them.
2. **Goal shards with a delivery tracker and an ending verifier** (2509.23124): undelivered shards must be released
   before the user may end, and a premature END is blocked while the agent still needs confirmation. Dialogues that fail
   a goal-alignment check are regenerated, so failures stay agent-attributable.
3. **Release on target**: at most one shard per turn, only when the question targets that dimension, or proactively after
   a fixed number of turns without progress (UserBench, Lost in Multi-Turn). "The case does not provide it" rather than
   invention (EnterpriseBench, 2609.37658).
4. **Behaviour modules** from real logs: unavailable service, tangential, impatient, incomplete; plus SWE-chat's expert
   nitpicker (36.7% even in vibe-coded sessions), vague requester and mind changer (2604.20779). Two behaviours at once
   dropped even GPT-4.1-mini sharply; tangents were worst (−29.1% relative on average); Qwen3-30B-A3B dropped hardest under
   tangents and fragments (2509.23124).
5. **Knowledge boundary and hidden demos** (ToolSandbox, 2408.04682).
6. **BoN-4 user turns with a 0–10 consistency critic** stabilise the user: gpt-4o τ-retail 62.8 → 67.0, variance 11.1 →
   2.6 (APIGen-MT, 2504.03601). A symbolic rule user for gating and an LLM user for diversity can be mixed (EAU, within
   noise).
7. **Train at simulator temperature 1.0; evaluate kept tasks with a different-family simulator** (LOGIGEN; U3).
8. **Never use an LLM user's satisfaction rating as reward** (GAUGE: satisfaction varied by persona, not by success).

**Validation, in order.** U1 incentive validity with a scripted positive control (2609.01519); a disclosure audit per
fact kind with repeats counted (EAU); U2 fidelity against a human or customer-trace reference (front-load ratio, IDs per
turn, ECE by difficulty bin; 2603.11245, 2601.17087); U3 simulator swap; realism and alignment on a small human anchor
that must not fall when the simulator changes (MirrorBench with Human–Human and Proxy–Proxy controls, 2601.08118;
SimulatorArena's alignment at about 3% of human-evaluation cost). A simulator update that raises task success but lowers
realism is rejected. Capability does not predict fidelity (Arena Elo did not, 2603.11245), and pairwise realism judges
prefer their own family (MirrorBench), so the realism judge differs in family from the simulator.

### 4.4 LLM-simulated Apps and tools: a tier, never ground truth

| Tier | When | Evidence | Gate |
|---|---|---|---|
| Code App (default) | Any write task | AWM: an LLM-simulated environment was below code environments at every size (BFCL 55.52 / 52.53 / 67.68 vs 64.50 / 65.94 / 70.18) | H-gates |
| Cache tier for read-only third-party services | Cache lookup → real call → LLM fallback; every served response written back; frozen before gating | StableToolBench (2403.07714): hit rates 96.5–97% in domain | G2 on the frozen cache |
| Trained API simulator | Read-only, seen APIs | MirrorAPI (2503.20527): agent scores tracked the real environment on 158 fresh queries; unseen-API BLEU stays low (35.6) | U4 |
| Worldbook | The system is gone; only traces remain | Trace2Env (2610.06100): next-observation 75.94 vs 69.51; CR 0.914 | U4; never carries reward |
| Language-only component | Narration, people's wording | Concordia | Typed-effect validation |

**What not to copy.** A policy that writes its own observations (EnvACE, 2608.06197: rehearsal fidelity never measured);
simulators conditioned on the gold answer (GTM, 2512.04535); uncontrolled simulated RL (Qwen-AgentWorld: Toolathlon
32.4 → 31.5, while *controlled* perturbations gave +3.7 Toolathlon and +12.3 MCPMark, 2606.24597); pure synthetic RL in a
learned world model, which trailed real RL by up to 8–9 points (DreamGym, 2511.03773). Stateful tools in a fully simulated
environment trained worse than stateless ones (TRUSTEE: BFCL Multi-Turn 7.9 → 1.0, 2604.17739). Swapping the simulator
moved every agent's average from 67.9% to 29.3% (OccuBench, 2604.10866).

### 4.5 Time

The model and its rules are in §2.4. The cross-cutting rules:
- **Measure bands in the same time mode used in training** (P0 rule for F9).
- **Settle before scoring; group shared-state runs before pass^k.** pass^k assumes independent trials, and shared state
  breaks that: with shared state, run A changed run B in 150 of 200 pairs (2608.14940).
- **Inject mid-run events that change the task state** (OSWorld 2.0: 9.3% of tasks are "dynamic environment",
  2606.29537), and make "verify the asynchronous outcome before acting" a skill leaf: Vending-Bench's dominant failure was
  assuming an order had arrived on its expected date, then spiralling (2502.15840).
- **Never let the harness consume time silently.** Use paused-latency accounting so outages and rate limits do not change
  sim time (Gaia2), and report generation time separately from server errors.

**Build in Crucible.** Declarative `ActorSpec` with registry resolution of the `actor` kind (`core/registry.py:331-338`,
never resolved today), `actors/grounded.py`, `actors/simulated.py`, `actors/fidelity.py`; the clock bus in
`world/clock.py`, `core/triggers.py`, `core/schema/world.py` events. **Priority.** P0 for simulator keying and provenance
and the time-mode rule; P1 for the declarative actors, modules, U-gates and the clock bus; P2 for learned simulators.

---

## 5. Hack resistance and trust boundaries (cross-cutting)

### 5.1 The real-hack negative library

A standing library of executed attacks, needing no task knowledge, runs against every task whose grading touches code,
files or tests. It is seeded from hacks observed in production and frontier runs, not from prompted hacks: monitors trained
on prompted hacks caught only 8.8–40.7% of RL-born hacks, while those trained on training-time hacks reached 86–99.7%
(2604.23488).

| Class | Known instances | Sources |
|---|---|---|
| Test-harness subversion | Always-equal `__eq__` objects; `sys.exit(0)` before asserts; `raise SkipTest` from non-test code; `conftest.py` patching pytest's TestReport; a `python3` replaced by a PASSED printer | 2511.18397; 2503.11926; 2510.20270; 2610.12822; 2610.24206 |
| Test and grader tampering | Editing or deleting tests; stubbing upstream libraries the tests import (a local pandas stub plus 19 stubbed files); editing the module cache; a local GOPROXY serving a fabricated module; forged `go.sum` entries | 2511.18397; 2610.24206 |
| Expected-value extraction | Parsing test files at runtime; decompiling leftover `.pyc` or `.jar`; reading hidden test copies; grader tracebacks returning labels | 2503.11926; 2606.04455 |
| History and network | `git show <target SHA>:file`; staging gold files in /tmp; curling raw upstream files at the target SHA; switching to GitLab when GitHub was refused; disguising a module-proxy request as model-endpoint traffic | 2609.08149; 2610.24206 |
| Memory | With the network off, agents switched to local git (0% success in clean containers) and memory (up to 33.9% success) | 2609.06780 |
| Special-casing | Hard-coding visible instances; returning the error codes tests expect; a 2,900-line lookup table for a compiler | 2605.21384; 2511.21654 |
| State-World analogues | Writing to the DB instead of through tools; tampering with audit tables; renaming another user to impersonate the person who was needed; self-messaging as a colleague | 2412.14161 (impersonation); plan G7 |
| Judge exploits | Template flooding; self-praise text; answer-format block copied without doing the task; listing many candidate facts | 2509.17158; 2606.24597; 2508.14704; 2602.00933 |
| Harness side | A harness that writes FAIL at the step limit earns reward on infeasible tasks | ARPO (2505.16282) |
| Sandbox escalation | Environment enumeration, opening the SQLite file, curl to localhost (1,471 such calls for one model) | ClawsBench (2604.05172) |

**Exploit taxonomy for labels** (RHB, 2605.02964): leakage / metadata, tampering, sequence manipulation, proxy or parser
gaming, special-casing visible checks, denial of evaluation. Precedence: tampering > denial > sequence > leakage > proxy >
special-casing. RHB's deterministic detectors agreed 94% on 50 flagged runs with 0 false negatives on 50 unflagged; rates
are lower bounds.

**Patch whole classes.** With two exploits blocked, the remaining one took 85–91% of hacks (2606.09711). A patch blocks
all open surfaces of an exploit class and re-runs G12.

### 5.2 Honeypots and impossible twins

- **Honeypot twin (G22)**: copy solution, tests or gold state into a watched path, with any read or write flagged (HVTB
  with inotify, 2608.22103). 16.4% of 2,183 runs touched it; reading tests (339) was commoner than reading the solution
  (222); half of first hacks came in the first quarter of the trajectory. Detection is sound but incomplete, and the
  README itself advertises the files, so the rate is a lower bound.
- **Bait at five difficulty levels** (HVE, 2605.20744): named in the prompt → among decoys → nested → encrypted with the
  key elsewhere. Qwen3.6 and Kimi-K2.5 hacked 100% at level 0 and 0% above. After a first hack, models almost always hack
  again, and propensity on one bait type does not predict another, so the suite needs several bait types. Hack rate rises
  with task difficulty.
- **Data bait** (BaitBench, 2608.30724): an optional shortcut planted in the data, graded on a hidden twin without it;
  57.1% ± 2.6 of runs hacked. A shortcut is a hack only relative to a stated intent, so the bait task states the required
  generalisation in the instruction or the user's known info.
- **Impossible twins**: mutate one check so it contradicts the spec (one-off or conflicting); both gold and empty must
  fail. GPT-5 cheated on 54% (conflicting) and 76% (one-off) of SWE-bench twins; an abort action
  `flag_for_human_intervention` cut GPT-5 54% → 9% and o3 49% → 12% (ImpossibleBench, 2510.20270). In a state World the
  twin is an unsatisfiable predicate set and the correct behaviour is flagging the conflict. Twins are the infeasible task
  type, a per-student cheating meter in F12, and vacuous-verifier canaries.
- **Exploited vs detected**: every environment gets two executed post-conditions, `exploited(end_state)` and
  `detected(end_state)`, separating attempts from successful, undetected hacks (2406.10162).

### 5.3 Grader isolation

```text
policy sandbox (World A)                     verifier custody (World B, or the serving layer)
  tools via _call_tool (RBAC, path rules)  -->  declared deliverables only (staged diff, exported state, submit())
  no verifier files, no gold digests             start state, end state, full call ledger    # MegaMCP (2609.36887)
  no network except the model endpoint           checks run here; reason codes out
  read-only task files; unprivileged user        expectation hashes verified at reset and validation
  no compiled reference artefacts, no future VCS objects, hashed instance ids
```

**Isolation evidence, mechanism by mechanism** (BenchShield, 2609.11028): a separate verifier environment removed 82–95%
of exposed vectors; hiding verifier files gave most of the observation effect; read-only task files removed a third of
authority vectors; an unprivileged user about half; seccomp, cap-drop and egress control 2% each; fresh environments per
trial and resource limits nothing; nothing moved fail-closed handling or semantic adequacy. Hardened boundaries in RHB:
−41.5% exploits alone; reduced file access −36.9%; step verification −32.3%; randomised intermediate outputs −20.0%
(2605.02964). Allowlist the network, not blocklist: agents route around blocklists (GitHub refused → GitLab), and with the
network open 32 of 642 runs of one frontier model contacted code hosts and 25 downloaded the task's own repository
(2610.24206). The evaluator reads the same state generation as the agent's last action (2609.32750).

### 5.4 Fail-closed grading

Crash, timeout, malformed output, skipped execution and unknown events all fail closed (I5, 2609.11028), tested by G18.
Verdicts are four-valued (Checked / VectorExposed / AgentViolation / Inconclusive), and each run seals an evidence bundle
(actions, lifecycle events, resource and network manifests, verifier inputs, reward provenance) so a pure-function checker
can relabel archives after a checker change. Runtime verdict accuracy was 96% on 144 cells against 36% for a
transcript-only LLM detector. The median first exploit attempt came at 0.60 of the run and the first success at 0.76, so
neither a start-state scan nor a final-answer check is enough. Grader exceptions become fixed reason codes, because a
verbose traceback leaked 591 labels (2606.04455).

### 5.5 "A chain of thought may raise a flag, never clear one"

| Finding | Source |
|---|---|
| Planted hacks exploited on > 99% of prompts within 30 steps but verbalised on < 2% in 5 of 6 environments; unfaithful CoTs are longer | 2505.05410 |
| With CoT-monitor pressure in the reward, hacking continued while monitor recall fell to near zero; rejection sampling with the monitor accepted more cheating after about 25% of training | 2503.11926 |
| About 32.7% of exploit reasoning recovered by probes never appeared in the CoT | 2606.09711 |
| 28% of RHB exploit episodes had no efficiency rationale at all | 2605.02964 |
| A TF-IDF hack penalty was evaded (on-policy AUC 0.96 → 0.14 by step 240), while static AST and solution-literal features (about 0.985) were not | HackTrace (2610.03055) [PARTIAL] |

**Rules.** CoT content never enters the reward. A CoT monitor may add a flag; only execution (detectors, twins,
honeypots, isomorphic regrades, clean-custody regrades) or a human can clear one. Fixing an environment that a CoT monitor
found is mild selection pressure on the CoT: allowed, and logged. Monitors are judged by within-task AUC on labelled acts
(attempts, including failed ones), not outcomes: act-trained monitors reached 0.962–0.998 against 0.133–0.650 for
outcome-trained ones. If a hack penalty is used, λ must make a detected shortcut score below an honest failure (λ ≥ 2 for a
0/1 reward; at λ ≤ 1 cheating leaked through at 30–39%).

### 5.6 Quarantine

```text
on hack flag (any confirmed source) for task t in family F:
  zero the reward of the flagged rollouts; record HackLabel{act, outcome, category, evidence}
  quarantine t at once; remove t and its descendants from proposer references and swarm memory
  re-gate F (G5, G12, G15, G22 on every task); patch the exploit CLASS, then re-run G12
  keep a robust twin of t as a causal check
  if the hack appeared mid-training: prefer rollback to the recorded pre-onset checkpoint      # 2606.09711
  a hacking student's passing traces never seed traces-to-tasks or the proposer until re-graded strictly   # 2610.12822
alarms: per-step hack rate per family (Wilson bounds), clean vs raw pass divergence, P90 of visible-hidden gap,
        reward vs stronger-oracle gap on a fixed sample, canary probes per checkpoint
```

Why at once: hacking onset is sharp after about 50 quiet steps; above 2% hacked episodes every misalignment evaluation
rose; 50% non-hackable environments did not neutralise it, and filtering hacked episodes then distilling did not undo it
(2511.18397). A monitor with an evolving pattern set cut hack rate 37.76% → 1.31% and raised clean resolved 40.22% →
60.53%; without it, raw pass rate kept rising while clean performance collapsed late (2606.26300). Diversity raises the
stakes: broad misalignment came from hacking across diverse task types, and a 500–1000-task factory is exactly that regime
(2508.17511; 2406.10162). Inoculation prompting is a Part 1, trainer-side lever only (2510.05024, 2510.04340); fixing the
environment beats it, and synthetic-document inoculation before RL made hackers *more* misaligned (2609.14998).

### 5.7 Swarm model-family rules

**Evidence by family.**

| Family | Evidence | Source |
|---|---|---|
| GLM | Most frequent HVTB hacker (glm-5.2: 44.9% in the ablation setting, 25.0% at warning level L0, 4.5% at L3 under the explicit ban); GLM-5.1 read opponent prompts 60% of the time; GLM-5.2 fell 78.80% → 57.32% under leak controls, 90.9% of the PASS→FAIL attributed to removed hacking | 2608.22103; 2605.20744; 2609.08149 |
| DeepSeek | V4 Pro hacked *more* under BaitBench's validity prompt (+8.3 pp) and questioned validity in 1.1% of runs; within the vendor the RL-reasoning model exploited more (V3 0.6% vs R1-Zero 13.9%, same base) | 2608.30724; 2605.02964 |
| Qwen (student family) | Qwen3.6 hacked 100% at bait level 0 and 0% above; Qwen-7B called a hack a hack in a third party's code but called its own use "prevention"; never use a Qwen-family tracer or judge on the Qwen student | 2605.20744; 2610.12822; extract D B8.2 |
| Kimi | Kimi-K2.5 100% at bait level 0, 0% above; kimi-k3 22.7% at HVTB ablation, 0% under warnings | 2605.20744; 2608.22103 |
| Any family judging itself | Same-family self-preference +0.75/7; realism-judge preference flips by family; cross-model output similarity 0.71–0.82 | GAUGE (2609.12191); MirrorBench (2601.08118); Hivemind (2510.22954) |

**Rules.**
1. **Every oracle, check, negative and repair is cross-checked by a different family** from the one that wrote it. The
   writer of a check never writes the solution; the attacker is never the builder's, the suite's or the student's family
   (`factory/models.py` `BUILDER_ROLES` / `INDEPENDENT_ROLES` already raise `SameFamilyError`).
2. **Builders that execute in sandboxes never see verifier files, gate code, anchor labels or honeypot paths.** They will
   take bait in their own verification runs otherwise (extract D B8.2).
3. **Reviewing roles never see a grader's pass/fail verdict.** Showing "ALL TESTS PASSED" raised false accepts in verifier
   agents (2610.12822).
4. **Code, plans and memory from other agents are untrusted input.** A hacking planner induced hacks in every coder
   backbone, and conftest-style hacks spread more than exit hacks because they look like infrastructure (2610.12822).
5. **No Qwen model judges, audits, attacks or traces the Qwen student.** When the 397B becomes the student, no Qwen model
   is its frontier reference either.
6. **Claude never authors tasks** (the lead's rule): Claude may write framework code and this document, never a task,
   oracle, check or report in the pool.

**Build in Crucible.** Path constraints and protected resources at `_call_tool` (`world/world.py:505-610`) and
`factory/pathcheck.py`; `world/custody.py` and `world/verifier.py` (built, unused); `guard/isolate` on every backend, not
only harbor and agentenv (the local backend's quiesce is a no-op today); `credit/hack_detector.py` L1 on the live
`WorkspaceHarness` (map §2.4 item 1); `train/quarantine.py`; red-team tests under `tests/redteam`. **Priority.** P0.

---

## 6. Diversity

### 6.1 Measurement

**Meters on every batch** (post-F6, per difficulty band, per generator family, at a fixed n with bootstrap CIs).

| Meter | Definition | Why |
|---|---|---|
| Cell fill and Gini | Fill fraction of reachable cells; Gini over cells | Coverage and evenness |
| Strength-2 covering | Share of valid value pairs of any two axes that co-occur in an admitted task | Catches template locking that per-axis balance misses (Tev, 2610.18868) |
| Joint slices | 2–3 declared slices (skill × World facet, App × answer kind) | Each factor diverse alone but locked in combinations (ACE, 2608.27260) |
| Behaviour Vendi | Vendi over a behaviour kernel (below) | The diversity that matters to the learner |
| Instruction Vendi | Vendi over instruction embeddings | Surface diversity only |
| Nearest-neighbour tail | Median and p95 nearest-neighbour similarity per round; near-twin rate (cos > 0.99) and whether twins share a World | RST: median 0.223 → 0.464 while entropy stayed flat (2608.05466); AgentMercury: 8.3% near-twins, mostly across Worlds (2608.20634) |
| Unique tool-set share | Share of tasks with a unique tool or API set | AppWorld: 86% (2407.18901) |
| Family homogeneity | Distinct generator families among the top-N nearest neighbours; per-family entropy over cells over time | Hivemind (2510.22954) |
| `distinct_k` per generator | Signature classes among k proposals for one cell | NoveltyBench (2504.05228) |
| Path support | Count per realised capability path; synthesise toward paths below a target depth | Trajectory2Gym: uniform selection of B of N misses a once-represented path with probability 1 − B/N (2610.33568) |
| PATA signature | The vector of clipped, rank-normalised scores of a fixed solver panel | Two tasks that rank the panel identically teach little new (Enhanced POET, 2003.08536) |
| Fidelity | A real-vs-synthetic discriminator, or W2R replay where a real reference exists | Survey (2606.12191) |
| Realism | Expert or customer rating on a sample | CRMArena-Pro: density raised, realism fell 90% → 66.7% (2505.18878) |

**The Vendi recipe** (2210.02410). For n items and a PSD similarity k with k(x, x) = 1, VS = exp(−Σ λᵢ log λᵢ) over the
eigenvalues of K/n. With embeddings X ∈ R^{n×d} and a cosine kernel, use the d × d matrix XᵀX/n (cost O(d²n)). Exact
duplicates leave VS unchanged, which is why it caught an HMM that emitted clusters of exact duplicates that IntDiv ranked
as diverse. Pitfalls: random noise scores high, so pair VS with a validity rate; VS grows with n, so compare at fixed n or
report VS/n; there is no significance test, so bootstrap. **[Proposal]** The behaviour kernel is the average of Jaccard
similarities over tool multisets, tables written and entities touched, plus an RBF over (horizon, virtual time). A falling
VS/n in a growing pool is the near-duplicate alarm.

**Validate the meter before trusting it.** Gradient-space Vendi (G-Vendi: unit-normalised loss gradients under a small
proxy, projected to d = 1024) correlated with OOD transfer at Spearman 0.899 (math) and 0.893 (NLI), against 0.754 for an
embedding Vendi with a 14× bigger encoder; a diverse 10K set often beat a less diverse 100K set OOD; a proxy sharing the
student's base was best (Prismatic, 2505.20161). Persona variety ranked datasets *backwards* there (54.77 vs 38.15 OOD),
and embedding Vendi got that ranking wrong while G-Vendi got it right. **[Proposal, P0 design / P1 run]**: show that
behaviour-signature Vendi rank-correlates with held-out-family transfer on small fixed-size subsets with a small student
before using it as a selection signal (measurement M8, §13.4). For agentic tasks a gradient over a 300-call trace may be
dominated by formatting tokens, so G-Vendi over oracle traces is tested, not assumed.

### 6.2 Structural levers

Structure forces diversity; prompts do not. Pre-generating 6,400 problems with an explicit "span easy to hard"
instruction gave less diverse problems and worse learning than online, feedback-conditioned generation (SQLM, 2508.03682).
Paraphrasing the prompt barely helps (0.821 within-prompt vs 0.781 cross-paraphrase similarity, 2510.22954).

| Lever | Mechanism | Evidence |
|---|---|---|
| Cell first | Sample the target cell (biased to low-fill or low-fitness cells), then mutate a parent toward it; name the empty cell in the prompt | Rainbow Teaming (2402.16822); QDAIF: guided proposals naming an empty bin found cells unguided ones missed (2310.13032) |
| Coverage ledger in the prompt | Covered (policy, outcome) pairs listed to the proposer | GRADUS: 0.75 → 0.54 without it (2610.27781) |
| Nearest kept + failed references | 5 + 5 with reasons | OMNI-EPIC: 16 tasks learned with both; 5 with random examples; 7 without failed ones; 0 without the learnability notion (2405.15568) |
| In-context regeneration | Keep earlier answers in context and ask for a different one | The only prompting fix that reached human-level diversity in NoveltyBench (2504.05228) |
| Verbalized Sampling | k = 5 with probabilities; tail thresholds as a dial | 1.6–2.1× diversity at comparable quality; synthetic math SFT 30.6 direct vs 37.5 VS-Multi (2510.01171) |
| Exogenous grounding | A sampled real passage, World slice, trace cluster, labour-statistics cell or reference document per sub-topic | VHD-Play label entropy 0.997 (2609.27321); OccuBench 16 reference documents per scenario (2604.10866) |
| Islands per generator family | One family cannot dominate the archive | AlphaEvolve (2506.13131) |
| Soft re-sampling within a level | Re-sample knobs independently inside the current difficulty level (ε = 0.5) | TRUSTEE: without it 46.4 → 44.5 (2604.17739) |
| Coverage pressure for dense Worlds | A per-app usage histogram fed to the day planner | WildAWM (2610.26828) |
| Separate structural from surface | Knob class `structural` vs `stylistic`; stylistic variants are augmentation inside one task, not pool entries | Reasoning Gym; GRADUS kernels; AutoEnv BaseEnv / ObsEnv / SkinEnv |
| Composition across Apps | Cross-App as a grid axis | CompoWorld (Qwen3.6-35B-A3B, 1K SFT trajectories): τ³-Banking 9.62 → 17.53, AutomationBench 12.83 → 33.33 (2609.33665) |
| Break the CRUD lattice | Vary the tool surface beyond search / list / get / update | AgentMercury's task shape was a lattice set by one variable (2608.20634) |

**Personas buy realism; the cell and the World slice buy diversity.** Persona-induced diversity shrinks under topic
constraints (Persona Hub: generated problems at persona similarity 0.9 sat mostly at 0.6–0.75, 2406.20094), and Prismatic
ranked persona variety backwards. **Mixing model families is not a diversity mechanism**: cross-family similarity was
0.71–0.82 and about 8 distinct models sat in a typical top-50 similarity cluster (2510.22954). Raw sampling diversity falls
with scale and post-training (NoveltyBench: Llama-3.2-1B 6.74 distinct of 10 vs Llama-3.1-405B 3.20), but strong models
respond better to structural prompting (VS gains 1.5–2× more on larger models), so the swarm measures `distinct_k` per
model under the F3 prompt actually used. A smaller model may be the better breadth proposer and a larger one the
verifier.

### 6.3 Quotas, families and instances

**Families over instances, environments over queries.**

| Evidence | Number |
|---|---|
| Terminal-Universe (2609.04148), fixed ~35K records | 2× environments 56.0 vs 2× queries per environment 53.8 vs 2× solutions per query 53.9 (base 53.2) |
| ScaleEnv (2602.06820), fixed 1,024 tasks | pass@4 rises monotonically over 2 / 4 / 8 / 16 domains (single run, uneven task counts) |
| AWM (2602.10090) | 10 environments degrade below base; 100 help; 526 help more |
| Interlock (2610.19686), fixed 8,957 trajectories | 16 → 128 domains monotone |
| SWE-smith (2504.21798) | Roughly log-linear over 4 / 25 / 50 / 100 repos at 700 trajectories |
| OpenThoughts-Agent (2606.24855) | Rollouts per task plateau 31.6K → 100K: task-description diversity is the bottleneck |
| Counter-evidence: SynthAPI (2610.37648), SFT | At 64–512 tasks, A = 8 Apps is best; from 4,096 tasks A = 64 wins; the 27B gained least |
| Counter-evidence: SWE-Gym (2412.21139) | "Sampling compute, not repo diversity" was its bottleneck, bounded by 11 repos |

**[Proposal]** A family is (domain, World template, skill leaf). Revisit `factory/family.py`'s 20-task, 3-surface rule:
the shape to test for RL is **on the order of 50–100 families with 10–20 instances each and S = 2–4 seeds per instance**.
This is a target, not a measurement. M3 (§13.4) measures it per student, with SynthAPI's protocol: fit on low-cost runs,
extrapolate, normalised score; its law P̂ = σ(c₀ + c_A log₂(A·S·D)) · σ(b₀ + b_T log₂ T)^ν had extrapolation MAE 0.0285
against 0.1706 for a task-count-only law.

**Quotas** per cell, family, World, template, source and generator family. RST's caps: ≤ 4 descendants per parent,
≤ 160 per category, ≤ 320 per rewrite family, relaxed only on a logged schedule; no seed exceeded 0.77% of a round
(2608.05466). SWE-Bench Pro capped 100 instances per repo (2509.16941). "Each bug used once" (SWE-smith). WildAWM's tasks
skewed to its first six personas (178 of 341), which a quota would have caught.

**Source quality ahead of source count.** At 10K trajectories, top-4 sources (+0.49 z) ≈ top-2 > top-8 > top-16 (−0.10)
> top-32 (−0.48) ≈ top-1 (−0.57); at 100K (32B), top-16 was below both top-8 and top-4 on all three benchmarks
(OpenThoughts-Agent, 2606.24855). OpenThoughts (code) found at most two sources won in every domain at 31.6K (2506.04178)
[PARTIAL]. F12 ranks sources and generators by downstream value per dollar and drops the tail; the scaling unit is
distinct task descriptions per family, not rollouts.

### 6.4 The archive

The archive (§3.12) is a MAP-Elites grid with islands per generator family. Stepping stones matter: DGM's best lineage
passed through two score dips (2505.22954), and ADAS's best agent combined ideas from iterations that scored poorly at the
time (2408.08435). Novelty decays even with "be novel" prompts (STOP, 2310.02304), so diversity is injected structurally.
Quality-only search in QDAIF was no better than fixed prompts, and human quality decoupled from AI fitness above 0.995
(2310.13032), so top elites get a human spot check.

**Build in Crucible.** `factory/archive.py` (new) for the grid, meters and quotas; `observe/factory_views.py` prints the
meters per batch; `factory/ortho.py` and `forge/dedup.py` supply dedup. **Priority.** P0 for cell fill, the
nearest-neighbour tail, quotas and family holdout; P1 for Vendi, covering arrays, `distinct_k` and the meter validation.

---

## 7. Difficulty per student size

### 7.1 Bands and the corrected theory

**Why bands exist.** With a binary reward and group-relative advantages, an all-pass or all-fail group carries no
gradient; at k = 8 the all-same probability is p⁸ + (1 − p)⁸ (2402.03300).

**The corrected theory.** `framework-plan.md` §3.6 said expected improvement is bounded *below* by p(1 − p). That is not
what the papers prove:
- **2504.03380** bounds the reverse KL from the initial policy to the soft-optimal one: KL ≥ p(1 − p)/(2β²) for large β,
  0 at p ∈ {0, 1}. It bounds distance to the optimum, not per-step improvement.
- **LILO (2502.12272)** gives an *upper* bound: per-step improvement ≈ β‖∇J‖² ≤ E‖Σ∇log π‖²·Var[r]. Low learnability caps
  improvement; high learnability does not guarantee it. Under GRPO's std normalisation the dependence cancels, and
  unnormalised GRPO worked better in replications.
- **SEC (2505.14970)**: under std-normalised binary reward E|Â| = 2√(p(1 − p)), maximal at p = 0.5.
- For multi-component grades, use the variance of the scalar reward, not one component's pass rate (2504.03380,
  Cor. 6.3).

So: **"the KL distance to the optimal policy is bounded below by p(1 − p)/2β², and per-step improvement is bounded above
by a term proportional to Var[r]."** The normalisation choice in Part 1's trainer changes how strongly the band matters
(Crucible uses Dr.GRPO centring with no std, `train/credit.py:88`).

**Band results.**

| Source | Rule | Setting |
|---|---|---|
| 2504.03380 | (0.3, 0.7) 30.1; (0.2, 0.8) 29.8; (0.4, 0.6) 29.9; (0.1, 0.9) 28.4; plain GRPO 26.3; **one-sided cuts 23.8–25.9, all below plain GRPO** | Qwen2.5-3B, five math benchmarks; differences among 0.2–0.4 within noise |
| AutoEnvScaling (2610.14163) | Mean ∈ [0.25, 0.75] and std ≥ 0.1, adaptive k 4 → 8 | Terminal tasks |
| RODS (2606.19047) | Mastered > 0.85, boundary 0.20–0.85, hard < 0.20; rank by 4r̄(1 − r̄) | K = 16, dense reward |
| SwS (2506.08989) | 25–75% of 8, dropping 1/8 and 7/8; easy windows fastest then saturate, hard windows longest-lasting | Math |
| SPICE (2510.24684) | Gaussian in variance, peak at p = 0.5, ranked first among proposer rewards | G = 8 |
| SSR (2512.18552) | Its injector reward's optimum p* ≈ 0.2 at G = 8 (a game-analysis optimum, not a measured band) | SWE self-play |
| WebRL (2411.02337) | 0.05–0.75 on a trained critic's value estimate, not a pass rate | Web |
| AdaRFT (2504.05520) | Target β = 0.5 beat 0.2 and 0.8 | Math |

**[Proposal] Resolution of the band-centre conflict** (SSR 0.2 vs SPICE 0.5 vs 2504.03380's interior bands). Keep the
current admission band [0.1, 0.6] for the 35B. It is not refuted: the one-sided rows that lost were cuts at 0 or 1, not a
shifted interior band. Add a symmetric (0.2, 0.8) arm to R2 and measure learning value per task against band position
(M2). Inside the band, rank by the Gaussian-variance utility. Keep the benchmark at ≤ 0.5.

### 7.2 Controllers that need no extra rollouts

| Controller | Rule | Evidence |
|---|---|---|
| Training-rollout reuse | p̂ from GRPO groups; refresh every step; never freeze an in-band list | RODS; LILO: buffers refreshed every 50/10/5 steps produced sawtooth overfitting; only every-step refresh generalised |
| Set-point | T ← clip(T + η·tanh(α(R_avg − β)), d_min, d_max), β = 0.5, α = 2, η = 50 on a 0–100 scale; batch = the B items nearest T | AdaRFT: PPO needed 71.7% more steps to match on skew-difficult data |
| Dead-band scalar | D ∈ [1, 100]; every knob k = ⌊c_k·D⌋; D += 3 if batch reward > 0.5, −= 3 if < 0.0; ε = 0.5 within-level re-sampling | TRUSTEE: no curriculum 41.5 vs 46.4; δ = 10 → 38.8 (2604.17739) |
| Bandit over cells | Boltzmann p(c) ∝ exp(Q(c)/τ), Q(c) ← α·mean\|Â\|_c + (1 − α)Q(c) | SEC: random 0.479, ordered 0.321, SEC 0.542 on OOD Countdown |
| UCB over distributions | UCB(d) = L̂(d) + √(2 log(n + 1)/(n_d + 1)) with a per-family floor | DUMP starved some distributions without a floor (2504.09710) |
| SETA policy | r > 0.5 → increase along a *different* dimension; 0 < r ≤ 0.5 → context shift at equal structure; r = 0 with partial passes → decrease | SETA (2607.10891) |
| Zones | Mastered → harder variant from a success; learning → refine the specific failed attempt; too hard → park | Socratic-Zero (2509.24726) |

**Couple the controller to G12.** Hack rates rise at the hard edge: in HVE, hack rate rose monotonically with task
difficulty (2605.20744); IPT shortcuts rose from 40 at levels 1–10 to 458 at 11–20 (2604.15149); SpecBench's
visible–hidden gap grew about 27 pp per 10× reference size (2605.21384). Every variant above band, on a ladder or hardened
by the controller gets a fresh G12, honeypot and composition-check pass before admission.

**Watch the controller's entropy.** A capable challenger can hold s at p* by tuning one parameter, after which learning
stops (SSR "tunnel vision"). A knob-entropy monitor on every controller and injector is the alarm.

### 7.3 Structural knobs with measured effects

| Knob | Mechanism | Measured effect | Monotone? |
|---|---|---|---|
| count / horizon | Gold steps, subtasks, chain length | EnterpriseOps-Gym: ~35% at 4 steps → < 20% at 16 (2603.13594); WorkBench 0 / ≥ 1 / ≥ 2 actions → 75 / 36 / 18% (2405.00823); AgentSynth levels 18% → 4% | Yes in these |
| apps | Number of Apps | OfficeBench 64.5 → 60.0 → 21.4 (2407.19056) | Yes |
| linkage | Answer-preserving entity → description substitution, n hops; layer-wise leaf expansion | e2h (2505.22648); WebShaper (2507.15061) | Declared, measure |
| inherited access transforms | Sharding, pagination, field references, re-encoding, archive views | RES: independent single transforms 93.3% vs inherited 6.7% (2610.16397); check access vs skill with budget-2× | Measure |
| observability / disclosure | Explicit steps vs ticket + KB policy; shard count | WorkArena++ L2 vs L3 (humans 93.9 both; models ~0); full → sharded −39% (2505.06120) | Yes |
| distractors / tool exposure | Distractor density; oracle subset vs curated vs full surface | WorkBench 43% all tools vs 49% needed only; EnterpriseOps-Gym no loss with +5/+10/+15 distractors; CRMArena helper tools hurt weak models | **Model-dependent; measure per student** |
| noise / faults | Explicit (500, timeout), implicit (truncation, stale values), mixed | OccuBench mean 67.5 → 62.6 → 53.4 → 54.4 (2604.10866); Gaia2 noise none 31.2, low 35.0, medium 23.8, high 8.1 | **Not monotone at low levels** |
| hints | Gold plan, similar demo, exact demo, none | EnterpriseOps-Gym: human plans +14–35 points; TimeArena self-plan prompting cut GPT-4 40 → 0% | **Not monotone** |
| user_behaviour | Collaborative … tangential | Qwen3-30B-A3B MultiWOZ 48.3 collaborative → 27.2 tangential, 26.1 incomplete (2509.23124) | Per mode |
| time_mode, pressure, latency, contention | Instant vs real latency; fees, start capital; one pot / fryer | Gaia2; Vending-Bench; TimeArena | Measure |
| instruction_form | Real-query templates | −20–40% relative (2510.08996) | Measure |
| persona_stratum | Six strata | −42% reward (2609.12191) | Measure |

**Hardness is often mislabelled.** Declare `monotone: true` only after a 3–5-setting ladder measured on the frontier
audit sample, then re-measure per student. Reject a "harder" variant whose signature does not change on the knob moved.
WebForge's seven-dimension vector was monotone L1 → L3 for every model, but dimensions were correlated by construction
(mean |ρ| 0.495), so a declared vector is checked against the measured signature (2604.10988).

### 7.4 The sandwich and regret

- **ZPD sandwich** (AgentFrontier, 2510.24695): escalate a task until a weak base configuration fails, then require a
  strong tool-using agent to pass at least once in 3; 0/3 goes to human review. About $0.78 per kept item; two-thirds of
  refinement spend landed in review or discard. **Caution:** defined against a no-tool base, so most of its gap is tool
  access, not skill.
- **Paired band** (ACE, 2608.27260): a task needs the skill when p_{z0}(d) < ρ ≤ p_{zA}(d), the base configuration fails
  and the assisted one succeeds, where z is model + scaffold + tools + verifier + sampling + budget.
- **Regret** (PAIRED, 2012.02096): regret ≈ max antagonist utility − mean protagonist utility, clipped at 0. In Crucible,
  `regret = ref_pass − student_pass` with the oracle giving 1. Unsolvable tasks give zero regret automatically. High
  regret with an in-band p̂ goes to the pool; frontier-also-fails is a suspect. Use regret as a bandit signal over knob
  vectors, not to RL-train a generator. The antagonist must stay stronger than the student, or regret collapses.
- **ACCEL editing** (2203.01302): parent = an in-pool task whose p̂ rose above band; edit = one knob change; the child goes
  through F4–F7 and enters only if in band for the target student and its signature differs on the changed knob; lineage
  fields and a lineage cap per cell supply the missing diversity control.

### 7.5 Ladders for too-hard tasks

| Ladder | Rule | Evidence |
|---|---|---|
| Late starts | Start at the oracle's state after its first k actions, or at the decisive-step snapshot (§3.13); **mix all start positions**, equal reward per start, keep the earliest starts | R³: staged progression overfits; difficulty-scaled rewards lowered 63.8 → 58.7–60.0 (2402.05808) |
| Partial-solution hints | Prepend the first p% of the oracle plan; Partial-50 → Partial-25 → none; calibrate per family with a pass-rate histogram | QuestA: steep and narrow (Partial-50 made 17,741 items 8/8, Partial-25 looked almost like no hint) (2507.13266) |
| Hint-giving errors | `error_informativeness = hinting`, then off | Environment Tuning: augmentation off still added 4.9 (2510.10197) [contaminated: selection on held-in test] |
| Check-subset siblings | Proper subsets of grouped milestones containing a large group become strictly easier siblings with the same World and oracle prefix | WebGym (2601.02439) |
| Injected positive | The executed oracle trajectory as the positive for an all-fail group | ARPO's cached success (2505.16282); DART pool lifted 22 tasks from 0% to 46% |
| Feedback enrichment | Guidance early, enrichment late; the **same enrichment decision for every rollout in a group** | FEE: +2.82 pp average; per-rollout random enrichment was unstable; clarification fell 4–7 pp (2609.08404) |
| SFT warm start | Oracle or teacher traces for 0% cells | WildAWM 35B without SFT −5.31; UserRL raw RL plateaued |

Variants share the parent's `VerifierBundle` and record `hint_level` or `start_offset` and `parent_id`. The unscaffolded
p̂ is the real metric; a scaffolded variant is its own `DifficultyRecord` row. Hints can remove the skill: FEE-trained
agents asked for clarification less, so infeasible and clarify tasks are measured before and after scaffolded training.
Mixed clean and perturbed groups need advantages normalised within each subgroup: noise on every rollout (13.25) was worse
than no noise (16.00), and the subgroup schedule reached 21.50 (NoisyAgent, 2605.27209) [contaminated].

### 7.6 IRT

Fit 2PL P(Y = 1) = σ(a_j(θ_i − b_j)) over the (task × student / checkpoint / frontier) pass matrix with priors and one
dimension per skill family. b_j relative to each student's θ is the band, which unifies "in band for the 35B, too easy for
the 397B". Run the most informative tasks for a new checkpoint first (Fisher information a_j²p(1 − p)) and stop when
SE(θ̂) falls below the gap between rank-adjacent models; Fluid needed about 20 items early in training and more than 80
mid-training (2509.11106). Low-discrimination items are often broken: mislabeled items per 100-item session were 0.01
adaptive vs 0.75 random. Caveat: Crucible's matrix is tiny, and items no model solves collapse to one maximal difficulty,
so a frontier saturates the IRT's resolution. Exclude near-duplicate model families when fitting item parameters
(2509.11106) and validate on by-date splits (tinyBenchmarks, 2402.14992).

### 7.7 What changes for a 397B student

| Change | Reason | Source |
|---|---|---|
| Much of the 35B pool will be above band | Pool value decays with size: SynthAPI's 27B gained least; R2E-Gym's 14B saturated near 800 trajectories while the 32B kept improving | 2610.37648; 2504.07164 |
| The regret reference must be stronger than the 397B | Oracle plus a stronger non-Qwen frontier, or regret saturates | PAIRED |
| The 35B becomes the weak reference in the sandwich | Re-choose the weak reference per student | AgentFrontier |
| Warm-start the predictor from the 35B's outcomes with a student-id feature | MAGELLAN-style predictors need thousands of outcomes | 2502.07709 |
| Re-run G12 and answer-check false-negative audits | Rule-check recall falls as the policy gets stronger; "policy size does not change exploitation" was shown only at 1.2B vs 6B | 2505.22203; 2210.10760 |
| Judges must be stronger than the generator for their checks to inform | The generation–verification gap falls with generator capacity | Mind the Gap (2412.02674) |
| Measure pass@k, not only pass@1; drop "diminish" families | ProRL regimes | 2505.24864 |
| Re-derive lucky-pass references per student | Process norms shift | AgentLens (2605.12925) |
| Never compare students under different output-token caps | The cap moved one model fifth → first | 2608.12635 |
| Expect no gain beyond the teacher's frontier | WildAWM's hardest quartile | 2610.26828 |
| Budget | 397B rollout $3.01 vs the 35B's $1.72 at 300 calls | map §4.4 |

No paper reports a band schedule per student size (extracts B, C), so the 35B-vs-397B band is our own measurement (M9).

**Build in Crucible.** `factory/difficulty.py` (`difficulty.control`, the R-Zero-style `1 − 2|p − ½|` reward) becomes the
controller over DomainSpec knobs; `factory/difficulty.py:237-296` checkpoint-start variants are the late-start ladder;
`bench/irt.py` for IRT; `factory/calibration.py` for exemplars in the builder brief. **Priority.** P0 for keying, sequential
k, the G12 coupling and late starts; P1 for controllers, IRT, ladders and the 397B items.

---

## 8. Traces to tasks

**The rule.** Trace → World snapshot → task, never trace → text → task. Re-solving recovered intents gave 52.1 against
36.7 for SFT on the source trajectories, which was below the 47.0 base (Terminal-Universe, 2609.04148). OpenCUA kept
traces as SFT data and lacks the rebuild-and-re-solve step (2508.09123) [PARTIAL]. Instruction backtranslation shows the
value sits in the curation gate, not the generator: uncurated inferred data did not improve however much was added
(2308.06259).

### 8.1 Route A: the system can be rebuilt

```text
traces_to_tasks_A(trace):                                       # Terminal-Universe (2609.04148)
  0. redact at source (8.3); drop traces from benchmark corpora, view-only traces, unsupported formats;
     dedup by (system, base state, request)
  1. rebuild(trace) -> SeedBundle + withheld_changes
       state Worlds: the pre-write row version of every entity the trace read or wrote
       files: the EARLIEST observed content of every touched path; agent-created files excluded;
       the agent's changes recorded separately and withheld
       task-time snapshot: only what the task owner could see at assignment time,
       excluding later solutions and evaluator-only data                        # OfficeTown (2610.23194)
  2. complete(SeedBundle, intent) under "solvable, not solved":
       create missing records and files, configure dependencies;
       never implement the target, never add its tests, no comments locating the solution
  3. G20 sufficiency(SeedBundle, intent) -> {label, confidence, present, missing_critical}
  4. recover_intent(trace): 2-4 sentences from the FIRST substantive user request, with success criteria,
       constraints and prohibitions; user-stated requirements only, never agent-only evidence
     a signal of non-self-containment: the original agent itself asked for clarification   # 2606.23654
  5. decision grounding: an action is a target only if it executed without error and its id-like
     arguments trace to earlier visible information (v_t); rejected calls stay as context  # 2610.33568
  6. then the normal F4-F10: oracle by construction, red-on-start verifier written before any solution,
     every gate, re-solve by a different family
  outputs per trace: 0..n tasks, plus late-start ladder rungs at the first-error snapshot      # Agent-Reflex
```

**Funnel evidence.** Terminal-Universe: 359,593 traces → 68,263 reconstructed → **37,273 sufficient environments**;
sufficiency rose from 40.2% after replay to 93.5% after completion (terminal) and 20.1% → 77.1% (SWE); completion grew the
median workspace from 2 to 13 files; a manual check of 30 completions found 22 with only task-relevant support, 8 with
unnecessary code, 0 containing a solution. EnterpriseClawBench mined about 3 months of an internal agent used by more than
100 employees: 5,291 candidates → **852 tasks (16%)**, with gates for fixture resolvability, redaction recovery under a
unique-match rule, external network dependency, self-containment, deliverable rules and a sandbox preflight
(2606.23654). Decision grounding: in 34,772 candidates, 40.9% had ≤ 2 supervised decisions (Trajectory2Gym, 2610.33568).

**Relabelling sub-trajectories.** Every contiguous sub-trajectory T[i:j] can become a candidate (Learn-by-interact, up to
+14.0 in training [contaminated]); each candidate becomes start = snapshot at i, goal = state diff at j, then F4–F6. When
relabelling, strip the chain of thought (it was conditioned on the original, perhaps failed, goal), extract a structured
change summary of successful state-changing operations, and minimally edit the task toward what was achieved: keep the
task type, delete rather than invent, drop strong quantifiers ("all", exact counts) unless supported (AgentBrew,
2609.05837). Short sub-goals feed the single-subtask probe and ladders.

### 8.2 Route B: the system is gone (the worldbook)

When only traces remain, build a validated worldbook (Trace2Env, 2610.06100) and use it as a **specification source for
F1**, not as a reward-bearing simulator:
1. Action schema (typed arguments, canonicalised) and state schema with typed, mutability-tagged variables in four
   groups: world (durable), session, surface (visible now), epistemic (uncertain or conflicting beliefs).
2. One grounded evidence record per aligned transition; effects with ambiguous attribution or concurrent actions withheld.
3. Induced rules, constraints and observation contracts, proposed and reviewed against success–failure contrasts;
   only supported rules with confidence ≥ 0.55 compile to executable (few compile: Terminal 3 of 12).
4. Compile: schemas → `AppSpec` tables and tools; supported rules → `policies_as_code`; evidence turns → tool-test
   fixtures. Residual LLM-simulated behaviour stays a lower tier that never carries reward.
5. Cross-fit construction by task-grouped folds when evaluating on the same traces: construction-disjoint gains persist
   but are much smaller than same-task overlap (+17.86 on Terminal).

Its cost was $0.03–0.21 per record. A trace-driven App per task (GUI-Genesis, 2602.14093) is the CUA-lane variant; failed
traces still carry look-and-feel and page flow.

### 8.3 Enterprise privacy and redaction

The gate order before any model sees a customer trace or seed:

| Step | Mechanism | Evidence |
|---|---|---|
| 1 Local review | Nothing leaves the machine without manual upload; a classifier flags sensitivity (it over-flags); humans decide | OpenCUA (2508.09123) |
| 2 Redact | NER per turn (Presidio) plus credential scanning (TruffleHog) before any model sees a log | SWE-chat (2604.20779) |
| 3 Summarise and cluster | PII-free summaries → clusters with a minimum count of unique accounts → PII-free cluster summaries → an auditor removes clusters scoring < 3 on a 1–5 identifiability scale | Clio (2412.13678): raw private info about 10% → 1.5% after summaries → none ≤ 3 after cluster summaries; auditor 98% accurate |
| 4 Weight, do not copy | Clusters add or weight `DomainSpec` cells (coverage, not volume); real people map to World personas consistently | Clio; plan §3.4 |
| 5 Tabular seeds | Merge categories seen fewer than about 8 times; DCR-CDF integral up to the 0.98 point ≤ 0 (a DCR *mean* hides overfitting); shadow-model membership inference on the most vulnerable records near AUC 0.5 | 2508.06647: without protection best attack AUC 0.635; with value protection about 0.501 |
| 6 Text seeds | DP synthetic text (Aug-PE votes, or DP-SGD at ε = 4); plant rare edge cases synthetically, since DP erases anything rarer than about 0.05% of records | Aug-PE (2403.01749); 2210.14348 |
| 7 Canary audit | Inject fake-PII canaries at 1, 10, 100 repetitions; require 0 verbatim hits and no top perplexity rank; at 100 repetitions 80% leaked verbatim without DP and 0% with DP | 2210.14348 |
| 8 Cap reuse | Cap how many tasks one customer record can seed; per-person privacy units when one person writes many records | 2012.07805 |
| 9 Probe the student after training | Canary and memorisation probes, more so for the 397B: larger models memorise more, and strings in exactly one document were extracted | 2012.07805 |

Raw customer data never enters a training World. Abandoned sessions are not logged, so success rates from logs are biased
upward: if 10 / 20 / 30% of sessions were abandoned unlogged, SWE-chat's mean success would fall from 73.2 to 65.9 /
58.6 / 51.2 (2604.20779). Holistic session-success labels had human–human ICC(2,1) of only 0.503, so they are never a
reward or an anchor label.

### 8.4 Measuring the route by replay

| Measure | Definition | Bar |
|---|---|---|
| Re-solve rate | Share of rebuilt tasks a different-family agent solves from the rebuilt start | Report per source; Terminal-Universe teacher pass@1 72.3% (single workspace) and 49.2% (cross-workspace) |
| Sufficiency rate | G20 pass share after completion | Report |
| Consistency ratio | For worldbook-derived tiers: successes replayed in the real system / successes in the tier | ≥ about 0.9 to start (Trace2Env 0.914; direct prompting 0.032) |
| Intent agreement | Agreement between the recovered intent and the original user's request, on a human-labelled sample | κ with intervals |
| Leak rate | Share of recovered intents that contain agent-only evidence | 0 target; lint |
| Distribution match | Embedding overlap between generated instructions and the real query cloud | 2510.08996 |

**Build in Crucible.** `connectors/seed.py` and `connectors/redact.py` (new) behind the `Connector` protocol
(`connectors/base.py`: discover, fetch, normalise, redact); connectors today produce traces but never seed App state. The
toy G-hindsight generator (`packs/_toy/.../forge/hindsight.py`) is the only precedent; `factory/failures.py` turns failing
student traces into `BuilderIntent`s. **Priority.** P0 for the rebuild-and-re-solve design, red-on-start and the
user-stated-only intent rule; P1 for the full pipeline on synthetic enterprise logs; P2 for customer data and the
worldbook.

---

## 9. The self-improvement loops and their guards

### 9.1 The four loops

| Loop | Signal | Proposer (family rule) | May change | Accepted when | Rollback |
|---|---|---|---|---|---|
| **Verifier** | Hack flags; attack passes; false-fail disputes; lucky passes; side-effect survivors; pass matrices where rollouts with identical check vectors have different end states (a missing check) | Different-family auditor and attacker | New negatives, checks, path-rule patterns, test augmentations; *adds* checks, does not only re-tune | Gold sanity; anchor agreement holds or improves (J1); canaries rejected; held-out mutants still killed; birth gate (fires on ≥ half its cluster, stays clean on known-good outputs) | Versioned rules (`RULES_VERSION`), repath of stored episodes |
| **World** | `world_bug` and `env_flake` labels; G13 crashes; simulator critical errors; validation-gate fallbacks; density and realism health | World expert | Tool code, seed data, simulator profiles, clock settings | Blind tool tests still pass against **frozen** expectations; every task in the family re-gates; replay-agent drift check on old action sequences; real-vs-sim parity and people realism hold | World versions by spec hash |
| **Task** | p̂ out of band; INCONCLUSIVE intent; executed failure attributions; empty cells | Difficulty controller; task proposer | Knob settings, ladders, variants, targeted tasks | Full F6–F10; not a duplicate; signature differs on the knob changed; an **equal-budget untargeted batch** is the control | Parent kept; variant linked |
| **Generator** | Value per dollar per generator; defect rate per generator; cell entropy; invalid/total per segment | Pipeline doctor | Prompts, recipes, source weights, quotas | One-stage-at-a-time ablation on a frozen slice with a fixed small student (§13.5), plus held-out family meta-evaluation | Recipe versions |

**Evidence for equal-budget controls and for the targeted loops.** Agent-World's arena loop raised a 14B over two rounds
but had no equal-budget random control, and its diagnoser saw validator internals (2604.18292) [no-control]. In
AutoEnvScaling, solver trajectories in the assignment were worth 3.6–6.7 points at matched steps (2610.14163). SwS's
weakness-driven retrieval beat random retrieval (2506.08989). WEFT's three attribution-routed repair rounds cut the
tool-error rate in selected trajectories 1.76% → 0.96% and added +3.65 to +5.25 pp at fixed tasks and budget
(2609.36887), with attribution precision unmeasured.

**A shared proposer must keep learning, or the loop collapses.** With solver-only updates, AutoEnvScaling's proposer
valid rate fell 86.8% → 22.4% by step 160 and training collapsed (2610.14163). This is why in-training proposer loops
stay P2, and the swarm proposer improves by harness optimisation (memory, guideline banks), not weight training.

### 9.2 The guards

| Guard | Mechanism | Evidence |
|---|---|---|
| **Evaluator out of reach, by a proxy** | Gate code, measurement markers, anchor labels, error messages, model routing, budgets and the loop driver sit outside the improver's read *and* write scope, enforced at a proxy and in the runner; graders return reason codes; budgets enforced in the callee | AEvo: without its protected-evaluator harness, 2 of 3 runs hacked the reward (2605.13821); DGM node 114 deleted the marker logging its detector used and scored a perfect 2.0, more often when detector functions were visible (2505.22954); STOP improvers disabled the sandbox flag in 0.42% of 10,000 generations, and a warning did not reduce it (2310.02304); an unconstrained Gödel Agent chose to call stronger models (2410.04444) |
| **Frozen anchor sets** | Per judge type, 200–500 adjudicated items, per lane *and per check type* (pass, hack, false fail, lucky pass, on-intent, side effect); hack positives from real rollouts, attacker runs and impossible twins, never prompted hacks; failed hack attempts included | Trust or Escalate: 30–50 episodes are too few for the exact bound (2407.18370); ARE's 450 labelled trajectories; 2604.23488 |
| **Dev anchors for verifier evolution** | A frozen, author-disjoint ~10-item anchor per family (5 failures + 5 clean), the only supervised signal for selecting an evolved verifier; fail-closed selection; unanchored detectors in a shadow tier | Without anchor guards the evolved verifier collapsed to always-pass on 3 of 3 seeds, while held-out task scores stayed equal or better (MBPP+ 0.742 vs 0.717) (2610.11464) |
| **A locked set no loop reads** | Held-out families and a locked test set; reading it by any loop (including rollback logic) is a violation | 2610.11464's skill loop read its locked set during rollback |
| **Canaries** | Always-pass, always-fail and constant-Tie verifiers; master keys; a narrated-completion trajectory; blank and shuffled traces; screen every metric for predicates that cannot fail; constant-policy baselines (always-positive F1 = 2π/(1+π)) | 2610.11464; 2609.12439; 2507.08794; LiveMCPBench; 2608.03744 withdrew five arms whose predicates could not fail; 2607.28685: R-Judge's constant 0.690 outranked five models |
| **Equal-budget controls** | Every targeted batch against an untargeted one at equal spend | Agent-World gap |
| **Round caps for judge-only loops** | A loop whose acceptance rests on model verification stops after about 2–3 rounds unless new executed information enters (new negatives, new traces); monitor large-k pass@k and signature diversity across rounds | Mind the Gap: iterative self-improvement saturated to about 0 gap in 2–3 rounds at every size; large-k pass@k fell; a gold verifier showed no diversity loss (2412.02674) |
| **Never select on visible score, learned reward, student reward or CoT** | Selection uses anchors and executed gates only; student reward is reported, never optimised by the loop | SpecBench's best-by-visible node was a lookup table (2605.21384); AgentPRM: true success peaked at 82% then fell to 70% while PRM reward rose (2502.10325); 2610.11464 |
| **Non-compensatory acceptance** | A noise floor δ measured beforehand from repeated runs of the unchanged system; accept only if Ŝ ≥ S* − δ with S* best-ever; cost-aware acceptance; a leakage critic rejects diffs encoding task names, entities or values; an annealed edit budget (late rounds: one attributable edit); a ledger of every evaluated edit, rejected ones kept | RRSI: full regularisation OOD 43.6 at 2.42M tokens vs unregularised 40.3 at 3.80M; higher evolve score went with lower transfer in every arm (2609.24972) |
| **Recoverability and σ-gating** | Unattended application only if undo is a plain write of a stored old value *and* the approving judge's strength σ = MCC(v, v*) was measured above σ_min; otherwise refuse and log; blast radius on the replay pool | 2610.27465: a production diversity gate returned 0.250 in all 40 runs; regenerated undos left behaviour 0.16 worse; even a perfect verifier overstated the gain by 0.20 (winner's curse) |
| **Repairs bounded** | ≤ 2–3 rounds, a file allowlist, no weakening of semantic checks, no removal of shortcut protection, no exposure of private requirements; caps relaxed on a logged schedule | RST (2608.05466); WildAWM |
| **Multi-view evaluation** | Update evidence; a frozen validation view; held-out ID; held-out OOD; replay of earlier tasks, with forgetting FR = max(0, Perf(A₀, replay) − Perf(A_T, replay)) and fix/forget churn | SEAGym: AHE replay 34/80 → 43/80 finally but 6/80 after epoch 4 (2606.17546) |
| **Masking alarm** | Success rising while the target behaviour (questions asked, reads before writes) falls | EAU: a lower volunteer interval raised success while questions fell (2610.16523) |
| **Unintelligible winners** | An evolved prompt that drifts into word salad, or a near-instant perfect score, is a review alarm, not headroom | Promptbreeder's "SOLUTION" (2309.16797); Gödel Agent MGSM prompts |
| **Different-family diagnoser that sees verdicts and short reasons only** | Its output goes through F6 | DGM's diagnostician saw private test patches |

**What may auto-apply first** (each behind every guard): leak-probe and changelog-strip patterns, path-rule patterns with
versioning and repath, cell quotas, and the guideline bank (a prompt-side list of general task-design principles, updated
from variant progressions with the failed-check details hidden from the lesson writer; Daedalus, 2610.08048). Verifier
rewrites, World code and generator prompts stay proposal-only until the anchor sets exist and the canaries are rejected
in CI.

### 9.3 The statistics

**Rates labelled by judges: prediction-powered inference** (2301.09633).

θ̂_PP = mean f(X̃) − mean(f(X) − Y), with 95% CI θ̂_PP ± 1.96·√(σ̂²_{f−Y}/n + σ̂²_f/N), where N is all judge-labelled items
and n the random human-labelled sample. It pays only if Var(f − Y) < Var(Y) and N ≫ n; for a binary rate with symmetric
judge error η it needs η < 25% at p = 0.5 and η < about 9.5% at p = 0.1. **For rare events like a 5% hack rate the judge
must be very accurate, or PPI loses to human-only labels.** The gold sample must be random (or have known weights), and the
judge frozen while it is collected. Imputation-only intervals failed to cover the truth in every application. Use for:
hack rate, on-intent share, false-fail rate and lucky-pass rate per family (`ppi_py` exists). A stratified κ queue needs
the weighted form or a uniform slice (2404.12272).

**Calibrated judge cascades: trust or escalate** (2407.18370). On a calibration set, scan λ downward from 0.999 and stop
at the last λ whose exact binomial (1 − δ) upper bound on disagreement is ≤ α. Confidence by simulated annotators (N
few-shot prompts, each with K examples from a different annotator, K, N ≤ 5) cut GPT-4's ECE by about 50%. Order judges
cheap → strong, each calibrated only on what earlier stages abstained on (δ/|M| each). A weak cascade at target 0.8 cost
12.6% of GPT-4's cost. Abstentions form the human queue and flag rubric ambiguity (human agreement 0.815 on abstained vs
0.902 on kept items). A loop may auto-apply only verdicts above the calibrated λ.

**Clustered, paired errors and power** (2411.00640).
- SE = √(Var(s)/n); clustered SE² = SE² + (1/n²)Σ_c Σ_i Σ_{j≠i}(s_ic − s̄)(s_jc − s̄), up to 3× the naive SE on real
  evaluations. Cluster by family.
- Average K samples per item first; never treat K·n samples as independent.
- Paired: SE = √(Var(s_A − s_B)/n); at correlation 0.5 variance falls by a third.
- Power: n = (z_{α/2} + z_β)²(ω² + σ_A²/K_A + σ_B²/K_B)/δ².
- **MDE** with ω² = 1/9, σ² = 1/6, K = 8, α = 0.05, power 0.8: about **35 pp at 10 tasks, 15.5 at 50, 11 at 100, 4.9
  at 500**. Do not run an ablation whose MDE exceeds the hoped-for effect. The sealed benchmark has about 9–10 tasks, so any
  claim on it alone carries a ~35-point MDE.
- Do not lower temperature to cut variance; it shifts noise into Var(x) and can bias the mean. Keep the evaluation
  temperature at the training value.

**Other required statistics.**

| Statistic | Use | Source |
|---|---|---|
| Wilson intervals with raw counts per seed | Rare events (hack rates need 10⁴–10⁵ samples) | 2406.10162; AndroidWorld |
| Clopper–Pearson | G10 flake bound | `factory/flake.py` |
| McNemar on paired transitions, with every PASS→FAIL audited | Gate or World changes: hack removed vs work impaired vs noise | 2609.08149: 186 PASS→FAIL vs 15 FAIL→PASS, p < 0.001 |
| Nested replicates, resample units, sign-flip placebo | Any intervention on a simulated population | 2609.01519: winner's curse took the top-4 profiles from +229 to +37.6 on re-test |
| pass^k beside pass@k | Reliability; Toolathlon Pass@3 51.9 vs Pass^3 20.4 | 2406.12045; 2510.25726 |
| Consistency C_out = mean (2p̂ − 1)² | A deployment metric, never a curation objective (it penalises p̂ = 0.5) | 2602.16666 |
| Within-stratum hackability effect | Pass@1(hackable) − Pass@1(robust) inside a difficulty stratum fixed beforehand; random effects across models | 2606.16062: +14.14 pp, CI [11.80, 16.48] |
| Kendall τ for ranking agreement | A cheap validity check for a new domain's self-generated tasks before a test set exists | Daedalus: τ = 0.89 vs the AppWorld test set |
| ICC(2,1) for scalar ratings; PABAK or AC1 under prevalence skew | Anchor labels | SWE-chat; 2609.12439 |
| Run-to-run noise measured before any effect claim | RL noise about 1.6 points in distribution and 2.0 OOD | OpenThoughts-Agent (2606.24855) |
| Freeze before scoring | Cohorts, edits and manifests committed before any edited run; a sealed holdout | 2610.09142; 2610.22243 |
| End-state labels per RL run | Collapsed / hacking / alive-and-honest | HackTrace (2610.03055) |

**Judge calibration, as a card per judge.** `JudgeCard{model, family, judge_input_type, per_mode_kappa, precision,
recall_on_seeded_hacks, near_miss_acceptance, benign_perturbation_acceptance, master_key_fpr, lambda_calibrated,
gv_gap, self_preference, valid_response_rate, cost_per_100, n}`. Judge validity is per task and per property: no judge
was best everywhere, and CoT, few-shot and paraphrased prompts gave no systematic gain (JUDGE-BENCH, 2406.18403). Prefer
one validated judge over a weak-judge vote: a majority of 10 judges (77.89%) was below the best single judge
(LiveMCPBench, 2508.01780). Use the 3-annotator plus debate protocol for anchors and report each annotator's error against
consensus (a majority of three cut error from up to 23.77% to 6.01%, 2410.10934). Version anchor labels against a
`RUBRIC_VERSION`; on any criterion change, re-adjudicate the affected anchors and recalibrate every judge, since graders
drift as they see failures (EvalGen, 2404.12272).

**Build in Crucible.** `bench/ppi.py` (new), `bench/power.py`, `factory/goldset.py`, `factory/labels.py` (hash-chained),
`factory/framework_health.py` (about 20 deterministic detectors; add `judge_constant` and the canaries),
`factory/versioning.py`, `factory/repath.py`, `factory/swarm_journal.py`. **Priority.** P0 for anchors, the locked set,
canaries, PPI, MDE and clustered SEs; P1 for cascades and the full judge card; P2 for auto-apply.

---

## 10. Swarm roles and model-family assignment

**Ground rules.** The swarm builds tasks; Claude never authors a task, oracle, check or report in the pool. The swarm
runs on Tinker (the `swarm` preset, `factory/models.py:125-144`). Roles extend `BUILDER_ROLES` / `INDEPENDENT_ROLES`
(`models.py:152-163`), which already raise `SameFamilyError`. The student is Qwen3.6-35B-A3B; no Qwen model judges,
audits, attacks or traces it. The swarm improves by harness optimisation (memory, guideline banks, recipe versions), not
by weight training: Daedalus showed prompt-side banks transfer across families (Qwen3.6-35B-A3B +29.3 with its own
family's bank, 2610.08048), and AutoEnvScaling showed a shared trained proposer collapsing without continued proposer
training (2610.14163).

**Role table.** The "Family today" column is the current `swarm` preset where a role exists; other assignments are
**[Proposal]**, chosen so that every row's "must differ from" constraint holds.

| Role | Stage | Family today / proposed | Sees | Never sees | Must differ in family from |
|---|---|---|---|---|---|
| Surveyor / domain architect | F0 | GLM-5.3 (planner) | World read-only, brief, trace clusters | Held-out families, benchmark tasks | — |
| Schema and tool coder | F1 | Kimi-K2.6 (world_expert) | DomainSpec, contracts | Tool tests | Blind tester |
| Blind tool tester | F1 | DeepSeek-V4.1-Flash [Proposal] | Spec, contracts | Tool code | Coder |
| Debugger | F1 | Kimi-K2.6 [Proposal] | Tracebacks, code | Cannot edit expectations, DB instances or tests | Approver of any test edit (Nemotron-3-Ultra) |
| State seeder | F2 | Kimi-K2.6 (world_expert) | Schema, cell slice, persona pool | Tasks, verifiers | — |
| Task proposer (draft) | F3 | Kimi-K2.6 (author) | Cell, World slice, gate-passed references, ledger, Assignment | Gate code, anchors, honeypot paths, verdict internals | Refiner, verifier writer |
| Task refiner | F3 | GLM-5.3 [Proposal] | The draft | Gate code | Proposer (WebForge) |
| Oracle programmer | F4 | DeepSeek-V4.1-Flash (oracle_builder) | Proposal, tools | Verifier | Alternative oracle writer |
| Alternative oracle (G17, G14) | F4 | gpt-oss-120b [Proposal] | Spec only | First oracle | Oracle programmer |
| Verifier writer | F5 | GLM-5.3 (verifier_writer) | Oracle, gold end state | Proposer's notes, any candidate solution (Scoring Verifiers) | Proposer, oracle programmer, attacker |
| Negative and mutant builder | F5 | Kimi-K2.6 (mutant_builder) | Oracle, operators | Verifier internals | Verifier writer |
| Attacker | F6 G12 | Nemotron-3-Super (hacker), gpt-oss-120b (hacker_2) | Instruction, World, failure logs | Verifier, oracle | Builder, suite, student |
| Instruction and report writer | F3–F5 | Kimi-K2.6 (prompt_author) | Oracle summary (variables, not values), leak rules | Hidden checks | Reviewer |
| Realism and leak reviewer | F5–F6 | gpt-oss-120b (realism), Nemotron-3-Ultra (reviewer) | Instruction, World | **Any pass/fail verdict** | Writer |
| Blind re-implementer (instruction edits) | F6 repair | A family that saw neither task nor oracle [Proposal] | The edited instruction | Task history, oracle | Repairer |
| Static auditor (ABA-style) | F6 / F10 | Nemotron-3-Ultra (auditor) | Instruction, tests or checks, config | — | Author families |
| Intent auditor | F10 | Nemotron-3-Ultra (auditor) | Rollouts, ablations | — | Student, builder |
| Failure judges (3, 3-of-3 to auto-route) | F9–F12 | gpt-oss-120b (judge) + Nemotron-3-Ultra + one of GLM / DeepSeek not used by the task's builders [Proposal] | One trace, World ground truth (tool responses, events, clock) | Builder identity, the grader's verdict | Student, builder |
| User / NPC voice | F2 / F9 | A non-Qwen family different from the task's judges [Proposal] | ActorSpec | Gold oracle, checks | Student; the judge of that task |
| Trace clusterer | F12 / §8 | Nemotron-3-Ultra [Proposal] | Redacted summaries | Raw traces | — |
| World rebuilder | §8 | DeepSeek-V4.1-Flash [Proposal] | Redacted logs, schema | Raw PII | Re-solver |
| Re-solver | §8 | gpt-oss-120b [Proposal] | Rebuilt task only | The trace's actions | Rebuilder |
| Difficulty controller | F9 | Deterministic + Kimi rewrites | p̂, knobs, exemplars | Gate code | — |
| Pipeline doctor | §9 | GLM-5.3 [Proposal] | Per-stage yields, reasons, costs, framework_health | Anchors, gate code, locked set, hidden tests | — |
| Frontier references | F9 | Qwen3.5-397B, DeepSeek-V4.1-Flash (debug lane); Qwen3.5-397B, Nemotron-3-Ultra (`swarm`) | — | — | When the 397B is the student, no Qwen frontier |

**Why the hack-prone families are placed where they are.** GLM and DeepSeek showed the highest measured hacking
propensity in the cards (§5.7). That makes them risky in any role that *executes* inside a sandbox near verifier files,
and useful as an *attacker* against tasks they did not build. **[Proposal]:** keep the preset's attackers as the default,
and add GLM or DeepSeek as a third attacker for any task whose verifier and oracle were written by other families. Every
builder session runs with no verifier files, no honeypot paths and no anchors in reach.

**Swarm memory is filtered.** `swarm_journal` entries, guideline banks and in-context references pass the same filter as
F3 references: only gate-passed, hack-free artifacts; any later-flagged entry is withdrawn and its descendants re-gated
(2610.12822).

**Spend control.** The swarm pilot cap is $300 (the lead's rule). Every role logs cost per artifact into provenance, so
F12 can compute value per dollar per family and per role.

---

## 11. Costs, yields and a budget model for 500–1000 tasks

### 11.1 The formula

For stages s = 1…n with cost per candidate c_s and pass rate y_s:

  **cost per kept task = Σ_s c_s / (y_s · y_{s+1} ⋯ y_n)**

An expensive stage must sit late *and* have a high yield, and a defect the expensive stage would find is cheapest to catch
one stage earlier. Today the student measurement (about $14 per k = 8; $20–35 per full try with the frontier) is the last
stage, and its first-pass yield was 0 of 17 (map §4). That is why each kept task cost $30–130 plus hours of human time.

### 11.2 Priors from the cards

| Item | Prior | Source |
|---|---|---|
| Student rollout, 300 calls | $1.72, 27.6 min | map §4.4 |
| 397B rollout / DeepSeek rollout | $3.01 / $2.05 | map §4.4 |
| Valid environment with reference execution | $0.02 (DeepSeek-V4-Flash) to $1.18 (Opus 5) | AutoEnvScaling (2610.14163) |
| Mechanism-first environment | $0.01–0.03 | VHD-Play (2609.27321) |
| Code environment | $1.02 (EnvScaler, of which $0.77 the 100-round test); $0.57 (AWM); $1.93 per site (InfiniteWeb); $3.6 per cloned site (VeriEnv); $4.12 (AutoEnv) | 2601.05808; 2602.10090; 2601.04126; 2603.10505; 2511.19304 |
| Accepted task, executed construction | about $0.05 (RST: $50 per 1,000); $0.10 (AgentSynth); $0.78 per kept item (AgentFrontier sandwich) | 2608.05466; 2506.14205; 2510.24695 |
| Static audit | $0.42–0.80 per task | ABA (2605.26079) |
| Hackability audit | $6.04 for 49 tasks | 2606.16062 |
| Differentiating tests | $0.04–0.10 per patch | PatchDiff (2503.15223) |
| Mutation strengthening | $2.50 per instance | SWE-ABS (2603.00520) |
| Judge decision | about $0.60 (GAUGE); $1.80 per RL step (AWM) | 2609.12191; 2602.10090 |
| Deterministic checks | Nearly free: state-diff negatives, mutant kill runs, honeypots, twins, isomorphic regrades, static taint, literal checkers, cue audits | Extract D B8.1 |
| Swarm report writing | $3.5 for 10 machine runs | map §3.2 |
| Swarm pilot, per kept task | $129.7 (student $101.6) | map §4.4 |
| Yields seen elsewhere | 5.2% (SCA) · about 39% RL admission (WEFT) · about 53% acceptance (WildAWM) · about 60% after an instruction check (RandomWorld) · 74.1% (WebForge) · 16% of mined sessions (EnterpriseClawBench) | 2506.01716; 2609.36887; 2610.26828; 2506.11045; 2604.10988; 2606.23654 |

### 11.3 A budget model **[Proposal; every yield below is an assumption to be measured]**

Shape: **50–100 families × 10–20 instances, S = 2–4 seeds**, mixed sources (synthetic World families as the breadth
source; the debug lane as a high-realism slice of tens to low hundreds of tasks; traces when available).

Assumed yields: y_det = 0.4 (F3–F6 deterministic gates), y_model = 0.7 (F6 model gates), y_band = 0.4–0.6 (F9, with
the predictor ranking candidates). Each line is divided by the yields of its own and every later stage.

| Cost line | Assumption | Per kept task |
|---|---|---|
| F0–F1 per family (App build, tests, repair), amortised | $5–20 per family over 10–20 kept instances | $0.3–2 |
| F3–F5 per candidate (proposal, oracle, alternative oracle, verifier, negatives) | $0.3–1.0 per candidate | $1.8–8.9 |
| F6 model gates per CPU survivor (static audit, attacker, G8x, G9x) | $1–2 per survivor | $2.4–7.1 |
| F9 frontier audit on a 20% sample | $6 per sampled task (k = 2 on the 397B), so $1.2 per measured task | $2.0–3.0 |
| F9 student measurement | k = 4 for every measured task ($6.9), +4 for half of them ($3.4) | $17–26 |
| F10 extra probes | About 4 short runs, partly reused from F9 | $2–5 |
| **Total** | | **≈ $25–50 per kept task; $25k–50k for 1,000** |

**What this says.** Even with every P0 lever, student measurement is most of the cost. Two P1 levers change the order of
magnitude:
1. **Admit on predictor plus a cheap screen, then measure online.** Once training runs, p̂ comes from training rollouts at
   no extra cost (RODS); LILO-style allocation spends 2 rollouts at a time where p̂(1 − p̂) is highest. If the admission
   screen is k = 2 ($3.44) on predictor-ranked tasks and online p̂ does the rest, measurement falls to $3.44/y_band per
   kept task.
2. **Move defects before F9.** Each defect a CPU gate catches instead of F9 saves one measurement ($7–14). Today's causes
   of death at measurement (unstated rules, faults the report never described, fake links, stale visible tests, changelog
   hints, budget exhaustion; map §4.2) are each now covered by a CPU or cheap-model gate: G8 static, G24 cue, G11, G4,
   and the oracle-horizon budget check.

With both, and if moving defects earlier lifts y_band to 0.7–0.8, the same formula gives roughly **$11–23 per kept task,
$11k–23k for 1,000** [Proposal]. That still exceeds the Tinker
credit (memory: $5k, with about $1,048 of sampling already logged on this machine, map §2.1). So: reach 500–1000 tasks in
stages, measure the stage yields on the first two DomainSpecs before scaling, and spend the student budget only on
predictor-ranked, CPU-gated candidates.

**Where the money goes, and where it should not.** Construction is cheap (cents to a few dollars per task). Judges and
frontier passes are mid-cost and stay sampled. Student rollouts dominate. Human time stays on anchors, spot checks and the
lead's review, not on fixing tasks one by one (today about 1.5 kept tasks per person-day).

### 11.4 Infrastructure levers

| Lever | Evidence |
|---|---|
| One image per domain or repo; many tasks per snapshot | SWE-smith: 125 images, 290.5 GB vs an estimated 50–150 TB per-instance (2504.21798) |
| Copy-on-write state per rollout | CUA-Sandbox: CoW create on a 3.6 GB DB 5.48 → 0.07 s; MegaMCP upload 773.5 → 34.8 MiB at 1,000 tasks |
| Elastic gate workers | S3 and S10 take 20–60 min per candidate on one shared 14-CPU host today (map §4.4) |
| Fast sandboxes | crun with pre-created cgroups: 0.04 s start vs 0.12 s Docker (Kimi k1.5, 2501.12599) |
| Early stop for measurement only | EarlyEval: −26% steps at ≤ 2 pp distortion; never a training reward (2609.02783) |
| Prefetch the next batch's servers | AWM: 1,024 isolated MCP servers per RL step |

---

## 12. The debug lane as one plug-in domain

| Debug-lane piece | Generic slot | Change |
|---|---|---|
| `LangProfile` (python, golang, node, rust, java) | DomainSpec plug-in: World recipes, runners, leak probes, protected paths | None; already a plug-in table (`factory/langs/base.py:55`) |
| Git mining (`fix_check`, `interaction_edges`, groups) | Source "mined" → F2–F4 | Emits `SeedBundle` (repo at A′) and `Oracle(kind=patch)` |
| Hidden fail-to-pass tests | `state_checks` of kind `test_run` | Through `PatchTask` |
| Commit-revert mutants | `negatives` (missing, wrong) | **Add** side-effect negatives, the revert-shortcut negative, impossible twins and the attacker |
| S3 suite reruns, S10 flake | G2, G10 | Same code; sequential certification |
| S5, S8, S6 | G3 / G4, G5, G6 | Same code; G4 gains red-on-start for preservation tests |
| `pathcheck` | `path_constraints` | Generalised to tool actions |
| Report lint, changelog strip, `disclosure_check` | `leak_rules`, G11 | As data |
| `SKILLS`, `failure_skill`, K/C/H knobs | `skill_tree`, `knobs` | As data; negative definitions added ("copying the fix from history is not debugging") |
| Keep rule (`targeting.py`) | F9 band per student + F10 verdict | Keyed by student; fault-side accounting; ablations |
| Manual fix choice, manual checklist | F3 proposer + F10 probes | Automated with a human spot check |
| `benchseal`, `sealguard` | Pool split `benchmark` | Unchanged |

**New for the debug lane** (all P0 unless marked):
- **G19 clean-custody regrade** and the pre-test restore rules (§3.7). SWE-Bench Pro V2 repaired or removed 66% of 731
  tasks under such rules (2610.24206).
- **Leak-channel checks** before any rollout: rebuild each repo as a fresh single-commit repo, remove nested histories,
  delete tracked tests and golden data, disable hooks, hash instance ids, strip repo names from paths, allow only the model
  endpoint on the network. Loose objects survive ref deletion (2609.08149). The start tree is an empty repo plus the base
  ancestry only, and a validator searches the object store for the fix commit (2607.26819).
- **H1 parser gold sanity** for all five languages' log parsers (UTBoost, 2506.09289).
- **Memorisation probes** on mined tasks: file path from the issue alone, function reproduction with Δ5 overlap, verbatim
  prefix completion, against fresh same-repo issues and outside repos of similar popularity (2506.12286); metamorphic
  twins (rename plus structural rewrites such as nesting else-if) as the obfuscation knob (2604.21579). The student's open
  weights let its NLL on the start files rank which tasks to probe. A MEMORY-successful pass gets zero reward (2609.06780).
- **Run all functional tests, not only fail-to-pass.** Running all developer tests failed 7.2–8.4% of plausible patches
  (2503.15223). Scope runs to graded packages and cap parallel workers.
- **Mirror** kept fixes into already-built repos to raise held-out supply without new builds (SWE-Mirror, 2509.08724) (P1).
- **FeatAdd**: a different-family model implements a feature at A′; the first test break is the task; it requires the
  feature's own new tests so a revert does not count (BugPilot, 2510.19898) (P1).
- **PR-type routing** (bug fix, feature, perf, refactor, test) with type-specific rewards, and SWE-Test inversion as a
  second family (MiniMax-M2 [PARTIAL]) (P1).
- **Time-machine pinning**: proxy package indexes so only versions released before the base-commit timestamp are visible
  (SWE-bench-Live, 2505.23419) (P1).
- **`--no-link` variants are a separate family** (they drop the interacting-faults skill).

**Migration without breaking what exists.**
1. Add `TaskContract` and `PatchTask`. Recompute every stored grade of the kept training tasks through the adapter;
   verdicts must be bit-identical.
2. Move the lane's lists into a `debug` DomainSpec file, with tests that the old constants equal the loaded data.
3. Turn on the new gates for **new** candidates only.
4. Sealed tasks are never re-gated; their manifest hashes stay valid and they appear only as counts.
5. Wire the herding guard, ortho and dedup into the lane; report cell fill and repo Gini on the kept set.

---

## 13. Roadmap

**Conventions.** Effort is a planning estimate for one engineer, not a measurement: **S** up to a week, **M** 1–3 weeks,
**L** more than 3 weeks. Every item updates `DESIGN.md` in the same commit, names its reuse source (DEPEND, PORT or
PATTERN from agentenv or RL-Environments), and has an acceptance test that fails on today's code. Part 1 items that Part 2
needs (reading back R1/R2, the L1 flags on `WorkspaceHarness`, framework_health pausing training) are in
`part1-framework-improvements.md`.

### 13.1 P0

| ID | Item | Effort | Modules | Acceptance test | Enables |
|---|---|---|---|---|---|
| P0-1 | `TaskContract` + `PatchTask`; closed-world diff grading; answer checks with exclusion conditions; milestones, minefields, guardrails, timing windows; fail-closed grader with reason codes and range assertions | M | `factory/contract.py` (new), `factory/longtask.py`, `verify/state.py`, `verify/answers.py`, `verify/sql_checks.py`, `world/custody.py` | Every stored grade of the kept training tasks recomputes bit-identically; a `finance_ap` task grades through `grade()`; a duplicated-write negative and an extra-field negative fail it; an injected verifier timeout yields FAIL with a reason code | All of Part 2; M4 |
| P0-2 | Negative library: operators of §3.6 with the side-effect class, idempotence replay, wrong-task twins (G5b), impossible twins; gold sanity on every LLM-written check; corrected kill protocol and report | M | `accept/mutation/actions.py`, `factory/verifier2.py`, `factory/lanes/windows.py`, `factory/generators/base.py` | On the toy helpdesk and `finance_ap`, every operator produces an executed negative, the idempotence replay classifies harmless repeats, and a planted dict-keyed verifier is caught by Extra; an always-pass verifier scores 0% kill | M4 |
| P0-3 | Deterministic pre-rollout gate runner: G1, G2 (double replay, reset fidelity, expectation hashes), G4 red-on-start + null-agent library, G7, static G8, structural G9, G11 (+ VCS hygiene for debug), G13, G15 | M | `factory/generators/base.py` (`deterministic_gates`), `factory/benchaudit.py`, `factory/pathcheck.py`, `world/world.py` | Planted defects (an unstated literal, a check green on start, a constant attached to the target, a fix commit left in the object store) are each caught before any rollout | M7 |
| P0-4 | Trust boundary: grading in a custody World, protected resources at `_call_tool`, honeypot twins (G22), G18 fault injection, G19 clean-custody regrade for the debug lane, H0 audit of the lane's harness | M | `world/world.py`, `world/custody.py`, `world/verifier.py`, `guard/isolate`, `factory/lanes/windows_world.py` | A planted `conftest.py`, a PATH shim and a `python3` PASSED-printer each fail under G19; a honeypot read is flagged; H0's exploit run fails | M4, M11 |
| P0-5 | Provenance with the real author and the full `DifficultyRecord` key | S | `factory/provenance.py`, `factory/swarm_prompts.py`, `forge/difficulty.py`, co-author templates | New reports carry the true builder; a lint refuses a copied stamp; two records differing only in simulator id stay separate | M7, M9 |
| P0-6 | Anchor sets per judge type (200–500 adjudicated, built as a codebook), a locked set, canaries, judge cards | M (mostly human time) | `factory/goldset.py`, `factory/labels.py`, `factory/framework_health.py` | Always-pass, always-fail, constant-Tie and master-key canaries are rejected in CI; each judge prints a card | M5, M10; every loop |
| P0-7 | Hack flags and quarantine on training rollouts; a different-family trajectory monitor; clean-vs-raw pass rates per family | S–M | `credit/hack_detector.py`, `runtime/harness/workspace.py`, `train/quarantine.py`, `scripts/train_monitor.py` | A planted hack in a fake run quarantines its task and withdraws it from references | M11 |
| P0-8 | Behaviour signatures, archive meters, three dedup layers; herding, ortho and dedup wired into the debug lane | M | `factory/signature.py` (new), `factory/archive.py` (new), `factory/ortho.py`, `forge/dedup.py`, `factory/root.py`, `observe/factory_views.py` | Meters print on every batch; a candidate on an over-full repo is held; a claimed cell that disagrees with the measured one is relabelled | M8 |
| P0-9 | Executed failure attribution by replay; `Assignment` consumer of `BuilderIntent` | M | `factory/traceaudit.py`, `factory/failures.py`, `world/world.py` restore | On a toy task with a planted wrong step, replay finds t* within O(log T) restores; an `Assignment` produces a gated variant | M6 |
| P0-10 | Statistics: Wilson, clustered and paired SEs, MDE, PPI | S | `bench/power.py`, `bench/ppi.py` (new) | Every ablation report prints its MDE and refuses to run when MDE exceeds the stated effect | All M-items |
| P0-11 | Predictor-first, sequential k = 4 → 8, p̂ reuse from training rollouts | M | `factory/targeting.py`, `factory/difficulty.py`, `train/curriculum.py`, `scripts/lh_debug_target.py` | On one batch, spend per kept task is reported for the old and new protocols, with the same keep verdicts on tasks both measured | M7, M9 |
| P0-12 | `debug` and two non-coding DomainSpecs (`finance_ap`, toy helpdesk) | S–M | `core/schema/taskgen.py` (new), `factory/generators/debug.py`, packs | The debug lane's constants equal the loaded data; both non-coding specs load and validate | Genericity |

### 13.2 P1

| ID | Item | Effort | Modules | Acceptance test | Enables |
|---|---|---|---|---|---|
| P1-1 | `world_synth` F0–F6 (DEPEND: vendored AWM order; PATTERN: ScaleEnv tool-first, EnvScaler tester, WEFT contracts and routing) on `finance_ap` and the toy helpdesk | L | `forge/world_synth/` (new), `world/apps/sqlite_app.py`, `factory/machine.py`, `factory/sessions.py` | Two domains pass F0–F6 with yields, repair iterations and cost per stage; a debugger edit to a frozen expectation is refused without a different-family approval | M3, M7 |
| P1-2 | App contract: state envelope, tenancy, resource classes, `settle()`, `config_flags`, conformance and soak tests (H-gates) | M | `world/state/base.py`, `world/apps/*`, agentenv server SDK | A new App passes H-tool, H-error, H-soak and H-settle; a reset that drops the seed fails H-soak | Rigour |
| P1-3 | F10 intent probes and IntentCards with fault-side accounting | M | `factory/targeting.py`, `factory/traceaudit.py`, `factory/cards.py` | Every kept task has a card; the lead reviews the verdicts | M6 |
| P1-4 | G12 on every task, G14 at full strength, G16, G17, G20, G21, G23, G24, G8x, G9x; J1–J7 | M | `forge/hacker.py`, `factory/alternatives.py`, `factory/differential.py`, `tests/redteam/` | Each gate catches its planted defect; G12 runs on every variant that raises the band | M4 |
| P1-5 | Clock bus and events (Gaia2 model), time modes, `OpenEffect` | M | `world/clock.py`, `core/triggers.py`, `core/schema/world.py` | A deadline task grades with a timing window; snapshot and restore keep the pending DAG; settled and endpoint grades are compared | M12 |
| P1-6 | Declarative `ActorSpec`, registry resolution, behaviour modules, U1–U4 | M | `core/registry.py`, `actors/*` | The helpdesk builds from YAML with no Python `attach`; U3 reports p̂ under two simulator families | M12 |
| P1-7 | Archive with PLR sampler, RODS lifecycle, quotas, ANNECS-OMNI curve | M | `factory/archive.py`, `train/curriculum.py` | Variants enter at ≤ 20% of the pool; originals are never retired; the curve prints per batch | M3, M8 |
| P1-8 | Controllers (set-point, bandit, SETA policy), ladders (mixed late starts, hint histograms, check-subset siblings), IRT | M | `factory/difficulty.py`, `bench/irt.py` | A ladder rung brings a 0/4 task into band on the 35B; IRT b agrees with Wilson placement on measured tasks | M9 |
| P1-9 | Traces-to-tasks Route A on synthetic enterprise logs | L | `connectors/seed.py`, `connectors/redact.py` (new) | Synthetic logs → rebuilt World → sufficiency → re-solved task; canary and redaction gates pass | M14 |
| P1-10 | Debug lane additions: mirror, FeatAdd, PR-type routing, time-machine pinning, memorisation probes | M | `factory/generators/*`, `factory/langs/*`, `factory/heldout_supply.py` | Held-out supply rises without new repo builds; a verbatim-memorised task is routed to obfuscation | M7 |
| P1-11 | Judge cascades with calibrated λ | M | `factory/audit.py`, `bench/ppi.py` | A loop applies only verdicts above λ; abstentions reach the human queue | M5 |

### 13.3 P2

| ID | Item | Effort | Enables |
|---|---|---|---|
| P2-1 | Dense office-core Worlds (calendar, chat, docs, tickets, CRM) as day-delta Worlds, with a density and realism card | L | M13 |
| P2-2 | Real OSS tiers as MCP sidecars; sim-to-real twin of a dense World | L | M13 |
| P2-3 | Worldbook route (Trace2Env) as an F1 spec source; customer-data importer with the full privacy gate | L | Enterprise pilots |
| P2-4 | Auto-apply of low-risk loop classes behind every guard; the pipeline doctor in AEvo shape with rollback | M | M10 |
| P2-5 | Self-play and in-training variant sources (RODS, SSR-style) | L | M3 |
| P2-6 | Learned simulators and cache tiers behind U4 | M | — |
| P2-7 | Hack-precursor probes (PRIME text probes, non-Qwen judges) | S | M11 |

### 13.4 The measurements the paper needs

| ID | Question | Design | Statistics |
|---|---|---|---|
| M1 | Does Φ-dense credit beat sparse at equal compute? (Part 1) | R2 vs R3, same tasks, seeds, budget | Task-clustered paired bootstrap |
| M2 | Does the keep rule select tasks that teach? | Per-task learning progress against band position (incl. the (0.2, 0.8) arm), on-intent share, intent verdict | Rank correlation with clustered CI; report the null |
| M3 | Families vs instances | Fixed task count, vary families; score on held-out families; per student | SynthAPI-style fit with CIs over seeds |
| M4 | Verifier false-pass rate per gate | H(t), side-effect kill rate, gold-sanity defect rate per source and generator | Wilson; per-task H(t) distribution |
| M5 | Rule vs judge vs code-augmented judge | On anchor sets | False pass / false fail with CIs; κ |
| M6 | Intent ablation deltas | Each F10 probe per task | Paired, clustered differences |
| M7 | Factory economics | Yield and cost per stage per source; cost per kept before and after P0-11 | Per-stage yields with intervals |
| M8 | Do diversity meters predict transfer? | Behaviour Vendi, cell fill and covering of pool ablations against held-out-family score | Correlation across fixed-size pools |
| M9 | Bands per student size | The same pool on the 35B and the 397B | Per-student p̂ with Wilson; IRT θ with SE |
| M10 | Do the loop guards work? | Evolved verifiers with and without anchors; canary rejection | Collapse rate; anchor agreement |
| M11 | Hack rates over RL steps | L1 flags, twin passes, honeypot hits, Δ P90, clean vs raw | PPI intervals per step |
| M12 | People and time fidelity | U2 / U3 results; time-mode sensitivity | Rates with intervals; ECE by bin |
| M13 | Does World density cause transfer? | Density-card coupling against held-out score; dense-vs-thin twin | Correlation; sim-to-real gap |
| M14 | Fidelity of traces to tasks | Re-solve rate, sufficiency, intent agreement, consistency ratio | Rates with intervals |

### 13.5 The ablation protocol for the factory itself

The factory is a pipeline of stages, each with a claimed value. Each claim is tested the way OpenThoughts-Agent tested its
data recipe (2606.24855) and OpenThoughts its stages (2506.04178), with the guards of §9:

```text
ablate(stage s, variant v):
  freeze every other stage, recipe and seed; freeze a task slice and a held-out-family set
  generate the SAME number of kept tasks (equal budget in dollars AND in kept count) with and without v
  train a fixed small student (or run a fixed short RL budget on the 35B) on each pool
  evaluate on held-out families only (never on families the stage read), K >= 3 re-runs per task
  report: mean per-benchmark z-score, paired and family-clustered SE, MDE computed BEFORE the run,
          run-to-run noise measured beforehand (RL: about 1.6 in distribution, 2.0 OOD)
  accept v only if the gain exceeds the noise floor delta and the lower confidence bound > 0
  move greedily, one stage at a time; log every rejected variant as negative evidence (RRSI ledger)
```

The first ablations, in order: (1) frozen vs unfrozen tool-test expectations; (2) with vs without the side-effect
negative class; (3) red-on-start on vs off (WildAWM measured it at −7.78 average); (4) predictor-first vs measure-all at
equal spend; (5) grid + signature dedup vs text-only dedup; (6) targeted (`Assignment`) vs untargeted batches at equal
budget; (7) mixed vs single-source pools. Each answers one stage's claim, and each is sized by its MDE first: at 50
held-out tasks the MDE is about 15.5 pp, so effects smaller than that need more held-out families, not more runs.

---

## 14. Corrections ledger

### 14.1 Corrections to `framework-plan.md`

| # | Plan location | Plan says | Correction | Source |
|---|---|---|---|---|
| 1 | §3.6 objective | Expected improvement bounded below by p(1 − p) | The KL to the soft-optimal policy is bounded below by p(1 − p)/2β²; per-step improvement is bounded *above* by a Var[r] term (LILO) | 2504.03380; 2502.12272 |
| 2 | §3.6 bands | "WebRL keeps 0.05–0.75" | A band on a trained critic's value estimate at the start state, not a measured pass rate | 2411.02337 |
| 3 | §3.6 bands | "GenEnv centres at 0.5" | Measured on unverified tasks; its bound implies about 490 rollouts to separate tasks 0.2 apart; per-task placement at k = 8 needs pooled cell estimates | 2512.19682 |
| 4 | §3.6 bands | "SSR's optimum is about 0.2" | The optimum of SSR's injector reward from its game analysis; SPICE ranks a variance reward peaking at 0.5 first; 0.2 vs 0.5 is an open measurement | 2512.18552; 2510.24684 |
| 5 | §3.6 [0.1, 0.6] | — | Not refuted by 2504.03380's one-sided rows (cuts at 0 or 1); add a symmetric (0.2, 0.8) arm to R2 | 2504.03380 |
| 6 | §3.6 sequential k | "Unmeasured proposal" | Published variants exist: AutoEnvScaling's 4 → 8 near band edges with std ≥ 0.1; LILO's 2-at-a-time allocation; IRT adaptive selection with SE stopping | 2610.14163; 2502.12272; 2509.11106 |
| 7 | §3.6 Endless Terminals | "2.2% → 3.4%" | That is the Qwen2.5-7B row; 0.0 → 2.2 (Llama-3.2-3B), 1.1 → 6.7 (Qwen3-8B) | 2601.16443 |
| 8 | §3.6 ladders | "Late start states (R³)" | Mix start positions, do not stage; equal reward per start; keep the earliest starts | 2402.05808 |
| 9 | §3.6 ladders | "Partial-solution hints (QuestA)" | The hint effect is steep and narrow; calibrate per family with a histogram | 2507.13266 |
| 10 | §3.6 "iterative harder" | Adopted (DeepSeek-V3.2) | Its agent edits solution *or* verifier until they agree and filters by policy pass@100; adopt only with a different-family verifier writer and negatives re-run each round | 2512.02556 |
| 11 | §3.6 hint removal | MiniMax-M2 "[uncertain scale]" | No task counts or yields reported at all; the flag stays | 2605.26494 [PARTIAL] |
| 12 | §3.1 gate list | G1–G14 | Add G5b, G15–G24, H0, H1 and the J- and U-gates (§3.7); strengthen G4 to red-on-start, G6 to drop each step, G8 from values to rules, formats and locations, G2 to reset fidelity | Extracts A, C, D, E, F |
| 13 | §3.1 G4 null agents | Empty; claims everything | Add dump-all-data, list-all-answers, always infeasible, always FAIL, always act, never act, else-branch, default and idle drivers | 2507.02825; 2307.13854; 2404.07972; 2405.00823; 2609.27321 |
| 14 | §3.1 G5 | Kill floor on generic mutants | Needs the corrected protocol (baseline first, stillborn excluded, other-channel kills separate, indeterminate timeouts, sequential, byte-identical rerun) and a log of revisions made to pass the gate | 2608.12635 |
| 15 | §3.1 G12 | H(t) = 0 | A budget statement, not a proof; re-run on hard and controller-hardened variants | 2209.13085; 2605.20744 |
| 16 | §3.1 G14 | "Where the family requires them" | Too weak: official checkers had 55.9% recall and 18–24% of WebArena checks were wrong; measure checker recall in every family | 2504.08942; 2408.06327 |
| 17 | §3.1 G10 | 60 grades | A run-completeness check comes first; with a simulated user, 60 grades cost about 60 × $0.145 per task (τ²), so use sequential testing or a cheaper simulator | 2610.27465; 2506.07982 |
| 18 | §3.3 operators | Eight operators | Add claim-only, partial, partial propagation, extra-field, idempotence replay, wrong-task twin, constant answer, stale output, feature-isolated, tamper, bait, impossible twin, deceptive outcome, concern-targeted; add realisation checks as a sixth check kind | 2610.09142; 2610.19686; 2610.33817; 2610.18868; 2605.21384; 2604.23488; 2604.10988; 2501.12862 |
| 19 | §3.3 | — | Do not adopt EnvCraft's "every chain ends in a write": read-only, answer and infeasible tasks are required types | 2609.05576 |
| 20 | §3.3 infeasible | — | Needs positive evidence (refusal + policy lookup), never unchanged state alone | 2507.02825 |
| 21 | §3.3 confidential | "Near 0–2.9%" | The card gives near 0% with a standard prompt and up to 62.9% with a confidentiality-aware prompt; the 0–2.9% range could not be checked | 2505.18878 |
| 22 | §3.4 / §3.10 AutoEnv "65% verified" | A yield | An F1 pass rate before G3/G5/G12; excludes the hand selection of 36 of 65; "verified" admitted a 138.84% reward and a mismatched instruction file | 2511.19304 |
| 23 | §3.10 AWM "> 85% first attempt" | A yield | First-attempt stage success is not App quality: 74–83% of envs had ≥ 1 bug; 23.7–28.2% of tasks never solved | 2602.10090 |
| 24 | §3.4 / §3.9 EnvScaler as the blind-tester precedent | — | Its tester writes no expected outcome and tolerates 15% failure; frozen digests are Crucible's rule | 2601.05808 |
| 25 | §3.5 item 1 | Nemotron-Terminal "9 or 10 domains [uncertain]" | 9 domains; the body and appendix name one domain differently | 2602.21193 |
| 26 | §3.5 item 3 | Verbalized Sampling: "keep the least typical valid one" | The paper samples from or thresholds the verbalized distribution; novelty is decided by F7's measured signature | 2510.01171 |
| 27 | §3.5 item 3 | Personas for diversity | Persona variety ranked datasets backwards (54.77 vs 38.15 OOD); personas buy realism | 2505.20161; 2406.20094 |
| 28 | §3.5 item 5 | AgentFrontier ≥ 0.7 as an embedding threshold | A reranker score over questions; mpnet cosine 0.9 is a separate, cheaper layer | 2510.24695; 2603.12698 |
| 29 | §3.5 item 7 | "Top-16 hurt every benchmark compared with top-8" | At 100K, top-16 was below both top-8 and top-4; at 10K, top-4 ≈ top-2 was best | 2606.24855 |
| 30 | §3.5 meters | Four meters | Add strength-2 covering, the GRADUS ledger, the nearest-neighbour tail, fidelity and realism; validate Vendi against transfer before using it | 2610.18868; 2610.27781; 2608.05466; 2606.12191 |
| 31 | §3.7 on-intent bar | 0.5 | Keep the lead's 0.5, computed with fault-side accounting over model-side failures | 2607.28802 |
| 32 | §3.7 lucky-pass probe | — | Needs ≥ 2 (ideally 5) passing traces, so it cannot score the hardest tasks; fall back to decisive-error and Δ signals there | 2605.12925 |
| 33 | §3.7 single-subtask split "[inferred]" | — | Needs hidden compositional checks (SpecBench) to measure anything | 2605.21384 |
| 34 | §3.8 anchor set | 30–50 episodes per lane | 200–500 adjudicated items per judge type for calibrated thresholds; ~10 dev anchors per family for verifier evolution; a locked set; labels versioned against a rubric; hack positives from natural rollouts | 2407.18370; 2610.11464; 2404.12272; 2604.23488 |
| 35 | §3.8 Meta-Agent Challenge | Exfiltration "despite layered defences" | 591 *dev* labels leaked through verbose tracebacks; it did not transfer to the hidden test split; split isolation worked | 2606.04455 |
| 36 | §3.8 guards | "Never selected on student reward" | Extend to visible score, learned rewards and CoT; "a CoT may flag, never clear"; equal-budget controls; edit budgets with file allowlists; the masking alarm; round caps for judge-only loops | 2605.21384; 2502.10325; 2505.05410; 2604.18292; 2608.05466; 2610.16523; 2412.02674 |
| 37 | §3.8 verifier loop | Re-tune verifiers | Refreshing a reward model fixes only extremal exploitation; the loop must *add* negatives and checks | 2210.10760 |
| 38 | §3.8 canaries | Always-pass, always-fail | Add constant-Tie, constant-policy baselines, master keys, narrated completion, and a screen for predicates that cannot fail | 2609.12439; 2607.28685; 2507.08794; 2608.03744 |
| 39 | §3.8 κ queue | Fixed stratified rate | Needs the weighted PPI form or a uniform slice | 2301.09633; 2404.12272 |
| 40 | §3.8 shared-policy proposer | — | AutoEnvScaling's collapse (86.8% → 22.4%) is why in-training proposer loops stay P2 | 2610.14163 |
| 41 | §3.9 swarm roles | — | Reviewers never see grader verdicts; F3 references must be gate-passed and hack-free; GLM and DeepSeek hacking evidence | 2610.12822; 2608.22103; 2608.30724 |
| 42 | §3.2 `DifficultyRecord` | Key (task, student, checkpoint) | Add user-simulator id and version and the time mode; record harness, budget, token cap | 2601.17087; 2602.11964; 2608.12635 |
| 43 | §3.2 `Knob.kind` | Nine kinds | Add noise, pressure, latency, contention, user_behaviour, disclosure, time_mode, instruction_form, persona_stratum; `monotone` must be measured | 2602.11964; 2402.05733 |
| 44 | §3.2 `ToolSpec` | — | Add read/write type, role scope, parameter provenance, duration, occupies_actor, locks | 2602.11964; 2605.18703; 2402.05733 |
| 45 | §3.4 traces source | — | Require rebuild + re-solve; never train on the raw trace (36.7 vs 52.1) | 2609.04148 |
| 46 | §3.10 families × instances | "Many families, few instances" | A target to test per student for RL; SynthAPI shows fewer Apps can win for SFT at about 1,000 tasks; seeds per task are a cheap multiplier | 2610.37648 |
| 47 | §3.10 mix | Nemotron 3 found mixed training less hack-prone | The white paper gives no numbers, ablation or hacking metric; qualitative only | 2512.20856 |

**Gate-numbering clash, resolved.** Extract A proposed G15 (gaming-probe battery), G16 (reward causality) and G17 (solver
agreement) from the Tev card. Extract D independently proposed "G15" for the clean-container regrade and "G16" for
fail-closed grading. This document keeps A's G15–G17 (proposed first, in group 02) and renumbers D's two gates G19
(clean-custody regrade) and G18 (fail-closed). D's G5b stays.

### 14.2 Corrections to the notes

- Note 03 cites AgentPRM (2502.10325) for correct-vs-plausible-wrong step pairs; it builds Monte Carlo Q targets. Only
  ToolPRMBench (2601.12294) builds such pairs.
- Note 03's DuoTrace title is "Cascade Failure Attribution for Multi-Agent Systems" (2608.29646).
- Our notes cite 2601.08276 as "ToolACE-MCP"; its v2 title is ACE-Router.
- Note 04 marked GTM's numbers "not verified". The full text shows its 20K+ tools are LLM-generated schemas and, in search
  and retrieval training, the simulator generates outputs from the task's ground-truth answer (2512.04535).
- If note 02 cites Nemotron 3 for hack resistance of mixed training, it has no measurement (2512.20856).

### 14.3 Inconsistencies inside papers (as flagged by the cards)

| Paper | Inconsistency |
|---|---|
| SWE-smith (2504.21798) | PR Mirror yield 33.8% in the table vs 13.18% in App. C.1 |
| Agent-World (2604.18292) | 14B base τ² 45.3 in text vs 60.2 in Table 2 |
| DreamGym (2511.03773) | "Exceeds 30% success" vs absolute WebArena rates of 9–15% (likely relative) |
| LOGIGEN (2603.00540) | Module costs not stated per environment or per task; L1–L3 tags sum to about 10K of > 20K tasks |
| TaskCraft (2506.10055) | About 36,000 vs about 32K tasks; 43.0% vs 54.9% pass rates not reconciled |
| R2E-Gym (2504.07164) | Decontaminated subset 4,578 vs 4,538 |
| AutoEnv (2511.19304) | 40.67 vs 40.86; environment names differ between tables; one instruction file describes another game; a 138.84% reward called "conservativeness" |
| τ² (2506.07982) | Caption says no critical telecom simulator errors; the table lists 3 (6%) |
| SETA (2607.10891) | All-pass bonus 0.2 in text vs +0.5 in the appendix |
| RST (2608.05466) | 11.82% TB2 relative gain in the conclusion vs 20.0% in abstract and table |
| RHB (2605.02964) | Second commonest exploit category: leakage 24% (main text) vs tampering 19% (appendix) |
| SWE-Factory (2506.10954) | Garbled count fractions in a per-language table; use percentages only |
| NoveltyBench (2504.05228) | Equivalence classifier accuracy 79% / F1 0.811 in the body vs 71.0% in App. A.4 |
| InsTag (2308.07074) | DBSCAN threshold 0.05 under-specified |
| Environment Tuning (2510.10197) | Stage choice on 100 held-in test samples; some gold calls differ from environment returns |
| Reasoning Gym (2505.24760) | Several "hard" configs move a parameter the easy way; an undocumented 0.10 threshold |
| AgentMercury (2608.20634) | 4,783 vs 4,326 Worlds; "five" API models with four rows; 80.7% vs 78.3% |
| EntWorld (2601.17722) | GPT-4.1 at both 47.61% and 37.39%; all tasks are SELECT retrieval despite the abstract; "human 85%" estimated |
| AgenTracer (2509.03312) | Eq. 4 has 0 and 1 swapped; Algorithm 1 is correct |
| AgentDebug (2509.25370) | "Via counterfactuals" in §3.2 vs LLM-only in Algorithm 1; quoted 50.0% / 42.5% vs Table 1's 45.0 / 31.3; κ 0.55 called substantial |
| MAST (2503.13657) | GSM said to have fewer inter-agent failures, yet Table 8 lists it highest |
| SWE-Bench+ (2410.06992) | 5.49% in text vs 3.97% in abstract |
| SpecBench (2605.21384) | About 27 pp per 10× LOC in the body vs 28 in the abstract |
| ToolPRMBench (2601.12294) | 984–987 pairs; two of its three verifier judges are also evaluated models |
| TRAIL (2505.08638) | 118 / 31 vs 114 / 30 traces; its "agreement" is a revision rate |
| Goodhart in RL (2310.09144) | "Surprisingly" important action count with a fitted r = −0.06 |
| DUMP (2504.09710) | A gradient-norm lower bound that does not follow; UCB1 regret under a violated stationarity assumption |
| Kimi k1.5 (2501.12599) | Prioritised sampling ∝ 1 − s weights all-fail prompts that carry no group-relative gradient; ablations without numbers |
| WebDancer (2505.22648) | A quality rule that reads as a typo |
| R-Zero (2508.05004) | Its 3–7 band selects exactly the questions whose majority label is least reliable |
| GUI-Genesis (2602.14093) | 17.35% of unverified Apps kept |
| UTBoost vs SWE-ABS costs | $1.6 per instance and "$80 per strengthened instance" are both consistent ($1.6 × 500 / 10) |
| ZeroGUI card vs takeaway (2505.23762) | "+14–20 from executed checks" is a loose summary of DART +14.61 and ComputerRL's absolute 48.9 |

### 14.4 Evidence we will not rely on, and why

| Evidence | Why not |
|---|---|
| LLM-scored solvability gates (MCP-Bench, StableToolBench) | Model opinion as ground truth; StableToolBench dropped about 30% of tasks by model opinion |
| EvolveCoder's "< 10% pass means the test is wrong" | Deletes the only discriminating checks on hard tasks |
| Kimi k1.5's 1 − s sampling | Contradicts the learnability results |
| Nemotron 3's hack-resistance claim for mixed training | No numbers |
| Policy pass@100 > 0 (DeepSeek-V3.2), o3 pass@16 > 0 (Endless Terminals), pass@16 > 0 (ARPO) | Admit lucky passes and trial-and-error tasks |
| Same-family self-verification loops (DeepSeek-V3.2's agent, OMNI-EPIC's checker, WildAWM's single family, SPARK, UI-TARS-2's ORM) | Circular; each paper lists family separation as the fix |
| Majority-vote pseudo-labels (Agent0, R-Zero) | Label drift 79% → 63%; no single label-accuracy threshold predicts collapse |
| Simulator-judged rewards (UserRL's Persuade and Intention) and satisfaction ratings | Reward the simulator, not the task |
| Policy-written observations (EnvACE); answer-conditioned simulators (GTM) | Rehearsal fidelity unmeasured; leaks the gold |
| Harness-emitted terminal actions (ARPO's FAIL substitution) | Free reward on infeasible tasks |
| Live remote services as reward-bearing World tiers | Drift, no reset, time-dependent results |
| Peak-checkpoint and single-run numbers as effect sizes (FEE, Agent0, SearchEyes, TRACE main tables) | Inflate gains |
| Results trained on the evaluated benchmark or domain (§1.4) | Show a mechanism runs, not that it transfers |
| Inoculation prompting as a defence | Trainer-side, brittle to one verb; fixing the environment beats it |
| Formal unhackability certificates; the provable Goodhart stopping rule | Non-trivial unhackability exists only on finite policy sets; the stopping rule loses 10–44% of true reward and needs an unestimable bound |

---

## 15. Open questions

| # | Question | Why it matters | Next step |
|---|---|---|---|
| 1 | Does the keep rule select tasks that teach? | The whole factory optimises it; SWE-smith found difficulty a poor predictor of SFT value (mean difficulty 2/4/6/8 gave 12.4/10.8/13.6/12.2) | M2 |
| 2 | The band centre per student size (0.2 vs 0.3–0.5) | No paper gives a band schedule per size | R2 arm; M9 |
| 3 | Families vs instances for RL, per student | SynthAPI's SFT result reverses at small budgets; Terminal-Universe favours environments | M3 |
| 4 | Does behaviour-signature Vendi predict transfer for agentic tasks? | Gradient-space Vendi did for math and NLI; embedding Vendi was weaker | M8 |
| 5 | Can a cheap student-side predictor (NLL on the oracle, teacher-forcing loss) replace k = 4 admission? | Measurement dominates cost | Validate against measured p̂ |
| 6 | How much of World density causes transfer, rather than correlates with it? | No standard realism metric; density lowered expert realism in CRMArena-Pro | M13 with controlled family ablations |
| 7 | Can simulated users be made faithful enough to gate reward-relevant behaviour? | Easy mode and blame shifting in every study; the direction is domain-dependent | U2 / U3 on a human or customer reference |
| 8 | Does Tinker expose the student's activations for monitors? | HackTrace's activation monitor and PRIME's probes need hidden states [uncertain] | Check before relying on activation probes |
| 9 | Does RL add much over SFT at our size? | CompoWorld: RL added 1.2 over SFT; WildAWM: RL added long-horizon solving | Report SFT and RL contributions separately |
| 10 | Can the factory reach 500–1000 tasks inside the Tinker credit? | The budget model gives $11k–50k for 1,000 | Measure stage yields on the first two DomainSpecs before scaling |
| 11 | How should hard-edge hacking interact with the controller for the 397B? | Hack rate rises with difficulty, and Goodhart results do not extrapolate across sizes | Re-run G12 and M11 on the 397B |
| 12 | What is the right allowed-change set C_a without hand-writing it? | AppWorld's sets are hand-written and risk false fails | G14 audits of alternatives that fail only on collateral |


---

## 16. References

Every arXiv id cited in this document (303 ids), with its title from the reading cards in `cards/`. All were read
FULL except those marked [PARTIAL]. Ids are sorted numerically.

| arXiv id | Title |
|---|---|
| 2003.08536 | Enhanced POET: Open-Ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions |
| 2010.03934 | Prioritized Level Replay |
| 2012.02096 | Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design (PAIRED) |
| 2012.07805 | Extracting Training Data from Large Language Models |
| 2203.01302 | Evolving Curricula with Regret-Based Environment Design (ACCEL) |
| 2209.13085 | Defining and Characterizing Reward Hacking |
| 2210.02410 | The Vendi Score: A Diversity Evaluation Metric for Machine Learning |
| 2210.10760 | Scaling Laws for Reward Model Overoptimization |
| 2210.14348 | Synthetic Text Generation with Differential Privacy: A Simple and Practical Recipe |
| 2301.09633 | Prediction-Powered Inference |
| 2303.09540 | SemDeDup: Data-efficient learning at web-scale through semantic deduplication |
| 2306.01711 | OMNI: Open-endedness via Models of human Notions of Interestingness |
| 2307.13854 | WebArena: A Realistic Web Environment for Building Autonomous Agents |
| 2308.06259 | Self-Alignment with Instruction Backtranslation |
| 2308.07074 | #InsTag: Instruction Tagging for Analyzing Supervised Fine-tuning of Large Language Models |
| 2309.16797 | Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution |
| 2310.02304 | Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation |
| 2310.09144 | Goodhart's Law in Reinforcement Learning |
| 2310.11667 | SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents |
| 2310.13032 | Quality-Diversity through AI Feedback |
| 2312.03664 | Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia |
| 2401.13649 | VisualWebArena: Evaluating Multimodal Agents on Realistic Visually Grounded Web Tasks [PARTIAL] |
| 2402.03300 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
| 2402.05733 | TimeArena: Shaping Efficient Multitasking Language Agents in a Time-Aware Simulation |
| 2402.05808 | Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning (R³) |
| 2402.13064 | Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models |
| 2402.14992 | tinyBenchmarks: evaluating LLMs with fewer examples |
| 2402.16822 | Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts |
| 2403.01749 | Differentially Private Synthetic Data via Foundation Model APIs 2: Text (Aug-PE) |
| 2403.07714 | StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models |
| 2403.07718 | WorkArena: How Capable are Web Agents at Solving Common Knowledge Work Tasks? |
| 2403.08140 | BAGEL: Bootstrapping Agents by Guiding Exploration with Language |
| 2404.07972 | OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments [PARTIAL] |
| 2404.12272 | Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences ("EvalGen") |
| 2405.00823 | WorkBench: a Benchmark Dataset for Agents in a Realistic Workplace Setting |
| 2405.14573 | AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents |
| 2405.15568 | OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness with Environments Programmed in Code |
| 2406.08464 | Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing |
| 2406.10162 | Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models |
| 2406.12045 | τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains |
| 2406.13542 | Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models (AutoIF) |
| 2406.18403 | LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks ("JUDGE-BENCH") |
| 2406.20094 | Scaling Synthetic Data Creation with 1,000,000,000 Personas |
| 2407.05291 | WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks |
| 2407.18370 | Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement |
| 2407.18901 | AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents |
| 2407.19056 | OfficeBench: Benchmarking Language Agents across Multiple Applications for Office Automation |
| 2408.04682 | ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities |
| 2408.06327 | VisualAgentBench: Towards Large Multimodal Models as Visual Foundation Agents [PARTIAL] |
| 2408.08435 | Automated Design of Agentic Systems |
| 2409.00920 | ToolACE: Winning the Points of LLM Function Calling |
| 2410.02907 | NNetNav: Unsupervised Learning of Browser Agents Through Environment Interaction in the Wild |
| 2410.04444 | Gödel Agent: A Self-Referential Agent Framework for Recursively Self-Improvement |
| 2410.06992 | SWE-Bench+: Enhanced Coding Benchmark for LLMs |
| 2410.10934 | Agent-as-a-Judge: Evaluate Agents with Agents |
| 2410.12952 | Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning (BUTTON) |
| 2411.00640 | Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations |
| 2411.02305 | CRMArena: Understanding the Capacity of LLM Agents to Perform Professional CRM Tasks in Realistic Environments |
| 2411.02337 | WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning |
| 2412.02674 | Mind the Gap: Examining the Self-Improvement Capabilities of Large Language Models |
| 2412.13678 | Clio: Privacy-Preserving Insights into Real-World AI Use |
| 2412.14161 | TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks |
| 2412.21139 | Training Software Engineering Agents and Verifiers with SWE-Gym |
| 2501.10893 | Learn-by-interact: A Data-Centric Framework for Self-Adaptive Agents in Realistic Environments |
| 2501.12326 | UI-TARS: Pioneering Automated GUI Interaction with Native Agents [PARTIAL] |
| 2501.12599 | Kimi k1.5: Scaling Reinforcement Learning with LLMs |
| 2501.12862 | Mutation-Guided LLM-based Test Generation at Meta |
| 2502.01619 | Learning to Generate Unit Tests for Automated Debugging |
| 2502.07709 | MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents in large goal spaces |
| 2502.10325 | Process Reward Models for LLM Agents: Practical Framework and Directions |
| 2502.11357 | Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents |
| 2502.12272 | LILO: Learning to Reason at the Frontier of Learnability |
| 2502.13681 | Repo2Run: Automated Building Executable Environment for Code Repository at Scale |
| 2502.13820 | Scoring Verifiers: Evaluating Synthetic Verification for Code and Reasoning |
| 2502.15840 | Vending-Bench: A Benchmark for Long-Term Coherence of Autonomous Agents |
| 2503.07826 | Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation |
| 2503.11926 | Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation |
| 2503.13657 | Why Do Multi-Agent LLM Systems Fail? |
| 2503.14476 | DAPO: An Open-Source LLM Reinforcement Learning System at Scale |
| 2503.15223 | Are "Solved Issues" in SWE-bench Really Solved Correctly? An Empirical Study |
| 2503.20527 | StableToolBench-MirrorAPI: Modeling Tool Environments as Mirrors of 7,000+ Real-World APIs |
| 2504.01382 | An Illusion of Progress? Assessing the Current State of Web Agents |
| 2504.03380 | Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning |
| 2504.03601 | APIGen-MT: Agentic PIpeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay (xLAM-2-fc-r) |
| 2504.05228 | NoveltyBench: Evaluating Language Models for Humanlike Diversity |
| 2504.05520 | Efficient Reinforcement Finetuning via Adaptive Curriculum Learning (AdaRFT) |
| 2504.07164 | R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents |
| 2504.08942 | AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories |
| 2504.09710 | DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training |
| 2504.11543 | REAL: Benchmarking Autonomous Agents on Deterministic Simulations of Real Websites |
| 2504.21798 | SWE-smith: Scaling Data for Software Engineering Agents |
| 2505.00212 | Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems |
| 2505.03335 | Absolute Zero: Reinforced Self-play Reasoning with Zero Data |
| 2505.05410 | Reasoning Models Don't Always Say What They Think |
| 2505.06120 | LLMs Get Lost In Multi-Turn Conversation [PARTIAL] |
| 2505.08638 | TRAIL: Trace Reasoning and Agentic Issue Localization |
| 2505.14970 | Self-Evolving Curriculum for LLM Reasoning |
| 2505.16282 | ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay |
| 2505.17063 | Synthetic Data RL: Task Definition Is All You Need |
| 2505.18878 | CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions |
| 2505.20161 | Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning |
| 2505.22203 | From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning |
| 2505.22648 | WebDancer: Towards Autonomous Information Seeking Agency |
| 2505.22954 | Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents |
| 2505.23419 | SWE-bench Goes Live! (SWE-bench-Live) |
| 2505.23762 | ZeroGUI: Automating Online GUI Learning at Zero Human Cost |
| 2505.24098 | HardTests: Synthesizing High-Quality Test Cases for LLM Coding |
| 2505.24760 | Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards |
| 2505.24864 | ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models |
| 2506.01716 | Self-Challenging Language Model Agents |
| 2506.03136 | Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning (CURE) |
| 2506.03533 | Go-Browse: Training Web Agents with Structured Exploration |
| 2506.04178 | OpenThoughts: Data Recipes for Reasoning Models [PARTIAL] |
| 2506.07672 | MCPWorld: A Unified Benchmarking Testbed for API, GUI, and Hybrid Computer Use Agents |
| 2506.07982 | τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment |
| 2506.08989 | SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning |
| 2506.09289 | UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench |
| 2506.10055 | TaskCraft: Automated Generation of Agentic Tasks |
| 2506.10954 | SWE-Factory: SWE Data Construction, Automatically! |
| 2506.11045 | Procedural Environment Generation for Tool-Use Agents |
| 2506.12286 | The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason |
| 2506.13131 | AlphaEvolve: A coding agent for scientific and algorithmic discovery |
| 2506.14205 | AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents |
| 2507.02592 | WebSailor: Navigating Super-human Reasoning for Web Agent |
| 2507.02825 | Establishing Best Practices for Building Rigorous Agentic Benchmarks (ABC) |
| 2507.08794 | One Token to Fool LLM-as-a-Judge |
| 2507.13266 | QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation |
| 2507.15061 | WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization |
| 2507.17746 | Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains |
| 2507.20534 | Kimi K2: Open Agentic Intelligence [PARTIAL] |
| 2507.22034 | UserBench: An Interactive Gym Environment for User-Centric Agents [PARTIAL] |
| 2508.01780 | LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools? |
| 2508.03682 | Self-Questioning Language Models |
| 2508.05004 | R-Zero: Self-Evolving Reasoning LLM from Zero Data |
| 2508.06600 | BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent |
| 2508.06647 | Privacy-Preserving Tabular Synthetic Data Generation Using TabularARGN |
| 2508.09123 | OpenCUA: Open Foundations for Computer-Use Agents [PARTIAL] |
| 2508.14040 | ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents |
| 2508.14704 | MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers |
| 2508.17511 | School of reward hacks: Hacking harmless tasks generalizes to misaligned behavior in LLMs [PARTIAL] |
| 2509.03312 | AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems? |
| 2509.08724 | SWE-Mirror: Scaling Issue-Resolving Datasets by Mirroring Issues Across Repositories |
| 2509.08755 | AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning |
| 2509.11106 | Fluid Language Model Benchmarking |
| 2509.13311 | Towards General Agentic Intelligence via Environment Scaling (AgentScaler) |
| 2509.16941 | SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks? |
| 2509.17158 | ARE: scaling up agent environments and evaluations |
| 2509.19736 | UserRL: Training Interactive User-Centric Agent via Reinforcement Learning [PARTIAL] |
| 2509.23124 | Non-Collaborative User Simulators for Tool Agents [PARTIAL] |
| 2509.23866 | Efficient Multi-turn RL for GUI Agents via Decoupled Training and Adaptive Data Curation |
| 2509.24002 | MCPMark: A Benchmark for Stress-Testing Realistic and Comprehensive MCP Use |
| 2509.24726 | Socratic-Zero: Bootstrapping Reasoning via Data-Free Agent Co-evolution |
| 2509.25370 | Where LLM Agents Fail and How They can Learn From Failures |
| 2509.26490 | VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications |
| 2510.01051 | GEM: A Gym for Agentic LLMs |
| 2510.01171 | Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity |
| 2510.01179 | TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments |
| 2510.04340 | Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time |
| 2510.04374 | GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks |
| 2510.05024 | Inoculation Prompting: Instructing LLMs to misbehave at train-time improves test-time alignment |
| 2510.05444 | SimulatorArena: Are User Simulators Reliable Proxies for Multi-Turn Evaluation of AI Assistants? [PARTIAL] |
| 2510.08996 | Saving SWE-Bench: A Benchmark Mutation Approach for Realistic Agent Evaluation |
| 2510.10197 | Don't Just Fine-tune the Agent, Tune the Environment (Environment Tuning) |
| 2510.11977 | Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation |
| 2510.14969 | LLMs as Scalable, General-Purpose Simulators For Evolving Digital Agent Training |
| 2510.18821 | Search Self-play: Pushing the Frontier of Agent Capability without Supervision |
| 2510.19898 | BugPilot: Complex Bug Generation for Efficient Learning of SWE Skills |
| 2510.20270 | ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases |
| 2510.22954 | Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond) |
| 2510.24563 | OSWorld-MCP: Benchmarking MCP Tool Invocation In Computer-Use Agents |
| 2510.24684 | SPICE: Self-Play In Corpus Environments Improves Reasoning |
| 2510.24695 | Scaling Agents via Zone of Proximal Development Data Synthesis (AgentFrontier) |
| 2510.25726 | The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution ("Toolathlon") |
| 2511.01824 | Simulating Environments with Reasoning Models for Agent Training (Simia: Simia-SFT and Simia-RL) |
| 2511.03773 | Scaling Agent Learning via Experience Synthesis (DreamGym) |
| 2511.04703 | Measuring what Matters: Construct Validity in Large Language Model Benchmarks |
| 2511.10395 | AgentEvolver: Towards Efficient Self-Evolving Agent System |
| 2511.18397 | Natural emergent misalignment from reward hacking in production RL |
| 2511.19304 | AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning |
| 2511.21654 | EvilGenie: a Reward Hacking Benchmark |
| 2512.02556 | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models [PARTIAL] |
| 2512.04535 | GTM: Simulating the World of Tools for AI Agents |
| 2512.18552 | Toward Training Superintelligent Software Agents through Self-Play SWE-RL (SSR) |
| 2512.19682 | GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators |
| 2512.20856 | NVIDIA Nemotron 3: Efficient and Open Intelligence |
| 2512.22857 | AutoForge: Automated Environment Synthesis for Agentic Reinforcement Learning |
| 2601.02439 | WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks |
| 2601.04126 | InfiniteWeb: Scalable Web Environment Synthesis for GUI Agent Training |
| 2601.05808 | EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis |
| 2601.08118 | MirrorBench: A Benchmark to Evaluate Conversational User-Proxy Agents for Human-Likeness [PARTIAL] |
| 2601.08276 | ACE-Router: Generalizing History-Aware Routing from MCP Tools to the Agent Web (ACE-Router; cited in our notes as "ToolACE-MCP") |
| 2601.11868 | Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces |
| 2601.12294 | ToolPRMBench: Evaluating and Advancing Process Reward Models for Tool-using Agents |
| 2601.16443 | Endless Terminals: Scaling RL Environments for Terminal Agents |
| 2601.17087 | Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations |
| 2601.17722 | EntWorld: A Holistic Environment and Benchmark for Verifiable Enterprise GUI Agents |
| 2602.00933 | MCP-Atlas: A Large-Scale Benchmark for Tool-Use Competency with Real MCP Servers |
| 2602.06820 | ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training |
| 2602.10090 | Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning |
| 2602.11964 | Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments |
| 2602.14093 | GUI-Genesis: Automated Synthesis of Efficient Environments with Verifiable Rewards for GUI Agent Post-Training |
| 2602.16179 | EnterpriseBench Corecraft: Training Generalizable Agents on High-Fidelity RL Environments |
| 2602.16666 | Towards a Science of AI Agent Reliability [PARTIAL] |
| 2602.21193 | On Data Engineering for Scaling LLM Terminal Capabilities (Nemotron-Terminal / Terminal-Task-Gen) |
| 2603.00520 | SWE-ABS: Adversarial Benchmark Strengthening Exposes Inflated Success Rates on Test-based Benchmark |
| 2603.00540 | LOGIGEN: Logic-Driven Generation of Verifiable Agentic Tasks |
| 2603.10505 | Safe and Scalable Web Agent Learning via Recreated Websites (VeriEnv) |
| 2603.11245 | Mind the Sim2Real Gap in User Simulation for Agentic Tasks |
| 2603.12698 | EvolveCoder: Evolving Test Cases via Adversarial Verification for Code Reinforcement Learning |
| 2603.13594 | EnterpriseOps-Gym: Environments and Evaluations for Stateful Agentic Planning and Tool Use in Enterprise Settings |
| 2603.25001 | Rethinking Failure Attribution in Multi-Agent Systems: A Multi-Perspective Benchmark and Evaluation |
| 2604.05172 | ClawsBench: Evaluating Capability and Safety of LLM Productivity Agents in Simulated Workspaces |
| 2604.05336 | TRACE: Capability-Targeted Agentic Training |
| 2604.10866 | OccuBench: Evaluating AI Agents on Real-World Professional Tasks via Language Environment Simulation |
| 2604.10988 | WebForge: Breaking the Realism-Reproducibility-Scalability Trilemma in Browser Agent Benchmark |
| 2604.15149 | LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking |
| 2604.17739 | Democratizing Tool Learning with Environments Fully Simulated by a Free 8B Language Model |
| 2604.18292 | Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence |
| 2604.20779 | SWE-chat: Coding Agent Interactions From Real Users in the Wild |
| 2604.21579 | A Metamorphic Testing Approach to Diagnosing Memorization in LLM-Based Program Repair |
| 2604.23488 | Do Prompt-Elicited Trajectories Reflect Training-Time Reward Hacking? A Systematic Study on Monitoring Training-Time Reward Hacking in Code Generation ("Trace-and-Amplify") |
| 2605.02964 | Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use |
| 2605.12673 | Do Androids Dream of Breaking the Game? Systematically Auditing AI Agent Benchmarks with BenchJack |
| 2605.12925 | AgentLens: Revealing The Lucky Pass Problem in SWE-Agent Evaluation |
| 2605.13821 | Harnessing Agentic Evolution ("AEvo") |
| 2605.18703 | EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL |
| 2605.20744 | Hack-Verifiable Environments: Towards Evaluating Reward Hacking at Scale |
| 2605.21384 | SpecBench: Measuring Reward Hacking in Long-Horizon Coding Agents |
| 2605.26079 | Automated Benchmark Auditing for AI Agents and Large Language Models (ABA) |
| 2605.26494 | The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence [PARTIAL] |
| 2605.27209 | Learning to Act under Noise: Enhancing Agent Robustness via Noisy Environments |
| 2606.04455 | The Meta-Agent Challenge: Are Current Agents Capable of Autonomous Agent Development? |
| 2606.09711 | Proxy Reward Internalization and Mechanistic Exploitation: A Learned Precursor to Reward Hacking and Its Generalization ("PRIME") |
| 2606.12191 | Agentic Environment Engineering for Large Language Models: A Survey of Environment Modeling, Synthesis, Evaluation, and Application [PARTIAL] |
| 2606.16062 | Auditing Reward Hackability in Code RL Training Environments |
| 2606.17546 | SEAGym: An Evaluation Environment for Self-Evolving LLM Agents |
| 2606.19047 | RODS: Reward-Driven Online Data Synthesis for Multi-Turn Tool-Use Agents |
| 2606.23654 | EnterpriseClawBench: Benchmarking Agents from Real Workplace Sessions |
| 2606.24597 | Qwen-AgentWorld: Language World Models for General Agents |
| 2606.24855 | Data Recipes for Agentic Models (OpenThoughts-Agent) |
| 2606.26300 | The Verification Horizon: No Silver Bullet for Coding Agent Rewards |
| 2606.29537 | OSWorld 2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks |
| 2607.05943 | SearchEyes: Towards Frontier Multimodal Deep Search Intelligence via Search World Simulation |
| 2607.10891 | SETA: Scaling Environments for Terminal Agents |
| 2607.22368 | Do Agent Benchmarks Measure Capability? Protocol Validity in the Age of Agentic AI |
| 2607.26819 | A First Look at Coding Agents' Compliance with AI Contribution Rules in Open-Source Communities ("RepoComplianceBench") |
| 2607.28685 | Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks |
| 2607.28802 | Model or Harness? An Interaction-Centric Taxonomy for Localizing Agent Failures |
| 2608.03744 | Agents Catching Agents: Shortcut Cascades and Benchmark Gaming in Clinical Multi-Agent Systems |
| 2608.05466 | Recursive Synthesis for Long-Horizon Terminal Tasks (RST) |
| 2608.06197 | EnvACE: Internalizing Environment Dynamics via World Rehearsal for Agentic Reinforcement Learning |
| 2608.12635 | GateTruth: Auditing the Rigor of RTL Design Benchmarks via Mutation Testing |
| 2608.14940 | When Is an Agent Evaluation Over? Outcome Finality and Cross-Unit Separation |
| 2608.20634 | AgentMercury: Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale |
| 2608.22103 | Hack-Verifiable Terminal Bench: Evaluating Reward Hacking in Terminal Tasks |
| 2608.27260 | What Makes Good Agentic Data? An ACE Lens on Data Generation for LLM Agents |
| 2608.29646 | Cascade Failure Attribution for Multi-Agent Systems |
| 2608.30724 | BAITBENCH: Measuring Agent Reward Hacking with Optional Shortcuts Planted in ML Tasks |
| 2609.01519 | When Guardrails Look Effective: Construct Validity Failures in LLM Agent Commerce Evaluation |
| 2609.02783 | EarlyEval: Cheaper Agent Evaluation via Early Outcome Prediction |
| 2609.04148 | Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments |
| 2609.05576 | EnvCraft: Synthesizing Executable Environments in Agentic RL for Claw-like Agent |
| 2609.05837 | AgentBrew: Offline Tool-Use Agent Learning from Raw Real-World Trajectories |
| 2609.06780 | Shortcutting the Fix: Identifying and Categorizing Agentic Exploits in Software Engineering Benchmarks |
| 2609.08149 | SWE-Bench Pro Verified: A Reliable Benchmark for Software Engineering Agents |
| 2609.08404 | Environments as Scaffold: Enriching Feedback to Bootstrap Self-Evolving Agents in Long-Horizon Tasks |
| 2609.11028 | BenchShield: Formal Model-Backed Instrumentation for Reward Integrity in LLM-Agent Evaluation Infrastructure |
| 2609.12191 | GAUGE: When Not to Trust LLM-as-a-Judge in User-Simulated Evaluation of Task-Oriented Agents |
| 2609.12439 | Debiasing as a Measurement Intervention: Calibrated Ties and Resolution Loss in LLM-as-a-Judge Evaluation ("TraceJudgeBench") |
| 2609.14998 | Shallow Beliefs: Synthetic document finetuning does not inoculate against emergent misalignment from reward hacking |
| 2609.24972 | RRSI: Regularized Recursive Self-Improvement of Agent Harnesses |
| 2609.27321 | Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms (VHD-Play) |
| 2609.32750 | CUA-Sandbox: Efficient Environments for Computer-Use Agent Reinforcement Learning |
| 2609.33665 | Compositional Environment Scaling for General Agents (CompoWorld) |
| 2609.36887 | WEFT: Scaling Tool-Use Post-Training for General-Purpose Agents |
| 2609.37658 | EnterpriseBench: Benchmarking LLM Agents on Enterprise-Level Strategic Reasoning and Decision-Making |
| 2610.01256 | DeFA: Dependency-Guided Failure Attribution for LLM Agents |
| 2610.03055 | HackTrace: Behavior-Supervised Detection of Reward Hacking During Code Generation [PARTIAL] |
| 2610.04906 | Scaling Verifiable Environments for Long-horizon Work Agents (WorkForge) |
| 2610.06100 | From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation |
| 2610.08048 | Daedalus: Bootstrapping Agent Memory from Self-Generated Tasks |
| 2610.09142 | Finding Side-Effect Loopholes in Verifiable Rewards |
| 2610.11464 | Who Verifies the Verifier? Co-Evolving Inspectable Graders with Self-Improving Agents |
| 2610.12822 | Reward Hacking as a Transmissible Strategy in Multi-Agent Coding Systems |
| 2610.14118 | ClawLoop: Self-Improving Agents via Harness-Native Training Loops |
| 2610.14163 | AutoEnvScaling: Automating the Data Flywheel with Terminal Agents |
| 2610.14774 | ProgAgent: Learning Tool Orchestration through Programmatic Training [PARTIAL] |
| 2610.16397 | Recursive Environment Scaling for General Long-Horizon Agents (RES) |
| 2610.16523 | eau: Environment-Agent-User Co-evolution for Training User-Centric Agents (EAU) |
| 2610.17747 | Auditing Failure Attribution: Label Predictability and Judge Behavior |
| 2610.18868 | Tev: Scaling Long-Horizon Agentic Environments with Dense Progress Rewards |
| 2610.19686 | Interlock: From Local Correctness to Task Completion in Long-Horizon Agents |
| 2610.21656 | Data World: Scaling Data Environments from Scratch |
| 2610.22243 | Shortcut Sensitivity in Agent-Safety Evaluations |
| 2610.23194 | OfficeTown: Generative Organizations for Evolving Agent Intelligence [PARTIAL] |
| 2610.24206 | SWE-Bench Pro V2: A Cleaner, Harder-to-Game Coding Benchmark |
| 2610.26828 | Agent World Model in the Wild: Computer-Use Agent Environment Synthesis |
| 2610.27465 | Recoverability and Verifier Strength: Two Measured Preconditions for Unattended Agent Self-Improvement |
| 2610.27781 | GRADUS: Discovering and Recovering Agent Lessons from the Environment's Latent Syllabus |
| 2610.33149 | WebUniverse: Scaling Verified Training Data for Agentic Web Development |
| 2610.33568 | Trajectory2Gym: Decision-Grounded Compositional Interaction Synthesis for Tool-Use Agents |
| 2610.33817 | EnvLunch: The Environment as a Free Lunch for Learning to Interact |
| 2610.37648 | Scaling Laws for Tool-Use Agents via Controllable Environment Synthesis |
