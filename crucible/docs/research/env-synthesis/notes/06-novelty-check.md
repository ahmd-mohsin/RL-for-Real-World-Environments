# Novelty check: counterexample-guided environment synthesis (C1–C7)

Date: 2026-10-29. Sources: the 337-paper card set (`crucible/docs/research/env-synthesis/cards/group-01..12.md`, `extract-a..f.md`), plus web and arXiv searches. Papers found on the web that are **not in the card set** are marked **[NEW]**. Only the abstracts or HTML were read for those, as noted.

Ratings: **novel** = no close prior art found. **partly novel** = the parts exist, and the combination, target or measurement is new. **known** = already done; drop it or recast it as substrate.

---

## Summary table

| Claim | Rating | Closest prior work | What stays new |
|---|---|---|---|
| C1 closed-world declarative contracts plus one generic grader | **known** (as a mechanism); partly novel only as a cross-substrate contract language | AppWorld 2407.18901 (expected ⊆ Δ ⊆ expected ∪ allowed, verbatim); **[NEW] Agent-Diff 2602.11224** (LLM-written declarative assertions over the canonical diff; any other change fails the task); τ-bench 2406.12045 / LOGIGEN 2603.00540 (exact canonical-state diff); ScaleEnv 2602.06820 (per-column match policies); WebArena/OSWorld JSON evaluator configs; JML `assignable` / Dafny `modifies` | One contract type covering SQL apps, file trees, repos, chat and Gym, with answer and trace constraints. The contract is also the object that gets mutated (C4) and composed (C7). |
| C2 typed falsifier population, per-class Clopper–Pearson certificates, CEGIS loop, counterexample archive | **partly novel** | Side-Effect Loopholes 2610.09142 (edit operators over the oracle's action program, a referee end-state diff, a wrong-task grid); **[NEW] Hacker-Fixer loops 2606.08960** (hacker / fixer / solver loop, defenses shared across tasks, 16% of 1,968 tasks hackable); BenchJack 2605.12673 (attacker-defender iteration, null-agent rule); GateTruth 2608.12635 (mutation-kill floor ≥95%); SWE-ABS 2603.00520; Rajan 2606.16062 (exploit attacker plus a sanity-gated repair loop); ImpossibleBench 2510.20270; EvolveCoder 2603.12698; CEGIS (Solar-Lezama 2006) | One typed population that spans all falsifier kinds. A per-task, per-class certificate with an exact bound used as the admission rule. A monotone counterexample archive replayed as a regression suite. 2606.08960 has **no** regression suite, no statistics, no planted-defect evaluation and no RL. |
| C3 communicability by taint over the oracle program | **partly novel** (the strongest of the "mechanism" claims) | Trajectory2Gym 2610.33568 (deterministic grounding mask: identifier-like arguments must be supported by earlier visible information; used for SFT masking); Interlock 2610.19686 (carrier maps plus a public-support critic, LLM-based); RST 2608.05466 (discoverability rule, prompt-enforced); LOGIGEN (instruction projected from the executed episode); EnvCraft 2609.05576 (an "every expected value derivable" invariant, LLM-checked); SETA 2607.10891; ABA 2605.26079 (instruction axis, LLM audit); BenchShield 2609.11028 (static taint, but over the *reward path*); CaMeL / FIDES / TaintGate (security provenance) | Dynamic taint during oracle re-execution as an *admission gate*. It covers the contract's checked constants as well as the action arguments. It emits a typed verdict ("hidden requirement") and is executed, not judged. |
| C4 second-order mutation of the contract or environment into planted-defect environments, with falsifier recall as the yardstick | **partly novel, close to novel** in this setting | **[NEW] 2607.27518** (UK AISI-style scanners; synthetic violations made by "modifying task environments and prompts", used *only for scanner development*, not for reported metrics); BenchGuard 2604.24955 / ABA (recall on *natural* gold-issue sets of 12–50 items); **[NEW] Auditing Sabotage Bench 2604.16286**, PaperAudit-Bench 2601.19916 (planted defects to score auditors, other domains); LAVA (Dolan-Gavitt et al., S&P 2016: injected bugs as ground truth for bug-finder recall); spec mutation (Budd & Gopal 1985; Black, Okun & Yesha 2000); oracle assessment (Jahangirova et al., ISSTA 2016 / OASIs 2018); mutation analysis as a yardstick for test generators (Andrews et al. ICSE 2005; Just et al. FSE 2014); Mills' error seeding (1972); μPRL 2408.15150 (mutating RL agents) | A typed operator set over *task contracts and environments* (drop a required effect, widen the frame, leak the answer, green-on-start, writable grader path, remove instruction information). It is used as a locked, label-free benchmark of a *validation pipeline* for RL environments, with per-class recall. No paper found does this for agent RL environments. Do **not** call it "higher-order mutation": that term (Jia & Harman 2009) means multi-fault program mutants. |
| C5 evaluator-side recursive self-improvement (new operators from counterexamples, birth gate, cross-domain transfer) | **partly novel** | Who Verifies the Verifier 2610.11464 (new typed detector ops born from clustered misses; a **birth gate** that requires firing on ≥ half the cluster and staying clean on known-good outputs; anchors; a locked test set); AutoRedTeamer 2503.15754 and AutoDAN-Turbo 2410.05295 (lifelong attack-strategy libraries); Rainbow Teaming 2402.16822; 2606.08960 (defenses transfer across tasks, "no universal defenses"); **[NEW] VerifierContractBench** (NeurIPS 2026: optimized verifiers regress on protected unsafe cases); RRSI 2609.24972; Recoverability 2610.27465 | The unit is a *falsifier* (it executes and produces a counterexample), not a detector. The birth gate requires recall on planted defects of **other tasks and domains** and no false alarms on certified anchors. Acceptance never reads student reward. A cross-domain transfer hypothesis, tested on a locked planted-defect set. 2610.11464 already owns "birth gate", anchors, locked test and "downstream score cannot certify a verifier", so cite it as the template. |
| C6 online certification during RL | **partly novel** | Verification Horizon 2606.26300 (trajectory monitor plus token penalty in training); PRIME 2606.09711 (patching one exploit mid-training makes the policy retarget within ~15 steps); **[NEW] 2606.04923** (an agent scans training rollouts for hack onset); HackTrace 2610.03055; EvolveCoder / CURE 2506.03136 (tests evolved from policy behaviour); **[NEW] 2609.35677** (theory: training records alone cannot identify accepted errors, while independent audits give "selective control"); **[NEW] LessWrong, Aug 2026** (pause training on a buggy environment, patch, resume; untested); 2606.08960 lists inline-RL as future work | Every passing rollout is a falsification trial that feeds the *task's certificate*. A flagged pass quarantines the task and starts repair, rather than penalizing the policy. Certificates tighten during training. 2609.35677 gives the theoretical motive: an independent falsifier is the audit that the theory says is needed. |
| C7 compositional contracts for long horizons | **known** in its main form | τ²-Bench 2506.07982 (atomic init/sol/assert subtasks composed, verified, with "not resolved before all sol steps"); WorkArena++ 2407.05291 (the composite oracle calls atomic oracles in order; validation per subtask or global); RHB 2605.02964 (chained tasks; downstream needs upstream artifacts, enforced by hashes); BUTTON 2410.12952; TaskCraft 2506.10055; CompoWorld 2609.33665; RES 2610.16397 / RST 2608.05466; ToolVerse 2607.15660; CoDE (Gur et al., NeurIPS 2021); Hoare-logic sequential composition | Only an algebra of frame and certificate composition: when the composite's validity certificate follows from the components'. That needs a theorem or a measured result to count. Otherwise demote C7 to a difficulty knob. |

---

## C1. Declarative closed-world task contracts

**Prior art found**
- **AppWorld (2407.18901, group-09).** Grading is a hash-based DB diff. A task passes iff C^expect ⊆ D^Δ and D^Δ ⊆ C^expect ∪ C^allow. The paper names this the collateral-damage check. 15% are QA tasks with an answer check. This is C1's formula verbatim, but per-scenario evaluation code is still hand-written.
- **Agent-Diff (2602.11224) [NEW].** LLMs (Claude Opus 4.5, Gemini 3 Pro) write declarative assertions over the canonical state diff: diff type, entity, `where` constraints, `expected_count`. "Any other insertion, deletion, or mutation is treated as a side effect and causes the task to fail", with declared ignore fields. It covers Box, Linear, Slack and Calendar, and is an evaluation benchmark only (no RL). This is the closest prior art for "LLMs write contract data, not verifier code, with a closed world".
- **τ-bench (2406.12045), LOGIGEN (2603.00540).** Full canonical end-state equality. That is closed-world with an empty allowed set, path-independent, minus auto ids and timestamps.
- **ScaleEnv (2602.06820).** Declarative per-column match policies (exempt / hard / semantic).
- **Side-Effect Loopholes (2610.09142).** Names the frame-condition gap and recommends closed-world frame checks by construction. The card itself already proposed this for Crucible.
- **τ² (2506.07982).** Assertions as DB functions.
- **Interlock (2610.19686).** Typed effect predicates ("for every object in Ω where α holds, φ") as data, judged by an agent.
- **WebArena / OSWorld / AndroidWorld.** JSON evaluator configs interpreted by a shared evaluator library.
- **Formal specification.** Frame conditions (JML `assignable`, Dafny `modifies`, Z/B frames; Borgida, Mylopoulos & Reiter 1995, "the frame problem in procedure specifications").

**What stays new.** One contract type across substrates (SQL apps, repos and file trees, mail and chat, Gym), with answer checks and trace constraints in the same object, interpreted by one trusted grader. The contract is the shared object for C2–C4 and C7. That is engineering unification, not a research contribution.

**Rating: known as a mechanism.** **Phrasing:** "We adopt AppWorld's closed-world diff semantics (as also used by Agent-Diff) and generalise it into a typed contract over any canonical-state environment. The contract is the substrate on which falsification, mutation and composition act." Do not list C1 as a contribution. Present it as the design that makes C2–C4 possible. A cross-substrate demo (at least 3 substrates, same grader) is still worth showing.

---

## C2. Typed falsifier population, statistical certificates, CEGIS repair, counterexample archive

**Prior art, by falsifier class**
- **Oracle-program mutation.** Side-Effect Loopholes 2610.09142: Claim-Only, Omit, Partial, Wrong-Value, Extra/Extra-NI, Extra-Field, plus a wrong-task grid with one-parameter twins. A referee end-state diff, separate from the verifier, classifies each edit. A repair test checks that the patched verifier rejects the edit and still passes the correct and natural runs. It needs no model calls and uses hand-designed edits, and it reports existence proofs only, no rates. Also AgenTracer fault injection (group-08), GateTruth's generic mutant operators, SWE-ABS's mutation-driven test strengthening, Meta ACH 2501.12862 and SWE-smith's AST operators.
- **Null / degenerate agents.** BenchJack's checklist ("run a null agent; any task scoring > 0 has a V6 flaw"), AutoEnvScaling (empty solution < 0.5), SETA no-op, the SynthAPI / τ² two-sided filter.
- **Alternative solvers.** EvolveCoder (a 64-solution pass matrix), CodeT, PatchDiff 2503.15223, UTBoost 2506.09289.
- **Adversarial attackers.** BenchJack, Rajan 2606.16062 (iterative attacker plus a gold-sanity-gated repair loop), **Hacker-Fixer loops 2606.08960 [NEW]** (Zhong, Raghunathan et al., Jun 2026). Its hacker / fixer / solver loop shares defenses through a git pool across tasks and found 323 of 1,968 terminal tasks hackable. Per its HTML it has **no regression suite of past exploits, no statistical guarantees, no seeded-vulnerability meta-evaluation, and no RL use**.
- **Impossible / wrong-task twins.** ImpossibleBench 2510.20270, the wrong-task grid in 2610.09142, EnterpriseOps infeasible tasks.
- **Metamorphic / distractor perturbation.** IPT 2604.15149 (isomorphic renaming), 2604.21579, Shortcut Sensitivity 2610.22243, Saving SWE-Bench 2510.08996.
- **Honeypots.** HVE 2605.20744, HVTB 2608.22103, BaitBench 2608.30724, RHB 2605.02964.
- **Certificates.** GateTruth certifies a testbench at a ≥95% kill rate (a point estimate, no bound). Side-Effect Loopholes gives a rank-order break-even α. Recoverability 2610.27465 gives σ = MCC plus a refuse rule. Crucible's own G10 uses a Clopper–Pearson flake bound. "What AI Red-Team Evaluations Can and Cannot Prove" **[NEW] 2607.21735** formalises what a clean red-team run supports (evidential ceiling, zero-event Clopper–Pearson). No paper found issues per-task, per-falsifier-class bounds as an admission rule.
- **CEGIS.** Solar-Lezama et al. 2006. LLM variants exist in planning, regex and access control (AutoCedar 2607.03656), but none for environment synthesis.

**What stays new**
1. One typed population that holds all of the falsifier classes above, with a common counterexample record.
2. A per-task, per-class certificate (trials, confirmed counterexamples, exact upper bound) as the admission criterion.
3. A monotone archive replayed as a regression suite on every repair. This closes the retargeting and regression problem that 2606.08960 does not address and that VerifierContractBench measures.

**Caveat to state up front.** The bound is relative to the falsifier distribution, not to all possible defects. The trials are not i.i.d. draws from "all exploits". Say "falsifier-relative certificate" and pair it with C4 recall, so the falsifier distribution is itself measured.

**Rating: partly novel.** **Phrasing:** "We unify previously separate validity checks (mutation, null agents, alternative solvers, attackers, twins, metamorphic tests) into a typed falsifier population. We admit tasks by falsifier-relative statistical certificates, and repair them in a CEGIS loop whose counterexample archive is a monotone regression suite." Cite 2610.09142 and 2606.08960 prominently as the closest prior art.

---

## C3. Communicability by taint

**Prior art**
- **Trajectory2Gym (2610.33568, group-02). The nearest.** Its local eligibility v_t requires that "identifier-like arguments [are] supported by preceding visible information". The check is deterministic over logs, and the student context excludes private events. It is used to mask SFT targets on logged trajectories, not to admit tasks. It is string-level, covers arguments only, and does not cover the verifier's constants.
- **Interlock (2610.19686).** "Every obligation in H must be supported by q or the public workspace". It uses carrier maps and an LLM critic that sees only (q, W0, V). Coverage comes from LLM review, not execution.
- **RST (2608.05466) D.1.** "Every reward-changing condition is stated or discoverable". It is enforced by prompts and gates, not by tracing.
- **LOGIGEN.** Projects the instruction from the executed episode, which is communicable by construction but has no check.
- **EnvCraft (2609.05576).** A data-consistency invariant ("every expected value derivable from built files"), LLM-checked.
- **SETA (2607.10891).** Its all-fail case comes from test literals that the instruction never states. ClawLoop's answer contract and WebUniverse's failed-test review also belong here.
- **ABA / BenchGuard.** "Instruction" defects found by LLM audit. Verification Horizon's `instruct_clear` judge, and RandomWorld's privileged solver check.
- **Taint machinery outside this use.** BenchShield (static taint over the reward path, for integrity, not communicability); CaMeL 2503.18813, FIDES, TaintGate and "Ghost in the Agent" 2604.23374 (provenance of tool arguments for prompt-injection security); ToolFuzz 2503.04479; LAVA (dynamic taint used to *inject* bugs); classic dynamic taint (Newsome & Song 2005; TaintDroid 2010). Known weak point: transformed or paraphrased values break exact taint. You need derivation rules (arithmetic, formatting, aggregation) or the false-positive rate will be high.

**What stays new.** Dynamic taint during **oracle re-execution** as an executed admission gate. It covers both the action literals and **the constants the contract checks**, sourced from the instruction, earlier observations or start-state reads, and gives a typed "hidden requirement" verdict.

**Rating: partly novel (the best mechanism claim).** **Phrasing:** "Prior work checks communicability by LLM review (Interlock, ABA, RST) or masks unsupported arguments in logged trajectories (Trajectory2Gym). We make it an executed property of the task: every literal the oracle uses and every constant the contract checks must be taint-derivable from the instruction or from agent-visible observations." **Must show:** precision and recall against human labels or planted "remove information" defects (C4), and how it handles transformed values.

---

## C4. Second-order mutation into planted-defect environments, with falsifier recall as the yardstick

**Prior art**
- **2607.27518 [NEW] (Mohl et al., Jul 2026).** Naturally generated transcripts had few violations, so the authors "systematically modified task environments and prompts to produce failure modes", e.g. gold solutions placed in the environment, across 4 criteria. The data was used **only for scanner development**: "not used in the final evaluation". This is the closest LLM-agent precedent. Cite it and differentiate: it covers 4 coarse criteria, is dev-only, has no operator taxonomy, no locked set and no recall yardstick.
- **BenchGuard 2604.24955 and ABA 2605.26079.** Auditor recall on *natural* confirmed-issue sets (12 issues; 50 tasks). The sets are small, and recall carries wide uncertainty.
- **Auditing Sabotage Bench 2604.16286 [NEW] and PaperAudit-Bench 2601.19916 [NEW].** Planted defects to score auditors in ML codebases and in papers.
- **Classic SE.** LAVA (S&P 2016) injects bugs as ground truth for bug-finder recall. Mutation analysis is an accepted yardstick for comparing test generators (Andrews, Briand & Labiche ICSE 2005; Just et al. FSE 2014: mutants as substitutes for real faults). Specification mutation goes back to Budd & Gopal 1985 and Black, Okun & Yesha 2000. Oracle assessment (Jahangirova et al. ISSTA 2016; OASIs 2018) uses mutation to expose false negatives of test oracles and generated tests to expose false positives. That is "testing the tests", which C4 lifts one level to "testing the validation pipeline". Mills' **error seeding** (1972) uses seeded faults to estimate the faults that remain.
- **Within agent RL.** ImpossibleBench mutates *tests* to measure *agent* propensity, not validator recall. GateTruth uses an always-pass testbench as a sanity control, which is a single planted defect. μPRL 2408.15150 mutates RL *agents* and scores test generators by mutation score.
- **Terminology.** "Higher-order mutation" (Jia & Harman 2009) means a mutant with several faults. "Second-order" in C4 means the mutation is applied to the *task contract and environment* instead of the oracle solution. Use **"contract mutation"** or **"meta-mutation"** to avoid a reviewer collision.

**What stays new.** A typed contract and environment mutation operator set targeting validity defects of RL environments: drop a required effect, widen the allowed frame, relax a tolerance, leak the answer, green-on-start, a writable grader path, removed instruction information. It yields planted-defect environments with exact ground truth, scored as per-class falsifier recall on a locked set. It serves both to evaluate validation pipelines and as the fitness signal for C5. No paper found does this for agent environments.

**Rating: partly novel, close to novel** for the setting. The concept descends from spec mutation and LAVA and should be credited to them. **Phrasing:** "We bring specification mutation and fault seeding (Budd & Gopal; LAVA; Jahangirova et al.) to RL environment validation. Contract mutation operators create planted-defect environments with known ground truth, and per-class falsifier recall on a locked set measures any validation pipeline." **Threat to address:** realism of planted versus natural defects ("mutants as valid substitutes", Just et al.). Validate on natural defects found by C2, BenchGuard-style gold sets, or Terminal Wrench (2606.08960 release, 323 hackable environments).

---

## C5. Evaluator-side recursive self-improvement

**Prior art**
- **Who Verifies the Verifier (2610.11464, group-06; NeurIPS 2026).** Senses misses, clusters them under a closed taxonomy, and has an LLM author a typed op per cluster. Its **birth gate** requires the op to fire on ≥ half its cluster and stay clean on known-good outputs. It uses anchored dev items, a locked test set, a fail-closed rule, and an outer audit that is never a training signal. Key finding: downstream score cannot certify an evolved verifier, and without anchors it collapses to always-pass. This is the template for C5's birth gate. Its limits: one model family, three task families, ops that are *detectors* over outputs, and no cross-domain transfer test.
- **AutoRedTeamer 2503.15754 (NeurIPS 2025).** A lifelong attack library, with new attacks proposed from research papers and memory-guided selection. **AutoDAN-Turbo 2410.05295.** A strategy library discovered by self-exploration. **Rainbow Teaming 2402.16822.** A QD archive of attacks.
- **2606.08960.** Fixer patches pool across tasks, and "every line in the final verifier code was written by a fixer working on a different task". That is transfer of defenses, not of falsifiers, and it finds no universal defenses.
- **VerifierContractBench [NEW] (NeurIPS 2026 listings).** All 26 meta-agent-edited verifiers newly certify ≥1 protected unsafe trajectory, and instance-preserving gates lower but do not remove this. It directly motivates the "no false alarms on certified anchors" and monotone-archive gates.
- **RRSI 2609.24972, Recoverability 2610.27465, DGM, ADAS.** Guards for self-improvement loops.

**What stays new.**
1. The improved artifact is a **falsifier** (an executable generator of counterexamples), and the birth gate is execution-based.
2. The birth gate requires reproducing its source counterexamples, **recall on planted defects in other tasks and domains**, and no false alarms on certified anchors.
3. Acceptance is independent of student reward.
4. An explicit, testable **cross-domain transfer hypothesis** on a locked planted-defect set.

**Rating: partly novel.** **Phrasing:** "Extending gated detector growth (2610.11464) and lifelong attack libraries (AutoRedTeamer) to RL environment validation, we grow typed falsifiers from confirmed counterexamples. A birth gate demands cross-domain recall on planted defects and zero false alarms on certified anchors, and never consults student reward." The transfer result is the novelty. If operators do not transfer, the claim weakens to "a 2610.11464-style loop in a new setting".

---

## C6. Online certification during RL

**Prior art**
- **In-training hack monitors.** Verification Horizon 2606.26300 (pattern-set trajectory monitor plus token penalty), HackTrace 2610.03055, 2606.04923 [NEW] (an LLM agent finds hack onset in training logs, for rubric RL), Baker et al. 2503.11926 (CoT monitors, obfuscation risk), IPT's hacking gap tracked live.
- **Patching mid-training.** PRIME 2606.09711: after one exploit is patched, the policy moves to the next open one within about 15 steps. Patch whole classes.
- **Theory.** 2609.35677 [NEW] (Moya, Thornley & Lin, Sep 2026): observations available during RLVR are "in general insufficient to detect or identify accepted errors". Independent audits give selective control.
- **Tests that co-evolve with the policy.** EvolveCoder, CURE 2506.03136, SPARK 2509.22624.
- **Pause, patch and resume** an environment (LessWrong post, Aug 2026, untested). 2606.08960 lists inline-RL as future work.

**What stays new.** Passing rollouts as falsification trials (independent falsifiers re-check passes), feeding the **task's certificate**. Quarantine plus repair acts on the task and environment rather than on the policy's reward, so certificates tighten over training. Report it with retargeting measurements (PRIME) and class-level patches.

**Rating: partly novel.** **Phrasing:** "Following the result that training records alone cannot identify accepted errors (2609.35677), we audit passing rollouts with independent falsifiers. Each audit is a trial on the task's certificate, and a confirmed counterexample quarantines and repairs the task, so certificates tighten during training." **Must show:** the cost per audited pass, the false-quarantine rate, and the hack-rate trajectory versus a monitor-only baseline.

---

## C7. Compositional contracts

**Prior art**
- **τ² 2506.07982.** Atomic (init, sol, assert) subtasks are composed by picking ≤1 per group and concatenating, then verified: apply init and sol, all asserts true, and not resolved before all sol steps. That is solvable by construction (2,285 composed tasks).
- **WorkArena++ 2407.05291.** The composite oracle calls the atomic oracles in order, and validation is per subtask or a global DB query.
- **RHB 2605.02964.** Chains of length 2–7 where downstream steps need upstream artifacts, enforced by recomputed hashes. This is C7's "later tasks require artifacts from earlier ones".
- **Others.** BUTTON (sequential / parallel-then-sequential composition), TaskCraft (depth and width extension), CompoWorld (product environments over services), RES and RST (recursive transformation operators), Interlock (coupling hyperedges), ToolVerse 2607.15660, CoDE (NeurIPS 2021, compositional environment generation with a curriculum). Hoare-logic sequential composition and design-by-contract for the formal side.

**What stays new.** Only a contract algebra: frame composition (how allowed and required sets combine when step 2 may overwrite step 1's effects), plus a **certificate-composition result** (when component certificates bound the composite's validity, and which falsifier classes must re-run on the composite). A student-relative difficulty knob with measured non-additivity would add a little.

**Rating: known** in its main form. **Action:** drop it as a headline contribution, or recast it as "certificate-preserving composition" with a theorem or measured result (e.g., composite false-pass rate ≤ the sum of the components' bounds plus the interface-defect rate, then measured).

---

## Additional angles that would make the paper clearly novel

1. **Recall-calibrated certificates (C2 × C4).** Combine falsifier recall measured on planted defects with the per-class zero-counterexample bounds. That gives an estimate of the *remaining* defect rate per domain, an error-seeding / capture–recapture estimate in the style of Mills 1972. No agent-environment paper reports how many defects remain undetected. This turns "we found N hacks" into a calibrated statement and answers the falsifier-relative caveat.
2. **A dose–response RL experiment.** Train the student (Qwen3.6-35B-A3B) on pools admitted at different certificate strengths, or with and without the archive and online certification. Measure the onset of hacking, the clean-versus-raw pass gap and transfer. Rajan 2606.16062 and 2606.08960 both state that they never show verifier hardening helps RL. This experiment would be first-of-kind evidence and would lift C2, C4 and C6 together.
3. **Cross-substrate transfer as the central hypothesis.** Use the same contract type, grader and falsifier operators on at least three substrates (SQL apps, repos, Gym/continuous control). Report operators born in one domain catching planted defects in another (C5). 2606.08960 reports "no universal defenses", so positive transfer of *falsifiers*, where defenses did not transfer, is a crisp, falsifiable finding.

(A fourth, smaller option: release the locked planted-defect set as a benchmark for environment-validation pipelines. BenchGuard and ABA validate on 12–50 natural issues, so a larger ground-truth set fills a measured gap.)

---

## Claims to drop or reframe
- **C1:** reframe as substrate (AppWorld and Agent-Diff already do it).
- **C7:** drop, or recast as certificate-preserving composition with a result.
- **C4 naming:** avoid "second-order / higher-order mutation". Use "contract mutation".
- **C2 statistics:** always say "falsifier-relative". Never imply a bound over all defects.

## Search coverage notes
- Keyword greps over the cards (mutation, counterexample, falsif, taint, communicab, planted, closed-world, frame, side effect, Clopper, certif, CEGIS, second-order) found the AppWorld, 2610.09142, Trajectory2Gym, Interlock, RST, 2610.11464, τ², RHB and WorkArena++ cards cited above. "counterexample" and "CEGIS" are almost absent from the cards, which supports C2's CEGIS framing as new in this literature.
- The web searches added: 2606.08960, 2602.11224, 2607.27518, 2604.24955, 2604.16286, 2601.19916, 2609.35677, 2606.01066 (verifier fuzzing for RLVR: adversarial completions against buggy versus stricter verifiers; abstract only), 2606.04923, 2607.21735, 2607.02586, VerifierContractBench, AutoRedTeamer, LAVA, OASIs, and spec mutation. **None of these is in the card set.** Read 2606.08960 and 2602.11224 in full before submission. They are the two papers most likely to be cited against C2 and C1.
- Unverified details: AutoDAN-Turbo's id (2410.05295) and CaMeL's id (2503.18813) come from memory. VerifierContractBench has no arXiv id (NeurIPS 2026 listings only).
