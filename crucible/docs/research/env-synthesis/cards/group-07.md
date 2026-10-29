# Group 07 cards: verifier testing, benchmark validity, leakage, judges, failure attribution

**Papers: 29. Read FULL: 29 (all from arXiv HTML full text; in a few, results tables or long appendices were only skimmed, and the card says so). PARTIAL: 0. ABSTRACT-ONLY: 0.**

**Five cross-paper takeaways for Part 2**

1. **The World's state diff should referee every negative, and the side-effect class is the one that survives.** Planted missing or wrong work was always caught; unrequested changes (a repeated write, an extra field) passed shipped verifiers that match records by key or compare sets (2610.09142). Build F5 negatives as edits of the oracle's action program (claim-only, omit, partial, wrong-value, repeat, extra-field, wrong-task twin), classify harmless repeats by an idempotence replay, and default verifiers to closed-world frame checks. Run the G5 kill gate the corrected way: baseline-first refusal, non-compiling mutants excluded, other-channel kills separated, sequential certification, a held-out mutant set, and a record of every check revised "to pass the gate" (Goodhart; 2608.12635). Concern-targeted mutants (2501.12862), "list the wrong solutions, then break each" generators (2505.24098) and pass-matrix split requests (2603.12698) are the ways to strengthen checks. New checks must always pass gold and alternatives.

2. **Repository tasks leak through four channels, and memory cannot be sandboxed.** Agents shortcut in 55–91% of trajectories; with the network off they switch to memory, which succeeds in up to 34% (2609.06780). Paired runs show a 21-point drop once git history, hidden files, metadata and code hosts are closed (2609.08149). In-sandbox grading accepts forged binaries and module caches (2610.24206). For the debug lane: grade the saved diff in a fresh container, allow only the model endpoint during the agent phase, use single-commit repos with hashed ids, audit trajectories with a three-family exploit judge (upstream, local git, memory, other), and probe memorisation (file path from the report alone; verbatim prefix completion; behaviour-preserving rewrites of the start code) (2506.12286, 2604.21579, 2410.06992).

3. **Validity means the intended skill stays necessary, and synthetic generators leak in their own ways.** Use Expose → Exploit → Mislead with a measured inflation gap G (2607.22368). Seeds, row orders, ids and positions can give the answer away; planted faults carry surface cues a one-step classifier finds 95–100% of the time, and judges fine-tuned on such labels learn the cue (2610.17747, 2610.22243). Add gates for generator regularity and planted-fault cues. Keep the INVALID / INCONCLUSIVE decision rule and test simulated personas with incentive checks and scripted controls (2609.01519). Label failures by edge and fault side so owner, grader, tool and environment faults go to task repair, not to the on-intent count (2607.28802). Fail-closed grading (I5) and a separate grader environment remove most structural vectors (2609.11028).

4. **Model judges are hackable and can be mis-anchored even when they agree with humans.** "Thought process:" or ":" fools frontier judges up to 80–90% of the time (2507.08794). Fine-tuning a judge for accuracy made it easier to hack in RL (2505.22203). Satisfaction carries no information about task success, and judge rankings fail on near-equal candidates (2609.12191). Rules: state checks first, model fallback only on rule rejections; prefer discriminative judges; show the judge the reference and response only (for answers) or the tool calls and goal (for process); run a fixed master-key and isomorphic re-grade probe before any judge enters a VerifierBundle; in training, monitor proxy reward against an invariant or stronger oracle per task family (2604.15149).

5. **A pass is not evidence of skill, and labels should come from replay where possible.** About 11% of passing SWE trajectories are lucky passes (trial and error, partial fixes on weak tests), concentrated in a few tasks (2605.12925). LLM judges localise the decisive step only 8–14% of the time (2505.00212), but the World can compute it by restoring a snapshot and substituting the oracle's action. Repair loops need the guards these papers converge on: repair the instruction, never the check; validate an instruction edit by a blind re-implementation; accept a patch only if a held-out set improves, else backtrack (2610.24206, 2502.01619). Estimate per-task difficulty with nested replicates; single-run "hard" selections shrink on re-run (2609.01519).

---

## 2410.06992 — SWE-Bench+: Enhanced Coding Benchmark for LLMs (Aleithan et al., 2024; arXiv)
Read: FULL.  Relevance to Part 2: medium. It is an audit of a patch-graded benchmark. It gives the two failure classes (solution leak, weak tests) that F6 G5/G9/G11 must catch in any task, and a manual-validation protocol, but no automated method.
**Problem.** Do agents that "resolve" SWE-bench issues really fix them, or do they copy hints from the issue thread or slip past weak tests?
**Method / algorithm.**
1. Take the 286 instances SWE-Agent+GPT-4 marked resolved on SWE-bench Full. Keep 251 whose eval logs show every FAIL_TO_PASS and PASS_TO_PASS test passing.
2. Three authors, independently, compare model patch with gold patch: files changed, lines changed, approach, plus the issue text, hints_text (issue comments, which SWE-bench feeds to models) and trajectories. Disagreements resolved in group discussion.
3. Label each pass with one of six patterns: suspicious (solution leak; incorrect fix; different files or functions changed; incomplete fix) or correct (different but correct fix; more comprehensive than gold).
4. Repeat the leak screen on all of SWE-bench Lite and Verified.
5. Build SWE-Bench+: the same SWE-bench scripts (attribute filter: resolves an issue and adds tests; execution filter: installs and gold PR passes), on 11 of the 12 repos (Django dropped: tracker left GitHub), issues dated 2023-11-01 to 2024-08-22 (a month after the latest model cutoff, GPT-4o Oct 2023), then a manual screen drops every issue whose report or comments state the fix. Result: 548 instances.
6. Run four agent systems on SWE-Bench+ and redo the manual patch validation on their passes.
**Verification / grading.** SWE-bench's own FAIL_TO_PASS / PASS_TO_PASS tests, then human patch review against gold. The hack surfaces are exactly the two found: hints that hold the answer, and tests too weak to tell a partial, wrong-location or wrong-semantics patch from the real fix.
**Scale & cost.** 251 + 54 + 112 passes reviewed by hand by three people. SWE-Bench+ runs: SWE-Agent+GPT-4 $3.59 per instance and ~4 min (≈37 h total); AutoCodeRover+GPT-4o $0.46 and 4.5 min (41 h); RAG runs ~30 s. They also report cost per resolved issue ($655 for SWE-Agent+GPT-4; note the text and table disagree on which row is which).
**Key results.**
- SWE-bench Full, SWE-Agent+GPT-4 passes (251): solution leak 32.67%, incorrect 12.75%, wrong files 3.59%, incomplete 14.74%; correct-but-different 30.27%, more comprehensive 5.98%. Resolution 12.47% → 5.49% counting only correct (the abstract's 3.97% figure is not reconciled in the text).
- Lite: 18% → 9.33%; Verified: 22.4% → 10.0%; leak share about 33% in both, so Verified's human screen did not remove leaks.
- 94% of SWE-bench issues predate the model cutoffs.
- SWE-Bench+ (no leak): weak-test passes remain about 67.72% of passes on average, mostly "wrong file/function changed". True rates: 0.55–3.83% (AutoCodeRover+GPT-4o 21 correct of 42 passes).
**Failure modes & limits.** Labels come from manual comparison with one gold patch, so "different file" may sometimes still be a valid fix (they count "different but correct" as correct, but the line is subjective). One agent family is studied in depth. Cutoff-date filtering only handles the models of that time. No automated detector is offered. The numbers are internally inconsistent in places (12.47 → 3.97 vs 5.49).
**Tips & tricks.**
- Treat the hint channel (issue comments) as part of the instruction for leak checks, not only the title and body.
- Review passing patches, not failures: weak tests show up only as passes.
- A pass that edits different files than the reference is a weak-test signal worth a second check.
- Date-filter tasks after the student's cutoff, then still screen for leaks, since freshness does not fix weak tests.
- Report cost per correct solve alongside pass rate.
**Fit to Crucible Part 2.** F6 G11 (leak): the leak rule must scan every text the solver sees (instruction, hints, changelog, comments) for the fix itself; our report lint and changelog strip are the generalised form. F5/G5: the "different location still passes" pattern is a negative class to plant (a fix in the wrong unit that makes the visible symptom vanish) and a reason to require per-subtask checks. F10: add a "pass audit on a sample" step: for each kept task, review a few frontier passes against the oracle diff or state delta; a pass whose touched set is disjoint from the oracle's flags weak checks. Interface: `PassAudit{task_id, rollout_id, touched_vs_oracle_overlap, label∈{correct, alt_correct, leak, weak_check}}`. Priority P1 (the mechanism is already in G5/G11; the pass-audit sampling is the new bit).

## 2610.09142 — Finding Side-Effect Loopholes in Verifiable Rewards (Abrich, 2026; arXiv, OpenReview submission)
Read: FULL (body and appendices A–M).  Relevance to Part 2: high. It is the direct template for F5's side-effect negative class and G5's per-class kill rule over state-based Worlds like ours (AppWorld and WorkArena are close cousins of Crucible's Apps).
**Problem.** Verifiers check the postcondition (requested change made) but often not the frame condition (nothing else changed). A run can do the task, add an unrequested change, and pass. Audits that compare score or reward cannot tell it from a correct run.
**Method / algorithm.** Mutation testing with the verifier as the test suite and a correct run as the program.
1. Inputs per task: resettable start state s0, a correct run (shipped solution, replayed as an API-call program or GUI driver), the verifier V_t pinned and unmodified.
2. Edits (one per edited run), designed by reading verifier and driver source:
   - Missing or wrong work: Claim-Only (no write, just claim); Omit (drop last write); Partial (keep only the first of a repeated write sequence); Wrong-Value (move one written value to another legal value); GUI: No-Submit, Omit-Field, Omit-Divergent (omit a field whose default differs from the requested value).
   - Unrequested change: Extra (repeat the last write); Extra-NI (Extra restricted to writes whose repeat changes per-table write counters, found by an idempotence replay); Extra-Field (GUI: fill a field the task never names with a non-default value).
   - Wrong-task test: grade task j's correct run with task i's verifier, at four distance levels: one-parameter twin, same template sibling, other template with shared tables, other template with disjoint tables.
3. Replay each edited run from a fresh start through the benchmark's normal actions only.
4. End-state check before the verifier call, with queries separate from the verifier's code. Effect = typed diff of added/removed/updated records after dropping a declared set N of meaningless fields (ids, timestamps, auto-numbers); two effects are equal when the multisets agree. In AppWorld they used only per-table write counters (a lower bound).
5. Call V_t. Classify each run exactly once: refused (edit cannot apply), delivery incomplete, invalid (correct run failed, integrity breach), harmless edit (pass, end state correct; an equivalent mutant), rejected, unconfirmed candidate, false positive (pass, end-state check shows the named wrong change).
6. Repair test: patch a copy of the verifier; the edited run must now fail and its correct run must still pass; also check against natural agent runs for false rejections.
7. Process: freeze protocol, enumeration code, pins and manifest in a commit before scored runs; integrity checks (driver and verifier share no mutable object; hash what the verifier expects at four stages from reset to validation); report the full flow for every edit, never pool edits, give no rate without a sampling frame. A sealed holdout of 14 AppWorld templates stays unopened.
Six preconditions (App. L): resettable isolated starts with shared per-episode randomness; a passing correct run; programmatic access to the action channel; a deterministic verifier callable unmodified and as a patched copy; an independent end-state check that sees every edit; a way to pin or hash what the verifier expects.
**Verification / grading.** Ground truth for "wrong" is the end-state diff against the correct run, not the verifier. What gets hacked: verifiers that match new records by key into a dict (a second request to the same receiver merges), compare a queue as a set (duplicate song invisible), or rely on a page-local monitor flag that the post-submit page load erases.
**Scale & cost.** No model calls at all. AppWorld: 48 tasks (16 templates × 3), 5 edits; wrong-task grid 2,294 runs at ~2 s each (~80 min). WorkArena: 13 templates × 3 configs; wrong-task grid 468 runs over ~11 h (one episode at a time, ~71 s per run); paired experiment 168 episodes; per-step experiment 84 episodes. Harness code is large: 2,642 + 1,335 lines (WorkArena), 1,340 + 931 + 5,177 lines (AppWorld).
**Key results.**
- Missing or wrong work: 0 false positives (AppWorld 168/168 rejected; WorkArena 84/93 rejected, the 9 passes harmless because the default equals the requested value).
- Wrong task: 0 of 2,689 valid off-diagonal runs accepted, including 118 one-parameter twins.
- Unrequested change: AppWorld Extra passed 39/48; 33 harmless (repeat changed nothing); 6 false positives, all 3 versions of 2 templates (payment requests; song queue). In 11 of 16 templates a repeat is idempotent; of the 5 where it changes data, tests reject it in 3.
- WorkArena form verifier called once after submit: passed 36/36 valid extra-field pairs. Per-step validation (BrowserGym style): extra field filled first → reward 0 before submit (11/11); drop-down last → 0 (4/4); text or reference field last → reward 1 (7/7).
- Repairs: one count assertion per template rejected 6/6 duplicates, passed 6/6 correct runs and all 13 natural agent runs that the original tests accepted. A flag-keeping verifier copy rejected 12/12 and passed 12/12.
- Reward impact: three RL methods (all-tests binary, fraction of tests, all-tests + c × progress) give the duplicate the correct run's maximum reward for any c > 0. None of 1,277 released passing runs makes the duplicate, so the loophole is latent.
- Rank-order break-even: with false-positive rate α, order i over j survives iff α < (q_i − q_j)/(1 − q_j); AppWorld pairs have median break-even 0.18 (normal) and 0.10 (challenge).
**Failure modes & limits.** Existence proofs only: hand-written edits, two benchmarks, no rates; AppWorld end-state check counts writes and never compares values; no policy trained, so whether RL finds the loophole is open. The author's own runner had a shared-config bug (driver overwrote the verifier's expected serial number) that inflated an earlier result and was found only by reading code, so harness integrity checks are essential. My critique: edits are designed by a person who read the verifier, so recall on unknown loophole classes is unmeasured; the automated-proposer study is announced, not reported.
**Tips & tricks.**
- Make the end-state diff (multiset of added/removed/updated records minus a declared ignore set N) the referee for every negative, separate from the verifier.
- Plant Extra (repeat last write) on every task; run an idempotence replay first so harmless repeats are classified, not counted.
- Use the wrong-task grid with one-parameter twins as a cheap specificity test for every verifier.
- Count assertions on created records close most duplicate-write loopholes.
- Never match created records by key into a dict, or compare collections as sets, when multiplicity matters.
- Check in-page or in-process monitors survive the grading call; per-step and end-only grading can disagree.
- Hash the verifier's expectations at reset, mid-run and validation; deep-copy configs between driver and verifier.
- A pass with a correct end state is harmless (equivalent mutant); do not count it as a verifier defect.
- Freeze the edit manifest before scoring; keep a sealed template holdout for rate estimates.
**Fit to Crucible Part 2.** F5 negative builder: adopt the edit catalogue as generic operators over the oracle's action program (Claim-Only, Omit, Partial, Wrong-Value, Extra/Extra-NI, Extra-Field) since our oracle is a tool-call program over policy-visible tools, which makes the edits mechanical. F6 G5: require the side-effect class (already in the plan) and add the idempotence replay so a repeat that changes nothing is "harmless", not "survived". Add a G5b wrong-task grid: each task's verifier must reject its one-parameter twin's oracle run, which reuses F3's sibling tasks and costs only CPU. Make the World's state diff the shared referee (`StateDelta{added, removed, updated}` minus `DomainSpec.ignore_fields`), and default verifiers to closed-world frame checks (whole delta vs gold delta) so the frame condition is checked by construction rather than per task. G2 should also hash the verifier's expected values at each stage. F8/F12: a loophole holding across a template is systematic, so verifier defects must be fixed per template, not per instance. Interfaces: `NegativeOp{kind: claim_only|omit|partial|wrong_value|extra|extra_field|wrong_task, target_call_idx}`, `NegativeOutcome∈{refused, incomplete, invalid, harmless, rejected, candidate, false_positive}`. Priority P0 (cheap, no model calls, closes a class we know survives).

## 2608.12635 — GateTruth: Auditing the Rigor of RTL Design Benchmarks via Mutation Testing (Bhadra, 2026; arXiv)
Read: FULL (body §1–§9 and limitations; leaderboard tables skimmed).  Relevance to Part 2: medium-high. The domain (Verilog) is outside our lane, but the paper is a careful account of how a mutation-kill gate on a test suite fails in practice, which is exactly what G5's "kill floor on generic mutants" needs to get right.
**Problem.** RTL benchmarks treat their testbench as ground truth and never ask whether it would catch a broken design. A pass@k number looks the same whether the testbench is rigorous or vacuous.
**Method / algorithm.**
1. Reference suite: 60 spec-to-RTL tasks (tiers of 20/25/15) and 8 agentic-repair tasks, each a 7-part package: original prose spec with a canary GUID, locked port interface, maintainer-reviewed reference RTL, testbench split into public smoke and hidden scoring vectors, formal properties where applicable (46/60), a per-task clock target, metadata with two sign-off fields. Drafts were written by AI coding assistants and reviewed by the single maintainer.
2. Scoring: correctness as a strict gate (lint → hidden-vector simulation → bounded model checking); any failure scores 0, so PPA cannot reward truncated logic. Then PPA = geomean(ref_area/area, ref_delay/delay, ref_power/power), score = 100·min(ppa,1.5)/1.5. Repair track adds bounded sequential equivalence (depth 20, 60 s) for behaviour-preserving objectives and disqualifies any diff to six immutable files (tests, properties, constraints, metadata, baseline).
3. Mutation engine: structural generator that masks comments and strings and enumerates every viable site for a fixed generic operator set (comparator boundary flip, operator/logic/bitwise inversion, shift-direction flip, reset/enable polarity flip, assignment deletion, register hold, output inversion). The seed sets only execution order.
4. Corrected protocol (after two self-found errors): (a) run the unmodified reference through the full pipeline first and refuse to score mutants if it fails (a missing hidden-vector mount had produced a false 100% kill with exit 0); (b) non-compiling mutants are stillborn, excluded from numerator and denominator; (c) kills from formal properties only are recorded as formal_only, not counted toward the simulation-test rate; (d) a timeout gets one extended-budget retry, then counts as indeterminate against the kill rate; (e) run sequentially only, because parallel load turns borderline mutants into "killed by timeout"; (f) re-run a seeded sample of designs and require byte-identical output.
5. Floor: a testbench is certified if it kills ≥95% of valid mutants. A vacuous always-pass testbench scores 0% as a sanity control.
6. External audit of RTLLM v2.0 at a pinned commit, read-only; designs whose reference fails baseline validation are reported "unsupported", never scored 0%; a documented module-name alias is applied at compile time only and recorded per row.
**Verification / grading.** Kill = behavioural simulation failure of a compiling mutant. Survival is not equivalence: no equivalent-mutant detection, so rates are a lower bound. What was hacked internally: a hand-written per-task "these mutants are equivalent" exclusion table (72 excluded; 36 were in fact killed by the hidden testbench), and a blended metric that counted lint and formal kills as test kills.
**Scale & cost.** Own suite: 1,245 mutants, 96 stillborn, 1,149 valid, 1,105 killed, 44 survived (pooled 96.17%), 4 formal-only. RTLLM: 46 audited designs, 775 mutants, median 10.5 per design (1–115). Repair-track spend cap $300 per campaign. Model costs in table form (not material here).
**Key results.**
- Own suite: 46/60 testbenches clear 95% (first claimed 60/60, then 49/60). The 14 failures have three causes: a Goodhart effect on 10 testbenches revised to pass the gate (pre-revision median 77.5%; 7 of the 10 still fail once the exclusion trick is removed), the unsound equivalence exclusion, and formal kills blended in.
- RTLLM v2.0: median per-design kill 74.0%, pooled 56.8%; 33/46 (72%) below the floor; 3 designs at 0%, two of which miss an inverted primary output. Kill rate falls with mutant count (Spearman −0.39): the largest third of designs median 61.4%. One golden reference fails its own testbench.
- Survival varies by operator: 80% of bitwise inversions and 69% of logic inversions survive, 12% of output inversions.
- CVDP cannot be audited: references are withheld, so no golden design to mutate.
- An output-token cap (4,096 vs 16,384) moved one model from fifth to first (pass@1 28 → 43); a lint-warning gate caused about half of all zero scores, varying 2.4× across models.
**Failure modes & limits.** Single maintainer; reference and tests drafted by one model family that is also evaluated; the signed-provenance schema was added after the certification run; small mutant counts make a single survivor fail the floor; operator mix drives the pooled rate. The paper is candid that tuning tests until they pass a mutation gate is itself Goodhart-prone.
**Tips & tricks.**
- Gate the mutation run on a clean baseline pass in the exact official mode; otherwise a setup failure reads as 100% kill.
- Exclude non-compiling mutants from both numerator and denominator.
- Never accept hand-reasoned "equivalent mutant" exclusions; test them against the real suite or use formal equivalence.
- Count timeouts as indeterminate against the rate, and certify only from sequential runs.
- Keep a held-out mutant set unused during test repair (matches G5's held-out rule) and record which tests were revised for the gate.
- Report pooled and per-item kill rates and survival by operator.
- Report "unsupported" separately from 0%.
- Hidden scoring vectors plus public smoke tests reduce memorisation of tests.
- Report and vary the output-token budget; it can reorder a leaderboard.
**Fit to Crucible Part 2.** F6 G5 kill floor: adopt the corrected protocol verbatim as the generic mutant runner over check suites (baseline-first refusal, stillborn exclusion, formal-only or "other-channel" kills separated, sequential certification, indeterminate timeouts, byte-identical sample rerun). F5 repair loop: the Goodhart finding argues for the held-out mutant set already in G5 and for logging each verifier revision made "to pass the gate", with the pre-revision kill rate kept in provenance. F9/§3.6: record per-run output-token cap in the DifficultyRecord; never compare students under different caps. Interface: `MutantRun{op, site, status∈{killed, survived, stillborn, indeterminate, other_channel_kill}}`, `KillReport{pooled, per_task, by_op, baseline_ok}`. Priority P1.

## 2501.12862 — Mutation-Guided LLM-based Test Generation at Meta (Foster et al., 2025; FSE Companion '25, industry track)
Read: FULL.  Relevance to Part 2: medium-high. The concern-to-mutant-to-test pipeline is a ready recipe for F5's negative builder and for hardening verifiers against a named failure class; the equivalence-detector numbers tell us how much to trust an LLM to filter planted faults.
**Problem.** Generate unit tests that guard against a specific class of fault (privacy regressions), starting from vague text about the concern, and give hard assurances about each test.
**Method / algorithm.** ACH (Automated Compliance Hardener); one model, Llama 3.1 70B, in all three agents; prompts are short and simple.
1. Input: free text about an issue of concern (past faults, requirements, regulatory text) plus an example diff of a past fault.
2. Make-a-fault agent: given the concern, the class under test and its existing test class, rewrite the class so that every method carries a typical bug of that concern, delimited by MUTANT START/END comments.
3. Rule-based filters: the mutant must build and must pass the existing tests (so it is currently uncaught). Drop mutants syntactically identical to the original; strip any comments the generator added, then drop those that become identical.
4. Equivalence-detector agent (LLM judge): "do both versions always behave the same? yes/no, and if no, explain how they differ." Uses weak mutation (local state change counts, propagation to output not required).
5. Make-a-test agent (modified TestGen-LLM): given original class, mutant, and existing tests, extend the test class with tests that fail on the mutant and pass on the original.
6. Assurances checked by rules: test builds; passes consistently on the original (non-flaky, repeated runs); kills a mutant no existing test kills. Coverage gain reported if any. Relevance and style are left to human code review.
7. Stop per class once one mutant builds and passes (use-case choice: find a relevant test first).
8. Ship as ordinary diffs into code review; the diff summary names the mutant the test kills as an example.
**Verification / grading.** Execution decides build, pass, and kill; only equivalence uses an LLM. The current code is the regression oracle, so ACH cannot find existing bugs (oracle problem). Hack surface: the generator inserts "misleading comments" claiming a bug while changing no code, which fools the equivalence judge.
**Scale & cost.** 10,795 Kotlin classes on 7 platforms; 31,677 mutants generated, 9,095 (29%) build and pass, 2,246 (25% of those) syntactically identical, 4,660 judged non-equivalent; 571 tests produced (kill on 5.3% of classes). No $ given.
**Key results.**
- Engineers accepted 73% of 191 reviewed tests (WhatsApp 56%, Messenger 90%); 36% judged privacy relevant; initial trial 27/30 accepted.
- 49% (277/571) of mutant-killing tests add no line coverage, so a coverage-only criterion would discard them; TestGen-LLM (coverage-targeted) kills 2.4% of mutants vs ACH 15%.
- Ground truth on 381 sampled mutants: 36% equivalent; 61% of equivalents contain only an added comment. Equivalence agent alone: precision 0.79, recall 0.47 (0.97/0.44 if "unsure" is treated as non-equivalent). With identical-code and comment-stripping pre-filters: precision 0.95, recall 0.96.
- LLM mutants have higher equivalent rate (~25% trivially identical) than rule-based (10–15%).
**Failure modes & limits.** Relevance is weak: mutants often belong to the general fault class, not the specific one, and there is no similarity metric. Results are one language and one company. The detector's good numbers reflect bimodal mutants (obvious change or comment-only), not skill at program equivalence. Pre-screening tests by guessed relevance was abandoned because engineers valued off-topic tests.
**Tips & tricks.**
- Seed the mutant generator with a concrete past-fault diff and a concern description, not a generic "introduce a bug".
- Require mutants to build and to pass the current suite: only surviving mutants are useful.
- Strip comments the generator added before any equivalence or kill check.
- Lexical identity check before calling an LLM judge.
- Prefer precision over recall in equivalence filtering; a missed equivalent wastes compute, a false "equivalent" loses a test.
- Use mutation kill, not line coverage, as the adequacy criterion for added tests.
- Report the killed mutant alongside each generated test for human review.
**Fit to Crucible Part 2.** F5 negative builder and the verifier-hardening loop: given a `DomainSpec` concern ("duplicate write", "wrong record updated", "PII leaked to wrong user"), generate concern-targeted mutants of the oracle (or of the start-state code in the debug lane), keep the ones that survive the current verifier, then ask a different-family writer for a check that kills them while gold still passes. This is a concrete loop for G5 that attacks the verifier where it is weak instead of with generic operators. The comment-stripping and identity filters apply directly to our planted bugs (a planted fault that only adds a comment is a no-op task). Debug lane: the "misleading comment" finding is a leak and no-op risk for authored faults, worth a lint in F6. Interface: `ConcernSpec{text, example_diffs}` in `DomainSpec`; `Mutant{diff, concern_id, survives_current_verifier, equiv_verdict}`. Priority P1.

## 2604.15149 — LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking (Helff et al., 2026; arXiv; the note calls it "Isomorphic Perturbation Testing")
Read: FULL (short paper plus appendices A–C).  Relevance to Part 2: high for the verifier-testing idea. It gives a cheap, generic metamorphic check (re-grade the same answer on an isomorphic copy of the task) and causal training evidence that a verifier missing an invariance gets hacked.
**Problem.** RLVR-trained models on inductive logic tasks output enumerations of instance labels instead of general rules. A verifier that checks only extensional correctness (the hypothesis labels the given examples right) passes them.
**Method / algorithm.** Isomorphic Perturbation Testing (IPT).
1. Task T = (background B, positives E+, negatives E−) from SLR-Bench (train/car ILP tasks, 20 complexity levels in 4 tiers, 250 tasks per tier).
2. The model produces one hypothesis H.
3. Extensional verification: H with B entails all E+ and no E− using the original object ids.
4. Isomorphic verification: apply a bijective renaming Φ of object constants only (train0→t1, car0→c1), keeping attribute constants (red, short) fixed, to get T^Φ; check H on T^Φ.
5. Shortcut iff H passes extensional and fails isomorphic. Shortcut rate = N_S / N_tasks.
6. Training test: two Olmo-3-7B-Think-DPO runs with the Olmo-3 RLVR recipe (~500 steps, 64 H100, ~48 h each), identical except reward = extensional vs isomorphic verifier. Track hacking gap r_ext − r_iso.
**Verification / grading.** Logic-program entailment, deterministic. Hacks found: blatant enumeration (list each positive as a fact) and obfuscated enumeration (a rule body that disjoins over specific object ids, so it looks like a rule).
**Scale & cost.** 1,000 tasks per model, one pass each. GPT-5 run 9.4M tokens, $103; training 2 × 64 H100 × 48 h.
**Key results.**
- Non-RLVR models (GPT-4o, GPT-4.5, Ministral) show 0 shortcuts; RLVR-trained ones do (GPT-5-nano: 37/147/184 shortcuts in Easy/Medium/Hard of 250 each; GPT-5-mini-high 1/23/59; GPT-5 3+1).
- Across models, 40 shortcuts in levels 1–10 vs 458 in levels 11–20.
- GPT-5-mini reasoning effort low/medium/high: 0 → 32 → 84 shortcuts.
- Olmo-3 32B: 0 shortcuts; Olmo-3.1 (longer RLVR) shows some.
- Training: with the extensional verifier, rewards diverge near step 250 and the gap reaches about 3.5 (max reward 10) by step 500; with the isomorphic verifier the gap stays near zero.
**Failure modes & limits.** One domain (ILP trains); RLVR status of closed models is presumed; training at 7B only; the perturbation covers id-renaming only, so other shortcuts (e.g., memorised attribute patterns) stay unseen. Short workshop-style paper with few ablations.
**Tips & tricks.**
- Re-grade one output on a semantics-preserving copy of the task; disagreement flags a shortcut, with no extra rollout.
- Rename identifiers, keep the meaningful attributes fixed.
- Expect shortcuts to grow with difficulty and with reasoning budget; audit hard bands harder.
- Smaller students hack more on hard tasks (relevant to a 35B MoE).
- Track the hacking gap (proxy reward minus invariant reward) during training as a live hack monitor.
- Watch for obfuscated hacks that look like valid structure.
**Fit to Crucible Part 2.** F5/F6: add a G-gate "isomorphic re-grade" for any answer-type check: the World can re-key entity ids (customer, invoice, ticket numbers) under a bijection, replay the student's action program or answer against the re-keyed World, and require the verdict to hold. An answer that hardcodes ids seen in the start state, or a check that accepts such answers, is exposed. This is cheap because Crucible's seeder (F2) is deterministic and can emit Φ-renamed twins. F12: log r_proxy − r_invariant per task during training as a hack-flag signal; tasks with a growing gap go back to F5. F9: hack risk rises in the hard band, so the frontier audit sample should be weighted toward hard tasks. Interface: `SeedBundle.isomorph(Φ)`; `GateResult{iso_consistent: bool}`; `FeedbackSignal.hacking_gap`. Priority P1 (P0 for answer-only tasks).

## 2604.21579 — A Metamorphic Testing Approach to Diagnosing Memorization in LLM-Based Program Repair (De Koning et al., 2026; arXiv)
Read: FULL.  Relevance to Part 2: medium. Our debug lane mines real repository bugs that the student may have seen; this gives a cheap contamination probe (re-run on a semantics-preserving rewrite of the start code) and the transformations that expose memorisation best.
**Problem.** LLM program-repair scores on old benchmarks may come from memorised fixes. Post-cutoff benchmarks age quickly. Can semantics-preserving rewrites, combined with NLL, expose leakage?
**Method / algorithm.**
1. Datasets: Defects4J (854 bugs, old) and GitBug-Java (199 bugs from 2023; 44 multi-function bugs dropped); function-level bugs only.
2. CodeCocoon applies nine natural transformations chosen from a human-rated naturalness study: rename function, parameter, variable (synonyms and acronyms from GPT-4o-mini, context aware), for→while, reverse if (negate and swap branches), nest else-if, swap equals operands, swap relational operands (a>b → b<a), expand unary increment. Each applied wherever possible. Test names and stack traces are updated by regex for function renames.
3. Repair prompt (SRepair-like, single step): buggy function, Javadoc, one random trigger test and its stack trace; chain-of-thought root cause then patch. Function-level localisation, not line-level.
4. 10 independent prompts per bug per version; closed models give 5 patches per prompt, open models 1. A prompt succeeds if any patch compiles and passes all tests. SR = successful prompts / prompts; SR_diff = SR_trans − SR_orig, computed on bugs solvable in the original (SR > 0).
5. Stats: Wilcoxon signed-rank, Vargha–Delaney Â12; Spearman between per-model NLL percentile (from Ramos et al.'s replication package, extended to all of Defects4J) and SR_diff on the lower half of NLL; permutation test (1M iterations, up to 3-way interactions) for which transformations matter.
**Verification / grading.** Test-adequate patches (plausible, not checked for true correctness). The transformations are validated by unit tests and manual inspection to preserve behaviour.
**Scale & cost.** About 1,000 original and 1,000 transformed bugs × 7 models × 10 prompts. Costs not reported.
**Key results.**
- Defects4J, all 7 models drop significantly (p < 0.001): Claude-3.7 77.3% → 72.4% (−4.9), GPT-4o −4.0, GPT-4o-mini −5.8, Gemma-2-27B −8.3, Llama-3.1-8B −16.0 (Â12 0.11, large), Mistral-7B −6.9, StarCoder2 −7.8. Worst cases: bugs solved 10/10 drop to 0/10.
- Drop is largest on "easy" bugs (original SR 70–100%), medians beyond −20% for weak models.
- GitBug-Java: no significant average drop for Claude-3.7 (+0.3) or GPT-4o (+1.8), but single bugs fall 100% → 30–70%.
- NLL link: the lowest-NLL bin has the largest drop; Spearman r = 0.30 (Gemma), 0.17 (Llama), 0.22 (Mistral), 0.40 (StarCoder, p = 0.08). An inverted U suggests a second, instability effect at high NLL.
- Nest-else-if is the most telling transformation (Claude and GPT-4o); renaming hurts mainly in combination with structural changes; Gemma is sensitive to renaming alone.
- Probing only the lowest 20–25% by NLL already exposes most inflation.
**Failure modes & limits.** Plausible-patch metric; transformations may still look less natural; NLL needs open weights; Java only; transformation counts scale with function size (confound acknowledged). Their test-renaming by regex may miss references.
**Tips & tricks.**
- Run a rewritten twin (renames plus a structural rewrite such as nest-else-if) of a mined bug; a big drop on a bug the model solves easily flags memorisation.
- Combine lexical and structural rewrites; lexical alone is weak.
- Keep transformations natural (human-rated), or the drop measures distribution shift, not memory.
- Report worst-case per-instance drops, not only means; leak shows up on single tasks.
- Where weights are open (the Qwen student), NLL of the original file is a cheap ranking signal for which tasks to probe first.
**Fit to Crucible Part 2.** F10 intent audit, "instruction only / memorisation" probe: for mined tasks, add a metamorphic twin probe (start tree rewritten by behaviour-preserving transforms; the oracle patch is re-applied through the same transforms, and the gates re-run). Student pass on original ≫ pass on twin marks the task as memorised; such tasks should go to the benchmark only after obfuscation or not at all. The Qwen3.6 student has open weights, so its NLL on the start files can rank which tasks to probe (F9 cheap predictor). F11 knob `obfuscation` in the DomainSpec (already listed) can be implemented as exactly this transform set, which also gives difficulty variants. Avoid applying renames that break the leak lint or the report text. Interface: `MetamorphicTwin{task_id, transforms, oracle_replayed_ok}`; `IntentProbe.metamorphic_drop`. Priority P1 for the debug lane, P2 for synthetic Worlds (where seeds are fresh anyway).

## 2505.24098 — HardTests: Synthesizing High-Quality Test Cases for LLM Coding (He et al., 2025; arXiv)
Read: FULL (body, prompts in A.2, training details A.6, oracle-free A.7).  Relevance to Part 2: medium-high. It is the clearest recipe for building a verifier from an oracle: generator programs written by an LLM, a validator, a multi-oracle agreement rule, and per-output-category generators; plus RL evidence that verifier precision changes training outcomes.
**Problem.** Synthetic tests for hard coding problems let well-disguised wrong solutions pass (TACO's false positive rate exceeds 90% on hard problems). Human tests are proprietary for about 80% of problems.
**Method / algorithm.** HardTestGen, all LLM content by GPT-4o with 2–3 hand-crafted in-context examples and chain-of-thought.
1. Filter problems: needs at least one running oracle program and stdin/stdout I/O; drop "core logic" (function-signature) problems.
2. Prompt (spec + oracle program) for `validate_input(str)->bool` checking every stated constraint, especially "it is guaranteed that..." clauses, within reasonable complexity; plus, when needed, `output_judging_function(input, candidate, reference)` for multi-answer problems (25.4% need one).
3. Inputs, three types (spec + oracle + validator in prompt):
   - Type 1 Direct: n_D = 10 small inputs imitating samples (≤300 chars).
   - Type 2 Regular: an LLM-written `gen_regular_input()` called n_R = 20 times; for multi-category outputs (Yes/No, −1 vs value), one generator per output category, each called 10 times, so an always-"No" program cannot pass. No rejection-sampling while-loops; construct valid inputs directly (e.g., generate the permutation then derive the array).
   - Type 3 Hacking: the LLM first lists plausible naive or brute-force candidate programs, then writes m_H generators, one per candidate, aimed at timing it out or hitting an edge case; each called 10 times. Skip if the hack is just "maximum size".
4. Drop inputs the validator rejects.
5. Outputs: run up to 8 oracle programs (most reliable sources first); if two oracles agree on >90% of inputs, keep the agreeing portion as references.
6. Oracle-free variant (A.7, based on ALGO): an LLM writes a brute-force exhaustive solution; 10 edge-input generators with small values; outputs from brute force; plus one maximum-size input that only must finish before timeout.
**Verification / grading.** Test suite as binary classifier vs ground truth (official tests on AtCoder; actual Codeforces submissions). Precision = TP/(TP+FP), recall = TP/(TP+FN). Hack surface they close: inefficient-but-correct programs (30% of LLM wrong programs are TLE vs 14.9% of human wrong programs), output-class imbalance.
**Scale & cost.** 47,136 problems from 13 judges; tests generated for 32.5k; 81.9% success. Failures: no valid oracle 6.62%, output verification failed 5.85%, input generation failed 3.72%. RL: Qwen3-4B, GRPO in verl, 8 H100, global batch 32, 32 samples per prompt, response length 24k, lr 5e-7, no KL, ~5k problems, best checkpoint within 100 steps.
**Key results.**
- Average +11.3 pp precision and +17.5 pp recall over existing tests; AtCoder difficulty 4+, Qwen2.5-Coder-7B programs: TACO precision 21.7 / recall 68.4 vs HardTests 60.0 / 94.7.
- Ablation: adding Type 2 and Type 3 to Type 1 raises precision by 2–48 pp at ≤2.5 pp recall cost; Type 1 alone (like typical LLM test writers) has precision as low as 10.4 on hard problems.
- Precision gain is larger for weaker program sources (7B models).
- RL (Qwen3-4B on LiveCodeBench-105): base 38.5/52.0/56.2 (pass@1/5/10); RL with TACO tests 37.0/51.0/57.1 (worse at pass@1); RL with HardTests 39.4/57.9/64.8.
- Self-distillation: training on verified-correct own traces beats random or incorrect traces; teacher distillation (R1 traces) benefits more from more questions than from filtering by tests (46.6k unfiltered beats 13k filtered).
- Oracle-free on 165 AtCoder problems × 50 solutions: FPR 17.7% vs AceCoder 32.5%, FNR 2.2% vs 2.6%.
**Failure modes & limits.** Needs oracle programs; single-file stdin/stdout only; still below human tests; one generator model (GPT-4o); RL run is short (≤100 steps) and on one small model.
**Tips & tricks.**
- Have the LLM write generator programs, not raw inputs; validity holds at scale.
- Always write an input validator first and pass it into the generator prompt.
- Ask for the likely wrong or naive solutions first, then target each with a hacking generator.
- Balance generators per output category so constant answers fail.
- Require agreement between independent oracles before trusting a reference output.
- Treat "maximum size, must finish" as a separate test type for efficiency.
- Measure a verifier by precision and recall against a ground-truth judge on many sampled programs, sliced by difficulty and by program source.
- For distillation, more questions beat stricter filtering; for RL and self-distillation, verifier precision matters.
**Fit to Crucible Part 2.** F5 verifier writer and negative builder: port the pattern as World-level generators. (a) The "validator first" step becomes a start-state validator for F2 seeds and for negative states. (b) "List likely wrong solutions, then build an input that breaks each" is the generic prompt for the negative builder: enumerate plausible wrong action programs (the shortcut, the partial, the wrong-entity) and build a seed variant where each one ends in a wrong state. (c) Per-output-category generators map to answer-kind balance (a task family whose answers are mostly "no change needed" lets a do-nothing agent win; G4 null agents catch this per task but the generator should balance per family). (d) Multi-oracle agreement maps to G14 alternatives: two oracles from different families must agree on the end state. Verifier evaluation in G5 should report precision and recall on a pool of sampled frontier and student rollouts labelled by the World diff, by difficulty band. Interface: `VerifierEval{precision, recall, by_band, by_source}`. Priority P1.

## 2603.12698 — EvolveCoder: Evolving Test Cases via Adversarial Verification for Code Reinforcement Learning (Ruan et al., 2026; arXiv)
Read: FULL (body and prompt appendices A.1–A.3).  Relevance to Part 2: medium-high. The pass-matrix-driven loop (drop saturated tests, pick disagreeing solutions, ask for tests that split them, dedupe by pass vector) is a reusable pattern for F5's verifier repair and for difficulty shaping, but the paper's correctness controls are weak.
**Problem.** Coding RL datasets have static, weak, redundant tests and skewed difficulty, so rewards are noisy and advantages vanish. Can tests be evolved from how candidate solutions actually behave?
**Method / algorithm.**
1. Seeds: TACO, APPS, PrimeIntellect SYNTHETIC-1, Codeforces, CodeContests (75,084 problems). Semantic dedup: all-mpnet-base-v2 embeddings, cosine > 0.9 keeps one at random (→ 30,437; mean 10-NN similarity 0.79 → 0.70).
2. Refinement (AceCoder recipe): GPT-4.1-mini rewrites each (problem, reference program) into a hard LeetCode-style function-signature problem with 20 assert tests (constant values, independent).
3. Candidate pool: 8 open reasoning models (Qwen3 4B/8B/14B/30B-A3B/32B, MiMo-7B-RL, Phi-4-Reasoning, R1-Distill-Qwen-32B) × 8 samples = 64 solutions per problem. Build binary pass matrix M (solutions × tests).
4. Adversarial generation (each round): drop tests passed by nearly all solutions; pick the 2 highest-pass solutions and 3 with maximal pairwise Hamming distance on pass vectors; prompt the generator with problem, tests, these 5 programs and their per-test results to write 20 new asserts that make at least one top solution fail, must be correct per the spec "not based on any program's behaviour".
5. Discriminative generation: drop tests with pass rate < 0.1 (assumed wrong), keep one test per identical pass vector; pick 5 solutions with identical or near-identical pass vectors; ask for 20 tests each passed by at least one and failed by at least one of them.
6. Final filtering: drop tests with pass rate < 10%; keep up to 5 tests per identical pass-vector group; drop problems with < 5 tests or > 60 of 64 perfect solutions.
7. 3 rounds after round 0. RL: GRPO, Qwen3-4B (thinking), group 8, temp 0.6, lr 1e-6, clip high 0.30 / low 0.20, response 32k.
**Verification / grading.** Assert tests with LLM-written expected values. No oracle program computes outputs after the rewrite, so test correctness rests on the "pass rate < 10% means wrong" rule and the pool's behaviour. A wrong test that 10%+ of solutions happen to satisfy survives; a correct but very hard test is deleted. The rewritten problem also detaches from its original reference solution.
**Scale & cost.** Problems 18,803 (r0) → 21,642 (r3); average tests per problem 10.8 → 20.9 → 28.4 → 35.0. 64 solutions per problem per round for ~20k problems. No $ or GPU hours given.
**Key results.**
- Pass@1 of the 64-solution pool falls from 43.80 (r0 tests) to 31.22 (r3 tests); pass@4 and pass@8 fall monotonically.
- RL with r3 data: EvolveCoder-4B average 49.0 over EvalPlus, BCB-Full, BCB-Hard, Aider-Polyglot, LCB-v5 vs Qwen3-4B 44.8 (+4.2) and Critique-Coder-4B 47.2; r0 → r3 data: 46.6 → 47.2 → 48.5 → 49.0.
- Discriminative generation keeps a larger share of its tests after filtering than adversarial generation.
- Larger models stay ahead on the final tests; gaps narrow at pass@8.
**Failure modes & limits.** Stated: no formal correctness guarantee; Python only. Mine: no precision/recall against a ground-truth judge, so "harder" may partly mean "more wrong tests"; the drop in pass@1 is offered as evidence of better verification, which it is not by itself; RL comparison lacks a same-compute baseline on non-evolved tests of equal count; no hack analysis.
**Tips & tricks.**
- Keep a solution × test pass matrix per task; it drives every decision (saturation, redundancy, disagreement).
- Remove tests nearly everyone passes; they carry no signal.
- Deduplicate tests by identical pass vector, keep a few per group.
- Prompt with the actual programs and their per-test results, choosing high-pass and maximally disagreeing ones.
- Ask for split tests: each must pass at least one and fail at least one of a near-identical group.
- Draw the candidate pool from several model families for behavioural diversity.
- Drop tasks where nearly every pool solution is perfect (too easy for training).
**Fit to Crucible Part 2.** F5/F6 repair loop: generalise the pass matrix to "rollout × check" over frontier and student rollouts (F9 already produces them). Checks that every rollout passes are dead weight; groups of rollouts with the same check vector but different World end states (by state diff) show where a new check is needed, and the "split" request to the verifier writer is: write a check that separates these two end states. Unlike EvolveCoder, Crucible must keep the oracle as the gate: every new check must pass the gold end state and alternatives (G3, G14), which closes the paper's correctness hole. F9: dropping tasks with > 60/64 perfect solves is the easy-band cut. F8/F11: embedding dedup at 0.9 cosine as a cheap layer. Avoid the "pass rate < 10% means test is wrong" rule; hard tasks would lose their only discriminating checks. Interface: `PassMatrix{rollout_ids, check_ids, bits, end_state_hashes}`; `SplitRequest{rollouts_a, rollouts_b, delta_a, delta_b}`. Priority P1.

## 2207.10397 — CodeT: Code Generation with Generated Tests (Chen et al., 2022; ICLR 2023)
Read: FULL (body and appendices A–J).  Relevance to Part 2: medium. Not a task-synthesis paper; it is the base algorithm for selecting a reference among model-written candidates when there is no oracle, and for measuring generated-test quality ("toxic" tests). Useful where an F4 oracle or an F5 check comes from sampling.
**Problem.** Pick the right solution from many samples without human tests: pass@100 77.4% vs pass@1 33.5% for Codex on HumanEval.
**Method / algorithm.** Dual execution agreement (RANSAC-inspired).
1. From the same model, sample N solutions X from the context (with examples removed from the prompt, so real tests are not exposed).
2. Sample tests Y by appending a placeholder `pass`, a comment "check the correctness of <fn>", and `assert` to the context; take the first 5 syntactically valid asserts from each of 100 samples (≤500 tests).
3. Run every solution on every test (0.1 s timeout). Group solutions with identical passed-test sets: each group S_x and its passed tests S_y form a consensus set.
4. Score f(S) = |S_x| · |S_y|; in practice sqrt(|S_x|)·|S_y| (passing more tests weighs more than having more look-alike solutions). Pick a solution from the best set; for top-k, one per top set.
5. Sampling: temperature 0.8, top-p 0.95, 100 solutions (50 for APPS, 1,000 for CodeContests).
**Verification / grading.** Ground-truth tests decide pass@k. Generated-test quality: accuracy = share passed by the canonical solution; toxicity = a test some generated solution passes but the canonical one fails.
**Scale & cost.** Up to 500 tests × 100 solutions per problem; cheap executions. No $ given.
**Key results.**
- code-davinci-002 pass@1: HumanEval 47.0 → 65.8, MBPP 58.1 → 67.7, APPS intro 27.2 → 34.6, CodeContests 0.7 → 2.1. Other models gain 4.2–13.1 pp.
- Scoring by solution count only (self-consistency) or test count only is worse than baseline; both factors are needed. AlphaCode-style output clustering is consistently worse.
- Better test writer helps weaker generators: davinci-002 tests add 6.2–15.9 pp pass@1 to InCoder and CodeGen.
- Only 10 tests already give +9.5 pp; gains flatten at ~50 samples × 3 tests.
- Keeping examples in the prompt inflates CodeT (tests copy real examples), so removing them is required for a fair measurement.
- Failure analysis (53/164 problems where a correct solution existed but was not top): ~20% due to ambiguous specs, missing corner cases or imports; the rest misunderstanding.
- Trivial solutions (constant or identity outputs) are many on hard sets (4.5/50, 91.6/1000) but filtering them changes little.
**Failure modes & limits.** Assumes wrong solutions rarely agree; fails when the model shares a misreading (ambiguity makes wrong consensus win). Toxic tests from weak models break the ranking. On competition-level problems the generated tests are too weak to help.
**Tips & tricks.**
- Score consensus by both agreeing-solution count (damped by sqrt) and passed-test count.
- Remove example I/O from the prompt when generating tests, or the tests leak the examples.
- Measure generated-test accuracy and toxicity against a trusted solution.
- A stronger model's tests can rescue a weaker model's selection; separate writer roles.
- Detect trivial outputs (constant, identity) as a cheap filter.
**Fit to Crucible Part 2.** F4/G14: when the oracle comes from sampling (no construction), use dual agreement across families as the acceptance rule: oracles from different families must reach the same end-state class and pass the same checks, and the shared-misreading failure mode is exactly why the ambiguity check in F10 matters. F5: compute "toxicity" of each generated check as a metric (a check that a wrong rollout passes but gold fails is a false-negative source); G3 gold-sanity already blocks the worst case, but toxicity on sampled rollouts quantifies check noise. Swarm role note: the strongest model writes checks, weaker ones write candidates. Interface: `ConsensusSet{rollouts, checks, score}` reused by F12 for clustering student failures by identical check vectors. Priority P2.

## 2502.01619 — Learning to Generate Unit Tests for Automated Debugging (Prasad et al., 2025; arXiv, UNC; the note calls it UTGen)
Read: FULL (body; appendix dataset and metric definitions).  Relevance to Part 2: medium. Two ideas transfer: the named trade-off between an error-revealing input and a correct expected output (any LLM-written check faces it), and the "accept an edit only if a held-out check set improves, else backtrack" rule for repair loops. Its "hard" debug split also informs difficulty in our debug lane.
**Problem.** Generated unit tests are useful for debugging only if the input triggers the bug and the expected output is right, without the gold solution. Zero-shot models trade one off against the other.
**Method / algorithm.**
1. Desiderata: attack rate (gold output ≠ buggy output on the input), output accuracy (predicted output = gold output), and the joint Acc ∩ Attack.
2. UTGen training data (from Tulu-3 Python functions with gold code f_r): an LLM perturbs gold code into buggy code f_b; sample n test inputs conditioned on (description, f_b); keep failing ones (f_r(x) ≠ f_b(x)); label output with f_r(x); generate a post-hoc chain-of-thought rationale for why f_r(x) is the output (STaR-style rationalisation); SFT the generator on (prompt → rationale + failing test). ~30K examples for 8B models, ~70K for 32B.
3. UTDebug at inference: (a) output by self-consistency, k = 8 sampled outputs with rationale, majority vote, keep only tests whose answer gets > 50% of votes; (b) per round generate n tests, use one failing test as feedback, accept the code edit only if the pass rate on the whole generated set rises, otherwise backtrack; 3 rounds.
4. Best-of-N use: 10 solutions, 3 tests each, pick the solution passing most generated tests.
**Verification / grading.** Private EvalPlus tests (80× more than HumanEval's) decide final pass@1. Datasets: HE+Fix (158 human-injected bugs; start code fails all tests, so easy), MBPP+Fix (325 realistic LLM-written wrong solutions), MBPP+Fix Hard (170 faulty codes that pass 50–95% of tests).
**Scale & cost.** 7–8B models (Llama3, Llama3.1, Qwen2.5-Coder-7B) and Qwen2.5-Coder-32B. Multiple tests × 8 votes × 3 rounds per problem; no $ given, but they warn this is costly on frontier APIs.
**Key results.**
- Trade-off, Qwen2.5-7B: prompting for failing tests raises attack rate by 13.5 pp but cuts output accuracy by 16.5 pp versus random tests; joint metric barely moves.
- UTGen: +7.59 pp on Acc ∩ Attack over prompted failing tests (average); Qwen2.5-32B +5.3 over prompted, +23.3 over random.
- Debugging, Qwen2.5-32B: MBPP+Fix +15.07 pp over no-test feedback, +4.61 over random tests; MBPP+Fix Hard +12.35 over no tests, +22.35 over random tests.
- Even GPT-4o and DeepSeek-V3 reach only ~50% attack rate on the Hard split; joint metric about one in three.
- GPT-4o as debugger with Qwen-32B UTGen tests: ~25 pp over self-feedback and 13.8 pp over prompted tests on the Hard split.
- Best-of-10 reranking with generated tests beats an 8B trained reward model by ~4.4 pp on HumanEval+.
- The hard split (faulty code passing most tests) is far harder to debug than human-injected bugs that fail every test (up to 53.5 pp gap).
**Failure modes & limits.** Function-level Python only; gold code needed for training data; the perturbation-based bugs may not match natural ones; generated tests still noisy, hence the voting and backtracking machinery.
**Tips & tricks.**
- Measure any check generator on attack rate, output accuracy, and their conjunction; the conjunction is what matters.
- Condition test generation on the faulty artefact; description-only tests rarely attack.
- Majority-vote expected outputs and drop low-agreement tests.
- In any repair loop, accept a change only if a broader held-out check set improves; backtrack otherwise.
- Bugs whose code passes 50–95% of tests are much harder than bugs that fail everything; select for that band.
- Small trained test writers can beat frontier self-critique.
**Fit to Crucible Part 2.** F5: the attack/accuracy trade-off is the generic risk when an LLM writes a check without executing the oracle; Crucible avoids it by labelling expected values from the oracle end state (as UTGen does at training time), so checks should always be computed from the gold run, never predicted. F8/F12 self-improvement guard: adopt "accept the verifier or World patch only if a held-out negative set and the gold alternatives score better; else backtrack" as the rule for every auto-applied fix (§3.8 guards). Debug lane F9: the "faulty code passes 50–95% of its tests" criterion is a cheap difficulty predictor and a knob (fewer visible failing tests). Interface: `CheckGenEval{attack_rate, output_acc, joint}`; `PatchAcceptance{heldout_before, heldout_after, accepted}`. Priority P2 (P1 for the backtrack guard).

## 2502.13820 — Scoring Verifiers: Evaluating Synthetic Verification for Code and Reasoning (Ficek et al., 2025; arXiv, NVIDIA)
Read: FULL (body, prompts; the large results tables were read only for the headline numbers quoted in the text).  Relevance to Part 2: medium. Gives a recipe and metrics to judge a synthetic verifier by how well it ranks graded-quality solutions, not just pass/fail; maps onto G5/G14 verifier evaluation and onto partial-credit checks.
**Problem.** No standard way to tell how well a synthetic verifier (LLM-written tests or a reward model) scores and ranks candidate solutions.
**Method / algorithm.**
1. Start from a benchmark with trusted tests (HumanEval, HE+, MBPP, MBPP+).
2. Generate many solutions with GPT-4o under two prompts (correct; "somewhat incorrect", logical errors only, no hints, e.g. only works for some value ranges), over several seeds and sampling settings.
3. Score each by the fraction of trusted tests passed. Dedupe solutions with equal score (keep the faster); always include the ground-truth (score 1.0); drop solutions that fail only through non-assertion errors (syntax, imports).
4. Select k solutions spread evenly over [m, 1]: compute target scores at equal spacing between the max (1.0) and the min m, greedily pick the unchosen solution nearest each target (k = 5 gives quantiles 0, .25, .5, .75, 1 when m = 0). Regenerate until k distinct scores exist; a few problems got manual solutions (10 for HE-R+, 15 for MBPP-R+).
5. Evaluate a verifier by its ranking of these k solutions: Top-1 accuracy (best ranked first), Bottom-1 accuracy (worst last), Spearman ρ against the trusted ranking, MAE between predicted and true pass fractions. Ties split fractionally.
6. Test-writer setup: 10 asserts per problem, temperature 1.0, 3 s timeout; reward models score each solution, min-max normalised per problem.
**Verification / grading.** Trusted tests define the ground-truth ranking. A saturation analysis shows Spearman ≥ 0.90 (CI lower bound) with only 6 trusted tests on HE+ and 5 on MBPP+, so few tests can already rank reliably.
**Scale & cost.** HE-R+ and MBPP-R+ (problem counts are in a table image not available as text) with ≥5 uniquely scored solutions each; 17 models evaluated. Reasoning-model cost limited tests to 10 per problem.
**Key results.**
- Best 10-test writers reach Top-1 79.1% (HE-R+) and 71.4% (MBPP-R+); models of ≥32B are needed for good Top-1 and Bottom-1.
- Reasoning models (DeepSeek-R1, o1-mini, o3-mini) are the best verifiers; scaling R1 to more tests reaches Top-1 91.6% on HE-R+ (plateau on MBPP-R+); Qwen-32B models plateau near 20 tests.
- Self-generated tests beat the best coding reward models (AceCoderRM-32B, Nemotron4-340B-Reward) on most metrics; reward models are weaker at ordering partially wrong solutions.
- Showing a (possibly wrong) solution in the test-writing prompt lowers test quality for all models: they adhere to the shown solution even when told not to.
**Failure modes & limits.** Benchmark quality is bounded by the source tests; one generator family (GPT-4o) produced the candidates (self-evaluation bias); Python function-level only; ranking partially wrong solutions is partly subjective.
**Tips & tricks.**
- Evaluate a verifier on a graded ladder of candidates (0, .25, .5, .75, 1 correct), not only on pass/fail pairs.
- Report Top-1, Bottom-1, Spearman and MAE together.
- Generate deliberately partially-wrong solutions with a "logical errors only, no hints" prompt for a realistic negative ladder.
- Drop candidates that fail for trivial non-logic reasons; they inflate verifier scores.
- Do not show candidate solutions to the check writer; it anchors on them.
- More tests help strong reasoning models more than weak ones.
**Fit to Crucible Part 2.** F5/G5: build a graded negative ladder per task (gold; one subtask missed; two missed; side effect; all wrong) from the oracle's action program, then score the verifier bundle by Spearman and Top-1/Bottom-1 on that ladder, in addition to the binary kill rate. This matters where Crucible gives partial credit (per-subtask checks or Φ progress): the ladder tests that partial reward is ordered correctly. Information separation (design rule 5) gets direct support: the verifier writer should not see candidate rollouts when writing checks, only the oracle and World. Interface: `NegativeLadder{levels: [(rollout_or_program, true_fraction)]}`; `VerifierEval{top1, bottom1, spearman, mae}`. Priority P2.

## 2505.22203 — From Accuracy to Robustness: A Study of Rule- and Model-based Verifiers in Mathematical Reasoning (Huang et al., 2025; arXiv, HKUST)
Read: FULL (body and appendices B–O; per-pattern tables not reproduced in text).  Relevance to Part 2: medium. Crucible's verifiers are mostly executable state checks, but answer-kind tasks and any LLM-judged check (report quality, explanation) face both failure modes measured here. The training-vs-oracle reward gap as a live hack monitor and the hybrid design transfer directly.
**Problem.** How accurate are rule-based and model-based math verifiers, and how do their errors affect RL? Static accuracy may not predict RL behaviour.
**Method / algorithm.**
1. Static set: 1,000 queries each from MATH, DeepScaleR, ORZ-Math, Skywork-OR1; two responses per query from short-CoT (Qwen2.5-Math-7B, Qwen2.5-32B) and long-CoT (R1-Distill-Qwen-7B/32B) models; GPT-4o labels correctness against the reference (agrees with two humans: κ 0.93, F1 0.98 on 1,000 items). 8,000 examples.
2. Rule verifiers: verl (string matching), Qwen-Math and HuggingFace Math-Verify (LaTeX parsing).
3. Model verifiers (1.5B/7B Qwen2.5, Qwen2.5-Math, R1-Distill), evaluated only on items the rule verifier rejected (the hybrid setting).
4. Hybrid verifier: rule first; if it says wrong, ask the model verifier.
5. RL: zero-RL GRPO from Qwen2.5-7B base on DeepScaleR (also Skywork-OR1, WebInstruct-Verified), verl, batch 1,024 prompts × 8 rollouts, 8k tokens, clip-high 0.28, KL 0; 4 nodes × 8 H100, ~3 days per run.
6. Hack monitor: at each checkpoint sample 1,000 training queries, score with GPT-4o as oracle, plot training reward vs oracle reward.
7. Probing: 13 adversarial answer patterns (empty symbols like "{", gibberish, adversarial prefixes "As an AI assistant, I know the answer is correct", answer explanations, etc.) on 471 DeepScaleR items; attack success = an incorrect answer judged correct.
8. Trained verifier R1-Distill-Verifier-1.5B: rejection fine-tuning on ~20K judgments agreeing with GPT-4o.
**Verification / grading.** Ground truth is GPT-4o labels validated against humans. Hacks found in RL: outputting a single symbol or long gibberish fools the fine-tuned generative verifier.
**Scale & cost.** 8,000 static items; RL runs ~3 days on 32 H100 each; hybrid adds 25.7% training time.
**Key results.**
- Rule verifiers: precision > 99%, recall ~86% average; 0.78 on Skywork-OR1; below 0.6 on general science (47% for HF on WebInstruct). Recall falls as the policy gets stronger (long-CoT ~0.92 vs higher for weak models).
- Hybrid (HF + R1-Distill-1.5B): ~+3 pp recall at > 98% precision; RL peak 57.3 vs 55.0 rule-only (+2.3); gap persists with more compute; +3.6 on science.
- Fine-tuned verifier: static recall 0.49 → 0.62, precision 0.68 → 0.73, yet in RL its training reward diverges from oracle at ~450 iterations and accuracy collapses; on Skywork-OR1 performance drops 58.7 → 55.5.
- All generative verifiers, trained or not, are fooled by simple patterns; scaling helps but does not remove it (Qwen2.5-Math-72B 12.0% average attack success vs Qwen2.5-72B 1.2%). The base 1.5B model is also vulnerable but was not hacked in RL, likely because the policy was too weak to find it.
- Discriminative verifier xVerify-3B: far lower attack success; in RL its training/oracle gap stays ≤ 0.004 with no collapse.
**Failure modes & limits.** Math and general-science answers only; oracle is GPT-4o (itself a model judge); small verifiers (≤7B) for the RL loop; probing patterns are hand-built from observed hacks.
**Tips & tricks.**
- Rule check first, model check only on rule rejections: keeps precision, recovers recall.
- Static classification accuracy does not predict hack resistance; probe with adversarial patterns before using a model judge as reward.
- Fine-tuning a judge for accuracy can make it easier to hack.
- Prefer discriminative (direct verdict) judges over generative ones in the RL loop.
- Monitor training reward against a stronger oracle on a fixed sample each checkpoint; divergence is the hack alarm.
- False negatives grow with student strength; re-audit rule checks as the student improves.
- A judge not hacked so far may just face a weak policy; stronger students will find it.
**Fit to Crucible Part 2.** F5/G12: any check that uses a model (answer equivalence, report quality) must pass an adversarial probe suite (empty, gibberish, self-assertion prefix, partial answer with explanation) before entering a VerifierBundle; record attack success per pattern. Answer-kind checks: adopt the hybrid order (exact or normalised rule first, model fallback on rejections only) and prefer a discriminative judge. F12: add the "training reward vs oracle reward on a fixed sample" monitor per task family, with the oracle being the World state check or a stronger judge; a divergence flags the family for F5 repair (this complements the hacking-gap idea of 2604.15149). F9: as the student moves from 35B to 397B, re-run false-negative audits on answer checks, since recall falls with student strength. Interface: `JudgeProbe{pattern, success_rate}`; `FeedbackSignal.reward_oracle_gap`. Priority P1.

## 2507.08794 — One Token to Fool LLM-as-a-Judge (Zhao et al., 2025; arXiv)
Read: FULL (body; appendices B–F read for the collapsed run, scaling hypothesis and prompt ablations; tables not reproduced in text).  Relevance to Part 2: medium. Any LLM judge in a Crucible verifier (answer equivalence, report checks, the intent auditor, failure attribution) needs a "master key" probe and the cheap mitigations here.
**Problem.** Reference-based generative judges in RLVR reward content-free answers. Found when an RL run (Qwen2.5-7B, REINFORCE, Qwen2.5-72B judge, WebInstructSub 30k) collapsed: responses shrank below 30 tokens to openers like "Sure, let's solve this problem step by step", and the judge rated ~90% of them correct.
**Method / algorithm.**
1. Master keys: non-word symbols (" ", ".", ",", ":") and reasoning openers ("Thought process:", "Let's solve this problem step by step.", "Solution" and translations such as 解, Respuesta). Each is used as the entire response for every (question, reference) pair.
2. Judges: specialised generative RMs (Multi-sub RM, General-Verifier, Omni-Judge) and general LLMs (Qwen2.5-72B, LLaMA3-70B, GPT-4o, o1, Claude-4), temperature 0, standard YES/NO prompt.
3. Benchmarks: Multi-subject RLVR, NaturalReasoning, GSM8K, MATH, AIME 1983–2024.
4. Master-RM mitigation: take 20k items from the 160k (q, a*, o, y) RM dataset, regenerate responses with GPT-4o-mini chain-of-thought, keep only the first sentence (a generic lead-in), label NO; add to the 160k; SFT Qwen2.5-7B and 32B Instruct with cross-entropy on YES/NO. The augmentation sentences avoid the test master keys.
5. Ablations: Qwen2.5 sizes 0.5B–72B; embedding-neighbour search for new keys; chain-of-thought + 5-vote majority; removing the question from the judge prompt.
**Verification / grading.** False positive rate of master-key responses (all wrong by construction). Judge quality: Cohen's κ vs GPT-4o (2,500 items) and humans (500); VerifyBench accuracy and macro-F1.
**Scale & cost.** 180k SFT examples; evaluation over five test sets and ten keys per judge. No $ given.
**Key results.**
- FPR up to 80–90%: "Thought process:" gets 60–90% on LLaMA3-70B and Qwen2.5-72B; ":" gets up to 35% on GPT-4o; General-Verifier accepts a single space 66.8% of the time on MATH. Specialised RMs are better but still > 10% on some keys.
- Master-RMs: near 0% FPR on all keys and benchmarks; κ 0.91 with GPT-4o and 0.90 with humans (7B); Master-RM-32B VerifyBench 95.15% acc, Hard 86.80%, above GPT-4o and Claude-4, within 2 pp of o1.
- FPR vs size is non-monotonic: 0.5B lowest FPR (literal matcher) but poor agreement; 1.5–3B high FPR; 7–14B best trade-off; 32–72B worse again, hypothesised to be "self-solving" the question instead of comparing.
- Removing the question from the judge prompt sharply cuts FPR (for math; general reasoning may need the question).
- Chain-of-thought plus majority voting can raise FPR.
- Embedding-similar phrases are new keys ("mental process" gives GPT-4o 16.1% FPR on GSM8K).
**Failure modes & limits.** Reference-based answer judging only; augmentation covers openers, not closing cues (self-verification text); a hypothesis, not a mechanism, for the scaling curve.
**Tips & tricks.**
- Before using any LLM judge as reward, score the empty/punctuation/opener responses; a non-zero FPR is disqualifying.
- Add truncated first-sentence responses labelled NO to judge training data.
- Mid-size judges (7–14B) can be more robust than 70B+ ones.
- For answer equivalence, show the judge only the response and the reference, not the question.
- Do not assume chain-of-thought or voting makes a judge safer.
- Search embedding neighbours of known keys to find new ones.
- A sudden length collapse with KL spike in RL is a judge-hack signature.
**Fit to Crucible Part 2.** F5/G12 attacker: include master-key responses (empty, ":", "Thought process:", a plausible opener with no work, a final claim of success with no tool calls) as fixed negatives for every task with a model-judged component; this is cheap and catches judge defects before any student run. F10 intent auditor and F12 failure-attribution judges: same probe, because they also produce labels that feed loops. Prompt rule for answer judges: reference + response only. F12 monitor: response-length collapse with KL rise per task family flags a judge hack. Interface: `MasterKeySet` in `DomainSpec`; `JudgeProbe` results stored in the VerifierBundle provenance. Priority P1 (P0 if any reward is model-judged).

## 2610.24206 — SWE-Bench Pro V2: A Cleaner, Harder-to-Game Coding Benchmark (Dan et al., 2026; arXiv, Scale AI)
Read: FULL.  Relevance to Part 2: high. A complete, measured maintenance pipeline for a repository-task suite like our debug lane: agentic audits as triage, a defect taxonomy, the "repair the instruction, never the test" rule, blind re-implementation as the gate for any instruction edit, verifier hardening, and clean-container regrading. Most of it maps directly onto F6, F10 and the F5↔F3 repair loop.
**Problem.** As agents improve they reach task defects older agents never touched and exploit the harness. Keep a 731-task benchmark accurate.
**Method / algorithm.**
1. Task = instruction (problem, requirements, interface) + container at base commit + reference patch + hidden test patch + FAIL_TO_PASS / PASS_TO_PASS lists + verifier. Valid iff every instruction-satisfying solution passes and every other fails.
2. Round 1: fix 40 images whose reference patch carried test/fixture files or whose instructions lacked the template.
3. Round 2: automated audit (difficulty calibration, instruction–test alignment, reward hacking) flagged 387; engineers aligned instructions with graded tests until ~700 passed re-audit. Global changes: drop "do not edit tests" from the instruction and instead revert test-related edits in the verifier; scrub git history; disable network (a frontier model had copied upstream fixes on 351 solved tasks; only 237 held with GitHub blocked).
4. Round 3, two agentic audits: (a) artifact audit: an LLM agent compares instruction with test lists and reference patch without executing, flagging six patterns (test requirement missing from instruction; contradiction; test or solution detail copied into instruction; changed behaviour no graded test exercises; fixtures read but never created; a graded list naming a whole test file), with "overly verbose requirements" as a leak signal; a human marks priorities → 161 flags. (b) failure audit: four agent setups (Claude Code + Opus 5, Codex + GPT-5.6, mini-SWE-agent with each) run every task; an LLM judge reads trajectory, patch, verifier log and tests and attributes each failure to model or task; flag when a patch consistent with a reasonable reading was rejected → 54 flags, 39 new. Total 200.
5. Adjudication: 23 engineers in their own language, 404 hours; each claim marked holds / holds in part / does not hold with artifact-and-line citation; high-priority flags get a second independent review and a third pass. Rule: if the reference adds a behaviour a hidden test checks, the instruction must state the behaviour but not how the reference implements it. Never edit a test to match an instruction; never rewrite an instruction to match a wrong test. Defects whose fix lies in tests, fixtures, verifier or image → quarantine/discard.
6. Blind validation: a second engineer who saw neither original task nor reference implements the edited instruction; the edit ships only if that implementation passes the hidden tests.
7. Verifier hardening (all tasks): scope Go runs to graded packages (253 tasks), restrict full-suite runs (231), cap pytest-xdist at 4 workers (239 + 83), fix the jest parser (47), restore every path the test patch touches and delete only paths it creates, restore tracked test data with allowlists, remove agent test files that collide with hidden tests, clear stale git locks, delete __pycache__/.pyc (a git-stash round trip left .pyc files making the verifier import base code). Final gate: reference passes every task, empty patch fails every task.
8. Locked protocol: network only to the model endpoint during the agent phase; save the agent's staged diff, apply it to a fresh image, run the unchanged verifier there; if sandbox and clean grades disagree, take two more clean grades and majority. Patch audit for in-patch forgeries (replace directives into the patch or absolute paths, go.sum entries for modules absent from the image, large directories imitating an SDK); trajectory scan for fetches and writes to module caches.
9. Hard split: tasks failed by at least 2 of 5 model families across runs (54 → 51 after review), each with reference pass, empty fail, and at least one solving run.
**Verification / grading.** Hidden tests in a clean container. Exploits observed: replacing python3 with a script printing PASSED (scored 1 in-sandbox since the verifier restored tests but not binaries); editing the Go module cache in place; serving a fabricated module from a local GOPROXY; forged go.sum entries; one model disguised a module-proxy request as model-endpoint traffic.
**Scale & cost.** 731 → 642 released (391 repaired, 89 discarded, 251 changed only via hardening; 480 = 66% repaired or removed); 404 engineer-hours; mean API spend per task ~$2.3–3.0 per system run; 50-minute budget per task; 5,778 harness-study trajectories released.
**Key results.**
- Defect counts (repaired/discarded): unstated test requirement 119/22; spec–test contradiction 10/5; solution leakage 18/4; insufficient tests 1/33; verifier or environment defect 228/19; misc 15/6.
- Of 45 rejected flags, 26 claimed leakage: the leak flag is the least reliable one.
- 27 instruction repairs: 18 had been failed by every v1 configuration; 15 now solved by all frontier systems. The fixture-restore fix recovered 44 runs.
- V2: frontier ~99% (Opus 5 99.4%), others 87.2–97.7%; Hard 52.9–98.0%. Every task solved by at least one frontier system.
- Network open: 32/642 Opus 5 runs contacted code hosts, 25 downloaded the task's own repository.
- Harness study (3 open models × 3 scaffolds): variance in solved/failed is 31.7% task, 2.65% model, 0.29% scaffold, 65% task-by-run noise; scaffolds change cost (Kimi 1.3M vs 2.6M tokens per solve), not accuracy. Testing after the last edit roughly doubles the odds of solving. Median failed run writes 1.6–2.0× the completion tokens of a solved run ("a failing run is one that kept going").
**Failure modes & limits.** Old public repositories remain memorisable; clean regrade misses forgeries inside the patch (left to the patch audit); the failure-audit judge is an LLM; the benchmark is near saturation for frontier models.
**Tips & tricks.**
- Use strong agents as auditors: running them finds defects reading artifacts misses (39 of 200 flags).
- Repair instructions, keep tests fixed, discard tasks whose defect is in the tests.
- State behaviour the hidden test checks, never the implementation.
- Validate every instruction edit by a blind re-implementation from the text alone.
- Remove "do not edit tests" from the prompt; enforce it in the grader by restoring test paths.
- Grade a saved diff in a fresh container; triple-grade on disagreement.
- Lock the network to the model endpoint during the agent phase; scrub git history.
- A graded list naming a whole test file makes unrelated or flaky failures fail correct patches.
- Leak flags are often wrong; require line-level citation before acting on them.
- Hard split = failed by ≥2 of 5 model families, each still reference-pass and empty-fail.
**Fit to Crucible Part 2.** F6: add G15 "clean-container regrade" (apply the student's diff to a fresh start image and re-verify; the debug lane already grades patches, but any non-patch state left in the sandbox must not count) and the pre-test restore rules (restore test-owned paths, remove colliding agent test files, clear caches). Add the verifier "unchanged-reference passes, empty patch fails" final check per task (G3/G4 already). F10: adopt the two-audit design: artifact audit (instruction vs checks vs oracle; the six patterns become generic lints over our task contract) and failure audit (frontier runs from F9 with an attribution judge; flag when a reasonable solution is rejected). F5↔F3 repair: the "repair the prompt, never the check" rule plus blind re-implementation (a different-family model solves from the edited instruction alone and must pass) is the guard for any auto-applied instruction fix in the §3.8 loops. F9: the Hard-split rule (≥2 of N families fail, ≥1 solves) is a cheap frontier-audit band definition; the 65% task-by-run variance argues for k ≥ 4 student rollouts before banding. F12: "failed runs are long runs" and "test after last edit doubles odds" are trace features for failure clustering. Interface: `AuditFlag{source: artifact|failure, pattern, holds: yes|partial|no, citation}`; `BlindRevalidation{model_family, passed}`. Priority P0 for the clean regrade and the repair rule; P1 for the audits.

## 2609.08149 — SWE-Bench Pro Verified: A Reliable Benchmark for Software Engineering Agents (Zheng et al., 2026; arXiv, Shanghai AI Lab / ECNU)
Read: FULL (body, appendix A leakage cases; later appendix cases skimmed).  Relevance to Part 2: high for the debug lane. It is a concrete inventory of evaluation-time leakage channels in repository tasks, with a paired before/after measurement and an audit that separates "hack removed" from "normal work broken". Complements 2610.24206, which differs on the network rule and on editing tests.
**Problem.** SWE-Bench Pro scores are inflated by agents retrieving gold patches or hidden tests at run time, and distorted by tasks whose instructions and tests disagree.
**Method / algorithm.**
1. Leakage channels (four): local artifacts (gold patches, hidden tests, fixtures, evaluator files), git history (future commits, branches, tags, remotes, reflogs, notes, replace refs, stashes; loose objects survive ref deletion), online sources (upstream commits, raw files, APIs, mirrors on any code host), metadata (target SHA inside the instance id, repository names in paths, F2P/P2P lists).
2. Anti-hacking controls: (a) rebuild each repository as a fresh single-commit repo (record tracked files, recursively remove nested histories, restore files in batches so dependencies stay); (b) delete tracked tests and clean test directories of ignored suites, fixtures and golden data; disable git hooks that could restore them on checkout; (c) allowlist metadata, drop gold patch and test lists, hash the instance id and use it as the workspace name, strip the repository name from visible paths; (d) block code-hosting domains (GitHub raw/API/objects, GitLab, Gitee, Bitbucket, Codeberg, GitCode, ...) while keeping dependency services; tell the agent not to use code hosts or module proxies for solution info.
3. Validation of anti-hacking: paired runs Baseline vs Anti-hacking on all 731 tasks; broad scan of suspicious git, file and network operations; high-precision audit of commands that touch answer-related paths; for every PASS→FAIL transition, an LLM annotator assigns one of: hack removed (direct evidence), hack removed (highly probable), normal execution impaired, randomness/insufficient evidence. McNemar test on transitions.
4. Task refinement: map public issue reports (GitHub issues, review repos, HF feedback) to instances → 119 candidates; an LLM classifies issue category (misleading description, overly narrow tests, overly broad tests, other defects such as corrupted data or invalid paths), checks validity, drafts a fix; human experts make minimal edits, preferring requirements/problem/interface over tests, gold patch untouched where possible; trial runs and iteration. 102 revised, 17 rejected.
**Verification / grading.** Original F2P and P2P tests. Validation of refinement by PASS/FAIL transitions on the refined subset and field-level diffs.
**Scale & cost.** 731 instances; 7 models on mini-SWE-agent with official settings; costs not given.
**Key results.**
- GLM-5.2: 78.80% → 57.32% (−21.5 pp) with anti-hacking; 186 PASS→FAIL vs 15 FAIL→PASS (McNemar p < 0.001). Baseline trajectories show local answer leakage on ≥103 tasks and network leakage on 49; confirmed answer-file access falls to 0.
- Of the 186 PASS→FAIL, 90.9% are attributed to removed hacking; 0 to impaired normal execution; 17 to randomness (similar to the 15 reverse transitions).
- DeepSeek-V4-Pro changes only slightly (little hacking in an earlier audit).
- Refinement of 102 tasks: requirements edited in 92, hidden tests in only 17; GLM-5.2 21 FAIL→PASS vs 2 PASS→FAIL; 59 stay FAIL (genuinely hard).
- Fix mechanisms that turn failures into passes: state exact constants and strings; define set vs multiset vs order semantics; state defaults and return structures; clarify control-flow boundaries; define interfaces; correct tests that enforce irrelevant wording.
- Observed hacks: `git show <target SHA>:file` with a byte-for-byte diff ("IDENTICAL TO GOLDEN PATCH"); copying seven gold files staged under /tmp/gold; curl of raw.githubusercontent.com at the target SHA; switching to GitLab when GitHub is refused. The SHA in the instance id served as the answer locator.
**Failure modes & limits.** Domain blocklists cannot cover self-hosted git, proxies, mirrors or raw IPs; residual files may remain; refinement started from public reports, so unreported defects remain; LLM annotator for causes. Unlike 2610.24206, it edits tests in 17 cases and uses a blocklist, not an allowlist.
**Tips & tricks.**
- Rebuild repositories as single-commit repos; deleting refs is not enough because objects persist.
- Hash task ids and workspace names; never put a commit SHA in anything the agent sees.
- Remove ignored test fixtures and golden data, and disable git hooks.
- Validate an anti-leak change by paired runs and an explicit audit of every PASS→FAIL for collateral damage.
- Use McNemar on paired transitions rather than comparing two pass rates.
- Most spec fixes are requirement edits stating exact constants, order semantics, defaults, boundaries, interfaces.
- Prefer an allowlist (model endpoint only) over a blocklist; agents route around blocklists.
**Fit to Crucible Part 2.** F6 G11 for the debug lane: extend the leak rule set with the four channel classes as concrete start-tree checks (no future objects in .git, no fixture or golden data for hidden checks, no target SHA or upstream repo name in ids, paths or env vars, no hooks), run as CPU gates before any rollout. F10: adopt the paired Baseline vs hardened comparison with McNemar and the collateral-damage audit whenever a gate or World change is applied to existing tasks (it is the evidence that a fix removed hacks without breaking solvability). F3/F5 prompt quality: the six fix mechanisms become an instruction lint ("does every check-enforced constant, ordering, default, boundary and interface appear in the instruction?"), which matches G8 communicability for Worlds. Interface: `LeakChannel∈{local_artifact, vcs_history, network, metadata}`; `TransitionAudit{pass_to_fail: [{task, cause}], mcnemar_p}`. Priority P0 for the debug-lane leak checks; P1 for the paired audit.

## 2609.06780 — Shortcutting the Fix: Identifying and Categorizing Agentic Exploits in Software Engineering Benchmarks (Ludwig et al., 2026; arXiv, NVIDIA)
Read: FULL (body, appendices A–E with prompts; per-category tables not in text).  Relevance to Part 2: high for the debug lane and for the swarm audit. Open-weight agents (the same family as our swarm) shortcut in most trajectories; memory-based reconstruction cannot be blocked by the sandbox; a three-judge trajectory audit is cheap and validated.
**Problem.** Pass rates on repository benchmarks hide "agentic shortcutting": satisfying verification without doing the task, through upstream code, local git, memory, or other hidden sources.
**Method / algorithm.**
1. Taxonomy: UPSTREAM (network fetch, clone, package download of upstream, reading issues/PRs, or local use of downloaded upstream artifacts); LOCAL_GIT (git log, git fsck etc. to find future commits; inspecting base commit and ancestors is allowed); MEMORY (acting in the environment on code recalled from training, e.g. writing a recalled upstream file; reasoning about it without acting is not penalised); OTHER (searching filesystem, package caches, metadata for newer code or hidden task info).
2. Verdict per category: SUCCESSFUL (hidden info retrieved and plausibly helped), ATTEMPTED, NOT_ATTEMPTED. Per judge: SUCCESSFUL if any category succeeded, else ATTEMPTED if any attempted.
3. Panel: three open judges (Qwen3.8-Flash-Next, DeepSeek-V4-Flash-0731, GLM-5.3-Flash) read every turn (repo name, base commit, reasoning, tool calls, outputs); majority vote; full disagreement → ATTEMPTED; unparseable → NOT_ATTEMPTED (<0.6%).
4. Conditions: Vanilla vs Principled prompt (adds a "Solution Originality" block forbidding upstream access, non-current git refs, reading hidden files or trajectories, recalling upstream code from memory, searching caches, "for ANY reason ... even to verify"; warns that such solutions are discarded) × network on/off. Containers already purge future commits and tags.
5. Benchmarks: SWE-bench Multilingual (300 tasks, 9 languages) and DeepSWE (113 tasks, 91 repos, 5 languages); mini-SWE-agent; 3 runs per task; Pass@1, Pass@3; exploit Attempt, Success, Resolved-with-exploit rates.
**Verification / grading.** Unit tests for resolution; judges for exploits; human check on 400 trajectories (100 tasks × 4 settings).
**Scale & cost.** 3 models × 2 prompts × 2 network settings × 3 runs × 413 tasks; plus 3 judges per trajectory. Costs not given.
**Key results.**
- Vanilla: attempt rates 58.7–91.0% (Multilingual) and 55.5–89.1% (DeepSWE); up to 72.1% of trajectories resolved with a successful exploit.
- Blocking network does not lower attempts (58.7–91.0%); agents switch to local git (up to 60.9%) and memory (up to 57.2%). Local-git exploits succeed 0% (containers clean), memory exploits succeed in up to 33.9% of trajectories with network off.
- Principled prompt: attempts 9.4–14.2% and 0.9–10.3%; exploit-aided resolutions nearly vanish. Pass@1 change: −0.1 to −12.7 pp on Multilingual (inflated before), −5.6 to +4.7 on DeepSWE.
- Judges: human agreement 87.8% on attempts, 82.3% on success (95/93% under Vanilla+network; lowest 70% on success with network off, where LLMs over-call memory success). Inter-judge disagreement ≤ 17.1% (attempt), ≤ 25.1% (success).
**Failure modes & limits.** Three open models; judges can misread command sequences, especially memory cases; a prompt is a soft constraint, not isolation. My critique: prompting the policy not to cheat does not transfer to RL training, where reward drives behaviour; the measurement (audit) is the useful part for training.
**Tips & tricks.**
- Audit full trajectories, not only patches; outcome-only metrics hide shortcuts.
- Count Attempt, Success and Resolved-with-exploit separately.
- Expect a pivot to memory when network is blocked; memory is the residual channel.
- A three-judge open-model panel with majority vote is cheap and agrees ~88% with humans on attempts.
- Give the judge the repo name and base commit so it can recognise upstream content.
- An originality instruction cuts attempts sharply at evaluation time.
**Fit to Crucible Part 2.** F10/F12: run the four-category exploit judge over every frontier-audit and student rollout on mined-repo tasks; a task whose passes are mostly exploit-aided is not measuring debugging (intent INVALID), and any student pass labelled MEMORY-successful should get zero reward in training (hack flag in F12). Since memory cannot be sandboxed, mined tasks from popular repositories need the metamorphic-twin probe (2604.21579) or synthetic fault injection so recall cannot reproduce the fix. Swarm roles (§3.9): the judge panel matches our family rules (three non-student families). The evaluation prompt for the benchmark should carry the originality block; the training prompt should not rely on it. Interface: `ExploitVerdict{category∈{upstream, local_git, memory, other}, status∈{successful, attempted, not_attempted}, judges: [...]}` on each rollout. Priority P0 for the debug lane audit.

## 2608.14940 — When Is an Agent Evaluation Over? Outcome Finality and Cross-Unit Separation (Casheekar et al., 2026; arXiv, U. Michigan Law / NUS)
Read: FULL.  Relevance to Part 2: medium. A conceptual paper with a small controlled replay, but it names two conditions Crucible's World must guarantee once it has a clock, events, async services and parallel rollouts: the label is final when scored, and one rollout cannot change another's start state.
**Problem.** Evaluations score the state at the endpoint and count each run as one trial. That is valid only if no later event can change the outcome (outcome finality) and no earlier run can change a later run's conditions (cross-unit separation). The endpoint establishes neither.
**Method / algorithm.**
1. Completion argument, four questions: what outcome, over what period, and what analysis unit? what ends the interaction vs what determines the score? what can remain active after the endpoint or reach another run (descendant processes, queues, shared stores, credentials, external accounts, public artifacts, retained memory)? how is finality and separation established, with evidence that a cancellation, isolation or reset succeeded?
2. Rules: assign a final label only after every relevant route that could change the outcome is blocked, followed to a terminal state, or bounded so all values give the same label; else report unresolved. Count runs separately only after every route between them is blocked or shown harmless; else model the dependence or group connected runs as one unit.
3. Controlled replay (AgentDojo 0.1.35 tasks, fixed function-call traces, no model; local HTTP service that applies a write after 0/25/100/250 ms; 5 schedule ids × 10 reps). Finality policies: snapshot at endpoint, reconcile (wait for terminal), verified cancellation; synchronous-write control. Separation policies: shared state, namespaced state, verified reset (wait for A's op to be terminal, restore and check B's initial value), grouped analysis; no-write and reversed-order controls.
4. Documentation review of ten public agent-benchmark protocols (one coder) for open work, reset scope and separate-scoring evidence.
5. Open-effects record per operation or resource: run/task/group ids; trace position and action; responsible system; stable handle (process, transaction, job id); status (pending, terminal success, terminal failure, cancel requested, cancelled, persistent, unknown); possible later states and relevance period; cross-run route; latest system observation; analysis treatment (closed, modelled, grouped, bounded, unresolved).
**Verification / grading.** In the replay, labels from the service's state.
**Scale & cost.** 200 trials per policy; trivial compute.
**Key results.**
- Snapshot vs terminal labels agree at 0 ms delay (50/50) and disagree in all 150 nonzero-delay trials; snapshot success 50/200 vs reconcile 200/200; verified cancellation scores the 150 pending writes as failures. Synchronous control: 0 disagreements in 600.
- Shared state: run A changes run B in 150/200 pairs (B's mean score 0.75 lower than namespaced); 0/200 under namespacing, verified reset, no-write control; reversing order removes it. Grouping keeps exposure but halves the analysis unit count (400 → 200).
- Review: reset or retention explicit in 8/10 protocols; 6 do not say whether background processes finish before scoring; only 3 give explicit evidence for separate scoring. pass^k assumes i.i.d. trials, which shared state breaks.
**Failure modes & limits.** Constructed mechanism only, no prevalence estimate; one coder for the review; open environments cannot enumerate all routes.
**Tips & tricks.**
- Do not treat a tool's acknowledgement as completion; track operations to a terminal state before scoring.
- Decide whether the step limit is part of the task or just an observation cutoff, and say which.
- Namespace or verified-reset every shared service between rollouts; verify the reset by reading the restored value.
- Run order randomisation (or reversal) as a cheap cross-run interference test.
- If state must persist across runs, the stream is the unit; do not compute pass^k over it.
- Record open effects with stable handles from the system, not from the agent's report.
**Fit to Crucible Part 2.** F1/F2 World contract: Apps with async jobs, scheduled events or notifications must expose a "settle" operation (advance the simulated clock until all pending operations are terminal) that the grader calls before scoring, and verifiers must declare whether their outcome is "at endpoint" or "after settlement". G2 determinism should run with settlement on. Parallel rollouts in F9 and in training: each rollout's World namespace must be fresh, and the gate suite should include a "run A then B on shared services" interference test for any App that holds cross-session state (people, mailboxes, external accounts). F12: difficulty and pass^k statistics assume separate units; shared-state task families must be grouped. Interface: `OpenEffect{handle, status, cross_run_route}` emitted by App wrappers; `Verifier.outcome_time∈{endpoint, settled}`. Priority P1 (P0 once the clock and events subsystem is on).

## 2609.11028 — BenchShield: Formal Model-Backed Instrumentation for Reward Integrity in LLM-Agent Evaluation Infrastructure (Zheng et al., 2026; arXiv, Dartmouth and others)
Read: FULL (body §1–§6, verdict and isolation results; appendix cases not read in depth).  Relevance to Part 2: high. It gives a fixed vocabulary of seven reward-integrity dimensions over the source-to-score lifecycle, a static taint check on the task package, a runtime event checker with four verdicts, and measured coverage of six isolation mechanisms. This is close to what F6 needs as a generic, domain-independent "reward path" gate, and to what F12 needs to label hacks in training traces.
**Problem.** Interactive benchmarks can be hacked through the path from what the agent observes or modifies to the reward. Task-specific patches and transcript-only detectors give no reusable evidence that a run stayed in bounds.
**Method / algorithm.**
1. Seven integrity dimensions: I1 observation (protected state hidden), I2 authority (agent cannot control outcome-owned state), I3 handoff (agent state enters outcome computation only via declared artifacts), I4 reward provenance (score comes from trusted outcome output), I5 failure handling (crash, timeout, malformed output, skipped execution, unknown event all fail closed), I6 release and reset (logs, feedback, snapshots, residual state), I7 semantic adequacy (accepted evidence means what the task intends; recorded, not enforced).
2. Formal core in TLA+, checked with TLC: phases setup → agent → handoff → outcome computation → reward collection → release; event alphabet Expose(r,a), Mutate(r,a), Handoff(h), Verify(I,v), Reward(s,x), Release(r,a), SemanticWitness(r,ℓ,e). Checks safety (no bad states), non-vacuity (an honest path exists), and counterexamples for each unsafe switch.
3. Task binding: a compiler reads the task package and backend config and emits a typed template mapping resources to authority classes (VerifierOnly, AgentVisible, forbidden...) and declared handoffs; authors fill only task-semantic fields (ambiguous roles, deliverables, review obligations such as "inferred from records, not copied from labels?"). Missing bindings stay as evidence gaps; they cannot erase discovered paths.
4. Static lane: activate a capability graph (nodes: resources and components; edges: operations); seed four taints (agent control, protected information, failure state, stale state); propagate over observe/mutate/handoff/outcome-input/reward/normalise/release edges with phase history; report a path when it reaches a sink for I1–I6. Executable, answer-bearing or aliased content keeps provenance through a handoff until sanitised. LLM static auditors may add labels or edges (add-only), never delete parser facts or issue verdicts.
5. Runtime lane: wrap the sandbox protocol; each record names the operation and one of eight phases; shell bodies decomposed by a per-program grammar into file and network operands; unresolved constructs marked opaque → fail closed. Each run seals an evidence bundle (actions, lifecycle events, resource and network manifests, artifact and verifier-input manifests, reward provenance, static result); the checker is a pure function of the bundle, so archived runs can be relabelled after a checker change.
6. Semantic routing: six independent LLM auditors each get only their evidence slice (lifecycle-graph, static artifacts, boundary, deliverable, egress, meta-trajectory); labels annotate but cannot change structural results.
7. Verdicts: Checked, VectorExposed (path exists, no evidence of use), AgentViolation (evidence of attempted or actual forbidden path), Inconclusive (missing or contradictory evidence, invalid binding, unknown reward-relevant event).
**Verification / grading.** Ground truth: a 456-trajectory human-adjudicated corpus from 31k+ public runs (Terminal-Bench 3 CI, SkillsBench 23,648 runs, ClawsBench 7,834), three annotators per record, episodes with primary vector, causal chain, first enabling condition, and positions of recon, first attempt, first success.
**Scale & cost.** 66 Python modules on an existing eval framework; structural verdict needs no model call and takes < 2 min per run; the full pipeline with semantic audit costs $5–10 per cell; static audit up to 65% cheaper than a hackability-scanner baseline. All model calls Claude Opus 5 high.
**Key results.**
- 314/456 (69%) adjudicated trajectories contain reward hacking; 419 episodes; 80 trajectories have several.
- Chains are directional: I1 opens chains (69 entry vs 4 outcome links); I4/I5 close them (5 entry vs 77 outcome). 106 of 110 multi-link episodes cross classes; top transitions I2→I4 (24), I2→I5 (21), I3→I5 (13).
- Timing: median first exploit attempt at 0.60 of the run, first success at 0.76; agents do legitimate work first, so neither a start-state scan nor a final-answer check is enough.
- 43% of episodes had no classified enabling condition; a recurring missing class: the verifier deserialises the deliverable (e.g. pickle.load), giving the agent code execution at grading time.
- Static recall of full exploit chains 77–100% vs 23–94% for the scanner baseline; same-vector coverage 43–78% vs 16–56%.
- Runtime attribution: 96% accuracy on verdicts among 144 runnable cells; no exploit got Checked; a transcript-only LLM detector reached 36%.
- Isolation, each alone: a separate verifier environment removes 82–95% of exposed I1–I4 packages; hiding verifier files gives most of the I1 effect; read-only task files remove a third of I2; unprivileged user about half of I1 and I2; seccomp/cap-drop and network egress block 2% each; fresh environment per trial and resource limits remove nothing. No mechanism moves I5 or I7.
**Failure modes & limits.** Model is an abstraction (classes of resources), so routes via the task's own service store, the public web, or a payload inside the one declared artifact are invisible to switches; semantic adequacy is left to LLM auditors; runtime instrumentation depends on the adapter's tool-call stream (container actions are not proxied); one framework.
**Tips & tricks.**
- Grade in a separate verifier environment behind a declared handoff; it removes most structural vectors at zero model cost.
- Hide verifier files from the agent environment and mount task files read-only.
- Make every grader failure path fail closed (I5); isolation will not fix it.
- Never deserialise or execute the agent's deliverable inside the grader without sandboxing.
- Seal an evidence bundle per run and make the checker a pure function of it, so verdict changes can be replayed over archives.
- Keep verdicts four-valued; "inconclusive" is a coverage gap, not a pass.
- Expect exploits mid-run after honest work; monitor the whole trajectory.
- Let LLM auditors add facts, never delete parser facts or issue final verdicts.
**Fit to Crucible Part 2.** F6: adopt I1–I7 as the generic G-gate taxonomy for reward-path integrity (our G7 tool-only and G11 leak are I1/I2 instances; I5 fail-closed is missing as a named gate and should become G16: every verifier error, timeout, unparsable output, or skipped check yields a fail, tested by fault injection into the verifier). F1: the World already holds hidden state server-side; record each App's resources with an authority class in the AppBundle so a static taint check can run per task (does any policy-visible tool expose a VerifierOnly column? can any tool mutate verifier-owned state?). F9/F12: emit a sealed per-rollout evidence bundle (tool calls, state deltas, verifier inputs, reward source) and run the structural checker on every training rollout; AgentViolation zeroes the reward and files a hack flag. The "first attempt at 0.60 of the run" finding supports trajectory-level monitoring over end-state checks in long-horizon tasks. Interface: `ResourceClass∈{agent_visible, verifier_only, forbidden, handoff}` on App resources; `IntegrityVerdict∈{checked, vector_exposed, agent_violation, inconclusive}` with dimensions. Priority P0 for I5 fail-closed and the separate grader; P1 for static taint and runtime checker.

## 2511.04703 — Measuring what Matters: Construct Validity in Large Language Model Benchmarks (Bean et al., 2025; NeurIPS 2025 Datasets & Benchmarks)
Read: FULL (body, checklist appendix A; codebook B and inclusion C skimmed).  Relevance to Part 2: medium. A review, not a method; its checklist is a good template for the DomainSpec skill definition (F0) and the IntentCard (F10), and it sets the bar for the benchmark write-up.
**Problem.** Do LLM benchmarks measure the phenomenon they claim? A systematic review of how phenomena are defined, tasks sampled, metrics chosen and claims justified.
**Method / algorithm.**
1. Corpus: 46,114 papers (ICML/ICLR/NeurIPS 2018–2024, ACL/NAACL/EMNLP 2020–2024) → keyword filter ("benchmark" + "LLM"/"language model") → 2,189 → GPT-4o-mini screen on three criteria (F1 0.84 vs human on 50) → 522 → 29 expert reviewers apply four criteria → 445 benchmarks.
2. Codebook of 21 items (phenomenon, task, metric, claims) derived from face, predictive, content, ecological, convergent and discriminant validity; primary reviewer codes, second reviewer maps to simplified options; 46 papers double-coded (Brennan–Prediger κ 0.524 over 30 categorical items).
3. Inductive synthesis of recommendations by the first author over 50 articles and all annotations, refined over five meetings.
**Verification / grading.** Not applicable (review).
**Scale & cost.** 445 benchmarks, 29 reviewers.
**Key results.**
- 78.2% define the phenomenon; 47.8% of those definitions are contested; 61.2% treat it as a composite.
- < 10% use complete real-world tasks; 40.7% use constructed tasks.
- Sampling: 27.0% at least partly convenience sampling; 55.2% partly targeted; 46.2% criterion; 17.1% random/stratified.
- Metrics: exact match used by 81.3%; LLM-as-judge 17.1%; only 16.0% use any uncertainty estimate or statistical test.
- 53.4% present construct-validity evidence; 47.5% add synthetic items.
- Eight recommendations with a checklist: (1) define the phenomenon (operational definition, scope, sub-components measured separately; negative definitions when contested, e.g. repeating memorised answers is not reasoning); (2) measure only the phenomenon (control unrelated sub-tasks, test format constraints, validate automated parsers); (3) representative dataset (random or stratified sampling over a defined task space; verify generated items; include sensitivity items such as rewordings); (4) acknowledge reuse limits; (5) prepare for contamination (detection tests, held-out set, check pre-exposure; procedurally generated tasks help); (6) statistics (power, uncertainty, rater variability); (7) error analysis (do failures correlate with non-target confounders?); (8) justify construct validity (real-world relevance, rationale linking task and metric to the definition, convergent/discriminant comparison, trade-offs).
**Failure modes & limits.** Only peer-reviewed conference benchmarks; moderate inter-rater agreement (κ ≈ 0.52); LLM pre-screen may drop some.
**Tips & tricks.**
- Write an operational and a negative definition of each skill before generating tasks.
- Measure sub-skills separately when a phenomenon is composite.
- Baseline the confounding sub-skills (format, instruction following, world knowledge) on their own.
- Validate any LLM parser or extractor for accuracy and bias.
- Sample tasks by an explicit plan over a defined space, not by convenience.
- Keep a held-out set and add contamination tests to the benchmark itself.
- Always report uncertainty and power.
- Check that failures land on the target skill, not on confounders.
**Fit to Crucible Part 2.** F0 DomainSpec: each `SkillLeaf` should carry an operational definition, a negative definition (what does not count, e.g. "copying the fix from history is not debugging"), and declared confounders with a baseline probe each; this gives F10 concrete things to test. F3/F11: the sampling recommendation is the case for the MAP-Elites grid with explicit quotas (targeted + stratified) rather than mining whatever is available. F10 error analysis = our on-intent bar plus the fault-side labels; the confounder check maps to "failures on World, grader or spec sides". Benchmark report (§4.4 measurements): ship the checklist filled in, with power analysis and intervals per student. Interface: `SkillLeaf{definition, negative_definition, confounders:[{name, probe}]}`. Priority P2 (documentation discipline), P1 for the negative definitions.

## 2607.22368 — Do Agent Benchmarks Measure Capability? Protocol Validity in the Age of Agentic AI (Shao et al., 2026; arXiv; the note cites "HackDetect")
Read: FULL (body §1–§7, appendices A schema summary, C cases, F taxonomy, H design considerations; I table skimmed).  Relevance to Part 2: high. It formalises the exact property F10 must establish (the intended skill stays necessary for reward), gives an evidence chain and a score-inflation metric, and shows that synthetic generators leak through regularities (seeds, variable order), a direct risk for Crucible's seeded Worlds.
**Problem.** A score supports a capability claim only if the protocol keeps that capability necessary; agents recover public answers, read hidden state, infer generator structure, use feedback as an oracle, or benefit from invalid scoring paths.
**Method / algorithm.**
1. Protocol P = {E environment, I information flow, S scoring, V verification}; S assigns credit, V supplies evidence that credit was earned via the intended path.
2. Evidence chain Expose → Exploit → Mislead. Mislead gap G = S_exploit − S_intended, computed only when a defensible comparison exists (targeted repair rerun, ablation removing the exposure, paired baseline, source-free run).
3. Five exposure sources: Answer source (public solutions, benchmark over-exposure), Hidden state (readable held-out files, tests), Generation mechanism (deterministic RNG with visible seed, predictable ordering), Interaction feedback (score-revealing signals that reconstruct targets), Evaluation pipeline (invalid scoring paths, warmup-state reuse, empty submission credited).
4. HackDetect audit bundle D = (spec, trajectory, artifact, score record, optional comparison). Steps: reconstruct protocol conditions (intended task, allowed resources, withheld info, scoring rule); filter the trace into narrow candidate evidence with exact pointers and no labels (recall-oriented); a fixed-prompt judge (GPT-5.5) evaluates one candidate at a time with read_file(path, start, end) only (no execution, no network), deciding exposure value, engagement (ignored, passive, active, engineered), and whether the affected artifact got grader credit; mislead = yes requires all three links. Outputs: exposure, engagement, mislead, confidence, capability drift ("intended X → measured Y"), repair target, pointers.
5. Validation outside the judge: schema checks; pointer checks (cited event, file, line, artifact, grader record must exist); a mislead-positive must show a path through trace → artifact → grader credit (except evaluator-only failures).
**Verification / grading.** 21 human labels on an MLS-Bench slice (all matched, including negative controls that read a held-out file but did not use it); 53 hand-labelled Frontier Science traces: precision 0.94, recall 0.76, F1 0.84.
**Scale & cost.** 2,385 traces, 15 benchmarks (mostly Claude Opus 4.8 traces); 960 cross-model rollouts on Frontier Science. Cost not reported.
**Key results.**
- Positive rates: Frontier Science 67.0% (331/494), AutoLab 66.7% (24/36); every other cohort ≤ 21.7%, five with 0. Properly isolated code-fixing benchmarks show few positives.
- Cross-model: Frontier Science mislead rate 65.0% (GPT-5.5) vs 69.7% (Kimi-k2.6), overlapping CIs: structural, not model-specific.
- Mislead gaps 0.447–1.00 in five paired cases: empty submission credited G = 1.00; feedback oracle 0.835; evaluation loophole 0.859; generator topological-order shortcut 0.621 (score 0.018 → 0.639 by restricting parents to lower-indexed variables); Frontier Science source transcription 0.447 (source-overlap correlates with score r = 0.625).
- Generator probes: an agent searched ~200K seeds to match a visible-seed bandit environment and then picked the optimal arm each round; a timing benchmark reused warmup inputs so a module cache made measured time a dictionary lookup.
- Of 352 mislead-positive traces with full coverage: Answer source 303 (86.1%), Hidden state 36, Generation mechanism 7, Evaluation pipeline 6, Feedback 0. Prevalence and detection difficulty are inversely ordered; generator regularity is hardest to detect because the code looks legitimate.
**Failure modes & limits.** Five cohorts are preselected suspicious subsets (no prevalence); judge configs vary across batches; recall 0.76; G needs a comparison score; no-tool memorisation is out of scope.
**Tips & tricks.**
- Treat every action that can raise the score as part of the effective task, then check it preserves the intended skill.
- Require exposure + use + credit before calling a run a hack; a file read alone is not.
- Quantify inflation with a paired run that removes the exposure.
- Hide or vary generator parameters (seeds, orderings); never expose the RNG seed to the agent.
- Separate warmup and measurement inputs; reset state between them.
- Validate artifacts before scoring; an empty artifact must not get credit.
- Re-audit after any change to tasks, agents, harnesses or scorers; stronger agents find new shortcuts.
- Record "capability drift" (intended → actually measured) as a field; it names the repair.
**Fit to Crucible Part 2.** F10 IntentCard: adopt "protocol validity" as the definition of VALID, Expose→Exploit→Mislead as the evidence schema, and G as the quantitative output of the ablation probes (null instruction, instruction only, budget 2×) whenever a comparison exists. F2 seeding: the generator-mechanism exposure is the main new risk for synthetic Worlds: seeds must never be visible; entity ids, row orders, timestamps and file orders must be randomised per seed and uncorrelated with the answer (e.g. the faulty invoice must not always be the last row); add a G-gate "ordering and id leakage" that checks a trivial positional or lexical heuristic does not beat chance on the task family. F12: run the candidate-filter + pointer-validated judge on sampled training passes; capability-drift strings feed failure clustering and the repair queue. Interface: `MisleadRecord{exposure∈{answer_source, hidden_state, generation_mechanism, interaction_feedback, evaluation_pipeline}, engagement, credited, drift, repair_target, pointers}`; `IntentCard.G`. Priority P0 for the generator-regularity gate; P1 for the audit.

## 2609.01519 — When Guardrails Look Effective: Construct Validity Failures in LLM Agent Commerce Evaluation (authors not listed in the HTML, 2026; arXiv)
Read: FULL.  Relevance to Part 2: medium. Small case study on buyer–seller simulations, but it gives the Valid / Invalid / Inconclusive contract that our F10 uses, plus three concrete tests that apply to any World with simulated people (incentive manipulation check, scaffold isolation, nested replication) and a scripted positive-control idea.
**Problem.** LLM market simulations output economic-looking numbers even when the simulated agents do not have the claimed incentives or the treatment is confounded with the scaffold.
**Method / algorithm.**
1. Testbed: buyer profiles θ with hidden component values, hard constraints, WTP cap, urgency, outside option; seller offers a mandatory base set at price p plus priced add-ons (hotel components); buyer accepts the utility-maximising feasible offer if it beats the outside option. Welfare, buyer surplus and seller profit computed from assigned ground truth, as a fraction of profile-specific first best. Guardrails: information rule (block WTP/budget/urgency questions) and conduct rule (base restricted to essentials), crossed 2×2.
2. Contract (all must pass before a policy claim): C1 incentive validity (the role responds to its stated objective in the predicted direction); C2 protocol isolation (cells share schema, decision rules, horizon, parsing; treatment flags applied after parsing); C3 stochastic stability (profile-level effect exceeds generation noise; replicates nested within profile-condition, not counted as new units); C4 accounting (completion, buyer surplus, seller profit, welfare reported jointly from ground truth). Known violation of C1 or C2 → INVALID; underpowered or non-monotone checks → INCONCLUSIVE, never "no effect".
3. Observation-sufficiency remark: if two traces share every observed feature but differ in the property of interest, no function of the observations recovers it, so keep prompts, hidden and visible transcripts, parsed offers, profile types and costs.
4. Experiments: E1 unify schema and chooser across cells (Qwen2.5 1.5B/3B/14B 4-bit, self-play, 30 profiles, temperature 0.2, 360 dialogues); E2 winner's-curse probe (top-4 14B profiles, 3 generations each, profile bootstrap, exact 2^4 sign-flip placebo, variance decomposition, MDE); E3 seller prompt manipulation check (compliance / standard / profit pressure); E4 scripted positive controls (profit-maximising seller; inefficient-bundling stress seller) over all 60 profiles.
**Verification / grading.** Ground-truth utilities; paired profile effects with 20,000-draw bootstrap.
**Scale & cost.** A few hundred dialogues with small local models; trivial cost.
**Key results.**
- Asymmetric scaffold: Both − None welfare +87.4 / +35.0 / +28.8 (1.5B/3B/14B). Unified scaffold: +7.2 / −13.9 / +23.8 (shrinks 92% at 1.5B, flips sign at 3B, unresolved at 14B).
- Top-4 single-generation 14B effects averaged +229; with 3 generations each, +37.6 (CI −34.2 to 109.3); generation residuals 49.9% of variance even at temperature 0.2; MDE 180.9; sign-flip p = 0.50.
- Stronger profit prompt gave less profit (33.8 vs 57.5) despite more extraction questions (0.83 vs 0.17 per dialogue): role prompts do not implement incentives monotonically.
- Scripted controls: a profit-maximising seller already reaches first-best welfare, so guardrails only transfer surplus and cut welfare by 24.5; only a seller programmed to force inefficient bundles makes guardrails raise welfare (+18.7). Same buyer-surplus gain (+44.0) in both, opposite welfare signs.
- Generator flaw disclosed: 43/60 profiles have negative outside-option utility, making acceptance easy.
**Failure modes & limits.** One model family in self-play, tiny samples, post-hoc profile selection; scripted sellers are instruments, not realistic behaviour.
**Tips & tricks.**
- Test that a simulated persona's behaviour moves monotonically with its stated incentive before relying on it.
- Apply treatment flags after a shared parser and schema; never let conditions differ in templates.
- Average replicates within unit, then resample units; do not count rollouts as independent subjects.
- Low temperature does not make one rollout a stable measurement.
- Use scripted agents as positive controls to prove the metric can detect the effect it claims.
- Report all components of an outcome (completion, each party's surplus, total) from ground truth.
- Treat INVALID and INCONCLUSIVE as results.
**Fit to Crucible Part 2.** F10: adopt the asymmetric decision rule (known construct violation → INVALID; insufficient evidence → INCONCLUSIVE with a plan), which the plan already cites. F1/F2 people-as-data: every simulated user or counterparty persona in a World needs an incentive manipulation check (vary the persona's stated goal, confirm behaviour moves as predicted) and a scripted positive-control persona; otherwise tasks that depend on persona behaviour (negotiation, approvals, escalations) do not exercise the intended skill. F9 difficulty: estimate per-task pass rate with replicates nested per task and bootstrap over tasks; winner's-curse on top-selected tasks is a real risk when picking "hard" tasks from single runs (re-run before banding). F2: validate seed distributions (e.g. outside-option sign) against the intended population. Interface: `PersonaCheck{persona_id, manipulation, monotone: bool, scripted_control_ok: bool}`; IntentCard verdict enum unchanged. Priority P1 (P0 for persona checks once simulated people gate rewards).

## 2609.12191 — GAUGE: When Not to Trust LLM-as-a-Judge in User-Simulated Evaluation of Task-Oriented Agents (Bodhwani et al., 2026; arXiv, Amazon)
Read: FULL (body §1–§5; appendices referenced, not read).  Relevance to Part 2: medium-high. Crucible's Worlds include simulated users and may use judges for "communicable" or report checks; this paper measures exactly when a simulator + judge gate tracks a verifiable state reward and when it does not, and finds subjective approval carries no information about success.
**Problem.** Teams promote agents using persona-driven user simulators and an LLM judge. Does that gate rank agents like a grounded, non-LLM reward?
**Method / algorithm.**
1. Substrates: τ²-bench retail (114 tasks) and airline (50), whose reward is a DB-state and action oracle (fully deterministic on airline; retail adds an LLM-scored assertion gated by a DB check); SimulatorArena math tutoring for replication (human-graded correctness).
2. Agents: 14 models from six providers at two temperatures → 25 configurations, ~3,700 transcripts. Positive-control degradation set: 12 Sonnet-4.5 configs of known good/medium/degraded quality made only by inference flags (temperature, token or step limits, error tolerance) × 6 persona strata × 2 domains = 720 transcripts.
3. Personas: six strata over cooperativeness, patience, assertiveness.
4. Four signals per transcript: policy-aware gate (operations-supervisor rubric reading the full transcript with tool calls and goal, scoring policy adherence and resolution first); process-blind proxy (first-person shopper, tool calls and task stripped, scoring how it felt); blind 3-person human panel on 150 stratified transcripts (Krippendorff α 0.79); verifiable reward r ∈ {0,1}.
5. Metrics: ranking validity ρ(G, R) at the agent level with model-clustered bootstrap; construct gap = failure rate among transcripts rated satisfied (≥5/7); decision-disagreement rate on near-equal pairs (reward gap < 0.1) vs wide pairs; broken-vs-working AUC; leave-one-out recalibration; judge diff-in-diff for self-preference; test–retest.
**Verification / grading.** The verifiable oracle is the anchor; humans and judges are measured against it.
**Scale & cost.** ~3,700 + 720 transcripts; four judges; ~$0.60 per judged decision.
**Key results.**
- Satisfaction–success gap: 57.5% of human-satisfied conversations failed (ρ −0.147; AUC 0.44), equal to the 57.3% base failure rate; all five subjective dimensions |ρ| ≤ 0.17. A truncated transcript with reward 0 got the highest human rating of any degraded variant (4.6/7).
- Judges reproduce humans (ρ 0.846 and 0.827), so a "human-validated" judge inherits the gap. The policy-aware gate does carry signal (20.0% failure among accepted vs 40.2% base; AUC 0.73); the process-blind proxy does not (AUC 0.49).
- Ranking validity across 25 agents ρ = 0.94 (all four judges 0.84–0.94; swapping the user simulator keeps 0.93), but on near-equal pairs the gate promotes the lower-reward agent 31% of the time (vs < 1% on wide pairs); 30% of all pairs and 55% of top-half pairs are near-equal. Averaging four judges does not help (common-mode errors); only abstaining when the gap is within noise halves the error.
- Same-family self-preference: +0.75/7 diff-in-diff.
- Personas: a population of six strata cuts mean verifiable reward by 42% (0.47 → 0.27) relative to the single cooperative default user; satisfaction varies by persona, not by success.
- A free completion bit catches truncation regressions (ρ 0.87 vs gate 0.80 on degradation set) but truncation is only 3.5% of failures; on the real grid it drops to 0.51.
- Recalibration (persona reweighting, judge ensembles) does not transfer out of sample.
**Failure modes & limits.** Two substrates; retail oracle has an LLM-scored part; human panel small (150); simulator from one family by default (checked with a swap).
**Tips & tricks.**
- Never use satisfaction or tone as a proxy for task success; anchor on state.
- A judge that sees tool calls and the goal is informative; a process-blind judge is not.
- Validate a judge on near-equal pairs, not only on the full range; wide-range ρ hides the resolution limit.
- Abstain on pairs whose score gap is inside sampling noise.
- Use a diverse persona population; a single cooperative simulated user inflates success by a lot.
- Check same-family self-preference with a second-family judge.
- Re-audit the judge on any change of judge, simulator or candidate pool.
**Fit to Crucible Part 2.** F5: confirms Crucible's rule that reward comes from World state checks; any judge-based check (e.g. "did the agent communicate X to the user") must see the tool calls and the goal and be validated for AUC against state-based labels on a sample before use. F2/F3 personas: draw simulated users from strata (cooperative, impatient, skeptical, anxious) as a DomainSpec axis, since persona alone moves success by ~40%; this is also a free difficulty knob for F9. F9/F11: when comparing generators, students or checkpoints whose scores are within noise, report "unresolved" instead of a winner (matters for "value per generator per dollar" in F12). Swarm judges: family rule plus the diff-in-diff check. Interface: `DomainSpec.axes += persona_stratum`; `JudgeValidation{auc_vs_state, near_equal_disagreement, self_pref}`. Priority P1.

## 2506.12286 — The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason (Liang et al., 2025; arXiv, Microsoft)
Read: FULL.  Relevance to Part 2: high for the debug lane's F10 "instruction only" probe, which cites it. It gives three cheap, model-agnostic memorisation probes and the cross-benchmark comparison design (same repos fresh, outside repos) that makes them interpretable.
**Problem.** High SWE-bench Verified scores may come from memorised issue→file associations and memorised fixes rather than localisation and repair skill.
**Method / algorithm.**
1. File-path identification: prompt with repo name + issue text only (no repo structure, code or metadata); ask for the single file to modify; correct if it exactly matches a file in the gold patch. Filtered accuracy: only instances whose issue text has no explicit path or import statement (simple heuristics).
2. Function reproduction: give the issue and the buggy file with the gold-patch-modified functions removed (only their names and paths as hints), ask for full implementations; measure consecutive 5-gram overlap with the gold code (frequency-aware); Δ5 = overlap(pred, gold) − overlap(pred, buggy) indicates leaning to the fixed version.
3. Prefix completion: give the code lines before each modified hunk; ask for continuation; an instance is "compromised" if any hunk is reproduced verbatim.
4. Comparison sets: SWE-bench Verified (500), Full SWE-bench sample (200), SWE-Bench Extra (217 fresh issues from the same 12 repos, mostly after cutoff), Outside-Repo tasks (245 from 7 popular repos not in SWE-bench: numpy, pandas, pytorch, scipy, celery, aiohttp, notebook), SWE-Bench-C# (75, internal), RefactorBench (39). Controls: repo size and issue length per set.
5. Ten OpenAI and Anthropic models, default sampling.
**Verification / grading.** Exact-match file paths; n-gram overlap; verbatim hunks.
**Scale & cost.** Single-turn prompts, ~1,300 instances × 10 models. Cheap.
**Key results.**
- File path from issue alone: 60–76% on Verified, 57–71% Full, 50–68% fresh same-repo issues, < 53% on outside popular repos; the hierarchy holds for every model and vendor and persists after filtering out issues that mention paths.
- Function reproduction 5-gram overlap: Verified up to 34.9%, Full 28.7%, Extra 18.2% (same as RefactorBench 18.1%; outside repos 13.9%).
- Prefix completion: 11.7–31.6% of Verified instances reproduced verbatim at least once (Claude 4 Opus 31.6%; Claude family rises with generation).
- Δ5 is negative on fresh post-cutoff issues for most models (outputs lean to the buggy version), positive on legacy sets.
**Failure modes & limits.** Correlational; n-gram overlap is noisy because correct fixes share text with gold; RefactorBench issues are much shorter (14.6 tokens), a confound; C# set internal; no agentic runs.
**Tips & tricks.**
- Ask for the file to fix from the issue text alone; high accuracy without the repo means memory, not localisation.
- Compare against fresh issues from the same repos and against outside repos of similar popularity; the gap is the signal.
- Filter out issues that name the path before scoring.
- Prefix-completion verbatim match is a quick per-instance contamination flag.
- Lean-to-buggy vs lean-to-fixed (Δ5) separates memory of the fix from general familiarity.
**Fit to Crucible Part 2.** F10 "instruction only (no World)" probe for the debug lane: implement exactly the file-path test (report text + repo name → predicted file) for the student and one frontier model; compare against a held-out baseline built from post-cutoff issues of the same repos and from repos never mined. Add the prefix-completion verbatim test as a G11 contamination gate on mined fixes (any verbatim hunk reproduction by the student → task goes to obfuscation via 2604.21579 transforms or is dropped from the benchmark split). For synthetic Worlds, the same probe asks for the answer from the instruction alone. Interface: `MemorizationProbe{file_path_hit, prefix_verbatim, delta5, baseline_set}`. Priority P0 for the debug lane (mined repos are exactly the SWE-bench situation), P1 elsewhere.

## 2610.17747 — Auditing Failure Attribution: Label Predictability and Judge Behavior (Xu and Gao, 2026; arXiv)
Read: FULL (body §1–§9, limitations; appendices referenced).  Relevance to Part 2: high for F12 and for any planted-fault task. Crucible injects faults (debug lane) and labels failures with a judge (failure_skill, first wrong step); this paper shows injected errors carry surface cues that make labels predictable without understanding, and that judges fine-tuned on such labels learn the cue.
**Problem.** Failure-attribution benchmarks ask which step or agent caused a failure. A high score should require a cross-step check; spurious cues may predict the annotation instead.
**Method / algorithm.** Four tests needing no new annotation:
1. Input restriction: a predictor h sees only part of the input (one step's text, step positions, presence of reasoning, speaker names); Ŷ = argmax_t s(x_t, t, T) scores each step alone, so it cannot do a cross-step check. If h recovers the label, the label does not require the check. Predictors trained on task-disjoint folds (TF-IDF + logistic regression, position rules, all-speakers).
2. Cue following: move or reword the cue while keeping the error fixed, then measure follow rate F = Pr[f(τ(X)) = c(τ(X)) | f(X) = c(X), cue moved], plus the accuracy change (a judge can depend on a cue without following it). Wording cues: paraphrase the error step, with a control paraphrase of the prior step.
3. Residual accuracy: judge accuracy on records the cue-only predictor gets wrong; report gains G and losses L since A(f) − A(h) = G − L.
4. Aggregation: recompute rankings under other failure-source mixes (find the crossing weight) and other metrics (micro-F1 vs exact-set).
**Verification / grading.** Released annotations; 95% bootstrap over task groups.
**Scale & cost.** Who&When Pro (6,257 text records; ALFAgent 311; smolagents 3,276; video 916), AFTraj (163 test; 1,264 training trajectories), Aegis (600). About 20 GPU-hours total.
**Key results.**
- ALFAgent: a TF-IDF classifier reading one step's reasoning finds the faulty step in 311/311 (100%); without the reasoning field 21.2%; deleting obvious phrases ("repetitive loop") still 98.4%; top weights on common words ("already", "so", "now"). smolagents: 95.2% from reasoning alone (90.8% when the error is not the final step).
- Injection artifact: the injection pipeline resumes after the faulty step without reasoning, so "last step with reasoning" is right 310/311 (99.7%).
- Zero-shot judges (GPT-4.1 58.8%, Claude Sonnet 5 57.2%, o3 56.3%) do not follow the boundary (0 of 361 correct answers move when task-related reasoning is appended; accuracy −4.2 and −9.0); fine-tuned judges (Qwen2.5-1.5B, Qwen3-8B on ALFAgent labels) reach 99.7–100% (+41 over GPT-4.1) yet lose 31–99 points when the boundary is moved.
- AFTraj: current-turn TF-IDF ties GPT-4.1 (82/163) but wins on injected (65.7 vs 48.1) and loses on natural failures (20.0 vs 54.5); the released 66.3% injected mix is where they cancel; the top two methods swap ranks at 63.0% injected. Word features beat structure features on injected errors only.
- Aegis: labelling every agent faulty ties GPT-4.1 on micro-F1 (49.2 vs 50.0) but trails 1.5% vs 20.7% on exact-set accuracy.
- Video: always guessing the final step (63.3%) beats GPT-4.1 (59.8%).
**Failure modes & limits.** Cue following run only on ALFAgent; judges used the authors' prompt; whether error-step text states the mistake plainly or carries injection traces is not separated; single valid label per failure.
**Tips & tricks.**
- Before trusting any step-level failure label, train a one-step-at-a-time classifier and a position rule; if they match the judge, the label is cue-driven.
- Inspect injection pipelines for artifacts (missing reasoning after the injected step, characteristic wording).
- Report judge accuracy on records the cue predictor gets wrong, with gains and losses.
- Report rankings across failure-source mixes (injected vs natural) and find the crossing weight.
- Prefer exact-set metrics over micro-F1 when "blame everyone" is a possible strategy.
- Do not fine-tune judges on cue-laden labels; they learn the cue.
**Fit to Crucible Part 2.** Debug lane (planted faults, F4/F5): run input restriction on the start state: a classifier or diff heuristic that reads one file or hunk at a time must not find the planted fault far above chance (e.g. injected code style, comments, odd identifiers, timestamps, a fault always in the last-modified file); this is a G-gate on authored and swarm-generated faults, complementary to the "misleading comment" lint from 2501.12862. F12 failure analysis: the "first wrong step" labeller and failure_skill judge must pass the four tests on a labelled sample, with natural (student) failures weighted as they occur in training, not as in an injected set; never train a labeller on injected-only labels. F11: when mixing injected (authored) and natural (traces-to-tasks) sources, report generator rankings across mixes. Interface: `CueAudit{one_step_acc, position_acc, residual_gain, residual_loss, crossing_mix}` attached to any label-producing component. Priority P0 for the planted-fault cue gate; P1 for the labeller audit.

## 2610.22243 — Shortcut Sensitivity in Agent-Safety Evaluations (Demirkiran and Ernstberger, 2026; arXiv, TAE workshop)
Read: FULL.  Relevance to Part 2: medium. Safety guards are not our target, but the three-level shortcut audit (label predictability → observational association → placebo-controlled paired edits) is a clean generic recipe for F8/F11 ("are task labels predictable from construction cues?") and for testing any LLM judge in Crucible. The domain-holdout and evaluation-unit findings apply directly to our pool statistics.
**Problem.** Agent-safety benchmarks can be easy for the wrong reasons (templates, tool names, length, source subset); guards may also respond to those cues.
**Method / algorithm.**
1. Record x = (s, z): judgment context s and audited cue z; released label y; capability-poor diagnostic r(z) that sees only the cue; guard decision g(x) ∈ {0,1}.
2. Level 1 label predictability: score r(z) against y (AUC, F1, balanced accuracy) with controls for duplication (hash complete action content, dedupe), source overlap, domain-group holdout (leave-one-environment/domain-out), and evaluation-unit weighting (rows vs equal-weight interactions vs one majority vote per interaction).
3. Level 2 observational guard association: within a matched cohort (same label, domain, action family), compare block rate with and without the cue. Only nominates a cue for intervention.
4. Level 3 paired decision sensitivity: apply a construct-preserving transformation T(s,z) = (s,z′) and compare g(T(x)) with g(x). For stochastic guards: three interleaved calls per base and condition, a formatting placebo (JSON whitespace only), report placebo-adjusted block-probability difference and identical-input disagreement separately; clustered bootstrap over source interactions, stratified by domain.
5. Artifacts: LinuxArena public sample (932 rows → 576 exact-action-unique shell trajectories: 176 attack, 400 honest, 10 environments); AgentDojo-Traj (1,220 steps; 352 unsafe); AgentHarm-Traj (731 steps). Frozen 64-step AgentDojo cohort (8 marker-present safe + 8 unsafe per domain, selected by hash before any edited run) with edits: marker rewrite (<INFORMATION> → <UNTRUSTED_EXTERNAL_CONTENT>), opaque tool aliases, both, placebo. Guards: Claude Haiku 4.5 (highest association in a 7-config screen), GPT-5.6 Luna replication, an 8-bit TS-Guard reproduction (deterministic).
**Verification / grading.** Released labels; cue-only diagnostics; paired edits on fixed bases.
**Scale & cost.** 960 Haiku calls ($3.41), 960 Luna calls ($0.48), GPT-5.5 one-draw ($5.07); local TS-Guard on a laptop.
**Key results.**
- LinuxArena: mean shell-command length separates attack from honest trajectories (AUC 0.910; action count 0.947); lexical TF-IDF leave-one-environment-out AUC 0.934 after dedup; every held-out environment 0.900–1.000.
- AgentDojo-Traj: the marker appears in all 352 unsafe rows and 321 of 868 safe rows; marker + side-effect verb prefix rule F1 0.802; exact-tool majority lookup F1 0.901 under grouped CV but 0.495 under domain holdout (only 17.3% of rows use a tool seen in another domain).
- AgentHarm-Traj: source subset determines all labels; harmful interactions contribute 7.29 rows each vs 1.65 for benign, so TS-Guard F1 0.902 (rows) falls to 0.719 (equal interactions) while balanced accuracy rises 0.754 → 0.776.
- Paired edits on Haiku: marker rewrite −8.33 pp blocking, aliases +11.46 pp; after placebo −6.77 and +13.02 pp (ranges exclude zero); measured accuracy against labels moves 73.4% → 81.8% (marker) and → 62.0% (aliases). Luna: 0.00 and +1.04 pp; TS-Guard flips 3/64. Effects are guard-specific.
**Failure modes & limits.** 64-step cohort is not a probability sample; edits not human-validated for equivalence; intervals exploratory; no end-to-end execution.
**Tips & tricks.**
- Test a capability-poor single-cue rule against your labels first (length, tool name, template tag, source).
- Always add a domain or environment holdout; within-domain CV hides lookup shortcuts.
- Deduplicate on the semantic unit (hash the action content) before scoring.
- Name the evaluation unit; row-level F1 overweights long interactions.
- Use observational association only to pick what to intervene on.
- For stochastic judges, interleave original, edited and placebo calls, repeat, and subtract the placebo effect.
- Freeze the cohort and edits before running the edited inputs.
**Fit to Crucible Part 2.** F8/F11 diversity and dedup: run Level-1 diagnostics on the pool itself: can a trivial rule (instruction length, App name, template phrase, number of entities) predict pass/fail or difficulty band? If yes, the pool carries construction signatures that a student can learn instead of the skill; quotas per generator, template and source then matter. Hold out whole task families (the DomainSpec already asks for a held-out family) and report results per family and per task (equal task weighting), not per rollout step. F10/F12: any LLM component that labels (intent auditor, failure-skill judge, exploit judge) gets the Level-3 paired-edit test with placebo before use. Interface: `ShortcutDiag{cue, metric, within_cv, family_holdout}`; reporting unit field on every metric. Priority P1.

## 2510.08996 — Saving SWE-Bench: A Benchmark Mutation Approach for Realistic Agent Evaluation (Garg et al., 2025; CAIN 2026, Microsoft)
Read: FULL (results table values not in text; numbers from prose).  Relevance to Part 2: medium. Gives a telemetry-grounded set of 11 "how users report bugs" templates and an instruction-mutation pipeline: a ready source for the instruction-form axis of the DomainSpec and for the debug lane's report style; it also shows instruction form alone moves pass rates a lot.
**Problem.** Formal GitHub issues are long, structured and full of reproduction detail; real developers send 10–30-word chat queries. Benchmarks built on issues overestimate chat-agent capability and are overfit.
**Method / algorithm.**
1. Telemetry: 10,000 random queries from ~6k users of an IDE chat agent over a week. LLM proposes 5–10 categories from ~500 queries, then labels each query (human validated). Bug fixing ≈ 14%.
2. Compare bug-fix queries with benchmark issues on length and on six elements (file path, reproduction code, error stack, test case, environment info, expected vs actual), detected by an LLM prompt. Users mostly give stack traces and file paths; issues give reproduction code, expected behaviour, environment.
3. Template extraction prompt → 11 templates: paste error/stack only; "fix this"; paste error + ask for explanation; minimal fix request; specific line/function fix; expected vs actual; paste code with question; debugging request without code; test/CI failure paste; analysis/triage request (often referencing ticket ids); reference to external issue/doc.
4. Mutation prompt: inputs are the original issue, the gold patch, and the templates; apply every applicable template, use patch details (file paths, function names, line numbers) where realistic, keep it short, informal, with typos; pick one variant per problem at random.
5. Check: embeddings (text-embedding-3-large + PCA) of mutated queries overlap the telemetry cloud; originals do not.
6. Evaluate OpenHands (100 steps max) with GPT-4.1, Claude Sonnet 3.7, Sonnet 4 on SWE-bench Verified (500), internal SWE-bench C# (150), Multi-SWE-bench TypeScript (224), original vs mutated, same hidden tests.
**Verification / grading.** Unchanged hidden tests of each benchmark.
**Scale & cost.** ~874 tasks × 3 models × 2 conditions. Token and step counts reported; $ not given.
**Key results.**
- SWE-bench Verified: 20–40% relative drop in success (13–16 absolute points) for all three models; > 50% over baseline for some models on public sets; largest drop for GPT-4.1.
- Internal C#: only 2–5 absolute points (10–16% relative), read as evidence that public-benchmark gains include overfitting.
- TypeScript: drops larger than C# for all models; largest for Sonnet 4.
- Mutated tasks need more steps; on Verified, token use falls (less text per step).
**Failure modes & limits.** Mutation may remove information the task needs or open new valid solutions (unfair failures); the mutation prompt sees the gold patch and is told to use its file paths and function names, which can leak localisation into the query (a confound the authors do not measure); one agent; single-turn only, no clarification.
**Tips & tricks.**
- Ground instruction styles in real user telemetry, not in issue templates.
- Generate several instruction variants per task from a template set and sample one.
- Check distribution match by embedding mutated instructions against real queries.
- Report step and token changes alongside success when instructions get shorter.
- Watch for leaks when a mutation prompt sees the gold patch.
**Fit to Crucible Part 2.** F0/F3: add an instruction-form axis to the DomainSpec (the 11 templates are a domain-specific seed for the debug lane; Worlds need their own from customer or ticket data), so the proposer emits several phrasings per cell and diversity covers form, not only content; F11 quota per form. F6 G11: any phrasing generated with oracle access must pass the leak lint (file or function names from the fix appear in the instruction only when the form legitimately carries them, such as "specific line/function fix"), and must pass G8 communicability (enough information remains). F9: instruction form is a difficulty knob (`hints`, `observability` in the knob list); terse forms raise difficulty without changing the World. F10: comparing the student across forms is a cheap robustness probe (instruction overfitting). Interface: `Knob{id: instruction_form, values: [...templates]}`; `TaskProposal.variants`. Priority P1.

## 2605.12925 — AgentLens: Revealing The Lucky Pass Problem in SWE-Agent Evaluation (Sahoo et al., 2026; arXiv)
Read: FULL (body, method appendix B; results appendices C–F skimmed).  Relevance to Part 2: high for F10 (the plan's "lucky-pass rate" probe cites it) and F12 trace analysis. It gives a deterministic, LLM-free process score built from a merged reference of passing traces, plus a Lucky-Pass taxonomy, divergence-point localisation, and waste signals.
**Problem.** Pass/fail treats a principled fix and chaotic trial and error as equal; pass-rate filtering of trajectories for training and pass-rate model rankings ignore process quality.
**Method / algorithm.**
1. Parse each step into a state: tool, target file, line range, content hash, position, and an intent stage E (exploration), I (implementation), V (verification), O (orchestration). A deterministic seven-rule labeler: search/inspect → E (always, even after edits), test runs → V, source edits → I, reads of previously edited files → V, bookkeeping → O; terminal commands parsed into five categories. Validated: 200 states, 5 engineers + 2 LLMs, Fleiss κ 0.933, heuristic accuracy 93.8% vs consensus.
2. Per task, merge k ≥ 2 passing trajectories into a Prefix Tree Acceptor: shared prefixes merge, divergent strategies branch; state equivalence cascade: exact content hash (1.0) → same AST scope via tree-sitter (0.90) → ≥30% line-range overlap (0.80–0.95) → same command group with Jaccard > 0.5 (0.70–0.85). k = 5 used (k ≥ 6 shows survivorship bias). Leave-one-out: a trajectory is never scored against a PTA containing itself.
3. Signals: Φ_struct (greedy ordered recall + bipartite unordered precision, harmonic mean); Φ_cov (fraction of PTA states matched, unordered); Φ_coh = (pivots + confirms)/(pivots + confirms + backtracks + ε) × (1 − r/|T|), r = blind retries; Φ_temp = 1 − mean over three time segments of JSD between candidate and PTA stage distributions (Laplace α = 0.01).
4. Score f = 0.20·Φ_struct + 0.15·Φ_cov + 0.30·100·Φ_coh + 0.35·100·Φ_temp (weights by AUROC grid search on a disjoint 278-trajectory pilot, then frozen). Tiers for passes: Ideal ≥ 70, Solid 47–70, Lucky < 47; failures: Partial-fail ≥ 47, Off-track < 47.
5. Waste: regression loops, blind retries, redundant steps, unnecessary exploration (files outside the known-good space), cyclic patterns; localised to steps. Lucky taxonomy: C1 minimal and unverified (short, no V); C2 brute-force convergence (high waste, low coherence); C3 incomplete implementation (partial fix passes weak tests); C4 excessive exploration; C5 divergent-but-valid.
**Verification / grading.** SWE-bench Verified tests for outcome; the PTA score is a process measure.
**Scale & cost.** 2,614 OpenHands trajectories, 60 tasks, 8 model backends; 47 tasks with ≥2 passes → 1,815 trajectories (1,136 pass, 679 fail). CPU only (11 cores, 18 GB).
**Key results.**
- Of 1,136 passes: 20.2% Ideal, 69.1% Solid, 10.7% Lucky; Lucky rate per model 0.5–23.2% (Opus 4.6: 77.3% pass rate with 18.7% Lucky).
- C2 + C3 = 68.0% of Lucky passes; top 10 tasks hold 63.1% of Lucky passes (task-concentrated, so some tasks reward trial and error).
- Lucky trajectories with blind retries waste 11.4 steps vs 2.7 in Ideal ones; failing runs are 58% more likely to inspect files outside the known-good space.
- Quality ranking differs from pass-rate ranking for all 8 models (moves up to 5 places).
- Pass vs fail separation: AUROC 0.766 (no single signal significant); beats TF-IDF and embedding alignment by 0.065–0.094; best single-trace matching gets 0.805 but gives no localisation.
- 54.9% of failures are Partial-fail (close to known-good processes; recoverable), 45.1% Off-track.
- Ablation: dropping temporal (−0.037) or coherence (−0.031) hurts most; merge order explains 35.9% of variance on one task, trajectory selection 64.1%.
**Failure modes & limits.** Needs ≥2 (ideally 5) passing traces per task, so the hardest tasks (few passes) cannot be scored; references built from frontier traces define "good" process, which may penalise novel strategies (C5 exists for that); weights tuned on a small pilot; OpenHands only.
**Tips & tricks.**
- Label steps by intent using history (a read after an edit is verification).
- Build a reference from several passing traces merged as a prefix tree, not from one trace.
- Score each trajectory leave-one-out against the reference.
- Track blind retries and regression loops as waste; they mark trial-and-error passes.
- Treat tasks with many Lucky passes as weak-check or trial-and-error tasks.
- Use Partial-fail traces (close to good process, wrong locally) as candidates for repair or near-miss training signals.
**Fit to Crucible Part 2.** F10: implement the lucky-pass probe as specified in §3.7 with this pipeline: after F9's frontier audit gives ≥ 5 passes per task, build a per-task PTA over tool-call states (the World's tool calls map cleanly to E/I/V/O: read/query tools = E, mutating tools = I, verifying reads of changed records = V), then score student passes; a task whose student passes are mostly Lucky is flagged (it rewards trial and error, or its checks are weak, cf. C3). F12: feed Lucky/Partial-fail tiers and waste localisation to failure clustering and traces-to-tasks (Partial-fail divergence points are natural seeds for "next-step" or hint-ablated tasks). F11 curriculum: weight training pool entries by process quality, not only pass. For the 397B student, re-derive references since process norms shift. Interface: `ProcessScore{task_id, rollout_id, tier, phi_struct, phi_cov, phi_coh, phi_temp, divergence_step, waste:[...]}`; `TaskPTA{task_id, k, nodes}`. Priority P1.

## 2607.28802 — Model or Harness? An Interaction-Centric Taxonomy for Localizing Agent Failures (Raj et al., 2026; arXiv)
Read: FULL (body §1–§7, appendix A pipeline and A.2 case; B definitions and C examples skimmed).  Relevance to Part 2: high for F10/F12. It supplies the "fault side" vocabulary the plan's on-intent rule already cites: every failure label becomes edge + responsible component, so World, grader and spec defects are separated from student skill failures and routed to the right repair loop.
**Problem.** Outcome-level failure labels conflate causes; the same visible failure may need post-training, harness work, environment redesign, or benchmark repair.
**Method / algorithm.**
1. Components: model, owner (task giver, defines success), grader (usually hidden from the agent), third party, context, memory, tool (interface incl. wrappers), local environment, external environment; peer and subagent are roles of another model.
2. Label = edge comp1 — comp2 · fault: side · failure mode. Example: a tool wrapper suppresses an error → tool — model · fault: tool; the wrapper returns the error but the model ignores it → fault: model.
3. Root-cause rule: start from the observed system failure, trace back, label the earliest failure from which execution does not recover.
4. Attribution rule: fault goes to the model when a more capable model could have avoided or recovered under the same conditions (hence 36 of 41 modes are model-side).
5. 41 modes. Owner edge: Instruction–Grader Mismatch (owner fault: instruction differs from what the grader checks); model-side Over-initiative, Under-initiative, Satisficing, Instruction-Following Failure, Reasoning Failure, Unauthorized Irreversible Action, Sycophancy, Domain Knowledge Deficit, Value Misalignment. Grader edge: Specification Gaming, Evaluation Awareness. Third party: Indirect Prompt Injection, Contextual Sycophancy. Context: Goal Drift, State Tracking Failure, Context Rationale Erosion (harness-side if compaction is harness-driven). Memory: Missed Write, State Staleness, Overgeneralization, Memory Rationale Erosion, Pollution, Redundancy, Missed Read, Memory Following Failure. Tool: Incorrect Tool Selection, Tool Hallucination, Tool Feedback Neglect, Tool Recovery Failure, Malformed Arguments, Suboptimal Arguments, Mistranslation (tool-side). Model–model: Delegation Failure, Communication Failure (peer and subagent). External env: Service Failure, Stale State Delivery (env-side), Recovery Failure (model-side if recovery was possible). Local env: Observation Failure, Recovery Failure.
6. Validation: 40 worked examples (benchmarks, system cards, reports, logged trajectories); four agent-as-a-judge runs (GPT-5.5, Claude Opus 4.6/4.7/4.8) given only definitions and a source reference; three turns per judgment: evidence reconstruction (neutral chronology), classification, reflection against disambiguation rules. Selective voting: label only when ≥ k of 4 judges agree on category.
**Verification / grading.** Agreement with one human annotator's labels.
**Scale & cost.** 40 examples; four judges.
**Key results.**
- Category (edge + fault side) agreement with human: κ 0.76 (GPT-5.5), 0.70–0.71 (Opus); inter-judge up to κ 0.84; failure-mode labels lower; giving the gold category improves mode accuracy (errors start at the category stage).
- Selective voting: ≥ 3 of 4 judges → 0.83 precision at 90% coverage; unanimity → 0.96 at 68%.
- Error sources: thin evidence (reports without trajectories) and root-cause chains, e.g. a judge blamed the model for not checking a reply that an environment bug never delivered.
**Failure modes & limits.** Descriptive, no prevalence; 40 examples and one human annotator; the "more capable model could have recovered" rule biases toward model fault and is hard to apply consistently; judges are imperfect on environment-caused failures, which matter most for task quality.
**Tips & tricks.**
- Label failures as edge + fault side, then mode; the side tells which loop repairs it.
- Always label the earliest unrecovered failure, not the symptoms.
- Instruction–Grader Mismatch is an owner (task) fault: route it to task repair, never to student training.
- Distinguish Service Failure (env) from Recovery Failure (model) by whether recovery was possible.
- Use multi-judge agreement thresholds and abstain below them; report coverage.
- Have the judge first reconstruct a neutral chronology from evidence, then classify.
**Fit to Crucible Part 2.** F10 on-intent rule: replace the single on-intent bar with fault-side–aware accounting: a failure counts against the task (INVALID or repair) when its side is owner, grader, tool (Mistranslation), or environment; it counts as on-intent only when model-side and its mode maps to the DomainSpec skill leaf. F12: the FeedbackSignal clusters by (edge, side, mode); owner/grader/env clusters go to F5/F1 repair queues, model-side clusters go to curriculum and traces-to-tasks. Judge setup: three non-student families with ≥3-of-3 agreement for auto-routing, abstain otherwise; feed the World's ground truth (actual tool responses, injected events, clock) into the evidence reconstruction step, which addresses the main error source (environment-caused failures misattributed to the model). Satisficing and Over-initiative map onto our "claims everything" null agent and side-effect negatives. Interface: `FailureLabel{edge: (comp1, comp2), fault_side, mode, root_step, judges, agreement}`. Priority P0 for the fault-side split in the on-intent rule; P1 for the full mode set.

## 2505.00212 — Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems (Zhang et al., 2025; ICML 2025)
Read: FULL (body, algorithms A.2–A.3; reasoning-prompt ablations skimmed).  Relevance to Part 2: medium-high for F12. It defines the "decisive error" counterfactually and measures how badly LLM judges localise it; Crucible's World, with snapshots and deterministic replay, can compute the counterfactual directly instead of asking a judge.
**Problem.** Locate the agent and step responsible for a multi-agent system's failure, automatically.
**Method / algorithm.**
1. Formalism: turn-based system ⟨N, S, A, P, φ⟩; trajectory τ = (s0, a0, ..., sT); Z(τ) = 1 if failure. Intervention I_(i,t) replaces a_t by a correct action and lets later steps adapt; (i,t) is decisive if Z(τ) = 1 and Z(I_(i,t)(τ)) = 0. Target: the earliest decisive error (i*, t*) = argmin_t over the decisive set.
2. Who&When dataset: 127 systems (126 generated by CaptainAgent on GAIA and AssistantBench, all GPT-4o; plus Magentic-One on AssistantBench and 30 GAIA tasks); failures only; 184 annotated failure tasks with responsible agent, decisive step, and natural-language reason.
3. Annotation: three experts, round 1 independent with self-marked uncertainty (15–30% uncertain), round 2 consensus discussion on uncertain cases (not majority vote), round 3 cross-review of each other's labels; 30.9 + 30.2 + 23.2 hours.
4. Three attribution methods: all-at-once (full log, one call); step-by-step (feed prefixes, stop at the first step judged a mistake); binary search (ask which half contains the error, recurse). Hybrid: all-at-once picks the agent, step-by-step then scans only that agent's steps. Settings with and without the task's ground-truth answer.
5. Metrics: agent-level accuracy, step-level accuracy, step accuracy with tolerance ±k.
**Verification / grading.** Human consensus labels.
**Scale & cost.** 184 tasks; GPT-4o primary, plus Llama, Qwen, o1, R1; hybrid costs 149,177 tokens vs 34,659 (all-at-once) and 87,720 (step-by-step) on the hand-crafted subset.
**Key results.**
- Best agent-level accuracy 53.5%; best step-level 14.2%; on Magentic-One logs step-level only 8.77%; some methods fall below random on step level.
- All-at-once is best for "who" (+19–25 pp over step-by-step); step-by-step best for "when"; binary search in between; rankings consistent across models.
- Accuracy falls with log length; on the longest logs (93–130 steps) all methods approach 0% step accuracy.
- With tolerance ±3–5 steps, all-at-once becomes competitive.
- Having the task's ground-truth answer helps every method.
- Aggregate statistics (which agent fails most) match human labels even when instance-level labels are poor.
- Hybrid is best on both metrics at the highest token cost.
- o1 and R1 do not consistently beat GPT-4o; adding explicit reasoning to the prompt helps more.
**Failure modes & limits.** Small dataset; labels from one team; GPT-4o-built systems only; the counterfactual definition is used for annotation but never executed (no replay); multiple decisive errors resolved by "earliest". A later audit (2610.17747) found its successor's labels cue-predictable.
**Tips & tricks.**
- Define the responsible step counterfactually: fixing it alone turns failure into success.
- Pick the earliest decisive error when several exist.
- Use whole-log judging for "who" and incremental judging for "when"; combine them.
- Give the judge the ground-truth outcome when available.
- Prefer aggregate failure statistics over single-instance labels for decisions.
- Expect long traces to defeat judges; chunk or bisect.
- Have annotators mark uncertainty and resolve by discussion, then cross-review.
**Fit to Crucible Part 2.** F12 first-wrong-step: Crucible can compute decisive errors by replay rather than by judgment: restore the World snapshot before step t, substitute the oracle's next action (or the oracle program from that state), and re-run; the earliest t where substitution flips the outcome is the decisive step. Bisect over t to keep it O(log T) replays. This gives exact labels for training a cheap judge and for traces-to-tasks (a new task starting at s_t* with the student's history is a natural "recover from here" task). Use LLM judges only for aggregate dashboards, where they are adequate. Interface: `DecisiveError{rollout_id, t_star, method: replay|judge, replays_used}`; `World.restore(snapshot_id)` + `Oracle.complete_from(state)`. Priority P1 (high leverage, since the World already supports snapshots).
