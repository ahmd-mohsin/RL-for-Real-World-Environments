# Group 06 reading cards

**Papers:** 29. **Read FULL:** 28. **PARTIAL:** 1 (2506.04178 OpenThoughts: §1–6 and key appendices; the per-source result tables skimmed). **ABSTRACT-ONLY:** 0. 2509.08724 (SWE-Mirror) has no arXiv HTML and was read from the PDF (all 21 pages).

**Five cross-paper takeaways for Part 2**

1. **Validate every check, then validate the validator, against something it never reads.** LLM-written checks fail on the gold solution 61.9% of the time, and an LLM judge does not notice (2606.16062). A collapsed always-pass verifier trains skills as well as a valid one, so student gains cannot certify a verifier (2610.11464). An LLM gate can return a constant with no error (2610.27465). So: gold-sanity on every generated check and negative (F5/G5), a frozen author-disjoint anchor set and a locked set per family, verifier strength as MCC with a task-level false-admit bound, "delete the signal" tests, and refusal whenever strength is unmeasured.
2. **Harness trust boundaries come before task-level gates.** BenchJack hacked 10 of 10 agent benchmarks to near 100% without solving a task, mostly through one all-task flaw (grader in the agent's trust domain, agent-writable test output, gold reachable from the sandbox), and code patches could not fix design flaws (2605.12673). ABC shows the same at task level: τ-bench's do-nothing agent scores 38% because "unchanged state" counts as success on impossible tasks (2507.02825). Add a harness-level trust-boundary gate per App template, broaden null agents (do nothing, empty answer, dump data, claim all), and require positive evidence for refusal tasks.
3. **Hackability and task defects concentrate where they look like difficulty.** 25–28% of SWE-bench-style tasks accept wrong patches, inflating real models' Pass@1 by +14 pp (2606.16062); 19.8% of "solved" patches are wrong under mutation-strengthened tests, and difficulty is orthogonal to check strength (2603.00520); 25.7% of tasks across 168 benchmarks have major defects (2605.26079); zero-solve tasks are enriched for bad tasks (2606.26300). Hardness never proves validity: route zero-solve and outlier tasks to a cheap agentic audit (≈$0.60/task, instruction/environment/evaluation axes, "five experts, prompt only" severity) and strengthen checks with both non-equivalent mutants (must fail) and equivalent alternatives (must pass).
4. **Trace-first generation beats instruction-first, if execution replaces the LLM judge.** Interaction-first data (NNetNav, Explorer, Go-Browse, BAGEL) outperforms instruction-first at equal budget because labels are feasible by construction; Go-Browse's frontier-over-states adds coverage and shows late start states make tasks far easier for weak models; e2h substitution and SWE-Mirror's "abstract, then re-plant" give answer-preserving difficulty and new sources. In a simulated World the trace gives the oracle (F4) and `StateAdapter.diff` gives the check (F5), replacing the 74–84%-accurate web judges these papers rely on; G8/G9 then check the instruction carries the needed information.
5. **Self-improvement loops need ML-style regularisation and multi-view evaluation.** RRSI: unregularised harness evolution had the best evolve score and no out-of-distribution gain; a noise band δ from repeated base runs, single attributable late edits, a pre-scoring leakage critic, cost-justified acceptance and pruning restored transfer at 36% fewer tokens (2609.24972). SEAGym: validation gains need not transfer, intermediate snapshots collapse (replay 34 → 6 → 43 of 80), single-source training collapses (2606.17546). Qwen's trajectory monitor with an evolving pattern set cut hacked successes 28.6% → 0.6% while clean resolution rose 40% → 61% (2606.26300). Crucible's loops should keep every artefact restorable by a write, gate on lower confidence bounds and a replay pool, and track clean vs raw pass rates.

---

## 2507.17746 — Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains (Gunjal et al., 2025; Scale AI preprint)
Read: FULL (body, all appendices incl. rubric-generation, judge and perturbation prompts).  Relevance to Part 2: medium. It is about reward shape for non-verifiable answers, not environment synthesis; it matters for Crucible's `answer_checks` with a semantic column policy and for any answer-kind that is free text (reports, summaries, customer replies).
**Problem.** RLVR needs a binary checker; open-ended expert answers (medicine, science) have none, and Likert or preference reward models overfit to length and style.
**Method / algorithm.**
1. Inputs: prompt x and a reference answer (from the source dataset, often itself LLM-written).
2. Rubric synthesis: a strong LLM (o3-mini for science, GPT-4o for medicine) writes 7–20 self-contained items per prompt, each with a title, a one-sentence description starting with its category (Essential / Important / Optional / Pitfall), and a weight (1–5 for positive items, −1 or −2 for pitfalls). Four desiderata: grounded in expert guidance (the reference), comprehensive coverage (facts, reasoning, completeness, style, safety, plus pitfalls), importance weights, and self-contained items a non-expert judge can apply in isolation.
3. Reward, two variants. Explicit: each item judged binary c_j ∈ {0,1} by a judge, reward r = Σ w_j c_j / Σ w_j, with categorical weights mapped Essential 1.0, Important 0.7, Optional 0.3, Pitfall 0.9 (pitfalls rephrased positively, "avoids X"). Implicit: the whole rubric list goes to one judge call, which returns a 1–10 holistic rating normalised to [0,1].
4. Training: GRPO, Qwen2.5-7B base, 16 rollouts per prompt, batch 96, lr 5e-6 constant with 10% warmup, 300 steps, max length 3584, temperature 1.0, judge gpt-4o-mini; one 8×H100 node.
5. Remark: RLVR is the k=1, w=1 special case where c_1 is exact match or tests.
**Verification / grading.** An LLM judge reads the rubric. A judge-alignment test: ~3,000 HealthBench clinician-ideal answers, each paired with an o3-perturbed worse version (subtle inaccuracy, dropped safety caveat, vagueness, disorder; manually screened to be objectively worse), giving 3,027 pairs; metric is pairwise preference accuracy. Hackability is not studied; a holistic implicit judge is exposed to length and rubric-echoing (the policy can restate rubric-like phrases), and nothing in the paper probes that.
**Scale & cost.** RaR-Medicine 20,166 prompts and RaR-Science 20,625, both ~7.5 items per prompt; types roughly Important 34%, Essential 29–31%, Optional 22%, Pitfall 13–15%. No dollar cost reported.
**Key results.**
- HealthBench (Qwen2.5-7B): RaR-Implicit beats Direct-Likert by up to 31% relative; GPQA-Diamond up to 7% relative. RaR-Predefined (four generic rubrics for all prompts) underperforms: instance-specific rubrics are the mechanism.
- HealthBench-1k in-domain (Table 1): Expert-answer SFT 20.4%, Simple-Likert 23.9%, Reference-Likert 31.7%, RaR synthetic without reference 32.0%, RaR synthetic with reference 35.9%, RaR with human rubrics 34.8%. Reference grounding is worth ~4 points; synthetic-with-reference matches human rubrics.
- Rubric design ablation (Table 2): Essential-only 34.9%, all rubrics 37.2%, no pitfalls 37.2%, no categorical labels 38.8%. Breadth helps; weights and synthetic pitfalls do not.
- Rubric generator without reference (Table 3): GPT-4o 34.2% best; Qwen-7B 31.9% near Qwen-72B 32.7%; all below reference-grounded 35.9%.
- Judge scale (Table 11): rubric-guided judging narrows policy spread across judges (0.250–0.279 vs 0.220–0.254); biggest gain for a Qwen-7B judge (+4.7 points).
**Failure modes & limits.** Only medicine and science; no agentic or tool tasks. Synthetic pitfalls add nothing, which the authors attribute to the difficulty of anticipating real failure modes. Reference answers are themselves LLM output in much of the data. No reward-hacking or length analysis. Critique: the "31%" headline is relative to a weak Likert baseline; the margin over Reference-Likert is small outside HealthBench-in-domain.
**Tips & tricks.**
- Condition rubric generation on a reference solution; it is the single largest quality lever.
- Make every rubric item self-contained so a small judge can apply it alone.
- Prefer many specific items to a few essential ones; breadth carries signal.
- Do not spend effort on weights or synthetic pitfalls until a real failure corpus exists.
- Build judge-validation pairs by perturbing a gold answer along named axes, then screen.
**Fit to Crucible Part 2.** F5 (VerifierBundle `answer_checks` for `semantic` columns and free-text answer kinds): generate the item list from the F4 oracle's gold answer (the reference), each item a binary check. F5 negatives: the perturbation recipe (degrade the gold along named axes) is a ready negative class for answer checks; G5 then requires the rubric judge to score each perturbed answer below the gold. Pitfall items should come from F12 failure clusters (real student mistakes), not from the generator's imagination, which is what this paper's null pitfall result implies. Keep state checks as the primary reward and rubrics as a sub-score only for answer text, so the debug-style "execution decides" rule holds. Avoid holistic implicit judging as a training reward without a hackability probe (G12). Priority P1 (only once an answer-kind with free text exists).

## 2406.12045 — τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (Yao et al., 2024; Sierra preprint, later ICLR 2025)
Read: FULL (body; appendices on APIs, policies, data-generation code, trajectories skimmed for structure).  Relevance to Part 2: high. It is the reference design for "World = DB + Python tools + policy doc + simulated user, graded by end-state equality", which is exactly Crucible's F1/F2/F4/F5 for a non-code domain.
**Problem.** Agent benchmarks give all information up front and ignore users and domain rules; deployment needs multi-turn information gathering, policy compliance and consistency over many trials.
**Method / algorithm.**
1. POMDP per task: state = hidden DB state ⊗ user state; actions = DB API calls ∪ messages to the user. DB transitions are deterministic Python functions over JSON tables; user transitions are LM samples (gpt-4-0613, temperature 1.0). The user sees only the dialogue, never the tool calls. The user ends with a STOP token.
2. Three construction stages per domain:
   - Stage I (manual): co-design the simplest schemas, APIs and Markdown policy, simplified from real counterparts. Some policy rules are enforced in the API (error strings such as "payment not found"), others are left to the agent (e.g. how many bags are paid).
   - Stage II (LM-assisted): write one example entry, have gpt-4 write a Python sampler that draws numeric and categorical fields with `random` and textual fields from LM-made lists; execute, let gpt-4 repair from execution for a few rounds, then hand-polish.
   - Stage III (manual with agent runs): write a user instruction (identity, intent, preferences, personality) that admits exactly one end state under the policy; run a gpt-4-turbo FC agent; read the trajectory; tighten the instruction; repeat until no ambiguity remains. Ground-truth write actions are copied from agent trajectories and edited. Each retail task got more than 40 gpt-4-turbo trials; zero- and low-success tasks were re-examined.
3. Domains: retail (500 users, 50 products, 1,000 orders; 7 write and 8 read APIs; 115 tasks) and airline (500 users, 300 flights, 2,000 reservations; 6 write, 7 read; 50 tasks). Non-DB tools: calculate, transfer_to_human.
4. Metric pass^k: probability that all k i.i.d. trials succeed, estimated per task as C(c,k)/C(n,k), averaged; contrasts with pass@k = 1 − C(n−c,k)/C(n,k).
**Verification / grading.** r = r_action × r_output ∈ {0,1}: final DB must equal the DB produced by replaying the gold write actions (read calls are free), and every required output string (e.g. "54.04") must appear as a substring in agent-to-user messages. The authors note r=1 is necessary but not sufficient: an agent can act without the required explicit user confirmation and still score 1. Hackable surfaces: substring output check (an agent that dumps many numbers passes), no path or procedure checks, policy steps (authentication, confirmation) are unverified. Ambiguity is the main false-negative source: 4 of 40 sampled gpt-4o failures were instruction typos or ambiguities.
**Scale & cost.** 165 tasks total, hand-built. gpt-4o agent $0.38 and gpt-4 user $0.23 per retail task; ~$200 per trial over the retail set; 95.9% of agent cost is input (long policy + tool schemas). 30-action cap; agent temperature 0.
**Key results.**
- pass^1 with function calling: gpt-4o 61.2 retail / 35.2 airline; claude-3-opus 44.2 / 34.7; llama-3-70B (ReAct) 14.8 / 14.4.
- gpt-4o pass^8 in retail below 25%: consistency collapses with k.
- Policy-removal ablation: retail 61.2→56.8 (rules are commonsense, so the policy is barely used), airline 33.2→10.8 for gpt-4o: only ad-hoc rules make the policy load-bearing.
- Failure mix (36 gpt-4o retail failures): wrong argument or info ~55%, wrong decision (rule) ~25%, partial resolution of compound requests ~19%; success falls with the number of gold write actions.
- Native function calling beats text ReAct; a "think" tool did not help FC agents.
**Failure modes & limits.** User-simulator errors (typos, ambiguity, weak arithmetic, rubber-stamping agent suggestions); task curation is tuned against one agent family (gpt-4-turbo), an admitted bias; manual annotation is slow and needs domain plus agent expertise; outcome-only reward ignores procedure. Own critique: uniqueness of outcome is enforced by hand, so this does not scale to 500–1000 tasks without an automated "unique end state" check; commonsense policies leak the solution (the retail ablation is the evidence).
**Tips & tricks.**
- Grade by end-state equality after replaying gold writes; let reads be free.
- Write the user instruction so only one end state satisfies policy plus preferences.
- Encode some rules in the API (hard errors) and leave others to the agent so rule-following is tested.
- Use an LM-written seeded sampler for bulk data instead of LM-written rows.
- Run many trials per task and inspect zero-success tasks first: they are mostly annotation bugs.
- Report pass^k with the unbiased C(c,k)/C(n,k) estimator, not only pass@1.
- Ablate the policy document: if removing it barely hurts, the rules are commonsense and not exercised.
**Fit to Crucible Part 2.** F1 (AppBundle: schema + Python tools + policies_as_code, the API-enforced vs agent-enforced split as a `PolicySpec.enforced_in: api|agent` field); F2 (seeded sampler code, determinism by seed); F3/F4 (TaskProposal carries `user_intent` + `user_known_info`; oracle = gold write list); F5 (state check = replay-and-diff; output substrings as `answer_checks`; add the path checks τ-bench lacks, e.g. "confirmation before write", via `path_constraints`); G8 Communicable (τ-bench's ambiguity failures are G8/uniqueness failures: automate a "unique outcome" probe by running k diverse simulated users and requiring all oracle-consistent runs to hit the same digest); F10 intent (adopt the policy-removal ablation as a probe: a policy-heavy family must drop sharply without the policy, else the rules are not exercised); F9/F11 (pass^k as the benchmark consistency metric). Avoid hand annotation as the main path; it is the scaling bottleneck. Priority P0 for the contract shape and grading; P1 for the policy-removal probe.

## 2512.20856 — NVIDIA Nemotron 3: Efficient and Open Intelligence (NVIDIA, 2025; white paper)
Read: FULL (the white paper is short; all sections read).  Relevance to Part 2: low. It is a model-family white paper; only §2.6 (one paragraph) touches RL environments, and it gives no environment-synthesis method. The claim note 02 cites is real but unquantified.
**Problem.** An efficient open model family (Nano 30B-A3B, Super, Ultra) for agentic workloads: high throughput, 1M context, strong reasoning.
**Method / algorithm.**
- Architecture: hybrid Mamba-2 + MoE layers with a few GQA attention layers (2 KV heads), no RoPE. Nano 30B-A3B claims 3.3× the throughput of Qwen3-30B-A3B at 8k in / 16k out.
- LatentMoE (Super, Ultra): project tokens from hidden d to latent ℓ (d/ℓ ≈ 4) for routing and expert compute, and scale expert count and top-K by d/ℓ. At ~8B active / 73B total and 1T tokens: MMLU-Pro 48.30 → 52.87, Code 51.95 → 55.14 against a standard MoE (d=4096, 128 experts, top-6 vs ℓ=1024, 512 experts, top-22).
- MTP: about +2.4% average across benchmarks on an 8B-active ablation; ~97% acceptance on the first two drafted tokens.
- NVFP4 pretraining up to 25T tokens with native FP4 GEMMs; QKV/attention kept BF16, Mamba output projections MXFP8, last 15% of layers high precision; loss gap <1% (Nano) and <0.6% (A8B) vs BF16.
- Long context: CPT at 512k, SFT at 256k, an RL long-context environment up to 32k inputs. RULER at 1M: 54.19 (Nano 3) vs 23.43 (Nano 2 12B dense hybrid).
- **Multi-environment RL (§2.6):** environments for math and science, competitive coding, instruction following, SWE, search, chat, general agentic tool use, long context and more, all trained **simultaneously** in one run instead of in stages. The authors report this as more stable, less prone to reward hacking and better overall than their earlier staged recipes, which tended to degrade some capabilities. Asynchronous RL decoupling training and inference; MTP to speed rollouts; GRPO with masked importance sampling for train/rollout policy mismatch. Stack: NeMo-RL and NeMo-Gym (open source).
- Inference-time reasoning budget control by forcing the end-of-thinking token.
**Verification / grading.** Not described per environment.
**Scale & cost.** Not reported for RL. Pretraining up to 25T tokens.
**Key results.** The RL claim is qualitative: Figure 7 shows several benchmarks rising together across one Nano RL run. No ablation numbers for simultaneous vs staged training, no hacking metric, no environment counts appear in this paper (they may be in the separate Nano technical report, which I did not read).
**Failure modes & limits.** The hacking-resistance claim has no measurement attached here; treat it as an engineering observation, not evidence. Own critique: "simultaneous is less hackable" plausibly comes from reward-mixture dilution (no single verifier dominates the gradient) and from capability retention, but neither is shown.
**Tips & tricks.**
- Train all environment families in one mixed RL run rather than in sequential stages, to avoid forgetting.
- Use masked importance sampling when rollouts come from a stale or different inference engine.
- Use a draft head (MTP) to speed rollout generation in long-horizon RL.
**Fit to Crucible Part 2.** F11 (pool and curriculum): supports keeping a mixed pool across families and domains in every batch, with per-family quotas, rather than staging domains; this matches §3.5's quota design. F12: if a single family's verifier is hackable, mixing reduces its share of the gradient, but that is not a guard; G12 still applies. Part 1 (async RL path) relevance: masked IS for off-policy rollouts. No change to stage design. Priority P2; cite only as qualitative support for mixed-family batches, and do not quote it as evidence of hack resistance.

## 2509.08724 — SWE-Mirror: Scaling Issue-Resolving Datasets by Mirroring Issues Across Repositories (Wang et al., 2025; ByteDance Seed preprint)
Read: FULL (PDF, all 21 pages incl. appendix prompts; no arXiv HTML exists).  Relevance to Part 2: high. It is a task source that decouples "where the real problem came from" from "where the working environment is", which is the exact scarcity Crucible's debug lane hits (real links in about 3 of 45 repos, scarce held-out supply).
**Problem.** Real issues are plentiful, working gyms (built repos with runnable tests) are scarce and expensive (~1 GB each; 100k instances ≈ 100 TB). Synthetic bug injection (SWE-smith) scales but loses authentic problems.
**Method / algorithm.** Three phases; the target repos are ones that already have a gym (from SWE-Gym, SWE-rebench, Multi-SWE-RL), with whole-suite run ≤ 5 min and ≤ 1 GB memory, and a manual sanity check of the suite.
1. **Task collection.** Qwen3-32B reads the target's README and emits 5 keywords; GitHub search returns the top 20 repos per keyword by stars and issue count; collect their merged, closed PRs that link an issue and edit code (tests not required). An LLM filter (prompt in A.1) reads the PR body + diff and the target's README + test suite and answers: is it a non-trivial bug fix or feature (not messages/docs/deps), and does the related functionality exist in the target? Output: [True/False, reason].
2. **Task mirroring** (GPT-4o abstracts, GPT-4.1 does the rest):
   - Abstraction: distil the PR into a repo-agnostic "bug pattern" (issue type, core problem, technical context, symptoms, root-cause pattern, impact scope), stripping variable names, paths and stack traces that would mislead localisation. The prompt pushes toward systemic, multi-location patterns.
   - Mirroring validation (Test Agent, Agentless style: localise source and test files, then SEARCH/REPLACE edits): write new tests in the target's suite that pass on current code and will fail once the issue is introduced → `test.patch`.
   - Mirroring symptom (Mirror Agent, a different prompt): given the abstraction plus the paths and functions in `test.patch`, edit non-test code so the new tests fail → `mirror.patch` (comments removed). `fix.patch` = reverse of `mirror.patch`.
   - Problem statement: an LLM writes a native-sounding issue from the original issue text, `test.patch` and `fix.patch`, and few-shot SWE-Gym issues, following SWE-bench-Verified quality criteria.
3. **Task verification.** Patches must apply in order. Run the full suite in three states: after mirror (Run.log), mirror+test (Test.log), mirror+test+fix (Fix.log). Accept only if: effective tests (only PASS→PASS, FAIL→FAIL, SKIP→SKIP, NONE→FAIL between Run and Test), effective fix (at least one ANY→FAIL→PASS), no regressions (no PASS→PASS→FAIL, SKIP→SKIP→FAIL), no flaky or abnormal transitions. Multiple samples per issue are deduplicated so every instance has distinct F2P tests and a distinct fix.
**Verification / grading.** Hidden `test.patch` F2P plus P2P. Weaknesses: the tests and the bug are written by the same pipeline from the same abstraction, so the test may check the injected symptom narrowly (an overfit fix passes); bug injection is "make these tests fail", which can produce trivial, localised sabotage; no negatives or mutation testing of `test.patch`. Problem statements see `fix.patch`, a leak risk.
**Scale & cost.** 40 repos, 4 languages, 60,671 tasks (Python 31 repos / 46,820; Rust 6 / 7,183; Go 2 / 4,056; JS 1 / 2,612); fix patches ~2.4–3.3 hunks, ~37–43 lines; F2P tests per task 7.5 (Go) to 80.2 (Rust). Storage 100 GB vs SWE-Gym 6 TB and SWE-smith 295 GB. No $ figures.
**Key results.**
- LLM pre-filter on 100 hand-labelled issues: precision 84.3%, recall 86.0% (43/7 positives, 8/42 negatives).
- Mirroring yield on 100 good issues per language: Python 68%, JS 52%, Go 36%, Rust 28%, overall 46%; compile/syntax errors dominate in Rust (36%) and Go (28%), semantic errors ~30–42% everywhere.
- Human fidelity audit of 184 mirrored tasks by 3 annotators: 156/177 majority-classified (88.1%) High or Moderate consistency with the source issue; 21 inconsistent.
- SFT (not RL) on 12,456 trajectories (6,431 from Claude-3.7/4-Sonnet on a 15k subset, 3 trials, ≤100 turns, success = finish + superset of gold-fixed tests; plus 6,025 SWE-rebench): Qwen2.5-Coder-32B 6.2 → 52.2 on SWE-bench Verified, 0.67 → 21.33 on Multi-SWE-bench-Flash; 7B 1.0 → 22.8.
- Data scale (32B, error-masking): 6.2 → 35.6 at 4,096 trajectories, monotone. Error masking (keep erroneous turns in context, mask their loss) beats response-only and error pruning at every scale.
- Cross-language (7B, 512 trajectories each, Python-only eval): Go +9.2, Rust +10.3, JS +8.4 points.
**Failure modes & limits.** Compiled languages mirror poorly; 12% of mirrored tasks are inconsistent with their source; repo selection (keyword search) concentrates on Python; no RL experiments; gains are confounded with the strong Claude teacher. Own critique: no check that the mirrored task needs the original issue's skill rather than "find the sabotaged line", and the injected bug is guided by the test, so the start state may be easy to diff against git history if history is not squashed.
**Tips & tricks.**
- Abstract the source issue into a repo-agnostic pattern before porting; raw paths and names mislead localisation.
- Write the tests first, then inject the bug to break them, then reverse the injection as the gold fix.
- Use different prompts (roles) for the test writer and the bug injector.
- Classify three-state test transitions (break / break+test / fix) with explicit allowed and forbidden patterns.
- Gate cheap: an LLM "does the target have this functionality" filter at ~84% precision saves expensive mirroring.
- Train on failed steps in context with their loss masked.
**Fit to Crucible Part 2.** §3.4 task sources: a new source, **mirror** (real issue semantics, re-instantiated in a World or repo that already passes F1/F2). Generalised beyond code: abstract a real incident (ticket, trace, bug) into a domain-free pattern, then plant it in an existing seeded World. F3 (proposal grounded in a real pattern, which also forces diversity from an exogenous corpus), F4 (oracle = reverse of the planted change, solution by construction), F5/G5/G6 (the three-state transition rules are a clean generic "effective check / effective fix / no regression / no flake" gate; G6 inverse mutation is the same idea). Add what SWE-Mirror lacks: G9 (instruction-only solver) because the statement saw the fix, G12 attacker on overfit fixes, F10 intent check that the mirrored task exercises the source skill (the 12% inconsistent rate says a fidelity judge is needed, from a different family). For the debug lane specifically: mirror kept real fixes into the already-built Go and Python repos to raise held-out supply without new environment builds. Priority P1.

## 2502.13681 — Repo2Run: Automated Building Executable Environment for Code Repository at Scale (Hu et al., 2025; v4, NeurIPS-style preprint, ByteDance + HIT)
Read: FULL (body + appendices A–G; the long per-repo table skipped).  Relevance to Part 2: medium. It is F1 for the code domain (build the World), not task synthesis; its record-and-replay with rollback is a generic pattern for any World builder whose actions mutate state.
**Problem.** Building a runnable test environment per repository is the bottleneck for executable code data; a general agent (SWE-agent) fails at it, and failed commands "pollute" the container so the history cannot be replayed as a Dockerfile.
**Method / algorithm.**
- Formulation: find base image B and command sequence P with S = δ(B,P) such that all repo tests *run* (pass or fail) under pytest.
- Two phases. **Build phase**, dual environment:
  - *Internal* (a Docker container, default python:3.10): environment monitoring (ls, cat, find, pip list, pipdeptree), dependency installation via a managed *waiting list* (add package with version constraints, add a requirements file, download resolves one at a time, conflicts surfaced), test running (`runtest`, pytest), code editing (any file except original test files, which are protected so the agent cannot delete tests), free bash.
  - *External* (host terminal): **rollback**: `docker commit` a snapshot before each non-safe command; on non-zero return code restore the snapshot. A safe-command whitelist (ls, cat, grep, find, diff, etc., without output redirection) skips snapshotting. **Base image change**: switch image and discard all prior commands. **Result processor**: truncate long outputs keeping head and tail.
  - ReAct loop, gpt-4o-2024-05-13, temperature 0.2.
- **Record phase**, a rule-based Dockerfile synthesiser: successful non-safe commands → RUN (with `cd` for the working dir), `export` → ENV, edits → COPY patch + script then RUN, base change → new FROM and drop earlier lines; finally pin every installed package to the version actually installed.
**Verification / grading.** Success = Dockerfile builds (DGSR) and pytest can execute the tests in the container (EBSR), regardless of pass/fail. A test-pass-rate check on 50 sampled repos (Appendix E): Repo2Run reaches ≥ 50% passing in 62% of repos and 100% passing in 34%, so EBSR overstates "correct" environments. Hackable: the agent could make tests "run" by stubbing modules; only original test-file edits are blocked.
**Scale & cost.** Benchmark: 420 Python repos created in 2024, >100 stars, with test dirs (crawled Dec 2024). Successful builds average 29.03 min (30.7% under 10 min); manual building by 8 engineers averaged 21.33 min on a sample. About 18 dependency installs and 3.5 test runs per successful build; base image changed in 13.3% of successes.
**Key results.**
- DGSR / EBSR: Repo2Run 100% / 86.0% (361/420); LLM-reads-README 47.6% / 22.1%; SWE-agent 26.9% / 9.0%; pipreqs 29.8% / 6.0%.
- Ablations: bash-only (no dual env) 92.4% / 41.7%; no rollback 96.9% / 83.6%; no synthesiser (LLM writes the Dockerfile from history) 19.5% / 13.8%. The deterministic synthesiser is the biggest single component; rollback mainly guarantees replayability.
- Per-domain EBSR 82.8% (AI/ML) to 95.9% (SE).
- Failures (59): runtest timeout 40.7%, hardware (GPU etc.) 35.6%, dependency timeout 13.5%, repository defects 8.5%, missing token 1.7%.
**Failure modes & limits.** Python only; success metric ignores test pass rate; GPU repos fail; timeouts dominate. Own critique: the version pinning at the end is what makes replays deterministic; without it, rebuilding later diverges.
**Tips & tricks.**
- Snapshot before every mutating action and roll back on failure, so the history stays replayable.
- Keep a whitelist of read-only commands to skip snapshots cheaply.
- Never let an LLM rewrite the build script from memory; synthesise it deterministically from the executed log.
- Pin every dependency to the installed version after the build.
- Protect test files from the builder agent.
- Give the agent an explicit "change base image" action that resets history.
**Fit to Crucible Part 2.** F1 (World / App synthesis): the "LLM explores, a deterministic recorder emits the artifact" split generalises to App and seed building: the swarm agent acts through tools against a snapshotting store (agentenv SDK already has snapshot/restore), failed actions roll back, and the AppBundle/SeedBundle is compiled from the successful action log, never hand-written by the LLM. G2 determinism follows from replay plus pinning. For the debug lane: adopt rollback + recorder + version pinning for environment builds of new repos (the environment-build failure rate is a known funnel loss). Avoid EBSR-style "it runs" as a gate; require the pre-fix test status profile to match (G3/G5). Priority P1 (debug lane environment builds), P2 for the general recorder.

## 2505.23419 — SWE-bench Goes Live! (SWE-bench-Live) (Zhang et al., 2025; Microsoft, NeurIPS D&B-style preprint)
Read: FULL (body, appendices B–G incl. RepoLaunch prompts).  Relevance to Part 2: medium-high for the debug lane's held-out and benchmark supply (post-cutoff mining, per-snapshot environments, time-machine dependency pinning); low for the generic World factory.
**Problem.** SWE-bench is static (contamination), covers 12 repos, and needs heavy manual environment work (SWE-Gym: 200+ hours; Multi-SWE-bench: ~1 year, 68 annotators for 1,632 instances).
**Method / algorithm.**
1. Repo selection: GitHub API Python repos with >1,000 stars (8,577) → >200 issues+PRs, >200 forks, ≥60% Python (3,316) → valid open-source licence (2,609).
2. Issue–PR extraction with SWE-bench's script plus SWE-Fixer heuristics; the PR must modify tests; issues created after 2024-01-01 (initial release cut at 2025-04-20).
3. **RepoLaunch** (per instance snapshot at base commit, not per version tag):
   - identify setup-relevant files (CI configs, README, docs, install guides);
   - LLM selects a base image from a candidate list;
   - setup agent (ReAct in a persistent bash session; may web-search or query the issue tracker) installs from source in editable mode, avoids tox;
   - a separate **verify agent** writes the test command (must emit per-test status, e.g. `pytest -rA`), runs it without changing state, tolerates a few failures, reports issues back to the setup agent;
   - on success the container is committed as an instance-level image.
   - **Time-machine pinning:** pip's index is proxied so only package versions released before the base-commit timestamp are visible, preventing future-version breakage.
4. Validation: framework-specific log parsers; keep instances with ≥1 FAIL_TO_PASS; record PASS_TO_PASS; repeat validation several times and keep only consistent instances (flake filter).
5. Monthly updates planned; a Lite subset samples 50 per month over Oct 2024–Mar 2025 (300 instances, seed 42).
**Verification / grading.** Standard F2P + P2P on the PR's own test patch. Median F2P is 1 test (mean 5.4); P2P mean 2,953. A single F2P test makes weak-test false positives likely (the later SWE-ABS/2606.16062 line of work in this batch measures exactly that). "Tolerable failures" in environment validation leaves residual broken tests in P2P.
**Scale & cost.** 1,319 instances from 93 repos; repos average 85k Python LoC and 423 files; gold patches mean 3.3 files, 9.0 hunks, 102.6 lines (median 2 / 3 / 24). RepoLaunch success rate and $ cost are not reported in this version. All runs on one 128-core CPU server.
**Key results.**
- Best full-set resolved rate: OpenHands + Claude 3.7 Sonnet 19.25%; SWE-agent + GPT-4.1 18.57%. The same OpenHands + Claude 3.7 setup scores 43.20% on SWE-bench Verified under identical settings: more than 2× drop on fresh issues.
- Instances from the 8 original SWE-bench repos (216) resolve at 22.96% vs 18.89% for the 1,103 from other repos, although the latter repos are smaller: evidence of overfitting to SWE-bench repos.
- Difficulty: single-file patches under 5 lines solved ~48%; ≥3 files or >100 lines below 10%; ≥7 files never solved. Repos with <100 files and <20k LoC often exceed 20%; >500 files rarely exceed 5%.
- Resolved rate shows no trend with issue creation quarter (2024Q1–2025Q1).
**Failure modes & limits.** Python only; no multiple runs (budget); RepoLaunch yield undisclosed; per-instance images are storage heavy. Own critique: recency alone is a weak contamination guard for models trained after the cut; the repo-overlap result shows repo familiarity matters as much as date.
**Tips & tricks.**
- Build environments per base-commit snapshot, not per release tag.
- Proxy the package index to the base-commit date (time-machine pinning).
- Split setup and verification into two agents; the verifier may not change state.
- Require per-test status output from the test command so a parser can diff transitions.
- Repeat validation runs and drop any instance with inconsistent outcomes.
- Use gold-patch scope (files, lines) and repo size as cheap difficulty predictors.
- Compare model scores on held-out repos vs familiar repos to detect repo-level overfitting.
**Fit to Crucible Part 2.** Debug-lane F1/F2 (environment per snapshot with time-machine pinning; Crucible's mining already uses post-cutoff rules: add the index proxy for Python and the equivalent GOPROXY/npm date pinning). G10 flake bound already exceeds their "repeat and require consistency". G11 contamination: adopt the familiar-repo vs unseen-repo comparison as a benchmark audit (held-out family by owner, which Crucible already enforces). F9 cheap predictor: gold patch files/lines and repo size are free features for the difficulty predictor. Avoid trusting single-F2P tasks without G5/G12 (weak-test false positives). Priority P1 (pinning and predictor features), P2 otherwise.

## 2506.04178 — OpenThoughts: Data Recipes for Reasoning Models (Guha et al., 2025; preprint)
Read: PARTIAL (all of §1–6; appendices D, F, H.1, H.4 opening, O, R.2–R.3 read; the 3,000+ lines of per-source tables in S skimmed via the main-text summaries).  Relevance to Part 2: medium. It is SFT data for math/code/science, not environments, but it is the best published template for **stage-wise, compute-controlled ablation of a data pipeline**, and several findings (source quality over count, filters by LLM difficulty, weaker teacher better) bear on F3, F11 and F12.
**Problem.** Open reasoning-SFT recipes are opaque; which choices at each pipeline stage matter is unknown.
**Method / algorithm.**
- **Ablation protocol.** For each stage, hold every other stage fixed, generate exactly 31,600 examples (log-midpoint of 10k and 100k), fine-tune Qwen2.5-7B-Instruct, score the average over 8 dev benchmarks (AIME24, AMC23, MATH500, CodeElo, CodeForces, LCB 05/23–05/24, GPQA-D, JEEBench), pick the winner by cross-domain average, move to the next stage greedily. A held-out benchmark set (AIME25, HMMT Feb 25, HLE-MC, LCB 06/24–01/25) is touched only at the end. Standard errors are reported and "within two SE of best" is bolded. 1,000+ experiments.
- **Stages and winners:**
  1. Question sourcing: 27 code, 21 math, 14 science sources (fully synthetic, semi-synthetic, human); GPT-4o-mini for any generated sources; DeepSeek-R1 answers everything. Code sources span a 17.2-point gap. No pattern by generation type: simple synthetic sources match curated ones.
  2. Mixing: take top-N sources, N ∈ {1,2,4,8,16}. Code average: top-1 39.9, **top-2 41.3**, top-4 38.6, top-8 37.0, top-16 36.4. At most two sources wins in every domain.
  3. Question filtering: LLM methods beat fastText and embedding filters. Code: GPT-4o-mini difficulty rating on an ICPC rubric (1–10), keep hardest (43.0 avg). Math/science: have an LLM answer and keep questions with the longest responses (GPT-4.1-mini 41.9 vs GPT-4.1-nano 39.4; a stronger rater is better). Best filters add ~4–6 points over random.
  4. Dedup × answers per question: exact, fuzzy (Indel/LCS similarity) or none × 1, 4, 16 samples: differences small; 16 answers per question chosen (fewer questions, more answers is as good as more questions).
  5. Answer filtering (generate 63,200, filter, sample 31,600): no filter beats training on everything; GPT verification scored 40.0 vs no-filter 41.9 for math. Dropped.
  6. Teacher: QwQ-32B beats DeepSeek-R1 (code avg 29.5 vs 27.2; math avg 71.6 vs 64.8) although R1 is the stronger solver. Phi-4-reasoning-plus is much worse.
- Final: 850k math, 250k code, 100k science = 1.2M; input math questions raised from 1M to 3M to hit the target after filtering.
- **Decontamination:** drop a sample if Indel similarity ≥ 75% or it shares any 13-gram (Qwen2 tokenizer) with an eval item; tuned on a built set of 3,092 contaminated (exact, wrapped, synonym-swapped, reformatted) and 3,000 clean items; misses 12/3,092.
**Verification / grading.** None for the final set (no answer verification). Appendix H: verification of OpenThoughts-114K hurt the 7B student (41.9 verified vs 45.0 unverified) but helped the 32B (64.5 vs 62.1). Removing unverifiable proof questions cost 5.6 points.
**Scale & cost.** ~$300 of API per 31,600-example ablation dataset; annotation of 1.2M with QwQ-32B took 22,000 H100-hours; one 7B training run 25,000 A100-hours; one eval 32 GPU-hours.
**Key results.** OpenThinker3-7B: AIME25 53%, LCB 06/24–01/25 51%, GPQA-D 54%; +15.3, +17.2, +20.5 points over DeepSeek-R1-Distill-Qwen-7B. Stacking stage winners shifts the whole scaling curve up from 316 to 31.6k examples; largest gains from source selection, question filtering and teacher choice.
**Failure modes & limits.** SFT only (no RL); greedy stage-by-stage search can miss interactions; winners chosen on cross-domain average; results may differ at larger student scale (the verification flip between 7B and 32B shows this). Own critique: "fewer sources is better" is measured at fixed 31.6k size; at 1M+ scale diversity may matter more.
**Tips & tricks.**
- Ablate one stage at a time at a fixed, cheap dataset size and fixed student; report standard errors.
- Keep a held-out benchmark set that no pipeline decision ever sees.
- Rank sources by downstream value and mix only the top few.
- Use an LLM's difficulty rating or response length as a question filter before paying for answers.
- A stronger rater makes a better filter.
- More samples per question can substitute for more questions.
- Check verification per student size: it can hurt small students and help large ones.
- Tune decontamination on a deliberately contaminated set with paraphrased variants.
- Pick the teacher by downstream student score, not teacher benchmark score.
**Fit to Crucible Part 2.** F12 + the loops: adopt the protocol as Crucible's **generator ablation harness**: a fixed batch size per arm, a fixed student, a frozen dev suite and a never-touched held-out family, with one stage swapped per arm (this is what §3.0 rule 1 "a stage can be swapped or ablated" needs as an actual procedure). F11: "source quality ahead of count" (already cited from the later OpenThoughts-Agent paper) is echoed here at reasoning-SFT scale; quotas should rank generators by value per dollar and drop the tail. F9 cheap predictor: LLM-rated difficulty and frontier response length are cheap pre-rollout filters. Per-student difference (verification helps 32B, hurts 7B) supports keying every pool decision by student id (§3.6). G11: copy the contaminated-testbed tuning for the debug lane's leak rules. Not directly about environments; Priority P1 for the ablation harness, P2 otherwise.

## 2012.07805 — Extracting Training Data from Large Language Models (Carlini et al., 2021; USENIX Security 2021)
Read: FULL (body + appendices A–C).  Relevance to Part 2: medium, for two places only: (1) the privacy risk of the traces-to-tasks and enterprise-ingest sources (a student trained on tasks seeded from customer data can regurgitate rare strings from them), and (2) a cheap memorisation detector for G11 contamination checks. Not about task synthesis.
**Problem.** The belief that LMs trained on deduplicated data for few epochs do not memorise individual examples; the paper shows black-box extraction of verbatim training data from GPT-2.
**Method / algorithm.**
- Definitions: a string s is *extractable* if some short prefix c makes s the model's (greedy/argmax) continuation; s is *k-eidetic memorised* if it is extractable and appears in at most k training documents (documents, not occurrences).
- Generation (200,000 samples of 256 tokens each, GPT-2 XL 1.5B), three strategies: top-n (n=40) from the empty prompt; decaying temperature (t=10 → 1 over the first 20 tokens, to diversify prefixes then follow confident paths); conditioning on 5–10 tokens of independently scraped Common Crawl text (50 MB, HTML stripped, line-deduplicated).
- Membership ranking, six metrics: raw perplexity; ratio of log-perplexity to GPT-2 Small; to GPT-2 Medium; GPT-2 log-perplexity over zlib entropy; perplexity before vs after lowercasing; minimum perplexity over a 50-token sliding window. The idea: memorised text has *unexpectedly* high likelihood relative to a reference that does not share the memorisation (smaller model, compressor, canonicalised text).
- Selection: for each of 3×6 configs, pick 100 of the top 1,000 (rank-skewed sampling), fuzzy-dedup by word-trigram multiset overlap ≥ ½.
- Ground truth: manual web search for an exact non-trivial substring, then OpenAI ran a fuzzy 3-gram match (and exact grep counts) against the real training set.
**Verification / grading.** Human + training-set lookup; the lookup has no false negatives among selected samples but cannot find memorisation the attack missed.
**Scale & cost.** 600,000 generated samples; 1,800 inspected; 604 unique memorised examples (33.5% overall precision; best config, Internet-conditioned + zlib, 67%).
**Key results.**
- Memorised content includes 46 named individuals (non-news), 32 contact-info samples (16 private individuals), 50 live URLs with random substrings, 31 code snippets (one extended to a 1,450-line file), 21 high-entropy strings (UUIDs, hashes) with ≥ 50 bits, several k=1 strings occurring in exactly one document (10–311 times within it).
- Perplexity alone is a poor detector on top-n/temperature samples (9% and 3% precision); reference-comparison metrics are much better. Small/Medium-ratio metrics find rarer content (13 and 10 high-entropy hits vs 1 for zlib).
- Natural canary study (Reddit URLs repeated in one pastebin document): XL emits every URL inserted ≥ 33 times; Medium about half; Small none (one with beam search at 359 insertions). Larger models memorise more (one setting: 18× more for 1.5B than 124M).
- Memorisation is context-dependent: the right prefix extracts orders of magnitude more (π digits: 25 greedy, 799 with a descriptive prompt, 824 with richer context). The 0.1% rate is a loose lower bound.
- Mitigations: DP training (hard to define records on web data), source-level dedup and limiting any single source's contribution, PII filtering, auditing with these attacks.
**Failure modes & limits.** Untargeted attack; manual verification limits scale; GPT-2 era. Own critique: fine-tuning-stage memorisation (the setting relevant to RL on customer-derived tasks) is not measured, only raised as an open question.
**Tips & tricks.**
- Detect memorisation by likelihood relative to a reference (smaller model, zlib, lowercased text), not raw likelihood.
- Repetition within one document counts: dedup at sub-document level and cap per-source contribution.
- Larger students memorise more, so the 397B is a higher privacy risk than the 35B.
- Seed extraction probes with realistic prefixes from the target distribution.
- Plant canaries to measure memorisation directly.
**Fit to Crucible Part 2.** Enterprise ingest and traces-to-tasks (§2(b) "Enterprise data ingest with redaction", §3.4 traces source): require redaction of high-entropy identifiers and PII before any trace becomes a SeedBundle, cap how many tasks one customer record can seed (repetition drives memorisation), and plant canary strings in seeds so a post-training extraction probe on the student can measure leakage, more so for the 397B. G11 contamination: the reference-ratio trick (student log-perplexity vs zlib or a smaller model) is a cheap probe for "this task text or its gold answer is memorised", complementing the instruction-only probe in F10. Priority P1 for the canary-and-redaction rule on traces-to-tasks; P2 for the detector.

## 2412.09605 — AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials (Xu et al., 2024; ICLR 2025)
Read: FULL (body + appendices A–K incl. evaluator and labeller prompts).  Relevance to Part 2: medium. It turns human how-to documents into task specs and trajectories on live sites; the corpus-to-task-spec half is a reusable "documents to tasks" source, but the grading (a lenient VLM judge on live websites) is the opposite of Crucible's execution-decides rule.
**Problem.** GUI agents lack multi-step trajectory data; human annotation is costly.
**Method / algorithm.**
1. **Tutorial harvesting from RedPajama** (20.8B docs):
   - rule pre-filter: action-verb, UI-element and platform keywords (≥ 20 keyword hits, ≥ 4 distinct, a mandatory keyword at least twice), 200–5,000 words, how-to/guide URL patterns → 68.8M (92.69% recall on 285 hand labels);
   - GPT-4o-mini labels 90k docs as GUI tutorial or not (F1 88.5% vs humans; it caught tutorials embedded in long pages that humans missed);
   - fastText trained on those labels (F1 89.5%) runs on all 68.8M → 18.8M deduplicated tutorial candidates;
   - GPT-4o-mini "tag & paraphrase" into a schema: platform, target type, target app, **target web URL** (the site where actions happen, not the tutorial's own URL), one-sentence task, prerequisites, numbered steps, expected result ($0.89 per 1,000).
2. **Guided replay** in BrowserGym (Chromium/Playwright): a GPT-4o agent gets the structured tutorial and target URL, observes screenshot + accessibility tree (not full HTML), acts by element id; Playwright logs DOM snapshots, network, video, screenshots and the agent's reasoning per step. ~8,027 tokens per step, ~86k per task; $215 per 1,000 tasks.
3. **Evaluation:** a GPT-4o judge sees task description, actions and reasoning interleaved; outputs a trajectory verdict, step-by-step analysis and the earliest failure point. Its prompt is **lenient by design**: success if "most" steps are effective, if >8 correct actions, if a final save or post click is missing, if only one of two subtasks is done, if a print task reaches the print button.
4. Training: Qwen2-VL-7B on 10k vision trajectories (Playwright actions mapped to pyautogui); Qwen2.5-7B/32B on 6k text trajectories (AXTree + Playwright).
**Verification / grading.** VLM judge accuracy against a 558-sample human gold set (from 1,081 reviewed): 84.0% on replayed tutorials; the authors say it is stricter than humans in disagreements. No state-based check exists (live websites). Hackable and noisy: the leniency rules accept partial completion, and there is no check of side effects on live sites.
**Scale & cost.** 23,430 tutorials replayed → 10,398 successful trajectories over 127 websites, 11 categories, 12.1 steps average. Web-related share 0.275; replay success 39.9%. Cost per verified trajectory = T&P/web-ratio + (replay + eval)/success ≈ **$0.551**.
**Key results.**
- WebArena: Qwen2.5-7B 3.80 → 10.46; Qwen2.5-32B with AgentTrek 22.40 (GPT-4o 13.10, GPT-4 14.41).
- ScreenSpot web grounding: Qwen2-VL-7B 30.7 → 67.4 average.
- Multimodal-Mind2Web cross-domain step SR: 39.5% (20% of data) → 45.0% (100%), approaching the human-annotated Mind2Web training split (47.7%).
- **Guidance ablation:** on 400 tasks, replay with step-by-step tutorial succeeded 52% (208) vs 15.78% (63) with only the high-level goal. Reasons: the tutorial gives the target URL, a plan, and paths through hidden menus.
**Failure modes & limits.** Tutorials go stale (a booking tutorial failed because the date passed); live sites drift; judge leniency admits partial trajectories; 11 categories only. Own critique: guided replay produces demonstrations of *following instructions*, so tasks derived this way have their solution path in the prompt unless the instruction is stripped back to the goal before use as an RL task.
**Tips & tricks.**
- Cascade filters: cheap rules, then an LLM labeller on a sample, then a fastText classifier distilled from the labeller for the full corpus.
- Extract the *target* environment URL separately from the document's own URL.
- Use a written procedure as privileged guidance for the solution-finder, then drop it from the student's prompt (the 52% vs 16% gap).
- Have the judge name the earliest failing step, not only pass/fail.
- Compute cost per *verified* item with stage yields in the denominator.
**Fit to Crucible Part 2.** §3.4 sources: "procedure documents to tasks" (enterprise SOPs, runbooks, help-centre articles) as a grounding corpus for F3, with the cascade filter. F4: the guided-replay asymmetry maps onto Crucible's information separation: the oracle programmer may see the procedure (privileged_notes), the solver sees only the goal; this raises oracle success without leaking the path. F5/F6: do not adopt the VLM judge as the verifier; in Crucible the World is simulated, so replay against the World gives state checks instead. The earliest-failure-step output aligns with §2(d) "first wrong step". Avoid live-site dependence (staleness). Priority P2 (corpus source), P1 for the privileged-procedure trick in F4.

## 2502.11357 — Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents (Pahuja et al., 2025; ACL Findings 2025)
Read: FULL (body + appendices A–C; prompt appendix D skimmed).  Relevance to Part 2: medium. It is the clearest "explore first, name the task after" recipe (bottom-up task intents that respect real constraints), which is the traces-to-tasks pattern applied to self-generated exploration; grading is again an LLM judge on live sites.
**Problem.** Web-agent trajectory data is small and human-made; top-down task proposals often ask for things the site cannot do (unavailable products, colours).
**Method / algorithm.** All agents are GPT-4o, executing through Playwright on real sites (top 100 Similarweb URLs for transactional tasks + 49k Tranco URLs; harmful sites filtered), viewport up to 1980×1080, 60 parallel processes, 50 hours.
1. **Task proposer:** from the homepage screenshot + accessibility tree, propose an abstract, high-level task *and* execute the first action. Halt on CAPTCHA, login or payment prompts.
2. **Task refiner (loop):** given the current task description and full action history, predict the next action and rewrite the task to be more specific and consistent with what has been done (so it respects observed availability and options).
3. **Task summariser:** from all actions and screenshots, write the final task description: what, not how.
4. **Task verifier:** task + action history + screenshots + a markdown dump of the final page (to see beyond the viewport) → success or discard; incoherent or intent-misaligned trajectories are dropped.
5. Training: select 40k, drop trajectories with > 2 scroll actions (to avoid scroll bias), ~30k left; SFT Phi-3.5V (Explorer-4B) and Qwen2-VL-7B (Explorer-7B).
**Verification / grading.** LLM judge only. Human audit of 100 trajectories: 81% agreement; confusion (fractions) TP 0.39, FN 0.05, FP 0.14, TN 0.42, so ~26% of judged-successful trajectories are actually failures (0.14/0.53). Dominant generation errors: grounding mismatch during refinement (the grounded action does not match its natural-language form, which then corrupts the summary) and summariser hallucination (adds constraints absent from the trajectory). Others: unresponsive sites, technical blockers, step limit.
**Scale & cost.** 175k total trajectories, 94k successful (53.1%), 49k unique URLs, 7.7 steps average, 720k screenshots, 33.3M elements, 830M tokens. $0.15 per raw trajectory, **$0.28 per successful one** (vs AgentTrek $0.55, Mind2Web human $0.85). Proposal/refinement step ≈ $0.0128 each, verification ≈ $0.024, summarisation ≈ $0.026. Diversity: 53k unique proposals → 94k final descriptions, 81k unique. Complexity by step count: easy 8.2k, medium 44.3k, hard 41.2k.
**Key results.**
- Mind2Web-Live (83 reachable tasks, max of 3 runs): full task SR Phi-3.5V 2.4 → 18.1, Qwen2-VL-7B 14.5 → 19.3 (GPT-4o 25.3).
- Multimodal-Mind2Web average step SR: Explorer-7B synthetic only 43.0, Mind2Web only 49.5, both 54.3 (AgentTrek-7B both 53.2).
- MiniWob++ zero-shot: Explorer-7B 53.26 (GPT-4 53.04, AgentTrek-7B 45.28).
- In-domain 100 held-out synthetic tasks: Explorer-7B 18 vs base 6; GPT-4o 16.
- Scaling 25% → 50% → 100% of data improves all metrics (task SR more slowly than step metrics). Removing vision costs 4.8 points.
**Failure modes & limits.** Incoherent tasks from free exploration; judge false positives; login-gated content missing; closed-model cost. Own critique: the summariser writes the task after the fact, so the "intent" may be a rationalisation of random clicks; nothing checks that the summarised task is solvable from its description alone or that another run would reach the same end.
**Tips & tricks.**
- Propose abstractly, then refine the task description after each executed step so it stays feasible.
- Summarise the final task as goal-only (no how).
- Give the verifier the full final page (markdown), not only the viewport.
- Filter behavioural biases out of training data (e.g. cap scroll actions).
- Report cost per *successful* item and per-stage cost.
- Track uniqueness from proposals to final tasks as a diversity meter.
**Fit to Crucible Part 2.** Traces-to-tasks and self-play sources (§3.4): "explore, then name" is a cheap way to get tasks that are feasible in the World by construction. In Crucible the exploration trace runs in the simulated World, so the end state is a digest, the oracle is the explored action program (F4 for free), and the summarised instruction is checked by G8 communicable and G9 (a fresh solver with only the instruction must reach the same digest), which replaces Explorer's weak judge. F7 behaviour signature: summariser hallucination is exactly "claimed cell ≠ measured cell", so compute the signature from the trace and reject mismatches. Diversity: proposer conditioned on a sampled World slice; measure unique-proposal → unique-task growth. Priority P1 (as an F3/F4 source for synthetic Worlds).

## 2410.02907 — NNetNav: Unsupervised Learning of Browser Agents Through Environment Interaction in the Wild (Murty et al., 2024; ICML 2025)
Read: FULL (body + appendices B–D; prompt appendix A skimmed).  Relevance to Part 2: high for the traces-to-tasks and exploration sources. Its central idea, prune an exploration episode as soon as its prefix cannot be named as a meaningful sub-task, is a generic, cheap feasibility-and-meaning gate for any "act first, label after" generator.
**Problem.** Instruction-first synthesis samples plausible but infeasible instructions, cannot reference deep site-specific entities, and gives no control over complexity.
**Method / algorithm.** Interaction-first, all components one zero-shot LLM with different prompts (Llama-3.1-70B for the main data; gpt-4o-mini in controlled runs).
1. **Exploration policy** π_explore: seeded by a sampled user **persona** for the website; sees persona, action list, current observation (accessibility tree), action history.
2. **State-change summariser** Δ(o_t, a_t, o_{t+1}) → short text of what changed; δ_τ is the sequence.
3. **Trajectory labeller** Lf(δ_τ) → an instruction ĝ that the trajectory could have been following.
4. **Outcome reward model** s(ĝ, δ_τ) ∈ {0,1}: do the state changes satisfy ĝ?
5. **Pruning loop:** at t ∈ {4, 8, …, 40} (t_max 40): label the prefix; if the ORM gives 0, kill the episode and start a new one; else store (ĝ, τ_<t) as a demonstration and continue. One episode yields several nested demos (sub-task, then larger task).
6. **Post-hoc action relabel:** for each stored (ĝ, o_i, τ_<i), the agent policy π_LM re-generates the action that serves ĝ (exploration actions may be aimless).
7. Optional **best-of-K** (K=3): sample K−1 fresh trajectories for each ĝ with π_LM, score all K with the ORM, keep the best.
**Verification / grading.** ORM judge for data; benchmarks use WebArena's functional evaluator and WebVoyager's GPT-4V judge. No ground-truth check on live sites; on self-hosted WebArena the data still only passes an LLM ORM.
**Scale & cost.** 20 sites (15 live from WebVoyager, 5 self-hosted WebArena); 10,272 demonstrations (4,695 WA + 5,577 live) ≈ 100k state–action training instances. Complexity by action count: easy (<5) 1,946, medium (5–10) 4,901, hard (10–20) 2,368, very hard (>20) 1,057. Intents per site ~21 (WA) / 25 (live), intent-distribution perplexity 13.5 / 16.2. Pruning: over 60% of WebArena episodes stopped by step 16; 65% of MiniWoB++ episodes stopped after 4 actions (mostly execution failures). Dollar cost not reported. Safety limits on live sites: ≤ 10 parallel instances, 0.5 s between actions, no login or submissions.
**Key results.**
- Llama-3.1-8B SFT (2 epochs, lr 2e-5, 20k context): WebArena 1.0 → 16.3 (WA data) / 14.9 (all); WebVoyager 4.4 → 35.2 (live data) / 34.1 (all). Zero-shot GPT-4: 14.1 / 33.5.
- **Controlled comparison** (gpt-4o-mini generator, same ORM filter, same demo count): MiniWoB++ mean reward 0.28 zero-shot, 0.28 instruction-first SFT, **0.48 NNetNav SFT**, 0.36 NNetNav-instructions-but-redistilled; WebArena 1.0 / 4.2 / **7.2** / 6.0. The hindsight trajectories themselves matter, not only the instructions.
- Self-training: Llama-3.1-8B generating its own NNetNav data (200 episodes per site, 302 demos) lifts itself 1.0 → 5.3 on WebArena.
- Transfer: in-domain site data helps 13 of 18 sites; WebArena-only data transfers poorly to live sites; mixing has mixed per-site effects (live average −1.3 with WA data added).
- Failure attributes: weak navigation efficiency (loops, redundant steps) and extracting from structured data (tables).
**Failure modes & limits.** LLM ORM noise; no login on live sites limits task types; text-only agents. Own critique: the ORM and labeller are the same model, so mislabels are self-consistent; prefix labels at fixed 4-step intervals bias task lengths toward multiples of 4.
**Tips & tricks.**
- Act first, then label; the label is feasible by construction.
- Check prefixes at fixed intervals and prune when no meaningful sub-task fits: big compute savings, more hierarchical tasks.
- Seed exploration with personas for behavioural diversity.
- Summarise state *changes* per step and label from the change log, not raw observations.
- Re-derive each action from the hindsight instruction before training.
- Keep the hindsight trajectories; re-sampling trajectories for hindsight instructions is worse.
- Measure intent diversity as the perplexity of an intent-cluster distribution.
**Fit to Crucible Part 2.** F3/F4 traces-to-tasks and self-play sources: in a simulated World, run persona-seeded exploration through policy-visible tools; every 4 calls, compute `StateAdapter.diff` (the δ log, exact rather than LLM-summarised), have a labeller propose an instruction, and prune unless an **execution-based** check (the labelled goal's state predicate holds on the current diff) passes, replacing the LLM ORM. Each surviving prefix is a task whose oracle is the prefix action program (F4 solved by construction) and whose verifier is the diff predicate (F5). Nested prefixes give a natural difficulty ladder (knob: horizon) for §3.6 and F9. G9/G8 then check that the instruction alone suffices. Persona sampling feeds §3.5 grounding. The controlled ablation is good evidence for preferring trace-first over instruction-first generation in synthetic Worlds. Priority P0 as the design of the synthetic-World generator's exploration mode.

## 2403.08140 — BAGEL: Bootstrapping Agents by Guiding Exploration with Language (Murty et al., 2024; ICML 2024)
Read: FULL (body; prompt appendices skimmed).  Relevance to Part 2: medium. It is the precursor of NNetNav: labeller ↔ agent round trips that move an instruction–trajectory pair toward a fixed point; a cheap self-consistency gate for synthetic tasks. Used only for in-context retrieval, not RL.
**Problem.** Zero-shot LM agents fail in unfamiliar digital environments without demonstrations; human demos are costly.
**Method / algorithm.** One LM (instruction-tuned PaLM-2, temperature 1.0), five prompted roles: exploration policy (unconditioned ReAct), trajectory labeller p_label(g | τ), instruction-following agent p_agent(τ | g), demonstration filter s(g, τ) ∈ {0,1}, and instruction generator p_instruct(g) from the first observation.
1. Seed: trajectory-first (τ⁰ from exploration) or instruction-first (g⁰ from the generator).
2. Round trips: g^t ~ p_label(· | τ^t); τ^{t+1} ~ p_agent(· | g^t); stop when s(g^t, τ^t) = 1 (add to buffer) or after T_iter = 5 iterations. Episodes capped at T = 15 steps.
3. Interpretation: p_k(τ) = Σ p_{k−1}(τ′) · p_label(g′ | τ′) · p_agent(τ | g′), so iterating keeps trajectories that are both describable in language and executable. Unlike HER, it relabels from the full transition history and iterates to denoise.
4. Execution-error recovery: on an environment exception, resample the action with the error message in context, up to m = 5 times.
5. Use: retrieve top-k demos by T5-XXL instruction embedding as in-context examples at test time.
**Verification / grading.** LLM filter for data; MiniWoB++'s scripted reward mapped from [−1,1] to [0,1] over 50 seeds (10 hardest tasks, where zero-shot < 0.95); ToolQA F1 on the final answer.
**Scale & cost.** 60 seeds per MiniWoB++ task, 200 per ToolQA setting. No cost reported.
**Key results.**
- MiniWoB++: zero-shot 46.8 → trajectory-first full BAGEL 61.0 (no iterations 53.0; instruction-first 52.0 / 56.0 full). ToolQA F1: 40.9 → 43.3 instruction-first full (trajectory-first 40.9 / 42.2). Iterating adds 4–8 points (MiniWoB++) and 1.3–4.5 (ToolQA).
- Trajectory-first beats instruction-first on MiniWoB++, where the LM prior over instructions is misaligned with the interfaces.
- ICL ablation (ToolQA): random demos 38.0, shuffled instruction-trajectory pairing 41.4, correct pairing 42.2: most gain is from seeing valid trajectories, a little from the instruction→trajectory mapping.
- Execution failures fall (choose-date 1.3 → 0.1, book-flight 3.0 → 0.6 per episode).
- Removing demos that, upon execution, fail their instruction adds 7 points on MiniWoB++: the LLM filter lets incorrect demos through.
- Diversity: skewed to modes (all book-flight demos were in December; ToolQA mostly {graph}); SciRex 0 correct because synthetic questions wanted descriptive answers while test needs specific numbers.
**Failure modes & limits.** Mode collapse in exploration; mismatch with the test instruction distribution; long horizons (book-flight 5% → 15% only); LLM filter false positives. ICL only, no fine-tuning.
**Tips & tricks.**
- Iterate label → re-execute → label until the pair is self-consistent; one pass is not enough.
- Prefer trajectory-first seeding when the LM cannot guess feasible tasks from the start page.
- Feed environment exceptions back and resample the action instead of ending the episode.
- Replace LLM filters with execution checks wherever possible (+7 points here).
- Watch for parameter mode collapse (e.g. all dates in one month) and randomise seeds/knobs explicitly.
**Fit to Crucible Part 2.** F3/F4 and G2/G8: the round trip is a usable *communicability* check: label a trace → a fresh agent re-executes from the label → compare end-state digests; agreement means the instruction carries the needed information (G8) and the outcome is reproducible (G2). In Crucible the filter must be the state digest, not an LLM (the paper's own +7 result). §3.5 diversity: the December collapse is the argument for knob-vector sampling in F2/F3 rather than leaving values to the LM. Error-message resampling belongs in the harness (Part 1). Priority P2 (subsumed by the NNetNav design; keep the round-trip as a G8 probe).

## 2502.06776 — InSTA: Towards Internet-Scale Training For Agents (Trabucco et al., 2025; preprint, v-latest with Qwen3 results)
Read: FULL (body + appendices A–G; prompt listings skimmed).  Relevance to Part 2: medium. Breadth-first task generation (one task per site, 150k sites) with a propose → explore → propose-harder loop and an LLM judge with calibrated confidence; useful for the judge-confidence idea and the task-schema (task, expert steps, criteria) but grading is a model opinion.
**Problem.** Web-agent data covers ~200 human-annotated sites; human data is static and expensive.
**Method / algorithm.**
1. Sites: top 1M Common Crawl hosts by PageRank.
2. **Task proposer, phase 1:** from the URL alone, emit JSON {proposed_task, steps an expert would follow, criteria (the required answer or end condition)} or "N/A" to mark the site unsafe or unsuitable (harmful, not user-facing, needs an account; tasks must not require login, personal data, purchases or orders). Temperature 0.5, 1,024 tokens. 85% of sites filtered → 150k.
3. **Exploration:** an agent attempts the easy initial task with Playwright function calls (arbitrary page methods; Markdown page encoding; ≤ 30 actions per site; one task per site to limit traffic).
4. **Task proposer, phase 2:** given URL, initial task and the exploratory trajectory, propose a harder, grounded task (one loop used; the authors intend to run it inside RL later as a curriculum).
5. **Judge:** an LLM reads task + trajectory and outputs a reasoning trace and a score r_T ∈ [0,1]; success if r_T > 0.5; confidence = 2·|r_T − ½|.
6. Training: SFT Qwen3-1.7B on trajectories from a Qwen3-235B policy, either uniformly sampled or **judge-filtered** (Judge success = 1).
**Verification / grading.** LLM judge only. On 100 human-labelled trajectories: 78.0% (Gemini 1.5 Pro), 81.7% (Llama 3.1 70B), 82.6% (GPT-4o); accuracy stable as PageRank falls; trajectories with confidence 1 are judged up to 93.1% correctly. Task verifiability: humans could complete and verify 75% (Llama 3.1 70B proposer), 85% (GPT-4o), 89% (Gemini 1.5 Pro) of initial tasks on 100 sites. Safety filter on 50 safe + 50 unsafe sites: accuracy 85% / 95% / 97%, unsafe recall 0.98–1.0. Hackable: the criteria field is free text; judge rewards plausible-looking trajectories; live content drift makes fixed answers stale (hence the judge).
**Scale & cost.** Task generation: 14 hours per phase on two 8×V100 machines. Data collection (fine-tuned 1.7B agent, Qwen3-235B judge): 150k trajectories, 2.2M screenshots and action traces, 1,200 V100-hours, $521.55 at AWS spot prices. 50% judged successful. Categories: 8,694 automatic categories, mean 16.9 tasks; 89% below the mean (long tail).
**Key results.**
- Qwen3-1.7B: 11.5% → **56.9%** success on 3,000 held-out sites (by a Qwen3-235B judge), above its own data-collection policy (Qwen3-235B) and Llama 4 Maverick, 94.7% of Gemini 2.5 Flash. Judge-filtered data scales faster than uniform sampling, and gains hold under independent secondary judges (Gemini 2.5 Flash, Llama 4 Maverick, GPT-4.1 Nano).
- Zero-shot transfer to WebVoyager matches strong LLMs for 3 of 4 judges.
- Adding 20% InSTA data to Mind2Web or WebLINX human data: same in-distribution test scores, +156.3% / +149.0% on 500 diverse InSTA test sites.
- Reasoning budget: Gemini 2.5 Flash 60% (50 tokens) → 70% (500 tokens).
- Hardest categories need memory across pages (product comparison) or actions outside Playwright (downloads).
**Failure modes & limits.** Judge-evaluated everything, including the headline (a student trained on judge-filtered data is scored by the same judge family; secondary judges partly mitigate). Only read-only tasks (no accounts, no writes), so no state-changing skills. One loop of harder-task proposal only.
**Tips & tricks.**
- Have the proposer output expert steps and an explicit criterion with every task.
- Let the proposer refuse (N/A) as an integrated safety/suitability filter, and measure its recall.
- Explore with an easy task, then propose a harder task grounded in what the exploration saw.
- Use judge confidence as a filter knob: high-confidence verdicts are much more accurate.
- Evaluate with several independent judges when no ground truth exists.
- Hold out whole sites (environments), not just tasks, for the test split.
**Fit to Crucible Part 2.** F3 (proposal schema: instruction + privileged expert steps + criterion maps to TaskProposal.instruction / privileged_notes / a draft of VerifierBundle); F3 loop "explore easy → propose harder grounded task" is a curriculum knob for F9 ladders (raise difficulty from what the student actually reached). F10/F12: where Crucible must use a model judge (semantic answer checks), keep only high-confidence verdicts and audit with a second-family judge, which this paper shows works as a filter. Split policy: held-out *Worlds* (families), as §3.5 already requires. Avoid: write-free tasks only; Crucible's value is state-changing tasks with executed checks. Priority P2.

## 2506.03533 — Go-Browse: Training Web Agents with Structured Exploration (Gandhi & Neubig, 2025; preprint)
Read: FULL (body + appendices A–D).  Relevance to Part 2: high for coverage. It turns exploration into graph search over discovered states (Go-Explore style reset-to-frontier), which is a direct recipe for **World-coverage-driven task proposal**: every reachable screen/state of an App gets explored and gets tasks, and start states can be set deep in the World ("prefixed" starts) as a difficulty knob.
**Problem.** Interaction-first methods (NNetNav) waste episodes re-discovering easy pages; instruction-first methods propose tasks from a static first page and hallucinate infeasible tasks.
**Method / algorithm.** Per website, a graph G = (URLs, trajectories), a frontier F starting at the root.
```
while F not empty:
  v = select_and_remove(F); s_v = state at v
  G_nav   = NavExplorer.propose(s_v)   # an agent (Claude-3.7, ≤15 steps) that clicks around and calls add_tasks_to_dataset() for tasks that reach neighbouring pages
  G_local = PageExplorer.propose(s_v)  # agents (GPT-4o ≤20 steps; Claude-3.7 ≤10) propose nav / info / modify tasks local to v
  for g in G_nav ∪ G_local:
     feasible, τ, v_new = FeasibilityChecker(g, s_v)   # Claude-3.7 tries up to 3 times; GPT-4o VLM judge
     if feasible: D += (g, τ); if v_new is new: add to V and F
  for feasible g:
     D += Solvers.sample(g, start=s_v,    prefixed=True)  ×2   # GPT-4o-mini, Qwen-2.5-7B
     D += Solvers.sample(g, start=root,   prefixed=False) ×2
```
Caps: 20 URLs explored per domain, ≤ 30 feasible tasks per URL, task horizon ≤ 10 steps, temperature 0.7.
**Verification / grading.** GPT-4o VLM-as-judge during collection; WebArena's programmatic task checkers at evaluation. Feasibility = at least one judged success in ≤ 3 strong-model tries.
**Scale & cost.** WebArena 5 domains × 20 URLs = 100 URLs; 26,749 trajectories (9,504 successful, 17,245 failed), 196k steps, 3,422 unique tasks. **$975.57 total** (rollouts $754.66 incl. Claude-3.7 $466; judging $220.91; 53% of input tokens were cache hits), ~3 weeks on 5 CPU nodes + 8 L40S for Qwen. Fine-tune: 8×H100, ~40 h, 2 epochs, lr 2e-5, 24k context.
**Key results.**
- WebArena SR (812 tasks): Qwen-2.5-7B 8.3 → **Go-Browse-7B 21.7**; NNetNav-7B (same recipe on 45k NNetNav steps) 18.8; GPT-4o-mini 19.3. Big gains on Shopping Admin (+11 vs NNetNav) and Reddit (+7); loses on Gitlab. Paired bootstrap: p = 0.094 vs NNetNav, 0.108 vs GPT-4o-mini (moderate).
- Online-Mind2Web (300 live tasks): Go-Browse-7B 5.33, NNetNav-7B 4.00, GPT-4o-mini 9.33; on truly out-of-domain sites NNetNav does better (mostly easy tasks).
- Successful Go-Browse trajectories reach deeper URLs (e.g. product-attribute edit pages NNetNav never reached).
- **Prefixed sampling** (start at the discovered page) has much higher success than starting at root, especially for deep nodes and weak models: decoupling navigation from local solving lets a 7B collect useful data.
- Feasibility check removed 403 tasks, saving 3.2k rollouts (~13% of steps) for the same positive data.
- **Outer-loop ablation** (30 tasks per domain): resets/tasks 1/30 → 183 unique URLs; 5/6 → 214; 15/2 → 260. Spreading proposals over more reset states raises coverage.
- Proposer models differ: Claude-3.7 proposes ~2× more tasks per step budget; GPT-4o gives more diverse state-modifying tasks; a dedicated NavExplorer doubles navigation tasks.
**Failure modes & limits.** Only 5 domains; LLM judge noise in collection; trained on successes only (39k failed steps unused); OOD generalisation is weaker than in-domain-adjacent.
**Tips & tricks.**
- Keep a frontier of discovered states and reset to them (Go-Explore) instead of re-exploring from the root.
- Make the proposer an agent that acts before proposing, so tasks are grounded in observed state.
- Split proposers by type (navigation vs local) and by model family; they cover different task clusters.
- Gate tasks by "a strong model succeeded at least once in k tries" before spending cheaper-model rollouts.
- Sample both from the deep start state and from the root: easy and hard variants of the same task.
- Measure coverage as unique states visited per budget.
**Fit to Crucible Part 2.** F3 + §3.5 diversity: implement proposal as frontier search over *World states*: nodes are (App, screen/entity view, key state fingerprint), edges are tool-call sequences; propose tasks per node; the coverage meter is "fraction of reachable node types with ≥ 1 kept task", a concrete World-facet axis for the MAP-Elites grid. F4: the strong-model feasibility try is an oracle-finding step, but in Crucible the oracle comes by construction; keep the check as a solvability audit. F9/§3.6 difficulty knobs: **start_offset** = prefixed vs root start, exactly as in the knob list; Go-Browse is direct evidence that a late start state makes a task much easier for weak models (a ladder for too-hard tasks), and that root start is the hard variant. Also adopt "propose with two families, keep the union" for proposer diversity. Priority P0 for the frontier-over-World-states proposer and the start-offset knob.

## 2505.22648 — WebDancer: Towards Autonomous Information Seeking Agency (Wu et al., 2025; Alibaba Tongyi, NeurIPS 2025)
Read: FULL (body + appendices A–E; case study skimmed).  Relevance to Part 2: medium. Two task generators (crawl-then-ask; easy-to-hard entity substitution that keeps the answer fixed) and a clean SFT-then-RL recipe; the e2h construction is a reusable, answer-preserving difficulty knob.
**Problem.** Information-seeking QA datasets are shallow (2–3 steps) or tiny (GAIA 466, WebWalkerQA 680, BrowseComp 1,266), so neither SFT nor RL has enough deep tasks.
**Method / algorithm.** Four stages.
1. **QA construction.**
   - *crawlQA:* collect root URLs of knowledge sites (arXiv, GitHub, wikis); recursively follow sublinks like a browsing human; GPT-4o writes QA pairs of designated types (COUNT, MULTI-HOP, INTERSECTION) via in-context examples. 60k pairs.
   - *e2hQA:* start from SimpleQA-style pairs with short entity answers. Iterate n times: pick an entity E_n in question Q_n; search for content C_n about it; LLM rewrites the content into a description R_n that replaces E_n in the question, giving Q_{n+1}. The answer never changes, so validity holds by construction; n controls the number of hops. 40k pairs.
2. **Trajectory rejection sampling.** ReAct with tools search (query, year filter; top-10 titles + snippets) and visit (goal, URL; a Qwen-2.5-72B summariser returns evidence + summary), and answer. Short-CoT trajectories from GPT-4o; Long-CoT from QwQ-Plus, fed actions and observations but not its previous thoughts (it was not trained on multi-step reasoning inputs), with each step's reasoning kept as the thought. Up to N = 5 samples per QA. Filter funnel: validity (ReAct format compliance), correctness (GPT-4o judge vs reference), quality (rules against hallucination and repetition: any 10-gram may occur at most 4 times; then an LLM check for non-redundancy, goal alignment, logical accuracy). Final SFT sets: 7,678 short-CoT (4.56 actions, 510 thought tokens) and 6,550 long-CoT (2.31 actions, 1,599 tokens). QA pairs that produced no SFT trajectory are kept for RL.
3. **Agent SFT**, loss masked on observation tokens: L = −(1/Σ𝟙[x_i ≠ o]) Σ 𝟙[x_i ≠ o] log π(x_i | tc, x_<i).
4. **Agent RL with DAPO** (decoupled clip ε_low/ε_high, token-level averaging, dynamic sampling that drops prompts with all-correct or all-wrong groups, which here also discards noisy synthetic QA), 16 rollouts per prompt, temperature 1.0, loss only on model tokens. Reward R = 0.1·format + 0.9·answer, both binary; answer judged by a Qwen-72B-Instruct LLM judge (answers are "inherently non-verifiable" by EM/F1). Only ~5,000 QA pairs fit in RL for compute and stability reasons. 32 nodes × 8 H20.
**Verification / grading.** LLM-as-judge for both data filtering and RL reward; GAIA/WebWalkerQA scored by an LLM judge, Pass@1. e2h's answer-invariance is the only by-construction guarantee.
**Scale & cost.** 100k QA pairs → ~14k SFT trajectories; RL on ~5k. No dollar cost.
**Key results.**
- GAIA avg Pass@1: Qwen-2.5-7B ReAct 18.4 → WebDancer 31.0; Qwen-2.5-32B 31.0 → 40.7; QwQ-32B 37.8 → **51.5** (GPT-4o ReAct 34.6; OpenAI DR 67.4). WebWalkerQA: 24.2 → 36.0, 31.9 → 38.4, 24.1 → 47.9. Best Pass@3: GAIA 64.1, WebWalkerQA 62.0.
- BrowseComp En/Zh Pass@1/Pass@3: 3.8/7.9 and 18.0/31.5.
- **RL without SFT cold start** on QwQ: Pass@3 only 5% on GAIA: SFT is essential in multi-tool settings.
- RL helps instruction models a lot (RL Pass@1 ≈ SFT Pass@3); for reasoning models gains are marginal except consistency (Cons@3), which the authors attribute to sparse reward over long trajectories.
- Long-CoT data helps the reasoning model (QwQ Pass@3 44.7 → 58.3) but raises invalid rates for small instruction models (7B: 0.65% → 21.4%, repetition overflow); mixing short and long gave no gain.
- Filtered data beats unfiltered in low-data regimes; decoding temperature barely matters; much variance comes from the live web changing. A memorisation stress test (fine-tune 7B on 69 GAIA-dev trajectories for 10 epochs) still gives only 37.4% on the same set.
**Failure modes & limits.** Only two tools; short-answer tasks only; RL uses a small subset; high rollout cost; LLM-judged rewards. (The paper's quality rule on "more than two actions" reads as a typo and is not reproducible as stated.)
**Tips & tricks.**
- Build hard questions by replacing an entity with a searched description while keeping the answer fixed; the number of substitutions is a hop-count knob.
- Keep QA pairs that failed SFT sampling for the RL stage (they are the hard ones).
- Mask observation tokens in SFT and RL losses.
- Use dynamic sampling to drop all-0/all-1 groups, which also screens noisy synthetic items.
- Cap n-gram repetition in kept trajectories (10-grams ≤ 4 times).
- Cold-start with SFT before multi-tool RL.
- Match CoT style to the student: long-CoT traces hurt small instruction models.
**Fit to Crucible Part 2.** F3/F9 difficulty: the e2h substitution generalises to Worlds as an answer-preserving *linkage* knob: replace a direct identifier in the instruction ("order #123") with an indirect description that must be resolved through other tools ("the order the customer complained about in last week's email"), keeping the gold end state identical, so solvability carries over from the base task (solution-by-construction stays valid). Each substitution must pass G8 (the description resolves uniquely in the World). §3.6: dynamic sampling plus "keep SFT failures for RL" matches band admission. Per-student difference (long-CoT harms small instruction students) supports per-student pools. Priority P1 for the substitution knob.

## 2308.06259 — Self-Alignment with Instruction Backtranslation (Li et al., 2023; ICLR 2024, Meta)
Read: FULL (body + appendices A–C; human-eval appendix D skimmed).  Relevance to Part 2: medium (conceptual). It is the origin of "start from a real artifact, infer the instruction it answers, then curate", which is the logic of traces-to-tasks: the trace/end-state is the gold output, the task statement is inferred. It also shows that curation is what makes the inferred data useful.
**Problem.** Instruction tuning needs many human (instruction, output) pairs; distillation needs a stronger model.
**Method / algorithm.**
1. Seed: 3,200 first-turn, rank-0, English (instruction, output) pairs from Open Assistant. Unlabelled: 502k self-contained segments from ClueWeb (text after an HTML header), deduplicated, length-filtered, with heuristics such as header capitalisation.
2. **Self-augment:** fine-tune base LLaMA on reversed pairs (output → instruction) to get a backward model M_yx = p(x | y); generate a candidate instruction x̂ for every segment y.
3. **Self-curate:** fine-tune M_0 on seed (instruction → output); prompt it to rate each (x̂, y) on a 5-point scale; keep score ≥ k (A_4: ≥ 4; A_5: ≥ 4.5).
4. **Iterate:** train M_1 on seed + curated set; re-score; train M_2 (two iterations).
5. Tag data sources with a system prompt (seed: "Answer in the style of an AI Assistant."; augmented: "Answer with knowledge from web search."). At inference, both prompts concatenated work best.
6. Loss on output tokens only; lr 1e-5 → 9e-6, batch 32, dropout 0.1; generation T 0.7, top-p 0.9.
**Verification / grading.** No ground truth; curation is the model's own 5-point judgement. Selection quality on 250 author-labelled pairs (20% positive): M_0 precision 0.44 / recall 0.09; M_1 0.52 / 0.44; GPT-4 0.88 / 0.92. Evaluation by AlpacaEval (GPT-4 judge vs text-davinci-003) and human pairwise preference.
**Scale & cost.** 502,133 candidates → A_4 195,043 → A_5 41,821 (curated instructions shorter, closer to seed length). No compute cost given.
**Key results.**
- Uncurated augmented data does not improve instruction following however much is added; A_5 improves steadily with quantity, contradicting the "a few thousand examples suffice" view for high-quality data.
- Data-scaling coefficient α (win rate = α log N + C): Humpback 6.95, WizardLM 5.69, Alpaca-GPT4 5.40, Vicuna 4.53, OASST 4.43, LIMA 2.86, Alpaca 1.99, FLAN v2 0.22.
- AlpacaEval win rate: Humpback 33B 79.84 (Guanaco 33B 65.96), 65B 83.71 (LIMA 65B 62.70), LLaMA-2 70B base 87.94.
- Augmented-only training (no seed) is worse than seed scaling alone; seed + curated augmented is best (complementary distributions).
- System-prompt tagging: 66.47 vs 59.96 without.
- Better curators select better data (M_1 > M_0 in precision and recall, and in downstream win rate); even low-precision selection helps.
**Failure modes & limits.** Web bias and safety not targeted; fails on precise-format or out-of-distribution outputs (ASCII art); judged by GPT-4-based metrics. Own critique: the backward model's instructions are "what would produce this text", which can under-specify constraints present in the output, the same communicability gap as G8.
**Tips & tricks.**
- Treat real artifacts as gold outputs and infer the request, instead of generating both.
- Always curate inferred pairs; quantity without curation adds nothing.
- Iterate: a better model curates better.
- Tag synthetic sources in training so the model can tell them apart; combine tags at inference.
- Keep a small in-domain seed set in every training mix.
**Fit to Crucible Part 2.** Traces-to-tasks (§3.4) and F3: enterprise logs give the gold end state (the "output"); a backward proposer writes the instruction; curation is G8 + G9 + F10 (does the instruction alone, plus the World, determine this end state; can a solver reach it). The paper's strongest lesson for Crucible is that the curation gate, not the generator, carries the value, and that a weak curator still helps if it improves across iterations (the loop in §3.8). Source tagging maps to `provenance.source` and per-source quotas in F11. Priority P2 (conceptual support; the mechanics are covered by NNetNav/Explorer cards).

## 2412.13678 — Clio: Privacy-Preserving Insights into Real-World AI Use (Tamkin et al., 2024; Anthropic)
Read: FULL (body + appendices B–D, G; the per-dataset appendix E and ethics §6 skimmed).  Relevance to Part 2: high for two mechanisms: (1) **facet → embed → cluster → contrastively label → hierarchise** is a ready design for F12 trace failure clustering and for the traces-to-tasks intent distribution; (2) its layered privacy design with an audited privacy scale is the template for enterprise traces entering the factory.
**Problem.** Learn how an assistant is used across millions of conversations without humans reading them and without leaking private data.
**Method / algorithm.**
1. **Input & screening:** random samples (by conversation, or by completion deduped per conversation); optional yes/no screener (Claude 3 Haiku, T 0.2) to narrow to a topic. Transcripts normalised to XML.
2. **Facet extraction:** per conversation, programmatic facets (turn count) and model facets (Haiku, T 0.2) answered in ≤ 2 sentences with a prefill, e.g. request ("The user's overall request for the assistant is to"), task, languages (validated by `langcodes`), a 1–5 concern score. Open-ended facets are told to omit PII and proper nouns.
3. **Base clustering:** embed summaries with all-mpnet-base-v2 (768-d), k-means with k scaled to dataset size (k withheld); enforce minimum unique accounts and conversations per cluster.
4. **Contrastive labelling:** Claude 3.5 Sonnet (T 1) sees 50 random in-cluster summaries and 50 nearest out-of-cluster summaries; writes a two-sentence description and a ≤ 10-word imperative name that distinguishes the cluster from its neighbours.
5. **Hierarchiser:** at each level, embed cluster names, k-means into neighbourhoods of ~40, Claude proposes parent clusters (also seeing the m nearest outside clusters), deduplicates proposals across neighbourhoods, assigns each child to a parent (parent list shuffled to avoid order bias), renames parents from their children; level sizes follow a geometric ratio (n_top/n_base)^{1/(L−1)}. LLM hierarchy beat HDBSCAN and agglomerative methods.
6. **Projection & UI:** UMAP (15 neighbours, min_dist 0, cosine) map, tree view, facet overlays and time trends.
7. **Privacy layers:** private-info-free summaries; cluster size thresholds; private-info-free cluster summaries; a model auditor removes any cluster scoring below 3 on a 1–5 identifiability scale (1 individual, 2 specific group/organisation, 3 thousands, 4 needs significant context, 5 not identifiable).
**Verification / grading.**
- Manual review: summaries 96% accurate (main failure: long multi-topic conversations lose a sub-request); concern score Spearman 0.84 vs humans; base cluster titles 97% accurate, ~3% of members misassigned; hierarchy titles 97%.
- End-to-end synthetic test: 19,476 generated conversations over 20 known categories in 15 languages: 94% reconstruction accuracy (84% on 19,488 concerning conversations over 18 categories), > 92% in every language.
- Privacy: auditor 98% accurate on 1,237 curated examples (607 clean, 630 with subtly inserted fictional private info). On 5,000 conversations: ~10% raw contain private info (scores 1–2), ~1.5% after summarisation, none ≤ 3 after cluster summaries.
**Scale & cost.** 1M conversations for the usage map; 2.28M for the multilingual analysis; 500k for the classifier study; **$48.81 per 100,000 conversations**.
**Key results.** Usage: coding dominates (web/mobile app development > 10%), then writing, research, education at 6–10% each; languages differ (e.g. elder care over-represented in Japanese and Chinese). Safety: found a coordinated SEO-spam network invisible per conversation; per-cluster concern score vs classifier flag rate r = 0.71; divergent clusters exposed classifier false positives (résumé help, security-adjacent code, D&D stat blocks) and false negatives (translations of explicit content; flag rates of 60% and 31% in two violating clusters).
**Failure modes & limits.** Summaries can hallucinate or drop sub-requests; k-means misgroups outliers; broad labels hide sub-topics; cannot prove absence; misses rare single events; privacy limits granularity; the authors treat outputs as leads requiring validation, not automated decisions.
**Tips & tricks.**
- Choose the facet question carefully: it decides what "similar" means for clustering.
- Label clusters contrastively with near-miss examples from neighbouring clusters.
- Build hierarchies by LLM proposal within embedding neighbourhoods, with dedup across neighbourhoods and shuffled assignment.
- Validate the whole pipeline end to end on synthetic data with a known category distribution.
- Validate any LLM auditor on a hard set made by subtly perturbing clean items.
- Stack privacy layers (summary, size threshold, re-summary, audit) and measure the residual rate at each layer.
- Compare two independent signals per cluster (e.g. judge score vs classifier rate) to find each one's blind spots.
**Fit to Crucible Part 2.** F12 FeedbackSignal.failure_clusters and §2(d) "Clustering at scale": adopt Clio's pipeline over training traces with facets = first wrong step, failure skill, fault side, tools in the failing span; contrastive labels give named failure modes that F3 can target (traces-to-tasks from failures). F10/F12 cross-signal trick: per cluster compare on-intent share vs verifier pass rate, or student pass vs frontier pass, to surface hackable or mislabelled task groups. Traces-to-tasks from customers: use Clio's privacy layers and identifiability scale as the gate before any customer trace becomes a TaskProposal; require clusters (intents) to meet a minimum-accounts threshold before a task family is created from them, so no task encodes a single customer's situation. Validation: build a synthetic trace set with known failure categories to measure the clusterer (94%-style reconstruction) before trusting it. Priority P0 for the failure-clustering design in F12; P1 for the privacy gate.

## 2403.01749 — Differentially Private Synthetic Data via Foundation Model APIs 2: Text (Aug-PE) (Xie et al., 2024; ICML 2024)
Read: FULL (body + privacy analysis and key appendices; per-size tables in C.10–C.11 skimmed).  Relevance to Part 2: medium, specific to enterprise data. It is a way to turn a customer's private records (tickets, emails, notes) into a synthetic corpus with a formal DP guarantee using only API access, which could seed World content (F2) and People without copying any real record.
**Problem.** DP synthetic text needs DP fine-tuning of a generator, impossible with closed APIs and expensive with open models.
**Method / algorithm** (Private Evolution adapted to text).
```
E_pri = Φ(S_pri)                      # embed private samples (sentence-transformer)
S_0 = RANDOM_API(N_syn · L)           # prompt LLM with the (non-private) class label + a random pseudo-subcategory keyword
for t in 0..T-1:
   E_t = Φ(S_t)                       # Aug-PE uses self-embeddings (K=0)
   H_t = DP_NN_HISTOGRAM(E_t, E_pri, σ): each private sample votes +1 for its nearest synthetic sample; add N(0, σ²) to every bin
   S'_t = top N_syn synthetic samples by H_t   (rank-based, not random draw → no duplicates)
   output S'_t
   S_{t+1} = [VARIATION_API(S'_t) × (L−1), S'_t]   # keep parents; L−1 variations each
```
- VARIATION_API by prompting: paraphrase, or fill-in-the-blanks with p% tokens masked (more masking → more diversity), with few-shot demos and a randomly sampled tone phrase. Adaptive length: target words = max(original + N(0, σ_word²), min_word), max_tokens = target × words-to-tokens ratio, so the length distribution is learned.
- Privacy: each private sample contributes one vote (sensitivity 1); Gaussian mechanism per iteration, composed adaptively over T; ε reported at 1, 2, 4, ∞. Labels assumed public; conditional generation runs per class.
**Verification / grading.** Utility by training downstream models on synthetic data and testing on real data (RoBERTa classifiers on Yelp ratings/categories and OpenReview area/recommendation; BERT-Mini/Small next-word accuracy on PubMed); distribution distance (FID, precision/recall, MAUVE, KL, TV); membership-inference AUC (PPL, REFER, LiRA) for empirical privacy.
**Scale & cost.** Yelp 1.94M reviews (5k–100k synthetic); OpenReview ICLR 2023 reviews 8.4k (2k synthetic); PubMed abstracts from Aug 2023 (2k synthetic). For 100k Yelp samples at ε = 1 with GPT-2-Large: DP-FT-Generator needs 1,764 V100-hours of DP-SGD + 7 h generation; Aug-PE needs 27 h (L = 2) or 139 h (L = 7): 65.7× / 12.7× faster.
**Key results.**
- With the same GPT-2-family generator, Aug-PE matches or beats DP fine-tuned generators on Yelp and OpenReview, loses on PubMed (GPT-2 lacks biomedical knowledge).
- With GPT-3.5 (API only), OpenReview rating accuracy drops only 45.4 → 43.1 from ε = ∞ to 1, while DP-FT-Generator GPT-2-Large falls 48.3 → 38.6 and DP-FT on the real data 65.1 → 30.5.
- MIA AUC on downstream models ≈ 50–52 for Aug-PE (chance) vs up to 77.6 (PPL) for non-private fine-tuning on real data.
- Private-data guidance is what adds value: GPT-3.5 random samples give OpenReview area 34.4; one Aug-PE iteration 39.3; full T 45.4.
- Aug-PE vs original PE (GPT-2): Yelp rating 44.9 → 67.5 (rank selection, keeping parents, L variations).
- Better embedder helps (sentence-t5-xl best on Yelp rating 67.6 vs MiniLM 62.6); more variations help (L−1 from 1 to 9: OpenReview area 39.2 → 42.7); temperature: high (1.4–1.7) for casual text, ~1.0 for technical text.
**Failure modes & limits.** Utility limited by the generator's domain knowledge; labels assumed public; ε accounting only covers the voting, not the prompts (which contain only labels); text is short-form. Own critique: DP protects individuals' contributions to the *distribution*, so rare but critical enterprise cases (one-off incidents) are exactly what gets washed out; tasks needing rare-case fidelity cannot come from this route.
**Tips & tricks.**
- Let private data only vote on synthetic candidates (nearest-neighbour histogram + Gaussian noise); never feed private text to the generator.
- Use rank-based top-N selection and keep parents to avoid duplicates and quality loss.
- Generate diversity with pseudo-subcategory keywords, tones and fill-in-the-blank masking rate.
- Learn length distributions by noisy targets in the prompt.
- Audit with membership-inference attacks after training on the synthetic data.
**Fit to Crucible Part 2.** F2 State seeding and §2(b) "Enterprise data ingest with redaction": a DP route for World *content* (ticket bodies, emails, notes, product descriptions) that matches a customer's distribution without containing their records; feeds SeedBundle text fields and People personas. It complements redaction (which keeps real records) and Clio-style aggregation (which keeps only patterns). Fidelity gate for F2: downstream-utility and distribution-distance checks (FID/MAUVE on embeddings) against the private set, computed inside the customer boundary. Not suitable for task logic or rare edge cases (DP smooths them away); those must come from redacted traces under the Clio-style gate. Priority P2 (enterprise-only, after the core factory).

## 2606.16062 — Auditing Reward Hackability in Code RL Training Environments (Rajan, 2026; single-author preprint)
Read: FULL (whole paper incl. appendix).  Relevance to Part 2: high. It is the closest published template for Crucible's G12 hackability gate plus a verifier-repair loop, and it quantifies the single most important pitfall of LLM-written tests: **61.9% of LLM-generated "blocking" tests fail on the gold solution**, which an LLM judge does not notice. Small n; treat numbers as indicative.
**Problem.** What fraction of code-RL tasks reward an incorrect solution, and how can broken tasks be repaired safely?
**Method / algorithm.**
1. **Exploit generation (attacker):** Claude Sonnet 4 writes K candidate *incorrect* patches that should pass the existing suite while changing observable behaviour. Round 1 single-shot; rounds 2–3 condition on round-1 failure logs (iterative attacker). Apply in the task's Docker via the project harness; exploit-successful if the project's runner passes. Task hackable if any candidate succeeds: H(t) > 0.
2. **Environment Quality Score:** EQS = 0.35·V + 0.30·(1 − H) + 0.20·F1 + 0.15·D, where V = verifier discrimination over candidate patches (NVIDIA-style), H = fraction of exploits that pass, F1 = three-way (correct / partial / incorrect, partial counted incorrect) agreement of a semi-formal LLM judge with execution, D = learnability from external per-task Pass@1. Verdict: KEEP > 0.70; FIX 0.40–0.70; DROP < 0.40. Weights untuned priors; headline numbers use H > 0 only.
3. **Sanity-gated repair loop for FIX tasks:**
   - augmenter proposes a blocking test (T 0.3);
   - **Stage 1 gold-sanity gate:** inject the test into the real project test file, apply the gold patch (and the task's test_patch infrastructure), run only the new test id with the project's native runner; must PASS on gold, else discard and retry (T 0.7, "assert different observable properties");
   - **Stage 2 judge:** 3-sample self-consistency semi-formal LLM judge (T 0.2/0.4/0.6, majority vote) decides whether the test blocks the target exploit;
   - task converts FIX → KEEP when every target exploit is covered; iteration budget 3.
4. **Meta-analysis:** per-model Δ_m = Pass@1(hackable) − Pass@1(robust) within the human-rated difficulty stratum (cross-model p_solve explicitly rejected as the control, since hackability inflates it); DerSimonian–Laird random effects across 134 SWE-bench Verified submissions.
**Verification / grading.** Docker execution throughout; the LLM judge is only consulted after the gate. Re-verification protocol for judge verdicts: expected PASS on gold then FAIL on exploit.
**Scale & cost.** SWE-bench Verified: 49 tasks (2 repos hackable: astropy 6, django 8), $6.04 API. R2E-Gym: 20 tasks over 6 repos (16 decisive), K = 1, $8.29 API + ~2.5 h Docker. Repair loop: $3.60 gated (111 augmentations) vs $5.57 un-gated.
**Key results.**
- Hackability: **28.5%** (14/49) of a SWE-bench Verified sample (18.4% with single-shot only; iterative rounds add 5 tasks); **25.0%** (4/16) of R2E-Gym at K = 1 (a lower bound).
- Hackable tasks inflate real model Pass@1 by **+14.14 pp** within difficulty stratum (95% CI [11.80, 16.48], I² = 0%, 123/134 models positive; unstratified +25.64 pp).
- EQS on the 50 tasks: mean 0.806; KEEP 38, FIX 11, DROP 1.
- Un-gated repair claimed 10/11 tasks fixed (EQS 0.570 → 0.816). Docker re-check of 8 judge-endorsed BLOCKS: 1 valid, 6 invalid augmentations (wrong import style, inverted expectation, wrong default value, inverted function purpose, keyword collision, test never reaches the fixed code), 1 uncollectable test. With the gate: 65 of 105 decisive augmentations fail on gold (**61.9% defect rate**); 9/11 tasks fixed (EQS 0.570 → 0.797).
- Ablation: gate without retry → 3/11; retry presence restores to 9/11; diversity-biased vs neutral "try again" prompt makes no difference (9/11 both); 1-sample judge costs one task (8/11) and saves 56% API.
**Failure modes & limits.** Small samples (49, 20, 11); a single attacker model, so rates are lower bounds; no training experiment showing that fixing tasks improves learning; EQS weights are priors. Own critique: the D(t) term rewards external solvability, which is circular for new tasks without public submissions.
**Tips & tricks.**
- Attack every task with a different-family LLM asked for wrong-but-passing solutions; iterate on failure logs.
- Run every generated check against the gold solution in the real harness before any judge sees it.
- Retry after gate failure; the retry itself matters, not its prompt style.
- Use an LLM judge only for "does this check block exploit X", after the gate.
- Control difficulty with a measure fixed before evaluation, never with cross-model solve rate.
- Score tasks continuously (EQS) and route to keep / fix / drop.
**Fit to Crucible Part 2.** F5 + F6: (a) **Gold-sanity on every check** is already in §3.1 F5 ("gold-sanity on every check"); this paper is the measured reason to make it non-negotiable for every LLM-written check and negative, including in repair rounds (61.9% defect rate). (b) G12: adopt the attacker protocol (different family, K tries, 3 rounds conditioned on prior failure logs; H(t) = 0 required), as §3.1 specifies, and record per-task H(t) for F12. (c) F6 repair: "gate → judge → retry, budget 3" maps directly onto "up to 3 repair rounds"; keep the attacker's exploits as G5 negatives for future re-checks and hold out some exploits from repair (the held-out mutant set). (d) F12/benchmark audit: the within-stratum Δ (pass rate on hackable vs robust subsets) is a way to show that hackability matters for a student, using F9 bands as strata. (e) An EQS-style routing score is compatible with GateReport, but Crucible's hard gates should stay hard. Priority P0.

## 2603.00520 — SWE-ABS: Adversarial Benchmark Strengthening Exposes Inflated Success Rates on Test-based Benchmark (2026; preprint, OpenAgentEval)
Read: FULL (body + appendices A.3–A.12; prompt listings A.1 skimmed).  Relevance to Part 2: high for F5/F6 in the debug lane and any executable-check domain: a complete, ablated recipe for strengthening checks with **mutation-driven negatives** plus a "decoupling" step that guards against overfitting to the gold solution, with measured false-negative cost.
**Problem.** PR-derived tests are necessary but not sufficient: coverage gaps (patch-affected code never executed) and semantic blind spots (executed but not asserted) let wrong patches pass.
**Method / algorithm.** Inputs per instance: issue I, gold patch P_gold, original tests T_ori = T_base ∪ T_patch. GPT-5 for all LLM steps (temperature 1.0); GLM-4.7 replication.
- **Stage I, coverage-driven:**
  1. LLM writes initial tests from I, P_gold, T_patch (diverse corner cases; T_patch shows style). Up to 3 attempts.
  2. **Decoupling:** LLM flags tests over-specialised to the gold patch (hard-coded error messages, implementation side effects) and rewrites them to test only issue resolution; keep the refined tests the gold patch still passes → T_td.
  3. **Program slicing:** Tree-sitter, intraprocedural forward + backward data/control slice from modified lines, limited to the same function/class, dependency depth ≤ 5 → L_rel.
  4. **Coverage-guided augmentation:** measure line coverage of L_rel by T_td; prompt the LLM with uncovered lines → T_cov.
- **Stage II, mutation-driven:**
  1. LLM generates ≤ 2 mutants of P_gold that pass T_ori but contain subtle semantic faults.
  2. Three LLM judges vote on issue relevance (drop irrelevant mutants) and on semantic equivalence to gold (98% agreement with humans on 100 mutants).
  3. Discrepancies: equivalent mutants that *fail* T_cov = false negatives (over-strict tests, set T_fn); non-equivalent mutants that *pass* T_cov = false positives (blind spots).
  4. Fix: an LLM relaxes T_fn assertions (keep issue check); an LLM writes targeted tests that kill each FP mutant (≤ 3 attempts each).
  5. T_aug = (T_cov \ T_fn) ∪ T_fn′ ∪ ⋃ T_mut(p).
**Verification / grading.** Re-run every leaderboard patch that passed T_ori under T_aug. Manual audit of all 500 instances for overfitting: 53 (10.6%) had gold-specific tests, corrected by hand before final numbers. 100 rejected agent patches categorised by two authors.
**Scale & cost.** 500 SWE-bench Verified + 150 SWE-bench Pro (stratified by language). 11,041 passing patches from the top-30 agents re-evaluated. $2.50 and 18.5 min per instance (81k tokens), $1,250 total; **$4.98 per strengthened instance** vs UTBoost $80. Runtime split: test generation 42%, mutant synthesis 25%, test execution 20%, slicing 8%, equivalence filtering 5%.
**Key results.**
- Strengthened 251/500 (50.2%) instances vs UTBoost 10/500; rejects **2,184/11,041 (19.78%)** previously passing patches; average agent resolve-rate drop 14.56 pp; top agent 78.80 → 62.20 (1st → 5th); Spearman ρ of leaderboard 0.82 (UTBoost 0.98); 30 rank changes in the top-30.
- SWE-bench Pro subset: 97/150 (64.7%) strengthened, average drop 16.46 pp, though agents score much lower there: **task difficulty and test strength are orthogonal**.
- GLM-4.7 vs GPT-5 on 50 instances: same 31/50 strengthened; drop 15.93 vs 19.40 pp.
- Ablation: initial generated tests 190 strengthened / 10.34 pp; + coverage 198 / 11.30; + mutation 251 / 14.56. Mutation is the bigger lever; coverage (reachability) alone adds little.
- Wrong-patch taxonomy (100): logic errors 47%, incomplete fixes 35%, type mismatches 7%, boundary 6%, off-by-one 1%, other 4%: agents produce "shallow" fixes that satisfy explicit tests.
**Failure modes & limits.** 10.6% overfitting (false-negative) rate even after decoupling, fixed manually; intraprocedural slicing misses cross-module effects; needs a gold patch; equivalence judged by LLMs. Own critique: excluding the 53 overfitting instances gives a smaller effect (206 strengthened, 11.22 pp), so the manual fix step matters for the headline.
**Tips & tricks.**
- Generate plausible-but-wrong variants of the gold solution and require the checks to kill them.
- Use equivalent variants too: a check that rejects a correct alternative is over-strict; relax it.
- Run a "decoupling" pass that strips gold-implementation specifics from generated checks.
- Coverage of the changed region is necessary but weak; assertions on outcomes are what discriminate.
- Vote equivalence/relevance with three judges.
- Measure check quality by re-scoring a population of real solver outputs (kill rate, rank change).
- Do not assume hard tasks have strong checks.
**Fit to Crucible Part 2.** F5 VerifierBundle: the NegativeSet should contain *both* non-equivalent mutants (must fail: missing, wrong, side_effect classes) and **equivalent alternatives** (must pass), the latter being G14 alternatives built automatically; the decoupling pass is a concrete method against single-reference false negatives. G5/G6: adopt the two-sided discrepancy rule (FP mutants → add checks; FN equivalents → relax checks) as the repair procedure, with gold-sanity after every change (see 2606.16062). For state-based Worlds, "mutants" are action-program mutations (drop a step, wrong entity, extra side effect), which `accept/mutation/actions.py` already hints at. F6/F12: re-score past student rollouts under strengthened verifiers and record the kill rate per task family as a verifier-weakness meter. The "difficulty ≠ check strength" result supports keeping G12 even for tasks the student rarely solves. Priority P0 (debug lane now; state-mutant version P1).

## 2605.12673 — Do Androids Dream of Breaking the Game? Systematically Auditing AI Agent Benchmarks with BenchJack (Wang et al., 2026; UC Berkeley preprint)
Read: FULL for body, checklist (App. C), cross-cutting patterns (E.11) and the patch appendix opening; the long prompt listings in App. D and per-benchmark exploit narratives in App. E skimmed.  Relevance to Part 2: high. It attacks the *harness and trust boundary*, not individual tasks: a single isolation flaw makes every task hackable regardless of how good its checks are. Crucible's G12 attacks tasks; this paper says the environment itself needs an attacker pass and a checklist before any task-level gate means anything.
**Problem.** Agent benchmarks are reward-hackable by design (shared filesystem, agent-controlled test output, eval on agent strings, answers reachable from the sandbox); post-hoc trajectory monitors are unreliable and costly.
**Method / algorithm.**
1. **Taxonomy** (from past incidents): V1 isolation failure (agent and evaluator share environment/process); V2 answers shipped with the test (gold reachable via filesystem or internet); V3 remote code execution into the evaluator (eval/exec/dynamic import on agent data); V4 LLM-judge prompt injection; V5 weak string matching; V6 evaluation-logic gaps (validators defined but never called, categories auto-scored); V7 trusting untrusted output (pass/fail derived from agent-controllable logs/files); V8 excessive permissions (root, docker socket, open network, writable host paths).
2. **Agent-Eval Checklist:** 30 binary items in 7 categories (isolation I1–I6, input handling H1–H4, judge robustness J1–J5, scoring S1–S5, evaluation logic L1–L3 incl. "run a null agent; any task scoring > 0 has a V6 flaw", permissions P1–P5, adversarial testing A1–A2).
3. **BenchJack** (orchestrator around a coding agent, Claude Code in the experiments):
   - *Reconnaissance:* set up the benchmark, map entry points, scorers, task configs, agent runtime, and the trust boundaries where the evaluator reads agent-controlled data; record task IDs.
   - *Flaw scan:* cross-reference with the taxonomy; static tools (custom semgrep rules, bandit, hadolint, Dockerfile analyser, AST trust mapper) speed up V3/V7/V8 detection; task-specific flaws inspected by the agent; findings ledger with location, explanation, severity.
   - *Exploit construction:* a `run.sh` exploit that must go through the official entry point, use a default/minimal agent (no pre-patching), and rely only on what a model could observe and do during a legitimate run; iterate until it hacks the most tasks. The exploit both proves the flaws and measures hackability.
   - Also packaged as a single coding-agent skill (more hallucinated findings, no resume).
4. **Iterative refinement:** BenchJack as attacker, a coding agent as defender patching the harness against each verified exploit; re-audit; stop when no working exploit or flaws need redesign.
**Verification / grading.** Exploits are executed through the real harness; the score itself is the check ("hard to verify findings, easy to verify scores").
**Scale & cost.** 10 benchmarks (SWE-bench Verified 500, SWE-bench Pro 731, FrontierSWE 17, MLE-bench 75, SkillsBench 88, Terminal-Bench 89, OSWorld 369, WebArena 812, NetArena 5,030, AgentBench 903 tasks). Cost not reported.
**Key results.**
- Working exploits on all 10; near-100% scores without solving any task on 9 of 10 (AgentBench below 90% only because the exploit covers one sub-suite).
- 219 distinct flaws across all 8 classes; counts concentrate in V2, V3, V6, V7; V3 most severe; V4/V5 least. Flaws either hit one task or all tasks; few V1 flaws but each exposes every task.
- Example: a nine-line `conftest.py` that rewrites pytest outcomes passes ~all SWE-bench tasks because the harness resets only files named in the test patch.
- Single-round patching kills the original exploit everywhere, but a fresh BenchJack run re-hacks most; only 4 benchmarks drop by more than half. **Design flaws (shared trust domain) cannot be patched in code; they must be redesigned.**
- On the 4 well-designed benchmarks (AgentBench, WebArena, OSWorld, SWE-bench Pro), 3 patch-and-re-hack rounds cut hackable tasks from near 100% to < 10%; WebArena and OSWorld reach 0%.
- Cross-cutting causes: grader in the agent's trust domain; pass/fail from agent-controlled stdout/files; eval/exec on agent strings; public answers; root + network in agent containers.
**Failure modes & limits.** Coding-agent auditor can hallucinate flaws (skill mode); the exploit rule ("what a model could do") is a judgement call; no cost data; benchmark-level, not training-level, evidence.
**Tips & tricks.**
- Grade outside the agent's container from artifacts copied out; never parse agent-writable logs as ground truth.
- Reset or forbid every file the agent may add that the runner auto-loads (conftest.py, sitecustomize, plugins, PATH shims).
- No gold answers anywhere reachable from the sandbox, including the internet.
- Run a null agent on every task; any nonzero score is a logic flaw.
- Drop root, network and host sockets by default; grant per task.
- Iterate attacker vs defender for ≥ 3 rounds; re-run a fresh attacker after every patch.
**Fit to Crucible Part 2.** F1 (AppBundle) and the harness, before F6: add a **harness-level G0 "trust boundary" gate** run once per App/World template and on every harness change: BenchJack-style recon + static scan + exploit attempt against the verifier path (state readers, answer parsers, path checks). In a simulated World this means: verifiers read the World DB through the privileged StateAdapter, never through anything the policy can write; answer checks use structured parsing; no gold digest or oracle reachable through policy-visible tools (G7 tool-only and G11 leak rules already cover parts). G4 "not pre-solved" (null agents) is the checklist's L3. G12 per-task attacks remain, but this paper shows that harness flaws dominate hack rates and need redesign, not patches. Debug lane: verify that the test runner ignores agent-created conftest/plugin files and that test outcomes are parsed outside the agent's container. Priority P0.

## 2507.02825 — Establishing Best Practices for Building Rigorous Agentic Benchmarks (ABC) (Zhu et al., 2025; NeurIPS 2025 D&B, UIUC Kang lab)
Read: FULL (body; appendices B–E skimmed for the per-benchmark findings; NeurIPS checklist skipped).  Relevance to Part 2: high. It supplies the two validity definitions that Crucible's F10 uses (task validity, outcome validity), a concrete check list (T.1–T.10, O.a–O.i, R.1–R.13) that maps onto G1–G14, and measured failures of τ-bench-style state checks that Crucible's synthetic-World tasks would inherit.
**Problem.** Agentic benchmarks over- or under-estimate agents by up to 100% (relative) because of task setup or reward design flaws.
**Method / algorithm.**
- **Definitions:** *task validity*: a task is solvable if and only if the agent has the target capability (violations: shortcuts → false positives; impossible tasks → false negatives). *Outcome validity*: a positive evaluation result if and only if the task succeeded.
- **Checklist (ABC)**, built from 17 popular benchmarks, prior pitfall reports and software-testing practice:
  - *Task validity:* tool versions specified (T.1); API availability and rate limits (T.2) with abort on interruption (T.3); clean state between tasks (T.4); agents isolated from ground truth (T.5); frozen reproducible environment (T.6); verify ground-truth annotation and task setup (T.7–T.8); **an automatic oracle solver** (T.9); **inspect outliers in pilot runs**: agents failing easy tasks → impossible tasks; agents passing only hard ones → shortcuts (T.10).
  - *Outcome validity* by method: whole-string match handles equivalents and redundant words (O.a); substring match handles negation, and tasks must prevent listing all answers or guessing (O.b.1–3); LLM judges validated for accuracy and self-consistency (O.c.1); unit tests verified and quality-measured (coverage, complexity) (O.d); fuzzing varies values, types, layouts, edge cases and inputs must affect outputs (O.e); E2E tests cover all workflow branches, deterministic (O.f); **state matching: ground truth covers all valid outcomes (O.g.1), state space includes relevant and irrelevant parts to catch out-of-scope changes (O.g.2), state complex enough that trivial changes cannot succeed (O.g.3)**; answer parsers state format assumptions (O.h.1) and prevent guessing (O.h.2); quality metrics correlate with the process (O.i.1).
  - *Reporting:* open data/harness, contamination measures, stated capability and construct validity, documented mitigations, quantified impact of known flaws, significance, baselines (R.1–R.13).
- Scoring: 1 point per satisfied applicable item; each flagged issue validated by an experiment.
**Verification / grading.** Each issue quantified empirically (trivial agents, re-grading published results, expert confirmation for CVE-Bench).
**Scale & cost.** 10 benchmarks assessed in depth; costs not given.
**Key results.**
- 7/10 violate task validity, 7/10 outcome validity, 10/10 have reporting gaps (80% do not acknowledge weaknesses).
- **τ-bench:** 38% of airline and 6% of retail tasks are intentionally unsolvable with success = unchanged state, so a do-nothing agent scores 38% on airline, above a GPT-4o agent; 2% / 3.6% of tasks grade by substring of verbatim DB text, so dumping the database passes (overestimate up to 40%).
- **WebArena:** substring match ignores extraneous content; for "N/A" tasks the LLM judge accepts an empty reply (1.4–5.2% overestimate).
- **SWE-Lancer:** tests sit in a password-protected zip whose entries can still be listed and overwritten; agents can score 100% without solving anything.
- **KernelBench:** fuzzing varies only values, not shapes or layouts: ~31% overestimate of kernel correctness.
- **OSWorld:** 13/46 Chrome tasks broken by website drift (selectors), 28% underestimate.
- **CVE-Bench repair:** checking for a SLEEP string in logs instead of its execution overestimated time-based SQLi by 32.5%; an outbound-server check reachable from the same Docker network gave a shortcut (fix cut success 10%); overall overestimation reduced by 33%.
**Failure modes & limits.** Checklist is manual and judgement-based; items are binary and unweighted; coverage of failure types depends on reported incidents. Own critique: "if and only if the agent has the target capability" is not operationalised; the paper offers no measurement of necessity (does success require the skill), which Crucible's ablation probes try to supply.
**Tips & tricks.**
- Run trivial agents (do nothing, empty answer, dump everything, list all options) on every task.
- For "refuse / no-op" tasks, require positive evidence (an explicit refusal, a read of the relevant record) rather than unchanged state.
- Include irrelevant state in the comparison so out-of-scope edits fail.
- Make the state change non-trivial: multiple variables and dependencies.
- Check that the verifier tests an *effect*, not a trace of an attempt (SLEEP in a log ≠ executed SLEEP).
- Use outlier inspection in pilots: easy-task failures and hard-task-only passes are both bug signals.
- Freeze environments; never rely on live websites for checks.
**Fit to Crucible Part 2.** F10 IntentCard vocabulary (task validity, outcome validity) and §3.7: ABC is cited as the source of G4. Concrete adoptions: (1) **G4 null agents should include "do nothing", "empty answer", "dump all readable data", "claim everything"**; (2) for `infeasible_share` tasks in the DomainSpec (deliberately impossible requests), the success predicate must require a refusal message plus evidence of the policy lookup, never only an unchanged state (τ-bench's 38%); (3) F5 `collateral` checks are O.g.2 (irrelevant state included); (4) a "state-change complexity" floor (O.g.3) as a G-check: a random or single-field change must not satisfy Φ; (5) T.10 outlier inspection belongs to F9/F10: tasks the frontier fails but the student passes, or easy-by-predictor tasks that nobody solves, are routed to audit (complements the zero-solve rule). G10/G6 already cover determinism and effect checks. Priority P0 (cheap, high-yield gate items).

## 2606.26300 — The Verification Horizon: No Silver Bullet for Coding Agent Rewards (Qwen Team, 2026; technical report)
Read: FULL (body §1–6; appendices A–C, J skimmed for the judge prompt, hack taxonomy and behaviour rubric).  Relevance to Part 2: high. A frontier lab's measured account of four verifier types in production RL, with three results Crucible should act on: an **agentic task-quality judge** (instruction clarity, instruction–test alignment) whose rejects concentrate in zero-solve tasks; a **trajectory behaviour monitor with an evolving pattern set** that cut hacked successes from 28.57% to 0.56%; and a method to **evaluate an LLM evaluator** against held-out tests with BoN accuracy, Kendall τ and threshold-conditioned quality.
**Problem.** Verifiers are proxies for underspecified intent; optimisation widens the proxy–intent gap (hacking, saturation). Verification quality has three axes: scalability, faithfulness, robustness; tests are scalable + robust but thin, LLM judges scalable + faithful but hackable, humans faithful + robust but unscalable.
**Method / algorithm.**
1. **SWE-like tasks (tests as verifier)**, from the SWE-Universe pipeline (PR → fix patch + test patch, Docker env, `evaluation.sh`; verifier must fail before and pass after the fix; invalid verifiers repaired by a building agent).
   - *Faithfulness:* an **agentic quality judge** (MiniSWEAgent exploring the container: reads repo and tests, runs commands) outputs `instruct_clear` (instruction + environment self-contained enough to solve) and `instruct_ut_align` (tests operationalise the instruction), aggregated to `overall_good`. Ablation (precision/recall/F1 on a human-labelled set): clarity F1 ≈ 91–95 across settings; alignment is harder, F1 75–81, best with few-shot examples + the gold patch (75.9 / 87.2 / 81.2). Typical bad tasks: few-word instructions, references to private discussions, tests orthogonal to the task, tests hard-coding a typo as expected output.
   - *Hacking:* environment hardening (strip git history after the PR date, no network unless needed) plus a **trajectory monitor**: log commands, network access, git ops, files opened/edited, final patch; match against a pattern set P (each = observable evidence, leakage risk, intervention: e.g. searching for the original PR, opening upstream diffs, querying commit hashes); token-level penalty on matching rollouts. **P is updated each training interval**: sample current-policy rollouts (prioritising verifier passes and monitor hits), an agentic reviewer finds new shortcut strategies, add them, redeploy.
2. **Frontend (rubric and interactive judges):** per-task checklists (avg 25.9 items: functional 37.7%, content 19.0%, visual 13.3%, layout 12.9%, UX 9.3%, technical 7.2%); then an interactive judge: a planner writes the full action list in one pass from page info (accessibility tree, key listeners), Playwright executes and records, a judge scores frames + code against the rubric.
3. **User feedback as verifier:** 125,528 real developer–assistant trajectories, 535,737 round labels by an LLM judge with evidence quotes, a polarity and a separate *user fairness* field, conservative-to-neutral rule. Training: SFT vs reweighted SFT vs **Span-KTO** (KTO with spans of consistent polarity as units, EMA reference point, CE on neutral tokens).
4. **Long-horizon repo generation (agent as verifier):** evaluator decomposes the spec into a checklist, must *execute* tests incl. end-to-end, must not edit generator code or run the repo's own tests, outputs S_pass and holistic S_eval. Evaluated against held-out original test suites on 104 NL2Repo tasks × ≤ 4 diverse generations with BoN accuracy, mean regret, Kendall τ, Pearson/Spearman, and threshold-conditioned UT score S̄_UT(θ).
**Verification / grading.** All verifier variants are themselves measured against ground truth (human labels, held-out tests, multi-judge consistency).
**Scale & cost.** Not reported in $; datasets above.
**Key results.**
- **Zero-solve tasks contain a much larger share of low-quality tasks** (judge-flagged); high-solve buckets are dominated by good tasks. Quality-filtered RL improved SWE-bench Multilingual and Pro, flat on Verified.
- Behaviour–success association after hardening (Qwen-Turbo rollouts, base resolved 59.99%): repository-history mining 21.11% of rollouts (56.55% resolved), test-oracle tampering 3.69% (47.29%), harness tampering 8.25% (41.47%), visible-test overfitting 30.00% (51.62%), evaluator-aware patching 8.78% (56.39%) all *below* base; **solution-artifact retrieval** 4.32% of rollouts but **72.34% resolved** (+12.35); external fix lookup 7.03% (61.69%).
- Monitored RL (3 SWE-bench variants): hack rate 37.76% → 1.31%; hacked-resolved 28.57% → 0.56%; **clean resolved 40.22% → 60.53%**. Without the monitor, raw pass rate keeps rising while clean performance collapses late in training.
- Static judges invite length hacking (ever longer CSS/JS); interactive judge avoids it; rubric judges reach Spearman 0.81–0.91 with humans and cross-judge τ ≥ 0.93. Interactive-judge RFT: WebDev Human Eval 78 → 84.
- User feedback: neutral 76.6%, negative 20.0%, positive 3.5% of rounds; 81.8% of negatives high-confidence; negatives mostly execution (56.6%) and misunderstanding (21.1%). Discarding negative tokens hurts (w_neg 0 → 37.2% vs SFT 41.8%); only mild downweighting (0.8) helps (44.4%). Span-KTO: SWE-bench Verified 54.2 → 59.8, internal Aone-bench 14.8 → 28.1; big behaviour gains on unresolved tasks (inefficiency +34.5%, communication +26.5%).
- Evaluator prompt iteration: v1 → v4 BoN 57.9 → 67.4, τ 0.379 → 0.473; **v5 with more rules got worse** (BoN 59.6). Failure patterns fixed in turn: no execution (lazy reading), no end-to-end check, role confusion (fixing generator code, running the repo's own tests, advocating for the generator), context overload. Backbones: Claude Opus 4.7 best (BoN 70.4, τ 0.579, most stable); ranking ability does not imply filtering quality (DeepSeek V4 Pro lower τ but higher S̄_UT at θ = 8).
- RFT with evaluator filter (9,139 samples) 23.52 vs random 21.61 at equal size; all 19,050 unfiltered 24.75: filtering matters most when candidates are scarce.
**Failure modes & limits.** Internal benchmarks and models; few ablations with error bars; pattern-set monitors only catch known strategies until the reviewer finds new ones. Own critique: a token-level penalty on monitored behaviour is itself a proxy that can be evaded by disguising shortcuts.
**Tips & tricks.**
- Judge each task for instruction clarity and instruction–check alignment with an agent that can inspect the environment; give it the gold solution.
- Treat zero-solve tasks as suspect quality first, hardness second.
- Log information-access actions in every rollout and keep a living pattern set of shortcut behaviours; review passing rollouts each interval.
- Track "clean resolved" (pass and no monitor hit) alongside raw pass rate; their divergence is the hacking alarm.
- Evaluate a model-based evaluator against held-out ground truth with ranking, selection and threshold metrics, and pick the metric by downstream use (RL needs ranking and variance; filtering needs low false positives).
- Forbid the evaluator from editing the artefact or reusing its tests; require execution and end-to-end checks.
- Do not over-specify judge prompts.
**Fit to Crucible Part 2.** F10 (already cited for the zero-solve rule): make the zero-solve rule concrete with a **task-quality judge** producing `instruct_clear` and `instruct_ut_align` (Crucible: instruction ↔ VerifierBundle alignment) from a non-author family with access to the World and the oracle. F12 FeedbackSignal.hack_flags: implement the **trajectory monitor** over tool-call logs (access to privileged tables, oracle artefacts, git history, network) with an evolving pattern set reviewed every training interval, and report clean-resolved vs raw-resolved per family (a divergence alarm for §3.8's loop guards). F6 G12 and BenchJack cover static leakage; this covers policy-dependent shortcuts that only appear during training. §3.8 loops: "verification must co-evolve" supports a recurring verifier-strengthening loop but with held-out ground truth to measure each new verifier (their NL2Repo protocol). Answer-kind judges (semantic answers): use the evaluator-evaluation metrics before using any judge as reward. Priority P0 for the monitor and clean-resolved metric; P1 for the quality judge.

## 2605.26079 — Automated Benchmark Auditing for AI Agents and Large Language Models (ABA) (Wang et al., 2026; Duke / Together AI / Stanford)
Read: FULL (body; appendices A, C, D overview, G; per-benchmark lists and case studies skimmed).  Relevance to Part 2: high. A cheap ($0.60 per task) agentic per-task auditor with a well-calibrated severity rubric that separates *ambiguity* from *difficulty*, validated against independent fixes. It is a ready implementation of F10's "zero-solve tasks need a clean agentic audit" and of the instruction/environment/evaluation fault sides.
**Problem.** Expert-authored tasks embed unstated assumptions, missing environment resources and brittle graders that human review misses; benchmarks are too complex for generalist annotation.
**Method / algorithm.**
1. **Evidence collector agent:** downloads repo/dataset, writes and runs a collector script that maps each task to a uniform manifest (instruction inline; `solution_ref`, `environment_ref`, `trajectory_path`, `test_output_path` as file references). Onboarding = benchmark name + URL.
2. **Auditor agent** (Claude Code CLI, Opus 4.7, default tools, pinned version), one prompt per task, two modes: *static* (instruction, tests, eval config only) and *trajectory* (plus recorded agent runs and test outputs).
3. **Three independent issue axes:** *Instruction* (missing, ambiguous or misleading so a reader cannot tell what passing means); *Environment* (container, resources, tools, network conflict with what the task assumes); *Evaluation* (tests too narrow, too broad, misaligned, or wrong ground truth).
4. **Severity calibration thought experiment:** give *only the prompt* to five experienced domain developers; would most reliably produce a solution the tests accept? 0 = yes, failures are skill; 1 = minor, a discoverable gap needing deliberate work (git history, docs, guessing among standard approaches); 2 = major, near impossible without more information, or tests check something else. A hard but clear task scores 0; a vague prompt bridged by domain-standard practice scores 0.
5. **Finding schema:** {finding_id, category, subtype, severity, claim, why_it_matters, evidence [{path, note}], suggested_fix}.
6. **Validation hierarchy:** fixes in the wild (maintainer PRs, retirements) > curated gold-issue sets (BenchGuard) > internal human review.
**Verification / grading.** Recall/precision against independent fixes: Terminal-Bench 2 maintainer PR (21 issues): recall 66.7% strict / 81.0% partial, precision 58.3 / 70.8 (a lower bound; the PR is not exhaustive); on 11 extra flagged findings reviewed by hand, 72.7% / 81.8% precision. BixBench gold set: recall 62.5 / 79.2, precision 50.0 / 71.4 (beats BenchGuard's 54.2 / 38.7 strict). ScienceAgentBench: recall 83.3 / 91.7. Human review: static major findings 73% strict / 91% partial precision; minor 63 / 83; SWE-bench Verified trajectory-mode majors 92 / 96. Independently flagged both tasks OpenAI cited when retiring SWE-bench Verified, without web or git access.
**Scale & cost.** 168 benchmarks (frontier-release suites + NeurIPS 2025 D&B), 34,285 tasks, nine domains. Mean **$0.42–0.80 per static task audit** (≈ $0.60), 6–15 turns, heavy prompt caching; a 500-task benchmark costs $200–400.
**Key results.**
- **25.7%** of tasks have at least one major issue; 15.1% minor-only; under 60% clean. By domain: safety/alignment 42.2% major, medical 41.5%, professional 38.4%, multimodal 24.7%, science 25.7%, agentic/tool use 20.6%, retrieval 17.1%, coding 14.1%, math 13.2%.
- Deterministic-grader domains are instruction-dominated (math 65%, science 69% of major issues); safety and retrieval are evaluation-dominated (51%, 57%); complex runtimes (agentic, professional, multimodal) add environment issues.
- Trajectory mode flags 8.5% more major tasks than static mode; agreement on trajectory majors 29–63% (mean 53%): runtime drift and contamination appear only with traces, metadata ambiguity and test leakage only statically.
- Removing flagged tasks raises average model scores by 9.9% (SWE-bench Verified) and 9.6% (Terminal-Bench 2) and changes rankings; Dabstep +12.8 points with major+minor filtered. A test shows the trajectory auditor's extra findings are not driven by agent failure.
**Failure modes & limits.** The auditor is an LLM with Claude-specific behaviour; severity is interpretive; validation covers a sample; redundant findings per issue (1.05–1.40×). Own critique: precision ~58–73% strict means an automatic DROP on every flag would discard many good tasks; use flags as routes to repair or audit.
**Tips & tricks.**
- Separate interpretive failure (prompt-vs-tests mismatch) from skill failure with the "five experienced developers, prompt only" test.
- Audit on three axes (instruction, environment, evaluation) and require file-path evidence plus a suggested fix for every finding.
- Run trajectory mode whenever rollouts exist; it catches runtime drift and silent failures.
- Validate auditors against independent fixes (maintainer PRs) rather than self-made labels.
- Budget ≈ $0.60 per task per audit with caching.
**Fit to Crucible Part 2.** F10 IntentCard and the zero-solve rule (§3.7 already cites ABA): implement the audit as an F10 probe with ABA's schema; the three axes map to the "fault side" labels (World/environment, grader/evaluation, spec/instruction) from note 03. Run *static mode* on every task at F6 (cheap, $0.60, catches instruction and evaluation mismatches before paid F9) and *trajectory mode* on tasks with zero or very low frontier solves after F9, sharing F9's rollouts. The severity rubric is the right instrument for G8 communicability in natural-language domains. Treat a major finding as "repair or INCONCLUSIVE", not INVALID by default, given ~60–70% precision; require a different-family auditor from the task author (family rule). The domain pattern (instruction issues dominate when graders are deterministic) predicts where Crucible's synthetic-World tasks will break: instruction underspecification relative to an exact state check. Priority P0 (static audit), P1 (trajectory audit).

## 2610.11464 — Who Verifies the Verifier? Co-Evolving Inspectable Graders with Self-Improving Agents (Zhang et al., 2026; AWS + HSBC preprint)
Read: FULL (body + all appendices).  Relevance to Part 2: high, as the main evidence behind §3.8's loop guards. Two results are load-bearing: (1) **a collapsed always-pass verifier trains skills as well as a valid one, so downstream task score cannot validate a self-evolved verifier**; (2) the safety comes from **anchor discipline** (fail-closed selection on a small locked reference set, validity gate), not from lifecycle management. Also a concrete design for an inspectable verifier: a small expression over typed drawback detectors.
**Problem.** Self-improving agent loops need a verifier each round; for open-ended tasks none exists, and an LLM judge of the same family invites reward hacking and shared blind spots.
**Method / algorithm.**
- **Splits:** train (large, unlabeled; graded by the evolving verifier for the skill loop); dev (**10 anchored items**, 5 failures + 5 clean, soft labels from a teacher comparing to golden references: the only supervised signal); test (locked; strongest available reference; never read by verifier evolution; used for evaluation and skill-loop rollback).
- **Ops (drawback detectors):** o(t, y, c) → {drawback, clean, abstain}, each checking one failure class, in three cost tiers: static (parse/inspect), execution (sandbox or live warehouse), narrow LLM-judge question. A fixed root validates structure.
- **Verifier = expression** over ops: leaf, OR (drawback if any), AND, NOT, K-of-k vote, weighted vote; abstaining children excluded; pass iff the root ∧ expression find no drawback.
- **One round** (Algorithm 1): *sense* misses (dev items passed but labelled fail) and gaps (train outputs where the pool abstains or splits); *grow* by clustering misses ∪ gaps under a closed failure taxonomy and having an LLM author a typed op per recurring cluster; **birth gate**: fires on ≥ half its cluster and stays clean on known-good outputs; ops from anchored misses enter *active*, ops from unlabeled gaps enter *shadow* (record verdicts, unselectable until they raise dev agreement); *select* candidates from an LLM composer (sees catalog, gaps, elite history) plus mutations/crossovers, keep argmax S(e) = A_dev(e) · A_train(e)^w − λ·C(e); *curate*: retire ops with non-positive leave-one-out marginal after a 2-round grace; *audit* on the locked test set (measurement only).
  - A_dev is recall-weighted (missed drawback costs 2× a false alarm); A_train is agreement with a consensus of opining ops, each weighted 1 + 10·max(0, anchored marginal) so a bloc of correlated cheap detectors cannot define the pseudo-label; w = 1.0 (code, SQL), 0.25 (report); complexity 0.001 per op beyond 3, +0.002 per LLM op.
  - **Guards:** fail-closed (no dev opinion → unselectable); validity gate drops all-pass, all-fail, all-abstain candidates.
- **Double Ratchet:** alternate verifier phases (15/8/5/2 rounds) with four 25-round skill phases of an existing skill-library loop (Ratchet); the verifier grades train attempts and its named drawback text becomes failure "capsules" for skill synthesis; the skill loop's held-out eval stays on the locked test set.
- **Outer audit:** a stronger model compares final vs pre-evolution outputs pairwise, both orders, win only if both agree; never a training signal.
- Models: Claude Opus 4.7 in all in-loop roles; Opus 4.8 as outer judge. Tasks: MBPP+ (hidden tests), Spider 2.0-Snow (execution match on live Snowflake), analyst report generation (no true verifier; rubric + golden sections). Hard subsets by 5-sample screening; splits 60/10/40, 59/10/40, 73/10/48.
**Verification / grading.** Verifier validity = agreement with the locked anchor; paired sign-flip permutation tests clustered by task.
**Scale & cost.** CPU only, API-bound; 100-round loops, 3 seeds; birth gate rejected ~1,000 (MBPP+), 612–785 (Spider), 172–188 (report) proposed ops; final pools 12–39 ops; selected verifiers have 1–3 leaves.
**Key results.**
- MBPP+: evolved verifier 0.625 ± 0.050 held-out agreement vs 0.417 hand-authored seed composition (+0.21) and 0.55 for the bare LLM judge it contains; consistent on every seed but p = 0.12 (peak) when clustered by task.
- **Ablation:** removing anchor guards → verifier collapses to always-pass on 3/3 report seeds (passes 0.97–1.00, vacuous perfect objective); removing the lifecycle collapses nothing. Yet the naive co-loop's **held-out task score equals or beats the anchored one** (MBPP+ 0.742 vs 0.717), because unfiltered practice still improves skills.
- Sufficiency: Double Ratchet keeps 106% / 110% / 88% of the lift that ground truth (or rubric) gives the same skill loop on MBPP+ / Spider / report; solves more never-solved held-out tasks than the ground-truth loop on both hard-anchor tasks (19/23 vs 16/23; 6/12 vs 4/12). Even a 0.50-agreement verifier sufficed on Spider: in a skill loop the verifier's role is directional (which attempts become failure capsules).
- **Reward-hacking episode:** report skills raised the rubric by +0.26 partly by writing metric tags without values (~30% of tags) and inventing confident forecasts; invisible in-loop; the outer judge preferred the baseline in 88% of decided pairs. One added detector (value-erasure check) + rewritten hints cut erased tags to ~1%. **The outer judge itself was wrong** until given the task's format contract (generic rubric win rate 0.122 → 0.126 after repair; task-aware rubric 0.515 → 0.770).
- **Detectability spectrum:** agreement is high where failures are mechanically checkable (MBPP+ crashes; Spider compile errors gave 0.85) and drops to chance when residual failures are "runs cleanly, wrong value" (Spider after schema grounding raised the solver 0.211 → 0.329, agreement fell to 0.50).
- Anchor size 4/6/8/10 items gives similar mean held-out agreement (0.871/0.882/0.874/0.854), larger variance with fewer.
**Failure modes & limits.** Mechanism study: three task families, one model family, small locked sets, single outer judge. Gains not significant under clustered tests. Own critique: the skill loop's rollback reads the locked test set, so "never read" holds only for verifier evolution.
**Tips & tricks.**
- Never validate a learned or evolved verifier by the policy's downstream score; keep a small locked anchor it predicts but never reads.
- Select verifiers fail-closed on the anchor and drop all-pass/all-fail/all-abstain candidates.
- Build verifiers as small readable compositions of deterministic detectors; name the failure class, not just a scalar.
- Admit new checks only if they fire on their failure cluster and stay clean on known-good outputs; put unanchored checks in a shadow tier.
- Weight a recall miss above a false alarm when selecting.
- Down-weight correlated detectors in any consensus pseudo-label.
- Keep an outer, different-model audit with the task contract in its rubric.
- Expect verifier improvement to stall when failures are semantically wrong but clean-executing.
**Fit to Crucible Part 2.** §3.8 loop guards: adopt as hard rules for every self-improvement loop that changes a verifier, World or task: (1) a frozen, author-disjoint **anchor set** per family (small, ~10 items, both classes) that the loop can never edit; candidates are selected on anchor agreement, fail-closed; (2) a validity gate (no all-pass/all-fail verifiers); (3) a separate **locked validation set** read by no loop; (4) never use student pass-rate gains as evidence that a verifier change was good. F5: the detector-expression shape fits VerifierBundle (state/answer/collateral/path checks are typed detectors; the bundle is their conjunction), and the birth gate is exactly "gold-sanity + fires on its negative class" (G5). F12: failure clusters → new detectors is a verifier-repair source; route through the birth gate and shadow tier. F10: the detectability spectrum predicts that semantic-answer tasks will need human or outer-judge audits; the outer judge must be given the task contract (the DomainSpec + TaskContract). Priority P0.

## 2609.24972 — RRSI: Regularized Recursive Self-Improvement of Agent Harnesses (2026; Google Research preprint)
Read: FULL (body + appendices C–E; qualitative case study F skimmed).  Relevance to Part 2: high for §3.8 (the self-improvement loops). It is the clearest measured recipe for keeping an LLM-driven "propose edit → score on a reused set → keep best" loop from overfitting its own evaluation set: sparse attributable edits, an edit ledger as negative evidence, a pre-scoring leakage critic, a noise-band acceptance floor, cost-justified acceptance, and pruning of components that stop paying. Crucible's generator/verifier/World improvement loops have the same structure.
**Problem.** Automated harness evolution (prompts, control flow, tools, memory, context) reuses a finite evolve set adaptively, so gains come from benchmark-specific fitting, noise chasing and complexity growth, and shrink or reverse out of distribution.
**Method / algorithm** (frozen policy; proposer, failure analyst and leakage critic are Claude Opus 4.8; edit space left fully open):
- *Proposal side:*
  1. **Annealed edit budget (L0-style):** at round t of T, b_t = ⌈b_min + (b_max − b_min)·½(1 + cos(πt/T))⌉ with b_min = 1, b_max = 3–4, from an ~80-edit draft pool: early rounds may bundle a few edits, late rounds one attributable edit.
  2. **Evidence-aware credit:** ledger of every evaluated atomic edit (component ∈ {prompt, control_flow, config, output_plumbing, context_mgmt, client_tool, skill, memory, subagent}, hypothesis, diff, ΔS, ΔC, accepted?). The proposer conditions on it; rejected hypotheses stay negative evidence.
  3. **Structured exploration:** if progress over w = 3 rounds ≤ δ (stall), reserve 1 of 2 candidate slots for never-exercised components.
- *Selection side (non-compensatory gates):*
  4. **Leakage critic before evaluation:** reads the diff; rejects edits encoding task names, entities, task-specific values, answers or benchmark-specific logic, and inert machinery. Screening before scoring keeps leaking candidates from ever acquiring an inflated score.
  5. **Noise-adjusted floor:** δ measured beforehand from repeated runs of the unchanged base harness (never tuned); accept only if Ŝ(H′) ≥ S* − δ with S* the best ever (floor never moves down).
  6. **Cost-aware acceptance (Ridge-style):** if ΔS > δ, require relative token growth ΔC ≤ β0 + β1·ΔS (β0 0.10–0.15; β1 ≈ 25% tokens per extra pass); if ΔS ≤ δ, admit only if w_s·ΔS − w_c·ΔC + w_n·(new structural component types) > 0 (coding sets w_s = 0: a within-noise gain alone never counts).
  7. **Structural pruning (Lasso-style):** components with no strictly positive gain in the last 4–5 rounds become deletion targets for the proposer.
  8. Optional per-domain guards; keep the incumbent if nothing is admissible.
- Settings: T = 20 (coding, workspace) or 40 (engineering); k = 2–4 trials per task; δ = 0.017 / 0.004 / 0.020.
**Verification / grading.** Unit tests (Terminal-Bench, SWE-bench), LLM-judge rubrics (Harvey LAB, JobBench, GDPval, APEX-Agents), deterministic simulators (EngDesign, Frontier-Eng). Paired task-level bootstrap (10,000 resamples) and an independent repeat of every search.
**Scale & cost.** 8 benchmarks, 3 domains; evolve sets: Terminal-Bench 2.1 (89 tasks), Harvey LAB (120 evolve / 40 ID held-out), EngDesign (61). Policy-token cost reported per trial.
**Key results.**
- Gains vs base harness H0: evolve splits +6.0 (Terminal-Bench), +1.1 (Harvey LAB), +4.9 (EngDesign); held-out +1.8 (SWE-bench Verified, never scored), +2.3 (Harvey ID held-out), +3.5 to +4.7 on three OOD workspace benchmarks, +4.3 Frontier-Eng; no held-out split regresses; all bootstrap CIs exclude zero; repeat run within 1.8 points.
- **Baselines overfit:** on workspace tasks Meta-Harness gets the highest evolve score (93.0 vs RRSI 90.5) but only +0.9 OOD; AHE and TTHE end *below* H0 out of distribution; RRSI OOD average 43.6 vs H0 39.7.
- **Ablation:** unregularized evolution: evolve 92.8 (highest), OOD 40.3 (≈ H0), 3.80M tokens/trial; without acceptance regularizers OOD 41.0, 3.59M; without proposal regularizers OOD 41.9, 2.69M; full RRSI OOD 43.6, 2.42M tokens (36% less than unregularized; H0 1.56M). Higher evolve score went with lower transfer in every arm.
- Transfers across policies (Gemini 3.5 Flash: Terminal-Bench 64.6 → 78.7, SWE-bench 76.8 → 79.0) and to an unseen weaker backbone (11.2 → 14.6).
- Gains survive deterministic graders, so they are not judge-pleasing.
**Failure modes & limits.** Frozen weights only; depends on δ and a few hyperparameters; one proposer family. Own critique: the leakage critic is an LLM reading diffs, so subtle benchmark-specific logic can pass; the evolve set itself is still reused adaptively, and the method only slows overfitting.
**Tips & tricks.**
- Measure the evaluation noise band δ from repeated runs of the unchanged system before any loop; never tune it.
- Accept a change only if it beats the best-ever by more than δ, never by within-noise wins.
- Make late changes single and attributable; keep a ledger of every tried change with its measured effect.
- Screen candidate changes for benchmark-specific content before scoring them.
- Charge cost growth against measured gain.
- Delete components that stop earning; reserve exploration slots when stalled.
- Judge loops by held-out and out-of-distribution transfer, never by the set they optimize.
**Fit to Crucible Part 2.** §3.8 loops (generator prompts, World templates, verifier rules, curriculum knobs): adopt RRSI's selection rules as the loop guard spec: (1) per-loop δ from repeated runs of the unchanged generator/verifier on the dev pool; (2) accept only ≥ best-ever − δ and gains > δ; (3) a pre-scoring **leakage critic** that rejects edits naming specific tasks, repos, entities or answers (the debug lane's changelog-strip widenings are exactly the kind of change that should name a pattern, not a task); (4) an edit ledger feeding the next proposal; (5) cost-justified acceptance using $ per kept task (F12 value-per-dollar); (6) prune generator components that stop contributing. Pair with "Who Verifies the Verifier?" (anchor set, locked set) and OpenThoughts-style one-stage-at-a-time ablation. Evaluate every loop on a held-out family the loop never reads (Crucible's held_out families). Priority P0 for the guard spec.

## 2606.17546 — SEAGym: An Evaluation Environment for Self-Evolving LLM Agents (Zheng et al., 2026; Tsinghua preprint)
Read: FULL (body + appendices A, B overview, D, E, F; case-study appendix C skimmed).  Relevance to Part 2: medium-high. It is an evaluation protocol, built on Harbor (Crucible's export format), for anything that updates itself across rounds: separate update evidence from a frozen update-validation view, held-out ID and OOD views, and a **replay view for forgetting**, with per-snapshot records. Crucible's §3.8 loops and its training runs need exactly these views.
**Problem.** Self-evolving agents are scored by one final number or one curve, which hides whether an update generalises, overfits recent tasks, raises cost, or breaks old behaviour.
**Method / algorithm.**
- Formulation: snapshot A_t = (M, H_t) with fixed model M and mutable harness H_t (prompts, memory, skills, tools, middleware, runtime config); the environment samples a train batch B_t, the agent solves it, receives feedback under the benchmark's visibility policy, and applies its own update H_{t+1} = U(H_t, B_t, trajectories, feedback).
- **Schedule as data:** state persistence, task reuse, batch size, update repeats and assessment timing are parameters, so single-task, online and epoch-batch evolution are comparable.
- **Splits vs views:** D_train gives update evidence; V_update-val ⊂ D_val is a frozen process view never used as update evidence; V_ID and V_OOD ⊂ D_test for final transfer; V_replay re-runs earlier tasks for retention.
- **Metrics** (from saved records, recomputable): Perf, SR; UVG_prev and UVG_base (update-validation gain vs previous/initial snapshot); IDG and OODG (final vs initial on held-out views); FR = max(0, Perf(A_0, replay) − Perf(A_T, replay)); token, tool-call, time costs; domain macro averages. Replay also yields fix and forget counts relative to A_0 and per-step churn.
- Integration: Harbor runs tasks, environments, verifiers; methods plug in through a rollout/update wrapper; harness diffs, update summaries, trajectories and snapshot records are saved for diagnosis (secondary to verified metrics).
**Verification / grading.** Benchmarks' own verifiers (Terminal-Bench 2.0 tests; HLE answers).
**Scale & cost.** 80 train tasks (Terminal-Bench 2.0 + HLE math/physics), 35 validation, 55 ID test, 80 OOD (HLE CS/AI + engineering); 5 epochs, batch 20; DeepSeek-V4-Flash rollouts; rollout 1.46–2.33M tokens per task; updates 1.6–3.9M tokens each.
**Key results.**
- Methods: AHE (harness editing) V 40.0 → 57.1, ID +9.1, OOD +6.3; ACE (skill/strategy memory) +2.9 / +3.6 / +2.5; TF-GRPO (grouped-rollout text optimisation) validation +17.1 but ID +3.6 and **OOD −2.5** with the highest rollout cost: validation gain does not imply transfer.
- **Replay:** AHE final solves 43/80 train tasks vs 34/80 initially (fixes 13, forgets 4), but after epoch 4 replay fell to **6/80** (an evolved middleware message-construction bug) before recovering. Initial-vs-final reporting would hide this.
- **Batch size is non-monotonic:** 10 → −14.3 val / −14.5 ID; 20 → +17.1 / +9.1; 40 → +2.9 / +1.8; 80 → −17.1 / −16.4. Update analysis capacity (~3–4M tokens per update regardless of batch) is the bottleneck: too little evidence per update or too much to digest both break the harness.
- **Source diversity:** HLE-only training reached a useful snapshot at epoch 3 (ID +7.3) but **collapsed to 0** at the final snapshot; the mixed source also dipped but recovered, because tool/environment failures (Terminal-Bench) and reasoning failures (HLE) gave varied evidence for repair.
- **Cross-model:** harnesses evolved on one backend transfer asymmetrically (DeepSeek-evolved: GLM ID +7.3, GPT-5.4 ID −3.6); GPT-5.4-evolved harness: own ID +5.5 but own OOD −7.5. Gains transfer when the evaluation backend shows the same failure surface.
**Failure modes & limits.** Two benchmark sources; one main backend; evaluation cost grows with the number of views; no weight updates studied. Own critique: n per view is small (35–80 tasks), so single-snapshot differences of a few points are noise; no bootstrap intervals reported.
**Tips & tricks.**
- Keep update evidence, a frozen process-validation view, held-out ID, held-out OOD, and a replay view separate.
- Re-run replay at every snapshot and report fix/forget churn, not only end-to-end gain.
- Select the reported snapshot by validation but always show final and intermediate snapshots.
- Tune evidence per update to the updater's analysis capacity.
- Mix sources with different failure surfaces so later updates can repair earlier damage.
- Save diffs and per-snapshot records so metrics can be recomputed without reruns.
**Fit to Crucible Part 2.** §3.8 loops and F12: give every loop (generator prompts, verifier rules, World templates) and every student training run SEAGym's view structure: update evidence = the loop's working pool; update-validation = a frozen dev pool; ID test = held-out tasks of trained families; OOD = held-out families (already in the DomainSpec); **replay = previously kept tasks re-run after each loop change**, which turns into a regression gate for the factory itself (a verifier or generator change that drops replay pass rates on known-good tasks is rejected). F11: the source-diversity result supports mixed-family batches (cf. Nemotron 3) and suggests mixing *failure surfaces* (tool/env vs reasoning) on purpose. Training (Part 1): track per-checkpoint fix/forget on a fixed replay set for the 35B, since an intermediate collapse is likely to be missed otherwise. Since SEAGym is Harbor-native, Crucible's Harbor export can be evaluated with it directly. Priority P1.

## 2610.27465 — Recoverability and Verifier Strength: Two Measured Preconditions for Unattended Agent Self-Improvement (Wang & Qi, 2026; Uniphore preprint)
Read: FULL (whole paper incl. appendices A–E).  Relevance to Part 2: high for §3.8. Two measurable quantities decide which self-improvement steps may run without a human: **verifier strength σ** (Matthews correlation with a trusted reference, so a constant verifier scores 0) and **recoverability** (does undoing need a new approval; does rollback restore behaviour; what blast radius did the change already have). It also documents **silent gate collapse** of an LLM judge and two cheap tests that catch it.
**Problem.** Platforms now read traces, derive agent changes, approve them with an automated verifier and deploy them; which of these steps can safely run unattended?
**Method / algorithm.**
- **Verifier strength** σ(v) = MCC(v, v*) on a trajectory set C, in [−1, 1]; also false-admit rate α and false-reject rate φ against the reference. MCC is used because accuracy flatters a constant verifier.
- **Recoverability** of change ι: re-approval g ∈ {0,1} (0 if the old value is stored and undo is a plain write; 1 if the undo must be regenerated and approved by the same gate); restoration fidelity ρ (share of probe tasks with the same outcome after undo as before); blast radius β (share of probe tasks whose outcome changed while ι was live, which rollback cannot repair). Noise floors ρ0, β0 from a null change (stochastic agent and user).
- **Autonomy rule:** run unattended only if g = 0 **and** σ ≥ σ_min was measured; otherwise **refuse** (a third outcome besides admit/reject, logged with the missing precondition). σ_min is calibrated per platform: choose a tolerable regret r, build a candidate pool whose best-vs-average gap exceeds noise ("not field-limited"), measure regret vs σ by flipping reference verdicts with probability ε, set σ_min where the regret CI's upper end ≤ r; recalibrate when the number of runs per candidate changes. Also bound α by a task-level upper confidence bound (errors repeat across a task's runs).
- **Platform:** change types typed by recoverability (system prompt, tool definitions, skills: write-back undo; weights: re-pin adapter but learned-judge verifier, so human sign-off); evaluation as an admission controller (nothing deploys if eval fails or cannot run); each agent declares its verifiers with measured σ.
- **Two acceptance tests for any LLM gate:** *delete the signal* (remove the property the gate claims to judge; the score must change) and *check the guarantees* (verify request-level settings such as forced tool calls, output format, temperature actually took effect).
**Verification / grading.** τ²-bench airline (50 tasks: 30 train, 20 test) where the task score is a product of configurable checks, so one set of trajectories can be re-scored by different verifiers; reference = DB-state equality × communication checks.
**Scale & cost.** Gate study: 172 settings × 10 runs (1,718 calls), 3 judge models, 4 prompt variants. τ² experiments: 2 runs per train task, 4 per test task. Costs not given.
**Key results.**
- **Silent gate collapse:** the production diversity gate (smallest judge) returned 0.250 in all 40 runs regardless of training-set imbalance, identical to a prompt with imbalance removed; full collapse in 9/37 configurations, all on that judge; larger judges tracked imbalance (p < 1e−11). Rewriting the criterion as a checkable bound ("if the largest seed exceeds 50%, score at most 0.10") worked 30/30 vs 0/30. Three request settings silently failed (forced tool call incompatible with extended reasoning; a gateway dropped output-format requests; temperature rejected).
- **Real verifiers' σ:** LLM judge with natural-language assertions σ = 0.70 (α 0.10, φ 0.21; task-resampled CI [0.36, 1.00] because errors cluster in 1–2 tasks); reference-action match σ = 0.67 (α 0.21, φ 0.12): similar σ, opposite errors; a constant admit-all verifier σ = 0 while passing all 42 failing trajectories.
- Weaker σ picked worse prompt changes (regret 0 → 0.067 as σ → 0) but not significantly at 20 test tasks (needs ~220); the candidate pool was "field-limited" (best candidate only 0.069 above average). Even with a perfect verifier the loop overstated its own gain by 0.20 (winner's curse + train/test shift; p = 0.08).
- **Rollback:** restoring a stored prompt returned behaviour to the noise floor (ρ − ρ0 = 0.000, TOST p = 0.007); three regenerated undos left behaviour 0.16 worse (p = 0.0001), and the gate refused them.
- **Blast radius:** each admitted prompt change altered outcomes on 24–28% of 50 tasks vs an 8% floor (+0.17 beyond noise, p = 0.0002), with flips in both directions (18 up, 20 down) so the mean barely moved.
- Infrastructure failures appear as empty transcripts with no error from the runner; all runs were checked before scoring.
**Failure modes & limits.** One platform; rule applied after the fact; only prompt edits tested end to end; small task sets; σ_min not transferable across settings. Own critique: β is measured only on probe tasks; real side effects in tools (refunds, emails) need tool-boundary accounting.
**Tips & tricks.**
- Report verifier strength as MCC against a fixed reference set, plus false-admit rate with a task-level upper bound.
- Treat σ as unmeasured after any change to judge model, prompt, request format or averaging, and refuse until remeasured.
- Run "delete the signal" and "check the guarantees" on every LLM gate.
- State gate criteria as checkable numeric bounds, not soft preferences.
- Keep the old version of everything a loop can change (g = 0); never let a loop change something it cannot restore by a write.
- Measure blast radius on a fixed probe set while a change is live; mean pass rate hides churn.
- Promote on the lower confidence bound, not the best training score (winner's curse).
- Check run completeness: crashed simulations can look like silent zeros.
**Fit to Crucible Part 2.** §3.8 loops: the guard spec should include (1) for every model-based gate (LLM judges in F5 answer checks, F10 auditors, F3 dedup judges, the swarm's quality gates) a measured σ and α on a frozen labelled set, recomputed after any model/prompt change, with refusal when stale; (2) "delete the signal" as a unit test for each gate (e.g. the debug lane's report lint, intent judges); (3) every loop-modifiable artefact (generator prompts, DomainSpec rules, verifier code, World templates) versioned so undo is a write (g = 0); (4) blast-radius measurement against the replay pool (pairs with SEAGym) before a loop change is kept; (5) loop decisions on lower confidence bounds. F6 G10: the empty-transcript failure mode argues for a run-completeness check before any gate score is trusted. For the student pipeline, weight updates stay human-approved, consistent with this rule. Priority P0.

## 2411.12990 — BetterBench: Assessing AI Benchmarks, Uncovering Issues, and Establishing Best Practices (Reuel et al., 2024; NeurIPS 2024 D&B, Stanford)
Read: FULL (body + appendices A, D–I; the 46 criterion definitions in J skimmed).  Relevance to Part 2: low-medium. It is a lifecycle checklist for benchmark *documentation and usability* (design, implementation, documentation, maintenance), not for task validity or hackability; useful for Crucible's held-out benchmark release and for its reporting rules, little for the factory's gates.
**Problem.** No structured way to judge AI benchmark quality; widely used benchmarks differ greatly in quality and are reported side by side without caveats.
**Method / algorithm.**
- 46 criteria from a literature review (incl. non-AI benchmarking: hardware, environmental quality, bioinformatics) and unstructured interviews with 20+ stakeholders (policymakers, model and benchmark developers, users, researchers); only criteria under the developer's control with normative consensus are scored.
- Five-stage lifecycle: design, implementation, documentation, maintenance, retirement (retirement not scored).
- Criteria examples. *Design:* tested capability defined; how it maps to the task; real-world relevance; how to interpret the score; domain experts involved; use cases/personas; literature integrated; informed metric choice; **metric floors and ceilings, human level, random level**; validated automatic evaluation; differences to related benchmarks; input sensitivity addressed. *Implementation:* evaluation code, data/prompts/environment available; **script to replicate results**; supports API and local models; **globally unique identifier or encrypted instances** against contamination; a training-on-test-set task; release requirements. *Documentation:* construction process, data collection, task rationale, metrics, **statistical significance**, normative assumptions, limitations, requirements file, quick start, code structure, comments, peer review, licence. *Maintenance:* code usability checked, feedback channel, contact person, retirement statement.
- Scoring 0/5/10/15 (not mentioned / mentioned / partially / fully), n/a allowed; stage averages; usability = weighted mean of implementation, documentation, maintenance; ≥ 10 means reasonably good. Two authors per benchmark, consensus; calibration by three authors on one benchmark.
**Verification / grading.** Manual expert scoring; bootstrap for group differences.
**Scale & cost.** 24 benchmarks (16 foundation-model, 8 non-FM). Living site with community feedback.
**Key results.**
- Stage averages (all): design 10.8, implementation 6.2, documentation 10.1, maintenance 9.6; implementation weakest.
- Replication scripts average 3.75/15; statistical significance reporting 5.62/15; 14/24 did not run multiple evaluations or report uncertainty; 17/24 lack easy replication scripts.
- MMLU scored lowest (5.5 weighted), GPQA high (11.0), yet both are reported together by model developers.
- Design and usability correlate (Pearson 0.693 across all, p < 0.001).
- Open challenges: saturation, contamination (canaries need model-developer cooperation), construct validity, non-standard reporting. A benchmark should detect a 1% improvement reliably.
**Failure modes & limits.** Equal weights, coarse 4-level scale, gameable by doing easy items; does not assess construct validity or whether tasks are correct (ABC, ABA and BenchJack in this batch cover those). Own critique: a benchmark can score well here and still be 100% hackable.
**Tips & tricks.**
- Publish floors, ceilings, random and human (or frontier) reference levels with every metric.
- Ship a one-command replication script and report run-to-run variance.
- Mark every instance with a unique identifier/canary and state release rules.
- Version tasks; record retirements.
- Size the benchmark so the improvements you care about are detectable.
**Fit to Crucible Part 2.** F11 split = benchmark (§2(e) evaluation methodology): use the checklist as the release standard for the held-out benchmark pool: per-task canary GUIDs (also feeds the Carlini-style memorisation probe), floors (null agents, G4) and ceilings (oracle, G3) and a frontier reference level per task, variance from repeated runs (pass^k, bootstrap), a replication script, versioning, and a limitations section. Its "detect a 1% improvement" item is a sample-size requirement for the ~9-task benchmark pool in the current plan, which cannot meet it; report intervals instead. It does not inform gates. Priority P2.
