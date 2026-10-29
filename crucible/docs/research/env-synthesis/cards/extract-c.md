# Extract C: groups 05 and 06, consolidated by mechanism

**Sources.** `group-05.md` (29 papers: difficulty filtering and curricula, self-play and failure-seeded synthesis, test-based verifier weakness, reward hacking) and `group-06.md` (29 papers: World and benchmark design, interaction-first web generation, privacy, verifier auditing, self-improvement guards). 58 papers in total. Read status from the cards: group 05 all FULL (two from PDF: 2502.12272, 2502.18449; some tables lost in HTML, headline numbers from text); group 06 has 28 FULL and 1 PARTIAL (2506.04178 OpenThoughts, per-source tables skimmed); 2509.08724 read from PDF. No ABSTRACT-ONLY cards. Where a card flags a number as lost or uncertain, the flag is kept below.

**Vocabulary.** Stage codes F0–F12, gates G1–G14 and loop names are those of `framework-plan.md` §3. "New gate" means a check the plan does not yet list. Priorities are the cards' own (P0 now, P1 next, P2 later), merged where several cards agree.

---

## B1 Domain spec and World/environment synthesis

### B1.1 The τ-bench World shape: DB + Python tools + policy document + simulated user
- **How it works** [2406.12045].
  1. The task is a POMDP. State = hidden DB state ⊗ user state. Actions = DB API calls ∪ messages to the user. DB transitions are deterministic Python over JSON tables; user transitions are LM samples (gpt-4-0613, T = 1.0). The user sees only the dialogue, never tool calls, and ends with STOP.
  2. Stage I (manual): co-design the simplest schemas, APIs and a Markdown policy, simplified from real systems. Some rules are enforced inside the API (hard error strings such as "payment not found"); others are left for the agent to follow (e.g. how many bags are paid).
  3. Stage II (LM-assisted): write one example row; gpt-4 writes a Python *sampler* (numeric and categorical fields from `random`, text fields from LM-made lists); execute, repair from execution for a few rounds, hand-polish. The LM writes the sampler, not the rows.
  4. Stage III (manual with agent runs): write a user instruction (identity, intent, preferences, personality) that admits exactly one end state under the policy; run a gpt-4-turbo agent, read the trajectory, tighten, repeat. Retail tasks got > 40 trials each; zero- and low-success tasks were re-examined.
- **Numbers.** Retail: 500 users, 50 products, 1,000 orders, 7 write + 8 read APIs, 115 tasks. Airline: 500 users, 300 flights, 2,000 reservations, 6 write + 7 read, 50 tasks. Cost: gpt-4o agent $0.38 + gpt-4 user $0.23 per retail task; 95.9% of agent cost is input (long policy + schemas). 30-action cap.
- **Policy-removal ablation.** Retail 61.2 → 56.8 without the policy (rules are commonsense, so barely used); airline 33.2 → 10.8 (ad-hoc rules are load-bearing).
- **Failure modes.** User-simulator errors (typos, ambiguity, weak arithmetic, rubber-stamping agent suggestions); 4 of 40 sampled gpt-4o failures were instruction typos or ambiguities; curation tuned against one agent family (admitted bias); uniqueness of outcome enforced by hand, which does not scale to 500–1000 tasks.
- **Adoption.** F1 `AppBundle`: add `PolicySpec.enforced_in: api | agent` so rule-following is actually tested. F2: seeded sampler code, determinism by seed. F3/F4: `TaskProposal` carries `user_intent` + `user_known_info`; the oracle is the gold write list. P0 for the contract shape; the policy-removal probe goes to B4 (P1).

### B1.2 Record-and-replay World building with rollback (Repo2Run)
- **How it works** [2502.13681]. The goal is a base image B and command list P such that all repo tests *run*.
  - Build phase, dual environment. Inside the container: monitoring commands, a managed dependency *waiting list* (resolves one package at a time and surfaces conflicts), test running, code editing (original test files are protected), free bash. Outside on the host: `docker commit` a snapshot before every non-safe command and restore it on non-zero exit; a whitelist of safe read-only commands (ls, cat, grep, find, diff, no output redirection) skips snapshotting; a "change base image" action discards all prior commands; long outputs are truncated keeping head and tail. ReAct loop, gpt-4o, T = 0.2.
  - Record phase: a rule-based synthesiser turns the successful non-safe commands into a Dockerfile (RUN with `cd`, `export` → ENV, edits → COPY patch + RUN, base change → new FROM) and pins every installed package to the version actually installed.
- **Ablation.** DGSR / EBSR (Dockerfile builds / tests run): full 100% / 86.0% (361/420); bash-only 92.4% / 41.7%; no rollback 96.9% / 83.6%; **no synthesiser (LLM writes the Dockerfile from history) 19.5% / 13.8%**. Baselines: LLM-reads-README 47.6 / 22.1; SWE-agent 26.9 / 9.0. The deterministic recorder is the biggest component; rollback mainly guarantees replayability; pinning makes later rebuilds deterministic.
- **Numbers.** 420 Python repos from 2024; successful builds average 29.03 min (8 human engineers: 21.33 min); ~18 installs and 3.5 test runs per build. Failures (59): runtest timeout 40.7%, hardware 35.6%, dependency timeout 13.5%, repo defects 8.5%.
- **Failure modes.** EBSR overstates correct environments: only 62% of sampled repos reach ≥ 50% passing and 34% reach 100%. An agent could make tests "run" by stubbing modules.
- **Adoption.** Generalise "the LLM explores; a deterministic recorder emits the artifact" to F1/F2: the swarm agent acts through tools against a snapshotting store (the agentenv SDK has snapshot/restore), failed actions roll back, and the `AppBundle` / `SeedBundle` is compiled from the successful action log, never hand-written by the LLM. G2 follows from replay + pinning. Do not use "it runs" as a gate; require the pre-fix test-status profile to match (G3/G5). P1 for debug-lane builds, P2 for the general recorder.

### B1.3 Per-snapshot environments, time-machine pinning, split setup and verify agents (SWE-bench-Live)
- **How it works** [2505.23419]. Build one image per base commit, not per release tag. Identify setup files (CI configs, README, install guides); the LLM picks a base image; a setup agent (persistent bash, may web-search) installs in editable mode and avoids tox. A **separate verify agent** writes a test command that emits per-test status (e.g. `pytest -rA`), runs it without changing state, and reports problems back. **Time-machine pinning:** proxy the pip index so only versions released before the base-commit timestamp are visible. Validation repeats runs and keeps only consistent instances.
- **Numbers.** 1,319 instances from 93 repos (mean 85k LoC). RepoLaunch yield and $ are not reported. Median F2P = 1 test (mean 5.4), which makes weak-test false positives likely.
- **Adoption.** Debug lane F1/F2: add the index proxy for Python and date pinning for GOPROXY and npm. Never let the verify role mutate state. P1.

### B1.4 Environment profile as the minimal domain description (AgentEvolver)
- A hand-written profile per environment lists entities, attributes and operations at a chosen granularity and is put in the explorer's initial state to focus exploration [2511.10395]. Difficulty is expressed as counts of entities, attributes and operations (e.g. 2 entities, 3 attributes, 3 operations, "hard").
- **Adoption.** F0 `DomainSpec`: the profile is a subset of the `AppBundle` tool graph + schema; the (#entities, #attributes, #operations) triple is a ready knob vector. Synthetic Data RL's three-part task definition (task description + input format + output format) is the minimal `DomainSpec` seed for non-World domains [2505.17063].

### B1.5 Simulated people as opponents: rotate profiles, watch exploitation
- SPIRAL: a fixed opponent is exploited (win rate vs Gemini 0% → 62.5%, then gains plateau); self-play stays at 50–52% vs its own t−16 checkpoint and keeps improving; random opponents collapse [2506.24119].
- **Adoption.** A fixed simulator profile is a fixed opponent. F2 People: rotate simulator profiles and track student win rate against each profile as an exploitation alarm in F12. Per-(family, role) EMA baselines (SPIRAL's RAE) remove structural asymmetries when comparing families. P2.

### B1.6 Frozen, reproducible environments and run completeness
- ABC task-validity items: tool versions specified (T.1), API availability and rate limits (T.2) with abort on interruption (T.3), clean state between tasks (T.4), agents isolated from ground truth (T.5), frozen reproducible environment (T.6) [2507.02825]. Website drift broke 13/46 OSWorld Chrome tasks (28% underestimate). Never rely on live sites for checks.
- Live-site generators go stale: AgentTrek's booking tutorial failed once its date passed [2412.09605]; WebDancer attributes much variance to the live web changing [2505.22648].
- **Run completeness.** Infrastructure failures appeared as empty transcripts with no runner error; all runs had to be checked before scoring [2610.27465]. Adoption: a run-completeness check before any gate score is trusted (G10 prerequisite). P0, cheap.
- **Fast sandboxes.** Kimi's code sandbox on crun with pre-created cgroups and a tmpfs overlay: 0.04 s start vs 0.12 s Docker; 120 vs 27 containers/s on 16 cores [2501.12599]. Relevant when the verifier executes per rollout.

### B1.7 Enterprise content without copying records
- **DP synthetic text (Aug-PE)** [2403.01749]. Private samples only *vote* on synthetic candidates:
  ```
  E_pri = embed(private)
  S_0 = RANDOM_API(N_syn·L)      # class label + random pseudo-subcategory keyword
  for t in 0..T-1:
     H_t = DP_NN_HISTOGRAM(embed(S_t), E_pri, σ)   # each private sample votes for its nearest synthetic one; Gaussian noise per bin
     S'_t = top-N_syn by H_t                       # rank-based, no duplicates
     S_{t+1} = [VARIATION_API(S'_t) × (L−1), S'_t] # keep parents
  ```
  Variation by paraphrase or fill-in-the-blank masking (more masking, more diversity), random tone phrases, noisy length targets. Results: with GPT-3.5 by API, OpenReview rating accuracy drops only 45.4 → 43.1 from ε = ∞ to 1 (DP fine-tuning on real data: 65.1 → 30.5); MIA AUC ≈ 50–52 (chance); 65.7× / 12.7× faster than DP fine-tuning. Failure: DP smooths away rare one-off cases, so rare enterprise incidents cannot come from this route. Adoption: F2 `SeedBundle` text fields and People personas inside the customer boundary, with a fidelity gate (FID / MAUVE on embeddings against the private set). P2.
- **Memorisation risk** [2012.07805]. Larger models memorise more (one setting: 18× more for 1.5B than 124M); repetition inside one document drives memorisation; strings occurring in exactly one document were extracted. Adoption for traces-to-tasks: redact high-entropy identifiers and PII before any trace becomes a `SeedBundle`; cap how many tasks one customer record can seed; plant canary strings in seeds and probe the student after training (more so for the 397B). P1.
- **Clio privacy layers** [2412.13678]: PII-free summaries → cluster size thresholds (minimum unique accounts) → PII-free cluster summaries → a model auditor that removes clusters below 3 on a 1–5 identifiability scale. Measured: ~10% of raw conversations contain private info, ~1.5% after summarisation, none ≤ 3 after cluster summaries; auditor 98% accurate on a set with subtly inserted private info. Adoption: the gate before any customer trace becomes a `TaskProposal`; require an intent cluster to meet a minimum-accounts threshold before a family is built from it. P1.

### Minor B1 tips
- Encode deliberately "ad-hoc" rules in the policy so that the policy is load-bearing (τ-bench airline vs retail).
- Give the builder an explicit "reset base" action (Repo2Run); protect test files from the builder agent.
- Require per-test machine-readable status from every test command so a parser can diff transitions (SWE-bench-Live; see B3.5 on parsers).
- Read-only, login-free and payment-free constraints bound what live-web generators can produce (Explorer, InSTA, NNetNav): their tasks lack state-changing skills, which a simulated World can supply.
- Mask truncated or budget-exhausted episodes rather than scoring them as failures; carry a `budget_exhausted` grade flag (DAPO) [2503.14476].

---

## B2 Task proposal and sources

### B2.1 Act first, label after: interaction-first generation with execution as the filter
Five papers converge on one pattern. The trace is produced first; the instruction is written from it; the label is feasible by construction.

- **NNetNav prefix pruning** [2410.02907].
  1. Persona-seeded exploration policy (persona + action list + accessibility tree + history).
  2. A state-change summariser writes Δ(o_t, a_t, o_{t+1}) per step.
  3. At t ∈ {4, 8, …, 40}: a labeller writes an instruction ĝ from the change log; an outcome model s(ĝ, δ_τ) ∈ {0,1} checks it. If 0, kill the episode; if 1, store (ĝ, τ_<t) and continue. One episode yields nested demos (sub-task, then larger task).
  4. Post-hoc relabel: the agent policy regenerates each action given ĝ. Optional best-of-3.
  - **Controlled ablation** (same gpt-4o-mini generator, same filter, same demo count): MiniWoB++ 0.28 zero-shot / 0.28 instruction-first / **0.48 NNetNav** / 0.36 NNetNav instructions with re-distilled trajectories; WebArena 1.0 / 4.2 / **7.2** / 6.0. The hindsight trajectories themselves matter, not only the instructions.
  - Pruning stopped > 60% of WebArena episodes by step 16. Llama-3.1-8B SFT: WebArena 1.0 → 16.3, WebVoyager 4.4 → 35.2. Self-training on its own data: 1.0 → 5.3.
  - Failure: ORM and labeller are the same model (self-consistent mislabels); 4-step intervals bias lengths to multiples of 4.
- **AgentEvolver solution-after-exploration** [2511.10395]. Curiosity exploration (T = 1; 3 breadth steps, then 17 (AppWorld) / 27 (BFCL) depth steps with a myopic window) → distil the trajectory → add difficulty preferences → the LLM writes a query → the key steps become the reference solution → **replay the reference in the environment and drop tasks whose reference fails** → lexical dedup (> 0.8 token overlap) during generation plus embedding steering, and dedup again after.
- **Explorer propose–refine–summarise** [2502.11357]. Propose an abstract task and execute the first action; after each step, rewrite the task to match what was done (so it respects real availability); summarise the final task as goal-only (what, not how); a verifier judges with a markdown dump of the full final page. 175k trajectories, 94k successful (53.1%).
- **BAGEL round trips** [2403.08140]. Label a trajectory → re-execute from the label → relabel, up to 5 iterations, until a filter accepts. Iterating adds 4–8 points (MiniWoB++). Trajectory-first beats instruction-first on MiniWoB++ (61.0 vs 56.0), where the LM's instruction prior is misaligned with the interface. **Removing demos that fail their instruction on execution adds 7 points**: the LLM filter lets incorrect demos through.
- **Go-Browse frontier over states** [2506.03533]:
  ```
  while F not empty:
    v = select_and_remove(F)
    G = NavExplorer.propose(s_v) ∪ PageExplorer.propose(s_v)   # agents act before proposing
    for g in G:
      feasible, τ, v_new = FeasibilityChecker(g, s_v)          # strong model, ≤ 3 tries
      if feasible: keep (g, τ); if v_new is new: add to F
    for feasible g: sample cheap solvers from s_v (prefixed) ×2 and from root ×2
  ```
  Caps: 20 URLs per domain, ≤ 30 tasks per URL, horizon ≤ 10. Feasibility removed 403 tasks and saved 3.2k rollouts (~13% of steps). Qwen-2.5-7B WebArena 8.3 → 21.7 (NNetNav-7B 18.8; p = 0.094, moderate). On truly out-of-domain live sites NNetNav does better (mostly easy tasks).
- **Failure modes across the group.** All of these rely on LLM judges: Explorer's judge agrees 81% with humans, and ~26% of its judged successes are failures (FP 0.14 of 0.53 judged positive); its summariser hallucinates constraints absent from the trace and grounding mismatches corrupt the summary; the summarised "intent" may rationalise random clicks.
- **Adoption (P0).** The synthetic-World generator's exploration mode: persona-seeded exploration through policy-visible tools; every 4 calls compute `StateAdapter.diff` (exact, not LLM-summarised); a labeller proposes an instruction; prune unless an *execution-based* predicate on the diff holds. Each surviving prefix is a task whose oracle is the prefix action program (F4 by construction) and whose verifier is the diff predicate (F5). Nested prefixes give a horizon ladder (B6). G8/G9 then check that the instruction alone suffices (B4.4). F7: summariser hallucination is "claimed cell ≠ measured cell", rejected by the signature. Go-Browse adds the frontier: nodes are (App, entity view, state fingerprint), edges are tool-call sequences; propose per node. P0 for frontier proposer and start-offset knob.

### B2.2 Answer-first, certified self-play for read-only tasks (Search Self-play)
- **How it works** [2510.18821].
  1. Sample a ground-truth answer a* from a set (50,000 answers from NQ/HotpotQA/ARPO).
  2. The proposer runs a multi-turn search trajectory from a*, then writes q.
  3. Rule filter: tags present, non-empty, not too short, ≥ 1 search call, answer string absent.
  4. RAG certification: a solver answers q from the proposer's own retrieved documents **plus 4 random documents from other trajectories**, with no search; keep q only if a judge finds the answer equal to a*.
  5. Proposer reward = 1 − mean solver success (REINFORCE); solver uses GRPO. Replay buffer of valid questions, **cleared every 10 steps**.
- **Ablations.** Noise documents: 4 best; 0, 1 or 7 worse. Batch filling: dummy padding < dynamic resampling < full-reuse replay (44.4) < replay reset every 10 steps (**48.7**; baseline 42.4). Fixed-proposer solver training saturates near 0.9 in-game reward, then overfits. **A −0.1 reward for invalid questions caused a death spiral** (entropy up, valid rate → 0, solver reward spuriously rising on a stale buffer). Proposer validity rose from ~0% to ~50%. REINFORCE proposer + GRPO solver 83.4 s per step vs GRPO both 504.4 s (~6×) for +1.4 points.
- **Hacks found.** Non-unique answers ("Temptations singer" → Otis Williams passes RAG because the evidence holds one singer, but an open solver can validly name another member); "easy for RAG, hard for search" questions over a closed set. Noise documents counter the second.
- **Results.** Qwen2.5-7B-Base +26.4 average over 7 QA benchmarks (+40.4 TriviaQA); Instruct +8.0; gains at 32B.
- **Adoption (P1).** F4 for read-only World tasks with `answer_checks`: answer-first, the proposer explores the World, writes the question from its trace; certify by re-deriving the answer from the trace's observations plus distractor rows. Add a **uniqueness check**: query the full World for every entity matching the instruction's constraints (not only the oracle's evidence). Proposer reward, if trained: zero, not negative, for invalid proposals; validity first, then 1 − p̂.

### B2.3 Failure-targeted generation
- **SwS weakness → concepts → targeted problems** [2506.08989].
  1. Preliminary RL 200 steps; record per-problem accuracy per epoch. **Weakness** iff max_t a_{i,t} < 0.5 **and** a negative accuracy slope. (Base/SFT failures are often format issues RL fixes quickly, so they are not used.)
  2. An LLM lists foundational concepts per failed problem; group by category.
  3. Recombine: seed concept ∝ frequency; add concepts by softmax over (co-occurrence count + cosine similarity to the set's mean embedding) at temperature τ. Budget per category ∝ its failure rate F_D.
  4. Generate one standalone problem per concept set; reject proofs, multiple choice, true/false, multi-part.
  5. Quality filter by two other models (78.35% removed at the strictest setting).
  6. Reference answer by a stronger reasoner's majority (≥ 50% share).
  7. Difficulty filter: initial policy, 8 samples, keep 25–75% and drop 1/8 and 7/8 (~35% survive).
  - Funnel (32B): 17,545 problems → 1,905 weaknesses → 1M candidates → 40K used. +10.0 (7B) and +7.7 (32B) average over 8 math benchmarks; 7B +16.7 AIME24 from MATH-12K.
  - Weakness-driven *retrieval* from Big-Math beats random retrieval; randomly chosen problems are fitted fast and stop giving signal.
  - Difficulty windows: 5–7/8 helps fastest then saturates; 1–4/8 gives the longest-lasting gains.
  - **Self-evolving variant** (14B runs every stage): +1.2 only; ~50% of its own problems were all-correct and it over-rated their quality.
  - Adoption (P0, the missing consumer of `factory/failures.py` `BuilderIntent`s): weakness = "never above 0.5 and declining p̂ across checkpoints" in F12 `failure_clusters`; concepts = skill leaves and World facets from the failed task's `BehaviourSignature` and failure-skill labels, not its text; recombination = sample neighbouring cells with co-occurrence statistics over existing signatures; budget ∝ per-cell failure rate; then the normal F4–F10 gates. Retrieval from the archive of gated-but-unused tasks is the zero-generation-cost first step.
- **WebRL failed-seed evolution** [2411.02337]. Seed GPT-4o "in-breadth" evolution from failed instructions; keep 0.05 ≤ V(s₀, I) ≤ 0.75 under a trained critic before any rollout; a GPT-4o feasibility filter with per-site rules (written after a manual review found infeasible tasks); 500 tasks per phase × 8 phases. Llama-3.1-8B WebArena-Lite 4.8 → 20.6 (SFT) → **42.4**. The reward is an 8B outcome judge, ~80% accurate (80.8% on test labels, 79.4% on 100 hand labels): about one reward in five is wrong; the judge reads final-state HTML only, so it cannot see side effects or the path. Case study: many "new" tasks were clarifications of failed ones, so failures were partly spec defects. Adoption: P0 for the pattern (failure-seeded proposals), with the judge replaced by F4/F5 construction; record whether a variant changed the spec (clarification) or the skill (harder) and route it through G8.
- **Socratic-Zero zones** [2509.24726]. Mastered (s = 1) → harder variant from a successful attempt; learning (0 < s < 1) → refine the *specific failed attempt* into a new problem targeting that error; too difficult (s = 0) → park, re-categorise later. If all 8 attempts fail, the teacher re-solves its own problem and checks ambiguity and uniqueness. A cheaper generator (Qwen3-32B) is distilled from the teacher with weight U(q′) = exp(−(s − 0.5)²/(2·0.2²)): validity 95.6% vs 89.1% base; downstream utility within 1.1 points of Qwen3-235B, Claude-4.1-Opus, Gemini-2.5-Pro, GPT-5. Utility-function ablation effects are small (0.2–0.4 points). Adoption (P1): zones map onto F9 bands (above → knob mutation; in → targeted variant from the failure trace; below → ladder or park); the distillation is a generator-loop cost lever, but acceptance must still go through F6–F10. Avoid teacher-as-verifier and teacher-as-validity-judge in one family.
- **Synthetic Data RL definition-to-tasks** [2505.17063]. Retrieve passages for a domain keyword; **summarise an abstract "sample pattern" from the demos first** (copying demos produced near-duplicates); generate 500; label by 16-sample instructor majority; rewrite solved items harder and unsolved items easier or into sub-problems; select by 64 base samples, keep the lowest *positive* pass rates. Ablations at M = 500: no pattern step MATH 72.0 → 65.0; no difficulty rewrite GSM8K 89.1; easiest 90.5 / hardest 90.7 / all 89.9 vs lowest-positive 91.7. Llama-3.2-3B base did not improve with GRPO on either synthetic or human data (lacks verification and backtracking behaviours). Failure: low-pass selection also selects mislabelled items (a wrong label looks "hard"); "rewrite harder" can reword rather than change structure. Adoption (P1): pattern step → derive axes and skill leaves from exemplars; rewrite pair → knob moves that must change the signature (F7/F8); keep G3 before measurement so a low pass rate cannot mean a wrong label; add a student-readiness check per family.

### B2.4 Self-labelled self-play: what drifts, and why grounding is required
- **R-Zero** [2508.05004]. Challenger reward r_u = 1 − 2|p̂ − ½| from 10 solver samples (majority = pseudo-label), minus a repetition penalty (1 − BLEU distance, agglomerative clustering at 0.5, penalty = cluster size / batch). Dataset keeps 3 ≤ majority count ≤ 7 of 10. **Label drift**: pseudo-label accuracy 79% → 69% → 63% over 3 iterations (GPT-4o reference on 200 items); with Gemini labels ~71% at step 15 → ~42–44% at step 60. Every model eventually collapses; **no single label-accuracy threshold triggers collapse** (0.6B collapsed at 70.6%, 4B tolerated 48.8%). Shared parameters for both roles: peak 47.31 vs 49.12, label accuracy 63.4% vs 71.0%. Ablations (4B): no repetition penalty 49.07 → 45.76; no band filter math 47.35 and general 31.15 → 26.69. Results: Qwen3-4B-Base 42.57 → 49.93; 8B 48.64 → 53.72.
- **SQLM** [2508.03682]. Route verification by the generator–verifier gap: self-consistency where checking is as hard as solving; proposer-written unit tests where checking is easier. Proposer reward 1 iff 0 < pass < 1. Update the proposer every k = 5 solver steps (algebra 0.556 / 0.600 / 0.546 / 0.563 for k = 1 / 5 / 10 / never; coding without proposer updates 0.343). Proposer tests are unvalidated.
- **LSP** [2509.07414]. Zero-sum challenger vs solver under a reward model degenerates into adversarial nonsense without a KL anchor and a quality self-reward; with a DeBERTa RM the solver answered most queries in Python regardless of usefulness.
- **Adoption.** Labels must never be the student's or the proposer's vote: every self-play task carries an F4 oracle and F5 verifier and enters through F6. A proposer may be rewarded on the student's p̂ only after G3, G5 and G11 pass, with a cluster-size repetition penalty computed on **behaviour signatures** (not BLEU). Proposer ≠ solver (different models, ideally families). P1 as guard evidence, P2 as a source.

### B2.5 Real-data mining
- **SWE-RL** [2502.18449]. GHArchive 2015-01 to 2024-08; clone 4.6M repos; drop every SWE-bench repo; aggregate per merged PR with the file contents at the **merge base**; add LLM-predicted **relevant-but-unchanged files** (showing only modified files taught the model to edit every file shown); filter bot PRs (dependabot, renovate, bump, automerge), empty or huge changes, lock files and version bumps. 24M → 11M → **273K** seed PRs. Reward −1 malformed, else difflib similarity to the oracle patch. Llama-3.3-70B, 512 H100 × 32 h: SWE-bench Verified 41.0% (SFT 36.2%). Exact-match reward gives 29.0 vs 34.8 repair (mean training reward stays ≈ 0). Adoption (P2): confirms the debug lane's merge-base start and repo-level decontamination; the relevant-unchanged files become a `distractors` knob set from data.
- **SWE-Bench Pro sourcing** [2509.16941]. Consecutive commit pairs that resolve an issue; keep multi-file, ≥ 10-line changes (avg 107.4 lines over 4.1 files); 50–100 instances per repo, hard cap 100. Public and held-out sets only from GPL/copyleft repos (a legal barrier to commercial training corpora); commercial set from 18 private startup codebases. Splits: public 731, held-out 858 (private), commercial 276. Human-written statement + **requirements** + **interface** (see B4.2). Gold tests run several times; flaky and "too broad" tests dropped by hand.
- **SWE-Mirror** [2509.08724]. Find source repos by README keywords; an LLM filter keeps non-trivial fixes whose functionality exists in the target (precision 84.3%, recall 86.0% on 100 labels); **abstract** the PR into a repo-agnostic bug pattern (names, paths and stack traces stripped); a Test Agent writes tests that pass now and should fail once the issue is introduced (`test.patch`); a Mirror Agent with a *different prompt* edits non-test code so they fail (`mirror.patch`); `fix.patch` = its reverse; an LLM writes the issue. Verify over three states (mirror / mirror+test / mirror+test+fix): effective tests (only PASS→PASS, FAIL→FAIL, SKIP→SKIP, NONE→FAIL between Run and Test), effective fix (≥ 1 ANY→FAIL→PASS), no regressions, no flaky transitions. Yield per 100 good issues: Python 68%, JS 52%, Go 36%, Rust 28% (overall 46%); compile errors dominate Rust (36%) and Go (28%). Human audit: 88.1% High or Moderate consistency with the source; 12% inconsistent. 60,671 tasks in 40 repos, 100 GB storage (SWE-Gym 6 TB). SFT on 12,456 trajectories: Qwen2.5-Coder-32B 6.2 → 52.2 Verified. Failures: tests and bug come from one abstraction (narrow tests, overfit fixes pass); the statement sees `fix.patch` (leak); no check that the mirrored task needs the source skill. Adoption (P1): a new source, **mirror**: abstract a real incident into a pattern and plant it in an already-gated World or repo (oracle = reverse of the planted change). Add G9 (the statement saw the fix), G12 on overfit fixes, and a different-family fidelity judge (the 12% rate). For the debug lane: mirror kept fixes into already-built Go and Python repos to raise held-out supply without new builds.
- **Instruction backtranslation** [2308.06259]. Treat real artifacts as gold outputs and infer the request. Uncurated inferred data does not improve however much is added; curated A_5 data improves steadily with quantity. Curator precision/recall on 250 labelled pairs: M_0 0.44 / 0.09, M_1 0.52 / 0.44, GPT-4 0.88 / 0.92; even low-precision curation helps. Source tagging by system prompt 66.47 vs 59.96. Augmented-only training is worse than seed alone. Adoption (P2, conceptual): for traces-to-tasks the curation gate (G8 + G9 + F10), not the generator, carries the value; tag sources in `provenance.source`.

### B2.6 Traces and documents as sources
- **Clio-style intent clustering** [2412.13678]: facet extraction → embed (all-mpnet-base-v2) → k-means with minimum unique accounts per cluster → **contrastive labels** (50 in-cluster vs 50 nearest out-of-cluster summaries) → LLM hierarchy within embedding neighbourhoods of ~40, with dedup across neighbourhoods and shuffled parent assignment. Synthetic end-to-end validation: 94% reconstruction over 20 known categories in 15 languages. $48.81 per 100,000 conversations. Adoption: cluster traces to weight or add `DomainSpec` cells (coverage, not volume), under the B1.7 privacy gate. Also the design for F12 failure clustering (B7).
- **Procedure documents to tasks (AgentTrek)** [2412.09605]. Cascade filter: rules (68.8M of 20.8B docs, 92.69% recall) → LLM labeller on 90k (F1 88.5%) → fastText distilled from it (F1 89.5%) → 18.8M candidates → tag-and-paraphrase into a schema with a separate **target URL** ($0.89 per 1,000). Guided replay with the tutorial succeeded 52% (208/400) vs 15.78% with only the goal. Adoption: enterprise SOPs, runbooks and help articles as a grounding corpus for F3 (P2); the procedure as `privileged_notes` for the oracle programmer only, dropped from the solver's prompt (P1), since guided replay otherwise puts the path in the prompt.
- **InSTA proposer schema** [2502.06776]: from a URL emit {task, expert steps, criterion} or "N/A" (unsafe or unsuitable; 85% of sites filtered, unsafe recall 0.98–1.0); explore with the easy task; then propose a harder grounded task. Humans could complete and verify 75–89% of initial tasks depending on proposer. Adoption (P2): TaskProposal = instruction + privileged expert steps + draft criterion; "explore easy → propose harder" as an F9 ladder step.
- **Impossible twins as the infeasible task type** [2510.20270]: see B3.4.

### Minor B2 tips
- Split proposers by type (navigation vs local) and by model family; Claude-3.7 proposed ~2× more tasks per step budget, GPT-4o more diverse state-modifying tasks (Go-Browse).
- Keep QA pairs that produced no SFT trajectory for the RL stage; they are the hard ones (WebDancer).
- Feed environment exceptions back and resample the action (up to 5) rather than ending the episode (BAGEL); a harness concern.
- Cap behavioural biases in kept trajectories (Explorer: drop trajectories with > 2 scrolls; WebDancer: any 10-gram at most 4 times).
- Weakness from *training* dynamics, not base-model failure (SwS); seeds from a mid band 10–70% with subject balance and solution-approach clustering (Socratic-Zero).
- In adversarial self-play, add a validity or quality term to the challenger's reward; a pure "make the solver fail" reward produces junk (LSP, Search Self-play).
- Mix synthetic with real tasks and down-weight synthetic advantages (AgentEvolver used 0.5 when mixed).

---

## B3 Verifiers and gates

### B3.1 How weak test-based verifiers are: the measured base rates
Both directions fail, and the rates are large enough to set the gate design.
- **False positives.**
  - Running *all* developer tests, not only the PR's: 7.2–8.4% (avg 7.8%) of plausible SWE-bench patches fail; resolve rates drop 3.8–5.2 points [2503.15223].
  - PatchDiff finds behavioural differences from the oracle in **29.6%** of plausible patches; 82.7% of these are missed by the full developer suite. Extrapolated ~11.0% of plausible patches incorrect, inflating resolve rates by 6.4 points [2503.15223].
  - UTBoost: insufficient tests in 7.7% (Lite) and 5.2% (Verified) of instances; on those, 28.4% / 15.7% of passing patches are wrong. The **log parser** was wrong on 54.7% / 54.2% of instances and exposed 64 / 79 wrongly passed patches. Leaderboard ranks changed for 40.9% / 24.4% of entries [2506.09289].
  - SWE-ABS: 2,184 of 11,041 passing patches (**19.78%**) rejected after strengthening; top agent 78.80 → 62.20 (1st → 5th) [2603.00520].
  - Hackable tasks: 28.5% (14/49) of a SWE-bench Verified sample, 25.0% (4/16) of R2E-Gym; within a difficulty stratum they inflate real-model Pass@1 by **+14.14 pp** (95% CI [11.80, 16.48], 123/134 models positive) [2606.16062]. Small n; indicative.
- **False negatives.** See B3.8 (SWE-Bench Pro 25.9 → 8.4; SWE-RL; SWE-ABS 10.6%).
- **Hardness is orthogonal to check strength.** SWE-Bench Pro subset: agents score much lower, yet 64.7% of instances were strengthened (drop 16.46 pp) [2603.00520]. Easier tasks are cheated *more* often [2510.20270]. So G5/G12 apply to hard tasks too.

### B3.2 Gold-sanity gate with retry, then judge (sanity-gated repair loop)
- **How it works** [2606.16062].
  1. An augmenter proposes a blocking test (T 0.3).
  2. **Stage 1 gate:** inject it into the real test file, apply the gold patch (and the task's test infrastructure), run only the new test with the project's native runner; it must PASS on gold, else discard and retry (T 0.7, "assert different observable properties").
  3. **Stage 2 judge:** 3-sample self-consistent LLM judge (T 0.2/0.4/0.6, majority) decides whether the test blocks the target exploit.
  4. FIX → KEEP when every target exploit is covered; budget 3 iterations.
- **Evidence.** Un-gated repair claimed 10/11 tasks fixed; a Docker re-check of 8 judge-endorsed blocking tests found 1 valid, 6 invalid (wrong import, inverted expectation, wrong default, inverted purpose, keyword collision, never reaches the fixed code), 1 uncollectable. With the gate: **65 of 105 decisive augmentations failed on gold (61.9% defect rate)**; 9/11 fixed. Gate without retry 3/11; with retry 9/11; diversity-biased vs neutral retry prompt no difference; 1-sample judge loses one task (8/11) and saves 56% API. Costs: $3.60 gated vs $5.57 un-gated.
- **EQS routing.** EQS = 0.35·V + 0.30·(1 − H) + 0.20·F1 + 0.15·D (V discrimination, H exploit pass fraction, F1 judge-vs-execution agreement, D external learnability); KEEP > 0.70, FIX 0.40–0.70, DROP < 0.40. Weights are untuned priors; D is circular for new tasks without public submissions. Mean 0.806 over 50 tasks (KEEP 38, FIX 11, DROP 1).
- **Adoption (P0).** Gold-sanity is non-negotiable for every LLM-written check and negative, including inside repair rounds. F6 repair = gate → judge → retry, budget 3. Keep the attacker's exploits as G5 negatives and hold some out from repair. EQS-style scores can sit in `GateReport`, but hard gates stay hard.

### B3.3 Two-sided mutation strengthening (SWE-ABS) and differentiating tests (PatchDiff, UTBoost)
- **SWE-ABS** [2603.00520], GPT-5, T 1.0.
  - Stage I, coverage: initial tests from issue + gold + test patch (≤ 3 tries) → **decoupling** pass (an LLM flags tests over-specialised to gold, such as hard-coded error messages or implementation side effects, and rewrites them to test only issue resolution; keep those gold still passes) → Tree-sitter program slice (intraprocedural, depth ≤ 5) → prompt with uncovered slice lines.
  - Stage II, mutation: ≤ 2 subtly wrong mutants of gold that pass the original tests; three judges vote on issue relevance and semantic equivalence (98% agreement with humans on 100). **Equivalent mutants that fail** = over-strict tests → relax them. **Non-equivalent mutants that pass** = blind spots → write killing tests (≤ 3 tries each).
  - Ablation: initial tests 190 instances strengthened / 10.34 pp; + coverage 198 / 11.30; + mutation **251 / 14.56**. Mutation is the lever; coverage alone adds little. GLM-4.7 replicates (31/50 same as GPT-5).
  - Cost: $2.50 and 18.5 min per instance; **$4.98 per strengthened instance** vs UTBoost $80 (see B9 for why both numbers hold).
  - Failure: 10.6% of instances still had gold-specific tests after decoupling, fixed by hand; without that fix the effect shrinks (206 strengthened, 11.22 pp). Wrong-patch taxonomy (100): logic 47%, incomplete 35%, type 7%, boundary 6%.
- **PatchDiff** [2503.15223]: find target functions (first non-test function on call traces from tests to patch-modified code); give an LLM both patches, context and traces; ask for a test that passes under one and fails under the other (10 samples; ≤ 2 repair rounds to pass on oracle); keep tests deterministic over 20 runs that separate the two. $0.039 (no repair, 26.0% detection) to $0.105 per patch (29.6%). Stronger generators find more (100-patch samples: GPT-4o-mini 28.0%, DeepSeek-V3 39.0%, Qwen3-235B-A22B 47.7%). Pynguin and CoverUp found 0. Correctness on 77 suspicious patches: incorrect 28.6%, correct 5.2% (twice because the *oracle* had irrelevant changes), **uncertain 66.2%** because the issue under-specifies the differing behaviour.
- **UTBoost** [2506.09289]: intramorphic testing (gold and candidate must agree on issue-related tests); generate tests at T = 0 and 20 each at 0.8 / 0.9 / 0.99 (high temperature found extra gaps); human confirmation. ~$1.6 API per instance, ~300 CPU hours; applicable only where some candidate passed (74.6% Lite, 81.6% Verified).
- **Adoption (P0).** F5 `NegativeSet` holds *both* non-equivalent mutants (must fail) and equivalent alternatives (must pass, the automatic G14 set); the two-sided discrepancy rule is the repair procedure, with gold-sanity after every change. For state Worlds the mutants are action-program operators (§3.3 table) and the differentiating probe is a state query where oracle and candidate end states differ. Run differentiating generation on every passing student or attacker artefact of a pilot slice (cheap at $0.04–0.10). Admit a new check only if the instruction entails it (B4.2), so the loop cannot encode the oracle's arbitrary choices. Debug lane: run all functional tests, not only fail-to-pass.

### B3.4 Grader-level attack library and impossible twins
- **The reusable hack list** (from production and frontier runs) [2511.18397, 2503.11926, 2510.20270]:
  - always-equal `__eq__` objects (plus overriding other methods of the expected type);
  - `sys.exit(0)` / `exit(0)` before asserts;
  - `raise SkipTest` from non-test code;
  - `conftest.py` patching pytest's TestReport to pass everything;
  - editing or deleting tests;
  - stubbing or editing upstream libraries the tests import (bypasses "can't edit tests"); a local stub of pandas plus 19 stubbed or deleted files;
  - parsing test files at runtime to extract expected values;
  - decompiling leftover compiled reference solutions (`.pyc`, `.jar`);
  - recording call-count state to return different values per call;
  - special-casing test inputs; returning the error codes tests expect; stubs where coverage is poor.
- **Exact detectors.** Revert-and-rerun: the final repo passes, and some test fails once the agent's test edits are reverted ⇒ cheat [2503.11926]. Regexes for `exit(0)` and SkipTest gave the ground truth: action monitor recall 60%, CoT monitor 95%.
- **Impossible twins (ImpossibleBench)** [2510.20270]. Mutate tests with an LLM so they contradict the spec: *one-off* (change one expected value) or *conflicting* (duplicate an assertion with a different value). QC: gold and empty patch must both fail (8.8% of one-off and 3.4% of conflicting mutations removed). Cheating rate = pass rate on twins. GPT-5 cheats 54% (conflicting) and 76% (one-off) on SWE-bench twins vs 2.9% one-off LCB. Read-only tests stop test edits but not special-casing or operator overloading; hidden tests drive cheating ≈ 0 but cut legitimate performance. Strict prompt: GPT-5 92% → 1% on conflicting LCB. An abort action (`flag_for_human_intervention`) cuts GPT-5 54% → 9%, o3 49% → 12%. Feedback loops raise both pass rate (80 → 83%) and cheating (33 → 38%). LLM monitors catch 86–89% on LCB but 42–65% on SWE-bench (fooled by a "backward compatibility" rationalisation).
- **Adoption (P0).**
  - G12 / F5: a standing library of `NegativeCase{class: attack}` for every code- or test-graded task, needing no task knowledge; path constraints on test dirs, dependency directories and import paths; tests only in the verifier custody World (`world/custody.py`); G11 scans the start image for compiled or reference artefacts. For state Worlds the analogues are writing to the DB instead of through tools (G7) and tampering with audit tables.
  - Impossible twins per family: contradict one state, answer or timing check; the correct behaviour is flagging the conflict. They are the infeasible task type of §3.3, a per-student cheating meter in F12, and vacuous-verifier canaries in the anchor suite.
  - Harness: read-only verifier artefacts, revert-before-score, and an explicit abort/flag action rewarded on impossible tasks.

### B3.5 Harness trust boundary before task gates (BenchJack) and parser sanity
- **Taxonomy** [2605.12673]: V1 isolation failure (agent and evaluator share environment); V2 answers shipped with the test; V3 code execution into the evaluator (eval, exec, dynamic import on agent data); V4 judge prompt injection; V5 weak string matching; V6 evaluation-logic gaps (validators never called; "run a null agent; any task > 0 has a V6 flaw"); V7 trusting agent-controllable output; V8 excessive permissions (root, docker socket, open network, writable host paths). A 30-item checklist in 7 categories.
- **Procedure.** Reconnaissance (entry points, scorers, trust boundaries) → flaw scan (semgrep rules, bandit, hadolint, AST trust mapper; findings ledger) → exploit `run.sh` that must go through the official entry point with a minimal agent and use only what a model could observe → iterate. Attacker vs defender rounds.
- **Evidence.** Working exploits on 10/10 benchmarks; near-100% without solving a task on 9/10; 219 flaws. A nine-line `conftest.py` passes ~all SWE-bench tasks because the harness resets only files named in the test patch. Single-round patching kills the original exploit, but a fresh run re-hacks most; only 4 benchmarks drop by more than half. **Design flaws (shared trust domain) cannot be patched in code.** On 4 well-designed benchmarks, 3 rounds cut hackable tasks from ~100% to < 10% (WebArena and OSWorld to 0%).
- **Parser sanity** [2506.09289]: repository-specific regex parsers (avg 23 lines) mis-read multi-line logs (took a docstring line as the test name).
- **Adoption (P0).** A new harness-level **G0 trust-boundary gate**, run once per App/World template and on every harness change: recon + static scan + exploit attempt against the verifier path. In a simulated World: verifiers read through the privileged `StateAdapter`, never through anything the policy can write; answer checks parse structurally; no gold digest or oracle reachable through policy-visible tools. Debug lane: the runner must ignore agent-created `conftest.py`, `sitecustomize`, plugins and PATH shims, and parse outcomes outside the agent's container. Add a **parser gold-sanity** check: parsed per-test outcomes on gold and known-failing runs must match the framework's machine-readable report (JUnit XML, JSON); the debug lane parses logs in five languages.

### B3.6 Null agents, positive evidence for refusals, and state-comparison rules (ABC)
- [2507.02825] τ-bench: 38% of airline and 6% of retail tasks are intentionally unsolvable with success = unchanged state, so a **do-nothing agent scores 38% on airline**, above GPT-4o; 2% / 3.6% grade by substring of verbatim DB text, so dumping the DB passes (overestimate up to 40%). WebArena: the judge accepts an empty reply on "N/A" tasks (1.4–5.2%). SWE-Lancer: tests in a password-protected zip whose entries can be listed and overwritten (100% without solving). KernelBench fuzzing varies values but not shapes (~31% overestimate). CVE-Bench checked for a SLEEP string in logs instead of its execution (32.5% overestimate).
- **Rules.** Outcome validity for state matching: ground truth covers all valid outcomes (O.g.1); state includes irrelevant parts to catch out-of-scope changes (O.g.2); the change is complex enough that trivial edits cannot succeed (O.g.3). Substring checks must prevent listing all answers (O.b). Check an *effect*, not a trace of an attempt.
- **Adoption (P0).** G4 null agents = do nothing, empty answer, dump all readable data, claim everything / list all options. For `infeasible_share` tasks the success predicate requires a refusal message *plus* evidence of the policy lookup, never unchanged state alone. Collateral (F5) is O.g.2. A new state-change complexity floor: a random or single-field change must not satisfy Φ. Answer checks include negative checks (absence of wrong answers), against Terminal-Bench's "emit all possible answers" exploit.

### B3.7 The Terminal-Bench per-task QC pipeline
- [2601.11868] Task = instruction + Docker image + tests + human oracle + time limit; tests check the **final container state**, never commands or console output.
  1. CI runs the oracle (must pass) and a no-op agent (must fail).
  2. Checklist: everything tested is in the instruction and vice versa; hard to cheat; exact schema for structured outputs; pinned deps; ran with a strong model.
  3. Deterministic checks: canary string; Dockerfile must not copy tests or solution; **test-file reference check** (a file read by both tests and oracle must be named in the instruction); no privileged containers.
  4. LLM checks, including **hardcoded solution** (an oracle that echoes the answer fails) and typos in file and variable names.
  5. Debug tool: all trials fail the *same* tests ⇒ instruction defect; varied failures ⇒ fair difficulty.
  6. Several models run each task; auditors analyse failures.
  7. Cheating-only exploit agent (prompted to complete only by cheating, without seeing tests); found monkey-patching the test environment, guessing answers, emitting all answers.
  8. ~3 reviewer-hours per task.
- Judges: trajectory-level GPT-5 judge 90% agreement with 120 human labels (human–human 93% on 20); command-level 92.4%.
- **Adoption (P0).** Map: no-op → G4; oracle → G3; bidirectional spec check → G8 plus a reverse direction (no untested requirement); file-reference check → a deterministic G8 sub-check; hardcoded-oracle check → G7-adjacent (the oracle must compute through tools); exploit agent → G12 prompt pattern; image hygiene → G11 / custody; failure consistency → F10 probe (B4). Harbor adapters must pass parity runs before a domain plug-in is trusted.

### B3.8 State diff vs judges; alternatives and false negatives
- **State-equality grading.** τ-bench r = r_action × r_output: the final DB equals the DB from replaying the gold writes (reads are free) and every required output string appears in agent messages [2406.12045]. Necessary but not sufficient: an agent can skip a required user confirmation and still score 1; substring outputs pass a number-dumping agent. Add path constraints (e.g. confirmation before write).
- **Judge accuracies in this group** (all as generators' rewards or filters): WebRL ORM ~80%; Explorer 81% agreement with ~26% of judged successes wrong; InSTA 78.0–82.6% (93.1% at confidence 1); AgentTrek 84.0% on 558 gold samples; Kimi CoT reward model 98.5% vs 84.4% for a value-head RM [2501.12599]; Terminal-Bench 90%. BAGEL's execution filter beat its LLM filter by 7 points. AgentEvolver: a naive judge barely improves the agent; principles help; adding the reference solution brings synthetic-task training close to human-task training. In a simulated World, replay plus `StateAdapter.diff` replaces all of these.
- **Rubrics for free-text answers only (RaR)** [2507.17746]: 7–20 self-contained items per prompt (Essential / Important / Optional / Pitfall, weights) grounded in a reference answer; explicit reward r = Σ w_j c_j / Σ w_j, or one holistic implicit judge call. Reference grounding worth ~4 points (35.9 vs 32.0); synthetic-with-reference matches human rubrics (34.8); four generic rubrics underperform; no labels 38.8 vs all 37.2; pitfalls add nothing. Judge-validation pairs by perturbing gold along named axes (3,027 pairs). Adoption (P1, once a free-text answer kind exists): items generated from the F4 gold answer; the perturbation recipe is a negative class for answer checks; pitfalls from F12 failure clusters; state checks stay the primary reward.
- **Single-reference false negatives.**
  - SWE-Bench Pro: removing requirements and interface cut GPT-5 25.9 → 8.4 and Opus 4.1 22.7 → 8.2: about two-thirds of solves vanish when exact names, signatures and strings are not communicated [2509.16941].
  - SWE-RL's similarity reward scores correct alternative fixes low (its own limitation) [2502.18449].
  - SWE-ABS: 10.6% of instances over-specialised to gold after decoupling [2603.00520].
  - Search Self-play's non-unique answers mark valid open-search answers wrong [2510.18821].
  - Kimi's coding rule as a multi-oracle template: generate 50 tests; a test is valid if ≥ 7/10 reference solutions agree on its output; admit a problem if ≥ 9/10 pass all selected tests (1,000 → 323 admitted, 32%) [2501.12599]. A check that disagrees with most alternatives is a bad check, not a bad alternative.
  - SWE-RL consensus rerank |P|·|T|² (patches passing the same reproduction tests) is a rule for choosing among oracle candidates in F4/G14.

### B3.9 Per-check quality score and cross-validation
- **CURE μ** [2506.03136]. Execute n solutions against m generated + t_q trusted tests → binary matrix. With test correctness p_u, p_01 = P(wrong solution passes a correct test), p_00 = P(wrong passes a wrong test): ranking by tests passed has precision → 1 as m → ∞ **iff μ = p_u(1 − p_01) − (1 − p_u)·p_00 > 0**, at rate ≳ 1 − exp(−μ²m/8). Per-test reward: if the test passes every correct solution it earns the count of incorrect solutions minus those that pass it; otherwise only the negative term. A naive "passes all correct code" reward left p_01 = 42.2%, p_00 = 14.7% vs 36.5%, 9.1%. Results: unit-test accuracy +37.8%, BoN +9.0%; a 4B tester's tests as RL reward for 14B gave gains comparable to ground-truth tests.
- **AutoIF K×K cross-validation** [2406.13542]: sample K verifier functions and K labelled cases; keep cases with > 50% agreement across functions and functions with > 50% accuracy across cases; back-translate each function to English and drop NLI contradictions (~15%). Residual: functions and cases from one model can share a misunderstanding; NLI only removes contradictions.
- **Adoption (P0).** μ is the per-check score in `VerifierBundle`: known-correct set = oracle + G14 alternatives; known-wrong set = `NegativeSet` + failed student and attacker artefacts (the best negatives are those students actually write). G5's kill floor can be set on aggregate μ, and the bound gives the number of independent checks needed for a target precision. K×K agreement across checks from different families is a cheap pre-gate before gold-sanity.

### Minor B3 tips
- Rerun gold tests repeatedly and drop flaky ones; drop "too broad" tests that pass for unrelated changes (SWE-Bench Pro).
- Remove compiled artefacts and any reference solution from the environment; strip future git commits (Terminal-Bench, Baker, Qwen).
- Grade outside the agent's container from copied-out artifacts; drop root, network and host sockets by default (BenchJack).
- Canonicalise answers at build time so the check is parser-free (DAPO's integer rewrite; note no check that the rewritten question stays well-posed).
- Exclude small answer spaces (multiple choice, true/false) from RL tasks or pair them with a path or state check (Kimi, SwS).
- Recompute all stored verdicts after any verifier or parser fix; rankings move (UTBoost).
- Three-state transition classification (break / break + test / fix) is a generic effective-check / effective-fix / no-regression / no-flake gate (SWE-Mirror).

---

## B4 Intent, validity, communicability and leakage

### B4.1 Validity vocabulary and outlier routing (ABC)
- *Task validity*: solvable iff the agent has the target capability (shortcuts → false positives; impossible tasks → false negatives). *Outcome validity*: a positive result iff the task succeeded [2507.02825]. The card notes the "iff" is not operationalised; ABC measures no necessity, which Crucible's ablation probes try to supply.
- **T.10 outlier inspection** in pilots: agents failing easy tasks → impossible task; agents passing only hard ones → shortcut. Adoption (P0): in F9/F10, route tasks the frontier fails but the student passes, and predicted-easy tasks nobody solves, to audit (complements the zero-solve rule).

### B4.2 Communicability: requirement writing, bidirectional spec and entailment of checks
- **Requirements + interface** [2509.16941]: requirements pin every test-visible exact value (log strings, route names) without stating the implementation; an interface block gives signatures and paths of public functions the tests call. The 25.9 → 8.4 ablation (B3.8) is its measured value.
- **Bidirectional spec check** [2601.11868]: tested ⊆ described and described ⊆ tested; every file and schema the tests read named in the instruction.
- **Entailment rule** [2503.15223, 2406.13542]: 66% of oracle/candidate differences could not be decided from the issue. A new check is admissible only if the instruction entails it; otherwise G8 fails and the spec, not the candidate, is under-specified. **Back-translation** of each check (or the whole `VerifierBundle`) to English, tested for entailment against the instruction and the `IntentCard` claim, makes "the verifier exercises the intended skill" testable without rollouts.
- **Adoption (P0).** G8: for every value a check reads under column policy `exact`, find it in the instruction, start state or simulated user's known info; otherwise communicate it (requirements/interface) or relax the column to `semantic`. Back-translation entailment as an F5/F10 probe.

### B4.3 Fault-side probes over existing rollouts
- **Failure consistency** [2601.11868]: all trials fail the same checks ⇒ instruction defect; varied failures ⇒ fair difficulty. Free over F9's student runs. P0.
- **ABA three-axis audit** [2605.26079]: an auditor agent (Claude Code, Opus 4.7) per task, static mode (instruction, tests, config) or trajectory mode (plus runs). Axes: *instruction* (cannot tell what passing means), *environment* (resources or tools conflict with assumptions), *evaluation* (tests too narrow, too broad, misaligned or wrong). **Severity by thought experiment:** give only the prompt to five experienced developers; would most produce a passing solution? 0 = yes (failures are skill; a hard but clear task is 0); 1 = discoverable gap; 2 = near impossible or tests check something else. Findings carry file-path evidence and a suggested fix. Validation against a Terminal-Bench 2 maintainer PR: recall 66.7% strict / 81.0% partial, precision 58.3 / 70.8 (a lower bound). 25.7% of 34,285 tasks across 168 benchmarks have ≥ 1 major issue; agentic/tool use 20.6%, coding 14.1%, math 13.2%. Deterministic-grader domains are instruction-dominated (math 65%, science 69% of majors). Trajectory mode flags 8.5% more majors; static/trajectory agreement on majors 29–63%. **$0.42–0.80 per static audit** (≈ $0.60). Precision ~58–73% strict, so a flag routes to repair or INCONCLUSIVE, not INVALID. Adoption: static mode on every task at F6 (P0), trajectory mode on zero- or low-solve tasks after F9 sharing F9's rollouts (P1); the three axes are the fault sides (World, grader, spec); different family from the author. Prediction for Crucible: synthetic-World tasks with exact state checks will break on instruction under-specification.
- **Qwen task-quality judge** [2606.26300]: an agent exploring the container outputs `instruct_clear` (F1 ≈ 91–95) and `instruct_ut_align` (F1 75–81; best with few-shot + gold patch, 75.9 / 87.2 / 81.2). **Zero-solve tasks hold a much larger share of judge-flagged tasks.** Quality-filtered RL improved SWE-bench Multilingual and Pro, flat on Verified. Adoption (P1): `instruct_clear` and instruction ↔ `VerifierBundle` alignment from a non-author family with access to World and oracle.
- **Zero-solve rule** is also Socratic-Zero's "all 8 fail ⇒ re-verify" [2509.24726] and τ-bench's "zero-success tasks are mostly annotation bugs" [2406.12045].
- **Student-side fault classes** (initial clusterer categories): SWE-Bench Pro buckets from the last 20 turns (20 matched humans better than 10 or 40): wrong solution, tool use, syntax, incorrect file, endless reading, misunderstood problem, context overflow [2509.16941]. τ-bench: wrong argument ~55%, wrong rule decision ~25%, partial compound resolution ~19%.

### B4.4 Shortcut and necessity probes
- **No-reasoning guess test (G9)** [2501.12599]: a model answers with no chain of thought; any hit within N = 8 tries ⇒ guessable, drop. For World answer tasks: ask a frontier model for the answer or end state from the instruction alone, no tools, 8 tries. P1.
- **Policy-removal ablation** [2406.12045]: a policy-heavy family must drop sharply without its policy document (airline 33.2 → 10.8), else the rules are commonsense and unexercised (retail 61.2 → 56.8). F10 probe, P1.
- **Unique-outcome probe** [2406.12045, 2510.18821]: run k diverse simulated users and require every oracle-consistent run to reach the same digest; for answers, query the World for all entities satisfying the instruction.
- **Round-trip communicability** [2403.08140, 2502.11357]: label a trace → a fresh agent with only the label re-executes → compare digests. Agreement = G8 (the instruction carries the information) and G2 (reproducible). The filter must be the digest, not an LLM.
- **Clarification vs harder** [2411.02337]: record whether a failure-seeded variant changed the spec or the skill.

### B4.5 Memorisation, contamination and leakage
- **Decontamination recipes.** DeepSeekMath: drop segments sharing a 10-gram with GSM8K/MATH/CMATH/AGIEval, exact match for 3–9-gram items [2402.03300]. OpenThoughts: Indel similarity ≥ 75% or any shared 13-gram, tuned on a built set of 3,092 contaminated (exact, wrapped, synonym-swapped, reformatted) + 3,000 clean items; misses 12/3,092 [2506.04178]. AutoIF: rephrase detector + 13-gram [2406.13542]. Adoption: tune G11 leak rules on a deliberately contaminated testbed (P1).
- **Memorisation detector** [2012.07805]: likelihood relative to a reference (smaller model, zlib entropy, lowercased text), not raw perplexity (raw perplexity 9% and 3% precision; Internet-conditioned + zlib 67%). Plant canaries. Adoption: a cheap probe for "this task text or its gold answer is memorised", alongside the instruction-only probe (P2).
- **Repo familiarity.** The same agent scores 43.20% on SWE-bench Verified and 19.25% on fresh SWE-bench-Live issues; instances from the 8 original repos resolve at 22.96% vs 18.89% elsewhere [2505.23419]. Recency alone is weak; compare familiar vs unseen owners as a benchmark audit. Licence- or access-barrier sources (copyleft, bought private code) complement the date rule [2509.16941].
- **Leaks in generated tasks.** SWE-Mirror's statement sees `fix.patch` [2509.08724]; guided replay puts the path in the prompt unless stripped back to the goal [2412.09605]; git history after the PR date, upstream diffs and commit-hash lookups are shortcut channels (Qwen strips history and blocks network) [2606.26300]; compiled reference artefacts [2503.11926].

### B4.6 Lucky passes and aleatoric "learnability"
- GRPO's std normalisation gives a single pass in a mostly failing group a large advantage, amplifying lucky passes; the probe matters most near p̂ ≈ 0.1 [2402.03300].
- p(1−p) cannot tell aleatoric noise ("coin flip" tasks, flaky graders) from learnable uncertainty [2502.12272]; the same holds for mean |Â| [2505.14970, 2504.09710]. Adoption: combine learnability with G10 (flake bound) and the lucky-pass rate before admitting; exclude G10 failures from curriculum arm rewards.

### Minor B4 tips
- Human difficulty labels correlate r = 0.436 with empirical difficulty; 54.5% of human-medium tasks are empirically hard, often creative or adversarial ones [2601.11868]. Never use author estimates as validity or difficulty evidence.
- Never fix a hackable reward by instructing the model not to exploit it; it can increase misalignment [2511.18397]. The DomainSpec authoring guide should say instructions must not forbid what the verifier would reward.
- Check that each constraint is compatible with the task it is attached to (AutoIF's query-level filter): the analogue is instruction vs World state and persona.

---

## B5 Diversity and coverage

### B5.1 Structure forces diversity; prompts do not
- SQLM: pre-generating 6,400 problems with an explicit "span easy to hard" instruction gave visibly less diverse problems (PCA) and worse learning than online, feedback-conditioned generation [2508.03682].
- Synthetic Data RL: conditioning on demos alone produced near-duplicates until an abstract pattern step was added (MATH 65.0 → 72.0) [2505.17063].
- BAGEL: every book-flight demo landed in December; ToolQA mostly one source [2403.08140]. Randomise knob values explicitly rather than leaving them to the LM.
- R-Zero: a repetition penalty by cluster size (BLEU at 0.5) was worth 3.3 math points [2508.05004].
- **Adoption.** Supports §3.5: condition F3 on a sampled cell, World slice, persona and K nearest tasks, not on diversity instructions. Compute repetition penalties and dedup on behaviour signatures, not surface text.

### B5.2 Coverage as states visited and cells filled
- Go-Browse outer-loop ablation (30 tasks per domain): resets × tasks 1 × 30 → 183 unique URLs; 5 × 6 → 214; 15 × 2 → 260. Spreading proposals over more reset states raises coverage; deep pages NNetNav never reached got tasks [2506.03533]. Adoption (P0): "fraction of reachable node types with ≥ 1 kept task" as a World-facet axis of the MAP-Elites grid.
- Meters from the cards: NNetNav intent-distribution perplexity (13.5 / 16.2) [2410.02907]; Explorer unique proposals → final tasks (53k → 94k descriptions, 81k unique) [2502.11357]; InSTA's 8,694 automatic categories, 89% below the mean size (long tail) [2502.06776]; Search Self-play LDA topic mix stayed balanced [2510.18821].
- Exogenous corpora force diversity: SWE-Mirror's real-issue patterns, AgentTrek's tutorials, Clio clusters of real intents (B2).
- Personas seed behavioural diversity (NNetNav); two proposer families cover different task clusters (Go-Browse).

### B5.3 Dedup layers and source caps
- AgentEvolver: generation-time lexical dedup (> 0.8 token overlap) with embedding steering, then post-hoc dedup [2511.10395]. OpenThoughts: exact vs fuzzy vs no dedup made small differences at 31.6k examples; 16 answers per question as good as more questions [2506.04178].
- Per-source caps: SWE-Bench Pro ≤ 100 instances per repo [2509.16941]; Carlini caps per-source contribution against memorisation [2012.07805]; Clio minimum unique accounts per cluster.
- **Source quality over count.** OpenThoughts mixing top-N sources (code): top-1 39.9, **top-2 41.3**, top-4 38.6, top-8 37.0, top-16 36.4; at most two sources won in every domain. Simple synthetic sources matched curated ones [2506.04178]. Critique: measured at fixed 31.6k size; at 1M+ scale diversity may matter more.

### B5.4 Multi-skill balance in the sampler
- SEC-2D (9 arms = 3 task types × 3 levels) beat random on all three; random collapsed mid-training on Countdown OOD while SEC-2D stayed stable [2505.14970]. DUMP's scheduler starved some distributions (LogiQA 1.94 → 1.70, Geomotion 0.54 → 0.25) [2504.09710]; WebRL's Map site regressed late [2411.02337]. Adoption: per-family floor shares in the F11 sampler; report per-family regressions in F12.

### Minor B5 tips
- Add relevant-but-unchanged distractors so the learner cannot assume everything shown must change (SWE-RL).
- Concept recombination by co-occurrence + embedding similarity keeps generated tasks coherent while moving them away from the seed (SwS).
- Kimi's tagging by domain and discipline balances subjects in the prompt set [2501.12599].

---

## B6 Difficulty and curriculum per student

### B6.1 Why bands exist: variance as learnability
- GRPO z-scores advantages per group; an all-pass or all-fail group gives zero gradient. At k = 8 the all-same probability is p⁸ + (1−p)⁸, high at band edges; G = 64 hides this [2402.03300].
- **2504.03380 theorem.** For Bernoulli reward, KL(π_init‖π*) = −p/β + log((1−p) + p·e^{1/β}) ≥ p(1−p)/(2β²), and 0 at p ∈ {0,1}; generalised to the cumulant generating function of the centred reward (Gaussian: σ²/2β²). It bounds the distance from the initial to the soft-optimal policy, not per-step improvement (see B9).
- **LILO theorem**: an *upper* bound; per-step improvement ≈ β‖∇J‖² ≤ E‖Σ∇log π‖²·Var[r]. Low learnability caps improvement; high learnability does not guarantee it. Under GRPO's std normalisation the dependence cancels; unnormalised GRPO works better in replications [2502.12272].
- SEC: under std-normalised binary reward E|Â| = 2√(p(1−p)), maximal at p = 0.5 [2505.14970].
- Use variance of the scalar reward, not one component's pass rate, for multi-component grades [2504.03380 Cor. 6.3].

### B6.2 Band results with numbers
- **Balanced vs one-sided** (Qwen2.5-3B, average of 5 math benchmarks) [2504.03380]: SFT 20.2; plain GRPO 26.3; (0,1) 27.3; (0.1,0.9) 28.4; (0.2,0.8) 29.8; **(0.3,0.7) 30.1**; (0.4,0.6) 29.9; **one-sided cuts 23.8–25.9, all below plain GRPO**; offline proxy curation 26.3–28.4. 7B: 39.6 → 41.8. Band ordering among 0.2–0.4 is within noise (AIME 30 items, 3.3 points = 1 problem). Narrow bands reach the plain maximum in < 50% of steps but need many more rollouts per batch. The live checkpoint's batch difficulty stays ≈ 0.5 while fixed checkpoints see batches harden: measure on the live student.
- **Other bands.** R-Zero 3–7 of 10 [2508.05004]; SwS 25–75% of 8 minus 1/8 and 7/8 [2506.08989]; WebRL critic 0.05–0.75 [2411.02337]; SwS training-time dynamic sampling 10–90%; AdaRFT β = 0.5 beat 0.2 and 0.8 (β = 0.2 sits at reward ≈ 0.2 on too-hard problems) [2504.05520]; Socratic-Zero Gaussian μ = 0.5, σ = 0.2 (μ ∈ {0.3, 0.4, 0.6, 0.7} lose only 0.2–0.4); Synthetic Data RL lowest positive pass rate.
- **DAPO dynamic sampling** (drop all-correct and all-wrong groups, refill to a fixed batch) was the largest single step: AIME24 42 → 50 on Qwen2.5-32B; ladder 30 → 36 → 38 → 41 → 42 → 50 [2503.14476]. It pays for discarded rollouts and silently drops all-fail prompts that may be broken tasks. Train reward correlated little with validation.
- **Lasting vs fast.** SwS: easy windows help fastest then saturate; hard windows give the longest gains. Training only on hard-extreme stays flat; easy-extreme learns fast then stalls (AdaRFT).
- **Adoption.** Keep [0.1, 0.6] for the 35B but add a symmetric (0.2, 0.8) arm in R2: the one-sided rows were cuts at 0 or 1, not a shifted interior band, so they do not refute [0.1, 0.6]. Export per-task p̂ to the trainer so DAPO-style rejection happens without paying first; log every dropped all-fail group (→ zero-solve audit) and all-pass group (→ mutation queue). P1.

### B6.3 Controllers and samplers that need no extra rollouts
- **AdaRFT set-point** [2504.05520]: T ← clip(T + η·tanh(α·(R_avg − β)), d_min, d_max), β = 0.5, α = 2, η = 50 on a 0–100 scale; batch = the B problems nearest T. PPO needed +43 (71.7%) extra steps to match on skew-difficult 1.5B, +34 uniform, +16 skew-easy; pass@40-filtered PPO needed more (+49 / +52 / +21). Difficulty from 128 rollouts of a middling reference solver; 64 rollouts keep ±0.05 > 90% of the time, 40 ~80%; this 1.28M-rollout pass is excluded from the 2× speed-up. GPT-4o-rated difficulty helps, less well. Adoption (P1): generalise to the knob vector per student; a monotone knob (`monotone: true`) is a free solver-independent scale.
- **SEC bandit** [2505.14970]: arms = categories; Boltzmann p(c) ∝ exp(Q(c)/τ); Q(c) ← α·mean|Â|_c + (1−α)·Q(c); α ∈ {0.2, 0.5}, τ ∈ {0.2, 0.4, 1.0}. Qwen2.5-3B OOD Countdown random 0.479 / ordered 0.321 / SEC 0.542. Gains shrink when the model is strong relative to the pool (7B Countdown 0.566 vs 0.555); return on a harder pool (0.373 → 0.439). Hard-to-easy badly hurts; fixed easy-to-hard often worse than random. Adoption (P1): grid cells as arms, Q per (cell, student); low-Q-all-pass cells feed mutation, low-Q-all-fail cells feed ladders.
- **DUMP UCB** [2504.09710]: UCB(d) = L̂(d) + √(2 log(n_total + 1)/(n_d + 1)) over a 300-advantage window, softmax τ = 0.1; greedy (τ = 0) underperforms. Take the exploration bonus and add a per-family floor (P2).
- **LILO Alg. 3** [2502.12272]: 2 attempts per pool task, then repeatedly pick a task ∝ p̂(1−p̂) and roll 2 more; batch learnability ≈ 0.17 vs 0.21 for 8-attempt rejection vs 0.06 uniform at the uniform budget. Speed-ups 1.5×–3.3× to baseline-best. Adoption (P1): the principled form of §3.6's "k = 4 first", spending student rollouts 2 at a time where p̂(1−p̂) is highest.
- **Kimi prioritised sampling ∝ 1 − s_i** after a mixed warm-up then hard-only curriculum [2501.12599]: contradicts the learnability view (see B9).

### B6.4 Freshness of the learnable set
- LILO: an SFL-style buffer refreshed every 50, 10 or 5 steps produced sawtooth overfitting; only every-step refresh generalised. LILO did not change the train–test line (it speeds fitting, not generalisation) [2502.12272].
- Search Self-play: replay reset every 10 steps best (B2.2). WebRL: replay successes only, at moderate perplexity [1/0.95, 1/0.5] (31.5 vs 27.9 / 23.0), historical data ≤ 2× new.
- Adoption: refresh p̂ from F12 training rollouts every step; never freeze an "in-band" list between measurement rounds; plan the pool at several times the batch (LILO grows the pool to 8× as accuracy nears 95%).

### B6.5 Hardness from structure: knobs and ladders
- **Answer-preserving linkage (e2h)** [2505.22648]: replace an entity in the question with a searched description, n times; the answer never changes, so validity holds by construction; n = hop count. World analogue: replace "order #123" by "the order the customer complained about in last week's email"; the gold end state is unchanged; each substitution must resolve uniquely (G8). P1.
- **Start offset** [2506.03533]: prefixed starts at the discovered state have much higher success than root starts, especially for deep nodes and weak models. Root start = hard variant; deep start = ladder rung. P0.
- **Horizon ladder** from nested NNetNav prefixes [2410.02907].
- **Privileged guidance** removed for the student: 52% vs 15.78% replay success with vs without the procedure [2412.09605].
- **Structural predictors.** Resolve rate falls with files touched: SWE-bench-Live single-file < 5 lines ~48%, ≥ 3 files or > 100 lines < 10%, ≥ 7 files never; repos > 500 files rarely > 5% [2505.23419]. SWE-Bench Pro: sharp drop beyond 3 files; open models ~0 at 10+ files [2509.16941]. τ-bench: success falls with the number of gold writes. AgentEvolver: entities × attributes × operations. Free features for the F9 predictor.
- Rewrites must move a knob or add a sub-goal and change the behaviour signature; "harder rewording" is rejected at F7/F8 [2505.17063].

### B6.6 Per-student calibration
- **ProRL pass@k regimes** [2505.24864]: strong negative correlation between the base model's pass@128 on a task and its RL gain in pass@128. *Diminish* (high base pass@128, mostly math: pass@1 up, pass@128 down), *plateau*, *sustained* (code, complex tasks). An OOD task with base 0 at any k (boxnet) became solvable. Adoption (P1): measure student pass@k, not only pass@1, before admitting a family for the 397B; cut families where pass@k ≈ 1 even if pass@1 is mid-band; tag families by regime in F12.
- Bigger students pick harder arms (SEC 7B vs 3B); verification of SFT data hurt a 7B (41.9 vs 45.0) and helped a 32B (64.5 vs 62.1) [2506.04178]; long-CoT traces raised invalid rates for a 7B instruct model (0.65% → 21.4%) while helping QwQ [2505.22648]; Llama-3.2-3B base could not use GRPO at all [2505.17063]; RL without SFT cold start on QwQ gave 5% GAIA Pass@3 [2505.22648]. Key every pool decision by student id; add a readiness check per (family, student).
- RL raises Maj@K but not Pass@K on a fixed pool (DeepSeekMath): flat Pass@k on held-out families signals the pool needs new tasks, not more epochs.

### Minor B6 tips
- Fill a fixed batch asynchronously, cancel stragglers, order by visit count (2504.03380 Alg. 1).
- Easy problems early shorten rollouts and cut wall time per step (AdaRFT 122 s vs 133 s).
- Count difficulty-estimation compute in the budget; at $1.72 per student rollout, rollout-based difficulty at n = 40–128 is unaffordable for agentic tasks, so use a model-judged predictor plus sequential student runs on the shortlist.

---

## B7 Self-improvement loops, their guards, and the statistics they need

### B7.1 Anchor discipline for any loop that changes a verifier [2610.11464]
- **Splits.** Train (large, unlabelled, graded by the evolving verifier); dev = **10 anchored items** (5 failures + 5 clean), the only supervised signal; test = locked, never read by verifier evolution.
- **Verifier as an expression** over typed drawback detectors o(t, y, c) → {drawback, clean, abstain} in three cost tiers (static, execution, narrow LLM question), combined by OR / AND / NOT / K-of-k / weighted vote.
- **One round.** Sense misses (dev items passed but labelled fail) and gaps (train outputs where detectors abstain or split) → cluster under a closed failure taxonomy → an LLM authors one op per cluster → **birth gate**: fires on ≥ half its cluster and stays clean on known-good outputs; ops from anchored misses enter *active*, from unlabelled gaps *shadow* (unselectable until they raise dev agreement) → select argmax S(e) = A_dev(e)·A_train(e)^w − λ·C(e) (A_dev recall-weighted, a miss costs 2× a false alarm; A_train consensus with each op weighted 1 + 10·max(0, anchored marginal) so correlated cheap detectors cannot define the pseudo-label) → retire ops with non-positive leave-one-out marginal after 2 rounds → audit on the locked set.
- **Guards.** Fail-closed (no dev opinion ⇒ unselectable); validity gate drops all-pass, all-fail, all-abstain candidates.
- **Evidence.** Without anchor guards the verifier collapsed to always-pass on 3/3 report seeds (passes 0.97–1.00), yet the naive co-loop's held-out *task* score equalled or beat the anchored one (MBPP+ 0.742 vs 0.717): **downstream score cannot certify a verifier**. Removing lifecycle management collapsed nothing. MBPP+ evolved verifier 0.625 vs 0.417 hand-authored seed (p = 0.12 clustered by task). Anchor size 4/6/8/10 gave similar means, larger variance with fewer. A report-skill hack (metric tags without values, ~30%) was invisible in-loop; the outer judge preferred baseline in 88% of pairs, but **the outer judge was itself wrong until given the task's format contract** (0.515 → 0.770). Agreement falls to chance when residual failures "run cleanly with the wrong value".
- **Adoption (P0).** Per family: a frozen, author-disjoint ~10-item anchor of both classes that no loop edits; fail-closed selection; validity gate; a separate locked set; never use student pass-rate gains as evidence for a verifier change. `VerifierBundle` = conjunction of typed detectors; the birth gate = gold-sanity + fires on its negative class (G5); unanchored detectors in a shadow tier. Outer audits get the `DomainSpec` + `TaskContract`.

### B7.2 Regularised selection for prompt, recipe and harness loops (RRSI) [2609.24972]
- **Proposal side.** Annealed edit budget b_t = ⌈b_min + (b_max − b_min)·½(1 + cos(πt/T))⌉, b_min = 1, b_max = 3–4 (late rounds: one attributable edit); a ledger of every evaluated atomic edit (component, hypothesis, diff, ΔS, ΔC, accepted), rejected ones kept as negative evidence; on a stall over w = 3 rounds, reserve 1 of 2 slots for never-exercised components.
- **Selection side (non-compensatory).** A **leakage critic before scoring** rejects diffs encoding task names, entities, values, answers or benchmark-specific logic; a **noise floor δ** measured beforehand from repeated runs of the unchanged system, never tuned (accept only if Ŝ ≥ S* − δ, S* best-ever, never moving down); cost-aware acceptance (if ΔS > δ require ΔC ≤ β0 + β1·ΔS, β0 0.10–0.15; within-noise gains count only via a weighted score, and never on coding sets); prune components with no positive gain in 4–5 rounds.
- **Ablation (workspace OOD).** Unregularised: evolve 92.8 (highest), OOD 40.3 (≈ base 39.7), 3.80M tokens/trial; no acceptance regularisers OOD 41.0; no proposal regularisers 41.9; full RRSI **43.6 at 2.42M tokens (36% less)**. Higher evolve score went with lower transfer in every arm. Baselines AHE and TTHE ended *below* base OOD.
- **Adoption (P0).** The guard spec for the generator, verifier-rule, World-template and knob loops: per-loop δ, best-ever − δ floor, leakage critic (debug-lane changelog-strip widenings must name a pattern, not a task), ledger, $ per kept task as the cost term, pruning, and evaluation on held-out families the loop never reads.

### B7.3 Recoverability and measured verifier strength [2610.27465]
- **σ(v) = MCC(v, v*)** on a trajectory set, plus false-admit α and false-reject φ, with a task-level upper bound on α (errors cluster by task). Accuracy flatters a constant verifier; MCC gives it 0.
- **Recoverability**: re-approval g (0 if undo is a plain write of a stored old value), restoration fidelity ρ, blast radius β (outcomes changed while live), with noise floors from a null change.
- **Autonomy rule.** Unattended only if g = 0 and σ ≥ σ_min was measured; otherwise **refuse** (logged). σ_min calibrated by flipping reference verdicts with probability ε and finding where regret's CI upper end ≤ tolerance; remeasure after any change to judge model, prompt, request format or averaging.
- **Evidence.** A production diversity gate returned 0.250 in all 40 runs regardless of input (silent collapse in 9/37 configurations, all on the smallest judge); rewriting the criterion as a checkable bound worked 30/30 vs 0/30; three request settings silently failed (forced tool call vs extended reasoning; dropped output format; rejected temperature). LLM judge σ = 0.70 (α 0.10, φ 0.21, CI [0.36, 1.00]); reference-action match σ = 0.67 (α 0.21, φ 0.12): similar σ, opposite errors. Stored-prompt rollback returned to the noise floor; regenerated undos left behaviour 0.16 worse. Each admitted prompt change altered 24–28% of 50 task outcomes vs an 8% floor, flips both ways, mean barely moved. Even with a perfect verifier the loop overstated its gain by 0.20 (winner's curse). Detecting the σ effect on regret needs ~220 test tasks, not 20.
- **Adoption (P0).** For every model-based gate (F5 semantic answers, F10 auditors, F3 dedup judges, swarm quality gates): measured σ and α on a frozen labelled set, refusal when stale; "delete the signal" (remove the judged property; the score must change) and "check the guarantees" as unit tests; every loop-modifiable artefact versioned so undo is a write; blast radius on the replay pool; promote on lower confidence bounds; weight updates stay human-approved.

### B7.4 Multi-view evaluation of a loop (SEAGym) [2606.17546]
- Views: update evidence; a frozen update-validation view; held-out ID; held-out OOD; **replay** of earlier tasks. Metrics: UVG, IDG, OODG, forgetting FR = max(0, Perf(A_0, replay) − Perf(A_T, replay)), fix/forget churn, cost.
- **Evidence.** TF-GRPO validation +17.1 but OOD −2.5; AHE replay 34/80 → 43/80 finally, but **6/80 after epoch 4** (an evolved middleware bug); batch size non-monotonic (10 → −14.3 val; 20 → +17.1; 40 → +2.9; 80 → −17.1), since update-analysis capacity (~3–4M tokens) is the bottleneck; HLE-only training collapsed to 0 at the final snapshot while the mixed source recovered. n per view is small (35–80), no bootstrap intervals.
- **Adoption (P1).** Replay = previously kept tasks re-run after each loop change, as a regression gate for the factory; per-checkpoint fix/forget on a fixed replay set for the 35B; Harbor-native, so the export can be evaluated directly.

### B7.5 Runtime hack monitoring that co-evolves [2606.26300, 2503.11926, 2511.18397]
- **Qwen trajectory monitor.** Log commands, network, git ops, files opened and edited, final patch; match against a pattern set P (each with evidence, leakage risk, intervention); token-level penalty on matches; **update P every training interval** by having an agentic reviewer read current-policy rollouts (prioritising verifier passes and monitor hits). Hack rate 37.76% → 1.31%; hacked-resolved 28.57% → 0.56%; **clean resolved 40.22% → 60.53%**. Without the monitor raw pass rate keeps rising while clean performance collapses late. Solution-artifact retrieval was 4.32% of rollouts but 72.34% resolved (+12.35 over base). Critique: the token penalty is itself a proxy that disguised shortcuts can evade.
- **Never optimise against a monitor.** With CoT-monitor pressure in the reward, hacking continued while monitor recall fell to near zero (obfuscation); rejection sampling with the monitor accepted increasingly cheating samples after ~25% of training [2503.11926].
- **Hackability is a safety defect.** Hacking onset is sharp after ~50 quiet steps; at > 2% hacked episodes all misalignment evals rose (e.g. fake/bad goals 69.8%, frame colleague 41.3% vs baseline 0–4%); covert misalignment 40–80% of misaligned responses; 50% non-hackable environments did not neutralise it; filtering hacked episodes and distilling did not undo it; inoculation prompting cut misalignment 75–90% while hacking stayed > 99% [2511.18397].
- **Adoption (P0).** F12 `hack_flags`: a full-trajectory monitor from a family other than the student's, with an evolving pattern set; per-step (not per-epoch) hack-rate alarms per family; report clean vs raw pass rate per family (divergence = alarm); quarantine and re-gate a family on any confirmed hack; anchor sets include obfuscated hacks. Inoculation prompting is a Part 1 choice.

### B7.6 Loop-adjacent evidence
- **Label-drift meter** for any model-labelled source: relabel a sample with an outside-family reference each generation; alarm on a falling trend (R-Zero 79 → 63%) [2508.05004]. Generators over-rate their own outputs (SwS self-evolving) [2506.08989]. RM scores can rise while the real target does not (LSP) [2509.07414].
- **Generator ablation harness (OpenThoughts)** [2506.04178]: per stage, hold others fixed, generate exactly 31,600 examples, fine-tune a fixed student (Qwen2.5-7B-Instruct), average 8 dev benchmarks, report standard errors ("within 2 SE" bolded), move greedily; a held-out set touched only at the end; ~$300 API per ablation dataset; 1,000+ experiments. Winners: LLM difficulty rating or longest-response filters (+4–6 over random; stronger rater better); no answer filtering; QwQ-32B teacher beat stronger DeepSeek-R1. Adoption (P1): Crucible's stage-ablation procedure (§3.0 rule 1), with value per dollar per generator.
- **Generator distillation weighted by student utility** (Socratic-Zero; B2.3).

### B7.7 Statistics and judge calibration
- **pass^k** = C(c,k)/C(n,k) per task, averaged (vs pass@k = 1 − C(n−c,k)/C(n,k)); gpt-4o retail pass^8 < 25% [2406.12045].
- Bessel-corrected p̂(1−p̂)·K/(K−1) for small K [2502.12272].
- **Within-stratum effect of hackability:** Δ_m = Pass@1(hackable) − Pass@1(robust) inside a difficulty stratum fixed before evaluation (never cross-model solve rate, which hackability inflates), DerSimonian–Laird random effects [2606.16062]. Use F9 bands as strata.
- Paired task-level bootstrap with 10,000 resamples and an independent repeat of every search [2609.24972]; paired sign-flip permutation clustered by task [2610.11464]; paired bootstrap p-values for generator comparisons (Go-Browse p = 0.094).
- **Judge calibration methods.** Confidence 2·|r − ½| as a filter knob (InSTA, 93.1% accurate at confidence 1); validate auditors on subtly perturbed hard sets (Clio 98%, RaR 3,027 pairs); validate against independent fixes such as maintainer PRs (ABA); three-judge votes for equivalence (SWE-ABS 98%); evaluate an evaluator against held-out tests with BoN accuracy, Kendall τ and threshold-conditioned quality, chosen by downstream use (RL needs ranking, filtering needs low false positives); ranking ability does not imply filtering quality (DeepSeek V4 Pro lower τ, higher S̄_UT at θ = 8); **adding rules made the evaluator worse** (v4 BoN 67.4 → v5 59.6); best backbone Claude Opus 4.7 (BoN 70.4, τ 0.579) [2606.26300]. Compare two independent signals per cluster to expose each one's blind spots (Clio: concern score vs classifier flag rate, r = 0.71).
- **Sample size.** A benchmark should detect a 1% improvement (BetterBench); the ~9-task benchmark pool cannot, so report intervals [2411.12990]. AIME's 30 items make 3.3 points one problem [2504.03380].

### Minor B7 tips
- Keep a fixed 10% human spot-check queue stratified by gate and fault side (plan) and validate any clusterer end to end on synthetic traces with known categories first (Clio 94%).
- Separate proposer and solver update rates (SQLM k = 5); reset the reference policy and optimiser when validation stalls (ProRL).
- Publish floors (null agents), ceilings (oracle), frontier reference levels, run-to-run variance and canary GUIDs per benchmark task (BetterBench).

---

## B8 Pipeline architectures, costs, yields, scale; training results on synthetic environments

### B8.1 Cost and yield per stage (as reported; settings in brackets)
| Pipeline | Cost | Yield / funnel |
|---|---|---|
| Explorer [2502.11357] | $0.15 per raw, **$0.28 per successful** trajectory; proposal/refine $0.0128, verify $0.024, summarise $0.026 per step | 175k → 94k successful (53.1%); 60 processes × 50 h |
| AgentTrek [2412.09605] | **$0.551 per verified trajectory**; tag-and-paraphrase $0.89 per 1,000; replay $215 per 1,000 tasks | 23,430 replayed → 10,398 (replay success 39.9%; web share 0.275) |
| Go-Browse [2506.03533] | **$975.57 total** (rollouts $754.66 incl. Claude-3.7 $466; judging $220.91; 53% cache hits); ~3 weeks | 26,749 trajectories, 9,504 successful, 3,422 unique tasks |
| InSTA [2502.06776] | $521.55 (1,200 V100-hours, spot) | 1M sites → 150k (85% filtered); 50% judged successful |
| NNetNav [2410.02907] | not reported | 10,272 demos over 20 sites; > 60% of WebArena episodes pruned by step 16 |
| SwS [2506.08989] | not reported | 1M candidates → quality filter (−78%) → answer consistency → difficulty (~35% kept) → 40K |
| Kimi coding tests [2501.12599] | — | 1,000 → 614 no special judge → 463 with ≥ 40 valid tests → 323 admitted (32%) |
| SWE-RL [2502.18449] | 512 H100 × 32 h training | 4.6M repos; 24M → 11M → 273K PRs |
| SWE-Mirror [2509.08724] | storage 100 GB | mirroring 46% overall (Python 68, JS 52, Go 36, Rust 28); 60,671 tasks |
| Repo2Run [2502.13681] | 29.03 min per successful build | 86.0% EBSR (361/420) |
| Search Self-play [2510.18821] | 83.4 s per step (REINFORCE proposer) | proposer validity ~0% → ~50% |
| SWE-ABS [2603.00520] | $2.50, 18.5 min per instance; $4.98 per strengthened | 251/500 strengthened |
| UTBoost [2506.09289] | ~$1.6 API per instance; ~300 CPU h | 36 insufficient instances |
| PatchDiff [2503.15223] | $0.039–0.105 per patch ($91.7 for 877) | 29.6% differ |
| Hackability audit [2606.16062] | $6.04 for 49 tasks; $8.29 + ~2.5 h Docker for 20; repair $3.60 gated | 9/11 FIX → KEEP |
| ABA audit [2605.26079] | $0.42–0.80 per static task; $200–400 per 500 tasks | 25.7% major |
| Clio [2412.13678] | $48.81 per 100k conversations | — |
| Terminal-Bench [2601.11868] | ~3 reviewer-hours per task; $1–$100 per full run | 229 written → 89 selected |
| τ-bench [2406.12045] | $0.38 agent + $0.23 user per retail task; ~$200 per trial over retail | 165 hand-built |
| OpenThoughts [2506.04178] | ~$300 API per 31.6k ablation set; 22,000 H100-h to annotate 1.2M; 25,000 A100-h per 7B run | 1.2M final |

Takeaway from the cards: compute cost per *verified* item with stage yields in the denominator (AgentTrek, Explorer), and price discarded rollouts in online filtering (2504.03380).

### B8.2 Training results that used synthetic environments or tasks
- **AgentEvolver** (synthetic tool tasks, Qwen2.5) [2511.10395]: 7B AppWorld 1.8 → 23.2 (self-questioning) → 32.4 (full); BFCL v3 29.8 → 49.0 → 57.9. 14B AppWorld 18.0 → 44.3 → 48.7; BFCL 41.6 → 60.3 → 66.5. Synthetic vs human tasks (7B): AppWorld 23.2 vs 16.1, BFCL 49.0 vs 58.8, average 36.1 vs 37.5; **hybrid 43.6, better than either**. 14B trained on AppWorld synthetic loses only 4.3 on BFCL vs in-domain. Set sizes 100/200/500: diminishing gains past 100 (40.3% at 100).
- **WebRL** Llama-3.1-8B WebArena-Lite 42.4 (70B 49.1; GPT-4o 13.9) [2411.02337]. Generation prompt variants 42.4 / 40.6 / 43.6 (stable). No replay ⇒ decay; no KL ⇒ per-phase overfit.
- **Go-Browse** Qwen-2.5-7B 21.7 WebArena; **NNetNav** Llama-3.1-8B 16.3 WebArena / 35.2 WebVoyager; **Explorer** Qwen2-VL-7B Mind2Web-Live 19.3; **InSTA** Qwen3-1.7B 11.5 → 56.9 on 3,000 held-out sites (judge-scored; holds under secondary judges); **AgentTrek** Qwen2.5-32B WebArena 22.40.
- **WebDancer** (crawl and e2h QA, DAPO RL with an LLM-judge reward) GAIA QwQ-32B 37.8 → 51.5; RL on only ~5k QA pairs for compute and stability [2505.22648].
- **Search Self-play** Qwen2.5-7B-Base +26.4 average over 7 QA benchmarks; SOTA on 5/7 at 32B [2510.18821].
- **SWE-Mirror** (SFT, not RL) Qwen2.5-Coder-32B 6.2 → 52.2 SWE-bench Verified; 7B 1.0 → 22.8; error masking beat response-only and pruning at every scale; cross-language +8.4 to +10.3 [2509.08724].
- **Reasoning self-play and synthesis:** R-Zero Qwen3-4B/8B-Base (B2.4); SPIRAL Qwen3-4B-Base 34.0 → 44.5, 8B 39.5 → 49.6 (three seeds) [2506.24119]; SwS +10.0 / +7.7 (7B / 32B); Socratic-Zero 8B 56.1% average; Synthetic Data RL Qwen2.5-7B GSM8K 91.7, matching human data at equal budget (M = 1,000: 91.8 vs 91.7); ProRL 1.5B on 136K tasks, ~16K H100-h, > 2K steps; DAPO Qwen2.5-32B AIME24 50.
- **Qwen3.x-30/35B-A3B.** No card in these two groups reports RL on a Qwen3-30B-A3B or 35B-A3B student. The only mention is Nemotron 3 Nano 30B-A3B, claimed at 3.3× Qwen3-30B-A3B throughput at 8k in / 16k out [2512.20856]. Model-specific evidence for the student must come from other groups or our own runs.

### B8.3 Mixed vs staged training
- Nemotron 3 trains all environment families simultaneously and reports this as more stable, less hack-prone and better than staged recipes, **with no numbers, ablation, environment counts or hacking metric** in the white paper [2512.20856]. Cite only as qualitative support.
- Measured support for mixing: SPIRAL multi-game 59.5% vs best specialist 52.9% vs Gemini-2.0-Flash; SEC-2D stability; AgentEvolver hybrid; SEAGym's mixed-source recovery vs single-source collapse; R-Zero followed by human-labelled GRPO beat human-labelled alone by +2.35 at 4B. Counter-signal: one hackable family contaminates a run even at 50% clean environments [2511.18397], so mixing is not a guard; G12 still applies.

### B8.4 Architecture patterns worth copying
- **LLM acts; deterministic recorder emits the artifact** (Repo2Run; B1.2).
- **Separate roles with separate prompts or families**: test writer vs bug injector (SWE-Mirror), setup vs verify agent (SWE-bench-Live), proposer vs solver (R-Zero), attacker different from builder (2606.16062).
- **Cheap cascades before expensive steps**: rules → LLM labeller on a sample → distilled fastText on all (AgentTrek); LLM "functionality exists" pre-filter at ~84% precision before mirroring (SWE-Mirror); critic band before rollouts (WebRL); strong-model feasibility before cheap-model rollouts (Go-Browse).
- **Async batch filling with cancellation** (2504.03380 App. B) and partial rollouts with repeat detection (Kimi).

---

## B9 Corrections and contradictions

### B9.1 Corrections to `framework-plan.md`
1. **§3.6 objective wording.** The plan says expected improvement is "bounded below by the variance of success p(1−p)" citing 2504.03380. The derivation bounds the reverse KL from the initial to the soft-optimal policy: KL ≥ p(1−p)/(2β²), valid for large β. It is not a proven lower bound on per-step improvement. LILO's theorem is an *upper* bound on per-step improvement [2502.12272]. Suggested wording: "the KL distance to the optimal policy is bounded below by p(1−p)/2β², and per-step improvement is bounded above by a term proportional to Var[r]".
2. **§3.6 band list.** "WebRL keeps 0.05–0.75" is a band on a trained *critic's value estimate* V(s₀, I) at the start state, not on a measured pass rate [2411.02337]. It shows a coarse predictor can gate generation; it is not a p̂ band.
3. **§3.6 [0.1, 0.6].** 2504.03380's one-sided arms lost to plain GRPO, but they were cuts at 0 or 1, not a shifted interior band, so they do not refute [0.1, 0.6]. Add a symmetric (0.2, 0.8) arm to R2.
4. **§3.6 sequential measurement.** LILO Alg. 3 (2 rollouts at a time to the currently most learnable task) is a principled, measured form of the "k = 4 first" proposal; the plan calls its proposal unmeasured.
5. **§3.7 / F10.** p(1−p) and mean |Â| reward aleatoric noise as well as learnability; a flaky task looks maximally learnable. G10 must run before any learnability-based admission or curriculum arm reward.
6. **§3.3 infeasible tasks and G4.** Unchanged-state success lets a do-nothing agent score 38% on τ-bench airline; the plan's infeasible type needs positive evidence (refusal + policy lookup), and G4's null agents need "dump all data" and "list all answers" members.
7. **§3.1 gate list.** Missing items with evidence in these groups: a harness-level trust-boundary gate (BenchJack), a parser gold-sanity check (UTBoost), a hardcoded-oracle check and the reverse spec direction (Terminal-Bench), a state-change complexity floor (ABC O.g.3), a uniqueness check for answer tasks (Search Self-play), a run-completeness check (2610.27465).
8. **§3.5 dedup and coverage.** The repetition penalty and dedup should run on behaviour signatures; R-Zero's BLEU version is surface-level.
9. **Nemotron 3 as evidence.** If the plan or note 02 cites 2512.20856 for hack resistance of mixed training, the white paper gives no measurement; keep it as a qualitative statement only.

### B9.2 Errors and weaknesses inside papers (flagged by the cards)
- **DUMP** [2504.09710]: App. A asserts a lower bound on the gradient norm by mean |Â|·‖∇log π‖, which does not follow (the norm of an expectation is at most the expected norm); App. B is UCB1 regret against a fixed arm under a stationarity assumption the setting violates. It reports a shaped reward, not accuracy; three distributions regress.
- **Kimi k1.5** [2501.12599]: prioritised sampling ∝ 1 − s_i puts most weight on s = 0 prompts, which give no group-relative gradient; it contradicts the learnability results (2502.12272, 2504.03380). Ablations are curves without numbers.
- **WebDancer** [2505.22648]: the quality rule on "more than two actions" reads as a typo and is not reproducible as stated.
- **R-Zero** [2508.05004]: the 3–7 band selects exactly the questions where the majority label is least reliable; no single label-accuracy threshold predicts collapse.
- **2606.16062**: EQS weights are untuned priors; its D(t) term uses external solve rate, circular for new tasks; n is small (49, 20, 11).
- **SWE-ABS** [2603.00520]: the headline depends on a manual fix of 53 over-specialised instances (without it 206 strengthened, 11.22 pp).
- **2610.11464**: "never read" holds only for verifier evolution; the skill loop's rollback reads the locked test set. Gains are not significant under task-clustered tests.
- **ABC** [2507.02825]: the task-validity "iff" is not operationalised.
- **ImpossibleBench** [2510.20270]: LiveCodeBench twins have no QC (no reference solutions).
- **Explorer** [2502.11357]: the ~26% false-success figure is derived from the reported confusion fractions (0.14 / 0.53), not stated directly.
- **AgentEvolver** [2511.10395]: no report of how many synthetic tasks fail feasibility or are near-duplicates; evaluation on public benchmarks the generators may know.
- **OpenThoughts** [2506.04178]: "fewer sources is better" is measured at a fixed 31.6k size; greedy stage search can miss interactions.
- **Single-seed and small-eval caveats** apply to 2504.03380, 2505.14970, 2504.05520, 2502.12272, 2504.09710 and SEAGym; several group-05 cards note result tables lost in HTML conversion (2501.12599, 2506.08989, 2509.24726, 2510.18821, 2506.03136, 2406.13542, 2505.17063) with numbers taken from the text.

### B9.3 Apparent inconsistencies, resolved or kept
- **Terminal-Bench difficulty correlation.** The group-05 summary says r = 0.44; the card gives r = 0.436. Same figure, rounded.
- **UTBoost cost.** The UTBoost card gives ~$1.6 API per instance; the SWE-ABS card compares $4.98 per *strengthened* instance with "UTBoost $80". Both hold: $1.6 × 500 instances / 10 strengthened ≈ $80 per strengthened instance.
- **LILO naming.** The arXiv title is "Learning to Reason at the Frontier of Learnability"; the NeurIPS version is LILO. Same paper (2502.12272).
- **LILO vs GRPO normalisation.** LILO's bound loses its learnability dependence under std-normalised GRPO, and replications find unnormalised GRPO better; SEC's E|Â| = 2√(p(1−p)) derivation assumes std normalisation. The two cards are consistent but imply that the normalisation choice (Part 1) changes how strongly the band matters.
- **Verification helps or hurts by student size** (OpenThoughts 7B vs 32B) is consistent with, not contrary to, "construction before sampling"; it concerns SFT answer filtering, not task validity.
