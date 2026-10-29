# Group 01 cards: environment and task synthesis, tool-use data, SWE environments, gyms

**Papers: 29.** Read FULL: 28. PARTIAL: 1 (2507.20534 Kimi K2: post-training, agentic data and RL read in full; pre-training and architecture read only enough to summarise). ABSTRACT-ONLY: 0. Where an arXiv HTML had lost appendices, the local or fetched PDF was read for those pages (DreamGym App. C, Simia Figs. 9–14, Synatra App. G+). Agent-World's case-study appendix is cut off in the HTML.

**Five cross-paper takeaways for Part 2**

1. **Many Worlds beat many tasks per World, but the ideal mix depends on scale and student.** ScaleEnv (more domains at a fixed 1,024 tasks), AWM (10 envs worse than base, then 100 → 526 better), Agent-World (diminishing after about 500 envs) and EnvScaler (steepest from 0 to 20 envs) all point one way. SynthAPI adds the nuance: at small budgets fewer Apps win, and at our 500–1,000 scale they partly still do. Several start states per instruction and dense distractors raise the ceiling, and its largest student gains least. So F0/F11 should treat Apps (A), tasks per App (T), seeds per task (S) and distractor density (D) as explicit budget knobs, and measure the allocation per student instead of assuming it.
2. **Every paper's verifier is weaker than its World, and none has negatives.** The best designs pass a two-sided filter, oracle passes and start state fails (SynthAPI, τ²'s init/solution/assertion triples), or use exact canonical state diff (LOGIGEN). The rest use partial-credit checklists (EnvScaler), LLM judges (AWM, Agent-World rubrics, Kimi K2) or single-target equality (AutoForge, ScaleEnv). No paper generates negatives, checks alternative solutions, or audits verifier correctness. R2E-Gym's test metrics are the exception (distinguishing rate; toxic tests reaching up to 10%). AutoEnv's own appendix shows a "verified" reward scoring 139% of maximum. Crucible's G5/G6/G12/G14, a collateral diff and an answer check for read-only intents are what this literature lacks. Build them before scaling.
3. **Prove solvability by construction; sampling is not proof.** AWM has no oracle, and 23–28% of its training tasks were never solved. Agent-World keeps a task if 2 of 5 ReAct runs pass, and EnvScaler never proves feasibility. Pipelines that execute an oracle (SynthAPI, ScaleEnv code chains, LOGIGEN exploration, τ², APIGen-MT blueprints, SWE-smith) keep tasks clean. The oracle must also use only agent-visible tools and information. SynthAPI's oracle reads privileged IDs, and only LOGIGEN projects the instruction from the executed episode, which is G7 + G8.
4. **The simulated user is a World component with its own fault side and hack surface.** LOGIGEN: a low-temperature user simulator gave higher training reward and lower transfer. AutoForge: switching the eval user from GPT-4.1 to GPT-5 moved τ² Telecom from 76.3 to 90.4. GEM: a strong user plus a weak agent beat the reverse. VitaBench: 9.2% of failures came from the simulator. The fixes are known: give the user tools and state (τ² dual control); Best-of-N with a critic (APIGen-MT); strict disclosure rules; mask user-caused failures without penalising them (AutoForge MEU). F9/F10 must key p̂ on the simulator, cross-check kept tasks with a different-family simulator, and extend G8 to the user's known info.
5. **Hardness must come from structure that changes behaviour, and loops need guards the papers skip.** Structural knobs with evidence include boundary-adjacent seeding (LOGIGEN), DAG merges with reasoning nodes (AutoForge), weak or independent edges and hidden tool names (Agent-World), subtask count (τ²: pass^1 near 0 beyond 7 actions), depth and width extension with a no-tool check (TaskCraft = G9), and World-side hints switched off late (Environment Tuning). SWE-smith's harder tiers gave no SFT gain, and Reasoning Gym's own "hard" knobs sometimes ran backwards, so measure monotonicity. Agent-World's diagnose → targeted-synthesis loop is the F12 → F3 template. But it runs with no untargeted control, no verifier audit, and the diagnoser sees validator internals. Adopt it only behind §3.8's anchors, an equal-budget random baseline, and gate code hidden from the swarm.

---

## 2602.06820 — ScaleEnv: Scaling Environment Synthesis from Scratch for Generalist Interactive Tool-Use Agent Training (Tu et al., 2026; arXiv preprint, Nanjing Univ. + Meituan)
Read: FULL (body, appendices A–E; code listings in D skimmed).  Relevance to Part 2: high. It is the closest published template for F1 (App synthesis with blind tool tests), F2 (seeding with distractors), F4 (oracle-as-code first) and state-based grading with column policies, and it gives the domain-count-vs-task-count result that motivates forced diversity.

**Problem.** Agentic RL needs many interactive, stateful, verifiable environments; real APIs are few and unsafe, LLM-simulated tools hallucinate state, and earlier synthesis pipelines (AutoForge, EnvScaler) either depend on documents or produce tasks that are inconsistent with the environment state.

**Method / algorithm.** Two phases, all driven by a mix of DeepSeek-V3.2, GLM-4.7, GPT-5.1 and Qwen3-32B in different agent roles.
- Formalism: domain foundation B = ⟨Σ (DB schemas), T (tools)⟩; environment E = ⟨s0, T⟩; task ψ = ⟨E, hidden user goal u, user profile P_user⟩; POMDP whose state is (env state, history, u); tool actions change env deterministically, text actions only history; reward r = R(s_T, u).
- **Phase 1, executable graph.**
  1. *Top-down tool schema*: from a domain keyword ("Job Seeking"), an LLM writes ~50 atomic tool schemas with descriptions, parameters and pre/post-conditions (submit_application needs upload_resume).
  2. *DB schema by reverse engineering*: a Database Agent infers the tables each tool implies, dedups, and writes a tool→table mapping plus integrity constraints (5–22 tables per domain).
  3. *Reward specification*: every DB column is assigned one of three matching policies: exempt (generated IDs, optional fields), hard (exact string/number: prices, quantities, timestamps), semantic (fuzzy match for free text).
  4. *DB code first*: LLM writes the DB code (pydantic models in a thread-safe base) plus integrity test scripts; a debugger agent iterates on tracebacks until all pass.
  5. *Procedural testing for tools*: a Code Agent writes the tool; a separate Test Agent writes unit tests together with matched DB instances. Each test outcome is classified Success (state transition equals the expected DB state), Anticipated Rejection (the tool raises the pre-declared exception on bad input) or Unexpected Failure (anything else) → Debug Agent edits the tool *or the DB instance* until the test passes.
  6. *Tool dependency graph*: a Tool Dependency Agent judges every pair of verified tools on data flow (argument passing), pre/post-conditions and shared tables, adding directed edges.
- **Phase 2, task instantiation by graph expansion.**
  1. *Seed chain*: an LLM, given the graph and schema, writes an executable *code* tool chain C1 = (a1..ak) with arguments; outputs flow into inputs programmatically.
  2. *Initial state*: an LLM synthesises s0 satisfying C1 with entity consistency (FKs resolve); C1 is executed on s0 to prove feasibility; distractor rows (schema-valid, orthogonal to the gold path) are injected with density scaled by target complexity.
  3. *Instruction*: an LLM writes P_user and u grounded only in C1; the reward target s_gt is the state after executing C1.
  4. *Dependency-aware BFS*: grow H1 = K(C1) by adding any tool whose inputs and outputs are satisfiable from tools already in H1; execute each added tool with arguments derived from H1 and patch the env when it errors (goal: interaction completeness, every valid call returns a meaningful observation).
  5. *LLM-gated chain expansion*: with D_n = G \ H_n the unused tools, a strong LLM outputs a score p ∈ [0,1] from (|D_n|, structural complexity c(H_n) = (|V| + 0.5|E|)/50, feasibility g(D_n) = success rate of an oracle agent, Qwen3-235B-A22B best-of-16, at finding executable chains in D_n). If p ≥ τ (τ not reported) sample C_{n+1} from D_n, expand, execute, patch, merge. Enforce |H_n| ≥ 20 tools by merging random valid auxiliary chains.
- **Training**: GRPO, step-wise ratio, group-normalised outcome reward; Qwen2.5-72B-Instruct as user simulator; Qwen3-8B batch 1024, Qwen3-32B batch 2048, 48 steps, lr 1e-6; "Zero RL" (no SFT).

**Verification / grading.** Rule-based comparison of final DB state with s_gt under the per-column policies. Hackable or weak points: (a) read-only goals produce s_T = s0 = s_gt, so a no-op passes (no answer check); (b) only one gold path defines s_gt, so a valid alternative that writes a different but acceptable state fails; (c) no collateral check is described beyond equality, so extra writes in exempt columns are invisible; (d) the user simulator is a 72B LLM that may leak or garble u; (e) exact-match columns require the user to communicate exact values, which the paper does not gate (communicability). Tests are written by a different agent than the code, but the debugger may edit DB instances to make tests pass, so expected states are not frozen.

**Scale & cost.** 16 domains, ~50 tools each (25–70+), 2,560 tasks (64–512 per domain, unevenly weighted toward wedding planning and knowledge management). ~546K tokens per domain foundation (379K prompt + 167K completion) and ~93.2K tokens per task. No yield per stage, no wall time, no $ reported.

**Key results.** Strict OOD (tool-embedding t-SNE shows test domains separate from train).
- Qwen3-8B → SE-8B: τ²-Retail 38.4→50.9, Airline 30.5→37.5, Telecom 21.5→27.2; VitaBench Cross 1.5→3.0, Delivery 18.3→26.3, Instore 14.8→23.8, OTA 4.5→7.0.
- Qwen3-32B → SE-32B: Retail 59.5→63.6, Airline 48.0→48.0, Telecom 27.2→30.9, Vita Cross 5.3→10.8, Instore 22.5→34.5, OTA 4.5→12.5. VitaBench pass@4 avg 36.0→46.8 (32B).
- *Domain scaling*: N ∈ {2,4,8,16} domains at a fixed 1,024 tasks, pass@4 rises monotonically on both benchmarks and has not plateaued at 16.
- *Execution-verification ablation* (8B, avg@4 τ²): without EV 42.3/30.0/25.2 vs with EV 50.9/37.5/27.2; Airline without EV is below base.
- *Reward ablation* (τ² avg of 3 domains): LLM judge avg@4 36.5, pass@4 58.8 vs rule-based 38.5, 62.9.
- *Domain stability*: two disjoint 4-domain sets give 13.8 and 13.3 Vita avg vs 9.8 base.

**Failure modes & limits.** Authors: generative misuse only. My critique: single run per setting, no variance; 48 RL steps is short; the 32B Airline gain is 0; no per-stage yields, no measure of how many tasks are unsolvable or broken after patching; gating threshold τ unpublished; task counts per domain are uneven, which confounds the domain-scaling curve with domain identity; no policy documents in training, so "format generalisation" is asserted rather than isolated; the paper never shows that the user simulator reveals exact-match values; no dedup or diversity measurement beyond t-SNE.

**Tips & tricks.**
- Write DB code and its tests before tool code; tools are tested against concrete DB instances.
- Classify tool test outcomes three ways; "anticipated rejection" makes error paths first-class tests.
- Express the reference solution as executable code so data flow between calls is enforced.
- Derive the goal state by executing the reference solution, never by asking the LLM for it.
- Per-column match policies (exempt / exact / semantic) make DB-diff rewards workable.
- Inject schema-valid, orthogonal distractor rows with density as a difficulty knob.
- Grow the reachable tool set by dependency-closed BFS and execute every added tool to find env gaps.
- Hold domain count, not task count, as the diversity lever (fixed 1,024 tasks, more domains = better).

**Fit to Crucible Part 2.**
- **F1**: adopt procedural testing as the blind tool tester role, but *freeze* expected state digests at test-write time and forbid the debugger from editing DB instances or expectations without a different-family checker (the plan's `ToolTestCase.expected_state_digest` already says this; ScaleEnv shows why). Adopt the dependency-graph edge kinds (dataflow | precondition | shared_table) for `AppBundle.tool_graph`.
- **F2**: distractor density as a knob in `SeedBundle`; entity-consistency check is G1.
- **F4**: oracle as executable action program; gold end state = executed result; run twice (G2).
- **F5**: column policies map to `AppSpec.column_policy`; add the answer check (read-only goals) and collateral diff that ScaleEnv lacks; G8 communicability for exact columns.
- **G13**: their dependency-aware BFS execution is a concrete way to implement interaction completeness.
- **Diversity / F0**: the domain-scaling result argues for many DomainSpecs/families over many tasks per family; set per-family quotas in F11.
- **Avoid**: LLM-gated expansion with an unpublished threshold as the only complexity control; use explicit knobs instead.
- Priority **P0** (F1/F4/F5 design reference).

## 2610.37648 — Scaling Laws for Tool-Use Agents via Controllable Environment Synthesis (Szot et al., 2026; arXiv preprint, Apple)
Read: FULL (body, appendices A–E including all generation prompts).  Relevance to Part 2: high. It is the only paper in this batch that measures how to *allocate* a synthesis budget between tools (Apps), tasks and start states, and it gives a cheap, reimplementable solver-first-then-verifier task contract.

**Problem.** Executable environment synthesis papers scale quantity; nobody knows whether a fixed budget should go to new tools, new tasks on existing tools, or new database states for existing tasks.

**Method / algorithm.** SynthAPI, with Gemini-3.1 Pro (internet off) as the single builder LLM.
- **App generation** (agentic, inside Gemini CLI): a hand-made taxonomy of 8 categories (shopping, music, productivity, finance, fitness, messaging, utilities, social), balanced. Per app: (1) metadata = name, description, category, exhaustive list of in-app functionalities (prompt excludes cross-app features and pure account management); (2) `db.py`: dataclasses and Enums covering all functionalities; (3) `apis.py`: public functions that may touch state only through four CRUD primitives, so every change is logged in the DB object; functions not needing the current user are marked `no_auth`.
- **Task tuple** T = (x instruction, v verifier, Db0 start state, π* oracle).
  1. Sample an app and F functionalities; prompt for n short (15–40 words), casual, unambiguous, action-only instructions that mention the app, with previously generated tasks in context as "do not duplicate". Difficulty should come from one constraint or decision, not stacked steps. No user in the loop.
  2. Verifier `verify(db, answer=None) -> bool`, written from the instruction and API docs *before* the start state exists; it may read the change history.
  3. Start state: `setup_database()` returns the minimum core rows (Db_core) plus the main user's fixed identity; then D−1 extra distractor stages `setup_additional_stage_k()`, each told what earlier stages added and to differ from them, to use non-colliding IDs, to relate to the main user, and not to pre-solve or block the task.
  4. Oracle `solve(db)` written with privileged access to Db0 but restricted to public APIs (no private helpers, no CRUD primitives, no setattr, no direct db access).
  5. Filter: keep iff v(Db0) = 0 and v(Exec(π*, Db0)) = 1.
- **Controls**: A apps, T tasks per app, S start states per instruction (same x, v, π*; new Db0), D distractor batches. N = A·T·S tasks; D adds complexity without adding tasks.
- **Interface**: Gymnasium, deterministic. Actions are Python code blocks in a persistent REPL (AppWorld style); tools are discovered through meta-tools `list_apps`, `list_tools`, `get_tool_documentation`, and the episode ends only via `complete_task(answer=None)`. Observations truncated at 2,048 tokens; errors do not end the episode; 32-step cap; success only if complete_task is called and v is true.
- **Training for the analysis**: SFT, not RL (RL per setting too costly). GLM-5.2 expert, up to 4 attempts per task, keep the first verified success, one trajectory per task. Loss on reasoning + action tokens only. lr 2e-5, ≤1 epoch, early stop after 5 consecutive validation-loss rises, 4k held-out trajectories for validation. Students Qwen3.5 2B/4B/9B/27B on 8×B200 nodes.
- **Scaling law**: per (model size, A) a generalised logistic P̂(N) = U·σ(b + b_N log2 N)^ν. Joint law P̂(A,T,S,D) = σ(c0 + c_A log2(A·S·D)) · σ(b0 + b_T log2 T)^ν: diversity and complexity set the ceiling, tasks per app set the rate of approach. Fit with SciPy trust-region least squares on all runs except the top 25% of FLOPs (C ≈ 6·M·L supervision tokens), then extrapolate up to 16×.
- **Normalised score** per benchmark: (succ − zero-shot)/(best observed − zero-shot), averaged over AppWorld and a SynthAPI holdout (4 new apps × 25 tasks, max complexity).

**Verification / grading.** Generated Python verifier over end state (+ change log + optional answer). Gates are only the two-sided filter (not pre-solved, oracle solves). Weak points: the verifier is written by the same model as the oracle and the start state, so a shared misunderstanding passes both sides; there are no negatives (wrong-value or side-effect mutants), so a lax verifier is undetected; the filter does not check that the oracle uses only information the agent can discover (it reads privileged IDs); downstream eval counts unintended changes as failure, but the training verifier need not. Read-only questions are excluded by prompt, removing the no-op hole.

**Scale & cost.** 64 apps, 10,472 APIs (6,266 write, 4,206 read; median 144 per app, range 51–443), median 55 dataclasses, median 81 functionalities per app; 458,752 tasks; median core state 59 rows; median 2 functionalities per task. SFT corpus 5.9B tokens (721M supervised). 849 trained models. Diagnostic eval alone needed 1.8M episodes. Expert trajectories: median 8 tool calls (4–18). No $ or yield-per-stage figures reported (filter rejection rate not given).

**Key results.**
- *Apps vs tasks* (9B, S=1, D=2): A=8 best at 64–512 tasks; A=64 clearly best from 4,096 tasks, and its ceiling is higher. Even at T=1,024 tasks per app only a median 51% of an app's tools are used, so the gain is not functionality exhaustion. Same trend on 2B/4B/27B and on another pipeline's data (SimSAT: at N=1,024, A=32 beats A=16; at ≤512 fewer apps win).
- *Start states per instruction*: best S = 1 at 32–512 tasks, 2 at 512–4,096, 4 beyond.
- *Distractors*: D=4 gives a large gain over D=1,2 at high budgets.
- *Law*: extrapolation MAE 0.0285 vs 0.1706 for a task-count-only law.
- *Fixed budget 32,768 tasks*: Opt (A=64, T=128, S=4, D=4) beats SubOpt (A=32, T=1,024, S=1, D=2) everywhere; e.g. AppWorld Test-C 4B 38.4→45.0, holdout 4B 51.3→56.0.
- *Big* (all data, 4 epochs): AppWorld Test-C 66.4/77.0/81.5 (4B/9B/27B) vs zero-shot 14.7/26.4/61.5; beats SimSAT by 7.6/11.6 pp; BFCL multi-turn on Qwen3 4B/8B 50.5/55.6 vs EnvScaler 45.0/51.9, EnvFactory 48.5/49.0, AWM 40.8/42.3. Nemotron-Agentic (LLM-simulated, short) is below zero-shot in 5 of 8 settings.
- 27B gains are small or negative on ASTRA (SubOpt 77.3 < zero-shot 80.2).

**Failure modes & limits.** Authors: no simulated user, no long-document retrieval, single-app tasks only, law validated on one pipeline, SFT not RL; baselines differ in training regime so data efficiency is not isolated. My critique: the law predicts SFT distillation gains; RL with a group-relative advantage depends on per-student pass rate, which the law ignores; the largest student gains least and can lose (ASTRA 27B), the first sign that easy synthetic pools stop teaching bigger students; one builder model makes "diversity" one model's diversity (no homogeneity check); no measure of verifier correctness.

**Tips & tricks.**
- Gate state writes through 4 CRUD primitives so every change is logged and diffable.
- Write the verifier from the instruction before the start state exists, then build state to fit.
- Two-sided filter: v(start)=0 and v(oracle end)=1.
- Ban the oracle from private helpers and raw DB access; it must use agent-visible APIs.
- Add distractors in stages, each told to differ from earlier stages and to target the main user.
- Re-instantiate the same instruction on several start states (S) before writing new instructions, once the budget is large.
- Discovery meta-tools keep context small with 100+ tools per app and teach transfer to new tool sets.
- Fit allocation curves on small runs and extrapolate; hold out the top-FLOP quartile to test.

**Fit to Crucible Part 2.**
- **F0/F11 quotas**: make A (Apps/families), T (tasks per App), S (seeds per task) and D (distractor density) explicit budget knobs; the default allocation at our 500–1000 scale is *low T, several S*. At ~1,000 tasks their curves say few Apps beat many Apps for SFT; we should test this for RL rather than assume diversity always wins at our scale.
- **F2**: S = multiple `SeedBundle`s per `TaskProposal` is the cheapest way to multiply tasks; requires the verifier to be state-generic (no hard-coded IDs). Staged distractor prompts are a ready F2 recipe.
- **F4/F6**: the two-sided filter is G3 + G4; the public-API restriction is G7. Add what they lack: G5 negatives, G8 communicability (their oracle reads privileged IDs), and a verifier written by a different family.
- **F1**: CRUD-gated access log is the cheap route to `StateAdapter.diff` and collateral checks.
- **F9**: the 27B result warns that pool value decays with student size; key DifficultyRecord by student.
- **Measurement (§4.4)**: reuse their protocol (normalised score, fit on low FLOPs, extrapolate) to measure our own generator value per dollar in F12.
- Priority **P0** for the allocation knobs and filter; P1 for the scaling-law methodology.

## 2602.10090 — Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning (Wang et al., 2026; ICML-format preprint, UNC Chapel Hill + Snowflake; v3 May 2026)
Read: FULL (body, appendices A–B, all prompts; code listings skimmed).  Relevance to Part 2: high. Its pipeline order is already vendored in Crucible (`_vendor/openenv_awm`); this card records the details we inherit, their measured defect rates, and what our gates must add.

**Problem.** Agentic RL needs many reliable, resettable environments. Real services are costly and unexposed; LLM-simulated transitions hallucinate and cost an LLM call per step; prior code-based syntheses rely on documentation, task sets or game priors, or are small.

**Method / algorithm.** "Software-engineering order", all with GPT-5, each stage executed and self-corrected.
1. **Scenarios**: Self-Instruct expansion from 100 popular domain names (similarweb top sites) to 1,000 scenarios of 150–300 words; an LLM classifier keeps CRUD-heavy, synthesizable platforms (rejects news, wikis, search, AI inference); embedding dedup at cosine 0.85; category caps (e.g. e-commerce).
2. **Tasks first, as requirements**: k = 10 tasks per scenario, single sentence, self-contained with concrete parameters, API-solvable (no UI), post-authentication (user id = 1). 10,000 tasks.
3. **Database**: SQLite DDL with keys, FKs, indexes inferred from the tasks (tables only when a task needs them; no auth fields). Executed; failures summarised by a separate LLM call and fed back; accept if < 10% of tables fail.
4. **Sample data**: INSERTs that satisfy each task's preconditions plus variation data, executed in FK order; same < 10% failure threshold.
5. **Interface spec then code**: first a toolset schema (minimal endpoints that make every task executable: name, method, path, summary, typed params, response schema), then one Python file with SQLAlchemy models, Pydantic models and FastAPI handlers exposed as MCP tools (≈2,000 lines, ≈35 tools). Validated by server start, health check and list-tools (0% error tolerance).
6. **Verification**: per task, a read-only Python function `verify_task(initial_db_path, final_db_path) -> dict` that queries both DBs and returns task-relevant signals, plus natural-language success and failure criteria. The final reward comes from a **code-augmented LLM judge** (GPT-5) that sees the trajectory and the verifier output and returns Completed | Partially Completed | Agent Error | Environment Error (priority order in that sequence), told that verifier results may be wrong.
7. **Self-correction**: up to 5 retries per component with a 200–500-token error summary; if still above threshold, keep the best attempt.
- **RL**: GRPO via AgentFly + verl; Qwen3 thinking 4B/8B/14B; 526 envs, 3,315 tasks; batch 64 × 16 rollouts = 1,024 isolated MCP servers per step, each on its own SQLite copy restored after each rollout; background prefetch of the next batch's environments; lr 7e-7, ≤96 steps, KL 0.001, clip-high 0.28, sequence-level importance sampling, temp 1.0, max 20 turns.
- **Reward**: R = 1.0 Completed, 0.1 Partially Completed, 0 otherwise; any step-level format violation (missing think block, hallucinated tool name, schema-invalid JSON, list_tools not called exactly once first, no successful real call) ends the rollout with −1; server error → 0 (environment error, not blamed on the agent).
- **History-aware training**: sliding window w = 3 at train time, with each rollout split into T samples (system prompt + first message + list_tools exchange + last 3 turns), loss only on turn t; w = 10 at eval for long tasks.
- Agent interface: two meta-tools, `list_tools` and `call_tool(tool_name, arguments_json)`; format converters map τ², BFCL and MCP-Universe onto them.

**Verification / grading.** State diff by generated SQL + LLM judge. The judge rescues false negatives from imperfect envs (idempotent success with no state delta) and the code evidence prevents false positives (wrong-entity writes). Reliability on 100 trajectories × 5 judgments: GPT-5.1 self-consistency 90.8%, pairwise 95.5%, κ 0.891, reward flip 9.2%; Claude-4.5 flip 18.0%; Qwen3.5-122B flip 23.7%. Hackability: the reward is still a judge opinion, so a fluent trajectory plus a lax verifier can pass; verifier and env are written by the same model; no negatives or mutation tests; "partially completed" reward 0.1 can be farmed; the format reward penalises refusal, which hurts BFCL hallucination (the authors note this).

**Scale & cost.** 1,000 envs, 35,062 tools, 10,000 tasks. GPT-5 synthesis cost per 100 samples: scenario $0.43, task $0.56, DB $3.59, sample data $13.75, toolset schema $23.74, env code $12.81, verification $2.21; total $57.09 per 100 (≈ $0.57 per environment with its 10 tasks). First-attempt success > 85%; DB 88.3%, data 88.2%, code 86.8%; 1.13 correction iterations on average. Judge cost ≈ $1.80 per RL step (≤1,024 samples). Env complexity: 18.5 tables, 129 records, 35 tools, 1,985 lines; Claude-4.5 agent needs 8.5 steps, 7.1 unique tools; 62.6% completion at a 20-step budget, 13.7% exceed it. Env error rate during RL ≈ 4%.

**Key results.**
- BFCLv3 overall: 4B 54.92→64.50, 8B 53.83→65.94, 14B 61.25→70.18; LLM-simulated env baseline (GPT-5 as transition model, same tasks) gives 55.52/52.53/67.68.
- τ²-bench pass@1: 4B 15.83→22.57, 8B 26.44→33.45, 14B 36.69→39.03 (EnvScaler better on τ²: 28.96/39.39, plausibly because its seed tasks overlap τ²). MCP-Universe: 8B 6.70→11.17.
- Gains shrink with task hardness: BFCL simple +26.7, medium +15.3, hard +1.1.
- Verification ablation (8B): LLM-only 55.46 BFCL / 26.44 τ², code-only 60.00 / 29.59, code-augmented 65.94 / 33.45.
- History window aligned at train and test: BFCL 64.50 vs 55.35 full-history; misalignment collapses τ² pass@4 to 15.11.
- Env scaling (4B): 10 envs degrades below base; 100 gives large gains; 526 better still.
- Quality audit on 100 envs (LLM judges): 74–83% of envs have ≥1 bug (2.7–4.1 bugs each; 44% unhandled edge inputs, 14% DB-constraint conflicts); blocked tasks 11.5–14.0% vs EnvScaler 46.8–57.1%. Generator swap: Claude-4.5 99–100% stage success; Qwen3.5-122B 77–97% with more bugs; diversity (mean pairwise distance 0.31–0.35) is generator-independent.
- Difficulty: GPT-5.1 pass@1 36.1%, Sonnet-4.5 62.6%; 69% of 11+-call tasks unsolved by both; 23.7–28.2% of training tasks never solved once by the students.
- Code-level diversity: 0.0% AST-duplicate functions across envs, endpoint-name Jaccard 0.004.

**Failure modes & limits.** Authors: fixed (non-evolving) pipeline; runtime-only self-correction misses logic bugs; single-scenario tasks; no adversarial or access-control training; 74% of envs buggy. My critique: about a quarter of training tasks are never solved, which is wasted compute and possibly broken tasks (no solvability proof: there is no oracle program, so "task feasible" is a judge score); the 10%-error acceptance means some tables and rows silently missing; the judge sees trajectories, so persuasive narration may sway it; the env-count scaling curve has 3 points and one seed; no read-back of whether the student exploited Partially Completed.

**Tips & tricks.**
- Generate tasks before the schema, so the DB and tools exist only to serve tasks.
- Spec the toolset before writing code; direct generation of 30+ tools drifts.
- Summarise tracebacks with a separate cheap call before feeding them back.
- Restore a per-rollout SQLite copy; prefetch the next batch's servers during the gradient step.
- Separate "environment error" (reward 0, not agent's fault) from "agent error" in the reward taxonomy.
- Early-terminate on format errors with −1: 27% less rollout time; format errors otherwise stay > 20%.
- Train with the same history truncation used at inference.
- Measure judge reliability with repeated judgments (self-consistency, κ, flip rate) before using it as a reward.
- Check code-level diversity with AST-duplicate rate and endpoint-name Jaccard, not only embeddings.

**Fit to Crucible Part 2.**
- **F1** already follows AWM's order. Add what AWM lacks: blind tool tests with frozen expected states (ScaleEnv/EnvScaler style) instead of runtime-only checks, and zero tolerance for missing tables in kept Apps (their 10% tolerance explains the 74% bug rate).
- **F4**: AWM has no oracle; Crucible's "construction before sampling" (oracle program executed → gold end state) closes the 23–28% never-solved hole and turns "feasible" into G3 evidence.
- **F5**: keep AWM's `verify_task(initial, final)` + success/failure criteria as the *answer/state check generator*, but the reward must be the deterministic check; an LLM judge may only adjudicate `env_error` vs `agent_error` (fault side, §3.7), never award pass. Their Environment Error class maps to our `world_bug`/`env_flake` labels feeding the World loop.
- **G13**: their quality audit metrics (blocked tasks, bugs per env) are a ready density-card health report.
- **F12/§2(a)**: history-aligned training and format-error early termination are RL-path settings to confirm in Part 1.
- **Diversity**: reuse scenario-level embedding dedup (0.85) + category caps at F0, and AST/endpoint Jaccard as a cheap F8 check across Apps.
- Priority **P0** (already vendored; the deltas above are the work).

## 2604.18292 — Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence (Dong et al., 2026; arXiv preprint, Renmin Univ. + ByteDance Seed)
Read: FULL body + appendices 7–8 (diagnosis prompt, environment cards); the appendix 9 case-study trajectories are cut off in the HTML.  Relevance to Part 2: high. It grounds Worlds in mined real data, makes two task generators (graph walks, programs), and runs the only closed train → diagnose → targeted synthesis loop in this batch, which is the F12 → F3 path of §3.4 and §3.8.

**Problem.** Synthetic tool environments are often LLM-invented and simple, and training on them is single-round; there is no mechanism that uses the environment pool to find agent weaknesses and regenerate data for them.

**Method / algorithm.** GPT-OSS-120B is the policy for mining, task synthesis, code and rubric generation, and diagnosis.
- **Theme collection**: real MCP server specs from Smithery (M1), tool-definition docs from open datasets mapped back to topics by an LLM (M2), industrial product requirement documents (M3).
- **Agentic database mining**: a deep-research agent (search, browser, code, OS tools) mines topic-aligned web data and stores it as files (json, csv, sql, html, tex, yaml…); a *complexification* operator φ re-prompts the agent N rounds to enlarge and enrich the DB.
- **Tools + tests**: a coding agent writes candidate tools and, per tool, a set of unit tests. Keep a tool iff it compiles, passes > 50% of its own tests, and its environment has ≥1 valid tool and ≥1 valid test.
- **Taxonomy**: Ward hierarchical clustering of themes into 50 clusters, GPT-OSS-120B names them (second tier, seeded by TOUCAN's taxonomy), three annotators merge into 20 first-tier types; > 2K third-tier labels.
- **Graph-based tasks**: per environment, a fully connected weighted digraph over tools, edges labelled by an LLM: strong dependency (w = 3, input strictly needs the predecessor's output), weak (w = 2, input derivable but obtainable otherwise), independent (w = 1, fallback so no dead ends). Random walk from a node with output and no strong predecessors, next node sampled ∝ edge weight; arguments filled from predecessor outputs or sampled from the DB; an LLM prunes and checks the chain → τ*. Draft query without tool or schema names; execute τ* in a Python sandbox; refine the query from observed fields; LLM writes a JSON ground-truth answer and a rubric (field completeness, schema match, numeric tolerance).
- **Programmatic tasks**: the LLM writes a complex query from tool schemas + DB description, then a solver script with loops, branches, aggregation, debugged in a ReAct loop until it runs; its output is a*. Then an LLM writes `V_code(a, a*)` with multi-level assertions over answer and DB state, also debugged in the sandbox.
- **Solvability filter (both)**: a ReAct agent solves each task 5 times; keep iff ≥ 2 runs reach a consistent / passing answer.
- **Difficulty knobs**: longer random walks; higher sampling probability for weak and independent edges (less obvious data flow); rewrite queries to hide tool names and execution logic; for programs, more unique tools and calls, mandated branches, cross-DB aggregation, sorting, filtering.
- **RL**: cold-start SFT on 40K trajectories from an in-house Doubao-Seed-1.8; then GRPO on 5K RL samples, Qwen3-8B/14B, clip 0.2/0.28, 80K-token trajectories, 32K per step, 32 tasks × 8 rollouts per step, temperature 1.0. Reward: graph tasks = 1 iff a rubric-conditioned LLM judge passes *every* criterion; programmatic tasks = 1 iff V_code passes.
- **Self-evolving arena (Algorithm 1)**: arena = K = 5 environments per first-tier category (stratified). Each round r: synthesise fresh tasks + validators for arena envs; evaluate π_r; a diagnosis agent (Python + search) reads failure traces (tool logs, validator feedback, per-tool success counts), error stats per env and category, and env metadata, and outputs a ranked weak-environment list W_r plus per-env task-generation guidelines (e.g. state-update mistakes); complexify those envs' DBs if the weakness is state diversity; synthesise targeted tasks with the guidelines; continue RL from π_r.

**Verification / grading.** Two verifier kinds: (i) a rubric judged per criterion by an LLM (all must pass), (ii) generated assertion code over answer and state. Ground truth comes from executing a generated chain or script, so the answer is real; but the rubric judge is an LLM opinion, the verifier is written by the same model family as the solver script, there are no negative tests, and tools are kept at > 50% test pass, so up to half of a tool's own tests may fail. The 2-of-5 solvability filter uses a model, not the oracle, as the proof of solvability. The diagnosis agent sees validator internals and rubrics; nothing stops it from writing guidelines that target verifier blind spots.

**Scale & cost.** 1,978 environments retained of > 2,000; 19,822 tools (avg > 10 per env, max > 40); tasks average > 20 turns, minimum 7, some > 40. Doubao-Seed-2.0-pro pass@10: most tasks solved once in 10, few solved 10/10, some never. RL set only 5K samples; 40K SFT trajectories. No $ or GPU-hours, no per-stage yield (how many tools or tasks are filtered is not given).

**Key results.** (8 runs averaged per eval.)
- Agent-World-8B: τ² avg 61.8 (Qwen3-8B 26.2; EnvScaler-8B 37.9; AWM-8B 34.4; ScaleEnv-8B 38.5); BFCL-V4 51.4 (base 40.4); MCP-Mark 8.9 (base 2.4). 14B: τ² 65.4, BFCL-V4 55.8 (> DeepSeek-V3.2's 54.1), MCP-Mark 13.3.
- Environment-count scaling (0, 10, 100, 500, 1,000, 1,978 envs): average of 4 subdomains 18.4 → 38.5; biggest jumps 10→100 and 100→500; diminishing but positive after 500. MCP-Mark Postgres 4.8 → 19.9; BFCL WebSearch 7.0 → 47.0.
- Self-evolution, two rounds: Agent-World-14B τ²/BFCL/MCP-Mark-Postgres 60.2/52.4/29.5 → 63.5/54.9/36.3 → 65.4/55.8/38.1; applied to EnvScaler-8B 37.9/47.6/9.5 → 40.2/49.1/13.9 → 41.6/50.0/15.1. Round-2 gains smaller.
- Also reported (figures only): gains on SWE-bench Verified, Terminal-Bench, GAIA, HLE and no loss on math; SkillsBench/ARC-AGI-2/Claw-Eval 9.2/6.5/30.5 at 8B.

**Failure modes & limits.** Not discussed by the authors beyond diminishing returns. My critique: the cold start is 40K trajectories from a stronger in-house model, so the comparison with RL-only baselines (AWM, ScaleEnv) is not like-for-like and the environment contribution is not isolated; the text and Table 2 disagree on the 14B base τ² score (45.3 in text vs 60.2 in the table); self-evolution is 2 rounds with no control (same compute spent on more random tasks), so targeting is not shown to beat volume; the arena is synthesised from the same generator as the training data and its environments are reused for targeted training, so the "held-out" diagnosis can overfit; no audit of verifier or rubric correctness; a tool kept at 51% test pass is a real World bug source.

**Tips & tricks.**
- Seed themes from real MCP server specs and PRDs, not only from an LLM's imagination.
- Mine real web data into the World and enlarge it in rounds (complexification).
- Weight tool-graph edges (strong 3 / weak 2 / independent 1) and walk them; raise weak/independent edge probability to hide the data flow and raise difficulty.
- Hide tool names and schema from the instruction; refine the instruction after executing the chain so field names match reality.
- Two generators: linear chains (graph walk) and programs (loops, branches, aggregation) cover different reasoning shapes.
- Stratify the diagnostic arena by taxonomy category, K per category, fresh tasks each round.
- Feed diagnosis per-tool success counts and validator feedback, and require ranked weak envs plus concrete generation guidelines.

**Fit to Crucible Part 2.**
- **F0**: theme sources (MCP specs, tool docs, PRDs) and a 3-tier taxonomy are a direct template for `DomainSpec.families` and the coverage axes; the stratified arena is a held-out family meta-eval (§3.8 guard).
- **F1/F2**: web-mined, complexified DBs are a realism tier for Apps; keep our frozen-digest tool tests and a 100% pass bar instead of > 50%.
- **F3/F4**: adopt both generators as proposal styles (graph walk → action program; program → oracle script); both already give an executed oracle (F4) and real gold answer.
- **F5/F6**: replace the rubric-LLM reward with deterministic state/answer checks where possible; add G5 negatives and G12 attacker that Agent-World lacks.
- **F9**: replace "≥2 of 5 ReAct passes" by the oracle as proof, with the student band as the last gate.
- **Task loop (§3.8) and traces-to-tasks**: the diagnosis agent → weak envs + guidelines → targeted synthesis is exactly the "variant of task T aimed at failure cluster C" route from `factory/failures.py`; adopt with our guards: diagnosis sees verdicts and short reasons, never validator code (DGM lesson); targeted tasks go through full F6–F10; compare against an equal-budget untargeted batch to prove targeting helps.
- **Diversity**: their env-count curve (gains to ~500 envs, then diminishing) is one more reason to set family quotas in F11 rather than more tasks per family.
- Priority **P1** (loop design and generators), P2 for web-mined Worlds.

## 2603.00540 — LOGIGEN: Logic-Driven Generation of Verifiable Agentic Tasks (Zeng et al., 2026; arXiv preprint, Baidu + Tsinghua)
Read: FULL (body, appendix A case study skimmed, appendix B prompt and cost table).  Relevance to Part 2: high. Three ideas map straight onto Part 2: policy compiled into DB triggers (World rules that cannot be bypassed), boundary-adjacent seeding (a principled F2 difficulty knob), and forward exploration that yields the goal state by execution. It also has the cleanest evidence of user-simulator hacking.

**Problem.** Reverse synthesis (sample a tool chain, write a query for it) imitates happy paths, has few decision points where intent collides with policy, and gives no state target to verify. τ-style tasks need policy-governed state transitions.

**Method / algorithm.** Task package D = ⟨P wiki policy, I task description for the user simulator, E = ⟨schema with tables + triggers, atomic CRUD tools, Python+SQLite executor⟩, s_origin, s_target⟩. Models: DeepSeek-V3.2(-Thinking).
1. **Architect (policy compiler)**, four stages each followed by a Verifier-Agent Check–Fix–Verify (semantic checklist against the upstream artifact + DDL executes):
   - *Analyze*: plan in complex logic on purpose: multi-variable branches, state-dependent constraints, role permissions, irreversible transitions.
   - *Wiki policy*: expand seed domain notes (scenario + a few business rules) into a full natural-language policy.
   - *Static tables*: normalised schema, keys, types.
   - *Triggers*: BEFORE INSERT/UPDATE triggers reject invalid writes with a structured POLICY_VIOLATION; AFTER triggers apply mandated side effects (audit rows, inventory deduction).
   - *Tools*: per-table insert / query / update only (no delete; soft-delete via status); SQLite errors mapped to {code, message, violated_rule, hint}; tool descriptions state each BEFORE trigger as a precondition and each AFTER trigger as a side effect.
2. **Set Designer (seeding)**: *boundary-adjacent* state: numeric thresholds seeded at N−1; discrete states one valid action away from a violation. Resource injection by four strategies (trade-offs, distractors, substitutes, high-cardinality noise) and a cast of four user archetypes (mismatch, entangled in dependency chains, rookie with little context or privilege, edge at a constraint boundary).
3. **Explorer (forward synthesis)**: a Client agent (owns the goal, makes imperfect, changing, descriptive requests) and a Consultant agent (reads state and policy, offers a menu of feasible options, resolves IDs, executes; on POLICY_VIOLATION explains and offers alternatives). Loop: menu → goal → preference queries → parameter resolution → operate/negotiate; stop early on irreversible divergence. s_target = the actual DB after the episode (executed, so no ground-truth drift).
4. **User-view projection**: rewrite the episode into a goal-first, procedure-hidden task description I (no tool names, schema or IDs). The trainee gets (P, I, E, s_origin); s_target only for reward.
- **Reward**: DIFF(a, b) = Σ_tables |rows(a) Δ rows(b)| over canonicalised rows, excluding auto IDs and timestamps. R_final = 1 iff DIFF(s_final, s_target) = 0. Dense proximity P_t = 1 − min(DIFF(s_t, s_target), Δ0)/(Δ0 + ε) with Δ0 = DIFF(s_origin, s_target); r_t = P_t − P_{t−1} on success, −λ_err on a policy violation (trigger rollback).
- **Turn-aware GRPO**: A_it = A_i + min(r_t, 0): only negative step progress is added, so harmful steps inside successful trajectories are penalised, with no positive bonus.
- **Training**: SFT on 20K verified trajectories (DeepSeek-V3.2 user simulator + DeepSeek-V3.2-Thinking teacher, kept only if R_final = 1): 2 epochs, lr 5e-6, batch 64, 32K context. RL: cold start on ≈3K trajectories, then TA-GRPO in verl, KL 0, 1,000 steps, batch 64 × 8 rollouts, lr 1e-6, 50-turn cap, simulator and agent at temperature 1.0; tool and user tokens masked.

**Verification / grading.** Exact canonical state equivalence, path-independent; policy enforced by the DB engine so a violating write cannot land. Strengths: catches extra writes (the symmetric difference includes collateral), and the policy cannot be talked around. Weaknesses: single target state, so any valid alternative ending (a different but acceptable flight) fails; read-only intents leave DIFF = 0 from the start, so a no-op passes unless every task has a write (not stated); free-text columns must match exactly unless excluded; the target is whatever the Consultant did, including its arbitrary choices, so I must communicate every choice (no communicability gate is described); the user simulator is the main hack surface (below).

**Scale & cost.** > 2,000 environments, ≈10 tasks each, > 20,000 tasks across 8 core domains (Table 2: travel, retail/supply chain, healthcare, finance/insurance, education, enterprise ops/HR, real estate, tech/media). Difficulty tags: L1 51, L2 6,161, L3 3,788 (the counts sum to ≈10K, so only part of the set is tiered). Module success rates: Architect 68.5%, Set Designer 72.8%, Explorer 99.3%, trajectory distillation 74.2%. Cost at ¥2/¥3 per M input/output tokens, no prefix cache: Architect ¥1.22 (policy 0.09, tables 0.36, triggers 0.77), Set Designer ¥1.61, Explorer ¥1.24, distillation ¥1.43 — the paper does not say whether these are per environment or per task; total roughly ¥5.5 (≈ $0.75) per unit.

**Key results.** τ²-bench, pass^1 averaged over 4 runs, DeepSeek-V3.2 as user simulator (note: other rows use GPT-4.1 or others, so comparisons are confounded).
- 8B: base 30.8 → cold start 53.5 → TA-GRPO 71.8 (SFT on all 20K: 72.1). 32B: 40.7 → 62.7 → 79.5 (SFT 77.5). Retail/Airline/Telecom for 32B-RL 85.1/64.7/88.6.
- Beats AgentScaler-30B (61.8), EnvScaler-8B, TOUCAN-32B (31.6), MUA-RL-32B (47.0).
- TA-GRPO vs vanilla GRPO: faster, higher reward and higher τ² across steps, gap larger at 8B (curves only).
- Pass^k collapses with k: 32B-SFT Retail 81.6 at k=1 → 56.1 at k=4.
- **Simulator hacking**: Qwen-32B trained from scratch with the DeepSeek-V3.2 simulator at T = 0.1 gets higher, faster training reward than at T = 1.0, but against a mismatched GPT-4.1 simulator its Retail score rises then falls after steps 600–700; against the training simulator it stays high. T = 1.0 stays robust and ends higher on GPT-4.1.

**Failure modes & limits.** Authors: simulator hacking; capability–consistency gap. My critique: evaluated only on τ², and the 8 synthetic domains (airline-like travel, retail, telecom-like tech support) are the τ² domains, so OOD transfer is not shown; the user simulator also differs from the official τ² setup; RL adds ≈0 at 8B over full SFT; no ablation of boundary seeding, triggers or forward exploration individually, so which pillar matters is unknown; the Explorer's 99.3% success says little about task quality; no dedup or diversity measure.

**Tips & tricks.**
- Put business rules in DB triggers so violating writes are rejected by the engine, not by tool code.
- Return structured violation errors with the violated rule and a hint.
- Document each trigger's precondition and side effect in the tool description.
- Seed numeric limits at N−1 and states one action from a violation.
- Seed with trade-offs, distractors, substitutes and noise; cast mismatch, entangled, rookie and edge users.
- Get the goal state by running a Client–Consultant exploration, then project it into a goal-only instruction.
- Canonicalise DB diffs (drop auto IDs, timestamps) and use the symmetric difference size as a distance and a dense reward.
- Add only negative step progress to the advantage (turn-aware GRPO).
- Train against a high-temperature simulator and evaluate against a different-family simulator to detect simulator hacking.

**Fit to Crucible Part 2.**
- **F1**: add a `policies_as_code` tier in `AppSpec` compiled to triggers (or a StateAdapter pre-write hook) so path constraints are enforced inside the World as well as checked post hoc by F5.
- **F2**: boundary-adjacent seeding and the 4×4 resource/archetype menu become `DomainSpec` knob values (distractors, observability, start offset) and a persona axis for people-as-data.
- **F4**: Client–Consultant exploration is a third oracle route beside action programs; the executed end state is the gold digest (G2/G3). Require G8 communicability: every arbitrary choice the Consultant made must appear in I or the simulator's known info.
- **F5**: their symmetric-difference DIFF is our collateral check; their P_t is a ready milestone function for Φ. Add G14 alternatives for single-target tasks.
- **F9/F10 and the World loop**: simulator hacking is a fault-side signal. Add an intent probe: re-evaluate kept tasks with a different-family user simulator and flag tasks whose pass rate falls (simulator-dependent passes). Default simulator temperature to 1.0 in training.
- **§2(a)**: turn-aware negative-only advantage is cheap to test in the RL path.
- Priority **P0** for seeding and collateral DIFF; P1 for trigger-compiled policies and the simulator cross-check.

## 2512.22857 — AutoForge: Automated Environment Synthesis for Agentic Reinforcement Learning (Cai et al., 2025; arXiv preprint, Tongyi Lab, Alibaba)
Read: FULL (body, appendices A–B; short paper).  Relevance to Part 2: medium. The synthesis pipeline is a thin variant of AgentScaler's (docs → state keys → tool code → graph walks), but two pieces matter for us: the DAG composition of tool sequences with explicit reasoning nodes (a structural hardness knob), and masking trajectories that fail because the simulated user erred (a fault-side mechanism for the World loop).

**Problem.** Earlier pipelines are semi-automatic or produce easy tasks; LLM-simulated users err and make correct agent actions look wrong; multi-environment RL is treated as single-environment for advantage estimation.

**Method / algorithm.** Qwen3-Thinking-235B-A22B synthesises everything.
1. **Environment**: state is a key–value database S = [(K_i, V_i)]. From tool documentation the LLM first writes only the keys (attribute names); values are deferred to task time so state and task stay consistent. Then Python tool code against that key structure.
2. **Tool sequences**: tool graph with an edge when an LLM judges one tool's output a plausible input to another (AgentScaler style); random walks give thousands of sequences. Then three complexity operators:
   - *Merge* two sequences and let an LLM drop redundant tools (e.g. two different id lookups for the same entity);
   - *Reasoning nodes*: insert LLM-inferred computation steps on preceding outputs (e.g. total price of items);
   - *Reasoning edges*: LLM adds directed edges where a parent's output, after reasoning, yields a child's input → a DAG G_k.
3. **Task**: instantiate state values S_k for the DAG; LLM writes an initial intent Q̃_k from (S_k, G_k); fill tool arguments from Q̃_k in topological order and execute on S_k → gold end state S*_k; refine the intent to the minimal natural information from (S_k, S*_k). Sample D_k = (Q_k, S_k, S*_k, F).
4. **ERPO (RL)**: GPT-4.1 user simulator issues requests from Q; agent calls tools or asks the user; rollout ends when the user says all requirements are met. Reward R = 1 iff the end state equals S*, else 0. Changes to GRPO: (a) *interleaved thinking*, keep all prior reasoning across user turns in training and inference; (b) *MEU*: an LLM judge reads the user intent and the dialogue and flags trajectories where the simulated user gave wrong, out-of-scope or missing information; flagged trajectories get zero weight in loss and advantage; (c) *environment-level advantage*: A_ij = (R_ij − mean_j R_ij) / std over all trajectories of all questions in the same environment (mean per question, std per environment) to stabilise small-group std; (d) DAPO dynamic sampling drops all-pass and all-fail groups. Cold-start SFT on correct trajectories of the backbone Qwen3-Thinking-30B-A3B itself; batch 32 × 8 rollouts; 64 GPUs.

**Verification / grading.** Exact end-state equality. Problems: single gold end state from one argument filling of one DAG (alternatives fail); read-only DAG parts leave no state trace, so reasoning nodes are not verified unless their result is written (no answer check described); key–value state with no schema constraints or FKs; no negatives or solvability check beyond executing the gold sequence; MEU is itself an LLM judge, and a policy can learn behaviours that make the *user* look wrong (e.g. confusing the user into an out-of-scope reply) to get bad trajectories masked rather than penalised.

**Scale & cost.** Only 10 environments and 1,078 tasks. Appendix B: environment execution is about 1/7 of RL wall time (env 1 : LLM 6.04, units of average env execution time). No token or $ cost.

**Key results.** Qwen3-Thinking-30B-A3B → AutoForge-30B-A3B: τ-bench Retail 67.8→73.1, Airline 48.0→56.5; τ² Retail 58.8→74.8, Airline 58.0→62.0, Telecom 26.3→76.3; VitaBench Delivery 35.0→46.0, In-store 40.0→54.5, OTA 20.5→24.0, Cross 16.0→17.5. Beats AgentScaler-30B-A3B on all but τ Airline is close. ACEBench-zh (Chinese, custom format, unseen tools): SFT and RL both above base (figure only). Better user simulator raises measured τ² scores: Telecom 76.3 with GPT-4.1 → 90.4 with GPT-5-thinking as user; prompt optimisation gives +0.5–0.6. Ablations shown only as training-reward curves: env-level advantage gives a smoother, higher curve; without MEU the reward drifts down late in training; interleaved thinking helps (figure).

**Failure modes & limits.** Authors: needs tool documentation; only a few environments; outcome reward only. My critique: the paper calls τ/τ²/VitaBench "in-domain", meaning its 10 environments target those domains, so it is not evidence of transfer; ablations are reward curves, not benchmark numbers; MEU's false-mask rate is never measured; env-level std mixes questions of different difficulty, which down-weights hard questions' advantages; the user-simulator result shows benchmark scores can move 14 points with the simulator alone.

**Tips & tricks.**
- Generate state keys first and fill values per task, so the state fits the task.
- Merge walked sequences and dedupe redundant lookups to get multi-requirement tasks.
- Insert explicit reasoning nodes (aggregation, comparison) between tool calls to force computation.
- Fill arguments in topological order from the drafted intent, execute, then rewrite the intent from start and end states.
- Mask, don't penalise, trajectories that failed because the simulated user erred.
- Keep reasoning across user turns for multi-turn agents (interleaved thinking).
- Report benchmark results with the user simulator named; it moves scores by > 10 points.

**Fit to Crucible Part 2.**
- **F3/knobs**: "merge + reasoning node + reasoning edge" is a concrete *linkage* and *count* knob for synthetic World tasks, and it changes the behaviour signature (F7), which is what §3.6 requires of a real hardness increase. Reasoning nodes need an answer check (F5) or a written result, or they are unverifiable.
- **F9/F10 fault sides**: MEU is the "simulator side" fault label. Adopt as a labelled fault side, but (a) measure the masker against the anchor set, (b) never mask on the policy's own claim, (c) track mask rate per task: a high rate is a World-loop signal that the task's user profile or known info is broken (G8 communicability), not a reason to keep masking.
- **Evaluation (§2(e))**: record the user simulator model and prompt in every DifficultyRecord; band verdicts are not comparable across simulators.
- **Avoid**: KV-only state (no FKs) and single-target state equality; our F5 collateral + alternatives cover both.
- Priority **P1** (MEU-as-fault-side, DAG composition knob); the pipeline itself P2.

## 2601.05808 — EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis (Song et al., 2026; arXiv preprint, Renmin Univ.; v2 Apr 2026)
Read: FULL (body, appendices A–C with all prompts).  Relevance to Part 2: high. Its dual-agent blind tool assessment is the precedent for F1's blind tester (the plan cites it), its checklist-of-state-predicates verifier is a cheap F5 pattern, and its cost table is one of the few per-environment and per-scenario $ figures.

**Problem.** Build many stateful, rule-bearing tool environments without real systems, LLM simulators or tool documentation, plus tasks and rule-based rewards for them.

**Method / algorithm.**
- **SkelBuilder** (GPT-4.1 and Qwen3-235B-Instruct for discovery and code; GPT-4.1-mini and Qwen3-30B-A3B-Instruct for assessment):
  1. *Task-guided discovery*: from existing task sets (API-Bank, ToolACE), an LLM keeps tasks that live in a persistent, domain-specific, stateful environment (both reads and writes, state-dependent answers), infers an environment description for each, then embedding-dedups the descriptions.
  2. *Logic planning*: LLM enriches the description into a state definition, domain rules (≈4.6 constraint rules per env) and a tool list.
  3. *Program modelling*: state → Python class attributes; each tool → a class method written given rules + attributes; merged into one class file; AST-parse check (invalid files discarded); signatures extracted by AST + regex into the tool interface.
  4. *Dual-agent assessment*: a **frontend testing agent** sees only the tool interface and the current state (not the code) and issues one call per round, positive or negative (wrong types, missing IDs, out-of-range, empty, extreme, special characters), balanced across tools; the env executes; a **backend checking agent** sees the method source, arguments, return value and state diff and labels Pass / Warning / Fail with a reason. N = 100 rounds; env score = mean pass; keep if ≥ 0.85.
- **ScenGenerator**:
  1. Initial state: LLM writes a JSON config for the class with cross-linked valid IDs, 3–5+ entities per collection, varied values and dates, realistic names.
  2. Task: LLM writes a realistic, feasible, *state-modifying* (no pure queries), multi-object or multi-condition task from the state, tools and rules.
  3. Verifier: LLM decomposes the task into independent "Has …" checklist items, each with exact values; each item becomes `check_func(final_state) -> bool`; random IDs and timestamps checked only for presence and type; non-fixed targets (e.g. "add a remark") checked for non-empty. Reward = fraction of checks passed.
- **Interaction modes**: Non-Conversation (full task given) and Conversation (an LLM user reveals sub-goals piece by piece, ends with ###STOP###).
- **Training**: 140 envs for SFT, 51 disjoint envs for RL. SFT on ≈9K Qwen3-30B-A3B-Thinking trajectories (both modes; drop invalid-format or LLM-judged-impossible ones); n-round samples split into n sub-samples supervising only the last round's reasoning (Qwen3 strips past reasoning); 3 epochs, lr 1e-6, batch 256. RL: Reinforce++ in ROLL, KL 0.1, lr 1e-6, 64 tasks × 8 rollouts, ≤200 steps, Non-Conversation only.

**Verification / grading.** Checklist of end-state predicates; process-agnostic, so alternative paths pass. Hackable: partial credit per check; no collateral check, so extra or destructive writes outside the checklist are free; "non-fixed target" checks accept any non-empty value; the task generator writes the checklist with no oracle, so feasibility is never proven (impossible tasks just give low reward); checklist written by an LLM can miss a required condition (verifier too lax) and nothing tests it with negatives. Env assessment is an LLM judging code, not frozen expectations, and a 15% failure tolerance remains.

**Scale & cost.** 266 synthesised envs → 191 kept (28.2% rejected, mostly type/attribute errors, then state-consistency). Avg 18.6 tools (10.4 query, 8.1 write), 21.4 state categories, 4.6 rules. ≈7K scenarios (≈37 per env). Cost (GPT-4.1 / mini): per environment $1.02 (discovery $0.008, construction $0.25, 100-round assessment $0.77 = 1.76M tokens); per scenario $0.064 (state $0.029, task $0.010, check functions $0.024).

**Key results.** Qwen3 thinking models, averages of 3 runs.
- 8B: BFCL-v3 multi-turn 28.88 → 37.00 (SFT) → 41.88 (SFT+RL); τ-bench 38.19 → 41.35 → 44.81; ACEBench-Agent 60.00 → 71.67 → 72.50. 4B and 1.7B similar on BFCL/ACE; 1.7B RL slightly hurts τ (17.44 → 16.28).
- Direct RL without SFT helps but less (8B BFCL +8.74, τ +4.11).
- Train–test similarity: SFT on the 50% of envs most similar to BFCL vs the 50% least similar: 32.00 vs 32.50 (random 50%: 31.25; all: 34.88); similarity does not drive the gain.
- Env count (SFT, 4B): steepest gain from 0 to 20 envs, then slower but rising.
- Interaction mode: Non-Conv SFT better on complete-information subsets, Conv SFT better on missing-parameter; both together best (37.00). Miss-Func does not improve because no training task lacks a tool.
- Non-thinking 4B: EnvScaler SFT beats APIGen and APIGen-MT on BFCL/ACE, but APIGen-MT (built on τ domains) wins on τ.
- Pilot on 50 scenarios: Qwen3-8B 57.8, max@8 69.4; ≈15 steps Non-Conv, > 25 Conv.

**Failure modes & limits.** Authors: LLM bias in business logic; no web/IR; no latency or error-pattern simulation; text only; ≈200 envs. Others (AWM) report 46.8–57.1% blocked tasks in EnvScaler envs vs 11.5–14.0% for AWM, and suspect τ² overlap via seed tasks. My critique: no solvability proof for tasks; partial-credit reward rewards doing some of the task; the 0.85 threshold keeps envs where 15 of 100 probes fail; RL only in Non-Conv mode; small eval sets (ACEBench-Agent 50 tasks) make +1–2 point differences noise.

**Tips & tricks.**
- Mine environment themes from existing task sets by inferring "which stateful system would this task live in".
- Plan state, rules and tool list first; generate each tool method separately against the fixed attributes.
- Run a blind tester (interface only) and a code-aware checker together; include negative probes; score = pass rate over 100 calls.
- Require tasks to write state; reject query-only tasks to avoid no-op passes.
- Split the task into independent "Has …" predicates with exact values; relax only generated IDs and timestamps.
- Train with both full-information and progressive-disclosure dialogues; add missing-tool cases if that skill is wanted.
- The 100-round assessment is 75% of env cost; tune N.

**Fit to Crucible Part 2.**
- **F1**: adopt the frontend/backend split as the blind tool tester and debugger roles, but make the tester write *expected outcomes* before execution (frozen digest), so the checker cannot rationalise the code's behaviour; keep negative probes; per-App report = pass rate, repair iterations, abandoned tools (matches `AppBundle.report`). G13 interaction completeness can reuse their probe generator.
- **F3/F5**: the "Has …" checklist is a good *state check* generator, provided F4 executes an oracle first (they have none) and F5 adds collateral (full diff inside the allowed set) and negatives (missing, wrong value, side-effect). Score binary at F6 (all checks), partial credit only as Φ milestones.
- **Diversity**: their similarity ablation supports measuring behaviour signatures over topic similarity; their env-count curve (0→20 steep) suggests even our first families pay off early.
- **Cost model (§3.10)**: ≈$1 per App and ≈$0.06 per scenario with GPT-4.1-class models is the cheapest published F1/F2–F5 rate; use it as a prior for swarm budgets.
- **Avoid**: seeding themes from public tool-use benchmarks' task sets (contamination); tolerance thresholds below 100% for kept Apps.
- Priority **P0** (blind tester design, checklist verifier); cost prior P1.

## 2509.13311 — Towards General Agentic Intelligence via Environment Scaling (Fang et al., 2025; arXiv preprint, Tongyi Lab / Alibaba) [AgentScaler]
Read: FULL (the arXiv HTML v1 is the whole paper: sections 1–7 plus Limitations; there is no appendix, so no prompts, thresholds or counts beyond the main text exist to read).  Relevance to Part 2: high for the F1/F2/F3 skeleton (tools → domains by graph communities → DB schema → executable tools → tool-graph walks → tasks with a gold state) and for coverage-as-countable-communities (§3.5 already cites it); medium overall, because the paper is thin on numbers, has no RL, no environment-quality gate beyond trajectory filtering, and no difficulty control.

**Problem.** Function-calling breadth tracks the diversity of training environments, but stateful, verifiable tool environments (τ-bench style) are hand-built. The paper wants to (i) build many fully simulated, database-backed environments automatically and (ii) learn from simulated user–agent experience in them.

**Method / algorithm.**
*Design principle.* Every tool is a read or write operator on a domain database D: `API(func, α) ≡ op(func)(α; D)`, op ∈ {read, write}. Tools in one domain share a schema S_k. So environment design = partition the tool space into domains {T_1..T_M} and give each a schema. (Their stated basis is StableToolBench-MirrorAPI's view; the same abstraction as τ-bench.)

*Stage A — environment build.*
1. **Scenario collection.** >30,000 APIs from ToolBench, APIGen and an internal repository. Filter low-quality APIs (criteria not given), rewrite some descriptions to carry explicit input/output specs (pointer to SynWorld). Build tool compositions from I/O relations. Pool size N > 30k.
2. **Tool dependency graph.** Nodes = tools. Edge (i, j) iff cos-sim(φ(params_i), φ(params_j)) > τ, i ≠ j, where φ embeds a tool's parameter list. Neither the embedding model nor τ is reported.
3. **Domain partition.** Louvain community detection on that graph; each community = one domain/environment. Because parameter-vector matching misses true dependencies, an LLM then checks every tool pair *within* a domain to correct edges (and, implicitly, give them direction for the walk below). Result: M > 1,000 domains.
4. **Programmatic materialisation.** From all tool parameters in a domain, an LLM generates a domain DB schema; then writes each tool as Python code doing reads/writes on that schema. On τ-bench's domains, manual inspection found the generated schema and code "highly consistent" with the official implementation (qualitative claim, no metric).

*Stage B — task construction (forward, solution-first).*
1. Initialise a DB state from the schema, "encouraging as much diversity as possible" (method unspecified).
2. Build a **directed** dependency graph over the domain's APIs; start at a random node; directed walk until max steps or a sink node → a "logically coherent" tool sequence.
3. For each step, generate arguments and **actually execute** the call against the DB, tracking the evolving state. Output: gold tool sequence + arguments + gold final DB state.
4. Integrate the sequence into one overall user intent (the simulated user's goal).

*Stage C — experience collection.* τ-bench-style interplay: a simulated user with the overall intent talks to the agent, which uses the domain's tools, until the user deems the task done. Models used for the user and the agent are not stated.

*Stage D — three-stage funnel filter.*
1. Validity: well-formed alternating user/assistant turns; n-gram filter drops trajectories with heavily repetitive reasoning.
2. Environment-state alignment: keep only if final DB state = gold state (checks writes).
3. Function-call exact match: tool sequence and arguments must exactly match the gold sequence. Applied for read-only sequences in particular, where state alignment is vacuous (the text says stage 3 is "the most stringent"; it is ambiguous whether it applies to all trajectories or only read-only ones).
Trajectories whose intermediate tool calls returned errors are **kept** if they pass the filters, for robustness.

*Stage E — training.* SFT with loss only on tool-call tokens τ_t and assistant-reply tokens y_t; user turns h_t and tool responses ρ_t are in context but masked. Two phases: (1) broad general domains for basic tool use and user interaction; (2) vertical, target-domain scenarios for specialisation. Backbones: Qwen3-Thinking-4B-2507, Qwen3-8B, Qwen3-Thinking-30B-A3B-2507. No RL (listed as future work).

Pseudo-code for the task synthesizer as described:
```
for domain d in louvain(G_param_sim > τ) refined by LLM pairwise edge check:
    S_d = LLM_schema(params(T_d)); tools_d = LLM_code(T_d, S_d)
    repeat:
        D0 = init_state(S_d)                       # "diverse"
        seq = directed_walk(G_d, start=random, max_steps)
        D = D0; for f in seq: α = gen_args(f, D); obs = exec(f, α, D)   # D mutates
        intent = LLM_summarise(seq, args)          # user goal
        emit Task(D0, intent, gold_seq, gold_args, gold_state=D)
```

**Verification / grading.** Two granularities: final DB state equals gold state (writes), and exact tool+argument match (reads). Hackable/fragile points: (a) exact sequence match rejects valid alternative solutions (single-reference false negatives; the plan's G14 problem); (b) state equality without column policies is brittle to harmless fields (timestamps, generated ids) and silent on read-only answers; (c) the user simulator decides "done", so a wrong user can end an episode early; the filters, not the user, carry correctness; (d) no check that the environment's tools are correct beyond the trajectory's agreement with the gold run, which was produced by the same tool code. A buggy tool yields a consistent but wrong gold state.

**Scale & cost.** >30k APIs → >1,000 domains. Number of tasks, trajectories, yields per filter stage, tokens, $ and GPU hours: **not reported.** τ and the embedding model: not reported.

**Key results.** (pass^1 on τ-bench / τ²-Bench, accuracy on ACEBench-en)
- AgentScaler-30B-A3B: τ Retail 70.4, Airline 54.0; τ² Retail 70.2, Airline 60.0, Telecom 55.3; ACEBench-en Overall 75.7 (Normal 76.7, Special 82.7, Agent 60.0). Base Qwen3-Thinking-30B-A3B: 67.8 / 48.0; 58.8 / 58.0 / 26.3; 67.2. Largest gain: τ² Telecom +29.0.
- AgentScaler-4B: τ² Telecom 48.2 vs base 28.7; ACEBench-en Overall 65.9 vs 49.5; but Special 76.7 vs 84.7 (a drop).
- AgentScaler-8B: τ Retail 50.4 vs 45.2; ACEBench-en Overall 67.4 vs 65.9 (small).
- OOD ACEBench-zh Overall: 4B 43.9→65.6, 8B 71.3→73.7, 30B 74.2→81.5; Agent subset 4B 6.7→38.4. Special subset drops for 4B (−15.3) and 30B (−3.4).
- For context: Kimi-K2-1T τ² Telecom 65.8, GPT-5-think 96.7; AgentScaler-30B does not beat them on Telecom.
- Ablation (Fig. 3, numbers only in a figure): both stage 1 and stage 2 beat the base on ACEBench-en; stage 2 raises the Agent subset further. No ablation of the graph, the LLM edge check, the filters, domain count, or data size.
- pass^k on τ² falls with k for all models; AgentScaler stays above its base at every k.
- Accuracy falls with tool-call count on τ-bench (scatter, Fig. 5); long horizons stay unsolved.

**Failure modes & limits.** Authors: SFT only; validated only up to 30B; long-horizon tool use still poor. My critique:
- **Contamination by construction.** The pipeline reproduces τ-bench's schemas and tool code "highly consistently", and stage 2 trains on vertical domains matching the eval targets. Gains on τ/τ² may largely be in-domain fitting; ACEBench-zh is the only OOD check, and it is the same task family in another language.
- No environment-quality gate: no blind tool tests, no check of tool code against a spec, no interaction-completeness check. Correctness of 1,000+ LLM-written environments is assumed.
- No per-stage yields, no counts, no ablations of the core claim (that more/more-diverse environments help). The "environment scaling" thesis is not tested as a scaling curve.
- Random directed walks over a similarity graph give coherent-but-arbitrary sequences; nothing ties a task to a skill, and nothing controls difficulty beyond max steps.
- Exact-match filtering biases the SFT data toward the gold path, which teaches path imitation, not goal reaching.
- Special-subset regressions (refusing / handling missing info) suggest the data under-represents infeasible or under-specified requests.

**Tips & tricks.**
- Treat each tool as a read or write over a shared schema; group tools into domains by parameter-embedding similarity + Louvain, then repair edges with an LLM pairwise check inside each community.
- Generate the schema from the union of a domain's tool parameters, then code the tools against it.
- Build tasks forward: sample a directed tool walk, execute it on a seeded state, and freeze the end state as gold before any agent runs.
- Use state equality for write tasks and exact call match only for read-only tasks (or better, an answer check).
- Keep trajectories that recover from tool errors if they still reach the goal.
- Mask user and tool-response tokens in the SFT loss.
- Train broad-then-vertical (two phases).

**Fit to Crucible Part 2.**
- **F0/F1 (DomainSpec, App synthesis).** Adopt the tool-graph → community → schema → code recipe as one F1 recipe for domains that start from an API catalogue (MCP servers, OpenAPI specs). The communities give a countable `families` list and a natural `world_facet` axis for the §3.5 grid. **Add what it lacks:** our F1 gates (blind tool tests with frozen expected states, schema integrity, G13 interaction completeness), because AgentScaler trusts generated code.
- **AppBundle.tool_graph.** Their directed dependency graph is exactly our `tool_graph: [Edge{from,to,kind}]`; record the evidence (embedding similarity, LLM pair check) per edge. Use it in F3 to make walks, and in F7 to compute `cross_app_flows`.
- **F2.** "Diverse initial state" is unspecified; our `SeedBundle` with determinism and FK gates (G1, G2) is the stronger version.
- **F3/F4.** Their walk-execute-freeze is a solution-by-construction oracle (`Oracle(kind=action_program)`, `gold_end_state_digest`). Adopt; but condition the walk start and length on a sampled grid cell and knob vector (horizon, tool count) rather than uniform random, and **reject walks whose behaviour signature duplicates an existing task** (F8).
- **F5.** Replace exact sequence match with our five check kinds: state checks under column policies, collateral, answer checks for read-only intents (their read-only exact match is our "answer check" gap), path constraints; and generate negatives from the oracle (drop step, wrong value, duplicate write) — none exist here.
- **G14.** Their exact-match rule is the anti-pattern G14 exists to fix; avoid it for RL rewards.
- **Diversity / held-out.** Use Louvain communities as families and hold out whole communities (15–20%) for OOD transfer; they did not, which is why their OOD claim is weak.
- **Contamination (G11).** Their τ-bench reproduction warns that construction models regenerate public benchmark domains; add a similarity check of generated schemas/tools against known benchmark schemas before a domain enters training.
- **Interface implied:** `ToolCatalogue → [Domain{tools, schema, tool_code, tool_graph}]`; `Domain × seed × walk → TaskProposal + Oracle + gold_state`.
- **Priority:** P1 (an F1/F3 recipe for API-catalogue domains, after the generic contract exists). The idea of communities as countable families is P0-cheap and already in §3.5.


## 2510.01179 — TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments (Xu et al., 2025; arXiv preprint, UW + MIT-IBM Watson AI Lab)
Read: FULL (main text, Appendix A schema, B server analysis, C annotation correlation / SFT hyper-parameters / ablation, D all five prompt templates).  Relevance to Part 2: medium. It is an SFT data pipeline over *real* MCP servers with LLM-judged quality, not a verifiable-environment or RL-task factory: no state, no ground-truth answer, no oracle. Useful for F3 proposal prompts (single / multi / featured server), task-quality rubric dimensions (notably "stability" and "verifiable"), the irrelevance (infeasible) extension, and as a negative example of what LLM-judge-only filtering cannot certify.

**Problem.** Open tool-agent training data is small, single-turn, or uses simulated tool responses. Toucan builds a large, permissively licensed set of trajectories with real tool execution over many MCP servers, including parallel, multi-step, multi-turn and no-tool (irrelevance) cases.

**Method / algorithm.** Five core stages plus three extensions.
1. **MCP server onboarding.** Crawl ≈2,800 server specs from GitHub and Smithery (June 2025). Keep remote servers reachable over streamable HTTP and those needing no third-party credentials → 871 (30.6%). Generate a few test questions per tool and drop servers whose tools error or misbehave → **495 servers, >2,000 tools**. An LLM labels each server with a primary category from 26 predefined ones (plus up to 2 secondary and an optional custom label).
2. **Task synthesis.** A task = question + target tool names. Five generator models to spread model bias: Mistral-Small, DevStral-Small, GPT-OSS, Kimi-K2, Qwen3-32B. Max tools per task N = 3. Three strategies:
   - *Single server:* 1..N tools from one server; tool choice balanced by Smithery usage statistics.
   - *Multi-server:* sample N servers from the same or different categories; the LLM first writes a server analysis and a cross-server workflow, then a task using 2..N tools across ≥2 servers.
   - *Featured server:* 25 hand-picked servers all placed in context; the LLM brainstorms a scenario and picks tools freely.
   Prompt rules worth copying: no tool or server names in the question; "cannot reasonably be answered with a single tool / single server"; list target tools in expected call order; XML output with analysis before the question.
3. **Task filtering.** Kimi-K2 rates each task 1–5 on six dimensions: tool-selection difficulty, tool-selection uniqueness (are there viable alternative tool combinations?), question quality, scenario realism, verifiable (rule-checkable answer vs free-form), stability (does the answer change with time, geolocation, OS, network, versions?). Reasoning is written before each rating. Kimi-K2 was chosen on Pearson correlation with human labels on 50 samples (GPT-4.1 and Kimi-K2 highest; Kimi-K2 cheaper locally). Exact thresholds for the core filter are not stated; the SFT subset uses the thresholds below.
4. **Trajectory generation.** Three teacher models from different families (GPT-OSS-120B, Kimi-K2, Qwen3-32B) × two agent frameworks (Qwen-agent, OpenAI-agent), calling the real servers over streamable HTTP.
5. **Post-filtering.** Rules drop trajectories that failed to start or connect, have no tool calls, have failed tool responses, or contain local file-system paths. Two automatic metrics: *desired tool use percentage* (coverage of target tools) and *order correctness* (target order respected). GPT-OSS-120B judges completeness and conciseness (1–5).

Extensions (each re-runs stages 2–5 on new tasks):
- **Ext.1 Irrelevance:** shuffle server metadata across instances so the shown tools cannot solve the task; keep only trajectories with zero tool calls.
- **Ext.2 Persona diversification:** (a) same tools, same order, same outcome, new context/persona; (b) same context, more constraints (stakeholders, compliance, deadlines), step-wise prompts rewritten as goal-oriented.
- **Ext.3 Multi-turn:** self-simulation by the trajectory model, either splitting a multi-tool task into sequential sub-questions or appending LLM-written follow-ups.

SFT subset selection: question quality = 5, scenario realism = 5, completeness ≥ 4, conciseness ≥ 4, desired-tool-use = 1.0, then category re-balancing → 28.3K core + 40K irrelevance + 15.8K diversify + 35.2K multi-turn = **119.3K** (≈8% of the 1.5M). SFT: Hermes tool template, lr 2e-5, 2 epochs, effective batch 64, AdamW, ZeRO-3, max length 32,768; Qwen2.5-7B/14B/32B-Instruct.

**Verification / grading.** No ground truth. Correctness = rules (no errors, tools used, order) + an LLM judge's completeness score. Hackable/weak: a trajectory can call the target tools in order and confidently state a wrong answer; "completeness" is judged without the true answer; real tool outputs drift over time (they rate "stability" but on the question, not by re-running); tool-order correctness penalises valid alternatives (they measure uniqueness, which helps, but do not use alternatives as references). Multi-turn follow-ups are written by the same model, so the user is not adversarial.

**Scale & cost.** 1,527,259 trajectories; 567,262 multi-turn; 495 servers, >2,000 tools; most servers expose <10 tools. Yield: 2,800 → 871 → 495 servers (17.7% overall). Per-stage task/trajectory yields, token counts, $ and wall time: not reported. Evaluation on one 8×H100 server; SFT on 8 or 64 H100.

**Key results.**
- BFCL V3 overall: 7B 55.10→58.26 (+3.16), 14B 57.69→65.09 (+7.40), 32B 61.73→70.45 (+8.72; above GPT-4.5-Preview 70.32 and DeepSeek-V3 64.71). Multi-turn subset 32B 26.38→46.50, the largest gain.
- 7B regressions: BFCL non-live AST 84.19→78.52; relevance 72.22→66.67; τ² Telecom 16.70→10.50.
- τ-bench avg: 7B 15.03→22.48, 14B 30.85→35.24, 32B 38.76→42.33. τ² avg: 7B +1.69, 14B +5.97, 32B +2.20 (32B Telecom flat, 21.11→20.20).
- MCP-Universe (231 tasks, 11 servers, mostly *not* in the training pool because they need configuration): Toucan models above similar-size open models; 32B best on 3D Design; numbers only in figures.
- Ablation on 14B (cumulative): core single-turn only gives BFCL 60.16 but **hurts** τ-bench Retail (44.46→36.95) and BFCL irrelevance (68.46→46.88); adding irrelevance restores irrelevance to 77.85 and BFCL to 64.74; diversify ≈ flat (64.56); multi-turn brings τ Airline 17.25→22.00 and Retail to 48.48. So the multi-turn and irrelevance extensions carry most of the agentic gain; single-turn tool-calling data alone trains over-calling.

**Failure modes & limits.** Authors: June-2025 snapshot; servers needing keys or setup (Notion, GitHub) excluded; real execution is slow and costly (plan an LLM tool simulator). My critique:
- No verifiable reward, so it cannot be used for RL as is, and quality rests on two LLM judges validated on 50 samples.
- The server pool is skewed to keyless, read-mostly, public-data servers (search, weather, calculators); write-heavy, stateful enterprise tools are under-represented, and stateful side effects are never checked.
- Only 8% of the data was used for SFT; no evidence that the 1.5M scale itself matters (no data-size ablation).
- Gains are small or negative on τ²'s Telecom (dual-control, stateful), which is closest to our World tasks.
- Multi-turn by self-simulation teaches the model to talk to itself; the persona of the user is not grounded.

**Tips & tricks.**
- Filter servers by live probes (a few test calls per tool) before generating tasks.
- Prompt for "requires exactly K tools from ≥2 servers, cannot be done with one", with a cross-server workflow written first.
- Rate tasks on stability (time, place, version dependence) and verifiability; keep both as task metadata.
- Rate tool-selection uniqueness to know when a single reference sequence is unsafe.
- Build infeasible tasks by shuffling the tool list away from the task; grade them by "no tool call" plus a refusal.
- Generate persona variants that keep the tool sequence and outcome fixed, and constraint variants that keep the context fixed.
- Validate each LLM judge against ~50 human labels and pick by correlation per cost.
- Single-turn tool data alone can hurt agentic benchmarks and irrelevance handling; mix in multi-turn and no-tool data.

**Fit to Crucible Part 2.**
- **F3 (proposal).** Reuse the three prompt shapes as proposal templates, but condition them on a sampled grid cell and World slice (§3.5) instead of free choice, and add our K-nearest "do not repeat" references.
- **F3/F10 task-quality metadata.** Add `stability` and `verifiable` to `TaskProposal` as proposer self-ratings that a later gate checks: our G2 determinism and G10 flake are the executed versions of "stability"; an answer-check or state-check existing is the executed version of "verifiable". Do not trust the LLM ratings in place of the gates.
- **Infeasible tasks (§3.3).** Ext.1's shuffle is a cheap generator for the `infeasible` task type: present a tool set that cannot satisfy the intent; grade "no write + explicit infeasibility report". Adopt at P1, with a check that the shuffled tools really cannot solve it (an oracle search over the shown tools).
- **Diversity.** Ext.2(a) is "same behaviour signature, new text": by our F8 rule these are duplicates to the learner. Use them only as instruction-paraphrase augmentation inside one task, not as new pool entries. Ext.2(b) "harder via constraints" must change the signature to count (§3.6 rule).
- **MCP sidecars (§2(b)).** The onboarding filter (remote, keyless, live-probed) is a useful checklist for which real MCP servers can back a World App; real servers fail our G2/G10 unless snapshotted or mocked.
- **Avoid:** LLM-judge completeness as a reward; target-tool order as a grader.
- **Interface implied:** `ServerCatalogue → probe → [App]`; `TaskProposal.meta {tool_selection_uniqueness, stability, verifiable}`.
- **Priority:** P2 overall; the infeasible-task generator and the two rubric fields are P1.


## 2601.08276 — ACE-Router: Generalizing History-Aware Routing from MCP Tools to the Agent Web (Yao et al., 2026; arXiv preprint v2, 19 Apr 2026; SJTU / Huawei and others) [cited in our notes as "ToolACE-MCP"; the v2 title is ACE-Router]
Read: FULL (v2 HTML: sections 1–5, Limitations, Appendices A–F incl. mutation taxonomies, both mutation prompts, agent-route benchmark example, variance tables).  Relevance to Part 2: low–medium. It trains a tool/agent *router* by SFT on fully LLM-simulated trajectories; no executable environment, no verifier. Two transferable ideas: (1) **LLM mutation operators that grow a tool catalogue with near-duplicate, confusable tools** (hard-negative distractors), and (2) **cutting each multi-turn trajectory into per-step (history, query) → next-tool labels**. Both matter for F1 distractor density and difficulty knobs (tool count, distractors), not for verification.

**Problem.** With MCP, the candidate tool/agent set is too large to put in context, and embedding retrieval (a) cannot separate functionally similar tools and (b) ignores the interaction history (prior results, failures, prerequisites). The paper trains a router π_θ(c | Q, H, C) that picks the next tool or agent given the current query Q, history H and candidate space C, and wraps it in a two-tool "Light Routing Agent" (router-invoke + execute).

**Method / algorithm.**
1. **Candidate graph.** Embed each tool's serialized description + input schema with all-MiniLM-L6-v2; undirected edge if cosine > τ = 0.82.
2. **Self-evolutionary mutation.** Repeat: sample a candidate c and an operator m; an LLM writes c′ = m(c) as JSON schema; add node c′ with an edge to c. Tool operators: *usage extension* (same logic, new domain), *function enhancement* (add ≥2 major features), *workflow chain* (a pre- or post-processing tool), *helper tool* (e.g., a validator), *parameter redesign* (single lookup → filtered query). Agent operators: domain transfer, capability enhancement, workflow specialization, tool composition, scenario adaptation (4–8 tools per agent). Prompt constraints: keep domain tags, meaningful differentiation, no copying, valid JSON. No dedup or validity filter on mutants is described. Tool bank: 627 MCP tools (from MCP-Universe and LiveMCP) → **2,005** after mutation.
3. **Subset sampling.** From seed nodes, a DFS-style random walk over the graph collects a locally coherent subset (similar or dependent tools).
4. **Trajectory synthesis.** GPT-4o writes a task and a coarse plan for the subset, then role-plays a multi-turn dialogue: user turns, assistant tool calls, and **LLM-simulated tool responses** (no real execution).
5. **Label extraction.** For each step t where the assistant invokes candidate c: label = c; input = history (o_0 … a_{t−1}) with o_{t−1} as the current query. → **15,092** training instances.
6. **Training.** Qwen3-8B, LoRA r = 8 on all linear layers, 3 epochs, global batch 64, lr 1e-4 cosine, warmup 0.1, max 32,768 tokens, BF16.
7. **Inference.** Light Routing Agent: the reasoning model (fixed to Gemini-2.5-Pro for all router methods) calls the router tool to get a candidate, then the execution tool to run it; tool descriptions never enter the reasoning model's context.

**Verification / grading.** Training labels = the simulator's own tool choices. Human check: 100 synthesized trajectories annotated by experts, Cohen's κ = 0.93 with the synthesized labels. That validates "the label is a sensible tool for this step", not that the simulated tool outputs are faithful. Evaluation: MCP-Universe and MCP-Mark (easy mode), "accuracy", avg@5 at temperature 1. The paper does not say clearly whether accuracy is routing accuracy or end-task success; MCP-Mark per-server scores are multiples of 10, which suggests about 10 tasks per server (≈50 total), so a 14-point overall gap is ≈7 tasks.

**Scale & cost.** 627 → 2,005 tools; 15,092 training instances; 40+ agents → 156 agent-routing test samples (built by the same mutation + synthesis pipeline). Costs not reported.

**Key results.**
- Overall (router + Gemini-2.5-Pro executor): MCP-Universe 53.44 ± 0.57 vs Gemini-2.5-Pro as router 49.79, Claude-Sonnet-4 48.25, GPT-4o 47.41, base Qwen3-8B 46.14, ReAct (Gemini-2.5-Pro) 41.80, MiniLM (Q+H) 43.62, Qwen3-Embedding-8B 44.88, BM25 30.00. MCP-Mark 60.00 ± 2.00 vs best baseline Gemini-2.5-flash 48.00; Qwen3-8B 42.00.
- Scale/noise (MCP-Universe / MCP-Mark): merged multi-server pool 53.02 / 56.00; + mutated confusable tools 53.02 / 54.00; + real LiveMCP tools 52.59 / 56.00. Baselines fall more on MCP-Mark (Gemini-2.5-Pro 46→32 with LiveMCP noise).
- Ablations: without history 53→48 (Universe) and 60→52 (Mark); without mutation 47.12 / 44.00; vanilla fine-tuning (no history, no mutation) 46.70 / 42.00. So most of the MCP-Mark gain comes from the mutation-enlarged candidate pool.
- Agent routing (own 156-sample benchmark): 91.6%, trained on tool data only.
- Embedding with history (text-embedding-3-large Q+H) is *worse* than query-only (35.20 vs 40.95): naive history concatenation hurts dense retrieval.

**Failure modes & limits.** Authors: LoRA on 8B only; trained on tool data, agent routing is transfer. My critique:
- **Contamination:** the 627 seed tools include MCP-Universe's own tools, so the router has seen the evaluation tool set (plus mutants of it).
- The agent benchmark is generated by the same pipeline as training, so 91.6% is in-distribution.
- Small MCP-Mark sample (≈50 tasks) makes the headline gap fragile despite the ±2.00 std over 5 runs (std over runs is not uncertainty over tasks).
- Simulated tool outputs mean no error-recovery signal is real; the "history helps error recovery" claim rests on simulated failures.
- Mutants are never checked for being distinct and valid; some may be functionally identical to the original, which makes the "correct" label ambiguous.

**Tips & tricks.**
- Grow a tool catalogue with five mutation operators (usage extension, function enhancement, workflow chain, helper, parameter redesign) to create confusable distractors.
- Link each mutant to its parent in the tool graph; sample task tool subsets by DFS walks over similarity edges so distractors are near neighbours.
- Turn every multi-turn trajectory into per-step (history, current observation) → next-action examples.
- Do not append raw history to an embedding retriever; it hurt here.
- Report variance over runs and evaluate with injected real and synthetic distractor tools.

**Fit to Crucible Part 2.**
- **F1 / knob `distractors` and `tools` (§3.6).** Use the mutation operators to generate *distractor tools* inside a World App: near-duplicate tools with different semantics (a validator, a bulk variant, a wrong-domain variant). This is a structural hardness knob that changes the behaviour signature only when the solver must discriminate. Each distractor must still be real code over the App's schema and pass F1 blind tool tests and G13; otherwise it is a crash source, not a hard negative.
- **F7/F8.** A mutant tool that does the same thing as its parent breaks the oracle's uniqueness; record `mutated_from` on the tool and make G14 alternatives aware of it.
- **F12 / traces-to-tasks.** The per-step label extraction is a cheap way to turn student traces into step-level probes ("from this history, which tool next?"), useful for first-wrong-step analysis (§2(d)) rather than as RL tasks.
- **Avoid:** LLM-simulated tool responses as training ground truth; evaluating on a benchmark whose tools seeded the training catalogue (our G11 must check tool-catalogue overlap with held-out benchmarks, not only text overlap).
- **Interface implied:** `ToolSpec.mutated_from`, `AppBundle.distractor_tools`, knob `distractors: {count, operator mix}`.
- **Priority:** P2 (a distractor generator once the F1 App synthesis exists).


## 2409.00920 — ToolACE: Winning the Points of LLM Function Calling (Liu et al., 2024; arXiv v2 Jul 2025, ICLR 2025 per later citations [venue not stated in the text]; Huawei Noah's Ark / USTC / SJTU)
Read: FULL (v2 HTML: sections 1–5, Appendices A–H incl. rule table, hyper-parameters, case studies, extra ablations on data source / data types / complexity evaluator, ICL comparison, limitations).  Relevance to Part 2: medium. It is SFT data synthesis for single-call function calling with *simulated* tool responses, so it has no executable World or verifier. Two ideas carry over: (1) a **hierarchical API context tree** used to plan coverage and to measure diversity by cluster count, with an ablation showing diversity matters; (2) **learner-loss as a per-student complexity signal**, with an ablation showing the evaluator must be the learner itself. Both bear on §3.5 diversity and §3.6 per-student difficulty.

**Problem.** Function-calling data built from public APIs is narrow (few domains, simple, single-turn) and often inaccurate. ToolACE synthesises a diverse API pool, generates dialogs whose complexity is matched to the model being tuned, and filters them with rule and model checks.

**Method / algorithm.**
*1. Tool Self-evolution Synthesis (TSS)* → 26,507 APIs over 390 domains.
- **Speciation.** From API-related documents in pretraining data (manuals, API docs, product specs, tutorials), a frontier-LLM agent extracts a domain and all functionalities; children nodes are generated recursively → an **API context tree** (domain → functionality leaves such as "get stock price").
- **Adaptation.** For each new API, sample a subtree; the number of nodes sets how broad/specific the API is (one node = simple API; many nodes = rich API). Distinct subtrees per API keep functionalities distinct.
- **Evolution.** An LLM writes a new API from a sampled subtree + an example from an API example buffer; then "diversity indicators" mutate it: add functionality or parameters, add constraints, mutate parameter types (nested list-of-dicts etc.), change return values. New APIs go back into the buffer for the next generation.

*2. Self-Guided Dialog Generation (SDG).*
- Three LLM agents: **user** (requests, supplies info, follows complication instructions), **assistant** (call API / ask for info / summarise / answer without tools; uses a structured function-call thinking format), **tool** (simulates execution output from the API definition).
- Four data types: single, parallel, dependent, non-tool-use (irrelevance and missing-parameter cases); plus nested-argument cases.
- **Self-consistency:** each assistant action is sampled several times; only actions with a consistent decision across samples are kept.
- **Complexity evaluator = the model to be tuned, M.** Complexity H_M(x, y) = mean token NLL of response y given query x under M (Eq. 1). Loss rises with (a) number of candidate APIs, (b) number of APIs used, (c) query–description dissimilarity (Fig. 2).
- **Band calibration:** on a small prior set spanning complexity levels, samples M already answers correctly give a lower bound on loss (already mastered); samples whose loss stays high after fine-tuning give an upper bound (too hard). Exact numeric bounds not reported.
- **Self-guided complication:** if a sample is below the band, the user agent is told to need more APIs or to phrase further from the API description; if above, to simplify.

*3. Dual-Layer Verification (DLV)*, with human oversight.
- **Rule layer** (Appendix B): API definition is valid JSON Schema with required fields; called name in the tool list; all required params present; param formats match patterns (regex); dialog fields complete, response not too long, no invalid characters or mixed language; tool-response names match calls; role order correct; tool response follows a call. No real execution.
- **Model layer:** decomposed into separate expert-LLM sub-checks: hallucination detection (are argument values grounded in the query or system prompt?), consistency (does the reply complete the task and respect constraints?), tool-response check (is the simulated output consistent with the API definition?), plus repetition / junk filters.

*4. Training.* LLaMA-3.1-8B-Instruct, LoRA r = 16, α = 32 on all modules, lr 1e-4, warmup 0.1, cosine, batch 48, 3 epochs → ToolACE-8B. Also Qwen1.5 0.5B–7B for scaling.

**Verification / grading.** Correctness = schema/format rules + LLM judges; the tool is an LLM, so "executability" is format-validity, not behaviour. What can slip through: semantically wrong argument values that are grounded in the query but wrong for the API; simulated tool outputs that are plausible but inconsistent across turns; dependent calls whose second step does not actually use the first step's output. Evaluation is on BFCL (AST match and executable categories) and API-Bank.

**Scale & cost.** 26,507 APIs, 390 domains; 30 clusters at the top of the tree (used in the diversity ablation). The total dialog count is not stated in the main text; the complexity ablation draws bottom/middle/top 60,000 samples, which implies at least 180K samples. The public subset is ~11.3K (per Toucan's table, not this paper). Cost and frontier models used: not reported.

**Key results.**
- BFCL-v3 (leaderboard 2024-09-20): ToolACE-8B overall 59.22, rank 3 behind GPT-4-turbo FC 59.49 and GPT-4o FC 59.29; non-live AST 89.27, exec 90.07, live AST 73.21, multi-turn 14.37, relevance 85.37, irrelevance 83.81. xLAM-7b-fc-r 51.45.
- API-Bank: Call 75.94 (= GPT-4-0613), Retrieval+Call 47.41 (base LLaMA-3.1-8B 71.18 / 37.04).
- Same 25K samples, same base (Table 6): ToolACE 58.19 overall vs xLAM 40.51 vs ToolLLM 24.90; irrelevance 86.42 vs 11.87 vs 4.41.
- **Data-type ablation (25K fixed):** without parallel → overall 50.60, multi-turn 1.75; **without multi-type (non-tool / clarification) → overall 42.71, irrelevance 6.99**, multi-turn 1.75; without dependent 57.97; without nested 57.19.
- **Verification ablation** (figure only): rule layer beats no verification; both layers best, despite fewer samples.
- **Complexity ablation** (figure only): the middle-loss 60K subset slightly beats easy and hard subsets.
- **Evaluator ablation (Table 8):** learner LLaMA-8B as its own evaluator 59.22; Qwen1.5-7B as evaluator 57.61; Qwen1.5-14B 57.67. A stronger evaluator drops samples it finds easy but the learner does not (non-live AST falls, live AST rises); a weaker one keeps samples too easy for the learner.
- **Diversity ablation** (figure only): APIs from 6 / 14 / 30 clusters, ~30K samples each; overall accuracy rises with cluster count, and relevance detection gains most.
- Scaling on Qwen1.5 0.5B–7B: gains at every size; 0.5B and 1.8B go from near-zero structured output to usable.
- 3-shot RAG in-context learning on LLaMA-8B underperforms zero-shot fine-tuning and even the zero-shot base, as the model copies tools from the examples.

**Failure modes & limits.** Authors: per-sample loss evaluation scales with model size and sample count; one-round non-uniform sampling may keep the model in its comfort zone (they suggest iterative rounds); general abilities still trail GPT-4. My critique:
- Simulated tool responses: no state, no consequences, so multi-turn scores stay low (14.37) and nothing teaches recovery from real tool failures.
- The band bounds are heuristic and unreported; the complexity gain is "slight" and shown only in a figure.
- NLL of a *reference* response measures how surprising the teacher's answer is, not whether the learner would fail the task; a correct-but-unusual phrasing raises loss. For RL we need pass rates, not loss.
- Diversity ablation confounds cluster count with API novelty relative to BFCL.
- No dedup or contamination check against BFCL / API-Bank APIs.

**Tips & tricks.**
- Build a domain → functionality tree first, then sample subtrees per API; count coverage as filled tree nodes or clusters.
- Mutate APIs along explicit axes (new params, constraints, nested types, changed returns) to get structural, not textual, variety.
- Include non-tool and missing-parameter cases; without them irrelevance collapsed to 7%.
- Sample each assistant decision several times and keep only consistent decisions.
- Decompose model verification into narrow sub-questions (grounding of argument values, task completion, tool-output vs schema) instead of one "is this correct?" prompt.
- Use the learner itself, not a proxy model, to score per-sample difficulty; a stronger or weaker proxy shifts the kept set the wrong way.

**Fit to Crucible Part 2.**
- **F0 DomainSpec / §3.5 grid.** The API context tree is a ready shape for `skill_tree` and the `world_facet` axis: domain → functionality leaves, with coverage = filled leaves. Adopt the "subtree size = API breadth" idea as a knob on App richness in F1.
- **F9 per-student difficulty.** The evaluator ablation is direct evidence for our rule that p̂ is keyed by student: difficulty measured on a different model, stronger or weaker, picks the wrong tasks. For the cheap predictor before F9, consider the student's **NLL on the oracle's action program** (tool calls + args, rendered as the student would emit them) as a feature: it costs one forward pass per task instead of k rollouts. Validate it against measured p̂ on our kept tasks before relying on it (their evidence is SFT-loss, not RL pass rate).
- **Self-guided complication ↔ knobs.** Their "more APIs" and "further from the description" map to our `tools` and `obfuscation` knobs; "simplify" maps to ladders. Our rule that a harder variant must change the behaviour signature applies: "further from the description" changes only text, so it counts only as obfuscation, not as a new task.
- **F5/G-gates.** The rule layer is a subset of our F1 tool tests and G13; the hallucination sub-check (argument values grounded in visible info) is the LLM analogue of G8 communicability. Our executed version is stronger; keep theirs only as a cheap pre-filter on proposals.
- **Task types (§3.3).** Their non-tool / missing-parameter data is evidence that infeasible and clarification-needed tasks must be in the pool from day one.
- **Avoid:** simulated tool outputs; loss as the only difficulty measure.
- **Priority:** P1 for the student-NLL predictor feature (cheap experiment on existing tasks); P2 for the tree-shaped DomainSpec, which §3.5 already covers.


## 2511.03773 — Scaling Agent Learning via Experience Synthesis (Chen et al., 2025; arXiv v2 10 Nov 2025; Meta Superintelligence Labs / FAIR, UChicago, UC Berkeley) [DreamGym]
Read: FULL. Sections 1–7, Appendix A (per-environment settings) and Appendix B (theorem and proof) from the arXiv HTML. The HTML dropped Appendix C (prompts), so I read pp. 19–26 of the local PDF (`crucible/docs/research/papers/2511.03773.pdf`) for the task-variation, variation-selection, AX-tree mapping and agent prompts.  Relevance to Part 2: high as a design reference and as a warning. It is the clearest case of an **LLM-simulated environment that also produces the reward and the tasks**, trained in the RL loop with a reward-variance curriculum. Its curriculum rule and sim-to-real warm start are reusable. Its core choice, letting a model decide transitions *and* success, breaks our rule that "the LLM proposes; execution decides". So in Crucible it can only be a cheap proxy or a warm-start tier, never the ground truth.

**Problem.** Real-environment RL for LLM agents is costly (long rollouts, slow or unsafe environments, no resets), short of tasks (fixed instruction sets whose feasibility needs human checking) and noisy in reward (dynamic web pages, broken evaluators). DreamGym replaces the environment with a reasoning "experience model" that synthesises next states and rewards in an abstract text state space. A replay buffer grounds it and a task generator builds a curriculum.

**Method / algorithm.**
*Components.*
1. **Experience model M_exp** (Llama-3.1-8B-Instruct in all main results). The state space is an abstract, cleaned text form. For WebArena, an LLM maps the AX-tree to 3–10 relevant actionable elements plus a short reasoning block: page description, progress so far, next-action analysis. Each step's input:
   - the full interaction history {(s_i, a_i)};
   - the task instruction τ;
   - top-k past transitions retrieved from the replay buffer by cosine similarity of an embedding φ(s_t, a_t). The encoder and k are not specified.
   The output is a CoT reasoning trace R_t, then (s_{t+1}, r_{t+1}). Reward is outcome-only: r = 1 at the final step if *the experience model judges* the task complete, else 0. Invalid actions go to a failure state with r = 0.
2. **Training M_exp (SFT).**
   - Offline transitions (s_t, a_t, s_{t+1}, r_{t+1}).
   - A strong teacher LLM annotates each transition with a reasoning trace R*_t that explains why the action leads to that next state and reward.
   - The loss is joint: L = −log P(R*_t | s_t, a_t, H_t, D_k) − log P(s_{t+1} | s_t, a_t, R*_t, H_t, D_k).
   - Offline data per environment:
     - **WebShop:** 1,600 human demos plus 2,000 oracle/random-policy trajectories. Test tasks are excluded.
     - **ALFWorld:** 3,200 expert demos plus 2,000 oracle/random.
     - **WebArena:** 4,800 trajectories. These are successful demos of top leaderboard agents (IBM CUGA, ScribeAgent, Learn-by-Interact, AgentOccam, all with AX-tree observations) plus a high-performing agent and a random policy.
3. **Replay buffer.** Seeded with the offline data, then enriched with the synthetic trajectories generated online. The model and the agent "co-evolve": retrieval stays close to what the current policy visits.
4. **Curriculum task generator M_task.** It shares parameters with M_exp.
   - Task value is the group reward variance: V_τ = (1/n) Σ_i (r_i − r̄)². With binary rewards this equals p(1 − p), maximal at p = 0.5.
   - For GRPO the group is the training group. For PPO, tasks are clustered by an embedder and each cluster is a group.
   - High-V_τ tasks are seeds for new variations, τ_t = M_task({τ^i_{t−1}}). A hyper-parameter λ caps the share of synthetic tasks per iteration, so the original task distribution stays covered. Its value is not reported.
   - The prompts (App. C) first generate K variations per seed. They keep the **same action type** (search, filter, upvote, purchase, delete…) and change entities, filters, time ranges or constraints, "slightly harder but solvable". A second prompt then **selects the single most challenging-yet-feasible variation** from the candidates and gives a short reason. For WebShop and ALFWorld the selection prompt is grounded in product attributes or room objects, so attribute combinations stay realistic.
5. **Training loop.**
   ```
   buffer ← offline data; tasks ← seed set
   repeat:
     for τ in sample(tasks): rollout π_θ against M_exp (s, r from M_exp; k neighbours from buffer)
     update π_θ with GRPO or PPO (Verl-Agent defaults)
     buffer += new synthetic trajectories
     seeds ← tasks with V_τ > 0, highest first
     tasks += select_best(M_task(seeds, K variants))   # capped by λ
   ```
6. **Sim-to-real (DreamGym-S2R).** Train purely in DreamGym, then run ordinary RL in the real environment with 5K real transitions. The same rule-based or small-model state mapping keeps the two state spaces aligned.
7. **Theory (App. B).** Under a trust-region update with per-state KL ≤ δ, the real improvement is at least (synthetic surrogate gain) − 4γV_max δ/(1−γ)² − 2(ε_R/(1−γ) + 2γR_max ε_P/(1−γ)²). Here ε_R is the sup reward error and ε_P the sup TV error of the transitions. The authors' point: only reward accuracy and "domain consistency" matter, not pixel- or HTML-level fidelity.

**Verification / grading.** In training, success is the experience model's verdict, and no executable check exists. Final evaluation runs in the real environments with their own evaluators (WebShop score, ALFWorld success, WebArena-Lite's 165-task functional checks). Quality of the simulator is judged by GPT-4o on 100 sampled trajectories, each scored 0/1/2 for consistency, diversity, informativeness and hallucination (Fig. 4, figure only).

What can be hacked:
- The policy is optimised against M_exp's reward. A lenient or inconsistent simulator can be exploited: the policy finds actions the simulator wrongly calls success. ε_R and ε_P are never measured.
- "Feasible" generated tasks are feasible only in the simulator. A task that has no solution on the real site can still show V_τ > 0 there.
- GPT-4o as the fidelity judge is another model opinion, not a measurement.

**Scale & cost.**
- Compute: 8 A100 nodes + 4 H100 nodes for all runs, baselines included.
- Real-data budgets: SFT 20K transitions, DPO 40K, GRPO/PPO 80K, DreamGym 0, S2R 5K (a trajectory is about 10 steps).
- On WebArena, DreamGym reaches its gains at about 1/3 to 1/5 of the real-RL training time, counting rollout time and GPU hours (figure only).
- The WebArena real-RL baseline had only 4 AWS servers, so at most 4 parallel sessions. Servers were reset by hand after each sweep. Known-broken evaluator verdicts were kept as-is.

**Key results** (success %, Table 1; backbones L3.2-3B / L3.1-8B / Q2.5-7B).
- **WebShop.**
  - GRPO real 62.1 / 65.0 / 66.1; DreamGym-GRPO 59.3 / 63.9 / 68.3; S2R-GRPO 70.5 / 75.0 / 72.1.
  - PPO real 59.9 / 64.2 / 68.1; DreamGym-PPO 60.5 / 58.1 / 65.0; S2R-PPO 66.0 / 63.9 / 73.7.
  - SFT 32.0 / 35.1 / 32.9.
- **ALFWorld.**
  - GRPO real 65.3 / 70.9 / 79.8; DreamGym 62.1 / 66.3 / 71.0; S2R 65.0 / 75.9 / 82.4.
  - PPO real 47.0 / 72.9 / 81.1; DreamGym 40.5 / 70.8 / 72.7; S2R 49.1 / 73.3 / 79.9.
  - Pure synthetic trails real RL by up to 8–9 points (Q2.5-7B).
- **WebArena-Lite.**
  - GRPO real 7.3 / 6.1 / 6.1; DreamGym 13.3 / 9.1 / 12.7; S2R 13.9 / 9.7 / 11.2.
  - PPO real 6.7 / 4.8 / 7.3; DreamGym 14.5 / 10.9 / 10.0.
  - The abstract and §5.2's "over 30%" must be a relative gain. The absolute rates are 9–15%, so the sentence that agents exceed 30% success does not match the table.
- **Ablation (Table 2; WebShop / WebArena).** Full 63.9 / 13.3; without replay 59.2 / 9.7; without experience reasoning 55.8 / 7.3; without task generation 57.3 / 7.3. Without the generator, WebShop curves plateau early.
- **Simulator ablation (Fig. 4).** Without history, consistency falls. Without reasoning, informativeness falls and hallucination rises.
- **Data efficiency (Fig. 5).** The 8B model passes 50% on WebShop with 10K offline transitions. The 3B reaches about 55% with 20K. On WebArena, WebDreamer starts ahead (about 13%) at low data and the gap closes with more data.
- **Transfer (Fig. 3 middle).** WebShop-trained policies beat WebArena-SFT on WebArena, and the reverse also holds. Web → ALFWorld transfer collapses.

**Failure modes & limits.** Authors: single-environment models only; a universal multi-environment model is future work. My critique:
- **Ground truth is a model's opinion.** Training reward, transitions and task feasibility all come from one fine-tuned 8B model. The theorem's bound is vacuous in practice: with sup errors, γ near 1 and any non-zero ε_P, the penalty exceeds any plausible gain. They never estimate ε_R, for example by replaying synthetic-success trajectories in the real environment.
- **Unfair WebArena baseline.** Real RL was starved: 4 sessions, manual resets, broken evaluator labels kept. The headline gap is partly infrastructure, not method.
- **Possible contamination.** The WebArena simulator was trained on successful leaderboard trajectories. Nothing says demos on the 165 Lite evaluation tasks were excluded. WebShop explicitly excluded test tasks; WebArena did not say so.
- In two of three environments, pure synthetic training *trails* real RL. The robust win is S2R, which is a warm start and needs a real environment anyway.
- Curriculum variations keep the action type. That widens surface coverage but cannot create new skills, so diversity is bounded by the seed set's action types.
- With an 8B simulator, the policy can learn the simulator's quirks. Smoother training curves may reflect a more predictable, easier world, not better learning.

**Tips & tricks.**
- Rank tasks for curriculum by group reward variance, p(1 − p), and generate variations only from tasks with mixed outcomes.
- Cap the share of generated tasks per iteration (their λ) so the seed distribution is never crowded out.
- Generate K variations under a "same action type, different entity/constraint/time" rule, then have a separate call pick the hardest feasible one with a stated reason.
- Ground the selection prompt in real world facts (product attributes, room objects) to block impossible combinations.
- Compress observations to a short list of relevant actionable elements plus progress notes. This made the simulator data-efficient, and the same idea can shrink agent context.
- Put retrieved real transitions in a simulator's context; it cut hallucination in their ablation.
- Add teacher-written "why this action led to this state" traces when distilling a simulator.
- Use a cheap simulated stage as a warm start, then a small real-environment RL stage.

**Fit to Crucible Part 2.**
- **Where it must not go.** It cannot be a reward source for kept tasks or the benchmark. Design rule 2 (execution decides) and F4's solution-by-construction both exclude a simulator-judged reward. An "experience model" verdict is an LLM opinion and would need G3–G12 on every task, and simulator transitions cannot pass G2 determinism.
- **F9 cheap predictor (§3.6, §3.10 lever 2).** A DreamGym-style simulator of a *specific* World, distilled from that World's real traces, can estimate a student's p̂ before real rollouts. Rank candidates by simulated V_τ = p̂(1 − p̂), then spend real student runs only on the top of the queue. The cost is one small model per World. Measure its rank correlation with real p̂ on the 13 kept tasks before trusting it.
- **Task loop (§3.8).** Their curriculum *is* our band rule. V_τ > 0 with a maximum at p = 0.5 matches the [0.1, 0.6] admission band. The variation generator fits the task loop: seed = in-band task, generate K variants, pick the hardest feasible one. Adopt it **with our guards**:
  - every variant goes back through F4–F10 (oracle in the real World, gates, real p̂);
  - "same action type" variants must change the behaviour signature on the knob changed, or F8 rejects them as duplicates;
  - λ becomes a quota on the share of variants per cell and family (F11).
- **World loop / fidelity.** If any LLM-simulated App is ever used (e.g., for an un-buildable long-tail service), add a **sim-fidelity gate**: replay N synthetic-success trajectories in the real App, or re-grade them with the executed verifier. That measures ε_R directly. Block the simulator if disagreement exceeds a bound. This is the missing experiment in the paper.
- **Curriculum stage (Part 1 RL path).** S2R suggests a two-stage schedule: a cheap tier first (e.g., LLM-simulated Apps or reduced Worlds), then the real World at a small budget. This is P2 and should be measured by our own runs, not assumed.
- **Observation abstraction.** Their AX-tree refinement prompt is a reusable component for CUA/Harbor Worlds (§2(b)) to keep student context short. It must not hide information the verifier relies on (G8 communicability).
- **Interface implied:** `SimWorld(World) {experience_model_ref, buffer_ref, fidelity: {eps_R_hat, n_replayed}}`; `TaskVariant {parent, knob_changed, action_type_preserved, selection_reason}`; `DifficultyRecord.predictor_score` from the simulated V_τ.
- **Priority.** P1: variance-seeded variant generation inside the task loop, under F6–F10. P2: the simulator as a predictor and as a warm-start tier, gated on measured fidelity. Avoid: simulator-judged reward for any task in the pool.


## 2511.01824 — Simulating Environments with Reasoning Models for Agent Training (Li et al., 2025; arXiv v1 3 Nov 2025; UW / Microsoft / CMU) [Simia: Simia-SFT and Simia-RL]
Read: FULL. I read sections 1–4, Limitations and Appendices A–F from the arXiv HTML. The HTML drops the prompt figures (Figs. 9–14), so I read their text from the local PDF (`crucible/docs/research/papers/2511.01824.pdf`, pp. 12–16).  Relevance to Part 2: high as a counter-design, medium as a source. It is the extreme of "no environment at all". A frontier reasoning model (o4-mini / GPT-5) simulates the user, the tools and the reward from a tool spec, the policy rules and one seed trajectory. It is used for SFT data amplification and for GRPO. The results are real but come with heavy caveats: in-distribution seeds, a judge-only reward, short RL runs. One finding matters for our World design: **RL against the simulator beat RL against the real OfficeBench**, apparently because the simulator returned explanatory error messages.

**Problem.** Agents struggle in "simple-task / complex-environment" settings: many tools, schemas and edge cases. Building a bespoke environment per domain is heavy and brittle. Simia asks whether an LLM can stand in for the environment, both to make SFT trajectories and to provide RL feedback and reward.

**Method / algorithm.**
*Simia-SFT: end-to-end trajectory amplification* (4 stages).
1. **LLM pre-filter of seeds** on three checks: completeness (the task finished, with all parts present), logic (reasoning and actions consistent) and format (roles, JSON).
2. **Prompt.** Tool specifications, policy rules and constraints, input/output format, and **one seed trajectory** as the exemplar. Main prompt rules (Fig. 9):
   - use only the listed tools;
   - keep the exemplar's format and **about the same number of turns**;
   - write a **completely new scenario that needs similar problem-solving patterns**;
   - optionally put 1–3 sentences of reasoning in `<think>` inside function-call turns only;
   - start with a HUMAN turn.
   The OfficeBench prompt (Fig. 10) instead asks to **extend a 1-app task into a natural multi-app workflow**. It also hard-codes "error-avoidance" practices: `ls` before file operations, `mkdir -p`, `switch_app` before an app's API, no exact file names in the task (discover them with shell), end with `finish_task`.
3. **Simulation.** One call writes a whole trajectory: user turns, assistant reasoning, tool calls and simulated observations, until done. Temperature 1.0 and several passes per seed give diversity. Synthesizers: GPT-5 and o4-mini.
4. **Rule post-processing:**
   - repair malformed JSON arguments;
   - drop trajectories that call tools not in the list, or that cannot be parsed;
   - normalise calls to Hermes XML `<tool_call>` or JSON;
   - swap the system prompt for the target deployment's.
   No semantic correctness check is applied. The authors say quality is "bounded by the synthesizer's semantic correctness".

*Seeds → outputs.*
- APIGen-MT: about 1.5k Airline-like and 3.5k Retail-like trajectories (τ-bench in-distribution) → **90k**.
- AgentTuning: 195 OS, 351 WebShop, 122 Mind2Web → **15k**.
- OfficeBench: 76 one-app tasks with o4-mini trajectories → **30k** multi-app.

*SFT.* LLaMA-Factory, full-parameter, lr 5e-6, 2 epochs, 8×A100-80GB, per-device batch 2, cosine, max length 16,384. Trajectories are split at assistant turns, with loss on assistant tokens only. Backbones: Qwen2.5 (1.5B–32B), Qwen3-8B, Llama-3.1-8B, Llama-3.2-3B.

*Simia-RL: GRPO in a simulated environment.*
- **Environment-feedback prompt, OfficeBench (Fig. 11).** Inputs: the current app, the allowed action formats, response guidance, **the task's testbed data**, the task and the action history. The simulator first checks the action is valid (right app, right name, all parameters) and fails it at once if not. Otherwise it simulates a realistic result and returns JSON `{success, observation}`.
- **Feedback prompt, APIGen-MT (Fig. 13).** It plays both the human and the tool. It enforces `<tool_call>` wrapping and valid JSON, and returns an error message otherwise. A tool response reuses matching results from the reference conversation when they exist and otherwise invents plausible results. The simulator emits `[TERMINATE]` when done.
- **Reward prompt, OfficeBench (Fig. 12).** Judge success from the full history and whether the outcome meets the requirements, explicitly not from the model's claim of "task finished". Output JSON `{reasoning, evidence, task_success, confidence}`.
- **Reward prompt, APIGen-MT (Fig. 14).** Four criteria: understood the task, followed the system-prompt policies, used correct tool-call format, handled tool responses. Completing the request while violating policy scores 0, and any malformed tool call scores 0. Output `{reasoning, reward ∈ {0,1}}`.
- Simulator and judge: o4-mini at temperature 1.0 with a 60K context.
- RL settings: RAGEN on VeRL, GRPO, 16 rollouts per task, **64 steps**, lr 1e-6, KL 0.001, clip-high 0.28, mini-batch 32, policy temperature 0.7, max 25 agent turns (40 interaction rounds), 12K context, 8×A100.
- RL always follows SFT. One setting trains on OfficeBench 1-app and evaluates on 2- and 3-app. The other trains on APIGen-MT-5k and evaluates on τ²-Bench.

**Verification / grading.** In training, success is an o4-mini verdict on a trajectory whose observations o4-mini also wrote. No executed state exists, so the judge rules on a story it partly authored. Evaluation runs in the real benchmarks: τ²-Bench with a GPT-4.1 user at temperature 0, OfficeBench, AgentBench.

What can be hacked or go wrong:
- The policy can satisfy the judge's rubric (format, apparent policy adherence) without reaching a real end state.
- The simulator may invent tool results that make a wrong plan look successful.
- Consistency across turns rests only on context. There is no state store, so the simulator can contradict itself or quietly agree with impossible requests.
- The paper claims the simulator needs no testbed data, yet the OfficeBench feedback prompt includes `{testbed_text}` and the APIGen-MT prompt includes the seed's reference conversation. So it is grounded by real task data where the authors had it.

**Scale & cost.** 135k synthetic trajectories in total (90k + 15k + 30k). RL used 64 GRPO steps on 8×A100. Simulator token cost and dollar cost are not reported. Every RL step calls o4-mini once per environment turn plus once for the reward, so cost grows with rollouts × turns.

**Key results.**
- **τ²-Bench Airline / Retail / avg.**
  - Simia-Tau Qwen2.5-32B 56.0 / 61.7 / 58.9. For comparison: GPT-4o 54.2 avg, GPT-4.1 59.1, o4-mini 63.2, GPT-5 67.6, xLAM-2-70B 56.3, base Qwen2.5-32B 36.9.
  - Qwen3-8B: base 23.8 → APIGen-MT-5k SFT 35.7 → Simia SFT 49.3 → SFT + Simia-RL 51.0 (49.0 / 52.9).
  - Qwen2.5-7B: Simia SFT 40.1 → +RL 42.8.
- **Pass^k on τ²** (Qwen2.5-32B Simia vs xLAM-2-70B).
  - Airline 56.0 / 48.0 / 46.0 vs 49.3 / 40.0 / 34.0.
  - Retail 61.7 / 47.7 / 38.6 vs 63.2 / 51.5 / 46.5. Retail robustness is *worse* than xLAM-2-70B.
- **OfficeBench 2-app / 3-app / avg** (trained only on 1-app seeds).
  - Qwen2.5-7B: base 19.2 avg; seed-only SFT 25.9; Simia SFT 42.6; SFT + sim-RL 49.6 (64.7 / 34.5).
  - **RL in the real OfficeBench environment reached only 60.8 / 28.6**, so RL in the simulator beat RL in the real environment. Their case study puts this down to explanatory feedback: the simulator explained a conflict with a lunch break, where the real environment gave a fixed terse failure.
  - Qwen3-8B base scores 2.0 because thinking-mode loops run to excessive length. Simia SFT brings it to 44.0.
- **AgentBench avg (Mind2Web / OS / WebShop).** Simia-AB Qwen3-8B 42.6 vs seed-SFT 40.6 vs base 32.8, a small gain over seed SFT. WebShop 67.3 vs 68.6 for seed-only, so no gain there.
- **Real vs simulated data at equal size** (Fig. 6, figure only). Comparable on OfficeBench and AgentBench, better on τ². Larger simulated sets keep improving.
- **Synthesizer choice** (15k, Qwen2.5-7B). o4-mini ≈ GPT-5 on most suites. GPT-5 is much better on τ²: Retail 40.2 vs 28.7, Airline 30 vs 20.
- **Joint training** of Qwen3-8B on all three sets: τ² Airline 16.0 → 47.7; the average across suites beats GPT-4.

**Failure modes & limits.** Authors: a finite set of domains; simulation may carry distributional bias against real deployments. My critique:
- **"Environment-agnostic" holds for engineering only.** The τ² gains come from 90k trajectories amplified from APIGen-MT seeds, which were themselves built in τ-bench's own Airline and Retail environments with their policies. The training data is in-domain for the evaluation. The honest OOD tests are OfficeBench 1-app → multi-app (same apps) and AgentBench, where the gain over seed SFT is 2 points.
- **No semantic filter.** Trajectories can show policy violations or invented tool facts and still be kept. SFT gains do not show the data is correct.
- **The RL evidence is thin.** 64 steps, gains of 1.7–7 points, and no measurement of reward agreement between o4-mini and the real evaluator. The sim-beats-real result is one benchmark, one backbone, no seeds, figure plus one table.
- **Cost** is never reported, though frontier-model calls per environment step are the obvious scaling cost of Simia-RL.
- **Retail pass^3 below xLAM-2-70B** suggests simulated data trains less consistent behaviour.

**Tips & tricks.**
- Pre-filter seeds with three LLM checks (complete, logical, well-formatted) before amplifying them.
- Put the full tool spec, policy rules, I/O format and one seed in the synthesis prompt; ask for a new scenario with "similar problem-solving patterns" and a similar turn count.
- For compositional generalisation, prompt to extend single-app seeds into natural multi-app workflows (1-app → 2/3-app gave large gains).
- Keep the task free of exact file names so the agent must discover them; bake known failure-avoidance practices into the synthesis prompt.
- Repair JSON and validate tool names in post-processing; swap in the deployment system prompt.
- In an LLM reward prompt, forbid judging by the agent's own "task finished" claim and ask for evidence plus a confidence.
- Make the simulator reject malformed or out-of-spec actions *first*, before simulating results.
- Explanatory error messages in the environment can speed RL markedly (sim 64.7 / 34.5 vs real 60.8 / 28.6).

**Fit to Crucible Part 2.**
- **Not as a ground-truth source.** A Simia reward is a model judging a model-written trajectory, so it fails design rules 2 and 3. No Simia-RL episode can stand in for a TaskContract grade.
- **World design: an error-informativeness knob.** The sim-beats-real result is the one structural lesson. Our Apps' tool errors should say *why* an action failed (violated constraint, missing precondition, conflicting record) without revealing the solution. Make this an `observability`/`hints` knob value in the DomainSpec: terse, explanatory, or hinting. That links to Environment Tuning's hint-giving errors (§3.6 ladders). Two cautions:
  - Explanatory errors may lower difficulty and must not break G8/G9: an error must not hand the answer to a no-World solver.
  - Measure the effect in our own RL runs; do not assume it.
- **Part 1 warm start (SFT before RL).** Simia-SFT is a cheap way to amplify our *kept training tasks'* oracle traces into SFT data for the student, using the "new scenario, similar pattern" prompt. Rules:
  - never seed from held-out families or the sealed benchmark (G11, pool splits);
  - stamp provenance;
  - prefer replaying each amplified trajectory in the real World and keeping only those that reach a goal state (the executed filter Simia lacks).
  This is P2.
- **F3 proposal amplification.** The OfficeBench "extend 1-app to multi-app" prompt is a good F3 template for `cross_app_flows` cells. The proposal must still get an F4 oracle in the real World, and F7 must confirm the claimed multi-app signature.
- **People-as-data / simulated users.** The APIGen-MT simulator plays both the human and the tools. Keep these roles apart: the user simulator is an `ActorSpec` with hidden intent and known info (G8), and tools run as code.
- **F10 judges.** The reward prompt's "evidence + confidence, not the agent's claim" pattern is reusable for failure-labelling judges in F9/F10, which are diagnostic, not reward.
- **Interface implied:** `DomainSpec.knobs.observability ∈ {terse, explanatory, hinting}` applied to tool error rendering. An `Amplifier(seed_trace) → candidate trace` whose outputs must be replayed through `TaskContract.grade` before use.
- **Priority.** P1: the error-informativeness knob and a small RL A/B on one World. P2: SFT amplification under executed replay. Avoid: an LLM-judged reward on pool tasks.


## 2508.08791 — Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments (Ye et al., 2025; arXiv v3 Apr 2026; Fudan + ByteDance Seed) [FTRL]
Read: FULL. Main text, App. 8 (datasets, metrics), App. 9 (per-scenario tables 6–9), App. 10 (case studies), App. 11 (all five construction prompts) from the arXiv HTML text.  Relevance to Part 2: medium. It is a small, clean example of **solution-first tool synthesis**: start from a question with known sub-answers, write one tool per sub-question, merge, inflate, then generate Python code that is guaranteed to return the planted answer. Its reward-shaping ablation (precision vs. completeness) is directly reusable. But its "World" is a set of stateless lookup functions over hand-written factual questions, with no database, no state change and no people, so it covers only Crucible's read-only answer-task corner. The verifier is substring matching, which is weak.

**Problem.** RL for tool use needs (a) stable environments (online APIs rate-limit and drift) and (b) a verifiable reward that does not lean on an LLM judge. FTRL builds local, code-only tool environments from questions and scores the policy only from environment feedback.

**Method / algorithm.**
*Data model.* An environment = user input U (question q, answer a, sub-questions q_1..q_n with intermediate answers a_i) + toolset T. Each tool t_i = document d_i (name, description, parameters with name/type/description/required flag) + implementation c_i.

*Five construction stages* (GPT-4o, temperature 0, max_tokens 2048):
1. **Scenario decomposition.** Four logical structures of sub-questions: single-hop; parallel single-hop (independent q_i); multi-hop (q_{i+1} needs a_i); parallel multi-hop (mixed). **User inputs per scenario are written by hand** (the paper says "manually construct"); the case studies show factual questions about late-2023 events.
2. **Document generation.** For each sub-question q_i, the model designs a tool whose function solves q_i, with parameters abstracted from q_i. The prompt asks for a JSON {analysis, tool-schema} and pushes for a "versatile" tool that handles similar queries and edge cases (two few-shot examples: distance calculator, forest locator).
3. **Function integration.** The model compares all n documents, groups functionally identical or overlapping ones and merges each group into one schema (output: list of {ids, merged document} or null). Yields m ≤ n tools.
4. **Complexity scaling.** Four strategies: functional generalisation (broader function), parameter expansion, parameter-type generalisation (strings → arrays, dicts, nested), and toolset extension (add non-essential distractor tools). The prompt requires backward compatibility with the original function.
5. **Localized deployment.** Each final document becomes a Python function (stdlib only): names, argument names and type annotations copied from the document. The (q_i, a_i) pairs are given as *prior conditions*: with valid arguments (and defaults) the return value must contain a_i as a substring; the function should also "simulate" other plausible returns and have error handling for bad inputs.
6. **Code verify** (only in App. 11, not described in the main text). A further prompt reads the question and the generated code and must emit a concrete call `func(param="value")` with values drawn from the question, or null if the function cannot solve it. Presumably calls are then executed and checked against a_i; the paper does not state the filtering rule or yield.

*Reward* (per sampled output o; p = number of tool calls in o, q ≤ p = sub-questions those calls solved, t = sub-questions still unsolved, a = final answer):
```
if p > 0:                 R = 2q / (p + 1)
elif o is None:           R = -0.5
elif format error:        R = -0.3
elif a is a substring of o: R = 1 / (t + 1)
elif t == 0:              R = 0.5
else:                     R = 0
```
My reading of "at each step, we record the trajectory, available tools, final answer, remaining unsolved sub-questions and answers" is that **each turn is its own training instance** and R is a per-step reward: a tool-calling step is scored on call precision and progress, a final-answer step on correctness, discounted by unfinished sub-questions. The paper does not say how "a sub-question is solved" is detected; it is most likely that the tool's return contains a_i. Note 2q/(p+1) with q = p = k equals 2k/(k+1), so correct parallel calls earn more than 1, which rewards parallelism.

*Training.* VeRL; Reinforce++ or GRPO; lr 1e-6; batch 512; mini-batch 32; 16 rollouts per update; max response per step 1024 (non-reasoning) / 8192 (reasoning); 3 epochs, **trajectories resampled with the current model at each epoch**; 8 A100. Students: Qwen2.5-7B/14B-Instruct, Qwen3-8B/14B in both modes.

**Verification / grading.** Fully programmatic: substring match of a_i in tool output and of a in the final answer. Hackable or noisy in several ways: (1) the planted function is keyed on argument values the generator chose, so a correct but differently phrased argument returns "not found" (the 7B case study fails on "Emir Nawaf…" vs "Nawaf…"); (2) substring match rewards answer-dumping; (3) the questions are real-world facts, so a model can answer from memory without tools (the reward then pays 1/(t+1) only when p = 0, which partly blocks this); (4) no check that distractor tools were *not* called beyond the precision term.

**Scale & cost.** Train: 2,215 questions (500 single-hop, 500 parallel single-hop, 500 multi-hop, 715 parallel multi-hop); average sub-questions 1.00 / 2.10 / 4.71 / 6.97; average tools per instance 8.00–10.92. In-domain test: 200 (50 per scenario), avg tools 9.26. No token cost, yield per stage, or rejection counts reported.

**Key results.** (Table 2; averages over in-domain Solve-F1, ToolHop AC, τ-bench Pass^1, RoTBench TS/PI/CF.)
- Qwen3-8B non-reasoning: avg 31.01 → 45.43 (GRPO); in-domain Solve-F1 23.48 → 49.54; τ-bench 10.13 → 23.35; ToolHop 28.54 → 38.93.
- Qwen2.5-14B: 34.33 → 44.25 (Reinforce++); τ-bench 16.74 → 26.83.
- Trained 8B/14B models beat every closed model on the *average* (best closed: GPT-4o 42.79, Claude-4-Sonnet 42.71), but this average is dominated by in-domain Solve-F1 and RoTBench TS. On τ-bench they stay far below (Claude-4-Sonnet 50.22 vs best FTRL 32.52).
- τ-bench gains happen although training has no user simulator or multi-turn dialogue.
- **Reward ablation** (Qwen2.5-7B, Fig. 4, numbers only in the plot): precision-only q/p leads to incomplete tasks; completion-only q leads to tool overuse; product q²/p is too discrete to train stably; 2q/(p+1) balances both.
- General benchmarks (MMLU, BBH, GSM8K, MATH, HumanEval, MBPP) roughly unchanged (figure only).
- Parameter analysis: top-changed modules concentrate in layers 0–2 MLPs (31 of 36 top-3 slots across 12 models).
- Reasoning mode helps multi-hop and hurts single-hop and RoTBench (e.g. Qwen3-8B RoTBench TS 75.52 non-reasoning vs 54.00 reasoning).

**Failure modes & limits.** Authors: the work trains invocation, not reasoning for tools. My critique: (1) the hand-written questions are the bottleneck and the diversity source; nothing forces coverage beyond the four structural scenarios; (2) tools are stateless lookups, so there is no state verification, no collateral, no write tasks; (3) per-scenario yields and the code-verify filter are unreported, so cost per kept environment is unknown; (4) RoTBench and ToolHop come from the same group, and gains on τ-bench airline are small and noisy (50 tasks); (5) no seeds or error bars; (6) the "lower-layer MLP" claim is descriptive, with no causal test.

**Tips & tricks.**
- Write the tool *from* the sub-question and its answer, then make the code return the answer: solvability by construction.
- Merge overlapping tools after generation to avoid one-tool-per-task leakage (the tool name otherwise gives away the step).
- Inflate tool complexity separately from task complexity: nested parameter types and distractor tools raise selection difficulty without new questions.
- Have a second model derive a concrete call from the question and the code; a null call flags an unusable tool.
- Use a precision × completeness reward of the form 2q/(p+1): it penalises spray-calling without zeroing partial progress.
- Penalise empty output (−0.5) more than format errors (−0.3).
- Resample trajectories with the current policy every epoch.

**Fit to Crucible Part 2.**
- **F1 (World / App synthesis):** the document → merge → complexity-scale → deploy chain is a lightweight template for curated tools. Adopt the *merge* and *distractor tools* steps as knobs (`tools`, `distractors` in the DomainSpec knob list). Avoid its stateless design: Crucible's tools must act over a seeded database (F2) and be checked by state.
- **F4/F5:** "code must return the planted answer for valid arguments" is a weak form of an Oracle plus answer check. Our version should execute the oracle program and freeze the gold answer, with the answer check in `verify/answers.py` doing normalised match, not substring.
- **G13 (interaction completeness) and G8 (communicable):** the argument-phrasing false negative in their case study is exactly what G8/G14 must catch: every argument value the oracle uses must be derivable from the instruction, and argument matching must be canonicalised (entity IDs, not free strings).
- **G9 (no shortcut):** their factual questions are answerable from memory. Our instruction-only probe (F10) should be run on any read-only task family; prefer synthetic entities in the World so memory cannot help.
- **Reward shaping (task contract `grade`):** take 2q/(p+1) as a candidate *auxiliary* milestone reward for multi-subtask tasks (q = milestones reached, p = state-changing calls), but keep the binary end-state verdict as the main reward.
- **Diversity (§3.5):** their four sub-question structures (single, parallel, chain, mixed) are a useful scenario-type axis for the coverage grid; measure it from the oracle's dataflow graph (F7), not from the label.
- Priority: **P2** overall; the reward ablation and the merge/distractor knobs are **P1** small adoptions.


## 2504.03601 — APIGen-MT: Agentic PIpeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay (Prabhakar et al., 2025; arXiv v4 Jul 2025; Salesforce AI Research) [xLAM-2-fc-r]
Read: FULL. Sections 1–6, App. A (benchmarks) and App. B (all five prompts: task generation, alignment judge, feedback summariser, user simulator, BoN user critic) from the arXiv HTML text.  Relevance to Part 2: **high (core)**. It is the canonical **blueprint-first** pipeline for multi-turn, simulated-user, stateful tool tasks: a task is (instruction q, ground-truth write actions a_gt, expected outputs o_gt), validated by execution, a policy-as-unit-test check and a judge committee with a reflection loop, then turned into dialogues by a simulated user, with rejection by state and output match. Almost every piece maps onto F3–F6 of our stage graph. Its weak points (LLM committee as the semantic gate, training and testing on the same τ-bench databases, no diversity control beyond samplers, no negatives) are exactly the gaps Crucible's design is meant to close.

**Problem.** Multi-turn agent data (a user who reveals intent gradually, policy-bound API calls over a stateful backend) is scarce and costly to label. Generating whole dialogues in one shot fails because one hallucinated step breaks everything and each turn depends on prior tool outputs. APIGen-MT splits the problem: first build and verify a *task blueprint* with ground-truth actions, then simulate the conversation that realises it.

**Method / algorithm.**

*Formalism.* POMDP (U, S, A, O, T, R). Actions are {tool_call, response}. Observations are tool outputs O_E or human messages O_H. The environment state is latent to both agent and user. The episode ends on a user stop token or a turn cap. Reward R(ΔS_E, a) depends on the cumulative state change and the agent's responses to the human.

*Phase 1: blueprint generation and validation* (instantiated on τ-bench retail and airline; 15 read and 13 write APIs in total).
1. **API dependency graph.** Nodes are APIs. An edge A → B exists if B's inputs can come from A's output **and** the pair is allowed by domain policy. Random walks on this graph propose plausible action sequences.
2. **Context samplers** feed the generator:
   - API sampler: picks the **write** APIs that form the core of a_gt. Read APIs are left out of a_gt on purpose, so the agent is free in how it gathers information.
   - Policy sampler: samples domain rules; complexity comes from the number of writes and the policy constraints touched.
   - Domain data sampler: samples user and order records with metadata (cost, time, attributes) to fit context.
   - Persona sampler: PersonaHub personas.
   - Example sampler: few-shot well-formed tasks for the sampled APIs.
   - The *sampling frequency of each sampler is randomised per iteration* to vary prompts.
3. **Generator** (gpt-4o and DeepSeek-V3) emits `<thought>`, `<instruction>` q, `<actions>` a_gt, `<outputs>` o_gt. Prompt rules: actions focus on database-modifying calls; for pure information requests, the generator reads the sampled user data itself and writes only the answer into o_gt (no tool call); give all parameters; obey policy; "do not copy the examples".
4. **Stage 1, action validation** (deterministic):
   - format check (tags present; tool calls valid JSON; outputs are strings);
   - execution check: run a_gt in the τ-bench environment, checking API names, argument names and types; record the cumulative state change as a git-style **diff_patch**;
   - **policy compliance check:** domain policies are translated into **Python unit tests** that run on the execution trace of a_gt. They catch cross-action violations (B invalid after A's state change) and return specific feedback.
5. **Stage 2, alignment validation** (committee of diverse LLM judges). Each judge sees the task, the tools and the diff_patch and gives 0/1 scores on four rubric items: **Correctness** (a_gt implements q), **Completeness** (q is detailed enough and fully addressed), **Satisfaction** (o_gt answers the information requests), **Creativity** (non-trivial, plausible, interesting). The judge is told the diff must contain *only* changes the intent requires. Output includes a reflection and a suggested correction. Majority vote per metric.
6. **Stage 3, final review and refinement.** Tasks with average score above a threshold are accepted (the threshold value is not given). Failures go to a **feedback summariser** that merges the judges' points into actionable feedback; the generator revises the task (Reflexion-style). Max reflection rounds: **3 for retail, 5 for airline**.
7. **Reverse task recombination** (compositional hardening):
   ```
   pick validated tasks T1, T2, … with the same user persona
   a_comb = a_gt1 ∘ a_gt2 ∘ …;  o_comb = o_gt1 ⊕ o_gt2 ⊕ …
   rerun the policy unit tests on a_comb      # e.g. return and cancel of the same order conflict
   generator writes one coherent q_comb for (a_comb, o_comb)
   re-validate from Stage 2 (skip Stage 1: parts were already executed)
   ```
   Note: skipping execution of the concatenated sequence is a shortcut; only the policy tests are rerun on the combined trace.

*Phase 2: simulated human–agent interplay.*
- The **user LLM** gets q (with persona) and rules: one line at a time; reveal only what the current step needs; never invent facts not in the intent; paraphrase rather than repeat the intent; stay in persona; emit `###STOP###` when satisfied. It does **not** see the APIs or the environment.
- The **agent** is gpt-4o in function-calling mode.
- **Rejection sampling:** keep a trajectory only if the end database state matches the state from a_gt *and* the agent's responses contain o_gt (r = 1). Up to **3 attempts per task**; the union of distinct successes is kept.
- **Stabilised user (BoN, N = 4, with self-critique).** For each user turn, sample 4 candidates; a judge scores each 0–10: full marks if specific item/order/personal details match the task description, lower for each wrong detail; normal conversational moves (asking, STOP) are fine; unhelpful replies ("I don't remember") score low. Keep the best.

*Training.* Filtered behavioural cloning: split each trajectory at every assistant turn; loss only on assistant tokens. Mixed with single-turn APIGen function-calling data and other xLAM agentic data. LLaMA-Factory, full fine-tune, DeepSpeed ZeRO-3, bf16, AdamW, ≤3 epochs, one H200 node. Bases: Llama-3.1/3.2-Instruct and Qwen-2.5-Instruct, 1B–70B. **No RL.**

**Verification / grading.**
- Blueprint: execution + policy unit tests (hard) and an LLM committee (soft, majority vote, unknown threshold).
- Trajectory: end-state equality with the a_gt state plus output inclusion, the τ-bench reward.
- Hackable or weak points: (1) the committee is the only check that q actually *requires* a_gt (intent alignment by opinion); (2) "creativity" is scored by models of the same families as the generator (gpt-4o and DeepSeek-V3 in both roles); (3) o_gt inclusion is substring-like and rewards verbose answers; (4) a single reference end state rejects valid alternatives (no alternative oracles); (5) since read APIs are not in a_gt, an information answer that the generator copied from sampled data may not be reachable by the agent through the tools if a read path is missing; nothing checks this communicability.

**Scale & cost.**
- Phase 1 task success rate **70% with the agentic feedback loop vs 28% without** (2.5×).
- Phase 2 trajectory success rate **67%**.
- Trajectories: 1–29 turns; average ~12 turns for gpt-4o, 7 tool calls and 6 user turns per trajectory.
- Released set: 5K trajectories (APIGen-MT-5k). The total number generated internally is not stated, nor are token or dollar costs.

**Key results.**
- **τ-bench** (pass@1, ≥5 trials, naive user, think tool only): xLAM-2-70b-fc-r 67.1 retail / 45.2 airline / 56.2 overall vs gpt-4o-2024-11-20 52.9, Claude-3.5-Sonnet(new) 60.1, Llama-3.1-70B-Instruct 38.2. xLAM-2-32b 54.6 (Qwen-2.5-32B base 24.7); 8b 46.7; 3b 38.2; 1b 21.8.
- **BFCL v3** (04/03/2025 leaderboard): 70b overall 78.19 (rank 1), 32b 75.83 (rank 2), 8b 72.83. Multi-turn accuracy: 70b 75.12, 8b 69.25, 3b 56.00, 1b 43.12, against gpt-4o FC 41 and o1 36.
- **Consistency:** pass^k curves on τ-airline: the 70b's pass^5 exceeds Claude's despite a lower pass^1 (figure only).
- **BoN user stabilisation** (τ-retail, 5 trials): gpt-4o mean 62.8 → 67.0, variance 11.1 → 2.6; xLAM-2-70b 67.1 → 68.8, variance 9.7 → 4.0.
- By task length (short/medium/long by Claude's turn percentiles over 8 trials): the 70b beats gpt-4o on long tasks and trails Claude; it needs more user turns than Claude because it gathers user details in stages.
- **No ablations** isolate recombination, committee size, samplers, or the policy unit tests on downstream score; the only pipeline ablation is the 28% → 70% yield.

**Failure modes & limits.** Authors: user simulation stays stochastic; failed trajectories are discarded though they could give contrastive signal; multi-stage validation is costly; no RL yet. My critique:
- **Contamination / in-distribution evaluation.** Tasks are generated from τ-bench's own APIs, policies and user/order databases and evaluated on τ-bench test tasks. Nothing in the paper says test users or orders were held out. The τ-bench gains are therefore in-domain; BFCL multi-turn is the better transfer evidence.
- The gate that matters most for intent (does success *need* the actions?) is an LLM vote.
- Diversity is controlled only by random sampler frequencies and persona; there is no coverage metric, dedup or behaviour signature.
- Recombination concatenates independent subtasks of one persona. That makes long tasks, but the parts do not interact (no dataflow between them), so it adds length more than reasoning.
- The 67% Phase-2 rate with gpt-4o as the agent means the kept data is biased toward what gpt-4o can do; the hardest 33% of blueprints never produce training data.
- The same model families generate, judge and act; no family separation.

**Tips & tricks.**
- Build the blueprint (instruction, write actions, expected outputs) before any dialogue; simulate dialogue only to realise it.
- Put only **write** actions in the ground truth; leave reads free so the agent's information-gathering path is not over-constrained, and grade by end state.
- Express domain policies as **unit tests over the executed trace**; they catch cross-action conflicts that judges miss.
- Give judges the **diff_patch**, not just the action list, and tell them extraneous changes fail.
- Feed failure reasons back to the generator through a summariser; this alone moved yield from 28% to 70%.
- Cap reflection rounds per domain (3 or 5).
- Randomise how much each context sampler contributes per prompt to vary tasks cheaply.
- Grow hard tasks by composing validated easy ones, and re-run policy checks on the composed trace.
- Stabilise the user simulator with Best-of-4 plus a critic that checks factual consistency with the hidden intent; it cut evaluation variance roughly 2.5–4×.
- User prompt rules worth copying: one line at a time, reveal only what the current step needs, never invent facts, paraphrase the intent, explicit stop token.

**Fit to Crucible Part 2.**
- **F3 (cell-grounded proposal):** their samplers (API, policy, data, persona, examples) are a concrete version of "cell + World slice + persona + K nearest". Adopt the **API dependency graph** (edges by dataflow *and* policy permission) as the `tool_graph` in `AppBundle`, and sample write-API subsets from it per cell. Replace their random frequencies with our MAP-Elites cell sampling and add the K-nearest "do not repeat" references they lack.
- **F4 (solution by construction):** a_gt over write APIs + o_gt is our `Oracle(kind=action_program)` + `gold_answer`. Keep their write-only ground truth; grade by end-state digest. Add what they lack: run the oracle twice (G2), and execute composed sequences rather than skipping Stage 1.
- **F5 / G5:** their policy unit tests are a direct template for `policies_as_code` and `path_constraints`. Add the negatives they do not have: mutation operators over a_gt (drop a step, wrong value, duplicate write, touch a distractor) must each fail the state checks.
- **F6 / F10 (gates and intent):** their committee rubric (Correctness, Completeness, Satisfaction, Creativity) can be an *advisory* realism reviewer at F5–F6, run by a family different from the proposer. It must not be the intent gate; F10's ablations (null instruction, instruction only) and G8 communicability must decide. G8 is especially needed here: their prompt lets o_gt be copied from sampled data the agent may not be able to read.
- **Simulated users (TaskContract.user):** adopt the user prompt rules and the **BoN-4 + consistency critic** as the default `ActorSpec` driver for multi-turn tasks, and measure user-induced variance per task (their table is the baseline: variance 11 → 2.6). Our G10 flake bound should be computed *with* the simulated user in the loop.
- **Difficulty (§3.6):** reverse recombination is a `count` / `horizon` knob. Prefer compositions with dataflow between parts (one subtask's output is an input of the next), so the behaviour signature (F7) changes beyond length; reject compositions whose signature is the union of the parts with no new edges.
- **Feedback loops (§3.8):** the reflection loop is a per-task repair round (F6 allows up to 3, same as their retail cap). Report the yield with and without repair per stage, as they did; it is the cleanest single number for our cost model.
- **Held-out discipline:** avoid their train-on-benchmark-DB pattern. Seed train and benchmark tasks from disjoint entity sets (separate `SeedBundle`s) and keep held-out families.
- **Phase-2 rejection with a frontier agent** = our frontier audit; it biases the pool toward frontier-solvable tasks. In Crucible, construction is the solvability proof and the frontier run is a sampled audit, so keep failed-by-frontier tasks if the oracle passes and F10 is VALID.
- Priority: **P0** for the blueprint-first contract, write-only oracle + end-state grading, policy-as-tests and the user-simulator recipe; **P1** for recombination as a knob and the committee as an advisory reviewer.


## 2507.20534 — Kimi K2: Open Agentic Intelligence (Kimi Team, 2025; arXiv v2 Feb 2026; Moonshot AI technical report)
Read: FULL for post-training (§3.1–3.3: agentic data synthesis, verifiable-rewards gym, self-critique rubric reward, RL algorithm and infra), evaluation (§4, App. C), App. B (tool-call template), App. F (critic rubrics), App. G; PARTIAL (read for a summary) for pre-training (§2: MuonClip, rephrasing, architecture, infra) and App. D–E.  Relevance to Part 2: **medium**. It is a frontier lab's description of an agentic task-synthesis pipeline at scale (3,000+ real MCP tools, 20,000+ evolved synthetic tools, thousands of agents, rubric-graded tasks, an LLM tool simulator with state) and of a mixed RL gym. It is useful as a **map of what a production pipeline contains** and for a few concrete practices (hierarchical domain evolution, moderate-difficulty filtering by SFT pass@k, a hack-check layer for instruction claims, critic refinement on verifiable rollouts). It gives **no yields, counts per stage, thresholds or ablations** for the synthesis pipeline, and its central grading choice (an LLM judge against an LLM-written rubric, over an LLM-simulated tool world) is what Crucible's "execution decides" rule forbids for ground truth.

**Problem.** Agentic skills (multi-step tool use, planning) are rare in natural data. Real environments are costly, private or hard to reach. The report aims to synthesise tool-use trajectories at scale for SFT and then run joint RL over verifiable and non-verifiable tasks.

**Model in one paragraph (summary only).** 1.04T-parameter MoE, 32.6B active, DeepSeek-V3-like with MLA; 384 experts (8 active + 1 shared, sparsity 48), 64 attention heads (half of DeepSeek-V3, to cut long-context inference cost: 128k context with 128 heads would cost 83% more FLOPs), 1 dense layer. Pre-trained on 15.5T tokens with **MuonClip** (Muon + weight decay + RMS matching + per-head **QK-Clip**: rescale W_q, W_k when the head's max logit exceeds τ = 100), with no loss spike; QK-Clip touched 12.7% of heads in the first 70k steps and then went inactive. Knowledge data is rephrased chunk-wise with fidelity checks: SimpleQA 23.76 (raw, 10 epochs) → 27.39 (1 rephrase × 10 epochs) → 28.94 (10 rephrases × 1 epoch). Context 4k → 32k → 128k via YaRN.

**Method / algorithm (the parts in Part 2's scope).**

*A. Agentic SFT data synthesis (§3.1.1).* Three stages.
1. **Tool repository.**
   - 3,000+ real MCP tool specs fetched from GitHub.
   - 20,000+ synthetic tools by **hierarchical domain evolution**: start from top categories (e.g. financial trading, software applications, robot control), evolve many specific application domains per category, then synthesise tools per domain with interfaces, descriptions and operational semantics. A t-SNE plot shows the two sources cover complementary regions; no coverage metric is given.
2. **Agents and tasks.**
   - Thousands of agents, each a synthesised system prompt plus a sampled tool combination.
   - Per agent, tasks from simple to complex, **each paired with a rubric**: success criteria, expected tool-use patterns, evaluation checkpoints.
3. **Multi-turn trajectories.**
   - **User simulator:** LLM personas with distinct communication styles.
   - **Tool simulator** ("functionally equivalent to a world model"): an LLM executes tool calls, **keeps and updates state** across calls, and injects **controlled stochasticity** (successes, partial failures, edge cases).
   - **Judge:** an LLM grades each trajectory against the task rubric; only successes are kept (large-scale rejection sampling).
   - **Hybrid real execution:** for coding and SWE, real sandboxes with test-suite pass rates as ground truth.

*B. Verifiable Rewards Gym (§3.2.1).* A Gym-like extensible framework with many task families:
- **Math, STEM, logic:** a tagging system raises coverage of under-covered domains; logic includes structured-data tasks (multi-hop table reasoning, cross-table aggregation) and puzzles (24-game, Sudoku, cryptarithms, Morse). **Moderate difficulty:** keep problems by the SFT model's pass@k (thresholds not given).
- **Complex instruction following:** hybrid verification (code interpreter for checkable constraints such as length; LLM judge for nuanced ones) plus a **hack-check layer** that detects claims of compliance without actual compliance. Instructions come from three sources: expert-written conditional prompts with rubrics, AutoIF-style agentic augmentation, and a fine-tuned model that writes instructions aimed at specific failure modes and edge cases.
- **Faithfulness:** a sentence-level judge (FACTS-Grounding-style) flags unsupported factual claims; used as a reward model.
- **Coding and SWE:** competition problems with judges, synthetic problems with human unit tests mined from pre-training data; GitHub PRs and issues turned into environments with executable unit tests, on Kubernetes with **10,000+ concurrent sandboxes**.
- **Safety:** human seed prompts by risk category; an attack model evolves jailbreaks against the target; a judge with a task rubric gives binary success.

*C. Self-Critique Rubric Reward (§3.2.2).* For non-verifiable prompts:
- The K2 actor samples responses; the K2 critic ranks them by **pairwise comparison** against three rubric sets: core rubrics (clarity and relevance; conversational fluency; objective, grounded interaction without flattery or metacommentary), **prescriptive anti-hacking rubrics** (no opening praise of the user; no sentences justifying why the response is good), and human-written task-specific rubrics. The critic may weigh rubrics against its priors.
- The critic was bootstrapped in SFT from open and in-house preference data.
- **Closed-loop critic refinement:** on-policy rollouts from *verifiable* prompts continuously update the critic, so objective signals calibrate subjective judgement.
- Authors' stated side effect: the rubric favours confident, unhedged answers, so the model can overstate certainty.

*D. RL algorithm (§3.2.3).* K1.5's objective, per prompt x with K samples:
L = E_x [ (1/K) Σ_i ( r(x, y_i) − r̄(x) − τ log π_θ(y_i|x)/π_old(y_i|x) )² ], with r̄ the group mean, optimised with Muon. Additions:
- **Budget control:** per-sample max token budget by task type; over-budget responses are truncated and penalised.
- **PTX loss** on hand-picked high-quality data to prevent forgetting.
- **Temperature decay:** high early for exploration, lowered later.

*E. Infra (§3.3).* Colocated train/infer engines; a checkpoint engine broadcasts full parameters (full update of the 1T model in under 30 s); heavy environments run as separate scalable services; many concurrent rollouts amortise slow environments; **partial rollout** pauses long-tail trajectories and resumes them next iteration; one Gym-style interface for new environments. Tool calls use a TypeScript declaration format (shorter than JSON Schema) with a constrained-decoding "enforcer" for the call section.

**Verification / grading.**
- Agentic SFT data: LLM judge vs LLM-written rubric over an LLM-simulated stateful tool world. Nothing in the pipeline is executed except in the real-sandbox subset. Failure modes: the simulator can make an impossible plan succeed; the judge and simulator can share blind spots; the rubric author can encode wrong success criteria; no negatives or canaries are described.
- RL: verifiable rewards (tests, answer match, code-checked constraints) plus judges; a dedicated hack-check for instruction-following claims; prescriptive rubrics to block flattery-style hacks of the critic.

**Scale & cost.** 3,000+ MCP tools; 20,000+ synthetic tools; "thousands" of agents; "tens of thousands" of trajectories; 10,000+ concurrent sandboxes; H800 cluster. No GPU-hours, token costs, per-stage yields or kept-vs-generated counts for any synthesis stage.

**Key results** (non-thinking mode; Table 3).
- τ²-Bench Avg@4: retail 70.6, airline 56.5, **telecom 65.8**; micro average 66.1 vs DeepSeek-V3-0324 48.8, Qwen3-235B-A22B 37.3. Claude Opus 4: 81.8 / 60.0 / 57.0. K2's telecom score (the dual-control domain) is the highest in the table, ahead of Opus 4 (57.0) and Sonnet 4 (45.2).
- ACEBench (En) 76.5 vs GPT-4.1 80.1.
- SWE-bench Verified agentic single attempt 65.8 (multi-attempt 71.6), Multilingual 47.3; LiveCodeBench v6 53.7; AIME 2025 49.5; GPQA-Diamond 75.1; IFEval 89.8; Multi-Challenge 54.1.
- **No ablation** of the synthesis pipeline, the tool simulator, the rubric judge, the critic loop or the RL additions is reported, so no number can be attributed to any single mechanism here.

**Failure modes & limits.** Authors: excessive tokens on hard reasoning or unclear tool definitions, truncated or incomplete tool calls; performance can drop when tools are enabled unnecessarily; one-shot full-project building is weaker than agentic use. Critic rubric favours overconfidence. My critique: (1) the agentic pipeline is described at the level of components with no thresholds, yields or ablations, so it is not reimplementable from the paper; (2) an LLM tool simulator with "controlled stochasticity" is a modelling choice, not ground truth, and nothing checks that its state transitions follow consistent rules; (3) rubrics are generated with the tasks, so the task author also writes the grader (no information separation); (4) the strong telecom result suggests the synthetic multi-turn data covered user-in-the-loop troubleshooting patterns, but the paper does not say whether τ²-like domains were in the synthetic set (contamination cannot be ruled out).

**Tips & tricks.**
- Grow tool and domain coverage by **hierarchical evolution** (category → application domain → tools), and mix in real tool specs (MCP) for realism.
- Make an "agent" a (system prompt, tool subset) pair and generate tasks per agent; it gives a natural diversity unit.
- Attach a rubric (success criteria, expected tool patterns, checkpoints) to every synthetic task at birth.
- Inject controlled failures (partial failures, edge cases) in tool responses to teach recovery.
- Filter RL prompts to moderate difficulty with the current SFT model's pass@k.
- Add a **hack-check** layer for any judge-graded constraint: detect "claims of compliance" separately from compliance.
- Have a fine-tuned generator write tasks aimed at known failure modes.
- Write explicit **prescriptive anti-hack rubrics** for any learned judge (e.g. ban self-justification and user flattery).
- Recalibrate a learned critic on verifiable rollouts during RL.
- Per-task-type token budgets with truncation penalty; partial rollouts for long-tail episodes.

**Fit to Crucible Part 2.**
- **F0 DomainSpec / diversity (§3.5):** hierarchical domain evolution is a way to *propose* skill-tree leaves and World facets for the coverage grid. Adopt it in the domain-architect role, but measure coverage with our cell-fill and behaviour-signature meters, not with a t-SNE plot.
- **F1 (World / App synthesis):** their LLM tool simulator is the opposite of our executable App. Use it at most as a **cheap pre-screen tier** (does a proposal make sense before we code an App), never as the grading World. Their real-sandbox subset is the pattern we follow everywhere.
- **F1 knob:** "controlled stochasticity" (partial failures, edge cases) should be a World knob in executable form: deterministic, seed-driven fault injection in tools, so G2 determinism still holds per seed.
- **F3:** "agent = system prompt + tool subset" plus a fine-tuned generator for failure modes maps to F3 proposals conditioned on failure clusters from F12 (traces-to-tasks loop).
- **F5 / G12:** the hack-check layer and prescriptive anti-hack rubrics generalise to our negative class `attack` and to judge canaries (§3.8): every judge-graded check needs a "claimed but not done" negative.
- **F9:** moderate-difficulty filtering by SFT pass@k is the same idea as our per-student band; nothing new on thresholds.
- **§3.8 loops:** closed-loop critic refinement on verifiable rollouts is a verifier-loop pattern, but it updates the judge from student behaviour, which our "never selected on student reward" guard forbids for gates. Allow it only for advisory judges, anchored to the frozen anchor set.
- **Infra (Part 1 relevance):** partial rollouts, environments as separate services, per-task token budgets.
- Priority: **P2** overall; **P1** for the hack-check negative class and hierarchical domain evolution as an F0 aid.


## 2509.26490 — VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications (Meituan LongCat Team; corresponding authors Wei He and Qi Gu, 2025; arXiv v2 Oct 2025)
Read: FULL. Sections 1–6, App. A (trait definitions), App. B (agent, user-simulator and sliding-window evaluator prompts in full), App. C (a full cross-scenario trajectory) from the arXiv HTML text.  Relevance to Part 2: **medium**. It is a hand-curated benchmark (400 tasks), not a synthesis pipeline, so it says little about scaling task creation. Its value for us is in four design ideas: (1) **policies encoded as tool pre/post-conditions on a dependency graph** instead of a policy document; (2) a **three-axis complexity model** (reasoning, tool, interaction) with measurable proxies we can turn into knobs; (3) **deliberate distractor entities** in the seeded database with only a handful of valid solutions; (4) a **rubric-based sliding-window LLM evaluator** with a measured agreement (κ = 0.828) and an ablation showing rubrics are what make the judge work. It also gives a clean estimate of how many runs a stochastic multi-turn evaluation needs (k = 4).

**Problem.** Existing agent benchmarks either test function-call accuracy in isolation or, like τ-bench, constrain agents with long policy documents and narrow action spaces. They underplay information-heavy reasoning, inter-tool dependencies, cross-domain composition and the user as a source of uncertainty. VitaBench asks what makes real-world agent tasks complex and builds a benchmark along those axes in life-service apps (food delivery, in-store dining and services, online travel).

**Method / algorithm.**

*Formalism.* Per environment e, a POMDP with state S = S_db ⊗ S_user. Tool calls follow deterministic Python transitions T_db; user turns follow a stochastic LLM transition T_user. Reward r(e, u, τ) ∈ [0, 1] after the episode.

*Complexity model* C_task = ⟨C_reason, C_tool, C_interact⟩:
- **Reasoning:** entropy of the observation space H(O) and partial observability η = 1 − |O|/|S|; operationalised as the number of "reasoning points" per task and the search-space size (candidate entities).
- **Tool:** tool graph G = (V, E); |V|, edge density ρ = |E| / (|V|(|V|−1)), and coverage |V_task|/|V| of the task-relevant subgraph; cross-scenario tasks enlarge the action space.
- **Interaction:** user profile attributes (age, diet…), behaviour attributes (cooperation, goal ambiguity), and user state that changes over the dialogue.

*Stage I, framework design.*
1. Abstract three domains from real app implementations into simplified API tools: **66 tools** in total (27 write, 33 read, 6 general such as distance, weather, time). Per domain: delivery 20, in-store 24, OTA 38.
2. Build a directed tool-dependency graph. Each tool description carries **pre-conditions** (state required before the call) and **post-conditions** (expected result), e.g. `modify_order` needs a prior `get_order_detail`. Domain rules live in this structure; there is **no policy document**. That also makes cross-domain composition trivial.
3. User simulator (τ-bench style, gpt-4.1): it holds the full instruction but reveals requirements progressively and gives implicit constraints only when asked; each user has a profile and behaviour attributes. **Knowledge boundaries:** the agent can see the profile, but some facts (e.g. dietary restrictions) are hidden and must be inferred from order history or asked.

*Stage II, task creation* (four parts per task):
1. **User profiles** from anonymised real platform data, enriched with personality (impatient, anxious, indifferent) and interaction style (detail-oriented, dependent, logical).
2. **Instructions** that merge **several real user requests** into one composite objective (single-domain multi-goal, or cross-domain), then are **manually reviewed** for clarity and feasibility.
3. **Environment data:** real provider and product records plus model-generated augmentation under human supervision. **Target options that satisfy all constraints are mixed with distractors that each violate a specific requirement**; large search spaces, few valid solutions. Transaction histories support requests such as "same meal as last time".
4. **Rubrics:** hand-written atomic criteria per task (e.g. "restaurant within 500 m", "user eats only vegetarian food").
Tasks are refined over several trial runs with human verification, removing ambiguity while keeping several valid solution paths. A task has its own environment, typically 5–20 providers and sometimes 100+ products.

*User simulator rules (App. B), worth copying:* one line per turn; context then need; ask the agent to choose ("which would you recommend?"); split the instruction into independent points over turns but make sure every detail is eventually said; answer "don't know" if not in the instruction; never invent, assume, expand or substitute requirements; resist the agent's attempts to change needs; **become impatient and refuse after the agent repeats an already answered question 3 times**; explicit rules for when not to end (needs not fully stated, tasks incomplete or wrong) and when to end (all done, or agent says it cannot due to technical limits).

*Rubric-based sliding-window evaluator* (claude-3.7-sonnet, chosen to avoid overlap with evaluated agents):
```
rubrics R = {r_1..r_k}; state s ∈ {0,1}^k, all 0
split trajectory into windows of w = 10 turns, overlap δ = 2
for each window W_i:
    judge sees env info, full user instruction, W_i, current s
    may set s_j = 1 if the assistant met r_j in this window
    may reset s_j = 0 only if the assistant overturned an earlier correct conclusion
score = 1[Σ s_j = k]        # all-or-nothing for the benchmark
```
Judge instructions include: tool results are visible only to the assistant and do not count as recommendations; the assistant may *believe* it placed an order that failed; quantities and times must match strictly; addresses and notes use functional equivalence.

*Protocol.* Function-calling agents with OpenAI tool schemas; no turn limit; ends on `###STOP###` or failure; temperature 0; **4 runs per task**; Avg@4, Pass@4, Pass^4.

**Verification / grading.**
- An LLM judge over hand-written rubrics, not a state check. The authors argue state comparison cannot grade recommendations or planning that leave the DB unchanged, and gives no intermediate signal.
- **Evaluator ablation** on GLM-4.5 cross-scenario trajectories vs human labels: full method score 20.0, task accuracy 95.0, rubric accuracy 88.5, κ 0.828; without sliding window 19.0 / 90.0 / 87.6 / κ 0.604; **without rubric checklist 91.0 / 22.0 / κ 0.018**; without both 82.0 / 32.0 / κ 0.067. Without rubrics the judge passes almost everything.
- Hackable: the judge reads the assistant's claims; the prompt warns about false "order placed" claims, but a rubric met in words and not in the DB can still slip through. Rubrics are monotone (once true, stays true unless explicitly overturned), so a correct statement followed by a wrong final action could pass.

**Scale & cost.** 400 tasks: 100 cross-scenario (main), 100 per single domain. Databases: cross-scenario 1,324 providers / 6,946 products / 447 transactions; delivery 410 / 788 / 48; in-store 611 / 3,277 / 28; OTA 1,437 / 9,693 / 154. Trajectories span roughly 50–100 turns (Table 1). Human effort per task and dollar cost are not reported.

**Key results.**
- Cross-scenario Avg@4: best o3 (high) 30.0, Claude-4.1-Opus thinking 29.0, LongCat-Flash-Thinking 24.3, Gemini-2.5-Pro 23.5, GPT-5 (high) 22.8; best non-thinking Claude-4.1-Opus 21.8. **Pass^4 on cross-scenario is at most 6.0** for every model; Pass@4 reaches 61.0 (o3).
- Single domain best Avg@4: delivery 54.0 (GPT-5 high), in-store 56.8 (LongCat-Flash-Thinking), OTA 37.8 (o3).
- Thinking vs non-thinking average 23.8 vs 17.9 with similar turns (61.1 vs 69.9).
- **Complexity vs difficulty** (all-model average): in-store 42.1 (5.6 reasoning points, search space 3,916, 24 tools, 68 edges, density 12.3%); delivery 38.0 (7.4, 1,246, 20, 50, 13.2%); OTA 20.7 (9.7, 11,284, 38, 309, 22.0%); cross-scenario 16.2 (10.3, 8,717, 66, 512, 11.2%). **Search-space size does not predict difficulty; reasoning points and tool-graph size do.** (Four points, so this is suggestive, not a fit.)
- Interaction ablation (figure only): default persona user < neutral user < solo agent with the full instruction upfront; persona styles hurt GPT-4.1-mini more than Claude-4-Sonnet.
- **User simulator reliability:** information fidelity 9.48/10 and persona consistency 9.34/10 over 100 human-rated conversations each (two annotators); "scattered" personas are the least controllable.
- **Runs needed:** resampling 32 runs, k = 4 cuts the MSE of the mean by 77.5% vs k = 1; k = 8 adds little.
- **Error analysis** (Claude-4.1-Opus, 76 failed rubrics on cross-scenario): reasoning 61.8%, tool use 21.1%, interaction 7.9%, **user-simulator errors 9.2%**. Agents abandon tasks despite having tools and repeat failed calls instead of adapting.

**Failure modes & limits.** Authors: simulator stochasticity (9.2% of failures) is mitigated only by repeated runs; data mostly Chinese (English version in progress). My critique: (1) all 400 tasks and rubrics are hand-built, so the method does not scale to our 500–1000 synthetic tasks without automation of rubric writing; (2) grading by an LLM judge over a 50–100-turn transcript is weaker than a state check for any task that writes (orders, bookings), and those are most tasks here; a hybrid would have been better; (3) monotone rubric state can reward an agent that says the right thing and then does the wrong thing; (4) the complexity "framework" is measured on 4 domain aggregates, so it is a heuristic, not a validated predictor per task; (5) the evaluator is a single model family (Claude 3.7) with κ measured on one agent's trajectories.

**Tips & tricks.**
- Encode domain rules as tool **pre/post-conditions** on a dependency graph; agents must explore instead of reading a policy, and domains compose without merging policy texts.
- Seed the database with **distractors that each violate exactly one constraint**, so success requires checking every constraint; keep only a handful of valid answers.
- Hide some user facts behind knowledge boundaries (inferable from history or by asking), to force clarification and inference.
- Build composite tasks by **merging several real user requests**.
- Use **atomic rubrics**; a judge without them is close to useless (κ 0.018).
- For long transcripts, use a **sliding-window judge** (10 turns, 2 overlap) with persistent rubric state.
- Run multi-turn evaluations **4 times**; report Pass^k to expose instability.
- Make the simulated user impatient after 3 repeated questions and resistant to persuasion.
- Count simulator-caused failures separately in error analysis.

**Fit to Crucible Part 2.**
- **F1 (World / App synthesis):** adopt **pre/post-conditions per tool** in `ToolSpec` and derive `tool_graph` edges of kind `precondition` from them. This also feeds G13 (interaction completeness) and lets F4 check that an oracle program respects preconditions before execution.
- **F2 (state seeding):** adopt the **one-violation distractor** pattern as a seeding rule tied to the `distractors` knob: for each constraint in the task, seed at least one near-miss entity that fails only that constraint. Each near-miss is also a free negative for F5 (an oracle variant that picks it must fail).
- **F2 / G8:** their "knowledge boundary" (fact hidden in history, discoverable by tools or by asking) is exactly the `observability` knob; G8 must then verify the hidden fact is reachable through some tool or in `user_known_info`.
- **F3 / diversity:** the three axes (reasoning points, tool-subgraph size and density, interaction attributes) are candidate coverage axes and difficulty predictors. Measure them from the oracle trace (F7): reasoning points ≈ number of constraints checked; tool coverage ≈ |tools used| / |tools exposed|; cross-app flows from `BehaviourSignature`. Their table suggests search-space size is a poor knob; prefer constraint count and tool-graph breadth.
- **F5 verifier:** keep state checks as the primary verdict for write tasks; use their **atomic-rubric, sliding-window judge** only for answer checks on read-only or recommendation intents (which our §3.3 contract calls `answer_checks`) and as an auxiliary dense reward. Require the rubric list to be generated from the oracle (each rubric = one state predicate or one answer fact), so rubrics and state checks stay aligned.
- **F6 / G10:** with a simulated user in the loop, budget 4 grades per task per measurement (their MSE curve), and track the simulator-error share as a World-side fault label (§3.7 fault sides).
- **TaskContract.user:** add their user rules (impatience after 3 repeats, resist persuasion, explicit end conditions) to our `ActorSpec` driver, alongside APIGen-MT's BoN critic.
- **§3.8 guards:** their κ = 0.018 without rubrics is a concrete canary: a judge prompt without structured criteria must fail the anchor set.
- Priority: **P1** for pre/post-conditions, near-miss distractor seeding and the sliding-window rubric judge for answer checks; **P2** for the complexity model as difficulty predictor.


## 2506.07982 — τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment (Barres et al., 2025; arXiv v1 Jun 2025; Sierra + University of Toronto / Vector Institute)
Read: FULL. Sections 1–5; App. A (telecom task distribution, personas, a full task spec, Default and No-User trajectories); App. B (how the original retail/airline tasks were verified and fixed); App. C (agent and user system prompts, example user instructions); App. D (retail, airline and telecom policies, original and workflow technical-support policies; skimmed for structure, the decision-graph figures are images); App. E (user-simulator error taxonomy and cases) from the arXiv HTML text.  Relevance to Part 2: **high (core)**. Three things map directly onto Crucible: (1) the **compositional task generator**: atomic subtasks defined as (init, solution, assertion) function triples, grouped by mutual exclusion, composed combinatorially, and **verified by execution** (init → solution satisfies all assertions; the task is not solved until every solution step is applied). It is the cleanest published version of "solution by construction" plus an "inverse" check, and it gave 2,285 tasks from 15 groups. (2) **Dual control**: the simulated user owns tools over its own state (a mocked phone), which makes the user both a skill target and a far more reliable simulator. (3) **Ablation modes** (No-User, Oracle Plan) that separate reasoning failures from communication failures, which is our F10 intent-audit idea in benchmark form.

**Problem.** τ-bench-style benchmarks are single-control: only the agent acts on the world and the user just supplies information. Real support work (technical troubleshooting) needs the user to act (toggle airplane mode, reseat a SIM) under the agent's guidance. Also, hand-written tasks are brittle and user simulators are unreliable (the paper measures 40–47% simulator error rates in the original domains).

**Method / algorithm.**

*Formalism: Dec-POMDP* (S, {A_i}, {O_i}, T, R, U, M), players i ∈ {agent, user}.
- S = S_world ⊗ S_history, S_world = S_db,agent ⊗ S_db,user (e.g. CRM records vs phone status).
- Each turn exactly one player acts: a tool call on its own DB or a message.
- Agent tool calls can change the user's state (enabling roaming on the CRM lets the phone roam).
- R: S → [0, 1] on the final state (DBs and history).
- U holds the user scenario and the agent's domain policy.

*Domain creation, five stages (telecom as the worked case).*
1. **Agent DB and tools.** An LLM writes a product requirements document (business logic, schema, functions); an LLM then writes the implementations, a mock DB and **unit tests**; humans refine the code until all tests pass.
2. **User DB and tools.** Same process for the user side: a mocked phone with status fields (signal, SIM, airplane mode, APN, app permissions…) and functions.
3. **Programmatic task creation.**
   ```
   atomic subtask t = ( {f_init_t,k}, {f_sol_t,k}, {f_assert_t,k} )
     f_init   : any DB function; sets up the fault       e.g. set_airplane_mode(True)
     f_sol    : must be agent or user TOOLS               e.g. toggle_airplane_mode()
     f_assert : any DB function; must hold at the end     e.g. assert_service_status("connected")
   groups  : mutually exclusive / alternative subtasks share a group
   compose : pick ≤ 1 subtask per group; concatenate init, sol, assert lists
   verify  : apply init, then sol  → all asserts true
             and the task is NOT resolved before all sol steps are applied
   ```
   Telecom: **15 atomic subtask groups** across 3 user intents of increasing complexity (service_issue, mobile_data_issue, mms_issue; the latter two often require fixing a service issue first). Composition yields **2,285 tasks**; a **balanced subsample of 114** is released, stratified by intent and number of subtasks (2–9). Solution lengths: service 2.31 ± 2.25 actions (1–8), mobile data 4.31 ± 1.79 (2–8), MMS 6.00 ± 2.85 (2–12). The number of subtasks is the difficulty knob.
4. **Agent policy.** An LLM writes the domain policy *from the curated tasks and their solutions* (step-by-step diagnostic procedures per intent). A second, "workflow" variant adds explicit decision procedures (and decision graphs).
5. **Manual refinement** of tools, policy and subtasks together.

*Personas.* Each telecom task is randomly given None, Easy (41-year-old office administrator, average skills, patient, confirms each step) or Hard (64-year-old retired librarian, low tech knowledge, anxious, only shares info when asked, worries a reboot will lose photos).

*Task spec format* (App. A.2): id; purpose ("test resolution path"); user scenario = reason for call, **known info**, **unknown info**, task instructions (behavioural conditions, e.g. show mild frustration after the first failed attempt; resolution criterion "status bar shows signal"); a **ticket** (used in No-User mode); initial state = initialisation actions on the user env; evaluation criteria = expected actions with requestor (user/agent) + environment assertions.

*User simulator* (gpt-4.1; prompt in App. C.2): one message or one tool call per turn, never both; never invent information or tool results; paraphrase; disclose progressively and wait to be asked; **call a tool only when the agent requests it**; if asked for several actions, refuse and ask for one at a time; tool-call messages are not shown to the agent; stop tokens `###STOP###`, `###TRANSFER###`, `###OUT-OF-SCOPE###`. **User tools return only human-readable output**, which keeps the "complexity asymmetry": the user can act but cannot plan the fix.

*Evaluation criteria* (any subset per task):
- **DB check** (final DB equals expected; from τ-bench);
- **communication info check** (the agent conveyed required info);
- **status assertions** (assertion functions over S_world);
- **natural-language assertions** over S_history (e.g. "the agent diagnosed the cause"), meant for debugging;
- **action matching** (every f_sol appears in the trajectory).
Telecom uses **only assertion functions**, so any valid fix path passes.

*Ablation modes.* **Default** (dual control); **No-User** (agent gets the ticket and controls all tools, user tools included: pure reasoning and tool use); **Oracle Plan** (agent is given the full tool-call sequence for both sides: pure coordination and communication).

*Verification of the original domains* (App. B): a unified tool formalism, a data model per domain, a mock domain for unit-testing the framework; tasks enriched with purpose metadata and structured user instructions (intent, known/unknown info); evaluation split into environment, communication, NL and action assertions; an **iterative review loop anchored in simulation**: run each task; reviewers patch transient agent or simulator errors to let the run finish; inspect for under-specification, over-specification and **non-unique solutions**; refine. This yields the "verified" retail (115) and airline (50) sets.

**Verification / grading.**
- Telecom: end-state assertion functions over the user's device and the CRM. They are code, deterministic and path-agnostic.
- Generation-time checks: the composed task must pass after init → sol, and must **fail before the last solution step** (a "not pre-solved / every step matters" check, close to our G4 and G6).
- Remaining weak spots: (1) the agent's own claims are not graded (only the state), which is right; (2) a user simulator can still end the dialogue early (all 8 telecom simulator errors were premature `###TRANSFER###`); (3) assertions check that the fault is fixed, not that nothing else was changed (no explicit collateral check unless the DB check is also used); (4) the "not resolved before all sol steps" check is on the full prefix, not on dropping each step individually, so redundant steps in the middle could slip through (I infer this from the wording; the code may do more).

**Scale & cost.**
- Domains: retail 500 users / 50 products / 1,000 orders, 7 write + 6 read agent tools, 115 tasks; airline 500 users / 300 flights / 2,000 reservations, 6 write + 6 read, 50 tasks; telecom 5 plans / 9 lines / 4 customers, 6 write + 7 read agent tools, **15 write + 15 read user tools**, 114 tasks (2,285 full).
- Cost with gpt-4.1 as both agent and user: **$0.086 agent + $0.059 user per task**; one trial of all domains is about **$40**.
- Each task run 4 times at temperature 0.

**Key results.**
- **Pass^1 telecom** (Default): gpt-4.1 34%, o4-mini 42%, claude-3.7-sonnet 49%, gpt-4.1-mini about 50% (stated as "around 50%"). gpt-4.1 drops from 74% retail / 56% airline to 34% telecom. Claude-3.7's telecom pass^1 equals its airline score but falls faster with k.
- **Dual control costs ~20 points:** No-User → Default drops pass^1 by 18 points (gpt-4.1) and 25 points (o4-mini).
- **Oracle Plan** helps both, more for o4-mini. The **workflow policy** slightly helps Default and No-User but **hurts Oracle Plan** for both models (extra guidance confuses an agent that already has the plan).
- **Length:** pass^1 falls with the number of required actions and is near 0 beyond 7 actions in Default; the No-User advantage shrinks from about 0.3–0.4 to under 0.2 as actions grow, so long horizons are hard even without the user. Same trend with the number of subtasks.
- **Issue types:** failures concentrate on mobile_data and mms issues (multi-stage, conditional).
- **Personas:** Easy > Hard; surprisingly None ≤ Hard.
- **User-simulator reliability** (2 annotators; gpt-4.1 agent and user): airline 100 conversations, 13% critical / 34% benign / 47% total errors; retail 50, 12% / 28% / 40%; **telecom 50, 6% / 10% / 16%**. Error types: retail mostly conversation-structure violations (11/20), missing constraints (4/20, mostly critical), premature termination (3/20), ungrounded references (2/20); airline structure 19/47, ungrounded 15/47, missing constraint 11/47, premature 2/47; telecom all premature transfer (8/8). (The Table 2 caption says no critical telecom errors were reported while the table lists 3 critical; the text treats 6% as critical.)

**Failure modes & limits.** Authors: user tools have not yet been applied to retail and airline; domain creation still needs human experts (PRD and code refinement, policy refinement); the expert-novice gap (adapting explanations to the user's mental model) is not modelled. My critique: (1) the telecom DB is tiny (4 customers, 9 lines), so the diversity of 2,285 tasks is purely combinatorial over 15 fault groups; many tasks share one behaviour pattern and differ only in which faults are stacked, so the released 114 are a better unit than the 2,285; (2) the policy is written *from* the tasks, which risks the policy encoding exactly the generated fault set (agents can succeed by following the policy's procedure without diagnosis); (3) four models and 4 runs each, no confidence intervals; (4) personas are randomly assigned rather than crossed with tasks, so persona effects are confounded with task mix; (5) the No-User mode gives the agent the user's tools *and* a ticket, so it changes two things at once.

**Tips & tricks.**
- Define each atomic subtask as **(init, solution, assertion)** functions; solution functions must be policy-visible tools, init and assertions may touch the DB directly.
- Put alternative or conflicting subtasks in one **group** and draw at most one per group when composing; this prevents contradictory tasks by construction.
- At generation time, check **solved after all solution steps and not solved before the last one**.
- Subsample the combinatorial space **balanced by intent and subtask count** instead of keeping everything.
- Give the simulated user **tools over its own state** with human-readable outputs; it cut simulator errors from 40–47% to 16%.
- Separate **known info** and **unknown info** in user instructions; it directly encodes what the user can and cannot communicate.
- Ship a **ticket** per task so the same task runs in No-User mode; the Default–No-User gap measures communication and coordination skill; Oracle Plan measures reasoning load.
- Grade troubleshooting by **assertions on the end state**, not by action matching, so any valid fix path passes.
- Verify a benchmark by running each task, letting reviewers patch transient simulator errors so the run finishes, then checking for under-, over-specification and non-unique solutions.
- Have the user refuse multi-action instructions and act one step at a time.
- Track simulator errors by type; premature termination and missing constraints are the critical ones.

**Fit to Crucible Part 2.**
- **F3–F5 (proposal, oracle, verifier) as one typed unit:** adopt the (init, solution, assertion) triple as the **atomic building block** of a `TaskContract` in any `world_synth` domain:
  - `init` = start-state mutation on the `SeedBundle` (F2);
  - `solution` = the `Oracle(kind=action_program)` over policy-visible tools (F4; enforces G7 tool-only by construction);
  - `assert` = `state_checks` (F5).
  The composed task's oracle is the concatenation; its verifier is the conjunction. Store `groups` (mutual exclusion) in the `DomainSpec` so F3 cannot propose contradictory compositions.
- **F6 gates:** their two generation checks map onto ours: init → sol passes all asserts = **G3 Solvable**; init alone (and init + any strict prefix of sol) fails = **G4 Not pre-solved**, which we should strengthen to **G6 Inverse mutation** by dropping *each* solution step individually (our drop-a-step operator), not just suffixes. Add the collateral check they lack: the start→end diff must lie inside the union of the subtasks' declared write sets.
- **Difficulty (§3.6):** number of subtasks is a clean `count` knob with a measured monotone effect (pass^1 near 0 beyond 7 actions). Their intent hierarchy (mobile data and MMS depend on service first) is a `linkage` knob: a composition where one fault masks another is harder than independent faults, which is what we want from structure-based hardness. Prefer compositions with dependencies over pure stacking, and check the behaviour signature (F7) changes.
- **Diversity:** combinatorial composition gives volume without behavioural diversity. Use F7 signatures and F8 dedup on the composed set and sample the pool **balanced by intent × subtask count**, as they did, rather than taking all combinations. Count the atomic group set, not the composed count, as the family's real size.
- **Dual control in the World (`ActorSpec`):** give simulated people **their own tools and state** in the World (a person's device, inbox, or local files). It is both a new skill axis (guiding a user to act) and a reliability fix for the simulator. Their asymmetry rules (human-readable outputs, act only on request, one action at a time) belong in the generic person driver. Make G8 communicable check the user side too: every user-side solution step must be something the user can be asked to do with its tools.
- **F10 intent audit:** their **No-User** and **Oracle Plan** modes are exactly the ablations we need for multi-turn tasks. Add them to `IntentCard.ablations`: No-User (gives the agent all tools and the ticket) isolates reasoning; Oracle Plan isolates communication. A task meant to train communication should show a large Default–No-User gap; if the gap is near zero, the user part is decorative.
- **Policy generation:** they write the policy from tasks; we should invert that and write the policy (as `policies_as_code` plus a prose rendering) **before** tasks, then check tasks against it, to avoid the policy leaking the fault list. Their finding that a workflow policy hurts when the plan is known suggests including policy detail level as an `hints` knob.
- **Benchmark verification loop (App. B):** their simulation-anchored review (run, patch transient errors, look for under/over-specification and non-unique solutions) is a good manual template for our human spot-check queue (§3.8) and for G14 alternatives.
- **Cost (§3.10):** $0.145 per task-run with gpt-4.1 on both sides is a useful reference for the per-run cost of a simulated-user grade; our G10 flake bound with a simulated user would cost about 60 × that per task, so flake runs should use a cheaper simulator or a smaller k with sequential testing.
- **Students:** Kimi K2 reports 65.8 on τ²-telecom (non-thinking), well above the models here, so the 114-task telecom set is near the top band for frontier models but probably in band for a 35B student; it is a candidate external held-out probe, never a training source.
- Priority: **P0** for the (init, solution, assertion) triple with groups and the solved/not-pre-solved checks as the generic synthetic-task builder; **P0** for user-side tools and the asymmetry rules in the person driver; **P1** for No-User / Oracle Plan ablations in F10 and the balanced subsampling.


## 2504.21798 — SWE-smith: Scaling Data for Software Engineering Agents (Yang et al., 2025; arXiv v2, May 2025; venue not stated in the text)
Read: FULL (main text §1–6 and appendices A–G: infrastructure, all four bug strategies with Algorithms 1–2, yield tables 10–12, issue generation, difficulty rater, training config, agent behaviour and failure taxonomy; skipped only the per-repo description table and references).  Relevance to Part 2: **high, core.** It is the canonical "environment first, then synthesise many tasks inside it" pipeline, the closest published ancestor of Crucible's debug lane (fault planting into a real repo, F2P validation, issue writing, multi-bug combination), and it reports yield and cost per strategy, which is exactly the per-stage accounting §3.0 rule 1 and §3.10 want.

**Problem.** SWE-bench-style training data is scarce (≤ thousands of instances from ≤ 11 repos), costs hundreds of human hours, and needs terabytes because every instance gets its own Docker image keyed to a historical commit. Mining real PRs is low-yield: SWE-bench's own funnel keeps 2294 of 93,139 PRs (2.46%).

**Method / algorithm.** The core inversion: build one execution environment per (repo, commit), then manufacture hundreds of tasks that break that environment's tests, instead of finding tasks first and building an environment per task.

1. **Repo selection.** Top 5,000 PyPI packages (snapshot 2024-11-18), sorted by GitHub stars, drop < 1,000 stars, drop the 12 SWE-bench test repos, keep permissive or checked custom licences. 128 repos kept; 17 more abandoned at manual review.
2. **Environment build.** SWE-agent with Claude 3.5 Sonnet (Oct 2024), ≤ $2 and ≤ 150 calls, starting from a bare clone on Ubuntu 22.04; the system prompt tells it to find CONTRIBUTING/README, `pip install -e .[dev]`, apt-get system deps, and run pytest. Average $0.72 and 17 steps, ~2 minutes. A human then reads the trajectory, extracts install and test commands (3–20 min, ~7 min typical), checks that > 80% of existing tests pass, and writes a test-output parser (~1 min; reused across pytest repos). One Docker image per repo at its latest commit. Total human time ~18–20 h for 128 repos.
3. **Candidate generation**, four strategies, each emitting a `.diff`:
   - **LM Modify.** Enumerate every function (and, separately, class) via Python `ast`, excluding test directories. o3-mini is prompted as a "chaos monkey" to rewrite it with a logical bug. The prompt holds a pool of 9 bug-type hints (calculation order, data-transform errors, wrong variable assignment, edge-case mishandling, conditional/loop logic, off-by-one, default/constant changes, reordering, swallowed exceptions) of which **3 are sampled per call**, plus shuffled constraints: no compile or syntax errors, keep the signature and docstring, be subtle, add several bugs in long functions, and **no comments marking the bug** (models do this by default otherwise). Output: explanation + bugged code.
   - **LM Rewrite.** Remove the function body, give the whole file minus that body plus the signature/docstring, and ask for a fresh implementation with no external libraries. No bug is requested; bugs come from imperfect reimplementation.
   - **Procedural (AST).** 13 transforms in four families: Class (remove methods and their references, remove base class, shuffle methods), Control flow (invert if/else bodies, shuffle lines), Expressions (±1 on a numeric constant, break operator chains `a+b+c→a+b`, swap operands, change operator), Removal (loops, conditionals, assignments, try/with wrappers). Each has precondition filters (has loop, has if-else, has operators, etc.) and a complexity window, where complexity = count of conditionals + loops + boolean ops + exception blocks + comparisons; a min-complexity floor drops getters/setters. Each transform applies per eligible site with a "likelihood" (e.g. 0.5) so edits stay modest. Zero LM cost.
   - **Combine.** Merge already-validated single bugs. *Same file* (Algorithm 1): per file, sample up to `max_combos=40` sets of `num_bugs∈[2,4]` validated bugs, apply sequentially; on clean apply save and re-validate as one instance, stop at `limit_per_file=3`, and drop every remaining combo that shares a bug with the saved one (each bug used once). *Same module* (Algorithm 2): build a path trie from patch paths, collapse to `depth=2` (e.g. `sklearn/feature_extraction`), sample `num_bugs∈[2,5]`, `max_combos=100`, `limit_per_module=10`, keep only combos touching ≥ 2 files. Motivation: 75% of multi-file SWE-bench instances stay within one submodule.
   - **PR Mirror.** Crawl PRs touching Python files (mostly from 2023-01-01 on). For each changed file, give o3-mini the file's current source plus that file's hunk of the PR diff, and ask for a full-file rewrite that reverts the PR; concatenate into one patch. `git apply --reverse` fails on most PRs because code has moved (checked on 100 sqlfluff PRs). Skip PRs touching > 8 files. Drops SWE-bench's "must link an issue" and "must touch tests" filters; validation decides. Sanity check: 92/100 django SWE-bench instances recovered, 84 with identical F2P sets, 8 with a subset because tests were later removed.
4. **Execution validation.** Apply candidate, run the full suite, keep it iff ≥ 1 previously passing test now fails (F2P); P2P = all other tests. Discard if the suite exceeds a **2-minute** limit. Bug pushed as a branch on a mirror repo; instance fields are repo, instance_id, base_commit (the bugged branch), patch (the bug; its reverse is the gold fix), problem_statement, created_at (validation time), FAIL_TO_PASS, PASS_TO_PASS. **No hidden tests**: F2P tests are present and runnable in the repo at solve time.
5. **Issue generation.** An LM gets one random SWE-bench Verified issue as a style demonstration, the bug patch, the F2P list, the source of one F2P test, and the test-run output, and writes a GitHub-style issue. Rules: mimic the demo's structure, length and "fluff"; never explain the fix or cause; never name pytest or the failing test; include a reproduction that prints actual vs expected. ~2.54¢ per issue.
6. **Trajectories and training.** SWE-agent (bash, str_replace_editor, submit), Claude 3.7 Sonnet as expert, ≤ 75 steps, ≤ $2. Rejection sampling fine-tuning (keep resolved trajectories only), full-parameter with torchtune, lr 5e-5, ≤ 3 epochs, 32,768 context, 2–8 H100. Convert function-calling trajectories into the student's XML thought/action format (training on raw function-call traces "does not work"). Student inference at T = 0, last 5 tool outputs kept in context.

**Verification / grading.** Standard SWE-bench harness: F2P must pass and P2P must stay passing after the agent's edit. Hackability: because F2P tests are in the repo and visible, an agent can read them, and the instruction tells it not to touch tests but nothing enforces that beyond running the original tests; the paper itself says the data is **not fit for evaluation** without deleting F2P tests and checking issue leakage and underspecification. Gold is the reverse of the planted bug, so solutions that fix the behaviour differently still pass if tests pass (tests are the only oracle). The 1-test F2P minimum is a weak signal: a bug might break a test incidentally (e.g., an import-time error) rather than exercise the intended logic; the prompt only discourages compile errors.

**Scale & cost.** 100,074 candidates → 50,137 instances (50.1%) from 128 repos (avg 381/repo, max 2,277 for pandas). $1,360 total: $1,000 bug generation, $160 automated install, $200 issues for ~10K bugs. 2.32¢ per candidate on average. Per strategy (yield; instances; ¢/candidate; median F2P; median lines): Combine 96.9%, 10,092, 0, 15, 11; LM Modify 56.0%, 17,887, 0.38, 4, 3; LM Rewrite 35.0%, 4,173, 3.93, 4, 24; PR Mirror 33.8%, 2,344, 5.53, 3, 14; Procedural 40.2%, 15,641, 0, 7, 5. Procedural spread is wide: shuffle class methods 1.9%, invert-if 49.4%, remove-assign 48.7%, swap operands 22.6%. Appendix C.1 also states PR Mirror's yield as 13.18% in prose, which conflicts with its own table (33.8%); I could not resolve which is right. Per-repo yield: 10 repos < 25%, 27 repos > 75%; low test coverage → low yield. Storage: 125 images, 290.5 GB, against an estimated 50–150 TB for SWE-bench-style per-instance images (SWE-Gym: 6 TB for 2,438). Flask case study: 402 candidates → 267 instances in 23 min of generation + 14 min validation for $2.47, against 11 SWE-bench flask instances from 2,434 PRs costing ~38 human hours; "bug coverage" (share of tests broken by ≥ 1 instance) 89% vs 7.25%. Expert solve: 6,457 resolved out of 17,906 attempts on 8,686 instances (36%).

**Key results.**
- SWE-agent-LM-32B (Qwen2.5-Coder-32B-Instruct + 5,016 trajectories): **40.2% pass@1 SWE-bench Verified**, 30.7% Lite (base +33.4 pp); 7B: 15.2% Verified. Pass@1 averaged over 6 runs.
- Same-size comparison at 500 trajectories (Claude 3.5 / GPT-4o experts): 28.2% Verified; the paper calls this a relative +8.2% over SWE-Gym and +0.7% over R2E-Gym at the same size.
- **Strategy ablation** (7B, 1,000 instances per strategy, capped to 507 trajectories): PR Mirror 9.2 ±1.7, LM Rewrite 8.8 ±1.7, Procedural 8.6 ±1.8, **LM Modify 5.7 ±1.5**. Rated difficulty explains it: LM Modify is mostly trivial variable swaps despite the 9-type hint list (score 3.30 vs Rewrite 5.27, Combine 5.72).
- **Issue-text ablation** (600 PR-mirror instances, capped to 259): fixed templates 6.4, F2P-test-as-issue 7.3, LM issue 7.7, original issue 7.8. Templates produced 31% fewer unique actions (379 vs 550). Giving the F2P test made the student skip writing a reproduction script (127 vs 379 of 500 Verified runs reproduced the bug): leaking the grader changes the learned behaviour, not just the score.
- **Difficulty does not predict training value.** Expert resolve on easy/medium/hard 58.6/41.0/17.0%, but SFT on 500-trajectory sets of mean difficulty 2/4/6/8 gave 12.4/10.8/13.6/12.2% (no trend; no error bars).
- **Repo count scales ~log-linearly** at fixed 700 trajectories over 4/25/50/100 repos.
- **Specialisation:** 700 SymPy trajectories at a pre-cutoff commit, evaluated on 22 post-2022 SymPy Verified tasks: 7B 21.2% vs 13.6% general; 32B stage-2 42.4% vs 33.3%, with small general loss.
- RFT beats random trajectory sampling at every n in 100–1,600. Capping at 3 trajectories per instance (easy-instance bias, after SWE-Gym) gave the final set.
- Multilingual: on their new 300-task, 9-language SWE-bench Multilingual, the 32B gets 8.4% vs base 6.5% vs Claude 3.7 43%; edits drift toward Python syntax. Python-only synthetic data did not transfer across languages.
- Behaviour: > 25% of student trajectories have a ≥ 10-long repeated base command (vs < 4% for Claude); length-10 repetition ≈ 89% failure; 53% of failures hit step/cost limits, often before any source edit (stuck at localisation). Scaffold anti-repetition (warnings at 4/6 repeats, resampling at 6/8, temperature bumps) suppressed repetition but dropped resolves slightly (38.4%): repetition is a symptom of being lost, not a cause.

**Failure modes & limits.** Authors: Python-only (AST tooling), SFT only (no RL), no hidden tests, issues not screened for leakage or underspecification, synthetic bugs not shown faithful enough for evaluation, PR mirroring runs against today's code so line numbers and context drift, and installs still need a human pass. My critique:
- The F2P ≥ 1 criterion is the whole validity gate. Nothing checks that the broken test exercises the planted logic, that the bug is not trivially visible (e.g., a deleted method raising AttributeError), or that the issue does not reveal the location. That is weaker than Crucible's S5/S6/S8 gates.
- Combine instances are near-free to make but almost absent from the final SFT set (130 of 5,016), and the paper never ablates them, so "combining makes harder tasks" is shown only by the difficulty rater, not by training value. No evidence the combined bugs interact; they are co-located, not linked.
- The difficulty rater is trained on time-to-fix labels for real issues (75.3% 3-way accuracy) and then applied to synthetic bugs: a distribution shift the paper does not check.
- Ablations are single-run on a 7B with ±1.5–1.9 std errs, so the 8.6 vs 9.2 "comparable" claim is within noise in both directions.
- Repo-scaling and difficulty results are for SFT; their meaning for RL with group advantages (where band matters) is untested.

**Tips & tricks.**
- Environment per (repo, commit), many tasks per image: ~500× less storage.
- Sample 3 of N bug-type hints per call and shuffle constraints, but expect that open-ended "make a bug" still collapses to trivial swaps; ask for reimplementation (Rewrite) or real reversions for harder, more realistic faults.
- Always forbid comments that mark the bug; models add them by default.
- Use a reasoning model for patch reversal against moved code; `git apply -R` mostly fails.
- Drop issue/test-linkage filters at mining time and let execution decide: more real changes become tasks.
- Cap trajectories per instance (3) to stop easy-instance overweighting.
- Never give the F2P test as the problem statement; it teaches the student to skip reproduction.
- Style-transfer issues from a random real issue (structure, length, fluff); forbid naming the test or the cause.
- Combine validated bugs with "each bug used once" and per-file/module caps to avoid over-representation.
- Time-limit the suite (2 min) at validation to keep CPU cost bounded.
- Count "bug coverage" (share of tests a pool can break) as a breadth metric per repo.

**Fit to Crucible Part 2.**
- **Source table (§3.4) and F1–F5 for the debug domain.** SWE-smith is the "synthetic fault in a real repo" source that Crucible's lane lacks: the lane mines real fixes (~1% of fix units yield, links in ~3/45 repos). Adopting PR Mirror (LM revert of a real fix against current HEAD) directly widens the mined supply without historical environment builds, and keeps realism (it is a real fix). **Adopt** as a mining fallback for fixes whose `git revert` fails, with the same S-gates. P0 for the lane's yield problem.
- **Procedural and LM Rewrite faults as a second family**, not a substitute: per §3.10, the lane is a high-realism slice; synthetic faults would be a separate family (as `--no-link` is) so transfer is measured on held-out families. Use LM Rewrite over LM Modify (Modify gives the worst training value). P1.
- **Combine ≠ linked faults.** SWE-smith's combination is co-location only. Crucible's multi-bug tasks require executed interaction edges (S6, G6 inverse mutation). Use Algorithm 1/2's bookkeeping (each fault used once, per-file and per-module caps, ≥ 2 files) as the **quota and dedup mechanics** for F11, but keep G6 as the link test. Avoid importing "combined = harder" as a difficulty knob without a behaviour-signature change (§3.6's rule).
- **F5 negatives for free.** The planted-bug patch set per repo is a ready generic-mutant pool for G5's kill floor and held-out mutant set: apply 13 AST operators to the oracle fix to get "near-miss" wrong patches. P1.
- **G11 / instruction writer (§3.9).** Adopt the issue prompt's rules as leak rules (no test name, no cause, no fix), plus the finding that F2P-as-issue changes behaviour: add a G11 check that the report does not quote hidden test identifiers or assertion text. The style-demo trick suits the report writer role. P0 (cheap).
- **F9 / §3.6.** Confirms "hardness by itself is no proof of value" (already cited). The cheap predictor here is a rater on (issue, patch) — a template for Crucible's advisory predictor, but trained on our own p̂ labels, not time-to-fix.
- **F12 / trace analysis.** Their failure taxonomy (error/context, stuck at localisation, stuck at reproduction, incorrect localisation, incorrect edit, by comparing touched files to gold) is a deterministic fault-side labeller that fits `failure_skill`; the repeated-base-command detector (strip env, last chained command, drop args) is a cheap "lost agent" flag. P1.
- **§3.10 cost model.** Supplies per-stage yields and ¢ costs; the cheapest-first ordering (CPU AST before LM, validated singles before combos) matches rule 4.
- **Avoid:** visible F2P tests (Crucible's hidden tests are right); single-run 7B ablations as decision evidence; Python-only assumptions (their multilingual transfer failed, so Crucible's multi-language profiles must each supply their own data).
- **Interface implied:** a `FaultSource` plug-in in the debug DomainSpec with `kind ∈ {mined_fix, mirrored_fix, lm_rewrite, ast_op, combine}`, each emitting `Oracle(kind=patch)` + candidate F2P set + provenance and per-kind yield/cost meters.


## 2412.21139 — Training Software Engineering Agents and Verifiers with SWE-Gym (Pan et al., 2024; arXiv v2 June 2025; the text carries ICML template keywords, venue otherwise unstated)
Read: FULL (§1–6 and appendices A–B: concurrent-work comparison, capping table, trajectory-sampling breakdown, training configs, both ORM prompts).  Relevance to Part 2: **medium.** It is the "mine real issues, hand-build environments" baseline that SWE-smith, R2E-Gym and SWE-rebench all position against. Its environment recipe is too manual to copy, but three findings carry over directly: per-instance capping against easy-task bias, the failure of naive on-policy self-improvement, and the on-/off-policy mix for training an outcome verifier.

**Problem.** No public SWE training set combined real GitHub tasks, executable environments and test-based reward. The SWE-bench train split has patches but no environments; R2E is synthetic and tiny (246).

**Method / algorithm.**
1. **Repo selection** via SEART GitHub search: Python, created before 2022-07-01, > 500 stars, ≥ 300 LOC, > 500 PRs, > 100 contributors → 358 repos, none of SWE-bench's 12.
2. **Instance extraction** with SWE-bench's script (PR links an issue, touches tests) → **SWE-Gym Raw: 64,689 instances**, no environments.
3. **Pick 11 repos with many instances** (pandas ≈ 1/3 of the final set; bokeh ≈ 1%). Generalise SWE-bench's versioning script so each instance maps to a release version (from pyproject, git tags, scripts); set up environments once per version, not per instance.
4. **Manual dependency configuration** per version from requirements files, CI scripts and docs at the issue's snapshot. The stated reason for doing it by hand: an automatic setup script that drops failures biases the set away from complex-dependency tasks (their criticism of the Nebius SWE-bench-extra approach).
5. **Execution validation** with SWE-bench's script: the gold patch must make more tests pass than the original code. ~200 annotator hours (authors) and ~10,000 CPU core-hours → **2,438 instances**, one Docker image each, 6 TB.
6. **SWE-Gym Lite (230):** SWE-bench Lite's filter (single file, clear statement, small diff, no error-message-text tests).
7. **Policy training (OpenHands CodeAct 2.1, bash + editor, no browser).** Rejection-sampling fine-tuning on 491 successful trajectories from GPT-4o and Claude 3.5 Sonnet (Oct 2024) at temperatures 0–1, max 30 or 50 turns. Yield was low: GPT-4o solved 4.5–9% per pass; Claude 3.5 29.1% of Lite at 50 turns. torchtune full FT, lr 1e-4, ≤ 5 epochs, batch 8, 32k context; eval ≤ 100 turns.
8. **Specialised-workflow self-improvement (MoatlessTools).** Iterative RFT: 30 rollouts per Lite task at T = 1.0, keep successes, fine-tune; 2 iterations (more gave nothing). 7B full FT lr 2e-5; 32B LoRA rank 64, lr 5e-4, 10,240 context; FP8 serving.
9. **Per-instance capping:** at most c successful trajectories per task, preferring those with fewer model turns.
10. **Outcome verifier (ORM).** Qwen2.5-Coder-32B fine-tuned to emit `<judgement>YES/NO</judgement>` after the full interaction log (system prompt asks: main task done? edge cases? errors? did it verify?). Reward = softmax over the YES/NO logprobs. Data: 1,318 successes (443 off-policy + 875 on-policy, ≤ 32k tokens) + an equal number of failures sampled from each source = 2,636. For MoatlessTools the verifier sees issue + before/after hunks + code spans and answers True/False; successes capped at 2 per instance, failures subsampled to balance.

**Verification / grading.** SWE-bench harness: F2P from the PR's test patch must pass, P2P must not regress. Real developer tests, hidden from the agent. Known weaknesses inherited: tests can be narrower than the issue; the validity rule (gold passes more tests than the original) is loose, admitting instances where few tests separate gold from base. The ORM is a learned judge with a YES/NO token over the whole trajectory, so it can reward a trajectory that *claims* verification; Best@k selection is only as good as that judge (gap below).

**Scale & cost.** 358 repos → 64,689 raw → 2,438 executable (11 repos). 200 human hours, 10k CPU-core-hours, 6 TB images. Trajectory sampling: 5,557 failed vs 491 successful OpenHands trajectories; successes median 33 messages, ~16k tokens. Gold patches are larger than SWE-bench's (69.8 vs 32.8 lines, 2.5 vs 1.7 files, 10 vs 9 F2P), and models score lower on SWE-Gym than SWE-bench.

**Key results.**
- OpenHands + 491 trajectories: 32B 7.0 → **20.6% Verified**, 3.0 → 15.3% Lite; 14B 4.0 → 16.4%; 7B 1.8 → 10.6%. Stuck-in-loop (same action 3× in a row) falls 4.6–18.6 pp for most sizes; empty-patch rate falls.
- **On-policy self-improvement hurt:** fine-tuning the base 32B on 868 on-policy + 491 off-policy successes dropped Lite from 15.3% to **8.7%**.
- MoatlessTools RFT: 7B 7.0 → 9.0 → 10.0% Lite; 32B 19.0 → 19.7% then flat.
- **Capping** (7B Moatless, Lite): cap 1 → 9.0%, **cap 2 → 9.7%**, cap 3 → 7.7%, no cap (172 traj) → 9.3%; zero-shot 7.0%. Success counts over 30 samples are long-tailed toward easy tasks.
- **Verifier scaling (OpenHands, 32B agent + 32B ORM, Verified):** Pass@k 20.6 → 37.8 (k=8) → 42.8 (k=16); **Best@k 20.6 → 29.8 (k=8) → 32.0 (k=16)**, roughly log-linear. LoRA verifier beat full FT (29.8 vs 27.2 @8). Off-policy-only verifier data plateaus near 22%; on+off mix best (27 @8 under full FT).
- Moatless verifiers: 7B 10 → 13.3%, 32B 19.7 → 26.3% Lite at k = 8; the 7B verifier plateaus after k = 4.
- Data scaling: resolve rises with trajectory count up to 491 with no saturation; dedup-by-instance and repo-ordered subsets are comparable, so the authors read the bottleneck as sampling compute, not task count or repo diversity (a conclusion bounded by 11 repos and ≤ 491 trajectories).

**Failure modes & limits.** Authors: 11 repos, Python only, heavy manual setup, modest self-improvement, no RL. My critique:
- The "diversity is not the bottleneck" conclusion conflicts with SWE-smith's log-linear repo scaling at 100 repos; with 11 repos and one repo at a third of the data, the experiment cannot see repo effects.
- The self-improvement drop is unexplained (no ablation of mix ratio, cap, or the instability of SFT on its own easy successes). It is evidence that naive filtered BC on own samples can degrade, not that on-policy learning fails.
- One Docker image per instance (6 TB) is the cost SWE-smith later avoids.
- Verifier gap (42.8 pass@16 vs 32.0 best@16) shows a learned trajectory judge leaves ~10 pp on the table and is a hackable target if used as RL reward.

**Tips & tricks.**
- Version instances by release and build one environment per version, not per instance.
- Cap successful trajectories per instance at 2–3, preferring shorter ones.
- Train outcome verifiers on a mix of on-policy and off-policy trajectories, balanced success/failure; LoRA may regularise better than full FT.
- Score with normalised YES/NO token probabilities to get a continuous reward.
- Report empty-patch and stuck-in-loop rates beside resolve rate; they move first.
- Report Pass@k and Best@k from 100 random size-k subsamples of M rollouts (Lightman-style), excluding the T = 0 first attempt from the regression.

**Fit to Crucible Part 2.**
- **F9 / F11 (difficulty, pool).** The long-tailed success distribution and the capping result support keeping per-task trajectory caps and the [0.1, 0.6] band: easy tasks dominate any success-filtered dataset unless capped. Adopt "cap per task" as a pool rule for any SFT or replay use of traces. P1.
- **F12 / §3.8 guards.** The on-policy self-improvement collapse (15.3 → 8.7) is another reason the task and generator loops must be accepted on frozen anchors and held-out families, not on the student's own successes. Cite beside 2610.11464. P1.
- **Verifier loop and judges (§3.9 Judge role).** The ORM recipe (on+off-policy mix, balanced, cap 2 per instance) is the right recipe if Crucible trains a trajectory judge for F10 lucky-pass or hack flags; but never as the reward. The Pass@k/Best@k gap is the measure of judge strength σ (§3.8 "σ-gated application"). P2.
- **F1 environment build.** Avoid manual per-version setup; the authors' own criticism of automatic setup (it biases away from complex dependencies) is worth tracking as a **coverage meter**: log which repos and dependency profiles drop at environment build, so the debug lane's repo mix is not silently skewed (the kept set already clusters on Go and four repos). P1, cheap.
- **Task sources (§3.4).** Confirms the low yield of real-issue mining (64,689 raw → 2,438 usable with heavy labour), consistent with the lane's ~1% fix-unit yield; scale must come from synthesis or automated build (SWE-rebench, SWE-Factory).


## 2504.07164 — R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents (Jain, Singh et al., 2025; arXiv v1 April 2025; venue not stated in the text)
Read: FULL (§1–6 and appendices A–E: commit-filter thresholds, install search, test generation, issue back-translation prompt and examples, patch minimisation, SFT and verifier configs, distinguishability and toxicity definitions, attention analysis, example generated tests).  Relevance to Part 2: **high, core.** Two halves, both on point. (1) SweGen turns *commits* (not issue-linked PRs) into tasks by generating missing fail-to-pass tests and back-translating issues from test execution, the nearest published analogue of Crucible's real-fix mining with privileged-gold test generation. (2) The verifier half is a rare quantitative study of *how test-based and learned verifiers fail* (distinguishability, toxicity, attention to self-reported success), which feeds G5, G12, G14 and the F10 judges directly.

**Problem.** Executable SWE training sets are capped by human-written issues and tests (SWE-Gym: 2,438). And at test time, both test-based and judge-based selection saturate early.

**Method / algorithm (SweGen, the data half).**
1. **Repos.** SEART GitHub search for Python repos with many commits (no numeric thresholds given). Full set spans more repos than the 10 used for training (figure only; no count in text).
2. **Commit filtering**, at line and AST-entity level, then an LLM-as-judge filter (prompt and model not given). Rule thresholds on non-test code: ≤ 5 files; ≤ 100 edited lines; patch ≤ 2,000 characters; ≤ 1 deleted entity; ≤ 3 added entities; ≤ 3 edited entities; ≤ 10 statement-level changes. Prefer non-documentation changes with correlated code and test edits.
3. **Historical install by search.** Per commit, parse requirements/setup files, enumerate candidate pins (Python version, numpy, setuptools, …), and run a script that tries combos in order and exits on the first that builds and checks. Docker image per commit. Described as semi-manual.
4. **F2P tests.** Use the commit's own tests: run them at the parent (buggy) and at the commit (fixed); F2P = fail before, pass after. **If the commit has no usable tests, generate F2P tests** with an Agentless-style reproduction-test generator that is **given the ground-truth patch as context** (privileged gold), then keep tests that actually fail before and pass after.
5. **Patch minimisation.** Iteratively delete hunks of the gold patch while the F2P tests still pass, leaving the minimal fix (used for localisation signal; removes style noise).
6. **Issue back-translation.** The LLM gets commit hash and message, the non-test diff, the test-file diff, old (failing) and new (passing) execution output, the relevant test function source, extracted assertion failures, and example issues. Rules: concise; describe the failing input, the nature of failure and expected behaviour; **include brief test code** (truncate > 5–6 lines) but **do not name test functions or files**; never reveal the fix; pick the most informative failing test if several; concrete terms only; human tone. Output: title, description with buggy example code, expected and actual behaviour. Naive back-translation from the diff alone gave generic statements; adding execution traces made them precise.
7. **Decontamination.** Remove repos overlapping SWE-bench's test repos: 8,135 → **R2E-Gym-Subset 4,578** (appendix says 4,538; inconsistency in the paper) over 10 repos. Repo-level only; no n-gram or date check.
8. **Policy SFT.** Minimal OpenHands-derived ReAct scaffold with 4 tools (file_editor, search, execute_bash, submit), no internet. Claude 3.5 Sonnet v2 at T = 0.2, ≤ 40 steps, ≤ 32k tokens, 10-min trajectory and 90-s per-action timeouts; keep successes graded by the environment's tests (real and synthetic) → **3,321 trajectories over 2,048 tasks**. Full SFT with LLaMA-Factory on thoughts + actions, 2 epochs, batch 8, lr 1e-5, warmup 0.1, 20k context.

**Method (hybrid verifier, the test-time half).**
- **Execution-based (EB).** A testing agent (Qwen2.5-Coder-32B SFT'd on 2,203 Claude test-generation trajectories, positive and negative, ≤ 40 steps, 5-min/60-s timeouts) explores the repo and writes one script with M = 10 tests that print "Issue resolved / reproduced / Other issues". A fixed Django in-context starter template helped formatting (~2% of problems). Plus regression tests from the repo. Score: `s_EB_k = Σ_i Pass(P_k, test_i)` if patch k has the maximal regression score among the K, else 0.
- **Execution-free (EF).** Qwen2.5-Coder-14B LoRA (rank 64, 2 epochs, lr 1e-5, 32k) trained on 5,700 balanced YES/NO trajectories (off-policy + on-policy from their 32B, the SWE-Gym recipe). `s_EF = P(YES) / (P(YES)+P(NO))`.
- **Hybrid.** `s_H_k = Top_n(s_EF_k) + s_EB_k`, where Top_n keeps EF scores only for the top-n patches (others −∞); regression filtering is applied after the top-n cut. EF breaks ties among equal test counts; Top_n limits toxic tests to already-plausible patches. Value of n not stated in the text I read.
- **Test-quality metrics (App. C.3).** For test t, incorrect patches P_i, correct P_c: `Distinguish(t) = 1[max_{p∈P_i} Pass(p,t) ≠ max_{p∈P_c} Pass(p,t)]`; `Toxic(t) = 1[max_{p∈P_i} Pass(p,t) > max_{p∈P_c} Pass(p,t)]`; rates are means over the test set.

**Verification / grading.** Training reward is the commit's F2P (and P2P) tests, real or synthetic. Hack surfaces: (a) synthetic tests are written *with the gold patch in view*, so they can encode the specific implementation (overfit to gold), producing false negatives on valid alternative fixes; nothing in the paper measures this. (b) Issues include test code, so the solver sees part of the grader. (c) The EF verifier keys on the agent's own success claims ("Successfully reproduced… verified the fix…"): removing the trajectory and keeping only the patch drops Best@26 from 42.8% to 37.6% (accuracy 71.8 → 68.0%); removing thoughts drops it to 41.4%. Attention windows peak on self-congratulatory summaries in an incorrect sympy trajectory. A policy optimised against such a judge would learn to *claim* verification. (d) Generated tests are weak: for most problems < 20% of tests distinguish the top correct from the top incorrect patch; most are Pass→Pass (do not reproduce) or Fail→Fail (fail even the gold, due to errors in the test); toxic tests reach up to 10% of tests for a minority of problems.

**Scale & cost.** 8,135 tasks total; 4,578 decontaminated over 10 repos; "> 2.5×" more than issue-based collection on the same repos, and "> 3×" more executable environments than prior sets. No dollar, token or CPU figures are reported for any stage, and no per-stage yields (commits screened → filtered → installed → F2P-valid). Inference: 26 editing rollouts (1 at T = 0, 25 at T = 0.8/0.9) and 7 test-agent rollouts at T = 0.8 per problem.

**Key results.**
- SFT on SweGen data, Pass@1 Verified (Qwen2.5-Coder): **7B 19.0, 14B 26.8, 32B 34.4** vs SWE-Gym's 10.6 / 16.4 / 20.6. Lite: 11.0 / 20.7 / 23.8.
- **Synthetic vs real issues, 400 trajectories each: 27.8 vs 28.0%** Pass@1. Back-translated issues are as good as human ones for SFT (agrees with SWE-smith's 7.7 vs 7.8).
- **Thoughts in SFT targets:** 34.4 (34.2 in prose) vs 30.4 without.
- Trajectory scaling 100 → 3,200: 14B saturates near 800; 32B keeps improving with diminishing returns.
- Test-time: EB alone plateaus at 43.7%, EF alone 42.8% (Best@26); Pass@26 = 64.4%. **Hybrid Best@26 = 51.0%** (Best@16 49.4%).
- Ablations inside the hybrid: regression tests only 47.4% (from 42.9%); Agentless zero-shot tests instead of the testing agent 48.8%; without Top_n 49.8%.
- Compute: going 16 → 21 editing rollouts gives 47.6 → 48.4%; adding 5 test-agent rollouts instead gives 49.3%, and test rollouts are cheaper.

**Failure modes & limits.** Authors: installation still semi-manual; EB tests have low distinguishability and rare toxicity; EF verifiers rely on heuristics. My critique:
- **No funnel numbers.** We cannot price SweGen per kept task or know how many commits needed generated tests versus had their own; the 2.5× claim is a figure, not a table.
- **Synthetic-test validity is unmeasured.** Tests generated from the gold patch were never checked against alternative correct fixes or mutants of the fix; the paper's own analysis of the *same kind* of generated tests (at inference) shows mostly non-distinguishing and some toxic tests. Training reward from such tests inherits those defects silently.
- **Decontamination is repo-level only**, and 10 training repos is narrow; the full 8,135 set includes SWE-bench repos and is used nowhere for evaluation claims, which is correct, but the subset still trains on commits that may overlap in time with SWE-bench Verified's distribution.
- **Including test code in the issue** is a choice that SWE-smith found changes behaviour (the student stops writing reproductions). R2E-Gym does not ablate it.
- All training is SFT; none of the verifier insights are tested as RL reward.
- The hybrid ranks *K given patches*; the gain from 43 to 51 depends on Pass@26 = 64%, i.e. on diversity among rollouts.

**Tips & tricks.**
- Mine commits, not issue-linked PRs: back-translate the issue and generate missing F2P tests to unlock ~2.5× more tasks.
- Feed execution traces (failing assertions, old vs new runs) into issue back-translation; diff-only back-translation is generic.
- Minimise the gold patch by hunk deletion while F2P still passes; gives a cleaner oracle and localisation labels.
- Concrete commit-size caps (≤ 5 files, ≤ 100 lines, ≤ 2,000 chars, ≤ 3 edited entities, ≤ 10 statements) keep tasks focused.
- Install historical commits by ordered search over dependency-pin combos, exit on first success.
- Measure every test set's distinguishability and toxicity against known-correct and known-incorrect patches.
- Never let a learned judge see the agent's narrative if it will be optimised against; patch-only scoring is less accurate but harder to sway.
- Use a judge only to break ties among execution-equal candidates, and only within its top-n (limits damage from bad tests).
- Spend marginal compute on extra test generation before extra solver rollouts when selecting.
- Give the test generator a fixed starter template for the framework (settings, fixtures); it fixes formatting failures.
- Train SFT on thoughts, not just actions.

**Fit to Crucible Part 2.**
- **Task source "real-fix mining" (§3.4) and §3.11.** SweGen is the move from "fixes that come with tests" to "fixes, with tests generated from privileged gold, validated by execution". If the debug lane today drops fix units without fail-to-pass tests, this is the most direct yield lever after PR mirroring (SWE-smith). **Adopt with a guard:** a generated test is admitted only if it (i) fails at A′ and passes at the fix (their rule), and (ii) **passes at least one alternative valid fix** (G14) and (iii) **kills the commit-revert and generic mutants** (G5). Without (ii)–(iii) generated tests overfit to the gold. P0 for the lane's supply, P0 for the guard.
- **F5 / G5 / G14 metrics.** Make distinguishability and toxicity, as defined in App. C.3, first-class verifier-quality numbers in `GateReport`: over the oracle, the alternatives (correct set) and the negatives (incorrect set), every check must be distinguishing for at least one negative, and **toxicity must be 0** (any check that passes a negative while failing a valid alternative is a verifier bug). This generalises beyond code to state checks. P0, cheap given F5 already builds both sets.
- **F4 oracle minimality and G6.** Patch minimisation is the code-world version of "every planted change matters": a hunk that can be removed with F2P still passing is either noise or an under-tested part of the fix. Run minimisation before G6; a fix hunk that survives deletion means either drop it from the oracle or add a check that needs it. P1.
- **G11 / report writer (§3.9).** Adopt the back-translation inputs (assertion failures, old/new runs) for the debug lane's report writer, but follow SWE-smith rather than R2E-Gym on test code: **do not quote hidden test code** in the report; describe input, failure and expectation in prose. Add a G11 lint that rejects reports containing hidden-test identifiers or verbatim assertion lines. P0.
- **F10 judges and §3.8 verifier loop.** The EF finding is the clearest evidence for a Crucible rule: **any judge used for lucky-pass, hack or intent labels must score the end state or patch with the agent's self-narrative stripped**, or at least be ablated with it stripped; a judge whose accuracy depends on the narrative fails the frozen-anchor canaries. Add a "self-report canary" to the anchor suite: an incorrect trajectory with a confident success summary must be scored NO. P0.
- **G12 attacker.** "Claim success loudly" is a cheap attacker strategy to include in the attacker's repertoire for any task where a judge contributes to reward. P1.
- **§3.6 difficulty and §3.10 cost.** The trajectory-scaling saturation differing by size (14B at ~800 vs 32B still rising) is weak evidence that larger students need more and harder data from the same pool; consistent with the 35B vs 397B band question but not a band schedule. Provides no cost inputs for the formula (a gap to note).
- **Avoid:** repo-only decontamination for benchmark claims (Crucible's G11 date rule and owner-disjoint held-out are stronger); semi-manual pin search as a scaled path (SWE-rebench and SWE-Factory automate it).
- **Interface implied:** `VerifierBundle.quality = {distinguish_rate, toxicity_rate, per_check: {kills: [negative_id], passes_alternatives: bool}}`, computed in F6 from `NegativeSet` and `alternatives()`; and a `TestSynth` step in the mined source that emits candidate checks tagged `provenance=generated_from_gold`, which G5/G14 must clear before they count.


## 2505.20411 — SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents (Badertdinov, Golubev et al., 2025; Nebius; arXiv v2 Nov 2025; venue not stated in the text)
Read: FULL (§1–5 and appendices A–M: install prompts and recipe schema, failure modes, the 18-instance install study, quality-rater classification report, refinements to SWE-bench, benchmark filters, agent system prompt, eval setup, schema, funnel, dataset statistics).  Relevance to Part 2: **high for the mining source and for G2/G10/G11 practice; medium overall.** It is the fully automated, large-scale version of "real issue + real tests → executable task", with a per-stage funnel and concrete refinements (head-commit diffing, flake reruns, dependency freezing, date-based decontamination) that map almost one-to-one onto Crucible's debug-lane gates. It has no training results of its own and no fault synthesis.

**Problem.** Executable, verifiable SWE tasks are scarce because environment setup is manual, and static benchmarks become contaminated as models train on them. The paper wants a pipeline that runs continuously with no human in the loop, serving both RL training data and a fresh benchmark.

**Method / algorithm.** Four stages, run on a distributed platform (TractoAI) so the whole funnel can be re-run when the pipeline changes.
1. **Preliminary collection.** GitHub Archive (issues, discussion, linked PRs, labels, merge status) plus full local clones (avoids API limits, enables version analysis). Scope: ~450K issue-linked PRs created before 2025-05-01 from > 30K repos with permissive licences (MIT, Apache-2.0, BSD-2/3/4, ISC, CC0, ZPL-2.1, manually checked "Other") and > 75% Python LOC. Filters: issue resolved; PR merged to main; PR linked to exactly one issue; issue text > 10 chars; PR adds or modifies tests **and** changes non-test code; 1–15 files touched. Split the diff into solution patch (non-test) and test patch (test files). → ~153K candidates.
2. **Automated install configuration (agentless).** Group tasks by version from `git tag`, normalised to major.minor (available for ~95% of tasks; the rest get their own environment). Per group, use the PR base commit with the **most recent date** (later commits within a minor version tend to carry environment fixes). Qwen2.5-72B-Instruct, three calls: (a) from the repo file list, pick ≤ 10 install-relevant files, root first; (b) from their concatenated content (truncated to 100K chars), reason then emit a JSON recipe {python, packages: requirements.txt | environment.yml | list, install (always a local `pip install -e .[dev]`-style command), test_cmd (direct pytest with `-rA --tb=line`, no placeholders), pre_install apt commands, reqs_path, env_yml_path, pip_packages}, executed by a fixed bash template (clone, checkout, pre_install, conda create, install deps, install, test); (c) on failure, give the recipe, the tail of the error log and the files, ask for < 100 words of diagnosis and a full corrected JSON. Up to 3 candidate recipes per task, each refinable. An interactive install agent was sometimes better but much more expensive.
3. **Execution-based verification.** Build in an isolated container with buildah on tmpfs, from a base image with conda/gcc, using internal PyPI/APT mirrors. Run the test-patch tests before and after the solution. Valid iff (1) ≥ 1 test-patch test fails before, (2) every test-patch test that failed before passes after, (3) every test-patch test that passed before still passes. **Run tests several times and drop any task that fails in at least one run** (flake guard). After success, **freeze exact dependency versions** (pip freeze, conda env export) and store them in the instance for reproducible rebuilds.
4. **Automated quality assessment.** Qwen2.5-72B-Instruct fine-tuned on the ~3,800 SWE-bench Verified human annotations (input: issue, gold patch, test patch), one binary label per head, 75/25 split (413 validation): **Task Complexity** (> 1 h) 81% acc (vanilla 68%); **Issue Clarity** 79% acc but recall on the minority class only 0.20; **Test Patch Correctness** 67% acc, recall on "high-score" 0.42. Labels are shipped as metadata, not used as hard filters for the training set.

**Refinements over SWE-bench (App. G), each a validity fix.**
- Diff the PR **head commit** against base, not the merge commit, so main-branch merges do not inject unrelated changes or tests (the reason sympy-14821 was excluded from Verified). Fall back to merge commit and record it.
- Drop deleted test files from test directives.
- Run tests with `--tb=line` (not `tb=no`) so AttributeError/ImportError in the pre-fix run is visible; flag tasks whose tests reference not-yet-existing names (agent must guess exact signatures). Flagged, not dropped, since some are valid.
- Freeze dependencies on first success.

**Benchmark construction (App. H).** From fresh tasks: no AttributeError/ImportError before the fix; ≤ 3 files; solution ≤ 500 words; issue 16–1,000 words, English; issue created in 2025; LLM difficulty < 3; ≤ 50 F2P tests → 294 tasks from 169 repos. All models run by the authors with one fixed ReAct text-command scaffold (no function calling), identical prompt, developer-recommended sampling, 128K context, **5 runs**, reporting mean, SEM and pass@5. Tasks whose issue predates a model's release are flagged as possibly contaminated. The agent prompt forbids editing or adding tests and ends with a self-check list the agent must answer before submitting.

**Verification / grading.** SWE-bench semantics (F2P from the PR's test patch must pass; P2P must stay passing), with hidden tests. Weak points: real PR tests can be implementation-specific (their own rater estimates test-patch correctness at 67% accuracy, so many unfit tests remain); the training set is not filtered on any quality label; "fails in ≥ 1 of N runs" catches flakes only up to N (N unstated). The ImportError flag addresses one known false-negative class (signature guessing).

**Scale & cost.** Funnel (App. L): ~10M PRs, 6M issues, 32K repos → ~450K matched (~5% of PRs) → ~150K filtered (~33%) → **~21K valid (~14%)** → 21,336 annotated tasks from **3,468 repos**. Install success: a working recipe for at least one task in **31% of repos**. Install study on 18 SWE-bench instances: agentless 1 candidate 6/18, 3 candidates 8/18, 10 candidates 9/18; interactive agent 8/18; several failures were log-parser failures (Django's custom runner). Dataset medians: issue 91 words, 2 files, 37 lines, 2 F2P, 22 P2P; means are much larger (3.46 files, 142 lines, 14.6 F2P), so a long tail. No dollar or GPU-hour cost for collection is given; serving the evaluated open models used 2 nodes × 8 H200, ~7 h per full DeepSeek-V3 run.

**Key results.**
- Contamination signal: DeepSeek-V3-0324 39.7% Verified vs 21.3% on Mar–Apr 2025 SWE-rebench; V3-1226 35.2 vs 21.9 (the two versions diverge on Verified but tie on fresh tasks). Llama-3.3-70B 18.1 vs 11.2; Qwen2.5-Coder-32B 4.9 vs 3.2. Every model drops; the authors read the *gap pattern*, not the drop, as the contamination hint (difficulty differs between sets, so absolute drops are not evidence by themselves).
- GPT-4.1 is the only model clearly lower on Mar–Apr than Jan (31.1 → 26.7%).
- Qwen3 think vs no-think: no gain from thinking (e.g., 235B 12.2 vs 16.6% Mar–Apr).
- High pass@5 / low mean (Llama-4-Maverick 12.2% mean, 27.6% pass@5) marks inconsistency; SEMs are 0.3–1.8 pp.
- Files-changed as difficulty proxy (DeepSeek-V3-0324, Jan–Jul 2025 leaderboard): 1 file 28.6% (n=201), 2 files 20.6% (n=144), ≥ 3 files 17.5% (n=64).
- Qwen2.5-Coder-32B fails mostly by hallucinating environment responses and looping on format errors.

**Failure modes & limits.** Authors: automated quality checks admit underspecified or unsolvable tasks, lowering ceilings; Python only; the install prompt was tuned on 18 repos. My critique:
- **No training evidence in the paper.** The RL-suitability claim rests on citations of others' use; there is no experiment showing that training on SWE-rebench beats a smaller curated set, or that the quality labels predict training value.
- **The quality rater is weakest exactly where RL needs it.** Test-correctness recall on the "bad tests" side is not reported in a way that shows bad tests are caught; at 67% accuracy, a large share of the 21K may carry tests that reject valid alternative fixes, which in RL is a silent false-negative reward.
- **Install success biases the mix** (31% of repos), toward simple, well-packaged projects, the bias SWE-Gym warned about. No coverage report by dependency profile.
- **Decontamination is date-based against model release dates**, which ignores that a fix's code may be on GitHub before the issue date and that pretraining cutoffs differ from release dates.
- The benchmark's "LLM difficulty < 3" filter shapes the benchmark toward what the rater calls easy, which may compress the range where models differ.

**Tips & tricks.**
- Diff the PR head commit against base, never the merge commit.
- Group by major.minor version and pick the newest base commit per group for the shared environment.
- Generate several install recipes and refine each on its error log; 3 candidates captured most of the gain in their study (8/18 vs 9/18 at 10).
- Use a fixed JSON recipe schema run by a fixed template, so the LLM never writes arbitrary shell.
- Freeze dependencies on first success; keep package mirrors.
- Rerun tests N times; reject on any failure.
- Run tests with tracebacks on so missing-name errors are visible; flag tests that require guessing new signatures.
- Keep a per-stage funnel table; re-run the whole funnel when a stage changes.
- Benchmark protocol: fixed scaffold, ≥ 5 runs, mean ± SEM and pass@k, contamination flags by date.

**Fit to Crucible Part 2.**
- **Mined source (§3.4) and §3.11.** The debug lane already does most of this; the specific refinements to check are present or adopt: **head-commit diffing** (fix-unit extraction must not include merge noise), **ImportError/AttributeError flag** (a fault whose hidden test needs a not-yet-existing name is a spec-guessing task, a G8 communicability failure in generic terms), and **freeze-on-success** (part of G2 determinism across rebuilds). P0, cheap.
- **G10 flake.** Their "fail once in N runs → drop" is cruder than Crucible's 60-grade Clopper–Pearson bound; keep ours. But adopt their attribution of flakes (external API calls, exact float equality) as **leak/flake lint rules** in the DomainSpec, run before the expensive 60-grade loop (cheap-before-paid). P1.
- **F1 environment build for new repos/languages.** The 3-call agentless recipe (file pick → JSON recipe → error-driven refinement, ≤ 3 candidates) is the cheap first tier; escalate to an agentic builder only for high-yield repos (their own suggestion). This fits the `LangProfile` plug-in table: each language supplies the recipe schema and the fixed template. P1.
- **Coverage meter (§3.5).** Their 31% repo install rate is exactly the kind of silent skew to report: log install failures by repo, language and dependency profile, and count them against cell fill. P1.
- **G11 contamination.** Date-based contamination flags are the right shape for the sealed benchmark (Crucible already uses a date rule); add their **model-release-date flag per evaluated model** to benchmark reports, and keep owner-disjoint held-out (stronger than theirs). P1.
- **F9 predictor.** The fine-tuned rater recipe (Verified annotations → complexity, clarity, test-correctness heads) is a template for Crucible's advisory predictor, but their numbers (67% on test correctness) say a text-only rater cannot replace G5/G14 executed checks. Use such a rater only to rank, never to admit. P2.
- **Benchmark protocol (sealed benchmark).** Five runs with SEM and pass@k, fixed scaffold, no function-calling differences: adopt as the reporting rule for the sealed benchmark and held-out families. P1.
- **Avoid:** shipping quality labels as metadata without gating; training on unscreened real tests. Crucible's G5/G14 should apply to mined tasks too (do the hidden tests accept an alternative valid fix?).


## 2506.10954 — SWE-Factory: SWE Data Construction, Automatically! (Guo, Wang et al., 2025; Sun Yat-sen University and Huawei; arXiv v3 Jan 2026, ACM template, venue not stated)
Read: FULL (§1–10; the agents' prompts are in an online appendix PDF on GitHub that is not part of the arXiv text, so I have not read them).  Relevance to Part 2: **medium-high for F1 environment build in the debug domain, low elsewhere.** It is the multi-language (Python, Java, JS, TS) automated environment builder with a role-split agent loop, a memory of past environments, and per-language yields and costs. Crucible's lane yields 0 on Java today and is Go-heavy, so a measured 30–55% build rate on Java/JS/TS at ~$0.05 per issue is directly useful. Its verifier novelty (exit codes) is a simplification Crucible should mostly not adopt.

**Problem.** The SWE-bench construction pipeline has three manual or broken steps: binary files in test patches arrive empty from the GitHub API; environment setup is manual per repo and version; fail-to-pass validation needs hand-written log parsers per repo and version.

**Method / algorithm.**
1. **Binary test-file recovery.** Detect "Binary files … differ" placeholders in the test patch by pattern, emit download commands (e.g., wget of the file at the PR commit) into the test script, and strip the broken binary hunks so `git apply` succeeds. Affected share: Pillow 30.4%, Manim 27.3%, matplotlib 22.1% of instances; without it these tasks silently fail F2P and vanish.
2. **SWE-Builder, four agents** (base LLM at T = 0.1, ≤ 5 outer iterations, ≤ 10 explorer rounds, 20 parallel processes):
   - *Repository Explorer*: loop of tool calls `browse_file(fp, q)` (an LLM extracts the query-relevant part of a file), `browse_dir(fp, depth)`, `search_keyword(kw)`; collects dependencies (requirements.txt, pom.xml, …), test commands (pytest, mvn test, …) and setup notes (CONTRIBUTING, scripts); stops when it judges the summary sufficient.
   - *Environment Manager*: writes or updates a Dockerfile; keeps its generation history; falls back to the previous Dockerfile on a failed generation.
   - *Test Manager*: writes the evaluation shell script (apply test patch, fetch binaries, delete removed files, run target tests) and must end it with `rc=$?; echo "OMNIGRIL_EXIT_CODE=$rc"`.
   - *Test Analyst*: `build_image` → `start_container` → `run_eval` **with the gold patch applied**; success iff the target tests pass. On failure it reads build or test logs and routes targeted guidance to only the responsible agent(s) (explorer, Dockerfile or script), which are reset and rerun.
   - *Evaluation Environment Memory Pool*: every validated (Dockerfile, script) is stored per repo; a new issue retrieves the setup from the **closest version** of the same repo as the starting reference.
3. **Exit-code F2P validation.** Run the script before and after the gold patch; parse the marker; valid iff non-zero before and zero after; missing marker → error.
4. **Training data (exploratory).** 10 Python repos not in SWE-bench → 2,877 instances with GPT-4.1-mini → 2,809 after installing the DeepSWE agent's four tools into each image. One Kimi-K2 trajectory per environment (T = 0.2, ≤ 40 iterations), **no success filter is described**. Full-parameter SFT with MS-Swift on 8 A800, "3 iterations", 65,536 sequence length, lr 1e-5, warmup 0.05, packing with YaRN, sequence parallelism 2–8. Eval with DeepSWE scaffold at T = 0, ≤ 40 iterations, 32k context.

**Verification / grading.** One bit per run: the exit code of the target test command. Weaknesses (mine; the paper does not discuss them):
- **No per-test granularity.** F2P and P2P collapse into one status, so a gold patch that fixes the target and breaks an unrelated test in the same command is still "fail after"; worse, a test command that runs only the new tests never checks regressions at all. Multi-fault tasks lose per-fault milestones.
- **"Fail before" is not "fails for the right reason."** A non-zero exit before the gold patch is satisfied by collection errors, ImportErrors on names the fix introduces, or a broken script; SWE-rebench flags exactly this class and SWE-Factory does not.
- **The Test Analyst iterates until the gold passes.** An agent rewriting the test script toward "gold passes" can narrow or skip tests, and nothing measures test strength (no mutant, no alternative-fix or distinguishability check).
- The human check (F1 0.99 over 1,201 instances; 2,380 logs at F1 1.0) validates *that the exit code matches the log's pass/fail*, not that the task is a good task.
- Script-generation errors: 16 false negatives, 4 from a typo in the marker, 12 from `set -e` exiting before the echo; catchable by lint.

**Scale & cost.** SweSetupBench: 2,441 pre-2025-03-01 issues from 12 repos (> 2.5k stars) → 20% stratified by version (≥ 1 per version) = **671 issues** (Python 198, Java 186, TS 176, JS 111). Results (valid F2P, manually verified; output rate; min/issue; $/issue):
- GPT-4.1-mini: **50.2% (337/671)**, 64.8%, 26.3 min, **$0.047**.
- Kimi-K2: 47.8%, 63.2%, 30.2 min, $0.056.
- DeepSeek-V3-0324: 42.0%, 53.4%, 23.0 min, $0.037 (11.4 valid tasks per $, the best rate; GPT-4.1-mini 10.7, Kimi 8.5).
- Per language (F2P %, GPT-4.1-mini / Kimi / DeepSeek): Python 54.0 / 54.0 / 54.5; Java 43.5 / 33.9 / 30.1; TS 55.1 / 56.3 / 42.0; JS 46.8 / 46.8 / 39.6. (The count fractions printed beside the per-language percentages in the extracted table are garbled, e.g. "181/186" for 43.5%; I use the percentages only.)
- Ablations (GPT-4.1-mini, F2P): no Repository Explorer 42.9% (cost falls to $0.021); no Memory Pool 45.0% (mean −5.2 pp over the three models, more time); **no execution feedback 0.3%** (static review of Dockerfile and script only; Kimi 2.7%, DeepSeek 0.7%). Binary recovery on the 44 binary-patch issues: 45.5–50% with it, **0% without**.

**Key results (training).** Verified resolve rate base → SFT: Qwen2.5-Coder-3B 0.0 → 3.4; 7B 0.0 → 13.6; **14B 5.8 → 21.0**; Qwen3-8B (non-thinking) 3.4 → 16.2; Llama-3.1-8B 1.2 → 5.2. Lite: 14B 4.3 → 14.7. Most of the small-model gain is format: empty-patch rate falls from ~91% to 4–16% and tool-call failures from ~80–93% to 1–9% for the 3B/7B. No comparison against training on other datasets at equal size, and no seeds or error bars.

**Failure modes & limits.** Authors: 12 repos in the evaluation set, ≤ 14B students, stochastic script errors. My critique, beyond the verification points above:
- The training experiment cannot attribute the gain to SWE-Factory's data versus any 2.8K agent trajectories from Kimi-K2 (no baseline dataset); without a success filter, some trajectories may be failures.
- "Fully automated" still needs manual repair for the 16 parse failures and depends on gold-patch-pass as the only quality bar.
- The memory pool's nearest-version retrieval can propagate a wrong-but-passing environment across a repo's versions.
- Repository Explorer's benefit is smallest for Kimi-K2, which the authors attribute to memorised repos: an install agent may succeed by memory on popular repos and fail on the long tail Crucible needs for held-out families.

**Tips & tricks.**
- Detect "Binary files … differ" in API-fetched diffs and fetch the blobs; or diff from a local clone, which keeps binaries.
- Split environment building into explorer, Dockerfile writer, test-script writer and analyst; route failures only to the responsible role.
- Execution feedback is non-negotiable; static review of a Dockerfile gets ~0%.
- Keep a per-repo memory of validated environments and start from the nearest version.
- Make every generated script print a unique, greppable status marker; lint scripts for `set -e` and marker typos before running.
- Track valid tasks per dollar per builder model; the cheapest model can win on that metric.

**Fit to Crucible Part 2.**
- **F1 for the debug domain (`LangProfile` plug-ins, §3.11).** Adopt the four-role builder and the memory pool as the environment-build stage for new repos, especially Java/JS/TS where the lane yields nothing today. Map roles onto §3.9: Explorer and Environment Manager are the "schema and tool coder" analogue, the Test Analyst is a builder-side checker, so add Crucible's family rule: the role that edits the test script must not be the one that approves it (the paper has no such separation, and its loop optimises toward gold-passes). Prefer cheap builder models scored by valid tasks per $ under the swarm preset. P1.
- **G2 / G10 / G5 still decide.** SWE-Factory's validity bar is weaker than Crucible's: keep per-test parsing for F2P, P2P and per-fault milestones, and use the exit-code marker only as a **fallback for unknown runners**, logged as reduced-granularity. Add a G-check that the pre-fix failure is not a collection/ImportError (SWE-rebench's flag). Run G5 mutants and G14 alternatives on every built environment so a narrowed test script is caught. P0 (as a rule for any adopted builder).
- **Generic World analogue (F1).** The pattern "build artefact → execute → analyst routes failure to the responsible sub-builder → memory of validated builds" is the same loop F1 needs for Apps (schema/tool coder, blind tester, debugger). The no-execution-feedback collapse (50% → 0.3%) is strong evidence for §3.0 rule 2 (execution decides). Cite it there. P1.
- **G11 / coverage.** Binary-file loss is a silent selection bias against image and visualisation repos; if Crucible's miner reads diffs via the API anywhere, check for it; if it uses local git, note it as covered. P2, cheap.
- **§3.10 cost inputs.** $0.037–0.056 and 23–30 min per issue for environment + F2P at 42–50% yield, per language. Use as the F1 cost line for the mined source in the cost formula.
- **Avoid:** using its training result as evidence of data quality (no baseline), and gold-pass-only validity.


## 2510.10197 — Don't Just Fine-tune the Agent, Tune the Environment (Lu et al., 2025; arXiv v2 Jan 2026, Ant Group AWorld / Zhejiang U.) [Environment Tuning]
Read: FULL (main text, App. A–G including both augmentation prompts, progress-reward definition, training details, KL ablation and all case studies).  Relevance to Part 2: medium-high. It is not a task-synthesis paper (it trains on 400 existing BFCL V3 tasks), but it is the cleanest evidence for two §3.6 ladder levers: **hint-giving environment errors** that are generated by an LLM and policed by a leak judge, and a **staged scaffold that is removed before evaluation**. Its per-turn progress reward is directly a milestone Φ for multi-turn World tasks.

**Problem.** Multi-turn tool-use RL from very few tasks (BFCL V3 multi-turn has 800 samples, 8 domains, 84 tools). SFT on synthetic trajectories overfits; plain GRPO cold-starts badly and collapses (Qwen2.5-7B single-stage RL on 400 tasks: gradient-norm explosion and accuracy collapse after about 70 steps, peak gain only 10–15 pp near step 60).

**Method / algorithm.** Three levers, scheduled by a 4-stage curriculum. Train set: 100 tasks from each of the four BFCL V3 splits (Base, Missing Function, Missing Parameter, Long Context) = 400; the other 400 are the in-distribution test.

*Stage 1 — syntax only.* A small data slice (size unstated). Per-episode counters: C_correct (right tool name, valid args), C_error (right name, invalid args), C_format (turns breaking the XML-like `<think>/<tool_call>/<answer>` format). R_format = (N − C_format)/N over N turns; R_tool = C_correct/(C_correct + C_error); R_stage1 = I_tool · (R_format + R_tool), where I_tool = 1 iff at least one tool call was attempted (kills "void turns").
*Stage 2 — Base split only*, progress reward R_P, augmented environment ON.
*Stage 3 — all four splits*, R_P, augmentation ON. System prompt adds two rules: try other tools before asking the user; never hallucinate a call when a function or parameter is missing, ask instead.
*Stage 4 — all splits, R_P, augmentation OFF*, so the policy learns to act on the standard (uninformative) errors it will see at evaluation.
*Stage transition rule:* advance only when validation accuracy plateaus **and** the gradient norm is stable (qualitative; no numeric thresholds given).

*Progress reward.* For each user turn t of T: r_t = r_t^state · r_t^exec, both binary. State check: environment state after the turn equals the ground-truth state after that turn (outcome, not path). Exec check: for read-type calls, the returned values match the expected ground-truth results. R_P = (1/T) Σ r_t. The direct-GRPO baseline uses 0.9·R_P + 0.1·R_format from the start.

*Actionable environment augmentation (App. B).* Goal: replace cryptic errors ("No available route", "No such file or directory") with messages that name the root cause and a direction, not the answer ("Invalid airport code[s] … use another tool to find the code"; "Paths are not allowed. Specify only file/directory name"). Two LLM roles, few-shot (3–5 hand-written domain examples bootstrap a whole domain):
1. **Augmentation Generator.** Input: domain, question, trajectory, tool name, the call, the original error, few-shot pairs. Output JSON (reasoning first): need_augmentation; error_category ∈ {dependency_missing, parameter_constraint, state_prerequisite, format_mismatch, resource_unavailable}; augmentation_type ∈ {clarify_constraint, suggest_prerequisite, explain_dependency, correct_format}; augmented_feedback. Guidelines: root cause, directional guidance, keep exploration space, pedagogical.
2. **Constitutional Judge.** Sees the same plus the augmented text **and the ground-truth calls** (for judging only). Four booleans: solution_leakage (gives the exact call, concrete parameter values or the whole sequence), exploration_constraint (only one path allowed), actionability, hint_quality. Approve iff no leakage, no over-constraint, actionable, high quality.
3. Loop: generator → judge → retry with the judge's suggestions, at most 3 attempts.
4. Which errors get augmented: during early training, tasks the agent repeatedly fails are flagged automatically; their error trajectories feed the workflow. The paper does not say clearly whether approved hints are compiled into the tool code (they look like edited error strings in the case studies) or looked up at run time. I read it as an offline edit of the environment's error messages.

*Optimiser.* GRPO-style group-normalised advantage, PPO clip decoupled (ε_low 0.2, ε_high 0.28, DAPO style), **KL penalty β = 0.1** to the reference (ProRL style). Group size, batch size, LR, steps, rollout length and hardware: not reported.

**Verification / grading.** BFCL's own checker: per-turn state comparison plus execution-result comparison against ground-truth call results. Hackable or weak points: (a) exec-result matching needs the expected return values, so it is single-reference for reads; (b) the progress reward gives partial credit for early turns, so a policy can farm easy turns (fine for training, but it is not the eval metric); (c) the leak judge sees ground truth and is an LLM, so leakage is policed by opinion, not execution; nothing checks that the augmented environment still behaves identically on correct calls.

**Scale & cost.** 400 training tasks; 3–5 few-shot hint examples per domain; ≤3 generator–judge rounds per hint. Count of augmented messages, compute, wall time, tokens and GPU hours: not reported. Models for the generator and judge: not named.

**Key results.** (BFCL V3 multi-turn, held-in 400, accuracy %)
- Qwen2.5-7B-Instruct 7.00 → 36.92 (ToolRL baseline 18.00). Llama-3.1-8B 5.48 → 28.25 (ToolRL 11.25). ToolACE-2-8B 37.99 → 47.18 (ToolRL **drops** it to 33.75). watt-tool-8B 35.74 → 54.34 (ToolRL 42.00). xLAM-2-8b (SFT on synthetic data) is still higher in-distribution at 70.50.
- Curriculum ablation, Qwen2.5-7B: base 7.00; direct GRPO 17.42; +Stage 1 15.50 (syntax alone is slightly *below* direct GRPO); +Stage 2 25.83; +Stage 3 32.00; +Stage 4 36.92. Removing the scaffold (Stage 4) still adds 4.9 pp.
- Augmentation ablation (figure only): over 20 pp gain on the Missing Parameter and Missing Function splits.
- Progress vs binary reward (figure only): no visible difference on Base in Stage 2; with binary reward the Missing Parameter/Function splits **fail completely (near 0)** in Stage 3.
- KL ablation (Stage 2, figure): β = 0 → entropy collapse and accuracy drop; β = 0.001 only delays it; β = 0.1 keeps entropy and keeps improving.
- OOD (Table 2): SFT models collapse (xLAM-2 web search 5.00 despite 70.50 in-distribution). Env Tuning on Llama-3.1-8B: BFCL V4 avg 8.46 → 16.53 (web search 1.00 → 15.00); τ²-bench 20.00 → 21.67; ACEBench Agent 1.65 → 4.17. On ToolACE-2: ACEBench 8.34 → 15.00 and τ² 10.83 → 20.83, but BFCL V4 16.79 vs ToolRL's 18.91. The OOD gains are small and mixed. Step sizes of 2.5/5 pp imply about 20–40 tasks per τ² domain, so most OOD differences are within noise.

**Failure modes & limits.**
- Authors: the curriculum and feedback are hand-scheduled; automating both is future work.
- **Possible test leakage in checkpoint choice:** validation is "100 samples from the remaining BFCL data", and the remaining data is the 400-task held-in test set. Stage transitions are chosen on it.
- No seeds or error bars; OOD sets are tiny; key ablations are figures only.
- The augmented hints include fairly directive text ("You can use alternative tool to find the correct airport code"), close to the judge's own leak line. The judge's precision is never measured.
- Stage transitions are judgement calls (plateau and "stable" gradient norm), so this is not a reproducible controller.
- The case-study gold calls contain codes ('CRH', 'PHV') that differ from what the environment returned ('CWH', 'PHW'), which hints at noisy ground truth in BFCL itself.
- Gains come on the training environments' own domains; it shows learnability from few tasks, not a generator of tasks.

**Tips & tricks.**
- Give the student a syntax-only warm-up whose reward counts valid calls and well-formed turns, gated on at least one tool call.
- Turn opaque tool errors into root-cause hints for too-hard tasks, then **switch them off** for the last stage so the policy does not depend on them.
- Generate hints with an LLM from a few hand-written exemplars. Approve each with a judge that sees ground truth and rejects exact calls, concrete values and single-path prescriptions. Retry at most 3 times.
- Per-turn state × exec product, averaged, is a cheap Φ that rescues tasks that a binary reward leaves at zero.
- In long multi-turn GRPO, keep a real KL penalty (β ≈ 0.1); β = 0.001 did not prevent entropy collapse here.
- Watch the gradient norm as an early warning of collapse; advance curricula only when it is stable.

**Fit to Crucible Part 2.**
- **§3.6 ladders (F9 → task loop), P1.** Add `hint_level` as a real knob with a generated implementation: `simplify(knob="hints")` returns a variant whose World tools emit root-cause errors. Build it as a **World-side error-message overlay** (a versioned table keyed by tool and error class), not as prompt text, so the scaffold can be switched off per task and per stage without touching the verifier. The ladder schedule is: hints on while p̂ < band low, hints off as p̂ rises; the pool records both variants (`shares_fixes_with`).
- **Leak gate on hints (G11 extension).** The constitutional judge's four criteria (no exact call, no concrete values, no full sequence, not single-path) are a ready rubric for any scaffold text, including QuestA-style partial hints. Run it with a different family from the hint writer. Add an **executed** check that the judge lacks: replay the oracle with hints on and off; the end state and grade must be identical (the overlay must change messages only, never behaviour). Also run G9 with the hint overlay on: a solver given only the instruction and hints must still fail.
- **F5 milestones / Φ.** The per-turn state-and-exec product maps directly onto `milestones` in the `VerifierBundle` for multi-turn tasks with a simulated user. Use it for shaping only; the pool and benchmark grade stays the full conjunction.
- **F1 World quality.** Their finding is an argument that the World's error messages are part of task quality: G13 (interaction completeness) should also flag tools whose errors cannot distinguish causes (for example "not found" for both a missing file and a disallowed path). A World-loop signal: clusters of student failures right after an ambiguous error.
- **F12 → ladders.** Their hint source is "tasks the agent repeatedly fails" plus their error trajectories. That is exactly our failure-cluster feed. Route clusters with an ambiguous-error signature to the hint generator instead of to a new task.
- **§2(a) RL settings.** Evidence for keeping KL in long multi-turn runs and for a gradient-norm alarm; P2 for Part 2 itself.
- **Avoid:** choosing curriculum transitions on the test split; and treating the 36.92 as evidence that hints transfer OOD, since the OOD gains are small and noisy.


## 2506.10055 — TaskCraft: Automated Generation of Agentic Tasks (Shi et al., 2025; arXiv v2, OPPO PersonalAI)
Read: FULL (sections 1–7 and App. A–E: data statistics, the depth-extension verification rules with their failure examples, the atomic-verification scoring prompt, the three optimised prompts, the superset-verification prompt, training details). The paper is short; some prompt bodies and all output formats are elided ("…") in the source.  Relevance to Part 2: **high** as a pattern for F3/F4 difficulty-by-construction (depth = linkage, width = count) and for the two cheapest intent probes we have (G9 no-shortcut: agent-with-tools must beat the same model without tools; leak check after each composition). Medium as evidence: the numbers are thin, the domain is read-only web/PDF/image QA, and every check after the atomic stage is an LLM opinion, not execution.

**Problem.** Agentic benchmarks (GAIA 466 Q, BrowseComp 1,266, HLE 2,500 from about 1,000 experts) cannot scale by hand. Asking an LLM to write a tool-needing question for an answer gives low tool-use rates, uncontrolled difficulty and inconsistent verification. TaskCraft builds tasks bottom-up from documents, so difficulty grows by composition and the answer and trajectory are known by construction.

**Method / algorithm.**
*Notation.* A tool T takes an input index i_T (page title, PDF name, image path, song name) and returns content C. The model infers a relation R between C and the answer a. A question is q = f(i_T, R) → a, where f is an LLM that verbalises.

*1. Corpus.* Unlabelled seed documents by modality: web pages 75% (recent news across domains), images 15% (charts and tables from financial reports and papers, filtered to images carrying information beyond their caption text), PDFs 10% (English financial reports and papers). "Up-to-date news" is their contamination guard, implicitly.

*2. Atomic task (one target tool call).* For each document: take i_T (title/path); run the tool to get C; an LLM extracts **atomic conclusions** from C (optimised prompt rules: one indivisible fact each; must contain a definite identifier such as a number, date or unique name; time-sensitive facts carry explicit time ranges; inline any cited text); pick candidate answer a and relation R; write q from (i_T, R). The assumption is an "ideal search engine": i_T is specific enough that finding the source is finite navigation; search and file-system steps are not counted as hops.

*3. Atomic verification (the only agentic check).* Run a **tool agent** with at most about 3 tool steps, and separately an **infer-LLM with no tools**. A judge-LLM scores each answer vs the gold on 0/1/2 (2 semantically equivalent; 1 contains all gold info plus valid extras; 0 omits or contradicts). Keep iff AgentScore > LLMScore **and** AgentScore > 0. This is rejection sampling for "really needs the tool".

*4. Depth extension (n hops → n+1).* Goal: hide the current index i_T^n behind a new lookup.
- A search agent retrieves pages that contain i_T^n and an LLM picks a **minimal unique superset** i_T^{n+1} plus relation R^{n+1} (prompt examples: a paragraph → its containing text; a term → its discipline; a date → its week or month; a short event → the larger event; a page → its parent page; "Avatar 3" → "Avatar film series", relation "the third film"; ≤3 keywords per search; one relation; avoid strongly specific proper nouns).
- Build the intermediate question q̂^{n+1} = f(i_T^{n+1}, R^{n+1}) → i_T^n (its answer is the old index).
- Merge: q^{n+1} = f(q̂^{n+1}, R^n) → a, i.e. substitute the old index in q^n with the description, keeping all earlier relations. Merge prompt rules: slightly more complex than the core query; the answer stays unique; **the answer is concealed**; polish without changing proper nouns. About 20 few-shot examples.
- The trajectory is recorded while extending: the chain of (q̂, tool, result) steps is the gold trajectory, so tasks can exceed the generator agent's own ability (reverse construction).

*5. Extension verification (no agent; "linguistic").* A judge-LLM checks (a) **strict superset**: i_T^{n+1} properly contains i_T^n, is not a synonym (Car/Automobile), and R uniquely identifies the subset (no many-to-one); (b) the substitution in q^{n+1} replaced exactly i_T^n; then (c) **leak check**: an infer-LLM tries to answer the merged task without tools, and the judge drops tasks whose answer it gets. The appendix shows the two failure classes this targets: the *pseudo-superset* (a synonymous title gives a hop that needs no tool) and *information leakage* (the merged question states "$34.5 billion" and asks for the exact amount).

*6. Width extension.* q_width = q_1 + q_2 → a_1 + a_2, where + is an LLM merge and rephrase of two independent subtasks. No interaction between subtasks is required.

*7. Prompt self-evolution.* DSPy-style bootstrap few-shot: for each of four prompts (conclusion extraction, superset search, relation articulation, substitution) append randomly sampled **validated** tasks from the pipeline's own output as exemplars (20 for atomic, 10 for depth), try several exemplar sets, keep the set with the best pass rate (atomic) or the most hops (depth).

*8. Downstream use.* SFT on 3,202 synthesised multi-hop trajectories (search-tool content masked, Chain-of-Action style), then Search-R1 PPO (2018 Wikipedia, E5 retriever, 3 passages, NQ+HotpotQA train, policy LR 1e-6, value LR 1e-5, 500 steps, GAE λ=γ=1, KL 0.001, clip 0.2, action budget 4).

Pseudo-code:
```
for doc in corpus:
    C = tool(i_T(doc)); for (a,R) in LLM_atomic_conclusions(C):
        q = LLM_verbalise(i_T, R)
        if judge(agent_tools(q, steps<=3), a) > judge(llm_no_tools(q), a) > -1 and judge(agent..)>0: keep q
for q^n in kept, up to n_k=6 attempts:
    C' = search_agent(supersets of i_T^n); (i', R') = LLM_min_unique_superset(C', i_T^n)
    q̂ = LLM_verbalise(i', R') -> i_T^n
    q^{n+1} = LLM_substitute(q^n, i_T^n -> q̂)
    if judge_strict_superset(i', i_T^n, R') and judge_substitution_ok and not judge(llm_no_tools(q^{n+1}), a): keep
width: q_w = LLM_merge(q1, q2) -> (a1, a2)
```

**Verification / grading.** Ground truth is the extracted fact from a real document, so the answer is grounded. Grading is a 0/1/2 LLM judge for semantic match. Weak or hackable points:
- After the atomic stage, no tool agent ever re-solves the composed task. Superset uniqueness, substitution correctness and answer uniqueness are judged by an LLM, not checked against the web. A wrong or ambiguous hop gives an unsolvable or multi-answer task that still "verifies".
- The leak check uses one infer-LLM. A stronger student may know the answer anyway (the hop entities are proper nouns, but news gets into later pretraining).
- Width tasks are concatenations; a solver can split them trivially. They add length, not interaction.
- The 1-point "extra details" score rewards hedged, verbose answers if used as a reward.

**Scale & cost.** About 36,000 tasks claimed in the abstract and conclusion, but about 32k in the contributions list (inconsistent). 22,053 atomic tasks (web-based 77.78%, image 9.80%, PDF 12.41%). Hop distribution: PDF 1-hop 2,737 (33.62%), 2-hop 1,820, 3-hop 1,514, 5–7 hops 11.80%; image 1–3 hops 4,342/5,397 (80.45%), 5–7 hops 467; HTML 1-hop 17,154 (74.84%), 5–7 hops 1,089 (4.75%). Yields: atomic pass rate 54.9% → 68.1% with prompt optimisation (29.1 s → 23.5 s per document); depth extension pass rate within 6 attempts 41.0% → 51.2% (31.5 s → 30.2 s per task). $ cost, token counts and generator model names: not reported (GPT-4.1 is named only in the direct-prompting ablation).

**Key results.**
- Difficulty spread (Smolagents on a 1,000-task sample, accuracy): PDF 54.4, HTML 50.7, image 22.1, avg 42.4, against GAIA L1/L2/L3 = 54.71/43.02/26.92 for the same agent. Image tasks are the hardest, near GAIA L3. Failure rates rise web → PDF → images in PDFs.
- Construction vs direct prompting (Table 4, GPT-4.1 asked to write a one-tool question for an answer): pass rate 18.5% vs 43.0%; time 119.7 s vs 86.7 s; tools per query 2.8 vs 2.1; tool-count variance 1.2 vs 0.4. Conditioning on (i_T, R) makes atomicity and difficulty controllable. (The 43.0% here differs from the 54.9% atomic pass rate of Table 2; the settings are not reconciled.)
- SFT on 3,202 trajectories, Qwen2.5-3B, EM: Base workflow avg 0.034 → 0.174 with SFT; Search-R1 0.140 → 0.245 with SFT first (Bamboogle 0.088 → 0.280, Musique 0.049 → 0.111). Instruct: 0.113 → 0.173; Search-R1 0.230 → 0.236 (no real gain). So the trajectories help a base model as an RL warm start, and barely help an instruct model.
- No ablation of depth vs width, of the leak filter, or of the superset check; no measurement of how many kept tasks are actually wrong.

**Failure modes & limits.** Authors: only common tools (browse, PDF, image); user-specific tools are future work. My critique:
- **Verification beyond the atomic stage is not executed.** That is their stated efficiency gain ("verify only the increment"), but it means compositions are never shown solvable. The plan's rule 3 (construction before sampling) needs the composed solution to *run*.
- Depth by superset climbing tends to produce artificial riddle chains ("the recurring AP sports feature that…") whose difficulty is lookup and disambiguation, not reasoning. That is BrowseComp-style obfuscation, useful for search skills but a risk of intent drift if the intended skill is something else.
- Hop count is the generator's label, not a measured behaviour; the "ideal search engine" assumption ignores search steps, so real horizon is uncontrolled.
- Self-evolving prompts select exemplars by pass rate on the same LLM judges they are graded by, which is circular: it raises yield under the judge, not validity.
- No dedup or diversity control; the corpus mix (75% web) decides coverage.
- 36k vs 32k inconsistency and missing cost data weaken the scale claim.

**Tips & tricks.**
- Keep a task only if the tool agent beats the **same-strength model without tools** on it (AgentScore > LLMScore, agent > 0). This is the cheapest "needs the World" probe there is.
- Build atomic tasks from extracted atomic facts that carry a definite identifier (number, date, unique name) and an explicit time range; vague facts make ungradable tasks.
- Raise difficulty by **substituting an identifier with a uniquely-identifying description** that needs one more lookup; check (i) strict containment, not synonymy, (ii) uniqueness, (iii) answer concealment.
- After every composition, re-run the no-tool solver on the composed text; drop it if it gets the answer.
- Verify incrementally: only the new hop needs checking if the earlier chain is fixed (cheap), but re-solve the whole chain on a sample.
- Bootstrap few-shot exemplars for the generator from the pipeline's own validated outputs to raise yield (54.9% → 68.1% atomic).
- Cap extension attempts (they use 6) and log pass rate per attempt as a stage yield.

**Fit to Crucible Part 2.**
- **G9 No shortcut (F6), P0, already cited.** Implement TaskCraft's rule literally for read-only answer tasks: solver with World tools vs the same model with instruction only; reject when the instruction-only score ≥ the tool score, or when the instruction-only score is non-zero for exact answers. For write tasks the analogue is "instruction only gives no state change that passes" (G4 null agent already covers part). Run it after **every** knob change, not just at creation (their leak check after each merge).
- **F3/F4 hardness from structure (§3.6), P1.** Depth extension = our `linkage` knob; width = our `count` knob. In a World this becomes executable rather than linguistic:
  - *Depth (World version):* replace a direct key in the instruction (vendor id, invoice number) with a description that requires a lookup ("the vendor whose last invoice was disputed in March"). The **strict-superset/uniqueness check becomes a query**: the description must select exactly one row in the start state (`StateAdapter.query` count = 1). The oracle program gains the lookup step, so F4 still executes end to end and G3 holds.
  - *Width (World version):* merge two verified subtasks. Require interaction, or label it honestly as `count` only. The F10 single-subtask probe (joint pass ≈ product) will flag width tasks as non-interacting, which is correct.
  - Record the measured extra hop in the F7 `BehaviourSignature` (extra tables read, extra calls) and reject "harder" variants whose signature does not change (§3.6 rule).
- **Incremental verification (F6 cost lever, §3.10).** Adopt the idea that a child task re-uses the parent's verified gates and only the delta is new, but keep an executed full re-solve of the oracle (cheap in a World; it is the step TaskCraft skipped).
- **Leak rules (G11).** Their pseudo-superset and answer-in-question cases are two concrete `LeakRule` classes for answer tasks: "instruction contains the gold answer or a near-paraphrase" and "description is a synonym of the hidden key". Both can be checked by string/entity match against the oracle's gold answer before any model call.
- **Generator loop (§3.8).** Bootstrap few-shot from accepted tasks is a cheap generator-loop lever, but select exemplars on executed gate yield **and** held-out anchor agreement, not on LLM-judge pass rate (avoid their circularity). Feed the K nearest accepted tasks as "do not repeat" in the same slot so exemplars do not cause herding.
- **Avoid:** LLM-only checks for uniqueness or solvability where the World can execute them; counting hops by label; riddle-style obfuscation for families whose intended skill is not search (the intent card should name obfuscation as a knob, `obfuscation` already exists in the DomainSpec).
- **Interface implied.** `Knob{id: linkage, op: substitute_key_with_unique_description}` and `Knob{id: count, op: merge_independent_subtask}`, each with a precondition query (uniqueness) and a post-check (G9 re-run, signature delta). Priority: G9 rule P0; depth/width operators P1.


## 2511.19304 — AutoEnv: Automated Environments for Measuring Cross-Environment Agent Learning (Zhang et al., 2025; arXiv v2 Dec 2025, HKUST-GZ / DeepWisdom)
Read: FULL (main text; App. A three-layer base classes; App. B generation algorithm, feature tables, file layout, a full DSL example, environment code and validator, human theme review; App. C metrics, per-environment results, the >100% reward case, agent code and prompt; App. D learning algorithm, prompts, learned artefacts and trajectories; App. E SFT setup, skin-inversion ablation, multimodal skins).  Relevance to Part 2: **high** for F1 (World/App synthesis) structure and its gate ladder: theme → spec → DSL → code by a coding agent → self-repair → execution test → level generator + validator → differential-model reliability check, at $4.12 per environment and 65% end-to-end yield. The BaseEnv/ObsEnv/SkinEnv split is a clean way to make observability and surface form into **knobs that do not touch the rules**. Medium as evidence for task quality: the environments are toy text games, the gates are weak (no oracle, a heuristic LLM-written max-reward, no hack test), and its own appendix shows reward exploits and a mismatched instruction file.

**Problem.** Cross-environment agent learning is unmeasured because there is no extensible pool of environments with different rules over transitions, observations and rewards. AutoEnv generates heterogeneous environments cheaply, and then asks whether fixed agent-learning methods (prompt/code optimisation) scale across many environments.

**Method / algorithm.**
*Formalism.* E = (S, A, T, R, Ω, τ), implemented as three code layers:
- **BaseEnv** — state, actions, transition, reward, termination; `reset(mode=load|generate, world_id, seed)`, `_generate_world(seed)` writes a level file, `transition(action)`, `reward(action) → (value, events, info)`.
- **ObsEnv** — an `ObservationPolicy` maps true state to a semantic observation (full, partial, radius).
- **SkinEnv** — `render_skin(ω)` maps the semantic observation to agent-facing text or images. `step()` = transition → reward → observe → render → done.
The same BaseEnv can carry different observation policies and skins, so rule changes, observability changes and surface changes are separable.

*Generation (Algorithm 1), generator model Claude-4-Sonnet:*
1. Theme θ (one paragraph, LLM-written; 25 of 100 were then human-reviewed) → **DesignAuthoring**: a detailed natural-language design (goals, rules, state variables, reward conditions).
2. **DSLSynthesis** → a YAML config with sections `meta`, `state_template`, `observation{policy, expose[...]}`, `reward{events: trigger→value}`, `transition{actions[name, params]}`, `skin{type, template}`, `termination{max_steps, conditions}`, `generator{mode: procedural, pipeline of named steps, randomization{seed_based, parameter ranges}}`, `world_loading`. **ValidateDSL** checks schema and interface alignment.
3. **CodeSynthesis** by coding agents (SWE-agent style): the three layer classes, a **level generator**, a **validator** (with a `max_reward` estimator per level), and agent-facing docs (`agent_instruction.txt`, `action_space.txt`).
4. **Self-repair**, up to 40 steps: BasicCodeTest = import, reset, step, call generator and validator; on failure the reflect model edits code from the error messages. Still failing → reject.
5. **Execution test**: a short ReAct rollout; crash or hang → reject.
6. **Level generation**: generate 15 candidate levels; keep those the validator accepts (checks such as action constraints, target reachability, impossible patterns, reward-structure fields, valid cell values); compute each level's max-reward bound. Zero valid levels → reject. Levels 11+ are kept as a validation split (`val_levels/`).
7. **Reliability (differential model testing)**: ReAct agents on GPT-4o-mini and DeepSeek-V3.1; if the **weaker model consistently beats the stronger**, the reward is judged near-random and the environment is rejected.
8. Package: code, levels, per-level max rewards, config.

*Dataset curation.* 100 themes → 65 pass all gates → **36 hand-selected** for coverage of reward type (binary/accumulative), observability (full/partial) and semantics (aligned/inverse: the text says poison heals, water hurts), and to drop too-easy or too-hard ones → AutoEnv-36 with 358 validated levels.

*Learning formalism (Algorithm 2).* Candidate = a version of the agent with components (prompt, agent code, tools, model) plus trajectories and metrics. Loop for T = 10 iterations: Selection F_s (Best, or Pareto over reward and cost) → Optimisation F_o (Claude-4-Sonnet reads trajectories; "dynamics" signal = reverse-engineer rules, rewards, failure causes, strategies, uncertainties, confidence; "instruction" signal = diagnosis, principles, when-then guidelines, guardrails, measurement) applied to a target component (prompt or agent code) → Evaluation F_e (3 runs per sample on fixed test levels; training levels drawn fresh from each environment's generator). 2 selections × 2 signals × 2 components = 8 methods. "Learning upper bound" = best method chosen per environment.

**Verification / grading.** Reward is the environment's own coded reward; the metric is reward ÷ validator-estimated max reward. Weak points, several shown in the paper itself:
- The **max-reward bound is a heuristic** written by the same coding agent. GPT-5 scores **138.84%** on 1_InterDimension, because profit reward accrues every positive increment without subtracting later losses and stability bonuses arrive almost every step. The authors call this the bound's conservativeness, "not a bug". I read it as a reward exploit: alternating gains and losses earn more than the final wealth.
- **The level validator checks structure, not solvability by execution** (the Connect-Four validator checks dimensions, disk counts, required keys and remaining steps). No oracle or planner proves a level is winnable; 24_MemoryMatching scores 0% for six of seven models (o3 10%), and 35_BattlefieldTactics 0–16.67%.
- **The reliability check is one-sided**: it only rejects environments where the weaker model consistently wins. It cannot detect a reward that both models hack, a trivially solvable level, or a broken one where both score 0.
- **Spec/document drift**: the published `agent_instruction` for 22_ColumnStrategy (a Connect-Four DSL) describes a 10×10 deceptive grid-navigation task. Environment names also disagree between Table A6 and Table A7 (24 MemoryMatching vs ChaosSlidePuzzle; 26 TileArrangement vs MismatchedMemoryGame). Nothing in the gate ladder checks that the instruction matches the code.
- Inverse-semantic environments score higher on average (40.69 vs 36.15). Inverting only the skin of aligned environments cuts scores by 68.8%, so the inverse environments were **generated easier**. The generator's difficulty is entangled with the semantic axis it was asked to vary.

**Scale & cost.** 100 themes (75 automated, 25 with human review). Stage pass rates: execution 90.0% (automated 88.0 / supervised 96.0), level generation 96.7% (97.0 / 95.8), reliability 74.7% (70.3 / 87.0); **overall 65.0% (60.0 / 80.0)**. Cost **$4.12 per environment** overall ($4.33 automated, $3.48 supervised; human time not costed). AutoEnv-36: average 6.10 actions, 471 lines of code; 18 binary / 18 accumulative; 15 full / 21 partial; 28 aligned / 8 inverse. Learning runs cost about $0.43–0.76 per environment per method (DeepSeek-V3.1 executor). SFT baseline: 800 action-level examples per environment, Qwen2.5-7B, 8 epochs, 4 GPUs.

**Key results.**
- Benchmark spread (normalised reward, 3 runs): o3 48.73, GPT-5 46.80, Claude-4-Sonnet 40.67 (40.86 in App. table), Gemini-2.5-Flash 39.41, DeepSeek-V3.1 34.01, Kimi-K2 31.49, GPT-4o-mini 11.96. Binary > accumulative (40.06 vs 32.25); full > partial observability (39.81 vs 33.54).
- Human review of themes raises yield 60% → 80%, mainly via reliability (70.3 → 87.0), by making state, action, reward and difficulty rules explicit (worked example App. B.7). This is the single largest lever in the paper.
- Learning, 6-env subset, Qwen2.5-7B: base 17.53; best single method SFT 25.09; upper bound 28.86. DeepSeek-V3.1: base 34.99; best single (Pareto + Instruction + Prompt) 42.99; upper bound with all 8 methods 46.34 (4 methods already give most of it; 8 add 1.23).
- All 36 envs, Gemini-2.5-Flash, 4 methods: base 39.41 → best single 42.40 (+3.0) vs +7.2 on the 6-env subset; per-env upper bound 47.75. Gains concentrate on few environments; some methods hurt (Dynamics + Agent under Pareto drops below baseline on 19-AS).
- Skin-inversion ablation (Gemini-2.5-Flash): numeric-label inversion 26.67% (understood conceptually), symbol inversion 0% (inferred by interaction but explored badly), pixel-value inversion 8.62% (never adapts).

**Failure modes & limits.** Authors: imperfect reliability verification; small, text-only set; restricted learning-method space. My critique:
- The headline "65% verified" means "runs, produces structurally valid levels, and the weak model does not consistently beat the strong one". That is far weaker than our G1–G14. **Verified ≠ correct**, as the >100% reward and the wrong instruction file show.
- 36 of 65 were chosen by hand for coverage and difficulty, so the dataset statistics describe a curated set, not the generator's natural output. The generator's own diversity is not measured.
- No dedup, no novelty check between themes; themes are LLM-written, so coverage depends on the theme prompt.
- No RL; "learning" here is prompt and code optimisation of a frozen model plus one SFT baseline. Results are 3 runs on about 10 levels per environment, with no confidence intervals.
- Toy rule-worlds have little to say about enterprise realism; the transferable part is the pipeline shape and the layer split.

**Tips & tricks.**
- Make the environment spec a **typed DSL file** (state template, observation `expose` list, reward events, actions, termination, generator pipeline with seeded parameter ranges) and validate it before any code is written.
- Split rules / observation policy / rendering into separate layers so observability and surface form become knobs that cannot change the reward.
- Give the coding agent a hard budget (40 repair steps) with a fixed smoke test (import, reset, step, generator, validator).
- Ship a **level generator plus validator** with every environment, and hold out generated levels as a validation split.
- **Differential model testing** is a cheap sanity check: a weaker model that consistently beats a stronger one signals a random or inverted reward.
- Human or strong-model review of the *spec* (not the code) is the cheapest yield lever: 60% → 80% here.
- Treat any score above the "max reward" as a reward-hack alarm, not as conservativeness.
- Test semantic robustness by inverting only the skin with the rules fixed; it separates generator difficulty from the property being varied.

**Fit to Crucible Part 2.**
- **F0 → F1 spec pipeline, P1.** AutoEnv's theme → design → DSL → code chain maps onto `DomainSpec` → `AppBundle`. Adopt: (a) an explicit intermediate design document reviewed before code (their 60 → 80% lever; our F0 already has a lead review gate, extend it to App specs); (b) a schema-validated spec file per App (our `AppSpec` with `schema_ddl`, `policies_as_code`, `column_policy`) checked before code synthesis; (c) a bounded self-repair loop with a fixed smoke test, logging `repair_iterations` and `abandoned` as our F1 yield metrics already name.
- **ObsEnv/SkinEnv as knobs (§3.6), P1.** Implement `observability` and `obfuscation` knobs as World-side view layers (column exposure, redaction, renaming), never as edits to state or verifier. Then `simplify(knob="observability")` is safe by construction and the F7 signature should be identical across skins (same tools, same rows written). Use the **skin-inversion test** as an intent probe for families that claim "reads the World, not the wording".
- **Gate ladder lessons for F1/F6.** AutoEnv shows what is missing when gates stop at "it runs":
  - Add an **instruction–code consistency** check (the 22_ColumnStrategy mismatch would pass all of their gates): a different-family reviewer reads the agent-facing docs against tool schemas, and G13 interaction completeness calls every documented action.
  - Replace the heuristic max-reward bound with **solution by construction** (F4 oracle) for every level; G3 must execute.
  - Add a **reward-exceeds-oracle alarm**: any rollout scoring above the oracle's reward, or above 1 on a normalised metric, triggers G12 hackability review. Their 138.84% is the canonical example.
  - Keep **differential model testing** as a cheap F1 pre-gate (weak > strong consistently → reject), but never as the reliability gate.
- **F2 levels.** Their "generator + validator + seeded randomization ranges" is the `SeedBundle` pattern; require determinism (G2: seed twice, same fingerprint), which AutoEnv does not check.
- **Diversity (§3.5).** Their reward/observability/semantics axes are a hand-made 2×2×2 grid; the inverse-semantics confound is a warning that **generator difficulty correlates with the axis requested**. Our cell membership must be measured (F7) and difficulty measured per cell (F9), never assumed balanced from the labels.
- **Generator loop (§3.8).** Their finding that no fixed learning method scales across heterogeneous environments supports per-family settings (knobs, ladders) rather than one global recipe; P2.
- **Cost model inputs (§3.10).** $4.12 per environment and 65% yield are already cited. Note the yield excludes the hand-selection step (36 of 65) and human review time; for our formula, treat AutoEnv's "verified" as an F1 pass rate before G3/G5/G12, not as a kept-task rate.
- **Avoid:** structural validators in place of executed oracles; one-sided reliability checks; calling exploits "conservative bounds"; publishing curated subsets as generator statistics.


## 2510.01051 — GEM: A Gym for Agentic LLMs (Liu, Sims, Duan et al., 2025; arXiv v2 Mar 2026, Sea AI Lab with OpenRLHF / ROLL / RL2 / NUS / Stanford / Oxford)
Read: FULL (main text, App. A registration code, App. B γ case studies, App. C ReBN algorithm, App. D reward-shaping ablation, vision-language and multi-agent τ-bench study, App. E related work, App. F hyperparameters).  Relevance to Part 2: **low-medium.** It is an environment *interface* and RL-baseline paper, not task or environment synthesis: no generator, no verifier testing, no difficulty control. Three findings matter to Part 2: (1) the simulated user's strength moves agent success as much as the agent's own strength does, (2) small grader differences change reported scores a lot, (3) the discount γ and the reward sign shape tool-use behaviour (speed, overuse). It is also the lineage our Part 1 trainer bridges sit in.

**Problem.** LLM RL is mostly single-turn; multi-turn agent RL lacks a Gym-like standard. Environments are coded into each trainer's source, so "same-name" environments differ between codebases and results do not compare. GRPO-style trajectory-level advantages also force γ = 1 and one reward per episode.

**Method / algorithm.**
*Interface.* Gym API: `gem.make(id)`, `reset() → (obs, info)`, `step(action) → (obs, reward, terminated, truncated, info)`. Environment = task + optional tools. Seven task categories: math, math with image, code, game (from TextArena), QA, ReasoningGym (100+ single-turn generators), terminal (Docker). Tools as wrappers: Python execution, search, MCP. Adding a dataset is a `register(id, EnvClass, dataset_name, question_key, answer_key)` call; a new task subclasses `Env` and writes `reset` (initial-state distribution) and `step` (transition and reward). Observation wrappers choose what history the agent sees (last output, all outputs, outputs + actions, or a summary). Vectorised asynchronous execution with **autoreset**; the `terminated` flag stops value bootstrapping across episodes. Multi-agent API: dict-keyed `step`/`reset`, an `AgentSelector` for sequential or parallel turn order. Single-file integrations with Oat, Verl, OpenRLHF, ROLL and RL2.

*Baseline algorithm — REINFORCE with Return Batch Normalization (ReBN).* Treat each response (one turn) as an action. For every transition compute the discounted return G_t = Σ_{k≥t} γ^{k−t} r_k; normalise all G_t in the batch: Ã = (G − mean(G))/std(G); update with Σ Ã ∇ log π(y_t | s_t), optionally with 2 proximal epochs. Unlike GRPO it allows per-turn rewards and γ < 1 without per-state resampling, and unlike PPO it needs no critic. Compared against multi-turn GRPO (trajectory reward normalised within a group of M trajectories per start state, length bias removed), vanilla REINFORCE and turn-level PPO with GAE.

*Settings (App. F).* 8×A100, about a day per experiment; max 4,096 tokens per turn; temperature 1.0 train / 0.0 eval; AdamW (0.9, 0.95), LR 1e-6 constant, grad clip 1.0; KL 0; clip 0.2; γ = 0.9 for game and QA, 1.0 otherwise; GAE λ 0.95; 500 steps. Base models Qwen3 {1.7B, 4B, 8B}-Base.

**Verification / grading.** Rewards come from each environment: math via HuggingFace `math_verify`, QA exact match, games by rule, terminal by test cases. The paper itself warns that **minor grader differences across codebases cause large score variation**, so its math numbers are only comparable inside GEM. Nothing in GEM tests a verifier; correctness is inherited from the source datasets.

**Scale & cost.** 24 environments with baselines; Terminal-Bench subset of 57 tasks; MCPMark Postgres subset of 20; τ-bench retail 115 tasks. Compute: 8×A100 for about a day per run. RL2 async rollout gives about 2× wall-clock speed on Sudoku.

**Key results.**
- ReBN beats vanilla REINFORCE in every shown environment and matches or beats PPO and GRPO in all eight benchmarked ones (curves only; no tables or seeds). GRPO is fine on single-turn ReasoningGym tasks and weak on dense-reward multi-turn games (GuessTheNumber, Sudoku). PPO is best on long-horizon Sudoku but poor on Minesweeper (critic quality).
- **γ matters:** on GuessTheNumber (1–50), γ < 1 drives the policy to about log₂50 ≈ 5.6 turns, i.e. binary search; γ = 0.999 exhausts its 10 trials. With GRPO-style γ = 1 people instead tune a turn limit.
- **Reward sign matters:** with 0/1 rewards, vanilla REINFORCE never learns to call tools; fixed ±1 rewards give *no* tool use on math:Orz57K but **tool overuse** on qa:HotpotQA; ReBN on 0/1 rewards works in both.
- Tools help after RL: Qwen3-4B-Base math average 35.3 (base, no tool) → 41.4 (RL, no tool) → 49.8 (RL, Python tool); AIME24 10.0 → 30.0. QA (search tool): 10.2 → 22.7 (RL no tool) → 43.7 (RL + tool, HotpotQA only) → 45.5 (HotpotQA + NQ mixed). Mixing two training environments helps slightly.
- Training on easy Sudoku improves some ReasoningGym evaluations (figure only).
- **User simulator strength (τ-bench retail, 115 tasks, 3×3 grid of user and assistant models: Gemini-2.0-Flash-Lite, GPT-4o-mini, GPT-4o):** a stronger simulated user raises success for every assistant; GPT-4o as assistant gains about 20 pp from weak to strong user, reaching 61.7%; a strong user with a weak assistant (44.3%) beats a weak user with a strong assistant (41.7%).
- Evaluation use: GPT-5 is best on both MCPMark-Postgres (20 tasks) and the Terminal-Bench subset (57), with the fewest turns (figure only).

**Failure modes & limits.** Mostly curves without seeds or confidence intervals; "consistent improvement" claims rest on single runs. Algorithm comparisons use small Qwen3 base models and short runs (500 steps). The benchmark environments are standard datasets and toy games, so none of this tests long-horizon enterprise tasks. The τ-bench user study evaluates, it does not train. My critique: ReBN normalises across all transitions in the batch, mixing turns from different tasks and depths; with heterogeneous task difficulty this can let easy tasks dominate the baseline. The paper does not test that.

**Tips & tricks.**
- Keep environments decoupled from the trainer behind a Gym-style `reset/step` with an explicit `terminated` vs `truncated` split.
- Use γ < 1 (they use 0.9) when speed is part of the intended skill, instead of a hard turn cap.
- Avoid raw 0/1 returns with plain REINFORCE; normalise returns per batch or the policy may never explore tool use. Fixed negative rewards can cause tool overuse.
- Pin and version the grader; the same benchmark under a different grader is a different benchmark.
- In multi-agent tasks, report results per (user model, agent model) pair; the user model is part of the task.

**Fit to Crucible Part 2.**
- **F9 difficulty and F10 intent, P1: the simulated user is a task parameter.** GEM's 3×3 grid shows that a weak user simulator can lower success as much as a weaker agent. For multi-turn tasks with a person in the loop (`TaskContract.user: ActorSpec`), `DifficultyRecord` must key p̂ on the user-simulator model and profile as well as the student, and the user model must be frozen per pool version. Add an F10 probe: rerun a sample with a stronger user simulator; if the pass rate jumps, the task's hardness is in the simulator ("World side" fault), not the student's skill. That belongs in `fault_sides`.
- **G10 / grader versioning.** Their grader-sensitivity warning supports versioned answer checks (`verify/answers.py` under `RULES_VERSION`) and recomputing stored grades when a checker changes, as §3.8 already plans.
- **Knobs and reward shaping (§3.6), P2.** γ is a training-side knob, not a task knob, but a task whose intended skill includes efficiency should say so in its `IntentCard`, and the trainer should run it with γ < 1 or a timing/budget check rather than a hidden turn cap. Their tool-overuse result under ±1 rewards is a reminder that "budget 2×" (F10) needs a matching "budget ½" check for tasks where calls are cheap.
- **Interface (§3.0 rule 7).** The Gym API, autoreset and observation wrappers are compatible with our agentenv/World sandboxes; no new adoption is needed beyond what Part 1 already bridges. A `DomainSpec` family could expose an observation-wrapper choice (full history vs summarised) as a `horizon`-related knob for long tasks.
- **Avoid:** using GEM's curated datasets as Part 2 sources (they are public and contaminated), and reading its algorithm curves as evidence for our 35B setting.


## 2505.24760 — Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards (Stojanovski, Stanley, Sharratt, Jones et al., 2025; arXiv v2 Oct 2025, Open-Thought / Scale AI / UCL)
Read: FULL (main text sections 1–8; App. A: category table of all generators with curriculum flags, worked task examples with metadata, the full easy/hard config list for zero-shot evaluation, training setup and a complete verl config). Per-task zero-shot heatmaps are figures only, so per-task numbers were not readable. The NeurIPS checklist and references were skimmed.  Relevance to Part 2: **medium.** The tasks are single-turn puzzles, far from World tasks, but RG is the cleanest worked example of **difficulty as a parameter vector of a generator** with solution-by-construction and a simple adaptive curriculum. Its parameter taxonomy (difficulty / structural / stylistic) maps directly onto our `Knob` schema, and its own easy/hard configs show why knob monotonicity must be measured, not declared.

**Problem.** RLVR needs verifiable data at scale; fixed datasets saturate, get memorised and contain errors. RG replaces datasets with procedural generators whose answers are verifiable by algorithm and whose difficulty is tunable.

**Method / algorithm.**
*Design principles:* (P1) every task is checked by algorithm, no human judgement; (P2) large solution spaces, to favour general strategies and resist hacking; (P3) parametric difficulty control for curricula.
*Generators.* Over 100, in categories (counts from Table 6): algebra 6, algorithms 32, arithmetic 11, cognition incl. ARC 14, code 2, games 18, geometry 4, graphs 6, induction 2, logic 7. All but a few support a curriculum.
*Each generator* is a seeded function from a config to (question, answer, metadata). Metadata carries the source dataset, index, the full latent instance (matrix, puzzle, statements), the **solution**, and the difficulty parameters used. Answers come from the generating process (solution-first) or an exact solver (e.g. orthocenter as an exact fraction, Sudoku solution grid, BF program compiled from a known C program).
*Three parameter kinds:*
- **Difficulty** parameters: node counts, polynomial degree, word length, number ranges.
- **Structural** parameters: dimensionality, constraint types, proof depth (e.g. knights_knaves `depth_constraint`, `width_constraint`, `n_people`).
- **Stylistic** parameters: names, number formats, framing (e.g. knights/knaves renamed sages/fools); these vary surface without changing difficulty.
Some generators include a controlled share of unsolvable instances (`p_solvable: 0.9` in the training config; shortest_path asks for "infeasible" when there is no path).
*Easy vs hard configs* (App. A) exist per generator for evaluation.
*Curriculum (verl integration):* start at the lowest level of one attribute; every `update_steps` (30) look at the last k = 20 steps; raise the level when accuracy > 0.70; a `failure_threshold` of 0.10 is in the config (I read it as the lower trigger, but the paper's text describes only raising).
*Reward:* accuracy (1.0) plus a format reward scaled 0.2; the config also has a length reward scaled 0.2. Evaluation tables report accuracy only.
*Training setup:* verl, GRPO, Qwen2.5-3B-Instruct, 4×A6000, `dataset_size` 20,000 procedurally drawn per run, batch 32, n = 8 rollouts, LR 1e-6, KL 0.001 (low-var KL), entropy 0.001, clip 0.2, max response 2,048, 500 steps typical (800 for RG-Math), 1 epoch. Total for all reported experiments: about 1,500 A6000-hours.

**Verification / grading.** Exact or canonicalised answer checks per generator (string equality, numeric tolerance, rounding rules stated in the prompt, e.g. "round to 3 decimals"). Because the latent instance is in metadata, a verifier can re-check alternative valid answers when the task has many (Sudoku, shortest path may have ties; the paper does not say which verifiers accept alternatives). Hack risks: format rewards are learned first (the reward spike at the start of training is partly format); single-answer string checks on tasks with several valid answers would be false negatives; the paper does not measure either.

**Scale & cost.** Unlimited instances; >100 generators; about 1,500 A6000-hours for all experiments. Evaluation sets: 50 problems per held-out task or level, 3 runs each.

**Key results.**
- Zero-shot (hard configs, average): o3-mini 63.5, DeepSeek-R1 59.5, Grok 3 Mini 55.1, Llama 4 Maverick 41.5, Claude 3.5 Sonnet 40.3, Gemma 3 27B 20.3. Visual-spatial-in-text categories (cognition, games, ARC) stay below 50% even for the best.
- **Difficulty cliff** easy → hard: o3-mini code −71.9, graphs −33.8, geometry −33.1, algorithms −25.6 pp; DeepSeek-R1 code −61.8, graphs −29.6, geometry −11.8, algorithms −27.9 pp.
- Intra-domain transfer (Qwen2.5-3B, train on a category composite, test on a held-out task of the same category, acc@3): algebra 5.0 → 16.7; algorithmic 52.3 → 59.7; arithmetic 89.7 → 96.0; cognition 40.3 → 42.3; games 0.0 → 3.3.
- Cross-domain transfer: RG-Algorithmic lifts algebra 23.83 → 52.89 and geometry 0.83 → 23.17; RG-Logic lifts cognition 11.62 → 24.94 and graphs 19.81 → 28.86; RG-Games lifts algebra to 45.61. Some transfers are negative (ARC −2.3 from algebra, games −0.8 from logic).
- External: RG-Math (800 steps) MATH 48.5 → 58.2, BBH 8.68 → 16.34, GSM8K 76.2 → 76.7; MMLU-Pro math 54.63 → 60.25 (RG-Algorithmic −0.74 there).
- **Curriculum vs uniform-over-levels** (one epoch, 50 hold-out per level): Spell Backwards word length 4: 70.67 vs 30.00 (base 12.00); length 6: 30.00 vs 10.67. Mini Sudoku 8–10 blanks: 20.00 vs 6.67. Count Primes range 100–500: 30.67 vs 4.00, even though **the curriculum never left level 1**, so the gain there comes from training in band rather than from progression.

**Failure modes & limits.** Authors: hard to cover knowledge-heavy or creative domains; verification misses quality aspects people value; single-turn text only; i.i.d. uniform sampling, no continual-learning study. My critique:
- **Declared difficulty is not always difficulty.** In the published easy/hard configs several "hard" settings move a parameter the easy way: mini_sudoku hard has *fewer* blanks (6–10 vs 8–12), puzzle24 hard has a *smaller* value range (≤6 vs ≤10), tsumego hard allows fewer stones, time_intervals hard has smaller ranges. Some may be intended, but nothing in RG checks that a knob is monotone in measured success. The aggregate "difficulty cliff" averages over such knobs.
- Small evaluation sets (50 items × 3 runs) and a 3B model; cross-domain numbers are large relative to these sizes, but no confidence intervals are given, and part of the transfer may be format learning.
- The cross-domain table is sparse (dashes) and the choice of pairs is not explained.
- Contamination resistance is claimed from "no two instances identical", but templates are fixed; a model can learn a template's solver, which is the skill only if the template is the skill.
- No dedup by behaviour: different seeds can give near-identical instances at low difficulty.

**Tips & tricks.**
- Put the full latent instance and the solution in each task's metadata; it lets verifiers accept alternative answers and lets you audit difficulty after the fact.
- Separate difficulty, structural and stylistic parameters; stylistic ones are free diversity that should not move p̂.
- Include a controlled infeasible share (`p_solvable`) with "infeasible" as the correct answer.
- A simple curriculum works: raise a level when accuracy over the last 20 steps exceeds 0.70; keep a lower threshold (0.10) to back off.
- Training on the in-band level beats sampling all levels uniformly, even without progression.
- Keep a format reward small (0.2 of the 1.0 accuracy reward) and report accuracy alone.

**Fit to Crucible Part 2.**
- **F0 DomainSpec knobs, P1.** Adopt RG's taxonomy in the `Knob` schema: add `class: difficulty | structural | stylistic` beside `kind` and `monotone`. Stylistic knobs (names, phrasing, formats, persona voice) feed §3.5 diversity at no difficulty cost; F7 must show identical behaviour signatures across stylistic variants, and F9 should show no p̂ shift. A p̂ shift on a stylistic knob is an intent finding (the task depends on wording).
- **Measured monotonicity (F9), P1.** RG's configs show the failure: declare `monotone: true` only after measurement. Add a cheap check per family: sample a knob ladder (3–5 settings), run the frontier audit sample, and fit a monotone trend; a non-monotone knob is flagged in the `DomainSpec` review. This also feeds the cheap predictor (§3.6).
- **Difficulty controller (§3.6 / F9), P1.** Their curriculum rule (raise at > 0.70 over the last 20, back off at < 0.10) is the in-training analogue of our band; it supports keeping p̂ windows per (task family, student) and moving along a knob when the window leaves the band. The Count Primes result is evidence that **in-band training alone** carries value, which supports sending out-of-band tasks to ladders or mutation rather than training on them.
- **Infeasible tasks (§3.3).** `p_solvable` is the generator-level form of our `infeasible_share` in the DomainSpec. Make it a per-family parameter with its own answer check ("report infeasible") and its own negatives (an agent that always says infeasible must fail the feasible share, G4).
- **Metadata as privileged notes (F3/F4).** RG's metadata (latent instance + solution + parameters) is our `privileged_notes` + `Oracle` + `knob_vector`; keep it in verifier custody only.
- **Not for the pool.** RG tasks are public and single-turn; they are not Part 2 tasks. At most they are a warm-up or a canary family for testing the knob machinery and the controller cheaply (P2).
- **Avoid:** treating generator parameters as difficulty without measurement, and reporting a single "hard" config as a band.


## 2409.15637 — Synatra: Turning Indirect Knowledge into Direct Demonstrations for Digital Agents at Scale (Ou et al., 2024; NeurIPS 2024, CMU / Amazon)
Read: FULL. The arXiv HTML text stops at the start of App. G, so App. G–M (example generations from random pages, baseline prompt, case studies, ClueWeb domain statistics, training settings, broader impacts, limitations) were read from the arXiv PDF (pages 20–30) fetched into the scratchpad.  Relevance to Part 2: **low-medium.** Synatra makes SFT demonstrations, not RL tasks: nothing is executed and there is no verifier. Two ideas transfer: (1) **procedural documents as an intent source** (tutorials there; SOPs, runbooks and help-desk articles for an enterprise World), turned into concrete scenarios with invented but consistent values; (2) **grounding proposals in a sampled real observation** with brainstormed task categories and temperature-sampled domains for coverage. Both belong at F0/F3; the rest of its pipeline is what Part 2 must not do (unexecuted, LLM-imagined states).

**Problem.** Web agents lack large sets of direct (state, action) demonstrations. Human demos are costly (about $0.80 each on Mind2Web) and narrow; exploration needs live environments with real accounts and content. Plenty of *indirect* knowledge exists: tutorials (procedural knowledge), descriptions of what happens after an action (environment knowledge), and unlabelled page snapshots (ungrounded observations).

**Method / algorithm.**
*Representation: trajectory as a Python program.* Prompt holds website, current accessibility tree (AXTree) and objective; the response is a `solve()` body whose comments give sub-task plans (task level) and per-action explanations or chain-of-thought (action level), with actions as API calls (`click`, `type`/`click_and_type`, `hover`, `key_press`, `scroll`, `goto`, tab ops, `stop(answer)`), plus a one-line **action summary** that replaces the CoT in later history to keep context short.

*Source A — tutorials → trajectories with imagined pages.*
1. Filter wikiHow with GPT-3.5-turbo (6-shot title classifier: task done purely through a computer or phone GUI, no physical steps, no CLI) → about 25k articles.
2. **Rewrite** each article into a hypothetical concrete trajectory (assistant LM): invent a concrete, consistent scenario with realistic values (no boilerplate like `email@example.com`), map free-text verbs onto the action API, unroll repeated steps, skip install/login, pick the most common path when the article branches, group calls into sub-tasks, and write the final task command.
3. **Generate the observation** between two randomly chosen consecutive actions: the LM writes an HTML page that reflects all past actions and contains the element needed for the next action, tagged `id="next-action-target-element"` (removed later). Target richness: about 80 elements, at least 20 interactable, DOM depth about 7, at least 3,000 tokens; no CSS/JS needed. The tag gives the grounding of the next action.

*Source B — real pages → imagined tasks and histories.*
1. Sample pages from ClueWeb22 (10B pages). Domain frequency is long-tailed (64.2% of domains appear once; App. J), and random sampling over-represents non-interactive news. They **temperature-sample by top-level domain** with T = 0.6: P′_i = p_i^{1/T} / Σ_k p_k^{1/T}, which up-weights frequent, interactive domains (Amazon, Reddit) while keeping rare ones.
2. Convert to AXTree with WebArena-style node ids; show a segment to the LM.
3. Prompt: analyse the page's domain, **brainstorm 8 task categories**, then write concrete unambiguous tasks for a subset (5 in the prompt) with a prescribed number of past actions each (about 7, 0, 6, 4, 10), the history grouped into sub-tasks, and the next action using an element id that must exist in the given tree.

*Filtering.* (1) Format/completeness: action in the action space; target element valid and present; no "…" abbreviations; no unfilled prompt placeholders. (2) **No-op filter:** an LLM predicts o_{t+1} from (o_t, a_t); if it predicts no change, the sample is dropped as an unresponsive action.

*Training.* CodeLlama-7B (chosen over Llama-2/3, DeepSeek Coder, Mistral by a Mind2Web pilot) fine-tuned on 99,920 samples from 21 domains, context 4,096, 6×A100-80G for about 40 hours, batch 48, LR 4e-5, cosine, warm-up 0.03.

**Verification / grading.** None for the synthetic data: no execution, no checker, no replay. Tutorial-sourced trajectories have human-written procedures but LLM-imagined pages; page-sourced ones have real pages but LLM-imagined procedures and histories. Correctness is assumed. App. G's own examples show the gaps: an "account management" task whose next action types a username into the site **search box**, and histories that start with `go_forward()` or `go_back()` from nowhere. Evaluation uses each benchmark's checker (Mind2Web step accuracy against references; MiniWoB++ and WebArena execution-based).

**Scale & cost.** 25k filtered articles; about 100k demonstrations (99,920 used); **$0.031 per kept sample** ($0.028 generation + $0.003 filtering, with filtered-out cost spread over kept ones), about 3% of a human demonstration ($0.80). The generation model is called the "assistant LM" and is not named in the method; the conclusion refers to GPT-4-class calls.

**Key results.**
- Synatra-CodeLlama-7B: Mind2Web step accuracy 15.85 (base CodeLlama-instruct-7B 6.62; GPT-3.5 12.79; GPT-4 29.09; Lemur-chat-70B 14.28); MiniWoB++ 38.20 (base 23.04; GPT-3.5 39.57); WebArena 6.28 (base 0.00; GPT-3.5 6.16; GPT-4 14.41; AgentLM-70B 3.07).
- Error analysis (Table 2): clicks on non-existent elements 94.70% (base) → 0.70%; invalid actions 58.88 → 4.91; click on non-clickable 100 → 5.77; repeated typing into the same field 6.79% vs 26.85% for GPT-4-turbo.
- **9k synthetic vs 9k human (Mind2Web) demonstrations:** human-only gives +3.96 on MiniWoB++ and 0 on WebArena; synthetic gives +17.81 and +4.56. Human + synthetic still trails synthetic-only (figure). The authors attribute this to coverage: Mind2Web lacks information-seeking tasks and stop conditions.
- Scaling 18k → 50k → 100k raises WebArena success steadily (figure only).
- Program format beats NL format at 30k (figure). Mixed sources beat either source alone at 9k (figure). RAG over raw tutorials (top-3, Llama-3.1-8B) helps WebArena only marginally, while fine-tuning on converted trajectories helps a lot.

**Failure modes & limits.** Authors: source quality varies; synthetic demos may miss real interaction complexity; risk of overfitting to source formats; needs a strong, costly LLM. My critique:
- **No execution anywhere**, so action correctness, page realism and task feasibility are unverified; the no-op filter is an LLM guess about state change, not a state diff.
- Gains are measured mostly as reduced format/grounding errors; absolute WebArena success stays at 6.28%. Much of the value is teaching the action grammar, which an executed environment teaches more reliably.
- The human-vs-synthetic comparison confounds source quality with task-type coverage (Mind2Web has no answer tasks or stop actions), so it shows coverage matters more than it shows synthetic beats human.
- Many ablations are figures without numbers or seeds.
- Temperature sampling with T < 1 *sharpens* toward frequent domains; it fixes "too much news" but reduces tail coverage, so its diversity effect depends on what "frequent" correlates with.

**Tips & tricks.**
- Convert procedural documents into a concrete scenario with invented, mutually consistent values before anything else; generic steps ("enter the keyword") do not make tasks.
- Ask the generator to brainstorm task categories for a given state first, then instantiate several; it spreads tasks over intents of the same page.
- Prescribe history lengths per generated task (0 to 10 prior actions) to spread start states along the horizon.
- Use an action summary instead of the full reasoning in long histories.
- Reject samples whose action does not change the state (here by LLM; in a World, by diff).
- Sample sources by temperature over domains to balance interactive vs passive content.

**Fit to Crucible Part 2.**
- **F0 / F3 task sources (§3.4), P2.** Add "procedural documents" as a source kind beside mined, synthetic, trace and self-play: enterprise SOPs, runbooks, policy manuals and help-centre articles describe real intents and step orders. In Part 2 they should seed **cells and proposals**, never trajectories: a document becomes (intent, step outline, entities needed) → F2 seeds the entities into the World → F4 writes an oracle program over real tools and **executes** it. This is Synatra's rewrite step with execution in place of imagination, and it fits the traces-to-tasks rule "trace → World snapshot → task, never trace → text → task".
- **F3 grounding and diversity (§3.5).** Synatra's Source B is our "condition F3 on a sampled World slice": show the proposer a real start-state view (records, a page, an inbox), ask for N categories then concrete tasks, and vary the prescribed `start_offset` (their 0–10 prior actions) as a knob. Measure cell membership from the executed oracle (F7), not from the brainstormed category.
- **G13 / no-op negatives (F5–F6).** Their no-op filter is a weak version of two gates we have: G4 (null agent fails) and the collateral/state checks. In a World it should be an executed diff: an oracle step whose state diff is empty and whose observation adds nothing is either a read (fine, if an answer depends on it) or dead weight to drop from the oracle.
- **SFT warm-start (Part 1, P2).** Their error table supports a small grammar warm-up for a new tool API before RL; for the 35B student this matters only if a new App's action grammar is unusual.
- **Avoid:** LLM-imagined observations or states as training ground truth; counting "categories brainstormed" as coverage; mixing human and synthetic sources without checking which task types each covers.

