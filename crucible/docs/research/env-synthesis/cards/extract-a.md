# Extract A: mechanisms from card groups 01 and 02

**Sources.** `group-01.md` (29 papers: environment and task synthesis, tool-use data, SWE environments, gyms; 28 FULL, 1 PARTIAL: Kimi K2 2507.20534, post-training read in full, pre-training summarised only) and `group-02.md` (29 papers: synthesis, verifier testing, self-improving loops; 27 FULL, 2 PARTIAL: OfficeTown 2610.23194 and ProgAgent 2610.14774, bodies in full, some appendices by heading only). No ABSTRACT-ONLY cards. Flags from the cards are kept: PARTIAL, [uncertain], "figure only", "n/r" (not reported) and paper inconsistencies (collected in B9).

**Conventions.** Stage and gate codes follow `framework-plan.md` §3 (F0–F12, G1–G14, the four loops of §3.8). "Adopt" lines say where a mechanism goes in Crucible, with a priority (P0 now, P1 with the generic core, P2 later). Numbers keep their setting (model, benchmark, k). Proposed new gate codes G15–G17 come from the group-02 header and the Tev card; they are proposals, not plan text.

---

## B1. Domain spec and World/environment synthesis

### 1.1 Five build orders for an App

All five end in code over a typed state; they differ in what is fixed first.

| Order | Steps | Evidence and weak point |
|---|---|---|
| **Tool-first** (ScaleEnv 2602.06820) | domain keyword → ~50 tool schemas with pre/post-conditions → DB schema reverse-engineered (5–22 tables) → per-column match policy → DB code + integrity tests first → tool code tested by a separate agent → dependency graph | Execution-verification ablation (8B, τ² avg@4): 42.3/30.0/25.2 without vs 50.9/37.5/27.2 with; Airline without it is below base. The debugger may edit DB instances, so expectations are not frozen |
| **Tasks-first** (AWM 2602.10090; InfiniteWeb 2601.04126) | scenarios → 10 tasks each → DB only for what tasks need → sample data → toolset spec → code (~35 tools, ~2,000 lines); InfiniteWeb: tasks → data models → one shared interface → task-centric TDD (≤ 8 iterations, 1.5% unfixed) | AWM accepts a stage at < 10% failing tables; its audit finds 74–83% of envs with ≥ 1 bug. Task-centric correctness leaves off-path functions buggy |
| **Catalogue-first** (AgentScaler 2509.13311; EnvScaler 2601.05808; Agent-World 2604.18292; CompoWorld 2609.33665) | real API/MCP catalogue or task sets → tool graph → communities or themes → schema → code | AgentScaler: >30k APIs → >1,000 Louvain domains, but reproduced τ-bench schemas "highly consistently" (contamination); no env quality gate |
| **Spec-DSL-first** (AutoEnv 2511.19304) | theme → design doc → typed YAML DSL (state template, observation `expose`, reward events, actions, termination, seeded generator) → schema check → code → ≤ 40 repair steps → smoke test → level generator + validator | 100 themes → 65 pass; **human review of the spec raised yield 60% → 80%** (reliability 70.3 → 87.0), the paper's largest lever |
| **Mechanism-first** (Data World 2610.21656; Tev 2610.18868; VHD-Play 2609.27321; ClawLoop 2610.14118; Reasoning Gym 2505.24760) | fix the mechanism (causal DAG, latent world, solver family, seeded generator) → derive state → render observations; the answer is computed | VHD-Play's matched control on Qwen3.6-35B-A3B: 2,200 EnvScaler scenarios gave BFCL 58.56, below base. Tev: 30 s CPU per env; generator authoring cost n/r |

- **Context scoping.** The same 600-task GRPO run on Claude-Code-generated sites gave +0.7/+0.7/−1.3 against clear gains on InfiniteWeb sites, attributed to per-step specs vs one session accumulating the codebase (4.9 vs 12.6 pages per site).
- **Adopt.** F1 recipes are DomainSpec plug-ins (`tool_first | tasks_first | catalogue | dsl | mechanism`); each gets a reviewed design document before code (spec review P0), a schema-validated `AppSpec`, a bounded repair loop with a fixed smoke test, and the meters `repair_iterations` and `abandoned`. Prefer mechanism-first where an answer is computable (finance_ap allocation, helpdesk scheduling) (P1).

### 1.2 Blind tool tests with frozen expectations (F1 gate)

Every pipeline tests tools; none freezes the expected state at write time.
- **ScaleEnv:** outcomes Success / Anticipated Rejection (the declared exception) / Unexpected Failure; a Debug Agent edits the tool *or the DB instance* until tests pass.
- **EnvScaler:** a frontend tester (interface + current state only) issues one positive or negative call per round (wrong types, missing ids, out-of-range, empty, extreme, special characters); a backend checker (code, args, return, state diff) labels Pass/Warning/Fail; N = 100, keep at ≥ 0.85; 191/266 kept. The rounds are $0.77 of the $1.02 per env.
- **EnvFactory (2605.18703):** four tiers (simple, medium, complex, boundary), each case tagged `expected_behavior`; three layers (L1 load, blocking; L2 2–3 cases per tool; L3 save and diff); 92.9% success, 192K + 31K tokens, ~3 min. **Defect:** the reviser may declare a failure a "scenario problem" with no independent check.
- **CompoWorld:** a separate session writes adversarial boundary tests from the spec only; failing tools are quarantined.
- **Agent-World:** keeps a tool at > 50% of its own tests passing.
- **SWE-Factory (2506.10954):** without execution feedback the build rate fell 50.2% → 0.3% (GPT-4.1-mini).
- **Adopt (P0).** The blind tester writes `expected_state_digest` and `expect` before execution; four tiers, three layers, EnvScaler's negative probes, a spec-only adversarial session; 100% pass for kept Apps; no edits to expectations, DB instances or "scenario problem" verdicts without a different-family checker. N is a cost knob.

### 1.3 Tool graphs, interaction completeness, distractor tools (G13)

- **Edges.** ScaleEnv: dataflow, pre/post-conditions, shared tables. Agent-World weights strong 3 / weak 2 / independent 1. APIGen-MT (2504.03601): dataflow *and* policy permission. EnvFactory: bge-m3 output→input similarity (threshold n/r), then an LLM adds missing edges (parameter-less tools would be isolated) and prunes. VitaBench (2509.26490): from pre/post-conditions. EnvCraft (2609.05576): a bi-level tool and environment graph.
- **Completeness.** ScaleEnv grows the reachable set by dependency-closed BFS and *executes* each added tool, patching the env on error. This is a concrete G13.
- **Distractor tools.** ACE-Router (2601.08276; "ToolACE-MCP" in our notes) mutates tools five ways (usage extension, function enhancement, workflow chain, helper, parameter redesign), 627 → 2,005; without mutation MCP-Mark 60 → 44.00. Mutants were never checked for distinctness. FTRL (2508.08791) merges overlapping tools so names do not give away steps. CompoWorld adds 5 distractor services per task.
- **Adopt (P1).** Edge `kind ∈ {dataflow, precondition, shared_table, policy_permitted}` with evidence; G13 = BFS-and-execute; distractor tools are real code that pass F1 tests and carry `mutated_from`, which G14 reads.

### 1.4 The App engineering contract and error messages

- **EnvCraft Table 15:** dict returns with structured errors; no mutable globals; immutable data dir during rollouts; atomic writes; bounded file locks; **timestamps from `base_time` + `action_index`, wall clock banned**; sessions bound by trainer-injected env vars; trainer commands hidden from agent help; deterministic, side-effect-free evaluator; full state retrieval for snapshot/restore.
- **CompoWorld:** Pydantic state rejecting unknown fields; one envelope `{success, data | error{code, message}}`. **SynthAPI (2610.37648):** state touched only through four CRUD primitives, so every change is logged. **EnvFactory:** `load_scenario`/`save_scenario`. **ProgAgent:** paginated returns; correct `isError` (disabling error detection 38.3 → 34.5). **AWM:** per-rollout SQLite copy; prefetch next batch's servers.
- **Error informativeness.** Simia (2511.01824): RL in an o4-mini simulator beat RL in real OfficeBench (64.7/34.5 vs 60.8/28.6), attributed by case study to explanatory errors (one benchmark, no seeds). Environment Tuning (2510.10197) rewrites opaque errors into root-cause hints approved by a judge that rejects exact calls, concrete values, full sequences and single-path prescriptions (≤ 3 rounds). EnvCraft: infrastructure errors generic, domain errors name the entity, field, state or valid alternative.
- **Adopt.** EnvCraft's list and CompoWorld's envelope go into the App contract (P0), mapped onto clock injection, custody and `StateAdapter`. Error informativeness is a knob (`terse | explanatory | hinting`) implemented as a versioned World-side overlay (P1, with a small RL A/B). G13 flags errors that cannot distinguish causes.

### 1.5 Policies as code

- **LOGIGEN (2603.00540):** SQLite BEFORE triggers reject with a structured POLICY_VIOLATION `{code, message, violated_rule, hint}`; AFTER triggers apply mandated side effects; tools are insert/query/update only (soft delete); tool descriptions state triggers as preconditions and side effects.
- **APIGen-MT:** policies as Python unit tests over the executed trace; they catch cross-action conflicts judges miss.
- **VitaBench:** rules as tool pre/post-conditions on the graph, no policy document, so domains compose.
- **GRADUS (2610.27781):** policies compiled into guarded operators (action, guard → outcome).
- **Warning (τ² 2506.07982):** τ² wrote its policy from the tasks and solutions, so it may encode the fault list; its "workflow" policy hurt the Oracle Plan mode.
- **Adopt (P1).** `policies_as_code` compiled to triggers or a pre-write hook, enforced in the World and re-checked in F5; policy written before tasks; policy detail level as a `hints` knob.

### 1.6 State seeding, distractors and determinism (F2)

- **Distractor rows.** ScaleEnv: schema-valid rows orthogonal to the gold path, density by target complexity. SynthAPI: a minimal core (median 59 rows) plus D−1 staged distractor batches, each told what came before, to use non-colliding ids, relate to the main user, and not pre-solve or block; D = 4 beat D = 1, 2 at high budgets.
- **Near-misses.** VitaBench: each distractor violates exactly one constraint; few valid answers. EnvCraft: dirty data, near-duplicates, outdated versions, decoys, odd formats under "exactly one correct answer". Each near-miss is a free F5 negative.
- **Boundary-adjacent (LOGIGEN):** thresholds seeded at N−1, states one action from a violation; resources by trade-offs, distractors, substitutes, noise; four user archetypes (mismatch, entangled, rookie, edge). Not ablated individually.
- **Several start states per instruction (SynthAPI):** best S = 1 at 32–512 tasks, 2 at 512–4,096, 4 beyond; needs state-generic verifiers.
- **Mechanism, then observation.** Data World: persona-seeded DAG with typed edges (monotonic, threshold, saturation, lagged, interactive) → fixed contract → canonical truth layer by layer (one session per layer, fixed seed, per-edge empirical checks in a ledger) → observation projection by a separate session with *named* defects (id spellings, delays, duplicates, unit drift, schema drift, unlabelled FKs) → read-only auditor. Tev: latent world validated against invariants → workflow DAG → render with asserted round trip → seed-derived, declared, invertible corruption, so the answer is computable from s0.
- **Real files (WorkForge 2610.04906):** one expert blueprint → K real resource pools; real workspaces had 5.4× more files and 279 vs 21.5 KB against synthesised ones; workspaces are profiled into checkable **workspace facts** and the hidden plan's claims are verified against files.
- **Perturbed copies (EnvLunch 2610.33817):** delete 30% of rows (tables ≥ 20 rows) with FK cascade, rescale measures, never touch keys.
- **Determinism.** AZR (2505.03335) runs twice and bans `random`, `datetime`, `time`, `subprocess` and similar modules. SWE-rebench freezes dependencies on first success. Kimi K2's "controlled stochasticity" becomes seed-driven fault injection in the card's adoption. AutoEnv never checks determinism.
- **Adopt (F2).** Distractor density and staged distractors (P0); boundary offsets and archetypes as knob values (P0); S > 1 seeds per proposal (P0); near-miss seeding per constraint (P1); `latent_traits` as mechanism DAG + observation systems (P1); `WorkspaceFacts{id, statement, computation_ref, evidence_path}` with each check citing a fact (P1). G2 = seed twice, same fingerprint.

### 1.7 Simulated people as World components

- **Dual control (τ²).** The user owns tools over its own state (a mocked phone) with human-readable outputs, so it can act but not plan. Simulator errors (two annotators): airline 47% (13% critical), retail 40% (12%), **telecom 16% (6%)**. No-User → Default costs 18 (gpt-4.1) to 25 (o4-mini) points of pass^1.
- **Known and unknown info.** τ² separates them per task, plus a ticket for No-User mode. VitaBench hides some facts behind knowledge boundaries. EnvFactory limits the user to external parameter values and gives it user tools (login, reset_password).
- **Recurring prompt rules.** One line per turn; reveal only what the step needs; never invent; paraphrase; stop tokens; act only on request, one action at a time (τ²); "don't know" if absent; resist persuasion; refuse after an answered question is repeated 3 times; explicit end and no-end conditions (VitaBench).
- **Stabilisers.** APIGen-MT BoN-4 user turns with a 0–10 consistency critic: gpt-4o τ-retail 62.8 → 67.0, variance 11.1 → 2.6. EAU (2610.16523) mixes a symbolic rule user with an LLM user (ε = 0.5; ColBench 0.200 / 0.177 / 0.213, within noise).
- **The user moves scores as much as the agent.** AutoForge (2512.22857): τ² Telecom 76.3 with a GPT-4.1 user vs 90.4 with GPT-5-thinking. GEM (2510.01051): strong user + weak assistant 44.3% beat weak user + strong assistant 41.7%. VitaBench: 9.2% of failed rubrics were simulator errors; fidelity 9.48/10.
- **Simulator hacking (LOGIGEN).** Trained at simulator T = 0.1, Qwen-32B's training reward rose faster, but against a GPT-4.1 simulator its Retail score fell after steps 600–700; T = 1.0 stayed robust.
- **Disclosure defects (EAU run record).** facts_per_reply 1 → 3 disclosed the reference solution; a lower volunteer_interval raised success while questions fell 40% → 26% (reverted as masking); max_reveals ignored repeats; the leak filter targeted the wrong fact kind.
- **Adopt.** The `ActorSpec` driver gets user tools and asymmetry rules (P0), the prompt rules, BoN-4 + critic (P0), a symbolic arm for gating and an LLM arm for diversity (P1), a disclosure audit per fact kind with repeats counted (P1), and p̂ keyed on simulator model, prompt and temperature (P1). Train at simulator temperature 1.0; cross-check kept tasks with a different-family simulator (B4.3).

### 1.8 LLM-simulated environments: a tier, never ground truth

- DreamGym (2511.03773): an 8B experience model emits states *and* rewards; pure synthetic RL trails real RL by up to 8–9 points (WebShop/ALFWorld, Q2.5-7B); the robust win is a sim-to-real warm start with 5K real transitions; its WebArena real baseline was starved; ε_R and ε_P never measured.
- Simia: o4-mini simulates user, tools and reward; 64 RL steps, gains 1.7–7 points, τ² seeds in-distribution.
- CompoWorld serves only non-implementable tools by an LLM (7.1% of tools; ~10% of samples), forced through the same Pydantic models. AWM's LLM-simulated env baseline was below code envs at every size (BFCL 55.52/52.53/67.68 vs 64.50/65.94/70.18).
- **Adopt (P2).** Pre-screen or warm-start tier only, behind a sim-fidelity gate (replay synthetic successes in the real App or re-grade with the executed verifier, measuring ε_R); world-model tasks excluded from the benchmark.

### 1.9 Time, events, organisations and noise

- **OfficeTown (2610.23194, PARTIAL):** roles from O*NET/ESCO with authority rules; permissioned state; shared simulated calendar with asynchronous activation and adaptive pacing; **save, accept and share as separate actions**; journalled updates. A six-month run grew 27 seed files to 497. A task = request + **task-time snapshot** of what its owner could see.
- **Agent-Reflex (2610.17953):** 24,510 real tool errors in eight families (server/gateway 33.6%, auth 17.9%, timeout 14.7%, resource missing 10.5%, argument 10.2%, generic 8.3%, rate limit 4.5%, network 0.3%), each with an admissible recovery; inject at p = 0.2 with families sampled uniformly; keep only if recovery stays possible. GPT-5.2 τ² clean 81.0 → tool noise 61.0.
- **Adopt.** `snapshot_at(actor, t)` (P0 for the trace source); permissioned views (P1); a per-App fault profile with G3 re-run under it (P1); timeout-after-write must be state-consistent.

### 1.10 Build stages as agentic sessions with host-side gates

- **Reference execution during the build** (AutoEnvScaling 2610.14163), valid envs off vs on: GPT-5.6-sol 19.5% → 84.8%, DeepSeek-V4-Flash 19.4% → 73.0%, Opus 5 52.0% → 94.0%. Prompt → feedback → agent-in-sandbox: DeepSeek-V4-Flash 3.5% → 12.1% → 89.0%. Checks run outside the sandbox; the builder edits its memory and tools only. Cross-domain admission 71.3% of 1,489 runs (terminal/SWE/GPU 87–100%, browser 26–81%, professional workflows 33–62%).
- **Role split and memory.** SWE-Factory: explorer, Dockerfile writer, test-script writer, and an analyst routing each failure to the responsible role, with a per-repo memory (no explorer 42.9%, no memory 45.0%, vs 50.2%). Data World: one session per stage sharing only on-disk artifacts, each with a ledger.
- **Adopt (P0).** F1–F5 run as machine/RLM sessions that execute the oracle during generation; gates host-side and hidden; the role that edits a test script never approves it.

### 1.11 Environment builds for the debug domain

- One image per (repo, commit): SWE-smith 125 images, 290.5 GB vs an estimated 50–150 TB per-instance (SWE-Gym 6 TB for 2,438).
- SWE-rebench agentless recipe: three calls, ≤ 3 candidates (18-instance study: 6/18 with 1, 8/18 with 3, 9/18 with 10); newest base commit per major.minor; diff the PR head commit; `--tb=line` with a flag for not-yet-existing names; freeze on success.
- SWE-Factory: binary test files arrive empty from the API (Pillow 30.4% of instances); 0% vs 45.5–50% F2P with recovery; lint scripts for `set -e` and marker typos (12 + 4 of 16 false negatives).
- **Adopt (P1).** Three-call recipe as the cheap tier for Java/JS/TS, SWE-Factory roles as escalation, valid tasks per dollar as the builder metric; per-test parsing, exit codes only as a logged fallback; check head-commit diffing, the ImportError flag and freeze-on-success in the lane (P0).

**Minor tips (B1).** Generate state keys first and fill values per task (AutoForge); summarise tracebacks with a separate cheap call (AWM); discovery meta-tools keep context small with 100+ tools; seed themes from real MCP specs and PRDs, never from benchmark task sets; live-probe real servers (TOUCAN kept 495 of ~2,800, 17.7%) and snapshot them for G2/G10; compress observations (DreamGym) without hiding what the verifier needs.

---

## B2. Task proposal and sources

### 2.1 Solution-first: the oracle runs before the task exists (F4)

Pipelines that execute a reference solution keep tasks clean; pipelines that "prove" solvability by sampling do not.

| Pattern | How the oracle arises | Card |
|---|---|---|
| Walk, execute, freeze | Directed walk over the tool graph; arguments generated and executed step by step on a seeded DB; end state frozen as gold | AgentScaler, ScaleEnv (code chain with programmatic dataflow), Agent-World graph walks |
| Blueprint | Instruction q + ground-truth **write** actions a_gt + expected outputs o_gt; reads left free; executed, recorded as a diff_patch | APIGen-MT |
| (init, solution, assert) triple | `init` sets the fault, `solution` must use agent or user tools, `assert` holds at the end; groups hold mutually exclusive subtasks; ≤ 1 per group when composing | τ² (15 groups → 2,285 tasks; 114 released) |
| Verifier first, then state, then oracle | `verify(db, answer)` from instruction and API docs before state exists; `setup_database()` fits it; `solve(db)` restricted to public APIs; keep iff v(Db0) = 0 and v(Exec(π*, Db0)) = 1 | SynthAPI |
| DAG with reasoning nodes | Merge walks, add LLM-inferred computation nodes and reasoning edges, fill arguments in topological order, execute, then rewrite the intent from (S, S*) | AutoForge |
| Tools from sub-answers | Write one tool per sub-question, merge overlapping tools, generate code that must return the planted answer | FTRL |
| Program as task | A solver script with loops, branches and aggregation, debugged until it runs; its output is the answer | Agent-World programmatic tasks; EnvLunch random valid programs |
| Solver family | A sampler plus an exact solver (DP, Held-Karp, HiGHS, SLSQP) gives the optimum u* and a default-policy value u₀ before the environment exists | VHD-Play |
| Seeded generator | build(seed) → (s0, answer key computed by code, answer contract); the LLM never writes task data | ClawLoop, Tev, Reasoning Gym |
| Probe by the author | The author executes the walk; the end joint state is the goal; each rubric criterion becomes an executable check with v(g) = 1 | CompoWorld |

- **What sampling-as-proof costs.** AWM has no oracle: 23.7–28.2% of its training tasks were never solved once by the students. Agent-World keeps a task if ≥ 2 of 5 ReAct runs pass. EnvCraft keeps 0 < p < 1 on 8 rollouts of a 27B. Endless Terminals (2601.16443) keeps o3 pass@16 > 0 (about half discarded), and about half of the kept tasks are solved by all 16 o3 attempts. EnvScaler never proves feasibility.
- **Oracle restrictions.** SynthAPI bans private helpers, raw CRUD primitives, `setattr` and direct DB access in `solve()`, but the oracle still reads privileged ids (a G8 gap). τ² requires `solution` to use policy-visible tools by construction (G7). ProgAgent requires a Python oracle restricted to the provided tools to score 1, then has an independent agent write its *own* solver at the hardest scale as an audit.
- **Minimality (R2E-Gym 2504.07164).** Delete gold-patch hunks while F2P still passes; a hunk that survives is noise or under-tested.
- **Adopt (F4, P0).** `Oracle(kind=action_program)` with `tools_used`, run twice. The τ² triple is the atomic builder for `world_synth` families, with groups stored in the DomainSpec so F3 cannot propose contradictions (P0). Mechanism families with a solver become a `world_synth` source where the domain has an optimum (P1). Frontier sampling is an audit on a sample. Oracle minimality runs before G6 (P1).

### 2.2 Act-then-label and explore-then-propose

- **LOGIGEN forward exploration:** a Client agent (owns the goal, makes imperfect, changing requests) and a Consultant agent (reads state and policy, offers a menu, resolves ids, executes) play out an episode; s_target is the actual DB after it; a user-view projection rewrites the episode into a goal-first, procedure-hidden instruction. Explorer success 99.3%. Risk: s_target includes the Consultant's arbitrary choices, so every choice must be communicated (G8).
- **SCA challenger (2506.01716):** the proposer first calls the environment's tools (≤ 15 steps) to read hidden state, may not invent ids it has not observed, runs its solution before answering, and ids the solver should discover are redacted from the instruction afterwards.
- **GRADUS grounding:** an agent with tools, policies, schemas and a state-probing tool grounds an unbound blueprint into a concrete task and reference sequence, which code replays at once.
- **Synatra (2409.15637):** show a real observation, brainstorm 8 task categories, write several concrete tasks with prescribed history lengths (0–10 prior actions).
- **Adopt.** F3 proposers explore the World through the same tools and may cite only observed ids (P0). LOGIGEN's Client–Consultant route is a third oracle route beside action programs, under G8 (P1).

### 2.3 Grounded, conditioned proposal (F3)

- **Samplers.** APIGen-MT feeds the generator an API sampler (write APIs only), a policy sampler, a domain-data sampler, a persona sampler (PersonaHub) and an example sampler, with each sampler's frequency randomised per iteration. Its feedback summariser raised blueprint yield from 28% to 70%.
- **References.** AZR conditions on K = 6 sampled past triplets and asks for a different one; replacing dynamic references with a fixed prompt cost about 5 math points (43.8 vs 46.8 overall; math −5.3).
- **Ledger.** GRADUS puts a corpus-level ledger of covered (policy, outcome) pairs in the prompt (B5.2: its largest ablation).
- **Three prompt shapes (TOUCAN):** single server, multi-server (write a server analysis and cross-server workflow first; "cannot reasonably be answered with a single tool or server"), featured server; no tool or server names in the question; target tools listed in call order.
- **Surface grounding.** VHD-Play grounds each scenario in one sampled real document passage (28 domains; topical label entropy 0.997, largest label 4.67%). ClawLoop instantiates workflow patterns in deliberately long-tail domains (beekeeping hives, ham-radio logs) to decouple skill from surface. WebUniverse keeps the interaction mechanism and changes the scenario. Endless Terminals draws category × complexity × persona at random.
- **Adopt (F3, P0).** Inputs = sampled cell + World slice + persona + K nearest tasks + the coverage ledger + a sampled real passage for surface. Self-ratings that TOUCAN collects (`stability`, `verifiable`, `tool_selection_uniqueness`) may be stored as proposal metadata but never replace G2/G10 or F5 (P1).

### 2.4 Composition operators

| Operator | Detail | Evidence and failure mode |
|---|---|---|
| τ² group composition | Concatenate init/sol/assert of ≤ 1 subtask per group | Subtask count is a monotone knob (B6.3). Combinatorial volume is not behavioural diversity |
| Reverse recombination (APIGen-MT) | Same-persona validated tasks; a_comb = a1 ∘ a2, rerun only policy tests, write one q_comb, re-validate from Stage 2 | Parts do not interact (no dataflow); skipping execution of the combined sequence is a shortcut |
| Sequential and parallel-then-sequential (BUTTON 2410.12952) | Add a task consuming the first output; or a parallel task, then one needing both | Bottom-up vs monolithic "write a complex task" prompt: −15.0% relative success on average (small model −16.1%) |
| Depth and width (TaskCraft 2506.10055) | Depth: replace an index with its minimal unique superset plus a relation, then substitute; width: merge two independent subtasks | Depth pass rate within 6 attempts 41.0% → 51.2% with prompt optimisation; width adds length, not interaction; compositions never re-solved by an agent |
| Merge + reasoning nodes (AutoForge) | Drop redundant lookups, insert aggregation or comparison nodes | Reasoning nodes need an answer check or a written result to be verifiable |
| Service walks (CompoWorld) | K = 5 services + 5 distractor services; walk length L ∈ [7, 12], covers every service, no consecutive repeats, L > K forces a revisit | Composed beat single-service at equal data (B5.3); "revisit forces stale re-read" asserted, not verified |
| 1-app → multi-app (Simia) | Prompt to extend a single-app seed into a natural multi-app workflow | OfficeBench 2/3-app gains from 1-app seeds |
| f(g(x)) (AZR) | Composition curriculum | **Collapsed**: f learned to return g(x) unchanged |
| Combine (SWE-smith) | Merge validated single bugs per file (≤ 40 combos, 2–4 bugs, ≤ 3 per file) or per module (depth-2 trie, 2–5 bugs, ≥ 2 files, ≤ 10 per module); each bug used once | 96.9% yield, but co-location only; only 130 of 5,016 SFT trajectories; never ablated |

- **Adopt (F3/F4, P1).** A composition is a program join over sub-oracles that each executed; the joined oracle re-runs (G2/G3). Prefer compositions with dataflow between parts; F7 rejects a composite whose signature is the union of its parents with no new edges (the AZR collapse is the warning). Keep the sub-task breakdown in `privileged_notes` for the F10 single-subtask probe (BUTTON card).

### 2.5 Traces to tasks and real-data mining

- **Task-time snapshots (OfficeTown):** task = request + what its owner could see at assignment time, excluding later solutions and evaluator-only data; each attempt starts fresh from these inputs.
- **Decision grounding (Trajectory2Gym 2610.33568):** in 34,772 candidates, 40.9% had ≤ 2 supervised decisions and 10.0% a certified decision after an authentic error. An action is a target only if it executed without error and its identifier-like arguments trace to earlier *visible* information (v_t); rejected calls stay as context and the grounded recovery becomes the target. Matched 8B ablations (single runs): targeted expansion +3.55, grounding +2.75, structural selection +2.32, parent normalisation +8.04.
- **First-error restore (Agent-Reflex):** an analyzer and the verifier label rollouts (disagreements dropped) and give the first erroneous turn t\* with a state snapshot; restoring to s_{t\*} yields recovery data; success/failure pairs across a nine-model, five-family pool become 1,170 skill records.
- **Other sources.** ACE-Router cuts trajectories into (history, observation) → next-tool labels (step probes, not RL tasks). Synatra turns tutorials into concrete scenarios ($0.031 per kept sample vs ~$0.80 human; nothing executed). WorkForge maps expert blueprints to real resource pools.
- **Code mining.** SWE-smith PR Mirror reverts a real PR per file against HEAD with an LM (`git apply -R` mostly fails; 92/100 django Verified instances recovered, 84 with identical F2P). R2E-Gym SweGen mines *commits*, generates missing F2P tests from the gold patch (kept only if fail-before, pass-after) and back-translates issues from execution traces (">2.5×" more tasks, figure only). SWE-rebench automates PR mining end to end (B8). SWE-Gym hand-built 2,438 of 64,689 raw.
- **Adopt.** Trace → World snapshot → task, never trace → text (P0); G8 reuses v_t over oracle programs (P0); failed traces give late-start ladder variants at s_{t\*} (P1); SOPs and runbooks seed cells with an executed oracle (P2); PR mirroring as the lane's fallback when `git revert` fails (P0); generated F2P tests only behind G5 and G14 (B3.6).

### 2.6 Self-play proposers

- **SCA:** one Llama-3.1-8B as challenger, executor and user; survival runnable verifier 47.7% → solution passes 9.5% → all failure cases fail **5.2%**; pass@1 12.0 → 23.5 (PAE 12.9; PAE's ungrounded tasks drove Airline pass@4 to 0). 200 or 400 tasks with more rollouts helped train, not test; 800 tasks gave steady test gains. Joint training over four environments gave no gain (23.0 vs 23.5).
- **AZR:** proposer reward 0 if r̄ = 0 else 1 − r̄ (G = 8); Coder 3B/7B/14B +5.7/+10.2/+13.2, Llama-3.1-8B +3.2; explicit complexity and diversity rewards gave nothing. The card notes 1 − r̄ rewards lucky near-impossible tasks (r̄ = 1/8 earns 0.875).
- **AutoEnvScaling:** base Qwen3.6-35B-A3B produced valid environments 32.3% of the time; SFT on 600 proposer trajectories raised it to 86.8% with no detectable change in solving (TB2.1 40.0 → 41.1, CI [−4.9, 7.4]). Proposer reward −1 / −0.25 / +1 for invalid / out of band / admitted.
- **VHD-Play:** a frozen Qwen3.6-35B-A3B setter admitted 70% (21/30, 2.5 attempts) once the mechanism was fixed; Qwen3.7-Max 100% (1.03); cost per admission about equal.
- **GenEnv:** generator rewarded by exp(−β(p̂ − 0.5)²) on LLM-made targets, no executed ground truth.
- **Adopt (P2).** Self-play tasks enter through F6; proposer scores apply after gates in the −1/−0.25/+1 shape; a student-sized setter is viable when the mechanism is fixed.

### 2.7 Infeasible, under-specified and clarification tasks

- ToolACE (2409.00920) data-type ablation at a fixed 25K: without non-tool and clarification data, overall 42.71 and irrelevance **6.99**.
- TOUCAN Ext.1 shuffles server metadata so the shown tools cannot solve the task, keeping zero-tool-call trajectories; core single-turn data alone dropped BFCL irrelevance 68.46 → 46.88 and τ Retail 44.46 → 36.95, and adding irrelevance data restored irrelevance to 77.85.
- Reasoning Gym's `p_solvable: 0.9` with "infeasible" as the correct answer.
- AutoEnvScaling's HiL-Bench: 500 generated under-specified tasks with a withheld-details registry served by an ask tool; Qwen3.6-35B-A3B success 6.8% → 33.9%, Ask-F1 29.6 → 55.3.
- AgentScaler's Special-subset regressions (4B −15.3, 30B −3.4 on ACEBench-zh) suggest under-represented infeasible requests.
- **Adopt (P1).** Infeasible and ask-the-user types from day one, with their own answer check ("report infeasible" or an ask event) and their own negative: an agent that always says infeasible must fail the feasible share (G4). Check that shuffled tools really cannot solve the task by an oracle search over the shown tools (TOUCAN card).

### 2.8 Fault sources for the debug domain (SWE-smith)

- Four strategies with yield, instances, ¢ per candidate: Combine 96.9%, 10,092, 0; LM Modify 56.0%, 17,887, 0.38; LM Rewrite 35.0%, 4,173, 3.93; PR Mirror 33.8%, 2,344, 5.53 (B9: the prose also says 13.18%); Procedural AST 40.2%, 15,641, 0 (13 transforms; shuffle class methods 1.9%, invert-if 49.4%).
- Training value (7B, 1,000 instances per strategy): PR Mirror 9.2 ± 1.7, LM Rewrite 8.8 ± 1.7, Procedural 8.6 ± 1.8, **LM Modify 5.7 ± 1.5** (mostly trivial variable swaps despite a 9-type hint list, 3 sampled per call).
- Always forbid comments marking the bug (models add them by default).
- **Adopt.** A `FaultSource` plug-in `kind ∈ {mined_fix, mirrored_fix, lm_rewrite, ast_op, combine}` with per-kind yield and cost meters; synthetic faults form separate families from mined ones (P1); prefer Rewrite over Modify.

**Minor tips (B2).**
- Generate tasks before schema so DB and tools exist to serve tasks (AWM, InfiniteWeb).
- Two generator styles cover different reasoning shapes: linear chains and programs with loops and aggregation (Agent-World).
- Require the proposer's atomic facts to carry a definite identifier and an explicit time range (TaskCraft).
- Bootstrap few-shot exemplars from the pipeline's own validated outputs (TaskCraft atomic pass 54.9% → 68.1%), but select them on executed yield and anchor agreement, not judge pass rate.
- Cap extension attempts and log pass rate per attempt (TaskCraft: 6).
- Style-transfer issue text from one random real issue; never name the test, cause or fix (SWE-smith, ~2.54¢ per issue).

---

## B3. Verifiers and gates

### 3.1 The two-sided filter and must-fail negatives (G3, G4, G5)

- **Two-sided filter.** SynthAPI keeps iff v(Db0) = 0 and v(oracle end) = 1. Endless Terminals requires final tests to fail on the untouched start state. SETA (2607.10891) requires a no-op run to pass zero tests and solve.sh to pass all. AutoEnvScaling requires reference = 1.0 and empty solution < 0.5. τ² requires init → sol to satisfy every assert *and* the task to be unresolved before the last solution step.
- **The filter alone selects lenient verifiers (SCA).** Human audit of 50 Retail rollouts per variant (200 trajectories, 8 human-hours): adding the solution check cut false negatives but *raised* false positives, because lenient verifiers pass the solution filter and the filter amplifies them (some passed with no action). Adding ≥ 3 near-miss failure cases (wrong item id, wrong payment method, cancel instead of update) removed all false positives. Under-specification false negatives remained (B4.1).
- **Benchmarks carry the same defect.** SCA found 40% of τ-bench Airline test tasks pass with no action.
- **Adopt (P0).** G3 + G4 + G5 are the minimum; negatives come from mutation operators and a different-family builder, not from the proposer (SCA's negatives are only the near-misses the proposer imagines). Add a side-effect class.

### 3.2 Mutation-based negatives and reward causality (G5, G6, proposed G16)

- **Reward causality (Tev).** In a passing workspace, mutate each cause and check the corresponding reward component drops.
- **Per-step dropping.** τ²'s check is on the full prefix ("not resolved before all sol steps"), so a redundant middle step could slip through [the card infers this from the wording]; strengthen to G6 by dropping *each* step.
- **Ready mutant pools.** SWE-smith's 13 AST operators applied to the oracle fix give near-miss wrong patches for G5's kill floor and a held-out mutant set. VitaBench/EnvCraft near-miss rows give "picks the decoy" negatives.
- **New negative classes from group 02.** *Partial propagation* (Interlock 2610.19686: the decision is right but some downstream records do not reflect it); *constant answer* (EnvLunch: an answer that does not change across perturbed Worlds); *stale output* (Tev counterfactuals: plausible wrong states must fail); *claimed but not done* (Kimi K2's hack-check layer; B3.4).
- **Adopt.** Extend the §3.3 operator table with "drop a realisation", "constant answer", "stale output" and "claim success" (P0 for the first two per the Interlock and EnvLunch cards). Propose **G16 reward causality** (P0 per the Tev card).

### 3.3 Exploit probes, hackability and hardening (G12, proposed G15)

- **Generic probe battery (Tev):** 14 gaming probes (empty output, minimal schema-valid output, copied input, single-seed hard-coding, path traversal, private lookup, …) must score 0 on both the verifier and Φ. Proposed as **G15**.
- **Exploit taxonomy and detectors (RHB 2605.02964).** Six categories with deterministic detectors over command logs, reads/writes, protected-file diffs and recomputation discrepancies: (1) leakage/metadata (`_meta/**` reads); (2) tampering (writes under the task dir, harness edits); (3) sequence manipulation (forged completion markers, fabricated intermediates, skipped verification); (4) proxy or parser gaming (minimal valid JSON, defaults, duplicate keys, self-reported runtimes); (5) special-casing visible checks (hard-coded instances); (6) denial of evaluation (timeouts, crashes; fail-closed). Precedence: tampering > denial > sequence > leakage > proxy > special-casing. Detector audit: 94% agreement on 50 flagged runs (3 borderline false positives), 0 false negatives on 50 unflagged. Rates are lower bounds.
- **Hardening works and is free.** All four toggles: exploits 6.5% → 0.8% (−87.7%), success 83.2 → 82.8 (n.s.). Single toggles: hardened boundaries −41.5%, reduced file access −36.9%, step verification −32.3%, randomised intermediate outputs −20.0%.
- **Where exploits rise.** Exploit rate jumps at chain length 5, the first step the agent cannot privately verify (MicroRHB chains 2.3% vs CoreRHB 8.1% at length 5). Hard variants that raise honest work 2–10× with the exploit surface unchanged raised exploits for all 13 models (o3 11.8 → 19.6; R1-Zero 13.9 → 22.3). Within every vendor the RL-reasoning model exploited more (DeepSeek-V3 0.6% vs R1-Zero 13.9%, same base). 72% of exploit episodes had an explicit efficiency rationale; 28% had none, so CoT monitoring alone misses a quarter.
- **Exploits that passed weaker gates.** AutoEnv's "verified" reward scored **138.84%** of its max on 1_InterDimension (profit reward accrues every positive increment, stability bonuses nearly every step); the authors call the bound conservative, the card calls it an exploit. AWM's "Partially Completed = 0.1" can be farmed. InfiniteWeb checkpoints reward attempted actions, so repeating the rewarded action can farm them. FTRL's substring reward rewards answer dumping. AutoForge's MEU masker can be gamed by confusing the user into an out-of-scope reply so the trajectory is masked.
- **Adopt (P0).** RHB's six categories become the attacker's checklist and the `hack_flags` labels; the detectors run as cheap deterministic G12 pre-checks before the paid attacker. Add G15 (Tev's battery plus "repeat the rewarded action"). Any rollout above the oracle's reward, or above 1 normalised, triggers G12 review. Re-run G12 on every variant that raises the knob band, and place at least one privately unverifiable check (hidden split or held-out state) inside long tasks.

### 3.4 Trust boundaries and fail-closed grading

- **RHB boundaries:** grade outside the sandbox by recomputation on hidden data; read submissions from one output path only; mount only the current step's inputs read-only, copy prior artifacts into a read-only `prev/`, unmount decoys and grader metadata; enforce chain dependencies by recomputed hashes or per-instance nonces, not file existence; parse fail-closed with strict schemas, reject duplicate keys and defaults; measure metrics in the grader, never trust self-reports.
- **Role independence (Tev):** reference solver, alternative solver, strict verifier and progress scorer have disjoint, audited import closures.
- **Others:** AutoEnvScaling runs admission checks outside the builder sandbox; ProgAgent hides verifier and oracle in a Bubblewrap sandbox; EnvCraft binds sessions by injected env vars and hides trainer commands; Data World's auditor is read-only and pass/fail only.
- **Fault classes in the reward.** AWM separates Environment Error (reward 0, not the agent's fault) from Agent Error and ends rollouts on format violations with −1 (27% less rollout time; format errors otherwise stay >20%). EnvCraft treats verifier crashes or timeouts as data defects (≤ 2 automatic verifier repairs, else discard), not agent failures.
- **Adopt.** Hardened boundaries = verifier custody; reduced file access = path constraints; step verification = milestone checks with nonces; randomised outputs = per-seed variation, with G2 comparing fingerprints modulo the nonce (RHB card). Role independence becomes a provenance check on code closures (P0, Tev card).

### 3.5 State diff versus judges

- **Canonical state diff (LOGIGEN).** DIFF(a, b) = Σ_tables |rows(a) Δ rows(b)| over canonicalised rows (auto ids and timestamps dropped); R = 1 iff DIFF(s_final, s_target) = 0. The symmetric difference catches extra writes too.
- **Column policies (ScaleEnv).** Each column exempt (generated ids, optional), hard (prices, quantities, timestamps) or semantic (free text). Reward ablation on τ² (avg of 3 domains): LLM judge avg@4 36.5, pass@4 58.8 vs rule-based 38.5, 62.9.
- **Judge plus code evidence (AWM).** 8B: LLM-only 55.46 BFCL / 26.44 τ²; code-only 60.00 / 29.59; code-augmented judge 65.94 / 33.45. Judge reliability over 100 trajectories × 5: GPT-5.1 self-consistency 90.8%, κ 0.891, reward flip 9.2%; Claude-4.5 flip 18.0%; Qwen3.5-122B 23.7%. Judge cost about $1.80 per RL step.
- **Rubrics make judges work (VitaBench).** Sliding-window judge (w = 10 turns, overlap 2, persistent monotone rubric state): task accuracy 95.0, κ 0.828; without the window κ 0.604; **without the rubric checklist κ 0.018** (task accuracy 22.0). Monotone state can reward a correct statement followed by a wrong action.
- **Selection signals (WorkForge, 1K trajectories, threshold 0.7):** LLM-only ALE 15.62, programmatic-only 14.89, hybrid 22.11. **Agent-as-judge vs text-only judge (Interlock):** AutomationBench 18.33 vs 13.17 (text filter below base 14.33), at 11% more judge tokens.
- **Trajectory judges overrate claims (R2E-Gym).** Removing the trajectory and keeping only the patch drops the execution-free verifier's Best@26 from 42.8% to 37.6%; attention peaks on self-congratulatory summaries in an incorrect trajectory. SWE-Gym's ORM: Pass@16 42.8 vs Best@16 32.0, about 10 pp left on the table.
- **Verifiers from the agent's view fail (Tev).** An LLM writing the verifier from the public view: 0.24 S@1 with **88% of GRPO groups discarded**, not above base (0.25).
- **Other check styles and their holes.** EnvScaler "Has …" checklists with exact values (IDs and timestamps checked for presence and type) give partial credit and no collateral check. AgentScaler's exact tool-and-argument match rejects valid alternatives. EnvFactory mixes trajectory match and state (α = 0.5 best, 41.38; α = 0 and α = 1 worse), which teaches path imitation. SWE-Factory's single exit code collapses F2P and P2P and cannot tell "fails for the right reason".
- **Adopt.** For write tasks the reward is the deterministic state check under column policies plus a collateral DIFF (P0). Judges may adjudicate fault side (`env_error` vs `agent_error`) or grade answer/recommendation intents with atomic rubrics generated from the oracle (each rubric item = one state predicate or answer fact), never award pass on write tasks. Any judge used for labels scores the end state with the agent's narrative stripped, or is ablated with it stripped (P0, R2E card).

### 3.6 Collateral, realisation, alternatives and false negatives (G14, proposed G17)

- **Collateral.** τ² assertions check the fix, not that nothing else changed. SCA, CompoWorld and EnvScaler have no collateral check, so "right thing plus extra damage" passes. APIGen-MT gives judges the diff_patch and tells them extraneous changes fail.
- **Realisation (Interlock).** Each required effect y = (public basis, object population Ω, applicability α, outcome φ): "for every object in Ω where α holds, φ holds". Checks: 44% content inspection, 41% recomputation, 11% initial–final comparison, 4% text matching. Credit propagates through declared dependencies; change criteria already true at start are excluded. Follow-through rate FTR = correct upstream decision AND all downstream effects.
- **Test quality (R2E-Gym App. C.3).** Distinguish(t) = 1[max over incorrect patches of Pass ≠ max over correct]; Toxic(t) = 1[max over incorrect > max over correct]. For most problems < 20% of generated tests distinguish the top correct from the top incorrect patch; toxic tests reach up to 10% for a minority of problems. SWE-rebench's rater estimates test-patch correctness at only 67% accuracy.
- **Two independent solvers (Tev).** A reference and an independently authored alternative must give identical output trees on every seed; disagreement = unresolved spec → reject. Proposed as **G17** (oracles from different families must agree on the end-state fingerprint).
- **Avoiding single-reference false negatives.** τ² telecom uses assertion functions only, so any fix path passes; ClawLoop tests locate records by business keys, not hard-coded ids; EnvFactory masks arguments that do not affect correctness (limit, page size); Reasoning Gym keeps the latent instance in metadata so verifiers can accept alternative answers; FTRL's planted function returned "not found" for a correct but differently phrased argument ("Emir Nawaf…" vs "Nawaf…").
- **Adopt.** Add **realisation checks** as a sixth check kind beside collateral (its dual: everything that should change did) with partial-propagation negatives (P0). `VerifierBundle.quality = {distinguish_rate, toxicity_rate, per_check kills}` computed over oracle + alternatives (correct set) and negatives (incorrect set); every check must kill at least one negative and toxicity must be 0 (P0, R2E card). Generated tests (R2E's gold-informed F2P) count only if they pass an alternative valid fix and kill the revert and generic mutants (P0). Business-key lookup is a column-policy convention.

### 3.7 Gold sanity, replay agreement and progress validity

- **Replay agreement (VHD-Play).** Admit only if replaying the solver's policy through the generated environment reproduces u* within tolerance; audit 80 held-out environments × (optimal, default, idle drivers) × 2 runs: 480 executions, no exceptions, all 240 repeated outcomes agreed, no realised utility exceeded u*.
- **Progress validity (Tev).** Φ(s) = Σ w_v · c̃_v(s) over checkable workspace nodes, credit gated by the weakest prerequisite; Φ must lie strictly in (0, 1) on partial states, rise monotonically to 1 along the reference and go negative on regression. Metamorphic rewrites must keep the grade; six seeds must give distinct world ids, public-tree hashes and task ids.
- **Flake.** SWE-rebench reruns N times and drops a task that fails once (N unstated); the card keeps Crucible's 60-grade Clopper–Pearson bound and adopts SWE-rebench's flake attributions (external API calls, exact float equality) as cheap lint before it. With a simulated user, τ²'s $0.145 per task-run makes 60 grades about 60× that per task, so the card suggests a cheaper simulator or sequential testing.
- **Adopt.** Replay agreement as a G3 variant for solver-backed domains; default and idle drivers as cheap G4 null agents (P1). `milestones` follow Tev's progress contract and F6 checks progress validity before Φ may be a reward (P0).

**Minor tips (B3).**
- Make every generated script print a unique greppable marker; lint for `set -e` and marker typos (SWE-Factory).
- In an LLM reward prompt, forbid judging by the agent's own "task finished" claim and ask for evidence plus confidence (Simia).
- Write prescriptive anti-hack rubrics for any learned judge (no flattery, no self-justification) and recalibrate learned critics only for advisory use (Kimi K2).
- Decompose model verification into narrow sub-questions (argument grounding, completion, output vs schema) instead of one "is this correct?" (ToolACE).
- Validate any judge on about 50 human labels and pick by correlation per cost (TOUCAN: Kimi-K2 chosen).
- Pin and version the grader; minor grader differences change reported scores a lot (GEM).
- Per-column exact policies need G8: the user must be able to communicate exact values (ScaleEnv card).

---

## B4. Intent, validity, communicability and leakage

### 4.1 G8 as a deterministic pre-rollout gate

The group-02 header calls G8 the most common silent defect. The cards give static tests from three sides:

- **Oracle argument provenance.** Label every tool parameter *external* (a user can supply it) or *internal* (system state, must be read from the World) (EnvFactory; EnvCraft's "user-specified vs tool-derived" with tool-derived values referred to only abstractly; InfiniteWeb's system-managed vs user-provided, with interfaces exposing only the latter). Trajectory2Gym's v_t: an identifier-like argument is valid only if earlier visible information supports it. Known violations: SynthAPI's oracle reads privileged ids; APIGen-MT's o_gt may be copied from sampled data the agent cannot read; LOGIGEN's target includes the Consultant's arbitrary choices.
- **Verifier assertions against the instruction and World.** SETA's worked case: five GPT-5.4 rollouts all failed the same two tests because the tests demanded four JSON literals the instruction never stated, and oracle and tests shared the assumption, so no-op/oracle checks could not catch it. RST (2608.05466) rule D.1: every reward-changing condition is stated, documented in a local file the instruction points to, a named secondary deliverable with a documented schema, or a pure anti-placeholder check. WebUniverse writes tests from the statement before any solution, then reviews *only failed tests* for unstated assumptions and regenerates them. SWE-rebench flags tests that reference not-yet-existing names (the agent must guess exact signatures).
- **Hidden decision rules against a declared contract.** ClawLoop's **answer contract** names each hidden decision (filter criterion, rounding, tie-break, join key), the deliverable schema and numeric tolerance, without values.
- **Recoverability.** Data World's auditor, without access to hidden truth, must re-derive each targeted finding from the observed files. Interlock's "public support" critic sees only (request, workspace, rubric) and checks criteria follow from public inputs.
- **What execution cannot catch.** SCA's residual false negatives came from under-specified instructions ("return one of my latest orders").
- **Adopt (P0).** G8 runs before any rollout over (1) oracle argument provenance via `ToolSpec` per-parameter `provenance: external | internal` (external values appear in the instruction or `user_known_info`; internal values are produced by an earlier oracle call), (2) every verifier assertion and literal against instruction + start state under RST's four-way classification, (3) every hidden rule against an `AnswerContract{hidden_decisions, schema, tolerance}`, and (4) for analytical tasks, a recoverability session without hidden state. G8 also covers the user side: every user-side solution step must be something the user can be asked to do with its tools (τ² card).

### 4.2 Shortcut and no-World probes (G9)

- **TaskCraft:** keep a task only if a tool agent beats the same-strength model without tools (AgentScore > LLMScore and > 0); after each depth merge, an infer-LLM tries the merged task without tools and the task is dropped if it succeeds. Two leak classes: a *pseudo-superset* (a synonymous title gives a hop that needs no tool) and *answer in question* ("$34.5 billion" stated while asking for the exact amount).
- **FTRL:** factual questions about real events are answerable from memory; prefer synthetic entities.
- **Adopt.** G9 literally for answer tasks, re-run after every knob change (P0); both leak classes as string/entity `LeakRule`s against the gold answer before any model call (P1).

### 4.3 Ablation probes that test what success needs (F10)

| Probe | Mechanism | Evidence |
|---|---|---|
| No-User / Oracle Plan (τ²) | No-User gives the agent the ticket and all tools (isolates reasoning); Oracle Plan gives the full tool sequence (isolates communication) | Default vs No-User gap 18–25 points; a near-zero gap means the user part is decorative. No-User changes two things at once (card critique) |
| Written-out / informed / agentic (VHD-Play) | The same draw presented written-out, with parameters revealed, or with parameters acquired by probing | Base 0.962 / 0.231 / 0.204 → trained 0.992 / 0.875 / 0.815: most of the gap is stateful execution, not problem solving |
| Interface (ProgAgent) | Same task under direct tool calls vs code that calls tools | DeepSeek V4 Pro pass@8 under DTC falls 0.95 → 0.42 from v0 to v3 but stays > 0.87 under PTC: a context-load test under one harness, not the other |
| Budget and access (RES 2610.16397) | Remove composed dependencies; give direct access | Removing composed dependencies → 100% pass (from 0%) on 15 tasks; the failure case ran out of turns while still acquiring evidence |
| Stronger or different simulator | Rerun with a stronger (GEM) or different-family (LOGIGEN) user | A pass-rate jump means hardness lives in the simulator |
| Skin inversion (AutoEnv) | Invert only the rendering with rules fixed | Numeric-label inversion 26.67%, symbol 0%, pixel 8.62% |
| Stylistic knob (Reasoning Gym) | Vary names, formats, framing | A p̂ shift on a stylistic knob means the task depends on wording |
| Masking alarm (EAU) | Success up while the target behaviour (questions asked) goes down | Caught a simulator volunteering answers |

- **Adopt.** IntentCard ablations add No-User and Oracle Plan for multi-turn tasks (P1), a written-out version of the same draw (P0 per the VHD-Play card), the interface under the target harness, and budget-2× plus direct-access controls on any RES-style or RST-style variant, labelling hardness "access/budget" when success returns (P0 whenever those knobs are used). Record harness and simulator in `DifficultyRecord` and the IntentCard.

### 4.4 Leakage and contamination

- **Leaking the grader changes behaviour, not just score (SWE-smith).** Issue text ablation (600 PR-mirror instances): templates 6.4, F2P test as issue 7.3, LM issue 7.7, original issue 7.8; given the F2P test, the student wrote a reproduction in 127 vs 379 of 500 Verified runs. R2E-Gym includes truncated test code in issues and does not ablate it.
- **Construction regenerates benchmarks.** AgentScaler reproduces τ-bench schemas; ACE-Router's 627 seed tools include MCP-Universe's own; APIGen-MT and Simia train on τ-bench's own domains; EnvScaler seeds from API-Bank/ToolACE task sets and AWM reports 46.8–57.1% blocked tasks in EnvScaler envs; LOGIGEN's 8 domains are τ²'s.
- **Checks in use.** AutoEnvScaling: 13-gram overlap with held-out benchmarks, benchmark pages blocked from builder web access. RST: 0 exact 13-token matches with TB2, TB-Hard, LHTB; max 5-gram Jaccard < 0.009. WebUniverse: 5-gram Jaccard > 0.6 or MiniLM cosine > 0.55 against 1,029 benchmark statements, all hits manually reviewed (514 retained as confirmed false positives, flagged); registered-domain matching for vision. SWE-rebench: date flags per evaluated model's release; DeepSeek-V3-0324 39.7% Verified vs 21.3% fresh (read as a *gap pattern*, since difficulty differs). Agent-Reflex: environment-family splits, near-duplicate checks against benchmark instructions *and tool schemas*.
- **Disclosure inside the World.** EAU's facts_per_reply leak (B1.7); Environment Tuning's hints come close to the judge's own leak line, and its stage transitions were chosen on the held-in test split; SCA redacts discoverable ids after construction.
- **Leak rules can cost signal (AZR).** Removing globals and comments that reveal answers *lowered* performance. The card says measure before stripping.
- **Adopt.** G11 rejects reports quoting hidden test identifiers or verbatim assertion lines (P0); add tool-catalogue and schema overlap against held-out benchmarks to G11, not only text (P1); block benchmark pages from builders and run 13-gram decontamination (P0 per AutoEnvScaling card); model-release-date flags per evaluated model on benchmark reports (P1).

### 4.5 Fault sides and failure taxonomies

- **Labels with evidence.** AWM's Environment Error; AutoForge's MEU (an LLM flags trajectories where the user gave wrong, out-of-scope or missing information; they get zero weight, false-mask rate never measured); VitaBench's 9.2% simulator share; τ²'s simulator error types (premature termination and missing constraints are critical; all 8 telecom errors were premature `###TRANSFER###`); SETA's post-rollout judge labels all-fail tasks `too_hard` or `design_flaw` from per-test failure frequencies (94 tasks, about 2%, dropped; a lower bound since only r = 0 tasks were audited); EAU attributes each bad trajectory to policy, simulator, environment/verifier or config.
- **Trace taxonomies.** SWE-smith: error/context, stuck at localisation, stuck at reproduction, incorrect localisation, incorrect edit (by comparing touched files to gold); a ≥ 10-long repeated base command ≈ 89% failure. Endless Terminals: command diversity after the first error 0.49 in successes vs 0.18 in looping failures. ClawLoop: misunderstanding, wrong tool, incorrect edit, premature termination, missing capability.
- **Adopt (P1).** MEU becomes the simulator-side fault label, measured against the anchor set, never triggered by the policy's own claim, with a per-task mask rate whose rise is a World-loop signal (G8 on user info). EAU's defect list seeds the fault-side taxonomy and the anchor set.

### 4.6 Instruction–check consistency

- AutoEnv's published `agent_instruction` for 22_ColumnStrategy (a Connect-Four DSL) describes a 10×10 grid-navigation task, and no gate checked it.
- **EnvCraft's six cross-artifact invariants:** path consistency, data consistency (every expected value derivable from the built files), objective consistency (verifier checks only what the request implies), answer uniqueness, information separation, code-only evaluation.
- **RST:** derive verifier checks from the transformation contract, never from the oracle's incidental commands; a cross-file pass repairs artifacts produced but not in the contract, verifier requirements without public evidence, and instruction requirements never checked.
- **Interlock carrier maps:** each generator role records where each requirement appears, and a source-coverage review requires every typed relation to reach instruction, workspace and rubric.
- **Adopt.** EnvCraft's invariants as a deterministic pre-gate before G3 (P1); an instruction–code consistency check by a different-family reviewer, with G13 calling every documented action (P1); a carrier-map coverage-trace gate (P1).

**Minor tips (B4).**
- Split every spec into an agent-visible brief and builder-only notes (SETA; Endless Terminals' privileged ground-truth block).
- Self-review removes tests that enforce unstated conventions (SETA), but self-review by the builder's own model is weak (AutoEnvScaling card).
- Report "misses a critical (P0) requirement" beside pass and weighted score (OfficeTown: GPT-6 Astra misses one on 20.8% of tasks).

---

## B5. Diversity and coverage

### 5.1 Families over instances, with a scale caveat

- **More Worlds beat more tasks per World:** ScaleEnv pass@4 rises monotonically over 2/4/8/16 domains at a fixed 1,024 tasks (single run, uneven task counts per domain); AWM 10 envs degrade below base, 100 help, 526 more; Agent-World gains to about 500 envs, diminishing after (0 → 1,978: 18.4 → 38.5 average of 4 subdomains); EnvScaler steepest 0 → 20 envs; EnvFactory 50 → 75 → 85 rising with diminishing returns; SWE-smith roughly log-linear over 4/25/50/100 repos at 700 trajectories; Interlock 16 → 128 domains monotone at a fixed 8,957 trajectories; ToolACE 6/14/30 API clusters (figure only); SCA only 800 tasks gave steady test gains.
- **The nuance (SynthAPI law, SFT not RL).** At 64–512 tasks A = 8 Apps is best; from 4,096 tasks A = 64 wins with a higher ceiling; even at 1,024 tasks per App only a median 51% of an App's tools are used. Fixed 32,768 tasks: Opt (A = 64, T = 128, S = 4, D = 4) beats SubOpt (A = 32, T = 1,024, S = 1, D = 2) everywhere (AppWorld Test-C 4B 38.4 → 45.0). Law P̂ = σ(c0 + c_A log2(A·S·D)) · σ(b0 + b_T log2 T)^ν: Apps, seeds and distractors set the ceiling; tasks per App set the rate. Extrapolation MAE 0.0285 vs 0.1706 for a task-count-only law. The 27B gains least and loses on ASTRA (77.3 < 80.2 zero-shot).
- **Counter-evidence.** SWE-Gym read "sampling compute, not repo diversity" as the bottleneck, bounded by 11 repos with one at a third of the data; EnvScaler's train–test similarity ablation found similarity does not drive the gain (32.00 vs 32.50).
- **Adopt (F0/F11, P0).** Apps (A), tasks per App (T), seeds per task (S) and distractor density (D) are explicit budget knobs; the default at 500–1,000 is low T with several S; test the allocation for RL per student rather than assume it. Reuse SynthAPI's protocol (fit on low-FLOP runs, extrapolate, normalised score) to measure generator value in F12 (P1).

### 5.2 Corpus-level coverage instruments

| Instrument | Mechanism | Evidence |
|---|---|---|
| (policy, outcome) ledger (GRADUS) | Guarded operators give enumerable (rule, outcome) pairs; a shared ledger of covered pairs goes into every proposer prompt | Removing it: telecom 0.75 → **0.54**, the largest ablation; 1.4–2.9× more pairs than baselines |
| Strength-2 covering array (Tev) | Every valid pair of values of any two of 19 axes must co-occur in an admitted env; extend greedily by the generator/seed/tier covering the most uncovered pairs | Detects template locking that per-axis balance misses |
| Capability-path support (Trajectory2Gym) | Count support per realised path; synthesise toward paths below a target depth d_s (1, 2, 3, 4, 8 from 1.7B to 30B) | Uniform selection of B of N misses a once-represented path with probability 1 − B/N; structural selection covered 942 more paths at the same budget |
| Trees and communities | ToolACE domain → functionality tree; AgentScaler Louvain communities; Agent-World Ward clustering into 50 second-tier clusters and 20 first-tier types; Kimi K2 hierarchical domain evolution | Count filled leaves; t-SNE plots are not coverage metrics |
| Caps and quotas | RST: ≤ 4 descendants per parent, ≤ 160 per category, ≤ 320 per rewrite family, ≤ 280 per cohort, relaxed on a logged schedule; lineage gate ≥ 80% recoverable ancestry or the round aborts. RODS per-type seed quotas; TOUCAN category rebalancing; AWM category caps; SWE-smith "each bug once" | RST: no seed > 0.77% of a round |
| Domain sampling (Synatra) | Temperature-sample domains, P′ ∝ p^{1/T}, T = 0.6 | T < 1 sharpens toward frequent domains, so tail coverage falls |

- **Adopt (P0).** Put the ledger in the F3 prompt beside the K nearest tasks; require a strength-2 covering array over the DomainSpec axes in F11 in addition to MAP-Elites fill; RST's caps and lineage gate go into archive quotas; path support counts become a coverage meter; never subsample rare cells at random.

### 5.3 Composition and mix as coverage axes

- **CompoWorld (Qwen3.6-35B-A3B, 1K SFT trajectories, equal environment count):** composed vs single-service τ³-Banking 9.62 → 17.53, AutomationBench 12.83 → 33.33; single-environment SFT slightly *hurt* τ³-Banking and DeepPlanning versus base.
- **Mixes:** WebUniverse vision-only SFT drove WebGen functional accuracy to 12.1 (below base 34.5), text-only helped function, the mixture was best everywhere; TOUCAN multi-turn and irrelevance extensions carried most agentic gain, single-turn data alone trained over-calling; EnvCraft general Tool data alone gave only +1.57/+1.74, Tool-then-Claw ≈ Claw only; ToolACE's data-type ablation (B2.7).
- **Adopt.** "Cross-App" is an axis of the coverage grid (P0 per the CompoWorld card); mix families for the student.

### 5.4 Structural versus surface variation

- Reasoning Gym separates difficulty, structural and stylistic parameters. GRADUS separates structural kernels (causal, parallel, repetition) from interaction kernels (indirect reference, ambiguity, user pressure, extra reasoning) and lets code schedule kernels while the LLM may decline incoherent ones. AutoEnv separates BaseEnv (rules), ObsEnv (observation policy) and SkinEnv (rendering).
- TOUCAN Ext.2(a) "same tools, order, outcome, new persona" is the same behaviour signature, so a duplicate to the learner; τ²'s 2,285 combinations from 15 groups over a 4-customer DB are combinatorial, so the balanced 114 (by intent × subtask count) is the better unit.
- **Adopt.** Knob schema gets `class: difficulty | structural | stylistic` (P1); stylistic and interaction variants must keep the F7 signature and should not move p̂; structural ones must change it; paraphrase variants are augmentation inside one task, not pool entries; a family's size is its atomic group set, not its composed count.

### 5.5 Measuring diversity and drift

- AWM: scenario dedup at cosine 0.85; code-level AST-duplicate functions 0.0% and endpoint-name Jaccard 0.004 across envs; mean pairwise distance 0.31–0.35, independent of generator model.
- RST over 15 rounds: domain entropy 0.821 → 0.817, rewrite-family entropy 2.26–2.31 bits (max 2.32), 36/40 operators used, but **nearest-neighbour similarity median 0.223 → 0.464 (p95 0.703)** and bootstrap coverage 98.3% → 60.1%.
- SWE-smith "bug coverage" (share of tests broken by ≥ 1 instance): 89% vs 7.25% for SWE-bench flask. AZR: program complexity and AST-edit diversity rose without being rewarded. AgentScaler, Endless Terminals, TaskCraft, AutoEnv and LOGIGEN report no dedup.
- **Homogeneity.** SynthAPI uses one builder model; RST one family builds, audits and probes difficulty; WebUniverse's kept trajectories are 75.3% from one teacher; TOUCAN spreads tasks over five generator models and Agent-Reflex mines skills from a nine-model, five-family pool. AutoEnv's inverse-semantic environments scored higher (40.69 vs 36.15), but inverting only the skin of aligned ones cut scores 68.8%, so the generator made the requested axis easier: difficulty was entangled with the axis.
- **Adopt (F8, P1).** Add AST-duplicate rate and endpoint Jaccard across Apps; track the nearest-neighbour similarity tail per round as RST's alarm; measure difficulty per cell rather than assume balance from labels.

### 5.6 Silent selection biases

- Environment builds skew the mix: SWE-rebench installs at least one task in only 31% of repos; SWE-Gym hand-configured to avoid biasing away from complex dependencies; SWE-Factory notes an install agent may succeed by memory on popular repos.
- Verification is a hidden curriculum: GRADUS shows recovered tasks are disproportionately hard (medium-hard + hard 41% of first-round accepts vs 75% and 83% of round-2 and round-3 recoveries); SCA filtering narrowed the difficulty distribution for an 8B challenger but not a 70B; APIGen-MT's 67% Phase-2 rate with gpt-4o drops the hardest third; Interlock keeps only tasks both students solve once.
- **Adopt (P1).** Log build failures and gate deaths by repo, language, dependency profile and cell; track the difficulty mix of first-pass vs repaired accepts as a meter.

**Minor tips (B5).** Hold out whole communities or families (15–20%) for OOD claims (AgentScaler card); keep the passing prefix of iterative sequences (WebUniverse); use mechanism families, not application domains, as the operator taxonomy (RST); the 3-tier Agent-World taxonomy and stratified arena (K = 5 per first-tier category) double as a held-out family meta-evaluation.

---

## B6. Difficulty and curriculum per student

### 6.1 Band rules in use

| Source | Rule | Notes |
|---|---|---|
| AutoEnvScaling | 4 rollouts, expand to 8 if the mean is within 0.15 of a band edge; keep if mean ∈ [0.25, 0.75] **and std ≥ 0.1**; below band: one diagnostic rollout with reference-solution hints, ≤ 2 revisions; pool of 256 re-reviewed every 16 solver updates | The published form of §3.6's sequential k; 3.1× more non-zero-variance groups than Tmax |
| RODS (2606.19047) | Reuse GRPO rollouts (K = 16): mastered r̄ > 0.85, boundary 0.20–0.85, hard < 0.20; rank seeds by 4r̄(1 − r̄) | Boundary-zone variance 2.0–2.2× the others over 4,800 measurements; uses a dense progress reward |
| ClawLoop | N = 8, keep 1–5 passes | In band 65.6/45.8/55.4% over three iterations |
| ProgAgent | 0 < p < 0.8 at **every** rung of a scale ladder, K = 8 on DeepSeek V4 Pro | Band set on a frontier model, not the student |
| EnvCraft | 0 < p < 1 on 8 rollouts of a 27B | Tuned to the 27B, not the trainee |
| Reasoning Gym | Raise a level when accuracy over the last 20 steps > 0.70; config has a 0.10 back-off [the card reads it as the lower trigger] | Count Primes: curriculum 30.67 vs uniform 4.00 though it never left level 1, so in-band training carries the value |
| GenEnv | Generator reward exp(−β(p̂ − 0.5)²); batches with \|p̂ − 0.5\| > 0.1 excluded | Measured on unverified tasks |
| DreamGym | Seeds = tasks with V = p(1 − p) > 0, cap λ on generated share | λ n/r |

- **Statistics.** GenEnv's bound: telling two task types apart whose band distances differ by Δ needs n ≈ (9/2)·ln(4/δ)/Δ²; the card computes about 490 rollouts for Δ = 0.2 at 95%, so per-task band placement at k = 8 is noisy and pooled cell-level estimates are needed. Tev: the all-fail group share is (1 − p)^G, 66% at p = 0.05, G = 8.
- **Adopt (F9, P0).** AutoEnvScaling's band + spread + adaptive k as the sequential rule; RODS's zero-cost reuse of training rollouts for p̂; pooled cell-level p̂ beside per-task p̂.

### 6.2 Per-student calibration

- **The evaluator must be the learner (ToolACE).** Learner LLaMA-8B as its own complexity evaluator 59.22; Qwen1.5-7B as evaluator 57.61; Qwen1.5-14B 57.67. A stronger evaluator drops samples it finds easy but the learner does not.
- **Pool value decays with student size.** SynthAPI's 27B gains least; R2E-Gym's 14B saturates near 800 trajectories while the 32B keeps improving; TaskCraft's trajectories helped a base 3B as a warm start (0.140 → 0.245) but barely an instruct one (0.230 → 0.236); CompoWorld's 397B baseline scored 5.50 on AutomationBench, below the 35B.
- **The harness and simulator are part of the band.** Tev reports difficulty per learner under a declared harness; WorkForge's Claude-Code-trained model beat OpenClaw-trained under Claude Code (24.2 vs 16.1) and the reverse on WildClaw (+6.2); OfficeTown's V4-Flash weighted score 61.9 / 48.6 / 40.7 across three executors.
- **Depth routing (GRADUS).** Depths 1–3 mostly SFT, depths 4–5 mostly RL.
- **Adopt.** `DifficultyRecord` keys on (task, student, checkpoint, harness, simulator) (P0); test the student's NLL on the oracle program as a one-forward-pass predictor feature against measured p̂ before relying on it (P1, ToolACE card; its evidence is SFT loss).

### 6.3 Structural knobs with evidence

| Knob | Evidence |
|---|---|
| Subtask count | τ² pass^1 near 0 beyond 7 actions in Default; failures concentrate in conditional mobile-data and MMS intents (a `linkage` knob: one fault masks another) |
| Reasoning points and tool-graph size | VitaBench across 4 domain aggregates: reasoning points and tool-graph size track difficulty, search-space size does not (suggestive, not a fit) |
| Inherited, composed access transforms | RES: 17 operators (sharding, pagination, field references, numeric re-encoding, archive views, index reconstruction, staged release…); frontier Claw-Eval-Live pass from about 85% to 5.71–28.57%; **independent single transforms 93.3% vs inherited 6.7%**; basic 6-operator library 80.0% |
| Recursive solution-first rewrite | RST: 40 operators in 5 mechanism families, a transformation contract per rewrite; DeepSeek-V4-Pro pass@4 90% → 2.5% over 15 rounds, mostly more work (solution 67 → 374 lines) |
| Walk length and revisits | CompoWorld L ∈ [7, 12] separate from service count K |
| Weak edges and hidden names | Agent-World: raise weak/independent edge probability; strip tool and schema names |
| Superset substitution | TaskCraft depth; World version = replace a key with a description that `StateAdapter.query` resolves to exactly one row |
| Data volume | ProgAgent rows 1/2/4/8×, files 1/2/3/4×; training on v0 only cut LOCA-bench 38.3 → 23.7 (14B) |
| Co-scaled problem size | VHD-Play knapsack 6×5 → 11×11: gains +0.41 → +0.86 while base falls 0.432 → 0.085 |
| Scale tier × DAG depth × corruption classes | Tev |
| Multi-hop by construction, constrained dirtiness | ClawLoop: index file → per-entity details (none sufficient alone) → a cross-cutting reference |
| Boundary seeding, distractor density, confusable tools, noise | LOGIGEN; SynthAPI D = 4; ACE-Router; Agent-Reflex p = 0.2 |

- **Hardness is often mislabelled.** Reasoning Gym's own "hard" configs move several parameters the easy way (mini_sudoku fewer blanks, puzzle24 smaller range). SETA's operators moved Qwen3-8B in the declared direction 77% (decrease: median 6% → 38%) and 60% (increase: 83% → 69%) of the time. SWE-smith: SFT on mean difficulty 2/4/6/8 gave 12.4/10.8/13.6/12.2 (no trend). Endless Terminals' o3-filtered tasks transferred little (TB2: 3B 0.0 → 2.2, 7B 2.2 → 3.4, 8B 1.1 → 6.7). AWM's gains shrink with hardness (BFCL simple +26.7, hard +1.1). RES and RST hardness may be access or endurance (B4.3).
- **Adopt.** Declare `monotone: true` only after a 3–5-setting ladder measured on the frontier audit sample (P1, Reasoning Gym card); re-measure every variant per student (P0); reject "harder" variants whose signature does not change on the knob moved; RES operators become App-level `obfuscation`/`observability`/`linkage` transforms with a replay-and-diff validity gate and a no-bypass check (P1).

### 6.4 Ladders and easing

- **Hints on, then off (Environment Tuning).** 400 BFCL tasks, Qwen2.5-7B: base 7.00; direct GRPO 17.42; + syntax stage 15.50; + Base split with augmented errors 25.83; + all splits 32.00; + augmentation off 36.92 (switching the scaffold off still added 4.9). Binary reward left Missing Parameter/Function near 0 in Stage 3 where progress reward did not (figure only). Single-stage RL collapsed after about 70 steps.
- **Other ladders.** SETA "decrease" (fewer steps, pre-seeded scaffolding, reduced scope, no hints, ≥ 2 steps); Agent-Reflex late starts at s_{t\*}; Synatra prescribed history lengths (start offset); ClawLoop raises difficulty at most one level per iteration (its controller overshot at k = 2: admission 45.8%, trajectories doubled); EAU allows difficulty increases only when recent training success ≥ 0.80.
- **Dense reward for p ≈ 0.** Tev PPRS: R = 1[pass] + λ·Φ(s_T), λ = 0.25, per-turn increments, per-turn-index group baseline; any pass outranks any failure; the shaped objective can prefer a less successful policy only if the pass-rate gap is < λ(1 − p₁)/(1 + λ) ≤ 0.2. 9B: PPRS 0.29 S@1 / 0.11 P@5 vs sparse 0.25 / 0.06; at 27B terminal-only reward **could not complete one optimiser step**. InfiniteWeb: 767 of 4,000 tasks discriminative under dense checkpoints vs 174 binary (4.4×). SETA partial reward raised groups with variance 49% → 86%. CompoWorld weights each criterion by w_j = 0.01 + (1 − p_j) so R = 1 only when all pass (eval about 0.76 vs 0.67 uniform at step 140).
- **Adopt.** `simplify(knob="hints")` is a World-side error overlay, replayed with hints on and off to prove identical end states and grades, with G9 re-run under hints (P1). PPRS is the candidate reward where the 35B's p < 0.05 (P1); keep the binary all-pass as the keep-rule metric.

### 6.5 Controllers and pool lifecycle

- **SETA-Evol policy:** r > 0.5 → increase (more steps, tighter constraints, larger scale, failure handling; not "more of the same", each variant harder along a different dimension); 0 < r ≤ 0.5 → context shift at equal structure (nginx → apache2); r = 0 with partial passes → decrease. 505 decrease, 143 increase, 553 change-context pairs.
- **RODS lifecycle:** variants enter at the next epoch, ≤ 20% of the active pool; burn-in eviction if the first observation < 0.20; drift eviction to > 0.95 or < 0.20 after n_min observations; prune by variance beyond P_max = 400; originals never retired; append-only logs replayed after a crash. Ablations vs 56.00 on BFCL (Qwen3-4B): random seed selection −4.75, binary detection −3.25, no coherence rewrite −5.13, no retirement −3.38, static pool −2.88.
- **Adopt.** SETA's thresholds as the controller's default per student, plus the variant-must-move check (P1); RODS's staged injection, eviction rules and originals-never-retired as archive rules keyed per student (P0 per the RODS card).

**Minor tips (B6).** Keep a real KL in long multi-turn GRPO (Environment Tuning: β = 0 and 0.001 collapsed entropy, β = 0.1 kept improving); use γ < 1 when speed is part of the skill instead of a hidden turn cap (GEM: γ = 0.9 found binary search); plain REINFORCE on 0/1 never learned tool calls and fixed ±1 caused overuse on HotpotQA (GEM); cap trajectories per instance at 2–3 for any SFT use (SWE-Gym cap 2 best: 9.7 vs 9.3 uncapped).

---

## B7. Self-improvement loops, their guards, and the statistics they need

### 7.1 Loops in the batch and what they lacked

| Loop | Signal → change | Result | Missing guard |
|---|---|---|---|
| Agent-World arena | Diagnosis agent reads failure traces, per-tool success counts, validator feedback → ranked weak envs + generation guidelines → targeted tasks, DB complexification | 14B τ²/BFCL/MCP-Mark-Postgres 60.2/52.4/29.5 → 63.5/54.9/36.3 → 65.4/55.8/38.1 over 2 rounds | No equal-budget random control; the diagnoser sees validator internals; arena reused for training |
| ClawLoop | Failed held-out trajectories → (failure point, root cause, missing capability) → clustered per skill → next synthesis | Qwen3.6-27B SkillsBench 46.3 → 57.2; saturates around K = 3 | Same model writes tasks, tests and diagnoses; held-out split from the same pool |
| AutoEnvScaling | Assignment = parent env, pass rate, modal failure, trajectories → simpler or harder variant | Matched steps TB2.1 26.7 → 38.9/39.3; without solver trajectories 35.3/32.6 (feedback worth 3.6–6.7 points) | Rubric review by the proposer's own model |
| EAU meta-agent | Reads training logs and rollouts → knob writes to simulator, rewards, episode length | 15 knob writes, 2 measured harms, 3 retracted attributions | Moving objective mid-run; up to 5 restarts then selection on training metrics; no matched compute |
| TaskCraft prompt evolution | Exemplars chosen by pass rate under the same LLM judges | Atomic 54.9% → 68.1% | Circular: raises yield under the judge, not validity |
| Kimi K2 critic | Verifiable rollouts recalibrate a rubric critic | No ablation | Updates a judge from student behaviour |
| SWE-Gym self-improvement | Fine-tune on own successes | Lite 15.3% → **8.7%** | Accepted on the student's own successes |

- **A shared proposer must keep learning.** AutoEnvScaling: with solver-only updates the proposer's valid rate fell 86.8% → 22.4% by step 160 and training collapsed; with proposer training it dipped to 79.6% then reached 94.7%.
- **Adopt.** The `Assignment{parent_ref, student_id, p_hat, failure_cluster, trace_refs}` object is F3's input and the missing consumer of `factory/failures.py` (P0). The swarm proposer improves by harness optimisation (memory, skills), not weight training (AutoEnvScaling card). Every targeted batch is compared against an equal-budget untargeted batch; the diagnoser is a different family, sees verdicts and short reasons only, and its output goes through F6 (P1).

### 7.2 Guard mechanisms found in the cards

- Admission checks outside the builder's sandbox; the builder may edit memory and tools, never checks (AutoEnvScaling).
- Declared knob ranges, per-write change caps (entropy ≤ 0.004 per write), difficulty increases only at recent success ≥ 0.80, revert paths, and an experience pool that records retractions (EAU). The card notes range checks do not guarantee behavioural correctness.
- Repairs bounded (≤ 2), restricted to a six-file allowlist, conditioned on the contract, and forbidden to weaken semantic checks, remove shortcut protection or expose private requirements; caps relaxed only on a logged schedule; the lineage gate (RST).
- Revisions bounded and versioned, findings with evidence references routed to the producing role (Interlock).
- Re-measure every variant, since declared direction fails 23–40% of the time (SETA).
- Originals never retired; generated share capped (RODS 20%; DreamGym λ); KL cap on a trained generator (GenEnv).
- Gate on real side attribution: the LLM diagnoses, execution certifies (GRADUS).
- **Adopt.** These extend §3.8's table: edit budgets and file allowlists for repairs (P0), logged cap-relaxation schedules (P1), a masking alarm (success rising while the target behaviour falls) as a loop health check (P1).

### 7.3 Repair rounds: attribute the side, keep the hard tasks

- **Yields with repair.** APIGen-MT feedback summariser 28% → 70% blueprint yield (3 rounds retail, 5 airline). GRADUS validity loop (telecom/CAR-disambiguation): none 0.65/0.40 → + parallel branches 0.68/0.43 → + reflection 0.71/0.55 → + task recovery 0.75/0.60. RODS: blind retry instead of the feedback loop −2.13; its repair patches the start-state config and grows a function blocklist (≤ 3 attempts).
- **Side attribution (GRADUS).** Replay reference and demonstration from the same start; accept only on outcome agreement plus programmatic policy checks; otherwise a diagnostic LLM labels demonstration error (search more) or task/reference error (return to initialisation). Recovered tasks are disproportionately hard (B5.6).
- **Adopt (P0).** F6 repair rounds attribute each failure to a side (task, oracle, verifier, World, simulator) and repair that side; every start-state patch re-runs G1/G2/G8; report yield with and without repair per stage.

### 7.4 Judge calibration and statistics

| Practice | Source | Numbers |
|---|---|---|
| Repeated judgments: self-consistency, pairwise agreement, κ, reward flip rate | AWM | GPT-5.1: 90.8%, 95.5%, κ 0.891, flip 9.2% |
| Rubric canary | VitaBench | κ 0.018 without rubrics |
| Balanced per-verifier audit, precision and recall | WebUniverse (one author, 100 accepted + 100 rejected) | Browser scripts 98.0/91.6; GUI agent 97.0/94.2; VLM judge 87.0/83.7 |
| Detector audit | RHB | 94% agreement on 50 flagged; 0 FN on 50 unflagged |
| Human audit of false positives/negatives | SCA | 200 trajectories, 8 human-hours |
| Judge chosen by correlation per cost | TOUCAN | 50 human labels |
| Runs per stochastic multi-turn task | VitaBench | Resampling 32 runs: k = 4 cuts MSE of the mean by 77.5% vs k = 1; k = 8 adds little |
| Minimum detectable effect | Data World | MDE = 2.802·s_Δ/√n, about 9–10 points for its cause/consequence test |
| Benchmark protocol | SWE-rebench | Fixed scaffold, 5 runs, mean ± SEM, pass@5, per-model date flags |
| Pass@k and Best@k from 100 random size-k subsamples | SWE-Gym | — |
| Exploit-rate statistics | RHB | Clopper–Pearson CIs, Fisher tests |
| Noise floors | EAU (0.228 on one metric); ACE-Router (std over runs is not uncertainty over ≈ 50 tasks); Environment Tuning (step sizes of 2.5/5 pp imply 20–40 tasks per τ² domain) | — |
| Annotator agreement for refreshed exams | GRADUS (scout ≤ 5/8, two annotators, 87.5–100% judged correct, AC1 0.79–1.00) | — |

- **Adopt.** WebUniverse's balanced 100/100 audit is the acceptance report for any new gate or judge (P1); VitaBench's κ 0.018 and R2E's confident-but-wrong summary are anchor-suite canaries (P0); k = 4 per measurement for simulated-user tasks; MDE before any "no difference" claim.

**Minor tips (B7).** Keep an experience pool of interventions with retractions (EAU); stratify a diagnostic arena by taxonomy category with fresh tasks each round (Agent-World); separate advantage baselines per (task type, role) in multi-role self-play (AZR, AutoEnvScaling).

---

## B8. Pipeline architectures, costs, yields and scale

### 8.1 Cost and yield per pipeline

| Pipeline | Unit cost | Yield | Scale |
|---|---|---|---|
| ScaleEnv | ~546K tokens per domain (379K prompt + 167K completion); ~93.2K per task | Not reported per stage | 16 domains, 2,560 tasks |
| AWM | $57.09 per 100 envs with 10 tasks each (≈ $0.57 per env); judge ≈ $1.80 per RL step | >85% first attempt; 1.13 iterations; env error ≈ 4% during RL | 1,000 envs, 35,062 tools, 10,000 tasks |
| EnvScaler | $1.02 per env ($0.77 of it the 100-round test); $0.064 per scenario | 191/266 envs (71.8%) | ≈ 7K scenarios |
| AutoEnv | $4.12 per env ($4.33 automated, $3.48 supervised) | 65% overall (60 / 80); then 36 of 65 hand-picked | 358 levels |
| LOGIGEN | ≈ ¥5.5 (≈ $0.75) per unit; per env or per task unstated | Architect 68.5%, Set Designer 72.8%, Explorer 99.3%, distillation 74.2% | >2,000 envs, >20,000 tasks |
| SWE-smith | 2.32¢ per candidate; $1,360 total | 50.1% (100,074 → 50,137); expert solve 36% | 128 repos |
| SWE-rebench | n/r | ~450K → ~150K → ~21K valid (~14%); installs in 31% of repos | 21,336 tasks, 3,468 repos |
| SWE-Factory | $0.037–0.056 and 23–30 min per issue | 42.0–50.2% valid F2P | 671 issues, 4 languages |
| SCA | challenge generation 8–32 (units of 8×A100-hours) | 5.2% survive | 800 tasks per setting |
| AutoEnvScaling | $0.02 (DeepSeek-V4-Flash) to $1.18 (Opus 5) per valid env | 73.0–94.0% with reference execution | 256-task active pool |
| VHD-Play | $0.01–0.03 per env; ≈ 900 in + 6,300 out tokens | 65.6% first try; 1.41 attempts | 3,300 envs |
| RST | ≈ $50 per 1,000 accepted | 498–572 per 1,000 seed attempts per round | 37,484 tasks |
| Tev | median 30 s CPU per env, no model calls (generator authoring n/r) | — | 1,170 envs |
| InfiniteWeb | $1.93 per site, median 20 min | 1.5% unfixed | 600 sites in ≈ 125 min |
| EnvFactory | 192K + 31K tokens, ≈ 3 min per env | 92.9% | 85 envs |
| EnvCraft | n/r | envs 139/324 (42.9%); tasks 19,777/35,464 (55.8%) | — |
| ClawLoop | n/r | ~89% agent-free pass; 45.8–65.6% in band | 1,024 candidates per iteration |
| Data World | n/r | 1,087/1,302 worlds (83.5%) | 2,235 tasks |
| EnvLunch | 1,600 tasks in 20 min CPU | — | warm-up only |
| Synatra | $0.031 per kept sample | — | ≈ 100k demos |
| RODS | 448 GPU-h training + 448 GPU-h synthesis (doubles footprint) | — | ≤ 400 seeds + 400 variants |
| τ² evaluation | $0.086 agent + $0.059 user per task-run (gpt-4.1); one trial of all domains ≈ $40 | — | — |

- **Infra notes.** One image per repo (SWE-smith, about 500× less storage); 1,024 isolated MCP servers per RL step with prefetch (AWM); 10,000+ concurrent sandboxes and partial rollouts for long tails (Kimi K2); environment execution about 1/7 of RL wall time (AutoForge); one connection per session is a throughput bottleneck for stateful MCP (EnvFactory).
- **Adopt.** These populate §3.10's cost formula as priors; VHD-Play's and AutoEnvScaling's per-valid-env costs and RST's per-task cost are the cheapest published rates for executed construction.

### 8.2 Training results on our student family (Qwen3.x-30/35B-A3B) and the 397B

| Paper | Student, regime | Result |
|---|---|---|
| CompoWorld | Qwen3.6-35B-A3B, SFT 3K (DeepSeek-V4-Flash) + GRPO 1K tasks, synthesis by GLM-5.3 | Average +9.17 on eight benchmarks; AutomationBench 10.33 → 32.33; VitaBench +10.50; RL added only 1.2 over SFT and lowered τ³-Banking. Qwen3.5-397B-A17B baseline 5.50 on AutomationBench |
| AutoEnvScaling | Qwen3.5-35B-A3B RL with GPT-5.6-sol or Opus 5 proposers; Qwen3.6-35B-A3B as self-proposer | TB2.1 avg@5 26.7 → 38.9 / 39.3; self-proposer 41.1 → 53.3, TB4.0 1.6 → 3.2; HiL-Bench 6.8% → 33.9% |
| Data World | Qwen3.6-35B-A3B RL-zero (GSPO), rubric judge = the base model itself | DataWorld-Bench 37.64 → 44.70; DAComp +14.28; DABstep-Hard +20.90; trajectories much longer (36.5 → 102.8 rounds) |
| VHD-Play | Qwen3.6-35B-A3B GRPO, 34 steps, 2,176 envs | F/I/A 0.962/0.231/0.204 → 0.992/0.875/0.815; BFCL V4 61.25 → 64.08; matched EnvScaler control 58.56 (below base) |
| WorkForge | Qwen3.5-35B-A3B-Base, SFT from GLM-5.3 | GDPval 45.5 → 73.6; ALE 11.4 → 28.7; Qwen3.5-397B GDPval 79.7 |
| WebUniverse | Qwen3.5-35B-A3B SFT | WebGen-Bench 34.5 → 63.9; matched 15K other-mix SFT 31.5 (below base) |
| Interlock | Qwen3.6-35B-A3B rejection-sampling SFT on own accepted rollouts | AutomationBench 9.00 → 13.50; ClawGym-SFT made it worse (5.17) |
| RES | Qwen3.5-35B-A3B SFT on RES vs original ClawGym trajectories | Claw 64.76 → 77.14; Terminal-Bench 2.1 23.60 → 38.20 |
| GRADUS | Qwen3-30B-A3B, SFT then GRPO, Opus 4.5 proposer | τ² airline/retail/telecom 0.63/0.77/0.38 → 0.80/0.89/0.97; matched 8B comparison: model-prior 0.45, graph 0.51, trajectory-first 0.59, GRADUS 0.76 |
| Trajectory2Gym | Qwen3 up to 30B-A3B SFT, in-environment train splits | τ2 Telecom (30B-A3B) 12.50 → 95.00; BFCL 77.58 → 80.93 |
| AutoForge | Qwen3-Thinking-30B-A3B, ERPO | τ² Telecom 26.3 → 76.3; 10 envs, 1,078 tasks targeting the eval domains |
| AgentScaler | Qwen3-Thinking-30B-A3B SFT only | τ² Telecom 26.3 → 55.3; ACEBench-en 67.2 → 75.7 |

- **Read with care.** Several of these train inside the evaluation environments or their domains (GRADUS, Trajectory2Gym, AutoForge, AgentScaler); several are SFT distillation from a stronger teacher (WorkForge, WebUniverse, CompoWorld's SFT stage); Data World self-judges during RL.
- **Other headline results (smaller students).** Agent-World-8B τ² 61.8 (cold start 40K trajectories from an in-house model); LOGIGEN 8B τ² 30.8 → 71.8 (RL ≈ 0 over full SFT at 8B); ScaleEnv 8B Retail 38.4 → 50.9; AWM 8B BFCL 53.83 → 65.94; SynthAPI Big 27B AppWorld Test-C 81.5 (SFT); RODS 4B BFCL V3 22.13 → 56.00; SETA Qwen3-8B TB2.0 3.1 → 10.7 ± 1.3; RST PPO 27B TB2 49.44; Tev-27B ALE-CM P@5 0.17 → 0.28; ClawLoop 27B SkillsBench 46.3 → 57.2; EnvLunch +9.3/+9.6 (9B/4B) on six data-analysis benchmarks; Agent-Reflex 8B five tool-use benchmarks 40.2 → 58.6.

---

## B9. Corrections and contradictions

### 9.1 Errors and inconsistencies inside papers (as flagged by the cards)

- **SWE-smith:** PR Mirror yield 33.8% in the table vs 13.18% in App. C.1 prose (unresolved).
- **Agent-World:** 14B base τ² 45.3 in the text vs 60.2 in Table 2.
- **DreamGym:** abstract and §5.2 say agents exceed 30% success, but absolute WebArena rates are 9–15% (likely a relative gain).
- **LOGIGEN:** module costs not stated as per environment or per task; difficulty tags L1–L3 sum to ≈ 10K of > 20K tasks.
- **TaskCraft:** about 36,000 tasks (abstract) vs about 32k (contributions); direct-prompting ablation pass rate 43.0% vs 54.9% atomic pass rate, not reconciled.
- **R2E-Gym:** decontaminated subset 4,578 (text) vs 4,538 (appendix).
- **AutoEnv:** Claude-4-Sonnet 40.67 vs 40.86; environment names differ between Tables A6 and A7; one instruction file describes a different game; a 138.84% reward called "conservativeness".
- **τ²:** Table 2 caption says no critical telecom simulator errors, the table lists 3 critical (6%).
- **SETA:** all-pass bonus 0.2 (text) vs +0.5 (appendix table).
- **RST:** conclusion's 11.82% TB2 relative gain vs 20.0% in abstract and table.
- **RHB:** second most common exploit category is leakage 24% (main text) vs tampering 19% (appendix).
- **SWE-Factory:** count fractions in the extracted per-language table are garbled (e.g. "181/186" beside 43.5%); use percentages only.
- **AgentScaler:** ambiguous whether exact-match filtering applies to all trajectories or only read-only ones; τ and the embedding model unreported.
- **FTRL:** a code-verify filter appears only in App. 11, with no rule or yield.
- **Simia:** claims no testbed data is needed, yet the OfficeBench prompt includes `{testbed_text}` and the APIGen-MT prompt the seed's reference conversation.
- **Environment Tuning:** checkpoint and stage choice used 100 samples from the held-in test set; some gold calls ('CRH', 'PHV') differ from environment returns ('CWH', 'PHW').
- **Reasoning Gym:** several "hard" configs move a parameter the easy way; the 0.10 `failure_threshold` is undocumented in the text.
- **SWE-Gym vs SWE-smith:** "diversity is not the bottleneck" (11 repos) conflicts with SWE-smith's log-linear repo scaling at 100 repos.
- **Baseline fairness:** DreamGym's WebArena real-RL baseline was starved; many comparisons mix user simulators (LOGIGEN uses DeepSeek-V3.2, other rows GPT-4.1) or training regimes (SynthAPI baselines).
- **Naming:** our notes cite 2601.08276 as "ToolACE-MCP"; its v2 title is ACE-Router.

### 9.2 Corrections and additions to `framework-plan.md`

1. **§3.4 and §3.10, AutoEnv "65% verified".** Read it as an F1 pass rate before G3/G5/G12; it excludes the hand selection of 36 of 65 and human review time, and "verified" admitted a 138.84% reward and a mismatched instruction file.
2. **§3.10, AWM ">85% first attempt".** First-attempt stage success is not App quality: AWM accepts at < 10% failing tables, and its own audit found 74–83% of envs with ≥ 1 bug and 23.7–28.2% of training tasks never solved.
3. **§3.4 / §3.9, EnvScaler as the blind-tester precedent.** Its frontend tester writes no expected outcome and its backend checker judges code against the observed state diff with a 15% failure tolerance; the plan's frozen-digest rule is stricter and must be stated as Crucible's addition, not EnvScaler's practice.
4. **§3.6, Endless Terminals "2.2% → 3.4%".** That is the Qwen2.5-7B row only; the card gives 0.0 → 2.2 (Llama-3.2-3B), 2.2 → 3.4 (Qwen2.5-7B) and 1.1 → 6.7 (Qwen3-8B).
5. **§3.6, GenEnv "centres at 0.5".** Its band is measured on unverified generator tasks; its own ranking bound implies about 490 rollouts to separate tasks 0.2 apart in band distance, so per-task placement at k = 8 needs sequential tests and pooled cell estimates.
6. **§3.6 sequential k.** A published variant exists (AutoEnvScaling: 4 → 8 near band edges, mean ∈ [0.25, 0.75], std ≥ 0.1); cite it beside the proposal.
7. **§3.1 gate list.** Add G15 exploit probes, G16 reward causality, G17 solver agreement (two oracles, different families, same fingerprint), a recoverability variant of G8, and progress-validity checks before Φ is admitted; strengthen G6 from prefix checks to dropping each step; G10 with a simulated user needs a cost plan (τ²: 60 × $0.145 per task).
8. **§3.3 check kinds and operators.** Add realisation checks (Interlock) and the negative classes partial propagation, constant answer (EnvLunch), stale output (Tev) and claim-success. Do not adopt EnvCraft's "every chain ends in a write": read-only, answer and infeasible tasks are required types.
9. **§3.5 coverage.** MAP-Elites fill alone misses template locking; add Tev's strength-2 covering array and GRADUS's (policy, outcome) ledger in the proposer prompt, and RST's nearest-neighbour similarity tail as a drift alarm.
10. **§3.5 family homogeneity.** The plan's caveat gains a measured case: AutoEnv's generator made the requested semantic axis easier, so cell difficulty must be measured, not assumed.
11. **§3.8 guards.** Add edit budgets with file allowlists for repairs and the rule that repairs may not weaken checks (RST); the masking alarm (EAU); equal-budget untargeted controls for every targeted batch (Agent-World gap); and, for any shared-policy proposer, the AutoEnvScaling collapse (86.8% → 22.4%) as the reason in-training proposer loops stay P2.
12. **§3.10 families × instances.** SynthAPI shows at about 1,000 tasks fewer Apps can win for SFT; the plan's "many families, few instances" stays a target to test for RL per student, with S (seeds per task) as a cheap multiplier.
