# 01 ScaleEnv: a deep read, a critique, the related work, and what Crucible should take

*Written 2026-10-29. Not committed.*

**Sources**
- The paper: `crucible/docs/research/papers/2602.06820.pdf`, all 27 pages read. Pages 1–10 are the body and references; pages 11–27 are appendices A–E.
- Related work: each related paper was checked on its arxiv.org abs page and HTML full text by three research passes on 2026-10-29.
- The 10 PDFs downloaded (list in §5) were also confirmed by their title pages.

**Legend**
- **[paper]**: the number is stated in the paper.
- **[derived]**: I computed it from the paper's numbers.
- **[inferred]**: my reading, not stated in the paper.
- **UNVERIFIED**: not confirmed.
- All summaries are in my own words.

---

## 0. TL;DR

**What ScaleEnv is.** ScaleEnv (Tu, Hao, Yang et al.; Meituan LongCat with Nanjing University, HIT Shenzhen and ECNU; arXiv 2602.06820, v1 6 Feb 2026) builds tool-use RL environments from a domain keyword:
- An LLM writes tool schemas, then a DB schema derived from them, then Pydantic/Python code for both.
- Each tool is checked by executing it against LLM-written test cases on matched DB instances, with a debug agent fixing failures.
- An LLM links tools into a dependency graph.
- Tasks are built solution-first. A sampled tool chain, written as code, is executed on a generated start state to produce the gold end state, and the user instruction is written from that chain.
- The environment around each task is then "snowballed": a dependency-closed BFS over the graph, plus further chains admitted by an LLM gate, until at least 20 tools are reachable.
- Reward is rule-based: the final DB is compared with the gold DB under three per-column match policies.

**How it is trained and scored.**
- Qwen3-8B and 32B are trained with GRPO directly ("Zero RL", no SFT) on 16 synthetic domains and 2,560 tasks, with Qwen2.5-72B-Instruct as the simulated user.
- On the out-of-domain τ²-Bench, Qwen3-SE-8B goes from 30.1 to 38.5 Avg@4 [derived].
- Pass@4 rises monotonically with the number of training domains, N = 2, 4, 8, 16, at a fixed 1,024 tasks.
- The paper reached ICML 2026 (PMLR vol. 306); no code or data has been released.

**Lessons for Crucible.**
1. **Domain diversity, not task count, drives transfer.** ScaleEnv, Apple's SynthAPI (2610.37648) and AWM (2602.10090) all point this way, so a 500–1000-task pool should be many families with few tasks each.
2. **Execution-verified tool code is worth about 6 τ² points over the same pipeline without execution checks.**
3. **State-diff rewards need per-column policies plus checks it lacks.** ScaleEnv has no answer check for read-only goals, no collateral-damage check, and no frozen clock.

Crucible already has most of the checking machinery and is stronger than ScaleEnv on difficulty, hacking and simulated-user fidelity. What it lacks is a generic **world synthesizer**: keyword to schema, code, tool tests, tool graph, and task chains. That is the one big item to adopt (§4).

---

## 1. ScaleEnv in full technical detail

### 1.1 Formalism (§3)
**Domain and environment.**
- A domain foundation B = ⟨Σ, 𝕋⟩: Σ is the set of DB schemas, which defines the valid state space; 𝕋 is the set of executable tools.
- An environment E = ⟨s₀, 𝕋⟩ adds a start DB filled to Σ.

**Task.** ψ = ⟨E, u, P_user⟩:
- u is the user's hidden intent.
- P_user is a profile with permissions, location and history.

**Episode model.** A POMDP:
- The state is (environment state, history, intent).
- Actions are either a natural-language reply or a tool call.
- A tool call updates the environment state and the history deterministically. A reply only adds the simulated user's answer to the history.
- The reward is R(s_T, u), computed on the final environment state only.

### 1.2 Domain foundation (§4.1.1–4.1.2, App. D)
1. **Top-down tool schema synthesis.** From a domain name such as "Job Seeking", an LLM designs the atomic tool set. Each tool gets:
   - a description, parameters, and an output structure;
   - logical **pre- and post-conditions**, e.g. `submit_application` requires a prior `upload_resume`.
   - Table 6 shows four Job-Seeking tools as an example.
2. **DB schema derived from the tools.** A "Database Agent" reads the tool definitions and works backwards to the tables they need; a submit tool implies an Application table and a Job table.
   - It de-duplicates and filters into one schema with integrity constraints: PKs, FKs, defaults, enum-like values (Table 7).
   - It also writes a **tool → tables mapping**, which later decides which DB instances each tool is tested on.
3. **DB code first, then tool code.** The DB is implemented first because tools depend on it.
   - Listing 1: each table is a Pydantic `BaseModel` with a `ThreadSafeBase` mixin and a `@with_instance_key("<pk>")` decorator. The DB class holds `Optional[Dict[pk, Row]]` per table, so state is a dictionary of tables, not SQL.
   - The LLM also writes DB test scripts against the integrity constraints. A debugger agent fixes the code from tracebacks until every test passes.
   - Listing 2: tools are methods on a class holding `self.db`. They validate arguments, raise `KeyError` or `ValueError` on bad input, mutate the tables, and return a dict.
4. **Scale per domain [paper].**
   - §5.1 says about 50 tools and 5–20 tables per domain.
   - Appendix B.1 says 25 tools (Online Learning) to more than 70 (Entertainment Media Query), and 5 tables (Job Seeking) to 22 (Agriculture). The two sections disagree slightly.

### 1.3 Roles in the multi-agent architecture
The paper names roles but publishes no prompts.

| Role | What it does |
|---|---|
| Schema LLM (Figure 1 calls it a "meta-agent") | Writes the tool schemas from the domain keyword |
| Database Agent | Writes the DB schema and the tool→table mapping |
| Code Agent | Implements the DB and tool code |
| Test Agent | Writes test cases and matched DB instances for each tool, in parallel with the code |
| Debug Agent | Reads error logs and fixes the tool code **or** the DB instance |
| Tool Dependency Agent | Judges every pair of tools and adds the edges |
| Chain generator | Writes the seed tool chain as executable code, from graph G and the DB schema |
| State-construction LLM | Writes s₀ and adds distractor rows |
| Instruction LLM | Writes P_user and u, grounded in the seed chain |
| Gating LLM (policy π) | Scores p ∈ [0,1] for admitting a new chain |
| "Oracle" | Qwen3-235B-A22B, best-of-16, estimates whether the remaining tools are feasible |
| User simulator | Qwen2.5-72B-Instruct during training |

The construction models were a mix of DeepSeek-V3.2, GLM-4.7, GPT-5.1 and Qwen3-32B. The paper does not say which model played which role.

### 1.4 Procedural testing (§4.1.2)
The Code Agent and the Test Agent work **in parallel**: the tests are not written from the code. The Test Agent writes unit tests and the matched DB instances. Each tool runs on its instances, and every case ends in one of three outcomes:
- **Success:** no error, and the state transition equals the expected DB state.
- **Anticipated rejection:** invalid input raises the exception the schema declared.
- **Unexpected failure:** any other runtime error or state mismatch. The Debug Agent then repairs the tool code or the DB instance, and the loop repeats until the procedural test passes.

Two consequences:
- **No failure or yield rates are reported.** There is no first-pass rate, no count of debug iterations, and no share of tools or domains abandoned. All the user asked for on failure rates is missing from the paper. For contrast:
  - EnvScaler reports 71.8% environment yield;
  - AWM reports more than 85% first-attempt success and 1.13 repair iterations on average.
- **The debugger may edit the DB instance as well as the code.** That makes it possible to "fix" a failing test by changing the expected state to match buggy code. The paper has no guard against this [inferred].

### 1.5 Tool dependency graph (§4.1.3, App. B.1)
- A Tool Dependency Agent judges each pair of verified tools on three axes:
  - **data flow:** an output feeds a parameter;
  - **pre- and post-conditions:** a logical prerequisite;
  - **state dependency:** the two tools share a table.
- A directed edge is a causal link. The result is one graph G per domain.
- Figure 1 draws two edge styles ("=" and "//"), but the text never defines edge types.
- Figure 5 plots graph density per domain, from 0 to about 0.2. Job Seeking and Knowledge Management are densest; Agriculture is sparsest.
- Pairwise LLM judging costs O(n²) calls, about 2,500 pairs for 50 tools [derived]. The paper does not discuss this.

### 1.6 Task initialization (§4.2.1)
1. **Seed chain sampling.**
   - An LLM, given G and the schema, writes a seed chain C₁ = (a₁…a_k) **as executable code**, so outputs feed later arguments programmatically.
   - Writing it as code is the paper's fix for arguments losing their link to the tool outputs they come from.
2. **Start state with distractors.**
   - An LLM writes s₀ so that C₁ succeeds on it (entity consistency), and C₁ is **executed** to confirm.
   - Then distractor rows are added. They satisfy the schema but are orthogonal to the gold path, so the agent must filter information.
   - Distractor density scales with a "predefined task complexity", which the paper never defines.
   - **Ground truth:** the gold end state is the result of executing C₁ on s₀.
3. **Instruction synthesis.**
   - An LLM writes P_user and the hidden intent u, both grounded strictly in C₁.
   - The reward criteria come from the gold state.
   - Figure 2 also lists "evaluation rubrics" as an output of task construction. The text never says how a rubric is used, and the reward described is rule-based only.

### 1.7 The snowball: controlled environment expansion (§4.2.2)
**Why expand.** An environment holding only C₁'s tools and rows lets the policy memorise sparse trajectories. ScaleEnv therefore grows the environment around each task to give **interaction completeness**: every valid tool call returns a meaningful observation.

1. **Dependency-aware BFS.**
   - Start with H₁ = C₁.
   - BFS over G. A tool v is added only if its input dependencies can be met by tools already in H₁, which avoids dead ends.
   - Each new tool is **executed** with arguments drawn from H₁, and the environment is refined if anything errors.
   - The result is K(C₁).
2. **LLM-gated chain expansion.**
   - After iteration n, the candidates are Dₙ = G \ Hₙ.
   - A gating LLM, prompted with |Dₙ|, c(Hₙ) and g(Dₙ), outputs p ∈ [0,1]. If p ≥ τ, it samples C_{n+1} ⊆ Dₙ, expands it to K(C_{n+1}), executes it, refines, and merges it into Hₙ.
   - **Structural complexity:** c(Hₙ) = (|V| + λ|E|)/S_sat, with λ = 0.5 and S_sat = 50.
   - **Feasibility:** g(Dₙ) is the success rate of an oracle (Qwen3-235B-A22B, best-of-16) at finding executable chains in Dₙ.
   - **The threshold τ is never given.**
3. **Floor.**
   - If |Hₙ| < 20, random valid auxiliary chains are merged in. The "Random Subgraph K(t)" box in Figure 2 is this step.
   - The final environment exposes Hₙ, at least 20 tools, possibly a subset of the domain's full graph [inferred from Figure 2: "Scaled Environment E = K(t) ∪ Hₙ"].

**What the expansion changes and what it does not.** The task's reward stays tied to C₁'s gold state. Expansion adds tools and data for exploration, not new goals [inferred: §4.2.1 ties u and R to C₁, and §4.2.2 never redefines R]. Figure 2 labels a "Task-specific new Tool Chain C_{n+1}" from a "Strong Solver", so it is possible that later chains also feed the task. The text is ambiguous here.

### 1.8 Distractors, user simulation, hidden goals
- **Distractors** come in two forms: schema-valid rows orthogonal to the gold path (§1.6), and extra tools from the expansion. There is no ablation of either.
- **User simulator.** Qwen2.5-72B-Instruct is initialised with the intent u from ψ and plays the user over many turns. The agent sees only the user's replies, so u is hidden.
- **What is not specified:**
  - the user prompt, temperature and stop rule;
  - whether the simulator knows P_user;
  - any check on simulator fidelity.
- **Behaviour in the Appendix E trajectory:**
  - The simulated user chains several requests across about 50 turns.
  - It also asks for something out of scope (study resources), which the agent declines.
  - Several turns are read-only analytics (conversion rates). The state-based reward cannot grade these (§2).

### 1.9 The verifiers (§4.1.1 "Reward Specification", §5.4)
**Final-state reward.** R compares the agent's final DB with the gold DB, column by column, under three policies:
1. **Exempt fields:** dynamically generated IDs and optional columns that do not affect success.
2. **Hard constraints:** exact character or numeric equality, for timestamps, prices and quantities.
3. **Semantic alignment:** fuzzy matching for descriptive text. **The method is not stated**: embedding, LLM or string similarity.

**Not stated:**
- whether the reward is binary or a partial credit;
- whether the whole DB is compared, which would catch collateral writes, or only the touched rows;
- who assigns columns to policies, an LLM or a rule;
- how timestamps written by `datetime.now()` inside tools (Listing 2: `deleted_at`, `updated_at`) are made deterministic.

**Executable action verification ("EV").** In the paper this is the build-time discipline, not a runtime reward:
- every tool, chain and expansion is actually executed;
- the environment is patched until the seed chain is solvable.
- §5.4 ablates exactly this; without it "tools are never subjected to actual parameter-driven execution".

There is **no** check on the agent's intermediate actions, its final answer text, or its policy compliance.

### 1.10 Solvability guarantees
**What exists:**
1. A solution path exists by construction: the gold state *is* the result of executing C₁ on s₀.
2. Every tool added in expansion has been executed from reachable arguments.
3. The oracle's best-of-16 feasibility score gates later expansion.

**What is not guaranteed:**
- **Communicability:** that the hidden intent u, as voiced by the simulated user, carries every argument value C₁ used. Examples are exact timestamps (a hard constraint) and the IDs the user must give.
- **Uniqueness:** that u has a single valid end state. Ambiguous intents give false negatives.
- **That an agent can actually solve the task:** no agent-solve or ladder check is run on the finished task.

### 1.11 RL setup (§5.1, App. C)
**Recipe.**
- **"Zero RL"** means RL directly from Qwen3-8B and Qwen3-32B with no SFT warm-up. These are released post-trained checkpoints, not base models. Thinking mode is not stated.
- **Algorithm:** standard GRPO (Eq. 1–2):
  - a group of G trajectories per task;
  - a token-level clipped ratio averaged over trajectory length L_i;
  - a KL penalty to the reference;
  - group-normalised advantage (r − μ)/σ.

**Stated hyperparameters [paper]:**
- learning rate 1e-6;
- 48 training steps;
- rollout batch 1,024 for 8B and 2,048 for 32B;
- user simulator Qwen2.5-72B-Instruct.

**Not stated:** group size G, ε, β (KL), max turns, max context, temperature, hardware, wall-clock.
- If "rollout batch" means trajectories per step, 8B saw about 49K trajectories in total [derived, interpretation uncertain].

**Data.**
- 16 domains and 2,560 tasks (Table 8).
- Tasks per domain are unbalanced, in powers of two: 512 each for wedding planning and knowledge management; 256 each for job seeking and healthcare telemedicine; 128 each for four domains; 64 each for eight domains.
- The scaling and stability studies use 1,024 tasks.

### 1.12 Results (Table 1, Avg@4 [inferred: Qwen3-8B's row equals the "Qwen3-8B" row of Table 3, which is labelled Avg@4])

| Model | τ² Retail | τ² Airline | τ² Telecom | Vita Cross | Delivery | In-store | OTA |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | 38.4 | 30.5 | 21.5 | 1.5 | 18.3 | 14.8 | 4.5 |
| **Qwen3-SE-8B** | 50.9 (+12.5) | 37.5 (+7.0) | 27.2 (+5.7) | 3.0 (+1.5) | 26.3 (+8.0) | 23.8 (+9.0) | 7.0 (+2.5) |
| Qwen3-32B | 59.5 | 48.0 | 27.2 | 5.3 | 27.0 | 22.5 | 4.5 |
| **Qwen3-SE-32B** | 63.6 (+4.1) | 48.0 (+0.0) | 30.9 (+3.7) | 10.8 (+5.5) | 31.3 (+4.3) | 34.5 (+12.0) | 12.5 (+8.0) |

**Reference rows** (the evaluation setup is not stated, so these may come from different user simulators):

| Model | Retail | Airline | Telecom |
|---|---|---|---|
| Qwen3-235B-A22B-2507 | 71.9 | 58.6 | 47.3 |
| Kimi-K2-0905 | 70.6 | 56.5 | 65.8 |
| GPT-OSS-120B | 57.0 | 38.0 | 45.6 |
| Seed-OSS-36B | 68.4 | 52.0 | 41.2 |
| xLAM-2-32B | 55.3 | 52.0 | 16.7 |

**Averages [derived]:**

| Model | τ² Avg@4 | VitaBench Avg@4 |
|---|---|---|
| 8B base → SE | 30.1 → 38.5 | 9.8 → 15.0 |
| 32B base → SE | 44.9 → 47.5 | 14.8 → 22.3 |

Gains shrink sharply at 32B on τ², and Airline does not move.

**Table 2: VitaBench Pass@4.**

| Model | Cross | Delivery | In-store | OTA | Average |
|---|---|---|---|---|---|
| 8B base | 6 | 43 | 44 | 15 | 27.0 |
| SE-8B | 12 | 56 | 57 | 18 | 35.8 |
| 32B base | 15 | 57 | 58 | 14 | 36.0 |
| SE-32B | 29 | 65 | 66 | 27 | 46.8 |

The paper reads the Pass@4 gains as a higher ceiling, not just more stable outputs.

**OOD claim (App. A).** A t-SNE of tool-description embeddings shows the 16 training domains as clusters apart from the τ² and Vita tools. This tests the semantics of tool descriptions only, not overlap in behaviours or policies.

### 1.13 Ablations (§5.4)

**Table 3: without executable verification, 8B, Avg@4.**

| | Retail | Airline | Telecom | Average [derived] |
|---|---|---|---|---|
| Base | 38.4 | 30.5 | 21.5 | 30.1 |
| Without EV | 42.3 | 30.0 | 25.2 | 32.5 |
| Full | 50.9 | 37.5 | 27.2 | 38.5 |

Most of the gain needs execution-checked environments.

**Table 4: reward, τ² average** (8B [derived: its Avg@4 of 38.5 equals the Table 1 average]).

| Reward | Avg@4 | Pass@4 | Pass^4 |
|---|---|---|---|
| LLM-as-judge | 36.5 | 58.8 | 14.6 |
| Rule-based | 38.5 | 62.9 | 15.0 |

The judge model and its prompt are not given. The gap is small, and there are no seeds or error bars.

**Table 5: two domain sets, 4 domains each, 1,024 tasks, 8B, VitaBench Avg@4.**

| | Average |
|---|---|
| Base | 9.8 |
| Set A (wedding, knowledge management, job seeking, telemedicine) | 13.8 |
| Set B (logistics, job seeking, email, pet care) | 13.3 |

**The sets are called "non-overlapping" but both contain job seeking.** The table also calls the method "EnvZero", apparently an earlier name.

**Not ablated:** distractors, graph expansion versus C₁-only environments, the 20-tool floor, the gating LLM, the choice of user simulator, the number of tools, SFT warm-up.

### 1.14 The environment-scaling curve (§5.3, Fig. 3)
- Training uses N ∈ {2, 4, 8, 16} domains at a **fixed 1,024 tasks**, and Pass@4 is measured.
- Both curves rise monotonically, and the paper says there is no plateau at N = 16. That rests on five points with no error bars.

| N | 0 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| VitaBench Pass@4 | 27.0 | 29.2 | 30.4 | 33.8 | 34.5 |
| τ² Pass@4 | 56.0 | 57.0 | 59.6 | 60.7 | 63.6 |

- The N = 16 points differ from Table 2 (35.8) and Table 4 (62.9). That is consistent with the main model being trained on 2,560 tasks rather than 1,024 [inferred].
- **Confound:** at a fixed task count, more domains also means more distinct tools, schemas and graph shapes. "Diversity" is not separated from "tool count" or "schema count". Apple's SynthAPI (2610.37648) later varies tool, task and database diversity independently; its abstract reports that tool and DB diversity matter more than task count at larger budgets.

### 1.15 Costs and scale (App. B.2, Table 9)

| Stage | Prompt tokens | Completion tokens | Total |
|---|---|---|---|
| Domain foundation, per domain | ~379K | ~167K | ~546K |
| One verifiable task | ~74.6K | ~18.6K | ~93.2K |

**Totals [derived]:**
- 16 domains ≈ 8.7M tokens.
- 2,560 tasks ≈ 239M tokens, so task generation is about 96% of synthesis tokens.
- 1,000 Crucible-scale tasks at this rate ≈ 93M tokens, plus about 0.55M per domain.

**Not reported:**
- dollar cost;
- whether the oracle's best-of-16 runs are inside the per-task figure;
- RL compute;
- simulated-user cost. AutoForge measured the simulated user at about 6× the environment's time.

### 1.16 Failure cases
**Reported:**
- **Without EV:** tool calls that look plausible fail at runtime on unmet preconditions or mismatched state, e.g. refunding an order that does not exist. The resulting noisy rewards conflict with each other.
- **LLM-judge reward:** the policy optimises for language that sounds right rather than for correct logic.
- No qualitative failure analysis of the trained agents on τ² or Vita is given.

**Observed in the paper's own artefacts [inferred, from Appendix D–E]:**
- **Non-deterministic tools.** `delete_job_application` and `archive_old_applications` stamp `datetime.now()`. AWM has the same defect: 531 of its 1,000 scenarios seed rows with `datetime('now')`, which Crucible pins in `interop/openenv/awm.py`.
- **Loose search.** A search for "energy analyst" returns a "Data Analyst" row, APP004. Either the keyword tool matches any token or the distractor design leaks into results.
- **Skipped records.** The agent queries APP001–005 and APP007–009 but skips APP006, and the reward cannot see that.
- **Invented timestamps.** The agent makes up `created_at = 2024-03-15 09:30:00` as a tool argument, and timestamps are hard constraints. Grading such a run correctly needs that column to be exempt.
- **Ungradable read-only turns.** Several turns are read-only computations (ROI, conversion rates) that change no state.

### 1.17 Appendix contents; prompts
- **A:** the t-SNE OOD check. Tables 6–7 give a tool schema and a DB schema example.
- **B:** domain statistics (tools against tables against graph density) and the token table.
- **C:** the GRPO objective.
- **D:** listings of the Pydantic DB code and three tool methods.
- **E:** a full Job-Seeking trajectory of about 52 turns.
- **No prompts are published, in the appendix or anywhere else.** I can describe them only by role (§1.3). Reproducing ScaleEnv means writing every prompt from scratch.

### 1.18 Status
- Published at ICML 2026 (PMLR 306, "tu26a"; poster #66374), per the citing-works pass. I did not check this myself.
- **No official code or data.** The HF paper page links nothing, and the meituan-longcat GitHub org has no ScaleEnv repo. An unrelated personal repo, `hangyu1234/ScaleEnv-DeepSeek-` with 0 stars, is not official.
- Citing papers (EnvCraft, SETA) list ScaleEnv as a baseline with 16 environments, 2,560 tasks and a rule-check reward. Agent-World (2604.18292) reports beating ScaleEnv-8B.

---

## 2. Critique

### 2.1 Gaps and hackable spots in the verifier
1. **Read-only goals cannot be checked.** The reward looks only at the final DB. If the intent is "tell me my conversion rate", the gold state equals s₀, so **a no-op earns full reward** and a wrong answer is never penalised. The Appendix E trajectory is mostly this kind of turn. Crucible already has answer checkers in `verify/answers.py` and AWM-style checks of the final answer; ScaleEnv has neither.
2. **Collateral damage is undefined.** If only the touched rows are compared, an agent can overwrite distractor rows, e.g. with a broad `batch_update_application_status`, and still pass. If the whole DB is compared, exempt-field leniency elsewhere could hide edits. The paper says neither.
3. **Semantic-alignment columns are an unspecified fuzzy matcher.** If it is an LLM, the judge the paper rejects comes back for the text columns, along with its hackability. If it is embedding similarity, short boilerplate text may pass.
4. **Hard constraints on values the user never states.** The gold state comes from C₁'s concrete arguments. If the simulated user does not say the exact timestamp or amount, a correct agent fails: a false negative. Wall-clock writes make the gold state non-reproducible unless those columns are exempt.
5. **A single reference path.** The gold state comes from one chain. Equally valid strategies that leave different, legitimate end states fail, e.g. recording a reminder as a note rather than a deadline. Nothing measures the false-negative rate. Crucible's `factory/alternatives.py` (at least two valid alternatives per task, one from a different model family) addresses exactly this.
6. **The debugger can edit the expected DB instance** to agree with buggy code (§1.4).
7. **No red-teaming of the reward.** No agent was run to search for reward without doing the work, and no mutation testing of the checks was done. Compare Crucible's `forge/hacker.py` and `accept/mutation/`.

### 2.2 Risks from the LLM-simulated user
- **One simulator model** (Qwen2.5-72B) is used throughout training, with unknown temperature and no fidelity check.
- **LOGIGEN (2603.00540) shows the failure directly.** Training against a low-temperature DeepSeek-V3.2 simulator (T = 0.1) gave higher and faster training reward. Evaluated against a different simulator (GPT-4.1), the policy rose and then collapsed, a bell-shaped curve in Retail. It had learned simulator-specific phrasings. ScaleEnv does not measure or mitigate this.
- **The simulator can leak or garble the hidden intent.** AutoForge (2512.22857) found enough simulated-user errors to justify an LLM judge that masks those trajectories. APIGen-MT uses a best-of-4 user to cut run-to-run variance from 11.1 to 2.6.
- **Benchmark scores depend on the evaluation simulator.** AutoForge's τ²-Telecom score moves from 76.3 to 90.4 just by changing it. ScaleEnv does not say which simulator it evaluated with, so Table 1's baseline rows may not be comparable.

### 2.3 Limits on diversity
- The domains come from keywords chosen by hand. There is no taxonomy, no novelty search, and no measure of diversity beyond a t-SNE picture.
- All environments share **one architecture**: CRUD over a dictionary of Pydantic tables. There are no external side effects, time, events, other actors, documents, browsers, files or long-running processes. Every task has the shape "one user, one DB, at least 20 tools".
- Difficulty is never measured on the policy. There is no pass-rate band, no filtering of easy or hard tasks, and no curriculum. The task counts per domain are arbitrary powers of two.
- **No policy documents.** τ² rewards following a policy, which ScaleEnv's tasks never test. The authors present the τ² gains as format generalisation; this is plausible but not separated from other effects.

### 2.4 What they did not measure
- **Pipeline numbers:**
  - yields per stage, debug iterations, abandoned tools or domains;
  - a human audit of task validity;
  - the reward's false-positive and false-negative rates;
  - simulated-user fidelity or leak rate;
  - the share of tasks whose GRPO groups were all-0 or all-1 (no signal);
  - training curves.
- **Comparisons:**
  - seeds or variance (every table is single-run);
  - a comparison with other synthesis pipelines (AutoForge, EnvScaler, AgentScaler, APIGen-MT) at equal model and budget;
  - an SFT baseline.
- **Unablated components:** distractors, expansion, the gate, the 20-tool floor, the user simulator.
- **Effects on the model:**
  - contamination: the construction models (GPT-5.1, DeepSeek-V3.2) know τ²'s domains;
  - general capability after RL, e.g. BFCL hallucination or refusal;
  - scaling beyond 16 domains, and tool count versus domain count.
- **Inconsistencies, all small, but signs of a rushed write-up:**
  - "non-overlapping" sets that overlap;
  - the "EnvZero" name;
  - the unlabeled metric in Table 1;
  - mismatched N = 16 points;
  - two different statements of tools per domain.

---

## 3. Related work

**How these were checked.**
- Every id was checked on arxiv.org (abs page plus HTML) on 2026-10-29.
- Rows marked ✱ were also downloaded and title-checked (§5).
- Numbers come from the paper text as read by the research passes. "n/r" means not reported in the text read.
- §3.3 papers dated 2026-06 to 2026-10 were checked on their abs pages, but most mechanisms come from their abstracts only.

### 3.1 Environment-synthesis work that ScaleEnv cites

| Paper | Mechanism | Scale | Verification | Key numbers |
|---|---|---|---|---|
| **AutoForge ✱** 2512.22857 (Cai et al., Tongyi/Alibaba, Dec 2025) | LLM writes a key-value state schema and Python tools from tool documents. An LLM links tool I/O into a graph; random walks are merged into a DAG with "reasoning nodes". Executing the DAG gives the gold state S*. The task text is cut to the minimum information needed. | 10 envs, 1,078 hard tasks | Binary state equality with S*. An LLM judge masks trajectories with simulated-user errors. DAPO-style dynamic sampling. Trained with ERPO (advantage normalised per environment), cold-start SFT then RL. | Qwen3-30B-A3B-Thinking on τ²: Retail 58.8→74.8, Airline 58.0→62.0, Telecom 26.3→76.3. VitaBench Cross 17.5. Telecom 76.3→90.4 with GPT-5 as the user. No yield or cost reported. |
| **EnvScaler ✱** 2601.05808 (Song et al., Renmin, Jan 2026) | SkelBuilder mines stateful task skeletons from API-Bank and ToolACE, plans state, rules and tools, and writes one Python class per environment. ScenGenerator writes a start state, a hard task, and a checklist of K validation functions. | 191 of 266 envs kept (71.8%). About 18.6 tools per env. About 7K scenarios. | Code: a blind dual-agent check over 100 rounds (a tester sends positive and negative calls without seeing code; a checker reads the code and the state diff), pass threshold 0.85. Reward: the fraction of validation functions true on the final state. | Qwen3-8B BFCL-MT 28.9→37.0 (SFT)→41.9 (SFT+RL); τ-bench 38.2→44.8. RL without SFT is weaker. About $1.02 per env and $0.064 per scenario (GPT-4.1). |
| **AgentScaler ✱** 2509.13311 (Fang et al., Tongyi, Sep 2025), "Towards General Agentic Intelligence via Environment Scaling" | 30K+ real APIs are clustered by Louvain into domains over an embedding-plus-LLM tool graph. Each domain gets a DB schema, and tools become DB read/write code. Tasks are walks over the graph, executed to a gold state. | 30K+ APIs, 1,000+ domains | Filters: validity; final state equals gold; read-only steps must match the gold calls exactly. | SFT only. AgentScaler-30B-A3B τ²-Telecom 55.3, ACEBench-en 75.7. |
| **Toucan ✱** 2510.01179 (Xu et al., UW/MIT-IBM, Oct 2025) (ScaleEnv's "real-world" category) | Real remote MCP servers. Five open LLMs synthesise tasks. Three teacher models run them with real tool execution. Extensions: irrelevance, persona, multi-turn. | 495 servers used (17.7% yield from about 2,800). 2K+ tools. 1.53M trajectories. | Real execution plus rule checks, plus LLM judges (Kimi-K2, GPT-OSS-120B) with strict score thresholds. | SFT. Qwen2.5-32B BFCL-v3 61.7→70.5. τ²-Telecom dropped for 7B and 32B. |
| **ToolACE-MCP / ACE-Router** 2601.08276 (Yao et al., ZJU/SJTU/Huawei, Jan 2026; retitled in v2) | Graph over 627 MCP tools; LLM mutation grows it to 2,005 variants. Walks become tasks. Dialogues are fully simulated. Trains a history-aware tool router. | 15K+ router samples | Human audit of 100 trajectories (κ = 0.93). No execution. | Qwen3-8B LoRA, MCP-Universe 46.1→53.4. Removing mutation drops it to 47.1. |
| **ToolACE** 2409.00920 (Liu et al., Huawei Noah's Ark, Sep 2024) (ScaleEnv's "LLM-simulated" category) | API context tree mined from pretraining text, with speciation, adaptation and evolution of APIs. Dialogues from user, assistant and tool agents. Complexity measured by the learner's own NLL. | 26,507 APIs, 390 domains | Rule layer plus model layer. Tool replies are simulated, not executed. | LLaMA-3.1-8B BFCL-v3 59.2. Medium-complexity data does best. |
| **DreamGym ✱** 2511.03773 (Chen et al., Meta, Nov 2025) | A reasoning "experience model" (Llama-3.1-8B), trained on offline transitions, predicts the next abstract state and the reward. A replay buffer seeds it. Task variants are chosen by group reward variance. | WebShop, ALFWorld, WebArena | The model predicts the reward. No real verifier during synthetic RL. | WebArena with zero real data: 13.3 against 7.3 for real GRPO (3B). Removing task generation costs about 6 points. |
| **Simia ✱** 2511.01824 (Li et al., UW/Microsoft/CMU, Nov 2025) | GPT-5 or o4-mini simulates whole trajectories (SFT) or each tool step plus a final 0/1 judgement (RL) from tool specs, a policy and seed examples. | About 135K synthetic trajectories [sum of three seed expansions] | An LLM judge. No execution. | Qwen2.5-32B SFT τ²(Airline+Retail) 58.9 (GPT-4o 54.2). OfficeBench RL: simulated environment 64.7 against real 60.8. |
| **FTRL** 2508.08791 (Ye et al., Fudan/ByteDance Seed, Aug 2025) | Questions are split into sub-questions. GPT-4o writes a tool per sub-question and scales complexity with distractor tools and nested parameters. Tools are local Python. | 2,215 training instances | Environment feedback per sub-question. Reward 2q/(p+1) plus format penalties. | Qwen3-8B (non-reasoning) average 31.0→45.4. |
| **APIGen-MT ✱** 2504.03601 (Prabhakar et al., Salesforce, Apr 2025) | Blueprints (instruction, gold actions, gold outputs) from API-graph walks plus policy, data and persona samplers. A best-of-4 self-critiquing simulated user rolls them out, and only reward-1 trajectories are kept. Reverse recombination makes harder tasks. | τ-bench domains, about 5K trajectories | Execution with a state diff, policies compiled to unit tests, an LLM committee, and reflection loops. | Task pass rate 28%→70% with feedback. xLAM-2-70B τ-bench 56.2, BFCL-v3 78.2 (rank 1 at the time). |
| **Kimi K2** 2507.20534 (Moonshot, Jul 2025) | 3K+ real MCP tools plus 20K+ synthetic tools. Thousands of synthetic agents, tasks with rubrics, persona users, and a stateful **tool simulator with injected failures**. Real Kubernetes sandboxes for code. | 23K+ tools | Rubric LLM judge for the simulated tools, unit tests for code, a self-critique rubric reward in RL. | τ²-Bench 66.1, SWE-bench Verified 65.8 (non-thinking). |
| Benchmarks: **VitaBench** 2509.26490 (Meituan) | Delivery, in-store and OTA life-service tasks; 66 tools in the cross-scenario set; 100 cross-scenario and 300 single-scenario tasks. | – | Rubric sliding-window judge (claude-3.7-sonnet), κ = 0.828 against humans; GPT-4.1 as the user. Reports Avg@4, Pass@4 and Pass^4. | Best Cross Avg@4: o3-high 30.0. |
| **τ²-Bench ✱** 2506.07982 (Sierra) | Dual control: both the user and the agent act. | Airline 50, Retail 115, Telecom 114 tasks | DB state, assertions, NL assertions, action matching. | GPT-4.1: Retail 74, Airline 56, Telecom 34. |

### 3.2 Environment and task scaling beyond tool APIs (SWE and general)

| Paper | Mechanism | Scale | Verification | Key numbers and lessons |
|---|---|---|---|---|
| **SWE-smith** 2504.21798 (Yang et al., Stanford, Apr 2025) | One environment per repo, many bugs per environment: LM modify or rewrite, 13 AST mutations, bug combination, PR mirroring. Issue text written from the diff. | 128 repos, 50,137 instances | Kept only if it breaks at least one previously passing test. | Overall yield 50.1% at 2.32¢ per candidate; about $1,360 in total. 32B Verified 40.2% with SFT. Difficulty predicted training value poorly. |
| **SWE-Gym** 2412.21139 (Pan et al., Berkeley, Dec 2024) | Real PRs with environments set up by hand. | 2,438 executable instances, 11 repos | F2P harness. | About 200 annotation hours. SFT on 32B, Verified 7.0→20.6. Training on its own on-policy data *hurt* (Lite 15.3→8.7). |
| **R2E-Gym** 2504.07164 (Jain et al., Berkeley, Apr 2025) | Mines commits, generates F2P tests where missing, and writes issues back from the commit. | 8,135 tasks | Hybrid verifier: execution plus a learned judge. | Hybrid Best@26 51.0% against about 43% for either verifier alone. |
| **SWE-rebench** 2505.20411 (Nebius, May 2025) | Automatic install recipes (an LLM writes up to 3 per task), execution validation, LLM quality labels, rolling decontamination by date. | 21,336 tasks from 3,468 repos | F2P plus P2P; flaky tests re-run. | Only 31% of repos install automatically. DeepSeek-V3 scores 39.7 on Verified against 21.3 on fresh tasks. |
| **SWE-Factory** 2506.10954 (SYSU/Huawei, Jun 2025) | Four-agent environment builder with a memory pool; grading by exit code. | 671 issues (eval) | Exit-code F2P. Precision 0.92, recall 1.0. | 33–40% valid instances at $0.024–0.045 each. Every false positive was "error2pass". |
| **Environment Tuning** 2510.10197 (Ant Group, Oct 2025) | Tunes the environment instead of synthesising tasks: a 4-stage curriculum, LLM hint feedback checked for leaks, a per-turn progress reward. | 400 BFCL-v3 multi-turn tasks | Per-turn state plus result. | Qwen2.5-7B BFCL-MT 7.0→36.9. SFT baselines collapse out of distribution. |
| **TaskCraft** 2506.10055 (Shi et al.; OPPO UNVERIFIED, Jun 2025) | Atomic tasks from documents, extended in depth (more hops) and width (merged subtasks). | About 32–36K tasks | LLM judge plus a check that the **tool is needed**: an agent with tools must beat one without. | Atomic pass rate 54.9→68.1% after prompt optimisation. |
| **AutoEnv** 2511.19304 (HKUST-GZ/DeepWisdom, Nov 2025) | Theme → three-layer YAML DSL → code. Coding agents build the environment, a level generator and a validator. | 65 of 100 environments verified | Execution, then reachability of levels, then a differential **weaker-model-must-not-beat-stronger** gate. | Overall 65% at about $4.12 per environment; human theme review raises it to 80%. |
| **GEM** 2510.01051 (Sea AI Lab, Oct 2025) | Gym API for agentic LLMs across 24 environments. | 24 envs | Rewards per environment. | ReBN (REINFORCE with batch normalisation) supports dense per-turn rewards. |
| **Reasoning Gym** 2505.24760 (Oct 2025 v2) | 102 procedural generator–verifier pairs with difficulty set by parameters. | Unlimited | Algorithmic. | Curriculum raises difficulty once accuracy is above 70% over 20 steps. |
| **Synatra** 2409.15637 (CMU, Sep 2024) | Turns wikiHow and web snapshots into synthetic demonstrations. | 100K | Nothing executed. | $0.031 per demonstration. A cautionary baseline for unverified data. |
| **BUTTON** 2410.12952 (Baichuan, Oct 2024) | Atomic tasks composed bottom-up; multi-agent playback top-down with simulated tools. | 8K trajectories | An LLM checks only that a composition is solvable. | Llama3-8B GTA 1.4→30.5. |
| **Self-Challenging Agents** 2506.01716 (Zhou et al., Meta, Jun 2025) | The agent explores the tools, then writes a Code-as-Task: instruction, verifier, example solution, failure cases. | 800 tasks | The solution must pass and every failure case must fail. | Only 5.2% of generated tasks survive every filter. The failure cases remove all false positives. 8B self-improvement 12.0→23.5. |
| **Absolute Zero** 2505.03335 (Tsinghua, May 2025) | One model proposes and solves code reasoning tasks. | Unlimited | Executor plus a determinism check (run twice). Learnability reward. | AZR-Coder-7B +15.2 on math. |

### 3.3 Papers citing ScaleEnv, or later environment-scaling work (2026)
Semantic Scholar lists **36** papers citing ScaleEnv. Google Scholar could not be queried (rate-limited), so this count is a lower bound. The most relevant:

| Paper | Mechanism | Scale | Verification | Key numbers |
|---|---|---|---|---|
| **Agent World Model (AWM) ✱** 2602.10090 (UNC/Snowflake, Feb 2026; ICML 2026) | Scenario → tasks → DB → sample data → tool schema → MCP environment code → verification code, with execution-based self-correction. Crucible already vendors it (`_vendor/openenv_awm`). | 1,000 envs, 35,062 tools, 10,000 tasks | **Code-augmented LLM judge.** In its ablation, code-only (state diff) verification is brittle under environment imperfections (false negatives), and adding an LLM over state-diff signals is best: 4B BFCL 64.5 against 55.7 for code-only and 51.9 for LLM-only. | More than 85% first-attempt success; 1.13 repair iterations. 8B BFCL-v3 53.8→65.9. This **contradicts ScaleEnv's "rule beats judge"**: the answer depends on how imperfect the environment is. |
| **LOGIGEN ✱** 2603.00540 (Baidu/Tsinghua, Feb 2026) | Policy-first. An Architect compiles natural-language policy into DB constraints, a Set Designer builds boundary states, and an Explorer finds paths. | 20K tasks, 8 domains | Exact state equivalence. | LOGIGEN-32B (RL) τ² 79.5% (base 40.7). The **user-simulator hacking** study is in §2.2. |
| **Agent-World ✱** 2604.18292 (Renmin/ByteDance Seed, Apr 2026) | Environments and tasks discovered from thousands of real MCP themes, with controllable difficulty. A **self-evolving arena** finds capability gaps and synthesises targeted tasks, so the policy and the environments co-evolve. | n/r in abstract | n/r in the pass | 8B: τ² 61.8, BFCL-v4 51.4, MCP-Mark 8.9. Reports beating ScaleEnv-8B. |
| **SynthAPI** 2610.37648 (Apple, Oct 2026), "Scaling Laws for Tool-Use Agents via Controllable Environment Synthesis" | Controls tool, task and DB diversity independently and fits scaling laws. | 458,752 tasks; 849 trained models | n/r | Tool and DB diversity beat task count at larger budgets (abstract). The direct test of ScaleEnv's curve. |
| **CompoWorld** 2609.33665 (AllSpark, Sep 2026) | Composes verified services into cross-service tasks by walks over dependency graphs. A world model stands in for tools that cannot be implemented. Rubric reward. | 448 services, 10,130 tools; 1K RL tasks | Rubric (completion-focused) | +9.17 on average over 8 benchmarks, on Qwen3.6-35B-A3B. |
| **EnvFactory** 2605.18703 (HKUST-GZ/Huawei, May 2026) | Explores real resources into stateful tool environments; topology-aware sampling of trajectories with implicit intent. | 85 envs, 7 domains | Execution | +15% BFCLv3, +6% τ² and Vita. |
| **EnvCraft** 2609.05576 (HIT, Sep 2026) | Sandboxed workspace environments for claw-style agents; topology-aware task generation. | 139 envs, about 20K tasks | Sandbox | Up to +11.9% on Claw benchmarks. |
| **SETA** 2607.10891 (CAMEL-AI, Jul 2026) | SETA-Synth turns sources into terminal environments; SETA-Evol expands difficulty under one shared verifier. | 4,500+ envs | Shared verifier | Qwen3-8B TB2 12%. |
| **RST** 2608.05466 (UGA/Tencent, Aug 2026) | Recursively lengthens verified seed tasks, realigns verifier and instruction, and re-validates in a fresh sandbox. | 37,484 tasks over 15 rounds, about $0.05 per task | Fresh-sandbox validation | DeepSeek-V4-Pro pass@4 falls from 90% to 2.5% across the rounds. |
| **WorkForge** 2610.04906 (Tencent/Fudan, Oct 2026) | Expert workflows plus real files. Checkable "factual anchors" yield programmatic and semantic verifiers. | 16.7K envs, 40 domains | Programmatic plus semantic | GDPVal 45.5→73.6. |
| **AutoEnvScaling** 2610.14163 (Northeastern/Microsoft/Stanford/UW/UNC, Oct 2026) | A proposer agent builds environments from the solver's rollouts, closing the loop; self-play. | n/r | n/r | 25× more valid environments than prompt-only generation; +12.2 on TB2.1 with self-play. |
| **EAU** 2610.16523 (Alberta/NUS/Kuaishou, Oct 2026) | A meta-agent diagnoses failures **during RL** and hot-swaps fixes to the user simulator, environment, reward and config. | n/r | n/r | +14.7–15.0% over the strongest baseline. |
| **RODS** 2606.19047 (Ant Group/Westlake, Jun 2026) | Uses GRPO reward variance to find tasks at the capability boundary and synthesises matched variants online. | 400 seeds | n/r | Matches a 17K offline pipeline with about 20× fewer trajectories. |
| **Data World** 2610.21656 (Meituan LongCat/NJU, Oct 2026) | World mechanisms first, then files. An audit checks that targets can be recovered; rubric reward. | 2,135 tasks from 1,037 worlds | Recoverability audit plus rubric | RL-zero +20.9 DABstep-Hard. |
| **Tev** 2610.18868 (Adobe/UCLA, Oct 2026) | A hidden latent-world spec, then tasks rendered from it. Progress-potential dense reward. | 1,170 envs | Spec-derived | n/r in the pass |
| **VHD-Play** 2609.27321 (Qwen, Sep 2026) | Solves a mathematical model first, then renders it as stateful tools, so the dynamics and the reference come from one source. | 3,300 envs at a few cents each | Model-derived | Agentic score 0.204→0.815. |
| **GenEnv** 2512.19682 (Dec 2025) | Agent and environment simulator co-evolve; an α-curriculum reward keeps tasks in the agent's zone of proximal development. | n/r | Simulator | Up to +40.3% over 7B baselines. |
| **Endless Terminals** 2601.16443 (Jan 2026) | Four-stage pipeline: task, container, tests, solvability filter. PPO with binary reward. | 3,255 tasks | Tests plus a solvability filter | Qwen2.5-7B 10.7→53.3 on the held-out dev set. |
| **InfiniteWeb** 2601.04126 (MSRA, Jan 2026; ACL 2026) | Test-driven synthesis of functional websites, with task evaluators that give a dense reward. | n/r | Generated evaluators | GUI agents improve on OSWorld and Online-Mind2Web (no figures in the abstract). |
| **Reward Hacking Benchmark** 2605.02964 (May 2026; ICML 2026) | Tool tasks with natural shortcuts. | n/r | n/r | Exploit rates 0–13.9%; environment hardening cuts exploits by 87.7%. |

**Other ScaleEnv citers whose titles were confirmed but whose mechanisms were not read:** Trajectory2Gym 2610.33568, EnvLunch 2610.33817, WebUniverse 2610.33149, GRADUS 2610.27781, OfficeTown 2610.23194, Interlock 2610.19686, Agent-Reflex 2610.17953, RES 2610.16397, ProgAgent 2610.14774, ClawLoop 2610.14118, DAEDALUS 2610.08048, WEFT 2609.36887, CUA-Sandbox 2609.32750, Environments as Scaffold 2609.08404, AgentBrew 2609.05837, "What Makes Good Agentic Data?" 2608.27260, EnvACE 2608.06197, SearchEyes 2607.05943, Qwen-AgentWorld 2606.24597, Text World Models 2606.09032, NoisyAgent 2605.27209, TRACE 2604.05336.

**Older and adjacent work** (verified, outside the window): AgentGym-RL 2509.08755, ARE 2509.17158, Learn-by-Interact 2501.10893, OS-Genesis 2412.19723, Magnet 2503.07826, RandomWorld 2506.11045, SPICE 2510.24684, Agent0 2511.16043.

**UNVERIFIED** (names seen only in other papers' tables): TermiGen, Tmax, RLVE-Gym, EnvGen, a LongCat "EnvFactory".

### 3.4 Cross-cutting findings that matter for us
1. **Execution-checked environments and state-diff rewards are now the default.** Every serious 2026 pipeline uses them: ScaleEnv, AutoForge, EnvScaler, AWM, LOGIGEN, CompoWorld. Purely simulated environments (Simia, DreamGym, Synatra, BUTTON) are cheaper and more scalable but rely on judges.
2. **Code-only versus judge rewards is not settled.**
   - ScaleEnv finds rule-based beats an LLM judge by 2 Avg@4 points.
   - AWM finds a code-augmented judge beats code-only by about 9 BFCL points, because code-only fails on imperfect environments.
   - The likely reconciliation: a deterministic state check where the state is clean, with the judge used only on imperfections. Crucible should *measure* this with its gold sets rather than pick a side.
3. **Yields reported:**

   | Pipeline | Yield |
   |---|---|
   | EnvScaler | 71.8% of environments |
   | AWM | more than 85% first attempt |
   | AutoEnv | 65% of environments |
   | SWE-smith | 50% of bug candidates |
   | SWE-Factory | 33–40% of instances |
   | SWE-rebench | 31% of repos |
   | Toucan | 17.7% of servers |
   | Self-Challenging | 5.2% of tasks |

   ScaleEnv reports no yield.
4. **Self-improvement has moved into the training loop.** Agent-World's arena, EAU's meta-agent, RODS's variance-triggered variants, GenEnv's curriculum, AutoEnvScaling's proposer that builds from rollouts, and DreamGym's variance-chosen tasks all generate during RL. ScaleEnv is a one-shot offline generator.
5. **The simulated user is the weakest link.** LOGIGEN's temperature study, AutoForge's masking, APIGen-MT's best-of-N user, and AutoForge's 76→90 swing when only the simulator changes all point to it. Few pipelines measure simulator fidelity.

---

## 4. Mapping to Crucible

Code areas are given from my own reading of `crucible/src/crucible` today. The lead's codebase map, being written separately, takes precedence where the two differ.

**Priorities:**
- **P0:** decide now for the 500–1000 plan.
- **P1:** build next.
- **P2:** later.
- **P3:** skip, or keep as a pattern only.

| ScaleEnv or related mechanism | Do we have it? | What adopting it would take | Priority |
|---|---|---|---|
| **Domain foundation:** keyword → tool schema → DB schema → DB and tool code | **Partial.** `world/apps/sqlite_app.py` takes a `tools_module` (MiMo format, which AWM apps reduce to). `world/apps/json_api.py` gives a JSON-doc CRUD App. AWM's 1,000 generated envs load through `interop/openenv/awm*.py` with vendored code verifiers. A toy `world_synth` Architect exists in `packs/_toy/.../forge/scenario.py`, writing tasks before data in AWM order. **No generic synthesizer from a keyword.** | A `world_synth` generator in `forge/`, running on the Tinker swarm. It should **depend on** AWM's pipeline order (scenario → tasks → DB → data → tools → verifier) and port ScaleEnv's tool-first schema derivation and tool→table map; that follows the build-on-frameworks rule. Output is an `sqlite_app` with a `tools_module`, so the existing World, snapshot and verifier stack applies unchanged. | **P1** (the main gap) |
| **Procedural testing:** Code and Test agents in parallel, three outcomes, debug loop | **Partial.** Acceptance A1–A13 and mutation in `accept/`. Asymmetric information is enforced by staging in `forge/orchestrator.py` (`visible`), so the test writer cannot see the code. No per-tool generated unit tests. | A tool-level gate: Test agent (code hidden) → cases plus DB instances → three-outcome classifier → repair. Take EnvScaler's **blind dual-agent check with a pass threshold** as the stronger form. **Forbid the repair role from editing the expected state** (the hole in §1.4), or require a different-family checker to approve any edit to it (`factory/provenance.py` already enforces the family rule). Report yield and repair iterations per stage in `forge/economy.py`. | P1, with world_synth |
| **Tool dependency graph** | **No.** `workflow/graph.py` is a graph of milestones, not tools. | Build it deterministically, not with O(n²) LLM pairwise calls: (a) data flow from matching output and parameter names and types; (b) state dependency from the tables each tool reads and writes, recorded by tracing tool execution over the `sql` StateAdapter; (c) pre- and post-conditions from anticipated-rejection cases. Use an LLM only to break ties. | P2 |
| **Seed chain as code, executed to the gold state** | **Yes, in substance.** The forge and factory are solution-first: the oracle comes before the verifier (`factory/driver.py` funnel; `forge/compose.py` oracle actions and component map). | Nothing new beyond expressing chains as tool-call programs for world_synth tasks. | – |
| **Distractor rows and tools** | **Yes.** `forge/perturb.py` provides `distractor` and `decoy` operators, and cosmetic perturbations must not change the solution. `world/noise.py` adds instruction and tool noise. | Expose distractor density as a difficulty knob in `factory/difficulty.py`'s controller, which ScaleEnv scales by hand. Add a **collateral-damage invariant:** distractor rows unchanged at the end. | P1 (cheap) |
| **Snowball expansion:** dependency-closed BFS, at least 20 tools | **No.** Composition (`forge/compose.py`, A14 necessity) composes tasks, not tool surfaces. | A "tool-surface closure" acceptance check: every exposed tool, called with schema-valid arguments from the start state, returns a non-crash observation (interaction completeness). Then grow the surface by dependency closure. The LLM gate and c(H) metric are not needed: use measured yield and the ladder instead. | P2 (closure check P1) |
| **Oracle feasibility:** strong solver best-of-k | **Yes, and stronger.** `forge/ladder.py` has rungs L0–L4 including frontier pass@k; `factory/alternatives.py`; `factory/targeting.py`. | – | – |
| **State-diff reward with three column policies** | **Mostly.** `verify/sql_checks.py` (EnterpriseOps `database_state` port), `verify/state.py` predicates, StateAdapter `diff` and `fingerprint`, AWM code verifiers, `world/verifier.py` separate-verifier custody. | Declare a column policy (exempt, exact, semantic) **in the schema at synthesis time**, generated IDs and wall-clock columns exempt by rule. Freeze the clock with `world/clock.py` (we already pin AWM's `now`). Grade semantic columns with validated judge items (`verify/judge_item.py`) only. Add answer checks (`verify/answers.py`) for read-only intents, since ScaleEnv lets a no-op pass these. | P1 |
| **Rule versus judge versus code-augmented judge** | Rule checks plus validated judge items; gold sets in `factory/goldset.py`. | Run ScaleEnv's and AWM's comparison on our own gold set: false pass and false fail for each mode. Do not assume either paper's result. | P2 |
| **Simulated user with hidden goal and profile** | **Yes, and stronger.** `actors/simulated.py` (persona, known and unknown info, outside-policy scope), `actors/grounded.py` (answers only from records), `actors/fidelity.py` (fabrication rate). | Add **simulator diversity** during training (model and temperature rotation, per LOGIGEN). Always **evaluate against a simulator from a different family than training**. Add a communicability check: every hard-constraint value in the oracle chain is either in the start state the agent can see or in the simulator's known info. | P1 |
| **Entity consistency:** FKs across tables | Partial: SQL invariants in `verify/`. | An FK-integrity and schema-constraint check on every generated start state, as an acceptance check. | P2 |
| **Difficulty** (ScaleEnv has none on the policy) | **Yes, and stronger.** `forge/learnability.py` (band p ∈ [0.1, 0.7] with a Wilson interval), `forge/difficulty.py` (online p̂), `factory/difficulty.py` (controller after R-Zero), `factory/calibration.py`. | – | – |
| **Domain diversity beats task count** (ScaleEnv Fig. 3; SynthAPI) | Family-level machinery: `factory/family.py` (at least 20 tasks on at least 3 surfaces, transfer proxy), `factory/ortho.py` (4-layer orthogonality), the diversity quota in `forge/economy.py`. | **Plan the 500–1000 pool as many families with few tasks each,** not a few deep families. Measure transfer on **held-out families**, as ScaleEnv's OOD setup does, rather than held-out tasks of a seen family. Track families against a held-out score as our own scaling curve. Today the factory produces only debugging tasks, so domain breadth would come from world_synth. | **P0** (planning decision) |
| **Execution verification ablation** (Table 3: about 6 τ² points) | Our acceptance contract is execution-first throughout. | Evidence for keeping the gates strict when moving to synthetic tool worlds; no work needed. | – |
| **Reward hacking** (ScaleEnv: nothing beyond "no judge") | **Yes.** `forge/hacker.py` (adaptive black-box hacker), `forge/specialists.py` (family memory), A13 checks. | Run the hacker on world_synth tasks, especially for no-op and broad-write exploits (§2.1). | P1 with world_synth |
| **Zero RL with GRPO** | **Yes.** `train/objectives.py` (GRPO, Dr.GRPO, RLOO); Tinker bridge (`train/bridges/tinker`). | – | – |
| **Cost accounting** (tokens per domain and per task) | **Yes.** `forge/economy.py` (dollars per accepted task, stop rules), `factory/swarm_journal.py`. | Report per stage in tokens as well as dollars, so ScaleEnv, EnvScaler and AWM numbers compare. ScaleEnv's rate (93K tokens per task, 546K per domain) prices 1,000 tasks across about 50 domains at roughly 120M tokens [derived]. | P2 |
| **Self-improving loop** (absent in ScaleEnv; present in Agent-World, EAU, RODS, AutoEnvScaling, GenEnv, DreamGym) | **Partial.** Factory outer loop (swarm-pipeline/00 §12: recipe search on executed fitness only); swarm journal with blame and proposals; `accept/repair.py` (RLM task manager); online difficulty. **No step that targets environment generation at weaknesses found during training.** | Add a gap-targeting stage: cluster training failures (`signals/`, `factory/failures.py`) → propose families or variants aimed at them (RODS-style variants of boundary tasks; Agent-World-style arena) → the same acceptance gates. Keep the existing exploit guard: any yield-raising change must pass the hack-replay library and a held-out family. | **P1** |

**What Crucible already does better than ScaleEnv:**
- difficulty measured on the policy;
- adversarial hacking;
- grounded simulated users with fidelity metrics;
- valid alternative solutions, which avoid single-reference false negatives;
- a separate verifier World;
- a frozen clock;
- cross-family provenance;
- yield and cost economics.

**What ScaleEnv has that we lack:** an automated **breadth engine** that turns a keyword into an executable, tested, graph-structured tool world. Crucible should build world_synth on AWM's pipeline (we already depend on its data and verifiers) with ScaleEnv's tool-first schema, tool unit tests and dependency-closed tool surfaces. Every world it produces then passes through the existing factory gates. A ScaleEnv clone without those gates would not be enough.

---

## 5. PDFs downloaded to `crucible/docs/research/papers/` (git-excluded)
Each was checked by its title page:

| File | Paper |
|---|---|
| 2512.22857.pdf | AutoForge |
| 2601.05808.pdf | EnvScaler |
| 2509.13311.pdf | AgentScaler |
| 2511.01824.pdf | Simia |
| 2511.03773.pdf | DreamGym |
| 2504.03601.pdf | APIGen-MT |
| 2510.01179.pdf | Toucan |
| 2602.10090.pdf | Agent World Model |
| 2604.18292.pdf | Agent-World |
| 2603.00540.pdf | LOGIGEN |

ScaleEnv itself (2602.06820.pdf) was already present.
