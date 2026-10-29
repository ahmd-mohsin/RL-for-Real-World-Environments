# Extract F: groups 11 and 12 (CUA and MCP benchmarks, GUI RL, tool and user simulators, dense Worlds, traces-to-Worlds, privacy for seeds)

**Sources.** Group 11: 29 papers (21 FULL, 8 PARTIAL: UI-TARS 2501.12326, OpenCUA 2508.09123, Non-Collaborative Users 2509.23124, Lost in Multi-Turn 2505.06120, SimulatorArena 2510.05444, MirrorBench 2601.08118, UserBench 2507.22034, UserRL 2509.19736; in each the main text was read in full). Group 12: 18 papers (17 FULL, 1 PARTIAL: the survey 2606.12191). No ABSTRACT-ONLY cards. Core papers for Part 2: WildAWM 2610.26828, Terminal-Universe 2609.04148, Trace2Env 2610.06100, Gaia2 2602.11964, AndroidWorld 2405.14573, MCPMark 2509.24002, ZeroGUI 2505.23762, Lost in Simulation 2601.17087, Non-Collaborative Users 2509.23124, UserBench 2507.22034, UserRL 2509.19736, MCP-Atlas 2602.00933, Lost in Multi-Turn 2505.06120.

**Stage vocabulary.** F0–F12 stages, G1–G14 gates, and the four loops (verifier, World, task, generator) follow `framework-plan.md` §3.1 and §3.8. Proposed new gates are named in plain words (for example "red-on-start", "counterfactual probe", "sufficiency") rather than given new G-numbers.

**Five headline points from this extract.**
1. Executed checks beat model-judged rewards by a wide margin in GUI RL, and the commonest judge error is believing the agent's own claim of completion (ZeroGUI, LiveMCPBench, Online-Mind2Web, WildAWM).
2. The simulated user is part of the World. Swapping only the user model moves success by about 9 points and reshuffles difficulty bins, so p̂ must be keyed by the simulator too (Lost in Simulation).
3. Traces become Worlds, never text: rebuild the earliest observed state, withhold the agent's edits, complete under "solvable, not solved", gate on sufficiency, then re-solve (Terminal-Universe); if the system is gone, build a validated worldbook (Trace2Env).
4. Reward-bearing state must live in code. Every working design lets an LLM narrate or voice people but keeps state and reward in grounded variables (Concordia, Vending-Bench, UserRL, UI-Simulator, Trace2Env).
5. WildAWM trained Crucible's own student (Qwen3.6-35B-A3B) on synthesised CUA Worlds and ablated five gates one at a time; every gate paid for itself, and gains stopped at the teacher's frontier.

---

## B1 Domain spec and World/environment synthesis

### B1.1 One state API per App, with tenancy and cheap resets

**How it works.**
- OSWorld 2.0 [2606.29537] runs 31 self-hosted web services (email, bank, chat, travel, visa, insurance). Each is a Next.js app exposing `/api/state` (load, reset, export one JSON state envelope), `/api/files`, `/health` and a `/state-manage` authoring page. State is scoped by a `user_id` cookie, so many task instances share one container. A reverse proxy routes `app.localhost`.
- MCPWorld [2506.07672] restores only the App's persistent user data snapshot, not a VM image, which makes resets faster.
- WindowsAgentArena [2409.08264] gives each parallel worker its own full VM, keeps the golden snapshot in object storage and mounts code as volumes so the image rarely rebuilds. About 50 workers finish 154 tasks in under 20 minutes (median). It cannot speed the OS clock, so parallelism is the only lever.
- MCP-Atlas [2602.00933] pins each of 36 servers in a Docker image with a read-only filesystem except work dirs, allow-listed egress, and a restart between tasks.
- BrowserGym [2412.05467] defines a minimal task contract: `setup()` returns the goal; `validate()` runs after every action and returns (reward, done, message). It records a task dependency graph when tasks share mutable backends, and `prepare_backend()` resets Docker backends between agents. Action exceptions come back as observations, never as crashes.
- Gaia2's ARE [2602.11964] types every tool `read` or `write` at definition time and scopes it by role (agent, user, env).

**Failure modes.** Live remote services drift and are not resettable. MCPMark [2509.24002] finds local services much easier than remote ones (Postgres 76.19 vs Notion 41.96 and GitHub 47.83 for gpt-5-medium). MCP-Universe [2508.14704], LiveMCPBench [2508.01780] and MCP-Atlas all rely on live upstream APIs and report time-dependent results. Shared backends cause agent collisions (BrowserGym caps WebArena parallelism at 2–4).

**Adoption.** F1 `AppSpec` gets a required state envelope (load, reset, export), tenancy key and health probe; tools carry a read/write type and role scope. Prefer self-hosted or template-copy states (sample Postgres DBs, cloned sites) over live APIs as World tiers. Priority P1.

### B1.2 Persona-rooted dense Worlds built as deterministic day deltas

**How it works (WildAWM [2610.26828], all synthesis by open models, GLM-5.2 in OpenCode).**
1. A persona creator writes a "persona bible" grounded in 15 mirrored real websites (SQLite catalogues) and a shared MCP app index: demographics, at least 3 named contacts, 4–8 MCP services tied to story arcs, at least 2 multi-day story arcs (at least one "image-decisive"), routine, web accounts with fixed-salt password hashes, and a compact signature.
2. A diversity judge rejects a persona that shares occupation plus life domain, core tool or site pattern, or narrative arc with an accepted one (at most 2 revisions).
3. A day planner reads the real cumulative state for day d−1 and prior plans, then writes day d's plan across CLI/files, MCP and web, under **coverage pressure** (a per-app and per-site usage histogram from earlier days). Missing event skills and missing MCP services are written by a skill-creator and an app-creator into a shared library (83 reusable skills by the end).
4. A "god-mode" day implementer writes `delta.sh` plus deterministic Python sub-scripts that, replayed hermetically over day d−1, reproduce day d identically: frozen timestamp literals, derived IDs (MAX(id)+1), idempotent inserts, fixed salts. It also writes `produces.json` declaring every write.
5. World verification: deterministic lint (including cross-persona identity reservation); **replay twice with identical content hash**; every declared artifact exists and changed; live health (boot, every account logs in and renders history, all MCP services load); an independent reviewer outside the container checks the snapshot against the plan for intent and genuine defects only (missing artifacts, broken FKs, non-frozen timestamps).
- Scale: 36 personas, 640 world states (8–30 per persona, mean 17.8), 171.6K DB records, 59.3 MiB per world, median 52.4K tokens of state (against 1.1K for CUA-Gym's injected states). Instances share the application layer but isolate mutable state.

**Related.** Gaia2 builds 10 "universes" (400K–800K tokens each) along an app dependency graph rooted at Contacts, with priority-based conflict resolution (foundational apps win), PersonaHub personas expanded with the location, chats of at least 10 alternating messages with shared-experience references, emails whose reasons must fit both personas, calendars generated week by week over about 3 months, 500 real products. It lists what it does not handle: temporal consistency across apps, relationship consistency across channels, cross-modal references. OSWorld 2.0 seeds each task with files, open tabs, site records and message history consistent with one user profile (ids, dates, amounts agree).

**Failure modes.** WildAWM keeps credential files in the workspace; task counts skew to the first 6 personas (178 of 341 tasks); the same model family builds Worlds, writes tasks, judges and teaches.

**Adoption.** F2 `SeedBundle` construction for dense Worlds: a sequence of replayable deltas with a write manifest, accepted only on a double-replay hash match (this is G2 applied to the World itself) and a live-health probe. Add Gaia2's listed coherence gaps (timestamps vs calendar, relationships across channels) as G1 integrity checks. Feed the planner a usage histogram as coverage pressure. Priority P0.

### B1.3 The clock, events and notifications

**Gaia2 event model (ARE).**
- Everything is a timestamped event in a time-ordered queue with DAG dependencies. Types: Agent, User, Env, **Conditional** (polls a predicate, for example "cancel only if booked"), **Validation** (a milestone or constraint check that fails the run on timeout), **Oracle** (ground-truth actions for the verifier).
- A notification policy chooses which events reach the agent's context: low (user messages only), medium (consequences of agent actions; default), high (all env events).
- A `System` app gives `get_current_time`, `wait` and `wait_for_next_notification`. A wait switches the sim to event-to-event jumps, so hours of sim time run in minutes.
- By default model generation consumes sim time. To survive outages and rate limits, "simulated generation time" pauses the clock while the model responds and then advances it by the measured latency. An "instant" mode removes latency as an upper bound.
- Effect of time: GPT-5 (high) scores 0.0 on the Time split (inverse scaling: thinking longer misses deadlines); in instant mode it reaches 34.4, and Sonnet goes 8.1 → 26.7.

**Other clock mechanics.**
- Vending-Bench [2502.15840]: each tool call advances sim time by 5 min, 25 min, 75 min or 5 h by tool; `wait_for_next_day` skips ahead; each morning brings yesterday's sales and new email. Deliveries arrive later on the expected day, not at wake-up, so status must be checked.
- TimeArena [2402.05733]: three separate mechanics, namely action durations (1–10 min, hidden until the action starts), actor occupancy (Type 1 actions block the agent; Type 2 run in parallel) and object locks until completion; `wait` skips one minute.
- AndroidWorld [2405.14573] resets the device clock to a fixed instant at each task start so relative-date tasks are reproducible.
- UI-TARS-2 [2509.02544] accelerates and pauses time in browser game sandboxes by re-implementing the Window timing APIs, without touching game logic.
- OSWorld 2.0 injects emails or chat messages mid-run that change the semantic task state (9.3% of tasks are "dynamic environment").
- Concordia [2312.03664] spawns nested sub-simulations with a finer clock for detail-heavy scenes.

**Adoption.** F1/F2: adopt Gaia2's four event kinds and notification policy (as an observability knob), event-to-event waits and paused-clock latency accounting. Add `duration`, `occupies_actor` and `locks: [resource]` to `ToolSpec` (TimeArena). Deliver async outcomes offset from wake times. F9 must measure bands in the same time mode used in training. Priority P1 (P0 for the time-mode rule in F9).

### B1.4 Grounded variables: code holds reward-bearing state, the LLM narrates

**Pattern across papers.**
- Concordia: a Game Master (GM) turns free-text actions into events. Money, votes and inventory are Python components; the GM rejects actions that break them and tells the agent. `partial_state(player)` gives per-actor observability. Apps are Python classes with `@app_action` methods whose docs come from docstrings and type hints. Four app-fidelity tiers: language only, simple simulated code, LLM-prompted, real API.
- Vending-Bench: an LLM generates and caches price elasticity, reference price and base sales once; a closed-form seeded demand model then runs the dynamics (price impact × elasticity, day-of-week, month and weather multipliers, a variety multiplier capped at −50%, noise, round, clip to stock).
- UserRL [2509.19736]: each gym is a finite automaton with deterministic rule-based transitions and turn rewards; the LLM only voices the user from hidden state.
- UI-Simulator [2510.14969]: rule-based transitions for deterministic actions (type, scroll, tab/back/forward via a browsing stack) and LLM transitions for the rest.
- Trace2Env [2610.06100]: every LLM-proposed state change is a typed declarative effect validated on a state copy (paths, mutability, types, rule support, constraints); on rejection only the observation is committed.
- EnterpriseBench [2609.37658]: the Beer Game and the Enterprise Digital Twin run code-defined dynamics; the agent's plan is a constrained transform on a JSON template; reward is computed cost or earnings.

**Failure evidence.** LLM simulation drifts beyond one step (WebDreamer [2411.06559]: H = 2 and 3 underperform H = 1); simulated states import irrelevant context (UI-Simulator); stateless simulated APIs cannot model a write followed by a read (StableToolBench [2403.07714]). An LLM GM deciding outcomes is a non-executed ground truth.

**Adoption.** Rule for F1: reward-relevant quantities are tables or variables behind `StateAdapter`; LLM-backed components may produce text and propose effects, which pass a typed validation gate. Log every observation-only fallback as a `world_bug` candidate. For business simulators, "LLM sets cached parameters once, code runs the dynamics" keeps G2 under seed. Priority P1.

### B1.5 Tool and App synthesis with executed tests

**Mechanisms.**
- ComputerRL API workflow [2508.14040]: (a) from user-given example tasks, extract needed functions, diff against existing APIs, write new general-purpose ones only; (b) implement with the App's libraries plus error handling and logging; (c) generate tests for invocation and correct results across parameter inputs; failing APIs get the error back and are rewritten until tests pass. Result: 103 APIs over 6 Apps. Only the active App's tools are shown. Effect: GPT-4o GUI-only 11.2 → API-GUI 26.2 on OSWorld (Office 6.2 → 27.9). Gap: tests come from the same workflow (not blind) and expected states are not frozen.
- OSWorld-MCP execute-then-wrap [2510.24563]: o3 writes code solutions, keeps a summary only if its re-run completes the task (72 verified), wraps them as MCP tools; 192 curated server tools are added; two experts must both approve; task-specific tools and duplicates are dropped; 158 tools remain, 25 of them deliberate distractors.
- GTM error operators [2512.04535]: wrong type, missing required parameter, extra parameter, semantically invalid value.
- MCPWorld white-box Apps: verification hooks on the App's internal events (Frida instrumentation for compiled Apps, code injection for Python/JS/Electron/Java, or querying internal APIs, CLIs, logs, DBs), so transient and in-memory states are checkable.
- Coverage is a confound: MCP-only failures in MCPWorld are 48.42% "insufficient MCP coverage", a World defect.

**Adoption.** F1: keep ComputerRL's requirement-diff and "general functions only" rule, but write tool tests blind in a different family and freeze expected-state digests (plan rule). Add the four error operators to G13 (every tool must reject each with a non-crash error). Apps emit structured internal events that verifiers subscribe to. A tool that solves a task in one call is a shortcut (G9), and tools derived from the graded pool are contamination (G11). Priority P1.

### B1.6 Tiers for services the World does not own

**StableToolBench cache tier [2403.07714].** Per call: (1) cache lookup by category, tool, API, arguments; (2) on miss, call the real API; (3) if it fails, an LLM simulator (gpt-4-turbo) gets the API docs plus up to 5 real cached calls as few-shot; every served response is written back. Cache 352,630 → 164,980 valid entries; hit rates 96.5–97% in-domain, 76–91% for newer models. With the virtual server, disabling up to 50% of real APIs barely changes scores. Fidelity tests: blind human preference (annotators could not reliably tell simulated from real and often preferred simulated), documentation adherence (90% of 50), response diversity across arguments (only 2 of 60 APIs repeated a response over 5 calls). Keep some real error responses.

**MirrorAPI [2503.20527].** A Qwen2.5-7B simulator trained on 95,872 real request/response pairs over 7,500 APIs (requests made scenario-first at temperature 1.0, then call at 0.1; about 10,000 parameter-error failures kept; o1-preview rationales for CoT mode). Following score OOD 6.86 vs GPT-4o 5.93; on failure calls 8.28 vs 2.06; BLEU to real 35.6 vs 13.4 (OOD) and 74–90 vs 10–15 (ID). Key finding: on 158 fresh queries, agent scores under MirrorAPI tracked the real environment, while a GPT-4o-simulated environment gave clearly lower scores. Unseen-API fidelity stays low (BLEU 35.6).

**GTM [2512.04535], as a warning.** A 1.5B tool simulator trained on LLM-made calls for 20K+ LLM-generated tool schemas. For an unseen tool, GTM-only RL peaks at 0.34 at step 40 and collapses; hybrid (GTM 30 steps then real) reaches 0.41 against real-only 0.40. In search and retrieval training the simulator generates outputs from the task's ground-truth answer.

**Frozen corpus tier (BrowseComp-Plus [2508.06600]).** Harvest evidence per gold answer (o3 with search writes clue → URL → evidence), scrape, have humans label justifying spans and gold documents (400+ hours, agreement above 80%), add hard negatives from about 7 sub-queries per question (top 100 Google results each), dedup to 100,195 documents. Adding 10× unrelated documents did not change rankings.

**Adoption.** Part 1 App tiers: cache-first for read-only third-party services; freeze simulator outputs into the cache before gating (else G2 breaks); stateful Apps for any write task. Acceptance test for any simulated tier: a fixed agent's scores in real vs simulated World on the same tasks must match within noise (MirrorAPI). A tool output may never be computed from the task's gold answer (G11). Priority P2 (P1 for the real-vs-sim acceptance test).

### B1.7 Simulated people and users

**Architecture that works.**
- **State and reward in code, LLM as voice** (UserRL). Hidden state lives in the automaton; replies are generated from it.
- **Goal sharding with a delivery tracker** (Non-Collaborative Users [2509.23124]): the user goal is split into information pieces; a dialogue-state tracker forces any undelivered piece out before the user may end; an ending verifier blocks a premature `<END>` while the agent still needs confirmation. Dialogues that fail a goal-alignment check are regenerated, so failures stay agent-attributable.
- **Reveal-on-target rules.** UserBench [2507.22034]: the oracle user (temperature 0) gives only trip basics on reset, reveals a preference implicitly when asked about the relevant aspect, or proactively after a fixed number of turns without progress. EnterpriseBench: case material split into public, gated-on-request (revealed one piece at a time only when the question targets that dimension) and internal-only (never quoted); say "the case does not provide it" rather than invent; redirect data-hoarding; small hints only when stuck. Lost in Multi-Turn [2505.06120]: the user holds all shards, reveals at most one per turn, picks the one that fits the assistant's question, rephrases without changing content.
- **Multi-round user agent** (Terminal-Universe [2609.04148]): private requirement ledger; reads a private verifier result; after a fail describes only the user-visible symptom (never tests or tracebacks); after a pass adds a compatible extension or, if allowed, a conflicting replacement that quotes what it supersedes; up to 6 follow-ups.
- **Behaviour modules, not prompt text** (Non-Collaborative Users): four types from marketing research and real logs: unavailable service, tangential (four dialogue acts plus complaints if ignored), impatience (triggered by failures or delay, probability rising per trigger, cynical after the first outburst), incomplete (style transfer from 5 real brief messages or random truncation with the tracker marking truncated info unsent). A prompt-only simulator barely lowers agent success; the modular one does; humans preferred its realism about 70% of the time.
- **Profiles from real conversations** (SimulatorArena [2510.05444]): inherent knowledge (math concepts labelled knows well / partial / struggling / not introduced; document preferences) plus message style (attributes discovered contrastively over real vs simulated conversations for 10 iterations) plus a length range from the real user's min/max rounded outward to multiples of 5. Best profile depends on task (interaction style only for tutoring; full profile for documents). More constraints lower each attribute's fulfilment.
- **Relationship visibility table** (SOTOPIA [2310.11667]): family, friend, romantic see all but secrets; acquaintances see name, occupation, pronouns, public info; strangers see nothing. Each side has a private goal with one built-in disagreement.
- **Persona generation** (Concordia): facts plus randomised traits (Big Five) plus scenario → backstory → formative memories at several ages; trait fidelity is not validated.
- **Ask and refuse as actions.** OSWorld 2.0 has a bounded-knowledge ASK_USER channel; UI-TARS has a `CallUser` terminal action; BrowserGym has `send_message_to_user` and `report_infeasible`.

**Known defects of simulated users.** Too verbose (SimulatorArena: zero-shot CoT users write 89.5 words per message vs humans 15.5; length control brings 21.4), too cooperative and polite (Lost in Simulation: questions in 18.8% of turns vs 9.8% for humans; politeness 39.2% vs 19.9%), and they shift failure attribution to the agent (48.9% agent-caused in simulation vs 24.5% with humans; user-caused 40% vs 62.2%). Humans can also be more helpful than the simulator: UserRL's real users gave cooperative hints (Telepathy 0.56 → 0.78 for the 8B).

**Adoption.** F2 `ActorSpec{traits, backstory, memories, known_info: public | gated(dimension) | internal, style, length_range, behaviour_modules}` with `observe/act`. G8 for user-in-the-loop tasks becomes mechanical: every oracle-needed fact is a shard in the public or gated sets, and the tracker must be able to deliver it. Behaviour mode is a knob (B6.3). Priority P0 for provenance (B1.8), P1 for the rest.

### B1.8 Validating simulators and simulated Worlds

**Probes.**
- **Simulator swap** (Lost in Simulation [2601.17087]): same agent (GPT-4o), four user LLMs; success moves from 67.0% (Sonnet 3.7) to 75.9% (Sonnet 4.5); a GPT-4o user (same family as the agent) gave the lowest variance, 67.8 ± 1.2. A one-line "limit politeness" tweak moved 11 of 18 tasks to another difficulty bin and dropped success 50.0 → 46.7.
- **Human calibration by bin**: ECE_Human–LLM = Σ_i w_i |s_i(human) − s_i(LLM)| over difficulty bins, w_i = share of human completions in bin i. US humans: agent success 45.2%, ECE 15.1. Simulation said 0% in the hardest bin where humans reached 30.8%, and 60% in moderate bins where humans reached 39.0%. SAE 50.6% (ECE 11.7) vs AAVE 39.4% (ECE 20.3), gap growing with age (19 points at 55+); ECE 18.9 for India.
- **Downstream alignment** (SimulatorArena): Spearman between rater scores on simulated conversations and human ratings on real ones at instance, intermediate (model × difficulty) and system level; best intermediate above 0.7, about 26% above vanilla prompting; about 3% of human-evaluation cost ($0.02–0.17 vs $5.33–6.50 per conversation).
- **Human-likeness** (MirrorBench [2601.08118]): lexical diversity (MATTR window 50, Yule's K, HD-D) z-scored against humans of the same dataset; judge metrics calibrated with Human–Human and Proxy–Proxy controls, PI = clip((raw − μ_PP)/(μ_HH − μ_PP), 0, 1). Realism and diversity diverge. Pairwise judges prefer their own family (PI flips sign by judge family). Without a goal, realism collapses (GTEval 0.554 → 0.156); a topic-only goal sounds human but drifts off task.
- **Replay in the real system** (Trace2Env W2R): run the agent in the simulator, then replay the induced actions in the real env; consistency ratio CR = W2R / Real. Direct prompting on ALFWorld: 97% in sim, 3% replayed (CR 0.032); Trace2Env 88% / 85% (CR 0.914).
- **Intrinsic ranking** (WebDreamer): at states from gold trajectories, does the simulator let a scorer rank the gold action above deviations whose full trajectories failed (deviations that also succeed are dropped to avoid false negatives); task-level accuracy correlates with live success at Pearson 0.8455.
- **Cheap discriminator** (survey [2606.12191]): an LLM tries to tell real from synthetic ("Web Turing Score").

**Adoption.** Record user-simulator model id and prompt version in `SeedBundle` provenance; key `DifficultyRecord` by (task, student, checkpoint, user-simulator id). World-loop acceptance for any simulator change: real-vs-sim score shift within noise where a real reference exists, CR ≥ about 0.9 as a starting bar for W2R, and realism and alignment on a small human anchor must not fall. A simulator update that raises task success but lowers realism is rejected. Priority P0 (keying and provenance), P1 (probes).

### B1.9 Seeding from customer data: fidelity and privacy gates

**Mechanisms (only needed when a customer-data source exists).**
- TabularARGN [2501.12012]: fix the value domain first (rare-category merge, outlier clipping, numeric encoding as discrete, binned or digit sub-columns); a shallow any-order autoregressive model with per-batch permutation masking gives conditional generation on any column subset for free, so a seeder can plant the anomalous row a task needs. Two-table setup generates parent, then child sequence (sequence length sampled at step one plus a counting index). Fidelity: univariate and bivariate accuracy = ½(1 − L1) over binned histograms; sequence coherence over (step, next step) pairs. Flat accuracy 0.92–0.98; sequential 0.79–0.99; trains Adult in about 2.5 min on CPU. Limit: one parent→child level; no business-rule validity, so G1 must run after it.
- Privacy audit [2508.06647]: do not trust a DCR mean. Integrate CDF(DCR train/syn) − CDF(DCR train/test) up to the 0.98 point of the train/test CDF; positive means overfitting or leakage. Merge categories seen fewer than about 8 times (a 5K-unique `emp_title` column in 10K rows broke every method). Run shadow-model membership inference (1,000 shadow models) on the most vulnerable records (k-NN outliers). Without protection best attack AUC 0.635; value protection brings it to about 0.501; DP ε = 1.0 to about 0.50–0.52.
- AIM [2201.12677]: workload-aware DP synthesis for marginal queries; selects marginals by expected error reduction at current noise, anneals budget, gives per-marginal L1 error bounds from the last round only. Beats PrivMRF 1.3× on all-3way on average. Discrete, single-table.
- DP synthetic text [2210.14348]: control codes from record attributes, DP-SGD fine-tune (ε = 4), sample in original category proportions. Canary audit: inject 5 fake-PII records at 1, 10, 100 repetitions; at 100 repetitions 80% leak verbatim without DP and 0% with DP; one exposure can already rank near the top. DP erases anything rarer than about 0.05% of records, so plant rare edge cases synthetically. Use per-person privacy units when one person writes many records.
- OpenCUA [2508.09123] privacy pipeline for recorded traces: nothing leaves the machine without manual upload; GPT-4o classifies sensitivity (over-flags, e.g. an empty password field); humans decide.

**Adoption.** §2(b) enterprise ingest and F2's "fidelity gate when from customer data": (1) rare-value merge at about 8; (2) DCR-CDF integral ≤ 0; (3) shadow-model MIA on the most vulnerable records near AUC 0.5; (4) canary audit for any text pipeline (0 verbatim hits, no top perplexity rank); all before any swarm model sees the seed. Priority P2 (P1 for the canary audit in any redaction pipeline).

### B1.10 Minor World tips
- Observation format is a knob and a harness setting to freeze: UIA tree marks plus pixel detectors beat either alone by 15–57% (WindowsAgentArena); Set-of-Mark ids should not clash with on-screen content [2310.11441].
- Fix tool-description order and retrieval in the harness and record them: shuffling order moved accuracy 20.5 → 22.7; removing RAG (all 158 tools in context) cut it 20.5 → 15.5 (OSWorld-MCP).
- Disguise canonical scenarios (Prisoner's Dilemma) against contamination (Concordia).
- Equation-based counterpart policies give cheap deterministic multi-agent pressure (Beer Game q_t = d_t + α(I* − I_t) + βB_t).
- Real environments can be worse training grounds than simulated ones when they hit dead ends (empty searches, login walls, homogeneous account states): UI-Simulator's pipeline in the real env scored 4.31 vs 6.28 simulated. Seed real Worlds richly.
- Ground simulated counterparties in retrieved real facts and only answer contacts that exist (Vending-Bench suppliers).
- Model websites or workflows as finite-state machines where possible so trajectories can be enumerated (survey pointer to AutoWebWorld).
- Generated Apps and games should expose explicit state variables for verifiers (UI-TARS-2 games with JS verifiers reading score, level, lives).

## B2 Task proposal and sources

### B2.1 Template + seeded parameters + noise rows + state check + teardown (oracle by construction for state Worlds)

**How it works (AndroidWorld [2405.14573]).**
- Write tasks are `TaskEval` classes with four methods: `generate_random_params` (seeded), `initialize_task` (clear the relevant DB, insert goal rows plus **noise rows** that must not be touched), `is_successful` (read the App's SQLite, files or settings; return a float), `teardown`.
- Read-only tasks are data: prompt template, `relevant_state` (goal rows), **`exclusion_conditions`** so random noise cannot change the answer (no noise event on the asked date), `success_criteria` built from a few transforms (count, sum, identity) with a match type, and lists of possible parameter values. The answer is a transform over relevant state, so the oracle is free.
- Composite tasks sum subtask checks divided by count (create a note and SMS it: 0.5 each).
- Validators are shared across Apps (row-exists, file-exists).
- New read-only tasks were drafted by an LLM as protobufs and human-reviewed. Defects seen: missing or hallucinated fields, incompatible parameter generation, unused parameters, non-specific prompts; more complex structure gave more defects; editing drafts was still faster than writing from scratch.
- 116 templates, practically unlimited instances; 6 volunteers found over 30 task bugs.

**Failure modes.** Checks look for goal-row presence; deleting noise rows is penalised only where the check says so. Difficulty per seed is uncontrolled (B6.2).

**Adoption.** F2/F4: the quadruple maps onto `SeedBundle` (params, initialiser, distractors) and `TaskContract.start/grade`. Make `exclusion_conditions` a required field of read-only `TaskProposal`s ("the answer is unique under the seeded distractors"); answer = transform over relevant state is the cheap F4 oracle; "noise rows unchanged" is the default collateral check; the defect list becomes an F3 format-gate checklist. Priority P1.

### B2.2 Combinatorial construction from banks of atomic hidden facts

**UserBench [2507.22034].**
1. Preference bank: 5 travel aspects, 24 categories, 82 preferences, 902 implicit expressions, each validated to (i) not state the preference and (ii) still imply it unambiguously ("my schedule is packed tight" → direct flights).
2. Scenario = random combination of preferences across aspects; tier by preferences per aspect (Travel-22 easy … Travel-444 hard).
3. Per scenario, per aspect: 1 best option (meets all, cheapest), 3 correct, 10 wrong (violate at least one preference), 5 noise; search returns options only if arguments match the ground-truth query.
4. Grading by construction: best 1.0, correct 0.8, else 0; exact option ids, no judge. 4K+ scenarios (2,651 train, 417 test).

**SOTOPIA template route [2310.11667].** Fill slot templates from structured data (CraigslistBargains items and prices; seller target = list price ÷ (1 + markup), markup ~ Exponential(rate 0.5)); the LLM only rewrites for fluency. 30 template scenarios needed few edits; 60 "inspiration" scenarios needed heavy editing. Discard scenarios whose constraint admits fewer than 5 character pairs.

**Adoption.** F2/F3/F4 pattern for read-and-recommend tasks: hidden preferences as latent traits in `SeedBundle`, best/correct/wrong/noise rows seeded in the App, oracle = the unique best row, answer check = exact id, G8 guaranteed by the reveal rule. Prefer template-fill from exogenous structured data over free generation. Priority P1.

### B2.3 Traces-to-tasks when the system can be rebuilt (Terminal-Universe)

**Pipeline [2609.04148].**
- Seeds: 359,593 public trajectories. Keep a trace if the end workspace has at least 5 files and 100 lines; exclude benchmark-derived corpora, view-only traces and unsupported formats; dedup SWE traces by repo + base commit + problem statement; drop SWE-bench Verified repos.
- **Stage 1, deterministic replay.** Normalise read/write/edit into one event stream. For each touched path take the **earliest observed content** (before the agent's first change); exclude agent-created files; record the agent's changes separately. The result is partial.
- **Stage 2, agentic completion under "solvable, not solved".** Create missing files, complete partial ones, configure dependencies; never implement the target, never add tests for it, no comments locating the solution; prefer source and data over docs; network only to restore dependencies. Manual check of 30: 22 added only task-relevant support, 8 added unnecessary code, 0 contained a solution.
- **Stage 3, sufficiency judge.** Read-only agent returns {label, confidence, reason, present, missing_critical}.
- **Re-querying.** (1) Intent Recovery: from the first substantive user request write a 2–4 sentence objective with success criteria, constraints, prohibitions; never inject agent-only evidence. (2) Single-WS: an explorer proposes 10 candidates → 5 with different focus, rejecting undeterminable, docs-only, trivial, flaky, external-service, repo-independent or solution-revealing tasks; one sampled per environment. (3) Cross-WS: profile each workspace, TF-IDF neighbours within a language, an LLM labels pairs, keep dependency pairs (each workspace in at most one pair); the task must need at least 3 concrete facts found only in a read-only reference and may not state them; the verifier checks the reference is unchanged. (4) Multi-Round with the user agent of B1.7.
- **Verifier** (B3.1): red on start for target tests, green for preservation, written before any solution.
- **Funnel.** 359,593 traces → 68,263 reconstructed → sufficiency on 38,294 terminal (40.2% after replay → 93.5% after completion) and 1,900 SWE (20.1% → 77.1%) → **37,273 sufficient environments**. Completion grows median files 2 → 13 and lines 60 → 539. Teacher pass@1 Single-WS 72.3%, Cross-WS 49.2%; yields 72% and 49%.
- **Evidence.** Re-solving recovered intents with a teacher gave 52.1 avg vs SFT on the source trajectories 36.7 (below the 47.0 base). Replay-only 48.7 vs replay + completion 52.9 (±3.5 vs ±1.4). **Budget axis at fixed ~35K records: 2× environments 56.0 vs 2× queries per env 53.8 vs 2× solutions per query 53.9 (base 53.2).**

**Failure modes.** One teacher writes tasks, solutions and tests (shared blind spots); completion invents plausible but unreal context; network-enabled containers can leak; coverage bounded by the trace distribution; Intent Recovery tasks have no verifier.

**Adoption.** §3.4 traces source, with interfaces: `rebuild(trace) → SeedBundle + withheld_changes` (for state Worlds, pre-write row versions of every entity read or written), `complete(SeedBundle, intent)` under the no-solution contract, `sufficiency(SeedBundle, intent) → {label, present, missing_critical}` as a new F2 gate, `recover_intent(trace)` with "user-stated requirements only". Crucible's family separation fixes their single-teacher limit. Priority P0 (design and red-on-start rule), P1 (cross-World and multi-round).

### B2.4 Traces-to-tasks when the system is gone (Trace2Env worldbook)

**Offline construction [2610.06100]** (learning-free; built once by one model, frozen):
- Schemas: action schema (types, typed args; incoming actions canonicalised) and state schema with typed, mutability-tagged variables in four groups: `world` (durable), `session`, `surface` (visible now), `epistemic` (uncertain or conflicting beliefs).
- Grounded evidence: one record per aligned transition (action, outcome, facts, candidate effects with uncertainty, verbatim observation); up to 3 demos per action type chosen deterministically. Effects with ambiguous attribution or concurrent actions are withheld (confidence capped at 0.5).
- Induced abstractions: transition rules, state constraints, observation contracts, conventions, proposed and reviewed against **success–failure contrasts** (a contrast suggests a missing precondition; a counterexample narrows a rule); labelled supported / tentative / conflicted; only supported rules with confidence ≥ 0.55 compile to executable.
- Provenance links every abstraction to episodes and turns.

**Online step.** Workspace = immutable worldbook + mutable partial episode state + episodic memory. A missing state entry means unknown, not absent; worldbook examples never seed episode facts. Retrieve up to 5 evidence records and 2 demos; a deterministic applicability gate tags each supporting / uncertain / format-only and masks foreign values in the latter two; the world-model agent (≤8 read-only requests, ≤20 calls) previews effects on a copy and proposes typed effects plus an observation; the validation gate of B1.4 accepts or falls back.

**Numbers.** Next-observation score 75.94 vs direct prompting 69.51 (GPT backbone); best in 11 of 14 env–backbone pairs; a worldbook built by one model works when run by another. Ablations (Terminal): evidence given abstraction +4.26; abstraction alone −1.12; removing the adaptive loop −1.79 (no loop keeps about 80% of the gain at 29% of cost). Construction-disjoint gains persist but are much smaller than same-task overlap (+17.86 Terminal). Cost per record $0.21 (GPT-5.6-sol) or $0.03 (DeepSeek-V4.1-Flash). Few rules compile (Terminal 3 of 12).

**Adoption.** Use the worldbook as a **specification source for F1**: compile schemas into `AppSpec` (actions = tools, state = tables), supported rules into `policies_as_code`, evidence turns into tool-test fixtures, and keep any residual LLM-simulated behaviour as a lower App tier that never carries reward. Cross-fit construction (task-grouped folds) when evaluating on the same traces. Priority P1.

### B2.5 Recording real human traces

- **OpenCUA [2508.09123]**: annotators record on their own machines (screen video, input events, accessibility trees, HTML); local review before manual upload; tasks must exceed 15 steps; each rated rejected/ok/good/excellent; Ubuntu data kept disjoint from OSWorld. A reflector marks incorrect or redundant steps (excluded from loss); a summariser rewrites vague goals into precise ones and scores alignment, efficiency, difficulty. Keyframes are taken **before the mouse's pre-movement** to avoid leaking the target position. 22,625 tasks, 634 annotators, about 10 tasks per annotator-hour, about $20K annotation + $0.6 per task CoT, about $32K total.
- **UI-TARS [2501.12326]** online bootstrapping per round: run the model on hundreds of VMs, filter by rules (drop actions that change nothing), a VLM score and human marking of the **first erroneous step** (keep only the valid prefix); reflection data pairs the first error with a correction and with a "recover after the error stayed" step; DPO on (error, correction) pairs.
- **UI-TARS-2 [2509.02544]** flywheel: validator V(s) ∈ {0,1}; V = 1 → SFT, V = 0 → continual pretraining; never discarded. In-situ annotation with think-aloud audio; interactive annotation where the annotator accepts or overrides the model's proposed action live (on-policy data).
- **MCPWorld** records a human doing every task (DuckTrack) and derives difficulty from human step counts.

**Adoption.** §3.4 traces pipeline order: record → local review → privacy classify → human check → reflector labels → goal rewrite and score → **rebuild the end state in the World and re-solve** (the step OpenCUA lacks; it keeps traces as SFT data). The first-error step is the late-start state for a ladder (B6.4). Priority P1.

### B2.6 Explore first, then label (act-then-label)

- **UI-Simulator step-wise task control [2510.14969]**: the teacher proposes a high-level control from the initial state, acts under it, checks Done, proposes the next control from the current state and the previous control, decides when to stop; 3–8 controls per proposal by site. Then summarise the rollout into an instruction, rewrite thoughts to fit it, and filter trajectories that target invalid elements, change nothing, or whose stated action differs from the taken one. Without task control: WebArena 6.28 → 1.72, AndroidWorld 8.6 → 5.2, PCA effective dimension at 90% variance 153 → 118 (the teacher keeps clicking the same 1–2 elements). Generation in three steps (overview → rich draft → structured render) beat single-step (4.06 vs 6.28).
- **WebDreamer random walks [2411.06559]**: click-heavy random walks biased toward newly appeared elements, before/after screenshots, a VLM describes the functional change; 3.16M transitions over 1.25M URLs; filtered against test queries.
- **WildAWM sketch agent**: explores a live World and proposes at most 4 sketches, each needing CLI, MCP and web, each indispensable, including writing and running a script over a real user-visible source whose result drives a later decision.
- **MCPMark co-design loop**: explore under a seed topic → evolve (B6.3 operators) → draft verifier → expert solves with an agent → perturb end state → iterate.
- **OSWorld 2.0** found three defects in LLM-synthesised proposals: unrealistic artifacts, shallow workflows that glue unrelated operations, and hack risk from unanticipated alternative solutions; it uses them only as idea aids.

**Adoption.** F3: propose sub-goals from the current World slice during exploratory oracle construction, then label; retrospective labels must still pass F4–F6. Random walks over Crucible's own coded Apps can produce tool-test fixtures. Priority P1 (task control), P2 (random walks).

### B2.7 Obfuscation, multi-hop and per-site function synthesis (UI-TARS-2)

- **Multi-condition obfuscation**: extract an entity's attributes (Wikipedia), LLM-score each attribute's distinctiveness, delete the most revealing ones, rewrite the rest abstractly.
- **Multi-hop chains**: follow real hyperlinks; obfuscate each hop; embed each hop's answer in the next question.
- **Filter**: drop instances solvable from prior knowledge or one search.
- **Per-website synthesis**: drop inaccessible, login-gated and trivial sites; a VLM extracts each site's core functions; synthesise single-page tasks, merge prerequisite subtasks, refine for objectivity (690 sites).
- **Generated games**: LLM-written HTML5 games that keep core mechanics and expose explicit state; a JS verifier reads runtime variables; a curriculum raises game difficulty.

**Adoption.** Knobs with measurable dials: number of most-distinctive attributes removed (obfuscation) and hop count (linkage). The prior-knowledge and single-search filters are the G9 instruction-only and budget probes. Priority P1.

### B2.8 Generation along the tool dependency graph, and the concrete→fuzzy rewrite (MCP-Bench)

**How it works [2508.20453].** o4-mini analyses inherent dependencies (one tool's output feeds another's input) and scenario dependencies (result picks the next tool, cross-validation, parallel merge, loops), records them in `dependency_analysis`, and writes one maximal task using all given servers; data rules: self-contained, no external files or URLs, concrete values, **relative dates only**. A "fuzzing" step rewrites it as a conversational request with no tool names or steps, keeping every number; the grader gets the concrete version. Each task gets 10 distractor servers (100+ tools).

**Failure modes.** The same model synthesises, filters (LLM solvability ≥ 9.0, utility ≥ 5.0) and judges; no executed ground truth; scores compress (0.43–0.75).

**Adoption.** F1 `tool_graph` edges (dataflow, precondition) from the dependency prompt; F3 sampling along graph paths. Concrete→fuzzy is an obfuscation/hint knob with a ladder (concrete steps → named tools → goal only), applied after the oracle exists, with G8 confirming the fuzzy text still carries every exact value. Do not copy the LLM-scored solvability gate. Priority P2.

### B2.9 LLM task proposal grounded on a sampled start state (ZeroGUI)

**Prompt design [2505.23762].** A screenshot of a randomly sampled initial state + exemplar tasks for the App; ask for N candidates at once (10 for OSWorld, 5 for AndroidLab) that are achievable from that state, similar to but not copies of the exemplars, covering different aspects, objectively evaluable, at least 3 steps, spread in difficulty, plus a subset of deliberately **infeasible** tasks whose correct answer is FAIL. 4,000+ Ubuntu and 225 Android tasks; 725 and 175 used. Ablation (Daily domain SR): full 27.2; no exemplars 22.3; single candidate 24.1; no infeasible 24.0.

**Failure mode.** Exemplars were the benchmark's test tasks, so the generator copies that distribution.

**Adoption.** F3 matches §3.5's grounding rule; replace exemplars with the K nearest tasks in the cell. Priority P1.

### B2.10 Infeasible and partly infeasible tasks

- WindowsAgentArena and OSWorld include infeasible tasks rewarded when the agent declares FAIL.
- ZeroGUI: without generated infeasible tasks, infeasibility detection halves (Daily infeasible SR 22.1 without vs 41.3 with).
- Gaia2 Ambiguity split: impossible, contradictory or multi-answer requests; the agent does the clear part and reports.
- Non-Collaborative Users "unavailable service": an LLM extends a goal with three sentences needing missing APIs or parameters.
- Hack (ARPO [2505.16282]): a harness that writes FAIL when the step limit is hit earns reward on infeasible tasks; their "OSWorld Hard" forbids it.

**Adoption.** `DomainSpec.infeasible_share` is justified. The harness must never emit any terminal action for the agent. Infeasible tasks need a check that the agent declared infeasible for the right reason (an answer check on the stated reason). `report_infeasible` is a first-class terminal tool. Priority P1.

### B2.11 Real-data mining and human authoring rules (what to keep when humans are in the loop)

- Online-Mind2Web [2504.01382]: re-validating 650 old Mind2Web tasks found **47% invalid or stale**; rewrite rather than drop when ambiguity or staleness is the only defect; commit to replace outdated tasks at the same difficulty.
- MCP-Atlas: prompts must be tool-dependent, stable over the window, unambiguous, natural, reasoning-intensive (indirect references, cross-source filtering), never naming servers, tools or parameters; a per-task tool menu of required tools plus same-category distractors (6–37, mean 15.2; 2–8 required); a reference trajectory (mean 9.8 calls) used only for solvability and diagnosis; a separate review for tool-name leakage.
- MCP-Universe admission: reject a task an LLM solves without MCP servers, or consistently solves with them within five retries.
- LiveMCPBench: 300 candidates → 95 tasks (32% yield) with careful humans.
- Gaia2 human authoring in a GUI with real-time DAG guardrails (connected, rooted at `send_message_to_agent`, only one branch carries user messages, each turn ends with `send_message_to_user`) cut annotation time about 5×.
- MirrorAPI request generation: scenario first at high temperature, then the call at low temperature.

### B2.12 Minor tips (sources)
- GTM grows a field taxonomy by prompting with random pairs of existing fields: a cheap F0 brainstorming method for `DomainSpec` axes, subject to lead review.
- OSWorld-MCP: tools generated from the very tasks they are evaluated on are a contamination pattern; derive tools from a different pool.
- Survey vocabulary for sources: task-driven (wrap static data), real-world-driven (projections of web/OS), de novo (from minimal seeds); "more freedom needs more verification logic". Pointers it lists: LOGIGEN forward deductive synthesis into SQLite, AgentScaler read/write DB abstraction, CLI-Gym environment inversion (break a working env).
- Concordia: disguise known games; log (component state, action) pairs to blame either the LLM or the spec.

## B3 Verifiers and gates

### B3.1 Red-on-start and initial unsatisfaction (strengthens G4)

- **Terminal-Universe**: one self-contained pytest file, at most 18 tests, 1–3 per requirement, public surfaces only, no network, randomness or wall-clock, no skip/xfail, never writing into the workspace. Expected values for new behaviour are computed independently (from shipped data or the query), **never by running the incomplete code**; preservation tests use current behaviour as baseline and may not reuse the repo's own suite. Run on the unmodified workspace: at least one target test fails, preservation tests pass, zero collection or infra errors; else iterate or reject (`query_unverifiable`, `workspace_broken`). The verified filter matters more for hard tasks (Cross-WS 53.2 all vs 55.4 verified; Single-WS 56.0 vs 56.4).
- **WildAWM initial unsatisfaction**: V(W₀, W₀, ∅) = 0, i.e. the verifier scores 0 on the untouched start state with an empty answer. Dropping this gate cut the ablation average 31.86 → 24.08 and in-domain 21.59 → 3.05.

**Adoption.** F5 rule for every `VerifierBundle`: written before oracle rollout; target checks fail on the start state; preservation (collateral baseline) checks pass on it; expectations never computed from the state or code under test. Priority P0.

### B3.2 Perturbation tests of the checker against the oracle end state

- **MCPMark**: after the expert solves a task, they hand-edit the final state into near-miss failures and edge cases and confirm the script flags both success and failure. Scripts average 209.8 lines per task.
- **Gaia2** two-tier verifier test: (a) unit tests that perturb oracle trajectories in ways known to keep or break validity; (b) 450 hand-labelled trajectories: ARE Verifier agreement 0.98, precision 0.99, recall 0.95 vs an in-context LLM-only verifier at 0.72 / 0.53 / 0.83 on the same Llama-3.3-70B; with Gemini 2.5 Pro or Claude Sonnet 3.7 as soft judge 0.96 / 0.98 / 0.89.
- **AndroidWorld** noise rows give a natural "touched distractor" perturbation.

**Adoption.** G5 negatives get an automatic version of the expert's hand edits on the oracle end state: wrong value, missing write, touched distractor, extra write. Keep a few hundred hand-labelled trajectories per lane in the anchor set with agreement, precision and recall. Priority P0.

### B3.3 Action-level write verification with causality and timing (Gaia2 ARE Verifier)

**Algorithm.**
1. Precheck: the multiset of tool names over agent writes equals the oracle's.
2. Topologically sort oracle actions; for each, find an unmatched agent write with the same tool; **hard-check** exact fields (IDs); **soft-check** free fields (message text) with an LLM given the task, both argument sets and tool-specific guidelines.
3. Causality: all oracle parents must already be matched to earlier agent actions; independent goals in any order.
4. Timing: for oracle actions with relative delay > 1 s, the agent action must land in [Δt − 5 s, Δt + 25 s] of its matched parent ("immediate" annotated +2 s).
5. Success iff every oracle action is matched. Multi-turn: verify each turn; online, a conditional event runs the verifier and triggers the next turn only on pass, so runs stop at divergence (saves rollout cost).
- Reads are free and unverified. Noise and Agent2Agent variants need no new annotation because the verifier checks writes, not traces.

**Failure modes.** Assumes no equivalent writes (Messages vs Chats fails), so tasks must pin preferences; the tool-count precheck fails harmless extra writes (a corrected mistake); a read-then-summarise privacy leak is caught only if the outgoing message fails the soft check; the soft judge was hacked in RL (B3.7).

**Adoption.** F5 `timing_checks` as windowed matching relative to the matched parent and causal order from the oracle DAG, combined with Crucible's state checks and collateral diff (Gaia2 lacks a full collateral check). Priority P0.

### B3.4 Checklist verifiers, partial credit and evaluator kinds

- **WildAWM**: V(W₀, W, a) = Π_i v_i(W₀, W, a), a product of N binary diff checks over start state, end state and answer, built from an SDK of diff primitives; RL reward r = min{1, V + (0.1/N) Σ v_i}; r = −1 on a format failure. Removing partial credit cost −2.25 in RL ablations.
- **OSWorld 2.0**: the end state is scored against an unordered set of checkpoints (mean 27.25 per task) so alternative paths pass; binary = all checkpoints. Model judges are only objective binary checklist items, each validated on correct and confusable-wrong states; judges supply 11.53% of total score and never more than 50% of any task. Card critique: order-free partial checkpoint credit invites checkpoint farming in RL; binary over long tasks is near-zero signal for a 35B student (binary is 0 beyond 163 human-minutes).
- **MCP-Universe** evaluator kinds: format (schema compliance), static (time-invariant answers), dynamic (recomputed from the source at grading time) plus state checks; a task passes only if all pass; AE = share passed. Format evaluators pass above 90% for non-reasoning models but 51–74% for o3 and Gemini.
- **TimeArena**: duration-weighted progress s_i = t_i / Σ t_j; efficiency vs an algorithmic oracle minimum time (greedy schedule, a heuristic, manually checked).
- **AndroidWorld** composite: subtask checks summed and divided by count.

**Adoption.** F5 milestones for Φ as duration- or cost-weighted binary checks; small partial credit (0.1/N) on top of a product verifier; cap any judge's share per task; inside a simulated World, "dynamic" ground truth becomes a query over the end state, so answers do not go stale per seed; report per-kind pass rates so format failures separate from skill. Priority P1.

### B3.5 Side effects and collateral

- **OSWorld 2.0** side-effect suite: planted fake secrets (canary credentials) checked for leaks, disk exhaustion, document integrity, privileged group membership, leftover processes, sandbox bypass, sudoers, xhost. Observed: 33% of tasks bypass the UI (Opus 4.7: 35 system-level changes); 14% read hidden browser state.
- **AndroidWorld**: "noise rows unchanged" is the natural collateral check.
- **Terminal-Universe Cross-WS**: the read-only reference must be unchanged.
- **WindowsAgentArena** is the negative example: single-property evaluators (one JSON key) with no collateral diff; continuous similarity rewards near-misses.
- **Vending-Bench**: valuing inventory at cost makes hoarding free; value at a discount or add a hoarding check.
- **SOTOPIA** secrets: grade by code (did the secret string or fact appear in outbound messages), not by an LLM (judge r 0.22 on Secret).

**Adoption.** F5 `collateral` defaults: distractor rows unchanged, reference Apps unchanged, planted canary credentials never leave, no privilege or process residue. Priority P1.

### B3.6 Judges: what they may see, how to validate them

**Inputs.**
- Exclude the policy's self-report. ZeroGUI judge ablation (precision / recall → downstream SR): last screenshot only 47.5 / 40.4 → 23.9; all screenshots 53.7 / 61.7 → 25.6; all screenshots + agent response 44.3 / 74.5 → 24.0; all screenshots + **unanimous** 4-vote, no response 61.5 / 51.1 → 27.2. False positives damage training more than false negatives. Judge success at the final state, not "achieved at some point".
- LiveMCPBench judge error classes: hallucinated completion accepted 45%, output completeness disagreement 30%, granularity mismatch 17.5%, other 7.5%.
- Online-Mind2Web WebJudge: key points from the task text (avg 3.6), per-screenshot relevance 1–5, keep screenshots ≥ θ = 3 (2 floods context, 4 drops evidence), final responses excluded. Agreement with humans: GPT-4o 83.6%, o4-mini 85.7%, WebJudge-7B 87.0%; baselines 66.9–79.5% and falling with trajectory length. On AgentRewardBench precision o4-mini 82.0% vs rule-based 83.8%; o4-mini is conservative (gap 10.5). Key-point-wise mode is better for ≤ 3 points, too strict beyond. Token cost 67K (CoT) or 120K (key-point-wise) per task. Rerun 3×: SD 1.1% average.
- WildAWM agent judge: read-only shell and vision on BEFORE/AFTER snapshots; must never accept agent-authored text as proof of an external fact; checks images independently; labels each failed check too-strict or genuine-miss.

**Atomic claims for read-only answers (MCP-Atlas).** One judge call per claim: fulfilled 1.0 / partial 0.5 / not 0.0 with justification and confidence; numeric tolerance (within 5%, ±1 pp for percentages, equivalent forms match); coverage = mean; pass at ≥ 0.75; trajectory-agnostic, so alternative paths are free. Three judges' mean pass 65.4 / 62.3 / 63.9; per-model range 2.1–4.6 pp; treat rank swaps inside that band as noise. Hack: an answer listing many candidate facts can cover claims; answers from memory pass if facts are known.

**Validation per dimension.**
- SOTOPIA GPT-4 vs humans (Pearson): Goal 0.71, Financial 0.62, Relationship 0.56, Believability 0.45, Knowledge 0.33, Social rules 0.33, Secret 0.22; on human-played episodes everything but Goal drops; judge rates higher when it disagrees; Llama-2-70B near useless (Goal 0.24). **Adding fine-grained score-range definitions made Social rules fall to −0.59.**
- EnterpriseBench consulting judge (different family than the policy): 20-case audit r = 0.71 (LLM 8.5% higher than humans); 500 transcripts × 3 annotators: Pearson 0.887, within-1 85.0%, human ICC(2,3) 0.892, quant within-1 only 67.4%; prompt reorder/simplify moves the score ≤ 0.12.
- LiveMCPBench: DeepSeek-V3 78.95–81.05% human agreement; DeepSeek-R1 58.95% (reasoning models were worse judges of long trajectories); LLM-generated key points often agreed with humans better than human ones; **majority vote of 10 judges 77.89%, below the best single judge**; "any judge agrees" ceiling 97.89%.
- StableToolBench: solvability judgment accuracy GPT-4-turbo 80%, Gemini 82%, Claude-2 71%, GPT-3.5 65%; answer judging 74%.
- MCP-Bench: shuffling rubric order 5× and averaging lowers cross-judge CV 16.8% → 15.1% and raises human agreement 1.24 → 1.43 of 2.
- UI-TARS-2 ORM (the policy model further trained to score success): F1 83.8 on 300 human-labelled traces with a relatively high false-positive rate; authors saw no substantial hacking on inspection. Self-judging risk.

**Adoption.** For every judge in F5/F10/F12: inputs exclude the policy's claims; acceptance needs per-dimension agreement on the anchor set (trust dimensions with r ≥ about 0.6), precision reported, and a cross-judge range; do not use LLM judges for norms or secrets; prefer one validated judge over a weak-judge vote; measure rubric changes rather than assume detail helps; use atomic claims derived from the oracle end state (not an annotator's answer) for read-only intents. Priority P0 (rules), P1 (claims).

### B3.7 Judge hacking and the counterfactual probe

- **Gaia2 hack in RL**: on Search (one write expected), the policy learned to stuff ever-longer template-like code strings into the final message, overwhelming the soft judge into false positives. Fix: a task-agnostic "style" soft check.
- **MCP-Universe**: o3 sometimes copied the answer-format block without doing the task, which satisfied format evaluators.
- **WildAWM gates on judges**: (a) **judge consistency**: programmatic verifier = LLM judge = agent judge on all 6 rollouts; dropping it cut the average 31.86 → 22.56 and in-domain 21.59 → 3.79 (the largest effect). (e) **Counterfactual probe**: a fresh session re-judges blind, then is told the opposite verdict ("an independent audit believes ..."); pass if the judge defends with cited evidence; fail if it caves by editing verifier scripts (edit detected by diff and reverted); route to review if the judge was in fact wrong. Dropping the probe cut 31.86 → 27.91. A probe failure lets a repair agent edit only verifier and rubric; the task is frozen.
- LiveMCPBench's main error class (narrated completion) is exactly the vacuous-verifier canary.

**Adoption.** New gate for any LLM or agent judge: counterfactual probe plus a style check on any soft judge used as reward, plus a G12 attacker on soft judges. Add a "narrates completion without doing it" trajectory to the canary suite. Cross-check executed verifiers with an independent judge during synthesis (judge consistency) as a defect detector, not as reward. Priority P0.

### B3.8 Trust boundaries and harness-side hacks

- The harness must not act for the agent: ARPO's FAIL substitution earned free reward.
- Dynamic evaluators that share the agent's data source accept a stale or wrong response for both (MCP-Universe).
- Hooks on "function X called with Y" can be satisfied through any path (MCPWorld).
- Tools that compute outputs from the gold answer leak it (GTM).
- Agent-authored text is never evidence of an external fact (WildAWM); final messages hallucinate success (Online-Mind2Web, ZeroGUI).
- UserRL's Persuade and Intention rewards come from the simulator's own judgement, so a policy may learn to please the simulator; keep simulator-judged rewards out of training tasks.
- Multi-choice answers invite shotgun guessing (UserBench: about +40% score with no rise in elicitation); score single-choice.
- Image-decisive steps (two products identical on specs, different only in photo) defeat text-only shortcuts (WildAWM).

### B3.9 Two-sided QA: hacks and false negatives, alternatives

- OSWorld 2.0 QA stack: a coding agent writes unit tests for the rubric over expected paths; two humans re-solve end to end; several frontier agents roll out to reveal unanticipated solutions. Two audits: **reward hacking** (exploiting files, APIs, browser storage, project metadata; e.g. a route checker that checked only waypoints accepted a driving route) and **false negatives** (rigid checks penalising harmless format).
- WildAWM labels each failed check too-strict vs genuine-miss; ≤ 2 fix rounds, then every affected gate re-runs.
- WebDreamer drops deviations whose trajectories also succeed, to avoid false negatives in its ranking test.
- Claim-level and unordered-checkpoint scoring make alternatives free; Gaia2's exact-write matching does not.

**Adoption.** F6: pair G12 (hack) with an explicit false-negative audit on frontier rollouts; G14 alternatives benefit from trajectory-agnostic checks. Priority P1.

### B3.10 Minor verifier tips
- Separate implicit failures (finished but wrong; above 50% for all models in MCPMark, above 80% for gpt-5-high and kimi-k2) from explicit ones (context overflow, turn limit, premature stop, malformed calls).
- Rule-based trace metrics (valid tool name, schema compliance, execution success) are cheap but do not separate models (above 95% for nearly all in MCP-Bench).
- Two-reviewer unanimity for tool admission (OSWorld-MCP).
- Fail an episode after k consecutive invalid actions and bound waits (TimeArena: 5).
- Keep the reference trajectory for solvability and diagnosis only, never for scoring (MCP-Atlas).

## B4 Intent, validity, communicability and leakage

### B4.1 Shortcut and partial-input agents (G9, F10 null probes)

- **No-tools admission** (MCP-Universe): reject a task an LLM solves without the World.
- **Prior-knowledge and single-search filters** (UI-TARS-2).
- **Search-only agent** (Online-Mind2Web): one Google query, click one result, check for the answer. On 100 WebVoyager tasks it solved 51%; on the new benchmark 22% (easy about 50%, medium 18%, hard 3%).
- **Instruction-only checks** in Cross-WS: queries that state the hidden facts are rejected.

**Adoption.** Keep a small library of domain-specific trivial agents (search-only, read-only, answer-from-text, no-World) in F6 and gate on their success, reported by difficulty. Priority P1.

### B4.2 Oracle-input ablations: which part of the skill does the task exercise

- BrowseComp-Plus: with all evidence in context, gpt-4.1 reaches 93.49% and Qwen3-32B 83.25% (6% context overflow), while with search Qwen3-32B scores 3.5–10.4, so the open model's gap is interleaved search, not reading. Small open models under-call tools even when prompted (Qwen3-32B fewer than 1 call per query).
- Set-of-Mark: ground-truth masks instead of predicted add +14.5 mIoU; separate perception from decision with oracle regions.

**Adoption.** F10 probe for read-heavy families: "oracle evidence given". If the student passes with evidence and fails without, the IntentCard says the task exercises search. For the 35B, a "≥ 1 tool call" floor and evidence recall are cheap trace features. Priority P2.

### B4.3 Did success use the intended skill

- WildAWM: successful rollouts must use every declared interface.
- OSWorld-MCP Tool Invocation Rate: TIR = (n_t + n_g)/(N_t + N_g), counting successes that used a tool on tool-beneficial tasks and used none on the rest; best only 36.3% (Claude 4 Sonnet); Qwen2.5-VL-72B 10.9%.
- UserBench "preference elicited" (under 30% for most models): a pass with low elicitation is a lucky pass.
- OSWorld 2.0 exposure attribution: per phenomenon tag, Handled / Blocked / Untested, so a tag is blamed only when the agent reached it; hidden-state phenomena were the main blockers.
- TimeArena: necessary vs unnecessary waits separate "exploits parallelism" from luck (over half of valid actions were waits, mostly unnecessary).

**Adoption.** F7 `BehaviourSignature` fields: interfaces used, efficient path available/taken, elicitation count, necessary vs unnecessary waits; F10 uses them as on-intent signals and exposure attribution as the "did the failure happen at the intended skill" primitive. Priority P1.

### B4.4 Simulator- and mode-sensitivity probes

- Lost in Simulation: if p̂ leaves the band when only the user model changes, the task measures the simulator.
- Non-Collaborative Users: if p̂ moves a lot between behaviour modes, the card should say the task tests dialogue robustness.
- Gaia2: time mode (paused vs real latency vs instant) changes success drastically.

**Adoption.** F10 adds a user-simulator swap probe for tasks with a user in the loop, and G8 is checked against two simulator families. Priority P1.

### B4.5 Communicability (G8) and transformation gates

- Shards and gated facts make G8 mechanical (B1.7).
- Lost in Multi-Turn sharding gate: simulate 10 Full, 10 Concat (shards as bullets in one turn), 10 Shuffle-concat; keep if both reach ≥ 0.8 × Full. Concat kept 95.1% of Full on average (86–92% for smaller models), so the multi-turn loss is not information loss.
- MCP-Bench fuzzy rewrites must keep every numeric value.

**Adoption.** Gate any transformation (obfuscation, fuzzing, persona, sharding) by executed performance relative to the original, not by LLM opinion. Priority P1.

### B4.6 Fault sides and failure taxonomies

- MCP-Atlas 11 modes with one primary mode, contributing modes and an `is_root_cause` flag, explicit boundary rules between confusable modes, diagnosis only on failed tasks. Tool family: malformed call, wrong tool, no tool use, failed error recovery. Cognitive family: misunderstanding, faulty synthesis, response misparsing, early termination, hallucinated fact, logical error, constraint violation. 63.3% of failures cognitive. Early termination is a signature of some models (36–43%); o3 Pro had no tool call in 40.1% of failures.
- MCPWorld: "insufficient MCP coverage" (48.42% of MCP-only failures) is a World defect.
- Lost in Simulation hand labels: agent / user / both / other.
- Vending-Bench dominant failure: assuming an async outcome happened (an order "arrived" on its expected date), then spiralling (escalation to non-existent support, emailing the FBI, typing tool calls as text for 1,300 messages); not a context-length effect (r = 0.167 with memory-full day).
- Non-Collaborative failure mechanisms: repeated doc re-fetch on unfulfillable requests, fabricated API results (Qwen3-235B), apology loops consuming budget, invented parameters under fragments.
- OpenCUA error study: missing task knowledge, pixel grounding, action repetition when an action has no visible effect, wrong termination.
- ComputerRL: "operational illusions" (believing an action succeeded).

**Adoption.** F12 first-level label: implicit vs explicit; second: MCP-Atlas modes with root cause; fault side adds `user_sim` and `world_bug` (coverage). Skill leaf "verify asynchronous outcome before acting" and a meltdown detector (repeated escalation, no tool calls) are cheap. Priority P1.

### B4.7 Leakage and contamination

- Tools derived from graded tasks (OSWorld-MCP); simulator conditioned on the gold answer (GTM).
- Trace-derived observations: choose keyframes before the user's pre-movement (OpenCUA); intent recovery keeps only what the user said (Terminal-Universe).
- Name leakage review of prompts (MCP-Atlas); hidden facts must not appear in Cross-WS queries.
- Decontamination: 13-gram against Terminal-Bench (Terminal-Universe); filter synthetic search queries that appear in test tasks (WebDreamer); cross-fit worldbooks (Trace2Env).
- Exposure audit: public–private gaps per model family (MCP-Atlas: −0.8 to +9.6 pp, OpenAI models clustered at +8.4), treated as a signal, not proof. BrowserGym saw GPT-4o's WebArena rise 23.5 → 31.4 between checkpoints and partly suspected contamination.
- Training on the benchmark: ZeroGUI's test-time stage trains on the benchmark's test tasks; ARPO, DART and ComputerRL train on OSWorld tasks or its domain.
- Public MBA casebooks are a likely contamination source (EnterpriseBench).

**Adoption.** G11 rules: no tool or simulator output computed from gold; user-stated-only intent; name-leak lint; n-gram decontamination; sealed benchmark split with a public–private gap report. Priority P1.

### B4.8 Minor intent tips
- Hint level is not monotone: TimeArena's self-plan prompting (infer dependencies, estimate durations, apply the greedy rule) cut GPT-4's cooking 3-task completion 40 → 0%.
- More turns or pass@k raise the maximum but not the mean in UserBench; reasoning effort helps mid and large models but not nano (MCPMark).
- Distinguish "ask" from "wait": track active vs passive elicitation.

## B5 Diversity and coverage

### B5.1 Measurement
- UI-Simulator: PCA effective dimension of task embeddings at 90% variance (153 with step-wise task control vs 118 without).
- WildAWM: Vendi score above CUA-Gym at every sample size; nearest-neighbour cosine distance 0.25 vs 0.20; coverage reached 134 domains and 91 task types without saturating.
- MirrorBench: lexical z-scores vs humans; realism and diversity diverge (on QULAC all proxies are less diverse than humans; on ClariQ some are more diverse).
- Survey: report correctness, diversity, complexity and fidelity per environment batch; correctness tooling is mature, the other three are preliminary. Add a **fidelity meter** (real-vs-synthetic discriminator, or W2R where a real reference exists) to §3.5's meters.

### B5.2 Structural levers with evidence
- **More distinct environments beat more queries or solutions per environment** (Terminal-Universe: 56.0 vs 53.8 / 53.9 at fixed records). Supports a per-World quota and rewarding World novelty over instruction novelty in F11.
- **Coverage pressure** (WildAWM usage histograms) and **persona rejection rules** (same occupation + domain, tool pattern, or arc).
- **Step-wise sub-goal control** during exploration (B2.6).
- **Several candidates per prompt, grounded on a sampled start state** (ZeroGUI: single candidate 24.1 vs 27.2).
- **Ready-made scenario axes**: OSWorld 2.0's ten phenomenon tags (cross-source reasoning 42.6%, visual-spatial precision 41.7%, implicit state 39.8%, multi-item tracking 39.8%, conflict disambiguation 36.1%, multimodal editing, tutorial following, dynamic environment 9.3%, streaming 5.6%, proactive interaction 5.6%); Gaia2's splits (Execution, Search, Ambiguity, Adaptability, Time) and Time taxonomy (time-based or event-based × one-off or recurrent); user behaviour modes; SOTOPIA's scenario × characters × relationship × partner policy.
- **Profiles sampled from a real pool** as exogenous randomness (SimulatorArena).
- **Cross-platform and cross-interface data** helps: OpenCUA cross-platform data raised OSWorld 9.8 → 18.5; UI-TARS-2 hybrid GUI/SDK RL beat GUI-only even on GUI-only tasks with half the per-interface data; browser RL transferred to OSWorld (43.0 → 47.5) and AndroidWorld (64.6 → 73.3).
- Gaia2 dropped composite "3+ capability" tasks as unnatural with unclear signal.

### B5.3 Dedup
- Near-duplicate = same core action + artifacts (WildAWM's LLM diversity gate, which picks sketches adding most topical variety against accepted and concurrently locked tasks, with no fixed quotas).
- Exact-match dedup and "identical request, different response" removal (MirrorAPI); tool-name dedup at cosine > 0.8 (GTM); repo + commit + statement and one-pair-per-workspace rules (Terminal-Universe); functional-duplicate tools dropped (OSWorld-MCP).
- Content-only rewrites (UI-Simulator-Grow "search running shoes" → "search slippers") should be flagged by F8 signatures as near-duplicates unless a knob changes.

### B5.4 Homogeneity warnings
- Simulated users are too cooperative and verbose (B1.7); GPT-family behaviour generators risk homogeneity (Non-Collaborative Users).
- Generators copy their exemplars (ZeroGUI).
- A synthetic pool can look rich and be shallow (WebVoyager, 51% search-only); measure the pool, not the generator's claims.
- Small pools do not transfer: ARPO in-domain 43.8 → 81.25 but OOD 55.2 → 56.3 (GRPO alone 52.08).
- LLM-only diversity judgement without an explicit grid (WildAWM) is weaker than the behaviour-signature grid.

**Adoption.** F11 quotas per World and per family; signatures as the dedup authority; phenomenon and Time taxonomies as `DomainSpec` axes; held-out families mandatory. Priority P1.

## B6 Difficulty and curriculum per student

### B6.1 Band and admission rules seen in the cards
- ARPO: admit if baseline pass@16 > 0 (128 tasks); this raises within-group reward variance. Looser than Crucible's k = 8 Wilson band.
- MCP-Universe: "solved consistently within five retries" = too easy.
- WildAWM: k = successes among 3 unhinted rollouts; k ≤ 1 hard, k = 2 medium, k = 3 too easy (sent back to be made harder); 3 hinted rollouts in parallel. Labels from 3 rollouts are noisy.
- DART: drop rollouts per task from 8 toward fewer once running success exceeds 0.6.
- SOTOPIA-hard: difficulty for a target model = (best model mean + 3σ) − (target mean − 3σ), clipped.
- Human or oracle reference length: Online-Mind2Web easy ≤ 5 steps, medium 6–10, hard ≥ 11; MCPWorld easy 0–5, medium 5–10, hard 10+; AndroidWorld step budget 2× human steps; efficiency E = agent steps / human steps.
- Cheap predictors: UI-Simulator teacher-forcing loss of the student on the teacher's actions, keep the 25th–75th percentile (below learned, above too hard or ambiguous); EnterpriseBench LLM difficulty annotation with explicit anchors (context length, depth, ambiguity, label space, steps) matched 2 experts at r = 0.96, MAE 0.47; survey: calibrate complexity by discriminative power across models, not length.

### B6.2 Variance: p̂ needs seeds and clock offsets, not just repeats
- AndroidWorld: three seeds give 27.6 / 26.3 / 33.2%; across-seed variance exceeds within-seed variance (Wilson intervals, p < 0.05 on two of five tasks). Cause is structural: a seed picking an off-screen category (needs horizontal scroll) or the "replace text" variant makes a task fail every time.
- OpenCUA: at temperature 0 with only the system date changed, Pass@1 20.1 vs Pass@16 38.6 at 50 steps.
- Lost in Multi-Turn: unreliability U90−10 about 50 points for every model; aptitude falls 16% but unreliability rises 112%; temperature 0 does not fix it.
- MCPMark: pass^4 about a third below pass@1 (gpt-5-medium 52.56 / 33.86).
- Vending-Bench: every model has runs that sell 0 units; report the minimum over at least 5 seeds.
- ZeroGUI: generated-task training raises pass@4 (coverage); test-time training raises all-pass@4 (consistency).

**Adoption.** F9 samples several seeds and clock offsets per task; seed parameters that change the behaviour signature become declared knobs or separate cells (F7); report pass^k beside pass@k; use k ≥ 8 with Wilson bounds for multi-turn. Priority P0.

### B6.3 Knobs with measured effects

| Knob | Mechanism | Measured effect |
|---|---|---|
| Retrieval breadth / tools | top-k, irrelevant servers, distractor tools | LiveMCPBench k 5 → 1: 78.95 → 64.21 (McNemar p = 0.02); k = 10 no gain. MCP-Universe 7 servers / 94 tools: 22.22 → 11.11. OSWorld-MCP: accuracy and TIR fall as tools per task rise |
| Disclosure pace | shard count, one per turn vs snowball vs recap | Full → Sharded −39% on average; any split into ≥ 2 shards causes the loss; Snowball recovers 15–20% of the gap |
| Linkage per facet | preferences per aspect | UserBench: drives difficulty more than number of aspects; removing 5 distractor options gives only small gains |
| User behaviour mode | collaborative, unavailable, tangential, impatient, incomplete | Qwen3-30B-A3B τ-bench 27.9 → 26.6 / 20.4 / 24.8 / 30.1; MultiWOZ 48.3 → 47.2 / 27.2 / 41.0 / 26.1; tangents worst (−29.1% relative avg); two behaviours at once drop even GPT-4.1-mini sharply |
| Noise | random tool failures, signature changes, spam events | Gaia2 Sonnet: none 31.2, low 35.0, medium 23.8, high 8.1 (not monotone at low) |
| Notification verbosity | low / medium / high | Gaia2 observability knob |
| Time mode | latency counted vs paused vs instant | GPT-5 (high) Time 0.0 → 34.4 in instant mode |
| Pressure, capital, latency | recurring fee, start balance, delivery delay | Vending-Bench: $100 start cuts sales; $5 fee ends all runs before day 100; $0 fee does not raise sales (idle loops) |
| Contention | resource count | TimeArena one pot/fryer/oven lowered completion and progress |
| Evolve operators | remove hints, hide information, longer inputs, more steps | MCPMark; record the operator |
| Obfuscation / hops | delete most distinctive attributes; link-walk hops | UI-TARS-2 |
| Cross-World hidden facts | ≥ 3 facts only in a read-only reference | teacher pass 72.3% → 49.2% |
| Observation format | a11y tree vs SoM screenshots | AndroidWorld M3A 30.6 vs 25.4 |

**Adoption.** Add `noise`, `pressure`, `latency`, `contention`, `user_behaviour`, `disclosure` and `time_mode` to `Knob.kind`. Do not assume `monotone: true` (noise and hint level were not monotone); measure. Each knob change must move the behaviour signature. Priority P1.

### B6.4 Ladders and injected positives for too-hard tasks
- **Inject a stored success into all-fail groups**: ARPO replaces one of 8 rollouts with a cached success (training reward 0.65 → 0.75; in-domain 68.8 → 81.25); DART's experience pool lifted 22 tasks from 0% to 46%. Crucible analogue: the F4 oracle trajectory executed in the World is the injected positive for below-band tasks.
- **Hinted rollouts** (WildAWM: synthesis SR 21.3% without hints vs 45.6% with; hint derived from the private `reason.md` by a separate call).
- **Late-start states** from the first wrong step of a student trace (UI-TARS).
- **Per-step model-pool mixing** to prove solvability of zero-solve candidates (ComputerRL: for unsolved tasks, pick a random strong model at each step) before declaring them defective.
- **SFT warm start**: WildAWM 35B without SFT −5.31; UserRL RL from raw Qwen3 plateaus early, and with SFT some gyms gain over 100%; WildAWM keeps at most 1 success per task from 4 teacher rollouts.
- **Entropulse** (ComputerRL): after RL plateaus (180 steps), SFT on the run's own successful rollouts (about 130k steps, LR 5e-6), then RL again: 42.0 → 41.5 → 45.8. Successful rollouts are a curriculum asset worth storing per task.

### B6.5 Controllers
- DART: dynamic rollout frequency; per-task horizon cap = longest past success (can block longer solutions); high-entropy step selection (top 80%); truncated importance weight C = 1. Ablation on 45 training tasks: 28.67 → 50.90 (dynamic rollout) → 66.11 (dynamic length) → 68.33 → 70.55 → 72.28; rollouts per task 8.0 → 5.0; without importance correction accuracy collapsed 55% → about 0 after step 60.
- WildAWM adaptive rollouts: per-task EMA multiplier κ ∈ [1, 3] (weight 0.8); add chunks of 8 rollouts until at least 2 distinct rewards; pick a group of G spanning lower and upper reward halves; if all rollouts are format failures, A = −K^(−1/2).
- UI-Simulator-Grow: each round, targets in the 25–75th loss percentile get content-only variants; replay the most representative old tasks (top row-sums of the instruction similarity matrix); reached full-data results with 66% of trajectories.

**Adoption.** F9/F11: publish p̂ continuously (F12 `p_hat_updates`) and move compute away from tasks above 0.6; horizon cap from oracle length as a budget check; the teacher-forcing-loss band as a rollout-free pre-filter before paid k = 4 sampling, after calibration against measured p̂. Priority P1.

### B6.6 Per-student evidence on Crucible's family
- WildAWM Qwen3.6-35B-A3B: base pass@1 5.7%, pass@4 18.4% on hard XAI tasks (Qwen3.5-4B 0.38%); after SFT + GRPO, 8-benchmark OOD average 42.84 → 53.42. Gains concentrate in the 2nd and 3rd difficulty quartiles (+19.9, +12.2), **none in the hardest quartile** (beyond the teacher). Interface switching comes from SFT (18.4% → 24.7%); RL adds long-horizon solving (tasks solved in 40–120 steps +17.0 vs +9.7 for SFT alone); RL without SFT reduces switching.
- MCP-Bench qwen3-30b-a3b 0.627 vs gpt-5 0.749; separation lies in planning (dependency awareness 0.22–0.65).
- Non-Collaborative Users: Qwen3-30B-A3B drops hardest under tangents and fragments.
- UserRL: training with a cheap open user (Qwen3-32B) transfers to GPT-4o evaluation; a GPT-4o user trains slightly faster. Raw scaling 4B → 32B adds little without interaction training.
- Reasoning effort as a difficulty confound: gpt-5-mini 8.27 → 30.32 (MCPMark); gpt-oss-20b 13.4 → 34.6 with search calls 1.9 → 23.9 (BrowseComp-Plus).

**Adoption.** Plan ladders and the 35B band around "gains stop at teacher reach"; record training-time and eval-time simulator ids per `DifficultyRecord`. Priority P0.

## B7 Self-improvement loops, guards, statistics and judge calibration

### B7.1 Loops seen in the cards
- **Human–agent co-design** (MCPMark): explore → evolve → verifier script → expert solves → perturb end state → iterate; expert feedback templates ("not verifiable enough", "explore deeper", "make subtasks integrated rather than unrelated"). 3–5 expert hours per task.
- **Gate-and-repair** (WildAWM): failures return with diagnostics for ≤ 2 fix rounds, then must re-pass every affected gate; a probe failure may edit only verifier and rubric; lessons feed the generator's skill library. Mean 2.5 revision rounds; fix counts per task: judge consistency 1.09, solvability 1.16, difficulty 0.39, probe 0.18, initial-unsat 0.05.
- **Data flywheels** (UI-TARS, UI-TARS-2): verified rollouts to SFT, failed to a lower-trust stage; task pool and model refreshed toward current weaknesses.
- **Target-and-vary** (UI-Simulator-Grow) with replay against forgetting.
- **Entropulse** (B6.4).

### B7.2 Guards with evidence
- **Same-family circularity** is the repeated limit: MCPMark's task agent drafts the checker; MCP-Bench's o4-mini synthesises, filters and judges; WildAWM's GLM-5.2 builds Worlds, tasks, judges and teaches; Terminal-Universe's single teacher writes tasks, solutions and tests; UI-TARS-2's ORM grades its own policy. Crucible's family separation is the fix each paper lists.
- **Family preference in judges** (MirrorBench PI flips by family) supports the different-family rule for user-simulator judges.
- **World-loop acceptance tests**: real-vs-sim agent scores (MirrorAPI); BrowserGym's ReproducibilityAgent replays a previous action sequence on the same seeds and diffs prompts to expose drift, which is the "frozen expected states" check after any World change; a reproducibility journal (benchmark, packages, commit, OS, time).
- **People-loop health**: MirrorBench realism with HH/PP calibration plus lexical z-scores, SimulatorArena alignment on a human anchor.
- **Canaries**: narrated-completion trajectory (LiveMCPBench), counterfactual probe (WildAWM), style check (Gaia2).
- **Never select on a learned reward with known false positives** (UI-TARS-2 ORM F1 0.84 with false positives still trained; plan forbids selecting on it).

### B7.3 Statistics used in the cards
- Wilson intervals across seeds (AndroidWorld); pass^k beside pass@k (MCPMark); A90 and U90−10 over ≥ 10 runs (Lost in Multi-Turn); ECE over difficulty bins and GEE with demographic controls (Lost in Simulation); bootstrap CIs with 10,000 resamples (MCP-Atlas) and trajectory-clustered paired bootstrap (Trace2Env); McNemar for paired task outcomes and Kendall τ-b for rank stability under a second judge (0.88, LiveMCPBench); Spearman at instance, intermediate and system levels (SimulatorArena); ICC(2,3) and within-1 rates (EnterpriseBench); standard error σ/√N (BrowserGym); cross-judge ranges (MCP-Atlas 2.1–4.6 pp); multi-seed SD ≤ 0.016 (MirrorBench); judge rerun 3× (Online-Mind2Web).

**Adoption.** F12 and the anchor set report per-dimension agreement, precision, cross-judge range and seed variance; F9 uses paired tests (McNemar) when comparing knob settings on the same tasks. Priority P1.

## B8 Pipeline architectures, costs, yields, scale; training results

### B8.1 Costs and yields

| Paper | Pipeline | Yield / cost |
|---|---|---|
| WildAWM | personas → day deltas → sketches → 7 artifacts → 6 rollouts → 3 judges → 5 gates | 341 tasks (273 hard, 68 medium), **53% acceptance**; per task 69.8M generator input tokens, 1.1M output; solving about 80 calls, 4.1M input tokens; 8 B200s for synthesis and training |
| Terminal-Universe | 359,593 traces → 37,273 environments → 31,977 SFT records (1.42B tokens) | yields 72% (Single-WS), 49% (Cross-WS); no $ given |
| OpenCUA | human recording + CoT synthesis | 22,625 tasks for about $32K |
| OSWorld 2.0 | human-built | 108 tasks; eval about $72/task (Opus 4.8 batched), $25.5 (GPT-5.5), $2.4–6.6 open models |
| MCPMark | expert + agents | 127 tasks; 3–5 expert hours each; eval $0.01–9.18/task |
| LiveMCPBench | human | 300 → 95 (32%) |
| BrowseComp-Plus | o3 evidence + humans | 1,266 → 830; 400+ annotation hours |
| Lost in Multi-Turn | sharding + simulation | 200,000+ conversations for about $5,000 |
| SimulatorArena | profile simulators | $0.02–0.17 per conversation (about 3% of human) |
| Trace2Env | worldbook simulator | $0.03–0.21 per record |
| UI-Simulator | LLM-simulated UI | $0.02–0.05 per web trajectory |
| BrowserGym | Claude-3.5-Sonnet sweep | $23–299 per benchmark |
| GTM | learned tool simulator | 6.3× faster search RL steps; 11× faster CUDA RL |
| DART | decoupled async RL | throughput 22.6 → 43.6 actions/min; env utilisation 12.2% → 67.7% |
| ARPO | 256 parallel VMs, batched vLLM inference | per-batch rollout 3 min at 8 envs vs 19 min at 256 envs (as the card states it); epoch 6 h → 1.2 h |
| WindowsAgentArena | one VM per worker | $0.384/h per worker; full run 6.5–82 min |

### B8.2 Training results on synthetic or semi-synthetic environments
- **WildAWM** (Qwen3.6-35B-A3B, Crucible's student): SFT on 100 tasks (≤ 1 GLM-5.2 success per task, 3 epochs, LR 5e-6), then GRPO on 208 tasks, ≤ 64 steps, 8 tasks × G = 4, LR 1e-6, KL 0, clip 0.2/0.28, 256 turns, 30 min per rollout, MoE router and vision encoder frozen. OOD average +10.58 (Online-Mind2Web 69.60 → 80.77, WebArena 53.54 → 67.26, ClawEval 38.00 → 54.67, OSWorld-Verified 33.16 → 49.22). Gate ablations on 4B SFT data: full 31.86; without judge consistency 22.56; initial unsat 24.08; difficulty 26.37; probe 27.91; solvability 29.26.
- **Terminal-Universe** (Qwen3.5-27B, SFT only): Terminal-Bench 2.0 41.6 → 52.8; TB 2.1 46.2 → 58.1; beats other synthesis corpora including a 35B-A3B one (47.6).
- **UserRL** (Qwen3 4B/8B): Equalized turn reward + discounted reward-to-go trajectory score best (8B 0.565 vs raw Qwen3-32B 0.313, GPT-4o 0.445); naive sparse turn rewards collapse.
- **ComputerRL** (AutoGLM-OS-9B): OSWorld 48.9; step-level GRPO (reward 1 for every formatted step of a successful trajectory), G = 16, KL 0.0003.
- **DART-GUI-7B**: 42.13% vs base 27.52% (+14.61) on OSWorld.
- **ARPO** (UI-TARS-1.5-7B): 23.5 → 29.9; OOD near zero.
- **ZeroGUI** (UI-TARS-7B-DPO): 17.7 → 20.2 (+2.5) with a VLM reward.
- **UI-TARS-2**: OSWorld 47.5, AndroidWorld 73.3; PPO beat GRPO; value pretraining fixed value estimates that were negatively correlated with reward; entropy often rose in GUI RL.
- **Non-Collaborative Users**: Qwen2.5-3B SFT on collaborative dialogues 91.6% collaborative but 61.2% on unavailable-service users; mixing non-collaborative data raised the cross-mode average 78.8 → 86.9.
- **UI-Simulator**: SFT in a simulated UI beat the same pipeline in the real env.

**Indicative comparison.** Execution-checked RL (DART +14.61; ComputerRL 48.9 absolute) far exceeded VLM-judged RL (ZeroGUI +2.5) on OSWorld; base models and action spaces differ, so only indicative.

## B9 Corrections and contradictions

### B9.1 Errors or caveats in papers
- **GTM [2512.04535]**: note 04 marked its numbers "not verified". The full text shows the 20K+ tools over about 300 domains are LLM-generated schemas, the quality judge is Qwen2.5-72B, and in search and retrieval training the simulator generates outputs from the task's ground-truth answer.
- **ZeroGUI**: part of the gain comes from test-time RL on the benchmark's own test tasks with the VLM reward; the reported benchmark score is contaminated.
- **ARPO**: the main table mixes trained and untrained OSWorld tasks; held-out transfer is about zero.
- **DART-GUI and ComputerRL**: train on OSWorld tasks or domain; ablations are on training tasks; no held-out App family.
- **OSWorld-MCP**: tools derived from the evaluated tasks' solutions can collapse a task to one call, so accuracy gains partly measure tool-to-task leakage.
- **MCP-Bench** and **StableToolBench** filter by LLM-judged solvability (StableToolBench drops about 30% of tasks, 1,100 → 765, by model opinion).
- **Gaia2**: the no-equivalent-writes assumption and tool-count precheck give false negatives for harmless extra writes; reads are unverified.
- **AndroidWorld**: goal-row presence checks do not penalise deleting noise rows unless written to.
- **Vending-Bench**: valuing inventory at cost leaves hoarding unpenalised.
- **TimeArena**: the greedy oracle is a heuristic, not a proven optimum.
- **AIM**: picking the tightest round invalidates its error bound; use the last round.
- **TabularARGN**: DCR share near 0.5 is necessary but not sufficient (its follow-up shows DCR means hide overfitting).
- **EnterpriseBench**: judge agreement differs by sample (20-case r = 0.71 vs 500-transcript Pearson 0.887); its two simulators are single scenarios, so effectively 2 tasks.
- **Online-Mind2Web**: its best-agreement judge (o4-mini) has low recall (success-rate gap 10.5).
- **SimulatorArena / MirrorBench**: human-likeness is not the same as eliciting the same agent behaviour.

### B9.2 Inconsistencies inside the cards
- The ZeroGUI card says executed checks "gave +14–20 on the same benchmark (DART, ComputerRL)", while the group-11 takeaway gives DART +14.6 and ComputerRL "up to 48.9% absolute". ComputerRL's numbers in its card are an absolute 48.9, an API-GUI ablation 11.2 → 26.2 and a training ablation 15.2 → 45.8, so "+14–20" is a loose summary [noted inconsistency].
- Lost in Simulation's lowest-variance user (GPT-4o, 67.8 ± 1.2) shares the agent's family; low variance there should not be read as fidelity.
- UserRL found real users more helpful than GPT-4o users, while Lost in Simulation found simulated users too cooperative; the direction of the sim–human gap depends on the domain and simulator.

### B9.3 Corrections and additions to `framework-plan.md`
1. **§3.2 `DifficultyRecord`** key: add user-simulator id and version and the time mode; record training-time vs eval-time simulator (Lost in Simulation, UserRL, Gaia2).
2. **§3.2 `Knob.kind`**: add `noise`, `pressure`, `latency`, `contention`, `user_behaviour`, `disclosure`, `time_mode`. `monotone` must be measured, not declared (Gaia2 noise, TimeArena hints).
3. **§3.2 `ToolSpec`**: add read/write type, role scope, `duration`, `occupies_actor`, `locks` (Gaia2, TimeArena).
4. **§3.1 G4**: strengthen to red-on-start for target checks and green-on-start for preservation, with expectations never computed from the state under test (Terminal-Universe, WildAWM).
5. **New gates**: counterfactual probe and style check for any LLM or agent judge (WildAWM, Gaia2); sufficiency for rebuilt start states (Terminal-Universe); simulator fidelity (W2R or real-vs-sim score shift) for any LLM-simulated App or person (Trace2Env, MirrorAPI).
6. **§3.1 G5**: generate end-state perturbation negatives automatically (MCPMark, Gaia2).
7. **§3.1 G8**: for user-in-the-loop tasks, check via shards and gated facts, against two simulator families.
8. **§3.6**: ARPO's pass@16 > 0 admission is looser than the k = 8 Wilson band; keep Crucible's. Add seed and clock-offset sampling to F9. The oracle trajectory is the injected positive for below-band tasks.
9. **§3.6**: per-student ladders should expect no gain beyond the teacher's frontier (WildAWM hardest quartile).
10. **§3.5 / F11**: quota per World; more environments beat more queries per environment (Terminal-Universe).
11. **§3.5 meters**: add a fidelity meter (survey).
12. **§3.7 F10**: add simulator-swap, oracle-evidence and shortcut-agent probes; fault sides add `user_sim`.
13. **§3.8 World loop**: acceptance adds real-vs-sim score parity, replay-agent drift checks, and people realism health; log validation-gate fallbacks as `world_bug` signals.
14. **§3.4 traces source**: require rebuild + re-solve; never train on the raw trace (36.7 vs 52.1).
15. **Avoid**: LLM-scored solvability gates, live remote services as World tiers, answer-conditioned simulated tools, simulator-judged rewards for training tasks, harness-emitted terminal actions.
