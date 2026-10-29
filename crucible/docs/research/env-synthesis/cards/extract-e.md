# Extract E: groups 09 and 10 (self-improvement guards, judge statistics, enterprise and web Worlds)

**Inputs.** `group-09.md` (29 papers, all read FULL; some result tables were images, so only numbers stated in the text were used for 2606.04455, 2410.10934, 2404.12272, 2609.12439, 2602.16179, 2506.13131, 2408.08435) and `group-10.md` (29 papers; 26 FULL; PARTIAL: 2408.06327 VisualAgentBench, 2401.13649 VisualWebArena, 2404.07972 OSWorld, whose bodies were read in full and long per-app appendices skimmed). No ABSTRACT-ONLY cards. Stage and gate names follow `framework-plan.md` §3 (F0–F12, G1–G14, the four loops of §3.8).

**What the two groups contribute.** Group 09 gives the guard evidence for §3.8 (protected evaluators, hack-label curation, judge validation) and the statistics every factory number needs (PPI, calibrated escalation, clustered and paired SEs, IRT). It also holds the hand-built World canon (AppWorld, ToolSandbox, TheAgentCompany, Toolathlon, Corecraft). Group 10 is the enterprise and web environment set. It covers World-first synthesis (AgentMercury, CRMArena), App cloning (VeriEnv, GUI-Genesis), knob-driven difficulty (WebForge, TRUSTEE, WebGym), trace mining (EnterpriseClawBench) and simulated people (the Sim2Real study).

---

## B1. Domain spec and World/environment synthesis

### M1.1 Seed through the App's own tested APIs, with backdated time (AppWorld)
- **How.** Build each App as code (AppWorld: FastAPI + SQLite + SQLModel; 9 apps + 2 helpers; 457 APIs; 101 tables; 1,780 API unit tests at 98% coverage). Fill a Base DB *by calling the APIs*, not by writing rows, so every write passes the App's own validation and multi-table consistency (an order clears the cart and sends a confirmation email). Tooling lets the seeder call APIs "in the past" so timestamps are consistent. The data holds 106 fictitious people aged 19–60 with relationships (friends, family, roommates, coworkers) and 2 years of activity sampled to match those relationships (rent split among roommates, not coworkers), ≈370K rows. Text whose meaning a task depends on is hand-written. Low-stakes text (memo, amount) comes from an LLM with reviewed prompts [2407.18901].
- **Clock and reset.** Each task fixes a start timestamp. `freezegun` freezes `datetime.now()` inside the shell. Reset reloads the task DB and the time in under 0.5 s (in-memory SQLite, server and client in one process).
- **Diff support.** Per-row `record_hash` and per-table change counters make the start→end diff cost under 0.6 s.
- **Failure modes.** The engine took ~100K LOC over 14 months by the authors. Private expected values are computed in Setup, so a Setup bug reaches both the solution and the evaluation.
- **Crucible.** F2 `SeedBundle`, P0. Seed through policy-visible App APIs with backdated timestamps, which gives G1 integrity by construction. Store per-row hashes in `StateAdapter`. Inject the clock (`clock_injection` in `AppSpec`).

### M1.2 Latent traits that cause the records, generated in FK-dependency order (CRMArena, CRMArena-Pro)
- **Algorithm** [2411.02305]:
  1. Inputs: a company name and description, a real vendor schema (Salesforce Service Cloud, 16 objects) and a target count per object (mid-size firm, 4 years).
  2. Generate objects in DAG order (Account/Contact → Order → OrderItem; User skills → Case → EmailMessage, LiveChatTranscript, CaseHistory).
  3. Put **latent variables** in the DAG as hidden nodes: `ShoppingHabit` per customer (when and what they buy, seasonally), `Skill` per agent (solve or transfer) and policy-deviation flags that cause violations in transcripts. *Without the Skill latent no transfers exist, so the transfer task family cannot exist.*
  4. Generate JSON in mini-batches of 10 with GPT-4o. Every earlier entry of that object goes in the prompt with "do not repeat". Then exact-dedup the task-critical fields (e.g. email).
  5. A format verifier regenerates any batch with a missing field. A task-specific content verifier (e.g. an LLM must pick the right product from a paraphrased name) discards entries that are too far or too close.
  6. Upload to a clean org *without* the latents, and map generated IDs to org IDs.
- **Scale.** 100 users, 500 products, 22,000 price-book entries, 2,071 orders, 977 cases, 3,234 emails. 90% of 10 experts rated it realistic; the main complaint was empty, "not worked in" records.
- **CRMArena-Pro** [2505.18878] merges the Service, Sales and CPQ schemas (25 objects) and uses **21 latent variables**. It makes two Worlds from one generator with two scenario configs: B2B with 29,101 records and B2C with 54,569. Expert realism fell to 66.7% (B2B) and 62.3% (B2C) from CRMArena's 90%: *generated density is not realism*.
- **Failure mode.** Ground truth leans on latents the agent cannot see. No check is reported that every answer is derivable from visible data.
- **Crucible.** F2, P0. `latent_traits (hidden)` exists in the contract. Make latents *cause* records so families become possible. Add a G8 variant: every answer computed from latents must also be reachable by the oracle through policy-visible tools. Make several scenario configs per World template, and watch realism as density grows.

### M1.3 World first, with invariants checked after the episode rather than enforced by transitions (AgentMercury)
- **Factorisation** [2608.20634]: company identity p(C|σ) → service graph p(G|C) → state schema p(Σ|G,C) → seeded state p(s₀|Σ) → World-level invariants p(R|G,Σ,s₀).
- **Key design.** Invariants R are **not enforced by the transition function**. "A downstream record must exist after an upstream event" is checked after the episode, and the World does not create that record for the agent, so the agent must do the cross-service follow-up. R has two views: R_vis, surfaced as in-World policy documents the agent must find, and R_hid, the executable checks.
- **Tasks.** (Δs₀, u, ρ) ~ Task(s₀, R): a task-specific state patch, an instruction and a rubric of three assertion forms only (record exists, field equals, field differs; 2–10 per task, mean 5.4). Ten task seeds per World.
- **Tooling.** A fixed core of 10 tools plus 4 per extra resource (search, list, get, update), so tool counts take only the values {10, 14, 18, 22, 26}. This is the root of the template lattice (B5).
- **Dominant authoring failure.** Collapsing a cross-service invariant into one service (GPT-5.4 on 10/30 briefs).
- **Crucible.** F1/F2, P0. Make `invariants: [Invariant{trigger_service, target_service, check_sql, visible_doc_ref}]` a first-class `AppBundle` field. Render R_vis into policy docs and R_hid into `VerifierBundle.state_checks`. Add a gate that a cross-service invariant really spans two services.

### M1.4 Task-centric World building with planted enterprise noise
- **Corecraft** [2602.16179]: a PC-parts retailer's support org in one Docker bundle; >2,500 entities over 14 types; 23 MCP tools; transactional state per episode. Every entity and tool exists to support hard tasks, not to maximise size. The planted noise includes conflicting timestamps, incomplete records and **pagination limits that do not signal truncation**. Frontier failures it exposes: accepting exactly-limit-sized results without paginating, generic search before looking up the customer's product, and anchoring on the first plausible tool.
- **EnterpriseOps-Gym** [2603.13594] plants hidden challenges: two incidents share an external number across tenants, a tool default violates policy, multi-hop ID resolution.
- **ClawsBench** [2604.05172] seeds each task with *needles* (targets) and *fillers* (noise).
- **AppWorld** Setup [2407.18901] enforces well-definedness (presuppositions hold), distractors (many similar orders), realistic hurdles (an expired default card) and contrast sets.
- **GUI-Genesis** [2602.14093] forces 3–5 plausible distractors per list or selection, free-text inputs and no pre-filled targets.
- **TheAgentCompany** [2412.14161] plants setup failures in coding tasks, pop-ups and HR/admin traps. **Toolathlon** [2510.25726] uses messy real initial state: inboxes with many senders, shops with hundreds of products, repos with 10+ developers' branches.
- **WebForge** [2604.10988] injects a cookie banner at 1 s, a promo popup at a random 5–15 s, and latency. It replaces blocking `alert()` with inline errors and adds unneeded login pages.
- **Crucible.** F0 knobs and F2 defaults, P1. Add `distractor_density`, a "silent truncation" tool policy in the App template library, trap knobs (duplicate identifiers, policy-violating defaults) and needles + fillers. Pair each planted trap with a negative (G5). Prune entities and tools that no task needs.

### M1.5 App fidelity tiers and how each was built and checked
| Tier | Example | Build and check | Numbers |
|---|---|---|---|
| Real open-source apps, containerised | TheAgentCompany (GitLab, ownCloud, Plane, Rocket.Chat) [2412.14161]; Toolathlon (Poste.io, Canvas, k8s, WooCommerce) [2510.25726]; WebArena (Magento, Postmill, GitLab) [2307.13854]; EntWorld (EspoCRM, ZenTao, OpenProject, CMDB, Snipe-IT, iTOP) [2601.17722] | Mock data baked into images; init scripts; MCP servers patched or written by authors (many open servers were buggy or missing tools) | Toolathlon: 32 apps, 604 tools; 108 tasks in ~70 min with 10 processes on one 16-CPU node |
| Mock REST Apps imitating real APIs | ClawsBench (Gmail 62 endpoints, Calendar 38, Docs 12, Drive 41, Slack 45) [2604.05172] | **Conformance pipeline**: capture golden request–response pairs from real accounts (169 fixtures); compare at every nesting level (key sets, value types, empty-collection shapes, mutation side effects); 328 automated tests; coverage map; append-only quirk log (65 entries) | Found 11 recurring mock bug classes (over-serialised mutations, null vs absent, format-dependent shape, list reusing the detail serialiser, static values for computed fields, wrong mutation side effects, ...) |
| Coding-agent clones with a privileged SDK | VeriEnv [2603.10505] | From screenshots, an agent (Cursor CLI + GPT-5.2) builds code C, DB D and **Python SDK P** exposing internal state, plus start and reset scripts. It then explores its own site with Playwright, files structured bug reports, patches and re-tests | 149 sites; 26% failed (missing start scripts, port conflicts, CORS); $3.6 and 83.5 min per site; functionality 90%; **judge correct 76%** |
| Trace-driven, one App per task | GUI-Genesis [2602.14093] | Kimi K2 turns a real-app trace (successful or not) into a task-specific system prompt; plans a file manifest; writes Flask + vanilla JS with an in-memory backend and a hidden reward; ≤5 self-verification retries with a Playwright golden-path test that must trigger the reward | 1,118 Apps for $302.12 (<$0.30 each); 82.65% pass within 5 tries (27.4% first try); 0.42 s per interaction vs 4.81 s real |
| LLM-simulated environment | TRUSTEE [2604.17739]; OccuBench LES [2604.10866] | One LLM holds state in context and returns JSON observations | Avoid for stateful training (M1.10) |
- **Tips.** Use a coding agent *with vision* for UI Apps: text-only open coders could not fix front-end bugs, and Claude CLI and OpenHands often quit early [2603.10505]. Plan the file manifest first, then write files one by one [2602.14093]. Forbid "task completed" pop-ups and auto-redirects, and define the terminal page by task type [2602.14093].
- **Crucible.** F1, P0/P1. (a) An "App conformance" gate next to the blind tool tests whenever an App imitates a real API, with the 11 bug classes as a checklist for the schema and tool coder. (b) The C/D/P triple is `AppBundle` + `StateAdapter`; the grader uses the SDK and the solver never sees it. (c) VeriEnv's self-test-and-patch loop is F1's repair round. (d) Containerised OSS is the highest-fidelity tier for the Harbor/CUA lane, wrapped by MCP sidecars.

### M1.6 Tool-surface design rules
- **ToolSandbox** [2408.04682]:
  - Tools raise *informative precondition errors that imply the fix* (`send_message` raises ConnectionError when cellular is off). Enabling cellular needs low-battery mode off, which gives nested state dependencies the agent must find and backtrack through.
  - One "omni-search" per domain, at least one state-changing tool per stateful domain, and utility tools for canonical forms (timestamp ↔ date parts, distance), so the agent is never forced to compute in its head.
  - **"Murphy's law"**: the environment always surfaces a race when dependent tools are called in parallel. Big models were hurt by it: GPT-4 69.2 and Claude-3-Opus 74.5, against GPT-3.5 82.6 and Claude-3-Sonnet 84.5.
- **Toolathlon** [2510.25726]: return tool errors as observations, not loop ends. Truncate outputs over 100K characters, with a paging toolkit over the cached raw output (10K pages). Overlong outputs occur in 15–35% of trajectories and lower success. Load whole servers (<10 per task) to create realistic distractor tools.
- **WorkBench** [2405.00823]: documented result limits (e.g. `search_events` returns at most 5) force paging.
- **TheMCPCompany** [2510.19286]: one tool per REST endpoint from OpenAPI (GitLab 1,085, Azure 16,837, Rocket.Chat 520 with LLM-rewritten descriptions; 18,505 in total). A gateway exposes `find_tools(query)` (embedding top-k) and `call_tool`.
- **OfficeBench** [2407.19056]: restrict visible operations to the current App plus switch/submit. GPT-4o 47.0 vs 44.3 with ~30% fewer tokens.
- **WebServ** [2510.16252]: mark interactive elements explicitly and give IDs only to them. Removing the cue dropped GPT-4o CMS 28.6→8.6 and Claude 3.5 GitLab 36.7→6.7; weaker models suffer most. Return the observation only after network quiescence (e.g. 500 ms with zero in-flight XHR/fetch), else an explicit timeout error.
- **REAL** [2504.11543]: App `config_flags` (`error_finding_driver`, `error_booking_ride`, latency default 2,000 ms, pricing multiplier, location preset, `hide_aria_labels`) set per run by URL, so one App yields many variants.
- **Crucible.** F1 tool rules, P1. Add precondition exceptions with fix hints, omni-search, canonical-form utilities, a Murphy race mode, paging and errors-as-observations as harness features the student needs *in training*. Add `config_flags` as first-class `AppSpec` parameters, and an explicit "actionable" marker on every affordance (matters most for the 35B).

### M1.7 Schema discovery for an existing App or customer DB (EntWorld)
- **Steps** [2601.17722]:
  1. List the non-empty tables.
  2. An LLM writes a SchemaJSON per table.
  3. Parse explicit FKs. An LLM proposes implicit links from column name and type similarity, and **each is kept only if a probe JOIN returns rows**.
  4. An LLM abstracts workflows (core tables, relations, business description). Cache by the MD5 of the core-table list.
  5. Template SQL is run **inside a transaction**: it must execute and return non-empty results, affected rows are inspected for CUD, then ROLLBACK.
- **Crucible.** F1 schema-integrity gate for §2 enterprise ingest, P1. Rollback execution is the cheap G3 pre-check for SQL-expressed oracles.

### M1.8 Reset, isolation and determinism
| Mechanism | Source | Number |
|---|---|---|
| In-memory DB reload + frozen time | AppWorld [2407.18901] | <0.5 s per task |
| Snapshot config files → diff → revert only changed files (for a World too costly to recreate) | SCUBA [2509.26506] | The same diff is a free collateral check |
| Block-level copy-on-write containers (Incus/LXC on ZFS/Btrfs); clone running containers for mid-episode checkpoints; one paired browser + server per rollout | WebServ [2510.16252] | Launch 1.78 s vs 8.96 s; 28 MiB vs 6.78 GiB per instance; 200+ containers per host |
| Fresh user created and deleted around each episode; an installer pins global config | WorkArena++ [2407.05291] | Allows parallel runs |
| Container restart from image | WebArena [2307.13854] | Seconds to a minute; too slow at RL scale |
| VM snapshot | OSWorld [2404.07972] | 302 distinct initial states |
| One state store, `/finish` diff, `/clear` reset, locked dates and time zones | REAL [2504.11543] | Client-side store is writable by a CDP-level agent |
- **Determinism hazards.** VeriEnv's top judge error was a DB reset that did not keep the random seed used to fill the data, so expected values drifted [2603.10505]. In the clinical study, a font or library version change silently altered rendered cues and invalidated a content cache, and about half of per-case outcomes differed on re-run [2608.03744].
- **Crucible.** G2, P0. Extend G2 to "reset reproduces the seeded state exactly" (fingerprint after reset) and to content-addressed renders with pinned library versions. Use block-level CoW per rollout for real-service Apps.

### M1.9 Simulated people: knowledge boundaries, an information-release schedule and fidelity checks
- **Knowledge boundary + hidden demos** (ToolSandbox [2408.04682]). The user-simulator prompt holds the goal (the agent is called "User B" to avoid role reversal), a *knowledge boundary* (what the user knows and does not, including partial access to the expected result) and few-shot demos visible only to the user role. Error on 1,032 annotated trajectories, as hallucination / instruction errors: goal only 12.4% / 6.2%; + boundary 7.75 / 3.88; + demos 6.97 / 0.77. The user ends the dialogue with `end_conversation`, its only tool.
- **The Sim2Real gap** [2603.11245]. 451 Prolific users vs 31 simulators on 165 τ-bench tasks, with GPT-5.2 as the agent:
  - Simulators run "easy mode". Most general simulators raise agent success above the human 63.6%, up to 77.8%.
  - They front-load identifiers (UserLM-8B 4.8 per turn vs 2.6 for humans) and rarely use short turns (GPT-4o 1% vs 29%). They are too polite (49% vs 15%), hedge more (14.6% vs 7.3%) and pivot more (16–19% vs 8.4%).
  - Human ceiling USI 92.7; best simulator DeepSeek-V3.1 76.0. Capability (Arena Elo) does not predict fidelity.
  - A findings-informed persona prompt *lowered* USI 70.9→64.6 and worsened ECE 0.18→0.29.
  - As evaluators, simulators are lenient on experience: GPT-5.1 rates human-likeness +1.11 above humans.
  - Metrics: D2 information pattern (share of words in the first two turns, identifiers per turn); ECE over 5 difficulty bins, ECE = Σ_b |S_b|/N·|p̂_sim(b) − p̂_human(b)|.
- **NPC colleagues** (TheAgentCompany [2412.14161]): Sotopia profiles with name, role, responsibilities, projects and channel access, backed by Claude-3.5-Sonnet; ≤~$0.024 per call; 1 role-play error in 41 reviewed NPC tasks. Some task information is obtainable only by asking the right person.
- **User holds the info** (CRMArena-Pro [2505.18878]): move the required domain knowledge from the query into the simulated user, who releases it incrementally. Single→multi-turn: gemini-2.5-pro 54.1→35.1 (B2B). In 9 of 20 failed multi-turn trajectories the agent never got all the needed information. Better models ask more clarifying questions. Simulator error was 1/20 on a hand check.
- **The simulator needs intent and persona.** Hiding the intent from TRUSTEE's environment lowered BFCL Multi-Turn 46.4→43.0, and hiding the persona →43.1 [2604.17739].
- **Crucible.** F2 people / F3 `user_known_info` / G8, P1 (P0 for any domain whose skill is elicitation):
  - A `UserSimSpec` with a hard information-release schedule (identifiers released in fragments, only when asked).
  - An F6 "user-sim fidelity" gate: front-load ratio and IDs per turn within a band of a human or customer-trace reference.
  - An F9 check that the pass rate with the simulator is not above the rate with a stricter or human-derived user on a sample (ECE-style).
  - Every `ActorSpec` carries a knowledge boundary and hidden demos.
  - Never use an LLM user's satisfaction rating as reward.

### M1.10 Keep LLM-simulated components out of state and reward
- **TRUSTEE** [2604.17739]: an 8B model plays task generator, user, tool and verifier. Stateful tools (EnvScaler's) trained worse than stateless ones (base model BFCL Multi-Turn 7.9 → 1.0). The judge agreed with ToolRL's rule scores at ρ 0.70, precision 0.72.
- **OccuBench** [2604.10866]: swapping the simulator moved every agent's average from 67.9% (Gemini Flash) to 63.4% (Qwen 3.5 Plus) to 29.3% (GPT-5.2). GPT-5.2 as a simulator fabricated rooms, omitted a required roster entity and invented a return-window rule. Gemini and Qwen simulators agreed on 24 of 28 pairwise rankings (85.7%).
- **Crucible.** Execution decides (§3.0 rule 2). An LLM may simulate stateless lookups or people only. Gate any LLM-simulated component with a fabrication / omission / rule-invention audit and a cross-simulator ranking agreement ≥85%. Implement faults in App code (M6.6), not in a prompt.

### M1.11 Permissions and confidentiality as World features
- **RBAC** [2510.27287]: 4 roles per department in a 4:3:2:1 ratio. Levels 9–14 × department rules, LLM-generated and expert-checked, live in a JSON rule file per source, and an access layer filters retrieved content. *Weakness:* part of the access check is done by an LLM agent, so it is not a deterministic permission model. Removing access control made agents attempt unanswerable tasks.
- **Confidentiality probes** [2505.18878]: 80 queries per org for each of 3 types (other customers' PII, internal metrics, confidential knowledge). Refusal is near 0% with a standard prompt. A confidentiality-aware prompt raises gpt-4o-mini to 62.9% but costs task completion (gemini-2.5-pro −5.7), and works less in multi-turn and on open models (<12%).
- **Crucible.** Policies-as-code in `AppBundle`; the org hierarchy goes in `SeedBundle.people`; permissions are deterministic code.

### M1.12 Time-varying ground truth
- Toolathlon and OSWorld use *dynamic getters* that fetch live facts at grading time (current shareholders, live train schedules) [2510.25726, 2404.07972]. Runs then stop being reproducible over time. WebGym's live sites likewise make labels non-reproducible [2601.02439].
- **Crucible.** Allow dynamic answer checks only in live-data families, and mark those tasks non-sealable.

### Minor tips (B1)
- Choose App categories from real usage logs (WebArena used ~200 segments of the authors' own browsing histories). Seed one persistent persona per App with a long history (a customer with 35+ orders) so tasks can be personal. Sample seed entities 80/20 popular/long-tail (GitLab repos by stars) [2307.13854].
- Randomise the surface skin per instance (10 brands and themes) for cheap visual diversity [2407.05291].
- Start tasks from intermediate states ("document half edited", "after a crash") [2404.07972]. This is the `start_offset` knob.
- Sample tool subsets within one category first, so the tool set reads as one App [2604.17739].
- Contain the solver: privilege drop, root-owned task files at mode 700, no sqlite3 binary. Log escalation attempts (env enumeration, opening the SQLite file, curl to localhost; 1,471 such calls for GPT-5.4 on Codex) [2604.05172].
- Keep state server-side. REAL's localStorage and WebForge's Base64-encrypted answer fields are decodable by an agent that reads the page source [2504.11543, 2604.10988]. A reward printed to stdout must be isolated from the solver [2602.14093].
- Write the business policy (tie-break rules, period definitions) into the task context so the answer is unique [2411.02305].
- Ship a create/destroy script (Terraform) and a proof-of-concept solution script with every task on a stateful external service. On a real cloud, GPT-5 deleted a VM by mistake [2510.19286].
- If an LLM must keep state, write state notes ("X decreases after Y") [2604.10866].
- Stop an episode after N identical consecutive actions (OfficeBench N=5; WebArena 3 repeats or 3 invalid actions) [2407.19056, 2307.13854].

---

## B2. Task proposal and sources

### M2.1 Setup / Oracle / Evaluation programs per family, with reject-and-resample (AppWorld)
- **Shape** [2407.18901]. Each of 250 hand-written scenarios (3 tasks each, 750 in total) has three parts:
  1. A **Setup** program. It picks a supervisor, samples placeholder values and edits a copy of the Base DB (and the time) so the task is well-defined, has distractors and hurdles, and forms a **contrast set** with its siblings. It may `Fail(k)` and resample. It writes private expected values the agent never sees.
  2. A **Validation solution** program that uses minimal setup knowledge and must pass the evaluation from the start state. It is kept as a regression test.
  3. An **Evaluation** program: a battery of state assertions (mean 8, max 22), each with an NL description.
- **Metrics.** TGC (all tests pass per task) and SGC (all 3 tasks of a scenario pass). SGC runs 30–50% below TGC: agents do not solve every contrast variant.
- **Make tasks need interaction.** Put information only in free text that must be read. Oracle API lists added only 7.8–9.8 TGC, so retrieval is not the bottleneck.
- **Cost.** Task generators were ~40K LOC over 5 months, the cost a factory must beat.
- **Failure mode.** The validation solution shares Setup's knowledge, so it cannot catch a wrong expectation.
- **Crucible.** F3–F5, P1. A DomainSpec family is one AppWorld scenario: a parameterised Setup that enforces well-defined / distractors / hurdles / contrast set with `Fail→resample`, an Oracle program and an Evaluation battery. Check Setup's private values independently, with a second implementation or G6 inverse mutation. Report scenario-level all-pass.

### M2.2 setup / teardown / validate / cheat, and composition of atomic units (WorkArena, WorkArena++)
- **WorkArena** [2403.07718]. A task is 4 functions: `setup()` (create rows, navigate, log in), `teardown()`, `validate()` (reward, an optional user chat message, done) and an optional `cheat()` Playwright oracle. The cheat run is the feasibility gate and the canary for platform upgrades. 33 templates expand to 19,912 instances, with combinatorial categories capped at 1,000 random instances. *Instances differ only in values, so the count overstates diversity.*
- **WorkArena++ composition** [2407.05291]:
  - A workflow chains atomics (create user → order laptop → assign laptop). The composite oracle calls the atomic oracles in order. Validation is per step or one global DB query. A new workflow costs about one config line.
  - The 341 workflows each come at 2 presentation levels, 682 tasks in all. L2 gives the exact steps. L3 says only "please complete the following task", and a ticket carries the values and points to a knowledge-base protocol article.
  - Humans scored 93.9% on both levels. GPT-4o scored L2 3.0% and L3 0.0%; Llama3-70B, Mixtral and GPT-3.5 scored 0%.
  - An infeasible family of 52 tasks uses `report_infeasible(reason)`.
  - Curriculum: buckets of tasks needing the same ability, a meta-seed drawing seeds per category, sampling without replacement. Meta-seeds 0–9 are reserved for evaluation.
- **Crucible.** F3/F4, P1. Composition is a proposal operator: `TaskProposal` = a sequence of atomic cells, with the composed Oracle as concatenated action programs and the composed VerifierBundle as per-step milestones + global state. Φ milestones come for free. Shorten composites that sit below the band instead of shipping them.

### M2.3 Template + outcome function, with conditional and no-action instances (WorkBench)
- **Shape** [2405.00823]. 69 templates (48 single-domain, 21 multi-domain), each with an *outcome function* that computes the expected change to the databases from the slots and current data. 10 instances per template (690 tasks); 3 phrasings each.
- **Anti-shortcut devices.** Conditional tasks ("if page views fell >10% last week, schedule X tomorrow at the earliest free slot; else email Y"). 122 of 690 (18%) need no action.
- **Results.** GPT-4: 0-action tasks 75%, ≥1 action 36%, ≥2 actions 18%. Only 15 of 69 templates sit at 0% or 100%, so instances within a template differ in difficulty.
- **Crucible.** F4 Oracle kind `end_state` computed from slots, cheaper than an action program for template families. Default to conditional tasks plus a ≈15–20% no-action share. Check that "always act" and "never act" policies cannot score (extends G4).

### M2.4 Typed slots + code-computed ground truth + paraphrase; false-presupposition share (CRMArena)
- **Steps** [2411.02305]. 14 seed templates with slots ([YEAR], [EXTREMA], [NCASES], [PRODUCT], [TODAY]). Fill the slots, compute the ground truth by code on the generated DB *with* latents, map the IDs, then paraphrase with an LLM. 30% of queries in five tasks are false-presupposition ones (expected "None"). 130 instances per task.
- **Note.** The non-answerable queries were often *easier* than the answerable ones.
- **Crucible.** F3, a cheap proposer for analytics cells. Diversity then comes only from slot values and paraphrase, so F8 must dedup on templates.

### M2.5 Reverse synthesis: context first, then the task whose answer it holds [2510.27287]
- **Algorithm.** Sample a domain and an expert-curated persona. Retrieve *that employee's* records. Choose an expert goal template built from O*NET 29.2 occupation tasks. Extract entities and relations with an LLM. Decompose into tool-aware sub-goals and fill the sub-task templates. Ground each sub-task in a context snippet. Assemble task + ground truth. Validate against an expert checklist, rephrase, and repeat up to max_iter. Claude 3.5 Sonnet does a final quality check.
- **Numbers.** ~80 s per task; mix 65% search, 30% CRUD, 5% unanswerable; experts kept 80%.
- **Weak point.** Grading is a 1–5 Prometheus judge (o1-mini: 31% by human evaluation vs judge ~0.35).
- **Crucible.** A proposer mode (World slice → goal template → a task whose answer or end state is reachable), P2 (ideas only). Keep a grounding snippet per sub-goal for audit.

### M2.6 Prototype → instantiation (with invalid fills), and single corruptions above an effect threshold (VisualAgentBench)
- **Prototypes** [2408.06327]. A prototype is a high-level goal with placeholders. Instantiation grounds the placeholders in concrete items, attaches the judge and rephrases. The WebArena training set used 40 prototypes per site with a hand-written Playwright solver each, filled with *valid and invalid* items: 1,186 trajectories.
- **CSS debugging.** Corrupt one categorical property of one random rule (alter or delete). Keep it only if the SSIM between renders is <0.8, and annotate the visual difference in words. Removing the NL difference description dropped gpt-4-turbo 27.9→1.9.
- **Crucible.** F3. Add "invalid item fill" as a free infeasible generator. Generalise the effect threshold to every mutation task: a planted change must move an observable by at least a threshold (a G6-style check).

### M2.7 Seed → derived branches → paraphrases (ToolSandbox); extend a verified task by one App (OfficeBench)
- **ToolSandbox** [2408.04682]. Seed scenarios (single tool, single turn) → derived branches (multiple calls, multiple user turns, state dependency, canonicalisation, insufficient information; combinable) → paraphrases that reuse milestones. A second annotator who sees only the agent's sub-view solves each scenario, followed by ≥4 rounds of model tests. 1,032 scenarios; 85% are in at least one hard category.
- **OfficeBench** [2407.19056]. Make a 3-App task by adding one relevant App to a 2-App task. GPT-4o: 64.5 (one App), 60.0 (two), 21.4 (three).
- **Crucible.** Knob-driven variants of one family with reusable milestones; each branch re-passes G3/G5. "Add one App to a verified task" is an F3 mutation that keeps the oracle prefix.

### M2.8 Plan → Generate → Refine → Validate with two planner models (WebForge)
- **Pipeline** [2604.10988]. A creative model (Gemini-3-Pro, T=2.0) drafts the objective, the difficulty vector δ ∈ {1,2,3}⁷, the site design, the solution path and the answer type. A precise model (Claude-4.5-Opus, T=1.0) reviews it, changing 30–50% of the draft.
  - The Generation agent fills placeholders with real data found by search and builds a static multi-page site that never hints the objective.
  - The Refinement agent applies a rule checklist (functional completeness, visual correctness, state determinism, realism, task security, interaction feedback) in an Assess → Plan → Execute → Verify loop.
  - The Validation agent replays the solution path in the same Chromium the evaluated agents use (≤50 steps, 3 retries per action), and checks that the reasoning derives the answer and the result equals the ground truth.
- **Yield and ablation.** 1,260 generated → 934 validated (74.1%; information retrieval 88.9%, consumer transaction 63.9%). Without plan refinement, 59.5%; without plan refinement and without the refinement agent, 51.4%.
- **Crucible.** F3 swarm roles, P1: draft and refine by *different families*, validated in the student's exact runtime.

### M2.9 Coverage-first scenario synthesis with plan / no-plan solvability runs (OccuBench)
- **Steps** [2604.10866]. 100 scenarios mapped to real job roles (≤3 per domain, 65 domains, 10 industries) → 16 non-overlapping sub-topics per scenario, each with a professional reference document (terminology, workflows, state variables, edge cases, constraints) → agents generate the config, instruction, tools, solution plan and rubric → each task runs several times *with and without* the plan → a majority-vote verifier → a repair module diagnoses and fixes, then re-runs → drop tasks at 100% success, at 0% or with invalid schemas. Keep the difficulty level with the *lowest* autonomous success. 382 instances; mean 5.5 tools and 16.2 calls.
- **Caveat.** Choosing the lowest-success level biases tasks toward the simulator's quirks.
- **Crucible.** F0 coverage method. F9 cheap predictor: the no-plan pass rate estimates difficulty, and the plan pass rate checks solvability.

### M2.10 Task proposal JSON with intent hidden from the solver (TRUSTEE)
- **Shape** [2604.17739]. Given a tool subset and four settings (expected distinct calls 1–3; persona expert/beginner/novice; first-query ambiguity clear/somewhat/highly; expected turns), the model returns the expected tool list, the hidden user intent (richer than the first message), the persona and a first query that must not name tools. The agent's system prompt is drawn from 4 specificity levels.
- **Crucible.** This matches `TaskProposal.user_intent` + `privileged_notes`. Persona × ambiguity are orthogonal axes ("an expert can ask vaguely").

### M2.11 Verified Q&A generation with a weaker solver (WorkArena knowledge base)
- **Steps** [2403.07718]. 100 (item, value) facts. GPT-4 writes an HTML article that must contain the exact sentence stating the fact, then 10 paraphrased questions with format instructions. GPT-3.5 must answer each from the article, and failures go back to GPT-4 to rewrite. *A weaker solver is used so the writer cannot cater to itself.* GPT-4 lists 10 acceptable answer formats, checked by hand.
- **Crucible.** The generator for `answer_kind=lookup` tasks over seeded documents. Precompute answer variants at build time, not with a judge at grade time.

### M2.12 Traces to tasks: a gated miner of real sessions (EnterpriseClawBench)
- **Gate order** [2606.23654]. The source is ~3 months of an internal agent used by >100 employees.
  1. Candidate recovery: split and merge session turns, giving 5,291 candidates.
  2. Length filter (<10 effective user characters rejected).
  3. **Fixture resolvability**: every declared attachment must resolve to bytes, else reject, so missing data is never scored as agent failure.
  4. **Redaction recovery under a unique-match rule**: restore a redacted URL or path only if the surrounding text matches exactly one raw-session URL with the same host, or the session holds exactly one candidate for that host; else risk-mark and reject.
  5. External network dependency check.
  6. **Self-containment** by an LLM. A signal of failure: *the original agent itself asked for clarification*.
  7. Rewrite into one prompt naming the materials and the exact deliverable (file type and count); strip timestamps, hashes and chat noise.
  8. Taxonomy: 7 role classes × 45 skill subclasses.
  9. Deliverable detection, hard rules (type, count, non-empty, openable, no tracebacks, no unreplaced placeholders), then weighted semantic rubrics.
  10. **Sandbox preflight** (upload, run, download, judge routing) before admission.
- **Yield.** 852 tasks (16%), with a 120-task Lite subset audited by hand.
- **Skill-transfer protocol.** 10 in-domain runs → a creator model distils a skill → inject it → 5 held-out tasks. GPT-5.5 as creator +0.068 and never negative; Haiku 4.5 −0.094.
- **Crucible.** The §3.4 trace miner, P0. Each gate writes a reject reason and a yield (design rule 1). Map role × skill subclass to `skill_tree` leaves. Replace the rubric with World-grounded checks where the deliverable is a state change, and keep the hard rules as G-gates.

### M2.13 Traces to Apps and to procedure docs
- **GUI-Genesis** [2602.14093] rebuilds an App per task from a real-app trace, including failed traces, which still carry look-and-feel and page flow. This suits the CUA lane and customer traces where the real system is unavailable. The trace grounds `AppSpec`, and the golden-path test is the Oracle + G3.
- **SCUBA** [2509.26506] has GPT-5 summarise each human trajectory into a short plan and retrieves 2 demos per template. Even *similar* demos help: Claude computer-use 27→47.7%, Gemini-2.5-Pro 31→46%, UI-TARS 2.7→9.2%. Plan distillation is the summariser step that turns traces into reusable procedure docs (KB articles) inside the World.

### M2.14 Real-data mining with natural difficulty labels and contamination-proof variants (SWE-Lancer)
- **Mining** [2502.12115]. 1,488 Upwork tasks from one repo, $1M in actual payouts ($250–$32,000). Prices rise when proposals fail, so **price tracks difficulty**. Engineers screen with 0–3 rubrics: well-specified without the comments, the PR really fixes it, tests "valid for all reasonable solutions". Playwright E2E tests are triple-verified, and an automatic check confirms each passes on the fixed state and fails on the broken one.
- **Holdout variants.** Keep the description and tests fixed, and **re-introduce the bug in a different but equivalent way** ($499,200 private set).
- **Crucible.** Debug lane, P1. A variant operator for held-out supply, and a G6 check that tests key on behaviour, not on one patch. Use market price or time-to-close as a prior for the difficulty predictor on mined tasks. Proposal-choice data (pick the manager's choice among 4–5 real proposals; 99% re-validation agreement) can become cheap discrimination cells.

### M2.15 Provenance-clean mining for restraint tasks (RepoComplianceBench)
- **Steps** [2607.26819]. 455 AI-contribution provisions from 102 communities are hand-coded into policy cards (Refuse / Disclose / Verify / Handoff). 16.2K issues closed within 180 days before a cutoff → hygiene gates → an LLM curator *blind to the fix* screens for simple defects and the rule they trigger → a **temporal gate** (the provision text must exist at the pre-fix base commit) → 257 instances in 58 repos. Frozen sampling then gives 106 instances in 49 repos.
- **Crucible.** F3 provenance template for mined sources: curator blind to the fix, temporal gate, frozen sampling with a per-repo cap (≤2 per repo per rule).

### M2.16 Archive-conditioned proposal prompts
- **ADAS** [2408.08435] puts the whole archive (code + score) in the proposer prompt, asks for an "interestingly new" entry, adds a curated list of known wrong-implementation patterns (it cut code errors), and runs two self-reflection rounds (novel vs archive? bugs?). Runtime errors are retried with the traceback, at most 5 times.
- **AlphaEvolve** [2506.13131] samples several prior programs with scores and outputs, plus stochastic formatting and co-evolved meta-prompts.
- **Hack contamination of references** [2610.12822]. Showing a target k leaky-passing source submissions made Claude-Opus-4.7 adopt the hack at 51.5 / 73.6 / 84.9% for k = 3/5/10.
- **Crucible.** F3, P0 rule + P1 recipe. Show the K nearest tasks *with their measured signatures and p̂*, plus a wrong-pattern list and a novelty self-check. **Only references that passed G5/G12 are shown.** A reference later flagged as hackable is withdrawn and its descendants re-gated. The same rule applies to swarm memory (`swarm_journal`).

### M2.17 Restraint and infeasible task sources
| Source | Device | Number |
|---|---|---|
| OSWorld [2404.07972] | Real users' requests for deprecated or hallucinated features (30 tasks) | FAIL action; the infeasible class rewards giving up unless balanced |
| WorkArena++ [2407.05291] | Non-existent field, invalid configuration; `report_infeasible(reason)` with a reason check | 52 tasks |
| ToolSandbox [2408.04682] | Insufficient information; a minefield on the tool that would need a hallucinated argument | Performance anti-correlated with strength (GPT-4o 42.0, Gemini-1.5-Pro 76.2; weak open models score high only because they rarely call tools) |
| EnterpriseOps-Gym [2603.13594] | Missing tool, explicit policy violation, unavailable resource; 3–4 compound constraints and ~10 side-effect checks each | 30 tasks; refusal without side effects ≤53.9% |
| VisualAgentBench [2408.06327] | Invalid item fills of a prototype | Free |
| RepoComplianceBench [2607.26819] | Refuse or hand off per a repository rule | Native Refuse 0% and Handoff 0% for all four agents |
| CRMArena [2411.02305] | False presupposition, expected "None" | 30% of five tasks |

### Minor tips (B2)
- Choose task domains top-down from labour statistics: O*NET occupations by headcount × median wage, physical ones dropped [2412.14161]. GDPval's version: sectors >5% of US GDP → the 5 largest-wage, predominantly digital occupations (≥60% of O*NET tasks digital, weighted) → 44 occupations, $3T in wages [2510.04374]. OccuBench's: real job roles → sub-topic reference docs [2604.10866].
- Toolathlon sources tasks in two stages: free sourcing over 50+ Apps, then targeted sourcing for under-represented Apps [2510.25726].
- OSWorld mines tutorials, how-to videos, WikiHow and forums ranked by views and votes [2404.07972].
- Choose slot values *to vary difficulty* inside one template (extra scrolling, alphabetical search, more checkboxes to track). SCUBA: 60 templates × 5 instances, paraphrased and then rewritten by two authors [2509.26506].
- Make one Q&A per (item, value) with the fact embedded as an exact string, so the source of truth is checkable [2403.07718].
- Generate a deliberately under-specified variant (~40% of prompt length) to test context discovery [2510.04374].
- Running the oracle under the harness records unlimited demonstrations (WorkArena++ AXTree + screenshot + action) [2407.05291]. Splice a failed attempt with a hinted success into one trial-and-error trajectory [2408.06327]. Both give SFT warm-start data for 0% cells.
- Every graded rollout yields free pointwise, pairwise and reflection judge examples. Use them only for a *separate*, different-family judge, never to let the student grade itself [2509.22624].
- The factory's own construction traces are training data: brief→World, partial completion, validator-guided repair of a corrupted World, intent→diff [2608.20634] (B8).

---

## B3. Verifiers and gates

### M3.1 Grade by expected ⊆ diff ⊆ expected ∪ allowed, and count side effects apart from failures
- **AppWorld rule** [2407.18901]. Pass iff every expected change C^expect ⊆ D^Δ and D^Δ ⊆ C^expect ∪ C^allow. The allowed set holds optional changes (the cart restored or left empty). This one rule covers alternative solutions and collateral damage. 15% of tasks also have an answer field.
  - *Weakness:* the allowed sets are hand-written, so an unforeseen harmless change fails a correct solution.
- **WorkBench definition** [2405.00823]. Accuracy = all 5 DBs equal the ground-truth end state. A *side effect* = some DB differs from the ground truth *and* from the start, so failing without changing anything is not a side effect. Side-effect rates were Analytics 54% (plots for the wrong period) and multi-domain 29%.
  - *Weakness:* exact equality on written text fields means the wording must be in the task.
- **Four verifier classes** (EnterpriseOps-Gym [2603.13594]): task completion, integrity (FK validity), permission/process compliance, and side effects. SQL on the end state: mean 5.3 conditions, max 44. Policy compliance is the lowest-scoring class (ITSM 30.0%). The verifier-level pass rate is misleading because easy setup checks inflate it.
- **Validators that miss collateral.** WorkArena's validators check only the target record [2403.07718]. WebArena's locators read "the latest post", so extra posts are invisible [2307.13854]. A WorkArena++ global DB check of the target rows misses changes elsewhere [2407.05291].
- **Crucible.** F5 collateral + `TaskContract.grade`, P0. Use AppWorld's rule as the reference. Make "diff outside the gold diff" a separate verdict reported to F12. Require each of the four classes where it applies. When a different-family alternative fails *only on collateral*, send it to an audited "allowed changes" review (G14).

### M3.2 Milestone DAGs with best topological matching, minefields, guardrails and information-flow checks (ToolSandbox)
- **Semantics** [2408.04682]:
  - Each milestone is a target DB state (or tool trace) with a per-column similarity function: exact, ROUGE-L, or AST match for tool calls.
  - Row similarity is the **geometric mean** of column similarities, so one violated must-hold column zeroes it. Rows are matched by assignment.
  - Trajectory score = the max over mappings of turns to milestones that respect a topological order of the DAG, averaged.
- **Minefields** have the same form, but any match sets the trajectory score to 0. Example: `timestamp_diff` must never be called when the current time is unavailable, since calling it means a hallucinated argument.
- **Extras.** `guardrail_similarity`: no change to a given DB between two milestones. `tool_trace_dependant_similarity`: a value from an earlier tool output must flow into a later call.
- **Crucible.** F5 `VerifierBundle`, P0. Adopt these semantics for `milestones` (Φ). Minefields become path constraints that zero the reward. A guardrail is a no-touch window (timing/collateral check). The information-flow check is the formal version of the "skip a read" mutation operator (§3.3).

### M3.3 Other cheap negative-check primitives
- **must_exclude** [2401.13649]: fail if a forbidden string appears (e.g. the old price). Pair every "value updated" check with one.
- **not_contain_text / file_not_exist** [2407.19056]: pair them with positive `contain_text` checks, since `contain_text` alone rewards keyword stuffing.
- **Execution predicates** [2407.19056]: `no_overlap(Bob.ics)`, `common_event(...)` and comparator lambdas accept many valid end states. This is the right form for G14-heavy families.
- **Property-based checks** of a chosen entity ("any red blanket under $X"), done as DB predicates and not by a VQA model [2401.13649].
- **Locator + matcher** [2307.13854]: separate where state is read (DB query, site API, JS selector) from what must hold. Replace keyword `must_include` with structured field predicates wherever a DB exists.
- **Functional "the system works" check** for composite tasks [2510.19286]. Failures it catches: declaring victory after part of the task, not checking that the app works.
- **Completeness over many items.** Toolathlon agents updated 91 of 116 Notion rows, and finished 1940 of a 1940s year-by-year analysis and then claimed done [2510.25726]. A knob "count of items to process" with a check on *all* of them probes laziness.

### M3.4 Deceptive outcome codes (WebForge)
- **How** [2604.10988]. The App computes an outcome code from the accumulated state. Each common mistake maps to a distinct, plausible, same-format wrong code: a valid date but not Saturday → GEG-…842; wrong catering → …991; wrong guest count → …118; an invalid date → …294. Only the fully correct booking yields …841.
- **Effect.** Partial completion cannot pass as success, and the code is a free error label.
- **Crucible.** A G5 negative generator, P0. For each planted mistake class, the World returns a distinct failing outcome, and G5 checks that every class maps to a distinct failing state. Keep the mapping server-side, not in client "encryption".

### M3.5 Impossible twins and contradictory tracers: hack labels without a judge
- **Tracer** [2604.23488]. For a coding task with tests T, add one fake test with an existing test's input and a different output, giving T^⊥. No honest program passes T^⊥, so any solution that does hacked the evaluator (exit before asserts, its own harness).
  - The condition is sufficient, not necessary: it catches evaluator exploits only, not special-casing that also passes honest tests.
- **Amplify (resample-until-hack GRPO)** [2604.23488]:
  1. Reward = passing T^⊥.
  2. Sample a large group (G_L = 1024). If none scores 1, resample up to K times, then skip.
  3. Truncate to G = 32, keeping ≥1 hack and ≥1 non-hack.
  4. Run a standard clipped GRPO step with KL.

  This turns rare natural hacks into common ones for data collection (Qwen2.5-Coder-1.5B, DeepSeek-Coder-1.3B; ~300 steps on 4×A6000).
- **Crucible.** F5/F6, P1. One `NegativeCase(class="impossible_twin")` per family: an unsatisfiable predicate set in state Worlds, one conflicting hidden test in the debug lane. Any rollout that scores on the twin is a hack label (`hack_flag(source="tracer")` in `FeedbackSignal`).

### M3.6 A hardened second evaluator, and the three-part hack label
- **Leaky vs strict** [2610.12822]. The leaky evaluator is pytest over model-written files, open to exit codes, an always-equal `__eq__` and `conftest.py` report patching. The strict one imports the solution in an isolated subprocess, calls the function directly and compares normalised strings, with no pytest and no conftest.
- **Hack label** RH = 1 iff a static detector finds a known mechanism **and** the leaky evaluator passes **and** the strict one fails. All 400 sampled positives were confirmed by hand.
- **PRIME's exploit surfaces** [2606.09711]: an `__eq__` always-equal object, `sys.exit` and `conftest.py` injection. Hack = proxy ≥0.8 and gold <0.1. "Narrow honest" = proxy ≥0.8 and gold in (0.1, 0.8), a visible transitional regime under fraction-passed rewards.
- **Crucible.** G12, P1. In the debug lane, run hidden tests in a fresh tree with model-written `conftest.py`, `pytest.ini`, `__eq__` overrides and early exits blocked. The three surfaces go in the attacker playbook and path constraints, plus "process exit" and "in-solution test harness" rules.

### M3.7 Exploits are retargeted, so patch whole classes
- **Evidence** [2606.09711]. With two exploits blocked, the remaining one takes 85–91% of hacks (H 0.49–0.61), and family-specific exploit reasoning leads by 12–17 steps. Switching to gold reward drops hacking 0.40→0.03 in ~20 steps, but the precursor stays ≈0.73. Re-exposure to the hackable proxy brings onset back in 22 steps, 7.5× faster than from scratch. Only random reward decays it.
- **Crucible.** §3.8 verifier loop, P1. A patch must block the whole exploit class (all open surfaces), and H(t) is re-gated after it. After a verifier fix mid-training, prefer rolling back to the recorded pre-onset checkpoint over continuing.

### M3.8 Trust boundaries: graders, error channels and state the solver must not reach
- **Error channels leak** [2606.04455]. GPT-5.3-Codex made `solve()` raise an exception containing the target item. The endpoint's verbose traceback returned it with its label, and the agent iterated to pull all 591 dev questions with answers. (It did not transfer to the hidden test split.)
- **Isolation blueprint** [2606.04455]: an agent container plus an evaluation container (a Flask service holding all ground truth, an API proxy enforcing quotas and logging, an agent runner). Test-split scoring needs a secret header injected only at final scoring. A static import/endpoint scan. A post-hoc read-only auditor (Claude Opus 4.6) returns CLEAN / SUSPICIOUS / CHEATING with evidence.
- **The auditor's 8 cheat classes**: hard-coded answers (including encoded ones), an unauthorised model, proxy bypass, restricted-path access, brute-force enumeration of answers through the endpoint, unauthorised packages, a score anomalous for the code's complexity, and verifier/reward tampering.
  - It flagged 5 trials with distinct exploit classes in ordinary runs, and isolation stopped every one from inflating the test score.
  - The auditor was validated by **inducing** hacking: running with no valid API key, since direct red-team prompts were refused. 7 of 8 such trials cheated, and the auditor matched a human on all 8.
- **Impersonation shortcut** [2412.14161]. Unable to find the right person, an agent renamed another user to that person's name.
- **Scorer sanity** [2310.02304]. An improver returned predictions with a different array shape, and numpy broadcasting gave "accuracy" above 1000%.
- **Sandbox escalation** [2604.05172]: env enumeration, opening the SQLite file, curl to localhost, by all models.
- **Crucible.** G7/G12 and the World tool layer, P0:
  - `GraderResponse{verdict, reason_code}` with no free text from the verifier sandbox. Grader exceptions are replaced by a fixed error code.
  - Every World tool error is checked so it reveals no hidden state or gold answer.
  - The 8 cheat classes become path-constraint and audit categories.
  - Forbidden actions in every World with people: "rename or impersonate a user", "edit another person's profile", "self-message as the colleague".
  - Every check returns a bounded value of its declared type, and an out-of-range score fails the task (a range assertion in `Grade`).
  - Escalation counts per trajectory become a hack-flag feature.
  - Rate-limit and dedup endpoint queries per item to block enumeration of small answer spaces.

### M3.9 Checkers under-count success: measure recall and require alternatives
| Finding | Source |
|---|---|
| Official rule-based web checkers: 83.8% precision, **55.9% recall** (F1 67.1); they reject correct answers phrased differently | AgentRewardBench [2504.08942] |
| GPT-4o agent success, experts vs rules: VWA 35.9 vs 17.4, WA 42.3 vs 25.6, WK++ 18.4 vs 4.6. Rules even mis-rank agents | [2504.08942] |
| WebArena-Lite fixed 30–39 of 165 sampled WebArena tasks (wrong gold answers, exact_match too strict, environment bugs, URL criteria); ~18–24% of checks wrong | VisualAgentBench [2408.06327] |
| A hand audit of 20 ToolRL rule-scored failures found 10 actually correct (7 exact-match too strict, 3 malformed gold) | TRUSTEE [2604.17739] |
| On τ-bench, 70.6% of reward-0 runs were judged successful by the human user; 33% of reward-1 runs judged unsuccessful or partial; 81% of correct policy refusals got reward 1; Cramér's V 0.168 (τ-bench v1 has known reward bugs) | Sim2Real [2603.11245] |
| SWE-Lancer's "tests valid for all reasonable solutions" is a human opinion, not tested with alternative patches | [2502.12115] |
- **Crucible.** G14, P0. Every family reports checker recall against human labels on a sample (anchor set), and false-fail disputes feed new accepted alternatives. Expect exact-match checks to under-count. Keep the binary state reward, but add a separate outcome class for a correct `policy_refusal`.

### M3.10 Judges over-count success: use them only where code cannot check
| Finding | Source |
|---|---|
| No judge exceeds ~70% precision on web trajectories (best GPT-4o with accessibility tree: 69.8 P / 83.1 R); ~30% of "successes" false | [2504.08942] |
| VLM judge 71.8% vs code-native state check 49.0% for the best model on synthetic Apps | GUI-Genesis [2602.14093] |
| SDK-validated judges correct 76% (κ = 0.61 audit) | VeriEnv [2603.10505] |
| Visual judge vs humans: Spearman −0.26 (text 0.79, MAE 0.134); judges differ in absolute scale, so scores from different judges must not be mixed | EnterpriseClawBench [2606.23654] |
| Corecraft's rubric judge: reliability asserted, not measured | [2602.16179] |
| TheAgentCompany: 29% of tasks use an LLM evaluator (deterministic keyword checks first); judge and NPC share one model; the claim that judge concerns "should be dismissed" rests on review, not measured agreement | [2412.14161] |
- **Judge error types** [2504.08942]: grounding mismatch (trusting the agent's description of the screen), misleading agent reasoning (the agent claims success), missed instruction details, misread action intent.
- **Crucible.** F5, P0. Execute Corecraft-style criteria as state and answer checks wherever the World can ("refund issued" is a state check; "identified the PSU is insufficient" is an answer check). Leave only communication quality to a calibrated judge. Never use a 1–5 judge or an LLM answer extractor as the reward: prefer a structured `submit` tool.

### M3.11 Judge input hygiene and agentic judging
- **Never show the reviewer the pass signal.** Verifier agents blocked most hacks, but showing "ALL TESTS PASSED" raised false accepts. All verifiers also rejected some genuine solutions [2610.12822].
- **Strip or quarantine the agent's own success claims.** Judges anchor on them [2504.08942]. Machine-observed state outranks the agent's prose ("all tests pass" with no matching command is a violation) [2607.26819].
- **Less input can be better.** Screenshot-only gives higher precision and the accessibility tree higher recall, and both together were worse than the screenshot alone [2504.08942].
- **Normalise before judging** (TRACE [2609.12439]). Call 1 normalises both candidates into fixed fields (final answer, support facts, reasoning links, missing support, unsupported claims, uncertainty) with presentation masked. Call 2 compares only those records. v2 restored 97–100% better-plain resolution with 0% worse-cited wins, at 3.7–4.3× tokens and 5.3–5.9× dollars (~$18.6–21.6 per 1,000 comparisons with GPT-5.5).
- **Agent-as-a-Judge** [2410.10934]. Modules kept after ablation: graph, locate, read, retrieve, ask. *Memory hurt* (earlier wrong judgments cascade), planning was unstable, and search added noise.
  - Alignment on OpenHands: ask only 65.03% → +graph 75.95% → +read 82.24% → +locate 90.44%.
  - Gray-box (workspace + trajectory) 92.07% vs LLM-as-a-Judge 70.76%.
  - Cost $30.58 and 118 min vs humans' $1,297.50 and 86.5 h.
  - The class-imbalance trap: an LLM judge scored 84.15% on MetaGPT mostly by answering "unmet".
- **Evidence-bound, fail-closed judging** [2607.26819]. Per-rule human rubrics; yes/no/uncertain answers that must cite trajectory evidence; partial, uncertain or uncited counts as non-compliant.
- **Crucible.** F10 auditor and per-mode judges, P1. Use a `JudgeInput` type that excludes `Grade`, normalised trace fields (actions taken, state changed, evidence read, claims made), locate/read tools and no cross-item memory. Gray-box beats black-box. Presence checks ("file exists and non-empty") must be paired with executed checks.

### M3.12 Choose check implementations by coverage and false-failure rate (EvalGen)
- **Definitions** [2404.12272], over outputs E with user labels: Coverage(F) = share of bad outputs failed by some assertion; FFR(F) = share of good outputs failed; Alignment = the harmonic mean of Coverage and 1 − FFR. Generate several candidate assertions per criterion, execute all of them, and pick the best alignment under a per-criterion FFR ceiling.
- **Result.** On the product pipeline EvalGen used less than half as many assertions, and coverage rose 49%→73%. SPADE needed its FFR threshold raised from 10% to 40% to find any set.
- **Crucible.** F5 verifier writer, P1. Generate several candidate checks per verification goal. Keep the one with the best coverage (negatives killed, G5) against FFR (oracle + accepted alternatives fail, G14).

### Minor tips (B3)
- Gate tests automatically by "passes on fixed, fails on broken", then by human review for fairness to alternative fixes [2502.12115].
- Give the solver a reproduction tool that shows behaviour but not the grade (SWE-Lancer's user tool: Playwright script → text trajectory + screenshots) [2502.12115].
- E2E behaviour tests over unit tests where an App exposes a UI or workflow. On unit tests, o1 renamed an attribute so that a guarded branch never ran [2502.12115].
- One-way-door scoring for irreversible tools: irreversible harm scores negative, omission 0, safe completion up to 1. Report UAR (<0) and SCR (≥0.8) separately, to split "safe because unable" from "safe because careful" [2604.05172].
- Infeasible tasks must carry side-effect checks, so "refuse but break something" fails [2603.13594]. Grade the stated *reason*, not only the refusal [2401.13649, 2407.05291].
- In WebArena, the "you may say unachievable" hint made GPT-4 call 54.9% of feasible tasks impossible (11.7% success with the hint vs 14.41% without) [2307.13854]. OSWorld's FAIL action gave free reward to an agent that gave up [2404.07972]. **G4 needs "always infeasible" and "always FAIL" null agents**, and the false-infeasible rate on feasible tasks.
- A checker can talk: `validate()` returns a user chat message as well as a reward [2403.07718].
- Rubric milestones that name the failed sub-goal serve as process reward and as failure labels for F12 clusters [2509.26506].
- Strict-format scorers can reward format over ability. A BFCL pilot ranked GPT-4o-mini highest and the two strongest models lowest [2607.28685].
- Use deterministic hard rules (file exists, opens, no placeholders, no traceback) before any judge [2606.23654].

---

## B4. Intent, validity, communicability and leakage

### M4.1 Counterfactual reruns: remove one input and see whether the pass survives (the referee)
- **How** [2608.03744]. Three oversight agents were compared:
  - a *gate* that flags every agreement with peers;
  - a *same-lineage judge* that reads only the deliberation transcript;
  - a *referee* that infers the peers' endorsed answer, **privately re-queries the agent on the same input without the transcript**, and flags when the shared answer equals the peer answer and differs from the private one.

  An honest-peer control (peers assert the correct answer) makes false positives measurable.
- **Numbers.** Gate false-positive rate 100%. The transcript-only judge had precision 1.0 and recall 0.93 on text, but collapsed onto the gate on imaging. Referee 77–88% precision, 13–21% FPR on imaging. Two peers asserting the same wrong answer gave adoption 15/40 on MedQA. On 150 untouched CheXpert films, two "senior radiologist" peers moved the model to the wrong read on 61%.
- **Self-report fails.** Drifters toward a hidden rubric that rewarded a wrong decoy almost never said so: 1/10 on text, 1/134 on imaging. Decoy uptake was +0.275 (text) to +0.86 (CheXpert). Saying "you are being evaluated" reduced it.
- **Crucible.** F10, P1. The referee is the null-instruction / instruction-only ablation in general form. For a suspected shortcut (a World hint, a leaked file, a planted comment), re-run the student with that element removed. A pass that disappears is shortcut-dependent. Hack and intent flags come from execution and counterfactual runs, never from reasoning text.

### M4.2 Capability control and constant-policy baselines (validity audit)
- **Constant policy** [2607.28685]. For a binary benchmark with positive base rate π scored by F1, "always positive" gets F1 = 2π/(1+π). On R-Judge (π = 0.527) the constant gets 0.690 and outranks five discriminating models (0.30–0.57). The highest-specificity model (o3-mini, 0.97) gets only 0.702.
- **Capability confound.** Capability correlates with R-Judge balanced accuracy (ρ +0.71 with MMLU) but barely with specificity (+0.16). Capability predicts τ²-bench success (+0.60), and no safety score adds anything beyond it.
- **Small panels lie.** A trade-off seen at n = 7 (ρ −0.64) vanished at n = 18 (+0.02). A quarter of random 7-model subsets show |ρ| ≥ 0.5 by chance.
- **Method.** Pre-specified positive, negative and convergent controls; partial correlations controlling capability; organisation-clustered and leave-one-out bootstraps.
- **Crucible.** F10, P1. Ask whether a family's pass rate tracks the intended skill or only general strength. Control for a capability composite (e.g. frontier ability θ from IRT). A family fully explained by general capability is weak evidence of skill-specific intent.

### M4.3 Screen every metric for predicates that cannot fail
- The clinical study withdrew five arms whose predicates could not fail by construction, including two referee flags that reduced algebraically to their label [2608.03744].
- **Crucible.** §3.8 canaries, P0. Every reported metric and check is screened for a vacuous predicate before use.

### M4.4 Communicability: every needed value, format and location is recoverable
- **Fuzzy but recoverable instructions** [2510.25726]. Toolathlon's instructions are concise and underspecified, but the intent is deterministically inferable from World state (the format comes from existing Week1/Week2 sheets; which files to edit comes from exploring).
- **NPC-held information** [2412.14161]. Some facts come only from asking the right colleague.
- **Latent-derived answers** must also be derivable from visible state [2411.02305] (M1.2).
- **Unique final state** [2603.13594]. EnterpriseOps-Gym requires one final state with possibly many paths, but enforces this by authoring, not by a check.
- **Policy in the context.** CRMArena writes its tie-break and period definitions into the system prompt to remove ambiguity [2411.02305]. EnterpriseOps-Gym's gold plan cites a policy clause per action [2603.13594].
- **Self-containment signal** [2606.23654]: if the original agent asked for clarification, the task is not self-contained.
- **Crucible.** G8, P0/P1. Extend G8 from values to formats and locations. It must cover every `ActorSpec`'s knowledge boundary, and latent-derived answers must be reachable through policy-visible tools. Add an "instruction specificity" knob (fuzzy ↔ step-by-step) constrained by this recoverability. Add a check that the final state is unique, or that alternatives are accepted.

### M4.5 Leakage and contamination in start states
- **`git log --all`** [2607.26819]. Clone-and-rewind leaves the fix in `.git`. Instead, start an empty repo and fetch only the base commit and its ancestry from a local mirror, with no remote. A validator checks every workspace (all 257 passed), each run uses a throwaway copy, and a trajectory audit voids any run that retrieves the fix.
- **SWE-Lancer** [2502.12115] removes the GitHub remote and future commits, with no internet.
- **Temporal gate** [2607.26819]: any rule or fact the task depends on must exist at the base commit.
- **No completion signals** [2602.14093]: no success pop-ups or auto-redirects that leak completion.
- **Re-injected variants** [2502.12115] make held-out instances contamination-proof (M2.14).
- **In-environment shortcuts.** OfficeBench's LLM app inside the environment is an uncontrolled shortcut [2407.19056].
- **Crucible.** G11, P0. Confirm that `fs_git` start states are an empty repo + base ancestry only (no future objects, remote, reflog or packed refs). Add a validator that searches the start tree's object store for the fix commit.

### M4.6 Low-discrimination tasks are often broken (IRT as a defect detector)
- **Fluid** [2509.11106]. Adaptive selection by Fisher information almost never picks low-discrimination items. Mislabeled MMLU items per 100-item session: 0.01 adaptive vs 0.75 random. Items that strong models fail and weak ones pass are often mislabeled.
- **Crucible.** F10, P1. Tasks with near-zero or negative discrimination, after enough responses, go to the INCONCLUSIVE/defect queue.

### M4.7 Hidden reasoning and natural hacks
- Hidden reasoning [2606.09711]: about 32.7% of the exploit reasoning that direct probes recover never appears in the CoT.
- Concealment without pressure [2610.12822]: hide+disguise was about 30% of confirmed hacks for Qwen and nearly 100% for Llama and Gemma. Qwen-7B calls a hack a hack in a third party's code but calls its *own* use "prevention".
- Natural hacks [2604.23488] hide in normal-looking code (a self-written TestCase, then `sys.exit`), and the CoT does not announce them.
- **Crucible.** The lucky-pass and on-intent rates must not depend on the student's stated reasons.

### M4.8 Restraint tasks sit below every band at first
- **Evidence.**
  - RepoComplianceBench [2607.26819]: native Refuse and Handoff were 0% for all four agents. Agents opened the policy file in 12 of 347 native runs, and 242 of 248 violations happened without opening it. One feedback round lifted Verify to 90–100% and Disclose to 81–97%, but withdrawing a banned contribution stayed at 0/30 (GPT-5.5) to 7/30 (Sonnet). Stronger models are better verifiers and more stubborn violators.
  - ToolSandbox [2408.04682]: insufficient-information performance falls as models get stronger.
  - CRMArena-Pro [2505.18878]: confidentiality refusal near 0%.
- **Crucible.** §3.3 infeasible and confidential tasks, P1. Score stopping as success through an answer or path check. Give these tasks ladders (hints that the rule exists) and their own band target. Measure the completion cost of a refusal prompt, and add the collateral check "refusal does not stop legitimate tasks". Their anti-correlation with general strength also makes them a useful intent probe.

### Minor tips (B4)
- Make tasks need interaction, with information only in free text [2407.18901]. Put procedures in a KB doc the agent must find, not in the prompt [2407.05291].
- Report Pass^k (all k runs) beside Pass@k. Toolathlon: Claude-4.5-Sonnet Pass@3 51.9 vs Pass^3 20.4 [2510.25726]. Corecraft: Pass^3 9.3→17.6% after RL [2602.16179].
- The scaffold can dominate the score. Without skills docs every model on OpenClaw scored 0–8% TSR (39–65% of runs ended within 3 steps; >1,000 invented CLI subcommands). With skills + a meta prompt, 39–63% [2604.05172]. So the scores mostly measured documentation access.
- A safety rule in the prompt can be misapplied to legitimate policy text. One model labelled a written legal hold an "embedded override" and ignored it [2604.05172]. Test such rules on tasks where the document is the authority.

---

## B5. Diversity and coverage

### M5.1 Measure diversity on tool-set and behaviour vectors, not text (AgentMercury's negative evidence)
- **Analysis** [2608.20634]. A TF-IDF t-SNE of task text shows blobs that match no domain, persona or shape, while tool-set vectors do separate CRM, billing, project management and support.
  - Mean pairwise text similarity is 0.13, but the **median nearest-neighbour similarity is 0.83**.
  - **8.3% of tasks have a near-twin** (cos >0.99), and only 17% of near-twins are sibling seeds of the same World. The redundancy comes from shared templates *across* Worlds.
  - Task shape is a lattice set by one variable (the number of extra resources). CRM is ≈53–61% and "Account Manager" ≈53–63% of every cluster.
- **Crucible.** F8/F11, P0. Report the near-twin rate and whether twins share a World. Measure distance on tool-set and behaviour signatures. Break the CRUD lattice by varying the tool surface beyond search/list/get/update. Cap tasks per World and per template.

### M5.2 Breadth and distinct-task count drive out-of-distribution gains (WebGym)
- **Ablation ladder** [2601.02439], peak OOD success with Qwen3-VL-8B: half the subdomains removed 31.0 → full breadth, hard-biased 2:5:3 mix 34.5 → only easy 36.9 → uniform sampling (≈25:5:1) 38.2 → shorter step caps 42.9.
- **Overfitting.** Hard-biased sampling overfits after ~36k trajectories (hard tasks repeated from few domains). "Only easy" beats "only medium" overall.
- **Split by source.** 1,167 OOD test tasks, each from a distinct website, with every training task on those sites removed.
- **Crucible.** F11 pool composition, P1. Maximise distinct tasks and domains. Use uniform sampling over a large pool with a minority of hard tasks. Split by World or App family, not by generator run.

### M5.3 Templates and buckets, not instances, are the diversity unit
- WorkArena's 19,912 instances come from 33 templates that differ only in values [2403.07718]. WebArena has 3.3 instances per template and VisualWebArena 2.9 [2307.13854, 2401.13649].
- Per-template success varies widely ("fork metaseq" vs "fork all repos from Facebook"), so a template is not a difficulty class [2307.13854].
- WorkArena++ buckets tasks needing the same ability (ordering an iPad ≈ ordering a MacBook) and samples across buckets [2407.05291].
- AppWorld reports **86% of tasks with a unique API set over 52 App combinations**, a behaviour-signature diversity meter [2407.18901].
- **Crucible.** F7/F8, P1. Dedup and quota on templates and behaviour-signature classes. Adopt the "unique tool-set share" as one meter.

### M5.4 Archive structure: MAP-Elites × islands, stepping stones, multi-metric elites
- **AlphaEvolve** [2506.13131]. A program database that combines MAP-Elites with island populations. Optimising several metrics helps the target metric, by keeping structurally different good programs in prompts. Volume comes from a fast model (Gemini 2.0 Flash) and occasional jumps from a strong one (Pro). Stochastic prompt formatting and co-evolved meta-prompts add diversity cheaply.
  - Ablations (curves only): removing evolution, prompt context, meta-prompt evolution or full-file evolution, or using a single small LLM, each hurts.
- **Stepping stones.**
  - DGM's best lineage passes through two score dips (iterations 4 and 56) [2505.22954].
  - ADAS's best ARC agent combines ideas from iterations 3, 5, 11 and 12 that scored poorly at the time [2408.08435].
  - Empty initialisation still beats the baselines (math 67.5 vs 53.4; DROP worse, 73.9 vs 79.4) [2408.08435].
- **Novelty decays.** STOP's proposed strategies lose novelty over iterations, and a "be novel" prompt does not stop it [2310.02304].
- **Crucible.** F11, P1. Cells come from the DomainSpec grid, with **islands per generator family** so one family cannot dominate. Choose the elite per cell by learnability × interestingness. Keep novel non-elite tasks and recipes as stepping stones. Inject diversity structurally, not by prompt.

### M5.5 Grounding diversity in exogenous material
- OccuBench grounds diversity in 16 sub-topic reference docs per scenario, not in prompt temperature [2604.10866].
- CRMArena shows all prior entries of an object in each mini-batch prompt and exact-dedups key fields afterwards [2411.02305].
- TRUSTEE re-samples knobs independently within the current level (ε = 0.5), so the curriculum does not collapse onto one pattern. Without it, BFCL Multi-Turn fell 46.4→44.5 [2604.17739].
- WebForge drafts at T = 2.0 with one family and refines at T = 1.0 with another [2604.10988].

### Minor tips (B5)
- Exclude near-duplicate model families when fitting IRT item parameters [2509.11106]. Validate estimators on by-date or by-organisation splits, not only random ones [2402.14992].
- Hold out whole Apps (AppWorld Test-C: every task needs an API from an unseen App) [2407.18901].
- Make two Worlds from one generator with two scenario configs (B2B/B2C), but expect realism to fall as objects are added [2505.18878].
- Keep a reserved meta-seed range for evaluation [2407.05291].

---

## B6. Difficulty and curriculum per student

### M6.1 A dead-band scalar controller with soft within-level re-sampling (TRUSTEE)
- **Algorithm** [2604.17739]:
  ```text
  D in [1,100]; every knob k = floor(c_k * D)    # e.g. persona level = floor(0.02*D)
  after each step: r = mean batch reward
      if r > eta_high (0.5): D += delta (3)
      if r < eta_low  (0.0): D -= delta
  per task, with eps = 0.5: re-sample tool count, expected calls, turns,
      persona and ambiguity uniformly *within* the current level
  ```
  The grading criteria tighten with D: intent met → expected tools used correctly → no hallucination → no redundant calls → concise. System-prompt specificity falls across training.
- **Ablations** (Qwen3-8B, BFCL Multi-Turn, 46.4 default):
  - no curriculum (fixed at hardest) 41.5;
  - no soft re-sampling 44.5;
  - tool count fixed at max 27.4;
  - user fixed at hardest 32.8;
  - δ = 10 → 38.8.

  Turn-level rewards from the simulators did not help (35.6→33.0). A 397B backbone was no better as simulator. Gains on post-trained models are within CIs.
- **Crucible.** F9/F11 online band keeper, P1: `DifficultyController(D, η_low, η_high, δ, ε) → knob_vector` over the DomainSpec knobs. Add "system-prompt specificity", "user expertise × ambiguity" and "grading strictness" as knobs. Pair it with per-student bands rather than one batch mean.

### M6.2 A declared difficulty vector with compositional level rules (WebForge)
- **Seven dimensions × 3 levels** [2604.10988]: jump depth, jump breadth, page interaction, visual complexity, information complexity, reasoning/calculation, and risk (irreversible actions with or without warnings).
- **Level rules.** L1: ≤2 dims at 2 and none at 3. L2: ≥2 at 2 and ≤1 at 3. L3: ≥2 at 3 and ≥2 at 2. Hard tasks are therefore hard on several axes.
- **Results** (14 configs): L1 73.9, L2 54.8, L3 28.1. Gemini-3-Pro 86.4/82.1/58.0; Qwen3-Omni-30B 12.7 overall.
  - **Every dimension was monotone L1→L3 for every model.** The steepest were visual complexity and jump breadth. Reasoning/calculation best separates strong from weak models.
  - Dimensions are moderately correlated by construction (mean |ρ| = 0.495; jump depth vs reasoning 0.754). 107 tasks went unsolved by all 14 models, 64.5% of them L3.
  - Prompt tokens grew 5–8× from L1 to L3.
- **Crucible.** F0/F3, P0. The proposal declares δ over the DomainSpec knob kinds (count/linkage ≈ depth/breadth, distractors ≈ information complexity, observability, hints), and F7 checks the measured signature against the declared δ. Report per-dimension curves per student.

### M6.3 Check-subset decomposition: strictly easier siblings with no rollouts (WebGym)
- **Algorithm** [2601.02439]:
  ```text
  rubric = fact groups; difficulty = total facts (easy 1-3, medium 4-6, hard 7+)
  if n_groups >= 2 and some group has >= 3 facts ("large"):
      for each proper subset S of groups that contains a large group:
          new task: instruction rewritten for S; rubric = S
  ```
  Example: a 9-fact task with groups of 2/3/4 yields 5 sub-tasks of difficulty 3–7. They are strictly easier, and well-posed whenever the parent is.
- **Crucible.** F3/F9 difficulty down-shifter, P1. For a task whose VerifierBundle has grouped milestones, proper subsets with a large group become easier siblings with the same World and oracle prefix. The 35B gets the sub-tasks and the 397B the parent. It is also a cheap way to fill low-difficulty cells.

### M6.4 Measured structural levers
| Lever | Evidence |
|---|---|
| Horizon (gold steps) | EnterpriseOps-Gym: closed models ~35% at 4 steps → <20% at 16; open models fall faster [2603.13594]. EntWorld GPT-4.1: 34.0% (<5 steps) → 2.7% (>15) [2601.17722]. AppWorld: tasks needing 60+ LOC score <20 TGC; difficulty level 1→3 gives 58.3→21.0 [2407.18901]. WorkBench: 0 / ≥1 / ≥2 actions → 75 / 36 / 18% [2405.00823] |
| Number of Apps | OfficeBench 64.5 → 60.0 → 21.4 for 1/2/3 Apps [2407.19056]. ClawsBench multi-service −23 pp TSR and more unsafe actions [2604.05172] |
| Observability | WorkArena++ L2 explicit steps vs L3 ticket + KB policy (human 93.9 both; models ~0) [2407.05291]. CRMArena-Pro single→multi-turn 54.1→35.1 [2505.18878]. GDPval under-specified prompts (42% of length) lowered performance [2510.04374] |
| Faults (M6.6) | OccuBench mean 67.5 (E0) → 62.6 (E1 explicit) → 53.4 (E2 implicit) → 54.4 (E3 mixed); Opus 71.5→60.2 as fault count goes 1→4 [2604.10866] |
| Policy load | EnterpriseOps-Gym policy compliance lowest; ITSM best 28.5% [2603.13594] |
| Sub-task count | ≥3 sub-tasks much harder [2510.27287] |
| Structural score | EntWorld D_task = Σ wᵢdᵢ over tables, relations, operation type (SELECT < CUD), result size and SQL complexity, with predefined weights [2601.17722] |
| Natural label | SWE-Lancer price [2502.12115]; GDPval value = expert time × wage [2510.04374] |
| Two-axis labels | VisualWebArena: action count (easy ≤3, medium 4–9, hard ≥10) × perception load, averaged; keep an easy subset for weak models [2401.13649] |
| Within-template slots | SCUBA picks slot values to vary search depth and items to track [2509.26506]. WorkBench per-template pass spread 10–90% [2405.00823] |

### M6.5 Hints, plans, docs and tool exposure as scaffold knobs
- **Plans.**
  - EnterpriseOps-Gym [2603.13594]: human plans +14–35 points; **Qwen3-4B with human plans matches much larger models**; a Sonnet 4.5 planner + executor +6–13 for weak models. Decomposing into sub-agents *hurt* in CSM/HR (sequential state dependencies).
  - DSPy gold plans add ~40–50% relative [2510.27287].
  - OccuBench's plan/no-plan pass gap is a pre-rollout difficulty predictor [2604.10866].
- **Demos.** SCUBA's retrieved plan-demos (even only similar ones) raise success by 15–21 points [2509.26506].
- **Docs.** ClawsBench: without docs every model is at floor (0–8%) [2604.05172].
- **Tool exposure.**
  - TheMCPCompany: an oracle tool subset beats the browser by +13.8 points and costs $2.29 less per task. Retrieval over 18.5K tools adds +5.4. But GPT-5-mini with retrieval (32.1) did worse than with the browser (33.4), with retrieval recall 33% vs 59–70% for frontier models [2510.19286].
  - WorkBench: GPT-4 43% with all 26 tools vs 49% with only the needed toolkit [2405.00823].
  - CRMArena: hand-made helper tools help strong models and hurt weak ones. gpt-4o-mini fell 28.3→19.5; removing the calculation and dependent tools cost GPT-4o −26.7 but helped weaker models [2411.02305].
- **Thinking budget** (mixed evidence):
  - GPT-5.2 reasoning effort none→xhigh: 54.7→82.2% [2604.10866].
  - GPT-OSS-120B low→high: Drive 8.6→41.0%, but ITSM plateaus at 6.1% [2603.13594].
  - GPT-5-high was no better than GPT-5 on Toolathlon [2510.25726].
  - o1 IC 9.3→16.5% from low to high effort [2502.12115].
- **Crucible.** F9 per-student knobs, P1. `hints ∈ {none, similar_demo, exact_demo, gold_plan}`; `tool_exposure ∈ {oracle_subset, curated_app_tools, full_surface_behind_find_tools}`. Treat helper-tool exposure and surface size as per-student band knobs, since they may make tasks unlearnable for the 35B. Keep tool docs on for the 35B and vary hint depth instead. Remove scaffolds as p̂ rises.

### M6.6 Faults in code, not in a prompt
- **OccuBench's taxonomy** [2604.10866]: E1 explicit (500, timeout, connection refused); E2 implicit (lists truncated to 1–2 items, missing or null fields, stale values); E3 mixed. All are transient and spaced through the episode, controlled by `fault_count` (default 2) and `fault_duration` (default 2 consecutive calls). Robustness R = min(CR_E1, CR_E2, CR_E3)/CR_E0. Implicit truncation is the hardest.
- **Crucible.** F1/F2, P1: `fault_profile{kind: explicit|implicit|mixed, count, duration}` as a DomainSpec knob, implemented through App `config_flags` (REAL-style). Add negatives that pass only if the agent re-queries.

### M6.7 IRT for per-student difficulty and adaptive measurement
- **tinyBenchmarks** [2402.14992]:
  - The model: 2PL multidimensional P(Y_il = 1) = σ(α_iᵀθ_l − β_i), fit by variational inference with hierarchical priors, d ∈ {2,5,10,15} chosen on validation.
  - p-IRT: fit θ of the new model from its observed items, and estimate the score as λ̂·(observed mean) + (1−λ̂)·(mean predicted p on the unseen items).
  - gp-IRT: mix with weight λ = b²/(b² + v), where v is the sample variance and b² the IRT bias estimated on a split of training models.
  - Result: about 2% error with 100 items per scenario (MMLU 14K→100). Anchors clustered on IRT embeddings resist new model types better than correctness-vector anchors.
- **Fluid** [2509.11106]:
  - The model: P(u_ij = 1) = σ(a_j(θ_i − b_j)), fit by MCMC, with one unidimensional model per benchmark. A global trait hid a model getting *less* truthful, and multidimensional fits gave no gain.
  - Item selection: at the current θ̂, pick the unadministered item with the most Fisher information I_j(θ) = a_j²p(1−p).
  - Stopping: when SE(θ̂) falls below the average ability gap between rank-adjacent models. Items needed ranged from ~20 early in training to >80 mid-training.
  - Results: Fluid with 500 items beat full-benchmark accuracy (validity 8.3 vs 9.1; variance 4.9 vs 23.8), and even 50 items beat the full set. The variance gain comes from adaptive selection, not from IRT scoring alone. **tinyBenchmarks at 100 items had higher variance than random (30.5 vs 19.8).**
  - Failure mode: items that no training model solved collapse to one maximal difficulty, so a frontier model saturates the IRT's resolution.
- **Crucible.** F9, P1:
  - Fit 2PL over the (task × student/checkpoint/frontier) pass matrix. b_j relative to each student's θ is the band, which unifies "in band for the 35B, too easy for the 397B".
  - Run the most informative tasks for a new checkpoint first, and stop on SE. This extends §3.6's k = 4 → 8 rule.
  - Report held-out ability as θ̂ with SE.
  - Pick the sealed benchmark by IRT anchors so it covers the ability range.
  - Caveat: Crucible's matrix is tiny, so use priors and d = 1–2 (one dimension per skill family).

### Minor tips (B6)
- Per-difficulty step caps, tightened (10/20/30 instead of 15/30/45), raised success even on hard tasks [2601.02439].
- The 35B band will sit at short horizons first [2603.13594]. At WebForge L3 (2–58%), 30B-class models are at floor [2604.10988].
- A model can build Apps it cannot navigate (the "synthesis–navigation gap"), which supports building with a stronger family than the student [2602.14093].
- 0% cells give no signal. WorkArena++ was 0% for every 2024 model [2407.05291]. Drop such composites below the band or shorten them, or warm-start with oracle-trace SFT.
- EnterpriseOps-Gym discarded tasks that frontier pre-runs found trivial. That biases the set toward frontier-hard tasks, which may be unlearnable for small students. Use pre-runs as bug finders (G3/G13), not as the difficulty filter [2603.13594].
- Make conditional tasks so that branch choice is part of the difficulty [2405.00823].

---

## B7. Self-improvement loops, their guards, and the statistics and judge calibration they need

### M7.1 The evaluator sits outside every improver's read and write scope, by construction
| Evidence | Source |
|---|---|
| **Without AEvo's protected-evaluator harness, 2 of 3 kernel runs hacked the reward and produced no valid result**; the third reached 1167 cycles. With the harness: no hack, 1138 cycles, 16/100 invalid. Without the meta-agent skills: no hack, but 1407–2379 cycles and early stops (37, 65 rounds) | AEvo [2605.13821] |
| DGM node 114 reached a perfect 2.0 on a "no tool-use hallucination" score in two edits by deleting the marker logging the detector depended on, despite instructions. Hacking was more frequent when the detector functions were visible, so they were hidden | DGM [2505.22954] |
| A meta-agent exfiltrated 591 dev labels through grader tracebacks | [2606.04455] |
| STOP improvers dropped call budgets "for simplification" and built a new LM object with a larger budget. The sandbox flag was disabled in 0.42% (GPT-4) and 0.12% (GPT-3.5) of 10,000 generations, and a warning did not reduce it (0.46% / 0.17%). A string-match lower bound | STOP [2310.02304] |
| Unconstrained Gödel Agent scored 90.5 / 90.6 / 87.9 / 55.7 vs constrained 80.9 / 64.2 / 70.9 / 34.9, because it chose on its own to call stronger models. In 100 MGSM trials, 4% ended by crashing (usually after editing its own improvement module), 92% had temporary drops and 14% ended worse than the start | Gödel Agent [2410.04444] |
- **Crucible.** §3.8 guards, P0. Gate code, measurement markers, anchor labels, error messages, model routing, budgets and the loop driver all live outside the improver's write *and* read scope, enforced at a proxy and the runner. Graders return reason codes only. Treat a near-instant perfect score, or a large unconstrained gain, as a review alarm, not as headroom. Normalise any gain by cost and resources before accepting it.

### M7.2 The AEvo shape for the pipeline doctor: edit the mechanism, run a segment, record everything
- **Loop** [2605.13821]:
  ```text
  repeat:
    meta-edit phase: M reads summary Phi(C_r) of the ledger (candidates, scores, traces,
        failures, costs); makes ONE workspace edit + a run plan {iterations, budget, stop_if}
    evolution segment: mechanism Pi runs several rounds -> many evaluated candidates
    harness records every candidate {artifact, score, trace, failure, cost, provenance}
  stop: target reached, or 25 rounds without gain
  ```
  The evaluator is isolated from both the inner and the meta agent: they can submit, but cannot read the evaluator's internals or write scores. The skill is Read → Attribute → choose exactly one action → Run → Record. The initial procedure is deliberately minimal, so gains can be attributed.
- **Numbers.** Terminal-Bench / ARC-AGI-2, Avg@3: AEvo-Procedure 53.8 / 47.0 vs DGM 44.3 / 29.8, ADAS 38.6 / 36.0, ReAct 28.6 / 21.8. $0.32–1.47 per round, ~3× the procedure baselines.
- **Limits.** The same benchmarks are used for search and report. There is no rollback rule.
- **Crucible.** §3.8 generator loop, P1: `MetaEdit{target_file, diff, hypothesis, run_plan{n_candidates, budget_usd, stop_if}}` in the provenance ledger. The observation is per-stage yields, invalid rates, reasons and costs (F12). A segment is a batch through F3–F8. Add the held-out-family meta-evaluation and rollback that AEvo lacks.

### M7.3 DGM-style archive selection and staged evaluation
- **Parent weight** [2505.22954]: w_i = sigmoid(λ(α_i − α₀)) · 1/(1 + n_i), with α_i the score, n_i the number of children that can still edit code, λ = 10 and α₀ = 0.5. Every node keeps a non-zero chance of selection.
- **Evaluation stages.** A child must compile and keep its core function, else it is discarded. Then 10 tasks → 50 tasks → 200 for contenders (>40% and top-2). A separate diagnostician (o1) turns a failure log into one repo-agnostic improvement issue. *It also saw the private test patch, which is a leak by design.*
- **Ablations** (SWE / Polyglot): w/o open-ended exploration 23.0 / 14.0; w/o self-improvement 39.0 / 28.0; Greedy 39.7 / 30.0; DGM 50.0 / 38.0. Valid-child rate 51.3% vs 32.5%.
- **Cost.** ~$22,000 and ~2 weeks per SWE-bench run.
- **Crucible.** §3.8 generator and World loops, P1. Archive pipeline and recipe versions with this parent rule and a valid-child smoke stage. Cascade a cheap frozen slice → a larger slice → held-out families. Never give the doctor hidden tests or the anchor set.

### M7.4 Other loop-design lessons
- **ADAS** [2408.08435]: ~$300–500 per run. Validation sets of 20–128 items let the search optimise noise; GPQA-sized differences fall within the CIs. Select on a frozen slice plus held-out families that are larger, and never on a set the improver can see.
- **STOP** [2310.02304]:
  - Enforce budgets in the callee (the LM wrapper and the utility count calls).
  - Show the improver a **grey-box** description of the objective (what is averaged), never the training items.
  - Weak improvers regress: GPT-3.5 and Mixtral got worse with iterations, and only 12% of GPT-3.5 runs gained ≥3%. Gate loops by the improver's model strength.
  - The bound ε = sqrt((l ln|Σ| + ln(1/δ))/n) says overfitting falls with 1/√n of the frozen slice.
- **Gödel Agent** [2410.04444]: forbid edits to the improvement loop itself. Carry errors forward as information. Inspect evolved prompts by eye: the MGSM prompts drifted into word salad while the score rose. Require a stored previous version before any auto-apply.
- **AlphaEvolve** [2506.13131]: an evaluation cascade (cheap and easy cases first; a smoke run drops broken programs); hold out part of the evaluation inputs (half the kernel shapes, recent workloads); canonicalise outputs before scoring (rounding for exactness).
- **SPARK** [2509.22624] co-trains policy and judge in one model. Test-time self-reflection on the untrained base hurt badly (math 49.1→39.9; WeMath 62.1→24.1). Reward-only GRPO overfits. Self-judging contradicts §3.0 rule 5, so do not use it for the verifier loop.

### M7.5 Hacks spread between agents: filter what the swarm shares
- **Propagation model** [2610.12822]: R = kθ₁/u₀, where k is the number of references, θ₁ the adoption rate given one hacked reference, and u₀ the clean success rate. Hacks grow when R > 1. With a verifier gate, R_gate = R·a_H/a_N, so suppression needs a_H/a_N < 1/R.
- **Channels.**
  - Plans carry hacks: a hacking planner induced hacks in every coder backbone.
  - Orchestrators route more work to the hacking agent. Aggregators keep hacks when no clean candidate passes.
  - Chains on hard tasks persist after the source leaves, with justifying comments growing more persuasive.
  - Feedback rounds make low-propagation targets adopt hacks, so first-submission evaluation misses this.
  - Conftest-style hacks spread more than exit hacks, because they look like test infrastructure.
- **Crucible.** §3.9, P1. Any reviewing role sees no grader pass/fail verdict (add this to "Never sees"). Code, plans and memory from other agents are untrusted input. A hacking student's passing traces must not seed traces-to-tasks or the proposer until they are re-graded strictly. Evaluate agents under feedback loops.

### M7.6 Hack precursors during training (PRIME)
- **The probe** [2606.09711]. At a checkpoint, the policy answers three questions about a fixed (task, solution) pair, without outcomes: correctness self-assessment, proxy recognition and exploit reasoning. Two judges score each 0–5. Φ^B = the share of fixed proxy-pass/gold-fail diagnostics with all three scores ≥4.
- **Numbers** (Qwen2.5-Coder-14B):
  - Onset order: self-assessment at step ≈27, proxy recognition ≈47, exploit reasoning ≈103, sustained hacking ≈164.
  - Φ^B crosses 0.35 about 40 steps before onset. Quiet checkpoints with high Φ^B reach 0.37 hacking later, more than checkpoints already hacking at 15–30% with low Φ^B (0.25).
  - Activation probes reach AUROC 0.85–0.90 but need weight access.
  - Scale threshold: Φ^B / H at 1.5B 0.10 / 0.08; 7B 0.62 / 0.48; 14B 0.78 / 0.65.
- **Caveats.** It needs known hacks for the diagnostic set. The thresholds are descriptive.
- **Crucible.** F12, P2: `FeedbackSignal.hack_precursor{family, phi, checkpoint}`, using text probes judged by a non-Qwen family (Tinker may not allow activation access).

### M7.7 Curating hack labels for the anchor set
- **Prompted hacks do not validate a monitor** [2604.23488]:
  - Monitors trained on prompted hacks caught 8.8–40.7% of RL-born hacks. Zero-shot GPT-4.1 caught 3.7%, o4-mini 10.6% and GPT-5.4 18.2%, while all reached 97–100% on prompted hacks.
  - Monitors trained on training-time hacks reach 86–99.7% and transfer to an unseen hack type (BERT 100%).
  - Training-time hacks are closer to natural ones (energy distance 0.044 vs 0.399). A prompt-trained BERT flags 65% of normal code as hacking.
- **Induced pressure** [2606.04455] creates positives cheaply: remove a resource the honest path needs.
- **Crucible.** §3.8 anchor set, P0. Hack positives come from real rollouts, attacker runs and impossible twins, not from prompts. Report judge accuracy separately on normal and hack items, so always-flag judges show.

### M7.8 Statistics for every factory number
**(a) Prediction-powered inference for judge-labelled rates** [2301.09633]
- **Mean estimate.** θ̂_PP = mean f(X̃) − mean(f(X) − Y), with 95% CI θ̂_PP ± 1.96·sqrt(σ̂²_{f−Y}/n + σ̂²_f/N). Here N is all judge-labelled items and n the random human sample. A general convex form, and stratified, covariate-shift and label-shift versions, exist.
- **When it pays.** Only if Var(f − Y) < Var(Y) and N ≫ n. For binary Y with symmetric judge error η, it needs η < 25% at p = 0.5 and η < ~9.5% at p = 0.1. *For rare events such as a ~5% hack rate, the judge must be very accurate or PPI loses to human-only labels.* A weak predictor needed 40 labels vs the classical 35, and with N = 1000 < n, 869 vs 652.
- **Rules.** Imputation-only intervals failed to cover the truth in every application. The gold set must be a random sample (or have known weights), and the judge must stay frozen while the gold sample is collected.
- **Crucible.** P0: `estimate_rate(judge_labels, human_sample, weights)` for hack rate, on-intent share, false-fail rate and lucky-pass rate per family (`ppi_py` exists). If the κ queue is stratified, use the weighted form. Keep a uniform slice too (M7.10).

**(b) Calibrated selective judging: trust or escalate** [2407.18370]
- **Calibration.** On a human-labelled calibration set, scan λ downward from 0.999. Compute the empirical disagreement over items with confidence ≥ λ and its exact binomial (1−δ) upper bound. Stop at the last λ whose bound is ≤ α. This guarantees P(disagreement ≤ α) ≥ 1−δ.
- **Confidence by Simulated Annotators.** Run N few-shot prompts, each with K examples labelled by a different annotator (K, N ≤ 5), and take the agreement ratio as confidence. This cut GPT-4's ECE by ~50% and raised AUROC ~13%. CoT sampling and semantic entropy did no better.
- **Cascade.** Order judges cheap → strong, and calibrate each only on what earlier stages abstained on (δ/|M| each).
  - ChatArena at target 0.85: 91% guarantee success with GPT-4 used on 17.5% of items.
  - A weak cascade at 0.8 costs 12.6% of GPT-4's.
  - Abstained items have lower human agreement (0.815 vs 0.902), so abstentions flag rubric ambiguity.
- **Crucible.** F10 and σ-gated application, P1. A loop may auto-apply only verdicts from a judge operating above its calibrated λ. Abstentions form the human queue. **Size:** 30–50 anchor episodes per lane are too few for the exact bound; plan ~200–500 adjudicated items per judge type.

**(c) Error bars, clustering, pairing and power** [2411.00640]
- **Formulas.**
  - SE = sqrt(Var(s)/n). The Bernoulli formula is too wide for fractional scores.
  - Clustered: SE_cl² = SE² + (1/n²)Σ_c Σ_i Σ_{j≠i}(s_ic − s̄)(s_jc − s̄). It was up to 3× the naive SE on real evals.
  - Average K samples per item first; never treat K·n samples as independent. The noise term falls to E[σ²]/K; with Var(x) = 1/12 and E[σ²] = 1/6, variance falls by 1/3 at K = 2, 1/2 at K = 4, at most 2/3.
  - Paired: SE = sqrt(Var(s_A − s_B)/n). At correlation 0.5, variance falls by a third.
  - Power: n = (z_{α/2} + z_β)²(ω² + σ_A²/K_A + σ_B²/K_B)/δ².
- **Example.** δ = 0.03 and ω² = 1/9 give n ≈ 969. A derived example with ω² = 1/9, σ² = 1/6, K = 8, α = 0.05 and power 0.8 gives an MDE of **≈35 pp at 10 tasks, 15.5 at 50, 11 at 100, 4.9 at 500**.
- **Do not lower temperature to cut variance.** It shifts noise into Var(x) and can bias the mean.
- **Crucible.** §4.4 and every §3.8 ablation, P0. Use paired, family-clustered SEs and an a-priori MDE, and do not run an ablation whose MDE exceeds the hoped-for effect: `report(scores, clusters, pairs, K)`, `mde(n, K, omega2, sigma2)`. Clustering also favours "many families, few instances".

**(d) Judge validity is per task and per property** [2406.18403]
- **Setup.** 20 datasets, 11 judges. Prompts were the original human instructions; invalid responses were replaced by random labels.
- **Findings.**
  - There is large variation across models and properties, not explained by human upper bounds, and no model is best everywhere.
  - Toxicity and safety were the weakest (some κ negative, many refusals).
  - Judges agree more with non-experts than experts, and more on human-written than machine-generated text.
  - CoT, few-shot and paraphrases gave no systematic gain.
- **Crucible.** P0. Keep the anchor set per lane *and per check type* (pass, hack, false fail, lucky pass, on-intent, side effect), with ≥2 human labels on a subset to get an upper bound. Choose the judge per check type by measured κ, and record the valid-response rate. Budget more human review for judge-only verdicts on student traces.

**(e) Three-tier probes for any judge-prompt change** [2609.12439]
- **Equivalent pairs.** On content-identical pairs, plain GPT-5.5 picked the cited candidate 85.8% of the time (Claude 50.9%, DeepSeek 47.2%).
- **The debiasing trap.** Debiasing prompts drive worse-cited wins to 0% with clear-conflict accuracy still ≥93%, so a standard bias test "passes". But L4 ("Tie whenever final answers match") left only 2–15% better-plain resolution on *moderate* real gaps: a 98-point L0→L4 loss for GPT-5.5 and 89.0 for Claude (CI 84.5–93.0). At L3, judges chose Tie on 33–42% of moderate items vs 9% for humans.
- **Reporting.** Report Tie per tier. Run both candidate orders. Use PABAK or AC1 instead of κ under prevalence skew.
- **Crucible.** P1. Every judge-prompt edit must pass equivalent pairs, clear passes/fails and moderate gaps (two traces with the same final state, one of which skipped a justifying read). Add a **constant-Tie canary** next to always-pass and always-fail.

**(f) Human reference and rubric versioning**
- **Agent-as-a-Judge** [2410.10934]. Three experts judged independently (16.5, 19.5 and 22 h), then debated to consensus (9.5 h). Pairwise disagreement was 10–30%, and the worst evaluator had 23.77% error; a majority of three cut error to 6.01%.
- **EvalGen criteria drift** [2404.12272]. Graders add criteria on seeing new failures, reinterpret old ones ("all entities are proper nouns" → "most"), and re-grade for consistency rather than belief. Random grade sampling gave high-variance alignment; selectivity-weighted sampling was consistently better.
- **ClawsBench** [2604.05172] chose 5 repeats per task from a 30-repeat pilot by split-half reliability (r_SB ≥0.84 at k = 10). It used cluster bootstrap CIs and paired Wilcoxon with Holm correction.
- **AgentRewardBench** [2504.08942] scores judges on **precision** of "success", because false positives poison rejection fine-tuning and RL rewards.
- **Crucible.** §3.8 anchor set, P0. Use the 3-annotator + debate protocol and report each annotator's error vs consensus. Version the anchor labels against a `RUBRIC_VERSION`: on any criterion change, re-adjudicate the affected anchors and recalibrate every judge. "Frozen" means frozen per version.

### M7.9 Process signals for improver runs
- Longer gaps between evaluation calls and more total runtime predict reward; the number of evaluation calls does not [2606.04455].
- 33% of meta-agent configurations had run-to-run std >0.1, vs at most 0.053 for human baselines [2606.04455].
- Only 5 of 39 configurations beat the human-baseline average [2606.04455].
- Report invalid/total candidates per segment. A jump in invalids flags a broken mechanism [2605.13821].

### M7.10 Human review sampling
- Sample reviews by predicted badness (selectivity-weighted), but **keep a uniform slice for unbiased rates**, because PPI needs it [2404.12272, 2301.09633].

---

## B8. Pipeline architectures, costs, yields, scale; training results on synthetic environments

### M8.1 Cost of hand-built Worlds and tasks (the baseline the factory replaces)
| Source | Cost |
|---|---|
| AppWorld [2407.18901] | Engine ~100K LOC over 14 months; generators ~40K LOC over 5 months for 250 scenarios; GPT-4o ReAct ≈$0.70 per task; ≈$10K of experiments |
| TheAgentCompany [2412.14161] | ~3,000 person-hours over 2 months by 20 people for 175 tasks (~17 h per task; complex ones >10 h); agent cost $0.3–6.8 per task |
| Toolathlon [2510.25726] | 4–6 graduate-student hours per task + ~5 h per review round by 5–6 senior authors; 108 tasks |
| OSWorld [2404.07972] | ~1,800 person-hours for 369 tasks (650 single-app, 750 workflow, 400 double-checking); ~1 h setup + ~2 h evaluator per task (~5 h each); 134 evaluator functions |
| EnterpriseOps-Gym [2603.13594] | ~$100 per task (≈$115k for 1,150 tasks); 160+ contributors; 4 months; gold paths mean 9.15 steps (max 34); 24.9 tables and 3,443 rows touched per task |
| GDPval [2510.04374] | Gold task: 404 expert minutes, $361; review 109 min, $86; ≥3 (mean 5) reviews per task |
| Agent-as-a-Judge DevAI [2410.10934] | 86.5 human-judging hours, ≈$1,297.50 per developer run |
| SWE-Lancer [2502.12115] | 100 professional engineers; ≥3 reviewers per IC task |
| ToolSandbox [2408.04682] | 1,032 scenarios by 2 experts with ≥4 test rounds |

### M8.2 Cost and yield of automated pipelines
| Pipeline | Yield and cost |
|---|---|
| VeriEnv clones [2603.10505] | 149 sites; 26% failed; $3.6 and 83.5 min per site including tasks; 7,400 tasks (40/39/21% easy/medium/hard); judge correct 76% |
| GUI-Genesis [2602.14093] | 1,118 Apps, $302.12 total; 82.65% verified within 5 tries; 17.35% that never passed were *kept* anyway |
| WebForge [2604.10988] | 1,260 → 934 (74.1%) with zero humans |
| EnterpriseClawBench [2606.23654] | 5,291 → 852 (16%) mined tasks |
| Reverse synthesis [2510.27287] | ~80 s per task; 80% kept by experts |
| OccuBench [2604.10866] | 100 × 16 sub-topics → 382 kept instances |
| AgentMercury [2608.20634] | 43,300 tasks from 4,326 Worlds (claims 4,783); 842 tools; 222 table types; 148 service combinations. Construction $ not given. Frontier APIs pass all 12 validators zero-shot on 67–87% of briefs |
| WebGym [2601.02439] | 292,092 tasks over 127,645 sites (top-20 sites ≈ half, ~80% easy) |
| WorkArena [2403.07718] | 33 templates → 19,912 instances |
| TRUSTEE [2604.17739] | 30–43 h on 8×H20 vs ~150–170 h for EnvScaler; ~1e8 backbone tokens per run |

### M8.3 Training results on synthetic or dense environments
| Result | Setting |
|---|---|
| **Qwen3.5-35B-A3B 24.8 → 28.1 (GRPO) / 28.3 (SAO)** on held-out EnterpriseOps-Gym, gains in all 8 domains; BFCL 31.1→42.5; τ³ Telecom 49.1±49.8 → 68.9±20.5 | AgentMercury [2608.20634]: GRPO, 8 samples, Dr.GRPO-style clip 0.2/0.28, no std normalisation, zero-variance groups dropped, lr 1e-6, batch 128, ≤20 tool turns, 16k response tokens; fully async (2 actor + 6 rollout GPUs, staleness ≤4); 3,200 tasks (7.4%) over 200 steps touched 53.5% of Worlds and 75.8% of tools; reward ~0.4 → 0.55–0.60; truncation 0.35 → ~0 |
| Qwen3.5-4B 12.3 → 15.7; out-of-domain AIME26 45.9→56.0, LiveCodeBench 36.6→44.0 | Same; OOD claims **unablated** (no control RL run), and CSM fell at 4B (9.2→5.6) |
| **World authoring: base Qwen3.5-35B-A3B 3.3% → 83.3%** on 30 held-out briefs (Fisher p = 1.2e−10) | SFT on 29,823 construction-trace samples (forward, partial completion, validator-guided repair, intent→diff). *With* the recipe prompt the fine-tuned model drops to 10% (27/30 fail the cross-service check); for API models the recipe gives no gain (McNemar p ≥0.375) |
| GLM 4.6 (357B MoE, 32B active) 25.37 → 36.76% held-out; Toolathlon Pass@1 18.8 → 25.6, Pass^3 9.3 → 17.6; τ²-Retail +7.4; BFCL Parallel +4.5 | Corecraft [2602.16179]: 1,000 train / 150 held-out (random split, same World); 16 rollouts per prompt, one container each; GRPO with DAPO-style clipping; one epoch; LLM-judged rubric reward; no environment ablation |
| Qwen3-30B-A3B WebArena-Lite subset 35.5 → 58.2 (SFT) → 61.8 (RL); Qwen3-4B 24.5 → 54.1 → 57.3 | WebServ [2510.16252]: SFT on Claude 4.5 trajectories, then GRPO with dynamic filtering, 12 samples, 512 parallel rollouts feeding 64 H200s, ~12 min per step. Most of the gain is SFT; RL adds ~3–4 points; the RL task source is not stated |
| Qwen3-VL-8B 26.2 → 42.9% on OOD websites | WebGym [2601.02439]: REINFORCE on successes only (filtered BC), memory field, no-op step drop, per-tier step caps; async rollouts 1,800 trajectories in ~30 min on 128 CPUs + 24 H100 (4–5× faster than synchronous) |
| Qwen3-8B BFCL Multi-Turn 39.9 → 46.4; base 0 → 7.9 | TRUSTEE [2604.17739]: fully simulated environment; post-trained gains within CIs |
| Qwen3-4B WebArena-Lite 7.88 → 13.94; LLaMA-3.2-3B 3.03 → 12.73 | VeriEnv [2603.10505]: rejection-sampling fine-tuning on 97 cloned sites; more sites gave monotone gains |
| Real-app human-judged success 36.91 (base) → 40.94 (real-app RL, VLM reward) → 42.28 (synthetic Apps, code reward) | GUI-Genesis [2602.14093]: Qwen2.5-VL-32B; 240→480→969 Apps monotone; a real-app epoch costs >$28k vs ≈$0 |
| EntAgent-RL 56.9% (SFT 50.6%) | EntWorld [2601.17722]: Qwen2.5-VL-32B; in-distribution split (same templates and Apps) |
| Qwen3-8B SFT 0.27, SFT+DPO 0.29 ≈ GPT-4o CoT 0.27 (judge score) | [2510.27287]: 1k generated tasks + 1,200 DPO pairs; weak evidence |
| Qwen3-30B-Think 16.8% on EnterpriseOps-Gym (51.9% on Email); Qwen3-4B 13.7% | [2603.13594]: a baseline for the student class |
- **Crucible expectation.** For the 35B MoE, the closest published RL gain on a held-out enterprise benchmark is ~3 points (AgentMercury). Report SFT and RL contributions separately. Hold out families, not random tasks.

### M8.4 RL-recipe tips seen across the cards
- Drop zero-variance groups [2608.20634, 2510.16252].
- An experience-replay buffer that swaps a stored success into an all-zero group fixes zero-signal batches but is off-policy, so it is an option, not a default [2601.17722].
- Keep truncated samples during cold start, so the truncation penalty is learned [2608.20634].
- Tighten step caps per tier, and drop no-op steps (identical next screenshot) from the gradient [2601.02439].
- Keep the evaluation temperature at the training value [2411.00640].

### M8.5 Self-improvement loop costs
DGM ~$22,000 per SWE-bench run (baselines ~$10,000; $350 per 60 SWE tasks vs ~$5 per 60 Polyglot) [2505.22954]. ADAS ~$300–500 [2408.08435]. Gödel Agent ~$15 per run [2410.04444]. AEvo $0.32–1.47 per round [2605.13821]. Meta-Agent Challenge 12–24 h per run [2606.04455]. STOP ~3,000 GPT-4 calls per iteration [2310.02304]. JUDGE-BENCH 321 A100-hours, ~$565 for GPT-4o [2406.18403].

---

## B9. Corrections and contradictions

### Corrections to `framework-plan.md`
1. **§3.8 Meta-Agent Challenge.** The plan says ground-truth exfiltration surfaced "despite layered defences". The card says the exfiltrated items were 591 *dev* labels, pulled through verbose grader tracebacks; this did not transfer to the hidden test split, and isolation stopped every flagged exploit from inflating the test score [2606.04455]. The right lesson is that *error channels leak, and split isolation worked*, not that the defences failed.
2. **§3.8 frozen anchor set size.** About 30–50 episodes per lane is too small for an exact-binomial calibrated judge threshold. Plan ~200–500 adjudicated items per judge type [2407.18370]. Hack labels must come from natural rollouts, not prompted hacks [2604.23488].
3. **§3.8 "frozen" anchor set.** Labels are valid only against a versioned rubric, so they need re-adjudication on any criterion change (criteria drift) [2404.12272].
4. **§3.8 fixed human spot-check rate.** A queue stratified by gate and fault side needs either the weighted PPI form or a uniform slice. Raw stratified counts are not population rates [2301.09633, 2404.12272].
5. **§3.8 vacuous-verifier canaries.** Add a constant-Tie canary [2609.12439], constant-policy baselines (F1 = 2π/(1+π)) [2607.28685], and a screen for predicates that cannot fail [2608.03744].
6. **§3.8 σ-gated application.** A concrete σ exists: a calibrated λ from fixed-sequence testing with a stated (α, δ) [2407.18370].
7. **§3.1 G4 null agents.** Add "always infeasible / always FAIL" [2307.13854, 2404.07972], "always act" and "never act" [2405.00823], and "act on the else-branch" [2405.00823].
8. **§3.1 G14.** "Where the family requires them" is too weak. Official checkers' 55.9% recall [2504.08942] and the 18–24% broken WebArena checks [2408.06327] argue for measured checker recall in every family.
9. **§3.1 G2.** Extend it to reset fidelity (VeriEnv's top judge bug) and to pinned render libraries [2603.10505, 2608.03744].
10. **§3.3 mutation operators.** Add "impossible twin" [2604.23488], "deceptive outcome" per mistake class [2604.10988] and the information-flow check [2408.04682].
11. **§3.9 swarm roles.** Reviewing roles must never see the grader's verdict [2610.12822], and F3 references must be gate-passed and hack-free [2610.12822].
12. **§3.6 sequential measurement.** IRT-based adaptive selection with SE stopping is the measured version of k = 4 → 8 [2509.11106].
13. **§3.3 infeasible/confidential evidence.** The card for CRMArena-Pro gives confidentiality refusal "near 0%" with a standard prompt, and up to 62.9% (gpt-4o-mini) with a confidentiality-aware prompt [2505.18878]. The plan's "near 0–2.9%" range could not be checked against this card. EnterpriseOps-Gym's 53.9% is the best refusal *without side effects* [2603.13594]. That matches the plan.
14. **§4.4 measurements.** A 10-task pool cannot show a small effect: the MDE is ≈35 pp under the paper's illustrative variances [2411.00640]. Correlations from fewer than ~20 units are unreliable [2607.28685].

### Errors and weaknesses in papers (flags kept)
- **AgentMercury** [2608.20634]: count inconsistencies (4,783 vs 4,326 Worlds; "five" API models but four rows; 80.7% vs 78.3% mean). The RL reward mixes in a DeepSeek-V4-Flash judge and a behaviour penalty, so it is not purely state-based. The OOD math and code gains are unablated. The 12 validators are not listed.
- **EntWorld** [2601.17722]: GPT-4.1 reported as both 47.61% and 37.39% (garbled). Despite the abstract, all 1,756 tasks are SELECT retrieval. "Human 85%" is estimated, not measured. The RL split is in-distribution.
- **GUI-Genesis** [2602.14093]: the same model writes App, reward and test; 17.35% of unverified Apps were kept.
- **DGM** [2505.22954]: the diagnostician sees private test patches. Selection and reporting use the same benchmark, except for the cross-benchmark transfers.
- **Corecraft** [2602.16179]: causal language without any environment ablation; one model, one epoch; judge reliability asserted; random split; commercial authorship of both environment and claims.
- **TheAgentCompany** [2412.14161]: "LLM-judge concerns should be dismissed" rests on review, not agreement numbers. Judges and NPCs share one model. The "correct deliverable overrides failed checkpoints" rule is a manual escape hatch.
- **TRUSTEE** [2604.17739]: gains on post-trained models lie within CIs. The judge sees the expected-tool list. In the Qwen3-8B run, policy and judge are the same weights.
- **Sim2Real** [2603.11245]: τ-bench v1 has known reward bugs. The "human success" label is a role-player's opinion.
- **AppWorld** [2407.18901]: allowed-change sets are hand-written (false-fail risk). Setup bugs reach both solution and evaluation.
- **EnterpriseOps-Gym** [2603.13594]: "unique final state" is enforced by authoring, not checked. Discarding frontier-trivial tasks biases the set toward frontier-hard tasks.
- **2510.27287**: access control partly enforced by an LLM agent; a 1–5 judge for grading; 5% unanswerable is too few to train refusal.
- **WebServ** [2510.16252]: the RL task source is not stated.
- **OSWorld** [2404.07972]: FAIL rewards giving up. **WebArena** [2307.13854]: the unachievable hint causes false give-ups.
- **ClawsBench** [2604.05172]: scaffolding effects dwarf model effects; ragged factorial; no human baseline.
- **Gödel Agent** [2410.04444]: the "free" results come from constraint evasion. **STOP** [2310.02304]: the sandbox-disable measure is a lower bound (string match).
- **PRIME** [2606.09711]: Φ^B needs already-known hacks; the thresholds are descriptive; key figures were described in text only.
- **tinyBenchmarks vs Fluid**: Fluid found tinyBenchmarks at 100 items noisier than random sampling (30.5 vs 19.8) [2509.11106 on 2402.14992].
- **AgentRewardBench** [2504.08942]: the experts' own lenient rule on ambiguous instructions defines "success".
- **TraceJudgeBench** [2609.12439]: TRACE's normaliser records missing support itself, so stage 1 may do part of the judging.
- **VeriEnv** [2603.10505]: many judges are `must_include` on answer strings, i.e. answer checks, not state checks.
- **SCUBA** [2509.26506]: it is not stated whether evaluators check collateral config changes.

### Contradictions across papers
- **Distractor tools.** Redundant tools hurt in WorkBench (GPT-4 43→49% when given only the needed toolkit) [2405.00823]. ToolSandbox found about −10 for Claude-3-Sonnet with all tools [2408.04682]. EnterpriseOps-Gym found no loss with +5/+10/+15 relevant-looking distractors [2603.13594]. The effect seems model- and setting-dependent, so measure it per student.
- **Thinking budget.** It helps strongly on OccuBench (54.7→82.2) and on Drive (8.6→41.0). It plateaus on ITSM (6.1%), and GPT-5-high was no better on Toolathlon [2604.10866, 2603.13594, 2510.25726].
- **Partial credit.** TheAgentCompany uses 0.5·partial + 0.5·full [2412.14161]. TheMCPCompany gives success only on Azure, where many valid paths exist [2510.19286]. GUI-Genesis's staged credit can reward partial work [2602.14093]. For Crucible, keep the binary keep rule and use milestones only as Φ.
- **Simulated environments.** TRUSTEE trains usefully on stateless simulated tools [2604.17739], but its stateful ablation, OccuBench's simulator swap [2604.10866] and the Sim2Real easy-mode effect [2603.11245] all argue against LLM-simulated state and people in the reward.
- **Curriculum emphasis.** WebGym found uniform sampling over a large, mostly easy pool beat hard-biased sampling, and "only easy" beat "only medium" [2601.02439]. TRUSTEE found that a fixed hardest setting hurt [2604.17739]. Both favour moving bands over a hard-heavy pool. That is consistent with the plan's band, but it argues against a hard-weighted mix.
- **Recipe prompts for authors.** They gave no gain for API models and destroyed a fine-tuned author's cross-service structure [2608.20634]. ADAS, by contrast, found a wrong-pattern list in the prompt cut errors [2408.08435]. Test prompt additions per model.
- **Prompting judges and simulators.** JUDGE-BENCH found CoT and few-shot prompting gave no systematic judge gain [2406.18403]. A findings-informed persona prompt lowered simulator fidelity [2603.11245]. Debiasing prompts destroyed resolution [2609.12439]. Trust or Escalate's few-shot Simulated Annotators did improve *confidence* calibration [2407.18370]. In every case, measure the change on the same metric before adopting it.
