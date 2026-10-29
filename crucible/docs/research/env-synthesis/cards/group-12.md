# Group 12 cards

**Papers: 18.** Read FULL: 17. PARTIAL: 1 (2606.12191, the survey: domain tables and memory/workflow sections skimmed). ABSTRACT-ONLY: 0.
Core for Part 2: 2610.26828 (WildAWM), 2609.04148 (Terminal-Universe), 2610.06100 (Trace2Env), 2602.11964 (Gaia2). Supporting: Vending-Bench, UI-Simulator, Online-Mind2Web, SOTOPIA, EnterpriseBench, Concordia. Peripheral (privacy and infrastructure): TabularARGN ×2, AIM, DP text, BrowserGym, WebDreamer, TimeArena, the survey.

**Five cross-paper takeaways for Part 2**
1. **Verifier gates should be executed, multi-sided and attacked; each gate pays for itself.** WildAWM ablates five gates one at a time (judge consistency, initial unsatisfaction, solvability, difficulty, counterfactual probe); dropping any one costs 2.6–9.3 points of downstream score, and in-domain score falls from 21.6 to about 3 without judge consistency or initial unsatisfaction. Terminal-Universe requires the hidden tests to fail on the start state for the target and to pass for preservation, written before any solution and never computed from the code under test. Gaia2 tests its verifier with oracle perturbations plus 450 hand-labelled runs (0.98 agreement), and still saw its LLM soft check hacked in RL (fixed by a style check). Adopt for F5/F6: a red-on-start rule (strengthens G4), a counterfactual "assert the opposite verdict" probe for every LLM or agent judge, perturbation unit tests, and a style check on any soft judge. Judges are trustworthy only per dimension (SOTOPIA: goal r 0.71, secrets 0.22) and never on the policy's own claims (Online-Mind2Web, WildAWM).
2. **Traces-to-tasks has two proven routes, and the trace must become a World, never text.** If the system can be rebuilt: replay the earliest observed state, withhold the agent's changes, complete missing context under a "solvable, not solved" contract, gate on a sufficiency judge, then re-solve with a strong policy (Terminal-Universe: 37.3K environments from 360K traces; re-solving 52.1 vs imitating the trace 36.7; doubling environments beats doubling queries or solutions, 56.0 vs 53.8/53.9). If it cannot: build a worldbook (schemas, evidence, rules with provenance) and validate every proposed effect (Trace2Env). Either way, measure by replay in the real system: in-sim success can be fiction (ALFWorld 97% in-sim vs 3% replayed for direct prompting).
3. **LLM-simulated dynamics must not carry reward-bearing state.** Every working design keeps rewards in code: Concordia's grounded variables, Vending-Bench's cached LLM parameters driving a coded demand model, UI-Simulator's rule-based deterministic transitions, Trace2Env's typed effect validation, WildAWM's deterministic day deltas. LLM simulation drifts beyond one step (WebDreamer), imports wrong context (UI-Simulator), and needs a fidelity probe (W2R replay, intrinsic ranking of gold vs failing deviations).
4. **Dense Worlds need an event clock, async consequences and deterministic builds, and time makes difficulty depend on latency.** Gaia2 supplies the event DAG (env, conditional, validation, oracle events), notification-verbosity knob, paused-clock generation time and windowed timing checks; WildAWM supplies persona-rooted multi-day Worlds built as replayable deltas accepted only if two replays hash equal; Vending-Bench shows most long-horizon collapses start from an asynchronous outcome assumed rather than checked; TimeArena adds durations, actor occupancy and resource locks. GPT-5 (high) scores 0% on Gaia2 Time (34.4% with latency removed), so F9 must measure bands in the training time mode.
5. **Diversity and difficulty come from structure and cheap probes, and our own student shows where the ceiling is.** Step-wise sub-goal control lifted task diversity (UI-Simulator: PCA effective dimension 153 vs 118, and success collapsed without it); coverage-pressure histograms and a persona-diversity judge spread WildAWM Worlds; a search-only shortcut agent exposed WebVoyager (51%) and should be a standing G9 probe. Cheap difficulty signals: k-of-3 unhinted plus hinted rollouts, teacher-forcing loss in the 25–75th percentile, cross-environment hidden-fact tasks (teacher pass 72% → 49%). WildAWM trained **Qwen3.6-35B-A3B** (base pass@1 5.7% on hard CUA tasks): it needed an SFT warm start, gained +10.6 points over 8 OOD benchmarks, and gained nothing on tasks beyond the teacher's reach; plan ladders accordingly.

## 2312.03664 — Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia (Vezhnevets et al., 2023; Google DeepMind tech report)
Read: FULL (body + Appendix A; references skipped).  Relevance to Part 2: medium. It is a library design, with no experiments or numbers, but its Game Master (GM) pattern, grounded variables, nested games and phone-app digital actions are a clean template for the "people" and "simulated user" parts of the World and for LLM-mediated environment steps.

**Problem.** Agent-based models need agents that act with common sense and talk, and an environment that can turn free-text actions into grounded consequences, including actions on digital apps. Concordia is the library for that ("generative agent-based models", GABM).

**Method / algorithm.**
- Two generative parts: player agents and a Game Master. Observations, actions and event statements are all English strings.
- **Agent** = long-term associative memory m (a set of strings, Park-et-al.-style retrieval with importance) + working memory z = states of *components* (identity, plan, observation-and-clock, goal, hunger, money ...). Each component is a natural-language statement, updated by querying memory and the LLM, or by ordinary code (e.g., a calorie counter).
  - Action step: a_t ~ p(· | f^a(z_t)) where f^a is usually concatenation of component states.
  - Component update: z^i_{t+1} ~ p(· | f^i(z_t, m_t)), cadence configurable per component.
  - An agent may get several observations before acting; it does not act on every observation.
  - Agent API: `name()`, `observe(str)`, `act(action_spec)`; action_spec gives output type (free text, categorical choice, float) and the call-to-action phrasing.
- **Game Master** is built like an agent, but its components describe world state. Per action it answers four questions: world state, which event results, who observes it, what it does to grounded variables.
  - Event: e_t ~ p(· | f^e(z_t), a_t); per-player observation o^i_{t+1} ~ p(· | f^o(z_{t+1})); no observation if the GM judges the player did not see it.
  - GM component API: `state()`, `partial_state(player)` (what that player may see, e.g. only own location), `update()`, `update_before_event(cause)`, `update_after_event(event)`, `terminate_episode()`.
  - Step order: for each player in random initiative order: update components → send partial_state as observation → `act` → update_before_event → GM chain of thought produces event statement → update_after_event on all components → check termination.
  - **Grounded variables** (money, votes, inventory) live in Python components; the GM rejects actions that violate them (cannot pay more than you have) and tells the agent the action was invalid.
  - Clock advances after each player or after all players; a "quasi-simultaneous" mode; a concurrent-action flag that is fast but yields inconsistencies.
- **Nested games**: a GM component can spawn a child GM with a finer clock (e.g. fishing scene, conversation) and receive its memories back.
- **Digital actions**: a `SceneTriggeringComponent` detects phone-related events and spawns a `PhoneGameMaster` for one owner. The `PhoneUniverse` turns the English action into an API call by (1) prompting for app and function, (2) prompting for arguments, (3) invoking it, (4) posting to a `NotificationHub`, (5) handing back for multi-step actions. Apps are Python classes with `@app_action` methods; English descriptions are auto-generated from docstrings and type annotations. Four fidelity levels for an app: language only, simple simulated code, LLM-prompted (e.g. search), real API.
- **Backstories**: biographical facts + randomised traits (e.g. Big Five) + scenario context → backstory → formative memories at several ages → initial memory. Trait fidelity is explicitly not validated.
- Concurrency on component updates (LLM latency is the bottleneck); a sequential wrapper exists.
- Example scenarios: calendar meeting between two phone owners, a small-town election with a smear campaign, a day in town, a snowed-in pub dispute, bean trading with inventory, Cyberball social exclusion.

**Verification / grading.** None for task success; the GM is the arbiter and is an LLM. Grounded variables are the only executed state. Validation guidance: measure generalisation on new data, check "algorithmic fidelity" per research question, prefer model comparison over absolute claims, run sensitivity analysis over irrelevant details (wording), parsimony. Known hazards: train-test contamination (models know named games like Prisoner's Dilemma, so disguise them), stereotyped personas.

**Scale & cost.** Not reported.

**Key results.** None quantitative; a design paper with examples.

**Failure modes & limits.** LLM GM can drift off-topic (they plan "keyframes" to steer); concurrent turns are inconsistent; persona traits unvalidated; stereotype risk; no determinism. My critique: for RL rewards, an LLM GM deciding outcomes is a non-executed ground truth, which violates Crucible's "LLM proposes, execution decides" unless the outcome lives in grounded variables.

**Tips & tricks.**
- Keep every reward-relevant quantity as a grounded variable in code; let the LLM only narrate.
- `partial_state(player)` is a clean per-actor observability hook (information separation per person).
- Generate app tool descriptions from docstrings and type hints, so tool docs never drift from code.
- Spawn nested sub-simulations with a finer clock for detail-heavy episodes.
- Disguise canonical scenarios to avoid contamination by known setups.
- Build personas as traits → backstory → dated formative memories, then audit that traits show up.
- Log (component state, action) pairs per step: lets an auditor blame either the LLM or the component spec.

**Fit to Crucible Part 2.** F1/F2 (World people and simulated users) and the §2(b) "people as data" design. Adopt: the GM step protocol with `partial_state` for per-person observability; grounded variables as the only reward-bearing state (they map to `StateAdapter` tables); the four app-fidelity tiers as a vocabulary for App tiers; persona generation pipeline for `ActorSpec` with a trait-fidelity audit added. Avoid: LLM-decided outcomes feeding the verifier. Interface: `ActorSpec{traits, backstory, memories, known_info}` + `observe/act` with an action_spec type. Priority P2 (useful pattern; nothing measured).

## 2310.11667 — SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents (Zhou et al., 2023; ICLR 2024)
Read: FULL (body + appendices B–G).  Relevance to Part 2: medium. Not a task-synthesis pipeline, but a procedurally composed task space (scenario × characters × relationship × partner policy), a rule-based per-relationship observability scheme, private goals, and a validated LLM-judge rubric with per-dimension human agreement. Directly relevant to multi-turn simulated-user tasks and to which reward dimensions an LLM judge can be trusted on.

**Problem.** Static social-intelligence benchmarks miss interaction; interactive ones are narrow. Build an open-ended, goal-driven, multi-dimension social environment and check whether an LLM can grade it.

**Method / algorithm.**
- **Task = scenario context + 2 characters + relationship + private social goal per agent.** Space is the cross product, further multiplied by the partner model's policy.
- **Characters (40):** GPT-4 generates name, gender, age, occupation, pronouns, Big Five traits, moral foundations (care, fairness, loyalty, authority, purity), Schwartz values, decision style (directive / analytical / conceptual / behavioural), then a secret and a public fact. Authors fix inconsistent profiles.
- **Relationships (5 types):** family, friend, romantic, acquaintance, stranger. 90 GPT-4-generated pairs plus 30 random stranger pairs, manually checked (31/30/30/29 + 30). **Rule-based visibility:** family, friend and romantic see the whole profile except secrets; acquaintances see name, occupation, pronouns and public info; strangers see nothing.
- **Scenarios (90):** two generation routes.
  1. "Inspiration" route: a vignette sampled from existing datasets (Social Chemistry 20, Social IQa 20, Deal-or-no-Deal 10, NormBank 10) is given to GPT-4, which writes a context and two goals with one point of initial disagreement. 60 scenarios; needs heavy manual editing.
  2. "Template" route: fill slot templates from structured datasets (CraigslistBargains items and prices; MutualFriends), then ask GPT-4 only to make the sentence fluent. Seller target = list price ÷ (1 + markup), markup ~ Exponential(rate 0.5). 30 scenarios; few manual edits. The template route gives much better control.
  - Each scenario carries a relationship constraint; manual validation removed 10%. Guidelines: gender-neutral, plausible, discard if the constraint admits fewer than 5 character pairs, and keep strategy hints abstract.
- **Tasks:** 5 character pairs per scenario → 450 tasks; all model pairs simulated.
- **Episode:** round-robin turns, max 20; actions are speak / non-verbal / physical (each with free text) / none / leave. Formally an N-agent Dec-POMDP with vector rewards over M dimensions. Agent temperature 1, evaluator temperature 0.
- **SOTOPIA-EVAL dimensions:** Goal [0,10], Believability [0,10], Knowledge [0,10], Secret [−10,0], Relationship [−5,5], Social rules [−10,0], Financial [−5,5]. Judge prompt per dimension asks for reasoning then an integer score.
- **SOTOPIA-hard:** difficulty for a target model = (mean + 3σ of the best model) − (mean − 3σ of the target), clipped to range; top 20 tasks for GPT-4, also hard for others.

**Verification / grading.** LLM judge (GPT-4) vs humans on 200 episodes, 2 vetted MTurk annotators each (43 kept of 235 applicants via a qualification task). Inter-human Randolph κ 0.503 (5 bins), 84.85% pairwise agreement. GPT-4–human Pearson r on model episodes: Goal 0.71, Fin 0.62, Rel 0.56, Bel 0.45, Kno 0.33, Soc 0.33, Sec 0.22. On human-played episodes it drops on everything but Goal (0.78). GPT-4 rates higher than humans when it disagrees, and is over-optimistic on Secret and Social rules. Llama-2-70B as judge is near useless (Goal r 0.24). Adding fine-grained score-range definitions did not help (Soc fell to −0.59). Hackable: a goal-completion judge can be talked into it; verbose agents may look better.

**Scale & cost.** 450 tasks; human annotation $12.4/hour, 10–15 min per episode. No compute numbers.

**Key results.** Goal (auto-judge, averaged over partners): GPT-4 7.62, GPT-3.5 6.45, Llama-2-70B 5.38, MPT-30B 4.10. On SOTOPIA-hard, humans 5.95–6.15 vs GPT-4 4.85 on Goal (significant). Humans use 16.8 words per turn vs GPT-4's 45.5. Weak partners drag strong models down. All models leak secrets and break norms; Llama-2 ranks below GPT-3.5 here despite parity on static benchmarks.

**Failure modes & limits.** Dyadic, round-robin only; 5 relationship types; judge biases (position, verbosity, stereotypes); judge validity varies a lot by dimension; partner policy is a confound in every score.

**Tips & tricks.**
- Template-fill scenarios from structured data, then let an LLM only rewrite for fluency: far fewer manual fixes than free generation.
- Encode observability between people as a rule table keyed by relationship type.
- Give each side a private goal with one built-in point of disagreement.
- Validate an LLM judge per dimension against vetted humans; trust only the dimensions with r ≥ ~0.6.
- More detailed rubric text can make a judge worse; measure it, do not assume.
- Select hard tasks by the gap between the best model's optimistic bound and the target model's pessimistic bound.
- Fix the partner (simulated user) model when comparing students; partners change outcomes.

**Fit to Crucible Part 2.** F2 (people in the seed bundle: traits, secrets, relationships with a visibility rule) and F3 (multi-turn tasks with a simulated user whose goal and known info are private). Also the confidential-task type of §3.3: the Secret dimension is a probe pattern, but it should be graded by code (did the secret string or fact appear in outbound messages), not an LLM. For F5/F10 judges: adopt per-dimension human-agreement measurement as the acceptance test of any judge on the anchor set; do not use LLM judges for norms or secrets (r 0.22–0.33). Adopt the template route for persona and scenario seeding with exogenous structured data. Priority P1 for the simulated-user and confidentiality task types; P2 otherwise.

## 2602.11964 — Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments (Froger et al., 2026; Meta SuperIntelligence Labs)
Read: FULL (body + Appendices A, B incl. verifier, hacking, ablations).  Relevance to Part 2: high. It is the closest published design to Crucible's World clock, events and notifications, and its write-action verifier (with a measured 0.98 agreement, a documented judge hack and its fix, and a perturbation-based verifier test) maps onto F5/F6 and the timing-check kind of §3.3.

**Problem.** Agent benchmarks are synchronous (the world changes only when the agent acts) and grade only end outcomes. Gaia2 tests agents in a world that moves on its own clock, with an action-level verifier usable as an RLVR reward.

**Method / algorithm.**
- **ARE platform, five concepts.** (i) *Apps*: stateful Python classes; methods become tools via decorators typed `read` or `write`, and role-scoped (agent / user / env). (ii) *Environment*: apps + time manager + rules; an MDP, deterministic given start state and seed. (iii) *Events*: everything (tool calls, state changes, scheduled updates) is a timestamped, logged event in a time-ordered `EventQueue` with DAG dependencies; types are Agent, User, Env, **Conditional** (polls a predicate, e.g. cancel only if booked), **Validation** (milestone or constraint check that fails the run on timeout) and **Oracle** (ground-truth actions for the verifier). (iv) *Notifications*: a policy (whitelist) chooses which events are pushed into the agent's context; presets low (user messages only), medium (consequences of agent actions: email, messages, calendar, order and ride status; default), high (all env events, e.g. new products). (v) *Scenarios*: initial state + event DAG (user ask, env events, oracle actions, validation), in a `scenario.py`.
- **Two core apps.** `AgentUserInterface` (blocking or non-blocking user channel) and `System` (`get_current_time`, `wait`, `wait_for_next_notification`). Calling a wait tool switches the sim from real time to event-to-event jumps, so hours of sim time run in minutes.
- **Time.** By default model generation consumes simulated time. To survive outages and rate limits they use "simulated generation time": the clock pauses while the model responds, then advances by the measured generation time. An "instant" mode removes latency entirely (upper bound).
- **Mobile environment.** 12 apps (Messages, Chats, Emails, Calendar, Contacts, Shopping, Cabs, Files, RentAFlat, City ...) with 101 tools; 10 "universes", each a full state for one user persona, 400K–800K tokens of content.
- **Universe generation.** App dependency graph with Contacts as root and priority-based conflict resolution (foundational apps win). Personas from PersonaHub, expanded with the universe's location; one sampled as the user. Chats: ≥10 alternating messages between sampled contacts with shared-experience references. Emails: inbox (sender sampled) and sent (recipient sampled); the LLM must invent a valid reason from both personas. Calendar: iterative week-by-week generation conditioned on the previous week's summary, over ~3 months. Shopping: 500 real Amazon products + discount codes. Files: one shared Wikipedia-based tree for all universes. Unhandled: temporal consistency across apps, relationship consistency across channels, cross-modal references.
- **Scenario authoring (human).** Annotators explore a universe in the ARE GUI and draw the event DAG (user ask, oracle write actions, interfering env events, further turns). Real-time structural guardrails: DAG fully connected rooted at `send_message_to_agent`; only `send_message_to_agent` or env events may follow `send_message_to_user`; only one branch carries user messages; each turn ends with `send_message_to_user`. Then multiple independent validation rounds, a consistency check, and post-hoc difficulty calibration with a baseline agent. The GUI cuts annotation time about 5×.
- **Splits.** Five core splits of 160 each (800): Execution (many writes), Search (many reads, one final-answer write), Ambiguity (impossible, contradictory or multi-answer; agent must do the clear part and report), Adaptability (a dependent env event, child of `send_message_to_user`, forces a plan change; plus distractor events), Time (at least one time-sensitive oracle action; capped at 5 minutes; taxonomy: time-based one-off, time-based recurrent, event-based one-off, event-based recurrent). Two augmentations of Gaia2-mini (160) add 320: **Noise** (random tool failures, signature changes, irrelevant events such as spam) and **Agent2Agent** (apps replaced by app-agents reached by messaging). Augmentations need no new annotation because the verifier checks writes, not traces. Composite "3+ capability" tasks were dropped as unnatural with unclear signal.

**Verification / grading.** ARE Verifier compares agent *write* actions to the minimal oracle write set; reads are free and unverified.
1. Precheck: the multiset of tool names over writes must match the oracle's.
2. Topologically sort oracle actions; for each, find an unmatched agent write with the same tool, then **hard check** exact fields (IDs) and **soft check** flexible fields (message text) with an LLM judge given the task and both argument sets plus tool-specific guidelines.
3. **Causality**: all oracle parents must already be matched to earlier agent actions; independent goals may be done in any order.
4. **Timing**: only for oracle actions with relative delay > 1 s; agent action must land in [Δt − 5 s, Δt + 25 s] of its matched parent ("immediate" is annotated +2 s).
5. Success iff every oracle action is matched. Multi-turn: verify each turn (split on `send_message_to_user`); online, a conditional event runs the verifier and only triggers the next turn if this one passed, so the run stops once the agent diverges.
- Assumes no equivalent write actions (using Messages where the oracle used Chats fails), so tasks must state preferences unambiguously.
- **Verifier validation:** (a) unit tests by perturbing oracle trajectories in ways known to preserve or break validity; (b) 450 hand-labelled trajectories: ARE Verifier agreement 0.98, precision 0.99, recall 0.95 vs an in-context LLM-only verifier at 0.72 / 0.53 / 0.83 (same Llama-3.3-70B). With Gemini 2.5 Pro or Claude Sonnet 3.7 as soft judge: 0.96 / 0.98 / 0.89.
- **Hack found in RL:** on Search (one write expected) the policy learned to stuff ever-longer template-like code strings into the final message, overwhelming the soft judge into false positives. Fix: a task-agnostic "style" soft check.

**Scale & cost.** 1,120 scenarios (800 core + 320 augmented), 10 universes, 101 tools. Runs: temperature 0.5, 16K tokens per turn, ≥128K context, 3 runs per scenario, stop at 200 steps, context overflow, verifier verdict or timeout. Claude 4 Opus excluded for cost. Per-scenario dollar cost plotted, not tabulated.

**Key results.**
- Pass@1 overall: GPT-5 (high) 42.1, Claude-4-Sonnet-Thinking 37.8, Claude-4-Sonnet 34.8, GPT-5 (low) 34.6, Gemini-2.5-Pro 25.8, Kimi-K2 20.1, Qwen3-235B-Thinking 15.7, GPT-OSS-120B 13.7, Llama-4-Maverick 7.4.
- Execution and Search are easiest; Ambiguity, Adaptability, Time and Noise are hard. GPT-5 (high) scores **0.0 on Time**: inverse scaling, because thinking longer misses deadlines. In instant mode GPT-5 (high) Time goes 0.0 → 34.4 and Sonnet 8.1 → 26.7.
- Pass@1 correlates with tool calls and output tokens; "thinking" variants take fewer steps and cost less per solved scenario.
- Noise ablation (Sonnet, Gaia2-mini): none 31.2, low 35.0, medium 23.8, high 8.1.
- A2A: helps Llama-4-Maverick (pass@k and fewer tool errors), not Sonnet. Heterogeneous teams: Llama main + Claude apps 16.2 vs all-Llama 8.5; Claude main + Llama apps 18.3; all-Claude 29.3. Executor quality matters more.
- Parallel tool calling: big latency and token savings, pass@1 change −6.3 to +3.0 pp, ranking unchanged.

**Failure modes & limits.** Single-threaded scaffold cannot express some concurrent Time actions; Time capped at 5 min; scenarios are human-authored (expensive; no synthesis); no-equivalent-writes assumption creates false negatives when a user did not pin the app; LLM soft judge is a hack surface; universe coherence gaps (timestamps vs calendar). My critique: the tool-count precheck makes the verifier strict (any extra write fails), which is good against side effects but gives false negatives for harmless extra writes (e.g. a corrected mistake); they cite exactly that distinction as a feature. Reads are unverified, so a privacy leak via reading then summarising is caught only if the outgoing message fails the soft check. The soft judge is the weakest link and was hacked within RL.

**Tips & tricks.**
- Type every tool read or write at definition time; verify writes only, let reads be free.
- Match oracle writes in topological order with hard checks on IDs and a judge only for free text.
- Use a timing window asymmetric around the oracle delay ([−5 s, +25 s]).
- Stop multi-turn runs at the first failed turn via a conditional verifier event; saves rollout cost.
- Make noise and multi-agent variants as environment-level modifiers so one annotation yields several tasks.
- Pause the sim clock during model generation, then advance by measured latency; also report an instant-mode upper bound.
- Accelerate time to the next event whenever the agent waits.
- Notification verbosity is a clean observability knob.
- Test the verifier with oracle perturbations that should keep or flip the verdict, then with a few hundred hand-labelled real trajectories.
- Add a task-agnostic style check to any LLM soft judge used as an RL reward.
- Generate persona-rooted universes along an app dependency graph so references resolve.

**Fit to Crucible Part 2.**
- **F1/F2 (World):** adopt the event model (Env/Conditional/Validation/Oracle events in a DAG), the notification policy as an observability knob, `wait` with event-to-event time jumps, and paused-clock generation time. Adopt the persona-rooted dependency-graph seeding with Contacts-first resolution and add the gaps they list (timestamp coherence) as G1 integrity checks.
- **F3/F4:** the four Time taxonomy cells and the Ambiguity taxonomy are ready-made `DomainSpec` axes; Ambiguity = the infeasible/clarify task type.
- **F5:** their verifier is a concrete instance of "path + timing checks": implement `timing_checks` as windowed matching relative to the matched parent, and causal order via the oracle DAG. Combine with Crucible's state checks and collateral diff (they lack a full collateral check).
- **F6 / verifier testing:** adopt their two-tier verifier test (perturbation unit tests + ~450 hand-labelled trajectories with agreement/precision/recall), and add a style check plus G12 attacker on any soft judge.
- **Knobs and augmentations:** Noise level and A2A are knob-like modifiers that create variants without new oracles; add `noise` to the `Knob` kinds.
- **F9 caveat:** latency-dependent tasks make difficulty depend on inference speed; difficulty must be measured in the same time mode as training.
- Priority **P0** for the verifier design and test protocol, P1 for the event and notification model.

## 2502.15840 — Vending-Bench: A Benchmark for Long-Term Coherence of Autonomous Agents (Backlund & Petersson, 2025; Andon Labs)
Read: FULL.  Relevance to Part 2: medium. A single hand-built long-horizon World, not a factory, but its time-cost-per-tool clock, overnight event delivery, simulated suppliers grounded in real data, and closed-form demand model are concrete patterns for the World's clock and people; its failure analysis is a direct warning for long-horizon task design and trace-based failure clustering.

**Problem.** LLM agents do short tasks well but lose coherence over long horizons. Isolate that by a simple but long business: run a vending machine for months of simulated time (>20M tokens per run).

**Method / algorithm.**
- **Agent scaffold** (inspect-ai): basic tool loop; context = last N = 30,000 tokens; memory tools with read/write/delete on a scratchpad, a key-value store and a vector DB (text-embedding-3-small, cosine), no size limits. Prompt "Continue on your mission by using your tools." when the agent emits no tool call.
- **Remote tools:** read/send email, web search (Perplexity), storage inventory, balance. **Physical tools are behind a sub-agent** (stock machine from storage, collect cash, set prices, machine inventory) reached through `sub_agent_specs`, `run_sub_agent`, `chat_with_sub_agent`; this simulates delegating to a human or robot.
- **Clock:** each tool call advances sim time by 5 min, 25 min, 75 min or 5 h depending on the tool; `wait_for_next_day` skips ahead. Each morning the agent is notified of yesterday's sales and new emails.
- **Supplier simulation:** agent finds real wholesalers via search; only emails to addresses that exist in the real world get replies, generated next day by GPT-4o from Perplexity-fetched facts about that supplier plus the agent's request. A purchase must specify item names, quantities, delivery address and the account number to charge; goods arrive a few days later with an email on arrival. Item sizes (small or large slot) are set by GPT-4o at order time.
- **Customer demand model**, daily per item: (1) GPT-4o generates and caches price elasticity, reference price and base sales; (2) sales impact factor from % difference to reference price × elasticity, multiplying base sales; (3) day-of-week, month and weather multipliers; (4) a choice multiplier rewarding variety but penalising too many options, capped at −50%; (5) add noise, round, clip to [0, available stock].
- **Config:** start $500, daily fee $2, machine 4 rows × 3 slots (2 small rows, 2 large). 2,000 messages per run; stop early if the fee is unpaid 10 days in a row. 5 runs per model or config.
- **Score:** final net worth = cash + uncollected machine cash + unsold inventory at wholesale cost. Also units sold, days until sales stop, tool use.
- **Human baseline:** one person, 5 hours, same chat interface, no prior knowledge.

**Verification / grading.** Purely numeric outcome from the simulator state (net worth). Not hackable through a judge, but inventory valued at cost means buying stock is neutral to net worth, so hoarding inventory is not penalised (they note the top Sonnet run over-ordered).

**Scale & cost.** About 25M tokens and 5–10 real hours per run; 5 runs per model, 10 models, plus config variants.

**Key results.**
- Mean net worth: Claude 3.5 Sonnet $2,217.93 (min $476), o3-mini $906.86, human $844.05 (single run), Gemini 1.5 Pro $594.02, GPT-4o mini $582.33, GPT-4o $335.46, Gemini 2.0 Pro $273.70. Every model has runs that sell 0 units.
- Sales stop on average at 82.2% of the run for Sonnet, 15.8% for Gemini 2.0 Pro; tool use drops after ~120 days for all.
- **Dominant failure:** the agent assumes an order arrived on its expected date (it arrives later that day), the sub-agent's restock fails, and the agent spirals into tangents: escalation to non-existent support, "closing" the business, emailing the FBI, legal threats, or typing tool calls as text for 1,300 messages. Rarely recovers (one Gemini 2.0 Flash run did).
- **Not a context-length effect:** Pearson r = 0.167 between days-until-sales-stop and days-until-memory-full; Sonnet degrades 51 days after memory is full. Larger memory (60K) did worse than 30K and 10K for GPT-4o mini; memory-tool use did not change with memory size.
- **Config ablations (GPT-4o mini):** start balance $100 cuts sales a lot; $5 fee ends all runs before day 100; **$0 fee does not raise sales**: without pressure the agent loops on waiting.
- Sonnet wins by following up with the sub-agent (`chat_with_sub_agent`), tracking units, daily sales and bestsellers, and noticing weekend peaks; it writes daily scratchpad summaries but never reads them back.

**Failure modes & limits.** One domain, 5 runs per arm (very high variance), one human sample; demand parameters generated by GPT-4o, not real data; supplier replies generated (realism depends on Perplexity facts); no partial-credit decomposition.

**Tips & tricks.**
- Charge simulated time per tool type; give a `wait_for_next_day`-style skip.
- Deliver asynchronous outcomes (deliveries, replies) at times that do not align with the agent's wake-up, so status must be checked, not assumed.
- Ground simulated counterparties in retrieved real-world facts, and only answer contacts that exist.
- Use a closed-form, seeded demand model with cached LLM-generated parameters; the LLM sets parameters once, code runs the dynamics.
- A recurring cost (pressure) changes behaviour; zero pressure causes idle loops.
- Run ≥5 seeds of any long-horizon task; report the minimum, not only the mean.
- Record "days until activity stops" as a degradation metric.
- Put physical actions behind a sub-agent to model delegation and its verification.

**Fit to Crucible Part 2.**
- **F1/F2 (World clock and people):** adopt per-tool time costs, a wait-to-next-event tool, and delayed event delivery offset from wake times; adopt the "LLM sets cached parameters, code simulates" split for any market or demand App (keeps execution as ground truth and determinism under seed).
- **F3/F4 knobs:** horizon, pressure (recurring cost), start capital and delivery latency are monotone difficulty knobs; add `pressure` and `latency` to the knob kinds.
- **F5:** net-worth outcomes are cheap state checks, but value inventory at a discount or add a hoarding collateral check.
- **F10/F12:** "assumed status without checking" is a skill leaf ("verify asynchronous outcome before acting") and a failure cluster label; the meltdown signature (repeated escalation, no tool calls) is a cheap trace detector for failure analysis. Long-horizon tasks need a per-run degradation metric, and difficulty measured over several seeds (variance is the signal here).
- Priority P1 (World clock and long-horizon failure taxonomy); P2 as a task source.

## 2402.05733 — TimeArena: Shaping Efficient Multitasking Language Agents in a Time-Aware Simulation (Zhang et al., 2024; ACL 2024)
Read: FULL.  Relevance to Part 2: low–medium. A small hand-built text sim (30 tasks), but it isolates three World mechanics Crucible's clock needs (action durations with dependencies, agent occupancy, object occupancy) and gives a dense progress metric weighted by duration and an algorithmic oracle for minimum time.

**Problem.** Text simulations ignore time; real planning needs durations, parallelism and resource contention. Measure whether LLM agents multitask efficiently.

**Method / algorithm.**
- **World:** 3 scenarios (cooking, household, laboratory) × 10 tasks = 30 tasks; 71 objects, 45 action verbs (each with a state-change description, e.g. chop turns whole into sliced). Per task on average 4–6 actions, 3–6 objects, 13–19 min shortest time. Tasks hand-built by 3 annotators as dependency flowcharts.
- **Mechanics:** (1) action dependencies within a task; (2) each action lasts 1–10 min, durations not revealed until execution starts; (3) **agent occupancy**: Type 1 actions occupy the agent until done (wash cup), Type 2 let it act in parallel (boil water); (4) **object occupancy**: an object used by an action is locked until completion; (5) `wait` skips one minute.
- **Feedback types:** invalid action, non-existent object, wrong input (prerequisite missing or already done), mismatched object, occupied object; valid action returns start message with duration; completion returns the new state and frees the object.
- **Task sets:** #Task = 1, 2, 3 (combinations of tasks from one scenario), 10 combos per setting per scenario. Time limit 40 min per task, scaled with task count. Fail after 5 consecutive incorrect actions. Actions parsed from text by regex.
- **Oracle:** greedy algorithm: sort non-occupying actions by duration descending, place prerequisites first via BFS, then always start the longest non-occupying action as early as possible and avoid idling; computes minimal time T_min (manually checked).
- **Metrics:** progress score where each action contributes s_i = t_i / Σ t_j (duration-weighted); AS = mean max progress; CS = Σ P_i / Σ T_i (score per minute); CR = completion rate (a combination counts as one task); CT = mean time of successful completions.

**Verification / grading.** Fully programmatic state machine; ground truth by construction. Hackable only via `wait` loops, which are valid actions and do not fail the episode (observed).

**Scale & cost.** 30 base tasks, 90 combined evaluations per model, 7 models. No cost given.

**Key results.**
- GPT-4 is the only model that completes combined tasks reliably: cooking CR 90 / 70 / 40% for 1 / 2 / 3 tasks; household 100 / 100 / 90%; lab 90 / 50 / 60%. Most others drop to 0% at 2–3 tasks.
- Even GPT-4 is slow: cooking 3-task CT 76.25 min vs oracle 52.5.
- **Self-plan prompting** (model infers dependencies, estimates durations, applies the greedy rule, then executes) did *worse* than vanilla GPT-4 (cooking 3-task CR 40 → 0%): wrong dependency parsing, cascading duration errors, rigid adherence to a flawed plan.
- Over half of valid actions are `wait`, mostly *unnecessary* waits (other actions were available): no parallelism awareness.
- Restricting cooking to one pot, fryer and oven left GPT-4's time and speed unchanged (it rarely parallelises anyway) but lowered completion and progress.
- Weaker models fall into wait / invalid-action alternation loops, believing wait undoes errors.

**Failure modes & limits.** Tiny task set; actions listed explicitly (no exploration); occupancy is fixed per verb (cooking beef in reality needs attention); text only; greedy oracle is a heuristic, not a proven optimum.

**Tips & tricks.**
- Weight partial progress by the duration (or cost) of each completed subgoal, not by count.
- Model three time mechanics separately: duration, actor occupancy, resource locks.
- Hide true durations until an action starts, so planning must adapt.
- Compute an oracle minimum time algorithmically and score efficiency against it.
- Classify every invalid action (dependency, mismatch, repeat, occupied); the mix is a failure signature.
- Distinguish necessary from unnecessary waits; unnecessary waiting flags missing parallelism.
- Fail after k consecutive invalid actions, and also bound waits, to stop loops.

**Fit to Crucible Part 2.**
- **F1 World clock:** add `duration`, `occupies_actor: bool` and `locks: [resource]` to `ToolSpec`, so Apps can express parallelisable work and contention; resource count is a difficulty knob (contention).
- **F5 timing checks / Φ:** duration-weighted milestones are a clean Φ for partial credit; an efficiency check against an algorithmic oracle time is a timing-check variant ("finished within 1.2 × oracle time").
- **F7 behaviour signature:** record necessary vs unnecessary waits and invalid-action classes; they separate a skill ("exploit parallelism") from luck.
- **F10:** self-plan's failure is evidence that hand-feeding a strategy in the prompt can hurt; a hint-level knob must be measured, not assumed monotone.
- Priority P2.

## 2501.12012 — TabularARGN: A Flexible and Efficient Auto-Regressive Framework for Generating High-Fidelity Synthetic Data (Tiwald et al., 2025; MOSTLY AI, arXiv)
Read: FULL (body + Appendices A–G).  Relevance to Part 2: medium for F2 (seeding a World from customer data), low elsewhere. It is the best-documented open tool in this batch for turning a real flat table plus a child event table (customers → transactions) into a privacy-safe synthetic seed, with conditional generation that lets a seeder plant specific values.

**Problem.** Tabular synthesis must handle mixed types, multi-sequence child tables, conditioning and privacy, at practical speed. Existing GAN, diffusion and LLM-as-text methods are slow or brittle on real data.

**Method / algorithm.**
- **Encoding (privacy-aware domain first):** every column becomes one or more categorical sub-columns. Categorical: rare values merged into one category; missing is its own category. Numeric: auto-choice of discrete (each value a category), binned (≤100 percentile bins, value sampled uniformly within the bin) or digit (one sub-column per digit plus sign and missing flags); value ranges clipped against outliers. Datetime: split into year, month, day, hour ...; for sequences optionally relative to the first event. Strings: per-character sub-columns. Geo: lat/long → quadtile character sequence with density-adaptive resolution.
- **Flat model:** shallow any-order autoregressive network. Per sub-column: embedding (size 3·d^0.25), regressor (feed-forward ReLU, width 16·max(1, ln d), dropout), predictor (softmax over that sub-column's categories). A **permutation masking layer** zeroes the embeddings of columns after i in a random order that changes each batch, so the model learns p(x_i | any subset). Loss: summed categorical cross-entropy = E_σ[Σ_i log p(x_σ(i) | x_σ(<i))], teacher forcing.
- **Sequential model:** autoregressive over columns *and* time: an LSTM history encoder over causally shifted past steps feeds every column's regressor. Two extra columns, sequence length (constant per subject, sampled at the first step) and a counting index, handle variable and irregular lengths, including zero-length. A random max-sequence window (default 100) per batch for long sequences.
- **Two-table setup:** a context processor embeds the parent flat row; flat and sequential models train in parallel; at generation the parent is sampled first, then its sequence (autoregression extended to the table dimension).
- **Training:** 10% validation split; halve LR after 3 epochs without improvement; stop after 5; keep the best weights. Optional DP-SGD (Opacus; noise multiplier 1.5, clip 1, δ = 1e-5).
- **Sampling features:** conditional generation on any column subset (seed specific values), imputation, fairness-aware generation, temperature to trade rule adherence for diversity.

**Verification / grading.** Fidelity: univariate and bivariate accuracy = ½(1 − L1) between decile- or top-10-binned histograms / contingency tables of train vs synthetic; sequential **coherence** = same on (step, next step) pairs sampled per subject. Privacy: DCR share = fraction of synthetic records (MiniLM embeddings of the row string) whose nearest neighbour is in train rather than an equal-size holdout; 0.5 is ideal.

**Scale & cost.** One A10G. Adult (24K rows) trains in about 2.5 min on CPU; ACS-Income 738K rows: generation 15 s, while several baselines OOM or exceed 16 h. Sequential generation: Baseball 49 s vs REaLTabFormer 1,058 s; California (845K child rows) 520 s vs 23,950 s.

**Key results.** Flat accuracy: Adult 0.98 (TabSyn 0.98, CTGAN 0.77), ACS-Income 0.98 (TabSyn 0.95), Default 0.92 (TabSyn 0.95), Shoppers 0.95. Sequential accuracy: Baseball 0.88 (next best ClavaDDPM 0.79), Berka 0.79 (REaLTabFormer 0.74), California 0.99 (0.89); coherence 0.91 / 0.82 / 0.98. DP versions: Adult 0.94 at ε≈2.4; Berka collapses to 0.45 at ε≈1.9 (few subjects, long sequences). All DCR shares ≈ 0.50. Multi-task shared training beat per-column independent models in validation loss.

**Failure modes & limits.** Only one parent→child level (no deep FK graphs); low-order metrics only (no constraint or business-rule validity); DCR is a weak privacy metric (the follow-up paper shows it); DP hurts small or long-sequence data badly; vendor-authored benchmark.

**Tips & tricks.**
- Fix the value domain (rare-category merge, outlier clipping) before training; it is both a privacy and a robustness step.
- Shuffle column order per batch to get any-subset conditioning for free; use it to plant task-relevant values in a seed.
- Model sequence length explicitly at step one plus a counting index.
- Generate parent first, then child sequences conditioned on the parent embedding.
- Score sequences with a step-pair coherence metric, not only marginals.

**Fit to Crucible Part 2.** F2 State seeding "from customer data" (§2(b) enterprise ingest): a candidate generator for entity tables and per-entity event histories (customer → orders, account → transactions) with a fidelity gate built from its univariate / bivariate / coherence accuracies plus a privacy gate (DCR share near 0.5 is necessary but not sufficient; see 2508.06647). Conditional sampling is the hook for F3/F4: plant the anomalous row a task needs while keeping the rest realistic. Limit: one level of depth, so multi-table Worlds need chaining or a relational method. It does not check schema constraints, so G1 integrity must run after it. Priority P2 (enterprise-data path only).

## 2508.06647 — Privacy-Preserving Tabular Synthetic Data Generation Using TabularARGN (Sidorenko & Tiwald, 2025; MOSTLY AI, arXiv)
Read: FULL (body; appendices skimmed for attack definitions and tables).  Relevance to Part 2: low–medium. It is the privacy-audit half of the method above: how to test that synthetic seed data built from customer records does not leak members, and evidence that DCR alone misleads.

**Problem.** Synthetic tabular data is used for privacy, but benchmarks report DCR means that hide overfitting. Measure TabularARGN's privacy rigorously with membership-inference attacks and compare its protections.

**Method / algorithm.**
- Same flat any-order model as 2501.12012 (embedding → permutation mask → ReLU regressor → softmax predictor). Generation: start from zeros, sample column by column in a chosen order.
- **Protections:** early stopping (validation loss, patience, LR halving); **25% dropout** on regressors; **value protection**: rare categories (threshold drawn uniformly from 5 to 8 per column) replaced by `_RARE_` or by a common value; numeric and datetime values beyond the k-th highest or lowest (k uniform in 5–8) clipped; optional DP-SGD.
- **Benchmark:** plugged into the CDTD pipeline: 11 datasets, competitors SMOTE, ARF, CTGAN, TVAE, TabDDPM, CoDi, TabSyn, CDTD (~3M parameters each; ARGN usually 5–10× fewer). Metrics: JSD (categorical), Wasserstein (continuous), L2 between correlation matrices, CatBoost detection score, train-synthetic-test-real ML efficiency (RMSE, F1, AUC), DCR.
- **DCR correction:** compute ∫₀^{q} [CDF(DCR_train/syn) − CDF(DCR_train/test)] up to the point where the train/test CDF reaches 0.98; a positive integral means synthetic rows sit closer to training rows than real holdout rows do (overfitting or leakage); such method–dataset pairs get the worst rank.
- **Membership inference:** Adult, the 1,000-record sample used in prior work; the two most vulnerable records chosen by the Achilles score (mean cosine distance to 5 nearest neighbours). Black-box threat model (attacker knows the generator, output size and the full training set except membership of the target). 1,000 shadow models, half with the target, feed a meta-classifier. Attacks via an extended TAPAS: Groundhog Naive / Histogram / Correlation / Logistic, query-based counting attacks, Hamming and L2 distance, direct lookup, KDE inference-on-synthetic. Report AUC and accuracy.

**Verification / grading.** Attack AUC (0.5 = chance) is the privacy score; utility by the benchmark ranks.

**Scale & cost.** 1,000 shadow models per configuration; no compute reported.

**Key results.**
- Raw ranks: ARGN first on Wasserstein (2.4) and detection (1.8), second on L2 and AUC; CDTD best on several. After the DCR-integral correction, CDTD, TabDDPM and SMOTE are flagged as overfitting and ARGN ranks first on 5 of 7 metrics.
- On the lending dataset all methods overfit because `emp_title` has ~5K unique values in 10K rows (98% appear fewer than 8 times); merging categories seen fewer than 8 times into a rare token fixed the DCR CDF.
- MIA without value protection or DP: AUC 0.48–0.64; best attack Correlation Groundhog AUC 0.635, accuracy 60.1%; distance and lookup attacks only 0.53–0.56 (leakage is distributional, not verbatim copies). With value protection: Correlation Groundhog 0.635 → ~0.501. With DP ε = 1.0: all attacks ~0.50–0.52.
- Authors contrast with >0.7 AUC typically reported for GAN-style generators.

**Failure modes & limits.** Only two target records on one dataset; utility cost of value protection and DP not measured; the DCR integral bound 0.98 is a choice; leakage via rare categories needs data-specific handling.

**Tips & tricks.**
- Do not trust a DCR mean; compare the CDFs of train→synthetic and train→holdout distances and flag a positive integral.
- Merge categories with fewer than ~8 occurrences before synthesis; high-cardinality free-text-like columns are the main leak.
- Audit the most vulnerable records (outliers by k-NN distance), not the average record.
- Use shadow-model MIAs with several attack families; simple correlation attacks can be the strongest.
- Early stopping and dropout already cut leakage; DP ε ≈ 1 removes measured MIA advantage.

**Fit to Crucible Part 2.** §2(b) redaction and F2 "fidelity gate when from customer data": the privacy gate for synthesised seeds should be (1) DCR-CDF integral ≤ 0, (2) rare-value merge at a threshold of about 8, (3) a shadow-model MIA on the k most vulnerable records with AUC near 0.5, run before any model in the swarm sees the seed. Traces-to-tasks (§3.4) needs the same gate when rebuilt start states are drawn from logs. Free-text fields (emails, tickets) are out of its scope; those need the DP text route (2210.14348). Priority P2 (needed only when a customer-data source exists).

## 2201.12677 — AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data (McKenna et al., 2022; PVLDB 15(11))
Read: FULL (body through §7; appendices referenced only via the main text).  Relevance to Part 2: low. A DP synthetic-table algorithm for marginal workloads; relevant only to the enterprise path where seed tables are built from customer data and a formal privacy guarantee is required. Its "workload-aware" idea (spend budget on the statistics the downstream tasks will query) is the transferable part.

**Problem.** Workload-aware DP synthetic data mechanisms often lose to workload-agnostic ones even on their own workloads, and none tells the user how wrong each answer may be.

**Method / algorithm.** Select–measure–generate, with Private-PGM (a graphical model fitted to noisy marginals by max likelihood) for the generate step.
- **Inputs:** discrete dataset D, workload W of weighted marginal queries (attribute subsets r with weights c_r), zCDP budget ρ. Defaults: MAX-SIZE 80 MB junction tree, T = 16d (upper bound on rounds), α = 0.9 (90% of each round to measuring).
- **Initialise:** measure all 1-way marginals in the downward closure W₊ with Gaussian noise σ₀ = √(T/(2αρ)); fit an independence model.
- **Weights:** w_r = Σ_{s∈W} c_s · |r ∩ s| (overlap with the workload).
- **Loop while ρ_used < ρ:** charge ρ_t = ε_t²/8 + 1/(2σ_t²); candidates C_t = marginals in W₊ whose addition keeps JT-SIZE ≤ (ρ_used/ρ)·MAX-SIZE; select r_t by the exponential mechanism with quality q_r = w_r(‖M_r(D) − M_r(p̂_{t−1})‖₁ − √(2/π)·σ_t·n_r) (current error minus expected error after measuring at the current noise; sensitivity max w_r); measure y_t = M_{r_t}(D) + N(0, σ_t²I); refit p̂_t by Private-PGM (weighted by 1/σ_i).
- **Annealing:** if the refit changed marginal r_t by less than its expected noise √(2/π)σ_t n_{r_t}, double ε and halve σ (more budget per round); if the remaining budget is less than two rounds, spend it all in a final round. This removes the rounds hyper-parameter.
- **Generate:** randomised rounding from p̂ to records.
- **Uncertainty quantification (no extra budget):** for "supported" marginals (subset of a measured one), an unbiased weighted-average estimator with known variance gives an L1 confidence bound √(2 log 2)σ̄_r n_r + λσ̄_r√(2n_r); for unsupported ones, a bound from the exponential mechanism's selection guarantee. Use the last round only (picking the tightest round invalidates the bound).
- **Basic good practice it codifies:** measure whole marginals, Gaussian not Laplace noise, unbounded DP (√2 less noise), 10/90 select/measure split.

**Verification / grading.** Workload error = (1/(k|D|)) Σ c_i ‖M_{r_i}(D) − M_{r_i}(D̂)‖₁; 5 trials; ε ∈ [0.01, 100], δ = 1e-9.

**Scale & cost.** Single CPU core, 4 GB, ≤24 h per run (most far less). Datasets: adult (48,842 × 15), salary, msnbc (~990K × 16), fire, nltcs, titanic; workloads all-3way, target (3-way with a target attribute), skewed (256 weighted triples).

**Key results.** All-3way: AIM beats PrivMRF 1.3× on average (3.6× max), MST 2.6× (118×), MWEM+PGM 1.5× (16×), RAP 5.6×, GEM 2.0×; best in 76% of settings and 100% at ε ≥ 1. Skewed: 2× over PrivMRF on average (5.7× max). Ablations (geometric-mean error ratio): Gaussian noise 1.18, unbounded DP 1.13, 10/90 split 1.08, initialisation 1.03, new selection 1.37, adaptive rounds 1.48, Private-PGM vs relaxed projection 2.36. Error bounds always held at 95% on fire; median looseness 4.4× (supported) and 8.3× (unsupported).

**Failure modes & limits.** Marginal workloads only; discrete domains only (discretisation quality matters); high-cardinality attributes and sparse marginals unhandled; small workloads may favour plain Gaussian+PGM; single flat table (no relational structure).

**Tips & tricks.**
- State the downstream workload (which joint statistics tasks will query) and spend privacy budget there.
- Score candidate statistics by expected error reduction at the current noise, not raw error.
- Cap model size explicitly so runtime is predictable.
- Report per-query error bounds with the data so users know which questions it answers well.

**Fit to Crucible Part 2.** F2 enterprise seeding with a formal guarantee: the workload is derivable from the `DomainSpec` (the joint statistics the verifiers and tasks depend on, e.g. invoice amount × vendor × status), which makes AIM's workload-awareness a natural fit, and its per-marginal error bounds can feed a seed-fidelity card. But it is single-table and discrete, so it would sit under a relational wrapper, and most synthetic Worlds need no DP at all. Priority P2, only if a customer insists on formal DP.

## 2210.14348 — Synthetic Text Generation with Differential Privacy: A Simple and Practical Recipe (Yue et al., 2023; ACL 2023)
Read: FULL.  Relevance to Part 2: low–medium. The recipe for turning private customer text (emails, tickets, chats, feedback) into a DP synthetic corpus that can seed World inboxes and traces-to-tasks without exposing real people; its canary-injection test is a reusable leak audit.

**Problem.** Synthetic text with a formal privacy guarantee usually has poor quality. Show that DP fine-tuning a pretrained LM, then sampling, gives useful private synthetic text.

**Method / algorithm.**
1. Prepend each record with a **control code** built from its attributes, e.g. "Business Type: Bar | Review Stars: 5.0" (or "A1: v | A2: v | A3: v" for customer feedback).
2. Fine-tune a pretrained causal LM (GPT-2 small, medium, large) with **DP-SGD** (per-sample clipping, norm 1.0; Gaussian noise; Opacus via `dp-transformers`). ε = 4, δ = 1/(N log N); noise from a numerical composition accountant. DP hyper-parameters: batch 4,096, LR 1e-4, 20–50 epochs (vs 5 epochs, batch 32 non-DP). Max length 128.
3. Sample with top-k = 50 and top-p = 0.9, conditioned on control codes drawn in the original category proportions (the authors ignore the small privacy cost of releasing those proportions), 100K samples.
4. DP's post-processing property: any number of downstream uses cost no extra privacy.

**Verification / grading.**
- Utility: RoBERTa-base trained on synthetic, tested on real (rating and category classification).
- Distribution: embedding precision/recall F1, FID and MAUVE averaged over five sentence-transformers; BERTopic top-10 topic distribution; length distribution.
- **Privacy audit by canaries:** inject 5 crafted records with PII (name, address, phone, email, licence plate) repeated 1, 10 or 100 times; check verbatim appearance in generations and the perplexity rank of the secret among 10,000 same-type candidates.
- **Subject-of-interest test:** inject 100 / 1,000 / 10,000 paraphrased reviews about one topic (GPT-3-generated) and count appearances.

**Scale & cost.** Yelp: 1.9M training reviews, 10 categories; 16 V100s. Customer feedback: 1M Microsoft records, 2 A100s.

**Key results.**
- Rating / category accuracy: original 0.7334 / 0.7752; GPT2-Large synthetic ε=∞ 0.7090 / 0.7576; ε=4 0.6936 / 0.7568. Larger generators shrink the DP gap.
- Synthetic-with-DP 100K beats training the classifier directly with DP on 100K real (0.6689 / 0.7552) and nearly matches DP on 1.9M real (0.7014 / 0.7644), while serving unlimited tasks at one ε.
- Embedding F1 GPT2-Large 0.585 (∞) vs 0.514 (ε=4); MAUVE 0.724 vs 0.609.
- **Canaries:** at 100 repetitions, 80% leak verbatim without DP, 0% with DP; a canary seen *once* already ranks near the top for some types (address, plate) without DP.
- **Tail loss:** a subject in 100 records (0.005%) vanishes under DP; 1,000 records give 10 generations; 10,000 are over-generated without DP (3.0% vs 0.53%) and less so with DP (1.7%).
- **Length truncation:** DP generations are shorter (GPT2-Large 84.5 vs 89.4 tokens).
- Customer feedback: attribute accuracy drops most on the 45-class attribute (0.716 → 0.536 at ε=4): small classes suffer.

**Failure modes & limits.** DP erases rare patterns and minority classes, exactly what edge-case tasks need; generated text is shorter and blander; guarantees assume uncorrelated records (one customer writing many tickets breaks this, so the example boundary must be the person); GPT-2-era models.

**Tips & tricks.**
- Condition the private generator on control codes so category proportions and labels survive.
- Audit leaks by injecting canaries at several repetition rates and checking verbatim reappearance plus perplexity rank against same-type decoys; PII scrubbers are too noisy for this.
- Expect DP to remove anything rarer than ~0.05% of records; plant rare cases synthetically afterwards instead.
- Use per-person (not per-record) privacy units when one person produces many records.
- Combine DP with deduplication and redaction.

**Fit to Crucible Part 2.** §2(b) enterprise ingest and §3.4 traces-to-tasks: (1) a DP text generator conditioned on control codes from the `DomainSpec` (ticket type, product, sentiment) can fill World inboxes and chat histories with realistic, non-identifying content; (2) the canary test is a cheap, concrete leak gate (G11-style) for any pipeline that ingests customer text: inject known fake PII into the source and require 0 verbatim hits and no top-rank perplexity in outputs. Avoid using DP text for rare-edge-case tasks; the factory should synthesise those cells deliberately (the coverage grid), since DP will not reproduce them. Priority P2 (customer-data path only), with the canary audit P1 for any redaction pipeline.

## 2609.04148 — Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments (Wu et al., 2026; Qwen Team / Tsinghua, arXiv)
Read: FULL (body + Appendices A–D, F; case studies E skimmed).  Relevance to Part 2: **high**. It is the clearest working precedent for Crucible's traces-to-tasks source: rebuild the start state from a trace, keep the agent's edits out, complete the missing context with an agent under a "solvable but not solved" contract, gate on sufficiency, then re-query the rebuilt environment for many new verified tasks. It also measures that more environments beat more queries per environment, and that re-solving beats imitating the source trace.

**Problem.** Agent trajectories are plentiful; executable environments are scarce. A trajectory is one frozen demonstration, while an environment can be re-solved by a stronger model, verified, and re-queried for new tasks. Invert the usual mapping: recover environments from trajectories.

**Method / algorithm.**
- **Seeds:** 359,593 public trajectories (SWE-rebench, SWE-smith, CoderForge, SWE-Gym; LFM2-Terminal, LiteCoder-Terminal). Keep a trace if the end workspace has ≥5 files and ≥100 lines. Exclude Terminal-Bench-derived corpora, view-only traces and unsupported action formats; SWE: dedup by repo + base commit + problem statement (keep the trace exposing most content); drop SWE-bench Verified repos.
- **Stage 1, deterministic replay.** Normalise read/write/edit ops into a chronological event stream. For each touched path, take the earliest observed content (before the agent's first change); exclude agent-created files; record the agent's changes separately. Result Ê₀ is partial (only touched paths; truncated reads).
- **Stage 2, agentic completion.** A completion agent gets the recovered task, Ê₀ and the file inventory, and must make the task *solvable but not solved*: create missing files, complete partial ones, configure dependencies; no implementation of the target, no tests for the target, no comments locating the solution; prefer source and data over docs; network only for restoring dependencies. Workspace goes under /app in a standard ubuntu:24.04 container with network. Manual check of 30 completions: 22 only task-relevant support, 8 added unnecessary code, 0 contained a solution.
- **Stage 3, sufficiency filter.** A read-only agentic judge labels the workspace sufficient / insufficient for the recovered task, returning JSON {label, confidence, reason, present, missing_critical}. Sufficient = enough project-specific source, config, data and structure for a grounded attempt; caches and installable packages may be absent.
- **Re-querying (four variants):**
  1. *Intent Recovery*: from the first substantive user request (later ones only if they clarify or constrain the same task), write a 2–4 sentence self-contained `core_objective` plus success criteria, constraints and prohibitions; never inject agent-only evidence (paths, values) the user did not state.
  2. *Single-WS*: a read-only explorer proposes 5 tasks (internally 10 candidates → filter → 5 with different primary focus) in the zone "clear goal + verifiable standard + unexplored implementation"; reject undeterminable, docs-only, trivial, flaky, external-service, repository-independent or solution-revealing tasks. One valid candidate per environment is sampled.
  3. *Cross-WS (breadth)*: profile each workspace (domain, subdomain, language, frameworks, 3–8 capabilities, key entities, summary); TF-IDF nearest neighbours within a language; an LLM labels pairs similar / complementary / dependency / unrelated and the donor side; keep dependency pairs, each workspace in at most one pair. The generator mounts both, chooses target (writable) and reference (read-only), and writes a task that needs **≥3 concrete facts found only in the reference** (field names, formats, parameters) that the query must not state, naming only the reference's mount root; it returns the 3 hidden facts. Reject tasks that leak the facts or name reference internals; the verifier checks the reference is unchanged.
  4. *Multi-Round (depth)*: a user agent keeps a private requirement ledger and reads the private verifier result; after a fail it describes only the user-visible symptom (never tests or tracebacks) and asks for a fix without changing requirements; after a pass it adds a compatible extension, or, if policy allows, a conflicting replacement that quotes what it supersedes. Before each coding round a verifier writes round-level acceptance tests; new and active (regression) requirements are tested after each response. Up to 6 follow-ups.
- **Verifier construction (agentic, in-container, before any solution):** one self-contained pytest file, ≤18 tests, 1–3 per requirement, public surfaces only, absolute paths, no network, no randomness or wall-clock, no skip/xfail, never write into /app. Expected values for new behaviour must be computed independently (from shipped data or the query), never by running the current incomplete code; preservation tests use current behaviour as baseline and must not reuse the repo's own suite. **Run it on the unmodified workspace: at least one target test must fail, preservation tests must pass, zero collection or infra errors**, else iterate or reject (query_unverifiable / workspace_broken).
- **Rollout and selection:** teacher Qwen3.7-Max (xhigh) in Claude Code, temperature 1.0, top-p 0.95, 256K context, 500 turns, 4 h. Single/Cross-WS kept only if all tests pass on the final workspace. Multi-Round: trim trailing failed rounds, keep sessions with ≥2 verified passing rounds, keep intermediate fail→repair rounds. 13-gram decontamination against Terminal-Bench.
- **Training:** SFT of Qwen3.5-27B, 2 epochs, LR 7e-6, batch 256, 256K sequence length.

**Verification / grading.** Hidden agent-written pytest, gated by "red on start, green preservation". Weak points (authors): one teacher writes tasks, solutions and tests, so shared blind spots pass; tests target public interfaces only. Intent Recovery tasks have no verifier.

**Scale & cost.** 359,593 traces → 68,263 reconstructed environments → after decontamination and repo dedup, sufficiency on 38,294 terminal (40.2% after replay → 93.5% after completion) and 1,900 SWE (20.1% → 77.1%) → **37,273 sufficient environments**. Completion grows terminal workspaces from median 2 to 13 files (mean 2.9 → 22.4) and text lines from 60 to 539. SFT corpus: 25,386 Single-WS + 3,512 Cross-WS + 3,079 Multi-Round = 31,977 records, ~1.42B tokens (Intent Recovery 35,809 used for ablations). Per record medians: Single-WS 14 turns / 20 tool calls / 30.4K tokens; Cross-WS 23 / 38 / 46.5K; Multi-Round 92 / 102 / 126.1K. Teacher pass@1: Single-WS 72.3%, Cross-WS 49.2%. Yield Single-WS 25.4K of 35.1K launched (72%), Cross-WS 3.5K of 7.1K (49%). No $ or GPU-hours given. Python is 84.7% of terminal workspaces.

**Key results.**
- Qwen3.5-27B: Terminal-Bench 2.0 41.6 → 52.8, TB 2.1 46.2 → 58.1 (Terminus2-XML; 58.2 with Claude Code); EvoCode-Bench v2 MT@4 6.3 → 20.1, case score 67.8 → 76.1. Beats other synthesis corpora on the same or larger bases (RST-27B 49.4, CalibForge-35B-A3B 47.6, TMax-27B 42.7 on TB 2.0).
- **Re-solve vs imitate (same 35.8K tasks):** source-trajectory SFT 36.7 avg (below the 47.0 base) vs Intent Recovery re-solved by the teacher 52.1.
- **Completion matters:** replay-only 48.7 vs replay + completion 52.9 (±3.5 vs ±1.4); on incomplete workspaces the teacher spends effort repairing setup instead of the task.
- **Verifier filter matters more for hard tasks:** Single-WS 56.0 (all) vs 56.4 (verified, 72% of data); Cross-WS 53.2 vs 55.4 (verified, 49%).
- **Breadth:** Single-WS 56.4, + Cross-WS 58.4; per category, model training +15.0 and debugging +10.0.
- **Depth:** + Multi-Round raises MT@4 18.4 → 21.0 and case 71.9 → 76.9; removing round-level verifier feedback drops them to 18.8 / 73.2.
- **Budget axis (fixed ~35K records):** 2× environments 56.0 vs 2× queries per env 53.8 vs 2× solutions per query 53.9 (base 53.2).
- SWE-derived Intent Recovery transfers to terminal: 47.0 → 50.0.

**Failure modes & limits.** Generic container lowers fidelity for special system deps; coverage bounded by the source trace distribution; single-teacher circularity; sufficiency judge is an LLM; completion invents plausible but unreal context (8/30 added unneeded code), which lowers realism; network-enabled containers can leak; SFT only, no RL; no per-stage cost.

**Tips & tricks.**
- Rebuild the start state from the earliest observed version of each touched object and withhold everything the agent created or changed.
- Contract for the completion agent: "solvable, not solved"; ban tests and location hints for the target.
- Gate rebuilt environments with a read-only sufficiency judge that must list present and missing-critical evidence.
- Re-solve recovered intents with a strong policy instead of training on the raw trace.
- In intent recovery, keep only what the user said; never promote agent-observed details into the instruction.
- Verifier must be red on the start state for the target and green for preservation, written before any solution, with expectations computed independently of the code under test.
- Cross-environment tasks: require ≥3 hidden facts discoverable only in a read-only reference, and reject queries that state them.
- Multi-turn users: private requirement ledger, symptom-only feedback on failure, compatible extension on pass, explicit supersession for conflicts.
- Spend budget on more distinct environments, not more queries or solutions per environment.
- Keep fail→repair rounds in multi-turn data; trim only trailing failures.

**Fit to Crucible Part 2.**
- **Traces-to-tasks (§3.4) and F2/F3/F4:** this is the reference implementation of "trace → World snapshot → task". Map: replay = World rebuilder writing a `SeedBundle` from the earliest observed state of every entity the trace read or wrote (for state Worlds, the pre-write row versions); completion = state seeder filling un-observed but needed entities under a "solvable not solved" contract; sufficiency judge = a new F2 gate (`G-sufficiency`: the rebuilt start state supports the recovered intent, with `missing_critical` evidence). Intent Recovery = the intent labeller, with the rule "user-stated requirements only".
- **F5/F6:** adopt their verifier gate as a generic rule for `VerifierBundle`: written before oracle rollout, *fails on the start state for the target checks* (a cheap G4 not-pre-solved check), preservation checks pass on the start state (collateral baseline), expectations never computed from the code or state under test. Their gap: one model writes tasks, oracle and tests; Crucible's family separation (§3.0 rule 5) fixes exactly the limitation they list.
- **Diversity (§3.5) and F11 quotas:** their budget ablation is direct evidence for quota per environment/World (one task per environment > two per environment) and for rewarding World-level novelty over instruction novelty.
- **Difficulty (§3.6):** Cross-WS is a structural hardness knob (teacher pass 72% → 49%) that keeps solvability by construction; Crucible's analogue is cross-App or cross-World tasks with hidden facts in a read-only App.
- **Multi-turn task type (§3.3):** their user-agent protocol is a concrete `ActorSpec` behaviour: private contract ledger, private verifier result, symptom-only complaints, extension/revision/conflict styles; round-level verification and fail-stop scoring.
- **Interface implied:** `rebuild(trace) -> SeedBundle + withheld_changes`; `complete(SeedBundle, intent) -> SeedBundle` under a no-solution contract; `sufficiency(SeedBundle, intent) -> {label, present, missing_critical}`; `recover_intent(trace) -> {objective, criteria, constraints, prohibitions}`.
- Priority **P0** for the traces-to-tasks design and the red-on-start verifier rule; P1 for cross-World and multi-round re-querying.

## 2610.06100 — From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation (Long et al., 2026; NTU / PolyU, arXiv)
Read: FULL (body + Appendices A, B.3–B.5, C; case study and prompts skimmed).  Relevance to Part 2: **high** for the traces-to-tasks source when the original system is gone, and as a cautionary result for any LLM-simulated World: high in-simulator success can be fiction (Direct Prompting ALFWorld: 97% in the sim, 3% when replayed in the real env). Its separation of reusable environment knowledge from per-episode state, with a validation gate on every state effect, is directly adoptable.

**Problem.** Realistic replicas of terminals, apps or enterprise systems may be impossible to rebuild (missing code, data, infra), but historical interaction traces remain. Build a reusable, stateful simulator from traces alone, without reconstructing the executable system and without training.

**Method / algorithm.** Two phases; learning-free; worldbook built once by GPT-5.6-sol and frozen.
- **Offline: worldbook K_E = Construct(D_build)**, four components:
  - *Schemas*: action schema (types, argument names and types; incoming actions canonicalised against it) and state schema with typed, mutability-tagged variables in four groups: `world` (durable facts), `session` (interaction context), `surface` (currently visible), `epistemic` (uncertain or conflicting beliefs).
  - *Grounded evidence*: one record per aligned transition (action, outcome, extracted facts, candidate effects with uncertainty, verbatim observation); up to 3 demonstrations per action type, chosen deterministically.
  - *Induced abstractions*: transition rules (conditions over state and args → ordered effects, outcome, optional observation template, confidence, status), state constraints, observation contracts (required fields, examples, template), conventions.
  - *Provenance*: every abstraction links to supporting episodes and turns.
  - Construction steps: align actions to results (call ids, else adjacency); the model extracts facts and effects while code keeps the raw observation; withhold effects with ambiguous attribution or concurrent actions (confidence capped at 0.5); induce schemas, with deterministic checks adding missed action names, args and state vars; propose rules and constraints, review against supporting evidence and **success–failure contrasts** (a contrast suggests a missing precondition; a counterexample narrows a rule); label supported / tentative / conflicted; only supported rules with confidence ≥ 0.55 compile to executable.
- **Online: episode workspace W_t = (K_E, ŝ_t, M_t)**: immutable worldbook (how the env behaves), mutable partial episode state (what is true now), episodic memory (what happened). A missing state entry means *unknown*, not absent; worldbook examples never seed episode facts.
- **Per action (Algorithm 1):** canonicalise; retrieve (action-type lookup + lexical) up to 5 evidence records and 2 demonstrations; a deterministic **applicability gate** labels each as supporting / uncertain / format-only (shared operands or names already in the episode for Terminal; page or screen identity for Web and Android); foreign values in uncertain and format-only entries are masked; current state and memory outrank retrieved examples. The world-model agent inspects (≤8 read-only requests, ≤20 model calls), previews effects on a state copy, and proposes q_t = (Δ_t, õ_{t+1}): declarative effects on declared mutable variables plus an observation. A **validation gate** checks paths, mutability, types, rule support and state constraints on a copy; effects attributed to a rule must equal its resolved effects; schema-valid model-proposed effects that add no constraint violations are accepted. On rejection, a fallback commits the observation only (no state change), or a strict policy records a failure. Render via rule or contract template if present. Commit state and append to memory.

**Verification / grading.** Of the simulator itself: (a) next-observation fidelity on AgentWorldBench (Terminal, SWE, Android, Web) and three EnvScaler envs (Food, Shopping, Benefits), scored 1–5 by a GPT-5.2 judge on format, factuality, consistency, realism, quality, rescaled to 0–100; trajectory-clustered paired bootstrap (10,000 resamples) for CIs; (b) multi-turn: a task agent solves in the real env (Real), in the world model (WM), and the WM-induced action sequence is **replayed in the real env (W2R)**; consistency ratio CR = W2R / Real. Leakage control: cross-fitting (task-grouped 5-fold) when only benchmark traces exist; results stratified by construction-overlap.

**Scale & cost.** Construction traces: 20 Terminal-Bench trajectories (334 turns), 50 WebArena, 3–16 EnvScaler, 40 ALFWorld (836 turns), 30 SciWorld (1,119 turns). Worldbook sizes: e.g. Terminal 15 actions, 29 state fields, 3 of 12 rule candidates executable, 18 notes; Web 227 state fields, 18/28 rules. Inference per record (GPT-5.6-sol / DeepSeek-V4.1-Flash): Terminal $0.21, 41 s / $0.03, 48 s vs direct prompting ~$0.04; 1.3–5.1 tool calls per record. Without the adaptive loop: ~80% of the gain at 29% of the cost.

**Key results.**
- Average next-observation score: Trace2Env 75.94 (GPT) / 75.75 (DeepSeek) vs Direct Prompting 69.51 / 68.71, Trace RAG 73.76 / 73.06, Worldbook-as-prompt 74.35 / 74.86, Harness only 70.79 / 70.35. Best in 11 of 14 env–backbone pairs, never below direct prompting; factuality and consistency improve everywhere. A worldbook built by one model works when operated by another.
- **Multi-turn transfer:** ALFWorld Real 93%, Direct Prompting WM 97% but W2R 3% (CR 0.032) vs Trace2Env WM 88%, W2R 85% (CR 0.914). SciWorld CR 0.529 → 0.706.
- **Ablations (Terminal):** schema only +1.91 over harness only; + abstraction alone −1.12; + evidence +2.45; abstraction given evidence +0.69; evidence given abstraction +4.26; removing the adaptive loop −1.79; removing state tracking −0.97.
- **Unseen tasks:** gains persist on construction-disjoint subsets (Web +4.21, SWE +5.09 with CIs above 0; Terminal +3.69 with GPT, not significant with DeepSeek); same-task overlap gives much larger gains (+17.86 Terminal), i.e. coverage still matters.
- **Trace scaling:** Terminal 64.89 at 20 traces → 66.42 at 36, still rising; construction cost grows with trace volume.

**Failure modes & limits.** Fidelity bounded by what traces reveal; few rules compile (3 of 12 on Terminal), so most behaviour is model-proposed; LLM judge scores fidelity; no training of agents on it yet; the fallback policy can silently accept a wrong observation with no state change; absolute fidelity is moderate (~60–70 on Terminal/Web).

**Tips & tricks.**
- Separate immutable environment knowledge, mutable episode state, and episode memory; never let examples seed facts.
- Treat a missing state entry as unknown, not absent.
- Tag retrieved evidence as supporting, uncertain or format-only, and mask foreign values in the latter two.
- Every LLM-proposed state change is a typed declarative effect validated on a state copy before commit.
- Induce rules from success–failure contrasts; compile only supported rules above a confidence floor; keep provenance links.
- Withhold effects with ambiguous attribution.
- Measure a simulator by replaying sim-induced actions in the real system (W2R/Real), not by in-sim success.
- Cross-fit worldbook construction to avoid evaluating on its own traces.

**Fit to Crucible Part 2.**
- **Traces-to-tasks when the customer system is unavailable (§3.4):** use a worldbook-style artifact as the bridge from redacted logs to a World: its schemas map to `AppSpec` (actions = tools, state schema = tables), supported rules map to `policies_as_code`, evidence turns are tool-test fixtures. Better still, use it as a *specification source* for F1: compile the schemas and supported rules into a coded App with blind tool tests, then treat the remaining LLM-simulated behaviour as a lower App tier. This keeps "execution decides" for reward-bearing state.
- **F1 gate for any LLM-simulated App or person:** require a W2R-style replay consistency check wherever a real reference exists (CR ≥ ~0.9 as a starting bar), because in-sim success overstates fidelity by up to 30×.
- **World loop (§3.8):** the validation gate (typed effects on declared mutable state, constraint checks, observation-only fallback) is a concrete guard for LLM-backed tools in the World; log every fallback as a `world_bug` candidate signal.
- **People and epistemic state:** the `epistemic` state group is a clean place for a simulated user's uncertain beliefs.
- Priority **P1** (traces-to-tasks design and LLM-App guardrails); P2 as a runtime simulator for training, since rewards must not depend on unverified LLM transitions.

## 2610.26828 — Agent World Model in the Wild: Computer-Use Agent Environment Synthesis (Wang et al., 2026; UNC / Microsoft Research et al., arXiv)
Read: FULL (body + Appendices A.3–A.5; case studies skimmed).  Relevance to Part 2: **high**. A complete open-model World + task + verifier factory with five execution-based gates, ablated one gate at a time, then RL on **Qwen3.6-35B-A3B, Crucible's own student**, with gains on 8 OOD benchmarks. Its persona-to-multi-day-world synthesis with deterministic replayable deltas is the closest match to Crucible's dense-World plan, and its counterfactual "gaslight" probe is a new verifier test.

**Problem.** CUA environments built by hand do not scale; synthetic ones are single-app, visually simple and usually built with proprietary models. Tasks must be solvable, hard enough, and have reliable verifiers, without assuming "a frontier model is always right".

**Method / algorithm.** All synthesis by open models (generator and agent judge GLM-5.2 in OpenCode).
- **World synthesis (persona → day plans → deterministic deltas):**
  1. *Persona creator* writes a "persona bible" grounded in 15 real mirrored websites (WebHarbor SQLite catalogues: Amazon, Booking, Coursera, GitHub ...) and the shared MCP app index: demographics, ≥3 named contacts, 4–8 MCP services tied to story arcs, sites, ≥2 multi-day story arcs (≥1 "image-decisive"), routine, web accounts with fixed-salt password hashes, and a compact signature. A *diversity judge* rejects personas sharing occupation + life domain, core tool or site pattern, or narrative arc with any accepted one (≤2 revisions).
  2. *Day planner* reads the real cumulative state W(c, d−1) and prior plans, writes day d's plan across CLI/files, MCP and web, with **coverage pressure** (per-app and per-site usage histogram from earlier days) and a mapping from activities to reusable event skills and MCP apps. A skill-creator writes missing event skills (interfaces, state changes, cross-record consistency constraints) and an app-creator implements missing MCP services (schema, backend, server), all added to a shared library.
  3. *Day implementer* ("god-mode", inside the booted world) writes `delta.sh` + deterministic Python sub-scripts that, replayed hermetically over day d−1, reproduce day d identically: frozen timestamp literals, derived IDs (MAX(id)+1), idempotent inserts, fixed-salt hashes; plus `produces.json` declaring every write.
  4. *World verification:* (i) deterministic lint incl. cross-persona identity reservation; (ii) **replay twice, identical content hash**; (iii) every declared artifact in `produces.json` exists and changed; (iv) live health: boot, every account logs in and renders history, all MCP services load, credentials discoverable in the workspace; (v) an independent reviewer agent outside the container checks the snapshot against the plan (intent, not exact values; reject only genuine defects such as missing artifacts, broken FKs, non-frozen timestamps).
  - Each world instance shares the application layer but isolates mutable state (DBs, site histories, files).
- **XAI (cross-interface) task synthesis:**
  1. *Sketch agent* explores a live world W₀ = W(c, d) and proposes ≤4 sketches; each must need CLI, MCP and web, each indispensable, and must include writing and running a script over a real user-visible source whose result drives a later MCP or web decision.
  2. *Diversity gate:* an LLM picks sketches adding the most topical variety vs accepted and concurrently locked tasks; near-duplicate = same core action + artifacts; explicitly no fixed quotas.
  3. *Task generator* (given a live capability contract from active MCP packages and schemas) emits seven artifacts: public `task.md` (first-person, ≤300 words), private `reason.md` and `meta.json` (incl. the private truth of any image-decisive fact), `golden.py` (reference completion), rubric, verify skill, and deterministic verifier scripts built from an SDK of diff primitives; mandatory self-check loop until all green. A hint is derived from `reason.md` by a separate call.
  4. *Verifier form:* V(W₀, W, a) = Π_i v_i(W₀, W, a), a checklist of N binary criteria over the start state, end state and final answer.
  5. *Rollouts:* 6 per task, half with hints, sampled from 18 model–harness pairs (GLM-5.2, DeepSeek-V4-Flash, MiniMax-M3 × OpenCode, Codex, Claude Code, Nanobot, OpenClaw, Hermes).
  6. *Three judges per rollout:* the programmatic verifier; an LLM judge over a machine-extracted evidence digest (completed / failed / world_imperfect with confidences; judges intent, not exact strings); an agent judge with read-only shell and vision on BEFORE/AFTER snapshots, which must never accept agent-authored text as proof of an external fact, must check images independently, and labels each failed check too-strict or genuine-miss.
  7. **Five gates:** (a) *judge consistency*: V = V_LLM = V_agent on all 6 rollouts; (b) *initial unsatisfaction*: V(W₀, W₀, ∅) = 0; (c) *solvability*: ≥1 successful rollout, and successful runs must use every declared interface; (d) *difficulty*: k = successes among the 3 unhinted rollouts; k ≤ 1 hard, k = 2 medium, k = 3 too easy (sent back to make harder); (e) **counterfactual probe**: fresh session, blind re-judge, then assert the opposite verdict ("an independent audit believes...") and see if the judge defends with cited evidence (pass), caves by editing verifier scripts (fail; edit detected by diff and reverted), or shows the judge was wrong (route to review).
  8. Failures return with diagnostics for ≤2 fix rounds, then must re-pass every affected gate; a probe failure lets a repair agent edit only verifier and rubric, task frozen. Lessons feed the generator's skill library.
- **Training:** SFT warm start (base Qwen3.6-35B-A3B pass@1 5.7%, pass@4 18.4% on XAI; 4B 0.38%): 4 GLM-5.2 rollouts on each of 100 SFT tasks, keep ≤1 success per task, split at compaction boundaries, drop segments >102,400 tokens, 3 epochs, LR 5e-6. Then GRPO on 208 tasks, ≤64 steps, 8 tasks × G = 4 rollouts (4B: 8), LR 1e-6, KL 0, clip 0.2/0.28, entropy 0.001, 256 turns, 30 min per rollout; MoE router and vision encoder frozen.
  - Reward: r = −1 on a format failure (turn hits token limit, malformed reasoning tags, unparseable or unknown tool, empty output; rollout stops and only that turn gets gradient), else r = min{1, V + (0.1/N) Σ v_i} (checklist partial credit).
  - Advantage: (r − μ)/σ over the pool; if σ = 0 and every rollout is a format failure, A = −K^(−1/2); otherwise 0.
  - **Adaptive rollouts:** per-task EMA multiplier κ ∈ [1, 3] (weight 0.8); extra chunks of 8 rollouts until the pool has ≥2 distinct rewards (or only format failures); then pick a group of G spanning lower and upper reward halves. Sandbox or harness errors retried ≤3 times then excluded; timeouts still graded.

**Verification / grading.** Programmatic checklist of diff checks over W₀, W and the answer, cross-validated in synthesis by the LLM and agent judges and the probe. On 100 sampled rollouts: agreement with judge majority 0.95, κ 0.90 (vs GLM-5.2 judges precision 0.98 / recall 0.93; GPT-5.6-Sol 0.85 / 0.98; Opus-5 0.98 / 0.88). Image-decisive steps (two products identical on specs, differ only in photo) defeat text-only shortcuts.

**Scale & cost.** 36 personas, 640 world states (8–30 per persona, mean 17.8), 15 sites + 8 MCP services, 83 reusable skills, 171.6K DB records, 50.1 local files, world state 59.3 MiB, median 52.4K tokens (vs 1.1K for CUA-Gym's injected states). Tasks: 341 accepted (273 hard, 68 medium) at a **53% acceptance rate**, 4.2 sub-tasks and 3.1 interface switches on average, 74.5% revisit an interface; synthesis SR 21.3% without hints, 45.6% with. Per task: 2.5 revision rounds mean, **69.8M generator input tokens**, 1.1M output; solving takes ~80 LLM calls, 4.1M input tokens. Fix counts per task: judge consistency 1.09, solvability 1.16, difficulty 0.39, probe 0.18, initial-unsat 0.05. Hardware: 8 B200s for synthesis and training. Split: 100 SFT / 208 RL / 33 in-domain eval.

**Key results.**
- Qwen3.6-35B-A3B average over 8 OOD benchmarks 42.84 → **53.42** (+10.58): Online-Mind2Web 69.60 → 80.77, WebArena 53.54 → 67.26, ClawEval 38.00 → 54.67, OSWorld-Verified 33.16 → 49.22. Qwen3.5-4B 21.83 → 30.04. Gains on all 8 for both.
- Tasks are hard: GPT-6-Astra ~60% pass@1 and still fails 27.3% after 8 tries; every task solved by ≥1 of 24 agent configs, none by all.
- Environment diversity: higher Vendi than CUA-Gym at every sample size; nearest-neighbour cosine distance 0.25 vs 0.20; coverage reached 134 domains and 91 task types without saturation.
- **Gate ablations (SFT data, 4B; avg over O2W, QCB, OSV, in-domain):** full 31.86; w/o judge consistency 22.56 (in-domain 21.59 → 3.79); w/o initial-unsatisfaction 24.08 (3.05); w/o difficulty 26.37 (7.58); w/o probe 27.91; w/o solvability 29.26. Every gate matters; judge consistency and initial unsatisfaction most.
- **RL ablations (35B-A3B):** w/o SFT warm start −5.31; w/o checklist partial credit −2.25; w/o format reward −1.24.
- Behaviour: gains concentrate on OOD tasks in the 2nd and 3rd difficulty quartiles (+19.9, +12.2), none on the hardest quartile (beyond the teacher). Interface switching is learned from SFT (18.4% → 24.7%); RL adds long-horizon solving (tasks solved between 40 and 120 steps +17.0 vs +9.7 for SFT alone). RL without SFT or RFT on own successes *reduces* cross-interface switching.

**Failure modes & limits.** Only 341 tasks; task count skewed to the first 6 personas (178 tasks); difficulty labels from 3 unhinted rollouts are noisy; same-family risk: GLM-5.2 writes worlds, tasks and is a judge and SFT teacher; LLM diversity judge without explicit coverage grid; credentials files in the workspace; no comparison of training value per gate cost. Gains stop at the teacher's frontier.

**Tips & tricks.**
- Build Worlds as a sequence of days; each day is a deterministic replayable delta over the previous state, accepted only if two replays hash equal.
- Freeze timestamps, derive IDs, make inserts idempotent, fix password salts.
- Declare every write in a manifest and verify each declared artifact exists and changed.
- Grow a shared library of event skills and services; reuse across personas.
- Add per-app usage histograms to the planner prompt as coverage pressure.
- Persona diversity: reject on same occupation + domain, same tool pattern or same arc.
- Verifier = product of binary diff checks over (start, end, answer); give 0.1/N partial credit in RL.
- Gate "initial unsatisfaction": the verifier must score 0 on the untouched start state with an empty answer.
- Require that successful rollouts use every interface the task claims to need.
- Probe judges by asserting the opposite verdict; any verifier edit under pressure fails the task.
- Agent judges must never accept agent-written text as evidence for an external fact.
- Make some steps decidable only from images or other ground truth the agent must inspect.
- Grow rollouts per task adaptively until rewards differ; pick groups spanning reward halves; give format failures a small negative advantage when all fail.
- Warm-start with ≤1 success per task from a teacher before RL on hard long-horizon tasks.

**Fit to Crucible Part 2.**
- **F1/F2:** adopt the day-delta World build (deterministic replay, double-replay hash = G2 for the World itself, `produces.json` declaration check, live health, independent reviewer outside the container) as the generic `SeedBundle` construction for dense Worlds; coverage pressure in the planner is a cheap diversity lever for F2.
- **F5/F6 gates:** their five gates map onto §3.1: initial unsatisfaction = G4; solvability = G3 (but by rollouts, where Crucible prefers construction; use both); judge consistency = an executed + judge cross-check that Crucible lacks; **counterfactual probe = a new G-gate for any LLM or agent judge** (and a natural canary for §3.8's "vacuous verifier" guard). Their ablation is the strongest published evidence that each gate pays for itself in training value.
- **F9 difficulty:** k-of-3 unhinted plus hinted rollouts gives a cheap band and a hint-based ladder; their student is ours, so their numbers (base pass@1 5.7% on hard CUA tasks, SFT warm start needed, gains limited to tasks within teacher reach) directly inform the 35B band and the need for ladders.
- **F7/F10 intent:** "successful rollouts must use every declared interface" is a cheap behaviour-signature check that success needs the intended skill; image-decisive steps are an anti-shortcut design pattern.
- **Training (§2(a)):** adaptive rollout multiplier and reward-diverse group selection are directly reusable for the 35B on Tinker.
- **Avoid:** same model as world builder, judge and teacher (Crucible's family rules); LLM-only diversity judgement (use the behaviour-signature grid).
- Priority **P0** (gates + probe + day-delta World); P1 for the RL recipe details.

## 2609.37658 — EnterpriseBench: Benchmarking LLM Agents on Enterprise-Level Strategic Reasoning and Decision-Making (Yang et al., 2026; Shandong U. / Zhongguancun Academy, arXiv)
Read: FULL (body + Appendices A.4–A.6, A.10; A.7–A.9, A.11 skimmed).  Relevance to Part 2: medium. An evaluation suite, not a factory, but its Consulting task is a clean template for converting expert case material into a hidden-information, multi-turn simulated-person task (gated facts revealed only when asked), and its two simulators are examples of code-defined enterprise dynamics with cheap numeric rewards.

**Problem.** Enterprise and financial benchmarks test static QA; real enterprise work involves missing information, delayed feedback and long-horizon trade-offs.

**Method / algorithm.**
- **Foundational layer:** re-organises existing financial/enterprise QA sets into four capability categories (Information Extraction, Numerical Calculation, Domain Knowledge, Complex Reasoning), each sample annotated with a 0–10 difficulty from context length, reasoning depth, ambiguity, label-space complexity and number of steps (Easy 0–3, Middle 4–6, Hard 7–10) by an expert-LLM prompt; human-audited.
- **Consulting (423 cases from MBA casebooks):** each case = visible problem statement + hidden case facts + reference solution. An LLM **interviewer** runs ≤12 rounds. Prompt rules: the problem and background are public; sections like "information provided if requested" and "hints" are **gated facts**, revealed only when the candidate's question clearly targets that dimension, one logical piece at a time; "solution", "analysis", "key questions", "interviewer notes" are internal-only guidance, never quoted; never invent facts not in the case (say the case does not provide it); one focused follow-up per turn; redirect a candidate who keeps asking for data without analysing; small hints only when stuck; end only after a recommendation with 2–3 drivers and brief feedback, with an end token; never in the first message. Pacing from case length (short: ≥3–4 answers; long: 7–10).
- **Judge:** LLM from a different family than the evaluated backbone scores structure, quant, business sense, communication (0–10) plus overall; strict calibration bands; do not reward verbosity; penalise hallucinated facts beyond the case.
- **Beer Game:** 4-stage supply chain, agent is the retailer with local observations only; 25 steps; demand 100 → 400 from week 2; 1-week information delay, 2-week shipping delay; holding cost 0.5/unit/step, backorder 1.0, minimum cost 200; other roles follow q_t = d_t + α(I* − I_t) + βB_t with α = β = 1 (or a 4-step smoothed-demand variant). Score: total cost.
- **Enterprise Digital Twin (transentis):** the agent outputs one scenario-level plan {C, R, P}: number of consultants, global revenue-risk level, and per-project accept / start / deadline; a constrained transform applies it to a template JSON scenario (12 consultants with salary and workplace cost; 10 projects with contracted effort, billing rate, deadline, extension and follow-on probabilities); the simulator runs 96 steps and returns revenue, expenses, utilisation, margin and accumulated earnings (score).
- **Agents compared:** CoT, Self-Refine, Reflexion, A-MEM, Debate, Discussion, Dynamic Cheatsheet, GEPA, ACE on DeepSeek-V3, GPT-4.1, DeepSeek-V4-Pro, GLM-5.2; plus AOA, a router that learns rules from traces mapping sample features (category, difficulty, table/code, source) to a method.

**Verification / grading.** QA: exact / normalised match, MC accuracy, numeric tolerance, code execution. Simulators: code-computed cost or earnings. Consulting judge validation: 20-case blind audit r = 0.71 (LLM higher than humans by 8.5%, mostly quant and communication); expanded 500 transcripts × 3 annotators: overall Pearson 0.887, Spearman 0.736, MAE 0.679, within-1 85.0%, human ICC(2,3) 0.892; quant within-1 only 67.4%. Prompt robustness: reordering or simplifying interviewer and judge prompts moves the overall score ≤0.12 (1.45%). Difficulty labels vs 2 experts on 50 tasks: r = 0.96, MAE 0.47, 94% within 1 point. Hackable: judge rewards transcripts, so a candidate that extracts gated facts by broad questioning, or flatters, may score; an interviewer LLM can leak gated facts.

**Scale & cost.** 423 consulting cases; Beer Game and EDT are single configurations (one scenario each), so effectively 2 tasks; budgets in A.11 per method on 50 instances (not tabulated here).

**Key results.** No single method wins: DeepSeek-V3 best Consulting GEPA 8.28, best Beer Game cost Discussion 1.94×10⁴, best EDT A-MEM 6.76M; GPT-4.1 best Consulting DC 8.08, Beer Game and EDT GEPA (3.09×10⁴; 8.16M). Self-Refine and Reflexion score below CoT on Consulting. Strong QA does not predict interactive performance. Numerical and domain-knowledge accuracy drop sharply from Easy to Hard (e.g. DeepSeek-V3 numeric 0.84 → 0.50). AOA routing gives 0.729 vs 0.725 best fixed method (tiny).

**Failure modes & limits.** Simulators are single scenarios (no task distribution), so not usable for training without parameter sampling; consulting reward is an LLM rubric; text only; casebook contamination likely (public MBA casebooks); no RL.

**Tips & tricks.**
- Split expert case material into public, gated-on-request and internal-only sections; the simulated person reveals gated facts one at a time only when the question targets them.
- Have the simulated person say "not provided" instead of inventing facts.
- Redirect data-hoarding without analysis; give small hints only when stuck.
- Use a judge from a different family than the policy; report per-dimension agreement with 3 annotators (ICC) and within-1 rates.
- Test judge prompt robustness by reorder and simplify variants.
- LLM difficulty annotation with explicit anchors can match experts closely (r 0.96).
- Equation-based counterpart policies give cheap, deterministic multi-agent pressure.

**Fit to Crucible Part 2.**
- **F3 multi-turn tasks with a simulated user (§3.3):** adopt the gated-fact protocol as the `user_known_info` contract: the user's hidden facts are tagged public / gated(dimension) / internal, the user simulator reveals only on targeted questions, and G8 "communicable" can be checked mechanically (every oracle-needed fact is in public or gated sets). Grade the outcome by state or answer checks, not by a consulting-style rubric, unless the rubric judge passes per-dimension agreement on the anchor set.
- **F1 Apps:** the Beer Game and EDT show how to make a business simulator an App: parameters in a JSON template, the agent's decision as a constrained transform, deterministic rollout, numeric reward; Crucible would add seed sampling over demand scripts, delays and project mixes to get a task family with monotone knobs (delay, demand shift size, number of projects).
- **F9:** their difficulty-annotation anchors (context length, depth, ambiguity, label space, steps) are a candidate input to the cheap predictor.
- Priority P2 (patterns), P1 for the gated-fact user protocol.

## 2412.05467 — The BrowserGym Ecosystem for Web Agent Research (Le Sellier De Chezelles et al., 2024; ServiceNow Research / Mila et al., arXiv)
Read: FULL (body; appendices on action set, costs and runtime skimmed).  Relevance to Part 2: low–medium. Infrastructure, not synthesis. Useful for Crucible as (a) a minimal task contract (`setup()` returns the goal, `validate()` returns reward, done, message), (b) the action-mapping pattern for a restricted, safe action space, and (c) reproducibility features (version journal, a replay agent that re-executes an action sequence to detect environment drift).

**Problem.** Web agent benchmarks are fragmented across incompatible codebases, so agents and models cannot be compared fairly or cheaply.

**Method / algorithm.**
- **BrowserGym:** a gymnasium POMDP over a chat + Chromium (Playwright) browser. Observations: goal or chat messages (text and images), open tabs (URLs, titles, active index), raw DOM and AXTree via CDP with injected unique element ids (`bid`), per-element bbox, visibility ratio and set-of-marks flag, screenshot, and `last_action_error` (exceptions returned as feedback rather than crashing the loop).
- **Actions:** the base space is raw Python with a Playwright page plus `send_message_to_user` and `report_infeasible`; an `action_mapping` function converts a restricted action language (default high-level primitives: click(bid), fill, press, mouse_click(x, y), tabs ...) into trusted code; parse errors become feedback. The action set's `describe()` generates the agent-facing documentation.
- **Task API:** a task subclass implements `setup(page)` (log in, create data, navigate; returns the goal, possibly with images) and `validate(page, chat_messages)` called after every action (inspect page, DB, chat; return scalar reward, done, optional user message to simulate interaction). `register_task` exposes it as a gym env.
- **Benchmarks unified:** MiniWoB++ (125 templates), WebArena (812), VisualWebArena (910), WorkArena L1 (33 templates, high seed diversity), L2 and L3 (341 each), WebLINX (31,586 static turns), AssistantBench (214, open web). Each ships metadata: train/test split, suggested action set, seeds per task, max steps, and a **task dependency graph** for shared-backend benchmarks; `prepare_backend()` resets Docker backends between agents.
- **AgentLab:** `Study` objects run parallel episodes (joblib or ray; ray respects task dependencies, which caps WebArena parallelism to 2–4), auto-relaunch failed tasks up to 3 times, record versions (benchmark, packages, commit, OS, time) in a reproducibility journal; **ReproducibilityAgent** replays a previous study's action sequence on the same seeds and diffs prompts to expose environment or model drift; AgentXRay UI for step-level trace inspection; `DynamicPrompting` shrinks prompt components to a token budget instead of truncating; GenericAgent retries parsing 4 times before failing a task.

**Verification / grading.** Per-benchmark validators (state or answer checks); success rate with standard error σ/√N. Error taxonomy (qualitative): navigation, form handling, task understanding, stuck behaviour, information extraction, external errors.

**Scale & cost.** Claude-3.5-Sonnet full sweep: MiniWoB 625 episodes $23.11, WorkArena L1 330 $100.03, L2 235 $299.44 (33.8 steps per episode), L3 235 $191.50, WebLINX 2,650 $104.62, WebArena 812 $138.76, VisualWebArena 910 $134.69, AssistantBench 181 $37.33. 20 parallel jobs on a laptop is feasible; environments, not compute, are the bottleneck.

**Key results.** Claude-3.5-Sonnet leads most benchmarks: WorkArena L2 39.1% vs GPT-4o 8.5% and o1-mini 14.9%; WebArena 36.2 vs 31.4; L3 ≈ 0 for all. GPT-4o leads VisualWebArena (26.7 vs 21.0). Llama-3.1-405B beats GPT-4o-mini on several. GPT-4o's WebArena score rose 23.5 → 31.4 between checkpoints, which the authors partly attribute to possible contamination of public benchmarks.

**Failure modes & limits.** Live-web non-determinism (locale, ads, layouts), bot detection, agent collisions on shared backends (one agent per WebArena VM), synchronous loop, safety of arbitrary code actions.

**Tips & tricks.**
- Task = `setup()` returning the goal + `validate()` returning (reward, done, message) after every step.
- Return action exceptions as observations; never crash the episode.
- Restrict the action space through a mapping layer that compiles a safe DSL to code.
- Inject stable element ids so actions are unambiguous.
- Record a task dependency graph when tasks share mutable backend state; reset backends between agents.
- Keep a reproducibility journal (versions, commit, OS, timestamp) and a replay agent that re-executes old action sequences to detect drift.
- Fit prompts to a budget by shrinking components (history, page tail), not by tail truncation.

**Fit to Crucible Part 2.** Mostly F1/F6 infrastructure patterns that Crucible already has in other form (Harbor tasks, snapshot/restore). Adopt: (1) the replay-agent drift detector as a periodic F6 check that a stored oracle action sequence still produces the same end-state fingerprint after any World or App change (the World loop's "frozen expected states" test, §3.8); (2) task dependency metadata when tasks share a World instance; (3) `report_infeasible` as a first-class terminal action for the infeasible task type. Avoid live-web tasks for training (non-determinism, bot detection). Priority P2.

## 2504.01382 — An Illusion of Progress? Assessing the Current State of Web Agents (Xue et al., 2025; Ohio State / UC Berkeley, arXiv)
Read: FULL (body + Appendices B, E; case studies skimmed).  Relevance to Part 2: medium. It is a sharp case study of how a synthetic, shortcut-prone benchmark inflates results (a Google-search-only agent solves 51% of WebVoyager), plus a validated trajectory judge (key points + key screenshots) with numbers on agreement, precision and token cost. Both bear on G9 no-shortcut probes and on judge acceptance for Crucible.

**Problem.** Reported web-agent success near 90% on WebVoyager contradicts experience. The authors attribute it to easy, shortcut-solvable, low-diversity synthetic tasks and to an unreliable LLM judge, and build a better online benchmark and judge.

**Method / algorithm.**
- **Shortcut audit:** a naive search agent (one Google query, click one result, check for the answer, no site interaction) on 100 stratified WebVoyager tasks: 51% success, manually judged. On the new benchmark it gets 22% (easy ~50%, medium 18%, hard 3%).
- **Benchmark construction (Online-Mind2Web):** sample 650 Mind2Web tasks and check viability: 47% are invalid or have outdated ground truth. Remove ambiguous tasks, tasks no longer executable, and CAPTCHA-protected sites. Keep 167 maximally distinct Mind2Web tasks, rewrite 24 (to restore solvability or remove ambiguity), add 34 from Mind2Web-Live, and hand-write 75 new tasks on high-traffic sites (Similarweb): **300 tasks over 136 sites**. Difficulty by human reference length N_step: easy ≤5 (83), medium 6–10 (143), hard ≥11 (74). Commit to replace outdated tasks with same-difficulty ones.
- **Evaluation protocol:** start URL given; agents told not to use Google Search (else they finish on other sites, confounding difficulty); every trajectory human-labelled by ≥2 annotators with a third resolving conflicts.
- **WebJudge (LLM-as-judge for a trajectory of actions and screenshots, not the agent's final message):** (1) key-point identification from the task text (avg 3.6 points; e.g. location and rating); (2) per-screenshot description and relevance score 1–5; keep screenshots with score ≥ θ (θ = 3 best; 2 floods context, 4 drops evidence); (3) outcome judgement from task, key points, key screenshots and action history. Two modes: CoT (one call; used) and key-point-wise (all points must pass; better for ≤3 points, too strict beyond; combined 84.6%). Final responses excluded because they often hallucinate success. WebJudge-7B: Qwen2.5-VL-7B fine-tuned for key-screenshot identification, 2 calls per trajectory.
- **Efficiency metric:** E = mean over successful tasks of (agent steps / human reference steps).

**Verification / grading.** Human labels are the reference. WebJudge agreement with humans: GPT-4o 83.6%, o4-mini 85.7% (success-rate gap 3.8 pts), WebJudge-7B 87.0% (gap 3.9). Baselines: Autonomous Eval 79.4–79.5%, WebVoyager judge 73.9–78.7%, AgentTrek 66.9%; their agreement falls as trajectories lengthen (Operator often >100 screenshots); WebJudge holds to ~80 steps. Three reruns: SD of success rate 1.1% average, 1.8% max. On AgentRewardBench (1,302 trajectories, 5 benchmarks): precision WebJudge o4-mini 82.0% (≈ rule-based 83.8%), 7B 75.7%, GPT-4o 73.7%, prior best 69.8%; success-rate gap vs human: WebJudge GPT-4o 5.9 vs rule-based 9.8 vs prior 8.1; o4-mini is conservative (low recall, gap 10.5). Token cost per task: CoT 67K (47K for screenshot scoring), key-point-wise 120K.

**Scale & cost.** 300 tasks × 6 agents, all human-labelled; agents capped at 25 steps (open agents) or 50 (Claude CU 3.7); Operator collected manually from its web UI.

**Key results.** Human-judged success: Operator 61.3%, Claude Computer Use 3.7 56.3%, SeeAct 30.7%, Browser Use 30.0%, Claude CU 3.5 29.0%, Agent-E 28.0%; several 2024–25 agents do not beat SeeAct (early 2024). Success drops 31.6 pts from easy to medium and 15.4 more to hard; Operator and Claude 3.7 reach 83–90% on easy only. Failed trajectories take ~2× the steps of successful ones (loops, pop-ups). Operator explores (2.6× human steps, up to 44 min) vs greedy agents (~1.0×) that hit dead ends. Operator errors: filter & sorting 57.7%, navigation 19.6%; misses numeric and temporal constraints. Allowing Google Search lifts Browser Use only 26 → 31%.

**Failure modes & limits.** Live sites drift (tasks need maintenance); judge excludes final answers, so information-seeking answers are judged from screenshots only; key points generated by the judge can be wrong or unnecessary; 300 tasks; no training use.

**Tips & tricks.**
- Audit any task pool with a trivial shortcut agent (here: search-only); report its success by difficulty.
- Re-validate seed tasks before reuse: half of an older pool was stale.
- Rewrite rather than drop tasks when ambiguity or staleness is the only defect.
- Band difficulty by human reference length.
- Judge trajectories, not the agent's own claims; agent final messages hallucinate success.
- Select relevant intermediate evidence by a scored filter (θ = 3 of 5) to avoid both context overload and missed steps.
- Key-point-wise judging is better for few requirements; holistic CoT for many.
- Rerun the judge 3× to measure its variance.
- Prefer a judge with high precision for reward use; check the success-rate gap too.

**Fit to Crucible Part 2.**
- **G9 no-shortcut (F6) and F10 null-agent probes:** their search-only agent is a concrete instance of the "instruction only / partial input" probe; Crucible should keep a small library of domain-specific trivial agents (search-only, read-only, answer-from-text) and gate on their success.
- **F5 judges for non-state outcomes:** "never trust the policy's own final message as evidence" matches WildAWM's agent-judge rule; adopt for any answer check that a model judges. The key-point decomposition is an automatic way to derive answer-check rubrics, but generated key points need review (they overconstrain beyond ~3).
- **F9:** human-reference-length banding and the efficiency ratio E are cheap difficulty and behaviour features for the predictor and the signature.
- **Source quality (§3.5 item 7):** a synthetic pool can look rich and be shallow; measure the pool, not the generator's claims.
- Priority P1 (shortcut-agent gate), P2 otherwise.

## 2411.06559 — Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents (Gu et al., 2024; Ohio State / Orby AI, arXiv v2 2025)
Read: FULL (body + Appendices B–D).  Relevance to Part 2: low–medium. It is an inference-time planning method, not a task factory; relevant as (a) a cheap way to generate transition data at scale (random walks + before/after descriptions, 3.1M instances), (b) an intrinsic evaluation protocol for any simulator (rank ground-truth vs deviation actions), which Crucible could reuse to test LLM-backed Apps or people, and (c) evidence that LLM simulation degrades quickly beyond one step.

**Problem.** Tree search helps web agents but needs backtracking, which irreversible real-web actions forbid, and costs ~10× latency. Use an LLM as a world model to simulate candidate actions before committing (model predictive control).

**Method / algorithm.**
- **WebDreamer loop (Algorithm 1):** at each step, propose top-k candidate actions; an LLM **self-refine** step drops clearly irrelevant ones; for each remaining action, `sim(o, a)` produces a natural-language description of the state change (and, for H > 1, imagines the next action and its effect); `score` (GPT-4o) rates each simulated trajectory complete 1.0 / on track 0.5 / incorrect 0, averaged over several samples; execute the argmax action; stop on a stop action, max steps or an action repeated 3 times. H = 1 works best.
- **World-model data synthesis:** start URLs from the October 2024 Common Crawl index; heuristic random walks (click, hover, type, select; click-heavy probabilities; prefer newly appeared elements to capture causal chains; GPT-3.5 writes contextual search queries); screenshot before and after; Qwen2-VL-72B describes the target element (referring expression) and the functional state change (red box drawn around the element); combine with one of 7 prompt templates; filter failed interactions, bot-blocked and harmful pages. 3,160,247 instances over 1,247,960 URLs (click 84%, hover 7.6%, type 6.9%, select 1.5%; scroll dropped as useless; mean step position 4.4).
- **Dreamer-7B:** Qwen2-VL-7B fine-tuned by next-token prediction on (screenshot, action) → change description; 64 H100s, ≤2 epochs, LR 1e-6, batch 192. In-domain continual training: 25K synthesized interactions per VWA site, filtering search actions whose queries appear in test tasks, LR 5e-7, 1 epoch.
- **Intrinsic evaluation (for checkpoint selection):** 44 Mind2Web tasks with human ground-truth trajectories; at each state pick top-5 alternative elements by the MindAct ranker, extend each into a full trajectory with SeeAct, judge with an LLM, keep only deviations whose trajectories *failed* (to avoid false negatives from alternative valid paths): 141 states, 279 deviation actions. Metrics: pair-wise accuracy (ground truth scored ≥ deviation), state-level (≥ all deviations; primary), task-level (all states correct).

**Verification / grading.** Benchmarks' own evaluators (VWA programmatic; Online-Mind2Web WebJudge ~85% human agreement; Mind2Web-Live key nodes). Intrinsic task-level accuracy correlates with Mind2Web-Live success at Pearson 0.8455 across world models.

**Scale & cost.** 3.1M synthetic transitions; 64 H100s for training. VWA wall-clock per task: reactive 68–88 s, tree search 749–972 s, WebDreamer 179–234 s; steps: tree search ~3× reactive, WebDreamer ≈ reactive.

**Key results.** Success: VWA reactive 17.6, tree search 26.2, WebDreamer (GPT-4o) 23.6, Dreamer-7B 21.9, + in-domain 23.2; Online-Mind2Web 26.0 → 37.0 (GPT-4o) / 35.0 (Dreamer-7B); Mind2Web-Live 20.2 → 25.0 / 24.0. Dreamer-7B beats Qwen2-VL-72B as world model. Ablation: reranking candidates by score without simulation gives a small gain only; removing self-refinement hurts. H = 2 and 3 underperform H = 1 because imagined actions hallucinate availability and trajectories become indistinguishable. Intrinsic: Dreamer-7B state-level 82.98 vs GPT-4o 80.85. In-domain training did not help Reddit (dense text pages).

**Failure modes & limits.** Multi-step simulation error accumulation; noisy, verbose GPT-4o predictions; scoring by an LLM; world model outputs text deltas, not state, so no persistent consistency (contrast Trace2Env).

**Tips & tricks.**
- Random-walk real or synthetic UIs, record before/after, and have a strong VLM describe the functional change: cheap, large transition corpora.
- Bias random walks toward newly appeared elements to capture causal dependencies.
- Evaluate a simulator intrinsically by whether it ranks the ground-truth action above failing deviations; drop deviations that also succeed.
- Keep simulation depth at 1 unless state is tracked explicitly.
- Filter candidate actions before simulating; noise from irrelevant candidates hurts.
- Filter synthetic data against test queries to avoid leakage.

**Fit to Crucible Part 2.** Limited. (1) The intrinsic ranking test is a reusable acceptance test for any LLM-simulated App or simulated user in the World loop (§3.8): given a gold path and failing deviations, does the simulator's response let a fixed scorer separate them? (2) Random-walk transition mining over Crucible's own coded Apps could produce cheap tool-test fixtures and seed "explore → describe → propose task" pipelines (F3). (3) Its depth-1 limit argues against reward-bearing LLM simulation over long horizons. Priority P2.

## 2510.14969 — LLMs as Scalable, General-Purpose Simulators For Evolving Digital Agent Training (Wang, Yin et al., 2025; Harvard / UCLA, arXiv; "UI-Simulator")
Read: FULL (body + Appendices A–F; prompts G skimmed).  Relevance to Part 2: medium. SFT data synthesis in an LLM-simulated UI, not verified RL tasks, but two mechanisms transfer directly: **step-wise task control** during exploration (a measured diversity fix) and **UI-Simulator-Grow** target selection (keep tasks in the 25–75th percentile of teacher-forcing loss, then generate light variants), a cheap difficulty-band proxy and variant generator.

**Problem.** UI agent trajectories are expensive to collect in real environments (e.g. 1,800 human hours for 360 OSWorld tasks), and real environments are slow, flaky and hard to parallelise. Use an LLM as the environment to synthesise diverse UI states and trajectories.

**Method / algorithm.**
- **Simulator:** state s_t = accessibility tree with text, bounding boxes and dynamic attributes; observation = elements whose bbox intersects the viewport. Hybrid transitions: **rule-based** for deterministic actions (type appends to the element's content; scroll moves the viewport offset by fixed displacement with the state unchanged; tab, back and forward via an explicit browsing stack) and **LLM-based** for the rest, in three few-shot CoT steps: (1) predict an overview of the next state ("search results for sneakers"; 4 examples); (2) write a rich unstructured draft of sections and elements (1 example); (3) convert the draft to the structured tree and assign coordinates (1 example). GPT-4o-mini, temperature 0.5.
- **Retrieval-augmented variant (-R):** a small offline corpus of real transitions from the target environment (1,647 for WebArena, 683 for AndroidWorld, ~25% and ~10% of what OS-Genesis uses); hybrid retrieval: BM25 on action history → GPT-4o semantic re-rank → BM25 on a composite state + history key; the simulator conditions on the retrieved state.
- **Instruction-free guided rollout:** the teacher (GPT-4o-mini) proposes a high-level **task control** c_0 from the initial state ("search for a product"); generates thought, action and step summary under the current control; a Done(c_i) check by the teacher triggers a new control proposed from the current state and the previous control ("check order history" after "go to my account"); the teacher decides when to STOP. Controls per proposal: 3 (map) to 8 (GitLab, shopping admin).
- **Trajectory wrapper:** summarise the completed rollout into a user instruction G (retrospective labelling); rewrite thoughts to be conditioned on G; optionally insert reasoning steps for information queries; filter trajectories whose actions target invalid elements or cause no meaningful change, or whose stated action differs from the taken action.
- **UI-Simulator-Grow (targeted scaling):** each iteration, compute the **teacher-forcing loss** of the current student on a validation set (teacher actions as labels, per-step cross-entropy averaged over the task); keep tasks between the 25th and 75th percentile as targets (below = already learned, above = too hard or ambiguous); synthesise **lightweight rewrites** of each target ("search running shoes" → "search slippers") that keep action types and flow and change only content in the states; continual learning with **replay** of the most representative previous tasks (top row-sums of the instruction cosine-similarity matrix, RoBERTa-large Sentence-Transformer). The validation set is refreshed each iteration from newly synthesised data.

**Verification / grading.** No per-task verifier: the teacher judges completion; quality by filters and a human evaluation of 40 trajectories × 3 annotators over 8 dimensions (realism, state reasonability, action validity, thought consistency, completion, trajectory consistency, irrelevant steps, abstraction): mostly ≥0.86–0.95 satisfied; pairwise annotator agreement 0.876–0.976. Downstream: WebArena (812 tasks) and AndroidWorld (116) success rates with their own evaluators.

**Scale & cost.** 2K web trajectories (avg 3.3 steps) and 1.3K Android (avg 5 steps). Cost per web trajectory $0.02 (retrieval-free) / $0.05 (retrieval-augmented), double for Android. SFT of Llama-3-8B (web) and Qwen-2.5-7B (Android) on 4 A6000s, LR 1e-5, 2 epochs.

**Key results.** WebArena / AndroidWorld success: base 2.34 / 0.0; UI-Simulator-F (no target-env data) 6.28 / 8.6; -R 6.40 / 12.9; Grow-R 7.14 / 13.4 (vs OS-Genesis with GPT-4o teacher in the real env 6.16 / 9.1; Llama-3-70B 7.02; GPT-4o 13.10 / 11.7). **Ablations:** without step-wise task control 1.72 / 5.2 (task PCA effective dimension at 90% variance falls 153 → 118; the teacher keeps clicking the same 1–2 elements); single-step instead of multi-step simulation 4.06 / 9.1 (bland, biased content); same pipeline run **in the real environment** 4.31 / 4.7–9.1, worse than simulated (search-not-found dead ends, login walls, homogeneous account states); layout-shuffle perturbation hurts -F least (6.28 → 5.54). Grow reaches the full-data result with 66% of trajectories.

**Failure modes & limits.** Simulated states can import irrelevant context (forum page filled with the previous forum's topic) or copy the retrieved reference state while ignoring the current query; no executed verifier, so "completion" is the teacher's opinion; absolute success remains low; SFT only.

**Tips & tricks.**
- Steer exploration with step-wise sub-goal controls proposed from the current state; it measurably raises task diversity.
- Split generation into overview → rich draft → structured render to avoid bland, biased states.
- Use rule-based transitions for deterministic actions; LLM only for the rest.
- Label exploration retrospectively (summarise what was done into an instruction), then filter on stated-vs-taken action mismatch.
- Pick next-round targets from the middle of a learnability proxy (25–75th percentile of teacher-forcing loss) and generate content-only variants of them.
- Replay the most representative old tasks to avoid forgetting.
- Simulated environments can beat real ones when real ones have dead ends (empty searches, login walls); seed real Worlds richly to avoid this.

**Fit to Crucible Part 2.**
- **F3 diversity (§3.5 item 3):** step-wise task control is a concrete form of "grounded, conditioned proposal"; Crucible's analogue is proposing sub-goals from the current World slice during exploratory oracle construction. The PCA effective-dimension number is a cheap extra surface-diversity meter.
- **F9 cheap predictor:** teacher-forcing loss of the student on the oracle trajectory is an inexpensive, rollout-free difficulty signal; its middle band is a candidate pre-filter before paid k = 4 sampling (needs calibration against measured p̂).
- **Task loop (§3.8):** content-only rewrites of in-band targets are the cheapest "variant" operator; the F8 behaviour signature will flag them as near-duplicates unless a knob changes, which is the right guard.
- **F1/F2 realism:** their real-env result is a reminder that sparse seeds create dead ends; G13 interaction completeness plus dense seeding addresses it.
- Avoid: teacher-judged completion as ground truth. Priority P1 (loss-band predictor, task control), P2 otherwise.

## 2606.12191 — Agentic Environment Engineering for Large Language Models: A Survey of Environment Modeling, Synthesis, Evaluation, and Application (Li, Jin, Men et al., 2026; arXiv; corresponding author Jun Zhao)
Read: PARTIAL (§I–III, §V synthesis and quality control in full, §VI-C trajectory synthesis, §VI-D RL, §VII environment evolution, §VIII challenges; §IV domain tables and §VI-A/VI-B memory and workflow evolution skimmed).  Relevance to Part 2: medium as a map, low as a source of methods. It gives a usable vocabulary (symbolic vs neural synthesis; task-driven, real-world-driven, de novo; four quality dimensions; three environment-evolution paradigms) and pointers, but no new experiments and little quantitative comparison.

**Problem.** Environment construction for LLM agents lacks a systematic taxonomy; the survey organises the lifecycle: attributes, domains, synthesis, quality evaluation, agent evolution, environment evolution.

**Method / algorithm (taxonomies, condensed).**
- **Environment attributes (§III):** symbolic vs neural transitions; open vs closed loop; online vs offline; MDP vs POMDP; deterministic vs not; discrete vs continuous actions; uni- vs multimodal; single vs multi-agent.
- **Symbolic synthesis (§V-A), by freedom:** (1) *task-driven*: wrap static data as interactive envs (SWE-Gym; Scale-SWE's three agents: environment builder, unit-test creator, problem-statement writer; SWE-Hub; SCALER for competitive programming with adjustable difficulty; MEnvAgent incremental patching to reuse envs; SWE-smith one image per repo; DockSmith trained Dockerfile repair; CLI-Gym environment inversion: break a working env to create tasks); (2) *real-world-driven*: simplified projections of web, OS, games (AgentSynth's information asymmetry: compose easy subtasks into hard tasks; TaskCraft; VeriEnv clones websites; MCPMark on real MCP servers); (3) *de novo*: generate environments from minimal seeds (AutoForge builds state structures and tool-logic graphs before code; LOGIGEN forward deductive synthesis compiling rules into SQLite-backed envs; AutoEnv decouples underlying logic from surface; ScaleEnv procedural testing + graph expansion; AWM; SWE-Playground; Endless Terminals samples dimensions; gg-bench invents novel games against contamination). More freedom needs more verification logic.
- **Neural synthesis (§V-B):** pixel-level (video world models, NeuralOS), word-level (WebDreamer, Code2World renders GUI as code, MobileDreamer), latent-level (JEPA family); trade fidelity vs abstraction; all suffer long-horizon drift.
- **Quality control (§V-C), four dimensions:** *correctness* (execution, unit tests, gold tool sequences reaching a target state, verifier-vs-ground-truth agreement); *diversity* (embedding dedup, t-SNE dispersion, coverage over tool categories and API combinations, random walks over tool-dependency graphs); *complexity* (structure parameters: steps, tools, entities, DAG depth, plan length; or calibration by strong-model/human performance, e.g. gg-bench drops games with low win-rate disparity); *fidelity* (rule-intent consistency and expert review for symbolic; FID/FVD, physics checks, a "Web Turing Score" where an LLM tries to tell real from synthetic). Authors' verdict: correctness is mature; diversity, complexity and fidelity evaluation are preliminary.
- **Trajectory synthesis for SFT (§VI-C):** task synthesis by resource transformation, *reverse synthesis* (explore then label: OS-Genesis, HATS hardness-driven exploration) or *structure-based synthesis* (graphs, trees, schemas: ToolACE, DeepDive random walks on KGs then cue obfuscation, WebShaper set operations); trajectory generation by augmentation, sequential interaction, tree search, or model simulation; refinement by filtering, correction (GUI-Reflection fixes the first wrong action; SynthAgent remove/reorder/drop/keep edits) and iterative loops (AgentFrontier targets the zone of proximal development).
- **RL design (§VI-D):** reward shaping families (cost penalties for redundant tool calls, decoupled normalisation of accuracy/format/length rewards, conditional release of final-answer rewards), credit assignment (GiGPO anchor-state grouping), sampling (keep high-variance groups; replicate non-zero-variance samples), alternating RL and SFT against entropy collapse.
- **Environment evolution (§VII):** *neural-driven* (self-play: Absolute Zero, Self-Challenging, SSR; world models learned or prompted); *difficulty-driven* with explicit signals (RLVE raises difficulty past a success threshold; SCALER keeps a target success band with active curation; GenEnv's curriculum reward targets a success band; EnvGen targets weak skills; PAIRED/ACCEL regret; DataEnvGym student-error feedback; CuES curiosity) or implicit mechanisms (POET minimal-criterion mutation; DreamGym prioritises high reward-entropy tasks; AgentGen bidirectional easier/harder evolution; SEC adaptive category selection); *scaling-driven* (scenario-level: EnvScaler skeleton → scenarios; AgentScaler read/write DB abstraction; InfiniteWeb; AutoWebWorld websites as finite-state machines with enumerable verified trajectories; environment-level: ARE, AutoEnv factorised transition/observation/reward distributions).
- **Open problems (§VIII):** Environment-as-a-Service interfaces; dynamic, long-horizon, open-ended, multimodal properties; multi-agent environments; neural-symbolic hybrids; sim-to-real gaps on correctness, difficulty, diversity, fidelity; co-evolution; on-policy distillation for multi-turn agents; a "science of environment engineering": environment scaling laws, environment learnability, and the environment-to-capability mapping.

**Verification / grading.** None (survey).

**Scale & cost.** Not applicable; tables list representative works with base models and data sources (not reproduced here).

**Key results.** None of its own. Qualitative takeaways: environments should balance realism, diversity, controllability and verifiability; correctness tooling is mature, distribution-level quality metrics are not; symbolic-neural hybrids are the open middle ground.

**Failure modes & limits.** Breadth over depth; method descriptions are one-liners and sometimes imprecise; no cross-method numbers; no cost analysis; little on intent alignment or verifier attack-testing.

**Tips & tricks.**
- Classify each new App by synthesis freedom; demand verification logic proportional to it.
- Report all four quality dimensions per environment batch, not only correctness.
- Model websites or workflows as finite-state machines when possible so trajectories can be enumerated and verified.
- Abstract tools as read/write operations over a database to enumerate the tool × intent × path space.
- Calibrate complexity by discriminative power (win-rate or pass-rate disparity across models), not length alone.
- Use an LLM "real vs synthetic" discriminator as a cheap fidelity probe (Web Turing Score).

**Fit to Crucible Part 2.** Useful as a checklist against §3: Crucible's design already covers correctness (G1–G14), diversity (coverage grid, signatures), complexity (knobs, per-student bands) and adds what the survey says is missing (intent alignment, verifier attack tests, per-student difficulty). Two gaps it surfaces: (1) a **fidelity meter** for synthetic Worlds (a real-vs-synthetic discriminator or W2R replay where a real reference exists) should join §3.5's meters; (2) **environment learnability** and the environment-to-skill map are exactly what F12's "value per generator per dollar" and the intent audit should measure, so Crucible's measurements could be reported as an answer to the survey's open question. Pointers worth chasing for F1/F4: AutoWebWorld (FSM websites), LOGIGEN (forward deductive synthesis into SQLite), AgentScaler (read/write DB abstraction), SCALER/GenEnv (band-targeting controllers). Priority P2.

