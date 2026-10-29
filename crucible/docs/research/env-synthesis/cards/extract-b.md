# Extract B: groups 03 and 04, consolidated by mechanism

**Inputs.** `group-03.md` (29 papers, all FULL) and `group-04.md` (29 papers: 27 FULL; PARTIAL for 2605.26494 MiniMax-M2
and 2512.02556 DeepSeek-V3.2, where every data, synthesis and RL section was read and the architecture was skimmed). No
card is ABSTRACT-ONLY. Stage and gate names follow `framework-plan.md` §3 (F0–F12, G1–G14, the four loops of §3.8).

**How to read it.** Each bucket lists named mechanisms. Each mechanism gives the steps, the thresholds and numbers with
their settings, the evidence with arXiv ids, its failure modes, and the adoption for Crucible (stage, interface,
priority). Priorities are the card authors' (P0 cheap and evidenced, P1 core, P2 later). Smaller tips close each bucket.

**Contamination warning that applies to every bucket.** Four papers fed benchmark material into synthesis: TRACE
(2604.05336, τ² discovery rollouts on the evaluated tasks), NoisyAgent (2605.27209, trains on τ² and VitaBench domains it
evaluates), Magnet (2503.07826, BFCL-v3 implementations in its pool) and Learn-by-interact (2501.10893, SWE-bench Verified
files used as "documents"). SearchEyes (2607.05943) evaluates on a benchmark built by its own pipeline. For Crucible this
hardens the sealed-split rule: nothing from benchmark tasks may feed F0–F3, the skill selector or the failure clusterer.

---

## B1. Domain spec and World/environment synthesis

### B1.1 Executed survey before a coverage plan (Daedalus Surveyor)
- **Steps** (2610.08048). A Surveyor explores the live environment read-only and may write its plan only after it has
  answered five probes per area from calls it actually ran: record contents and cross-area keys; what lies past page 1;
  behaviour on wrong and stale ids; what a write does to related records, read back; what it refuses and with which
  error. It then writes (a) a coverage goal: one chosen axis (app, area or task shape), a weighted distribution justified
  by real use rather than endpoint count, generator guidance and pitfalls; (b) a closed **tag manifest**, each tag with a
  definition and a `target_share` in (0,1]. A tag must be able to be false, so anything with a target above about 0.5 is
  a baseline requirement, not a coverage tag.
- **Evidence.** Adding the survey cut total generation cost 48% and raised session acceptance from 86.7% to 97.8% on
  AppWorld (cumulative ablation).
- **Adoption.** F0: the Surveyor is a cheap bootstrap of `DomainSpec.axes` for a new World. Interface
  `SurveyReport{axis, distribution, tags:[{tag, definition, target_share}]}` → DomainSpec draft, then lead review. The
  live UNDER/OK/OVER tag tally becomes an F3 quota signal next to MAP-Elites cell fill. P0-cheap for `world_synth`.

### B1.2 Taxonomy-first skill trees, and how to normalise them
- **GLAN** (2402.13064). An LLM lists fields → sub-fields → disciplines from several prompts; humans vote keep or remove
  only at the top levels (removing a node removes its descendants; 126 disciplines kept). Each discipline is decomposed
  into subjects, then syllabi of 10–30 sessions with about 5 key concepts each. Instructions sample 1 session + 1–5
  concepts (basic) or 2 sessions + 2–5 concepts across both (harder, combinatorially more). Content generation and format
  conversion are separate calls. Weakness: the taxonomy weights by structure, not usage, and gains were uneven (strong on
  STEM, weak on some humanities disciplines).
- **InsTag normalisation** (2308.07074). Turn open-set LLM labels into a stable ontology in four passes: frequency floor
  (drop tags seen < 20 times; 100K+ raw → 8,541), lexical canonicalisation (lowercase, stemming → 7,157), embedding
  clustering (PhraseBERT + DBSCAN → 6,587; the stated threshold 0.05 is under-specified), and association-rule merging of
  always-co-occurring tags (FP-Growth, support 40, confidence 99%). Audit on two axes: per-item precision (96.1%) and
  per-label consistency across items (86.6%), with planted wrong labels to confirm the auditor catches errors.
- **Nemotron-Terminal prompt modules** (2602.21193). One shared system template plus a per-domain prompt module, a
  curated primitive-skill list per domain across 6 dimensions, and the domain's pre-built Dockerfile given to the
  generator. 9 shared domain images instead of per-task Dockerfiles: no Dockerfile repair loop.
- **Task regimes as an axis** (ACE survey, 2608.27260): clarify, refuse or infeasible, partial completion and recovery
  are separate regimes. **Capability splits** from Gaia2 (2509.17158): search, execution, adaptability, time,
  ambiguity, noise, collaboration.
- **Adoption.** F0 builds the skill tree GLAN-style (LLM hierarchy, human prunes the top), runs InsTag's four passes on
  any swarm- or trace-derived labels, and gates the taxonomy with the precision/consistency audit plus planted wrong
  labels. `SkillLeaf` gains `aliases`, `merged_from`, `support`. Add a usage-weighting step (Surveyor, trace clusters).
  Task regimes and Gaia2 splits go into `DomainSpec.axes`. The template + module prompt structure is the F3 existence
  proof (P0). Others P1.

### B1.3 Contract-first stateful App synthesis with error-routed repair
- **WEFT** (2609.36887), steps. (1) MCP tool definitions from public docs and business requirements. (2) An agent infers
  shared entities, read/write dependencies and state transitions across the whole tool set, then instantiates a shared DB
  schema. (3) It writes a per-tool **contract**: argument → state mapping, reads, writes, returns. (4) It generates a
  FastAPI backend; workspace state binds to isolated file runtimes. (5) Validation: interface names and argument schemas;
  generated normal and boundary calls; queries checked against state, updates against the DB/workspace diff and a
  read-back. (6) **Repair is routed by error class**: structural → schema, mapping → contract, runtime or state access →
  backend.
- **DeepSeek-V3.2** (2512.02556, PARTIAL) runs the same order from a task category: retrieve or generate data into a
  sandbox DB → synthesise task-specific tools as functions over the DB → task + solution + verifier (B2.2).
- **RandomWorld** (2506.11045) shows the determinism trick for generated tools: outputs are **memoised per input**, so a
  tool is deterministic within an environment, and invalid input types return an error. GPT-4o plausibility rating ≥ 4
  kept about 11% of sampled tool signatures.
- **Failure modes.** WEFT's attribution precision is not measured; checks written against one reference execution may
  be too strict. DeepSeek's DB holds internet data (staleness, contamination).
- **Adoption.** F1: contract-first App synthesis and error-class repair routing; WEFT's normal and boundary calls are G13.
  Memoise any generated or stubbed tool per input so G2 holds. P0 for routing, P1 otherwise.

### B1.4 Read/write tool tags, events, clock and notifications (ARE)
- **Abstractions** (2509.17158). Apps are stateful tool collections; Python methods are auto-exposed; decorators mark
  `read` vs `write`; tools are role-scoped (agent, user, env). Events: agent, user and env tool calls; **conditional**
  events that poll a predicate; **validation** events that fail the run on timeout; **oracle** events (ground-truth
  actions). Events form a DAG with absolute or relative timing; all are logged. Notifications pass a policy whitelist at
  low, medium or high verbosity, an observability knob. A system app offers get_current_time, wait and
  wait_for_next_notification; simulated time jumps event to event while the agent waits. Optional natural-language
  hints (step-by-step solution) for QA and for RL when tasks are too hard.
- **Numbers.** Mobile environment: 12 apps, 101 tools. Gaia2: 800 scenarios (160 per split) plus Agent2Agent and Noise
  augmentations of Gaia2-mini, 1,120 total. Time shows **inverse scaling with reasoning**: GPT-5 (high) 0% with real
  generation time vs 34.4% in instant mode; Claude 4 Sonnet 8.2% → 26.7%. Noise split: most models < 20.
- **Augmentation by invariance.** Agent2Agent and Noise splits reuse the oracle solutions, because replacing apps by
  app-agents or adding noise (tool signature changes, failure probability 0.1, 10 random env events per minute) does not
  change the correct writes.
- **Authors' advice.** Simple tasks in complex environments, not complex tasks in simple ones.
- **Adoption.** F1/F2 reference model for §2(b): read/write tags, role-scoped tools, conditional and validation events,
  notification verbosity as the observability knob. Account for generation time separately from server errors. P0 for
  read/write tags, P1 for events and notifications.

### B1.5 Persona-rooted, dependency-ordered universe seeding
- **ARE universes.** Llama 3.3 70B generates content rooted in a PersonaHub persona → locations → apps in app-dependency
  order (Contacts first; Messages, Emails, Calendar and Cabs query existing state so referenced people exist), first as
  unstructured text, then by schema-constrained decoding. About 400K raw and 800K structured tokens per universe, 10
  universes. **Known gaps:** temporal consistency across apps, relationship consistency, cross-modal references.
- **Persona Hub** (2406.20094). Text-to-persona ("who would read, write, like or dislike this?") over web text, then
  persona-to-persona expansion ("who is close to this persona?") for six iterations to reach low-visibility people. Dedup
  by MinHash on 1-grams (128-signature, threshold 0.9), then embedding cosine > 0.9 (tighter, e.g. 0.5, for fewer but more
  diverse personas). 1,015,863,523 personas. Persona-driven "what tools would this user need" is a cheap App-surface
  brainstorm.
- **WEFT progressive state grounding** in three rounds: target entities and minimal executable state; then cross-tool
  relations and constraints; then history and distractors. Completion conditions are preserved, schema and reachability
  are re-checked, and the result is snapshotted as a resettable s0.
- **Adoption.** F2: persona → relationships → contacts/colleagues, seeded per universe in dependency order; enterprise
  variant derives personas from redacted org documents. Add ARE's three coherence gaps as G1 extensions. WEFT's three
  rounds map onto `SeedBundle` with `distractor_density` as the last round. P1.

### B1.6 State-scoped isolation, exact reset and soak tests
- **CUA-Sandbox** (2609.32750). Mutable state, not the runtime, is the unit of independence. A task resource contract
  classes each resource as **authoritative** (DB, user files; private copy-on-write delta over an immutable base),
  **derived** (caches, indexes; private, rebuildable) or **ephemeral** (IPC, displays; recreated). A resource without a
  fine-grained adapter goes private, never shared. Each action carries an env id and a generation lease; reset, snapshot,
  clone and fork freeze the route, drain in-flight work, stage a successor and atomically publish generation g+1; stale
  requests are rejected. The evaluator reads the same generation as the last action.
- **Numbers.** At 8 concurrent envs on one host: throughput ×2.22 (WebArena), ×6.20 (VisualWebArena), ×3.08 (OSWorld);
  WebArena memory 2,103 → 464 MiB; incremental storage 6,757 → 13 MiB; CoW create on a 3.6 GB DB 5.48 → 0.07 s.
  Transactional publication: 0/100 mixed-generation events vs 100/100 with direct publication. Success ≥ Docker in all
  12 model–benchmark pairs; matched SFT→RL of Qwen3.5-4B gives the same gain (OSWorld +3.88 vs +3.87).
- **Card critique.** The uplift over Docker in every pair suggests the Docker baseline had flake scored as failure:
  environment flake silently depresses measured success.
- **MegaMCP** (WEFT): shared MCP processes outside sandboxes, a private CoW DB and workspace per (rollout, MCP),
  content-addressed registry, async debounced snapshots, generation fencing. Upload 773.5 → 34.8 MiB at 1,000 tasks,
  memory −77.6% to −95.7%, cold TTFT 10.5 → 4.8 s. **The verifier runs in the serving layer** on start DB, end DB and the
  full call ledger, invisible to the agent.
- **AgentGym-RL** (2509.08755): a full-reset endpoint per episode, and memory-leak fixes in long-running env servers
  (TextCraft, SciWorld). It excluded WebArena write tasks to save reset cost.
- **Adoption.** F2/World: the three resource classes and generation fencing are the shape for `StateAdapter`
  snapshot/restore and "one World image per domain, many tasks per snapshot" (§3.10 lever 6). G2 and G10 depend on exact
  reset. Log env errors separately (`env_flake` in F12). Add a long-run soak test (thousands of resets) per App to the
  World gates. Never drop write tasks to save reset cost. P2 (P1 if CUA or web tiers train).

### B1.7 Controlled perturbations and noise as World operators
- **Qwen-AgentWorld** (2606.24597). A learned simulator helps RL only with a detailed initial state and per-turn control
  instructions. On MCP, **uncontrolled sim RL ≈ no gain** (Toolathlon 32.4 → 31.5), while **controlled** perturbations
  gave +3.7 Toolathlon and +12.3 MCPMark. Perturbation classes: pagination, withheld intermediates, injected errors
  (an error must be emitted, not a fake success), batch partial failure, required final state.
- **NoisyAgent** (2605.27209). User-side noise, applied by perturbing the simulated user's pattern while keeping the
  objective: ambiguous, inconsistent (needs change), redundant, out-of-scope. Tool-side noise on a random subset of
  returns: failure (e.g. 429), truncation, misleading or inconsistent field, redundant detail. Under noise, tool calls
  fall 13.9 → 11.4 while output tokens double; in 78% of the 23 newly solved tasks the base model had gathered all the
  information but never executed the final write. Training curriculum in B6.6.
- **Failure modes.** "Misleading" noise can make a task unsolvable or reward a wrong action; NoisyAgent asserts
  solvability but shows no check. Rewards were rubric judges throughout.
- **Adoption.** F1/§3.6: implement perturbations as World-side operators on code Apps and `ActorSpec`s (never
  LLM-generated returns), each tagged *solvability-preserving* or not. A misleading-field operator must be paired with a
  second source of truth in the World so G8 holds. Each perturbation needs a G13 check. Run G2 and G10 with noise seeds
  fixed, so injected failures are reproducible and not counted as flake. P1.

### B1.8 Learned simulators and self-rehearsed environments: a hazard with narrow uses
- **EnvACE** (2608.06197): one policy plays ACT and REHEARSE, so it writes its own tool responses, and reward comes from
  a verifier or a checklist judge (Qwen3-30B-A3B in training). Nothing stops the rehearsal role from writing
  observations that make the checklist pass; rehearsal fidelity is never measured. Gains over env-scaling baselines are
  ≤ 1 pp overall (EnvACE-8B 32.91 vs AWM-14B 32.54). Test-time private rehearsal (N=2) lifted overall 36.7 → 40.9.
- **Qwen-AgentWorld fictional worlds.** 1k search worlds, each backed by a 300–500-row fictional relational DB; ground
  truth by SQL; queries reverse-generated; snippets "tease, never answer". WideSearch F1-item 34.0 → 50.3 with sim RL vs
  45.6 with real-search RL; the sim-trained agent raised page-extract calls 2.5 → 4.0 while the real-RL agent dropped to
  1.5. The simulation instruction carries the success state (a leakage risk the authors argue away with OOD evals).
- **Text world-model survey** (2606.09032): prefer code-as-world-model where dynamics are specifiable; back any
  simulator with structured state; measure a simulated component by its **consistency ratio** (share of simulated
  successes that also succeed in the real component), not single-step accuracy.
- **Adoption.** Avoid policy-written observations and LLM world models as reward (violates "execution decides"). Use a
  schema-checking LLM stub only to draft blind tool tests for unbuilt tools. Fictional DB + SQL is a clean F5 answer-check
  construction (P1). The consistency ratio is the check for any LLM-simulated colleague or user (P2).

### B1.9 Simulated people must not be cooperative
- **Evidence cited by the survey** (2606.09032, citations not verified by the card): prompted assistant LLMs as users
  overestimate agent ability; a user model trained on 384k real dialogues lowers agent success sharply; 31 simulators vs
  451 participants show an "easy mode" of over-politeness; LLMs fail to predict latent cascading side effects in
  enterprise systems.
- **WEFT views.** One objective, state and condition set under an Agentic view (full brief) or a SimUser view
  (persona-sampled user discloses requests in dependency order, one atomic task per turn) × vague, partial or full
  disclosure × ReAct, OpenClaw or Hermes harnesses. Mixed views beat either alone (+3.18 avg over Agentic); adding two
  harnesses gave +9.71 mean (+13.58 Toolathlon) at fixed tasks.
- **Verbalized Sampling for replies** (2510.01171): probability-weighted choice among k = 5 candidate turns avoids
  identical persona replies across episodes. **Magpie** (2406.08464) template-prefix sampling is a near-free sampler for
  a user's opening request (about 3% of its compute), but its prior collapses to information seeking (> 50% of Pro).
- **Adoption.** `ActorSpec` profiles for vague, contradicting, impatient and wrong users; validate user simulators against
  a small human-labelled anchor (part of §3.8). Disclosure level and view are a cheap knob family (observability, hints).
  P1 for views, P2 for the rest.

### B1.10 Closed, deterministic retrieval Worlds with checkable milestones
- **SearchEyes** (2607.05943): Wikidata5M ∩ Wiki6M ∩ Wikipedia images → about 1.2M entities (340K visual), 5.8M triples,
  822 predicates in 4 domains. Hubs (degree > 500) and 47 meta predicates removed. One summary document per entity, so
  the environment logs exactly which entity each step retrieved (an id ↔ document bijection). BM25 + dense search fused by
  RRF, lookup, visual search, sandboxed Python. Deterministic.
- **Adoption.** Any retrieval-shaped World should keep an id ↔ record bijection, so milestones (gold intermediate
  entities) are checkable (B3.2, B6). P1.

### B1.11 Minor tips (B1)
- Make the observation/context rule a versioned property of the World or `TaskContract`
  (`context_rule: append|window(k)|summary(model)`): in Agent-R1 (2511.14460) a sliding window beat append-only, which
  beat an LLM summary (one task, GSM8K). It changes difficulty, so it belongs in the difficulty key.
- Snapshot per turn so retries and GRPO groups start from identical state (WEFT).
- Capacity surplus Σc_m ≥ ⌈ρk⌉ with ρ > 1 before composing a task across Apps, and an LLM check that the App set supports a
  coherent business objective (WEFT).
- Clean traces for world models by skipping retry cycles without breaking the state chain and dropping no-change GUI
  turns (Qwen-AgentWorld); a four-statistic turn classifier (overlap, novelty, Jaccard, length ratio) drops echo and
  boilerplate turns before failure clustering.
- RL gains are largest where feedback is causal and explicit (rule-based simulators), smaller on WebArena and search
  (AgentGym-RL).
- Filter GUI env failures (CAPTCHA, HTTP errors) and tools outside the declared action space out of traces at source
  (Qwen-AgentWorld).
- Hand-designed enrichment per environment does not scale (FEE); the F4 oracle and milestones Φ supply it automatically.

---
## B2. Task proposal and sources

### B2.1 Solution-first typed call DAG with dead-step trimming (RandomWorld)
- **Steps** (2506.11045). (1) A type system of 73 fine-grained base types in a subtype hierarchy, each with a generator
  and a recognizer; constructors list, dict, union. (2) Tools sample input and output types; an LLM names them and rates
  plausibility (keep ≥ 4). (3) **Skeleton first**: sample user-input types and a length ℓ ∈ [ℓ_min, ℓ_max]; repeatedly add
  a tool whose inputs are subtypes of available variables; the last output is the goal. (4) **Trim** every call whose
  outputs do not feed the goal, directly or indirectly, and resample until all ℓ calls are used; dedup skeletons.
  (5) The instruction writer sees the skeleton with **intermediate values replaced by variables** (no output leakage).
  (6) **Informativeness check**: a privileged solver (only the needed tools, in gold order, with type signatures and
  examples, no login) must reach the goal, else discard. (7) Tool inventory = needed tools + distractors at r_dist = 1.0;
  reward is exact match of the returned value to the goal.
- **Numbers.** 12,000 environments from 556 tools (550 generated + 6 hand-made), ℓ 2–8, about $150 total OpenAI cost;
  about 60% environment yield after dedup. Llama GRPO doubles the base on its own test (0.483 → 0.721) and NESTFUL F1F
  0.53 → 0.89. In-distribution over-fitting (Llama-SFT 0.855) correlated with worse transfer; the authors suggest
  stopping near 0.72 in-distribution and raising ℓ instead. In distribution, task count mattered more than tool count;
  fewer tools helped ToolQA-Hard (+15%).
- **Failure modes.** Random-valued tools make tasks semantically empty; the instruction check may filter the hardest
  tasks; tool names and descriptions can leak the chain (App. C).
- **Adoption.** F4 oracle lint: every oracle step must contribute to the goal end state or answer (also an F8 normaliser
  before signatures). The privileged check is G8 made executable (B4.1). Instruction writer sees variables, not values
  (G11). ℓ range and distractor ratio are knobs. P1.

### B2.2 Data → tools → task + solution + verifier, tool-only oracle, iterative hardening (DeepSeek-V3.2)
- **Steps** (2512.02556, PARTIAL). Given a category: (1) put retrieved or generated data in a sandbox DB; (2) write
  task-specific tools over it; (3) propose a **simple** task with a Python solution function and verification function;
  **the solution may only call tools or do logic, never other functions or the DB**; the agent edits solution or
  verifier until the output passes; (4) iteratively increase difficulty, updating solution and verifier each time and
  adding tools when needed; (5) keep only instances with **pass@100 > 0** under the policy. Design target: hard to solve,
  easy to verify (e.g. constraint-laden trip plans).
- **Numbers.** 1,827 environments and 4,417 general-agent tasks (of about 85K agent prompts across code 24,667, search
  50,275, code interpreter 5,908). Pass@1 on 50 sampled synthetic tasks: DeepSeek-V3.2-Exp 12%, Sonnet-4.5 34%, Gemini-3.0
  Pro 51%, GPT-5-Thinking 62% (pass@4 26/62/74/82). **RL only on synthetic general-agent tasks** gave substantial gains on
  τ²-Bench, MCP-Mark and MCP-Universe; RL restricted to code + search did not.
- **Failure modes (card).** One agent writes solution and verifier and edits either until they agree, so a verifier can
  be weakened to match a wrong solution, with no independent negatives. Pass@100 with the policy admits lucky passes and
  trial-and-error tasks. Hardening is judged by the generator, not a student band. No diversity control across envs.
- **Adoption.** The `world_synth` pipeline template (P0): G7 tool-only, harder-with-updated-oracle-and-verifier. Fixes:
  the verifier writer is a different family from the solution writer; every hardening round re-runs F5 negatives;
  replace pass@100 by solvability by construction plus F9's band; version the tools added during hardening in
  `AppBundle`, so a World change re-gates the family.

### B2.3 Certified atomic tasks composed into a grounded task graph (WEFT)
- **Steps** (2609.36887). (1) Per MCP, a task agent writes atomic tasks (one intent, bounded calls) plus E2E tests; keep
  one only if execution from an independent initial state satisfies its completion conditions on returns and persistent
  state. (2) Drop a task whose ordered tool-name chain is a non-contiguous subsequence of another's (B5.3); iterate with
  failures, redundancy and uncovered tools fed back until tool coverage ≥ η or a round budget R. (3) Five difficulty
  profiles (Simple … Expert) set ranges of atomic-task count, MCP count and domain breadth plus a corpus share;
  largest-remainder allocation turns shares into quotas; the sampler favours under-used domains and MCPs and ranks combos
  by shared entities, complementary ops and low redundancy. (4) A composition agent links atomic tasks into a DAG (nodes:
  intent, inputs, outcome; edges: information or state dependencies; entity bindings across MCPs); deterministic checks:
  nodes use selected tasks, bindings conflict-free, every input comes from the initial state or an upstream outcome.
  (5) Progressive grounding of state (B1.5).
- **Numbers.** 8,172 MCPs, 64,755 tools, 41,695 certified atomic tasks, 11,884 composed tasks (median 20 atomic turns;
  profile mix 27/36/15/10/12%; more than half multi-domain).
- **Adoption.** F3/F4 for composed `world_synth` tasks; the DAG input-provenance check is a cheap G8 precursor. P1.

### B2.4 Explorer doctrine: under-specify, harden by scope, write end-state deltas (Daedalus)
- **Steps** (2610.08048). An Explorer session (5 concurrent, 40-turn cap) sees the survey, the live tag tally, the
  accepted tasks ("do something different") and a guideline bank. It explores, writes one task and completes it itself
  (feasibility). Output: a short natural request; the minimal tool-call solution; success conditions **on the end state,
  as changes from the start, path-agnostic**. Doctrine: delete every id, amount or date the solver could find; get
  hardness from **scope** (superlatives, universals, relative time windows, pagination, none or many matches, policy
  limits), not from more steps or more apps.
- **Evidence.** A single Explorer trajectory as memory *hurt* (35.7 < 44.3 no-memory) until solver traces were added
  (+15.8). Removing τ² tasks whose reference exchanged more than one item per call exposed a real verifier bug (the
  environment grades argument order), found by construction.
- **Adoption.** The F3 proposer prompt core, with F4 executed oracle and F5 code checks instead of Daedalus's LLM judge.
  P1.

### B2.5 Structure-first generators over a graph
- **Magnet** (2503.07826). Build a local dependency graph (an LLM marks an edge when one function's output is a
  precondition for, equals, or is part of another's input; 30 candidates per function from the same category); take a
  random walk of S = 7 steps as the signature path; apply node operations: **Merge** consecutive turns into a multi-call
  turn (p = 0.3); **Insert** a nested function the query does not name, or insert it later for a long dependency;
  **Split** to make an empty turn labelled missing-function or missing-parameter, where the right behaviour is to ask.
  Back-translate each turn to a query that refers to prior outputs indirectly and never names functions, then
  forth-translate to an executable call and run it. Ablation: the graph alone ~+8 multi-turn; Merge and Insert add most;
  Split helps missing-function/parameter. Magnet-14B-mDPO BFCL-v3 68.01 overall, multi-turn +30 over base, above its
  teacher.
- **SearchEyes PKC paths** (2607.05943). Random walk of K ∈ {3,4,5} hops (P 0.3/0.5/0.2) from a visual anchor with
  ≥ 2 out-edges. Constraints: perception/knowledge hop alternation with ≥ 2 P-hops; at least one one-to-few predicate plus
  a **disambiguating constraint edge** no sibling satisfies (treewidth 2); adjacent hops in different domains and ≥ 3
  domains per path; hub exclusion, predicate blacklist, dedup by (anchor, answer) and by entity sequence. Question
  generation conceals the anchor (image only), never names predicates (one domain hint from a random middle hop) or path
  entities. 22K questions.
- **WebShaper** (2507.15061): tasks as graphs of (variable, relation, constant) triplets, expanded layer-wise (B6.4).
- **WebSailor** (2507.02592): seed from a rare Wikidata entity, expand non-linearly (sometimes return to an earlier node)
  until an edge budget, sample subgraphs with diverse topologies, then write the QA.
- **Adoption.** `AppBundle.tool_graph` + Magnet's operations give generic knobs: `linkage` (insert for long dependency),
  `hidden_steps` (implicit nested call), `missing_info` (split → infeasible or clarify task type). For multi-entity World
  tasks, walk the World's relation graph under SearchEyes-style constraints. P1.

### B2.6 Traces to tasks: relabel, rebuild, re-solve
- **OS-Genesis** (2412.19723): rule-based traversal of UI elements gives ⟨s_pre, a, s_post⟩ triples; an annotator maps
  each to a low-level step, then lifts it to a plausible high-level task; the task is **re-executed forward** to get the
  trajectory; a GPT-4o trajectory reward model (1–5) weights samples instead of discarding partial ones. AndroidWorld SR
  4.02 → 15.18 (InternVL2-4B), 4.46 → 16.96 (8B); keeps > 80% of the value of human trajectories. Human data had varied
  instructions but narrow action sequences.
- **Learn-by-interact** (2501.10893): self-instruct 3 tasks per document, roll out, then relabel **every contiguous
  sub-trajectory** T[i:j] (quadratic), as a summary or as an abstraction of purpose; replace a derailed run's goal with its
  actual purpose; drop consecutive duplicate (a, o) pairs; a two-model committee must all judge each pair coherent. Funnel
  example SWE-bench: 6,464 docs → 4,568 trajectories → 41,237 pairs → 10,232 kept (filters keep 25–60%). Backward
  construction worth up to +14.0 in training; short sub-trajectories (< 5 steps) helped most; mixing lengths best.
- **AgentBrew** (2609.05837): **strip the chain of thought** (it was conditioned on the original, maybe failed, goal);
  extract a structured change summary of successful state-changing operations; minimally edit the original task to what
  was achieved (keep the task type, keep completed subsets, delete rather than invent, drop strong quantifiers such as
  "all", "each", exact counts, unless supported). Relabelling on Notion: Acc 7.1 → 14.2. Proposal grounding: exactly one
  read-only call to find real entities, then a request that names only verified ids.
- **MiniMax-M2** (2605.26494, PARTIAL): evidence-driven synthesis (run real tools first, then derive tasks strictly
  entailed by the traces) and the **workbook walk** (atomic spreadsheet operations from a seed workbook, intermediate
  states recycled as seeds, answer from the trajectory, question from the answer; acceptance by recalculated cell match).
- **Qwen-AgentWorld OpenClaw recipe**: a real trace → seed scenario (initial state + query) → vary concrete state,
  rephrase intent, adjust difficulty, compose goals (4k environments).
- **AgentSynth reviser** (2506.14205): when a subtask is partial, rewrite its description to what was achieved.
- **Adoption.** §3.4: each relabel candidate becomes (a) start = World snapshot at step i, (b) goal = state diff at step
  j, (c) re-solved and gated through F4–F6; never the label alone. The intent labeller uses AgentBrew's rules. Workbook
  walk = World random-walk synthesis for `world_synth`. Short sub-goals feed F10's single-subtask probe and ladders.
  Avoid committee-only acceptance and benchmark-derived documents. P1.

### B2.7 Failure-targeted generation from traces (TRACE, WEFT, OMNI-EPIC)
- **TRACE contrastive selector** (2604.05336). (a) An analysis agent writes a canonical dictionary of named capabilities
  from tool calls, results and replies. (b) With the dictionary fixed, label each trajectory × capability NA, PRESENT or
  LACKING. ER⁺(c) = lacking-rate on successes, ER⁻(c) on failures (NA excluded); gap Δ(c) = ER⁻ − ER⁺; coverage Cov(c) =
  share of failures where c is LACKING. **Keep c if Δ ≥ 0.20 and Cov ≥ 0.10**, and only if selected consistently across
  repeated labellings (10 runs; top 3 in all 10, tool-calling precision in 8/10). About 5 min per pass.
- **TRACE env contract** (App. E). Same tool schemas as the target; reset(seed) deterministic, different seeds
  meaningfully different (DB contents, requests, constraint combinations, distractor counts); one skill per env (≤ 2);
  ground truth per seed; reward from final DB state (hash) and correct communicated result; **target base success
  30–60%** (> 80% too easy, < 15% too hard); multi-level reward α·Π components + (1−α)·Σ w·components; **builder self-check
  with 100+ rollouts: mean 0.3–0.6, std > 0.2, > 60% of groups with variance.**
- **TRACE results** (Qwen3-30B-A3B, τ² overall): base 32.9; GRPO on target 37.8; AWM 38.4; GEPA prompting of the same
  capabilities 39.6; one capability adapter 40.3; TRACE 48.2. SWE-bench Verified 26.0 → 41.0 (SWE-RL 32.6). Qwen3.6-27B τ²
  50.0 → 59.1, SWE-V 68.0 → 73.2. Equal rollouts: +10.4 over GRPO, which was unstable (35.4 at 3,840 rollouts).
  Composition: token-level MoE gate over frozen per-skill LoRAs 48.2 ≫ multi-capability GRPO 40.9, SFT 37.8, on-policy
  distillation 37.8, TIES merge 39.6.
- **Caveats.** τ² discovery used the evaluated tasks; capability labels are LLM-made with no human audit; "success depends
  primarily on c" is asserted, not tested by ablation. Single greedy run in main tables (3 seeds at T = 0.7 in App. G).
- **WEFT attribution routing**: an attribution agent reads graphs, traces, checkpoint outcomes and state evidence;
  cross-view and cross-harness patterns localise faults; env → backend, task/state → graph and s0, eval → verifier; policy
  failures leave the task unchanged.
- **OMNI-EPIC failure context** (2405.15568): see B2.9.
- **Adoption.** F12 → F3 (P0 for the selector): compute Δ(c) and Cov(c) per family over student rollouts using
  `failure_skill` labels, keep Δ ≥ 0.2 and Cov ≥ 0.1, stable over ≥ 3 relabel runs by a different-family labeller, and emit
  them in `FeedbackSignal.failure_clusters`. F3 then proposes "variant of family F targeting skill c" under the TRACE
  contract mapped to gates: G2 (seeded determinism), F4 (ground truth per seed), F10 single-subtask probe (single-skill
  intent), and a frontier pre-check of the reward distribution before F9. Interface
  `FailureAttribution{side: policy|world|task|verifier, evidence_refs, target_component}` (P0 for WEFT routing). GEPA vs
  training is an "intent needs training" control.

### B2.8 Realistic fault origin and self-generated bugs (BugPilot, SSR)
- **BugPilot FeatAdd** (2510.19898): an agent invents and implements a feature while preserving behaviour; if ≥ 1
  existing test fails, the repo state is a bug; otherwise it keeps changing code until tests fail. Patches: 4,376 diff
  tokens, 4.2 files, 415.9 net lines (SWE-smith 598 tokens, 1.2 files, −3.2 lines). Claude Sonnet 4 solve rate: R2E-Gym
  63.5%, SWE-smith 65.9%, BugInstruct 54.6%, **FeatAdd 41.4%** (GPT-5: 68.7/77.5/67.8/53.4). Category histogram most even.
  Qwen3-32B pass@1 over 3 seeds: base mix 49.87; + BugInstruct 49.87 (no gain); **+ FeatAdd 51.93 SFT, 52.40 RL (25
  steps)**; all data 54.6, pass@3 67.4. RL gave no gain over SFT on frontier and multi-file subsets.
- **BugPilot shortcut risk (card).** Huge patches let a solver "fix" by reverting the whole feature; bug reports written
  from failing-test output leak the test.
- **SSR bug artifacts** (2512.18552): `test_script.sh`, `test_files.txt` (restored before grading), `test_parser.py`,
  `bug_inject.diff`, `test_weaken.diff` whose reversal is the solver's spec. Injection by (a) direct prompting, (b)
  **removing code hunks or files** with compatibility fixes, (c) **reverting historical changes**. Direct prompting
  collapsed to trivial one-line edits; removal + history reversion was best. **Higher-order bugs**: a failed solver
  state becomes a new start, capped at second order. **Natural-language issue generation failed** (copied test patches,
  incoherent, identical patterns).
- **MiniMax SWE routing**: tag mined PRs by type (bug fix, feature, perf, refactor, test) because each needs its own
  reward (bug fix: F2P + P2P with golden patch passing; feature: new tests; perf: P2P tests with a stable significant
  speed difference); **SWE-Test inversion** (write a test that fails before and passes after the patch).
- **Adoption.** Debug lane: a FeatAdd generator (different-family swarm model on a real repo at A′; first test break is
  the task; the feature diff is the oracle's inverse) with a **revert-shortcut negative** (B3.8). Type routing and test
  inversion give separate families. For `world_synth`: a plausible but mistaken operator workflow leaves the World
  inconsistent; the corrupted state is the start, the repaired state the goal; "remove or corrupt a record the oracle
  depends on" is SSR's removal analogue. Task text comes from F3 with grounding, not from self-play. P1.

### B2.9 Proposer conditioning that measurably helps
- **OMNI-EPIC** (2405.15568): the generator (Claude 3 Opus, T = 0) sees the **5 most similar learned and 5 most similar
  failed** tasks (embedding cosine over description + code) and is asked for a novel, specific next task with an explicit
  success condition. Ablation, tasks learned in a short run: 16 full; **5 with uniform rather than nearest examples; 7
  without failed examples** (the generator regenerates already-failed tasks); **0 without the learning-progress notion in
  the prompt**. Warm-starting from the nearest learned policy: 63.2% vs 36.8% success.
- **Eurekaverse** (2411.01775): the evolve prompt gets the learner's success rate and per-reward-term statistics. "No
  Feedback" (always "make it harder") improves early, then degrades as tasks outrun the policy; "Diversity Only", initial
  only and final only all underperform.
- **Rainbow Teaming** (2402.16822): sample the **target cell first** (biased toward low-fitness cells, temperature 0.1),
  then mutate a parent toward it, so no placement classifier is needed.
- **Self-Instruct** (2212.10560): 8 in-context items per call, **6 human seeds + 2 recent generations**; output-first
  generation for small answer spaces (input-first collapses onto one label).
- **NoveltyBench in-context regeneration** and **Verbalized Sampling** (B5.5).
- **Adoption.** F3 context = 5 nearest kept + 5 nearest failed or rejected tasks in the cell with their F6/F9 failure
  reasons + an explicit band statement for the named student + the student's p̂ and failure-cluster summary for variants.
  `TaskProposal` carries the ids of in-context exemplars, so F11 can measure pool drift. P0.

### B2.10 Self-play sources: what sustains them
- **SPICE** (2510.24684): one model; the Challenger sees a sampled document and proposes (q, a*) with a* extracted from
  it; the Reasoner answers without the document. Challenger reward r_C = exp(−(Var(l) − 0.25)² / (2·0.01)) over G = 8
  correctness bits (peaks at p = 0.5), −0.1 for invalid questions. Qwen3-4B-Base 35.8 → 44.9, Qwen3-8B 43.0 → 48.7,
  OctoThinker-3B 14.7 → 25.2, -8B 20.5 → 32.4. Beats R-Zero (+3.7; R-Zero degrades after 5 iterations), Absolute Zero
  (+4.9) and a fixed Qwen3-32B Challenger (+7.2). No corpus: 40.7 vs 43.9. Co-evolution: against a fixed Reasoner, later
  Challengers' pass rate falls 55% → 35%; against a fixed Challenger, later Reasoners rise 55% → 85%.
- **Agent0** (2511.16043): proposer reward = format · max(0, λ_unc·(1 − 2|p̂ − 0.5|) + λ_tool·γ·min(#calls, 4) − R_rep),
  with p̂ the executor's self-consistency over k = 10 and R_rep the BLEU-cluster size in the batch. The executor trains on
  |p̂ − 0.5| ≤ 0.25 and is rewarded against **majority-vote pseudo-labels** (no ground truth). Qwen3-8B-Base math 49.2 →
  58.2. Ablations: untrained curriculum −9.3; no tool reward −7.2; no repetition penalty a large drop; ADPO → GRPO −1.9.
  Tables bold peak-during-training checkpoints.
- **SSR game** (B7.6): solve-rate feedback is weak at G = 8; shallow, grounded self-play.
- **Adoption.** The agentic analogue of SPICE's asymmetry is "proposer sees the World snapshot and privileged notes;
  solver sees only the instruction and tools" (rule 5). Any trained proposer uses Agent0's composite reward with F4
  oracles and F5 checks in place of pseudo-labels. P2 (in-training self-play).

### B2.11 Real-data mining and source choice
- **Terminal-Gym** (MiniMax, PARTIAL): Stack Overflow rebuilt chronologically; drop posts without accepted answers, low
  score or too long; annotate quality, verifiability, category, complexity, env needs; keep scriptable Linux/Docker,
  moderate difficulty; rewrite into a schema (context, tools, I/O formats, success criteria), grade into 4 tiers, keep the
  top 2; an agent writes Dockerfile + test and repairs from diagnostics up to a retry cap. **No task counts or yields are
  reported.**
- **DeepSeek code agent**: millions of issue–PR pairs; an env-setup agent builds environments; one counts only if the
  gold patch gives **F2P > 0 and P2F = 0**; JUnit-format output for cross-language parsing; tens of thousands of
  environments in 8 languages.
- **Search QA hard-and-checked rule** (DeepSeek): keep an item only if the ground truth is verified and **every candidate
  answer from heterogeneous answerers is verifiably wrong**.
- **Nemotron-Terminal sources** (Qwen3-8B, TB2.0): math adapter 5.39, code 6.29, SWE 7.02, all adapters 9.66;
  seed-based synthetic 6.18, **skill-based 12.4**, all synthetic 12.4 (seed data adds no mean gain, lowers variance).
- **OpenThoughts-Agent source ranking** (2606.24855): see B5.7.
- **Adoption.** F2P > 0 and P2F = 0 as the mined-environment acceptance rule; DeepSeek's QA rule as a hard-negative
  generator for F5 answer checks. P1.

### B2.12 Minor tips (B2)
- Blocklist task kinds the solver cannot serve (modalities, tools) at generation time (Self-Instruct).
- Measure a quality rate per field (instruction, input, output) on a sample: Self-Instruct's audit of 200 gave instruction
  valid 92%, input 79%, output correct 58%, all fields 54%. Output quality, not count, bound its gains (regenerating
  outputs with a stronger model gave +10% human eval; gains plateaued after about 16K instructions).
- Separate content generation from format conversion (GLAN).
- Use the reference solution only to derive test expectations, never in the prompt (Nemotron-Terminal).
- Persona + initial screenshot → a simple task in a few atomic actions; feed failed proposals back so the follow-up
  proposer goes simpler (AgentSynth).
- Expand MCQ and free-form formats together: MCQ stabilises verification, free-form trains reasoning (SPICE).
- Build multi-source seeds from mutually similar k-NN triplets (similarity > 0.8) rather than combinatorial search
  (AgentFrontier, 2510.24695).
- Seed injection locations to control duplicates (SSR).
- Frontier labs invert sources into second skills: fix → test-writing task (MiniMax), code review tasks ("approximately
  verifiable", LLM-verified; do not use for RL reward).

---
## B3. Verifiers and gates

### B3.1 Oracle write-DAG matcher with hard, soft, causal and timing checks (ARE Verifier)
- **Steps** (2509.17158). (1) The multiset of write-tool names must equal the oracle's. (2) Walk the oracle graph in
  topological order and map each oracle write to an agent write by **hard** checks (exact ids and arguments) and **soft**
  checks (an LLM judge with tool-specific guidelines, for free text only). (3) **Causality**: every oracle parent must be
  matched to an earlier agent action. (4) **Timing**: within a window around the oracle's relative delay, [Δt − 5 s,
  Δt + 25 s]. (5) Multi-turn: verify per turn and **release the next turn only if the current one passed**; placeholder
  arguments are resolved through the oracle ↔ agent mapping. Reads are not verified (design rule: a unique set of writes
  solves the scenario).
- **Validation.** On 450 human-labelled trajectories: agreement 0.98, precision 0.99, recall 0.95, vs an in-context LLM
  judge at 0.72 / 0.53 / 0.83. Unit tests come from **perturbed oracle trajectories** (validity-preserving vs breaking).
- **Observed RL hack.** On Search (single write = final message) the policy stuffed templated pseudo-code into the
  message, which overwhelmed the soft judge; fixed by a task-agnostic style/sanity soft check.
- **Blind spot.** Equivalent alternative writes (Messages vs Chats) fail.
- **Adoption.** F5: implement `path_constraints` + `timing_checks` as an ARE-style DAG matcher beside state checks; hard
  checks or code wherever possible; soft checks only for free text and always with a sanity check; G14 fixes the
  unique-write weakness; perturbed-oracle unit tests are G5 negatives; aim the anchor set at about 450 labelled
  trajectories. P0.

### B3.2 Deterministic checkpoints over state diffs, run server-side; judges only as audits (WEFT, SearchEyes, TRACE)
- **WEFT.** Given completion conditions and a reference execution accepted by a semantic LLM, an agent writes
  deterministic checks over state diffs and tool returns, plus a rubric; checks are tested on the reference evidence.
  Reward R(τ) = mean over K checkpoints of r_k ∈ {0,1} (atomic tasks + overall goal). The rubric judge is **never RL
  reward**, only a consistency audit. The verifier runs in the serving layer on start DB, end DB and the call ledger
  (B1.6), with DB files hidden.
- **SearchEyes milestones.** Keep the construction path and use each gold intermediate entity as a milestone: a
  trajectory is anchored at entity v_k at the first step v_k appears in its observations; within each anchor group (≥ 2
  members, non-zero variance) compute the group-relative advantage of the final outcome from that step on (latest anchor
  wins); final A = 0.3·A_ep + 0.7·A_hop. HaPO vs GRPO +4.0 (GiGPO +0.7, DAPO +0.3); hop-only +3.3. **Hack surface
  (card):** anchors reward retrieving gold entities, so broad searches that surface many entities are not penalised.
- **TRACE** grades by final-DB hash plus correct communicated result.
- **Adoption.** Each gold intermediate state the F4 oracle touches is a milestone for Φ; the in-training use is a Part 1
  trainer item. Penalise over-retrieval if milestones count retrieval. P1.

### B3.3 Inverse mutation and consistency validation (SSR)
- **Steps** (2512.18552). All must pass: test files exist and cover the weakening patch's files; the parser emits valid
  JSON; the test script on the original passes ≥ min_passing_tests; the bug touches ≥ min_changed_files; ≥
  min_failing_tests previously passing tests fail after the bug; some buggy-state failures pass after weakening;
  **inverse mutation**: for each file in the bug patch, revert only that file in the buggy state and run the non-weakened
  tests; at least one failing test must pass, else the file is unnecessary and the artifact fails.
- **Grading hygiene.** Remove and re-initialise `.git` to block history leakage; restore the oracle test files from the
  original tag before grading; reward +1 if all pass, −1 otherwise.
- **SSR's own gap.** The full oracle tests are visible to the solver; there are no hidden tests.
- **Adoption.** G6 as written, plus a **hidden oracle split**: F5 keeps checks and tests the solver never sees. For
  `world_synth`, G6 generalised: restoring any single corrupted record flips some check. P0.

### B3.4 Stress-test every judge with near-misses and benign perturbations
- **AgentSynth** (2506.14205). The WebJudge-style verifier (extract key requirements, choose key screenshots, judge
  binary + completion %) agreed with humans 88% (inter-rater κ 0.74) on 100 tasks. **Near-miss states** (almost-right
  filename, similar wrong folder) were **accepted as success 12%** of the time; benign UI perturbations accepted 94%.
- **Daedalus judge** (2610.08048) against official verifiers: AppWorld P .943 / R .846 / κ .807; τ²-retail κ .749;
  AutomationBench κ .729; inter-judge κ .85–1.0; precision > recall. Rules: trust effects, not narration; a group outcome
  ("every …") is UNMET unless the trajectory proves the group size (anti-pagination hack). It reads the trajectory, so a
  solver that prints convincing tool output could fool it.
- **Adoption.** G5 negatives must include near-miss end states; report a judge's near-miss and benign acceptance rates.
  Trajectory-reading judges never grade reward. P0.

### B3.5 A catalogue of judge hacking and decoupling
Every LLM-judged reward in group 03 was hacked or decoupled at the top. Execution and pairwise checks held.
| Paper | What broke | Fix that worked |
|---|---|---|
| ARE 2509.17158 | Templated pseudo-code flooded the soft judge | Task-agnostic sanity/style check |
| Qwen-AgentWorld 2606.24597 | Self-praise text in predicted observations under a 5-dimension rubric reward | A 9:1 rubric:rule mix (10% rule anchor), content-type matching (exact for deterministic content), strict tag extraction so thinking never reaches the judge |
| QDAIF 2310.13032 | Human quality decouples from AI fitness in 0.995–1.0; embedding feedback more hackable than LLM feedback | Human spot checks of top elites |
| Rainbow Teaming 2402.16822 | Score-based updates on Llama Guard raised its ASR, lowered GPT-4 ASR, and stopped updating at 1.0 | Pairwise judge with position swap and majority vote; evaluate with classifiers not optimised against |
| AgentBrew 2609.05837 | An LLM-judge trajectory filter collapsed Notion yield to Acc 0 | No judge filter |
| OpenThoughts-Agent 2606.24855 | An LLM-verified RL source's reward rose smoothly 0.54 → 0.73 while the policy compacted (self-correction −59%) | Card: a smoothly rising LLM-verified reward is a warning sign |
| EnvACE 2608.06197 | Policy writes the observations the checklist judge reads | Avoid the design |
| Agent0 2511.16043 | Majority-vote pseudo-labels can converge on consistent wrong answers; tool-call reward farmable (cap 4) | Card: replace with executed oracles |
| OMNI-EPIC 2405.15568 | The same model writes world and success checker; no negative test; human agreement 72.7% | F5 negatives + G5/G12 |
| SearchEyes 2607.05943 | Answer-extraction fallbacks (last line) gameable; hop anchors reward over-retrieval | Strict extraction; penalise over-retrieval |
- **WEFT disagreement filter.** 16 pilot rollouts per task (Nex-N2-mini) scored by the rubric judge and the executable
  verifier; keep tasks where both Pearson and Spearman pass thresholds: 1,954 of 5,000 kept (39%). Thresholds not stated.
- **Adoption.** A rubric-vs-executable disagreement over k frontier pilot rollouts is a cheap verifier-defect G-check
  before F9 (P0). Any soft check: pairwise with swap, a rule anchor, content-type routing, and thinking hidden from it.

### B3.6 Independence of solution and verifier, and verifier strength
- **DeepSeek-V3.2**: "edit solution or verifier until they agree" is the failure rule 5 exists for (B2.2).
- **Mind the Gap** (2412.02674): verification helps only to the extent it adds information over generation. The gap
  rises with verifier capacity and falls with generator capacity (stronger generators' errors are subtler); **AND
  ensembles of verifiers raise the gap** (Qwen-2-1.5B: MC 5.27, CoT-Score 2.78, MC + CoT-Score 7.68). Details in B7.1.
- **Adoption.** Verifier writer ≠ solution writer in family; every hardening round re-runs F5 negatives; judges stronger
  than the generator, combined by AND across families and methods for VALID/INVALID verdicts; this matters when the 397B
  becomes the student. P0 (rule), P1 (ensembles).

### B3.7 Alternatives and false negatives
- **Non-unique answers.** WebSailor admits obfuscation can leave several entities satisfying all conditions; it only
  guarantees the given answer satisfies them, so a correct alternative is marked wrong by its single-answer judge.
- **Unique-write assumption.** ARE fails equivalent alternative writes.
- **One-reference strictness.** WEFT's checks written against one reference may be too strict.
- **Survey rule** (ACE 2608.27260): keep alternative valid paths, because repair against a fixed verifier narrows data to
  verifier-friendly patterns.
- **Homogeneous alternatives.** Artificial Hivemind (2510.22954) suggests "two alternatives from different families" may be
  the same solution.
- **Adoption.** After any obfuscation, query the World for every entity satisfying the stated constraints and reject (or
  accept the full set) unless exactly one remains: exact in a World, impossible on the web. G14 must check that
  alternatives' action traces differ. P1.

### B3.8 Generating hard negatives
- **Verbalized Sampling for wrong answers** (2510.01171): direct prompting follows a "write a plausible wrong solution"
  instruction only 34% of the time and collapses to one wrong answer; VS-CoT reaches 0.89 incorrect rate and 0.57
  coverage of distinct wrong answers; offline RL with 1K positive + 1K VS negatives 36.63 vs 34.12 positive-only.
- **Magnet's five error types** (nested, short dependency, long dependency, wrong summary, missed function/parameter),
  labelled on sampled SFT-model errors and injected back as misleading hints for DPO negatives.
- **Rainbow attacker archive** (2402.16822): MAP-Elites over (verifier check kind × exploit style: empty answer, claim
  success, edit tests, touch protected path, shortcut read, format abuse, judge flooding) with fitness "attack passes the
  verifier"; parent-child BLEU ≥ 0.6 discarded (about 24%); stepping-stone cross-cell mutation beat from-scratch and
  same-cell mutation. Similarity filter: ASR 0.92 vs 0.99 without, Self-BLEU 0.42 vs 0.79.
- **Revert shortcut** (BugPilot): "revert all changes since A′" must not count as a fix unless intended, or the task must
  also require the feature's own new tests.
- **AgentSynth near-misses** (B3.4); **DeepSeek "all candidates wrong"** (B2.11).
- **Adoption.** G12 as a Rainbow-style archive by a different-family attacker; the archive becomes reusable F5 negatives
  and a regression suite for §3.8 verifier changes (P0). VS for F5's wrong and shortcut classes: ask for k plausible wrong
  end states or action programs with probabilities (P0). Map Magnet's error types onto the mutation operators (drop a
  step ≈ missing nested call; wrong value ≈ wrong dependency; missed clarification ≈ infeasible-task negative).

### B3.9 Validate the verifier before it grades
- ARE: 450 labelled trajectories plus perturbed-oracle unit tests (B3.1).
- ARE QA: annotator A writes prompt and oracle; B and C solve it blind; D reconciles and rejects or tightens the prompt.
  Post-QA, a scenario at 100% model success is too easy, at 0% likely misspecified.
- Qwen-AgentWorld: choose and tune judges by a double-blind real-vs-simulated discrimination test; three judges agreed on
  rankings (Spearman 0.92–0.99). Rule verifiers on controllability, error handling and write→read consistency are kept
  only where rule and rubric labels agree.
- Mind the Gap: measure each judge's GV-gap on held-out ground truth (B7.1).
- **Adoption.** F10 uses blind independent re-solves by two different-family models (ARE's B and C) to catch ambiguous
  instructions. P1.

### B3.10 Accuracy is conjunctive; type-specific rewards; layered app verifiers
- **ACE** (2608.27260): A(d) = V_E(E) ∧ V_q(q|E) ∧ V_τ(τ|E,q) ∧ V_v(v|E,q,τ); relational checks matter because separately
  generated factors drift apart; passing a layer is no evidence for properties that layer cannot observe; record failure
  class and provenance.
- **MiniMax Agent-as-a-Verifier** (AppDev): deploy the app in a sandbox; layered binary checks with mandatory evidence:
  execution layer as a hard gate (files, deps, build, server, no JS errors), interaction layer (Playwright drives core
  workflows), aesthetics layer.
- **Adoption.** State the F6 contract as ACE's conjunction. P1.

### B3.11 Minor tips (B3)
- Mask only positive advantage on degenerate turns (no-progress repetition, malformed calls, interface violations):
  min(A, 0) (WEFT). Zero the advantage after ≥ 3 consecutive tool errors and clamp the prefix to max(A, 0) (SearchEyes).
- Format penalty (> 10 answer tags → reward/4) and a repetition kill-switch (SearchEyes).
- Tag every tool read or write; verify writes, let reads vary (ARE).
- Partial-credit components can be farmed (TRACE); a small partial reward for well-formed wrong answers helps, too large
  invites gaming (R³: best ε = 0.2).
- Accuracy and reward can diverge when rewards mix outcome and format terms (Agent-R1, Reinforce++ on GSM8K).
- An evaluator must read the same state generation as the agent's last action (CUA-Sandbox).

---
## B4. Intent, validity, communicability and leakage

### B4.1 Executable communicability: the privileged-solver check
- **RandomWorld** (2506.11045): a solver given privileged help (only the needed tools, in gold order, with signatures and
  examples) must reach the goal from the instruction; otherwise the instruction is under-specified and the task is
  discarded. Its failure is not task infeasibility: it may drop the hardest tasks.
- **WEFT DAG check**: every node input must come from the initial state or an upstream outcome.
- **AgentSynth caution**: the summariser that merges k subtasks can drop a subtask's constraint (subtask coherence 90%),
  creating an instruction–trajectory mismatch.
- **Adoption.** G8 made executable and cheap, before F9: a frontier solver given only the needed tools in order must
  reach the goal; plus the static check that every exact value the oracle uses is visible in the start state or the
  simulated user's known info. Run it on composed (summarised) instructions. P1.

### B4.2 Shortcut and memorisation probes
- **Closed-book probe** (WebShaper 2507.15061): reject any sub-question a no-tool model answers with the constant itself
  (too easy or memorised). It used one model (QwQ); the student's memorisation was not tested.
- **SearchEyes's five filters**: degenerate path, single-hop shortcut, knowledge contamination, cross-modal grounding,
  answer verifiability. Drop answers among the top-100 common entities or single-token answers. Ablation (avg of 4):
  full 61.8; **without anti-shortcut filters 54.1 (−7.7)**; without P/K alternation 57.6; without retrieval boost 57.1;
  without concealment 59.3; without treewidth constraint 59.3; without domain balancing 60.0; free-form GPT-4o synthesis
  without the graph 49.2 (−12.6). Single runs.
- **Structural shortcut checks** (WebShaper): random expansion attaches constants to constants (**redundancy**, no extra
  reasoning); sequential chains leave a constant directly attached to the target (**reasoning shortcut**: the solver can
  guess from the near constant). Layer-wise expansion of leaf constants blocks both.
- **Adoption.** G9 gains two structural checks computable from the oracle's query plan or trace: **no constant node
  attached directly to the target**, and **no constant–constant edges**. F10's instruction-only ablation runs the
  closed-book probe on the student family as well as a frontier model. Store `TaskProposal.formal_graph` privately; F7
  compares the oracle's traversed edges with it. P0 (the anti-shortcut ablation is the strongest single gate evidence in
  these cards).

### B4.3 Paired "base fails, assisted succeeds" band, and why hardness is not intent
- **ACE** (2608.27260): complexity C_z(d) = 1 − Pr[v = 1 | d, z], with z = model + scaffold + tools + verifier + sampling
  + budget. Structural features are explanatory controls, not difficulty. A task needs agentic skill when
  p_{z0}(d) < ρ ≤ p_{zA}(d): the base configuration fails and the assisted one succeeds.
- **AgentFrontier caution** (2510.24695): its ZPD is defined against a no-tool base model, so most of the measured gap
  is tool access, not skill.
- **Other cards' unmeasured intent claims.** TRACE asserts "success depends primarily on c" without ablation;
  Nemotron-Terminal's "3–5 skills per task" is never checked; AgentSynth's composed difficulty is mostly horizon and
  memory span, and failures are dominated by GUI grounding, which may not be the intended skill; WebSailor's obfuscation
  raises search cost without necessarily changing reasoning structure.
- **Adoption.** F10 adds the paired band as a probe (instruction-only or no-tools configuration vs full). Record z in
  every `DifficultyRecord` (scaffold, tool set, budget, verifier version). Run the budget-2× probe on composed and
  obfuscated families. F7 must show the claimed skills' observables (`SkillLeaf.observable_in`), else relabel. P1.

### B4.4 Leak rules that the cards found necessary
- The instruction writer sees variables, not intermediate values (RandomWorld).
- Never name path entities; at most one coarse domain hint; describe the constraint entity relationally (SearchEyes).
- Back-translated queries refer to earlier outputs indirectly and never name tools (Magnet).
- Tool names and descriptions can leak the call chain (RandomWorld App. C); shuffle tool order in prompts (Magnet).
- The reference solution derives test expectations only (Nemotron-Terminal); no solution leakage in the prompt.
- Bug reports written from failing-test output leak the failing test (BugPilot).
- Remove and re-init `.git`; restore oracle test files before grading (SSR).
- Strip the trace's chain of thought before relabelling (AgentBrew).
- A learned simulator's instruction carries the success state (Qwen-AgentWorld); the oracle's earlier actions must not be
  readable from a late-start snapshot (R³ card).
- When easing a too-hard task, shrink the sweep; never reveal ids or steps (Daedalus).
- **Adoption.** These become `DomainSpec.leak_rules` entries checked at G11. P1.

### B4.5 Benchmark contamination inside synthesis
- Cases: TRACE, NoisyAgent, Magnet, Learn-by-interact (header warning); SearchEyes's own-pipeline benchmark measures
  in-distribution skill; DeepSeek's internet-sourced DB data; Magpie may regurgitate alignment-set prompts (unchecked).
- Procedures that worked: GLAN's train/test loss-gap check on ARC, GSM8K, MATH; Prismatic's 10-gram match, then
  embedding nearest-neighbour + LLM paraphrase judge; Nemotron's 14-gram decontamination vs TB2.0 plus identity-leak
  filters; Self-Instruct seeds written without reference to test sets (mean ROUGE-L 0.21 to SuperNI test instructions).
- **Adoption.** G11 runs n-gram, then embedding-NN + paraphrase judge against the sealed benchmark; the sealed split
  never feeds F0–F3 or the selector. P1.

### B4.6 Lucky-pass guards
- **Failure then streak** (Daedalus): accept a task only after ≥ 1 failure followed by N_s = 3 consecutive successes; stop
  at N_f = 8 total failures (too hard) or no failure (too easy). Card critique: this is a recoverable-failure band, not
  a p̂ band, and calibrates to one solver.
- **Weak solvability filters admit lucky passes**: DeepSeek's policy pass@100 > 0; AgentFrontier's strong-agent ≥ 1/3
  rule (a 1/3 item may be ambiguous).
- **Adoption.** Solvability by construction replaces sampled solvability. F10's lucky-pass rate can use an N-streak test
  with the learned lesson. P1.

### B4.7 Instruction–check consistency and fault sides
- **ACE rules**: added obligations must be checked by the verifier, otherwise they are label noise; hidden information
  counts as complexity only if it has an intended source and a recovery action.
- **MiniMax** runs model-based validation of description–test consistency on mined SWE tasks; Terminal-Gym grades every
  hint variant with one unified test suite so tests check the logic, not the phrasing (B6.5).
- **WEFT attribution sides**: policy, World, task, verifier (B2.7).
- **Scaffold side effects**: FEE-trained agents asked for clarification less (Miss-Param −7 pp, 4B DAPO; Miss-Func −4 pp,
  8B GRPO); a scaffolded variant may not exercise the parent's intended skill (QuestA card).
- **Adoption.** F10 labels scaffold variants with the reduced skill set; infeasible and clarification tasks are measured
  before and after scaffolded training. Fault sides per failure label feed F12. P1.

### B4.8 Minor tips (B4)
- The GEPA-vs-training gap in TRACE (39.6 vs 48.2) is a control that the skill needs training, not just prompting.
- An "equal / not equal" judge for no-information-gain mutations is a cheap semantic no-op detector (WizardLM), but the
  real test is a changed behaviour signature.
- Accept search answers only when grounded in retrieved evidence, not memory (MiniMax evidence specification).
- Concealment ablation as an intent probe: leak entity names and measure the pass-rate jump (SearchEyes card).
- Instruction under-specification is a deliberate hardness lever only when every hidden value is discoverable through
  tools (Daedalus doctrine + G8).

---
## B5. Diversity and coverage

### B5.1 Vendi score: exact definition and safe use
- **Definition** (2210.02410). For n items and a PSD similarity k with k(x,x) = 1, let λ_i be the eigenvalues of K/n.
  **VS = exp(−Σ λ_i log λ_i)**, the effective rank of K. With embeddings X ∈ R^{n×d} and a cosine or inner-product kernel,
  use the d×d matrix XᵀX/n: cost O(d²n) instead of O(n³); otherwise Nyström sampling. The empirical kernel entropy
  converges at about 1/√n.
- **Properties.** VS ∈ [1, n]; exact duplicates leave it unchanged; for mutually dissimilar groups VS(union) =
  exp(H(group shares)) × Π VS(group)^{share}; averaging kernels that capture different attributes raises it, more when
  the attributes are uncorrelated (average pairwise similarity does neither).
- **Evidence.** On molecules, IntDiv ranked an HMM among the most diverse generators while VS ranked it least diverse,
  because it emitted clusters of exact duplicates, the swarm's failure mode. StackedMNIST: two GANs both hit all 1,000
  modes, yet VS 746.7 vs 866.6 vs data 943.7, so mode counts hide within-mode collapse. Beam search reads as about 3
  distinct captions out of 5.
- **Pitfalls (card).** Random noise scores high, so pair it with a validity rate; the kernel decides meaning; VS grows
  with n, so compare at a fixed n (or report VS/n); no significance test, so bootstrap over subsamples for alarm
  thresholds.
- **Adoption.** F8/F11/§3.8 meters as specified, at fixed n per batch with bootstrap CIs, per generator family. A concrete
  behaviour kernel: the average of Jaccard over tool multisets, tables written and entities touched, plus an RBF over
  (horizon, virtual time). A falling VS/n in a growing pool is the near-duplicate alarm. Interface
  `DiversityMeter{kernel_id, n, vs, vs_ci, per_family}` written per batch to F12. P0.

### B5.2 Validate the diversity metric against transfer; accept into sparse cells (Prismatic Synthesis)
- **G-Vendi** (2505.20161): unit-normalised loss gradients of each (x, y) under a small frozen instruction-tuned proxy
  (Qwen2.5-0.5B-Instruct), projected to d = 1024 by a Rademacher matrix; Vendi of the gradient matrix.
- **Validation protocol.** 300 subsets at fixed sizes spread across the diversity range (random; up via balanced-cluster
  sampling; down via similarity growth from a seed; mixtures); train a small model per subset; OOD = mean over 7 unseen
  benchmarks relative to a model trained on the full pool.
- **Numbers.** Spearman ρ / R² with OOD at 100K: G-Vendi 0.899 / 0.823 (math), 0.893 / 0.791 (NLI); embedding Vendi with
  a 14× bigger encoder 0.754 / 0.659 (math), 0.841 (NLI); LLM skill-set entropy 0.812; 2-gram entropy 0.736; perplexity
  −0.631. In distribution, scale dominates. A diverse 10K set often beat a less diverse 100K set OOD. **Personas ranked
  backwards**: DiverseReason (10K seeds × 0.1K personas) beat DiversePersona (0.1K seeds × 10K personas) OOD 54.77 vs
  38.15; G-Vendi ranked them right (63.2 vs 51.9), embedding Vendi wrong (22.0 vs 24.5). A proxy sharing the student's
  base was best (Llama-3.2-1B-It ρ 0.909; base Qwen2.5-0.5B 0.772). Vanilla and persona-guided generation saturate at
  50–100K; Prismatic kept improving to 1M.
- **Loop.** k-means on gradient features with k = 1% of pool size; generate 5-shot from the pool; keep only items landing
  in the **sparsest k/2 clusters**; quality by majority vote (N = 3, ≥ 2 agree); recluster each round. Heuristics (k = 1%,
  smallest half) not ablated; sparse acceptance can favour broken outliers.
- **Card critique.** For agentic tasks gradients need an oracle trace, and a 300-call trace on a 0.5B proxy may be
  dominated by formatting tokens. The persona finding is math-only. Correlation holds only with quality controlled.
- **Adoption.** Before trusting "Vendi on behaviour signatures", show it rank-correlates with held-out-family transfer on
  small fixed-size task subsets with a small student (a cheap measurement; P0 for its design). Sparse-cell acceptance as
  the F11 rule. Test a G-Vendi variant over oracle traces under a student-family proxy before adopting (P1). Reframe §3.5
  item 3: personas buy realism; diversity comes from cell and World slice.

### B5.3 Dedup by behaviour, with calibrated text layers
- **Tool-chain subsequence dedup** (WEFT): drop a task whose ordered tool-name chain is a non-contiguous subsequence of
  another's; identical chains keep the earlier.
- **Functional equivalence** (NoveltyBench 2504.05228): two outputs are equivalent unless a user who saw one would benefit
  from the other. Partition sequentially against a random class representative; distinct_k = classes among k samples.
  The DeBERTa equivalence classifier: ROC-AUC 0.81; accuracy is reported as 79% / F1 0.811 in the body and 71.0% in App.
  A.4 (inconsistent).
- **SemDeDup** (2303.09540): embed, k-means, compare all pairs within a cluster, pairs with cos ≥ 1 − ε are duplicates,
  keep one per group; tune ε on 10% of clusters at two values and interpolate. Redundancy taxonomy: perceptual duplicates,
  semantic duplicates (same content, different surface, e.g. one name swapped), semantically redundant (different
  instances of one concept), misleading. LAION: removing 37% cost nothing on ImageNet zero-shot; 50% cost < 0.5%; which
  duplicate is kept barely matters. **Card critique**: two tasks with near-identical instruction embeddings can differ in
  the one World fact that changes the solution.
- **Lexical thresholds used in the cards**: Self-Instruct ROUGE-L < 0.7 to the whole pool; Rainbow parent-child BLEU
  ≥ 0.6 discarded; Promptbreeder BERT-cosine > 0.95 filtered before the mutator sees the population; Persona Hub MinHash
  0.9 then embedding 0.9; AgentFrontier reranker ≥ 0.7 over questions.
- **Solution-preserving edits are augmentation** (ACCEL card): renames and distractor reorderings that keep the optimal
  path are a different label from new tasks.
- **Adoption.** F8 labels = SemDeDup's taxonomy: MinHash layer = perceptual; embedding layer = semantic duplicate, a
  **hold-for-review flag**, calibrated on a small human-labelled set of task pairs (duplicate vs legitimate variant);
  equal `BehaviourSignature` = semantically redundant, the **drop rule**. Tool-chain subsequence is a cheap behavioural
  layer. Interface `DedupDecision{layer, nearest_id, sim, signature_equal, action: drop|hold|keep}`. A per-generator
  `distinct_k` meter: sample k proposals for one cell and count signature classes. P1.

### B5.4 Homogeneity: why model mix and scale do not diversify
- **Artificial Hivemind** (2510.22954): in 79% of queries one model's 50 responses average > 0.8 similarity at T = 1
  (random cross-pool pairs 0.1–0.2); min-p at T = 2 still leaves 81% of pairs > 0.7. Cross-model mean similarity
  0.71–0.82 (DeepSeek-V3 vs qwen-max 0.82, vs GPT-4o 0.81); about 8 distinct models in a typical top-50 similarity cluster.
  Paraphrasing the prompt barely helps (0.821 within-prompt vs 0.781 cross-paraphrase). LM, reward-model and judge scores
  correlate with humans markedly less on similar-quality and high-disagreement subsets.
- **NoveltyBench**: distinct out of 10 — Claude-3.5 Sonnet 1.76, GPT-4o 2.88, Llama-3.1-405B 3.20 vs Llama-3.2-1B 6.74,
  Gemma-2-2B 5.66. Each post-training stage (SFT → DPO → RLVR) lowers diversity, most at DPO; more reasoning effort lowers
  diversity. Paraphrase and "be unique" prompts help marginally.
- **Verbalized Sampling theory** (2510.01171): rewards carry a typicality term, r = r_true + α·log π_ref + ε, with α̂ =
  0.57 ± 0.07 (Llama-3.1-405B base) and 0.65 ± 0.07 (GLM-4.5 base) on 6,874 matched HelpSteer pairs; under KL-regularised
  RLHF the policy sharpens to π_ref^γ with γ = 1 + α/β > 1 wherever true utility is flat.
- **Magpie** prior collapse (> 50% information seeking).
- **Adoption.** Never count family mix as a diversity mechanism. Add the "distinct generator families among the top-N
  nearest neighbours" statistic to per-batch meters. A smaller model may be the better breadth proposer, a larger one the
  verifier. P0 (design constraint, no build cost).

### B5.5 Structural levers inside the proposer call
- **In-context regeneration** (NoveltyBench): keep previous answers in context and ask for a different one; the only
  prompting fix that reached the human group's diversity; GPT-4o and Gemini-2.0-Pro then exceeded human utility.
- **Verbalized Sampling**: VS-Standard ("k responses with probabilities"), VS-CoT, VS-Multi (multi-turn, "k more with
  probabilities"); tail thresholds ("each response < p", p from 1 to 0.001) as a diversity dial. Creative writing
  diversity 1.6–2.1× direct at comparable quality; VS-Standard +12% cost for +86% diversity (2,000 poems). Tulu-3-70B:
  direct diversity 20.8% after SFT → 10.8% after DPO, VS stays ≈ 30%. Synthetic math SFT (avg of 3 benchmarks, 3
  students): direct 30.6 (below the no-SFT 32.8), sequence 34.3, VS-Standard 36.1, VS-Multi 37.5. Larger models gain
  1.5–2× more; small models can lose quality. Verbalized probabilities are a steering device, not calibrated
  likelihoods.
- **Archive-driven mutation** (QDAIF 2310.13032): LMX (archive members as few-shot), rewrite-to-a-different-bin for
  strong models, and **guided** proposals that name an empty target bin (LMX-rewrite missed limericks unless guided).
  Quality-only search was no better than fixed prompts. Adding the second axis at iteration 1,000 of 2,000 approached the
  full-2D QD score with better best quality. Use non-uniform bins when an axis comes from logits.
- **Rainbow**: cell first, low-fitness bias, cross-cell stepping stones (B2.9, B3.8).
- **Decoding sweep** (Magpie): T 1.0 → 1.25 and top-p 1 → 0.99 lower judged quality slightly but raise difficulty and
  min-neighbour distance; generic system prompts made queries easier.
- **Generation budget per seed** (2410.15226): more topics raise diversity; more generations per topic help, then saturate
  or hurt; multi-topic prompts reduce diversity; Cosmopedia v0.2's "optimised" prompts were less diverse and worse than
  v0.1; stronger generators gave more diverse data; mixing generators helped slightly; a balanced real:synthetic ratio was
  best.
- **Adoption.** F3 per cell: VS-Multi with k = 5, the K nearest tasks *and the batch's own earlier proposals* in context,
  candidates taken from a mix of high and low stated probability, tail thresholds when a cell stalls; **F7's measured
  signature, not the stated probability, decides novelty**. Cap generations per cell or seed. Name the empty cell in the
  prompt. Staged axis expansion when a DomainSpec gains an axis. Interface `TaskProposal.verbalized_p`,
  `proposal_batch_id`, `turn`. P0.

### B5.6 Coverage meters and selection
- **Tag manifest** with target shares and a live UNDER/OK/OVER tally (Daedalus).
- **Per-App tool coverage** = distinct tools in retained chains / App tools; generate toward uncovered tools (WEFT).
- **Coverage-first selection** (InsTag Alg. 1): sweep the pool by complexity, add an item only if it adds an uncovered tag
  to the batch set, reset when all tags are covered. At a fixed 6K, coverage raised MT-Bench with a plateau between 50%
  and 90%. "Tag count = complexity" rewards verbose multi-intent items; tags are claims about text.
- **Leaf and leaf-pair coverage** (GLAN).
- **Diversity only on valid, in-band tasks**, with per-factor coverage and normalised entropy plus a few **joint slices
  to catch template locking** (each factor diverse alone but only in fixed combinations) (ACE).
- **LLM Cluster-agent metric** (2410.15226): derive clustering criteria from samples, cluster 10-item draws with a
  verification pass, repeat 5,000 times; D = mean(C_i / S_i). It correlated linearly with pretraining and SFT accuracy where
  n-gram, perplexity and k-means did not (k-means ranked 300K topics below 100K). Costs about 5,000 LLM calls per
  measurement.
- **PATA-EC** (Enhanced POET 2003.08536): describe a task by the vector of clipped, rank-normalised scores of a fixed
  solver population; distance in that space is task novelty with no domain features.
- **Action-sequence diversity** over instruction diversity (OS-Genesis: humans write varied instructions but narrow
  action sequences).
- **Adoption.** Meters computed post-F6 and per difficulty band; 2–3 declared joint slices (skill × World facet, App ×
  answer kind). F11 admits tasks that fill an uncovered measured cell first, then by learnability. A PATA-style signature
  from a fixed panel (frontier models, student checkpoints, scaffold variants) complements the trace signature: two tasks
  that rank the panel identically teach little new (P1). The cluster-count meter is a monitoring-only fifth meter for
  intent and scenario diversity (P2).

### B5.7 Source quality, not source count
- **OpenThoughts-Agent** (2606.24855): source spread up to about 30 pp on SWE-Bench-V-100 and about 10 pp on TB2.0 between
  ranks 1 and 95; top-4 sources swe-smith, stackexchange-superuser, stackexchange-tezos (997 unique tasks), issue-tasks;
  naive self-instruct ranked 17th. Mixing at 10K: Top-4 (+0.49 z) ≈ Top-2 > Top-8 > Top-16 (−0.10) > Top-32 (−0.48) ≈
  Top-1 (−0.57). At 100K (32B), Top-16 below both Top-8 and Top-4 on all three benchmarks (SWE 40.33 vs 49.00 / 45.33).
  Rollouts per task plateau 31.6K → 100K: **task-description diversity is the bottleneck**; rewriting Tezos' 997 problems
  into > 21K surface forms kept improving.
- **Nemotron-Terminal**: skill-based 12.4 vs seed-based 6.18 (B2.11).
- **Adoption.** F12 ranks sources and generators by downstream value and drops the tail; count distinct task
  descriptions per family as the scaling unit, not rollouts. P1.

### B5.8 Minor tips (B5)
- Dedup mined prompts by user so heavy users do not dominate (NoveltyBench).
- Inspect the largest duplicate clusters: templated "one name swapped" items dominate them (SemDeDup).
- Measure diversity inside one model and across models separately (Hivemind).
- Densely label a few items (25 raters) to see disagreement rather than sparsely labelling many (Hivemind).
- Tune persona-dedup thresholds to the diversity needed (Persona Hub); persona-induced diversity shrinks under topic
  constraints, so measure it per cell (at persona similarity 0.9, generated problems sit mostly at 0.6–0.75).
- Track joint distributions, not only marginals (ACE).
- Use a domain-relevant embedding model, though its training set matters little for duplicate detection (SemDeDup).

---
## B6. Difficulty and curriculum per student

### B6.1 Band targets and proposer rewards, side by side
| Source | Target or rule | Setting |
|---|---|---|
| SPICE 2510.24684 | Gaussian in variance, peak at p = 0.5: r = exp(−(Var − 0.25)²/0.02); −0.1 invalid | G = 8; ranking: variance > R-Zero's 1−2\|p−0.5\| > threshold (0<k<8) > Absolute Zero's 1−p |
| SSR 2512.18552 | Injector reward 1 − (1+α)s for 0<s<1, −α at s ∈ {0,1}, −1 if inconsistent; α = 0.8 | Game analysis: optimum p* ≈ 0.2 at G = 8; a Beta reward s^a(1−s)^b peaks near a/(a+b) |
| Agent0 2511.16043 | Executor trains on \|p̂ − 0.5\| ≤ 0.25 (p̂ ∈ [0.3, 0.8]) | k = 10 self-consistency |
| TRACE 2604.05336 | Base success 30–60% (> 80% too easy, < 15% too hard); std > 0.2; > 60% groups with variance | Builder self-check with ≥ 100 rollouts |
| QuestA 2507.13266 | Keep hinted items at 0–4 / 8 | Hard pool first filtered at 0–1 / 8 |
| WebSailor 2507.02592 | Drop 8/8 before RL; drop zero-variance groups | DUPO |
| OpenThoughts 2606.24855 | Best RL source had a moderate non-saturating reward ≈ 0.47–0.51 | RLOO, 8B |
| Daedalus 2610.08048 | ≥ 1 failure, then 3 consecutive successes; 8 failures = too hard | One solver |
- **Adoption.** The Gaussian-variance score is the F9 *ranking* utility inside the band (`archive_score` in F11). SPICE
  supports a centre near 0.5 at G = 8 for learning signal; SSR's 0.2 is the optimum of its injector reward (B9 notes the
  tension). The TRACE triplet (mean, std, informative-group share) is a builder-side pre-check on the frontier, before the
  paid student run. P1.

### B6.2 Regret, replay and editing: the UED family
- **PAIRED regret** (2012.02096): regret ≈ max over antagonist episodes U(τ_A) − mean over protagonist episodes U(τ_P),
  clipped at 0. Unsolvable tasks give zero regret automatically; with an efficiency bonus the adversary is drawn to the
  easiest task the student still fails. Only PAIRED kept increasing solved path length; transfer Labyrinth 40% vs 10%
  (minimax) vs 0% (DR), Maze 18% vs 0% vs 0%. A generator whose random policy mostly makes infeasible maps failed (solved
  length ≈ 2 vs ≈ 15). The antagonist must stay stronger than the protagonist, or regret collapses.
- **PLR sampler** (2010.03934): P_replay = (1 − ρ)·P_S + ρ·P_C; score S_i = mean |GAE| on the last episode; **rank
  prioritisation** h = 1/rank, P_S ∝ h^{1/β}, β = 0.1 (proportional and greedy were worse); **staleness** P_C ∝ c − C_i
  with ρ = 0.1 (Procgen) or 0.3 (MiniGrid); gains only for 0 < ρ < 1. P_D(replay) = |seen| / |train| anneals toward
  replay. Procgen easy: 128.3% of uniform (10/16 games significant), with UCB-DrAC 176.4%; MiniGrid
  ObstructedMazeGamut-Easy 0.53 → 0.85. Scores must fall as a task is mastered, else replay collapses. For GRPO there is no
  value function: use p̂(1 − p̂) or mean |advantage|.
- **ACCEL** (2203.01302): a replay buffer of 4,000–10,000; replay rate 0.8–0.9; on replay, train on θ, then **edit**
  θ → θ′ with a few random operations, roll θ′ out without gradient, admit it if its score beats the buffer's lowest;
  score = positive value loss. Lineages edited 40+ times stay at the frontier for the checkpoint that sees them. Mazes:
  aggregate IQM near 100% vs < 80% for Robust PLR; PerfectMaze 51×51 53% vs 25%; BipedalWalker 8D about 3× PLR. Ablation
  PLR → +Edit → +StartSimple: editing helps in both domains; starting simple matters most in BipedalWalker. **No explicit
  diversity mechanism**; random edits were competitive with an RL editor.
- **Minimal criterion against all past solvers + ANNECS** (Enhanced POET 2003.08536): admit a child only if not too easy
  and not too hard for current agents; ANNECS = cumulative count of environments that passed the minimal criterion
  against **all agents ever produced** and were eventually solved. Original POET's ANNECS plateaued after about 20K
  iterations; Enhanced POET kept rising nearly linearly. Improved transfer (beat the max of the incumbent's 5 most recent
  scores) cut false replacements 50.4% → 22.3%.
- **Adoption.** `DifficultyRecord.regret = ref_pass − student_pass`, with the oracle giving 1 for the upper term; high
  regret with in-band p̂ goes to the pool, frontier-also-fails is a suspect (§3.7 zero-solve). Use regret as a bandit
  signal over knob vectors, not to RL-train a generator. PLR as the default in-training sampler over the kept pool, keyed
  per (task, student, checkpoint), β = 0.1, ρ = 0.1–0.3, `SamplerState{task_id, student_id, score, last_seen_step,
  visits}` (P1). The ACCEL loop is the task loop's core: parent = in-pool task whose p̂ rose above band; edit = one knob
  change; the child goes through F4–F7 and enters only if in band for the target student and its signature differs on
  the changed knob; lineage fields (`parent_id`, `generation`, `edit_ops`) and a **lineage cap per cell** supply the
  missing diversity control (P1). An ANNECS-style factory curve is a P0-cheap measurement; the minimal criterion against
  all past checkpoints guards against re-admitting learned tasks.

### B6.3 Predict before paying, and the ZPD sandwich
- **MAGELLAN** (2502.07709): goal prompt → LLM last-token state → 1-layer MLP (128, tanh) → success probability, trained by
  BCE on the last 5,000 outcomes with recency weighting, updated every 32 policy updates; **the LLM is fine-tuned on its
  own LoRA, separate from the policy**. Absolute learning progress ALP(g) = |C_t(g) − C_{t−N}(g)| over the last N = 100
  predictor snapshots; goals sampled ∝ ALP with ε annealed 1 → 0.2. Held-out goals: mean |observed − predicted| 0.11
  (MAGELLAN) vs 0.53 (online per-goal) vs 0.09 (expert groups). Frozen representation plateaued; a shared adapter with
  the policy was slower. Infeasible goals cluster in its embedding space, even by reason. **Card caveat**: needs thousands
  of binary outcomes; predicts nothing for a new student until warm-started.
- **OMNI LP estimator** (2306.01711): normalise success by a random-policy baseline; p_recent = EMA(t_norm, 0.1),
  p_gradual = EMA(p_recent, 0.1); reweight f(p) = (1 − p_θ)p / (p + p_θ(1 − 2p)), p_θ = 0.1; bidirectional LP =
  |f(p_recent) − f(p_gradual)| also catches forgetting.
- **Cheap predictor features**: frontier token spend (keeping tasks GPT-5 spends more tokens on ≈ +3 pp, OpenThoughts);
  a reference solver's historical pass rate and the number of env-repair iterations (MiniMax Terminal-Gym).
- **AgentFrontier ZPD sandwich** (2510.24695): escalate by one of four named moves (knowledge expansion, conceptual
  abstraction, factual grounding, computational formulation) until the no-tool base model fails, or K_max = 30 (mean
  7.81); then the tool-using strong agent gets N = 3 attempts: ≥ 1 correct → post-training; 0/3 → **human review**.
  Costs: refinement ≈ $0.24 per candidate; verification ≈ $0.18 with a 33% pass → ≈ $0.54 amortised; **≈ $0.78 per kept
  item**. Two-thirds of refinement spend lands in human review or discard.
- **Adoption.** A per-student competence predictor `p̂_pred(task, student)` trained on all F9 outcomes (task text +
  signature features), starting from the existing advisory predictor (MAE 0.204), used to rank before the k = 4
  sequential measurement and to compute ALP per cell for F12; warm-start the 397B predictor from the 35B's outcomes with a
  student-id feature; an infeasible-cluster placement feeds the §3.7 suspicion. Interface `predictor_score`,
  `predictor_version`, `ALP(cell, window)`. For the 397B, the 35B is a natural weak reference in the sandwich. All-fail
  routes to INCONCLUSIVE/human, as §3.7 says. P1.

### B6.4 Hardness from structure, and why textual hardening fails
- **Composition of verified subtasks** (AgentSynth 2506.14205): a proposer writes a simple subtask; an executor (≤ 10
  steps) runs it; a verifier judges; a reviser relabels partials; a follow-up proposer builds on the current state and
  earlier failed proposals; a summariser merges the first k of n subtasks, so k = 1…6 gives six levels from one chain.
  o4-mini 18% at level 1 → 4% at levels 5–6; GPT-5.1 with Agent S3 62% → 14%; humans 70% at level 6. Horizon 5 → 45 steps,
  apps 1.2 → 3.3, app switches 0.5 → 4.3, memory span 2 → 18 steps; perception share flat at ~40%. **Asymmetry ablation**:
  asking for long tasks directly gives generation success 64/33/11% (easy/medium/hard) with evaluation flat at 60/51/48%
  (survivors are easy); AgentSynth holds generation at 65/57/52% while evaluation falls 62 → 14%. Cost ≈ $0.011 per step,
  ≈ $0.60 per 6-subtask chain, ≈ $0.10 per task.
- **Layer-wise leaf expansion** (WebShaper 2507.15061): with the answer fixed, turn each leaf constant into a variable
  defined by a new sub-question whose answer is that constant; difficulty = number of layers (depth) and intersections
  (width). Layer-wise beat sequential and formal beat natural-language expansion on all backbones; trajectories needed
  > 3 searches 3–4× more often than baselines. GAIA: 60.1 (Qwen2.5-72B), 53.3 (QwQ-32B); RL after SFT +7.8 (32B) and
  +13.5 (72B).
- **Constraint-edge structure** (SearchEyes): disambiguating constraint edges and concealment (B2.5, B4.2).
- **Scope words** (Daedalus): all, latest, last N days, beyond page 1, none or many matches, policy limits.
- **Graph operations** (Magnet): merge, insert, split.
- **Realistic faults** (BugPilot FeatAdd 41.4% vs SWE-smith 65.9%) and **removal + history reversion** (SSR).
- **Composition profiles** (WEFT) and **3–5 primitives per task** (Nemotron-Terminal; unverified that tasks exercise them).
- **Obfuscation** (WebSailor): exact dates → vague periods, names → initials, quantities → qualitative; SailorFog-QA
  before filtering: o4-mini 47.3% (WebDancer-QA 90.2%, BrowseComp-en 26.3%). Requires an executed uniqueness check
  (B3.7).
- **What fails.** Evol-Instruct (2304.12244): six operators (add constraint, deepen, concretize, more steps, complicate
  input, mutate-to-rarer) capped at 10–20 added words, judged "harder" by an LLM rating text, not by any solver; average
  benchmark score did rise with evolution round (C0 → C4) and the seed's skill mix propagated (ShareGPT 4.3% math vs
  Alpaca 11.8%). OpenThoughts: **all task augmentations (combine, add constraints, harden) failed to beat untouched
  descriptions at 10K**. AgentSynth's direct-hard ablation above. Eurekaverse "No Feedback" overshoots.
- **Adoption.** Chain-and-summarise is a direct `world_synth` family generator (P0): oracle = concatenated subtask
  programs executed in the World; verifier = state diff after k subtasks; `count` knob = k; memory span and app switches
  go into `BehaviourSignature`. Layer-wise expansion over World relation graphs with SQL-exact answers (P0). Map Evol
  operators to knobs (add constraint → collateral or path constraint; concretize → specific World entities; more steps →
  horizon/composition; complicate input → distractors/observability; mutate-to-rarer → move to an empty cell), each
  variant recording `parent_id`, `operator`, `knob_delta`, and rejected unless its signature changes on that knob (P1).

### B6.5 Ladders for too-hard tasks, and their removal
- **QuestA partial solutions** (2507.13266): prepend the first p% (by tokens) of the *final solution block* as a hint;
  RL on Partial-50 until saturation (≈ 100 steps, when entropy starts falling), re-filter, then Partial-25 (1,900 steps);
  Partial-25 → Partial-0 gave nothing. Hints never at evaluation. Nemotron-1.5B average 53.14 → 63.26 (AIME24 61.77 →
  72.50); same data without hints 60.44, so hints add +2.82 and reach the no-hint score in about half the steps;
  Partial-50 only 60.26. pass@k kept or lifted across k (RL on easy prompts lowers large-k). **Calibration table**:
  Partial-50 made 17,741 items 8/8, while 0/8 counts were 3,155 / 3,589 / 3,812 at 25 / 10 / 0%: the useful hint range is
  steep and narrow. Theory: a hint of probability ≥ δ^{1/2−ε} cuts the sampling budget to roughly its square root.
- **R³ late starts** (2402.05808): start from M = 5–6 intermediate states of a correct demonstration; **mix all start
  positions** (staged progression overfits early stages and drops at switches); equal reward for every start
  (difficulty-scaled rewards lowered the average 63.8 → 58.7–60.0); keep the hardest (earliest) starts. Llama2-7B CoT
  avg: SFT 60.19, RL 61.48, staged 62.81, R³ 65.62; KL β = 0 still worked.
- **Feedback-Enriched Environments** (2609.08404): action guidance (next valid actions, suggested action) and observation
  enrichment (internal progress tracker, expanded tool returns); each applied with p = 0.5; best schedule **guidance
  early (steps 0–100), enrichment late (100–200)**; **the same enrichment decision for every rollout in a GRPO group**
  (per-rollout random enrichment was unstable). Average +2.82 pp over standard env, +4.3 on hard-tier envs; entropy
  collapse delayed; regressions on Miss-Param (−7) and Miss-Func (−4); still zero on AppWorld. Single runs with peak
  checkpoints reported.
- **Terminal-Gym hint removal** (MiniMax): abstract or remove hints, paths and expected outputs while keeping semantics;
  **every variant graded by one unified test suite**; prefer variants with fewer hints and lower zero-shot pass. No counts
  or yields reported.
- **ARE hints**: optional step-by-step natural-language solutions for RL when tasks are too hard.
- **Adoption.** `hint_level ∈ {oracle prefix 50%, 25%, 0}` implemented as (a) checkpoint start after the oracle's first k
  actions (the existing `difficulty.py:237-296` variants) or (b) a text plan hint; calibrate per family with a pass-rate
  histogram per level; schedule removal by entropy or p̂. **Variants share the parent's `VerifierBundle`** and record
  `hint_level` or `start_offset` and `parent_id`; the unscaffolded p̂ is the real metric; a scaffolded variant is its own
  `DifficultyRecord` row. Train late starts mixed, with equal reward, with the oracle's action log hidden. Enrichment is
  keyed by (task, step bucket, training phase), constant within a group; the milestone set Φ supplies observation
  enrichment and the F4 oracle supplies guidance. P1.

### B6.6 Perturbation curricula with group-safe normalisation (NoisyAgent)
- **Steps** (2605.27209). Of N rollouts per task, N_noise are noisy; **normalise advantages separately within the clean
  and noisy subgroups**, loss over the union (GSPO base). Robustness gap Δ = success(clean) − success(noisy); start at
  ρ = 0; when Δ < θ = 0.05 raise the noise share (cap ρ ≤ 0.5) and the severity. Drop all-pass and all-fail groups.
- **Numbers.** AgentNoiseBench-τ² Retail Avg@4: Qwen3-8B 31.80 → 36.40; 32B 37.72 → 43.20; clean τ² Retail 8B 46.49 →
  47.59. Ablation (Delivery, 8B, noisy): full 21.50; **noise on all rollouts 13.25, worse than no noise 16.00**; fixed
  difficulty 20.00. 32–64 H800 for 3–5 days per run.
- **Adoption.** Report a per-student noise gap Δ in `DifficultyRecord`; a task in band clean but far below band noisy is
  a robustness task, not a capability task. The Part 1 trainer needs subgroup-normalised advantages whenever a group
  mixes knob settings. P1.

### B6.7 Horizon and budget as their own knob
- **ScalingInter-RL** (AgentGym-RL 2509.08755): grow the max-turn cap h_1 < h_2 < … every Δ steps; on Deep Search a
  10-turn cap learned faster early and then collapsed, a 5-turn cap was stable but plateaued, the progressive cap beat
  both. ScalingInter-7B: WebArena 26.0 (GPT-4o 16.0), SciWorld 57.0 (base 1.5 → 50.5 for AgentGym-RL-7B).
- **Adoption.** A budget/horizon knob distinct from task hardness; F10's budget-2× probe is the measurement side;
  ScalingInter is a Part 1 schedule (P2). Record `context_rule` in the difficulty key (B1.11).

### B6.8 Per-student calibration: what changes for the 397B
- Difficulty is relative to the full configuration z (ACE); record it per `DifficultyRecord`.
- PAIRED: the regret reference must be stronger than each student; for the 397B use the oracle plus a stronger frontier,
  or regret saturates.
- AgentFrontier: re-choose the weak reference per student (the 35B can be the 397B's weak reference).
- MAGELLAN: warm-start a new student's predictor from another student's outcomes.
- Enhanced POET: minimal criterion against all past checkpoints, so tasks learned by an earlier checkpoint are not
  re-admitted.
- Daedalus calibrates to one solver; its tasks still ranked 9 models like the AppWorld test set (Kendall τ = 0.89, 34/36
  pairs), so calibrated self-generated tasks can serve as a ranking proxy.
- Mind the Gap: the judge must be stronger than the generator for its check to inform, which matters when the 397B is
  the student (B7.1).

### B6.9 Minor tips (B6)
- Loop outcomes too_easy / too_hard → targeted refinement in the same spirit, up to N_r = 5; the easier direction must
  reduce scope or remove one requirement, never name records or spell steps (Daedalus).
- Every generated World or task program should expose a difficulty argument, so one program yields a ladder; > 50% of
  GPT-4o terrain programs passed executability checks (Eurekaverse, ≈ $15 of API per 24 h run).
- Evolve from the environments that produced the best learner; soft-select among top policies to resist a noisy proxy
  (Eurekaverse).
- Do not scale reward by start distance (R³).
- Track sampled-difficulty mass over time as a curriculum diagnostic (PLR); test old checkpoints on late generations to
  confirm the frontier moved (ACCEL).
- A random generator must already produce mostly feasible instances (PAIRED).
- When in-distribution accuracy saturates, raise structural difficulty instead of training longer (RandomWorld).
- Hardness by itself is not value: for the task loop, a variant counts only if the student band and the signature
  change (ACCEL, Evol, OpenThoughts).

---
## B7. Self-improvement loops, their guards, and the statistics they need

### B7.1 The generation–verification gap bounds every model-judged loop (Mind the Gap)
- **Framework** (2412.02674). Generator f, proxy utility û_g from verifier g, reweighted distribution f[w(û_g)] ∝
  f·w(û_g) (rejection sampling or KL-regularised RL). Improvement J(f_{t+1}) − J(f_t) ≥ gap(f_t, g) − ε_update, with
  gap(f, g) = J(f[w(û_g)]) − J(f) and the **relative gap** normalised by headroom (U_max − J).
- **Findings.** Small models (Qwen 0.5B, Llama-2-7B) have non-positive gaps. CoT verification is more stable than
  multiple-choice. The relative CoT-Score gap grows roughly linearly in log pre-training FLOPs. Natural Questions: gap
  < 1% at all sizes. Sudoku: only about 70B models get a gap (Qwen-2-72B 8.82% → +16.99, relative 20.81). **Iterative
  self-improvement saturates to ≈ 0 gap in 2–3 rounds, independent of model size; pass@k at large k falls across rounds**
  (effective diversity lost, converging on wrong answers the verifier accepts). **A gold verifier shows no diversity
  loss**; a stronger but imperfect verifier still does. Round-1 gains partly come from answer-format adherence. Verifier
  methods disagree; AND-ensembles raise the gap; CoT-Score has a self-bias (mode = 10); thresholds transfer across sizes
  within a method.
- **Adoption** (§3.8). (1) A **GV-gap test for every model-judged component** in F6/F10/F11 on the frozen anchor set:
  filtering by the judge must raise precision against human labels; report the relative gap per judge version (this is
  what "agreement with the anchor set holds or improves" means operationally). (2) Cap loops that rely on model
  verification at about 2–3 rounds unless new executed information enters (new negatives, new traces). (3) Monitor
  large-k pass@k and signature diversity across rounds as the collapse alarm. (4) AND-ensembles of different-family,
  different-method judges for VALID/INVALID. P1.

### B7.2 Execution-driven component repair (WEFT)
- **Loop.** Rollouts → an attribution agent localises the fault (cross-view and cross-harness patterns; rubric-vs-exec
  disagreement flags the verifier) → targeted revision of the backend, the task graph and s0, or the verifier (policy
  failures leave the task alone) → re-instantiate, rerun construction checks, fresh rollouts. Three rounds.
- **Numbers.** Tool-error rate in selected trajectories 1.76% → 0.96%; +3.65 to +5.25 pp downstream at fixed tasks and
  budget.
- **Gaps (card).** No cost accounting; attribution precision unmeasured; "policy failure ⇒ keep task" rests entirely on
  the attribution agent; no anchor set guards its drift; no hackability audit.
- **Adoption.** The template for §3.8's World and verifier loops, with our guards added: anchor set, **frozen expected
  states** for blind tool tests, re-gating of every task in the family. P0 for routing, P1 for the loop.

### B7.3 Prompt-side learning artifacts: guideline banks and lesson writers (Daedalus)
- **Mechanism.** After any session with more than one variant, an LLM updates a compact list of general task-design
  principles from the progression of variants and their solver traces (add, modify, delete; never entity-specific). The
  Extractor that writes solver heuristics **does not see which conditions failed**, which pushes general lessons over
  task-specific fixes. One consolidation call merges heuristics into atomic, deduplicated items.
- **Numbers.** The guideline bank cut cost 12%; repeated-success validation added +4.5; solver traces +15.8. Counterfactual
  replay on 81 accepted tasks: heuristic in context raises MSR 66.7 → 77.0 (paired bootstrap CI [2.1, 18.5]). Beyond 90
  sessions MSR falls (140-session bank 53.7) because consolidation misstates heuristic scope. Cross-family banks helped
  every base agent; Qwen3.6-35B-A3B +29.3 with its own family's bank.
- **Adoption.** A guideline bank is a low-risk generator-loop artifact (prompt-side, versioned under `RULES_VERSION`-style
  recipe versions), behind §3.8's guards; watch scope loss at consolidation. Hide failed-check details from any lesson or
  repair writer. P1.

### B7.4 Evolving generator prompts (Promptbreeder) and its overfitting signal
- **Mechanism** (2309.16797). Units of two task-prompts + one mutation-prompt; population 50; binary tournament; 20–30
  generations; fitness = accuracy on a random 100-item training batch. Nine operators: zero-order generation (a reset
  from the problem description), first-order mutation, EDA over a population filtered so no two prompts exceed 0.95
  BERT-cosine **with no fitness shown** (the LLM copied the best), EDA rank-and-index (ascending order labelled
  "descending" to avoid copying), lineage, zero- and first-order **hyper-mutation** of the mutation-prompt, **Lamarckian**
  reverse-engineering of an instruction from a correct working, crossover (10%), few-shot shuffling. Removing
  self-referential operators hurt almost everywhere; GSM8K zero-shot 83.9% with the evolved prompt "SOLUTION".
- **Qwen-AgentWorld AutoResearch**: an optimiser agent revises system prompts against a judge's score on held-out real
  trajectories (10 iterations × 12 parallel runs seeded with different styles), then a human approves.
- **Adoption.** The mechanism the pipeline doctor would use for F3 or F5 prompts. Fitness = guarded signals only
  (anchor-set agreement, gate pass rates on a frozen slice, held-out family meta-evaluation), never student reward.
  Reject unintelligible winners (like "SOLUTION") as an overfitting flag. Hyper-mutation stays proposal-only until the
  anchor set and canaries exist. P2.

### B7.5 Open-ended archives that keep making new, learnable tasks
- **OMNI** (2306.01711): learnability alone gets stuck on "learnable but boring" variants. A model of interestingness,
  given mastered and failed tasks, down-weights boring ones (× 0.001, not deletion). Crafter (15 interesting + 90 boring +
  1,023 impossible): tasks learned uniform 4, LP 55, OMNI 82; interesting tasks learned 3 / 9 / 14. AI2-THOR: 0 / 2 / 13.
  The FM judge ≈ an oracle judge. In the infinite space, the FM proposes each task with **executable success predicates**.
- **OMNI-EPIC** (2405.15568): archive of descriptions + environment code labelled learned or failed; the generator sees 5
  nearest learned + 5 nearest failed; an environment generator writes reset/step/reward and a **separate `get_success`**;
  compile errors fed back ≤ 5 times; a post-generation check (GPT-4o) against the 10 most similar archive tasks admits only
  novel, creative, not-too-easy or useful tasks; failures go back for modification. **ANNECS-OMNI** = ANNECS + judged
  interesting. Ablations in B2.9. Fraction of attempted tasks learned decreases over time.
- **QDAIF / Rainbow**: per-bin elites; quality-only selection does not produce diversity (B5.5).
- **Caution** (Hivemind): an interestingness judge is exactly the miscalibrated-judge case among equally valid options;
  use it only to reject near-copies, never to rank valid alternatives; calibrate it on a densely labelled set. OMNI's
  authors flag Goodhart risk if a generator is trained to please the judge.
- **Adoption.** F11 archive score = learnability (band membership or LP) × interestingness, the latter a
  different-family judge voting "interestingly different / near-copy" against the cell's elites and the student's
  mastered tasks; normalise LP by G4's null-agent pass rate; bidirectional LP schedules re-measurement of forgotten tasks.
  **ANNECS-OMNI is the headline factory progress curve** (cumulative tasks novel by cell or signature, in band at
  admission, later solved by some checkpoint, judged interesting). Human spot check of top elites in the 10% κ queue. P0
  for the failed-examples context and the gate, P1 for LP.

### B7.6 Self-play stability and its failure modes
- **SSR game analysis**: a capable challenger has **dominant strategies**, e.g. tests that pass pseudo-randomly by a hash
  of the code, or obfuscating the codebase so nothing is fixable; and a **tunnel-vision** strategy, holding s at p* by
  tuning one parameter (operand length, obfuscation level, chain length), after which learning stops. Mitigations: ground
  the challenger in large, diverse real data; keep it close to its instruction-based start (shallow self-play); do not
  self-play natural language. Ablations over 1,231 tasks: **self-play > repair-only** (training on deduplicated valid bugs
  from earlier runs) **> injection-only** (degrades), so the online, evolving distribution matters. Solver-feedback reward
  was only slightly better than a consistency-only ±1 reward. Training on 23 benchmark-repo images did not beat more
  diverse repos. Instability (gibberish) at scale.
- **SPICE**: a fixed strong Challenger trains less than a co-trained one; R-Zero degrades after 5 iterations.
- **Agent0**: majority-vote labels; peak-checkpoint reporting.
- **Adoption.** A **knob-entropy monitor** on any injector or controller (fraction of removal vs history vs other
  operators, files touched) as the tunnel-vision alarm; injection locations sampled from the coverage grid; treat
  solve-rate feedback as weak and rely on consistency gates plus F9 measurement. In-training self-play is P2.

### B7.7 One-stage-at-a-time ablation as the generator-loop acceptance test (OpenThoughts-Agent)
- **Protocol** (2606.24855). Six SFT stages (sourcing, mixing, augmentation, task filtering, teacher, rollout filtering),
  each ablated alone at a fixed 10K-trajectory budget; student Qwen3-8B; 160 GH200-hours per ablation; selection by **mean
  per-benchmark z-score** over 3 core benchmarks with 3 stochastic re-runs per task; 5 OOD benchmarks sealed until the
  end. RL run-to-run noise ≈ 1.6 points in distribution and 2.0 OOD; RL sources spanned 7.6 points.
- **Findings beyond sources.** The strongest model (GPT-5.3-Codex) was a worse teacher than GLM-4.7-AWQ (≈ −5 pp TB2.0);
  BugPilot found GPT-5 trajectories with little reasoning text trained a student to 31.4% vs Claude's 49.87. Dropping
  < 5-turn traces was the best rollout filter (+3.5 pp avg at an equal ~145M token budget). LLM description filters +3 pp.
- **RL monitor signature.** The best RL source drove more exploration (think tokens +116%, tool calls +31%), flipped 18
  SWE-V tasks fail → pass, then collapsed (reward ≈ 0.13, about 80% timeouts) from over-exploration; checkpoint taken
  before.
- **Adoption.** §3.8 generator-loop test as written: a fixed small-student run on a frozen slice, mean z-score, sealed
  held-out families, and measured run-to-run noise before claiming an effect (P0). The collapse signature (rising think
  tokens + timeouts while reward plateaus) becomes a `train_monitor` lever (P1).

### B7.8 Agent-steered RL operations (MiniMax)
- M2.7 absorbs 30–50% of daily RL-iteration work (reads logs, diagnoses anomalies, edits configs) under human-steered
  goals; a 100-round autonomous scaffold-optimisation loop added loop detection and better parameters, +30% on in-house
  evals. Evidence is internal and qualitative.
- **Adoption.** Supports §3.8's "agent proposes, human steers" default and the human-review gate.

### B7.9 Statistics and calibration toolkit drawn from the cards
- **Anchor sizes and agreement** (beyond B3): QDAIF 280 annotations, human–AI 73% overall, 82% where humans agree, 95%
  on non-neutral labels, human–human 73%; Rainbow human–human 83%, human–GPT-4 81%, human–Llama Guard 78%, and
  CyberSecEval 100% malicious vs experts 80–94%; InsTag human–GPT κ 0.92 / 0.75 on 40 cases.
- **Paired and ranking statistics.** Paired bootstrap CI for counterfactual replay (Daedalus); Kendall τ for model-ranking
  agreement (0.89); Spearman for judge-ranking agreement (0.92–0.99, Qwen-AgentWorld); Welch t-test over 10 runs (PLR);
  rliable IQM and optimality gap (ACCEL); paired SE ≈ 2% on SWE-bench Verified (SSR).
- **Dense vs sparse labelling.** 25 raters per item revealed disagreement that sparse labels hide; judge calibration drops
  on similar-quality and high-disagreement subsets (Hivemind).
- **Noise-robust selection.** Replacement must beat the max of the incumbent's 5 most recent scores (Enhanced POET);
  soft selection among top policies (Eurekaverse); 3 judgements averaged for MT-Bench noise (InsTag).
- **Reporting cautions.** Peak-of-curve or peak-checkpoint reporting inflates gains (FEE, Agent0); single runs (SearchEyes
  ablations, Qwen-AgentWorld, TRACE main tables); small test sets where ±7 pp is one task (AgentBrew 28/28/21) or GAIA's
  ~100 IS items (WebShaper); WebArena 50 test tasks (AgentGym-RL).
- **Adoption.** Every judge on the anchor set gets a GV-gap, κ against humans and near-miss acceptance; every claimed
  pipeline effect needs run-to-run noise; model-ranking agreement (Kendall τ) is a cheap validity check for a new
  domain's self-generated tasks before a real test set exists.

### B7.10 Minor tips (B7)
- Keep a zero-order "reset from the problem description" operator to escape divergence (Promptbreeder).
- Self-referential improvement must be scored by the improvement it causes, not by its own judgement (Promptbreeder).
- Human approval of every evolved prompt variant before use, with disjoint variants per stage (Qwen-AgentWorld).
- The guideline bank and failed-examples context are the evidenced cheap loops; verifier rewrites stay proposal-only.
- A judge-rated quality score at the top of its range is not trustworthy (QDAIF > 0.995).

---
## B8. Pipeline architectures, costs, yields and scale; training results

### B8.1 Cost and yield table (every figure with its setting)
| Pipeline | Unit cost | Yield / funnel | Scale | Notes |
|---|---|---|---|---|
| Daedalus 2610.08048 | $109.7 generation for AppWorld (90 sessions) | 97.8% sessions accepted; 118 refinements; 13.6 solver rollouts per task | 90 sessions per benchmark | Most cost in Explorer + Judge |
| RandomWorld 2506.11045 | ≈ $150 total OpenAI | ≈ 11% of tool signatures kept; ≈ 60% env yield after dedup | 12,000 envs, 556 tools | Random-valued tools |
| AgentSynth 2506.14205 | ≈ $0.011/step; ≈ $0.60 per 6-subtask chain; ≈ $0.10 per task | Gen success 65/57/52% by difficulty | > 6,000 tasks | Humans: OSWorld 4.4 h/task, TheAgentCompany 17 h/task |
| AgentFrontier 2510.24695 | $0.24 refine + $0.54 amortised verify ≈ $0.78 per kept item | Strong-agent pass 33% | 12,000 RFT trajectories | DeepSeek API pricing |
| WEFT 2609.36887 | not reported | RL admission 1,954 / 5,000 (39%) | 8,172 MCPs; 41,695 atomic; 11,884 composed | — |
| DeepSeek-V3.2 2512.02556 | not reported | kept pass@100 > 0 | 1,827 envs, 4,417 general tasks; ~85K agent prompts | PARTIAL |
| ARE 2509.17158 | ≈ 400K raw + 800K structured tokens per universe | 4-annotator QA per scenario | 10 universes; 1,120 scenarios | Human-built oracles |
| Learn-by-interact 2501.10893 | many LLM calls, no $ | Filters keep 25–60% of pairs | ≈ 10K pairs per benchmark | Committee only |
| Qwen-AgentWorld 2606.24597 | not reported | SFT 7,094 / 10,250 kept (69.2%) | 10M+ CPT trajectories; 92,308 RL turns | — |
| SearchEyes 2607.05943 | 8×H20 SFT; 64 H20 RL 200 steps | SFT accept on exact answer ≈ 42% | 22K questions; 30K trajectories | — |
| WebShaper 2507.15061 | not reported | 18K seeds → 5,000 trajectories | — | — |
| Self-Instruct 2212.10560 | ≈ $600 total | 54% all fields valid | 52,445 instructions | — |
| Magpie 2406.08464 | ≈ $0.12 (Air) and $1.1 (Pro) per 1K instances | Filtered 300K beat raw 4M | 3M + 1M | Instruction step ≈ 3% of compute |
| Eurekaverse 2411.01775 | ≈ $15 API per 24 h run on 8 A6000s | > 50% of terrains executable | 5 iterations × 8 runs | — |
| Nemotron-Terminal 2602.21193 | not reported | no oracles generated | 226,313 adapter + 264,207 synthetic | SFT only |
| OpenThoughts-Agent 2606.24855 | 160 GH200-h per 10K ablation; RL ≈ 46 h on 24 A100 | — | > 100 ablations; 100K traces | — |
| SSR 2512.18552 | 512 H100 per run; 150 steps ≈ 2.5B tokens | — | — | Same images as CWM's RL |
| NoisyAgent 2605.27209 | 32–64 H800 × 3–5 days | — | — | Judge rewards |
| TRACE 2604.05336 | 4–8 A100 per adapter; ≈ 5 min per analysis pass | 2–4 capability envs per benchmark | 5,120 rollouts to 48.2 on τ² | — |
| AgentBrew 2609.05837 | 9 h on 4×H100 for credit | all trajectories kept except T_max | ≈ 8.5k trajectories | — |
| Enhanced POET 2003.08536 | ≈ 12 days on 750 CPU cores per run | — | 60,000 iterations | — |
| CUA-Sandbox / MegaMCP | see B1.6 | — | — | Infra savings, not synthesis |
- **Reading.** Unit costs span $0.10 per task (AgentSynth, LLM judge) to $0.78 per kept item (AgentFrontier, two-model
  sandwich); none includes a student band measurement, which dominates Crucible's cost (§3.10). AgentFrontier's
  "refine + amortised verify" is the right shape for §3.10's formula. Planning priors from these cards: > 50%
  executability of generated environment code (Eurekaverse), about 60% yield after an instruction check (RandomWorld),
  about 39% RL admission after a judge–executor consistency filter (WEFT), 33% strong-agent solvability after escalation
  (AgentFrontier).

### B8.2 Architecture patterns worth copying
- **WEFT** is the only end-to-end loop in these cards that goes World → certified tasks → composition → grounded state →
  views → deterministic verifier with rubric audit → attribution-routed revision → SFT (prefix-preserving rejection
  sampling: retry only the failed atomic turn from the restored snapshot, k = 3) → RL filter → atomic-turn GRPO.
- **Role decompositions** to reuse as swarm roles: AgentSynth (proposer, executor, verifier, reviser, follow-up proposer,
  summariser); Daedalus (Surveyor, Explorer, Solver, Judge, Extractor, Consolidator; auxiliary agents use the larger
  model of a family, the solver the smaller); ARE (Apps, Environment, Events, Notifications, Scenarios, Verifier).
- **MiniMax** samples every task under multiple perturbed scaffolds; its stacks (SWE, AppDev, Terminal-Gym, Cowork,
  Finance) report no yields or costs.

### B8.3 Training results on Qwen3.x-30/35B-A3B and nearby students
| Paper | Student | Training | Result |
|---|---|---|---|
| TRACE 2604.05336 | Qwen3-30B-A3B | Per-capability LoRA RL on synthesised envs + MoE router | τ² 32.9 → 48.2 (GRPO on target 37.8; AWM 38.4); SWE-bench Verified 26.0 → 41.0 |
| TRACE | Qwen3.6-27B | same | τ² 50.0 → 59.1; SWE-V 68.0 → 73.2 |
| WEFT 2609.36887 | WEFT-35B-A3B | SFT + atomic-turn GRPO on synthesised MCP worlds | Toolathlon-Verified 45.99; AutomationBench 26.50 (WEFT-14B beats Agent-World-14B by +6.41 BFCL V4, +2.23 τ², +12.27 Claw-Eval) |
| Qwen-AgentWorld 2606.24597 | Qwen3.5-35B-A3B-SFT | Language-world-model RL as warm-up (no tool calls) | Terminal-Bench 2.0 33.25 → 39.55; SWE-V 64.5 → 67.9; SWE-Pro 42.2 → 47.4; Claw-Eval +11.3; BFCL v4 +9.0 |
| Qwen-AgentWorld | 35B (LWM) | LWM training | AgentWorldBench 47.73 → 56.39; 397B LWM 58.71 vs GPT-5.4 58.25 |
| Qwen-AgentWorld | 35B | Fictional-world search sim RL | WideSearch F1-item 34.0 → 50.3 (real-search RL 45.6) |
| AgentFrontier 2510.24695 | Qwen3-30B-A3B | RFT on ZPD data (12K trajectories); + CPT 50B tokens | HLE text-only 25.67 (TaskCraft 19.87, MegaScience 20.15, MiroVerse 19.64 at equal budget); with CPT 28.6 |
| Daedalus 2610.08048 | Qwen3.6-35B-A3B | Memory bank, no weight updates | +29.3 with its own family's bank |
| EnvACE 2608.06197 | (Qwen3-30B-A3B as training judge) | — | Not a student result |
- **Reading.** The two RL-on-synthetic-world results at this size (TRACE, WEFT) both used executable state-based rewards
  and targeted or composed tasks. Qwen-AgentWorld's warm-up result is a candidate Part 1 auxiliary objective, not a task
  source.

### B8.4 Other training results that used synthetic environments or tasks
- **DeepSeek-V3.2**: RL only on synthetic general-agent tasks improved τ²-Bench, MCP-Mark and MCP-Universe; code + search
  RL alone did not.
- **Nemotron-Terminal** (SFT): TB2.0 Qwen3-8B 2.47 → 13.0; 14B 4.04 → 20.2; 32B 3.37 → 27.4 (above Qwen3-Coder-480B 23.9).
  Per-category at 32B: Data Querying 0 → 60, Model Training 0 → 50, Data Processing 5 → 50, Debugging 0 → 33.3; Scientific
  Computing and Mathematics stay ≈ 0. **Filtering hurt** (no filter 12.4 vs complete-only 6.74 vs success-only 5.06; an
  SFT finding, not transferable to rewards). Long context (64K) did not help. **Mixed single-stage beat two-stage: 13.03
  vs 10.39.**
- **OpenThoughts-Agent** final 32B: TB2.0 26.2, OT-TBLite 41.3, SWE-V-100 55.7; 7-benchmark avg 44.8 vs Nemotron-Terminal-32B
  40.9. SFT "undertrained" checkpoints benefit more from RL; base Qwen3-8B could not.
- **SSR** (CWM-sft 32B): +10.4 SWE-bench Verified and +7.8 SWE-Bench Pro over the trajectory, beating a human-data RL
  baseline with the same images, without ever seeing an issue.
- **BugPilot** (Qwen3-32B): FrogBoss 54.6 pass@1, 67.4 pass@3; FrogMini (14B) 45.27.
- Results for NoisyAgent, FEE, AgentGym-RL, QuestA, R³, SPICE, Agent0, Magnet, RandomWorld, OS-Genesis and
  Learn-by-interact are in B2 and B6.
- **WebSailor** (72B): BrowseComp-en / zh / Xbench-DeepSearch / GAIA 12.0 / 30.1 / 55.0 / 55.4; an RFT cold start
  (about 2K trajectories with > 5 tool calls) converged higher than direct RL; RL raised pass@1 much more than pass@3.
- **SearchEyes-27B** avg 68.1 over six benchmarks vs OpenSearch-VL-32B 61.9.
- **AgentBrew** (Qwen3-32B, weighted SFT on relabelled logs): Acc 9.5 → 18.2, above Qwen3-235B 15.9; PMI credit > InfoPO >
  uniform > LLM self-score; TOUCAN-14B (1.5M synthetic trajectories) and AWM-14B gave little or negative transfer at 14B.
- **Prismatic** PrismMath-7B 75.25 vs R1-Distill-7B 72.80.

### B8.5 Trainer items surfaced by these cards (Part 1, listed for completeness)
- **Atomic-turn GRPO** (WEFT): G segments from an identical restored state per turn, one passing segment continues;
  beats trajectory-level GRPO on τ² by +5.69 (8B) and +4.43 (14B); trajectory-level GRPO regressed τ² below SFT.
- **Subgroup normalisation** when a group mixes clean and perturbed rollouts (NoisyAgent); **constant enrichment within a
  group** (FEE).
- **DUPO** (WebSailor): fill zero-variance slots by duplicating other same-batch groups with variance, 2–3× faster than
  DAPO resampling.
- **Off-policy sequence masking, keep routing, keep sampling mask, unbiased K3 KL** (DeepSeek-V3.2); relevant to the MoE
  student.
- **Windowed FIFO** (W = 0.3N) against easy-first distribution shift; mixed-domain RL in every stage (MiniMax).
- **One prediction turn per trajectory** in RL pools (shared-prefix expansion collapsed reward variance; Qwen-AgentWorld).
- **Per-role baselines** when one policy plays several roles (EnvACE, SPICE: mean-centred, no std).
- **Advantage weighting by label confidence and looser upper clip on ambiguous items** (Agent0 ADPO, −1.9 when removed).

---
## B9. Corrections and contradictions

### B9.1 Corrections to `framework-plan.md`
| Plan location | Plan says | Cards say | Source |
|---|---|---|---|
| §3.5 item 1 | Nemotron-Terminal "9 or 10 domains [uncertain count]" | **9 domains.** The body lists dependency management where App. Table 10 names system administration: a naming inconsistency, not a tenth domain | 2602.21193 |
| §3.5 item 3 | Verbalized Sampling: "keep the least typical valid one" | The paper **samples from or thresholds** the verbalized distribution (tail thresholds). Novelty should be decided by F7's measured signature, not the stated probability | 2510.01171 |
| §3.5 item 3 | Personas as a grounding input for diversity | Persona variety ranked datasets **backwards** in Prismatic (54.77 vs 38.15 OOD); constraints shrink persona-induced diversity (Persona Hub). Personas buy realism; diversity comes from cell, World slice and signatures | 2505.20161; 2406.20094 |
| §3.5 item 5 | AgentFrontier's ≥ 0.7 as a starting embedding threshold | It is a **reranker score over questions**, not an embedding cosine; calibrate on labelled Crucible task pairs before relying on it | 2510.24695 |
| §3.5 item 7 | "Mixing the top 16 sources hurt every benchmark compared with the top 8" | At 100K (32B), top-16 is worse than **both top-8 and top-4**; at 10K, **top-4 ≈ top-2 is best** (top-16 −0.10 z, top-32 −0.48, top-1 −0.57) | 2606.24855 |
| §3.6 hint removal | MiniMax-M2 "[uncertain scale]" | The report gives **no task counts or yields** for Terminal-Gym and no data-pipeline ablations; the flag stays | 2605.26494 (PARTIAL) |
| §3.6 band centre | "SSR's optimum is about 0.2 for group size 8" | Correct as stated, but it is the optimum of SSR's **injector reward function** from its game analysis (App. A), and solver-feedback reward was only slightly better than consistency-only. SPICE's head-to-head ranks a variance reward peaking at 0.5 first. Treat 0.2 vs 0.5 as an open measurement | 2512.18552; 2510.24684 |
| §3.6 ladders | "Late start states (R³)" | Add: **mix start positions, do not stage**; equal reward per start; keep the hardest starts | 2402.05808 |
| §3.6 ladders | "Partial-solution hints (QuestA)" | Add: the hint effect is steep and narrow (Partial-25 looks almost like no hint); calibrate per family with a histogram | 2507.13266 |
| §3.6 "iterative harder" (DeepSeek-V3.2) | Adopted as a design row | DeepSeek's agent edits solution **or** verifier until they agree and filters by policy pass@100; adopt only with a different-family verifier writer and re-run negatives each round | 2512.02556 |
| §3.7 zero-solve rule | Requires 2 alternatives or an agentic audit | Consistent with AgentFrontier's 0/3 → human review and PAIRED's "frontier-also-fails is not high regret" | 2510.24695; 2012.02096 |
| §3.8 task loop | "Signature differs from the parent's on the knob changed" | Strongly supported: Evol-Instruct's hardening is textual, OpenThoughts' augmentations never beat untouched tasks at 10K, AgentSynth's direct-hard survivors are easy | 2304.12244; 2606.24855; 2506.14205 |
| §3.10 mix | "Nemotron-Terminal found mixed beat two-stage, 13.03 against 10.39" | Confirmed (SFT, Qwen3-8B) | 2602.21193 |
| §3.6 hardness proxy | "+3 pp from GPT-5 token spend" (OpenThoughts) | Confirmed (≈ +3 pp, task-filter stage at 10K) | 2606.24855 |
| G12 hackability | Attacker with K tries, 3 rounds | Add a Rainbow-style archive over check kind × exploit style, reused as F5 negatives | 2402.16822 |
| G6 / F5 | Inverse mutation from SSR | SSR itself shows the full oracle tests to the solver; Crucible must add **hidden** checks | 2512.18552 |

### B9.2 Errors and inconsistencies inside papers
- **NoveltyBench**: equivalence-classifier accuracy is 79% / F1 0.811 in the body but 71.0% in App. A.4 at the chosen
  threshold.
- **InsTag**: the DBSCAN similarity threshold (0.05 as stated) is under-specified.
- **Nemotron-Terminal**: the domain list differs between body and appendix (above). Its "3–5 skills per task" claim is
  never measured.
- **TRACE, NoisyAgent, Magnet, Learn-by-interact**: benchmark material inside synthesis (header). TRACE's "success depends
  primarily on c" is asserted, not tested; NoisyAgent asserts solvability under misleading noise without a check;
  Learn-by-interact's committee is same-family (Claude judges Claude).
- **SearchEyes**: VisSearch Bench is built by the same pipeline; training World is Wikipedia while evaluation uses live
  Serper search (sim-to-real gap unmeasured).
- **CUA-Sandbox**: systematic uplift over Docker in all 12 pairs suggests a flaky baseline.
- **FEE, Agent0**: peak-of-curve or peak-checkpoint numbers in the main tables.
- **WEFT**: Pearson/Spearman thresholds for the RL filter are not stated; attribution precision is not measured.
- **AgentBrew**: test sets of about 28/28/21 tasks, so ±7 pp is one task.
- **Daedalus**: an early near-duplicate filter on expected tool paths was used once and is not part of the method; the
  calibration target is a recoverable-failure band, not a p̂ band.
- **EnvACE**: rehearsal fidelity against real tool returns is never measured.
- **OMNI-EPIC**: the checker and world come from the same model; the 72.7% human agreement is the only check.
- **Text world-model survey**: many 2026 citations are not verified by the card.

### B9.3 Contradictions across papers, with how to read them
- **Bigger models: less or more diverse?** NoveltyBench and Hivemind: larger models give fewer distinct outputs (Llama-3.2-1B
  6.74 vs Llama-3.1-405B 3.20 distinct of 10). 2410.15226: stronger generators produce more diverse data. Verbalized
  Sampling: larger models gain 1.5–2× more diversity from VS. Reading: raw sampling diversity falls with scale and
  post-training, but strong models respond better to structural prompting (VS, in-context regeneration). The swarm should
  measure `distinct_k` per model under the F3 prompt actually used.
- **Does mixing generators help?** 2410.15226: mixing helps slightly. Hivemind: cross-family similarity 0.71–0.82. Both
  are compatible with "mixing is not a mechanism; measure induced coverage" (ACE).
- **Short or long trajectories?** Learn-by-interact: short sub-trajectories (< 5 steps) help most for relabelled data
  (mixing lengths best). OpenThoughts: dropping < 5-turn traces is the best rollout filter. WebSailor keeps only > 5 tool
  calls. Reading: short units help as relabelled *tasks*, long traces help as *SFT demonstrations*; different uses.
- **Tasks or tools?** RandomWorld: task count matters more than tool count in distribution, and fewer tools helped
  ToolQA-Hard. The plan cites SynthAPI (2610.37648, abstract) for tool and DB diversity beating task count at larger
  budgets. Different regimes (in-distribution, small budgets vs larger budgets); keep it a measurement (§3.10 M3).
- **Augmentation: fails or works?** OpenThoughts: augmentations fail at 10K, yet rewriting 997 Tezos problems into > 21K
  surface forms kept improving at 100K. Effects are scale-dependent; for RL, the behaviour-signature rule still decides.
- **Filters: help or hurt?** Nemotron-Terminal: success-only filtering hurt SFT (12.4 → 5.06), and AgentBrew: lower-quality
  data still helped weighted SFT. Both are SFT findings; for RL rewards the cards insist on executed checks. Do not
  transfer "no filter" to reward design.
- **Hints: protect or erode skills?** QuestA: hint-trained models keep pass@k. FEE: guidance-trained agents ask for
  clarification less (−4 to −7 pp). Reading: hints in math remove search cost; hints in interactive tasks can remove the
  question-the-input behaviour. Measure infeasible and clarification tasks before and after scaffolded training.
- **Strongest model as teacher?** OpenThoughts (GPT-5.3-Codex worse than GLM-4.7-AWQ) and BugPilot (GPT-5 trajectories
  31.4% vs Claude's 49.87%) agree: visible reasoning text between tool calls matters more than solve rate.
- **Self-play reward signal.** SPICE's ablation shows the proposer reward shape matters (variance best, 1 − p worst); SSR
  found solve-rate feedback only slightly better than consistency-only. Reading: with G = 8, the solve-rate estimate is
  coarse; reward shape matters more in QA self-play than in SWE self-play under strong consistency gates.

### B9.4 Notes for other extracts and the plan's notes
- The ARE verifier's 450-trajectory validation is the only anchor-set size with a reported agreement in these cards; the
  §3.8 "30–50 episodes per lane" target is far below it. Consider 450 as the upper reference for a mature lane.
- Daedalus shows Qwen3.6-35B-A3B benefits from prompt-side memory banks (+29.3), relevant if Part 1 tests a memory or
  context-engineering baseline against RL.
- No card in groups 03–04 reports a band schedule per student size; the 35B vs 397B question remains our own
  measurement.
