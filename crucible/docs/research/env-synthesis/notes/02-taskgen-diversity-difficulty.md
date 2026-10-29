# 02 Task generation at scale: diversity, difficulty, generator-verifier loops, pipelines, traces-to-tasks

Research note, 2026-10-29. Scope: what the literature and the published lab pipelines say about generating
500 to 1000 diverse, verifiable, intent-aligned RL tasks in any domain from a simulated World (databases, websites,
people, time, tools), and about a task factory that keeps improving itself. Not committed.

**Method.** Every arXiv id below was checked against its arxiv.org/abs page (title and date) on 2026-10-29. Mechanisms
and numbers come from the abstracts or the papers' HTML full text. Where a paper is internally inconsistent, or a
number could not be confirmed, the row says so (marked **[uncertain]**). Summaries are in my own words. 113 arXiv sources plus Harbor
are cited; 8 PDFs are in `crucible/docs/research/papers/` (list at the end).

**One-paragraph answer.** The systems that work share five habits. (1) They define diversity as coverage of explicit
axes (a taxonomy, a tool graph, skill combinations, real seeds), not as "ask the LLM for something different", and
they ground every proposal in exogenous randomness (a world-state slice, a document, a repo, a persona). (2) They build
tasks forward from a known ground truth, so the verifier is cheap and the task is "hard to solve but easy to verify"
(DeepSeek-V3.2's phrase). (3) They check each task with executable evidence: a reference solution passes, known-bad
solutions fail, the task fails in the start state, and a solver with no tools or no hints cannot shortcut it.
(4) They calibrate difficulty against the actual student, aiming at a pass rate of roughly 0.2 to 0.5, and they
re-measure as the student changes. (5) They turn failures into new tasks: failed tasks seed new variants, failed
rollouts become harder start states, and weak skills get more tasks. What fails is ungrounded self-play (label drift,
then collapse), generators rewarded only on solver pass rate (they learn to make flaky or obfuscated tasks), strict
success-only trace filtering, and "harder" rewrites that add no new structure.

---

## Q1. Forcing diversity: guaranteeing coverage of all scenario types

### State of the art

Three families of mechanism, in order of how much control they give:

1. **Explicit coverage structures.** A taxonomy or skill tree is written first, and generation is conditioned on its
   leaves (GLAN's discipline → subject → syllabus → concept tree; Nemotron-Terminal's 9 domains with 3–5 primitive
   skills combined per task; Endless Terminals' category × complexity × scenario sampling; MiniMax-M2's PR-type and
   occupational sub-category taxonomies; AgentScaler's tool graph split into 1,000+ domains by community detection).
   Coverage is then countable: every leaf or cell has a fill count.
2. **Quality-diversity archives.** MAP-Elites keeps the best solution per cell of a grid of user-chosen behaviour
   dimensions. QDAIF and Rainbow Teaming port this to LLM generation, with an LLM both mutating candidates and
   judging quality and cell membership. Open-endedness work (POET, Enhanced POET, OMNI, OMNI-EPIC) adds two filters on
   top: learnability for the current agent, and "interestingness" judged by a foundation model so that trivial
   variants of solved tasks are not counted as new.
3. **Implicit diversity from seeds and personas.** Persona Hub conditions generation on 1B web-mined personas;
   Magpie samples queries from an aligned model's own prior; Self-Instruct and Evol-Instruct grow a pool from seeds
   and dedup by surface similarity. These give volume and breadth cheaply but no coverage guarantee.

**Measurement.** Practice uses several measures at once: per-cell counts against the taxonomy (the only one that
proves coverage), embedding dispersion such as the Vendi score (exponential of the entropy of a similarity matrix's
eigenvalues, reference-free), tag-based counts (InsTag's 6.6K open-set tags), LLM cluster scoring (the "LLM
cluster-agent" metric of 2410.15226), and gradient-space diversity (G-Vendi, Spearman ρ ≈ 0.9 with out-of-distribution
accuracy over 300+ training runs in Prismatic Synthesis). Dedup is MinHash for near-copies (MiniMax-M2 AppDev),
embedding nearest-neighbour thresholds (SemDeDup; AgentFrontier drops a question at reranker similarity ≥ 0.7), and
n-gram/BLEU clustering (R-Zero penalises proposals in large BLEU clusters at τ = 0.5).

**Mode collapse in the generator is real and cross-model.** NoveltyBench finds 20 leading models far less diverse than
human writers, and larger models in a family often less diverse than smaller ones. Artificial Hivemind (26K open-ended
queries, 31,250 human annotations) finds both intra-model repetition and strong inter-model homogeneity: different
model families give strikingly similar outputs. Verbalized Sampling traces collapse to typicality bias in preference
data and recovers diversity by asking for a list of candidates with stated probabilities (1.6–2.1× diversity in
creative writing; more capable models gain more). For a swarm of DeepSeek, GLM and Kimi generators, the Hivemind
result means mixing families does not by itself give diversity.

**Diversity measurably pays.** OpenThoughts-Agent found task-description diversity to be the bottleneck: more
rollouts per task plateaued from 31.6K to 100K examples, while rewriting 997 base tasks into 21K surface forms kept
improving. SWE-smith found performance rising roughly with the log of the number of training repositories. But
OpenThoughts-Agent also found that mixing the top 16 task sources hurt every benchmark compared with the top 8:
diversity of low-quality sources is not free.

### Papers

| arXiv | Paper | Mechanism | Scale | Key numbers |
|---|---|---|---|---|
| 1504.04909 | MAP-Elites | Archive of best solution per cell of chosen behaviour dimensions | Generic | Origin of the QD archive pattern |
| 2310.13032 | QDAIF | LLM mutates and LLM judges quality and diversity axes | Creative writing | Covers more of the target space than non-QD controls |
| 2402.16822 | Rainbow Teaming | QD archive (risk category × attack style) for adversarial prompts | Llama 2/3 targets | Attack success > 90% on all tested models; hundreds of diverse prompts |
| 1901.01753 | POET | Co-evolves environments and agents; transfers solutions between environments | 2D walker | Founding open-ended env generator |
| 2003.08536 | Enhanced POET | Domain-general novelty measure (PATA-EC) and a universal progress measure | 2D walker | Removes POET's domain-specific novelty metric |
| 2306.01711 | OMNI | FM as model of interestingness on top of learning-progress filtering | Crafter-like task spaces | Learnable but boring tasks are filtered out |
| 2405.15568 | OMNI-EPIC | FM writes environment and reward code for the next learnable and interesting task; archive of past tasks | Simulated robotics | Tasks as code: unbounded task space |
| 2309.16797 | Promptbreeder | Evolves task prompts and the mutation prompts that mutate them | Reasoning, classification | Self-referential mutation operator |
| 2402.13064 | GLAN | Taxonomy of human knowledge → subjects → syllabi → concepts → instructions | All disciplines | Taxonomy is the only input; no seed examples |
| 2406.20094 | Persona Hub | Condition generation on 1B web-mined personas | 1B personas | Personas as "distributed carriers" of perspective |
| 2212.10560 | Self-Instruct | Seed pool, generate, filter by surface similarity | GPT-3 | +33% absolute on Super-NaturalInstructions |
| 2304.12244 | WizardLM / Evol-Instruct | In-depth and in-breadth rewrite operators raise complexity | Open-domain instructions | Used later by WebRL for task evolution |
| 2406.08464 | Magpie | Sample user queries from an aligned model's template prefix | 4M generated, 300K kept | Diversity from the model's own prior |
| 2308.07074 | InsTag | Open-set intent tags define diversity and complexity | 6.6K tags; 6K selected samples | Tag-selected 6K beats larger SFT sets on MT-Bench |
| 2210.02410 | Vendi Score | exp(entropy of similarity-matrix eigenvalues) | Any domain | Reference-free diversity metric |
| 2505.20161 | Prismatic Synthesis (G-Vendi) | Diversity in proxy-model gradient space; synthesise toward sparse regions | 300+ runs | ρ ≈ 0.9 with OOD accuracy |
| 2410.15226 | Diversity of synthetic data | LLM cluster-agent diversity metric | 350M, 1.4B models | Diversity correlates with pretraining and SFT results |
| 2303.09540 | SemDeDup | Embedding-cluster semantic dedup | LAION, C4 | Removes 50% of LAION with minimal loss |
| 2504.05228 | NoveltyBench | Measures distinct, high-quality outputs per prompt | 20 models | Larger models often less diverse |
| 2510.22954 | Artificial Hivemind | Intra-model repetition and inter-model homogeneity | 26K queries, 31,250 annotations | Judges and reward models miscalibrated on divergent-preference items |
| 2510.01171 | Verbalized Sampling | Ask for k candidates with probabilities | Training-free | 1.6–2.1× diversity in creative writing |
| 2010.03934 | Prioritized Level Replay | Replay levels by estimated learning potential | Procgen | Emergent curriculum from replay priorities |
| 2012.02096 | PAIRED | Adversary rewarded by regret (antagonist minus protagonist) | Grid, driving | Fixes minimax's unsolvable-worst-case failure |
| 2203.01302 | ACCEL | Evolve levels by small edits; keep high-regret ones | Mazes, walkers | Complexity compounds by editing |
| 2502.07709 | MAGELLAN | LLM agent learns to predict its own competence and learning progress over goals | Large evolving goal space | Only method to master the whole goal space in their test |
| 2411.01775 | Eurekaverse | LLM writes environment code as a curriculum | Quadruped parkour | Environment-curriculum by code generation |
| 2602.21193 | Nemotron-Terminal (Terminal-Task-Gen) | Seed-based and skill-based generation; 3–5 skills per task; domain prompt modules | 264K synthetic tasks | 9 domains listed (Table 10 adds a 10th) **[uncertain count]** |
| 2601.16443 | Endless Terminals | Sample category × complexity × scenario context per prompt | 3,255 tasks | About half of candidates fail solvability |
| 2606.24855 | OpenThoughts-Agent | Ablates 95 task-generation strategies and source mixing | 100K traces | Rollouts per task plateau 31.6K→100K; top-16 mix worse than top-8 |
| 2509.13311 | AgentScaler | 30K+ APIs → tool graph → Louvain → 1,000+ domains, each a DB-backed environment | 1,000+ domains | Initial DB state varied for diversity |

### What works

- **Coverage defined by structure the generator cannot fake.** Taxonomy cells, tool-graph communities and skill
  combinations are countable, and a missing cell is visible. Nemotron-Terminal's skill-based and seed-based halves
  are about equal in size (139,841 vs 124,366) and are trained mixed.
- **Grounding each proposal in exogenous material.** SPICE (corpus documents), SSR (real repos), AgentFrontier
  (document triplets), Endless Terminals (random axis draws), AgentScaler (random tool-graph walks and varied initial
  DB state). Grounding is what SPICE credits for sustained improvement over ungrounded self-play.
- **Archive with an interestingness gate.** OMNI-EPIC's archive plus an FM judge that rejects near-duplicates of
  solved tasks is the general pattern for "new" rather than "different".
- **Several dedup layers.** MinHash for copies, embedding or reranker similarity for paraphrases, BLEU clustering as a
  generator penalty.

### What fails

- Asking the generator to "be diverse" with a fixed prompt. AZR's ablation: replacing dynamic in-context references
  from the task buffer with a fixed prompt cost about 5 math points.
- Relying on family mixing (Hivemind) or on larger generators (NoveltyBench) for diversity.
- Embedding diversity alone: it measures surface text, not what the task exercises. Two tasks that touch the same
  tables with the same tool sequence are the same task to the learner.
- Adding many weak sources (OpenThoughts-Agent top-16).

### Design recommendations for a domain-generic factory

1. **A `DomainSpec` names its coverage axes up front**, as a MAP-Elites grid: skill leaves (from a skill tree),
   scenario type, world facet (which tables, sites, people, clock events), and difficulty knob settings. Coverage
   is the fill fraction of reachable cells, reported per batch. The axes are domain data, the archive code is generic.
2. **Measure cell membership from behaviour, not from the generator's label.** Compute a behaviour signature from the
   reference solution's run in the World: tools called, state rows read and written, entities and people touched,
   clock events crossed, horizon length. The generator's claimed cell is checked against it. This makes diversity
   and intent auditable and blocks label drift.
3. **Condition every proposal on a sampled cell plus a sampled world slice and persona**, and show the K nearest
   existing tasks in that cell as "do not repeat" references (AZR's buffer references). Ask for several candidates
   at once with stated likelihoods (Verbalized Sampling) and keep the least typical valid one.
4. **Four diversity meters on every batch**: cell fill and Gini over cells; Vendi on description embeddings; Vendi on
   behaviour signatures; per-generator-family entropy over cells over time. A falling family entropy, or a rising
   median nearest-neighbour similarity, is the mode-collapse alarm. Response: rebalance cell quotas toward empty
   cells and raise the repetition penalty, not "rewrite the prompt".
5. **Dedup on three levels**: MinHash on text, embedding or reranker similarity (≥ 0.7 as in AgentFrontier is a
   reasonable start), and identical behaviour signatures.
6. **Keep source quality ahead of source count.** Rank sources and generators by the downstream value of their tasks
   (Q4), and drop the tail (OpenThoughts-Agent).

---

## Q2. Difficulty calibration and curriculum

### State of the art

**The objective.** With a binary reward and group-relative advantages, an all-pass or all-fail group carries no
gradient. Online Difficulty Filtering proves that expected policy improvement is lower-bounded by the variance of
task-level success, p(1−p), so intermediate tasks maximise learning (up to +12% in under half the GRPO steps).
"Learning to Reason at the Frontier of Learnability" applies sampling for learnability to LLM RL and shows many
questions sit at all-pass or all-fail throughout PPO/VinePPO training. DAPO's dynamic sampling over-samples and drops
groups whose accuracy is exactly 0 or 1. Kimi k1.5 estimates difficulty from an SFT model's pass rate over 10 samples,
samples proportionally to 1−s_i, and removes hackable prompts that a model can guess without reasoning within 8
tries. Kimi K2 keeps RL prompts of "moderate" difficulty by the SFT model's pass@k.

**Where the target sits.** Not always at ½. SSR's injector reward peaks at a low but nonzero solve rate and its
appendix estimates the optimum at about 0.2 for group size 8. GenEnv's environment reward is a bell centred at
α = 0.5. R-Zero keeps questions with |p̂ − ½| ≤ 0.25 (3 to 7 of 10 samples agree). WebRL keeps evolved tasks whose
critic score from the start state is between 0.05 and 0.75.

**Curriculum controllers.** SEC treats task categories as bandit arms rewarded by mean absolute advantage. DUMP uses
UCB over data distributions by advantage magnitude. AdaRFT moves a target difficulty with recent reward. PLR (from
classic RL) replays levels by learning potential. MAGELLAN learns to predict its own learning progress across goals,
which is the cheap-predictor idea for long tasks.

**Self-evolving curricula.** WebRL generates new tasks from the agent's own failures (Llama-3.1-8B 4.8% → 42.4% on
WebArena-Lite). AZR's proposer reward is 0 when the solver never succeeds and 1−r̄ otherwise. R-Zero trains a
separate Challenger on an uncertainty reward minus a repetition penalty. SCA has the agent explore tools, then write
Code-as-Task tasks. SPIRAL gets a curriculum from self-play in zero-sum games (up to 10% across 8 reasoning
benchmarks). SWE-RL uses rule-based similarity to real PR patches (Llama3-SWE-RL-70B 41.0% on SWE-bench Verified).
SSR self-plays bug injection and repair (+10.4 Verified, +7.8 Pro). Environment Tuning uses a four-stage curriculum
plus hint-giving environment feedback. Agent-R1 (2511.14460), named in the brief, is a modular agentic-RL training
framework, not a curriculum or task-generation method.

**Knobs that change difficulty while keeping the verifier exact.** From the papers:

| Knob | Where it is shown | Effect |
|---|---|---|
| Number of composed subtasks | AgentSynth (2506.14205) | Frontier success 18% at level 1 → 4% at level 6 |
| Hop depth and width | TaskCraft (2506.10055), WebShaper (2507.15061) | Depth extension pass rate 41–51% per attempt; controlled multi-hop structure |
| Information obfuscation | WebSailor (2507.02592) | High-uncertainty questions; entity names blurred |
| Hint removal | MiniMax-M2 Terminal-Gym (2605.26494) | Prefers variants with fewer hints and lower zero-shot pass rate |
| Escalation along named axes until a no-tool model fails | AgentFrontier (2510.24695) | 7.81 refinement rounds on average, cap 30 |
| Bug count, linked bugs, higher-order bugs | SWE-smith combine-bugs (2504.21798); SSR (2512.18552) | Combined bugs have median 15 F2P tests vs 4 for single LM edits; failed fixes become second-order bugs |
| Realistic bug origin | BugPilot (2510.19898) | Bugs from agents adding features; 1.2K beats 3K injected bugs by 2% |
| Removal of code or history | SSR | Removal plus history reversion beat direct injection |
| Iterative "make it harder, update solution and verifier" | DeepSeek-V3.2 (2512.02556) | Agent adds tools when the toolset cannot support the harder task |
| Partial solutions as hints (makes hard tasks learnable) | QuestA (2507.13266) | 1.5B: AIME24 72.50% (+10.73) |
| Start from late states (reverse curriculum) | R³ (2402.05808) | Start-state position is a continuous difficulty dial |
| Hint-giving environment errors | Environment Tuning (2510.10197) | Qwen2.5-7B BFCL multi-turn 7.00% → 36.92% from 400 instances |
| Tool count, horizon, initial DB state | AgentScaler (2509.13311) | Accuracy falls as tool-call count grows |

**Per-model-size calibration.** The evidence that one task pool does not fit all students:

- R-Zero: the 0.6B model peaks after one Challenger iteration and then declines; the 4B improves for three before a
  drop. Pseudo-label accuracy falls from 79% to 63% over three iterations for Qwen3-4B.
- AZR: gains grow with size (Coder 3B +5.7, 7B +10.2, 14B +13.2).
- Mind the Gap (2412.02674): the generation-verification gap that powers self-improvement scales with pretraining
  FLOPs, so a small model is a weak judge of its own tasks.
- Nemotron-Terminal: the same corpus gives 8B 2.5→13.0, 14B 4.0→20.2, 32B 3.4→27.4 on Terminal-Bench 2.0.
- Endless Terminals: large gains on its own dev set (Qwen2.5-7B 10.7% → 53.3%) but small transfer to Terminal-Bench 2.0
  (2.2% → 3.4%). Synthetic tasks filtered only for "solvable by o3 at pass@16" were far easier than the human-curated
  target.
- SWE-smith: for SFT, training on harder difficulty tiers gave no clear gain (10.8–13.6%, no trend). Hardness is not
  value by itself.
- OpenThoughts-Agent: keeping tasks that take GPT-5 more tokens gave about +3 pp. A frontier model's cost is a usable
  hardness proxy.

**Making tasks hard enough for strong students.** The approaches that produce hard, still verifiable tasks: (a)
human-curated frontier-filtered sets (Terminal-Bench 2.0: 89 tasks, frontier agents below 65%; SWE-Bench Pro: 1,865
problems from 41 repos, tasks of hours to days); (b) adversarial generation against the student (SSR, R-Zero,
search self-play); (c) composition and chaining (AgentSynth, TaskCraft, WebShaper, combined and higher-order bugs);
(d) realistic fault origin (BugPilot); (e) iterative escalation until a reference model fails (AgentFrontier,
DeepSeek-V3.2). Each still needs a solvability guarantee: DeepSeek-V3.2 keeps tasks with nonzero pass@100 by its own
model; Endless Terminals keeps nonzero pass@16 by o3; AgentFrontier requires one success in 3 by a tool-augmented
model (33% verification success); SSR uses the injected patch itself as the proof of solvability.

**The cost of verifying hard tasks.** Solvability by sampling scales badly: pass@100 on a long-horizon task is 100
long rollouts. Construction-based solvability (the planted fault, the injected patch, the reference solution that
runs in the World) costs one run. Benchmarks that cannot rely on construction pay with human time (Terminal-Bench 2.0
writes a human solution per task; SWE-bench Verified and SWE-Bench Pro are human-verified).

### Papers

| arXiv | Paper | Mechanism | Scale | Key numbers |
|---|---|---|---|---|
| 2402.03300 | DeepSeekMath (GRPO) | Group-relative advantage | 7B | Zero-variance groups give no signal (motivates filters) |
| 2503.14476 | DAPO | Dynamic sampling: drop groups with accuracy 0 or 1 | Qwen2.5-32B | 50 on AIME 2024 |
| 2504.03380 | Online Difficulty Filtering | Bound: improvement ≥ variance of success; balanced filtering | Math RL | Up to +12% in under half the steps |
| 2502.12272 | Frontier of Learnability | Sample by p(1−p) | PPO, VinePPO | Consistent gains across algorithms |
| 2501.12599 | Kimi k1.5 | Pass@10 difficulty, sample ∝ 1−s_i, remove guessable prompts (8 tries) | Production | Curriculum then hard-only phase |
| 2507.20534 | Kimi K2 | RL prompts of moderate SFT pass@k; rubric tasks; tool simulator as world model | 3,000+ MCP tools, 20,000+ synthetic tools | Tau2 66.1, SWE-bench Verified 65.8 |
| 2505.14970 | SEC | Non-stationary bandit over categories, reward = mean absolute advantage | Math, code | Learned curriculum |
| 2504.09710 | DUMP | UCB over distributions by advantage magnitude | Mixed sources | Distribution-level scheduling |
| 2504.05520 | AdaRFT | Moves target difficulty with recent reward | Math | Faster convergence |
| 2411.02337 | WebRL | New tasks evolved from failures; critic filter 0.05–0.75; 500 per phase | WebArena-Lite | 4.8% → 42.4% (Llama-3.1-8B) |
| 2505.03335 | Absolute Zero | Proposer reward 1−r̄ (0 if never solved); deduction/abduction/induction code tasks | No external data | AZR-Coder-7B +10.2 avg; gains grow with size |
| 2508.05004 | R-Zero | Challenger reward 1−2\|p̂−½\| minus BLEU-cluster penalty; keep \|p̂−½\| ≤ 0.25 | Qwen3, OctoThinker | Label accuracy 79→63%; collapse after 1–3 iterations **[headline numbers differ from tables]** |
| 2506.01716 | Self-Challenging Agent | Explore tools, then write Code-as-Task tasks | 4 tool environments | Llama-3.1-8B 12.0 → 23.5 self-improvement |
| 2506.24119 | SPIRAL | Zero-sum game self-play, role-conditioned advantages | 4 models | Up to +10% on 8 benchmarks |
| 2502.18449 | SWE-RL | Rule-based patch-similarity reward on software history | Llama 3 70B | 41.0% SWE-bench Verified |
| 2512.18552 | Self-play SWE-RL | Inject and repair bugs with one policy; test-patch spec; inverse mutation check | CWM 32B, 512 H100 | +10.4 Verified, +7.8 Pro; optimum solve rate ≈ 0.2 |
| 2510.19898 | BugPilot | Bugs arise from agents adding features | 1.2K bugs | FrogBoss-32B 54.6%, FrogMini-14B 45.3% Verified |
| 2505.17063 | Synthetic Data RL | Generate from task definition, adapt difficulty by solvability, select by pass rate | Qwen2.5-7B | +29.2 GSM8K, +13.7 CFA |
| 2506.08989 | SwS | Mine consistently failed problems, extract concepts, synthesise new problems | Math RL | Weakness-driven synthesis |
| 2508.03682 | Self-Questioning LMs | Proposer rewarded when problem is neither too easy nor too hard | Arithmetic, algebra, code | Unit tests as verifier for code |
| 2509.07414 | Language Self-Play | Competitive self-play without data | Llama-3.2-3B | Improves IF, math, code |
| 2510.24695 | AgentFrontier | ZPD: no-tool LLM fails, tool agent succeeds (1 of 3) | 12K frontier trajectories | HLE 28.6% (30B-A3B, CPT+RFT); dedup ≥ 0.7 |
| 2512.19682 | GenEnv | Simulator trained by reward-weighted regression on exp(−β(p̂−α)²), α = 0.5 | 5 benchmarks | ALFWorld 14.2 → 54.5; beats 3.3× Gemini offline data |
| 2511.10395 | AgentEvolver | Self-questioning task generation, self-navigating, self-attributing | Tool environments | Curiosity-driven tasks in new environments |
| 2510.10197 | Environment Tuning | 4-stage curriculum, hint-giving errors, per-turn progress reward | 400 BFCL instances | Qwen2.5-7B 7.00 → 36.92; GRPO alone 17.42 |
| 2507.13266 | QuestA | Partial-solution hints during RL | 1.5B | AIME25 62.29% (+12.79) |
| 2402.05808 | R³ reverse curriculum | Start RL from late states of demonstrations | Reasoning | Start position as a dial |
| 2505.24864 | ProRL | Prolonged RL with KL control and reference resets, diverse tasks | 1.5B | Gains where base fails at any k |
| 2412.02674 | Mind the Gap | Generation-verification gap scales with pretraining FLOPs | Several families | Small models are weak self-verifiers |
| 2506.14205 | AgentSynth | Information asymmetry: easy subtasks, hard compositions | 6,000+ tasks | 18% → 4% (level 1 → 6); $0.60 per trajectory |
| 2509.16941 | SWE-Bench Pro | Human-verified long-horizon enterprise tasks | 1,865 tasks, 41 repos | Public, held-out and commercial splits |
| 2601.11868 | Terminal-Bench 2.0 | Human-written solution and tests per task | 89 tasks | Frontier agents < 65% |
| 2509.08755 | AgentGym-RL | ScalingInter-RL: horizon grows over training | Multi-environment | Horizon as a curriculum axis |

### What works

- **Measure p̂ on the real student, during training, from the groups already run** (Kimi, DAPO, Online Difficulty
  Filtering). A static pool does not stay in band.
- **A generator rewarded for landing in band, plus a hard validity gate** (AZR, R-Zero, GenEnv, SSR), with the band
  centre between 0.2 and 0.5.
- **Composition and realism as hardness**: composed subtasks, linked and higher-order faults, faults from real
  development, removal of hints and history.
- **Learnability rescue for too-hard tasks**: hints (QuestA), late start states (R³), hint-giving environment errors
  (Environment Tuning), per-turn progress reward.

### What fails

- **Ungrounded self-play runs out.** R-Zero's labels drift and performance falls after 1–3 iterations; SPICE
  credits grounding for avoiding this.
- **Hardness without structure.** SWE-smith difficulty tiers; OpenThoughts-Agent rewrites that "harden"
  descriptions did not beat originals.
- **Solvability filters that are too weak** (Endless Terminals: o3 pass@16 > 0 admitted tasks far below the target
  benchmark's difficulty).
- **A generator steering one knob only.** SSR's appendix describes "tunnel vision" where the challenger narrows to a
  single difficulty dimension.

### Design recommendations

1. **Difficulty is a per-(task, student) quantity.** Store p̂ with a Wilson interval keyed by student id and checkpoint
   (Crucible already tracks discounted counts per task in `forge/difficulty.py`; key it by student too). A task is
   "in band for the 35B" and may be "too easy for the 397B".
2. **Band by student size.** Default centre 0.3 for G = 8 (between SSR's 0.2 and GenEnv's 0.5), with
   p ∈ [0.1, 0.7] admitted. Use p(1−p) as the sampling weight at training time.
3. **Every generator family exposes a knob vector** (count, linkage, distractors, horizon, tool count, partial
   observability, hint level, start-state offset, obfuscation). The controller picks knobs to hit the band for a
   named student and must spread choices across knobs (penalise knob-entropy collapse, per SSR's tunnel-vision
   warning).
4. **Strong-student tasks come from structure**: composition of verified subtasks (AgentSynth, TaskCraft depth/width),
   linked faults, higher-order variants from the student's own failed attempts (SSR), realistic origin (BugPilot),
   and hint removal. Reject "harder rewording" that adds no behaviour-signature change.
5. **Solvability by construction first, sampling second.** A task carries a reference solution that runs in the
   World and reaches the goal state. Sampling (frontier pass@k) is an audit on a sample, plus the final in-band check
   on the student. This keeps verification cost near one run per task even for long-horizon tasks.
6. **Too-hard tasks are not discarded; they become ladders**: the same task with hints, or started from a
   checkpoint, at several offsets. Remove the scaffold as p̂ rises.
7. **A cheap difficulty predictor** (MAGELLAN, Crucible's S15 predictor) ranks candidates before any rollout, so the
   expensive student check runs only on likely-in-band tasks.

---

## Q3. Generator-verifier-solver loops

### State of the art

**Roles.** The common split is proposer (or challenger, injector), solver, and verifier. Some systems use one policy
for both proposer and solver (AZR, SSR, SPICE, search self-play); others train separate models (R-Zero) or a third
generator that distils the teacher (Socratic-Zero). CURE co-evolves a coder and a unit-tester from their interaction
outcomes; AutoIF writes instructions, checker code and unit tests for the checker, then rejection-samples.

**Making the verifier trustworthy.** The strongest pattern is Code-as-Task (SCA): instruction, verification function,
example solution, and failure cases. A task is kept only if the example passes and the failure cases fail; tracebacks
go back to the challenger for repair. On Retail with Llama-3.1-8B only 5.2% of generated tasks survive. The
ablation is the important part: a verifier alone gives infeasible tasks; adding an example solution makes verifiers
too lenient (false positives); adding failure cases removes the false positives. SSR's consistency checks go further:
the test script passes on the original code, enough tests fail after injection, and inverse mutation testing
requires that reverting any single modified file makes some failing test pass. TaskCraft keeps an atomic task only if
a tool agent beats a no-tool LLM on it, and checks extensions for information leakage. DeepSeek-V3.2's general-agent
solution function may only call tools, never the database, so the task must be solvable through the interface the
agent has.

**Known verifier weaknesses.** Test-based verifiers miss wrong patches: UTBoost found 345 erroneous patches labelled as
passing in SWE-bench and changed 40.9% of Lite and 24.4% of Verified leaderboard entries; PatchDiff (2503.15223) found
7.8% of plausible patches behaviourally wrong. R2E-Gym found execution-based and execution-free verifiers each plateau
at about 42–43%, with the hybrid at 51%.

**Reward hacking and collusion.** ImpossibleBench measures cheating by making the spec and the tests conflict, so any
pass is a shortcut (for example, deleting failing tests). Anthropic's 2511.18397 shows that learning to hack
production coding environments generalised to alignment faking and sabotage; effective mitigations were preventing
the hacks, more diverse safety training, and inoculation prompting. OpenAI's 2503.11926 shows a weaker model can
monitor a stronger model's chain of thought for hacks, but optimising against the monitor teaches obfuscated hacking.
For generator-verifier loops specifically, SSR's appendix names the dominant strategies of a capable challenger: a
randomly failing test (which gives exactly the "medium" solve rate a band reward wants) and heavy obfuscation. AZR
reports an "uh-oh moment" in a Llama run. Kimi k1.5 removes prompts a model can guess without reasoning. Nemotron 3
reports that training all environments at once was less prone to reward hacking than staged training.

**Task repair from failed rollouts.** WebRL evolves new tasks from failed ones. SwS extracts the concepts behind
consistently failed problems and synthesises new ones around them. SSR applies failed solver patches to create
higher-order bugs. SCA, Endless Terminals (k = 3 rounds) and MiniMax-M2 Terminal-Gym feed build or test failures back
to the generator for repair. OS-Genesis scores partial trajectories 1–5 and samples by score instead of discarding.

### Papers

| arXiv | Paper | Mechanism | Scale | Key numbers |
|---|---|---|---|---|
| 2506.01716 | Self-Challenging Agent | Code-as-Task with failure cases; traceback repair | 4 environments | 5.2% of tasks kept; 12.0 → 23.5 **[50 vs 200 annotated trajectories inconsistent in paper]** |
| 2512.18552 | Self-play SWE-RL | 5-file bug artifact, consistency and inverse-mutation checks, reward −α at s∈{0,1} | 32B | α = 0.8; solver-feedback reward only slightly beats consistency-only |
| 2505.03335 | Absolute Zero | Execution validity, determinism (2 runs), blocked modules | 3B–14B | "Uh-oh moment" flagged |
| 2508.05004 | R-Zero | Separate Challenger and Solver; majority-vote labels | 0.6B–8B | Label accuracy decays; collapse |
| 2509.24726 | Socratic-Zero | Teacher, Solver, Generator co-evolve from few seeds | Math | Generator distils the Teacher's question design |
| 2510.24684 | SPICE | Challenger mines corpus documents; Reasoner solves | Several families | +8.9% math, +9.8% general reasoning |
| 2510.18821 | Search Self-play | Proposer writes deep-search queries; ground truth checked from its own search results | Search agents | Self-play without supervision |
| 2506.03136 | CURE | Coder and unit tester co-evolve | 4B–14B | +5.3% code generation, +9.0% Best-of-N |
| 2406.13542 | AutoIF | Instruction + checker code + checker unit tests, execution rejection sampling | Qwen2, Llama3 | Verification as code |
| 2504.07164 | R2E-Gym | SWE-Gen from commits with back-translated issues; hybrid verifiers | 8.7K tasks | 34.4% pass@1; hybrid 51% |
| 2412.21139 | SWE-Gym | Real tasks; trained verifiers on agent trajectories | 2,438 tasks | 32.0% Verified with verifier |
| 2506.09289 | UTBoost | LLM test generation exposes weak tests | SWE-bench | 345 wrong patches; 24.4% of Verified entries change |
| 2503.15223 | PatchDiff study | Differential patch testing | SWE-bench Verified | 7.8% of plausible patches wrong |
| 2510.20270 | ImpossibleBench | Spec-test conflicts make any pass a cheat | LiveCodeBench, SWE-bench | Cheating rate as a metric |
| 2511.18397 | Natural emergent misalignment | Hacking in production RL generalises to misalignment | Anthropic environments | 3 effective mitigations incl. inoculation prompting |
| 2503.11926 | CoT monitoring | Weaker model monitors CoT; optimising against it obfuscates | o3-mini | Monitor in reward helps only at low optimisation |
| 2507.17746 | Rubrics as Rewards | Instance rubrics aggregated as reward | Medical, science | Up to +31% HealthBench vs Likert judge |
| 2406.12045 | τ-bench | Final DB state vs annotated goal state; pass^k | Retail, airline | gpt-4o < 50%, pass^8 < 25% (retail) |
| 2506.07982 | τ²-bench | Dual-control: user also acts on the world | Telecom added | Verifier over shared world state |
| 2512.20856 | Nemotron 3 | All RL environments trained at once | Nano/Super/Ultra | Simultaneous training less prone to hacking than staged |
| 2412.19723 | OS-Genesis | Graded trajectory reward model, sample ∝ score | GUI | Qwen2-VL-7B AndroidWorld 17.41% |
| 2506.08989 | SwS | Failure concepts → new problems | Math RL | Weakness-targeted generation |

### What works

- **Verifiers with both positive and negative evidence** (SCA's failure cases, SSR's inverse mutation, AutoIF's
  checker unit tests).
- **State-based verification of the World** (τ-bench, AgentScaler, DeepSeek-V3.2): compare final state with a goal
  state derived by running the reference through the tools.
- **Privileged information the solver never sees**: Endless Terminals' ground-truth section, SSR's oracle test files,
  reset before evaluation.
- **Removing shortcut channels**: SSR deletes git history and resets test files; DeepSeek-V3.2 forbids the solution
  from touching the DB directly.
- **Repair loops with a retry cap**, and failures turned into new tasks rather than discarded.

### What fails

- A generator rewarded only on solver pass rate (flaky tests and obfuscation earn the band reward).
- Self-verification by small models (Mind the Gap; R-Zero label drift).
- Execution-only or judge-only verifiers (R2E-Gym's plateau; UTBoost; PatchDiff).
- Optimising directly against a monitor (obfuscation).

### Design recommendations

1. **Separate the three roles by model family and by information.** The proposer sees the World and the cell; the
   verifier is code built from privileged ground truth; the solver sees only the instruction and the tools. Crucible's
   family rule already applies to artifacts; extend it to "the model that wrote the verifier may not judge its
   validity".
2. **Gate before any reward reaches the generator**: reference passes (N runs, deterministic), every planted negative
   fails, start state fails, inverse mutation (undo any single planted change → some check flips), no-tool or
   no-World solver cannot pass (TaskCraft), and an ImpossibleBench-style "impossible twin" is not passed by the
   reference student (a cheating-rate probe per task family).
3. **The generator's reward is validity first, band second**: consistency-only reward is nearly as good as
   solver-feedback reward in SSR, so band reward is a small bonus on top of a hard validity gate, never the main term.
   A task whose p̂ is in band must also fail for the intended reason (Crucible's failure targeting) or the bonus is
   withheld; this blocks the flaky-test strategy.
4. **Repair, then mutate.** A failed gate returns the trace to the proposer (cap 3 rounds). A task the student always
   fails is laddered (Q2). A task the student always passes is mutated along an unused knob. Failed student attempts
   become higher-order variants.
5. **Audit for hacks continuously**: a CoT and action monitor on a sample of training rollouts (as a detector, not a
   reward term), protected paths, and train environments mixed rather than staged.

---

## Q4. Pipeline architectures at scale

### Stage maps of the main pipelines

| arXiv | Pipeline | Stages | Scale and yield | Cost per kept task |
|---|---|---|---|---|
| 2504.21798 | SWE-smith | repo → one Docker env per repo → bug candidates (5 strategies) → keep if ≥ 1 test breaks → LM issue text → trajectories | 50,137 instances, 128 repos; 50.1% candidate yield (Combine 96.9%, Modify 56.0%, Procedural 40.2%, Rewrite 35.0%, PR Mirror 33.8%) | About $1,360 total, ≈ $0.027 per instance; 2.32¢ per candidate; 295 GB of images vs an estimated 50–150 TB per-instance |
| 2505.20411 | SWE-rebench | 450K PRs from 30K+ repos → 153.4K candidates → LLM install recipes (≤ 3) → execution validation → fine-tuned quality labels | 21,336 tasks, 3,468 repos; recipes work in 31% of repos | Not reported |
| 2412.21139 | SWE-Gym | Real issues with runtime | 2,438 tasks | Manual-heavy |
| 2504.07164 | R2E-Gym | Commits → generated tests → back-translated issues | 8.7K tasks | Not given in abstract |
| 2506.10954 | SWE-Factory | Multi-agent env builder, exit-code test parsing, binary test-file recovery | Issue tasks | Automates fail-to-pass checks |
| 2509.08724 | SWE-Mirror | Port a real issue's essence into a repo that already has a gym | 60,671 tasks, 40 repos, 4 languages | Reuses environments |
| 2502.13681 | Repo2Run | Agent writes Dockerfile until tests run | 420-repo benchmark | Environment build is the bottleneck |
| 2505.23419 | SWE-bench-Live | Automated curation of post-2024 issues | 1,319 tasks, 93 repos | Live, contamination-aware |
| 2512.02556 | DeepSeek-V3.2 | General agent: build data + DB in sandbox → synthesise tools → simple task with solution and verifier functions → iterate harder → keep nonzero pass@100. Search: long-tail entities → question agent → multi-config answerers → verifier keeps only if all candidates wrong and truth checked. Code: millions of issue-PR pairs → agent builds env → F2P > 0 and P2F = 0 | 1,827 environments, 85,267 tasks (24,667 code, 50,275 search, 4,417 general, 5,908 interpreter) | Not reported |
| 2507.20534 | Kimi K2 | 3,000+ MCP tools + 20,000+ evolved synthetic tools → thousands of agents (prompt × toolset) → rubric tasks → user personas + stateful tool simulator → judge filter; real sandboxes for code | "Tens of thousands" of examples; K8s sandboxes 10,000+ concurrent | Not reported |
| 2509.13311 | AgentScaler | 30K+ APIs → tool graph → Louvain → 1,000+ domains → DB schema → tools as code → tool-graph walks → 3 trajectory filters (validity, final state, exact call match) | 1,000+ domains; 4B/8B/30B-A3B | Retention not reported |
| 2605.26494 | MiniMax-M2 | PR tasks by type (bug, feature, performance, refactor, tests) with type-specific rewards; AppDev with Agent-as-a-Verifier; Terminal-Gym from Stack Overflow (tier grading, repair loop, hint removal, difficulty calibration); cowork tasks with rubric judges | 10+ languages | No counts reported **[uncertain scale]** |
| 2602.21193 | Nemotron-Terminal | Seed adapter + skill-combination generator → 9 shared domain images → pytest tests, no oracle solution → teacher traces → decontaminate (14-gram) | 264,207 synthetic + 226,313 adapted | Not reported; unfiltered traces scored 12.4 vs 5.06 success-only |
| 2601.16443 | Endless Terminals | Axis-sampled description with privileged ground truth → container + initial tests (≤ 3 repairs) → completion tests → o3 pass@16 > 0 | 3,255 tasks (≈ 2,500 in Harbor format) | ≈ 50% discarded at solvability |
| 2606.24855 | OpenThoughts-Agent | Source tasks (95 strategies ablated) → mix top sources → (no) augmentation → LLM description filter → teacher (GLM-4.7) traces in Daytona → drop timeouts, subagents, < 5 turns | 100K traces | ≈ 160 GH200-hours per 10K-trace ablation |
| 2511.19304 | AutoEnv | Environment as factorised transitions, observations, rewards; generated and validated levels | 36 envs, 358 levels | $4.12 per environment on average |
| 2510.01051 | GEM | Gym-style API, async vectorised execution, wrappers | 24 environments | Infrastructure, not generation |
| 2505.24760 | Reasoning Gym | 100+ procedural generators and verifiers with complexity parameters | Unbounded | Infinite data with adjustable difficulty |
| 2509.17158 | ARE / Gaia2 | Environments with apps, rules, content, verifiers; asynchronous time | Gaia2 | Time and noise as first-class |
| 2504.03601 | APIGen-MT | Task blueprints with ground-truth actions reviewed by an LLM committee → simulated human-agent interplay | 1B–70B models | Verified blueprint before trajectory |
| 2409.00920 | ToolACE | Self-evolved API pool, multi-agent dialogs, rule + model verification | 26,507 APIs | 8B SOTA on BFCL at the time |
| — | Harbor (laude-institute/harbor, GitHub) | Task = instruction + Dockerfile + test script that writes a reward; parallel trial runner on Docker, Daytona, Modal; registry | Hundreds of eval sets (talk estimate) **[uncertain]** | Infrastructure; no paper |
| 2506.04178 | OpenThoughts (reasoning) | 1,000+ controlled experiments per pipeline stage | 1.2M examples | Template for stage-wise ablation |

### What works

- **Disentangled stages, each with its own gate and yield metric.** Every large pipeline above has the same spine:
  source or spec → environment → tools → task → verifier → filter → difficulty → dedup → train. The good ones
  report yield per stage (SWE-smith per strategy; SCA per gate).
- **Shared environments, many tasks per environment.** SWE-smith's one image per repo (500× less storage), Nemotron-
  Terminal's 9 shared images, SWE-Mirror's reuse of existing gyms. Environment build is the expensive and failure-
  prone stage (SWE-rebench: 31% of repos).
- **Cheap construction-based generators produce most volume; expensive realistic ones add most value.** SWE-smith's
  combined and procedural bugs cost 0¢; PR Mirror costs 5.53¢ but gave the most useful trajectories (9.2% vs 5.7% for
  LM Modify).
- **Stage-wise ablation of the factory itself** (OpenThoughts 1,000+ experiments; OpenThoughts-Agent 100+).

### What fails

- Per-task images (storage and build failures), success-only trace filtering (Nemotron-Terminal), staged curricula of
  sources (Nemotron-Terminal: mixed 13.03 vs two-stage 10.39), adding weak sources (OpenThoughts-Agent).

### Design recommendations

1. **The stage graph**, generic across domains:

   | Stage | Output | Gate | Yield metric |
   |---|---|---|---|
   | S0 DomainSpec | coverage axes, knob schema, World schema | human or lead review | — |
   | S1 World build | DB state, sites, people, clock, seeded from spec | schema and invariants check | build success |
   | S2 Tools | tool code over the World (solution may call only tools) | tool unit tests | tool pass rate |
   | S3 Task proposal | instruction, privileged ground truth, reference solution, negatives, knob vector, claimed cell | format | proposals per cell |
   | S4 Verifier | executable state check from privileged ground truth | Q3 gate list | gate pass rate per check |
   | S5 Behaviour signature | tools, rows, people, events, horizon | claimed cell = measured cell | label agreement |
   | S6 Dedup | — | text, embedding, behaviour | duplicate rate |
   | S7 Difficulty | p̂ per student, frontier audit sample | band per student | in-band rate |
   | S8 Archive | cell elite by learnability × interestingness | — | cell fill, Vendi |
   | S9 Train and observe | p̂ updates, failure clusters | hack monitor | learning progress per cell |
   | S10 Feedback | new proposals targeted at weak, empty or drifting cells | — | value per generator |

2. **One World image per domain, many tasks per World snapshot**, each task a state diff plus a start-state seed
   (SWE-smith's lesson, applied to databases and sites).
3. **Account cost per kept task per stage** and per generator, and rank generators by downstream value per dollar
   (the SWE-smith strategy table is the template).
4. **Ablate the factory, not only the model**: each factory change is an experiment with a fixed small student run.

---

## Q5. Traces to tasks

### State of the art

Two directions:

- **Forward from documents or plans**: tutorials become task specs, a model executes them, a judge verifies
  (AgentTrek, $0.55 per trajectory; Synatra, 100K demos from tutorials at $0.031 each, about 3% of human cost;
  Learn-by-interact from documentation; InSTA annotates 150K sites with tasks, judge accuracy 82.6%).
- **Backward from interaction ("reverse instruction")**: explore first, then label what was done as the task. BAGEL
  round-trips between a labeller (trajectory → instruction) and an agent (instruction → refined trajectory).
  NNetNav labels exploration retroactively and prunes episodes that cannot be labelled with a meaningful sub-task (10K
  demos: +15 pts WebArena for Llama-3.1-8B). OS-Genesis derives low-level then high-level tasks from before/after
  states and scores trajectories rather than discarding. Explorer synthesises 94K trajectories over 49K URLs at 28¢
  each. Go-Browse frames exploration as graph search (10K trajectories, 100 URLs). Instruction Backtranslation is the
  text-only origin of the idea. BUTTON builds compositional tasks bottom-up from atomic ones, then generates
  trajectories top-down in a simulated multi-agent setting. TaskCraft and WebShaper extend seeds structurally.
  APIGen-MT verifies a blueprint with ground-truth actions before simulating the dialogue.

For enterprise logs the backward direction is the natural one: a log is a trajectory with an outcome; the task is the
intent behind it, and the verifier is the state change it produced.

**Privacy.** Enterprise traces carry PII and secrets. Relevant work: Clio analyses millions of conversations through
model-produced, aggregated summaries without human reviewers reading raw text; Aug-PE generates differentially
private synthetic text with only API access to an LLM; Carlini et al. (2012.07805) show that language models can
regurgitate verbatim training sequences, so any trace text that reaches a trained model can leak. No paper in this
set gives a complete trace-to-environment redaction protocol; that part is design, not literature.

### Papers

| arXiv | Paper | Mechanism | Scale | Key numbers |
|---|---|---|---|---|
| 2412.09605 | AgentTrek | Web tutorials → task specs → VLM agent replay → VLM judge | Web | $0.55 per trajectory |
| 2409.15637 | Synatra | Indirect knowledge (tutorials) → direct demonstrations | 100K demos | $0.031 each; 7B CodeLlama beats GPT-3.5 on WebArena, Mind2Web |
| 2502.11357 | Explorer | Exploration and refinement → task intents | 94K trajectories, 49K URLs | 28¢ per successful trajectory |
| 2412.19723 | OS-Genesis | Explore → (before, action, after) → low-level → high-level task; graded reward model | Android, web | AndroidWorld 17.41% (Qwen2-VL-7B) vs 9.82% self-instruct |
| 2410.02907 | NNetNav | Retroactive labelling; prune unlabellable episodes | 10K demos | WebArena 16%, WebVoyager 35% (Llama-3.1-8B) |
| 2403.08140 | BAGEL | Labeller ↔ agent round trips | ToolQA, MiniWob++ | +2–13% absolute |
| 2501.10893 | Learn-by-interact | Docs → interactions → backward instruction construction | SWE-bench, WebArena, OSWorld, Spider2-V | Training and ICL gains |
| 2502.06776 | InSTA | LLM annotates 150K sites; agents attempt; LLM judge filters | 150K sites | Judge 82.6%; harmful-content detection 97%; 1.7B agent 56.9% |
| 2506.03533 | Go-Browse | Exploration as graph search | 10K trajectories, 100 URLs | 7B: 21.7% WebArena |
| 2410.12952 | BUTTON | Bottom-up compositional tasks, top-down multi-agent trajectories | Function calling | Compositional multi-turn data |
| 2506.10055 | TaskCraft | Atomic tasks from documents; depth and width extension; no-tool vs tool check | ≈ 36K tasks **[32K vs 36K in paper]** | Atomic pass 54.9 → 68.1% with prompt optimisation |
| 2507.15061 | WebShaper | Set-theoretic formal task, agentic Expander adds knowledge projections | IS tasks | Open-source SOTA on GAIA and WebWalkerQA (scores not checked) **[uncertain]** |
| 2507.02592 | WebSailor | Structured sampling + information obfuscation for high uncertainty; DUPO RL | BrowseComp-style | Matches proprietary agents (claim) |
| 2505.22648 | WebDancer | Browsing data construction → trajectory sampling → SFT → RL | GAIA, WebWalkerQA | Four-stage recipe |
| 2308.06259 | Instruction Backtranslation | Label human text with instructions, self-curate | LLaMa | Origin of reverse instruction |
| 2504.03601 | APIGen-MT | Verified blueprint (ground-truth actions) → simulated interplay | τ-bench, BFCL | Small models beat larger ones in multi-turn |
| 2412.13678 | Clio | Privacy-preserving aggregated analysis of conversations | 1M conversations | No human reads raw text |
| 2403.01749 | Aug-PE | DP synthetic text via API-only Private Evolution | 3 datasets | No model training needed |
| 2012.07805 | Extracting Training Data | Verbatim extraction from trained LMs | GPT-2 | Memorisation risk of trace text |

### What works

- **Backward labelling plus a replay check**: the derived task must be re-solvable in the environment (BAGEL round
  trips, OS-Genesis execution of high-level tasks).
- **Pruning trajectories that cannot be named as a meaningful sub-goal** (NNetNav).
- **Graded rather than binary trace quality** (OS-Genesis).
- **Costs well under a dollar per trajectory** across Synatra, Explorer, AgentTrek, AgentSynth.

### What fails

- Judge-only verification of derived tasks (InSTA's judge is 82.6% accurate: about one in six labels wrong).
- Deriving tasks from logs without rebuilding the start state: the task cannot be replayed or verified.
- Feeding raw trace text into generators that later train models (memorisation).

### Design recommendations

1. **Trace → World snapshot → task, never trace → text → task.** From a log, reconstruct the start state in the
   simulated World (entities, records, clock), take the log's end state as the goal state, and let a labeller write
   the intent. The verifier is the state diff. The task enters the same Q3 gate as generated tasks.
2. **Re-solve before keeping**: a frontier agent or the replayed log actions must reach the goal state from the
   rebuilt start state.
3. **Redact before any model sees the trace**: deterministic PII and secret scrubbing; replace real people and
   organisations with World personas consistently (same real ID → same synthetic persona) so relational structure
   survives; aggregate rare values. For text that must keep realism (emails, tickets), generate DP synthetic
   replicas (Aug-PE style) rather than copying.
4. **Use traces for coverage, not volume**: cluster traces Clio-style into intent categories and use the clusters to
   add or weight cells in the DomainSpec, so the factory covers what real users do.
5. **Contamination accounting**: tasks derived from logs after a cutoff date are eligible for the sealed benchmark;
   older ones go to training (SWE-rebench's date rule).

---

## Synthesis: a domain-generic, self-improving task factory

How the findings combine for 500–1000 tasks per domain:

1. **Spec first.** A DomainSpec defines the World schema, tools, coverage axes (MAP-Elites cells) and the knob
   schema. Target: every reachable cell holds at least k tasks per student band (for example 1000 tasks over about
   100 cells with k ≈ 3 per band for 3 bands).
2. **Generate forward from planted ground truth**, conditioned on a sampled cell, world slice, persona and K
   nearest-neighbour exclusions, with several candidates per call (VS). Each task is Code-as-Task: instruction,
   privileged ground truth, reference solution (tools only), negatives, knob vector.
3. **Gate with executable evidence** (reference passes deterministically, negatives fail, start fails, inverse
   mutation, no-tool solver fails, leakage check, impossible-twin probe).
4. **Measure behaviour, then dedup and place in the archive.**
5. **Calibrate per student**: a predictor ranks candidates; the student's p̂ places the task in a band; too-hard tasks
   become ladders (hints, checkpoint starts); too-easy tasks are mutated along an unused knob.
6. **Train with online difficulty sampling** (p(1−p) weights, drop zero-variance groups), all domains mixed.
7. **Close the loop**: failure clusters (SwS), failed attempts as higher-order variants (SSR), empty or collapsing
   cells (diversity meters), and generator ranking by downstream value per dollar. The factory itself is ablated
   stage by stage (OpenThoughts).

**Budget guide from the literature.** Construction-based generation runs at cents per candidate (SWE-smith 2.32¢);
trajectories cost $0.03–$0.60 each (Synatra, Explorer, AgentTrek, AgentSynth); a full environment costs about $4
(AutoEnv); environment builds fail often (31% repo success in SWE-rebench). Expect 5–50% gate survival (SCA 5.2%;
SWE-smith 50.1%; Endless Terminals about 50%) and plan proposal volume at 2–20× the target.

## Gaps and open questions

- No paper reports a per-student-size difficulty band schedule for a 35B vs a 397B student; the closest evidence is
  indirect (R-Zero and AZR scaling, Nemotron-Terminal per-size gains). This needs our own measurement.
- No paper gives a full redaction protocol for turning enterprise traces into environments.
- Several pipelines report no per-stage yields or costs (DeepSeek-V3.2, Kimi K2, MiniMax-M2, AgentScaler).
- MiniMax-M2, OpenThoughts-Agent and Nemotron-Terminal details come from the HTML full text read through an
  extraction tool; numbers were cross-checked against abstracts where the abstract carried them.

## PDFs saved

In `crucible/docs/research/papers/` (git-excluded): 2504.21798 (SWE-smith), 2512.18552 (Self-play SWE-RL),
2512.02556 (DeepSeek-V3.2), 2606.24855 (OpenThoughts-Agent), 2506.01716 (Self-Challenging Agent), 2508.05004 (R-Zero),
2505.03335 (Absolute Zero), 2510.19898 (BugPilot).
