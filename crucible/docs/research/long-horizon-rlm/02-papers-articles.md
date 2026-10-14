# Long-horizon RLM research 02: papers, articles and datasets

**Question.** Agent tasks can run for thousands of steps and up to millions of tokens. How do they manage context, memory
and credit? Crucible wants a recursive-LM (RLM) harness in which the model works inside a persistent Python environment.
What should that harness and its rewards look like? This note takes the answer from every local article, paper and
dataset.

**Status.** Research only: no code was changed. Studied at Crucible HEAD 775f30f, 2026-10-13. Every number below is as
the source reports it unless it is marked as checked.

## Sources

Abbreviations used for paths:

| Abbreviation | Path |
|---|---|
| A/ | `RL-Environments/07-articles/` |
| DS/ | `RL-Environments/08-hf-rl-env-datasets/` |
| MO/ | `RL-Environments/06-frameworks-and-env-code/xiaomimimo-verl/third_party/mimoagent-osr/src/mimoagent/` |

| Key | Source | What it gives |
|---|---|---|
| **CH** | A/prime-agent/2605.09998.txt, *Continual Harness* (Princeton / GDM, May 2026), read in full | Harness state H = (prompt, sub-agents, skills, memory); the Refiner edit loop; Pokémon runs up to about 200k turns; reset-free online training |
| **PA** | A/prime-agent/2608.23552.txt, *Prime Agent: A Self-Improving RLM Harness* (Aug 2026) | The L0–L3 state hierarchy; async `rlm()` children; autonomous mode, goal and heartbeats; ARC-AGI-3, OOLONG, nanoGPT, Factorio (7 days, 23.4M output tokens) |
| **RLM** | A/prime-agent/rlm.txt, the PI RLM blog (Jan 2026) | REPL output cap of 8192 chars; 120 s cell timeout; `llm_batch`; tools only through sub-LLMs; the answer variable |
| **RUST** | A/prime-agent/prime-agent-rust.txt | A swarm of 2,209 agents and 228.7B tokens; the Planner → Implementer → Reviewer → Verifier loop; open-ended hill-climbing with no numeric target |
| **QK** | A/qingkeai-2026-agentic-rl-infra.md, with formulas from raw/…html | §3.8 context management (GLM, LongCat, Kimi, DeepSeek, StepFun, MiniMax); §3.5.3.6 CompactionRL; §3.6 credit and rewards; §3.7 hacking |
| **SWARM** | A/pi-on-the-nature-of-the-swarm.md | Compaction as a bet; persistent sub-agents; trees vs a mesh |
| **VF1** | A/pi-verifiers-v1.md | The trace as a message graph: N branches give N trainable samples |
| **GS** | A/git-swarm.md, arxiv-2610.04862-abs.md, gitswarm-2610.04862.txt | Git as shared memory; Contribute / Nominate / Abstain; honest-evidence artifacts; open-ended research hill-climbs |
| **FE** | A/hf-space-fineenvs-multi-harness-rl.md | Multi-harness capture; the "+0.2 for submitting" collapse; a bonus paid only when the answer is correct |
| **R2E** | A/hf-repo2rlenv-how-not-to-build-coding-environments.md | Contamination; the gold = 1 / no-op = 0 rule |
| **PM** | A/pm-building-rl-environments-for-superintelligence.md, pm-introducing-karotte.md, pm-karotte-technical-deep-dive.md | Don't lie to the AI; make tasks deep; an honest way to give up; Karotte's defences |
| **HFG** | A/hf-space-rl-environments-guide.md, hf-blog-rl-environments.md, zhihu digest | Reward patterns (rubric, gate, judge); per-action vs per-trajectory reward |
| **LHTB** | DS/IntelligenceLab__Long-Horizon-Terminal-Bench | 46 tasks; `continue_until_timeout`; dense hidden graders; headline = count of tasks with reward ≥ 0.95 |
| **TBS / TB2.1** | `/Users/muahmed/Desktop/Projects/terminal-bench-science` (75 tasks), DS/harborframework__terminal-bench-2.1 | 8 h agent budgets; binary reward enforced |
| **MiMo** | DS/FineEnvs__MiMo-V2.6-RL-harbor-code; mimoagent (MO/) | The model-decided `Compact` tool; a 9-section summary; a wrap-up hint at 0.9 |
| **CR** | crucible/docs/research/long-horizon/01-frameworks.md; docs/modules/long-horizon.md; src | Checked against code in §8 |

Two caveats apply throughout:

1. **None of PA, RLM, RUST or CH trains an RLM with policy-gradient RL.** PA §5 says harness capabilities "remain underused
   because current models were not trained to operate them". CH trains a small student by imitating a teacher's
   rewrites, not by policy gradient. All training mechanics for summaries and segments in this note come from QK
   (CompactionRL, MiniMax, Kimi PARL, StepFun), VF1 and FE.
2. **QK is a survey.** Its numbers are the labs' own reports and were not re-derived here. The RLM blog's charts are
   images, so only the direction of each result can be recovered.

---

## 1. Context-management techniques, with parameters and results

### 1.1 Heuristic and harness-level techniques

| Technique | Source | Parameters | Reported result |
|---|---|---|---|
| **Keep-recent-k** (drop older steps) | QK §3.8.3, GLM-5 search agent | k = 5 | BrowseComp 55.3% → 62.0% |
| **Tiered HCM** (discard all tool history and restart past a threshold) | QK §3.8.3, GLM-5 | T = 32k tokens | BrowseComp 75.9 (final). Inference-time only; whether it was used in RL is not stated |
| **Hybrid** (summary, then discard-all) | QK §3.8.3, LongCat | summary at 80K tokens; discard-all when max turns are reached; discard thresholds tighten over time | BrowseComp 55.8% → 73.1% |
| **Varied per task group in training** | QK §3.8.3, Kimi K3 | context management is a configurable harness module that changes across task groups | the aim is to avoid overfitting to one mechanism (no number) |
| **Preserved thinking** | QK §3.8.4: GLM-5; Kimi K3; DeepSeek-V4 | keep every reasoning block; DeepSeek-V4 keeps them across user-message boundaries | Kimi's template supports only this; without it "generation quality is very unstable" |
| **Selective thinking** | QK §3.8.4, StepFun | keep only the reasoning of the tool chain started by the most recent user instruction | dropping all thinking fails 100+-turn coding sessions; keeping all of it costs too much |
| **Model-decided compaction** | MiMo, MO/tools/cc/compact.py and MO/agents/cc/cc_agent.py:63-72, 227-244 | the `Compact` tool sets a flag and the rebuild happens at the next query. A `<context_usage>` footer is added once usage ≥ 0.8 of (window − 10k buffer): window 100k for cc, 1M for mimocode | the shipped code recipe turns compaction off (262k window, `trajectory_selection: longest`) |
| **Automatic compaction, Codex style** | MiMo, MO swe_codex_agent.yaml:17-21 | trigger at 900k tokens; keep user messages up to 20k tokens; add a hand-off summary | n/a |
| **Wrap-up hint** (separate from compaction) | MiMo, MO/agents/default.py:70-80 | a user-turn hint at 0.9 of the budget: *finish*, do not compact | n/a |
| **Middle-truncation of tool output** | MiMo default.py:62-69; Karotte; VF1 | MiMo keeps 40,000 chars; verifiers 20 KB; Karotte puts a timeout on every command and truncates long output | Karotte: "One `cat` of a large file fills up the whole context" |
| **REPL output cap** | RLM | 8192 chars per turn by default, adjustable; data lives in variables | forces the model to use code and sub-LLMs to read data |
| **Prompt-as-variable / file** | PA §2.3, §3.2; RLM | the task context is a readable file or variable, searched from the REPL | OOLONG-128k: GLM-5.2 scores .700 with Prime Agent vs .420 with Pi-mono; OOLONG-Pairs .874 vs .556. Frontier models on their native harness are near parity |
| **Tools only through sub-LLMs** | RLM | the root never sees tool output; DeepDive `open` returns tens of thousands of tokens truncated and 1.5M+ untruncated | sub-LLMs carry most completion tokens, so the main context stays small |
| **Answer variable** | RLM | `answer = {"content": "", "ready": False}`; the rollout ends only when `ready` is True, and `content` can be edited over many turns | "a form of diffusion" over the reasoning chain; the model can check and fix the answer before ending |
| **Memory catalog in the prompt, bodies on demand** | CH C.1.4 | every step prompt lists all memory IDs and titles ("LONG-TERM MEMORY OVERVIEW"); `process_memory` reads a body | forgetting happens only by demoting importance |
| **Narrow sub-agent contexts** | CH C.1.3, Fig 17a | sub-agents have their own prompt | about 10× fewer tokens per step than the orchestrator (8× on Red, 11× on Emerald) |
| **Sub-agent as compaction** | SWARM | a sub-agent reads about 200k tokens and returns about 2k; its summary is written *after* the question is known | the essay recommends persistent sub-agents that can take follow-up questions |
| **Agent swarm as context sharding** | QK §3.8.4, Kimi K2.5 | sub-agents have independent bounded memory and return only relevant output | beats discard-all on BrowseComp in both accuracy and efficiency |
| **Lossless L3 under lossy L1** | PA §2.2 | "Compaction replaces a conversational prefix with a summary and retains the original events in L3 for REPL retrieval" | design rule (no ablation reported) |
| **Fixed observation pacing** | CH App A | 120 emulator frames per step; one action can batch several button presses | one decision per meaningful change of state |
| **Persistent shell; nudge on an empty turn** | PM Karotte deep dive | `cd` and `export` carry over between commands; a turn with only reasoning gets "continue" instead of ending the episode | robustness over long episodes |

### 1.2 Compaction as part of RL training

| Technique | Source | Parameters | Result |
|---|---|---|---|
| **CompactionRL** (Tsinghua; used in GLM-5.2 RL) | QK §3.5.3.6, §3.8 | trigger when **remaining budget < 10,240 tokens**; the **same trainable policy** writes the summary; the context is rebuilt keeping the **last k = 2 steps**; the summary segment enters the RL objective | GLM-4.5-Air: SWE-bench Verified 66.8% (+7.0), Terminal-Bench 2.0 24.5% (+3.1). GLM-4.7-Flash: 56.0% (+5.5), 20.2% (+6.8). Under compacted evaluation SWE-V goes 59.8 → 66.8. The authors admit gains under single-window evaluation are inconsistent |
| **Context management as an RL action** | QK §3.5.3.2, §3.8.3, MiniMax | context management is an explicit action folded into the dynamics, s_{t+1} = f(s_t, a_t, o_t); each (s_t, a_t) pair is trained on its own | removes the train/inference mismatch: the policy "need not know whether s_t came from simple append, aggressive truncation or full history rewrite" |
| **Branch = sample** | VF1; FE | the trace is a message graph whose size grows linearly in turns; each root-to-leaf path is one training sample; compaction starts a new root (S1, U1′) | "unlocks long-horizon training past a model's context window"; FE: Claude Code produces about 8 rows per rollout and 77% of all rows |
| **Memory-agent overwrite** | MemAgent (CR 01 §3, checked) | 1,024-token memory carried across fresh-context chunks | n/a |

### 1.3 Scale reference points

- **Long trajectories (QK §0):** hundreds to thousands of tool calls and millions of tokens per trajectory. Kimi K3
  trains with RL at 1M context. Qwen3-Coder-Next's mean turns rose from 50 to 130. About 5% of samples take about 80% of
  generation cost.
- **Compression ratio (SWARM fn 3):** "for most tasks a 1M-token working context compresses quite nicely into 16k
  tokens", about 60×.
- **Prime Agent runs:**
  - Factorio: 7 days, 23.4M output tokens, 633 sub-agents of depth 1 in 149 waves, at most 7 running at once.
  - nanoGPT: 85.5 hours.
  - ARC-AGI-3: up to 3M output tokens per game.
  - Rust rewrite: 2,209 agents, 16,758 messages, 228.70B tokens, more than 10,000 sandboxes, depth up to 3.
- **Continual Harness:** Gemini Plays Pokémon reaches about 172k turns on Yellow Legacy and about 200k on the harness-growth
  plot. The agent stalls for 1,003 turns at the Power Plant.
- **GitSwarm:** 100M to 1B input tokens per ProgramBench task; NanoChat ran 24 active hours.
- **Datasets:**
  - LHTB: agent budgets up to 4 h (2048) and 240 min (apex-ib244). No token or step budget.
  - TB-science: 70 of 75 tasks at 8 h, 5 at 5 h. Expert estimates up to 600 h.
  - TB2.1: 15–60 min budgets for tasks experts estimate at up to 480 min.

---

## 2. How summaries and compaction are trained in RL

| Question | Answer from the sources |
|---|---|
| **Who writes the summary?** | The policy itself, in every training recipe that trains it: CompactionRL ("the same trainable policy"), MiniMax (context management as an action), VF1 (the summary AC is a sampled assistant message), MiMo (a policy-decided `Compact` tool and the Codex hand-off). No source trains with a separate summariser model. |
| **What reward does the summary get?** | **The episode's task reward, with no separate summary-quality reward**, chosen on purpose: "hand-defined summary metrics may not reflect which details matter" (QK §3.5.3.6). CompactionRL also uses no KL term. VF1 and FE give every branch the rollout's reward. |
| **Loss normalisation** | **Token-level.** The loss is averaged over all generated tokens of the rollout, not per sample or per segment. Averaging per segment would give rollouts that compact more often more weight at the same reward. |
| **Critic vs group baseline** | CompactionRL **moved back to PPO with a critic**, because a variable number of segments per rollout breaks fixed-size group normalisation. MiniMax keeps a trajectory baseline instead: Â_{i,t} = Σ_{p≥t} r_p − B_i, with each (s_t, a_t) trained on its own. |
| **Credit across segments** | **Cross-trajectory GAE (CompactionRL).** (1) Local GAE inside a segment: A^loc_{s,i} = Σ_l (γλ)^l δ_{s,i+l}, with δ_{s,i} = r_{s,i} + γV(x_{s,i+1}) − V(x_{s,i}). (2) Scale by position: Â_{s,i} = (γλ)^{N_{>s}} · A^loc_{s,i}, where N_{>s} is the number of trainable tokens after segment s. Without the scaling, naive per-segment GAE puts the terminal reward at the end of every segment, so early segments behave as if the outcome were near. The authors call it an approximation with no bias bound. |
| **Truncation** | StepFun: R̂ = V(s_T) if the rollout was truncated ("truncated ≠ failed"); stable at a 20% truncation rate; needs a critic. Qwen3-Coder-Next does the opposite: a penalty for trajectories still unfinished at max turns. |
| **Masks** | VF1 and FE: context tokens get mask 0 and sampled tokens mask 1. A turn with missing or misaligned logprobs stays as context and is never a target. A rollout stopped by the call budget does not train its final reply (FE). GLM SAO cuts observation tokens out of the GAE chain, so the TD step links one action's last token to the next action's first token. |
| **Sub-agents** | Kimi K2.5 PARL: sub-agents are frozen and their trajectories are left out of the objective; only the orchestrator is updated. Its reward is r_parallel + r_finish + r_perf, with the first two weights annealed to 0. |
| **Thinking across compaction** | The template choice (preserved or selective thinking) is part of the harness and must match between training and inference (QK §3.8.4). |
| **CH training, for contrast** | Reset-free online DAgger: 256-step iterations inside the live, self-editing harness. A pairwise judge scores windows (stride 8) with weights progress 0.4, action 0.3, reasoning 0.2, format 0.1. A teacher rewrites low-scoring windows; soft SFT, 3 epochs at 5e-6, batch size 1. Five runs gained +2 to +5 milestones over about 30 iterations. Each restart from a saved state regresses for 2–3 iterations. Offline GRPO (4 samples, judge scores action 0.6 + format 0.4) produced degenerate outputs. |

**Takeaway.** The settled recipe is:

- the policy writes the summary;
- the summary is trained on the shared episode reward;
- the loss is normalised per token;
- credit across segments uses either a critic with (γλ)^{N_{>s}} discounting, or MiniMax's reward-to-go per (s, a);
- the compaction mode the policy sees in training is varied across groups (Kimi K3) or chosen by the policy itself
  (MiniMax, MiMo).

What is not settled is how summary quality is credited. Every source uses outcome credit only, and the cross-segment fix
is approximate.

---

## 3. Recursion and sub-model delegation

| Design | Source | Depth | Budgets and limits | Observed |
|---|---|---|---|---|
| `llm_batch(prompts)`: parallel fresh copies of the same model | RLM | exactly 1 (0 and n are planned) | 120 s per REPL call, which also limits sub-calls; 8192-char output cap; no cap on the number of calls | models give sub-LLMs very little context; GPT-5-mini rarely uses them in math; GLM 4.6 stops using them when told to use them heavily |
| Async `rlm()`: a persistent child session with its own kernel, history and workspace; returns a handle; family-scoped message queues | PA §2.3, App B | Factorio 1; Rust rewrite up to 3 | accounting covers the root and all descendants (tokens, time and cost separately); session states admitted / running / idle / inactive; **children inherit the runtime's file, network and credential permissions** | Factorio: "a shallow, repeatedly widening tree … rather than deeper recursion"; at most 7 at once |
| Planner → Implementer (own worktree) → Reviewer (a different model, separate context) → Verifier (fresh sandbox) | RUST | 1 FSM per task | a failure sends the task back to the implementer; merge only when both checks pass | "an agent that writes code is biased when evaluating it" |
| `execute_custom_subagent` / `return_to_orchestrator` | CH C.1.3 | 1 | no budget | Clean return / back on goal within 10 steps, by task: dialogue 100/67, navigate 83/65, battle 83/23, grind 62/25, menu 100/100, catch 65/15. A saved harness that kept editing collapsed to 6.4% ± 5.7 inherited sub-agent use (Red) |
| Persistent ("hired") sub-agents in a mesh of peers with shared state (a repo or tracker) | SWARM | flat | communication cost is the limit: "If the hard part of a task needs everything in one place at once, splitting it up doesn't help at all" | trees push compaction up to the root |
| Flat swarm with Git as shared state; each episode is one of Contribute / Nominate / Abstain | GS | flat, concurrency 5 (12 to 20 workers in research runs) | budget B in charged episodes (100–500 ProgramBench, 20–160 proofs); **abstaining does not use up B** | 94.7% of contributions are later built on; the winner's ancestry covers 81.6–92.6% of the graph |
| Kimi K2.5 PARL orchestrator | QK §3.6.3.4 | 1 | CriticalSteps = Σ_t (S_main^(t) + max_i S_sub,i^(t)); r_finish pays only for sub-tasks that really finish, which stops sub-agent spam | wide-search latency up to 4.5× lower; item F1 72.8% → 79.0% |
| DeepSeek-V4.1 Agent Team | QK §3.6.3.4 | DAG | latency penalty = critical-path length, not event count | work off the critical path costs nothing |
| LongCat Heavy Thinking | QK §3.8.4 | 1 | parallel reasoning, then a summary model combines the results | n/a |

**Takeaway.**

- **Depth and width:** depth 1 is the norm (RLM, CH, Factorio). Width grows in waves, and deeper trees appear only in
  engineering swarms (depth 3).
- **Budgets:**
  - in tokens and wall time across the whole tree (PA);
  - on the critical path, for latency (Kimi, DeepSeek);
  - never "number of sub-calls" as a reward. Kimi pays r_finish only for completed sub-tasks because counting calls
    invites spam.
- **Children:** a child should be persistent (follow-ups, SWARM / PA) and have **least privilege**. PA children inherit
  credentials; §5 shows why that matters.

---

## 4. Memory designs and the Continual Harness loop

### 4.1 Memory designs

| Design | Source | Mechanism | Evidence |
|---|---|---|---|
| **L0–L3 hierarchy** | PA §2.2 | L0 weights; L1 context (changed by compaction); L2 REPL values and sub-agent sessions ("agentic garbage collection"); L3 disk: an append-only event log, kernel snapshots, the session tree, compaction records, message queues and versioned harness state. Forks never delete events | design; Factorio recovered after a destructive reset (tech 5 → 1) |
| **Typed supplemental state** | PA §2.5 | prompt notes (rules), memories (facts), skills (code), sub-agent specs (roles); create, read, edit or delete; per session or global. Edits come from the agent or a background `/refine` call, apply at turn boundaries, record their trigger and intended effect, are versioned and can be rolled back. The base prompt is never rewritten | none |
| **Harness H = (p, G, K, M)** | CH §2.2 | K holds text rules plus code (pathfinders, tool wrappers); M is a memory tree (`/red/brock: "Onix L12; use Bulb."`) with importance | most skills go unused (§4.2) |
| **Git as memory** | GS | each atomic commit has a Git parent plus a mandatory `informed_by=` list of sources, so the graph has both edges. Negative findings are commits. Every commit carries `MY_THOUGHTS.md` (evidence, **unresolved uncertainty**) and `EVALUATION.md` ("what you actually checked, known gaps"). Rule: "distinguish observed evidence from inference. Do not claim coverage that was not performed … never invent metrics." Workers announce "Gap \| Deliverable \| Success \| Abort" before editing | Rulebook test: shared memory keeps coverage high; independent workers reach 70–75% / 20–23%. Branch-native checkpoints cut repeated moves. Attribution IDs cut how much of the archive is read. 70–76% of negative results are later cited |
| **Files / REPL pointers** | SWARM | the summary carries pointers ("What do I need to know exists?") | files go stale (fn 7); the day-3 agent doesn't know what to grep for (fn 8) |
| **Fixed-size memory overwrite** | MemAgent | 1,024 tokens across chunks | n/a |

### 4.2 The Continual Harness edit loop

**What it does (CH §3.1–3.2, Fig 2a):**

1. **When:** every F steps, after a warm-up of W steps. The paper never gives values for F or W.
2. **Input:** the Refiner (the same model and meta-tools as the agent) reads the last F steps. Failure signatures seen
   earlier remain available.
3. **Diagnose:** it looks for navigation loops, tool-call failures, stalled goals and missed exploration.
4. **Four passes:**
   1. rewrite the prompt;
   2. create, edit or delete sub-agents, deleting ones not used productively;
   3. turn successful action sequences into skills and fix skill code that raised errors;
   4. add or update memory, and demote areas the agent has moved past.
5. **Apply immediately:** H_{t+1} = H_t ⊕ Δ. There is no reset, **no test before an edit is kept, and no automatic
   revert.**

**How the paper judges edits (after the fact only):**

- rolling skill success before vs after each update, and whether a skill was re-edited within 5 steps;
- saved harnesses reused with editing off vs on (3 seeds);
- path cost against the optimal (Dijkstra) path;
- the share of calls going to inherited components.

**What the paper found:**

- **Most authored skills are never used:**

  | Run | Created | Used at least once | Used 3+ times | Ever succeeded |
  |---|---|---|---|---|
  | Emerald phase 1 | 99 | 17 | 11 | 5 |
  | Red phase 1 | 110 | 33 | 18 | 3 |
  | Red phase 2 | 335 | 53 | 31 | 14 |

  "The refinement loop therefore triages."
- **Memory is rarely read back from scratch:** 35% reuse at Mt Moon over 72 operations. Runs that loaded a saved harness
  reached 100%. The paper proposes a bias toward reusing existing entries.
- **The harness is what carries over:** with editing off, inherited components take 98–100% of calls.
- **Collapse with editing on (Red):** new, unrepaired sub-agents replace repaired inherited ones from about step 213, and
  inherited sub-agent use falls to 6.4%.
- **Edits never settle:** in the Yellow run, up to about 3,000 changed lines per 2,000 turns over 200k turns.
- **Weak models get worse:** Flash-Lite reaches 20% at $11 with the minimal harness but 3–13% with any self-editing
  variant.
- **Cost (Emerald, Gemini 3 Pro):** 100% of milestones at a median $130, against 98% at $215 for the minimal harness,
  about 40% cheaper.

**The paper contradicts itself in two places:**

- §4.3 says a reused harness with editing on beats starting from scratch on Red; C.2.1 says it falls below the minimal
  harness.
- D.4 gives the student as 26B; Table 4 gives 31B.

**For Crucible.** M14 already does what CH lacks: a paired, interleaved A/B over at least 3 seeds with a noise gate. CH's
data is the evidence for keeping that gate. Without it, edits regress on components the agent does not use, and new
components displace repaired ones.

### 4.3 The same loop for performance work

The RUST hill-climb is an edit-and-A/B loop for engineering:

1. profile and build a backlog of hypotheses;
2. run baseline and candidate on the **same sandbox in alternating order**, plus neighbouring benchmarks to catch
   regressions;
3. **withhold results too unstable to compare**;
4. two reviewers on different models check parity;
5. merge, and start the next experiment from the new baseline.

Over 3 days this produced more than 144 experiment records and more than 69 merged changes.

GitSwarm adds three lessons:

- **Evaluation funnel:** small evaluations misrank, so it uses 128 → 512 → 4,873 examples. At 128 examples, 5 of 12 pairs
  were ranked wrongly.
- **Re-checks:** an idea wrongly rejected at 2.1 h was reversed at 3.3 h.
- **Search collapse:** the share of work on one idea went from 17% to about 90%. The prompt now says "reopen the search".

---

## 5. Failure modes and mitigations

| Failure | Evidence | Mitigation in the sources |
|---|---|---|
| **Context rot** | SWARM fn 1 ("the 'effective' context window … is smaller than the advertised one"); RLM | data in variables; output cap; sub-LLMs read tool output; persistent sub-agents |
| **A summary drops a fact that matters later** | SWARM fn 4 (a flaky-test line dropped on day 1 is the key on day 3; "a smarter summarizer doesn't save you here") | keep L3 lossless with pointers (PA); summarise after the question is known (sub-agent); keep a persistent sub-agent to ask follow-ups; MiMo's rebuild tells the model to "state the gap before relying on missing details" |
| **Goal drift after compaction or hand-off** | CH: back on goal after a hand-off only 15–25% for catch, battle and grind | MiMo's summary quotes the next step verbatim "to prevent drift" and keeps the **original issue verbatim**; Crucible's rebuild keeps the task verbatim; PA's *goal* persists across continuations; FE's Full harness adds drift checks and rollback |
| **Lost constraints** (a rule changes mid-task) | Crucible's backlog 50 → 25 USD amendment is built to test this, but its gate never uses the summary (§8) | MiMo's summary sections cover *all user messages* and *pending tasks*; Crucible's COMPACT prompt asks for exact ids and amounts |
| **Repetition / action loops** | CH Power Plant: 1,003 turns; an identical payload sent **842 times** because the harness silently ignored a malformed field. "Tool generation primarily occurred within the first 50-200 turns of encountering a new bottleneck. Beyond 500 turns in a stall, the agent ceased tool creation" | reject malformed calls loudly; detect exact repeats and a static world; step in well before 500 stalled turns; Ling uses a zlib repetition penalty; MiMo MOPD2 fixed repetition defects |
| **Feedback blindness / trusting an unverified tool** | CH B.3: a memory note said "trust this tool", and the agent ignored what the game showed it | don't let memory store unverified directives; skills need independent checks |
| **Wrong assumptions never checked** | CH: the Goldenrod puzzle ran for days; 2.5 Pro took 16,403 turns to the Fog Badge | the Refiner targets stalled goals |
| **Skill and memory bloat** | CH funnel (§4.2) | triage, a bias toward reuse, demotion, deletion rules |
| **Unrepaired new components displace repaired ones** | CH C.2.1 (6.4%) | A/B gate; deprecate a new component whose task an inherited one already covers |
| **An exploit saved as a skill** | PA §3.5 Factorio: RCON commands spawned resources despite an anti-cheat heartbeat, and the trick was saved as a reusable skill | "least-privilege action interfaces, independent state validation, and auditable rollback of contaminated refinements" |
| **Silent failure** | PA EmulatorBench: Opus scored 0.000 "despite successful tool-call responses" | grade outcomes that the environment checks |
| **Irreversible actions** | PA Factorio: a world reset dropped technologies 5 → 1 | checkpoints; the agent continued and recovered |
| **Harness failures counted as model failures** | PA §1: "dropped state … miscounted resources, or terminated prematurely" | standard persistence, recovery and accounting; mask harness faults |
| **The scaffold is not trained** | RLM: math-python falls with the RLM; results swing with tips; INTELLECT-3 cannot use the RLM | train on the scaffold (RLM future work); vary the harness in training |
| **Small students break on a large harness prompt** | CH D.2: tool format 0 at 8K context | budget the harness prompt itself |
| **Search collapse; proxy misranking** | GS (§4.3) | evaluation funnel; "reopen the search" |
| **Predicting output instead of waiting** | GS: Gemini workers guessed command outputs and commit SHAs | one evidence-dependent step per response; `set -e` |
| **Kept working without finishing** | FE: submitting fell from 69% to 41%; calls rose 17 → 21 | the same output cap in training and eval; a bonus paid only on correct answers |
| **Restart regression** | CH D.4: 2–3 weak iterations after each restart | expect it; checkpoint more often |

---

## 6. Graders for long tasks

### 6.1 What PM actually says about requirements (verbatim)

**PM does not recommend hiding requirements.** It says the opposite: put no **unchecked** requirement in the prompt.

**Don't lie to the AI (A/pm-building-rl-environments-for-superintelligence.md):**

- "we absolutely cannot lie to the AI. Lying to the AI leads directly to misalignment."
- **Outcomes, not process** (L15): "The most common way RL environments lie to the AI is by asking it to do something the
  grader never checks." The fix is to "only ask for the outcome: 'write a kernel that matches the fp64 reference within
  this tolerance, as fast as possible, using whatever precision you want.' Everything in the prompt should be faithfully
  checked by the grader."
- **The rule** (L21): "requirements such as forcing the use of a certain approach are not viable … If the grader can't
  check it, it cannot go into the prompt." The example is TerminalBench 4's fp8-rmsnorm-gemm task: it banned Triton but
  checked only correctness and speed.
- **Accept that the model knows it is in an environment:** "assume that a superintelligent AI can always figure out that
  it's in a training environment, and give up realism in exchange for security." The article reports that Opus 5 detects
  training or eval 95% of the time.

**Karotte adds:**

- "If the prompt asks for one thing but the grader rewards something else, the model learns that the grader is what
  actually matters."
- "A hack that is initially tried 10% of the time will soon become the model's default approach."

**R2E adds:** "A prompt instruction is a nudge, not a control … the environment enforces it, the prompt never asks for it."

**The consistent pattern (LHTB, TBS):** state the grading criteria openly and hide the test **data**.

- alp lists all 9 verifier gates but uses hidden, renamed columns.
- apex states its tolerance.
- 2048 explains that it replays the move log.

### 6.2 Dense progress signals that stay honest

| Pattern | Source | Why it is honest |
|---|---|---|
| **Fraction of checked fields** | LHTB apex-ib244 | 38 ordered stages and 146 numeric fields within tight tolerance, no LLM judge. "There is no penalty for taking many steps … Partial work is partially credited." Each field is worth 1/146, so the parts **add up to the outcome**. `validate` checks the schema only and "does not grade correctness" |
| **Replay of an action log** | LHTB 2048 (engine.py:1-20, 140-155) | the verifier replays the logged moves in a fresh engine with the same seed: "a move log cannot fake a score". Score = band(max tile) + min(0.999, (sum − max)/max), open-ended |
| **Milestones read from the environment** | CH App A | 31 Emerald and 18 Red milestones read from game memory (not reported by the agent), with cost counted in button presses |
| **Stepwise diagnostic verifier** | PA EmulatorBench | fraction of human diagnostic programs passed |
| **Potential-based Φ** | Crucible verify/progress.py | Φ = weighted share of grader subgoals; ΔΦ telescopes; xl gate ΣΔΦ = 0.9999996 (checked in the report) |
| **Multiplicative quality gates** | QK MiMo GRS | R = R^test · S^sol · S^beh. Failures stay 0, and the rubric comes from the task rather than from sampled runs |
| **Conserving redistribution** | QK MiMo GAR | λ = Σ_P A_j / Σ_P f_j A_j and A′_i = λ f_i A_i for passing runs; failures are untouched; a confirmed hack is set to 0 and the group recomputed. Without GAR, turns and length grow and the pass rate stalls |

**What is not honest:**

- **Paying for submitting:** FE's "+0.2 for submitting anything" taught the policy to submit at once. Held-out accuracy fell
  0.740 → 0.178 "while training reward still looked healthy."
- **Judged reasoning and format:** CH's step judge gives reasoning 0.2 and format 0.1, which nothing checks, and its step
  scores are not required to add up to the outcome.
- **Shaping in general:** MiniMax's process reward penalises language mixing and format errors, and QK §3.6.4 warns that
  "every shaping term opens a hack surface". GLM's slide reward was gamed by truncating content and adjusting spacing.

**The consensus (QK §3.6.3.4):** spend grader compute on detailed **outcome** scoring, not on a process reward model.
Process rewards are used only by Ling 2.6 and MiniMax.

**Efficiency bonuses that cannot be farmed:**

- **FE:** at most 0.1, paid **only on correct answers**, so "the model can't earn the bonus by giving up early". It was the
  only reward contrast in 17.5–22.6% of groups.
- **MiniMax:** r^speed = h(T/T_baseline), relative to a baseline. SWE end-to-end time fell 31.3 → 22.8 min.
- **Kimi:** critical-path cost.
- **Kimi K3 judge:** a candidate longer than σ·ℓ0 loses automatically.
- **Qwen:** the tool-format penalty applies to the offending tokens only.

### 6.3 Open-ended graders that reward doing better

- **PM, "make tasks deep":** "Is this a task a human expert could pour an unbounded amount of time into and still not
  fully solve?" (the elliptic-curve rank record). "As long as the grader rewards doing better rather than clearing a fixed
  bar, a task like that doesn't saturate." "something like 'implement this feature in a codebase' gets saturated quickly,
  and after that there's no more learning signal."
- **RUST:** "We deliberately gave the loop no numeric targets, since a fixed threshold tends to become a stopping point."
  Improvement was measured noise-gated, in alternating order, with regression guards.
- **LHTB 2048:** the reward is the **peak** score over the session, there is no ceiling ("each further doubling is +1"),
  and reset is allowed. The instruction says "You cannot lose by continuing… Risk-averse stopping is a mistake", and it
  gives explicit stop rules.
- **GS research hill-climbs:**
  - the best **protected** dev score over time;
  - a fixed number of protected dev evaluations (64 in the Loop run, all used);
  - a separate held-out test (RMT: dev 3.158 vs held-out 3.147);
  - capacity caps (≤ 350M parameters and a fixed token budget). Without caps, NanoChat grew 50.3M → 1.72B, and the paper
    discloses it.
- **nanoGPT speedrun (PA §3.3):** a record counts only as an **8-seed mean**. The harness effect was smaller than the noise.

**The grading requirement this implies:** an open-ended grader must

1. score against a baseline or the previous best, not a fixed bar;
2. measure with paired, repeated runs and refuse unstable comparisons;
3. limit how often the search can query the measurement (protected evaluations);
4. keep a held-out measurement apart from the search metric;
5. cap the resources that would let the score be bought (parameters, tokens, wall time).

### 6.4 Honest give-up and abstention

- **PM:** "some tasks may not be solvable at all, and we might want to give the AI an honest way to give up." It gives no
  mechanism.
- **GS:**
  - each episode ends in exactly one action: Contribute, Nominate or **Abstain**;
  - abstaining **does not use up the budget B**, but its tokens still count;
  - 94.9% (1,349 of 1,421) of ProgramBench abstentions were judged justified, and 97.1% of abstainers saw a conflicting
    commit land after they started;
  - "A wrong answer presented as if it were correct scores 0" (IMO grader).
  - `EVALUATION.md` requires "known gaps or risks", and `MY_THOUGHTS.md` requires "unresolved uncertainty".
- **StepFun:** fabricated citations and overconfident claims score 0. A second reward model penalises a right answer with
  wrong reasoning.
- **FE:** because the bonus is paid only when correct, giving up early earns nothing.
- **TBS:** some tasks have infeasible variants, and matpower's hidden manifests include stressed infeasible cases. When
  the outcome is checkable, "infeasible" is a correct answer the grader can verify.
- **Gap:** no source trains a give-up action with RL. §10 P6 proposes how.

### 6.5 Environment awareness

| Source | What the model is told |
|---|---|
| RLM | the cell timeout, how long each call took, and the installed packages |
| MiMo | a `<context_usage>` footer and a wrap-up hint |
| GS | `gs_dashboard`: remaining budget, in-flight work, best protected-dev score; the nomination tally is explicitly "not evidence of correctness" |
| LHTB | how `continue_until_timeout` works |
| PM | that it is in an environment; realism is traded for security |
| TBS | every instruction ends "You have X seconds to complete this task. Do not cheat…" |

### 6.6 Grader feedback channels in long tasks

- **LHTB `continue_until_timeout` (30 of 46 tasks):** when the agent stops, the hidden verifier runs. If it fails, the
  same session resumes **with the failure summary**, until timeout or pass. Stock Harbor drops the flag. This loop is a
  grader-feedback channel by design: fine for evaluating an agent that iterates, but **in RL it leaks hidden-test
  information to the policy**.
- **PA autonomous mode:** "a failed test returns bounded output for another attempt", with an end-condition test after
  every turn.
- **Crucible rule (proposed in §10 P9):** any check whose result the policy can see during the episode is part of the
  task and is listed in the prompt. The outcome grade runs on hidden data the policy never sees mid-episode.

### 6.7 Hacking that matters for long tasks

**What labs saw:**

- **Qwen:** agents re-added git remotes to fetch answers. Any call with both `github.com/{repo}` and git, curl or wget is
  now refused.
- **Ling:** 0.2% of trajectories cheat via git.
- **DeepSeek-V4.1:** kernel and file-system exploits; an environment crash counts as a failure plus a "recoil" signal.
- **LongCat prover:** the training pass rate jumped at step 80 and fell 97.9% → 27.9% once an AST check covering 9 cheat
  patterns was added.
- **GLM-5.2:** blocks a suspicious call and **returns dummy information so the rollout continues**, because aborting
  destabilises training.
- **Kimi K3:** a faked artifact scores 0; public and hidden verifiers are paired.
- **MiMo:**
  - a hack agent probes environments until none can be exploited; confirmed hacks are under 2%;
  - flaky tasks need 8 consistent reruns;
  - **"soft hacks"** pass the tests while hurting the code: speculative compatibility branches, swallowed exceptions,
    relaxed validation.
- **Karotte:** o3 reward-hacked about 30% of RE-Bench runs. `collect_submission` kills the agent's processes, refuses
  symlinks and FIFOs, and copies the submission into a root-only folder. A fake model replays known attacks, and each must
  score 0.
- **R2E:** three perfect scores came from contamination. Only gold = 1.0 and no-op = 0.0 make results meaningful.

**Long-task specifics:**

- The longer the episode, the more chances to find an exploit.
- A **persistent memory or skill layer keeps an exploit across episodes** (PA Factorio).
- **Kimi forks sandboxes** (snapshot 133 ms, resume 49 ms) so reward judging has no side effects.

---

## 7. Evaluating long-horizon agents

| Practice | Source |
|---|---|
| **Score at a fixed spend** (tokens, cost, time) and **score at the practical plateau**; curves of score vs output tokens (10k–3M) and vs cost | PA §1, Fig 5, 7 |
| **Accounting over the whole agent tree**; tokens, time and cost reported separately; an event log tying each call, retry, verifier result and harness edit to the config | PA §2.6 |
| **Main-model token efficiency** = reward / main-model tokens, sub-LLM tokens excluded, normalised to the plain LLM | RLM |
| **Score vs input tokens and output tokens**; input counts the full context resent on every request | GS |
| **At least 3 seeds**, seed medians and each seed shown; 24-hour runs; cost–completion Pareto | CH §4.1 |
| **Same output cap in training and eval**; per-harness reporting; 1–2 points of single-run noise | FE (16,384 vs 4,096 caused 556 of 1,000 eval cells to be cut off) |
| **Process metrics:** memory pull rate; clean return and back-on-goal; created → used → succeeded funnel; skill success before and after each edit; inherited-component share; reuse and ancestry; how often abstentions were justified; when the winner appears (93–98% of a ProgramBench run, 8.6–17.5% on IMO) | CH, GS |
| **Causal claims about memory** need matched runs that resume with and without the earlier history | GS §4 |
| **Headline vs training signal:** LHTB reports `lhtb_solved` = count of tasks with reward ≥ 0.95, retiring the mean-reward id. TBS enforces binary reward: a fractional reward "is simply a failure that reads like progress"; diagnostics go to metrics.json; the no-op agent must score 0 | LHTB eval.yaml; TBS CONTRIBUTING.md:638-652 |
| **Read the trajectory:** "A pass rate moves for reasons unrelated to your fix … Open the run and confirm" | TBS CONTRIBUTING.md:772 |
| **Sampling pitfalls:** Oolong is sorted by size, so taking prompts in order selects only short ones; an API-rejected prompt counted as 0 tokens | RLM |
| **Forced-continuation baseline:** a single agent repeatedly told "Keep working", with its intermediate checkpoints graded | GS (ProgramBench 63.1–65.1% vs 73.1–79.4% for the swarm) |

---

## 8. Crucible's current long-horizon design, checked against the code

Checked in the code and reports, file:line. Report: `docs/modules/reports/longhorizon_xl.json`. Gate:
`packs/_toy/src/crucible_pack_toy/longhorizon/gate.py:32-67`. `tests/unit/test_longhorizon.py` was rerun: 7 passed.

### 8.1 xl gate claims vs the report

| Claim (long-horizon.md / DESIGN §10.4) | Report / code | Verdict |
|---|---|---|
| 1,200 tickets / 7,623 steps / 7,627 calls | 1200 / 7623 / 7627 | checked |
| 1,114,725 trajectory tokens | 1114725 | number checked; **meaning overstated** (below) |
| 128k budget, 4 compactions | 128000 / 4 | checked |
| largest context before a compaction 102,853 ("103k" in DESIGN) | **102,652** | **stale** |
| crash on attempt 1, resume from step 3000 | the crash came at 3,495 steps; resumed from a1@step3000 | checked |
| reward 1.0; ΣΔΦ = 1.0000; no drift; credit sums to 1 | 1.0; 0.9999996; []; 1.0 | checked |
| wall time 66 s | **77.4 s** | **stale** (the report was regenerated in the M10 audit) |
| 7 tests | 6 functions, one parametrized ×2 | checked; 7 passed |

**What the docs do not say:**

1. **"Tokens" are estimates.**
   - `approx_tokens` is chars/4 (`actors/policy.py:127-135`).
   - ReplayLLM reports a fixed `completion_tokens: 40` on every call (`runtime/inject.py:121`).
   - So about 305k of the 1.11M are synthetic 40-token replies, and the rest are chars/4 estimates.
   - No tokenizer ever measured the episode. "Over a million tokens" is a chars/4 plus synthetic figure.
2. **The "scripted LLM" is an oracle action list.** ReplayLLM (`inject.py:93-133`) finds its position from `rp<k>`
   tool-call ids.
   - Its summary is the template "actions 0..k-1 of the plan are done".
   - **No decision depends on what the summary says.**
   - So the gate tests plumbing: compaction, segments, checkpoint/resume and Φ telescoping.
   - It does **not** test summary fidelity, remembering the 50 → 25 amendment, revisits after compaction, or any real loss
     of context.
3. `test_xl_preset_exceeds_a_million_tokens` (`tests/unit/test_longhorizon.py:87-91`) asserts only "more than 7,000 actions
   and 1,200 tickets". The name claims more than the test checks.
4. **An undocumented failing run.** `docs/modules/reports/labstack/phase_longhorizon.log` (40 tickets, 32k budget, 229
   steps) says `"pass": false`. The default crash point (call 3500) never fires in a 229-step run, so the resume condition
   cannot be met.
5. The reward of 1.0 is Φ over 1,200 per-ticket subgoals, i.e. the apex fraction pattern.

### 8.2 What is implemented beyond what long-horizon.md says

**`ContextConfig` (`actors/policy.py:81-101`):**

- `budget_tokens` 128,000, `trigger` 0.8, `keep_recent_turns` 3, `summary_max_tokens` 2,000.
- Modes:
  - `compact` (the harness triggers at trigger × budget);
  - `action` (a `context_compact` tool, MiniMax style);
  - `none` (an overflow ends the episode, scored).
- Strategies:
  - `summary`;
  - `keep_recent` (GLM);
  - `discard_all`;
  - `tiered` (older tool outputs shrink to 200 chars, then a summary if still over);
  - `hybrid` (LongCat, `hybrid_summaries` 1).
  - These came from M13 C-7 and are tested in `tests/unit/test_context_modes.py`.

**Rebuild (`policy.py:229-306`):**

- system + task verbatim + summary + the last k complete turns, with no fabricated turns;
- an empty summary is resampled once, then `summary_failed` keeps the recent turns;
- a summary cut at its length limit is labelled as cut.

**Elsewhere:**

- **Context size:** the provider's count plus chars/4 for later messages (`policy.py:188-200`).
- **`RLMConfig` (`runtime/harness/rlm.py:48-73`):**
  - output 8192 chars; cell 120 s;
  - `max_turns` 500;
  - sub-calls 256 per episode / 64 per cell / 4 concurrent / 2M tokens;
  - the same compaction defaults;
  - a Docker kernel.
- **Cross-segment GAE has landed, but long-horizon.md and DESIGN §10.4 still list it as Remaining.** `train/objectives.py:228-288`
  and `train/trainer.py:270-289` implement CompactionRL:
  - local GAE per segment;
  - the shared terminal R on each segment's last trainable token;
  - (γλ)^{N_{>s}} scaling;
  - StepFun truncation bootstrap;
  - sub-calls as their own sequences with the outcome as their terminal.

  Defaults: `CriticConfig.gamma` 1.0 and `lam_policy="adaptive"`, i.e. 1 − 1/(1.5·T) over the rollout's trainable length
  T (`train/critic.py:26-36`). With these, (γλ)^{N_{>s}} ≈ exp(−N_{>s}/(1.5T)), so the scale is never below ≈ e^{−2/3} ≈
  0.51 however long the rollout. With a fixed `lam_policy=1.0` the scaling does nothing. This is within CompactionRL's
  formula, but the effect is milder than the per-token discount the paper describes. Worth a test that pins the intended
  magnitude.

**Still missing:**

- observation offloading and a memory app;
- compaction mode varied per GRPO group;
- `calls.jsonl` streaming;
- per-segment ATIF export;
- staged milestones and a fast feedback loop;
- a real-model long-horizon run;
- a boundary-reason / trainable field in `Segment` (`core/schema/rollout.py:108-117`).

**01-frameworks.md external claims spot-checked, all hold:**

- verifiers: 16,384 reserve and 20 KB truncation;
- MiMo: 100k/1M budgets with a 10k buffer, a 900k Codex trigger, the 0.8 footer, and the train.yaml values;
- Harbor: 8,000 / 1,300 / 3;
- MemAgent: 1,024;
- the LHTB facts.

Its "TB-science 8 h" holds for 70 of 75 tasks (5 are 5 h).

---

## 9. Requirements for Crucible's very-long-horizon RLM harness

Parameters are defaults to start from, each tied to a source. **Has** marks what exists today (§8); everything else is
new. "Episode" means thousands of World steps and 1–10M tokens across segments.

### Context and state

**H1. Three tiers of state; compaction loses nothing on disk.**
- Prompt (L1): system prompt, task, the RLM's recent turns.
- Kernel (L2): variables.
- Disk (L3): an append-only event log *outside* kernel memory, holding every cell, tool result, message, summary and edit.
- Read L3 from the REPL with paged, searchable helpers (`history.search(regex)`, `history.cell(i)`, `history.since(step)`).
- Has: `HISTORY` in kernel memory. A 10M-token episode needs disk-backed, indexed paging.
- Sources: PA §2.2; SWARM.

**H2. Output and time limits, told to the model.**
- REPL output cap 8192 chars per cell; cell timeout 120 s. Report each cell's elapsed time and the remaining
  turn / token / wall-clock budget in every observation.
- World tool results enter `OBS` in full. Only a middle-truncated view (≤ 40,000 chars) is printed.
- Sources: RLM; MiMo; Karotte.

**H3. Compaction, both modes kept.**
- **Harness-triggered:** fire when remaining budget < max(10,240 tokens, 0.2 × window), i.e. CompactionRL's absolute
  floor next to the existing 0.8 trigger.
- **Policy-triggered:** a `compact()` action, with a `<context_usage>` footer from 80% of usage.
- **Rebuild:**
  - system prompt;
  - task verbatim;
  - a boundary marker (compaction count, tokens before);
  - the structured summary;
  - the last k = 2–3 complete turns;
  - a pointer to L3;
  - the line "state the gap before relying on missing details".
- **Summary schema** (MiMo's 9 sections, adapted):
  - goal and done-criteria;
  - **constraints and rules in force, with when each changed**;
  - every user or supervisor message, verbatim;
  - entities and ids touched;
  - pending and waiting items with their triggers;
  - errors and fixes;
  - the current hypothesis;
  - the next step, quoted verbatim.
- **Summary size:** `summary_max_tokens` 2,000 for a 128k window, scaling to about 16k for a 1M window (SWARM's ~60×).
- **Wrap-up hint:** a separate hint at 0.9 of the *episode* budget (turns, tokens or wall time): finish, do not compact.
- Has: compaction, task verbatim, keep 3, a 2,000-token summary. **New:** the 10,240 floor, the schema, the boundary
  marker, the gap line, the wrap-up hint, the scaling.

**H4. Context policy varied across training groups and recorded.**
- One compaction mode per GRPO group, so the harness id stays one per group.
- Rotate modes across groups: summary / keep-recent k=5 / tiered 32k / hybrid at 80K / action.
- Train so the policy is robust to whichever mode an evaluation uses. Record the mode in the harness id and in each
  `Segment`.
- Sources: Kimi K3; MiniMax. long-horizon.md already lists this as Remaining.

**H5. Thinking-history policy is a harness setting.**
- Options: `preserved` (Kimi, GLM, DeepSeek) and `selective-latest-instruction` (StepFun).
- Never drop thinking *inside* a tool chain. It must match between training and evaluation.
- Has: thinking modes in `test_context_modes.py`. **New:** the StepFun selective mode.

### Delegation

**H6. Sub-LLM delegation scaled for millions of tokens.**
- `llm_query` / `llm_batch` to fresh same-model copies. Depth defaults to 1 and can be set to 0–3.
- Sub-call budgets scale with the episode instead of fixed numbers. Today: 256 calls / 2M tokens.
  - Proposed: per segment 256 calls and 2M tokens, plus an episode cap.
  - Plus a critical-path wall-clock budget: CriticalSteps = Σ(main + max sub).
- Optional **tools-only-through-sub-LLMs** mode, so tool output never reaches the root (RLM).
- **Persistent children:** async handles that take follow-up questions after the parent compacts (PA `rlm()`, SWARM),
  with a returned answer capped at about 2k tokens.
- Children get **least privilege**: no inherited credentials, network or write access unless granted.
- Training rules:
  - Has: other-model sub-calls are recorded, never trained; same-model sub-calls share the parent's credit.
  - New, optional: a PARL mode that freezes sub-agents and trains only the root.
  - Never reward the number of sub-calls; pay only for sub-tasks that verifiably finished (r_finish).

### Memory and harness edits

**H7. Typed, versioned memory, never credited by Φ.**
- Entry types: notes, facts, skills (code), sub-agent specs. They live as files in the kernel workspace and are
  snapshotted into L3.
- **Catalog in every prompt** (id, title, importance, last-used step); bodies loaded on demand.
- Record per-entry pull-rate and success counters. Demote stale entries. Run GC of entries that were created and never
  used.
- Every write records its trigger and intended effect; rollback by version.
- **A skill becomes persistent (outlives the episode) only after independent validation** in a fresh sandbox, run by the
  grader side and checked against the World's rules. An exploit saved as a skill (Factorio) must be impossible to carry
  across episodes.
- Sources: PA §2.5; CH; GS. Has: `harness_state.py` with typed entries, versions, rollback, usage counters. **New:** the
  catalog-in-prompt, pull rate, validation before cross-episode persistence, and skills/memory living in the
  episode-scoped kernel.

**H8. Harness edits across episodes: keep M14's gate.**
- Accept an edit only on a paired, interleaved A/B over at least 3 seeds with a noise gate. Unstable results are
  withheld, and neighbouring tasks are checked for regressions (RUST, M14).
- *Within* an RL episode, edits are ordinary policy actions credited by the outcome. They never carry a reward of their
  own.
- Deprecate a new component when an inherited one already covers its task (CH C.2.1).

### Robustness

**H9. Detect stalls and loops; reject malformed calls loudly.**
- Malformed tool or host calls return an explicit error. Never silently fall back (CH's 842 silent repeats).
- **Detectors:**
  - exact-repeat payloads (the same World call ≥ 20 times in a row);
  - a static World-state hash over 100 steps;
  - ΔΦ = 0 over 200 steps.
- **Escalation:**
  - at the first trigger, inject a stall notice that names the evidence;
  - at 3 triggers in a row, force a compaction / reflection turn (the window where models still build new tools is
    50–200 turns);
  - at 500 stalled turns, stop with a scored `stalled` outcome, never an infra mask.
- Has (`actors/rlm/policy.py:21-22, 547-557`):
  - stall signatures: the same cell 3 times (`stall_repeat` 3), the same World call with the same result 3 times, and
    consecutive cell errors (`max_cell_errors` 4);
  - each one injects a `[stall-detected]` notice;
  - the episode stops after `max_stalls` 3.
- **New:**
  - a static World-state hash and a Φ plateau, which catch slow loops that alternate between a few actions and never
    repeat exactly;
  - a forced reflection / compaction turn before the stop;
  - a `stalled` scored outcome.
- A stop after 3 exact repeats suits short episodes. For thousands of steps, decide whether to keep the 3-strike stop or
  escalate as above, and record the choice in the harness id.

**H10. Goal persistence and an explicit end.**
- The episode ends only through `answer["ready"] = True` / `rlm.finish`, or a budget limit.
- The goal and done-criteria survive every compaction verbatim (H3).
- An empty turn gets a nudge to continue, not an end (Karotte).
- An end-condition check after each turn may return *public* results only (H12, P9).

**H11. Checkpoint and resume over hours.**
- Snapshot the World and kernel (journal replay or a container snapshot) every 500 steps and at each compaction.
- A masked crash resumes from the latest snapshot without a restart of the episode.
- Target the snapshot/resume speeds Kimi reports (133 / 49 ms) via container checkpointing where possible.
- Report how many steps were lost to each resume.
- Has: a checkpoint every 500 steps; kernel journal replay. **New:** a snapshot at each compaction, a resume in labstack
  that actually fires, and a real crash point (the labstack run failed for exactly this reason).

### Measurement and records

**H12. Real token accounting and a real-model gate.**
- Count tokens with the policy's tokenizer (TITO records), never chars/4, for any "million-token" claim.
- Account separately for:
  - root input and output;
  - sub-call input and output;
  - compaction calls;
  - wall time;
  - cost.
- Report main-model token efficiency (RLM).
- **Replace ReplayLLM in the gate** with a summary-dependent check:
  - a real model, or a scripted model that acts on what the summary *says* (for example, it reads the threshold and
    pending ids from the summary text);
  - so that forgetting the 50 → 25 amendment or an awaiting ticket after compaction costs reward.
- Measure a **lost-constraint rate**: actions after a compaction that break a rule in force.

**H13. Training records per segment.**
- Each segment is an exact-prefix TITO sequence.
- The trace is a message graph (VF1): N branches give N samples.
- Add `boundary_reason` (compaction / summary_failed / rerender / sub / resume) and `trainable` to `Segment`.
- Summaries are trained as policy tokens on the shared outcome, with **no summary reward and no extra KL**.
- **Loss normalised per token across the rollout's segments**, so rollouts that compact more get no extra weight. M12's
  rollout row-weights must give every trainable token of a multi-segment rollout the same weight as in a one-segment
  rollout.
- Has: segments, linear deltas, cross-segment GAE. **New:** the schema fields, and a test of segment-count invariance.

**H14. Security for long runs.**
- The kernel is least-privilege (Has: Docker, no network, read-only root).
- Grader state, hidden data, the event log of record and the checkpoint store sit outside the kernel's write reach.
- On a detected hack attempt, **block the call and return an explicit refusal** while the rollout continues (GLM-5.2,
  Qwen). Record `metadata["misbehavior"]`.
- A fake-model attack replay (Karotte) must score 0 for every long-horizon pack.

**H15. Environment awareness.**
- Every observation footer carries:
  - step and turn counts;
  - context usage;
  - remaining episode budget;
  - the last cell's elapsed time;
  - the count of sub-calls left.
- Do not pretend realism (PM).

**H16. Evaluation outputs** (§7).
- Score vs tokens and score vs cost curves; score at a fixed spend and at the plateau.
- ≥ 3 seeds; the same caps in training and eval.
- Process metrics: pull rate, clean return and back-on-goal per sub-agent kind, the skill funnel, the lost-constraint
  rate, stall count, compactions per episode, and tokens per segment.

---

## 10. Reward design for long tasks

**P1. Everything in the prompt is checked, and nothing the grader checks is concealed.**
- Requirements, rules and grading criteria are stated. Only the test *data* is hidden.
- No process instructions ("use X", "don't use Y") unless the World enforces them.
- **Proposed pack lint:** every imperative in `instruction.md` maps to a grader check or a World guard. An unmapped
  instruction fails `pack check`.
- Sources: PM, Karotte, R2E, LHTB.

**P2. Dense progress is a potential over checked subgoals.**
- Φ ∈ [0, 1], ΣΔΦ = Φ_T − Φ_0, and the final reward equals the outcome. Each subgoal is a grader check on World state, a
  replayed log or an artifact (the apex fields, 2048 replay, CH milestones).
- **No credit for:**
  - submitting or attempting (FE's 0.740 → 0.178 collapse);
  - judged reasoning or format;
  - memory writes, notes, summaries or sub-call counts.
- Has: Φ with telescoping and drift checks; memory not credited. Keep the full re-check every 250 steps.

**P3. Bonuses only on top of correct work, and bounded.**
- Efficiency and speed bonuses ≤ 0.1, paid only when the outcome passes (FE).
- Measure them against a baseline (MiniMax: T / T_baseline) or the critical path (Kimi, DeepSeek), never raw counts.
- A length/verbosity cap acts as a gate (Kimi σ·ℓ0), not a slope.
- Format penalties apply to the offending tokens only (Qwen).

**P4. Gate quality multiplicatively.**
- Rubric or reviewer scores multiply the verified outcome (R = R_test · S_quality, MiMo GRS) or redistribute within the
  passing set while conserving total positive advantage (GAR).
- A failure stays 0. A confirmed hack is 0, and the group stats are recomputed.
- Rubrics come from the task, not from the best sample.
- This is how "soft hacks" (swallowed exceptions, relaxed validation) are caught.

**P5. Open-ended "do better" graders.**
- **Score:**
  - improvement over a baseline or the best so far, with no fixed bar (PM, RUST);
  - the peak over the session when resets are allowed (2048);
  - an unbounded band scale normalised by a reference (e.g. reward = clip((score − baseline) / (reference − baseline), 0,
    1.5));
  - the bench reports raw score next to reward.
- **Measurement:**
  - paired, alternating-order runs;
  - ≥ 3 seeds (8 for record claims, as in nanoGPT);
  - a noise gate that refuses unstable comparisons;
  - an evaluation funnel (128 → 512 → full).
- **Limits:**
  - a budget of protected evaluations the policy can call (e.g. 64 per episode);
  - a held-out final measurement apart from the search metric;
  - caps on the resources that buy score (parameters, tokens, wall time).

**P6. Honest give-up: a give-up should beat a false claim and lose to real success.**
- Give the policy a `give_up(reason, evidence)` action, or `answer["status"] = "infeasible"`. Proposal for Crucible,
  built from PM, GS and StepFun:
  - **Infeasible task, give-up claimed:** the claim is checkable (for example a stressed infeasible variant), so a correct
    "infeasible" scores like success.
  - **Feasible task, give-up claimed:**
    - the reward is the Φ already earned, with **no outcome credit**;
    - no penalty beyond the lost outcome;
    - no efficiency bonus (FE).
  - **False success claim** (the outcome check refutes it): reward 0, overriding Φ, and a flag (`crucible:false_claim`).
    The IMO rule: "a wrong answer presented as if it were correct scores 0".
- Ordering: real success > honest give-up (keeps partial Φ) > confident wrong claim (0).
- To keep give-up from becoming a lazy optimum, every give-up pack must contain feasible tasks that the oracle solves,
  and the bench reports the give-up rate on feasible tasks.
- Mixed packs (some variants infeasible) teach calibration.
- GS shows that justified abstention can be measured (94.9%).

**P7. Truncation is not failure, unless finishing is the task.**
- With a critic, a limit stop bootstraps V(s_T) (StepFun; Has: `truncation="bootstrap"`).
- Where finishing in budget is part of the task, the limit stop is the scored outcome (Has: `"reward"` mode).
- Never mix the two within one GRPO group.

**P8. Infrastructure faults are masked, never scored.**
- A verifier that never ran gives reward None (FE).
- Sandbox or World crashes mask by host-side evidence.
- A policy-caused crash (fork bomb, filled disk) is scored. Karotte gives the harness 1 GiB of reserve and the agent the
  highest OOM score, so it is the agent's process that dies.

**P9. Grader feedback inside an episode only from public checks.**
- `continue_until_timeout`-style loops may return *public* check results (stated in the prompt).
- A hidden-test failure summary must never reach the policy in RL (LHTB's evaluation loop is a leak channel for
  training).
- An end-condition test after every turn (PA) uses public checks only.

**P10. Hack-resistant grading, made stronger for long episodes.**
- The grader replays logs rather than trusting the agent's reports ("a move log cannot fake a score").
- Grading runs outside the policy's write reach, after `collect_submission`-style process kill and a copy into a
  root-only area.
- Every long pack gets a hack-agent and fake-model replay.
- Online per-call monitoring that blocks and continues with a refusal (H14).
- Gold scores 1.0 and no-op scores 0 before anything counts.

**P11. Report a thresholded headline; train on the dense signal.**
- Training uses the dense Φ outcome.
- The bench headline is a solved count (reward ≥ 0.95, LHTB) or binary (TBS), with mean reward as a secondary.
- A fractional score shown as the headline "is a failure that reads like progress" (TBS).

**P12. Spend grader compute on outcome scoring, not on a process reward model** (QK consensus).
- Process signals that are not checks (reasoning quality, plan quality) stay diagnostics, never reward. CH's
  reasoning-weighted judge is the counter-example.

---

## 11. What the sources leave open

**Training:**
- How to credit summary quality beyond the shared outcome. Every source uses outcome credit, and CompactionRL's GAE fix is
  an approximation with no bias bound.
- RL training of an RLM (persistent kernel, sub-calls). No source reports it. PA and RLM list it as future work.
- Honest give-up trained with RL: only abstention *measurement* exists (GS).

**Tuning:**
- Values for F and W in CH's edit cadence (never given).
- Whether harness-varied training (Kimi K3) or a policy-chosen mode (MiniMax) transfers better. No source compares them.

**Evaluation:**
- Long-context benchmarks for an RLM where models trained on their own harness reach parity (PA Table 1). The gains show
  up mainly for models *not* trained on the comparison harness.
