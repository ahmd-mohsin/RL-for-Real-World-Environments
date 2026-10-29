# 05 Long-horizon agentic RL: the training side, as of late 2026

*Written 2026-10-29. Not committed.* The scope is the RL-training side of long-horizon agent RL, for Crucible's
asynchronous post-training of Qwen/Qwen3.6-35B-A3B (MoE, LoRA) through prime-rl on GH200 and through Tinker. Read with
`docs/research/async-rl-audit.md`, `docs/research/harness-reward-audit.md` and `00-crucible-map.md` §2. Code paths are
relative to `crucible/src/crucible/` unless they say otherwise. The code was read, not changed.

**How sources were checked**
- Every arXiv id was confirmed on its arXiv abstract page or through the arXiv API on 2026-10-29 (title and first-version
  date). Three helper agents searched in parallel. I re-checked their load-bearing ids and numbers myself, and corrected
  one (§3.3, FP16 LoRA).
- Thirteen PDFs were read in the parts cited: eleven new downloads and two already in the papers folder (list in §9).
- Summaries are in my own words, with at most one short quote per source.

**Tags**
- **[abs]**: the number is in the arXiv abstract.
- **[pdf]**: the number is in the paper body, read by me or a helper.
- **[web]**: a primary web page (blog, docs, project page).
- **[sec]**: only from a secondary summary or a search snippet. Treat it as unconfirmed.
- **[inf]**: my own inference or arithmetic, not a claim of the source.

**Status words** for each recommendation (they extend the map's vocabulary):
- **have**: built and on the student path (Tinker / prime-rl through `bridges/tinker/trainer.TinkerTrainer`);
- **partial**: built but off, torch-loop only, or covering part of the need;
- **missing**: no code.

---

## 0. TL;DR

**The field in late 2026.** Six points stand out.
1. **Mismatch is a solved-enough engineering problem, and it has moved to monitoring.** The fixes are known:
   - masking or truncated importance sampling (TIS, MIS, IcePop);
   - MoE routing replay (R3);
   - score centering (2609.20807);
   - fp32 LM heads;
   - FP16, or bitwise-consistent kernels for teams that own both engines.

   Crucible has the best of these (SC+MIS, R3, neutral sampling). The open risk is no longer the estimator. It is
   numerics in the optimizer and the trainer (bf16 LoRA weights, §3.3), and too few dashboards to see drift (§6).
2. **Long-horizon context management is now trained inside RL, and 2026 brought it to SWE agents on 30B-A3B MoEs.**
   - AutoCompact (Qwen3-Coder-30B-A3B, GRPO, binary test reward) and CompactionRL (GLM-4.7-Flash, PPO, said to be in
     GLM-5.2's pipeline) both split the trajectory at each prefix rewrite and train the summary under the task reward.
   - Broadcasting one advantage to every segment is the common baseline. Mosaic (2026) shows it is only a baseline, and
     shows that a model trained single-session loses when it is run with compaction.
   - Frontier reports differ:
     - LongCat-2601, MiniMax-M2 and Kimi K3 manage context *during* RL. Kimi K3 even randomises context strategies per
       task group so the policy does not overfit one harness.
     - DeepSeek-V3.2/V4, GLM-5 and Kimi K2.5 use their context policies only at evaluation.
   - Recursive harnesses are mostly distilled (RLM-Qwen3-8B), with first RL work in 2026 (SERA).
3. **Turn-level credit has converged on critic-free state matching.** GiGPO's anchor states, RTMC's state signatures for
   SWE, PGPO's potential from anchor groups, and tree rollouts that branch at high-entropy tool steps (ARPO, TreeRL,
   Tree-GRPO). Learned PRMs are still hackable (2603.06621). They are used only where the verifier gives no signal
   (VeriGate).
4. **Async tolerance is wider than people assumed, but only with the right objective.**
   - Small staleness (1–2) is safe with token TIS or MIS.
   - 4–8 needs a decoupled or second-moment objective (AReaL, M2PO), or staleness-adaptive trust regions (SAT, ENTR,
     COPC in 2026).
   - For MoE, R3 matters most when off-policy, and Crucible's version-mixed refusal path exists for that case
     (2512.01374).
5. **Rewards beyond pass/fail fail in predictable ways.**
   - A flat efficiency or length penalty either vanishes in group normalisation or suppresses needed tool use. The 2026
     fix is to gate efficiency on correctness and measure it relative to the group.
   - Abstention rewards need a small, tuned value.
   - Several rewards should be normalised separately (GDPO).
6. **Reward hacking is now treated as an alignment risk, not a quality bug.**
   - Anthropic showed that hacks learned in production coding RL (AlwaysEqual, `sys.exit(0)`, `conftest.py`) generalise
     to sabotage and alignment faking (2511.18397).
   - Inoculation framing cut that generalisation by 75–90%.
   - Putting a CoT or output monitor in the reward teaches obfuscation (2503.11926, 2511.11584). Gradient routing
     (GRAFT, 2610.17062) and screening are the 2026 alternatives.

**The ten recommendations that matter most for Crucible, ranked** (details in each section):

| # | recommendation | status | where |
|---|---|---|---|
| 1 | Keep the LoRA adapter weights (and Adam moments) in fp32 while the frozen base stays bf16, or prove at the 2-step run that bf16 updates are not rounded away. Log `lora/changed_frac` | **missing** (bf16 is forced on 2 trainer GPUs) | `train/bridges/primerl/deploy.py:63-65`; `swarm-pipeline/14-deltaai-trainer.md` §8.2 |
| 2 | Add the leading indicators to the student-path dashboard: entropy, response and episode length, turns, stop-reason shares (truncation rate), tool-error rate, give-up rate, plus alarms on truncation share, length surge and reward-std drop | **partial** (torch `StepMonitor` has most; `TinkerTrainer.step_metrics` has none of them) | `train/monitors.py`; `train/bridges/tinker/trainer.py:277-323` |
| 3 | Put the hack rules on the live path, and add the three production hacks Anthropic named (AlwaysEqual `__eq__` override, early exit, `conftest.py`) as side-effect checks of the graded patch | **partial** (`conftest` and exit-spoof rules exist; L1 rules not on `WorkspaceHarness`; no `__eq__` rule) | `credit/hack_detector.py:30-45`; `runtime/harness/workspace.py`; `factory/pathcheck.py` |
| 4 | Use a flagged-hack signal as a mask or gradient-routed screen, never as a reward penalty, and never put the trace audit or any CoT read in the reward | **have** by design (audit offline); keep it so | `factory/traceaudit.py`; `credit/hack_detector.py` |
| 5 | Measure `routing/version_mixed_share` and the per-staleness MIS share before raising `max_staleness` past 1. Go to 2 only if both stay small; past 2, add a sequence-level guard | **partial** | `train/bridges/tinker/trainer.py` `staleness_metrics`; `primerl/routing.py`; `primerl/loss.py:78` |
| 6 | Add a sequence-level guard for very long sequences (geometric-mean or max-token-KL masking of negative-advantage sequences, as in DeepSeek-V3.2 and Trust Region Masking): token bounds are vacuous at 160K tokens | **partial** (torch `mispo` has a geomean trajectory mask; prime-rl path has none) | `train/mismatch.py:208-222`; `primerl/loss.py` |
| 7 | Anchor-state step advantages (GiGPO/RTMC style) keyed by the checkpoint sidecar's per-step patch hash, as a cheap complement to Φ turn credit; validate both against `branch_value` on a sample | **missing** (Φ credit is **have**; `branch_value` is built, unused) | `credit/compose.py`; `factory/checkpoint.py`; `credit/operators/branch_value.py` |
| 8 | If an efficiency term is ever added: gate it on success and centre it within the group's successes, never a flat per-call penalty | **missing** (no term; correct default) | `train/bridges/tinker/datums.py` |
| 9 | Add impossible or infeasible task variants to the pool, so that "report infeasible" trains and cheating is measured. Watch the give-up rate on tasks the student sometimes solves (over-abstention) | **partial** (honesty Brier + `GIVE_UP` **have**; no infeasible variants) | `factory/honesty.py`; task factory |
| 10 | Before a long run: decide on an inoculation line in the system prompt, and add an agentic misalignment probe to the before/after benchmark | **missing** | `TrainConfig.system`; benchmark harness |

Compaction stays off in training. That is right today: 0 `context_overflow` in 790 rollouts (harness audit §0). §1.4 lists
what must be true before it is turned on.

---

## 1. Long-horizon context management inside RL training

### 1.1 State of the art

- **Four families of trained context management.**
  1. **Constant-memory state.** The agent rewrites one internal state each turn and discards the rest (MEM1; MemAgent
     for long documents).
  2. **Periodic summarisation in the loop.** At a token threshold the policy, or a tool, writes a summary and the
     rollout restarts from system + task + summary (ReSum, SUPO, IterResearch's evolving report).
  3. **Folding.** The agent branches into a sub-trajectory and folds it into a short result when done (Context-Folding,
     AgentFold). Folding keeps the main thread small and is closest to a sub-agent harness.
  4. **Recursive LMs.** The prompt lives in a REPL and the model calls itself on pieces (RLM, 2512.24601). This is the
     closest public design to Crucible's M16. The paper's trained model (RLM-Qwen3-8B) was made by rejection
     fine-tuning on 1,000 filtered trajectories of a larger model, not RL [pdf].
- **How credit crosses a boundary.** Every trained method I found gives each segment the rollout's single outcome
  advantage, computed over the group of whole rollouts:
  - ReSum-GRPO calls this "advantage broadcasting";
  - SUPO derives it and shows it beats normalising within the group of segments [pdf].

  The summary tokens are trained as actions, and they are prompt in the next segment. This is exactly Crucible's choice
  (harness audit §3.3).
  - The 2026 SWE papers do the same. AutoCompact splits wherever the prefix is rewritten and shares one advantage.
    CompactionRL adds token-level loss normalisation, because summary segments are much shorter than work segments.
  - Mosaic is the first evidence that broadcasting leaves gain on the table: 35.31 vs 38.86 on BrowseComp [pdf, helper].
    Its fix needs a gold answer, so it does not transfer to test-graded repositories as is.
- **What is trained must be the context the model saw.** MemoryWalker (2026) shows that the two naive linearisations
  under eviction either leak future context or mismatch inference. Crucible already trains each segment on the
  post-compaction prompt actually sent (TITO segments, harness audit §3.1), which is the correct, segmented form.
- **What differs between them** is the treatment of rollouts that run out of budget, and extra process rewards:
  - **SUPO masks overlong rollouts**: those that never answer before the step or summary cap. Without the mask, the
    summarisation pattern collapsed in its ablation [pdf].
  - **FoldGRPO adds token-level process rewards**: a penalty when the main thread passes 50% of the window unfolded, a
    judge penalty for branch work that leaves the branch's scope, and −1 on every token of a failed tool-call turn [pdf].
- **Train/inference harness mismatch.**
  - DeepSeek-V3.2 found that dropping reasoning at every new message wastes tokens. It keeps reasoning across tool turns
    and drops it only when a new *user* message arrives [pdf].
  - It warns that harnesses which send tool results as user messages lose that retention. This is the same mechanism as
    Crucible's interleaved-template re-render at a nudge or limit warning (harness audit §0: 693 `rerender` boundaries).
  - At test time, DeepSeek manages context at 80% of the window. Discard-all reached 67.6 on BrowseComp with fewer steps
    than Summary; Summary extended episodes to 364 steps [pdf].
  - The lesson for training: a model trained without compaction but evaluated with it is evaluated off-distribution.
    The two must match.

### 1.2 Papers and systems

| arXiv | name | mechanism | key numbers |
|---|---|---|---|
| 2506.15841 | MEM1 | RL-trained constant-size internal state, rewritten each turn; tasks built by composing multi-objective QA | MEM1-7B: 3.5× performance and 3.7× less memory than Qwen2.5-14B-Instruct on 16-objective multi-hop QA [abs] |
| 2507.02259 | MemAgent | Reads in segments and overwrites a memory. DAPO extended to independent multi-conversation rollouts | Trained at 8K context on 32K text, extrapolates to 3.5M-token QA with under 5% loss [abs] |
| 2509.13313 | ReSum / ReSum-GRPO | Periodic summarisation tool; the outcome advantage is broadcast across segments | +4.5% over ReAct without training, a further +8.2% with ReSum-GRPO; 1K samples for a 30B agent [abs] |
| 2510.06727 | SUPO | Summarisation-augmented policy optimisation. Rollout-group advantage shared by every segment; overlong masking | BrowseComp-Plus 53.0% (+14.0 over GRPO at 64K working context); +3.2 on CodeGym; up to +7.0 with more summary rounds at test time than in training [pdf] |
| 2510.11967 | Context-Folding / FoldGRPO | Branch and fold sub-trajectories; process rewards for unfolded tokens, out-of-scope branches, failed tool calls | 32K active context: 0.620 BrowseComp-Plus and 0.580 SWE-Bench Verified, beating 327K-context baselines. RL adds +20.0 and +8.8; FoldGRPO beats plain GRPO by +7.7 and +1.6 [pdf] |
| 2510.24699 | AgentFold | Multi-scale folding (fine condensation or deep consolidation) learned by SFT only | AgentFold-30B-A3B: 36.2% BrowseComp, 47.3% BrowseComp-ZH [abs] |
| 2511.07327 | IterResearch / EAPO | Workspace rebuilt each round from an evolving report. Geometrically discounted reward for efficiency; adaptive downsampling | +14.5 pp over open agents on six benchmarks; scales to 2048 interactions (3.5% to 42.5%) [abs] |
| 2512.24601 | Recursive Language Models | Prompt as a REPL variable; the model recursively calls itself on pieces | GPT-5 RLM beats compaction by a median 26% and Claude Code by 13%. RLM-Qwen3-8B +28.3% over Qwen3-8B, trained by rejection fine-tuning on 1,000 trajectories [abs][pdf] |
| 2512.02556 | DeepSeek-V3.2 | Thinking retained across tool turns, dropped at a new user message. Test-time Summary / Discard-75% / Discard-all at 80% of the window | Discard-all 67.6 BrowseComp; Summary extends to 364 steps [pdf] |
| 2510.24701 | Tongyi DeepResearch | Agentic mid-training plus post-training for long-horizon research, with per-stage environments | 30.5B total, 3.3B active; state of the art on HLE, BrowseComp and others at release [abs] |

### 1.2b 2026: compaction trained for coding agents, multi-session credit, and the frontier reports

**Research papers.**

| arXiv | name | mechanism | key numbers |
|---|---|---|---|
| 2610.02163 | AutoCompact | `compact()` as a policy tool. SFT warm start on judge-corrected compaction decisions, then fully async GRPO on SWE-Gym with a binary test reward. Split at prefix rewrites, one shared advantage, token-mean loss, no KL or entropy bonus | Qwen3-Coder-30B-A3B: SWE-bench Verified 30.4% → 39.6%, SWE-PolyBench-V +5.0 to 24.5% [pdf] |
| 2607.05378 | CompactionRL | Compaction forced when under 10,240 tokens remain (64K budget, at most 3 per rollout). Summary and work segments share the task reward. Token-level loss normalisation against summary/work length imbalance; cross-segment GAE (needs a critic). Async in slime | GLM-4.7-Flash +5.5 to 56.0% SWE-V (200-task subset), +6.7 to 20.2% TB2. Ablations: removing either component hurts [pdf][abs] |
| 2610.22174 | Mosaic | Compaction or subagent fork = new session. Trajectory advantage that counts sessions, plus a per-session score (gold-answer likelihood gain, QA only) | Qwen3.5-4B BrowseComp with compaction: broadcast 35.31, Mosaic 38.86, trained single-session then run multi-session 34.68 [pdf, helper] |
| 2609.00865 | MemoryWalker | Eviction turns the history into a tree. Training on the rightmost path leaks the future; depth-first replay mismatches inference. Exact LogitTree or 4D mask; cheap SDCC (KL to a teacher that saw the full prefix) | qualitative: naive training widens the train/rollout logprob gap [abs] |
| 2512.22733 | FoldAct | Separate losses for summary and action tokens, a full-context consistency loss, and selective segment training | 5.19× training speedup; names gradient dilution and self-conditioning collapse [abs] |
| 2609.37082 | Traverse | "Seal Memory" self-compression tool; RL caused "Seal Collapse", fixed by training only the final segment after context management | 72.83 BrowseComp (35B) [abs, helper] |
| 2609.35540 | Continuous Context Management | Compact every turn; GRPO plus privileged full-history distillation | beats full-history GRPO on WebShop at 4B, not 8B [abs, helper] |
| 2512.22087 | Context as a Tool (SWE-Compressor) | Compression as a callable tool for SWE agents, SFT only | 57.6% SWE-V [abs, helper] |
| 2610.04902 | SERA | RL for recursive agents: the parent writes a rubric per child; rubric writing trained against verified outcomes; leaf-coverage credit for decomposition | +5.38 TextCraft-Synth, +13.14 TextWorld [abs, helper] |
| 2511.21689 | ToolOrchestra | 8B orchestrator RL-trained to route to models and tools (outcome, efficiency, preference rewards) | HLE 37.1% vs GPT-5 35.1% [abs, helper] |
| 2610.31778 | Benchmarks Are Surprisingly Compaction-Friendly | Five compaction strategies in OpenCode with frontier models (evaluation only) | the first summary drops about 1/3 of ProgramBench's anti-cheating rules; instructions dropped about 6× more often than specs [abs, helper] |

**Frontier reports: where context is managed and how RL runs** (all [pdf] by a helper unless marked).

| arXiv | report | context management | RL infrastructure |
|---|---|---|---|
| 2512.02556 | DeepSeek-V3.2 | reasoning kept across tool rounds, dropped at a new user message; Summary / Discard-75% / Discard-all at evaluation only | off-policy negative-sequence masking, Keep Routing, Keep Sampling Mask [pdf, mine] |
| 2606.19348 | DeepSeek-V4 | keeps all reasoning in tool-calling conversations (1M window); Discard-all only for BrowseComp evaluation | preemptible rollout service; sandboxes with chained snapshots and a command log, so preempted tasks resume by replay |
| 2602.15763 | GLM-5 | interleaved and "preserved" thinking; keep-recent-k and Discard-all at evaluation (BrowseComp 55.3 → 62.0 with k = 5) | fully async, more than 1k concurrent rollouts; a token-in-token-out gateway called critical; double-sided token IS; stale samples dropped by version |
| 2602.02276 | Kimi K2.5 | Agent Swarm as "proactive context management"; only the orchestrator is trained (subagents frozen, for clean credit) | BrowseComp 60.6 → 74.9 with Discard-all, 78.4 with Swarm |
| 2607.24653 | Kimi K3 | **during RL, harness settings (tools, prompts, context strategies, memory, subagents) randomised per task group** | partial rollouts carried over across iterations, a per-token regulariser for staleness, microVM sandboxes with checkpoint and resume |
| 2605.26494 | MiniMax-M2 series | context management treated as environment. Each (state, action) is a sample with episode-level credit. White-box agents register their context operations so training rebuilds exact states; black-box agents are trained from the request stream | Forge RL system, CISPO |
| 2601.16725 | LongCat-Flash-Thinking-2601 | context management "necessary" for RL with tools: self-summary at 80K, Discard-all when turns run out, a 15K-sample summary cold start (about +3%) | DORA multi-version async, 2–4× vs sync |
| 2510.24701 | Tongyi DeepResearch | Markovian workspace rebuild from question, report and last step; its role inside RL not confirmed | step-level async on rLLM |
| web (HF model card) | Kimi K2-Thinking | hides all earlier tool outputs past 256K; 300-step limit | [web, helper] |

No Qwen3.5 or Qwen3.6 text-model technical report exists on arXiv (only Qwen3.5-Omni), so the student's own RL recipe is
unknown.

### 1.3 Credit across compaction boundaries, and the harness mismatch, for Crucible

- **What Crucible does matches the field.** It gives one outcome advantage per rollout, copied to every segment. The
  accepted summary is trained; rejected summary attempts are not. Group statistics stay on whole rollouts (harness audit
  §3.1–3.3).
- **Budget-limited rollouts.**
  - **What Crucible does differently.** It scores truncations (`max_steps`, `context_overflow`) on the final state.
    SUPO masks them.
  - **Why scoring is correct here.** The debug lane's tasks are graded on the repository's state, not on an answer
    string, so a truncated episode still has a measurable outcome (verifiers, Harbor and OpenEnv do the same).
  - **The SUPO lesson still applies:** if a run penalises long summarising rollouts, the summarisation pattern can
    collapse. The cue would be shrinking summaries, or compaction counts dropping to 0, in a run with compaction on.
- **Turn credit across segments.** Φ turn credit runs on the root chain's calls, and summary nodes take credit 0
  (harness audit §3.3). FoldGRPO's evidence is that process terms matter most for the context action itself: its
  unfolded-token penalty and its failed-call penalty.
  - **[inf]** Crucible's analogue would be a small negative term on main-thread tokens past a threshold. It is only worth
    adding if compaction or folding is ever trained.
- **The train/inference mismatch is the real risk today.**
  - The student is trained with no compaction and an interleaved template that re-renders at harness user turns.
  - Any eval or production harness that compacts, summarises or drops thinking differently is off-distribution.
  - Mosaic measured the cost directly: a model trained single-session and run with compaction scored below both trained
    variants. Kimi K3 goes further and randomises context strategies during RL so the policy generalises across
    harnesses.
  - **Open contradiction:** `runs/handoff-deltaai-inference.md` reports "2 compactions" in a debug episode, while
    training is documented as compaction-off (`00-crucible-map.md` §2.4 point 4). Resolve it before reading any
    before/after benchmark.
- **What a summary drops matters for honesty and hacking.** 2610.31778 found the first summary drops about a third of
  anti-cheating rules. Crucible's rebuild keeps the system prompt and the task verbatim (`actors/policy.py:253-270`), so
  the protected-path rules survive. Mid-episode harness notices do not (harness audit §3.4).

### 1.4 Recommendations

| recommendation | status | where |
|---|---|---|
| Keep compaction off in training while `context_overflow` stays near 0 (0 of 790 today); re-measure on the 300-call debug episodes (median 160K-token sequences) | **have** (decision) | `factory/envs.py:284-286`; harness audit R7 |
| Make the eval and benchmark harness match training: no compaction, the same template and nudge path; record the harness id with every benchmark number | **partial** (ids recorded; "2 compactions" contradiction open) | `runs/handoff-deltaai-inference.md`; `runtime/harness/*` |
| Before compaction is ever on in training: pass `ContextConfig` through `NativeEnv.policy_kwargs`, fit the summary call to the window, save `_compactions_in_row` (harness audit R7). Add a dashboard line for compactions per episode and summary length, to catch SUPO's collapse | **partial** | `actors/policy.py:330-387,421`; `factory/envs.py` |
| Prefer the cheapest strategy first if needed: DeepSeek's Discard-all or `keep_recent` keeps no model call on the boundary, so there is nothing new to train or hack | **have** (strategies exist) | `actors/policy.py` (`keep_recent`, `discard_all`) |
| If compaction is trained, follow AutoCompact, the closest public recipe (Qwen 30B-A3B coder, GRPO, test reward). Use compaction as a policy action (Crucible's `context_compact` action mode), an SFT warm start on judged compaction decisions, segments split at prefix rewrites, one shared advantage, token-mean loss | **partial** (action mode, segments, shared advantage and token-mean loss exist; no warm-start data) | `actors/policy.py` (mode `action`); `train/bridges/common.py`; `datums.py` |
| Train each segment on the context actually sent; never linearise across an eviction (MemoryWalker) | **have** | `train/world_env.py`; `runtime/record.py:140-195`; `bridges/common.flatten` |
| Harness randomisation during RL (Kimi K3): vary the tool limit, nudge wording, context strategy and limit warnings per group, so the student does not overfit one harness. Record the drawn configuration in the rollout | **missing** | `factory/envs.make_env`; `EpisodeLimits` |
| Watch for summary collapse (Traverse's "Seal Collapse", FoldAct's self-conditioning collapse): compactions per episode, summary length, and the share of episodes that compact | **missing** | step metrics |
| If the RLM is ever trained: fix C1 (kernel restart masking) and H1 (`compaction_failed` as a truncation) first, and consider distillation (RLM-Qwen3-8B route) before RL | **partial** | `actors/rlm/kernel.py`; `actors/rlm/agent.py`; harness audit §6 |
| Thinking retention: if `enable_thinking` is turned on, adopt DeepSeek-V3.2's rule (keep reasoning across tool turns, drop at a real user turn). Check that the interleaved template does this at nudges, which are user turns | **missing** (thinking off by default) | `TrainConfig.enable_thinking`; `train/world_env.py:175-181` |

---

## 2. Credit assignment for multi-turn agents

### 2.1 State of the art

- **Group-relative outcome advantages are the baseline everywhere**: GRPO, Dr.GRPO, RLOO. The question is how to give
  turn-level signal without a critic.
- **The critic-free, state-matched family.**
  - **GiGPO** builds step groups after the fact from identical environment states across a group's rollouts.
  - **RTMC** (2026) makes this practical for SWE agents with hand-made state signatures:
    - tool category;
    - file views bucketed by 100 lines;
    - a hash of the edit content.

    With 8 rollouts most states are seen only once, so smoothing toward the group success rate does much of the work
    [web].
  - **PGPO** (2026) turns the anchor-group return into a potential and uses potential differences, so good actions
    inside failed trajectories are separated. Plain GiGPO cannot do that.
- **Learned or judged step rewards.**
  - TCPO (2026) converts per-turn verifier scores into credit relative to the best earlier state, plus hindsight and
    counterfactual re-checks.
  - HCAPO (2026) has the policy re-score steps in hindsight.
  - TRACE (2026) uses a frozen reference model's log-probability of the gold answer as a value. That needs a gold answer.
- **Critics.**
  - Turn-PPO (2025) and SORL (2025) argue for turn-level, not token-level, critics or IS in multi-turn settings.
  - ArCHer and SWEET-RL are the older hierarchical and privileged-critic designs.
  - A helper found no turn-level critic in SkyRL itself (its paper and docs); the turn-critic papers are Turn-PPO and
    SORL.
  - ARLArena (2026) is a controlled testbed that factors policy gradient into four design axes. Its SAMPO recipe uses
    sequence-level clipping, turn-level advantages and dynamic filtering.
- **Tree rollouts.**
  - TreeRL branches at high-uncertainty tokens.
  - Tree-GRPO makes each node a whole agent step and shows that the within-tree objective equals step-level DPO.
  - ARPO and AEPO branch after tool results, where entropy jumps. AEPO penalises chains of consecutive high-entropy
    branches.
  - PATR (2026) uses a process scorer to choose branch points: up to +5.0 on SWE-Bench [abs].
  - A shared prefix is the cost saving.
- **Process-reward hacking is documented.**
  - Sum-form credit over a PRM collapses training from the start. Min-form credit (PURE) fixes it [abs].
  - 2603.06621 shows PRM reward above 0.9 while true accuracy stays below 4% on AIME [abs].
  - VeriGate (2026) uses PRM scores only in groups where the verifier gives every member the same reward.
- **Hindsight relabelling.**
  - ECHO, AgentHER and Hindsight SL relabel failed runs with goals they did reach. Mostly for SFT or prompting.
  - For RL on fixed hidden tests it is a data source, not a reward [inf].

### 2.2 Papers

| arXiv | name | mechanism | key numbers |
|---|---|---|---|
| 2505.10978 | GiGPO | Episode-level groups plus step-level anchor-state groups; no critic, no extra rollouts | > +12% ALFWorld, > +9% WebShop over GRPO; QA 42.1% (3B), 47.2% (7B) [abs] |
| 2604.11037 | RTMC | Rollout-tree Monte Carlo advantages over state signatures for SWE agents | +3.2 pp pass@1 over GRPO on SWE-bench Verified [abs]; 52.2% with Qwen3-Coder-30B-A3B, 49.7% without prior smoothing [web, sec] |
| 2609.02236 | PGPO | Potential from anchor-group returns; advantage = potential difference of adjacent states | strong ALFWorld/WebShop, negligible overhead, no figures in abstract [abs] |
| 2608.01667 | TCPO | Dense verifier scores turned into credit relative to the best earlier state, plus hindsight and counterfactual checks | best or tied best turn Pass@8 on Qwen3-4B, R1-Distill-8B [abs] |
| 2603.08754 | HCAPO | The policy re-scores step Q-values in hindsight, multi-scale advantage | +7.7% WebShop, +13.8% ALFWorld over GRPO [abs] |
| 2607.13988 | TRACE | Value = log-ratio of a frozen model's probability of the gold answer; TD reward per tool call | BrowseComp-Plus Qwen3-30B-A3B 8.4 → 42.6 with pure RL [abs] |
| 2505.11821 | Turn-level credit | Terminal, delayed and per-turn reward MDPs with GRPO and PPO variants for each | dense per-turn beats sparse; PPO with per-turn rewards most stable [abs] |
| 2512.17008 | Turn-PPO | PPO with a turn-level critic MDP | WebShop, Sokoban [abs] |
| 2511.20718 | SORL | Turn-level IS and clipping-triggered normalisation for off-policy multi-turn RL | qualitative stability gains [abs] |
| 2602.21534 | ARLArena / SAMPO | Testbed factoring policy gradient into loss aggregation, IS clipping, advantage design and filtering | SAMPO +25.2% average over GRPO [sec] |
| 2402.19446 | ArCHer | Hierarchical: off-policy utterance-level critic plus token-level policy | about 100× sample efficiency on agent tasks [abs] |
| 2503.15478 | SWEET-RL | Step critic with training-time privileged information | +6% absolute on ColBench [abs] |
| 2505.20732 | SPA-RL | Learned progress estimator redistributes the final reward to steps | +2.5% success on average [abs] |
| 2506.11902 | TreeRL | On-policy tree search, branch at high-uncertainty steps, no separate PRM | beats chain RL on math and code [abs] |
| 2509.21240 | Tree-GRPO | Node = agent step; intra- and inter-tree advantages; equivalent to step-level DPO | 11 QA datasets [abs] |
| 2507.19849 | ARPO | Entropy-triggered branching after tool calls, with advantage attribution | 13 benchmarks; half the tool-call budget [abs] |
| 2510.14545 | AEPO | Penalises consecutive high-entropy branching; stop-gradient in the high-entropy clip | Qwen3-14B, 1K samples: GAIA 47.6 [abs] |
| 2607.15610 | PATR | Process-scored branch points with prefix reuse and early stop | up to +5.0 SWE-Bench [abs] |
| 2504.15275 | PURE | Min-form credit over a PRM, which stops sum-form hacking | sum-form collapses at once; min-form matches verifiable reward in 30% of the steps [abs] |
| 2502.01456 | PRIME | Implicit process reward model updated online from outcome labels | +15.1% average over SFT [abs] |
| 2603.06621 | Reward Under Attack | PRMs ignore style changes but miss logic corruption; RL exploits it | PRM reward > 0.9 at < 4% true accuracy [abs] |
| 2605.30451 | VeriGate | PRM used only in zero-variance verifier groups | about +20% (1.5B), +12% (7B), less hacking [abs] |
| 2504.20073 | RAGEN / StarPO-S | Multi-turn "echo trap": reward variance cliffs and gradient spikes; trajectory filtering, critic, gradient stabilisation | qualitative [abs] |
| 2510.10304 | ECHO | Hindsight trajectory rewriting for LM agents (prompting) | up to +80% over vanilla agents [abs] |
| 2603.21357 | AgentHER | Hindsight relabelling of failed trajectories into demos for achieved goals | +7.6 to 11.4% over success-only SFT [abs] |

### 2.3 For Crucible

- **What Crucible has.**
  - Φ turn credit `A_t = A + η(ΔΦ_t − c)·σ_R/σ_ΔΦ` with η 0.1: centred within the episode, bounded by the outcome
    advantage's sign, off for hacked or gate-failed episodes (`train/credit.py:52,122`; `credit/compose.py`).
  - This is a potential-based design over *verified* milestones (hidden fail-to-pass tests per fix). It is safer than a
    learned PRM, since the potential comes from the grader, not a model.
  - It matches PGPO's idea with a grader-defined Φ instead of an estimated one.
- **What the field adds that Crucible lacks.**
  - **Anchor-state grouping over the World state.** Group members on one task often reach the same repository state:
    the same patch after the same edits. The checkpoint sidecar already snapshots the patch at each step.
    - A hash of the normalised patch (with `checkpoint.same_change` semantics) is a natural RTMC signature.
    - **[inf]** It gives step-level contrast at zero extra rollouts, including inside failed groups, where Φ is flat
      when no milestone moves.
    - RTMC's caveat applies: at group size 8, most states are unique, so pool with a prior toward the group mean.
  - **Branching from snapshots.** `credit/operators/branch_value.py` (built, unused) already restores a World snapshot
    and runs K continuations. Use it in two ways:
    - **To validate Φ.** Does ΔΦ track ΔV̂? This is the predictive-validity gate, `credit/validity.py`, which has no
      caller.
    - **Later, as TreeRL-style branching** at high-entropy steps after a tool result (ARPO). A 300-call episode is
      about 28 minutes, so a branch from call 150 halves the cost of each extra sample [inf].
- **What not to do.** Do not add a learned PRM or an LLM-judged turn score as reward. If one is added, follow VeriGate:
  only in zero-variance groups, and with min-form credit (PURE).

| recommendation | status | where |
|---|---|---|
| Keep Φ turn credit at small η, centred and sign-bounded | **have** | `credit/compose.py:107-177`; `train/credit.py` |
| Wire the predictive-validity gate for Φ weights, so Φ's milestone weights are fitted to outcome prediction | **partial** (`weights_from_validity` has no caller) | `credit/validity.py:133`; `factory/checkpoint.py:128` |
| Anchor-state step advantages from patch hashes (GiGPO/RTMC), logged first as a diagnostic (how often members share a state), then trained at small weight | **missing** | `factory/checkpoint.py` (snapshots); `credit/compose.py` |
| Use `branch_value` on a sampled subset to check ΔΦ against ΔV̂ before trusting Φ at larger η | **partial** (built, unused) | `credit/operators/branch_value.py` |
| Any learned step score: only in zero-variance groups (VeriGate), min-form (PURE), never the main reward | **have** as policy (`credit/operators/prm.py` says baseline only); keep | `credit/operators/prm.py` |
| A turn-level critic (Turn-PPO): not now; the GAE critic exists for n = 1 enterprise cases | **partial** | `train/critic.py` |

---

## 3. Stabilising agentic and MoE RL

### 3.1 Training–inference mismatch

- **Where it comes from.**
  - Different kernels and reduction orders between the sampler and the trainer.
  - BF16 rounding.
  - MoE top-k routing flips.
  - Sampling truncation (top-p and top-k).
  - FP8 or INT rollouts.

  The bias of token-level IS grows like O(T²·D_TV) with length ("When Speed Kills Stability" [web]; Trust Region
  Masking, 2512.23075, which derives O(T) bounds [abs]). That makes it a long-horizon problem in particular.
- **The corrections, in roughly the order they appeared.**
  1. **TIS.** A one-sided cap on the train/sample ratio (Yao et al. blog, Aug 2025; verl's default cap is 2 [web]).
  2. **MIS and IcePop.** Drop tokens whose ratio leaves a band. Ring-1T's IcePop uses [0.5, 5]: about 1–2‰ of tokens
     masked, rising over training. A narrow [0.5, 2] band destabilised [pdf, helper].
  3. **Sequence-level and geometric masks.** "When Speed Kills Stability" Geo-RS. DeepSeek-V3.2 masks only
     negative-advantage sequences whose mean log-ratio exceeds δ, because learning from one's own mistakes helps most
     and very off-policy negatives mislead [pdf].
  4. **Routing replay (R3).**
     - With R3, Qwen3-30B-A3B's train/infer KL falls from 1.5e-3 to 7.5e-4, near the dense 8B's 6.4e-4 [pdf].
     - Rollout overhead is under 3% [pdf].
     - DeepSeek calls Keep Routing "crucial" for MoE RL since V3-0324 [pdf].
  5. **Score centering** (2609.20807, Together AI). An additive correction that cancels the persistent drift term. It
     composes with IS: SC + MIS is best under heavy staleness and quantisation [pdf].
  6. **Precision fixes.**
     - FP32 LM head: in MiniMax-M1 it raised the train/infer probability correlation to about 0.99 [pdf, helper]; in
       ScaleRL it raised the asymptote 0.52 → 0.61 [pdf].
     - FP16 everywhere: about 24× smaller mismatch [pdf].
     - Batch-invariant or bitwise-consistent kernels: Thinking Machines (Sep 2025) [web]; vLLM and NVIDIA for
       DeepSeek-V4-Flash and a 30B-A3B hybrid on 2026-10-14 [web].
- **Counter-evidence worth keeping in mind.**
  - 2602.01826 finds that mismatch grows with gradient noise. FP16 still collapsed in its setup; MIS did not prevent
    collapse there, while token TIS did. Lowering the LR when response length surges fixed it [pdf, helper].
  - The vLLM post shows a baseline whose KL *fell* while entropy rose 0.05 → 2.6 and SWE-bench Pro fell 0.56 → 0.22
    [web]. Low KL alone is not health.

### 3.2 Objectives, entropy, length and KL

- **Objectives.**
  - **GSPO.** Sequence ratio and sequence clip. It notes about 10% of activated experts change per update on
    Qwen3-30B-A3B, and that GRPO needs routing replay to converge where GSPO does not [pdf, helper].
  - **CISPO.** A clipped, stop-gradient IS weight that keeps every token's gradient (MiniMax-M1). It is ScaleRL's
    choice for robustness.
  - **GMPO.** Geometric mean of token rewards.
  - **Qwen's "Formulation and Practices" (2512.01374).** The token surrogate is valid only when both mismatch and
    staleness are small. On Qwen3-30B-A3B, on-policy, R3 gave no gain. Off-policy: at 2× R2 beat R3; at 4× and 8× R3 won
    and R2 collapsed [pdf, helper].
- **Entropy.**
  - Entropy collapse is predictable: R ≈ −a·e^H + b. The entropy change follows the covariance of log-prob and advantage
    [abs]. 73% of the entropy drop and 76% of the gain happen in the first 200 steps [pdf, helper].
  - Clip-Cov and KL-Cov act on the few high-covariance tokens.
  - An entropy-loss coefficient of 0.01 made entropy explode [pdf, helper].
  - Clip-higher (DAPO ε_high 0.28) is the cheap standard [pdf].
  - Only about 20% high-entropy "forking" tokens carry the RLVR gain (2506.01939) [abs].
- **Length and turn blow-up.**
  - Dr.GRPO removed GRPO's length bias (std and length normalisation) [abs].
  - DAPO uses token-level loss and a soft overlong penalty.
  - **ScaleRL: training instabilities were "often linked to truncations"; it calls a high truncation rate a reliable
    warning signal** [pdf].
  - 2602.01826: a response-length surge led collapse by roughly 190 steps in its runs [pdf, helper; lead time inferred].
- **KL choices.**
  - Differentiating through a KL estimate as a loss gives the wrong gradient (2506.09477) [abs].
  - "k3 as loss", as in GRPO, is a biased approximation (2510.01555) [abs].
  - 2512.21852: only k1 in the reward gives an unbiased gradient, and some KL stabilises async training [abs][pdf,
    helper].
  - DeepSeek-V3.2 uses an IS-corrected k3 and drops or weakens KL for some domains [pdf].
  - ProRL uses KL with reference resets for long runs [abs].

### 3.3 LoRA RL and precision

- **"LoRA Without Regret"** (Thinking Machines, 2025-09-29) [web]:
  - RL absorbs about O(1) bits per episode, so low rank suffices; rank 1 matched full fine-tuning.
  - The best LoRA LR is about 10× the full fine-tuning LR (fit 9.8×; about 15× for runs under ~100 steps), roughly
    independent of rank.
  - Apply LoRA to MLP and MoE layers; attention-only underperforms.
  - For MoE, one LoRA per expert at rank = total rank ÷ active experts.
  - α = 32 with 1/r scaling.
  - LoRA tolerates large batches less well than full fine-tuning.
- **Dissent.** 2512.23165 finds rank 1 and VeRA "severely bottleneck" reasoning RL, and DoRA, AdaLoRA and MiSS beat LoRA
  [abs]. 2601.06677 has r = 8 failing where r = 256 worked on ≤1.5B models [abs]. Rank 32 sits between the two camps.
- **BF16 weights swallow small updates.**
  - **FP16 paper, checked by me.** BF16 LoRA (r 32, α 64, lr 4e-5, "following Schulman and Lab") collapsed after about
    600 steps, while FP16 stayed stable [pdf].
    - That experiment was **Qwen2.5-Math-1.5B** with GRPO-Token-TIS.
    - Its MoE runs (Qwen3-30B-A3B, full fine-tuning) are separate.
    - A helper's summary merged the two; this is corrected here.
  - **TRL write-up (2026-04-04, hf.space mirror)** [web]:
    - BF16 weight storage rounds small updates away;
    - with PPO clipping this silences unchanged tokens ("phantom clipping");
    - FP32 master weights with BF16 autocast fixed it.
- **Arithmetic for Crucible [inf].**
  - BF16 keeps 8 significant bits, so an update smaller than about |w|·2⁻⁹ to |w|·2⁻⁸ rounds back to w.
  - Adam's per-element step is at most about the LR, 4e-5 here.
  - So a weight of magnitude above about 0.01–0.02 cannot move by one step.
  - LoRA's A matrix is initialised at a scale of about 1/√d_in. That is about 0.02 for d_in ≈ 2048; the student's
    hidden size was not checked.
  - **So in bf16, many A entries may never change, while B (zero-initialised) trains.** The adapter would degrade toward
    a fixed random projection.
  - Whether prime-rl keeps Adam moments in fp32 when `optimization_dtype = bfloat16` was not checked; it should be.
- **Crucible's numbers.**
  - Tinker: rank 32, LR 4e-5 (`TrainConfig`).
  - prime-rl: rank 32, α 64, AdamW 4e-5, clip 1.0 (`deploy.py:66-70,144`).
  - The 35B probe forced `optimization_dtype = bfloat16` on 2 trainer GPUs because fp32 also holds the frozen base
    (`14-deltaai-trainer.md` §8.2, which already flags the rounding risk).
  - Episodes are about 32 × 160K tokens ≈ 5M trained tokens per step. That is a large LoRA batch by the Thinking
    Machines result [inf].

### 3.4 Papers

| arXiv / source | name | mechanism | key numbers |
|---|---|---|---|
| blog (fengyao.notion.site, 2025-08) | TIS | One-sided cap on the train/sample ratio | adopted in verl (cap 2) and slime [web, sec] |
| blog (yingru.notion.site, 2025-09-17) | When Speed Kills Stability | Sequence TIS, sequence MIS, geometric-mean rejection; O(T²) bias of token IS | collapse seen as a grad-norm explosion [web, partial render] |
| 2510.18855 | Ring-1T / IcePop | Token masking outside [α, β]; C3PO++ budgeted rollouts | [0.5, 5]; 1–2‰ masked; baseline collapse near step 180–200 [pdf, helper] |
| 2510.11370 | R3 | Record inference routes, replay in training | KL 1.5e-3 → 7.5e-4; overhead < 3%; prevents collapse; beats GSPO and TIS [pdf][abs] |
| 2609.20807 | Score centering | Additive drift-cancelling correction; composes with IS | SC matches or beats IS methods 0.6B–30B; SC + MIS best under staleness [pdf] |
| 2510.26788 | FP16 | FP16 in both engines | about 24× smaller mismatch; BF16 LoRA collapse at about 600 steps (1.5B) [pdf] |
| 2602.01826 | Beyond Precision | Mismatch rises with gradient noise; length-triggered LR decay | FP16 still collapsed in their setup [abs][pdf, helper] |
| 2512.01374 | Stabilizing RL with LLMs (Qwen) | Token surrogate validity; MiniRL; R2 vs R3 by off-policy degree | R3 needed at 4–8× off-policy [pdf, helper] |
| 2512.23075 | Trust Region Masking | Mask whole sequences whose max token KL breaks the region; bounds O(T²) → O(T) | theory; in verl presets [abs] |
| 2507.18071 | GSPO | Sequence ratio and clip | stabilises MoE RL; used for Qwen3 [abs] |
| 2507.20673 | GMPO | Geometric mean of token rewards | up to +4.1% Pass@1 at 7B [abs] |
| 2506.13585 | MiniMax-M1 / CISPO | Clipped IS weight with stop-gradient; FP32 head | 512 H800 for 3 weeks, $534,700 [abs] |
| 2510.13786 | ScaleRL | PipelineRL-8, CISPO, FP32 logits, prompt-level aggregation, zero-variance filter, no-positive-resampling | 400K GPU-hours; truncations as the warning signal [abs][pdf] |
| 2512.02556 | DeepSeek-V3.2 | Unbiased k3, off-policy negative-sequence masking, Keep Routing, Keep Sampling Mask | [pdf] |
| 2503.14476 | DAPO | Clip-higher, dynamic sampling, token-level loss, overlong shaping | 50 on AIME24 with Qwen2.5-32B [abs] |
| 2503.20783 | Dr.GRPO | Removes GRPO's length and std biases | 43.3% AIME24 at 7B [abs] |
| 2505.22617 | Entropy mechanism | R = −a·e^H + b; Clip-Cov, KL-Cov | [abs][pdf, helper] |
| 2506.01939 | High-entropy minority tokens | Train only the ~20% forking tokens | +11.04 AIME25 on Qwen3-32B [abs] |
| 2505.24864 | ProRL | KL control plus reference resets for prolonged RL | [abs] |
| 2506.09477 | KL pitfalls | Differentiating through a KL estimate is wrong | [abs] |
| 2512.21852 | A Comedy of Estimators | Which KL configurations give unbiased gradients; KL helps async | [abs] |
| blog (thinkingmachines.ai/blog/lora) | LoRA Without Regret | Rank, LR, layers and batch size for LoRA RL | rank 1 suffices for RL; LR about 10× [web] |
| 2512.23165 | PEFT for RLVR | 12+ PEFT methods compared | rank 1 and VeRA bottleneck reasoning [abs] |
| blog (vllm.ai, 2026-10-14) | Bitwise-consistent RL | Trainer calls vLLM kernels; zero diff | 91.3% rollout speed; untuned baselines [web] |
| blog (thinkingmachines.ai, 2025-09-10) | Defeating Nondeterminism | Batch-invariant kernels | sampler–trainer KL 0 [web, helper] |

### 3.5 Recommendations

| recommendation | status | where |
|---|---|---|
| **LoRA weights fp32, base bf16.** Patch the prime-rl model dtype so adapter params (and Adam moments) are fp32 while the frozen base stays bf16; the adapter is tiny. Failing that, give the trainer more GPUs and keep `float32`. Either way, log the share of LoRA elements that changed per step and the norm of A and B separately at the 2-step run | **missing** | `primerl/deploy.py:63-65,137`; `14-deltaai-trainer.md` §8.2 |
| Keep SC + MIS [0.5, 5] as the default correction; the band equals IcePop's and the paper's | **have** | `primerl/loss.py:78-83` |
| Keep neutral sampling (top-p 1, top-k 0) so no Keep-Sampling-Mask is needed | **have** | `primerl/backend.py:63-68` (B1 fix) |
| Keep routing replay; refuse version-mixed sequences (R5) and watch the refused share | **have** | `primerl/routing.py`; async audit R5 |
| Add a sequence-level guard on the prime-rl path for long sequences: DeepSeek-style negative-advantage masking by mean log-ratio, or max-token-KL masking. The torch loop's `mispo` has the geomean trajectory mask; port its rule | **partial** | `train/mismatch.py:208-222`; `primerl/loss.py` |
| FP32 LM head in both engines, if prime-rl and vLLM allow it for `qwen3_5_moe` | **missing** (not checked) | prime-rl model config; `primerl/deploy.py` |
| FP16 instead of BF16: not now. It needs loss scaling with FSDP, and 2602.01826 saw FP16 collapse too | decision | |
| Entropy: log it first (the student path does not). Add Clip-Cov or KL-Cov, or the existing mimo4 entropy-adaptive band, only if entropy falls fast in the first ~50 steps | **partial** (torch only) | `train/trainer.py:116-152,468-492` |
| No reference KL (β 0) is fine for a 40-step LoRA run. For runs of hundreds of steps, or at staleness ≥ 2, add k1-in-reward or DeepSeek's IS-corrected k3, never k3-as-loss | **have** (β 0); KL form **missing** | `primerl/deploy.py:68` (`kl_coeff`) |
| LoRA rank 32 on attention, MLP and experts with LR 4e-5 is consistent with LoRA Without Regret. Two checks: α 64 doubles the α/r factor against that post's α = 32 recipe, and Tinker's α was not checked | **have**; α parity **uncertain** | `primerl/deploy.py:66-67`; Tinker backend |
| An LR-decay trigger on response-length surge (2602.01826) as a safety valve for long runs | **missing** | `TrainConfig` |

---

## 3b. Asynchronous RL systems

### 3b.1 State of the art

- **Two designs dominate.**
  1. **Disaggregated, fully async with a staleness bound** (AReaL, slime, verl fully-async, ROLL Flash, NeMo-RL async,
     Laminar, DORA).
  2. **In-flight weight updates that keep the generation going** (PipelineRL, prime-rl). Prime-rl continues on the
     stale KV cache across a swap.
- **Partial rollouts.** APRIL over-provisions and carries unfinished rollouts into the next step. AReaL and verl
  interrupt and resume on new weights.
- **Staleness is an algorithm problem first.**
  - AReaL's decoupled PPO (a proximal policy recomputed per step) kept near-oracle accuracy up to η = 8. Without it, η = 4
    dropped MATH500 from 89.2 to 66.9 [pdf, helper].
  - M2PO bounds the second moment of IS weights and trains with data 256 updates stale [abs].
  - 2026 adds staleness-adaptive trust regions:
    - SAT shrinks the clip on the high-mismatch tail;
    - ENTR scales the region by token entropy (stable to 30 versions);
    - COPC also corrects "advantage staleness" (stable at 64 steps);
    - VESPO uses a sequence-level kernel (64×).
- **Measured speedups** (all at scales far above Crucible's):

  | system | speedup |
  |---|---|
  | AReaL | up to 2.77× |
  | PipelineRL | about 2× faster learning on 128 H100 |
  | ROLL Flash | 2.24× RLVR, 2.72× agentic |
  | verl fully async | 1.72–2.01× on Qwen3-30B-A3B at a 96:32 split [web, helper] |
  | Laminar | up to 5.48× on 1024 GPUs |
  | SkyRL-Agent's dispatcher | 1.55× over naive async batching |

- **Defaults people actually use:**

  | system | staleness |
  |---|---|
  | NeMo-RL | age 1 |
  | ROLL | α = 2 (good range 1–4) |
  | AReaL | η 4 (code) to 8 (math) |
  | prime-rl / INTELLECT-3 | `max_off_policy_steps` 8 with IcePop [0.5, 5] |
  | ScaleRL | PipelineRL k = 8 |

  Crucible's own matrix is in the async audit §3.1.

### 3b.2 Papers and systems

| id | name | mechanism | key numbers |
|---|---|---|---|
| 2505.24298 | AReaL | Fully decoupled; staleness bound η; decoupled PPO; interruptible rollouts | up to 2.77× [abs]; η = 4 without decoupling: MATH500 66.9 vs 89.2 [pdf, helper] |
| 2512.06547 | A-3PO | Interpolated proximal policy, no extra forward pass | 1.8× speedup; in AReaL [abs] |
| 2509.19128 | PipelineRL | In-flight weight updates with the stale KV cache kept | about 2× faster learning, 128 H100 [abs] |
| 2509.18521 | APRIL | Over-provision; recycle partial rollouts | +22.5% throughput (max 44%), +2.1% accuracy [abs] |
| 2505.07291 | INTELLECT-2 (prime-rl) | Decentralised async GRPO; TOPLOC; SHARDCAST | two-step async [abs][pdf, helper] |
| 2512.16144 | INTELLECT-3 (prime-rl) | Continuous batching with in-flight updates; agentic multi-turn | 106B-A12B, up to 512 H200 [abs]; off-policy 8, IcePop [0.5, 5] [pdf, helper] |
| 2505.24034 | LlamaRL | Single-controller PyTorch, async off-policy | up to 10.7× at 405B [abs] |
| 2506.06122 / 2510.11345 | ROLL / ROLL Flash | Rollout–train decoupling, asynchrony ratio | 2.24× RLVR, 2.72× agentic [abs] |
| 2409.19256 | HybridFlow (verl) | Hybrid controller; 3D-HybridEngine | 1.53–20.57× throughput [abs] |
| 2510.12633 | Laminar | Relay workers, trajectory-level async, long-tail repacking | up to 5.48× [abs] |
| 2604.26256 | DORA (LongCat) | Multi-version streaming rollout, bounded staleness | 2–3× vs SOTA [abs] |
| 2504.15930 | StreamRL | Disaggregated stream generation; length ranker | up to 2.66× [abs] |
| 2511.14617 | Seer | Synchronous: divided rollout, context-aware scheduling, grouped speculative decoding | 2.04× rollout throughput; tail latency −72–94% [abs] |
| 2511.16108 | SkyRL-Agent | Async pipeline dispatcher; Tinker, verl and SkyRL-train backends | 1.55×; SA-SWE-32B 39.4% SWE-bench Verified [abs] |
| 2510.01161 | M2PO | Second-moment trust region on IS weights | stable at 256 stale updates; clipped tokens 1.22% → 0.06% [abs] |
| 2607.18722 | SAT (Stale but Stable) | Staleness-adaptive trust region on the high-mismatch tail | Qwen3-30B-A3B with R3 [abs] |
| 2607.22186 | ENTR | Entropy-normalised trust region | stable to 30 versions; 2.6× [abs] |
| 2610.09597 | COPC | Couples the policy-ratio and advantage-staleness corrections | stable at 64-step staleness; 1.7× [abs] |
| 2602.10693 | VESPO | Variational sequence-level reshaping | stable up to 64× staleness [abs] |
| docs | verl fully async, NeMo-RL async, AReaL-lite, slime | Bounds and knobs (async audit §3) | verl Qwen3-30B-A3B 1.72–2.01× [web, helper] |

### 3b.3 For Crucible: concrete defaults for a 35B MoE on a few GPUs

No source tests a 35B MoE on 4 GPUs, so these are inferences from the defaults above plus Crucible's measurements.
- **Measurements.** About 1 h per 4 × 8 step, of which training is about 0.2 h. Episodes run about 28 min. Calls switch
  version only between calls.

| knob | default for the first async run | why |
|---|---|---|
| `max_staleness` | **1**, with `async_groups` 4 (async audit R1) | ROLL α 1–2, NeMo age 1; token MIS suffices at 1 |
| raise to 2 | only after `is/masked_frac/s1` and `routing/version_mixed_share` stay small for ~10 steps | Qwen 2512.01374: R3 matters from about 4× off-policy |
| beyond 2 | only with a decoupled or second-moment objective (AReaL decoupled PPO, M2PO), or a 2026 adaptive trust region | AReaL η = 4 collapse without decoupling |
| in-flight calls at a swap | keep call-level switching (never mix versions in a call) | unique in the async audit matrix; makes R5's refusal rare |
| partial rollouts | keep finishing calls on their own adapter; do not add APRIL-style carry-over across steps until staleness 1 is proven | an episode spans many versions already |
| prefix cache | keep per-adapter salts | async audit §1.4 |
| LoRA slots | `max_loras = 2` at staleness 1 | TRL's rule `max_loras ≥ staleness + 2` is for in-call switching; Crucible needs one older version |
| group size | 8 | GRPO variance; RTMC notes state pooling needs more |
| trust region | SC + MIS [0.5, 5], no PPO clip | have |

| recommendation | status | where |
|---|---|---|
| Run the overlap (staleness 1, `async_groups` 4) after the 2-step real-model run shows `kl/mismatch`, `sc/*`, `is/masked_frac` at staleness 0 | **partial** (open by design, async audit R1) | `scripts/delta_rl.py`; `bridges/tinker/trainer.py:156-157` |
| Per-staleness KL and masked share on the prime-rl path need a per-sequence channel into `compute_loss` | **partial** (token shares only) | async audit R2 |
| AReaL's decoupled proximal policy, if the masked share grows with staleness | **missing** | async audit R2 |
| Do not adopt APRIL carry-over or in-place LoRA reload: they mix versions inside a call | decision | |

---

## 4. Reward design for agents beyond pass/fail

### 4.1 State of the art and the risks of each kind

- **Milestones and partial credit.**
  - ADMIRE (2026) rewards successful runs only at distilled milestones, and gives failed runs per-milestone credit.
  - BEACON (2026) segments at milestones so a late failure does not punish earlier correct work: ALFWorld long-horizon
    92.9% vs GRPO 53.5% [abs].
  - MiRA (2026) brings WebArena-Lite from 6.4% to 43.0% [abs].
  - **Risk:** milestones defined by a model are "prone to bias and reward hacking" (ADMIRE's motivation). Milestones
    defined by hidden tests, as in Crucible, avoid most of it.
- **Efficiency and cost.**
  - **OTC-PO** cuts tool calls by up to 68.3% at similar accuracy, efficiency scored only for correct answers [abs].
  - **Risks, documented in 2026:**
    - a mild flat penalty is washed out by accuracy variance in group normalisation;
    - a strong one suppresses tool use the task needs (HDPO/Metis 2604.08545);
    - coarse penalties teach blanket tool suppression (AKBE 2605.26952);
    - a round-count penalty made a privilege-escalation agent drop the recovery loops it needed (2603.17673) [sec].
  - **Fix:** gate on correctness and measure relative to the cohort of correct runs (SlimSearcher, HDPO, DRPO and DDCA
    for length).
- **Length penalties.** They can cause a sudden, early accuracy collapse (GRPO-λ 2505.18086). Correct-but-long rollouts
  get a negative advantage unless length is normalised within the correct ones (DRPO 2510.04474) [abs].
- **Honesty and abstention.**
  - TruthRL's three-way reward (+1, 0, −1) halves hallucination (43.5% → 19.4%) [abs].
  - 2601.20126 sweeps the abstain reward: about −0.25 to 0.3 works; higher values cause over-abstention [abs].
  - "Why Language Models Hallucinate" argues binary grading rewards guessing [abs].
  - Agentic abstention (2606.28733): agents stop too late or never [abs].
  - **ImpossibleBench** (2510.20270) builds impossible variants, so any pass is cheating. Offering an explicit "flag as
    impossible" option cut GPT-5's cheating from 54% to 9% [web, helper].
  - Confessions (2512.08093) train a separate honesty channel [abs].
- **Format penalties.** FoldGRPO penalises failed tool-call turns. NeMo-RL has an optional negative advantage. Mostly
  harmless; the risk is net negative mass that lengthens or flattens outputs, which Crucible's `signed` mode avoids.
- **Several objectives.** GDPO shows that GRPO on a summed reward maps different reward combinations to identical
  advantages. Normalise each reward within the group, then sum [abs].

### 4.2 Papers

| arXiv | name | mechanism | key numbers |
|---|---|---|---|
| 2602.11524 | ADMIRE | Milestones distilled from successes; asymmetric credit | > +10% AndroidWorld [abs] |
| 2605.06078 | BEACON | Milestone segmentation, dual-scale advantages | ALFWorld long-horizon 92.9% vs 53.5% [abs] |
| 2603.19685 | MiRA | Dense milestone rewards for web agents | 6.4% → 43.0% WebArena-Lite [abs] |
| 2504.14870 | OTC-PO | Tool-use reward tied to correctness | up to −68.3% tool calls [abs] |
| 2606.07074 | SlimSearcher | Correctness gate, then cohort-relative efficiency | −17 to 58% tool rounds [abs] |
| 2604.08545 | HDPO / Metis | Separate efficiency channel inside correct trajectories | [abs] |
| 2605.26952 | AKBE | With/without-tool paired rollouts find when a tool is needed | −18% calls, +1.85 accuracy [abs] |
| 2504.13958 | ToolRL | Systematic study of tool-use reward types, scales, granularity | +17% over base [abs] |
| 2505.18086 | GRPO-λ | Fall back to 0/1 reward when a group's correctness is low | −47.3% length [abs] |
| 2510.04474 | DRPO | Length signal normalised within correct rollouts | 77% shorter, −1.1% GSM8K [abs] |
| 2602.02099 | DDCA | Length advantage among correct only; penalty scaled by pass rate | [abs] |
| 2503.04697 | L1 / LCPO | Length control in the prompt | [abs] |
| 2501.12599 | Kimi k1.5 | Length penalty, long2short | [abs] |
| 2601.05242 | GDPO | Per-reward group normalisation before summing | beats GRPO on tool calling, math, code [abs] |
| 2507.17746 | Rubrics as Rewards | Rubric items aggregated into a reward | up to +31% HealthBench [abs] |
| 2509.25760 | TruthRL | Ternary reward | hallucination 43.5% → 19.4% [abs] |
| 2601.20126 | Rewarding Intellectual Humility | Abstain-reward sweep | r_abs −0.25 to 0.3 [abs] |
| 2509.04664 | Why LMs Hallucinate | Binary grading rewards guessing | [abs] |
| 2606.28733 | Agentic Abstention | When agents should stop on impossible tasks (benchmark) | timely recall 26.7 → 57.4 with CONVOLVE [abs] |
| 2510.20270 | ImpossibleBench | Spec-contradicting tests; any pass is cheating | flag option: GPT-5 54% → 9% [web, helper] |
| 2512.08093 | Confessions | Separate honesty-only channel after the answer | [abs] |

### 4.3 For Crucible

- **Have.** The reward is `phi_weight·Φ + (1 − phi_weight)·[full pass and no path violation]`. A hack-severity
  violation gives 0; a gate-severity one removes the bonus. Honesty is a Brier term ≤ 0, added to the advantage, not to
  R, so it keeps the outcome's sign. `GIVE_UP` exists. The format penalty is off on every backend (harness audit §4.1,
  R5).
- **This already follows two 2026 lessons.**
  - Honesty is composed as a separate channel, not summed into R before centring. That is the GDPO and Confessions
    lesson, and `factory/honesty.py` explains why: an honest failure would otherwise gain a positive advantage in an
    all-fail group.
  - The milestones are executable, not judged.
- **Gaps.**
  1. **No infeasible variants.** The honesty term can only train calibration on tasks that are hard, not on tasks that
     cannot be done. An ImpossibleBench-style variant has two uses:
     - **For RL:** contradicting hidden and visible tests, or a bug report whose requested behaviour conflicts with
       existing tests, trains the honest "not met" claim.
     - **For measurement:** it measures cheating.

     Keep the reward for a correct "infeasible" small (TruthRL: 0 vs −1; 2601.20126: about −0.25 to 0.3).
  2. **Over-abstention is not monitored.** Track `GIVE_UP` and low-claim rates on tasks with p̂ > 0.3. A rise means the
     honesty term is teaching quitting.
  3. **Efficiency.**
     - 148 of 790 targeting rollouts hit the 150-call limit (harness audit §0). There is pressure to want a cost term.
     - If one is added, it must be HDPO-style: only within successful members, relative to the group.
     - A flat per-call penalty would teach the agent to stop investigating, the AKBE and PrivEsc failure.
     - The truncation share is a dashboard signal first (§6).
  4. **Several rewards later** (Φ, pass, honesty, format): if any second scalar enters R itself, normalise each within
     the group before summing (GDPO).

| recommendation | status | where |
|---|---|---|
| Executable milestones as partial credit, with full-pass bonus and hack zeroing | **have** | `factory/lanes/windows.py:1704-1753`; `factory/pathcheck.py` |
| Honesty as a separate ≤ 0 advantage channel; flat groups kept when composed advantages differ | **have** | `factory/honesty.py`; `TinkerTrainer._has_signal` |
| Infeasible task variants in the pool (train "report infeasible", measure cheating) | **missing** | task factory (`factory/generators/debug.py`) |
| Over-abstention monitor: give-up and low-claim rates on solvable tasks | **missing** | `train/bridges/tinker/trainer.py` step metrics |
| Efficiency term only success-gated and group-relative; none today | **missing** (correct default) | `train/bridges/tinker/datums.py` |
| Per-reward group normalisation if a second scalar enters R | **missing** (not needed yet) | `credit/compose.py` |
| Format penalty stays off; if on, use `signed` | **have** | `TrainConfig.tool_format_penalty`, `format_penalty_mode` |

---

## 5. Alignment and safety during agentic RL

### 5.1 State of the art

- **Reward hacking generalises.** Anthropic (2511.18397) taught a pretrained model hacking strategies, then ran RL on
  real production coding environments.
  - **The hacks.** The model learned three:
    1. returning an object whose `__eq__` always returns True;
    2. `sys.exit(0)` before the asserts;
    3. a `conftest.py` that patches pytest's report.
  - **The generalisation.** The model went on to alignment faking, cooperation with malicious actors, and sabotage. In
    Claude Code, on the paper's own codebase, it attempted sabotage 12% of the time, and its sabotaged classifiers were
    65% as effective [pdf].
  - **What fixed it only partly.** Chat-style RLHF fixed chat evals, while agentic misalignment persisted (up to 70% of
    pre-RLHF misalignment remained on some evals) [pdf].
  - **What worked.**
    - Preventing the hack (preference-model or hack-classifier penalties).
    - Broader safety RLHF.
    - **Inoculation prompting**: one system-prompt line framing hacking as acceptable cut misalignment by 75–90%,
      despite hack rates above 99% [pdf]. Anthropic says it has started using it in production training [pdf].
  - **What did not work.** Filtering hack episodes and distilling the rest. Telling the model not to hack can increase
    misalignment [pdf].
- **Other generalisation evidence.**
  - Emergent misalignment from narrow fine-tuning (2502.17424) [abs].
  - School of Reward Hacks (2508.17511): SFT on harmless hacks generalises in GPT-4.1 [abs].
  - Iterative DPO reproduces it cheaply (2609.06649) [abs].
  - Documents that only *describe* hacking raise hacking (Anthropic blog: XL 4.7% → 9.1%) [web, helper].
- **Inoculation papers.**
  - 2510.04340: a less surprising trait means less pressure to update globally.
  - 2510.05024: prompts that elicit the behaviour more strongly before training inoculate better [abs].
- **Monitors and obfuscation.**
  - OpenAI (2503.11926): a weaker model monitoring the CoT catches far more than an action-only monitor. Putting the CoT
    monitor in the reward helps at low optimisation, then produces obfuscated hacking [abs].
  - Even *output-only* supervision can obfuscate the CoT (2511.11584), and the obfuscation transfers to new tasks
    (2601.23086) [abs].
  - "Chain of Thought Monitorability" (2507.11473) asks developers to weigh training decisions against monitorability
    [abs].
  - 2026 alternatives to penalties:
    - **GRAFT** (2610.17062): flagged rollouts update a throwaway adapter that is deleted later, so the monitor never
      enters the reward [abs]. With a standard penalty, an unmonitored hack appeared in about 30% of runs [web, helper].
    - **Screening vs penalty** (Wong, Engels, Nanda, ICML 2026, OpenReview only): penalties suppress more; screening
      hurts learning less [web, helper].
    - **Train monitors on hacks that arise in training**, not prompted ones (Trace-and-Amplify 2604.23488) [abs].
    - **A precursor probe** forecasts hack onset (PRIME 2606.09711) [abs].
  - **RHB** (2605.02964): exploit rates from 0% to 13.9% across 13 models; 72% of hacks came with an explicit CoT
    rationale [abs].
- **Sycophancy.** RLHF amplifies agreement when it covaries with reward (2602.01002) [abs]. No 2026 paper on
  sycophancy drift under *agentic* RL was found.

### 5.2 For Crucible

- **The debug lane's grader is pytest-shaped**, so Anthropic's three named hacks are directly relevant.
  - **Covered or partly covered.**
    - `conftest.py` and test plumbing are hack-severity path rules (`factory/pathcheck.py`, harness audit R6).
    - Protected tests are restored before the hidden run.
    - `EXIT_SPOOF` (`os._exit(0)`, `sys.exit(0)` near "test", `pytest.exit`) is in `credit/hack_detector.py:32`.
    - The map (§2.4) found that `WorkspaceHarness` never calls `rule_flags`, so L1 rules do not run on the student path.
  - **Not covered: the AlwaysEqual hack.** A patch that makes a returned object compare equal to anything passes
    equality asserts.
    - Path rules cannot see it, because it lives in source code, not in test files.
    - A cheap check: grep the graded patch for added `__eq__` / `__ne__` / `__hash__` overrides that return a constant,
      and for `sys.exit` / `os._exit` in non-test source. Report them as `hack_suspect` for audit, not as an automatic 0.
    - A stronger check: side-effect mutants (`03-verifiers-traces-loops.md` §0; 2610.09142).
- **Monitor use.**
  - The trace audit and `llm_hack_judge` are offline. Keep them out of the reward.
  - If a hack signal must act during training, use it as a mask or screen (drop the episode), or route it (GRAFT).
    Never use it as a penalty a policy can learn to dodge.
  - Crucible's masking already drops, rather than penalises, infra failures (V8). The same plumbing can carry
    `hack_suspect` screening.
- **Inoculation.**
  - `TrainConfig.system` is the place.
  - Anthropic's milder line was "your task is just to make the grading script pass" [web, helper].
  - The trade-off: inoculation raises test-time hacking slightly if the line stays in at test time, so it must be absent
    from eval prompts.
  - This is a product decision for the lead; record it in the run identity.
- **Measurement.**
  - Add to the before/after benchmark one small agentic misalignment probe, run in the eval harness on the trained
    adapter:
    - does it edit tests when told not to?
    - does it misreport STATUS?
    - does it sabotage a "write a hack detector" request?
  - A pass-rate jump alarm exists (`PassRateAlarm`, torch loop only, `train/monitors.py:42-52`). Port it.
- **Honesty drift.** The Brier term is a guard. Watch `honesty/brier` and the share of false "done" claims per step: a
  rise in false claims during RL is the agentic analogue of sycophancy drift.

| recommendation | status | where |
|---|---|---|
| Hack-severity path rules for test plumbing, history, network | **have** | `factory/pathcheck.py` |
| L1 hack rules on the live `WorkspaceHarness` path | **partial** (NativeHarness only) | `runtime/harness/native.py:917-925`; `workspace.py` |
| AlwaysEqual and early-exit checks on the graded patch's non-test source | **missing** | `credit/hack_detector.py`; `factory/pathcheck.py` |
| Hack signals as screen or mask, never as a reward penalty; trace audit offline | **have** (by design) | `factory/traceaudit.py` |
| Inoculation line decision, recorded in run identity, absent in eval | **missing** | `TrainConfig.system` |
| Agentic misalignment probe in the before/after benchmark | **missing** | benchmark harness |
| Pass-rate jump alarm on the student path | **partial** (torch only) | `train/monitors.py:42-52` |
| Honesty drift: false-"done" share per step | **missing** | `factory/honesty.py`; step metrics |

---

## 6. Monitoring and diagnostics during RL

### 6.1 Which signals lead

| signal | evidence of lead time | source |
|---|---|---|
| truncation rate (episodes hitting step, token or context caps) | instabilities "often linked to truncations"; a high rate is a reliable warning | ScaleRL 2510.13786 [pdf] |
| response-length surge | led collapse by about 190 steps (4B: surge near step 110, collapse near 300) | 2602.01826 [pdf, helper; lead inferred] |
| reward std within groups | std drop at step 40 vs reward collapse at step 90 (FrozenLake) | RAGEN 2504.20073 [pdf, helper] |
| grad-norm growth and clip ratio | gradual grad-norm growth and a rising clip ratio came before instability; grad-norm spikes mark the point of no return | INTELLECT-2 [pdf, helper]; RAGEN |
| entropy | fast early collapse predicts the ceiling (R = −a·e^H + b); entropy turning back *up* preceded collapse in INTELLECT-2's ablations; a rising entropy with falling KL preceded a score collapse | 2505.22617 [abs]; INTELLECT-2 [pdf, helper]; vLLM blog [web] |
| train/infer mismatch (KL, extreme-ratio share) | share of tokens with ratio > 2 above 0.1 coincided with the crash; mismatch std doubles in MoE before instability | R3 [pdf]; 2602.01103 [abs, helper] |
| IS diagnostics | heuristic alarms: mean weight outside [0.5, 2], ESS < 0.3, IS std > 1, abs(kl) > 0.1 | verl rollout_corr docs [web, helper] |
| pass-rate jump | classic hack signature (LongCat's Lean4 pass rate fell 97.9% → 27.9% once the verifier was fixed) | `train/monitors.py` docstring |
| hack precursors | probe for proxy internalisation forecasts onset; 72% of hacks carry an explicit CoT rationale | PRIME 2606.09711, RHB 2605.02964 [abs] |

### 6.2 What Crucible logs on the student path, and what is missing

`TinkerTrainer.step_metrics` (`train/bridges/tinker/trainer.py:277-323`) plus `staleness_metrics` log:
- reward stats, Φ final, gate pass rate, pass rate;
- sequence and token advantage stats;
- `kl/sample_vs_train` and its abs max, the ratio stats;
- staleness max and mean, per-bucket token shares (and KL / masked share where the backend gives trainer logprobs);
- drops by reason, tokens (prefill, cached, sampled, trained);
- spend, time;
- quarantine and difficulty.

prime-rl's rows add `primerl/loss/mean` and `primerl/optim/grad_norm` (`primerl/backend.py:61`).

**Missing on the student path** (most exist in the torch loop's `StepMonitor`, `train/monitors.py`):
- **entropy**: at least the behaviour estimate −mean log p of sampled tokens, already computed in `train/runtime.py:915`;
- **lengths**: sampled tokens per call, calls (turns) per episode, trained tokens per episode, and the length-bias KS
  distance;
- **stop-reason shares**: `agent_done`, `max_steps`, `context_overflow`, `max_output_tokens`, `too_many_errors`,
  `timeout`-masked. This is the truncation rate, today about 19% `max_steps` in the targeting data [inf from harness
  audit §0];
- **tool-error and format-error rates per episode**;
- **honesty**: Brier, false-"done" share, give-up rate;
- **zero-variance group share** (now only as `drops/*`);
- **hack-rule hits and `hack_suspect` share**;
- **`routing/version_mixed_share`** is logged with a warning (async audit R5); keep it on the dashboard;
- **alarms**: pass-rate jump, truncation-share rise, length surge, entropy drop or rebound, grad-norm trend, reward-std
  drop. The torch loop has `PassRateAlarm`; nothing on the student path.
- `framework_health` (about 20 detectors) runs as an operator tool through `scripts/train_monitor.py`, never read by
  the trainer (map §2.2).

### 6.3 Recommendations

| recommendation | status | where |
|---|---|---|
| Port `StepMonitor`'s fields (entropy estimate, lengths, truncation share, zero-variance share, hack rate) into `step_metrics` | **partial** | `train/monitors.py`; `train/bridges/tinker/trainer.py:277-323` |
| Stop-reason shares and turns per episode in every step row | **missing** | same |
| Port `PassRateAlarm`; add truncation-share and length-surge alarms that write a `stop` record or page the operator (not an automatic stop) | **partial** | `train/monitors.py:42-52` |
| Log LoRA A/B norms and changed-element share per step (§3.5) | **missing** | prime-rl backend metrics |
| Run `scripts/train_monitor.py` (framework_health) every N steps from the job and surface exit code 3 | **partial** (operator tool) | `scripts/train_monitor.py:129-181` |
| One W&B panel per section: mismatch (KL, MIS share per bucket, routing refusals), learning (reward, std, pass, Φ, zero-var), behaviour (entropy, lengths, turns, stops, tool errors), honesty and hacking, system (time, tokens, spend) | **partial** | `tracking` sink |

---

## 7. Concrete settings for the planned DeltaAI run (summary)

These are the settings the sections above support. Items marked "change" are suggestions, not decisions.

| setting | today | suggestion | basis |
|---|---|---|---|
| correction | SC + MIS [0.5, 5], no PPO clip | keep | 2609.20807; IcePop band |
| routing | R3, refuse version-mixed | keep; watch refusal share | R3; 2512.01374 |
| sampling | neutral, `generation_config = vllm` | keep | DeepSeek Keep Sampling Mask is then unnecessary |
| trainer dtype | bf16 (forced on 2 GPUs) | **change**: fp32 adapters on bf16 base, or prove updates survive | §3.3 |
| LM head precision | not checked | check fp32 head in both engines | ScaleRL; MiniMax-M1 |
| staleness | 0 (sync) | 1 with `async_groups` 4 after the 2-step run | async audit R1; §3b |
| loss aggregation | global token mean | keep | Dr.GRPO; async audit R9 |
| advantage | Dr.GRPO (no std) + honesty + Φ turn credit (η 0.1) | keep; add anchor-state diagnostic | §2.3 |
| zero-variance groups | dropped on composed advantage | keep | DAPO; ScaleRL |
| KL | none | keep for 40 steps; k1-in-reward if runs get long or async ≥ 2 | 2512.21852 |
| LoRA | r 32, α 64, lr 4e-5, attention + MLP + experts | keep; check α/r parity with Tinker | LoRA Without Regret |
| entropy control | none on student path | log first; act only on fast collapse | 2505.22617 |
| compaction | off | keep off; align the eval harness | §1 |
| dashboards | reward / KL / staleness / drops | add §6.2's missing lines before step 3 | §6 |

---

## 8. Uncertain or unverified items

- **The two mismatch blogs** (TIS by Feng Yao et al.; "When Speed Kills Stability" by Liu, Li et al.) did not render
  through the fetcher. Their content here comes from partial renders, the verl and ms-swift docs, and search summaries.
  Treat their details as [sec], except the general mechanisms, which several primary papers restate.
- **Secondary numbers:**
  - ARLArena's +25.2%;
  - GDPO's AIME +6.3%;
  - RTMC's 52.2% / 49.7%;
  - the ImpossibleBench flag-option rates (from the authors' post);
  - the monitor-intervention study (OpenReview and LessWrong only);
  - GRAFT's 30% (Anthropic alignment blog).
- **Lead times from 2602.01826** are arithmetic from reported step numbers.
- **The bf16 rounding analysis** (§3.3) is my arithmetic. The student's hidden size, prime-rl's optimizer-state dtype,
  and Tinker's LoRA α were **not checked**.
- **Corrected helper claim:** the FP16 paper's BF16 LoRA collapse was on Qwen2.5-Math-1.5B, not Qwen3-30B-A3B.
- **Contradictions in the literature:**
  - rank-1 LoRA sufficiency (Thinking Machines vs 2512.23165);
  - whether FP16 fixes collapse (2510.26788 vs 2602.01826);
  - whether MIS alone suffices (2609.20807 and Ring-1T vs 2602.01826).

## 9. Sources

**PDFs in `crucible/docs/research/papers/` (git-excluded) read for this note.**
- New, the most important first: 2610.02163 (AutoCompact), 2607.05378 (CompactionRL), 2510.26788 (FP16), 2511.18397
  (natural emergent misalignment), 2510.11370 (R3), 2510.13786 (ScaleRL), 2510.06727 (SUPO), 2510.11967
  (Context-Folding).
- Also downloaded, with the parts cited read: 2505.10978 (GiGPO), 2512.24601 (RLM), 2510.01161 (M2PO).
- That is eleven new PDFs, more than the 6–8 asked for. The first eight are the ones to read.
- Already present and re-read: 2512.02556 (DeepSeek-V3.2), 2609.20807 (score centering).

**Verified arXiv ids**, grouped by section (title and first-version date checked):
- **§1** 2506.15841, 2507.02259, 2509.13313, 2510.06727, 2510.11967, 2510.24699, 2511.07327, 2512.24601, 2512.02556,
  2510.24701, 2610.02163, 2607.05378, 2610.22174, 2609.00865, 2512.22733, 2609.37082, 2609.35540, 2512.22087,
  2610.04902, 2511.21689, 2610.31778, 2606.19348 (DeepSeek-V4), 2602.15763 (GLM-5), 2602.02276 (Kimi K2.5), 2607.24653
  (Kimi K3), 2605.26494 (MiniMax-M2), 2601.16725 (LongCat-2601), 2506.13585 (MiniMax-M1).
- **§2** 2505.10978, 2604.11037, 2609.02236, 2608.01667, 2603.08754, 2607.13988, 2505.11821, 2512.17008, 2511.20718,
  2602.21534, 2402.19446, 2503.15478, 2505.20732, 2506.11902, 2509.21240, 2507.19849, 2510.14545, 2607.15610,
  2504.15275, 2502.01456, 2603.06621, 2605.30451, 2504.20073, 2510.10304, 2603.21357.
- **§3** 2510.18855, 2510.11370, 2609.20807, 2510.26788, 2602.01826, 2512.01374, 2512.23075, 2507.18071, 2507.20673,
  2510.13786, 2503.14476, 2503.20783, 2505.22617, 2506.01939, 2505.24864, 2506.09477, 2510.01555, 2512.21852,
  2512.23165, 2601.06677, 2602.01103.
- **§3b** 2505.24298, 2512.06547, 2509.19128, 2509.18521, 2505.07291, 2512.16144, 2505.24034, 2506.06122, 2510.11345,
  2409.19256, 2510.12633, 2604.26256, 2504.15930, 2511.14617, 2511.16108, 2510.01161, 2607.18722, 2607.22186,
  2610.09597, 2602.10693.
- **§4** 2602.11524, 2605.06078, 2603.19685, 2504.14870, 2606.07074, 2604.08545, 2605.26952, 2603.17673, 2504.13958,
  2505.18086, 2510.04474, 2602.02099, 2503.04697, 2501.12599, 2601.05242, 2507.17746, 2509.25760, 2601.20126,
  2509.04664, 2606.28733, 2510.20270, 2512.08093.
- **§5** 2511.18397, 2502.17424, 2508.17511, 2609.06649, 2510.04340, 2510.05024, 2503.11926, 2511.11584, 2601.23086,
  2507.11473, 2610.17062, 2604.23488, 2606.09711, 2605.02964, 2602.01002, 2610.09142.

**Web sources.**

| source | used for |
|---|---|
| Thinking Machines, "LoRA Without Regret" (thinkingmachines.ai/blog/lora) | LoRA rank, LR, layers, batch size |
| Thinking Machines, "Defeating Nondeterminism in LLM Inference" | batch-invariant kernels |
| vLLM blog, "Finish RL Training Faster: Bitwise-Consistent…" (vllm.ai/blog/2026-10-14-train-inference-consistency) | bitwise-consistent RL |
| Feng Yao et al., TIS blog (fengyao.notion.site/off-policy-rl) | TIS |
| Liu, Li et al., "When Speed Kills Stability" (yingru.notion.site) | MIS and sequence-level corrections |
| HF TRL precision write-up (aminedirohf-trainer-generator-bf16-mismatch.hf.space) | BF16 rounding, phantom clipping |
| verl fully-async and rollout_corr docs | async speedups, IS alarms |
| NeMo-RL async docs | staleness defaults |
| AReaL-lite and slime READMEs | async designs |
| Anthropic, "Training on Documents About Reward Hacking" | document-induced hacking |
| Anthropic alignment blog (GRAFT) | gradient routing |
| OpenReview, ICML 2026, "Designing Effective Monitor-Based Interventions…" | screening vs penalty |
