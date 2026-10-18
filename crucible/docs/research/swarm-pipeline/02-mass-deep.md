# Swarm pipeline, study 02: MASS (Sakana AI, Multi-Agent Self-Supervision) traced in code

Source: `RL-Environments/06-frameworks-and-env-code/sakanaai-mass` (HEAD `fb40358`, 107 files, ~6k lines of Python/shell).
Paper arXiv 2610.12176. Every claim cites `file:line`, relative to that repo root. `RHI` means
`harness_improvement/iterate_multi_agent_prompt.py` (2535 lines, the workflow optimizer). Read-only study: no code run,
no network.

## 0. Licence: PATTERN only, no PORT

- There is **no LICENSE file**. `NOTICE.md:12-13`: "No public open-source license has been assigned to the MASS-specific
  code in this repository." Default copyright applies, so all rights are reserved.
- Consequence for Crucible: **we copy no code and no prompt text** (prompts are copyrightable text too). Every idea below is
  **PATTERN**: we re-implement it from the description, in our own words and our own structure. Third-party parts (qwen-code,
  Harbor, ScienceAgentBench, MLR-Bench) keep their own licences (`NOTICE.md:3-6`) and do not matter for us here.

## 1. What the repo is, end to end

The stage driver is `mass/pipeline.py`. `search` runs RHI, `collect` gathers teacher trajectories, `rank` does self-judged
pairwise matches with Bradley-Terry, `select` takes ranks 1-15 for train and rank 16 for val, `render` builds masked SFT
windows, `train` fits the LoRA, `export` merges and quantizes to FP8, `rollout`/`report` run external evaluation
(`pipeline.py:328`, `README.md:140-148`). One model plays executor, optimizer and in-loop judge: `model_id` is passed to all
three (`pipeline.py:80`, `:119`, `mass/judging.py:24-27`; `docs/paper_to_code.md:5`). L0 is Qwen3.6-27B-FP8 behind vLLM
(`configs/paper.json:4`, `configs/models.json:2-3`). Cycle 2 repeats everything from the merged L1 (`configs/cycle2.json:2-8`).

## 2. Workflow representation, runtime, contracts, workspaces

**Representation is plain prompt text, not a schema object.** A "workflow" is one block of the task prompt. It runs from the
marker `Create an agent team with following agent candidates to solve this problem:` up to the last `Use uv for all Python
workflows` (`RHI:35-36`, `:67-81`; `rfind` because the phrase can recur inside the block, `:70-72`). The prompt is split into
`before_design / design_block / after_design` (`RHI:39-47`, `:80-98`). A new version is re-spliced (`RHI:922-923`).
- v0 (`tasks/initial/query*_ourTeam.txt`) is a JSON-ish list of `{agent_id, role, instruction}`
  (`tasks/initial/query51_ourTeam.txt:3-25`). There are 4 agents (finance) or 6 (pharmacy, robotics). The **same team is
  reused for every task of a domain**: 51/53/56/60 share one team and 202/205/207/221 share another (agent ids identical
  across files). v0 has **no contract or hop fields**.
- "Contract" and "hop" exist only as **optimizer instructions**. The optimizer is told to strengthen each subagent's
  "Output to orchestrator" schema and to add orchestrator-subagent hops with recall and re-delegation (`RHI:733-736`,
  `:902-904`). Whatever structure appears in v1+ is free text the model wrote. Nothing parses it.

**Runtime.** `runtime/run_episode.sh` runs one headless qwen-code 0.20.0 process: `qwen --approval-mode yolo
--output-format stream-json < prompt.txt` in the workspace dir (`run_episode.sh:172-189`). The orchestrator is the main
qwen-code thread. It spawns subagents through qwen-code's built-in `agent` tool. MASS adds **no multi-agent runtime code**.
Coordination rules sit in the appended teacher directive (`tasks/teacher_directive.txt`, collection only,
`pipeline.py:147`):
- every non-orchestrator role must be a real `agent` call with `subagent_type "general-purpose"`, and the orchestrator
  writes no pipeline code itself (`:3`);
- every call runs in the foreground (`run_in_background: false`, `:5`); work goes stage by stage, and work runs in parallel
  only when inputs exist and outputs are disjoint (`:7`);
- each worker prompt is self-contained: role instruction, absolute path, files to write, upstream interfaces, a verify
  command, and the expected report (`:9`);
- verify, then re-delegate a narrower task to the same role instead of fixing the work yourself (`:11`); one deliverable
  per call (`:13`); a final summary of every worker call (`:15`).

**Contract enforcement: none in code.** The only programmatic checks on a design come after the optimizer writes it:
- the required start marker (`RHI:940-944`);
- optional regex rejection of numeric caps on agents, hops, turns or time (`RHI:964-985`, `:2411-2434`; *off* in the
  paper driver, which does not pass `--forbid-fixed-agent-limits`, `pipeline.py:112-125`);
- an optional size cap (`RHI:2435-2454`, also off).

At runtime no one checks that subagents returned the schema. Worker outputs are just files and tool results.

**Workspaces.** Each episode starts in an empty dir. The runner refuses a non-empty workspace or an existing
`run_status.json` (`run_episode.sh:45-52`). It has a fixed config: T=0.6, top_p 0.95, top_k 20, max_tokens 32768/response,
context 262144, shell timeout 600 s, HTTP 1800 s, a seed, thinking on (`run_episode.sh:112-163`). It runs on the **host with
no sandbox** (`execution_mode: "headless_host_visible_yolo"`, `:305`; `docs/running.md:13-16`). A logging reverse proxy
records every request and response byte-faithfully, for SFT (`runtime/api_proxy.py:1-16`, started at
`run_episode.sh:85-103`). The status record is `exit_code`, `trace_has_success_result`, and a `failure` fingerprint: the
final error plus the last 5 tool calls (`run_episode.sh:229-261`). An "API Error … ECONNREFUSED/timeout" final turn is
relabelled as an infra failure, never judged (`:199-224`, `:263-266`). The deliverables the task asks for are what gets
judged: `research_report.md`, `deliverables/index.json`, `deliverables/plots/*.png`, `results/*.json`, code
(`tasks/bare/query51.txt:1`).

## 3. Workflow search (RHI)

**Budget per task.** `search.iterations = 11` (`configs/paper.json:17`) gives:
- 1 bare-task reference episode (`pipeline.py:100-101`);
- 12 workflow executions, v0..v11 (`pipeline.py:103-107`);
- 11 optimizer calls; the last RHI call is `--evaluate-only`, so it proposes nothing (`pipeline.py:129-130`, `RHI:2196-2202`);
- 12 baseline judgments (v0 seed + v1..v11, `RHI:1799-1805`), plus 0-11 champion arbitrations.

Each workflow version is executed **once** (one seed per version: `seed_base + q + k*1000`, `pipeline.py:107`).

**Comparison rule actually used: baseline-anchored + champion arbitration** (not the paper's Algorithm 1;
`docs/reproduction.md:11-15`, `README.md:217-222`).
- From k ≥ 1 the driver passes `--judge-vs-baseline --baseline-repo-root reference/` (`pipeline.py:126-128`). Each version
  is judged once against the **fixed single reference run**, key `v{k}_vs_no_harness` (`RHI:1702-1797`). v0 is judged once
  as a seed (`RHI:1804`).
- Champion (`_derive_champion_baseline`, `RHI:532-573`). The first version to beat the baseline takes the crown outright.
  After that, a new baseline winner must beat the champion in **one** direct arbitration call, key
  `champion_v{c}_vs_v{k}`, kept in a separate ledger (`RHI:2104-2190`). Losses and ties never move the champion. If no
  version ever beats the reference, v0 holds the crown "without a proven win" and `beats_reference = false`
  (`RHI:2191-2199`). The task is then dropped from collection (`pipeline.py:142-144`). This happened to task 221
  (`tasks/splits.json:5-7`).
- Note (`docs/paper_to_code.md:45-50`): the champion record does **not** make the executed sequence monotonic. v{k+1} is
  written from v{k} even when v{k} lost. The champion is only shown as context.

**Optimizer prompt** (`_build_single_submission_improvement_prompt`, `RHI:597-919`). The system role is "principal prompt
engineer". It edits **only** the design block and states a priority for multi-agent-only advantages: parallel
specialists, cross-agent critique, agent-to-agent talk, reconciliation (`RHI:727-741`). The user message has these
sections (`RHI:877-918`):
1. task + current design;
2. the champion design, with the judge's rationale for why it beat the current one (BEST-H, `--include-best-design`,
   `RHI:745-808`);
3. the v0 design as the root, to show drift (`RHI:839-852`);
4. workspace evidence: tree, preflight, report/JSON/code excerpts, then *all* `results/*.json` capped at 24k/file and
   120k total (`RHI:1986-2001`, `:988-1094`; caps `pipeline.py:125`);
5. execution metadata from `run_status.json`. If the run failed, a FAILURE NOTICE asks for anti-stall contracts and
   re-delegation, with no numeric caps and no collapse to a single agent (`RHI:2003-2053`). A `prior_failures`
   recurrence note (`RHI:2054-2081`) is dead code here, because nothing in the repo writes that field;
6. the pairwise history, with a W/L/T scoreboard and the champion starred (`RHI:576-594`). Each entry carries the verdict,
   a 2400-char rationale excerpt, and "actionable takeaways": sentences matching cue words like *fail, missing,
   inconsistent, nan* (`RHI:303-337`). It also carries a **unified diff of the design change behind that verdict**
   (DIFF-HISTORY, `--history-design-diffs`; 1500 chars for the 3 newest, 400 for older, `RHI:347-397`);
7. a recurring-issue checklist: regex buckets counted over all rationales (`RHI:447-489`);
8. instructions. Output is JSON `{improved_multi_agent_design, changes_from_previous, why_it_should_improve,
   evidence_used, expected_impact, verification_checks}` (`RHI:620-632`), with a plain-text fallback mode
   (`RHI:633-643`, `:2383-2400`).

Sampling for optimizer calls: T 0.6, top_p 0.95, top_k 20, 49152 output tokens, thinking on, context guard
chars/4 + output ≤ 262144-8192 (`pipeline.py:120-125`, `RHI:162-188`).

**Noise and instability handling (thin):**
- one judgment per comparison: no repeats, no majority vote;
- the optimizer is merely *told* that each verdict is a single judgment and that trends matter more than flips
  (`RHI:867-874`);
- no position swap in RHI. The reference is always A and the version always B (`RHI:1723-1749`); the champion is always A
  and the current version B (`RHI:2124-2150`). Any B-bias in the judge favours the newer design;
- run-type labels reach the judge: "no-harness-single-agent-baseline" vs "multi-agent-design-v{k}"
  (`RHI:1747-1748`, rendered at `evaluation_claudecodex/pairwise_judge.py:50-52`);
- no automatic retries. Invalid judge JSON raises (`harness_improvement/llm_backend.py:76-113`), and a nonzero episode exit
  stops the driver (`pipeline.py:50`, `docs/reproduction.md:60-63`).

The engineering is solid where it is used:
- write-once artifacts that refuse to overwrite different content (`RHI:143-159`, `:191-206`, `:2311-2322`);
- an orphan-result guard (`RHI:1857-1862`);
- a frozen config per `run_dir` (`pipeline.py:57-62`);
- completed episodes are reused only if the prompt bytes match (`pipeline.py:73-77`);
- optional provenance checks on reused verdicts (`RHI:251-300`);
- `--require-local-vllm`, which refuses any non-loopback backend for self-RHI (`RHI:1464-1484`).

## 4. Evaluator

**Evidence bundle** (`evaluation/bundle.py:31-101`, `evaluation/artifacts.py`):
- a file tree of at most 400 entries (`artifacts.py:171-243`);
- a "programmatic preflight" that only checks **file existence**: 4 core paths, plus the PNG paths regex-extracted from the
  task text (`artifacts.py:129-168`, `:383-403`);
- capped excerpts of `research_report.md`, `deliverables/index.json`, `results/metrics.json`,
  `results/model_comparison.json` and `requirements.txt`, up to 4 entry scripts (`run_*.py`, `main.py`, `pipeline.py`) and
  one test file (`artifacts.py:317-380`).

PNGs and binaries are listed by name only (`artifacts.py:26-50`). **Nothing is executed. Figures are not seen. Metrics are
read as the agent wrote them.**

**Pairwise prompt.** The system prompt gives a domain persona by task-id band: quant 1-99, pharmacy 200-299, robotics
300-399 (`evaluation_claudecodex/pairwise_prompts.py:13-46`). Rules (`:4-10`):
- the task block is the only contract;
- workspaces are evidence only;
- cite concrete evidence;
- answer `tie` only when the two are genuinely comparable.

The rubric has 6 axes: deliverable coverage, rigor, reproducibility, presentation, engineering, task alignment
(`pairwise_prompts.py:74-96`). The output is `{"winner": "A"|"B"|"tie", "rationale"}`, validated by pydantic
(`pairwise_schema.py:8-10`).

**Within-task ranking for SFT** (`mass/pipeline.py:162-198`).
- 18 candidates per task (`configs/paper.json:18`) on a seeded connected ring schedule: each candidate meets 6 opponents,
  giving 54 unique pairs (`pipeline.py:162-172`; test `tests/test_pipeline.py:38-43`). A/B order is randomized per pair
  (`pipeline.py:171-172`), but each pair gets **one** self-judgment with neutral labels `candidate-a/b`
  (`mass/judging.py:18-33`).
- Only episodes with `exit_code 0` and a success result are eligible (`pipeline.py:65-70`, `:184`).
- Bradley-Terry by MM iterations (500), a tie counts 0.5/0.5, and a weak prior gives 0.5 virtual games against the mean
  (`mass/ranking.py:4-34`). Scores are log-strengths (`:34`).
- Ranks 1-15 go to train, rank 16 to val, and 17-18 are discarded (`pipeline.py:201-219`). **So BT removes only 2 of 18.**
  Selection pressure really sits at the workflow gate (`beats_reference`) and the success filter.

**External reporting** (outside the loop, `mass/judging.py:36-58`, `pipeline.py:304-323`). GPT-5.5 at effort xhigh and
Claude Opus 4.8 at effort max each judge every pair 3 times. Seeds 42-44 set a random A/B swap, which is un-swapped
afterwards (`judging.py:50-56`). The summary is W/(W+L) with ties excluded (`pipeline.py:319-323`). Rollouts use the
**bare** task only (`pipeline.py:292-301`).

**Calibration, blind-spot checks and agreement: not in the repo.** There is:
- no judge calibration set;
- no seeded-defect control;
- no position-bias measurement;
- no self-vs-external agreement code ("Evaluator agreement across generations … additional code required",
  `docs/experiment_coverage.md:33`).

The paper's eligibility filter (success plus 0.6 deliverable) is also absent (`docs/reproduction.md:56-58`). MLR-Bench uses
the official rubric with two external reviewers. A missing paper or one under 200 bytes scores 0 (`benchmarks/scripts/bench_mlr_judge.py:66-90`).

## 5. Tasks

- **12 hand-written open-ended research tasks**: finance 51/53/56/60, pharmacy 202/205/207/221, robotics 302/305/307/324
  (`tasks/splits.json:8-12`). The split is 9 designated train, 3 test (60, 207, 305), and 8 SFT (221 excluded,
  `tasks/splits.json:2-7`).
- The split is validated as disjoint (`pipeline.py:18-26`). Test tasks are refused at collection (`pipeline.py:139-140`).
  RHI may still run on test tasks (`tasks/README.md:7-9`).
- **No task-generation code ships.** Hints of a larger private catalog: the judge rules cite
  `multi_agent_design/*/queries.json` (`pairwise_prompts.py:6`), and RHI hard-codes exclusions for query57/220, which are
  not released (`RHI:1029-1049`). The 25 KB of task text in total (`tasks/bare/*.txt`) is the whole spec.
- A task is shaped as: a study goal; public data sources, often with download recipes (`tasks/bare/query202.txt:1`); then a
  **Deliverables** list. That list asks for a conference-style report of 6-12 pages, named plots, pinned reproducible code
  with one entry point, a metrics JSON, model-comparison and ablation JSONs (`tasks/bare/query51.txt:1`), and a `logs.txt`
  history (`:3`).
- Why the tasks are long-horizon: live data acquisition, model training, required ablations, multi-artifact consistency
  between report and JSON, no turn cap, and a 262k context with 32k-token responses (`run_episode.sh:305-311`).
- **No hidden tests and no verifiable outcome.** "Success" is only "qwen-code ended without an error" (`run_episode.sh:195-198`).

## 6. Distillation

**Teacher prompt** = retained workflow prompt + teacher directive (`pipeline.py:145-150`). 18 seeds per eligible task
(`pipeline.py:151-159`).

**Rendering** (`training/prepare_sft_data.py`):
- Requests come from the proxy log (`:43-57`). Threads are grouped by a hash of system + first user message and segmented
  by message-prefix continuity (`:91-119`). The last request plus its response gives the full conversation (`:368-376`).
  A truncated final response is dropped by default (`:374-376`, `:423`).
- **Role** comes from the system-prompt prefix: the main thread (matched prompt, or "You are Qwen Code") is the
  orchestrator, "You are a general-purpose subagent" is a worker, anything else is auxiliary (`:182-188`).
- **Workflow-text removal:** only in the orchestrator's first user message, the teacher prompt is replaced by the bare
  task. It tolerates qwen-code's normalisations of `@` tokens and whitespace, and asserts exactly one hit (`:128-179`,
  `:377-381`). Worker prompts keep the role instructions the orchestrator copied into them (`docs/paper_to_code.md:31-32`).
  So the *team design* is internalised by the orchestrator, while workers still see their delegated briefs.
- **Loss masking:** labels −100 on system, user, tool and all context tokens. The loss runs from after the generation
  prefix `<|im_start|>assistant\n<think>\n` through `<|im_end|>` of each assistant block (`:252-273`), with an empty-think
  edge case (`:258-265`). Per user-query segment, only that segment's assistant turns are trained, because the Qwen
  template strips earlier reasoning (`:222-281`, `in_segment` `:253`). Block-wise tokenization must equal whole-render
  tokenization, or rendering errors (`:276-279`).
- **Windows:** header (system + first user) + ≤ 8192 masked overlap tokens that always include the segment's query +
  contiguous loss blocks, in 49152-token windows (`:284-344`; `configs/paper.json:21`). `--strict` fails on any thread
  error, skipped block or missing main thread (`:484-495`). The driver also fails if any selected episode yields no window
  (`pipeline.py:236-243`).

**Training** (`training/train_sft_lora.py`):
- LoRA r 64, α 128, dropout 0.05, on q/k/v/o/gate/up/down plus the linear-attention projections `in_proj_qkv`,
  `in_proj_z`, `out_proj` (`:47-50`, `:194-195`).
- AdamW β (0.9, 0.95), lr 3e-5, warmup 5, cosine to 10% (`:356-366`; `configs/paper.json:22-24`). 1624 steps, 3 replicas ×
  1 window, 2 GPUs per replica (`pipeline.py:38-39`, `:246-256`).
- Sampler: task uniform, then orchestrator with p = 1/3 else worker/auxiliary, then a uniform window, with replacement
  (`:140-165`; `pipeline.py:252`). Loss is token-mean per window (`:450`).
- Checkpoint = minimum validation loss on the rank-16 trajectories (`pipeline.py:259-274`). Then merge, then block-FP8
  (`pipeline.py:277-289`). The paper run used 6 H100 for about 19 h of SFT alone (`docs/running.md:10-11`).

**Outcome as reported** (`README.md:57-66`):
- research benchmarks go up from L0 to L2: SAB 28.8 to 31.7, MLR 1.71 to 2.25, MLR delivery 57% to 76%;
- **coding goes flat or down**: Terminal-Bench 2.0 0.375 to 0.352, SWE-bench Verified 0.626 to 0.611; DSBench is flat;
- much of the MLR/Asta gain is *delivery rate*. Asta's score among delivered reports *falls*, 0.105 to 0.088
  (`benchmarks/README.md:40-44`);
- the brief's "1.2-1.6× score/token" claim is not checkable here, because the token-efficiency analyses are not included
  (`docs/experiment_coverage.md:35`).

## 7. `harness_improvement/` = the RHI optimizer, not a self-modifying runtime

It holds `iterate_multi_agent_prompt.py` (judge + champion + optimizer, one step per call) and `llm_backend.py` (local vLLM
chat or OpenAI Responses JSON client, `llm_backend.py:162-354`). "Harness" means the multi-agent design *text*. The
qwen-code runtime is never changed.

How an improvement is accepted:
1. it is **written unconditionally** as the next version if it has the marker (and passes the optional cap/size
   validators), then executed (`RHI:2405-2459`);
2. it becomes **retained** only by beating the single reference run, and then the champion, under one self-judgment each
   (§3);
3. the retained version goes to collection only if `beats_reference` (`pipeline.py:141-144`).

A separate `--judge-llm-backend` lets an external judge steer a local optimizer (`RHI:1194-1229`, `:1618-1669`), but the
paper config does not use it.

## 8. Weaknesses for our purpose (verified long-horizon coding RL tasks)

1. **Non-verifiable grade end to end.** The judge reads agent-written claims (metrics JSON, report text) and file existence
   (§4). A fabricated `metrics.json` that agrees with a fluent report is indistinguishable from real work. Success is
   "process exited cleanly" (`run_episode.sh:195-198`).
2. **Homogeneous blind spots.** The same model executes, judges its own output and rewrites the workflow
   (`pipeline.py:119`, `judging.py:24`). Systematic errors the model cannot see (wrong leakage handling, silently broken
   backtests) are invisible at every stage. No goldset or seeded defects measure that blindness (§4).
3. **Judge-steering loop (reward hacking at the recipe level).** The optimizer sees the verdict rationales, cue-word
   "takeaways" and a checklist built by regex over judge prose (`RHI:303-337`, `:447-489`). So it optimises toward what the
   judge *says*, not toward what is true.
4. **Biases that inflate wins.** The optimizer is told to beat the "no-harness" baseline (`RHI:810-833`) and is biased
   toward more hops and agents by the prompt (`RHI:733-736`), not by evidence. Meanwhile the judge sees run-type labels
   (`RHI:1747-1748`) and a fixed A/B position (§3).
5. **Statistical fragility.** The reference is n = 1 and each version is n = 1. `beats_reference` is one sample against one
   sample under one judgment. RHI has no MDE and no CI. BT selection removes 2/18 (§4).
6. **Safety.** It runs yolo on the host with internet and live data (`docs/running.md:13-16`). It is not reproducible:
   data drift, unpinned images (`docs/reproduction.md:73-75`).
7. **The transfer to coding failed in their own numbers** (§6). Distilling self-ranked research-report trajectories does
   not buy coding skill.

## 9. Mapping into Crucible's task-building swarm (all PATTERN; licence §0)

Crucible already has much of what MASS has only in prompts:
- the M17 machine is data, with typed ports, guarded transitions, bounded loops and loop-back findings to the owner
  (`src/crucible/factory/machine.py`);
- the funnel runs spec → author → realism → oracle → verifier → ledger → floors → probes → hackers → second_verifier →
  solvers → differential → audit → flake → horizon → gate (`factory/driver.py`);
- outcomes are executed and verified, and the verifier's error is measured (`factory/goldset.py`).

MASS adds a **search loop over the recipe** and a **distillation loop over the swarm**.

### USE

| # | MASS idea (cite) | Crucible form | Fitness / guard |
|---|---|---|---|
| U1 | Workflow search with a fixed reference + champion that advances only on direct wins (`RHI:532-573`, `:2104-2190`) | **Recipe search over the factory itself.** The search space is `driver_machine()` role prompts, loop bounds, model assignments and stage order. The reference is the production recipe. The champion is promoted only on a paired direct win | **Executed metrics only**, on a frozen cell set (the same K intents/windows, same seeds): accepted-task yield per $, mutant catch rate, hacker confirmed-hack count = 0, solver pass-rate inside the target band (study 11 had 0/123, too hard), differential disagreement and goldset false-pass ≤ bound, flake bound. Paired sign test / MDE over K cells, never one sample |
| U2 | DIFF-HISTORY: each verdict shown with the design diff that caused it (`RHI:347-397`) | The recipe optimizer sees per-stage funnel deltas next to the YAML/prompt diff that produced them | Credit assignment from measured stage counts, not judge prose |
| U3 | Failure fingerprint into the optimizer: final error + last 5 tool calls (`run_episode.sh:229-261`, `RHI:2020-2053`) | Escalations and masked episodes carry a fingerprint into the recipe optimizer and into the owning role's loop-back | Already close to the machine's `carry`. Add the tool-call tail |
| U4 | Post-generation validators that reject a proposed design (`RHI:964-985`, `:2435-2454`) | The machine validator refuses a proposed recipe that breaks `isolated_from`, `different_family_from`, the USD cap or the canary screening | Hard refusal, recorded like `rejected_result_*.json` |
| U5 | Self-contained delegation checklist (`tasks/teacher_directive.txt:9-13`): path, files, upstream interfaces, verify command, report shape; re-delegate narrower on failure | A **lint over role prompts** in the swarm (like M17's prompt-ledger lint). Loop-back goes to the owner with *narrower* findings | Our typed ports/verdict `{ok, owner, findings, measured}` replace MASS's free-text "Output to orchestrator" contracts. Keep ours (enforced). Take MASS's recall/re-delegation emphasis as loop-back policy |
| U6 | Champion/pairwise + BT for *variants* (`pipeline.py:162-198`, `ranking.py`) | Only for **non-verifiable secondary axes** among variants that already passed every executed gate: prompt naturalness/realism, spec clarity | Cross-family judges, both orders, neutral labels, ties allowed, goldset-calibrated. Never a reward, never a gate on correctness |
| U7 | Distil the swarm's best trajectories with the recipe text removed (`prepare_sft_data.py:161-179`, `:377-381`) | SFT the factory's role models (spec author, oracle writer, verifier writer, hacker) on trajectories from **gate-accepted** tasks. The long recipe prompt is replaced by the bare role brief so the behaviour is internalised | Selection by executed acceptance (oracle 1, floors 0, mutants caught, flake clean), not self-rank. Held-out families sealed (`factory/sealing.py`) |
| U8 | Byte-faithful proxy log → render exactly what the served model saw; segment-aware Qwen-think masking; block-vs-whole tokenization check; strict audit; task × role balanced sampler (`api_proxy.py:1-16`, `prepare_sft_data.py:222-344`, `:484-495`, `train_sft_lora.py:140-165`) | The same discipline for any SFT/RL data we render from multi-thread episodes (the RLM tree of M16, factory roles): per-thread role labels, loss on assistant tokens only, no silent drops | Directly relevant to M12 trainer data hygiene |
| U9 | Write-once artifacts, frozen run config, prompt-byte match on reuse (`RHI:191-206`, `pipeline.py:57-77`) | Already in the forge `Journal`/`Ledger`. Add the prompt-hash match on episode reuse if it is missing | |

### REJECT

| # | MASS choice | Why not |
|---|---|---|
| R1 | Self-judging as the selection/reward signal (`pipeline.py:175-198`, `RHI:1702-1797`) | We reward executed hidden tests and milestones. A self-judge has the blind spots of §8.2 and invites §8.3 |
| R2 | One judgment per verdict, fixed position, labels visible (`RHI:1723-1749`) | If we judge anything (U6): ≥ 2 orders, neutral labels, repeated calls, measured position bias |
| R3 | "More hops/agents, no numeric caps" as an optimizer prior (`RHI:733-736`, `:712-722`) | Our recipe objective includes $ and wall clock (ledger caps). Agent count is a cost, not a virtue |
| R4 | Single-sample reference and versions (`pipeline.py:100-107`) | Our acceptance needs paired, multi-cell evidence with an MDE (study 11 metrics) |
| R5 | Host yolo, live internet data (`run_episode.sh:186`, `docs/running.md:13-16`) | We use a World with `--network none` and executed leak probes |
| R6 | SFT on self-ranked research trajectories to gain coding skill | Their own coding scores fell (`README.md:65-66`). Use RL on verified tasks for the policy. Use distillation (U7) only for factory roles, gated by execution |
| R7 | Task creation | MASS ships none (12 fixed prompts, domain-generic teams). M17's mining + ledger + oracle + mutants is far ahead. Nothing to take |

### Concrete next step (if the lead approves)

A recipe-search experiment on M17 should keep the production machine as the reference and draw K ≈ 10 frozen cells
(option-B set). The optimizer is a cross-family model. It gets the funnel table + the YAML diff history (U2) + the
fingerprints (U3) and proposes one recipe edit per round. Each edit is checked by the machine validator (U4) and executed
on all K cells with the same seeds. The champion is promoted on a paired win across cells on the executed fitness (U1),
under a fixed USD cap per round. Stop after N rounds without a promotion. The output is a better recipe, plus a corpus of
accepted role trajectories for U7.
