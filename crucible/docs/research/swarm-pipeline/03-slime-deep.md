# 03 · slime deep study: what to depend on, port, or copy as a pattern

Scope: THUDM slime at HEAD `859938d` (shallow clone), at `RL-Environments/06-frameworks-and-env-code/thudm-slime`.
Paths below are relative to that root unless they start with `crucible/`. The study was read-only: nothing was run and
no network calls were made. Every claim cites file:line.

**Licence.** Apache-2.0 (`LICENSE:1-3`). We may **DEPEND** on it or **PORT** code, provided we keep the licence text and
attribution and mark modified files (Apache §4). The patent grant is included; nothing here is copyleft.

**Provenance.** slime trains GLM-4.5 through GLM-5.3 (`README.md:20`). It claims rollout-only and train-only debugging,
reproducibility controls, recovery and CPU/GPU tests (`README.md:23`). The agentic CI covers real Codex and Claude Code
training (`examples/coding_agent_rl/README.md:293-294`).

---

## 1. Architecture

### 1.1 Processes and placement
- **One entrypoint, `train.py`; there is no `train_async.py`.** The rollout function decides sync vs async.
  - Loop (`train.py:47-100`): `rollout_manager.generate(rollout_id)` (:51) → critic then actor train (:60-67) →
    checkpoint (:73-84) → `pause_rollout_admission`, `update_weights()`, `resume_rollout_admission` (:89-94) → periodic
    eval (:99). An initial weight push happens at :22.
- **Ray placement** (`slime/ray/placement_group.py`).
  - Colocate: one placement group of `max(actor,rollout)` GPUs, rollout offset 0 (:117-118). `--colocate` forces
    `--offload` (`arguments.py:72-77`).
  - Disaggregated: two placement groups, so a restarted trainer can resize without moving engines (:126-136). Bundles
    are `{GPU:1,CPU:1}` with PACK (:45-100).
- **Train actors.** `RayTrainGroup` (`slime/ray/actor_group.py:16`: `async_train` :148, `update_weights` :178);
  `TrainRayActor` (`slime/ray/train_actor.py:25`); Megatron actor `train` at
  `slime/backends/megatron_utils/actor.py:400` (advantages at :431/:530), `update_weights` at :629.
- **Rollout side.** `RolloutManager` (`slime/ray/rollout.py:35`): `generate` (:317) → `call_rollout_fn` (:545) →
  `BatchBuilder.convert` (:385) → `split_by_dp` (:390). The SGLang router is a subprocess
  (`slime/backends/sglang_utils/deployment.py:18-62`). `ServingCluster` (`slime/ray/serving.py:35`) is a **named,
  detached** Ray actor that owns routers, engines and placement groups, and outlives a trainer crash.
- **Data buffer.** `DataSource.get_samples/add_samples` (`slime/data/data_source.py:17-55`). `RolloutDataSourceWithBuffer`
  (:191) offers `pop_first`, `pop_oldest` (`--buffer-sort-by-staleness`, :248-262) or `--buffer-filter-path` (:198).
  With `--rollout-data-transport straw` it uses a persistent lease-based `QueueDataSource`
  (`slime/data/queue_data_source.py:49,1292`): priority ready=2 > partial=1 > fresh=0, keyed by oldest weight version
  (:1176-1193).
- **Weight push** (`slime/backends/megatron_utils/update_weight/__init__.py:18-56`): delta mode → disk delta; disk
  transport → disk; colocate → CUDA IPC; otherwise NCCL broadcast. NCCL flow (`update_weight_from_distributed.py:103-143`):
  `weight_version += 1` → `pause_generation` (`abort` if the cache is flushed, else `in_place`) → buckets →
  `continue_generation`. `should_flush_cache` (`slime/utils/weight_sync.py:4-13`) flushes every
  `--flush-cache-interval` syncs; any interval ≠1 is a PipelineRL-style in-place pause.

### 1.2 Sync rollout, over-sampling, dynamic filtering, partial rollouts
- **`generate_rollout`** (`slime/rollout/sglang_rollout.py:772-793`) runs `generate_rollout_async`, then returns aborted
  partials to the buffer via `data_source.add_samples` (:790-792).
- **Over-sampling + dynamic filter.** When `remaining < rollout_batch_size` it pulls `over_sampling_batch_size` groups
  (:530-537); `call_dynamic_filter` drops groups and each drop decrements `remaining` (:555-571); it stops at
  `rollout_batch_size` and calls `abort()` (:589). Built-ins: `check_reward_nonzero_std` (DAPO) and `_with_fallback`
  (`slime/rollout/filter_hub/dynamic_sampling_filters.py:9-25`). **Agent-aware:** a trajectory is a `list[Sample]` of
  segments and its terminal reward counts once (:10-12; `filter_hub/base_types.py:33`).
- **Partial rollout.** `abort()` collects ABORTED samples with output and tags `metadata.start_rollout_id` (:473-484);
  `generate()` resumes with `max_new_tokens - response_length` (:173). `--mask-offpolicy-in-partial-rollout` zeroes the
  old prefix's loss mask, with no IS reweighting (:294-297).

### 1.3 Fully async
- **Entry.** `--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async`
  (`fully_async_rollout.py:381-389`; no eval). `AsyncRolloutWorker` (:82) is a background thread + loop with
  concurrency `server_concurrency × engines / n_samples × prefetch_factor` (:86-90). The only backpressure is
  `output_queue.qsize()` (:178-244). ABORTED/PENDING groups are requeued (:246-275); "interrupted segments cannot reopen
  their closed agent sessions", so inputs are deep-copied first (:247-249). Multi-node: `RolloutScheduler` /
  `DistributedRollout` (`fully_async_distributed.py:36,484`).
- **Staleness is measured, never bounded.** `sample_staleness = current − min(weight_versions)`
  (`slime/utils/staleness.py:10-29`) is only logged; a grep for `max_staleness|staleness_threshold|max_off_policy` finds
  nothing. Staleness only orders work (`pop_oldest`, the straw key). One response can span several weight versions
  (`weight_versions` appended per chunk, `slime/utils/types.py:493-494`).
- **Off-policy correction.** `--use-rollout-logprobs` (`arguments.py:1174`); `--use-tis` (clip 2, clip-low 0,
  :1184-1203; `vanilla_tis_function`, `loss.py:904-925`); ICE-POP mask (`loss.py:928-949`); `--custom-tis-function-path`
  (:1205; MIS at token/sequence/geometric level, `examples/train_infer_mismatch_helper/mis.py:150`); OPSM (:1236-1245);
  score centering (`slime/utils/score_centering.py`, `arguments.py:949`). `train_rollout_logprob_abs_diff` is always
  logged (`loss.py:1345-1363`). `generate_streaming` (`slime/rollout/sglang_streaming_rollout.py:48`) keeps partial
  state on the sample when an abort fires.

### 1.4 Reward hub, sample hooks, every extension point
- **Reward hub.** `async_rm` (`slime/rollout/rm_hub/__init__.py:55-96`) resolves per-sample `custom_rm_path` →
  `--custom-rm-path` → `metadata.rm_type`/`--rm-type` (`remote_rm`, `deepscaler`, `dapo`, `math`, `f1`, `gpqa`,
  `ifbench`, `random`; `boxed_` prefix). `batched_async_rm` (:99-110) serves `--group-rm`.
- **Sample hooks.** `--rollout-sample-hook-path` (repeatable) `hook(args, sample, *, rollout_id, evaluation)` runs after
  generate, before reward (`slime/rollout/sample_hooks.py:37-64`; `sglang_rollout.py:356`).

| Hook flag (arguments.py line) | Signature | Called at |
|---|---|---|
| `--rollout-function-path` :377 | `(args, rollout_id, data_source, evaluation=False)` | ray/rollout.py:127 |
| `--custom-generate-function-path` :527 | `async (args, sample, sampling_params[, evaluation]) -> Sample \| list[Sample]` | sglang_rollout.py:340 |
| `--custom-rm-path` :1487 | `async (args, sample)->float` (list in batch mode) | rm_hub/__init__.py:62,106 |
| `--dynamic-sampling-filter-path` :493 | `(args, samples) -> DynamicFilterOutput\|bool` | sglang_rollout.py:515; fully_async_rollout.py:298 |
| `--rollout-sample-filter-path` :1555 | `(args, groups)`; sets `remove_sample` | base_types.py:73 |
| `--custom-reward-post-process-path` :1497 | `(args, samples) -> (raw_rewards, rewards)` | batch_builder.py:211-212 |
| `--custom-convert-samples-to-train-data-path` :1505 | `(args, samples) -> dict` | batch_builder.py:263-264 |
| `--rollout-data-postprocess-path` :598 | `(args, rollout_id, rollout_data)` | megatron actor.py:533 |
| `--custom-advantage-function-path` :1091 | `(args, rollout_data)`; fills `advantages`, `returns` | loss.py:779-782 |
| `--custom-loss-function-path` :1048, `--custom-tis-function-path` :1205, `--custom-pg-loss-reducer-function-path` :1211 | loss side | loss.py:1547,1274,1305 |
| `--data-source-path` :732, `--buffer-filter-path` :569, `--eval-function-path` :879 | data side | rollout.py:87; data_source.py:198 |

All hooks load by dotted path through `load_function` (`slime/utils/misc.py:50-60`). Advantage estimators: grpo, gspo,
cispo, reinforce_plus_plus(_baseline), ppo (`arguments.py:1064-1080`). Group normalization is rollout-side
(`batch_builder.py:210-257`). grpo broadcasts one scalar to every token (`loss.py:784-788`; `ppo_utils.py:533-540`).
The only token-level rewards are PPO's KL shaping plus the terminal scalar (`loss.py:790-803`). **There is no native
per-turn or per-token task reward.**

---

## 2. The agentic path (`slime/agent/`, `examples/coding_agent_rl`)

### 2.1 Layers
Four layers (`examples/coding_agent_rl/generate.py:5-15`): sandbox contract (`slime/agent/sandbox.py`), harness
(`slime/agent/harness/`), task layer (`examples/coding_agent_rl/swe.py`), protocol adapter + TrajectoryManager
(`slime/agent/adapters/`, `slime/agent/trajectory.py`).

**Sandbox contract** (`sandbox.py:32-73`). The `Protocol`: `sandbox_id`; `async __aenter__/__aexit__`;
`exec(cmd, *, user="root", env, timeout=120, check, idempotent=True) -> (rc, out, err)`;
`write_file(path, str|bytes|Path, *, user)`; `read_file(path, *, user) -> str` (`""` if missing; lifecycle and
permission errors propagate). Optional `emulates_root` (:44-46); `SandboxTerminatedError` (:28) means the sandbox died.
- **`exec_and_wait`** (:99-162) runs long commands `setsid`-detached with output to a file, an exit-code marker, a
  `mkdir` lock that dedupes a replayed spawn, and a 5 s `read_file` poll (:79-96). Every RPC is short and idempotent, so
  gateway stream cuts are survivable. Timeout returns `EXIT_TIME_BUDGET_EXCEEDED=-1` (:76).
- Providers: `E2BSandbox` (transient-error classes, jittered backoff, pool reset; :177-406) and `SunabakoSandbox` (PRoot
  rootfs over SSH, per-sandbox memory/PID caps; `sunabako_sandbox.py:20-84`, `emulates_root` :30). Selected by
  `create_sandbox` (`examples/coding_agent_rl/sandbox.py:8-16`).

**Harness** (`slime/agent/harness/common.py`). `BaseHarness` abstracts `install_cli`, `write_config`,
`launch_and_wait(sb, ctx, prompt, budget) -> exit_code` (:63-79). `run()` = `ensure_agent_user` → `write_config` →
`launch_and_wait` (:81-104). `HarnessContext(workdir, session_id, adapter_url, model_label)` has no task fields
(:42-54). `run_agent` runs the CLI as user `agent`, stdout to `{workdir}/.harness/trajectory.jsonl` (:111-126).
- **Claude Code** (`claude_code.py`): `claude -p <prompt> --permission-mode bypassPermissions --output-format
  stream-json …` (:26-30, :87). `ANTHROPIC_BASE_URL=adapter_url`, `ANTHROPIC_AUTH_TOKEN=session_id`: **the session id is
  the API key** (:91-96). Non-essential traffic off (:32-36). PreCompact/PostCompact hooks in `settings.json` (:57-84);
  the CLI PID is recorded for the hook (:103-108).
- **Codex** (`codex.py`): `codex exec --dangerously-bypass-approvals-and-sandbox --json`, because the outer sandbox owns
  isolation and nested bwrap cannot run in PRoot (:36-40). `config.toml` sets the provider `base_url` to the adapter
  with `wire_api="responses"` (:45-60), and `model_context_window` / `model_auto_compact_token_limit` from env (:90-100).
  Hooks go to `hooks.json` with stop mode `json` (:111-124).
- **Compaction budget** (`harness/compaction.py`, stdlib, copied into the sandbox). PreCompact increments `attempts`
  under `flock`; at `attempts >= limit` it sets `limit_reached` (:37-49) and either prints `{"continue":false}` (Codex)
  or SIGTERMs the CLI after verifying it is a true ancestor through `/proc` host PIDs (:50-67). PostCompact increments
  `count`; `collect_metrics` reads these (`claude_code.py:111-120`). 70% comes from `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=70`
  and Codex `COMPACT_TOKEN_LIMIT = 0.7·ctx` (`run_qwen38_27b_fully_async.sh:49-53,60-70`). Limit 2 per CLI; a third
  attempt stops the agent, which is graded on its existing patch (`README.md:226-227`).

**Adapters** (`adapters/common.py`, `anthropic.py`, `openai.py`, `responses.py`). An aiohttp app in a thread exposing
`/v1/messages` (+ `count_tokens`, which returns 0; `anthropic.py:300-304`), `/v1/chat/completions` and `/v1/responses`
(rejects `previous_response_id`/`store`; `responses.py:182-187`). The sid is the bearer token (`anthropic.py:218`;
`openai.py:291`).
- `_run_turn` (`common.py:387-476`): translate → render with the **served chat template** → reuse the original model
  tokens of a replayed assistant prefix (`_session_prompt_ids`, :93-136; a changed or compacted history is a separate
  branch) → `call_sglang_generate` posts `input_ids` to `/generate` with `return_logprob` (:423, :524-700) → parse with
  the SGLang tool/reasoning parsers → **flush the reply, then `record_turn`** (:441-469), so an undelivered turn is never
  recorded.
- Hard validations: logprobs present, counts match, top-p nucleus ids present, R3 routing present (:637-660). The
  context budget is clamped by speculative-decoding draft slots (:544-568).
- **Partial rollout inside an agent turn:** an SGLang `abort` during a weight update is continued in a loop on the same
  token prefix (:600-680), so the turn's `weight_versions` spans versions.
- Session affinity: `X-SMG-Routing-Key: sid` (:572) with `--router-policy consistent_hashing`. `max_turns_per_sid`
  returns 429 (:354-375).

### 2.2 TrajectoryManager: multi-turn tool trajectory with compaction → training samples
File: `slime/agent/trajectory.py`.
- **`TurnRecord`** (:40-54) is the adapter↔manager contract: `prompt_ids`, `output_ids`, `finish_reason`,
  `output_log_probs`, `ill_formed`, replay (top-p, top-k, routed experts), `weight_versions`.
- **Per-sid message tree** (:62-124). `record_turn` (:381-403): `_find_mount_point` matches messages by role and dict
  equality (:450-466) → `_try_merge_assistant_rewrite` demotes a re-rendered short assistant to routing-only instead of
  forking (:468-524) → mount remaining prompt messages → attach a generated assistant leaf carrying the `TurnRecord`.
- **Sub-agents and compaction need no special code:** "any divergence in the prompt prefix forks into a new leaf"
  (`anthropic.py:8-10`). The compaction request (history + summarize) hangs under the old branch and its summary is a
  generated turn; the post-compaction context (system + summary …) starts a new subtree.
- **Linearization**, `get_trajectory` (:405-442): one root→leaf chain per leaf. `_split_chain_into_builders` (:554-575)
  packs turns into `_SampleBuilder`s by token drift (:189-228):
  - **CLEAN**: prompt exactly extends held tokens; append the tail with `loss_mask=0`.
  - **REALIGN**: short drift inside the latest response, under `fork_threshold` (default 1024, :369); overwrite that
    span from the prompt with `loss_mask=0`, logprob 0 (:293-301).
  - **FORK**: anything else, any R3 replay, or mismatched replay keys (:203-214) → new builder, i.e. a new `Sample`.
- **Loss masks.** Model outputs get `loss_mask=1` with sampled logprobs; tool, observation and context tokens get 0
  (:240-250). A turn shared by sibling leaves is trained **once**, by the first leaf (`response_trained`, :96-98,
  :567-568). `to_sample` strips the first-turn prompt and truncates to `max_context_tokens` (:311-341). Builders with no
  trained token are dropped (:596-600). Metadata: `truncated`, `use_tool`, `ill_formed` (:586-595).
- **Reward is outcome only:** "`reward` is assigned in full to every emitted Sample (not split across them)"
  (:417-419, :437-438).
- **Group statistics over unique trajectories.** `BatchBuilder._post_process_rewards` groups segments by `rollout_id`
  and **raises** if they disagree on group or reward (`slime/data/batch_builder.py:219-235`); it normalizes over
  distinct trajectories and copies each advantage to all segments (:236-255). `rollout_mask_sums` gives a
  per-trajectory token-weighted denominator across micro-batches (:308-323). `dp_schedule` keeps a rollout's segments in
  one optimizer step (`slime/utils/dp_schedule.py:1-20`).
- **Consequences.** (a) The compaction summary turn *is trained*, with the outcome reward. (b) **No step credit**: every
  trained token gets the same group-normalized scalar. (c) TITO is exact; sampled tokens are never rebuilt from text
  (`README.md:277-280`). Drift costs a new segment, never wrong tokens.

### 2.3 `examples/coding_agent_rl`: one rollout, end to end
`generate()` (`generate.py:225-413`):
1. Pick `(harness, adapter)` from `sample.metadata.agent` or `SWE_AGENT` (:50-64).
2. `adapter.open_session(sid, sampling_defaults, max_context_tokens)` (:269-279); sid
   `cagent-{instance}-{index}-{group}` (:438-443).
3. Everything runs under `asyncio.timeout(rollout_guard_sec)` (:282).
4. `boot_agent_sandbox` (:110-164): sandbox from the task image, CLI installed offline from host tarballs,
   `SWE_BOOT_RETRIES=2` with jitter, optional boot semaphore.
5. `swe.prepare_workspace` (`swe.py:187-209`): agent user, pre-commands, `PROBLEM_STATEMENT.md`, and a **baseline git
   tree** from a temp index that captures untracked files and excludes `PROBLEM_STATEMENT.md` and `.harness` (:241-262).
6. `harness.run(...)` with `agent_time_budget_sec` → `collect_metrics` → `swe.git_diff(baseline, current)` (:265-272).
7. **Grading in a fresh sandbox from the same image** (`swe.py:279-340`): apply the diff (`git apply` → `--3way` →
   `patch -p1`, :352-366), then `swepro` (F2P∪P2P ⊆ passed, :369-391), an `eval_cmd` (exit 0) or an `f2p_script`.
   **Reward ∈ {0,1}** (:415-428). For MiMo, `prepare_mimo.evaluation_command` **resets test-patch paths to HEAD**, then
   applies the hidden test patch (`prepare_mimo.py:18-40`); hidden tests never enter the agent sandbox (:3-5).
8. Train path: `adapter.finish_session(sid, reward=…)` returns segments (:325-336).
   - `compact_limit_reached` or `EXIT_TIME_BUDGET_EXCEEDED` → `TRUNCATED` but **trained with the graded reward**
     (:340-349).
   - `SandboxTerminatedError` → **trained with reward 0** ("an observed trajectory outcome", :369-399).
   - Wall-clock timeout or other exception → `_abort_result` (`remove_sample=True`, `loss_mask=[0]`; :400-410,
     :446-463). Eval returns a reward-only placeholder (:466-491).
9. A per-worker trace shard records prepare, model-request, agent, patch and grading spans (:225-251).

**Scale** (`run_qwen38_27b_fully_async.sh`; `README.md:14-24,216-224`): Qwen3.8-27B, training TP4·PP3·CP2 on 24 GPUs,
12 SGLang engines at TP2 on 24 GPUs, two sandbox nodes outside Ray. `SERVER_CONCURRENCY=14 × 12 = 168` slots;
`GROUP_PREFETCH_FACTOR=2` → 42 prefetched groups. Batches are 8 prompts × 8; a group trains only when all 8 finish with
nonuniform rewards (sh:168). `--partial-rollout --flush-cache-interval 3` (sh:176); `--use-tis`, score centering,
`--pg-loss-type reinforce --disable-grpo-std-normalization`, KL 0, stateless Adam, LR 1e-6 (sh:175-188). Timeouts:
agent 3600 s, grader 600 s, whole rollout 7200 s.
- **Doc/code drift:** the README says context/output 131,072 / 8,192 (`README.md:22`); script defaults are 262144 /
  131072 (sh:23-24).
- **The sandbox must reach the adapter over HTTP** (`README.md:43-44`; `generate.py:175-180`): isolation is "no egress
  except the adapter", not "no network".
- "Do not modify tests" (sh:58) is **enforced only for test-patch paths** (`prepare_mimo.py:29-33`); an edit to an
  untouched `conftest.py` still applies in grading.

### 2.4 Other agentic examples
- **multi_agent** (`examples/multi_agent/agent_system.py:198-290`): solver, rewriter and selector each get their own
  `batched_async_rm` reward, scaled ×1.2/×0.8 by whether the final selection was right. **Role-level credit**, the only
  non-outcome credit in the repo.
- **tau-bench** (`examples/tau-bench/trainable_agents.py:177-336`): masks by **re-tokenizing** chat-template deltas
  (`token_delta.py:4-30`); env final reward. The pre-adapter, non-TITO pattern.
- **retool**, **search-r1**: hand-rolled loops; `loss_mask` 1 for model tokens, 0 plus dummy logprobs for tool output
  (`search-r1/generate_with_search.py:179-272`). Outcome rewards; retool adds a tool-call shaping term when wrong
  (`retool/generate_with_retool.py:407-430`).

---

## 3. Reliability features
- **Engine recovery is automatic.** `RolloutHealthMonitor` probes `health_generate` and retires dead engines
  (`slime/utils/health_monitor.py:126-153`); they restart before the next weight update (`slime/ray/serving.py:256-266`;
  `actor.py:633-638`). `--use-fault-tolerance` is a compatibility flag (`arguments.py:675-682`).
- **Trainer recovery is semi-manual:** resubmit `train.py` against the live detached `ServingCluster`
  (`--rollout-session-id`, `arguments.py:669`).
  - Journal (`slime/ray/training_recovery.py`) written with tmp + fsync + rename + dir fsync (:262-293).
    `remember_raw` stores a batch with its data cursor (:359-371); `remember_converted` stores it before DP sharding, so
    a new layout re-shards without re-running rewards (:373-390). Batches release only on `checkpoint_committed`
    (:424-437); reconcile paths at :295-333.
  - Requires torch_dist, RNG and optimizer state, frozen rollout/hook config (:29-150); parallelism may change. Losing
    the ServingCluster or Ray cluster is not recovered (`docs/en/advanced/fault-tolerance.md:121-129`).
  - Straw `commit_checkpoint` checksums payloads and writes immutable `rollout/committed_{id}.json`
    (`slime/data/checkpoint.py:77-155`). Data-source offset, epoch, buffer and consumer state are saved
    (`data_source.py:137-183`).
- **Reproducibility.** Bitwise with `--sglang-enable-deterministic-inference`, flashinfer, `--deterministic-mode`
  (`initialize.py:96-101`), `NCCL_ALGO=Ring`, `CUBLAS_WORKSPACE_CONFIG` (`docs/en/advanced/reproducibility.md:3-26`).
  Per-sample `sampling_seed = rollout_seed + i` (`sglang_rollout.py:113-115`). GLM-5 train/rollout logprob alignment is
  gated at `<1e-6` (`arguments.py:1656-1662`).
- **Debug modes.** `--debug-rollout-only`, `--debug-train-only`, `--save-debug-rollout-data {rollout_id}`,
  `--load-debug-rollout-data` (forces train-only, skips SGLang; `arguments.py:1381-1427, 2093-2098`). `sleep_rollout.py`
  for engine stress tests. `check_rollout.py` runs real coding rollouts and graders against a running SGLang server with
  no trainers (`examples/coding_agent_rl/check_rollout.py:1-5`). Dumps hold all of `Sample.__dict__`
  (`types.py:246-251`).
- **Train–infer mismatch helper** (`examples/train_infer_mismatch_helper/`): standard PPO, bypass, decoupled 3-policy
  IS; `mis.py` gives TIS, rejection sampling and a veto, and logs `mismatch_kl`/`k3_kl` (:383-430). Routing replay / R3:
  `--use-routing-replay`, `--use-rollout-routing-replay` (`arguments.py:1218-1230`;
  `slime/utils/routing_replay.py:83,173-205`).
- **Delta weight sync** (`update_weight_from_disk_delta.py:33`): pinned-CPU snapshot seeded by a full HF publish
  (:119-131); per-tensor XOR or overwrite, zstd, `weight_v{N}` safetensors dir (:144-267); engines `/pull_weights` and
  verify checksums. Disk transport only, not with colocate. XOR applies exactly once; overwrite is idempotent
  (`docs/en/advanced/delta-weight-sync.md`).

---

## 4. Against Crucible
Sources: `crucible/DESIGN.md` §6.2–6.4 (:455-540), §7.6 M17 (:698-750), §10.1 (:886-930), §10.3 (:1044-1063),
§10.4–10.5 (:2711-2760); `crucible/src/crucible/train/credit.py`; `crucible/src/crucible/runtime/harness/{workspace,native}.py`;
`crucible/src/crucible/factory/{checkpoint,honesty}.py`; `crucible/src/crucible/factory/lanes/windows.py`.

### 4.1 What slime has that Crucible should take

| slime piece | Verdict | Why / how |
|---|---|---|
| Megatron trainer + SGLang serving + fully-async loop at 27B–GLM scale | **DEPEND** (pinned commit; new `train/bridges/slime_bridge.py`) | Crucible has no Megatron backend; its FSDP trainer (`train/trainer.py:932`) won't reach 27B+ at 128k. slime owns `train.py`, so Crucible plugs in through hooks (§5); the only source change is one field list (§4.3). |
| `Sandbox` Protocol + `exec_and_wait` + idempotent-retry discipline (`sandbox.py:32-162`) | **PORT** into `runtime/harness/` as the provider seam under `WorkspaceHarness` | Four methods; our Docker World becomes one provider, E2B and Sunabako come free. Detached spawn + marker is the right answer for 1-hour test runs. |
| CLI harnesses (`claude_code.py`, `codex.py`, `common.py`) + `compaction.py` budget hook | **PORT** | Exactly the "harnesses vary across groups" contract (DESIGN:1062): counted, bounded compaction with a verified-ancestor kill. M16 `context.py` stays for the RLM actor. |
| Adapters + `TrajectoryManager` | **PORT, then extend** | Exact TITO, message tree, CLEAN/REALIGN/FORK, train-each-response-once and flush-before-record satisfy DESIGN contract 1 (TITO + per-span version) and 7 (compaction = segments). **Needed:** builders must export per-turn token spans (`last_response_start_idx` exists, :182, but is not emitted) so per-call credit maps to tokens. |
| Segment → trajectory normalization (`batch_builder.py:219-255`), `rollout_mask_sums`, same-step DP scheduling | **PATTERN** | Our GRPO must normalize over unique rollouts when one rollout gives N segments. Check `train/batching.py` for the same invariant. |
| Fresh-sandbox grading of `git diff` against a temp-index baseline (`swe.py:241-272`) | **PATTERN** (already ours: `WindowTask.grade_patch` on a pristine A tree, `snapshot_patch` windows.py:1218) | Confirms our design; copy the `--3way` / `patch -p1` apply ladder. |
| Health monitor, detached ServingCluster, recovery journal, debug dump/replay, `check_rollout.py` | **DEPEND** (they come with slime) / **PATTERN** for Crucible's runtime | `check_rollout.py` is the model for a "rollouts without trainers" smoke test of a Crucible task. |
| R3, score centering, delta weight sync, MIS helper | **DEPEND** | Our `train/mismatch.py` menu already names TIS/MIS; under slime use its flags. |

### 4.2 Where Crucible is ahead, and where slime conflicts with our rules
1. **Step credit.** slime is outcome only (`trajectory.py:417-419`; `loss.py:784-788`) plus multi_agent role scaling.
   Crucible has milestone Φ with a prerequisite product (DESIGN:455), ΔΦ in `progress.jsonl`, and `turn_credit_map`
   centred and bounded by the outcome advantage (`credit.py:52-84`). The M17 sidecar grades snapshots at commits, every
   K calls and compaction boundaries; a lost milestone counts negative (DESIGN:714).
2. **Honesty.** slime has none. Crucible Brier-scores STATUS.json, supports `GIVE_UP`, and `compose_advantage` never makes
   a failure positive (`factory/honesty.py:156`).
3. **Measured verifiers.** slime trusts dataset tests. Crucible measures them: oracle reruns with a flake bound,
   commit-revert mutants, leak probes, hacker episodes, and a prompt ledger + lint refusing unenforced instructions
   (DESIGN:704-712). slime's "do not modify tests" is enforced only for test-patch paths (§2.3).
4. **Infra blame.** slime trains `SandboxTerminatedError` trajectories at reward 0 (`generate.py:369-399`). Crucible
   masks every non-policy failure (DESIGN:917). **Under slime, our generate must return removed samples
   (`_abort_result`-style) for masked failures.**
5. **Staleness bound.** slime only logs it (`staleness.py:10-29`); Crucible contract 3 drops above τ. Implement as a
   `--dynamic-sampling-filter-path` (drop groups with too-old `min(weight_versions)`) or a
   `--rollout-sample-filter-path` setting `remove_sample`.
6. **Truncation.** slime trains `TRUNCATED` runs with the graded reward (`generate.py:340-349`); Crucible says
   truncation is not failure (mask or Φ-bootstrap, DESIGN:1061). Our generate decides.
7. **Single-rollout critic mode** (DESIGN:1060): slime's `ppo` has a critic (`loss.py:790-803`), so it is usable.

### 4.3 Step credit through slime: three hooks and one patch
- **(a) Generate** records per-turn ΔΦ in `sample.metadata["turn_credit"]` with token spans from the ported
  TrajectoryManager.
- **(b) Honesty** via `--custom-reward-post-process-path`: it replaces default normalization and composes Â + Brier
  term with the failure floor per trajectory, returning `(raw, adv)` that grpo broadcasts.
- **(c) Per-token advantage:** `--custom-convert-samples-to-train-data-path` adds a per-token `turn_advantages` key;
  `--custom-advantage-function-path` sets `advantages[i] = A_seq + η·turn_adv` (`loss.py:779-782`).
- **The patch.** The DP split forwards only a **fixed field whitelist** (`slime/data/batch_builder.py:467-487`), so
  unknown keys are dropped. One line there (or an upstream PR) is the only source change. No-patch fallback: one Sample
  per turn with the prefix as context and a per-sample reward; correct but quadratic in tokens.
- **Where to snapshot.** While the adapter handles a turn, the CLI is blocked on the model, so a snapshot in an adapter
  subclass's `_run_turn` *before* `call_sglang_generate` (`common.py:423`) sees a quiescent workspace (except background
  jobs the agent left running). `debug_callback` (:377) fires after the reply is flushed: too late. Compaction is
  visible both to the PreCompact hook and as a FORK in the tree. This replaces the World-step `on_step` mapping:
  `credit_from_progress` assumes one World step per action (`runtime/record.py:33-45`), and a CLI agent's tool calls are
  invisible to us.

---

## 5. The interface a Crucible task must expose to train under slime
1. **Data row** (JSONL; `--input-key prompt --label-key label --metadata-key metadata`, sh:169). `prompt`: the window
   prompt + `EPILOGUE` (windows.py:755). `metadata`: `instance_id` (= task_id), `image` (World image at A), `workdir`,
   `agent` (`claude_code`|`codex`, one CLI per group), `crucible_task_ref` (loads the `WindowTask` record: milestones,
   requires, weights, checkset), `split=train`. Held-out windows never appear in this file (sealing).
2. **Sandbox**: a provider implementing `slime.agent.sandbox.Sandbox` (`exec`, `write_file`, `read_file`, async context,
   `sandbox_id`), raising `SandboxTerminatedError` on infra death, with egress only to `adapter_url`. Crucible's
   `task.open(workdir)` World becomes this provider.
3. **Generate**: `async generate(args, sample, sampling_params, evaluation=False) -> list[Sample]` via
   `--custom-generate-function-path`. Steps: open session → `task.open` → sandbox, record base SHA → harness under
   budget with checkpoint snapshots (§4.3) → `task.snapshot_patch` → `task.grade_patch(patch)` **in a fresh container**
   → `R = 0.5·Φ_final + 0.5·[all gates pass]` (windows.py:30-35) →
   `finish_session(reward=R, extra_metadata={turn_credit, status_claims, give_up, blame})`. Contract: every segment keeps
   `group_index`, `index`, `rollout_id`; a masked or infra failure returns `remove_sample=True`, `loss_mask=[0]`.
4. **Filters and post-processing**: `check_reward_nonzero_std` (per trajectory); a staleness-τ sample filter; honesty in
   `--custom-reward-post-process-path`; turn credit via the convert + advantage hooks plus the whitelist patch (§4.3).
5. **Run flags** (sh:165-188): `--rollout-function-path …fully_async_rollout.generate_rollout_fully_async`,
   `--partial-rollout`, `--use-tis`, `--router-policy consistent_hashing`, `--rollout-max-context-len`, and compaction
   env knobs `SLIME_AGENT_{CC,CODEX}_MAX_COMPACTIONS`, `*_COMPACT_PCT` / `*_COMPACT_TOKEN_LIMIT`.
