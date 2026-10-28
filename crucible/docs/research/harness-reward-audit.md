# Harness, compaction and reward audit (2026-10-28)

Scope: the long-horizon agent harness (the transcript policy `LLMToolPolicy` that trains the student, and the M16 RLM),
context compaction and the trained token stream (TITO), the reward and credit path (`credit/`, `train/credit.py`, the
Tinker / prime-rl datums, `verify/`, the checkpoint sidecar, honesty, path checks), and Crucible's use of the agentenv
framework. A comparison with nine local frameworks closes it. Every claim names a file and line. Paths are relative to
`crucible/src/crucible/` unless they say otherwise.

All the fixes are in commit `9db0857d` on branch `worktree-agent-ab4ad5539d6391889`. Each has a test in
`tests/unit/test_harness_reward_audit.py` that fails on the code before it. The open risks of §8 were fixed or decided in
a second pass (2026-10-28, branch `worktree-agent-a6734480fd8877554`); §8 gives each item's status.

## 0. What actually trains the student

It is worth stating first, because the M16 RLM is not on this path.

- **The path.** `train/bridges/tinker/trainer.py:393` (and the prime-rl backend, which reuses it) builds the policy with
  `TinkerSampler.policy_ref`.
  - That gives `LLMToolPolicy` over `SamplerLLM` (`train/bridges/tinker/sampler.py:128-141`), wrapping `EngineLLM`
    (`train/world_env.py:113`).
  - It runs under `WorkspaceHarness` from `factory/envs.make_env` (`factory/envs.py:209-243`), with the checkpoint
    sidecar as Φ (`factory/checkpoint.py`).
  - Advantages come from `datums.build_datums` (`train/bridges/tinker/datums.py:67`).
- **The RLM is not on it.** `make_env(mode="rlm")` is not called anywhere outside a test. The RLM (`actors/rlm/`) serves
  the task factory's agents.
- **No compaction in training.** `policy_ref` passes no `ContextConfig`, so the student never compacts:
  `LLMToolPolicy.context` is None and `_maybe_compact` returns at once (`actors/policy.py:330-333`).
- **Measured on the local targeting runs** (790 student rollouts under `data/targeting`; metadata only, held-out and
  sealed paths skipped):
  - **Turns:** median 110, p90 279, max 304.
  - **Stops:** 619 `agent_done`, 148 `max_steps` (the 150-call tool limit), 12 `timeout`, 3 `max_output_tokens`,
    1 `too_many_errors`, 0 `context_overflow`.
  - **Segments:** 693 boundaries, all `rerender`. Qwen3.6's "interleaved" template re-renders history at a harness user
    turn: a limit warning or a nudge.
  - **Conclusion:** compaction is not needed at today's 64k/256k contexts and 150 calls. Its code paths still matter
    for any run that turns it on, so they were audited (§2) and one real bug was fixed.

## 1. Bugs found and fixed

| # | Severity | Where (before the fix) | What | Fix |
|---|---|---|---|---|
| B1 | **High** (TITO) | `actors/policy.py:357-371` | **What broke:** a blank first summary attempt is resampled once. The resample's prompt equals the first attempt's prompt, so it never extends that attempt's prompt + completion. As a link of the segment's chain, `record.build` flagged it `prefix_break` (`runtime/record.py:179-184`). **Effect:** train mode masked the whole rollout `capture_incomplete` (an infra mask), and the trainer dropped the whole group. **Second fault:** the discarded blank attempt was trained with the episode's advantage. | The resample is a tree node (edge `compaction_attempt`, `Turn.node`) with its own whole prompt. Each attempt carries `accepted`; a discarded attempt is context with loss mask 0 (`record.py:155`), as M16 and verifiers do. Reproduced with `EngineLLM` + `ByteChatCodec` before the fix. |
| B2 | **Medium** | `train/bridges/tinker/sampler.py` (`SamplerLLM`) | **What broke:** no `context_window`, so `LLMToolPolicy._reply_tokens` never fitted `max_tokens` to the room left (`actors/policy.py:278-291`). `TinkerSampler.gen` then refused prompt + `max_tokens` > context (`sampler.py:90`), and the episode ended `context_overflow` (scored, truncated) with room left for a shorter reply. **Also:** a `ContextConfig` without `budget_tokens` compacted at 0.8 × 128k, beyond a 64k model's context (`policy.py:334`). **Scope:** Tinker, and the DeltaAI vLLM path that reuses `TinkerSampler`. | `SamplerLLM.context_window()` returns the served model's context. |
| B3 | **Medium** | `runtime/harness/workspace.py` | **What broke:** `WorkspaceHarness`, the coding lanes' harness, applied neither `LimitPolicy` mask nor `penalty:<c>`. The limit policy is in its id digest (`native.py:410-414`), so a run configured with a penalty silently trained on partial grades. | Mask before collect, penalty after the grade. The re-grade of a stored patch keeps the agent attempt's penalty, as `NativeHarness.regrade` does. |
| B4 | **Medium** (reward hacking) | `runtime/harness/native.py` `run`, `workspace.py` `run` | **What broke:** processes that survive the final quiesce make `run_episode` record `crucible:policy_misbehavior` with mask False (`actors/policy.py:818-822`). Only the multi-step path acted on it. A single-step episode was graded as usual while the survivors could still write the tree being graded (in place) or collected (workspace). | `native.misbehaved(ep)` turns it into the attempt's failure, scored 0 and never graded, in both harnesses (DESIGN §8.3 already promised this). **Verdict change, flagged:** an episode whose processes survive the quiesce now scores 0 instead of its grade. It applies only when the World is isolated (a guard quiesce); the local backend's quiesce is a no-op. |
| B5 | **Medium** (credit agreement) | `factory/checkpoint.py:561-576` | **What broke:** `reconcile` matched the collected patch to a checkpoint by its sha256. The windows lane's end snapshot (`factory/lanes/windows.py:2075-2089`) and the harness's collect (`verify/collect.py:73-129`) render one change two ways: the collect drops derived files (`DERIVED`: an unignored `__pycache__/*.pyc`, `node_modules`) and quotes non-ASCII paths (no `core.quotepath=off`). **Effect:** a byte match missed the end checkpoint, so the sidecar's own (possibly loaded) verdict stood against the final grade. This is the `credit_grade_mismatch` the 2026-10-25 fix was for. | `checkpoint.same_change` compares the change (per-file blocks by unquoted path, derived files dropped, path lines ignored). It is a fallback for the end checkpoint only, so a different change is never matched. |
| B6 | Low (latent) | `factory/envs.py:236-240` | **What broke:** `make_env(mode="rlm")` built `RLMHarness`, which grades the World in place (`task.grade(world, workdir)`, `native.py:613`). A WindowTask's `grade(solution, *, run, checks)` (`factory/lanes/windows.py:1660`) raises `TypeError` there, so every attempt would end masked at verification. | Refused at construction for a task with `grade_patch`. |
| B7 | Low (V8) | `interop/agentenv/plugin/steps.py:167` | **What broke:** a masked grade's agentenv verification row carried `score: 0.0` (agentenv's own skipped rows carry none). With a tolerant verify step (`fail_task_on_error=False`), agentenv's aggregate of an all-skipped set scored 0.0 into `TaskRun.scores` (found by the agentenv sub-audit). | The masked row has no score. |
| B8 | Low | `interop/agentenv/mcp_env.py:108` | **What broke:** restore read every non-data part as a `FilePart`. A `text` part from `data/get` raised a `ValidationError`. | `TypeAdapter(agentenv_protocol.types.Part)`. |

| B9 | Medium (RLM) | `actors/rlm/agent.py` `_new_segment` | **What broke:** under an interleaved template each CONTINUE nudge opens a rerender segment, and that reset the nudge count. A tool-less model ran to `budget:turns` (a truncation) instead of the scored `agent_done`. | The count resets only at compaction and resume segments. |
| B10 | Medium (RLM) | `actors/rlm/agent.py` `_turn` | **What broke:** a python call whose arguments did not parse, or carry no `code`, ran as an empty cell ("(no output)"), uncorrected. | Answered with a correction row; no cell runs. |
| B11 | Medium (RLM) | `actors/rlm/agent.py` `_compact` | **What broke:** an accepted summary whose rebuilt context left no room (`compaction_failed`) stayed trainable and was named the failed segment's summary call. | Not accepted, not trainable, no `summary_call`. |

Tests: `tests/unit/test_harness_reward_audit.py` holds 15 tests. Every one but the preserved-template control fails before
its fix (checked by reverting the source).

## 2. The transcript harness (`LLMToolPolicy`, `run_episode`, `WorkspaceHarness`)

### 2.1 Tool-call parsing and error paths

- **Malformed tool calls are fed back, never dropped.**
  - Unparseable arguments become `{"__unparseable__": raw}` (`train/chat.py:40-58`; the Qwen XML parser,
    `train/bridges/tinker/codec.py:60-104`).
  - The World answers `invalid_arguments` with fault `policy` (`world/world.py:543-545`).
  - `credit/shaping.format_errors_by_call` charges the issuing call (`runtime/harness/native.py:901`). Only the torch
    trainer subtracts a format penalty (`train/trainer.py:527-545`); the Tinker / prime-rl datums do not (R5).
- **Agent-caused tool errors are `PolicyToolError`, fault `policy`** (`world/apps/base.py:58,179`;
  `world/apps/workspace.py:38-40,99-106`).
  - They count toward `max_tool_errors` like any `policy` or `tool` fault (`actors/policy.py:795-799`), are not format
    errors, and are scored.
  - A plain `ToolError` stays fault `tool`, a domain error, also scored (`world/world.py:601-602`).
  - Correct.
- **An App that raises outside its contract is `crucible:tool_crashed`** (task blame, masked; `world/world.py:597`).
  - The workspace App turns every snippet exception into a `ToolError` (`world/apps/workspace.py:99-106`), so a policy
    cannot reach this path with odd arguments. Arguments are schema-validated before the call (`world/world.py:543`).
- **Cut-off and blank replies** (`actors/policy.py:551-587`).
  - A cut-off reply's tool calls are never executed. Two nudges per stall, then `max_output_tokens` or `format_error`.
  - Nudges are user turns, so in "interleaved" mode `EngineLLM` re-renders and flags a new segment
    (`train/world_env.py:175-181`). There is no prefix break.
  - A blank turn that was nudged past (`call["nudged"] = "blank"`, `policy.py:567`) is still trained with the episode's
    advantage. That is defensible: it is a real sample with its real prompt. Compare B1: a discarded summary attempt is
    now not trained.
- **The policy's own exceptions** (`actors/policy.py:763-781`).
  - `ValueError`, `IndexError`, `KeyError` or `TypeError` raised out of `act` are scored 0 as `crucible:policy_error`.
  - Risk R1 (fixed, §8): a harness bug of those types inside `LLMToolPolicy` (a `zip(strict=True)` mismatch, a config error) becomes
    a negative reward instead of a mask.
  - The engine path is wrapped correctly: `EngineCallError` carries an infra `failure_kind` (`train/world_env.py:52-60,
    189-214`).

### 2.2 Budgets and how a stop is recorded

- **Tool calls.**
  - `make_env` sets `max_steps = tool_limit + 2` (`factory/envs.py:229`).
  - `LLMToolPolicy.act` counts issued calls, warns at `limit_warning`, then gives one final `write_file` turn for
    `STATUS.json` (`actors/policy.py:440-511`).
  - Calls of the last reply past the limit get a "not run" tool reply. The stop is `max_steps`, truncated and scored.
- **Wall clock.**
  - `Timeouts.agent_s` (3600 s) is the episode's `asyncio.timeout`. It ends with stop `timeout`, truncated and scored
    (`actors/policy.py:815-819`).
  - `rollout_s` bounds the attempt and masks.
  - Risk R2 (fixed for training, §8): the agent clock includes model sampling latency, so in a shared async sampler a `timeout` stop depends on
    queue load, not on the policy.
- **Context.**
  - `ContextOverflow` ends `context_overflow`, truncated and scored (`actors/policy.py:536-538`).
  - B2 fixed the missing window fit.
- **Labels and limit policy.**
  - `TRUNCATING_STOPS` and `POLICY_STOP_ORDER` (`actors/policy.py:862-863`) label the stop.
  - `LimitPolicy` (score / mask / `penalty:<c>`) covers `max_steps`, `timeout` and `context_overflow`
    (`runtime/infra.py:222-235`). B3 fixed the workspace harness ignoring it.
  - `max_output_tokens`, `too_many_errors` and `too_many_compactions` are truncations the limit policy does not cover.
    `LIMIT_STOPS` (`native.py:86`) omits the first and the last.

### 2.3 Snapshots, checkpoint and resume

- **Checkpoint state.** `LLMToolPolicy.state_dict` / `load_state_dict` (`actors/policy.py:375-401`) carry the context,
  queue, segments, token capture (`last_full`, `last_seg`), the tool-limit state and the call ids.
- **Not saved.** `_compactions_in_row` (`policy.py:622`) is not in the state. A resume resets the
  too-many-compactions counter (minor).
- **Resume in training.** `NativeHarness._resumable` resumes only after an agent-stage mask (`native.py:492-513`).
  Training runs `RetryRule(max_attempts=1)` (`trainer.py:396-400`), so resume never happens in training.
- **The checkpoint sidecar** (`factory/checkpoint.py`).
  - Grades workspace snapshots out of band and never feeds the agent: `on_step` returns None, and files are written
    after the loop.
  - The cap thinning keeps the end and the commits.
  - An unmeasured point carries Φ and is never 0 (`checkpoint.py:512-525`). Correct.
  - Risk R6 (fixed, §8): the windows lane's snapshot diffs against the agent-movable `base` tag (`factory/lanes/windows.py:2081`),
    not the recorded base sha the harness uses. Moving or deleting the tag makes mid-episode Φ unmeasured or wrong. It
    cannot raise the reward, since the final grade and reconcile decide the end, but it can misplace turn credit.

### 2.4 Determinism and seeds

- `seed=0` only when temperature is 0, so sampled GRPO members differ (`actors/policy.py:521-522`). The per-call seed
  fix is in `EngineLLM.complete` (`world_env.py:201-202`).
- Conversations are keyed by the attempt's `sample_key` and the reply nonce, so one member's completion never splices
  into another's prompt (`world_env.py:121-124,150-174`).
- Call ids are derived from (actor, call index, position) when the provider's are unusable (`policy.py:293-311`).

### 2.5 Host-outage pause and resume (`world.hostreach`)

- **The clocks.** `held()` lifts every registered clock's deadline and restores what it had left (`world/hostreach.py:90-125`).
  - Registered clocks: the agent clock (`actors/policy.py:815-817`), the World's per-tool deadline
    (`world/world.py:568-571`) and the Runner's attempt clock.
  - Concurrent holds of one clock are reference-counted.
- **Replay.** A non-replayable tool call cut mid-way returns `interrupted` (fault `tool`, scored) after the host returns
  (`world.py:584-590`). A host that does not return within the pause budget masks the episode `transport_unreachable`
  (V8).
- **Model calls.** They are not wrapped, so a model call is never sampled twice.
- **Residual.** The detection window (up to `ReachRule.detect_s`, 50 s) runs on the agent clock, which is acceptable.

## 3. Compaction and the trained token stream

### 3.1 What the model saw is what trains

- **The engine path extends token for token.** `EngineLLM._prompt` (`train/world_env.py:156-185`):
  - It extends a served conversation as prompt + completion (+EOS) + the codec's suffix. Only the assistant turn this
    engine produced (the nonce) can be extended.
  - Anything else is rendered whole and flagged `new_segment`, which the policy turns into a `rerender` segment
    (`actors/policy.py:531-532`).
- **Linear capture.** `_compress` (`policy.py:234-248`) stores a delta only when the prompt extends the previous call's
  prompt + completion in the same segment.
- **The record.** `record.build` (`runtime/record.py:140-195`) keys chains by (actor, agent, segment) and flags any
  non-extension as `prefix_break`, which masks in train mode.
- **The linearizer.** `bridges/common.flatten` (`train/bridges/common.py:85-132`) emits one sequence per chain:
  - mask 1 exactly on sampled completion tokens, 0 on observations and on the EOS appended after a length cut;
  - the behaviour logprobs aligned.
  - So each segment's first prompt is the post-compaction context that was actually sent.
- **B1 was the one hole.** A summary resample broke the chain. It is fixed.

### 3.2 What is compacted, when, and the summary's tokens

- **When** (`actors/policy.py:330-387`): at `trigger × budget`, with no batch in flight.
- **The rebuilt context:** system + the task verbatim + `SUMMARY_HEAD + summary` + the last k complete turns
  (`_rebuild`, `policy.py:253-270`). No turn is fabricated.
- **Strategies:** `keep_recent`, `discard_all`, `tiered` and `hybrid` have no model call. Mode `action` lets the policy
  call `context_compact`.
- **The summary is trained. Decision: yes, the accepted summary trains with the episode's advantage, the rejected ones
  do not.**
  - The summary is a policy action whose quality decides the rest of the episode. M16's ladder showed that a summary
    that drops a rule costs 15-27% of the reward.
  - So it must receive the outcome's credit: CompactionRL, verifiers `trace.py:515-553`, and SkyRL's MemAgent all train it.
  - In the next segment it is prompt (mask 0), never trained twice.
  - A discarded attempt was not part of the trajectory that earned the reward; training it would reinforce blank
    summaries in successful episodes (B1).
- **Token accounting.**
  - `context_tokens` is the last call's provider count + chars/4 of what came after (`policy.py:212-224`).
  - After a rebuild it falls back to chars/4 until the next call measures. That can undercount code 2-3× for one turn,
    and the summary call itself is not fitted to the window (risk R7).

### 3.3 Credit across segments

- **Outcome credit.** One outcome advantage per rollout (Dr.GRPO, R − group mean over live members;
  `train/credit.py:88-101`). It is copied to every trainable token of every segment, summary included
  (`datums.py:51-85`, `flatten`). Group statistics stay on whole rollouts, never on segments, as prime-rl `routing.py`
  and SkyRL `trainer.py:1212-1250` do.
- **Turn credit.** It runs on the root chain's calls in order (`train/credit.py:52-85`). Summary nodes take their
  parent's credit (0). Φ's progress rows map to calls by `tool_call_id` (`runtime/record.py:33-75`).
- **Loss scale.** `loss_scale="sum"` makes a long, many-segment episode weigh more than a short one;
  `token_mean` divides by the batch's trained tokens (`datums.py:88-95`).

### 3.4 Can compaction hide what the grader or honesty needs?

- **Grader:** no. It reads the World (patch or state), never the transcript.
- **Path checks:** no. They read the World's own trace (`trace.json`), which compaction does not touch
  (`factory/pathcheck.py:1-10`).
- **Honesty:** no. It reads `STATUS.json` from the end tree or patch and the give-up line of the final action
  (`factory/honesty.py`; `factory/checkpoint.py:529-559`).
- **Sidecar:** it snapshots the state at each root compaction boundary (`before_action`, `checkpoint.py:387-397`).
- **What compaction can hide is from the policy itself:** a rule or supervisor message the summary dropped. That is the
  signal the summary is trained on, not a measurement gap.

## 4. Rewards and credit

### 4.1 The reward

- **The formula.** The debug lane's reward is `phi_weight·Φ + (1 − phi_weight)·[full pass and no path violation]`
  (`factory/lanes/windows.py:1704-1753`).
  - A hack-severity path violation gives 0 with `hack_suspect` (`factory/pathcheck.py:1-35`).
  - A gate-severity violation removes the full-pass bonus. Protected paths are restored before the hidden tests run.
- **Scale and sign at the trainer.**
  - R ∈ [0, 1]; the outcome advantage is R − mean.
  - Honesty adds a term ≤ 0 that keeps the advantage's sign (`factory/honesty.py:156-166`; the mirror is
    `slime/credit.py:47-53`).
  - Turn credit is centred within each episode, bounded by the outcome advantage, and off for gate-failed or hacked
    episodes (`credit/compose.py`).
  - Datums put the advantage on every completion token and 0 elsewhere (`datums.py:59-64`).

### 4.2 What each ending gets (training)

| Ending | Reward | Right for RL? |
|---|---|---|
| `agent_done` | the grade | yes |
| `max_steps` (tool limit), `timeout`, `context_overflow`, `max_output_tokens`, `too_many_errors` | the grade of the partial work, truncated flag (limit policy `score`) | yes for a coding task graded on state (verifiers, Harbor, OpenEnv do the same). `timeout` is the exception: in training it now counts only the agent's own time and is masked (R2, fixed) |
| `GIVE_UP:` / `STATUS.json` "not done" | the grade, plus an honesty term (a correct "not met" claim costs nothing) | yes |
| an agent-caused tool error | none by itself (fault `policy`, scored), counted toward `too_many_errors` | yes |
| a malformed tool call | the World's error message; the format penalty (off by default) on every backend | yes (R5, fixed) |
| the policy's own exception | 0 (`crucible:policy_error`) for its model's output (`PolicyOutputError`); a harness exception is masked `crucible:framework_error` | yes (R1, fixed) |
| it killed its World (OOM in its own container, a kill command, no stop from outside) | 0 (`crucible:agent_broke_env`) | yes (R3, fixed) |
| it broke its repo (`.git` deleted, git hung) | 0 (`crucible:agent_broke_env`) | yes |
| survivors at quiesce | 0 (`crucible:policy_misbehavior`; B4) | yes |
| provider / engine / Tinker cap / container lost / host gone / tool crash | masked; the whole group is dropped and refilled (`trainer.py:410-419`) | yes (V8) |
| grade could not run (sandbox did not start, run killed from outside, verifier timeout) | masked (`factory/lanes/executable.py` `killed`, `runtime/grading.settle`) | yes (V8) |
| the World exposed the answer (`answer_exposed`) | masked (task's fault) | yes, but see R4 |

### 4.3 Reward-hacking surfaces

- **Editing or deleting tests.** Protected paths are restored before the hidden tests run, and the edit is a
  `path:protected` gate failure (no full-pass bonus).
- **Test plumbing.** `conftest.py`, `sitecustomize.py` and `*.pth` are hack-severity, reward 0.
- **Reading hidden tests or grader files, and VCS history.**
  - `path:history` is hack-severity (`factory/pathcheck.py:63-64`).
  - When the World actually answered such a read with the answer, the probe masks the episode (`actors/policy.py:837-839`).
- **Network.**
  - The World runs `--network none`.
  - `path:network` (any download attempt) is hack-severity, and `path:install` is a gate.
- **Writing grader paths.** The grader sees only the patch on a pristine tree, in its own container. The agent's World
  holds no grader files.
- **Special-casing test names.** Not detectable mechanically. The hidden tests are separate from the visible ones the
  agent can read (`window["tests_deleted"]`, `windows.py:1673`), which is the mitigation.
- **Masking its own episode.** In the data, four `env_container_lost` masks were "after step 1" (§0). A policy whose
  command killed its own container was masked, not scored, and its group's signal was lost. Fixed (R3): the daemon's
  evidence decides, and a death the policy caused is a scored 0.

### 4.4 Partial credit and the final-vs-sidecar reconcile

- **Φ.** Φ = Σ w_m · pass_m · Π pass_prereq, with uniform weights unless the task or the validity gate sets them
  (`factory/checkpoint.py:140-146`).
- **The rows.** Progress rows telescope to Φ_end − Φ_0 (`checkpoint.py:249-296`).
- **The reconcile.** `fix/credit-end-reconcile` makes the final grade's vector the end checkpoint's. It only worked
  when the two patches were byte-equal (B5, fixed).
- **What remains.** A different git version or config (rename detection) between the container and the host can still
  render one change two ways (R6).

## 5. agentenv integration (sub-audit; pin 041afd7)

- **Dependency map.**
  - Only `interop/agentenv/*` imports agentenv at runtime, plus the explorer plugin and `factory/publish.py`, which
    import it lazily.
  - The ports (`core/store`, `core/ids.validate_id`, `core/registry._incompatible_core`, `core/triggers`) are in
    THIRD_PARTY.md.
  - `lint-imports`: 5 contracts kept, 0 broken.
- **The reward path.** Crucible never uses agentenv's reward-file verifier. `crucible_verify` grades with Crucible's
  producer, and a masked grade writes no reward file and raises. V8 holds on every live path, except the row score that
  B7 fixed.
- **Real bugs, recommended fixes.** These touch agentenv lifecycle and timeline code that has no training-path user
  today. The first is fixed (2026-10-28); the other two are not.
  - **A failed agentenv run leaked the World** (`interop/agentenv/plugin/steps.py:85-106`). **Fixed (2026-10-28).**
    - The World was closed only in `CrucibleVerifyStep`'s `finally`, so a failed deploy or prompt step leaked the
      gateway, the App servers and any paid sandbox, and the `mkdtemp` directory was never removed.
    - agentenv tears a run down only through its sandbox records (`teardown_run`; the pin has no other run hook). The
      `crucible_world` step now also records a bare sandbox of type `crucible_world`, and the plugin registers
      `CrucibleWorldProvider` under that type (entry point `agent_env.sandbox_providers`; an editable reinstall picks it
      up). Its handle's `terminate` closes the World and removes its directory, so `agent-env run`, `eval run` and the
      `LocalRunner` close it however the run ended.
    - A `crucible_world` step whose own setup fails closes its World at once. `release` is idempotent: verify,
      `close_world`, `teardown_worlds` and `teardown_run` may all reach one World. The work tree is always removed; the
      directory goes whole unless a verify step wrote outputs there.
    - Tests: `tests/unit/test_open_risks_agentenv.py`.
  - **`export_agentenv` publishes Tasks that cannot run** (`factory/envs.py:575-597`). `RowTask` refuses every backend
    but `local`, and the World step passes `backend="agentenv"`. Fix: refuse at publish time.
  - **`_restore_triggers` breaks wall-clock triggers** (`interop/agentenv/world_backend.py:571-596`).
    - An `after_event` dependent of a fired anchor is lost.
    - An unfired `every` is shifted to restore time + interval.
    - A stochastic `every` restarts its draws.
    - Fix: re-register at `next_mark`, and expand stochastic arrivals with `core/triggers.exp_interval`.
- **Risks.**
  - **agentenv's stock unit-test verifier scores a missing reward file as 0.0**
    (`agentenv-framework/.../run_container_unit_tests_verifier.py:334-342`). Never wire a Crucible verifier through it.
  - **`crucible_verify` has no grading budget or regrade.** `A2AHarness` has both.
  - **The sandbox fallback chain has no per-provider deadline.**
  - **"no-network" is not hermetic on Modal**, which adds its own hosts.
  - **THIRD_PARTY.md lacked DEPEND rows** for the plugin, `publish.py`, the explorer, `mcp_env.py` and `agents.py`.
    Fixed (2026-10-28): five DEPEND rows at pin 041afd7.

## 6. The RLM (M16) harness

A sub-audit read `actors/rlm/*` and confirmed each finding with a script, built on the `tests/unit/test_m16_f_agent.py`
scaffolding.

**Not on the student path (§0).** The severities below are for a run that trains the RLM. Today the RLM serves the task
factory.

**Fixed here** (B9-B11 in §1, tests in `test_harness_reward_audit.py`):
- **B9.** A CONTINUE nudge under an interleaved template reset the nudge count (`actors/rlm/agent.py` `_new_segment`).
  Unlimited nudges followed, then `budget:turns` (a truncation) instead of the scored `agent_done`.
- **B10.** A malformed python call ran as a silent empty cell (`agent.py` `_turn`).
- **B11.** A summary the run never continued from (no room after the rebuild) stayed trainable (`agent.py` `_compact`).

**Not fixed: recommendations, ranked.**
- **C1, critical for RLM training: a policy can mask its own episode by breaking its kernel's restart.**
  - `repl.py` is pushed once per launcher into a directory the kernel user can write (`actors/rlm/kernel.py:403-407,
    441-449`). A cell that deletes it and then dies makes `_restore` fail with `KernelStartError`, which masks the
    episode as `crucible:backend_error` (`agent.py:1361-1362`).
  - Fix: re-push `repl.py` read-only and root-owned on every start. Count a restore failure after model code ran as the
    policy's damage unless a fresh container also fails.
- **H1, the policy's own unusable summaries become `compaction_failed`, a truncation** (`agent.py:219, 1531-1539`;
  `context.py:719-734`).
  - The causes: an empty reply, a tool call, or a summary too long.
  - Rejected attempts are never trained, so a hopeless episode can trade its 0 for a truncation. That matters with
    overlong filtering or truncation bootstrap.
  - Fix: keep truncation for "cannot shrink". Score content failures as their own scored stop.
- **H2, an infra fault in a child agent or sub-call is scored; on the root it is masked** (`agent.py:1831-1835, 1116`;
  `tree.py:693-699`).
  - Fix: catch every non-cancel exception in a child's `run` and in `_sub`, then set `tree.fault`.
- **H3, the context-loop stall rule counts only World steps as progress** (`agent.py:1805-1811`). An agent reading a
  large `CONTEXT` in order is stopped `stalled`.
- **M2, refused tool rows land after notice user rows** (`agent.py:1738-1747`). OpenAI-style APIs reject that order, a
  policy-triggerable `agent_request_failed` mask.
- **M4, rows appended after the model's last turn are dropped when the tail does not fit** (`context.py:455-482`).
  - What is lost: an INBOX announcement, a wrap-up notice, child events.
  - Fix: always keep the rows after the last assistant turn.
- **Low.**
  - L1: the recovery notice's `history.since` step is off by the cell's earlier actions.
  - L2: a resumed cell answers `tool_calls[0]`, not the python call.
  - L3: the boundary marker's `window=w-1` points at a rerender window under interleaved templates.
  - L4: the snapshot cadence subtracts journal length from cell counts.
  - L5: the tokens of a discarded silent-overflow reply are not charged.
  - L6: the kernel container is renewed only at the loop top.

**Checked and found correct:**
- Overflow retries discard the reply and charge no turn.
- Rejected summary attempts are never trained.
- Chains are keyed per agent, and capture deltas reset per segment.
- A resume opens a root segment.
- The ledger cap keeps World results whole.
- `budget:<kind>` stops reach `res.truncated` (`runtime/harness/rlm.py:136-138`).

## 7. Comparison with other frameworks

Roots under `RL-Environments/06-frameworks-and-env-code/`. Each row is how a framework handles one concern, with
Crucible's state.

| Concern | Best practice seen | Crucible |
|---|---|---|
| TITO and prefix checks | verifiers stores per-node engine ids and raises on a non-extending bridged turn (`verifiers/v1/graph.py:798-822`, `train.py:403-425`); NeMo-RL asserts "Non-contiguous" (`nemo_rl/environments/nemo_gym.py:1225-1232`); OpenEnv's capture proxy never tokenizes (a local re-render matched the engine on 0 of 28 turns, `core/harness/rollout.py:138-161`) | Equivalent: `EngineLLM` extends ids and flags re-renders; `record.build` masks a prefix break. Crucible splits at a template re-render instead of failing (the "interleaved" Qwen template), which matches SkyRL's prefix-exact merging (`skyrl/train/generators/utils.py:920-934`). |
| Segments after compaction | verifiers: one sample per graph leaf; the summary is trained at the end of its segment, then prompt (`trace.py:192-193, 515-553`) | Same (§3.1). |
| Advantage across segments | one per episode, copied to all segments (prime-rl `orchestrator/algo/routing.py:11-40`; SkyRL `trainer.py:1212-1250`; TRL `async_rollout_worker.py:1072-1074`) | Same (§3.3). |
| Summary attempts | trained only when a `compaction` edge accepts them (verifiers `trace.py:519-549`); prime-agent types each compaction's outcome (`semantic_edges/compaction.rs:12-168`); Harbor flags copied context "not for training" (`models/trajectories/step.py:64-72`) | M16 already did; `LLMToolPolicy` now does too (B1). |
| Ungraded runs | `None`, not 0; dropped per member, baseline over survivors (prime-rl `algo/base.py:11-18`); typed verifier errors (Harbor `verifier/verifier.py:20-122`). Anti-patterns: masked members in the GRPO mean (SkyRL `ppo_utils.py:1374-1383`), infra scored 0 (prime-environments, rLLM `integrations/harbor/runtime.py:165`) | Stricter: a masked member drops its whole group (no partial group), and a failed grade is masked. |
| Truncation | scored on the final state and flagged (verifiers `trace.py:620-632`); optional overlong masking (NeMo-RL `grpo.py:3571-3580`; DAPO); prime-rl drops timeouts in training (`dispatcher.py:668-674`) | Scored and flagged; `LimitPolicy` can mask or penalise per stop. Training now counts only the agent's own time and masks a `timeout`, as prime-rl does (R2). |
| Group gates | zero-variance and min-members filters (SkyRL `trainer_utils.py:661-677`, rLLM `buffer.py:203`); abort when the masked fraction is too high (NeMo-Gym `rollout_collection.py:2666-2674`) | Zero spread, infra, staleness and objective filters, and `too_many_drops` (`trainer.py:410-419`). The masked-share stop (`max_masked_frac`, §8) now stops a training run whose recent samples are mostly masked. |
| Grading hygiene | tests staged only at verify time, `/tests` wiped (verifiers `tasksets/harbor/taskset.py:318-354`); test edits stripped from the patch and a held-out test patch applied (NeMo-Gym `swemer_v2/verification.py:129-131, 161-164`); fail-closed egress (NeMo-Gym `sandbox/agent_tools.py:62-69`) | Comparable or stronger: a pristine-tree grade in its own container, protected paths restored, path checks on the World's trace, `--network none`. |
| Format errors | cap consecutive format errors (mini-swe-agent `agents/default.py:100-114`); optional negative advantage on invalid tool calls (NeMo-RL `grpo.py:2319-2406`) | Capped (`max_nudges`, `max_tool_errors`); the optional format penalty is now applied by every backend (R5). |
| Logprob alignment | sampled tokens get mask 1 only if their logprobs line up (OpenEnv `capture/graph.py:443-473`) | `Turn.missing()` masks the whole rollout on a length mismatch (`core/schema/rollout.py:172-182`), stricter. |

Bugs seen in the references, which Crucible avoids:
- rLLM's `TITOCompleter` prepends the previous token input even when `is_prefix` is False (`rllm/engine/rollout/completer.py:98-112`).
- skyrl-agent's train backend fills missing logprobs with 0.0 (`skyrl_train_backend.py:38`).
- NeMo-Gym hard-codes `training_eligible=False` for post-compaction segments (`compaction/session.py:237`).

## 8. Open risks and recommendations

The open-risks pass (2026-10-28, branch `worktree-agent-a6734480fd8877554`) fixed or decided every item below. Each fix
has a unit test that fails on the code before it: `tests/unit/test_open_risks_loop.py` (R1, R2, R5, R8, R9, the
masked-share stop, the host checkpoint snapshot of R6), `test_open_risks_integrity.py` (R3, R6) and
`test_open_risks_agentenv.py` (R10).

| Item | Status | Where |
|---|---|---|
| R1 harness exceptions scored 0 | fixed | `actors/policy.py` `run_episode`, `Policy.harness_owned`, `PolicyOutputError` |
| R2 the agent clock counts sampler time | fixed (training default) | `EpisodeLimits.agent_clock`, `model_time`, `make_env(training=True)` |
| R3 a policy masks its own episode by killing its World | fixed | `factory/envs.py` `InfraWatch`, `death_evidence`, `native.broke_world` |
| R4 an answered history probe is masked | unchanged (correct per P21) | |
| R5 format penalty in the torch trainer only | fixed (one rule on every backend, off by default) | `datums.build_datums`, `TrainConfig.tool_format_penalty` |
| R6 snapshots and grades that trust the agent's tree | fixed | `verify/collect.py`, `windows.snapshot_patch`, `pathcheck.runner_plumbing` |
| R7 compaction in training | unchanged (out of scope: not used in training) | |
| R8 honesty in flat groups | decided: keep groups whose composed advantages differ | `TinkerTrainer._has_signal` |
| R9 `loss_scale="sum"` | decided: default `token_mean` | `TrainConfig.loss_scale` |
| R10 agentenv | fixed (World leak, THIRD_PARTY rows) | see R10 |
| masked-fraction abort (§7) | fixed | `TrainConfig.max_masked_frac`, `masked_window` |

- **R1. Policy exceptions were scored 0** (`actors/policy.py:776-778`). **Fixed.**
  - A `ValueError`, `IndexError`, `KeyError` or `TypeError` from harness code inside `act` was scored as the policy's.
  - Now a policy declares `harness_owned`. `LLMToolPolicy` does: it turns every fault of its model's output into feedback
    inside `act` (unparseable arguments become an `invalid_arguments` tool reply, a cut-off reply a nudge), so anything
    that still escapes `act` is the framework's.
  - Such an exception masks the episode as `crucible:framework_error` (blame `harness`, detail `framework`). It reaches
    `framework_health`'s `masked_failure` cluster (component `harness`) and never trains the policy away from a bug it
    did not cause.
  - A policy reading its model's reply raises `PolicyOutputError` where the reply itself cannot be acted on; that stays
    `crucible:policy_error`, scored 0. Policies whose own logic parses actions (scripted, game seats) keep the old rule.
  - Model-caused tool errors were never on this path: they are World answers (`PolicyToolError`, fault `policy`).
- **R2. The wall-clock agent timeout counted sampler latency.** **Fixed; the training default.**
  - `EpisodeLimits.agent_clock="own"`: each model call holds the agent clock (`actors.policy.model_time`, a
    `hostreach.held(only=...)` hold on that one clock). The World's tool deadline and the Runner's attempt clock
    (`rollout_s`, masked) keep running, so a sampler that never answers still ends the attempt.
  - `make_env(training=True)` (set by `factory.tinker.train`, the DeltaAI run's path) uses it and masks a `timeout` stop
    (`LimitPolicy(on_timeout="mask")`), as prime-rl drops timed-out rollouts in training
    (`orchestrator/dispatcher.py:668-674`).
  - Evals and benchmarks are unchanged: the wall clock and a scored `timeout`. `EpisodeLimits` leaves `agent_clock` out
    of a harness id at its default, so existing ids are unchanged.
  - Not covered: the RLM's model calls do not go through `model_time` (the RLM is not on the training path).
- **R3. A policy could mask its own episode by killing its container** (`env_container_lost`). **Fixed.**
  - `InfraWatch` now asks the daemon why a dead World died (`envs.death_evidence`: `DockerExec.lost` and the container's
    `.State`) and records the last shell command the policy ran.
  - The death is the policy's outcome only when the daemon saw no stop from outside and the container either ran out of
    its own memory limit (`OOMKilled`) or exited right after a kill or shutdown command (`envs.KILL_COMMAND`). Then it is
    `crucible:agent_broke_env`: scored 0, never collected or graded (`native.broke_world` in both harnesses), and the
    episode stops at once.
  - Everything else stays `env_container_lost`, masked, with the evidence in the failure, so it can be measured.
  - The window World's PID 1 is `sleep infinity`, which no signal from inside the container can kill, so this branch
    should be rare; the four measured "after step 1" masks stay infrastructure.
- **R4. A history probe the World answered is masked as `answer_exposed`.** The policy gets no negative signal for
  trying. This is correct per P21 (the task leaked), but such tasks should leave the training pool. The masks name them.
  Unchanged.
- **R5. Format penalties were torch-trainer only.** **Fixed: one rule on every backend.**
  - `TrainConfig.tool_format_penalty`, `format_penalty_mode` (`reduction` | `signed`) and `format_kappa` are the torch
    trainer's settings, applied by `datums.build_datums` on the offending call's own tokens: `reduction` subtracts the
    penalty per malformed call, `signed` rebalances by the sign of the rollout's advantage with each sign's batch mass
    kept (`credit.signed`).
  - The default stays 0 (off) on every backend, as the torch trainer's was. A malformed call already costs one of the
    tool calls, is answered with the error and counts toward `too_many_errors`, so its outcome already carries the cost;
    `reduction` adds net negative mass, which `signed` avoids. A run that sets one is a different objective (it is in
    the run identity then).
- **R6. Patch identity between the sidecar and the collect, and what else trusts the agent's tree.** **Fixed.**
  - `open_world` records the base commit before the agent acts (`world._crucible_base_sha`). `WindowTask.snapshot_patch`
    and the slime snapshot diff against it with one hardened in-container command (`verify.collect.snapshot_command`):
    - no system or global config, the git dir and work tree named explicitly;
    - no hooks, fsmonitor or submodule recursion;
    - the repository's own filter drivers made pass-throughs;
    - the collect's derived-file excludes, and a fixed diff format.
  - A World with no recorded base is unmeasured, never diffed against the tag.
  - The audit of what else trusts the agent's tree found three holes, all fixed:
    1. **Filter drivers.** Host git in the agent's repository ran its own filter drivers at `git add`: code the policy
       chose ran on the host and could put any content in the graded patch. Every host git call now neutralises them
       (`verify.collect.safe_flags`). The host checkpoint snapshot of lanes without their own snapshot
       (`checkpoint.workspace_snapshot`) does the same at command scope (`fs.git_diff_command(config=)`).
    2. **A redirected repository.** A `.git` replaced by a symlink or `gitdir:` file, a `commondir`, `core.worktree` or
       `core.bare` made host git index another repository or tree. `repo_broken` now refuses these as `agent_broke_env`
       before any host git runs (`verify.collect.repo_hazards`).
    3. **Test-runner plumbing.** The grader runs pytest with the tree on `PYTHONPATH` and plugin autoload on, so a patch
       adding a `[pytest11]` entry point or a module shadowing the runner (pytest, `_pytest`, pluggy, iniconfig, py,
       `pytest_timeout`, m17guard) steered the test run. These are now `path:plumbing` (hack) when the patch adds them
       outside the project's own names (`pathcheck.runner_plumbing`). Plugin autoload stays on: turning it off would
       change kept verdicts (`--timeout` needs the autoloaded plugin).
  - **Verdict-affecting, flagged:** `RULES_VERSION` moved to `path-2026-10-28`, which changes every window task's
    verifier version (and so the objective id); stored path verdicts read as stale in `framework_health` and `repath`.
    No kept oracle or floor verdict changes. Only a policy that games these surfaces scores differently.
  - Checked and sound: the final collect's recorded base, the pre-agent base reads (`_base_sha`,
    `checkpoint.start_commit`), protected test files and the conftest (protected, restored, plumbing). Other
    gitattributes (eol, `-diff`, `export-ignore`) only change content the agent chose and never reach the pristine grade
    tree.
  - Not hardened: `world/state/fs.FsAdapter.q_git_diff` (the native harness's verifier view) still honours repo-local
    filter drivers when it runs without a `trusted_git_dir`. It is off the training path, and the layering keeps the
    helpers in `verify` out of `world`.
- **R7. Compaction is untested in training, and its accounting is approximate.** Out of scope for this pass (training
  runs without compaction).
  - Before any run turns compaction on:
    - pass a `ContextConfig` through `NativeEnv.policy_kwargs` (today it carries only the tool limit);
    - fit the summary call to the window as `_reply_tokens` does;
    - add `_compactions_in_row` to `state_dict`.
  - With 0 `context_overflow` in 790 rollouts, it is not urgent.
- **R8. Honesty in flat groups.** **Decided: a flat group trains when its composed advantages differ.**
  - `drop_zero_spread` now drops a group whose members' composed sequence advantages (outcome + honesty, exactly what
    `build_datums` trains) are all equal (`TinkerTrainer._has_signal`), not one whose rewards are.
  - Why: on hard tasks most groups fail together. There the honesty term alone separates an honest "not done" (0) from
    a false "done" (< 0), and that is the only place calibration can train. Dropping on reward spread is what DAPO,
    slime (`check_reward_nonzero_std`) and prime-rl (`no_signal`) do because a flat GRPO group has no gradient; this
    group has one. A group whose composed advantages are equal (every member equally miscalibrated) is still dropped:
    with no honest member in it, the term would push down whole trajectories with no contrast.
  - Every group with outcome spread keeps its spread after composition, so the change only adds groups. The quarantine
    still reads rewards and Φ.
- **R9. `loss_scale="sum"` weighs episodes by length.** **Decided: the default is now `token_mean`.**
  - Both modes weigh every trained token equally within a step, so long episodes carry more weight in both. That is
    the Dr.GRPO choice, kept on purpose: a per-episode mean would weigh a long failing episode's tokens less and favour
    long failures.
  - They differ in the step's scale. `sum` scales the gradient by the batch's trained-token count, which swings several
    fold between batches of 20-call and 300-call episodes. With prime-rl's `max_norm = 1` the clip then binds on almost
    every step and gives every step the same length, whatever its signal.
  - `token_mean` divides by the batch's token count (a global token mean). That is how prime-rl (global rl weight),
    AReaL, SkyRL, verl, NeMo-RL and TRL normalise (async-rl-audit §3.3). On the prime-rl path it is prime-rl's own
    normalisation (`wire.undo_token_mean` and `datums.scale` cancel). The score-centering audit verified that both modes
    keep SC's per-token weights correct.
  - A constant normaliser (Dr.GRPO's generation budget; async audit R9) would also keep the step size steady. Under
    AdamW its only gain over `token_mean` is when clipping stops binding, so it is not adopted.
  - **Resume, flagged:** a run directory created with the old default records `loss_scale = "sum"`; resuming it needs
    `--loss-scale sum`, since the trained objective must not change mid-run.
- **R10. The agentenv items in §5.** See §5: the World leak on a failed run is fixed and the THIRD_PARTY rows are added
  (`interop/agentenv/plugin/steps.py`, `CrucibleWorldProvider`; tests in `test_open_risks_agentenv.py`). The unusable export and the trigger restore are unchanged (no training-path user).
- **The masked-fraction abort** (§7, NeMo-Gym `rollout_collection.py:2666-2674`). **Fixed.**
  - Over the last `masked_window` landed samples (default 64), more than `max_masked_frac` masked (default 0.5) stops
    the run with `stop = "masked_frac"`. The state is saved, and the mask kinds go to `summary.json` and a `stop` record.
  - NeMo-Gym's own setting is off by default, with 0.5 in its example and 50 minimum samples.
  - Both settings are resumable (operational). `None` turns it off. Masked training timeouts (R2) count, and the kinds
    tell them apart.
