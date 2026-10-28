# Harness, compaction and reward audit (2026-10-28)

Scope: the long-horizon agent harness (the transcript policy `LLMToolPolicy` that trains the student, and the M16 RLM),
context compaction and the trained token stream (TITO), the reward and credit path (`credit/`, `train/credit.py`, the
Tinker / prime-rl datums, `verify/`, the checkpoint sidecar, honesty, path checks), and Crucible's use of the agentenv
framework. A comparison with nine local frameworks closes it. Every claim names a file and line. Paths are relative to
`crucible/src/crucible/` unless they say otherwise.

All the fixes are in commit `9db0857d` on branch `worktree-agent-ab4ad5539d6391889`. Each has a test in
`tests/unit/test_harness_reward_audit.py` that fails on the code before it.

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
  - Risk R1: a harness bug of those types inside `LLMToolPolicy` (a `zip(strict=True)` mismatch, a config error) becomes
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
  - Risk R2: the agent clock includes model sampling latency, so in a shared async sampler a `timeout` stop depends on
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
  - Risk R6: the windows lane's snapshot diffs against the agent-movable `base` tag (`factory/lanes/windows.py:2081`),
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
| `max_steps` (tool limit), `timeout`, `context_overflow`, `max_output_tokens`, `too_many_errors` | the grade of the partial work, truncated flag (limit policy `score`) | yes for a coding task graded on state (verifiers, Harbor, OpenEnv do the same). `timeout` is the exception: see R2 |
| `GIVE_UP:` / `STATUS.json` "not done" | the grade, plus an honesty term (a correct "not met" claim costs nothing) | yes |
| an agent-caused tool error | none by itself (fault `policy`, scored), counted toward `too_many_errors` | yes |
| a malformed tool call | the World's error message; a format penalty only in the torch trainer | partly (R5) |
| the policy's own exception | 0 (`crucible:policy_error`) | mostly; R1 |
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
  command kills its own container is masked, not scored, and its group's signal is lost (R3).

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
- **Not fixed: real bugs, recommended fixes.** These touch agentenv lifecycle and timeline code that has no
  training-path user today.
  - **A failed agentenv run leaks the World** (`interop/agentenv/plugin/steps.py:85-106`).
    - The World is closed only in `CrucibleVerifyStep`'s `finally`, so a failed deploy or prompt step leaks the gateway,
      the App servers and any paid sandbox.
    - The `mkdtemp` directory is never removed.
    - Fix: close on failure, and register teardown with agentenv's `teardown_run`.
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
  - **THIRD_PARTY.md lacks DEPEND rows** for the plugin, `publish.py`, the explorer, `mcp_env.py` and `agents.py`.

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
| Truncation | scored on the final state and flagged (verifiers `trace.py:620-632`); optional overlong masking (NeMo-RL `grpo.py:3571-3580`; DAPO); prime-rl drops timeouts in training (`dispatcher.py:668-674`) | Scored and flagged; `LimitPolicy` can mask or penalise per stop. A wall-clock `timeout` in training is load-dependent (R2): prime-rl's choice (drop it) is worth adopting. |
| Group gates | zero-variance and min-members filters (SkyRL `trainer_utils.py:661-677`, rLLM `buffer.py:203`); abort when the masked fraction is too high (NeMo-Gym `rollout_collection.py:2666-2674`) | Zero spread, infra, staleness and objective filters, and `too_many_drops` (`trainer.py:410-419`). No masked-fraction abort for the factory's measurement arms. |
| Grading hygiene | tests staged only at verify time, `/tests` wiped (verifiers `tasksets/harbor/taskset.py:318-354`); test edits stripped from the patch and a held-out test patch applied (NeMo-Gym `swemer_v2/verification.py:129-131, 161-164`); fail-closed egress (NeMo-Gym `sandbox/agent_tools.py:62-69`) | Comparable or stronger: a pristine-tree grade in its own container, protected paths restored, path checks on the World's trace, `--network none`. |
| Format errors | cap consecutive format errors (mini-swe-agent `agents/default.py:100-114`); optional negative advantage on invalid tool calls (NeMo-RL `grpo.py:2319-2406`) | Capped (`max_nudges`, `max_tool_errors`); the format penalty is not in the Tinker / prime-rl datums (R5). |
| Logprob alignment | sampled tokens get mask 1 only if their logprobs line up (OpenEnv `capture/graph.py:443-473`) | `Turn.missing()` masks the whole rollout on a length mismatch (`core/schema/rollout.py:172-182`), stricter. |

Bugs seen in the references, which Crucible avoids:
- rLLM's `TITOCompleter` prepends the previous token input even when `is_prefix` is False (`rllm/engine/rollout/completer.py:98-112`).
- skyrl-agent's train backend fills missing logprobs with 0.0 (`skyrl_train_backend.py:38`).
- NeMo-Gym hard-codes `training_eligible=False` for post-compaction segments (`compaction/session.py:237`).

## 8. Open risks and recommendations

- **R1. Policy exceptions are scored 0** (`actors/policy.py:776-778`).
  - A `ValueError`, `IndexError`, `KeyError` or `TypeError` from harness code inside `act` is scored as the policy's.
  - Recommend: classify by origin. Only exceptions raised while parsing the model's own output should be the policy's,
    for example a `PolicyOutputError` raised where `LLMToolPolicy` reads arguments. Everything else should be
    `agent_run_error` with unknown blame.
- **R2. The wall-clock agent timeout counts sampler latency.**
  - In async RL on a shared sampler, `timeout` stops (12 of 790 targeting rollouts) carry queue load into the reward.
  - Recommend, either:
    - mask `timeout` in training (`LimitPolicy(on_timeout="mask")`; B3 makes this work for the workspace harness), or
    - hold the agent clock during model calls with `hostreach.held()` around `llm.complete`, leaving a model-time budget
      separate.
- **R3. A policy can mask its own episode by killing its container** (`env_container_lost`, `factory/envs.py:127-141`).
  - Recommend: when the World dies right after a policy shell command, check that command against a kill-pattern list
    (`kill -9 1`, `pkill`, `shutdown`, `rm -rf /`). Score a match as `crucible:agent_broke_env` instead of masking it.
  - Measure first. The four "after step 1" masks in the targeting data look like infrastructure (the first step is
    usually a read).
- **R4. A history probe the World answered is masked as `answer_exposed`.** The policy gets no negative signal for
  trying. This is correct per P21 (the task leaked), but such tasks should leave the training pool. The masks name them.
- **R5. Format penalties are torch-trainer only.** `Turn.format_errors` is recorded but `datums.build_datums` ignores it.
  - Recommend: if a format penalty is wanted on Tinker / prime-rl, apply it in `build_datums` on the offending call's
    span, with the signed mode's rebalancing (`credit/signed.py`).
- **R6. Patch identity between the sidecar and the collect.**
  - Recommend: make the windows lane's `snapshot_patch` diff against the recorded base sha with the collect's own
    flags and excludes. B5's change-equality is the fallback.
  - Not done here: it changes the lane's Φ inputs mid-episode, and the lead owns the lanes.
- **R7. Compaction is untested in training, and its accounting is approximate.**
  - Before any run turns compaction on:
    - pass a `ContextConfig` through `NativeEnv.policy_kwargs` (today it carries only the tool limit);
    - fit the summary call to the window as `_reply_tokens` does;
    - add `_compactions_in_row` to `state_dict`.
  - With 0 `context_overflow` in 790 rollouts, it is not urgent.
- **R8. Honesty in flat groups.** `factory/honesty.compose_advantage` (`honesty.py:156-166`) says the honesty term alone
  separates honest from miscalibrated members in a group without outcome spread. `TrainConfig.drop_zero_spread=True` (the default) drops those groups first, so
  honesty never trains there.
  - Recommend: keep a flat group whose honesty terms differ, or document that honesty only reshapes groups with spread.
- **R9. `loss_scale="sum"` weighs episodes by length.** With episodes from 20 to 300 calls, the long ones dominate the
  gradient. `token_mean` (a constant per batch) keeps the Dr.GRPO property and bounds the step size.
- **R10. The agentenv items in §5** (World leak, unusable export, trigger restore).
