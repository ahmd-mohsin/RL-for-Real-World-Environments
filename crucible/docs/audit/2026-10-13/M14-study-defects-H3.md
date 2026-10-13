# M14 defect hunt H3: the RLM kernel, policy, continual harness and manager

Crucible HEAD `f2d9e18`. Scope: `actors/rlm/{repl,kernel,policy,harness_state}.py`, `runtime/harness/rlm.py`,
`train/world_env.rlm_world_rollout`, `runtime/record.build` (sub-calls), `accept/repair.py` (RLMAgent, ForgeWorkspace,
TaskFSM). No source or test was edited.

Reproducers are in `scratchpad/m14/h3/` (session scratchpad), one `rNN_*.py` per finding, each with its captured `rNN.out`.
Run them from `crucible/` with `PYTHONPATH=src .venv/bin/python <file>`. Docker 29.8 was used locally for V3. Every
container the reproducers started was removed afterwards, and nothing was pruned. The credential canaries are fake
values planted in the launching process's environment. No real secret or `.env` content was read.

## Summary

| id | sev | where | defect |
|---|---|---|---|
| V1 | P0 | `kernel.py:161`, `policy.py:139`, `repair.py:146,512,543` | The default launcher of `RLMPolicy`, `RLMAgent`, `rlm_implementer` and `RLMTaskManager` is `LocalLauncher`, so model code runs on the host. It can read the parent's API keys (via `ps -E` on macOS or `/proc/<ppid>/environ` on Linux), find the repo `.env` from `sys.argv[0]`, reach the network, write outside its directory, and leave processes behind that outlive the kernel. |
| V2 | P0 | `repl.py:35,44-48`; `kernel.py:194-206` | The protocol shares fd 1 with cell code, and the host parses any JSON line on it as an event. A forged `done` with an escaped lone surrogate reaches the next prompt, and the codec failure becomes `crucible:harness_unsupported`. The policy can turn its own failing episode into an infra mask. |
| V3 | P1 | `kernel.py:174-187,225-229` | `close()` and the wedged-cell kill only kill the local `docker exec` client. The kernel process inside the container keeps running. With `DockerLauncher` a CPU-pinned process is left behind after each wedge. With `ExecLauncher` into a World sandbox, model-started work keeps changing the sandbox after the episode is closed. |
| V4 | P1 | `policy.py:263-265,271-274` | Sub-LLM calls have no budget. One cell can issue any number of calls, and `max_tokens` is caller-chosen and unbounded. |
| V5 | P1 | `kernel.py:245-246`; `policy.py:271-280` | A sub-LLM provider fault becomes a cell exception. The episode is scored and not masked, and the failed call leaves no record, so the rollout looks token-complete. |
| V6 | P1 | `policy.py:202-207,288-290`; `kernel.py:167-172` | A sandbox infra fault (daemon down, container lost, restore cannot start) is classified `crucible:policy_error`: scored 0 and trainable. |
| V7 | P1 | `harness_state.py:163-199` | The A/B can keep edits without real evidence. A missing or NaN guard never counts as a regression. NaN runs are dropped per arm, so the comparison is unpaired. One seed with margin 0 is allowed, and `fn` determinism is never checked. |
| V8 | P1 | `repair.py:303-333` | `commit` trusts whatever verification is stored last. `verify(checks=[subset])` records `passed=True` over the subset and overwrites the full, failing verification, so a task that fails A16 gets committed. |
| V9 | P1 | `policy.py:447-452` | Under an interleaved or strip thinking template, the compaction call's `new_segment` is ignored. The result is a prefix_break, and every compacting rollout is masked `capture_incomplete`. |
| V10 | P2 | `kernel.py:276-299`; `policy.py:306-310` | Exact restore needs cells to be deterministic. One `t0 = time.time()` makes every later restore raise `SnapshotMismatch`, which becomes `policy_error`. |
| V11 | P2 | `repair.py:123-136` | `rlm.spawn` limits depth only. Breadth and total turns, tokens and cost across the agent tree are unbounded. |
| V12 | P2 | `world_env.py:347-350` | `rlm_world_rollout` passes no seed, so every rollout's kernel `random` uses seed 0. A `launcher=` argument is one shared instance across concurrent rollouts: one rollout's `aclose` removes the others' container, and `/tmp` state persists between episodes. Found by reading the code; not reproduced. |
| V13 | P2 | `policy.py:225-269`; `repair.py:94-113` | The host bridge does not check its inputs. A direct `host("rlm.note", text=...)` bypasses the 512-character cap. A World call made while `_current_call is None` (during `_deliver` or `_archive`) is stepped but attached to no model call, which shifts per-call progress credit. Found by reading the code; not reproduced. |

## V1 (P0): model code runs on the host by default, and parent credentials are readable

**Where.** `Kernel.__init__` falls back to `LocalLauncher()` (`kernel.py:161`), and so does `RLMPolicy` (`policy.py:139`).
`RLMAgent.run` calls `Kernel(self.launcher)` with `launcher=None` by default (`repair.py:146`), and neither
`rlm_implementer` (`:512`) nor `RLMTaskManager.manage` (`:543`) passes one. The module doc says `LocalLauncher` is for
"trusted code only", but the manager path runs it with live LLMs, and the 2026-10-10 live smoke did exactly that.
`safe_environ` scrubs only the child's own environment.

**Reproducer.** `r01_local_launcher.py`, launched with `OPENAI_API_KEY=sk-canary-H3-exec-0123 HF_TOKEN=hf_canary_H3_exec`.
`r01b_linux_proc.sh` covers the Linux `/proc` path.

```
own_env_keys []
parent_env_canaries ['OPENAI_API_KEY=sk-canary-H3-exec-0123', 'HF_TOKEN=hf_canary_H3_exec']
repo_from_argv0 /Users/.../RL-for-Real-World-Environments | .env exists: True readable: True
network open
wrote_outside_cwd True
after Kernel.close(): orphan still alive -> 60329 sleep 123
--- Linux (python:3.12-slim) ---
child env has key: False
parent /proc environ: ['OPENAI_API_KEY=sk-canary-H3-linux']
```

**Minimal fix.**
1. Remove the `LocalLauncher` fallback. `Kernel`, `RLMPolicy` and `RLMAgent` should require a launcher, or default to
   `DockerLauncher`.
2. Have `LocalLauncher` refuse to start unless the caller passes `trusted=True`.
3. When it does run, start it from a re-exec'd launcher process that never held the credentials. Copy `repl.py` to the
   temp directory so `argv[0]` does not reveal the repo, and kill its process group on close (`os.killpg`).

**Regression test (fails now).** `RLMAgent("m", FakeLLM([...]))` without a launcher must either raise or run in Docker.
With the canary in `os.environ` when the test process starts, a cell that reads `/proc/{ppid}/environ` or `ps -E`
returns no canary.

## V2 (P0): protocol injection on fd 1 lets the policy mask its own episode

**Where.** The kernel writes events to `sys.__stdout__` (`repl.py:35`), the same fd 1 that cell code can reach with
`os.write(1, ...)` or `sys.__stdout__`. The cell id is readable from `sys.modules["__main__"]._cell`. `Kernel._read`
accepts any JSON line as an event (`kernel.py:194-206`), and `execute` takes the first `done` whose id matches.
`json.loads` decodes `"\ud800"` to a lone surrogate. That string goes into the tool message, the next
`EngineLLM.complete` cannot encode it, and the resulting `EngineCallError(failure_kind="crucible:harness_unsupported")`
is a masking kind (`run.py:53`).

**Reproducer.** `r02_protocol_mask.py` runs the real M12 `rollout_fn`. The cell forges a `done` event with an escaped lone
surrogate in `stdout`.

```
control (gives up): reward=0.0 mask=False trainable=True stop=agent_done
forged protocol line: rollout_fn returned None (episode MASKED: not trained, not scored, retried as infra)
direct: masked = True | failures = [('crucible:harness_unsupported', "... chat codec: UnicodeEncodeError: 'utf-8' codec can't encode c")]
```

A forged `done` can also set `finished`, `timed_out` or `digest`. A wrong `digest` makes the next restore raise
`SnapshotMismatch` (see V10).

**Minimal fix.**
1. Move the protocol off fd 1. Pass a dedicated pipe as fd 3 and `dup2` `/dev/null` (or a capture pipe) onto fds 1 and
   2 before any cell runs.
2. On the host, validate every event: known `event` names, field types, length caps, and `done` only for the cell in
   flight. Scrub strings with `.encode("utf-8", "replace").decode()`.
3. When the content that broke a model call came from the policy's own cell or tool output, classify the codec failure
   as `crucible:policy_error`, never as `harness_unsupported`.

**Regression test (fails now).** A cell that runs `os.write(1, json.dumps({"event": "done", "id": cid, "stdout": "\ud800"}).encode() + b"\n")`
must yield a real `done` whose stdout does not contain the forged text, and the episode must not be masked.

## V3 (P1): killing the `docker exec` client leaves the kernel running in the container

**Where.** `Kernel.close` and the wedged-cell path call `p.kill()` on the local client (`kernel.py:174-187,225-229`).
`docker exec` does not forward that kill to the process it started (the same issue `world/exec.with_deadline`
documents). `DockerLauncher` gets away with this only because `aclose` eventually removes the container. `ExecLauncher`
into a World sandbox, the documented coding-World placement, has no such cleanup.

**Reproducer.** `r03_exec_orphans.py`.

```
(a) wedge 0: killed=True timed_out=True
(a) wedge 1: killed=True timed_out=True
(a) kernel processes alive in the container after 2 kills + replays:
pid 14 python3 -u /tmp/crucible_rlm_repl.py
pid 21 python3 -u /tmp/crucible_rlm_repl.py
pid 28 python3 -u /tmp/crucible_rlm_repl.py
(a) container CPU: 97.92%
(b) kernel closed at 04:45:08
(b) 3 s after close, the World file reads: tampered 11
(b) 5 s after close, the World file reads: tampered 16
```

After two wedges and replays, three kernel processes share the 1-CPU quota, so the live kernel is starved. In part (b),
a sandbox file kept changing after `close()` returned.

**Minimal fix.**
1. Start the kernel through `with_deadline`-style wrapping, e.g. `setsid` plus a pid file.
2. On close or kill, run `docker exec <c> kill -KILL -<pgid>` and confirm the process is gone before returning.
3. For `ExecLauncher` in a World sandbox, require a quiesce of the policy's process group before grading.

**Regression test (fails now).** After a wedged cell and a replay, `docker exec <c> pgrep -f crucible_rlm_repl` lists one
process. After `close()`, a file written by a thread the cell started does not change again.

## V4 (P1): sub-LLM calls have no budget

**Reproducer.** `r04_subcalls.py` (a):

```
(a) sub-calls in one cell: 300; largest max_tokens sent: 1000000 (sub_max_tokens=1000)
```

**Fix.** Clamp the per-call limit to `min(max_tokens or sub_max_tokens, sub_max_tokens)`. Add `max_sub_calls` (per cell
and per episode) and a sub-call token budget to `RLMConfig` (they are hashed into the harness id). When a limit is hit,
return a policy-visible error.

**Regression test.** `rlm.batch(['x'] * 300, max_tokens=10**6)` makes at most `max_sub_calls` calls, each with
`max_tokens <= sub_max_tokens`.

## V5 (P1): a sub-LLM provider fault is scored and invisible in the record

**Where.** The handler turns every exception into a `__error__` reply (`kernel.py:245-246`), and `_sub` appends a record
only when the call succeeds (`policy.py:271-280`). On the root path, the same `LLMError` sets `provider_unavailable`
and masks.

**Reproducer.** `r04_subcalls.py` (b):

```
(b) cell output: cell saw LLMError: provider 503
(b) masked: False | failures: [] | reward: 0.0 | sub records: 0 | stop: agent_done
```

**Fix.** In `RLMPolicy._host`, catch `LLMError` and `EngineCallError` from `_sub`, record the failure kind on the policy
(`self.failure_kind`, `self.error`), and stop the episode as the root path does, so it masks. Leave policy-caused errors
such as a context overflow on the sub prompt as cell errors.

**Regression test.** If the sub-LLM raises `LLMError("503")`, `ep.masked` is True with kind `provider_unavailable`.

## V6 (P1): sandbox infra faults are scored as `policy_error`

**Where.** `act()` maps any driver exception to `crucible:policy_error` (`policy.py:202-207`). That covers a failed
bootstrap (`:288-290`), `KernelError` from `start` (`kernel.py:167-172`) and a failed restore replay. In
`rlm_world_rollout` such an episode is graded, and if it had any root call it is trainable at reward 0.

**Reproducer.** `r05_infra_scored.py`:

```
daemon down at start: masked=False failures=[('crucible:policy_error', 'RuntimeError: docker run failed: Cannot connect to the Docker daemon a')]
sandbox lost mid-episode: masked=False failures=[('crucible:policy_error', 'RuntimeError: docker exec failed: No such container: crucible-rlm-xyz')]
```

**Fix.** Give launcher and kernel-start failures their own exceptions carrying `failure_kind`. Use
`crucible:env_container_lost` or `crucible:backend_error` for a missing container or daemon, and
`crucible:env_image_unavailable` for a pull failure. Let them propagate with that kind, as `EngineCallError` does.
Replay divergence caused by the policy's own nondeterminism stays scored (V10).

**Regression test.** A launcher whose `start` raises "Cannot connect to the Docker daemon" yields a masked episode with
infra blame.

## V7 (P1): the A/B can keep an edit without real evidence

**Where.** `HarnessState.evaluate` (`harness_state.py:163-199`):
- `gain(g) < -tolerance` is False for NaN, so a guard the candidate stops reporting passes.
- `mean` drops NaN runs separately in each arm, so the comparison is not paired.
- There is no minimum number of seeds, `margin` defaults to 0, and nothing checks that `fn` returns the same metrics
  for the same state and seed.

**Reproducer.** `r06_harness_ab.py`:

```
(a) accepted: True {'base': 1.0, 'candidate': nan}
(b) accepted: True {'base': 0.5, 'candidate': 1.0}
(c) no-effect edits kept with one seed and margin 0: 58/100
```

**Fix.**
1. Reject when any primary or guard value is missing or NaN on either arm.
2. Pair the runs per seed and drop a seed from both arms, or reject.
3. Require `len(set(seeds)) >= min_seeds` (e.g. 3).
4. Require the paired improvement to clear `margin` and a sign test, or a CI lower bound above 0.
5. Optionally re-run the base on one seed and refuse if `fn` is not deterministic.

**Regression test.** Cases (a), (b) and (c) above must each be rejected.

## V8 (P1): commit trusts a partial verification

**Where.** `verify(..., checks=...)` computes `passed` over only the named checks and stores the result as the version's
verification, replacing any earlier one (`repair.py:303-326`). `commit` checks only `verification["passed"]`
(`:328-333`). The manager RLM reaches `verify` with arbitrary keyword arguments through `forge.verify`. Unknown check
names raise `KeyError`, so they are not a bypass.

**Reproducer.** `r07_forge_verify.py`, on the weak B.15 task that fails A16:

```
default checks: False ['A1', 'A16']
checks=['A2']: True {'A2': {'status': 'pass', 'reasons': []}}
commit: {'task_id': 't2', 'committed': 1}
```

**Fix.**
1. Store verifications keyed by check set.
2. Have `commit` require a passing verification whose check set covers `self.checks`, run against the current audit
   floor.
3. Keep a subset `verify` from overwriting a full one.

**Regression test.** `verify(t, v)` fails, then `verify(t, v, checks=["A2"])` passes; `commit(t, v)` must raise
`PermissionError`.

## V9 (P1): compaction under an interleaved or strip template is a prefix break

**Where.** `_maybe_compact` records the summary call in `self.segment` and runs `_compress` without reading
`r.tokens["new_segment"]` (`policy.py:447-452`). Root turns do read it (`:357`). `LLMToolPolicy._maybe_compact` has the
same gap (`actors/policy.py:298`).

**Reproducer.** `r08_compaction_rerender.py` uses a byte codec whose full render drops earlier `<think>` content, as
Qwen3-style templates do:

```
preserved template: reward=1.0 compactions=6 prefix_breaks=[] trainable=(True, 'ok')
interleaved template: reward=1.0 compactions=6 prefix_breaks=['agent:c3', ...6 calls] trainable=(False, '... agent:c3: prefix_break')
```

**Fix.** When the summary call's tokens carry `new_segment`, open a `rerender` segment before recording the call. One
option is to stamp the call with `segment + 1` and reset `_last_full`.

**Regression test.** The `r08` interleaved run is `valid_for_training()`.

## V10 (P2): exact restore fails after any nondeterministic cell

**Reproducer.** `r09_replay_nondeterminism.py`:

```
pure cell: stop=agent_done failures=[]
cell keeps time.time(): stop=agent_done failures=[('crucible:policy_error', 'SnapshotMismatch: RLM kernel replay diverged after cell 2 ...')]
```

**Fix.** Either journal and replay nondeterministic sources (patch `time.time`, `os.urandom` and `uuid4` in the kernel
to record and replay), or fall back on a divergence: keep the live kernel, tell the model the restore was not exact, and
do not end the episode. Document that restore is exact only for deterministic cells.

**Regression test.** After `t0 = time.time()` and then a wedged cell, the episode continues.

## V11 (P2): spawn breadth and tree budgets are unbounded

**Reproducer.** `r10_spawn_breadth.py`:

```
finished: True | model calls across the tree: 27 | children spawned from one cell: 25
```

**Fix.** Add a shared run budget across the agent tree (`max_spawns`, `max_total_turns`, token and cost caps) that every
`RLMAgent` checks.

**Regression test.** With `max_spawns=3`, the 4th spawn raises inside the cell.

## V12, V13 (P2, found by reading the code)

**V12.**
- `rlm_world_rollout`'s `make` passes no `seed` (`world_env.py:347-350`), so every rollout's kernel is seeded 0.
  `RLMHarness` derives the seed from `attempt.sample`; do the same here from `rid`.
- A `launcher=` instance passed to `rlm_world_rollout` is reused across concurrent rollouts. Take a factory instead.

**V13.**
- Bound `rlm.note` on the host side.
- In `_host`, refuse `world.call` and `world.send` while `_current_call is None`. That happens during the
  bootstrap, `_deliver` and `_archive` executions, which model code can redefine. Alternatively, attribute those actions
  explicitly; otherwise `credit_from_progress` silently shifts per-call credit.

## Checked and found sound

- Within one cell, World calls are serialized: the host handles one request at a time. Each `world.call` is one World
  step.
- Sub-calls are recorded as `kind="sub"` with `parent`, are stand-alone sequences in `build` and `full_prompts`, and do
  not disturb the root's `EngineLLM` conversation.
- A sub-call that lacks a token record does mask the rollout in train mode (`Turn.missing`).
- `HarnessState.rollback_to` restores whole-state versions, and base-prompt edits are refused.
- Spawn depth is enforced.
- `DockerLauncher` blocks the network, has a read-only root and runs as uid 65534. Its container is removed at
  `aclose`, and none were left after these runs.
