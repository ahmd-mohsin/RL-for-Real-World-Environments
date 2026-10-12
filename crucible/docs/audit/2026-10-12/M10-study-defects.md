# M10 study: defects in `runtime` (runner, harnesses, infra, budget, checkpoint, journal, records)

Scope: `C/runtime/` (`run`, `infra`, `keys`, `budget`, `checkpoint`, `journal`, `collect`, `inject`, `remote`, `gpu`, `perf_guard`, `record`, `single_turn`, `control_task`, `harness/{base,native,workspace,rlm,imported}`), `CP/runtime/r1.py`, `TP/runtime.py`, `TP/longhorizon/`, and the pieces M10 promises that live elsewhere: `C/interop/harbor/trajectories.py`, `C/verify/progress.py`, `C/actors/policy.py` (`ContextConfig`, `run_episode`, `LLMToolPolicy`), `World.stream_trace`. The two files the M7 audit changed were re-checked: `runtime/collect.py` (`tree_changed` ignores derived files) and `runtime/harness/workspace.py` (records the base sha before the agent runs).

Spec: ARCHITECTURE "### M10 `runtime`" (contents, Takes, exit gate); DESIGN §10.1 "As built (M10)", §10.4 (long horizons), P16, Appendix B.5; `docs/modules/M10-runtime.md`; `docs/research/m9-m10/02-runtime.md`; `docs/research/improvements/M09-M11.md` (M10 section); `docs/research/long-horizon/01-frameworks.md`; reports `docs/modules/reports/m10_r1.json` and `longhorizon_xl.json`.

Paths: `C` = `crucible/src/crucible/`. `CP` = `crucible/packs/coding/src/crucible_pack_coding/`. `TP` = `crucible/packs/_toy/src/crucible_pack_toy/`.

Method: a read-only study on HEAD `526fdb5`, using `crucible/.venv` (Python 3.12.9) on macOS. **[ran]** means reproduced with a probe; **[read]** means established from the code alone. The probes are in the session scratchpad at `m10hunt/p1_workspace_masks.py … p10_secrets.py` and `g1_longhorizon_gate.py`. All probes are offline: scripted policies, `FakeLLM`/`ReplayLLM`, temp directories. No paid call, GPU, lab host, Docker or network was used. Any credential in a probe is a synthetic canary built at run time.

Offline gates re-run:
- **B.5 injection gate (CPU part):** `tests/unit/test_runtime.py` passes inside the coverage run below (118 passed). The Linux-container classes in `tests/redteam/test_runtime_docker.py` were not re-run.
- **Long-horizon xl gate** (`TP/longhorizon/gate.py`), with `REPORT` redirected to the scratchpad so the repo report is untouched: see §4.
- **R1** (`CP/runtime/r1.py`) needs the lab RTX 5090 over ssh. It was not re-run. `ImportedHarness`, which R1 uses, has 0% unit coverage.

Note: the ARCHITECTURE build-status table has **no M10 row** (nor M9, M11 or M12). `git log -G` shows that no such row was ever committed. The only M10 status is in `docs/modules/M10-runtime.md` ("core built; exit gate in progress").

## 1. Requirement ledger

Status: **live** = built and behaves as specified; **partial** = built, but missing a specified part, or with a reproduced hole; **wrong** = contradicts the spec; **missing** = not built.

| # | Requirement (source) | Status | Evidence |
|---|---|---|---|
| L1 | Every infra failure is masked, never scored 0 (P16, ARCH exit gate) | partial | The 8 B.5 injections pass in `test_runtime.py`. A policy-caused `TimeoutError` is scored 0 as `timeout`, while any other exception is masked (V7) |
| L2 | Policy failures are **not** masked (DESIGN §10.1: "only policy failures are scored") | **wrong** | A policy can turn its own failure into a mask in several ways: corrupt `.git`, leave an unreadable file (V1); send bad compaction arguments, or a policy bug raises (V6); leave a background load running (V5); a provider 400 caused by the policy's own message (V7) |
| L3 | Blame attribution (`infra.py`, `DEFAULT_BLAME`) | partial | The taxonomy and `normalize` are correct. `boundary` maps every unknown exception to `unknown` blame, so a policy-side crash is masked, and enough of them trip the run-wide unknown-fraction stop (V6) |
| L4 | Retries: grade-only regrade or re-collect, never re-running the agent (DESIGN §10.1, `workspace.py:7-11`) | **wrong** | The third attempt after an empty collection re-runs the agent (V1, p1). In train mode the regrade is masked `capture_incomplete` and the next retry re-runs the agent (V3, p4). The re-collect diffs against the movable `base` tag, not the recorded sha (V2, p2) |
| L5 | Budget: reserve/settle, caps, credit exhaustion as infra, uncertain spend (ARCH Takes; DESIGN §10.1) | partial | Credits exhausted stop the run and the sample stays owed (tests pass). The default estimate is 0, so concurrent attempts overspend the cap 3× (V4). `over_uncertain_tolerance` and `release` are never called. Settle is unbounded |
| L6 | Checkpoint/resume: resume from the right step, no double counting (DESIGN §10.4) | partial | Resume from `a1@step100` works once (test). The resumed attempt's cost re-charges every call restored from the checkpoint (V8). A second crash cannot resume: the store blobs are in `a1/work` (V9). A real process crash never resumes (V9). There are no digest or identity checks on the checkpoint (research item still open) |
| L7 | Journal crash safety and idempotence (`journal.py` docstring) | partial | Writes are fsynced and `record.json` is atomic. A torn tail swallows the next appended row, and the sample is re-run (V10). There is no process lock, so a second runner on the same run id moves the first runner's in-flight attempts to `aborted/` (research item, still open) [read] |
| L8 | Harness exceptions contained (research M09-M11 "bugs found") | **wrong** | `harness.run` and `_rollout` are unguarded (`run.py:139-142`). One exception aborts the whole `asyncio.gather` and leaves the ledger op `reserved` (V11, p3) |
| L9 | k samples and seeding (`keys.py` docstring: the seed is "sent to the endpoint and used as the LLM cache key") | partial | Sample ids are deterministic. `derive_seed` is computed and recorded (`run.py:86`, `226`), but nothing uses it: harnesses pass `sample_key=attempt.id` (`native.py:180,252`, `workspace.py:115`), so a retry changes the key, and the endpoint gets `seed=None` (`policy.py` act) [read] (V16) |
| L10 | Cost accounting per attempt | partial | Own-call costs work for concurrent attempts (`test_runtime_cost.py`). A masked agent-stage attempt has no `res.calls` (`save_episode` is never reached), so it falls back to the shared `PolicyRef.spend` delta: 0 when unset (p6: a1 made 150 calls and records `cost_usd 0.0`). The resume double count is V8 |
| L11 | Perf guard: one perf grade per host, only on a quiet host (B.5 row 7) | partial | The lock and the host check work (tests). The grade runs before the policy's World is closed, so the policy's own sandbox and processes count as contention (V5) |
| L12 | GPU hosts and preflight (`gpu.py`; B.5 row 4) | partial | `preflight` (PID 1 init, GPU arch) has no caller in `src/` or `packs/` (grep). `world/sandbox.py:87` passes `--init`, which is the real fix. There is no per-attempt GPU selection: `NVIDIA_VISIBLE_DEVICES=all` (`gpu.py:72`). `GpuHost.get` ignores rsync failures (`gpu.py:69`). 0% coverage |
| L13 | Control channel and self-kill (B.5 row 8, `remote.py`) | partial | `control()` has no caller outside tests. The lint misses `pgrep -af`, `-fl` and `--full`, and it flags the safe quoted form `pkill -f "[f]oo"` (V15) |
| L14 | Records: ATIF step ids line up with `steps.jsonl` (ARCH `trajectories.py`) | partial | `to_atif` and `build` join calls to ATIF steps (tests pass). `align` and `cross_check` have no caller outside `tests/unit/test_runtime_records.py`. `Runner._rollout` always passes `rows=()` (`run.py:212`), so no `steps.jsonl` is written or aligned per attempt. The file lives in `interop/harbor/`, not `runtime/` (the improvements item about Harbor imports in runtime is resolved) |
| L15 | Train-mode masking of incomplete token records | live, with a side effect | `record.py:102-106`. It also masks every grade-only retry (V3) |
| L16 | Single-turn parsing failures score 0, never crash | live | p5: `"["*20000`, `'{"a": '*5000`, `1e999999`, `nan` and NUL bytes all score 0, not masked (the M7 V19 fix holds). A judge outage masks `check_infra` |
| L17 | Long horizons: compaction and segments, streamed trace, dense progress (DESIGN §10.4) | partial | The xl gate passes (§4). A bad `context_compact` argument from the model masks the episode (V6). `_run_steps` resets the World on resume (V14) |
| L18 | Collect hook (B.5 row 1; the M7 changes) | partial | `tree_changed` ignores derived files, which is correct. The base sha is recorded before the agent runs, which is correct on the first collection only (V2). `git diff`'s return code is unchecked (`verify/collect.py:69`). The patch is still decoded with `errors="replace"` (`:69`; M7 V20 item 3 is still open) |
| L19 | Remote and keys: no secrets in records | partial | Nothing scrubs at the record boundary. A provider error that echoes a credential lands in 14 files: record, results, failures, trace and ATIF trajectory (V12, p10) |
| L20 | `ControlTask` reward from the environment return (M1 item 10) | partial | `score(nan) = 1.0` (V13) |
| L21 | `harness/harbor.py` (Harbor installed agents through `Trial`; ARCH contents and Takes DEPEND) | **missing** | There is no file. The module doc lists it under "Remaining". Verifiers harnesses are also absent. The A2A harness lives in `C/interop/agentenv/harness.py` |
| L22 | Exit gate (a): "one historical frontier run (mini-swe-agent on task02) reproduced end to end" | partial | R1 is a **re-grade** of the recorded GPT 5.6 sol patch through `ImportedHarness` (`m10_r1.json`, profile identical). No agent was run. R2 (replay) and R3 (live mini-swe-agent through Harbor) are pending and depend on L21 |
| L23 | Exit gate (b): all 8 B.5 classes injected, classified infra and masked | live (as injected) | CPU tests pass. The docker classes were not re-run here. The injections prove that infra is masked, but nothing proves that the policy cannot reach those masks (L2) |
| L24 | Resume refuses a different spec; a finished spec dispatches nothing | live | `journal.py:65-68`; tests pass. The identity hashes task **ids**, not content (`run.py:54`), so a task edited under the same id resumes silently [read] |
| L25 | Imported harness (R1, dataset submissions) | partial | 0% coverage. A missing recorded patch is classified `capture_incomplete` (infra) and retried 3 times (`imported.py:33-34`), not raised as a configuration error [read] |

## 2. Defect candidates

Severity: **P0** = a policy obtains reward it did not earn, or a failure it caused is masked or mis-scored in a way it can exploit, or secrets leak into records. **P1** = a wrong classification or result, or a crash, in normal use. **P2** = an edge case.

### Masks a policy can trigger (P16: masks are for infrastructure only)

**V1 [P0][ran] A policy that breaks collection is masked instead of scored 0, and the third attempt re-runs the agent.**
- `classify` (`runtime/collect.py:54-61`) masks every empty patch from a changed tree, or one with hook errors, as `crucible:empty_collection` (harness blame). It does not ask why the hook failed. `FibTask.broken` only checks that `.git` exists (`TP/runtime.py:95-96`).
- p1, `WorkspaceHarness` + `FibTask` + scripted policies through `Runner`:
  - `printf garbage > .git/HEAD` gives `reward None, masked, empty_collection, attempts 3`;
  - `echo x > notes.txt && chmod 000 notes.txt` gives the same;
  - `printf junk > .git/index` gives the same;
  - controls: no-op 0.0 and oracle 1.0, not masked.
- A do-nothing policy escapes its 0 with one shell command. Under GRPO the masked sample drops out of the group, which is the exploit P16 forbids.
- p1 detail, per attempt: a1 runs the agent; a2 is `recollected`; **a3 is `agent_done`**, a fresh agent run. `_prior` needs `prev/preserved_ws` (`workspace.py:43`), but a re-collect attempt never preserves its tree, so the third attempt falls through to a full re-run. This contradicts "neither re-runs the agent" (DESIGN §10.1).
- Fix:
  - collection errors on paths the agent wrote (permission denied, unreadable or corrupt `.git` files) are the policy's outcome (`crucible:agent_broke_env`, scored 0);
  - only hook or transport errors are infra; `broken()` should run `git rev-parse --verify <base_sha>` and `git status`, not just test that `.git` exists;
  - carry the preserved tree forward (or look back to the first attempt that preserved it), so no retry re-runs the agent.

**V5 [P0][ran] Load the policy leaves running masks perf grades (its own and other samples').**
- `WorkspaceHarness.run` grades inside the `PerfGuard` slot (`workspace.py:133`, `_grade` `:69-73`) **before** `world.close()` (`:137-139`). The guard counts any GPU or CPU load and any running `crucible*` container (`perf_guard.py:49`, `78-79`).
- p8: a `perf = True` FibTask, and a probe that reports 95% GPU utilization while the policy's sentinel process lives:
  - the oracle alone gives 1.0;
  - the oracle plus `nohup bash -c 'exec -a <tag> sleep 30' &` gives `masked node_contention, attempts 3`;
  - the sentinel is still alive after the World closed.
- [read] With the docker backend, the attempt's own `crucible-sbx-*` container (`world/sandbox.py:81`) is still running at grade time, so every perf grade waits `wait_s` (900 s) and is then masked. Concurrent samples' sandboxes mask each other.
- Fix:
  - close (and kill the process group of) the policy's World before the perf slot, as the separate-verifier path does;
  - exclude the attempt's own sandbox from the guard's count;
  - a load the attempt itself left behind is `agent_broke_env`, not contention.

**V6 [P1][ran] Model output or policy code that raises inside the episode is masked with `unknown` blame.**
- `run_episode` does not catch `policy.act` exceptions (`policy.py` loop). `boundary("agent")` maps any unknown exception to `agent_run_error`, blame `unknown` (`infra.py:84`).
- p5 results:
  - an LLM reply `context_compact {"keep_recent_turns": "all"}` under `ContextConfig(mode="action")` gives `ValueError: invalid literal for int()` (`policy.py:429`), then masked, unknown;
  - a scripted or neural policy raising `IndexError` gives masked, unknown.
- Unknown masks are not retried, and above 5% of terminal samples they stop the whole run (`infra.py:169-170`). A policy that emits that tool call can therefore halt a training run.
- Fix:
  - in `LLMToolPolicy`, validate `context_compact` arguments: bad ones are a format error, scored;
  - in `run_episode`, catch exceptions from `policy.act` and record a `crucible:policy_error` failure. For a model-driven policy this is policy blame; keep `unknown` only for harness code.

**V7 [P1][ran] Policy-caused provider rejections are masked, while timeouts are scored, whatever their cause.**
- `RequestRejected` (any non-retryable 4xx, `llm.py:299-300`) carries `failure_kind = "agent_request_failed"`, which is harness blame and so masked and retried 3 times (p5 `provider_400`). A 400 is often caused by the policy's own earlier output: malformed tool-call JSON replayed in the history, or an over-long argument.
- A `TimeoutError` raised inside the loop (`policy.py:587`) becomes stop `timeout`, which is a scored limit (p5: reward 0.0), even when it came from an inference-server socket rather than the episode clock.
- Fix:
  - give `run_episode` its own deadline flag, and treat only the episode's `wait_for` as `timeout`;
  - map other `TimeoutError`s to `transport_timeout`;
  - split 4xx into request-shape errors caused by policy content (scored as a format failure) and auth or model errors (infra).

### Retries, budget, resume, journal

**V2 [P1][ran] The re-collect retry diffs against the agent-movable `base` tag, not the recorded sha.** `workspace.py:95` reads `attempt.dir / "base_sha"`, the **new** attempt's directory. The sha was written to the previous one (`:110-112`). It falls back to `"base"`, which the M7 audit says the agent can move or delete.
- p2: preserved tree; the agent committed `fib.py` and moved `base` to HEAD; `a1/base_sha` holds the true base. Got `stop recollected, reward 0.0, patch 0 bytes`. The recorded sha gives a 185-byte patch.
- Fix: read `prior[1].parent / "base_sha"`. Treat a missing sha as infra, never as the tag.

**V3 [P1][ran] In train mode every grade-only retry is masked, and the next retry re-runs the agent.**
- A regrade or re-collect attempt has no model calls, so `valid_for_training` fails with "no model calls recorded" (`core/schema/rollout.py:159-160`). `Runner._rollout` then marks `crucible:capture_incomplete` (`run.py:219-220`).
- `_prior` only reuses a previous attempt masked at `verification` (`workspace.py:45`), and `capture_incomplete` is at stage `collect`, so the third attempt re-runs the agent.
- p4, the same flaky perf mask on a1:
  - eval: agent runs 1, then a1 masked and a2 `regraded` 1.0;
  - train: **agent runs 2**, then a1 masked, a2 `regraded` masked `capture_incomplete`, and a3 a fresh `agent_done` 1.0.
- The measured solution is silently replaced by a new sample, which DESIGN §10.1 forbids.
- Fix: a regrade or re-collect attempt inherits the token record (or a pointer to it) from the attempt whose work it grades, and `_rollout` builds the rollout from that attempt's calls.

**V4 [P1][ran] The budget cap is checked against reservations of 0, so concurrent attempts overspend it.**
- `PolicyRef.estimate_usd` defaults to 0.0 (`native.py:94`), and `reserve` only refuses once the cap has **already** been passed (`budget.py:51`). `settle` accepts any amount (`:56-59`).
- p3, cap $1.00, cost $0.60 per attempt, concurrency 4:
  - estimate 0 gives 5 attempts and **$3.00 spent**;
  - estimate 0.6 gives 1 attempt, $0.60.
- A single attempt that settles $5.00 under a $1.00 cap is accepted.
- `over_uncertain_tolerance()` is never called: p3 ran with uncertain spend at 100% of spend, and `stopped: None`. `release()` is never called, and `Failure.delivery` is unused.
- Fix:
  - require a positive estimate (or hold `cap / concurrency`);
  - check `over_uncertain_tolerance` in `stop_check`;
  - release reservations whose failure is `delivery = not_sent`;
  - give the in-attempt LLM client the remaining sample budget, so one attempt cannot exceed it.

**V8 [P1][ran] A resumed attempt re-charges every model call restored from the checkpoint.** `LLMToolPolicy.state_dict` carries `model_calls` (`policy.py:310`). After `load_state_dict`, `res.calls` holds them, and `NativeHarness` sums `cost_usd` over all of them (`native.py:301-303`).
- p6, crash at call 150, checkpoint every 50 steps: the checkpoint carries **104 calls** at $0.01 each; a2's record has 260 calls and `cost_usd 2.6`, so $1.04 of that was already spent by a1.
- a1 itself records `cost_usd 0.0`: a masked agent-stage attempt has no `res.calls`, and `PolicyRef.spend` is unset.
- The xl gate shows the same pattern: `model_calls` 7,627 in attempt 2 covers the whole episode.
- Fix:
  - charge only calls whose `call_id` is not in the checkpoint;
  - for a masked attempt, charge the policies' `model_calls` (they exist even when `save_episode` is skipped), and write `calls.json` for masked attempts too.

**V9 [P1][ran] Resume works once: a second crash, or a real process crash, cannot resume.**
- `_resumable` reopens `prev/work` (`native.py:114`, `232`). After one resume, the World of a2 lives in `a1/work`, but a3 opens `a2/work`.
- p7, crashes at a1 call 150 and a2 call 90:
  - a2 gives `resumed_from a1@step100`;
  - a3 gives `agent_run_error StoreError: missing blob sha256:…`, masked, unknown;
  - the sample ends masked after 3 attempts.
- A real process death leaves no `record.json`. `next_attempt` moves the directory to `aborted/` (`journal.py:103-107`), and `_resumable` requires `record.json` (`native.py:117`).
- p6(b): after a kill at step 150 with a checkpoint at step 100, the re-run gives `a2 resumed_from: None` and 266 fresh calls. That is the "dies at step 3,000" case `checkpoint.py:3-4` names as the motivation.
- Fix:
  - record the World's work dir in the checkpoint and reopen that;
  - let `_resumable` resume from `aborted/a<n>/checkpoint.json` when no record exists;
  - add checkpoint identity (spec hash, code version, blob digests), as the improvements doc asks.

**V10 [P2][ran] A torn tail in `results.jsonl` swallows the next row.** `_append` writes without making sure the file ends in a newline (`journal.py:29-33`). The next record is concatenated to the fragment, and `_read_jsonl` drops the joined line (`:43-46`).
- p9, with a half-written terminal row:
  - the second run re-runs the harness and still reads 0 terminal rows;
  - the third run re-runs it again.
- Each re-run re-samples a sample that already had a result. A crash between `record.json` and the results append also re-runs the sample, because nothing reconciles from `record.json`.
- Fix: before appending, truncate a tail that does not end with `\n`. On resume, rebuild terminal rows from `record.json` files that have no results row.

**V11 [P1][ran] A harness or rollout exception aborts the whole run** (research item, still open).
- p3: a harness raising `RuntimeError` escapes `Runner.run`. The sibling sample's result is lost, `results.jsonl` is never written, and the ledger op stays `reserved`.
- The same path covers:
  - `save_episode`, `rule_flags` and `format_errors_by_call` (outside any boundary, `native.py:257-263`, `workspace.py:118`);
  - `_rollout` parsing a trajectory (`run.py:209`);
  - `_base_sha`/`task.workspace` (`workspace.py:110`);
  - `RLMPolicy.aclose` (`rlm.py:121-123`).
- Fix: wrap `harness.run` and `_rollout` per attempt. Settle the op as uncertain, record a masked `agent_run_error` with blame `harness`, and continue.

### Records and secrets

**V12 [P0][ran] Provider error text, including any credential it echoes, is stored verbatim in records and traces.**
- `LLMClient` wraps `str(e)` (`llm.py:282`, `290-305`). `LLMToolPolicy` puts it into `Done(reason=f"policy error: {e}")` (`policy.py:381`) and into `self.error`, which becomes `Failure.reason` (`native.py:268`, `infra.py:57`).
- p10, a synthetic `sk-ant-api03-<random>` inside an upstream error: the canary appears in **14 files**, including `failures.jsonl`, `results.jsonl`, each attempt's `record.json`, `trace.json`, `trace.jsonl` and `trajectory.agent.json`.
- httpx `HTTPStatusError` messages include the request URL (`llm.py:366-371`), so a query-string key (`?key=`) reaches records the same way.
- Nothing at the record boundary calls the existing scrubbers (`world/leaks.scrub_text`, `connectors/redact.py`).
- Fix: scrub `Failure.reason`, `Done.reason`, journal events and `calls.json` with the M3 redaction patterns before writing them. Never include URLs or headers in `LLMError` text.

**V16 [P2][read] The derived seed is recorded but never used** (L9). k samples differ only through `sample_key = attempt.id`, so a retried sample gets a new cache key and a different completion. In replay mode the retry misses the cache and is masked `provider_unavailable`. Fix: pass `sk.seed` to the policy (endpoint `seed` at temperature > 0) and key the cache on the sample id, not the attempt id.

### Environment results and edge cases

**V13 [P1][ran] `ControlTask.score(nan)` is 1.0.** `max(0.0, min(1.0, (ret - lo) / (hi - lo)))` returns 1.0 for NaN (`control_task.py:54-56`), and the gate reads `pass` because `s > 0` is checked on the clipped value. Actions are validated as finite (`core/schema/spaces.py:112`), but a simulator that blows up under in-range actions (MuJoCo instability) or a remote environment that returns NaN gets full reward. Fix: a non-finite return is masked as `verifier_error` (environment fault), or scored 0 when the episode's own actions caused it.

**V14 [P2][read] `_run_steps` resets the World when resuming a multi-step task.** `checkpoint_every` with `task.steps` restores the checkpoint (`native.py:244-245`). `_run_steps` then calls `world.reset()` at step 0 (`:169-170`) and never checkpoints (the `Checkpointer` is only built in the single-episode branch, `:250`). Fix: refuse `checkpoint_every` for multi-step tasks, or checkpoint per step and skip the reset on resume.

**V15 [P2][ran] The self-matching lint misses combined flags and flags the safe form.**
- `SELF_MATCH` (`remote.py:47`) flags `pkill -f foo`, `pkill -9 -f foo` and `pkill -f -9 foo`.
- It misses `pgrep -af foo`, `pgrep -fl foo` and `pkill --full foo`. The DESIGN B.10 incident was a `pgrep -f` liveness check, and `-af` is its common form.
- It flags the documented safe form `pkill -f "[f]oo"`: the optional quote backtracks.
- `control()` treats rc 255 as a self-kill (`:32`), but 255 is also an ordinary ssh connection failure.
- Fix: parse the argv (any short-flag cluster containing `f`, or `--full`); match the bracket inside quotes; treat 255 as `transport_unreachable`.

**V17 [P2][read] Smaller gaps.**
- A missing imported patch is `capture_incomplete` (infra, retried 3 times; `imported.py:33-34`), not a configuration error.
- `GpuHost.get` ignores rsync failures (`gpu.py:69`). `GpuHost.path` over ssh builds `/<workroot>` when `echo $HOME` fails (`:52-53`).
- `preflight` accepts a host if **any** GPU matches the declared arch (`gpu.py:81-82`).
- `RunSpec.identity_doc` hashes task ids, not content (`run.py:54`).
- `ImportedHarness` stamps `imported_at` into timings, so records are not byte-reproducible (`imported.py:39`).
- `verify/collect.collect` still defaults `base="HEAD"` and ignores `git diff`'s return code (`verify/collect.py:46`, `69`).

## 3. Coverage

Run from `crucible/` with `COVERAGE_FILE` in the scratchpad (the repo's `.coverage` is untouched):

```
pytest -q -p no:cacheprovider --cov=crucible.runtime tests/unit/{test_runtime,test_runtime_cost,test_runtime_records,test_longhorizon,
  test_single_turn,test_control,test_rlm_harness,test_train_runtime,test_harness_branch,test_multistep,test_m7_audit,test_a2a_harness,
  test_train_control_records,test_linear_capture}.py
118 passed in 61.9 s; TOTAL 1547 stmts, 86%
```

| Module | Cover | Untested risky paths |
|---|---|---|
| `harness/imported.py` | 0% | the whole R1 path (missing patch, regrade from an import) |
| `gpu.py` | 0% | `preflight`, `put`/`get`, `path` over ssh |
| `single_turn.py` | 73% | media and content-part parsing (`135-188`), NeMo option shapes |
| `remote.py` | 76% | the self-kill branch (`35`), `kill_by_pidfile` |
| `infra.py` | 88% | the `Health` stop rules (`167-171`), `boundary`'s `TimeoutError`/`OSError` mapping (`79-84`) |
| `budget.py` | 90% | `release`, `increase_limit`, `over_uncertain_tolerance` (all unused by the runner) |
| `harness/workspace.py` | 90% | re-collect that fails again (`100`), a masked episode (`120-121`), the perf-guard slot (`70-73`) |
| `journal.py` | 91% | torn-tail skip (`45-46`), the aborted-attempt move (`105-107`) |
| `harness/native.py` | 91% | multi-step masked, misbehaviour and broken branches (`183-191`), separate-verifier custody (`131-133`) |
| `perf_guard.py` 92%, `control_task.py` 95%, `record.py` 96%, `rlm.py` 96%, `run.py` 97%, `checkpoint.py` 97%, `collect.py`, `keys.py`, `inject.py`, `base.py` 100% | | |

High line coverage hides the defects above. No test asks whether a **policy** can reach a mask (V1, V5, V6). No test retries more than twice, resumes twice, or runs train mode through a retry (V1, V3, V9). No test runs concurrency against a cap (V4). p1, p3, p4 and p7 are the cheapest regression seeds; each runs in under 15 s offline.

## 4. Offline gates

- **B.5 injections (CPU):** they pass inside the run above. The docker classes (`tests/redteam/test_runtime_docker.py`) were not re-run.
- **Long-horizon xl gate:** passes in 73 s (`pass: true`, reward 1.0, 2 attempts, `resumed_from a1@step3000`, 7,623 steps, 7,627 calls, 4 compactions, 1,114,725 trajectory tokens, drift `[]`, credit sum 1.0). It matches `longhorizon_xl.json` field for field, except `wall_s` (66.1 → 73.0) and `max_context_before_compaction` (102,853 → 102,652). The second difference means the context estimate is not byte-reproducible across runs; the temp-dir path length in observations is the likely cause [read]. Two caveats:
  - the gate only crashes once, in-process, so it cannot see V9;
  - its "7,627 model calls" in attempt 2 includes the calls restored from attempt 1, which is the V8 double count.
- **R1, toy runtime with an LLM, R2, R3:** not re-run (they need the lab GPU, a paid model or Harbor). `m10_r1.json` was compared to its claims only: the gate profile and failing test ids match the recording, and `pass: true`.
