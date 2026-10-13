# M11 study, hunt H1: defects in `interop` (Harbor bridge, canonical identity, replay)

Scope: `C/interop/harbor/` (`task_io`, `verify_dir`, `inplace`, `multi_step`, `atif`, `atif_import`, `trajectories`, `metrics`, `gpu_docker`, `world_backend`, `role_runner`), `C/interop/canonical.py`, `C/interop/replay.py`, and what they rely on: `C/verify/runtime/plan.py` (`write_reward_files`, `flat_reward`), `C/runtime/record.py`, `C/runtime/harness/imported.py`.

Spec: ARCHITECTURE "### M11 `interop`" (contents, exit gate). DESIGN "As built (M10)" Harbor paragraphs (patch mode, in-place mode, A-3), "As built, M11 part 1", "M11 importers ... Harbor multi-step", Appendix B.17. Research `docs/research/m11-m12/05-interop-diversity.md` (§2 identity, §2.4 replay, line 147 `call_id`). Harbor source for comparison: `RL-Environments/06-frameworks-and-env-code/harbor-framework-harbor/src/harbor` (`H/`).

Paths: `C` = `crucible/src/crucible/`. `H` = the Harbor checkout above.

Method: a read-only study on HEAD `af402b4`, using `crucible/.venv` (Python 3.12.9) on macOS. **[ran]** means reproduced with a probe. **[read]** means established from the code alone. Probes are in the session scratchpad at `m11hunt/h1/` (`p1_inplace_forge_mask.py` … `p10_agent_phase.py`, helper `fakeenv.py`). All probes are offline. Container probes use `fakeenv.py`: a fake Harbor environment that runs commands with host bash and maps `/app`, `/tests` and `/logs` to temp dirs. It mirrors Harbor's mounted `/logs` (Harbor makes the log dirs mode 777, `H/environments/base.py:584`). Harbor's own `Verifier`, network plan and `Task` run unchanged. No paid call, GPU, lab host or network was used. No secret was read.

## 1. Requirement ledger

Status: **live** = built and behaves as specified. **partial** = built, but a specified part is missing or a hole was reproduced. **wrong** = contradicts the spec. **missing** = not built.

| # | Requirement (source) | Status | Evidence |
|---|---|---|---|
| L1 | Task dir I/O through Harbor's own `TaskConfig`/`Task`, content hash, report keys Harbor drops (`task_io.py` doc) | partial | Validation and hash are Harbor's. `ignored_keys` flags keys Harbor migrates (`version`, `memory`, `storage`) as dropped (V16, p6) |
| L2 | Patch mode: grade with the task's own verifier, Harbor layout, declared network, limits, GPU; no reward, timeout or stale image masks (DESIGN patch mode) | **wrong** | NaN, inf and string rewards pass as ok; a valid single-key `reward.json` crashes (V5, p2). Raw TOML: `[verifier].env`, `user`, the network baseline and legacy fields are ignored (V6, p2). A `mask.json` from any verifier wins over its reward (V7). 0% unit coverage |
| L3 | In-place mode: Harbor's `Verifier` grades the agent's container under Harbor's phase network plan (DESIGN in-place mode) | **wrong** | No reset of `/logs/verifier` or `/tests` before the verifier: planted rewards and a planted `conftest.py` score 1.0 (V1, p1, p1b). Separate-verifier tasks are accepted and graded in the agent's container (V2, p6). The agent phase is never applied (V3, p10) |
| L4 | Reward scalar rule: `reward`, else single key, else `reward_key`, else mask `ambiguous_reward` | live | `test_harbor_inplace.py` (Docker ran); `scalar_reward` |
| L5 | Missing reward, timeout, unapplicable phase masks, never 0; only infra is masked (DESIGN M10 rule) | partial | Infra cases mask (tests). Policy-caused verifier failures also mask (V4, p1 `mask`) |
| L6 | Multi-step: steps in order, Harbor `_prepare_step`, shared verifier reset, Harbor `min_reward`, `mean`/`final` (DESIGN 2566, `multi_step.py` doc) | live, with edges | Tests pass on Docker. p7: the multi-step grade resets `/logs/verifier`, so the p1 plant scores 0.0 (control for V1). `oracle_actions` crashes on a binary solution file (V17). `_trial_paths` sits outside the `try` (V19) |
| L7 | Separate-verifier steps refused (multi-step) | live | `multi_step.py:71-73`. The single-step class has no such refusal (V2) |
| L8 | ATIF export through Harbor's models, `crucible_seq` join keys (`atif.py`) | live | `test_runtime_records.py`, gate test. Tool failures are exported as `extra.ok/error` only (V15) |
| L9 | ATIF import maps every step and result (`atif_import.py`) | partial | The importer reads only `tool_result_is_error`, so Crucible-exported failures come back clean. Image parts are dropped with no marker (V15, p8) |
| L10 | Trajectories join (calls, ATIF steps, rows) | live | `trajectories.py` 95% covered; M10 audit covered `align`/`cross_check` callers |
| L11 | Harbor job metric that honours the mask (`metrics.py`, DESIGN §4.3) | partial | Masks are not 0. A graded single-key row (`{"score": 1}`) and a bool reward count as masked, unlike Harbor's `Mean` and `scalar_reward` (V12, p3) |
| L12 | GPU Docker overlay (`gpu_docker.py`) | live | 84% covered; Docker tests pass |
| L13 | Harbor World backend, remote daemon via `DOCKER_HOST` (`world_backend.py`) | partial | Process-wide `DOCKER_HOST` is cleared by the first of two concurrent Worlds (V8, p5) |
| L14 | Forge roles as Harbor wrapper tasks (`role_runner.py`) | live | `test_role_runner.py` passes, 90% covered. Harbor templatizes sensitive `AgentConfig.env` on dump (`H/models/trial/config.py:156-163`) [read] |
| L15 | Canonical JSON, "RFC 8785 style" (`canonical.py`, research §2 line 86) | partial | Not JCS: `1.0` stays `1.0`, keys sort by code point, not UTF-16. `canonical_args` NFC-normalizes the values replay re-issues (V14, p3, p9) |
| L16 | `call_id = H(episode_key, model_call_ordinal)` (research line 147; DESIGN "derived call_ids agree") | **wrong** | Replay derives ids from the progress-row ordinal, one per action plus the final `done`. Native records use `agent:cN` (`policy.py:384`) (V10, p4) |
| L17 | Replay gives runtime-independent grades (research §2.4) | partial | A truncated or unmapped transcript is graded as a real score (V9, p4). Temp dirs are never removed (V18) |
| L18 | Exit gate: identical `reward.json` and `steps.jsonl` through Harbor, agentenv, verifiers, NeMo-Gym, OpenEnv, on task02 and one toy task (ARCH M11) | partial | Toy only; no task02 part. The "native" leg is a replay of the native trace, so the native harness's own `steps.jsonl` is never compared. `reward_doc` returns `None` for both "masked" and "nothing written", and `same(None, None)` is true (V13, p3) |
| L19 | A masked grade travels as a mask, never 0 (B.17) | live | `write_reward_files` writes no reward file when masked; `test_masked_grade_travels_as_mask_never_zero` passes |
| L20 | Unique task identity for imported Harbor tasks (journal, World id) | **wrong** | `id = "harbor/<short_name>"`: `org-one/hello` and `org-two/hello` share one id and one World id (V11, p9) |
| L21 | Imported harness (`imported.py`) | live | M10 audit fix holds (`harness_unsupported`, non-retryable). It grades through `HarborDirTask` in R1, so V5-V7 apply to it |
| L22 | Rollout records (`record.py`) | live | `test_runtime_records.py` passes |

## 2. Defect candidates

Severity: **P0** = silently wrong reward or score, a mask scored as 0 or a 0 as a mask, a leaked answer, or a cross-runtime mismatch in reward. **P1** = a wrong classification or result, or a crash, in normal use. **P2** = an edge case.

### Rewards the policy can forge or escape (in-place Harbor tasks)

**V1 [P0][ran] Single-step in-place grading reads files the policy planted in `/logs/verifier` and `/tests`.**
- `HarborInPlaceTask.grade` (`C/interop/harbor/inplace.py:190-213`) runs Harbor's `Verifier` in the agent's container without emptying `/logs/verifier` or `/tests` first. Harbor's `Verifier.verify` reads `reward.json` before `reward.txt` (`H/verifier/verifier.py:257-260`). The multi-step path does reset both (`multi_step.py:181`, Harbor `_reset_shared_step_verifier_dirs`).
- The World's `grader_paths` intercept is a regex on the literal `/logs/verifier` and `/solution` (`C/world/intercept.py:56`). It does not cover `/tests`, and any indirection passes.
- p1, a task whose `test.sh` writes `reward.txt` 1 or 0:
  - no-op 0.0, solve 1.0 (controls);
  - `cd /logs && echo '{"reward": 1}' > verifier/reward.json`: intercept none, **reward 1.0, ok**.
- p1b, a task whose `test.sh` runs `pytest /tests/test_outputs.py`:
  - no-op 0.0, solve 1.0;
  - the policy writes `/tests/conftest.py` with a `pytest_runtest_makereport` hook that marks every test passed: intercept none, **reward 1.0, ok**.
- p7 (control): the same plant on a multi-step grade scores 0.0, because that path empties the dirs.
- Fix:
  - call `_reset_shared_dirs` (empty `/logs/verifier` with chmod, empty `/tests`) at the start of `grade`, as `_grade_step` does;
  - add `/tests` to the intercept, and match on normalized paths after `cd`/variables, or drop the regex as a defense and rely on the reset.

**V2 [P0][ran] A task whose verifier runs in a separate environment is graded in the agent's container.**
- `HarborMultiStepTask` refuses separate-verifier steps (`multi_step.py:71-73`). `HarborInPlaceTask.__init__` does not check `resolve_task_verifier_mode`. `grade` always runs the shared `Verifier` in the agent's container. `[[verifier.collect]]` hooks and artifact upload never run.
- p6, a task with `environment_mode = "separate"`, a collect hook that copies `/app/answer.txt` to `/logs/artifacts/`, and a `test.sh` that reads the artifact:
  - the task loads; `load_harbor_task` even reports `verifier_mode separate`;
  - no-op 0.0; **solve 0.0** (the hook never ran, so the artifact is missing); **plant 1.0**.
- Harbor would run the verifier in a fresh container of `[verifier].environment` (or a copy of `[environment]`), after the collect hooks, with only the declared artifacts (`H/trial/trial.py:885-925`).
- Fix: refuse separate mode in `HarborInPlaceTask.__init__`, as the multi-step class does, until a separate-verifier grade is built on Harbor's `_run_separate_verifier` path (collect hooks, artifacts, fresh environment).

**V3 [P0][ran] The agent's network phase is never applied for single-step in-place tasks.**
- The World's network is the environment baseline (`_world_network`, `inplace.py:131-134`). The agent phase is only registered with the backend (`:118-120`). Nothing applies it: `grade` switches only to the verifier phase. `_setup_step` does apply it, but only for multi-step tasks (`multi_step.py:145`).
- For a public baseline the spec also sets `intercept: ["grader_paths"]` (`inplace.py:147`), which turns off the `remote_code_fetch` rule.
- p10: `[agent] network_mode = "no-network"` on a public baseline gives agent phase `no-network`, World network **public**, and no code path that applies the phase.
- Effect: a task that forbids the agent the internet (so it cannot fetch answers or upstream fixes) runs the agent with full internet and no fetch intercept.
- Fix: after `w.reset()` in `open`, apply `plan.agent_phase` when it differs from the baseline (fail closed when the environment lacks dynamic policy). Or build the World with the agent phase as its network and switch to the verifier phase in `grade`. Keep `remote_code_fetch` unless the agent phase is public.

**V4 [P0][ran] Verifier failures the policy causes in a shared container are masked, not scored 0.**
- `inplace.grade` masks every exception (`:216-217`); `_grade_step` does the same (`multi_step.py:189-190`). Harbor's `RewardFileNotFoundError`, `RewardFileEmptyError` and parse errors are often the policy's doing in a shared container.
- p1 `mask`: `d=/logs; mkdir -p $d/verifier/reward.txt` gives **masked `IsADirectoryError`**. The control (no-op) scores 0.0.
- Multi-step resets `/logs/verifier`, so this route is closed there. A policy that breaks the interpreter or shell the tests need (`mv /bin/bash`, a full disk) still ends with no reward and a mask [read].
- Under GRPO a masked sample leaves its group: the exploit DESIGN M10 ("only policy failures are scored") forbids. This is the M10 V1 class for Harbor tasks.
- Fix:
  - run a trusted probe before the tests (as root: write and read `/logs/verifier/.probe`, run `true` through the verifier user);
  - if the probe fails after the policy ran, it is `crucible:agent_broke_env` (scored 0);
  - mask only typed infra errors (`AddTestsDirError`, `DownloadVerifierDirError`, timeouts, environment errors) and reward-file errors whose probe also failed before the agent ran.

### Patch mode (`verify_dir.HarborDirTask`)

**V5 [P0][ran] Patch mode accepts non-finite and non-numeric rewards, and crashes on valid ones.**
- `grade_patch` parses rewards itself (`verify_dir.py:125-132`) instead of using Harbor's rules (`H/verifier/verifier.py:67-122`, which reject non-numeric and non-finite values).
- p2 results:
  - `reward.txt` `nan` → **reward nan, ok**; `{"reward": NaN}` → nan, ok; `{"reward": "inf"}` → **inf, ok**; `{"reward": "1"}` → 1.0, ok (Harbor masks it);
  - `{"score": 1.0}` (valid in Harbor, single key) → `TypeError`; `[1]` → `AttributeError`; `{` → `JSONDecodeError`. The workspace boundary turns these into masks with `unknown` blame;
  - `reward.txt` `1\n0` → 0.0 (Harbor rejects it).
- A NaN reaches `res.reward` (`runtime/harness/workspace.py:107`) and poisons every mean. `ImportedHarness` (R1) grades through this path.
- Fix: parse with Harbor's `Verifier._parse_reward_json` / `_parse_reward_text` rules, then `scalar_reward`. Non-finite or non-numeric → masked `verifier_error`. Parse errors → masked with a typed kind, never a raise.

**V6 [P1][ran] Patch mode reads raw TOML and ignores verifier env, user and the network baseline.**
- `HarborDirTask.__init__` uses `tomllib` (`verify_dir.py:53-66`), not `TaskConfig`. p2 argv for a task with `[verifier].env`, `user = "tester"`, `[environment] network_mode = "public"`, `memory = "2G"`:
  - `--network none` (Harbor's verifier phase defaults to the environment baseline, here public);
  - no `-e` (Harbor merges `[verifier].env` into the test run, `H/verifier/verifier.py:189-203`);
  - no `--user`; no `--memory` (Harbor migrates `memory` to `memory_mb`).
- `allow_internet`, `version` and a separate `[verifier.environment].docker_image` are also ignored [read].
- A test script that needs its env or network then fails or writes 0. That is a wrong reward, not a mask.
- Fix: load with `TaskConfig`; take the network from `resolve_trial_network_plan`; pass the merged verifier env (`resolve_env_vars`) as `-e`; pass `--user`; use `[verifier.environment]` when set.

**V7 [P1][ran] Any verifier's `mask.json` overrides its reward.**
- `grade_patch` checks `mask.json` before the reward (`verify_dir.py:122-124`). p2: `mask.json` + `reward.txt 0` → masked `check_infra`.
- Only the Crucible kit writes `mask.json` from a trusted, privilege-dropped process. In a foreign `test.sh` the code under test often runs as root with `/logs/verifier` writable, so the policy's code can mask its own 0 [read].
- Fix: honour `mask.json` only when the task declares the Crucible kit (a marker in `task.toml` or the kit's `crucible_reward.json` written beside it), else ignore it.

### Runtime plumbing

**V8 [P1][ran] Two Harbor-backend Worlds on one remote daemon break each other's `DOCKER_HOST`.**
- `provision` sets the process-wide `os.environ["DOCKER_HOST"]` and `teardown` pops it (`world_backend.py:165-170`, `269-273`).
- p5: A and B both on `ssh://lab`. After A's teardown, `DOCKER_HOST` is **unset while B is alive**. B's later compose calls (exec, stop) go to the local daemon. After B's teardown it is set again to `ssh://lab` (B restores A's value), so the process stays pointed at the lab.
- Fix: never mutate `os.environ`. Pass `DOCKER_HOST` per call (Harbor's compose subprocess env), or reference-count the setting under a lock.

**V9 [P1][ran] Replay scores truncated or unmapped transcripts as real results.**
- `replay` runs at most `max_steps=500` actions (`replay.py:88`) and grades whatever state results. Unknown tool names become tool errors (`max_tool_errors=None`) and are graded.
- p4 (toy helpdesk, gold transcript):
  - full: 1.0;
  - one action over `max_steps`: **0.0, not masked**;
  - an unmapped tool name (what `from_atif` returns with its default `known=()` for a bare name such as `issue_refund`): **0.0, not masked**.
- Long foreign transcripts (hundreds of terminal calls) and foreign tool names are the normal case for Harbor imports. The reward then differs by runtime.
- Fix: mask `crucible:replay_incomplete` when fewer actions executed than the transcript holds, or when any action hit an unknown tool. Make `known` required in the parsers.

**V10 [P1][ran] Replay `call_id`s are per action row, not per model call.**
- `replay.py:97` uses `call_id(episode_key, i)` with `i` the progress-row index (every action plus the final `done`). The research spec is `H(episode_key, model_call_ordinal)`, with the ordinal counting sampled assistant messages (05 line 147). Native records use `agent:c1`, `agent:c2` (`C/actors/policy.py:384`).
- p4: nine rows, nine distinct ids, including one for `done`. A call that emitted two tool calls gets two ids in replay and one in the native record, so per-call credit cannot be joined across runtimes [read].
- Fix: keep the call ordinal on each `Action` (one ATIF agent step, one Responses turn, one gateway model call), give every action of a call the same id, give `done` none, and derive native ids with the same function.

**V11 [P1][ran] Harbor tasks from different orgs with the same short name share one task id.**
- `inplace.py:103-104` uses `task.short_name`, which drops the org (`H/models/task/task.py:89-93`).
- p9: `org-one/hello` and `org-two/hello` → both `harbor/hello`, both World id `harbor:hello`. Journals, resume and results key on the task id.
- Fix: use the full `task.name`, and add the content hash for the World id.

### Edges

**V12 [P2][ran] `masked_mean` masks graded single-key and bool rewards.** `metrics.py:36-37`, `49-50` read only `reward`. p3: rows `{"score": 1}`, `{"score": 0}`, `null` → Harbor `Mean` 0.333; `scalar_reward` accepts the single key; `masked_mean` → `None`, mask rate 1.0. `{"reward": true}` is masked; Harbor counts it as 1. Fix: apply `scalar_reward`'s rule in both the function and `MASKED_MEAN_PY`; count ambiguous rows apart from masks.

**V13 [P2][ran] The identity gate cannot tell "masked" from "nothing written".** `reward_doc` returns `None` when `reward.json` is absent (`canonical.py:66-69`). p3: runtime A masked (`mask.json`), runtime B wrote nothing; `same(...)` is true. Fix: compare `crucible_reward.json` and `mask.json`; fail when a runtime wrote neither. Also add a leg that compares the native harness's own `steps.jsonl`, and the task02 leg the exit gate names.

**V14 [P2][ran] `jcs` is not RFC 8785, and `canonical_args` changes the bytes it replays.** p3: `jcs({"r": 1.0})` is `{"r":1.0}` (JCS: `{"r":1}`); keys sort by code point (JCS: UTF-16 units), so `U+E000` and `U+1F600` swap. p9: `canonical_args` turns `café` into `café`, and replay writes those bytes into the World. Fix: call it "Crucible canonical JSON" or implement JCS; NFC-normalize only for comparison, never the arguments replay sends.

**V15 [P2][ran] ATIF round trip loses tool failures and images.** p8: the export writes `extra {ok: false, error: "invalid_arguments"}` (`atif.py:48-49`); the importer reads only `tool_result_is_error` (`atif_import.py:105`), so the imported event has `error: None`. An image content part is dropped (`text_of`, `:35-40`) with no marker in `data`. Fix: read `extra.ok is False` / `extra.error`; record `data.media` (count and types) when parts are dropped.

**V16 [P2][ran] `ignored_keys` reports keys Harbor migrates.** p6: `version`, `environment.memory`, `environment.storage` are reported as dropped, yet Harbor maps them (`memory_mb` = 2048, `schema_version` = 1.0). Fix: skip the legacy keys Harbor's validators consume, or diff against a re-dump of the validated config.

**V17 [P2][ran] `oracle_actions` crashes on a binary solution file.** `multi_step.py:238` calls `read_text()`. p7: a `data.bin` in `steps/one/solution` → `UnicodeDecodeError`. Fix: send binary files base64-encoded through `exec`, or use `oracle_step`.

**V18 [P2][ran] Replay never removes its temp dirs.** `replay.py:81` makes `replay-*` with `mkdtemp`; nothing deletes it. `$TMPDIR` on this machine holds **4,986** of them. Fix: a `TemporaryDirectory` closed in `finally` after `world.close()`.

**V19 [P2][read] Exceptions can escape the grade.** In `inplace.grade` the `finally` calls `set_network_policy(base)` unguarded (`:218-220`), so a failure there replaces the report with a raise. `_trial_paths(world)` runs before the `try` in both grades (`inplace.py:202`, `multi_step.py:177`). Fix: guard the restore as `_grade_step` does, and move `_trial_paths` inside the `try`.

**V20 [P2][read] `export_ref` caches by ref name.** `verify_dir.py:29-33` reuses `dest/<task>@<ref>` whenever `task.toml` exists. A branch ref that moved returns the old grader. A crash mid-extract leaves a partial tree that is reused. Fix: resolve the ref to a commit sha, key the cache by sha, extract to a temp dir and rename.

**V21 [P2][read] A patch-mode timeout leaves the staged files on the host.** On `TimeoutExpired`, `grade_patch` removes the container but returns before `host.exec("rm -rf ...")` (`verify_dir.py:114-119`). Fix: clean up in `finally`.

## 3. Gates re-run offline

- `tests/unit/test_harbor_backend.py`, `test_harbor_inplace.py`, `test_harbor_multistep_import.py`, `test_interop_gate.py`, `test_runtime_records.py`: **25 passed** in 122 s. A local Docker daemon was up, so the Docker-marked in-place and multi-step tests ran too. Nothing was pulled beyond cached images.
- With `test_role_runner.py`, `test_m3_audit_capture.py`, `test_m4_audit.py`, `test_m7_audit.py` added: **97 passed** in 145 s.
- The toy identity gate passes. Its limits are in L18 and V13.

## 4. Coverage of `crucible.interop.harbor`

Run: `crucible/.venv/bin/python -m pytest <tests> --cov=crucible.interop.harbor` with `COVERAGE_FILE` in the scratchpad (the repo `.coverage` was not touched).

| File | Stmts | Miss | Cover (M11 tests) | Cover (+ role, M3, M4, M7 tests) |
|---|---|---|---|---|
| `atif.py` | 40 | 5 | 88% | 88% |
| `atif_import.py` | 102 | 21 | 0% | 79% |
| `gpu_docker.py` | 73 | 12 | 84% | 84% |
| `inplace.py` | 142 | 13 | 91% | 91% |
| `metrics.py` | 13 | 0 | 0% | 100% |
| `multi_step.py` | 155 | 17 | 89% | 89% |
| `role_runner.py` | 128 | 13 | 0% | 90% |
| `task_io.py` | 84 | 12 | 86% | 86% |
| `trajectories.py` | 78 | 4 | 95% | 95% |
| `verify_dir.py` | 107 | 107 | **0%** | **0%** |
| `world_backend.py` | 190 | 46 | 76% | 76% |
| **total** | 1112 | 250 | 59% | **78%** |

Also: `interop/canonical.py` 88%, `interop/replay.py` 91%, `runtime/record.py` 75%. `runtime/harness/imported.py` is never imported by these tests.

`verify_dir.py` (patch mode, R1's grader) has no unit test at all, which is why V5-V7 went unseen. The in-place tests need Docker and cover the happy paths; no test plants a file before grading (V1, V2) or sets an agent network phase (V3).
