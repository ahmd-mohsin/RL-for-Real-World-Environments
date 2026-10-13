# M11 study, hunt H3: defects in `interop` (OpenEnv import and export, AWM, MiMo, EnterpriseOps-Gym)

Scope: `C/interop/openenv/` (`import_env`, `export`, `awm`, `awm_task`, `awm_check`, `catalog`), `C/interop/mimo/` (`importer`, `verify`, `check`, `export`), `C/interop/eog/` (`importer`, `oracle`), and what they use: `C/world/world.py` (choke point, `ends_episode`, `VerifierView.unchanged`), `C/world/apps/sqlite_app.py`, `C/world/state/snippets/sql.py`, `C/verify/sql_checks.py`, `C/_vendor/openenv_awm/`.

Spec: ARCHITECTURE "### M11 `interop`" and the §3 rows tagged M11 (OpenEnv contract and rubric tree, MiMo/verl −999 masking, 267 MiMo envs); DESIGN "As built, M11 part 1" (MiMo round trip), "M11 part 4: OpenEnv import", "E: AWM-1K tasks", "M11 importers: EnterpriseOps-Gym", B.18; `docs/research/m11-m12/05-interop-diversity.md` §2.1 (N1–N4), §3.5, §3.7.

Sources compared: `RL-Environments/06-frameworks-and-env-code/{meta-pytorch-openenv, servicenow-enterpriseops-gym}`, `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss/general/envs` (all 266 `verify.py` files are byte-identical, md5 `e3f0b527…`).

Paths: `C` = `crucible/src/crucible/`. `OE` = `RL-Environments/06-frameworks-and-env-code/meta-pytorch-openenv/`. `EOG` = `…/servicenow-enterpriseops-gym/`. `MM` = `…/xiaomi-mimo-v2.6-rl-oss/general/envs/`.

Method: a read-only study on HEAD `af402b4`. `crucible/.venv` (Python 3.12.9) and `crucible/.envs/openenv` (same Python, OpenEnv installed) on macOS. **[ran]** means reproduced with a probe; **[read]** means established from the code alone. Probes live in the session scratchpad at `m11hunt/h3/` (`p1_openenv.py`, `p1b_openenv_timeout.py`, `p2_export.py`, `p3_mimo.py`, `p4_eog.py`, `p5_awm.py`, `p6_mimo_check.py`, `p7_mimo_unpinned.py`). All probes are offline: synthetic environments, the local AWM cache, the local EOG clone and `gym_dbs.zip`, the local MiMo release. The only "judge" is a localhost fake. Credentials are synthetic canaries built at run time. No Docker, GPU, network or paid call was used. The repo's `.coverage` is untouched.

## 1. Requirement ledger

Status: **live** = built and behaves as specified; **partial** = built, with a reproduced hole or a missing part; **wrong** = contradicts the spec or a design principle; **missing** = not built.

| # | Requirement (source) | Status | Evidence |
|---|---|---|---|
| L1 | OpenEnv App: `factory`, reset kwargs, MCP tools via `ListToolsAction`, Action schema as one `step` tool (DESIGN part 4) | live | `test_openenv_import.py` 12/12 pass (PortSim, grid_world, connect4, maze, wildfire, client mode, images) |
| L2 | Blame: invalid args / unknown tool → policy, execution error → tool, transport / timeout → infra; an exception in `step` masks (DESIGN part 4) | **wrong** (vs P16) | Built as written, but the rule lets a policy mask itself: an out-of-range action that the environment rejects by raising, or a slow tool call (V2) |
| L3 | Restore = rebuild + reset + re-issue, every result digest verified, never silently a different state (DESIGN part 4, B.18) | partial | Step digests omit `metadata`, `reward` and `done` (V1). 20/20 restores of a hidden-state env were accepted; the hidden state differed in 10 |
| L4 | Environment-terminated episodes (`ToolResult.ends_episode`) | live | `world.py:625-629`; test passes. The export pays the final reward again on every later `crucible__finish` (V5) |
| L5 | Reward: last or sum, normalized by `reward_range`; no terminal step = 0, scored (N4) | partial | NaN → 0 and inf → 1, both scored; a terminal step with reward `None` is reported "never terminated" (V3) |
| L6 | Catalogue: PortSim (`catalog.py`) | live | Native = imported rewards (test) |
| L7 | Two-player seats, turn refusal, zero-sum attribution | live | Test passes. The non-zero-sum branch (`import_env.py:552-555`) has no test |
| L8 | Client mode and reconnect replay | partial | Client test passes. The `CONNECTION_LOST` replay and `RemoteActionError` branches (`451-459`) have no test |
| L9 | OpenEnv export: Ψ-telescoping step rewards; masked → `reward None`, `metadata.masked` (DESIGN part 1, research §3.5) | partial | An episode whose World recorded an infra fault is graded and sent as a normal reward (V4) |
| L10 | Export "rubric tree plus `openenv.yaml`" (ARCH M11 contents; research §3.5 `CrucibleRewardRubric`) | missing | No rubric class and no `openenv.yaml` writer under `C/interop/openenv/` (grep) |
| L11 | AWM scenarios as Worlds (M1 gate 2b, `awm_check.py`) | not re-run | `awm_check.py` has 0% coverage; the sweep needs 40 service starts. `test_awm_tasks.py` starts `social_network_8` and passes |
| L12 | AWM tasks: code-mode verifier in AWM's runner; `complete` → 1, `others` → 0, runner error masked (DESIGN "E: AWM") | partial | A policy's final message can push a verifier past its CPU limit, which masks (V16). One verifier is masked forever by its function name (V17) |
| L13 | AWM: freeze `now` (B.18 "AWM sample data uses `datetime('now')`") | missing | 531/1000 scenarios seed rows from the wall clock; two provisions 1 s apart differ (V18) |
| L14 | MiMo import: one App per tools file, writable workspace, verifier files withheld | live | p6: `check_env` passes all 9 checks on env 0000; no verifier file anywhere in the World dir |
| L15 | MiMo verify: fresh grading root; crash or no judge → masked, never 0; items → `check/<id>` (DESIGN part 1, research §3.7) | partial | `judge=None` is not "no judge" when `GA_JUDGE_KEY2/3` exist (V6). `outcome` is always 0 (V7). The scorer path is not pinned (V8). The grading root and log leak (V9) |
| L16 | MiMo byte-identical re-export (ARCH exit gate) | partial | It copies the source directory and compares hashes taken from that same source. Nothing of the World or the run passes through it, so the gate cannot fail on an importer bug. It deletes its source when `out` contains it (V10) |
| L17 | MiMo envs run through the M10 runtime | missing | `mimo.verify.grade` has no caller in `src/` or `packs/`; only `test_interop_gate.py` calls it. There is no MiMo `RunTask`, and `agent_output` (MiMo's chat answer) has no producer |
| L18 | verl-fork dataset export, masked → −999 (ARCH §3 row "−999 / judge-failure masking", research §3.7) | missing | No `C/interop/verl/` |
| L19 | EOG schema inference: union of inserted columns plus verifier columns; unique integer `*_id` → `INTEGER PRIMARY KEY` | partial | No column affinity, so typed comparisons differ from the server (V12). Function and type names become columns (V19) |
| L20 | EOG `now` pinned to the task's t0 (DESIGN "M11 importers") | partial | Pinned while seeding only. `sql_query` / `sql_execute` see the wall clock (V14) |
| L21 | EOG `SqlAssertion` = EnterpriseOps' result reduction and comparisons (`sql_checks.py` docstring) | **wrong** | `gt`/`lt` on text always fail; `equals` coerces numeric strings and uses `isclose` (V13) |
| L22 | EOG refusal mapping: all-zero-count verifiers → hard `unchanged` gate | partial | A deletion task matches the rule and becomes unwinnable (V11). The `unchanged` gate itself compares content fingerprints, so insert-then-delete nets to unchanged [read] |
| L23 | EOG: `response_check` and `tool_execution` verifiers refused | live | `strict=True` is the default and `import_eog_dir` / `import_hf_rows` never pass `strict=False`. With `strict=False` they are dropped silently (V20) |
| L24 | EOG witness scores 1 on derivable tasks; no-op scores 0 | live | Test: 12 tasks import, all no-ops 0, 7 witnesses 1. p4f: no real verifier query errors at the start state (0/12) |
| L25 | SQL tools stay inside the World | **wrong** | `VACUUM INTO` writes a new database file anywhere the host process can write, from `sql_execute` and even from the read-only `sql_query` (V15) |
| L26 | Exit gate (toy): identical `reward.json` / `steps.jsonl` across runtimes; MiMo round trip | live (toy) | `test_interop_gate.py` passes. The MiMo test asserts `status in ("masked", "ok")`, so it accepts any grade; the `ok` branch (where V7 lives) never runs in tests |

## 2. Defect candidates

Severity: **P0** = silently wrong reward; a mask scored 0, or a policy outcome masked; oracle or answer leaked to the agent; restore of a different state; byte non-identity in the MiMo round trip; SQL escape of the sandbox. **P1** = a wrong classification or result in normal use, or nondeterminism. **P2** = an edge case or hygiene.

### OpenEnv import

**V1 [P0][ran] Restore accepts a different environment state: step digests leave out `metadata`, `reward` and `done`.**
- `_obs_text` pops `reward` and `done` (`import_env.py:399-400`) and pops `metadata` unless `show_metadata` (`:404-405`). `_step_digest` hashes only that text plus images (`:433-434`). `_replay` compares only that digest (`:384`).
- Many OpenEnv environments keep their hidden state in `metadata` (that is why `show_metadata` defaults to off). Rewards are not checked at all.
- p1a, `HiddenEnv`: step 1 draws a hidden card into `metadata`, step 2 pays it. Snapshot after step 1, restore: **20/20 restores accepted, hidden card different in 10/20.** The restored branch pays a different reward than the journal's.
- p1a, `RewardNoise`: per-step reward random, text constant. Restore accepted; the journal keeps the old rewards while the live environment produced new ones.
- Fix: digest the full observation for replay checks: text **plus** canonical JSON of `metadata`, `reward` and `done` (keep the policy-visible text separate). Check `done` and the turn after replay. Keep the existing test of a noisy text, and add one with noise in `metadata` only and one in `reward` only.

**V2 [P0][ran] A policy can mask its own episode: an action the environment rejects by raising, or a tool call it makes slow.**
- An exception from `env.step` propagates out of `OpenEnvApp.call` (`import_env.py:418`, `450`) to the choke point, which records `crucible:tool_crashed` and masks (`world.py:477-481`).
- OpenEnv's own environments raise on bad actions: Atari raises `ValueError` for an out-of-range `action_id` (`OE/envs/atari_env/server/atari_environment.py:174-178`), FinRL and dm_control raise on wrong action shapes. The JSON schema the policy sees allows those values.
- p1c: `StrictEnv` (raises like Atari) with action `n=99`: `masked=True`, `crucible:tool_crashed`. Control `n=2`: scored.
- MCP timeouts: OpenEnv returns `error_type="timeout"` when a call exceeds `MCP_TOOL_CALL_TIMEOUT` (30 s, `OE/src/openenv/core/env_server/mcp_environment.py:81,568-575`). The importer maps every type it does not list to infra (`import_env.py:474`).
- p1b: an MCP `solve(horizon)` tool, timeout shortened to 0.5 s. `horizon=2` scored; `horizon=50` **masked** (`crucible:tool_crashed`). Control: a tool that raises (`execution_error`) is scored.
- A do-nothing policy escapes its 0 with one bad action. This is the M10 V1/V6/V7 class (P16) in the importer.
- Fix: wrap `_issue` in `call`. On an exception, rebuild a fresh environment and replay the journal (the digest check already exists). If the replay succeeds, the environment is healthy and the failure is the action's: return `ok=False, error="invalid_action", fault="policy"` and do not journal it. Only a replay that fails is infra. Map `timeout` to policy when the same health replay passes, and record the elapsed time.

**V3 [P1][ran] Non-finite or missing terminal rewards are scored, not masked.**
- `plan.env_reward` normalizes with `min(1, max(0, (r-lo)/(hi-lo)))` (`import_env.py:592`). `max(0.0, nan)` is `0.0`.
- p1b/d: terminal reward `nan` → reward 0.0, infra ok. `inf` → 1.0, infra ok. `None` on a `done` step → 0.0 with the reason "the episode never terminated" (`episode_reward` returns `final_reward`, which is `None`, `:577-580`).
- A NaN or infinite reward is an environment bug, so P16 says mask. A terminal `None` is a real termination with a wrong reason.
- Fix: in `env_reward`, `not math.isfinite(r)` → raise `InfraError("crucible:invalid_reward")`. In `episode_reward`, distinguish "not done" from "done without reward" and report the second honestly (mask, or 0 with its own reason, per task).

### OpenEnv export

**V4 [P0][ran] The exported Environment never reads `World.failures`: an infra-faulted episode goes out as a normal reward.**
- `_final` grades with `self._task.grade(...)` and masks only when the grade itself is masked (`export.py:115-122`). The M10 runner masks on `world.failures`; the export does not.
- p2: an imported environment crashes inside `step` on the first call (`World.failures = ['crucible:tool_crashed']`); the second call ends the episode. The export returns `reward 1.0, metadata.masked False`.
- The same path turns an infra crash that stops the policy into a scored 0 at `crucible__finish`. TRL and OpenEnv's TrainingTrace then train on it (research N3).
- Fix: in `_final`, `masked = grade masked or bool(self._world.failures)`; carry the failure kinds in `metadata["crucible"]`. Also check `world.failures` on every step and end the episode as masked at the first infra fault.

**V5 [P1][ran] After `done`, the export pays the final reward again on every `crucible__finish`; other calls raise.**
- `_step` for `FINISH` skips the World step when the agent is already done, then calls `_final` again (`export.py:98-102`). `_final` never moves `self._psi` to the final value.
- p2: environment-terminated step reward 1.0, then `crucible__finish` 1.0, then again 1.0: **a summing consumer gets 3.0 for a 1.0 episode.** DESIGN part 1 promises that the step rewards sum to the final reward.
- Any other tool call after done raises `WorldError 'agent' already declared done` into the OpenEnv server.
- Two resets without `episode_id` and `seed` share the episode id `openenv|None` (`:74`), so derived `call_id`s collide across episodes.
- Fix: latch the final observation. After done, return it again with `reward=0.0` (or `None`), never re-grade. Return a `done=True` error observation for tool calls after done. Default the episode id to a fresh uuid.

### AWM tasks

**V16 [P0][ran] The policy's final message can push an AWM verifier past its CPU limit, and the runner error masks the episode.**
- `AwmTask.final_answer` is the policy's `Done` text (`awm_task.py:85-87`), passed unbounded to the verifier (`:98`). Any `execution_status != "success"` raises `InfraError` (`:99-100`), which masks.
- 2,380 of 10,010 code verifiers read `final_answer`. p5b sampled 150. A benign answer runs cleanly on all of them. A 5,000-character answer of zeros turns **4/150** into `runner rc -24` (SIGXCPU at the 20 s `RLIMIT_CPU`, regex backtracking in e.g. `fleet_management_6` t8). Each is a masked sample the policy chose.
- Upstream maps every execution error to `judge_error` (`OE/envs/agent_world_model_env/server/verifier.py:220-221`); Crucible chose masking, which a policy can reach.
- Fix: run each verifier once at admission on the initial DB with a benign answer (self-check). For a verifier that passes the self-check, a later execution error caused by the episode's inputs is the policy's (score 0, kind `policy_error`). Cap `final_answer` length (for example 4,000 characters) before the call. Keep the mask for a runner that fails its self-check.

**V17 [P2][ran] One AWM verifier is masked on every sample: the function name is hard-coded.**
- `run_verifier` always asks for `verify_task_completion` (`awm_task.py:40`). Upstream takes the first `def verify_…` line (`OE/…/server/verifier.py:205-214`).
- `habit_tracking_1` task 5 defines `verify_task_complete`. p5a: `execution_status=error, "Function 'verify_task_completion' not found"` → masked forever. 1 of 10,010.
- Fix: the upstream rule; the V16 admission self-check would also reject the task.

**V18 [P2][ran] AWM initial databases depend on the wall clock (B.18 open item).**
- `AwmServiceApp.provision` calls the vendored `create_database` with no `pin_now` (`awm.py:53-56`). 531 of 1,000 scenarios' sample data calls `datetime('now', …)`.
- p5 aside: `booking_and_appointments_1` provisioned twice, 1 s apart: initial fingerprints differ. Grading within an episode is consistent (the same `initial.db` copy), but replays, restores across runs and relative-date tasks drift.
- Fix: open the connection in `create_database` through `sqlite_app.pin_now` at a fixed epoch (the task's or a dataset constant).

### MiMo

**V6 [P1][ran] `judge=None` is not "no judge": MiMo's verifier also reads `GA_JUDGE_KEY2/3`, which `grade` neither clears nor restores.**
- `grade` saves five variables and pops only `GA_JUDGE_KEY` when no judge is given (`mimo/verify.py:58`, `63-65`). MiMo's `_judge_chain_sc` reads `GA_JUDGE_KEY`, `KEY2`, `KEY3` and their `MODEL/API/URL` (`MM/*/verify.py:63-80`).
- p3a: `GA_JUDGE_KEY2` (a canary) and `GA_JUDGE_URL2` (a localhost fake) in the environment, `grade(judge=None)`: **5 judge requests carried the canary bearer**, and the grade came back `infra ok, reward 1.0`. DESIGN says "without one, MiMo masks".
- With a judge given, `KEY2/3` silently join the chain as failover, so a grade can come from a different model than the one configured.
- MiMo prints `key[:12]` of every chain entry. The scrub covers only `judge["key"]` and `sk-…` (`:78-80`). p3c: the canary prefix stays in `verify.log`.
- No `GA_JUDGE_*` variable is in the repo's `.env` files today (names checked, values not read), so this needs a MiMo-style environment to bite. It is then a paid call and an unvetted grade.
- Fix: before the import, clear every `GA_JUDGE_*`, `GRADE_*`, `VERIFY_*` variable and set only the configured judge. Restore all of them after. Scrub every key that was set, not just one. Better: run `verify.py` in a subprocess with an explicit environment (also fixes V8 and V21).

**V7 [P1][ran] MiMo `outcome` is always 0: it reads `strict_pass`, which MiMo's verifier never returns.**
- `outcome = float(bool(res.get("strict_pass")))` and the `mimo` dict read `strict_pass`, `deterministic_score`, `judge_score`, `judge_all_passed` (`mimo/verify.py:93-94`). MiMo's `verify()` returns `reward`, `score`, `detail`, `results` and maybe `reward_error` (`MM/*/verify.py:458-469`). None of the 266 `verify.py` files contains `strict_pass` (grep).
- p3b: all 6 components pass and reward is 1.0, yet `outcome 0.0` and every `mimo` field is `None`.
- `outcome` is the success bit for pass@k, acceptance and credit composition. Every MiMo env reads as never solved.
- Fix: `outcome = float(bool((res.get("detail") or {}).get("passed")))`; fill the `mimo` dict from `detail` (`passed`, `gate_failed`, `by_method`). Add a test of the `ok` branch (lines 91-93 have no coverage).

**V8 [P1][ran] The MiMo scorer is not pinned: whatever `synthesis` package is importable decides the grade.**
- MiMo's `_grade` prefers `synthesis.office_gen.grade_rubric` when it imports (`MM/*/verify.py:408-417`). Research §3.7: "Import must pin the self-contained path." `grade` execs `verify.py` in the Crucible process with its `sys.path` (`mimo/verify.py:68-74`).
- p7: a no-op run, no judge: masked `judge_failed` on a plain path. With a stand-in package named `synthesis` on `sys.path`: **`infra ok, reward 1.0`.**
- Fix: run `verify.py` in a subprocess with `python -I` and a `PYTHONPATH` of the grading root only, or install a `sys.modules["synthesis"] = None` blocker for the duration of the in-process call.

**V9 [P2][ran] The MiMo grading root and its log are left in the temp dir and never reported.**
- `g = mkdtemp("mimo-grade-")` holds a full env copy (verifier included), the run's DBs and workspace, and `verify.log` (`mimo/verify.py:51`, `80`). Nothing deletes it and the reward document does not name it.
- p3c: one `mimo-grade-*` directory per grade remains; its path is not in the reward document.
- Fix: grade under a caller-supplied work dir (as the other tasks' `grade(world, workdir)` do), return the log path in the reward detail, and clean up.

**V10 [P2][ran] The MiMo export deletes its own source when the output dir contains it, and it tests nothing of the importer.**
- `export` runs `rmtree(out)` and then `copytree(env.path, out)` (`mimo/export.py:16-18`). p: `export(env, env.path.parent)` → `FileNotFoundError`, **source env gone**.
- The "byte-identical" check compares a fresh copy of `env.path` with hashes taken from `env.path` at import. It does not read the WorldSpec, the bundle or any run state, so the ARCH gate cannot fail on an import bug (ledger L16).
- Fix: refuse an `out` that equals, contains or lies inside `env.path`. If the gate is meant to prove a round trip, rebuild the env from the WorldSpec plus the bundle (the files Crucible actually serves) and compare that.

**V21 [P2][read] MiMo grade hygiene.**
- It mutates `os.environ` for the whole process (`mimo/verify.py:58-88`); two concurrent grades with different judges race.
- It is synchronous and can block for minutes (rule subprocesses up to 300 s each, judge calls), which would stall an asyncio runner.
- A `verify()` that returns a non-dict raises `TypeError` at `"error" in res` (`:87`), outside the try.
- A NaN `score` passes straight into `reward` (`:91`).
- Fix: the subprocess runner of V6/V8, with a timeout that masks, a dict check, and `math.isfinite`.

### EnterpriseOps-Gym

**V11 [P0][ran] A deletion task is imported as a refusal task and becomes unwinnable.**
- `zero` is true when every `database_state` verifier is `COUNT(*) … = 0` (`eog/importer.py:352-354`); such a task gets the hard `unchanged` gate (`:355`, `242-250`) and a `Done` witness (`:273-274`). A request to delete or close-out a record has exactly this shape.
- p4a: seed holds case 1; the only verifier is `SELECT COUNT(*) FROM customer_case WHERE case_id = 1` expecting 0. Imported as a refusal. **Correct delete: 0** (`unchanged` fails). No-op: 0 (`eog/state` fails). Derived witness: 0.
- None of the 13 local tasks has this shape; the HF dataset is larger and was not checked offline.
- Fix: at import, run the zero-count verifiers on the seeded database. A refusal needs all of them to pass already at the seed. Otherwise the task is a normal task (and its witness is the DELETE that `derive` already builds).

**V12 [P1][ran] The inferred schema has no column types, so the same write grades differently from the real server.**
- `infer_schema` emits bare column names (`eog/importer.py:188-197`). A typeless column keeps `'427'` as text, and SQLite then compares `'427' = 427` as false. The benchmark's servers declare typed columns, which coerce on write.
- p4b: the CSM witness scores 1.0; **the same witness with integer parameters sent as JSON strings scores 0.0** (`link_new_case_sla` checks fail). Direct: `COUNT(assigned_to = 427)` is 0 on a typeless column and 1 on an `INTEGER` one.
- LLM policies often send numbers as strings, so this silently lowers rewards.
- Fix: infer an affinity per column from the seed values (all integers → `INTEGER`, numeric → `REAL`, else `TEXT`) and declare it. Re-run the 12-task gate.

**V13 [P1][ran] `SqlAssertion` comparisons differ from EnterpriseOps' `_compare_values`.**
- EnterpriseOps uses plain Python `==`, `>`, `<`, `in` (`EOG/benchmark/verifier.py:374-399`). `sql_checks.compare` converts to numbers for `gt`/`lt` and returns false when that fails, and `equals` coerces numeric strings and uses `isclose` (`verify/sql_checks.py:57-73`).
- p4c: `'2026-01-02' gt '2025-12-31'`: Crucible false, EnterpriseOps true. `'b' lt 'c'`: false vs true. `'1233' equals 1233`: true vs false. `1.0000000001 equals 1`: true vs false.
- A `greater_than` check on a date column can never pass, so the task scores 0 for every policy. The local tasks use only `equals`; HF rows were not checked.
- Fix: give `from_enterpriseops` assertions EnterpriseOps' semantics (a `semantics="eog"` field that `compare` honours), including its `contains` (`expected in str(actual)`, false when `expected` is not text).

**V14 [P1][ran] At run time the SQL tools see the wall clock, not the task's pinned `now`.**
- `pin_now` is applied only while seeding (`sqlite_app.py:92-93`). `tool_sql_query` / `tool_sql_execute` open plain connections (`:181-200`).
- p4d: World clock t0 `2025-12-05T15:33:30Z`; the policy's `SELECT datetime('now')` returns `2026-10-13 03:27:25`. The same `UPDATE … datetime('now')` run twice 1 s apart gives **different final fingerprints**.
- The World contradicts itself (the instruction's day versus the database's). Replays and cross-runtime identity checks drift, and relative-date verifiers depend on the day the run happens.
- Fix: pin `now` to the World clock in both SQL snippets (pass the clock's current time in the snippet arguments and call `pin_now`).

**V15 [P0][ran] The SQL tools write files outside the World: `VACUUM INTO` from `sql_execute`, and even from the read-only `sql_query`.**
- `sql_execute` runs any single statement on a plain connection (`sqlite_app.py:188-200`). The query snippet opens `mode=ro` with `PRAGMA query_only` (`state/snippets/sql.py:126-130`), but `VACUUM INTO` creates its target before failing. EOG Worlds use the local backend, so the snippet runs on the host.
- p4e: `VACUUM INTO '<scratch>/outside/via_execute.db'` → ok, a 1.2 MB database file **outside the World dir**. `sql_query` with `VACUUM INTO` → error "attempt to write a readonly database", **but the file was still created**.
- A policy can create files anywhere the Crucible process can write: other Worlds' directories, grading roots, run stores, or the disk until it fills.
- Fix: install a `sqlite3` authorizer on both tool connections. Deny `SQLITE_ATTACH`, `SQLITE_DETACH`, `VACUUM` and writable `PRAGMA`s. On `sql_query`, allow only `SELECT`, `READ` and `FUNCTION`. Keep `load_extension` disabled. Add a red-team test with `VACUUM INTO` and `ATTACH`.

**V19 [P2][ran] `referenced_columns` turns function and type names into columns and misses unqualified columns in joins.**
- The single-table branch keeps every identifier that is not in `_SQL_WORDS` (`eog/importer.py:143-148`). p: `julianday(...)`, `CAST(x AS INTEGER)`, `IFNULL(...)` add columns `julianday`, `CAST`, `INTEGER`, `IFNULL`. In a join, `WHERE stage = 'x'` adds `stage` to no table.
- p4f: no real verifier query errors at the start state (0/12), so no local task is unwinnable from this today.
- Fix: drop identifiers followed by `(` and the type names after `AS` inside `CAST`. For joins, give an unqualified column to the one joined table whose seed holds it, else report it at import.

**V20 [P2][read] `strict=False` and import errors.**
- With `strict=False`, `response_check` and `tool_execution` verifiers are dropped and recorded only in `meta.unported_verifiers` (`eog/importer.py:298`, `305-307`). The reward is then 1 on the ported subset, while EnterpriseOps requires every verifier.
- A task whose verifiers are all unported imports with `refusal=False` and no assertions; `plan()` raises "nothing to grade" only at grade time (`:251-252`).
- `import_eog_dir` catches `ValueError`, `FileNotFoundError`, `KeyError` only (`:377`). A malformed seed raises `sqlite3.Error` in `infer_schema` and aborts the whole directory import.
- Fix: refuse at import when no assertion remains. Mark a partially ported task `provisional` in a field that acceptance reads. Catch `sqlite3.Error` per task.

## 3. Coverage

Run from `crucible/` with `COVERAGE_FILE` and a coverage rc in the scratchpad. The OpenEnv cases run as subprocesses in `.envs/openenv`, so a scratch `sitecustomize.py` (`coverage.process_startup()`) and `COVERAGE_PROCESS_START` collected them:

```
coverage run -m pytest -q -p no:cacheprovider tests/unit/{test_openenv_import,test_interop_eog,test_awm_tasks,test_interop_gate}.py
27 passed in 141.0 s; subprocess data combined; TOTAL 1475 stmts, 76%
```

| Module | Cover | Untested risky paths |
|---|---|---|
| `openenv/import_env.py` | 83% | reset-digest mismatch on replay (`380`), `CallToolObservation.error` text (`395-396`), `RemoteActionError` and `CONNECTION_LOST` replay (`451-459`), the error blame map (`473-475`), MCP `isError` (`478`), non-zero-sum seats (`552-555`) |
| `openenv/export.py` | 96% | reset of a live World (`64`), the async twins (`82`, `128`), `close` (`132`) |
| `openenv/awm_task.py` | 92% | a missing verifier (`35`), the verifier timeout (`48-49`), a non-JSON runner reply (`55-56`), the mask itself (`100`) |
| `openenv/awm.py` 91%, `catalog.py` 89% | | |
| `openenv/awm_check.py` | 0% | the whole M1 gate 2b sweep |
| `mimo/verify.py` | 83% | env restore of a pre-set variable (`86`), the crash mask (`88`), **the `ok` branch** (`91-93`, V7) |
| `mimo/check.py` | 0% | the M1 gate 2 sweep (p6 ran it on one env: 9/9 checks pass) |
| `mimo/export.py` 87%, `mimo/importer.py` 93% | | `out` already existing (`17`), the diff failure (`23`) |
| `eog/importer.py` | 92% | legacy single-gym config (`81`), the zip-less seed lookup (`110-117`), two servers on one app id (`302`), a verifier naming an unknown gym (`310-312`), `import_hf_rows` (`384-390`) |
| `eog/oracle.py` | 88% | `TRUE`/`FALSE` literals, non-`equals` refusal (`85`), several `_bump` / `_choose` branches (`155-175`), an empty insert (`230`) |

High line coverage hides the defects above. No test restores an environment whose hidden state lives in `metadata` (V1). No test asks whether a policy can reach a mask (V2, V16). No test grades a MiMo env on the `ok` path (V7). No test sends a numeric value as a string to an EOG World (V12). No test exercises a SQL tool against the filesystem (V15).

## 4. Offline gates

- **OpenEnv import gate** (`test_openenv_import.py`, in `.envs/openenv`): 12/12 pass (inside the 27 above).
- **EOG gate** (`test_interop_eog.py`): passes. 12 of 13 local tasks import (one seed is missing from `gym_dbs.zip`), all no-ops score 0, 7 witnesses score 1, one through the Runner.
- **AWM tasks** (`test_awm_tasks.py`): passes on `social_network_8` t0 (correct 1, wrong 0, no-op 0, disallowed import masked).
- **Interop toy gate and MiMo round trip** (`test_interop_gate.py`): passes. The MiMo grade without a judge is masked `judge_failed`; the test would also accept `ok`.
- Not re-run: the AWM 100-scenario sweep (`awm_check`), the MiMo 267-env sweep (`mimo/check`), anything needing Docker images, a real judge or the HF dataset.

## 5. Suggested fix order

1. V15 (SQL authorizer): a sandbox escape, one small change, protects every `sqlite_app` World.
2. V2, V16, V4: masks a policy can reach, and a mask that goes out as a score. They share one rule: health-check the environment or the verifier; if it is healthy, the failure is the policy's.
3. V1: the restore contract.
4. V11, V12, V13: EOG rewards that are silently wrong. Re-run the 12-task gate after each.
5. V6, V7, V8 together: run MiMo's `verify.py` in a subprocess with an explicit environment and read `detail.passed`.
6. The rest (V3, V5, V14, V17-V21, V9, V10).

The cheapest regression seeds are p1 (V1, V2, V3), p2 (V4, V5) and p4a/p4e (V11, V15). Each runs offline in under 30 s.
