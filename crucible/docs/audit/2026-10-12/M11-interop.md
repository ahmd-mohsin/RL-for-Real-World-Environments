# M11 interop: audit (2026-10-12)

Scope: `src/crucible/interop/` (harbor, agentenv, verifiers, nemo_gym, openenv, mimo, eog, canonical, replay), the App
MCP server it fronts (`world/packaging/app_server.py`), the SQLite App's policy tools (`world/apps/sqlite_app.py`,
`world/state/snippets/sql.py`), `verify/sql_checks.py` (EOG semantics), the score-gate outcome rule in
`verify/runtime/plan.py`, the failure taxonomy (`core/schema/run.py`, `runtime/infra.py`), the vendored AWM
`create_database`, a new top-level `crucible_vf` (the verifiers plugin id), and the suite's collection rule
(`tests/conftest.py`).

Checked against ARCHITECTURE §M11 and the §3 rows tagged M11, DESIGN "As built, M11 part 1", "part 4", "M11 importers",
B.17/B.18, `docs/research/m11-m12/05-interop-diversity.md`, with the M0 protocol. Framework comparison:
`M11-study-sweep.md` (five group sweeps). Defect hunts: `M11-study-defects-H1.md` (Harbor, replay), `-H2.md` (agentenv,
verifiers, NeMo-Gym), `-H3.md` (OpenEnv, MiMo, EOG). 25 regression tests (`tests/unit/test_m11_audit.py`, with
`_m11_openenv_cases.py`) fail on the M10 audit commit (`af402b4`) and pass after the fixes (five of them fail there
first on a helper the fix adds, `instance_for`, `function_name`, `rewards.py`, `_acquire_docker_host`, `grade(workdir=)`;
each also asserts the behaviour). The 25th tests the collection rule below.

**The cross-runtime gate was never in a full run.** `tests/conftest.py` skips a test module whose *text* has a line
starting `from verifiers` / `import openenv` (an optional framework that is not installed). `test_interop_gate.py`
carries such lines inside code strings it runs in the framework Pythons, so it, `test_remote_ws.py` and this audit's
own test file were silently dropped from every `pytest tests/unit` (36 tests). The rule now reads the module's
top-level import statements (`ast`); all 36 pass.

There was no `docs/modules/M11-interop.md` and no M11 row in the ARCHITECTURE status table; both are added.

## 1. Requirement ledger (rows that were wrong, missing or partial)

| # | Requirement | Before | After |
|---|---|---|---|
| L1 | Every infra fault is masked on every foreign path (P16, B.17) | **wrong**: the NeMo-Gym export, the OpenEnv export and the agentenv gateway path graded a World with a recorded fault and scored it; the agentenv backend relabelled every App crash as a scored tool error | the NeMo-Gym `/verify`, the OpenEnv final step and `crucible_verify` mask on `World.failures`; App servers carry the fault label across the MCP hop; a grading exception masks (`judge_failsafe`) |
| L2 | An imported framework's own failure is never the policy's 0 | **wrong**: a NeMo-Gym server's 500, death or missing / NaN reward scored 0, its rewards were clamped; OpenEnv NaN scored 0, inf 1 | non-2xx, transport loss, missing or non-finite rewards and out-of-range rewards mask (`reward_range`); a terminal step without a finite reward masks |
| L3 | A policy cannot turn its own failure into a mask | **wrong**: a raising OpenEnv action, a slow MCP call, a 4xx NeMo-Gym tool reply, a long AWM answer, a broken Harbor container, malformed A2A model output all masked | a health replay decides OpenEnv faults (healthy → policy); 4xx → policy, 5xx → tool; an AWM verifier that runs on the untouched DB failed on the episode (scored 0); in a shared Harbor container a missing reward after the dirs were reset, or a verifier user that cannot write, is the policy's 0; `LLMToolAgent` reports its own errors as `agent_error` |
| L4 | The policy cannot reach the grader, the answer or the control plane (P21) | **wrong**: `verify` was callable as a NeMo-Gym tool mid-episode; the agentenv agent got the gateway root (re-enabled a revoked tool, read actor tokens); planted `reward.json`, `/tests/conftest.py`, `mask.json`, `steps.jsonl` were read; the separate verifier ran in the agent's container; policy SQL wrote files outside the World | one grade per NeMo-Gym session ends it; agents get a role-pinned `/mcp`-only proxy; the grader's dirs are emptied first and grader-only files are removed; separate-verifier tasks are refused in place; an SQLite authorizer denies ATTACH / VACUUM INTO / setting PRAGMAs |
| L5 | Outside agents act through the World (agentenv) | **wrong**: gateway calls reached `App.call` directly (no log, clock, events, schema checks, RBAC, quotas) | the App server's caller runs a World step for unsigned (outside) calls; the World's own tool calls in that step go straight to the Apps |
| L6 | Contracts of each framework, field by field | partial: NeMo-Gym seed / close / reverify replies failed its models, its own agent could not seed, a retried seed leaked; verifiers could not load the export; verifiers v1 traces masked recovered retries, crashed on `None` rewards, kept one branch and 4-decimal logprobs; agentenv `teardown_run` failed, verifier rows were per gate, concurrent runs shared a World | NeMo-Gym routes validate against its models (test in `.envs/nemo`); `vf-eval crucible-vf --taskset.path` loads; traces by `ok`, branches and token records checked, exact logprobs; reward-mode row, `metadata["score"]`, per-run World handle, `sandbox_id` empty, Worlds closed after grading |
| L7 | Harbor semantics (rewards, network phases, verifier env) | **wrong**: patch mode took NaN / strings, crashed on a single-key file, graded with no network by default, dropped `[verifier].env` / `user`, trusted any `mask.json`; the agent's network phase was never applied in place; same-named tasks of two orgs shared an id; `DOCKER_HOST` was unset under a live World | Harbor's reward rules (`rewards.py`) everywhere; the verifier phase from the environment baseline; env file, `--user`, legacy `memory`; `mask.json` only from Crucible's kit; the agent phase applied after `open` (fails closed); the full task name; a reference-counted `DOCKER_HOST` |
| L8 | Replay and cross-runtime identity | partial: a cut or unmapped transcript graded as a real 0; final answers were dropped (NeMo-Gym import and ATIF cut them to 2,000 characters); temp dirs leaked | `crucible:replay_incomplete` masks; final answers travel as `Done(answer=...)`; replay dirs removed |
| L9 | OpenEnv import and export | **wrong**: restore accepted a different hidden state; masked episodes paid their Ψ steps; finish re-paid the final reward | digests cover `metadata`, `reward`, `done`; a masked episode raises `CrucibleMaskedEpisode`; the final observation is latched |
| L10 | MiMo grading | **wrong**: `GA_JUDGE_KEY2/3` reached the grader with `judge=None`; an importable `synthesis` package decided the grade; `outcome` read a field MiMo never returns; out-of-range scores were scored; the export could delete its source | `verify.py` runs in `python -I` with only the configured judge and `synthesis` blocked; `outcome = detail.passed`; non-finite or out-of-[0, 1] scores mask; overlapping export targets refused |
| L11 | EOG grading | **wrong**: a "drive the count to zero" task became an unwinnable refusal; typeless columns failed numbers sent as strings; comparisons differed from the benchmark; the SQL tools saw the host clock | refusal only when every zero-count check holds at the seed; column affinities from seed values; EOG's `_compare_values` verbatim (`semantics="eog"`); the tools pin `now` to the World clock |
| L12 | Score-gated imports report success | **wrong**: outcome 1.0 and gate pass at reward 0 (A1 could never pass on them) | a score check may set `detail["success"]`; outcome needs it; the reward is unchanged |
| L13 | Taxonomy | partial: M11's own failure names were unmapped (unknown blame, counted by the stop rule) | mapped; `crucible:replay_incomplete` / `replay_divergence` registered |
| L14 | Exit gate (cross-runtime identity on task02 and a toy task; MiMo round trip) | partial: the toy gate passes but several legs pass by construction; no task02 case; the MiMo round trip copies bytes | unchanged in scope; the NeMo-Gym leg now validates every route, the verifiers export loads through verifiers' own loader; DESIGN states what each leg proves |
| L15 | Exports to verl, MiMo-from-World, SkyRL, TRL; `manifest.yaml`; imports I0-I10 | **missing** | still missing (queued; the trainer-side bridges are M12) |

## 2. Defects and fixes

| Ids | Sev | Fix (files) |
|---|---|---|
| H2 V1, G3 D1 | P0 | `interop/nemo_gym/server.py` (`_grade`: `World.failures`, judge failsafe) |
| H2 V5, V10, V11, G3 D2-D6, D18 | P0/P1 | `nemo_gym/server.py` (one grade per session, idempotent seed, legacy body, strict replies, `/crucible/session`, `/crucible/message`, `/crucible/done`, session lock, reaper skips busy sessions, temp dirs removed, seed `episode_id` keys `call_id`s); `nemo_gym/client.py` |
| H2 V4, G3 D9, D10, D21, G5 D1, D2, D7 | P0/P1 | `nemo_gym/import_server.py` (verify errors mask, reward range, 4xx/5xx tool replies, full outputs, `close_session`, full answer, roles kept) |
| H2 V2, V3, V6, V7, V14-V16, G4 D1, D3-D8, D16 | P0/P1 | `world/packaging/app_server.py`; `interop/agentenv/world_backend.py` (`_DIRECT`, `_external_step`, `AgentProxy`, contained teardown); `agentenv/plugin/steps.py`; `agentenv/harness.py`; `agentenv/agents.py` |
| G2 D1-D6, D10, D11, D14, H2 V8, V13, G5 D11 | P0/P1 | `interop/verifiers/vf_taskset.py` (new), `taskset.py`, `src/crucible_vf/` (new), `v1_import.py`; Harbor installed (`--no-deps`, local source) in `.envs/verifiers` |
| H1 V1-V4, V11, V19, G1 D2, D4, G5 D3 | P0 | `interop/harbor/inplace.py`, `multi_step.py` |
| H1 V5-V7, V20, V21, G1 D6, G5 D8, D9 | P0/P1 | `interop/harbor/rewards.py` (new), `verify_dir.py` |
| H1 V8, V12 | P1/P2 | `harbor/world_backend.py`, `harbor/metrics.py` |
| H1 V9, V18, G5 D6, D2b | P1 | `interop/replay.py`, `harbor/atif.py` |
| H3 V1-V5, G1 D1, G5 D4, D5 | P0/P1 | `interop/openenv/import_env.py`, `openenv/export.py`, `verify/runtime/plan.py` (`success`) |
| H3 V16-V18 | P0/P2 | `openenv/awm_task.py`, `openenv/awm.py`, `_vendor/openenv_awm/db_manager.py` (+ ORIGIN note) |
| H3 V6-V10, V21, G3 D14 | P1/P2 | `interop/mimo/verify.py`, `mimo/export.py` |
| H3 V11-V15, V20, G3 D11, D12 | P0/P1 | `interop/eog/importer.py`, `verify/sql_checks.py`, `world/apps/sqlite_app.py`, `world/state/snippets/sql.py` |
| G5 D10 | P1 | `core/schema/run.py`, `runtime/infra.py` |
| suite rule | P0 (test gap) | `tests/conftest.py` (`_top_imports`) |
| found while fixing | P1 | trigger actions and state checks the agentenv gateway runs itself are signed as the event's actor (`world_backend._trigger`): once outside calls became World steps, an unsigned trigger action waited on the step that provoked it |
| not fixed | — | H1 V10 (replay `call_id`s are per action, not per model call), V13-V17 (identity-gate "masked vs nothing written", JCS naming, ATIF tool-error and image fidelity, ignored-key noise, binary oracle files); H2 V17 (static actor token; mitigated: the agent no longer reaches `/trajectory`), V19-V21; G4 D2, D9, D10, D17 (TRL / verl / SkyRL / AReaL bridges: M12), D11 (MiMo-from-World and `recipes/code` exports), D12-D15; G3 D7 (NeMo-RL baselines count masked 0.0: M12 bridge), D8 (NeMo-Gym's `harbor_agent` scores a missing reward 0: unsupported path, documented), D13, D15-D17, D20 (`.envs/nemo` not built from NeMo-Gym's lock); G2 D12, D13, D15; G5 D12-D20 (grading-key refusal, I0-I10) |

## 3. Gates

- Toy cross-runtime gate (`tests/unit/test_interop_gate.py`): passes. The NeMo-Gym leg reads tools from
  `/crucible/session`; the agentenv leg now takes its rows from the World trace.
- NeMo-Gym route validation against NeMo-Gym's own models (`.envs/nemo`): seed, masked verify, close, `reverify_mode`
  and `aggregate_metrics` all validate.
- verifiers: `import_taskset("crucible-vf")` resolves `CrucibleTaskset` and `HarborEnv` in `.envs/verifiers`.
- OpenEnv import suite (PortSim, generic, noisy, stochastic simulation, connect4, pixels, MCP): passes with full-
  observation digests.
- EOG 12-task gate: no-op 0, witness 1 on the 7 derivable tasks, refusal variant, unchanged.
- Docker in-place Harbor tests pass (local daemon), including the planted-reward regression.
- Not run: task02 through any foreign runtime (lab GPU), NeMo-Gym's `single_agent_turn` driving the export, a live
  `vf-eval` episode on Docker, MiMo with a real judge (paid).
- `crucible.interop` coverage over the full suite: **76%** in process (64% before the collection fix). Modules that run
  only inside a framework's Python (`openenv/import_env.py`, `openenv/export.py`, `verifiers/vf_taskset.py`,
  `mimo/verify.py` through its subprocess) are exercised by tests there but measured as 0% here.
- Suite (`tests/unit`, `tests/toy_e2e`): **1087 passed, 3 skipped** (1053 collected before: the 36 tests the
  collection rule dropped, plus this audit's file). `lint-imports`: 4 kept; `ruff` clean. `test_launch.py`'s
  kill-and-restart chaos test failed once under full-suite load and passes alone (3/3) and in the final run.
