# M11 framework sweep, group 3: NeMo-Gym, NeMo-RL, MiMo (verl fork + v2.6 release), EnterpriseOps-Gym, Karotte

Read-only sweep of M11 `interop` at Crucible HEAD `af402b4` (2026-10-12). I opened every Crucible citation and every external citation myself. Nothing under `crucible/src`, `crucible/packs` or `crucible/tests` was changed.

Path roots:
- `NG` = `RL-Environments/06-frameworks-and-env-code/nvidia-nemo-gym/`
- `NR` = `RL-Environments/06-frameworks-and-env-code/infra-nvidia-nemo-rl/`
- `EOG` = `RL-Environments/06-frameworks-and-env-code/servicenow-enterpriseops-gym/`
- `KA` = `RL-Environments/06-frameworks-and-env-code/preferencemodel-karotte/`
- `XV` = `RL-Environments/06-frameworks-and-env-code/xiaomimimo-verl/`
- `ME` = `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss/`; `S` = `ME/general/envs/s3k_0000_accounting_audit_tax_en_t1_rl_008/`
- `C` = `crucible/src/crucible/`; `T` = `crucible/tests/unit/`

Probes live in `scratchpad/m11sweep/g3/` (`p1`..`p11`). They run from `crucible/` with `crucible/.venv/bin/python`, or with `crucible/.envs/nemo/bin/python` where NeMo-Gym code is needed. `[ran]` means a probe reproduced it. `[read]` means I found it by reading code only.

Verdicts: **DEPEND** (use their code or contract as is), **PORT** (copy and adapt; licence given), **PATTERN** (reimplement the idea), **nothing** (Crucible is as good or better), **anti-pattern** (do not copy).

## Pins and licences

| Root | Commit (date) | Licence | What Crucible runs |
|---|---|---|---|
| `NG` | `3ef478d` (2026-10-03), `nemo_gym 0.7.0rc0` | Apache-2.0 | `crucible/.envs/nemo` installs this checkout editable, so code = local source. **But the env is not built from `NG/uv.lock`:** Python 3.13.7 (`NG/pyproject.toml:28` requires ≥ 3.13.14); pydantic 2.14.0 (lock 2.13.4); fastapi 0.143.0 (lock 0.135.2); starlette 1.7.0 (lock 1.3.1); mcp 1.30.0 (lock 1.29.0); aiohttp 3.14.4 (lock 3.14.1). openai 2.44.0 matches. See D20. |
| `NR` | `c9193bc` (2026-10-10) | Apache-2.0 | not installed; read only |
| `XV` | `a2ad9f6` (2026-09-26) | Apache-2.0 (mimoagent MIT) | not installed; read only |
| `ME` | HF release, no git | README front matter `license: apache-2.0`; no LICENSE file | env dirs imported; `mcp_http.py`/`mcp_bridge.py` vendored in `C/_vendor/mimo_sidecar/` with `ORIGIN.md` |
| `EOG` | `271f2c3` (2026-08-19) | Apache-2.0 (tasks are data) | `data/revised` + `gym_dbs.zip` imported |
| `KA` | `31c693e` (2026-10-08) | MIT | nothing in M11 |

---

## 0. Corrections and confirmations of prior claims

| Prior claim | What the code says now |
|---|---|
| DESIGN "As built, M11 part 1": "Responses validate against NeMo-Gym's own pydantic models in its Python 3.13 env" | **Only the unmasked `/verify` body is validated** (`T/test_interop_gate.py:113-118`). The seed body is written to `seed.json` (`:111`) and never validated. `[ran p1]`: the seed, close and `reverify_mode` bodies all fail NeMo-Gym's own models. The aggregate body validates only because its keys are silently dropped. |
| ARCHITECTURE M11 exit gate: identical files "through ... NeMo-Gym" | The gate never runs NeMo-Gym code. It drives `make_app` with FastAPI's `TestClient` and replays the items with Crucible's own `from_responses` (`T/test_interop_gate.py:42-71`). No NeMo-Gym agent or environment server sends a request. That is why D2-D6 pass the gate (D17). |
| `C/interop/nemo_gym/server.py:13-14`: masked `reward` is 0.0 "only because the schema requires a float (NeMo-RL drops masked samples)" | **Half true.** NR zeroes the masked row's loss (`NR/nemo_rl/algorithms/grpo.py:2524-2537`, applied at `:3582`). But it computes group baselines first with `torch.ones_like(rewards)` as the valid mask (`grpo.py:3458-3499`). The 0.0 enters every sibling's baseline. `[ran p10]` (D7). |
| DESIGN B.17: "a masked grade writes no `reward.json` ... (a missing reward is an error in every runtime)" | **False for NeMo-Gym's `harbor_agent`.** A trial without verifier rewards gives `reward = 0.0` and no `mask_sample` (`NG/responses_api_agents/harbor_agent/utils.py:92-113`, `app.py:297-319`). `[ran p11]` (D8). Research doc 05 line 170 already noted it; B.17 did not carry it. |
| `M08-M10-sweep-G3.md` D9: "MiMo importer scores a missing `score` as 0.0" | **Refined.** A missing score is 0.0 in MiMo's own `S/run_verify.py:125` too, so Crucible matches it. The real gap is range: MiMo's runner masks a non-finite or out-of-[0, 1] reward (`XV/recipes/general/general_agent/environment.py:575-578`). Crucible's `C/interop/mimo/verify.py:91` passes 1.5 and −0.2 through as scored. NaN and inf are masked later by `write_reward_files` (`C/verify/runtime/plan.py:367-371`). `[ran p9]` (D14). |
| DESIGN "As built, M11 part 1": "MiMo round trip ... re-exported byte-identical" | **True but vacuous.** `export()` copies the source directory recorded at import and compares it with hashes of that same source (`C/interop/mimo/export.py:14-24`). It never reads the imported World. `[ran p8]`: an import with every App removed still exports "identical" (D13). |
| Research doc 05 §3.4: "`failure_kind` (Crucible names are already NeMo-legal)" | **Confirmed.** NeMo-Gym accepts its 22 names or any `<server>:<kind>` (`NG/nemo_gym/failure_kinds.py:148-165`). Crucible's own kinds are `crucible:*`. |
| `M08-M10-sweep-G3.md` §0: Crucible writes `mask_sample` in both places NR reads | **Confirmed.** `C/interop/nemo_gym/server.py:147-148`. NR reads only `instance_config.mask_sample` (`NR/nemo_rl/experience/rollouts.py:314-322`). NR honours it unless `env.should_mask_flagged_samples` is `false` (`rollouts.py:1913-1925`, default true). |
| `M08-M10-sweep-G5.md` D10: workplace_assistant rows with empty ground truth score 1.0 for doing nothing | Still open; not re-run here. It is an import-side gap of `NemoGymTask` (`C/interop/nemo_gym/import_server.py:227-238`). |

---

## 1. Per framework

### 1.1 NeMo-Gym resources server (`NG`, Apache-2.0): export (`C/interop/nemo_gym/server.py`)

NeMo-Gym has two callers of a resources server.
- The **legacy agent path**: `SimpleAgent.run` posts the dataset row to `/seed_session` (`NG/responses_api_agents/simple_agent/app.py:306-315`), carries the session in cookies, posts `POST /<tool>` with the arguments, puts the raw response body into `function_call_output` (`:215-223`), then posts `/verify` with the row plus `response`.
- The **environment-server path**: `single_agent_turn` sends `ResourcesSeedSessionRequest`, validates the reply with `ResourcesSeedSessionResponse` (`NG/environment_servers/single_agent_turn/app.py:109-130`), verifies with `task_data | {responses_create_params, response}` and the session cookie (`:254-274`), and closes with `ResourcesCloseSessionRequest` (`:92-101`).

Field-by-field contract (`NG/nemo_gym/base_resources_server.py`):

| Contract | NeMo-Gym source | Crucible | Status |
|---|---|---|---|
| Seed request `{resources_session_id, episode_id, task_id, task_data}`, extra forbidden, "Idempotently initialize" | `:184-192` | Accepts a superset (`server.py:99-103`) | OK as a reader. **Not idempotent** (D4). **Legacy body (no `resources_session_id`) → 500** (D3). Unknown `task_id` → 500 (D3). |
| Seed response `{resources_session_id, resources_tools: MCPServerMetadata?, sandbox_access?}`, extra forbidden | `:195-202` | Adds `tools`, `instruction` (`server.py:119-120`) | **Fails validation** `[ran p1]` (D2). |
| `MCPServerMetadata{server_name, url_path="/mcp", transport, headers}` (headers required) | `:175-181` | `{server_name, url_path: "/", transport: "http"}`, no headers; no MCP served at `/` | **Fails validation**; advertises a transport it does not serve (D2). |
| Session identity: cookie from seed, sent back on every call | `single_agent_turn/app.py:119-122`; `server_utils.py:1200-1217` | `resources_session_id` cookie, header or body (`server.py:85-86,121`) | OK for the env-server path. |
| `POST /<tool>` → JSON body shown to the model verbatim | `simple_agent/app.py:215-223` | `{"output": ..., "ok": ...}` (`server.py:170-180`) | OK (workplace_assistant also returns `{"output": ...}`, `NG/resources_servers/workplace_assistant/app.py:101-105`). An exception inside `world.step` is a 500 (P2, folded into D19). |
| Verify request = `BaseVerifyRequest{responses_create_params, response}` + row fields | `:92-105` | Reads `responses_create_params`, `response`, optional `episode_key` | OK. `episode_key` is never sent by NeMo-Gym (D18). |
| Verify response `BaseMultiRewardVerifyResponse{reward, reward_components, mask_sample, failure_kind?, failure_reason?}` | `:107-164` | `server.py:143-150` | **Validates** (`T/test_interop_gate.py:93-118`). `reward == sum(reward_components)` holds (`{"crucible": reward}`), as NR requires (`NR/nemo_rl/environments/nemo_gym.py:1483-1508`). |
| `mask_sample` means "the environment or its infrastructure failed" | `:110-122` | Set only from the grade (`server.py:140`) | **An infra fault during the episode is scored** `[ran p3]` (D1, P0). |
| Grading failures masked by `judge_failsafe` (mask in both places, `failure_kind judge_failed`) | `NG/nemo_gym/judge.py:85-117`; wired at `:247-252` | A grade exception is a 500 | P2 (pick 3). A 500 is a retryable episode failure in `single_agent_turn` (`app.py:331-335`), so it is not trained, but the record is lost. |
| `/close_session` → `ResourcesCloseSessionResponse{resources_session_id}` | `:218-232`, `:291-298` | `{"closed": bool}` (`server.py:153-158`) | Fails validation `[ran p1]`. `single_agent_turn` only checks the status, so P2 (D6). |
| `GET /reverify_mode` → a bare `ReverifyMode` string | `:75-78`, `:317-318`; read by `ReverifyMode(await get_response_json(res))` (`NG/nemo_gym/rollout_reverification.py:786-788`) | `{"reverify_mode": "unsupported"}` (`server.py:160-162`) | **`ValueError` in NeMo-Gym's guard** `[ran p1]`: `ng reverify` crashes instead of refusing (D5). |
| `/aggregate_metrics` → `AggregateMetrics{group_level_metrics, agent_metrics, key_metrics, ...}` | `NG/nemo_gym/config_types.py:895-931` | `{mean_reward, masked, n}` (`server.py:164-168`) | Validates with every value dropped `[ran p1]` (D6). |
| `manifest.yaml` (`EnvironmentManifest`: reward range, determinism, licensing, datasets) | `NG/nemo_gym/environment/manifest.py:215-313` | none in `C/interop` | **Missing** (D16). ARCHITECTURE M11 promises "a resources server plus `manifest.yaml`". |
| Dataset rows (`responses_create_params.input` + `.tools`) and a server config | `NG/resources_servers/*/data/*.jsonl`, `configs/` | none | **Missing** (D16). Stock NeMo-Gym agents take tools from the row, not from the seed reply. |
| ATIF v1.7 (`AtifTrajectoryV1_7`, `schema_version: Literal["ATIF-v1.7"]`, extra forbidden, strict) | `NG/nemo_gym/atif_v1_7.py:33-37,187-190`; exporter `atif_export.py:868` | `C/interop/harbor/atif.py` emits Harbor's default `ATIF-v1.8` | Rejected as emitted; **validates once the version is set to v1.7** `[ran p4]` (D15). |

Mechanisms:

| Mechanism | Source | Crucible equivalent | Verdict |
|---|---|---|---|
| The pydantic request/response models as the export contract | `NG/nemo_gym/base_resources_server.py:92-232` | Hand-written dicts; only verify is checked | **DEPEND** (test-time: validate every route, D17) |
| `single_agent_turn` environment server as a gate driver | `NG/environment_servers/single_agent_turn/app.py:83-298` | `TestClient` | **DEPEND** (test only; runs in `.envs/nemo`) |
| `judge_failsafe`: a judge or grader exception becomes `mask_sample` + `failure_kind` in both places | `NG/nemo_gym/judge.py:85-117` | 500 | **PATTERN** (S) |
| `EnvironmentManifest` + `environment/validation.py` (stale-manifest and reward-range errors) | `manifest.py:96-313`; `validation.py:448-711` | none | **DEPEND** (generate and validate) |
| `normalize_tool_name` and `RESERVED_MCP_TOOL_NAMES` | `base_resources_server.py:50-72` | `C/interop/canonical.py` `canonical_tool`; `app__tool` names cannot collide | nothing |
| Admission (503 on queue timeout) and idle reaper | `NG/nemo_gym/base_environment_server.py:37-44,132` (`max_concurrent_episodes`, `queue_timeout_seconds`) | `server.py:40-80` | nothing (equivalent; 503 is retryable in NG, `single_agent_turn/app.py:331-335`) |
| Strict ATIF v1.7 subset model | `atif_v1_7.py` | Harbor models | **DEPEND** (test-time validator for a v1.7 export) |

### 1.2 NeMo-Gym import (`C/interop/nemo_gym/import_server.py`, `client.py`)

| Contract | NeMo-Gym behaviour | Crucible | Status |
|---|---|---|---|
| Tool errors are model-visible outputs | `simple_agent/app.py:215-223`: any status, body goes to the model; workplace_assistant returns errors as 200 (`app.py:102-105`) | `raise_for_status()` (`import_server.py:129-134`) → `World._call_tool` masks `crucible:tool_crashed` (`C/world/world.py:477-481`) | **A 422 or 500 masks the episode** `[ran p5a]` (D9). |
| `reward: float`; non-finite is an error | `NG/nemo_gym/verifier_fixture.py:95-98` | `float(doc.get("reward", 0.0))`, clamped to [0, 1] (`import_server.py:251-253`) | **NaN → 0 scored; 1.8 → 1; −1 → 0; missing → 0** `[ran p5b]` (D10). |
| Declared range `reward.range` | `manifest.py:100-108`; 171 manifests [0, 1], 2 [0, 0], 1 [0, 1.8] (my count over `NG/environments`, `NG/benchmarks`) | ignored | D10. |
| `mask_sample` top level or `instance_config` | `base_resources_server.py:110`; `judge.py:105-115` | both read (`import_server.py:247`) | OK. |
| Session lifecycle: seed then close | `single_agent_turn/app.py:92-101` | Re-seed on restore opens a new session; the old one is never closed; `stop()` never closes (`import_server.py:97-115,142-148`) | P2 (D21). |
| Remote client: masks admission and connection loss | — | `client.py:55-60,77-82` | OK. Other non-2xx tool replies and a 404 at `/verify` raise instead (`:83,89-90`) (D22). |

### 1.3 NeMo-RL consumption (`NR`, Apache-2.0)

| Where NR reads the verify response | Source | Effect on Crucible's export |
|---|---|---|
| `mask_sample` only from `instance_config` | `NR/nemo_rl/experience/rollouts.py:314-322`; `rollout_manager.py:2324-2330` | Crucible writes it (OK). |
| `should_mask_flagged_samples` (default true) | `rollouts.py:1913-1925` | OK. |
| `reward == sum(reward_components)` or `ValueError` | `NR/nemo_rl/environments/nemo_gym.py:1483-1508`, called at `rollouts.py:3205-3216` | OK (`{"crucible": reward}`). GDPO then sees one component `reward/crucible`. |
| Baseline and std with `torch.ones_like(rewards)` | `NR/nemo_rl/algorithms/grpo.py:3458-3499` | Masked 0.0 shifts siblings' advantages. `[ran p10]`: group `[1, 1, 1, masked]` gives advantages +0.333 on each success; honouring the mask gives 0 (D7). |
| `mask_sample` loss filter after advantages | `grpo.py:2524-2537`, `:3582` | Masked row's own loss is zero. |

Verdict: **anti-pattern** for the baseline (do not count a masked row). The MiMo fork does it right: invalid rows are filled with the group's valid mean, then their advantage is zeroed (`XV/verl/trainer/ppo/core_algos.py:304-350`). **PATTERN** for Crucible's M12 NR bridge.

### 1.4 MiMo v2.6 release (`ME`) and verl fork (`XV`): `C/interop/mimo/`

| Contract | MiMo source | Crucible | Status |
|---|---|---|---|
| Env layout: `manifest.json`, `instruction.md`, `tools/*.py`, `system/<app>/state.db`, `workspace/`, `verify.py`, `verifier_meta.json`, `run_verify.py`, `_helpers.py`, sidecar scripts (all 267 envs; one lacks `verify.py`/`verifier_meta.json`) | my inventory of `ME/general/envs/*` | `importer.py:41-98` | OK. Tool discovery copies `sidecar_entrypoint.py`'s rule (`importer.py:41-46`). |
| Physical DB isolation: the agent container never mounts `/work/system` | `XV/recipes/general/general_agent/environment.py:33-41` | Policy gets MiMo tools + a `workspace` doc library confined to `mimo/workspace` (`C/world/apps/doc_library.py`, `confine`); `sql_tools` off | OK. |
| `verify(env_path, ground_truth=None, model_answer={}, pre_state={}, post_state, agent_output)` | `S/run_verify.py:76-92` | `verify.py:73-74` | OK, same call and the same `post_state` (`verify.py:32-46` = `run_verify.py:34-51`). |
| `agent_output` = last assistant text of the session logs | `S/run_verify.py:14-31` | Caller passes it by hand; no production caller exists (only `T/test_interop_gate.py:245-259`) | P2 (D14): there is no MiMo `RunTask`, so imported envs are not graded through the M10 runtime. |
| Rubric filter `VERIFY_DETERMINISTIC` / `VERIFY_AGENT_JUDGE` over `rubrics.json` | `S/run_verify.py:54-73` | not applied | nothing: 0 of 267 envs ship `rubrics.json`. |
| Mask contract: crash or judge failure → no `reward` key | `S/run_verify.py:102-151` | `verify.py:87-90` masks both | OK. |
| Runner masks unparseable, non-finite or out-of-[0, 1] reward | `XV/.../environment.py:569-578` | 1.5, −0.2 scored `[ran p9]` | D14. |
| `reward_binary_fullscore` per instance | `environment.py:580-582` | none | nothing for import (instance flag of MiMo's own dataset rows). |
| Infra categories → −999 → mask | `XV/recipes/general/agent_loop.py:76-87`; `core_algos.py:304-350` | M12 territory | PATTERN (see 1.3). |
| `TokenTrace`: response mask 1 for generated tokens, 0 for injected | `XV/recipes/general/token_trace.py:73-97` | M12 `rollout/v1` | nothing for M11. |
| Export target: `DatasetEnvActor` instance row `{instance_id, dataset_type: "general_agent", docker_image, env_task_dir, cwd, problem_statement, verifier_timeout_sec?}` | `environment.py:43-56`; `env_actor.py:134-185` | **No Crucible-World → MiMo export** | D13. ARCHITECTURE M11 "verl: Worlds exported in MiMo format for `recipes/general`". |

Verdicts: env layout and `verify()` call **DEPEND** (done). Range mask **PATTERN** (S). DatasetEnvActor row **TARGET** (build D13). The vendored sidecar is a correct **PORT** with `ORIGIN.md`.

### 1.5 EnterpriseOps-Gym (`EOG`, Apache-2.0): `C/interop/eog/`

| Contract | EOG source | Crucible | Status |
|---|---|---|---|
| `BenchmarkConfig{system_prompt, user_prompt, verifiers, number_of_runs, gym_servers_config, selected_tools, restricted_tools, ...}` | `EOG/benchmark/models.py:54-82` | `load_task` (`importer.py:68-83`) decodes HF JSON-text fields | OK. |
| `GymServerConfig{mcp_server_name, mcp_server_url, seed_database_file, auth_config?, context?}` (+ `user_info` in the data) | `models.py:41-50` | One `sqlite_app` per server; identity from the **first** server only (`importer.py:339-344`) | OK for the 13 local tasks (one gym each). Multi-gym identities are dropped (P2 note). |
| Three verifier types | `EOG/benchmark/verifier.py:48-80` | `database_state` only; others refused (`importer.py:64-65,296-299`) | OK, declared. With `strict=False` the dropped verifiers are only listed in `meta` (P2 note). |
| `database_state`: query, reduce, compare | `verifier.py:81-127,291-399` | `C/verify/sql_checks.py:33-83,217-227` | **Comparison diverges** `[ran p7]` (D12). Local data uses only `equals` (91 int, 11 str; my count), so local tasks are unaffected. |
| Unconfigured `gym_name` | EOG skips it, and `all()` over the rest passes vacuously (`EOG/benchmark/executor.py:371,440-451`) | Re-addressed to the only App, or refused (`importer.py:309-312`) | nothing (Crucible is stricter). |
| Refusal tasks | EOG has no abstain verifier; infeasibility is an HF config (`EOG/evaluate.py:284-298`) | Inferred when every check is `COUNT(*) = 0` (`importer.py:351-355`) | **Misfires on feasible "drive to zero" tasks** `[ran p6]` (D11). |
| Tools: 512 named MCP tools in Docker images | `EOG/README.md`; gym servers | `sql_query` / `sql_execute` | Declared substitution (`importer.py:15-18`). Row visibility per user is not reproduced (`:36-37`). |

Verdicts: the verifier comparison **PORT** (Apache-2.0, ~25 lines, `verifier.py:291-399`). The executor's vacuous `all()` is an **anti-pattern**.

### 1.6 Karotte (`KA`, MIT)

- Format: a task is a Python class with `id`, `system_prompt`, `tools` and `steps`; each step has `instructions` and a `judge` (`KA/docs/tasks/tasks-and-steps.md:1-80`). Discovery is `environment.get_tasks()` (`KA/src/karotte/load_tasks.py:21-49`). The run score is the last scored step (`KA/docs/tasks/scoring.md:18`). `Scoring{score, metadata, continue_task}` (`KA/src/karotte/schemas/scoring.py`).
- M11 has no Karotte contract: ARCHITECTURE M11 lists no Karotte folder, and nothing in `C/interop` reads Karotte. **nothing** for M11 now.
- If an importer is built later, it maps like Harbor multi-step (`C/interop/harbor/multi_step.py`). Two rules: Karotte `status="error"` becomes a mask, and an `ExecutableJudge` crash (scored 0, `KA/docs/tasks/scoring.md:52`) becomes a mask too. The judge crash rule is an **anti-pattern** to avoid.

---

## 2. Best picks (ranked)

Effort: S = under a day, M = a few days.

| # | Pick | Source | Verdict | Change in Crucible | Effort |
|---|---|---|---|---|---|
| 1 | Mask episodes with World infra faults in every export | Crucible's own rule (`C/actors/policy.py:626`); NG `mask_sample` meaning (`base_resources_server.py:110-122`) | PATTERN | `server.py:124-151`: if `world.failures` holds a non-policy kind, return `mask_sample` with that `failure_kind`. Same at `C/interop/openenv/export.py:116-120` | S |
| 2 | Validate every export route with NeMo-Gym's own models, driven by NeMo-Gym's own environment server | `base_resources_server.py:167-232`; `single_agent_turn/app.py:83-298` | DEPEND (test) | Gate drives `make_app` through `single_agent_turn` with a scripted model in `.envs/nemo`; validates seed, verify (masked and not), close, `reverify_mode`, aggregate | M |
| 3 | `judge_failsafe` around `/verify` | `NG/nemo_gym/judge.py:85-117` | PATTERN | A grade or replay exception becomes `mask_sample` + `failure_kind: verifier_error` in both places | S |
| 4 | Package the export for stock NeMo-Gym | `manifest.py:215-313`; `validation.py` | DEPEND | `interop/nemo_gym/package.py`: `manifest.yaml` (`reward.range [0, 1]`, `determinism`, `licensing`), `config.yaml`, JSONL rows with `responses_create_params.{input, tools}` and `task_id`; validated by `EnvironmentManifest` | M |
| 5 | Reward-range mask on every importer | `XV/.../environment.py:575-578`; `NG/nemo_gym/verifier_fixture.py:95-98` | PATTERN | `mimo/verify.py:91` and `nemo_gym/import_server.py:251-253`: non-finite, missing (NeMo) or out-of-range → masked `verifier_error`; NeMo import normalizes with the manifest range | S |
| 6 | Valid-mean fill for masked rows in the NR bridge | `XV/verl/trainer/ppo/core_algos.py:304-350` | PATTERN | M12 NR bridge passes `valid_mask = ~mask_sample` to `calculate_baseline_and_std_per_prompt` (a one-argument upstream change) or fills masked rewards with the group's valid mean | S |
| 7 | EOG comparison verbatim | `EOG/benchmark/verifier.py:374-399` | PORT (Apache-2.0) | `sql_checks.compare` gets an `eog_strict` mode used by `from_enterpriseops`; lenient coercion only for inferred untyped columns, recorded in the check details | S |
| 8 | ATIF v1.7 for the NeMo-Gym target | `atif_v1_7.py:33,187-190` | DEPEND (validator) | `episode_to_atif(..., schema_version="ATIF-v1.7")` for NG consumers; test with `AtifTrajectoryV1_7` | S |
| 9 | MiMo `DatasetEnvActor` row as the MiMo export target | `XV/.../environment.py:43-56` | TARGET | `mimo/export.py`: build the env dir from the World (tools files, `system/*/state.db` at t0 from State Adapters, `instruction.md`, verifier files) and emit the instance row; the byte-identity check then compares a real round trip | M |

Anti-patterns (do not copy): NG `harbor_agent` reward default 0.0 (`harbor_agent/utils.py:92-113`); NR baselines over masked rows (`grpo.py:3458-3499`); EOG `all()` over skipped verifiers (`executor.py:371,440-451`); Karotte judge crash = 0.

---

## 3. Crucible defects and spec gaps

| # | Sev | Defect | Where | Evidence | Fix |
|---|---|---|---|---|---|
| D1 | **P0** | An infra fault during an episode driven through the NeMo-Gym export is scored. `World._call_tool` records `crucible:tool_crashed` in `World.failures` and shows "internal error" to the policy. `/verify` grades the World and never reads `World.failures`. Only `run_episode` turns them into a mask (`C/actors/policy.py:626`). The M12 remote rollout path (`client.py`) inherits it. The OpenEnv export has the same shape | `C/interop/nemo_gym/server.py:124-151`; `C/interop/openenv/export.py:116-120` [read] | [ran p3]: `reward 0.0, mask_sample False, failure_kind None`, with `World.failures = [crucible:tool_crashed]` | Pick 1. Map each failure through `runtime.infra.normalize`; mask unless the kind is a policy outcome. Regression test: p3 |
| D2 | P1 | The seed reply fails `ResourcesSeedSessionResponse` (extra `tools`, `instruction`; `resources_tools.headers` missing; MCP advertised at `/`, which is not an MCP endpoint). `single_agent_turn` validates it, so every episode fails at seed | `server.py:117-122` vs `NG/nemo_gym/base_resources_server.py:175-202`; `NG/environment_servers/single_agent_turn/app.py:124` | [ran p1]: 3 validation errors | Reply `{resources_session_id}` only; set `resources_tools` only when an MCP route is served (with `headers`). Give Crucible's client its tools through a separate `GET /crucible/session/{id}` or a dataset row |
| D3 | P1 | The legacy agent path cannot seed: `SimpleAgent.run` posts the row, which has no `resources_session_id`, so `body["resources_session_id"]` raises (500). An unknown `task_id` is also a 500 | `server.py:101-103`; `NG/responses_api_agents/simple_agent/app.py:306-315` | [ran p2]: legacy body 500; unknown task 500 | Fall back to the NeMo-Gym session cookie (`server_utils.py:1200-1217`) or mint an id and set the cookie; unknown task → 404 |
| D4 | P1 | `/seed_session` is not idempotent. A retried seed with the same id replaces the session without closing the first World or releasing its slot. Capacity shrinks by one per retry | `server.py:102-116`; contract `base_resources_server.py:185` ("Idempotently initialize") | [ran p2]: after one retry and one close, capacity 2 with 1 live session answers 503 | If the id exists for the same task, return the existing session; otherwise close it first |
| D5 | P1 | `/reverify_mode` returns `{"reverify_mode": ...}`; NeMo-Gym parses a bare enum, so `ng reverify` raises `ValueError` instead of refusing | `server.py:160-162` vs `NG/nemo_gym/rollout_reverification.py:786-788` | [ran p1] | Return `"unsupported"` |
| D6 | P2 | `/close_session` reply fails `ResourcesCloseSessionResponse`; `/aggregate_metrics` reply validates as `AggregateMetrics` with every value dropped | `server.py:153-168` vs `config_types.py:906-931` | [ran p1] | Return `{resources_session_id}`; return `agent_metrics`/`key_metrics` (mean over unmasked, masked count) |
| D7 | P1 (spec) | The masked sample's `reward: 0.0` enters NR's GRPO baselines; the docstring says NR drops it | `server.py:13-14,143`; `NR/nemo_rl/algorithms/grpo.py:3458-3499,3582` | [ran p10]: `[1, 1, 1, masked]` → +0.333 per success instead of 0 | Correct the docstring and DESIGN B.17; pick 6 in the M12 NR bridge |
| D8 | P1 (spec) | DESIGN B.17 says a missing `reward.json` is an error in every runtime. NeMo-Gym's `harbor_agent` turns it into reward 0.0 with no mask. Crucible Harbor tasks run through it train every masked grade as a real 0 | DESIGN B.17; `NG/responses_api_agents/harbor_agent/utils.py:92-113`, `app.py:297-319` | [ran p11] | Fix B.17. Document the NG `harbor_agent` path as unsupported; route NeMo-Gym through the resources-server export; offer upstream a `mask_sample` when `verifier_result` has no rewards |
| D9 | P1 | The NeMo-Gym importer masks any non-2xx tool reply as `crucible:tool_crashed`. NeMo-Gym shows that body to the model. A policy can trigger a mask with arguments the server rejects (422) or that crash the tool (500) | `C/interop/nemo_gym/import_server.py:129-134` → `C/world/world.py:477-481` | [ran p5a]: 422 and 500 both `masked=True` | 4xx → `ToolResult(ok=False, fault="policy", content=body)`; 5xx → `fault="tool"` (scored, shown); only connection loss and 502-504 → infra |
| D10 | P1 | The NeMo-Gym importer clamps the server's reward: NaN → 0 scored, 1.8 → 1, −1 → 0, missing → 0. It ignores the manifest's `reward.range` | `import_server.py:251-253` | [ran p5b] | Pick 5 |
| D11 | P1 | The EOG refusal heuristic turns a feasible "drive to zero" task (only `COUNT(*) = 0` checks) into a refusal task with a hard `unchanged` gate. No action can score 1 | `C/interop/eog/importer.py:351-355` | [ran p6]: do nothing 0.0 (count fails); close the cases 0.0 (`unchanged` fails) | Infer refusal only when every zero-count check already holds on the seeded DB (evaluate at import); otherwise a normal task |
| D12 | P2 | `sql_checks.compare` diverges from EOG's `_compare_values`: `greater_than`/`less_than` on date strings is always false; `equals` and `contains` coerce str and int where EOG does not | `C/verify/sql_checks.py:44-83` vs `EOG/benchmark/verifier.py:374-399` | [ran p7]: 4 of 7 cases differ | Pick 7 |
| D13 | P1 (gate) | The MiMo "byte-identical re-export" copies the source directory and compares it with itself. It cannot detect a lossy import. No function exports a Crucible World to MiMo format (ARCHITECTURE M11 `verl` bullet) | `C/interop/mimo/export.py:14-24` | [ran p8]: import with 0 Apps exports `identical: True` | Pick 9 |
| D14 | P2 | MiMo grading: out-of-range rewards are scored (MiMo's runner masks them); no MiMo `RunTask`, so `agent_output` is passed by hand and imported envs are not graded through the M10 runtime | `C/interop/mimo/verify.py:49,91`; only caller `T/test_interop_gate.py:245-259` | [ran p9]: 1.5 and −0.2 scored | Pick 5; a `MimoTask` whose `grade` takes `agent_output` from the World trace's last policy message (`S/run_verify.py:14-31`) |
| D15 | P2 | No ATIF v1.7 export for NeMo-Gym (ARCHITECTURE §3 TARGET row). Crucible's ATIF is v1.8 and NeMo-Gym rejects it; the steps it emits already fit v1.7 | `C/interop/harbor/atif.py:24-61` | [ran p4]: v1.8 rejected; forced v1.7 validates | Pick 8 |
| D16 | P1 (spec) | No `manifest.yaml`, config or dataset rows for the NeMo-Gym export. Stock NeMo-Gym cannot run it; only Crucible's own client can | `C/interop/nemo_gym/` (absent); ARCHITECTURE M11 `nemo_gym` bullet | [read] | Pick 4 |
| D17 | P1 (gate) | The NeMo-Gym leg of the M11 gate runs no NeMo-Gym code and validates only the unmasked verify body. The seed body is written to a file and never checked | `T/test_interop_gate.py:42-71,93-118` (`:111`) | [read]; p1 shows what it misses | Pick 2 |
| D18 | P2 | `/verify` derives `call_id`s from `episode_key`, which NeMo-Gym never sends (default `"e0"`). Every rollout of a task gets the same `call_id`s | `server.py:138` | [read] | Store the seed's `episode_id` in the session and use it |
| D19 | P2 | `/verify` replays the episode in a second World outside the slot budget. N concurrent verifies hold 2N Worlds. A replay or tool-route exception is a 500 after a finished grade | `server.py:131-139,170-180`; `C/interop/replay.py:81-82` | [read] | Count the replay World in the semaphore, or attach the progress tracker at seed and skip replay; wrap in pick 3 |
| D20 | P2 | `.envs/nemo` is not built from `NG/uv.lock` (Python 3.13.7 < 3.13.14; pydantic, fastapi, starlette, mcp, aiohttp differ). The model validation runs against other versions than NeMo-Gym pins. No script records how the env is built | `crucible/.envs/nemo/pyvenv.cfg`; site-packages | [ran] `ls` + `uv.lock` | Build with `uv sync --frozen` and Python ≥ 3.13.14; add the command to a script and DESIGN |
| D21 | P2 | NeMo-Gym import: restore re-seeds without closing the old session; `stop()` never closes; `/verify` sees tool outputs cut at 64 KiB while the policy saw the full text | `import_server.py:97-115,138,142-148,165-167` | [read] | `/close_session` before re-seed and on stop; keep full outputs (store pointer) for the verify response |
| D22 | P2 | Remote client: non-2xx tool replies other than 404, and a 404 at `/verify`, raise instead of mapping to mask (`env_session_lost`) or a shown tool error | `C/interop/nemo_gym/client.py:83,89-90` | [read] | Map like `:81-82`; 4xx shown to the policy |

---

## 4. Probe index (`scratchpad/m11sweep/g3/`)

| Probe | Shows | Result |
|---|---|---|
| `p1_export_contract.py` (crucible venv + `.envs/nemo`) | Seed, close, `reverify_mode`, aggregate replies vs NeMo-Gym models | seed FAIL (headers missing, `tools` extra, `instruction` extra); close FAIL; reverify `ValueError`; aggregate OK with all fields empty |
| `p2_seed_retry_leak.py` | Retried seed; legacy body; unknown task | `admitted 2`, `live 1`; after close, a second live session gets 503; legacy body 500; unknown task 500 |
| `p3_infra_fault_scored.py` | Tool crash during an NG-driven episode | policy sees `internal error`; `World.failures = [crucible:tool_crashed]`; verify `reward 0.0, mask_sample False` |
| `p4_atif_v17.py` | Crucible ATIF vs `AtifTrajectoryV1_7` | as emitted `ATIF-v1.8` rejected; forced `ATIF-v1.7` OK |
| `p5_import_errors_and_reward.py` | Importer vs an NG-shaped server | 422 and 500 tool replies mask the episode; reward NaN → 0.0, 1.8 → 1.0, −1 → 0.0, missing → 0.0, all scored |
| `p6_eog_zero_count.py` | Zero-count task imported as refusal | `refusal: True`; do nothing 0.0; closing the cases 0.0 |
| `p7_eog_compare.py` | `sql_checks.compare` vs EOG `_compare_values` | date `greater_than`: EOG True, Crucible False; `'1'` vs `1`, `1` vs `'1'`, `'1233'` vs `1233`, `contains 12`: EOG False, Crucible True |
| `p8_mimo_export_tautology.py` | Export of an import with every App removed | `{'files': 27, 'identical': True}` |
| `p9_mimo_reward_range.py` | MiMo verify scores 1.5, −0.2, NaN, inf | 1.5 and −0.2 scored with a `reward.json`; NaN and inf masked only by `write_reward_files`; MiMo's runner masks all four |
| `p10_nr_masked_baseline.py` | NR's own baseline function, exec'd from source | ones mask: advantages +0.333; mask honoured: 0.0 |
| `p11_ng_harbor_agent_missing_reward.py` (`.envs/nemo`) | `HarborAgentUtils.extract_reward` without rewards | 0.0 |
