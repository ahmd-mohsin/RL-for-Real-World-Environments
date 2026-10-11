<!-- M1 taxonomy 01: Harbor, SWE-smith, mini-swe-agent, Repo2RLEnv, GitSwarm, autoresearch. Read-only study by a subagent, 2026-10-12. -->

# Environment-family catalog for 6 codebases (read-only research)

**Path prefixes (all relative to ROOT):**
- `H/` = `harbor-framework-harbor/`
- `HA/` = `harbor-framework-harbor/adapters/`
- `SS/` = `swe-bench-swe-smith/`
- `MS/` = `swe-agent-mini-swe-agent/`
- `R2/` = `huggingface-repo2rlenv/`
- `GS/` = `veds12-git-swarm/`
- `AR/` = `karpathy-autoresearch/`

**Licenses:**
- Harbor: Apache-2.0 (`H/LICENSE`). Each adapter inherits its upstream license, noted per family.
- SWE-smith: MIT (`SS/LICENSE`).
- mini-swe-agent: MIT (`MS/LICENSE.md`).
- Repo2RLEnv: Apache-2.0 (`R2/LICENSE`).
- GitSwarm: no license file. The repo is a project page only (`GS/README.md`).
- autoresearch: MIT (`AR/README.md:90-92`).

---

## 0. Harbor substrate (shared by every Harbor family and every Repo2RLEnv output)

Almost every family below is expressed in Harbor's task format, so this section lists the primitives once.

- **Task layout:** `task.toml`, `instruction.md`, `environment/` (a Dockerfile or `docker-compose.yaml`), `tests/test.sh` and an optional `solution/solve.sh` (`H/CLAUDE.md` "Tasks").
- **Rewards:**
  - Written to `/logs/verifier/reward.txt` as one float, or to `reward.json` as a multi-key dict.
  - JSON values must be finite numbers (`H/src/harbor/verifier/verifier.py:89-122, 257-268`).
- **Hidden tests:** `tests/` is uploaded into the container only at verify time (`verifier.py:175-187`).
- **Verifier placement** (`[verifier]`, `H/src/harbor/models/task/config.py:562-611`):
  - `environment_mode = shared | separate`, plus a separate verifier image.
  - `[[verifier.collect]]` hooks snapshot sidecar state into files (`config.py:726-757`).
- **Artifacts:** per-service artifacts with excludes (`config.py:640-723`). `trial/regrade.py:1-12` re-grades recorded artifacts with a new separate verifier.
- **Multi-step tasks:**
  - `[[steps]]` carry per-step instruction, tests, timeouts, healthcheck and artifacts.
  - `min_reward` gating can abort the remaining steps (`config.py:760-791`; `H/src/harbor/trial/multi_step.py:195-241`).
  - Trial reward is `mean` or `final` over steps (`config.py:794-824`).
- **Environment config** (`config.py:348-376, 422-470`):
  - Healthcheck (Docker HEALTHCHECK semantics, runs before agent setup).
  - Resources: cpus, memory, storage, gpus, gpu_types and a TPU spec.
  - `mcp_servers` (stdio, sse or streamable-http; `config.py:617-637`), `skills_dir`, `workdir`, `os = linux | windows`, prebuilt `docker_image`.
- **Network policy:** `no-network | public | allowlist` (hostnames, wildcards, IPs, CIDRs), set per phase for agent, verifier and steps (`config.py:37-80, 219-269`).
- **Timeouts and users:** per-phase agent/verifier timeouts with multipliers (`H/src/harbor/trial/trial.py:568-658, 835-882`). Per-phase `user` (`config.py:340-346, 565-568`).
- **Actors:**
  - Oracle agent runs `solution/` (`H/src/harbor/agents/oracle.py:57-82`).
  - Simulated user: a persona plus Jinja template, connected to the target agent via an ACP bridge (`H/src/harbor/trial/simulated_user.py:10-151`; `H/src/harbor/bridges/base.py:10-50`; `H/src/harbor/job.py:148`).
  - Computer-use agent `computer-1` (xdotool click/double-click/drag/scroll/zoom/type/key/done; `H/src/harbor/agents/computer_1/runtime.py:61-99, 333-403`).
  - Terminus-2 drives a tmux pane, i.e. keystrokes (`H/src/harbor/agents/terminus_2/terminus_2.py:28`).
- **Sandboxes:** about 35 providers (docker, daytona, modal, e2b, gke, ec2, singularity, use_computer, cua_cloud, hyperbrowser, …; `H/src/harbor/environments/`). Capabilities are declared as flags (gpus, tpus, disable_internet, allowlist variants, dynamic_network_policy, windows, mounted, docker_compose; `H/src/harbor/environments/capabilities.py:12-60`). No generic snapshot/restore exists; only provider-level image snapshots such as `environments/langsmith.py:144-203`.
- **Grading helpers:**
  - **Rewardkit:** criteria, LLM or agent judges, nested dimensions → multi-key `reward.json`, weighted-mean or all-pass aggregation, overlayfs workspace isolation (`H/packages/rewardkit/README.md:1-110`; `H/packages/rewardkit/src/rewardkit/isolation.py:1-40`).
  - **Task-quality rubric:** includes `anti_cheating_measures`, `tests_or_solution_in_image` and `hardcoded_solution` (`H/src/harbor/cli/quality_checker/default-rubric.toml:1-52`).
  - **Red-team "cheating agent" prompt** (`H/examples/prompts/cheating_agent.md.j2:1-12`).

---

## 1. Harbor example-task families (`H/examples/tasks/`, Apache-2.0)

### H-E1. Single-container terminal/file task (the "hello-world" archetype)
- **Tasks:** hello-world, hello-alpine, hello-workdir, hello-user, hello-skills, hello-world-bat (Windows). The same archetype covers the registry-only datasets listed in §2-R.
- **State:** container filesystem. **Actions:** bash or tmux via whichever agent runs. **Actors:** single agent.
- **Time:** agent/verifier `timeout_sec` (`hello-world/task.toml`).
- **Observations:** text.
- **Grader:** `tests/test.sh` plus pytest `test_state.py` → binary `reward.txt`. Terminal reward only.
- **Infra:** Dockerfile; `skills_dir="/skills"` in hello-skills; `user="agent"` in hello-user; `os="windows"` in hello-world-bat.
- **Anti-cheat:** tests uploaded only at verify time.
- **Port:** `H/src/harbor/models/task/config.py` (whole task schema, Apache-2.0).

### H-E2. Multi-step staged task
- **Tasks:** hello-multi-step-simple, -bat, -advanced; separate-verifier-multi-step.
- **State:** files persist across steps.
- **Grader:** per-step tests, `steps/<name>/tests/test.sh`, with shared top-level tests as fallback.
- **Time and gating:**
  - Per-step agent/verifier timeouts.
  - `min_reward = 1.0 / 0.5` aborts later steps (`hello-multi-step-advanced/task.toml` `[[steps]]`).
  - A per-step healthcheck (`test -x /app/greet.sh`) runs before the step's agent.
- **Reward density:** one reward per step (milestone-like), aggregated by mean or final.
- **Infra:** step-scoped `workdir/` uploads with `setup.sh`, per-step verifier env, per-step artifacts.
- **Port:** `H/src/harbor/trial/multi_step.py` (step loop, gating, artifact archiving).

### H-E3. Compose/sidecar service world with separate verifier
- **Tasks:** sidecar-artifacts, environment-env-multi, hello-healthcheck, separate-verifier-environment, verifier-mode-matrix (about 8 variants).
- **State:** processes in sidecars (e.g. an HTTP API server holding order stats).
- **Grader:**
  - `[[verifier.collect]] service="api" command="curl …/stats > /tmp/stats.json"`.
  - Artifacts are collected from `service="api"` and the separate verifier reads only those artifacts (`sidecar-artifacts/task.toml`).
- **Infra:** compose; healthcheck `curl -sf http://localhost:8080/` with retries and start_period (`hello-healthcheck/task.toml`).
- **Anti-cheat:** the verifier runs in a fresh container with only the declared artifacts, and can be `no-network` (`separate-verifier-environment/task.toml`).
- **Port:** `H/src/harbor/trial/artifact_handler.py` and `trial/regrade.py` (artifact manifest, then regrade).

### H-E4. MCP-tool world
- **Tasks:** hello-mcp (FastMCP sidecar exposing `get_secret`; `hello-mcp/environment/mcp-server/server.py:10-11`), daytona-computer-use (stdio MCP `daytona-cli mcp start`).
- **Actions:** MCP tool calls declared via `[[environment.mcp_servers]]` with `url = "http://mcp-server:8000/mcp"`.
- **Grader:** pytest on outputs.

### H-E5. Computer-use desktop/browser
- **Tasks:**
  - `computer-1-click-calibration`: a local HTML suite, `network_mode="no-network"`. The grader reads the window title via `wmctrl` for a PASS CODE and compares it with `/logs/agent/final_answer.txt` (`tests/test.sh`).
  - `hello-computer-use` (use.computer macOS/Ubuntu VM) and `hello-cua-cloud` (Cua Cloud Ubuntu/Windows VM pool, claimed and released with a TTL keepalive; `H/src/harbor/environments/cua_cloud.py:1-30`).
- **State:** VM or browser. **Actions:** keyboard/mouse (computer-1 actions). **Observations:** screenshots plus text.
- **Port:** `H/src/harbor/agents/computer_1/runtime.py` (action schema → xdotool, coordinate scaling).

### H-E6. Simulated-user conversational coding (deepswe)
- **Task:** `deepswe_tomlkit-toml-table-converters`.
- **State:** git repo in a prebuilt image.
- **Actors:** policy plus a simulated developer persona (`persona.md`: "privately inspect the repository state before you answer … never mention hidden tests").
- **Time:** agent 5400 s, verifier 1800 s.
- **Grader:** `tests/test.patch` applied, then tests run → binary reward.
- **Port:**
  - `H/src/harbor/trial/simulated_user.py` (template validation, persona injection).
  - `H/src/harbor/bridges/` (protocol bridge between user agent and target agent).

### H-E7. Judge-graded and rubric tasks
- **llm-judge-example:** `verifier.env ANTHROPIC_API_KEY`, `MODEL_NAME=claude-haiku-4-5`, grader `tests/llm_judge.py`.
- **reward-kit-example:**
  - Nested dimensions `correctness/`, `structure/` with `reward.toml` weighted-mean `{most_common=2, word_count=3, pipeline=1}`.
  - Criteria return fractional scores, giving partial credit (`tests/criteria.py:19-51`).
  - Runner: `uvx … rewardkit /tests`.
- **Port:** `H/packages/rewardkit/` (Apache-2.0). Multi-key reward, judge TOML, overlay isolation.

### H-E8. Multimodal input
- **Task:** `describe-image`. The agent reads `/workspace/image.png`; the grader checks for an expected string and writes `reward.json`.

### H-E9. GPU task
- **Task:** `hello-cuda` with `gpus=1`, `gpu_types=["A100","H100"]`.

### H-E10. Harness-conformance fixtures
- **Tasks:**
  - environment-env-single/multi: env injection without leaking to sidecars.
  - network-policy-matrix: about 25 static and dynamic variants of env/agent/verifier/step network policies, including allowlist IPv4 CIDR and an uncontrolled compose sidecar.
  - islo-network-policy-demo: agent allowlist `api.anthropic.com`, `gist.github.com`, …
  - hello-load-atif-trajectory-*: trajectory replay.
- **What they are:** these exercise primitives rather than RL problems, but they define the acceptance surface.

---

## 2. Harbor adapter families (`HA/`, 86 adapters; counts from each README and `H/registry.json`)

All of these share §0 unless stated otherwise. "Binary" means reward.txt 0/1, terminal only.

### HA-1. Repo-level issue/feature resolution (patch, then F2P/P2P tests)
- **Members and counts:**
  - `swebench` 500, `swebench_multilingual` 300, `swebenchpro` 731, `swegym` 2438 full / 230 lite, `swesmith` 100.
  - `multi-swe-bench` 1632 (7 languages), `featbench` 156, `featurebench` 200 / lite 30 (44 need a GPU; `HA/featurebench/README.md:14`).
  - `devopsgym` 733 (Java/Go), `crustbench` 100 (C→Rust, GPL), `abc-bench` 224 (backend, DinD).
  - `swelancer` IC 198 plus manager 265.
  - Registry-only (not inspectable locally): swe-gen-js 1000 (`H/registry.json:71816`), vmax-tasks 1043 (`:98565`), scale-ai/swe-atlas-qna 124 (`:308329`), legacy-bench 10 (`:309627`).
- **State:** git repo in a per-instance image (`FROM {docker_image}`).
- **Actions:** bash and file edit. **Actors:** single agent.
- **Time:** `timeout_sec={max_timeout}`; swelancer uses 3000 s.
- **Grader:**
  - Embedded SWE-bench report parser tracks `patch_exists` / `patch_successfully_applied`, then runs FAIL_TO_PASS and PASS_TO_PASS (`HA/swebench/src/swebench_adapter/task-template/tests/test.sh:33-91`; `HA/swegym/template/tests/test.sh:128-166`).
  - `featurebench` rejects agent edits to masked test files by diffing `git status` against an image-baseline file (`HA/featurebench/template/tests/test.sh:10-40`).
  - swelancer manager tasks are exact match on `selected_proposal_id` (`HA/swelancer/src/swelancer/task-template/manager/tests/test.sh:4-18`).
  - `abc-bench` waits for an in-container dockerd (`HA/abc-bench/src/abc_bench/task-template/tests/_docker_wait.sh:1-10`).
- **Reward density:** binary, terminal only.
- **Anti-cheat:** tests hidden until verify. swebenchpro writes the reward via an EXIT trap (`HA/swebenchpro/src/swebenchpro/task-template/test.sh:7-24`).
- **Licenses:** MIT for most; GPL for crustbench; upstream terms otherwise.

### HA-2. Test-writing tasks
- **Members:** `swtbench` 433; `deveval` 63 (3 phases × 21 repos: implementation / unit_testing / acceptance_testing); registry swe-atlas-tw 90 (`H/registry.json:309080`).
- **Grader:** reverse grading. The model's test patch must fail on base code and pass on gold code (`HA/swtbench/src/swtbench/task-template/tests/test.sh:6-53`). The gold code patch is embedded only in the test script.

### HA-3. Performance-optimization tasks (continuous or threshold)
- **`algotune` (154, MIT):**
  - Reward = raw speedup over baseline, an unbounded scalar (`HA/algotune/src/algotune/task-template/tests/test_outputs.py:116-153`).
  - 8 cpus, 16 GB, 3600 s.
- **`gso` (102):**
  - The agent's diff is captured via `git diff` (`HA/gso/template/tests/test.sh:30-41`).
  - `opt_commit` is turned into a binary reward (`tests/gso_evaluate.py:49-55`).
  - Agent 10800 s (3 h).

### HA-4. Cleanroom program reconstruction (ProgramBench, 200)
- **State:** cleanroom image holding a reference binary and docs.
- **Network:** agent `network_mode="allowlist"` to LLM APIs only. Agent 14400 s (4 h).
- **Grader:**
  - Separate verifier built `FROM` the cleanroom image with `/workspace` wiped.
  - The artifact handoff copies `/workspace` but excludes `executable`, so the reference binary cannot be passed through (`HA/programbench/src/programbench/task-template/task.toml:1-90`).
  - Hidden behavioral tests; reward = pass rate. Multi-key `{reward, resolved, almost_resolved≥0.95}` (`HA/programbench/README.md:11-33`).
  - The oracle is known to score below 1.0 (`README.md:17-21`).
- **Port:** this `task.toml` pattern: separate verifier, artifact exclude, hash-scrub.

### HA-5. Function/problem-level code generation with hidden unit tests
- **Members and counts:**
  - `aider_polyglot` 225 (6 languages, MIT), `autocodebench` 200, `bigcodebench_hard` 145, `ds1000` 1000, `evoeval` 100.
  - `humanevalfix` 164, `quixbugs` 80, `livecodebench` 100 (of 1055), `usaco` 304, `scicode` 80 (reward.json).
  - `research-code-bench` 212: reward.json `{reward, code_lines}` for line-weighted metrics (`HA/research-code-bench/template/tests/test.sh:51-85`).
  - `frontier-cs-algorithm` 172.
  - Registry code-contests 9644 (`H/registry.json:113093`).
- **frontier-cs-algorithm details:**
  - Privileged `judge` sidecar (go-judge) (`HA/frontier-cs-algorithm/.../environment/docker-compose.yaml`).
  - Partial score 0-100 (`tests/evaluate.py:17-44`). Agent 10800 s.
- **Grader:** pytest or judge. Mostly binary, except partial scores in scicode and frontier-cs.

### HA-6. Exact/structured-answer QA and reasoning (answer file vs hidden key)
- **Members and counts:**
  - Math and science: `aime` 60, `gpqa-diamond` 198 (CC-BY-4.0), `mmmlu` 210,630 (150 in registry), `arc_agi_2` 167 (pass@2).
  - Logic and reasoning: `satbench` 2100, `reasoning-gym` 576 (96 generators × easy/hard).
  - Domain QA: `labbench` 181 (figure QA, CC-BY-SA), `lawbench` 120, `pixiu` 435 (of 54,083).
  - Audio and agentic QA: `mmau` 1000 (audio), `dabstep` 450, `gaia` 165.
  - Function calling: `ace-bench` normal 823 / special 150, `bfcl` 3641.
  - Other: `kramabench` 104 (633 sub-tasks), `gdb` 33,786 (39 graphic-design sub-benchmarks), `qcircuitbench` 28.
- **ace-bench note:** the agent split (150) is not built because it needs a live simulated user (`HA/ace-bench/README.md:35-41`).
- **bfcl:** AST match on function calls emitted as free text (`HA/bfcl/README.md:9`).
- **gdb:** outputs are SVG/JSON/PNG/MP4; metrics include IoU, SSIM, BERTScore, … (`HA/gdb/README.md:5-15`).
- **Observations:** text, images (labbench, gdb), audio (mmau).
- **Grader:** deterministic scorer, mostly binary. `reasoning-gym` and `pixiu` give continuous reward. `gdb` always writes reward.json and fails closed (`HA/gdb/src/gdb_adapter/task-template/tests/test.sh:6-20`).

### HA-7. LLM-judge-graded answers
- **Members and counts:**
  - Single-question judges: `simpleqa` 4326, `hle` 2500 (judge o3-mini), `aa-lcr` 99 (about 100k-token contexts), `omnimath` 4428, `ineqmath` 100.
  - Safety and adversarial: `strongreject` 12,207 (313 prompts × 39 jailbreaks; 150 parity), `seal0` 111 (Apache-2.0, Claude judge).
  - Agentic research: `bixbench` 205 (Jupyter via the `nbcli` server; `HA/bixbench/src/bixbench/task-template/environment/nbcli/`), `clbench` 1899 (judge with rubric), `financeagent` 50.
  - Structured search: `deepsynth` 40 (F1 over JSON keys), `widesearch` 200 (MIT, reward.json).
  - `locomo` 10 conversations: reward.json `{reward, score_sum, num_questions}` plus a dataset-level metric (`HA/locomo/README.md:41`).
  - `webgen_bench` 101: the verifier launches the agent's app and runs a WebVoyager Selenium + VLM agent; fraction of YES/PARTIAL verdicts (`HA/webgen_bench/README.md:17-39`).
  - `scienceagentbench` 102 (Category B uses a visual judge).
- **financeagent:** compose with SEC EDGAR, SerpAPI and model keys (`HA/financeagent/template/environment/docker-compose.yaml`). An oracle shortcut skips the judge on an exact match (`tests/test.sh:19-25`).
- **Grader:** `verifier.env` carries `OPENAI_API_KEY` / `MODEL_NAME`, so the verifier needs network.

### HA-8. Data / SQL / dbt / spreadsheet engineering (stateful DB/files)
- **Members:** `spider2-dbt` 64, `adebench` 48 (dbt + DuckDB, buggy project; filters tests by `DB_TYPE`; `HA/adebench/src/adebench/task-template/tests/test.sh:24-63`), `bird_bench` 1534 (150 parity; SQL execution match), `spreadsheetbench-verified` 400 (CC BY-SA), `dacode` 479 (MIT; vendored metric suite under `tests/evaluators/metrics/{dbt,ml,table,image,text}.py`).
- **State:** SQLite, DuckDB, xlsx files.

### HA-9. Scientific discovery with continuous metrics
- **Members:**
  - `codepde` 5: one solver × 100 instances, aggregate nRMSE < 0.05 (`HA/codepde/README.md:12`).
  - `llmsr_bench` 240: reward = R², rounded to 1.0 at a threshold (`HA/llmsr_bench/template/tests/test.sh:36-438`).
  - `sldbench` 8 (clipped R²).
  - `refav` 1500 (CC BY-NC-SA; HOTA-Temporal on AV2 logs; 16 GB).
  - `qcircuitbench` 28 (semantic score).
- **Reward density:** continuous terminal scalar.

### HA-10. Long-horizon ML research (GPU)
- **Members:**
  - `mlgym-bench` 11/12 (CC-BY-NC): 24 cpus, 60 GB, `gpus={gpus}`, continuous metric by default (`HA/mlgym-bench/README.md:7-47`).
  - `rexbench` 2 released (of 12): `timeout_sec = 43200.0 #12 hours`, `gpus=1`, 40 GB (`HA/rexbench/src/rexbench/task-template/task.toml:20-32`).
  - `ml_dev_bench` 33 (MIT, 16 GB, compose bind-mount, 7 tasks without an oracle).
  - `replicationbench` 90.
- **Grader:** evaluation script versus baseline, writing reward.txt. rexbench uses a fail-closed `trap` (`rexbench/.../tests/test.sh:5`).

### HA-11. Simulated-user customer-service tool world (tau3-bench, 375, MIT)
- **Domains:** airline, retail, telecom, banking_knowledge (`HA/tau3-bench/src/tau3_bench/adapter.py:36-58`).
- **State:** tau2 domain DB inside the `tau3-runtime` sidecar.
- **Actions:** MCP tools `start_conversation`, `send_message_to_user`, domain tools and `end_conversation` (`task-template/instruction.md`).
- **Actors:** policy plus an LLM `UserSimulator` with its own user tools (`environment/runtime-server/server.py:146-200`).
- **Time:** `max_steps=200`, `max_errors=10`; termination reasons are agent stop, user stop, max steps or too many errors (`server.py:160-164, 298-308`).
- **Grader:** tau2 `evaluate_simulation` combining DB/action/communicate checks and NL assertions via an LLM (`tests/evaluate.py:126-164`).
- **Infra:** sidecar healthcheck (`environment/docker-compose.yaml`).
- **Port:** `runtime-server/server.py` (wraps a turn-based simulator plus user simulator as an MCP App with a persisted state log).

### HA-12. Simulated-time event-driven assistant world (GAIA2 on Meta ARE, 800)
- **State:** ARE scenario apps (email, messaging, calendar, filesystem) inside the `are-sidecar`.
- **Actions:** every ARE app tool is auto-registered as an MCP tool, plus `are_wait_for_notification` and `are_get_time` (`HA/gaia2/template/environment/are-sidecar/are_mcp_server.py:200-437`).
- **Time (simulated clock):**
  - The environment starts paused with `start_time` and `time_increment_in_seconds` (`:61-78`).
  - Real "thinking time" is added as a sim offset capped at 300 s (`:155-168`).
  - Waiting advances event by event to the next scheduled event or notification (`:327-360`).
- **Grader:** dual-mode LLM judge (`HA/gaia2/template/tests/llm_judge.py:1-9`).
  - search / ambiguity: oracle answer comparison.
  - execution / time / adaptability: match the agent's write-actions from the event log against oracle actions.
- **Infra:** event log on a shared volume (`docker-compose-cli.yaml`). An alternative non-MCP path uses `run_parity.py`.
- **Port:** `are_mcp_server.py` (pause-and-tick clock, notification drain, event log; license follows upstream ARE).

### HA-13. Stateful API-server and enterprise worlds
- **`medagentbench` 300 (MIT):**
  - A FHIR server runs inside the image; the agent uses GET/POST/FINISH via `medagentbench_cli.py` (`HA/medagentbench/template/environment/medagentbench_cli.py:1-30`).
  - The grader checks results plus the required POST transcript (`README.md:142-153`) and polls server readiness (`tests/test.sh:13`).
- **`crmarena` 1170 (CC BY-NC):**
  - Live external Salesforce org reached with `simple_salesforce` and `SF_*` credentials (`HA/crmarena/template/task.toml:24`; `instruction.md:3-23`).
  - Reward: F1 for `knowledge_qa`, exact match for the rest (`README.md:12-30`).
- **`theagentcompany` 174:**
  - Shared host-run GitLab, Plane, OwnCloud and RocketChat, with RocketChat NPC coworkers (`HA/theagentcompany/README.md:7-32`).
  - The healthcheck runs `/utils/init.sh`, which resets services (`template/task.toml:63-67`).
  - Reward = earned / total checkpoints, with a `bonus_for_completing_final` rule (`template/tests/test.sh:37-65`; `README.md:184-191`).
  - Must run sequentially (`-j 1`) because resets would collide (`README.md:308`).
  - Per-difficulty agent timeouts of 1800 / 3600 / 7200 s (`README.md:48`).
  - Graders are encrypted, using `DECRYPTION_KEY` (`README.md:45`).
- **Reward density:** TheAgentCompany gives partial, milestone-like credit; the others are terminal.

### HA-14. Interactive hidden-state games
- **`textarena` 62 single-player games:**
  - The game lives in a `game` sidecar; the agent plays via `/app/play.py` over stdin and HTTP.
  - The result is fetched only with a `RESULT_TOKEN` held in `verifier.env` (`HA/textarena/src/textarena/task-template/environment/game_server.py:13-61`; `tests/test.sh:17-46`; `task.toml [verifier.env]`).
  - Reward = game reward, which can be fractional.
- **`kumo` 5300 (Apache-2.0):**
  - A verifier service is the environment. The agent sends `POST /act` via `kumo_cli.py`.
  - The `secret.json` truth exists only in the verifier image. There is a `max_steps` action budget and an `actions.jsonl` log; the final truth is exact match (`HA/kumo/README.md:7-39`; `template/environment/docker-compose.yaml`).
- **Port:** the token-gated result endpoint and secret-only-in-verifier-image patterns.

### HA-15. Multi-agent cooperative coding (CooperBench, 652 tasks; 199 features across 30 tasks)
- **Actors:** two LLM agent sidecars (`agent1`, `agent2`, mini-swe-agent based) plus `main`.
- **Messaging:** Redis inbox with a `send_message <recipient>` command; inbound messages are auto-injected (`HA/cooperbench/src/cooperbench/task-template/environment/docker-compose.yaml:1-70`; `environment/cooperbench_messaging.py:1-40`; `environment/send_message`).
- **Time:** `AGENT_TIMEOUT=1650`, `MAX_ITERATIONS=100`.
- **Grader:**
  - Wait for both `agent.done` files, then git-apply each patch and merge (naive, then union fallback).
  - Run both features' tests; write reward.json `{reward, feature1_passed, feature2_passed, merge_ok}` (`tests/test.sh:46-80`).
- **Port:** the compose plus Redis messaging pattern for N-agent worlds.

### HA-16. Security exploitation (CyberGym, 1507; Apache-2.0)
- **State:** a `task-server` sidecar holds the vulnerable and fixed binaries and ground truth. The agent submits PoCs over HTTP with an auth token.
- **Network:** `restrict-network.sh` applies an iptables allowlist (NET_ADMIN) (`HA/cybergym/template/environment/docker-compose.yaml:1-50`; `environment/restrict-network.sh`).
- **Difficulty levels 0-3** reveal more files (`README.md:31-48`).
- **Grader:** reward = 1 if any PoC crashes the vulnerable build and not the fixed one (`tests/test.sh:34-91`). The verifier fails as an infra error if the server is unhealthy (`:10-11`).
- **Related:** registry binary-audit 46 (backdoor finding; `H/registry.json:4874`).

### HA-17. Full desktop VM (OSWorld-Verified, 361 / 369)
- **State:** Ubuntu qcow2 guest under QEMU/KVM inside Docker (`devices: /dev/kvm, /dev/net/tun`, NET_ADMIN, a 32 GB disk, healthcheck POST `/screen_size` with `retries: 720`, `start_period: 600s`; `HA/osworld/src/osworld/task-template/environment/docker-compose.yaml`). Build timeout 7200 s.
- **Actions:** screenshot plus pyautogui-style actions via the custom `osworld-agent` (`README.md:33, 157-164`).
- **Grader:** vendored upstream getters and metrics for chrome, libreoffice, gimp, vscode, vlc, thunderbird, … reading live VM state. Output is reward.txt plus reward.json (`tests/test.sh:6-15`; `tests/evaluators/upstream/`).
- **Port:** the upstream evaluator vendoring (`tests/evaluators/`, upstream license).

### HA-R. Registry-only terminal datasets (task code not in this checkout)
- **Entries:** terminal-bench 2.0 (89; `H/registry.json:94012`), terminal-bench-pro 200 (`:94553`), openthoughts-tblite 100 (`:170964`), termigen-environments 3566 (`:236471`), seta-env 1376 (RL training; `:104830`), otel-bench 26 (`:39022`), compilebench 15 (`HA/compilebench/README.md:9-28`; fetched in Harbor format upstream).
- **Archetype:** presumably H-E1. Not verified locally.

---

## 3. SWE-smith (`SS/`, MIT)

### SS-1. Synthetic bug-fix SWE-gym (any repo → unlimited tasks; published set 52k instances)
- **Count:** 52k (`SS/README.md:51, 64`). About 847 repository profiles: cpp 214, typescript 136, python 134, java 107, golang 89, javascript 83, rust 52, ruby 23, php 4, c 3, csharp 2 (`SS/swesmith/profiles/*.py`).
- **State:**
  - One image per repo, named `swesmith.{arch}.{owner}_1776_{repo}.{commit}`.
  - Each instance is a git branch checked out in a container (`SS/swesmith/profiles/base.py:80-110, 207, 493-520`).
- **Actions:** bash and file edit via SWE-agent tool bundles `registry`, `edit_anthropic`, `submit` (`SS/agent/swesmith_infer.yaml:97-102`).
  - Generation runs use `per_instance_call_limit: 75` and `cost_limit: 2.` (`SS/agent/swesmith_gen_claude.yaml:61-71`).
- **Time:** test-suite timeout 90 s per instance and 900 s for the full reference run (`profiles/base.py:108-109`).
- **Grader:**
  - Apply the patch, run `/eval.sh` in the container with a timeout (`SS/swesmith/harness/utils.py:101-268`).
  - F2P/P2P report with FULL / PARTIAL / NO resolution status (`SS/swesmith/harness/grading.py:109-234`).
  - Validation keeps only bugs that break one or more tests (statuses `0_f2p` / `1+_f2p` / timeout; `SS/swesmith/harness/valid.py:56-115`).
- **Anti-cheat:** the instance branch commits "Remove F2P Tests", deleting fail-to-pass test files from the agent's view (`SS/swesmith/harness/gather.py:540-551`).
- **Distinctive feature, bug injection** (all MIT):
  - 135 procedural AST modifier classes across 10 languages: control-flow invert/shuffle, operator flip/swap, change constants, remove loop/conditional/assign, class remove-bases/shuffle-methods, ruby nil-introduction, … (`SS/swesmith/bug_gen/procedural/base.py:14-160`; `procedural/<lang>/*.py`).
  - LLM `modify` / `rewrite` (`bug_gen/llm/`); PR mirroring (`bug_gen/mirror/`); same-file and same-module combination of bugs (`bug_gen/combine/`).
  - Issue-text generation from tests, PRs or static sources (`swesmith/issue_gen/`).
- **Port:**
  1. `SS/swesmith/bug_gen/procedural/` (mutation operators).
  2. `SS/swesmith/harness/{valid,grading}.py` (break-tests validation plus F2P/P2P).
  3. `SS/swesmith/profiles/base.py` (RepoProfile: image build, test command, log parser).

---

## 4. mini-swe-agent (`MS/`, MIT). An agent harness and runner, not an env author

### MS-1. SWE-bench-family runner
- **Datasets:** full, verified, lite, multimodal, multilingual, smith, rebench (`MS/src/minisweagent/run/benchmarks/swebench.py:53-62`).
- **State:** `swebench/sweb.eval.x86_64.*` image, cwd `/testbed`.
- **Actions:** bash only, one command per step, no tools (`MS/src/minisweagent/environments/docker.py:101`). Commands time out after 60 s.
- **Time:** `step_limit 250`, `cost_limit 3.` (`MS/src/minisweagent/config/benchmarks/swebench.yaml:112-117`).
  - The agent has `wall_time_limit_seconds` (`MS/src/minisweagent/agents/default.py:26-31, 132-141`).
- **Termination:** the first output line is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (`MS/src/minisweagent/environments/local.py:45-54`). The submission is `patch.txt` (`swebench.yaml:81-101`), written to `preds.json`.
- **Grading:** external (sb-cli / swebench harness).

### MS-2. ProgramBench runner
- **Container:** `docker --network none --cpus 20 --memory 60g --user agent` (`MS/src/minisweagent/config/benchmarks/programbench.yaml:186-197`).
- **Time:** `wall_time_limit_seconds: 21600` (6 h), `step_limit 1000`, `container_timeout 7h`. The prompt injects remaining steps and time (`programbench.yaml:4, 181-185, 238-250`).
- **Output:** the submission is a tar of `/workspace` for `programbench eval` (`MS/src/minisweagent/run/benchmarks/programbench.py:1-60`).

### MS-3. Local / human-in-the-loop shell
- **Modes:** `human` / `confirm` / `yolo`; `confirm_exit` (`MS/src/minisweagent/agents/interactive.py:3-34`).
- **Sandboxes:** local, docker, singularity, bubblewrap (`--unshare-user-try`), swerex docker/modal, contree (`MS/src/minisweagent/environments/`).
- **Port:** `MS/src/minisweagent/agents/default.py` (about 200-line loop with limit exceptions) and `environments/local.py:_run` (kills the whole process group on timeout).

---

## 5. Repo2RLEnv (`R2/`, Apache-2.0). A generator that emits Harbor tasks

**Families shared across all pipelines:**
- **State:** a git repo or terminal fixture in a bootstrap image built by an LLM ReAct agent with a spend cap (`R2/src/repo2rlenv/bootstrap/agent.py`).
- **Actions:** bash and file edit. **Actors:** single learner (owned recipes run `user="learner"`).
- **Anti-contamination:** enforced "by the environment, never requested in the prompt" (`R2/CLAUDE.md:48-56`).
  - Git-history scrub to the base commit, deleting future refs and reflog (`R2/src/repo2rlenv/pipelines/_env_guard.py:64-90`).
  - Compose `extra_hosts` denylist blackholing PyPI and the GitHub/GitLab forges (`_env_guard.py:1-50, 91`).
  - Owned recipes run agent and verifier with `no-network` and a **separate** fresh verifier (`R2/src/repo2rlenv/pipelines/recipes/repository/export.py:269-276`; `recipes/scaler/export.py:70-75`).

| ID | Pipeline / recipe (upstream license) | Grader and reward | Evidence |
|---|---|---|---|
| R2-1 | `pr_diff` (text-only diff tasks) | 6-component weighted [0,1]: format_valid, size_sanity, file_targeting F1, region_overlap, difflib similarity, LLM judge. Default weights 0/.08/.12/.20/.10/.50; capped at ≤0.40 when size_sanity < 0.10. Multi-key `reward-details.json`. | `R2/src/repo2rlenv/pipelines/_pr_diff_verifier.py:1-48, 383-397`; `R2/docs/reference/REWARD_SCHEMA.md:20-65`; `R2/src/repo2rlenv/reward.py` |
| R2-2 | Runtime F2P/P2P: `pr_runtime`, `commit_runtime`, `cve_patches`, plus recipes swe_next (Apache), r2e_gym (Apache), sec_bench (MIT), swe_gen (`pr_to_env`, Apache), tasksmith | Dense `reward = f2p_rate × p2p_rate`, plus `resolved` and `command_resolved` flags, regressions and untracked failures. Log parsers for pytest, go, cargo and jest. | `REWARD_SCHEMA.md:80-143`; `R2/src/repo2rlenv/pipelines/_pr_runtime_verifier.py`; `R2/src/repo2rlenv/log_parsers/` |
| R2-3 | Synthesized: `code_instruct`, `equivalence_tests`, recipe r2e (MIT) | Binary hidden test, kept only if it fails without the solution and passes with it. Equivalence is checked against a `reference_<name>` oracle. | `REWARD_SCHEMA.md:150-160`; `R2/docs/pipelines/code_instruct.md`, `equivalence_tests.md` |
| R2-4 | `repo_mutate / swe_smith` (MIT) | Seeded single-site mutation, then issue writer. The original code is the reference. Separate verifier. | `R2/src/repo2rlenv/pipelines/recipes/swe_smith/mutations.py`, `grade.py`; `R2/docs/pipelines/repo_mutate.md` |
| R2-5 | `repo_reconstruct`: swe_flow (MIT, 100 tasks), codemidas | Feature removal or runtime-traced missing functions, graded by private behavioral tests. | `R2/docs/pipelines/repo_reconstruct.md`, `codemidas.md` |
| R2-6 | `terminal_synth` (seta_seed2synth, tmax, endless_terminals, dataarc), `task_evolve / seta_evol`, `terminal_reconstruct / terminalworld` (from asciinema recordings) | Generated reference plus tests. Fresh baseline must fail and reference must pass. Agent and verifier `no-network`, verifier runs as root. | `R2/src/repo2rlenv/pipelines/recipes/terminal/draft.py:162-163`; `R2/docs/pipelines/terminal_synth.md`, `terminalworld.md` |
| R2-7 | `env_repair / cli_gym` (MIT) | Break a healthy environment; the agent repairs it. Grader restores original tests and **fails if protected source hashes changed** (`reason.txt`). | `R2/src/repo2rlenv/pipelines/recipes/cli_gym/grade.py:1-30`; `cli_gym/export.py:92-93` |
| R2-8 | `reasoning_synth / scaler` (Apache) | Answers come from released generator and reference programs, no LLM calls. `result.json {score, accuracy}`. Separate no-network verifier. | `recipes/scaler/export.py:70-75`; `R2/docs/pipelines/scaler.md` |
| R2-9 | `optimization_synth / frontiersmith` | Continuous [0,1] feasibility × quality score. The verifier runs the submission under rlimits (CPU 3 s, AS 256 MB, NPROC 32, …). | `recipes/frontiersmith/grade.py:1-40`; `R2/docs/pipelines/frontiersmith.md` |
| R2-10 | Tasksmith (agent-authored task from one PR; picks its own dependencies, tests and **GPU count**) | Private verifier plus quality loop (review, bounded repair, blind rollout). | `R2/docs/pipelines/tasksmith.md`; `R2/examples/tasksmith-gpu-options.json` (`"gpus": 1`, Modal); `R2/src/repo2rlenv/quality/loop/` |

- **Recipe catalog:** 17 recipes, each with upstream commit and license (`R2/src/repo2rlenv/pipelines/recipes/catalog.json`).
- **Port:**
  1. `_env_guard.py` (history scrub and egress denylist).
  2. `_pr_diff_verifier.py` (dense diff reward that needs no execution).
  3. `_pr_runtime_verifier.py` (graded F2P × P2P).
  4. `cli_gym/grade.py` (protected-file hash contract).
  5. `frontiersmith/grade.py` (rlimit sandboxed scorer).

---

## 6. GitSwarm (`GS/`). Project page only; no license file and no environment code

### GS-1. Shared-repo multi-agent compounding inference
- **State:** one shared, branchable git repository. Every contribution is an immutable commit carrying an `informed_by=` link to other commits ("semantic inheritance") (`GS/source/template.html:357`).
- **Actors:** a pool of identical workers with no roles. C = 5 concurrent workers, replaced asynchronously until a budget of B episodes is spent (`:377`). Each episode runs in its own isolated git worktree (`:406`).
- **Actions:** bash and git, plus read-only shared views `gs_dashboard` and `gs_inflight [inspect]` (`:392-399`).
- **Episode endings:** exactly one of Contribute, Nominate (uses budget) or Abstain (does not) (`:423-441`).
- **Grader:**
  - Publication validation checks only required files and allowed paths (`:423`).
  - Final answer = most nominations (`:450`); for research tasks, best validation score with a held-out check (`:481`).
  - External scoring: ProgramBench 50-task hidden tests and IMOProofBench-Advanced 30 proofs (`:470-471`).
- **Infra:** research tasks use 12-20 workers sharing a GPU pool via SLURM for 24-72 h (`:480`).
- **Data:** `GS/source/data/{activity,fig3,graphs,research}.json`.
- **Port:** nothing; there is no code. The concepts worth reusing are worktree-per-episode, an `informed_by` provenance edge, the nomination ledger kept outside the commit graph, and the live-peer dashboard.

---

## 7. autoresearch (`AR/`, MIT)

### AR-1. Autonomous single-GPU LLM-training research loop (1 task)
- **State:** a git branch `autoresearch/<tag>`. The editable file is `train.py`. `prepare.py` (data, tokenizer, `evaluate_bpb`) is read-only. Data cache lives in `~/.cache/autoresearch` (`AR/program.md:9-31`; `AR/prepare.py:30-45`).
- **Actions:** edit, `git commit`, `uv run train.py > run.log`, then `git reset` on regression (`AR/program.md:94-104`).
- **Time:** fixed `TIME_BUDGET = 300` s of training per experiment (`AR/prepare.py:31`; `AR/train.py:555, 603`). Kill after 10 min (`program.md:108`). "NEVER STOP": the loop runs indefinitely, about 100 experiments overnight (`program.md:112-114`).
- **Grader:** `val_bpb` on a pinned validation shard (`prepare.py:43-44, 344-360`; `train.py:613-627`). Continuous; lower is better.
- **Reward density:** a progress signal every 5-minute run, logged to `results.tsv` with keep / discard / crash (`program.md:66-88`).
- **Infra:** one NVIDIA GPU, H100 tested (`AR/README.md`).
- **Anti-cheat:** prompt-only. The rule not to modify `prepare.py` is not enforced.
- **Port:** `AR/prepare.py` (fixed-budget evaluator with a pinned validation shard) and the `program.md` loop. GitSwarm's AutoResearch NanoChat task builds on this.

---

## 8. Requirements an environment framework must meet to express all of these

The family IDs in parentheses are the ones that need each primitive.

**State and apps**
1. **Files+git state** with per-instance prebuilt images and branch checkout (HA-1/2/4/5, SS-1, R2-*, AR-1, GS-1).
2. **Multi-service compose worlds** with service DNS, `depends_on: service_healthy`, shared volumes and per-service resource limits (H-E3, HA-11/12/13/14/15/16).
3. **Embedded DB state:** SQLite, DuckDB and dbt projects, spreadsheets (HA-8); tau2 domain DB (HA-11); FHIR server (HA-13).
4. **External live SaaS or host-shared services** with explicit reset hooks and non-parallelizable flags (crmarena, TheAgentCompany, financeagent).
5. **Full VM / desktop state** with KVM device passthrough, long boot healthchecks and VM pools with TTL claims (HA-17, H-E5).
6. **Browser state** (H-E5 calibration, webgen_bench verifier).
7. **Game or simulator process holding hidden state** (HA-14, HA-12).
8. **GPU/TPU requests with type constraints and long budgets**; shared GPU queues (H-E9, HA-10, featurebench, R2-10, AR-1, GS-1 SLURM).
9. **Nested Docker (DinD)** inside the task (abc-bench).
10. **Any MCP server as an App** over stdio, sse or streamable-http, including auto-wrapping a simulator's tool registry (H-E4, HA-11, HA-12).

**Actions and actors**

11. **Bash/tmux, file edit, MCP, HTTP, CLI-wrapped HTTP** (kumo `kumo_cli.py`, medagentbench CLI), **stdin game moves** (textarena), **keyboard/mouse** (H-E5, HA-17), **free-text function calls** (bfcl, ace-bench).
12. **Simulated grounded users:** LLM user plus persona, private goal, and user-side tools (HA-11, H-E6). ace-bench agent split is still unbuilt.
13. **Multi-agent:** N policy agents with a message bus and per-agent output volumes (HA-15); swarm of identical async workers with budget, abstain and nominate (GS-1).
14. **Simulated NPC coworkers** inside services (TheAgentCompany RocketChat).
15. **Humans in the loop** (human / confirm / yolo modes; MS-3).
16. **Oracle solution runner,** plus tolerance for oracles known to score below 1.0 (all Harbor; `agents/oracle.py`; programbench, gso, omnimath).

**Time**

17. **Per-phase wall-clock timeouts** with multipliers; step, cost and wall limits with remaining-budget prompts (MS-1/2); max_steps and max_errors (HA-11); action budgets (kumo).
18. **Simulated clock** that advances on tool calls and waits, with a scheduled-event queue and notifications (HA-12).
19. **Hours-long and "never stop" episodes:** 4-12 h tasks (rexbench, programbench, gso); indefinite loop (AR-1); 24-72 h swarms (GS-1).
20. **Multi-step episodes** with per-step setup, uploads, healthchecks and `min_reward` gating (H-E2).

**Observation**

21. **Text, images, audio and screenshots** (describe-image, labbench, gdb, mmau, HA-17, H-E5).

**Grading and reward**

22. **Hidden verifier assets** uploaded only at grade time; secrets only in the verifier image or sidecar (kumo `secret.json`, cybergym task-server, textarena `RESULT_TOKEN`).
23. **Shared (in-place) grading in the agent container** and **separate fresh-container grading** fed only by declared artifacts (with excludes), plus **collect hooks** that snapshot sidecar or runtime state (H-E3, HA-4, R2 owned recipes).
24. **Regrade from a recorded artifact manifest** (`trial/regrade.py`).
25. **Multi-key reward dicts** with a designated primary key: programbench `{reward, resolved, almost_resolved}`; cooperbench; locomo; research-code-bench `code_lines`; R2 `reward-details.json`; rewardkit dimensions.
26. **Continuous, unbounded rewards** (algotune speedup), **fractional** (R², nRMSE, HOTA, F1, pass-rate, f2p×p2p, checkpoint ratio) and **binary**.
27. **LLM/VLM/agent judges** that need verifier-phase network and API keys (HA-7, gaia2, webgen_bench, R2-1).
28. **Patch-apply + F2P/P2P reports, reverse grading** (swtbench), **merge of multiple agents' patches** (cooperbench), **trajectory- or event-log-based grading** (gaia2 write-action matching, medagentbench POST history, TheAgentCompany optional trajectory).
29. **Dense or milestone signals:** per-step rewards (H-E2), checkpoints (TheAgentCompany), per-run val metric (AR-1), and component breakdowns usable as shaping (R2-1).
30. **Fail-closed reward writing:** EXIT/ERR traps, defaulting to 0. Distinguish infrastructure error from reward 0 (cybergym health, hle judge failure, gdb).

**Isolation and anti-cheat**

31. **Per-phase network policy:** no-network, allowlist by hostnames, wildcards and CIDRs, or a denylist via `extra_hosts`. Agent and verifier policies can differ (HA-4, cybergym iptables, R2 `_env_guard`, network-policy-matrix).
32. **Git-history scrub and test-file removal** in the task branch (R2 `_env_guard`, SS `gather.py:540-551`); masked-test baseline diffing (featurebench); protected-file hash contracts (cli_gym).
33. **Reference-binary exclusion across the agent→verifier boundary** plus hash scrub (programbench).
4. **Resource rlimits** on graded submissions (frontiersmith); per-phase users such as learner, solver, root.
35. **Contamination controls:** held-out splits (labbench 20% held back), frozen package versions, red-team cheating-agent prompts and quality rubrics (`cheating_agent.md.j2`, `default-rubric.toml`, R2 quality loop).

**Generation and infrastructure**

36. **Mutation / bug-injection operators** and the "keep if it breaks ≥1 test" validation loop (SS-1, R2-4); environment-breaking operators (R2-7); task evolution strategies (R2-6 seta_evol).
37. **LLM-driven environment bootstrap** with a cost cap and content-addressed cache (R2 `bootstrap/`).
38. **Healthcheck / setup hooks** at environment level and per step; service reset endpoints (H-E3, TheAgentCompany `init.sh`, OSWorld).
39. **Artifact collection** from main and sidecar services, plus trajectory export (ATIF).
40. **Windows containers** (hello-world-bat, multi-step-bat), skills directories, configurable workdir, env-var injection that does not leak to sidecars (H-E10).
41. **Parallelism constraints:** some worlds must run serially because of shared mutable services (TheAgentCompany `-j 1`).

**Gaps relative to Crucible's model** (observed, not judged against Crucible's code):
- None of these codebases implements a generic **snapshot/restore with fingerprints** at world level. The closest are image caching and `regrade`.
- **Virtual clock** exists only in GAIA2/ARE.
- **Dense in-episode progress reward** exists only via multi-step rewards and checkpoints; there are no per-tool-call milestones.
- **Single tool-call gate** exists only implicitly, per protocol (MCP, CLI, bash).
