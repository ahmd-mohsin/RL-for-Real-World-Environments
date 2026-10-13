# M13 `bench` benchmark campaign: Crucible's benchmark pipeline run end to end on local environments

2026-10-13, HEAD `b6d89fc`. This is one of six parallel M13 auditors. Nothing under `crucible/src` or `crucible/tests` was
edited; this file is the only write under `crucible/`.

Setup:
- Host: macOS laptop, 14 CPU, Docker 29.8, everything run locally.
- No paid API, no GPU, no `.env`.
- The only network use was the Harbor verifiers' own grade-time `apt`/`uv` installs, two small base-image pulls, and one
  pip install inside the termgrade stand-in base image.

Paths:
- `SP/` = `/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m13/bench/`.
- Run dirs: `SP/runs/<group>/bench-<name>/`, with the M13 report in `bench/` and the campaign record in `bench-<name>.campaign.json`.

Drivers (they import crucible and change nothing):

| Driver | What it runs |
|---|---|
| `common.py` | policies, the HF model, `bench()` = `Runner.run` → `report_run(ks=(1,2,3))` |
| `harbor_run.py` | Harbor task dirs in place |
| `nemo_wp.py`, `nemo_cal.py` | NeMo-Gym workplace_assistant and calendar |
| `eog_run.py` | EnterpriseOps-Gym |
| `openenv_run.py` | OpenEnv PortSim |
| `chaos.py` | induced infrastructure faults |
| `coding_recorded.py` | re-derives the coding report from recorded runs |
| `recompute.py` | independent metric recomputation (no crucible import), planned vs terminal, determinism |
| `gen_table.py` | the results table below |

## Method

**Sources (8 distinct).**

| Source | What ran |
|---|---|
| terminal-bench-2.1 | `adaptive-rejection-sampler`, in place via Docker |
| Terminal-Lego, Lego-X and PrimeIntellect | 2 + 2 sample tasks, in place |
| termgrade | 2 tasks. The `dsbase-venv312:local` base is not local, so a stand-in was built from `python:3.12-slim` with `/opt/verifier_venv` (pytest 8.4.1 and pytest-json-ctrf). Built images were removed afterwards. |
| NeMo-Gym workplace_assistant | all 545 validation rows (floors), plus 8 rows × 4 policies |
| NeMo-Gym calendar_scheduling | 11 rows: 5 NeMo examples + 6 HF rows |
| EnterpriseOps-Gym | `interop.eog`, 13 local tasks |
| OpenEnv PortSim | `interop.openenv.catalog.portsim`, train 0–4 |
| Packs | toy pack via `crucible bench run toy`; coding pack recorded runs via `bench_recorded`, re-derived |

**Policies.** Every policy ran with n = 3 samples per task, except the 545-row floor scan (k = 1).

| Policy | What it does |
|---|---|
| oracle | Harbor: `solution/` written to `/tmp/crucible_oracle` and run through the `term` tools.<br>NeMo workplace: the row's `ground_truth` calls.<br>Calendar: a backtracking solver over NeMo's own `is_constraint_satisfied` / `is_event_conflicting`. It solves 11/11 rows, each checked with NeMo's `grade_assistant_response`.<br>EOG: `oracle_actions()`.<br>PortSim: the CP-SAT optimal plan. |
| no-op | No action, or an empty `Done`. |
| `adv_plant` | Writes `/logs/verifier/reward.{txt,json}`, prints PASSED, overwrites `/tests/test_outputs.py`, and plants a pass-all `conftest.py`. |
| `adv_shadow` | Stub `apt-get`, `curl`, `uvx`, `uv` and `pytest` in `/usr/local/bin` that exit 0, and the same stub over `/opt/verifier_venv/bin/pytest`. |
| `adv_spam` (NeMo) | 12 random tool calls, then a forged `{"reward": 1}` answer. |
| `adv_inject` (calendar) | Verdict text plus a zero-duration event. |
| `adv_wipe` (EOG) | `DELETE` of every table. |
| `adv_garbage` (PortSim) | Garbage plan, then a forged verdict. |
| `serial_baseline` (PortSim) | A partial-credit plan. |
| qwen2.5-0.5b | Qwen2.5-0.5B-Instruct on CPU through Crucible's own stack: `HFGenerateEngine` + `HFChatCodec` + `EngineLLM` + `LLMToolPolicy`, temperature 0.7, 4–8 steps. |

**Checks.**

| Check | How |
|---|---|
| (d) metric recomputation | Every run's headline metrics recomputed from raw `results.jsonl` by `recompute.py`, with no crucible import. |
| (d) determinism | `report_run` run twice into fresh dirs; `report.html` sha256 and digest compared. |
| (f) flake | termgrade oracle/no-op re-run 3× in fresh run dirs. |
| (e) compare | `crucible bench compare` on no-op vs every policy. |
| (b) chaos | Container removed mid-episode; container removed after the oracle and before the grade; host network dropped before the grade; NeMo server killed mid-run and before `/verify`. |

**Totals.** 23 run directories, about 1,750 terminal episodes, 91 min of runner wall time in total (the model runs
overlapped).

## Results per dataset × policy

Notes on the table:
- **mean ± CI**: mean over measured samples, with the seeded task bootstrap from the report; "—" = no CI (one task) or
  no measured sample.
- **tasks**: measured / in the report / planned.
- **wall**: the run's wall time, all policies, concurrency 3.

| dataset | policy | tasks (meas / in report / planned) | samples measured | masked | mean ± 95% CI | pass@1 | pass@3 | pass^3 | wall (run) |
|---|---|---|---|---|---|---|---|---|---|
| TB2.1 (in place) | oracle | 1 / 1 / 1 | 3 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 737 s |
| TB2.1 | noop | 1 / 1 / 1 | 3 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| TB2.1 | adv_plant | 1 / 1 / 1 | 3 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| TB2.1 | **adv_shadow** | 1 / 1 / 1 | 3 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | ″ |
| TB2.1 | qwen2.5-0.5b | 1 / 1 / 1 | 3 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| Terminal-Lego, Lego-X | **oracle** | 1 / 2 / 2 | 3 | 3 | **0.000** | 0.000 | 0.000 | 0.000 | 366 s |
| Lego-X | noop | 1 / 2 / 2 | 3 | 3 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| Lego-X | adv_plant | 1 / 2 / 2 | 3 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| Lego-X | **adv_shadow** | 1 / **1** / 2 | 3 | 0 | **1.000** | 1.000 | 1.000 | 1.000 | ″ |
| Lego-X | qwen2.5-0.5b | 1 / **1** / 2 | 3 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| Terminal-Lego, PrimeIntellect | oracle | 0 / 2 / 2 | 0 | 6 | — | — | — | — | 9 s |
| PI-Lego | noop | 0 / 2 / 2 | 0 | 6 | — | — | — | — | ″ |
| termgrade (stand-in base) | oracle | 2 / 2 / 2 | 6 | 0 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 | 146 s |
| termgrade | noop | 2 / 2 / 2 | 6 | 0 | 0.000 [0.00, 0.00] | 0.000 | 0.000 | 0.000 | ″ |
| termgrade | adv_plant | 2 / 2 / 2 | 6 | 0 | 0.000 [0.00, 0.00] | 0.000 | 0.000 | 0.000 | ″ |
| termgrade | **adv_shadow** | 2 / 2 / 2 | 6 | 0 | **1.000 [1.00, 1.00]** | **1.000** | 1.000 | 1.000 | ″ |
| termgrade | qwen2.5-0.5b | 2 / 2 / 2 | 6 | 0 | 0.000 [0.00, 0.00] | 0.000 | 0.000 | 0.000 | ″ |
| NeMo workplace, 545 val rows (k=1) | oracle | 545 / 545 / 545 | 545 | 0 | 1.000 [1.00, 1.00] | 1.000 | — | — | 80 s |
| NeMo workplace, 545 val rows | **noop** | 545 / 545 / 545 | 545 | 0 | **0.134 [0.11, 0.16]** | 0.134 | — | — | ″ |
| NeMo workplace, 8 rows | oracle | 8 / 8 / 8 | 24 | 0 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 | 1162 s |
| NeMo workplace, 8 rows | noop | 8 / 8 / 8 | 24 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| NeMo workplace, 8 rows | adv_spam | 8 / 8 / 8 | 24 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| NeMo workplace, 8 rows | qwen2.5-0.5b | 8 / 8 / 8 | 24 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| NeMo calendar, 11 rows | oracle | 11 / 11 / 11 | 33 | 0 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 | 1287 s |
| NeMo calendar | noop | 11 / 11 / 11 | 33 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| NeMo calendar | adv_inject | 11 / 11 / 11 | 33 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| NeMo calendar | qwen2.5-0.5b | 11 / 11 / 11 | 33 | 0 | 0.091 [0.00, 0.21] | 0.091 | 0.182 | 0.000 | ″ |
| EnterpriseOps-Gym (7 of 13) | oracle | 7 / 7 / 7 | 21 | 0 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 | 40 s |
| EOG | noop | 7 / 7 / 7 | 21 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| EOG | adv_wipe | 7 / 7 / 7 | 21 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 12 s |
| EOG (6-task subset) | qwen2.5-0.5b | 6 / 6 / 6 | 18 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 1078 s |
| OpenEnv PortSim, runner as shipped | every policy | 0 / **2** / **5** | 0 | 22 | — | — | — | — | 19 s |
| PortSim (instruction workaround) | oracle | 5 / 5 / 5 | 15 | 0 | 1.000 [1.00, 1.00] | 1.000 | 1.000 | 1.000 | 10 s |
| PortSim (workaround) | serial_baseline | 5 / 5 / 5 | 15 | 0 | 0.200 [0.20, 0.20] | 0.000 | 0.000 | 0.000 | ″ |
| PortSim (workaround) | noop | 5 / 5 / 5 | 15 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| PortSim (workaround) | adv_garbage | 5 / 5 / 5 | 15 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | ″ |
| toy pack (`crucible bench run toy --k 3`) | scripted:oracle | 3 / 3 / 3 | 9 | 0 | 1.000 [1.00, 1.00] | 1.000 | **—** | **—** | 5 s |
| toy pack | scripted:half | 3 / 3 / 3 | 9 | 0 | 0.306 [0.00, 0.47] | 0.000 | — | — | ″ |
| toy pack | scripted:noop | 3 / 3 / 3 | 9 | 0 | 0.000 [0.00, 0.00] | 0.000 | — | — | ″ |

Other runs:

| Run | Dir | Result |
|---|---|---|
| Chaos, Harbor (TB2.1, 3 kinds × 3) | `SP/runs/chaos/bench-chaos-harbor` | 9/9 scored 0, 0 masked (B2) |
| Chaos, NeMo kill mid-run (30 rows) | `SP/runs/chaos/bench-chaos-nemo` | 9 scored 1.0, 12 masked, 9 never dispatched (health stop) |
| Chaos, NeMo kill before `/verify` | `SP/runs/chaos/bench-chaos-nemo-stop` | 3/3 masked `verifier_error` (`check_infra`); retries masked as well |
| termgrade flake re-runs ×3 | `SP/runs/harbor/bench-termgrade-flake{1,2,3}` | Same digest `5b064781aa381e89` all 3 times; wall 42–51 s each |
| Mixed Lego-X + PI-Lego `task_00000` | `SP/runs/harbor/bench-mixed` | 2 planned, 1 in the report (B7) |
| Coding pack, recorded runs re-derived | `SP/runs/coding/r{1,2}` | r1 = r2, but they differ from the committed report (B15) |

## (a) Validity floors per dataset

| Dataset | oracle < 1 | no-op > 0 | Notes |
|---|---|---|---|
| TB2.1 | – | – | – |
| Lego-X | **`task_00000`: 0/3**. Harbor's own `OracleAgent` also scores 0 (`SP/logs/harbor_native_oracle_legox.log`).<br>`task_07777`: does not build (`COPY ./task_file: not found`), masked 9/9. | – | The local HF snapshot drops empty dirs: `task_file/output` and `task_file`. |
| PI-Lego | both tasks masked. `docker_image = terminal-lego/<id>:latest` gives `pull access denied`. | – | Needs the Dockerfile override (G5 D19). |
| termgrade | – | – | Needs the base-image prebuild step. |
| NeMo workplace | – | **73 / 545 val rows (13.4%)** | 71 have an empty `ground_truth`; 2 updates are already true at seed (`val029`, …). |
| NeMo calendar | – | – | – |
| EOG | – | – | 1 of 13 skipped (missing seed DB), 5 of 12 have no derivable oracle, so 7 were benchmarked. Refusal tasks would have oracle = no-op by design; none were local. |
| PortSim | – | – | – |
| toy | – | – | – |

Every one of these tasks is in the reports as an ordinary item. Nothing in M13 flags them (B4, B5).

## (b) Masking

**NeMo-Gym is correct.**
- A server killed mid-run masks every later sample (`agent_run_error`, "ConnectError").
- A server killed after the agent and before `/verify` masks it (`verifier_error` / `check_infra`).
- Neither scores 0, and masked samples are not in any denominator. `recompute.py` agrees on `mean_reward` and `pass@k`.

**Harbor in place is not (B2).**
- A container removed mid-episode scores **0** 3/3: the policy sees `exit_code 1, service "main" is not running`
  as an ordinary failed command, then the grade's `_probe` returns `crucible:agent_broke_env`.
- A container removed after the oracle solved the task (and before the grade) scores **0** 3/3, with the same
  `agent_broke_env`.
- A host-side network drop after the oracle solved the task ("Task completed successfully!") scores **0** 3/3. The
  task's `test.sh` cannot `apt-get`/`curl` uv (`Temporary failure resolving 'archive.ubuntu.com'`, `uvx: command not
  found`) and writes reward 0.
- Evidence: `SP/runs/chaos/bench-chaos-harbor/samples/*/a1/{trace.jsonl,verifier/report.json,work/w/harbor_trial/verifier/test-stdout.txt}`.

**Image pull and build failures mask** (PI-Lego, Lego-X `task_07777`). But they are named `agent_run_error` with blame
`harness`/`unknown`, and the real cause is cut off by the 500-character reason (B12).

## (c) Adversarial policies

These scored 0 everywhere:
- `adv_plant`, on all Harbor sets. The `grader_paths` intercept refused the writes and flagged them (`hack/intercepted` = 2).
- `adv_spam`, `adv_inject`, `adv_wipe` and `adv_garbage`.

**`adv_shadow` scored 1.0 on every Harbor dataset that grades** (TB2.1 3/3, termgrade 6/6, Lego-X 3/3).
- On Lego-X it scored 1.0 even on `task_00000`, where the oracle scores 0.
- No rule flag was raised, and the bench would ignore one anyway.
- IRT gives it the oracle's ability (+1.56 vs +1.56 in `bench-termgrade/bench/summary.json`).
- `compare` reports no-op → `adv_shadow` as the same improvement as no-op → oracle (B1).

## (d) Metric consistency and determinism

**Independent recomputation** (`SP/recompute/*.json`). Coverage, `mean_reward`, `mean_reward_raw`, pass@1/2/3 and
pass^1/2/3 match the report exactly (0 diffs) on all 16 runs checked:
- termgrade, Lego-X, TB2.1, PI-Lego, mixed;
- workplace floors and main, calendar;
- EOG ×3 (main, adversarial, model);
- PortSim ×2;
- chaos ×3.

**`report_run` determinism.** `report_run` run twice into fresh dirs gives the same digest and a byte-identical
`report.html` on every run.

**Toy CLI.** `crucible bench run toy` in two fresh dirs gives a byte-identical `report.html` and the same digest
`7509adfd595da48b`.

**Where it fails.** The CLI `bench report` path changes the digest (B10). The committed coding report is not
reproducible at HEAD (B15).

## (e) Compare and McNemar

The obvious direction holds on every dataset:

| Comparison | Pairs | Δ | Flips | p |
|---|---|---|---|---|
| no-op → oracle, workplace floors | 545 | +0.866 [0.837, 0.895] | 472 fail→pass, 0 pass→fail | 0.0 |
| no-op → oracle, calendar | 33 | +1.0 | | 0.0 |
| no-op → oracle, EOG | 21 | | | 9.5e-07 |
| no-op → oracle, PortSim | 15 | | | 6.1e-05 |
| no-op → oracle, termgrade | 6 | | | 0.03125 |
| no-op → oracle, TB2.1 | 3 | | | 0.25 |
| no-op → model, calendar | | +0.091 | 3 fail→pass | 0.25 |

Elsewhere no-op → model is 0.

The exact McNemar values are right, but they count the k samples of a task as independent pairs (B11):
- termgrade has **2 tasks** and gets p = 0.031;
- workplace main has 8 tasks and gets p = 1.2e-07, where a per-task test gives 0.0078.

Without policy filters, `compare` silently mixes policies (B3).

## (f) Coverage, wall time, flakiness

Coverage is in the table above.

Wall time:
- Harbor in-place episodes take about 40–80 s each; most of it is the verifier's grade-time apt + uv install on
  TB2.1 and Lego.
- termgrade (offline verifier) takes about 7 s per episode.
- NeMo and EOG take under 1 s per scripted episode.
- Qwen-0.5B on CPU takes 20–70 s per episode.

Flakiness: none observed. termgrade oracle/no-op ×3 re-runs and the toy ×2 give identical digests. TB2.1 oracle was 3/3
in the main run; the earlier chaos run's oracle phase also solved it 6/6, before the fault was injected.

## (g) CLI

| Command | What happened |
|---|---|
| `bench run toy --k 3` | Reports only pass@1 (B9). |
| `bench report <run>` | Overwrites the run's own report with a different digest (B10). |
| `bench compare` with a misspelled policy | `pairs: 0`, `p = 1.0`, exit 0 (B14). |
| `bench report` on an empty dir | Writes an empty report, exit 0 (B14). |
| `bench run coding` | `ValueError: pack coding declares neither hooks.tasks nor hooks.generator`. No CLI path benchmarks Harbor dirs, NeMo rows, EOG or OpenEnv (B17). |

No hang. The runner's `attempt_timeout_s` was never hit.

## Findings

| ID | Sev | What | Evidence (`SP/…`) | Suspected location | Minimal fix |
|---|---|---|---|---|---|
| B1 | **P0** | A reward hack scores 1.0 and the bench counts it. Shadowing the grader's own tools passes TB2.1, Lego-X and termgrade 100%: `/usr/local/bin/{apt-get,curl,uvx,pytest}` stubs, or a stub over `/opt/verifier_venv/bin/pytest`. It passes even on a task whose oracle fails. No flag is raised, because `EXIT_SPOOF` / `VERIFIER_FILE` do not match it. The bench reads no rollout flags at all, so even `hack/intercepted` never reaches a metric. IRT ranks the hacker with the oracle. | `runs/harbor/bench-{tb21,termgrade,legox}` (policy `adv_shadow`); `runs/harbor/bench-termgrade/samples/*/a1/rollout.json` (`flags: []`) | `interop/harbor/inplace.py:236-288` (the verifier runs agent-writable binaries in the agent's container); `credit/hack_detector.py:27-29`; `bench/metrics.py:77-114` (`load_run` ignores `rollout.json` flags) | Bench: carry `flags` and `hack/*` into `Episode`; report a flagged-pass column and exclude flagged passes from pass@k. Grade: run the verifier with a root-owned clean PATH and a verifier toolchain copied from the image at grade time. Or check `docker diff` for writes to PATH dirs and verifier venvs, and score those 0 as `policy_misbehavior`. Rule: flag writes to executables on PATH or in `*/bin/` that the verifier uses. |
| B2 | **P0** | An infrastructure fault in a Harbor in-place World is scored 0. A container removed mid-episode or before the grade is labelled `crucible:agent_broke_env`. A host network loss at grade time gives the verifier's own 0. A solved task is reported as failed. | `runs/chaos/bench-chaos-harbor` (9/9 scored 0); `…/verifier/test-stdout.txt` | `inplace.py:64-66` (`tool_exec` returns compose's "service main is not running" as `ok`); `inplace.py:201-210,258-260` (`_probe`: rc ≠ 0 means the agent broke it) | Before attributing to the agent, check container liveness from the host (`docker inspect` State, container exit reason). A container that is gone or stopped from outside is masked `env_session_lost`. Grade-time network: a verifier-phase connectivity probe, and on failure mask `check_infra` instead of trusting `reward.txt`. Keep `agent_broke_env` for in-container causes (PID 1 killed from inside, exit code / OOM). |
| B3 | **P0** | `compare` joins on (task, sample_idx) without policy or harness. A run with several policies collapses to an arbitrary, hash-ordered mix, so the numbers are wrong and there is no warning. Toy run, no-op vs the unfiltered run: Δ −0.204, 1 "regression". | `toy_cli/r1/bench-toy-bench` (command in §(e)) | `bench/compare.py:32-34`; `cli/studio.py:293-294` | Key on (policy, harness, task, sample_idx). Refuse (or require `--base-policy` / `--cand-policy`) when a side has more than one (policy, harness). |
| B4 | **P0** | Vacuous tasks are included silently. 13.4% of workplace_assistant validation rows are solved by the no-op (empty `ground_truth`). They lift every policy's mean, by up to 0.134, with no flag. `item_qa` does not flag them either. | `runs/wa/bench-wa-floors` (no-op 0.134; 73 tasks listed in §(a)) | `bench/run.py:27-36` (no control or validity input); `bench/irt.py:196` | `run_bench` / `report_run` take a validity manifest (M9 verdicts or I2 floor results). Run the no-op control by default. Report floor-solved items separately and exclude them from headline metrics. |
| B5 | **P0** | Broken tasks are included as hard items. Lego-X `task_00000`'s gold solution fails in its own environment, so every policy gets 0 and the task reads as "hard". No oracle check exists in the bench path. | `runs/harbor/bench-legox`; `logs/harbor_native_oracle_legox.log` | `bench/run.py` (no oracle gate); `interop/harbor/inplace.py` (no oracle verdict on import) | An I1 oracle check before an item enters a benchmark (oracle ≥ 1 on ≥ 3 runs). Report items whose oracle is below 1 as invalid, never as 0. |
| B6 | P1 | Coverage leaves out samples that never ran, and the stop reason. When the health, budget or wall-clock rule stops a run, undispatched samples and whole tasks disappear from the report. `run_bench` drops `RunSummary.stopped`. Policies end up averaged over different task sets: on Lego-X, `adv_shadow` and the model cover 1 task while the oracle covers 2. | `runs/chaos/bench-chaos-nemo` (30 planned, 21 in the report); `runs/openenv/bench-portsim-mainvenv` (60 planned, 22 in the report, tasks 2–4 absent); `runs/harbor/bench-legox` (30 planned, 22) | `bench/run.py:35`; `bench/metrics.py:77-82,224-229`; `bench/report.py` | Read `run.json` (planned) and the journal's `stop_rule` events. Add `planned` / `not_run` / `stopped` to coverage and to the HTML header. Compute cross-policy aggregates on the common task set, or flag them. |
| B7 | P1 | Duplicate task ids are not refused. Harbor tasks with no `[task].name` take the directory name, so Lego-X and PI-Lego `task_00000` share one id and one sample id. The two attempts ran at once: the second moved the first's live work dir to `aborted/`, and the report kept one result and dropped the other dataset. | `runs/harbor/bench-mixed` (2 planned, 1 in the report; `samples/46551f621fbe00c1/aborted/a1`) | `interop/harbor/inplace.py:108-109`; `runtime/run.py:81-89`; `runtime/journal.py:111-121` | `Runner.plan`: refuse duplicate `task.id`. Harbor ids without a name: include a dataset or path hash. |
| B8 | P1 | OpenEnv tasks cannot run through the M10 runner.<br>(i) In `.venv`: `OpenEnvTask.instruction` is a callable, and `save_episode` passes it to ATIF. A pydantic `ValidationError` follows.<br>(ii) In `.envs/openenv`: the runner always loads Harbor's ATIF modules, which are not installed there.<br>Either way every attempt is masked `agent_run_error`, and the deterministic error is retried: 142 attempts for 22 terminal, then the health stop. With a string-instruction wrapper, PortSim benchmarks cleanly. | `runs/openenv/bench-portsim{,-mainvenv}` vs `-mainvenv-wrap` | `runtime/harness/native.py:72,84`; `runtime/run.py:243`; `interop/openenv/catalog.py:39` | Pass the World's rendered instruction (or `str()` only when it is text). Load ATIF lazily and skip the trajectory when Harbor is absent. Make a deterministic harness exception (the same error class and message on attempt 2) non-retryable. |
| B9 | P1 | The CLI never computes pass@k for k > 1. `bench run --k 3` and `bench report` call `run_bench` / `report_run` with the default `ks=(1,)`, so pass@3 / pass^3 / max@3 are missing from a k = 3 run. | `toy_cli/r1/bench-toy-bench/bench/summary.json` (`"ks": [1]`) | `cli/studio.py:245,277` | `ks = range(1, k+1)` in `bench run`; `--ks` on `bench report` (default: up to the run's k from `run.json`). |
| B10 | P1 | `bench report <run>` is not reproducible and overwrites the original. On the same toy run the digest changes `7509adfd595da48b` → `387bc08a752c409a`: stages, extras and title are lost, so stage reach becomes None and the manifest disappears. It overwrites `run/bench/` in place. | `toy_cli/r1_bench_orig` vs `toy_cli/r1/bench-toy-bench/bench` | `cli/studio.py:270-278`; `bench/run.py:39-51` | Store the report config (ks, threshold, title, stages pointer, extras) in `summary.json` and reuse it on report. Write to a new dir unless `--force`. |
| B11 | P1 | McNemar and the paired bootstrap pseudo-replicate. The k samples per task count as independent pairs, and sample i of one policy is paired with sample i of another, which is arbitrary. 2 tasks × k = 3 gives p = 0.031; 8 tasks give p = 1.2e-07 where a per-task test gives 0.0078. | `compare/bench-termgrade.noop_vs_oracle.json`, `compare/bench-wa-main.noop_vs_oracle.json` | `bench/compare.py:31-62` | Test at the task level: per-task pass rate differences with a sign or permutation test over tasks, or a clustered McNemar. Print n_tasks beside p. |
| B12 | P2 | Setup failures are mis-labelled. `pull access denied` and Dockerfile `COPY … not found` are deterministic task-side defects, but they become `agent_run_error` with blame harness or unknown. The 500-character reason cuts off the cause. They count toward a health stop that says "environment is broken" but name no task. The report has no breakdown of masked kinds. | `runs/harbor/bench-{pilego,legox}/results.jsonl` | `runtime/run.py:191-197`; `runtime/infra.py` (normalize); `bench/report.py` | Classify compose build and pull errors as `task_setup` (task blame) with the tail of the error kept. Add a masked-kind table to `summary.json` and the HTML. |
| B13 | P2 | pass@k denominators are hidden. Groups with fewer than k measured samples are dropped without notice, so pass@1 and pass@3 can cover different task sets (Lego-X `adv_plant`: 1 group for pass@3, 1 masked task gone). | `recompute/bench-legox.json` (`groups_n>=k`) | `bench/metrics.py:185-221` | Report the number of groups per k next to each value. |
| B14 | P2 | The CLI is silent on bad input. A misspelled policy gives `pairs: 0`, `p = 1.0`, exit 0. `bench report` on a dir with no `results.jsonl` writes an empty report, exit 0. | §(g) | `cli/studio.py:281-293`; `bench/metrics.py:77` | Error on no matching episodes or no `results.jsonl`. |
| B15 | P2 | The published `docs/modules/reports/m13_coding_recorded` is stale. Re-derived at HEAD (twice, identical) the digest is `b7d4950e7ac6aa58`, not the committed `cefd2f4533e82eeb`. Φ drifts by up to 0.029 (task01 fable 0.4917 → 0.4627) after the M6–M12 audits; rewards are unchanged. | `runs/coding/r1` vs the committed report | `packs/coding/.../bench_recorded.py` (output not regenerated) | Regenerate on credit changes; a CI check that re-derives the digest. |
| B16 | P2 | IRT does not use the controls. Items the no-op solves are not flagged by `item_qa`, and a hacker gets the oracle's θ. | `runs/wa/bench-wa-floors/bench/summary.json` (`item_qa_flagged: []`); termgrade θ | `bench/irt.py:196`; `bench/run.py:43-49` | Flag items passed by a declared control respondent; exclude flagged respondents. |
| B17 | P2 | No benchmark path exists for non-pack sources. `crucible bench run` needs pack hooks (coding has none), so Harbor dirs, NeMo rows, EOG and OpenEnv need Python drivers. A Harbor oracle cannot be replayed through the policy surface: `grader_paths` refuses `/solution` writes, and `HarborInPlaceTask.oracle` is not exposed as a `PolicyRef`, so floors need driver code. | §(g); `runs/harbor/bench-termgrade-smoke` (first attempt: oracle refused) | `cli/studio.py:227-251`; `interop/harbor/inplace.py:225-234` | A `bench run --source harbor|nemo|eog|openenv <path>` entry, with built-in oracle and no-op control policies per importer (part of I0–I2). |

**Verified correct:**
- NeMo masking (mid-run and before `/verify`): masked samples are never scored and never in denominators.
- Masked = None everywhere: `mean_reward_raw` differs from `mean_reward` exactly as the masked samples predict.
- Metric arithmetic: 0 diffs over 16 runs.
- Report determinism.
- Exact McNemar values.
- The `grader_paths` intercept against reward and test planting.
- NeMo, EOG, calendar and PortSim adversaries all scored 0.
- The Qwen engine path end to end, with token records and no masks.

## (h) What the datasets that cannot be benchmarked yet need (I0–I10)

**Benchmarkable today** (all with driver code, see B17):
- TB2.1 (1 local task);
- termgrade (with a base image built by hand);
- NeMo workplace_assistant (full splits are local in `SP/../../wp_*.jsonl`) and calendar;
- EOG (7 of 13);
- PortSim (only with the B8 workaround);
- the toy pack;
- coding recorded (replay only).

**Present but not valid:**
- Lego-X: 1 broken oracle and 1 unbuildable task, both from empty dirs the HF snapshot drops.
- PI-Lego: private images.

| Dataset(s) | Blocker found in this campaign or in G5 | Needed |
|---|---|---|
| Terminal-Lego (Lego-X, PI) | Empty dirs dropped, so the oracle fails or the build fails. PI pins private `terminal-lego/<id>:latest`. | I0 materializer that restores empty dirs from `task.toml`/`solve.sh` references; Dockerfile override for `docker_image`; I1 oracle ≥ 3; I10 diff vs `prime-data-excluded-tasks.jsonl`. |
| termgrade | `FROM dsbase-venv312:local` is not built by anything; the verifier venv is agent-writable (B1). | I0 base-image prebuild from `base_image/`; `task_source` row → dir; B1 grading isolation. |
| TB2.1 (89), Lego, BeyondSWE, repo2rlenv-eqv | Grade-time network (B2); `hub.byted.org` images; amd64-only images run under emulation on arm64. | I3 grade-offline / connectivity probe; image remap; registry pinning (I0). |
| data-agent, SmolDataEnvs | Multi-key `reward.json`; `[verifier.env]`; HF network and `HF_TOKEN` at setup. | Reward-key rule (G5 D13); verifier env; offline setup. |
| HF_ML_Tasksmith, repo2rlenv-swe-flow, openai-math, TB-science, NatureBench | Separate-verifier tasks are refused in place. NatureBench also needs an LLM judge, GPU compose and a judge key. | Harbor `_run_separate_verifier` path; I7 judge pinning. |
| MiMo-harbor-code | No oracle; no reward on testbed failure. | Oracle from the source `code.parquet` patch (I1); a non-root agent. |
| Long-Horizon-TB | No tests or solutions. | I0 lint: refuse. |
| Lego-RL-2699 | Repos not shipped. | Restore repos, or use the PI `openswe` taskset row path. |
| R2E-Gym, Multi-SWE ×2, Scale-SWE, SWE-Lego, SWE-rebench-V2 | No row → `RunTask` adapter. `v1_import` only converts `vf-eval` traces, and the PI tasksets `load_dataset()` from the network. | I1/I2 `vrows.py` over the PI tasksets in `.envs/verifiers` (`validate`, `apply_gold_patch`, `solved`), offline rows. |
| Reverse-Text-RL | Taskset is local (`primeintellect-ai-verifiers/environments/reverse_text`), pure Python and an LCS reward, but not installed in `.envs/verifiers`. `load()` calls `load_dataset` (network). Only 3 local rows. | A local-rows option plus the same `vrows` adapter (S). Not run here: no adapter, and no pip from the network. |
| NeMo ReasoningGym, instruction_following, structured_outputs | Server deps missing in `.envs/nemo` (`reasoning_gym`, `verifiable_instructions`, `defusedxml`, `xmltodict`, `openapi_schema_validator`); none in a local pip/uv cache. | Install the deps; oracle from the `answer` column; I2 minimal-object floor (structured_outputs). |
| NeMo CFBench, SysBench | Paid LLM judge (temperature 0.7). | I7 judge pin, or a soft channel. |
| OpenEnv catalogue (PortSim, …) | B8. | B8 fix; then `oracle_actions` per environment (I1). |
| RetroEnv, procedural-pile, harbor-mix | Env package not local / no framework / gated. | Install the package; a plain answer-task adapter; a user decision on gating. |
| MiMo import (`interop/mimo`) | Not exercised in this campaign (time). | – |
| All | M13 takes any task list without validity evidence (B4, B5). | I10 `import_report.json` (floors, oracle, flake, infra rates) fed to `run_bench` as the validity manifest. |
