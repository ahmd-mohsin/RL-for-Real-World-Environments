# M15 sweep G2: how frameworks shape a user-facing CLI, pipelines, run directories, viewers and exports, and how Crucible's studio compares

**Scope.** This sweep covers the user path of M15 at HEAD 076ab65: the `crucible` CLI, the `crucible build` pipeline, the agentenv explorer plugin and `crucible export`.

**Method.** Every README command was run through the real CLI (`.venv/bin/crucible`) in a scratch directory, timed, run twice, re-run with changed flags, interrupted and resumed. The explorer plugin was mounted in agentenv's real `create_app()` through `TestClient` and probed with adversarial inputs. The frameworks below were read for comparison.

**Labels.**
- `[ran]` means a probe in this sweep produced the evidence.
- `[read]` means the evidence comes from reading code.

**Constraints.**
- No GPU and no paid API were used.
- Nothing under `crucible/src`, `crucible/packs` or `crucible/tests` was edited.
- No `.env` file was read; the dotenv files in the probes are synthetic.

**Probe location.** `/private/tmp/claude-501/-Users-muahmed-Desktop-Projects-RL-for-Real-World-Environments/54b8300a-e533-4d3f-9f31-c0f8369fa981/scratchpad/m15/g2/` (`SP/`). Section 6 indexes it.

**Path keys.**
- Crucible: `C/` = `crucible/`; `CLI` = `C/src/crucible/cli/studio.py`; `EXP` = `C/src/crucible/studio/explorer.py`; `EX` = `C/src/crucible/studio/export.py`; `B` = `C/src/crucible/studio/build.py`; `PK` = `C/src/crucible/studio/pack.py`.
- Sources under `RL-Environments/06-frameworks-and-env-code/`: `H` = harbor-framework-harbor; `V` = primeintellect-ai-verifiers; `NG` = nvidia-nemo-gym; `RL` = infra-rllm-org-rllm; `SK` = infra-novasky-ai-skyrl; `OE` = meta-pytorch-openenv; `FE` = adithya-s-k-fineenvs; `R2E` = huggingface-repo2rlenv; `SS` = swe-bench-swe-smith.
- `AE` = `agentenv-framework/`.

## 0. Headline defects

| # | Sev | Defect | Where | Evidence |
|---|---|---|---|---|
| D1 | **P0** | `crucible export --to nemo-gym\|openenv` copies the pack's `data/` and `src/` whole. That includes the dotenv the pack declares as its `redaction_env` and the raw workflow logs that `label` redacts. The result is a bundle meant "to move to another machine" (EX:10-11). | EX:94-106 (`_bundle`: `copytree(src, data, worlds, families)`, no ignore list beyond `__pycache__`) | [ran] `SP/repro_p0.sh`. A scaffolded pack has `ingest: … redaction_env: data/.env` and `data/.env` holds `FAKE_API_KEY=sk-fake-probe-0000`. `pack check` reports ok. `label` stores 0 hits (redacted). After `export --to nemo-gym`, both `exp/pack/data/.env` and `exp/pack/data/ticket_logs/tickets.jsonl` contain the value, and `src/.env.local` is copied too. |
| D2 | **P0** | `GET /runs/{id}/report` reads `<run>/bench/report.html` with no `resolve()`/`is_relative_to` check, so a symlink inside a run directory serves any file outside the data root. EXP:16 claims "the API cannot read outside it". | EXP:120-128 | [ran] `SP/probe_link.py`. `data/shared/evil-run/bench/report.html` is a symlink to `SP/outside.txt`. The response is 200 with `OUTSIDE-ROOT-CANARY-123` as `text/html`. |
| D3 | **P0** | Stored XSS. The report route serves any `report.html` bytes in the data root (an imported or shared run directory) on the explorer's own origin. It goes through the app-wide CSP `script-src 'self' 'unsafe-inline'`, not through the `sandbox` CSP that agentenv gives agent-produced HTML (`AE …/routers/objects.py:20-25,97-98`). Same-origin script passes the CSRF guard, which refuses only `Sec-Fetch-Site: cross-site` (`AE …/explorer/app.py:148-156`). It can therefore POST to `/api/v1/tasks/{id}/run` (`AE …/routers/runs.py:144,185`). The report Crucible itself generates escapes every value correctly. | EXP:120-128 | [ran] `SP/probe_csp.py`: a planted `<script>fetch("/api/v1/runs",{method:"POST"})</script>` comes back 200 `text/html` with CSP `script-src 'self' 'unsafe-inline'`. [ran] `SP/probe_xss.py`: task and policy ids `"><script>…<img onerror>` render escaped (9× `&lt;script&gt;`, 0 `<script`). |
| D4 | P1 | `GET /packs/{name}/check` runs the pack's code in the explorer process. The static level imports the package and calls `register()` and every hook. A bare name that is not a known pack resolves against the server's working directory. `SAFE` also admits `.` and `..`. | EXP:92-101 → PK:194-196 (CWD fallback), PK:640-668 (`resolve` + `reg(r)`); `pack check --all` already isolates in a subprocess (CLI:59-60) | [ran] `SP/probe_exec.py`: `./evilpk` in the CWD with a marker in `__init__.py`. The GET returns 200 and the marker reads "pack code ran in pid 799", the explorer's own pid. |
| D5 | P1 | Explorer runs are keyed by directory basename. `/runs` lists every directory holding `results.jsonl` + `samples/`, including M9's internal accept sub-runs. `/runs/{id}` silently serves the first hit, so the other runs with that name cannot be addressed. | EXP:42-54 | [ran] `SP/probe_app.py` over the README output: `bench-toy-build` appears 5 times (build, build2, …) and `control_noop`, `blank_usage` and others appear 30 times each (`build/accept/<hash>/a11/…`). `/runs/bench-toy-build/summary` always returns `build/…`. |
| D6 | P1 | A refused re-run leaves the run directory inconsistent. `configs/resolved/build.json` is rewritten with the new flags, then forge, accept and export run again, and only then does `run_bench` refuse the identity mismatch. The same ordering exists in `label` (written before ingest). | B:47 (before B:120); `C/src/crucible/studio/label.py:118` | [ran] `build toy --out build_k --k 3` on a k=1 directory exits 1 with "different identity … refusing to resume". Afterwards `build.json` says `"k": 3`, `run.json` says `k=1`, and the message names no way out (a new `--out`). Harbor (`H/src/harbor/job.py:267-278`), verifiers (`V/verifiers/v1/cli/eval/main.py:87-113`) and R2E (`R2E/…/repository/runner.py:65-91`) compare before writing anything. |
| D7 | P1 | Several steps report success on nothing (exit 0). | EX:129-149; `label.py:146-153`; EX:116-118; CLI:115,401 (no `IntRange`) | [ran] All four cases exit 0: (a) `export --to atif --run label` and `--run build` (a build root) give `trajectories: 0`; (b) `label --run label` gives `labelled 0 episodes` plus empty `sft/dpo/kto` and "next: train on x3/sft.jsonl"; (c) `export --to canonical --pack ./demo` (the README's own scaffold) gives `tasks: {}`; (d) `build toy --k 0` measures 0 of 21 planned samples with models `None`. `bench report label` refuses the same non-run input (exit 1), so the commands disagree. |
| D8 | P1 | `build`'s exit code and human output hide failure. Rejected tasks, export errors, owed or invalid samples and bench warnings are kept only in `summary.json`. `accept` exits 1 on a rejection; `build` never does. | CLI:120-135 (no `SystemExit`, no warnings line); B:76-78 (`export_errors` collected silently) | [ran] The k=0 build printed `scripted:oracle None` and exit 0. The "21 of 21 planned samples have no result" warning appears only in `summary.json`. `bench run --k 0` prints it. |
| D9 | P1 | An invalid `$CRUCIBLE_BENCH_KEY` is ignored silently. `check_key`'s `ValueError` (short, repeated or public key) is swallowed, a random key is written to `bench.key`, and the run exits 0. The user's held-out items are then not keyed by the user's key and cannot be regenerated elsewhere. | `C/src/crucible/bench/run.py:163-166` | [ran] `CRUCIBLE_BENCH_KEY=deadbeef bench run ./demo` exits 0 and creates `bench.key`. A valid 32-byte key is used and no file is written (correct). |
| D10 | P1 | A next-step hint can point at another pack. `pack new` prints `crucible pack check {dest}`, where `str(Path("./toy"))` = `toy`, and `find_pack` resolves names before paths. | PK:351, PK:178-198 | [ran] `pack new bar --dir ./toy` prints "next: crucible pack check toy". Running that checks `packs/_toy` and prints `toy (static): ok`; the user's pack is never checked. |
| D11 | P1 | ATIF export loses attribution. All 21 trajectories carry `agent: {name: crucible-policy, version: 0.1.0}` with no task id, policy id, reward or run id. File names are sample hashes. Oracle, half and no-op episodes cannot be told apart. | EX:129-149 (copies only `trajectory.*.json`); `C/src/crucible/interop/harbor/atif.py:26,65` | [ran] `exp_atif/*.json`: `extra` = {actor, crucible_trace, harness, world}; `final_metrics` = {total_steps}; 0 files mention a reward. A Harbor trial directory keeps `result.json` and `config.json` beside the trajectory (`H/src/harbor/models/trial/paths.py:190-298`). |
| D12 | P1 | Contract break and provenance. M15-studio.md:13 and `studio/__init__.py:10` say "every command writes … `configs/resolved/<cmd>.json`", but only `label` and `build` do. `bench run`, `export`, `forge`, `accept`, `model` and `ingest` write none. Nothing records the crucible version, git sha or dirty flag, the Python version, the pack version or content hash, or the agentenv version (`run.json` = {run_id, identity, spec, created}). | CLI:397-423, 497-508; `C/src/crucible/runtime/run.py:101` | [ran] `ls` and a grep over `runs/bench-demo-bench/{run.json,bench/summary.json}`: the only seed is `run_seed 0`, and there is no version or sha. Harbor `JobLock` records `harbor.version` and `git_commit_hash` (`H/src/harbor/models/job/lock.py:60-63`). Every verifiers trace records `VersionInfo(version, commit)` (`V/verifiers/v1/trace.py:91-101`). |
| D13 | P1 | Vacuous test. `test_explorer_plugin.py:41` asserts that `..%2Fetc` gets 400/404, but Starlette decodes `%2F` and returns its own route 404 before `_run_dir` runs. There is no symlink, HTML-serving, duplicate-id or pack-check-execution test. | `C/tests/unit/test_explorer_plugin.py:41` | [ran] The 404 body is `{"detail":"Not Found"}` (router), not the plugin's `no run …`. |
| D14 | P1 (cross-group, label) | `label --select correct` (the default) puts expert-labelled **bad** traces into SFT. `curate` selects by gated reward only, and the credit hook gives reward 1.0 to traces whose outcome is `bad`. | `C/src/crucible/studio/datasets.py:233-234` | [ran] `label/sft.jsonl` holds 18 rows. Two of them, T1012 and T1018, have `label: bad, reward: 1.0, resolved: true` (`rewards.jsonl`: every gate 1, `outcome: bad`). |

P2 items are in §2 (ledger) and §3.

## 1. The README user path, run end to end [ran]

The path ran in a fresh directory `SP/run2`. It took 60.2 s wall clock in total, and every command exited 0.

| Step | Wall | Notes |
|---|---|---|
| `pack new demo --dir ./demo` | 0.3 s | The hint prints `crucible pack check demo` (bare name; see D10). It works only because `./demo` is in the CWD: from `/tmp` it fails with `no pack 'demo'`. |
| `pack check ./demo --level semantic` | 1.4 s | ok. Two warns (template text, authors). |
| `bench run ./demo --out runs` | 1.5 s | Writes `bench-demo-bench` plus `bench-demo-bench-controls`, `runs/bench.keys.jsonl` and `bench.key` (the secret) inside the run directory. No `configs/resolved` is written (D12). |
| `label --pack toy --out label` | 0.3 s | 24 traces; sft 18, dpo 0, kto 22; 48 signals. The doc gate says **34** (M15-studio.md:83, drift). The next hint is `python -m crucible.train.launch <job.toml>` with no template. |
| `build toy --out build` | 35.5 s | No progress output for 35 s. The result is "accepted 0/5, provisional 5". The doc gate says "5 accepted" (M15-studio.md:89; drift since the M9 V2 change). |
| `pack check finance_ap --level semantic` | 11.8 s | ok |
| `bench run finance_ap --out runs` | 8.9 s | ok. AUC is printed for finance_ap and is `None` for demo. |
| `label --run build/runs/bench-toy-build` | 0.2 s | 21 episodes and 7 DPO pairs, matching the gate. |
| `bench compare … --base-policy scripted:oracle --cand-policy scripted:half` | 0.2 s | 7 regressions, McNemar p = 0.0156. |

**Reproducibility.**
- I ran `build` twice into `build/` and `build2/` [ran].
- The trees differ only in timestamped `accept/*/evidence/grades/log-<epoch>.json` names. `summary.json` differs in the key commitment, digest and split seal, which follows from the random per-directory key.
- Paths inside `summary.json` are relative to the CWD (`"dir": "build/export"`, B:52,79,134), so a summary read from another directory points nowhere.
- Two `label` runs into separate directories are identical apart from those paths and the binary `store/index.sqlite`.

**Idempotence and resume.**
- Re-running `build toy --out build` takes 1.9 s and adds no results (21 to 21) [ran].
- After `SIGTERM` at 10 s, during accept, the re-run finishes in 27 s with 5 provisional and 21 measured samples, the same as an uninterrupted run [ran].
- The run identity refuses changed `--k` or `--fresh` (good), but it does so after rewriting the resolved config (D6).

**Dry run.**
- `build --dry-run` and `label --dry-run` write nothing [ran]. `build`'s dry run only echoes the manifest hooks and ingest globs (B:42-43).
- It does not say which steps would resume, whether `--out` holds an incompatible run (the D6 refusal), which acceptance checks stay pending without `--docker`/`--paid`, or the planned sample count.
- verifiers' `--dry-run` writes the resolved config (`V/…/eval/main.py:114-117`). Harbor's validates config, preflight and task metadata (`H/src/harbor/cli/jobs.py:1275-1285`).

**Errors.**
- User errors print one line with exit 1 [ran]: missing pack; `label --pack --run`; `export --to atif` without `--run`; `bench compare` on a non-run directory.
- `--verbose` works only before the subcommand. `crucible build x --verbose` fails with "No such option" [ran].
- `crucible build ./demo` on the README's own scaffold says "pack demo has no hooks.forge: build needs them" and gives no next step (`bench run ./demo`) [ran].
- `USER_ERRORS` includes `LookupError` and `ValueError` (`C/src/crucible/cli/main.py:37,56-62`). An internal `KeyError` or `IndexError` therefore prints as a one-line user error with exit 1. Only `pack check` separates exit 3 (internal). OpenEnv `validate` uses 0/1/2/3 (`OE/src/openenv/cli/commands/validate.py:150-151`). agentenv maps only `ValueError`, `NotFoundError` and `DockerBuildError` (`AE/src/agent_env/cli/__init__.py:21-55`).

**JSON.**
- `build --json` and `bench run --json` give pure JSON on stdout with an empty stderr [ran].
- `export` always prints a JSON dump whether or not `--json` is given, with no human summary or `next` (CLI:508).
- `export --to canonical --run X --pack toy` ignores `--run` silently [ran].

**Help.**
- `crucible --help` takes 0.05 s (lazy table works) [ran].
- The `accept` help says "model-free acceptance checks" (`main.py:32`), but the command runs the full contract (CLI:200).

## 2. Requirement ledger: user-facing RL-environment studio

Status values: done / partial / missing / WRONG.

| # | Requirement (source) | Crucible location | Status | Evidence |
|---|---|---|---|---|
| R1 | Run directory holds the resolved config. Sources: verifiers `configs/resolved/eval.json` plus a verbatim launch TOML (`V/verifiers/v1/cli/output.py:71-109`); Harbor `config.json` (`H/src/harbor/job.py:1022-1023`). | `label.py:86-89,118`; B:47 | partial / WRONG | Only `label` and `build` write one (D12), and both write it before the identity check (D6). The `build` config omits `needs` (`--docker`/`--paid`) and the dry-run flag. |
| R2 | Versions, git sha and seeds in every run. Sources: Harbor JobLock (`H/…/lock.py:60-63,668-672`); verifiers trace VersionInfo; AE build metadata (`AE/src/agent_env/utils/build_metadata.py:10-65`). | `C/src/crucible/runtime/run.py:101` (`run.json`) | missing | D12 [ran]. `PK:526 crucible_version()` exists but is used only by `pack check`. |
| R3 | Unique, collision-free run names. Sources: verifiers `<env>--<model>--<harness>--<uuid8>` (`V/…/configs/cli/eval.py:17-30`); Harbor trial `<task>__<shortuuid>`. | `bench-<pack>-bench`, `bench-<pack>-build` | partial | Names are deterministic, which makes resume possible. The explorer collapses equal names (D5). |
| R4 | Resume compares config and refuses on mismatch, before writing. Sources: H job.py:267-278; V main.py:87-113; R2E receipt `config_sha256` (`R2E/…/repository/runner.py:65-91`). | `runtime/run.py` journal identity | partial | Identity refusal works [ran]. It fires after side effects (D6). `label` has no identity at all: a re-run into the same `--out` silently overwrites. |
| R5 | Idempotent and interruptible pipeline. | B: store dedup, accept cache, journal | done | [ran]: 1.9 s re-run; SIGTERM then resume reach the same result. |
| R6 | Dry run shows the plan. Sources: V `--dry-run` writes the resolved config; H validates preflight; NG `eval submit --dry-run --resolve-only` (`NG/nemo_gym/cli/main.py:1196-1214`). | B:42-43; `label.py:111-116` | partial | Writes nothing [ran]. Misses the planned samples, resume state, pending needs and conflicts. |
| R7 | One-line user errors, distinct exit codes, a traceback flag. Sources: AE `_UserErrorsAreNotCrashes`; OE 0/1/2/3; V Ctrl-C exit 130. | `main.py:37,56-68`; CLI:53-79 | partial | One-line errors [ran]. Internal and user errors are conflated (`LookupError`). Exit codes 2 and 3 exist only in `pack check`. `--verbose` is global-only. |
| R8 | A failure is never reported as success. | CLI:111-135, 497-508; EX:129-149 | WRONG | D7 and D8 [ran]. |
| R9 | Secrets never on argv. Sources: V `api_key_var`; H `--ae`, scrubbed afterwards. | CLI options | done | grep finds no key or token option. `--key-file` takes a path. The env key is never written; only its commitment is recorded (`build.json`) [ran]. |
| R10 | Secrets never in resolved configs or logs. Sources: H `templatize_sensitive_env` and `_scrub_jobs_dir` (`H/src/harbor/trial/trial.py:1107-1160`); NG `hide_secrets` (`NG/nemo_gym/secret_utils.py:21-36`); AE describe masking. | `label.py:70-73` (counts only) | done for configs | [ran]: no key material appears in `configs/`, `summary.json` or `run.json`. However, `bench.key` (the HMAC secret) sits in the output directory (`bench/run.py:167-174`), and nothing keeps it out of what a user shares (P2). |
| R11 | Exports exclude secrets and unpublished data. Sources: H uploader allowlist "secrets a user happens to drop … don't leak" (`H/src/harbor/upload/uploader.py:78-113`); OE push ignores `.*` (`OE/src/openenv/cli/commands/push.py:23`); OE template `.dockerignore` lists `.env`. | EX:94-106 | **WRONG (P0)** | D1 [ran]. |
| R12 | Provenance in exported artifacts. Sources: R2E `manifest.json` plus pinned `registry.json` (`R2E/…/hub.py:504-537,585-594`); FE per-file sha256 manifest (`FE/05-multi-harness-rl/prepare.py:89-106`). | EX:53-55,66 (`crucible_export.json` for nemo-gym only); `openenv.yaml` | partial / WRONG | nemo-gym and openenv carry the pack name, version and licence. Canonical carries no manifest. ATIF carries no task, policy or reward (D11). None carries a crucible version, git sha or file hashes. |
| R13 | Exports consumable by other tools. | EX | partial | ATIF passes Harbor's validator (21/21) [ran]. The nemo-gym launcher and openenv module are written. OpenEnv is not importable in `.venv`, so the export is `built_here: false` with a note [ran]. |
| R14 | Progress and cost reporting. Sources: H Rich progress plus `cost_usd` (`H/src/harbor/job.py:1033-1057`; `H/src/harbor/models/job/result.py:41,169`); V dashboard cost; OE collect counts; R2E $5 cap plus BudgetLedger. | B, CLI | partial | `bench` reports cost in the summary and report. `build` and `label` print nothing for 35 s and record no elapsed time [ran]. |
| R15 | JSON output for scripting. | CLI `_emit` | done (except `export`) | [ran] |
| R16 | Next steps printed and correct. | CLI `_emit`; PK:351; B:141; `label.py:160` | partial | Hints are present. They misresolve in D10. A non-run `label --run` still says "train". |
| R17 | Viewer: no path escape. Sources: H `resolve()` with parents check (`H/src/harbor/viewer/server.py:1143-1173,2703-2710`); RL `resolve()` plus `relative_to` (`RL/rllm/eval/visualizer.py:864-878`); AE SPA `is_relative_to`. | EXP:46-54,120-128 | **WRONG (P0)** | D2 [ran]. Run-id traversal is blocked [ran]: `%2e%2e` gives "no run '..'", `a%00b` gives 400. |
| R18 | Viewer: no XSS. Source: AE sandbox CSP for agent-produced HTML. | EXP:120-128; `C/src/crucible/bench/report.py:51-52,128-233` (`html.escape`) | **WRONG (P0)** | The generated report escapes correctly [ran]. Served bytes are trusted (D3). JSON routes return raw strings as `application/json` with `nosniff`, which is fine. |
| R19 | Viewer: auth and binding. | inherited from AE | done (inherited) | Loopback bind, Host allow-list (421 for `testserver` [ran]), CSRF guard [read]. GET routes have side effects: `report` writes `report.html` into the data root [ran], and pack check executes code (D4). |
| R20 | Viewer: bounded pages. | EXP:130-135 | partial | `limit` is clamped to 1..1000. A negative `offset` returns tail rows [ran]. Every page reloads the whole run. `/runs` and `/datasets` scan the whole root on every request. |
| R21 | Viewer: corrupt data surfaces. | EXP:112-118 | missing | A `results.jsonl` of `not json` returns 200 with an empty summary [ran]. |
| R22 | Viewer configuration is not silently replaced. | EXP:32-39 | WRONG (P2) | A malformed `[plugins.crucible]` (`ConfigError`) is swallowed and the plugin falls back to `$CRUCIBLE_DATA` or `./data`. |
| R23 | The README covers the viewer. | README | missing | There is no explorer mention. The README's `--out` directories (`runs`, `build`, `label`) are outside the plugin's default root `data` (EXP:39), so a user who starts `agent-env up` sees nothing. |
| R24 | Validate before publishing. Sources: articles `hf-blog-rl-environments.md:57,68`; `hf-repo2rlenv-how-not-to-build-coding-environments.md:58,72`. | `export` | missing | `export` runs no `pack check` or gold/no-op gate before writing a launcher. NeMo-Gym's own upload returns exit 0 on a failed format check (`NG/nemo_gym/hf_utils.py:137-140`), which is the same anti-pattern. |
| R25 | Input validation on numeric flags. | CLI:115,401 | missing | `--k 0` and `--fresh -1` are accepted [ran]. |

## 3. Explorer probe log [ran] (`SP/probe_*.py`, real `agent_env.explorer.app.create_app()`, `base_url=http://127.0.0.1`)

| Request | Result |
|---|---|
| `/runs/..%2F..%2Fetc/summary`, `/runs/../summary`, `/runs/./summary` | 404 from the router (never reach the plugin) |
| `/runs/%2e%2e/summary` | 404 "no run '..'". `SAFE` admits `..`, but the basename lookup misses. |
| `/runs/a%00b/summary` | 400 "invalid run id" |
| `/packs/%2e%2e/check` | 404 "no pack at ..". `..` is treated as a path relative to the server's CWD. |
| `/packs/demo/check`, `/packs/demo_env/check` | 200. These check `./demo` and `./demo_env` in the server's CWD (D4). |
| `/packs/evilpk/check` | 200, and the pack's module ran in the server process (D4) |
| `/runs/bench-toy-build/summary` | 200 with the first of 5 equal names (D5) |
| `/runs/<id>/episodes?offset=-3&limit=2` | 200 with the last 3 rows sliced to 2 |
| `?offset=99999999999999999999` | 200 with empty items |
| `?offset=abc` | 422 |
| `/runs/evil-run/report` (symlink) | 200 with outside-root bytes (D2) |
| `/runs/evil-run/report` (planted script) | 200 `text/html`, CSP allows inline script (D3) |
| `/runs/bench-xss-run/report` (evil ids) | escaped |
| `/runs/norep/report` | 200. The GET wrote `norep/bench/report.html` via `report_run` with threshold 1.0, not the pack's threshold (EXP:126-127). |
| `/runs/broken/summary` (`results.jsonl` = `not json`) | 200 with empty models (R21) |
| Error bodies | Include the absolute data-root path (P2 information leak on a local tool) |

Every response carried the agentenv CSP, `nosniff` and `X-Frame-Options: DENY`. The plugin adds nothing of its own.

## 4. How the frameworks do it (facts used above; [read], citations checked at their lines)

**Harbor (Apache-2.0)**
- Typer CLI with eager imports (`H/src/harbor/cli/main.py:11-34`). One-line `ValueError` errors and `SystemExit(1)` (`H/src/harbor/cli/jobs.py:1977-1988`).
- The job directory is `jobs/<timestamp>` with `config.json`, `lock.json` (harbor version and git commit) and `result.json`. Trials sit in `<task>__<shortuuid>/` with `config.json`, `result.json`, `agent/`, `verifier/` and `exception.txt`.
- Resume refuses a changed config (job.py:267-278) or lock (job.py:925-937). `harbor job resume` drops cancelled trials (jobs.py:2414-2472). `--dry-run` runs preflight.
- Rich progress and `cost_usd`.
- Secrets: `templatize_sensitive_env` stores `${KEY}` or a masked value, and `_scrub_jobs_dir` redacts env values in every trial file after the run.
- The viewer binds 127.0.0.1, guards paths with `resolve()` plus a parents check (so symlink escapes get 403), caps text at 1 MiB, and sets no CSP (`H/src/harbor/viewer/server.py`).
- Upload uses an allowlist (`H/src/harbor/upload/uploader.py:78-113`). The task packager respects `.gitignore`, but its defaults do not exclude `.env` (`H/src/harbor/publisher/packager.py:13-20`).

**verifiers v1 (MIT)**
- Plugin and config errors become a one-line `SystemExit`. Ctrl-C exits 130.
- Runs go to `outputs/<env>--<model>--<harness>--<uuid8>/` with `configs/resolved/eval.json`, a verbatim `configs/eval.toml`, `traces.jsonl` and `logs/attempt_n/`.
- A non-empty run without `--resume` is refused. `--resume` diffs the configs and lists the changed keys, then keeps good rollouts by `task.hash`.
- Each trace records `VersionInfo(version, commit)`. `--dry-run` writes the resolved config.
- Gap: `client.headers` is dumped unmasked into `eval.json` (`V/verifiers/v1/configs/client.py:49`, `V/verifiers/v1/cli/output.py:80`). The platform push deliberately sends counts only (`V/verifiers/v1/utils/platform.py:54-67`).

**agentenv (Apache-2.0)**
- Click CLI with `_UserErrorsAreNotCrashes` and `--verbose`. `--json` on `config` and `plugin`. `env:`/`secret:` references, and `describe` masks secrets.
- Explorer:
  - loopback-only bind, Host allow-list, CSRF guard, app CSP;
  - the objects proxy serves HTML, SVG and XML under `CSP: sandbox` and `nosniff`;
  - the SPA uses `is_relative_to`.
- There is no written guidance for plugins that serve files or HTML. This is the seam Crucible fell through.
- Gap: artifact put uploads a whole directory with no `.env` exclusion (`AE/src/agent_env/cli/artifact/file_artifact_universe.py:76-81`).

**NeMo-Gym (Apache-2.0)**
- argparse `COMMANDS` table with a global `--json` that is rejected where unsupported (`NG/nemo_gym/cli/main.py:1294,1430-1434`). Unknown flags get "did you mean".
- Config order is YAMLs, then `env.yaml` (secrets), then overrides. `gym env resolve` uses `hide_secrets`.
- The resolved config is not saved to the output.
- `resume_from_cache` skips finished (task, rollout) keys but does not compare config.
- `dataset upload` exits 0 on a failed format check.

**rLLM (Apache-2.0)**
- Click `_LazyGroup` (`RL/rllm/cli/main.py:19-56`), the pattern Crucible ported. `CliError` escapes Rich markup.
- `meta.json` records no version or sha.
- The viewer guards with regex, `..` rejection and `resolve()` plus `relative_to`. Gaps: an unescaped root path in the HTML, and JSON inside `<script>` without escaping `</`.

**SkyRL (Apache-2.0)**
- No CLI. The config is logged unredacted.
- Resume checks only `global_step`, and loads the pickle with `weights_only=False`.

**OpenEnv (BSD-3-Clause)**
- Typer CLI. `validate` exit codes 0/1/2/3 with `--json`. `init` refuses a non-empty directory.
- `collect --resume` overwrites `metadata.json` without comparing it.
- `push` ignores `.*` and `__pycache__` but not `.gitignore`. It checks structure only, not `validate`. `--secret` values are redacted in errors.

**FineEnvs (Apache-2.0)**
- `prepare.py` pins revisions, checks sha256, writes a per-file hash manifest and fails on train/test overlap.
- Gaps: stale files in the target are hashed in, and the `deploy.py` exclude list has no `.env`.

**Repo2RLEnv ("Apache-2.0 AND MIT")**
- Errors go through `report_error` with exit 2, as JSON under `--json`.
- The spec stores env-var names, not values.
- The run receipt hashes config plus runtime wheel plus source. Re-running needs `--resume`, and a changed config fails with "use a new run ID".
- Hub push writes `manifest.json` and a pinned `registry.json`. Gap: task directories are copied whole, so a `.env` would be uploaded.

**SWE-smith (MIT)**
- Per-module argparse with skip-if-done (`--redo_existing`) and no config check.
- An unseeded `random.shuffle` sits in `SS/swesmith/bug_gen/llm/modify.py:71`.

**Articles**
- Run oracle and verifier before any model (`hf-blog-rl-environments.md:57,68`).
- Gold = 1.0 and no-op = 0.0 first (`hf-repo2rlenv-how-not-to-build-coding-environments.md:58,72`).
- Keep gold and hidden tests off the box (`:63,77`).
- Publish files, a run command and the reward rule (`hf-blog-rl-environments.md:191`).
- Splits and versions (`hf-space-rl-environments-guide.md:291`).

The pattern across sources: the mature tools (Harbor, verifiers, R2E) compare config before writing and stamp versions. Every packaging path that copies directories whole (AE artifact put, R2E task copy, Harbor packager defaults, FE deploy) has the D1 hole; only Harbor's job upload uses an allowlist. Harbor and rLLM resolve before serving; Crucible's report route does not.

## 5. Picks

| # | Kind | What | Licence | Effort | Fixes |
|---|---|---|---|---|---|
| 1 | PATTERN | An **allowlist bundle**, as in Harbor's job upload (`H/src/harbor/upload/uploader.py:78-113`), with OpenEnv's `.*` ignore (`OE/src/openenv/cli/commands/push.py:23`). Copy `pack.yaml`, README, LICENSE, `pyproject.toml` and `src/<package>/` without dotfiles, plus only the runtime data the manifest declares. Never copy ingest globs or `redaction_env`. Fail if any copied file matches a secret pattern or a loaded dotenv value (reuse `RedactionPolicy`). | Apache-2.0 / BSD-3 (pattern only) | S | D1, R11 |
| 2 | DEPEND | **agentenv's sandbox rule for agent-produced HTML** (`AE/src/agent_env/explorer/routers/objects.py:20-25,97-98`). Serve `report.html` with `Content-Security-Policy: sandbox` and `nosniff`, or re-render it from `summary.json` on request instead of trusting file bytes. Resolve, then `is_relative_to(root)`, for every file read (`H/src/harbor/viewer/server.py:1143-1150`; `RL/rllm/eval/visualizer.py:864-878`). | Apache-2.0 | S | D2, D3 |
| 3 | PORT | **The verifiers / R2E resume receipt** (`V/verifiers/v1/cli/eval/main.py:87-113`; `R2E/…/repository/runner.py:65-91`). Hash the resolved config and runtime, compare it to `<out>/configs/resolved/<cmd>.json` before any step runs, refuse with the changed keys and "use a new --out", and only then write. Apply this to `label`, `build`, `bench run` and `export`. | MIT / Apache-2.0 | S-M | D6, D12, R1, R4 |
| 4 | PATTERN | **A version stamp**, as in Harbor `JobLock.harbor` (`H/src/harbor/models/job/lock.py:60-63`) plus verifiers `VersionInfo`. Record `{crucible version, git sha, dirty, python, agentenv version, pack name, version and content hash}` in `run.json`, every `summary.json` and every export manifest (canonical and ATIF included). `PK:526 crucible_version()` already exists. | Apache-2.0 / MIT | S | D12, R2, R12 |
| 5 | PATTERN | **Exit-code contract** from OpenEnv `validate` (0/1/2/3), already in `pack check`, extended to every command. Narrow `USER_ERRORS` to a Crucible `UserError` plus `FileNotFoundError` (agentenv's narrower set). `build`, `export` and `label` exit 1 when something is rejected, an export errors, nothing is measured, or the output is empty. `click.IntRange(min=1)` on `--k`. Print bench warnings in `build`. | BSD-3 / Apache-2.0 | S | D7, D8, R7, R8, R25 |
| 6 | PATTERN | **Harbor's trial layout for ATIF export**: write a `manifest.jsonl` (or `<name>.result.json`) next to each trajectory with task id, policy, harness, reward, gates, mask kind and run identity. Name files `<task>__<policy>__<sample>`. | Apache-2.0 | S | D11 |
| 7 | PATTERN | **Explorer run ids are paths**, not basenames (Harbor job/trial paths). Use the run directory relative to the root, URL-encoded, or the journal identity. Hide `accept/*/a11/*` sub-runs. Clamp `offset >= 0` and paginate from the journal instead of reloading. | n/a | S | D5, R20 |
| 8 | PATTERN | **Isolate pack check** in the explorer: index names only, no CWD fallback, and `check_in_subprocess` (already used by `--all`, CLI:59-60), or drop the route. | n/a | S | D4 |
| 9 | FIX | `bench_key` re-raises an invalid `$CRUCIBLE_BENCH_KEY` instead of falling back. `pack new` prints `./`-prefixed or absolute paths in its hints (PK:351). | n/a | XS | D9, D10 |
| 10 | PATTERN | **Rich progress and elapsed time** in `build` and `label` (`H/src/harbor/job.py:1033-1057`; OE collect counts). Add a README section for the explorer (`[plugins.crucible] data_root`) and default `--out` under `data/`. | Apache-2.0 | S | R14, R23 |
| 11 | TEST | Replace the vacuous traversal assertion (`C/tests/unit/test_explorer_plugin.py:41`) with plugin-level probes: a symlinked `report.html`, a planted script, duplicate run names, `/packs/<cwd dir>/check`, and a negative offset. Add CLI tests for D6, D7, D9 and D10 and the D1 bundle scan. | n/a | S | D13 |

**Cross-group note for the label owners (D14).** `--select correct` should also require that the trace's outcome is not `bad`, or the gated reward should be capped by the outcome. As it stands, the default SFT set trains on traces the expert signals reject.

## 6. Probe index (`SP/`)

| Probe | What it does |
|---|---|
| `run1/` | Exploratory README-path runs: build, build2, the `build_k`/`build_f` re-runs (D6), x4 (k=0), x5 (fresh=-1), xk (SIGTERM and resume), `xkey*` (bench key, D9), `exp_*` (all four export targets), `demo_env` (D1), `evilpk` (D4), `toy`-named scaffold (D10) |
| `run2/` | Clean timed README path (§1) |
| `repro_p0.sh` | Self-contained D1 reproduction, verified in `SP/repro/` |
| `probe_app.py` | Explorer route sweep and duplicate ids (D5, §3) |
| `probe_exec.py` | Pack code executed by GET (D4) |
| `probe_link.py` | Symlink escape (D2) |
| `probe_csp.py` | Planted HTML and CSP header (D3) |
| `probe_xss.py` | Hostile task and policy ids through `run_bench` and the report, episode and summary routes (escaped) |
| `probe_rep.py` | GET-generated report and corrupt run |

Each probe runs as `CRUCIBLE_DATA=<root> HOME=SP/home .venv/bin/python SP/probe_*.py`.
