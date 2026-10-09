# M1 `world` — runtime for any environment

**Status: built; exit gate passed (2026-10-09).** Code: `src/crucible/world/`, `src/crucible/interop/{harbor,agentenv,mimo,openenv}/`, `src/crucible/_vendor/{mimo_sidecar,openenv_awm}/`. Tests: `tests/unit/test_{state_adapters,world_local,agentenv_backend,harbor_backend,postgres_app,website_browser,ci_app}.py`, `tests/redteam/`, `tests/toy_e2e/`, `tests/gpu/`.

## What it is
A gym-like runtime over any World: Apps (state + tools), Actors, a virtual Clock and Events. Policies call `reset` / `step`. Verifiers get a `VerifierView` of State Adapters, never handed to a policy. Whole worlds snapshot and restore exactly, and that is verified by fingerprints.

### Pieces
| Piece | What | Built on |
|---|---|---|
| `exec.py` | Where state lives and commands run: `LocalExec`, `DockerExec`; `HarborExec` in interop | — |
| `state/` | State Adapters: `fs` / `fs_git` (managed set = tracked + untracked − ignored, so the build outputs a task's collect hook ignores are derived state), `sql` (SQLite; online backup; restore *into* the live DB so open connections see it; canonical row-order-free dump; a missing DB is a valid state), `json_doc`, `process` (builds, tests, GPU probes) | canonicalization written once as standalone stdlib snippets (`state/snippets/`) that run in-process locally and are shipped verbatim into containers |
| `apps/` | `common/workspace`, `common/git_repo`, `common/sqlite_app` (MiMo tool modules, in-process or over MCP; or generic SQL tools), `common/mailbox`, `common/doc_library` (docx/xlsx/pptx/pdf extraction), `common/json_api`, `common/toolchain` | MiMo tool-module format |
| `packaging/` | `mcp_tools.py`: MiMo modules loaded with MiMo's own loader and wrapped as FastMCP `Tool`s, so in-process calls are byte-identical to served ones. `sidecar.py`: serve modules with the ported `mcp_http`, supervise, MCP client. `app_server.py`: **any** App served over MCP (namespaced tools, `/ready`) | MCP SDK low-level server, ported MiMo `mcp_http.py` |
| `clock.py` | Virtual clock (step / wall), event scheduler, per-actor ACL (ordered grants / revokes) | agentenv gateway clock and trigger semantics |
| `world.py` | `World`: lifecycle, tool routing + ACL, messages, responders run *inside* `step` (simulated people are environment dynamics), events, snapshot / restore / reset, trace export | — |
| `backends/local.py` | Everything on this machine (no isolation; refuses GPU Apps and policy shells unless opted in) | — |
| `interop/harbor/world_backend.py` + `gpu_docker.py` | Sandbox from Harbor's `EnvironmentFactory`; GPU via a compose overlay subclass (`runtime: nvidia`, `network_mode: none`, `init`, shm), remote Docker through `DOCKER_HOST=ssh://…` (inherited by Harbor's compose subprocess) | Harbor `DockerEnvironment`, `EnvironmentFactory`, `TrialPaths`, prebuilt-image path |
| `interop/agentenv/world_backend.py` | Apps behind a **real agentenv gateway** (subprocess). Every App is served over MCP. One gateway role per actor (`/tools/disable|enable`), so access is enforced by agentenv itself. Wall clock via `/clock/set-time`; events become gateway triggers (time / action with tool, permission, nl); nl goes through an agentenv-protocol A2A relay agent into the world inboxes | agentenv `Gateway`, triggers, clock, `agentenv_protocol.a2a_agent` |
| `interop/mimo/` | Any MiMo env → World (one `sqlite_app` per `tools/<app>.py`, a writable workspace, verifier files withheld) + sweep check | MiMo release format |
| `interop/openenv/awm.py` | Any AgentWorldModel-1K scenario → World (generated FastAPI service with FastApiMCP; SQLite state) + sweep check | ported OpenEnv `data_loader` / `db_manager` / `_patch_env_code` |

## Added after the gate (2026-10-09): enterprise Apps and isolation

These were deferred because the coding MVP did not need them. They are what the finance pack and hidden-check verification (M7) build on.

| Piece | What | Built on |
|---|---|---|
| `state/postgres.py`, `apps/postgres_app.py` | **Database server** as a system of record: `postgres:16` in its own container; tools `list_tables`, `describe_table`, `sql_query` (read-only transaction), `sql_execute` (opt-in); MiMo-style tool modules get `DATABASE_URL` and are served over MCP. Canonical fingerprint = schema (columns, constraints, indexes, **sequences**) + per-table sorted `row_to_json` hashes; snapshot `pg_dump -Fc`, restore `pg_restore` + verify. Restoring a snapshot resets sequences, so the next insert gets the same id | Postgres, psycopg2 |
| `packaging/session.py`, `apps/mcp_server.py` | **Any MCP server** as an App (stdio command or streamable-HTTP URL) over one persistent session per App, so stateful servers (a browser) keep their state between calls; tool allowlist by glob; `{app:<id>}` resolves a peer App's URL | MCP SDK |
| `apps/website.py`, `website_runner.py` | **Website** App: an ASGI app (`create_app(db_path)`) served by uvicorn over a SQLite state, so the site's data snapshots and restores like any SQL App | Starlette / uvicorn |
| `apps/browser.py` | **Browser** App: Playwright MCP (`@playwright/mcp`, headless, isolated profile, origins limited to the world's sites); `browser_evaluate` withheld from the policy. `BrowserStateAdapter` fingerprints URL, cookies and localStorage, and restores them | Playwright MCP |
| `apps/ci.py` | **Hidden-check CI as a tool**: `run_ci` snapshots the source App, restores it into a separate runner (local directory or a `--network none` container, optionally a task's verifier image, optionally on a remote GPU host), runs checks it never shows, and returns `name: PASS/FAIL/TIMEOUT` only (output only for `show_output` checks, scrubbed). Runs are budgeted and cost virtual time; the budget is App state, so restoring a snapshot gives runs back | — |
| `world/leaks.py` | Leak scrubber for anything the policy reads (instructions, replies, CI output) | Repo2RLEnv patterns (Apache-2.0) |
| `world/guard/` | **Isolation guard** (below) | Karotte (MIT), Repo2RLEnv |

The toy pack's vendor portal shows the website and browser Apps end to end: a clerk disputes invoice INV-1003 through the browser; the site's database records the dispute; restoring the earlier snapshot rewinds both the database and the page; navigating to GitHub is blocked (`tests/unit/test_website_browser.py`).

### Isolation (`WorldSpec.isolation`)

`isolation.policy_user` turns a container world into a hardened one. After the Apps seed their files as root, the backend installs `guard.py` (a standalone stdlib script, root 0600) and:
1. creates the policy user, hands it the writable App roots, and makes `root_only` paths and the harness directory 0700 root;
2. scrubs git history for Apps with `scrub_history` and **verifies** it (every reachable commit is an ancestor of the kept base, nothing unreachable remains, forbidden SHAs are absent);
3. routes every policy tool (shell, file tools, search) through `guard run` as the demoted uid, and pins the verifier side to the resolved interpreter with `python -I` and a root-only staging directory;
4. runs an egress self-test as the policy uid and refuses to start the world if anything answers;
5. kills the policy's whole process cohort before verification snapshots (`World.quiesce`, `snapshot(quiesce=True)`; `run_episode` takes its final state this way).

`tests/redteam/test_attacks.py` attacks this on local Docker. Each attack first asserts it took effect, then that it was defeated: root-only answers read directly and via symlink; `git log --all` / `git show <future-fix>`; socket connect and `pip download`; writing harness paths, replacing `/usr/local/bin/python3`, killing PID 1; a fake `python3` planted in the policy's `PATH`, rc files, workspace and `/tmp`; a FIFO and a 20 GB sparse file (snapshot in under 30 s, both reported as anomalies); lowering `oom_score_adj`; a fork bomb (`pids_limit`); a background process rewriting the code after submission (identical fingerprints across two quiesced snapshots); reading the hidden CI checks. Passed on 3 of 3 runs.

## Exit gate

| Requirement (ARCHITECTURE M1) | Result |
|---|---|
| **Snapshot round-trip on the 5 seed worlds** (reset → oracle → snapshot → restore into fresh containers → identical fingerprints, 3/3) | ✅ **15/15 restores identical** on the lab RTX 5090 through Harbor (GPU visible in every sandbox; `network_mode: none`). See `reports/m1_seed_worlds_gpu.json` |
| … and the toy world | ✅ Local and agentenv backends (`tests/toy_e2e`); restore into a fresh world (`test_restore_into_fresh_world`) |
| **Generality: ≥ 20 MiMo envs** load, tools match, fingerprints stable | ✅ **266/266** complete MiMo envs pass every check in-process: App ids = MiMo discovery = manifest; tool names = public functions (13,052 tools in 1,779 apps); verifier files absent; reset stable; mutation visible and undone (`reports/m1_mimo_sweep_inprocess.json`). The same checks pass on stratified samples served over **MCP by MiMo's ported `mcp_http`** (30/30, `m1_mimo_sweep_mcp.json`) and **behind the agentenv gateway** (21/21, `m1_mimo_sweep_agentenv.json`) |
| **Generality: ≥ 20 AWM envs** | ✅ **100/100** evenly spaced AgentWorldModel-1K scenarios pass every check: generated FastAPI service up, tools over MCP (3,494 tools over 1,912 tables), reset stable, mutation visible and undone. See `reports/m1_awm_sweep.json` |
| **Clock / events** fire at the right virtual time on the agentenv backend | ✅ Time trigger (nl via A2A relay) and permission trigger fire through agentenv's own driver; the action trigger runs a tool through the gateway (`test_agentenv_backend.py`) |
| Cross-backend canonical fingerprints | ✅ Same repo + DB fingerprint identically on macOS (in-process) and in a Linux container; a local snapshot restores bit-exactly in the container |

Seed-world numbers (lab RTX 5090, remote Docker over SSH):

| seed | main repo files | oracle diff (files) | snapshot | restore (×3) |
|---|---|---|---|---|
| task01_prims_omp | 27 | 47 | — | 3/3 equal |
| task01v2_tessera_omp | 82 | 108 | — | 3/3 equal |
| task02_pcops_triton | 9 | 29 | 2.6 s | 3/3 equal, 5.5 s each |
| task03_mdlite_pallas | 19 | 31 | 2.6 s | 3/3 equal, 5.5 s each |
| task04_pcl_gpu_cuda12 | 3,468 (68.6 MB tar) | 56 | 9.0 s | 3/3 equal, 15.6 s each |

## Framework findings (each handled in code; worth knowing before relying on these frameworks)
1. **agentenv:** the gateway starts its time-trigger driver in the FastMCP server *lifespan*, which only runs while an MCP client session is open. REST-only use (`/step`) never fires time triggers. The backend holds one MCP session open for the world's lifetime.
2. **agentenv:** tool names must be unique across backing servers (the gateway raises otherwise), so the backend namespaces them `<app>__<tool>`.
3. **agentenv:** relative time triggers re-anchor to the new `t0` whenever the clock is re-armed. A fresh episode therefore re-arms at the spec's `t0` and re-registers triggers; a mid-episode restore re-arms at the snapshot's time.
4. **agentenv:** A2A URLs are base URLs; the client and the gateway executor append `/a2a` themselves.
5. **Harbor:** the local Docker provider has no GPU path; we added `GpuDockerEnvironment`, following Kata's overlay pattern. Docker never uses the no-network compose file; without a `network_policy` a task starts public. We isolate with `network_mode: none`, as our migration runs did.
6. **Harbor:** `exec` merges stderr into stdout and has no stdin; `HarborExec` stages stdin through a file.
7. **MiMo:** `sidecar_entrypoint.py` probes readiness via `/proc/net/tcp` (Linux-only) and advertises `/dump` and `/verify`, which are not implemented. `--start-and-detach` loses the health thread. Replaced by our sidecar. One env defines a tool twice (the last wins); one has a tool without a DB (now a valid empty state); one has an interrupted download (skipped).
8. **AWM data:** some generated sample INSERTs are invalid (OpenEnv's loader logs and skips them), and several scenarios seed rows with `datetime('now', …)`, so their initial DB depends on wall-clock time. Tasks forged from AWM must freeze "now" to be reproducible (M8).
9. **mcp 1.30:** `streamablehttp_client` is deprecated; we use `streamable_http_client` with our own httpx client.
10. **Harbor 0.23 / agentenv (per Karotte's comparison):** neither demotes the agent, kills leftover processes before grading, limits resources or blocks egress by default. With `isolation.policy_user`, Crucible does all four on the Harbor backend (above); the agentenv backend does not yet (its tools run in the gateway's process).
11. **Playwright MCP:** element arguments are `target` (an accessibility ref), not `ref`. `browser_evaluate` replies are Markdown; only the `### Result` section is JSON. Navigations race evaluations ("Execution context was destroyed"), so reads retry. `--allowed-origins` is documented as not a security boundary, so network isolation still has to come from the world.
12. **git:** `ls-files --others` never lists FIFOs, sockets or devices, so a git-aware state adapter cannot see them without a separate lstat sweep.
