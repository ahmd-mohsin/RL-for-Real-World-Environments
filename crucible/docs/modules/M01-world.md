# M1 `world` — runtime for any environment

**Status: built; exit gate passed (2026-10-09).** Code: `src/crucible/world/`, `src/crucible/interop/{harbor,agentenv,mimo,openenv}/`, `src/crucible/_vendor/{mimo_sidecar,openenv_awm}/`. Tests: `tests/unit/test_{state_adapters,world_local,agentenv_backend,agentenv_sandbox,harbor_backend,sandboxes,postgres_app,website_browser,ci_app,virtual_env,episode_loop,audit}.py`, `tests/redteam/`, `tests/toy_e2e/`, `tests/gpu/`.

## What it is
A gym-like runtime over any World: Apps (state + tools), Actors, a virtual Clock and Events. Policies call `reset` / `step`. Verifiers get a `VerifierView` of State Adapters, never handed to a policy. Whole worlds snapshot and restore exactly, and that is verified by fingerprints.

### Pieces
| Piece | What | Built on |
|---|---|---|
| `exec.py` | Where state lives and commands run: `LocalExec`, `DockerExec`; `HarborExec` in interop | — |
| `state/` | State Adapters: `fs` / `fs_git` (managed set = tracked + untracked − ignored, so the build outputs a task's collect hook ignores are derived state), `sql` (SQLite; online backup; restore *into* the live DB so open connections see it; canonical row-order-free dump; a missing DB is a valid state), `json_doc`, `process` (builds, tests, GPU probes) | canonicalization written once as standalone stdlib snippets (`state/snippets/`) that run in-process locally and are shipped verbatim into containers |
| `apps/` | `common/workspace`, `common/git_repo`, `common/sqlite_app` (MiMo tool modules, in-process or over MCP; or generic SQL tools), `common/mailbox`, `common/doc_library` (docx/xlsx/pptx/pdf extraction), `common/json_api`, `common/toolchain` | MiMo tool-module format |
| `packaging/` | `mcp_tools.py`: MiMo modules loaded with MiMo's own loader and wrapped as FastMCP `Tool`s, so in-process calls are byte-identical to served ones. `sidecar.py`: serve modules with the ported `mcp_http`, MCP client (no supervisor: a restart would hide an outage). `app_server.py`: **any** App served over MCP (namespaced tools, `/ready`) | MCP SDK low-level server, ported MiMo `mcp_http.py` |
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

`isolation.policy_user` turns a container World into a hardened one, on every container backend (Harbor, or agentenv with a sandbox). One module does it, `world/guard/isolate.py`, after the Apps seed their files as root and before the policy's first step:
1. **A fresh account.** The guard (`guard.py`, standalone stdlib, root 0600) creates the policy user and never reuses an image account. Ubuntu-based images ship `ubuntu` at uid 1000, which might own files or have sudo, so the uid is allocated. The world refuses to start if root's `PATH` or interpreter chain contains anything the policy could write: it would plant `git` or `tar` there, and the verifier would run them as root.
2. **Git the verifier can trust.** Where asked, history is scrubbed and the scrub is **verified**: every reachable commit is an ancestor of the base, nothing unreachable remains, and forbidden SHAs are absent. Every repo's `.git` is also copied to root-only storage. Verifier-side git (fingerprints, snapshots, diffs) uses only that copy, through `--git-dir/--work-tree`, so hooks, filters, `fsmonitor`, external diff drivers and `info/exclude` planted by the policy are never read.
3. **Ownership.** The policy-writable App roots are handed to the policy. App-owned state (SQLite, mailbox, JSON documents) may not sit inside them and is root `0600`. The sandbox workdir is `0711`.
4. **A pinned verifier side.** The verifier uses `python -I -S` at the path resolved inside the sandbox, and stages snippets, stdin and snapshot payloads in a root-only directory, never `/tmp`.
5. **Demoted policy tools.** Every policy tool (shells, file tools, the doc library, the toolchain shell) runs through `guard run`. That means: `setsid`; `PR_SET_NO_NEW_PRIVS`, so setuid binaries and sudo are inert; `oom_score_adj=1000` pinned with `CAP_SYS_RESOURCE`; `RLIMIT_NPROC` below the container's pids limit, so the harness can always exec the guard; and the image's `PATH`. When a command exits, its process group goes with it; work detached with `setsid` lives until the next quiesce.
6. **No egress.** An egress self-test runs as the policy against the internet, cloud metadata, the default gateway, IPv6 and UDP DNS (plus the agentenv gateway's host port). The world refuses to start if anything answers.

Between episodes, `World.restore` first kills the policy's process cohort, then writes the snapshot with directory fds and `O_NOFOLLOW`. Files are created `O_EXCL` and owned by the policy, so a symlinked parent can't redirect a root write. Then the policy's `.git` is reset to the trusted copy, and `/tmp`, `/var/tmp`, `/dev/shm`, its home and its SysV IPC objects are emptied. Root lacks `CAP_IPC_OWNER` in Docker, so the IPC removal runs as the owner. Verification snapshots quiesce first (`snapshot(quiesce=True)`; `run_episode` does it). A cohort that cannot be killed is scored as `crucible:policy_misbehavior`, not masked.

`tests/redteam/test_attacks.py` attacks this on local Docker. Each attack first asserts it took effect, then that it was defeated:
- root-only answers read directly or through a symlink, by the policy *and* by verifier-side reads of the policy's tree;
- a FIFO answer;
- `git log --all` and `git show <future-fix>`;
- git-ref injection into a root shell;
- git config traps (fsmonitor, external diff, clean filter, hooks): these fire for the policy and never for root;
- hiding a file through `.git/info/exclude`;
- restoring through a parent replaced by a symlink to `/etc`;
- leftovers in `/tmp`, `/dev/shm`, home, rc files and SysV shared memory across a reset;
- an image account with passwordless sudo and a setuid-root `bash`;
- network, including `pip download`;
- harness paths;
- a fake `python3` planted everywhere;
- a FIFO and a 20 GB sparse file;
- the OOM floor;
- a background job without redirection;
- a verifier timeout that must kill the process inside the container;
- a fork bomb;
- a post-submission tamper loop;
- CI candidate code that rewrites the hidden check and leaves a daemon behind.

A second world, whose image puts a policy-writable directory on root's `PATH`, must refuse to start.

### Sandboxes and GPUs

| Where the World's state and commands run | Backend + params | GPUs | Verified |
|---|---|---|---|
| This machine's Docker | `harbor` (`environment_dir`/`task_dir`/`docker_image`) or `agentenv` + `sandbox: {provider: docker, image}` | nvidia runtime (`gpu_mode: runtime`) or `--gpus` | red-team suite, `test_agentenv_sandbox.py` |
| A GPU host's Docker over SSH (the lab RTX 5090) | the same + `gpu_host: lab5090` (`crucible.toml [gpu_hosts]`) | `runtime: nvidia` (that host's `--gpus` lacks the MPS mount) | seed worlds 3/3, CI on the verifier image, agentenv sandbox with `nvcc 12.8` (`reports/m1_gpu_sandboxes.json`) |
| Any Harbor provider (Modal, Daytona, GKE, E2B, Runloop, …) | `harbor` + `environment_type: modal` (+ `environment_kwargs`) | `cfg.gpus` + `cfg.gpu_types` from `resources.gpu_type`; network policy passed through | wiring unit-tested; live run needs provider credentials |
| Any agentenv sandbox provider (Modal, E2B, Sail, registered ones) | `agentenv` + `sandbox: {provider: modal, image}` | Modal: agentenv's `ModalSandboxProvider(gpu="H100" / "A100:2" / ...)` | adapter unit-tested through a fake provider; `CRUCIBLE_MODAL=1` runs it live |

`Resources.gpu_type` asks for a GPU class on any backend. A WorldRef records the digest of the container image's filesystem layers, and restoring it into a World that runs a different image raises `SnapshotMismatch` (DESIGN B.5: stale images). Layers, not image ids, because Compose stamps project labels into the images it builds. The same guard hardens every container option. In a sandboxed agentenv World the gateway and App servers stay on the host as trusted routers. Apps that would spawn host-side servers are refused, and so is a host-side tool module: it would run outside the sandbox.

### The tool-call choke point (`World._call_tool`)

Every call goes through one place: the policy's, simulated people's and events'. It checks names and the ACL, validates arguments against the tool's JSON schema (bad arguments come back as `invalid_arguments` with field errors), suggests near matches for unknown tools, enforces a deadline (`limits.tool_timeout_s`), and contains App crashes as `internal_error`. The traceback stays in the trace, and the episode is masked as an infrastructure failure. It labels every failure's `fault` (`policy` / `tool` / `infra`) and bounds output once (head + tail with a banner; the full text goes to the store).

MiMo-style tool modules read the World's **virtual clock and seed** instead of wall time and `os.urandom` (`packaging/virtual_env.py`). `datetime`, `time`, `uuid` and `random` are rebound inside the loaded modules only, so replaying a call from the same WorldRef gives the same database. Before this, 1,174 of the 1,796 MiMo modules made every replay different.

### World audit (`crucible world validate`)

`world/audit.py` runs OpenEnv RFC 008-style checks on any World:
- reset stability and the state contract;
- tool declarations and a smoke call of read tools;
- static and dynamic leakage (fix pointers, forbidden strings in any observation);
- episode isolation and determinism (gold replayed twice);
- oracle passes / no-op fails (with a verifier predicate);
- for isolated Worlds: network policy, planted setuid/ownership anomalies, and root-only paths read as the policy.

A check that cannot run is a named SKIP. Each planted defect in `test_audit.py` is caught, and the hardened red-team World passes.

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
13. **agentenv:** the gateway grants tools by a self-asserted `AgentEnv-Role` header, and passes tool arguments to backing servers unauthenticated. Crucible's environment dynamics now call Apps directly; the system role has no tools; the actor identity carried to App servers is HMAC-signed. Relative time triggers re-anchor whenever the clock is set (so restored worlds re-fire them). Triggers are now registered at absolute instants; snapshots wait for in-flight triggers to settle (agentenv's own `prompt_agent` logic); restores re-register only what had not fired.
14. **Harbor / Docker:** Harbor's exec timeout kills only the local `docker` client; the command keeps running in the container. Deadlines are now enforced inside the sandbox (`timeout -s KILL`). `--security-opt no-new-privileges` breaks `--init` on the lab host (`exec /sbin/docker-init: operation not permitted`), so no-new-privs is set per policy process tree instead. Container root has no `CAP_IPC_OWNER`, so it cannot remove the policy's SysV segments; the owner has to.
15. **Images:** Ubuntu-based images ship a uid-1000 user, and a guard that "creates" uid 1000 silently becomes that user. Task images run Python 3.8–3.10, so `shutil.rmtree(dir_fd=)` (3.11+) cannot be used in snippets. Some images have no `timeout`, and BSD/busybox `base64` flags differ, so commands use the portable forms.
16. **MiMo:** 1,174 of the 1,796 tool modules read the wall clock, and 707 return `{"error": {...}}` as a successful value. Both are now handled (virtual time; envelope → `ok=False`, `fault=tool`).
17. **Docker:** `docker exec` into a container that was removed, paused or stopped from outside, or on a daemon that cannot be reached, is an ordinary non-zero exit, so the policy saw `[exit 1] No such container` and was blamed. `DockerExec` and `GVisorExec` now ask the daemon (`DockerExec.lost`: `inspect`, then `events` for `kill`/`stop`/`destroy`/`pause`) and raise `EnvContainerLost`; the World records `crucible:env_container_lost` (infra), for Harbor too (it was `crucible:tool_crashed`). The words alone never decide: a live container printing them is the command's output. A container whose own main process exited is the episode's doing (M1 audit 2026-10-14 R-1).
18. **Output size:** `Popen.communicate` and Harbor's exec keep every byte a command prints, so one `yes` grew the orchestrator by ~1 GB/s (only the guard capped output, and only with `isolation`); `communicate` also waited for background jobs holding the pipes (`sleep 30 &` held the call 30 s). `_blocking_run` now reads both pipes as they fill and keeps the first 1 MiB and last 16 MiB of each (`OUT_HEAD`, `OUT_TAIL`, a `...[N bytes dropped]...` marker between), returning 1 s after the command exits; `HarborExec` cuts the same head and tail inside the container (`capped_output`: `head -c` / `tail -c`, exit status from `PIPESTATUS`, unmarked so what fits is exact). The tail is large because `run_python` returns its result at the end of stdout (M1 audit 2026-10-14 R-2).
19. **Deadline layering:** the World's `tool_timeout_s` and the workspace shell's cap were both 3600 s (the toolchain shell had no cap), so the World's timeout fired first on a policy's hung command and masked it as `transport_timeout`: a policy could escape a 0 by hanging. A tool's own deadline is now at most `tool_timeout_s` (`App.deadline`, which also turns a requested 0 into 1 s), and the World waits `TOOL_GRACE_S` (60 s) longer, so the policy gets its scored `[TIMEOUT]` and only a tool hung past its own deadline is masked (NeMo-Gym's outer deadline = tool timeout + cleanup + 30 s; M1 audit 2026-10-14 R-4).
20. **Leftover containers:** sandboxes ran `sleep infinity`, so a crashed or killed run's containers lived forever, and the perf guard (no other `crucible*` container on the host) then masked every performance grade there as `node_contention`. A sandbox now runs `sleep <timeout_s + 15 min>` (`SandboxSpec.timeout_s`, already its documented lifetime; `--rm` removes it when it ends), carries `crucible.owner=<host>:<pid>`, and the perf guard ignores containers whose owner (ours or the RLM kernel's label) is a dead process on this host. `--rm` also deletes a container the moment its own main process dies, so `DockerExec.lost` reads the events for a removed container: died with no `kill`/`stop` before its lifetime ran out is the episode's doing, anything else (or no evidence) is lost (M1 audit 2026-10-14 R-6). Since the 2026-10-20 merge into dev the keeper, labels, perf guard rule and sweep are M10 lab audit R-6's `world/lifecycle.py` (renewable life loop, `crucible.owner`/`crucible.expires`, expired orphans ignored); `DockerExec.expires_at` is the keeper's deadline, moved out by each renewal.
21. **Teardown under cancellation:** `World.close` set `_closed` first, stopped the Apps with no bound and called the backend's teardown outside any `finally`, so a cancel during close (a second Ctrl-C, a task group shutting down) skipped the teardown for good and a hung App stop held it forever. `close()` now runs the teardown as one task awaited to its end through repeated cancels (verifiers `run_shielded`), then re-raises the cancel; Apps stop in reverse start order within `APP_STOP_S` (150 s) each, the backend's teardown runs in a `finally` within `TEARDOWN_S` (600 s), and `closed` holds once all of it finished; every close() waits for the same teardown (M1 audit 2026-10-14 R-9). Since the 2026-10-20 merge into dev the shielded, bounded close is M10 lab audit R-6's (`close_timeout_s`, recorded in the attempt scope, a teardown error raised to the first caller); the per-App bound and the reverse order stay.
22. **Outages at grading time:** `SqlAssertion` scored any failed query as a fail (a dead container or a transport error included), and since R-1 a confirmed container loss raises `EnvContainerLost`, which the check runner counted as the candidate's crash. A check whose exception is (or wraps) `EnvContainerLost` is now `infra` and the grade is masked as `crucible:env_container_lost`; `SqlAssertion` applies the rule `state_predicates` already used: a schema error (`no such table`, a renamed column) is the policy's failure, any other failed query is infrastructure (M1 audit 2026-10-14 R-11).
23. **Runaway SQL:** the SQLite App's `sql_query` / `sql_execute` ran the policy's statement with no deadline of its own: on the local backend `LocalExec.run_python` runs the snippet in a thread with no timeout, so a recursive CTE ran until the World masked the call and the thread burned CPU after; in a container it ran 600 s. Postgres only killed the `psql` client after 300 s, leaving the query running in the server. Both now stop the statement at `sql_deadline_s` (default 30 s, `SQL_DEADLINE_S`, capped by `App.deadline`): SQLite through a progress handler in the snippet (a write is rolled back), Postgres through `SET statement_timeout`; the policy gets a scored `query_timeout: the statement ran past Ns and was stopped` (M1 audit 2026-10-14 R-12).
24. **State a snapshot misses:** every `audit_world` check started from reset, so an App whose state lives partly outside its adapter (an in-memory counter reset in `after_restore`) passed them all, yet diverged on resume, which restores into a fresh World. `runtime.restore_roundtrip` (major) snapshots halfway through the gold run, restores into a fresh World of the same spec and backend, finishes both and requires identical end fingerprints; the first differing tool result is kept as evidence (agentenv's universe round trip; M1 audit 2026-10-14 R-16).
25. **Restores before step 1:** `start()` takes the initial snapshot, `reset()` restored it, and a harness's own reset restored it again: two full restores of every App before the policy's first call (about 31 s per attempt for task04). `World.restore` now skips an App when the World's state was last set to this same WorldRef, no tool call or verifier query ran since, and the App's live fingerprint still matches; anything else restores in full, because a fingerprint cannot see file times, owners or what an App keeps outside its adapter. `World.restored_apps` names what the last restore restored (M1 audit 2026-10-14 R-5).
26. **Unenforced network rules:** the local backend has no network enforcement, yet its `validate()` accepted any `network` and `audit_world` passed `no-network` as if it held (`network_enforced` was read by nothing). The local backend now refuses `allowlist` (a spec naming its hosts expects them enforced) unless `backend_params.accept_unenforced`; `no-network`, every spec's default, stays allowed; and `audit_world` adds `network.enforced`, a named SKIP for any non-public world on a backend without isolation (local, ssh) (M1 audit 2026-10-14 R-13).
27. **Images by tag:** seeds name images by tag (`ahmedmohsin/task01-prims-omp:v1`) and the image identity was checked only when a WorldRef was restored, passing when none was recorded, so a re-pushed or rebuilt tag changed every task unseen. `crucible world pin IMAGE` prints the image's identity (`layers:<sha256>`, the digest of its filesystem layers, which also works for local builds); set as `image_digest` (a sandbox's `SandboxSpec`, or Harbor's `backend_params`), a start in any other image is refused as `crucible:stale_image` (infra, never retried) and the container removed (M1 audit 2026-10-14 R-14).
28. **Swap:** `--memory` without `--memory-swap` lets a container swap as much again as its limit, so a runaway allocation thrashes the host instead of stopping at the limit. Every memory limit Crucible's World sets now caps swap at the same value: sandboxes (`--memory-swap`), Harbor's compose overlay (`memswap_limit`), Harbor directory verifiers, the CI App's runner and RLM kernel containers (M1 audit 2026-10-14 R-26).

## Article cross-check (2026-10-09)

Source: Preference Model, Repo2RLEnv and Prime Intellect, re-checked after M3. See `docs/reviews/2026-10-09-articles-crosscheck.md`.

- **Egress self-test** (`guard.py egress`):
  - **Targets:** 17 answer-source hosts (code hosts, package registries, wheel and archive mirrors), in addition to DNS, cloud metadata, IPv6 and the gateway.
  - **TLS/SNI on port 443:** a TLS handshake carrying the host name, because Harbor's transparent egress sidecar accepts every TCP connect.
  - **Allowlists:** allowlisted hosts are excluded from the deny set and probed as lanes (`allowed_reachable`, a warning if a lane is closed).
  - Red-team suite: 3/3 on local Docker.
- **Audit `network.answer_sources`:**
  - critical failure when an allowlist opens a code host or public package index (serve a date-frozen mirror instead);
  - major failure for `public` worlds.
- **World traces:**
  - every edit call records `data.writes` (M4 credits the agent or peer that wrote the file);
  - every policy tool call is scanned for answer-key probes;
  - `run_episode` masks an episode whose probe actually exposed an answer source (`crucible:answer_exposed`).
  - Tests: `tests/unit/test_probes.py`, including two peer policies sharing a workspace.
- **Open: gold = 1 / no-op = 0 for the seed worlds.**
  - The seed-world gate above tested snapshot and restore only. It never graded the oracle and a no-op through the World.
  - The audit also found grader holes in four of the five seed tasks (DESIGN Appendix B.10).
  - Both need the task graders re-versioned and re-validated on the lab GPU.

## Gate re-verified after the hardening (2026-10-09)
With the choke point (schema validation, deadlines, fault labels), virtual time for tool modules and the new error envelopes in place, the generality gate still holds: **266/266** complete MiMo envs (`reports/m1_mimo_sweep_inprocess.json`) and **100/100** AWM scenarios (`reports/m1_awm_sweep.json`) pass every check. The test suite is `tests/unit` + `tests/toy_e2e` (82) and `tests/redteam` (3 tests, 25+ attacks), and the lab gates are `tests/gpu` (seed worlds, CI on the verifier image, agentenv GPU sandbox).

## Added for environment diversity (2026-10-10; research `docs/research/m11-m12/05` §5.2)
What the OpenEnv / FineEnvs / NeMo-Gym catalogue (simulations, games, vision, budgeted tools) needs from every World.
Tests: `tests/unit/test_world_diversity.py`, `tests/unit/test_openenv_import.py`.
- **Environment-terminated episodes:** `ToolResult.ends_episode` marks the calling actor done (logged
  `environment_terminal`), for environments that end themselves (a final answer, a step budget).
- **Per-tool call budgets:** `LimitsSpec.tool_quota` maps fnmatch patterns over qualified names to calls per episode
  (all policy actors together). It is enforced at the choke point after argument validation; an exhausted budget is a
  policy-fault result (`quota_exhausted`). The counts live in `WorldRef.tool_counts`, are part of the fingerprint, and
  are restored on restore and cleared on reset. The field is `None` by default, so specs without quotas keep their content hash.
- **Sim clock:** `ClockSpec.mode = "sim"`. Time moves only when an App calls `ctx.clock.advance_to` /
  `advance_sim_seconds` (a discrete-event simulator, or an imported environment reporting its time through
  `clock_from`), never per step and never backwards. Time-triggered events fire when the simulation passes their due time.
- **Per-App seeds:** `AppContext.seed = derive_seed(WorldSpec.seed, app id)` (sha256, 63-bit): stable, distinct per App,
  reproducible from the spec hash.
- **Image tool outputs:** `ToolResult.images` (`ImageRef`: mime + content-addressed store key; `App.image(bytes)`).
  - Traces record refs only. `data_b64` rides on the live observation and is excluded from every serialization.
  - `LLMToolPolicy(vision=True)` sends the images as a user turn after the tool batch (OpenAI tool messages carry
    text only); text models get a placeholder line instead.
  - `approx_tokens` counts an image as a fixed 768 tokens.
  - M12 token-in-token-out training is text-only for now (a multimodal processor codec is open).
- **Step overhead:** `World.step` on an in-process App stays under 5 ms on average (tested).

## M1 build item 1 (2026-10-12): a separate verifier World and submission custody

Source: the M1 taxonomy (`docs/research/m1-taxonomy/07-capability-matrix.md`, primitive "separate verifier environment").
Ports Karotte `save_submission.py` and `untrusted_paths.py` (MIT) and the Harbor `ArtifactHandler` manifest (Apache-2.0); see THIRD_PARTY.md.

- **`WorldSpec.verifier`** (`VerifierSpec`):
  - `mode` is `in_world` (default, the previous behaviour) or `separate`;
  - `backend` and `backend_params` select the verifier World's backend or image;
  - `artifacts` and `protected` take `"<app>:<glob>"` patterns over fs/fs_git Apps;
  - Karotte's limits apply: `allow_symlinks`, `max_file_bytes`, `max_total_bytes`, `max_entries`, `max_depth`.
  - A spec without `verifier` hashes exactly as before.
- **Hand-off** (`world/verifier.py`):
  - `hand_off` runs quiesce (a surviving process counts as misbehaviour), then snapshot, then `check_snapshot`, then `collect_artifacts` into `attempt/artifacts/<app>/` with `manifest.json`.
  - `verifier_world` closes the policy's World, starts a fresh one from the same spec, refuses shared absolute state roots, restores, checks every App fingerprint, then copies in the trace and the episode-start ref. `VerifierView.unchanged` and trace checks therefore work as they do in place.
- **Harness** (`runtime/harness/native.py`):
  - With `mode: separate`, the grade phase runs in the verifier World.
  - It writes `verifier/handoff.json`, or `verifier/custody.json` with the problems found.
  - A violation is `crucible:policy_misbehavior` with policy blame: scored 0 and never masked.
- **World**: `close()` is idempotent, and `World.episode_start` records the ref the episode was reset to. That ref is the baseline for protected paths.
- **Tests** (`tests/unit/test_verifier_separate.py`, 13):
  - a clean hand-off graded in a different World, with the policy's World already closed;
  - six attacks (symlink, FIFO, tampered, added and deleted hidden test, nesting depth) plus an oversize file. Each ends as misbehaviour, and the grader never runs;
  - symlinks inside the tree are allowed with the flag; symlinks leaving the tree are refused;
  - a late writer beats the in-place grader but not the separate one;
  - shared roots are refused;
  - spec validation;
  - the Karotte copy rules on host paths.
- **Open:** item 1 is verified on the local backend. A run on a container backend (verifier image without the policy's tools, `network none`) belongs on the lab GPU node with the M1 container suite.

## M1 build item 2 (2026-10-12): single-turn tasks with output parsers

Source: taxonomy family 1 and primitive P1 (single-turn task type), plus the parser half of P16.
Ports NeMo-Gym `mcqa`, `equivalence_rule` and `math_with_judge`, and verifiers v1 `utils/score.py`; see THIRD_PARTY.md.

- **`runtime/single_turn.py`**:
  - `SingleTurnTask(id, prompt, answer, parser, criteria, options, system, judges)` is a `RunTask` with a World of no Apps and one tool-less actor.
  - `completion(world)` reads the full final reply. `Done.answer` keeps it when `reason` is cut at 2000 characters; the World logs it on the `done` event.
  - `single_turn_policy(llm)` sends the system prompt and few-shot turns, exposes no tools and never nudges. `answer_policy(fn)` gives scripted completions.
  - `load_tasks(rows | path, dataset=...)` reads NeMo-Gym, verifiers and plain rows, with row-level `parser` and `criteria`.
- **`verify/answers.py`**:
  - `ParserSpec` kinds: full, boxed, answer_colon, xml, code, json, mcq, regex, last_line. Each takes the `think` handling and `fallback_full`.
  - `CheckerSpec` kinds: exact, exact_raw, numeric, math, mcq, contains, regex, seq_match, weighted_seq_match, json_subset, any_of, judge. Each takes a `threshold`.
  - `Criterion` is one weighted rubric line. `score_answer` runs the rubric. `parse_judge_choice` reads a judge model's verdict.
  - `CheckerUnavailable` masks a sample instead of scoring it 0.
- **Extra:** `crucible[answers]` installs `math-verify`. Without it, the `math` checker masks unless both sides are numbers.
- **Tests** (`tests/unit/test_single_turn.py`):
  - parsers and multiple-choice modes, including ambiguity refusal and letter folding;
  - checkers, judge verdicts, row shapes and validation;
  - end to end, gold scores 1, wrong answers 0 and unformatted replies fail the gate;
  - a 12,000-character reasoning reply keeps its answer;
  - weighted rubric thresholds;
  - a judge outage masks the sample;
  - the LLM policy sees the system prompt and few-shot turns, gets no tools and makes exactly one call;
  - a blank reply is the answer.
- **Open:** a code-execution checker (taxonomy family 2: single-turn code with hidden tests) belongs with the verify trusted runner (M7). Image prompts depend on item 8.

## M1 build item 3 (2026-10-12): enterprise record-world primitives

Source: taxonomy family 6 and primitives P4–P6. Ports EnterpriseOps-Gym `benchmark/` (Apache-2.0).

- **Identity** (`ActorSpec.identity`, `IdentitySpec`):
  - `App.identity(actor)` gives Apps the caller's identity.
  - `common/mcp_server` over HTTP opens one session per identity carrying `x-user-id`, `x-user-email`, `x-user-name`, `x-user-roles` and `x-<header>`. It also accepts static `headers`, and `identity: required` refuses stdio servers.
- **RBAC** (`WorldSpec.rbac: {role: {allow, deny}}`):
  - enforced in `World._call_tool` for actors with an identity;
  - a refused call is `permission_denied` (policy fault), and the tool stays visible;
  - spec validation catches unknown roles, roles without rbac, and patterns over unknown apps.
- **Catalogs** (`ActorSpec.catalog`, `CatalogSpec{oracle, distractors, hide}`):
  - `World._catalogs()` draws distractors with `derive_seed(world seed, "catalog:<actor>")` over every tool;
  - `tools_for` filters to the catalog, and a call outside it is `unknown_tool`;
  - actors without a catalog, and system calls, are unaffected.
- **Seeding** (`common/sqlite_app`):
  - `seed_sql` (one or more data scripts) runs after `schema_sql` and may extend a `seed_db`;
  - `pin_now` overrides `datetime`/`date`/`time`/`julianday`/`unixepoch`/`strftime` so that `now` is the clock's `t0`;
  - the sql adapter's snapshots carry column names (`meta.columns`) for row diffs.
- **Verifier side**:
  - `VerifierView.start` gives the episode-start WorldRef;
  - `VerifierView.changes(app)` diffs the start against a snapshot taken now;
  - `verify/sql_checks.py` provides `SqlAssertion` and `Collateral` (allowances per table with `match` and `ops`; updates paired by `key`). Both become M7 `Check`s, load from task data, and convert from EnterpriseOps-Gym.
- **Tests** (`tests/unit/test_enterprise_primitives.py`, 9):
  - `pin_now`, and byte-identical seeding;
  - catalog modes: oracle, +2, all with hide;
  - RBAC refusals, identity reaching an App, identity validation and headers;
  - identity headers reaching a live HTTP MCP server, with one session per identity;
  - SQL assertions and collateral on a service-desk World: a policy that also closes an unrelated case passes every assertion and fails only the collateral check;
  - all 102 verifiers of the 13 local EnterpriseOps-Gym tasks convert.
- **Open:** an EnterpriseOps-Gym importer (its MCP servers ship as Docker images) is M11, and needs the lab node. The `response_check` (LLM judge over a SQL result) and `tool_execution` verifier types are not ported yet.

## M1 build item 4 (2026-10-12): multi-step tasks with per-step gating

Source: taxonomy primitive "multi-step tasks with per-step gating". Ports Harbor `trial/multi_step.py` and `StepConfig` (Apache-2.0).

- **`TaskStep(name, instruction, grade, min_reward, max_steps, fresh_context, setup)`**: a `RunTask` with `steps` (and `step_reward`: mean | final | mean_executed) runs them in one World.
- **`World.begin_task_step(name, index, instruction)`** opens the next step in the running episode:
  - policy actors leave `done`;
  - a `task_step` trace event marks the boundary;
  - the next observation carries the step's instruction and tools.
- **`run_episode(..., start_obs=...)`** continues an episode from given observations.
- **Harness**:
  - every step is graded under `steps/NN-name/` and its gate is checked;
  - masked segments, policy misbehaviour, limit masks and a broken environment are judged per step;
  - model calls from all policy objects (fresh contexts included) are merged into the attempt's evidence;
  - `verifier_world(close_source=False)` gives intermediate steps a separate verifier while the policy's World stays open.
- **Records**: `meta.task_steps` and `meta.step_reward`, with gates `step/<name>`.
- **Tests** (`tests/unit/test_multistep.py`, 8):
  - three steps with a setup hook carry state and context, with trace markers and per-step reports;
  - a gate stops the remaining steps, under each reward strategy;
  - with no gate, every step runs;
  - dict gates, including a missing key;
  - a fresh context per step;
  - separate verification at every step: three copies, and the World is closed after the last step.
- **Open:**
  - dense progress from step boundaries (each passed step is a natural milestone) goes to M6;
  - importing Harbor multi-step task directories (`steps/<name>/instruction.md`, per-step `tests/`) goes to M11.

## M1 build item 5 (2026-10-12): the per-uid egress firewall

Source: taxonomy primitive "per-user egress firewall". Ports Karotte `confinement.py` (MIT).

- **Guard ops**:
  - `firewall {uid, mode: allowlist|deny|ports, allowed, blocked_ports}` installs iptables/ip6tables OUTPUT owner-match rules. Blocked localhost ports (TCP and UDP) are dropped first. Then come localhost, the sandbox's own addresses (from `/proc/net/fib_trie` and `if_inet6`) and the allowlisted IPs (names resolved at setup; wildcards reported). Everything else is rejected.
  - On gVisor it uses iptables-legacy with DROP and reports `took: false`.
  - `firewall_lift {uid}` removes the uid's rules so that a reused sandbox starts clean.
- **`IsolationSpec.firewall`** (None = auto | off | required) and **`blocked_ports`**:
  - `isolate_world` applies the rules after the policy account exists and before the egress probe;
  - `app_ports(world)` adds the localhost ports of App servers (`url`, `mcp_url`);
  - `DockerSandbox` adds `--cap-add NET_ADMIN` for root when a policy user is set.
- **Tests** (`tests/unit/test_firewall.py`, Docker, 2):
  - in a real container with internet, the policy uid keeps localhost and the allowlisted 1.0.0.1, and loses 1.1.1.1 and an App's blocked localhost port;
  - root is unaffected;
  - the egress probe reports isolated;
  - lifting restores the uid's network;
  - `ports` mode leaves a public world's internet, and `deny` mode keeps only localhost.

## M1 build item 6 (2026-10-12): health checks, reset hooks, run-serially

Source: taxonomy primitive P14. Ports the Harbor healthcheck (Apache-2.0); patterns from TheAgentCompany (`init.sh`, `-j 1`) and OpenEnv `SUPPORTS_CONCURRENT_SESSIONS`.

- **`ProbeSpec`** is a `command` or an `http` probe (method, body, `expect_status`, `timeout_s`). **`HealthcheckSpec`** adds interval, start period, start interval and retries.
- **`AppSpec.healthcheck`** runs after the App starts (spec order, so dependants start after it is healthy) and after every fresh reset. `World.check_health(app, spec)` is public, and `TaskStep.healthcheck = (app, spec)` runs before a step opens.
- **`AppSpec.reset_hook`** runs on every fresh `reset()`, never on `reset(start=ref)`.
- **`HealthcheckError` and `ResetHookError`** map to `crucible:backend_error` (masked) in `runtime/infra.NORMALIZE`.
- **`WorldSpec.serial`**: `SerialLock` (`flock` under `$CRUCIBLE_LOCK_DIR`, default `~/.crucible/locks`) is held from `World.start` to `World.close`, waiting up to an hour. `World.close` releases it even when teardown fails.
- **Tests** (`tests/unit/test_health.py`, 6):
  - the start period absorbs early failures;
  - an unhealthy World does not start and its attempt is masked;
  - HTTP probes, including expected status, an unreachable server and spec validation;
  - reset hooks run on fresh resets only, and a failing hook raises;
  - serial worlds run one after the other while non-serial worlds overlap;
  - the lock holds across processes.

## M1 build item 7 (2026-10-12): native turn order for multi-agent tasks

Source: taxonomy family 11 and primitive P7. Follows OpenSpiel and OpenEnv game semantics (no code copied).

- **`TurnSpec{mode: sequential|simultaneous|app, seats, app, per_turn}`** sits on `WorldSpec.turns`. Validation catches seats that are not policy actors, duplicate seats, app mode without an app, and an unknown app.
- **`TurnManager`**:
  - tracks to-move, legal actions per seat, rewards, terminal state, the round buffer and deferred results, and saves and restores them through `WorldRef.turns`. The fingerprints of non-turn worlds are unchanged.
  - The App protocol is `turn_state()` and an optional `resolve_round(actions)`.
- **World**:
  - the gate in `_step` refuses moves out of turn (`not_your_turn`) and illegal moves (`illegal_action`);
  - `_simultaneous` buffers moves and resolves the round;
  - `_end_if_terminal` ends every seat with `game_over` and the rewards;
  - observations carry `to_move` and `legal`.
- **`run_episode`** asks only the seats that may move, and hands a simultaneous round's deferred results to each seat on its next turn.
- **Harness**: `meta.seat_rewards` in run records.
- **Tests** (`tests/unit/test_turns.py`, 6):
  - sequential seats, including a message counting as a move and Done leaving the rotation;
  - Nim with App rules: legal actions, out-of-turn refusal, a seeded first mover (chance), terminal rewards, snapshot and restore mid-game;
  - a full Nim game through `run_episode`;
  - simultaneous rock-paper-scissors, both through direct steps and a whole episode;
  - seat rewards reach the run record.
- **Open:**
  - per-seat credit for training (one record per seat, M12);
  - imperfect-information observation views beyond per-actor observations (the App filters what each seat sees);
  - an OpenSpiel adapter App (M11).

## M1 build item 8 (2026-10-12): images and audio in messages and instructions

Source: taxonomy observation gaps "images or audio in messages and instructions" (R19). Uses the OpenAI chat content-part formats.

- **`MediaRef{kind: image|audio, mime, ref | url, data_b64 (excluded)}`** appears in `Message.attachments`, `Inbound.attachments` and `Observation.instruction_media`.
- **World**:
  - `put_media(kind, mime, bytes)` stores the bytes, and `hydrate(media)` attaches them to live observations;
  - the `instruction_media` constructor argument and attribute hold the instruction's media;
  - `_send` keeps references only, and the trace logs them;
  - `_inbound_json` leaves `attachments` out of media-free snapshots, so their fingerprints are unchanged.
- **LLM policy**:
  - `media_parts(text, media, vision, audio)` renders the content;
  - `LLMToolPolicy(audio=...)` is a new flag;
  - `approx_tokens` counts `input_audio`, taking the clip length as 16 kHz, 16-bit mono.
- **Single-turn**:
  - `_split_prompt` returns the instruction's media;
  - `_media_of` decodes data URLs and keeps remote URLs;
  - `single_turn_policy(vision=, audio=)` passes the model's capabilities.
- **Tests** (`tests/unit/test_media.py`, 5):
  - instruction media and message attachments reach the policy with their bytes, while the trace and snapshot hold references only;
  - media-free snapshots are unchanged;
  - content parts, placeholders, token estimates and `MediaRef` validation;
  - multimodal single-turn rows in chat, Responses and audio form score through the harness, with no bytes in any written file;
  - a text model is told an image was there.
- **Open:**
  - M12's chat codec is still text only, so training vision or audio models needs a multimodal processor codec;
  - a policy checkpoint (`state_dict`) holds the rendered data URLs of vision turns.

## M1 build item 9 (2026-10-12): a WebSocket client for remote environments

Source: taxonomy row "OpenEnv HTTP/WebSocket client mode absent". The earlier `ClientEnv` wrapped OpenEnv's own client, which needs OpenEnv installed. It had no test, dropped MCP tools, and did not recover from connection loss.

- **`world/remote.py`**:
  - `RemoteEnvClient(url, session_id=, message_timeout_s=, ping_interval_s=, max_message_bytes=100 MB, headers=)` provides `reset`, `step`, `state`, `rpc`, `list_tools`, `call_tool` and `close`;
  - `ws_url()` normalizes http(s) and ws(s) URLs;
  - `RemoteActionError` (policy) and `RemoteEnvError` (infrastructure) separate the blame;
  - requests are serialized per socket, a socket is dropped on timeout or cancel, and `session_lost` is set after a reconnect.
- **`ClientEnv`** (OpenEnv client mode):
  - runs on `RemoteEnvClient`;
  - `docker_image` starts the container itself (port on 127.0.0.1, `/health` wait) and removes it on close;
  - takes `mcp` and `session_id`.
- **`OpenEnvApp`**:
  - client MCP discovery and calls go through `list_tools` / `call_tool` steps, and wire-format errors (dicts) are handled;
  - `_replay(journal)` is shared by restore and connection recovery;
  - a server-refused action becomes `invalid_action` (policy) and is not journaled.
- **Extra:** `crucible[remote]` installs `websockets`.
- **Tests** (`tests/unit/test_remote_ws.py`, 5):
  - URL normalization;
  - protocol errors and the desync guard after a timeout;
  - a World over a remote counter environment: the schema check comes before the server's check, the server refusal counts as policy, restore makes a new replayed session, and a dropped socket mid-episode is recovered;
  - an MCP-style remote environment;
  - a real OpenEnv server (`create_app`) run from `.envs/openenv` and driven from the main venv, which has no OpenEnv installed.

## M1 build item 10 (2026-10-12): continuous control and simulators, both styles

Source: taxonomy families 10 and 12, now expressible. The user decided "both properly": LLM-driven simulators, and Gym-style tasks whose policies are neural networks, with their own trainer.

- **10a `core/schema/spaces.py`**:
  - `Space` covers box, discrete, multi_discrete, multi_binary, dict and tuple;
  - methods: `problem` / `contains` / `clip` / `sample` / `flat_dim` / `flatten` / `json_schema`;
  - readers: `from_gymnasium` and `from_spec_dict`.
- **10a `world/apps/control.py`** (`ControlApp`, template `common/control`):
  - sources are `env: gymnasium:<id>`, `factory` or `client`;
  - time: `frame_skip`, `sim_dt_s`, `max_episode_ticks`, and the sim clock follows the simulator;
  - `seed` (world or int) and `reseed(seed)`;
  - state: the journal with digest-checked replay, `restore: replay|none`, and `journal: eager|lazy`;
  - tools: `step` (schema from the action space), `wait` (`idle_action`), `macros` and `observe`;
  - `act(tool, args)` is the shared entry point.
- **App hook `before_snapshot`**: `World.flush()` calls it before snapshots and before `VerifierView` query, fingerprint and unchanged checks.
- **10b, the LLM path**: tested through the native harness. An `LLMToolPolicy` makes three decisions over 40 simulator ticks, and the `ControlTask` grade is the normalized return.
- **10c `world/gym.py`** (`WorldEnv`, `VectorWorldEnv`, `gym_spaces()`) and **`train/control.py`** (PPO, `evaluate`): the neural-network path.
- **10d `RemoteControlEnv`**: dm_control-style servers over WebSocket. Actions go out as `values` and observations come back as `observations`; spaces and `control_timestep` come from the server's state. `done_is` defaults to truncated, since dm_control ends at a time limit.
- **`runtime/control_task.py`**: `ControlTask(env | factory | client, return_range, success_return, ...)`.
- **Extra:** `crucible[control]` installs Gymnasium.
- **Tests**:
  - `test_control.py` (6): spaces; Pendulum ticks, sim clock and replay restore; CartPole termination, truncation and reseeding; wait and macros; refusal of non-deterministic simulators; the LLM path through the harness;
  - `test_worldenv_ppo.py` (4, one slow): the sync API and trace; vector auto-reset; PPO learning CartPole; PPO learning Pendulum;
  - `test_remote_control.py` (2): a remote dm_control-style simulator in a World, including timestep, time limit and new-session replay; and a neural PD controller over `WorldEnv`.
- **Open:**
  - a real MuJoCo / dm_control run (the wheels are not installed here);
  - CARLA and Unity need GPU hosts, which means the lab;
  - SAC and other off-policy learners;
  - vectorized simulators inside one World (thousands of environments per process), for speed beyond ~3k steps/s;
  - wiring M12 records and checkpoints for control runs.

## M1 build item 11 (2026-10-12): a desktop App for computer use

Source: taxonomy family 10 (computer use), now expressible.

- **`world/apps/desktop.py`** (`DesktopApp`, template `common/desktop`):
  - its own container, built once from an embedded Dockerfile (Debian, Xvfb, xdotool, ImageMagick, python3-tk);
  - params: `screen`, `max_width` (screenshot scale), `files`, `setup`, `settle_s` and `wait_after_action_s`;
  - `reset: reprovision|replay`, as described above;
  - one `computer` tool with 12 actions plus `modifier`; validation of the action enum, coordinates on screen, keys and durations;
  - a screenshot on every result (`ToolResult.images`); the journal records actions with screenshot digests; `inspect` and `screenshot` serve graders.
- **World**: `check_health` uses an App's `health_exec` when it has one.
- **Tests** (`tests/unit/test_desktop.py`, Docker, 4):
  - a Tk app driven by screenshot-space clicks and typing, with files verified inside the container; off-screen and unknown actions refused; screenshots journaled correctly;
  - both reset tiers, the replay tier stable across three runs;
  - a health check inside the desktop;
  - a vision LLM policy receiving every screenshot as an `image_url` part while it clicks Approve.
- **Open:**
  - ~~full VMs and pre-booted pools~~: built 2026-10-11 (`common/vm`, below) and run on the lab's KVM;
  - accessibility trees (AT-SPI);
  - Windows and macOS desktops (cloud providers);
  - Android emulators.

## Full virtual machines with pre-booted pools (2026-10-11; `common/vm`, `world/apps/vm.py`)
- **What:** a Debian 12 cloud guest under QEMU/KVM in its own container (`--device /dev/kvm`, `--network none`; QEMU user networking with `restrict=on` gives the guest a NIC with no route out). SSH for commands, QMP for snapshots. Tools: `exec`, `read_file`, `write_file`; graders use `inspect`.
- **Pre-booted starts:** the first VM on a host cold-boots once and saves `ready` (RAM, devices and disk) into a golden overlay on a shared volume. Every later VM copies it and starts with `-loadvm ready`. `VMPool` keeps N warm and resets each released VM with `loadvm ready` in the background; a VM that fails to reset is replaced, never handed out dirty.
- **World snapshots:** `before_snapshot` → `savevm <tag>` (the tag derives from the journal); `after_restore` → `loadvm <tag>`. This is exact RAM-and-disk branching, so running processes come back. In another container, `restore: replay` re-runs the commands and `restore: start` refuses mid-episode restores.
- **Measured on the lab** (`docs/research/m12-infra/live/vm_check*.json`, driven over `DOCKER_HOST=ssh://`):
  - cold boot to SSH 8.8 s, then `savevm` 0.6 s;
  - a warm VM from the golden state 2.4–2.8 s (SSH 0.5 s after `loadvm`);
  - a reset 1.1–1.4 s;
  - a mid-episode snapshot 0.6–0.7 s and its restore 1.0–1.2 s, with a background process alive again;
  - 8 VMs × 64 concurrent acquire / exec / release cycles: all clean, no file leaked between users.

## Deferred from the survey (with the module that owns them)
- **Distractor tools and per-actor identity / row-level RBAC** (EnterpriseOps `plus_N_tools`, identity headers): task construction, so M8 forge and the finance pack.
- **A cgroup just for the policy** (Karotte `cgroups.py`: `memory.max`, atomic `cgroup.kill`) and **disk quotas** (`file_quota.py`, needs `SYS_ADMIN`). For now, container-wide `pids_limit` / `mem_limit` plus `RLIMIT_NPROC` headroom.
- ~~**Per-uid firewall**~~: built in M1 item 5 (below). Harbor compose environments still need `cap_add: [NET_ADMIN]` in their compose file for `firewall: required`.
- **agentenv driver lifetime:** an external MCP client that connects and disconnects stops the gateway's trigger driver even while Crucible's keepalive session is open. External harnesses must keep their session open for the episode.
- **Harbor and `DOCKER_HOST`:** Harbor's compose subprocesses read `DOCKER_HOST` from the process environment, so one process cannot drive worlds on two different Docker hosts at once (the backend refuses rather than racing).
- **Baked world images** (SWE-smith / Repo2RLEnv): provision once, commit, start from the image, so the 68 MB task04 seed push disappears. M10 runtime.
