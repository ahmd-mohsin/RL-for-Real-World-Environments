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

## Deferred from the survey (with the module that owns them)
- **Distractor tools and per-actor identity / row-level RBAC** (EnterpriseOps `plus_N_tools`, identity headers): task construction, so M8 forge and the finance pack.
- **A cgroup just for the policy** (Karotte `cgroups.py`: `memory.max`, atomic `cgroup.kill`) and **disk quotas** (`file_quota.py`, needs `SYS_ADMIN`). For now, container-wide `pids_limit` / `mem_limit` plus `RLIMIT_NPROC` headroom.
- **Per-uid firewall** (Karotte `iptables --uid-owner`, needs `NET_ADMIN`) for allowlist worlds where root keeps network; Harbor's allowlist sidecar for the same. Today allowlist worlds rely on the provider's policy and the egress probe.
- **agentenv driver lifetime:** an external MCP client that connects and disconnects stops the gateway's trigger driver even while Crucible's keepalive session is open. External harnesses must keep their session open for the episode.
- **Harbor and `DOCKER_HOST`:** Harbor's compose subprocesses read `DOCKER_HOST` from the process environment, so one process cannot drive worlds on two different Docker hosts at once (the backend refuses rather than racing).
- **Baked world images** (SWE-smith / Repo2RLEnv): provision once, commit, start from the image, so the 68 MB task04 seed push disappears. M10 runtime.
