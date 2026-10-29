# 00 Crucible as it exists today: a map for env-synthesis design

Date 2026-10-29, `main` at `f07ff594`. Read-only survey of the code (`src/crucible/**`, `scripts/`, `packs/`), `DESIGN.md`,
`docs/research/**` and the sibling `agentenv-framework`. Paths are relative to `crucible/` unless they start with
`agentenv-framework/`. Nothing under `data/sealed/**` was opened; held-out tasks appear only as counts.

Status words used throughout:

| word | meaning |
|---|---|
| **live** | implemented and used by a real run (a real task, a real model or a GPU) |
| **built, unused** | implemented and unit-tested, but no real task, pack or run uses it |
| **toy-only** | used only by the `_toy` or `finance_ap` packs (no real task, no training) |
| **partial** | implemented in part, or wired but switched off on the live path |
| **stub** | a shim or a re-export |
| **schema-only** | a declared field or registry entry that no runtime code reads |
| **absent** | no code |

## Executive summary

1. **One live lane.** Everything that produces tasks today is the multi-language **debugging lane**. It takes 2 to 5
   real post-cutoff fix commits of one GitHub repo, reverts them at a head B, and writes a symptom-only bug report. Each
   fix's own fail-to-pass tests become a hidden milestone, and B is the oracle (`src/crucible/factory/generators/debug.py:1-35`,
   `docs/research/swarm-pipeline/07-debug-lane.md:7-13`). The GPU kernel lane was deleted on 2026-10-28.
2. **The World is a repo in a container.** The World layer is broad: 15 core App templates plus 5 interop templates,
   six backends, snapshot and restore, clock, events, actors and RBAC. The live lane uses exactly one App
   (`common/git_repo`), the `local` backend, and a raw `docker run --network none` (`src/crucible/factory/lanes/windows_world.py:698-754`).
   - Simulated people, the clock and events, mail and docs run only in toy packs.
   - Messaging (Slack-like), calendars and CRM are absent.
   - No code derives a task from World state.
3. **The RL framework is deep, but barely trained.** Async rollouts, staleness bounds, MIS and SC+MIS losses, Φ turn
   credit, honesty rewards, masking, path rules and a trace audit all exist. Real gradient steps on the student
   Qwen3.6-35B-A3B: **2**, both on a toy window (Tinker smoke). No RL run on a debug task has ever been read back.
   - Tinker spend on this machine: $1,047.6 sampling, $0.1 training (`data/**/spend.jsonl`, summed).
4. **Task generation is real but expensive and hand-driven.** Mining, gates and measurement are automated and
   executed, so ground truth is never a model opinion. People (and Claude) still do the costly steps:
   - picking repos and fixes;
   - writing or rewriting most bug reports;
   - the pre-measurement checklist;
   - fixing tasks until they reach the band.
5. **Yield is low.** Measured on this machine's runs:
   - 1,676 fix units → 706 real fixes → 51 interacting groups → 19 candidates through every CPU gate → 17 accepted
     candidates (`07-debug-lane/funnel_table.md`). **0 of those 17 passed the keep rule as built**
     (`07-debug-lane/fixing/README.md:9`).
   - After about a week of manual fixing and new mining by three people there are about 13 kept training tasks and
     5 benchmark tasks.
   - Cost per kept task: about $30–130 of Tinker plus hours of human time.
6. **The generic contracts are coding-shaped.** `LongTask.grade_patch(patch)` (`src/crucible/factory/longtask.py:72`), path
   rules, the language layer, mutants-as-commit-reverts and GitHub scouts all assume a git repo with a test suite. The
   machine, keep rule, flake bounds, audit taxonomy, provenance and family rules are domain-neutral and reusable.

---

## 1. The World: how dense is it really?

### 1.1 The runtime core

| Piece | Status | Evidence |
|---|---|---|
| `World`: start, reset, step, snapshot, restore, verifier view | live | `src/crucible/world/world.py` (1,212 lines) |
| `crucible.world/v1` schema | live | see below |
| Declarative actor fields (`persona`, `grounded_on`, `policy_doc`, `transport`, `WorldSpec.policies`) | schema-only | see below |
| Registry kinds `actor` / `transport` | schema-only | see below |
| Tool choke point `_call_tool` | live | see below |
| Responder loop in `World.step` | toy-only | see below |

**`World`** (`src/crucible/world/world.py`):
- **Start** (`:294-329`): provision the backend, seed the bundles, build the Apps from the registry, then take an
  initial snapshot.
- **Reset** (`:633-670`) restores that snapshot.
- **Snapshot** (`:1069-1091`) is a `WorldRef` covering every App's `StateRef`, the clock, pending events, actor state,
  ACL, inboxes and turns.
- **Restore** (`:1104-1159`) checks the spec hash and fingerprints, and skips Apps that did not change.

**The `crucible.world/v1` schema** (`src/crucible/core/schema/world.py`, 543 lines; exported to
`schemas/crucible.world/v1.*.json`). It declares:
- Apps;
- actors of kind policy, simulated, peer, human or system (`:19`);
- identity and RBAC (`:114-143`) and catalogs (`:145`);
- the clock (`:175`) and events (`:213-309`);
- isolation (`:342`), turns (`:379`) and verifier custody (`:390`);
- cross-reference validation (`:455-527`).

**The declarative actor fields** (`schema/world.py:157-169`, `:312`). The World runtime never reads them. Actors are
built in Python and attached with `World.attach` (`world.py:446`); for example
`packs/_toy/src/crucible_pack_toy/smoke.py:82-84`. `persona` is a constructor argument of `GroundedResponder` and
`SimulatedActor` (`src/crucible/actors/grounded.py:47`), not something a spec sets.

**The `actor` and `transport` registry kinds.** They are registered (`src/crucible/core/registry.py:331-338`), but no
code resolves either kind from the registry. A spec cannot instantiate a simulated person.

**`_call_tool`, the one tool choke point** (`world.py:505-610`). It applies RBAC (`:485-494`), catalogs, quotas, the
default interceptors (`:203-212`: remote code fetch and grader paths) and output bounding. It is live on the debugging
lane.

**The responder loop in `World.step`** (`world.py:875-916`). At most 8 inner turns. Messages to policy actors are
scrubbed.

### 1.2 Component by component

App templates are registered at `src/crucible/core/registry.py:300-322`.

| Component | Status | What exists | Evidence |
|---|---|---|---|
| **Database: SQLite** (`common/sqlite_app`, `sql` adapter) | toy-only | Domain tools come from a pack `tools_module`, as plain functions loaded in-process or over MCP, or as generic `sql_query`/`sql_execute`. `now` is pinned for seeding. Tool modules need a local exec. | `world/apps/sqlite_app.py` (205 lines), `:117-118`; `world/state/sql.py`; used by `packs/_toy/worlds/helpdesk.yaml:8-12` and `packs/finance_ap/src/crucible_pack_finance_ap/world.py:393-397` |
| **Database: Postgres** (`common/postgres_app`) | built, unused | A `postgres:16-alpine` container, with a dump-based adapter. 1 docker-marked test. | `world/apps/postgres_app.py`, `world/state/postgres.py`; `tests/unit/test_postgres_app.py` |
| **Website / web app** (`common/website`) | built, unused | Runs a pack's ASGI factory under uvicorn. The only site is a 67-line toy portal, in one slow test. | `world/apps/website.py`, `website_runner.py`; `packs/_toy/data/portal/site.py`; `tests/unit/test_website_browser.py` |
| **Browser** (`common/browser`) | built, unused | Playwright MCP. State is URL, cookies and localStorage. | `world/apps/browser.py` (136 lines) |
| **REST resource store** (`common/json_api`) | built, unused | One JSON document. | `world/apps/json_api.py` |
| **Imported web services** (`openenv/awm_service`, `openenv/env`) | partial | Importers for AWM FastAPI services and OpenEnv envs. No AWM worlds are in the repo. | `interop/openenv/import_env.py`, `awm.py`, `_vendor/openenv_awm/` |
| **People: grounded responder** | toy-only | Answers from records: verbatim, quote or LLM. Grounding checks cover numbers, overlap and polarity. | `actors/grounded.py`, `actors/grounding.py`; toy smoke and `m2_gate.py`; `packs/coding/.../grounding_gate.py` |
| **People: scenario persona** (`SimulatedActor`) | toy-only | A NeMo-Gym-style persona with a consistency judge. | `actors/simulated.py` (262 lines); only `packs/_toy/.../m2_gate.py` |
| **People: human / peer** over file_queue or A2A | built, unused | | `actors/human.py`, `actors/transports.py`, `interop/agentenv/a2a.py` |
| **People: fidelity audit** | toy-only | | `actors/fidelity.py`; toy `m2_gate`, coding `grounding_gate` |
| **Time: clock** (step / wall / sim), scheduler, access windows | toy-only | | `world/clock.py:51,131,247` |
| **Triggers** (seeded Poisson arrivals, predicates) and **events** (at, after, every, on_tool, state, after_event → nl / tool / permission) | toy-only | The only real event worlds are the toy helpdesk (3 events) and toy longhorizon. finance_ap has no clock or events. | `core/triggers.py` (ported from agentenv's gateway); `schema/world.py:213-309`; `packs/_toy/worlds/helpdesk.yaml:30-34`; `finance_ap/world.py:390-401` |
| **File system** (`common/workspace`, `common/git_repo`, `fs` / `fs_git` adapters) | **live** | list, read, write and edit files; search; shell; git status and diff. Snapshot is tar.gz plus a manifest; restore checks the fingerprint. | `world/apps/workspace.py` (176 lines), `world/state/fs.py:149-177` |
| **Terminal / shell** | **live** | The shell tool runs through `policy_exec`. On the lane that is a `DockerExec` into the task container. | `world/exec.py` (LocalExec, DockerExec); `windows_world.py:753` |
| **Toolchain** (`common/toolchain`, stateless `process` adapter) | partial | Its only pack use was the deleted GPU lane. | `world/apps/toolchain.py`, `world/state/process.py:35-39` |
| **CUA / desktop** (`common/desktop`: Xvfb, xdotool, one `computer` tool) | built, unused | Reset by reprovision or replay. 1 docker test. | `world/apps/desktop.py` (259 lines) |
| **VM** (`common/vm`: QEMU/KVM, golden `loadvm`) | built, unused | The real test is skipped without `CRUCIBLE_VM_DOCKER_HOST`. | `world/apps/vm.py` (379 lines), `tests/unit/test_vm_app.py:23` |
| **Messaging (Slack-like)** | **absent** | Only the World's built-in actor inboxes and `Message` actions exist. The `chat_messages` connector ingests transcripts; it is not an App. | `world.py:860-874`, `connectors/chat_messages.py` |
| **Docs (Notion-like)** (`common/doc_library`) | toy-only, partial | A folder of files: list, read (renders docx, xlsx, pptx, pdf), search, optional write. It has no page or block model and no comments. | `world/apps/doc_library.py` (133 lines); toy and finance_ap kb |
| **Email** (`common/mailbox`) | toy-only | SQLite-backed, with virtual-clock timestamps; system delivery bypasses the ACL. | `world/apps/mailbox.py` (131 lines); only `helpdesk.yaml:18-24` |
| **Calendar** | **absent** | Mentioned only in a docstring. | `world/apps/mcp_server.py:3` |
| **CRM** | **absent** | | — |
| **ERP** | toy-only | No ERP App. An ERP is a pack `tools_module` over `common/sqlite_app`: finance_ap's 130-line AP tools. The MiMo and EnterpriseOps-Gym importers map their apps onto sqlite. They have tests but no imported data. | `packs/finance_ap/.../ap_tools.py`; `interop/mimo/importer.py`; `interop/eog/importer.py` |
| **MCP servers** (`common/mcp_server`: any stdio or HTTP server, per-identity sessions) | built, unused | No pack mounts one. | `world/apps/mcp_server.py` (107 lines), `world/packaging/session.py` |
| **agentenv MCP envs** (`agentenv/mcp_env`) | built, unused | Snapshot is `data/get`; restore is `data/reset` + `data/add`. | `interop/agentenv/mcp_env.py:56,103-104` |
| **CI runner** (`common/ci`: hidden checks in a separate runner) | built, unused | | `world/apps/ci.py` (252 lines) |
| **Control** (`common/control`: Gymnasium or a WebSocket sim) | live in M12 only | CartPole and Pendulum PPO through `train/control.py`. No enterprise use. | `world/apps/control.py`, `world/gym.py` |
| **Tool registry** | live | `ToolSpec` has app, name, description, JSON-schema input and `read_only`, exported as `<app>__<tool>`. The registry has 12 kinds; packs register through the `crucible.plugins` entry-point group. | `core/schema/actions.py:30-46`; `core/registry.py:27-40,300-345`; `pyproject.toml` |

**Density in one line:**
- 20 App templates in all;
- 4 used by any pack world (sqlite_app, doc_library, mailbox, workspace / git_repo);
- **1 used by the live lane** (git_repo);
- 3 absent (messaging, calendar, CRM).

### 1.3 How tools and data sources are added

**A domain tool on a database** (the finance_ap pattern):
1. Write plain Python functions with docstrings and type hints. The vendored MiMo loader turns the docstrings and
   signatures into `ToolSpec`s (`world/packaging/mcp_tools.py:121-150`; example `packs/finance_ap/.../ap_tools.py:24-40`).
2. Write `schema_sql` and `seed_sql`.
3. Declare an App with `template: common/sqlite_app`, `state: {adapter: sql}` and `params: {tools_module, transport}`,
   and ship the files in `backend_params.bundles`.

**A new App type:**
1. Subclass `App` (`world/apps/base.py:103`). Implement `list_tools()` and `tool_<name>` coroutines. Raise
   `ArgumentError`, `PolicyToolError` or `ToolError` so faults are blamed correctly (`:49-68`).
2. Pick or write a `StateAdapter` (`world/state/base.py`: fingerprint, snapshot, restore, diff, query).
3. Register it from a pack's `register(registry)` as `("app_template", "<pack>/<name>", "module:Class")`
   (e.g. `packs/coding/src/crucible_pack_coding/__init__.py:26-27`).
4. Scaffold with `crucible pack new` from `packs/_template` and check with `crucible pack check`
   (`src/crucible/studio/pack.py`).

**An existing MCP server.** Mount it with `common/mcp_server` (`command` or `url`, `allow_tools`, per-identity headers).
No code is needed. No pack does this yet.

**A new data source (the ingest path).** Implement the `Connector` protocol (discover, fetch, normalize, redact;
`src/crucible/connectors/base.py`). Register it under kind `connector` (core connectors at `core/registry.py:340-345`:
file_drop, sql_audit_log, otel_import, capture_proxy, chat_messages, atif; the coding pack adds github_pr, agent_runs,
claude_code_sessions, ci_logs and policy_docs). Then declare it in `pack.yaml` `ingest:` (`IngestSpec`,
`core/schema/pack.py:50-88`).
- **Connectors produce traces** (`crucible.trace/v1`, DESIGN §4.2) for signals, milestones and credit.
- **They do not populate App state.** Seeding a World from an enterprise export is manual: you write the SQL or seed
  files yourself.

### 1.4 Build, snapshot, restore, isolate

**Build.** `World(spec)` → backend from the registry (`world.py:226-229`) → `provision` → `_seed_bundles` (`:331-368`) →
each App's provision, start and healthcheck → `list_tools` → initial snapshot.

**Snapshot per adapter.** Each adapter snapshots its own state:

| adapter | snapshot |
|---|---|
| `fs` | tar.gz |
| `sql` | database copy |
| `postgres` | dump |
| `json_doc` | the document |
| `browser` | URL, cookies and localStorage |
| desktop / VM | journal replay, or QMP savevm/loadvm |
| `process` | none (stateless) |

The World adds the clock, the events, the ACL, inboxes, actor `state_dict`s and turns (`world.py:1069-1091`).

**Backends.**

| backend | real use |
|---|---|
| `local` (`world/backends/local.py`, no isolation) | the live lane and the toy packs |
| `apptainer` | Marlowe smoke scripts |
| `harbor` (`interop/harbor/world_backend.py`) | the deleted GPU lane |
| `agentenv` (`interop/agentenv/world_backend.py`: the real agentenv gateway as a subprocess, `:543`) | tests only |
| `container_exec` | none |
| `ssh_exec` | tests only |

**Isolation on the live lane is ad hoc.** `guard/isolate` (policy uid, git scrub, firewall, quiesce, egress probe) is
called only by the harbor and agentenv backends (`interop/harbor/world_backend.py:418-421`,
`interop/agentenv/world_backend.py:465-468`). The local backend's quiesce is a no-op (`world/backends/base.py:107-110`).
The lane instead uses:
- `docker run --network none --pids-limit 512 --memory --cpus` (`windows_world.py:744-746`);
- an offline check that reads the network mode back with `docker inspect` (`:712-722`);
- grading in a separate `--network none` sandbox, not a World.

`SandboxPool`, custody and the separate verifier (`world/custody.py`, `world/verifier.py`) are built but unused.
`world/lockdown.py` is used once (`scripts/delta_rl.py:183-190`).

**Mid-episode state on the live lane bypasses World snapshots.** Φ checkpoints use a git private-index diff through
`DockerExec` (`factory/lanes/windows.py:2075-2097`). Per-language setup is re-run after every reset by monkey-patching
`world.reset` (`factory/generators/debug.py:254-266`).

### 1.5 How tasks are "extracted" from the World today

**They are not.** No code derives a task from World state, and none builds a World from ingested enterprise data in
the general case.

| Source of tasks or Worlds | What it does | Evidence |
|---|---|---|
| The debugging lane | Derives tasks from **git history**: reverts real fixes; the tests are the checks. The World is only the sandbox, built ad hoc. Authored task records hold no WorldSpec. | `windows_world.py:698-707`; `data/debug/authored/*.json` keys: prompt, requirements, spans, checkset, findings, repro, provenance |
| `forge/fuse.py:31` `fuse_worlds` | Composes two existing specs | |
| Toy `forge/world_synth.py` | Writes a helpdesk variant from a fixed template | |
| finance_ap generator | Seeded Python that emits a spec plus SQL. The data is substantive: 7-14 invoices, tolerances, aliases, near misses. | `finance_ap/world.py:1-50,380-408` |
| `packs/_toy/.../forge/hindsight.py` | The only traces-to-task generator (G-hindsight), on toy ticket logs | |
| `factory/failures.py:198-279` | Turns failing student traces into `BuilderIntent`s with status `awaiting_human` (`:214`). No generator consumes them. | |

### 1.6 DESIGN §4.4's AP-invoice World: real vs aspirational

**The example is aspirational.** DESIGN.md:274-291 shows four Apps, four actors, a clock and an event. Validating it
against the shipped `WorldSpec` gives 5 errors: extra `state.engine` fields, a string `grounded_on`, and a missing event
`id`. The templates `finance_ap/erp` and `finance_ap/vendor_portal` do not exist.

**What is real** is `finance_ap/invoices`:
- an ERP sqlite_app and a kb doc_library;
- one policy actor;
- no mail, portal, simulated vendor or approver, clock or events (`finance_ap/world.py:390-401`).

The closest assembled multi-app World is the toy helpdesk: tickets, kb, mail, two grounded customers, a supervisor, an
ops system actor, a step clock and 3 events. Even there the responders are attached in Python.

DESIGN §12.3 ("horizontal extension") is honestly labelled "designed, not built" (DESIGN.md:3775-3782).

### 1.7 What agentenv contributes to the World

agentenv is an optional extra, an editable path dependency (`pyproject.toml:26,112-113`). An import-linter contract
forbids it below `interop`, `studio` and `factory` (`pyproject.toml:157-165`). CI does not install it.

**Used:**
- the gateway as a World backend (role-based tool access, clock, triggers);
- the A2A client and agent;
- the protocol data plane in `mcp_env`;
- sandbox providers (local, modal, modal_vm, e2b, sail_vm);
- the Task / Eval / Artifact classes, for export (`factory/publish.py:64-126`);
- the explorer UI.

**Not used,** although ARCHITECTURE.md:225-227 says "DEPEND": `ServiceDBEnv`, `MultiEnv`, `WebsiteEnv`, `snapshot_env`,
`reset_env`, env-state providers, its verifiers and its bundle runner.

**agentenv has no simulated enterprise apps and no user simulator.** What it ships:
- a server SDK (`agentenv-framework/packages/agentenv-protocol/.../agent_env_environment.py:229`), whose
  `@tool` / `@reset_data` / `@add_data` / `@get_data` give any app MCP tools plus snapshot and restore;
- a Postgres + postgres-mcp env (`agentenv-framework/src/agent_env/env/envs/service_db.py:13-29`);
- a Playwright browser MCP;
- the gateway.

**The live lane does not exercise agentenv.** `RowTask.BACKENDS = ("local",)` (`factory/envs.py:677`), so
`export_agentenv` refuses every row (`envs.py:698-716`).

**Crucible also depends on the sibling `RL-Environments` checkout.** Harbor and verifiers come from it as editable path
sources (`pyproject.toml:111,114`).

---

## 2. Part 1: the RL framework

### 2.1 What has actually trained

| Run | Steps | Evidence |
|---|---|---|
| Tinker smoke: Qwen3.6-35B-A3B, LoRA r32, group 4, toy 3-milestone window, 2026-10-20 | **2** (rewards [1,0,.17,0] then [.17,.17,0,1]; KL 0.0; $0.089) | `docs/research/swarm-pipeline/09-tinker-bridge.md:261-280`; `runs/tinker_smoke.json` |
| DeltaAI prime-rl smoke: Qwen3-0.6B, synthetic word reversal | 3, including a resume; step 3 had zero gradient | `swarm-pipeline/14-deltaai-trainer.md:298-317` |
| Modal M12 torch trainer: Qwen3-0.6B on 4×A10G | 20 (held-out success 0.44 → 1.00); 1.7B 9 steps (already solved) | `docs/research/m12-infra/02-live-validation.md:24-41` |
| 35B verification job on DeltaAI (go-toml, SC+MIS, routing replay) | submitted; **outcome never read** | `14-deltaai-trainer.md:674-677` |
| 8×H100 (Marlowe smoke8), the 40-step 35B run | queued, never run / still a plan | `swarm-pipeline/05-marlowe.md:106-108`; `runs/handoff-deltaai-trainer.md:76-78` |

About 790 student rollouts in `data/targeting/` measured difficulty. None took a gradient step.

### 2.2 Component status

| Component | Where | Status |
|---|---|---|
| Async control loop (torch runtime; topologies colocated, partial, disaggregated stream, elastic) | `train/runtime.py:17-29,235`, `train/scheduler.py:157`, `train/pool.py:74` | live on Modal (M12) |
| Async on the Tinker / prime-rl loop (the student path) | `train/bridges/tinker/trainer.py:156-157,197-198` | partial: defaults are synchronous (`--max-staleness 0`, `--async-groups 0`, `scripts/tinker_rl.py:429-430`) |
| Staleness bound | torch: τ=4, wait/drop (`scheduler.py:72-91`). Tinker: `_stale` and `_cancel_doomed` (`tinker/trainer.py:528-562`) | torch live; Tinker on the path at τ=0 |
| Weight sync | `train/weights.py:25,61`; NCCL buckets `train/weightsync.py:170`; Tinker `save_for_sampler`; prime-rl LoRA broadcast | live (Modal bitwise; Tinker 2 steps; prime-rl 3 hot-loads) |
| Remote leased rollout workers, env pool | `train/workers.py:80,264`, `train/envpool.py:39` | built, unused on the student path |
| `train/rollout.py` | re-export of `core/schema/rollout` | stub |
| Score centering | Dr.GRPO R − mean, no std (`train/credit.py:88`, `train/objectives.py:136`) | live. Zero-spread groups are dropped on the composed advantage (`tinker/trainer.py:518-525`) |
| Group ops (GRS, GAR, BT, tournament, zero_hacks) | `train/group_ops.py:62-191`, `credit/gar.py:35` | built, unused |
| Mismatch / MIS corrections (11 variants: tis, icepop, mispo, cispo, gspo, …) | `train/mismatch.py:138` | torch live; others tested only |
| SC+MIS loss (prime-rl default), MoE routing replay | `train/bridges/primerl/loss.py:15-40,79`, `primerl/routing.py` | built, **not run live** (`14-deltaai-trainer.md:332-345`) |
| Tinker loss | `importance_sampling` / `ppo` / `cispo` (`tinker/trainer.py:142`) | live (IS only; no MIS on Tinker) |
| Φ milestone potential | checkpoint sidecar `factory/checkpoint.py:140-146,399,621` | live in the smoke. It reimplements `workflow/potential.py:33`; `credit/operators/potential.py` is a 4-line stub |
| Turn credit `A_t = A + η(ΔΦ_t − c)·σ_R/σ_ΔΦ`, η=0.1 | `train/credit.py:52,122`; `credit/compose.py:107,115,177` | on the live path; cancelled for hacked or gate-failed episodes |
| Other credit operators (branch_value, prm, rubric_item, channels, properties, judge, signed) | `credit/operators/*`, `credit/*` | built, unused |
| defect_to_check | `credit/operators/defect_to_check.py:28,36` | partial (pack-only) |
| Predictive-validity gate for Φ weights | `credit/validity.py:133`; `checkpoint.py:128 weights_from_validity` | built, **no caller** |
| Honesty (STATUS.json Brier, GIVE_UP) | `factory/honesty.py:132,156`; `bridges/tinker/datums.py:50-62` | on the live path |
| Format penalty | `bridges/tinker/datums.py:77-131`; default 0 (`tinker/trainer.py:178`) | partial: off. `WorkspaceHarness` never fills `format_errors` |
| Loss masks (only sampled tokens; tool output 0), untrainable-rollout drops, masked-share stop (>50% of last 64) | `train/bridges/common.py:70-125`; `core/schema/rollout.py:172-182`; `tinker/trainer.py:183-184,606-614` | live |
| Training timeouts masked | `LimitPolicy(on_timeout="mask")`, `factory/envs.py:324` | live |
| Path rules / protected paths / forbidden ops (installs, network, history reads) | `factory/pathcheck.py:93,587-635`; `LongTask.path_rules` | live; a hack-severity violation zeroes the reward |
| Protected tests | hidden tests graded in a fresh container; protected test paths in `LangProfile` | live |
| Filter-driver neutralisation (snapshots never trust the agent's tree) | `verify/collect.py:61-92`; `world/state/fs.py:44-53` | live |
| hack_detector L1/L2 | `credit/hack_detector.py:62,131` | partial: L1 runs only in `NativeHarness` (`runtime/harness/native.py:917-925`), **not on the live WorkspaceHarness path** |
| Quarantine | `train/quarantine.py:100,123`; `tinker/trainer.py:407` | on the live path |
| Red-team acceptance (A16) | `accept/checks/redteam.py` | used by the acceptance contract, not by the debug lane |
| RLM harness (M16): ledger, context, tree, kernel snapshots, REPL, stall | `actors/rlm/*`; `runtime/harness/rlm.py:64` | live for factory swarm sessions (`factory/sessions.py:280-284`) and CPU ladders (5,000 calls in a 32k window, `long-horizon-rlm/06-validation.md`). **Never trained.** Open audit items C1, H1-H3 |
| Compaction | `actors/policy.py:114-132,362-377` (trigger 0.8× budget; summary / keep_recent / tiered / hybrid; action mode `context_compact`); RLM `context.py:83` | built; **off in training** (`NativeEnv.policy_kwargs` passes only the tool limit, `factory/envs.py:284-286`). Harness audit R7 open |
| Trace audit | `factory/traceaudit.py:427,634,896` | live in the factory, offline; not in the training loop. See below |
| framework_health (about 20 deterministic detectors: masked_failure, credit_grade_mismatch, path_verdict_drift, trainer_numeric, …) | `factory/framework_health.py:6-29,897-922` | operator tool via `scripts/train_monitor.py:129-181` (exit code 3 on a critical defect); the trainer never reads it |
| Curriculum (online p̂ sampler) | `train/curriculum.py`; `tinker/trainer.py:107,161,410` | opt-in; default round-robin |
| Admission, teachers (OPD), critic (GAE) | `train/admission.py`, `teachers.py`, `critic.py` | built, unused on the student path |

**The trace audit, in detail:**
- **Pipeline.** Deterministic facts, then `first_wrong_step`, then the `RLMAuditor`, then typed proposals
  (`verifier_gap`, `false_fail`, `spec_defect`, `unstated_format`, `world_bug`, `intent_drift`; `:79`).
- **Labels.** Every label must quote the trace (`factory/audit.py:45-59`). The closed cause taxonomy is skill_gap,
  spec_defect, unstated_format, world_bug, verifier_defect, budget, env_flake, abstain (`factory/cards.py:33`).
- **On-intent** is a deterministic rule (`on_intent_default`, `traceaudit.py:1642`). For the debug lane it is
  `failure_skill` plus the dispute rule (`scripts/lh_debug_target.py:839-856,939`).
- **Judges.**

  | preset | auditor | judge |
  |---|---|---|
  | `tinker` | GLM-5.3 | GLM-5.3-Flash |
  | `swarm` | Nemotron-3-Ultra | gpt-oss-120b |

  A Qwen model never audits the Qwen student (`factory/models.py:71-72,122-123,159`).

### 2.3 Trainer bridges

| Bridge | Status |
|---|---|
| Tinker (`train/bridges/tinker/`: backend, sampler, datums, spend meter, fake) | live (2-step smoke; 790 sampling rollouts; bit-exact logprobs on 633 tokens, `09-tinker-bridge.md:281-284`). Resume is tested only against the fake |
| prime-rl v0.10.0 (`train/bridges/primerl/`: SC head, top-k server, routing replay, wire) | live smoke at 0.6B on DeltaAI GH200; 35B probes ran on prime-rl's fake data only (bf16 64K, peak 86.2 GiB) |
| In-house torch trainer (FSDP2, HSDP, HF engine, SPMD) | live on Modal at 4 GPUs; never on 8×H100 |
| vLLM / SGLang serving | vLLM live (DeltaAI 35B TP2 at 262K); SGLang smoke only |
| verl (`verl_bridge.py`, `verl_agent_loop.py`, `verl_plugin.py`), SkyRL, rLLM, TRL, slime | tested against pinned upstream source; **never trained**. slime is not installed (`train/bridges/slime/__init__.py:1-10`) |

### 2.4 Open defects from the two audits

**`docs/research/async-rl-audit.md`.**
- **Fixed** (the `4717ac39`, `9db0857d`, `32ffaca1` and `ce042240` passes, all merged): B1-B9, R2-R7.
- **Open by design:** R1 (overlap sampling with training).

**`docs/research/harness-reward-audit.md`.** B1-B11 and R1-R6 are fixed. Still open:
- **R7: compaction in training.** No `ContextConfig` is passed, and `_compactions_in_row` is not saved in the policy's
  `state_dict` (`actors/policy.py:421,657-662`).
- **RLM C1** (critical if the RLM is ever trained). `repl.py` is pushed once, into a directory the kernel user can
  write (`actors/rlm/kernel.py:407,439-449`).
- **RLM H1-H3, M2, M4, L1-L6.** Examples: `compaction_failed` is still a truncation (`agent.py:219,1539`), and the stall
  rule counts only `tree.step` as progress (`agent.py:1834-1835`).
- **agentenv sandbox pause/resume** re-arms the gateway clock.

**Not in either audit.** Found in this survey, and spot-checked in code:
1. On the live path, the L1 hack rules and the format penalty have no data. `WorkspaceHarness` never calls
   `rule_flags` or `format_errors_by_call`, so `hack/confirmed` is never set.
2. `weights_from_validity` has no caller.
3. The `prefix_break` docstring (`bridges/common.py:13`) says the segment splits; the code drops the whole rollout.
4. `runs/handoff-deltaai-inference.md` reports "2 compactions" in a debug episode, which contradicts "off in training".
   Unresolved.

---

## 3. Part 2: task generation

### 3.1 The pipeline as it runs (the debug lane)

| # | Stage | Entry point | Output | Who |
|---|---|---|---|---|
| 0 | Repo registry and clone | `factory/lanes/windows.py:131` (`DEBUG_REPO_URLS`, 84 repos); held-out draw `:154-212` | `data/debug/src/<owner>__<repo>` | **manual** |
| 0b | Held-out supply | `factory/heldout_supply.py:137-196` (`gh api` search, ranking) | sealed draw | ranking automated, draw manual |
| 1 | Scout (local) | `scripts/lh_debug.py:65` | `data/debug/scout.json` (licence, language, fix units) | automated, $0 |
| 2 | Mine: S1 source / executed / groups, S3_world, S4 oracle | `lh_debug.py:118-133` → `Debug.candidates_for` `generators/debug.py:808`, `fix_check` `:199`, `rebreak.interaction_edges` `generators/rebreak.py:198-218`, `group_fixes` `:237`, `synthesize` `debug.py:963`, `precheck` `:1128`, `build` `:1176`; `windows.validate_window` `lanes/windows.py:837-913` | store `data/debug/gen/store/{train,heldout}/tasks/<id>/window.json` + A′; `funnel.jsonl` | automated (Docker, no model) |
| 2b | Subset (a task from named fixes; `--no-link`, `--hide-tests`, `--b`) | `lh_debug.py:136-151` → `Debug.subset` `debug.py:867-919` | same store | **manual** fix choice (helper `scripts/lh_debug_links.py`) |
| 3 | Gates S6 → S5 → S8 → S3 → S10 (stop at first failure; up to 3 automatic test exclusions per gate) | `lh_debug.py:194-252` → `Debug.gates` `debug.py:1243-1304` → `deterministic_gates` `generators/base.py:394-458` | `data/debug/gates/<id>.json` | automated |
| 4 | Bug report (S7 draft) | swarm `lh_debug.py:460` → `swarm_prompts.write_prompt` `factory/swarm_prompts.py:126`; or a hand-written JSON | `data/debug/prompts/<id>.json` | swarm (Kimi) **or by hand** |
| 5 | Report lint and two-way ledger (S7); S7r repro | `author_debug` `debug.py:538`, `lint_debug` `:519`, `unstated_messages` `:474`; `generators/debug_repro.py` | `data/debug/authored/<id>.json`, `data/debug/repro/` | automated |
| 6 | Intent card | `task_intent` `debug.py:679`, skills `:592-615` | `data/debug/candidates.json`, `intent_cards.json` | automated |
| 6b | Pre-measurement checklist (stale assertions, hints, report vs tests, real link) | `07-debug-lane/fixing/README.md:102-120` | — | **manual** |
| 7 | Measure (S13-S15): student k=8, frontier k=2, at 300 tool calls; auditor and judge label failures | `scripts/lh_debug_target.py` → `factory/tinker.py:251` (the exact RL env, `factory/envs.py:295`); budget `debug.py:118-135` | `data/targeting/<run>/…/report.json`, `spend.jsonl` | automated, paid; launched by hand with `--cap` |
| 8 | Keep rule | `targeting.keep_verdict` `factory/targeting.py:523-580` (`KeepRule` `:486`) | `report.json` `keep` | automated verdict; **sometimes overridden** |
| 9 | Fixing / easing | manual loop (`fixing/README.md:68-100`); automated alternative `scripts/lh_difficulty_pilot.py` → `difficulty.control` `factory/difficulty.py:117`, `revise_debug.DebugReviser` `:129`, `DebugGates` `:213` | variant windows `<parent>-<knob>` | mostly manual |
| 10 | Contamination audit | `factory/benchaudit.py:56-293` (git-grep probes, 13-gram overlap) | `runs/benchmark_contamination.json` | automated |
| 11 | Seal | `lh_debug.py:577` → `factory/benchseal.py:69-153` (held out, 5 gates, audited; hashed manifest, HMAC seeds, power plan); guard `factory/sealguard.py:227-292` + `.githooks/pre-commit` | `data/sealed/benchmark/` | automated checks; run on the lead's word |

**What each gate checks, in code:**
- **S6** (`generators/base.py:409-414`). At least one *executed* `requires` edge, unless `allow_independent`
  (`--no-link`), and the `check_longtask` conformance check passes (`factory/longtask.py:116+`). An edge is executed when
  the milestone fails alone and passes with its prerequisite.
- **S5** (`factory/verifier2.py:734-751`). Clean-World solvable: the oracle scores 1, Φ is 1, and there is no path
  violation (`longtask.py:270-298`). The **null agent** scores 0 with Φ 0, for both the empty patch and a "STATUS.json
  claims everything" patch (`longtask.py:262-267`). The fairness lint flags no check (`factory/fairness.py:215`).
- **S8** (`verifier2.py:754-777`). The oracle and its unit steps score 1, floors score 0, every mutant fails its
  milestone, and the "ten competent people" check passes (`fairness.py:248`). Mutants are **commit reverts**
  (`lanes/windows.py:686-713`), not defect-class operators. In this lane the ≥2-alternatives requirement is set aside as
  `alternatives_pending` (`base.py:422-426`).
- **S3** (`lanes/windows_s3.py:115+`). The image builds, A′'s existing suite runs, the leak probes hold, and 60 reruns
  (half at concurrency 4) stay under a 5% flake bound.
- **S10** (`base.py:435-439`, `factory/flake.py:25-33,125`). 60 oracle grades. The Clopper–Pearson upper bound on false
  negatives must be ≤ 0.05.
- **Not run in the debug lane:** S9 second verifier and S9b alternatives (one smoke, `07-debug-lane.md:474-486`); S11
  review; S12 hackers; S16 horizon; counterfactual, paraphrase and prerequisite probes (`targeting.py:217-456`).

**The keep rule** (`targeting.py:462-580`):

| | condition |
|---|---|
| Training | student full-pass rate in [0.1, 0.6] over 8 attempts; on-intent share of labelled failures ≥ **0.5** (`SKILL_MIN`; was 0.7 until 2026-10-27); ≥ 3 labels, otherwise "unmeasured"; at least one frontier full pass ("fair") |
| Benchmark | rate ≤ **0.5** (`BENCH_HI`; was 0.3) |
| Any task | an open World or spec issue gives `world_bug` / `spec_defect` |

`band_power` (`:500`) is a flag. Some docs and strings still state the old bars: `07-debug-lane.md:16`,
`fixing/README.md:42-45` and `debug.py:713`.

### 3.2 The swarm

**Presets** (`factory/models.py`):

| Preset | Builders | Checkers |
|---|---|---|
| `tinker` (the ModelBook default, `:80-98`) | `claude-code:opus-5.5` in every builder role | DeepSeek-V4.1-Flash as second verifier; GLM-5.3 reviewer and auditor; GLM-5.3-Flash judge and realism; DeepSeek and Nemotron-3-Super hackers |
| `swarm` (`:125-144`, from 2026-10-23, "no Claude") | GLM-5.3: planner, verifier_writer. DeepSeek-V4.1-Flash: oracle_builder, oracle_author. Kimi-K2.6: author, prompt_author, world_expert, mutant_builder | Nemotron-3-Ultra: second_verifier, reviewer, auditor. gpt-oss-120b: realism, judge, hacker_2, tie_break. Nemotron-3-Super: hacker. Qwen3.5-397B and Nemotron: rewriters |

- **Family rules.** These are enforced on roles (`BUILDER_ROLES` and `INDEPENDENT_ROLES`, `:152-163`, which raise
  `SameFamilyError`) and on artifacts (`factory/provenance.py`).
- **Models.**

  | role | model |
  |---|---|
  | student | `Qwen/Qwen3.6-35B-A3B:peft:262144` (`lh_debug_target.py:63`) |
  | frontier (debug lane) | Qwen3.5-397B-A17B, DeepSeek-V4.1-Flash (`:64`) |
  | frontier (`swarm` preset) | Qwen3.5-397B, Nemotron-3-Ultra |

**What the swarm actually does in the debug lane.** It never writes ground truth: the World, oracle, hidden tests and
mutants are mined and executed.
1. **The bug-report writer** (`swarm_prompts.prompt_machine`, `:66-89`).
   - A Kimi M16 RLM session in an offline clone of A′ writes the report (`WRITER_BRIEF`, `:24-41`).
   - Then the deterministic lint runs, then one Nemotron review (it blocks once, then is advisory).
   - Limits: 1-4 sessions of up to 60 turns; $1.5 per window and $10 per run (`lh_debug.py:668-674`).
   - Measured: $3.5 for 10 machine runs (`13-swarm-self-improvement.md:211-215`).
2. **The difficulty controller.**
   - `ControlController` (`factory/difficulty_bridge.py:88`) picks a knob (K1-K6, C1, H1-H3; `factory/difficulty.py:44-62`,
     `factory/calibration.py:31-40`).
   - The swarm rewrites the report, and `DebugGates` re-runs every executed gate.
   - The first pilot (2026-10-23) kept **1 task for $129.7** (`13-swarm-self-improvement.md:184-215`).

**Swarm-built windows** (`factory/generators/swarm.py:638-699`, `scripts/lh_swarm.py`) are the one place a model writes
the oracle and the verifier:
- the states are plan → oracle session → oracle check → oracle plan → isolated verifier session → review → build →
  prompt → gates → second verifier → hackers → accept;
- Python only;
- 5 of 14 were accepted on OpenRouter in P5b (DESIGN.md:1149-1157);
- **never run on the swarm preset**.

**Provenance of stored reports.** `data/debug/prompts/` holds 23 reports stamped `claude-code:opus-5.5` and 32 stamped
Kimi-K2.6. All 17 co-author reports under `07-debug-lane/{fixing,new-tasks}/prompts/` carry the Claude stamp, including
the 3 newest kept tasks (2026-10-28/29), because authors copy the template's field (`fixing/README.md:76-77`). The
`builder` field therefore cannot separate human, Claude and swarm authorship. "The swarm builds tasks" holds only for
the pilot.

### 3.3 Domain-specific vs generic

| Generic (reusable as is) | Coding/debugging-specific |
|---|---|
| The machine: declarative states, budgets, escalation, recovery (`factory/machine.py`); RLM builder sessions (`factory/sessions.py`); hand-offs; `ModelBook` and family rules; provenance ledger | `LongTask.grade_patch(patch)` and `oracle_patch()` (`factory/longtask.py:72-76`): the contract takes a **patch** |
| Keep rule, `cause_mix`, `band_power` (`factory/targeting.py`); flake bounds (`factory/flake.py`); null-agent and clean-World checks (`longtask.py:262-298`) | The language layer `LangProfile` (`factory/langs/base.py:55`; python, golang, node, rust, java): test definitions and hiding, runners, parsers, World recipes, leak probes, changelog stripping (`langs/base.py:338-383`) |
| Audit taxonomy and quote rule (`factory/audit.py`, `factory/cards.py:33`); hash-chained labels (`factory/labels.py`); trace audit's typed proposals | Ground truth from **git**: `fix_check`, `interaction_edges`, commit-revert mutants, B points, `--no-link` |
| Difficulty controller logic and R-Zero reward `1-2|p-½|` (`difficulty.py:117`); curriculum sampler | Knob texts, `SKILLS` and `failure_skill` (`debug.py:592-615,717-755`); `train_monitor` mode detectors |
| Sealing and seal guard structure (`benchseal.py`, `sealguard.py`); eval lease (`factory/sources/lease.py`) | Contamination probes (code lines, test names); sources (`sources/scouts.py`, `sources/github.py`, `heldout_supply.py`): all GitHub |
| `floors.py` (supports World tool-call tasks and state answers); `accept/mutation/actions.py` (drop_step / drop_read on tool-call sequences) | `pathcheck.py` (installs, network, history reads, protected files); the report lint (function names, test names, versions, asserted messages, `debug.py:367-472`) |

**What a non-coding domain needs instead.** Each coding piece has a non-coding counterpart:

| coding piece | non-coding counterpart |
|---|---|
| real fix + its fail-to-pass tests | a real record with an executed outcome |
| the Docker repo | a multi-App World (§1) |
| hidden tests | state checks |
| commit-revert mutants | action-sequence or state operators |
| S3 suite reruns | a World determinism check |
| `pathcheck` | forbidden tool actions |
| the report lint | domain leak rules |
| GitHub scouts | domain sources |

A `grade_state(world_ref)` member would be needed beside `grade_patch`.

### 3.4 Other lanes and generators

| Component | Status |
|---|---|
| `generators/debug.py`, `debug_repro.py` | **live** (the only live lane) |
| `generators/rebreak.py` | base of debug; one standalone Python run (P5a: 1 window, DESIGN.md:1144) |
| `generators/stitch.py`, `open.py` | ran once in P5a (5 stitched, 2 open; DESIGN.md:1142-1146); unused since |
| `generators/swarm.py` | P5b only (above) |
| `lanes/executable.py` + `factory/driver.py` + `factory/root.py` (M17 machine, `scripts/factory_run.py`) | first run: 210 validated single-commit tasks for $0.79; "family acceptance has not run yet" (DESIGN.md:680-687) |
| `lanes/windows*.py` (the Python window lane) | infrastructure for debug; produced 26 train and 10 held-out windows earlier (`swarm-pipeline/00-task-factory-workflow.md:90`) |
| `accept/**` (Acceptance Contract A1-A16, defect registry, mutation) | used by `lanes/executable.py:1786-1793` and the studio CLI; **not by the debug lane**. `accept/mutation/__init__.py` mentions patch operators that do not exist |
| `forge/**` (M8/M13 Task Forge) | partly reused: `forge/budget.py`, `learnability.py`, `difficulty.py`, `dedup.py`, hacker, judge, economy. `perturb`, `fuse`, `stitch`, `tighten`, `curator`, `swarm`, `compose` and `ladder` are imported only by tests |
| `factory/family.py`, `ortho.py`, `transfer.py`, `goldset.py` | built; reached only from `factory_run.py` / `scripts/lh_p2.py` |
| `bench/irt.py` (2PL), `bench/longctx/*` | used by the M11 benchmark runner, not the factory |
| `packs/_toy` (helpdesk), `packs/finance_ap` (AP invoices) | toy-only non-coding generators (seeded templates) |
| GPU kernel debug and migration lane | **deleted 2026-10-28** (DESIGN.md:1393) |

### 3.5 Quality mechanisms

**Diversity.**
- **In the lane:** a fix joins at most one group; at most 4 groups per repo (`debug.py:780,839-847`);
  `shares_fixes_with` marks variants (`lh_debug.py:380-386`); the held-out split is disjoint by owner and org
  (`sources/lease.py:42-66`).
- **Built but not wired into debug:** the 40% herding guard (`factory/root.py:29-31`), 4-layer orthogonality
  (`factory/ortho.py`) and `forge/dedup.py`.
- **Result:** the kept tasks cluster on a few repos (mvdan-sh, carbon, black, excelize).

**Difficulty calibration.**
- Measured, not predicted: the k=8 band at 300 calls.
- A predictor exists but is advisory only (LOO MAE 0.204 vs 0.257 baseline, `12-difficulty.md:144-148`).
- Checkpoint-start variants (`difficulty.py:237-296`).
- Measured training exemplars go into the builder brief (`factory/calibration.py`).
- **The pilot's lesson:** the checkpoint estimate predicted 0.5, and the measured rates were 0 and 0.125
  (`13-swarm-self-improvement.md:217-224`).

**Intent alignment.** `task_intent` derives skills from the executed window. `failure_skill` maps each failure to a
skill, with an on-intent bar of 0.5. `disclosure_check` (`debug.py:646`) catches faults the report never describes. A
human approves intent cards (`failures.py:214`).

**Verifier testing.**
- **Built and run in the debug lane:** null agent, oracle, floors, commit-revert mutants, flake at S3 and S10, and the
  executed-edge check.
- **Built, smoke-only here:** second verifier (shadow tests), differential and alternatives (`verifier2.py`,
  `differential.py`, `alternatives_gate.py`).
- **Not used here:** defect-class mutation (`accept/mutation`).

**Traces to tasks.** Failing traces become `BuilderIntent`s awaiting a human (`factory/failures.py:198-279`), and
`train_monitor` intents feed `diagnose`. **No generator consumes them.**

**Self-improvement loops.** All of them *propose*; none applies its own fix.

| Loop | What it does | Evidence |
|---|---|---|
| `repath.scan` | Re-applies the current path rules to stored episodes | `factory/repath.py:115`; 321 attempts, 17 flips, `07-debug-lane.md` §6.3 |
| `framework_health` | Reports ranked framework defects | `factory/framework_health.py:897` |
| `train_monitor` levers | Proposals marked `applied: False` | |
| `swarm_journal.diagnose` | Proposals with `applied: false` | `factory/swarm_journal.py:341` |
| `traceaudit` | Typed proposals with executed acceptance and versioning | `factory/versioning.py` |
| The fixing loop | Humans change tasks and commit pipeline fixes (changelog strip, leak probe, OS-only Go packages) | `07-debug-lane/new-tasks/README.md:57-73` |

---

## 4. The numbers so far

These are aggregates over both splits, from this machine's `data/debug/funnel.jsonl` (9,934 rows, 2026-10-20 to
2026-10-28) and `07-debug-lane/funnel_table.md`, plus the docs. Co-authors' machines are not in the funnel file.

### 4.1 Funnel (mining run of 2026-10-20 plus the held-out draw 3)

**Sources.** 178 clones scouted; 72-75 mined (`07-debug-lane.md:133`).

| stage | go | node | rust | python | java | total |
|---|---|---|---|---|---|---|
| fix units | 594 | 575 | 226 | 188 | 93 | 1,676 |
| real fixes (executed fail-to-pass) | 288 | 242 | 99 | 39 | 38 | 706 |
| interacting groups | 28 | 9 | 11 | 3 | 0 | 51 |
| validated windows | 22 | 9 | 9 | 3 | 0 | 43 |
| dependent + valid mutants | 14 | 5 | 1 | 3 | 0 | 23 |
| all CPU gates | 13 | 2 | 1 | 3 | 0 | 19 |
| report written (S7) | 11 | 4 | 1 | 3 | 0 | 19 |

- **Accepted candidates:** 17, 10 training and 7 held out (`07-debug-lane.md:138-160`).
- **Measured against the keep rule:** **0 of 17 kept** as built (`fixing/README.md:9`).
- **`data/debug/candidates.json` today:** 79 rows, 44 training and 35 held out, over 25 repos.
  - Language: 53 Go, 13 Rust, 9 Node, 4 Python.
  - Level: 48 hard, 8 medium, 23 easy.
  - Gate status: `gates_ok` is true for 37 and false for 17; 25 have not been gated.

### 4.2 Where tasks die

**At mining.** The funnel holds 2,735 failure rows out of 9,934; 2,431 of them are S1_executed. The top reasons:

| rows | reason |
|---|---|
| 946 | the revert no longer applies at B |
| 443 | no own test fails after the revert (not a behavioural fix) |
| 338 | no executed interaction |
| 253 | the World image does not build |
| 192 | the revert only stops tests from building (fixes that add API) |
| 91 + 40 + 24 + 23 + 20 | the suite does not run (various runners) |

At grouping, 99 rows had no interacting group. At S7, 136 rows were refused by the builder as unstateable from symptoms
(`07-debug-lane/prompt_refusals.json`). Gates kill few: S3 12, S3_world 24, S4 9, S6 14, S10 4, S5 2, S8 1 failure
rows.

**At measurement** (the costly kill). This is why 0 of 17 kept, with the fixes made (`fixing/README.md`,
`uzair-fixed-tasks.md`):

| cause | fix |
|---|---|
| a bug no model ever fixed (too hard) | dropped |
| attempts ran out of the 300-call budget | swap a slow bug |
| faults left reverted that the report never described | apply them |
| missing rules in the report | add a sentence |
| fake links (two fixes' cases in one coarse test, or load-dependent timing tests) | `--no-link` |
| changelog hints | strip widened |
| stale visible tests that expected the bugs | `--hide-tests` |
| too easy (7/8) | add bugs |

**At new-repo screening** (2026-10-28/29, `07-debug-lane/new-tasks/README.md:33-55`): about 45 repositories were screened
for 3 kept tasks. Most have no real link between post-cutoff fixes, or only fake ones.

### 4.3 Kept so far

- **Training: about 13.**
  - 7 from the Uzair and Haseeb fixing rounds (`fixing/README.md:13-24`);
  - 2 from Azaan (commit `5aa83dab`);
  - 1 from the swarm pilot (`13-swarm-self-improvement.md:205`);
  - 3 new repos (commit `5d718ae4`).
  - Several are variants of the same repos, and the pilot's black variant may overlap Azaan's black task.
  - Target: 10 training (met).
- **Benchmark: 5**, 4 of them accepted as contaminated after the contamination audit (commit `860fdd60`).
  - Target: about 9.

### 4.4 Cost and time

| Item | Number | Source |
|---|---|---|
| Tinker spend logged on this machine | $1,047.6 sampling, $0.1 training. By model: student 35B $645 (609 episodes), Qwen3.5-397B $190, DeepSeek $136, Nemotron-Ultra $42, GLM $30, gpt-oss $4. $169 of it was the deleted GPU lane | `data/**/spend.jsonl` (summed, excluding `data/sealed`) |
| One student rollout at 300 calls | $1.72, 27.6 min, 208 calls. 397B: $3.01, 13.6 min. DeepSeek: $2.05, 26.8 min | `07-debug-lane.md:408-411` |
| One k=8 measurement (student only) | about $14 | `12-difficulty.md:72` |
| Per task, manual fixing | quick pass $5-10, full measurement $20-35 per try. Haseeb about $100 for 3 kept tasks | `fixing/README.md:87,96`; `fixing/haseeb.md:64` |
| Per kept task, swarm pilot | $129.7 for 1 kept (student $101.6, DeepSeek $15.4, auditor $7.6, judge $1.5, writing $3.5) | `13-swarm-self-improvement.md:207-215` |
| Mining + gating wall time | about 11 + 6 h on 14 shared CPUs for the 17 candidates; S3 and S10 take 20-60 min per candidate | `07-debug-lane.md:222-224` |
| Human throughput | 3 new kept training tasks in about 2 days (about 45 repos screened); 8 tasks fixed by one person in about 1-2 days | `new-tasks/README.md`; `uzair-fixed-tasks.md` |
| CPU-only stages | $0 (mining, World, dependencies, mutants, flake) | `00-task-factory-workflow.md:539` |

**At these rates:**
- 1,000 tasks would cost about $30k-130k of Tinker for measurement alone.
- Mining yield is about 1 accepted candidate per 100 fix units, or about 4 repos.
- Each accepted candidate still needs a fixing loop.

---

## 5. Gaps, ranked

### 5.1 What blocks 500-1,000 diverse tasks

Ranked by how much each one limits scale.

1. **Ground truth comes from one scarce source.** Tasks need several real post-cutoff fixes in one repo that revert
   cleanly *and* interact in an executed test.
   - Yield is about 1% of fix units, and screening found real links in about 3 of 45 repos.
   - `--no-link` raises the yield but drops the "interacting faults" skill the lane claims to train.
   - There is no second ground-truth source in production: stitch, open, swarm-built windows and the executable lane
     are idle.
2. **Measurement is the cost centre, and it runs after the kill.** Every candidate pays about $14-35 per k=8 try, and
   most need 2-4 tries. The predictor is advisory, with MAE 0.20, so nothing cheap filters candidates before Tinker.
3. **The fixing loop is human.** Repo choice, fix choice, report writing, the checklist and keep overrides are manual.
   - The automated controller kept 1 of 5 tasks.
   - People wrote or rewrote the reports; Claude-stamped reports dominate the stored set.
   - At about 1.5 tasks per person-day, 1,000 tasks is years of work.
4. **Diversity guards exist but are not wired in.** Herding, ortho and dedup are not in the debug lane, and the kept
   set clusters on a handful of repos and on Go.
   - No held-out Python or Rust instance exists.
   - Java yields 0.
   - Breadth beyond "multi-fault debugging" is one skill family (`SKILLS`, `debug.py:592-615`).
5. **CPU and host contention.** S3 and S10 take 20-60 minutes per candidate on one 14-CPU host shared with other work,
   and load-dependent flakes kill candidates (axios, lo).
   - No remote or elastic gate workers run in production.
   - The `apptainer` and agentenv sandbox providers (Modal, E2B) exist but the lane uses local Docker.
6. **Verifier depth is thinner than designed.** The second verifier, alternatives (S8's ≥2 is waived), hackers (S12)
   and counterfactual and paraphrase probes are off in the debug lane.
   - Mutants are commit reverts only.
   - At scale, unhackability rests on hidden tests plus path rules.
   - The live harness never runs the L1 hack rules (§2.4 item 1).
7. **Contamination and sealing do not scale.** 4 of 5 benchmark tasks are already exposed. Held-out supply depends on
   fresh post-cutoff fixes from disjoint owners, which are rarer still.
8. **No training signal yet that the tasks teach anything.** No RL run on a debug task has been read back, so the
   factory's keep rule (band, on-intent) is unvalidated as a predictor of learning value.

### 5.2 What blocks a generic, domain-agnostic pipeline

Ranked.

1. **The task contract is patch-shaped.** `LongTask.grade_patch` / `oracle_patch` / `mutants` (`factory/longtask.py:62-86`),
   `verify/collect.py` and `pathcheck` assume a diff against a repo. A state-based contract is needed:
   - grade a `WorldRef`;
   - oracle = an action trace or an end state;
   - mutants = state or action operators.

   `floors.py` and `accept/mutation/actions.py` hint at it.
2. **No World-to-task or data-to-World path.**
   - Connectors ingest traces but never seed App state.
   - World specs are hand-written or come from seeded Python templates (finance_ap, toy).
   - Nothing samples entities, relations or histories into a multi-App World.
   - Nothing derives tasks, checks and oracles from World state or logged workflows.
3. **The enterprise App catalogue is thin and unused.**
   - Messaging, calendar and CRM are absent. Docs are a folder of files. Email, website, browser, desktop, VM, Postgres
     and MCP are built but unused.
   - No real multi-App World exists. The DESIGN §4.4 example does not even validate against the schema.
   - agentenv's server SDK (`@tool` + data plane) is the cheapest way to mint new Apps with snapshot and restore, but
     nothing uses it to author Apps.
4. **Simulated people are not declarative.** `persona`, `grounded_on`, `policy_doc` and `transport` are schema-only; the
   `actor` registry kind is never resolved. Every responder is Python wiring. A generator cannot emit "a vendor who
   disputes the invoice" as data.
5. **The swarm has no world-building role in production.**
   - `world_expert`, `oracle_builder` and `verifier_writer` ran only in Python swarm windows (P5b, OpenRouter), never on
     the swarm preset.
   - The live swarm only writes bug reports.
   - A generic pipeline needs model-built Worlds, oracles and checks, held to executed gates (the "LLM proposes, reality
     decides" rule, DESIGN.md:28).
6. **Intent, skills, knobs and lint are hard-coded per lane.** `SKILLS`, `failure_skill`, the K/C/H knob texts, the
   report lint and the `train_monitor` detectors are debugging content. A generic pipeline needs them as data: an
   intent card, a skill taxonomy and knob definitions per domain.
7. **Isolation for non-repo Worlds is untested in production.** guard/isolate, custody, the separate verifier, the pool
   and the agentenv gateway backend are tested only. The live lane's isolation is a bespoke `docker run`.
8. **The training path cannot use multi-App or long Worlds yet.**
   - Compaction is off in training.
   - The RLM harness is untrained and has open critical items.
   - Async on the student path is off.
   - Simulated-user turns, clocks and events have never been in a training rollout.

### 5.3 Reusable as is for a new framework

| What to reuse | Where |
|---|---|
| Executed gates and their statistics | S5 null agent and clean World, S8 floors and mutants, S10 Clopper–Pearson flake; `factory/longtask.py`, `factory/flake.py` |
| Keep rule and intent share | `factory/targeting.py` |
| Closed failure taxonomy with quoted evidence | `factory/audit.py`, `factory/cards.py` |
| Family rules and provenance | `factory/models.py`, `factory/provenance.py` |
| The machine and RLM builder sessions | `factory/machine.py`, `factory/sessions.py` |
| Sealing and the seal guard | `factory/benchseal.py`, `factory/sealguard.py` |
| World runtime, schema, snapshot and restore, tool choke point | `src/crucible/world/world.py`, `core/schema/world.py` |
| The Tinker loop with Φ turn credit, honesty and masking | `train/bridges/tinker/` |
| framework_health and trace-audit proposals | as the self-diagnosis layer, once something applies them |
