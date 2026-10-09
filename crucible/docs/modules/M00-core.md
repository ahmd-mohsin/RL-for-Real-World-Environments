# M0 `core` — contracts

**Status: built; exit gate passed (2026-10-09).** Code: `src/crucible/core/`. Tests: `tests/unit/test_core.py`.

## What it is
The typed contracts every other module is written against. It contains no domain logic and imports no framework. Harbor's task model is checked at the interop boundary (`interop/harbor/task_io.py`), not here: that keeps the core installable without Harbor's dependency tree. This changes ARCHITECTURE decision 4, which had allowed `core.schema` to re-export Harbor models.

| File | Contents |
|---|---|
| `schema/base.py` | `CrucibleModel`: unknown keys rejected, canonical JSON, content hash, YAML/JSON I/O. `VersionedDocument` refuses an unknown major `schema_version` |
| `schema/world.py` | `crucible.world/v1`: Apps, State specs, Actors (fnmatch tool access), Clock (step / wall), Events (`at` / `after` / `every` / `count` / `on_tool` / `repeat`; nl / tool / permission actions), Policies. Cross-reference validation (unknown apps, actors, duplicate ids, time events without a clock, at least one policy actor) |
| `schema/state.py` | `StateRef`, `StateDiff`, `Query`, `QueryResult`, `ClockState`, `WorldRef` (all Apps + clock + pending events + fired counts + ACL rules + inboxes + actor memory + backend state) |
| `schema/actions.py` | `ToolSpec` (MCP field names, OpenAI export), `ToolCall` / `Message` / `Done` (ATIF field names), `Observation`, `StepResult` |
| `schema/trace.py` | `crucible.trace/v1` (non-agent events, actors, outcome, signals) and `Record` (citable span for grounded actors) |
| `schema/signal.py` | Generic expert-signal taxonomy (13 types) |
| `schema/milestone.py` | Milestone checks as adapter Queries + expectations, prerequisite DAG (cycle check) |
| `schema/reward.py` | `reward.json` (a failed gate forces 0) and `steps.jsonl` rows; NeMo-Gym-style `reward_components()` |
| `schema/run.py` | `RunRecord` and failure kinds: NeMo-Gym's shared vocabulary verbatim, plus `crucible:*` incident kinds. Infrastructure failures are masked |
| `schema/task.py` | `crucible.task/v1`, `crucible.card/v1` (provenance + value proof), `HarborTaskSummary` |
| `ids.py` | Harbor-compatible directory hash; tree-manifest hash used for state fingerprints |
| `prov.py` | `Provenance` (created_by, derived_from, params) |
| `registry.py` | Lazy `module:Class` registry (agentenv pattern); plugins register through the `crucible.plugins` entry-point group |
| `config.py` | `crucible.toml` discovery; `env:NAME` secrets (process env, then nearest `.env`); model aliases; GPU hosts; prices; LLM budget |
| `store/` | Content-addressed blobs (sha256, read-only, corruption detected), documents with a SQLite index, tags |

## Exit gate

| Requirement | Result |
|---|---|
| Every schema round-trips and exports a JSON Schema | ✅ 11 documents; JSON-Schema export runs with warnings as errors |
| Validation catches mistakes | ✅ 7 mutation cases rejected: duplicate ids, no policy actor, unknown app in tool pattern, time event without a clock, unknown actor, extra key, wrong major version |
| The 5 seed task directories validate | ✅ Through Harbor's own `TaskConfig` / `Task` / `Packager` (Harbor content hashes recorded). Importing them surfaced **3 real defects** (below) |
| The toy World validates | ✅ `packs/_toy/worlds/helpdesk.yaml` |
| Import-linter contracts declared | ✅ In `pyproject.toml`: core/world/actors never import frameworks, `crucible.interop` or packs; layers core < world < actors |
| `THIRD_PARTY.md` written | ✅ |

## Defects found by importing the seeds through Harbor (fixed or recorded)
1. **`authors` format.** All 5 seed `task.toml` files had `authors = ["…"]`; Harbor requires `[{ name = … }]`. The seeds were not valid Harbor tasks. **Fixed** in `migration_tasks/*/task.toml`.
2. **Silently dropped keys.** Harbor drops unknown keys (pydantic `extra="ignore"`). The importer now lists them (`HarborTaskSummary.ignored_keys`). For task04, `verifier.environment.dockerfile = "environment/Dockerfile.verifier"` is ignored, so **under Harbor the separate verifier would be built from the agent's Dockerfile and would lack the hidden CUDA 11.8 reference**. Recorded for M7/M11; the fix is a prebuilt `docker_image` for the verifier environment.
3. **Network mode.** The seeds set `no-network` only for the agent and verifier phases; the `[environment]` baseline defaults to `public`. The importer now reports both.
