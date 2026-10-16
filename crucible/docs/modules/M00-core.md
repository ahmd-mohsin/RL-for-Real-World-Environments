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
| `schema/run.py` | `RunRecord` (`crucible.run/v2`: the lock fingerprint and launch number it ran under, task digest, usage, timing) and failure kinds: NeMo-Gym's shared vocabulary verbatim, plus `crucible:*` incident kinds. Infrastructure failures are masked. `RunLock`, `RunLaunch` |
| `schema/usage.py` | `Usage` (one call's tokens, disjoint: uncached input, cache reads, cache writes, output; every provider's shape reads to the same numbers), `UsageTotals` (an attempt's live calls with coverage; replays apart), `FinishReason`, `Timing` (per stage, with the model's share of the agent stage) |
| `schema/task.py` | `crucible.task/v1`, `crucible.card/v1` (provenance + value proof), `HarborTaskSummary` |
| `ids.py` | Harbor-compatible directory hash; tree-manifest hash used for state fingerprints |
| `prov.py` | `Provenance` (created_by, derived_from, params) |
| `registry.py` | Lazy `module:Class` registry (agentenv pattern); plugins register through the `crucible.plugins` entry-point group |
| `config.py` | layered `crucible.toml` (defaults < user `~/.config/crucible` < project, never above its repository < `--set`; or `$CRUCIBLE_CONFIG` alone), operator-only settings a project file cannot override (provider endpoints and credentials, GPU hosts, raising the budget), `Config.sources` / `crucible config`; `env:NAME` secrets (process env, then nearest `.env`); model aliases; GPU hosts; prices; LLM budget |
| `store/` | Content-addressed blobs (sha256, read-only, corruption detected), documents with a SQLite index (versioned migrations), tags moved by compare-and-set on a revision with an append-only history (`tag(name, key, expect=rev)`, `TagConflict`, `tag_info`, `tag_history`). Interfaces `BlobStore` / `DocIndex` (`store/protocols.py`, `Store.backend = "local"`), verified reads (`get_bytes`, `open_blob`, `export`, `verified_path`), a conformance kit any backend passes (`store/conformance.py`) |

## Audit (2026-10-12)

`docs/audit/2026-10-12/M00-core.md` reproduced 18 defects. Fixed here (regression tests in
`tests/unit/test_core_audit.py`; reference hashes pinned in `tests/unit/golden_hashes.json`):
- **Round trip and identity:** no nullable field has a non-`None` default (`IsolationSpec.pids_limit` uses 0 for no limit,
  `CrucibleTask.milestones` uses `""` for none; `None` is still read); canonical JSON lives in `core/canonical.py`
  (finite floats, `-0.0` folded, NFC strings), shared with the interop identity gate; `HASH_SCHEME = 1` in `core/ids.py`.
- **Versions:** `crucible.reward/v1` and `crucible.run/v2` are versioned; `VersionedDocument` explains a newer minor
  version, upgrades older majors through `register_upcaster`, and rejects an unknown major on assignment. The first
  upcaster in use reads `crucible.run/v1` records as v2 (R-1 step 3).
- **Validation:** the world spec rejects allowlists without hosts, hosts without an allowlist, negative time steps,
  malformed `t0`, non-positive tool limits, negative quotas, and events or grants naming unknown apps; `StepRow.phi` is in
  [0, 1]; `RunRecord` rejects contradictory masks and malformed timestamps; `validate_id` (agentenv rules).
- **Store:** staged, fsynced writes under flock, hashed from the copied bytes; corrupt blobs repaired on put; WAL with a
  busy timeout; files of dead writers swept on open; every name and provenance a document is stored under is kept.
- **Registry and config:** plugin load failures are recorded and cited in lookups and `crucible registry`;
  `Config.redacted()` masks secret-shaped values (NeMo-Gym rules) and keeps `env:` references.
- **Failure kinds:** packs register `<pack>:<kind>` with a default blame; failure reasons are bounded at 2,000 characters.

- **Layering (D3):** 18 upward imports had tied 13 packages into one import cycle. Seven modules moved to their layer
  (`core/agent_events.py`, `core/schema/rollout.py`, `workflow/potential.py`, `studio/datasets.py`, `accept/repair.py`,
  `interop/harbor/trajectories.py`, `studio/explorer.py`; the old paths of the first three remain as re-exports), `Attack`
  moved into `forge/schema.py`, and three adapters are reached through pointers. The full layer order is an import-linter
  contract (ARCHITECTURE §1.5); framework imports are forbidden below `interop`. CI: `.github/workflows/crucible.yml`.

Open from the audit: the ranked features in the audit's §4.2.

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
