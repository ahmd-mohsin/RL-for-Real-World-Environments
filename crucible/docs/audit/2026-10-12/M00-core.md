# M0 core: audit (2026-10-12)

Scope: `src/crucible/core/` (2,269 lines: schemas, ids, provenance, registry, config, store, triggers) against
ARCHITECTURE.md §M0, `docs/modules/M00-core.md`, and the core layers of agentenv and the local frameworks.
Method: requirement ledger, coverage, import contracts, and one executable probe per suspected defect. Every
finding below was reproduced, not inferred.

## 1. Requirement ledger

| # | Requirement (source) | Status | Evidence |
|---|---|---|---|
| R1 | Every schema round-trips model → JSON → model (ARCH exit gate) | **wrong** | `IsolationSpec.pids_limit=None` reloads as 4096; `CrucibleTask.milestones=None` reloads as the default path (D1) |
| R2 | Canonical JSON gives a stable content hash; equal objects share an id (base.py docstring) | **wrong** | `load(save(x)).content_hash() != x.content_hash()` for D1 fields; NaN serializes as the non-JSON token `NaN` (D2) |
| R3 | JSON Schema export for every document | live | 72 models export; `test_core` |
| R4 | Unknown keys rejected | live | `extra="forbid"` on `CrucibleModel` |
| R5 | Unknown major `schema_version` refused | partial | refused on load; accepted on assignment (`doc.schema_version = "…/v9"`) (D9) |
| R6 | Versioned documents for every persisted format (ARCH: trace, signal, world, state, milestone, task, reward, card, bench, run) | partial | 7 versioned documents; `crucible_reward.json`, `steps.jsonl`, `RunRecord`, signals and bench reports carry no version (D8) |
| R7 | Validation catches spec mistakes | partial | 7 planted cases rejected; 9 further mistakes accepted (D5) |
| R8 | Seed tasks and the toy world validate | live | `test_core` |
| R9 | Import-linter contracts active in CI (ARCH exit gate) | **missing / wrong** | no CI exists; `lint-imports` reports 1 broken contract (actors → train/accept/runtime → interop, Harbor) (D3) |
| R10 | Content-addressed store: immutable, deduplicated, corruption detected | **wrong** | `put_file` stores a blob under a hash that does not match its bytes when the source changes during the copy; concurrent `put_file` of one new file crashes (D4) |
| R11 | Store index lists documents by kind and name | partial | a document stored a second time under a new name keeps only the first name; the second provenance is dropped (D10) |
| R12 | Registry: lazy pointers, duplicate registration fails, plugins via entry points | partial | all 61 builtin pointers resolve; a plugin that fails to load is logged and then reported as "known: none", with no cause (D7) |
| R13 | Secrets only as `env:NAME`; resolved secrets never serialized; `redacted()` shows references (config.py docstring) | **wrong** | `Config.redacted()` returns raw keys and bearer tokens when a file holds them (D6) |
| R14 | `reward.json`: a failed gate forces 0 | live | validator; but NaN and ±inf rewards pass (D2) |
| R15 | Failure vocabulary: every kind has a default blame; infrastructure masked | live | probe: 0 kinds without blame |
| R16 | `RunRecord` states are consistent | partial | `outcome="completed"` with `mask_sample=True` and a reward is accepted; timestamps unchecked (D11) |
| R17 | Milestone DAG: unknown prerequisites and cycles rejected | live | validator; Kahn's algorithm |
| R18 | Seeded event schedules reproducible after restore | live | `triggers.py:180` uses `random.Random(seed)` |
| R19 | `THIRD_PARTY.md` | live | present |

## 2. Defects (reproduced)

| ID | Severity | Defect | Reproduction | Fix |
|---|---|---|---|---|
| D1 | P0 | `exclude_none` in canonical JSON and `dump()` drops an explicit `None` from fields whose default is not `None`. The reloaded value differs (unlimited pids becomes 4096) and the content hash changes across a round trip | probe in this audit | make such fields non-nullable with an explicit sentinel, or serialize explicit `None` for fields with a non-`None` default; add a property test over every model |
| D2 | P0 | NaN and ±inf pass every float field; canonical JSON emits `NaN`, which strict JSON readers reject; a NaN reward poisons every aggregate | `RewardJson(reward=float("nan"))` | `allow_inf_nan=False` on `CrucibleModel`; `json.dumps(allow_nan=False)` in `canonical_json`; `StepRow.phi` in [0, 1] |
| D3 | P0 | Layering broken: `crucible.actors` imports `train`, `accept`, `runtime`, and through them `crucible.interop.harbor` and `harbor`; no CI runs the contracts | `lint-imports` | move the shared pieces down (ATIF types to core, the RLM manager out of actors), declare the full layer order, add a CI job |
| D4 | P0 | Store `put_file` hashes the source, then copies it: a change in between stores wrong bytes under the key. Threads writing the same new key share one temp name and crash | probe (monkeypatched copy; 6 threads) | copy to a unique temp file, hash the copy, then `os.replace`; treat an existing destination as success |
| D5 | P1 | Accepted mistakes: `allowlist` with no hosts; `no-network` with hosts; negative `step_dt_s`; non-timestamp `t0`; negative tool timeout, output cap and quotas; tool events and permission grants naming unknown apps | probe | field and cross-reference validators |
| D6 | P1 | `Config.redacted()` returns raw secrets | probe | redact every `api_key` that is not an `env:` reference and secret-looking header values |
| D7 | P1 | Plugin load failures are swallowed; a later lookup says "known: none" | probe | keep load errors and cite them in `RegistryError` and `crucible registry` |
| D8 | P1 | Persisted formats without `schema_version`: reward documents, step rows, run records, signals, bench reports | probe | version them, with a reader that accepts the unversioned form as v1 |
| D9 | P2 | `schema_version` can be reassigned to an unknown major version | probe | field validator instead of `model_post_init` only |
| D10 | P2 | Store index keeps the first name and provenance of a re-stored document | read `put_doc` | a `names` table (key, name, provenance) |
| D11 | P2 | `RunRecord` accepts contradictory states and malformed timestamps | probe | model validator |

## 3. Coverage and contracts

- `tests/unit/test_core.py` alone covers 78% of `crucible.core`; `schema/pack.py` 0% and `triggers.py` 18% from it
  (full-suite figure in §5).
- Import contracts: 3 kept, 1 broken (D3). No CI configuration exists in the repository.

## 4. Comparison with the local frameworks

A read-only study compared `crucible.core` with agentenv (Apache-2.0), Harbor (Apache-2.0), OpenEnv (BSD-3-Clause),
NeMo-Gym (Apache-2.0), verifiers (MIT) and prime-environments (Apache-2.0). Findings that are defects join §2; the rest
are ranked features.

### 4.1 Additional defects

| ID | Severity | Defect | Reference implementation | Fix |
|---|---|---|---|---|
| D12 | P0 | Canonical JSON does not fold `-0.0` or normalize Unicode: `0.0` and `-0.0`, and NFC and NFD spellings of one id, hash differently. Crucible's own `interop/canonical.py:28-41` already normalizes floats, outside core | Harbor `models/task/step_name.py:29-31` (NFC); agentenv `store/ids.py:50-86` | one canonicalizer in core (finite floats, `-0.0` fold, NFC strings), used by `canonical_json` and interop |
| D13 | P1 | Content hashes include defaults and carry no scheme version, so adding a field with a non-`None` default silently changes every stored id | agentenv `bundle/ledger.py:38-40` (`SCHEME`); Harbor `models/dataset/manifest.py:245-262` | hash with defaults excluded, tag the scheme, and pin golden hashes in a test |
| D14 | P1 | No migrations: a newer minor version fails with a vague extra-key error; old documents strand on any schema change | Harbor `models/task/config.py:503-553`; agentenv `artifact/registry.py:105-124` | per-document upcaster registry run before validation; explicit "newer minor" error |
| D15 | P1 | Store: default rollback journal on one connection (no WAL, no busy timeout); no fsync before `os.replace`; temp files of dead writers never swept; a corrupt blob is never rewritten by a later put | agentenv `store/document_store/sqlite_document_store.py:51-73`, `store/object_store/local/store.py:259-418` | WAL + busy timeout, fsync file and directory, sweep stale temp files on open, rewrite a blob whose size or hash mismatches |
| D16 | P1 | Ids are unchecked strings, and some become path segments (traversal, case collisions, over-long names) | agentenv `store/ids.py:50-109`; Harbor `models/task/config.py:313-321` | a constrained `Id` type (NFC, character set, byte limit, no dot segments) |
| D17 | P2 | Packs cannot add namespaced failure kinds; `Failure.reason` is unbounded; `attempt` is a free string | NeMo-Gym `failure_kinds.py:56-60, 138-170`, `episode_types.py:23-33` | registered `<pack>:<kind>` names with a default blame; bounded reason; typed attempt |
| D18 | P2 | Two unrelated classes named `Provenance` (`prov.py`, `schema/milestone.py:32`); core `manifest_hash` is unused | read | rename the milestone class; remove or use `manifest_hash` |

### 4.2 Ranked features

| Rank | Feature | Reference | Owner |
|---|---|---|---|
| 1 | Packs declare `requires_crucible`; the registry skips an incompatible pack with a recorded reason | agentenv `plugins/_requirements.py:29-56`; prime-environments per-environment pins | M0 registry, M14 |
| 2 | Typed `Usage` (cached and reasoning tokens, cost) and `Timing` (boot, setup, agent, scoring) on run records; `finish_reason` and sampling parameters per call | verifiers `trace.py:53-174`, `types.py:122-133`; ATIF `step.py`, `metrics.py` | M0, M10 |
| 3 | Provenance records the code commit, a dirty flag, pack versions and a config hash; `params` accepts JSON values | verifiers `utils/version.py:11-30`; agentenv `config/provenance.py:17-24` | M0 |
| 4 | Tag history with compare-and-swap and integer revisions | Harbor `models/package/version_ref.py`; agentenv `document_store.py:341, 428` | M0 store |
| 5 | Store protocols (`BlobStore`, `DocIndex`) with remote backends through the registry | agentenv `store/object_store/object_store.py:57-215`, `store/routing.py` | M0 store, M10 |
| 6 | Multimodal `ContentPart` (text, image, audio) on messages and observations | ATIF `content.py:47-121`; verifiers `types.py:10-35` | M0, M1 |
| 7 | Richer pack manifest: judge pin, task-distribution pin, resources and network, SPDX license check, upstream `adopted_from`, checked-in JSON Schemas | OpenEnv `validation/manifest.py:39-242`; NeMo-Gym `environment/manifest.py:101-285` | M14 |

### 4.3 Where Crucible is stronger

Verified reads on every `get_bytes`; `extra="forbid"` on every model (OpenEnv's `StepRequest` allows extras); a failure
model with blame, stage, delivery and masking, and `scored_reward()` returning `None` for masked samples; reward
invariants (a failed gate forces 0); trace validation of sequence order and reply targets; Harbor-compatible directory
hashing; live-only image bytes; a typed tool fault field; seeded per-name randomness.

## 5. Pending results

- Full unit suite coverage of `crucible.core`.
- Framework comparison findings, ranked.
