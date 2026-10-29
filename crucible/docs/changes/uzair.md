# Branch `uzair`: change log

What this branch changes relative to `main`, kept for merging with other work. One entry per change, newest last.
Each entry names the audit item it implements, the files and public names it touches, any change to persisted
formats or behaviour, the tests that cover it, and where a merge is likely to conflict.

- Base: `main` at `c27f5ca` (2026-10-15).
- Last merged from `main`: `c27f5ca`.
- Source of the items: `docs/research/improvements/2026-10-14/M00-core.md` (R-1 to R-31).
- From entry 36: the M0 re-audit `docs/research/improvements/2026-10-16/M00-core.md`, whose items are named A16-R1 to
  A16-R31 here (the 2026-10-14 items stay R-1 to R-31). Only the items with a realistic failure today, plus two
  guard tests, are implemented (the user's choice, to avoid over-engineering); the rest wait for a consumer.

## 1. Add the M00 core audit

- **Item.** none (documentation).
- **Files.** `docs/research/improvements/2026-10-14/M00-core.md` (new), `docs/changes/uzair.md` (new).
- **Code, formats, behaviour.** none.
- **Merge notes.** New files only; no conflicts expected.

## 2. R-4: run ids are checked before they become directories

- **Item.** R-4 (carry-over AUD-D16, SCOUT-2, NEW-2), narrowed: sample ids and attempts are 16-hex hashes made by
  `runtime/keys.py`, and pack names already match `core/schema/pack.py:NAME`, so only `run_id` needed a check.
- **Files.** `src/crucible/core/ids.py`, `src/crucible/runtime/journal.py`, `tests/unit/test_core_audit.py`.
- **Public names.** New `crucible.core.ids.child_path(root, *names) -> Path`. `validate_id` is unchanged.
- **Behaviour.** `RunJournal(root, run_id, ...)` raises `ValueError` before creating anything when `run_id` has a `/`,
  is `.`/`..`/empty, has a NUL or invisible character, is not NFC, or resolves outside `root` (a symlink pointing out).
  Every `RunSpec(run_id=...)` and `run_bench(...)` reaches this check first.
- **Formats.** none.
- **Tests.** `test_child_path_refuses_unsafe_names`, `test_child_path_refuses_a_symlink_out_of_root`,
  `test_run_journal_refuses_a_run_id_that_escapes_the_root`.
- **Merge notes.** One-line change in `RunJournal.__init__` (`self.dir = child_path(...)`) plus one import; a conflict
  there means someone else changed how the run directory is built. Any new caller passing a `run_id` with `/` will now
  fail; grep found none.

## 3. R-5: resolved secrets are scrubbed by value

- **Item.** R-5 (new in the audit; H2, C6). PATTERN from Harbor `trial/trial.py:1118-1170` (Apache-2.0).
- **Files.** `src/crucible/core/secrets.py`, `src/crucible/core/config.py`, `tests/unit/test_core_audit.py`.
- **Public names.** New `crucible.core.secrets.register_secret(value)` and `MIN_SECRET_LEN = 8`. `resolve_secret` keeps
  its signature and results, and now registers every value it returns (env, .env or literal).
- **Behaviour.** `scrub_secrets` (and so `scrub_obj`, `scrub_json`, every journal and record write that uses them)
  first replaces each registered value with `<secret>`, longest first, then applies the existing patterns. A key with
  no vendor prefix that a provider echoes in an error no longer reaches `Failure.reason` or `record.json`. Values under 8
  characters are not registered. The registry is per process.
- **Formats.** none.
- **Tests.** `test_a_resolved_secret_with_no_vendor_shape_is_scrubbed`, `test_short_values_are_not_registered_as_secrets`.
- **Merge notes.** `resolve_secret` was rewritten (same behaviour, plus the register call); a conflict there means someone
  else changed secret resolution, so keep the `register_secret(value)` call on every return path. New import of
  `core.secrets` in `core/config.py` (no cycle: secrets imports only `re`).

## 4. R-6: new stores are private to their owner

- **Item.** R-6 (new in the audit; AE-6). PATTERN from agentenv `store/local_state.py:20-37` (Apache-2.0). Scope: the
  core `Store` only; run directories (`RunJournal`, M10) are left for the M10 work.
- **Files.** `src/crucible/core/store/__init__.py`, `tests/unit/test_core_audit.py`.
- **Public names.** none (private helper `_make_private_root`).
- **Behaviour.** A store root that `Store` creates is mode 0700 and gets a `.gitignore` of `*`; a new `index.sqlite` is
  created 0600 before SQLite opens it. A root that already exists keeps its mode and gets no `.gitignore` (it may be
  shared on purpose). Blobs stay 0444; others cannot reach them through a 0700 root.
- **Formats.** none. A new `.gitignore` file appears at the root of newly created stores.
- **Tests.** `test_a_new_store_is_private_to_its_owner`, `test_an_existing_store_root_keeps_its_mode`.
- **Merge notes.** `Store.__init__` gained two steps (root creation, index pre-creation); a conflict there means someone
  else changed store opening, so keep both. Anything that lists a store root's files now also sees `.gitignore`.

## 5. R-7: documents with a duplicate key or a non-finite JSON number are refused

- **Item.** R-7 (new in the audit; C5). PORT with notice of OpenEnv `discovery/serialization.py:19-40` (BSD-3-Clause);
  THIRD_PARTY.md row added.
- **Files.** New `src/crucible/core/serialize.py`. Changed `src/crucible/core/schema/base.py` (`from_yaml`, `load`),
  `src/crucible/studio/pack.py` (3 sites, incl. `load_manifest`), `src/crucible/accept/checks/extended.py` (`s1_signoff`),
  `src/crucible/accept/registry/model.py` (`load_registry`), `packs/coding/.../forge/forge_gpu.py`,
  `packs/_toy/.../forge/env.py`, `packs/_toy/.../forge/calibrate.py`, `pyproject.toml` (ruff), `THIRD_PARTY.md`,
  `tests/unit/test_core_audit.py`.
- **Public names.** New `crucible.core.serialize.load_yaml(text, *, source)` and `load_json(text, *, source)`.
  `CrucibleModel.from_yaml` gained an optional `source=` keyword.
- **Behaviour.** A key given twice in one YAML mapping raises `yaml.constructor.ConstructorError` (a `yaml.YAMLError`)
  naming both lines; a YAML merge (`<<: *base`) with overriding keys is still allowed. In JSON read by
  `CrucibleModel.load`, a duplicate key or `NaN`/`Infinity` raises `ValueError`. The 361 YAML/JSON files in the
  repository were scanned: none is refused.
- **Lint.** `[tool.ruff.lint] extend-select = ["TID251"]` bans `yaml.safe_load`, `yaml.safe_load_all`, `yaml.load`,
  `yaml.full_load` outside tests. New code that loads YAML must use `load_yaml`.
- **Formats.** none.
- **Tests.** `test_a_duplicate_yaml_key_is_refused_naming_both_lines`, `test_a_yaml_merge_is_not_a_duplicate`,
  `test_json_refuses_duplicate_keys_and_non_finite_numbers`, `test_documents_and_pack_manifests_load_strictly`.
- **Merge notes.** The most likely conflict: someone else's branch adds a new `yaml.safe_load` call. Ruff then fails
  with TID251; replace it with `load_yaml(text, source=...)`. `load_manifest` and `load_registry` now go through
  `PackManifest.load` / `Defect.load` (same result for valid files).

## 6. R-8: provider keys and secret headers are env: references only

- **Item.** R-8 (new in the audit; H3, PRIME-C5, NEW-5). PATTERN from Harbor `models/trial/config.py:157-164` (Apache-2.0).
- **Files.** `src/crucible/core/config.py`, `src/crucible/actors/llm.py` (one line), `tests/unit/test_core_audit.py`.
- **Public names.** New `crucible.core.config.KEY_PLACEHOLDERS` (`""`, `"EMPTY"`, `"none"`) and
  `LLMProvider.resolved_headers(*, search_from=None)`.
- **Behaviour.** `LLMProvider.api_key` must be `env:NAME` or a placeholder; a literal key fails validation (so a
  `crucible.toml` holding one no longer loads). A header whose name looks secret (`looks_like_secret_key`:
  Authorization, X-Api-Key, Cookie, ...) must be `env:NAME`, the variable holding the whole value (e.g. `Bearer ...`);
  other headers stay literal. `LLMClient` sends `resolved_headers()`: before, headers went out as written, so an
  `env:` header was sent literally. Resolved header values are registered with the scrubber (R-5).
- **Formats.** none. Config dumps now carry references only for these fields.
- **Tests.** `test_a_literal_api_key_is_refused`, `test_references_and_local_placeholders_load`,
  `test_secret_headers_are_references_resolved_at_use`; `test_redacted_config_never_contains_raw_secrets` now masks a
  literal inside `request_defaults` (the provider fields can no longer hold one).
- **Merge notes.** Anyone adding an `LLMProvider(api_key="sk-...")` or a literal `Authorization` header in tests or
  configs will get a ValidationError; use `env:NAME` with `monkeypatch.setenv`, or `"EMPTY"` for a local engine.
  `request_defaults` can still hold literal secrets (e.g. `extra_headers`); not covered by R-8.

## 7. R-9: agentenv's NOTICE is reproduced; THIRD_PARTY.md and LICENSES/ are checked against each other

- **Item.** R-9 (new in the audit; AE-11). Apache-2.0 §4(d) obligation. The audit's digest manifest was not built: the
  coverage test below catches a missing or unnamed file, which is how this one was missed.
- **Files.** New `LICENSES/NOTICE-agentenv.txt` (agentenv-framework `NOTICE` at ac413aa, byte-identical), new
  `tests/unit/test_third_party.py`; `THIRD_PARTY.md` gains a "Notices" section.
- **Public names, formats, behaviour.** none.
- **Tests.** `test_every_licence_file_is_named_and_every_named_one_exists`, `test_the_agentenv_notice_is_reproduced`.
- **Merge notes.** Anyone adding a file to `LICENSES/` must name it in `THIRD_PARTY.md` (and vice versa), or the new
  test fails. The wheel still ships code only (`[tool.hatch.build.targets.wheel]`); add LICENSES/ and THIRD_PARTY.md
  to it if the wheel is ever distributed outside the team.

## 8. R-10: failure reasons naming non-UTF-8 text stay writable

- **Item.** R-10 (new in the audit; SW-4), narrowed to free text from the environment. PATTERN from karotte
  `src/karotte/text.py:4-24` (MIT). The model-wide variant was not taken: it would escape a policy's tool call before
  `World._call_tool` sees it, so the World could no longer refuse the call with policy blame (M14 campaign B1).
- **Files.** New `src/crucible/core/text.py`; `src/crucible/world/world.py` (the helpers moved out, imported back);
  `src/crucible/core/schema/run.py` (`Failure`); `tests/unit/test_core_audit.py`.
- **Public names.** New module `crucible.core.text` with `safe_text`, `safe_json`, `unencodable` (moved verbatim from
  `world/world.py`, which still imports them under the same names) and the field type `SafeText`.
- **Behaviour.** `Failure.reason` escapes lone surrogates to `\udcff` text before the secret scrub and the length cap;
  `Failure.detail` is `SafeText`. A failure naming a non-UTF-8 file name now serializes, hashes and reaches the journal;
  before, it validated and then raised in `RunJournal.finish`, outside the runner's try/except. Tool calls are unchanged.
- **Formats.** none.
- **Tests.** `test_a_validated_failure_always_serializes_and_hashes` (property test, 200 examples),
  `test_a_failure_naming_a_non_utf8_file_reaches_the_journal`. The B1 World test still passes.
- **Merge notes.** A branch that edits `safe_text` / `safe_json` / `unencodable` in `world/world.py` will conflict: apply
  the edit in `core/text.py`. Other free-text fields can adopt `SafeText` (not the policy's tool-call fields).

## 9. R-11: build_info() identifies the running code

- **Item.** R-11 (carry-over AUD-F3; PRIME-C2 corrected, AE-4, C8, NG-8, MV-1). PORT with notice of NeMo-Gym
  `orchestration/jobs.py:78-127` (Apache-2.0); THIRD_PARTY.md row added. Plugins (the audit's fifth field) wait for R-13.
- **Files.** New `src/crucible/core/build.py`; `src/crucible/studio/runs.py` (`provenance`, its `_git` helper removed);
  `THIRD_PARTY.md`; `tests/unit/test_core_audit.py`.
- **Public names.** New `crucible.core.build.BuildInfo` (crucible_version, source_sha256, git_commit, git_dirty,
  lock_sha256), `build_info()` (cached per process) and `source_digest(package_dir)`.
- **Behaviour.** The commit comes from PEP 610 `direct_url.json` (a git install), else from git only when the checkout
  tracks the package's `__init__.py`; a virtualenv inside another repository gets `None`, where studio used to report
  that repository's commit. `git_dirty` now covers `src/crucible/` only (before: the whole monorepo, tracked files).
  Studio `provenance()` keeps `git_sha` and `git_dirty` and adds `source_sha256` and `lock_sha256`.
- **Formats.** Studio provenance dicts (build summaries, export and bench manifests) gain two keys; nothing versioned
  changes. `Provenance` and `RunRecord` are untouched: recording BuildInfo there waits for R-18 and R-1.
- **Tests.** `test_a_foreign_repository_never_lends_its_commit`, `test_dirty_covers_the_package_only`,
  `test_source_digest_follows_content_not_bytecode`.
- **Merge notes.** A branch that edits studio's `provenance()` or its `_git` helper conflicts; the git logic now lives in
  `core/build.py`.

## 10. R-12: the store index is versioned and migrated

- **Item.** R-12 (new in the audit; TC-2). PATTERN: Crucible's own document rule (refuse newer, upgrade older),
  applied to the index; rLLM `rllm/data/dataset.py:283-318` is the counter-example (unknown versions overwritten).
- **Files.** `src/crucible/core/store/__init__.py`, `tests/unit/test_core_audit.py`.
- **Public names.** New `crucible.core.store.INDEX_VERSION` (now 1) and the private `_MIGRATIONS` list, which replaces
  the `_SCHEMA` script (same tables and indexes).
- **Behaviour.** `Store()` reads `PRAGMA user_version`: equal to `INDEX_VERSION` costs one read; lower applies the
  missing migrations and sets the version inside one `BEGIN IMMEDIATE` (concurrent opens migrate once: checked with 8
  processes on one unversioned index); higher raises `StoreError` ("newer than this Crucible ... upgrade") and closes
  the connection. Existing indexes are version 0 and become 1 on first open (migration 1's CREATEs are no-ops).
- **Formats.** The index file now carries `user_version = 1`. An older Crucible (before this change) still opens it,
  since it ignores the pragma.
- **Tests.** `test_a_new_index_carries_the_current_version`, `test_an_unversioned_index_is_adopted_with_its_rows`,
  `test_an_index_from_a_newer_crucible_is_refused`, `test_a_later_migration_reaches_an_existing_index`.
- **Merge notes.** Any schema change to the index (a new column, table or index) must be appended to `_MIGRATIONS`,
  never made by editing migration 1. A branch that still edits `_SCHEMA` conflicts here: move its DDL into a new
  migration. R-16 (tag revisions) is meant to be migration 2.

## 11. R-13: plugins are staged and merged, independent of install order

- **Item.** R-13 (carry-over IMP-M0-5, SCOUT-1, NEW-6; H8, AE-2, SW-3). PATTERN from agentenv
  `plugins/_registration.py:95-187` (Apache-2.0). Decision: a contested name is withheld only from the claimants; their
  other names still load.
- **Files.** `src/crucible/core/registry.py` (`Registry.load_plugins`, new `_Staged`, `_dist_label`),
  `tests/unit/test_core_audit.py`.
- **Public names.** none new; `Registry.failures` now also holds per-plugin refusals (withheld or built-in names).
- **Behaviour.** Entry points run sorted, each against a `_Staged` recorder: a `register()` that raises adds nothing
  (before: names registered before the exception stayed). A `(kind, name)` claimed by two or more plugins is withheld
  from all of them (before: the first in install order won, the second failed), a plugin claiming a built-in name is
  refused even with `replace=True`, and a plugin registering one name twice fails. Merged entries carry
  `source = <distribution>==<version>` (e.g. `crucible-pack-toy==0.3.0`) instead of the string the plugin passed.
- **Formats.** none. `describe()` output shows the new source labels.
- **Tests.** `test_a_plugin_that_fails_halfway_registers_nothing`,
  `test_a_contested_name_is_withheld_whatever_the_install_order` (both orders),
  `test_a_plugin_cannot_take_a_built_in_name`.
- **Merge notes.** A pack's `register(registry)` now receives a `_Staged` recorder, not the `Registry`: only
  `register(...)` is available during registration (all current packs use only that). Code matching on a plugin's own
  `source=` string should match `name==version` instead.

## 12. R-14: plugins for another Crucible are skipped; `crucible registry --check`

- **Item.** R-14 (carry-over AUD-F1, IMP-M0-5; AE-3). PORT with notice of agentenv `plugins/_requirements.py:29-55`
  (Apache-2.0); THIRD_PARTY.md row added. `packaging` is used when installed (not a declared dependency: harbor,
  huggingface-hub and pytest bring it); without it the gate is skipped with one warning.
- **Files.** `src/crucible/core/registry.py` (`_incompatible_core`, `_warn_once`, `Registry.check`, a gate in
  `load_plugins`), `src/crucible/cli/main.py` (`registry --check`), `../.github/workflows/crucible.yml` (new step),
  `THIRD_PARTY.md`, `tests/unit/test_core_audit.py`.
- **Public names.** New `Registry.check() -> {"broken", "unavailable", "failed_plugins"}`; CLI flag
  `crucible registry --check` (exit 1 on a broken pointer or a failed plugin).
- **Behaviour.** A plugin whose unconditional `crucible` requirement excludes the running `__version__` is skipped before
  its module is imported (`failures[ep] = "incompatible core: needs crucible<0.2,>=0.1 (installed: ...)"`); a
  requirement under an extra does not count. `check()` resolves every pointer: a missing module of the pointer's own
  package or of Crucible, a missing attribute or an import error is `broken`; a missing third-party module is
  `unavailable` (an optional dependency).
- **CI.** New step "Registry pointers resolve and plugins load" after the import contracts. Checked locally with
  CI's extras and Harbor at CI's pin a2fafbc: 0 broken, 1 unavailable (`transport/a2a`, agentenv is not in CI).
- **Formats.** none.
- **Tests.** `test_a_plugin_for_another_crucible_is_skipped_before_import`,
  `test_check_separates_broken_pointers_from_missing_optional_dependencies`, `test_registry_check_exits_1_on_a_broken_pointer`.
- **Merge notes.** When the version moves to 0.2, every pack's `crucible>=0.1,<0.2` must move with it or the packs are
  skipped (by design). A new built-in pointer to a missing Crucible module fails CI at the new step.

## 13. R-15: every CRUCIBLE_* variable is declared and read typed

- **Item.** R-15 (new in the audit; TC-4). PATTERN from rLLM `rllm/env.py:1-46` (Apache-2.0). Scope chosen: every
  Crucible-side read converted, training included.
- **Files.** New `src/crucible/core/env.py`. Reads converted in `core/config.py`, `actors/llm.py`, `world/docker_tunnel.py`,
  `world/health.py`, `connectors/redact.py`, `studio/explorer.py`, `studio/pack.py`, `bench/dynamic.py`, `bench/run.py`,
  `train/cluster.py`, `train/spmd.py` (its `_env_float` removed), `train/bridges/verl_agent_loop.py`,
  `packs/_toy/.../m12/gate.py`, `packs/_toy/.../m12/gate_dist.py`. Comments only in `actors/rlm/repl.py` and
  `verify/runtime/patterns/hidden_tests.py` (sandboxed readers that cannot import Crucible keep `os.environ`).
  `tests/unit/test_core_audit.py`.
- **Public names.** `crucible.core.env`: `KNOBS` (46 declared: 19 user, 8 internal, 19 test/script), `Knob`, `EnvError`,
  `get(name)`, `parse(name, raw)`, `require(name)`, `undeclared(environ)`. Convention: `from crucible.core import env as knobs`.
- **Behaviour.** A malformed value raises `EnvError` naming the variable (before: silent fallback or a bare
  `ValueError` deep in a call): `CRUCIBLE_LLM_CACHE=reply` no longer turns the cache off, and `LLMClient(cache_mode=...)`
  refuses values other than record/replay/off. Booleans accept 1/0, true/false, yes/no, on/off:
  `CRUCIBLE_CPU_ONLY=0` now means GPU (any non-empty value meant CPU), `CRUCIBLE_DOCKER_SSH_TUNNEL=false` now disables
  the tunnel (only "0" did). `CRUCIBLE_SPMD_*` no longer fall back on a malformed number. Secret knobs
  (`CRUCIBLE_REDACTION_SALT`, `CRUCIBLE_BENCH_KEY`, `CRUCIBLE_VLLM_KEY`) are registered with the scrubber when read
  and never echoed in errors. The first read warns once about undeclared `CRUCIBLE_*` variables, naming the closest knob.
- **Formats.** none.
- **Tests.** `test_a_malformed_knob_fails_loudly_naming_it`, `test_knobs_parse_to_their_types_and_defaults`,
  `test_a_secret_knob_is_scrubbed_and_never_echoed`, `test_an_undeclared_crucible_variable_is_named_with_the_closest_knob`,
  `test_every_crucible_variable_in_the_code_is_declared` (scans src/ and packs/), `test_the_llm_client_refuses_an_unknown_cache_mode`.
  26 affected suites incl. training (cluster, FSDP, verl bridge, M12): 468 passed; failures were pre-existing (2 studio)
  or a Ray timeout that also fails on main (`test_ray_launcher_places_per_node_groups`; passes alone on this branch).
- **Merge notes.** A new `CRUCIBLE_*` name anywhere in src/ or packs/ must be added to `KNOBS`, or the scan test fails;
  read it with `knobs.get(...)`. A branch still calling `os.environ.get("CRUCIBLE_...")` in the converted files will
  conflict on those lines; `_env_float` in `train/spmd.py` no longer exists.

## 14. R-18: every persisted format is frozen per version

- **Item.** R-18 (carry-over AUD-D8, D13, D14, SCOUT-3, DOC-gate-1, DES-M0-1). PATTERN from OpenEnv
  `tests/test_validation/test_schema_sync.py` (BSD-3-Clause), one file per version instead of one overwritten file;
  THIRD_PARTY.md row added. Decisions: rollout/acceptance/control keep their `schema` key; unversioned files
  (`steps.jsonl` rows, `WorldRef`) are snapshotted, with no version field added yet.
- **Files.** `src/crucible/core/schema/base.py` (shared `migrate_document`, `check_same_major`, new `SchemaKeyed`;
  `VersionedDocument` uses them, same behaviour and messages), `core/schema/rollout.py`, `accept/model.py`,
  `train/control_record.py` (`schema_: Literal[...]` becomes `str` on `SchemaKeyed`), `core/schema/pack.py`
  (newer integer version refused by name). New: `schemas/` (16 formats: `<name>/<version>.schema.json` and
  `.example.json`), `tests/unit/_formats.py` (the format list and rendering), `tests/unit/test_formats.py`,
  `scripts/freeze_schemas.py`. `THIRD_PARTY.md`.
- **Public names.** `crucible.core.schema.base.SchemaKeyed`, `migrate_document(owner, schema, data, keys)`,
  `check_same_major(owner, schema, v, key)`.
- **Behaviour.** Rollout, acceptance and control records now refuse a newer major or minor version with an explicit
  message and can be upcast from older majors (before: any other string failed a Literal); their key and key order are
  unchanged. `pack.yaml` with `schema_version: 2` is refused as "written for a newer Crucible".
- **Formats.** No document changes. The current shapes are frozen as v1 (rollout v1 includes the `agent` and
  `trainable` fields added in place on 2026-10-13). Snapshots hold the shape only (JSON Schema without titles and
  descriptions), so docstring edits are not format changes.
- **Tests.** `test_formats.py`: each format matches its version's snapshot; every example of every version loads; every
  `VersionedDocument`/`SchemaKeyed` class and `ALL_DOCUMENTS` entry is frozen; an unbumped change is caught; schema-keyed
  rules; pack version refusal; the snapshot normalizer. 45 suites using these formats (training included): 706 passed;
  3 failures pre-existing, 1 (`test_v1_oracle_replay_crash_fails_the_build`) flaky in that long batch only (passes
  alone here and on main, and in its own file here).
- **Merge notes.** Any branch that adds or changes a field on a persisted model now fails `test_formats.py` until it
  bumps the version and runs `uv run --frozen python scripts/freeze_schemas.py`; files under `schemas/` are never edited
  by hand. `--refresh` is only for a pydantic upgrade that re-renders unchanged models.

## 15. R-1 (step 1 of 4): the run lock, a fail-closed run identity

- **Item.** R-1 step 1 (carry-over IMP-M0-3, IMP-M0-7, AUD-F3, NEW-5). PATTERN from NeMo-RL
  `rollout_checkpoint.py:62-204` and slime `training_recovery.py` (Apache-2.0); THIRD_PARTY.md row added. Steps left:
  2 launch records and 3 run-record lock/launch ids (both with R-2/R-3/R-19 as `crucible.run/v2`), 4 training-checkpoint
  identity (M12).
- **Files.** New `src/crucible/core/fingerprint.py`, `src/crucible/runtime/lock.py`, `tests/unit/test_run_lock.py`,
  `schemas/crucible.run_lock/v1.*`. Changed `core/schema/run.py` (`LockChange`, `RunLock`), `runtime/run.py`
  (`RunSpec.allow_change`, `_lock` called by `Runner.run` and `Runner.regrade`), `runtime/journal.py` (the old identity
  check is skipped once a lock exists), `runtime/harness/native.py` (`PolicyRef.settings`, `lock_settings()` on
  `PolicyRef` and `NativeHarness`), `actors/llm.py` (`LLMClient.lock_settings()`), `runtime/single_turn.py` (the
  single-turn policy declares its settings), `tests/unit/_formats.py`, `THIRD_PARTY.md`.
- **Public names.** `crucible.core.fingerprint`: `SCHEME`, `project`, `fingerprint`, `diff`, `describe`, `excluded`,
  `flatten`. `crucible.runtime.lock`: `OPERATIONAL`, `RunLockMismatch`, `lock_doc`, `build`, `ensure`, `jsonable`.
  `RunLock`/`LockChange` (`crucible.run_lock/v1`). `RunSpec.allow_change`, `PolicyRef.settings`, `lock_settings()`.
- **Behaviour.** Before the first dispatch the runner writes `<run>/lock.json`: every `RunSpec` field (new fields count
  automatically) and each policy's and harness's declared settings, minus `OPERATIONAL` (concurrency, budget, wall clock,
  retry backoff, health thresholds, allow_change) and secrets, fingerprinted. A resume whose settings differ raises
  `RunLockMismatch` (a `ValueError`) naming up to 20 changed dotpaths, unless each matches a glob in `allow_change`; an
  allowed change is recorded in `lock.json` and the journal (`run_lock` event). Policies without settings (scripted refs,
  pack-provided refs) count by id only and are listed in `opaque`. A run directory from before this change is checked by
  its old identity, then adopted. Task content is not covered yet (R-2).
- **Formats.** New `crucible.run_lock/v1` (lock.json). Run records unchanged.
- **Tests.** 9 in `test_run_lock.py` (globs, projection, diff, field coverage, refusal naming temperature and max_steps,
  operational changes, allowed change recorded, legacy adoption, opaque listing). 43 runner suites + toy e2e: 677
  passed, 3 pre-existing failures.
- **Merge notes.** Code that resumes a run with changed settings now raises unless it passes `allow_change`. A new
  operational `RunSpec` field must be added to `runtime.lock.OPERATIONAL`, or it counts in the identity. A harness
  attribute that is not a setting should be private (`_name`) or listed in the class's `LOCK_EXCLUDE`. Packs can make
  their policies non-opaque with `PolicyRef(..., settings={...})`.

## 16. R-1 (step 2 of 4): a launch record per invocation

- **Item.** R-1 step 2 (the RunLaunch part of the audit's recommendation).
- **Files.** `src/crucible/core/schema/run.py` (`RunLaunch`; imports `core.build.BuildInfo`), `src/crucible/runtime/lock.py`
  (`launch`, `LAUNCH_DIR`), `src/crucible/runtime/run.py` (`_lock` records the launch), `tests/unit/_formats.py`,
  `schemas/crucible.run_launch/v1.*`, `tests/unit/test_run_lock.py`.
- **Public names.** `RunLaunch` (`crucible.run_launch/v1`), `crucible.runtime.lock.launch(run_dir, spec, lock, status)`.
- **Behaviour.** Every `Runner.run` / `Runner.regrade` writes `<run>/launches/<n>.json` (n = 1, 2, ...; created
  exclusively) after the lock check: argv with secrets scrubbed, the `OPERATIONAL` values used (concurrency, budget, ...),
  `Config.redacted()` (scrubbed), `build_info()` (source digest, commit, dirty flag, lock hash), the Python version, the
  lock fingerprint and its status (created/adopted/same/changed). The journal gets a `launch` event with n and the commit.
  An unreadable configuration is recorded as an error, never a reason not to run.
- **Formats.** New `crucible.run_launch/v1`. Run directories gain a `launches/` folder. Run records unchanged (their
  launch id comes with `crucible.run/v2`, step 3).
- **Tests.** `test_each_invocation_is_recorded_as_a_launch` (numbering, operational values, same lock, scrubbed argv,
  journal events). 43 runner suites + toy e2e: 679 passed, 3 pre-existing failures.
- **Merge notes.** Code that lists a run directory now also sees `launches/` (and `lock.json` from step 1).

## 17. R-2: task content digests, in the lock and checked around every attempt

- **Item.** R-2 (correction of `core/schema/run.py:161-162`, documented as hashes but filled with names; merges
  openenv-hf:C2). PATTERN from OpenEnv `validation/runner.py` (`source_digest`, BSD-3-Clause); THIRD_PARTY.md row added.
  Decision: tasks that cannot easily be hashed declare no digest and are listed as uncovered.
- **Files.** `src/crucible/core/ids.py` (`tree_digest`), `src/crucible/core/schema/run.py` (kind `crucible:task_changed`:
  task blame, non-retryable), `src/crucible/runtime/lock.py` (`task_digest`, `definition_digest`; digests enter the lock;
  tasks without one are listed in `opaque`), `src/crucible/runtime/run.py` (`_digest_now`, `_task_changed`, the checks in
  `_attempt`, `_record(..., task_digest)`, `_lock` returns the lock), `interop/harbor/inplace.py`
  (`HarborInPlaceTask.digest`: Harbor's `Packager.compute_content_hash`), `interop/harbor/verify_dir.py`
  (`HarborDirTask.digest`: `tree_digest`), `runtime/single_turn.py` and `runtime/control_task.py` (`digest`: the
  definition in canonical JSON), `THIRD_PARTY.md`, `tests/unit/test_task_digest.py`, `tests/unit/test_run_lock.py`.
- **Public names.** `crucible.core.ids.tree_digest(root)`; the optional task method `digest() -> str`;
  `crucible.runtime.lock.task_digest`, `definition_digest`; failure kind `crucible:task_changed`.
- **Behaviour.** A task's digest is taken when the lock is built (so a resume with an edited task raises
  `RunLockMismatch` naming `tasks.<i>.digest`), again at dispatch (a task edited since the lock was taken is never
  dispatched: the attempt is recorded as `crucible:task_changed`, masked, not retried, nothing spent) and again after
  the attempt (a task edited during it has its grade discarded: `crucible:task_changed`, cost kept). The record's
  `meta.task_digest` holds the dispatch digest. `tree_digest` hashes regular files only (sorted
  `relpath\0sha256\n` lines, skipping .git, __pycache__, *.pyc, .DS_Store) and refuses symlinks and files changing
  mid-read. A Harbor task's digest equals Harbor's own task hash (checked on the task04 seed). Toy and pack Python
  tasks declare none yet. No code was found writing into a task directory; Docker-run Harbor tasks were not run here.
- **Formats.** none new; `RunRecord.meta` gains `task_digest` (free-form), a typed field comes with `crucible.run/v2`.
  `Runner.regrade` does not check digests yet.
- **Tests.** `test_task_digest.py` (8): digest stable and content-sensitive, symlink refused, Harbor equality, digest
  recorded, edited during the attempt, edited after the lock (never dispatched), resume refused naming the task.
  50 suites (runner, Harbor, single-turn, control): 720 passed, 3 pre-existing failures.
- **Merge notes.** A task class can opt in by adding `digest()`; it must be deterministic and cheap (it runs at every
  dispatch and after every attempt). A harness that writes into its task directory would now mask every attempt as
  `crucible:task_changed`: write under the attempt directory instead.

## 18. R-3: each turn records the sampling its call actually sent

- **Item.** R-3 (carry-over AUD-F2 sampling; merges openenv-hf:C3, mimo-verl:MV-3, PRIME-C1's sampling part). PATTERN from
  OpenEnv `core/harness/capture/graph.py` (BSD-3-Clause); THIRD_PARTY.md row added. Decisions: the trainer refuses an
  unrecorded temperature by default (opt-out `allow_unrecorded_sampling`); the verl bridge's parameters wait for M12.
- **Files.** `src/crucible/core/schema/rollout.py` (`SamplingParams`, `NOT_SAMPLING`, `Turn.sampling` typed, rollout
  `v1.1`), `src/crucible/actors/llm.py` (`LLMResponse.sampling`; `_request` factored out of `_complete_paid`; `_sent`;
  cache entries keep `sampling`; the token-in-token-out path records what the engine was sent),
  `src/crucible/actors/policy.py` (call records carry `sampling`), `src/crucible/runtime/record.py` (a turn takes its
  call's sampling; the `sampling` argument is only a fallback), `src/crucible/runtime/run.py` (no longer passes
  `RunSpec.sampling` to the rollout), `src/crucible/train/trainer.py` (`TrainerConfig.allow_unrecorded_sampling`; the
  temperature guard; truncation by `SamplingParams.truncated`), `src/crucible/train/teachers.py`, `schemas/crucible.rollout/v1.1.*`,
  `tests/unit/_formats.py`, `THIRD_PARTY.md`, `tests/unit/test_sampling.py`, and test fixtures in 11 training test files
  and 3 worker scripts (turns now record `temperature: 1.0`).
- **Public names.** `crucible.core.schema.rollout.SamplingParams` (`from_request`, `truncated`), `LLMResponse.sampling`,
  `TrainerConfig.allow_unrecorded_sampling`.
- **Behaviour.** `Turn.sampling` is what the call sent: the policy's parameters plus the provider's `request_defaults`
  and the call's extras (`extra_body` keys lifted; non-sampling keys such as `logprobs` and `return_token_ids` dropped).
  Before, every turn held a copy of `RunSpec.sampling` (default `{}`), blind to `request_defaults`. The trainer now
  refuses a trained turn without a recorded temperature (it used to skip the check), and treats top_k and min_p, not only
  top_p < 1, as truncation that needs candidate sets. An unset parameter is left out, so an empty record still dumps as `{}`.
- **Formats.** `crucible.rollout/v1` -> `v1.1` (minor): `Turn.sampling` is typed. v1 records read (their dicts mapped,
  unknown keys in `extra`); a v1 reader refuses v1.1 records with the newer-minor message.
- **Tests.** `test_sampling.py` (5): parameters from a request and from a v1 dict, the client records provider defaults,
  a cached reply keeps its sampling, a turn takes its call's sampling, the trainer refuses unrecorded/different
  temperatures and accepts with the opt-out. 42 training suites: 418 passed (63 failed before the fixtures recorded a
  temperature); 59 other suites: 858 passed, 3 pre-existing failures.
- **Merge notes.** Code reading `turn.sampling.get("x")` or `turn.sampling["x"]` must use attributes
  (`turn.sampling.temperature`). Test rollouts that are trained must record a temperature, or set
  `allow_unrecorded_sampling=True`. A new key in an LLM request that is not a sampling parameter belongs in `NOT_SAMPLING`.

## 19. R-19: typed token usage, finish reasons and stage timings

- **Item.** R-19 (carry-over AUD-F2, IMP-M1-8, NEW-4; merges harbor:H6, PRIME-C1, nemo-gym:NG-7, mimo-verl:MV-4, SW-8).
  PATTERN from verifiers `v1/types.py` (`Usage.from_openai`, `aggregate`, `FinishReason`) and `v1/trace.py` (`Timing`),
  MIT, no code copied; THIRD_PARTY.md row added. Decisions: `Turn.finish` typing amends `crucible.rollout/v1.1` (never
  pushed) instead of a v1.2; a missing cache price is the input price; `RunRecord` gets `meta["usage"]` and
  `meta["timing"]` now, typed fields and the removal of `RunRecord.tokens` wait for `crucible.run/v2` (R-1 step 3).
- **Files.** `src/crucible/core/schema/usage.py` (new: `Usage`, `UsageTotals`, `FinishReason`, `finish_reason`,
  `finish_fields`, `Timing`), `src/crucible/core/schema/rollout.py` (`Turn.finish` typed), `src/crucible/core/config.py`
  (`prices` take cache read/write rates), `src/crucible/actors/llm.py` (`LLMResponse.usage` normalized, `.finish`,
  `.finish_fields()`; `_price` by token kind; `_usage_dict`), `src/crucible/actors/policy.py`, `actors/harness.py`,
  `actors/rlm/agent.py`, `actors/rlm/context.py` (call records' finish; "length" checks on the normalized reason),
  `src/crucible/runtime/harness/base.py` (`HarnessResult.usage`, `.model_s`, `settle_usage`), `runtime/harness/native.py`,
  `runtime/harness/workspace.py`, `interop/agentenv/harness.py` (settle usage; captured calls read through `Usage`),
  `src/crucible/runtime/run.py` (`meta["usage"]`, `meta["timing"]`; `_task_changed` keeps them),
  `src/crucible/runtime/record.py` (rollout timings from `meta["timing"]`), `interop/harbor/trajectories.py` (ATIF metrics
  from `Usage`), `bench/metrics.py`, `bench/report.py`, `train/world_env.py`, `runtime/scripted_rlm.py`,
  `runtime/inject.py` (usage source `engine` / `estimate`), `schemas/crucible.rollout/v1.1.*` (regenerated),
  `tests/unit/_formats.py` (the v1.1 example has a finish), `crucible.toml`, `docs/modules/M00-core.md`,
  `THIRD_PARTY.md`, `tests/unit/test_usage.py` (new), `tests/unit/test_runtime_records.py` (the blank-usage marker).
- **Public names.** `crucible.core.schema.usage.{Usage, UsageTotals, UsageSource, FinishReason, FINISH_REASONS,
  finish_reason, finish_fields, Timing, STAGES}`, `LLMResponse.finish`, `LLMResponse.finish_fields()`,
  `HarnessResult.usage`, `HarnessResult.model_s`, `runtime.harness.base.settle_usage`; bench `Episode.tokens_cached`,
  `.tokens_replayed`, `.model_calls`, `.calls_reported`.
- **Behaviour.**
  - A call's usage is one shape whatever the provider: `input_tokens` (uncached), `cache_read_tokens`,
    `cache_write_tokens`, `output_tokens`, `reasoning_tokens` (inside output), `source` (`provider | engine | estimate |
    none`), plus the totals `prompt_tokens` / `completion_tokens`. OpenAI chat, Anthropic and Responses usage of the same
    call give the same numbers (before, an Anthropic-shaped capture counted its prompt without the cached part).
  - Configured prices charge cache reads and writes at their own rates when given (`[in, out, read, write]`); the audit's
    900-of-1000-cached call is booked 0.0013125 USD instead of 0.003 (2.29x). Without cache prices, unchanged.
  - Call records' `finish` is a `FinishReason` (`stop | length | tool_calls | content_filter | refusal | abort | error`);
    the provider's own word is kept in `finish_raw` when it differs. `length` checks (nudges, silent overflow, the
    trainer's truncation) now also see Anthropic's `max_tokens` and Responses' `max_output_tokens`.
  - Each attempt records `meta["usage"]` (its own live calls summed: never calls restored from a checkpoint or inherited
    by a re-grade; replays counted apart, with coverage) and `meta["timing"]` (stages plus `model`, the summed call
    latency inside `agent`), replacing `meta["timings"]`.
  - Bench reads `meta["usage"]` (records before R-19: `calls.json` summed, replays apart) and never `RunRecord.tokens`;
    it adds `tokens_cached`, `tokens_replayed`, `model_calls`, `calls_reported` and `observability.usage_reported`, and
    fills Harbor's `n_cache_tokens`. Replayed calls no longer count as live tokens.
- **Formats.** `crucible.rollout/v1.1` amended (unpushed): `Turn.finish` is an enum; v1 records with raw words read
  normalized (an unknown word reads as not stated). `crucible.run/v1` unchanged (`meta` is free); `meta["timings"]` is
  now `meta["timing"]`. `calls.json` usage dicts and LLM cache entries take the `Usage` shape (older ones still read).
- **Tests.** `test_usage.py` (8): provider shapes agree (4 shapes, cache writes), cache-rate pricing and replay, replays
  and `RunRecord.tokens` not counted as live (old and new records), an attempt's own usage and model time in its record,
  one finish vocabulary (v1 rollouts, Anthropic captures). `test_runtime_records.py`: a blank usage is now
  `source: "none"`, not `{"blank": 1}`. 151 non-training suites: 1613 passed, 7 failures that fail at HEAD too (the 3
  known ones, plus `test_agentenv_mcp_env` card tools, `test_ci_app` hidden checks, `test_cluster` Ray placement,
  `test_triggers` agentenv triggers). The 15 training suites that read finish, usage or timings: 225 passed.
- **Merge notes.** Code reading `usage["cached_tokens"]` or `usage["blank"]` must read `cache_read_tokens` /
  `source == "none"` (or `Usage.model_validate`). Code reading `rec.meta["timings"]` must read `meta["timing"]`
  (`Timing.model_validate(...).stages()`). New call-record producers should store `**r.finish_fields()` and a usage dict
  (it is normalized on `LLMResponse`). A new provider finish word belongs in `_FINISH_ALIASES`.

## 20. R-1 (step 3 of 4): `crucible.run/v2`, the run record names its lock and launch

- **Item.** R-1 step 3, with the record fields of R-2 (task digest) and R-19 (usage, timing); the first upcaster in use
  (IMP-M0-6). Mechanism: Crucible's own `register_upcaster` (the pattern of Harbor's `mode="before"` migrations in
  `models/task/config.py`; no code copied). Decisions: v1's `tokens` is dropped by the upcaster (nothing ever filled
  it, on `main` or `origin/haseeb`); `world_digest` (suggested by R-2) waits until something computes it (a v2.1 field).
  Step left: 4, training-checkpoint identity (M12).
- **Files.** `src/crucible/core/schema/run.py` (`RunRecord` v2 fields, `_run_v1_to_v2`, registered upcaster),
  `src/crucible/runtime/run.py` (`_lock` returns the lock and a `{lock, launch}` stamp; `_record(..., lock=, launch=)`
  fills the typed fields; `run` and `regrade` stamp every record), `src/crucible/runtime/record.py` (rollout timings from
  `rec.timing`), `src/crucible/bench/metrics.py` (reads `rec.usage`, `rec.timing`), `schemas/crucible.run/v2.*` (new;
  v1 files kept), `tests/unit/_formats.py` (v2 example), `tests/unit/test_run_lock.py` (2 tests),
  `tests/unit/test_usage.py`, `tests/unit/test_task_digest.py`, `tests/unit/test_m13_audit_metrics.py` (its record
  fixture sets `timing` instead of `meta["timings"]`), `docs/modules/M00-core.md`.
- **Public names.** `RunRecord.task_digest`, `.usage` (`UsageTotals`), `.timing` (`Timing`), `.lock`, `.launch`;
  `RunRecord.tokens` removed.
- **Behaviour.** Every record the runner writes (attempts, re-grades, records of failed record/rollout builds) carries
  the lock fingerprint and launch number of its invocation, so a resumed run's records can be told apart by settings
  and code (`launches/<n>.json`). `meta` no longer holds `task_digest`, `usage` or `timing`. Records made outside the
  runner (NeMo-Gym client, `train/world_env`) leave the new fields `None`.
- **Formats.** `crucible.run/v1` -> `v2` (major). The upcaster moves `meta.task_digest`, `meta.usage`, `meta.timing`
  into their fields, turns pre-R-19 `meta.timings` into `timing`, drops `tokens`, and leaves `lock`/`launch` `None`.
  v1 files on disk are not rewritten; a `results.jsonl` mixing v1 and v2 rows reads. A v1 reader (code on `main`)
  refuses v2 records as a newer major version.
- **Tests.** `test_run_lock.py`: records name the lock and launch they ran under (a resume with an allowed change gets
  launch 2 and the new fingerprint); a v1 record reads as v2 (with R-19 meta, with pre-R-19 timings, a mixed
  `results.jsonl`). `test_formats.py`: the v2 schema is frozen and the v1 example still loads. 151 non-training
  suites and the 8 training suites that build or read run records: 1743 passed, the same 7 failures as at HEAD
  (entry 19).
- **Merge notes.** Code reading `rec.tokens`, `rec.meta["task_digest"]`, `rec.meta["usage"]` or `rec.meta["timing"]`
  must read the fields. Code writing `RunRecord(tokens=...)` fails (unknown key). Code that reads `record.json` or
  `results.jsonl` as plain dicts sees `schema_version: crucible.run/v2` and the new keys; validate through `RunRecord`
  to read v1 and v2 alike.

## 21. R-1 (step 4 of 4): a training checkpoint records the settings it was trained with

- **Item.** R-1 step 4 (the M12 part of the audit's protocol: "M12 checkpoints store the train-config identity",
  `train/state.py` checked none). PATTERN from NeMo-RL `rollout_checkpoint.py:60-110,490-552` and slime
  `training_recovery.py:60-151` (Apache-2.0), on step 1's `core/fingerprint.py`; THIRD_PARTY.md row extended.
  Decision (option a): `lr` counts in the identity, and an allowed change is really applied.
- **Files.** `src/crucible/train/identity.py` (new), `src/crucible/train/state.py` (`capture` stores the identity,
  `save` writes its fingerprint into `manifest.json`, `restore` checks before installing anything, `configured_lr`),
  `src/crucible/train/runtime.py` (`RuntimeConfig.allow_change`, `RunReport.identity_check`, the sources' identity taken
  at construction, a `resume` tracking record), `tests/unit/test_train_state.py` (3 tests; the helper takes lr, n,
  prompts, allow_change), `docs/modules/M12-train.md`, `THIRD_PARTY.md`.
- **Public names.** `crucible.train.identity`: `OPERATIONAL`, `FUNCTIONS`, `TrainResumeMismatch`, `identity`,
  `identity_doc`, `sources_doc`, `check`. `crucible.train.state.configured_lr`. `RuntimeConfig.allow_change`,
  `RunReport.identity_check`.
- **Behaviour.**
  - Every checkpoint carries `identity`: every field of `RuntimeConfig`, `SchedulerConfig`, the trainer's config (and
    the critic's), the trainer class, `vocab_id`, a digest of the model's parameter names and shapes, each source's
    prompt count and digest, and the rollout / reward / environment / group / audit functions by qualified name (listed
    in `opaque`), minus `OPERATIONAL` (steps, wall clock, polling, workers, checkpoint cadence and directory, sinks, kept
    records, breaker thresholds, offload and KV placement, reissue pacing, heartbeat, `dp`, `token_budget`,
    `prefix_sharing`, `allow_change`), fingerprinted.
  - `Runtime.resume()` under other settings raises `TrainResumeMismatch` (a `ValueError`) naming up to 20 changed
    settings, before anything is installed, unless each matches a glob in `RuntimeConfig.allow_change`; an allowed change
    is in `RunReport.identity_check`, a `resume` tracking record, and the `changes` history of every later checkpoint.
  - A checkpoint from before this change (no identity) resumes once as `adopted`, recorded the same way.
  - After the optimizer state loads, each optimizer runs at its configured `lr` (and the critic's): loading had put the
    checkpoint's rate back, so an `lr` change on resume was silently ignored (configured 1e-4, ran at 3e-3).
- **Formats.** `crucible.run_checkpoint/v1` blobs gain an `identity` key and `manifest.json` an `identity`
  fingerprint (both optional on read; no version bump: the checkpoint is not a frozen format).
- **Tests.** `test_train_state.py` (7, 3 new): a resume with another group size, learning rate or prompt set is
  refused by name with nothing installed, more steps resume as `same`; an allowed `lr` change is applied and carried in
  later checkpoints; a checkpoint without an identity resumes as `adopted`. The 58 suites that import `crucible.train`:
  630 passed, 1 failure that fails at HEAD too (`test_cluster` Ray placement, entry 19).
- **Merge notes.** A training resume with changed settings now raises unless `RuntimeConfig.allow_change` names them.
  A new `RuntimeConfig` / `SchedulerConfig` / `TrainerConfig` field counts automatically; an operational one must be
  added to `train.identity.OPERATIONAL`. A trainer that schedules its learning rate would need `configured_lr` revisited.

## 22. R-16: tags move by compare-and-set, with revisions and a history

- **Item.** R-16 (carry-over AUD-F4; merges harbor:H5, agentenv:AE-1, nemo-gym:NG-10, SW-1, trainers-c:TC-1). PATTERN
  from SkyRL `tinker/api.py:1145-1176` (Apache-2.0); THIRD_PARTY.md row added. From here on the remaining M00 items are
  implemented without per-item approval (the user's instruction of 2026-10-16); each decision below is the recommended
  option. Decision: a fetch-order guard on trace heads (option a), so N ingests keep the newest head.
- **Files.** `src/crucible/core/store/__init__.py` (index migration 2; `Store.tag(name, key, *, expect, note)`,
  `tag_info`, `tag_history`, `TagInfo`, `TagEvent`, `TagConflict`, `check_tag_name`, `TAG_MAX_LEN`),
  `src/crucible/connectors/trace_store.py` (`put(..., fetched_at=)` by compare-and-set with retries, `head()`,
  `PUT_ATTEMPTS`), `src/crucible/connectors/base.py` (`ingest` stamps the fetch time before each fetch),
  `tests/unit/test_tags.py` (new), `tests/unit/test_core.py`, `tests/unit/test_core_audit.py`,
  `docs/modules/M00-core.md`, `THIRD_PARTY.md`.
- **Public names.** `Store.tag` takes a required keyword `expect` (0: create only; n: only from revision n; None: no
  check) and returns the revision; `Store.tag_info`, `Store.tag_history`; `crucible.core.store.{TagInfo, TagEvent,
  TagConflict, check_tag_name, TAG_MAX_LEN}`; `TraceStore.head`, `TraceStore.put(..., fetched_at=)`.
- **Behaviour.**
  - The check and the move are one `BEGIN IMMEDIATE` transaction (safe across processes); a stale writer gets
    `TagConflict` naming the current revision and key, and nothing changes. Moving a tag to the key it already holds is
    a no-op. Every move is appended to `tag_history` (name, rev, key, previous key, time, note).
  - Tag names: a string of 1 to 512 characters, no control characters, never starting with `sha256:` (such a tag was
    accepted but `resolve()` could never return it; an integer was accepted too).
  - `TraceStore.put` re-reads the head and carries its signals again when it loses the head to another put (a stale
    read dropped signals), and a version whose source was fetched before the head's (`fetched_at` in the head's history
    note) is stored but never becomes the head. Before, a slow ingest that tagged last moved the head back to the
    older version and `purge_superseded()` then deleted the newer one.
- **Formats.** Store index version 1 -> 2: `tags.rev` (existing tags: 1) and the new `tag_history` table (existing
  tags get one row). Code with index version 1 (`main`) refuses a store this branch has opened.
- **Tests.** `test_tags.py` (10): compare-and-set and history; 2 connections racing 40 moves, each revision won once;
  5 refused names; an index from before revisions migrates; 8 shuffled ingests of a growing source keep the newest head
  through a purge; a put that lost the head carries the new head's signals. Connector, accept and core suites: 248
  passed.
- **Merge notes.** Every `Store.tag(name, key)` call needs `expect=` (read it with `tag_info`). Code that read the
  `tags` table directly (`TraceStore._head_key` did) should use `tag_info`. A new caller of `TraceStore.put` that knows
  when it read its source should pass `fetched_at`.

## 23. R-17: a rollout records its engine, log-prob mode and tokenizer

- **Item.** R-17 (new; TB-2, TC-6). PATTERN from AReaL `areal/dataset/swe_sft/pipeline.py:88-136` (Apache-2.0);
  THIRD_PARTY.md row added. Decision: the field joins the unpushed `crucible.rollout/v1.1` (as R-19 did) rather than a
  v1.2.
- **Files.** `src/crucible/core/schema/rollout.py` (`TokenizerId`, `Capture`, `RolloutRecord.capture`),
  `src/crucible/train/runtime.py` (`_capture_of`: engines and their log-prob mode when a record finishes),
  `src/crucible/train/world_env.py` (`codec_capture`: the codec's tokenizer and template arguments, once per rollout
  function), `src/crucible/train/trainer.py` (`TorchTrainer.tokenizer_id`, `_check_capture` called by `_gate`),
  `src/crucible/train/identity.py` (the trainer's tokenizer joins the training identity), `schemas/crucible.rollout/v1.1.*`
  (regenerated), `tests/unit/_formats.py`, `tests/unit/test_capture.py` (new), `docs/modules/M12-train.md`,
  `THIRD_PARTY.md`.
- **Public names.** `crucible.core.schema.rollout.{TokenizerId (of), Capture}`, `RolloutRecord.capture`,
  `TorchTrainer.tokenizer_id`, `crucible.train.world_env.codec_capture`.
- **Behaviour.** `TokenizerId.of(tokenizer)` has a `token_map` part (vocabulary and special tokens: what ids mean) and a
  `render` part (chat template, fast-tokenizer backend: how text becomes ids); two tokenizers with one vocabulary and
  another template differ in `render` only (the audit's p_tb02; `vocab_signature` called them equal). Training records
  carry `capture` (engine classes, the pool's log-prob mode, the trainer's tokenizer, the codec's tokenizer and
  template arguments), so a restart under another engine, mode or tokenizer shows in the records. The trainer refuses a
  batch whose records come from different token maps or another than its declared `tokenizer_id`, and records with
  `raw` log-probs unless at temperature 1 without top-p truncation. Records without `capture` are not judged.
- **Formats.** `crucible.rollout/v1.1` amended (unpushed): optional `capture` (left out when unset, so a record without
  it keeps its bytes).
- **Tests.** `test_capture.py` (4): one vocabulary, two templates differ in render only; the trainer's three refusals;
  the runtime records `TorchEngine` / `processed` and the trainer's tokenizer; a codec's capture and an old record read.
  The 58 suites that import `crucible.train` plus this one: 634 passed, the known `test_cluster` failure.
- **Merge notes.** A trainer that knows its tokenizer should set `tokenizer_id = TokenizerId.of(tok)`. An engine that
  reports `capabilities()["logprobs"] == "raw"` now has its records refused at temperatures other than 1.

## 24. R-20: documents are stored as written; NFC is for identity only

- **Item.** R-20 (new; NG-4). PATTERN from NeMo-Gym `token_id_capture/fingerprint.py:118-123` (Apache-2.0);
  THIRD_PARTY.md row added.
- **Files.** `src/crucible/core/canonical.py` (`dumps_stored`; one `_form` for both forms; NFC key collisions refused),
  `src/crucible/core/schema/base.py` (`CrucibleModel.stored_json`), `src/crucible/core/store/__init__.py` (`put_doc`
  stores `stored_json`), `tests/unit/test_lossless_store.py` (new), `THIRD_PARTY.md`.
- **Public names.** `crucible.core.canonical.dumps_stored`, `CrucibleModel.stored_json()`.
- **Behaviour.** `put_doc` stores sorted, compact JSON with the text as given; `get_doc` returns decomposed (NFD) text as
  it was written (it came back NFC). Identities are unchanged: `content_hash` still hashes the NFC form, and for text
  already in NFC the stored bytes, and so the store key, are what they were. A document with NFD text gets a new store
  key (the hash of its own bytes). An object with two keys that are one key under NFC is refused by `content_hash`
  and `put_doc` (the canonical form kept one silently). In the only `put_doc` caller, `TraceStore`, a signal on an NFD
  event (named by its raw text) is now carried across a re-ingest (it was dropped: checked against the old storage).
- **Formats.** Stored documents with NFD text keep it (stores written before hold the NFC form; they still read).
- **Tests.** `test_lossless_store.py` (3): NFD round-trip with one identity and the old key for NFC text; NFC key
  collisions refused; the NFD signal survives a re-ingest. Core, store, tag, connector, M3 and interop gate suites: 227
  passed.
- **Merge notes.** Code that expected `get_doc` to return NFC text must normalize itself. A document key equals
  `"sha256:" + content_hash()` only for NFC text.

## 25. R-21: content ids name the scheme that made them

- **Item.** R-21 (carry-over AUD-D13, SCOUT-4; merges AE-7, NG-5, SW-2, TC-8). PATTERN from agentenv
  `bundle/ledger.py:144-157` (Apache-2.0); THIRD_PARTY.md row added.
- **Files.** `src/crucible/core/ids.py` (`format_content_id`, `parse_content_id`), `src/crucible/core/canonical.py`
  (`SCHEMES`: the serialization each scheme hashed, `{1: dumps}`), `src/crucible/core/schema/base.py` (`content_hash`
  through `SCHEMES[HASH_SCHEME]`, `matches_id`), `src/crucible/world/world.py` (`restore` checks the ref's world id by its
  own scheme), `tests/unit/test_content_ids.py` (new), `tests/unit/test_core_audit.py` (the golden test's remedy).
- **Public names.** `crucible.core.ids.{format_content_id, parse_content_id}`, `crucible.core.canonical.SCHEMES`,
  `CrucibleModel.matches_id(content_id)`.
- **Behaviour.** Every id made so far is scheme 1 and unchanged (the bare hex digest). A later scheme n writes
  `c<n>:<hex>`; `matches_id` recomputes an id under the scheme it names, so after a bump a stored WorldRef still
  restores (before, `HASH_SCHEME` entered no hash: the golden test's remedy "bump HASH_SCHEME" changed nothing and any
  canonical change broke `World.restore` of every stored ref). An id of an unknown scheme is refused by name. The
  golden test now says what to do: keep the old serialization in `SCHEMES`, add the new one, bump `HASH_SCHEME`.
- **Formats.** none today (scheme 1 ids are what they were).
- **Tests.** `test_content_ids.py` (2): id forms and refusals; a WorldRef from before a simulated scheme bump restores
  into a world whose id is `c2:...`, a wrong id is refused. World, core, runtime and format suites: 199 passed.
- **Merge notes.** Code that compares a stored content id with `x.content_hash()` should use `x.matches_id(stored)`.
  Changing `core.canonical.dumps` now means registering a new scheme (the golden test fails until then).
- **Fix (found by the final full run).** `World.restore` recomputed the id from the live spec, which a World may adjust
  once running (the browser World does), so `test_website_browser` refused its own ref ("belongs to world dfcbe07f801b,
  not dfcbe07f801b"). It now accepts the id taken at construction and checks any other against the spec as the World
  was built (`core.schema.base.matches_content_id` over a dump kept at construction). Test:
  `test_a_world_whose_spec_changed_after_it_was_built_restores_its_own_refs`.

## 26. R-22: store protocols, verified reads, a conformance kit

- **Item.** R-22 (carry-over AUD-F5; merges AE-8, NG-11; corrects `core/store/__init__.py:15-16`, which promised a
  replaceable interface that was never written down). PATTERN from agentenv `tst/store/conformance.py:390-450`
  (Apache-2.0); THIRD_PARTY.md row added.
- **Files.** `src/crucible/core/store/protocols.py` (new: `BlobStore`, `DocIndex`), `src/crucible/core/store/conformance.py`
  (new: `check`, `check_blobs`, `check_docs`, `check_races`), `src/crucible/core/store/__init__.py` (`Store.backend`,
  `verified_path`, `open_blob`, `export`), `src/crucible/world/state/base.py` (`_push_from_store`), `src/crucible/world/custody.py`
  (`_members`), `tests/unit/test_store_conformance.py` (new), `docs/modules/M00-core.md`, `THIRD_PARTY.md`.
- **Public names.** `crucible.core.store.protocols.{BlobStore, DocIndex}`, `crucible.core.store.conformance.check`,
  `Store.backend`, `Store.verified_path`, `Store.open_blob`, `Store.export`.
- **Behaviour.** The four readers outside core that took `blob_path` and read the file unchecked (World state pushes to
  a local or remote exec and their copies, custody's snapshot reader) now go through reads verified against the key: a
  corrupt payload is refused instead of restored into a World. `blob_path` stays the unverified location for core and
  its tests. Any other backend passes the kit (round trips, idempotent keys, corruption refused by every read, an
  export that fails leaves no copy, documents and tags, two clients racing puts and tag moves).
- **Formats.** none.
- **Tests.** `test_store_conformance.py` (2): the local store passes the kit; no `.blob_path(` outside core. World,
  store, custody and runtime suites: 192 passed, 3 skipped.
- **Merge notes.** New code reading a stored blob uses `get_bytes` / `open_blob` / `export` / `verified_path`, never
  `blob_path` (a test fails otherwise).

## 27. R-23: layered crucible.toml with a per-key scope policy

- **Item.** R-23 (carry-over IMP-M0-7; merges PRIME-C4, H4, NG-9, AE-12). PATTERN from Prime Agent
  `pa-core/src/settings/manager.rs:587-600` (MIT); THIRD_PARTY.md row added. Decisions: `$CRUCIBLE_CONFIG` stays the one
  file used (the operator's explicit choice, as before); a project file may set operator-only settings where the user
  file is silent (this repository's own `crucible.toml` defines the OpenRouter provider and the lab GPU host), may lower
  the budget, never raise it; a parent file is refused with an error, not ignored.
- **Files.** `src/crucible/core/config.py` (`load_config`, `project_config_path`, `user_config_path`, `OPERATOR_ONLY`,
  `ConfigError`, `set_overrides`, `Config.sources`; `_discover` removed), `src/crucible/cli/main.py` (root `--set
  KEY=VALUE`, `crucible config`), `src/crucible/core/env.py` (the `CRUCIBLE_CONFIG` description),
  `tests/unit/test_config_layers.py` (new), `docs/modules/M00-core.md`, `THIRD_PARTY.md`.
- **Public names.** `crucible.core.config.{load_config, project_config_path, user_config_path, set_overrides,
  ConfigError, OPERATOR_ONLY}`, `Config.sources`; CLI `crucible --set KEY=VALUE ...`, `crucible config`.
- **Behaviour.**
  - Layers, each over the one before: defaults, the user file, the project file (the nearest `crucible.toml` from the
    working directory up to the root of its git work tree; only `./crucible.toml` outside a repository), `--set`.
    Tables merge key by key: the user's prices and GPU hosts survive a project file (they were all dropped: the first
    file found won).
  - A `crucible.toml` above the working directory outside its repository raises `ConfigError` naming it and the remedy
    (`CRUCIBLE_CONFIG`, or the user file). Before, it was used, and could send the user's key to its own `base_url`.
  - Operator-only settings (`llm_providers.*.base_url`, `.api_key`, `.headers`, `gpu_hosts`, `llm_budget_usd`): a project
    file that changes one the user file sets, or raises the budget above the layers below it (default 1.0), is refused.
  - `Config.sources` lists the layers read (`user:`, `project:`, `explicit:`, `set:<key>`); `crucible config` prints them
    with the merged, redacted settings. `root` is the directory of the last file read.
- **Formats.** none (`crucible.toml` keys unchanged).
- **Tests.** `test_config_layers.py` (7): parent files refused in and out of a repository, the repository's file found
  from a subdirectory; user and project merge; the audit's redirect, a budget raise and a GPU-host override refused; a
  project may lower the budget and add its own provider; `--set` on top and `CRUCIBLE_CONFIG` alone. Config, CLI, core,
  env, secret and LLM suites: 142 passed.
- **Merge notes.** A `crucible.toml` kept in a parent directory outside a repository must move to
  `~/.config/crucible/crucible.toml` or be named by `CRUCIBLE_CONFIG`. Code that called `core.config._discover` uses
  `project_config_path` / `load_config`. A project that set a higher `llm_budget_usd` must set it in the user file or
  with `--set`.

## 28. R-24: forgetting a blob and indexing one are one transaction

- **Item.** R-24 (new; NG-6; TA-2 and TC-7 narrowed to the race; corrects `core/store/__init__.py:9`, which called
  writes safe across processes). PATTERN from agentenv `sqlite_document_store.py:354-362` (writes under `BEGIN
  IMMEDIATE`, Apache-2.0); this pattern is the store's existing agentenv row in THIRD_PARTY.md.
- **Files.** `src/crucible/core/store/__init__.py` (`forget_doc`, `put_doc`, `tag`, `_settle_forgotten`, `_alive`),
  `tests/unit/test_store_forget.py` (new).
- **Behaviour.** `forget_doc` deletes the rows, decides whether the blob is still used and renames it aside inside one
  `BEGIN IMMEDIATE` transaction, and removes the file after the commit. `put_doc` writes its rows in such a transaction
  too and writes the blob again if it was forgotten since `put_bytes`; `tag` re-checks the blob inside its
  transaction and refuses a forgotten one. Before, a put or tag in another process between the forget's commit and
  its unlink left an index row (or a tag) whose blob was deleted. A blob a crashed process set aside is put back on the
  next open if its rows remain (the forget never committed), else removed.
- **Formats.** none (`staging/forgotten-*` files exist only between a forget's commit and its unlink).
- **Tests.** `test_store_forget.py` (4): a put and a tag from a second connection landing while a forget holds its
  transaction (the put keeps its blob; the tag is refused, not left dangling); set-aside blobs of a dead process
  settled on open; 1.5 s of racing forgets and puts end with every row's blob present. Store, tag, connector, accept,
  world and content-id suites: 243 passed.
- **Merge notes.** `put_doc` now takes the index's write lock for its rows (it was an implicit transaction already).

## 29. R-25: every directory the store creates is synced into its parent

- **Item.** R-25 (new; SW-7). PORT from Karotte `durable_write.py:10-35` (MIT, `LICENSES/MIT-karotte.txt` already
  shipped); THIRD_PARTY.md row added.
- **Files.** `src/crucible/core/store/__init__.py` (`_mkdir_durable`, used for the root, `blobs/`, `staging/`, each new
  `blobs/<ab>/` and a settled blob's directory), `tests/unit/test_store_durable.py` (new), `THIRD_PARTY.md`.
- **Behaviour.** A new `blobs/<ab>/` is fsynced into `blobs/` before its first blob is renamed in (only the blob's own
  directory was synced, so after a power loss the directory, and the blob indexed in it, could be gone). The store root's
  entry is synced when it is created. An existing directory costs nothing extra.
- **Formats.** none.
- **Tests.** `test_store_durable.py` (1): a recording `_fsync_dir` sees the root, `blobs/` and the new shard synced, and
  no `blobs/` sync for a put into an existing shard. Store suites: pass.
- **Merge notes.** none.

## 30. R-26: media types are canonical and agree with the media's kind

- **Item.** R-26 (new; H9). PORT from Harbor `models/trajectories/content.py:10-44` (Apache-2.0, `LICENSES/Apache-2.0.txt`);
  THIRD_PARTY.md row added.
- **Files.** `src/crucible/core/schema/actions.py` (`MIME_ALIASES`, `canonical_mime`; `MediaRef` and `ImageRef`
  validate), `src/crucible/actors/policy.py` (`media_parts`: the audio format from the canonical type),
  `tests/unit/test_media_mime.py` (new), `THIRD_PARTY.md`.
- **Public names.** `crucible.core.schema.actions.{MIME_ALIASES, canonical_mime}`.
- **Behaviour.** A `MediaRef` type is lower-cased, its parameters dropped and its aliases resolved (`audio/wave` ->
  `audio/wav`, `audio/mp3` -> `audio/mpeg`, `image/jpg` -> `image/jpeg`), and its major type must equal the kind; an
  `ImageRef` must be an image type. Before, an image tagged `audio/wav` went out as an `image_url` part, and `audio/wave`
  as the provider audio format `wave`. `image/*` (a remote URL of unknown type) is kept.
- **Formats.** none (stored media types from producers already canonical read unchanged; an alias reads normalized).
- **Tests.** `test_media_mime.py` (11): six spellings normalized, four disagreements refused, audio formats `wav`/`mp3`.
  Suites using images or audio: 162 passed, the known `test_agentenv_mcp_env` failure.
- **Merge notes.** A producer that tagged media with a type of another kind now fails validation.

## 31. R-27: failure-kind names under fullmatch; an unknown kind read warns once

- **Item.** R-27 (carry-over AUD-D17; NG-1). PORT of NeMo-Gym `failure_kinds.py:52-60` (`_NAMESPACED_PATTERN`,
  `fullmatch`; Apache-2.0), the module this vocabulary already ports (its THIRD_PARTY.md row).
- **Files.** `src/crucible/core/schema/run.py` (`_NAMESPACED`, `register_failure_kind`, `default_blame`, `_WARNED`),
  `tests/unit/test_failure_kinds.py` (new).
- **Behaviour.** A pack kind is `<letter>[a-z0-9_]*:<letter>[a-z0-9_]*`, matched whole: `pack:kind\n` (`$` matched
  before the newline) and `9probe:_x` no longer register; a non-string is refused. An unknown kind met on read (a
  misspelling, a pack not installed here) still reads with blame `unknown` (masked), and is logged once per kind and
  process instead of silently.
- **Formats.** none.
- **Tests.** `test_failure_kinds.py` (10): eight malformed names refused, a well-formed one registers, an unknown read
  logs once. Core, runtime, run-lock, task-digest, usage and format suites: 222 passed. No pack registers a kind today.
- **Merge notes.** A pack kind starting with a digit or underscore must be renamed.

## 32. R-28: registry kinds are declared

- **Item.** R-28 (new; C4; corrects `core/registry.py:8-9`, whose list of kinds was out of date). PATTERN from OpenEnv
  `validation/graders/__init__.py:91-111` (BSD-3-Clause); THIRD_PARTY.md row added.
- **Files.** `src/crucible/core/registry.py` (`KINDS`, `Registry.kinds`, `declare_kind`, `_declared` in `register`, `get`,
  `has`, `names`; the plugin merge refuses an undeclared kind), `tests/unit/test_registry_kinds.py` (new),
  `tests/unit/test_core.py`, `tests/unit/test_core_audit.py` (their mechanism tests declare `thing` / `grader`),
  `THIRD_PARTY.md`.
- **Public names.** `crucible.core.registry.KINDS`, `Registry.declare_kind(kind, reader)`, `Registry.kinds`.
- **Behaviour.** Twelve kinds are declared, each with its reader (`segmenter` and `replayer`, read by `workflow`, were
  never declared or registered). Registering, reading or listing another kind raises `RegistryError` naming it, a near
  declared kind and the list; a plugin claiming one has that claim refused and reported in `failures` (its other
  claims register). Before, any kind registered: `segmentor` was accepted and never read.
- **Formats.** none.
- **Tests.** `test_registry_kinds.py` (3): a misspelled kind refused with the hint; a plugin's undeclared claim refused
  and reported; a scan of `src/` and `packs/` finds every kind used declared, and every installed pack loads without an
  unknown kind. `crucible registry --check`: 0 broken, 0 failed plugins. The 98 non-training suites that touch the
  registry, workflow, segmenting, replay or connectors: 1189 passed, 4 failures that fail at HEAD too (entry 19).
- **Merge notes.** A new registry kind must be added to `KINDS` (with its reader); tests of the registry mechanism use
  `declare_kind`.

## 33. R-29: `.env` files are read with python-dotenv's grammar

- **Item.** R-29 (new; SW-5; DEPEND changed to PORT, since ARCHITECTURE.md limits new dependencies). PORT (vendored
  verbatim) of python-dotenv 1.2.4 `dotenv/parser.py` (BSD-3-Clause); `LICENSES/BSD-3-Clause-python-dotenv.txt` and a
  THIRD_PARTY.md row added.
- **Files.** `src/crucible/_vendor/dotenv_parser.py` (new), `src/crucible/core/config.py` (`_dotenv`),
  `LICENSES/BSD-3-Clause-python-dotenv.txt` (new), `tests/unit/test_dotenv.py` (new), `THIRD_PARTY.md`.
- **Behaviour.** The nearest `.env` is parsed as python-dotenv parses it (quotes, escapes, `export`, inline comments,
  multi-line quoted values, a later binding of a key winning), with no interpolation. Before, a line was split at the
  first `=` and quotes stripped from the ends: `KEY=sk-...  # note` kept the comment, and the secret failed with 401.
- **Formats.** none.
- **Tests.** `test_dotenv.py` (2): the audit's fixture (plus empty, `#`-leading and multi-line values) reads as
  python-dotenv's own `dotenv_values` reads it; a resolved secret has no trailing comment. Config, core, secret, LLM and
  licence suites: pass.
- **Merge notes.** A `.env` value that relied on the old reading (a trailing `# ...` kept, or unbalanced quotes) now
  reads as python-dotenv does.

## 34. R-30: `core.ids.manifest_hash` removed

- **Item.** R-30 (carry-over AUD-D18; H10; corrects `core/ids.py:1-8` and the THIRD_PARTY.md row that called it the task
  hash). DEPEND: a Harbor task's id is Harbor's own hash (P18).
- **Files.** `src/crucible/core/ids.py` (`manifest_hash` removed; the module and `tree_digest` docstrings),
  `tests/unit/test_core.py` (its test kept for `tree_manifest_hash` only), `THIRD_PARTY.md`.
- **Behaviour.** none: `manifest_hash` had no caller outside its test. The docs now say a Harbor task's id comes from
  Harbor (`Packager.compute_content_hash` through `interop.harbor`), other task directories from `tree_digest`.
- **Formats.** none.
- **Tests.** Core, task-digest, state-adapter and licence suites: 39 passed.
- **Merge notes.** Code calling `crucible.core.ids.manifest_hash` uses `tree_digest` (a directory) or Harbor's hash (a
  Harbor task).

## 35. R-31: the unread `Config.backends` table is gone

- **Item.** R-31 (new; AE-10). The audit cites agentenv `config/describe.py:261-290` (warns on unread keys); dropping the
  field restores `extra="forbid"` for that table instead, so no code is taken and there is no THIRD_PARTY.md row.
- **Files.** `src/crucible/core/config.py` (`Config.backends` removed), `tests/unit/test_config_backends.py` (new).
- **Behaviour.** A `[backends]` table in a `crucible.toml` (e.g. `[backends.harbor] docker_host = ...`) is refused by
  name; before it was accepted and read by nothing. Docker hosts belong to `[gpu_hosts.<name>]`.
- **Formats.** `crucible.toml`: the `backends` key is no longer valid.
- **Tests.** `test_config_backends.py` (1). Config and core suites: 25 passed; `crucible config` loads this repository's
  file.
- **Merge notes.** A configuration that set `[backends]` must drop it (it never had an effect).

## 36. A16-R1: a credential goes only to an endpoint the operator paired it with

- **Item.** A16-R1 (carry-over SCOUT16-5, CHG-11, LED16-1, LED16-2; corrects R-23, entry 27). PATTERN from agentenv
  `config/model.py:149-153` (a key scoped to its endpoint, Apache-2.0); no code taken.
- **Files.** `src/crucible/core/config.py` (`load_config`, `_check_project`, new `_number`, `_endpoint`,
  `_credential_pairs`, `_check_credentials`; module doc), `tests/unit/test_config_layers.py`.
- **Public names.** none new; `ConfigError` gains a case.
- **Behaviour.** After the layers merge, every (endpoint, `env:` reference) pair a provider would send (its `api_key`
  unless a placeholder, and every `env:` header) must also be in the operator's view: the built-in defaults plus the
  user file, `$CRUCIBLE_CONFIG` and `--set`, without the project file. Otherwise `ConfigError` names the project file,
  the reference, the endpoint and the provider (never a value). The per-key check of R-23 compared a project file only
  with the user file, so with no user file a cloned `crucible.toml` could point openrouter at its own host with
  `env:AWS_SECRET_ACCESS_KEY`, or add `[llm_providers.mirror]` with `env:OPENROUTER_API_KEY` and route
  `models.policy-default` to it. The budget check now compares validated numbers: `llm_budget_usd = "500"` was skipped
  and then read as 500.0. A project file may still add a keyless local engine, lower the budget, or restate a pairing
  the operator has (this repository's `crucible.toml` repeats the default openrouter one and loads unchanged).
- **Formats.** `crucible.toml`: a project file that gives a provider its own credentialed endpoint is refused; declare
  that provider in `~/.config/crucible/crucible.toml` or with `--set`.
- **Tests.** `test_a_project_file_cannot_pair_a_credential_with_its_own_endpoint` (4: no user file, own provider,
  secret header, string budget; all 4 fail on the previous code), `test_a_project_file_may_restate_the_operators_pairings`.
  Config, core and dotenv suites: 130 passed.
- **Merge notes.** `load_config` now validates the operator layers a second time (without the project file).

## 37. A16-R2: provider settings carry no credentials

- **Item.** A16-R2 (carry-over CHG-2, LED16-3, the gap R-8 left, entry 9). PATTERN from agentenv `config/model.py:19-32`
  (call keywords a params table may not set, Apache-2.0); no code taken. Narrowed: the audit's extra masking in
  `_redact` is not needed once the settings refuse credentials.
- **Files.** `src/crucible/core/config.py` (`CREDENTIAL_NAME`, `_CLIENT_OWNED`, `LLMProvider._no_userinfo`,
  `LLMProvider._settings_carry_no_credentials`), `tests/unit/test_provider_settings.py` (new),
  `tests/unit/test_core_audit.py` (its redaction test now expects the refusal).
- **Public names.** `crucible.core.config.CREDENTIAL_NAME` (a credential-named key, matched on its last word).
- **Behaviour.** `LLMProvider` refuses, naming the setting and never its value: `extra_headers` and `extra_query` in
  `request_defaults` (headers go in `headers`, secret ones as `env:NAME`); the client-owned `model`, `messages`,
  `tools`, `stream` and `timeout` there or in its `extra_body`; any credential-named leaf (`api_key`, `auth_token`,
  `client_secret`, `X-Api-Key`, ...) in `request_defaults` or `template_kwargs`; a `base_url` with userinfo
  (`user@` or `user:pw@`). Before, a project's `request_defaults.extra_headers.Authorization` replaced the operator's
  key on every call, and operator literals (gateway tokens, `extra_body.api_key`, userinfo) reached `lock.json` and
  every `Turn.sampling.extra`. `max_tokens`, `return_token_ids` and other engine settings stay allowed.
- **Formats.** `crucible.toml`: those settings are refused.
- **Tests.** `test_provider_settings.py` (10). LLM, actor, lock, config and core suites: 225 passed.
- **Merge notes.** A provider that sent a header through `request_defaults.extra_headers` moves it to `headers`.

## 38. A16-R3: a blob write is complete or fails

- **Item.** A16-R3 (carry-over SCOUT16-2, NEW-1; corrects AUD-D4 and the store doc's "hashed from the staged bytes").
  PATTERN from slime `data/checkpoint.py:104-112,163-166` (a size mismatch is refused, Apache-2.0); no code taken.
- **Files.** `src/crucible/core/store/__init__.py` (new `_write_all`, `_synced`; `put_bytes`, `put_file`; module doc),
  `tests/unit/test_store_short_write.py` (new), `tests/unit/test_core_audit.py` (the D4 test rewritten),
  `tests/unit/test_config_layers.py` (a lint fix to entry 36's test).
- **Public names.** none.
- **Behaviour.** `put_bytes` and `put_file` write every byte (a loop over `os.write`, which may write fewer bytes than
  asked) and, after fsync, check that the staged file holds exactly the hashed count before it takes the key; otherwise
  the put raises and the staged file is removed, so no blob and no index row are written. Before, the count was
  dropped: under a 1 MiB file-size limit (the stand-in for a full disk) a 2 MiB `put_bytes` kept 1,048,576 bytes under
  the whole blob's key, `put_doc` indexed it, and every read then raised "corrupt"; a snapshot, which deletes its
  source after `put_file`, could not be restored.
- **Formats.** none.
- **Tests.** `test_store_short_write.py` (2: a half-writing `os.write` still stores every byte; under `RLIMIT_FSIZE` all
  three puts raise and leave no blob, row or staged file; both fail on the previous code). The D4 test patched
  `shutil.copyfile`, which `put_file` never calls; it now rewrites the source as the store opens it. Store, tag,
  conformance, core and config suites: 137 passed.
- **Merge notes.** A put that hits a full disk now raises `OSError` (or `StoreError` on a size mismatch) where it used to
  return a key.

## 39. A16-R4: an existing store open to other users is tightened on open

- **Item.** A16-R4 (carry-over of R-6's gap, entry 7: new roots only). PATTERN from Karotte `protected_store.py:89-120`
  (re-applies 0700 on every open, MIT); no code taken. Narrowed: no `store_shared` opt-out, since `TraceStore` already
  forced 0700 on every store it opened, so no store could be shared through group permissions.
- **Files.** `src/crucible/core/store/__init__.py` (new `_tighten`, called by `_make_private_root`),
  `src/crucible/connectors/trace_store.py` (its own chmod removed), `tests/unit/test_core_audit.py`.
- **Public names.** none.
- **Behaviour.** Opening a store whose root already exists and belongs to this user makes the root 0700 and
  `index.sqlite*` 0600 when any group or other bit is set, and logs one warning naming the old modes. A root owned by
  someone else is left alone, and no `.gitignore` is written into an existing directory. Before, roots made before R-6
  stayed 0755 with a 0644 index (this workspace's `data/store` is), readable by every user on a shared host, unless a
  `TraceStore` happened to open them.
- **Formats.** none.
- **Tests.** `test_an_existing_store_open_to_others_is_tightened_on_open` replaces
  `test_an_existing_store_root_keeps_its_mode`, which asserted the old rule. Store, tag, conformance, trace-store and
  core-audit suites: 210 passed.
- **Merge notes.** Code that expected a store root's mode to be left alone (none found) now sees 0700.

## 40. A16-R5: the `.env` search is bounded and owner-checked

- **Item.** A16-R5 (new: R-29, entry 33, changed only the parser). PATTERN from OpenEnv `auto/_discovery.py:340-449`
  (only files the user owns, BSD-3-Clause) and mini-swe-agent `src/minisweagent/__init__.py:26-36` (`.env` from the
  user directory, MIT); no code taken.
- **Files.** `src/crucible/core/config.py` (`_dotenv`; new `_project_dirs`, shared with `project_config_path`),
  `tests/unit/test_dotenv.py`.
- **Public names.** none.
- **Behaviour.** An `env:NAME` missing from the environment is read from the nearest `.env` between the working
  directory and the root of its git work tree (the working directory alone outside a repository), else from
  `~/.config/crucible/.env`. A `.env` there that another user owns or that is world-writable raises `ConfigError`
  naming it. Before, the search walked up to `/` unchecked: a job in `/tmp/job` took its keys from another user's
  `/tmp/.env`. Group-writable files are accepted (Linux's default umask 002 makes them).
- **Formats.** none.
- **Tests.** `test_dot_env_search_stops_at_the_repository_and_skips_other_users_files` (fails on the previous code).
  Config, dotenv, provider and core-audit suites: 124 passed.
- **Merge notes.** A `.env` kept above the repository's root (no case in this workspace) moves into the repository or
  `~/.config/crucible/.env`.

## 41. A16-R6: a refused setting never echoes its value

- **Item.** A16-R6 (new, continues R-8, entry 9). DEPEND on pydantic's `hide_input_in_errors`; PATTERN from Harbor
  `models/asp.py:12-13` (credential models hide their inputs, Apache-2.0).
- **Files.** `src/crucible/core/config.py` (`LLMProvider`, `GpuHost` and `Config` set `hide_input_in_errors`; new
  `_validated` used by `load_config`), `tests/unit/test_config_layers.py`, `tests/unit/test_provider_settings.py`.
- **Public names.** none.
- **Behaviour.** A configuration the models refuse raises `ConfigError` naming the layers read (`project:<path>`, ...)
  and each refused setting with the reason, never the value. Before, the pydantic error carried `input_value=` with the
  whole value: `crucible config` printed a prefixless literal key in full (the scrubber cannot know it) and named no
  file. Constructing the models directly still raises `ValidationError`, now without the input.
- **Formats.** none.
- **Tests.** `test_a_refused_setting_never_echoes_its_value`; `test_provider_settings_refuse_credentials` now checks the
  whole message. Config, dotenv, provider, core and core-audit suites: 142 passed.
- **Merge notes.** `load_config` raises `ConfigError` (a `ValueError`) where it raised `ValidationError`; no caller
  caught the latter.

## 42. A16-R8: two distributions with one entry-point name both load

- **Item.** A16-R8 (carry-over SCOUT16-4; corrects R-13, entry 17, whose port dropped the grouping by distribution).
  PATTERN from agentenv `plugins/_discovery.py:32-42` (claims grouped by distribution, Apache-2.0); no code taken.
  A16-R7 (old run records without a version) is not implemented: the audit found no such records anywhere.
- **Files.** `src/crucible/core/registry.py` (`Registry.load_plugins`), `tests/unit/test_core_audit.py`.
- **Public names.** none. `Registry.failures` keys a plugin as `<entry point> (<dist>==<version>)` when two
  distributions share its entry-point name; otherwise by the name alone, as before.
- **Behaviour.** Before, staging was keyed by the entry-point name: two packs that both exposed `pack` shared one slot,
  the first one's registrations vanished, `failures` stayed `{}` and `crucible registry --check` exited 0. Now both
  load; a name both claim is still withheld and reported under each distribution.
- **Formats.** none.
- **Tests.** `test_two_distributions_with_one_entry_point_name_both_load` (fails on the previous code). Registry, core
  and core-audit suites: 119 passed; `crucible registry --check`: 0 broken, 0 failed plugins.
- **Merge notes.** none.

## 43. A16-R11: every ported MIT or BSD row ships its notice

- **Item.** A16-R11 (corrects R-9, entry 10, which checked only files already named). PORT obligation: MIT and BSD
  require the copyright notice with every copy.
- **Files.** `LICENSES/MIT-verifiers.txt` and `LICENSES/MIT-mini-swe-agent.txt` (new, copied verbatim from the local
  clones' `LICENSE` and `LICENSE.md`), `THIRD_PARTY.md` (seven licence cells name their file),
  `tests/unit/test_third_party.py`.
- **Public names.** none.
- **Behaviour.** The verifiers ports (`verify/answers.py`, `actors/llm.py`) and the mini-swe-agent port
  (`actors/harness.py`) had no notice file; four Karotte and OpenEnv rows did not name theirs. Every MIT, BSD or ISC
  row in the two "Code ported" tables now names a `LICENSES/` file holding the upstream copyright line (pattern-only
  rows are exempt: nothing is copied).
- **Formats.** none.
- **Tests.** `test_every_ported_mit_or_bsd_row_names_its_notice` (flags rows 21, 30, 31, 35, 38, 43 and 47 of the
  previous `THIRD_PARTY.md`); the licence suite: 3 passed.
- **Merge notes.** A new MIT, BSD or ISC port row must name its notice file. The wheel still ships no `LICENSES/`
  (deferred in R-9).

## 44. A16-R17: a forgotten blob is settled by its lock, and the forget is durable

- **Item.** A16-R17 (corrects R-24, entry 28). PATTERN from agentenv `store/object_store/local/store.py:419-431`
  (sweeps only files it can lock, Apache-2.0); no code taken.
- **Files.** `src/crucible/core/store/__init__.py` (`forget_doc`, `_settle_forgotten`, new `_settle_one` and
  `_lock_free`), `tests/unit/test_store_forget.py`.
- **Public names.** none.
- **Behaviour.** `forget_doc` holds an exclusive file lock on the blob from before it is renamed aside until it is
  unlinked, fsyncs the blob's and the staging directory after the rename (before the rows' removal commits) and the
  staging directory after the unlink. Opening a store settles a set-aside file only when it can take that lock (the pid
  in the name decides only where flock does not exist). Before, the pid decided: a crashed forget whose name held a live
  pid, its own included (a restarted container's process is pid 1 again), was never settled, and every read of the
  document raised "missing blob"; and the rename and unlink were not synced, so after a power loss forgotten content
  could reappear.
- **Formats.** none.
- **Tests.** `test_a_set_aside_named_with_a_live_pid_is_settled_unless_its_lock_is_held` (fails on the previous code).
  Store, forget, tag, conformance and core-audit suites: 122 passed.
- **Merge notes.** none.

## 45. A16 gate (LED16-6): the capture test skips without torch

- **Item.** The one failing M0 test in the 2026-10-16 audit's gate run (§1; ledger LED16-6), from R-17 (entry 21).
- **Files.** `tests/unit/test_capture.py`.
- **Public names.** none.
- **Behaviour.** `test_a_chat_codec_records_its_tokenizer_and_template_arguments_and_old_records_read` imported
  `crucible.train.world_env`, which imports torch through the engine, so it failed in an install without the `train`
  extra. It is split: the codec half skips without torch; the old-record half (core only) runs everywhere.
- **Formats.** none.
- **Tests.** With torch hidden: 2 passed, 3 skipped (the previous file: 1 failed). With torch: 5 passed.
- **Merge notes.** The test's name changed (two tests now).

## 46. A16-R10: golden vectors pin canonical JSON, scheme 1

- **Item.** A16-R10 (carry-over SCOUT16-7; a guard test). PATTERN from NeMo-Gym
  `token_id_capture/staging/conformance/__init__.py:16-41` (golden digest vectors, Apache-2.0); no code taken.
- **Files.** `tests/unit/test_canonical_vectors.py` (new), `src/crucible/core/canonical.py` (the `SCHEMES` comment).
- **Public names.** none.
- **Behaviour.** none. Every content id hashes `SCHEMES[1]`, which is the live `dumps`, and only two ids were pinned:
  switching to `ensure_ascii`, another float spelling, unsorted keys, spaced separators, or dropping the NFC or -0.0
  fold passed every test while every stored id moved. Eleven vectors (one per rule: sorting, separators, UTF-8, NFC,
  -0.0, float spelling, big integers, literals, escapes, non-string keys, tuples) pin the exact text and id, three
  refusals (NaN, ±inf, keys that are one under NFC) and the stored form's raw text. Each of those six changes fails at
  least one vector.
- **Formats.** none.
- **Tests.** `test_canonical_vectors.py` (16).
- **Merge notes.** A change that fails a vector changes every content id: keep the old function as `SCHEMES[1]` and add
  scheme 2, never edit the vectors.

## 47. A16-R9: every frozen example's reading is frozen too

- **Item.** A16-R9 (carry-over SCOUT16-8; corrects R-18, entry 22, whose freeze held the shape only; a guard test).
  PATTERN from agentenv `tst/unit/serialization/test_golden_wire_format.py:184-231` (writer bytes against goldens,
  Apache-2.0); no code taken.
- **Files.** `tests/unit/_formats.py` (new `read_file`, `render_read`; module doc), `tests/unit/test_formats.py`,
  `scripts/freeze_schemas.py` (writes the new files, never rewrites them), `schemas/*/<version>.read.json` (21 new),
  `schemas/crucible.run/v1.timings.example.json` (new: a run/v1 record in the shape the runner wrote before R-19, with
  `meta.timings`, so the v1 to v2 upcaster's moves are exercised).
- **Public names.** none.
- **Behaviour.** none in Crucible. Next to every frozen example, `<version>.read.json` holds what Crucible reads it as
  today: its content id and its stored form (for an older version, the upcaster's output). `test_every_example_reads_as_frozen`
  fails when either changes. Before, the schema snapshot let an alias rename, a dropped `exclude_if`, a field that moves
  every content id and an edited upcaster through. Checked by mutation: dropping `Sampling.extra`'s `exclude_if` fails
  both rollout readings, and editing the run upcaster fails the new v1 example (both passed the previous tests).
- **Formats.** none (test fixtures only).
- **Tests.** `test_every_example_reads_as_frozen` (21). Format and core-audit suites pass; the freeze script writes
  nothing on a second run.
- **Merge notes.** A change that fails a reading is a format change: bump the version (and add an upcaster) instead of
  rewriting the `.read.json` file.

## 48. Path dependencies must match `uv.lock`

- **Item.** none (found while triaging the 2026-10-14 M1 audit's R-15). The user asked to stay on main's versions.
- **Files.** `tests/unit/test_path_dependencies.py` (new).
- **Public names.** none.
- **Behaviour.** none in Crucible. uv records only the version a path source had when the lock was written, and
  `uv sync --frozen` installs whatever the folder holds: the local Harbor and agentenv clones had moved to 0.24.0
  (`dc933be7`) and 0.9.1324 (`015c89b`) while `uv.lock` and CI pinned 0.23.0 (`a2fafbc`) and 0.9.1301 (`041afd7`), so
  local runs tested other code than CI. The test fails when a path source's folder (its `pyproject.toml` version) or its
  installed metadata differs from the lock; extras not installed (agentenv in CI) and dynamic versions (verifiers) are
  skipped. The clones were checked out at the pinned commits (outside the repository; no file here changed), which
  also fixed `test_triggers`, a "known" failure caused by the newer agentenv.
- **Formats.** none.
- **Tests.** `test_every_path_dependency_is_the_version_uv_lock_records` (fails with Harbor's clone at `dc933be7`).
- **Merge notes.** Pulling a reference clone used as a path source now fails this test until it is checked out at
  the pinned commit again, or the lock and `THIRD_PARTY.md` are updated on purpose.

## 49. R-31 reverted: `Config.backends` is read again (M17)

- **Item.** R-31 (entry 35) undone. Main's M17 task factory (`23fdf6f`, 2026-10-16) reads `[backends.factory]` in
  `factory/models.py` (`ModelBook.from_config`: role models, solver rungs, `budget_usd_per_client`, `no_json_mode`), so
  the table is no longer unread. At the user's request M17 is respected as written.
- **Files.** `src/crucible/core/config.py` (`backends: dict[str, dict[str, Any]]` back, as on main),
  `tests/unit/test_config_backends.py` (deleted).
- **Public names.** `Config.backends` again.
- **Behaviour.** As before R-31: any `[backends.<name>]` table is accepted; only `factory` is read.
- **Formats.** `crucible.toml`: `backends` is a valid key again.
- **Tests.** `test_config_backends.py` removed; M17's `test_m17_q_models` and `test_m17_q_root` cover the table.
- **Merge notes.** none: this restores main's line.

## 50. The slime bridge's `CRUCIBLE_SLIME_*` variables are declared

- **Item.** none (R-15 follow-up). Main's slime bridge (`c7a98da`, M17 factory v2 P6) reads seven `CRUCIBLE_SLIME_*`
  variables that `KNOBS` did not declare, so `test_every_crucible_variable_in_the_code_is_declared` failed on main's CI.
- **Files.** `src/crucible/core/env.py` (7 knobs).
- **Public names.** `KNOBS` gains `CRUCIBLE_SLIME_ENV`, `_WORKDIR`, `_ROLLOUT_TIMEOUT_S`, `_AGENT_TIMEOUT_S`, `_TURN_ETA`,
  `_MAX_STALENESS`, `_WEIGHT_VERSION` (53 declared: 26 user, 8 internal, 19 test/script). The two timeouts are read
  through an f-string in `slime/generate.py:_knob`, which the scan test cannot see, so they were added by hand.
- **Behaviour.** none: the bridge still reads `os.environ` itself; the declarations carry its defaults (provider
  `crucible.factory.envs:slime_provider`, 7200 s / 3600 s, eta 0.1, tau 2) for `crucible env` and `knobs.get`.
- **Formats.** none.
- **Tests.** The scan test passes again; core, config and slime suites: 140 passed, 4 skipped (slime not installed).
- **Merge notes.** A changed slime default must be changed in both `slime/` and `KNOBS`.

## 51. M1 R-1: a lost task container is infrastructure on every exec path

- **Item.** R-1 of `docs/research/improvements/2026-10-14/M01-world.md` (must; carry-over of M13 audit B2, which fixed
  only Harbor's exec). Pattern from verifiers `v1/harness.py:183` (after a nonzero exit, `runtime.alive()` before
  blaming the harness; MIT) and our Harbor backend's `container_lost`; no code copied, no THIRD_PARTY.md row.
- **Files.** `src/crucible/world/exec.py` (`CONTAINER_GONE`, `EnvContainerLost` moved here from
  `interop/harbor/world_backend.py`; `container_lost_cause`; `DockerExec.lost`, `DockerExec._unless_lost` on `run`,
  `push_file`, `pull_file`), `world/sandbox.py` (`GVisorExec._raw`), `world/apps/base.py` (`App.call`),
  `world/world.py` (`_call_tool`), `interop/harbor/world_backend.py` (imports the moved names),
  `docs/modules/M01-world.md` (finding 17), `tests/unit/test_container_lost.py` (new).
- **Public names.** `crucible.world.exec`: `CONTAINER_GONE`, `EnvContainerLost` (now with
  `failure_kind = "crucible:env_container_lost"`), `container_lost_cause`, `DockerExec.lost`. Both names stay
  importable from `crucible.interop.harbor.world_backend`.
- **Behaviour.** A failed `docker exec` / `docker cp` whose output reads as a lost container (`No such container`,
  `is not running`, `is paused`, `error during connect`, ...) makes `DockerExec` ask the daemon on the same
  `DOCKER_HOST`: removed, paused, stopped by a `kill`/`stop`/`destroy`/`pause` event, or an unreachable daemon raise
  `EnvContainerLost`; a running container, or one whose own main process exited (the policy, or out of memory), keeps
  the plain `ExecOut`. An ordinary failure asks nothing. `App.call` re-raises a loss wrapped in a `ToolError` (the
  workspace file tools wrap every exec error), and `World._call_tool` records `crucible:env_container_lost`
  (blame infra, the sample is masked) instead of `crucible:tool_crashed` (blame task); the policy sees
  `<tool>: the environment's container was lost` (`error="transport"`, `fault="infra"`).
- **Formats.** none (the failure kind already existed).
- **Tests.** `test_container_lost.py` (7: fake-daemon cases for removed / forged message / ordinary failure /
  stopped from outside vs exited by itself / paused / unreachable / `docker cp`, the World for wrapped and direct
  losses, and a Docker-marked run that kills and removes a real container).
- **Merge notes.** M17 lanes calling `DockerExec.run` directly (`factory/lanes/windows.py`, `factory/sessions.py`) now get
  `EnvContainerLost` instead of a failed `ExecOut` when the container is gone; its `failure_kind` makes
  `factory.machine.is_infra_error` and `train.runtime` count it as infrastructure. With `factory.envs.InfraWatch` on, a
  lost container can be recorded more than once (the World per call that hits it, InfraWatch after the step); every
  record is the same masked kind.

## 52. M1 R-2: a command's output is bounded while it is read

- **Item.** R-2 of `docs/research/improvements/2026-10-14/M01-world.md` (must). Pattern from agentenv
  `providers/sandbox_providers/local_sandbox.py:349-438` (a bounded tail per stream, dropped bytes counted, return at
  exit plus 1 s; Apache-2.0) and our guard's `run` loop; no code copied, no THIRD_PARTY.md row.
- **Files.** `src/crucible/world/exec.py` (`OUT_HEAD`, `OUT_TAIL`, `EXIT_GRACE_S`, `capped_output`, `_Kept`,
  `_blocking_run` rewritten with `_drain` / `_feed`), `src/crucible/interop/harbor/world_backend.py` (`HarborExec.run`),
  `docs/modules/M01-world.md` (finding 18), `tests/unit/test_exec_output.py` (new).
- **Public names.** `crucible.world.exec`: `OUT_HEAD` (1 MiB), `OUT_TAIL` (16 MiB), `EXIT_GRACE_S` (1 s),
  `capped_output(cmd)`.
- **Behaviour.** Every command run through `_run_subprocess` (local, Docker, gVisor, SSH, and every `docker` CLI call)
  keeps at most the first 1 MiB and the last 16 MiB of stdout and of stderr; between them stands
  `...[N bytes dropped]...`. The call returns 1 s after the command exits even when a background job still holds its
  pipes (before: until they closed or the timeout); the job keeps running in the command's session, as before, until
  `kill_sessions` or the World ends. Stdin is fed from its own thread. On Harbor, every command runs as
  `{ cmd ; } 2>&1 | { head -c OUT_HEAD; tail -c OUT_TAIL; }` with the command's exit status (`PIPESTATUS[0]`), inside the
  container; no marker there, so an output that fits is byte for byte (Harbor already merged stderr into stdout). An
  image without `head`/`tail` runs the command unchanged.
- **Formats.** none.
- **Tests.** `test_exec_output.py` (7: head + tail and the exact dropped count per stream, byte-exact output that fits,
  a `run_python` result longer than the head, 300 MB through `LocalExec` adding under 100 MB of peak RSS, `sleep 30 &`
  returning in about 1 s (30 s before), large stdin and timeouts, Harbor's in-container cut with its exit status and
  staged stdin, and a Docker-marked 50 MB flood through `DockerExec`).
- **Merge notes.** Internal commands whose stdout is data (a `run_python` result, the desktop App's base64 screenshot)
  are exact up to 17 MiB per stream; past that the middle is lost. Harbor's background jobs still hold a call until
  the command's deadline (the in-container cut cannot see the command's exit).

## 53. M1 R-4: a policy's hung command is scored, not masked

- **Item.** R-4 of `docs/research/improvements/2026-10-14/M01-world.md` (must). Pattern from NeMo-Gym
  `agent_utils/process_supervisor.py:38-44` (outer budget = tool timeout + cleanup + 30 s; Apache-2.0); no code copied.
- **Files.** `src/crucible/world/apps/base.py` (`AppContext.tool_timeout_s`, `App.deadline`), `world/world.py`
  (`TOOL_GRACE_S`, `_call_tool`, the context gets the limit), `world/apps/workspace.py` and `world/apps/toolchain.py`
  (`tool_shell`), `interop/harbor/inplace.py` (`TerminalApp.tool_exec`), `core/schema/world.py` (comment),
  `docs/modules/M01-world.md` (finding 19), `tests/unit/test_tool_deadlines.py` (new).
- **Public names.** `AppContext.tool_timeout_s`, `App.deadline(requested, cap=None)`, `crucible.world.world.TOOL_GRACE_S`.
- **Behaviour.** The workspace shell, the toolchain shell and Harbor's in-place `exec` run with
  `max(1, min(asked, max_timeout_s, limits.tool_timeout_s))` seconds (the toolchain shell had no cap; a requested 0
  meant no deadline). The World waits `tool_timeout_s + 60 s` for any tool, so a shell's own deadline (and the
  transports' +15 s backstops) always come first: a policy's hung command is a scored `[TIMEOUT]`. A tool still
  running after the grace is masked as `transport_timeout`, as before, with the reason
  `<tool> hung past its own deadline (<limit>s + 60s)`. A pack whose `tool_timeout_s` is below a shell's
  `max_timeout_s` now gets shorter shell deadlines.
- **Formats.** none.
- **Tests.** `test_tool_deadlines.py` (4: both shells score a 2 s World limit's `[TIMEOUT]` for `sleep 30` asked at
  99 s (both masked before), a tool hung past its deadline masks, and `App.deadline`'s caps and floor). Shell,
  toolchain, in-place and timeout suites: 436 passed, 5 skipped.
- **Merge notes.** An episode's worst case per hung tool call grows by 60 s. Apps with long calls of their own should
  take their deadline from `App.deadline` too.

## 54. M1 R-6: sandbox containers end on their own; R-1's removed-container rule corrected

- **Item.** R-6 of `docs/research/improvements/2026-10-14/M01-world.md` (must, narrowed in triage: no lease renewal, no
  reaper, no run label). Pattern from our RLM kernel containers (`actors/rlm/kernel.py:477-596`: owner label,
  `sleep <lifetime>`); no outside code.
- **Files.** `src/crucible/world/sandbox.py` (`OWNER_LABEL`, `LIFETIME_MARGIN_S`, `owner_id`, `owner_dead`,
  `DockerSandbox.lifetime_s`, `argv`, `start`), `world/pool.py` (hygiene keeps the container's own `sleep <n>`),
  `world/exec.py` (`OUTSIDE_ACTIONS`, `DockerExec.since` / `expires_at`, `lost`, `_actions`), `runtime/perf_guard.py`
  (`sample`), `docs/modules/M01-world.md` (finding 20), `tests/unit/test_sandbox_lifetime.py` (new),
  `tests/unit/test_container_lost.py`, `tests/unit/test_sandboxes.py`.
- **Public names.** `crucible.world.sandbox`: `OWNER_LABEL` (`crucible.owner`), `LIFETIME_MARGIN_S` (900),
  `owner_id()`, `owner_dead(owner)`, `DockerSandbox.lifetime_s`; `crucible.world.exec.OUTSIDE_ACTIONS`,
  `DockerExec.since`, `DockerExec.expires_at`.
- **Behaviour.** A `DockerSandbox` (and `GVisorSandbox`) runs `sleep <SandboxSpec.timeout_s + 900>` (default 8100 s)
  instead of `sleep infinity` and is labelled `crucible.owner=<host>:<pid>`; one left by a crashed process ends and is
  removed (`--rm`) within that time. The perf guard counts a `crucible*` container unless its `crucible.owner` or
  `crucible.rlm.owner` names a dead process on this host (unlabelled and other hosts' containers count as before; a
  leftover still doing work shows in the GPU and load checks). Correction to entry 51: `--rm` deletes a container the
  moment its main process dies, so a policy that killed its own `sleep` read as "removed" and was masked. `lost()` now
  reads a removed container's events since its exec was made: died with no `kill`/`stop`/`pause` and before its
  lifetime (`expires_at`, set by the sandbox) is the episode's doing (scored); a kill or stop, a passed lifetime, or no
  evidence is lost. A stopped container past its lifetime is lost too.
- **Formats.** none (a new container label).
- **Tests.** `test_sandbox_lifetime.py` (5: the command and label, `owner_dead`, the perf guard over a fake and a real
  daemon, and a real `--rm` container whose `sleep` the policy kills: scored, not masked); `test_container_lost.py`
  (+1: removed with die / with kill / with no events / past its lifetime, stopped past its lifetime);
  `test_sandboxes.py` (`sleep 8100`). Sandbox, pool, gVisor, agentenv, perf guard, World and core suites: 231 passed,
  3 skipped, 1 known failure (`test_agentenv_mcp_env::test_card_tools_only_hides_uncarded`).
- **Merge notes.** A sandbox that must live past `timeout_s` + 15 min needs a larger `timeout_s` (up to 24 h). The pool
  hands out a warm sandbox as it is, so one that waited in the pool has less than its full lifetime left (no pack
  uses the pool). Docker keeps a bounded event log: on a very busy host a removed container's `die` may have rolled
  off, which counts as lost (masked), never as the policy's.

## 55. M1 R-9: `World.close` finishes its teardown, bounded

- **Item.** R-9 of `docs/research/improvements/2026-10-14/M01-world.md` (must). Ported from verifiers v1
  `utils/aio.py:10-31` (`run_shielded`, MIT, e22533430): THIRD_PARTY.md row, `LICENSES/MIT-verifiers.txt` (already shipped).
- **Files.** `src/crucible/world/world.py` (`APP_STOP_S`, `TEARDOWN_S`, `_to_the_end`, `World._closing`, `close`,
  `_close`), `THIRD_PARTY.md` (row), `docs/modules/M01-world.md` (finding 21), `tests/unit/test_world_close.py` (new).
- **Public names.** `crucible.world.world.APP_STOP_S` (150 s), `TEARDOWN_S` (600 s).
- **Behaviour.** `close()` starts the teardown once, as a task, and awaits it to its end even when the caller is
  cancelled (repeated cancels absorbed), then re-raises the first cancel; a later or concurrent `close()` awaits the
  same task. The teardown stops the Apps in reverse start order (a browser before the sites it opens; was start
  order), each within `APP_STOP_S`, then, in a `finally`, the backend's teardown within `TEARDOWN_S`, removes an owned
  workdir and releases the serial slot; a timeout or error is logged and the teardown goes on. `closed` turns true
  when it finished (it was set before anything ran, so a cancelled close was never retried).
- **Formats.** none.
- **Tests.** `test_world_close.py` (3: a close cancelled during an App's stop still stops every App and tears down once,
  and the caller gets its cancel; a hung stop is cut off and a failing teardown never raises; concurrent closes tear
  down once, in reverse order). All 3 fail on the previous code.
- **Merge notes.** A close can now take up to `APP_STOP_S` per App plus `TEARDOWN_S` and cannot be cut short by a cancel;
  it never hangs past those bounds. An App whose stop needs more than 150 s is cut off: raise `APP_STOP_S` for it.

## 56. M1 R-11: a grade never blames the policy for an outage

- **Item.** R-11 of `docs/research/improvements/2026-10-14/M01-world.md` (must). Pattern from NeMo-Gym (sandbox
  failures are error types, never rewards; Apache-2.0); no code copied. The report's `QueryResult.fault` field is not
  added: engine errors arrive as text inside an `ExecError`, so the existing schema-error rule (`_policy_broke`) is
  what tells a policy's damage from an outage, and every consumer would have needed changing.
- **Files.** `src/crucible/verify/runtime/plan.py` (`_run_check`, the masked grade's `failure_kind`),
  `src/crucible/verify/sql_checks.py` (`SqlAssertion.evaluate`), `docs/modules/M01-world.md` (finding 22),
  `tests/unit/test_grade_container_lost.py` (new).
- **Public names.** none.
- **Behaviour.** A check that raises `EnvContainerLost`, or an exception caused by one, is `infra` with
  `failure_kind = crucible:env_container_lost` (was `fail`, the candidate's crash); a grade masked by an infra check
  takes that check's `failure_kind` when it has one (else `check_infra`, as before). `SqlAssertion` with a failed
  query: a schema error (`SCHEMA_ERROR`: no such table / column, syntax error) is a fail, anything else is `infra`
  (was a fail).
- **Formats.** none.
- **Tests.** `test_grade_container_lost.py` (3: a check dying on a lost container, directly or wrapped, is infra and any
  other crash still fails; `SqlAssertion` passes, fails on a dropped table and is infra on a transport error; a whole
  grade on a lost container is masked as `crucible:env_container_lost`). All 3 fail on the previous code. Verify,
  SQL-check, EnterpriseOps and pack-audit suites: 312 passed, 26 skipped.
- **Merge notes.** A SQL assertion whose own query is wrong in a way the schema-error pattern does not name now masks
  instead of failing; `pack check`'s gold run still catches it.

## 57. M1 R-12: a policy's SQL statement stops at its own deadline

- **Item.** R-12 of `docs/research/improvements/2026-10-14/M01-world.md` (must). Pattern from SkyRL (interrupt the
  connection past a deadline); no code copied. Not taken: relabelling policy SQL errors from fault `tool` to `policy`
  (every consumer treats the two alike) and a general timeout for `LocalExec.run_python` (the progress handler ends the
  SQL thread; other snippets do not run policy code).
- **Files.** `src/crucible/world/state/snippets/sql.py` (`_deadline`, `_timeout`, `query`, `execute`),
  `world/apps/base.py` (`SQL_DEADLINE_S`), `world/apps/sqlite_app.py` (`tool_sql_query`, `tool_sql_execute`),
  `world/state/postgres.py` (`_statement_timeout`, `psql_json`, `psql`, `q_sql`), `world/apps/postgres_app.py`
  (`_pg_error`, `tool_sql_query`, `tool_sql_execute`), `docs/modules/M01-world.md` (finding 23),
  `tests/unit/test_sql_deadline.py` (new), `tests/unit/test_postgres_app.py` (+1).
- **Public names.** `crucible.world.apps.base.SQL_DEADLINE_S` (30 s); App param `sql_deadline_s` (SQLite and Postgres
  Apps); snippet arg `deadline_s` (`query`, `execute`); `statement_timeout_s` on `PostgresAdapter.psql_json` / `psql` /
  `q_sql`.
- **Behaviour.** A policy's SQL statement through the SQLite or Postgres App's `sql_query` / `sql_execute` runs at most
  `App.deadline(sql_deadline_s)` seconds (default 30): SQLite's progress handler interrupts it (a write is rolled back),
  Postgres's server stops it (`statement_timeout`); either way the tool fails with
  `query_timeout: the statement ran past Ns and was stopped` (a scored tool error, never a masked hang). The snippet's
  and `psql`'s process limits are the deadline plus 30 s (were 600 s and 300 s). Verifier-side queries (state adapters)
  have no deadline, as before; other SQL errors are unchanged.
- **Formats.** none.
- **Tests.** `test_sql_deadline.py` (2: a recursive CTE stops at 1 s with `query_timeout`, scored, and the next query
  runs; a runaway `INSERT` is rolled back); `test_postgres_app.py` (+1, Docker: `pg_sleep(5)` against a 1 s deadline
  is stopped by the server for both tools and the stopped `UPDATE` is not kept). SQL App, state adapter, EnterpriseOps
  and M11 suites: 202 passed, 4 skipped.
- **Merge notes.** A pack whose policy legitimately runs SQL longer than 30 s sets `sql_deadline_s` on the App.

## 58. M1 R-16: audit check `runtime.restore_roundtrip`

- **Item.** R-16 of `docs/research/improvements/2026-10-14/M01-world.md` (should; R-5 depends on it). Pattern from
  agentenv `task_step/task_steps/multienv_validator/verify_universe_roundtrip.py:97-176` (reload an export and compare;
  Apache-2.0); no code copied.
- **Files.** `src/crucible/world/audit.py` (`_run` returns tool results, `_restore_roundtrip`, `audit_world`),
  `docs/modules/M01-world.md` (finding 24), `tests/unit/test_restore_roundtrip.py` (new).
- **Public names.** audit check id `runtime.restore_roundtrip` (major).
- **Behaviour.** With a gold trajectory of at least 2 steps, `audit_world` resets, plays the first half, snapshots,
  plays the rest, then starts a fresh World of the same spec, backend, params and store, restores the halfway snapshot
  and plays the rest there; it fails when the end fingerprints differ (`apps_differing`, and the first differing tool
  result as evidence; results alone never fail it: a test run's timings differ). Skipped with a reason for a 1-step
  gold or when simulated actors are attached (a fresh World needs its own). Without gold it is skipped like the other
  gold checks. A container World's audit starts one more container.
- **Formats.** none (a new check id in audit reports).
- **Tests.** `test_restore_roundtrip.py` (3: the audit's probe App, which resets an in-memory counter on restore, passes
  every check from reset and fails this one with `{"id": 2}` vs `{"id": 1}`; the same App re-deriving its counter
  passes; a 1-step gold is skipped). `audit_world` callers (audit, studio, packs, red team, toy end-to-end): pass, but
  for the known coding-pack failure.
- **Merge notes.** A World whose App keeps state outside its adapter now fails its audit; fix the App, not the check.

## 59. M1 R-5: a restore of a WorldRef already in place is skipped

- **Item.** R-5 of `docs/research/improvements/2026-10-14/M01-world.md` (should, after R-16), narrowed: the no-op
  restores only. Pattern from agentenv (restore work only where it pays); no code copied. Not taken: dropping the
  unused snapshot of a separate verifier's World (no pack uses separate-verifier mode) and baked bases.
- **Files.** `src/crucible/world/world.py` (`World._touched`, `_settled`, `restored_apps`, `_unchanged_apps`, `restore`,
  `_start`, `_call_tool`, `VerifierView.adapter`), `docs/modules/M01-world.md` (finding 25),
  `tests/unit/test_restore_skip.py` (new).
- **Public names.** `World.restored_apps` (the Apps the last restore restored).
- **Behaviour.** `restore(ref)` skips an App's state restore when all three hold: the World's state was last set to
  `ref` (by `start` or a restore), no tool call (`_call_tool`, policy or system) and no verifier query
  (`VerifierView.adapter`, which `query` and `changes` use) ran since, and the App's live fingerprint (after `flush`)
  equals `ref`'s. Everything else in a restore (backend hooks, `after_restore`, clock, events, inboxes, responders)
  runs as before. So `start(); reset()` and a harness's second reset restore nothing; a file changed underneath with
  no tool call (a setup hook) is caught by the fingerprint and restored; after any tool call every App is restored.
- **Formats.** none.
- **Tests.** `test_restore_skip.py` (5: two resets after start restore nothing; after a tool call, after a verifier
  query, and after a read-only call with unchanged fingerprints every App is restored; a file planted with no tool
  call is restored). World, runtime, harness, audit, Harbor, state adapter, Postgres, studio and toy end-to-end
  suites: 90 passed, 1 skipped.
- **Merge notes.** Code that changes App state between a reset and the next restore without a tool call or a verifier
  query, in a way the App's fingerprint does not see, must bump `World._touched`.

## 60. M1 R-13: the local backend refuses an allowlist; the audit names unenforced networks

- **Item.** R-13 of `docs/research/improvements/2026-10-14/M01-world.md` (should), narrowed further than the triage:
  the triage would also refuse `no-network`, but `no-network` is every spec's default, so that would refuse almost
  every local World (toy packs, CI, most tests). Pattern from agentenv (refuse before provisioning; Apache-2.0); no
  code copied.
- **Files.** `src/crucible/world/backends/local.py` (`validate`), `src/crucible/world/audit.py` (`network.enforced`),
  `docs/modules/M01-world.md` (finding 26), `tests/unit/test_local_network.py` (new), `tests/unit/test_audit.py`
  (the allowlist test opts in).
- **Public names.** local backend param `accept_unenforced`; audit check id `network.enforced`.
- **Behaviour.** `World.start` on the local backend fails for `network: allowlist` unless `backend_params.accept_unenforced`
  is set. `audit_world` adds `network.enforced` (status `skip`, major) for a world whose network is not `public` on a
  backend whose isolation is `none` (local, ssh), naming the backend and the rule it cannot enforce; a skip does not
  fail the audit. Container backends are unchanged (their egress checks run under isolation).
- **Formats.** none.
- **Tests.** `test_local_network.py` (3: an allowlist is refused; accepted unenforced it runs and the audit reports it;
  the default `no-network` runs and is reported, a public world is not). Audit suites: 105 passed.
- **Merge notes.** A local world with an allowlist must add `backend_params: {accept_unenforced: true}`.

## 61. M1 R-14: pinned image identities

- **Item.** R-14 of `docs/research/improvements/2026-10-14/M01-world.md` (should). Pattern from agentenv `pin_digest`
  (resolve a tag to a digest; Apache-2.0); not ported: the pin is our own layer digest (`world.sandbox.image_id`),
  which local builds have too and registry digests do not, so no THIRD_PARTY.md row. Not taken: recording identity in
  every provider's records (Docker and Harbor already record it in WorldRefs).
- **Files.** `src/crucible/core/schema/world.py` (`SandboxSpec.image_digest`), `src/crucible/world/sandbox.py`
  (`StaleImage`, `check_pinned`, `DockerSandbox.start`), `src/crucible/interop/harbor/world_backend.py` (`provision`,
  Docker path), `src/crucible/runtime/infra.py` (`NORMALIZE["StaleImage"]`), `src/crucible/cli/main.py`
  (`crucible world pin`), `docs/modules/M01-world.md` (finding 27), `tests/unit/test_image_pin.py` (new).
- **Public names.** `SandboxSpec.image_digest`; Harbor backend param `image_digest`; `crucible.world.sandbox.StaleImage`,
  `check_pinned`; CLI `crucible world pin IMAGE [--docker-host]`.
- **Behaviour.** With `image_digest` set, a `DockerSandbox` (and `GVisorSandbox`) compares its container's image
  identity after `docker run` and, on a mismatch or an unreadable identity, removes the container and raises
  `StaleImage`; Harbor's Docker path does the same after its environment starts. `boundary` records it as
  `crucible:stale_image` (infra blame, never retried). Without a pin nothing changes. `crucible world pin` prints the
  identity, or fails (exit 4) when the image is not on the daemon.
- **Formats.** `SandboxSpec` (read from `backend_params.sandbox`, not part of a frozen format) gains an optional field.
- **Tests.** `test_image_pin.py` (4: the check, including an unreadable identity; the failure's kind, blame and
  retry rule; Docker: an image pinned with `crucible world pin` starts, the same tag pointed at another image is
  refused and its container removed; pinning an absent image is a user error). Harbor, agentenv, pool, runtime,
  formats and core suites: 191 passed. Harbor's pin check is covered through `check_pinned` only (no Harbor test
  environment here).
- **Merge notes.** Pinning the coding seeds is a data change for their owners: run `crucible world pin` on the lab
  daemon and add `image_digest` to each seed's backend params.

## 62. M1 R-26: swap capped at the memory limit

- **Item.** R-26 of `docs/research/improvements/2026-10-14/M01-world.md` (should), step 1 only per triage (no
  per-backend limit contracts, no refusal of unenforceable limits, no `HostConfig` read-back). Pattern from Harbor and
  Karotte (memory limits that hold); no code copied.
- **Files.** `src/crucible/world/sandbox.py` (`DockerSandbox.argv`), `src/crucible/world/apps/ci.py` (runner),
  `src/crucible/interop/harbor/gpu_docker.py` (`_ensure_overlay`), `src/crucible/interop/harbor/verify_dir.py`,
  `src/crucible/actors/rlm/kernel.py` (container argv), `docs/modules/M01-world.md` (finding 28),
  `tests/unit/test_memory_swap.py` (new).
- **Public names.** none.
- **Behaviour.** Wherever these set a memory limit they now set swap to the same value (`--memory-swap m`, compose
  `memswap_limit`), so the container gets no swap beyond its limit. Without a memory limit nothing changes.
- **Formats.** none.
- **Tests.** `test_memory_swap.py` (3: the sandbox argv, Harbor's overlay, and Docker: a real sandbox's `HostConfig`
  has `MemorySwap == Memory`). Sandbox, CI App and RLM suites pass but for the known local `test_ci_hidden_checks`.
- **Merge notes.** Not changed, for M17's owner: the factory's own `docker run` calls set `--memory` without
  `--memory-swap` (`factory/sessions.py:200`, `factory/traceaudit.py:717,1745`, `factory/generators/open.py:387,472`,
  `factory/generators/swarm.py:612`, `factory/lanes/windows_world.py:539`, `factory/lanes/executable.py:259`,
  `factory/lanes/windows_s3.py:292`).

## 63. M1 R-42: the unused `Sidecar` supervisor is deleted

- **Item.** R-42 of `docs/research/improvements/2026-10-14/M01-world.md` (should). Reference: MiMo's restart loop
  (`xiaomimimo-verl/recipes/general/mcp_proxy.py:132-133`) hides outages; a crashed tool server must fail its calls as
  infrastructure, not come back silently mid-episode.
- **Files.** `src/crucible/world/packaging/sidecar.py` (`Sidecar` and `McpServerProcess.restarts` removed, module
  docstring), `src/crucible/_vendor/mimo_sidecar/ORIGIN.md`, `docs/modules/M01-world.md` (the packaging row).
- **Public names.** removed: `crucible.world.packaging.sidecar.Sidecar`, `McpServerProcess.restarts`. Nothing in src,
  packs or tests used either (M17's `EpisodeSidecar` is unrelated).
- **Behaviour.** none (the class never ran).
- **Formats.** none.
- **Tests.** none added (a deletion). Suites over the module (MCP servers, MiMo, AWM, agentenv, A2A, SQLite and
  Postgres Apps): 68 passed, 9 skipped.
- **Merge notes.** Code that imported `Sidecar` must supervise its servers itself, and should mask, not restart.

## 64. M1 R-11 fix: the shipped verifier runtime imports only itself

- **Item.** Fix of entry 56 (R-11), found by the full run's red-team graders: `crucible/verify/runtime` is copied alone
  into verifier images (`/crucible_verify`), and R-11 made `plan.py` import `crucible.world.exec`, so every grade there
  crashed (`ModuleNotFoundError: crucible.world`; 17 grader-attack and 2 runtime red-team tests failed).
- **Files.** `src/crucible/verify/runtime/plan.py` (`_container_lost` replaces the import), `tests/unit/test_grade_container_lost.py` (+1).
- **Public names.** none.
- **Behaviour.** as entry 56: a check dying on a lost container is `infra` as `crucible:env_container_lost`; the
  runtime now finds it by the exception's `failure_kind` along the cause chain instead of by class.
- **Formats.** none.
- **Tests.** `test_the_shipped_verifier_runtime_imports_only_itself` (every `crucible` import under
  `verify/runtime` is `crucible.verify.runtime.*`). Red-team grader and runtime suites: 25 passed.
- **Merge notes.** Code added under `verify/runtime` may import only the standard library and `crucible.verify.runtime`.

## 65. Merge of uzair, haseeb and Azaan into dev (2026-10-20)

- **Item.** Integration on `dev`: `uzair` (M1 audit, entries 51-64) fast-forwarded, `haseeb` (M6 R1-R3 per-turn credit,
  the `core.build` import-cycle fix) merged clean, `Azaan` (M10 lab audit R-1..R-6, branched from c27f5ca, 77 commits
  behind) merged with 18 conflicted files. Where two branches solved the same problem the merge keeps one design and says
  which below; nothing of either branch was dropped silently.
- **Overlaps and how they were resolved.**
  - *Run identity* (M0 R-1/R-2 run lock vs M10 R-4 identity v2; decided by the user): identity v2 is the one gate. A
    changed resume is refused by `IdentityDrift` and accepted by `RunSpec.accept_drift` (epochs in `run.json`). The lock
    still writes `lock.json` and `launches/`, stamps every record (`lock`, `launch`, `task_digest`) and gives R-2 its task
    digests (a task edited before or during an attempt is masked `crucible:task_changed`), but `lock.ensure` records a
    change instead of raising. Removed: `RunLockMismatch`, `RunSpec.allow_change`. Kept both component hooks
    (`lock_settings()` / `PolicyRef.settings`, `identity_config()` / `PolicyRef.config`). M10's `breaker`, `accept_drift`
    and `hygiene` RunSpec fields are operational in the lock; a harness's grading timeouts are out of `lock_settings`, as
    identity v2 leaves them out.
  - *Container lifecycle* (M1 R-6 vs M10 R-6): M10's `world/lifecycle.py` (owner `host:pid:start` and expiry labels, the
    renewable life loop, attempt scopes, the start sweep, the SIGTERM/atexit reaper, PerfGuard's expired-orphan rule)
    replaces M1 R-6's `sleep <timeout_s + 900>` keeper, `sandbox.OWNER_LABEL` / `owner_id` / `owner_dead` /
    `LIFETIME_MARGIN_S` and the pid-based PerfGuard skip (`perf_guard.py` and `pool.py` are Azaan's as written). Kept from
    M1: `--memory-swap` (R-26) on the sandbox and the Harbor grade container, `check_pinned` (R-14) inside M10's guarded
    `image_id` block (a `StaleImage` closes the container like any other error there), and R-1's `DockerExec.expires_at`,
    now the keeper's deadline, moved out by every `renew`.
  - *World.close* (M1 R-9 vs M10 R-6): M10's shielded, bounded close (`close_timeout_s`, recorded in the attempt scope)
    wraps M1's teardown body (Apps in reverse start order, each within `APP_STOP_S`, so one hung stop cannot use up the
    whole close budget before the backend teardown). Behaviour change for R-9: a backend teardown error is raised to the
    first caller again (M10's rule, as before both changes); a timeout is still recorded, never raised. `_to_the_end` and
    `TEARDOWN_S` are gone (unused), with `_to_the_end`'s THIRD_PARTY row.
  - *Grading* (M10 R-3 re-grades vs M6 R2 gate kinds): wherever M10's `grade_in_place` path sets `res.gates`, it now also
    sets `res.gate_kinds` (native, workspace, agentenv A2A, the regrade path).
  - *Smaller*: `scrub_secrets` keeps registered values (M0) and takes M10's `replace` (a registered value gets the keyed
    digest too); `Failure.detail` stays `SafeText` beside M10's `evidence` / `retry_after`; `HarnessResult` has `usage`,
    `model_s` and `regradable`; `settle_usage` (R-19) runs on M10's capture path; agentenv capture calls keep the R-19
    usage/finish shape plus M10's `status` / `error` / `complete` / `latency_s`; `RunJournal` keeps `child_path`; tasks keep
    both `digest()` and `identity_config()`.
  - *M0 rules over M10 code*: `CRUCIBLE_R3_PROXY_HOST`, `_PROXY_PORT`, `_UPSTREAM`, `_BUDGET_USD` (coding pack R3 gate)
    declared in `core/env.py`; `agent_harness._TREE_KILL`'s `pgrep -f {pattern}` marked for the self-match lint (every
    pattern it is formatted with is bracketed); `Failure.evidence` / `retry_after` get `exclude_if` and the run record is
    `crucible.run/v2.1` (frozen; a v1 record still upcasts to v2); a media prompt enters `run.json` as its digest
    (`SingleTurnTask.identity_config`: no media bytes in run files); the two M10 MIT rows of THIRD_PARTY name their notices.
  - *M10 R-4 over M17 code* (decided by the user; for Ahmed to review): identity v2 refuses what it cannot inspect, so
    `factory.envs.scripted_policy` declares `PolicyRef(config=...)` (the steps by digest), `ExecTask.identity_config()`
    (mined repo, sha, parent, `task_hash`, requirements) and `WindowTask.identity_config()` (plus the window, spans,
    weights, Phi share, `verifier_version`, the agent's world image and workspace mode); five M17 test policies declare a
    `config`. `WindowTask.checkpoint` is left out (the sidecar writes progress, not the grade).
- **Files.** the 18 conflicted files (`THIRD_PARTY.md`, `core/schema/run.py`, `core/secrets.py`, `interop/agentenv/harness.py`,
  `interop/harbor/verify_dir.py`, `interop/harbor/world_backend.py`, `runtime/control_task.py`,
  `runtime/harness/{base,native,workspace}.py`, `runtime/journal.py`, `runtime/perf_guard.py`, `runtime/run.py`,
  `runtime/single_turn.py`, `world/{pool,sandbox,world}.py`, `tests/unit/test_sandboxes.py`), plus `runtime/lock.py`,
  `core/env.py`, `interop/harbor/agent_harness.py` (one comment), `factory/envs.py`, `factory/lanes/{executable,windows}.py`,
  `schemas/crucible.run/v2.1.*`, `tests/unit/_formats.py`, `tests/unit/test_m17_{c_harness,e_run,w_world}.py` (policy configs),
  `docs/modules/M01-world.md` (findings 20, 21), `tests/unit/test_sandbox_lifetime.py` (rewritten for the
  removed M1 R-6 names: the exec learns the keeper's deadline and each renewal), `tests/unit/test_world_close.py` (a
  teardown error reaches the first caller), `tests/unit/test_run_lock.py` and `tests/unit/test_task_digest.py` (the
  refusal is `IdentityDrift`, the allowance `accept_drift`).
- **Public names.** removed: `crucible.runtime.lock.RunLockMismatch`, `RunSpec.allow_change`,
  `crucible.world.sandbox.OWNER_LABEL`, `owner_id`, `owner_dead`, `LIFETIME_MARGIN_S`, `DockerSandbox.lifetime_s` as a
  property (M10's constructor argument now), `crucible.world.world.TEARDOWN_S`, `_to_the_end`. Added: the four
  `CRUCIBLE_R3_*` knobs, `ExecTask.identity_config`, `WindowTask.identity_config`. Format: `crucible.run/v2.1`.
- **Tests.** ruff: no new finding (64 left, all on Azaan's tip); lint-imports: 4 kept. Full unit + toy_e2e run before the
  M17, format and media fixes: 3163 passed; after them, the suites they touch (core audits, run lock, task digest,
  formats, media, third party, M1 World, M6, M10, M17): all pass but the failures below. Full run on the final tree
  (with origin/dev b83aedd merged, 1ae5e64; the 15-min A2A test deselected): 3231 passed, 61 skipped, 15 failed, 1 error,
  every one listed below; `r5_host_calls::test_sibling_attempts_keep_stepping_while_one_git_call_blocks` fails under
  full-suite load only, and `test_native_two_timeouts` passes under it only (both timing). Main's own run at b83aedd
  also fails `test_core_audit::test_every_crucible_variable_in_the_code_is_declared` (six `CRUCIBLE_*` names in the
  kernel lane, Ahmed's).
- **Merge notes.** Left for the authors (they fail on Azaan's own tip too): on Azaan's tip: `test_m10_lab_audit_r1_harbor.py` does not collect (`CONNECT_ERROR` undefined),
  `r1_a2a::test_a2a_agent_error_after_upstream_5xx_is_masked` waits 15 min for the R-5 breaker's quarantine and fails,
  `r2_quiesce::test_counted_by_bench_and_studio_hack_kinds`, `r3_control` / `r3_regrade` Harbor in place, two `r4_identity`
  tests, `r5_breaker::test_a_key_still_failing_after_quarantine_s_is_quarantined_once`, and the 64 ruff findings; on Haseeb's
  tip: `test_sampling` expects `crucible.rollout/v1.1` (now v1.2); on main: `card_tools_only`, CI hidden checks, Ray
  placement, two studio tests, the coding pack fixture. Between M10 and M17: `test_m17_p6_envs::test_native_two_timeouts`
  expects a rollout past `rollout_s` to be masked, but M10 R-3's grading window lets its grade finish (scored 0.0): which
  rule wins is Azaan's and Ahmed's call. Open: RLM kernel orphans
  (`crucible.rlm.expires`) count as PerfGuard contention until they end, which M1 R-6's pid rule used to skip.

## 66. M2 R-3: `World.finished` and `World.failures` follow a restore

- **Item.** R-3 of `docs/research/improvements/2026-10-15/M02-actors.md` (must), narrowed: `WorldRef` is a frozen
  unversioned format (R-18), so no field is added to it. `finished` is read back from the responders' saved state, as
  the audit asks; `failures` is not carried in the ref but starts empty, because a retry resumes from its masked
  attempt's checkpoint, which can hold the fault that masked it. Pattern from NeMo-Gym's conversational tool-use server
  (terminal state derived from the replayed log; Apache-2.0); no code copied.
- **Files.** `src/crucible/world/world.py` (`World.restore`, `_run_responder`), `src/crucible/actors/base.py`
  (`ActorBase.finished_reason`, `state_dict`, `load_state_dict`), `docs/modules/M02-actors.md` (finding 6),
  `tests/unit/test_restore_episode_state.py` (new).
- **Public names.** `ActorBase.finished_reason` (the reason of the responder's `Done`, set by the World).
- **Behaviour.** When a responder returns `Done`, the World also sets its `finished_reason` (when it has one).
  `World.restore(ref)` sets `World.finished` to the responders in `ref` whose saved state has a `finished_reason`, and
  `World.failures` to `[]`. So each M6 `branch_value` continuation starts from the branch point (a user's DONE or a
  responder fault in one no longer cuts short or masks the others), a restore to before a DONE lets the user talk
  again, and a World restored from a ref taken after a DONE (a checkpoint resume) ends `user_stop` instead of running
  to `max_steps`. `reset` is unchanged.
- **Formats.** none. A responder's saved state gains `finished_reason` only once it has ended, so the state, and every
  `WorldRef` fingerprint, of a conversation still running is unchanged; a ref saved before this reads as not ended.
- **Tests.** `test_restore_episode_state.py` (4: branch_value continuations with a stopping user and with a responder
  that fails once; a restore before and after the DONE; the saved state). Actor, World, credit, core and format
  suites: 318 passed.
- **Merge notes.** A responder that is not an `ActorBase` keeps its end only until a restore unless it has a
  `finished_reason` attribute and saves it in `state_dict`.

## 67. M2 R-14: the compaction budget is the model's declared or discovered window

- **Item.** R-14 of `docs/research/improvements/2026-10-15/M02-actors.md` (should), narrowed: an undeclared window is not
  an error for a self-hosted engine (the local mlx-lm student and every fake endpoint would stop working); it stays
  unknown, with today's 128,000 default. Hosted models are read from config only, so a run served from the LLM cache
  never goes online for a window. Pattern from verifiers `discover_threshold` (the /models fields; MIT) and Harbor's
  declared model limits (Apache-2.0); no code copied, so no THIRD_PARTY row.
- **Files.** `src/crucible/core/config.py` (`Config.context_windows`), `src/crucible/actors/llm.py`
  (`LLMClient.context_window`, `window_tokens`, `window_source`, `WINDOW_FIELDS`, `WINDOW_DISCOVERY_S`),
  `src/crucible/actors/policy.py` (`ContextConfig.budget_tokens`, `DEFAULT_BUDGET_TOKENS`, `LLMToolPolicy._window`),
  `src/crucible/actors/rlm/policy.py` (`_config`), `crucible.toml` (a commented `[context_windows]` example),
  `tests/unit/test_context_window.py` (new).
- **Public names.** `Config.context_windows` (`crucible.toml` table `[context_windows]`: model id -> tokens, positive);
  `LLMClient.context_window()` (async, resolved once); `LLMClient.window_tokens`, `window_source` ("config", "models",
  "unknown"); `crucible.actors.policy.DEFAULT_BUDGET_TOKENS` (128,000).
- **Behaviour.** `ContextConfig.budget_tokens` defaults to `None`: the compaction budget is the model's window when the
  policy's client knows it, else 128,000 as before; an explicit value is used as given. `LLMClient.context_window()`
  reads `context_windows[model id]`, else, for a self-hosted endpoint (`_self_hosted`: localhost, `tito`, a
  `policy_version`) outside replay mode, one `GET /models` (15 s cap) and the first of `max_model_len`,
  `context_length`, `context_window`, `max_context_length` on the model's entry; anything else leaves it unknown, and it
  never raises. `RLMPolicy(context=ContextConfig())` still gets 128,000 (an unset budget is not "compaction off").
- **Formats.** none (`crucible.toml` gains an optional table).
- **Tests.** `test_context_window.py` (5: hb01 compacts on a 32,768-token stub that reports its window and overflows as
  before when it does not; discovery once per client, config first, hosted from config only; hosted, replayed, refused
  and windowless engines stay unknown without error; the RLM budget; config validation). Actor, M2 audit, RLM, config,
  core, long-horizon and LLM suites: 589 passed.
- **Merge notes.** Code that reads `ContextConfig.budget_tokens` directly must treat `None` as "the model's window, else
  `DEFAULT_BUDGET_TOKENS`".

## 68. M2 R-1: one window rule at every model-call boundary

- **Item.** R-1 of `docs/research/improvements/2026-10-15/M02-actors.md` (must), as the triage's smallest version: the
  zero-token rule on the engine path, and the reply fitted to the window R-14 resolves. The fit lives in
  `LLMToolPolicy` (the one caller whose episode ends on it), not in each client: it covers the hosted, chat and
  token-in-token-out paths of `LLMClient` at once. Not taken: a fit inside `EngineLLM` for the training loop (no caller
  there knows the engine's window yet; its refusals are now overflows). Pattern from AReaL's window clamp, slime, rLLM
  and ROLL (Apache-2.0); no code copied.
- **Files.** `src/crucible/actors/policy.py` (`LLMToolPolicy._reply_tokens`, `_act`, `_final_status_turn`,
  `WINDOW_MARGIN`), `src/crucible/train/world_env.py` (`EngineLLM.complete`), `src/crucible/train/vllm_http.py` (the
  comment on its 400 handling, which said the episode ended truncated), `tests/unit/test_window_fit.py` (new).
- **Public names.** `crucible.actors.policy.WINDOW_MARGIN` (0.02 of the window).
- **Behaviour.** `EngineLLM.complete` raises `ContextOverflow` for a `length` finish with no completion tokens (vLLM's
  refusal, as `VLLMServerEngine` returns it, or a full window), so the policy ends `context_overflow`, truncated and
  scored, after one call instead of two nudges and `agent_done`. `LLMToolPolicy` sends `min(max_tokens, window - context
  - 2% of window)` when its client knows the window (`context_window`, R-14), and raises `ContextOverflow` before the
  call when nothing is left; with an unknown window it sends `max_tokens` as before. A fitted call's record carries
  `max_tokens_requested` (the sent value is its `sampling.max_tokens`). The hosted chat path already had the zero-token
  rule.
- **Formats.** none (a key in `calls.json` call records, which `runtime.record.build` does not carry).
- **Tests.** `test_window_fit.py` (3: an engine refusal ends `context_overflow`, truncated, after one call and no nudge;
  a reply fitted to the room with both values recorded, unchanged in a roomy or unknown window; no room ends the
  episode with no call). Engine, training rollout, Tinker bridge, RLM, actor and M2 audit suites: 306 passed, 12
  skipped.
- **Merge notes.** Code that calls `EngineLLM.complete` outside a policy now gets `ContextOverflow` where it got an
  empty `length` reply.

## 69. M2 R-5: a spent nudge allowance never runs a cut-off call and never ends `agent_done`

- **Item.** R-5 of `docs/research/improvements/2026-10-15/M02-actors.md` (must). Pattern from verifiers (a last
  `length` finish truncates) and mini-swe-agent (repeated format errors end the run; MIT); no code copied. Not taken:
  carrying the nudge label onto the rollout `Turn` (a frozen format; the label stays in the call record), and new
  `LIMIT_STOPS` / `LimitPolicy` rules (the new label is truncating through `TRUNCATING_STOPS` and is scored, which is
  what a rule would default to).
- **Files.** `src/crucible/actors/policy.py` (`LLMToolPolicy._act`, `POLICY_STOP_ORDER`, `TRUNCATING_STOPS`),
  `docs/modules/M02-actors.md` (finding 7), `tests/unit/test_nudge_limits.py` (new).
- **Public names.** Stop label `max_output_tokens`.
- **Behaviour.** When a blank or cut-off (`finish=length`) reply comes after the nudge allowance is spent (or with none,
  as in single-turn), `LLMToolPolicy` ends the episode instead of falling through: `max_output_tokens` (truncated) when
  the reply was cut off, `format_error` (terminal) when it was blank with a clean finish. A cut-off reply's tool calls are
  never run (the third one's ran and reset the allowance), and the reply's text stays the `Done` reason, so a cut-off
  single-turn answer is graded as before under the new label. Each nudged call's record carries `nudged`: `blank`,
  `cut_off` or `turn`.
- **Formats.** none (a key in `calls.json` call records).
- **Tests.** `test_nudge_limits.py` (5: three cut-off replies with a write call run nothing and end `max_output_tokens`;
  blank cut-off, text cut-off and blank clean replies; a cut-off single-turn answer stays the answer). Actor, M2 audit,
  RLM, M16 context, M17 auditor, single-turn and harness suites: 385 passed, 10 skipped; the 8 failures in
  `test_sb_claude_builder.py` fail on main too (its fake `claude` script's shebang breaks on this checkout's path, which
  has a space).
- **Merge notes.** Training data: episodes that stalled at the output limit are now `truncated`, so `overlong_filter`
  drops them, single-turn samples cut at `max_tokens` included (M12).

## 70. M2 R-2: a budget hold prices its own call; pressure waits instead of failing

- **Item.** R-2 of `docs/research/improvements/2026-10-15/M02-actors.md` (must), narrowed: no new
  `crucible:budget_pressure` kind and no wait timeout. A call waits only while calls in flight hold money, and each of
  them settles (its hold is released in a `finally`), so the wait is bounded by their own timeouts. Not taken: naming
  the cap's source (local or provider) in the run's stop reason (`runtime/run.py`, M10's). Pattern from repo2rlenv's
  campaign budget (waits only on pressure from holds that can settle; Apache-2.0); no code copied.
- **Files.** `src/crucible/actors/llm.py` (`LLMClient._reserve`, `complete`, `_overrun`, `_holds`, `_settled`,
  `Spend.budget_waits`), `tests/unit/test_budget_holds.py` (new).
- **Public names.** `Spend.budget_waits` (calls that waited for money held by calls in flight).
- **Behaviour.** A priced call holds its own worst case (prompt estimate plus `max_tokens` at the configured price)
  times the largest cost/estimate ratio seen so far (a stale price or a prompt the chars/4 estimate undercounts), no
  longer floored at the dearest call so far; an unpriced one still holds the dearest call seen. `BudgetExceeded`
  (`credits_exhausted`, which stops the run) is raised only when the money already spent plus this call's own hold
  exceeds the budget; when it fits but for what calls in flight hold, the call waits until one settles and checks again.
  So after one 0.20 USD call, 16 cheap concurrent calls all run (5 stopped the run before), 30 calls on a 0.004 USD
  budget all run (21 were refused by holds alone), and AUD-D12's cap still holds under concurrency.
- **Formats.** none.
- **Tests.** `test_budget_holds.py` (4: one dear call no longer inflates later holds; pressure waits; true exhaustion
  raises at once with a call in flight; a provider dearer than its price scales later holds). Every unit file that
  builds an `LLMClient` or books spend: 319 passed, 2 skipped; the failures are the 8 known
  `test_sb_claude_builder.py` ones and `test_sampling`'s rollout version (entry 65).
- **Merge notes.** `_reserve` is now a coroutine returning `(hold, priced estimate)`.

## 71. M2 R-15: one overflow wording table for every model-calling path

- **Item.** R-15 of `docs/research/improvements/2026-10-15/M02-actors.md` (should). PORT of Prime Agent's overflow
  table (`crates/pa-ai/src/utils_inner/overflow.rs:8-55`, 3e0893260, MIT; notice already in
  `LICENSES/MIT-prime-agent.txt`; THIRD_PARTY row added). Not changed: the engine's own refusal check in
  `train/vllm_http.py` (M12's, a 400 from our own engine) and the importers of `CONTEXT_OVERFLOW_MARKERS`
  (`runtime/agent_evidence.py`, `train/bridges/tinker/backend.py`), which keep the nine markers they had.
- **Files.** `src/crucible/actors/llm.py` (`OVERFLOW_PATTERNS`, `NON_OVERFLOW_PATTERNS`, `is_overflow_text`,
  `is_context_overflow`, `_complete_paid`, `_tito_complete`), `src/crucible/actors/rlm/context.py` (`is_overflow`;
  `_NOT_OVERFLOW` removed), `THIRD_PARTY.md`, `tests/unit/test_overflow_wordings.py` (new).
- **Public names.** `crucible.actors.llm.OVERFLOW_PATTERNS`, `NON_OVERFLOW_PATTERNS`, `is_overflow_text(text)`.
  `CONTEXT_OVERFLOW_MARKERS` stays (every one of them is covered by the table).
- **Behaviour.** The hosted chat path (a 400/413, or a 200 without choices), the token-in-token-out path (a 400 or
  now 413 from the engine; it checked one wording) and the RLM's `is_overflow` decide with one table: Prime Agent's 24
  provider patterns, the nine verifiers markers, NeMo-Gym's and SGLang's "longer than the model's context length" and
  vLLM's "longer than the maximum model length" / "is too long and exceeds limit of N", minus rate-limit and throttle
  wordings. So SGLang, llama.cpp, LM Studio, Ollama, Bedrock and Cerebras overflows end the episode
  `context_overflow` (scored) instead of `agent_request_failed` or `provider_unavailable` (masked, the latter retried
  into the same overflow).
- **Formats.** none.
- **Tests.** `test_overflow_wordings.py` (44: one sample per pattern, the old markers covered, four non-overflows, and
  the probe's nine wordings on the hosted, RLM and engine paths). RLM, M16, M14, Tinker bridge, agent-evidence and actor
  suites: 394 passed, 2 skipped, 3 xfailed.
- **Merge notes.** A new provider wording goes into `OVERFLOW_PATTERNS` (with its sample in the test), not into a
  caller.

## 72. M2 R-24: a paid call lost after it was sent is booked as uncertain, and sent again once

- **Item.** R-24 of `docs/research/improvements/2026-10-15/M02-actors.md` (should), narrowed: no `Idempotency-Key` on
  the re-send (no provider or proxy we use deduplicates on it) and no failed-call entry in `model_calls`
  (`runtime.record.build` turns call entries into rollout turns). Pattern from repo2rlenv's campaign LLM client (a
  dispatched call that fails keeps its reservation as uncertain; Apache-2.0); no code copied.
- **Files.** `src/crucible/actors/llm.py` (`_maybe_billed`, `RESENDS_AFTER_SEND`, `LLMClient._complete_paid`,
  `complete`, `Spend.failed_calls`), `tests/unit/test_failed_calls_booked.py` (new).
- **Public names.** `crucible.actors.llm.RESENDS_AFTER_SEND` (1); `Spend.failed_calls`.
- **Behaviour.** In `_complete_paid`, a failed attempt that may have reached the model (a 5xx, a 200 without choices, or
  a transport failure after the request went out: a read timeout or a reset, as the OpenAI SDK's `APITimeoutError` /
  `APIConnectionError` over httpx) books the call's budget hold as spend and as uncertain spend (as an unpriced
  model's calls are booked), adds it to the task's spend scope, and allows only one more send. A refused connection
  or a connect timeout (never sent) and any 4xx book nothing and retry as before. Calls with no hold (self-hosted
  engines, cost 0) are unchanged. So the hard budget and the run's uncertain-spend rule now see the dearest failures
  (long reasoning calls that time out), and a lost call is billed at most twice.
- **Formats.** none.
- **Tests.** `test_failed_calls_booked.py` (4: a read timeout is sent twice and books two holds; a 5xx then a success
  books both; a refused connection and a 400 book nothing). Every unit file that builds an `LLMClient` or books spend,
  except `r1_a2a` (it waits 15 minutes on main: entry 65): 498 passed, 1 skipped; the failure is `test_sampling`'s
  rollout version (entry 65).
- **Merge notes.** The run ledger settles `cost_usd + cost_uncertain_usd`, so a lost call counts twice there, as
  fallback-priced calls already did (ledger item IDX-FIX-1); conservative, and it may reach the 10% uncertain-spend stop
  sooner on a flaky provider (the audit's stated risk).

## 73. M2 R-6: unique, provider-safe, deterministic tool-call ids

- **Item.** R-6 of `docs/research/improvements/2026-10-15/M02-actors.md` (should), narrowed: the repair is in the two
  policies; the engine codecs (`train/chat.py`, `train/bridges/tinker/codec.py`, M12's) keep minting `call_<8 hex>`
  ids, which are valid and unique, so the policy keeps them and the token path renders exactly as before. Not taken: a
  strict mode that charges a duplicate id to the policy (NeMo-Gym's rule; a serving stack's parser makes these).
  Pattern from NeMo-Gym's nemo_user_sim id checks (Apache-2.0); no code copied.
- **Files.** `src/crucible/actors/policy.py` (`CALL_ID`, `LLMToolPolicy._call_ids`, `_ids_seen`, `_act`,
  `_final_status_turn`, `load_state_dict`), `src/crucible/actors/harness.py` (`BashLoopPolicy.act`),
  `tests/unit/test_tool_call_ids.py` (new).
- **Public names.** `crucible.actors.policy.CALL_ID` (`[A-Za-z0-9_-]{1,64}`).
- **Behaviour.** `LLMToolPolicy` keeps a provider's tool-call id when it matches `CALL_ID` and is new in the episode;
  otherwise (empty, reused from an earlier reply, repeated in the reply, or with characters such as `:` or `.` that
  Bedrock and Anthropic refuse) the call gets `c` + 8 hex of sha256(actor:call index:position), a 9-character id that
  is the same on every replay. The id is used in the stored assistant turn, the `ToolCall` and the call's `actions`;
  a call record whose ids were replaced carries `provider_ids`. So a provider that numbers ids per reply no longer has
  turn 2's malformed call charged to turn 1 (`format_errors_by_call`), two calls sharing an id no longer let a message
  in between their results, and an id-less reply no longer gets a random id that made every later prompt, and so the
  replay cache key, differ. The ids already used are rebuilt from the history and call records on `load_state_dict`.
  `BashLoopPolicy` numbers its calls `bash-<model call index>` (was random).
- **Formats.** none (ids in traces and records are now deterministic; a key in `calls.json` call records).
- **Tests.** `test_tool_call_ids.py` (5: an id reused across turns charges only its own call; two calls sharing an id
  keep their batch together; id-less and unsafe ids are derived alike on two replays; a restored policy knows its ids;
  the bash loop numbers its calls). Every unit file that drives `LLMToolPolicy` or `BashLoopPolicy` or reads tool-call
  ids (training, `r1_a2a`, `r1_harbor` and `sb_claude_builder` left out): 921 passed, 38 skipped, 3 xfailed; the failure is the known `test_agentenv_mcp_env` card_tools_only (the agentenv clone's extra `items_whoami` tool).
- **Merge notes.** A fake LLM that returns the same id on every turn now sees the second and later ones replaced; a
  test that asserts a fixed id on a later turn must take it from the returned `ToolCall`.
- **Full run after all M2 items (2026-10-21, 49 min).** Unit (no training or M12 files, no `r1_a2a`), toy end-to-end
  and red-team tests: 3285 passed, 56 skipped, 3 xfailed, 23 failed, none from M2. On main's source (7176582) the same
  fail: the 8 `test_sb_claude_builder.py` tests (the fake `claude` shebang breaks on a path with a space), 4
  `r2_quiesce_docker` and 1 `r6_docker` red-team tests, 2 `r4_identity`, `r5_breaker` quarantine. Known from entry 65 and
  earlier: `card_tools_only`, `r1_harbor` 503, two studio tests, the coding pack fixture, `test_sampling`'s rollout
  version. `r5_host_calls::test_sibling_attempts_keep_stepping_while_one_git_call_blocks` fails under full-suite load
  only (it passes alone and with its neighbours). Ruff on the changed files and lint-imports (5 contracts) are clean.

## 74. M3 R-1: a stream the client abandons is "client disconnected" under uvicorn

- **Item.** R-1 of `docs/research/improvements/2026-10-20/M03-connectors.md` (should), as recommended. Not taken:
  widening M10's `verdict_of` to read Starlette's `CancelledError` text (the wrong cause would stay in M3's rows), a
  separate `cancelled` verdict (no consumer), a shielded row write (every probe wrote the row without one).
  Pattern from NeMo-Gym's `base_responses_api_model.py:1166-1169` (Apache-2.0); no code copied.
- **Files.** `src/crucible/connectors/capture_proxy.py` (`CaptureProxy.app` → `forward_call` → `relay`, module
  docstring), `tests/unit/test_capture_client_left.py` (new), `docs/modules/M03-connectors.md` (finding 6).
- **Public names.** none.
- **Behaviour.** A streamed exchange ended from the downstream side, by `GeneratorExit` or by the
  `asyncio.CancelledError` that Starlette 1.7 raises when the client disconnects under uvicorn 0.54 (ASGI 2.3), is
  recorded with `error: "client disconnected"`, `complete: false`, its status and the partial response. It read
  `CancelledError: Cancelled via cancel scope ...`, which `verdict_of` took for `infra`: when Crucible's deadline
  killed a streaming agent (claude-code or codex via Harbor, an A2A agent), its probe answered ok and the end became
  `agent_run_error` with blame `unknown` (masked, counted by the 5% stop) instead of an evidenced `timeout`. Upstream
  failures keep `"<Type>: <message>"`. Trade-off: a mid-stream shutdown of the proxy itself is labelled the same.
- **Formats.** none (the `error` string of such capture rows).
- **Tests.** `test_capture_client_left.py` (1: the proxy served by uvicorn, a client that leaves after 5 chunks gives
  one row `complete: false`, `error: "client disconnected"`, partial reply kept, verdict `client_left`, nothing left in
  flight; it failed before the fix with the `CancelledError` row). Every unit file that drives the capture proxy or its
  evidence (`r1_a2a` left out: 15 min), M3 audit and core files: 425 passed. Ruff on the changed files and lint-imports
  (5 contracts) are clean.
- **Merge notes.** One `except` clause in `relay()`; R-2 will reuse the new test's uvicorn harness.

## 75. M3 R-2: an exchange whose client left is evidence, never an action, a trained call or a retry copy

- **Item.** R-2 of `docs/research/improvements/2026-10-20/M03-connectors.md` (should), as recommended, with one
  narrowing: only an answer (status < 400) is marked; an upstream error whose client left keeps its own status as M10's
  evidence. Not taken: "the last attempt sent wins" (misses resends without a retry header, and the deadline kill),
  `after_end` (the key is revoked before the kill), recording only after the flush (M10 loses its row), replaying
  retries (changes what the agent gets). Pattern from slime `adapters/common.py:428-443` and AReaL `cache.py:620-639`
  (Apache-2.0); no code copied.
- **Files.** `src/crucible/connectors/capture_proxy.py` (`CLIENT_LEFT`, `client_left`, `forward_call`,
  `CaptureLogConnector.fetch`, `normalize`, module docstring), `src/crucible/interop/agentenv/harness.py`
  (`calls_from_capture`), `src/crucible/interop/harbor/reconcile.py` (`reconcile`), `src/crucible/runtime/agent_evidence.py`
  (`verdict_of` uses the shared check), `docs/modules/M03-connectors.md` (capture_proxy row, finding 4),
  `tests/unit/test_capture_client_left.py`.
- **Public names.** New `crucible.connectors.capture_proxy.CLIENT_LEFT` (`"client disconnected"`) and
  `client_left(row) -> bool` (a capture row or a `calls_from_capture` call).
- **Behaviour.**
  - Proxy: a non-streaming answer whose client already left (a deadline kill, a client timeout) is recorded with
    `error: "client disconnected"`, keeping status, response and usage; `complete` stays unset, so `capture_spend`
    still books it as certain. It was an ordinary 200. Starlette's `is_disconnected()` returns at once.
  - `normalize`: a client-left exchange (streamed or not) gives its `llm_call` with `data.error` and no `tool_call` or
    server-tool result. A deadline-killed mini-swe-agent call listed `echo hi > out.txt` with `writes: file:out.txt`.
  - `fetch`: a client-left attempt counts as failed, so a later attempt of the same body replaces it; a client-left
    row answered after any earlier attempt of the same body (the timed-out original answering after its retry, turns
    later) is dropped. A lone client-left row (the deadline kill) stays. The raw log keeps every row for M10.
  - `calls_from_capture`: a client-left call is `trainable: false` (`runtime/record.py` already honours it); Harbor's
    `reconcile` leaves it out of the comparison (it was read as an auxiliary call the agent did not log).
- **Formats.** A non-streaming capture row can now carry `error: "client disconnected"`; a `calls.json` call can carry
  `trainable: false` for it. M10 already read that error as `client_left`.
- **Tests.** `test_capture_client_left.py` (+6: a deadline kill under uvicorn gives a client-left row with usage, no
  tool_call, a non-trainable call and certain spend, while a waiting client's row is unmarked; a timeout then SDK retry
  gives 2 exchanges, not 3, and no abandoned write; `fetch` in three orders; reconcile). All 6 failed before the fix.
  Every unit file that drives the capture proxy, `calls_from_capture`, reconcile or the evidence code (`r1_a2a` and
  training left out), M3 audit and core files: 519 passed; `test_sb_claude_builder.py` has its 8 known failures (the
  fake `claude` shebang breaks on the path with a space). Ruff on the changed files and lint-imports (5 contracts) are
  clean (FURB162 at `agent_evidence.py:223` is pre-existing).
- **Merge notes.** Small edits in `capture_proxy.py` (forward_call's non-streaming record, fetch, three tool-call loops
  in normalize), one line each in `calls_from_capture`, `reconcile` and `verdict_of`. A test fake that answers after its
  client left now gets a marked row.

## 76. M3 R-3: a frontier run its provider or harness ended is `unknown` and unscored

- **Item.** R-3 of `docs/research/improvements/2026-10-20/M03-connectors.md` (should), as recommended, plus one addition
  the user chose: `AgentRunSignalMapper` also scores no run graded under an older task version. Not taken: a list of
  exception names (never complete), refusing such runs (MiMo; loses traces M5 and M2 use), parsing messages (M10's job).
  Pattern from mini-swe-agent's exit vocabulary (MIT) and MiMo's `InfraError` rule (Apache-2.0); no code copied.
- **Files.** `packs/coding/src/crucible_pack_coding/signals.py` (`SCORED_EXITS`, `unfinished`,
  `AgentRunSignalMapper.map`), `packs/coding/src/crucible_pack_coding/connectors/agent_runs.py` (`normalize`, module
  docstring), `docs/modules/M03-connectors.md` (agent_runs row), `tests/unit/test_agent_run_exits.py` (new).
- **Public names.** New `crucible_pack_coding.signals.SCORED_EXITS` (`CLEAN_EXITS` minus the two limits: `Submitted`,
  `RepeatedFormatError`) and `unfinished(exit_status, reward) -> bool`. They live in `signals.py`, not the connector,
  because the connector imports Harbor at module level and the mapper must load without it.
- **Behaviour.**
  - `normalize`: a run with reward < 1 whose `exit_status` is not one of the agent's own scored stops is `unknown`,
    "the run stopped on <status> before it was done" ("no exit status (killed)" for the empty status a kill leaves).
    Only `LimitsExceeded` and `TimeExceeded` were exempt, so a graded provider 402 (`OpenRouterAPIError`), a harness
    `RuntimeError` or a kill was `bad`. The stale-version and not-a-grade rules still come first, and reward >= 1 still
    goes through the contamination rule. `details.exit_status` is recorded in every branch (was: the limits branch only).
  - `AgentRunSignalMapper`: such a run gives no signal on any path (the limit exemption covered the reward path only, so
    a limit-stopped run with failing gates gave -1 per gate); neither does a run with `stale_version` (the two task04 v1
    passes gave +1 "every hidden gate passed" though the connector labels them `unknown` under v3). A trace stored
    before `exit_status` was recorded keeps the old rule.
  - The 11 recorded runs keep their labels (1 good, 5 bad, 5 unknown); their signals change only by the two stale +1s.
- **Formats.** `details.exit_status` in every `coding/agent_run` trace outcome.
- **Tests.** `test_agent_run_exits.py` (9: the two real crashed trajectories with a `{reward: 0.0}` report, with and
  without a failing gate, and a killed run give `unknown`, the exit status and no signal; `Submitted` and
  `RepeatedFormatError` at 0.0 stay `bad` with -1; a trace with no recorded exit keeps its -1; the real runs keep their
  labels and the stale ones give no signal). 8 failed before the fix (the old-store guard passes either way). Coding
  connectors, M3, M4, M8, M14, M15 audit, packs, span and stitch files: 289 passed, 12 skipped; the failure is the known
  coding pack fixture (`data/cache/git` absent here, as on main). Ruff on the changed files and lint-imports (5
  contracts) are clean.
- **Merge notes.** `signals.py` now imports `crucible.runtime.agent_evidence` (no Harbor); `agent_runs.py` imports
  `unfinished` from `signals.py`. M8 forge sees fewer -1s only for runs that crashed or were killed.

## 77. M3 R-4: a lone surrogate no longer aborts an ingest batch

- **Item.** R-4 of `docs/research/improvements/2026-10-20/M03-connectors.md` (should), as recommended. Not taken:
  escaping in each connector (misses the next one), in `Store.put_doc` or `core.canonical` (changes stored bytes core-wide,
  M0 R-20/R-21), moving `store.put` into `ingest`'s `try` (still loses the session). Pattern from karotte
  `src/karotte/text.py:4-24` (MIT), already ported as `core/text.py` (`safe_text`, `safe_json`); no new copy.
- **Files.** `src/crucible/connectors/redact.py` (`redact`, module docstring), `docs/modules/M03-connectors.md`
  (Redaction list), `tests/unit/test_ingest_surrogates.py` (new).
- **Public names.** none.
- **Behaviour.** `redact()` passes the trace's JSON dump through `core.text.safe_json` before redacting, so every string
  and key it returns is UTF-8 encodable: a lone surrogate (a cut emoji in tool output, the `\udXXX` escapes the capture
  proxy writes) becomes escape text such as `\ud83d`. It passed `normalize` and `redact`, then `Store.put_doc` raised
  `UnicodeEncodeError` outside `ingest`'s per-source `try`: that source and every later one were lost (0 of 3 stored)
  and the error named no source. `safe_json` returns its input unchanged when nothing needs escaping, so every other
  trace keeps its bytes and store key. Affects every JSON-fed connector, `crucible label` and the M3 gate.
- **Formats.** none (stored traces may hold escape text where a lone surrogate was).
- **Tests.** `test_ingest_surrogates.py` (2: a capture dir whose first session holds `\ud83d` in a user turn and
  `\udcff` in a tool argument stores 3 of 3 with the escape text, it failed before the fix with the
  `UnicodeEncodeError`; surrogate-free sessions get the same store keys as with `safe_json` bypassed). Redaction,
  ingest, store, M3, M4 and M14 audit, coding connector, observe, tags, packs and core files: 461 + 119 passed, 8
  skipped; the failure is the known coding pack fixture (`data/cache/git` absent here, as on main). Ruff on the changed
  files and lint-imports (5 contracts) are clean.
- **Merge notes.** One line and one import in `redact()`. The M3 should items (R-1..R-4) are all done; the 23 later items
  wait for their consumer.

## 78. M4 R-1: a policy turn graded wrong by any uncontradicted negative step signal is never a positive target

- **Item.** R-1 of `docs/research/improvements/2026-10-27/M04-signals.md` (must), as recommended; it corrects the M15 V8
  fix (`775f30f1`), which excluded corrective-graded turns only. Not taken: a type list in M15 (drifts from M4's
  verdicts), a reliability threshold (reliability is None on this path, so it excludes nothing), skipping the KTO row
  (gives `wrong` two meanings). No new counter: `skipped.sft_graded_wrong` already counts the dropped SFT turns. Pattern
  from MiMo `verl/trainer/ppo/signed_rebalance.py:14-22` (Apache-2.0); no code copied.
- **Files.** `src/crucible/signals/pivots.py` (`_against`, `pivots`, new `graded_wrong`, module docstring),
  `src/crucible/studio/datasets.py` (`from_trace`), `docs/modules/M04-signals.md` (decision points),
  `tests/unit/test_graded_wrong.py` (new).
- **Public names.** New `crucible.signals.pivots.graded_wrong(trace) -> dict[int, list[str]]` (graded seq -> signal ids).
- **Behaviour.** `graded_wrong` takes every step-level signal (`TAXONOMY[type].grades == "step"`) with polarity < 0 and
  grades, skipping `all_commits` attributions and signals `verdict` contradicts, as `pivots` does but without the
  corrective-only filter; `pivots` keeps that filter and still decides the DPO pairs. `from_trace` builds
  `Labelled.wrong` from `graded_wrong` through `turn_of_seq`, so only policy turns are marked (the toy `sla_breach` on a
  non-policy event marks nothing). Such a turn is no SFT target, KTO False, never DPO chosen. On the toy pack
  `crucible label` trained T1005 seq 2 ("The carrier shows it at your building's parcel room.", answered "There is no
  parcel room in my building") and T1007 seq 2 ("Please regenerate your API key and retry.", rejected then escalated):
  SFT 10 -> 8, KTO desirable/undesirable 10/2 -> 8/4, `sft_graded_wrong` 0 -> 2, DPO and pivot counts unchanged.
  Trade-off: a correct reply graded only by `sla_breach` on a policy turn is no longer a positive.
- **Formats.** none (fewer positive rows).
- **Tests.** `test_graded_wrong.py` (2: the toy label run's counts and both replies only as KTO False; `graded_wrong` on a
  failing check, the same check passing later, an `all_commits` revert, a trace-level signal and a correction, plus
  `from_trace`'s `wrong`). The label counts were 10 and 10/2 before the fix. Label, studio, M4, M5, M14 framework, packs
  and core files: 140 + 93 passed; failures are the known coding pack fixture and two `test_m15_audit_studio` tests (async
  export, site index), which fail without this change too. Ruff on the changed files and lint-imports (5 contracts) are
  clean.
- **Merge notes.** `from_trace` no longer adds to `wrong` inside the pivots loop; `wrong` is computed once before it.

## 79. M4 R-2: a trace whose extraction raised yields no training rows

- **Item.** R-2 of `docs/research/improvements/2026-10-27/M04-signals.md` (should), as recommended. It narrows the M4
  audit's A15 (one failing mapper no longer stops the run) so it fails closed for that trace. Not taken: aborting the run
  (the defect A15 fixed), keeping only its negative rows (a mode with no consumer). The contract is M4's; the edit is in
  M15's `studio/label.py`. Pattern from AReaL's PRM `error_policy="reject"` (`areal/api/cli_args.py:2701-2711`,
  Apache-2.0); no code copied.
- **Files.** `src/crucible/studio/label.py` (`label`), `docs/modules/M04-signals.md` (Extraction),
  `tests/unit/test_extract_failed_rows.py` (new).
- **Public names.** none.
- **Behaviour.** A trace whose `extract()` raised is still listed in `summary["extract_failed"]`, gets the credit hook's
  outcome and is written to `traces.jsonl` and `rewards.jsonl`, but is never passed to `from_trace`: no SFT, KTO or DPO
  row, counted in `summary["skipped"]["extract_failed"]` (only when non-zero, so other summaries are unchanged). With no
  signals its `wrong` set was empty, so a corrected or rejected step was an SFT target and KTO-desirable (the toy T1005
  kept 2 SFT and 2 KTO rows when its mapper failed). Other traces' rows are unchanged. No flag.
- **Formats.** `summary.json` `skipped` can hold `extract_failed`.
- **Tests.** `test_extract_failed_rows.py` (1: with extraction raising for T1005, no row comes from it, `skipped`
  counts it, the other traces' rows equal a run without the failure, and the trace is in `traces.jsonl`; it failed
  before the fix). Label, M4, M14 framework and studio files: 142 passed, 3 skipped. Ruff on the changed files and
  lint-imports (5 contracts) are clean.
- **Merge notes.** One filter on the `from_trace` comprehension in `label()`.

## 80. M4 R-3: a PR check run that timed out or never started is polarity 0, not -1

- **Item.** R-3 of `docs/research/improvements/2026-10-27/M04-signals.md` (should), as recommended. Not taken: dropping
  the signal (every CI run is labelled `check_result` in the gate, so a drop is a type miss). Pattern from NeMo-RL
  `nemo_rl/experience/failures.py:96-97,146` (a timeout is infra, never scored) and NeMo-Gym's `inconclusive`
  (Apache-2.0); no code copied.
- **Files.** `packs/coding/src/crucible_pack_coding/signals.py` (`FAILED`, new `INCONCLUSIVE`,
  `CodingSignalMapper.map`, module docstring), `docs/modules/M04-signals.md` (coding/pull_request row),
  `tests/unit/test_ci_inconclusive.py` (new), `tests/unit/test_m4_audit_packs.py` (M12/M13 spelling case).
- **Public names.** New `crucible_pack_coding.signals.INCONCLUSIVE` (`timed_out`, `startup_failure`); `FAILED` is now
  `failure`, `error`.
- **Behaviour.** A PR `ci_run` whose conclusion is `timed_out` or `startup_failure` gives `check_result` with polarity 0,
  `meta.inconclusive=<conclusion>` and `meta.reason` (P16), the flaky shape, which `judgeable` keeps out of reliability,
  pivots and `graded_wrong`. It was -1, so a later hotfix "confirmed" the timeout's -1 in the gate's
  `check_result/automated` reliability row, while the same pack's CI-log path masks GitHub's time-limit line as infra.
  Flaky now compares only `success` with `failure`/`error`, so a timeout then a pass of one check is a pass (+1), not
  flaky. `failure` and `error` stay -1. Trade-off: a hang the PR caused is no longer -1, as on the CI-log path.
- **Formats.** `check_result` meta can carry `inconclusive` and `reason`.
- **Tests.** `test_ci_inconclusive.py` (2: `timed_out`, `STARTUP_FAILURE`, `failure`, `success` give [0, 0, -1, +1] and
  the timeout's verdict is None after a hotfix while the failure's is True; a timeout then a pass of one check is +1,
  not flaky; both failed before the fix). `test_m4_audit_packs.py::test_M12_M13` used `STARTUP_FAILURE` to check the
  capital spelling and now uses `FAILURE` (same intent). M4, coding connector, agent-run, M8 and M14 pack files: 51 + 47
  passed (M14's finance tests need `PYTHONPATH=packs/finance_ap/src`). Ruff on the changed files and lint-imports (5
  contracts) are clean.
- **Merge notes.** Two strings move from `FAILED` to `INCONCLUSIVE`; one meta branch in the `ci_run` case.

## 81. M4 R-4: M4 gate reports name the traces and the code they scored

- **Item.** R-4 of `docs/research/improvements/2026-10-27/M04-signals.md` (should), as recommended, with one placement
  change: `signals` may import only core (it sits beside `connectors`, below `studio`), so the code digest uses
  `core.build.source_digest` of the pack's source package instead of `studio.runs.pack_hash`, and the coding gate script
  (a pack, free to import `connectors`) adds the redaction fields. Not taken: refusing on a mismatch (needs pins written
  with the labels, L-8), the TraceStore head key (coding only, includes carried signals). Pattern from Repo2RLEnv
  `quality/report.py:64-74` and OpenEnv `validation/runner.py` (ideas); no code copied.
- **Files.** `src/crucible/signals/gate.py` (new `inputs`, `code`, module docstring),
  `packs/_toy/src/crucible_pack_toy/m4/gate.py` (`run`), `packs/coding/src/crucible_pack_coding/m4/gate.py` (`run`),
  `docs/modules/M04-signals.md` (Exit gate), `tests/unit/test_gate_inputs.py` (new).
- **Public names.** New `crucible.signals.gate.inputs(traces) -> {"traces": {id: digest}, "traces_sha256": str}` and
  `crucible.signals.gate.code(pack_src) -> {"git_sha", "git_dirty", "source_sha256", "pack_sha256"}`.
- **Behaviour.** Both gate reports gain `inputs` (per labelled trace the `content_hash()` of the trace before
  extraction with its signals cleared, and a sha256 over the sorted `id\0digest` lines) and `code`; the coding report
  also gains `redaction` = {`version`: `REDACTION_VERSION`, `salt_id`: `pseudonym("salt", "id", tenant_salt())`}, so a
  report from another tenant (other salt, other digests) reads as not comparable. Reports named only `labels_sha256`:
  an edited toy reply moved type accuracy 1.0 -> 0.988 under the same hash, and the coding gate's next rebuild scores
  whatever changed upstream since 2026-10-09. Metrics, thresholds and labels are unchanged; nothing is refused.
- **Formats.** `m4_gate_*.json` reports gain `inputs`, `code` and (coding) `redaction` keys. The committed reports are
  not regenerated here.
- **Tests.** `test_gate_inputs.py` (3: two toy runs give equal `inputs`, 12 traces, and `code` fields; editing one event
  of T1013 moves only that trace's digest and the combined one, under the same labels hash; carried signals do not
  change a digest; the coding report carries `redaction.salt_id` for the tenant's salt, with a stub store). It errors
  before the fix (no `inputs`). M4, M4 packs, R-3 and core audit files: 139 passed. Ruff on the changed files and
  lint-imports (5 contracts) are clean.
- **Merge notes.** The coding gate's `traces` dict is built from a `raw` list (same ids, same order).

## 82. M4 R-5: one CI-check identity in `verdict`; a later pass never contradicts a pass

- **Item.** R-5 of `docs/research/improvements/2026-10-27/M04-signals.md` (should), as recommended. It restores the
  pre-R9 rule for passes. Not taken: the check-suite identity of L-7 (two workflows' same-named jobs as one flaky check in
  the mapper; an abstention, not a wrong sign). Pattern from verifiers' attempt ids (`verifiers/v1/trace.py:565-589`,
  MIT); no code copied.
- **Files.** `src/crucible/signals/reliability.py` (new `_same_check`, `verdict`, module docstring),
  `packs/coding/src/crucible_pack_coding/signals.py` (`check_result` meta gains `app`), `docs/modules/M04-signals.md`
  (Reliability), `tests/unit/test_ci_inconclusive.py` (+2).
- **Public names.** none.
- **Behaviour.** `verdict` matches a CI check by (`meta.app`, `meta.check`), the identity the coding mapper already uses
  for flaky (it now records `app` in the signal's meta); a source without an app (CI logs) matches by name as before. A
  pass is contradicted only by a later failure of the same check on the same step, not by any later run: a pass then a
  pass gave [False, True] and now gives [True, True] on a good trace; an Azure failure then an Actions pass of "build" on
  one commit gave the failure False and now None. A failure is still contradicted by a later pass of the same check.
- **Formats.** PR `check_result` meta gains `app`.
- **Tests.** `test_ci_inconclusive.py` (+2: pass then pass confirms both; two apps are two checks, one app with both
  outcomes is still flaky; both failed before the fix). M4, M4 packs, gate inputs, graded-wrong, label, extract-failed,
  studio, M5, M8, M14, coding connector, agent-run and core files: 43 + 342 passed, 11 skipped. Ruff on the changed files
  and lint-imports (5 contracts) are clean. With R-5 the M4 should items (R-1..R-5) are all done; L-1..L-9 wait for a
  consumer.
- **Merge notes.** Two list comprehensions in `verdict` now call `_same_check`.

## 83. Part 1 A1: the workspace harness records the level-1 hack flags and malformed tool calls

- **Item.** A1 of `docs/research/env-synthesis/part1-framework-improvements.md` (§0 item 1, §1.1, §2(a); also §3.2 #3,
  first half), record-only as the doc says. Not taken: the re-scan of stored episodes (flag rate per rule; wait for the
  first training runs), the `__eq__` rule (§3.2 #3, second half; its own item). Pattern: `NativeHarness.run`
  (`runtime/harness/native.py:917-926`); no new code paths.
- **Files.** `src/crucible/runtime/harness/workspace.py` (`WorkspaceHarness.run`),
  `tests/unit/test_part1_workspace_flags.py` (new).
- **Public names.** none.
- **Behaviour.** After the episode, before `save_episode`, `WorkspaceHarness.run` runs `rule_flags(world.to_trace())` and
  `format_errors_by_call(trace, ep.model_calls)`, as `NativeHarness.run` does. Before, this override skipped both, so a
  student-path rollout (`factory.envs.make_env`, mode `workspace`: Tinker and DeltaAI) carried only the grade's flags and
  every `Turn.format_errors` was 0. Now `RolloutRecord.flags` holds the level-1 flags (rule flags first, then the
  grade's), `reward_components["hack/intercepted"]` counts intercepted calls, and turns carry their malformed-call
  counts. The reward does not change: `detector_group_fn` (level 2) has no caller, so `hack/confirmed` is never set, and
  the Tinker format penalty is 0 by default. Side effect: `studio.datasets` skips a flagged rollout as a positive
  (`flagged`), so an honest episode that writes a test file is no longer a dataset positive. Re-grade and re-collect
  retries still carry no level-1 flags of the attempt that ran the agent (as before).
- **Formats.** none (existing fields are now filled).
- **Tests.** `test_part1_workspace_flags.py` (+1: a fake agent writes `tests/test_fib.py` and calls an unknown tool; the
  rollout has the `writes_test_file` flag and one format error, the reward stays 1.0; it failed before the fix with no
  flags). Harness-reward audit, M17 C harness, hack defence, M7 hardening, runtime, M10 lab audit R3/R5, M17 P6/W world,
  open risks, credit/grade agreement, smoke scripts, Tinker bridge, M6 audit, M17 debug langs, M15 studio: 330 passed,
  3 skipped, 3 failed, none from this change (two M15 studio failures known before; the R5 host-calls tick test is
  timing-flaky under load, 0/10 failures with and without the fix when alternated). Ruff check and lint-imports
  (5 contracts) clean.
- **Merge notes.** Nine lines after `res.steps, res.stop, res.truncated = ...` in `WorkspaceHarness.run`.

## 84. Part 1 §3.2 #1: the DeltaAI bridge logs `lora/changed_frac`, the share of LoRA weights a step changed

- **Item.** §3.2 #1 of `docs/research/env-synthesis/part1-framework-improvements.md` (§3.3; plan table line 670): "prove
  at the 2-step run that bf16 updates are not rounded away. Log `lora/changed_frac`". This is the measurement only. Not
  taken: the fix (an fp32 copy of the adapter and Adam moments on the bf16 base, or 4 trainer GPUs in fp32). It needs a
  prime-rl change and a GPU run, and waits for what this gauge shows on R1/R2. Background: swarm-pipeline/14 §8.2 (2
  trainer GPUs need `optimization_dtype = bfloat16`; fp32 ran out of memory, job 3479803).
- **Files.** `src/crucible/train/bridges/primerl/backend.py` (`lora_change`, `ADAPTER_FILE`, `_Trainer.forward_backward`,
  module doc), `tests/unit/test_part1_lora_change.py` (new).
- **Public names.** `backend.lora_change(prev, cur)`, `backend.ADAPTER_FILE`.
- **Behaviour.** After each step's adapter is hot-loaded, the bridge compares `broadcasts/step_<s>/adapter_model.safetensors`
  with `step_<s-1>`'s (prime-rl keeps both; for the first step the previous one is the startup adapter), one tensor at a
  time, in a thread. It adds to the step's metrics (`steps.jsonl` `fb_metrics`, the smoke's `metrics`):
  `lora/changed_frac` (share of LoRA elements whose stored value changed), `lora/changed_frac_A` and `_B`, `lora/norm_A`
  and `_B`, and `lora/check_s` (the comparison's time). prime-rl saves the parameters without a cast, so the files are
  what the trainer stores. Reading: B starts at 0, where bf16 is fine-grained, so its share is the control (an expert
  that got no tokens is unchanged in both kinds); A well below B means updates were rounded away. No adapter file
  (another weight transport, test fakes) gives no `lora/*` keys; any error logs a warning and the step goes on. Nothing
  else changes. On the Mac, a simulated rank-32 adapter with Adam-sized steps (±4e-5) reads A 0.71 / B 1.00 in bf16 and
  1.00 / 1.00 in fp32; two 1.1 GB adapters compare in about 6 s.
- **Formats.** new keys in the step's metrics (above).
- **Tests.** `test_part1_lora_change.py` (+3: a step of the learning rate on A = 0.02 changes 0% of A and 100% of B in
  bf16, 100% of both in fp32; the share reaches `forward_backward`'s metrics; no adapter file gives no keys, and an
  unreadable one logs a warning while the step completes). prime-rl bridge, routing, score centering (+paper), verify,
  DeltaAI vLLM, difficulty bridge, M12 bridges audit, Tinker bridge, train bridges: 229 passed, 10 skipped. Ruff check and
  lint-imports (5 contracts) clean.
- **Merge notes.** One function before `_Trainer`, eight lines after the `primerl/step` metrics in
  `_Trainer.forward_backward`, `import math`, `ADAPTER_FILE` after `FIRST_STEP`.
