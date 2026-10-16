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
