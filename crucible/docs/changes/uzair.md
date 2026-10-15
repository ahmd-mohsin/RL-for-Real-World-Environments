# Branch `uzair`: change log

What this branch changes relative to `main`, kept for merging with other work. One entry per change, newest last.
Each entry names the audit item it implements, the files and public names it touches, any change to persisted
formats or behaviour, the tests that cover it, and where a merge is likely to conflict.

- Base: `main` at `c27f5ca` (2026-10-15).
- Last merged from `main`: `c27f5ca`.
- Source of the items: `docs/research/improvements/2026-10-14/M00-core.md` (R-1 to R-31).

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
