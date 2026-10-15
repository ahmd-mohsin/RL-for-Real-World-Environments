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
