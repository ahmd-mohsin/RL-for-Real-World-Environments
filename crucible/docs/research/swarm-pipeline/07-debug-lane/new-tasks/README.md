# New debug-lane tasks (Uzair, from 2026-10-28)

New training tasks built from repositories never used for a debug task before. Each one is mined with the pipeline
(`lh_debug.py mine`), gets a bug report we write, and is measured on Tinker. Paths under `crucible/data/debug-uzair/`
and `crucible/data/targeting/uzair*` in the notes are on Uzair's machine (gitignored).

## Rules for these tasks

- A repository not used before, permissive licence, fixes after 2026-05-01. CPU only (no GPU tasks).
- **At least 3 bugs** per task.
- **At least one real link** between bugs: one bug's own hidden test fails until another bug is fixed too. `--no-link`
  (allowed while fixing Ahmed's tasks) is not used here. Two bugs' cases side by side in one coarse test, or a slow or
  concurrency test that fails under load in every reverted run, is not a link.
- The keep rules are the lane's: training = student 10-60% full passes over 8 attempts, at least half of its failures on
  the trained skill, a strong model passes. Checklist before measuring: `../fixing/README.md`.

## Tasks

| Task | Repository | Bugs | Link | Result |
|---|---|---|---|---|
| `db-go-huma-815e1ff8` | danielgtaylor/huma (Go) | 4 | m04 needs m03 (JSON-array defaults on an exploded query array need absent-array defaults) | **kept for training**: student 4/8, failures on skill 4/4, Qwen3.5-397B 2/2 (`notes/huma-815e1ff8.md`) |

## Repositories screened (2026-10-28, since 2026-05-01)

| Repository | Real fixes | Result |
|---|---|---|
| danielgtaylor/huma | 44 fix units, one linked group | the task above |
| PyCQA/isort | one linked group | 3-bug task built, being measured |
| golang/net | one linked pair (http2) | 3-bug task built (the pair + a third http2 fix), being measured |
| valyala/fasthttp | 39 | no two fixes linked |
| getkin/kin-openapi, apache/datafusion-sqlparser-rs | | no linked group |
| golang/crypto | 52 at 3 of 6 B points | no linked group; stopped early |
| tobymao/sqlglot | 46 | links were fake: `test_max_nodes` (62 s) timed out under load in every reverted run, and the rest were cases of one large test method |
| google/cel-go | 7 | links were fake: a concurrency test that flaked in every reverted run |

## Pipeline fixes made while building them

- The leak probe counted isort's two example plugins (separate distributions in the repository root, installed from
  PyPI as test dependencies) as the project's own installed source and refused the window. A sub-project now counts only
  when the task's fixes touch it (`windows_world.workspace_modules`, `window_touched`).
- A changelog line that rewords a fix's subject slightly ("at the start of imports" for "at top of imports") stayed in
  the start code. A subject of 6 or more words now also matches an entry holding all of its words but one
  (`langs/base.strip_ref_entries`).
- golang/net's `lif` and `route` packages build only on Solaris and BSD, so on Linux `go test` reports a failed build at
  the start code and at B alike, and the window was refused. A package whose files are all for another OS is no longer
  a failed build (`langs/golang.py`).
