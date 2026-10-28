# New debug-lane tasks (Uzair, from 2026-10-28)

New training tasks built from repositories never used for a debug task before. Each one is mined with the pipeline
(`lh_debug.py mine`), gets a bug report we write, and is measured on Tinker. Paths under `crucible/data/debug-uzair/`
and `crucible/data/targeting/uzair*` in the notes are on Uzair's machine (gitignored).

## Rules for these tasks

- A repository not used before, permissive licence, fixes after 2026-05-01. CPU only (no GPU tasks).
- **3 to 5 bugs** per task (5 at most).
- **At least one real link** between bugs: one bug's own hidden test fails until another bug is fixed too. Two bugs'
  cases side by side in one coarse test, or a slow or concurrency test that fails under load in every reverted run, is
  not a link. For golang/net and isort the user relaxed this on 2026-10-28 (built with `--no-link`) after their linked
  versions could not reach the band (see the golang/net note).
- Every report quotes the exact outputs a hidden test compares, and no hidden test may call an internal function
  (isort's wrap-mode fixes were left out for that reason).
- The keep rules are the lane's: training = student 10-60% full passes over 8 attempts, at least half of its failures on
  the trained skill, a strong model passes. Checklist before measuring: `../fixing/README.md`.

## Tasks

| Task | Repository | Bugs | Link | Result |
|---|---|---|---|---|
| `db-go-huma-815e1ff8` | danielgtaylor/huma (Go) | 4 | m04 needs m03 (JSON-array defaults on an exploded query array need absent-array defaults) | **kept for training**: student 4/8, failures on skill 4/4, Qwen3.5-397B 2/2 (`notes/huma-815e1ff8.md`) |
| `db-go-golang-net-6c36e4c8-e2` | golang/net (Go) | 5, in five packages (quic, http2, dns/dnsmessage, xsrftoken, html) | none (relaxed) | **kept for training**: student 4/8, failures on skill 4/4, Qwen3.5-397B 2/2 (`notes/golang-net-6c36e4c8-e2.md`) |
| `db-python-isort-f66fa65b` and variants | PyCQA/isort (Python) | 3 / 3 / 5 | m03 needs m02 in the first | **dropped**: the linked version was budget-bound (student 3/8, all 5 failures at the 300-call limit on the `check_code` bug); without that bug every usable fix is solved 7-8 of 8 times (3 bugs: 3/3, 5 bugs: 6/8), and the other fixes' tests call internal functions |

## Repositories screened (2026-10-28, since 2026-05-01)

| Repository | Real fixes | Result |
|---|---|---|
| danielgtaylor/huma | 44 fix units, one linked group | the task above |
| PyCQA/isort | one linked group | measured and dropped (above) |
| golang/net | one linked pair (http2) | the linked versions could not reach the band (note); the unlinked 5-bug task is kept |
| mswjs/interceptors | | no linked group |
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

## Notes for the lead

- **DeepSeek-V4.1-Flash on Tinker is very slow on long prompts now** (2026-10-28 probe: a fresh 60K-token prompt took
  375 s; the newer `DeepSeek-V4.1-Flash-1017` build took 6 s and Qwen3.5-397B 5 s). Agent attempts at 50-130K tokens
  stalled until the 90-minute limit. `lh_debug_target.FRONTIER` still names the old build; these runs used
  `--frontier "Qwen/Qwen3.5-397B-A17B:peft:262144"`.
- **The 1017 build's Hugging Face repo has no `chat_template.jinja`** (its tokenizer files are byte-identical to the old
  build's), so Crucible cannot render prompts for it. Used here only through a local cache copy of the old template with
  `HF_HUB_OFFLINE=1`; the proper fix is a tokenizer override for that row in `train/bridges/tinker/models.py`.
- **A single Tinker sampling call has no time limit of its own**: a stalled call blocks an attempt until `agent_s`.
