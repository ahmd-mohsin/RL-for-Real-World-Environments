# M3 study: event logs, databases, repositories, CI logs, ingest idempotence

Read-only comparison study for the M3 audit. HEAD `dc29c0f`. Every Crucible defect below was reproduced with a
probe run against the repo's `.venv` (scratch scripts, not committed). "Observed" is real output.
Line numbers refer to HEAD. While this study ran, uncommitted edits appeared in `base.py`, `file_drop.py` and `sql_audit_log.py`.
They touch D9 (file_drop sort only, not github_pr), D12 (BOM, missing case key, per-ref isolation), D18 (path quoting) and D19 (NULL case, stale ref).
Re-run those probes against the final tree. The other items are untouched.
Paths: `C/` = `crucible/`, `CP/` = `crucible/packs/coding/src/crucible_pack_coding/`, `FW/` =
`RL-Environments/06-frameworks-and-env-code/`, `AE/` = `agentenv-framework/src/agent_env/`.

## 1. Mechanism table

| Framework | Mechanism | Where |
|---|---|---|
| SWE-smith | PR mirror. Paginated GitHub API loop with rate-limit sleep | `FW/swe-bench-swe-smith/swesmith/bug_gen/mirror/collect/utils.py:49-75, 110-170` |
| SWE-smith | Linked issues from PR title, body and commit messages (keyword + `#N`, comments stripped) | same file `:77-108` |
| SWE-smith | Patch split into fix and test by path words | same file `:309-326` |
| SWE-smith | Resume. Skips PRs already in the output file | `.../mirror/collect/build_dataset.py:115-150` |
| SWE-smith | Log parsers per repo profile: pytest, xdist `[gwN] STATUS name`, CTest, GTest, Catch2, Boost.Test, QTest, lit | `swesmith/profiles/python.py:93-102, 164-172`; `profiles/cpp.py:24, 449-490, 491-560, 567, 659, 736-749, 752` |
| SWE-smith | CTest with no per-test lines invents `synthetic_pass_i` names | `profiles/cpp.py:473-484` |
| SWE-smith | Flaky repair. Drops gold-failing tests from F2P/P2P | `swesmith/harness/repair.py:1-20, 138` |
| repo2rlenv | GitHub and GitLab behind one provider. GitLab MR list uses `diff_refs.base_sha`. Single page only | `FW/huggingface-repo2rlenv/src/repo2rlenv/provider.py:18-22`; `gitlab.py:55-113, 115-134` |
| repo2rlenv | Pytest parser. Param ids with spaces, xdist prefix, summary-first and progress-first forms | `src/repo2rlenv/log_parsers/pytest_parser.py:32-113` |
| repo2rlenv | Local git log with `\x1f`/`\x1e` separators | `src/repo2rlenv/git_local.py:91-155` |
| EnterpriseOps-Gym | No ingest. SQL is read only by verifiers through an MCP sql-runner. Resume by output-file existence | `FW/servicenow-enterpriseops-gym/benchmark/verifier.py:81-127, 195`; `benchmark_utils.py:51-57` |
| Harbor | Content hash over a sorted file manifest (`rel\0sha\n`) | `FW/harbor-framework-harbor/src/harbor/publisher/packager.py:73-95` |
| Harbor | Idempotent upload. Upsert on stable ids (`on_conflict="id"`, `"trial_id,model_id"`), plus a `skipped` sentinel | `src/harbor/upload/db_client.py:387-416, 560-562, 628-630`; `upload/uploader.py:34-43` |
| NeMo-Gym | Streams JSONL line by line. Bad lines go to `offending_example_idxs` and the run continues | `FW/nvidia-nemo-gym/nemo_gym/train_data_utils.py:292-301, 559-586` |
| NeMo-Gym | Failure-kind vocabulary (`transport_*`, `provider_oom_killed`, `kill_shaped`, `cancelled`) | `nemo_gym/failure_kinds.py:15-33, 60-128` |
| agentenv | Versioned artifact store with monotonic version allocation. Latest = highest version | `AE/artifact/store.py:35-50, 112-126` |
| agentenv | Validated namespaced ids with bounded, hashed derived keys | `AE/store/ids.py:1-80` |
| git-swarm | Website only (`index.html`, figure JSON). No ingest mechanism | `FW/veds12-git-swarm/README.md` |
| prime-environments | No log, repo or DB ingest found (grep for `parse_log`, `junit`) | n/a |
| **Crucible** | Connector protocol and `ingest()`: fetch, normalize, redact, put. No per-ref isolation | `C/src/crucible/connectors/base.py:45-54` |
| **Crucible** | `file_drop` CSV/JSONL. `rows_to_trace` shared with `sql_audit_log` | `C/src/crucible/connectors/file_drop.py:46-104` |
| **Crucible** | `sql_audit_log`. SQLite `file:...?mode=ro` URI, or psycopg2 | `C/src/crucible/connectors/sql_audit_log.py:21-69` |
| **Crucible** | TraceStore. Content-addressed. "Latest" = newest `created_at` of first insert | `C/src/crucible/connectors/trace_store.py:18-25`; `C/src/crucible/core/store/__init__.py:211-245` |
| **Crucible** | `github_pr`. Offline git discovery, superseded commits, blame, 30-day post-merge tracking | `CP/connectors/github_pr.py:77-125, 154-355`; `CP/git_history.py` |
| **Crucible** | GitHub API cache (never expires, first page only) | `CP/github_reviews.py:30-75` |
| **Crucible** | `ci_logs`. Format detection, five parsers, P16 infra masking, `flaky()` | `CP/connectors/ci_logs.py:37-255` |
| **Crucible** | `policy_docs`. Modality regexes, sections, config rules | `CP/connectors/policy_docs.py:32-215` |
| **Crucible** | M3 gate idempotence check | `CP/m3/gate.py:215-231` |

## 2. Defect candidates (all reproduced)

### High

**D1. A failing CI run is labelled `good` when the parser misses the failure.** `ci_logs.py:101-102, 85, 218-223`.
Counts come from parsed tests only. The runner's own summary is used only when nothing parsed, so it is never
checked against the parsed counts.
- Probe A (pytest param id with a space): `tests/t.py::test_ok PASSED [ 50%]` / `tests/t.py::test_p[a b] FAILED [100%]` /
  `FAILED tests/t.py::test_p[a b] - assert 0` / `=== 1 failed, 1 passed in 0.1s ===`. Observed: `good`, 1 passed, 0 failed.
- Probe B (GTest value-parameterized): `[  FAILED  ] Inst/Suite.Test/1, where GetParam() = 4 (0 ms)` plus one OK line.
  Observed: `good`.
- Better: repo2rlenv `log_parsers/pytest_parser.py:40-46` (name group allows `[...]` with spaces). Also reconcile with
  `summary_counts()`: if the summary says more failures than were parsed, the run is `bad`.

**D2. Infra masking scans the whole log, so real test failures become `unknown`.** `ci_logs.py:37-44, 224, 234-235`.
- Probe: `tests/t.py::test_api FAILED` / `FAILED tests/t.py::test_api - requests.exceptions.ConnectionError: [Errno 111] Connection refused`
  / `=== 1 failed, 1 passed in 0.1s ===`. Observed: `failure_kind=infra`, outcome `unknown`. A code failure is masked (P16 inverted).
- Better: NeMo-Gym ties a kind to the stage that failed (`failure_kinds.py:49`). Match infra markers only outside
  test-failure sections, or only in runner lines.

**D3. A pytest collection error is `unknown`, not `bad`.** `ci_logs.py:102, 141, 156`.
- Probe: `ERROR tests/test_a.py` / `!!! Interrupted: 1 error during collection !!!` / `=== 1 error in 0.12s ===`.
  Observed: format `unknown`, outcome `unknown`. A syntax error in the PR is masked.
- Cause: `PYTEST_SHORT` needs `::`, and the summary line needs `passed|failed`. Better: repo2rlenv `pytest_parser.py:81-96` accepts any node id.

**D4. A run killed mid-way is `good`.** `ci_logs.py:37-44, 239`.
- Probe: `tests/t.py::test_ok PASSED` / `Error: Process completed with exit code 137.` Observed: `good`.
- No check that a final summary line exists, and no `exit code 137` / `Killed` pattern. Better: NeMo-Gym `kill_shaped`, `provider_oom_killed` (`failure_kinds.py:96-98`).

**D5. `#12` matches `#123`, so another PR's revert is recorded as a revert of this PR.** `github_pr.py:341-342`.
- Probe (git history): PR #12 merged as `Merge pull request #12 from alice/feat`. Three days later on `main`:
  `Revert "Tweak launch bounds (#123)"`, changing a base line of `gpu/k.cu` that #12 did not write.
  Observed: `post_merge_commit` with `is_revert=True, mentions_pr=True, touched_lines={}`. M4 labels the outcome from this.
- Also `_norm(title)[:40] in text`: a short title such as "CUDA" matches any revert that mentions CUDA.
- Fix: `re.search(rf"#{n}\b|/pull/{n}\b")`. No reference does better.

**D6. A rename after merge counts as a fix that touches every PR line.** `CP/git_history.py:66, 82, 92` (`--no-renames`); `github_pr.py:297-337`.
- Probe: after the merge, `git mv gpu/k.cu gpu/kernels.cu` with message `fix: move gpu/k.cu to gpu/kernels.cu`.
  Observed: `post_merge_commit` with `fixlike=True, touched_lines={'gpu/k.cu': [(3, 4)]}`. After the rename, later edits to the new path are not tracked.
- Fix: diff with `-M`, then carry ranges to the new path.

**D7. `discover_git` + offline payload crash on any repo whose default branch is not `master`.** `github_pr.py:82, 124, 301`.
- Probe: the same repo with branch `main`, scope `{"branch": "main"}`. Observed: `RuntimeError: git log ... bad revision '<merge>..master'`.
- `_git_payload` hard-codes `"ref": "master"`, and `_after_merge` falls back to `"master"`.

**D8. Re-ingesting a PR is not idempotent unless `now` is pinned.** `github_pr.py:54, 145, 151, 237`.
`fetched_at = datetime.now()` goes into `TraceSource`, so it goes into the content hash.
- Probe: two `GitHubPRConnector(..., offline=True)` with no `now`, same ref. Observed: hashes differ.
  The gate passes only because it pins `NOW` (`gate.py:42, 51`).
- Better: Harbor keys on stable ids and upserts (`db_client.py:560-562`). Keep `fetched_at` out of the hashed body.

**D9. Events are ordered by string time, so mixed offsets and epoch numbers misorder them.** `file_drop.py:80`; `github_pr.py:218, 266`.
- Probe: rows at `2026-03-02T10:30:00+02:00` (08:30Z), `09:00:00Z`, `08:00:00Z`. Observed order: 08:00Z, 09:00Z, 08:30Z.
  Epoch `999` vs `1000` sorts `1000` first (also via SQLite integer columns).
- Git `%cI` keeps the committer's offset while API times are `Z`, so `github_pr` hits this on real PRs.
- Generic M4 attribution is "the last writer", so wrong order grades the wrong step.
- Better (internal): `C/src/crucible/actors/records.py:88-94` compares instants, not strings.

**D10. Event seq is not stable when a log grows, and the source row id is dropped.** `file_drop.py:84-104`; `trace_store.py:18-25`.
- Probe: ingest `{"id":"1","time":"...T01:00Z","text":"refund approved"}`, then the same log plus an earlier row id `0`.
  Record `fd:T1#0` was "refund approved". It is now "customer opened". `event.data` holds no `id`.
- Probe: store v1, store v1 plus a Signal (M4's write-back), then ingest the grown log. Observed: `get()` returns 0 signals. The newest version replaces the signalled one.
- Fix: keep the source row id in `event.data`, and make Record ids `trace#<row id>`. Merge signals on re-ingest, or re-run M4.

**D11. Postgres access is not read-only, and the query is committed.** `sql_audit_log.py:22-28`.
- `with psycopg2.connect(dsn) as con` commits on exit and does not close the connection. psycopg2 runs multi-statement strings, and no `set_session(readonly=True)` is set.
- Probe (needs Postgres): scope query `UPDATE audit_log SET actor='x' RETURNING *` (or `SELECT 1; DELETE FROM audit_log`). The change is committed.
- `fetchall()` loads the whole table, with no server-side cursor and no keyset paging.
- Better: EnterpriseOps-Gym sends SQL only through a separate runner (`verifier.py:195`).

### Medium

**D12. `ingest()` aborts the whole batch on one bad source.** `base.py:51-53`. Earlier refs are stored and later ones are not, with no report.
Exposing inputs (each raises, observed):
- CSV with a BOM (`﻿case,time,...`): `KeyError 'case'`. Fix: `encoding="utf-8-sig"`.
- Latin-1 CSV (`Jos\xe9`): `UnicodeDecodeError`.
- `export.CSV` (upper case) is parsed as JSONL: `JSONDecodeError`.
- JSONL text with a raw U+2028 (valid JSON, `ensure_ascii=False`): `str.splitlines()` splits it, `JSONDecodeError`. Fix: `split("\n")`.
- A row without the case column: `KeyError`.

Better: NeMo-Gym `train_data_utils.py:559-586` streams `for line in f` and records bad line numbers.

**D13. `file_drop` reads every file again for each case.** `file_drop.py:63-71` (discover reads all, fetch re-reads all).
The cost is O(cases x bytes), with whole files held in memory. 10^5 tickets in a 1 GB export means 10^5 full reads. Fix: group in one streaming pass and cache by file digest.

**D14. Duplicate rows from overlapping exports become duplicate events.** `file_drop.py:65, 71`.
- Probe: `day1.jsonl` = {id 1 opened}; `day2.jsonl` = {id 1 opened, id 2 closed}. Observed events: `opened, opened, closed`. Fix: dedupe on `id`, or on the row hash.

**D15. Equal timestamps without an id: the hash depends on the export order.** `file_drop.py:80`.
- Probe: two rows with the same `time`, no `id`, in swapped order. Observed: different content hash.
  A re-export in another order creates a new version. Fix: tie-break on the canonical row hash.

**D16. "Latest" means first insert, so going back to earlier content never becomes latest.** `store/__init__.py:217, 224, 245`; `trace_store.py:22-25`.
- Probe: put v1, put v2 (grown), put v1 again (an audit row deleted, or an older export). Observed: `get()` returns v2.
- Better: agentenv allocates versions monotonically (`AE/artifact/store.py:112-126`).

**D17. Trace ids collide.**
- With `family` set, `file_drop` and `sql_audit_log` both give `toy/support_ticket:T1` (`file_drop.py:103`). They overwrite each other with no merge.
- Without `family`, the same ticket from both sources gives two traces and no cross-connector dedupe.
- Case `1` (int) and `"1"` merge, while `1.0` splits.
- `ci:<relative path>` collides across roots, and across reruns that write to the same path (`ci_logs.py:202, 240`). `flaky()` then sees only the last run.

**D18. A SQLite path is put into a URI unescaped.** `sql_audit_log.py:29`.
- Probe: dsn `/tmp/x/audit#2026.db`. Observed: SQLite opens `/tmp/x/audit`, **creates that empty file** (`mode=ro` was in the fragment), and a query on `sqlite_master` returns `[]` silently.
- `?` gives "no such table", and `%41` gives "unable to open". A SQLAlchemy DSN `postgresql+psycopg2://` falls through to SQLite.
- Fix: `Path(p).resolve().as_uri() + "?mode=ro"`.

**D19. NULL keys in an audit table.** `sql_audit_log.py:51-55, 64-66`.
- A NULL case id gives one trace `None` that mixes unrelated rows (observed: bob's invoice:2 delete and eve's vendor:9 insert).
- A NULL `row_id` gives no `writes`, so attribution is dropped silently.
- A ref whose case was deleted raises `KeyError 'INV9'`.

**D20. Non-RFC 3339 times are accepted at ingest and fail later.** `file_drop.py:98`.
- Probe: `"time": "03/02/2026 09:00"` is stored as is. Then `RecordStore.from_traces` raises `Invalid isoformat string`.
- psycopg2 `datetime` values become `str()` with a space and no `T`. Fix: normalize with `world.clock.parse_time` at ingest and reject bad rows into quarantine.

**D21. Actor kind and role.** `file_drop.py:86-88`.
- `actor_kind: "Agent"`, `"bot"` or `"AI"` becomes `human`. Observed: `('copilot','human','participant')`. Bot signals then count as human expert signals.
- An actor's role is fixed by its first row, so `ana` customer then approver stays `customer`.

**D22. CI logs as downloaded from CI lose every per-test check.** `ci_logs.py:65, 85, 101-102, 117, 149-156`. Every pattern is `^`-anchored.
- A GitHub Actions timestamp prefix (`2026-10-11T12:00:00.1234567Z tests/t.py::test_ok PASSED`) or ANSI colour (`\x1b[32mPASSED`) gives zero test checks. Observed: only summary counts survive, so `flaky()` and per-test attribution are lost.
- A JUnit file with a UTF-8 BOM or in UTF-16 is detected as `unknown` (observed). The cause is `lstrip()` keeping `﻿` and `read_text` assuming UTF-8.
- A pytest log containing "all tests passed, thanks" is detected as CTest. Observed: `unknown` for a failing run.
- Fix: strip `^\S+Z ` and ANSI codes first; parse JUnit bytes with `ET.fromstring(bytes)`.

**D23. GitHub API reads are incomplete or stale.** `github_pr.py:147-149`; `github_reviews.py:44-59`.
- Reviews, review comments and check runs read page 1 only (`per_page=100`). A PR with 150 comments is truncated silently.
- The cache never expires. Check runs cached while in progress (`completed_at: null`) are skipped forever (`github_pr.py:207`).
- The cache write is not atomic (`p.write_text`). No 429 handling or secondary-limit wait.
- Better: SWE-smith `utils.py:110-170` (paginate, rate-limit sleep).

**D24. Gate idempotence is checked only in the easy case.** `gate.py:215-231`. Same process, same order, pinned `NOW`, unchanged sources.
D8, D10, D14, D15 and D16 all pass it. Add probes for reorder, growth, a fresh process with `now=None`, and rollback.

**D25. Git path parsing drops or renames files.** `git_history.py:71, 97-99`.
- Observed `added`: `{'gpu/k.cu': ..., 'gpu/my file.cu\t': ...}`. Git appends a tab to names with spaces.
- `gpu/café.cu` is missing: core.quotePath gives `+++ "b/gpu/caf\303\251.cu"`, which fails the `+++ b/` test.
- Fix: `-c core.quotePath=false` and strip the trailing `\t`, or use `--numstat -z`.

### Low

**D26.** All PR commits are attributed to the PR author, even commits a maintainer pushed (`github_pr.py:194`). `git_author` is kept only in data.

**D27. `policy_docs` false positives and negatives.** `policy_docs.py:33-35, 79-112, 125, 129-134`. Observed:
- "Do not hesitate to open an issue." → `must_not`. "We are always happy to review" → `must`. "Please see the build docs" → `should`. "The required packages are listed" → `must`.
- 4-space indented code `// never call free()` → `must_not`. A `~~~` fence body → `must_not`. A Markdown table row → `must_not`.
- `.. warning:: Kernels must not allocate` is dropped (all directive bodies are skipped).
- CODEOWNERS `*.cu @gpu-team # GPU kernels` gives owners `['@gpu-team','#','GPU','kernels']`.
- A YAML list in `.clang-tidy` (`  - 'bugprone-*'`) adds an empty check `''`.

**D28. CTest lines are missed.** `ci_logs.py:65`. `***Exception: SegFault` and `***Not Run (Disabled)` lines are missed (observed: `segv` absent). They are recovered only when the "The following tests FAILED" block is present.

**D29. Flakiness inside one run is not seen.** `ci_logs.py:53-58, 101`. Surefire `<flakyFailure>`/`<rerunFailure>` and pytest `RERUN` count as plain passes (observed: `good`, no flaky mark).

**D30.** The `FIX_BLOCK` regex treats any code block after "use" or "should be" as a suggestion (`github_pr.py:32, 250`). `discover_git` can yield one PR key twice (re-landed squash), and the second overwrites the first.

## 3. Features Crucible lacks (ranked)

1. **`issues` connector (GitHub/Jira/Linear).** Listed in ARCHITECTURE M3 and missing.
   Ref: SWE-smith `mirror/collect/utils.py:77-108` (PR to issue links), `:171-200` (`get_all_issues`); repo2rlenv `gitlab.py:126`.
   Owner: new `CP/connectors/issues.py`.
2. **`gitlab` connector.** Listed and missing. Ref: repo2rlenv `gitlab.py:55-113` (MRs, `diff_refs.base_sha`), `provider.py:18-22` (one provider interface).
   Owner: `CP/connectors/github_pr.py`. Split the API part behind a provider; the git part already is host-neutral.
3. **Paginated, resumable API ingest with rate-limit waits.** Ref: SWE-smith `utils.py:49-75, 110-170`; `build_dataset.py:115-150`. Owner: `CP/github_reviews.py`.
4. **Per-record quarantine and streaming.** Ref: NeMo-Gym `train_data_utils.py:559-586`. Owner: `C/src/crucible/connectors/base.py` (`ingest` returns stored keys plus failures), `file_drop.py:46-50`.
5. **Stable event identity and an upsert/merge policy.** Source row ids on events, signals carried across versions, and monotonic versions.
   Ref: Harbor `db_client.py:560-562, 628-630`; agentenv `artifact/store.py:112-126`. Owner: `trace_store.py`, `file_drop.py`.
6. **A source digest on `TraceSource`** (file manifest hash, row count, max time, SQL watermark), so a re-ingest can say "same source" without diffing traces.
   Ref: Harbor `packager.py:83-95`. Owner: `C/src/crucible/core/schema/trace.py:22-27`.
7. **`vcs_history` as its own connector.** Reverts and hotfixes over the whole branch via blame, not only inside one PR's 30-day window.
   Listed in ARCHITECTURE. Today it lives in `github_pr._after_merge` (`github_pr.py:283-355`). No reference has it.
8. **More test formats.** Catch2, Boost.Test, QTest, lit, Python unittest. Ref: SWE-smith `profiles/cpp.py:24, 230, 567, 659, 752`. Owner: `CP/connectors/ci_logs.py:163`.
9. **A shared failure-kind vocabulary for CI infra** (transport, quota, OOM-killed, cancelled). Ref: NeMo-Gym `failure_kinds.py:60-128`. Owner: `ci_logs.py:37-44`.
10. **`mini_swe_agent_runs`** is present under the name `agent_runs` (`CP/connectors/agent_runs.py`). ARCHITECTURE `:379, 510` should say so.

## 4. Where Crucible is stronger

- **Redaction before storage, with proof.** No reference redacts at ingest. Crucible plants 20 canary formats per connector, and checks the real `.env` values against the raw transcript and the store (`gate.py:68-151, 198-213`).
- **Content-addressed store that verifies on read** (`store/__init__.py:158-175, 203-207`). SWE-smith and repo2rlenv append JSONL.
- **PR history from git alone** (`github_pr.py:77-125`). Discovery needs no API. Rebased-away commits are recovered (`:163-171`).
  Review comments are attributed by blame at the reviewed commit (`:254-271`). Suggestion application is checked against the final file (`:272-278`).
  Post-merge line ranges are carried through later commits (`:296-337`). The SWE-smith mirror keeps only the final diff and splits test from fix by path words (`utils.py:309-326`).
- **P16 infra masking and cross-run `flaky()`** (`ci_logs.py:224-239, 245-255`). SWE-smith has neither in its parsers. Its flaky repair only drops tests after a gold run.
- **No invented test names for summary-only logs** (`ci_logs.py:10-12, 222-223`). SWE-smith makes `synthetic_pass_i` (`cpp.py:473-484`).
- **Non-UTF-8 git output is decoded with replacement** (`git_history.py:43`). **SQLite opens read-only by default** (`sql_audit_log.py:29`, path escaping aside).
- **Human actor ids are pseudonymized**, not only handles. Canonical JSON uses NFC, so one name has one hash (`C/src/crucible/core/canonical.py`).
