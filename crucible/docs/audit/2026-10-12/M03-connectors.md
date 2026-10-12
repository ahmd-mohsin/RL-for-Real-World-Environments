# M3 connectors: audit (2026-10-12)

Scope:
- `src/crucible/connectors/` (1,016 lines: base, redact, trace_store, file_drop, sql_audit_log, otel_import,
  chat_messages, capture_proxy);
- `src/crucible/core/agent_events.py`, the shared probe and write vocabulary;
- `src/crucible/interop/harbor/atif_import.py`;
- the coding pack's connectors (`github_pr`, `ci_logs`, `claude_code_sessions`, `agent_runs`, `policy_docs`, with
  `git_history` and `github_reviews`);
- the M3 gate.

The checks were against ARCHITECTURE.md §M3 and `docs/modules/M03-connectors.md`. The method is the M0 protocol:
- a requirement ledger;
- coverage;
- import contracts;
- one executable probe per suspected defect.

Every finding was reproduced. 80 regression tests were written; on the unmodified `HEAD`, 76 fail and 4 pass (see §2).
Three read-only studies compared M3 with the local frameworks:
- `M03-study-redaction.md`;
- `M03-study-capture.md`;
- `M03-study-logs-repos.md`.

## 1. Requirement ledger

| # | Requirement (source) | Status before | Evidence |
|---|---|---|---|
| R1 | Redaction inside the tenant removes credentials before anything is stored (ARCH, DESIGN §5.1) | **wrong** | <ul><li>Many fields were never redacted: `trace_id`, `source.url`, targets, refs, dict keys, and values under keys such as `password` or `Cookie`.</li><li>Many formats were missed: URL userinfo, short passwords, Basic auth, cookies, Azure keys, `SECRET_KEY=` style names, Google OAuth tokens, Vault tokens, Slack webhooks.</li><li>Connectors cut text before redaction, so a cut private key was stored (A4–A8, B1–B6).</li></ul> |
| R2 | Stable salted pseudonyms keep threads linkable without exposing people (M03 doc) | **wrong** | the default salt `"crucible"` was public, so a guessed address could be reversed (A10); names in text, git authors and @mentions survived the actor pseudonym (B9) |
| R3 | Exit gate: redaction removes 100% of planted canaries | live, but weak | <ul><li>Before: it passed with formats the redactor already knew, planted only in free text, and checked only whole values.</li><li>Hardened: 31 canaries, including structured fields and a private key longer than any text cap, checked by 12-character fragments.</li><li>The hardened gate found a leak in this audit's own fix (G1).</li></ul> |
| R4 | Exit gate: re-ingest is idempotent (same content hash) | partial | <ul><li>It held only with a pinned clock and an unchanged row order. The PR fetch time was hashed (D8), and equal-time rows were hashed in export order (D15).</li><li>Unchanged sources held. A grown log dropped M4's signals (D10), and "latest" meant the first insert (D16).</li></ul> |
| R5 | Exit gate: ≥ 30 PR traces, the task 01–04 construction transcript, every frontier run, 100% schema-valid | live | re-run: 55 PRs, 9,684 transcript events, 11/11 runs, 106/106 valid |
| R6 | `ingest()` stores nothing unredacted and isolates sources (M03 doc) | partial | one bad source aborted the whole batch without saying which (A18) |
| R7 | `file_drop`: any tabular export → one trace per case | **wrong** | <ul><li>Crashes: a BOM, Latin-1, `.CSV`, U+2028 in JSON, a row without a case key, or numeric text (A11–A14, D12).</li><li>Wrong order: events sorted as strings (A12).</li><li>Wrong data: overlapping exports duplicated events (D14), and non-ISO times were stored as-is and broke M2 later (D20).</li></ul> |
| R8 | `sql_audit_log`: SQLite read-only, Postgres | **wrong** | <ul><li>NULL case ids merged unrelated objects into one trace (A15).</li><li>A path containing `#` opened, and created, another database (A16).</li><li>On Postgres the query was committed, the connection never closed, and multi-statement or UPDATE queries ran (D11).</li></ul> |
| R9 | `capture_proxy`: records live agents, streaming and not (M03 doc) | **wrong** | <ul><li>A streamed 429 reached the agent as HTTP 200 (C2).</li><li>A failed or interrupted stream left no record (C3).</li><li>Cache markers and side calls produced false context resets that re-emitted the whole history (C1).</li><li>Harness text was recorded as the human's (C14).</li><li>Choices and tool-call names were merged in stream assembly (C5–C7).</li></ul> |
| R10 | `atif` import, including subagents and interrupts (M03 doc) | **wrong** | Claude Code subagent prompts became human messages, and their writes were booked to the main agent (C20); copied-context steps were re-emitted as new probes (C21); error flags were dropped (C23) |
| R11 | `otel_import` on the GenAI conventions (M03 doc) | partial | events out of time order (C28); repeated span events overwritten (C27); OpenInference spans ignored; absolute export paths raised (C29) |
| R12 | Answer-key probes: ran / exposed, never scored (M03 doc, P27) | partial | <ul><li>False negatives: `git --no-pager log --all` and argv commands (C15); "not found" in a commit subject marked the probe failed (C16).</li><li>False positives: any `/root` path and every web search counted as exposures, which relabel good runs (C17).</li></ul> |
| R13 | `writes_of`: which calls wrote which files (M4 attribution) | **wrong** | `view` counted as a write and as a test edit; bash writes, MCP and Gemini tools, and freeform `apply_patch` were missed (C18) |
| R14 | `ci_logs`: infra failures masked (P16), outcome `good` iff nothing failed | **wrong** | <ul><li>Failing runs labelled `good` when the parser missed failures (D1); a collection error was `unknown` (D3).</li><li>Infra masking hid code failures (D2); a killed run was `good` (D4).</li><li>Downloaded logs lost every per-test check (D22).</li></ul> |
| R15 | `github_pr` with `vcs_history` (reverts and hotfixes) | **wrong** | <ul><li>Reverts and fixes misattributed: `#12` matched `#123` (D5); a rename counted as a fix touching every PR line (D6).</li><li>Crash: a repository on `main` failed (D7).</li><li>Path parsing: names with spaces or accents were lost (D25).</li><li>API reads: page 1 only, and in-progress check runs cached forever (D23).</li></ul> |
| R16 | `claude_code_sessions`: freeze, then convert one session | partial | `freeze: False` merged every session of the project into each trace (C24); a frozen copy never refreshed (C25) |
| R17 | Raw data at rest is protected (M03 "Not done") | **wrong** | <ul><li>Capture logs and frozen transcripts used the default umask.</li><li>Store blobs were 0444 and the index 0644, readable by any local user.</li><li>There was no purge for versions stored before a redaction fix (B3, B11).</li></ul> |
| R18 | ARCH coding connectors: `github`, `gitlab`, `ci_logs`, `vcs_history`, `claude_code_sessions`, `mini_swe_agent_runs`, `policy_docs`, `issues` | partial | `gitlab` and `issues` are missing (deferred, as the M03 doc says); `vcs_history` lives inside `github_pr`; `mini_swe_agent_runs` is `agent_runs` |
| R19 | Import contracts | live | `lint-imports`: 4 kept |

## 2. Defects (reproduced)

Tests: `tests/unit/test_m3_audit.py` (A, B), `test_m3_audit_capture.py` (C), `test_m3_audit_logs.py` (D). On `HEAD`,
76 of 80 fail. The 4 that pass:
- Two URL-credential cases (A4). On `HEAD` the password was already hidden, but inside an e-mail pseudonym that anyone
  can recompute with the public salt. The sibling test `test_A4_url_userinfo_keeps_the_host` fails on `HEAD`.
- Two code-redaction cases (B10). They guard regressions this audit's first redaction pass introduced, which the
  redaction study caught.

Severity: P0 means stored secrets or wrong labels in normal use; P1 a common edge case; P2 rare.

### 2.1 Redaction and storage

| ID | Sev | Defect | Reproduction | Fix |
|---|---|---|---|---|
| A1 | P1 | The `sk-` rule had no word boundary: "task-04-gpu-migration-harness" became `ta<secret>` | probe1 | prefix formats start at a word boundary; a bare `sk-` key must contain a digit |
| A2–A3 | P2 | `threading.local()` became a host, `icon@2x.png` an e-mail, a version string an IP | probe1 | not followed by `(`; retina names excluded; octets bounded |
| A4 | P0 | `https://alice:pw@host` was parsed as an e-mail: the user stayed, the password went into a recomputable pseudonym | probe1 | URL userinfo → `<secret>@`, host kept |
| A5–A7 | P1 | Short passwords, Azure `AccountKey`, GCP `private_key_id`, cards and phones passed | probe1 | rules added (cards: issuer prefix + length + Luhn) |
| A8 | P0 | `trace_id`, `source.url`, targets, refs and dict keys were never redacted | e-mail case id | every string field and key |
| A9 | P2 | Redacting twice changed the hash (handles re-hashed) | `redact(redact(t))` | pseudonyms recognized |
| A10 | P0 | Pseudonyms were SHA-256 under the public salt `"crucible"`: a guessed address reversed | dictionary probe | HMAC under a random 0600 tenant key (`redaction.salt`, or `CRUCIBLE_REDACTION_SALT`); `salt` never serialized |
| B1 | P0 | Values under `password`, `Authorization`, `Cookie`, `DB_PASS` and similar keys survived whatever their form | study probe | key-aware masking, narrowed for traces (`max_tokens`, `sort_key` stay) |
| B2 | P0 | A private key cut by a connector's text cap lost its END line and was stored | study (about 1,000 characters reached the store) | unterminated keys matched, in their own pass first |
| B4–B6 | P1 | Basic auth, cookies, `SECRET_KEY=` / `DB_PASS=`, escaped quotes, URL-encoded, quoted passphrases, `ya29.`, `hvs.`, webhooks passed | study probes | one assignment rule that removes only the value and skips code values; formats added |
| B7 | P1 | `with_env_file` missed DSNs, `*CREDENTIAL*`, `*AUTH*` and kept inline comments in the value | study probe | names widened, comments stripped |
| B8 | P0 | Unanchored e-mail and assignment rules were quadratic: 200k characters took 18.6 s (1 MB about 8 min) | timing probe | anchored and bounded: 1 MB in < 0.6 s for every adversarial shape |
| B9 | P1 | Git author names, sign-offs, @mentions of known people, IBANs and SSNs survived | study probes | known people (ids, handles) rewritten in text; identity keys (`author`, `assignee`, ...) pseudonymized; IBAN (mod-97) and SSN rules |
| B10 | P2 | Over-redaction of code: `token = tokenizer.encode(x)`, `x@self.w`, `git@github.com:o/r` | study probes | code values, matrix products and ssh remotes kept |
| B13–B14 | P2 | Agent handles (model ids) were pseudonymized; `Alice` and `alice` became two people | probe | humans only; case-folded |
| B3, B11 | P0 | Store blobs 0444 and index 0644; no purge; no record of which rules redacted a version | stat; probe | `TraceStore`: 0700 store, 0600 index; every version stamped `redaction=<rules hash>`; `stale()`, `purge_superseded()`, `Store.forget_doc()` |
| G1 | P0 | (found by the hardened gate, introduced in this audit) the new cookie rule ate to the end of the line, including a private key's BEGIN line, so the key body was stored | gate: `private_key` leaked in 3 connectors | private keys first, in their own pass; cookie values as `name=value` pairs only |

### 2.2 Ingest, event logs, SQL

| ID | Sev | Defect | Fix |
|---|---|---|---|
| A11–A14, D12 | P1 | `file_drop` crashed on a BOM, Latin-1, `.CSV`, U+2028 in JSON, numeric text and rows without a case key | decoded and read as exports are written; caseless rows counted (`skipped`) |
| A12, D9 | P0 | Events sorted as strings: `10:00+02:00` came after `09:00Z`; epoch `1000` before `999`; also in `github_pr` | sorted by instant (`connectors.base.parse_time`) |
| A15, A16, A17 | P1 | NULL case ids merged unrelated objects; `#` / `?` in a path opened another database; a deleted case raised a bare `KeyError` | NULL cases skipped and counted; path quoted; `LookupError` |
| A18 | P1 | One bad source aborted the batch | every source tried; `IngestError` (stored keys + failed refs) or an `errors=` list |
| D10 | P0 | The row id was dropped; re-ingesting a grown log replaced M4's signalled version with one without signals | `data.row_id`; signals carried to the new version by event identity; `signals_dropped` counted |
| D11 | P0 | Postgres: the query was committed, never closed, multi-statement and UPDATE queries ran, whole table fetched | one SELECT only (checked), read-only session, server-side cursor, always closed. **Not run against a live Postgres** |
| D13–D15 | P1 | Each file re-read per case; overlapping exports duplicated rows; equal-time rows hashed in export order | parsed once per file; deduplicated by id or row hash; ties broken by row hash |
| D16 | P1 | "Latest" was the first insert time, so returning to earlier content never became the head | an explicit head tag per trace |
| D17 | P1 | `file_drop` and `sql_audit_log` gave one id to one ticket and overwrote each other; a rerun at the same CI path overwrote the first run | `sql_audit_log` ids carry the connector; CI ids carry a content digest |
| D20, D21 | P1 | `03/02/2026 09:00` stored as-is (broke M2's record store); `Bot` / `AI` actors became humans | times normalized to RFC 3339 (raw kept in `time_raw` when unparseable); kind aliases |

### 2.3 Agent-session capture and import

| ID | Sev | Defect | Fix |
|---|---|---|---|
| C1 | P0 | A rolling `cache_control`, a title side call or a re-rendered system prompt made a false `context_reset` and re-emitted the whole history (3 user turns for 1) | requests matched by longest prefix against any earlier request, markers and system turns ignored. A request that continues none is a side call (`data.thread`) if the main thread continues after it, else a reset |
| C2 | P0 | A streamed 429 / 5xx reached the agent as HTTP 200, with no `retry-after`, so its SDK never retried | the status and rate-limit headers pass through |
| C3–C4 | P0 | A stream that failed, or that the user interrupted (Esc), left no record; a lone surrogate lost the record after the upstream billed it | recorded in a `finally` (`complete: false`, `error`); ASCII-safe owner-only log lines |
| C5–C7 | P1 | Stream assembly: n > 1 choices concatenated; streamed reasoning lost; names appended ("bashbash"); index-less calls merged; mid-stream errors ignored | per-choice assembly, reasoning kept, names set once, slots by id, `error` kept |
| C8–C9 | P1 | Server tools (web search) invisible; `count_tokens`, `/v1/models`, `/v1/responses` were 404; the query string was dropped | server tools are calls; routes added (Responses recorded); query forwarded |
| C10–C12 | P1 | Retry collapse dropped best-of-n repeats and kept marked retries that were not adjacent; two live keys per session; no body cap; a torn last line raised | SDK retry headers decide; one key per session plus `unregister`; 64 MB cap; torn line skipped |
| C14 | P1 | `<system-reminder>` and command output in the user role were recorded as the human; images vanished | `harness_message` events; `[image]` markers |
| C15–C17 | P0 | Probe false negatives (`--no-pager`, `-c`, argv arrays; "not found" in a commit subject) and false positives (`/root`, every web search, escaped newlines) relabelled runs | git global options; argv joined; decoded text scanned; error lines exclude commit lines; `/root` dropped; web counts only when it returns a change link or a diff |
| C18 | P0 | `writes_of`: `view` a write; bash writes, MCP / Gemini names and freeform `apply_patch` missed (frontier runs edit through bash only) | shell writes (`sed -i`, redirects, heredocs, `tee`, `cp`/`mv`, `touch`); read-only editor commands skipped |
| C19 | P1 | `REJECTED` / `INTERRUPT` matched anywhere ("user rejected the invite" in a commit) | the harness's sentences at the start of the text |
| C20–C23 | P0 | ATIF: sidechain prompts became human messages and subagent writes the main agent's; copied-context agent steps re-emitted; error flags dropped; continuation segments shared one id | `extra.is_sidechain` / `agent_id` read; prompts are `subagent_prompt`; copied steps skipped; `tool_result_is_error` read; ids carry the trajectory id; temp dirs removed |
| C24–C25 | P1 | `claude_code_sessions` without freeze merged all sessions; a frozen copy never refreshed | a per-session view; refresh when the live file grew; frozen dir 0700 |
| C26 | P1 | `chat_messages` linked unlinked results newest-first, swapping parallel results | oldest-first |
| C30 | P1 | (found by the full suite, after C12's one-key rule) the A2A harness named its capture session after the attempt id alone and never released the key: two runs of one task appended to one capture log, so the second run's record could hold the first run's calls | session name carries a hash of the attempt directory; the key is released in the harness's `finally` |
| C27–C29 | P1 | OTel: results out of time order; repeated events overwritten; duplicate exports counted twice; nanoseconds lost; OpenInference ignored; no `writes`; absolute paths raised | time-ordered with depth; event lists; deduplicated by span id; integer nanoseconds; OpenInference TOOL / LLM / AGENT; `writes_of`; `glob` |

### 2.4 CI logs and git history

| ID | Sev | Defect | Fix |
|---|---|---|---|
| D1, D3 | P0 | Failing runs were `good` when the parser missed failures (pytest ids with spaces, GTest `GetParam()` lines); a collection error was `unknown` | node ids with brackets and xdist lines parsed; parsed counts raised to the runner's own summary; collection errors are errors |
| D2 | P0 | A test failing on "Connection refused" made the run infra (masked) | infra markers read outside failure sections only |
| D4 | P0 | A run killed mid-way (exit 137, no summary) was `good` | killed-run markers; a pytest / gtest / ctest log with no runner summary is `unknown` (cut off) |
| D22, D28, D29 | P1 | Timestamped or coloured logs lost every test; BOM / UTF-16 JUnit `unknown`; prose "tests passed," read as CTest; CTest exceptions missed; reruns invisible | prefixes and ANSI stripped; decoded; CTest summary required; exception lines; JUnit `flakyFailure` and pytest `RERUN` mark `flaky` |
| D5 | P0 | `#12` matched `#123`, so another PR's revert was this PR's; a short title matched any revert | `#N` exactly; the title only in `Revert "..."` naming no other PR |
| D6 | P0 | A `git mv` "fix:" commit touched every PR line; later edits of the moved file were lost | exact renames followed (ranges move to the new path; the log has no path filter) |
| D7 | P0 | Repositories on `main` crashed | default branch from the clone |
| D8 | P1 | The fetch time was in the hash: re-ingest was idempotent only with a pinned clock | no `fetched_at` in the trace |
| D23 | P1 | API lists cut at page 1; in-progress check runs cached forever; non-atomic cache; no 429 wait | `get_all`; incomplete runs not cached; atomic writes; `retry-after` waits |
| D25 | P1 | Paths with spaces got a trailing tab; accented paths were dropped | `core.quotePath=false`; tab stripped |
| D30 | P2 | Any code block after "use" counted as a suggested fix; a re-landed PR yielded two refs | "use" dropped; one ref per PR |

## 3. Coverage and contracts

- Import contracts: 4 kept. `ruff check`: clean.
- Full unit suite: 883 passed, 1 failed (C30, fixed), 3 skipped; the failing test and the harness, capture and connector
  suites re-run after the fix: pass.
- Statement coverage of `crucible.connectors` + `core.agent_events`: **91%** (1,418 statements). Lowest: `base.py` 72%
  (Postgres paths, error branches), `chat_messages.py` 78%, `sql_audit_log.py` 82% (the Postgres branch has no live
  server here), `capture_proxy.py` 90%.

## 4. Comparison with the local frameworks

### 4.1 Ranked features (not built in this audit)

| Rank | Feature | Reference | Owner |
|---|---|---|---|
| 1 | `issues` (GitHub / Jira / Linear) and `gitlab` connectors (ARCH lists them) | SWE-smith `mirror/collect/utils.py:77-200`; repo2rlenv `gitlab.py:55-113` | coding pack |
| 2 | A call lineage graph for captures (branches for retries, subagents and compaction, beyond the prefix and thread rule here) | NeMo-Gym `token_id_capture/lineage.py:227-298`; OpenEnv `capture/graph.py` | `capture_proxy.py` |
| 3 | Known-value scrub in every encoding (shell-quoted, JSON-escaped, URL-encoded, base64 of each `.env` value) | agentenv `model_key.py:196-208` | `redact.py` |
| 4 | Redact the raw payload before normalization (today: cut keys are caught, other formats cut mid-token may not be) | OpenEnv `modal_provider.py:607-618` | `base.ingest` |
| 5 | A source digest on `TraceSource` (file manifest, row count, SQL watermark) | Harbor `packager.py:83-95` | `core/schema/trace.py` |
| 6 | Per-record quarantine with streaming reads of very large exports | NeMo-Gym `train_data_utils.py:559-586` | `file_drop.py` |
| 7 | Token ids and logprobs for owned engines through the proxy | NeMo-Gym `token_id_capture/records.py` | `capture_proxy.py` |
| 8 | A standalone `vcs_history` connector (reverts and hotfixes over a whole branch); renames with edits (needs blobs) | — | coding pack |
| 9 | More test formats (Catch2, Boost.Test, QTest, lit, unittest) | SWE-smith `profiles/cpp.py` | `ci_logs.py` |
| 10 | Name / NER redaction for free text beyond known people | — (no reference) | `redact.py` |

### 4.2 Where Crucible is stronger

- **One choke point.** `ingest` always redacts before `put`; the references scrub at each log site.
- **Format patterns, proven.** The gate checks the real `.env` values against the raw transcript and the store; the
  references only mask values they injected. No reference has a planted-canary test; this one now plants 31 formats,
  structured fields and a key longer than any text cap, and checks fragments.
- **Linkable pseudonyms** under a tenant key (HMAC). The references delete; none pseudonymizes or detects PII.
- **One event vocabulary for every agent source** (ATIF, capture, OTel, chat) with answer-key probes judged as ran /
  exposed. No reference tags git-history, registry, code-host, grader or privilege probes in captured traces.
- **Retries keep the attempt the agent saw** (verifiers keeps the first).
- **PR history from git alone**: rebased-away commits recovered, comments attributed by blame, suggestions checked
  against the final file, post-merge lines carried through later commits (now through exact renames too).
- **P16 masking with cross-run `flaky()`**, and no invented test names for summary-only logs.
- **`otel_import` exists**: the references only export ATIF to OTel.

## 5. Gate re-run

`python -m crucible_pack_coding.m3.gate` (offline, 0 GitHub API calls): **pass, 9/9 checks**.

| Check | Result |
|---|---|
| Sources | 106 traces: 55 PRs, the construction transcript (9,684 events), 11/11 frontier runs (3 good / 5 bad / 3 unknown), 4 CI reports, 11 policy docs (173 rules), 24 ticket logs. All counts as before |
| Canaries | 31 planted formats (was 20), now including structured fields and a long key cut by the text caps: 0 leaked, checked by 12-character fragments |
| Real secrets | none of the 4 workspace `.env` credentials in the store, all caught by format patterns alone |
| Idempotence | same keys on re-ingest; and on two new checks: a PR connector with its own clock, and every ticket log written in reverse row order |
| Cost | 188 s (was 62 s): post-merge history is read without a path filter so renames can be followed |

## 6. Resolution (2026-10-12)

Fixed: A1–A18, B1–B14, G1, C1–C29 except C22's ordering of `subagent_trajectories`, D1–D25, D28–D30.

Changed defaults:
- `RedactionPolicy.salt` defaults to the tenant key, so pseudonyms change once. Nothing compared stored pseudonyms.
- CI trace ids carry a content digest.
- `sql_audit_log` ids carry the connector name when a family is set; `file_drop` ids are unchanged.
- `ingest` raises `IngestError` after storing the good sources.

Open:
- C22: the ordering of nested `subagent_trajectories`. Claude Code's own sidechains, the common case, are handled.
- D26: commits are attributed to the PR author.
- D27: `policy_docs` false positives (low).
- Features §4.1.
- D11 has not been run against a live Postgres.
- Renames that also edit the file count as deletes, because similarity detection would fetch blobs into the blobless
  clone.

Suites: 884 passed after the C30 fix (883 in the full run + the fixed test), 3 skipped; the 80 audit tests included.
