# M3 `connectors` — capture and ingest

**Status: built (2026-10-09).**

- **Code:**
  - `src/crucible/connectors/`
  - `src/crucible/interop/harbor/atif_import.py`
  - `packs/coding/src/crucible_pack_coding/connectors/`
  - `packs/coding/src/crucible_pack_coding/m3/gate.py`
- **Tests:** `tests/unit/test_connectors.py`, `tests/unit/test_coding_connectors.py`.
- **Gate report:** `reports/m3_gate.json`.

## What it is
Raw records of real work become `crucible.trace/v1` documents, redacted inside the tenant and stored content-addressed. Every connector implements `discover → fetch → normalize`. `ingest()` always runs `redact()` before `TraceStore.put()`, so nothing unredacted is ever stored. A connector's `fetch` works from a ref alone (a ref from an earlier run re-reads its source), so re-ingest needs no in-memory state.

### Generic connectors (core)

| Connector | Source → events | Built on |
|---|---|---|
| `file_drop` | JSONL / CSV event logs (ticketing, ERP, CRM exports) → one trace per case. A `FieldMap` names the columns. `writes` lists the objects an event changed (M4 attributes signals to the last writer). `rows_to_trace` is shared with `sql_audit_log` | — |
| `sql_audit_log` | An audit table (SQLite read-only, or Postgres) → `state` events that write `<table>:<row id>` | — |
| `otel_import` | OTLP/JSON exports of instrumented agents, following the OTel GenAI semantic conventions. `execute_tool` → `tool_call` plus `tool_result` (span errors kept); `chat` / `text_completion` → `llm_call` (model, usage); `invoke_agent` → `agent_invocation`; other spans → `span`. Parent span ids are kept | OTel GenAI semconv (PATTERN) |
| `capture_proxy` | **`CaptureProxy`**: a recording proxy for `POST /v1/chat/completions` (OpenAI) and `POST /v1/messages` (Anthropic), streaming and non-streaming. Each session gets a capability key from `register()`. The real upstream key stays on the proxy, and an agent presenting an unregistered key gets 401. Streams are relayed as they arrive and reassembled for the record: OpenAI deltas, and the Anthropic `message_start` / `content_block_*` / `message_delta` events. Each record keeps a digest of the request body, so a client retry is collapsed to the attempt the agent actually saw. A streamed exchange is `complete` only when it ended with its dialect's terminal marker (`[DONE]` or a `finish_reason`; `message_stop`; `response.completed` / `response.incomplete`) and carried no error event; otherwise it is `complete: false` with `error` `stream_drop: ...` or `stream_error: ...` (M10 lab audit R-1). `register(session, max_calls=n)` caps the calls forwarded for a session: past the cap the proxy answers a non-empty assistant stop (`LIMIT_STOP_TEXT`) in the caller's dialect, forwards nothing and records only a `{"kind": "limit_stop"}` row (one per stop up to `LIMIT_STOP_ROWS`, then one coalesced row with `count` when the key is revoked, so a flood past the cap cannot grow the log without bound), which the connector counts (`payload["limit_stops"]`) and never turns into a turn. `probe(api, model)` sends one minimal generation of Crucible's own straight to the upstream, unlogged: the evidence that confirms an upstream verdict read from a session log the agent can write (M10 lab audit R-1). `inflight[session]` counts a session's forwarded calls not yet answered and recorded, kept past `unregister` (a killed client does not cancel the forward), and `drain(session, timeout_s)` waits for them, so a harness reads and bills them and never probes behind the session's own calls (M10 lab audit R-1 review round 3). **`CaptureLogConnector`** emits only the new messages of each request; a request that does not extend the previous one adds a `context_reset` event | verifiers `interception/server.py` (capability keys, one server for many sessions, retry guard); NeMo-Gym `anthropic_converter.py` (SSE event shapes) and `token_id_capture`; OpenEnv RFC 006; prime-agent `openai_completions/stream.rs` (terminal marker); OpenEnv `capture/sessions.py`, `server.py` (call cap) (PATTERN) |
| `atif` (`interop/harbor/atif_import.py`) | ATIF trajectories → user / agent / system messages, one `tool_call` per call, and a `tool_result` that answers its call (linked by step order when a harness leaves `source_call_id` empty). Also detected: interrupts, rejected calls, copied context, subagents. File-editing calls record `data.writes` | **DEPEND:** Harbor ATIF models, `convert_mini_swe_agent_to_atif`, `ClaudeCode._convert_events_to_trajectory` |
| `agent_events.py` | Shared by all agent-session connectors: which tools write files (`EDIT_TOOLS`, `writes_of`), and which user texts mean "stop" or "no" (`INTERRUPT`, `REJECTED`). M4 therefore sees the same event types whatever the source | — |

### Coding-pack connectors

| Connector | Source → events | Built on |
|---|---|---|
| `github_pr` | PRs with commits (including rebased-away ones), review threads, suggestions, check runs and post-merge history. `discover_git` reads merge and squash commits from git, with no API: the gate's PRs come from git only | git; GitHub REST only for cached payloads |
| `claude_code_sessions` | Our construction transcripts: freezes a copy of `<id>.jsonl` plus `subagents/`, converts it with Harbor, and tags events with the migration tasks they touched (`data.tasks`) so M5 can cut one segment per task | Harbor `ClaudeCode` converter (DEPEND) |
| `agent_runs` | Each frontier run on tasks 01–04: the trajectory (Harbor mini-swe-agent converter), a `final_patch` state event, one `grade_gate` check per verifier gate, and the outcome. The outcome comes from the hidden verifier's reward, or from the calibration record in `seeds.SEEDS`; otherwise it is `unknown` | Harbor (DEPEND) |
| `ci_logs` | One CI report or log = one trace: a `ci_run` event (counts, `failure_kind`, commit) plus one `test_case` check per test or failed build target. Formats are detected from the content: JUnit XML, CTest, GoogleTest, pytest, Ninja / Make / nvcc build failures, and our `grade_report.json`. Infra failures (disk full, OOM kill, lost network, no GPU, runner timeout) get outcome `unknown` (P16: masked, never scored 0). `flaky(traces)` lists tests that both passed and failed on one commit | **PORT:** SWE-smith `profiles/cpp.py` CTest / GoogleTest / pytest patterns (MIT). **PATTERN:** SWE-smith `harness/repair.py` (flaky tests) |
| `policy_docs` | CONTRIBUTING files, style guides (Markdown and reStructuredText), migration logs and configs, read from git at a pinned revision or from disk. Event types: `policy_section` per heading; `policy_rule` per normative sentence or list item (`must_not` > `must` > `should`; checklist items count as `must`); `config_rule` per `.clang-format` key, `.clang-tidy` check or CODEOWNERS line. Code blocks and comments are skipped. Revision history (number of commits, first and last change) is recorded. Rules are candidate rubric items with low reliability until §6.5 validates them | — |

### Redaction (`redact.py`)
Redaction runs on every trace before it is stored:

- **Credential formats**, each pattern tied to a provider: OpenAI, OpenRouter, Anthropic, GitHub, Hugging Face, AWS, Slack, GitLab, Google, Docker Hub, RunPod, npm, PyPI, Stripe, SendGrid, Replicate, Groq, xAI, Tailscale, Databricks, plus JWTs and private keys.
- **Generic rules:** any `Bearer` token, and `password=` / `"apiKey": …` style assignments.
- **The tenant's known secret values**, whatever their format, via `RedactionPolicy.with_env_file(".env")`. `secret_values` is excluded from serialization, so a stored policy cannot carry them.
- **Stable salted pseudonyms** for e-mails, internal hosts and private or Tailscale IPs. The same input gives the same token, so threads stay linkable.
- **Human actors:** both the handle **and the id** of a human actor are pseudonymized, and every reference to the id is rewritten. `file_drop` uses the exported name or e-mail as the id, so handles alone were not enough. Agents and systems keep their ids.

### Answer-key probes (`agent_events.py`, added in the article cross-check, 2026-10-09)
Every agent-session connector (ATIF, capture proxy, OTel), and the World, tags tool calls that probe for the answer: git history, package registries, code hosts and web tools, grader files, privilege escalation, `/proc` snooping and test edits.

- Each tag records whether the probe **ran** (an output, no failure) and whether it **exposed** an answer source (judged from the output). A `git log --all` that lists one `base` commit per repository ran but exposed nothing.
- **Probes are never scored.** An agent-run trace whose pass followed an exposure is relabelled `unknown` (`contaminated`).
- **All 11 frontier runs:** probes ran in 7 of them (history, installs, `/proc`, test edits), and **none exposed an answer source**. A manual audit of every command agreed.

## M3 audit (2026-10-12)

Report: `docs/audit/2026-10-12/M03-connectors.md`; 80 regression tests in `tests/unit/test_m3_audit*.py`. Behaviour that changed:
- **Redaction.**
  - Coverage: every string field and dict key; values under credential-named keys.
  - Formats: URL userinfo, short passwords, Basic auth, cookies, private keys cut by a text cap, Azure keys, Google OAuth and Vault tokens, webhooks, cards, IBANs, SSNs, phones.
  - Names: known people rewritten in text; identity keys (`author`, `assignee`) pseudonymized.
  - Mechanics:
    - linear time;
    - HMAC pseudonyms under a per-tenant key (`<data>/redaction.salt`, or `CRUCIBLE_REDACTION_SALT`);
    - idempotent;
    - code such as `threading.local()` and `tokenizer.encode(x)` is left alone.
- **Store.**
  - Access: the trace store is owner-only.
  - Provenance: each version is stamped with the redaction rules' hash. `stale()` and `purge_superseded()` find and remove old versions; `Store.forget_doc()` deletes.
  - Head: an explicit tag per trace.
  - Re-ingest: M4's signals are carried to a re-ingested version through `data.row_id` or event identity.
- **Ingest.** Failing sources no longer abort the batch (`IngestError` / `errors=`).
- **`file_drop`.** BOM, Latin-1, `.CSV` and U+2028 are read; overlapping exports are deduplicated; events are ordered by instant; times are normalized to RFC 3339; bot actors are agents.
- **`sql_audit_log`.** One SELECT, read-only (SQLite `query_only`; Postgres read-only session with a server-side cursor); NULL cases skipped; paths quoted.
- **Capture proxy.**
  - Upstream errors keep their status and `retry-after`.
  - Interrupted streams are recorded in a `finally`; logs are ASCII and owner-only.
  - `count_tokens`, `/v1/models` and `/v1/responses` are served, and the query string is forwarded.
  - Requests are threaded by prefix: side calls get `data.thread` and are not resets.
  - Harness text becomes a `harness_message`.
  - Retries collapse only when marked.
- **Probes and writes.**
  - Probes see git global options and argv commands; error lines exclude commit lines.
  - `/root` and plain web searches are not exposures.
  - Shell writes count; an editor's `view` does not.
- **ATIF.** Sidechains become `subagent_prompt` events with `data.subagent`; copied-context steps are skipped; tool errors are kept.
- **OTel.** Events are time-ordered; OpenInference spans map like GenAI ones; nanoseconds are kept.
- **`ci_logs`.**
  - Counts are checked against the runner's summary.
  - Collection errors are `bad`. Killed or cut-off runs are `unknown`.
  - Infra markers count only outside test failures.
  - Downloaded logs are cleaned. Flaky reruns are marked. Ids carry a digest.
- **`github_pr`.** Reverts match by exact PR number; exact renames are followed; the default branch is detected; no fetch time in the hash; API lists are paginated.

## Exit gate (`python -m crucible_pack_coding.m3.gate`, offline, 0 GitHub API calls, 188 s after the audit, 45 s before)

| Requirement (ARCHITECTURE M3) | Result |
|---|---|
| ≥ 30 public GPU-migration PR traces | ✅ **55** PCL PRs, read from git history with no API calls. Selection: merges touching `gpu/` or `cuda/` whose title matches `GPU_MIGRATION`. Results: 207 events; 8 PRs have a later fix-like commit touching the PR's lines within 30 days. Outcome labels are assigned by M4's `PullRequestOutcome`, so the connector leaves them `unknown` |
| Our task 01–04 construction transcripts | ✅ The one session that built tasks 01–04: 85 MB frozen, with 25 subagents → **9,684 events**. Task-tagged events: task01 111, task01v2 75, task02 106, task03 171, task04 209 |
| Every recorded frontier run | ✅ **11/11** runs, 2,532 events. Outcomes: 3 good, 5 bad, and 3 unknown (sol-v2, astra-v2 and astra-v3-aborted have no recorded result), matching the calibration records. Also ingested: their 4 grade reports through `ci_logs` (1 good, 3 bad, matching reward), 11 policy documents (PCL CONTRIBUTING, style guide, `.clang-format`, `.clang-tidy`, 5 migration logs, 2 design notes → 173 rules), and the toy pack's 24 ticket logs (172 events) |
| 100% schema-valid | ✅ 106/106 stored traces re-validate from their stored JSON |
| Redaction removes 100% of planted canaries | ✅ 20 formats (API keys of 13 providers, a JWT, a bearer token, a password assignment, an e-mail, two internal hosts, a private IP and a Tailscale IP) planted in raw records of five connectors: a ticket log, a real frontier trajectory (task text and a tool output), a capture log, a CI log and a policy doc. **0 leaked**, using the format patterns alone |
| … real secrets | ✅ All 4 credentials in the workspace `.env` appear in the raw construction transcript. **None** survives into the store, and each is caught by the format patterns alone, without the known-value list |
| Re-ingest is idempotent | ✅ A second ingest of every source returns identical content hashes; 106 docs before and after. Since the M3 audit also with a PR connector on its own clock and with every ticket log in reverse row order. Canaries: 31 formats including structured fields and a key longer than any text cap, checked by fragments, 0 leaked |

## Findings
1. **Two credential formats leaked on the first gate run.** Docker Hub PATs and RunPod keys were in the construction transcript, and the redactor had no pattern for them. The planted-canary test passed because it only planted formats the redactor already knew; the real-secret check caught it. Both formats now have patterns, and tenants can also list their known secrets.
2. **Actor ids leaked identity.** Only handles were pseudonymized. `file_drop` uses the exported actor name (often an e-mail) as the id, and it appeared in every event. Ids of human actors are now pseudonymized too.
3. **Old sources are not UTF-8.** A Latin-1 `©` in a 2011 PCL diff crashed the PR connector. Git output is now decoded with replacement.
4. **The capture log must not drop retries.** Recording "the first attempt only" (the verifiers guard, adapted) would keep a response the agent never saw when the client timed out and retried. Every attempt is recorded, and the connector keeps the last one.
5. **`fetch` depended on `discover` in the same process** for `sql_audit_log` and `otel_import`. A ref from an earlier run now re-reads its source.
6. **How a client's disconnect reaches a stream depends on the server.** The proxy named only `GeneratorExit` "client disconnected". Under uvicorn, which Crucible serves the proxy with, Starlette cancels the stream task instead, so an abandoned stream read `CancelledError: ...`. M10 then blamed the upstream, and a deadline kill of a streaming agent became unknown blame. `CancelledError` now counts as the client leaving too, and a test drives a real uvicorn-served disconnect (M3 audit 2026-10-20, R-1).

## Not done
- **`issues` (GitHub / Jira / Linear) and `gitlab`.** They need API calls, which were deferred on purpose. `github_pr` review threads and check runs exist only in the API cache from M4; the 55 gate PRs are git-only.
- **No live-agent captures yet.** The capture proxy is tested against fake upstreams only, with no paid API calls. Not proxied: the OpenAI Responses API (`/v1/responses`) and token-id capture for opaque harnesses (NeMo-Gym `token_id_capture`); logprobs are kept when the upstream returns them.
- **`otel_import` is tested on a synthetic OTLP export.** No enterprise telemetry was available. `harbor-atif2otel` converts in the opposite direction (ATIF → OTel), so it does not apply to import.
- **The frozen transcript copy is raw.** `data/cache/sessions` holds the unredacted transcript. It is local and gitignored, and only redacted traces enter the store.
- **Git may fetch missing blobs.** The PR cache is a blobless clone, so offline reading can still fetch blobs over the git protocol. This is not the REST API and has no rate limit.
