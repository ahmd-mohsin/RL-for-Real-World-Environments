# M3 study: agent-session capture and import

Scope: `connectors/capture_proxy.py`, `connectors/otel_import.py`, `connectors/chat_messages.py`, `interop/harbor/atif_import.py`, `core/agent_events.py`, coding pack `connectors/{claude_code_sessions,agent_runs}.py`. Spec: ARCHITECTURE §M3, `docs/modules/M03-connectors.md`. Read-only study, 2026-10-11.

Paths: `REF` = `RL-Environments/06-frameworks-and-env-code/`. `C` = `crucible/src/crucible/`. `CP` = `crucible/packs/coding/src/crucible_pack_coding/`.
Installed Harbor: `crucible/.venv` has `harbor-0.23.0` as an editable install of `REF/harbor-framework-harbor/src` (via `harbor.pth`). So Harbor file:line below is exactly the code Crucible runs. ATIF default schema is `ATIF-v1.8`. `harbor_atif2otel` is not importable in the venv (`opentelemetry.proto` missing).

Every candidate marked **[ran]** was reproduced with the Crucible venv on HEAD `dc29c0f`. Probe scripts are in the session scratchpad and are short enough to retype from the inputs given.

## 1. Mechanism table

| Mechanism | verifiers (`REF/primeintellect-ai-verifiers/verifiers/v1/`) | NeMo-Gym (`REF/nvidia-nemo-gym/nemo_gym/`) | OpenEnv (`REF/meta-pytorch-openenv/`) | Harbor (`REF/harbor-framework-harbor/`) | Crucible |
|---|---|---|---|---|---|
| Session identity | per-rollout secret, `register`/`unregister` `interception/server.py:368-388` | rollout id on lineage index `token_id_capture/lineage.py:378-414` | session registry `src/openenv/core/harness/capture/sessions.py`; admin key `compare_digest` `capture/server.py:715` | n/a | key -> session dict, no unregister `C/connectors/capture_proxy.py:59-66,91-94` |
| Dialects | chat, anthropic, responses `dialects/chat.py:425`, `anthropic.py:435`, `responses.py:372` | anthropic <-> responses converters `anthropic_converter.py:195-290`, `responses_converter.py` | detect by path/body `capture/detection.py:33-40`; catch-all incl. `/v1/responses` `capture/server.py:924` | n/a | chat + messages only `capture_proxy.py:146-147` |
| Aux routes | `count_tokens` relayed, not recorded `dialects/anthropic.py:436`, `base.py:277` | n/a | `/v1/messages/count_tokens` `capture/server.py:109`; `/v1/models` `:909` | n/a | none (404) |
| SSE parse | event-level, joins multi-line `data:` `dialects/base.py:228-243` | `anthropic_response_to_sse` `anthropic_converter.py:551-610` | `capture/sse.py:120-160` | n/a | line-level `capture_proxy.py:118-121` |
| Stream assembly | choice 0 only, reasoning, names set not appended, custom tools `dialects/chat.py:292-358`; signature_delta `anthropic.py:345-400` | thinking `signature` kept `anthropic_converter.py:237-260,724-735` | `capture/sse.py:175-224` | n/a | `assemble_openai` `capture_proxy.py:154-176`; `assemble_anthropic` `:179-209` |
| Status / errors | fast result keeps HTTP status; keepalive after grace `server.py:80-83,238-300,852` | n/a | `keepalive_response` keeps status `capture/sse.py:72-110` | n/a | streaming always 200 `capture_proxy.py:126` |
| Disconnect | turn keeps running, logged `server.py:270-300` | n/a | tested per RFC 012 `rfcs/012-harbor-capture-providers.md:66` | n/a | record skipped (no `finally`) `capture_proxy.py:113-125` |
| Retry guard | `x-stainless-retry-count` or `Idempotency-Key` gates replay `server.py:86-100,615-656` | n/a | retries become siblings in graph `capture/graph.py:17-19` | uuid dedup `agents/installed/claude_code.py:1155-1167` | adjacent same digest -> keep last `capture_proxy.py:245-248` |
| Context linking | content-hash graph, new branch at divergence (`rfcs/006...md:125`) | assistant fingerprint over any earlier call, fail closed `token_id_capture/fingerprint.py:50-74`, `lineage.py:235-298` | prefix graph; tolerant `_message_identity`; leading system stripped `capture/graph.py:85-140`; orphan adoption `:264` | n/a | exact prefix vs previous request only `capture_proxy.py:279-285` |
| Body cap | 1 GiB `server.py:72-77` | n/a | n/a | n/a | none `capture_proxy.py:95` |
| Ingest validation | n/a | conformance `token_id_capture/conformance.py` | graded checks, one-call rollout FATAL `rfcs/006...md:75`, `capture/validate.py` | ATIF validators `src/harbor/models/trajectories/trajectory.py:121-185` | none |
| ATIF ids | n/a | n/a | n/a | `session_id` run-scoped, MAY be shared `trajectory.py:29-46`; `trajectory_id` unique `:47-61` | trace id from `session_id` `C/interop/harbor/atif_import.py:136` |
| Subagents | n/a | n/a | subagent = own root `capture/graph.py:20-22` | `subagent_trajectory_ref` on results `observation_result.py:30`; nested under tool span `packages/harbor-atif2otel/src/harbor_atif2otel/convert.py:146,570-583`; Claude Code sidechains merged with `extra.is_sidechain`/`agent_id` `claude_code.py:1117-1121,1171-1176` | appended at the end, labelled by `session_id` `atif_import.py:92-93`; `step.extra` ignored |
| Tool error flag | n/a | n/a | n/a | `extra.tool_result_is_error` `claude_code.py:1432-1434` | dropped `atif_import.py:86-91` |
| OTel convention | n/a | n/a | n/a | OpenInference (`openinference.span.kind`, `tool.name`, `input.value`) `convert.py:357-371,539-544`; hex ids `:465-470` | GenAI semconv only `C/connectors/otel_import.py:29-30,108` |
| Session routing | key in auth header | n/a | n/a | n/a | key in auth header |
| Other | rllm gateway puts session in URL `REF/infra-rllm-org-rllm/rllm/gateway/manager.py:287-299`; agentenv `TRAJECTORY_V1` `agentenv-framework/packages/agentenv-protocol/src/agentenv_protocol/a2a_agent/extensions.py:949` | | | mini-swe tool name defaults to `bash` `agents/installed/mini_swe_agent.py:195-203,245` | |

## 2. Defect candidates

Severity: H = corrupts traces or labels in normal use; M = corrupts in a common edge case; L = rare or cosmetic.

### Capture proxy (`C/connectors/capture_proxy.py`)

**D1 [H][ran] False `context_reset` and full history re-emission.** `:279` compares raw dicts against the previous request only. `:285` then re-emits every message after a reset. Three normal inputs trigger it:
- Rolling `cache_control`. Request 1 `messages=[{"role":"user","content":[{"type":"text","text":"fix the bug","cache_control":{"type":"ephemeral"}}]}]`. Request 2 sends the same user block without `cache_control`, then the assistant `tool_use` and a `tool_result`. Got: `context_reset`, then `user_message "fix the bug"` a second time.
- A side call between main turns (Claude Code title/haiku calls, a concurrent subagent on the same key). Main, side (`system:"Generate a title"`), main. Got: 2 resets, 3 `user_message` events for one human message, 3 `system_message` events.
- A harness that re-renders its system prompt each turn (OpenAI path keeps system in `messages`). OpenEnv measured 13 turns -> 101 assistant messages for this `capture/graph.py:130-140`.
Better: NeMo-Gym resolves the parent against any earlier call by assistant fingerprint `token_id_capture/lineage.py:235-298`; OpenEnv compares `_message_identity` and strips leading system `capture/graph.py:85-140`. Emit only messages past the longest matching prefix.

**D2 [H][ran] Upstream 429/5xx on a streamed call reaches the agent as HTTP 200.** `:126` builds `StreamingResponse` with the default status. Input: `POST /v1/chat/completions {"stream":true,...}`, upstream answers `429 {"error":{...}}` with `retry-after: 7`. Got: client status 200, `text/event-stream`, body is the raw JSON line, no `retry-after`. The SDK does not retry; the agent sees an empty or broken stream. Non-stream path keeps the status but drops `retry-after` (`:130`). Better: verifiers keeps the status for fast results `server.py:80-83,852`; OpenEnv `capture/sse.py:72-110`.

**D3 [H][ran] The exchange is lost when a stream fails or the client leaves.** `_record` runs after the `async with` at `:123-125`, not in a `finally`. Input: upstream sends one chunk then raises `httpx.ReadError`. Got: 0 records added. The same holds for a client disconnect (Starlette cancels the generator) and for `httpx` connect/timeout errors on the non-stream path (`:127`, unhandled -> plain 500, no record). A user pressing Esc in Claude Code aborts the stream, so the interrupted turn, the strongest correction signal, is never captured. Better: verifiers runs the turn to completion and logs the disconnect `server.py:238-300`.

**D4 [M][ran] Lone surrogate in a request body: upstream is billed, agent gets 500, nothing recorded.** `json.loads` accepts `"\udcff"`; `_record` writes with `ensure_ascii=False` (`:70`) and fails on encode, after `client.post` already ran (`:127-129`). Input: `{"model":"m","messages":[{"role":"user","content":"bin \udcff"}]}` (JSON escape, as Python agents emit for `surrogateescape` tool output). Fix: `ensure_ascii=True` or `errors="surrogatepass"`, and `encoding="utf-8"` on open/read (`:69`, `:241`).

**D5 [M][ran] `assemble_openai` merges all choices into one.** `:162` ignores `choice.index`. Input: chunks with `index:0 "Hello"`, `index:1 "Bye"`, `index:0 " world"`, `index:1 " now"`. Got content `"HelloBye world now"`. verifiers skips `index != 0` `dialects/chat.py:299-300` and rejects `n != 1` `:513`.

**D6 [M][ran] Streaming loses reasoning and mangles tool names.** Only `delta.content` is kept (`:164-165`), so `reasoning_content`/`reasoning` deltas vanish; the non-stream path keeps them (`:326`). Names are appended (`:170`): a provider that repeats `"name":"bash"` per chunk gives `"bashbash"`. Calls without `index` default to slot 0 (`:167`): two parallel calls become `name "lscat"`, `arguments '{}{"p":1}'`. Better: `dialects/chat.py:301-303` (reasoning), `:352-354` (name set, not appended).

**D7 [M][ran] OpenAI mid-stream error chunk ignored.** `data: {"error":{"message":"overloaded"}}` after content. `assemble_openai` has no `error` branch; status is 200, so `normalize` (`:312`) does not flag the call. The Anthropic path does catch `type:"error"` (`:201`).

**D8 [M][ran] Anthropic server tools are invisible.** `normalize` only turns `type=="tool_use"` into calls (`:319-321`). Response `[{"type":"server_tool_use","name":"web_search","input":{"query":"django patch 4521"}}, {"type":"web_search_tool_result",...}]` yields no `tool_call` and no `web` probe. Same for `code_execution`/`web_fetch` server tools.

**D9 [M][ran] Missing routes break agents.** `/v1/messages/count_tokens` -> 404, `GET /v1/models` -> 404, `/v1/responses` (Codex) -> 404. The query string (`?beta=true`, sent by Claude Code) is dropped from the upstream URL (`:111`). Better: verifiers `dialects/anthropic.py:436`; OpenEnv `capture/server.py:109,909,924`.

**D10 [M][ran] Retry collapse is both too eager and too weak.** `fetch` (`:245-248`) drops the earlier of two adjacent records with the same body digest. Two identical `temperature:1` requests (best-of-n, self-consistency) kept 1 of 2. A real retry that is not adjacent (another call of the same session in between) is kept twice. `x-stainless-retry-count` and `Idempotency-Key` are not recorded. verifiers states the rule: "an unmarked repeated body can be a legitimate later turn" `server.py:617-620`.

**D11 [M] Within-session ordering is completion order.** `time` is stamped after the response (`:124,129`). Concurrent calls on one key (Claude Code subagents) append in finish order. User and tool messages get the response's end time, not the request time. Record `request_time` too.

**D12 [L-M][ran] Session lifecycle.** No `unregister`; keys never expire. `register("s1")` twice gives two live keys writing to the same `s1.jsonl`, so two runs interleave in one log. No body cap (`:95`; verifiers caps at 1 GiB `server.py:72-77`). Log size is quadratic: every record stores the full request (`:109`). `fetch` fails on a torn last line (`:244`) while the proxy is writing. Timing on key compare: dict lookup, not a practical oracle; `secrets.compare_digest` is cheap if keys move to a list (OpenEnv `capture/server.py:715`).

**D13 [L] Line-level SSE parsing.** A `data:` payload split over two `data:` lines is dropped (`:119-121`). verifiers joins them `dialects/base.py:228-243`. Not a defect: gzip (httpx decodes; the proxy sends decoded bytes without a `content-encoding` header), `: ping` comments and `[DONE]` (passed through, not parsed).

**D14 [M][ran] Harness text recorded as the human.** `user()` (`:262-264`) labels every user-role text as `actor="user"`. Input: `[{"type":"text","text":"<system-reminder>todo list empty</system-reminder>"},{"type":"text","text":"hi"}]` -> one `user_message` with the reminder. Compaction prompts, hook output and command caveats also become human turns. M4 then reads them as user feedback. Images are dropped without a marker (`:212-215`).

### Shared vocabulary (`C/core/agent_events.py`)

**D15 [H][ran] Answer-key probe false negatives.** `_GIT` (`:39`) allows only `-C`, `--git-dir`, `--work-tree` before the subcommand. `git --no-pager log --all --oneline` -> no probe. `git -c core.pager=cat log --all` -> none. Argv form `{"command":["git","log","--all"]}` (Codex shell) -> none; `["pip","download","django==5.0.1"]` -> none.

**D16 [H][ran] A probe that worked is scored as failed.** `ran` requires `not FAILED.search(out[:2000])` (`:144`), and `FAILED` (`:66-68`) matches `not found`, `denied`, `timed out`, `fatal:` anywhere. Input: call `git log --all --oneline`, result `"9f1e2d3 Fix file not found error in loader (#4521)\n1a2b3c4 base"`. Got `probe_ran=False`, nothing exposed. Commit subjects like these are common, so the most dangerous probe is the easiest to miss. Use exit status or `error` flags first; match `FAILED` only on the first lines or stderr markers.

**D17 [M-H][ran] Probe false positives that relabel good runs.** `privilege` matches any `/root` path (`:60`), and `exposed` returns True for `privilege` and `web` whenever they ran (`:111`). `cd /root/project && python -m pytest` -> strong exposure. `WebSearch {"query":"python asyncio gather docs"}` -> strong exposure. `agent_runs` then turns a reward 1.0 into `unknown`/contaminated (`CP/connectors/agent_runs.py:84-89`). Arguments are JSON text, so an escaped newline does not stop `[^\n|;&]*`: `"git log --oneline -3\ncat /tmp/upstream/notes.txt"` -> `git_history`. `git log --all` in a repo that keeps its base history lists many commits and counts as exposed (`:105-108`).

**D18 [M-H][ran] `writes_of` misses and false writes.** `str_replace_based_edit_tool {"command":"view","path":"tests/test_a.py"}` -> `["file:tests/test_a.py"]`, and `probes_of` adds a `test_edit` probe for a read. Misses: `mcp__filesystem__write_file {"path":...}` (only `.` is split, `:20`); Gemini CLI `write_file {"file_path":...}` (key is `path`, `:12`); freeform `apply_patch` whose input is a raw string (`:21` requires a dict); `*** Move to:` targets; every bash write (`sed -i`, `tee`, `>`, `cat <<EOF >`, `git apply`). mini-swe-agent edits only through bash (`REF/harbor-framework-harbor/src/harbor/agents/installed/mini_swe_agent.py:245`), so the 11 frontier-run traces have no `data.writes` except `final_patch`. Paths are not normalized: `/testbed/a.py` from an editor never equals `a.py` from `final_patch`.

**D19 [M][ran] `REJECTED`/`INTERRUPT` match anywhere in the text.** A `git log` line `"abc123 Show toast when user rejected the invite"` -> `rejected: True` (`:15`). A grep over `agent_events.py` itself, common in this repo's own Claude Code sessions, does the same. Anchor to the harness's exact sentences at the start of the result.

### ATIF import (`C/interop/harbor/atif_import.py`)

**D20 [H][ran] Claude Code subagent work is booked to the main agent and the human.** Harbor merges sidechains into the main step list and marks them only in `step.extra.is_sidechain` / `agent_id` (`claude_code.py:1171-1176`). `walk` ignores `extra` (`:64`). Input: step 3 `source:"user"`, `extra:{"is_sidechain":true,"agent_id":"sub1"}`, message "Explore the repo and report". Got a human `user_message`, and step 4's `Write` as a main-agent `tool_call`. The docstring's `data.subagent` is never set for Claude Code. This hits the 9,684-event construction transcript.

**D21 [M][ran] Copied-context agent steps are re-emitted.** Only user steps check `is_copied_context` (`:67`). An agent step with `is_copied_context:true` and `Bash "cd /root/x && git log --all"` produced a tool call and a strong exposure in `outcome.details.probes`. Harbor says such steps "should not be included in training data" (`src/harbor/models/trajectories/step.py:64-72`).

**D22 [M] ids and order.** `trace_id` uses `session_id` (`:136`), which Harbor defines as run-scoped and shareable by continuation segments (`trajectory.py:29-46`): two segments collide. Subagents are labelled by `s.session_id or s.trajectory_id` (`:93`), so siblings that inherit the parent's session id share one label; use `trajectory_id`. Subagent events are appended after all parent events (`:92-93`), so the `Task` result precedes the work it summarizes. Harbor resolves by `subagent_trajectory_ref` (`observation_result.py:30`); atif2otel nests the subagent under its tool span (`convert.py:570-583`). `continued_trajectory_ref` and observations on system steps are ignored.

**D23 [M][ran] Tool error flags dropped.** Harbor stores `extra.tool_result_is_error` (`claude_code.py:1432-1434`). The result event has no `error` (`:91`). `annotate_probes` then needs the `FAILED` regex (D16), and M4 loses failed-call evidence.

### Claude Code sessions (`CP/connectors/claude_code_sessions.py`)

**D24 [M][ran] `freeze: False` merges every session in the project.** `discover` falls back to the project dir (`:47`). Harbor globs `*.jsonl` there (`claude_code.py:1131-1133`). Two sessions `aaa`, `bbb` -> both traces contain `['task from aaa','task from bbb','ok aaa','ok bbb']`. Pass a per-session temp dir instead.

**D25 [M] Frozen once, stale forever.** `:45` freezes only when the copy is missing. A session that grows after the first ingest is never refreshed, though the docstring says the live file yields new versions. Compare size/mtime or a digest. Each convert also leaks a `mkdtemp` (`atif_import.py:116`).

### chat_messages (`C/connectors/chat_messages.py`)

**D26 [M][ran] Unlinked results go to the newest open call.** `:81-85` scans `reversed(order)`. Two parallel calls `read A`, `read B` without ids, results "contents of A", "contents of B" -> A's result `reply_to` B, B's to A. Use oldest-first, as `atif_import.py:90` does. Image parts add stray blank lines (`:30-32`).

### OTel import (`C/connectors/otel_import.py`)

**D27 [M][ran] Repeated span events overwrite each other.** `ev_attrs` is a dict by event name (`:116`). Two `gen_ai.user.message` events -> `input` keeps only "second". The current semconv event `gen_ai.client.inference.operation.details` carrying `gen_ai.output.messages` is not read (`:128`); text falls back to the span name.

**D28 [M][ran] Event order breaks time order.** The result is emitted right after its call with the end time (`:123-126`). A tool span [1µs, 9ms] with a nested chat span at 2µs gives `tool_call, tool_result(9ms), llm_call(2µs)`. Ties sort by input order (`:102`); exporters write children first, so with equal start times `execute_tool` precedes its `invoke_agent` parent and gets the service name as actor, not the agent. Sort by (start, depth) and emit results by end time.

**D29 [M] Coverage gaps.** OpenInference spans (Harbor's own atif2otel output, Phoenix, LangChain) have no `gen_ai.operation.name`, so every span becomes a generic `span` with no tool calls or probes (`:108,135-137`). No `writes_of` on OTel tool calls [ran: `write_file {"path":"a.py"}` -> no `writes`]. No `REJECTED`, no user messages, no `bytesValue` (`:33-43`). Duplicate span exports are not de-duplicated by `spanId` [ran: 2 events]. Links and most resource attributes are dropped. Nanosecond times go through a float (`:53`): microsecond precision only.

## 3. Features Crucible lacks (ranked)

| # | Feature | Reference | Owning Crucible file |
|---|---|---|---|
| 1 | Call lineage graph: parent = any earlier call, branches for retries, subagents, compaction | `REF/nvidia-nemo-gym/nemo_gym/token_id_capture/lineage.py:227-298`; `REF/meta-pytorch-openenv/src/openenv/core/harness/capture/graph.py:14-24,85-140,264` | `capture_proxy.py` `CaptureLogConnector` |
| 2 | Stream commit with status passthrough, keepalive, disconnect-safe recording | verifiers `interception/server.py:80-83,238-300`; OpenEnv `capture/sse.py:72-110` | `capture_proxy.py` `forward` |
| 3 | OpenAI Responses API dialect (Codex, trae-agent) | verifiers `dialects/responses.py:372`; OpenEnv `capture/detection.py:33-40` | `capture_proxy.py` |
| 4 | Ingest validation of captures (empty streams, one-call sessions, 4xx-truncated runs) | OpenEnv `rfcs/006-agentic-rl-harness-interception.md:75`, `capture/validate.py` | `CaptureLogConnector.fetch` |
| 5 | Retry semantics from `x-stainless-retry-count` / `Idempotency-Key` | verifiers `interception/server.py:86-100,615-656` | `capture_proxy.py:109,245` |
| 6 | Aux route relay (`count_tokens`, `/v1/models`) and header passthrough | verifiers `dialects/anthropic.py:436`; OpenEnv `capture/server.py:109,909` | `capture_proxy.py:146` |
| 7 | Subagent placement via `subagent_trajectory_ref`; sidechain identity from `step.extra` | Harbor `observation_result.py:30`, `claude_code.py:1117-1121`; atif2otel `convert.py:146,570-583` | `atif_import.py` `walk` |
| 8 | OpenInference attribute mapping in OTel import | atif2otel `convert.py:357-371,539-544` | `otel_import.py` |
| 9 | Token ids and logprobs for owned engines | NeMo-Gym `token_id_capture/records.py:108-235`; OpenEnv RFC 006 `:11` | `capture_proxy.py` (documented gap) |
| 10 | Session release and per-session budgets | verifiers `interception/server.py:379-388`; OpenEnv `rfcs/012-harbor-capture-providers.md:26` | `capture_proxy.py:59-66` |
| 11 | Thinking `signature` / `redacted_thinking` kept in the record | verifiers `dialects/anthropic.py:370-379`; NeMo-Gym `anthropic_converter.py:237-260` | `assemble_anthropic` |
| 12 | Session in URL path (agent keeps its own key scheme) | rllm `gateway/manager.py:287-299` | `capture_proxy.py` (option) |

## 4. Where Crucible is stronger

- **One event vocabulary for every source.** ATIF, capture, OTel and chat transcripts share `writes_of`, `INTERRUPT`, `REJECTED` and probe tagging (`agent_events.py`). No reference maps agent sessions onto workflow-trace events with `reply_to` links.
- **Answer-key probes with ran/exposed judgement.** No reference tags git-history, registry, code-host, grader-file or privilege probes in captured traces. `agent_runs` refuses to trust a green score that followed an exposure (`agent_runs.py:84-89`). The defects above are about precision, not the idea.
- **Retry policy keeps the attempt the agent saw.** verifiers' first-attempt guard would keep a response the agent never saw (M3 lesson 4). Crucible's rule is right; only its trigger needs the retry header (D10).
- **Capability keys with the upstream key held by the proxy**, the same model as verifiers, at a fraction of the code. Per-session JSONL is readable offline without the proxy.
- **DEPEND on Harbor converters** instead of re-parsing Claude Code and mini-swe-agent logs. Harbor's sidechain merge and uuid dedup come for free; Crucible only needs to read `step.extra` (D20, D23).
- **`otel_import` exists at all.** References only export ATIF to OTel (atif2otel); none import enterprise GenAI telemetry.
- **`context_reset` as an explicit event.** References treat a broken prefix as a new root silently. Crucible surfaces it to M4; it only needs fewer false positives (D1).

## 5. Suggested audit order

D1, D2, D3, D20, D15/D16, D17, D18, D24, D26, D5/D6, D27/D28. Each has an input above that fails on HEAD and should become a regression test.
