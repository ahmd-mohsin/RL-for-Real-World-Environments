# M03 connectors: audit against the lab RL repos (2026-10-20)

Crucible HEAD `d282251` (`uzair` = `main` = `dev`). Crucible source is cited relative to `crucible/src/crucible/`
(`connectors/redact.py:155`), other Crucible files relative to `crucible/`, and `coding/` stands for
`packs/coding/src/crucible_pack_coding/`. Reference code is cited as `<folder>/<path>:line`; inside a repo's own section 5
subsection the folder is dropped. Commits are in section 4. **[ran: probe]** marks a claim reproduced by a probe. Notes,
probes and outputs: `/Users/uzair/Desktop/RL env repos/.module-audit/2026-10-20/M03-connectors/` (`probes/`, `notes/*.out`).

## 0. Summary

- **Where the output matters.** Stored traces feed M4 signals, M5 steps, M8 forge and M15 labels. The capture proxy's raw
  logs are read directly by M10 for blame and for the harness call lists. Stored capture, ATIF, OTel and chat traces have
  no consumer yet, so most findings about them are later.
- **Established gaps (4 should, 0 must).**
  - R-1: under Crucible's own uvicorn server, a client that abandons a stream is logged as an upstream failure, so M10
    masks an evidenced timeout as unknown blame **[ran: v1_client_left S3]**.
  - R-2: a non-streaming exchange whose client already left is logged as an ordinary 200. It becomes the call list's
    last action, with a write that never happened **[ran: v1_client_left S1/S2]**.
  - R-3: `agent_runs` would label a run that a provider or harness ended `bad` once it is graded, and M4 would emit -1
    **[ran: v4_mv1_exception_exit]**.
  - R-4: one lone surrogate aborts a whole ingest batch (0 of 3 stored) **[ran: v4_c1_surrogate_ingest]**.
- **Counts.** must 0, should 4 (R-1..R-4), later 23 (L-1..L-23). Rejected candidates: 0. Seven proposed mechanisms were
  dropped or narrowed (section 7).
- **Top recommendations.** R-1 (impact H), then R-2, R-3 and R-4. Each is effort L and risk L.
- **Coverage.** 33 reference folders: 11 primary, 15 secondary, 7 not relevant. No sweep cluster failed. No candidate was
  left unverified (4 verify batches).
- **Against the 2026-10-12 module audit (`90f2539`).**
  - Introduced since: 1 of the 4 should items. R-1's label began to decide blame with `d99fde1`.
  - Present then and missed: R-2, R-3 and R-4 (at `90f2539` every graded reward < 1 was `bad`; `b65754b` exempted only
    the limits). Of the later items, L-5..L-8, L-11, L-14 and L-23 were already in the code then.
- **Ledger.** 71 items: 5 done, 17 partly, 48 not done, 1 superseded. Two ledger corrections: `context_reset` has an M5
  reader, and rLLM's cumulative token mode exists.

## 1. What the module does

M3 turns raw records of real work into redacted `crucible.trace/v1` documents in a tenant-local, content-addressed
store. Each source has a connector (`discover -> fetch -> normalize`). `ingest` stores `redact(normalize(fetch(ref)))`
(`connectors/base.py:115-120`), so the store never holds an unredacted trace. Generic connectors live in core:
`file_drop`, `sql_audit_log`, `otel_import`, `chat_messages`, `capture_proxy` with `CaptureLogConnector`, and `atif`
through Harbor. Domain connectors live in the coding pack: `github_pr`, `ci_logs`, `claude_code_sessions`, `agent_runs`
and `policy_docs`. `core/agent_events.py` marks writes (P28) and answer-key probes (P27) in every source. Since `d99fde1`,
M10 also reads the proxy's raw logs as runtime evidence.

| Item | Contract |
|---|---|
| Inputs | CSV/JSONL exports, SQL audit tables, OTLP/JSON, ATIF, chat JSONL, CI logs and grade reports, policy docs, Claude Code session dirs, mini-swe-agent run dirs, git + cached GitHub REST, live chat/Anthropic/Responses traffic via the proxy |
| Outputs | `Trace` versions under head tag `trace-head:<id>` (`connectors/trace_store.py:69-89`) with `params.redaction` provenance (:75); raw capture logs, 0600 in a 0700 dir (`connectors/capture_proxy.py:133,213-218`) |
| Called by | `crucible label` (`studio/label.py:71-82`), pack manifests via `core/registry.py:339-344`, `coding/m3/gate.py`, `coding/m4/build_traces.py`, `coding/runtime/r3.py`; M10 (`runtime/agent_evidence.py`, `interop/harbor/agent_harness.py:884-917`) |
| Layer | `crucible.connectors` imports only core (`pyproject.toml:160-164`); Harbor only under `interop/harbor/` |
| Principles | P16, P27, P28, P18 (DEPEND on Harbor), P19 (domain connectors in the pack); ARCH §7 rule 5 met, rule 4 (input hashes) partly (L-16) |

**Gate.** Passed. The last report, `docs/modules/reports/m3_gate.json` (committed at `b65754b`), shows 9/9 checks and 106
traces: 55 PRs, 1 construction session (9,684 events), 11 frontier runs (3 good / 5 bad / 3 unknown), 4 CI reports, 11
policy docs and 24 toy logs. It planted 31 canary formats and none leaked, and re-ingest was idempotent (106 -> 106). The
gate has not been re-run since `capture_proxy`, `agent_runs`, `redact` and `trace_store` changed. In this run, all 80
`tests/unit/test_m3_audit*.py` tests pass.

| Consumer | Uses M3 output? | Evidence |
|---|---|---|
| M4 signals | PR, agent_runs and CI traces; no capture traces | `coding/m4/build_traces.py:46-55` (PR only); `coding/signals.py:213-280` |
| M5 workflow | yes, `context_reset` included | `workflow/steps.py:123`; `coding/m56/rediscovery.py:94` |
| M8 forge | yes: G-trap tasks from M4 -1 signals | `coding/forge/derive.py:9-13,137` |
| M10 runtime | raw capture logs, not stored traces | `runtime/agent_evidence.py:141-146`; `interop/agentenv/harness.py:127-170` |
| M15 label | yes | `studio/label.py:71-90`; `coding/studio.py:158-175` |
| Stored capture/ATIF/OTel/chat traces | no consumer | `coding/m3/gate.py:138-145` (one synthetic row); `core/registry.py:341-344` |

This decides the tiers. The should items hit M10's direct reads (R-1, R-2), `agent_runs` labels (R-3) and every ingest
(R-4). An item that needs a consumer of stored capture, ATIF, OTel or chat traces is later.

## 2. Status ledger

| Item | State | Evidence |
|---|---|---|
| IMP-M02-S1 | done | World imports `core.agent_events` (`world/world.py:33`), enforced by the layers contract (`pyproject.toml:166-185`). |
| AUD-FIXED, AUD-C22 | done | 80 `tests/unit/test_m3_audit*.py` tests pass at `d282251`; nested subagents walked where the call returns (`interop/harbor/atif_import.py:63-121`). |
| SWP-M57-D5, SWP-M810-6 | done | `calls_from_capture` reuses `fetch` (`interop/agentenv/harness.py:127-142`); one pattern list scrubs records (`core/secrets.py:12-49`, `core/schema/run.py:163-166`). |
| RES-M3-3 | superseded | `prefix_break` is built in the runtime record (`runtime/record.py:148-158`), not in M3. |
| IMP-M3-1, AUD-F5, SWP-M57-280 | not done -> L-14, L-16 | `TraceSource` has no raw hash or connector version (`core/schema/trace.py:22-27`); `derived_from=[]` **[ran: p10_ledger]**. |
| IMP-M35-S2, IMP-M02-S3 | partly -> L-16 | Provenance is now recorded (`connectors/trace_store.py:75`), but the `RawRecord` is dropped (`connectors/base.py:116`). |
| IMP-M3-2, IMP-M35-S5 | not done | One episode via capture and chat gives `capture:ep1` and `chat:ep1`, with no link **[ran: p10_ledger]**. |
| IMP-M3-3, AUD-F2, HO-M01-1, HO-M02-1, SWP-M57-922 | not done -> R-2, L-1, L-6 | Lineage is the prefix rule (`connectors/capture_proxy.py:826-863`); an unmarked non-adjacent repeat is kept **[ran: p10_ledger]**. |
| IMP-M3-4, AUD-F10, STUDY-RED-F8, STUDY-RED-F9, STUDY-RED-D15 | not done | No entropy or NER detector, names come only from the trace's actors (`connectors/redact.py:188-196`), no per-string bound (:198-218), and redact does not import `QUERY_SECRET` (:29; `core/secrets.py:48`): `?key=` and IPv6 ULA survive **[ran: p10_ledger]**. |
| IMP-M3-5 | not done | PR selection is `title_re` only, with no `skip_reason` (`coding/connectors/github_pr.py:108-142`). |
| RES-M3-6 | partly | `is_fixlike` feeds post-merge signals only (`coding/connectors/github_pr.py:39-54,425`). |
| IMP-M3-6 | partly -> L-13 | The runtime uses each agent's converter (`interop/harbor/agent_harness.py:880-905`); import knows 2 (`interop/harbor/atif_import.py:132-145`) and already validates (:164). |
| IMP-M3-7, DOC-open-3, DOC-open-2 | not done -> L-12, L-10 | OTel tests use synthetic spans only (`tests/unit/test_m3_audit_capture.py:235`); proxy tests use fake upstreams and the live R3 run never ran (`coding/runtime/r3.py:1-30`); for token ids see AUD-F7. |
| IMP-M3-8 | not done | ATIF token ids and logprobs are excluded on import (`interop/harbor/atif_import.py:90`) **[ran: p11_atif]**. |
| IMP-M3-9, IMP-M3-10, RES-M3-5, AUD-F1, AUD-F8, DOC-open-1 | not done | No hf_dataset, experiment_log, commit-grammar, issues/gitlab or standalone vcs_history connector (`core/registry.py:339-344`); none has a consumer. |
| IMP-M4-3, SWP-M4-G4 | partly | The CI mapper masks infra runs (`coding/signals.py:253-280`), but reruns are a boolean, `flaky()` is uncalled, and PR `timed_out` scores -1 (:167-175). |
| AUD-D26, AUD-REN | not done | Commits are booked to the PR author (`coding/connectors/github_pr.py:234-235`); only exact renames are followed (`coding/git_history.py:99,115-132`). |
| AUD-D27, AUD-D11-live | not done | "Do not hesitate" gives `must_not` **[ran: p10_ledger]** (`coding/connectors/policy_docs.py:32-36`); the Postgres branch is untested (`connectors/sql_audit_log.py:30-48`). |
| AUD-F3 | not done -> L-20 | Known values match as exact strings (`connectors/redact.py:155-156`); URL-encoded and base64 copies leak **[ran: p10_ledger]**. |
| STUDY-RED-D6 | partly | ya29, hvs and Slack webhooks are now removed; bare AWS, Twilio `SK`, `shpat_` and `ATATT` keys are kept **[ran: p10_ledger]** (`core/secrets.py:12-36`). |
| AUD-F4 | not done | Redaction runs after `normalize` truncates to 200/4000 chars (`connectors/base.py:116`; `connectors/capture_proxy.py:82,692`). |
| AUD-F6 | not done -> L-17 | One bad JSONL line fails `discover` for the whole glob (`connectors/file_drop.py:83-88`) **[ran: p10_ledger]**. |
| AUD-F7, RES-M3-1, RES-M3-4 | partly -> L-10 | Non-streamed token ids are read (`interop/agentenv/harness.py:112-124`), streamed ones are dropped (`connectors/capture_proxy.py:509-551`); Responses and a 401 key exist. Correction: rLLM's cumulative mode does exist. |
| AUD-F9 | not done | Five parsers only (`coding/connectors/ci_logs.py:220`); other formats fall to `unknown` **[ran: p_swe_safety S2]**. |
| HO-M00a-1, HO-M00b-1 | partly | CAS head and ATIF reconcile are done (`connectors/trace_store.py:76-89`, `interop/harbor/reconcile.py`); `event_key` skips NFC (`core/schema/trace.py:62`); `store._db` is still read (`trace_store.py:109`). |
| STUDY-CAP-F2, STUDY-CAP-F11, STUDY-CAP-F12 | not done | No keepalive (`connectors/capture_proxy.py:338-387`); thinking `signature: None` **[ran: p10_ledger]**; the session comes from the key only (:235-240). |
| STUDY-CAP-F4 | partly | Per-exchange `complete`/`stream_drop`/status flags exist (`connectors/capture_proxy.py:334-383`), but no session-level verdict. |
| STUDY-CAP-D12 | not done -> L-7 | Each record stores the full request (`connectors/capture_proxy.py:319`): 230 MB at 800 calls **[ran: v2_lineage_scale]**. |
| STUDY-RED-F3, DOC-open-4, DOC-open-5 | not done (open-4/5 accepted) -> L-22 | Raw logs are kept indefinitely; frozen sessions are raw copies (`coding/connectors/claude_code_sessions.py:57-66`); the clone is blobless (`coding/git_history.py:43`). |
| STUDY-RED-F4 | not done -> L-14 | The stamp hashes the rules only (`connectors/redact.py:301-308`); hit counts are discarded (:167-170). |
| STUDY-LOG-F9 | not done | Corrected: NeMo-Gym's `failure_kinds.py` is a rollout vocabulary already used in `core/schema/run.py`, so this is not an M3 item. |
| STUDY-LOG-D21 | not done | An actor's role is fixed by its first row (`connectors/file_drop.py:171-172`) **[ran: p10_ledger]**. |
| DOC-stale-1, ARCH-M3-1 | not done | 20 vs 31 canaries (`ARCHITECTURE.md:40`); `/v1/responses` is served (`connectors/capture_proxy.py:423`); the unused atif2otel DEPEND (`ARCHITECTURE.md:218,509`). |
| NEW-M3-4, SWP-M57-262 | partly | The `ARCHITECTURE.md:386` tree still lists non-files; `docs/research/improvements/M03-M05.md:43` keeps the wrong NeMo-Gym cite. |
| RES-M3-2 | partly | `context_reset` is emitted (`connectors/capture_proxy.py:700-701`) and, correcting the ledger, read by M5 (`workflow/steps.py:123`); ATIF import emits none. |
| NEW-M3-1 | not done -> L-11 | File-form subagent refs are skipped (`interop/harbor/atif_import.py:68,111-121`) **[ran: p11_atif, p40_harbor H2]**. |
| NEW-M3-2 | not done -> R-2 | The docs say a retry is collapsed to the attempt seen (`connectors/capture_proxy.py:11-13`; `docs/modules/M03-connectors.md:23`). |
| NEW-M3-3 | partly -> L-8 | The 64 MiB cap is checked after the full read, and `passthrough` is uncapped (`connectors/capture_proxy.py:264,277-284`). |

## 3. Comparison questions

| # | Question | Axis |
|---|---|---|
| Q1 | Which attempt does the log say the agent saw when a call is retried, streamed, cut off or abandoned, and how is an incomplete exchange marked? | correctness (capture fidelity) |
| Q2 | How are a session's calls linked into conversations, and how are side calls, subagents, compaction and retries represented? | correctness (lineage) |
| Q3 | What is removed before anything is persisted (patterns, key names, known values and their encodings, PII), and does redaction see the raw payload? | security (redaction) |
| Q4 | Where do raw captures, caches and frozen transcripts live, with what permissions and lifetime, and is there a scrub or purge? | security (raw data at rest) |
| Q5 | How is a record identified and versioned (id, content hash, raw digest, normalizer and schema version), and how does re-ingest map onto the head? | identity and reproducibility |
| Q6 | When one episode arrives through several channels, how is it recognised as one, and how are ids namespaced? | identity (cross-source) |
| Q7 | Which dialects and trajectory formats are captured or imported, and what is lost (token ids, logprobs, signatures, subagent refs)? | interop |
| Q8 | How does ingest handle one bad record or source (quarantine, isolation, abort), stream large inputs, and report skips? | operability |
| Q9 | How are test logs parsed, reconciled with the runner's summary, separated into infra and code failures, and checked for flakiness? | correctness (CI logs, P16) |
| Q10 | How is PR and commit history obtained, how are reverts linked, and how are renames and force-pushes followed? | correctness (VCS) |
| Q11 | How are tool calls that reach answer sources detected, how is "ran" told from "exposed", and does exposure change the label? | evaluation integrity |
| Q12 | How do capture storage and lineage scale with session length, and how are bodies and logs bounded? | performance and scale |

## 4. Repo triage

33 reference folders: 11 primary, 15 secondary, 7 not relevant. No folder is absent and no sweep failed.

| Repo | Class | Reason | Commit | Licence |
|---|---|---|---|---|
| harbor-framework-harbor | primary | Owns ATIF, the converters M3 DEPENDs on, a validator and a post-trial scrub | a2fafbcf | Apache-2.0 |
| primeintellect-ai-verifiers | primary | Recording interception proxy with retry identity and message-graph lineage | e22533430 | MIT |
| nvidia-nemo-gym | primary | Capture in three dialects, token capture with lineage, strict ATIF import with digests | 7ece89985 | Apache-2.0 |
| meta-pytorch-openenv | primary | Recording proxy, rollout graph, capture validation and capture-to-ATIF reconcile | 8d3455ab | BSD-3-Clause |
| swe-bench-swe-smith | primary | Per-runner test-log parsers, PR mining, SWE-agent trajectory conversion | 9b74ac0 | MIT |
| huggingface-repo2rlenv | primary | Test-log parsers, PR and commit mining, build-log redaction, evidence digests | 27d04ee | Apache-2.0 |
| preferencemodel-karotte | primary | Native agent logs to a normalized transcript; surrogate-safe serialization | 31c693e | MIT |
| infra-inclusionai-areal | primary | Session proxy caching external-agent calls in a parent tree; retry-orphan rule | aedc21ec | Apache-2.0 |
| infra-thudm-slime | primary | Three-dialect adapters recording turns into a per-session message trie | 05e3b017 | Apache-2.0 |
| infra-novasky-ai-skyrl | primary | skycap: capture proxy with a context-graph forest and versioned records | a1e34dbf | Apache-2.0 |
| infra-rllm-org-rllm | primary | rllm-model-gateway: path-scoped capture proxy with token injection; ATIF bridge | 4e828c81 | Apache-2.0 (root classifier MIT: root `rllm/` PATTERN only) |
| agentenv-framework | secondary | No connector; per-site secret scrubs, OTel span reader, writer-version ledger | 041afd7 | Apache-2.0 |
| primeintellect-ai-prime-agent | secondary | Coding agent; request-id idempotency, JSONL tail rule, counted skips | 3e0893260 | MIT |
| primeintellect-ai-research-environments | secondary | Environments; stash-aware answer-source rules, vendored log parsers | f63f07c8a | Apache-2.0 (vendored parsers unclear) |
| adithya-s-k-fineenvs | secondary | Versioned derived index; per-rollout model allow-list proxy | 5e0b46e | Apache-2.0 |
| 08-hf-rl-env-datasets | secondary | 41 dataset dumps; run-identity and flaky-list patterns only | none (not a git repo) | per dataset: Apache-2.0, MIT, CC-BY-4.0, ODC-BY, AGPL-3.0, other |
| swe-agent-mini-swe-agent | secondary | Trajectory producer `agent_runs` ingests; typed exit statuses | 04d809ce | MIT |
| karpathy-autoresearch | secondary | One possible source format (results.tsv + git); no consumer | 228791f | none (README says MIT) |
| xiaomi-mimo-v2.6-rl-oss | secondary | HF task dump; session-reading verifiers, audit-table environments | none (HF rev 639865f) | none (README front matter apache-2.0) |
| xiaomimimo-verl | secondary | verl fork; ungradable infra statuses; capture submodules uninitialized | a2ad9f61 | Apache-2.0 |
| infra-volcengine-verl | secondary | Upstream verl; unredacted rollout tracing, sampled traces | 2035f58d | Apache-2.0 |
| infra-primeintellect-ai-prime-rl | secondary | Trainer; chunked append-only episode stream with update records | 1e5318108 | Apache-2.0 |
| infra-nvidia-nemo-rl | secondary | Trainer; fail-closed terminal attribution, k8s secret masking | 925f70d55 | Apache-2.0 |
| infra-openrlhf-openrlhf | secondary | Trainer; in-memory per-session token buffer server | dc2a7ad | Apache-2.0 |
| infra-huggingface-trl | secondary | Trainer; CLEAN/REALIGN/FORK reconciliation over OpenEnv captures | d9b9f39b | Apache-2.0 |
| infra-alibaba-roll | secondary | Engine-side proxy; hash-chained prefix, OTLP id normalisation | 0f37fe13 | Apache-2.0 |
| primeintellect-ai-prime-environments | not relevant | Eval environments; exit-code test runs only | cf220ee | Apache-2.0 |
| servicenow-enterpriseops-gym | not relevant | Benchmark runner; current-state DB seeds; a harness-retry scenario only | 271f2c3 | Apache-2.0 |
| veds12-git-swarm | not relevant | Project website; no commit-grammar parser | 9f66af0 | none |
| infra-moonshotai-checkpoint-engine | not relevant | Weight-update library (M11/M12) | 84631a2 | MIT |
| hf-space-fineenvs-multi-harness-rl | not relevant | Article site; prose on OpenEnv capture (claims only) | none (not a git repo) | CC-BY-4.0 |
| hf-space-fineenvs-simulation-rl-environments | not relevant | Article site; no M3 content | none (not a git repo) | CC-BY-4.0 |
| hf-space-rl-environments-guide | not relevant | Article site; reward patterns only | none (not a git repo) | CC-BY-4.0 |

## 5. Findings per repo

### harbor-framework-harbor
- Owns ATIF and the converters M3 DEPENDs on (HEAD = pin). Lineage is declared (subagent refs, `continued_trajectory_ref`,
  compaction, `models/trajectories/trajectory.py:29-82`) and followed (`utils/traces_utils.py:714-747,806-833`); Crucible
  resolves embedded refs only **[ran: p40_harbor]** (L-11).
- Corrections: import already validates (same file, :120-185); only ClaudeCode/Codex offer `convert_trajectory`
  (`agents/installed/base.py:356-358`); Terminus-2 segments lack `trajectory_id` (`terminus_2.py:2056-2089`).
- Verdicts: better on declared lineage and token fields; Crucible better on capture, redaction (`trial/trial.py:1107-1160`),
  versioning, P16, probes.

### primeintellect-ai-verifiers
- Recording proxy (since `ae819de` only logging/sampling changed): one record per exchange in a `finally`, no commit without a
  terminal event (`verifiers/v1/interception/server.py:887-910,210-236`); marked retries and `Idempotency-Key` get the stored
  response (`:89-101,630-715`), which would alter an external agent's run.
- Message graph keyed (parent, tools hash, message hash), each message stored once (`graph.py:241-341`), exact prefix by
  design (`:6-8`); no redaction; `VersionInfo` per trace (`trace.py:91-101`).
- Correction: no RFC files exist (`docs/audit/2026-10-12/M03-study-capture.md:12`). Verdicts: better on lineage cost and
  producer stamp (L-3, L-7, L-14); Crucible better on redaction, file modes, identity.

### nvidia-nemo-gym
- Three-dialect capture, token capture with lineage, strict ATIF import; since `3ef478d`: explicit retire, id-witness
  attribution, shielded write. `CancelledError` is its own class (`nemo_gym/base_responses_api_model.py:1166-1169`) (R-1);
  retry siblings resolved by continuation (`token_id_capture/builder.py:383-427`) (R-2, L-1).
- Fail-closed parents (`lineage.py:244-325`); `previous_response_id` by exact id (`harness_capabilities/checker.py:466-480`)
  (L-5); ATIF join pins `source_sha256`, refuses repeats (`atif_reverification.py:134-150,273-316`) (L-15, L-16).
- Corrections: AUD-F6's quarantine is a silent skip (`train_data_utils.py:750`); STUDY-LOG-F9's list is rollout-level;
  STUDY-RED-F9's strip covers spans and logs only. Verdicts: better on attribution, identity, scale; Crucible better on
  redaction, file modes, probes, CI.

### meta-pytorch-openenv
- Recording proxy, unchanged since `49aa302`; its reconcile is ported (`interop/harbor/reconcile.py`). Only successes become
  nodes, 429/5xx are retried inside (`src/openenv/core/harness/capture/upstream.py:404-412`), and a 200 with no choices is not
  a turn (`server.py:1250-1260`) (L-2).
- Parent = earlier request plus response as a prefix (`graph.py:299-366`); with no token ids every childless root is discarded
  **[ran: p_oe1_discarded_attempt]**; uuid ids, no body cap (`server.py:926-929`).
- Corrections: a full RolloutGraph port (IMP-M3-3) loses side-call vs reset, so take only its continuation check (L-1);
  IMP-M3-8's merge is `harbor/atif.py:498-563`. Verdict: Crucible better as evidence and on identity.

### swe-bench-swe-smith
- Per-repo log parsers, PR mining, SWE-agent conversion (HEAD = pin). Timeout = `resolved: False`, synthetic CTest names,
  flaky repair after one gold run (`harness/eval.py:69-79`, `profiles/cpp.py:473-486`, `harness/repair.py:138-215`); extra
  formats fall to `unknown` in Crucible **[ran: p_swe_safety S2]**.
- API PR mining without reverts (`bug_gen/mirror/collect/utils.py:49-170`); logs 10 token chars (`:67`); a cost-limit exit
  becomes a submit (`train/traj_mgr/utils.py:86-91`). Verdict: Crucible better; no candidate.

### huggingface-repo2rlenv
- Since `5555443`: unittest, pytest-footer and nextest parsers. Build logs lose credential-named env values, URL userinfo and
  auth headers before the cap (`tasksmith/build_logs.py:38-68`) (L-18); labels bind the subject hash
  (`emitter/evaluation.py:30-58`); no summary reconciliation or infra masking (`pipelines/_pr_runtime_verifier.py:476-547`).
- Correction: pytest parser now `log_parsers/pytest_parser.py:32-118`. Verdicts: better on process-held secrets and identity;
  Crucible better on CI verdicts and PR history (reverts by title only, `pipelines/commit_runtime.py:85,93`).

### preferencemodel-karotte
- Native logs to a transcript (HEAD = pin): only the successful attempt kept (`agents/builtin_source.py:219-300`), subagent
  lines dropped (`agents/grok_build/adapter.py:58-59`), bad lines skipped (`agents/mistral_vibe/adapter.py:125-137`).
- Surrogates escaped before serialization (`text.py:4-24`), ported to `core/text.py:44-59` but unused by M3 **[ran:
  p_swe_safety S3/S4]**. Verdicts: better on surrogate safety (R-4, L-17); Crucible better on subagents and redaction.

### infra-inclusionai-areal
- Session proxy caching calls in a parent tree (since `298412a` only image checks). The sibling a later turn continued wins
  (`areal/experimental/openai/cache.py:258-367`, opt-in) (L-1); undelivered interactions skipped at export (`:620-639`) (R-2),
  yet output is cached before delivery (`openai/client.py:1322-1343`).
- Correction: THIRD_PARTY.md:171's pin is absent; the pattern is at `proxy/workflow.py:585-630`. Verdicts: better on retry
  attribution and matching cost; Crucible better on incomplete marking, fidelity (`anthropic.py:25-85` flattens), redaction.

### infra-thudm-slime
- Three-dialect adapters recording turns into a per-session trie (since `c933787` a Responses adapter). A client gone before
  the flush gets 499 and no record (`slime/agent/adapters/common.py:428-443`) (R-1, R-2); trie walk O(L)
  (`trajectory.py:428-511`) **[ran: p60_areal_slime D: 0.13 s vs 6.23 s at 800 calls]** (L-7); cap while reading
  (`adapters/common.py:221`) (L-8).
- Verdicts: better on lineage cost and bounds; Crucible better on failure evidence, fidelity (tool ids dropped,
  `adapters/anthropic.py:79-140`), keys (sid = bearer, `adapters/common.py:468-473`).

### infra-novasky-ai-skyrl
- skycap (no change since `3d0ae22`): marked retries replayed, replies after `finish` not committed
  (`skycap/src/skycap/retry.py:92-123`, `skycap/text.py:143-146`); a leaving client not told from an upstream cut
  (`skycap/text.py:127-131`).
- Edits, resamples, subagents and compaction are forks (`graph.py:1-39`) (L-6), but stripped reasoning forks every turn
  **[ran: p_tc3_history_edit]**; trie matching 1.85 s vs 8.36 s at 800 calls **[ran: p_tc2_lineage_scale]**. Verdicts: better
  on edit forks and cost; Crucible better on durability (`record.py:20-21`), auth, redaction.

### infra-rllm-org-rllm
- Gateway (HEAD = pin): no status stored, any chunked stream an answer
  (`rllm-model-gateway/src/rllm_model_gateway/proxy.py:613-623`); Anthropic stored empty (`data_process.py:264-279`); no auth
  (`server.py:222-225`); streamed token ids kept (`data_process.py:104-139`) (L-10); the ATIF bridge follows chains
  (`rllm/integrations/harbor/atif_trajectory_bridge.py:43-77`) (L-11).
- Correction: `token_accumulator.py:53-153` exists, wired and tested (ledger RES-M3-4). Verdicts: better on streamed tokens;
  Crucible better on status, completeness, auth, dialects, identity.

### Secondary patterns
- **agentenv-framework:** 3-form key scrub, keys < 16 refused (`.../sail_vm/model_key.py:50-51,196-208`) (L-20, L-21).
- **primeintellect-ai-prime-agent:** `Idempotency-Key` (`.../semantic_edges.rs:37-39`) (L-3); JSONL tail rule (L-4).
- **primeintellect-ai-research-environments:** stashes and reflogs named as answer sources
  (`.../swesmith_env/taskset.py:57-58`) (L-23).
- **swe-agent-mini-swe-agent:** exception -> `exit_status=<class>`, kill -> empty (`.../default.py:74-121,174`) (R-3).
- **adithya-s-k-fineenvs:** index tied to revision and version (`.../hf-rl-explorer/app/catalog.py:583-617`) (L-14); model
  allow-list (L-9).
- **08-hf-rl-env-datasets:** RetroEnv run identity names the code digest; SWE-rebench V2 counts infra noise as flaky.
- **karpathy-autoresearch:** a crash is logged `val_bpb 0.000000`; discarded commits are reset away (`program.md:75,104`).
- **xiaomi-mimo-v2.6-rl-oss:** `general/envs/*/run_verify.py:14-31` (all 925 envs) reads subagent files as the answer;
  audit tables ingest unchanged.
- **xiaomimimo-verl:** infra statuses never graded (`.../mimoagent_runner.py:42,279-281`) (R-3); capture submodules absent.
- **infra-volcengine-verl:** rollout tracing ships raw prompts off-box (`verl/utils/rollout_trace.py:364-427`).
- **infra-primeintellect-ai-prime-rl:** chunked append-only episode stream (`.../monitors/file/traces/chunks.py:1-115`).
- **infra-nvidia-nemo-rl:** fail-closed attribution (L-1); name/value masking (`.../_logging.py:28-75`) (L-19).
- **infra-openrlhf-openrlhf:** per-session token buffer; a resend asserts (`.../agent_func_openai_server_executor.py:119`).
- **infra-huggingface-trl:** CLEAN/REALIGN/FORK, no side-call rejoin (`.../async_grpo/async_rollout_worker.py:102-194`) (L-6).
- **infra-alibaba-roll:** longest branch kept (`.../message_tracker.py:397-400`); hex OTLP ids (`.../otel_receiver.py:47`).

## 6. Gap matrix

Legend: ● built and tested, ◐ built with a stated weakness, ○ absent, – not applicable. The columns are the primary repos
of section 5 (verifiers = primeintellect-ai-verifiers, AReaL, slime, skycap = infra-novasky-ai-skyrl, rLLM).

| Q | Crucible | Harbor | verifiers | NeMo-Gym | OpenEnv | SWE-smith | repo2rlenv | karotte | AReaL | slime | skycap | rLLM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | ◐ every attempt; client-left mislabelled (R-1, R-2) | ○ no proxy; retries unseen | ● per exchange; replays retries | ● cancel class; witness attribution | ◐ successes only | ○ no attempt record | – | ◐ successful attempt only | ◐ adoption; abandoned looks complete | ◐ after flush; failures dropped | ◐ replay; leave vs cut unclear | ○ no status, no terminal check |
| Q2 | ◐ prefix threads; edits re-emit (L-6) | ● declared refs and chains | ● message graph, exact prefix | ● fingerprint parents, fail closed | ◐ graph; aux roots discarded | ○ linear | – | ○ subagents dropped | ◐ parent tree, no side calls | ● trie; forks on divergence | ◐ forest; forks on stripped reasoning | ○ positional join |
| Q3 | ● one choke point; values, patterns, PII | ◐ exact env values, post-trial | ○ none | ○ config keys, URL queries | ○ sandbox output only | ○ none; logs token prefix | ◐ env values in build logs | ○ none | ○ config keys only | ○ none | ○ none | ○ none |
| Q4 | ◐ 0700/0600; no retention (L-22) | ○ umask, in-place scrub | ○ default umask | ◐ umask; explicit retire | ◐ memory, deleted at end | ○ default umask | ◐ redacted retained streams | ○ default umask | ○ default umask | ○ sandbox file | ○ umask; S3 mirror | ○ umask SQLite, no auth |
| Q5 | ◐ content-addressed, CAS head; no raw digest | ◐ run-scoped ids; no digest | ◐ build stamp; uuid ids | ● source sha256, write-once | ○ uuid; unversioned | ○ folder-hash ids | ● recipe version, subject hash | ○ no version or hash | ○ task-index ids | ○ sample ids | ◐ format version; random ids | ○ uuid4 |
| Q6 | ○ per-connector ids, no link | ◐ atif2otel join keys | – single channel | ◐ refuses in-batch repeats | ◐ capture-ATIF reconcile | ○ per-source names | – | – | – | – | – | ○ positional join |
| Q7 | ◐ 3 dialects; streamed token ids dropped (L-10) | ● ATIF owner, 40+ converters | ◐ 3 dialects; choice 0 only | ● strict ATIF, tokens, signatures | ◐ 4 dialects; n = 1 only | ○ cost exit rewritten | ○ ATIF tool calls only | ◐ raw passthrough field | ◐ Anthropic flattened | ◐ tool ids dropped | ◐ chat only; token mode | ◐ Anthropic empty; tokens kept |
| Q8 | ◐ per source; one bad line fails glob (L-17) | ◐ silent line skips | ○ bad line raises | ● counts damaged lines | ◐ never raises; silent skips | ◐ per-trajectory skip | ◐ per-PR skip reasons | ◐ skips; logs raw line | – | – | ◐ failures listed | – |
| Q9 | ● 5 parsers, summary check, P16 | ○ any non-zero exit = 0 | – | ○ CTRF counts only | ○ Julia counts only | ◐ many formats; timeout = fail | ◐ many formats; no infra mask | – | – | – | ○ R2E parser | ○ R2E parser |
| Q10 | ◐ git-only; edited renames lost | – | – | – | – | ◐ API mining; no reverts | ◐ gh + git; reverts by title | – | – | – | – | – |
| Q11 | ◐ deterministic; no stash form (L-23) | ○ LLM rubric | ○ prevention only | ○ best-effort pre-run scrub | – | – | ○ construction-time guards | – | – | – | – | – |
| Q12 | ◐ cubic lineage, quadratic log (L-7) | – | ● linear graph, index lookup | ● deltas, hashed lookup | ◐ O(n²·L), no body cap | – | ◐ 5 M-char parse cap | ● bounded metadata | ◐ sorted prefix; no cap | ● trie; cap while reading | ◐ trie; written at end | ○ request stored twice |

## 7. Recommendations, ranked

- All tiers are the verifiers'. Verification lowered several candidates: C1 must -> should; OE-1 and NG-4 should -> later;
  FE-1, TA-C1, PE-2, NG-3, C3 and TA-C3 should -> later; PE-1 group and TA-C4 (its timeout trigger is R-2's) should ->
  later. The code agrees: `stale()` has no caller outside tests, and atif, otel and chat have no consumer.
- Duplicates were merged across verify batches. R-3 = swe-safety:C2 (verify-1) + mimo-verl:MV-1 (verify-4). They disagreed
  on an empty exit status; the code decides it (R-3, Reasoning). L-11 = harbor:H-1 (verify-2: one trace per segment) +
  trainers-c:TC-3 (verify-4: follow the chain into one trace). The smaller per-segment id is the default; follow the chain
  only if a consumer needs one trace per run.
- Order of work: R-1 first, because its uvicorn-served disconnect test is R-2's harness and R-2 reuses its string. R-3
  changes `coding/connectors/agent_runs.py` and `coding/signals.py` together. R-4 is independent. Among the later items,
  L-14 comes before L-13 and L-16, which ride in its head note.

| R | Title | Tier | Kind | Impact | Effort | Risk |
|---|---|---|---|---|---|---|
| R-1 | An abandoned stream is logged as client disconnected | should | defect | H | L | L |
| R-2 | A client-left non-streaming exchange is marked and never an action | should | defect | M | L | L |
| R-3 | agent_runs: a harness- or provider-ended run is `unknown`, unscored | should | defect | M | L | L |
| R-4 | Escape lone surrogates before storage | should | defect | M | L | L |

### R-1. A stream the client abandons is logged as "client disconnected" under uvicorn
- **Kind** defect · **Tier** should · **Status** correction of the 2026-10-12 fix D3; merged nemo-gym:NG-1 · **Origin**
  introduced since the 2026-10-12 audit (`d99fde1` made the `90f2539` label decide blame)
- **Problem.** `relay()` names only `GeneratorExit` "client disconnected" (`connectors/capture_proxy.py:374-375`). Under
  uvicorn 0.54.0 (`coding/runtime/r3.py:174`, ASGI spec 2.3), Starlette 1.7.0 cancels the stream task on disconnect
  (`starlette/responses.py:267-282`). The row then reads `CancelledError: ...`, and `verdict_of` returns `infra`
  (`runtime/agent_evidence.py:141-146`). When Crucible's deadline kills a streaming agent (claude-code or codex via Harbor,
  or an A2A agent), the probe answers ok and the end becomes `agent_run_error` with blame `unknown` (:322-327,371-376)
  instead of an evidenced `timeout`. The run is masked and counts toward the 5% stop. R3 does not stream.
- **Reference.** NeMo-Gym classifies `asyncio.CancelledError` on its own
  (nvidia-nemo-gym/nemo_gym/base_responses_api_model.py:1166-1169); slime calls it "client disconnected"
  (infra-thudm-slime/slime/agent/adapters/common.py:428-443).
- **Recommendation.** A relay ended by the downstream side records `error: "client disconnected"` and `complete: false`,
  keeping the status and partial response. Downstream endings are `GeneratorExit` and `asyncio.CancelledError`, the latter
  raised by the server's task group on disconnect or shutdown. Upstream failures keep `"<Type>: <message>"`. Only the
  exception check at `capture_proxy.py:375` changes.
- **Reasoning.** Widening `verdict_of` instead would leave the wrong cause in M3 and tie M10 to a Starlette message. A
  separate `cancelled` class has no consumer, and every probe wrote the row without a shielded write. Follows P16.
  Trade-off: a mid-stream proxy shutdown is labelled the same.
- **Established by.** Code read on both sides. **[ran: v1_client_left S3]** Real OpenAI SDK under uvicorn: HEAD
  `CancelledError` with verdict `infra`; patched copy `client_left`. **[ran: p_nemo_04b_disconnect_tcp]** A real TCP
  upstream gives the same row. **[ran: p_nemo_05_verdict]** `infra` gives `unknown` blame, while `client_left` gives an
  evidenced timeout. No existing test drives a served disconnect.
- **Impact / Effort / Risk.** H: masked deadline stops feed the 5% rule. L: one exception tuple and one test. L: only
  downstream ends change.
- **Mode.** PATTERN (NeMo-Gym Apache-2.0).
- **Acceptance.** A uvicorn-served test abandons a stream. It fails at HEAD and passes after the fix: one row with
  `complete: false`, error `client disconnected`, and verdict `client_left`. `test_m3_audit_capture.py` and the M10 R-1
  tests still pass.

### R-2. A client-left non-streaming exchange is marked; no client-left exchange is an action, trainable call or retry copy
- **Kind** defect · **Tier** should · **Status** new; merges TB-1, TC-1, TC-2, NG-2 · **Origin** present at `90f2539`
- **Problem.** `forward_call` records a non-streaming 200 once the upstream answers, even if the client left
  (`connectors/capture_proxy.py:55-56,393-395`). `normalize` emits its tool calls with `writes` (:754-763),
  `calls_from_capture` lists it (`interop/agentenv/harness.py:142-169`), and `fetch` folds only onto a failed or marked
  attempt in the last 5 rows (`capture_proxy.py:642-657`). A deadline kill of mini-swe-agent mid-call
  (`interop/harbor/agent_harness.py:850-858,888-890`) makes the late answer the trace's last action (`res.steps`, :917): a
  write that never happened. A timeout-then-retry also keeps the abandoned original. Hurt: Harbor/A2A call lists, rollout
  turns.
- **Reference.** slime records a turn only after the flush (infra-thudm-slime/slime/agent/adapters/common.py:428-443);
  AReaL skips undelivered calls (infra-inclusionai-areal/areal/experimental/openai/cache.py:620-639).
- **Recommendation.** Invariant: an exchange the agent never received stays as evidence but is never an action or a trained
  turn. (1) Before `_record`, the proxy checks `await request.is_disconnected()`; if so, it records `error: "client
  disconnected"` with status, response and usage, leaving `complete` unset so spend stays certain
  (`runtime/agent_evidence.py:507`). (2) One predicate, `client_left(row)`, for all readers. (3) `normalize` emits the
  `llm_call` with `data.error` and no `tool_call`. (4) `fetch` treats it as failed, in either order, session-wide. (5)
  `calls_from_capture` marks an unfolded row `trainable: false`; reconcile skips it (`interop/harbor/reconcile.py:220`).
  Fix the NEW-M3-2 docs (`capture_proxy.py:11-13`, `docs/modules/M03-connectors.md:23,109`).
- **Reasoning.** The disconnect is direct evidence. Rejected: TC-1's sent-last rule (misses unmarked resends and the
  deadline), TC-2's `after_end` (the key is revoked before the kill), continuation (cannot resolve the final call; L-1),
  record-after-flush (loses M10's row), retry replay (alters what the agent gets). P28, P16. Trade-off: one `receive()`.
- **Established by.** Code read on both sides. **[ran: v1_client_left S1]** Real SDK, 0.5 s timeout against a 2.0 s
  upstream: HEAD keeps 3 exchanges for 2 turns and `C-call` writes `file:a.py`; patched, 2. **[ran: v1_client_left S2]**
  Deadline kill: HEAD writes a 200 row and `K-call` writes `file:out.txt`; patched, `client_left` with usage. **[ran:
  p61_nonstream_disconnect]** `is_disconnected()` is true only for the client that left.
- **Impact / Effort / Risk.** M: every deadline-killed run with a call in flight lists an action the agent never received.
  L: one check, one predicate, four tests. L: only rows whose client is gone change.
- **Mode.** PATTERN (slime, AReaL Apache-2.0).
- **Acceptance.** The deadline-kill shape gives a client-left row with usage, no tool_call and a non-trainable call. v1 S1
  gives 2 exchanges, a waiting client's row is unmarked, and the 80 `test_m3_audit*.py` tests still pass.

### R-3. agent_runs: a run its harness or provider ended is `unknown`, never `bad`, and M4 does not score it
- **Kind** defect · **Tier** should · **Status** new; merged swe-safety:C2, mimo-verl:MV-1 · **Origin** present at `90f2539`
  (every graded reward < 1 was `bad`; `b65754b`, the M4 audit, exempted only the limits)
- **Problem.** `normalize` treats only `LimitsExceeded`/`TimeExceeded` with reward < 1 as unfinished
  (`coding/connectors/agent_runs.py:102`); any other exit is `bad` (:109-111). mini-swe-agent records a class name for an
  exception and an empty status for a kill (Reference). Two of the 11 recorded runs (`OpenRouterAPIError` HTTP 402,
  `RuntimeError`) are `unknown` only because they are ungraded. A grade report or SEEDS row would make them `bad`, as when
  M7 graded the v2 patch (`coding/bench_recorded.py:38-39`). `AgentRunSignalMapper` would then emit -1: it exempts only the
  limits (`coding/signals.py:243`), and its gate paths check no exit (:229-242). M8 turns -1s into traps
  (`coding/forge/derive.py:9-13`). Silent, against P16, latent today.
- **Reference.** mini-swe-agent's own stops (swe-agent-mini-swe-agent/src/minisweagent/exceptions.py:1-26;
  `agents/default.py:74-86,120-121,174`) are M10's `CLEAN_EXITS` (`runtime/agent_evidence.py:80`). MiMo never grades
  `InfraError` (xiaomimimo-verl/recipes/code/mimoagent_runner.py:42,279-281).
- **Recommendation.** Invariant (P16): a run is judged on its reward only if it ended on the agent's own stop. Add a
  predicate, `unfinished(exit_status, reward)`: reward < 1 and the status empty or outside `CLEAN_EXITS` (imported from
  `crucible.runtime.agent_evidence`). Such a run is `unknown` ("the run ended on <status> before it was done"). Limit exits
  keep today's rule, `RepeatedFormatError` stays scored, the stale-version and not-a-grade rules come first, and reward >= 1
  stays `good` under the contamination rule. Record `details.exit_status` in every branch (today only `agent_runs.py:103`).
  `AgentRunSignalMapper` checks the predicate before both gate paths.
- **Reasoning.** A list of exception names is never complete; refusing such runs (MiMo) loses traces M5 and M2 use; parsing
  messages duplicates M10; without the M4 change the -1 survives. The verify notes disagreed on an empty status; the code
  shows a kill saves one, so it is unfinished. Trade-off: a run that crashed after finishing is masked.
- **Established by.** Code read on both sides. **[ran: p_swe_safety S1]** Reward 0 with `OpenRouterAPIError`, `RuntimeError`
  or `''` gives `bad`. **[ran: v4_mv1_exception_exit]** The real trajectories with `{reward: 0.0}` give `bad` and a -1
  signal, also via a failing gate. Of the 11 runs, 9 are `Submitted`.
- **Impact / Effort / Risk.** M: stops silently wrong labels and traps. L: one predicate, two call sites, three tests. L: no
  change at `d282251`.
- **Mode.** PATTERN (mini-swe-agent MIT, xiaomimimo-verl Apache-2.0; vocabulary only).
- **Acceptance.** The two real trajectories with `{reward: 0.0}` (also with a failing gate) and an empty-status run give
  `unknown`, `details.exit_status` and no mapper signal; `gpt-6-astra-v3-run2` stays `bad`; the gate keeps 3/5/3.

### R-4. Escape lone surrogates at the redaction boundary so one record cannot abort an ingest batch
- **Kind** defect · **Tier** should (narrowed from must) · **Status** new; merged swe-safety:C1; completes the 2026-10-12
  C3-C4 fix · **Origin** present at `90f2539`, where `_record` began writing `\udXXX` escapes; missed
- **Problem.** `ingest` runs `redact(normalize(fetch(ref)))` inside a per-source `try`, but `store.put` is outside it
  (`connectors/base.py:115-120`). A lone surrogate loaded by `json.loads` passes `normalize` and `redact`
  (`connectors/redact.py:245-298`), then `Store.put_doc` raises `UnicodeEncodeError` (`core/store/__init__.py:446`). That
  source and every later one are lost, and the error names no source. The capture path writes such records on purpose
  (`connectors/capture_proxy.py:213-215,629-631`), and a JS harness that cuts a UTF-16 pair produces them. Affected: every
  JSON-fed connector, through `crucible label` (`studio/label.py:82`) and the M3 gate. The World already escapes them
  (`world/world.py:966-967`). Not must: the failure is loud, nothing stored is damaged, and current real inputs hold none.
- **Reference.** karotte escapes surrogates in every string and key before serialization
  (preferencemodel-karotte/src/karotte/text.py:4-24), already ported as `core/text.py:44-59`.
- **Recommendation.** Invariant: every trace handed to the store is UTF-8 encodable. `redact()` first passes
  `trace.model_dump(...)` through `core.text.safe_json`, so a lone surrogate becomes escape text (`\ud83d`) in strings, keys
  and structural fields. `safe_json` returns its input unchanged when nothing needs escaping, so other traces keep their
  bytes and keys. `store.put` stays where it is.
- **Reasoning.** Escaping in each connector would miss the next one. Escaping in `Store.put_doc` or `core.canonical` would
  change stored bytes core-wide (M0 R-20/R-21). Moving `put` into the `try` would still lose the session. `redact()` is ARCH
  §7 rule 5's choke point (`ARCHITECTURE.md:836`). Trade-off: the store holds escape text, which loses nothing.
- **Established by.** Code read on both sides. **[ran: v4_c1_surrogate_ingest]** When the first session ends `\ud83d`, HEAD
  aborts and stores 0 of 3; with `safe_json` it stores 3 of 3. **[ran: p_swe_safety S3/S4]** Chat and capture behave the
  same. **[ran: v4_c1_surrogate_census]** 0 of 307 Claude Code transcripts and 0 of 11 frontier trajectories hold one.
- **Impact / Effort / Risk.** M: removes an unattributed abort of the whole batch; the trigger is rare. L: one call, one
  test. L: other traces are unchanged.
- **Mode.** PATTERN (karotte MIT; the notice is already present).
- **Acceptance.** A capture dir whose first session holds `\ud83d` stores 3 of 3, with the escape text kept. A
  surrogate-free trace keeps its `d282251` key. The 80 `test_m3_audit*.py` tests pass.

### Later (not recommended now)

| L | Title | Why not now | What would make it due |
|---|---|---|---|
| L-1 | Keep the same-body attempt a later request continued (OE-2, TA-C4, TB-2, NG-2's rule; HO-M02-1, SWP-M57-922) | after R-2 the residual triggers have no current harness and no consumer reads capture traces **[ran: p60_areal_slime A/B, p_oe1]** | live capture traces feed M4/M5 credit, or token-id calls train without an ATIF check |
| L-2 | A 2xx with an error body or no output is a failed attempt in the fold (OE-1) | error bodies are already marked failed; no resend after an empty 200 is known **[ran: p_oe2_empty_200]** | a real log shows an empty 2xx followed by an identical resend |
| L-3 | Use the recorded `Idempotency-Key` as call identity (PE-3) | the OpenAI SDK sends none (`openai/_base_client.py:406`) **[ran: p50_prime P50b]** | a client sending it without stainless headers is proxied |
| L-4 | Repair a torn capture-log tail before the next append (PE-4) | needs a proxy crash mid-write plus a later append **[ran: p50_prime P50c]** | runs resume into an existing attempt dir after a proxy restart |
| L-5 | Link Responses calls chained by `previous_response_id` (NG-4) | no proxied harness chains; Harbor's `use_responses_api` defaults off **[ran: v2_ng4: 4 resets, 11 steps vs 0 and 4]** | the first proxied Responses agent that chains server-side |
| L-6 | An in-place history edit re-emits later turns and results (TC-5, TA-C5, TB-4; HO-M01-1) | proxied harnesses keep linear histories **[ran: p_tc3, p_ta1]** | a pruning or re-rendering harness (OpenCode, Claude Code) is proxied |
| L-7 | Prefix-trie lineage instead of cubic `_threads`, with STUDY-CAP-D12 (NG-6, TB-3, TC-6, PE-5) | sessions are short **[ran: v2_lineage_scale: 9.22 s, 230 MB at 800 calls]** | a captured session passes ~500 calls |
| L-8 | Enforce the body cap while reading, on every route (TB-5; NEW-M3-3) | only a key holder can send such a body | concurrent untrusted rollouts share the proxy, or a proxy OOM |
| L-9 | Flag captured calls to a model other than the configured one (FE-2) | needs a policy deliberately calling another model | multi-model proxied runs feed leaderboards or training selection |
| L-10 | Keep streamed token ids and logprobs; fix the doc claim (TC-4; AUD-F7, RES-M3-1, RES-M3-4, DOC-open-2) | proxied runs are eval only **[ran: p_tc6_stream_tokens]** | the first train-mode rollout via the proxy; doc fix with DOC-stale-1 now |
| L-11 | ATIF continuation segments collide on one id; file-ref subagents vanish (H-1, TC-3; NEW-M3-1) | no pack uses `connector: atif` **[ran: p40_harbor H1/H2, p_tc5]** | a pack ingests Harbor job dirs through `connector: atif` |
| L-12 | otel_import reads OpenInference tokens and `gen_ai.prompt`/`completion`; real fixture (H-2, AE-3; IMP-M3-7) | no OTel input is ingested **[ran: v4_otel_attrs]** | the first real OTel export (`ARCHITECTURE.md:218`) |
| L-13 | Call Harbor's public `convert_trajectory`, not a private method (H-3) | the private call works at the installed Harbor | the next Harbor update, or with L-14 |
| L-14 | Head stamp records rules, logic, policy and normalizer version; every put refreshes it (FE-1, TA-C1, AE-4, PE-2; STUDY-RED-F4) | `stale()` is test-only; every path rebuilds or re-ingests **[ran: p_oe3, p_ta2, p_v3]** | a step calls `stale()`, or a store outlives a rule, policy or connector change |
| L-15 | Refuse two sources mapping to one trace id in one ingest call (NG-3) | in-use connectors have unique ids **[ran: p_nemo_01, p_v3 V1]** | the first pack ingesting chat_messages or atif |
| L-16 | Raw-source digest on every version (NG-7; IMP-M3-1, AUD-F5, SWP-M57-280) | no consumer reads one | a replay or re-verification must name the bytes it checked |
| L-17 | Skip and count a torn final JSONL line; otel/chat errors name path:line (PE-1, NG-5, C4, TA-C2; AUD-F6) | the only discover consumer reads committed files, and fails loudly (`notes/v4_torn_jsonl.out`) | an external exporter's folder is ingested |
| L-18 | Redactor also removes registered and credential-named env values (C3) | no shipped path stores capture logs **[ran: p_swe_safety S5]** | a pack ingests capture logs, or a tenant has format-less keys |
| L-19 | Redact credentials held as name/value pairs (TA-C3) | no stored source has that shape **[ran: p_ta4]** | audit exports with field/new_value columns |
| L-20 | Match known values in JSON-escaped and shell-quoted forms (AE-1; AUD-F3) | the only configured `.env` holds vendor keys **[ran: p20 PA2/PA3]** | a tenant value with a quote, backslash or apostrophe |
| L-21 | Report short or word-like known values (AE-2) | no configured `.env` holds dev defaults **[ran: p20 PA5]** | a pack whose redaction_env is a dev `.env` |
| L-22 | Retire a session's raw capture log after storage (NG-8; STUDY-RED-F3) | owner-only logs are accepted; no tenant traffic | proxied tenant runs, or a retention rule |
| L-23 | Stash and reflog-walk reads are answer-key probes (PE-6) | worlds delete stashes and reflogs (`world/guard/guard.py:749-772`) **[ran: v2_pe6_stash]** | unscrubbed third-party traces feed P27 |

### Rejected in verification

No candidate was rejected outright. These proposed mechanisms were dropped, and the failures they targeted were kept:

| Ids | Reason |
|---|---|
| trainers-c:TC-2 (`after_end` marker) | The key is revoked before the agent is killed (`interop/harbor/agent_harness.py:850-858`), so a reply after revocation can reach a live client; R-2 covers the failure. |
| trainers-c:TC-1 (`request_time` rule) | It covers SDK-marked retries only, not tenacity resends or the deadline kill; replaced by R-2. |
| swe-safety:C1 (put inside the per-source try) | The surrogate is the only per-trace put failure, and the session would still be lost; R-4 escapes instead. |
| PE-1, NG-5, C4, TA-C2 (per-file isolation) | It cannot be fail-closed per case and needs a new reporting channel; L-17 keeps the torn-tail rule only. |
| trainers-a:TA-C3 (`kind: Secret` rule) | The probed Secret was text, so L-19 keeps the name/value rule only; base64 values stay with STUDY-RED-D6. |
| trainers-a:TA-C1 (`core/fingerprint` stamp) | `model_dump` is already fail-closed; one keyed HMAC hides terms and values (L-14). |
| openenv-hf:OE-2 (root grouping by prompt ids) | Without token ids it discards every childless root **[ran: p_oe1_discarded_attempt]**; L-1 groups by digest. |

## 8. What we already do better (keep)

- **Every attempt is evidence, appended as it ends; a call cap is a stop** (`connectors/capture_proxy.py:213-218,291-396`)
  vs success-only capture (meta-pytorch-openenv/.../capture/upstream.py:404-412), unmarked cut streams (rLLM), silent
  retries (Harbor), end-only writes (skycap), a 429 cap that SDKs retry (infra-thudm-slime/.../adapters/common.py:349-370).
- **Secrets: keys stay in the proxy, constant-time per-session keys, one redaction choke point, owner-only raw files**
  (`connectors/capture_proxy.py:105-157,235-240`, `connectors/base.py:116`) vs agent-held keys and an env-value scrub
  (harbor-framework-harbor/.../trial/trial.py:1107-1160), no auth (rLLM), sid = bearer (slime), default-umask (NeMo-Gym).
- **Content-addressed versions, CAS head, signal carry-over** (`connectors/trace_store.py:46-89`) vs uuid ids
  (primeintellect-ai-verifiers/verifiers/v1/trace.py:383), max-version heads (agentenv), overwrites (Harbor).
- **Lineage tolerant of re-rendering; side calls rejoin** (`connectors/capture_proxy.py:800-863`) vs forks on stripped
  reasoning (infra-novasky-ai-skyrl/skycap/README.md:37-44), longest branch only (ROLL), no rejoin (TRL).
- **Raw dialect bodies, every choice kept** (`connectors/capture_proxy.py:319,509-593`) vs LiteLLM flattening
  (infra-inclusionai-areal/.../openai/anthropic.py:25-54), dropped tool ids (slime), choice 0 only (verifiers).
- **P16 CI parsing, deterministic probes, subagents kept** (`coding/connectors/ci_logs.py:151-323`,
  `core/agent_events.py:186-343`, `interop/harbor/atif_import.py:63-121`) vs timeout = fail
  (swe-bench-swe-smith/swesmith/harness/eval.py:69-79), an LLM rubric (Harbor), dropped subagents (karotte).

## 9. Open questions

| Question | Tried | Missing (who can supply it) |
|---|---|---|
| Do Claude Code or Codex rewrite earlier messages, chain by `previous_response_id`, or send compaction turns (`isCompactSummary`, Responses `compaction` items)? Decides L-5, L-6. | Harbor converter, slime harness configs, NeMo-Gym converters, grep | CLI source or a real captured session (the user) |
| How often does OpenRouter answer 200 with an error or no choices? Which `x-stainless-retry-count` does Harbor's mini-swe-agent install send? Does the Anthropic SDK send `Idempotency-Key`? Decides L-1..L-3. | OpenEnv comments; openai 2.54 and litellm 1.106 in the probe venv | a live R3 capture; `crucible/data/runtime` is absent (an R3 run) |
| Is a stream whose client leaves right after the terminal event logged `complete: false`, and does skycap tell this case apart (aiohttp)? | Read `connectors/capture_proxy.py:368-373`, `infra-novasky-ai-skyrl/skycap/src/skycap/text.py:101-140` | a live uvicorn probe with the Anthropic TS SDK |
| Does Crucible's Harbor harness ever run Terminus-2 (continuation files)? | `interop/harbor/agent_harness.py:255-272`; grep finds no use | a run config (M10 owner) |
| Does litellm 1.96.2 (locked) fetch its price map at import, and does Harbor's Claude Code converter accept a session frozen mid-write? | litellm 1.106 `litellm/__init__.py:567` (probe venv); `coding/connectors/claude_code_sessions.py:46-66` | the 1.96.2 source; a converter run on a torn file |
| Will a crashed run get a grade (R-3 going live)? | Run dirs; `coding/bench_recorded.py:38-39` | `data/m7/history/task04` is not in this checkout (the user) |
| Should M3 label agent-reported limit stops `unknown` (today) or score them as M10 does? | `coding/connectors/agent_runs.py:102`, `runtime/agent_evidence.py:80` | an owner decision |
| What do real tenant exports hold: OTel tool spans with `gen_ai.prompt`, credentials as name/value pairs? Decides L-12, L-19. | agentenv tests; probe `p_ta4`; the gate store is too small | a scan of tenant inputs |
| How do MiMo's uni-agent gateway and mimoagent anti-hack record retries, lineage and probes? | README and configs | submodules uninitialized; fetching is forbidden here |
| Does the proxy's single `httpx.AsyncClient` degrade at RL concurrency? | A NeMo-Gym comment (`nvidia-nemo-gym/nemo_gym/anthropic_converter.py:28-31`) | a load probe |

## 10. Handoff: findings outside this module

| To | Finding |
|---|---|
| M10 runtime (Harbor harness) | The harness reads only `logs/trajectory.json` (`interop/harbor/agent_harness.py:900-904`), so Terminus-2 continuation and summarizer files (harbor-framework-harbor/src/harbor/agents/terminus_2/terminus_2.py:1871-1936,2056-2089) would reach reconcile as segment 0 only. Check also whether `endpoint_env` capture keys (`agent_harness.py:275-299,581-585`) can reach a verifier env; in OpenEnv this broke an LLM judge (meta-pytorch-openenv/src/openenv/harbor/seams.py:340-345). |
| M10 runtime (evidence) | `verdict_of` could read a legacy `CancelledError` row as `client_left` (`runtime/agent_evidence.py:141-146`). After R-2, check `classify_agent_end` and reconcile against client-left rows with completion tokens (`interop/harbor/reconcile.py:23-24`). M10 scores the limit exits (`agent_evidence.py:79-80`), while M3 and M4 mask them: one exit vocabulary would stop the drift. |
| M10 / M6 (reconcile) | Reconcile joins by order (`interop/harbor/reconcile.py:205-237`); NeMo-Gym joins by id witnesses (nvidia-nemo-gym/nemo_gym/token_id_capture/terminal.py:16-59); no failure was established. An empty 200 forces counts-only for the whole session (`reconcile.py:168-177,220`): exclude no-output calls, or adopt L-2's predicate. |
| M10 runtime (harness env) | verifiers suppresses Claude Code's title call and nonessential traffic (primeintellect-ai-verifiers/verifiers/v1/harnesses/claude_code/harness.py:82-86,97). Crucible sets neither, so side calls count against `max_model_calls`. |
| M10 / M0 (env knobs) | Harbor's Claude Code converter imports litellm (harbor-framework-harbor/src/harbor/agents/installed/claude_code.py:1036-1046), which fetches a remote price map at import (seen in 1.106) unless `LITELLM_LOCAL_MODEL_COST_MAP=True`. One env default would avoid an outbound GET during offline ingest. |
| M0 core (store) | `put_doc` inserts `doc_names` with `INSERT OR IGNORE`, and `tag` keeps an existing note (`core/store/__init__.py:458-463,557-559`), so every provenance or note stamp is first-write-wins. L-14 needs `tag()` to record a same-key move whose note changed. |
| M1 world (agentenv backend) | The agentenv gateway trajectory and log stay unscrubbed in a caller-given workdir at the process umask (`interop/agentenv/world_backend.py:220-222,299-301,608-611`; agentenv-framework/src/agent_env/env/gateway/gateway.py:121-123). |
| M4 signals | `AgentRunSignalMapper` emits -1 per failing gate with no exit check (`coding/signals.py:229-243`); R-3 closes this, otherwise it is an M4 item. IMP-M4-5's reference is still at HEAD (huggingface-repo2rlenv/src/repo2rlenv/emitter/evaluation.py:30-58). Low priority: a tool call written as text could be a signal (infra-nvidia-nemo-rl/nemo_rl/environments/nemo_gym.py:369-433). |
| M5 workflow | prime-agent's `golden_replay.rs` is a test-only parity corpus, not a runtime fallback (primeintellect-ai-prime-agent/crates/pa-core/src/tools/golden_replay.rs:1-9). slime's trained-once rule and AReaL's `clone_chain` are at HEAD (infra-thudm-slime/slime/agent/trajectory.py:532-553; infra-inclusionai-areal/areal/experimental/openai/cache.py:120-174). |
| M6 credit / interop | The A2A harness has no ATIF cross-check, so an undelivered captured call stays trainable (`interop/agentenv/harness.py:127-170`). After R-2, the trainer should mask client-left calls. |
| M7 verify / M8 forge | Hidden evaluator fields are masked by key name, and metrics named like them are refused (primeintellect-ai-research-environments/environments/office/ale_cli/ale_cli/evaluation.py:13,43-80). Sentinel-bracketed test output scores 0 when the marker is missing (.../swe/swerebench_v2/swerebench_v2/taskset.py:236-241). |
| M9 accept | Golden-patch k-pass sorts tasks into supported, broken, flaky and inconclusive; inconclusive covers runs where fewer verdicts completed (nvidia-nemo-gym/resources_servers/swemer_oml/aggregate_golden_patch.py:51-64). Crucible's counterpart was not checked. |
| M11 train | When Crucible drives TRL (`train/bridges/trl_bridge.py`), force off `log_completions`, its Hub push and tracker tables (infra-huggingface-trl/trl/trainer/grpo_trainer.py:1179-1200,3252-3290). checkpoint-engine has no M3 content. |
| M12 train | verl's rollout tracing sends unredacted prompts to Weave/MLflow/Trackio and dumps at the default umask (infra-volcengine-verl/verl/utils/rollout_trace.py:89-111,364-427): pin it off. References for `ARCHITECTURE.md:268` token capture: NeMo-Gym digest-verified parents (nvidia-nemo-gym/nemo_gym/token_id_capture/records.py:87-189) and rLLM cumulative mode (infra-rllm-org-rllm/rllm-model-gateway/src/rllm_model_gateway/token_accumulator.py:53-153). |
| M15 label | `recorded_outcome` grades a run without checking its exit status (`coding/studio.py:158-175`). It should return None when the M3 trace says the harness or provider ended the run (R-3). |

## 11. Sources

- **Repos:** the 33 folders of section 4 at the commits listed there. Where the earlier sweep's commit differs: verifiers
  `ae819de`, research-environments `d4fb80a`, NeMo-Gym `3ef478d`, OpenEnv `49aa302`, repo2rlenv `5555443`, prime-agent
  `2ee7e623a`, verl `5eb4826`, prime-rl `f7e8ab6`, NeMo-RL `c9193bc`, TRL `72cf37d`, AReaL `298412a`, slime `c933787`, SkyRL
  `3d0ae22`, ROLL `581046a`. Pins absent locally: `55347f62e2`, `6cad8710bb`, `dca5562ee6`, `4330be260a`, `e42f222fcf`,
  `1f555b97a1`, `7cbffa77b`, `859938d`.
- **Crucible docs:** `docs/modules/M03-connectors.md`; `docs/audit/2026-10-12/M03-connectors.md`,
  `M03-study-{capture,logs-repos,redaction}.md`; `docs/research/improvements/M03-M05.md`, `M00-M02.md`,
  `2026-10-14/{M00-core,M01-world}.md`, `2026-10-15/M02-actors.md`, `2026-10-16/M00-core.md`; `ARCHITECTURE.md` (§1-§3, M0
  and M3 cards, §7); `DESIGN.md` §3; `THIRD_PARTY.md`; `docs/modules/reports/m3_gate.json`.
- **Crucible code:** `connectors/*`; `core/{agent_events,secrets,text,canonical,prov,fingerprint,config,registry}.py`,
  `core/schema/trace.py`, `core/store/__init__.py`; `interop/harbor/{atif_import,agent_harness,reconcile}.py`,
  `interop/agentenv/{harness,agents,world_backend}.py`; `runtime/{agent_evidence,record,run}.py`; `workflow/steps.py`;
  `studio/label.py`; `world/guard/guard.py`; `coding/connectors/*`,
  `coding/{signals,git_history,github_reviews,bench_recorded,studio}.py`, `coding/{m3,m4}/gate.py`,
  `coding/forge/derive.py`, `coding/runtime/r3.py`.
- **Probes (`probes/`, outputs in `notes/`):** scout and ledger `p00_scout`, `p10_ledger`, `p11_atif`; sweeps
  `p20_agentenv`, `p30_mimo_verl`, `p40_harbor`, `p41_harbor_cc`, `p50_prime`, `p60_areal_slime`,
  `p61_nonstream_disconnect`, `p_oe1..3`, `p_swe_safety(_real)`, `nemo/p_nemo_01..06`, `trainers_a/p_ta1..4`,
  `trainers_c/p_tc1..6`; verify `v1_client_left`, `verify2/v2_*`, `p_v3_verify`, `verify4/v4_*`. Tests run: the 80
  `test_m3_audit*.py` (all passed), 119 M3 unit tests, 26 cross-module tests.
- **Work folder:** `/Users/uzair/Desktop/RL env repos/.module-audit/2026-10-20/M03-connectors` (`01-brief.md`,
  `02-ledger.md`, `sweep-*.md`, `verify-1..4.md`, `notes/`, `probes/`).
