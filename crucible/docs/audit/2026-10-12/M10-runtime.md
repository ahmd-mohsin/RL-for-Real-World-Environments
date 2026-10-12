# M10 runtime: audit (2026-10-12)

Scope: `src/crucible/runtime/` (run, infra, budget, journal, checkpoint, collect, remote, gpu, control task, harnesses
native/workspace/imported), the episode loop it drives (`actors/policy.run_episode`, `LLMToolPolicy` compaction), the
LLM client's error text, the local executor's process sessions (`world/exec.py`, `world/apps/workspace.py`), the
failure taxonomy (`core/schema/run.py`) and a new `core/secrets.py`.

Checked against ARCHITECTURE §M10, DESIGN §10 (P16) and Appendix B.5, and `docs/modules/M10-runtime.md`, with the M0
protocol. Framework comparison: `M08-M10-study-sweep.md` (M10 rows, found independently by all five sweeps). Defect
hunt: `M10-study-defects.md` (25 ledger rows, V1-V17). 18 regression tests (`tests/unit/test_m10_audit.py`) all fail on
the M9 audit commit (`6efff7a`) and pass after the fixes.

## 1. Requirement ledger (rows that were wrong, missing or partial)

| # | Requirement | Before | After |
|---|---|---|---|
| L2 | Policy failures are scored, never masked (P16) | **wrong**: junk in `.git/HEAD`, a corrupt index or a `chmod 000` file turned a do-nothing policy's 0 into a mask; a policy exception was an unknown-blame mask; background load masked its perf grade | a broken repository (status fails, base commit gone) is `agent_broke_env` (scored 0); files the agent made unreadable are its work product (tolerant collect parses git's per-path errors); a policy's own data error is `crucible:policy_error` (scored 0); its background jobs are killed with its World, before grading |
| L3 | Blame attribution | unmapped names had harness blame; pack kinds were renamed `agent_run_error`; socket timeouts were scored as episode timeouts | registered kinds keep their name and blame; unmapped foreign names have `unknown` blame (counted by the stop rule); `TimeoutError`/`ConnectionError`/`OSError` from a policy are transport (masked); only the episode deadline is `timeout` |
| L4 | Retries never re-run the agent | **wrong**: the third attempt after a re-collect re-ran the agent; train-mode re-grades were masked `capture_incomplete`, then re-run; the re-collect diffed against the movable `base` tag | the reusable work is found by looking back through attempts; the base is the one the agent-running attempt recorded (a missing record is non-retryable infra, never the tag); re-grades and re-collects inherit that attempt's model calls |
| L5 | Budget | reservations of 0 let concurrent attempts spend 3x the cap; uncertain spend never stopped the run; unsent requests were never released | an unpriced policy holds its share of the cap and at least the dearest settled attempt; uncertain spend above tolerance stops the run; `not_sent` failures release their reservation. A declared `estimate_usd` keeps the cap exactly; without one the first wave can overrun by concurrency x (cost - share) |
| L6 | Checkpoint/resume | worked once; a second crash opened the wrong store; a process crash never resumed; restored calls were charged twice; masked attempts charged 0; multi-step tasks reset on resume | the checkpoint names the attempt whose `work/` holds the World; a crash with no record resumes from `aborted/`; only new calls are charged; masked attempts charge their calls; multi-step tasks are not checkpointed |
| L7 | Journal crash safety | a torn tail swallowed the next row | the tail is cut before appending (NeMo-Gym); the forge and Curator journals skip torn lines too (M8) |
| L8 | Harness exceptions contained | **wrong**: one exception aborted `Runner.run` with reservations open | each attempt is contained: a harness, record or rollout exception is a masked harness failure with its spend held as uncertain; an optional per-attempt deadline (`attempt_timeout_s`); retries back off outside the concurrency slot |
| L11 | Perf guard | graded before the policy's World closed | the World is closed (and its sessions killed) first |
| L12, L13 | GPU preflight and control channel | any one GPU matching passed; rsync failures and `$HOME` failures were ignored; the self-match lint missed `pgrep -af`/`--full` and flagged the safe form; ssh rc 255 read as a self-kill | every GPU must match; staging failures raise; the lint parses flag clusters and bracket guards; ssh 255 is `transport_unreachable`. Still no production caller (L12, L13 partial) |
| L19 | No secrets in records | **wrong**: a credential echoed by a provider reached 14 files | `LLMError` scrubs its text at the source; `Failure.reason`, journal rows, `record.json`, `trace.json`, `calls.json` and trajectories are scrubbed structurally (string values only; `core/secrets.py`, the M3 patterns moved down one layer) |
| L20 | Control task reward | `score(nan) = 1.0` | a non-finite return is a masked `verifier_error` |
| L25 | Imported harness | a missing patch was infra, retried 3 times; records stamped wall time | a missing patch is `crucible:harness_unsupported` (not retried); no wall-clock stamp |
| L21 | `harness/harbor.py` | **missing** | still missing |
| L22 | Exit gate (a): a historical run reproduced end to end | partial (R1 is a re-grade) | unchanged |

## 2. Defects and fixes

| Id | Sev | Fix (files) |
|---|---|---|
| V1 | P0 | `runtime/collect.py` (`classify` with changed paths and git's per-path errors, `repo_broken`), `runtime/harness/workspace.py` |
| V5 | P0 | `runtime/harness/workspace.py` (close before grading), `world/exec.py` (`LocalExec.sessions`, `kill_sessions`), `world/apps/workspace.py` (`stop`) |
| V12 | P0 | `core/secrets.py` (new: patterns from `connectors/redact.py`, `scrub_secrets`, `scrub_obj`, `scrub_json`), `actors/llm.py`, `core/schema/run.py`, `runtime/journal.py`, `runtime/harness/native.py` |
| V2, V3 | P1 | `runtime/harness/workspace.py` (`_prior` looks back, `_inherit_calls`, base from the agent-running attempt) |
| V4 | P1 | `runtime/run.py` (reservations, uncertain-spend stop, release) |
| V6, V7 | P1 | `actors/policy.py` (act wrapper, compaction argument), harnesses score `crucible:policy_error`; `agent_request_failed` non-retryable |
| V8, V9, V14 | P1, P2 | `runtime/checkpoint.py` (`work_attempt`, `attempt_path`), `runtime/harness/native.py` |
| V10 | P2 | `runtime/journal.py` |
| V11 | P1 | `runtime/run.py` (`attempt_once`, deadline, backoff) |
| V13 | P1 | `runtime/control_task.py` |
| V15 | P2 | `runtime/remote.py` |
| V17 | P2 | `runtime/gpu.py`, `runtime/harness/imported.py` |
| sweeps | — | taxonomy: registered kinds kept (G3 D5), unmapped → unknown (G5), `harness_unsupported` registered (G4 D2), non-retryable kinds and masked limit stops (G2 D3, G3 D6) |
| V16 | P2 | not fixed: the derived seed still does not reach the endpoint; a retried sample misses the replay cache |
| — | — | not fixed: the journal does not rebuild terminal rows from `record.json` after a crash between the two writes (that sample re-runs once) |

## 3. Gates

- B.5 injections (CPU, `tests/unit/test_runtime.py`): pass. Stale-image and stale-build injections are now masked
  once and not retried (a retry cannot rebuild an image); contention is still re-graded in a fresh attempt.
- Long-horizon xl gate (`crucible_pack_toy.longhorizon.gate`): passes. The report changed in two values:
  `attempt1.steps` 0 → 3495 (the crashed attempt now records how far it got, because the crash is classified inside the
  episode) and `max_context_before_compaction` (102,853 → 102,652; the same drift the hunt saw before any change).
- The hunt's probes: every P0/P1 case now behaves as specified (p1, p2, p4, p5, p7, p8, p9, p10; p6 cost: a1 charges its
  150 calls, a2 only its 156 new ones; a killed process resumes from `aborted/a1@step100` and scores 1.0).
- R1-R3 and the docker B.5 classes were not re-run (lab GPU, Harbor, paid).
- `crucible.runtime` over the full suite: **90%** (86% in the hunt; `checkpoint.py` and `collect.py` 100%, `run.py` 96%,
  `native.py` 94%; still thin: `harness/imported.py` 0% (R1 only), `gpu.py` 55%, `remote.py` 74%).
- Suite (`tests/unit`, `tests/toy_e2e`): **1050 passed, 3 skipped** (1032 before, plus the 18 new tests).
  `lint-imports`: 4 kept; `ruff` clean.

## 4. Picks

Built: containment, deadline and backoff (agentenv `_run_safe`, rLLM, prime-rl, verifiers); re-collect from the
recorded base (our M7 fix completed); taxonomy fixes (Harbor, NeMo-Gym, prime-agent); journal torn-tail cut (NeMo-Gym);
secret scrubbing at the boundary (our M3 patterns); crash resume (prime-agent append-only history). Queued: sandbox
hygiene (liveness, start-failure cap, TTL, orphan sweep; rLLM WarmQueue, ROLL, agentenv `teardown_run`),
`harness/harbor.py` and a verifiers harness, Harbor's agent error classes mapped, single-writer journal lease, the
seed to the endpoint (V16).
