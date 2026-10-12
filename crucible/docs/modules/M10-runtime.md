# M10 `runtime` — running policies reliably

**Status: core built (2026-10-09); audited 2026-10-12 (`docs/audit/2026-10-12/M10-runtime.md`); exit gate in progress.**
- **Audit:** 17 reproduced defects fixed. A policy could turn its own failure into a mask (a broken `.git`, an
  unreadable file, a background job during perf grading, a raised exception); a credential echoed by a provider reached
  14 record and trace files; one harness exception aborted a run; retries re-ran the agent; resume worked only once and
  charged restored calls twice.
- **B.5 injections ✅:** all 8 harness and infra classes are injected and masked (CPU in `tests/unit/test_runtime.py`, Linux-container classes in `tests/redteam/test_runtime_docker.py`).
- **Policy controls ✅:** they are scored, not masked.
- **Historical run reproduction ⏳:** R1 (re-grade), R2 (replay), R3 (live mini-swe-agent through Harbor) are pending.

- **Code:** `src/crucible/runtime/`:
  - `keys.py`, `infra.py`, `journal.py`, `budget.py`, `run.py`, `collect.py`, `remote.py`, `gpu.py`, `perf_guard.py`, `trajectories.py`, `record.py`, `inject.py`;
  - `core/schema/rollout.py` (the `crucible.rollout/v1` schema, owned by M12);
  - `harness/` (`base.py`, `native.py`, `workspace.py`);
  - `verify/stage.py` (docker grading hygiene);
  - toy pack `runtime.py` (`HelpdeskTask`, `FibTask`, scripted and LLM policies).
- **Research:** `docs/research/m9-m10/02-runtime.md`.

## What it is

| Piece | What | Built on |
|---|---|---|
| `keys.py` | `SampleKey(task, world, policy, harness, sample_idx, seed, tier)`. Seeds are derived from the run seed and the key; attempt ids are `<sample>-a<N>`; the run identity is a hash of the spec minus cosmetic fields | FineEnvs identity |
| `infra.py` | Failure normalization at every boundary (Harbor exception names, MiMo categories, provider errors) into kinds with a **blame**. `RetryRule` never retries policy blame; `LimitPolicy` turns limits into scored stops; masked-aware `mean_reward`, `coverage`, unbiased grouped `pass_at_k`; `Health` stop rules; `classify_verifier` | NeMo-Gym `failure_kinds`/`rollout_health`; verifiers errors and `RetryRule`; Karotte limits |
| `journal.py` | `run.json` identity (a mismatch refuses to resume), fsynced `journal.jsonl`, `results.jsonl` (terminal), `failures.jsonl` (masked, never read by metrics), atomic `record.json` per attempt; aborted attempt dirs are kept as evidence | Repo2RLEnv `EventJournal`; NeMo-Gym sidecars, torn-tail repair |
| `budget.py` | SQLite reserve/settle ledger with caps; reservations left by a crash become *uncertain* spend; uncertain spend has its own tolerance | Repo2RLEnv `campaigns/budget.py` |
| `run.py` | `RunSpec` → plan → bounded-concurrency dispatch → reserve → harness → settle → record → retry. Credits exhausted → stop, resumable | agentenv `eval run --k` (shape) |
| `harness/native.py` | A World episode with an in-process policy: masked episode → failure; `task.broken()` → `crucible:agent_broke_env` (policy); limits → truncated and scored; grading → `classify_verifier`; cleanup never overrides a result | — |
| `harness/workspace.py` | Collects the agent's work as a patch (tolerant or legacy hook). An empty patch from a changed tree → `crucible:empty_collection`, workspace preserved; the retry re-collects only | our `run_agent.sh` collect fix |
| `collect.py` | `collect`, `tree_changed`, `classify` | — |
| `remote.py` | Sentinel control channel (`crucible:harness_self_kill`), pidfile kills, self-matching `pkill`/`pgrep` lint | B.5 row 8 |
| `gpu.py` | `GpuHost` (`--runtime=nvidia`), `preflight` (PID 1 init check → `crucible:zombie_wait`; GPU arch) | B.5 row 4 |
| `verify/stage.py` `docker_grade` | Pins the image digest (`expected_image` mismatch → masked `stale image` before grading); named, labelled container; timeout → `docker rm -f` + masked `verifier_timeout`; no reward file → masked `no_reward_json` | Harbor verifier behaviour |
| `perf_guard.py` | One perf grade per host (lock); host check before the grade (foreign GPU compute memory, GPU utilization, load per core); waits, then `node_contention` (masked) if never quiet | grader-side `LoadGuard`; Harbor regrade |
| `trajectories.py` | ATIF v1.8 per actor joined to model calls (`call_id`, `llm_call_count`, per-call `Metrics` with token ids/logprobs), `no_action` steps for blank calls, `FinalMetrics`; `align` sets `StepRow.atif_step_id`/`call_id`; `cross_check` against a proxy capture | Harbor ATIF models (DEPEND); OpenEnv proxy-vs-ATIF check |
| `record.py` + `core/schema/rollout.py` | `crucible.rollout/v1` per attempt (`rollout.json`): turns with token ids, behaviour logprobs, policy version, sampling, credit; `valid_for_training`; train mode masks incomplete token records as `crucible:capture_incomplete` | verl `TokenTrace` shape with per-span versions |
| `Runner.regrade` + grade-only retries | A masked grade is retried by re-grading the stored patch (no agent re-run); exhausted samples are re-graded later from artifacts | Harbor regrade-from-artifacts |
| `interop/harbor/verify_dir.py` | `HarborDirTask`: a Harbor task dir (optionally at a git ref) graded with its own verifier command in the pinned image on any Docker host; Harbor reward files; masked on no reward, timeout, stale image | Harbor task layout (DEPEND) |
| `harness/imported.py` | A recorded run's patch as an attempt (R1, dataset submissions) | — |
| `inject.py` | `FakeLLM` with scripted provider failures (blank usage, credits, context overflow), `FakeSpend`, `FlakyGradeTask`; toy pack `crashing_tool_world` | — |

## Exit gate

| Requirement (ARCHITECTURE M10) | Result |
|---|---|
| All 8 B.5 classes injected, each classified as infra and masked, never scored 0 | ✅ 8/8 classes:<br>• **empty patch:** `empty_collection`, then re-collect only;<br>• **blank usage:** masked, retried;<br>• **credits:** stop, resumable, resume finishes the run;<br>• **zombie PID 1:** `zombie_wait` preflight, container without `--init` vs with it;<br>• **stale image:** digest pin, masked before grading;<br>• **ABI / stale build:** `stale_build` from the verifier;<br>• **contention:** `node_contention` (time spread, and GPU OOM under foreign memory);<br>• **self-kill:** sentinel channel on macOS and over a Linux container.<br>Also masked: tool crash (environment bug), grader exception, verifier timeout. Controls scored: agent broke the environment, step limit, context overflow, no change (0) |
| Masked ≠ 0 | ✅ with 1 of 4 samples masked, the mean is that of the 3 scored samples; the Harbor-style average (masked as 0) is shown to differ |
| Resume | ✅ re-running a finished spec dispatches nothing; a different spec under the same run id is refused |
| One historical frontier run reproduced end to end | **R1 ✅** (`reports/m10_r1.json`): the GPT 5.6 sol task02 patch, imported and re-graded at task02@5e53057 with the image digest pinned, on the lab over `DOCKER_HOST=ssh://` inside a perf-guard slot. Gate profile identical to the recording (static ✓ install ✓ ops ✗ hard ✗ app ✓ perf ✗), the same 6 failing ops test ids, reward 0.0, one attempt, not masked. ⏳ R2 command replay, R3 live mini-swe-agent through Harbor (~$3–10) |
| Toy | ✅ scripted oracle 1.0 and no-op 0.0 over k samples; workspace `fib` task 1.0 / 0.0 |

| Trajectories and rollout records | ✅ every call, World action, ATIF step and `steps.jsonl` row joins (`tests/unit/test_runtime_records.py`); Harbor's validator accepts the trajectories; train mode masks a rollout without token records, eval mode does not need them |

## Remaining
- `harness/harbor.py` (installed agents through Harbor `Trial`, with the capture proxy as the model endpoint).
- GPU injections on the lab (burner process for contention, stale build, stale image); R1–R3.
