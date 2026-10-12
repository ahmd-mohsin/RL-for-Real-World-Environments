# M8 / M9 / M10 framework sweep, group 1: Harbor and the SWE and task-building ecosystem (2026-10-12)

Read-only, Crucible HEAD `526fdb5`. One reader. Every external file:line below was opened, except rows marked
"(prior)", which repeat a citation from an earlier study that this sweep did not reopen. Eight probes reproduce the
defects marked [ran]; they live in `scratchpad/m810sweep/g1/` and run with `crucible/.venv/bin/python` from `crucible/`.

**Moving target.** While this sweep ran, an uncommitted M8 audit edited `CR/forge/budget.py`, `CR/forge/orchestrator.py`,
`CR/interop/harbor/role_runner.py`, `CR/forge/repo.py` and `CR/actors/llm.py` in the working tree. Its diff already
targets parts of D4 (`settle` no longer raises; delta costs are clamped to the observed total, "V3/V4") and D13 (a
failed role trial now carries its cost, "V19"). D3 (forge part), D4 and D13 below describe HEAD `526fdb5`, which is
what P3 and P4 ran against. Re-run P3 and P4 on the committed audit before closing them.

Roots and pins:

| Root | Repo | Commit | Licence |
|---|---|---|---|
| `H` | `RL-Environments/06-frameworks-and-env-code/harbor-framework-harbor/src/harbor` | a2fafbc | Apache-2.0 |
| `R2` | `.../huggingface-repo2rlenv/src/repo2rlenv` | 5555443 | Apache-2.0 |
| `SS` | `.../swe-bench-swe-smith` | 9b74ac0 | MIT |
| `MS` | `.../swe-agent-mini-swe-agent/src/minisweagent` | 04d809c | MIT |
| `FE` | `.../adithya-s-k-fineenvs` | 5e0b46e | Apache-2.0 (PortSim data CC BY-SA 4.0) |
| `SP1` `SP2` `SP3` | the three HF Spaces (simulation, multi-harness, guide) | 46bb2be, b780039, ead51a4 | CC BY 4.0 (articles; no reusable code) |
| `OE` | `.../meta-pytorch-openenv` | 49aa302 | BSD-3-Clause |
| `CR` | `crucible/src/crucible` | 526fdb5 | ours |

Verdicts: **DEPEND** (call their code), **PORT** (copy; licence permits), **PATTERN** (rebuild the idea), **nothing**
(Crucible is equal or ahead), **anti-pattern** (do not copy; often a red-team or import fixture).

## 0. Corrections and confirmations of the prior studies

Confirmed still open:
- **Harness exceptions are not contained** (`docs/research/improvements/M09-M11.md:38-39`). Still true [ran P1]:
  `CR/runtime/run.py:139` awaits `harness.run` with no guard, and `:159` gathers without `return_exceptions`.
- **Retries are immediate; one semaphore covers every phase and is held across retries** (`M09-M11.md:41-42`). Still
  true: `CR/runtime/run.py:118-157`.
- **`run.json` has no code pins** (`M09-M11.md:43-44`). Still true: `CR/runtime/run.py:53-56` hashes ids only,
  though `CR/runtime/journal.py:6` promises "code/framework pins".
- **`over_uncertain_tolerance` and `release()` are never called** (`M09-M11.md:49`). Still true (grep of `src/`).
- **M9 `imports/` I0–I10 and M10 `harness/harbor.py` are absent** (`M09-M11.md:12-13`). Still true. Only the import
  defect fixtures exist (`CR/accept/registry/fixtures.py:213-256`).
- **Grade-store writes are not atomic** (`M09-M11.md:33`, `CR/accept/checks/base.py:107-110`). Still true, and worse
  than stated: a torn file is never re-graded [ran P8].
- **The forge should move onto the ported ledger** (`docs/research/m9-m10/02-runtime.md:676`). Not done.
  `CR/forge/budget.py:19-41` is still an in-memory `Ledger` with no persistence and no uncertain state [ran P4].
- **Patch mutation operators for coding tasks** (`docs/research/improvements/M06-M08.md` M8-1; `M09-M11.md:25`).
  Still absent: `CR/accept/mutation/actions.py:29-68` mutates World action lists only.
- **Repo2RLEnv `Review` / `Citation` / probe outcomes as the Auditor schema** (`docs/research/m7-m8/02-forge-synthesis.md:503-512`).
  Not ported. `CR/forge/schema.py:124-129` keeps `findings` with a `blocking` flag and nothing else. A3 gained valid
  alternatives (`CR/accept/checks/core.py:100-103,119-120`), but "an uninstalled probe is a failure, not a kill"
  was not taken (`core.py:132-134` counts any reward 0 as killed).

Corrections:
- **Repo2RLEnv `EventJournal` does not "tolerate a torn tail"** in the sense Crucible ported
  (`02-runtime.md:670`; `CR/runtime/journal.py:14,45-46`). Its `read()` drops the fragment
  (`R2/campaigns/events.py:46-53`), but `emit()` **refuses to append** after a torn tail (`events.py:35-40`).
  Crucible kept the tolerant read and appends anyway, so the next record joins the fragment and is lost [ran P3].
- **Crucible's runtime ledger is not a full PORT of Repo2RLEnv's.** `CR/runtime/budget.py:1` says PORT, but it drops
  duplicate-operation refusal (`R2/campaigns/budget.py:151-152` vs `CR/runtime/budget.py:53` `INSERT OR REPLACE`),
  evidence on settle (`R2 :226-245`), the persisted limit with an authorised increase (`R2 :84-106,184-212` vs
  `CR :30,66-67`, caps in memory), `BEGIN IMMEDIATE` (`R2 :108-120`), integer micro-dollars (`R2 :16,72-76`), and the
  metered-call receipts (`R2/campaigns/llm.py:17-100`).
- **OpenEnv AWM is not a world synthesizer** (ARCHITECTURE M8 "PATTERN: OpenEnv AWM (world synthesis)"). OpenEnv's
  `agent_world_model_env` downloads the pre-built AWM-1K dataset (`OE/envs/agent_world_model_env/server/data_loader.py:16-21,62-66`;
  `README.md:16`). It has no generation code. Crucible already uses it as data (`CR/interop/openenv/awm.py:1-9`). The
  synthesis pattern has to come from elsewhere (FineEnvs PortSim below is the best local example).
- **Harbor `compile` is not wired.** `02-forge-synthesis.md:414` recommends `CR/interop/harbor/compile_bridge.py`; the
  file does not exist. Forge roles do run as Harbor wrapper tasks (`CR/interop/harbor/role_runner.py:85-233`), which
  is the part of that recommendation that was built.
- **A11 does not run "the task's own runtime path"** (`docs/modules/M09-accept.md:35`). It always uses
  `NativeHarness()` (`CR/accept/checks/extended.py:166`). A workspace or imported task's collect path is never
  exercised. A11 also claims detectors `B.5-empty-patch` and `B.5-zombie` (`extended.py:136`) but injects neither;
  both are per-kit unit tests (`accept/registry/defects/B.5-empty-patch.yaml`, `B.5-zombie.yaml`).
- **`gpu.preflight` and `remote.control` have no production caller.** `B.5-zombie` is "fixed" by `--init + preflight`
  (`defects/B.5-zombie.yaml`). `--init` is applied (`CR/world/sandbox.py:87`, `CR/verify/stage.py:111`), but
  `preflight` (`CR/runtime/gpu.py:75-84`) is called only by `tests/redteam/test_runtime_docker.py:48-55`; `ssh.py:9`
  mentions it in a docstring only. `control`, `kill_by_pidfile` and `lint_self_matching` (`CR/runtime/remote.py`) are
  called only by tests.
- **"Harbor imported outside M11"** (`M09-M11.md:5-9`) is fixed: `CR/runtime/run.py:208` loads ATIF through
  `load_impl`, and `CR/runtime/record.py:20-21` imports Harbor under `TYPE_CHECKING` only.

## 1. Mechanism tables

### 1.1 Harbor (`H`)

| Mod | Mechanism | Source file:line | Crucible (file:line) | Verdict |
|---|---|---|---|---|
| M8 | `compile`: instructions × environments × verifiers cross product; auto-verifier checks artifacts and schemas only; deletes an existing output dir | `H/compile/compiler.py:52-95` (prior) | none (`compile_bridge.py` absent) | PATTERN (role wrappers only) |
| M8 | Role or check as a sandboxed wrapper task: reviewed task under `environment/`, deliverable JSON, `validate.py` gives reward 1 for a well-formed answer | `H/analyze/checker.py:144-207` | `CR/interop/harbor/role_runner.py:85-150` | nothing (ported) |
| M8/M9 | Adapter acceptance: oracle 100% before a WIP PR, then parity runs (mean ± SEM, same agent and model on both sides) in `parity_experiment.json`; a human process | `H/cli/adapter_review.py:32-44` (checklist text) | none (imports unbuilt) | PATTERN: record parity as provenance for imports, never as acceptance |
| M9 | `harbor check`: an installed agent reviews the whole task dir (solution and tests visible) against an 11-criterion rubric; outcomes pass / fail / not_applicable; a trial exception raises, an invalid deliverable raises `ValueError` | `H/analyze/checker.py:40-113,287-319`; `H/cli/quality_checker/default-rubric.toml:1-54` | A7 is structural only: check `states` ⊆ property ids, first 40 chars of each property in the instruction (`CR/accept/checks/core.py:225-239`) | DEPEND as advisory A7b (pick M9-6) |
| M9 | Oracle agent: a non-zero `solve.sh` exit only writes `exit-code.txt`; the verifier still runs | `H/agents/oracle.py:151-153` | A1 gold check: the oracle must change the state (`CR/accept/checks/core.py:33-35`) | nothing (Crucible ahead; import fixture `HF.D1`) |
| M9 | `harbor task check` / `task debug` removed; message points to `harbor check` / `harbor analyze` | `H/cli/tasks.py:505-537` | n/a | nothing (docs older than code may still cite them) |
| M10 | Typed installed-agent errors: `ApiRateLimitError`, `ApiUsageLimitError`, `ApiInternalServerError`, `ApiOverloadedError`, `ApiConnectionClosedError`, `ApiResponseStalledError`, `OutputTokenExceededError`, `ContextWindowExceededError`, `UnknownApiError`, `AgentSafetyRefusalError`, `AgentAuthenticationError`, `ModelNotFoundError`, `NetworkConnectionError` | `H/agents/installed/base.py:34-160` | `CR/runtime/infra.py:27-43` maps 2 of them; the rest become `agent_run_error`, harness blame, masked and retried [ran P7] | PORT the name map (pick M10-3) |
| M10 | Classification by regex over the agent's stdout+stderr, last match wins; soft refusals skipped | `H/agents/installed/base.py:487-583,912-950` | none | PATTERN, with a caveat: the output includes tool output the policy controls, so a printed "usage limit" string could become infra |
| M10 | Agent timeout and agent errors are recorded, then collection and the verifier still run: a trial can carry both a reward and `exception_info` | `H/trial/single_step.py:38-55,75-87,89-114` | no Harbor harness yet | anti-pattern to guard: classify by `exception_info` first |
| M10 | `RetryConfig`: `max_retries=0` by default; exclude list (timeouts, reward-file errors, usage limit, refusal, auth, model not found); backoff `min·mult^n`, no jitter | `H/models/job/config.py:291-322`; `H/trial/queue.py:83-110,194-235` | `CR/runtime/infra.py:87-96` (blame-based, no backoff); `CR/runtime/run.py:154-156` | PATTERN (pick M10-5) |
| M10 | A retry deletes the failed trial dir | `H/trial/queue.py:222` | every attempt kept (`CR/runtime/journal.py:98-108`) | anti-pattern |
| M10 | A cleanup failure is recorded as the trial's exception when none was set; with retries on, a graded trial is deleted and re-run | `H/trial/trial.py:545-566,1860-1881`; `H/trial/queue.py:205-222` | "cleanup never overrides a result" (`CR/runtime/harness/native.py:291-297`) | anti-pattern (Crucible ahead) |
| M10 | Phase-scoped concurrency: a per-agent pool acquired at `AGENT_START`, released at `AGENT_END`, plus a global trial semaphore | `H/trial/queue.py:45,114-120,131-192` | one semaphore over all phases and retries (`CR/runtime/run.py:101,118`) | PATTERN (pick M10-5) |
| M10 | Resume: refuse a changed config; delete trial dirs without `result.json`; keep errored trials unless their type is listed (default only `CancelledError`) | `H/job.py:258-340`; `H/cli/jobs.py:2294-2312,2422-2468` | identity refusal, aborted dirs kept, masked samples retried (`CR/runtime/journal.py:64-68,98-108`; `run.py:143-157`) | nothing (Crucible ahead) |
| M10 | `lock.json`: task digests, Harbor version and git hash; a mismatched lock refuses the job dir | `H/models/job/lock.py:56-170`; `H/job.py:909-941` | none | PATTERN (pick M10-6) |
| M10 | pass@k: a trial with no reward counts 0; returns nothing unless every reward is a single 0/1 key; k from the smallest group | `H/utils/pass_at_k.py:32-94` | masked samples excluded (`CR/runtime/infra.py:132-144`) | anti-pattern; but see D11 |
| M10 | `Mean`: missing reward = 0; the docs say so | `H/metrics/base.py:14-35`; `docs-mintlify/datasets/metrics.mdx:6` | `CR/runtime/infra.py:123-125` | nothing (export side already handled in M7 audit) |

### 1.2 Repo2RLEnv (`R2`)

| Mod | Mechanism | Source file:line | Crucible (file:line) | Verdict |
|---|---|---|---|---|
| M8 | Mutation task gen: the healthy suite must pass first; each candidate runs in the pinned image; kept only with an execution contrast; per-entity cap; seeded shuffle | `R2/pipelines/recipes/swe_smith/worker.py:46-55,57-103` | none for code | PORT (pick M8-2) |
| M8 | Execution contrast: test collection must be unchanged, the defective run must exit 1, and some healthy-passing test must fail; F2P/P2P derived from it | `R2/quality/test_results.py:55-65` | none | PORT (pick M8-2, M9-1) |
| M8 | libcst operator flips inside functions (arithmetic, boolean, comparison, membership, identity), ifs, integer constants | `R2/pipelines/recipes/swe_smith/mutations.py:27-130` | none | PORT (Apache-2.0) |
| M8 | Issue writer sees failing test sources and the defective log, never the diff | `R2/pipelines/recipes/swe_smith/issue.py:80-95` | DESIGN §7.2 says the Test writer cannot see the oracle diff; no rule for the instruction writer | PATTERN (pick M8-4) |
| M8 | Deterministic issue validator: no private test names, no talk of "the tests", ≤ 4000 chars, Python reproducers parse and have no undefined names; up to 3 metered attempts with feedback | `R2/pipelines/recipes/swe_smith/issue.py:26-77,98-125` | orchestrator validators + feedback exist (`CR/forge/orchestrator.py:343-360`); no instruction validator | PATTERN (pick M8-4) |
| M8 | LLM review with literal-quote citations, checked against the supplied documents | `R2/quality/loop/models.py:21-39`; `R2/quality/loop/protocol.py:14-35` | `CR/forge/schema.py:124-129` | PORT (pick M8-3) |
| M8/M9 | `Review.sound`: task, verifier and leakage assessments pass, no blocking issue, no reward-hack / task-defect / insufficient-evidence label, no open read requests | `R2/quality/loop/models.py:71-100` | `AuditReport.sound` = no blocking finding (`CR/forge/schema.py:127-129`) | PORT (pick M8-3) |
| M8/M9 | Semantic probes (wrong solution or valid alternative), bound to the bundle hash, run in a private variant; required probe foci learned from false acceptances (lazy output, numeric tolerance) | `R2/quality/loop/models.py:56-69`; `R2/quality/loop/runner.py:55-73` | negatives are authored (`CR/accept/checks/core.py:97-110`) | PORT (pick M8-3) |
| M9 | Controls before contrast: baseline and oracle must run cleanly (no exception, exit 0, reward present) | `R2/quality/loop/runner.py:76-90` | `ctx.grade` retries infra, then raises (`CR/accept/checks/base.py:97-112`) | nothing |
| M9 | A probe that does not install or execute cleanly is a failure, not a kill | `R2/quality/loop/runner.py:93-111` | any reward 0 is a kill (`CR/accept/checks/core.py:105,132-134`) | PORT (pick M9-1) |
| M9 | Behavioural kill: the failing run must be a completed failing test run, not an import or syntax error; evidence files are checksum-bound and size-capped | `R2/quality/loop/probe_behavior.py:13-91` | none | PORT (pick M9-1) |
| M9/M8 | Label vs evidence: a "legitimate success/failure" label must agree with the last rollout's reward and no infra exception | `R2/quality/loop/runner.py:114-138` | none (Calibrator labels are not cross-checked) | PORT (Calibrator) |
| M9 | Static verifier policy: tests may not read implementation source (`inspect.getsource`, opening `__file__`) | `R2/quality/loop/verifier_policy.py:30-79` | none in A13 | PORT (pick M9-7) |
| M9 | Readiness preflight: test selectors must exist in the frozen snapshot before any build or GPU | `R2/tasksmith/readiness.py:8-38` | none | PATTERN-low |
| M10 | `BudgetLedger`: SQLite `BEGIN IMMEDIATE`; micro-dollars; scoped caps; typed denials; duplicate op refused; settle needs evidence; persisted limit; increase needs the expected old limit and evidence | `R2/campaigns/budget.py:19-257` | `CR/runtime/budget.py:26-74`; `CR/forge/budget.py:19-41` | PORT (pick M10-4, M8-1) |
| M10 | Metered call: reserve, receipt `dispatched`, call, receipt `completed`, settle; missing usage or any exception marks uncertain; resume replays a receipt only for the same request hash | `R2/campaigns/llm.py:17-100` | none (forge roles and the runtime meter by client totals) | PORT (pick M8-1) |
| M10 | `EventJournal`: refuses to append after a torn tail; fsync on every non-progress event | `R2/campaigns/events.py:26-53` | appends after a torn tail (`CR/runtime/journal.py:29-47`) [ran P3] | PORT (pick M10-4) |
| M10 | `save_record`: mkstemp, fsync, `os.replace` | `R2/execution/lifecycle.py:19-31` | `CR/runtime/journal.py:50-57`; not used by the accept cache (D12) | nothing (reuse ours) |
| M10 | Retries only for observational reads; never for model calls or job dispatch | `R2/execution/read_retry.py:1-37` | none | PATTERN |
| M10 | Claim file as the recovery handle for a lost create response | `R2/execution/lifecycle.py:34-44` | none | PATTERN-low |
| M8 | Task-count allocation per source within one inventory snapshot | `R2/campaigns/scheduling.py:16-51` | diversity quota (`CR/forge/economy.py:160-170`) | nothing |

### 1.3 SWE-smith (`SS`)

| Mod | Mechanism | Source file:line | Crucible (file:line) | Verdict |
|---|---|---|---|---|
| M8 | Procedural, LM-modify, LM-rewrite, PR-mirror and combine bug generators | `SS/swesmith/bug_gen/` (prior: `02-forge-synthesis.md:144-200`) | none | PORT (MIT; via the Repo2RLEnv contrast rules, pick M8-2) |
| M8 | Issue prompt receives the bug patch, the test output and failing test source; "do not mention pytest" is a prompt rule only | `SS/configs/issue_gen/ig_v2.yaml:10-14,25,43,48-60` | — | anti-pattern (leaks location; prefer `R2` issue.py) |
| M9 | Validation: one buggy run against one reference run; a missing pre-gold output is written as `timed_out` and the candidate is dropped as `fail`; a timeout drops the candidate | `SS/swesmith/harness/valid.py:74-126,140-146` | — | anti-pattern (infra mislabelled and discarded; a hanging bug is lost) |
| M9 | F2P direction mismatch between `valid.py` and `gather.py`; history leak through a public 2-commit branch | prior: `02-forge-synthesis.md:183-200` | A5 leak audit (`CR/accept/checks/core.py:195-222`) | fixture for A5 / imports |

### 1.4 mini-swe-agent (`MS`)

| Mod | Mechanism | Source file:line | Crucible (file:line) | Verdict |
|---|---|---|---|---|
| M10 | Step, cost and wall-clock limits raise `LimitsExceeded` / `TimeExceeded` with an empty submission | `MS/agents/default.py:26-33,126-150` | `LimitPolicy` has timeout, steps and context only (`CR/runtime/infra.py:99-113`) | PATTERN: add a per-attempt cost limit stop |
| M10 | Consecutive format errors end with `RepeatedFormatError` | `MS/agents/default.py:100-113` | `too_many_errors` is a limit stop (`CR/runtime/harness/native.py:36`) | nothing |
| M10 | Model retry: tenacity, 10 attempts, exponential 4-60 s, abort on auth, context window, not-found, permission | `MS/models/utils/retry.py:9-25`; `MS/models/litellm_model.py:50-56` | — | nothing |
| M10 | Cost calculation: a zero or failed cost raises `RuntimeError` unless `ignore_errors` | `MS/models/litellm_model.py:108-126` | blank usage is masked and booked as uncertain spend (`CR/runtime/infra.py:41`; `CR/runtime/budget.py:56-59`) | nothing (Crucible ahead) |
| M10 | Batch: any exception becomes an empty submission in `preds.json`; resume skips every listed instance | `MS/run/benchmarks/swebench.py:122-177,229-232` | `ImportedHarness` masks a missing patch (`CR/runtime/harness/imported.py:33-34`) | anti-pattern; an import of historical runs must read `exit_status` |

### 1.5 FineEnvs (`FE`) and the three Spaces

| Mod | Mechanism | Source file:line | Crucible (file:line) | Verdict |
|---|---|---|---|---|
| M8 | PortSim generator: seeds from a string of (quay, week, seed, difficulty); a real week plus a seeded disruption | `FE/07-simulation-environments/portsim-v1/envs/berth_planning/core/berth_core/generate.py:176-190` | `CR/forge/` world synth uses toy seeds (`packs/_toy/.../forge/world_synth.py:124-143`) | PATTERN |
| M8/M9 | A candidate is kept only when CP-SAT proves the optimum; the frontier tier keeps a bounded gap and records `proven_optimal` and `lower_bound`; the oracle plan is re-evaluated to the same cost | `.../generate.py:195-204,318-331` | score gates record `best_known` with provenance (DESIGN §7.5); no "proven" flag | PATTERN (pick M8-5) |
| M8 | Headroom filter: the naive re-plan must cost ≥ 1.15 × optimum and ≥ 15 more; per-difficulty quotas; test split = weeks divisible by 5 | `.../berth_core/build.py:1-30,63-94` | A2 floors only require 0 (`CR/accept/checks/core.py:73-85`) | PATTERN (pick M8-5) |
| M9 | Import pins: instruction and grader hashes must match, per-file hash manifest, pinned dataset revision; train/test overlap by source or notebook raises | `FE/05-multi-harness-rl/prepare.py:38-125` | lineage sealing for forged tasks (`CR/forge/curator.py:58-84`); imports none | PATTERN (pick M9-8) |
| M10 | Eval identity: protocol source sha256, checkpoint, model, harnesses, task names, instruction hash; a mismatched output dir is refused | `FE/05-multi-harness-rl/eval/evaluate.py:191-210` | `CR/runtime/run.py:53-56` (no code hash) | PATTERN (pick M10-6) |
| M10 | Per-pair atomic files; only graded pairs (0/1) are reused; ungraded pairs are retried; secrets redacted from error text; coverage; exit 2 when incomplete | `FE/05-multi-harness-rl/eval/evaluate.py:140-157,214-246,289-290` | journal + coverage (`CR/runtime/journal.py`, `infra.py:128-129`); no redaction (`infra.py:83-84`) | PATTERN (redaction) |
| M10 | Over-masking: every exception, including "No verifier grade", is ungraded and retried forever | `FE/05-multi-harness-rl/eval/evaluate.py:49-50,234-243` | blame decides (`CR/runtime/infra.py:87-96`) | anti-pattern |
| M10 | Masked bounds in reporting: "even counting every ungraded cell as correct, its final score could not reach its starting 37.0%" | `SP2/app/src/content/chapters/training.mdx:384-386` | none (`CR/runtime/run.py:59-72`) | PATTERN (pick M10-6) |
| — | `SP1`, `SP3`: articles; code is review-comment tooling and a data export | `SP1/scripts/export_portsim_data.py`; `SP*/review/` | — | nothing |

### 1.6 OpenEnv (`OE`)

| Mod | Mechanism | Source file:line | Crucible (file:line) | Verdict |
|---|---|---|---|---|
| M8 | AWM env: 1,000 pre-built scenarios downloaded from the Hub; no synthesis | `OE/envs/agent_world_model_env/server/data_loader.py:16-21,62-66`; `README.md:16` | `CR/interop/openenv/awm.py:1-9` | correction (§0) |
| M9 | RFC 008 runtime "contracts" are declarations (plans, launch specs); the only grader is the static manifest | `OE/src/openenv/validation/runtime/contracts.py:79-238`; `validation/graders/static/manifest.py` | — | confirms the prior G1 finding |
| M9 | AWM rewards: `judge_error`, `server_error`, `reset_error`, `no_verifier` get 0.0; tool errors matched by keywords ("missing", "invalid") get −1.0; `incomplete` gets 0.1 | `OE/envs/agent_world_model_env/server/config.py:31-36`; `server/awm_environment.py:44-70,400-432,464-470` | Crucible uses AWM data only | anti-pattern (fixture for imports) |
| M10 | "Nothing raises out of `run_rollout`": a failure is a result with `reward=None` | `OE/src/openenv/harbor/rollout.py:14-17,505-509` | `CR/runtime/run.py:139` lets it raise [ran P1] | PATTERN (pick M10-1) |
| M10 | Harbor retry off: a Harbor-level retry re-runs the agent in the same capture session and merges two attempts into one graph | `OE/src/openenv/harbor/rollout.py:45-49` | one capture session per attempt (`CR/runtime/keys.py:7-8`) | PATTERN for `harness/harbor.py` |
| M10 | Per-key finite filtering; an unusable key is excluded, not 0; any `exception_info` sets `ok=False` (including agent timeouts) | `OE/src/openenv/harbor/rollout.py:513-553,573-577` | timeouts are policy outcomes (`CR/core/schema/run.py:56,62`) | nothing (OpenEnv over-masks timeouts) |

## 2. Best picks

### M8 forge

1. **Durable forge spend: one ledger, per-call receipts, per-node charges (PORT, M).**
   Source: `R2/campaigns/budget.py:19-257`, `R2/campaigns/llm.py:17-100`.
   Change: `CR/forge/budget.py` becomes a thin adapter over a hardened `CR/runtime/budget.py` (see M10-4), stored
   next to the forge journal, scope `forge/<plan>/<node>/a<k>`. `CR/forge/orchestrator.py:315-341` charges a node
   from its own role's meter (the role's `LLMClient.spend` delta or the Harbor trial's cost), never from the global
   `observed()` delta. `settle` never raises; an overrun denies the next reservation. `HarborRole` charges failed
   trials too (`CR/interop/harbor/role_runner.py:155-159,199,227`).
   Why: today two concurrent $1 nodes commit $4, and a settle past the cap aborts `run_plan` with no state [ran P4];
   a crash loses all spend; failed Harbor role trials are free.
2. **Coding bug generator with execution contrast (PORT, M).**
   Source: `R2/pipelines/recipes/swe_smith/mutations.py:27-130`, `worker.py:46-103`, `R2/quality/test_results.py:55-65`;
   SWE-smith procedural modifiers (MIT, prior `02-forge-synthesis.md:200-215`).
   Change: new `CR/accept/mutation/patch.py` (operators on the files the oracle touches, returned by `t.mutants()`),
   and a `G-bug` generator in `packs/coding/src/crucible_pack_coding/forge/` that keeps a candidate only if the
   healthy suite passes, collection is unchanged, the run exits "tests failed" and ≥ 1 healthy test fails.
   Why: A3's ≥ 90% mutant kill rule is unmeasurable on coding tasks today (D5), and the forge has no code-level
   generator. Repo2RLEnv's contrast rules fix SWE-smith's known holes (direction mismatch, mislabelled infra).
3. **Auditor and Adversary on Repo2RLEnv's review protocol (PORT, M).**
   Source: `R2/quality/loop/models.py:21-100`, `protocol.py:14-35`, `runner.py:55-138`.
   Change: `CR/forge/schema.py` `AuditReport` → `Review` (assessments, cited issues, rollout label, `sound`); a
   deterministic validator in the orchestrator checks every citation literally against the staged documents; the
   Adversary emits `SemanticProbe`s that become `NegativeSpec`s (`wrong` / `valid_alternative`) bound to the task
   hash; "required probe focus" entries live in the defect registry; the Calibrator's labels are checked against
   the recorded reward.
   Why: an uncited audit finding cannot be checked, and Crucible's auditor has no schema for "insufficient evidence".
4. **Instruction-writer rules (PATTERN, S).**
   Source: `R2/pipelines/recipes/swe_smith/issue.py:26-95`; anti-pattern `SS/configs/issue_gen/ig_v2.yaml:10-14,48-60`.
   Change: the Architect's node gets a validator (no hidden check ids or test names, no talk of "the tests", length
   cap, code blocks parse) and is staged with failing behaviour, never the oracle diff (`CR/forge/orchestrator.py`
   `Node.visible`; toy and coding role builders).
   Why: DESIGN §7.2 hides the oracle diff from the Test writer only; the instruction writer can leak the location.
5. **Generator acceptance from PortSim (PATTERN, S/M).**
   Source: `FE/.../berth_core/generate.py:176-204,318-331`, `build.py:1-30,63-94`.
   Change: the task card records `reference_proven` and `lower_bound` for score gates (`CR/core/schema/task.py`);
   `CR/forge/curator.py:58-84` rejects a candidate whose reference is unproven beyond a gap, or whose naive baseline
   leaves less than a declared headroom; seeds are derived from a string of the generator's inputs; splits follow
   source time or lineage.
   Why: a score gate normalises by `best_known − baseline`; without a proven reference and a minimum gap, a
   "record" can be noise and a task can have no room to learn.
6. **Forge journal durability (PATTERN, S).** `CR/forge/orchestrator.py:127-153`: fsync, tolerate and repair a torn
   tail at open (today it raises [ran P3]). Source: `R2/campaigns/events.py:26-53`.

### M9 accept

1. **Kills must be behavioural; zero mutants is a fail (PORT, M).**
   Source: `R2/quality/loop/probe_behavior.py:45-91`, `runner.py:93-111`, `R2/quality/test_results.py:55-65`.
   Change: `CR/accept/checks/core.py:97-145` counts a negative or mutant as killed only when the grade reports a
   completed failing run at the expected gate; build, import or collection errors are "stillborn" (reported, not
   scored); `scored == 0` is a fail ("mutation score unmeasured"). Needs a per-gate failure class in
   `CR/verify/runtime/plan.py` `GradeReport`.
   Why: A3 passes with zero mutants [ran P5a], and a mutant that only breaks the build proves nothing about a check.
2. **A11 hermetic and harness-true (S).** `CR/accept/checks/extended.py:145-189`: a fresh run root per evaluation
   (or clear `a11/`), run the task's own harness, and add the empty-collection injection for workspace tasks.
   Why: a second evaluation resumes the old journals and flips the credits case to "misclassified" [ran P6].
3. **Checks read evidence through the grade store, never the session memo (S).** `extended.py:214,228` iterate
   `ctx.grades`; call `ctx.grade(...)` for the oracle and each negative instead (store hits are free).
   Why: once A1 and A3 are cached, A12 errors on every run and can never pass [ran P5b].
4. **Atomic, parse-guarded acceptance files (S).** `CR/accept/cache.py:28-47`, `CR/accept/checks/base.py:88-111`:
   write with `journal.atomic_write`; a file that fails to parse is quarantined and treated as missing.
   Why: a torn grade makes its check error forever; a torn cached result crashes `accept_task` [ran P8].
5. **Score-gate-aware pass (S).** `extended.py:98,105`, `core.py:29`: a pass is `reward ≥ 1.0` (the threshold
   `CR/runtime/infra.py:132` uses). Why: A10 counts record-beating runs as failures [ran P5c].
6. **A7b: Harbor's quality checker as an advisory check (DEPEND, M).**
   Source: `H/analyze/checker.py:40-113,287-319`; rubric `H/cli/quality_checker/default-rubric.toml:1-54`.
   Change: new `CR/accept/checks/quality.py` (`Need.PAID`, `Need.HARBOR`), run on the exported Harbor dir for H3+;
   fail on `behavior_in_task_description`, `behavior_in_tests`, `anti_cheating_measures`,
   `tests_or_solution_in_image`, `hardcoded_solution`; a trial exception or invalid deliverable is `error`, not fail.
   Why: A7 checks author-declared mappings only; nothing reads the tests against the instruction.
7. **Static verifier lint in A13 (PORT, S).** `R2/quality/loop/verifier_policy.py:30-79` → `CR/accept/checks/extended.py:248-268`
   (prior M09-M11 #5, still open).
8. **Import pins when `imports/` is built (PATTERN, M).** `FE/05-multi-harness-rl/prepare.py:38-125`: hash-pinned
   instruction and grader, pinned revision, per-file manifest, lineage overlap rejects; Harbor parity files and
   adapter oracle screenshots are provenance only (`H/cli/adapter_review.py:32-44`).

### M10 runtime

1. **Contain every harness exception (PATTERN, S).** Source: `OE/src/openenv/harbor/rollout.py:14-17,505-509`.
   Change: `CR/runtime/run.py:139-142` wraps `harness.run` and `_rollout`; an escape becomes a masked
   `agent_run_error` with `blame="unknown"` (the stop rule counts it) and the reservation is settled as uncertain.
   Also put `CR/runtime/harness/native.py:256-263,269` and `CR/runtime/harness/workspace.py:110,118,122-131` inside
   `boundary`. Why: one adapter bug kills every in-flight sample and strands reservations [ran P1].
2. **Re-collect from the recorded base (S).** `CR/runtime/harness/workspace.py:95-96` reads
   `prior[1].parent / "base_sha"`; with no recorded sha, mask instead of using the movable tag.
   Why: the retry grades an empty patch as a 0 when the agent committed and moved `base` [ran P2].
3. **Harbor's error classes in the taxonomy (PORT names, S).** Source: `H/agents/installed/base.py:34-160`.
   Change `CR/runtime/infra.py:27-43`: `ApiUsageLimitError` → `crucible:credits_exhausted` (stop, resumable);
   `ApiRateLimitError`, `ApiConnectionClosedError`, `ApiResponseStalledError`, `NetworkConnectionError` → transport or
   `provider_unavailable` (retry with backoff); `ContextWindowExceededError`, `OutputTokenExceededError`,
   `AgentSafetyRefusalError` → policy (scored, not retried); `AgentAuthenticationError`, `ModelNotFoundError` → infra,
   not retryable, stop the run; `AgentSetupTimeoutError` → `crucible:backend_error`. When a Harbor result has both a
   reward and `exception_info`, classify by the exception first (`H/trial/single_step.py:75-87`).
   Why: today all of these are masked and retried as harness errors [ran P7]; a credits failure burns 3 attempts.
4. **Journal and ledger hardening (PORT, M).** Source: `R2/campaigns/events.py:26-53`, `R2/campaigns/budget.py:19-257`.
   Change: `CR/runtime/journal.py:29-47` repairs a torn tail at open (truncate to the last newline, keep the fragment
   as evidence) or refuses to append; `CR/runtime/budget.py` refuses duplicate ops, persists caps, requires evidence
   on settle and on `increase_limit`, uses `BEGIN IMMEDIATE` and integer micro-dollars, and calls
   `over_uncertain_tolerance` from the stop rule. Why: a torn tail silently loses the next terminal record [ran P3].
5. **Backoff and phase-scoped pools (PATTERN, S/M).** Source: `H/trial/queue.py:105-110,114-192`;
   `H/models/job/config.py:291-322`. Change: `CR/runtime/run.py:118-157`: wait `min·mult^n` with jitter between
   attempts, release the semaphore while waiting, separate agent and grading pools.
6. **Identity pins, secret redaction, masked bounds (PATTERN, S).** Sources: `FE/05-multi-harness-rl/eval/evaluate.py:191-210,237-242`;
   `H/models/job/lock.py:56-170`; `SP2/.../training.mdx:384-386`.
   Change: `RunSpec.identity_doc` (`CR/runtime/run.py:53-56`) adds harness and kit code hashes and task content
   hashes; `Failure.reason` redacts environment secrets (`CR/core/schema/run.py` validator, `CR/runtime/infra.py:83-84`);
   `RunSummary` adds the mean with masked samples as 0 and as 1, and the share of groups pass@k covers
   (`CR/runtime/infra.py:132-144`). Why: a comparison is decided only when the bounds do not overlap; pass@k silently
   drops groups today [ran P7].
7. **`harness/harbor.py` (DEPEND, M).** Source: `H/trial/trial.py`, `H/trial/single_step.py`; OpenEnv's reasoning for
   one session per attempt (`OE/src/openenv/harbor/rollout.py:45-49`). Run `Trial.create(...).run()` with Harbor
   retries off; Crucible owns retries; classify `exception_info` with pick 3; still the main gap for live
   installed-agent runs.

Do not adopt: Harbor's retry that deletes the failed trial (`H/trial/queue.py:222`) or lets a cleanup error replace a
graded result (`H/trial/trial.py:1860-1881`); Harbor and mini-swe-agent scoring errored runs as 0
(`H/utils/pass_at_k.py:41-43`, `MS/run/benchmarks/swebench.py:157-176`); SWE-smith's issue prompt with the patch
(`SS/configs/issue_gen/ig_v2.yaml:48-50`) and its timeout-drops-candidate rule (`SS/swesmith/harness/valid.py:84-126`);
AWM's 0.0 for judge and server errors (`OE/.../awm_environment.py:464-470`); FineEnvs' retry-every-exception
(`FE/05-multi-harness-rl/eval/evaluate.py:234-243`).

## 3. Crucible defects and spec gaps

- **D1 [ran P1] One harness exception aborts the whole run.** `CR/runtime/run.py:139,159`. A task whose `broken()`
  raises (`CR/runtime/harness/native.py:269`, outside any `boundary`) makes `Runner.run` raise `RuntimeError`; both
  ledger ops stay `reserved`; no `results.jsonl` or `failures.jsonl` is written. Same exposure at
  `native.py:256-263` and `workspace.py:110,118,122-131`.
- **D2 [ran P2] Re-collect retry ignores the recorded base sha.** `CR/runtime/harness/workspace.py:95-96` reads
  `base_sha` from the new attempt dir; `run()` wrote it into the previous one (`:110-112`). With the agent's commit
  and a moved `base` tag, the retry grades a 0-byte patch: reward 0.0, no failure. The recorded sha yields a
  171-byte patch with `fib.py`.
- **D3 [ran P3] Torn journal tails.** Runtime: after a torn `results.jsonl` line, the next terminal record is
  appended onto it and lost (`CR/runtime/journal.py:29-47`); `terminal()` returns `['s1']`, not `['s1', 's3']`, so
  resume re-samples a finished sample. Forge: `CR/forge/orchestrator.py:135-138` raises `JSONDecodeError` on a torn
  line, so the forge cannot resume at all. Neither journal fsyncs the forge side (`:148-153`).
- **D4 [ran P4] Forge budget double-counts and aborts.** `CR/forge/orchestrator.py:319-341` charges each node the
  global spend delta: two concurrent $1 nodes commit $4 (real $2). `CR/forge/budget.py:40-41` raises from `settle`,
  which escapes `_run_node`; with cap $2.5 and real spend $2, `run_plan` raises `BudgetExhausted` and the journal
  keeps only `draft`. The ledger is in memory, so a crash loses all spend.
- **D5 [ran P5a] A3 passes with zero mutants.** `CR/accept/checks/core.py:121-145`: `score=None`, `killed=0`,
  status pass. `tests/unit/test_accept.py:45` uses exactly this fixture (`mutants: list`).
- **D6 [ran P5b] A12 never recovers once A1 and A3 are cached.** `CR/accept/checks/extended.py:214,228` read
  `ctx.grades`, which cached checks never fill (`CR/accept/run.py:51-53`). After one infra error, A12 errors on
  every later run (`ValueError: need two equal-length samples of size >= 2`), blocking acceptance for that key.
- **D7 [ran P5c] A10 counts record-beating runs as failures.** `extended.py:105` passes only `reward == 1.0`: runs at
  1.3 and 1.1 give p = 0.0 and a band failure; `CR/runtime/infra.py:pass_at_k` gives 0.5 on the same rewards.
  Same equality in A1 (`core.py:29`) and A3 (`core.py:101`).
- **D8 [ran P6] A11 is not hermetic.** Fixed run ids under `workdir/a11` (`extended.py:145,166`) resume old journals.
  On a second evaluation of the same key (reuse off, a prior error, or a version bump), the credits sample is owed,
  attempt 2 runs without the injection, scores 1.0, and A11 fails a task it passed.
- **D9 [ran P7] Harbor error names fall through the taxonomy.** `CR/runtime/infra.py:27-52`: `ApiUsageLimitError`,
  `ContextWindowExceededError`, `AgentSafetyRefusalError`, `AgentAuthenticationError`, `NonZeroAgentExitCodeError`,
  `AgentSetupTimeoutError` all become `agent_run_error`, harness blame, masked and retryable. Crucible's own
  `ContextOverflow` is a scored policy outcome, so the same event is scored natively and masked under Harbor.
- **D10 [ran P7] Any `OSError` in a phase is infra.** `CR/runtime/infra.py:82-83` maps `PermissionError` and
  `FileNotFoundError` to `transport_unreachable`, masked. Low risk today (World tools catch their own errors), but
  it is a mask path a harness bug or a policy-shaped state can reach.
- **D11 [ran P7] pass@k drops groups silently.** `CR/runtime/infra.py:141-142` skips a group with fewer than k
  scored samples: pass@2 over {hard 0/2, easy 1/1 + 1 masked} is 0.0. Groups lost to masks are not reported.
- **D12 [ran P8] Torn acceptance files are permanent.** `CR/accept/checks/base.py:88-111` and
  `CR/accept/cache.py:28-47` write with `write_text` and parse without a guard. A torn grade makes A2 error on every
  run; a torn cached result makes `accept_task` raise `ValidationError` (`CR/accept/run.py:48`).
- **D13 [read] `HarborRole` drops cost and over-masks.** `CR/interop/harbor/role_runner.py:155-159,196-200`: any
  `exception_info` raises before the cost is returned, so failed role trials cost $0 in the forge ledger; every
  exception becomes `RoleInfraError` (`:228-229`), including `AgentTimeoutError` (a valid deliverable may exist,
  since Harbor still ran the verifier), `ApiUsageLimitError` (should stop the forge) and auth errors (retrying is
  pointless). The orchestrator retries three times with no backoff (`CR/forge/orchestrator.py:321-328`).
- **D14 [read] `PerfGuard` can block on the attempt's own sandbox.** It counts every `crucible*` container
  (`CR/runtime/perf_guard.py:49-50,78-79`); sandbox Worlds are named `crucible-sbx-*` (`CR/world/sandbox.py:81`);
  `WorkspaceHarness` grades before closing the World (`CR/runtime/harness/workspace.py:133` vs `:137-139`). Latent:
  today's perf grades go through `ImportedHarness` with no World.
- **D15 [read] Fixes with no production caller.** `gpu.preflight` (`CR/runtime/gpu.py:75-84`) and
  `remote.control` / `kill_by_pidfile` / `lint_self_matching` (`CR/runtime/remote.py:26-56`) are called only by tests.
  The registry marks `B.5-zombie` fixed by "--init + preflight".
- **D16 [read] A11 doc and detector drift.** See §0: `docs/modules/M09-accept.md:35` vs `extended.py:166`;
  detectors `B.5-empty-patch`, `B.5-zombie` attached to A11 (`extended.py:136`) without an A11 case.
- **D17 [read] A5 hides a found leak behind `pending`.** `CR/accept/checks/core.py:212-215` returns `PENDING` with
  only "needs docker" when the agent image cannot be audited, dropping text hits already found at `:200-206`.
  Acceptance stays blocked, but the report shows the wrong cause.
- **D18 [read] The acceptance key misses grading code outside `verify/runtime`.** `CR/accept/key.py:23,34-36` hashes
  `verify/runtime/*.py` and the plan files only. Edits to `CR/verify/stage.py`, `CR/verify/collect.py` or
  `CR/verify/leak_audit.py` keep old acceptances current.
- **D19 [read] Failure reasons are not redacted.** `CR/runtime/infra.py:83-84` stores `f"{name}: {ex}"`; nothing
  scrubs keys from provider error strings before they reach `failures.jsonl` and `results.jsonl`.
- **D20 [read] Imported runs retry a missing patch.** `CR/runtime/harness/imported.py:33-34` raises a retryable
  `capture_incomplete`; the runner retries it up to `max_attempts` with the same missing file (`CR/runtime/infra.py:93-96`).
- **Spec gaps [read]:** `harness/harbor.py`, `imports/` (I0–I10), `interop/harbor/compile_bridge.py` absent;
  `LimitPolicy` has no cost limit (`CR/runtime/infra.py:99-113`); no instruction-writer validator in the forge; no
  code-level bug generator; forge `Journal` and `Ledger` are not durable.

## 4. Probe index (`scratchpad/m810sweep/g1/`)

| Probe | Shows | Result |
|---|---|---|
| `p1_runner_uncontained.py` | D1 | `Runner.run raised: RuntimeError adapter bug in broken()`; ledger ops `reserved`, `reserved`; no results or failures files |
| `p2_recollect_base.py` | D2 | retry `stop recollected reward 0.0 failure None patch bytes 0`; recorded sha gives 171 bytes with `fib.py` |
| `p3_journal_torn_tail.py` | D3 | runtime `terminal()` = `['s1']` (s3 lost); forge `Journal` raises `JSONDecodeError` |
| `p4_forge_ledger.py` | D4 | (a) node costs 2.0 + 2.0, committed 4.0, real 2.0; (b) `run_plan` raises `BudgetExhausted` (4.0 committed vs cap 2.5), journal states `draft` only |
| `p5_accept_holes.py` | D5, D6, D7 | A3 pass with `score None`; A12 `error` in sessions 2 and 3; A10 `fail p = 0.0` vs runtime pass@1 0.5 |
| `p6_a11_rerun.py` | D8 | first A11 pass; second A11 fail "misclassified: credit exhaustion…"; credits run `scored 1, mean 1.0` |
| `p7_taxonomy.py` | D9, D10, D11 | Harbor names → `agent_run_error` harness masked retryable; `PermissionError` → `transport_unreachable` masked; pass@2 = 0.0 with the easy group dropped |
| `p8_accept_torn_files.py` | D12 | A2 `error` (JSONDecodeError) on two runs; `accept_task` raises `ValidationError` |
