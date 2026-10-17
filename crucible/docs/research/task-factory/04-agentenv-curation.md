# agentenv as the base of the curation method (study 04)

Question: for the M17 task factory (`03-design.md` §1, §2, §7, §9), what does agentenv-framework (Apache-2.0) already give,
what does Crucible already use, and where must we build? Paths below are relative to
`agentenv-framework/src/agent_env/` unless they start with `crucible/`.

**Finding in one line.** agentenv is a versioned store, a step DAG, a run recorder and a read-only browser. It has **no task
authoring from intent, no oracle or mutant run, no verifier test, no failure taxonomy and no label store**. It has two
useful seams we can use: a `review` step with a decision store, and an explorer plugin that mounts any FastAPI router.

## 1. agentenv's task model

### 1.1 Task, steps, verifier

- **Task** is an id, a version and an inline list of steps, nothing else (`task/task.py:296-315`). No prompt, intent or
  metadata field; the prompt lives in a `prompt_agent` step.
  ```python
  class Task:  type = "task"
      def __init__(self, id, version, steps: list[TaskStep] | None = None): ...
  ```
- **TaskStep** (`task_step/task_step.py`): `id, version, depends_on, fail_task_on_error, retry_config`, an abstract
  `execute(context)` and an optional `preflight() -> list[str]`. `depends_on=None` means "after all prior steps";
  independent steps run concurrently. Steps are registered by their `type` string (registry, entry points `agent_env.task_steps`).
- **DAG validation at creation** (`task/task.py:317-348`): duplicate step ids, forward or unknown `depends_on`, and an invalid
  `retry_from_step_id` raise `ValueError`. This is a structural check only.
- **Reward / verifier = a step that writes `context.metadata["verifications"][verifier_id]`** and a `score`. Built-ins
  (`task_step/task_steps/verifiers/`):
  - `verify_sandbox`: four probe types (`probe_file_exists`, `probe_dir_exists`, `probe_file_contains`, `bash_cmd_succeeds`);
  - `run_container_unit_tests_verifier`: runs a command in the agent's container, optional `reward_path` read as a float
    (Harbor style; it runs in the agent's container, so hidden tests are not isolated: DESIGN.md:788);
  - `env_outcome_verifier`: a script against the env;
  - `rubrics_verifier`: LLM or agent judge with `criteria`, formats incl. `rubric_binary`, `rubric_evidence` and
    `rubric_evidence_with_mistakes` (`judge_utils/judge_output_format.py:1198`), optional `grading_policy_prompt`,
    `context_facts_for_judge` (measured facts the judge must trust);
  - `aggregate_verifiers`, and `ScoreAggregator = all_pass | any_pass | weighted_average` with negative weights as
    penalties (`verifiers/scoring.py:12-48`). A criterion row is `{id, criterion, result|score, weight}`.
- **Run state** is `TaskStepContext` (`task_step/context.py:207`): `deployed_envs/agents/sandboxes`, `prompt_responses`
  (`PromptResponse` with `structured_output`, `tool_call_count`, `error_type/code/message`, trajectory object URLs), and a free
  `metadata` dict holding `seed`, `verifications`, `score`, `failed_steps`, `review_decisions`, `retry_resets`.
- **Seeds = parametrisation.** `context.metadata["seed"]` substitutes `<key>` in every string field of every step
  (`task/task.py:91-117`, applied on every start); `task run-batch --seeds file.csv` runs one task over CSV rows
  (`cli/task/run.py:548-647`). This is a task family: one template, many instances.

### 1.2 Artifacts, bundles, envs

- **Artifact** = immutable, versioned Pydantic object: `id, version, type` (`artifact/artifact.py:24-81`); every `put()` makes a
  new version; document goes to the document store, bytes to the object store. Built-ins: `file`, `file_artifact_universe`,
  `environment`, `environment_universe`, `docker_image`, `skill`, `cli`; extra types by `[artifacts] impls` or entry point
  (Crucible's `crucible_ref` is one).
- **Env** (`env/env.py`): `deploy() -> DeployedEnv`; MCP server, multi, website, DB, gateway. Env "cards" and validators
  (`task_step/task_steps/env_card_validator/`, `mcp_env_validator/`: schema, spec conformance, `validation_gate_aggregator`
  rolling `{passed, failed_gates, missing_gates, skipped_gates}`, fail-closed on a missing gate, `validation_gate_aggregator.py:27-48`)
  gate publication of an **env**, not a task.
- **Bundle** (`bundle/`): a folder of toml, json and files (`examples/hello/`) that `parse -> resolve -> plan -> preflight ->
  materialize -> run`. The plan checks id collisions, reference loops, wrong-type refs and tasks whose steps do not build
  before any write (`bundle/plan.py:1-9`); `preflight_run` checks deploys and judge agents (`bundle/preflight.py:76-83`);
  `run` runs each bundle task once, 4 at a time, with outcomes `PASSED | BELOW_ONE | UNSCORED | FAILED | CANCELLED`
  (`bundle/run.py:45-51`). `from_toml` is the authoring hook per entity type (`artifact/artifact.py:70`).

### 1.3 Versioning, storage

- **`(id, version)` is the unit**, unique index; `VersionedEntityStore.put` allocates the next version with retry
  (`store/document_store/document_store.py:412-486`). Collections: `tasks`, `evals`, `artifacts`, `envs`, `task_instances`,
  `task_step_journal`, `task_reviews`, `bundle_ledger`. Backends: SQLite (default), Mongo, DynamoDB, Firestore; objects on the
  filesystem, S3 or GCS. A new backend must pass the conformance kits (`AGENTS.md`, Tests).
- **Bundle ledger** (`bundle/ledger.py:1-41`): one row per written version with a digest of its inputs and one digest per
  input, so a re-run writes only what changed and a new version says why ("files changed: server.py"). This is
  the closest thing to **task lineage** agentenv has.
- **Eval** is a versioned list of pinned `(task_id, task_version)` (`eval/eval.py:14-67`), immutable (a change = a new version).
  This is a benchmark/suite object. It has no splits, labels or metrics.
- **Namespace routing** keeps `@local` ids in local stores (`store/routing.py`); one config file picks every backend.

### 1.4 Validation, review, quality checks

| What exists | Where | What it checks |
|---|---|---|
| DAG validation | `task/task.py:317` | ids and edges at construction |
| `Task.preflight()` and `agent-env task validate` | `task/task.py:350`, `cli/task/validate.py` | only that artifacts/envs a step names resolve; most step types have no preflight and the CLI prints coverage ("N of M steps have preflight checks") |
| Env validators and gate aggregator | `task_step/task_steps/mcp_env_validator/` | env publication, not tasks |
| Bundle plan and preflight | `bundle/plan.py`, `bundle/preflight.py` | writes and deploys, not task quality |
| Human review step | `task_step/task_steps/review.py:47`, `task_step/review_store.py:28-110` | see 1.5 |

**Not present:** an oracle run, a no-op or empty-action floor, mutants, a hack library, a flake bound, a verifier test or
accuracy measure, a second verifier, requirement-to-check traceability, a task lint, a "task card". Searched
`task/`, `task_step/`, `eval/`, `bundle/`, `cli/`, `explorer/` for these; no match. `run_container_unit_tests_verifier` and
`bash_cmd_succeeds` are checks, and nothing tests them.

### 1.5 Human-review and annotation surface

- **`review` step** (`task_step/task_steps/review.py:47-146`): registers `awaiting` in `ReviewStore`
  (`task_reviews`, unique `(instance_id, step_id)`, TTL 7 days, `review_store.py:19-25`), shows the previous step's
  `structured_output` bounded to 16 000 chars as `upstream`, polls every 5 s up to 3 h, heartbeats, then continues or raises
  `ReviewAbortedError`. A decision is exactly `continue | abort` plus `decided_by` (`set_decision`, `review_store.py:79-103`).
  **Nothing in this repo or in Crucible calls `set_decision`**: the writer is a hub outside the repo (module docstring:
  "the hub writes here"). Its schema has no label, reason, rubric score or free text.
- **Explorer** (`explorer/`, `agent-env up`, loopback `:8234`; FastAPI plus a prebuilt Next.js SPA):
  - read routes: versioned list/get/versions for artifacts, envs, tasks, agents, evals (`routers/common.py:97-170`);
  - runs: `POST /tasks/{id}/run`, `/runs` (k runs), `/cancel-run`, `GET instances`, `run-groups` and an SSE stream
    (`routers/runs.py:144-470`), a funnel tally `completed/failed/running`;
  - views: trajectory viewer, rubric matrix (`ui/src/components/rubric-matrix.tsx`), triggers timeline, "reviewer overview" HTML,
    task-instance viewer, an environments page.
  - **No write route for labels, comments, tags, decisions or edits** (entity routes are GET only). The UI has no hook for plugin
    pages (no `plugin` string in `ui/src` outside docs and the environments page).
  - **Two seams that matter:** `ExplorerPlugin` mounts any `APIRouter` under `/api/v1/...` before the core routers, from the
    `agent_env.explorer_plugins` entry point (`explorer/plugin.py:23`); and `[explorer] static_dir` swaps the SPA
    (`explorer/app.py:236`), so a Crucible-built UI is possible without forking.
  - The `/objects/content` route serves stored HTML under `Content-Security-Policy: sandbox` (`explorer/routers/objects.py`).

### 1.6 Generation and authoring tools

None for tasks. Authoring is by hand: toml/json bundles, `agent-env task create`, `eval create`, `eval add-tasks`
(`cli/task/create.py`, `cli/eval/*.py`). There is no LLM task generator, intent store, seed-to-variant operator or dedup. The
`mcp_cli_builder/codegen.py` step generates CLIs from an MCP server (env tooling, not tasks).

### 1.7 Eval runners and metrics

- `agent-env eval run --id E --k N --max-concurrency M --output-dir D` runs every pinned task N times; per run it writes a
  context JSON (`to_safe_dict`) and prints `PASSED score=...` or the exception (`cli/eval/run.py:70-185`). **The only
  summary is `n/total completed`** plus each run's `metadata["score"]`. No pass@k, no confidence interval, no per-cause
  count, no cross-run aggregation (grep for `pass_at_k` finds only a log-prefix tag, `task/task.py` comment).
- `Runner` seam (`runner/runner.py:89-127`): `submit -> RunHandle(run_id, instance_id)`, `status -> RunRecord`
  (`QUEUED|RUNNING|COMPLETED|FAILED|CANCELED`), `cancel`; `LocalRunner` built in; run groups tie k runs together
  (`metadata["run_group_id"]`).
- `TaskInstance` (`task/store.py:169-206`): `instance_id, task_id, task_version, status, current_step, completed_steps,
  context, error, rev, step_attempt_failures`; `StepAttemptFailure{attempt, step_id, step_type, error_class, error_message,
  duration_s}` (`:209`). Failures are recorded by **exception class**, not by cause of the failing trajectory.

## 2. What Crucible already uses, and what it does not

**Wired (all under `crucible/src/crucible/interop/agentenv/` and `studio/`; only `crucible.interop` and the studio import
`agent_env`, enforced by import-linter, `crucible/pyproject.toml:128-138`):**

| Piece | Where | Kind |
|---|---|---|
| `crucible_world` and `crucible_verify` task steps (World behind a real gateway; single reward producer; masked grade = `skipped: True` then raise) | `plugin/steps.py:1-25`, entry points `pyproject.toml:57-59` | DEPEND (plugin) |
| `crucible_ref` artifact (WorldRef, StateRef, grade, trace, rollout; `derived_from` lineage) | `plugin/artifacts.py` | DEPEND |
| `crucible` explorer plugin at `/api/v1/crucible`: packs, pack check, runs, summary, report, episodes, datasets; path containment, CSP sandbox | `studio/explorer.py:1-40`, `pyproject.toml:64` | DEPEND |
| GPU flags on the local sandbox provider | `plugin/gpu_docker.py` | DEPEND |
| A2A agents as harness; `LLMToolAgent` | `harness.py`, `agents.py`, `a2a.py` | DEPEND |
| agentenv env images as Apps | `mcp_env.py` | DEPEND (protocol client) |
| Trigger semantics, config loading, registries | `core/triggers.py`, `core/registry.py:3` | PORT / PATTERN (core may not import agentenv) |
| `eval run --k`, artifact versioning, trajectories, snapshots, explorer | `DESIGN.md:3031` | listed as reused |

**Not used yet (DESIGN.md and the forge show no import or mention):**
- agentenv's `Task` / `Eval` / `TaskStore` / `VersionedEntityStore` as the **task catalogue**: Crucible packs are its own task
  unit; `crucible/forge/` has its own `repo.py`, journal and `YieldBook`.
- `review` step and `ReviewStore`; `TaskInstance` / `RunRecord` / `Runner` as the factory's job record; the bundle `Ledger`
  (digest-based "why was this version written").
- `rubrics_verifier` with `rubric_evidence_with_mistakes` (the only built-in "findings with severity" judge), the
  `validation_gate_aggregator` roll-up, `Task.preflight`, the `step_journal` / `undo_steps` rollback, seed CSV batches.
- `[plugins.<dist>]` settings are used only for `data_root`.

## 3. Mapping to design 03
Legend: **DEPEND** = import and use as-is; **PORT** = copy the pure logic into `core` (Apache-2.0, as done for triggers);
**PATTERN** = copy the shape; **NOTHING** = build new.
### 3.1 Objects (§1)

| Object | agentenv piece | Verdict | Note |
|---|---|---|---|
| **IntentCard** | none (Task has no intent field). Storage shape: `Artifact` (versioned, immutable) | **PATTERN** for storage; content NOTHING | Model it as a Crucible pydantic type and store it as a `crucible_ref`-style artifact subtype (`kind="intent_card"`) so it gets versions and the explorer's list/get/versions routes for free. Immutability fits "a change = a new version". |
| **ObservationRecord** | `Artifact` + object store (hashed bytes); `PromptResponse` trajectory URLs; `crucible_ref.derived_from` | **DEPEND** (artifact + object store) | Pointer + hash + date + who is the artifact doc; trajectory bytes already live in the object store. The `type` taxonomy (human-graded, expert end state, ...) is NOTHING in agentenv. |
| **RequirementLedger** | `rubrics_verifier` criteria rows `{id, criterion, weight}` and `validation_gate_aggregator` fail-closed gating; `bundle/ledger.py` digests | **PATTERN** | agentenv has no prompt-span to check map. Reuse: the criterion-row shape (one id per requirement, so `verifications[...]` rows line up with ledger ids), fail-closed "required gate missing = fail" (`validation_gate_aggregator.py:27-48`). The two-way A7 check is NOTHING. |
| **VerifierGoldSet** | none | **NOTHING** | Store solutions as artifacts (bytes + label doc). Labels need a new collection (see 3.3). `aggregate_score` shows how to turn rows into a verdict, but nothing measures a verifier. |
| **FailureCause** (closed taxonomy) | `StepAttemptFailure.error_class`, `rubrics_verifier` mistakes `findings[severity]` | **PATTERN** | agentenv classifies exceptions. Our causes classify trajectories. Follow the judge format registry (prompt + schema + parser per format, `judge_output_format.py:1198-1215`) for the trace auditor: add a `failure_cause` format by plugin, if `JudgeOutputFormat` can be extended (it is a closed `Enum`; probably PORT the registry instead). |
| **TaskCard** | `Task` (steps) + `Eval` pin + `bundle/ledger.py` row (lineage) + `crucible_ref.derived_from` | **PATTERN** | Keep Crucible's TaskCard as the truth; publish a read-only mirror to agentenv as `Task` (steps = `crucible_world`, `deploy_agent`, `prompt_agent`, `crucible_verify`) so `eval run` and the explorer work on it. Measured stats go to the card, not the `Task`. |

### 3.2 Machine states (§2)

| State | agentenv | Verdict |
|---|---|---|
| Machine, journal, retries, resume | `Task.run` is a **DAG of steps with CAS-journaled context, rollback span and resume** (`task/task.py:157-296`, `step_journal.py`). It does not support cycles or an `escalate` outcome (`depends_on` refuses forward refs; `retry_config.max_retries` exhausted = the task fails, which is the right "escalate" signal). | **PATTERN**. The factory machine is cyclic (guard failure goes back to state 1..4), so keep Crucible's `forge/orchestrator.py` and `TaskFSM`; do not stretch `Task` to carry loops. Port the **journal + `undo_steps`** idea only if the forge's own journal lacks CAS rollback. |
| 1 Intent, 2 Spec, 3 World+prompt, 4 Verifier writer, 5 Oracle, 7 Mutants, 8 Hackers, 9 Second verifier, 12 Select | roles (LLM) | **NOTHING** in agentenv. Run roles as A2A agents if we want one harness (`interop/agentenv/harness.py`, `agents.py`), the one real reuse. |
| 6 Solvers (ladder rungs, parallel) | `eval run --k` / `run-batch --seeds`, `Runner.submit`, run groups, `teardown_run` | **DEPEND** for execution of rollouts against a published `Task`; Crucible's `forge/ladder.py` already owns policy choice. |
| 10 Trace audit | `rubrics_verifier` (direct LLM judge over a trajectory with `trajectory_filter`, retries, per-criterion frame grounding, `mistakes` rows) | **DEPEND or PORT** as the judge runner, with a Crucible `failure_cause` rubric. Cross-model rule is not enforced by agentenv (judge model is a step field): enforce it in the recipe config. |
| 11 Gate (A1-A18) | `validation_gate_aggregator` roll-up (`passed, failed/missing/skipped gates`) | **PATTERN**. `accept/model.py` already has the richer `Status` (pass/fail/NA/pending/error with evidence). Keep it. |
| Floors, flake bound, hack replay, differential | none | **NOTHING** (Crucible `accept/`, `forge/hacker.py`). |
| Human gate in the machine (sign-off, S1) | `review` step + `ReviewStore` (pause, `awaiting`, TTL, continue/abort) | **DEPEND for the pause/poll mechanism, extend the decision schema**: see 3.3. |

### 3.3 Human-labelling surface and decision store

**Can the explorer or eval UI be the human ground-truth intake?** Not as shipped, but it is a good **host**.

What is missing (all verified in 1.5): no write route; no label or reason field in `ReviewStore`; the UI lists
only fixed pages with no plugin-page hook; `eval run` prints pass/fail only. What helps: it already shows the three things a labeller
needs side by side (trajectory viewer, rubric matrix with per-criterion verdicts and grounded frames, run-group list), and the plugin
seam lets Crucible add routes at `/api/v1/crucible/...` with its existing containment and CSP-sandbox patterns.

Recommended split:
1. **Back end = Crucible explorer plugin** (extend `studio/explorer.py`): `GET /label/queue` (traces sampled for the 10% human sample,
   gold-set candidates, FP/FN candidates), `GET /label/item/{id}` (trace, ledger, verifier verdict, hidden for blind labelling),
   `POST /label/item/{id}` (label, cause, reason text, labeller id, timestamp). Write to a Crucible **append-only JSONL/SQLite
   store keyed by item id and labeller** (needed for κ: two labellers per item, no overwrite). Do not reuse `task_reviews`: it is
   unique per `(instance_id, step_id)`, expires after 7 days, and has one `continue|abort` decision.
2. **Front end = new page in a Crucible static bundle** (`[explorer] static_dir`) or a small standalone page served by the plugin
   under the sandbox CSP. A patch of the stock SPA is not needed. Reuse its trajectory and rubric components only by copying
   (MIT/Apache) if a native view is wanted; otherwise link to the stock instance page (`/tasks/{id}/instances/{iid}`).
3. **Decision store (§7)** = the same append-only store, one row per decision `{subject, gate, decision, reason, decider, model_or_human,
   time}`; criteria clustering and gate compilation are NOTHING in agentenv.
4. **Pause in the machine** = agentenv's `review` step when a machine runs as a `Task`; in the forge, a `pending` check status
   (`accept/model.py` already defines it) polled from the label store. Same pattern, no dependency needed.

### 3.4 The rest
| 03 item | agentenv | Verdict |
|---|---|---|
| Versioned catalogue of IntentCards, TaskCards, gold sets | `VersionedEntityStore` + document/object stores | **DEPEND** (store backends), shared with agentenv's explorer list routes |
| Recipe versions (`MACHINE.md`) | `Eval` (immutable versioned collection), bundle ledger digests | **PATTERN** |
| Yield ledger | none (`RunRecord` has status and times only) | **NOTHING**; Crucible `forge` `YieldBook` |
| Transfer test (held-out eval) | `Eval` pins tasks; no split concept | **PATTERN**: one `Eval` version per split (`train`, `heldout`), with splits sealed in Crucible's `splits.json` |
| Seed-driven task families | `metadata["seed"]` substitution, `run-batch --seeds` | **DEPEND** when a published `Task` is a template over a seed CSV |

## 4. Recommendations (ranked)

1. **Do not try to build the curation method on agentenv's task model; build the objects in Crucible and publish to agentenv.**
   agentenv has no intent, ledger, oracle, mutant, gold-set or cause concept. Keep IntentCard...TaskCard as Crucible types,
   and expose them as agentenv artifacts (`crucible_ref` kinds `intent_card`, `task_card`, `gold_set`, `label`) so versioning,
   storage and listing come free. Add the kinds to `CrucibleRefArtifact.kind` (`plugin/artifacts.py`).
2. **Build the human intake as a Crucible explorer plugin with its own append-only label store.** Reuse the plugin seam, the path
   containment and the CSP sandbox (`studio/explorer.py`). Do not reuse `ReviewStore` for labels (one decision, 7-day TTL, no
   reason, one row per step). Required fields: item id, labeller, label, FailureCause, reason, blind flag, timestamps; two labellers
   per item so κ is computable. This is the first thing the user needs for the first VerifierGoldSet (03 §10).
3. **Use the `review` step pattern for human pauses, with a richer decision.** The poll-with-heartbeat, idempotent `put_awaiting`
   and fail-closed timeout are right; port `ReviewStore` into Crucible with `reason`, `decided_by` and a closed decision enum
   (keep, drop, revise, escalate), no TTL for factory state.
4. **Keep the factory machine in the forge, not in `agent_env.Task`.** `Task` is an acyclic step DAG that fails on exhaustion;
   the machine is cyclic with `escalate`. Port only the journal-and-rollback idea (`step_journal`, `undo_steps_sync`) if the forge
   journal lacks compare-and-swap resume.
5. **Publish each accepted TaskCard as an agentenv `Task` + versioned `Eval`** (steps `crucible_world`, `deploy_agent`,
   `prompt_agent`, `crucible_verify`). Then the ladder, held-out transfer and training-time re-checks run through
   `eval run --k`, `Runner` and the explorer's run groups, and each split is one pinned `Eval` version.
6. **Use `rubrics_verifier` for the trace auditor (state 10) with a Crucible failure-cause rubric**, run by a model of a different
   family. Check whether the closed `JudgeOutputFormat` enum can take a plugin format; if not, port the format registry
   (`judge_output_format.py:1198-1215`) rather than patching agentenv. Enforce the cross-model rule in recipe config.
7. **Copy the `validation_gate_aggregator` semantics into the gate report**: a required gate with no verdict fails; `skipped` is
   advisory and must carry a reason (Crucible's `NA_REASONS` already does this). Extend with "gate ran on a stale verifier
   version = missing".
8. **Add a `Task.preflight()` equivalent for the ledger (A7 both ways) and surface it through `agent-env task validate`-like CLI**
   (`crucible task validate`), since agentenv's preflight covers only references, and the CLI honestly reports "N of M steps checked".
   Copy that honesty: print what was checked and what was not.
9. **Reuse the bundle ledger idea for lineage**: a digest of every input to a written version plus per-input digests, so a new
   TaskCard version says "prompt changed", "verifier v3 -> v4", "World bug fix" (`bundle/ledger.py:1-41`). This feeds the decision store
   and the yield ledger (which version was cheaper per accepted task).
10. **Add the missing eval metrics in Crucible, not in agentenv**: pass@k with Wilson / Clopper-Pearson intervals, cause mix per task,
    floors. agentenv's eval prints completed/failed counts only. Put them in the M13 summary and show them in the plugin's task
    page; offer upstream only after they are stable.

## 5. Risks
- Every use above stays in `crucible.interop` or the studio; forge and accept may not import `agent_env` (`crucible/pyproject.toml:128-138`).
- agentenv is v0.9.x with no written compatibility policy (`README.md`, Versioning); only the plugin surface is CI-protected. Depend on that, copy the rest.
- Not verified: a plugin page link in the SPA (no hook found); a plugin `JudgeOutputFormat` (an `Enum`); an outside hub writing `ReviewStore` (inferred from its docstring).
