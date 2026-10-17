# M17 task factory, study 10: checkpoint credit, honesty and the sealed benchmark (built)

What study 08 §6–§8 asked for, as built on branch `lh-coding`. Every piece reuses an existing seam: the observer protocol
of `run_episode`, the `progress.jsonl` → `Turn.advantage` path of the record builder, M6's `turn_credit_map` /
`token_credit`, C1's potential, the predictive-validity gate, `bench/contamination.py` and `factory/publish.py`.

## 1. Checkpoint Φ (`src/crucible/factory/checkpoint.py`)

**The contract consumed** (`MilestoneTask`, checkpoint.py:53): `milestones` (each `.id`, `.requires`, or a mapping with those keys, as the window lane's), `requirement_ids`,
`async milestone_status(patch) -> {milestone: bool}` run in a separate grading container. Masked grading raises or omits
a milestone; the sidecar reads that as *unmeasured*.

**Snapshots** (`CheckpointSidecar`, checkpoint.py:332):

| trigger | where | state captured |
|---|---|---|
| `commit` | `on_step` (checkpoint.py:389), `is_commit` (:151): `git` in command position, global options, then `commit` (not `commit-tree`) | after the commit step |
| `periodic` | every `every` tool calls (K = 10) | after that step |
| `compaction` | `before_action` (:377): the policy's root segments grew (`root_compactions`, :159; a resume is not one) | the state the new segment starts from: the action chosen after the compaction is seen *before* the World applies it |
| `end` | `on_end` (:408) | the final state |

- The snapshot is a patch against the start commit, resolved before the agent acts (`start_commit`, :178; the `base` tag
  can be moved later). The default diff is `world.state.fs.git_diff_command` (throwaway index, no hooks, no fsmonitor, no
  global config) on the task's host workspace (`workspace_snapshot`, :191). A task can supply `snapshot_patch(world)`
  instead (a container workspace), or the config an injected `snapshot`.
- Identical patches share one grading (sha256). The start state (empty patch) is graded once per task, outside the cap,
  so a milestone already true at A is never credited to the agent.
- **Cap** (default 20; `_admit`, :449): the end always has a slot. Over the cap, the lowest-priority snapshot not yet
  graded is dropped: periodic < compaction < commit, then the one closest to its kept predecessor (`_gap`, :475), so the
  end and the commit points survive. A dropped snapshot is not a checkpoint; its span merges into the next.
- **Async grading** (`_pump` / `_grade`, :483–:500): background tasks, `concurrency` 2, `grade_timeout_s` 1800 s; the
  agent waits only for the diff. `defer=True` grades nothing before the end (deterministic thinning).

**Φ** (`potential`, :139): Σ w_m · pass_m · Π_{r ∈ requires(m)} pass_r with Σ w = 1, C1 exactly (direct prerequisites;
a test asserts equality with `workflow.potential.potential`). Weights: `CheckpointConfig.weights`, else the task's own `weights` mapping,
else uniform; `weights_from_validity` (:122) derives them from `credit.validity.validate` results (kept: max(0, 2(AUC − ½)); dropped: 0). If no
milestone earns weight, Φ ≡ 0: no shaping rather than unvalidated shaping.

**Unmeasured** (`_status`, :502): an exception, a timeout, a non-dict, or no bool verdict for some milestone. The point
carries the last measured Φ with `measured: false`; its ΔΦ is 0 and the next measured point gets the whole change. Never
read as 0. A failed snapshot is also unmeasured.

**The series and the credit.** `series` (:224) gives `[(turn, Φ, newly passed, lost)]` (`CheckpointSidecar.phi_series`);
lost milestones are the credited set's shrinkage, so a loss is a negative ΔΦ at that point. `progress_rows` (:243) writes
the `verify.progress` row format, one row per World step with Φ carried:

- `assign="close"` (default): a span's ΔΦ goes on the row of the step that closes it (the commit, the K-th call, the
  last action before a compaction, the final action). Chosen because that turn produced the state that was measured,
  and because it keeps credit sparse and attributable.
- `assign="spread"`: ΔΦ / len(span) on every step of the span.
- A checkpoint at turn 0 (compaction before any action) puts its ΔΦ on a `step=-1` row: unattributed.
- The rows telescope to Φ_final − Φ0 (tested).

The runner already reads `progress.jsonl` (`runtime/run.py` `_rollout` → `runtime/record.build` →
`credit_from_progress`): the k-th row of an actor is the k-th action of its calls, so `Turn.advantage` = ΔΦ of the call
that issued the closing action. RLM sub-calls and children are folded onto the root turn by `_ancestors`.
`train/credit.turn_credit_map` → `token_credit` then apply unchanged: centred per episode, bounded by the outcome
advantage, none on gate-failed or hacked episodes, compaction calls on the episode advantage (tested end to end with the
workspace and the RLM harness).

**Evidence**, written only after the agent loop ends (a local-shell agent can never read it mid-episode): `progress.jsonl`,
`checkpoints.jsonl` (turn, triggers, patch sha256 and size, state, milestone vector, Φ, gained, lost, grading time,
reason), `checkpoints/<sha>.patch`, and the summary in `res.grade["progress"]` → `RunRecord.meta.progress` →
`RolloutRecord.progress` (`kind="checkpoint"`, phi0, phi_final, series, compaction_turns, graded / unmeasured / skipped
counts, give_up, honesty). No reward text anywhere.

## 2. The harness hook (opt-in; zero change when off)

- `runtime/harness/native.py:153` `EpisodeSidecar` Protocol: `on_start`, `on_step`, `before_action`, `on_end`. Runtime
  defines the seam and the factory implements it, so runtime never imports factory (all 4 import contracts kept).
- `episode_sidecar` (native.py:166): the harness's `sidecar(attempt)` param, else the task's `episode_sidecar(attempt)`;
  None = off.
- `probed` / `_ActProbe` (native.py:173–:200): only when a sidecar is on, each policy is wrapped so the sidecar sees the
  chosen action before the World applies it. Every attribute is the policy's; the action is returned unchanged; a sidecar
  error never ends the episode. The originals still go to checkpointing, `save_episode` and spend accounting.
- `NativeHarness.run` (native.py:368): the sidecar is the observer, replacing the subgoal `ProgressTracker` (one writer of
  `progress.jsonl`). It is not attached to a resumed or multi-step attempt. `RLMHarness(sidecar=...)` passes it through
  (rlm.py:72). `WorkspaceHarness.run` (workspace.py:149) adds it as the only observer and stores its summary.
- **How a task opts in:** define `episode_sidecar(self, attempt)` returning
  `CheckpointSidecar(self, attempt.dir, CheckpointConfig(...))`, or run with
  `WorkspaceHarness(sidecar=checkpoint_sidecar(cfg))` (checkpoint.py:610), which attaches a sidecar to every attempt whose
  task satisfies `MilestoneTask` and caches the start-state grading per task.

## 3. Honesty (`src/crucible/factory/honesty.py`)

- `STATUS.json` from the final tree (`status_from_tree`, :82; a symlink is refused) when the task has a host workspace,
  else from the final patch (`status_from_patch`, :89: a created file only; a partial edit cannot be read without its
  base). Missing or invalid file = no claim: **p = 0.5 for every verified requirement** (no information, Brier 0.25).
  Values outside [0, 1] or non-numeric count as missing; bools are 1 / 0.
- `score` (:132): Brier over requirements the verifier checked (`requirement_status(patch)` if the task has it, else the
  end checkpoint's verdicts through `requirement_milestones` {req: [milestones]}, or same-named milestones). Unknown keys are ignored and listed: nothing
  unverifiable is ever scored. Term = −λ · Brier, λ = 0.1. Properness is tested numerically on a 201-point grid.
- `GIVE_UP: <reason>` at a line start of the final message, or the RLM's `rlm.give_up` (`Done(reason="gave up: …")`)
  (`detect_give_up`, :39). The Done already stops the episode; the outcome is graded as it stands. The sidecar records
  the reason in the summary.
- **Composition** (`compose_advantage`, :156; `honesty_group_fn`, :174, an M12 group stage). The term is added to the
  outcome *advantage*, not the reward. Â ≤ 0 can only fall; Â > 0 is floored at 0. A reward-level fold would give an
  honest failure in an all-fail group a positive advantage; a test shows the naive fold doing that and the composition not.

## 4. Long-horizon metrics (`src/crucible/bench/lh_metrics.py`)

`LHEpisode` (:38; `from_progress` reads the sidecar summary).

- `summarize` / `by_intent` (:108, :141): pass@1, pass^k (τ-bench C(c,k)/C(n,k)), mean final Φ, milestones lost per
  episode, I6 compaction loss (losses per span starting at a boundary vs per other span, and their ratio), Brier and
  ECE of the claims, give-up rate, correctness (the outcome failed) and prematurity (another attempt solved the task),
  hack rate, and tokens and $ per solved task.
- `paired` (:161): per-task trained − base, a grouped seeded bootstrap CI, measured-sd MDE = 2.8·sd/√n, and the assumed
  sd = 0.25 MDE (0.128 at n = 30, 0.149 at n = 22), with a plain-language note.

## 5. Sealing (`src/crucible/factory/sealing.py`)

`seal` (:98) over two window stores:

- disjoint repos;
- every held-out window dated after every training window (violations listed with both dates);
- the base model's cutoff, reported, fatal only with `strict_cutoff`;
- undated windows are a finding;
- `bench.contamination` fingerprints of the held-out prompts and diffs scanned against training prompts and diffs, with
  harness boilerplate subtractable via `template`.

`assert_sealed` raises `SealError`. `publish_heldout` (:137) publishes train and held-out as separate splits through
`publish_family`, only on a clean report.

## 6. Tests (`tests/unit/test_m17_c_*.py`, 30 tests)

- **checkpoint:** triggers, the cap, prerequisites, a lost milestone, unmeasured points, spread, background grading.
- **harness:** WorkspaceHarness end to end (`Turn.advantage` → `turn_credit_map` / `token_credit`; no leak into
  observations, the trace or the attempt dir mid-episode; off by default; task opt-in; GIVE_UP), and the RLM harness
  with ≥ 2 compaction boundaries.
- **honesty:** parsing, Brier, properness, give-up, composition.
- **metrics:** the metrics, MDE and sealing.

## 7. Deviations and open points

- **One new file beyond the brief:** `factory/sealing.py` (sealing needs `factory.publish`, which `bench` may not import).
- **Commit triggers** fire on the command, not on its exit code. A failed commit costs at most one grading, and none if
  the tree is unchanged (dedup).
- **Compaction is the root agent's only.** A child agent's compactions do not trigger checkpoints.
- **Rows are written at the end, not streamed.** A crash mid-episode loses the checkpoint rows. The attempt is masked
  anyway, and retries do not attach a sidecar to a resumed attempt.
- **`bench/metrics.py` treats `progress.jsonl` as per-step Φ.** With checkpoint rows Φ is carried between measurements,
  so its progress AUC is a step-function approximation.
- **The window store's field names are not known yet.** `seal` uses accessor fallbacks (`repo`, `date_b`/`end_date`/…,
  `prompt`, `diff`), overridable per call.
