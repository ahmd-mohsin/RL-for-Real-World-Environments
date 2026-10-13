# M13 bench: defect hunt H1 (metrics, compare, report, run)

Scope: `src/crucible/bench/{metrics,compare,report,run}.py` at HEAD b6d89fc, and what they read: the M10 run directory
(`runtime/run.py`, `runtime/journal.py`, `runtime/keys.py`, `runtime/harness/native.py`), `core/schema/run.py`, and the
M6 progress stream (`verify/progress.py`). Tests: `tests/unit/test_bench.py` (4 tests).

Reproducers live in the session scratchpad, `scratchpad/m13/h1/`. Every one runs with
`PYTHONPATH=src .venv/bin/python <file>` from `crucible/`; the scipy checks use `.envs/nemo/bin/python`.
- `synth.py` writes synthetic M10 run directories with the real writers (`RunJournal.finish`, `RunRecord.mark_failure`,
  `SampleKey`, `attempt_id`). It adds masked retries (non-terminal), masked terminals, policy failures (`agent_timeout`,
  scored) and unfinished samples, and records the truth for each sample. It also has an independent reference: pass@k in
  the product form, pass^k, and max@k by brute force over k-subsets.
- `fuzz_metrics.py` runs 1000 random runs (1–5 tasks, 1–2 policies, k 1–5, mask rate 0 / 0.2 / 0.6 / 1.0, binary or
  continuous rewards) and compares coverage, the mean, pass@k / pass^k / max@k for every k ≤ n, and total spend against
  the reference.

## Summary

| id | sev | file:line | defect |
|---|---|---|---|
| V1 | P0 | metrics.py:121-163, 347-356 | The progress tracker's end-of-episode correction row (`step=-1`) sorts first. Final milestones, stage reach, regressions, AUC and the heatmap then use the uncorrected state, and the final Φ leaks back to t=0. |
| V2 | P0 | metrics.py:119-137, 275, 352 | The initial state is ignored: Φ0 is taken as 0, and milestones held at the start are never "gained". A fault-start task (the recovery-rate case) reports stage reach −1 and AUC 0 for a full solve. |
| V3 | P1 | metrics.py:128 | Φ reached at the last step has weight 0 in the AUC. A one-step oracle scores AUC 0, the same as a no-op. |
| V4 | P0 | metrics.py:83-85, 109, 311-315 | Cost counts terminal attempts only. The spend of masked attempts that were retried, and `cost_uncertain_usd`, are dropped, so "all spend / successes" is understated. |
| V5 | P0 | run.py:35, metrics.py:224-229 | A stopped or partial run is reported as the whole benchmark. `run_bench` discards the `RunSummary` (planned count, stop reason), and coverage has no planned denominator. |
| V6 | P1 | metrics.py:71-74, 83 | A torn tail in `results.jsonl` makes `load_run`, and with it every report and compare, raise. The journal itself tolerates the same file. |
| V7 | P0 | compare.py:32-36 | Pairs are keyed on (task, sample_idx) only. Several policies or harnesses in one run collapse silently, and two identical runs show "significant" flips (McNemar p = 0.031). |
| V8 | P1 | metrics.py:193, 204, 219, 289 | Masking silently changes which tasks a number covers, and with what weight. The mean reweights tasks by their mask rate, and pass@k drops groups with n < k without saying so (pass@2 < pass@1). |
| V9 | P1 | metrics.py:232-245 | The "95%" CI from the percentile task bootstrap undercovers at the task counts Crucible reports: 0.76 (3 tasks), 0.87 (5), 0.90 (10). |
| V10 | P1 | metrics.py:112 | The probe profile's `attempted` is always 0 on M10 runs. Only the training path writes `probe/attempted`. |
| V11 | P1 | run.py:28-43, metrics.py:175-181 | Recovery rate and first irrecoverable step are never computed by `run_bench`, `report_run` or the CLI. With C3's default k = 8 the Wilson rule can never fire anyway (it needs n ≥ 73). |
| V12 | P1 | tests/unit/test_bench.py | The tests are vacuous for masking: 11 of 14 mutants (masked counted, wrong attempt, CI, cost, heatmap) survive. |
| V13 | P2 | metrics.py:366 | The digest drops every dict key named `cost`, including a task id or gate name `cost`. |
| V14 | P2 | metrics.py:92 | Tool calls and messages count only the actor `"agent"`. Other policy actors (P28 multi-agent Worlds) count 0. |
| V15 | P2 | metrics.py:111, 315 | `gpu_s` is never written by M10, yet it is reported as 0 per success. Observability covers tokens and USD only. |
| V16 | P2 | run.py:29, cli/studio.py:245 | `run_bench` and `bench run` keep `ks=(1,)` while sampling k = 2, so pass@k and pass^k for k > 1 are never reported. |
| V17 | P2 | metrics.py:178-181 | The first irrecoverable step is not a suffix condition: [(3, 0/80), (5, 4/8)] returns 3. |
| V18 | P2 | compare.py:28, report.py:55, metrics.py:298 | McNemar p is rounded to 9 places (0/40 gives "p = 0.0"). The Pareto y axis assumes reward in [0, 1]. Stage reach averages raw indices across tasks with different stage counts. |

**Checked and clean:**
- **Estimators.** pass@k, pass^k, max@k, coverage, the measured mean, policy-failed samples scored 0, the latest
  terminal attempt per sample, and all-masked groups (coverage only) all agree with the reference over 1000 fuzzed runs.
  The only mismatch was spend (V4).
- **McNemar.** It equals `scipy.stats.binomtest(min(b, c), b + c).pvalue` to within 5e-10 for b, c < 120.
- **Wilson.** The formula matches the standard score interval.
- **XSS.** A payload in the task, policy, harness, gate, milestone, stage, extras keys and values, and the title yields no
  element or handler in `report.html` (`xss.py` parses the output). `html.escape` with `quote=True` covers the
  single-quoted attributes.
- **Determinism.** HTML and the digest are deterministic apart from cost, which is documented. No timestamps,
  timezone use or unsorted dicts reach the output.
- **Exposure.** `crucible:answer_exposed` masks by default, so an exposed pass is never counted.

---

## V1 (P0). The `step=-1` correction row is applied first

**Defect.** `ProgressTracker.on_end` (`verify/progress.py:166`) re-evaluates every subgoal at the end of the episode. When
a subgoal changed without its keys being touched (drift), it appends a correction row with `step=-1`: the true final Φ,
plus the milestones gained and lost. Bench sorts rows by `int(r["step"])` in `progress_auc`, `final_milestones`,
`regressions`, `stage_first_reached` and `heatmap`. The correction is therefore applied **before** step 1, and the later
rows overwrite it:
- the final milestone set ignores the correction;
- a loss caught only at the end is not a regression;
- Φ_T leaks into [0, t₁) of the AUC and into the heatmap's early bins;
- the last heatmap bin shows the uncorrected Φ.

**Reproducer.** `repro_progress.py` drives the real `ProgressTracker` with a fake subgoal set. Output (`repro_progress.out`):
```
== C: step 1 gains a and b; at the end 'a' no longer holds (drift caught by on_end: row step=-1, lost=['a'])
 final_milestones = {'b', 'a'} truth = {'b'}
 stage_reach = (0, 's_ab') truth = (-1, None)
 regressions = 0 truth = 1
 progress_auc(rows, 2) = 0.5 (bench convention, Φ0 = 0 on [0,1), 0.5 on [1,2): truth 0.25)
== B2: ... phi_final 1.0
 progress_auc = 1.0 (truth 0.0 under bench's convention)
 heatmap = {'p@h': {'t': [1.0, 1.0, 1.0, 0.5]}}  (last bin 0.5 although Φ_T = 1.0)
```

**Fix.** Sort rows by `(step if step >= 0 else +inf)` everywhere: a `-1` row is the end state, placed at T. One helper,
`ordered(rows, T)`, used by all five functions. Alternatively, M6 could write the row with `step=T` (with a `final` flag).

**Regression test.** Rows `[{step:1, gained:[a]}, {step:2, gained:[b]}, {step:-1, lost:[a], phi:0.5}]`. Assert:
- `final_milestones == {"b"}` and `regressions == 1`;
- `stage_reach` of stage [a, b] is −1;
- `heatmap(...)[-1] == 0.5` and `heatmap(...)[0] == 0.0`.

All of these fail now.

## V2 (P0). The initial state is ignored (Φ0 and the milestones held at the start)

**Defect.** `summarize_group` calls `progress_auc(e.rows, e.steps)` with the default `phi0=0.0`. `heatmap` also starts
each bin at `phi = 0.0`. Yet M10 records the real Φ0 in `meta.progress.phi0` (`ProgressTracker.on_end`, copied by
`runtime/run.py:266`). Bench reads `phi_final` from there, but not `phi0`.

Milestones that already hold at `on_start` are never in any row's `gained`. `final_milestones`, `stage_reach`,
`stage_first_reached` and `regressions` therefore start from the empty set:
- a stage whose exits include an initially held milestone is never reached;
- losing an initially held milestone (breaking working code) is not a regression.

This is exactly the DESIGN §11.1 recovery-rate setting: "the H-span task starts at an expert state with a known injected
defect".

**Reproducer.** `repro_progress.py`, scenario A, with `a` held at the start and step 1 gaining `b`:
```
 end {'phi0': 0.5, 'phi_final': 1.0, ...}
 bench progress_auc(rows, steps=1) = 0.0
 final_milestones = {'b'} truth = {'a','b'}
 stage_reach = (-1, None) truth = (0, 's_ab')
 stage_first_reached = {'s_ab': None} truth = {'s_ab': 1}
```

**Fix.**
- Add `Episode.phi0`, loaded from `meta.progress.phi0`, and pass it to `progress_auc` and `heatmap`.
- In M6, have `ProgressTracker.on_start` emit a `step=0` row with `gained` set to the subgoals held initially. The
  stream then carries the initial set, and the sum of `delta_phi` still telescopes, since Φ0 is the row's Φ.
- Until then, seed `have` from a `step=0` row when one is present.

**Regression test.** A tracker started on a state that holds `a`, with one step gaining `b` and stages `[("s", [a, b])]`.
Assert `stage_reach == 0`, `stage_first_reached == {"s": 1}`, and AUC ≥ 0.5. Then lose `a` at step 2 and assert
`regressions == 1`.

## V3 (P1). The last step's Φ has zero weight in the AUC

**Defect.** `progress_auc` keeps Φ_t over [t, t_next) and closes with `phi * (T - x)`. The row at step T (the state after
the last action) therefore gets weight 0. For a T-step episode the AUC is the mean of Φ0…Φ_{T−1}, and the state actually
reached never counts.

This contradicts "rewards getting further *sooner*": an episode that finishes the task at its last step scores as if it
never had, and short episodes are hit hardest. The toy gate's `o > h > n` holds only because its episodes are long.

**Reproducer** (`repro_progress.py`, scenario B):
- one step, Φ 0 → 1.0: AUC 0.0, the same as a no-op;
- two steps, Φ 0 → 0 → 1.0: AUC 0.0.

**Fix.** Use the right endpoint, `AUC = (1/T) Σ_{t=1..T} Φ_t`: Φ_t occupies step t's interval (t−1, t]. Equivalently, shift
each row to `[step−1, step)`. `heatmap` already reads Φ at the end of the bin, so it is consistent with this. Document
the convention in M13-bench.md.

**Regression test.** Assert `progress_auc([{step:1, phi:1}], 1) == 1.0` and
`progress_auc([{step:1, phi:0}, {step:2, phi:1}], 2) == 0.5`.

## V4 (P0). Cost leaves out the spend of retried masked attempts, and uncertain spend

**Defect.** The module states "per success = all spend (masked included) / measured successes", but `load_run` builds
`Episode`s from `results.jsonl` only. That file holds one terminal row per sample. A masked attempt that the runner
retried is non-terminal: it goes to `failures.jsonl` and its own `record.json`, which bench never reads. Its spend, real
money settled in the ledger, vanishes. `cost_uncertain_usd` (blank usage, harness exceptions) is never read either.
Spend is therefore lowest exactly when infrastructure is flakiest.

**Reproducer.**
- `repro_cost.py`: a real `run_bench` where attempts 1–2 hit `transport_timeout` after $1.00 each and attempt 3 passes
  for $0.10 plus $0.50 uncertain.
  ```
  bench cost.usd 0.1  per_success.usd 0.1
  masked attempts in failures.jsonl: 2 spend 2.0 + terminal 0.10 (+0.50 uncertain)
  truth: all spend = 2.10 (2.60 with uncertain) per success
  ```
- `fuzz_metrics.py` (1000 runs): `cost_all_spend` mismatches in every run that has a retry, for example 7.76 reported
  against 10.10 actual.

**Fix.**
- In `load_run`, sum `cost_usd`, `cost_uncertain_usd`, tokens and wall time over **every** attempt of the sample: every
  `samples/<sid>/a*/record.json`, or the terminal row plus that sample's `failures.jsonl` rows. Keep the terminal
  attempt for quality.
- Report `usd_uncertain` beside `usd`. Optionally cross-check against `ledger.summary()`.
- Records under `aborted/` have no record file, so count them under observability as "spend unknown".

**Regression test.** A synthetic run (the real writers) with 2 masked retries at $1 and a pass at $0.10. Assert
`cost.usd == 2.1` and `cost.per_success.usd == 2.1`.

## V5 (P0). A stopped or partial run is reported as the benchmark

**Defect.**
- **Discarded summary.** `run_bench` awaits `Runner.run` and throws away the `RunSummary`: `planned`, `stopped` (budget,
  wall clock, health), coverage against the plan.
- **No planned denominator.** `report_run` sees only terminal rows. Samples the runner never dispatched (a stop rule,
  `credits_exhausted`, which is deliberately non-terminal, a killed process) do not appear in `coverage` at all. Their
  tasks are absent from every aggregate.
- **No warning.** `summary.json` and `report.html` carry no stop reason and no "planned" count. Yet `run.json` holds the
  spec (tasks × policies × harnesses × k), so the plan can be recovered.

**Reproducer.**
- `repro_run_bench_stop.py`: a real `run_bench` over 40 planned samples (10 easy tasks that pass, 10 that are masked
  `stale_image`, 20 hard tasks that score 0). The health stop rule fires after 20 terminal samples:
  ```
  runner stop events: ["health: mask fraction 10/20 > 0.2: the environment is broken, not noisy (...)"]
  planned samples: 40; report coverage: {'episodes': 20, 'measured': 10, 'masked': 10, 'measured_tasks': 10, 'fully_masked_tasks': 10}
  report pass@1: 1.0  mean: 1.0
  summary.json mentions planned/stopped: False
  ```
  Measured over the full plan, pass@1 is 10/30 = 0.33. The report says 1.0 and does not say the run stopped.
- `repro_partial.py`, part 2: planned 16, terminal 8, and bench reports `{'episodes': 8, 'measured': 8, 'masked': 0}`.

**Fix.**
- `load_run` reads `run.json["spec"]` and re-derives the plan (`Runner.plan` or the same `SampleKey` ids). A planned
  sample with no terminal row becomes `Episode(reward=None, status="owed")`.
- Coverage gains `planned` and `owed`, and `fully_masked_tasks` counts tasks that are all owed.
- `run_bench` puts `RunSummary.stopped` and `planned` into `summary["run"]`. `report.html` shows a banner when
  `stopped` is set or `owed > 0`.
- Optionally refuse `pass@k` overall when `owed > 0` unless `allow_partial=True`.

**Regression test.** Run the stop scenario above. Assert that `summary["run"]["stopped"]` starts with `"health"`, that
`coverage["planned"] == 40` and `coverage["owed"] == 20`, and that the HTML contains the stop reason.

## V6 (P1). A torn tail in `results.jsonl` crashes `load_run`

**Defect.** `metrics._jsonl` calls `json.loads` on every line. The runner writes `results.jsonl` by appending with fsync,
and a crash mid-write leaves a fragment. `RunJournal._read_jsonl` skips such a fragment (journal.py:56), but bench raises
`JSONDecodeError`. That breaks `bench report`, `bench compare`, the studio explorer `/runs/{id}/summary` and
`datasets.from_run` on exactly the partial runs that need a report.

**Reproducer** (`repro_partial.py`, part 1):
```
journal.records() after the torn tail: 6 records
load_run raised JSONDecodeError Unterminated string starting at: line 1 column 54 (char 53)
```

**Fix.** Read through `RunJournal`'s tolerant reader (DEPEND rather than reimplement), or skip only an undecodable
**last** line. A bad line in the middle should still raise. Apply the same to `progress.jsonl` and `trace.jsonl`.

**Regression test.** Append a fragment to `results.jsonl` and assert that `len(load_run(d)) == 6`.

## V7 (P0). `compare` collapses episodes that share (task, sample_idx)

**Defect.** `b = {key(e): e for e in base}` uses `key = (task, sample_idx)`, so duplicate keys overwrite one another
silently. The overwrites happen:
- whenever a run has more than one harness;
- with more than one policy and no `--base-policy` / `--cand-policy` (the CLI default);
- for imported episodes that all carry `sample_idx=0`. `crucible_pack_coding/bench_recorded.py` gives
  `sample_idx=0` to every recorded run, for example GPT-5.6 Sol's 7 runs on one task.

Which episode survives depends on input order, so the result is neither paired nor deterministic.

**Reproducer.** `repro_compare.py` uses one run with 2 harnesses × 3 tasks × k = 4, compared with itself. The only change
in the second call is the candidate's order:
```
 pairs: 12 (truth 24)  unmatched: 0  flips: 0 0
 same episodes, candidate listed in another order -> pairs 12  pass_to_fail 0  fail_to_pass 6  mcnemar_p 0.03125  mean_delta 0.5
```
An identical run shows a "significant" improvement of +0.5.

**Fix.**
- Key on `(task, harness, sample_idx)` when comparing policies, or `(task, policy, harness, sample_idx)` when comparing
  runs, through a `pair_on=` argument.
- Raise on duplicate keys on either side, naming the key.
- Make the CLI require `--base-policy` / `--cand-policy` (and harness) when a run holds more than one.

**Regression test.** Assert that `compare(base, list(reversed(base)))` gives 24 pairs, 0 flips and delta 0, and that
`compare` raises on two episodes with the same key.

## V8 (P1). Masking silently changes which tasks a headline number covers, and their weights

**Defect.**
- **The mean reweights tasks.** `mean_reward` (the CLI's and `build`'s headline) pools episodes, so a task's weight is
  its count of *measured* samples. Masking shifts weight toward tasks with reliable infrastructure, typically the light
  ones.
- **Different task sets per column.** `pass@k`, `pass^k` and `max@k` average only groups with `n_measured ≥ k`. That is
  correct (nothing is fabricated), but nothing reports the exclusion, so each column covers a different task set.
  pass@2 can come out below pass@1, which is impossible for the quantity it estimates.

**Reproducer** (`repro_task_sets.py`):
```
mean_reward 0.8 | pass@1 0.5 | pass@2 1.0 | pass^2 1.0        (hard task: 3 of 4 masked, 1 measured fail)
pass@1 0.5  pass@2 0.0  (pass@2 < pass@1)
```

**Fix.**
- Report `mean_reward` as the macro mean over tasks (per-task means averaged), next to the pooled mean, or make the
  macro mean the headline.
- For every k, report `pass@k_groups` (groups counted) and `pass@k_excluded` (groups with n < k).
- In the overall row, compute every k over the same group set (n ≥ max k), or mark the column as partial.

**Regression test.** On the fixture above, assert `macro_mean == 0.5`, `pass@2_excluded == 1`, and that every reported
k uses the same group count, or carries the counts.

## V9 (P1). The "95%" CI undercovers at the task counts Crucible reports

**Defect.** `bootstrap_ci` is a percentile bootstrap over tasks. With few clusters the percentile interval is too narrow:
it has no small-sample correction and no t quantile. The toy benchmark has 6 tasks, the recorded coding benchmark about
4 per model, and the report labels the result "95% CI".

**Reproducer.** `ci_coverage.py` runs 1000 simulated benchmarks per row: p_t ~ Beta(1, 1) (target E[p] = 0.5), k = 4,
20% of samples masked at random. Output (`ci_coverage.out`):

| tasks | percentile bootstrap (current) | candidate: cluster-t (CR1, t with T−1 df) |
|---|---|---|
| 3 | 0.757 | 0.933 |
| 5 | 0.874 | 0.931 |
| 10 | 0.902 | 0.927 |
| 30 | 0.947 | 0.954 |

The paired-bootstrap delta CI in `compare` (`ci_coverage_compare.py`, true delta 0) covers 0.889 at 5 tasks and 0.930 at
10 (P2 part).

**Fix.**
- Use a cluster-robust t interval for the ratio mean, `t_{T−1} · sqrt(Σ(S_t − m·n_t)² / (T(T−1)·n̄²))`. A bootstrap-t
  or BCa interval would also do.
- Report the number of tasks with the CI, and below about 10 tasks label it "approximate".
- Apply the same to `compare`.

**Regression test.** A seeded simulation with 300 benchmarks of 5 tasks, asserting coverage ≥ 0.92. It fails now at
0.874.

## V10 (P1). The probe profile's `attempted` is always 0 on M10 runs

**Defect.** `load_run` reads `probe/attempted` from `rollout.json`'s `reward_components`. Only the training path writes
that key (`train/world_env.py:318`). The M10 runner's `_rollout` goes through `runtime/record.build`, which copies
`meta.components`, and the runner never sets that.

The data does exist on disk, in the attempt's World trace: `trace.json` → `outcome.details.probes.attempted`, written by
`world.py:1009`. Bench therefore reports `probes.attempted = 0` for every benchmark run.

**Evidence.** `grep -rn "probe/attempted" src` hits only `train/world_env.py:318` (the writer), the reader
`bench/metrics.py:112` and `train/runtime.py:903`. Nothing in `runtime/` writes it.

**Fix.** In `load_run`, read `trace.json["outcome"]["details"]["probes"]` and sum its `attempted` counts. Better, have
the M10 `_rollout` set `reward_components["probe/attempted"]` as the training path does: one writer, two readers.

**Regression test.** A native-harness run on a World with an answer-key probe that the policy tries and the World
blocks. Assert `probes.attempted ≥ 1`.

## V11 (P1). Recovery rate and first irrecoverable step are never computed by any entry point

**Defect.**
- `summarize` takes `branches`, and `load_run` takes `tags_of`. But `run_bench` and `report_run` accept neither and pass
  neither, and the CLI and `studio/build.py` call those two. So `recovery_rate` is None (no episode has the
  `fault_start` tag) and `first_irrecoverable_step` is None on every benchmark, although DESIGN §11.1 lists both.
- Even when wired, `first_irrecoverable_step` flags a step only when the Wilson upper bound is below 0.05. That needs
  **n ≥ 73** branches with 0 successes. C3's `branch_value` defaults to k = 8, whose 0/8 upper bound is 0.324, so the
  rule can never fire at the default C3 budget.

**Reproducer.**
- `grep -n "tags_of\|branches" src/crucible/bench/run.py src/crucible/cli/studio.py src/crucible/studio/*.py` finds
  nothing.
- `first_irrecoverable_step([(s, 0, 8) for s in range(1, 50)])` returns `None`. The smallest n with
  `wilson(0, n)[1] < 0.05` is 73.

**Fix.**
- Add `tags_of=` and `branches=` (or a `branches.json` read from the run directory) to `run_bench` and `report_run`, and
  read `fault_start` from task metadata (`getattr(task, "tags", ())`).
- Make the threshold and the branch budget consistent: either default `threshold` to what k = 8 can resolve, or have C3
  choose k from the threshold.
- Report `branches_insufficient` when no step could fire.

**Regression test.** `report_run(..., tags_of=lambda t: ["fault_start"])` gives a non-None `recovery_rate`. With 8
branches per step and 0 successes, the summary reports "insufficient branches", not a silent None.

## V12 (P1). The bench tests are vacuous for masking and most aggregates

`mutants.py` copies `src/`, applies one mutant at a time, and runs `tests/unit/test_bench.py`. Output (`mutants.out`):
```
SURVIVES | mean_reward counts masked as 0
SURVIVES | coverage counts masked as measured
SURVIVES | gate profile includes masked episodes
SURVIVES | compare keeps masked pairs (masked = fail)
SURVIVES | load_run keeps the FIRST terminal row per sample
SURVIVES | max@k ignores exposure
SURVIVES | bootstrap CI returns (0, 1)
killed   | digest includes cost
SURVIVES | cost per success over scored only
SURVIVES | heatmap always 0
killed   | pass^k = pass@k
SURVIVES | stage_first_reached never resets on loss
SURVIVES | recovery rate counts masked
killed   | paired CI unpaired
```
Eleven of 14 survive.
- `test_estimators` has one masked episode, and only for pass@k / pass^k.
- The toy benchmark has no masks, retries, gates, cost, fault starts or branches. It asserts nothing about
  `mean_reward_ci` or the heatmap's values.
- None of the 4 tests reads a run directory with a retried attempt, a regrade or a torn tail.

**Fix.** Add `tests/unit/test_m13_audit_*.py` built on a synthetic-run fixture that uses the real writers, as `synth.py`
does. Include a fuzz test against a reference implementation (the `fuzz_metrics.py` shape, with 50 seeds in CI), plus
the per-finding regression tests above.

## V13 (P2). The digest drops every key named `cost`

`deterministic_digest.strip` removes `k == "cost"` at **every** depth. The summary uses task ids
(`models[].tasks[<task>]`) and gate names (`gate_profile.gates[<gate>]`) as keys, so a task or gate named `cost` (a
plausible AP-invoice gate) leaves the digest. `repro_digest.py` prints: "digest equal although task 'cost' and gate
'cost' results differ: True".

**Fix.** Strip only the structural `cost` key, the sibling of `deterministic` in a group block, by path. Or hash
`deterministic` sub-blocks explicitly.

**Regression test.** Two summaries that differ only in task `cost`'s mean must have different digests.

## V14 (P2). Only the actor "agent" counts as the policy

`load_run` sets `policy_actors = {"agent"}`. A World whose policy actor has another id, or a P28 multi-agent World with
several policy peers, reports 0 tool calls and 0 messages, so the communication-cost metric is 0 and messages per
success is None.

**Fix.** Take the policy actors from the World spec in `trace.json`, or from the keys of `calls.json` and the
`trajectory.<actor>.json` artifacts. Count messages between policy actors separately (P28's communication cost).

## V15 (P2). GPU seconds are reported as 0, not as unobserved

Nothing in M10 writes `meta.gpu_s`; the only `gpu_s` hits are in bench. `cost.gpu_s` is therefore 0.0 and
`per_success.gpu_s` is 0.0, which reads as measured. `observability` covers tokens and USD only.

**Fix.** Report `None` when no episode carries `gpu_s`, and add `gpu_s` and `wall_s` to `observability`.

## V16 (P2). `ks` is not derived from k

`run_bench(k=2)` and `crucible bench run --k 2` keep `ks=(1,)`, so pass@2 and pass^2 (the reliability metric the module
advertises) never appear. `bench report` cannot pass `ks` at all.

**Fix.** Default `ks=tuple(range(1, k + 1))`, or `(1, k)`, read from `run.json["spec"]["k"]`, and add `--ks` to
`bench report`.

## V17 (P2). The first irrecoverable step is not a suffix condition

DESIGN defines it as "first t after which no continuation reaches success". The code instead returns the first step whose
own upper bound is below the threshold, even when a later step on the same trajectory still succeeds: with
`[(3, 0, 80), (5, 4, 8)]` it returns 3. The trajectory itself continued from step 3 to step 5, and 4 of 8 branches from
step 5 succeed, so step 3 was not irrecoverable.

**Fix.** Return the smallest t such that every sampled step ≥ t has an upper bound below the threshold.

## V18 (P2). Small presentation issues

- `mcnemar_exact` rounds to 9 decimals, so p = 2⁻³⁹ is shown as `0.0`. Show `< 1e-9` instead.
- `_pareto_svg` maps reward with `y = h − pad − (h − 2·pad)·reward`. Record-type rewards above 1 (P29) and negative
  penalty rewards (`LimitPolicy` `penalty:<c>`) draw outside the plot. Scale y to the data range.
- `stage_reach` averages raw stage indices across tasks with different numbers of stages, with −1 for "none". Normalize
  each by `len(stages) − 1`, or report it per task only.
