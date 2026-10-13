# M13 `bench`: step-level and OOD benchmarks

**Study:** `docs/research/bench-studio/01-framework-study.md` §1–2. It covers every local framework: what we adopt, and
what none of them has.

**Code:** `src/crucible/bench/`

| File | What |
|---|---|
| `metrics.py` | Every DESIGN §11.1 metric, from an M10 run directory. **Masked samples are never samples.** |
| `compare.py` | Task-level comparison of one model per side (two runs, or two policies in one run). |
| `splits.py` | Group-keyed splits, sealing and temporal eligibility. |
| `irt.py` | 2PL item response theory, pure Python. |
| `contamination.py` | Canaries, overlap, near duplicates, fingerprints and leaks. |
| `dynamic.py` | Fixed anchors plus fresh items behind a key. |
| `report.py` | `summary.json`, `episodes.json`, one self-contained `report.html`, `harbor_result.json` and `crucible_aggregate_metrics.json`. |
| `run.py` | `run_bench` (controls → M10 runner → report), `report_run` (an existing run), `report_saved` (re-report from the saved config). |

**`metrics.py`** (every number states its denominator; audit `docs/audit/2026-10-13`):
- `load_run` reads every planned sample: the latest terminal attempt, and each planned sample with none as **owed**.
  Spend sums **every attempt** (masked retries, uncertain USD). A torn tail in `results.jsonl` is skipped.
- Coverage: planned / terminal / measured / masked / owed, measured tasks, tasks with nothing measured, masks by kind.
  An all-masked group reports coverage only. `summary["run"]`: planned, owed, the runner's stop reason.
- The headline is the **mean of per-task means**, with the pooled mean and the raw mean (masked as 0) beside it, and a
  **t interval on task means** with `n_tasks` (approximate below 10 tasks).
- pass@k, **pass^k**, unbiased **max@k** on **one task set for every k**: a k some measured task cannot reach is refused,
  with `pass@k_tasks` / `pass@k_excluded`. Default ks: 1, the powers of two up to k, and k.
- An **exposed** episode (answer key reached) is a scored 0 with a flag; hack metrics count exposures, grader tampering
  and passes that carry a rule flag. With several models, `common_support` scores them on the tasks all measured.
- **Validity**: a manifest (controls run, M9 acceptance, import report) excludes tasks whose oracle fails or is flaky,
  or whose no-op passes, from every model's metrics, IRT and compare (`invalid_tasks`, with reasons). Without one the
  summary says `validity: unknown`.
- Gate profile (micro `verifier_level_pass_rate`, macro).
- Progress AUC: (1/T) Σ Φ_t over t = 1..T from Φ_0, the start state (only where Φ is instrumented). Progress rows are read
  in episode order: M6's `step=0` row is the start state (milestones held at the start), the `step=-1` correction is
  the end state.
- Stage reach (raw index for one stage map, `stage_reach_frac` normalized), and the step each stage was first reached.
- First irrecoverable step: the first t after which every sampled step has a Wilson upper bound below 0.05 (a suffix
  condition); branch budgets that cannot resolve it (fewer than 73 branches) are counted as `branches_insufficient`.
- Regressions, recovery rate (`fault_start` tag), spread across repeats, subset breakdowns, cost per success (all spend /
  measured successes; GPU seconds `None` when unobserved), latency percentiles, messages per success (policy actors from
  the attempt), probe profile (attempts from the World trace).
- The per-step Φ heatmap, and a deterministic digest (cost blocks excluded by path).

**`compare.py`:** one (policy, harness) per side, refused otherwise; the **task** is the unit (two policies share no
seeds, and a task's k samples are not independent). Task flips (pass rate above 0.5), **exact McNemar over tasks**, the
mean of per-task deltas with a t interval and an exact sign-flip permutation p, common support, invalid tasks left out,
and a markdown report.

**`splits.py`:**
- Group-keyed (lineage), salted HMAC assignment.
- OOD axes held out by value, and time blocks with a guard gap.
- Showcase tasks never published.
- `seal` / `verify_seal`, and `check` (invariant violations).
- `eligible` / `eligible_subsets` (temporal decontamination).

**`irt.py`:**
- MAP with priors; fitted by alternating damped Newton.
- Automatic Rasch below 15 respondents.
- **Fixed-anchor calibration**, and `ability` by bisection with its SE.
- `drift` (anchor quarantine).
- `next_item` (maximum Fisher information), `item_qa`.

**`contamination.py`:**
- A per-version canary GUID and `strip_canary`.
- 13-gram overlap, and 50-character code substrings.
- MinHash LSH (5-word shingles, 128 hashes, 16×8 bands), verified by exact Jaccard ≥ 0.8.
- File-tree fingerprints (git blob sha1), and an issue/PR/commit leak regex.

**`dynamic.py`:**
- HMAC item seeds, and commit–reveal (`sha256(K)` published).
- A caller-supplied validation chain, with top-up on rejection.
- The manifest holds ids, never seeds.

**`report.html`:** rLLM's single-file viewer and Harbor's waffle and Pareto views, plus the per-step Φ heatmap. A
warning banner (stop reason, owed samples, validity, refused ks, small task counts, hacks), validity, common support,
masks by kind, a per-task table and the configuration. It holds no timestamp, so equal runs render byte-identical HTML.
`harbor_result.json` (Harbor `JobStats` shape) and `crucible_aggregate_metrics.json` (NeMo-Gym aggregate shape with
`coverage/*`) sit next to it.

**CLI:**
- `crucible bench run <pack> | --harbor <dir>... | --tasks module:attr`: oracle and no-op controls first (`--no-controls`
  to skip), `--ks`, `--key-file`, `--anchors`.
- `crucible bench report <run dir>`: re-reports from the saved config into `<run>/bench-report` (`--force` overwrites).
- `crucible bench compare <a> <b>`: `--base-policy` / `--cand-policy` / `--base-harness` / `--cand-harness`; a side
  with several models, or none, is an error.

## Gate

`tests/unit/test_bench.py` (4) and `tests/unit/test_m13_audit_metrics.py` (one regression test per audit finding, a
50-seed fuzz test against an independent reference, CI coverage by simulation; 17 metric mutants all killed).

The estimators are checked against hand-computed values:
- pass@k, pass^k, max@k;
- AUC, regressions;
- first irrecoverable step;
- McNemar.

IRT on simulated data (24 respondents × 30 items):
- θ and b correlate > 0.9 with the truth;
- fixed anchors keep their parameters, and fresh items land on their scale;
- a contaminated anchor is quarantined.

Splits:
- the OOD, time and showcase rules hold;
- no invariant violations;
- the seal verifies with the salt and fails without it.

Contamination: canary, 13-gram, code-substring and near-duplicate hits, and no false hit.

**Toy benchmark (the exit gate's toy part).** Two anchors plus four fresh backlog items from HMAC seeds, each validated
(gold solves it, no-op does not). Three scripted policies × k = 2:
- oracle: pass@1 1.0, AUC highest, stage reach 3;
- half: Φ in (0, 1);
- no-op: everything 0.
- IRT puts oracle > half ≥ no-op, and the task-level comparison finds 6 task regressions (exact McNemar over tasks
  p = 2/64, sign-flip p = 2/64). The earlier "12 regressions, p < 0.001" counted the k = 2 samples of each task as
  independent pairs (audit G2-8).
- **A re-run in a fresh directory gives the same digest and byte-identical HTML.**

## Gate, real models

**Toy, live (`crucible_pack_toy.bench_live`, $2.15, `reports/m13_toy_live/`).**
- Setup: Claude Sonnet 5.5 and GPT-6.1 Sol (frontier), and Qwen3-235B (open), on 7 toy tasks (5 forged, helpdesk, a
  12-ticket backlog), k = 2, native harness, every metric.
- IRT ability: Sol +0.60 (pass@1 0.71), Sonnet −0.01 (0.57), Qwen3-235B −0.59 (0.43).
- No model solves the helpdesk task: all fail the t2 / t3 gates.
- Under the RLM harness, Sonnet solved the backlog (reward 1.0, stage reach 3, $0.35), where the native harness scored
  0.67.

**Coding, recorded (`crucible_pack_coding.bench_recorded`, $0, `reports/m13_coding_recorded/`).**
- Covers tasks 01–04 (each version its own item) plus the task04 people and NCV spans, from the recorded frontier runs:
  SEEDS, run dirs, grade reports, the task04 history and the M8 span grades.
- 13 episodes, 3 masked (runs that ended without a grade: an API error or an abort).
- Results: Claude Fable 5.1 1.0 (n = 1), GPT-6 Astra 0.5 (n = 2), GPT-5.6 Sol 0.14 (n = 7).
- End-state Φ per run. Per-step progress is **not tracked**: the recorded agents edited through python and `sed`, which
  replay cannot follow (0–12 states out of 46–200 calls), so the report shows no curve rather than a guessed one.
- **Still open:** the open-model part on the coding tasks. It needs a live run on the GPU Worlds.

**Fixes made while running the real-model benchmark:**
- Concurrent attempts sharing one client were charged each other's spend ($2.23 recorded for $1.32 spent): per-attempt
  cost now comes from the attempt's own model calls.
- Token counts are read from `calls.json` when run records carry none.
- `Episode.progress_tracked`: an untracked run reports AUC as unmeasured, not 0.
