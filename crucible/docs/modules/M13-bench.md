# M13 `bench`: step-level and OOD benchmarks

**Study:** `docs/research/bench-studio/01-framework-study.md` §1–2. It covers every local framework: what we adopt, and
what none of them has.

**Code:** `src/crucible/bench/`

| File | What |
|---|---|
| `metrics.py` | Every DESIGN §11.1 metric, from an M10 run directory. **Masked samples are never samples.** |
| `compare.py` | Paired comparison of two runs, or two policies in one run. |
| `splits.py` | Group-keyed splits, sealing and temporal eligibility. |
| `irt.py` | 2PL item response theory, pure Python. |
| `contamination.py` | Canaries, overlap, near duplicates, fingerprints and leaks. |
| `dynamic.py` | Fixed anchors plus fresh items behind a key. |
| `report.py` | `summary.json`, `episodes.json`, and one self-contained `report.html`. |
| `run.py` | `run_bench` (M10 runner → report) and `report_run` (an existing run). |

**`metrics.py`:**
- Coverage (measured / masked / fully-masked tasks; an all-masked group reports coverage only).
- The measured mean, with the raw mean beside it; a seeded bootstrap CI over tasks.
- pass@k, **pass^k**, unbiased **max@k**.
- Gate profile (micro `verifier_level_pass_rate`, macro).
- Progress AUC (only where Φ is instrumented).
- Stage reach, and the step each stage was first reached.
- First irrecoverable step (Wilson upper bound on C3 branch success).
- Regressions, recovery rate (`fault_start` tag), cost per success (all spend / measured successes) with observability
  coverage, messages per success, probe profile.
- The per-step Φ heatmap, and a deterministic digest (cost excluded).

**`compare.py`:** joined on (task, sample index), not row order. Flips with regressions first, **exact McNemar**, a
**paired bootstrap** CI of the delta, and a markdown report.

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

**`report.html`:** rLLM's single-file viewer and Harbor's waffle and Pareto views, plus the per-step Φ heatmap. It
holds no timestamp, so equal runs render byte-identical HTML.

**CLI:** `crucible bench run <pack>`, `crucible bench report <run dir>`, `crucible bench compare <a> <b>`.

## Gate

`tests/unit/test_bench.py` (4).

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
- IRT puts oracle > half ≥ no-op, and the paired comparison finds 12 regressions with McNemar p < 0.001.
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
