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

**Not yet run (needs spend):** the MVP gate's report on tasks 01–04 and their span tasks for 2 frontier models and 1 open
model. That needs paid model runs and the lab GPU.
