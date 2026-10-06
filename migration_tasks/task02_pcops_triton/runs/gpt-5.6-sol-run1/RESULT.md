# GPT 5.6 sol single-attempt run: task02_pcops_triton

**Reward: 0.**

- **Harness:** mini-swe-agent 2.4.6, bash only, 350-step / $80 cap.
- **Sandbox:** a `--network none` Docker container from `task02-pcops-triton:v1` on the lab RTX 5090. The hidden tests were mounted only in the fresh grading container after submission.
- **Usage:** GPT submitted on its own after **88 steps, $2.29**. The patch has 31 files and 3,020 lines (it deleted `csrc/` and added Triton kernels and a custom build backend).
- **Audit:** no network was available.

| gate | result |
|---|---|
| static (no native code, Triton present) | pass |
| install without a CUDA compiler | pass |
| ops | **132/138**: `scatter` mean/min/max wrong for int32/int64 |
| hard | **36/37**: FPS fails on a 150k-point example |
| app (pristine pcnet/pcdata, CUDA goldens, training trajectory) | pass |
| perf (grader skips it after a failed gate; measured separately) | **8 of 18 operator cases fail**; the benchmark crashed at grid_cluster before the pcnet cases |

| case | CUDA ms | GPT ms | ratio | bar at the time |
|---|---|---|---|---|
| fps 16×16k / 4×65k | 112.2 / 432.6 | 41.1 / 56.2 | 0.37 / 0.13 | 0.50 |
| knn k16 / k64 | 26.1 / 130.6 | 6.6 / 6.2 | 0.25 / 0.05 | 1.00 / 0.25 |
| three_nn | 6.67 | 1.33 | 0.20 | 0.75 |
| **radius** | 4.80 | 7.67 | **1.60** | 1.10 |
| **ball_query** | 1.93 | 13.05 | **6.76** | 1.10 |
| **group_points backward** | 3.22 | 8.45 | **2.62** | 1.20 |
| **three_interpolate backward** | 1.53 | 3.91 | **2.55** | 1.30 |
| **segment_csr sum / max** | 0.33 / 0.35 | 3.76 / 3.85 | **11.5 / 11.0** | 1.30 |
| **gather_csr** | 0.71 | 73.2 | **103** | 1.30 |
| scatter sum / mean / max / contended | ≈1.9–3.6 | ≈ same | ≤ 1.04 | 1.25 |
| **grid_cluster (4M points)** | 0.09 | **crash**: whole input in one block (numel > 1M) | – | – |

**What the run taught us, now folded back into the task:**
- **A per-query "whole example in one tile" kNN is fast.** GPT's ran at 0.25x CUDA. I re-tuned the reference kNN to many small programs (k16: 19.3 → 7.8 ms) and tightened the bars:
  - knn k16: 1.0 → 0.6;
  - pcnet train step: 0.8 → 0.65;
  - pcnet inference: 0.65 → 0.5.
- **GPT's single-block `grid_cluster` crashes above about 1M points.** I added a 3M-point hidden test. It exposed a second trap that I fixed in the reference: Triton's fp32 `/` is approximate and flips the cell of points on a cell boundary, so `tl.math.div_rn` is required to match CUDA's IEEE division.
