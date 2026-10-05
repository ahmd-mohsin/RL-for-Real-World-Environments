# GPT 5.6 sol single-attempt run: task01v2_tessera_omp

**Reward: 0.**

- **Harness:** mini-swe-agent 2.4.6, bash only, yolo mode, 350-step / $80 cap.
- **Sandbox:** fresh Runpod RTX 4090 pod from `ahmedmohsin/task01v2-tessera-omp:v1`. The hidden tests were uploaded only after submission.
- **Usage:** GPT submitted on its own after **95 steps, $3.03**, with a 6,632-line patch.
- **Audit:** no network commands; no access to the grader or hidden tests.

| gate | result |
|---|---|
| static | pass |
| build (GPU / host) | pass / pass |
| hidden tests, GPU | **63/64**. Failed `sort: 32-bit key instantiations…`: `radix_sort_pairs` no longer throws `tsr::Error` when the keys alias (in == out). |
| hidden tests, host device | **63/64** (same case) |
| golden frames vs CUDA | pass. It found and fixed the scene RNG evaluation-order trap. |
| offload probe | **FAIL**. `transform_faces` 84 B, `project_gaussians` 84 B, `shade_surface` 124 B device→host (budget 64): implicit `map(tofrom)` of the 84-byte `Camera` struct. |
| perf (measured separately; the grader skips it after a failed gate) | **FAIL, 12 of 14 cases** |

| case | CUDA ms | GPT ms | ratio | bar |
|---|---|---|---|---|
| splats frame | 32.5 | 116.3 | 3.58 | 1.25 |
| mesh frame | 15.3 | 38.8 | 2.55 | 1.25 |
| hybrid frame | 30.5 | 82.2 | 2.69 | 1.25 |
| 8 views / 4 threads | 30.9 | 97.6 | 3.16 | 1.25 |
| bin_splats splats / hybrid | 24.7 / 10.0 | 119.0 / 48.3 | 4.82 / 4.84 | 1.40 |
| shade_surface mesh / hybrid | 4.11 / 4.09 | 6.22 / 7.41 | 1.51 / 1.81 | 1.30 |
| rasterize_splats splats / hybrid | 0.78 / 0.46 | 2.31 / 1.19 | 2.98 / 2.62 | 1.50 |
| bin_mesh_faces mesh / hybrid | 1.45 / 0.83 | 18.9 / 15.0 | 13.0 / 18.0 | 1.25 |
| rasterize_mesh_fine mesh / hybrid | 2.40 / 2.13 | 1.16 / 0.82 | 0.48 / 0.39 | 1.25 (pass) |

**Takeaway:**
- GPT handled correctness well, including the hardest hidden trap: the compiler-dependent scene generation.
- It missed the error-path contract and the implicit-map transfer rule.
- It stopped at 95 of its 350 allowed steps without ever benchmarking against the stated bars.
- Its port is 2.5–18x slower than CUDA where the bars need 1.25–1.5x.
