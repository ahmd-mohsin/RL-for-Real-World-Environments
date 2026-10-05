# Task 01 v2: Tessera CUDA → OpenMP offload, reference port log (RTX 4090)

Task 01 v1 (`task01_prims_omp`, 6 primitives) was solved by Fable 5.1 on its first attempt. v2 applies the research findings in `plan.md` Step 5:
- whole-repo scope (85 files, 16 modules);
- a CUB-strength baseline;
- a ticket-style spec with no trap hints;
- adversarial hidden tests;
- golden frames from the original build.

## Making the original a correct, strong baseline

The CUDA build had only ever been compiled, never run on a GPU. Running it at 1080p on the 4090 exposed three bugs:

| # | Symptom on GPU | Cause | Fix in `repo/` |
|---|---|---|---|
| 1 | `raster: four pixels per thread` fails (864 pixels) | the test compared variants for exact equality; nvcc contracts the weight evaluation into FMAs differently per variant (1 ulp) | 1e-6 tolerance |
| 2 | `bin_mesh_faces` launch fails at 1080p (invalid argument) | coarse bit mask needs n_bins × 512 / 8 = 130 KB of dynamic shared memory | adaptive chunk size within 96 KB + opt-in attribute |
| 3 | `GEMM with 1132082 rows exceeds the grid limit` | rows mapped to `grid.y` (65,535 max) | rows on `grid.x` |
| 4 | (weak baseline) `bin_splats` 51 ms at 1M splats, 105.7M intersections | hand-written 4-bit radix | CUB `DeviceRadixSort` on real CUDA builds: 25 ms (18.4 ms sort, 5.7 G pairs/s) |

## Pitfalls found during the reference port (the hidden checks target them)

| # | Pitfall | Where it bites | Hidden check |
|---|---|---|---|
| 1 | **C++ argument evaluation order.** `make_float3(rng(), rng(), rng())` is unspecified; GCC (nvcc's host compiler) evaluates right to left, clang left to right. Every synthetic scene differs (x and z swapped), so the port never matches the CUDA build. | `make_synthetic_scene` (host code!) | golden frames; the spec requires identical scenes |
| 2 | Bin overflow drops faces in atomic order, so frames are nondeterministic even on CUDA | `rasterize_coarse` with dense meshes | golden settings avoid overflow and assert it; concurrency tests too |
| 3 | Global device state in a reduction (ticket counter) races when renderers run on several host threads | `prims::reduce` used by every frame | 16 threads × 32 views vs sequential; independent renderers on raw threads (caught on the host device) |
| 4 | Implicit `map(tofrom)` of non-scalar captures (`float3` camera position, bounds, tint) copies them back after every launch | `shade_surface` reached 68 B D2H (budget 64) | offload probe per stage |
| 5 | No dynamic team memory in portable OpenMP: the 96 KB bit mask must become a fixed 32 KB array with an adaptive chunk | coarse binning | golden frames at 1080p, bin-size variants |
| 6 | Lane state of the splat rasterizer persists across batches; with fewer threads than requested (host device) it must live in team memory | `rasterize_splats` | whole suite on the host-offload device |
| 7 | `__syncthreads_count` early exit: a reused team counter races unless double-buffered and ordered by barriers | `rasterize_splats` | tile variants 4/8/16 + 4 px/thread golden frames |
| 8 | clang's NVPTX math wrappers already define `min(int,int)`/`max` and `rsqrtf` as constexpr; redefining them fails, and the host pass lacks them | headers | build on both devices |
| 9 | CUDA fast-math intrinsics (`__logf`, `__expf`) compile in the device pass but are undefined on the host (link error) | projection, tonemap | build on both devices |
| 10 | Structured bindings captured by lambdas are rejected by clang with `-fopenmp` (again) | tests | the hidden suite is compiled with the library's flags |
| 11 | The original uploads two host-built −1 arrays per fine-raster call (16 MB at 1080p) | `rasterize_mesh_fine` | (performance: the port is 2.7x faster) |

## Performance (1080p, best of 2–3 interleaved rounds, ms)

| case | CUDA + CUB | straightforward port | ratio | reference port | ratio | bar |
|---|---|---|---|---|---|---|
| splats frame (1M splats) | 32.7 | 66.5 | 2.04 | 34.8 | 1.07 | 1.25 |
| mesh frame | 15.4 | 22.2 | 1.44 | 15.4 | 1.01 | 1.25 |
| hybrid frame | 24.2 | 43.3 | 1.41* | 26.8 | 1.10 | 1.25 |
| 8 views / 4 threads | 34.8 | 54.8 | 1.65 | 37.1 | 1.07 | 1.25 |
| `bin_splats` splats | 24.7 | 56.1 | 2.27 | 22.9 | 0.93 | 1.40 |
| `bin_splats` hybrid | 10.0 | 24.7 | 2.47 | 11.3 | 1.13 | 1.40 |
| `shade_surface` mesh | 4.04 | 6.01 | 1.47 | 3.25 | 0.80 | 1.30 |
| `shade_surface` hybrid | 4.11 | 7.01 | 1.70 | 3.13 | 0.76 | 1.30 |
| `rasterize_splats` splats / hybrid | 0.77 / 0.45 | 0.98 / 0.61 | 1.27 / 1.34 | 0.95 / 0.60 | 1.24 / 1.34 | 1.50 |
| `bin_mesh_faces` / `rasterize_mesh_fine` | 1.46 / 2.41 | | ≤ 0.68 | | ≤ 0.58 | 1.25 |

\* that run's CUDA hybrid frame was 30.6 ms.

What the reference port needed to pass:
- **Depth-first binning.** Stable-sort the n splats by depth, which costs O(n²) ranking below 8K. Emit intersections in depth order as 32-bit tile ids. Stable-sort on the 13 tile bits only (4 passes instead of 12 over 64-bit keys). Rebuild the 64-bit keys by gathering. The final order is identical to CUB's (tile, depth, id). It is faster than CUB.
- **Fully fused MLP** (one thread per row; weights in team memory; activations in registers via templated widths). 2.6x faster than the GEMM chain, with the same summation order.
- **Encoder:** one thread per point covering all levels, instead of a `collapse(2)` loop with a 64-bit index split. That was 2.3x.
- **Tonemap statistics:** reduction clause plus one team-local histogram mapped `tofrom` over the host stats array (10 launches/downloads down to 3).

## Grader validation (`tests/grade.py`, native on the pod)

| candidate | static | GPU tests | offload | host tests | perf | reward |
|---|---|---|---|---|---|---|
| original CUDA repo | FAIL (577) | build fails | – | build fails | – | 0 |
| straightforward port | pass | FAIL (golden: scene order) | FAIL (68 B) | FAIL | (8/14 cases over the bar) | 0 |
| straightforward port + RNG fix | pass | pass | FAIL (68 B) | pass | (8/14 cases over the bar) | 0 |
| reference port with a global reduce ticket | pass | pass | pass | FAIL (concurrency) | – | 0 |
| **reference port (`solution/solution.patch`)** | pass | 64/64 | pass | 64/64 | 14/14 | **1** |

Hidden suite: 64 test cases with 31,085 assertions, run on both devices. One grading run takes about 4 minutes.
