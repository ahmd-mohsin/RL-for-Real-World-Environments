# Task 04 migration log (not shown to the agent)

## What the reference migration had to fix

`solution/solution.patch` changes 31 files (+634 / −774) of PCL @ c47198828.

| area | legacy construct | CUDA 12 / sm_120 replacement | what breaks if done naively |
|---|---|---|---|
| CMake (kinfu, kinfu_large_scale, people) | modules disabled for CUDA >= 12 | enabled by default | modules not built |
| `gpu/containers/.../texture_binder.hpp` | `cudaBindTexture` on texture references | RAII `TextureObject<T>` (clamp, point, element, unnormalized; pitch2D / linear); the destructor syncs the stream before destroying | use-after-destroy with async launches |
| KinFu / KinFu LS marching cubes | `triTex`, `numVertsTex` texture references; a `__device__` global output counter; `__ballot`/`__any` with implicit masks | tables passed as pointers and read with `__ldg`; per-instance `DeviceArray<int> counter_` reset with memset, read back after the kernel; full-mask ballots, shuffle broadcast | concurrent `MarchingCubes` instances corrupt each other's counts |
| KinFu / KinFu LS `extract.cu` | volatile warp scan, global counter | shuffle scan, `__syncwarp`, bounds checks, per-call counter | racecheck hazards; wrong cloud under concurrency |
| block / warp reductions (`utils.hpp`, `estimate_combined.cu`, `gpu/utils` block / reduce / warp, `device.hpp` emulation) | warp-synchronous tails on `volatile` shared memory | shuffles in the same pairwise order (bitwise identical sums) | racecheck: millions of intra-warp hazards since Volta; different ICP sums if the order changes |
| people `multi_tree.cu`, `elec.cu`, `utils.cu` | texture references for depth, trees, edges, color map; `__constant__` threshold | texture objects as kernel parameters, threshold as a parameter | concurrent detectors overwrite each other's bindings and threshold |
| NCV `NPP_staging.cu` | 8u/32u/64u texture fetches with `alignmentOffset`; a texture path launched with NULL pointers (rectStdDev); 1D linear textures for the separable filters / supersampling, returning 0 out of range; a 2D linear-filtered texture for frame interpolation; a mirror / normalized 2D texture for bicubic resize | `__ldg` with the offset folded in; real pointers; a bounded `fetchLinear`; texture objects kept where the hardware filter or address mode is the semantics (blend: linear / clamp, 9-bit interpolation; bicubic: mirror / normalized) | interpolation no longer bit-identical (software lerp ≠ hardware 1.8 fixed point); bicubic edges differ (clamp ≠ mirror); memcheck errors or garbage reading past the end |
| NCV scans (`NPP_staging`, `NCVHaarObjectDetection`, `NCVAlg.hpp` subReduce) | warp-synchronous volatile scans and reductions | masked shuffles (mask derived from the scan width), read / `__syncwarp` / write phases | racecheck hazards in scanRows; wrong integral images for partial warps |

## Validation of the harness (RTX 5090, driver 580, CUDA 12.8.1)

- The original (CUDA 11.8, compute_86 PTX, JIT) and the oracle are deterministic run to run.
- The original runs from the verifier bundle on the 24.04 image with outputs bitwise identical to the native CUDA 11.8 container.
- **Oracle vs original:**
  - people labels: identical for all variants, including 5 `jitter-<seed>` variants;
  - NCV buffers: bit-identical for seeds 0 / 5 / 7 / 31337 / 123456;
  - KinFu: pose within float noise (orbit trans_err 1.127 vs 1.127 mm), TSDF / mesh / cloud within tolerance;
  - KinFu LS on the gallery scene: pose difference 0.1–0.2 mm and identical volume shifts on 8 walk variants.
- **The first KinFu LS scene was a corridor seen along its axis.** It was ICP-degenerate, and float-level differences between the two builds grew into 6–150 cm divergence. The original itself lost tracking at 480 frames. It was replaced by a gallery with hash-placed objects, walked sideways.

### Grading runs

The dense suite has these parts:
- 20 KinFu / KinFuLS workloads;
- 16 people variants;
- NCV with 12 seeds × 50 buffers;
- concurrency in 5 groups, including 2 lockstep stress groups that extract every frame, plus 6 people threads over 3 rounds;
- a fresh gate of 3 KinFu, 2 KinFuLS, 6 people and 4 NCV draws from a secret seed;
- sanitizer runs of racecheck, synccheck and memcheck on 4 workloads;
- perf over 3 rounds with one retry.

636 reference fingerprints are pinned.

| patch | reward | gates failed (why) |
|---|---|---|
| `solution.patch` (oracle) | **1.0** (37 min) | none. Perf ratios: KinFu frame 1.005, MC 0.43, cloud 0.81, KinFuLS frame 1.002, people 0.997 (two agents shared the GPU during this run) |
| empty patch | 0 | static (texture references); binary / consumers (modules not built) |
| warp fixes reverted (volatile warp-synchronous reductions / scans) | 0 | sanitizer only: racecheck finds 30 hazards in `TransformEstimatorKernel2` / `combinedKernel`. Every result is still bitwise right, so only racecheck catches it |
| process-global MC counter (`static int*` shared by instances) | 0 | concurrency only: the lockstep stress case crashes with an illegal address. The first suite, which extracted once per run, did not catch it |
| bicubic resize with clamp instead of mirror addressing | 0 | NCV + fresh (offset-ROI bicubic cases). The first suite, whole-image ROI only, did not catch it |
| `fetchLinear` without bounds (`__ldg` past the bound range) | 0 | NCV + fresh (guarded tiny-ROI filters) + memcheck (exact allocation). The first suite (zero padding inside the allocation) scored this **1.0** |

**Notes on the guard bands.** The guarded NCV source must start at a texture-aligned offset (1 KiB). With a 256 B offset, the original's `cudaBindTexture` rejects the pointer (`NCV_CUDA_ERROR`), and the statuses no longer match.

## Model runs on v1 (correctness bars, perf ≤ 1.10×) — single attempt, mini-swe-agent, no network

| model | steps / cost | reward | notes |
|---|---|---|---|
| GPT 6 Astra (`openai/gpt-6-astra`) | 71 / $11.08 | **1.0** (after the grader fix) | 4225-line patch. Bit-exact on everything, including 14 Haar configurations (texture-cached and non-atomic paths) and all `--api` checks. Perf ratios 1.00 / 0.99 / 0.86 / 1.00 / 1.04 |
| GPT 5.6 sol (`openai/gpt-5.6-sol`) | 128 / $3.63 | correct on every check run so far | 2523-line patch. Passes kinfu / people / ncv / concurrency / sanitizer and the new Haar and API checks |

The first grading of both runs returned 0 because of a flaky KinFuLS fresh check: a ~5% bifurcation of correct runs at volume-shift thresholds. The oracle produced byte-identical numbers on the same draw. The check now uses two tiers, calibrated on 80 natural and 40 induced bifurcations.

**Conclusion:** a correct legacy-CUDA migration is within reach of frontier models, so v2 adds speedup bars.

## v2: speedup bars and the optimized reference

The bars (fraction of the original's time; the oracle's result in parentheses):

| measure | bar | oracle |
|---|---|---|
| KinFu frame | 0.70 | 0.57 |
| KinFuLS frame | 0.70 | 0.58 |
| marching cubes | 0.55 | 0.43 |
| cloud | 0.75 | 0.61 |
| people | 0.50 | 0.33 |

Where the time went (oracle v1 profile on the 5090) and what v2 does:

- **KinFu frame (1.63 ms).**
  - The ICP `combinedKernel` cost 0.77 ms per frame: 19 iterations, each with 27 block-wide FP64 reductions and a host round trip (D2H, then an Eigen determinant and LLT solve).
  - v2 keeps the pose in device memory.
    - Warp sums run in FP32 (FP64 is 1/64 rate on GeForce), and per-block partials are accumulated in FP64.
    - A fused reduce + solve kernel follows: 27 blocks sum their rows in a fixed order, so it stays deterministic, and the last block runs a Cholesky solve (det = Π L_ii²) and the pose update.
    - There is one sync per frame, and the stream-ordered `cudaDeviceSynchronize` calls in maps and normals are gone.
  - Result: `combinedKernel` 41 → 8 µs, frame 1.63 → 0.93 ms.
  - Pitfall: a naive FP64 warp-shuffle reduction made the kernel *slower* (51 µs), and a single-thread solve with capped registers spilled (26 µs).
- **KinFuLS.** The same device ICP, with the volume-local translation offset (t − origin) and the original's loss threshold |det| < 1e5.
- **Cloud extraction (0.68 ms).**
  - The scan was latency-bound: 47% DRAM throughput and a 1.35-wave tail.
  - v2 carries the voxel value along z, takes the x+1 neighbour from the next lane, and splits each column's z range over 8 grid.z chunks. The point set is unchanged.
  - Result: 0.42 ms.
- **Marching cubes.** The original called `cudaGetDeviceProperties` on every run; removing it already gives 0.43.
- **People (40 ms).**
  - 88% of the time is the host `shs5` seeded hue region-grow, which is sequential and order-dependent (later seeds cannot pass through earlier seeds' points). Its window walk is skewed: from the second row on it starts one window width left of the box.
  - v2 keeps the exact index sequence and order, with structure-of-arrays split by index parity (so stride-2 runs are contiguous) and AVX2 tests of 8 candidates at a time.
  - It mirrors GCC's contracted arithmetic, `fma(dz, dz, fma(dx, dx, dy*dy))`, and turns the double compare into an equivalent float threshold.
  - Labels are bit-identical on 20 variants. Result: 40.6 → 13.3 ms.

### v2 validation (final suite)

| patch | reward | notes |
|---|---|---|
| `solution.patch` (oracle v2) | **1.0** (33 min) | Perf ratios: KinFu 0.569, MC 0.431, cloud 0.612, KinFuLS 0.584, people 0.34 |
| GPT 5.6 sol v1 patch, on the v1 suite | 1.0 | Like Astra, solved the correctness-only version |

809 reference fingerprints are pinned.

A Haar pitfall found during v2 validation: on some seeds the synthetic cascade exceeded the 16384-rect output capacity, and with atomic compaction the truncated set differs from run to run *in the original too*. The capacity is now 2^19, above the number of candidate windows over all scales.
