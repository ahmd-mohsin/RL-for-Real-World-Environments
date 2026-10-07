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
