# Bring PCL's GPU KinectFusion and people-detection modules to CUDA 12.8 (RTX 5090)

`/workspace` contains three repositories (each its own git repo):

| repo | what it is |
|---|---|
| `pcl` | The Point Cloud Library at upstream commit `c47198828`. Its GPU stack:<br>• `gpu/containers` and `gpu/utils`: device arrays, texture helpers, warp and block primitives<br>• `gpu/octree`, `gpu/features`, `gpu/surface`<br>• the applications `gpu/kinfu` (KinectFusion), `gpu/kinfu_large_scale` (KinectFusion with a moving cyclical volume) and `gpu/people` (random-forest body-part detection, including vendored NVIDIA NCV / NPP-staging primitives) |
| `kinfu_replay` | A downstream tool that runs KinFu and KinFu large-scale on synthetic depth sequences with known ground truth. It reports poses, TSDF statistics, meshes, extracted clouds and timing, and can run several trackers concurrently (`--concurrent`). |
| `people_eval` | A downstream tool that runs the people detector on an RGB-D frame and its variants (`people_eval`, with `--threads` for concurrent detectors), and exercises the NCV / NPP-staging primitives (`ncv_check`). |

The lab's new machine has an **RTX 5090 (Blackwell, sm_120)**, which needs **CUDA 12.8** or later, and that is the only toolkit installed (`/usr/local/cuda`).

On CUDA 12, PCL's CMake disables `gpu_kinfu`, `gpu_kinfu_large_scale` and `gpu_people` ("uses textures which was removed in CUDA 12"); see upstream issues #5571 and #5995. The code was written for pre-Volta GPUs.

**Make these three modules, and everything they depend on, build and run correctly on CUDA 12.8 / sm_120.** The two tools must keep working unchanged.

## Requirements

1. **Builds on CUDA 12.8.**
   - `pcl` configures and builds with CUDA 12.8 for sm_120, with `gpu_kinfu`, `gpu_kinfu_large_scale` and `gpu_people` enabled by default.
   - No texture or surface *references* remain anywhere in the GPU modules. You may use texture objects or other memory paths.
   - All device code (SASS and PTX) must target sm_120: do not lower the architecture, rely on JIT of an older target, or vendor another CUDA toolkit.
2. **Same results as the original pre-CUDA-12 code.** The original was built with CUDA 11.8, run on this GPU through PTX JIT, and recorded:
   - People detector body-part labels must match **exactly**, run with `OMP_NUM_THREADS=1`.
   - The NCV / NPP-staging primitives exercised by `ncv_check` must produce **bit-identical** buffers. That covers integral images (also from unaligned sources), decimation, rectangle std-dev, separable filters, resize and frame interpolation, plus compaction and transpose.
   - KinFu and KinFu large-scale must track and reconstruct as well as the original, measured against ground truth. Their TSDF volumes, meshes and extracted clouds must agree with the original's to within floating-point noise.
3. **Correct on independent thread scheduling.**
   - The modernized kernels must not depend on implicit warp-synchronous execution.
   - `compute-sanitizer` must report nothing on the KinFu, KinFu large-scale, people and NCV workloads with `--tool racecheck`, `--tool synccheck` and `--tool memcheck`.
4. **Safe for concurrent use.**
   - Several `KinfuTracker`, `kinfuLS::KinfuTracker`, `MarchingCubes` and `PeopleDetector` instances may run at the same time from different host threads.
   - Every result must then be bitwise identical to running each alone.
5. **Much faster than the original.** The lab moved to this GPU to run these pipelines faster. Measured on the same GPU against the original running through PTX JIT, each may take at most this fraction of the original's time:
   - per-frame KinFu and KinFu large-scale processing (`KinfuTracker::operator()`): 0.70×;
   - marching cubes (`MarchingCubes::run`): 0.55×;
   - zero-crossing cloud extraction (`TsdfVolume::fetchCloud`): 0.75×;
   - the people detector (`PeopleDetector::process`, with `OMP_NUM_THREADS=1`): 0.50×.

   The result requirements above still hold for the faster code.
6. **The tools are graded from their original sources.** Your changes to `kinfu_replay` and `people_eval` are discarded.

## Grading

The grader takes your diff of `/workspace/pcl` against its `base` tag and builds it from scratch with CUDA 12.8 (sm_120). It then builds the pristine `kinfu_replay` and `people_eval` against it and runs hidden tests that compare against results recorded with the original code. The tests cover:

- KinFu and KinFu large-scale on several trajectories and noise levels;
- the people detector on frame variants;
- the NCV primitives;
- concurrency;
- workloads drawn at grading time;
- compute-sanitizer race, sync and memory checks;
- the timing above (as measured by the tools).

All of it must pass.
