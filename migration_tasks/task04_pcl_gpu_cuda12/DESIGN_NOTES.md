# Task 04 design notes (not shown to the agent)

## Choice (2026-10-06)

Research found that legacy pre-Volta CUDA, modernized to CUDA 12.8 on sm_120, is the most realistic and best-gradeable legacy migration. Three CUDA 12 changes break such code:
- texture references were removed;
- implicit warp-synchronous code (`__ballot`/`__any` without a mask, `volatile` warp reductions) is unsafe under independent thread scheduling;
- legacy library APIs were removed.

The codebase is PCL's GPU stack (BSD-3), pinned at commit c47198828. Upstream CMake **disables** `gpu_kinfu`, `gpu_kinfu_large_scale` and `gpu_people` on CUDA >= 12 ("uses textures which was removed in CUDA 12"). Issues #5571 and #5995 have been open since 2023 with no PR.

**Hierarchy:**
- inside PCL: gpu_containers / gpu_utils (texture_binder, `__ballot` emulation) -> gpu_octree / gpu_features -> gpu_kinfu, gpu_kinfu_large_scale, gpu_people (vendored NVIDIA NCV Haar / NPP_staging);
- downstream consumers written for this task: `kinfu_replay` (synthetic depth sequences with ground-truth poses and an analytic scene SDF) and `people_eval` (body-part labelling of an RGB-D frame and its variants).

## Baseline

The original stack builds with CUDA 11.8 (the last CUDA with texture references) with `CUDA_ARCH_BIN=86`, which embeds compute_86 PTX. The RTX 5090 runs it through driver JIT (`environment/Dockerfile.legacy`, `install_legacy`).

The original therefore gives both the goldens and the performance baseline on the same GPU. Results:

- **kinfu orbit, 120 frames:** pose error 1.1 mm mean, mesh surface error 0.23 mm, 1.14M triangles, 1.6 ms per frame.
- **kinfuLS corridor, 240 frames:** 7 volume shifts, drift 1.5 cm mean over 6 m, 2.0 ms per frame.
- **people, 5 variants:** return 2, label histograms recorded, about 7 ms per frame.

## Harness notes

- `kinfuLS::setInitialCameraPose` only stores the pose (the constructor already called `reset()`), so pose errors are evaluated on motion relative to frame 0.
- The kinfuLS TSDF buffer is cyclical, so a raw marching-cubes mesh is wrapped. Grade it against goldens, not against the scene SDF.
- A straight corridor is degenerate for ICP. The walls and floor need relief, and a 0.8 m shift distance keeps tracking alive across shifts.
