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

## Grading design

- **References are computed at grading time** by the original build, never stored as goldens.
  - `environment/Dockerfile.verifier` builds the original PCL with CUDA 11.8 for compute_86 in a builder stage, plus the consumers built against it.
  - It copies them, with every non-glibc shared library they load (167 MB), into `/opt/legacy_rt` of the task image, root-only (0700).
  - The agent image never contains them. The bundle runs on the Ubuntu 24.04 / CUDA 12.8 image through driver JIT and gives results bitwise identical to the native CUDA 11.8 container.
  - `tests/fingerprints.json` pins the reference outputs (people label and NCV buffer hashes, key KinFu statistics), so a drifting verifier image is detected rather than silently grading against something else.
- **Candidate code never runs as root.**
  - CMake, the compiler and every consumer run linked against the candidate's libraries execute as uid 65534. That user cannot read the references (0700).
  - After each step the produced tree (install, build, consumer binaries, every output directory) is handed back to root, read-only. Later candidate code therefore cannot rewrite earlier outputs, for example to fake the concurrency comparison.
- **Seeded inputs for the fresh gate.** Workloads are drawn from a secret seed at grading time:
  - kinfu_replay `--noise`/`--frames`;
  - people_eval `jitter-<seed>` variants (seeded scale noise, a seeded hole, a one-pixel shift);
  - ncv_check `--seed` (data, image sizes and the unaligned source offset).

  Seeds 0/7/123456 and 5 jitter seeds were verified bitwise identical between the original and the oracle.
- **Sanitizer workloads are kept short.** Racecheck costs about 25 s per KinFu frame on the 5090.
  - KinFu runs 6 frames of the 120-frame orbit, with mesh, cloud and ray cast.
  - KinFuLS runs 10 frames of the walk with `--shift 0.1`, which shifts the cyclical volume twice and extracts slices to the world model.
  - People runs 2 variants; NCV runs all primitives.
- **Device code check.**
  - PCL's default build embeds sm_120 PTX next to the SASS, so PTX is accepted only for sm_120/compute_120.
  - Anything else (older SASS or PTX, which JIT would cover) fails, as do a CUDA 11 runtime/NPP in NEEDED and a foreign RUNPATH.
