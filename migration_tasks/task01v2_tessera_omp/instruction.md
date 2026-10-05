# Port Tessera from CUDA to OpenMP target offload

`/app` is **Tessera**, a hybrid Gaussian-splat and mesh renderer, written in CUDA. Per view it runs:
- **Surface:** mesh faces, then NDC transform, coarse binning and fine rasterization, then neural shading (hash-grid encoding and an MLP).
- **Splats:** EWA projection, SH colours, then tile binning (scan and sort).
- **Compositing:** the splats are blended over the surface tile by tile, then tone-mapped.

The library is `src/` and `include/tsr`. It ships its own device primitives (scans, reductions, histogram, radix and bitonic sorts, compaction), a CLI (`apps/tsr_render`) and a multi-view scheduler. Without a GPU it runs on a fiber-based CUDA emulator. With `-DTSR_ENABLE_CUDA=ON` it builds with nvcc, and large sorts then go through CUB.

We are dropping CUDA so that one code base can serve NVIDIA GPUs, other vendors' GPUs and a CPU fallback. **Port the whole renderer to OpenMP target offload** with clang 18 (`-fopenmp`). The port must keep its behaviour and its performance.

**The machine** has an RTX 4090 (sm_89), CUDA 12.8 (driver, nvcc and libdevice), and clang 18 with the NVPTX and host offload plugins (`/usr/lib/llvm-18`). `OMP_TARGET_OFFLOAD=MANDATORY` is set. Building the original with `-DTSR_ENABLE_CUDA=ON` gives you the reference to match.

## Requirements

1. **No CUDA and no emulator outside `tests/`.**
   - Remove every `.cu`/`.cuh` file, all CUDA syntax and runtime calls, and the CUDA language from CMake (no `enable_language(CUDA)`, `CUDAToolkit`, `cudart`, nvcc or CUB).
   - Remove the emulated backend and its fibers. The host thread pool used by `FrameScheduler` stays.
   - No vendor escape hatches: no `ompx_*`, no inline asm/PTX, no `__nvvm`/`__builtin_nvvm` builtins. Comments are not checked.
2. **Everything computes on the device.**
   - Every pipeline stage launches target regions on the device.
   - The library may not use host algorithms (`std::sort`, `std::partial_sum`, `std::accumulate`, `std::reduce`, …) in `src/` or `include/`.
   - A stage may copy at most 64 bytes from the device to the host per call. The exceptions are the results the API hands to the host: `tonemap`'s statistics, and a frame's images and statistics.
3. **Same public API.**
   - The headers keep their names, functions, types and members.
   - Every `cudaStream_t` becomes `tsr::Stream`, an opaque pointer type that accepts `nullptr`. Work may run synchronously.
   - Errors throw `tsr::Error` where the original does. Device runtime failures throw `tsr::DeviceError`, which derives from `tsr::Error`.
   - `TileBinning::keys` and `TileBinning::splat_ids` become scratch buffers with unspecified contents. All other fields keep their documented meaning.
4. **Same results as the original CUDA build.**
   - Frames and statistics (visible splats, intersections, covered surface pixels, exposure) must match the CUDA build. The only allowed differences come from floating-point rounding: a few rare pixels may differ by one or two 8-bit levels.
   - Binning keeps the original order: by tile, then depth, then splat index.
   - `make_synthetic_scene`, scene parsing and the procedural materials must produce exactly the same scenes as the original build.
   - Rendering is deterministic.
   - Renderers on different host threads, including `FrameScheduler` with any worker count, must work concurrently and produce the same frames as sequential rendering.
5. **Build interface.** `cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++` must work with these options:
   - `TSR_OFFLOAD_ARCH` (e.g. `sm_89`);
   - `TSR_OFFLOAD_TARGETS`, used when `TSR_OFFLOAD_ARCH` is empty (e.g. `x86_64-pc-linux-gnu`, the host-offload device);
   - `TSR_BUILD_TESTS` and `TSR_BUILD_APPS`.

   The library target stays `tessera`. An executable that links `tessera` and has no device code of its own must get everything it needs to offload, and must still be able to use C++ exceptions.
6. **Portable.** The same sources must be correct when offloading to the GPU and to the host device. Do not assume that the runtime grants the team size or thread count you ask for.

## Grading

`tests/` is replaced by a hidden suite. Your repository is built from scratch for `TSR_OFFLOAD_ARCH=sm_89` and for `TSR_OFFLOAD_TARGETS=x86_64-pc-linux-gnu`, with `CMAKE_BUILD_TYPE=RelWithDebInfo`. The suite has five parts:

- **Correctness tests on both devices:** the behaviour of every module, edge cases, large scenes, error paths, and concurrency.
- **Golden frames** rendered by the original CUDA build.
- **A static scan** for the rules above.
- **An offload probe** that counts the kernel launches and device-to-host bytes of every stage.
- **A benchmark** against the original CUDA build on the same GPU, in the same run. It uses 1080p scenes:
  - 1M splats;
  - a dense mesh with a neural material;
  - a hybrid of both;
  - 8 concurrent half-resolution views on 4 threads.

  Each time is the median of several calls. Every timed call includes a 1-element download of its output. Your time may not exceed the given multiple of the CUDA time:

| case | max time / CUDA time |
|---|---|
| full frame (`Renderer::render`): splats, mesh, hybrid | 1.25 |
| 8 views through `FrameScheduler` (4 workers) | 1.25 |
| `bin_splats` (splats, hybrid) | 1.40 |
| `shade_surface` (mesh, hybrid) | 1.30 |
| `rasterize_splats` (splats, hybrid) | 1.50 |
| `bin_mesh_faces`, `rasterize_mesh_fine` (mesh, hybrid) | 1.25 |

All gates must pass. A port that is correct but slow scores 0.
