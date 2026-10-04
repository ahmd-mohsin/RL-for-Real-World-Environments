# Tessera

Tessera is a hybrid real-time renderer that composites **3D Gaussian splats**
over **triangle meshes shaded by a neural material**, written in CUDA C++.

* Splats are projected with an EWA approximation, coloured with spherical
  harmonics, binned into screen tiles (two-pass intersection with exact
  ellipse/tile overlap), depth sorted (LSD radix sort or a bitonic network for
  short lists) and alpha-composited tile by tile.
* Meshes go through coarse binning and fine rasterization with
  perspective-correct barycentrics; the visible surface is shaded by a
  multiresolution hash-grid encoding feeding a small MLP.
* The composite stops each pixel's splat list at the mesh surface, then an
  automatic exposure / white-point pass tone maps the result to sRGB.
* Many views can be rendered concurrently, one renderer per worker thread.

Device code also runs **without a GPU**: when compiled by a host C++ compiler,
the `.cu` sources build against Tessera's emulated backend, which executes
every CUDA thread as a lightweight fiber so that barriers, shared memory, warp
shuffles/votes, cooperative groups and atomics behave as on hardware. The test
suite runs on both backends.

## Building

```bash
# CPU (emulated device) build, default
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure

# CUDA build (requires the CUDA toolkit)
cmake -S . -B build-cuda -G Ninja -DTSR_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda
```

Requirements: CMake >= 3.24, a C++20 compiler (GCC 12+ or Clang 16+). The
emulated backend supports Linux on x86-64 and AArch64.

## Rendering

```bash
./build/apps/tsr_render scenes/orbit.tsr --stats            # writes frame_<camera>.ppm
./build/apps/tsr_render scenes/orbit.tsr --camera 2 --hdr    # also writes a .pfm
./build/apps/tsr_render --synthetic 3 --gaussians 5000 --size 320 240 --threads 4
```

The scene format is described in [docs/scene_format.md](docs/scene_format.md).

## Layout

```
include/tsr/device   backend selection, launch API, memory, emulated backend
include/tsr/math     vector and small-matrix math
include/tsr/prims    scan, reduce, histogram, radix/bitonic sort, compaction
include/tsr/geom     camera, Gaussian projection, mesh transform, NDC helpers
include/tsr/bin      splat tile binning, mesh coarse binning
include/tsr/raster   splat compositing, mesh fine rasterization
include/tsr/encode   hash-grid encoding, GEMM, MLP
include/tsr/render   surface shading, the renderer
include/tsr/post     exposure and tone mapping
include/tsr/sched    host thread pool, multi-view scheduler
src/                 implementations (.cu files build on both backends)
apps/                tsr_render
tests/               doctest suite with host reference implementations
```

See [docs/architecture.md](docs/architecture.md) for the data flow and
[docs/emulated_backend.md](docs/emulated_backend.md) for how kernels run on
the CPU.

## Environment variables

| Variable | Effect |
|---|---|
| `TSR_EMU_WORKERS` | worker threads for the emulated backend (default `min(8, cores)`; `1` = fully deterministic block order) |
| `TSR_EMU_STACK_KB` | per-fiber stack size of the emulated backend (default 128) |
| `TSR_SYNC_LAUNCHES` | synchronize after every launch so errors are attributed to the right kernel |

## License

Tessera is derived in part from open-source projects; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
