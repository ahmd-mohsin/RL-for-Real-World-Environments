# single_task: a synthetic CUDA repository for long-horizon tasks

`repo/` is **Tessera**, a synthetic hybrid renderer assembled from five
open-source CUDA/C++ projects. Their code was extracted file by file, rewritten
against shared abstractions and wired into one pipeline, so no single upstream
repository resembles it. Tasks (issues + hidden tests) will be written against
this repository. The repository is the environment; the task comes next.

## Why this repository is hard

* **Dense CUDA everywhere**: shared-memory tiling, warp shuffles and ballots,
  cooperative groups, block barriers with early exit, atomics, dynamic shared
  memory, multi-pass device algorithms.
* **Deep cross-module dependencies**: a symptom in the final image can
  originate in the scan, the sort, the tile key encoding, the NDC convention,
  the mesh bins, the hash grid, the arena allocator, the fiber scheduler or
  the thread pool. Every stage feeds the next (see the diagram below).
* **Two coordinate conventions meet** (pixel space for splats, PyTorch3D-style
  flipped non-square NDC for meshes) and must agree to the sub-pixel.
* **Two execution backends**: real CUDA (nvcc) and an emulated backend that
  runs kernels on the CPU with per-thread fibers and hand-written x86-64 /
  AArch64 context switching. Correctness depends on both.
* **Size-dependent code paths**: e.g. intersection lists of at most 1024 use a
  bitonic network, larger ones a 4-bit LSD radix sort with ballot-based
  ranking, so behaviour can change with scene size.

## Provenance

| Upstream (license) | What was taken | Where it lives in Tessera |
|---|---|---|
| gsplat (Apache-2.0) | EWA projection, SH, tile intersection (SnugBox/AccuTile), tile offsets, tile rasterizer | `geom/gaussian_math.cuh`, `geom/projection.cu`, `bin/tile_intersect.cu`, `raster/splat_raster.cu` |
| PyTorch3D (BSD-3) | non-square NDC, coarse bit-mask binning, fine rasterizer, barycentric helpers | `geom/mesh_ndc.cuh`, `bin/mesh_bins.cu`, `raster/mesh_raster.cu` |
| NVIDIA cuda-samples (BSD-3) | helper_math, scan, reduction, histogram256, bitonic sort, matrixMul | `math/vec.h`, `prims/*`, `encode/mlp.cu` |
| tiny-cuda-nn (BSD-3) | multiresolution hash-grid encoding | `encode/hashgrid.cu` |
| BS::thread_pool (MIT) | thread pool, blocks, multi_future | `sched/thread_pool.h` |

New code written for Tessera: the emulated device backend (`device/emu/*`,
fibers, warp rounds, runtime API), the launch/memory layer, the 64-bit
generalized scan, the radix sort, the matrix library, the hybrid surface
compositing, the neural surface shading, tone mapping, the scene format, the
renderer, the multi-view scheduler, the CLI and the whole test suite (with
independent host reference implementations).

Upstream sources at the exact commits used can be re-fetched with
`scripts/fetch_upstream.sh` (into `upstream/`, git-ignored). Derived files
keep attribution headers, as the BSD and Apache licenses require.

## Data flow

```
scene ─► gaussians ─► projection ─► SH colours ─► tile binning ──────────────┐
                        (EWA)                     count → scan → keys →       │
                                                  sort (bitonic | radix) →    │
                                                  tile offsets                ▼
      ─► meshes ─► NDC transform ─► coarse bins ─► fine raster ─► neural ─► surface ─► tile compositing ─► tonemap ─► image
                                   (bit masks)    (top-K faces)   shading   layer      (stops at surface)   (reduce,
                                                                  (compact,                                  histogram,
                                                                   hash grid,                                scan)
                                                                   GEMM MLP)
all kernels ─► tsr::launch ─► CUDA  |  emulated backend (fibers on a thread pool)
```

## Numbers

* ~12k lines of C++/CUDA (excluding the vendored doctest), 70 files
* 49 test cases / ~30k assertions, run in < 1 s on the emulated backend
* verified: GCC 13 on AArch64 and x86-64 (emulated backend, all tests pass);
  nvcc 12.6 for sm_80 (CUDA backend compiles and links)

## Layout

```
single_task/
├── repo/                 the Tessera repository (becomes /testbed in the sandbox)
├── env/Dockerfile        task image: repo at /testbed, built + tested, single "base" commit
├── env/Dockerfile.dev    development image (GCC 13, CMake, Ninja)
├── env/Dockerfile.cuda   nvcc compile check (no GPU needed)
└── scripts/fetch_upstream.sh
```

## Commands

```bash
# build and test inside the dev image
docker build -t tsr-dev -f env/Dockerfile.dev env
docker run --rm -v "$PWD/repo":/src -w /src tsr-dev bash -c \
  "cmake -S . -B /tmp/b -G Ninja && cmake --build /tmp/b && ctest --test-dir /tmp/b --output-on-failure"

# task image
docker build -t rlsandbox/tessera:base -f env/Dockerfile .

# CUDA compile check
docker build -t tsr-cuda -f env/Dockerfile.cuda env
docker run --rm -v "$PWD/repo":/src -w /src tsr-cuda bash -c \
  "cmake -S . -B /tmp/b -G Ninja -DTSR_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 && cmake --build /tmp/b"
```
