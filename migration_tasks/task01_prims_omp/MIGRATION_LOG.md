# Task 01: migration log (reference port, done by hand on an RTX 4090)

This records every failure I hit while porting `tsr-prims` from CUDA to OpenMP
target offload (clang 18, `--offload-arch=sm_89`, CUDA 12.8 driver 570). Each
entry names the symptom an agent would see and the hidden check that targets it.
These are the places a single-attempt frontier run is expected to fail.

Environment facts discovered on the way:
- A synchronous OpenMP `target` launch costs **~42 µs** (`nowait` + `taskwait`
  ~23 µs) vs a few µs for an async CUDA launch. Small-input performance can
  never match CUDA, so gates are measured at large n.
- GPU performance counters are not available in the Runpod container
  (`ncu`: ERR_NVGPUCTRPERM). Profiling has to rely on `LIBOMPTARGET_PROFILE`
  traces, timing and compiler remarks (`-Rpass=openmp-opt`).
- clang 18 has no `ompx` warp shuffles/ballots (only thread/block ids and
  `sync_block`); a portable port must replace warp intrinsics algorithmically.

## Correctness / build pitfalls (GPU-only: none appear with the host-offload plugin)

| # | Symptom | Root cause | Fix | Hidden check |
|---|---|---|---|---|
| 1 | test TU fails to compile: doctest `static_assert` "Exceptions are disabled" | with `-fopenmp-targets=nvptx64` every TU is also compiled for the device, where exceptions are off | offload compile flags PRIVATE to the library; consumers only get `-fopenmp` + link flags | hidden tests use exceptions (`CHECK_THROWS_AS`) and link the library target |
| 2 | binaries run but `omp_get_num_devices() == 0` / "no OpenMP offload device" | `-Xopenmp-target -march=sm_89` does not record the arch for the offload link | use `--offload-arch=sm_89` | grader builds tests from a clean tree with its own flags |
| 3 | still 0 devices; no `.llvm.offloading` section in the executable | the offload linker only extracts device code from **static archive** members referenced by other device code; tests/tools have none | build the library as an OBJECT (or shared) library | hidden tests are separate executables with no device code of their own |
| 4 | `nvlink: Undefined reference to ...kSortTileE` | namespace-scope `constexpr` ODR-used in device code (`std::min` binds a reference) is not emitted for the device | put device constants in `begin/end declare target` | build of the hidden suite on GPU |
| 5 | (performance) | locals whose address escapes into a call in a parallel region are globalized (heap-allocated per thread) | return results by value from team helpers | performance gate |
| 6 | `error: capturing a structured binding is not yet supported in OpenMP` | clang with `-fopenmp` rejects lambdas that capture structured bindings, even in host code of an offloading TU | `std::get` / named locals | hidden tests are compiled with the library's flags |
| 7 | GPU passes, host-offload device: wrong results, then SEGV | team helpers assumed `num_threads(256)` is honoured (`if (tid < 16)`, one element per thread); the host plugin grants far fewer threads per team | every per-thread loop strides over 256 *virtual lanes* (`for v = tid; v < 256; v += nthreads`) | `tests_host` gate (same suite on `x86_64-pc-linux-gnu`) |
| 8 | `fatal error: error in backend: Cannot select: AtomicStore<(store seq_cst ...)>` | the LLVM 18 NVPTX backend cannot lower a `seq_cst` atomic store (`#pragma omp atomic write seq_cst`) | `#pragma omp flush` + relaxed `atomic write` | GPU build of the single-kernel reduction |

## Performance attempts, first round (n = 16.7M, ms per call, RTX 4090)

| primitive | CUDA original | v1 naive port | v2 team kernels | v2b radix: padded table, no local arrays |
|---|---|---|---|---|
| exclusive_scan | 0.282 | 1.052 | 0.409–0.441 | |
| reduce_max | 0.024 | 0.882 | 0.100–0.114 | |
| histogram256 | 0.059 | 3.487 | 0.106–0.111 | |
| radix_sort_pairs (44-bit) | 7.135 | 40.98 | 20.8–21.2 | **10.92** |
| compact_indices | 0.382 | 1.366 | 0.646–0.717 | |

What made the difference:
- **reduce**: the OpenMP `reduction` clause + a separate store kernel + host
  round trip (v1) → two team-kernel passes with a shared-memory tree.
- **histogram**: one global atomic per element (v1) → per-team shared-memory
  histograms merged with one atomic per bin per team.
- **scan**: one device thread scanning each 256-element chunk serially (v1) →
  2048-element tiles staged in team memory and scanned by the whole team.
- **radix sort**: one device thread per 1024-key tile (v1) → team counting and
  a stable rank from a digit-major scan of a [digit][thread] table (v2);
  then (v2b) removing 16-way shared-memory bank conflicts (padding) and
  dynamically indexed per-thread arrays (local memory).

## Final measurements (n = 2^26, hidden `perf_bench`, median of 15, RTX 4090)

Each timed call includes a 1-element download, so asynchronous CUDA work is
counted. Ratio = port / CUDA, both measured in the same grading run.

| case | CUDA ms | v1 naive | ratio | v2 (two-kernel reduce) | ratio | **v3 (oracle)** | ratio | bar |
|---|---|---|---|---|---|---|---|---|
| exclusive_scan_u32 | 1.179 | 4.047 | 3.42 | 1.144 | 0.98 | 1.146 | **0.97** | 1.30 |
| inclusive_scan_u32 | 1.180 | 3.914 | 3.41 | 1.166 | 1.01 | 1.179 | **1.00** | 1.30 |
| reduce_sum_f32 | 0.293 | 3.286 | 11.2 | 0.438 | 1.56 | 0.397 | **1.36** | 1.60 |
| reduce_max_f32 | 0.299 | 3.398 | 11.4 | 0.439 | 1.51 | 0.397 | **1.33** | 1.60 |
| histogram256 | 0.447 | 14.90 | 33.5 | 0.438 | 1.00 | 0.442 | **0.99** | 1.30 |
| radix_sort_u64_44bit | 26.37 | 182.3 | 6.98 | 32.16 | 1.23 | 32.25 | **1.22** | 1.50 |
| radix_sort_u32_32bit | 15.46 | 110.3 | 7.19 | 19.51 | 1.27 | 19.40 | **1.25** | 1.50 |
| compact_indices | 2.030 | 4.870 | 2.50 | 2.354 | 1.16 | 2.353 | **1.16** | 1.45 |

v3 = v2 + single-kernel reduction (last team to finish folds the partials,
ticket counter in device memory) + 4 independent accumulators per thread.

Why the reduce bar is 1.60: the OpenMP floor on clang 18 is ~63 µs per call
(empty `target` launch ~37 µs + 4-byte `omp_target_memcpy` ~26 µs), and the best
plain OpenMP streaming read of 256 MB measured 0.375 ms (vs ~0.27 ms CUDA kernel).
v3 is at that floor. A tighter bar would make the oracle flaky.

How the bars were chosen: the oracle needs at least 15% headroom on every case,
because run-to-run noise is up to ~10% (histogram 0.45–0.49 ms). The
straightforward port must fail every case. The bars are applied to the best
median of 3 interleaved rounds of reference and candidate.

## Grader validation (native on the pod, `tests/grade.py`)

| candidate | static | build gpu | tests gpu | offload | build host | tests host | perf | reward |
|---|---|---|---|---|---|---|---|---|
| original CUDA repo (empty patch) | FAIL (510) | FAIL | – | – | FAIL | – | – | 0 |
| v1 straightforward port | pass | pass | pass | pass | pass | pass | FAIL (all 8 cases, 2.5–33x) | 0 |
| v2 GPU-only (assumes 256 threads) | pass | pass | pass | pass | pass | FAIL (crash) | – | 0 |
| host-sort cheat (download, hand-written LSD on host, upload, token kernel) | pass | pass | pass | FAIL (12.6 MB D2H) | pass | pass | – | 0 |
| v2 two-kernel reduce | pass | pass | pass | pass | pass | pass | pass | 1 |
| **v3 oracle (`solution/solution.patch`) via `tests/test.sh`** | pass | pass | pass | pass | pass | pass | pass | **1** |

The static gate also caught a stale reference patch: two `cudaDeviceSynchronize`
calls were left in the ported `apps/prims_bench.cpp`. The patch has since been
regenerated against the exact `repo/` tree.

The Docker image (`environment/Dockerfile`) was not built here. The Runpod
container has no user namespaces, no CAP_SYS_ADMIN and no /dev/fuse, so
rootless Docker/Podman cannot run. The image uses the same toolchain as the
native validation (Ubuntu 24.04, apt clang-18, CUDA 12.8).
