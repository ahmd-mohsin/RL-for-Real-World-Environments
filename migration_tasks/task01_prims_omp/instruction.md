# Port `tsr-prims` from CUDA to OpenMP target offload

`/app` contains `tsr-prims`, the device-wide parallel primitives library of the
Tessera engine: prefix scans, reductions, a 256-bin histogram, LSD radix sort,
a bitonic small-sort and stream compaction. It is written in CUDA. Without a
GPU it runs through a fiber-based CUDA emulator (`include/tsr/device/emu`,
`src/device/emu`). With `-DTSR_ENABLE_CUDA=ON` it compiles with nvcc.

Port the library to **OpenMP target offload** (clang 18, `-fopenmp`), keeping
its public API and its performance.

The machine has an NVIDIA RTX 4090 (sm_89), CUDA 12.8 (driver, nvcc, libdevice),
and clang 18 with the NVPTX and host offload plugins (`/usr/lib/llvm-18`).
`OMP_TARGET_OFFLOAD=MANDATORY` is set. Build the original with
`-DTSR_ENABLE_CUDA=ON` to measure the baseline you have to match.

## What the finished repository must look like

1. **No CUDA anywhere outside `tests/`.** Remove every `.cu`/`.cuh` file. Remove
   the CUDA language from CMake (no `enable_language(CUDA)`, `CUDAToolkit`,
   `cudart` or nvcc). Remove all CUDA syntax and runtime calls, and remove the
   emulator and its thread pool. Do not use vendor escape hatches: `ompx_*`,
   inline asm/PTX, or `__nvvm`/`__builtin_nvvm` builtins. Use portable OpenMP
   only. Comments are not checked.
2. **Every primitive computes on the device.** Each call to a primitive must launch
   at least one `target` region. The library may not compute results on the host.
   It may not use host algorithms (`std::sort`, `std::partial_sum`,
   `std::accumulate`, `std::reduce`, …) in `src/` or `include/`. A call may copy
   at most 64 bytes from the device to the host, which is enough to return a
   scalar such as `compact_indices`' count. Device memory is allocated with the
   OpenMP device memory API (`omp_target_alloc`), and pointers passed to
   primitives are device pointers.
3. **Same public API** (`include/tsr/prims/prims.h`, `include/tsr/device/*.h`):
   - Function names, template instantiations and argument order are unchanged.
     The only change: every `cudaStream_t` parameter becomes `tsr::Stream`, an
     opaque pointer type that accepts `nullptr`. Primitives may run
     synchronously.
   - `tsr::DeviceBuffer<T>` and `tsr::Arena` keep every public member, with the
     same `cudaStream_t` → `tsr::Stream` substitution. They also keep their
     semantics: move, shrink without reallocation, alignment and arena overflow
     blocks.
   - Errors throw `tsr::Error` exactly where the CUDA version does (invalid bit
     ranges, aliasing, oversize bitonic input, out-of-range downloads, …).
   - Results are bit-identical to the CUDA version. That includes sort stability
     and the overwrite semantics of `histogram256`. Float sums may differ only by
     reassociation.
4. **Build interface.** `cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++`
   must work with these options:
   - `TSR_OFFLOAD_ARCH`: GPU architecture, e.g. `sm_89`.
   - `TSR_OFFLOAD_TARGETS`: offload target triple, used when `TSR_OFFLOAD_ARCH`
     is empty, e.g. `x86_64-pc-linux-gnu`, the host-offload device.
   - `TSR_BUILD_TESTS` and `TSR_BUILD_APPS`.

   The library target stays `tsr_prims`. An executable that links `tsr_prims`
   and has no device code of its own must get everything it needs to offload,
   and must still be able to use C++ exceptions.
5. **Portable.** The same sources must be correct when offloading to the GPU
   and when offloading to the host device. Do not assume that the runtime grants
   the team size or thread count you ask for.

## How it is graded

`tests/` is replaced by a hidden suite. Your repository is then built from
scratch twice, for `TSR_OFFLOAD_ARCH=sm_89` and for
`TSR_OFFLOAD_TARGETS=x86_64-pc-linux-gnu`, with
`CMAKE_BUILD_TYPE=RelWithDebInfo`. The hidden suite contains:

- **Correctness tests** on both devices. They cover edge cases, golden
  checksums from the CUDA version, in-place operation, error paths and the
  memory utilities.
- **An offload probe** that counts the kernel launches and device-to-host bytes
  of every primitive call (rule 2).
- **A performance benchmark.** It times each case below at n = 2^26 against the
  original CUDA library on the same GPU in the same run. Each time is the median
  of 15 calls, and every timed call includes a 1-element download of its output.
  Your time must not exceed the given multiple of the CUDA time:

| case | max time / CUDA time |
|---|---|
| exclusive / inclusive scan, `uint32_t` | 1.30 |
| reduce (Sum, Max), `float` | 1.60 |
| `histogram256` | 1.30 |
| `radix_sort_pairs`, 44-bit `uint64_t` keys + `int32_t` values | 1.50 |
| `radix_sort_pairs`, 32-bit `uint32_t` keys + `uint32_t` values | 1.50 |
| `compact_indices` | 1.45 |

All gates must pass. A port that is correct but slow scores 0.
