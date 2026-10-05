# The emulated device backend

When device sources are compiled by a host C++ compiler (`TSR_ENABLE_CUDA=OFF`),
`tsr/device/backend.h` includes `tsr/device/emu/*` instead of the CUDA toolkit.
The emulated backend provides the same language surface: qualifiers
(`__global__`, `__device__`, `__shared__`, ...), built-in variables
(`threadIdx`, `blockIdx`, ...), synchronization (`__syncthreads`,
`__syncthreads_count/and/or`, `__syncwarp`), warp shuffles and votes, atomics,
fast-math intrinsics, a cooperative-groups subset and the runtime API calls
Tessera uses (`cudaMalloc`, `cudaMemcpyAsync`, streams, events, errors).

## Execution model

* A launch validates its configuration like the CUDA runtime (block size,
  grid limits, dynamic shared memory against `cudaFuncSetAttribute`) and
  records failures in the per-thread "last error".
* Thread blocks are distributed over `TSR_EMU_WORKERS` host threads (the
  launching thread is one of them). Blocks of one launch can run concurrently,
  so global-memory atomics are real atomics.
* Within a block, every CUDA thread is a **fiber** with its own guard-paged
  stack. Fibers are scheduled cooperatively on the worker and switch only at
  synchronization points. On x86-64 and AArch64 the switch is a few
  instructions of assembly (`src/device/emu/fiber_context.cpp`); elsewhere it
  falls back to `ucontext`.
* `__shared__` variables are `static thread_local`: a worker runs one block at
  a time and that block's fibers share the worker's storage. Dynamic shared
  memory (`TSR_DYNAMIC_SMEM`) is a per-worker buffer sized at launch.
* `__syncthreads` releases when every live thread of the block has arrived;
  threads that already returned do not participate.
* Warp intrinsics are "rounds" keyed by the participation mask, so sub-warp
  tiles (e.g. `cg::tiled_partition<16>`) proceed independently. A round
  completes when every live lane of its mask has arrived; each lane then reads
  the values published in that round. Lanes that do not exist (partial warps)
  or have exited count as absent.
* A block that can make no progress (e.g. a barrier inside divergent code)
  aborts with a description of what each thread is waiting on.

## Direct mode

`tsr::launch_linear` launches element-wise kernels in *direct* mode: threads
run to completion one after another on the worker stack, which is roughly 50x
cheaper than fibers. Any synchronization primitive in such a kernel is a fatal
error, so only use `launch_linear` for kernels without barriers or warp
intrinsics; everything else goes through `tsr::launch`.

## Differences from hardware

* Memory is host memory; there is no distinction between host and device
  pointers, so a missing copy can go unnoticed on this backend.
* Execution within a block is sequential between synchronization points, so
  intra-block data races that would corrupt results on a GPU may stay hidden.
* Floating-point intrinsics (`__expf`, `__logf`, ...) are the precise libm
  functions, so results can differ from GPU results in the last bits.
