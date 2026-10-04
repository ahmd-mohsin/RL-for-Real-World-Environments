// Tessera - emulated device backend: execution model
//
// Kernels are ordinary C++ functions. A launch runs every thread block of the
// grid on a pool of host workers; inside a block each CUDA thread is a fiber
// (a ucontext with its own stack) and the block's fibers are scheduled
// cooperatively on one worker. A fiber yields only at synchronization points
// (__syncthreads, warp shuffles/votes, __syncwarp), which is what makes shared
// memory, barriers and warp intrinsics behave as on hardware.
//
// Two scheduling modes exist:
//   Cooperative  every thread is a fiber; barriers and warp operations allowed.
//   Direct       threads run to completion one after another on the worker
//                stack. ~50x cheaper, but any synchronization primitive is a
//                fatal error. Used by launch_linear() for element-wise kernels.
//
// Blocks may execute concurrently on different workers, so global-memory
// atomics are real atomics. Set TSR_EMU_WORKERS=1 for fully deterministic
// execution (blocks run in index order on the launching thread).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>

#include "tsr/device/emu/vector_types.h"

namespace tsr::emu
{
enum class Mode : std::uint8_t
{
    Cooperative,
    Direct,
};

struct GridShape
{
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem = 0;
};

// Type-erased kernel invocation: invoke(closure) runs one CUDA thread.
using InvokeFn = void (*)(const void *closure);

// Validates the shape against device limits, then runs the whole grid and
// returns when every block has finished. Returns 0 on success or a
// cudaError_t value describing why the launch was rejected.
int run_grid(const GridShape &shape, Mode mode, InvokeFn invoke, const void *closure, const void *kernel_id);

// ---- per-thread context (valid only inside a kernel) ------------------------
struct ThreadContext
{
    uint3 thread_idx;
    uint3 block_idx;
    dim3 block_dim;
    dim3 grid_dim;
    unsigned linear_tid; // thread index linearized within the block
    unsigned lane;       // linear_tid % 32
    unsigned warp;       // linear_tid / 32
    void *block;         // BlockState*, opaque to device code
    Mode mode;
};

ThreadContext &current_thread();
bool in_kernel() noexcept;

void *dynamic_shared_memory();

// Block-wide barrier. The predicate variants reduce `pred` over every live
// thread of the block (dead threads do not participate).
void block_barrier();
int block_barrier_count(int pred);
int block_barrier_and(int pred);
int block_barrier_or(int pred);

// Warp-synchronous exchange: every lane in `mask` publishes `value`; once all
// live lanes of the mask have arrived, the 32 published values of this round
// are returned (slots of lanes that did not participate are unspecified).
// `participants`, if given, receives the lanes that took part in the round.
// It must be taken from here rather than recomputed afterwards: by the time a
// lane resumes, faster lanes of the same round may already have exited.
const std::uint64_t *warp_exchange(unsigned mask, std::uint64_t value, unsigned *participants = nullptr);
void warp_barrier(unsigned mask);
unsigned warp_live_mask();

// Launch-time attributes (mirrors cudaFuncSetAttribute for dynamic smem).
void set_max_dynamic_smem(const void *kernel_id, int bytes);
int max_dynamic_smem(const void *kernel_id);

// Emulated device description.
struct DeviceLimits
{
    int warp_size                 = 32;
    int max_threads_per_block     = 1024;
    int max_grid_x                = 0x7fffffff;
    int max_grid_yz               = 65535;
    int shared_mem_per_block      = 48 * 1024;
    int shared_mem_per_block_optin = 99 * 1024;
};
const DeviceLimits &limits();
int worker_count();

// Bit-level helpers for moving arbitrary trivially-copyable values (up to
// 8 bytes) through the warp exchange slots.
template<typename T>
inline std::uint64_t pack_bits(const T &v)
{
    static_assert(sizeof(T) <= 8 && std::is_trivially_copyable_v<T>, "warp exchange supports <= 8 byte types");
    std::uint64_t out = 0;
    std::memcpy(&out, &v, sizeof(T));
    return out;
}

template<typename T>
inline T unpack_bits(std::uint64_t bits)
{
    T out;
    std::memcpy(&out, &bits, sizeof(T));
    return out;
}

namespace detail
{
template<typename Kernel, typename Tuple>
struct Closure
{
    Kernel kernel;
    Tuple args;
};

template<typename C>
void invoke_closure(const void *p)
{
    const C &c = *static_cast<const C *>(p);
    // Each CUDA thread receives its own by-value copy of the parameters.
    std::apply(c.kernel, c.args);
}
} // namespace detail

// Launch `kernel` over `shape`. Arguments are converted to the kernel's
// parameter types once, exactly as the CUDA runtime marshals them.
template<typename... KArgs, typename... Args>
int launch(void (*kernel)(KArgs...), const GridShape &shape, Mode mode, Args &&...args)
{
    static_assert(sizeof...(KArgs) == sizeof...(Args), "kernel argument count mismatch");
    using Tuple   = std::tuple<std::decay_t<KArgs>...>;
    using ClosureT = detail::Closure<void (*)(KArgs...), Tuple>;
    ClosureT closure{kernel, Tuple(static_cast<std::decay_t<KArgs>>(std::forward<Args>(args))...)};
    return run_grid(shape, mode, &detail::invoke_closure<ClosureT>, &closure, reinterpret_cast<const void *>(kernel));
}

} // namespace tsr::emu
