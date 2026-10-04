// Tessera - device backend selection.
//
// Every device translation unit includes this header first. Under nvcc it
// pulls in the CUDA toolkit; under a host compiler it pulls in the emulated
// backend, which provides the same language surface (qualifiers, built-ins,
// intrinsics, cooperative groups and runtime API).
//
// Kernels are launched through tsr::launch()/tsr::launch_linear() rather than
// the triple-chevron syntax so that both backends share one call site.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#if defined(__CUDACC__)
#    define TSR_BACKEND_CUDA 1
#    include <cooperative_groups.h>
#    include <cooperative_groups/reduce.h>
#    include <cuda_runtime.h>
#else
#    define TSR_BACKEND_EMU 1
#    include "tsr/device/emu/cooperative_groups.h"
#    include "tsr/device/emu/cuda_runtime.h"
#    include "tsr/device/emu/intrinsics.h"
#endif

#include "tsr/device/error.h"

namespace cg = cooperative_groups;

#define TSR_HD __host__ __device__
#define TSR_HDI __host__ __device__ __forceinline__
#define TSR_DI __device__ __forceinline__

#if defined(TSR_BACKEND_CUDA)
#    define TSR_DYNAMIC_SMEM(T, name) extern __shared__ T name[]
#    define TSR_UNROLL _Pragma("unroll")
#    define TSR_NO_UNROLL _Pragma("unroll 1")
#else
#    define TSR_DYNAMIC_SMEM(T, name) T *name = static_cast<T *>(::tsr::emu::dynamic_shared_memory())
#    define TSR_UNROLL
#    define TSR_NO_UNROLL
#endif

namespace tsr
{
struct LaunchConfig
{
    dim3 grid;
    dim3 block;
    std::size_t shared_bytes = 0;
    cudaStream_t stream      = nullptr;
};

constexpr std::uint32_t kLinearBlockSize = 128;

template<typename T>
constexpr TSR_HD T div_round_up(T value, T divisor)
{
    return (value + divisor - 1) / divisor;
}

// Launch a kernel that may use any CUDA feature (barriers, shared memory,
// warp intrinsics). Launch failures are reported through check_launch().
template<typename... KArgs, typename... Args>
inline void launch(void (*kernel)(KArgs...), const LaunchConfig &cfg, Args &&...args)
{
#if defined(TSR_BACKEND_CUDA)
    kernel<<<cfg.grid, cfg.block, cfg.shared_bytes, cfg.stream>>>(static_cast<KArgs>(std::forward<Args>(args))...);
#else
    const int status = ::tsr::emu::launch(
        kernel,
        ::tsr::emu::GridShape{cfg.grid, cfg.block, cfg.shared_bytes},
        ::tsr::emu::Mode::Cooperative,
        std::forward<Args>(args)...
    );
    if(status != 0)
    {
        ::tsr::emu::set_last_error(static_cast<cudaError_t>(status));
    }
#endif
    check_launch(cfg.stream);
}

// Launch an element-wise kernel over n_elements threads in 1D blocks of
// kLinearBlockSize. Such kernels must not synchronize: the emulated backend
// runs them without per-thread fibers.
template<typename... KArgs, typename... Args>
inline void launch_linear(void (*kernel)(KArgs...), std::uint32_t n_elements, cudaStream_t stream, Args &&...args)
{
    if(n_elements == 0)
    {
        return;
    }
    const dim3 grid{div_round_up(n_elements, kLinearBlockSize), 1, 1};
    const dim3 block{kLinearBlockSize, 1, 1};
#if defined(TSR_BACKEND_CUDA)
    kernel<<<grid, block, 0, stream>>>(static_cast<KArgs>(std::forward<Args>(args))...);
#else
    const int status = ::tsr::emu::launch(
        kernel, ::tsr::emu::GridShape{grid, block, 0}, ::tsr::emu::Mode::Direct, std::forward<Args>(args)...
    );
    if(status != 0)
    {
        ::tsr::emu::set_last_error(static_cast<cudaError_t>(status));
    }
#endif
    check_launch(stream);
}

} // namespace tsr
