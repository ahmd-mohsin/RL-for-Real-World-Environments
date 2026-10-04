// Tessera - device-wide reductions.
//
// Two-pass reduction modelled on reduce7 from the cuda-samples "reduction"
// sample (BSD-3-Clause): a grid-stride loop accumulates several elements per
// thread, warps reduce with shuffles and one warp folds the per-warp partials.
// The first pass writes one partial per block, the second pass runs a single
// block over the partials.
//
// Differences from the sample: the operation is a template parameter, and the
// final warp reduction runs over the full warp with identity padding instead
// of a ballot-derived partial mask (shuffling from lanes outside the mask is
// undefined behaviour that the sample relies on).
#include <algorithm>

#include "tsr/prims/block.cuh"
#include "tsr/prims/prims.h"

namespace tsr::prims
{
namespace
{
constexpr unsigned kReduceThreads = 256;
constexpr unsigned kMaxPartials   = 512;

template<typename T, typename Op, bool kTwoPerThread>
__global__ void __launch_bounds__(kReduceThreads)
    reduce_kernel(const T *__restrict__ in, T *__restrict__ out, std::uint64_t n)
{
    __shared__ T s_warp[kReduceThreads / kWarpSize];
    const Op op{};
    T acc = Op::identity();

    if constexpr(kTwoPerThread)
    {
        // Each iteration consumes two blocks' worth of elements.
        std::uint64_t i               = static_cast<std::uint64_t>(blockIdx.x) * kReduceThreads * 2 + threadIdx.x;
        const std::uint64_t grid_size = static_cast<std::uint64_t>(kReduceThreads) * 2 * gridDim.x;
        while(i < n)
        {
            acc = op(acc, in[i]);
            if(i + kReduceThreads < n)
            {
                acc = op(acc, in[i + kReduceThreads]);
            }
            i += grid_size;
        }
    }
    else
    {
        std::uint64_t i               = static_cast<std::uint64_t>(blockIdx.x) * kReduceThreads + threadIdx.x;
        const std::uint64_t grid_size = static_cast<std::uint64_t>(kReduceThreads) * gridDim.x;
        while(i < n)
        {
            acc = op(acc, in[i]);
            i  += grid_size;
        }
    }

    acc = block_reduce<kReduceThreads>(acc, op, s_warp);
    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = acc;
    }
}

template<typename T, typename Op>
void reduce_with(const T *in, std::size_t n, T *out, Arena &arena, cudaStream_t stream)
{
    // Enough blocks to keep every worker busy, few enough that the second
    // pass is a single block.
    const unsigned blocks = static_cast<unsigned>(std::clamp<std::size_t>(div_round_up<std::size_t>(n, kReduceThreads * 8), 1, kMaxPartials));
    T *partials           = arena.allocate<T>(blocks);
    const bool pow2       = n > 0 && (n & (n - 1)) == 0;
    if(pow2)
    {
        launch(reduce_kernel<T, Op, true>, LaunchConfig{dim3(blocks), dim3(kReduceThreads), 0, stream}, in, partials, static_cast<std::uint64_t>(n));
    }
    else
    {
        launch(reduce_kernel<T, Op, false>, LaunchConfig{dim3(blocks), dim3(kReduceThreads), 0, stream}, in, partials, static_cast<std::uint64_t>(n));
    }
    launch(reduce_kernel<T, Op, false>, LaunchConfig{dim3(1), dim3(kReduceThreads), 0, stream}, partials, out, static_cast<std::uint64_t>(blocks));
}
} // namespace

template<typename T>
void reduce(const T *in, std::size_t n, T *out, ReduceOp op, Arena &arena, cudaStream_t stream)
{
    switch(op)
    {
    case ReduceOp::Sum: reduce_with<T, SumOp<T>>(in, n, out, arena, stream); break;
    case ReduceOp::Max: reduce_with<T, MaxOp<T>>(in, n, out, arena, stream); break;
    case ReduceOp::Min: reduce_with<T, MinOp<T>>(in, n, out, arena, stream); break;
    }
}

template void reduce<float>(const float *, std::size_t, float *, ReduceOp, Arena &, cudaStream_t);
template void reduce<std::uint32_t>(const std::uint32_t *, std::size_t, std::uint32_t *, ReduceOp, Arena &, cudaStream_t);
template void reduce<std::int32_t>(const std::int32_t *, std::size_t, std::int32_t *, ReduceOp, Arena &, cudaStream_t);
} // namespace tsr::prims
