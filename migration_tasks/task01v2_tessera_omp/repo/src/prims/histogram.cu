// Tessera - 256-bin histogram.
//
// After histogram256 from the cuda-samples "histogram" sample (BSD-3-Clause):
// every warp owns a private sub-histogram in shared memory to spread atomic
// contention, the sub-histograms are merged per block into partial
// histograms, and a second kernel (one block per bin) sums the partials.
// The input here is one bin index per element instead of four packed bytes.
#include <algorithm>

#include "tsr/prims/prims.h"
#include "tsr/prims/warp.cuh"

namespace tsr::prims
{
namespace
{
constexpr unsigned kHistWarps       = 6;
constexpr unsigned kHistThreads     = kHistWarps * kWarpSize;
constexpr unsigned kHistBlockMemory = kHistWarps * kHistogramBins;
constexpr unsigned kMaxPartials     = 240;
constexpr unsigned kMergeThreads    = 256;

__global__ void __launch_bounds__(kHistThreads)
    histogram256_kernel(std::uint32_t *__restrict__ partials, const std::uint32_t *__restrict__ bins, std::uint64_t n)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ std::uint32_t s_hist[kHistBlockMemory];
    std::uint32_t *s_warp_hist = s_hist + warp_id() * kHistogramBins;

    TSR_UNROLL
    for(unsigned i = 0; i < kHistBlockMemory / kHistThreads; ++i)
    {
        s_hist[threadIdx.x + i * kHistThreads] = 0;
    }
    cg::sync(cta);

    const std::uint64_t stride = static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
    for(std::uint64_t pos = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; pos < n; pos += stride)
    {
        const std::uint32_t b = bins[pos];
        if(b < kHistogramBins)
        {
            atomicAdd(s_warp_hist + b, 1u);
        }
    }
    cg::sync(cta);

    for(unsigned bin = threadIdx.x; bin < kHistogramBins; bin += kHistThreads)
    {
        std::uint32_t sum = 0;
        for(unsigned w = 0; w < kHistWarps; ++w)
        {
            sum += s_hist[bin + w * kHistogramBins];
        }
        partials[blockIdx.x * kHistogramBins + bin] = sum;
    }
}

// One block per bin: each block adds up the same bin from every partial.
__global__ void __launch_bounds__(kMergeThreads)
    merge_histogram256_kernel(std::uint32_t *__restrict__ hist, const std::uint32_t *__restrict__ partials, unsigned partial_count)
{
    cg::thread_block cta = cg::this_thread_block();
    std::uint32_t sum    = 0;
    for(unsigned i = threadIdx.x; i < partial_count; i += kMergeThreads)
    {
        sum += partials[blockIdx.x + i * kHistogramBins];
    }

    __shared__ std::uint32_t data[kMergeThreads];
    data[threadIdx.x] = sum;
    for(unsigned stride = kMergeThreads / 2; stride > 0; stride >>= 1)
    {
        cg::sync(cta);
        if(threadIdx.x < stride)
        {
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
    }
    if(threadIdx.x == 0)
    {
        hist[blockIdx.x] = data[0];
    }
}
} // namespace

void histogram256(const std::uint32_t *bins, std::size_t n, std::uint32_t *hist, Arena &arena, cudaStream_t stream)
{
    const unsigned partial_count = static_cast<unsigned>(std::clamp<std::size_t>(div_round_up<std::size_t>(n, kHistThreads * 16), 1, kMaxPartials));
    std::uint32_t *partials      = arena.allocate<std::uint32_t>(static_cast<std::size_t>(partial_count) * kHistogramBins);
    launch(histogram256_kernel, LaunchConfig{dim3(partial_count), dim3(kHistThreads), 0, stream}, partials, bins, static_cast<std::uint64_t>(n));
    launch(merge_histogram256_kernel, LaunchConfig{dim3(kHistogramBins), dim3(kMergeThreads), 0, stream}, hist, partials, partial_count);
}
} // namespace tsr::prims
