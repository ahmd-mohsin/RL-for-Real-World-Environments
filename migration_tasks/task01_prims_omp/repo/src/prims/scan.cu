// Tessera - device-wide prefix sums.
//
// Hierarchical scan after the cuda-samples "scan" sample (BSD-3-Clause):
//   1. scan_tiles_kernel     scans 1024-element tiles (256 threads x 4 items)
//                            and records each tile's total
//   2. (recursion)           exclusive scan of the tile totals
//   3. uniform_add_kernel    adds each tile's offset to its elements
// Unlike the sample, lengths need not be powers of two: the tail tile is
// zero-padded on load and masked on store.
#include "tsr/prims/block.cuh"
#include "tsr/prims/prims.h"

namespace tsr::prims
{
namespace
{
constexpr unsigned kScanThreads = 256;
constexpr unsigned kScanItems   = 4;
constexpr unsigned kScanTile    = kScanThreads * kScanItems;

// `in` and `out` may alias (in-place scans), so they are not __restrict__.
template<typename T>
__global__ void __launch_bounds__(kScanThreads)
    scan_tiles_kernel(const T *in, T *out, T *__restrict__ tile_totals, std::uint64_t n, bool inclusive)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ T s_data[2 * kScanThreads];

    const std::uint64_t base = static_cast<std::uint64_t>(blockIdx.x) * kScanTile + threadIdx.x * kScanItems;
    T items[kScanItems];
    T original[kScanItems];
    TSR_UNROLL
    for(unsigned k = 0; k < kScanItems; ++k)
    {
        items[k]    = base + k < n ? in[base + k] : T(0);
        original[k] = items[k];
    }

    T tile_total;
    scan4_exclusive(items, s_data, kScanThreads, cta, tile_total);

    TSR_UNROLL
    for(unsigned k = 0; k < kScanItems; ++k)
    {
        if(base + k < n)
        {
            out[base + k] = inclusive ? items[k] + original[k] : items[k];
        }
    }
    if(tile_totals != nullptr && threadIdx.x == 0)
    {
        tile_totals[blockIdx.x] = tile_total;
    }
}

template<typename T>
__global__ void __launch_bounds__(kScanThreads)
    uniform_add_kernel(T *__restrict__ data, const T *__restrict__ tile_offsets, std::uint64_t n)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ T buf;
    if(threadIdx.x == 0)
    {
        buf = tile_offsets[blockIdx.x];
    }
    cg::sync(cta);

    const std::uint64_t base = static_cast<std::uint64_t>(blockIdx.x) * kScanTile + threadIdx.x * kScanItems;
    TSR_UNROLL
    for(unsigned k = 0; k < kScanItems; ++k)
    {
        if(base + k < n)
        {
            data[base + k] += buf;
        }
    }
}

template<typename T>
void scan_impl(const T *in, T *out, std::size_t n, bool inclusive, Arena &arena, cudaStream_t stream)
{
    if(n == 0)
    {
        return;
    }
    const std::uint64_t tiles = div_round_up<std::uint64_t>(n, kScanTile);
    TSR_CHECK(tiles <= 0x7fffffffu, "scan of " << n << " elements exceeds the grid limit");
    T *tile_totals = tiles > 1 ? arena.allocate<T>(tiles) : nullptr;

    launch(
        scan_tiles_kernel<T>,
        LaunchConfig{dim3(static_cast<unsigned>(tiles)), dim3(kScanThreads), 0, stream},
        in,
        out,
        tile_totals,
        static_cast<std::uint64_t>(n),
        inclusive
    );
    if(tiles > 1)
    {
        scan_impl(tile_totals, tile_totals, tiles, false, arena, stream);
        launch(
            uniform_add_kernel<T>,
            LaunchConfig{dim3(static_cast<unsigned>(tiles)), dim3(kScanThreads), 0, stream},
            out,
            tile_totals,
            static_cast<std::uint64_t>(n)
        );
    }
}
} // namespace

template<typename T>
void exclusive_scan(const T *in, T *out, std::size_t n, Arena &arena, cudaStream_t stream)
{
    scan_impl(in, out, n, false, arena, stream);
}

template<typename T>
void inclusive_scan(const T *in, T *out, std::size_t n, Arena &arena, cudaStream_t stream)
{
    scan_impl(in, out, n, true, arena, stream);
}

template<typename T>
T exclusive_scan_total(const T *in, T *out, std::size_t n, Arena &arena, cudaStream_t stream)
{
    if(n == 0)
    {
        return T(0);
    }
    // Read the last input before the scan: with in == out it is overwritten.
    T last_in = T(0);
    TSR_CUDA_CHECK(cudaMemcpyAsync(&last_in, in + n - 1, sizeof(T), cudaMemcpyDeviceToHost, stream));
    TSR_CUDA_CHECK(cudaStreamSynchronize(stream));
    scan_impl(in, out, n, false, arena, stream);
    T last_out = T(0);
    TSR_CUDA_CHECK(cudaMemcpyAsync(&last_out, out + n - 1, sizeof(T), cudaMemcpyDeviceToHost, stream));
    TSR_CUDA_CHECK(cudaStreamSynchronize(stream));
    return last_out + last_in;
}

#define TSR_INSTANTIATE_SCAN(T)                                                                    \
    template void exclusive_scan<T>(const T *, T *, std::size_t, Arena &, cudaStream_t);           \
    template void inclusive_scan<T>(const T *, T *, std::size_t, Arena &, cudaStream_t);           \
    template T exclusive_scan_total<T>(const T *, T *, std::size_t, Arena &, cudaStream_t);

TSR_INSTANTIATE_SCAN(std::uint32_t)
TSR_INSTANTIATE_SCAN(std::int32_t)
TSR_INSTANTIATE_SCAN(std::uint64_t)
TSR_INSTANTIATE_SCAN(std::int64_t)
TSR_INSTANTIATE_SCAN(float)

#undef TSR_INSTANTIATE_SCAN
} // namespace tsr::prims
