// Tessera - stable LSD radix sort of key/value pairs.
//
// Each pass sorts on a 4-bit digit in three steps:
//   upsweep    per-tile digit counts, stored digit-major so that one exclusive
//              scan turns them into global scatter offsets
//   scan       prims::exclusive_scan over 16 * num_tiles counters
//   downsweep  ranks every key within its tile and scatters it
//
// Ranking is stable without block-wide scans: a tile is split into one
// contiguous 128-key chunk per warp, chunks are processed 32 keys at a time in
// order, and lanes holding equal digits are found with one ballot per digit
// bit. Per-warp running digit counts live in shared memory and are folded
// across warps once per tile.
//
// Real CUDA builds hand large sorts to CUB's onesweep radix sort instead (the
// path production renderers ship); the hand-written passes above remain the
// implementation on the emulated backend.
#include <algorithm>

#if defined(__CUDACC__)
#    include <cub/device/device_radix_sort.cuh>
#endif

#include "tsr/prims/prims.h"
#include "tsr/prims/warp.cuh"

namespace tsr::prims
{
namespace
{
constexpr int kRadixBits          = 4;
constexpr unsigned kRadixDigits   = 1u << kRadixBits;
constexpr unsigned kSortThreads   = 256;
constexpr unsigned kSortItems     = 4;
constexpr unsigned kSortWarps     = kSortThreads / kWarpSize;
constexpr unsigned kSortWarpChunk = kWarpSize * kSortItems;
constexpr unsigned kSortTile      = kSortThreads * kSortItems;

template<typename K>
TSR_DI unsigned digit_of(K key, int shift, unsigned digit_mask)
{
    return static_cast<unsigned>(key >> shift) & digit_mask;
}

template<typename K>
__global__ void __launch_bounds__(kSortThreads) radix_upsweep_kernel(
    const K *__restrict__ keys,
    std::uint64_t n,
    int shift,
    unsigned digit_mask,
    std::uint32_t *__restrict__ counts, // [kRadixDigits][num_tiles]
    unsigned num_tiles
)
{
    __shared__ std::uint32_t s_counts[kRadixDigits];
    if(threadIdx.x < kRadixDigits)
    {
        s_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    const std::uint64_t base = static_cast<std::uint64_t>(blockIdx.x) * kSortTile;
    TSR_UNROLL
    for(unsigned k = 0; k < kSortItems; ++k)
    {
        const std::uint64_t idx = base + k * kSortThreads + threadIdx.x;
        if(idx < n)
        {
            atomicAdd(&s_counts[digit_of(keys[idx], shift, digit_mask)], 1u);
        }
    }
    __syncthreads();

    if(threadIdx.x < kRadixDigits)
    {
        counts[threadIdx.x * num_tiles + blockIdx.x] = s_counts[threadIdx.x];
    }
}

template<typename K, typename V>
__global__ void __launch_bounds__(kSortThreads) radix_downsweep_kernel(
    const K *__restrict__ keys_in,
    K *__restrict__ keys_out,
    const V *__restrict__ values_in,
    V *__restrict__ values_out,
    std::uint64_t n,
    int shift,
    unsigned digit_mask,
    const std::uint32_t *__restrict__ offsets, // exclusive scan of the upsweep counts
    unsigned num_tiles
)
{
    __shared__ std::uint32_t s_warp_counts[kSortWarps][kRadixDigits];
    const unsigned lane = lane_id();
    const unsigned warp = warp_id();
    if(lane < kRadixDigits)
    {
        s_warp_counts[warp][lane] = 0;
    }
    __syncwarp();

    const std::uint64_t chunk = static_cast<std::uint64_t>(blockIdx.x) * kSortTile + warp * kSortWarpChunk;
    K keys[kSortItems];
    V values[kSortItems];
    unsigned digits[kSortItems];
    std::uint32_t ranks[kSortItems];
    bool valid[kSortItems];

    TSR_UNROLL
    for(unsigned k = 0; k < kSortItems; ++k)
    {
        const std::uint64_t idx = chunk + k * kWarpSize + lane;
        valid[k]                = idx < n;
        digits[k]               = 0;
        ranks[k]                = 0;
        if(valid[k])
        {
            keys[k]   = keys_in[idx];
            values[k] = values_in[idx];
            digits[k] = digit_of(keys[k], shift, digit_mask);
        }

        const unsigned peers     = warp_match_label<kRadixBits>(digits[k], valid[k]);
        std::uint32_t warp_count = 0;
        if(valid[k])
        {
            warp_count = s_warp_counts[warp][digits[k]];
            ranks[k]   = warp_count + static_cast<std::uint32_t>(__popc(peers & lanemask_lt()));
        }
        __syncwarp();
        // The lowest lane of each peer group advances the group's counter.
        if(valid[k] && lane == static_cast<unsigned>(__ffs(static_cast<int>(peers)) - 1))
        {
            s_warp_counts[warp][digits[k]] = warp_count + static_cast<std::uint32_t>(__popc(peers));
        }
        __syncwarp();
    }
    __syncthreads();

    // Exclusive scan across warps for each digit: warp w's keys of digit d go
    // after every earlier warp's keys of digit d.
    if(threadIdx.x < kRadixDigits)
    {
        std::uint32_t running = 0;
        for(unsigned w = 0; w < kSortWarps; ++w)
        {
            const std::uint32_t c          = s_warp_counts[w][threadIdx.x];
            s_warp_counts[w][threadIdx.x]  = running;
            running                       += c;
        }
    }
    __syncthreads();

    TSR_UNROLL
    for(unsigned k = 0; k < kSortItems; ++k)
    {
        if(valid[k])
        {
            const std::uint32_t pos = offsets[digits[k] * num_tiles + blockIdx.x] + s_warp_counts[warp][digits[k]] + ranks[k];
            keys_out[pos]           = keys[k];
            values_out[pos]         = values[k];
        }
    }
}
} // namespace

template<typename K, typename V>
void radix_sort_pairs(
    const K *keys_in,
    K *keys_out,
    const V *values_in,
    V *values_out,
    std::size_t n,
    int begin_bit,
    int end_bit,
    Arena &arena,
    cudaStream_t stream
)
{
    TSR_CHECK(begin_bit >= 0 && end_bit <= static_cast<int>(8 * sizeof(K)) && begin_bit <= end_bit,
              "invalid bit range [" << begin_bit << ", " << end_bit << ")");
    TSR_CHECK(keys_in != keys_out && values_in != values_out, "radix_sort_pairs does not sort in place");
    if(n == 0)
    {
        return;
    }
    TSR_CHECK(n <= 0xffffffffu, "radix_sort_pairs supports at most 2^32-1 elements");

    const int passes = div_round_up(end_bit - begin_bit, kRadixBits);
    if(passes == 0)
    {
        TSR_CUDA_CHECK(cudaMemcpyAsync(keys_out, keys_in, n * sizeof(K), cudaMemcpyDeviceToDevice, stream));
        TSR_CUDA_CHECK(cudaMemcpyAsync(values_out, values_in, n * sizeof(V), cudaMemcpyDeviceToDevice, stream));
        return;
    }

#if defined(TSR_BACKEND_CUDA)
    std::size_t temp_bytes = 0;
    TSR_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys_in, keys_out, values_in, values_out,
                                                   static_cast<std::uint32_t>(n), begin_bit, end_bit, stream));
    void *temp = arena.allocate_bytes(temp_bytes);
    TSR_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp, temp_bytes, keys_in, keys_out, values_in, values_out,
                                                   static_cast<std::uint32_t>(n), begin_bit, end_bit, stream));
    return;
#endif

    const unsigned num_tiles = static_cast<unsigned>(div_round_up<std::size_t>(n, kSortTile));
    std::uint32_t *counts    = arena.allocate<std::uint32_t>(static_cast<std::size_t>(kRadixDigits) * num_tiles);
    K *tmp_keys              = passes > 1 ? arena.allocate<K>(n) : nullptr;
    V *tmp_values            = passes > 1 ? arena.allocate<V>(n) : nullptr;

    const K *src_keys   = keys_in;
    const V *src_values = values_in;
    for(int p = 0; p < passes; ++p)
    {
        // Alternate destinations so that the final pass lands in the outputs.
        const bool to_out    = ((passes - 1 - p) % 2) == 0;
        K *dst_keys          = to_out ? keys_out : tmp_keys;
        V *dst_values        = to_out ? values_out : tmp_values;
        const int shift      = begin_bit + p * kRadixBits;
        const int bits       = std::min(kRadixBits, end_bit - shift);
        const unsigned dmask = (1u << bits) - 1u;

        launch(
            radix_upsweep_kernel<K>,
            LaunchConfig{dim3(num_tiles), dim3(kSortThreads), 0, stream},
            src_keys,
            static_cast<std::uint64_t>(n),
            shift,
            dmask,
            counts,
            num_tiles
        );
        exclusive_scan(counts, counts, static_cast<std::size_t>(kRadixDigits) * num_tiles, arena, stream);
        launch(
            radix_downsweep_kernel<K, V>,
            LaunchConfig{dim3(num_tiles), dim3(kSortThreads), 0, stream},
            src_keys,
            dst_keys,
            src_values,
            dst_values,
            static_cast<std::uint64_t>(n),
            shift,
            dmask,
            counts,
            num_tiles
        );
        src_keys   = dst_keys;
        src_values = dst_values;
    }
}

template<typename K, typename V>
void sort_pairs(
    const K *keys_in,
    K *keys_out,
    const V *values_in,
    V *values_out,
    std::size_t n,
    int begin_bit,
    int end_bit,
    Arena &arena,
    cudaStream_t stream
)
{
    if(n <= kBitonicMaxElements)
    {
        // A single shared-memory network beats several radix passes for
        // small inputs. It compares whole keys, which matches the radix
        // ordering because callers only set bits inside [begin_bit, end_bit).
        bitonic_sort_pairs(keys_in, keys_out, values_in, values_out, n, stream);
        return;
    }
    radix_sort_pairs(keys_in, keys_out, values_in, values_out, n, begin_bit, end_bit, arena, stream);
}

#define TSR_INSTANTIATE_SORT(K, V)                                                                                  \
    template void radix_sort_pairs<K, V>(const K *, K *, const V *, V *, std::size_t, int, int, Arena &, cudaStream_t); \
    template void sort_pairs<K, V>(const K *, K *, const V *, V *, std::size_t, int, int, Arena &, cudaStream_t);

TSR_INSTANTIATE_SORT(std::uint64_t, std::int32_t)
TSR_INSTANTIATE_SORT(std::uint32_t, std::uint32_t)
TSR_INSTANTIATE_SORT(std::uint32_t, std::int32_t)

#undef TSR_INSTANTIATE_SORT
} // namespace tsr::prims
