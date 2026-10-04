// Tessera - single-block bitonic sort for short key/value arrays.
//
// After bitonicSortShared from the cuda-samples "sortingNetworks" sample
// (BSD-3-Clause). Changes: templated key/value types, arbitrary lengths up to
// the shared-memory limit (padded with maximal sentinels to the next power of
// two), and a comparator that orders by (key, value) so equal keys come out in
// a deterministic order.
#include <limits>

#include "tsr/prims/prims.h"

namespace tsr::prims
{
namespace
{
constexpr unsigned kSharedSizeLimit = static_cast<unsigned>(kBitonicMaxElements);

template<typename K, typename V>
TSR_DI void comparator(K &key_a, V &val_a, K &key_b, V &val_b, unsigned dir)
{
    const bool a_greater = key_a > key_b || (key_a == key_b && val_a > val_b);
    if(a_greater == (dir != 0))
    {
        const K tk = key_a;
        key_a      = key_b;
        key_b      = tk;
        const V tv = val_a;
        val_a      = val_b;
        val_b      = tv;
    }
}

template<typename K, typename V>
__global__ void bitonic_sort_shared_kernel(
    const K *__restrict__ keys_in,
    K *__restrict__ keys_out,
    const V *__restrict__ values_in,
    V *__restrict__ values_out,
    unsigned n,
    unsigned array_length, // power of two >= n
    unsigned dir           // 1 = ascending
)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ K s_key[kSharedSizeLimit];
    __shared__ V s_val[kSharedSizeLimit];

    const unsigned half = array_length / 2;
    const unsigned t    = threadIdx.x;
    TSR_UNROLL
    for(unsigned j = 0; j < 2; ++j)
    {
        const unsigned i = t + j * half;
        if(i < n)
        {
            s_key[i] = keys_in[i];
            s_val[i] = values_in[i];
        }
        else
        {
            s_key[i] = std::numeric_limits<K>::max();
            s_val[i] = std::numeric_limits<V>::max();
        }
    }

    for(unsigned size = 2; size < array_length; size <<= 1)
    {
        // Bitonic merge
        const unsigned ddd = dir ^ ((t & (size / 2)) != 0);
        for(unsigned stride = size / 2; stride > 0; stride >>= 1)
        {
            cg::sync(cta);
            const unsigned pos = 2 * t - (t & (stride - 1));
            comparator(s_key[pos], s_val[pos], s_key[pos + stride], s_val[pos + stride], ddd);
        }
    }

    // ddd == dir for the last bitonic merge step
    for(unsigned stride = array_length / 2; stride > 0; stride >>= 1)
    {
        cg::sync(cta);
        const unsigned pos = 2 * t - (t & (stride - 1));
        comparator(s_key[pos], s_val[pos], s_key[pos + stride], s_val[pos + stride], dir);
    }

    cg::sync(cta);
    TSR_UNROLL
    for(unsigned j = 0; j < 2; ++j)
    {
        const unsigned i = t + j * half;
        if(i < n)
        {
            keys_out[i]   = s_key[i];
            values_out[i] = s_val[i];
        }
    }
}
} // namespace

template<typename K, typename V>
void bitonic_sort_pairs(const K *keys_in, K *keys_out, const V *values_in, V *values_out, std::size_t n, cudaStream_t stream)
{
    TSR_CHECK(n <= kBitonicMaxElements, "bitonic_sort_pairs handles at most " << kBitonicMaxElements << " elements, got " << n);
    if(n == 0)
    {
        return;
    }
    unsigned length = 2;
    while(length < n)
    {
        length <<= 1;
    }
    launch(
        bitonic_sort_shared_kernel<K, V>,
        LaunchConfig{dim3(1), dim3(length / 2), 0, stream},
        keys_in,
        keys_out,
        values_in,
        values_out,
        static_cast<unsigned>(n),
        length,
        1u
    );
}

template void bitonic_sort_pairs<std::uint64_t, std::int32_t>(const std::uint64_t *, std::uint64_t *, const std::int32_t *, std::int32_t *, std::size_t, cudaStream_t);
template void bitonic_sort_pairs<std::uint32_t, std::uint32_t>(const std::uint32_t *, std::uint32_t *, const std::uint32_t *, std::uint32_t *, std::size_t, cudaStream_t);
template void bitonic_sort_pairs<std::uint32_t, std::int32_t>(const std::uint32_t *, std::uint32_t *, const std::int32_t *, std::int32_t *, std::size_t, cudaStream_t);
} // namespace tsr::prims
