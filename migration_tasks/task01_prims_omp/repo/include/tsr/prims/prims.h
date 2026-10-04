// Tessera - device-wide parallel primitives (host API).
//
// All functions enqueue work on `stream`; temporaries come from `arena` and
// stay valid until the arena is reset. Pointers are device pointers.
#pragma once

#include <cstddef>
#include <cstdint>

#include "tsr/device/backend.h"
#include "tsr/device/memory.h"

namespace tsr::prims
{
// ---- scan -----------------------------------------------------------------
// out[i] = sum(in[0..i))  (exclusive) or sum(in[0..i]) (inclusive). In-place
// operation (in == out) is supported. Instantiated for uint32_t, int32_t,
// uint64_t, int64_t and float.
template<typename T>
void exclusive_scan(const T *in, T *out, std::size_t n, Arena &arena, cudaStream_t stream);

template<typename T>
void inclusive_scan(const T *in, T *out, std::size_t n, Arena &arena, cudaStream_t stream);

// Exclusive scan that also returns the grand total to the host (synchronizes).
template<typename T>
T exclusive_scan_total(const T *in, T *out, std::size_t n, Arena &arena, cudaStream_t stream);

// ---- reduce ---------------------------------------------------------------
enum class ReduceOp : std::uint8_t
{
    Sum,
    Max,
    Min,
};

// Reduces n values into *out (device pointer). Instantiated for float,
// uint32_t and int32_t. An empty input produces the operation's identity.
template<typename T>
void reduce(const T *in, std::size_t n, T *out, ReduceOp op, Arena &arena, cudaStream_t stream);

// ---- histogram ------------------------------------------------------------
constexpr std::uint32_t kHistogramBins = 256;

// hist[b] = number of i with bins[i] == b, for b < 256. Values >= 256 are
// ignored. hist must hold kHistogramBins counters; it is overwritten.
void histogram256(const std::uint32_t *bins, std::size_t n, std::uint32_t *hist, Arena &arena, cudaStream_t stream);

// ---- sorting --------------------------------------------------------------
// Stable LSD radix sort of (key, value) pairs on key bits [begin_bit, end_bit).
// Bits outside that range are ignored by the radix passes. Inputs are left
// untouched; results are written to keys_out / values_out. Instantiated for
// (uint64_t, int32_t), (uint32_t, uint32_t) and (uint32_t, int32_t).
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
);

// Largest input handled by the single-block bitonic network.
constexpr std::size_t kBitonicMaxElements = 1024;

// In-block bitonic sort of up to kBitonicMaxElements pairs, ordered by
// (key, value). Same instantiations as radix_sort_pairs.
template<typename K, typename V>
void bitonic_sort_pairs(const K *keys_in, K *keys_out, const V *values_in, V *values_out, std::size_t n, cudaStream_t stream);

// Sorts pairs choosing the cheapest correct algorithm for n. Ordering is by
// key, then by original position (equivalently by value when values were
// emitted in increasing order).
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
);

// ---- compaction -----------------------------------------------------------
// Writes the indices i with flags[i] != 0 to out_indices in increasing order
// and returns how many there were (synchronizes).
std::uint32_t compact_indices(const std::uint8_t *flags, std::size_t n, std::uint32_t *out_indices, Arena &arena, cudaStream_t stream);

} // namespace tsr::prims
