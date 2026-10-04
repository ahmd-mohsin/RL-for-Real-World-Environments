// Tessera - block-level scan and reduction codelets (device only).
//
// The shared-memory scan codelets are derived from the scan sample in NVIDIA
// cuda-samples (BSD-3-Clause, see third_party/LICENSES/cuda-samples.txt),
// generalized from `uint` to any arithmetic type and from the 4-wide uint4
// variant to a 4-items-per-thread form.
#pragma once

#include <cstdint>

#include "tsr/device/backend.h"
#include "tsr/prims/warp.cuh"

namespace tsr::prims
{
// Naive inclusive scan over `size` threads (power of two, == participating
// threads): O(N log N) work. Uses 2 * size elements of shared memory; the
// first half is zero padding that removes the `pos >= offset` test.
template<typename T>
TSR_DI T scan1_inclusive(T idata, volatile T *s_data, unsigned size, cg::thread_block cta)
{
    unsigned pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_data[pos]  = T(0);
    pos         += size;
    s_data[pos]  = idata;

    for(unsigned offset = 1; offset < size; offset <<= 1)
    {
        cg::sync(cta);
        const T t = s_data[pos] + s_data[pos - offset];
        cg::sync(cta);
        s_data[pos] = t;
    }
    return s_data[pos];
}

template<typename T>
TSR_DI T scan1_exclusive(T idata, volatile T *s_data, unsigned size, cg::thread_block cta)
{
    return scan1_inclusive(idata, s_data, size, cta) - idata;
}

// Four consecutive items per thread: a sequential scan of the thread's items
// followed by a block-wide exclusive scan of the per-thread totals.
template<typename T>
TSR_DI void scan4_exclusive(T (&items)[4], volatile T *s_data, unsigned threads, cg::thread_block cta, T &block_total)
{
    T inclusive[4];
    inclusive[0] = items[0];
    inclusive[1] = inclusive[0] + items[1];
    inclusive[2] = inclusive[1] + items[2];
    inclusive[3] = inclusive[2] + items[3];

    const T thread_offset = scan1_exclusive(inclusive[3], s_data, threads, cta);

    // The last thread's inclusive total is the block total; publish it
    // through shared memory once every thread has read its own prefix.
    cg::sync(cta);
    if(threadIdx.x == threads - 1)
    {
        s_data[0] = thread_offset + inclusive[3];
    }
    cg::sync(cta);
    block_total = s_data[0];

    items[3] = thread_offset + inclusive[2];
    items[2] = thread_offset + inclusive[1];
    items[1] = thread_offset + inclusive[0];
    items[0] = thread_offset;
}

// Block-wide reduction for 1D blocks of BLOCK threads (multiple of 32).
// The result is valid in thread 0 only.
template<unsigned BLOCK, typename T, typename Op>
TSR_DI T block_reduce(T value, Op op, T *s_warp /* [BLOCK / 32] */)
{
    static_assert(BLOCK % kWarpSize == 0 && BLOCK <= 1024, "block size must be a multiple of 32");
    constexpr unsigned kWarps = BLOCK / kWarpSize;
    value                     = warp_reduce(value, op);
    if(lane_id() == 0)
    {
        s_warp[warp_id()] = value;
    }
    __syncthreads();
    if(warp_id() == 0)
    {
        // The whole first warp participates; lanes past the number of warps
        // contribute the identity so no lane reads an undefined partner.
        value = lane_id() < kWarps ? s_warp[lane_id()] : Op::identity();
        value = warp_reduce(value, op);
    }
    return value;
}

} // namespace tsr::prims
