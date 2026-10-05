// Tessera - warp-level building blocks (device only).
#pragma once

#include <cstdint>

#include "tsr/device/backend.h"

namespace tsr::prims
{
constexpr unsigned kFullMask = 0xffffffffu;
constexpr unsigned kWarpSize = 32;

// Lane of the calling thread for 1D blocks.
TSR_DI unsigned lane_id() { return threadIdx.x & (kWarpSize - 1); }
TSR_DI unsigned warp_id() { return threadIdx.x / kWarpSize; }

// Bits of lanes strictly below the calling lane.
TSR_DI unsigned lanemask_lt() { return (1u << lane_id()) - 1u; }

template<typename T>
struct SumOp
{
    TSR_HDI T operator()(const T &a, const T &b) const { return a + b; }
    TSR_HDI static T identity() { return T(0); }
};

template<typename T>
struct MaxOp
{
    TSR_HDI T operator()(const T &a, const T &b) const { return a < b ? b : a; }
    TSR_HDI static T identity();
};

template<typename T>
struct MinOp
{
    TSR_HDI T operator()(const T &a, const T &b) const { return b < a ? b : a; }
    TSR_HDI static T identity();
};

template<>
TSR_HDI float MaxOp<float>::identity()
{
    return -3.402823466e+38f;
}
template<>
TSR_HDI float MinOp<float>::identity()
{
    return 3.402823466e+38f;
}
template<>
TSR_HDI std::uint32_t MaxOp<std::uint32_t>::identity()
{
    return 0u;
}
template<>
TSR_HDI std::uint32_t MinOp<std::uint32_t>::identity()
{
    return 0xffffffffu;
}
template<>
TSR_HDI std::int32_t MaxOp<std::int32_t>::identity()
{
    return -2147483647 - 1;
}
template<>
TSR_HDI std::int32_t MinOp<std::int32_t>::identity()
{
    return 2147483647;
}

// Tree reduction towards lane 0 over the lanes in `mask`. Lanes outside the
// mask must not call this; lanes whose partner falls outside the warp segment
// keep their own value.
template<typename T, typename Op>
TSR_DI T warp_reduce(T value, Op op, unsigned mask = kFullMask)
{
    TSR_UNROLL
    for(unsigned offset = kWarpSize / 2; offset > 0; offset /= 2)
    {
        value = op(value, __shfl_down_sync(mask, value, offset));
    }
    return value;
}

// Butterfly reduction: every lane receives the result.
template<typename T, typename Op>
TSR_DI T warp_allreduce(T value, Op op)
{
    TSR_UNROLL
    for(unsigned offset = kWarpSize / 2; offset > 0; offset /= 2)
    {
        value = op(value, __shfl_xor_sync(kFullMask, value, static_cast<int>(offset)));
    }
    return value;
}

// Kogge-Stone inclusive scan across the full warp.
template<typename T>
TSR_DI T warp_inclusive_scan(T value)
{
    const unsigned lane = lane_id();
    TSR_UNROLL
    for(unsigned offset = 1; offset < kWarpSize; offset <<= 1)
    {
        const T other = __shfl_up_sync(kFullMask, value, offset);
        if(lane >= offset)
        {
            value += other;
        }
    }
    return value;
}

// Lanes of the warp whose `bits`-wide label equals the caller's, computed
// with one ballot per bit (a portable __match_any_sync). Only lanes with
// `valid` set are considered.
template<int BITS>
TSR_DI unsigned warp_match_label(unsigned label, bool valid)
{
    unsigned peers = __ballot_sync(kFullMask, valid);
    TSR_UNROLL
    for(int b = 0; b < BITS; ++b)
    {
        const bool bit      = (label >> b) & 1u;
        const unsigned vote = __ballot_sync(kFullMask, bit);
        peers              &= bit ? vote : ~vote;
    }
    return peers;
}

} // namespace tsr::prims
