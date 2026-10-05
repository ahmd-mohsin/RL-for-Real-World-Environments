// Tessera - emulated device backend: cooperative groups subset.
//
// Covers thread_block, grid_group (rank queries only), statically sized
// tiled partitions (<= 32 threads) and cooperative_groups::reduce, which is
// the surface used by Tessera's kernels.
#pragma once

#include <type_traits>

#include "tsr/device/emu/intrinsics.h"

namespace cooperative_groups
{
class thread_block
{
public:
    void sync() const { __syncthreads(); }
    unsigned thread_rank() const { return ::tsr::emu::current_thread().linear_tid; }
    unsigned size() const
    {
        const dim3 &b = ::tsr::emu::current_thread().block_dim;
        return b.x * b.y * b.z;
    }
    unsigned num_threads() const { return size(); }
    dim3 group_index() const { return dim3(::tsr::emu::current_thread().block_idx); }
    dim3 thread_index() const { return dim3(::tsr::emu::current_thread().thread_idx); }
    dim3 group_dim() const { return ::tsr::emu::current_thread().block_dim; }
    dim3 dim_threads() const { return group_dim(); }
};

inline thread_block this_thread_block() { return thread_block{}; }

class grid_group
{
public:
    unsigned long long thread_rank() const
    {
        const auto &t                  = ::tsr::emu::current_thread();
        const unsigned long long block = t.block_idx.x + static_cast<unsigned long long>(t.block_idx.y) * t.grid_dim.x
                                       + static_cast<unsigned long long>(t.block_idx.z) * t.grid_dim.x * t.grid_dim.y;
        const unsigned long long per_block = static_cast<unsigned long long>(t.block_dim.x) * t.block_dim.y * t.block_dim.z;
        return block * per_block + t.linear_tid;
    }
    unsigned long long size() const
    {
        const auto &t = ::tsr::emu::current_thread();
        return static_cast<unsigned long long>(t.grid_dim.x) * t.grid_dim.y * t.grid_dim.z * t.block_dim.x * t.block_dim.y
             * t.block_dim.z;
    }
    unsigned long long num_threads() const { return size(); }
    void sync() const { ::std::abort(); } // grid-wide sync needs a cooperative launch
};

inline grid_group this_grid() { return grid_group{}; }

template<unsigned N>
class thread_block_tile
{
    static_assert(N > 0 && N <= 32 && (N & (N - 1)) == 0, "tile size must be a power of two <= 32");

public:
    static constexpr unsigned num_threads_v = N;

    unsigned thread_rank() const { return ::tsr::emu::current_thread().linear_tid & (N - 1); }
    unsigned meta_group_rank() const { return ::tsr::emu::current_thread().linear_tid / N; }
    unsigned meta_group_size() const { return (thread_block{}.size() + N - 1) / N; }
    static constexpr unsigned size() { return N; }
    static constexpr unsigned num_threads() { return N; }

    // Lanes of the calling thread's tile within its warp.
    unsigned mask() const
    {
        if constexpr(N == 32)
        {
            return 0xffffffffu;
        }
        else
        {
            const unsigned lane = ::tsr::emu::current_thread().lane;
            return ((1u << N) - 1u) << (lane & ~(N - 1));
        }
    }

    void sync() const { __syncwarp(mask()); }

    template<typename T>
    T shfl(T var, int src) const
    {
        return __shfl_sync(mask(), var, src, N);
    }
    template<typename T>
    T shfl_down(T var, unsigned delta) const
    {
        return __shfl_down_sync(mask(), var, delta, N);
    }
    template<typename T>
    T shfl_up(T var, unsigned delta) const
    {
        return __shfl_up_sync(mask(), var, delta, N);
    }
    template<typename T>
    T shfl_xor(T var, int lane_mask) const
    {
        return __shfl_xor_sync(mask(), var, lane_mask, N);
    }
    unsigned ballot(int pred) const
    {
        const unsigned lane = ::tsr::emu::current_thread().lane;
        return __ballot_sync(mask(), pred) >> (lane & ~(N - 1));
    }
    bool any(int pred) const { return ballot(pred) != 0; }
    bool all(int pred) const { return __all_sync(mask(), pred) != 0; }
};

template<unsigned N, typename Parent>
inline thread_block_tile<N> tiled_partition(const Parent &)
{
    return thread_block_tile<N>{};
}

inline void sync(const thread_block &g) { g.sync(); }
template<unsigned N>
inline void sync(const thread_block_tile<N> &g)
{
    g.sync();
}

template<typename T>
struct plus
{
    T operator()(const T &a, const T &b) const { return a + b; }
};
template<typename T>
struct less
{
    T operator()(const T &a, const T &b) const { return b < a ? b : a; }
};
template<typename T>
struct greater
{
    T operator()(const T &a, const T &b) const { return a < b ? b : a; }
};
template<typename T>
struct bit_or
{
    T operator()(const T &a, const T &b) const { return a | b; }
};
template<typename T>
struct bit_and
{
    T operator()(const T &a, const T &b) const { return a & b; }
};

// Butterfly all-reduce: every lane of the tile receives the result.
template<unsigned N, typename T, typename Op>
inline T reduce(const thread_block_tile<N> &tile, T value, Op op)
{
    for(unsigned offset = N / 2; offset > 0; offset /= 2)
    {
        value = op(value, tile.shfl_xor(value, static_cast<int>(offset)));
    }
    return value;
}

} // namespace cooperative_groups
