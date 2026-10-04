// Tests for the execution model: barriers, shared memory, warp intrinsics,
// cooperative groups, atomics and launch validation. These run on whichever
// backend the library was built for.
#include <numeric>
#include <vector>

#include "doctest.h"
#include "tsr/device/backend.h"
#include "tsr/device/memory.h"

namespace
{
// Block-wide tree reduction through static shared memory.
__global__ void block_sum_kernel(const int *in, int *out, int n)
{
    __shared__ int s[256];
    const unsigned t = threadIdx.x;
    const unsigned i = blockIdx.x * blockDim.x + t;
    s[t]             = i < static_cast<unsigned>(n) ? in[i] : 0;
    __syncthreads();
    for(unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(t < stride)
        {
            s[t] += s[t + stride];
        }
        __syncthreads();
    }
    if(t == 0)
    {
        out[blockIdx.x] = s[0];
    }
}

__global__ void warp_shuffle_kernel(const float *in, float *sum_out, float *bcast_out, float *up_out)
{
    const unsigned t = threadIdx.x;
    float v          = in[t];
    float sum        = v;
    for(int offset = 16; offset > 0; offset /= 2)
    {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    const float lane0 = __shfl_sync(0xffffffffu, v, 0);
    const float up    = __shfl_up_sync(0xffffffffu, v, 1);
    if((t & 31) == 0)
    {
        sum_out[t / 32] = sum;
    }
    bcast_out[t] = lane0;
    up_out[t]    = up;
}

__global__ void vote_kernel(const int *pred, unsigned *ballots, int *counts, int *alls)
{
    const unsigned t = threadIdx.x;
    const unsigned b = __ballot_sync(0xffffffffu, pred[t]);
    const int c      = __syncthreads_count(pred[t]);
    const int a      = __all_sync(0xffffffffu, pred[t]);
    if((t & 31) == 0)
    {
        ballots[t / 32] = b;
        alls[t / 32]    = a;
    }
    if(t == 0)
    {
        counts[0] = c;
    }
}

// Two 16-lane tiles of the same warp reduce independently.
__global__ void tile16_kernel(const int *in, int *out)
{
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<16>(block);
    int v      = in[threadIdx.x];
    v          = cg::reduce(tile, v, cg::plus<int>());
    if(tile.thread_rank() == 0)
    {
        out[tile.meta_group_rank()] = v;
    }
}

__global__ void dynamic_smem_kernel(int *out)
{
    TSR_DYNAMIC_SMEM(int, s);
    s[threadIdx.x] = static_cast<int>(threadIdx.x) * 3;
    __syncthreads();
    out[blockIdx.x * blockDim.x + threadIdx.x] = s[blockDim.x - 1 - threadIdx.x];
}

__global__ void atomic_hist_kernel(const unsigned *in, unsigned n, unsigned *bins)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        atomicAdd(&bins[in[i] % 8], 1u);
    }
}

__global__ void early_exit_kernel(int *out, int n)
{
    // Threads beyond n exit before the barrier; the rest must still proceed.
    const int t = static_cast<int>(threadIdx.x);
    if(t >= n)
    {
        return;
    }
    __shared__ int s[64];
    s[t] = t;
    __syncthreads();
    out[t] = s[n - 1 - t];
}

__global__ void axpy_kernel(unsigned n, float a, const float *x, float *y)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void noop_kernel(int *) {}
} // namespace

TEST_CASE("emu: block reduction through shared memory and barriers")
{
    const int n = 1000;
    std::vector<int> host(n);
    std::iota(host.begin(), host.end(), 1);
    tsr::DeviceBuffer<int> in, out(4);
    in.upload(host);
    tsr::launch(block_sum_kernel, {dim3(4), dim3(256)}, in.data(), out.data(), n);
    const auto partial = out.to_host();
    CHECK(std::accumulate(partial.begin(), partial.end(), 0) == n * (n + 1) / 2);
    CHECK(partial[0] == 256 * 257 / 2);
}

TEST_CASE("emu: warp shuffles")
{
    std::vector<float> host(64);
    std::iota(host.begin(), host.end(), 0.0f);
    tsr::DeviceBuffer<float> in, sums(2), bcast(64), up(64);
    in.upload(host);
    tsr::launch(warp_shuffle_kernel, {dim3(1), dim3(64)}, in.data(), sums.data(), bcast.data(), up.data());
    const auto s = sums.to_host();
    CHECK(s[0] == doctest::Approx(31 * 32 / 2));
    CHECK(s[1] == doctest::Approx((32 + 63) * 32 / 2));
    const auto b = bcast.to_host();
    CHECK(b[5] == 0.0f);
    CHECK(b[40] == 32.0f);
    const auto u = up.to_host();
    CHECK(u[0] == 0.0f);  // lane 0 keeps its own value
    CHECK(u[33] == 32.0f);
    CHECK(u[32] == 32.0f);
}

TEST_CASE("emu: ballot, all and __syncthreads_count")
{
    std::vector<int> pred(64, 0);
    for(int i = 0; i < 64; i += 3)
    {
        pred[i] = 1;
    }
    for(int i = 32; i < 64; ++i)
    {
        pred[i] = 1;
    }
    tsr::DeviceBuffer<int> p, counts(1), alls(2);
    tsr::DeviceBuffer<unsigned> ballots(2);
    p.upload(pred);
    tsr::launch(vote_kernel, {dim3(1), dim3(64)}, p.data(), ballots.data(), counts.data(), alls.data());
    unsigned expected0 = 0;
    for(int i = 0; i < 32; i += 3)
    {
        expected0 |= 1u << i;
    }
    const auto b = ballots.to_host();
    CHECK(b[0] == expected0);
    CHECK(b[1] == 0xffffffffu);
    CHECK(counts.to_host()[0] == std::accumulate(pred.begin(), pred.end(), 0));
    const auto a = alls.to_host();
    CHECK(a[0] == 0);
    CHECK(a[1] == 1);
}

TEST_CASE("emu: sub-warp tiles reduce independently")
{
    std::vector<int> host(64);
    std::iota(host.begin(), host.end(), 0);
    tsr::DeviceBuffer<int> in, out(4);
    in.upload(host);
    tsr::launch(tile16_kernel, {dim3(1), dim3(64)}, in.data(), out.data());
    const auto r = out.to_host();
    for(int t = 0; t < 4; ++t)
    {
        int expect = 0;
        for(int i = t * 16; i < t * 16 + 16; ++i)
        {
            expect += i;
        }
        CHECK(r[t] == expect);
    }
}

TEST_CASE("emu: dynamic shared memory")
{
    tsr::DeviceBuffer<int> out(3 * 32);
    tsr::launch(dynamic_smem_kernel, {dim3(3), dim3(32), 32 * sizeof(int)}, out.data());
    const auto r = out.to_host();
    CHECK(r[0] == 31 * 3);
    CHECK(r[32 + 31] == 0);
}

TEST_CASE("emu: global atomics across concurrently running blocks")
{
    const unsigned n = 5000;
    std::vector<unsigned> host(n);
    std::iota(host.begin(), host.end(), 0u);
    tsr::DeviceBuffer<unsigned> in, bins(8);
    in.upload(host);
    bins.zero();
    tsr::launch(atomic_hist_kernel, {dim3(tsr::div_round_up(n, 128u)), dim3(128)}, in.data(), n, bins.data());
    const auto b = bins.to_host();
    CHECK(std::accumulate(b.begin(), b.end(), 0u) == n);
    CHECK(b[0] == 625u);
}

TEST_CASE("emu: threads that exit early do not block the barrier")
{
    tsr::DeviceBuffer<int> out(64);
    out.zero();
    tsr::launch(early_exit_kernel, {dim3(1), dim3(64)}, out.data(), 40);
    const auto r = out.to_host();
    CHECK(r[0] == 39);
    CHECK(r[39] == 0);
    CHECK(r[40] == 0);
}

TEST_CASE("emu: linear launches")
{
    const unsigned n = 1000;
    std::vector<float> x(n, 2.0f), y(n, 1.0f);
    tsr::DeviceBuffer<float> dx, dy;
    dx.upload(x);
    dy.upload(y);
    tsr::launch_linear(axpy_kernel, n, nullptr, n, 3.0f, dx.data(), dy.data());
    const auto r = dy.to_host();
    CHECK(r[0] == 7.0f);
    CHECK(r[n - 1] == 7.0f);
}

TEST_CASE("emu: invalid launch configurations are reported")
{
    CHECK_THROWS_AS(tsr::launch(noop_kernel, {dim3(1), dim3(2048)}, nullptr), tsr::DeviceError);
    CHECK_THROWS_AS(tsr::launch(noop_kernel, {dim3(1), dim3(32), 200 * 1024}, nullptr), tsr::DeviceError);
    CHECK_NOTHROW(tsr::launch(noop_kernel, {dim3(1), dim3(32)}, nullptr));
}
