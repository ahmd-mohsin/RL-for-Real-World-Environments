// Hidden performance benchmark. The same source builds against the original
// CUDA library and against an OpenMP port (public API only), so the grader
// can measure both on the same GPU.
//
// Every timed call is followed by a 1-element download of its output inside
// the timed region: a download waits for the data, so work that a primitive
// left running asynchronously is still counted.
//
//   tsr_perf_bench [--n N] [--reps R]   ->  one JSON line per case on stdout
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "tsr/device/memory.h"
#include "tsr/prims/prims.h"

namespace
{
template<typename F>
double median_ms(int reps, F &&f)
{
    f(); // warm up (first launch, allocations)
    f();
    std::vector<double> t;
    t.reserve(static_cast<std::size_t>(reps));
    for(int r = 0; r < reps; ++r)
    {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

void emit(const char *name, std::size_t n, double ms)
{
    std::printf("{\"case\": \"%s\", \"n\": %zu, \"median_ms\": %.4f}\n", name, n, ms);
    std::fflush(stdout);
}
} // namespace

int main(int argc, char **argv)
{
    using namespace tsr;
    std::size_t n = std::size_t{1} << 26;
    int reps      = 15;
    for(int i = 1; i + 1 < argc; ++i)
    {
        if(!std::strcmp(argv[i], "--n"))
        {
            n = std::strtoull(argv[++i], nullptr, 10);
        }
        else if(!std::strcmp(argv[i], "--reps"))
        {
            reps = std::atoi(argv[++i]);
        }
    }

    std::mt19937_64 rng(2026);
    std::vector<std::uint32_t> small(n), bins(n), keys32(n), idx32(n);
    std::vector<std::uint64_t> keys(n);
    std::vector<std::int32_t> vals(n);
    std::vector<float> floats(n);
    std::vector<std::uint8_t> flags(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        const std::uint64_t r = rng();
        small[i]              = static_cast<std::uint32_t>(r % 10);
        bins[i]               = static_cast<std::uint32_t>((r >> 8) % 256);
        keys32[i]             = static_cast<std::uint32_t>(r >> 32);
        idx32[i]              = static_cast<std::uint32_t>(i);
        keys[i]               = (r >> 12) & ((std::uint64_t{1} << 44) - 1);
        vals[i]               = static_cast<std::int32_t>(i);
        floats[i]             = static_cast<float>((r >> 40) % 1000) / 250.f - 2.f;
        flags[i]              = (r >> 20) % 3 == 0 ? 1 : 0;
    }

    Arena arena(std::size_t{1} << 30);
    DeviceBuffer<std::uint32_t> d_small, d_bins, d_keys32, d_idx32, d_u32(n), d_hist(prims::kHistogramBins), d_k32o(n), d_v32o(n);
    DeviceBuffer<std::uint64_t> d_keys, d_keys_out(n);
    DeviceBuffer<std::int32_t> d_vals, d_vals_out(n);
    DeviceBuffer<float> d_floats, d_red(1);
    DeviceBuffer<std::uint8_t> d_flags;
    d_small.upload(small);
    d_bins.upload(bins);
    d_keys32.upload(keys32);
    d_idx32.upload(idx32);
    d_keys.upload(keys);
    d_vals.upload(vals);
    d_floats.upload(floats);
    d_flags.upload(flags);

    std::uint32_t u32_sink = 0;
    std::uint64_t u64_sink = 0;
    float f_sink           = 0.f;

    emit("exclusive_scan_u32", n, median_ms(reps, [&] {
        arena.reset();
        prims::exclusive_scan(d_small.data(), d_u32.data(), n, arena, nullptr);
        d_u32.download(&u32_sink, 1);
    }));
    emit("inclusive_scan_u32", n, median_ms(reps, [&] {
        arena.reset();
        prims::inclusive_scan(d_small.data(), d_u32.data(), n, arena, nullptr);
        d_u32.download(&u32_sink, 1);
    }));
    emit("reduce_sum_f32", n, median_ms(reps, [&] {
        arena.reset();
        prims::reduce(d_floats.data(), n, d_red.data(), prims::ReduceOp::Sum, arena, nullptr);
        d_red.download(&f_sink, 1);
    }));
    emit("reduce_max_f32", n, median_ms(reps, [&] {
        arena.reset();
        prims::reduce(d_floats.data(), n, d_red.data(), prims::ReduceOp::Max, arena, nullptr);
        d_red.download(&f_sink, 1);
    }));
    emit("histogram256", n, median_ms(reps, [&] {
        arena.reset();
        prims::histogram256(d_bins.data(), n, d_hist.data(), arena, nullptr);
        d_hist.download(&u32_sink, 1);
    }));
    emit("radix_sort_u64_44bit", n, median_ms(reps, [&] {
        arena.reset();
        prims::radix_sort_pairs(d_keys.data(), d_keys_out.data(), d_vals.data(), d_vals_out.data(), n, 0, 44, arena, nullptr);
        d_keys_out.download(&u64_sink, 1);
    }));
    emit("radix_sort_u32_32bit", n, median_ms(reps, [&] {
        arena.reset();
        prims::radix_sort_pairs(d_keys32.data(), d_k32o.data(), d_idx32.data(), d_v32o.data(), n, 0, 32, arena, nullptr);
        d_k32o.download(&u32_sink, 1);
    }));
    emit("compact_indices", n, median_ms(reps, [&] {
        arena.reset();
        (void)prims::compact_indices(d_flags.data(), n, d_u32.data(), arena, nullptr);
        d_u32.download(&u32_sink, 1);
    }));
    return 0;
}
