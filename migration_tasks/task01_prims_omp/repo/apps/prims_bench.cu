// prims_bench - run every primitive on deterministic input and report a
// checksum and the wall time per call.
//
//   prims_bench [--n N] [--reps R] [--seed S]
//
// Checksums depend only on the input and the semantics of each primitive,
// so two correct implementations print identical checksum columns.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tsr/device/memory.h"
#include "tsr/prims/prims.h"

namespace
{
std::uint64_t mix(std::uint64_t h, std::uint64_t v)
{
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

template<typename T>
std::uint64_t checksum(const std::vector<T> &v)
{
    std::uint64_t h = 1469598103934665603ULL;
    for(const T &x : v)
    {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &x, sizeof(T) < 8 ? sizeof(T) : 8);
        h = mix(h, bits);
    }
    return h;
}

std::uint64_t splitmix(std::uint64_t &s)
{
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z               = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z               = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Kernel launches are asynchronous: synchronize so the time covers the work.
template<typename F>
double time_ms(int reps, F &&f)
{
    f(); // warm up
    TSR_CUDA_CHECK(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    for(int r = 0; r < reps; ++r)
    {
        f();
    }
    TSR_CUDA_CHECK(cudaDeviceSynchronize());
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}
} // namespace

int main(int argc, char **argv)
{
    std::size_t n      = 1 << 20;
    int reps           = 5;
    std::uint64_t seed = 42;
    for(int i = 1; i < argc; ++i)
    {
        if(!std::strcmp(argv[i], "--n") && i + 1 < argc)
        {
            n = std::strtoull(argv[++i], nullptr, 10);
        }
        else if(!std::strcmp(argv[i], "--reps") && i + 1 < argc)
        {
            reps = std::atoi(argv[++i]);
        }
        else if(!std::strcmp(argv[i], "--seed") && i + 1 < argc)
        {
            seed = std::strtoull(argv[++i], nullptr, 10);
        }
        else
        {
            std::fprintf(stderr, "usage: prims_bench [--n N] [--reps R] [--seed S]\n");
            return 2;
        }
    }

    try
    {
        std::uint64_t s = seed;
        std::vector<std::uint32_t> small(n), bins(n);
        std::vector<std::uint64_t> keys(n);
        std::vector<std::int32_t> vals(n);
        std::vector<float> floats(n);
        std::vector<std::uint8_t> flags(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            const std::uint64_t r = splitmix(s);
            small[i]              = static_cast<std::uint32_t>(r % 10);
            bins[i]               = static_cast<std::uint32_t>((r >> 8) % 300);
            keys[i]               = (r >> 12) & ((std::uint64_t{1} << 44) - 1);
            vals[i]               = static_cast<std::int32_t>(i);
            floats[i]             = static_cast<float>((r >> 40) % 1000) / 250.f - 2.f;
            flags[i]              = (r >> 20) % 3 == 0 ? 1 : 0;
        }

        tsr::Arena arena(std::size_t{64} << 20);
        tsr::DeviceBuffer<std::uint32_t> d_small, d_bins, d_out_u32(n), d_hist(tsr::prims::kHistogramBins), d_idx(n);
        tsr::DeviceBuffer<std::uint64_t> d_keys, d_keys_out(n);
        tsr::DeviceBuffer<std::int32_t> d_vals, d_vals_out(n);
        tsr::DeviceBuffer<float> d_floats, d_red(1);
        tsr::DeviceBuffer<std::uint8_t> d_flags;
        d_small.upload(small);
        d_bins.upload(bins);
        d_keys.upload(keys);
        d_vals.upload(vals);
        d_floats.upload(floats);
        d_flags.upload(flags);

        std::printf("%-18s %12s %20s\n", "primitive", "ms/call", "checksum");
        auto report = [](const char *name, double ms, std::uint64_t sum) { std::printf("%-18s %12.3f %20llu\n", name, ms, static_cast<unsigned long long>(sum)); };

        double ms = time_ms(reps, [&] { arena.reset(); tsr::prims::exclusive_scan(d_small.data(), d_out_u32.data(), n, arena, nullptr); });
        report("exclusive_scan", ms, checksum(d_out_u32.to_host()));

        ms = time_ms(reps, [&] { arena.reset(); tsr::prims::reduce(d_floats.data(), n, d_red.data(), tsr::prims::ReduceOp::Max, arena, nullptr); });
        report("reduce_max", ms, checksum(d_red.to_host()));

        ms = time_ms(reps, [&] { arena.reset(); tsr::prims::histogram256(d_bins.data(), n, d_hist.data(), arena, nullptr); });
        report("histogram256", ms, checksum(d_hist.to_host()));

        ms = time_ms(reps, [&] { arena.reset(); tsr::prims::radix_sort_pairs(d_keys.data(), d_keys_out.data(), d_vals.data(), d_vals_out.data(), n, 0, 44, arena, nullptr); });
        report("radix_sort_pairs", ms, mix(checksum(d_keys_out.to_host()), checksum(d_vals_out.to_host())));

        std::uint32_t count = 0;
        ms = time_ms(reps, [&] { arena.reset(); count = tsr::prims::compact_indices(d_flags.data(), n, d_idx.data(), arena, nullptr); });
        std::vector<std::uint32_t> idx = d_idx.to_host();
        idx.resize(count);
        report("compact_indices", ms, checksum(idx));
    }
    catch(const std::exception &e)
    {
        std::fprintf(stderr, "prims_bench: %s\n", e.what());
        return 1;
    }
    return 0;
}
