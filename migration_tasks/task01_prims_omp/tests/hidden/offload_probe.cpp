// Offload probe: proves that every primitive computes on the device.
//
// The probe interposes the libomptarget entry points that clang emits for
// target regions and data movement, plus the OpenMP device memory API, and
// counts, per primitive call:
//   kernels  number of target regions launched
//   d2h      bytes copied from the device to the host
// A port that downloads its input, computes on the host and uploads the
// result is caught by the d2h budget; one that never offloads is caught by
// the kernel count. Output: one JSON line per primitive on stdout.
#include <dlfcn.h>
#include <omp.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "tsr/device/memory.h"
#include "tsr/prims/prims.h"

namespace
{
std::atomic<long long> g_kernels{0};
std::atomic<long long> g_d2h{0};

constexpr std::int64_t kMapFrom = 0x002;

void count_from(std::int32_t n, const std::int64_t *sizes, const std::int64_t *types)
{
    if(sizes == nullptr || types == nullptr)
    {
        return;
    }
    for(std::int32_t i = 0; i < n; ++i)
    {
        if(types[i] & kMapFrom)
        {
            g_d2h += sizes[i];
        }
    }
}

template<typename F>
F real(const char *name)
{
    void *p = dlsym(RTLD_NEXT, name);
    if(p == nullptr)
    {
        std::fprintf(stderr, "offload_probe: cannot resolve %s\n", name);
        std::abort();
    }
    return reinterpret_cast<F>(p);
}

// Leading fields of libomptarget's KernelArgsTy (stable since LLVM 17).
struct KernelArgsPrefix
{
    std::uint32_t Version;
    std::uint32_t NumArgs;
    void **ArgBasePtrs;
    void **ArgPtrs;
    std::int64_t *ArgSizes;
    std::int64_t *ArgTypes;
};
} // namespace

extern "C" {
int __tgt_target_kernel(void *loc, std::int64_t dev, std::int32_t teams, std::int32_t threads, void *host_ptr, void *args)
{
    using Fn       = int (*)(void *, std::int64_t, std::int32_t, std::int32_t, void *, void *);
    static Fn next = real<Fn>("__tgt_target_kernel");
    ++g_kernels;
    const auto *a = static_cast<const KernelArgsPrefix *>(args);
    count_from(static_cast<std::int32_t>(a->NumArgs), a->ArgSizes, a->ArgTypes);
    return next(loc, dev, teams, threads, host_ptr, args);
}

#define TSR_PROBE_MAPPER(NAME)                                                                                       \
    void NAME(void *loc, std::int64_t dev, std::int32_t n, void **base, void **ptrs, std::int64_t *sizes,           \
              std::int64_t *types, void *names, void **mappers)                                                    \
    {                                                                                                                \
        using Fn       = void (*)(void *, std::int64_t, std::int32_t, void **, void **, std::int64_t *, std::int64_t *, void *, void **); \
        static Fn next = real<Fn>(#NAME);                                                                            \
        count_from(n, sizes, types);                                                                                 \
        next(loc, dev, n, base, ptrs, sizes, types, names, mappers);                                                 \
    }

TSR_PROBE_MAPPER(__tgt_target_data_end_mapper)
TSR_PROBE_MAPPER(__tgt_target_data_update_mapper)

#define TSR_PROBE_MAPPER_NOWAIT(NAME)                                                                                \
    void NAME(void *loc, std::int64_t dev, std::int32_t n, void **base, void **ptrs, std::int64_t *sizes,           \
              std::int64_t *types, void *names, void **mappers, std::int32_t dn, void *dl, std::int32_t ndn, void *ndl) \
    {                                                                                                                \
        using Fn       = void (*)(void *, std::int64_t, std::int32_t, void **, void **, std::int64_t *, std::int64_t *, void *, void **, std::int32_t, void *, std::int32_t, void *); \
        static Fn next = real<Fn>(#NAME);                                                                            \
        count_from(n, sizes, types);                                                                                 \
        next(loc, dev, n, base, ptrs, sizes, types, names, mappers, dn, dl, ndn, ndl);                               \
    }

TSR_PROBE_MAPPER_NOWAIT(__tgt_target_data_end_nowait_mapper)
TSR_PROBE_MAPPER_NOWAIT(__tgt_target_data_update_nowait_mapper)

int omp_target_memcpy(void *dst, const void *src, size_t length, size_t dst_off, size_t src_off, int dst_dev, int src_dev)
{
    using Fn       = int (*)(void *, const void *, size_t, size_t, size_t, int, int);
    static Fn next = real<Fn>("omp_target_memcpy");
    if(dst_dev == omp_get_initial_device() && src_dev != dst_dev)
    {
        g_d2h += static_cast<long long>(length);
    }
    return next(dst, src, length, dst_off, src_off, dst_dev, src_dev);
}

int omp_target_memcpy_rect(void *dst, const void *src, size_t elem, int ndims, const size_t *volume, const size_t *dst_off,
                           const size_t *src_off, const size_t *dst_dims, const size_t *src_dims, int dst_dev, int src_dev)
{
    using Fn       = int (*)(void *, const void *, size_t, int, const size_t *, const size_t *, const size_t *, const size_t *, const size_t *, int, int);
    static Fn next = real<Fn>("omp_target_memcpy_rect");
    if(dst != nullptr && volume != nullptr && dst_dev == omp_get_initial_device() && src_dev != dst_dev)
    {
        long long total = static_cast<long long>(elem);
        for(int d = 0; d < ndims; ++d)
        {
            total *= static_cast<long long>(volume[d]);
        }
        g_d2h += total;
    }
    return next(dst, src, elem, ndims, volume, dst_off, src_off, dst_dims, src_dims, dst_dev, src_dev);
}
} // extern "C"

namespace
{
template<typename F>
void section(const char *name, std::size_t n, F &&f)
{
    const long long k0 = g_kernels.load();
    const long long d0 = g_d2h.load();
    f();
    std::printf("{\"primitive\": \"%s\", \"n\": %zu, \"kernels\": %lld, \"d2h\": %lld}\n", name, n, g_kernels.load() - k0, g_d2h.load() - d0);
    std::fflush(stdout);
}
} // namespace

int main()
{
    using namespace tsr;
    const std::size_t n = 1 << 20;
    std::mt19937_64 rng(1234);
    std::vector<std::uint32_t> small(n), bins(n);
    std::vector<std::uint64_t> keys(n);
    std::vector<std::int32_t> vals(n);
    std::vector<float> floats(n);
    std::vector<std::uint8_t> flags(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        const std::uint64_t r = rng();
        small[i]              = static_cast<std::uint32_t>(r % 10);
        bins[i]               = static_cast<std::uint32_t>((r >> 8) % 300);
        keys[i]               = r >> 20;
        vals[i]               = static_cast<std::int32_t>(i);
        floats[i]             = static_cast<float>((r >> 40) % 1000) / 250.f - 2.f;
        flags[i]              = (r >> 30) % 3 == 0 ? 1 : 0;
    }
    Arena arena(std::size_t{256} << 20);
    DeviceBuffer<std::uint32_t> d_small, d_bins, d_u32(n), d_hist(prims::kHistogramBins), d_idx(n);
    DeviceBuffer<std::uint64_t> d_keys, d_keys_out(n);
    DeviceBuffer<std::int32_t> d_vals, d_vals_out(n);
    DeviceBuffer<float> d_floats, d_red(1);
    DeviceBuffer<std::uint8_t> d_flags;
    d_small.upload(small);
    d_bins.upload(bins);
    d_keys.upload(keys);
    d_vals.upload(vals);
    d_floats.upload(floats);
    d_flags.upload(flags);

    section("exclusive_scan", n, [&] { prims::exclusive_scan(d_small.data(), d_u32.data(), n, arena, nullptr); });
    section("inclusive_scan", n, [&] { prims::inclusive_scan(d_small.data(), d_u32.data(), n, arena, nullptr); });
    section("exclusive_scan_total", n, [&] { (void)prims::exclusive_scan_total(d_small.data(), d_u32.data(), n, arena, nullptr); });
    section("reduce", n, [&] { prims::reduce(d_floats.data(), n, d_red.data(), prims::ReduceOp::Sum, arena, nullptr); });
    section("histogram256", n, [&] { prims::histogram256(d_bins.data(), n, d_hist.data(), arena, nullptr); });
    section("radix_sort_pairs", n, [&] { prims::radix_sort_pairs(d_keys.data(), d_keys_out.data(), d_vals.data(), d_vals_out.data(), n, 0, 44, arena, nullptr); });
    section("bitonic_sort_pairs", 1000, [&] { prims::bitonic_sort_pairs(d_keys.data(), d_keys_out.data(), d_vals.data(), d_vals_out.data(), 1000, nullptr); });
    section("sort_pairs", n, [&] { prims::sort_pairs(d_keys.data(), d_keys_out.data(), d_vals.data(), d_vals_out.data(), n, 0, 44, arena, nullptr); });
    section("compact_indices", n, [&] { (void)prims::compact_indices(d_flags.data(), n, d_idx.data(), arena, nullptr); });
    return 0;
}
