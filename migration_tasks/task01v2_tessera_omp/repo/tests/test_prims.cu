// Tests for device-wide primitives against straightforward host references.
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "doctest.h"
#include "tsr/prims/prims.h"

using namespace tsr;

namespace
{
template<typename T>
std::vector<T> random_values(std::size_t n, std::uint32_t seed, T lo, T hi)
{
    std::mt19937 rng(seed);
    std::vector<T> out(n);
    if constexpr(std::is_floating_point_v<T>)
    {
        std::uniform_real_distribution<T> dist(lo, hi);
        for(auto &v : out)
        {
            v = dist(rng);
        }
    }
    else
    {
        std::uniform_int_distribution<long long> dist(static_cast<long long>(lo), static_cast<long long>(hi));
        for(auto &v : out)
        {
            v = static_cast<T>(dist(rng));
        }
    }
    return out;
}
} // namespace

TEST_CASE("prims: exclusive and inclusive scan over awkward lengths")
{
    Arena arena(1 << 20);
    for(std::size_t n : {std::size_t{1}, std::size_t{5}, std::size_t{1023}, std::size_t{1024}, std::size_t{1025}, std::size_t{4097}, std::size_t{70001}})
    {
        CAPTURE(n);
        const auto host = random_values<std::uint32_t>(n, static_cast<std::uint32_t>(n), 0, 9);
        DeviceBuffer<std::uint32_t> in, out(n);
        in.upload(host);

        prims::exclusive_scan(in.data(), out.data(), n, arena, nullptr);
        auto got = out.to_host();
        std::uint32_t acc = 0;
        bool ok           = true;
        for(std::size_t i = 0; i < n; ++i)
        {
            ok  = ok && got[i] == acc;
            acc += host[i];
        }
        CHECK(ok);

        prims::inclusive_scan(in.data(), out.data(), n, arena, nullptr);
        got = out.to_host();
        CHECK(got.back() == acc);
        arena.reset();
    }
}

TEST_CASE("prims: in-place 64-bit scan with total")
{
    Arena arena;
    const std::size_t n = 3000;
    std::vector<std::int64_t> host(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        host[i] = static_cast<std::int64_t>(i % 7) * (std::int64_t{1} << 33);
    }
    DeviceBuffer<std::int64_t> buf;
    buf.upload(host);
    const std::int64_t total = prims::exclusive_scan_total(buf.data(), buf.data(), n, arena, nullptr);
    CHECK(total == std::accumulate(host.begin(), host.end(), std::int64_t{0}));
    const auto got = buf.to_host();
    CHECK(got[0] == 0);
    CHECK(got[n - 1] == total - host[n - 1]);
}

TEST_CASE("prims: reductions")
{
    Arena arena;
    for(std::size_t n : {std::size_t{1}, std::size_t{256}, std::size_t{1000}, std::size_t{65536}, std::size_t{100003}})
    {
        CAPTURE(n);
        const auto host = random_values<float>(n, 7, -4.0f, 4.0f);
        DeviceBuffer<float> in, out(1);
        in.upload(host);
        prims::reduce(in.data(), n, out.data(), prims::ReduceOp::Max, arena, nullptr);
        CHECK(out.to_host()[0] == *std::max_element(host.begin(), host.end()));
        prims::reduce(in.data(), n, out.data(), prims::ReduceOp::Min, arena, nullptr);
        CHECK(out.to_host()[0] == *std::min_element(host.begin(), host.end()));
        prims::reduce(in.data(), n, out.data(), prims::ReduceOp::Sum, arena, nullptr);
        const double ref = std::accumulate(host.begin(), host.end(), 0.0);
        CHECK(out.to_host()[0] == doctest::Approx(ref).epsilon(1e-3).scale(static_cast<double>(n)));
        arena.reset();
    }
}

TEST_CASE("prims: integer sum reduction is exact")
{
    Arena arena;
    const auto host = random_values<std::uint32_t>(12345, 3, 0, 1000);
    DeviceBuffer<std::uint32_t> in, out(1);
    in.upload(host);
    prims::reduce(in.data(), host.size(), out.data(), prims::ReduceOp::Sum, arena, nullptr);
    CHECK(out.to_host()[0] == std::accumulate(host.begin(), host.end(), 0u));
}

TEST_CASE("prims: histogram256")
{
    Arena arena;
    auto host = random_values<std::uint32_t>(50000, 11, 0, 300); // values >= 256 are ignored
    DeviceBuffer<std::uint32_t> in, hist(prims::kHistogramBins);
    in.upload(host);
    prims::histogram256(in.data(), host.size(), hist.data(), arena, nullptr);
    std::vector<std::uint32_t> ref(prims::kHistogramBins, 0);
    for(auto v : host)
    {
        if(v < prims::kHistogramBins)
        {
            ++ref[v];
        }
    }
    CHECK(hist.to_host() == ref);
}

TEST_CASE("prims: radix sort is stable and honours the bit range")
{
    Arena arena;
    for(std::size_t n : {std::size_t{1}, std::size_t{300}, std::size_t{1024}, std::size_t{5000}, std::size_t{40000}})
    {
        CAPTURE(n);
        std::mt19937_64 rng(n);
        std::vector<std::uint64_t> keys(n);
        std::vector<std::int32_t> vals(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            // Few distinct keys in the sorted bit range to exercise stability;
            // garbage below begin_bit must not influence the order.
            keys[i] = ((rng() % 37) << 8) | (rng() & 0xff);
            vals[i] = static_cast<std::int32_t>(i);
        }
        DeviceBuffer<std::uint64_t> kin, kout(n);
        DeviceBuffer<std::int32_t> vin, vout(n);
        kin.upload(keys);
        vin.upload(vals);
        prims::radix_sort_pairs(kin.data(), kout.data(), vin.data(), vout.data(), n, 8, 20, arena, nullptr);

        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return (keys[a] >> 8) < (keys[b] >> 8); });
        const auto got_v = vout.to_host();
        bool ok          = true;
        for(std::size_t i = 0; i < n; ++i)
        {
            ok = ok && got_v[i] == static_cast<std::int32_t>(order[i]);
        }
        CHECK(ok);
        arena.reset();
    }
}

TEST_CASE("prims: bitonic sort orders by key then value")
{
    for(std::size_t n : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{100}, std::size_t{1024}})
    {
        CAPTURE(n);
        auto keys = random_values<std::uint32_t>(n, 5, 0, 20);
        std::vector<std::uint32_t> vals(n);
        std::iota(vals.begin(), vals.end(), 0u);
        DeviceBuffer<std::uint32_t> kin, kout(n), vin, vout(n);
        kin.upload(keys);
        vin.upload(vals);
        prims::bitonic_sort_pairs(kin.data(), kout.data(), vin.data(), vout.data(), n, nullptr);
        std::vector<std::pair<std::uint32_t, std::uint32_t>> ref(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            ref[i] = {keys[i], vals[i]};
        }
        std::sort(ref.begin(), ref.end());
        const auto gk = kout.to_host();
        const auto gv = vout.to_host();
        bool ok       = true;
        for(std::size_t i = 0; i < n; ++i)
        {
            ok = ok && gk[i] == ref[i].first && gv[i] == ref[i].second;
        }
        CHECK(ok);
    }
}

TEST_CASE("prims: compact_indices")
{
    Arena arena;
    const std::size_t n = 9999;
    std::vector<std::uint8_t> flags(n);
    std::vector<std::uint32_t> ref;
    for(std::size_t i = 0; i < n; ++i)
    {
        flags[i] = (i % 3 == 0 || i % 7 == 0) ? 1 : 0;
        if(flags[i])
        {
            ref.push_back(static_cast<std::uint32_t>(i));
        }
    }
    DeviceBuffer<std::uint8_t> f;
    DeviceBuffer<std::uint32_t> out(n);
    f.upload(flags);
    const auto count = prims::compact_indices(f.data(), n, out.data(), arena, nullptr);
    REQUIRE(count == ref.size());
    auto got = out.to_host();
    got.resize(count);
    CHECK(got == ref);
}
