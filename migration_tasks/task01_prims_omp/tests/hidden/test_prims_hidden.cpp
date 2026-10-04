// Hidden acceptance tests for the OpenMP-offload port of the primitives.
// Written against the public API only: tsr/prims/prims.h,
// tsr/device/memory.h and tsr/device/error.h.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "doctest.h"
#include "tsr/device/error.h"
#include "tsr/device/memory.h"
#include "tsr/prims/prims.h"

using namespace tsr;

namespace
{
template<typename T>
std::vector<T> rand_vec(std::size_t n, std::uint64_t seed, long long lo, long long hi)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<long long> d(lo, hi);
    std::vector<T> v(n);
    for(auto &x : v)
    {
        x = static_cast<T>(d(rng));
    }
    return v;
}

template<typename T>
std::vector<T> host_exclusive(const std::vector<T> &v)
{
    std::vector<T> out(v.size());
    T acc = T(0);
    for(std::size_t i = 0; i < v.size(); ++i)
    {
        out[i] = acc;
        acc   += v[i];
    }
    return out;
}

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
} // namespace

// ---------------------------------------------------------------- scan ----
TEST_CASE("scan: exclusive and inclusive over chunk and tile boundaries")
{
    Arena arena(1 << 20);
    for(std::size_t n : {1, 2, 255, 256, 257, 1023, 1024, 1025, 4096, 65537, 300001})
    {
        CAPTURE(n);
        const auto host = rand_vec<std::uint32_t>(n, n, 0, 9);
        DeviceBuffer<std::uint32_t> in, out(n);
        in.upload(host);
        prims::exclusive_scan(in.data(), out.data(), n, arena, nullptr);
        CHECK(out.to_host() == host_exclusive(host));
        prims::inclusive_scan(in.data(), out.data(), n, arena, nullptr);
        auto inc = host_exclusive(host);
        for(std::size_t i = 0; i < n; ++i)
        {
            inc[i] += host[i];
        }
        CHECK(out.to_host() == inc);
        arena.reset();
    }
}

TEST_CASE("scan: empty input is a no-op")
{
    Arena arena;
    DeviceBuffer<std::uint32_t> buf(4);
    std::vector<std::uint32_t> sentinel = {7, 7, 7, 7};
    buf.upload(sentinel);
    prims::exclusive_scan(buf.data(), buf.data(), 0, arena, nullptr);
    prims::inclusive_scan(buf.data(), buf.data(), 0, arena, nullptr);
    CHECK(buf.to_host() == sentinel);
    CHECK(prims::exclusive_scan_total(buf.data(), buf.data(), 0, arena, nullptr) == 0u);
}

TEST_CASE("scan: in-place 64-bit and signed scans")
{
    Arena arena;
    const std::size_t n = 70000;
    std::vector<std::int64_t> big(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        big[i] = static_cast<std::int64_t>(i % 13) * (std::int64_t{1} << 34) - static_cast<std::int64_t>(i % 5);
    }
    DeviceBuffer<std::int64_t> b;
    b.upload(big);
    const std::int64_t total = prims::exclusive_scan_total(b.data(), b.data(), n, arena, nullptr);
    CHECK(total == std::accumulate(big.begin(), big.end(), std::int64_t{0}));
    CHECK(b.to_host() == host_exclusive(big));

    const auto neg = rand_vec<std::int32_t>(5000, 3, -50, 50);
    DeviceBuffer<std::int32_t> d;
    d.upload(neg);
    prims::inclusive_scan(d.data(), d.data(), neg.size(), arena, nullptr);
    const auto got = d.to_host();
    std::int32_t acc = 0;
    bool ok          = true;
    for(std::size_t i = 0; i < neg.size(); ++i)
    {
        acc += neg[i];
        ok   = ok && got[i] == acc;
    }
    CHECK(ok);

    const auto u64 = rand_vec<std::uint64_t>(3333, 9, 0, 1000000);
    DeviceBuffer<std::uint64_t> u;
    u.upload(u64);
    CHECK(prims::exclusive_scan_total(u.data(), u.data(), u64.size(), arena, nullptr) == std::accumulate(u64.begin(), u64.end(), std::uint64_t{0}));
}

TEST_CASE("scan: float scan")
{
    Arena arena;
    std::vector<float> v(5000);
    for(std::size_t i = 0; i < v.size(); ++i)
    {
        v[i] = 0.25f * static_cast<float>(i % 7);
    }
    DeviceBuffer<float> in, out(v.size());
    in.upload(v);
    prims::inclusive_scan(in.data(), out.data(), v.size(), arena, nullptr);
    const auto got = out.to_host();
    double acc     = 0.0;
    bool ok        = true;
    for(std::size_t i = 0; i < v.size(); ++i)
    {
        acc += v[i];
        ok   = ok && std::abs(got[i] - acc) <= 1e-3 * (1.0 + acc);
    }
    CHECK(ok);
}

// -------------------------------------------------------------- reduce ----
TEST_CASE("reduce: sum, max and min across types and sizes")
{
    Arena arena;
    for(std::size_t n : {1, 31, 256, 1000, 65536, 100003})
    {
        CAPTURE(n);
        const auto fi = rand_vec<std::int32_t>(n, n + 1, -1000000, 1000000);
        std::vector<float> ff(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            ff[i] = static_cast<float>(fi[i]) / 4096.f;
        }
        DeviceBuffer<float> f, fo(1);
        f.upload(ff);
        prims::reduce(f.data(), n, fo.data(), prims::ReduceOp::Max, arena, nullptr);
        CHECK(fo.to_host()[0] == *std::max_element(ff.begin(), ff.end()));
        prims::reduce(f.data(), n, fo.data(), prims::ReduceOp::Min, arena, nullptr);
        CHECK(fo.to_host()[0] == *std::min_element(ff.begin(), ff.end()));
        prims::reduce(f.data(), n, fo.data(), prims::ReduceOp::Sum, arena, nullptr);
        const double ref = std::accumulate(ff.begin(), ff.end(), 0.0);
        CHECK(fo.to_host()[0] == doctest::Approx(ref).epsilon(1e-3).scale(static_cast<double>(n)));

        DeviceBuffer<std::int32_t> i32, io(1);
        i32.upload(fi);
        prims::reduce(i32.data(), n, io.data(), prims::ReduceOp::Min, arena, nullptr);
        CHECK(io.to_host()[0] == *std::min_element(fi.begin(), fi.end()));
        prims::reduce(i32.data(), n, io.data(), prims::ReduceOp::Max, arena, nullptr);
        CHECK(io.to_host()[0] == *std::max_element(fi.begin(), fi.end()));

        const auto fu = rand_vec<std::uint32_t>(n, n + 2, 0, 5000);
        DeviceBuffer<std::uint32_t> u, uo(1);
        u.upload(fu);
        prims::reduce(u.data(), n, uo.data(), prims::ReduceOp::Sum, arena, nullptr);
        CHECK(uo.to_host()[0] == std::accumulate(fu.begin(), fu.end(), 0u));
        arena.reset();
    }
}

TEST_CASE("reduce: empty input yields the identity")
{
    Arena arena;
    DeviceBuffer<float> f(1), fo(1);
    prims::reduce(f.data(), 0, fo.data(), prims::ReduceOp::Max, arena, nullptr);
    CHECK(fo.to_host()[0] <= -3.4e38f);
    prims::reduce(f.data(), 0, fo.data(), prims::ReduceOp::Sum, arena, nullptr);
    CHECK(fo.to_host()[0] == 0.f);
    DeviceBuffer<std::uint32_t> u(1), uo(1);
    prims::reduce(u.data(), 0, uo.data(), prims::ReduceOp::Min, arena, nullptr);
    CHECK(uo.to_host()[0] == 0xffffffffu);
    DeviceBuffer<std::int32_t> i(1), io(1);
    prims::reduce(i.data(), 0, io.data(), prims::ReduceOp::Max, arena, nullptr);
    CHECK(io.to_host()[0] == std::numeric_limits<std::int32_t>::min());
}

// ----------------------------------------------------------- histogram ----
TEST_CASE("histogram: counts, ignored values and overwrite semantics")
{
    Arena arena;
    for(std::size_t n : {0, 1, 4000, 123457})
    {
        CAPTURE(n);
        const auto v = rand_vec<std::uint32_t>(n, n + 5, 0, 400);
        DeviceBuffer<std::uint32_t> in, hist(prims::kHistogramBins);
        in.upload(v);
        // pre-fill: the histogram must be overwritten, not accumulated into
        hist.upload(std::vector<std::uint32_t>(prims::kHistogramBins, 99u));
        prims::histogram256(in.data(), n, hist.data(), arena, nullptr);
        std::vector<std::uint32_t> ref(prims::kHistogramBins, 0);
        for(auto x : v)
        {
            if(x < prims::kHistogramBins)
            {
                ++ref[x];
            }
        }
        CHECK(hist.to_host() == ref);
        arena.reset();
    }
}

// ---------------------------------------------------------------- sort ----
TEST_CASE("sort: radix sort is stable and ignores bits outside the range")
{
    Arena arena;
    // no structured bindings: clang refuses to capture them in lambdas when
    // -fopenmp is on ("capturing a structured binding is not yet supported")
    for(const auto &cfg : {std::tuple<std::size_t, int, int>{1, 0, 64}, {777, 3, 17}, {5000, 8, 20}, {40000, 0, 44}, {150001, 13, 31}})
    {
        const std::size_t n = std::get<0>(cfg);
        const int begin     = std::get<1>(cfg);
        const int end       = std::get<2>(cfg);
        CAPTURE(n);
        CAPTURE(begin);
        CAPTURE(end);
        std::mt19937_64 rng(n * 31 + begin);
        std::vector<std::uint64_t> keys(n);
        std::vector<std::int32_t> vals(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            keys[i] = rng(); // garbage everywhere, including outside [begin, end)
            if(i % 3 == 0 && i > 0)
            {
                keys[i] = keys[i - 1] ^ (std::uint64_t{1} << 63); // equal in-range bits
            }
            vals[i] = static_cast<std::int32_t>(i);
        }
        DeviceBuffer<std::uint64_t> kin, kout(n);
        DeviceBuffer<std::int32_t> vin, vout(n);
        kin.upload(keys);
        vin.upload(vals);
        prims::radix_sort_pairs(kin.data(), kout.data(), vin.data(), vout.data(), n, begin, end, arena, nullptr);
        const std::uint64_t mask = end - begin == 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << (end - begin)) - 1);
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ((keys[a] >> begin) & mask) < ((keys[b] >> begin) & mask); });
        const auto gv = vout.to_host();
        const auto gk = kout.to_host();
        bool ok       = true;
        for(std::size_t i = 0; i < n; ++i)
        {
            ok = ok && gv[i] == static_cast<std::int32_t>(order[i]) && gk[i] == keys[order[i]];
        }
        CHECK(ok);
        // inputs are untouched
        CHECK(kin.to_host() == keys);
        arena.reset();
    }
}

TEST_CASE("sort: 32-bit key instantiations and empty bit range")
{
    Arena arena;
    const std::size_t n = 20000;
    auto keys           = rand_vec<std::uint32_t>(n, 77, 0, 0xffffffffLL);
    std::vector<std::uint32_t> vals(n);
    std::iota(vals.begin(), vals.end(), 0u);
    DeviceBuffer<std::uint32_t> kin, kout(n), vin, vout(n);
    kin.upload(keys);
    vin.upload(vals);
    prims::radix_sort_pairs(kin.data(), kout.data(), vin.data(), vout.data(), n, 0, 32, arena, nullptr);
    auto sorted = keys;
    std::sort(sorted.begin(), sorted.end());
    CHECK(kout.to_host() == sorted);

    // begin == end: a stable copy
    prims::radix_sort_pairs(kin.data(), kout.data(), vin.data(), vout.data(), n, 5, 5, arena, nullptr);
    CHECK(kout.to_host() == keys);
    CHECK(vout.to_host() == vals);

    std::vector<std::int32_t> ivals(n);
    std::iota(ivals.begin(), ivals.end(), 0);
    DeviceBuffer<std::int32_t> iv, ivo(n);
    iv.upload(ivals);
    prims::radix_sort_pairs(kin.data(), kout.data(), iv.data(), ivo.data(), n, 16, 32, arena, nullptr);
    const auto order = ivo.to_host();
    bool ok          = true;
    for(std::size_t i = 1; i < n; ++i)
    {
        const auto a = keys[order[i - 1]] >> 16, b = keys[order[i]] >> 16;
        ok           = ok && (a < b || (a == b && order[i - 1] < order[i]));
    }
    CHECK(ok);
    CHECK_THROWS_AS(prims::radix_sort_pairs(kin.data(), kin.data(), vin.data(), vout.data(), n, 0, 32, arena, nullptr), tsr::Error);
    CHECK_THROWS_AS(prims::radix_sort_pairs(kin.data(), kout.data(), vin.data(), vout.data(), n, 0, 33, arena, nullptr), tsr::Error);
}

TEST_CASE("sort: bitonic network orders by key then value")
{
    for(std::size_t n : {1, 2, 3, 5, 64, 100, 513, 1000, 1024})
    {
        CAPTURE(n);
        auto keys = rand_vec<std::uint32_t>(n, n * 7, 0, 15);
        auto vals = rand_vec<std::int32_t>(n, n * 11, -50, 50);
        DeviceBuffer<std::uint32_t> kin, kout(n);
        DeviceBuffer<std::int32_t> vin, vout(n);
        kin.upload(keys);
        vin.upload(vals);
        prims::bitonic_sort_pairs(kin.data(), kout.data(), vin.data(), vout.data(), n, nullptr);
        std::vector<std::pair<std::uint32_t, std::int32_t>> ref(n);
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
    DeviceBuffer<std::uint32_t> k(1025), ko(1025);
    DeviceBuffer<std::int32_t> v(1025), vo(1025);
    CHECK_THROWS_AS(prims::bitonic_sort_pairs(k.data(), ko.data(), v.data(), vo.data(), 1025, nullptr), tsr::Error);
}

TEST_CASE("sort: sort_pairs gives the same order on both sides of the bitonic limit")
{
    Arena arena;
    for(std::size_t n : {std::size_t{1000}, std::size_t{1024}, std::size_t{1025}, std::size_t{9000}})
    {
        CAPTURE(n);
        std::mt19937_64 rng(n);
        std::vector<std::uint64_t> keys(n);
        std::vector<std::int32_t> vals(n);
        for(std::size_t i = 0; i < n; ++i)
        {
            keys[i] = ((rng() % 50) << 32) | (rng() % 4); // many ties
            vals[i] = static_cast<std::int32_t>(i);
        }
        DeviceBuffer<std::uint64_t> kin, kout(n);
        DeviceBuffer<std::int32_t> vin, vout(n);
        kin.upload(keys);
        vin.upload(vals);
        prims::sort_pairs(kin.data(), kout.data(), vin.data(), vout.data(), n, 0, 40, arena, nullptr);
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });
        const auto gv = vout.to_host();
        bool ok       = true;
        for(std::size_t i = 0; i < n; ++i)
        {
            ok = ok && gv[i] == static_cast<std::int32_t>(order[i]);
        }
        CHECK(ok);
        arena.reset();
    }
}

// ------------------------------------------------------------- compact ----
TEST_CASE("compact: order, empty and full selections")
{
    Arena arena;
    for(int pattern = 0; pattern < 3; ++pattern)
    {
        CAPTURE(pattern);
        const std::size_t n = 77777;
        std::vector<std::uint8_t> flags(n);
        std::vector<std::uint32_t> ref;
        for(std::size_t i = 0; i < n; ++i)
        {
            flags[i] = pattern == 0 ? 0 : (pattern == 1 ? 1 : static_cast<std::uint8_t>((i * 2654435761u) % 5 == 0 ? 3 : 0));
            if(flags[i])
            {
                ref.push_back(static_cast<std::uint32_t>(i));
            }
        }
        DeviceBuffer<std::uint8_t> f;
        DeviceBuffer<std::uint32_t> out(n);
        f.upload(flags);
        const std::uint32_t count = prims::compact_indices(f.data(), n, out.data(), arena, nullptr);
        REQUIRE(count == ref.size());
        auto got = out.to_host();
        got.resize(count);
        CHECK(got == ref);
        arena.reset();
    }
    DeviceBuffer<std::uint8_t> none;
    DeviceBuffer<std::uint32_t> out(1);
    CHECK(prims::compact_indices(none.data(), 0, out.data(), arena, nullptr) == 0u);
}

// -------------------------------------------------------------- memory ----
TEST_CASE("memory: device buffers")
{
    DeviceBuffer<int> a(10);
    CHECK(a.size() == 10);
    a.zero();
    CHECK(a.to_host() == std::vector<int>(10, 0));
    std::vector<int> v(10);
    std::iota(v.begin(), v.end(), 5);
    a.upload(v);
    int *p = a.data();
    a.resize(4); // shrinking keeps the allocation
    CHECK(a.data() == p);
    CHECK(a.size() == 4);
    CHECK(a.to_host() == std::vector<int>({5, 6, 7, 8}));
    DeviceBuffer<int> b(std::move(a));
    CHECK(a.data() == nullptr);
    CHECK(b.size() == 4);
    CHECK(b.to_host()[3] == 8);
    std::vector<int> two(2);
    b.download(two.data(), 2);
    CHECK(two[1] == 6);
    CHECK_THROWS_AS(b.download(two.data(), 5), tsr::Error);
    DeviceBuffer<int> c;
    c = std::move(b);
    CHECK(c.size() == 4);
    CHECK(c.bytes() == 16);
}

TEST_CASE("memory: arena alignment and overflow coalescing")
{
    Arena arena(1024);
    auto *x = static_cast<char *>(arena.allocate_bytes(1));
    auto *y = static_cast<char *>(arena.allocate_bytes(300));
    CHECK((y - x) == 256);
    arena.allocate_bytes(5000); // does not fit: overflow chunk
    CHECK(arena.capacity() >= 1024 + 5000);
    const std::size_t hw = arena.high_water();
    CHECK(hw >= 256 + 512 + 5000);
    arena.reset();
    CHECK(arena.used() == 0);
    CHECK(arena.capacity() >= hw);
    const std::size_t cap = arena.capacity();
    arena.allocate_bytes(256);
    arena.allocate_bytes(512);
    arena.allocate_bytes(5000);
    CHECK(arena.capacity() == cap); // the coalesced block now holds the whole working set
}

// --------------------------------------------------------- end to end ----
TEST_CASE("golden: checksums of the original CUDA implementation")
{
    // Inputs and checksums match apps/prims_bench of the CUDA version
    // (n = 1000000, seed = 42).
    const std::size_t n = 1000000;
    std::uint64_t s     = 42;
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
    Arena arena;
    DeviceBuffer<std::uint32_t> d_small, d_bins, d_out(n), d_hist(prims::kHistogramBins), d_idx(n);
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

    prims::exclusive_scan(d_small.data(), d_out.data(), n, arena, nullptr);
    CHECK(checksum(d_out.to_host()) == 12668895335224401473ULL);
    prims::reduce(d_floats.data(), n, d_red.data(), prims::ReduceOp::Max, arena, nullptr);
    CHECK(checksum(d_red.to_host()) == 12173694279284103456ULL);
    prims::histogram256(d_bins.data(), n, d_hist.data(), arena, nullptr);
    CHECK(checksum(d_hist.to_host()) == 9562380637836916104ULL);
    prims::radix_sort_pairs(d_keys.data(), d_keys_out.data(), d_vals.data(), d_vals_out.data(), n, 0, 44, arena, nullptr);
    CHECK(mix(checksum(d_keys_out.to_host()), checksum(d_vals_out.to_host())) == 10016139971090865801ULL);
    const std::uint32_t count = prims::compact_indices(d_flags.data(), n, d_idx.data(), arena, nullptr);
    auto idx                  = d_idx.to_host();
    idx.resize(count);
    CHECK(checksum(idx) == 9490549744518190041ULL);
}
