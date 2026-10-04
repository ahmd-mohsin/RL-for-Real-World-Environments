// Tests for splat tile binning.
#include <algorithm>
#include <set>
#include <vector>

#include "doctest.h"
#include "tsr/bin/tiles.h"
#include "tsr/scene/scene.h"

using namespace tsr;

namespace
{
struct Binned
{
    std::vector<std::uint64_t> keys;
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> offsets;
    std::vector<int2> radii;
    std::vector<float2> means;
    std::vector<float> depths;
    TileBinning info;
};

Binned run(const Scene &scene, const Camera &cam, bool accurate, std::uint32_t tile)
{
    DeviceGaussians dg;
    dg.upload(scene.gaussians, nullptr);
    ProjectedSplats splats;
    project_gaussians(dg, cam, ProjectionSettings{}, splats, nullptr);
    Arena arena;
    TileBinning tb;
    BinningSettings bs;
    bs.accurate  = accurate;
    bs.tile_size = tile;
    bin_splats(splats, cam, bs, tb, arena, nullptr);
    Binned b;
    b.keys    = tb.sorted_keys.to_host();
    b.ids     = tb.sorted_splat_ids.to_host();
    b.offsets = tb.tile_offsets.to_host();
    b.radii   = splats.radii.to_host();
    b.means   = splats.means2d.to_host();
    b.depths  = splats.depths.to_host();
    b.info.tile_size = tb.tile_size;
    b.info.tiles_x   = tb.tiles_x;
    b.info.tiles_y   = tb.tiles_y;
    b.info.tile_bits = tb.tile_bits;
    b.info.n_isects  = tb.n_isects;
    return b;
}

// Checks structural invariants that hold for any correct binning.
void check_invariants(const Binned &b)
{
    const std::uint32_t n_tiles = b.info.tiles_x * b.info.tiles_y;
    REQUIRE(b.offsets.size() == n_tiles + 1);
    CHECK(b.offsets.front() == 0);
    CHECK(b.offsets.back() == static_cast<std::int32_t>(b.info.n_isects));
    for(std::uint32_t t = 0; t < n_tiles; ++t)
    {
        CHECK(b.offsets[t] <= b.offsets[t + 1]);
        for(std::int32_t i = b.offsets[t]; i < b.offsets[t + 1]; ++i)
        {
            REQUIRE((b.keys[i] >> 32) == t);
            if(i > b.offsets[t])
            {
                // front to back inside a tile, ties broken by splat index
                const float d0 = b.depths[b.ids[i - 1]];
                const float d1 = b.depths[b.ids[i]];
                CHECK((d0 < d1 || (d0 == d1 && b.ids[i - 1] < b.ids[i])));
            }
        }
    }
}
} // namespace

TEST_CASE("bin: tile id bit width")
{
    CHECK(tile_id_bits(1) == 0);
    CHECK(tile_id_bits(2) == 1);
    CHECK(tile_id_bits(12) == 4);
    CHECK(tile_id_bits(16) == 4);
    CHECK(tile_id_bits(17) == 5);
    CHECK(tile_id_bits(1u << 20) == 20);
}

TEST_CASE("bin: axis-aligned binning matches a brute-force tile enumeration")
{
    SyntheticSceneParams p;
    p.n_gaussians = 200;
    p.with_mesh   = false;
    const Scene scene = make_synthetic_scene(p);
    const Camera &cam = scene.cameras[0];
    const Binned b    = run(scene, cam, false, 16);
    check_invariants(b);

    // Recompute every (tile, splat) pair from the projected radii.
    std::multiset<std::pair<std::uint32_t, std::int32_t>> expect, got;
    for(std::size_t g = 0; g < b.radii.size(); ++g)
    {
        if(b.radii[g].x <= 0 || b.radii[g].y <= 0)
        {
            continue;
        }
        // same float expression order as the kernel, so boundary cases agree
        const float ts = 16.f;
        const float tx = b.means[g].x / ts, ty = b.means[g].y / ts;
        const float rx = b.radii[g].x / ts, ry = b.radii[g].y / ts;
        const int x0 = std::clamp(static_cast<int>(std::floor(tx - rx)), 0, static_cast<int>(b.info.tiles_x));
        const int x1 = std::clamp(static_cast<int>(std::ceil(tx + rx)), 0, static_cast<int>(b.info.tiles_x));
        const int y0 = std::clamp(static_cast<int>(std::floor(ty - ry)), 0, static_cast<int>(b.info.tiles_y));
        const int y1 = std::clamp(static_cast<int>(std::ceil(ty + ry)), 0, static_cast<int>(b.info.tiles_y));
        for(int y = y0; y < y1; ++y)
        {
            for(int x = x0; x < x1; ++x)
            {
                expect.insert({static_cast<std::uint32_t>(y * b.info.tiles_x + x), static_cast<std::int32_t>(g)});
            }
        }
    }
    for(std::size_t i = 0; i < b.keys.size(); ++i)
    {
        got.insert({static_cast<std::uint32_t>(b.keys[i] >> 32), b.ids[i]});
    }
    CHECK(got == expect);
}

TEST_CASE("bin: accurate binning is a subset of the axis-aligned binning")
{
    SyntheticSceneParams p;
    p.n_gaussians = 300;
    p.with_mesh   = false;
    const Scene scene = make_synthetic_scene(p);
    const Binned loose = run(scene, scene.cameras[0], false, 8);
    const Binned tight = run(scene, scene.cameras[0], true, 8);
    check_invariants(tight);
    CHECK(tight.info.n_isects <= loose.info.n_isects);
    std::set<std::pair<std::uint64_t, std::int32_t>> loose_pairs;
    for(std::size_t i = 0; i < loose.keys.size(); ++i)
    {
        loose_pairs.insert({loose.keys[i] >> 32, loose.ids[i]});
    }
    bool subset = true;
    for(std::size_t i = 0; i < tight.keys.size(); ++i)
    {
        subset = subset && loose_pairs.count({tight.keys[i] >> 32, tight.ids[i]}) == 1;
    }
    CHECK(subset);
}

TEST_CASE("bin: small and large intersection lists use different sorts with the same order")
{
    // A handful of splats stays on the bitonic path, a dense cloud on the
    // radix path; both must satisfy the same ordering invariants.
    for(std::uint32_t n : {20u, 2000u})
    {
        CAPTURE(n);
        SyntheticSceneParams p;
        p.n_gaussians = n;
        p.with_mesh   = false;
        p.width       = 96;
        p.height      = 64;
        const Scene scene = make_synthetic_scene(p);
        const Binned b    = run(scene, scene.cameras[0], true, 16);
        check_invariants(b);
        if(n == 2000u)
        {
            CHECK(b.info.n_isects > 1024u);
        }
    }
}

TEST_CASE("bin: empty input")
{
    Scene scene;
    scene.cameras.push_back(Camera::look_at(make_float3(0.f, 0.f, -5.f), make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), 60.f, 40, 30));
    const Binned b = run(scene, scene.cameras[0], true, 16);
    CHECK(b.info.n_isects == 0);
    CHECK(std::all_of(b.offsets.begin(), b.offsets.end(), [](std::int32_t o) { return o == 0; }));
}
