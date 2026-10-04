// Tests for tile-based splat compositing against a brute-force reference.
#include <cmath>
#include <vector>

#include "doctest.h"
#include "reference.h"
#include "tsr/bin/tiles.h"
#include "tsr/geom/projection.h"
#include "tsr/raster/splat_raster.h"
#include "tsr/render/surface.h"
#include "tsr/scene/scene.h"

using namespace tsr;

namespace
{
struct Composite
{
    std::vector<float3> color;
    std::vector<float> alpha;
    std::vector<float3> ref_color;
    std::vector<float> ref_alpha;
};

// Renders splats over a synthetic surface layer (a depth ramp with a colour
// gradient) and the same thing with the host reference.
Composite render(const Scene &scene, const Camera &cam, std::uint32_t tile, std::uint32_t ppt, bool with_surface)
{
    DeviceGaussians dg;
    dg.upload(scene.gaussians, nullptr);
    ProjectedSplats splats;
    project_gaussians(dg, cam, ProjectionSettings{}, splats, nullptr);
    compute_splat_colors(dg, cam, splats, nullptr);
    Arena arena;
    TileBinning tb;
    BinningSettings bs;
    bs.tile_size = tile;
    bin_splats(splats, cam, bs, tb, arena, nullptr);

    const int W = static_cast<int>(cam.width), H = static_cast<int>(cam.height);
    std::vector<float3> s_color(static_cast<std::size_t>(W) * H);
    std::vector<float> s_depth(static_cast<std::size_t>(W) * H);
    for(int y = 0; y < H; ++y)
    {
        for(int x = 0; x < W; ++x)
        {
            const std::size_t i = static_cast<std::size_t>(y) * W + x;
            s_color[i]          = make_float3(x / float(W), y / float(H), 0.25f);
            s_depth[i]          = with_surface && x > W / 3 ? 3.0f + 2.0f * y / H : INFINITY;
        }
    }
    SurfaceLayer surface;
    surface.width  = W;
    surface.height = H;
    surface.color.upload(s_color);
    surface.depth.upload(s_depth);

    SplatRasterOutput out;
    SplatRasterSettings rs;
    rs.pixels_per_thread = ppt;
    rasterize_splats(splats, tb, surface, cam, rs, out, nullptr);

    Composite c;
    c.color = out.color.to_host();
    c.alpha = out.alpha.to_host();

    const auto radii = splats.radii.to_host();
    const auto means = splats.means2d.to_host();
    const auto depth = splats.depths.to_host();
    const auto conic = splats.conics.to_host();
    const auto opac  = splats.opacities.to_host();
    const auto color = splats.colors.to_host();
    std::vector<reference::HostSplat> host(radii.size());
    for(std::size_t i = 0; i < host.size(); ++i)
    {
        host[i] = {radii[i], means[i], depth[i], conic[i], opac[i], color[i]};
    }
    reference::composite(host, W, H, s_color, s_depth, c.ref_color, c.ref_alpha);
    return c;
}

std::size_t count_mismatches(const Composite &c, float tol)
{
    std::size_t bad = 0;
    for(std::size_t i = 0; i < c.color.size(); ++i)
    {
        const float d = std::fabs(c.color[i].x - c.ref_color[i].x) + std::fabs(c.color[i].y - c.ref_color[i].y)
                      + std::fabs(c.color[i].z - c.ref_color[i].z) + std::fabs(c.alpha[i] - c.ref_alpha[i]);
        bad += d > tol ? 1 : 0;
    }
    return bad;
}
} // namespace

TEST_CASE("raster: tile compositing equals per-pixel brute force")
{
    SyntheticSceneParams p;
    p.n_gaussians = 400;
    p.with_mesh   = false;
    p.width       = 70; // partial tiles on both axes
    p.height      = 45;
    const Scene scene = make_synthetic_scene(p);
    for(std::uint32_t tile : {16u, 8u, 4u})
    {
        CAPTURE(tile);
        const Composite c = render(scene, scene.cameras[0], tile, 1, false);
        CHECK(count_mismatches(c, 1e-4f) == 0);
    }
}

TEST_CASE("raster: four pixels per thread variant agrees with one pixel per thread")
{
    SyntheticSceneParams p;
    p.n_gaussians = 400;
    p.with_mesh   = false;
    p.width       = 64;
    p.height      = 50;
    const Scene scene = make_synthetic_scene(p);
    const Composite a = render(scene, scene.cameras[1], 16, 1, true);
    const Composite b = render(scene, scene.cameras[1], 16, 4, true);
    std::size_t diff = 0;
    for(std::size_t i = 0; i < a.color.size(); ++i)
    {
        diff += (a.color[i].x != b.color[i].x || a.color[i].y != b.color[i].y || a.color[i].z != b.color[i].z) ? 1 : 0;
    }
    CHECK(diff == 0);
    CHECK(count_mismatches(b, 1e-4f) == 0);
}

TEST_CASE("raster: the surface layer occludes splats behind it")
{
    SyntheticSceneParams p;
    p.n_gaussians = 300;
    p.with_mesh   = false;
    const Scene scene    = make_synthetic_scene(p);
    const Composite open = render(scene, scene.cameras[0], 16, 1, false);
    const Composite hid  = render(scene, scene.cameras[0], 16, 1, true);
    CHECK(count_mismatches(hid, 1e-4f) == 0);
    // A surface can only remove splats from a pixel, never add them, and
    // where there is no surface nothing changes.
    const int W = 64;
    double removed = 0.0;
    bool monotone = true, untouched = true;
    for(std::size_t i = 0; i < hid.alpha.size(); ++i)
    {
        monotone = monotone && hid.alpha[i] <= open.alpha[i] + 1e-6f;
        if(static_cast<int>(i % W) <= W / 3)
        {
            untouched = untouched && hid.alpha[i] == open.alpha[i];
        }
        removed += open.alpha[i] - hid.alpha[i];
    }
    CHECK(monotone);
    CHECK(untouched);
    CHECK(removed > 1.0);
}
