// End-to-end comparison with frames rendered by the original CUDA build.
//
// nvcc and an OpenMP compiler contract floating-point expressions into FMAs
// differently, so frames cannot be bit-identical; the tolerances below were
// calibrated against the reference port (worst case: 0.0002% of pixels off by
// more than 1/255, intersection counts off by 1 in 6.4M) and sit far below
// what any real rendering error produces.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "doctest.h"
#include "golden_cases.h"

#ifndef TSR_GOLDEN_DIR
#    define TSR_GOLDEN_DIR "data"
#endif

namespace
{
bool load_golden(const std::string &name, hidden::GoldenHeader &h, std::vector<uchar4> &px)
{
    const std::string path = std::string(TSR_GOLDEN_DIR) + "/" + name + ".golden";
    std::FILE *fp          = std::fopen(path.c_str(), "rb");
    if(fp == nullptr)
    {
        return false;
    }
    bool ok = std::fread(&h, sizeof(h), 1, fp) == 1 && h.magic == hidden::GoldenHeader{}.magic;
    if(ok)
    {
        px.resize(static_cast<std::size_t>(h.width) * h.height);
        ok = std::fread(px.data(), sizeof(uchar4), px.size(), fp) == px.size();
    }
    std::fclose(fp);
    return ok;
}

double rel(double a, double b) { return std::fabs(a - b) / std::fmax(1.0, std::fabs(b)); }
} // namespace

TEST_CASE("golden: frames match the original CUDA renderer")
{
    for(const hidden::GoldenCase &c : hidden::golden_cases())
    {
        CAPTURE(c.name);
        hidden::GoldenHeader g;
        std::vector<uchar4> want;
        REQUIRE(load_golden(c.name, g, want));

        const tsr::Scene scene        = c.scene();
        tsr::RenderSettings settings  = hidden::golden_settings();
        c.tweak(settings);
        tsr::Renderer r(settings);
        r.set_scene(scene);
        const tsr::Frame f = r.render(scene.cameras[c.camera]);
        REQUIRE(!f.stats.mesh_bins_overflowed);
        REQUIRE(f.width == g.width);
        REQUIRE(f.height == g.height);
        REQUIRE(f.rgba8.size() == want.size());

        std::size_t off1 = 0, off8 = 0;
        double sum = 0.0;
        int worst  = 0;
        for(std::size_t i = 0; i < want.size(); ++i)
        {
            const int d = std::max({std::abs(int(f.rgba8[i].x) - int(want[i].x)), std::abs(int(f.rgba8[i].y) - int(want[i].y)),
                                    std::abs(int(f.rgba8[i].z) - int(want[i].z)), std::abs(int(f.rgba8[i].w) - int(want[i].w))});
            off1 += d > 1 ? 1 : 0;
            off8 += d > 8 ? 1 : 0;
            sum  += d;
            worst = std::max(worst, d);
        }
        const double n = static_cast<double>(want.size());
        MESSAGE(c.name << ": >1 " << off1 / n * 100.0 << "%  >8 " << off8 / n * 100.0 << "%  mean " << sum / n << "  max " << worst
                       << "  isects " << f.stats.intersections << "/" << g.intersections << "  visible " << f.stats.visible_splats << "/"
                       << g.visible_splats << "  surface " << f.stats.surface_pixels << "/" << g.surface_pixels);
        CHECK(off1 / n <= 0.0005);
        CHECK(off8 / n <= 0.00002);
        CHECK(sum / n <= 0.01);
        CHECK(rel(f.stats.intersections, g.intersections) <= 1e-4);
        CHECK(rel(f.stats.visible_splats, g.visible_splats) <= 1e-4);
        CHECK(rel(f.stats.surface_pixels, g.surface_pixels) <= 2e-4);
        CHECK(rel(f.stats.tonemap.exposure, g.exposure) <= 1e-3);
        CHECK(rel(f.stats.tonemap.white_point, g.white_point) <= 0.05);
    }
}
