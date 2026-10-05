// Tests for tone mapping and the scene format.
#include <cmath>
#include <string>
#include <vector>

#include "doctest.h"
#include "tsr/post/tonemap.h"
#include "tsr/scene/scene.h"

using namespace tsr;

TEST_CASE("post: sRGB transfer functions are inverse")
{
    for(float x : {0.f, 0.001f, 0.0031308f, 0.2f, 0.5f, 1.f})
    {
        CHECK(srgb_to_linear(linear_to_srgb(x)) == doctest::Approx(x).epsilon(1e-4));
    }
    CHECK(linear_to_srgb(2.f) == doctest::Approx(1.f));
}

TEST_CASE("post: exposure maps the log-average luminance onto the key")
{
    const int W = 32, H = 16;
    std::vector<float3> hdr(W * H);
    for(int i = 0; i < W * H; ++i)
    {
        const float v = 0.05f + 4.0f * (i % 17) / 16.f;
        hdr[i]        = make_float3(v, v, v);
    }
    DeviceBuffer<float3> d;
    d.upload(hdr);
    DeviceBuffer<uchar4> out(W * H);
    Arena arena;
    TonemapStats stats;
    tonemap(d.data(), W, H, TonemapSettings{}, out.data(), &stats, arena, nullptr);

    double log_sum = 0;
    for(const auto &c : hdr)
    {
        log_sum += std::log(1e-4 + 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z);
    }
    const double log_avg = std::exp(log_sum / hdr.size());
    CHECK(stats.log_average_luminance == doctest::Approx(log_avg).epsilon(1e-3));
    CHECK(stats.exposure == doctest::Approx(0.18 / log_avg).epsilon(1e-3));
    std::uint32_t total = 0;
    for(auto h : stats.histogram)
    {
        total += h;
    }
    CHECK(total == static_cast<std::uint32_t>(W * H));
    // brighter input -> brighter output, and nothing exceeds 255
    const auto px = out.to_host();
    CHECK(px[16].x >= px[1].x);
    CHECK(px[0].w == 255);
}

TEST_CASE("post: disabled tone mapping clamps and encodes")
{
    std::vector<float3> hdr = {make_float3(0.f, 0.5f, 3.f)};
    DeviceBuffer<float3> d;
    d.upload(hdr);
    DeviceBuffer<uchar4> out(1);
    Arena arena;
    TonemapSettings s;
    s.enabled = false;
    tonemap(d.data(), 1, 1, s, out.data(), nullptr, arena, nullptr);
    const uchar4 p = out.to_host()[0];
    CHECK(p.x == 0);
    CHECK(p.y == static_cast<unsigned char>(linear_to_srgb(0.5f) * 255.f + 0.5f));
    CHECK(p.z == 255);
}

TEST_CASE("scene: parse a complete scene")
{
    const std::string text = R"(tessera-scene 1
# a comment
background 0.1 0.2 0.3
camera lookat 0 0 -5  0 0 0  0 1 0  fov 40 size 32 24
camera ortho 0 0 -5  0 0 0  0 1 0  size 16 16 scale 8
sh_degree 1
gaussian 0 0 0  1 0 0 0  0.1 0.1 0.1  0.8   0.1 0.2 0.3  0 0 0  0 0 0  0 0 0
mesh quad
v -1 -1 0
v 1 -1 0
v 1 1 0
f 1 2 3
end
material grid 4 2 10 4 1.5 mlp 8 1 seed 42 bounds -1 -1 -1 1 1 1 tint 1 0.5 0.5
)";
    const Scene s = parse_scene(text);
    CHECK(s.background.y == doctest::Approx(0.2f));
    REQUIRE(s.cameras.size() == 2);
    CHECK(s.cameras[0].width == 32);
    CHECK(s.cameras[1].model == CameraModel::Orthographic);
    CHECK(s.cameras[1].fx == 8.f);
    CHECK(s.gaussians.size() == 1);
    CHECK(s.gaussians.sh.size() == 12);
    REQUIRE(s.meshes.size() == 1);
    CHECK(s.meshes[0].faces[0].z == 2);
    CHECK(s.material.seed == 42u);
    CHECK(s.material.grid.n_levels == 4u);
    CHECK(s.material.tint.y == doctest::Approx(0.5f));
}

TEST_CASE("scene: errors carry the line number")
{
    auto error_of = [](const std::string &text)
    {
        try
        {
            parse_scene(text, "scene.tsr");
        }
        catch(const Error &e)
        {
            return std::string(e.what());
        }
        return std::string();
    };
    CHECK(error_of("tessera-scene 1\nbogus 1 2\n").find("scene.tsr:2") != std::string::npos);
    CHECK(error_of("tessera-scene 1\nmesh m\nv 0 0 0\nf 1 2 3\nend\n").find("scene.tsr:4") != std::string::npos);
    CHECK(error_of("tessera-scene 2\n").find("version") != std::string::npos);
    CHECK(error_of("tessera-scene 1\nmesh m\nv 0 0 0\n").find("missing its 'end'") != std::string::npos);
    CHECK(error_of("tessera-scene 1\ngaussian 0 0 0 1 0 0 0 0.1 0.1 0.1 1.5 0 0 0\n").find("opacity") != std::string::npos);
}
