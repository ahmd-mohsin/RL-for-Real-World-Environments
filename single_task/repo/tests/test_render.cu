// End-to-end tests of the renderer and the multi-view scheduler.
#include <cmath>
#include <vector>

#include "doctest.h"
#include "tsr/io/image.h"
#include "tsr/render/renderer.h"
#include "tsr/sched/frame_scheduler.h"

using namespace tsr;

namespace
{
bool same_pixels(const Frame &a, const Frame &b)
{
    if(a.rgba8.size() != b.rgba8.size())
    {
        return false;
    }
    for(std::size_t i = 0; i < a.rgba8.size(); ++i)
    {
        const uchar4 p = a.rgba8[i], q = b.rgba8[i];
        if(p.x != q.x || p.y != q.y || p.z != q.z || p.w != q.w)
        {
            return false;
        }
    }
    return true;
}
} // namespace

TEST_CASE("render: hybrid frame has sane statistics")
{
    SyntheticSceneParams p;
    p.n_gaussians = 500;
    const Scene scene = make_synthetic_scene(p);
    Renderer r;
    r.set_scene(scene);
    const Frame f = r.render(scene.cameras[0]);
    CHECK(f.width == 64);
    CHECK(f.height == 48);
    CHECK(f.rgba8.size() == 64u * 48u);
    CHECK(f.stats.visible_splats > 50u);
    CHECK(f.stats.visible_splats <= 500u);
    CHECK(f.stats.intersections >= f.stats.visible_splats);
    CHECK(f.stats.surface_pixels > 0u);
    CHECK_FALSE(f.stats.mesh_bins_overflowed);
    for(float a : f.alpha)
    {
        REQUIRE(a >= 0.f);
        REQUIRE(a <= 1.f);
    }
}

TEST_CASE("render: repeated renders are bit-identical")
{
    SyntheticSceneParams p;
    p.n_gaussians = 300;
    const Scene scene = make_synthetic_scene(p);
    Renderer r;
    r.set_scene(scene);
    const Frame a = r.render(scene.cameras[1]);
    const Frame b = r.render(scene.cameras[0]);
    const Frame c = r.render(scene.cameras[1]);
    CHECK(same_pixels(a, c));
    CHECK_FALSE(same_pixels(a, b));
}

TEST_CASE("render: without a mesh the background shows through")
{
    SyntheticSceneParams p;
    p.n_gaussians = 0;
    p.with_mesh   = false;
    Scene scene   = make_synthetic_scene(p);
    scene.background = make_float3(0.25f, 0.5f, 1.0f);
    Renderer r;
    r.settings().tonemap.enabled = false;
    r.set_scene(scene);
    const Frame f = r.render(scene.cameras[0]);
    CHECK(f.stats.visible_splats == 0u);
    CHECK(f.stats.surface_pixels == 0u);
    for(const float3 &c : f.linear)
    {
        REQUIRE(c.x == 0.25f);
        REQUIRE(c.z == 1.0f);
    }
}

TEST_CASE("render: splats behind the box are hidden, splats in front are not")
{
    // One bright splat behind the box and one in front, seen head on.
    Scene scene;
    scene.background = make_float3(0.f, 0.f, 0.f);
    TriangleMesh wall;
    wall.name     = "wall";
    wall.vertices = {make_float3(-0.5f, -0.5f, 0.f), make_float3(0.5f, -0.5f, 0.f), make_float3(0.5f, 0.5f, 0.f), make_float3(-0.5f, 0.5f, 0.f)};
    wall.faces    = {make_int3(0, 1, 2), make_int3(0, 2, 3)};
    scene.meshes.push_back(wall);
    scene.material.bounds_min = make_float3(-1.f, -1.f, -1.f);
    scene.material.bounds_max = make_float3(1.f, 1.f, 1.f);
    GaussianCloud &g = scene.gaussians;
    g.sh_degree      = 0;
    for(float z : {1.0f, -1.0f})
    {
        g.means.push_back(make_float3(z > 0 ? -0.2f : 0.2f, 0.f, z));
        g.quats.push_back(make_float4(1.f, 0.f, 0.f, 0.f));
        g.scales.push_back(make_float3(0.08f, 0.08f, 0.08f));
        g.opacities.push_back(0.95f);
        g.sh.insert(g.sh.end(), {1.5f, 1.5f, 1.5f});
    }
    const Camera cam = Camera::look_at(make_float3(0.f, 0.f, -4.f), make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), 40.f, 64, 64);
    Renderer r;
    r.settings().tonemap.enabled = false;
    r.set_scene(scene);
    const Frame f = r.render(cam);
    const float2 behind = cam.project(cam.world_to_camera(g.means[0]));
    const float2 front  = cam.project(cam.world_to_camera(g.means[1]));
    auto alpha_at       = [&](float2 p) { return f.alpha[static_cast<int>(p.y) * 64 + static_cast<int>(p.x)]; };
    CHECK(alpha_at(behind) == 0.f);
    CHECK(alpha_at(front) > 0.5f);
}

TEST_CASE("sched: parallel multi-view rendering equals sequential rendering")
{
    SyntheticSceneParams p;
    p.n_gaussians = 250;
    const Scene scene = make_synthetic_scene(p);
    std::vector<Camera> cams;
    for(int i = 0; i < 6; ++i)
    {
        const float a = 0.5f * i;
        cams.push_back(Camera::look_at(make_float3(3.5f * std::sin(a), 0.6f, -3.5f * std::cos(a)), make_float3(0.f, -0.2f, 0.f), make_float3(0.f, 1.f, 0.f), 50.f, 48, 40));
    }
    FrameScheduler sched(scene, RenderSettings{}, 3);
    const std::vector<Frame> par = sched.render_all(cams);
    Renderer r;
    r.set_scene(scene);
    REQUIRE(par.size() == cams.size());
    for(std::size_t i = 0; i < cams.size(); ++i)
    {
        CAPTURE(i);
        CHECK(same_pixels(par[i], r.render(cams[i])));
    }
}

TEST_CASE("io: PPM round trip and image comparison")
{
    Image8 img;
    img.width  = 5;
    img.height = 3;
    for(int i = 0; i < 15; ++i)
    {
        img.pixels.push_back(make_uchar4(static_cast<unsigned char>(i * 10), 7, static_cast<unsigned char>(255 - i), 255));
    }
    const std::string path = "tsr_test_roundtrip.ppm";
    write_ppm(path, img.width, img.height, img.pixels.data());
    const Image8 back = read_ppm(path);
    std::remove(path.c_str());
    const ImageDiff d = compare_images(img, back);
    CHECK(d.max_abs_diff == 0);
    CHECK(std::isinf(d.psnr));
    Image8 other = back;
    other.pixels[3].x += 10;
    CHECK(compare_images(img, other).differing_pixels == 1u);
}
