// Hidden stress tests: binning order invariants at scale (including massive
// depth ties), concurrency, determinism, MLP shapes and error paths.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "doctest.h"
#include "scenes.h"
#include "tsr/bin/tiles.h"
#include "tsr/encode/mlp.h"
#include "tsr/geom/projection.h"
#include "tsr/render/renderer.h"
#include "tsr/sched/frame_scheduler.h"

using namespace tsr;

namespace
{
std::uint32_t bits_of(float f)
{
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// A camera at z = -3 looking down +z, and a w x h grid of Gaussians on the
// plane z = 0: every splat has camera depth exactly 3, so all intersections
// within a tile tie on depth and must stay in splat-id order.
Scene plane_scene(int gw, int gh, float sigma, std::uint32_t W, std::uint32_t H, std::uint32_t seed)
{
    Scene s;
    s.background = make_float3(0.1f, 0.1f, 0.1f);
    GaussianCloud &g = s.gaussians;
    g.sh_degree      = 0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> jitter(-0.3f, 0.3f), col(0.f, 1.f);
    for(int j = 0; j < gh; ++j)
    {
        for(int i = 0; i < gw; ++i)
        {
            const float x = -1.6f + 3.2f * (i + 0.5f + jitter(rng)) / gw;
            const float y = -0.9f + 1.8f * (j + 0.5f + jitter(rng)) / gh;
            g.means.push_back(make_float3(x, y, 0.f));
            g.quats.push_back(make_float4(1.f, 0.f, 0.f, 0.f));
            g.scales.push_back(make_float3(sigma, sigma, 1e-4f));
            g.opacities.push_back(0.35f + 0.6f * col(rng));
            g.sh.push_back(col(rng) - 0.5f);
            g.sh.push_back(col(rng) - 0.5f);
            g.sh.push_back(col(rng) - 0.5f);
        }
    }
    s.cameras.push_back(Camera::look_at(make_float3(0.f, 0.f, -3.f), make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), 40.f, W, H));
    return s;
}

struct Binned
{
    std::vector<std::uint64_t> keys;
    std::vector<std::int32_t> ids, offsets;
    std::vector<std::int64_t> cum;
    std::vector<float> depths;
    std::vector<int2> radii;
    std::uint32_t n_isects = 0, n_tiles = 0;
};

Binned bin_scene(const Scene &scene, std::uint32_t tile_size, bool accurate)
{
    DeviceGaussians g;
    g.upload(scene.gaussians, nullptr);
    ProjectedSplats sp;
    project_gaussians(g, scene.cameras[0], {}, sp, nullptr);
    compute_splat_colors(g, scene.cameras[0], sp, nullptr);
    Arena arena(std::size_t{16} << 20);
    TileBinning tb;
    BinningSettings bs;
    bs.tile_size = tile_size;
    bs.accurate  = accurate;
    bin_splats(sp, scene.cameras[0], bs, tb, arena, nullptr);
    Binned b;
    b.n_isects = tb.n_isects;
    b.n_tiles  = tb.n_tiles();
    b.keys     = tb.sorted_keys.to_host();
    b.ids      = tb.sorted_splat_ids.to_host();
    b.offsets  = tb.tile_offsets.to_host();
    b.cum      = tb.tiles_per_splat.to_host();
    b.depths   = sp.depths.to_host();
    b.radii    = sp.radii.to_host();
    return b;
}

// Every invariant of a correct binning, checked on the output alone.
void check_binning(const Binned &b)
{
    REQUIRE(b.keys.size() >= b.n_isects);
    REQUIRE(b.ids.size() >= b.n_isects);
    REQUIRE(b.offsets.size() == static_cast<std::size_t>(b.n_tiles) + 1);
    REQUIRE(static_cast<std::int64_t>(b.n_isects) == (b.cum.empty() ? 0 : b.cum.back()));
    CHECK(b.offsets.front() == 0);
    CHECK(b.offsets.back() == static_cast<std::int32_t>(b.n_isects));
    std::size_t bad_order = 0, bad_key = 0, bad_offset = 0;
    std::vector<std::int64_t> per_splat(b.cum.size(), 0);
    for(std::uint32_t t = 0; t < b.n_tiles; ++t)
    {
        if(b.offsets[t] > b.offsets[t + 1])
        {
            ++bad_offset;
            continue;
        }
        for(std::int32_t j = b.offsets[t]; j < b.offsets[t + 1]; ++j)
        {
            const std::uint64_t k = b.keys[j];
            const std::int32_t id = b.ids[j];
            if(id < 0 || static_cast<std::size_t>(id) >= b.cum.size())
            {
                ++bad_key;
                continue;
            }
            ++per_splat[id];
            if((k >> 32) != t || static_cast<std::uint32_t>(k) != bits_of(b.depths[id]))
            {
                ++bad_key;
            }
            if(j > b.offsets[t])
            {
                const std::uint64_t kp = b.keys[j - 1];
                const bool ordered     = static_cast<std::uint32_t>(kp) < static_cast<std::uint32_t>(k)
                                     || (static_cast<std::uint32_t>(kp) == static_cast<std::uint32_t>(k) && b.ids[j - 1] < id);
                bad_order += ordered ? 0 : 1;
            }
        }
    }
    std::size_t bad_count = 0;
    for(std::size_t i = 0; i < b.cum.size(); ++i)
    {
        const std::int64_t want = b.cum[i] - (i == 0 ? 0 : b.cum[i - 1]);
        bad_count += per_splat[i] == want ? 0 : 1;
    }
    CHECK(bad_offset == 0);
    CHECK(bad_key == 0);
    CHECK(bad_order == 0);
    CHECK(bad_count == 0);
}
} // namespace

TEST_CASE("stress: binning order with massive depth ties")
{
    struct Cfg
    {
        int gw, gh;
        float sigma;
        std::uint32_t W, H, tile;
    };
    // intersection counts from a few hundred to millions, crossing every
    // size threshold a sort implementation is likely to have
    for(const Cfg c : {Cfg{6, 4, 0.01f, 64, 48, 16}, Cfg{20, 12, 0.02f, 160, 120, 16}, Cfg{40, 24, 0.03f, 320, 180, 8},
                       Cfg{90, 50, 0.02f, 640, 360, 16}, Cfg{200, 112, 0.012f, 960, 540, 4}, Cfg{300, 170, 0.03f, 1280, 720, 16},
                       Cfg{500, 280, 0.05f, 1920, 1080, 16}})
    {
        CAPTURE(c.gw);
        CAPTURE(c.tile);
        const Scene s  = plane_scene(c.gw, c.gh, c.sigma, c.W, c.H, static_cast<std::uint32_t>(c.gw * 7 + c.gh));
        const Binned b = bin_scene(s, c.tile, true);
        MESSAGE("plane " << c.gw << "x" << c.gh << " tile " << c.tile << ": " << b.n_isects << " intersections");
        CHECK(b.n_isects > 0);
        check_binning(b);
    }
}

TEST_CASE("stress: binning invariants on random scenes, both binning modes")
{
    for(const std::uint32_t n : {1u, 3u, 700u, 5000u, 60000u, 400000u})
    {
        for(const bool accurate : {true, false})
        {
            CAPTURE(n);
            CAPTURE(accurate);
            const Scene s  = hidden::splat_scene(n, n > 10000 ? 1280 : 333, n > 10000 ? 720 : 211, 900 + n);
            const Binned b = bin_scene(s, n % 3 == 0 ? 8 : 16, accurate);
            check_binning(b);
        }
    }
}

TEST_CASE("stress: no visible splats and fully culled scenes")
{
    Scene s = plane_scene(10, 10, 0.02f, 128, 96, 5);
    for(float3 &m : s.gaussians.means)
    {
        m.z = -10.f; // behind the camera
    }
    const Binned b = bin_scene(s, 16, true);
    CHECK(b.n_isects == 0);
    CHECK(std::all_of(b.offsets.begin(), b.offsets.end(), [](std::int32_t o) { return o == 0; }));
    Renderer r;
    r.set_scene(s);
    const Frame f = r.render(s.cameras[0]);
    CHECK(f.stats.visible_splats == 0);
    CHECK(f.stats.intersections == 0);
}

TEST_CASE("stress: 16 render threads x 32 views equal sequential rendering")
{
    const Scene scene = hidden::hybrid_scene(40000, 60, 400, 240, 77);
    std::vector<Camera> cams;
    for(int k = 0; k < 8; ++k)
    {
        cams.insert(cams.end(), scene.cameras.begin(), scene.cameras.end());
    }
    RenderSettings settings;
    settings.mesh_binning.bin_size          = 16;
    settings.mesh_binning.max_faces_per_bin = 8192; // overflow drops faces in atomic order
    std::vector<Frame> seq;
    {
        Renderer r(settings);
        r.set_scene(scene);
        for(const Camera &c : cams)
        {
            seq.push_back(r.render(c));
        }
    }
    FrameScheduler sched(scene, settings, 16);
    const std::vector<Frame> par = sched.render_all(cams);
    REQUIRE(par.size() == seq.size());
    std::size_t differing = 0;
    for(std::size_t i = 0; i < seq.size(); ++i)
    {
        REQUIRE(!seq[i].stats.mesh_bins_overflowed);
        differing += (par[i].rgba8.size() == seq[i].rgba8.size()
                      && std::memcmp(par[i].rgba8.data(), seq[i].rgba8.data(), seq[i].rgba8.size() * sizeof(uchar4)) == 0
                      && par[i].stats.intersections == seq[i].stats.intersections && par[i].stats.visible_splats == seq[i].stats.visible_splats
                      && par[i].stats.surface_pixels == seq[i].stats.surface_pixels)
                         ? 0
                         : 1;
    }
    CHECK(differing == 0);
}

TEST_CASE("stress: independent renderers on raw threads agree")
{
    const Scene scene = hidden::splat_scene(30000, 320, 200, 55);
    Frame ref;
    {
        Renderer r;
        r.set_scene(scene);
        ref = r.render(scene.cameras[1]);
    }
    std::vector<Frame> got(12);
    std::vector<std::thread> threads;
    for(std::size_t t = 0; t < got.size(); ++t)
    {
        threads.emplace_back([&, t] {
            Renderer r;
            r.set_scene(scene);
            for(int rep = 0; rep < 3; ++rep)
            {
                got[t] = r.render(scene.cameras[1]);
            }
        });
    }
    for(std::thread &th : threads)
    {
        th.join();
    }
    for(const Frame &f : got)
    {
        CHECK(f.rgba8.size() == ref.rgba8.size());
        CHECK(std::memcmp(f.rgba8.data(), ref.rgba8.data(), ref.rgba8.size() * sizeof(uchar4)) == 0);
        CHECK(f.stats.tonemap.exposure == ref.stats.tonemap.exposure);
    }
}

TEST_CASE("stress: repeated large renders are bit-identical")
{
    const Scene scene = hidden::splat_scene(300000, 1280, 720, 66);
    Renderer r;
    r.set_scene(scene);
    const Frame a = r.render(scene.cameras[0]);
    const Frame b = r.render(scene.cameras[2]);
    const Frame c = r.render(scene.cameras[0]);
    CHECK(std::memcmp(a.rgba8.data(), c.rgba8.data(), a.rgba8.size() * sizeof(uchar4)) == 0);
    CHECK(std::memcmp(a.linear.data(), c.linear.data(), a.linear.size() * sizeof(float3)) == 0);
    CHECK(a.stats.intersections == c.stats.intersections);
    CHECK(b.width == a.width);
}

namespace
{
std::vector<float> host_mlp(const Mlp &mlp, const std::vector<float> &in, std::uint32_t M)
{
    std::vector<float> cur = in;
    for(std::size_t l = 0; l < mlp.n_layers(); ++l)
    {
        const std::uint32_t K = mlp.layer_in(l), N = mlp.layer_out(l);
        const std::vector<float> W = mlp.weights_to_host(l), B = mlp.biases_to_host(l);
        std::vector<float> nxt(static_cast<std::size_t>(M) * N);
        const bool last = l + 1 == mlp.n_layers();
        for(std::uint32_t r = 0; r < M; ++r)
        {
            for(std::uint32_t o = 0; o < N; ++o)
            {
                double acc = 0.0;
                for(std::uint32_t k = 0; k < K; ++k)
                {
                    acc += static_cast<double>(cur[static_cast<std::size_t>(r) * K + k]) * W[static_cast<std::size_t>(k) * N + o];
                }
                const float v = static_cast<float>(acc) + B[o];
                nxt[static_cast<std::size_t>(r) * N + o] = last ? 1.f / (1.f + std::exp(-v)) : std::max(v, 0.f);
            }
        }
        cur.swap(nxt);
    }
    return cur;
}
} // namespace

TEST_CASE("stress: MLP forward matches a host reference for every network shape")
{
    struct Shape
    {
        std::uint32_t in, width, layers, out, M;
    };
    for(const Shape sh : {Shape{16, 16, 2, 3, 1}, Shape{32, 64, 2, 3, 70001}, Shape{64, 64, 3, 3, 4097}, Shape{64, 32, 4, 4, 1000},
                          Shape{24, 48, 2, 3, 999}, Shape{8, 16, 1, 3, 257}, Shape{96, 64, 2, 3, 300}, Shape{16, 128, 2, 3, 100},
                          Shape{32, 16, 0, 3, 50}})
    {
        CAPTURE(sh.in);
        CAPTURE(sh.width);
        CAPTURE(sh.layers);
        Mlp mlp;
        MlpConfig cfg;
        cfg.hidden_width  = sh.width;
        cfg.hidden_layers = sh.layers;
        mlp.initialize(sh.in, sh.out, cfg, 1234 + sh.width + sh.layers, nullptr);
        std::mt19937 rng(sh.M);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        std::vector<float> x(static_cast<std::size_t>(sh.M) * sh.in);
        for(float &v : x)
        {
            v = u(rng);
        }
        DeviceBuffer<float> dx, dy(static_cast<std::size_t>(sh.M) * sh.out);
        dx.upload(x);
        Arena arena(std::size_t{8} << 20);
        mlp.forward(dx.data(), sh.M, dy.data(), arena, nullptr);
        const std::vector<float> got = dy.to_host(), want = host_mlp(mlp, x, sh.M);
        float worst = 0.f;
        for(std::size_t i = 0; i < want.size(); ++i)
        {
            worst = std::max(worst, std::fabs(got[i] - want[i]));
        }
        CHECK(worst < 2e-5f);
    }
}

TEST_CASE("stress: error paths")
{
    Renderer r;
    const Scene s = hidden::splat_scene(100, 64, 48, 1);
    CHECK_THROWS_AS((void)r.render(s.cameras[0]), tsr::Error);
    r.set_scene(s);
    Camera empty = s.cameras[0];
    empty.width  = 0;
    CHECK_THROWS_AS((void)r.render(empty), tsr::Error);
    RenderSettings bad;
    bad.mesh_raster.faces_per_pixel = kMaxFacesPerPixel + 1;
    Renderer r2(bad);
    const Scene m = hidden::mesh_scene(10, 64, 48, 2);
    r2.set_scene(m);
    CHECK_THROWS_AS((void)r2.render(m.cameras[0]), tsr::Error);
    RenderSettings tiles;
    tiles.binning.tile_size = 32;
    Renderer r3(tiles);
    r3.set_scene(s);
    CHECK_THROWS_AS((void)r3.render(s.cameras[0]), tsr::Error);
}
