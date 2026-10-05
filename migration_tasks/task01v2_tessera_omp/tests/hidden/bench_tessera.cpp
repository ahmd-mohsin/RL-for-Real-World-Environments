// Hidden performance benchmark for the Tessera port.
//
// Builds against the original CUDA library and against a port through the
// public API only (streams are passed as nullptr). Every timed region ends
// with a 1-element download of the stage's output, so asynchronous work is
// counted. One JSON line per case: {"case", "median_ms"}.
//
//   tsr_bench [--reps R] [--quick]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "scenes.h"
#include "tsr/render/renderer.h"
#include "tsr/sched/frame_scheduler.h"

namespace
{
using namespace tsr;

template<typename F>
double median_ms(int reps, F &&f)
{
    f();
    f();
    std::vector<double> t;
    for(int r = 0; r < reps; ++r)
    {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

void emit(const std::string &name, double ms)
{
    std::printf("{\"case\": \"%s\", \"median_ms\": %.4f}\n", name.c_str(), ms);
    std::fflush(stdout);
}

template<typename T>
void touch(const DeviceBuffer<T> &b)
{
    if(b.size() > 0)
    {
        T x{};
        b.download(&x, 1);
    }
}

RenderSettings bench_settings()
{
    RenderSettings s;
    s.mesh_binning.bin_size          = 32;
    s.mesh_binning.max_faces_per_bin = 512;
    s.mesh_raster.faces_per_pixel    = 1;
    return s;
}

// Time every stage of the pipeline separately, in renderer order.
void bench_stages(const std::string &tag, const Scene &scene, const RenderSettings &settings, int reps)
{
    const Camera &cam = scene.cameras[0];
    const int W       = static_cast<int>(cam.width);
    const int H       = static_cast<int>(cam.height);
    Arena arena(std::size_t{64} << 20);

    DeviceGaussians gaussians;
    gaussians.upload(scene.gaussians, nullptr);
    DeviceMesh mesh;
    mesh.upload(scene.meshes, nullptr);
    NeuralMaterialModel material;
    if(mesh.n_faces > 0)
    {
        material.initialize(scene.material, nullptr);
    }

    ViewFaces faces;
    MeshBins bins;
    MeshFragments frags;
    SurfaceLayer surface;
    ProjectedSplats splats;
    TileBinning binning;
    SplatRasterOutput comp;
    DeviceBuffer<uchar4> ldr(static_cast<std::size_t>(W) * H);
    TonemapStats tstats;

    if(mesh.n_faces > 0)
    {
        emit(tag + "/transform_faces", median_ms(reps, [&] { transform_faces(mesh, cam, faces, nullptr); touch(faces.ndc); }));
        emit(tag + "/bin_mesh_faces", median_ms(reps, [&] { arena.reset(); bin_mesh_faces(faces, W, H, settings.mesh_binning, bins, arena, nullptr); touch(bins.faces_per_bin); }));
        emit(tag + "/rasterize_mesh_fine", median_ms(reps, [&] { rasterize_mesh_fine(faces, bins, W, H, settings.mesh_raster, frags, nullptr); touch(frags.zbuf); }));
        emit(tag + "/shade_surface", median_ms(reps, [&] { arena.reset(); shade_surface(material, frags, faces, cam, scene.background, surface, arena, nullptr); touch(surface.color); }));
    }
    else
    {
        clear_surface(W, H, scene.background, surface, nullptr);
    }
    if(gaussians.count > 0)
    {
        emit(tag + "/project_gaussians", median_ms(reps, [&] { project_gaussians(gaussians, cam, settings.projection, splats, nullptr); touch(splats.conics); }));
        emit(tag + "/compute_splat_colors", median_ms(reps, [&] { compute_splat_colors(gaussians, cam, splats, nullptr); touch(splats.colors); }));
        emit(tag + "/bin_splats", median_ms(reps, [&] { arena.reset(); bin_splats(splats, cam, settings.binning, binning, arena, nullptr); touch(binning.tile_offsets); }));
        emit(tag + "/rasterize_splats", median_ms(reps, [&] { rasterize_splats(splats, binning, surface, cam, settings.splat_raster, comp, nullptr); touch(comp.color); }));
        emit(tag + "/tonemap", median_ms(reps, [&] { arena.reset(); tonemap(comp.color.data(), W, H, settings.tonemap, ldr.data(), &tstats, arena, nullptr); touch(ldr); }));
    }
}

void bench_frame(const std::string &tag, const Scene &scene, const RenderSettings &settings, int reps)
{
    Renderer r(settings);
    r.set_scene(scene);
    std::size_t i = 0;
    emit(tag + "/frame", median_ms(reps, [&] { (void)r.render(scene.cameras[i++ % scene.cameras.size()]); }));
}

void bench_multiview(const std::string &tag, const Scene &scene, const RenderSettings &settings, int reps)
{
    std::vector<Camera> cams;
    for(int k = 0; k < 2; ++k)
    {
        cams.insert(cams.end(), scene.cameras.begin(), scene.cameras.end());
    }
    FrameScheduler sched(scene, settings, 4);
    emit(tag + "/multiview8", median_ms(reps, [&] { (void)sched.render_all(cams); }));
}
} // namespace

int main(int argc, char **argv)
{
    int reps   = 9;
    bool quick = false;
    for(int i = 1; i < argc; ++i)
    {
        if(!std::strcmp(argv[i], "--reps") && i + 1 < argc)
        {
            reps = std::atoi(argv[++i]);
        }
        else if(!std::strcmp(argv[i], "--quick"))
        {
            quick = true;
        }
    }
    const std::uint32_t W = quick ? 640 : 1920, H = quick ? 360 : 1080;
    const RenderSettings settings = bench_settings();

    const Scene splats = hidden::splat_scene(quick ? 100000 : 1000000, W, H);
    bench_stages("splats", splats, settings, reps);
    bench_frame("splats", splats, settings, reps);

    const Scene mesh = hidden::mesh_scene(quick ? 100 : 300, W, H);
    bench_stages("mesh", mesh, settings, reps);
    bench_frame("mesh", mesh, settings, reps);

    const Scene hybrid = hidden::hybrid_scene(quick ? 50000 : 400000, quick ? 80 : 200, W, H);
    bench_stages("hybrid", hybrid, settings, reps);
    bench_frame("hybrid", hybrid, settings, reps);

    const Scene mv = hidden::hybrid_scene(quick ? 20000 : 150000, quick ? 60 : 150, W / 2, H / 2, 31);
    bench_multiview("hybrid_half", mv, settings, std::max(3, reps / 3));
    return 0;
}
