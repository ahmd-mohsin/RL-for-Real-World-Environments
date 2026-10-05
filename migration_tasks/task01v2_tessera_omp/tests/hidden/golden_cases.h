// Golden frames: rendered by the original CUDA build (tools/gen_golden.cpp)
// and compared against the port by test_golden.cpp. Each case fixes a scene,
// a camera, an image size and renderer settings.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "scenes.h"
#include "tsr/render/renderer.h"

namespace hidden
{
using namespace tsr;

struct GoldenCase
{
    std::string name;
    std::function<Scene()> scene;
    int camera = 0;
    std::function<void(RenderSettings &)> tweak;
};

inline RenderSettings golden_settings()
{
    RenderSettings s;
    s.mesh_binning.bin_size          = 16;
    s.mesh_binning.max_faces_per_bin = 8192; // no overflow: dropped faces depend on atomic order
    return s;
}

inline std::vector<GoldenCase> golden_cases()
{
    return {
        {"splats_960x540", [] { return splat_scene(200000, 960, 540, 101); }, 0, [](RenderSettings &) {}},
        {"splats_tile8_1001x613", [] { return splat_scene(60000, 1001, 613, 102); }, 1, [](RenderSettings &s) { s.binning.tile_size = 8; }},
        {"splats_tile4_333x197", [] { return splat_scene(20000, 333, 197, 103); }, 2, [](RenderSettings &s) { s.binning.tile_size = 4; }},
        {"splats_ppt4_aa_800x450", [] { return splat_scene(120000, 800, 450, 104); }, 3,
         [](RenderSettings &s) {
             s.splat_raster.pixels_per_thread = 4;
             s.projection.antialiased         = true;
         }},
        {"splats_axis_aligned_640x640", [] { return splat_scene(80000, 640, 640, 105); }, 0, [](RenderSettings &s) { s.binning.accurate = false; }},
        {"mesh_960x540", [] { return mesh_scene(160, 960, 540, 106); }, 0, [](RenderSettings &) {}},
        {"mesh_k8_cull_717x405", [] { return mesh_scene(90, 717, 405, 107); }, 1,
         [](RenderSettings &s) {
             s.mesh_raster.faces_per_pixel = 8;
             s.mesh_raster.cull_backfaces  = true;
         }},
        {"hybrid_1280x720", [] { return hybrid_scene(150000, 120, 1280, 720, 108); }, 2, [](RenderSettings &) {}},
        {"hybrid_notonemap_513x1025", [] { return hybrid_scene(50000, 70, 513, 1025, 109); }, 3, [](RenderSettings &s) { s.tonemap.enabled = false; }},
        {"hybrid_bin16_480x272", [] { return hybrid_scene(30000, 60, 480, 272, 110); }, 1,
         [](RenderSettings &s) {
             s.mesh_binning.bin_size       = 16;
             s.mesh_raster.blur_radius     = 1e-5f;
             s.mesh_binning.blur_radius    = 1e-5f;
         }},
    };
}

// Binary layout of a golden file: magic, width, height, stats, rgba8[W*H].
struct GoldenHeader
{
    std::uint32_t magic = 0x54535247; // "GRST"
    std::int32_t width  = 0;
    std::int32_t height = 0;
    std::uint32_t visible_splats = 0;
    std::uint32_t intersections  = 0;
    std::uint32_t surface_pixels = 0;
    float exposure               = 0.f;
    float white_point            = 0.f;
};
} // namespace hidden
