// tsr_render - render the cameras of a .tsr scene to PPM images.
//
//   tsr_render SCENE [--out PREFIX] [--camera N] [--threads N] [--tile 4|8|16]
//              [--no-tonemap] [--antialiased] [--hdr] [--stats]
//              [--synthetic SEED [--gaussians N] [--size W H]]
//
// Every camera is rendered unless --camera selects one. With more than one
// view and --threads > 1 the views are rendered in parallel.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "tsr/io/image.h"
#include "tsr/render/renderer.h"
#include "tsr/sched/frame_scheduler.h"

namespace
{
struct Options
{
    std::string scene_path;
    std::string out_prefix = "frame";
    int camera             = -1;
    std::size_t threads    = 1;
    std::uint32_t tile     = 16;
    bool tonemap           = true;
    bool antialiased       = false;
    bool hdr               = false;
    bool stats             = false;
    bool synthetic         = false;
    std::uint32_t seed     = 1;
    std::uint32_t gaussians = 1000;
    std::uint32_t width    = 160;
    std::uint32_t height   = 120;
};

[[noreturn]] void usage(const char *msg)
{
    if(msg != nullptr)
    {
        std::fprintf(stderr, "tsr_render: %s\n", msg);
    }
    std::fprintf(stderr,
                 "usage: tsr_render SCENE [--out PREFIX] [--camera N] [--threads N] [--tile 4|8|16]\n"
                 "                  [--no-tonemap] [--antialiased] [--hdr] [--stats]\n"
                 "       tsr_render --synthetic SEED [--gaussians N] [--size W H] [options]\n");
    std::exit(2);
}

Options parse(int argc, char **argv)
{
    Options o;
    auto need = [&](int &i, const char *flag)
    {
        if(i + 1 >= argc)
        {
            usage((std::string("missing value for ") + flag).c_str());
        }
        return std::string(argv[++i]);
    };
    for(int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if(a == "--out")
        {
            o.out_prefix = need(i, "--out");
        }
        else if(a == "--camera")
        {
            o.camera = std::atoi(need(i, "--camera").c_str());
        }
        else if(a == "--threads")
        {
            o.threads = static_cast<std::size_t>(std::atoi(need(i, "--threads").c_str()));
        }
        else if(a == "--tile")
        {
            o.tile = static_cast<std::uint32_t>(std::atoi(need(i, "--tile").c_str()));
        }
        else if(a == "--no-tonemap")
        {
            o.tonemap = false;
        }
        else if(a == "--antialiased")
        {
            o.antialiased = true;
        }
        else if(a == "--hdr")
        {
            o.hdr = true;
        }
        else if(a == "--stats")
        {
            o.stats = true;
        }
        else if(a == "--synthetic")
        {
            o.synthetic = true;
            o.seed      = static_cast<std::uint32_t>(std::atoi(need(i, "--synthetic").c_str()));
        }
        else if(a == "--gaussians")
        {
            o.gaussians = static_cast<std::uint32_t>(std::atoi(need(i, "--gaussians").c_str()));
        }
        else if(a == "--size")
        {
            o.width  = static_cast<std::uint32_t>(std::atoi(need(i, "--size").c_str()));
            o.height = static_cast<std::uint32_t>(std::atoi(need(i, "--size").c_str()));
        }
        else if(a == "-h" || a == "--help")
        {
            usage(nullptr);
        }
        else if(!a.empty() && a[0] == '-')
        {
            usage(("unknown option " + a).c_str());
        }
        else if(o.scene_path.empty())
        {
            o.scene_path = a;
        }
        else
        {
            usage("more than one scene given");
        }
    }
    if(o.scene_path.empty() && !o.synthetic)
    {
        usage("no scene given");
    }
    return o;
}

void report(std::size_t index, const tsr::Frame &f)
{
    std::printf("view %zu: %dx%d splats=%u isects=%u surface_px=%u exposure=%.4f white=%.4f scratch=%zuKiB%s\n", index, f.width,
                f.height, f.stats.visible_splats, f.stats.intersections, f.stats.surface_pixels, f.stats.tonemap.exposure,
                f.stats.tonemap.white_point, f.stats.scratch_bytes / 1024, f.stats.mesh_bins_overflowed ? " MESH-BIN-OVERFLOW" : "");
}
} // namespace

int main(int argc, char **argv)
{
    const Options o = parse(argc, argv);
    try
    {
        tsr::Scene scene;
        if(o.synthetic)
        {
            tsr::SyntheticSceneParams p;
            p.seed        = o.seed;
            p.n_gaussians = o.gaussians;
            p.width       = o.width;
            p.height      = o.height;
            scene         = tsr::make_synthetic_scene(p);
        }
        else
        {
            scene = tsr::load_scene(o.scene_path);
        }
        if(scene.cameras.empty())
        {
            std::fprintf(stderr, "tsr_render: scene has no cameras\n");
            return 1;
        }

        std::vector<tsr::Camera> cameras;
        if(o.camera >= 0)
        {
            if(static_cast<std::size_t>(o.camera) >= scene.cameras.size())
            {
                std::fprintf(stderr, "tsr_render: camera %d out of range (scene has %zu)\n", o.camera, scene.cameras.size());
                return 1;
            }
            cameras.push_back(scene.cameras[static_cast<std::size_t>(o.camera)]);
        }
        else
        {
            cameras = scene.cameras;
        }

        tsr::RenderSettings settings;
        settings.binning.tile_size      = o.tile;
        settings.tonemap.enabled        = o.tonemap;
        settings.projection.antialiased = o.antialiased;

        auto write = [&](std::size_t i, const tsr::Frame &f)
        {
            const std::size_t view = o.camera >= 0 ? static_cast<std::size_t>(o.camera) : i;
            const std::string base = o.out_prefix + "_" + std::to_string(view);
            tsr::write_ppm(base + ".ppm", f.width, f.height, f.rgba8.data());
            if(o.hdr)
            {
                tsr::write_pfm(base + ".pfm", f.width, f.height, f.linear.data());
            }
            if(o.stats)
            {
                report(view, f);
            }
        };

        if(o.threads > 1 && cameras.size() > 1)
        {
            tsr::FrameScheduler sched(scene, settings, o.threads);
            const auto frames = sched.render_all(cameras);
            for(std::size_t i = 0; i < frames.size(); ++i)
            {
                write(i, frames[i]);
            }
        }
        else
        {
            tsr::Renderer r(settings);
            r.set_scene(scene);
            for(std::size_t i = 0; i < cameras.size(); ++i)
            {
                write(i, r.render(cameras[i]));
            }
        }
    }
    catch(const std::exception &e)
    {
        std::fprintf(stderr, "tsr_render: %s\n", e.what());
        return 1;
    }
    return 0;
}
