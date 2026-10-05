// Writes the golden frames of golden_cases.h. Built against the ORIGINAL CUDA
// library by the task author (not part of the graded build):
//   gen_golden OUT_DIR
#include <cstdio>
#include <string>

#include "golden_cases.h"

int main(int argc, char **argv)
{
    const std::string dir = argc > 1 ? argv[1] : ".";
    for(const hidden::GoldenCase &c : hidden::golden_cases())
    {
        const tsr::Scene scene = c.scene();
        tsr::RenderSettings settings = hidden::golden_settings();
        c.tweak(settings);
        tsr::Renderer r(settings);
        r.set_scene(scene);
        const tsr::Frame f = r.render(scene.cameras[c.camera]);
        if(f.stats.mesh_bins_overflowed)
        {
            std::printf("%s overflows its mesh bins\n", c.name.c_str());
            return 1;
        }
        hidden::GoldenHeader h;
        h.width          = f.width;
        h.height         = f.height;
        h.visible_splats = f.stats.visible_splats;
        h.intersections  = f.stats.intersections;
        h.surface_pixels = f.stats.surface_pixels;
        h.exposure       = f.stats.tonemap.exposure;
        h.white_point    = f.stats.tonemap.white_point;
        const std::string path = dir + "/" + c.name + ".golden";
        std::FILE *fp          = std::fopen(path.c_str(), "wb");
        std::fwrite(&h, sizeof(h), 1, fp);
        std::fwrite(f.rgba8.data(), sizeof(uchar4), f.rgba8.size(), fp);
        std::fclose(fp);
        std::printf("%s %dx%d visible=%u isects=%u surface=%u\n", c.name.c_str(), f.width, f.height, h.visible_splats, h.intersections,
                    h.surface_pixels);
    }
    return 0;
}
