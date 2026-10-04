// Tessera - the hybrid splat + mesh renderer.
//
// Per view:
//   surface   mesh faces -> NDC -> coarse bins -> fine raster -> neural shading
//   splats    projection -> SH colour -> tile binning (scan + sort)
//   composite tile rasterizer blends splats in front of the surface layer
//   post      exposure, white point and tone mapping to 8-bit sRGB
#pragma once

#include <cstdint>
#include <vector>

#include "tsr/bin/mesh_bins.h"
#include "tsr/bin/tiles.h"
#include "tsr/device/memory.h"
#include "tsr/geom/mesh.h"
#include "tsr/geom/projection.h"
#include "tsr/post/tonemap.h"
#include "tsr/raster/mesh_raster.h"
#include "tsr/raster/splat_raster.h"
#include "tsr/render/surface.h"
#include "tsr/scene/scene.h"

namespace tsr
{
struct RenderSettings
{
    ProjectionSettings projection;
    BinningSettings binning;
    SplatRasterSettings splat_raster;
    MeshBinningSettings mesh_binning;
    MeshRasterSettings mesh_raster;
    TonemapSettings tonemap;
};

struct FrameStats
{
    std::uint32_t visible_splats   = 0;
    std::uint32_t intersections    = 0;
    std::uint32_t surface_pixels   = 0;
    bool mesh_bins_overflowed      = false;
    std::size_t scratch_bytes      = 0; // arena high-water mark
    TonemapStats tonemap;
};

struct Frame
{
    int width  = 0;
    int height = 0;
    std::vector<float3> linear; // composite before tone mapping
    std::vector<float> alpha;   // splat coverage
    std::vector<uchar4> rgba8;  // tone-mapped sRGB
    FrameStats stats;
};

class Renderer
{
public:
    explicit Renderer(const RenderSettings &settings = {});
    ~Renderer();
    Renderer(const Renderer &)            = delete;
    Renderer &operator=(const Renderer &) = delete;

    // Uploads geometry and builds the neural material. Must precede render().
    void set_scene(const Scene &scene);

    Frame render(const Camera &camera);

    [[nodiscard]] const RenderSettings &settings() const { return settings_; }
    RenderSettings &settings() { return settings_; }

    // Intermediate results of the last render(), for inspection and tests.
    [[nodiscard]] const ProjectedSplats &splats() const { return splats_; }
    [[nodiscard]] const TileBinning &binning() const { return binning_; }
    [[nodiscard]] const MeshFragments &fragments() const { return fragments_; }
    [[nodiscard]] const SurfaceLayer &surface() const { return surface_; }
    [[nodiscard]] const SplatRasterOutput &composite() const { return composite_; }
    [[nodiscard]] cudaStream_t stream() const { return stream_; }

private:
    void render_surface(const Camera &camera);
    std::uint32_t count_visible_splats();

    RenderSettings settings_;
    cudaStream_t stream_ = nullptr;
    Arena arena_;
    bool has_scene_ = false;

    DeviceGaussians gaussians_;
    DeviceMesh mesh_;
    NeuralMaterialModel material_;
    float3 background_{};

    ProjectedSplats splats_;
    TileBinning binning_;
    ViewFaces faces_;
    MeshBins mesh_bins_;
    MeshFragments fragments_;
    SurfaceLayer surface_;
    SplatRasterOutput composite_;
    DeviceBuffer<uchar4> ldr_;
    bool mesh_overflow_ = false;
};

} // namespace tsr
