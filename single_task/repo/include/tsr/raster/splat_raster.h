// Tessera - tile-based compositing of splats over the surface layer.
//
// One thread block per tile walks the tile's depth-sorted intersection list
// in shared-memory batches and alpha-composites splats front to back for its
// pixels. A pixel stops at the first splat at or behind its surface depth;
// the remaining transmittance then reveals the surface colour.
#pragma once

#include <cstdint>

#include "tsr/bin/tiles.h"
#include "tsr/device/memory.h"
#include "tsr/geom/camera.h"
#include "tsr/geom/projection.h"
#include "tsr/render/surface.h"

namespace tsr
{
struct SplatRasterSettings
{
    // 1: one thread per pixel. 4: each thread shades four pixels of a column,
    // which lowers occupancy needs on small tiles (only valid for 16x16 tiles).
    std::uint32_t pixels_per_thread = 1;
};

struct SplatRasterOutput
{
    int width  = 0;
    int height = 0;
    DeviceBuffer<float3> color;        // [H][W] final composite (linear radiance)
    DeviceBuffer<float> alpha;         // [H][W] splat coverage: 1 - transmittance
    DeviceBuffer<std::int32_t> last_contributor; // [H][W] tile-relative index of the last blended intersection
};

void rasterize_splats(
    const ProjectedSplats &splats,
    const TileBinning &binning,
    const SurfaceLayer &surface,
    const Camera &camera,
    const SplatRasterSettings &settings,
    SplatRasterOutput &out,
    cudaStream_t stream
);

} // namespace tsr
