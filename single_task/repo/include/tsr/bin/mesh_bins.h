// Tessera - coarse binning of mesh faces for the fine rasterizer.
//
// Bins are bin_size x bin_size squares of RASTER space (raster index grows
// towards +x/+y NDC, i.e. right-to-left and bottom-to-top in the image).
// bin_faces[bin][0 .. faces_per_bin[bin]) lists the faces whose (blurred)
// screen bounding boxes overlap the bin; unused slots hold -1.
#pragma once

#include <cstdint>

#include "tsr/device/memory.h"
#include "tsr/geom/mesh.h"

namespace tsr
{
struct MeshBinningSettings
{
    int bin_size          = 16;
    int max_faces_per_bin = 256;
    float blur_radius     = 0.f; // squared NDC distance, as in the fine rasterizer
};

struct MeshBins
{
    int bin_size          = 0;
    int bins_x            = 0;
    int bins_y            = 0;
    int max_faces_per_bin = 0;
    DeviceBuffer<std::int32_t> faces_per_bin; // [bins_y][bins_x]
    DeviceBuffer<std::int32_t> bin_faces;     // [bins_y][bins_x][max_faces_per_bin]
    // True if some bin received more faces than it can hold. The overflowing
    // faces are dropped, so the output may be incomplete.
    bool overflowed = false;
};

void bin_mesh_faces(
    const ViewFaces &faces,
    int width,
    int height,
    const MeshBinningSettings &settings,
    MeshBins &out,
    Arena &arena,
    cudaStream_t stream
);

} // namespace tsr
