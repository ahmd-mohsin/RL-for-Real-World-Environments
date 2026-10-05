// Tessera - fine rasterization of binned mesh faces.
//
// For every pixel, keeps the K faces closest to the camera whose (blurred)
// footprint covers the pixel centre, sorted front to back. Outputs are in
// image layout: row 0 is the top row, column 0 the left column.
#pragma once

#include <cstdint>

#include "tsr/bin/mesh_bins.h"
#include "tsr/device/memory.h"
#include "tsr/geom/mesh.h"

namespace tsr
{
constexpr int kMaxFacesPerPixel = 8;

struct MeshRasterSettings
{
    int faces_per_pixel          = 1;
    float blur_radius            = 0.f;   // squared NDC distance
    bool perspective_correct     = true;  // perspective-correct barycentrics
    bool clip_barycentric_coords = false; // clamp barycentrics into the face
    bool cull_backfaces          = false;
};

struct MeshFragments
{
    int width           = 0;
    int height          = 0;
    int faces_per_pixel = 0;
    DeviceBuffer<std::int32_t> pix_to_face; // [H][W][K], -1 where empty
    DeviceBuffer<float> zbuf;               // [H][W][K] camera depth, -1 where empty
    DeviceBuffer<float3> bary;              // [H][W][K]
    DeviceBuffer<float> dists;              // [H][W][K] signed squared distance to the face edge
};

void rasterize_mesh_fine(
    const ViewFaces &faces,
    const MeshBins &bins,
    int width,
    int height,
    const MeshRasterSettings &settings,
    MeshFragments &out,
    cudaStream_t stream
);

} // namespace tsr
