// Tessera - triangle meshes on the device and their view transform.
//
// The mesh rasterizer works in the NDC convention it was written for:
//   +x points LEFT and +y points UP; the shorter image side spans [-1, 1] and
//   the longer side is scaled by the aspect ratio; z stays camera depth.
// mesh_ndc.cuh converts between this space and the pixel space used by the
// splat pipeline (origin top-left, +x right, +y down, pixel centres at +0.5).
#pragma once

#include <cstdint>
#include <vector>

#include "tsr/device/memory.h"
#include "tsr/geom/camera.h"
#include "tsr/scene/scene.h"

namespace tsr
{
// All meshes of a scene concatenated; face indices refer to the combined
// vertex array and face ids are global.
struct DeviceMesh
{
    DeviceBuffer<float3> vertices;
    DeviceBuffer<int3> faces;
    std::uint32_t n_vertices = 0;
    std::uint32_t n_faces    = 0;

    void upload(const std::vector<TriangleMesh> &meshes, cudaStream_t stream);
};

// Per-view face data consumed by the mesh rasterizer.
struct ViewFaces
{
    DeviceBuffer<float3> ndc;          // [F][3] vertices as (ndc_x, ndc_y, camera z)
    DeviceBuffer<float3> world;        // [F][3] vertices in world space (for shading)
    DeviceBuffer<float3> normals;      // [F] unit face normals in world space
    std::uint32_t n_faces = 0;
};

void transform_faces(const DeviceMesh &mesh, const Camera &camera, ViewFaces &out, cudaStream_t stream);

} // namespace tsr
