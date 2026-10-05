// Tessera - triangle meshes on the device and their view transform.
#include "tsr/geom/mesh.h"
#include "tsr/geom/mesh_ndc.cuh"

namespace tsr
{
namespace
{
__global__ void transform_faces_kernel(
    std::uint32_t n_faces,
    const float3 *__restrict__ vertices,
    const int3 *__restrict__ faces,
    const Camera camera,
    float3 *__restrict__ face_ndc,
    float3 *__restrict__ face_world,
    float3 *__restrict__ face_normals
)
{
    const std::uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if(f >= n_faces)
    {
        return;
    }
    const int3 tri = faces[f];
    const int idx[3] = {tri.x, tri.y, tri.z};
    const int W      = static_cast<int>(camera.width);
    const int H      = static_cast<int>(camera.height);
    float3 world[3];
    TSR_UNROLL
    for(int k = 0; k < 3; ++k)
    {
        world[k]            = vertices[idx[k]];
        const float3 pc     = camera.world_to_camera(world[k]);
        const float2 pix    = camera.project(pc);
        face_ndc[3 * f + k] = make_float3(geom::pixel_x_to_ndc(pix.x, W, H), geom::pixel_y_to_ndc(pix.y, H, W), pc.z);
        face_world[3 * f + k] = world[k];
    }
    const float3 n   = cross(world[1] - world[0], world[2] - world[0]);
    const float len2 = dot(n, n);
    face_normals[f]  = len2 > 0.f ? n * rsqrtf(len2) : make_float3(0.f, 0.f, 0.f);
}
} // namespace

void DeviceMesh::upload(const std::vector<TriangleMesh> &meshes, cudaStream_t stream)
{
    std::vector<float3> all_vertices;
    std::vector<int3> all_faces;
    for(const TriangleMesh &m : meshes)
    {
        const int base = static_cast<int>(all_vertices.size());
        for(const int3 &f : m.faces)
        {
            TSR_CHECK(f.x >= 0 && f.y >= 0 && f.z >= 0 && static_cast<std::size_t>(f.x) < m.vertices.size()
                          && static_cast<std::size_t>(f.y) < m.vertices.size() && static_cast<std::size_t>(f.z) < m.vertices.size(),
                      "mesh '" << m.name << "' has a face referencing a missing vertex");
            all_faces.push_back(make_int3(f.x + base, f.y + base, f.z + base));
        }
        all_vertices.insert(all_vertices.end(), m.vertices.begin(), m.vertices.end());
    }
    n_vertices = static_cast<std::uint32_t>(all_vertices.size());
    n_faces    = static_cast<std::uint32_t>(all_faces.size());
    vertices.upload(all_vertices, stream);
    faces.upload(all_faces, stream);
}

void transform_faces(const DeviceMesh &mesh, const Camera &camera, ViewFaces &out, cudaStream_t stream)
{
    out.n_faces = mesh.n_faces;
    out.ndc.resize(static_cast<std::size_t>(mesh.n_faces) * 3);
    out.world.resize(static_cast<std::size_t>(mesh.n_faces) * 3);
    out.normals.resize(mesh.n_faces);
    launch_linear(
        transform_faces_kernel, mesh.n_faces, stream, mesh.n_faces, mesh.vertices.data(), mesh.faces.data(), camera,
        out.ndc.data(), out.world.data(), out.normals.data()
    );
}
} // namespace tsr
