// Tests for mesh transform, coarse binning and fine rasterization.
#include <cmath>
#include <vector>

#include "doctest.h"
#include "reference.h"
#include "tsr/bin/mesh_bins.h"
#include "tsr/geom/mesh.h"
#include "tsr/raster/mesh_raster.h"
#include "tsr/scene/scene.h"

using namespace tsr;

namespace
{
struct Rasterized
{
    std::vector<std::int32_t> pix_to_face;
    std::vector<float> zbuf;
    std::vector<float3> ndc;
    bool overflow = false;
};

Rasterized rasterize(const std::vector<TriangleMesh> &meshes, const Camera &cam, int bin_size, int max_faces = 256)
{
    DeviceMesh mesh;
    mesh.upload(meshes, nullptr);
    ViewFaces faces;
    transform_faces(mesh, cam, faces, nullptr);
    Arena arena;
    MeshBins bins;
    MeshBinningSettings mb;
    mb.bin_size          = bin_size;
    mb.max_faces_per_bin = max_faces;
    bin_mesh_faces(faces, static_cast<int>(cam.width), static_cast<int>(cam.height), mb, bins, arena, nullptr);
    MeshFragments frags;
    rasterize_mesh_fine(faces, bins, static_cast<int>(cam.width), static_cast<int>(cam.height), MeshRasterSettings{}, frags, nullptr);
    Rasterized r;
    r.pix_to_face = frags.pix_to_face.to_host();
    r.zbuf        = frags.zbuf.to_host();
    r.ndc         = faces.ndc.to_host();
    r.overflow    = bins.overflowed;
    return r;
}
} // namespace

TEST_CASE("mesh: binned rasterization matches a naive rasterizer")
{
    SyntheticSceneParams p;
    p.width  = 72; // not a multiple of the bin size
    p.height = 40;
    const Scene scene = make_synthetic_scene(p);
    for(const Camera &cam : scene.cameras)
    {
        for(int bin_size : {8, 16})
        {
            CAPTURE(bin_size);
            const Rasterized r = rasterize(scene.meshes, cam, bin_size);
            CHECK_FALSE(r.overflow);
            std::vector<int> ref_face;
            std::vector<float> ref_z;
            reference::rasterize_nearest_face(r.ndc, static_cast<int>(cam.width), static_cast<int>(cam.height), ref_face, ref_z);
            std::size_t mismatches = 0, covered = 0;
            for(std::size_t i = 0; i < ref_face.size(); ++i)
            {
                covered += ref_face[i] >= 0;
                if(ref_face[i] != r.pix_to_face[i])
                {
                    ++mismatches;
                }
                else if(ref_face[i] >= 0)
                {
                    CHECK(r.zbuf[i] == doctest::Approx(ref_z[i]).epsilon(1e-4));
                }
            }
            CHECK(covered > ref_face.size() / 4);
            // only pixels exactly on shared edges may resolve differently
            CHECK(mismatches <= ref_face.size() / 200);
        }
    }
}

TEST_CASE("mesh: a triangle covers the image region its projected vertices span")
{
    // Seen from z = -4 looking down +z, world -x appears on the image's right.
    TriangleMesh tri;
    tri.name     = "tri";
    tri.vertices = {make_float3(-1.f, -0.5f, 0.f), make_float3(0.2f, 0.8f, 0.f), make_float3(0.4f, -0.6f, 0.f)};
    tri.faces    = {make_int3(0, 1, 2)};
    const Camera cam = Camera::look_at(make_float3(0.f, 0.f, -4.f), make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), 50.f, 64, 64);
    const Rasterized r = rasterize({tri}, cam, 16);

    float2 v[3];
    for(int k = 0; k < 3; ++k)
    {
        v[k] = cam.project(cam.world_to_camera(tri.vertices[k]));
    }
    CHECK(v[0].x > v[2].x); // world -x is on the right
    const float2 centroid = make_float2((v[0].x + v[1].x + v[2].x) / 3.f, (v[0].y + v[1].y + v[2].y) / 3.f);
    auto covered = [&](float2 p) { return r.pix_to_face[static_cast<int>(p.y) * 64 + static_cast<int>(p.x)] == 0; };
    auto towards = [](float2 a, float2 b, float t) { return make_float2(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)); };
    for(int k = 0; k < 3; ++k)
    {
        CAPTURE(k);
        CHECK(covered(towards(v[k], centroid, 0.2f)));   // just inside the vertex
        CHECK_FALSE(covered(towards(v[k], centroid, -0.3f))); // beyond the vertex
    }
    CHECK(covered(centroid));
    // depth is the camera distance to the plane
    CHECK(r.zbuf[static_cast<int>(centroid.y) * 64 + static_cast<int>(centroid.x)] == doctest::Approx(4.f).epsilon(1e-3));
}

TEST_CASE("mesh: bin overflow is reported")
{
    // Many tiny overlapping triangles in one bin.
    TriangleMesh m;
    m.name = "many";
    for(int i = 0; i < 64; ++i)
    {
        const float z = 0.01f * i;
        const int base = static_cast<int>(m.vertices.size());
        m.vertices.push_back(make_float3(-0.2f, -0.2f, z));
        m.vertices.push_back(make_float3(0.2f, -0.2f, z));
        m.vertices.push_back(make_float3(0.f, 0.2f, z));
        m.faces.push_back(make_int3(base, base + 1, base + 2));
    }
    const Camera cam = Camera::look_at(make_float3(0.f, 0.f, -4.f), make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), 50.f, 32, 32);
    CHECK(rasterize({m}, cam, 16, 8).overflow);
    CHECK_FALSE(rasterize({m}, cam, 16, 256).overflow);
}
