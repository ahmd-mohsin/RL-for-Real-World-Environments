// Hidden benchmark and stress scenes, built through the public API only so the
// same code runs against the original CUDA build and against a port.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "tsr/scene/scene.h"

namespace hidden
{
using namespace tsr;

// Heightfield terrain of (n x n) quads = 2 n^2 triangles over [-s, s]^2 at y ~ -0.6.
inline TriangleMesh make_terrain(int n, float s, std::uint32_t seed)
{
    TriangleMesh m;
    m.name = "terrain";
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> phase(0.f, 6.2831853f);
    const float p1 = phase(rng), p2 = phase(rng), p3 = phase(rng);
    m.vertices.reserve(static_cast<std::size_t>(n + 1) * (n + 1));
    for(int j = 0; j <= n; ++j)
    {
        for(int i = 0; i <= n; ++i)
        {
            const float x = -s + 2.f * s * static_cast<float>(i) / static_cast<float>(n);
            const float z = -s + 2.f * s * static_cast<float>(j) / static_cast<float>(n);
            const float y = -0.6f + 0.08f * std::sin(3.1f * x + p1) * std::cos(2.3f * z + p2) + 0.03f * std::sin(9.7f * (x + z) + p3);
            m.vertices.push_back(make_float3(x, y, z));
        }
    }
    m.faces.reserve(static_cast<std::size_t>(n) * n * 2);
    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < n; ++i)
        {
            const int a = j * (n + 1) + i, b = a + 1, c = a + (n + 1), d = c + 1;
            m.faces.push_back(make_int3(a, c, b));
            m.faces.push_back(make_int3(b, c, d));
        }
    }
    return m;
}

// Closed UV sphere with `rings` x `segments` quads.
inline TriangleMesh make_sphere(float3 c, float r, int rings, int segments)
{
    TriangleMesh m;
    m.name = "sphere";
    for(int i = 0; i <= rings; ++i)
    {
        const float th = 3.14159265f * static_cast<float>(i) / static_cast<float>(rings);
        for(int j = 0; j < segments; ++j)
        {
            const float ph = 6.2831853f * static_cast<float>(j) / static_cast<float>(segments);
            m.vertices.push_back(make_float3(c.x + r * std::sin(th) * std::cos(ph), c.y + r * std::cos(th), c.z + r * std::sin(th) * std::sin(ph)));
        }
    }
    for(int i = 0; i < rings; ++i)
    {
        for(int j = 0; j < segments; ++j)
        {
            const int a = i * segments + j, b = i * segments + (j + 1) % segments;
            const int c2 = a + segments, d = b + segments;
            m.faces.push_back(make_int3(a, b, c2));
            m.faces.push_back(make_int3(b, d, c2));
        }
    }
    return m;
}

inline Scene with_cameras(Scene s, std::uint32_t w, std::uint32_t h)
{
    s.cameras.clear();
    s.cameras.push_back(Camera::look_at(make_float3(0.4f, 0.9f, -3.6f), make_float3(0.f, -0.2f, 0.f), make_float3(0.f, 1.f, 0.f), 50.f, w, h));
    s.cameras.push_back(Camera::look_at(make_float3(-3.0f, 0.5f, 1.5f), make_float3(0.f, -0.3f, 0.f), make_float3(0.f, 1.f, 0.f), 45.f, w, h));
    s.cameras.push_back(Camera::look_at(make_float3(2.8f, 1.6f, 2.2f), make_float3(0.f, -0.3f, 0.f), make_float3(0.f, 1.f, 0.f), 55.f, w, h));
    s.cameras.push_back(Camera::look_at(make_float3(0.f, 3.5f, -0.5f), make_float3(0.f, -0.5f, 0.f), make_float3(0.f, 0.f, 1.f), 60.f, w, h));
    return s;
}

// Splat-dominated: many gaussians, no mesh.
inline Scene splat_scene(std::uint32_t n, std::uint32_t w, std::uint32_t h, std::uint32_t seed = 7)
{
    SyntheticSceneParams p;
    p.seed        = seed;
    p.n_gaussians = n;
    p.sh_degree   = 3;
    p.with_mesh   = false;
    p.width       = w;
    p.height      = h;
    return with_cameras(make_synthetic_scene(p), w, h);
}

// Mesh-dominated: dense terrain + sphere, neural material, a few splats.
inline Scene mesh_scene(int terrain_n, std::uint32_t w, std::uint32_t h, std::uint32_t seed = 11)
{
    SyntheticSceneParams p;
    p.seed        = seed;
    p.n_gaussians = 2000;
    p.sh_degree   = 1;
    p.with_mesh   = true;
    p.width       = w;
    p.height      = h;
    Scene s       = make_synthetic_scene(p);
    s.meshes.push_back(make_terrain(terrain_n, 2.5f, seed));
    s.meshes.push_back(make_sphere(make_float3(0.6f, -0.1f, 0.4f), 0.45f, 96, 192));
    s.material.grid.n_levels          = 16;
    s.material.grid.log2_hashmap_size = 19;
    s.material.grid.base_resolution   = 16;
    s.material.grid.per_level_scale   = 1.5f;
    s.material.mlp.hidden_width       = 64;
    s.material.mlp.hidden_layers      = 2;
    s.material.bounds_min             = make_float3(-2.6f, -1.f, -2.6f);
    s.material.bounds_max             = make_float3(2.6f, 1.f, 2.6f);
    return with_cameras(std::move(s), w, h);
}

// Hybrid: both workloads at once, splats partly occluded by the surface.
inline Scene hybrid_scene(std::uint32_t n_gauss, int terrain_n, std::uint32_t w, std::uint32_t h, std::uint32_t seed = 23)
{
    Scene s           = mesh_scene(terrain_n, w, h, seed);
    Scene g           = splat_scene(n_gauss, w, h, seed + 1);
    s.gaussians       = std::move(g.gaussians);
    return s;
}
} // namespace hidden
