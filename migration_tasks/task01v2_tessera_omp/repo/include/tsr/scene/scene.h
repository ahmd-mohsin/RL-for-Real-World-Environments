// Tessera - host-side scene description.
//
// A scene mixes two primitive families that are rendered into one image:
//   * 3D Gaussians with spherical-harmonic radiance (splatted), and
//   * triangle meshes shaded by a neural material (hash-grid encoding + MLP).
// Gaussians are composited front to back in front of the nearest mesh surface.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tsr/device/backend.h"
#include "tsr/geom/camera.h"

namespace tsr
{
struct GaussianCloud
{
    std::vector<float3> means;
    std::vector<float4> quats;  // (w, x, y, z), normalized on use
    std::vector<float3> scales; // standard deviations along the local axes
    std::vector<float> opacities;
    std::vector<float> sh;      // [N][(degree+1)^2][3], DC term first
    std::uint32_t sh_degree = 0;

    [[nodiscard]] std::size_t size() const { return means.size(); }
    [[nodiscard]] std::uint32_t coeffs_per_gaussian() const { return (sh_degree + 1) * (sh_degree + 1); }

    // Throws tsr::Error if array sizes disagree or values are out of range.
    void validate() const;
};

struct TriangleMesh
{
    std::string name;
    std::vector<float3> vertices;
    std::vector<int3> faces; // counter-clockwise when seen from outside
};

struct HashGridConfig
{
    std::uint32_t n_levels          = 8;
    std::uint32_t features_per_level = 2;
    std::uint32_t log2_hashmap_size = 12;
    std::uint32_t base_resolution   = 4;
    float per_level_scale           = 1.5f;
};

struct MlpConfig
{
    std::uint32_t hidden_width  = 16;
    std::uint32_t hidden_layers = 2;
};

// A procedurally initialized neural texture: parameters are drawn from a
// seeded generator so that a scene file stays small and renders reproducibly.
struct NeuralMaterial
{
    HashGridConfig grid;
    MlpConfig mlp;
    std::uint32_t seed = 1;
    float3 bounds_min  = {-1.f, -1.f, -1.f}; // region mapped onto the unit cube
    float3 bounds_max  = {1.f, 1.f, 1.f};
    float3 tint        = {1.f, 1.f, 1.f};
};

struct Scene
{
    GaussianCloud gaussians;
    std::vector<TriangleMesh> meshes;
    NeuralMaterial material;
    float3 background = {0.f, 0.f, 0.f};
    std::vector<Camera> cameras;

    [[nodiscard]] std::size_t face_count() const;
    [[nodiscard]] std::size_t vertex_count() const;
};

// Parses the line-based .tsr scene format (see docs/scene_format.md).
Scene load_scene(const std::string &path);
Scene parse_scene(const std::string &text, const std::string &source_name = "<string>");

// Deterministic synthetic scenes used by tests and benchmarks.
struct SyntheticSceneParams
{
    std::uint32_t seed          = 1;
    std::uint32_t n_gaussians   = 256;
    std::uint32_t sh_degree     = 1;
    bool with_mesh              = true;
    std::uint32_t width         = 64;
    std::uint32_t height        = 48;
};
Scene make_synthetic_scene(const SyntheticSceneParams &params);

} // namespace tsr
