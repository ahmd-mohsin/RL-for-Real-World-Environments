// Tessera - multiresolution hash-grid encoding of 3D positions.
//
// L levels with geometrically increasing resolution; each level stores F
// features per grid vertex in a table that is dense while it fits and hashed
// beyond that. A position in [0, 1]^3 is encoded by trilinear interpolation
// at every level, giving L * F features.
#pragma once

#include <cstdint>
#include <vector>

#include "tsr/device/memory.h"
#include "tsr/scene/scene.h"

namespace tsr
{
enum class GridInterpolation : std::uint32_t
{
    Nearest = 0,
    Linear  = 1,
};

class HashGrid
{
public:
    // Builds the per-level offset table and draws parameters uniformly from
    // [-init_scale, init_scale] with a PCG32 seeded by `seed`.
    void initialize(const HashGridConfig &config, std::uint32_t seed, float init_scale, cudaStream_t stream);

    [[nodiscard]] const HashGridConfig &config() const { return config_; }
    [[nodiscard]] std::uint32_t n_output_dims() const { return config_.n_levels * config_.features_per_level; }
    [[nodiscard]] const std::vector<std::uint32_t> &offsets() const { return offsets_; } // [L + 1], in grid vertices
    [[nodiscard]] std::uint32_t n_params() const { return offsets_.empty() ? 0 : offsets_.back() * config_.features_per_level; }
    [[nodiscard]] const float *params() const { return params_.data(); }
    [[nodiscard]] const std::uint32_t *device_offsets() const { return d_offsets_.data(); }
    [[nodiscard]] std::vector<float> params_to_host() const { return params_.to_host(); }

private:
    HashGridConfig config_{};
    std::vector<std::uint32_t> offsets_;
    DeviceBuffer<std::uint32_t> d_offsets_;
    DeviceBuffer<float> params_;
};

// Level scale and vertex resolution. `base_resolution` counts grid vertices,
// hence the -1 (power-of-two parameter grids for power-of-two bases).
float grid_scale(std::uint32_t level, float log2_per_level_scale, std::uint32_t base_resolution);
std::uint32_t grid_resolution(float scale);

// out[i * n_output_dims + level * F + f]; positions must lie in [0, 1]^3.
void encode_positions(
    const HashGrid &grid,
    const float3 *positions,
    std::uint32_t n,
    float *out,
    GridInterpolation interpolation,
    cudaStream_t stream
);

} // namespace tsr
