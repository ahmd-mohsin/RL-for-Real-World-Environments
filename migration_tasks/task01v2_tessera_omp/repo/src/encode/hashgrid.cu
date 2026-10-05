// Tessera - multiresolution hash-grid encoding of 3D positions.
//
// Derived from tiny-cuda-nn's GridEncoding forward pass (kernel_grid,
// grid_index, coherent_prime_hash, pos_fract and the offset table in
// encodings/grid.h and common_device.h; BSD-3-Clause, Copyright NVIDIA
// CORPORATION), see third_party/LICENSES/tiny-cuda-nn.txt. Specialised to
// three input dimensions, float parameters and row-major output.
#include <cmath>
#include <limits>

#include "tsr/encode/hashgrid.h"
#include "tsr/util/random.h"

namespace tsr
{
namespace
{
constexpr std::uint32_t kPosDims  = 3;
constexpr std::uint32_t kMaxLevels = 32;

TSR_HDI std::uint32_t coherent_prime_hash(const std::uint32_t (&pos_grid)[kPosDims])
{
    // The first prime is 1 so neighbouring cells along x stay coherent.
    constexpr std::uint32_t factors[kPosDims] = {1u, 2654435761u, 805459861u};
    std::uint32_t result                      = 0;
    TSR_UNROLL
    for(std::uint32_t i = 0; i < kPosDims; ++i)
    {
        result ^= pos_grid[i] * factors[i];
    }
    return result;
}

TSR_HDI std::uint32_t grid_index(std::uint32_t hashmap_size, std::uint32_t resolution, const std::uint32_t (&pos_grid)[kPosDims])
{
    // Largest resolution whose dense index cannot overflow 32 bits in 3D.
    constexpr std::uint32_t kMaxBase = 0x659;
    std::uint32_t stride             = 1;
    std::uint32_t index              = 0;
    if(resolution <= kMaxBase)
    {
        TSR_UNROLL
        for(std::uint32_t dim = 0; dim < kPosDims; ++dim)
        {
            index  += pos_grid[dim] * stride;
            stride *= resolution;
        }
    }
    else
    {
        stride = 0xffffffffu;
    }
    if(hashmap_size < stride)
    {
        index = coherent_prime_hash(pos_grid);
    }
    return index % hashmap_size;
}

// The 0.5 offset staggers the levels against each other so that fractional
// coordinates do not line up at integer scales (Instant-NGP, Appendix A).
TSR_DI void pos_fract(float input, float &pos, std::uint32_t &pos_grid, float scale)
{
    pos             = fmaf(scale, input, 0.5f);
    const float tmp = floorf(pos);
    pos_grid        = static_cast<std::uint32_t>(static_cast<int>(tmp));
    pos            -= tmp;
}

template<std::uint32_t F>
__global__ void encode_kernel(
    std::uint32_t n,
    std::uint32_t n_levels,
    float log2_per_level_scale,
    std::uint32_t base_resolution,
    const std::uint32_t *__restrict__ offsets,
    const float *__restrict__ params,
    const float3 *__restrict__ positions,
    GridInterpolation interpolation,
    float *__restrict__ out
)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
    {
        return;
    }
    const std::uint32_t level = blockIdx.y; // the same for the whole block
    const std::uint32_t n_out = n_levels * F;

    const float *grid               = params + offsets[level] * F;
    const std::uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale               = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
    const std::uint32_t resolution  = static_cast<std::uint32_t>(ceilf(scale)) + 1;

    const float3 p                   = positions[i];
    const float input[kPosDims]      = {p.x, p.y, p.z};
    float pos[kPosDims];
    std::uint32_t pos_grid[kPosDims];
    TSR_UNROLL
    for(std::uint32_t dim = 0; dim < kPosDims; ++dim)
    {
        pos_fract(input[dim], pos[dim], pos_grid[dim], scale);
    }

    float result[F];
    TSR_UNROLL
    for(std::uint32_t f = 0; f < F; ++f)
    {
        result[f] = 0.f;
    }

    if(interpolation == GridInterpolation::Nearest)
    {
        const std::uint32_t index = grid_index(hashmap_size, resolution, pos_grid) * F;
        TSR_UNROLL
        for(std::uint32_t f = 0; f < F; ++f)
        {
            result[f] = grid[index + f];
        }
    }
    else
    {
        // trilinear interpolation over the 8 cell corners
        TSR_UNROLL
        for(std::uint32_t idx = 0; idx < (1u << kPosDims); ++idx)
        {
            float weight = 1.f;
            std::uint32_t corner[kPosDims];
            TSR_UNROLL
            for(std::uint32_t dim = 0; dim < kPosDims; ++dim)
            {
                if((idx & (1u << dim)) == 0)
                {
                    weight      *= 1.f - pos[dim];
                    corner[dim]  = pos_grid[dim];
                }
                else
                {
                    weight      *= pos[dim];
                    corner[dim]  = pos_grid[dim] + 1;
                }
            }
            const std::uint32_t index = grid_index(hashmap_size, resolution, corner) * F;
            TSR_UNROLL
            for(std::uint32_t f = 0; f < F; ++f)
            {
                result[f] += weight * grid[index + f];
            }
        }
    }

    TSR_UNROLL
    for(std::uint32_t f = 0; f < F; ++f)
    {
        out[static_cast<std::size_t>(i) * n_out + level * F + f] = result[f];
    }
}

std::uint32_t next_multiple(std::uint32_t v, std::uint32_t m) { return (v + m - 1) / m * m; }
} // namespace

float grid_scale(std::uint32_t level, float log2_per_level_scale, std::uint32_t base_resolution)
{
    return std::exp2(level * log2_per_level_scale) * base_resolution - 1.0f;
}

std::uint32_t grid_resolution(float scale) { return static_cast<std::uint32_t>(std::ceil(scale)) + 1; }

void HashGrid::initialize(const HashGridConfig &config, std::uint32_t seed, float init_scale, cudaStream_t stream)
{
    TSR_CHECK(config.n_levels >= 1 && config.n_levels <= kMaxLevels, "hash grid levels must be in [1, " << kMaxLevels << "]");
    TSR_CHECK(config.features_per_level == 1 || config.features_per_level == 2 || config.features_per_level == 4
                  || config.features_per_level == 8,
              "features per level must be 1, 2, 4 or 8");
    TSR_CHECK(config.log2_hashmap_size >= 4 && config.log2_hashmap_size <= 24, "log2 hashmap size must be in [4, 24]");
    TSR_CHECK(config.per_level_scale >= 1.0f, "per-level scale must be >= 1");
    config_ = config;

    const float log2_scale = std::log2(config.per_level_scale);
    offsets_.assign(config.n_levels + 1, 0);
    std::uint32_t offset = 0;
    for(std::uint32_t level = 0; level < config.n_levels; ++level)
    {
        const std::uint32_t resolution = grid_resolution(grid_scale(level, log2_scale, config.base_resolution));
        const std::uint32_t max_params = std::numeric_limits<std::uint32_t>::max() / 2;
        const float dense              = std::pow(static_cast<float>(resolution), static_cast<float>(kPosDims));
        std::uint32_t params_in_level  = dense > static_cast<float>(max_params) ? max_params : resolution * resolution * resolution;
        // keep memory accesses aligned
        params_in_level = next_multiple(params_in_level, 8u);
        params_in_level = std::min(params_in_level, 1u << config.log2_hashmap_size);
        offsets_[level] = offset;
        offset         += params_in_level;
    }
    offsets_[config.n_levels] = offset;

    Pcg32 rng(seed, 0x4f1bbcdcbfa53e0bULL);
    std::vector<float> host(static_cast<std::size_t>(offset) * config.features_per_level);
    for(float &v : host)
    {
        v = rng.uniform(-init_scale, init_scale);
    }
    params_.upload(host, stream);
    d_offsets_.upload(offsets_, stream);
}

void encode_positions(
    const HashGrid &grid,
    const float3 *positions,
    std::uint32_t n,
    float *out,
    GridInterpolation interpolation,
    cudaStream_t stream
)
{
    if(n == 0)
    {
        return;
    }
    const HashGridConfig &c  = grid.config();
    const float log2_scale   = std::log2(c.per_level_scale);
    const dim3 blocks(div_round_up(n, 128u), c.n_levels, 1);
    const LaunchConfig cfg{blocks, dim3(128), 0, stream};
    switch(c.features_per_level)
    {
    case 1: launch(encode_kernel<1>, cfg, n, c.n_levels, log2_scale, c.base_resolution, grid.device_offsets(), grid.params(), positions, interpolation, out); break;
    case 2: launch(encode_kernel<2>, cfg, n, c.n_levels, log2_scale, c.base_resolution, grid.device_offsets(), grid.params(), positions, interpolation, out); break;
    case 4: launch(encode_kernel<4>, cfg, n, c.n_levels, log2_scale, c.base_resolution, grid.device_offsets(), grid.params(), positions, interpolation, out); break;
    case 8: launch(encode_kernel<8>, cfg, n, c.n_levels, log2_scale, c.base_resolution, grid.device_offsets(), grid.params(), positions, interpolation, out); break;
    default: TSR_CHECK(false, "unsupported features per level " << c.features_per_level);
    }
}
} // namespace tsr
