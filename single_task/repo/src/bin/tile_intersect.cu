// Tessera - binning of screen-space splats into image tiles.
//
// Two-pass intersection after gsplat's intersect_tile_kernel (Apache-2.0):
// pass one counts the tiles each splat overlaps, a prefix sum turns counts
// into write offsets, pass two emits the keys. The accurate path uses the
// SnugBox bound and AccuTile per-column ellipse clipping from SpeedySplat
// (arXiv:2412.00578) as implemented in gsplat; the offsets kernel follows
// gsplat's intersect_offset_kernel.
#include "tsr/bin/tiles.h"
#include "tsr/geom/gaussian_math.cuh"
#include "tsr/prims/prims.h"

namespace tsr
{
namespace
{
TSR_DI float2 ellipse_intersection(float A, float B, float C, float disc, float t, float2 p, bool is_y, float coord)
{
    const float p_u       = is_y ? p.y : p.x;
    const float p_v       = is_y ? p.x : p.y;
    const float coeff     = is_y ? A : C;
    const float h         = coord - p_u;
    const float sqrt_term = sqrtf(disc * h * h + t * coeff);
    return make_float2((-B * h - sqrt_term) / coeff + p_v, (-B * h + sqrt_term) / coeff + p_v);
}

// Walks the columns (or rows when is_y) of the tile rectangle and, for each,
// the range of tiles the ellipse covers. Counts them and, if keys != nullptr,
// emits one intersection per tile.
TSR_DI std::uint32_t accutile_tiles(
    float A,
    float B,
    float C,
    float disc,
    float t,
    float2 p,
    float2 bbox_min,
    float2 bbox_max,
    float2 bbox_argmin,
    float2 bbox_argmax,
    int2 rect_min,
    int2 rect_max,
    std::uint32_t tile_size,
    std::uint32_t tiles_x,
    bool is_y,
    std::uint64_t depth_bits,
    std::int32_t splat,
    std::uint64_t *keys,
    std::int32_t *ids,
    std::int64_t cursor
)
{
    const float block = static_cast<float>(tile_size);
    if(is_y)
    {
        rect_min    = make_int2(rect_min.y, rect_min.x);
        rect_max    = make_int2(rect_max.y, rect_max.x);
        bbox_min    = make_float2(bbox_min.y, bbox_min.x);
        bbox_max    = make_float2(bbox_max.y, bbox_max.x);
        bbox_argmin = make_float2(bbox_argmin.y, bbox_argmin.x);
        bbox_argmax = make_float2(bbox_argmax.y, bbox_argmax.x);
    }

    std::uint32_t count = 0;
    float2 intersect_max_line = make_float2(bbox_max.y, bbox_min.y);
    float min_line            = rect_min.x * block;
    float2 intersect_min_line = bbox_min.x <= min_line ? ellipse_intersection(A, B, C, disc, t, p, is_y, min_line) : intersect_max_line;

    TSR_NO_UNROLL
    for(int u = rect_min.x; u < rect_max.x; ++u)
    {
        const float max_line = min_line + block;
        if(max_line <= bbox_max.x)
        {
            intersect_max_line = ellipse_intersection(A, B, C, disc, t, p, is_y, max_line);
        }
        const float ellipse_min = (min_line <= bbox_argmin.y && bbox_argmin.y < max_line)
                                    ? bbox_min.y
                                    : fminf(intersect_min_line.x, intersect_max_line.x);
        const float ellipse_max = (min_line <= bbox_argmax.y && bbox_argmax.y < max_line)
                                    ? bbox_max.y
                                    : fmaxf(intersect_min_line.y, intersect_max_line.y);

        const int min_tile_v = max(rect_min.y, min(rect_max.y, static_cast<int>(ellipse_min / block)));
        const int max_tile_v = min(rect_max.y, max(rect_min.y, static_cast<int>(ellipse_max / block + 1)));

        TSR_NO_UNROLL
        for(int v = min_tile_v; v < max_tile_v; ++v)
        {
            const std::uint64_t tile_id = is_y ? static_cast<std::uint64_t>(u) * tiles_x + v : static_cast<std::uint64_t>(v) * tiles_x + u;
            if(keys != nullptr)
            {
                keys[cursor + count] = (tile_id << 32) | depth_bits;
                ids[cursor + count]  = splat;
            }
            ++count;
        }
        intersect_min_line = intersect_max_line;
        min_line           = max_line;
    }
    return count;
}

// first pass (keys == nullptr): tiles_per_splat[i] = overlap count
// second pass: emit keys at [cum[i] - count, cum[i])
__global__ void intersect_tiles_kernel(
    std::uint32_t n,
    const int2 *__restrict__ radii,
    const float2 *__restrict__ means2d,
    const float *__restrict__ depths,
    const float3 *__restrict__ conics,
    const float *__restrict__ opacities,
    std::uint32_t tile_size,
    std::uint32_t tiles_x,
    std::uint32_t tiles_y,
    bool accurate,
    std::int64_t *__restrict__ tiles_per_splat, // counts (pass 1) or inclusive cumsum (pass 2)
    std::uint64_t *__restrict__ keys,
    std::int32_t *__restrict__ ids
)
{
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n)
    {
        return;
    }
    const bool first_pass = keys == nullptr;
    const int2 radius     = radii[idx];
    if(radius.x <= 0 || radius.y <= 0)
    {
        if(first_pass)
        {
            tiles_per_splat[idx] = 0;
        }
        return;
    }

    const float2 mean2d          = means2d[idx];
    const std::int64_t cursor    = first_pass ? 0 : (idx == 0 ? 0 : tiles_per_splat[idx - 1]);
    const std::uint64_t depth_id = first_pass ? 0 : static_cast<std::uint64_t>(__float_as_uint(depths[idx]));
    const float tile_size_f      = static_cast<float>(tile_size);

    if(accurate)
    {
        // conic = (a, b, c): upper triangle of Sigma^{-1}; quadratic form
        // q = a dx^2 + 2 b dx dy + c dy^2 and disc = b^2 - a c = -det(Sigma^{-1}).
        const float3 conic = conics[idx];
        const float A = conic.x, B = conic.y, C = conic.z;
        const float disc = B * B - A * C;
        // opacity-aware level set alpha >= threshold, capped like the radius
        const float t = fminf(geom::kGaussianExtent * geom::kGaussianExtent, 2.0f * __logf(opacities[idx] / geom::kAlphaThreshold));

        const float neg_t_over_disc = -t / disc;
        const float x_extent        = sqrtf(neg_t_over_disc * C);
        const float y_extent        = sqrtf(neg_t_over_disc * A);
        const float2 bbox_min       = make_float2(mean2d.x - x_extent, mean2d.y - y_extent);
        const float2 bbox_max       = make_float2(mean2d.x + x_extent, mean2d.y + y_extent);
        const float bx_over_c       = B * x_extent / C;
        const float by_over_a       = B * y_extent / A;
        const float2 bbox_argmin    = make_float2(mean2d.y + bx_over_c, mean2d.x + by_over_a);
        const float2 bbox_argmax    = make_float2(mean2d.y - bx_over_c, mean2d.x - by_over_a);

        const int2 rect_min = make_int2(
            max(0, min(static_cast<int>(tiles_x), static_cast<int>(bbox_min.x / tile_size_f))),
            max(0, min(static_cast<int>(tiles_y), static_cast<int>(bbox_min.y / tile_size_f)))
        );
        const int2 rect_max = make_int2(
            max(0, min(static_cast<int>(tiles_x), static_cast<int>(bbox_max.x / tile_size_f + 1.f))),
            max(0, min(static_cast<int>(tiles_y), static_cast<int>(bbox_max.y / tile_size_f + 1.f)))
        );
        const int y_span = rect_max.y - rect_min.y;
        const int x_span = rect_max.x - rect_min.x;
        if(y_span * x_span == 0)
        {
            if(first_pass)
            {
                tiles_per_splat[idx] = 0;
            }
            return;
        }
        const std::uint32_t count = accutile_tiles(
            A, B, C, disc, t, mean2d, bbox_min, bbox_max, bbox_argmin, bbox_argmax, rect_min, rect_max,
            tile_size, tiles_x, y_span < x_span, depth_id, static_cast<std::int32_t>(idx), keys, ids, cursor
        );
        if(first_pass)
        {
            tiles_per_splat[idx] = count;
        }
        return;
    }

    // Axis-aligned fallback. tile_min is inclusive, tile_max exclusive.
    const float tile_radius_x = radius.x / tile_size_f;
    const float tile_radius_y = radius.y / tile_size_f;
    const float tile_x        = mean2d.x / tile_size_f;
    const float tile_y        = mean2d.y / tile_size_f;
    const int2 tile_min       = make_int2(
        min(max(0, static_cast<int>(floorf(tile_x - tile_radius_x))), static_cast<int>(tiles_x)),
        min(max(0, static_cast<int>(floorf(tile_y - tile_radius_y))), static_cast<int>(tiles_y))
    );
    const int2 tile_max = make_int2(
        min(max(0, static_cast<int>(ceilf(tile_x + tile_radius_x))), static_cast<int>(tiles_x)),
        min(max(0, static_cast<int>(ceilf(tile_y + tile_radius_y))), static_cast<int>(tiles_y))
    );
    if(first_pass)
    {
        tiles_per_splat[idx] = static_cast<std::int64_t>(tile_max.y - tile_min.y) * (tile_max.x - tile_min.x);
        return;
    }
    std::int64_t cur = cursor;
    for(int i = tile_min.y; i < tile_max.y; ++i)
    {
        for(int j = tile_min.x; j < tile_max.x; ++j)
        {
            const std::uint64_t tile_id = static_cast<std::uint64_t>(i) * tiles_x + j;
            keys[cur]                   = (tile_id << 32) | depth_id;
            ids[cur]                    = static_cast<std::int32_t>(idx);
            ++cur;
        }
    }
}

// e.g. tile ids of sorted keys: [1, 1, 1, 3, 3], n_tiles = 6
//      offsets:                 [0, 0, 3, 3, 5, 5, 5]
__global__ void tile_offsets_kernel(
    std::uint32_t n_isects,
    const std::uint64_t *__restrict__ sorted_keys,
    std::uint32_t n_tiles,
    std::int32_t *__restrict__ offsets // [n_tiles + 1]
)
{
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_isects)
    {
        return;
    }
    const std::int64_t tile_curr = static_cast<std::int64_t>(sorted_keys[idx] >> 32);
    if(idx == 0)
    {
        // offsets up to and including the first occupied tile
        for(std::int64_t i = 0; i < tile_curr + 1; ++i)
        {
            offsets[i] = 0;
        }
    }
    if(idx == n_isects - 1)
    {
        // every later tile, plus the terminating entry
        for(std::int64_t i = tile_curr + 1; i <= static_cast<std::int64_t>(n_tiles); ++i)
        {
            offsets[i] = static_cast<std::int32_t>(n_isects);
        }
    }
    if(idx > 0)
    {
        const std::int64_t tile_prev = static_cast<std::int64_t>(sorted_keys[idx - 1] >> 32);
        if(tile_prev == tile_curr)
        {
            return;
        }
        for(std::int64_t i = tile_prev + 1; i < tile_curr + 1; ++i)
        {
            offsets[i] = static_cast<std::int32_t>(idx);
        }
    }
}
} // namespace

std::uint32_t tile_id_bits(std::uint32_t n_tiles)
{
    std::uint32_t bits = 0;
    while(bits < 32 && (std::uint64_t{1} << bits) < n_tiles)
    {
        ++bits;
    }
    return bits;
}

void bin_splats(
    const ProjectedSplats &splats,
    const Camera &camera,
    const BinningSettings &settings,
    TileBinning &out,
    Arena &arena,
    cudaStream_t stream
)
{
    TSR_CHECK(settings.tile_size > 0 && (settings.tile_size & (settings.tile_size - 1)) == 0,
              "tile size must be a power of two, got " << settings.tile_size);
    out.tile_size = settings.tile_size;
    out.tiles_x   = camera.tiles_x(settings.tile_size);
    out.tiles_y   = camera.tiles_y(settings.tile_size);
    out.tile_bits = tile_id_bits(out.n_tiles());
    TSR_CHECK(out.tile_bits <= 32, "too many tiles: " << out.n_tiles());

    const std::uint32_t n = splats.count;
    out.tiles_per_splat.resize(n);
    out.tile_offsets.resize(static_cast<std::size_t>(out.n_tiles()) + 1);

    // pass 1: count
    launch_linear(
        intersect_tiles_kernel, n, stream, n, splats.radii.data(), splats.means2d.data(), splats.depths.data(),
        splats.conics.data(), splats.opacities.data(), out.tile_size, out.tiles_x, out.tiles_y, settings.accurate,
        out.tiles_per_splat.data(), static_cast<std::uint64_t *>(nullptr), static_cast<std::int32_t *>(nullptr)
    );

    std::int64_t total = 0;
    if(n > 0)
    {
        prims::inclusive_scan(out.tiles_per_splat.data(), out.tiles_per_splat.data(), n, arena, stream);
        TSR_CUDA_CHECK(cudaMemcpyAsync(&total, out.tiles_per_splat.data() + n - 1, sizeof(total), cudaMemcpyDeviceToHost, stream));
        TSR_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    TSR_CHECK(total < 0x7fffffff, "intersection count overflow: " << total);
    out.n_isects = static_cast<std::uint32_t>(total);

    if(out.n_isects == 0)
    {
        out.tile_offsets.zero(stream);
        out.keys.resize(0);
        out.splat_ids.resize(0);
        out.sorted_keys.resize(0);
        out.sorted_splat_ids.resize(0);
        return;
    }

    // pass 2: emit
    out.keys.resize(out.n_isects);
    out.splat_ids.resize(out.n_isects);
    launch_linear(
        intersect_tiles_kernel, n, stream, n, splats.radii.data(), splats.means2d.data(), splats.depths.data(),
        splats.conics.data(), splats.opacities.data(), out.tile_size, out.tiles_x, out.tiles_y, settings.accurate,
        out.tiles_per_splat.data(), out.keys.data(), out.splat_ids.data()
    );

    // group by tile, front to back within a tile
    out.sorted_keys.resize(out.n_isects);
    out.sorted_splat_ids.resize(out.n_isects);
    prims::sort_pairs(
        out.keys.data(), out.sorted_keys.data(), out.splat_ids.data(), out.sorted_splat_ids.data(), out.n_isects, 0,
        32 + static_cast<int>(out.tile_bits), arena, stream
    );

    launch_linear(tile_offsets_kernel, out.n_isects, stream, out.n_isects, out.sorted_keys.data(), out.n_tiles(), out.tile_offsets.data());
}
} // namespace tsr
