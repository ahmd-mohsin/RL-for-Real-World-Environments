// Tessera - tile-based compositing of splats over the surface layer.
//
// Derived from gsplat's rasterize_to_pixels_3dgs_fwd_kernel and the blending
// helpers in RasterizeToPixels3DGSDevice.cuh (Apache-2.0). Changes for
// Tessera: three colour channels, a per-pixel surface layer that both bounds
// the visible depth range and replaces the background, and a per-splat depth
// carried through the shared-memory batch.
#include "tsr/geom/gaussian_math.cuh"
#include "tsr/math/vec.h"
#include "tsr/raster/splat_raster.h"

namespace tsr
{
namespace
{
struct GaussianWeight
{
    float vis;   // exp(-sigma)
    float alpha; // min(kMaxAlpha, opacity * vis)
    bool valid;  // sigma >= 0 and alpha >= kAlphaThreshold
};

// sigma is the Mahalanobis exponent for the pixel offset (dx, dy) = mean - pixel.
TSR_DI GaussianWeight eval_gaussian_weight(const float3 &conic, float dx, float dy, float opac)
{
    const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
    const float vis   = __expf(-sigma);
    const float alpha = fminf(geom::kMaxAlpha, opac * vis);
    return GaussianWeight{vis, alpha, !(sigma < 0.f || alpha < geom::kAlphaThreshold)};
}

template<std::uint32_t TILE_SIZE, std::uint32_t CTA_SIZE>
__global__ void __launch_bounds__(CTA_SIZE) rasterize_splats_kernel(
    const float2 *__restrict__ means2d,
    const float3 *__restrict__ conics,
    const float3 *__restrict__ colors,
    const float *__restrict__ opacities,
    const float *__restrict__ depths,
    const std::uint32_t image_width,
    const std::uint32_t image_height,
    const std::uint32_t tile_width,
    const std::int32_t *__restrict__ tile_offsets, // [n_tiles + 1]
    const std::int32_t *__restrict__ flatten_ids,  // [n_isects]
    const float3 *__restrict__ surface_color,      // [H][W]
    const float *__restrict__ surface_depth,       // [H][W]
    float3 *__restrict__ render_colors,            // [H][W]
    float *__restrict__ render_alphas,             // [H][W]
    std::int32_t *__restrict__ last_ids            // [H][W]
)
{
    constexpr std::uint32_t BATCH_SIZE        = CTA_SIZE;
    constexpr std::uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
    constexpr std::uint32_t ROW_STRIDE        = CTA_SIZE / TILE_SIZE;
    constexpr std::uint32_t TILE_MASK         = TILE_SIZE - 1;
    constexpr std::uint32_t TILE_SHIFT        = TILE_SIZE == 16 ? 4 : (TILE_SIZE == 8 ? 3 : 2);
    constexpr std::uint32_t ALL_DONE          = (1u << PIXELS_PER_THREAD) - 1u;
    static_assert((TILE_SIZE & (TILE_SIZE - 1)) == 0 && TILE_SIZE >= 4 && TILE_SIZE <= 16, "TILE_SIZE must be 4, 8 or 16");
    static_assert(PIXELS_PER_THREAD > 0, "CTA_SIZE must not exceed TILE_SIZE * TILE_SIZE");

    const std::uint32_t tile_id = blockIdx.x;
    const std::uint32_t tile_x  = tile_id % tile_width;
    const std::uint32_t tile_y  = tile_id / tile_width;

    const std::uint32_t tid      = threadIdx.x;
    const std::uint32_t thread_x = tid & TILE_MASK;
    const std::uint32_t thread_y = tid >> TILE_SHIFT;

    // x is shared by the thread's pixels; y steps by ROW_STRIDE rows. Pixel
    // centres sit at +0.5.
    const std::uint32_t out_x = tile_x * TILE_SIZE + thread_x;
    const float px            = static_cast<float>(out_x) + 0.5f;

    std::uint32_t out_y[PIXELS_PER_THREAD];
    float py[PIXELS_PER_THREAD];
    std::int32_t pix_id[PIXELS_PER_THREAD];
    float z_limit[PIXELS_PER_THREAD];
    TSR_UNROLL
    for(std::uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        out_y[p]  = tile_y * TILE_SIZE + thread_y + p * ROW_STRIDE;
        py[p]     = static_cast<float>(out_y[p]) + 0.5f;
        pix_id[p] = static_cast<std::int32_t>(out_y[p] * image_width + out_x);
    }

    // Out-of-image pixels cannot return early: __syncthreads_count below
    // needs every thread of the block.
    std::uint32_t done_mask = (out_x >= image_width) ? ALL_DONE : 0u;
    TSR_UNROLL
    for(std::uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        if(out_y[p] >= image_height)
        {
            done_mask |= (1u << p);
        }
        z_limit[p] = (done_mask & (1u << p)) ? 0.f : surface_depth[pix_id[p]];
    }

    const std::int32_t range_start = tile_offsets[tile_id];
    const std::int32_t range_end   = tile_offsets[tile_id + 1];
    const std::int32_t num_batches = (range_end - range_start + static_cast<std::int32_t>(BATCH_SIZE) - 1) / static_cast<std::int32_t>(BATCH_SIZE);

    TSR_DYNAMIC_SMEM(int, s);
    std::int32_t *id_batch  = reinterpret_cast<std::int32_t *>(s);                   // [BATCH_SIZE]
    float3 *xy_opacity_batch = reinterpret_cast<float3 *>(&id_batch[BATCH_SIZE]);     // [BATCH_SIZE]
    float3 *conic_batch      = reinterpret_cast<float3 *>(&xy_opacity_batch[BATCH_SIZE]); // [BATCH_SIZE]
    float *depth_batch       = reinterpret_cast<float *>(&conic_batch[BATCH_SIZE]);   // [BATCH_SIZE]

    float T[PIXELS_PER_THREAD];
    std::int32_t last_offset[PIXELS_PER_THREAD];
    float3 pix_out[PIXELS_PER_THREAD];
    TSR_UNROLL
    for(std::uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        T[p]           = 1.0f;
        last_offset[p] = 0;
        pix_out[p]     = make_float3(0.f, 0.f, 0.f);
    }

    TSR_NO_UNROLL
    for(std::int32_t b = 0; b < num_batches; ++b)
    {
        // each thread fetches one splat, front to back
        const std::int32_t batch_offset = static_cast<std::int32_t>(BATCH_SIZE) * b;
        const std::int32_t batch_start  = range_start + batch_offset;
        const std::int32_t idx          = batch_start + static_cast<std::int32_t>(tid);
        if(idx < range_end)
        {
            const std::int32_t g  = flatten_ids[idx];
            id_batch[tid]         = g;
            const float2 xy       = means2d[g];
            xy_opacity_batch[tid] = make_float3(xy.x, xy.y, opacities[g]);
            conic_batch[tid]      = conics[g];
            depth_batch[tid]      = depths[g];
        }

        // wait for the batch to be loaded
        if constexpr(CTA_SIZE <= 32)
        {
            __syncwarp();
        }
        else
        {
            __syncthreads();
        }

        const std::int32_t remaining   = range_end - batch_start;
        const std::uint32_t batch_size = static_cast<std::uint32_t>(remaining < static_cast<std::int32_t>(BATCH_SIZE) ? remaining : static_cast<std::int32_t>(BATCH_SIZE));
        for(std::uint32_t t = 0; (t < batch_size) && (done_mask != ALL_DONE); ++t)
        {
            const float3 conic   = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float depth    = depth_batch[t];
            const float dx       = xy_opac.x - px;

            TSR_UNROLL
            for(std::uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
            {
                if(done_mask & (1u << p))
                {
                    continue;
                }
                if(depth >= z_limit[p])
                {
                    // At or behind the surface; so is every later splat.
                    done_mask |= (1u << p);
                    continue;
                }
                const float dy          = xy_opac.y - py[p];
                const GaussianWeight gw = eval_gaussian_weight(conic, dx, dy, xy_opac.z);
                if(!gw.valid)
                {
                    continue;
                }
                const float next_T = T[p] * (1.0f - gw.alpha);
                if(next_T <= geom::kTransmittanceThreshold)
                {
                    // saturated: exclusive of this splat
                    done_mask |= (1u << p);
                    continue;
                }
                const float vis = gw.alpha * T[p];
                pix_out[p]     += colors[id_batch[t]] * vis;
                last_offset[p]  = batch_offset + static_cast<std::int32_t>(t);
                T[p]            = next_T;
            }
        }

        // resync before the next batch; stop once every pixel is done
        if(__syncthreads_count(done_mask == ALL_DONE) >= static_cast<int>(BATCH_SIZE))
        {
            break;
        }
    }

    if(out_x < image_width)
    {
        TSR_UNROLL
        for(std::uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            if(out_y[p] < image_height)
            {
                render_alphas[pix_id[p]] = 1.0f - T[p];
                render_colors[pix_id[p]] = pix_out[p] + T[p] * surface_color[pix_id[p]];
                last_ids[pix_id[p]]      = last_offset[p];
            }
        }
    }
}

template<std::uint32_t TILE_SIZE, std::uint32_t CTA_SIZE>
void launch_variant(
    const ProjectedSplats &splats,
    const TileBinning &binning,
    const SurfaceLayer &surface,
    const Camera &camera,
    SplatRasterOutput &out,
    cudaStream_t stream
)
{
    const std::size_t smem = CTA_SIZE * (sizeof(std::int32_t) + 2 * sizeof(float3) + sizeof(float));
    TSR_CUDA_CHECK(cudaFuncSetAttribute(
        rasterize_splats_kernel<TILE_SIZE, CTA_SIZE>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)
    ));
    launch(
        rasterize_splats_kernel<TILE_SIZE, CTA_SIZE>, LaunchConfig{dim3(binning.n_tiles()), dim3(CTA_SIZE), smem, stream},
        splats.means2d.data(), splats.conics.data(), splats.colors.data(), splats.opacities.data(), splats.depths.data(),
        camera.width, camera.height, binning.tiles_x, binning.tile_offsets.data(), binning.sorted_splat_ids.data(),
        surface.color.data(), surface.depth.data(), out.color.data(), out.alpha.data(), out.last_contributor.data()
    );
}
} // namespace

void rasterize_splats(
    const ProjectedSplats &splats,
    const TileBinning &binning,
    const SurfaceLayer &surface,
    const Camera &camera,
    const SplatRasterSettings &settings,
    SplatRasterOutput &out,
    cudaStream_t stream
)
{
    TSR_CHECK(surface.width == static_cast<int>(camera.width) && surface.height == static_cast<int>(camera.height),
              "surface layer does not match the camera resolution");
    const std::size_t n = static_cast<std::size_t>(camera.width) * camera.height;
    out.width           = static_cast<int>(camera.width);
    out.height          = static_cast<int>(camera.height);
    out.color.resize(n);
    out.alpha.resize(n);
    out.last_contributor.resize(n);
    if(binning.n_tiles() == 0)
    {
        return;
    }

    switch(binning.tile_size)
    {
    case 16:
        if(settings.pixels_per_thread == 4)
        {
            launch_variant<16, 64>(splats, binning, surface, camera, out, stream);
        }
        else
        {
            launch_variant<16, 256>(splats, binning, surface, camera, out, stream);
        }
        break;
    case 8: launch_variant<8, 64>(splats, binning, surface, camera, out, stream); break;
    case 4: launch_variant<4, 16>(splats, binning, surface, camera, out, stream); break;
    default: TSR_CHECK(false, "unsupported tile size " << binning.tile_size << "; supported values are {4, 8, 16}");
    }
}
} // namespace tsr
