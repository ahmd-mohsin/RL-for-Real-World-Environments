// Tessera - shading of rasterized mesh fragments with a neural material.
//
//   1. flag pixels whose nearest fragment hit a face and compact their indices
//   2. interpolate the world position with the fragment's barycentrics and map
//      it into the material's unit cube
//   3. hash-grid encode -> MLP -> RGB
//   4. scatter RGB * tint * view-dependent Lambert term into the surface layer
#include <limits>

#include "tsr/math/vec.h"
#include "tsr/prims/prims.h"
#include "tsr/render/surface.h"

namespace tsr
{
namespace
{
// Parameter scale for the procedural hash grid: large enough that the
// texture shows visible structure through a freshly initialized MLP.
constexpr float kGridInitScale = 1.0f;

__global__ void fill_surface_kernel(std::uint32_t n, float3 background, float3 *__restrict__ color, float *__restrict__ depth)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        color[i] = background;
        depth[i] = __int_as_float(0x7f800000); // +inf
    }
}

__global__ void flag_hits_kernel(std::uint32_t n, int K, const std::int32_t *__restrict__ pix_to_face, std::uint8_t *__restrict__ flags)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        flags[i] = pix_to_face[static_cast<std::size_t>(i) * K] >= 0 ? 1 : 0;
    }
}

__global__ void fragment_positions_kernel(
    std::uint32_t m,
    const std::uint32_t *__restrict__ pixels,
    int K,
    const std::int32_t *__restrict__ pix_to_face,
    const float3 *__restrict__ bary,
    const float3 *__restrict__ face_world,
    const float3 bounds_min,
    const float3 inv_extent,
    float3 *__restrict__ unit_positions
)
{
    const std::uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j >= m)
    {
        return;
    }
    const std::size_t frag = static_cast<std::size_t>(pixels[j]) * K;
    const int f            = pix_to_face[frag];
    const float3 b         = bary[frag];
    const float3 world     = b.x * face_world[3 * f + 0] + b.y * face_world[3 * f + 1] + b.z * face_world[3 * f + 2];
    unit_positions[j]      = clamp((world - bounds_min) * inv_extent, 0.0f, 1.0f);
}

__global__ void scatter_surface_kernel(
    std::uint32_t m,
    const std::uint32_t *__restrict__ pixels,
    int K,
    const std::int32_t *__restrict__ pix_to_face,
    const float *__restrict__ zbuf,
    const float3 *__restrict__ bary,
    const float3 *__restrict__ face_world,
    const float3 *__restrict__ face_normals,
    const float *__restrict__ rgb, // [m][3]
    const float3 camera_position,
    const float3 tint,
    float3 *__restrict__ color,
    float *__restrict__ depth
)
{
    const std::uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j >= m)
    {
        return;
    }
    const std::uint32_t pix = pixels[j];
    const std::size_t frag  = static_cast<std::size_t>(pix) * K;
    const int f             = pix_to_face[frag];
    const float3 b          = bary[frag];
    const float3 world      = b.x * face_world[3 * f + 0] + b.y * face_world[3 * f + 1] + b.z * face_world[3 * f + 2];
    const float3 view_dir   = normalize(camera_position - world);
    // Two-sided headlight: 35% ambient, 65% diffuse.
    const float lambert = 0.35f + 0.65f * fabsf(dot(face_normals[f], view_dir));
    const float3 albedo = make_float3(rgb[3 * j + 0], rgb[3 * j + 1], rgb[3 * j + 2]);
    color[pix]          = albedo * tint * lambert;
    depth[pix]          = zbuf[frag];
}
} // namespace

void NeuralMaterialModel::initialize(const NeuralMaterial &desc, cudaStream_t stream)
{
    desc_ = desc;
    grid_.initialize(desc.grid, desc.seed, kGridInitScale, stream);
    mlp_.initialize(grid_.n_output_dims(), 3, desc.mlp, desc.seed ^ 0x5bd1e995u, stream);
}

void clear_surface(int width, int height, float3 background, SurfaceLayer &out, cudaStream_t stream)
{
    const auto n       = static_cast<std::uint32_t>(width * height);
    out.width          = width;
    out.height         = height;
    out.covered_pixels = 0;
    out.color.resize(n);
    out.depth.resize(n);
    launch_linear(fill_surface_kernel, n, stream, n, background, out.color.data(), out.depth.data());
}

void shade_surface(
    const NeuralMaterialModel &material,
    const MeshFragments &fragments,
    const ViewFaces &faces,
    const Camera &camera,
    float3 background,
    SurfaceLayer &out,
    Arena &arena,
    cudaStream_t stream
)
{
    clear_surface(fragments.width, fragments.height, background, out, stream);
    const auto n = static_cast<std::uint32_t>(fragments.width * fragments.height);
    const int K  = fragments.faces_per_pixel;

    std::uint8_t *flags = arena.allocate<std::uint8_t>(n);
    launch_linear(flag_hits_kernel, n, stream, n, K, fragments.pix_to_face.data(), flags);
    std::uint32_t *pixels = arena.allocate<std::uint32_t>(n);
    const std::uint32_t m = prims::compact_indices(flags, n, pixels, arena, stream);
    out.covered_pixels    = m;
    if(m == 0)
    {
        return;
    }

    const NeuralMaterial &desc = material.desc();
    const float3 extent        = desc.bounds_max - desc.bounds_min;
    TSR_CHECK(extent.x > 0.f && extent.y > 0.f && extent.z > 0.f, "material bounds must have positive extent");
    const float3 inv_extent = make_float3(1.f / extent.x, 1.f / extent.y, 1.f / extent.z);

    float3 *unit_positions = arena.allocate<float3>(m);
    launch_linear(
        fragment_positions_kernel, m, stream, m, pixels, K, fragments.pix_to_face.data(), fragments.bary.data(),
        faces.world.data(), desc.bounds_min, inv_extent, unit_positions
    );

    const std::uint32_t enc_dims = material.grid().n_output_dims();
    float *encoded               = arena.allocate<float>(static_cast<std::size_t>(m) * enc_dims);
    encode_positions(material.grid(), unit_positions, m, encoded, GridInterpolation::Linear, stream);

    float *rgb = arena.allocate<float>(static_cast<std::size_t>(m) * 3);
    material.mlp().forward(encoded, m, rgb, arena, stream);

    launch_linear(
        scatter_surface_kernel, m, stream, m, pixels, K, fragments.pix_to_face.data(), fragments.zbuf.data(),
        fragments.bary.data(), faces.world.data(), faces.normals.data(), rgb, camera.position(), desc.tint,
        out.color.data(), out.depth.data()
    );
}
} // namespace tsr
