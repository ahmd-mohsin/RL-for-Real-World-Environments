// Tessera - projection of 3D Gaussians into screen-space splats.
//
// The projection kernel follows gsplat's fused EWA projection
// (projection_ewa_3dgs_fused_fwd_kernel, Apache-2.0), specialised to one
// camera and to quaternion/scale parameterised covariances.
#include "tsr/geom/gaussian_math.cuh"
#include "tsr/geom/projection.h"

namespace tsr
{
namespace
{
__global__ void project_gaussians_kernel(
    std::uint32_t n,
    const float3 *__restrict__ means,
    const float4 *__restrict__ quats,
    const float3 *__restrict__ scales,
    const float *__restrict__ opacities,
    const Camera camera,
    const float eps2d,
    const float radius_clip,
    const bool antialiased,
    int2 *__restrict__ radii,
    float2 *__restrict__ means2d,
    float *__restrict__ depths,
    float3 *__restrict__ conics,
    float *__restrict__ out_opacities
)
{
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n)
    {
        return;
    }

    // transform Gaussian centre to camera space
    const float3 mean_c = camera.world_to_camera(means[idx]);
    if(mean_c.z < camera.near_plane || mean_c.z > camera.far_plane)
    {
        radii[idx] = make_int2(0, 0);
        return;
    }

    // transform covariance to camera space
    const math::mat3 covar   = geom::quat_scale_to_covar(quats[idx], scales[idx]);
    const math::mat3 covar_c = geom::covar_world_to_camera(camera.R, covar);

    math::mat2 covar2d = math::mat2::zero();
    float2 mean2d      = make_float2(0.f, 0.f);
    switch(camera.model)
    {
    case CameraModel::Pinhole:
        geom::persp_proj(mean_c, covar_c, camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height, covar2d, mean2d);
        break;
    case CameraModel::Orthographic:
        geom::ortho_proj(mean_c, covar_c, camera.fx, camera.fy, camera.cx, camera.cy, covar2d, mean2d);
        break;
    }

    float compensation;
    const float det = geom::add_blur(eps2d, covar2d, compensation);
    if(det <= 0.f)
    {
        radii[idx] = make_int2(0, 0);
        return;
    }
    const math::mat2 covar2d_inv = math::inverse(covar2d);

    float opacity = opacities[idx];
    if(antialiased)
    {
        opacity *= compensation;
    }
    if(opacity < geom::kAlphaThreshold)
    {
        radii[idx] = make_int2(0, 0);
        return;
    }
    // Opacity-aware extent: alpha = opacity * exp(-q / 2) drops below the
    // threshold at q = 2 ln(opacity / threshold) (arXiv:2402.00525, B.2).
    const float extent = fminf(geom::kGaussianExtent, sqrtf(2.0f * __logf(opacity / geom::kAlphaThreshold)));

    // tight axis-aligned bounding box (non differentiable)
    const float radius_x = ceilf(extent * sqrtf(covar2d[0][0]));
    const float radius_y = ceilf(extent * sqrtf(covar2d[1][1]));
    if(radius_x <= radius_clip && radius_y <= radius_clip)
    {
        radii[idx] = make_int2(0, 0);
        return;
    }

    // cull splats entirely outside the image
    if(mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= camera.width || mean2d.y + radius_y <= 0
       || mean2d.y - radius_y >= camera.height)
    {
        radii[idx] = make_int2(0, 0);
        return;
    }

    radii[idx]         = make_int2(static_cast<int>(radius_x), static_cast<int>(radius_y));
    means2d[idx]       = mean2d;
    depths[idx]        = mean_c.z;
    conics[idx]        = math::pack_sym(covar2d_inv);
    out_opacities[idx] = opacity;
}

__global__ void splat_colors_kernel(
    std::uint32_t n,
    std::uint32_t sh_degree,
    const float3 *__restrict__ means,
    const float *__restrict__ sh,
    const int2 *__restrict__ radii,
    const float3 camera_position,
    float3 *__restrict__ colors
)
{
    const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n)
    {
        return;
    }
    const int2 r = radii[idx];
    if(r.x <= 0 && r.y <= 0)
    {
        return;
    }
    const std::uint32_t K = geom::sh_coeff_count(sh_degree);
    const float *coeffs   = sh + static_cast<std::size_t>(idx) * K * 3;
    const float3 dir      = means[idx] - camera_position;
    // 3DGS convention: radiance = SH(dir) + 0.5, clamped at zero.
    colors[idx] = make_float3(
        fmaxf(geom::sh_to_color(sh_degree, 0, dir, coeffs) + 0.5f, 0.f),
        fmaxf(geom::sh_to_color(sh_degree, 1, dir, coeffs) + 0.5f, 0.f),
        fmaxf(geom::sh_to_color(sh_degree, 2, dir, coeffs) + 0.5f, 0.f)
    );
}
} // namespace

void DeviceGaussians::upload(const GaussianCloud &cloud, cudaStream_t stream)
{
    cloud.validate();
    count     = static_cast<std::uint32_t>(cloud.size());
    sh_degree = cloud.sh_degree;
    means.upload(cloud.means, stream);
    quats.upload(cloud.quats, stream);
    scales.upload(cloud.scales, stream);
    opacities.upload(cloud.opacities, stream);
    sh.upload(cloud.sh, stream);
}

void ProjectedSplats::resize(std::uint32_t n)
{
    count = n;
    radii.resize(n);
    means2d.resize(n);
    depths.resize(n);
    conics.resize(n);
    opacities.resize(n);
    colors.resize(n);
}

void project_gaussians(
    const DeviceGaussians &gaussians,
    const Camera &camera,
    const ProjectionSettings &settings,
    ProjectedSplats &out,
    cudaStream_t stream
)
{
    out.resize(gaussians.count);
    launch_linear(
        project_gaussians_kernel,
        gaussians.count,
        stream,
        gaussians.count,
        gaussians.means.data(),
        gaussians.quats.data(),
        gaussians.scales.data(),
        gaussians.opacities.data(),
        camera,
        settings.eps2d,
        settings.radius_clip,
        settings.antialiased,
        out.radii.data(),
        out.means2d.data(),
        out.depths.data(),
        out.conics.data(),
        out.opacities.data()
    );
}

void compute_splat_colors(const DeviceGaussians &gaussians, const Camera &camera, ProjectedSplats &splats, cudaStream_t stream)
{
    TSR_CHECK(splats.count == gaussians.count, "splats were projected from a different Gaussian set");
    TSR_CHECK(gaussians.sh_degree <= 3, "SH degree " << gaussians.sh_degree << " is not supported (max 3)");
    launch_linear(
        splat_colors_kernel,
        gaussians.count,
        stream,
        gaussians.count,
        gaussians.sh_degree,
        gaussians.means.data(),
        gaussians.sh.data(),
        splats.radii.data(),
        camera.position(),
        splats.colors.data()
    );
}
} // namespace tsr
