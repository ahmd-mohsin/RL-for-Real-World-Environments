// Tessera - projection of 3D Gaussians into screen-space splats.
#pragma once

#include <cstdint>

#include "tsr/device/backend.h"
#include "tsr/device/memory.h"
#include "tsr/geom/camera.h"
#include "tsr/scene/scene.h"

namespace tsr
{
// Device copy of a GaussianCloud.
struct DeviceGaussians
{
    DeviceBuffer<float3> means;
    DeviceBuffer<float4> quats;
    DeviceBuffer<float3> scales;
    DeviceBuffer<float> opacities;
    DeviceBuffer<float> sh;
    std::uint32_t count     = 0;
    std::uint32_t sh_degree = 0;

    void upload(const GaussianCloud &cloud, cudaStream_t stream);
};

struct ProjectionSettings
{
    float eps2d       = 0.3f;  // screen-space low-pass variance in pixels^2
    float radius_clip = 0.f;   // splats whose radii are all <= this are culled
    bool antialiased  = false; // fold the low-pass compensation into opacity
};

// Per-Gaussian screen-space parameters. A Gaussian is culled for this view
// when radii[i] == (0, 0); every other field is then unspecified.
struct ProjectedSplats
{
    DeviceBuffer<int2> radii;      // pixel half-extent along x and y
    DeviceBuffer<float2> means2d;  // pixel coordinates
    DeviceBuffer<float> depths;    // camera-space z
    DeviceBuffer<float3> conics;   // inverse 2D covariance, upper triangle (xx, xy, yy)
    DeviceBuffer<float> opacities; // effective opacity (compensated when antialiased)
    DeviceBuffer<float3> colors;   // view-dependent radiance, clamped at 0
    std::uint32_t count = 0;

    void resize(std::uint32_t n);
};

void project_gaussians(
    const DeviceGaussians &gaussians,
    const Camera &camera,
    const ProjectionSettings &settings,
    ProjectedSplats &out,
    cudaStream_t stream
);

// Evaluates SH radiance towards the camera centre for every visible splat.
void compute_splat_colors(const DeviceGaussians &gaussians, const Camera &camera, ProjectedSplats &splats, cudaStream_t stream);

} // namespace tsr
