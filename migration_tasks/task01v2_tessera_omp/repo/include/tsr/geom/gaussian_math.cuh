// Tessera - per-Gaussian geometry: covariance construction, EWA projection,
// low-pass filtering and spherical-harmonic radiance.
//
// Derived from gsplat (Apache-2.0, Copyright the Regents of the University of
// California, Nerfstudio Team and contributors; NVIDIA CORPORATION &
// AFFILIATES), see third_party/LICENSES/gsplat.txt. Modified for Tessera:
// rewritten against tsr::math matrices instead of glm, forward pass only.
#pragma once

#include <cstdint>

#include "tsr/device/backend.h"
#include "tsr/geom/camera.h"
#include "tsr/math/mat.h"
#include "tsr/math/vec.h"

namespace tsr::geom
{
// Alpha below which a splat is ignored by the rasterizer.
constexpr float kAlphaThreshold = 1.f / 255.f;
// Truncation of the Gaussian footprint, in standard deviations.
constexpr float kGaussianExtent = 3.33f;
// Upper bound on per-splat alpha. With this cap a fully opaque splat must be
// composited twice to drive transmittance below kTransmittanceThreshold.
constexpr float kMaxAlpha               = 0.99f;
constexpr float kTransmittanceThreshold = 1e-4f;
// Floor for the antialiasing compensation sqrt(det_orig / det_blur).
constexpr float kMinCompensation = 0.005f;

TSR_HDI float safe_rsqrt(float x) { return x > 0.f ? rsqrtf(x) : 0.f; }

// Quaternion (w, x, y, z), not necessarily normalized, to rotation matrix.
TSR_HDI math::mat3 quat_to_rotmat(const float4 &quat)
{
    float w = quat.x, x = quat.y, y = quat.z, z = quat.w;
    const float inv_norm = safe_rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    const float x2 = x * x, y2 = y * y, z2 = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    return math::make_mat3(
        1.f - 2.f * (y2 + z2), 2.f * (xy + wz), 2.f * (xz - wy), // 1st column
        2.f * (xy - wz), 1.f - 2.f * (x2 + z2), 2.f * (yz + wx), // 2nd column
        2.f * (xz + wy), 2.f * (yz - wx), 1.f - 2.f * (x2 + y2)  // 3rd column
    );
}

// Sigma = R S S^T R^T for scale S = diag(scale).
TSR_HDI math::mat3 quat_scale_to_covar(const float4 &quat, const float3 &scale)
{
    const math::mat3 R = quat_to_rotmat(quat);
    const math::mat3 M = R * math::mat3::diagonal(scale);
    return M * math::transpose(M);
}

TSR_HDI math::mat3 covar_world_to_camera(const math::mat3 &R, const math::mat3 &covar_world)
{
    return R * covar_world * math::transpose(R);
}

// First-order (EWA) projection of a camera-space Gaussian through a pinhole.
// The Jacobian is evaluated at a point clamped to 1.3x the field of view so
// that Gaussians far outside the frustum do not produce huge footprints.
TSR_HDI void persp_proj(
    const float3 &mean_c,
    const math::mat3 &cov_c,
    float fx,
    float fy,
    float cx,
    float cy,
    std::uint32_t width,
    std::uint32_t height,
    math::mat2 &cov2d,
    float2 &mean2d
)
{
    const float x = mean_c.x, y = mean_c.y, z = mean_c.z;

    const float tan_fovx  = 0.5f * width / fx;
    const float tan_fovy  = 0.5f * height / fy;
    const float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    const float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    const float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    const float rz  = 1.f / z;
    const float rz2 = rz * rz;
    const float tx  = z * fminf(lim_x_pos, fmaxf(-lim_x_neg, x * rz));
    const float ty  = z * fminf(lim_y_pos, fmaxf(-lim_y_neg, y * rz));

    // 3 columns x 2 rows
    const math::mat3x2 J = math::make_mat3x2(fx * rz, 0.f, 0.f, fy * rz, -fx * tx * rz2, -fy * ty * rz2);
    cov2d                = J * cov_c * math::transpose(J);
    mean2d               = make_float2(fx * x * rz + cx, fy * y * rz + cy);
}

TSR_HDI void ortho_proj(
    const float3 &mean_c,
    const math::mat3 &cov_c,
    float fx,
    float fy,
    float cx,
    float cy,
    math::mat2 &cov2d,
    float2 &mean2d
)
{
    const math::mat3x2 J = math::make_mat3x2(fx, 0.f, 0.f, fy, 0.f, 0.f);
    cov2d                = J * cov_c * math::transpose(J);
    mean2d               = make_float2(fx * mean_c.x + cx, fy * mean_c.y + cy);
}

// Adds an isotropic screen-space low-pass filter of variance eps2d (in
// pixels^2) and returns the blurred determinant. `compensation` rescales
// opacity so that the filtered Gaussian keeps its integrated weight.
TSR_HDI float add_blur(float eps2d, math::mat2 &covar, float &compensation)
{
    const float det_orig  = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0]          += eps2d;
    covar[1][1]          += eps2d;
    const float det_blur  = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation          = sqrtf(fmaxf(kMinCompensation * kMinCompensation, det_orig / det_blur));
    return det_blur;
}

// Number of SH coefficients per channel for a degree.
TSR_HDI constexpr std::uint32_t sh_coeff_count(std::uint32_t degree) { return (degree + 1) * (degree + 1); }

// Evaluates real spherical harmonics up to `degree` (<= 3) along `dir`
// (any length) for one colour channel. coeffs is laid out [K][3].
// Sloan, "Efficient Spherical Harmonic Evaluation", JCGT 2013.
TSR_HDI float sh_to_color(std::uint32_t degree, std::uint32_t c, const float3 &dir, const float *coeffs)
{
    constexpr std::uint32_t D = 3;
    float result              = 0.2820947917738781f * coeffs[c];
    if(degree >= 1)
    {
        const float inorm = safe_rsqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        const float x     = dir.x * inorm;
        const float y     = dir.y * inorm;
        const float z     = dir.z * inorm;

        result += 0.48860251190292f * (-y * coeffs[1 * D + c] + z * coeffs[2 * D + c] - x * coeffs[3 * D + c]);
        if(degree >= 2)
        {
            const float z2     = z * z;
            const float fTmp0B = -1.092548430592079f * z;
            const float fC1    = x * x - y * y;
            const float fS1    = 2.f * x * y;
            const float pSH6   = (0.9461746957575601f * z2 - 0.3153915652525201f);
            const float pSH7   = fTmp0B * x;
            const float pSH5   = fTmp0B * y;
            const float pSH8   = 0.5462742152960395f * fC1;
            const float pSH4   = 0.5462742152960395f * fS1;

            result += pSH4 * coeffs[4 * D + c] + pSH5 * coeffs[5 * D + c] + pSH6 * coeffs[6 * D + c]
                    + pSH7 * coeffs[7 * D + c] + pSH8 * coeffs[8 * D + c];
            if(degree >= 3)
            {
                const float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                const float fTmp1B = 1.445305721320277f * z;
                const float fC2    = x * fC1 - y * fS1;
                const float fS2    = x * fS1 + y * fC1;
                const float pSH12  = z * (1.865881662950577f * z2 - 1.119528997770346f);
                const float pSH13  = fTmp0C * x;
                const float pSH11  = fTmp0C * y;
                const float pSH14  = fTmp1B * fC1;
                const float pSH10  = fTmp1B * fS1;
                const float pSH15  = -0.5900435899266435f * fC2;
                const float pSH9   = -0.5900435899266435f * fS2;

                result += pSH9 * coeffs[9 * D + c] + pSH10 * coeffs[10 * D + c] + pSH11 * coeffs[11 * D + c]
                        + pSH12 * coeffs[12 * D + c] + pSH13 * coeffs[13 * D + c] + pSH14 * coeffs[14 * D + c]
                        + pSH15 * coeffs[15 * D + c];
            }
        }
    }
    return result;
}

} // namespace tsr::geom
