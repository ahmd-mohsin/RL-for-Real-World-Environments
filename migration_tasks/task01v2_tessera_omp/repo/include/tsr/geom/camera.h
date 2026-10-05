// Tessera - camera model.
//
// Conventions (OpenCV style): camera space has +x right, +y down and +z
// forward; pixel (u, v) has its origin at the top-left corner of the image
// and the centre of pixel (i, j) is at (i + 0.5, j + 0.5).
#pragma once

#include <cstdint>

#include "tsr/device/backend.h"
#include "tsr/math/mat.h"
#include "tsr/math/vec.h"

namespace tsr
{
enum class CameraModel : std::uint32_t
{
    Pinhole      = 0,
    Orthographic = 1,
};

struct Camera
{
    math::mat3 R;   // world -> camera rotation (rows are the camera axes in world space)
    float3 t;       // world -> camera translation
    float fx, fy;   // focal lengths in pixels (pixels per world unit for orthographic)
    float cx, cy;   // principal point in pixels
    std::uint32_t width, height;
    float near_plane    = 0.01f;
    float far_plane     = 1e10f;
    CameraModel model   = CameraModel::Pinhole;

    TSR_HDI float3 world_to_camera(const float3 &p) const { return R * p + t; }

    TSR_HDI float3 position() const
    {
        // C = -R^T t
        const float3 c = math::transpose(R) * t;
        return make_float3(-c.x, -c.y, -c.z);
    }

    // Projects a camera-space point to continuous pixel coordinates.
    TSR_HDI float2 project(const float3 &pc) const
    {
        if(model == CameraModel::Orthographic)
        {
            return make_float2(fx * pc.x + cx, fy * pc.y + cy);
        }
        const float rz = 1.f / pc.z;
        return make_float2(fx * pc.x * rz + cx, fy * pc.y * rz + cy);
    }

    TSR_HDI std::uint32_t tiles_x(std::uint32_t tile) const { return (width + tile - 1) / tile; }
    TSR_HDI std::uint32_t tiles_y(std::uint32_t tile) const { return (height + tile - 1) / tile; }

    // Builds a pinhole camera at `eye` looking at `target`; `up` is the world
    // up direction and fov_y is the vertical field of view in degrees.
    static Camera look_at(float3 eye, float3 target, float3 up, float fov_y_degrees, std::uint32_t width, std::uint32_t height);
};

} // namespace tsr
