// Tessera - camera helpers.
#include <cmath>

#include "tsr/geom/camera.h"

namespace tsr
{
Camera Camera::look_at(float3 eye, float3 target, float3 up, float fov_y_degrees, std::uint32_t width, std::uint32_t height)
{
    TSR_CHECK(width > 0 && height > 0, "camera resolution must be positive");
    TSR_CHECK(fov_y_degrees > 0.f && fov_y_degrees < 180.f, "vertical field of view must be in (0, 180) degrees");
    const float3 forward = normalize(target - eye);
    // Right-handed camera frame with +y pointing down the image, i.e. against
    // the world up vector: right = forward x up, down = forward x right.
    float3 right = cross(forward, up);
    TSR_CHECK(dot(right, right) > 1e-12f, "look_at: up vector is parallel to the viewing direction");
    right             = normalize(right);
    const float3 down = cross(forward, right);

    Camera cam{};
    // Rows of R are the camera axes expressed in world coordinates.
    cam.R = math::make_mat3(right.x, down.x, forward.x, right.y, down.y, forward.y, right.z, down.z, forward.z);
    const float3 r_eye = cam.R * eye;
    cam.t              = make_float3(-r_eye.x, -r_eye.y, -r_eye.z);

    const float fov_y = fov_y_degrees * 3.14159265358979f / 180.f;
    cam.fy            = 0.5f * static_cast<float>(height) / std::tan(0.5f * fov_y);
    cam.fx            = cam.fy;
    cam.cx            = 0.5f * static_cast<float>(width);
    cam.cy            = 0.5f * static_cast<float>(height);
    cam.width         = width;
    cam.height        = height;
    cam.near_plane    = 0.01f;
    cam.far_plane     = 1e10f;
    cam.model         = CameraModel::Pinhole;
    return cam;
}
} // namespace tsr
