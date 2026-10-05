// Tests for matrices, the camera model, Gaussian projection and NDC mapping.
#include <cmath>
#include <vector>

#include "doctest.h"
#include "tsr/geom/gaussian_math.cuh"
#include "tsr/geom/mesh_ndc.cuh"
#include "tsr/geom/projection.h"
#include "tsr/math/mat.h"

using namespace tsr;

TEST_CASE("math: 3x3 inverse and transpose")
{
    const math::mat3 A  = math::make_mat3(2.f, 0.5f, -1.f, 0.3f, 3.f, 0.2f, -0.7f, 0.1f, 1.5f);
    const math::mat3 I  = A * math::inverse(A);
    for(int c = 0; c < 3; ++c)
    {
        for(int r = 0; r < 3; ++r)
        {
            CHECK(I[c][r] == doctest::Approx(c == r ? 1.f : 0.f).epsilon(1e-5));
        }
    }
    const math::mat3 At = math::transpose(A);
    CHECK(At[0][1] == A[1][0]);
    const float3 v = make_float3(1.f, 2.f, 3.f);
    const float3 Av = A * v;
    // column-major: (A v).x = sum_c A[c][0] * v_c
    CHECK(Av.x == doctest::Approx(2.f * 1 + 0.3f * 2 - 0.7f * 3));
}

TEST_CASE("geom: quaternion rotation is orthonormal and handles unnormalized input")
{
    const math::mat3 R = geom::quat_to_rotmat(make_float4(2.f, 0.4f, -1.2f, 0.6f));
    const math::mat3 RtR = math::transpose(R) * R;
    for(int c = 0; c < 3; ++c)
    {
        for(int r = 0; r < 3; ++r)
        {
            CHECK(RtR[c][r] == doctest::Approx(c == r ? 1.f : 0.f).epsilon(1e-5));
        }
    }
    CHECK(math::determinant(R) == doctest::Approx(1.f).epsilon(1e-5));
    // identity quaternion
    const math::mat3 I = geom::quat_to_rotmat(make_float4(1.f, 0.f, 0.f, 0.f));
    CHECK(I[0][0] == 1.f);
    CHECK(I[1][0] == 0.f);
}

TEST_CASE("geom: look_at produces an upright right-handed camera")
{
    const Camera cam = Camera::look_at(make_float3(0.f, 0.f, -5.f), make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), 60.f, 64, 48);
    // The target is straight ahead at depth 5.
    const float3 pc = cam.world_to_camera(make_float3(0.f, 0.f, 0.f));
    CHECK(pc.x == doctest::Approx(0.f));
    CHECK(pc.z == doctest::Approx(5.f));
    // A point above the target appears above the image centre (smaller v).
    const float2 up = cam.project(cam.world_to_camera(make_float3(0.f, 1.f, 0.f)));
    CHECK(up.y < 24.f);
    CHECK(up.x == doctest::Approx(32.f));
    const float3 c = cam.position();
    CHECK(c.z == doctest::Approx(-5.f));
    CHECK(math::determinant(cam.R) == doctest::Approx(1.f));
}

TEST_CASE("geom: pixel/NDC mapping matches the rasterizer's pixel centres")
{
    for(int W : {48, 64, 65})
    {
        for(int H : {48, 31})
        {
            for(int c = 0; c < W; c += 7)
            {
                CHECK(geom::pixel_x_to_ndc(c + 0.5f, W, H) == doctest::Approx(geom::pix_to_non_square_ndc(W - 1 - c, W, H)).epsilon(1e-5));
            }
            for(int r = 0; r < H; r += 5)
            {
                CHECK(geom::pixel_y_to_ndc(r + 0.5f, H, W) == doctest::Approx(geom::pix_to_non_square_ndc(H - 1 - r, H, W)).epsilon(1e-5));
            }
        }
    }
}

TEST_CASE("geom: projection of an isotropic Gaussian")
{
    GaussianCloud cloud;
    cloud.sh_degree = 0;
    cloud.means     = {make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, -10.f), make_float3(50.f, 0.f, 0.f)};
    cloud.quats     = {make_float4(1.f, 0.f, 0.f, 0.f), make_float4(1.f, 0.f, 0.f, 0.f), make_float4(1.f, 0.f, 0.f, 0.f)};
    cloud.scales    = {make_float3(0.1f, 0.1f, 0.1f), make_float3(0.1f, 0.1f, 0.1f), make_float3(0.1f, 0.1f, 0.1f)};
    cloud.opacities = {0.9f, 0.9f, 0.9f};
    cloud.sh        = {0.5f, 0.f, -0.5f, 0.5f, 0.f, -0.5f, 0.5f, 0.f, -0.5f};

    const Camera cam = Camera::look_at(make_float3(0.f, 0.f, -5.f), make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), 60.f, 64, 48);
    DeviceGaussians dg;
    dg.upload(cloud, nullptr);
    ProjectedSplats out;
    ProjectionSettings ps;
    project_gaussians(dg, cam, ps, out, nullptr);
    compute_splat_colors(dg, cam, out, nullptr);

    const auto radii  = out.radii.to_host();
    const auto means  = out.means2d.to_host();
    const auto depths = out.depths.to_host();
    const auto conics = out.conics.to_host();
    const auto colors = out.colors.to_host();

    // visible, centred, at depth 5
    REQUIRE(radii[0].x > 0);
    CHECK(means[0].x == doctest::Approx(32.f));
    CHECK(means[0].y == doctest::Approx(24.f));
    CHECK(depths[0] == doctest::Approx(5.f));
    // isotropic: conic has no cross term and equal diagonal
    CHECK(conics[0].y == doctest::Approx(0.f).epsilon(1e-6));
    CHECK(conics[0].x == doctest::Approx(conics[0].z));
    // pixel std = f * s / z, plus the 0.3 px^2 low-pass
    const float sigma_px = cam.fx * 0.1f / 5.f;
    CHECK(1.f / conics[0].x == doctest::Approx(sigma_px * sigma_px + 0.3f).epsilon(1e-4));
    // behind the camera / far outside the image -> culled
    CHECK(radii[1].x == 0);
    CHECK(radii[2].x == 0);
    // degree-0 SH: 0.2820948 * c + 0.5, clamped at zero
    CHECK(colors[0].x == doctest::Approx(0.2820948f * 0.5f + 0.5f));
    CHECK(colors[0].y == doctest::Approx(0.5f));
    CHECK(colors[0].z == doctest::Approx(0.5f - 0.2820948f * 0.5f));
}

TEST_CASE("geom: antialiased projection folds the compensation into opacity")
{
    GaussianCloud cloud;
    cloud.means     = {make_float3(0.f, 0.f, 0.f)};
    cloud.quats     = {make_float4(1.f, 0.f, 0.f, 0.f)};
    cloud.scales    = {make_float3(0.01f, 0.01f, 0.01f)};
    cloud.opacities = {0.8f};
    cloud.sh        = {0.f, 0.f, 0.f};
    const Camera cam = Camera::look_at(make_float3(0.f, 0.f, -5.f), make_float3(0.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), 60.f, 64, 48);
    DeviceGaussians dg;
    dg.upload(cloud, nullptr);
    ProjectedSplats plain, aa;
    ProjectionSettings ps;
    project_gaussians(dg, cam, ps, plain, nullptr);
    ps.antialiased = true;
    project_gaussians(dg, cam, ps, aa, nullptr);
    CHECK(plain.opacities.to_host()[0] == doctest::Approx(0.8f));
    CHECK(aa.opacities.to_host()[0] < 0.8f);
}
