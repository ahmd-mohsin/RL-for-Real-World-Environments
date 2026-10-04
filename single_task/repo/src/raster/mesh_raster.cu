// Tessera - fine rasterization of binned mesh faces.
//
// Derived from PyTorch3D's RasterizeMeshesFineCudaKernel and the geometry
// helpers it uses (rasterize_meshes.cu, utils/geometry_utils.cuh,
// BSD-3-Clause, Copyright Meta Platforms, Inc. and affiliates). Reduced to one
// image; the clipped-face neighbour logic is dropped (Tessera does not clip).
#include <algorithm>
#include <vector>

#include "tsr/geom/mesh_ndc.cuh"
#include "tsr/math/vec.h"
#include "tsr/raster/mesh_raster.h"

namespace tsr
{
namespace
{
constexpr float kEpsilon = 1e-8f;

struct Pixel
{
    float z;
    std::int32_t idx; // face index
    float dist;       // signed squared distance of the pixel to the face
    float3 bary;
};

TSR_DI bool operator<(const Pixel &a, const Pixel &b) { return a.z < b.z || (a.z == b.z && a.idx < b.idx); }

TSR_DI float edge_function(const float2 &p, const float2 &v0, const float2 &v1)
{
    return (p.x - v0.x) * (v1.y - v0.y) - (p.y - v0.y) * (v1.x - v0.x);
}

TSR_DI float3 barycentric_coords(const float2 &p, const float2 &v0, const float2 &v1, const float2 &v2)
{
    const float area = edge_function(v2, v0, v1) + kEpsilon;
    const float w0   = edge_function(p, v1, v2) / area;
    const float w1   = edge_function(p, v2, v0) / area;
    const float w2   = edge_function(p, v0, v1) / area;
    return make_float3(w0, w1, w2);
}

TSR_DI float3 barycentric_perspective_correction(const float3 &bary, float z0, float z1, float z2)
{
    const float w0_top = bary.x * z1 * z2;
    const float w1_top = z0 * bary.y * z2;
    const float w2_top = z0 * z1 * bary.z;
    const float denom  = fmaxf(w0_top + w1_top + w2_top, kEpsilon);
    return make_float3(w0_top / denom, w1_top / denom, w2_top / denom);
}

TSR_DI float3 barycentric_clip(const float3 &bary)
{
    float3 w    = make_float3(fmaxf(bary.x, 0.f), fmaxf(bary.y, 0.f), fmaxf(bary.z, 0.f));
    const float sum = fmaxf(w.x + w.y + w.z, 1e-5f);
    w.x /= sum;
    w.y /= sum;
    w.z /= sum;
    return w;
}

// Squared distance from p to segment ab.
TSR_DI float point_line_distance(const float2 &p, const float2 &a, const float2 &b)
{
    const float2 ba = b - a;
    const float l2  = dot(ba, ba);
    if(l2 <= kEpsilon)
    {
        return dot(p - b, p - b);
    }
    const float t       = __saturatef(dot(ba, p - a) / l2);
    const float2 p_proj = a + t * ba;
    const float2 d      = p_proj - p;
    return dot(d, d);
}

TSR_DI float point_triangle_distance(const float2 &p, const float2 &v0, const float2 &v1, const float2 &v2)
{
    return fminf(fminf(point_line_distance(p, v0, v1), point_line_distance(p, v0, v2)), point_line_distance(p, v1, v2));
}

// Updates the per-pixel top-K queue (unsorted, with a cached maximum) with
// face `face_idx` if it covers pixel pxy.
TSR_DI void check_pixel_inside_face(
    const float3 *__restrict__ face_verts,
    int face_idx,
    int &q_size,
    float &q_max_z,
    int &q_max_idx,
    Pixel *q,
    float blur_radius,
    float2 pxy,
    int K,
    bool perspective_correct,
    bool clip_barycentric_coords,
    bool cull_backfaces
)
{
    const float3 v0 = face_verts[3 * face_idx + 0];
    const float3 v1 = face_verts[3 * face_idx + 1];
    const float3 v2 = face_verts[3 * face_idx + 2];

    const float2 v0xy = make_float2(v0.x, v0.y);
    const float2 v1xy = make_float2(v1.x, v1.y);
    const float2 v2xy = make_float2(v2.x, v2.y);

    // bounding box test, padded by the blur radius (in NDC units)
    const float r        = sqrtf(blur_radius);
    const float x_min    = fminf(v0.x, fminf(v1.x, v2.x)) - r;
    const float x_max    = fmaxf(v0.x, fmaxf(v1.x, v2.x)) + r;
    const float y_min    = fminf(v0.y, fminf(v1.y, v2.y)) - r;
    const float y_max    = fmaxf(v0.y, fmaxf(v1.y, v2.y)) + r;
    const float zmin     = fminf(v0.z, fminf(v1.z, v2.z));
    const float zmax     = fmaxf(v0.z, fmaxf(v1.z, v2.z));
    const bool outside   = pxy.x > x_max || pxy.x < x_min || pxy.y > y_max || pxy.y < y_min || zmin < kEpsilon;

    const float face_area      = edge_function(v0xy, v1xy, v2xy);
    const bool back_face       = face_area < 0.0f;
    const bool zero_face_area  = face_area <= kEpsilon && face_area >= -1.0f * kEpsilon;
    if(zmax < 0 || (cull_backfaces && back_face) || outside || zero_face_area)
    {
        return;
    }

    const float3 p_bary0     = barycentric_coords(pxy, v0xy, v1xy, v2xy);
    const float3 p_bary      = !perspective_correct ? p_bary0 : barycentric_perspective_correction(p_bary0, v0.z, v1.z, v2.z);
    const float3 p_bary_clip = !clip_barycentric_coords ? p_bary : barycentric_clip(p_bary);

    const float pz = p_bary_clip.x * v0.z + p_bary_clip.y * v1.z + p_bary_clip.z * v2.z;
    if(pz < 0)
    {
        return; // behind the image plane
    }

    // The inside test uses the unclipped barycentrics, otherwise every pixel
    // would count as inside.
    const float dist        = point_triangle_distance(pxy, v0xy, v1xy, v2xy);
    const bool inside       = p_bary.x > 0.0f && p_bary.y > 0.0f && p_bary.z > 0.0f;
    const float signed_dist = inside ? -dist : dist;
    if(!inside && dist >= blur_radius)
    {
        return;
    }

    if(q_size < K)
    {
        q[q_size] = {pz, face_idx, signed_dist, p_bary_clip};
        if(pz > q_max_z)
        {
            q_max_z   = pz;
            q_max_idx = q_size;
        }
        ++q_size;
    }
    else if(pz < q_max_z)
    {
        // Replace the current maximum, then find the new one.
        q[q_max_idx] = {pz, face_idx, signed_dist, p_bary_clip};
        q_max_z      = pz;
        for(int i = 0; i < K; ++i)
        {
            if(q[i].z > q_max_z)
            {
                q_max_z   = q[i].z;
                q_max_idx = i;
            }
        }
    }
}

template<typename T>
TSR_DI void bubble_sort(T *arr, int n)
{
    // Only used for tiny thread-local arrays, where divergence matters more
    // than asymptotic cost.
    for(int i = 0; i < n - 1; ++i)
    {
        for(int j = 0; j < n - i - 1; ++j)
        {
            if(arr[j + 1] < arr[j])
            {
                const T tmp = arr[j];
                arr[j]      = arr[j + 1];
                arr[j + 1]  = tmp;
            }
        }
    }
}

__global__ void rasterize_mesh_fine_kernel(
    const float3 *__restrict__ face_verts,
    const std::int32_t *__restrict__ faces_per_bin, // [BH][BW]
    const std::int32_t *__restrict__ bin_faces,     // [BH][BW][M]
    float blur_radius,
    int bin_size,
    bool perspective_correct,
    bool clip_barycentric_coords,
    bool cull_backfaces,
    int BH,
    int BW,
    int M,
    int H,
    int W,
    int K,
    std::int32_t *__restrict__ face_idxs,
    float *__restrict__ zbuf,
    float *__restrict__ pix_dists,
    float3 *__restrict__ bary
)
{
    // Can exceed H * W when the image is not divisible by bin_size.
    const int num_pixels  = BH * BW * bin_size * bin_size;
    const int num_threads = static_cast<int>(gridDim.x * blockDim.x);
    const int tid         = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    for(int pid = tid; pid < num_pixels; pid += num_threads)
    {
        // Pixels within a bin move fastest so neighbouring threads share a
        // bin and read the same face list.
        int i        = pid;
        const int by = i / (BW * bin_size * bin_size);
        i           %= BW * bin_size * bin_size;
        const int bx = i / (bin_size * bin_size);
        i           %= bin_size * bin_size;

        // raster indices
        const int yi = i / bin_size + by * bin_size;
        const int xi = i % bin_size + bx * bin_size;
        if(yi >= H || xi >= W)
        {
            continue;
        }
        const float2 pxy = make_float2(geom::pix_to_non_square_ndc(xi, W, H), geom::pix_to_non_square_ndc(yi, H, W));

        Pixel q[kMaxFacesPerPixel];
        int q_size    = 0;
        float q_max_z = -1000.f;
        int q_max_idx = -1;

        const int bin      = by * BW + bx;
        const int n_in_bin = min(faces_per_bin[bin], M);
        for(int m = 0; m < n_in_bin; ++m)
        {
            const int f = bin_faces[bin * M + m];
            if(f < 0)
            {
                continue;
            }
            check_pixel_inside_face(
                face_verts, f, q_size, q_max_z, q_max_idx, q, blur_radius, pxy, K, perspective_correct,
                clip_barycentric_coords, cull_backfaces
            );
        }
        bubble_sort(q, q_size);

        // Raster index -> image index: +x NDC is left and +y NDC is up.
        const int yidx    = H - 1 - yi;
        const int xidx    = W - 1 - xi;
        const int pix_idx = (yidx * W + xidx) * K;
        for(int k = 0; k < q_size; ++k)
        {
            face_idxs[pix_idx + k] = q[k].idx;
            zbuf[pix_idx + k]      = q[k].z;
            pix_dists[pix_idx + k] = q[k].dist;
            bary[pix_idx + k]      = q[k].bary;
        }
    }
}
} // namespace

void rasterize_mesh_fine(
    const ViewFaces &faces,
    const MeshBins &bins,
    int width,
    int height,
    const MeshRasterSettings &settings,
    MeshFragments &out,
    cudaStream_t stream
)
{
    TSR_CHECK(settings.faces_per_pixel >= 1 && settings.faces_per_pixel <= kMaxFacesPerPixel,
              "faces_per_pixel must be in [1, " << kMaxFacesPerPixel << "]");
    const int K         = settings.faces_per_pixel;
    const std::size_t n = static_cast<std::size_t>(width) * height * K;
    out.width           = width;
    out.height          = height;
    out.faces_per_pixel = K;
    out.pix_to_face.resize(n);
    out.zbuf.resize(n);
    out.bary.resize(n);
    out.dists.resize(n);
    TSR_CUDA_CHECK(cudaMemsetAsync(out.pix_to_face.data(), 0xff, out.pix_to_face.bytes(), stream));
    // -1.0f everywhere: 0xbf800000
    std::vector<float> minus_one(n, -1.f);
    out.zbuf.upload(minus_one, stream);
    out.dists.upload(minus_one, stream);
    out.bary.zero(stream);
    if(faces.n_faces == 0)
    {
        return;
    }

    const int threads = 128;
    const int pixels  = bins.bins_x * bins.bins_y * bins.bin_size * bins.bin_size;
    const int blocks  = std::min(1024, div_round_up(pixels, threads));
    launch(
        rasterize_mesh_fine_kernel,
        LaunchConfig{dim3(blocks), dim3(threads), 0, stream},
        faces.ndc.data(), bins.faces_per_bin.data(), bins.bin_faces.data(), settings.blur_radius, bins.bin_size,
        settings.perspective_correct, settings.clip_barycentric_coords, settings.cull_backfaces, bins.bins_y,
        bins.bins_x, bins.max_faces_per_bin, height, width, K, out.pix_to_face.data(), out.zbuf.data(),
        out.dists.data(), out.bary.data()
    );
}
} // namespace tsr
