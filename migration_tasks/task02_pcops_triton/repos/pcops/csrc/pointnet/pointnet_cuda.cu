// pcops - PointNet++ set-abstraction / feature-propagation operators.
//
//   ball_query         first `nsample` neighbours (in index order) within a radius
//   group_points       gather features of grouped neighbours (+ backward)
//   gather_points      gather features of sampled points (+ backward)
//   three_nn           three nearest known points of every unknown point
//   three_interpolate  inverse-distance interpolation from three neighbours (+ backward)
//
// Layouts follow PointNet++: xyz is [B, N, 3] (point-major), features are
// [B, C, N] (channel-major). All index tensors are int32. Distances are
// computed in the op-math type of the input (float for half/bfloat16).
#include "pointnet_cuda.h"

#include <ATen/OpMathType.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define PN_THREADS 256

namespace pcops
{
namespace
{
inline int64_t pn_blocks(int64_t n) { return (n + PN_THREADS - 1) / PN_THREADS; }

// One thread per query point. Neighbours are taken in increasing point
// index; the first neighbour found fills every slot so that rows with fewer
// than nsample neighbours are padded with it. Queries without any neighbour
// keep index 0 (the output is zero-initialised).
template<typename scalar_t>
__global__ void ball_query_kernel(int64_t b, int64_t n, int64_t m, float radius2, int64_t nsample,
                                  const scalar_t *__restrict__ new_xyz, // [B, M, 3]
                                  const scalar_t *__restrict__ xyz,     // [B, N, 3]
                                  int32_t *__restrict__ idx)            // [B, M, S]
{
    using acc_t     = at::opmath_type<scalar_t>;
    const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if(t >= b * m)
    {
        return;
    }
    const int64_t bi = t / m;
    const scalar_t *q = new_xyz + t * 3;
    const scalar_t *p = xyz + bi * n * 3;
    int32_t *out      = idx + t * nsample;
    const acc_t qx = q[0], qy = q[1], qz = q[2];
    int64_t cnt    = 0;
    for(int64_t k = 0; k < n && cnt < nsample; ++k)
    {
        const acc_t dx = static_cast<acc_t>(p[k * 3 + 0]) - qx;
        const acc_t dy = static_cast<acc_t>(p[k * 3 + 1]) - qy;
        const acc_t dz = static_cast<acc_t>(p[k * 3 + 2]) - qz;
        const acc_t d2 = dx * dx + dy * dy + dz * dz;
        if(d2 < static_cast<acc_t>(radius2))
        {
            if(cnt == 0)
            {
                for(int64_t s = 0; s < nsample; ++s)
                {
                    out[s] = static_cast<int32_t>(k);
                }
            }
            out[cnt++] = static_cast<int32_t>(k);
        }
    }
}

template<typename scalar_t>
__global__ void group_points_kernel(int64_t b, int64_t c, int64_t n, int64_t m, int64_t s,
                                    const scalar_t *__restrict__ features, // [B, C, N]
                                    const int32_t *__restrict__ idx,       // [B, M, S]
                                    scalar_t *__restrict__ out)            // [B, C, M, S]
{
    const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if(t >= b * c * m * s)
    {
        return;
    }
    const int64_t si = t % s;
    const int64_t mi = (t / s) % m;
    const int64_t ci = (t / (s * m)) % c;
    const int64_t bi = t / (s * m * c);
    const int32_t k  = idx[(bi * m + mi) * s + si];
    out[t]           = features[(bi * c + ci) * n + k];
}

template<typename scalar_t>
__global__ void group_points_grad_kernel(int64_t b, int64_t c, int64_t n, int64_t m, int64_t s,
                                         const scalar_t *__restrict__ grad_out, // [B, C, M, S]
                                         const int32_t *__restrict__ idx,       // [B, M, S]
                                         scalar_t *__restrict__ grad_features)  // [B, C, N]
{
    const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if(t >= b * c * m * s)
    {
        return;
    }
    const int64_t si = t % s;
    const int64_t mi = (t / s) % m;
    const int64_t ci = (t / (s * m)) % c;
    const int64_t bi = t / (s * m * c);
    const int32_t k  = idx[(bi * m + mi) * s + si];
    gpuAtomicAdd(grad_features + (bi * c + ci) * n + k, grad_out[t]);
}

template<typename scalar_t>
__global__ void gather_points_kernel(int64_t b, int64_t c, int64_t n, int64_t m, const scalar_t *__restrict__ features,
                                     const int32_t *__restrict__ idx, scalar_t *__restrict__ out)
{
    const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if(t >= b * c * m)
    {
        return;
    }
    const int64_t mi = t % m;
    const int64_t bi = t / (m * c);
    const int64_t ci = (t / m) % c;
    out[t]           = features[(bi * c + ci) * n + idx[bi * m + mi]];
}

template<typename scalar_t>
__global__ void gather_points_grad_kernel(int64_t b, int64_t c, int64_t n, int64_t m, const scalar_t *__restrict__ grad_out,
                                          const int32_t *__restrict__ idx, scalar_t *__restrict__ grad_features)
{
    const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if(t >= b * c * m)
    {
        return;
    }
    const int64_t mi = t % m;
    const int64_t bi = t / (m * c);
    const int64_t ci = (t / m) % c;
    gpuAtomicAdd(grad_features + (bi * c + ci) * n + idx[bi * m + mi], grad_out[t]);
}

// One thread per unknown point. Strict comparisons keep the earliest index on
// ties. With fewer than three known points the unused slots keep index 0 and
// distance 1e40 (+inf once stored in a float tensor).
template<typename scalar_t>
__global__ void three_nn_kernel(int64_t b, int64_t n, int64_t m,
                                const scalar_t *__restrict__ unknown, // [B, n, 3]
                                const scalar_t *__restrict__ known,   // [B, m, 3]
                                scalar_t *__restrict__ dist2,         // [B, n, 3]
                                int32_t *__restrict__ idx)            // [B, n, 3]
{
    using acc_t     = at::opmath_type<scalar_t>;
    const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if(t >= b * n)
    {
        return;
    }
    const int64_t bi  = t / n;
    const scalar_t *u = unknown + t * 3;
    const scalar_t *k = known + bi * m * 3;
    const acc_t ux = u[0], uy = u[1], uz = u[2];
    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int32_t besti1 = 0, besti2 = 0, besti3 = 0;
    for(int64_t j = 0; j < m; ++j)
    {
        const acc_t dx = static_cast<acc_t>(k[j * 3 + 0]) - ux;
        const acc_t dy = static_cast<acc_t>(k[j * 3 + 1]) - uy;
        const acc_t dz = static_cast<acc_t>(k[j * 3 + 2]) - uz;
        const acc_t d  = dx * dx + dy * dy + dz * dz;
        if(d < best1)
        {
            best3  = best2;
            besti3 = besti2;
            best2  = best1;
            besti2 = besti1;
            best1  = d;
            besti1 = static_cast<int32_t>(j);
        }
        else if(d < best2)
        {
            best3  = best2;
            besti3 = besti2;
            best2  = d;
            besti2 = static_cast<int32_t>(j);
        }
        else if(d < best3)
        {
            best3  = d;
            besti3 = static_cast<int32_t>(j);
        }
    }
    dist2[t * 3 + 0] = static_cast<scalar_t>(best1);
    dist2[t * 3 + 1] = static_cast<scalar_t>(best2);
    dist2[t * 3 + 2] = static_cast<scalar_t>(best3);
    idx[t * 3 + 0]   = besti1;
    idx[t * 3 + 1]   = besti2;
    idx[t * 3 + 2]   = besti3;
}

template<typename scalar_t>
__global__ void three_interpolate_kernel(int64_t b, int64_t c, int64_t m, int64_t n,
                                         const scalar_t *__restrict__ features, // [B, C, m]
                                         const int32_t *__restrict__ idx,       // [B, n, 3]
                                         const scalar_t *__restrict__ weight,   // [B, n, 3]
                                         scalar_t *__restrict__ out)            // [B, C, n]
{
    using acc_t     = at::opmath_type<scalar_t>;
    const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if(t >= b * c * n)
    {
        return;
    }
    const int64_t ni  = t % n;
    const int64_t ci  = (t / n) % c;
    const int64_t bi  = t / (n * c);
    const int64_t w   = (bi * n + ni) * 3;
    const scalar_t *f = features + (bi * c + ci) * m;
    const acc_t v     = static_cast<acc_t>(weight[w + 0]) * static_cast<acc_t>(f[idx[w + 0]])
                  + static_cast<acc_t>(weight[w + 1]) * static_cast<acc_t>(f[idx[w + 1]])
                  + static_cast<acc_t>(weight[w + 2]) * static_cast<acc_t>(f[idx[w + 2]]);
    out[t] = static_cast<scalar_t>(v);
}

template<typename scalar_t>
__global__ void three_interpolate_grad_kernel(int64_t b, int64_t c, int64_t n, int64_t m,
                                              const scalar_t *__restrict__ grad_out, // [B, C, n]
                                              const int32_t *__restrict__ idx,       // [B, n, 3]
                                              const scalar_t *__restrict__ weight,   // [B, n, 3]
                                              scalar_t *__restrict__ grad_features)  // [B, C, m]
{
    using acc_t     = at::opmath_type<scalar_t>;
    const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if(t >= b * c * n)
    {
        return;
    }
    const int64_t ni = t % n;
    const int64_t ci = (t / n) % c;
    const int64_t bi = t / (n * c);
    const int64_t w  = (bi * n + ni) * 3;
    const acc_t g    = grad_out[t];
    scalar_t *gf     = grad_features + (bi * c + ci) * m;
    for(int j = 0; j < 3; ++j)
    {
        gpuAtomicAdd(gf + idx[w + j], static_cast<scalar_t>(g * static_cast<acc_t>(weight[w + j])));
    }
}

void check_xyz(const torch::Tensor &t, const char *name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dim() == 3 && t.size(2) == 3, name, " must have shape [B, N, 3], got ", t.sizes());
}

void check_idx(const torch::Tensor &t, const char *name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kInt, name, " must be int32");
}
} // namespace

#define PCOPS_DISPATCH(TYPE, NAME, ...) \
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, TYPE, NAME, __VA_ARGS__)

torch::Tensor ball_query_cuda(torch::Tensor new_xyz, torch::Tensor xyz, double radius, int64_t nsample)
{
    check_xyz(new_xyz, "new_xyz");
    check_xyz(xyz, "xyz");
    TORCH_CHECK(new_xyz.size(0) == xyz.size(0), "batch sizes differ");
    TORCH_CHECK(new_xyz.scalar_type() == xyz.scalar_type(), "new_xyz and xyz must have the same dtype");
    TORCH_CHECK(nsample > 0, "nsample must be positive");
    c10::cuda::CUDAGuard guard(xyz.device());
    new_xyz       = new_xyz.contiguous();
    xyz           = xyz.contiguous();
    const auto b  = xyz.size(0), n = xyz.size(1), m = new_xyz.size(1);
    auto idx      = torch::zeros({b, m, nsample}, xyz.options().dtype(torch::kInt));
    if(b * m == 0)
    {
        return idx;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    PCOPS_DISPATCH(xyz.scalar_type(), "ball_query", [&] {
        ball_query_kernel<scalar_t><<<pn_blocks(b * m), PN_THREADS, 0, stream>>>(
            b, n, m, static_cast<float>(radius * radius), nsample, new_xyz.data_ptr<scalar_t>(), xyz.data_ptr<scalar_t>(),
            idx.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return idx;
}

torch::Tensor group_points_cuda(torch::Tensor features, torch::Tensor idx)
{
    TORCH_CHECK(features.is_cuda() && features.dim() == 3, "features must be a [B, C, N] CUDA tensor");
    check_idx(idx, "idx");
    TORCH_CHECK(idx.dim() == 3 && idx.size(0) == features.size(0), "idx must be [B, M, S]");
    c10::cuda::CUDAGuard guard(features.device());
    features     = features.contiguous();
    idx          = idx.contiguous();
    const auto b = features.size(0), c = features.size(1), n = features.size(2), m = idx.size(1), s = idx.size(2);
    auto out     = torch::empty({b, c, m, s}, features.options());
    if(out.numel() == 0)
    {
        return out;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    PCOPS_DISPATCH(features.scalar_type(), "group_points", [&] {
        group_points_kernel<scalar_t><<<pn_blocks(out.numel()), PN_THREADS, 0, stream>>>(
            b, c, n, m, s, features.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), out.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor group_points_backward_cuda(torch::Tensor grad_out, torch::Tensor idx, int64_t n)
{
    check_idx(idx, "idx");
    c10::cuda::CUDAGuard guard(grad_out.device());
    grad_out     = grad_out.contiguous();
    idx          = idx.contiguous();
    const auto b = grad_out.size(0), c = grad_out.size(1), m = grad_out.size(2), s = grad_out.size(3);
    auto grad    = torch::zeros({b, c, n}, grad_out.options());
    if(grad_out.numel() == 0)
    {
        return grad;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    PCOPS_DISPATCH(grad_out.scalar_type(), "group_points_backward", [&] {
        group_points_grad_kernel<scalar_t><<<pn_blocks(grad_out.numel()), PN_THREADS, 0, stream>>>(
            b, c, n, m, s, grad_out.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), grad.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}

torch::Tensor gather_points_cuda(torch::Tensor features, torch::Tensor idx)
{
    TORCH_CHECK(features.is_cuda() && features.dim() == 3, "features must be a [B, C, N] CUDA tensor");
    check_idx(idx, "idx");
    TORCH_CHECK(idx.dim() == 2 && idx.size(0) == features.size(0), "idx must be [B, M]");
    c10::cuda::CUDAGuard guard(features.device());
    features     = features.contiguous();
    idx          = idx.contiguous();
    const auto b = features.size(0), c = features.size(1), n = features.size(2), m = idx.size(1);
    auto out     = torch::empty({b, c, m}, features.options());
    if(out.numel() == 0)
    {
        return out;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    PCOPS_DISPATCH(features.scalar_type(), "gather_points", [&] {
        gather_points_kernel<scalar_t><<<pn_blocks(out.numel()), PN_THREADS, 0, stream>>>(
            b, c, n, m, features.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), out.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor gather_points_backward_cuda(torch::Tensor grad_out, torch::Tensor idx, int64_t n)
{
    check_idx(idx, "idx");
    c10::cuda::CUDAGuard guard(grad_out.device());
    grad_out     = grad_out.contiguous();
    idx          = idx.contiguous();
    const auto b = grad_out.size(0), c = grad_out.size(1), m = grad_out.size(2);
    auto grad    = torch::zeros({b, c, n}, grad_out.options());
    if(grad_out.numel() == 0)
    {
        return grad;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    PCOPS_DISPATCH(grad_out.scalar_type(), "gather_points_backward", [&] {
        gather_points_grad_kernel<scalar_t><<<pn_blocks(grad_out.numel()), PN_THREADS, 0, stream>>>(
            b, c, n, m, grad_out.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), grad.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}

std::tuple<torch::Tensor, torch::Tensor> three_nn_cuda(torch::Tensor unknown, torch::Tensor known)
{
    check_xyz(unknown, "unknown");
    check_xyz(known, "known");
    TORCH_CHECK(unknown.size(0) == known.size(0), "batch sizes differ");
    TORCH_CHECK(unknown.scalar_type() == known.scalar_type(), "unknown and known must have the same dtype");
    c10::cuda::CUDAGuard guard(unknown.device());
    unknown      = unknown.contiguous();
    known        = known.contiguous();
    const auto b = unknown.size(0), n = unknown.size(1), m = known.size(1);
    auto dist2   = torch::empty({b, n, 3}, unknown.options());
    auto idx     = torch::empty({b, n, 3}, unknown.options().dtype(torch::kInt));
    if(b * n == 0)
    {
        return {dist2, idx};
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    PCOPS_DISPATCH(unknown.scalar_type(), "three_nn", [&] {
        three_nn_kernel<scalar_t><<<pn_blocks(b * n), PN_THREADS, 0, stream>>>(
            b, n, m, unknown.data_ptr<scalar_t>(), known.data_ptr<scalar_t>(), dist2.data_ptr<scalar_t>(), idx.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dist2, idx};
}

torch::Tensor three_interpolate_cuda(torch::Tensor features, torch::Tensor idx, torch::Tensor weight)
{
    TORCH_CHECK(features.is_cuda() && features.dim() == 3, "features must be a [B, C, m] CUDA tensor");
    check_idx(idx, "idx");
    TORCH_CHECK(idx.dim() == 3 && idx.size(2) == 3 && weight.sizes() == idx.sizes(), "idx and weight must be [B, n, 3]");
    TORCH_CHECK(weight.scalar_type() == features.scalar_type(), "weight must have the dtype of features");
    c10::cuda::CUDAGuard guard(features.device());
    features     = features.contiguous();
    idx          = idx.contiguous();
    weight       = weight.contiguous();
    const auto b = features.size(0), c = features.size(1), m = features.size(2), n = idx.size(1);
    auto out     = torch::empty({b, c, n}, features.options());
    if(out.numel() == 0)
    {
        return out;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    PCOPS_DISPATCH(features.scalar_type(), "three_interpolate", [&] {
        three_interpolate_kernel<scalar_t><<<pn_blocks(out.numel()), PN_THREADS, 0, stream>>>(
            b, c, m, n, features.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), weight.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor three_interpolate_backward_cuda(torch::Tensor grad_out, torch::Tensor idx, torch::Tensor weight, int64_t m)
{
    check_idx(idx, "idx");
    c10::cuda::CUDAGuard guard(grad_out.device());
    grad_out     = grad_out.contiguous();
    idx          = idx.contiguous();
    weight       = weight.contiguous();
    const auto b = grad_out.size(0), c = grad_out.size(1), n = grad_out.size(2);
    auto grad    = torch::zeros({b, c, m}, grad_out.options());
    if(grad_out.numel() == 0)
    {
        return grad;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    PCOPS_DISPATCH(grad_out.scalar_type(), "three_interpolate_backward", [&] {
        three_interpolate_grad_kernel<scalar_t><<<pn_blocks(grad_out.numel()), PN_THREADS, 0, stream>>>(
            b, c, n, m, grad_out.data_ptr<scalar_t>(), idx.data_ptr<int32_t>(), weight.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad;
}
} // namespace pcops
