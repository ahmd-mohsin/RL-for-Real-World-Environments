// Tessera - exposure and tone mapping of the linear composite.
#include <cmath>
#include <vector>

#include "tsr/math/vec.h"
#include "tsr/post/tonemap.h"
#include "tsr/prims/prims.h"

namespace tsr
{
namespace
{
constexpr float kLumEpsilon = 1e-4f;

TSR_HDI float luminance(const float3 &c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

TSR_HDI float srgb_encode(float x)
{
    x = fminf(fmaxf(x, 0.f), 1.f);
    return x <= 0.0031308f ? 12.92f * x : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}

TSR_HDI unsigned char to_unorm8(float x) { return static_cast<unsigned char>(fminf(fmaxf(x, 0.f), 1.f) * 255.f + 0.5f); }

__global__ void log_luminance_kernel(std::uint32_t n, const float3 *__restrict__ hdr, float *__restrict__ log_lum)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        log_lum[i] = __logf(kLumEpsilon + luminance(hdr[i]));
    }
}

__global__ void luminance_bins_kernel(std::uint32_t n, const float3 *__restrict__ hdr, float exposure, std::uint32_t *__restrict__ bins)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
    {
        return;
    }
    const float L = luminance(hdr[i]) * exposure;
    if(L <= 0.f)
    {
        bins[i] = 0;
        return;
    }
    const float t = (__log2f(L) - kHistogramMinLog2) / (kHistogramMaxLog2 - kHistogramMinLog2);
    bins[i]       = static_cast<std::uint32_t>(fminf(fmaxf(t * 256.f, 0.f), 255.f));
}

__global__ void tonemap_kernel(
    std::uint32_t n,
    const float3 *__restrict__ hdr,
    bool enabled,
    float exposure,
    float inv_white_sq,
    uchar4 *__restrict__ out
)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
    {
        return;
    }
    float3 c = hdr[i];
    if(enabled)
    {
        c             = c * exposure;
        const float L = luminance(c);
        // extended Reinhard: L (1 + L / Lw^2) / (1 + L)
        const float Ld = L * (1.f + L * inv_white_sq) / (1.f + L);
        c              = L > 0.f ? c * (Ld / L) : make_float3(0.f, 0.f, 0.f);
    }
    out[i] = make_uchar4(to_unorm8(srgb_encode(c.x)), to_unorm8(srgb_encode(c.y)), to_unorm8(srgb_encode(c.z)), 255);
}
} // namespace

float linear_to_srgb(float x) { return srgb_encode(x); }

float srgb_to_linear(float x)
{
    x = std::fmin(std::fmax(x, 0.f), 1.f);
    return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

void tonemap(
    const float3 *hdr,
    int width,
    int height,
    const TonemapSettings &settings,
    uchar4 *out,
    TonemapStats *stats,
    Arena &arena,
    cudaStream_t stream
)
{
    const auto n = static_cast<std::uint32_t>(width * height);
    if(n == 0)
    {
        return;
    }
    TonemapStats local;
    TonemapStats &s = stats != nullptr ? *stats : local;
    s               = TonemapStats{};

    if(!settings.enabled)
    {
        launch_linear(tonemap_kernel, n, stream, n, hdr, false, 1.f, 1.f, out);
        return;
    }
    TSR_CHECK(settings.white_percentile > 0.f && settings.white_percentile <= 1.f, "white percentile must be in (0, 1]");

    // 1. exposure from the log-average luminance
    float *log_lum = arena.allocate<float>(n);
    float *sum     = arena.allocate<float>(1);
    launch_linear(log_luminance_kernel, n, stream, n, hdr, log_lum);
    prims::reduce(log_lum, n, sum, prims::ReduceOp::Sum, arena, stream);
    float host_sum = 0.f;
    TSR_CUDA_CHECK(cudaMemcpyAsync(&host_sum, sum, sizeof(float), cudaMemcpyDeviceToHost, stream));
    TSR_CUDA_CHECK(cudaStreamSynchronize(stream));
    s.log_average_luminance = std::exp(host_sum / static_cast<float>(n));
    s.exposure              = settings.key / s.log_average_luminance;

    // 2. white point from the luminance histogram
    std::uint32_t *bins = arena.allocate<std::uint32_t>(n);
    std::uint32_t *hist = arena.allocate<std::uint32_t>(prims::kHistogramBins);
    launch_linear(luminance_bins_kernel, n, stream, n, hdr, s.exposure, bins);
    prims::histogram256(bins, n, hist, arena, stream);
    TSR_CUDA_CHECK(cudaMemcpyAsync(s.histogram.data(), hist, sizeof(std::uint32_t) * prims::kHistogramBins, cudaMemcpyDeviceToHost, stream));
    prims::inclusive_scan(hist, hist, prims::kHistogramBins, arena, stream);
    std::vector<std::uint32_t> cdf(prims::kHistogramBins);
    TSR_CUDA_CHECK(cudaMemcpyAsync(cdf.data(), hist, sizeof(std::uint32_t) * prims::kHistogramBins, cudaMemcpyDeviceToHost, stream));
    TSR_CUDA_CHECK(cudaStreamSynchronize(stream));

    const double target = settings.white_percentile * static_cast<double>(n);
    std::uint32_t white_bin = prims::kHistogramBins - 1;
    for(std::uint32_t b = 0; b < prims::kHistogramBins; ++b)
    {
        if(static_cast<double>(cdf[b]) >= target)
        {
            white_bin = b;
            break;
        }
    }
    // upper edge of the bin
    const float log2_white = kHistogramMinLog2 + (white_bin + 1) * (kHistogramMaxLog2 - kHistogramMinLog2) / 256.f;
    s.white_point          = std::fmax(std::exp2(log2_white), 1e-3f);

    // 3. tone map
    launch_linear(tonemap_kernel, n, stream, n, hdr, true, s.exposure, 1.f / (s.white_point * s.white_point), out);
}
} // namespace tsr
