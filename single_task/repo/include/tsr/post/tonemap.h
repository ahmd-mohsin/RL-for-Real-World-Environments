// Tessera - exposure and tone mapping of the linear composite.
//
// 1. log-average luminance (prims::reduce) sets the exposure so the scene's
//    key value maps to `key`
// 2. a 256-bin histogram of log2 exposed luminance (prims::histogram256) and
//    its prefix sum (prims::inclusive_scan) give the white point as a
//    luminance percentile
// 3. extended Reinhard maps luminance into [0, 1]; colours are encoded as sRGB
#pragma once

#include <array>
#include <cstdint>

#include "tsr/device/memory.h"

namespace tsr
{
struct TonemapSettings
{
    bool enabled           = true;  // false: clamp linear values and encode sRGB
    float key              = 0.18f; // target for the log-average luminance
    float white_percentile = 0.99f; // fraction of pixels darker than the white point
};

struct TonemapStats
{
    float log_average_luminance = 0.f;
    float exposure              = 1.f;
    float white_point           = 1.f; // exposed luminance mapped to 1
    std::array<std::uint32_t, 256> histogram{};
};

// Log2 luminance range covered by the histogram.
constexpr float kHistogramMinLog2 = -10.f;
constexpr float kHistogramMaxLog2 = 6.f;

void tonemap(
    const float3 *hdr,
    int width,
    int height,
    const TonemapSettings &settings,
    uchar4 *out,
    TonemapStats *stats,
    Arena &arena,
    cudaStream_t stream
);

// Host helpers shared with tests.
float linear_to_srgb(float x);
float srgb_to_linear(float x);

} // namespace tsr
