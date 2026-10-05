// Host reference implementations used to validate the device pipeline.
// They are written independently of the library kernels (straightforward
// loops, no binning, no shared memory) so that a disagreement points at the
// pipeline rather than at a shared helper.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "tsr/device/backend.h"

namespace tsr::reference
{
struct HostSplat
{
    int2 radius;
    float2 mean;
    float depth;
    float3 conic;
    float opacity;
    float3 color;
};

// Front-to-back alpha compositing of every visible splat over a per-pixel
// surface (colour + depth), using the rasterizer's thresholds.
inline void composite(
    const std::vector<HostSplat> &splats,
    int W,
    int H,
    const std::vector<float3> &surface_color,
    const std::vector<float> &surface_depth,
    std::vector<float3> &out_color,
    std::vector<float> &out_alpha
)
{
    constexpr float kAlphaThreshold         = 1.f / 255.f;
    constexpr float kMaxAlpha               = 0.99f;
    constexpr float kTransmittanceThreshold = 1e-4f;

    std::vector<std::uint32_t> order;
    for(std::uint32_t i = 0; i < splats.size(); ++i)
    {
        if(splats[i].radius.x > 0 && splats[i].radius.y > 0)
        {
            order.push_back(i);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) { return splats[a].depth < splats[b].depth; });

    out_color.assign(static_cast<std::size_t>(W) * H, float3{0.f, 0.f, 0.f});
    out_alpha.assign(static_cast<std::size_t>(W) * H, 0.f);
    for(int y = 0; y < H; ++y)
    {
        for(int x = 0; x < W; ++x)
        {
            const std::size_t pix = static_cast<std::size_t>(y) * W + x;
            const float px = x + 0.5f, py = y + 0.5f;
            float T  = 1.f;
            float3 c = {0.f, 0.f, 0.f};
            for(std::uint32_t g : order)
            {
                const HostSplat &s = splats[g];
                if(s.depth >= surface_depth[pix])
                {
                    break;
                }
                const float dx    = s.mean.x - px;
                const float dy    = s.mean.y - py;
                const float sigma = 0.5f * (s.conic.x * dx * dx + s.conic.z * dy * dy) + s.conic.y * dx * dy;
                const float alpha = std::min(kMaxAlpha, s.opacity * std::exp(-sigma));
                if(sigma < 0.f || alpha < kAlphaThreshold)
                {
                    continue;
                }
                const float next_T = T * (1.f - alpha);
                if(next_T <= kTransmittanceThreshold)
                {
                    break;
                }
                const float vis = alpha * T;
                c.x += s.color.x * vis;
                c.y += s.color.y * vis;
                c.z += s.color.z * vis;
                T    = next_T;
            }
            out_color[pix] = float3{c.x + T * surface_color[pix].x, c.y + T * surface_color[pix].y, c.z + T * surface_color[pix].z};
            out_alpha[pix] = 1.f - T;
        }
    }
}

// Naive nearest-face rasterization in the mesh rasterizer's NDC convention
// (+x left, +y up) with perspective-correct depth. Returns the face index
// per image pixel (-1 where empty) and its camera depth.
inline void rasterize_nearest_face(
    const std::vector<float3> &face_ndc, // [F][3]
    int W,
    int H,
    std::vector<int> &pix_to_face,
    std::vector<float> &zbuf
)
{
    auto ndc_range = [](int s1, int s2) { return s1 > s2 ? 2.0f * s1 / s2 : 2.0f; };
    auto pix_ndc   = [&](int i, int s1, int s2)
    {
        const float r = ndc_range(s1, s2);
        return -r / 2.f + (r * i + r / 2.f) / s1;
    };
    auto edge = [](float px, float py, float ax, float ay, float bx, float by) { return (px - ax) * (by - ay) - (py - ay) * (bx - ax); };

    const std::size_t F = face_ndc.size() / 3;
    pix_to_face.assign(static_cast<std::size_t>(W) * H, -1);
    zbuf.assign(static_cast<std::size_t>(W) * H, -1.f);
    for(int y = 0; y < H; ++y)
    {
        for(int x = 0; x < W; ++x)
        {
            // image pixel (x, y) is raster index (W-1-x, H-1-y)
            const float px = pix_ndc(W - 1 - x, W, H);
            const float py = pix_ndc(H - 1 - y, H, W);
            float best_z   = std::numeric_limits<float>::infinity();
            int best_f     = -1;
            for(std::size_t f = 0; f < F; ++f)
            {
                const float3 a = face_ndc[3 * f], b = face_ndc[3 * f + 1], c = face_ndc[3 * f + 2];
                if(std::min({a.z, b.z, c.z}) < 1e-8f)
                {
                    continue;
                }
                const float area = edge(c.x, c.y, a.x, a.y, b.x, b.y) + 1e-8f;
                const float w0   = edge(px, py, b.x, b.y, c.x, c.y) / area;
                const float w1   = edge(px, py, c.x, c.y, a.x, a.y) / area;
                const float w2   = edge(px, py, a.x, a.y, b.x, b.y) / area;
                const float face_area = edge(a.x, a.y, b.x, b.y, c.x, c.y);
                if(std::fabs(face_area) <= 1e-8f || !(w0 > 0.f && w1 > 0.f && w2 > 0.f))
                {
                    continue;
                }
                // perspective-correct depth: 1 / z interpolates linearly
                const float t0 = w0 * b.z * c.z, t1 = a.z * w1 * c.z, t2 = a.z * b.z * w2;
                const float d  = std::max(t0 + t1 + t2, 1e-8f);
                const float z  = (t0 * a.z + t1 * b.z + t2 * c.z) / d;
                if(z < best_z || (z == best_z && static_cast<int>(f) < best_f))
                {
                    best_z = z;
                    best_f = static_cast<int>(f);
                }
            }
            const std::size_t pix = static_cast<std::size_t>(y) * W + x;
            pix_to_face[pix]      = best_f;
            zbuf[pix]             = best_f >= 0 ? best_z : -1.f;
        }
    }
}

// Multiresolution hash-grid encoding of one position, written from the
// Instant-NGP description.
inline std::vector<float> hash_encode(
    const float3 &p,
    std::uint32_t n_levels,
    std::uint32_t F,
    std::uint32_t base_resolution,
    float per_level_scale,
    const std::vector<std::uint32_t> &offsets,
    const std::vector<float> &params
)
{
    std::vector<float> out(n_levels * F, 0.f);
    const float in[3] = {p.x, p.y, p.z};
    for(std::uint32_t level = 0; level < n_levels; ++level)
    {
        const float scale         = std::exp2(level * std::log2(per_level_scale)) * base_resolution - 1.f;
        const std::uint32_t res   = static_cast<std::uint32_t>(std::ceil(scale)) + 1;
        const std::uint32_t size  = offsets[level + 1] - offsets[level];
        const std::uint64_t dense = static_cast<std::uint64_t>(res) * res * res;
        float frac[3];
        std::uint32_t cell[3];
        for(int d = 0; d < 3; ++d)
        {
            const float pos = std::fma(scale, in[d], 0.5f);
            const float fl  = std::floor(pos);
            cell[d]         = static_cast<std::uint32_t>(static_cast<int>(fl));
            frac[d]         = pos - fl;
        }
        for(int corner = 0; corner < 8; ++corner)
        {
            float w = 1.f;
            std::uint32_t c[3];
            for(int d = 0; d < 3; ++d)
            {
                const bool hi = (corner >> d) & 1;
                w            *= hi ? frac[d] : 1.f - frac[d];
                c[d]          = cell[d] + (hi ? 1u : 0u);
            }
            std::uint32_t index;
            if(res <= 0x659 && dense <= size)
            {
                index = c[0] + c[1] * res + c[2] * res * res;
            }
            else
            {
                index = c[0] * 1u ^ c[1] * 2654435761u ^ c[2] * 805459861u;
            }
            index %= size;
            for(std::uint32_t f = 0; f < F; ++f)
            {
                out[level * F + f] += w * params[(offsets[level] + index) * F + f];
            }
        }
    }
    return out;
}

} // namespace tsr::reference
