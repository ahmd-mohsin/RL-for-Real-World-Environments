// Tessera - pixel <-> non-square NDC conversions used by the mesh rasterizer.
//
// NonSquareNdcRange and PixToNonSquareNdc are derived from PyTorch3D
// (BSD-3-Clause, Copyright Meta Platforms, Inc. and affiliates), see
// third_party/LICENSES/pytorch3d.txt.
#pragma once

#include "tsr/device/backend.h"

namespace tsr::geom
{
// The shorter image side spans an NDC range of 2; the longer side is scaled
// by the aspect ratio. S1 is the dimension the range is computed for.
TSR_HDI float non_square_ndc_range(int S1, int S2)
{
    float range = 2.0f;
    if(S1 > S2)
    {
        range = (S1 * range) / S2;
    }
    return range;
}

// NDC coordinate of the centre of raster index i along a dimension of size
// S1. Raster indices grow with NDC, i.e. towards +x (left) / +y (up).
TSR_HDI float pix_to_non_square_ndc(int i, int S1, int S2)
{
    const float range  = non_square_ndc_range(S1, S2);
    const float offset = range / 2.0f;
    return -offset + (range * i + offset) / S1;
}

// Continuous pixel coordinate (origin top-left, +x right) -> NDC x.
// Pixel centre u = c + 0.5 maps to pix_to_non_square_ndc(W - 1 - c, W, H).
TSR_HDI float pixel_x_to_ndc(float u, int W, int H)
{
    return non_square_ndc_range(W, H) * (0.5f - u / static_cast<float>(W));
}

// Continuous pixel coordinate (origin top-left, +y down) -> NDC y.
TSR_HDI float pixel_y_to_ndc(float v, int H, int W)
{
    return non_square_ndc_range(H, W) * (0.5f - v / static_cast<float>(H));
}

} // namespace tsr::geom
