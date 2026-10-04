// Tessera - binning of screen-space splats into image tiles.
//
// Every (splat, tile) overlap becomes an intersection with a 64-bit key
//
//     [ unused | tile id (tile_bits) | depth as IEEE-754 bits (32) ]
//
// Sorting the keys groups intersections by tile and orders each group front
// to back (positive floats compare like their bit patterns). The rasterizer
// then walks tile_offsets[t] .. tile_offsets[t + 1] of the sorted list.
#pragma once

#include <cstdint>

#include "tsr/device/memory.h"
#include "tsr/geom/camera.h"
#include "tsr/geom/projection.h"

namespace tsr
{
struct BinningSettings
{
    std::uint32_t tile_size = 16;
    // true: count only tiles the splat's ellipse actually touches (SnugBox +
    // AccuTile); false: every tile of the axis-aligned radius box.
    bool accurate = true;
};

struct TileBinning
{
    std::uint32_t tile_size = 0;
    std::uint32_t tiles_x   = 0;
    std::uint32_t tiles_y   = 0;
    std::uint32_t tile_bits = 0; // bits needed to encode a tile id
    std::uint32_t n_isects  = 0;

    DeviceBuffer<std::int64_t> tiles_per_splat; // after binning: inclusive prefix sum
    DeviceBuffer<std::uint64_t> keys;           // unsorted intersection keys
    DeviceBuffer<std::int32_t> splat_ids;       // unsorted splat index per intersection
    DeviceBuffer<std::uint64_t> sorted_keys;
    DeviceBuffer<std::int32_t> sorted_splat_ids;
    DeviceBuffer<std::int32_t> tile_offsets;    // [n_tiles + 1]; last entry == n_isects

    [[nodiscard]] std::uint32_t n_tiles() const { return tiles_x * tiles_y; }
};

// Number of bits needed to represent tile ids 0 .. n_tiles - 1.
std::uint32_t tile_id_bits(std::uint32_t n_tiles);

void bin_splats(
    const ProjectedSplats &splats,
    const Camera &camera,
    const BinningSettings &settings,
    TileBinning &out,
    Arena &arena,
    cudaStream_t stream
);

} // namespace tsr
