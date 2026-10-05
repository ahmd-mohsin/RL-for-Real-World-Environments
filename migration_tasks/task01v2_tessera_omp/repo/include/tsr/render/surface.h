// Tessera - shading of rasterized mesh fragments with a neural material.
//
// Produces the opaque "surface layer" of a view: for every pixel the colour
// and camera depth of the nearest mesh surface, or the background colour and
// +inf depth where no mesh is visible. Splats are composited in front of it.
#pragma once

#include <cstdint>

#include "tsr/device/memory.h"
#include "tsr/encode/hashgrid.h"
#include "tsr/encode/mlp.h"
#include "tsr/geom/camera.h"
#include "tsr/geom/mesh.h"
#include "tsr/raster/mesh_raster.h"
#include "tsr/scene/scene.h"

namespace tsr
{
class NeuralMaterialModel
{
public:
    void initialize(const NeuralMaterial &desc, cudaStream_t stream);

    [[nodiscard]] const NeuralMaterial &desc() const { return desc_; }
    [[nodiscard]] const HashGrid &grid() const { return grid_; }
    [[nodiscard]] const Mlp &mlp() const { return mlp_; }

private:
    NeuralMaterial desc_{};
    HashGrid grid_;
    Mlp mlp_;
};

struct SurfaceLayer
{
    int width  = 0;
    int height = 0;
    DeviceBuffer<float3> color; // [H][W]
    DeviceBuffer<float> depth;  // [H][W], +inf where no surface
    std::uint32_t covered_pixels = 0;
};

void shade_surface(
    const NeuralMaterialModel &material,
    const MeshFragments &fragments,
    const ViewFaces &faces,
    const Camera &camera,
    float3 background,
    SurfaceLayer &out,
    Arena &arena,
    cudaStream_t stream
);

// Fills `out` with the background only (scenes without meshes).
void clear_surface(int width, int height, float3 background, SurfaceLayer &out, cudaStream_t stream);

} // namespace tsr
