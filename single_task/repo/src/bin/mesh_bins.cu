// Tessera - coarse binning of mesh faces.
//
// Derived from PyTorch3D's coarse rasterizer (rasterize_coarse.cu and
// bitmask.cuh, BSD-3-Clause, Copyright Meta Platforms, Inc. and affiliates),
// reduced to a single image. Each block takes a chunk of faces, marks every
// (bin, face) overlap in a shared-memory bit mask, then reserves space in each
// bin with one global atomic per bin and writes the chunk's faces out.
#include <algorithm>

#include "tsr/bin/mesh_bins.h"
#include "tsr/geom/mesh_ndc.cuh"

namespace tsr
{
namespace
{
constexpr float kFaceEpsilon  = 1e-8f;
constexpr int kChunkSize      = 512;
constexpr int kCoarseThreads  = 512;
constexpr int kCoarseBlocks   = 64;
constexpr int kMaxBinsPerSide = 22 * 4;

// A bool array of shape (H, W, N) packed into 32-bit words in shared memory.
class BitMask
{
public:
    TSR_DI BitMask(unsigned *data, int H, int W, int N) : data_(data), H_(H), W_(W), D_(N / 32) {}

    // Use all threads in the block to clear every bit.
    TSR_DI void block_clear()
    {
        for(int i = static_cast<int>(threadIdx.x); i < H_ * W_ * D_; i += static_cast<int>(blockDim.x))
        {
            data_[i] = 0;
        }
        __syncthreads();
    }

    TSR_DI void set(int y, int x, int d) { atomicOr(data_ + elem(y, x, d), 1u << (d % 32)); }

    TSR_DI bool get(int y, int x, int d) const { return (data_[elem(y, x, d)] >> (d % 32)) & 1u; }

    // Number of set bits in row (y, x, :).
    TSR_DI int count(int y, int x) const
    {
        int total = 0;
        for(int i = 0; i < D_; ++i)
        {
            total += __popc(data_[y * W_ * D_ + x * D_ + i]);
        }
        return total;
    }

private:
    TSR_DI int elem(int y, int x, int d) const { return y * W_ * D_ + x * D_ + d / 32; }

    unsigned *data_;
    int H_, W_, D_;
};

__global__ void triangle_bounding_box_kernel(
    const float3 *__restrict__ face_verts, // [F][3]
    int F,
    float blur_radius,
    float4 *__restrict__ bboxes, // [F] (xmin, xmax, ymin, ymax)
    std::uint8_t *__restrict__ skip_face
)
{
    const int f = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if(f >= F)
    {
        return;
    }
    const float sqrt_radius = sqrtf(blur_radius);
    const float3 v0         = face_verts[3 * f + 0];
    const float3 v1         = face_verts[3 * f + 1];
    const float3 v2         = face_verts[3 * f + 2];
    const float xmin        = fminf(v0.x, fminf(v1.x, v2.x)) - sqrt_radius;
    const float xmax        = fmaxf(v0.x, fmaxf(v1.x, v2.x)) + sqrt_radius;
    const float ymin        = fminf(v0.y, fminf(v1.y, v2.y)) - sqrt_radius;
    const float ymax        = fmaxf(v0.y, fmaxf(v1.y, v2.y)) + sqrt_radius;
    const float zmin        = fminf(v0.z, fminf(v1.z, v2.z));
    bboxes[f]               = make_float4(xmin, xmax, ymin, ymax);
    skip_face[f]            = zmin < kFaceEpsilon ? 1 : 0; // behind or touching the camera plane
}

__global__ void __launch_bounds__(kCoarseThreads) rasterize_coarse_kernel(
    const float4 *__restrict__ bboxes,
    const std::uint8_t *__restrict__ should_skip,
    int E,
    int H,
    int W,
    int bin_size,
    int max_elem_per_bin,
    std::int32_t *__restrict__ elems_per_bin, // [BH][BW]
    std::int32_t *__restrict__ bin_elems,     // [BH][BW][M]
    int *__restrict__ overflow_flag
)
{
    TSR_DYNAMIC_SMEM(unsigned, sbuf);
    const int M          = max_elem_per_bin;
    const int num_bins_x = 1 + (W - 1) / bin_size;
    const int num_bins_y = 1 + (H - 1) / bin_size;

    // Size of half a pixel in NDC units.
    const float half_pix_x = geom::non_square_ndc_range(W, H) / 2.0f / W;
    const float half_pix_y = geom::non_square_ndc_range(H, W) / 2.0f / H;

    BitMask binmask(sbuf, num_bins_y, num_bins_x, kChunkSize);
    const int num_chunks = 1 + (E - 1) / kChunkSize;

    for(int chunk = static_cast<int>(blockIdx.x); chunk < num_chunks; chunk += static_cast<int>(gridDim.x))
    {
        const int chunk_start = chunk * kChunkSize;
        binmask.block_clear();

        // One face per thread within the chunk.
        for(int e = static_cast<int>(threadIdx.x); e < kChunkSize; e += static_cast<int>(blockDim.x))
        {
            const int e_idx = chunk_start + e;
            if(e_idx >= E || should_skip[e_idx])
            {
                continue;
            }
            const float4 bb = bboxes[e_idx];
            for(int by = 0; by < num_bins_y; ++by)
            {
                // pix_to_non_square_ndc gives pixel centres; widen by half a
                // pixel for the true extent of the bin.
                const float bin_y_min = geom::pix_to_non_square_ndc(by * bin_size, H, W) - half_pix_y;
                const float bin_y_max = geom::pix_to_non_square_ndc((by + 1) * bin_size - 1, H, W) + half_pix_y;
                const bool y_overlap  = (bb.z <= bin_y_max) && (bin_y_min < bb.w);
                for(int bx = 0; bx < num_bins_x; ++bx)
                {
                    const float bin_x_max = geom::pix_to_non_square_ndc((bx + 1) * bin_size - 1, W, H) + half_pix_x;
                    const float bin_x_min = geom::pix_to_non_square_ndc(bx * bin_size, W, H) - half_pix_x;
                    const bool x_overlap  = (bb.x <= bin_x_max) && (bin_x_min < bb.y);
                    if(y_overlap && x_overlap)
                    {
                        binmask.set(by, bx, e);
                    }
                }
            }
        }
        __syncthreads();

        // One bin per thread: reserve space for this chunk's faces with a
        // single global atomic, then write them in face order.
        for(int byx = static_cast<int>(threadIdx.x); byx < num_bins_y * num_bins_x; byx += static_cast<int>(blockDim.x))
        {
            const int by    = byx / num_bins_x;
            const int bx    = byx % num_bins_x;
            const int count = binmask.count(by, bx);
            if(count == 0)
            {
                continue;
            }
            const int start = atomicAdd(elems_per_bin + by * num_bins_x + bx, count);
            if(start + count > M)
            {
                atomicExch(overflow_flag, 1);
                continue;
            }
            int next_idx = by * num_bins_x * M + bx * M + start;
            for(int e = 0; e < kChunkSize; ++e)
            {
                if(binmask.get(by, bx, e))
                {
                    bin_elems[next_idx] = chunk_start + e;
                    ++next_idx;
                }
            }
        }
        __syncthreads();
    }
}
} // namespace

void bin_mesh_faces(
    const ViewFaces &faces,
    int width,
    int height,
    const MeshBinningSettings &settings,
    MeshBins &out,
    Arena &arena,
    cudaStream_t stream
)
{
    TSR_CHECK(settings.bin_size > 0, "bin size must be positive");
    out.bin_size          = settings.bin_size;
    out.bins_x            = 1 + (width - 1) / settings.bin_size;
    out.bins_y            = 1 + (height - 1) / settings.bin_size;
    out.max_faces_per_bin = settings.max_faces_per_bin;
    out.overflowed        = false;
    TSR_CHECK(out.bins_x < kMaxBinsPerSide && out.bins_y < kMaxBinsPerSide,
              "too many bins (" << out.bins_x << " x " << out.bins_y << "); increase bin_size");

    const int n_bins = out.bins_x * out.bins_y;
    out.faces_per_bin.resize(static_cast<std::size_t>(n_bins));
    out.bin_faces.resize(static_cast<std::size_t>(n_bins) * settings.max_faces_per_bin);
    out.faces_per_bin.zero(stream);
    TSR_CUDA_CHECK(cudaMemsetAsync(out.bin_faces.data(), 0xff, out.bin_faces.bytes(), stream)); // -1 sentinels
    const int F = static_cast<int>(faces.n_faces);
    if(F == 0)
    {
        return;
    }

    float4 *bboxes      = arena.allocate<float4>(F);
    std::uint8_t *skip  = arena.allocate<std::uint8_t>(F);
    int *overflow_flag  = arena.allocate<int>(1);
    TSR_CUDA_CHECK(cudaMemsetAsync(overflow_flag, 0, sizeof(int), stream));
    launch_linear(triangle_bounding_box_kernel, static_cast<std::uint32_t>(F), stream, faces.ndc.data(), F, settings.blur_radius, bboxes, skip);

    const std::size_t smem = static_cast<std::size_t>(n_bins) * kChunkSize / 8;
    const int blocks       = std::min(kCoarseBlocks, 1 + (F - 1) / kChunkSize);
    launch(
        rasterize_coarse_kernel,
        LaunchConfig{dim3(blocks), dim3(kCoarseThreads), smem, stream},
        bboxes, skip, F, height, width, settings.bin_size, settings.max_faces_per_bin, out.faces_per_bin.data(),
        out.bin_faces.data(), overflow_flag
    );

    int overflow = 0;
    TSR_CUDA_CHECK(cudaMemcpyAsync(&overflow, overflow_flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
    TSR_CUDA_CHECK(cudaStreamSynchronize(stream));
    out.overflowed = overflow != 0;
}
} // namespace tsr
