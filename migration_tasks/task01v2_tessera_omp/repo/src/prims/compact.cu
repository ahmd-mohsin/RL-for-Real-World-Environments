// Tessera - stream compaction of index sets.
#include "tsr/prims/prims.h"

namespace tsr::prims
{
namespace
{
__global__ void flags_to_counts_kernel(std::uint64_t n, const std::uint8_t *__restrict__ flags, std::uint32_t *__restrict__ counts)
{
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n)
    {
        counts[i] = flags[i] != 0 ? 1u : 0u;
    }
}

__global__ void scatter_indices_kernel(
    std::uint64_t n,
    const std::uint8_t *__restrict__ flags,
    const std::uint32_t *__restrict__ positions,
    std::uint32_t *__restrict__ out
)
{
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i < n && flags[i] != 0)
    {
        out[positions[i]] = static_cast<std::uint32_t>(i);
    }
}
} // namespace

std::uint32_t compact_indices(const std::uint8_t *flags, std::size_t n, std::uint32_t *out_indices, Arena &arena, cudaStream_t stream)
{
    if(n == 0)
    {
        return 0;
    }
    TSR_CHECK(n <= 0xffffffffu, "compact_indices supports at most 2^32-1 elements");
    std::uint32_t *positions = arena.allocate<std::uint32_t>(n);
    const auto count         = static_cast<std::uint32_t>(n);
    launch_linear(flags_to_counts_kernel, count, stream, static_cast<std::uint64_t>(n), flags, positions);
    const std::uint32_t total = exclusive_scan_total(positions, positions, n, arena, stream);
    launch_linear(scatter_indices_kernel, count, stream, static_cast<std::uint64_t>(n), flags, positions, out_indices);
    return total;
}
} // namespace tsr::prims
