// Tessera - small fully connected network evaluated with tiled GEMMs.
//
// The GEMM kernel is the shared-memory tiling scheme of MatrixMulCUDA from the
// cuda-samples "matrixMul" sample (BSD-3-Clause), extended with bounds checks
// for arbitrary shapes and a fused bias + activation epilogue.
#include <cmath>

#include "tsr/encode/mlp.h"
#include "tsr/util/random.h"

namespace tsr
{
namespace
{
constexpr int kTile = 16;

TSR_DI float activate(float x, Activation act)
{
    switch(act)
    {
    case Activation::ReLU: return fmaxf(x, 0.f);
    case Activation::Sigmoid: return 1.f / (1.f + __expf(-x));
    case Activation::None: break;
    }
    return x;
}

template<int BLOCK_SIZE>
__global__ void gemm_bias_act_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    const float *__restrict__ bias,
    float *__restrict__ C,
    int M,
    int N,
    int K,
    Activation act
)
{
    // Rows map to grid.x (up to 2^31 - 1 blocks), columns to grid.y: M is the
    // number of shaded pixels and easily exceeds the 65535 limit of grid.y.
    const int by = static_cast<int>(blockIdx.x);
    const int bx = static_cast<int>(blockIdx.y);
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    const int row = by * BLOCK_SIZE + ty;
    const int col = bx * BLOCK_SIZE + tx;

    // Csub accumulates the element of the block sub-matrix computed by this thread.
    float c_sub = 0.f;

    // Loop over the BLOCK_SIZE-wide slabs of A's columns / B's rows.
    for(int t = 0; t < K; t += BLOCK_SIZE)
    {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Each thread loads one element of each sub-matrix, padding with
        // zeros outside the matrices.
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.f;

        // Make sure the matrices are loaded before computing.
        __syncthreads();

        TSR_UNROLL
        for(int k = 0; k < BLOCK_SIZE; ++k)
        {
            c_sub += As[ty][k] * Bs[k][tx];
        }

        // Make sure the preceding computation is done before loading two new
        // sub-matrices in the next iteration.
        __syncthreads();
    }

    if(row < M && col < N)
    {
        C[row * N + col] = activate(c_sub + (bias != nullptr ? bias[col] : 0.f), act);
    }
}
} // namespace

void gemm_bias_act(
    const float *A,
    const float *B,
    const float *bias,
    float *C,
    std::uint32_t M,
    std::uint32_t N,
    std::uint32_t K,
    Activation act,
    cudaStream_t stream
)
{
    if(M == 0 || N == 0)
    {
        return;
    }
    const dim3 grid(div_round_up(M, static_cast<std::uint32_t>(kTile)), div_round_up(N, static_cast<std::uint32_t>(kTile)), 1);
    TSR_CHECK(grid.y <= 65535u, "GEMM with " << N << " columns exceeds the grid limit");
    launch(
        gemm_bias_act_kernel<kTile>, LaunchConfig{grid, dim3(kTile, kTile), 0, stream}, A, B, bias, C, static_cast<int>(M),
        static_cast<int>(N), static_cast<int>(K), act
    );
}

void Mlp::initialize(std::uint32_t in_dim, std::uint32_t out_dim, const MlpConfig &config, std::uint32_t seed, cudaStream_t stream)
{
    TSR_CHECK(in_dim > 0 && out_dim > 0 && config.hidden_width > 0, "MLP dimensions must be positive");
    in_dim_  = in_dim;
    out_dim_ = out_dim;
    dims_.clear();
    dims_.push_back(in_dim);
    for(std::uint32_t l = 0; l < config.hidden_layers; ++l)
    {
        dims_.push_back(config.hidden_width);
    }
    dims_.push_back(out_dim);

    Pcg32 rng(seed, 0x9e3779b97f4a7c15ULL);
    weights_.clear();
    biases_.clear();
    weights_.resize(dims_.size() - 1);
    biases_.resize(dims_.size() - 1);
    for(std::size_t l = 0; l + 1 < dims_.size(); ++l)
    {
        const std::uint32_t fan_in  = dims_[l];
        const std::uint32_t fan_out = dims_[l + 1];
        const float limit           = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
        std::vector<float> w(static_cast<std::size_t>(fan_in) * fan_out);
        for(float &v : w)
        {
            v = rng.uniform(-limit, limit);
        }
        weights_[l].upload(w, stream);
        biases_[l].upload(std::vector<float>(fan_out, 0.f), stream);
    }
}

void Mlp::forward(const float *input, std::uint32_t M, float *output, Arena &arena, cudaStream_t stream) const
{
    TSR_CHECK(!weights_.empty(), "MLP is not initialized");
    if(M == 0)
    {
        return;
    }
    const float *src = input;
    for(std::size_t l = 0; l < weights_.size(); ++l)
    {
        const bool last      = l + 1 == weights_.size();
        float *dst           = last ? output : arena.allocate<float>(static_cast<std::size_t>(M) * dims_[l + 1]);
        const Activation act = last ? Activation::Sigmoid : Activation::ReLU;
        gemm_bias_act(src, weights_[l].data(), biases_[l].data(), dst, M, dims_[l + 1], dims_[l], act, stream);
        src = dst;
    }
}
} // namespace tsr
