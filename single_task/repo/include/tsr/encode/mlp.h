// Tessera - small fully connected network evaluated with tiled GEMMs.
#pragma once

#include <cstdint>
#include <vector>

#include "tsr/device/memory.h"
#include "tsr/scene/scene.h"

namespace tsr
{
enum class Activation : std::uint32_t
{
    None    = 0,
    ReLU    = 1,
    Sigmoid = 2,
};

// C[M x N] = act(A[M x K] * B[K x N] + bias[N]); all row-major. bias may be null.
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
);

class Mlp
{
public:
    // in -> [hidden_width] x hidden_layers (ReLU) -> out (sigmoid). Weights use
    // Xavier-uniform initialization from a PCG32 seeded with `seed`; biases
    // start at zero.
    void initialize(std::uint32_t in_dim, std::uint32_t out_dim, const MlpConfig &config, std::uint32_t seed, cudaStream_t stream);

    // input [M x in_dim] -> output [M x out_dim]
    void forward(const float *input, std::uint32_t M, float *output, Arena &arena, cudaStream_t stream) const;

    [[nodiscard]] std::uint32_t in_dim() const { return in_dim_; }
    [[nodiscard]] std::uint32_t out_dim() const { return out_dim_; }
    [[nodiscard]] std::size_t n_layers() const { return weights_.size(); }
    [[nodiscard]] std::vector<float> weights_to_host(std::size_t layer) const { return weights_[layer].to_host(); }
    [[nodiscard]] std::vector<float> biases_to_host(std::size_t layer) const { return biases_[layer].to_host(); }
    [[nodiscard]] std::uint32_t layer_in(std::size_t layer) const { return dims_[layer]; }
    [[nodiscard]] std::uint32_t layer_out(std::size_t layer) const { return dims_[layer + 1]; }

private:
    std::uint32_t in_dim_  = 0;
    std::uint32_t out_dim_ = 0;
    std::vector<std::uint32_t> dims_;
    std::vector<DeviceBuffer<float>> weights_; // [layer] row-major [in x out]
    std::vector<DeviceBuffer<float>> biases_;
};

} // namespace tsr
