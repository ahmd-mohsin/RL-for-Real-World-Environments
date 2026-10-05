// Tests for the hash-grid encoding, GEMM and MLP.
#include <cmath>
#include <vector>

#include "doctest.h"
#include "reference.h"
#include "tsr/encode/hashgrid.h"
#include "tsr/encode/mlp.h"
#include "tsr/util/random.h"

using namespace tsr;

TEST_CASE("encode: offset table follows the dense/hashed split")
{
    HashGrid grid;
    HashGridConfig c;
    c.n_levels          = 6;
    c.features_per_level = 2;
    c.log2_hashmap_size = 10; // 1024 entries
    c.base_resolution   = 4;
    c.per_level_scale   = 2.0f;
    grid.initialize(c, 3, 0.1f, nullptr);
    const auto &off = grid.offsets();
    REQUIRE(off.size() == 7);
    // level 0: scale 3 -> resolution 4 -> 64 dense entries
    CHECK(off[1] - off[0] == 64);
    // level 1: scale 7 -> resolution 8 -> 512 dense entries
    CHECK(off[2] - off[1] == 512);
    // level 2 onwards are capped at the hashmap size
    CHECK(off[3] - off[2] == 1024);
    CHECK(grid.n_params() == off.back() * 2);
}

TEST_CASE("encode: device encoding matches the host reference")
{
    HashGrid grid;
    HashGridConfig c;
    c.n_levels           = 8;
    c.features_per_level = 2;
    c.log2_hashmap_size  = 12;
    c.base_resolution    = 4;
    c.per_level_scale    = 1.6f;
    grid.initialize(c, 11, 1.0f, nullptr);

    Pcg32 rng(5);
    std::vector<float3> pos(300);
    for(auto &p : pos)
    {
        p = make_float3(rng.next_float(), rng.next_float(), rng.next_float());
    }
    pos[0] = make_float3(0.f, 0.f, 0.f);
    pos[1] = make_float3(1.f, 1.f, 1.f);
    DeviceBuffer<float3> dpos;
    dpos.upload(pos);
    DeviceBuffer<float> out(pos.size() * grid.n_output_dims());
    encode_positions(grid, dpos.data(), static_cast<std::uint32_t>(pos.size()), out.data(), GridInterpolation::Linear, nullptr);
    const auto got    = out.to_host();
    const auto params = grid.params_to_host();
    bool ok           = true;
    for(std::size_t i = 0; i < pos.size(); ++i)
    {
        const auto ref = reference::hash_encode(pos[i], c.n_levels, c.features_per_level, c.base_resolution, c.per_level_scale, grid.offsets(), params);
        for(std::size_t k = 0; k < ref.size(); ++k)
        {
            ok = ok && std::fabs(ref[k] - got[i * grid.n_output_dims() + k]) < 1e-4f;
        }
    }
    CHECK(ok);
}

TEST_CASE("encode: GEMM with bias and activation on ragged shapes")
{
    // Named copies, not structured bindings: clang with -fopenmp cannot
    // capture structured bindings (CAPTURE is a lambda).
    for(const auto &shape : {std::tuple{1u, 1u, 1u}, std::tuple{17u, 5u, 33u}, std::tuple{100u, 16u, 16u}, std::tuple{3u, 40u, 7u}})
    {
        const std::uint32_t M = std::get<0>(shape), N = std::get<1>(shape), K = std::get<2>(shape);
        CAPTURE(M);
        CAPTURE(N);
        CAPTURE(K);
        Pcg32 rng(M * 131 + N * 7 + K);
        std::vector<float> A(M * K), B(K * N), bias(N);
        for(auto &v : A)
        {
            v = rng.uniform(-1.f, 1.f);
        }
        for(auto &v : B)
        {
            v = rng.uniform(-1.f, 1.f);
        }
        for(auto &v : bias)
        {
            v = rng.uniform(-0.5f, 0.5f);
        }
        DeviceBuffer<float> dA, dB, db, dC(M * N);
        dA.upload(A);
        dB.upload(B);
        db.upload(bias);
        gemm_bias_act(dA.data(), dB.data(), db.data(), dC.data(), M, N, K, Activation::ReLU, nullptr);
        const auto C = dC.to_host();
        bool ok      = true;
        for(std::uint32_t i = 0; i < M; ++i)
        {
            for(std::uint32_t j = 0; j < N; ++j)
            {
                float acc = bias[j];
                for(std::uint32_t k = 0; k < K; ++k)
                {
                    acc += A[i * K + k] * B[k * N + j];
                }
                ok = ok && std::fabs(std::max(acc, 0.f) - C[i * N + j]) < 1e-4f;
            }
        }
        CHECK(ok);
    }
}

TEST_CASE("encode: MLP forward equals a host evaluation of its weights")
{
    Mlp mlp;
    MlpConfig cfg;
    cfg.hidden_width  = 12;
    cfg.hidden_layers = 2;
    mlp.initialize(10, 3, cfg, 99, nullptr);
    REQUIRE(mlp.n_layers() == 3);

    const std::uint32_t M = 37;
    Pcg32 rng(1);
    std::vector<float> x(M * 10);
    for(auto &v : x)
    {
        v = rng.uniform(-1.f, 1.f);
    }
    DeviceBuffer<float> dx, dy(M * 3);
    dx.upload(x);
    Arena arena;
    mlp.forward(dx.data(), M, dy.data(), arena, nullptr);
    const auto y = dy.to_host();

    std::vector<float> act = x;
    for(std::size_t l = 0; l < mlp.n_layers(); ++l)
    {
        const auto W  = mlp.weights_to_host(l);
        const auto b  = mlp.biases_to_host(l);
        const auto in = mlp.layer_in(l), out = mlp.layer_out(l);
        std::vector<float> next(M * out);
        for(std::uint32_t i = 0; i < M; ++i)
        {
            for(std::uint32_t j = 0; j < out; ++j)
            {
                float acc = b[j];
                for(std::uint32_t k = 0; k < in; ++k)
                {
                    acc += act[i * in + k] * W[k * out + j];
                }
                next[i * out + j] = l + 1 == mlp.n_layers() ? 1.f / (1.f + std::exp(-acc)) : std::max(acc, 0.f);
            }
        }
        act = std::move(next);
    }
    bool ok = true;
    for(std::size_t i = 0; i < y.size(); ++i)
    {
        ok = ok && std::fabs(y[i] - act[i]) < 1e-5f;
    }
    CHECK(ok);
}
