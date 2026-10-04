// Tessera - small deterministic random number generator (PCG32).
//
// Used to initialize neural parameters and to build synthetic scenes, so the
// sequence must be identical on every platform: no std:: distributions.
#pragma once

#include <cstdint>

namespace tsr
{
class Pcg32
{
public:
    explicit Pcg32(std::uint64_t seed = 0x853c49e6748fea9bULL, std::uint64_t stream = 0xda3e39cb94b95bdbULL)
    {
        state_ = 0u;
        inc_   = (stream << 1u) | 1u;
        next_uint();
        state_ += seed;
        next_uint();
    }

    std::uint32_t next_uint()
    {
        const std::uint64_t old = state_;
        state_                  = old * 6364136223846793005ULL + inc_;
        const auto xorshifted   = static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
        const auto rot          = static_cast<std::uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    // Uniform in [0, 1) with 24 bits of precision.
    float next_float() { return static_cast<float>(next_uint() >> 8) * (1.0f / 16777216.0f); }

    float uniform(float lo, float hi) { return lo + (hi - lo) * next_float(); }

    // Uniform integer in [0, bound).
    std::uint32_t next_bounded(std::uint32_t bound)
    {
        const std::uint32_t threshold = (~bound + 1u) % bound;
        while(true)
        {
            const std::uint32_t r = next_uint();
            if(r >= threshold)
            {
                return r % bound;
            }
        }
    }

private:
    std::uint64_t state_;
    std::uint64_t inc_;
};

} // namespace tsr
