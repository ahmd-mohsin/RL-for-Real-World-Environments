// Counter-based random numbers: Philox4x32-10 (Salmon et al., SC'11).
//
// key     = (seed low 32 bits, seed high 32 bits)
// counter = (c0, step low 32 bits, step high 32 bits, stream)
//
// Every random number in mdlite is a pure function of (seed, step, stream, c0),
// so runs are reproducible and restarts continue the same random sequence.
#pragma once

#include <cmath>
#include <cstdint>

namespace mdl {

enum RngStream : uint32_t {
  STREAM_LANGEVIN = 1,  // BAOAB O-step, c0 = 2*atom + {0,1}
  STREAM_GENVEL = 2,    // Maxwell-Boltzmann velocities, c0 = 2*atom + {0,1}, step = 0
  STREAM_CSVR = 3,      // thermostat, c0 = draw counter
  STREAM_CRESCALE = 4,  // barostat, c0 = draw counter
};

struct U4 { uint32_t x, y, z, w; };

__host__ __device__ inline void mulhilo32(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
  uint64_t p = (uint64_t)a * (uint64_t)b;
  hi = (uint32_t)(p >> 32);
  lo = (uint32_t)p;
}

__host__ __device__ inline U4 philox4x32_10(U4 c, uint32_t k0, uint32_t k1) {
  const uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u, W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
  for (int round = 0; round < 10; ++round) {
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(M0, c.x, hi0, lo0);
    mulhilo32(M1, c.z, hi1, lo1);
    c = U4{hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0};
    k0 += W0;
    k1 += W1;
  }
  return c;
}

__host__ __device__ inline U4 philox(uint64_t seed, uint64_t step, uint32_t stream, uint32_t c0) {
  return philox4x32_10(U4{c0, (uint32_t)step, (uint32_t)(step >> 32), stream}, (uint32_t)seed, (uint32_t)(seed >> 32));
}

// 53-bit uniform in the open interval (0, 1) from two 32-bit words
__host__ __device__ inline double uniform53(uint32_t a, uint32_t b) {
  return ((double)(a >> 5) * 67108864.0 + (double)(b >> 6) + 0.5) * (1.0 / 9007199254740992.0);
}

// Box-Muller: one Philox block -> two standard normal deviates
__host__ __device__ inline void gauss2(U4 r, double& g0, double& g1) {
  double u1 = uniform53(r.x, r.y);
  double u2 = uniform53(r.z, r.w);
  double rad = sqrt(-2.0 * log(u1));
  double th = 2.0 * 3.14159265358979323846 * u2;
  g0 = rad * cos(th);
  g1 = rad * sin(th);
}

// three normal deviates for atom `a` (two Philox blocks, the fourth deviate is unused)
__host__ __device__ inline void gauss3(uint64_t seed, uint64_t step, uint32_t stream, uint32_t a, double g[3]) {
  double g3;
  gauss2(philox(seed, step, stream, 2u * a), g[0], g[1]);
  gauss2(philox(seed, step, stream, 2u * a + 1u), g[2], g3);
}

// Sequential host-side stream for the global (per-system) random numbers of
// the thermostat and barostat. Each Philox block is consumed whole: uniform()
// takes words (x, y) of a fresh block, gauss() takes a fresh block and returns
// its first Box-Muller deviate.
struct HostStream {
  uint64_t seed, step;
  uint32_t stream;
  uint32_t counter = 0;
  HostStream(uint64_t seed_, uint64_t step_, uint32_t stream_) : seed(seed_), step(step_), stream(stream_) {}
  double uniform() {
    U4 r = philox(seed, step, stream, counter++);
    return uniform53(r.x, r.y);
  }
  double gauss() {
    double g0, g1;
    gauss2(philox(seed, step, stream, counter++), g0, g1);
    return g0;
  }
  // Marsaglia-Tsang gamma deviate with shape a >= 1 and unit scale
  double gamma(double a) {
    const double d = a - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
      double x = gauss();
      double t = 1.0 + c * x;
      if (t <= 0.0) continue;
      double v = t * t * t;
      double u = uniform();
      if (std::log(u) < 0.5 * x * x + d - d * v + d * std::log(v)) return d * v;
    }
  }
  // sum of n squared standard normal deviates (Bussi's resamplekin_sumnoises)
  double sum_noises(long long n) {
    if (n == 0) return 0.0;
    if (n == 1) { double g = gauss(); return g * g; }
    if (n % 2 == 0) return 2.0 * gamma(0.5 * (double)n);
    double g = gauss();
    return 2.0 * gamma(0.5 * (double)(n - 1)) + g * g;
  }
};

}  // namespace mdl
