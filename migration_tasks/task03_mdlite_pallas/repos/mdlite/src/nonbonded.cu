// Short-range nonbonded forces (LJ + real-space Coulomb) over the full neighbour
// list, and the corrections for excluded pairs.
//
// Each pair contribution is converted to fixed point before it is summed, so the
// result does not depend on the order or the buffer content of the list.
#include "engine.cuh"

namespace mdl {
namespace {

constexpr int NB_THREADS = 128;

// block-wide sum of fixed-point accumulators, then one atomic per block
template <int NV>
__device__ void block_flush(long long (&v)[NV], long long* acc, const int* slots) {
  __shared__ long long sh[NV][NB_THREADS / 32];
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int k = 0; k < NV; ++k) {
    long long x = v[k];
    for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
    if (lane == 0) sh[k][warp] = x;
  }
  __syncthreads();
  if (threadIdx.x < NV) {
    long long s = 0;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += sh[threadIdx.x][w];
    if (s) atomic_add_fixed(acc + slots[threadIdx.x], s);
  }
}

__device__ inline void lj_pair(const NbParams& p, double c6, double c12, double r2, double rinv, double& fscal, double& e) {
  double rinv2 = rinv * rinv;
  double rinv6 = rinv2 * rinv2 * rinv2;
  if (p.vdw_mod == 0) {
    double v6 = c6 * rinv6, v12 = c12 * rinv6 * rinv6;
    e = (v12 - c12 * p.sh12) - (v6 - c6 * p.sh6);
    fscal = (12.0 * v12 - 6.0 * v6) * rinv2;
  } else {
    // GROMACS force switch: the force of each r^-p term is smoothly switched to zero between rsw and rc
    double r = r2 * rinv;
    double f6 = 6.0 * rinv6 * rinv, f12 = 12.0 * rinv6 * rinv6 * rinv;
    double v6 = rinv6 - p.fsw_C6, v12 = rinv6 * rinv6 - p.fsw_C12;
    if (r > p.rsw) {
      double dr = r - p.rsw, dr2 = dr * dr, dr3 = dr2 * dr;
      f6 += p.fsw_A6 * dr2 + p.fsw_B6 * dr3;
      f12 += p.fsw_A12 * dr2 + p.fsw_B12 * dr3;
      v6 -= p.fsw_A6 / 3.0 * dr3 + p.fsw_B6 / 4.0 * dr3 * dr;
      v12 -= p.fsw_A12 / 3.0 * dr3 + p.fsw_B12 / 4.0 * dr3 * dr;
    }
    e = c12 * v12 - c6 * v6;
    fscal = (c12 * f12 - c6 * f6) * rinv;
  }
}

__global__ void k_nonbonded(const double* __restrict__ x, int N, double L0, double L1, double L2,
                            const int* __restrict__ count, const int* __restrict__ list, int cap,
                            const int* __restrict__ type, const double* __restrict__ q, const double* __restrict__ c6t,
                            const double* __restrict__ c12t, int ntypes, NbParams p, long long* __restrict__ f,
                            long long* __restrict__ acc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  long long a[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // elj, ecoul, vxx, vyy, vzz, vxy, vxz, vyz, unused
  if (i < N) {
    const double L[3] = {L0, L1, L2};
    double xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    double qi = q[i] * p.epsfac;
    int ti = type[i] * ntypes;
    long long fx = 0, fy = 0, fz = 0;
    int n = count[i];
    if (n > cap) n = cap;
    const int* li = list + (size_t)i * cap;
    for (int k = 0; k < n; ++k) {
      int j = li[k];
      double dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
      min_image(dx, dy, dz, L);
      double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= p.rc2) continue;
      double rinv = rsqrt(r2);
      // rsqrt is not correctly rounded; one Newton step makes it so to within an ulp
      rinv = rinv * (1.5 - 0.5 * r2 * rinv * rinv);
      double fs_lj = 0, elj = 0;
      double c6 = c6t[ti + type[j]], c12 = c12t[ti + type[j]];
      if (c6 != 0.0 || c12 != 0.0) lj_pair(p, c6, c12, r2, rinv, fs_lj, elj);
      double qq = qi * q[j];
      double fs_c = 0, ec = 0;
      if (qq != 0.0) {
        if (p.coul == 0) {
          double r = r2 * rinv;
          double br = p.beta * r;
          double erfc_br = erfc(br);
          ec = qq * erfc_br * rinv;
          fs_c = qq * (erfc_br * rinv + 1.1283791670955126 * p.beta * exp(-br * br)) * rinv * rinv;
        } else {
          ec = qq * (rinv + p.krf * r2 - p.crf);
          fs_c = qq * (rinv * rinv * rinv - 2.0 * p.krf);
        }
      }
      double fs = fs_lj + fs_c;
      double fpx = fs * dx, fpy = fs * dy, fpz = fs * dz;
      fx += to_fixed(fpx, FORCE_SCALE);
      fy += to_fixed(fpy, FORCE_SCALE);
      fz += to_fixed(fpz, FORCE_SCALE);
      a[0] += to_fixed(elj, ENERGY_SCALE);
      a[1] += to_fixed(ec, ENERGY_SCALE);
      a[2] += to_fixed(dx * fpx, VIRIAL_SCALE);
      a[3] += to_fixed(dy * fpy, VIRIAL_SCALE);
      a[4] += to_fixed(dz * fpz, VIRIAL_SCALE);
      a[5] += to_fixed(dx * fpy, VIRIAL_SCALE);
      a[6] += to_fixed(dx * fpz, VIRIAL_SCALE);
      a[7] += to_fixed(dy * fpz, VIRIAL_SCALE);
    }
    f[3 * i] += fx;
    f[3 * i + 1] += fy;
    f[3 * i + 2] += fz;
  }
  // every pair is visited twice; the caller halves these sums
  const int slots[8] = {A_LJ2, A_COUL2, A_V2XX, A_V2YY, A_V2ZZ, A_V2XY, A_V2XZ, A_V2YZ};
  long long v8[8] = {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]};
  block_flush<8>(v8, acc, slots);
}

// Excluded pairs: PME removes the reciprocal-space interaction erf(beta r)/r of
// every excluded pair (at any distance); reaction-field adds q_i q_j (k_rf r^2 - c_rf)
// for excluded pairs inside the cut-off.
__global__ void k_exclusions(const double* x, const int* pairs, int npairs, double L0, double L1, double L2, const double* q,
                             NbParams p, double rc2, long long* f, long long* acc) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  long long a[7] = {0, 0, 0, 0, 0, 0, 0};
  if (k < npairs) {
    const double L[3] = {L0, L1, L2};
    int i = pairs[2 * k], j = pairs[2 * k + 1];
    double qq = p.epsfac * q[i] * q[j];
    double dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
    min_image(dx, dy, dz, L);
    double r2 = dx * dx + dy * dy + dz * dz;
    double fs = 0, e = 0;
    if (qq != 0.0) {
      if (p.coul == 0) {
        if (r2 > 0) {
          double r = sqrt(r2);
          double br = p.beta * r;
          double erf_br = erf(br);
          e = -qq * erf_br / r;
          // d/dr [erf(br)/r] = (2b/sqrt(pi) exp(-b^2 r^2) r - erf(br)) / r^2 ; force = -dV/dr * dx/r
          fs = -qq * (erf_br / r - 1.1283791670955126 * p.beta * exp(-br * br)) / r2;
        } else {
          e = -qq * 1.1283791670955126 * p.beta;
        }
      } else if (r2 < rc2) {
        e = qq * (p.krf * r2 - p.crf);
        fs = -2.0 * qq * p.krf;
      }
    }
    if (fs != 0.0) {
      long long fx = to_fixed(fs * dx, FORCE_SCALE), fy = to_fixed(fs * dy, FORCE_SCALE), fz = to_fixed(fs * dz, FORCE_SCALE);
      atomic_add_fixed(f + 3 * i, fx); atomic_add_fixed(f + 3 * i + 1, fy); atomic_add_fixed(f + 3 * i + 2, fz);
      atomic_add_fixed(f + 3 * j, -fx); atomic_add_fixed(f + 3 * j + 1, -fy); atomic_add_fixed(f + 3 * j + 2, -fz);
    }
    a[0] = to_fixed(e, ENERGY_SCALE);
    a[1] = to_fixed(dx * fs * dx, VIRIAL_SCALE);
    a[2] = to_fixed(dy * fs * dy, VIRIAL_SCALE);
    a[3] = to_fixed(dz * fs * dz, VIRIAL_SCALE);
    a[4] = to_fixed(dx * fs * dy, VIRIAL_SCALE);
    a[5] = to_fixed(dx * fs * dz, VIRIAL_SCALE);
    a[6] = to_fixed(dy * fs * dz, VIRIAL_SCALE);
  }
  const int slots[7] = {A_COUL_EXCL, A_VXX, A_VYY, A_VZZ, A_VXY, A_VXZ, A_VYZ};
  block_flush<7>(a, acc, slots);
}

}  // namespace

void nonbonded_forces(const NeighborList& nl, const DeviceTopology& t, const NbParams& p, const double* x, const double L[3],
                      long long* f, long long* acc, cudaStream_t s) {
  // halving of the doubly counted pair sums happens at read-out (Simulation::energies)
  k_nonbonded<<<blocks_for(t.N, NB_THREADS), NB_THREADS, 0, s>>>(x, t.N, L[0], L[1], L[2], nl.count.p, nl.list.p, nl.cap, t.type.p,
                                                                 t.charge.p, t.c6.p, t.c12.p, t.ntypes, p, f, acc);
  MDL_CHECK_LAUNCH();
}

void exclusion_corrections(const DeviceTopology& t, const NbParams& p, double cutoff, const double* x, const double L[3],
                           long long* f, long long* acc, cudaStream_t s) {
  if (t.nexcl_pairs == 0) return;
  k_exclusions<<<blocks_for(t.nexcl_pairs, NB_THREADS), NB_THREADS, 0, s>>>(x, t.excl_pairs.p, t.nexcl_pairs, L[0], L[1], L[2],
                                                                            t.charge.p, p, cutoff * cutoff, f, acc);
  MDL_CHECK_LAUNCH();
}

}  // namespace mdl
