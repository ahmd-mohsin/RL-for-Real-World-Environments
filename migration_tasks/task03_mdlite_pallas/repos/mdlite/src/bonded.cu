// Bonded interactions: harmonic bonds and angles, periodic proper dihedrals and
// 1-4 pairs (scaled LJ and plain Coulomb, no cut-off). Forces and virial use
// GROMACS conventions; all sums are fixed point.
#include <cmath>

#include "engine.cuh"

namespace mdl {
namespace {

struct V3 { double x, y, z; };
__device__ inline V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ inline V3 add(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ inline V3 mul(double s, V3 a) { return {s * a.x, s * a.y, s * a.z}; }
__device__ inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
__device__ inline V3 pbc_dx(const double* x, int i, int j, const double* L) {
  double dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
  min_image(dx, dy, dz, L);
  return {dx, dy, dz};
}
__device__ inline void addf(long long* f, int i, V3 v) {
  atomic_add_fixed(f + 3 * i, to_fixed(v.x, FORCE_SCALE));
  atomic_add_fixed(f + 3 * i + 1, to_fixed(v.y, FORCE_SCALE));
  atomic_add_fixed(f + 3 * i + 2, to_fixed(v.z, FORCE_SCALE));
}
// accumulate r (x) f into a 6-component (xx,yy,zz,xy,xz,yz) fixed-point virial
__device__ inline void addvir(long long* v, V3 r, V3 f) {
  v[0] += to_fixed(r.x * f.x, VIRIAL_SCALE);
  v[1] += to_fixed(r.y * f.y, VIRIAL_SCALE);
  v[2] += to_fixed(r.z * f.z, VIRIAL_SCALE);
  v[3] += to_fixed(r.x * f.y, VIRIAL_SCALE);
  v[4] += to_fixed(r.x * f.z, VIRIAL_SCALE);
  v[5] += to_fixed(r.y * f.z, VIRIAL_SCALE);
}
__device__ inline void flush(long long* acc, int eslot, long long e, const long long* v) {
  if (e) atomic_add_fixed(acc + eslot, e);
  for (int c = 0; c < 6; ++c)
    if (v[c]) atomic_add_fixed(acc + A_VXX + c, v[c]);
}

__global__ void k_bonds(const double* x, const int* idx, const double* par, int n, double L0, double L1, double L2, long long* f, long long* acc) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n) return;
  const double L[3] = {L0, L1, L2};
  int i = idx[2 * b], j = idx[2 * b + 1];
  double r0 = par[2 * b], k = par[2 * b + 1];
  V3 d = pbc_dx(x, i, j, L);
  double r = sqrt(dot(d, d));
  double dr = r - r0;
  double e = 0.5 * k * dr * dr;
  long long v[6] = {0, 0, 0, 0, 0, 0};
  if (r > 0) {
    V3 fi = mul(-k * dr / r, d);
    addf(f, i, fi);
    addf(f, j, mul(-1.0, fi));
    addvir(v, d, fi);
  }
  flush(acc, A_BOND, to_fixed(e, ENERGY_SCALE), v);
}

__global__ void k_angles(const double* x, const int* idx, const double* par, int n, double L0, double L1, double L2, long long* f, long long* acc) {
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= n) return;
  const double L[3] = {L0, L1, L2};
  int i = idx[3 * a], j = idx[3 * a + 1], k = idx[3 * a + 2];
  double th0 = par[2 * a], kth = par[2 * a + 1];
  V3 rij = pbc_dx(x, i, j, L), rkj = pbc_dx(x, k, j, L);
  double nrij2 = dot(rij, rij), nrkj2 = dot(rkj, rkj);
  double nrij_1 = 1.0 / sqrt(nrij2), nrkj_1 = 1.0 / sqrt(nrkj2);
  double cth = dot(rij, rkj) * nrij_1 * nrkj_1;
  cth = fmin(1.0, fmax(-1.0, cth));
  double th = acos(cth);
  double dth = th - th0;
  double e = 0.5 * kth * dth * dth;
  double st = -kth * dth / sqrt(fmax(1.0 - cth * cth, 1e-12));
  double sth = st * cth;
  double cik = st * nrij_1 * nrkj_1;
  double cii = sth * nrij_1 * nrij_1;
  double ckk = sth * nrkj_1 * nrkj_1;
  V3 fi = mul(-1.0, sub(mul(cik, rkj), mul(cii, rij)));
  V3 fk = mul(-1.0, sub(mul(cik, rij), mul(ckk, rkj)));
  V3 fj = mul(-1.0, add(fi, fk));
  addf(f, i, fi);
  addf(f, j, fj);
  addf(f, k, fk);
  long long v[6] = {0, 0, 0, 0, 0, 0};
  addvir(v, rij, fi);
  addvir(v, rkj, fk);
  flush(acc, A_ANGLE, to_fixed(e, ENERGY_SCALE), v);
}

__global__ void k_dihedrals(const double* x, const int* idx, const double* par, int n, double L0, double L1, double L2, long long* f, long long* acc) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= n) return;
  const double L[3] = {L0, L1, L2};
  int i = idx[5 * d], j = idx[5 * d + 1], k = idx[5 * d + 2], l = idx[5 * d + 3], mult = idx[5 * d + 4];
  double phi0 = par[2 * d], kphi = par[2 * d + 1];
  V3 rij = pbc_dx(x, i, j, L), rkj = pbc_dx(x, k, j, L), rkl = pbc_dx(x, k, l, L);
  V3 m = cross(rij, rkj), nn = cross(rkj, rkl);
  double phi = atan2(sqrt(dot(cross(m, nn), cross(m, nn))), dot(m, nn));
  if (dot(rij, nn) < 0) phi = -phi;
  double mdphi = mult * phi - phi0;
  double e = kphi * (1.0 + cos(mdphi));
  double ddphi = -kphi * mult * sin(mdphi);
  long long v[6] = {0, 0, 0, 0, 0, 0};
  double iprm = dot(m, m), iprn = dot(nn, nn);
  if (iprm > 1e-24 && iprn > 1e-24) {
    double nrkj2 = dot(rkj, rkj);
    double nrkj = sqrt(nrkj2);
    double nrkj_2 = 1.0 / nrkj2;
    V3 fi = mul(-ddphi * nrkj / iprm, m);
    V3 fl = mul(ddphi * nrkj / iprn, nn);
    double p = dot(rij, rkj) * nrkj_2;
    double q = dot(rkl, rkj) * nrkj_2;
    V3 svec = sub(mul(p, fi), mul(q, fl));
    V3 fj = sub(fi, svec);
    V3 fk = add(fl, svec);
    addf(f, i, fi);
    addf(f, j, mul(-1.0, fj));
    addf(f, k, mul(-1.0, fk));
    addf(f, l, fl);
    addvir(v, rij, fi);
    addvir(v, rkj, mul(-1.0, fk));
    addvir(v, sub(rkj, rkl), fl);
  }
  flush(acc, A_DIH, to_fixed(e, ENERGY_SCALE), v);
}

__global__ void k_pairs(const double* x, const int* idx, int n, const int* type, const double* q, const double* c6t, const double* c12t,
                        int ntypes, double flj, double fqq, double L0, double L1, double L2, long long* f, long long* acc) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n) return;
  const double L[3] = {L0, L1, L2};
  int i = idx[2 * b], j = idx[2 * b + 1];
  V3 d = pbc_dx(x, i, j, L);
  double r2 = dot(d, d);
  double rinv2 = 1.0 / r2, rinv6 = rinv2 * rinv2 * rinv2;
  double c6 = flj * c6t[type[i] * ntypes + type[j]], c12 = flj * c12t[type[i] * ntypes + type[j]];
  double elj = c12 * rinv6 * rinv6 - c6 * rinv6;
  double flj_s = (12.0 * c12 * rinv6 * rinv6 - 6.0 * c6 * rinv6) * rinv2;
  double qq = fqq * ONE_4PI_EPS0 * q[i] * q[j];
  double rinv = sqrt(rinv2);
  double ec = qq * rinv;
  double fc_s = qq * rinv * rinv2;
  V3 fi = mul(flj_s + fc_s, d);
  addf(f, i, fi);
  addf(f, j, mul(-1.0, fi));
  long long v[6] = {0, 0, 0, 0, 0, 0};
  addvir(v, d, fi);
  if (elj != 0.0) atomic_add_fixed(acc + A_LJ14, to_fixed(elj, ENERGY_SCALE));
  flush(acc, A_COUL14, to_fixed(ec, ENERGY_SCALE), v);
}

}  // namespace

void bonded_forces(const DeviceTopology& t, const double* x, const double L[3], long long* f, long long* acc, cudaStream_t s) {
  const int TB = 128;
  if (t.nbonds) k_bonds<<<blocks_for(t.nbonds, TB), TB, 0, s>>>(x, t.bond_idx.p, t.bond_par.p, t.nbonds, L[0], L[1], L[2], f, acc);
  if (t.nangles) k_angles<<<blocks_for(t.nangles, TB), TB, 0, s>>>(x, t.angle_idx.p, t.angle_par.p, t.nangles, L[0], L[1], L[2], f, acc);
  if (t.ndih) k_dihedrals<<<blocks_for(t.ndih, TB), TB, 0, s>>>(x, t.dih_idx.p, t.dih_par.p, t.ndih, L[0], L[1], L[2], f, acc);
  if (t.npairs)
    k_pairs<<<blocks_for(t.npairs, TB), TB, 0, s>>>(x, t.pair_idx.p, t.npairs, t.type.p, t.charge.p, t.c6.p, t.c12.p, t.ntypes,
                                                    t.fudge_lj, t.fudge_qq, L[0], L[1], L[2], f, acc);
  MDL_CHECK_LAUNCH();
}

}  // namespace mdl
