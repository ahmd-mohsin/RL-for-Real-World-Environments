// Smooth particle-mesh Ewald (Essmann et al., J. Chem. Phys. 103, 8577, 1995).
//
// Charges are spread with cardinal B-splines of order p onto a fixed-point grid
// (exact, order-independent), transformed with a double-precision real-to-complex
// FFT, multiplied by the influence function, transformed back and interpolated
// to the atoms.
#include <cmath>
#include <vector>

#include "engine.cuh"

namespace mdl {
namespace {

constexpr int MAXORD = 8;
constexpr int RED_THREADS = 256;

// B-spline weights and derivatives of order p for fractional offset fr in [0, 1)
__device__ inline void bspline(double fr, int p, double* w, double* dw) {
  w[p - 1] = 0.0;
  w[1] = fr;
  w[0] = 1.0 - fr;
  for (int j = 3; j < p; ++j) {
    double div = 1.0 / (j - 1.0);
    w[j - 1] = div * fr * w[j - 2];
    for (int k = 1; k < j - 1; ++k) w[j - k - 1] = div * ((fr + k) * w[j - k - 2] + (j - k - fr) * w[j - k - 1]);
    w[0] = div * (1.0 - fr) * w[0];
  }
  dw[0] = -w[0];
  for (int j = 1; j < p; ++j) dw[j] = w[j - 1] - w[j];
  double div = 1.0 / (p - 1.0);
  w[p - 1] = div * fr * w[p - 2];
  for (int k = 1; k < p - 1; ++k) w[p - k - 1] = div * ((fr + k) * w[p - k - 2] + (p - k - fr) * w[p - k - 1]);
  w[0] = div * (1.0 - fr) * w[0];
}

// fractional grid coordinate: u = n * (x/L - floor(x/L)); returns the base index and the offset
__device__ inline int grid_base(double x, double L, int n, double& fr) {
  double s = x / L - floor(x / L);
  double u = s * n;
  int k = (int)floor(u);
  fr = u - k;
  return k;  // may equal n when s rounds to 1; callers wrap with % n
}

__global__ void k_spread(const double* x, const double* q, int N, double L0, double L1, double L2, int n0, int n1, int n2,
                         int p, long long* grid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double qi = q[i];
  if (qi == 0.0) return;
  double fr[3], w[3][MAXORD], dw[MAXORD];
  int k0 = grid_base(x[3 * i], L0, n0, fr[0]);
  int k1 = grid_base(x[3 * i + 1], L1, n1, fr[1]);
  int k2 = grid_base(x[3 * i + 2], L2, n2, fr[2]);
  for (int d = 0; d < 3; ++d) bspline(fr[d], p, w[d], dw);
  for (int a = 0; a < p; ++a) {
    int ix = (k0 + a) % n0;
    double wa = qi * w[0][a];
    for (int b = 0; b < p; ++b) {
      int iy = (k1 + b) % n1;
      double wab = wa * w[1][b];
      long long* row = grid + ((size_t)ix * n1 + iy) * n2;
      for (int c = 0; c < p; ++c) atomic_add_fixed(row + (k2 + c) % n2, to_fixed(wab * w[2][c], GRID_SCALE));
    }
  }
}

__global__ void k_unfix(const long long* in, double* out, size_t n) {
  size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (k < n) out[k] = from_fixed(in[k], GRID_SCALE);
}

// influence function, energy and virial; one thread per complex grid point,
// per-block partial sums (7 values) written to `red` in block order
__global__ void k_solve(cufftDoubleComplex* g, int n0, int n1, int n2, double L0, double L1, double L2, double beta,
                        const double* bm0, const double* bm1, const double* bm2, double epsfac, double* red) {
  __shared__ double sh[7][RED_THREADS];
  int nz = n2 / 2 + 1;
  size_t total = (size_t)n0 * n1 * nz;
  size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  double v[7] = {0, 0, 0, 0, 0, 0, 0};
  if (k < total) {
    int kz = (int)(k % nz);
    int ky = (int)((k / nz) % n1);
    int kx = (int)(k / ((size_t)nz * n1));
    if (kx != 0 || ky != 0 || kz != 0) {
      double mx = (kx <= n0 / 2 ? kx : kx - n0) / L0;
      double my = (ky <= n1 / 2 ? ky : ky - n1) / L1;
      double mz = kz / L2;
      double m2 = mx * mx + my * my + mz * mz;
      double V = L0 * L1 * L2;
      double denom = PI * V * m2 * bm0[kx] * bm1[ky] * bm2[kz];
      double eterm = epsfac * exp(-PI * PI * m2 / (beta * beta)) / denom;
      cufftDoubleComplex c = g[k];
      double s2 = c.x * c.x + c.y * c.y;
      double wgt = (kz == 0 || (n2 % 2 == 0 && kz == n2 / 2)) ? 1.0 : 2.0;
      double e = 0.5 * wgt * eterm * s2;
      double vf = 2.0 * (1.0 + PI * PI * m2 / (beta * beta)) / m2;
      v[0] = e;
      v[1] = e * (1.0 - vf * mx * mx);
      v[2] = e * (1.0 - vf * my * my);
      v[3] = e * (1.0 - vf * mz * mz);
      v[4] = -e * vf * mx * my;
      v[5] = -e * vf * mx * mz;
      v[6] = -e * vf * my * mz;
      g[k].x = c.x * eterm;
      g[k].y = c.y * eterm;
    } else {
      g[k].x = 0.0;
      g[k].y = 0.0;
    }
  }
  for (int c = 0; c < 7; ++c) sh[c][threadIdx.x] = v[c];
  __syncthreads();
  for (int st = RED_THREADS / 2; st > 0; st >>= 1) {
    if ((int)threadIdx.x < st)
      for (int c = 0; c < 7; ++c) sh[c][threadIdx.x] += sh[c][threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x < 7) red[(size_t)blockIdx.x * 7 + threadIdx.x] = sh[threadIdx.x][0];
}

__global__ void k_gather(const double* x, const double* q, int N, double L0, double L1, double L2, int n0, int n1, int n2, int p,
                         const double* phi, long long* f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double qi = q[i];
  if (qi == 0.0) return;
  double fr[3], w[3][MAXORD], dw[3][MAXORD];
  int k0 = grid_base(x[3 * i], L0, n0, fr[0]);
  int k1 = grid_base(x[3 * i + 1], L1, n1, fr[1]);
  int k2 = grid_base(x[3 * i + 2], L2, n2, fr[2]);
  for (int d = 0; d < 3; ++d) bspline(fr[d], p, w[d], dw[d]);
  double gx = 0, gy = 0, gz = 0;
  for (int a = 0; a < p; ++a) {
    int ix = (k0 + a) % n0;
    for (int b = 0; b < p; ++b) {
      int iy = (k1 + b) % n1;
      const double* row = phi + ((size_t)ix * n1 + iy) * n2;
      for (int c = 0; c < p; ++c) {
        double ph = row[(k2 + c) % n2];
        gx += dw[0][a] * w[1][b] * w[2][c] * ph;
        gy += w[0][a] * dw[1][b] * w[2][c] * ph;
        gz += w[0][a] * w[1][b] * dw[2][c] * ph;
      }
    }
  }
  f[3 * i] += to_fixed(-qi * gx * n0 / L0, FORCE_SCALE);
  f[3 * i + 1] += to_fixed(-qi * gy * n1 / L1, FORCE_SCALE);
  f[3 * i + 2] += to_fixed(-qi * gz * n2 / L2, FORCE_SCALE);
}

// cardinal B-spline M_p(x) by the recursion M_n(x) = x/(n-1) M_{n-1}(x) + (n-x)/(n-1) M_{n-1}(x-1)
double bspline_M(int n, double x) {
  if (n == 2) return (x < 0 || x > 2) ? 0.0 : 1.0 - std::fabs(x - 1.0);
  return x / (n - 1) * bspline_M(n - 1, x) + (n - x) / (n - 1) * bspline_M(n - 1, x - 1.0);
}

std::vector<double> bspline_moduli(int n, int p) {
  std::vector<double> mod(n);
  for (int k = 0; k < n; ++k) {
    double re = 0, im = 0;
    for (int j = 0; j <= p - 2; ++j) {
      double m = bspline_M(p, j + 1.0);
      double arg = 2.0 * PI * j * k / n;
      re += m * std::cos(arg);
      im += m * std::sin(arg);
    }
    mod[k] = re * re + im * im;
  }
  // odd orders have zeros at the Nyquist frequency; interpolate as in AMBER / GROMACS
  for (int k = 0; k < n; ++k)
    if (mod[k] < 1e-7) mod[k] = 0.5 * (mod[(k - 1 + n) % n] + mod[(k + 1) % n]);
  return mod;
}

}  // namespace

Pme::~Pme() {
  if (fwd) cufftDestroy(fwd);
  if (bwd) cufftDestroy(bwd);
}

void pme_setup(Pme& pme, int order, const int n[3], double beta) {
  pme.order = order;
  pme.beta = beta;
  for (int d = 0; d < 3; ++d) {
    pme.n[d] = n[d];
    pme.bmod[d].upload(bspline_moduli(n[d], order));
  }
  size_t nreal = (size_t)n[0] * n[1] * n[2];
  size_t ncplx = (size_t)n[0] * n[1] * (n[2] / 2 + 1);
  pme.qgrid_fixed.resize(nreal);
  pme.grid.resize(nreal);
  pme.kgrid.resize(ncplx);
  pme.red.resize(7 * (size_t)blocks_for(ncplx, RED_THREADS));
  if (pme.fwd) cufftDestroy(pme.fwd);
  if (pme.bwd) cufftDestroy(pme.bwd);
  if (cufftPlan3d(&pme.fwd, n[0], n[1], n[2], CUFFT_D2Z) != CUFFT_SUCCESS ||
      cufftPlan3d(&pme.bwd, n[0], n[1], n[2], CUFFT_Z2D) != CUFFT_SUCCESS)
    throw Error("PME: cuFFT plan creation failed");
}

double pme_forces(Pme& pme, const DeviceTopology& t, const double* x, const double L[3], long long* f, double vir6[6], cudaStream_t s) {
  const int* n = pme.n;
  size_t nreal = (size_t)n[0] * n[1] * n[2];
  size_t ncplx = (size_t)n[0] * n[1] * (n[2] / 2 + 1);
  pme.qgrid_fixed.zero(s);
  k_spread<<<blocks_for(t.N, 128), 128, 0, s>>>(x, t.charge.p, t.N, L[0], L[1], L[2], n[0], n[1], n[2], pme.order, pme.qgrid_fixed.p);
  MDL_CHECK_LAUNCH();
  k_unfix<<<blocks_for(nreal, 256), 256, 0, s>>>(pme.qgrid_fixed.p, pme.grid.p, nreal);
  MDL_CHECK_LAUNCH();
  cufftSetStream(pme.fwd, s);
  cufftSetStream(pme.bwd, s);
  if (cufftExecD2Z(pme.fwd, pme.grid.p, pme.kgrid.p) != CUFFT_SUCCESS) throw Error("PME: forward FFT failed");
  int nb = blocks_for(ncplx, RED_THREADS);
  k_solve<<<nb, RED_THREADS, 0, s>>>(pme.kgrid.p, n[0], n[1], n[2], L[0], L[1], L[2], pme.beta, pme.bmod[0].p, pme.bmod[1].p,
                                     pme.bmod[2].p, ONE_4PI_EPS0, pme.red.p);
  MDL_CHECK_LAUNCH();
  if (cufftExecZ2D(pme.bwd, pme.kgrid.p, pme.grid.p) != CUFFT_SUCCESS) throw Error("PME: backward FFT failed");
  k_gather<<<blocks_for(t.N, 128), 128, 0, s>>>(x, t.charge.p, t.N, L[0], L[1], L[2], n[0], n[1], n[2], pme.order, pme.grid.p, f);
  MDL_CHECK_LAUNCH();
  std::vector<double> red(7 * (size_t)nb);
  MDL_CUDA(cudaMemcpyAsync(red.data(), pme.red.p, red.size() * sizeof(double), cudaMemcpyDeviceToHost, s));
  MDL_CUDA(cudaStreamSynchronize(s));
  double sum[7] = {0, 0, 0, 0, 0, 0, 0};
  for (int b = 0; b < nb; ++b)
    for (int c = 0; c < 7; ++c) sum[c] += red[7 * (size_t)b + c];
  for (int c = 0; c < 6; ++c) vir6[c] += sum[c + 1];
  return sum[0];
}

}  // namespace mdl
