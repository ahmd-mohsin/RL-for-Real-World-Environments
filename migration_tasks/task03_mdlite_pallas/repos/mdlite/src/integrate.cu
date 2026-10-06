// Integrator building blocks and deterministic reductions.
#include <vector>

#include "engine.cuh"
#include "rng.cuh"

namespace mdl {
namespace {

constexpr int RT = 256;

__global__ void k_kick(double* v, const long long* f, const double* inv_mass, double h, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double s = h * inv_mass[i];
  for (int d = 0; d < 3; ++d) v[3 * i + d] += s * from_fixed(f[3 * i + d], FORCE_SCALE);
}

__global__ void k_drift(double* x, const double* v, double h, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * N) return;
  x[i] += h * v[i];
}

__global__ void k_langevin(double* v, const double* inv_mass, double c1, double c2, double kT, uint64_t seed, uint64_t step, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double g[3];
  gauss3(seed, step, STREAM_LANGEVIN, (uint32_t)i, g);
  double sig = c2 * sqrt(kT * inv_mass[i]);
  for (int d = 0; d < 3; ++d) v[3 * i + d] = c1 * v[3 * i + d] + sig * g[d];
}

__global__ void k_genvel(double* v, const double* inv_mass, double kT, uint64_t seed, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double g[3];
  gauss3(seed, 0, STREAM_GENVEL, (uint32_t)i, g);
  double sig = sqrt(kT * inv_mass[i]);
  for (int d = 0; d < 3; ++d) v[3 * i + d] = sig * g[d];
}

__global__ void k_scale(double* v, double lambda, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v[i] *= lambda;
}

// per-block sums: 0.5 m v_a v_b (6), m v (3), m (1)
__global__ void k_kinetic(const double* v, const double* m, int N, double* red) {
  __shared__ double sh[10][RT];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double a[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  if (i < N) {
    double vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2], mi = m[i];
    a[0] = 0.5 * mi * vx * vx; a[1] = 0.5 * mi * vy * vy; a[2] = 0.5 * mi * vz * vz;
    a[3] = 0.5 * mi * vx * vy; a[4] = 0.5 * mi * vx * vz; a[5] = 0.5 * mi * vy * vz;
    a[6] = mi * vx; a[7] = mi * vy; a[8] = mi * vz; a[9] = mi;
  }
  for (int c = 0; c < 10; ++c) sh[c][threadIdx.x] = a[c];
  __syncthreads();
  for (int st = RT / 2; st > 0; st >>= 1) {
    if ((int)threadIdx.x < st)
      for (int c = 0; c < 10; ++c) sh[c][threadIdx.x] += sh[c][threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x < 10) red[(size_t)blockIdx.x * 10 + threadIdx.x] = sh[threadIdx.x][0];
}

__global__ void k_sub_com(double* v, double cx, double cy, double cz, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  v[3 * i] -= cx; v[3 * i + 1] -= cy; v[3 * i + 2] -= cz;
}

// move every molecule rigidly so that its centre of mass scales by mu
__global__ void k_scale_mol(double* x, const double* m, const int* mol_ptr, const int* mol_atoms, int nmol, double mu) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nmol) return;
  double M = 0, cx = 0, cy = 0, cz = 0;
  for (int a = mol_ptr[k]; a < mol_ptr[k + 1]; ++a) {
    int i = mol_atoms[a];
    M += m[i]; cx += m[i] * x[3 * i]; cy += m[i] * x[3 * i + 1]; cz += m[i] * x[3 * i + 2];
  }
  double s = (mu - 1.0) / M;
  cx *= s; cy *= s; cz *= s;
  for (int a = mol_ptr[k]; a < mol_ptr[k + 1]; ++a) {
    int i = mol_atoms[a];
    x[3 * i] += cx; x[3 * i + 1] += cy; x[3 * i + 2] += cz;
  }
}

__global__ void k_reduce(const double* a, long long n, double* red) {
  __shared__ double sh[RT];
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  sh[threadIdx.x] = i < n ? a[i] : 0.0;
  __syncthreads();
  for (int st = RT / 2; st > 0; st >>= 1) {
    if ((int)threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
    __syncthreads();
  }
  if (threadIdx.x == 0) red[blockIdx.x] = sh[0];
}

}  // namespace

void kick(double* v, const long long* f, const double* inv_mass, double half_dt, int N, cudaStream_t s) {
  k_kick<<<blocks_for(N, RT), RT, 0, s>>>(v, f, inv_mass, half_dt, N);
  MDL_CHECK_LAUNCH();
}

void drift(double* x, const double* v, double h, int N, cudaStream_t s) {
  k_drift<<<blocks_for(3LL * N, RT), RT, 0, s>>>(x, v, h, N);
  MDL_CHECK_LAUNCH();
}

void langevin_o(double* v, const double* inv_mass, double c1, double c2, double kT, uint64_t seed, uint64_t step, int N, cudaStream_t s) {
  k_langevin<<<blocks_for(N, RT), RT, 0, s>>>(v, inv_mass, c1, c2, kT, seed, step, N);
  MDL_CHECK_LAUNCH();
}

void gen_velocities(double* v, const double* inv_mass, double kT, uint64_t seed, int N, cudaStream_t s) {
  k_genvel<<<blocks_for(N, RT), RT, 0, s>>>(v, inv_mass, kT, seed, N);
  MDL_CHECK_LAUNCH();
}

void scale_velocities(double* v, double lambda, int N, cudaStream_t s) {
  k_scale<<<blocks_for(3LL * N, RT), RT, 0, s>>>(v, lambda, 3 * N);
  MDL_CHECK_LAUNCH();
}

void kinetic(const double* v, const double* mass, int N, double ke6[6], double mom[3], cudaStream_t s) {
  int nb = blocks_for(N, RT);
  static thread_local DevBuf<double> red;
  red.resize(10 * (size_t)nb);
  k_kinetic<<<nb, RT, 0, s>>>(v, mass, N, red.p);
  MDL_CHECK_LAUNCH();
  std::vector<double> h(10 * (size_t)nb);
  MDL_CUDA(cudaMemcpyAsync(h.data(), red.p, h.size() * sizeof(double), cudaMemcpyDeviceToHost, s));
  MDL_CUDA(cudaStreamSynchronize(s));
  double sum[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int b = 0; b < nb; ++b)
    for (int c = 0; c < 10; ++c) sum[c] += h[10 * (size_t)b + c];
  for (int c = 0; c < 6; ++c) ke6[c] = sum[c];
  for (int c = 0; c < 3; ++c) mom[c] = sum[6 + c];
}

void remove_com(double* v, const double com_v[3], int N, cudaStream_t s) {
  k_sub_com<<<blocks_for(N, RT), RT, 0, s>>>(v, com_v[0], com_v[1], com_v[2], N);
  MDL_CHECK_LAUNCH();
}

void scale_molecules(const DeviceTopology& t, double* x, double mu, const double L_old[3], cudaStream_t s) {
  (void)L_old;
  k_scale_mol<<<blocks_for(t.nmol, 128), 128, 0, s>>>(x, t.mass.p, t.mol_ptr.p, t.mol_atoms.p, t.nmol, mu);
  MDL_CHECK_LAUNCH();
}

double reduce_sum(const double* a, long long n, cudaStream_t s) {
  int nb = blocks_for(n, RT);
  static thread_local DevBuf<double> red;
  red.resize(nb);
  k_reduce<<<nb, RT, 0, s>>>(a, n, red.p);
  MDL_CHECK_LAUNCH();
  std::vector<double> h(nb);
  MDL_CUDA(cudaMemcpyAsync(h.data(), red.p, nb * sizeof(double), cudaMemcpyDeviceToHost, s));
  MDL_CUDA(cudaStreamSynchronize(s));
  double sum = 0;
  for (double v : h) sum += v;
  return sum;
}

}  // namespace mdl
