// mdlite: shared types, units, error handling and fixed-point helpers.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace mdl {

// --- units: nm, ps, amu, e, kJ/mol, K, bar (GROMACS conventions)
constexpr double BOLTZ = 0.0083144626181532;    // kJ/(mol K)
constexpr double ONE_4PI_EPS0 = 138.935457833;  // kJ nm / (mol e^2)
constexpr double PRESFAC = 16.6054;             // kJ/(mol nm^3) -> bar
constexpr double PI = 3.14159265358979323846;

// --- fixed-point accumulation. Every force, energy and virial sum goes
// through 64-bit integers so that results are independent of summation
// order (neighbour-list layout, atomics, restarts). Values are rounded with
// llrint() once per contribution.
constexpr double FORCE_SCALE = 4294967296.0;        // 2^32 per kJ/mol/nm
constexpr double ENERGY_SCALE = 4294967296.0;       // 2^32 per kJ/mol
constexpr double VIRIAL_SCALE = 4294967296.0;       // 2^32 per kJ/mol
constexpr double GRID_SCALE = 1099511627776.0;      // 2^40 per e (PME charge grid)

__host__ __device__ inline long long to_fixed(double v, double scale) {
#ifdef __CUDA_ARCH__
  return __double2ll_rn(v * scale);
#else
  return llrint(v * scale);
#endif
}
__host__ __device__ inline double from_fixed(long long v, double scale) { return (double)v / scale; }

#ifdef __CUDACC__
__device__ inline void atomic_add_fixed(long long* addr, long long v) {
  atomicAdd(reinterpret_cast<unsigned long long*>(addr), static_cast<unsigned long long>(v));
}
#endif

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

#define MDL_CUDA(call)                                                                           \
  do {                                                                                           \
    cudaError_t err_ = (call);                                                                   \
    if (err_ != cudaSuccess)                                                                     \
      throw ::mdl::Error(std::string("CUDA error: ") + cudaGetErrorString(err_) + " at " +      \
                         __FILE__ + ":" + std::to_string(__LINE__));                            \
  } while (0)

#define MDL_CHECK_LAUNCH() MDL_CUDA(cudaGetLastError())

inline int blocks_for(long long n, int threads) { return (int)((n + threads - 1) / threads); }

// --- minimal device buffer
template <class T>
struct DevBuf {
  T* p = nullptr;
  size_t n = 0;
  DevBuf() = default;
  explicit DevBuf(size_t count) { resize(count); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  ~DevBuf() { release(); }
  void release() {
    if (p) cudaFree(p);
    p = nullptr;
    n = 0;
  }
  void resize(size_t count) {
    if (count == n) return;
    release();
    if (count) MDL_CUDA(cudaMalloc(&p, count * sizeof(T)));
    n = count;
  }
  void zero(cudaStream_t s = 0) {
    if (n) MDL_CUDA(cudaMemsetAsync(p, 0, n * sizeof(T), s));
  }
  void upload(const T* h, size_t count) {
    resize(count);
    if (count) MDL_CUDA(cudaMemcpy(p, h, count * sizeof(T), cudaMemcpyHostToDevice));
  }
  void upload(const std::vector<T>& h) { upload(h.data(), h.size()); }
  void download(T* h, size_t count) const {
    if (count) MDL_CUDA(cudaMemcpy(h, p, count * sizeof(T), cudaMemcpyDeviceToHost));
  }
  std::vector<T> to_host() const {
    std::vector<T> h(n);
    download(h.data(), n);
    return h;
  }
};

struct Box {
  double L[3];
};

// minimum-image displacement for an orthorhombic box
__host__ __device__ inline void min_image(double& dx, double& dy, double& dz, const double* L) {
  dx -= L[0] * rint(dx / L[0]);
  dy -= L[1] * rint(dy / L[1]);
  dz -= L[2] * rint(dz / L[2]);
}

}  // namespace mdl
