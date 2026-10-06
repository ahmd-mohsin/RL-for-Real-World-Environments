// SHAKE (positions) and RATTLE (velocities) for distance constraints.
//
// One thread per constraint cluster (connected component), Gauss-Seidel sweeps
// over the cluster's constraints in topology order until every constraint is
// within tolerance:
//   positions:  |d^2 - r^2| <= 2 tol d^2
//   velocities: |r . v_ij|  <= tol d^2 / dt
#include <vector>

#include "engine.cuh"

namespace mdl {
namespace {

__global__ void k_shake(double* x, const double* xref, double* v, double inv_dt, const int* cl_ptr, const int* cl_cons, int ncl,
                        const int* cidx, const double* cd, const double* inv_mass, double tol, int maxit, double L0, double L1,
                        double L2, int* fail) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncl) return;
  const double L[3] = {L0, L1, L2};
  int b = cl_ptr[c], e = cl_ptr[c + 1];
  bool done = false;
  for (int it = 0; it < maxit && !done; ++it) {
    done = true;
    for (int k = b; k < e; ++k) {
      int con = cl_cons[k];
      int i = cidx[2 * con], j = cidx[2 * con + 1];
      double d2 = cd[con] * cd[con];
      double rx = x[3 * i] - x[3 * j], ry = x[3 * i + 1] - x[3 * j + 1], rz = x[3 * i + 2] - x[3 * j + 2];
      min_image(rx, ry, rz, L);
      double diff = d2 - (rx * rx + ry * ry + rz * rz);
      if (fabs(diff) <= 2.0 * tol * d2) continue;
      done = false;
      double sx = xref[3 * i] - xref[3 * j], sy = xref[3 * i + 1] - xref[3 * j + 1], sz = xref[3 * i + 2] - xref[3 * j + 2];
      min_image(sx, sy, sz, L);
      double mi = inv_mass[i], mj = inv_mass[j];
      double rs = rx * sx + ry * sy + rz * sz;
      if (rs < 0.1 * d2) {  // the constraint rotated by more than ~84 degrees in one step
        atomicExch(fail, 2);
        return;
      }
      double g = diff / (2.0 * rs * (mi + mj));
      x[3 * i] += g * mi * sx; x[3 * i + 1] += g * mi * sy; x[3 * i + 2] += g * mi * sz;
      x[3 * j] -= g * mj * sx; x[3 * j + 1] -= g * mj * sy; x[3 * j + 2] -= g * mj * sz;
      if (v) {
        v[3 * i] += g * mi * sx * inv_dt; v[3 * i + 1] += g * mi * sy * inv_dt; v[3 * i + 2] += g * mi * sz * inv_dt;
        v[3 * j] -= g * mj * sx * inv_dt; v[3 * j + 1] -= g * mj * sy * inv_dt; v[3 * j + 2] -= g * mj * sz * inv_dt;
      }
    }
  }
  if (!done) atomicExch(fail, 1);
}

__global__ void k_rattle(const double* x, double* v, double dt, const int* cl_ptr, const int* cl_cons, int ncl, const int* cidx,
                         const double* cd, const double* inv_mass, double tol, int maxit, double L0, double L1, double L2,
                         double* vir, int* fail) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncl) return;
  const double L[3] = {L0, L1, L2};
  int b = cl_ptr[c], e = cl_ptr[c + 1];
  double w[6] = {0, 0, 0, 0, 0, 0};
  bool done = false;
  for (int it = 0; it < maxit && !done; ++it) {
    done = true;
    for (int k = b; k < e; ++k) {
      int con = cl_cons[k];
      int i = cidx[2 * con], j = cidx[2 * con + 1];
      double d2 = cd[con] * cd[con];
      double rx = x[3 * i] - x[3 * j], ry = x[3 * i + 1] - x[3 * j + 1], rz = x[3 * i + 2] - x[3 * j + 2];
      min_image(rx, ry, rz, L);
      double vx = v[3 * i] - v[3 * j], vy = v[3 * i + 1] - v[3 * j + 1], vz = v[3 * i + 2] - v[3 * j + 2];
      double rv = rx * vx + ry * vy + rz * vz;
      if (fabs(rv) <= tol * d2 / dt) continue;
      done = false;
      double mi = inv_mass[i], mj = inv_mass[j];
      double r2 = rx * rx + ry * ry + rz * rz;
      double g = -rv / (r2 * (mi + mj));
      v[3 * i] += g * mi * rx; v[3 * i + 1] += g * mi * ry; v[3 * i + 2] += g * mi * rz;
      v[3 * j] -= g * mj * rx; v[3 * j + 1] -= g * mj * ry; v[3 * j + 2] -= g * mj * rz;
      // constraint force on i: m_i dv_i / (dt/2) = 2 g r / dt
      double fs = 2.0 * g / dt;
      w[0] += rx * fs * rx; w[1] += ry * fs * ry; w[2] += rz * fs * rz;
      w[3] += rx * fs * ry; w[4] += rx * fs * rz; w[5] += ry * fs * rz;
    }
  }
  for (int k = 0; k < 6; ++k) vir[6 * (size_t)c + k] = w[k];
  if (!done) atomicExch(fail, 1);
}

}  // namespace

bool shake(const DeviceTopology& t, double* x, const double* xref, double* v, double inv_dt, double tol, int maxit, const double L[3],
           cudaStream_t s) {
  if (t.nclusters == 0) return true;
  static thread_local DevBuf<int> fail(1);
  fail.zero(s);
  k_shake<<<blocks_for(t.nclusters, 64), 64, 0, s>>>(x, xref, v, inv_dt, t.cl_ptr.p, t.cl_cons.p, t.nclusters, t.con_idx.p, t.con_d.p,
                                                     t.inv_mass.p, tol, maxit, L[0], L[1], L[2], fail.p);
  MDL_CHECK_LAUNCH();
  int h = 0;
  MDL_CUDA(cudaMemcpyAsync(&h, fail.p, sizeof(int), cudaMemcpyDeviceToHost, s));
  MDL_CUDA(cudaStreamSynchronize(s));
  return h == 0;
}

bool rattle(const DeviceTopology& t, const double* x, double* v, double dt, double tol, int maxit, const double L[3], double vir6[6],
            cudaStream_t s) {
  if (t.nclusters == 0) return true;
  static thread_local DevBuf<int> fail(1);
  static thread_local DevBuf<double> vir;
  vir.resize(6 * (size_t)t.nclusters);
  fail.zero(s);
  k_rattle<<<blocks_for(t.nclusters, 64), 64, 0, s>>>(x, v, dt, t.cl_ptr.p, t.cl_cons.p, t.nclusters, t.con_idx.p, t.con_d.p,
                                                      t.inv_mass.p, tol, maxit, L[0], L[1], L[2], vir.p, fail.p);
  MDL_CHECK_LAUNCH();
  int h = 0;
  MDL_CUDA(cudaMemcpyAsync(&h, fail.p, sizeof(int), cudaMemcpyDeviceToHost, s));
  // per-cluster virials summed on the host in cluster order (deterministic)
  std::vector<double> hv(6 * (size_t)t.nclusters);
  MDL_CUDA(cudaMemcpyAsync(hv.data(), vir.p, hv.size() * sizeof(double), cudaMemcpyDeviceToHost, s));
  MDL_CUDA(cudaStreamSynchronize(s));
  for (int c = 0; c < t.nclusters; ++c)
    for (int k = 0; k < 6; ++k) vir6[k] += hv[6 * (size_t)c + k];
  return h == 0;
}

}  // namespace mdl
