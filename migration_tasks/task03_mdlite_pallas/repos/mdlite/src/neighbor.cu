// Cell-list based Verlet neighbour list (full list: every pair appears for both atoms).
//
// The list holds all non-excluded pairs within rlist = cutoff + skin. It is
// rebuilt when any atom has moved more than skin/2 since the last build, or
// when forced (box change, restart, set_positions). Capacity per atom grows
// automatically: on overflow the list is rebuilt with 1.2x the largest count,
// rounded up to a multiple of 32.
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cmath>

#include "engine.cuh"

namespace mdl {
namespace {

__device__ inline int cell_coord(double x, double L, int nc) {
  double s = x / L - floor(x / L);
  int c = (int)(s * nc);
  return c >= nc ? nc - 1 : (c < 0 ? 0 : c);
}

__global__ void k_assign_cells(const double* x, int N, double L0, double L1, double L2, int n0, int n1, int n2, int* cell, int* order) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int cx = cell_coord(x[3 * i], L0, n0);
  int cy = cell_coord(x[3 * i + 1], L1, n1);
  int cz = cell_coord(x[3 * i + 2], L2, n2);
  cell[i] = (cx * n1 + cy) * n2 + cz;
  order[i] = i;
}

__global__ void k_cell_bounds(const int* cell_sorted, int N, int* start, int* end) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= N) return;
  int c = cell_sorted[k];
  if (k == 0 || cell_sorted[k - 1] != c) start[c] = k;
  if (k == N - 1 || cell_sorted[k + 1] != c) end[c] = k + 1;
}

__device__ inline bool excluded(const int* ptr, const int* idx, int i, int j) {
  for (int e = ptr[i]; e < ptr[i + 1]; ++e)
    if (idx[e] == j) return true;
  return false;
}

__global__ void k_build(const double* x, int N, double L0, double L1, double L2, int n0, int n1, int n2, double rl2,
                        const int* cell, const int* order_sorted, const int* start, const int* end,
                        const int* ex_ptr, const int* ex_idx, int cap, int* count, int* list, int* maxcount) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const double L[3] = {L0, L1, L2};
  double xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
  int c = cell[i];
  int cz = c % n2, cy = (c / n2) % n1, cx = c / (n1 * n2);
  // offsets per dimension, deduplicated for fewer than 3 cells
  int ox[3], oy[3], oz[3], nx, ny, nz;
  nx = n0 >= 3 ? 3 : n0; ny = n1 >= 3 ? 3 : n1; nz = n2 >= 3 ? 3 : n2;
  for (int k = 0; k < 3; ++k) { ox[k] = k - 1; oy[k] = k - 1; oz[k] = k - 1; }
  if (n0 < 3) for (int k = 0; k < n0; ++k) ox[k] = k;
  if (n1 < 3) for (int k = 0; k < n1; ++k) oy[k] = k;
  if (n2 < 3) for (int k = 0; k < n2; ++k) oz[k] = k;
  int n = 0;
  for (int a = 0; a < nx; ++a)
    for (int b = 0; b < ny; ++b)
      for (int d = 0; d < nz; ++d) {
        int ccx = (cx + ox[a] + n0) % n0, ccy = (cy + oy[b] + n1) % n1, ccz = (cz + oz[d] + n2) % n2;
        int nb = (ccx * n1 + ccy) * n2 + ccz;
        for (int k = start[nb]; k < end[nb]; ++k) {
          int j = order_sorted[k];
          if (j == i) continue;
          double dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
          min_image(dx, dy, dz, L);
          if (dx * dx + dy * dy + dz * dz >= rl2) continue;
          if (excluded(ex_ptr, ex_idx, i, j)) continue;
          if (n < cap) list[(size_t)i * cap + n] = j;
          ++n;
        }
      }
  count[i] = n;
  atomicMax(maxcount, n);
}

__global__ void k_displacement(const double* x, const double* xref, int N, double lim2, int* flag) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double dx = x[3 * i] - xref[3 * i], dy = x[3 * i + 1] - xref[3 * i + 1], dz = x[3 * i + 2] - xref[3 * i + 2];
  if (dx * dx + dy * dy + dz * dz > lim2) flag[1] = 1;
}

}  // namespace

void neighbor_build(NeighborList& nl, const DeviceTopology& t, const double* x, const double L[3], cudaStream_t s) {
  int N = t.N;
  const int TB = 128;
  int nc[3];
  for (int d = 0; d < 3; ++d) nc[d] = std::max(1, (int)std::floor(L[d] / nl.rlist));
  for (int d = 0; d < 3; ++d) nl.nc[d] = nc[d];
  int ncell = nc[0] * nc[1] * nc[2];
  nl.cell.resize(N); nl.order.resize(N); nl.cell_sorted.resize(N); nl.order_sorted.resize(N);
  nl.cell_start.resize(ncell); nl.cell_end.resize(ncell);
  nl.count.resize(N);
  nl.flag.resize(2);
  k_assign_cells<<<blocks_for(N, TB), TB, 0, s>>>(x, N, L[0], L[1], L[2], nc[0], nc[1], nc[2], nl.cell.p, nl.order.p);
  MDL_CHECK_LAUNCH();
  size_t tmp = 0;
  int bits = 1;
  while ((1 << bits) < ncell) ++bits;
  cub::DeviceRadixSort::SortPairs(nullptr, tmp, nl.cell.p, nl.cell_sorted.p, nl.order.p, nl.order_sorted.p, N, 0, bits, s);
  if (nl.sort_tmp.n < tmp) nl.sort_tmp.resize(tmp);
  cub::DeviceRadixSort::SortPairs(nl.sort_tmp.p, tmp, nl.cell.p, nl.cell_sorted.p, nl.order.p, nl.order_sorted.p, N, 0, bits, s);
  MDL_CHECK_LAUNCH();
  MDL_CUDA(cudaMemsetAsync(nl.cell_start.p, 0, ncell * sizeof(int), s));
  MDL_CUDA(cudaMemsetAsync(nl.cell_end.p, 0, ncell * sizeof(int), s));
  k_cell_bounds<<<blocks_for(N, TB), TB, 0, s>>>(nl.cell_sorted.p, N, nl.cell_start.p, nl.cell_end.p);
  MDL_CHECK_LAUNCH();
  double rl2 = nl.rlist * nl.rlist;
  for (;;) {
    if (nl.cap == 0) nl.cap = 64;
    nl.list.resize((size_t)N * nl.cap);
    MDL_CUDA(cudaMemsetAsync(nl.flag.p, 0, sizeof(int), s));
    k_build<<<blocks_for(N, TB), TB, 0, s>>>(x, N, L[0], L[1], L[2], nc[0], nc[1], nc[2], rl2, nl.cell.p, nl.order_sorted.p,
                                             nl.cell_start.p, nl.cell_end.p, t.excl_ptr.p, t.excl_idx.p, nl.cap, nl.count.p,
                                             nl.list.p, nl.flag.p);
    MDL_CHECK_LAUNCH();
    int maxc = 0;
    MDL_CUDA(cudaMemcpyAsync(&maxc, nl.flag.p, sizeof(int), cudaMemcpyDeviceToHost, s));
    MDL_CUDA(cudaStreamSynchronize(s));
    if (maxc <= nl.cap) break;
    nl.cap = ((int)std::ceil(1.2 * maxc) + 31) / 32 * 32;
  }
  nl.xref.resize(3 * (size_t)N);
  MDL_CUDA(cudaMemcpyAsync(nl.xref.p, x, 3 * (size_t)N * sizeof(double), cudaMemcpyDeviceToDevice, s));
  ++nl.builds;
}

bool neighbor_needs_rebuild(NeighborList& nl, const double* x, int N, double skin, cudaStream_t s) {
  if (nl.cap == 0) return true;
  MDL_CUDA(cudaMemsetAsync(nl.flag.p + 1, 0, sizeof(int), s));
  double lim = 0.5 * skin;
  k_displacement<<<blocks_for(N, 256), 256, 0, s>>>(x, nl.xref.p, N, lim * lim, nl.flag.p);
  MDL_CHECK_LAUNCH();
  int f = 0;
  MDL_CUDA(cudaMemcpyAsync(&f, nl.flag.p + 1, sizeof(int), cudaMemcpyDeviceToHost, s));
  MDL_CUDA(cudaStreamSynchronize(s));
  return f != 0;
}

}  // namespace mdl
