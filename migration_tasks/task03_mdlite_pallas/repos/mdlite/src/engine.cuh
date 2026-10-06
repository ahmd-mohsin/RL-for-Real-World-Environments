// mdlite device state and the GPU modules that act on it.
#pragma once

#include <cufft.h>

#include "common.cuh"
#include "config.h"
#include "system.h"

namespace mdl {

// energy / virial accumulator slots (fixed point, see common.cuh)
enum Acc : int {
  // pair sums over the full neighbour list: every pair is counted twice, halved at read-out
  A_LJ2 = 0,
  A_COUL2,
  A_V2XX, A_V2YY, A_V2ZZ, A_V2XY, A_V2XZ, A_V2YZ,
  // single-counted sums
  A_LJ14,
  A_COUL14,
  A_COUL_EXCL,  // exclusion corrections (PME: -erf(beta r)/r, reaction-field: k_rf r^2 - c_rf)
  A_BOND,
  A_ANGLE,
  A_DIH,
  A_VXX, A_VYY, A_VZZ, A_VXY, A_VXZ, A_VYZ,  // virial sum r_ij (x) f_ij
  A_COUNT
};

struct NbParams {
  double rc2;
  int vdw_mod;  // 0 potential-shift, 1 force-switch
  double sh6, sh12;  // potential-shift constants 1/rc^6, 1/rc^12
  double rsw;
  double fsw_A6, fsw_B6, fsw_C6, fsw_A12, fsw_B12, fsw_C12;
  int coul;  // 0 pme, 1 reaction-field (also plain cut-off)
  double beta;
  double krf, crf;
  double epsfac;
};

struct DeviceTopology {
  int N = 0, ntypes = 0;
  DevBuf<int> type;
  DevBuf<double> charge, inv_mass, mass;
  DevBuf<double> c6, c12;
  DevBuf<int> excl_ptr, excl_idx, excl_pairs;  // excl_pairs: 2 ints per pair
  int nexcl_pairs = 0;
  DevBuf<int> bond_idx;  DevBuf<double> bond_par;  int nbonds = 0;   // (i,j), (r0,k)
  DevBuf<int> angle_idx; DevBuf<double> angle_par; int nangles = 0;  // (i,j,k), (theta0 rad, k)
  DevBuf<int> dih_idx;   DevBuf<double> dih_par;   int ndih = 0;     // (i,j,k,l,mult), (phi0 rad, k)
  DevBuf<int> pair_idx;  int npairs = 0;
  DevBuf<int> con_idx;   DevBuf<double> con_d;     int ncons = 0;
  DevBuf<int> cl_ptr, cl_cons; int nclusters = 0;
  DevBuf<int> mol_ptr, mol_atoms; int nmol = 0;
  double fudge_lj = 1, fudge_qq = 1;
};

struct NeighborList {
  double rlist = 0;
  int cap = 0;      // neighbours per atom
  int nc[3] = {0, 0, 0};
  DevBuf<int> cell, order, cell_sorted, order_sorted, cell_start, cell_end;
  DevBuf<unsigned char> sort_tmp;
  DevBuf<int> count, list;
  DevBuf<double> xref;
  DevBuf<int> flag;  // [0] max count, [1] rebuild needed
  long long builds = 0;
};

struct Pme {
  int n[3] = {0, 0, 0};
  int order = 4;
  double beta = 0;
  DevBuf<long long> qgrid_fixed;
  DevBuf<double> grid;
  DevBuf<cufftDoubleComplex> kgrid;
  DevBuf<double> bmod[3];
  DevBuf<double> red;  // per-block energy and virial partial sums
  cufftHandle fwd = 0, bwd = 0;
  ~Pme();
};

// --- module entry points (implemented in the .cu files)
void neighbor_build(NeighborList& nl, const DeviceTopology& t, const double* x, const double L[3], cudaStream_t s);
bool neighbor_needs_rebuild(NeighborList& nl, const double* x, int N, double skin, cudaStream_t s);
void nonbonded_forces(const NeighborList& nl, const DeviceTopology& t, const NbParams& p, const double* x, const double L[3],
                      long long* f, long long* acc, cudaStream_t s);
void exclusion_corrections(const DeviceTopology& t, const NbParams& p, double cutoff, const double* x, const double L[3],
                           long long* f, long long* acc, cudaStream_t s);
void bonded_forces(const DeviceTopology& t, const double* x, const double L[3], long long* f, long long* acc, cudaStream_t s);
void pme_setup(Pme& pme, int order, const int n[3], double beta);
// returns reciprocal energy; adds the reciprocal virial (xx,yy,zz,xy,xz,yz) to vir6
double pme_forces(Pme& pme, const DeviceTopology& t, const double* x, const double L[3], long long* f, double vir6[6], cudaStream_t s);

// constraints; return false on non-convergence. vir6 receives the constraint virial (velocity stage only)
bool shake(const DeviceTopology& t, double* x, const double* xref, double* v, double inv_dt, double tol, int maxit,
           const double L[3], cudaStream_t s);
bool rattle(const DeviceTopology& t, const double* x, double* v, double dt, double tol, int maxit, const double L[3],
            double vir6[6], cudaStream_t s);

// integrator kernels
void kick(double* v, const long long* f, const double* inv_mass, double half_dt, int N, cudaStream_t s);
void drift(double* x, const double* v, double h, int N, cudaStream_t s);
void langevin_o(double* v, const double* inv_mass, double c1, double c2, double kT, uint64_t seed, uint64_t step, int N, cudaStream_t s);
void scale_velocities(double* v, double lambda, int N, cudaStream_t s);
// kinetic energy tensor diagonal + off-diagonal (6) and total momentum (3) and total mass, deterministic order
void kinetic(const double* v, const double* mass, int N, double ke6[6], double mom[3], cudaStream_t s);
void remove_com(double* v, const double com_v[3], int N, cudaStream_t s);
void scale_molecules(const DeviceTopology& t, double* x, double mu, const double L_old[3], cudaStream_t s);
void gen_velocities(double* v, const double* inv_mass, double kT, uint64_t seed, int N, cudaStream_t s);

// deterministic double reductions
double reduce_sum(const double* a, long long n, cudaStream_t s);

}  // namespace mdl
