#include "simulation.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include "crc32.h"
#include "rng.cuh"

namespace mdl {

const char* const TERM_NAMES[T_COUNT] = {"lj",        "lj_14",     "disp_corr", "coulomb_sr",  "coulomb_14", "coulomb_recip",
                                         "bond",      "angle",     "dihedral",  "potential",   "kinetic",    "total",
                                         "conserved", "temperature", "pressure", "volume"};

static const char CPT_MAGIC[8] = {'M', 'D', 'C', 'P', 'T', 0, 2, 0};
static const char TRJ_MAGIC[8] = {'M', 'D', 'T', 'R', 'J', 0, 1, 0};
constexpr double FLEXIBLE_CONSTRAINT_K = 400000.0;  // kJ/mol/nm^2, used when constraints = no

// GROMACS calc_ewaldcoeff_q: erfc(beta * rc) = rtol, by doubling then 60 bisection steps
double ewald_coefficient(double rc, double rtol) {
  double beta = 5.0, low, high;
  int i = 0;
  while (std::erfc(beta * rc) > rtol) {
    ++i;
    beta *= 2.0;
  }
  int n = i + 60;
  low = 0.0;
  high = beta;
  for (i = 0; i < n; ++i) {
    beta = 0.5 * (low + high);
    if (std::erfc(beta * rc) > rtol) low = beta;
    else high = beta;
  }
  return beta;
}

// smallest even n >= max(ceil(L / spacing), 2 * order) whose prime factors are all in {2, 3, 5, 7}
int pme_grid_size(double L, double spacing, int order) {
  int n = std::max((int)std::ceil(L / spacing), 2 * order);
  for (;; ++n) {
    if (n % 2) continue;
    int m = n;
    for (int p : {2, 3, 5, 7})
      while (m % p == 0) m /= p;
    if (m == 1) return n;
  }
}

Simulation::Simulation(const System& sys_in, const Config& cfg, const std::string& out_dir) : sys_(sys_in), cfg_(cfg), out_dir_(out_dir) {
  if (!cfg_.constraints) {
    for (auto& c : sys_.constraints) sys_.bonds.push_back(Bond{c.i, c.j, c.d, FLEXIBLE_CONSTRAINT_K});
    // exclusions are generated from the bond graph, so they do not change
    sys_.constraints.clear();
  }
  topo_ = Topology::build(sys_, true);
  for (int d = 0; d < 3; ++d) L_[d] = sys_.box[d];
  int N = sys_.natoms;
  dof_ = 3 * N - topo_.ncons - (cfg_.nstcomm > 0 ? 3 : 0);
  if (dof_ < 0) dof_ = 0;

  // nonbonded parameters
  double rc = cfg_.cutoff;
  nb_.rc2 = rc * rc;
  nb_.epsfac = ONE_4PI_EPS0;
  nb_.vdw_mod = cfg_.vdw_modifier == VdwModifier::PotentialShift ? 0 : 1;
  nb_.sh6 = std::pow(rc, -6.0);
  nb_.sh12 = std::pow(rc, -12.0);
  nb_.rsw = cfg_.rvdw_switch;
  if (nb_.vdw_mod == 1) {
    double r1 = cfg_.rvdw_switch, dr = rc - r1;
    auto coeffs = [&](double p, double& A, double& B, double& C) {
      A = -p * ((p + 4) * rc - (p + 1) * r1) / (std::pow(rc, p + 2) * dr * dr);
      B = p * ((p + 3) * rc - (p + 1) * r1) / (std::pow(rc, p + 2) * dr * dr * dr);
      C = 1.0 / std::pow(rc, p) - A / 3.0 * dr * dr * dr - B / 4.0 * dr * dr * dr * dr;
    };
    coeffs(6, nb_.fsw_A6, nb_.fsw_B6, nb_.fsw_C6);
    coeffs(12, nb_.fsw_A12, nb_.fsw_B12, nb_.fsw_C12);
  }
  for (int i = 0; i < N; ++i) {
    sum_q_ += sys_.charge[i];
    sum_q2_ += sys_.charge[i] * sys_.charge[i];
  }
  if (cfg_.coulombtype == Coulomb::PME) {
    nb_.coul = 0;
    nb_.beta = ewald_coefficient(rc, cfg_.ewald_rtol);
  } else {
    nb_.coul = 1;
    double eps = cfg_.coulombtype == Coulomb::Cutoff ? 1.0 : cfg_.epsilon_rf;
    nb_.krf = eps == 0.0 ? 1.0 / (2.0 * rc * rc * rc) : (eps - 1.0) / ((2.0 * eps + 1.0) * rc * rc * rc);
    nb_.crf = 1.0 / rc + nb_.krf * rc * rc;
  }
  if (cfg_.dispcorr) {
    std::vector<double> nt(sys_.ntypes, 0.0);
    for (int i = 0; i < N; ++i) nt[sys_.type[i]] += 1.0;
    double s = 0;
    for (int a = 0; a < sys_.ntypes; ++a)
      for (int b = 0; b < sys_.ntypes; ++b) s += nt[a] * nt[b] * topo_.c6[a * sys_.ntypes + b];
    avg_c6_ = s / ((double)N * (double)N);
  }
  nl_.rlist = rc + cfg_.skin;
  check_box();
  setup_device();

  if (cfg_.coulombtype == Coulomb::PME) {
    pme_ = std::make_unique<Pme>();
    int n[3];
    for (int d = 0; d < 3; ++d) n[d] = pme_grid_size(L_[d], cfg_.fourier_spacing, cfg_.pme_order);
    pme_setup(*pme_, cfg_.pme_order, n, nb_.beta);
  }

  // initial state
  x_.upload(sys_.x);
  std::vector<double> v0 = sys_.v.empty() ? std::vector<double>(3 * (size_t)N, 0.0) : sys_.v;
  v_.upload(v0);
  xold_.resize(3 * (size_t)N);
  f_.resize(3 * (size_t)N);
  acc_.resize(A_COUNT);
  if (cfg_.gen_vel) gen_velocities(v_.p, dt_.inv_mass.p, BOLTZ * cfg_.gen_temp, cfg_.seed, N, s_);
  if (dt_.nclusters) {
    MDL_CUDA(cudaMemcpyAsync(xold_.p, x_.p, 3 * (size_t)N * sizeof(double), cudaMemcpyDeviceToDevice, s_));
    constrain_positions(xold_.p, 0.0, false);
  }
  if (cfg_.nstcomm > 0) remove_com_motion();
  double scratch[6] = {0, 0, 0, 0, 0, 0};
  constrain_velocities(scratch);
  if (cfg_.gen_vel && dof_ > 0) {
    double ke6[6], mom[3];
    kinetic_energy(ke6, mom);
    double T = 2.0 * (ke6[0] + ke6[1] + ke6[2]) / (dof_ * BOLTZ);
    if (T > 0) scale_velocities(v_.p, std::sqrt(cfg_.gen_temp / T), N, s_);
  }
  compute_forces();
  // constraint virial at step 0: velocity constraint of the first half kick, without keeping the velocities
  if (dt_.nclusters) {
    DevBuf<double> vtmp(3 * (size_t)N);
    MDL_CUDA(cudaMemcpyAsync(vtmp.p, v_.p, 3 * (size_t)N * sizeof(double), cudaMemcpyDeviceToDevice, s_));
    kick(vtmp.p, f_.p, dt_.inv_mass.p, 0.5 * cfg_.dt, N, s_);
    for (double& c : cvir_) c = 0;
    if (!rattle(dt_, x_.p, vtmp.p, cfg_.dt, cfg_.shake_tol, cfg_.shake_maxiter, L_, cvir_, s_))
      throw Error("RATTLE did not converge at step 0");
  }

  if (!out_dir_.empty()) {
    std::string ep = out_dir_ + "/energy.csv";
    energy_file_ = std::fopen(ep.c_str(), "w");
    if (!energy_file_) throw Error("cannot open '" + ep + "'");
    std::fprintf(energy_file_, "step,time");
    for (int k = 0; k < T_COUNT; ++k) std::fprintf(energy_file_, ",%s", TERM_NAMES[k]);
    std::fprintf(energy_file_, ",box_x,box_y,box_z\n");
    if (cfg_.nstxout > 0) {
      std::string tp = out_dir_ + "/traj.mdtrj";
      traj_file_ = std::fopen(tp.c_str(), "wb");
      if (!traj_file_) throw Error("cannot open '" + tp + "'");
      int32_t hdr[2] = {N, cfg_.nstvout > 0 ? 1 : 0};
      int32_t tail[2] = {cfg_.nstxout, 0};
      std::fwrite(TRJ_MAGIC, 1, 8, traj_file_);
      std::fwrite(hdr, 4, 2, traj_file_);
      std::fwrite(&cfg_.dt, 8, 1, traj_file_);
      std::fwrite(tail, 4, 2, traj_file_);
    }
  }
}

Simulation::~Simulation() {
  if (energy_file_) std::fclose(energy_file_);
  if (traj_file_) std::fclose(traj_file_);
}

void Simulation::setup_device() {
  const System& s = sys_;
  const Topology& t = topo_;
  int N = s.natoms;
  dt_.N = N;
  dt_.ntypes = s.ntypes;
  dt_.type.upload(s.type);
  dt_.charge.upload(s.charge);
  dt_.mass.upload(s.mass);
  std::vector<double> im(N);
  for (int i = 0; i < N; ++i) im[i] = 1.0 / s.mass[i];
  dt_.inv_mass.upload(im);
  dt_.c6.upload(t.c6);
  dt_.c12.upload(t.c12);
  dt_.excl_ptr.upload(t.excl_ptr);
  std::vector<int> ei = t.excl_idx;
  if (ei.empty()) ei.push_back(0);
  dt_.excl_idx.upload(ei);
  std::vector<int> ep;
  for (auto& p : t.excl_pairs) { ep.push_back(p[0]); ep.push_back(p[1]); }
  dt_.nexcl_pairs = (int)t.excl_pairs.size();
  if (!ep.empty()) dt_.excl_pairs.upload(ep);

  std::vector<int> bi; std::vector<double> bp;
  for (auto& b : s.bonds) { bi.push_back(b.i); bi.push_back(b.j); bp.push_back(b.r0); bp.push_back(b.k); }
  dt_.nbonds = (int)s.bonds.size();
  if (dt_.nbonds) { dt_.bond_idx.upload(bi); dt_.bond_par.upload(bp); }
  std::vector<int> ai; std::vector<double> ap;
  for (auto& a : s.angles) { ai.push_back(a.i); ai.push_back(a.j); ai.push_back(a.k); ap.push_back(a.theta0 * PI / 180.0); ap.push_back(a.kth); }
  dt_.nangles = (int)s.angles.size();
  if (dt_.nangles) { dt_.angle_idx.upload(ai); dt_.angle_par.upload(ap); }
  std::vector<int> di; std::vector<double> dp;
  for (auto& d : s.dihedrals) {
    di.push_back(d.i); di.push_back(d.j); di.push_back(d.k); di.push_back(d.l); di.push_back(d.mult);
    dp.push_back(d.phi0 * PI / 180.0); dp.push_back(d.kphi);
  }
  dt_.ndih = (int)s.dihedrals.size();
  if (dt_.ndih) { dt_.dih_idx.upload(di); dt_.dih_par.upload(dp); }
  std::vector<int> pi;
  for (auto& p : s.pairs) { pi.push_back(p.i); pi.push_back(p.j); }
  dt_.npairs = (int)s.pairs.size();
  if (dt_.npairs) dt_.pair_idx.upload(pi);
  dt_.fudge_lj = s.fudge_lj;
  dt_.fudge_qq = s.fudge_qq;
  std::vector<int> ci; std::vector<double> cd;
  for (auto& c : s.constraints) { ci.push_back(c.i); ci.push_back(c.j); cd.push_back(c.d); }
  dt_.ncons = (int)s.constraints.size();
  if (dt_.ncons) { dt_.con_idx.upload(ci); dt_.con_d.upload(cd); }
  dt_.nclusters = (int)t.cl_ptr.size() - 1;
  if (dt_.nclusters) { dt_.cl_ptr.upload(t.cl_ptr); dt_.cl_cons.upload(t.cl_cons); }
  dt_.nmol = t.nmol;
  dt_.mol_ptr.upload(t.mol_ptr);
  dt_.mol_atoms.upload(t.mol_atoms);
}

void Simulation::check_box() const {
  for (int d = 0; d < 3; ++d)
    if (!(L_[d] >= 2.0 * nl_.rlist))
      throw Error("box length " + std::to_string(L_[d]) + " nm is smaller than twice the cut-off plus skin (" +
                  std::to_string(2.0 * nl_.rlist) + " nm)");
}

void Simulation::update_neighbors(bool force) {
  if (force || box_changed_ || neighbor_needs_rebuild(nl_, x_.p, sys_.natoms, cfg_.skin, s_)) {
    check_box();
    neighbor_build(nl_, dt_, x_.p, L_, s_);
    box_changed_ = false;
  }
}

void Simulation::compute_forces() {
  update_neighbors(false);
  f_.zero(s_);
  acc_.zero(s_);
  nonbonded_forces(nl_, dt_, nb_, x_.p, L_, f_.p, acc_.p, s_);
  exclusion_corrections(dt_, nb_, cfg_.cutoff, x_.p, L_, f_.p, acc_.p, s_);
  bonded_forces(dt_, x_.p, L_, f_.p, acc_.p, s_);
  for (double& v : vir_recip_) v = 0;
  e_recip_ = pme_ ? pme_forces(*pme_, dt_, x_.p, L_, f_.p, vir_recip_, s_) : 0.0;
  forces_valid_ = true;
  cache_valid_ = false;
}

void Simulation::kinetic_energy(double ke6[6], double mom[3]) { kinetic(v_.p, dt_.mass.p, sys_.natoms, ke6, mom, s_); }

const Energies& Simulation::energies() {
  if (!forces_valid_) compute_forces();
  if (cache_valid_) return cache_;
  std::vector<long long> a(A_COUNT);
  MDL_CUDA(cudaMemcpyAsync(a.data(), acc_.p, A_COUNT * sizeof(long long), cudaMemcpyDeviceToHost, s_));
  MDL_CUDA(cudaStreamSynchronize(s_));
  auto E = [&](int k) { return from_fixed(a[k], ENERGY_SCALE); };
  auto Vr = [&](int k) { return from_fixed(a[k], VIRIAL_SCALE); };
  Energies& e = cache_;
  double V = L_[0] * L_[1] * L_[2];
  for (double& t : e.term) t = 0;
  e.term[T_LJ] = 0.5 * E(A_LJ2);
  e.term[T_LJ14] = E(A_LJ14);
  e.term[T_COUL14] = E(A_COUL14);
  e.term[T_BOND] = E(A_BOND);
  e.term[T_ANGLE] = E(A_ANGLE);
  e.term[T_DIHEDRAL] = E(A_DIH);
  e.term[T_COUL_SR] = 0.5 * E(A_COUL2);
  double disp = 0;
  if (cfg_.dispcorr) {
    double rc = cfg_.cutoff;
    disp = -2.0 / 3.0 * PI * (double)sys_.natoms * (double)sys_.natoms * avg_c6_ / (V * rc * rc * rc);
  }
  e.term[T_DISPCORR] = disp;
  if (nb_.coul == 0) {
    double self = -ONE_4PI_EPS0 * nb_.beta / std::sqrt(PI) * sum_q2_;
    double net = -ONE_4PI_EPS0 * PI * sum_q_ * sum_q_ / (2.0 * V * nb_.beta * nb_.beta);
    e.term[T_COUL_RECIP] = e_recip_ + E(A_COUL_EXCL) + self + net;
  } else {
    e.term[T_COUL_SR] += E(A_COUL_EXCL) - 0.5 * ONE_4PI_EPS0 * nb_.crf * sum_q2_;
  }
  double pot = 0;
  for (int k = T_LJ; k <= T_DIHEDRAL; ++k) pot += e.term[k];
  e.term[T_POTENTIAL] = pot;

  // virial: (xx, yy, zz, xy, xz, yz)
  double w[6];
  for (int c = 0; c < 6; ++c) w[c] = 0.5 * Vr(A_V2XX + c) + Vr(A_VXX + c) + vir_recip_[c] + cvir_[c];
  // net-charge term: E ~ 1/V, isotropic virial E per diagonal component
  if (nb_.coul == 0) {
    double net = -ONE_4PI_EPS0 * PI * sum_q_ * sum_q_ / (2.0 * V * nb_.beta * nb_.beta);
    for (int c = 0; c < 3; ++c) w[c] += net;
  }
  for (int c = 0; c < 3; ++c) w[c] += 2.0 * disp;
  double ke6[6], mom[3];
  kinetic_energy(ke6, mom);
  double ke = ke6[0] + ke6[1] + ke6[2];
  e.term[T_KINETIC] = ke;
  e.term[T_TOTAL] = pot + ke;
  e.term[T_TEMPERATURE] = dof_ > 0 ? 2.0 * ke / (dof_ * BOLTZ) : 0.0;
  e.term[T_VOLUME] = V;
  const int map[9] = {0, 3, 4, 3, 1, 5, 4, 5, 2};
  for (int k = 0; k < 9; ++k) {
    e.virial[k] = w[map[k]];
    e.pressure[k] = (2.0 * ke6[map[k]] + w[map[k]]) / V * PRESFAC;
  }
  e.term[T_PRESSURE] = (e.pressure[0] + e.pressure[4] + e.pressure[8]) / 3.0;
  double cons = e.term[T_TOTAL] + integral_;
  if (cfg_.pcoupl != Pcoupl::None) cons += cfg_.ref_p * V / PRESFAC;
  e.term[T_CONSERVED] = cons;
  cache_valid_ = true;
  return cache_;
}

void Simulation::forces(double* out) {
  if (!forces_valid_) compute_forces();
  std::vector<long long> f(3 * (size_t)sys_.natoms);
  MDL_CUDA(cudaMemcpyAsync(f.data(), f_.p, f.size() * sizeof(long long), cudaMemcpyDeviceToHost, s_));
  MDL_CUDA(cudaStreamSynchronize(s_));
  for (size_t k = 0; k < f.size(); ++k) out[k] = from_fixed(f[k], FORCE_SCALE);
}

void Simulation::constrain_positions(const double* xref, double inv_dt, bool fix_velocities) {
  if (!dt_.nclusters) return;
  if (!shake(dt_, x_.p, xref, fix_velocities ? v_.p : nullptr, inv_dt, cfg_.shake_tol, cfg_.shake_maxiter, L_, s_))
    throw Error("SHAKE did not converge at step " + std::to_string(step_));
}

void Simulation::constrain_velocities(double* vir6) {
  if (!dt_.nclusters) return;
  if (!rattle(dt_, x_.p, v_.p, cfg_.dt, cfg_.shake_tol, cfg_.shake_maxiter, L_, vir6, s_))
    throw Error("RATTLE did not converge at step " + std::to_string(step_));
}

void Simulation::remove_com_motion() {
  double ke6[6], mom[3];
  kinetic_energy(ke6, mom);
  double M = 0;
  for (double m : sys_.mass) M += m;
  double c[3] = {mom[0] / M, mom[1] / M, mom[2] / M};
  remove_com(v_.p, c, sys_.natoms, s_);
}

void Simulation::step_vv() {
  int N = sys_.natoms;
  double dt = cfg_.dt;
  kick(v_.p, f_.p, dt_.inv_mass.p, 0.5 * dt, N, s_);
  MDL_CUDA(cudaMemcpyAsync(xold_.p, x_.p, 3 * (size_t)N * sizeof(double), cudaMemcpyDeviceToDevice, s_));
  drift(x_.p, v_.p, dt, N, s_);
  constrain_positions(xold_.p, 1.0 / dt, true);
  ++step_;
  forces_valid_ = false;
  compute_forces();
  kick(v_.p, f_.p, dt_.inv_mass.p, 0.5 * dt, N, s_);
  if (dt_.nclusters) {
    for (double& c : cvir_) c = 0;
    constrain_velocities(cvir_);
  }
  if (cfg_.tcoupl == Tcoupl::CSVR && step_ % cfg_.nsttcouple == 0) thermostat_csvr();
  if (cfg_.nstcomm > 0 && step_ % cfg_.nstcomm == 0) remove_com_motion();
  cache_valid_ = false;
}

void Simulation::step_baoab() {
  int N = sys_.natoms;
  double dt = cfg_.dt;
  double scratch[6];
  kick(v_.p, f_.p, dt_.inv_mass.p, 0.5 * dt, N, s_);
  constrain_velocities(scratch);
  MDL_CUDA(cudaMemcpyAsync(xold_.p, x_.p, 3 * (size_t)N * sizeof(double), cudaMemcpyDeviceToDevice, s_));
  drift(x_.p, v_.p, 0.5 * dt, N, s_);
  constrain_positions(xold_.p, 2.0 / dt, true);
  // project the velocities onto the constraint manifold before measuring the heat exchanged by the O-step
  constrain_velocities(scratch);
  double ke0[6], ke1[6], mom[3];
  kinetic_energy(ke0, mom);
  double c1 = std::exp(-dt / cfg_.tau_t);
  double c2 = std::sqrt(1.0 - c1 * c1);
  langevin_o(v_.p, dt_.inv_mass.p, c1, c2, BOLTZ * cfg_.ref_t, cfg_.seed, (uint64_t)step_, N, s_);
  constrain_velocities(scratch);
  kinetic_energy(ke1, mom);
  integral_ -= (ke1[0] + ke1[1] + ke1[2]) - (ke0[0] + ke0[1] + ke0[2]);
  MDL_CUDA(cudaMemcpyAsync(xold_.p, x_.p, 3 * (size_t)N * sizeof(double), cudaMemcpyDeviceToDevice, s_));
  drift(x_.p, v_.p, 0.5 * dt, N, s_);
  constrain_positions(xold_.p, 2.0 / dt, true);
  ++step_;
  forces_valid_ = false;
  compute_forces();
  kick(v_.p, f_.p, dt_.inv_mass.p, 0.5 * dt, N, s_);
  if (dt_.nclusters) {
    for (double& c : cvir_) c = 0;
    constrain_velocities(cvir_);
  }
  if (cfg_.nstcomm > 0 && step_ % cfg_.nstcomm == 0) remove_com_motion();
  cache_valid_ = false;
}

// Bussi, Donadio, Parrinello, J. Chem. Phys. 126, 014101 (2007)
void Simulation::thermostat_csvr() {
  if (dof_ == 0) return;
  double ke6[6], mom[3];
  kinetic_energy(ke6, mom);
  double K = ke6[0] + ke6[1] + ke6[2];
  if (K <= 0) return;
  double Kt = 0.5 * dof_ * BOLTZ * cfg_.ref_t;
  double c = std::exp(-cfg_.nsttcouple * cfg_.dt / cfg_.tau_t);
  HostStream rs(cfg_.seed, (uint64_t)step_, STREAM_CSVR);
  double r1 = rs.gauss();
  double sumr2 = rs.sum_noises(dof_ - 1);
  double Knew = K + (1.0 - c) * (Kt * (sumr2 + r1 * r1) / dof_ - K) + 2.0 * r1 * std::sqrt(Kt / dof_ * (1.0 - c) * c * K);
  if (Knew < 0) Knew = 0;
  scale_velocities(v_.p, std::sqrt(Knew / K), sys_.natoms, s_);
  integral_ -= Knew - K;
}

void Simulation::barostat() {
  const Energies& e = energies();
  double P = e.term[T_PRESSURE];
  double V = L_[0] * L_[1] * L_[2];
  double cf = cfg_.compressibility * cfg_.nstpcouple * cfg_.dt / cfg_.tau_p;
  double mu;
  if (cfg_.pcoupl == Pcoupl::Berendsen) {
    mu = std::cbrt(1.0 - cf * (cfg_.ref_p - P));
  } else {
    // Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020), isotropic
    double kT_bar = BOLTZ * cfg_.ref_t * PRESFAC / V;
    HostStream rs(cfg_.seed, (uint64_t)step_, STREAM_CRESCALE);
    double g = rs.gauss();
    double deps = -cf * (cfg_.ref_p - P - kT_bar) + std::sqrt(2.0 * kT_bar * cf) * g;
    mu = std::exp(deps / 3.0);
  }
  double Lold[3] = {L_[0], L_[1], L_[2]};
  scale_molecules(dt_, x_.p, mu, Lold, s_);
  for (double& L : L_) L *= mu;
  if (cfg_.pcoupl == Pcoupl::CRescale) {
    double K0 = e.term[T_KINETIC];
    scale_velocities(v_.p, 1.0 / mu, sys_.natoms, s_);
    integral_ -= K0 / (mu * mu) - K0;
  }
  box_changed_ = true;
  forces_valid_ = false;
  cache_valid_ = false;
  compute_forces();
}

void Simulation::run(long long nsteps) {
  for (long long k = 0; k < nsteps; ++k) {
    write_outputs(false);
    if (cfg_.pcoupl != Pcoupl::None && step_ % cfg_.nstpcouple == 0) barostat();
    if (cfg_.integrator == Integrator::VV) step_vv();
    else step_baoab();
  }
  write_outputs(true);
}

void Simulation::finish_outputs() { write_outputs(true); }

void Simulation::write_outputs(bool final_call) {
  if (out_dir_.empty() || last_output_step_ == step_) {
    if (final_call && energy_file_) std::fflush(energy_file_);
    return;
  }
  last_output_step_ = step_;
  if (cfg_.nstenergy > 0 && step_ % cfg_.nstenergy == 0) write_energy_row(energies());
  if (traj_file_ && step_ % cfg_.nstxout == 0) write_frame();
  if (cfg_.nstcheckpoint > 0 && step_ % cfg_.nstcheckpoint == 0)
    save_checkpoint(out_dir_ + "/state_" + std::to_string(step_) + ".mdcpt");
  if (final_call) {
    if (energy_file_) std::fflush(energy_file_);
    if (traj_file_) std::fflush(traj_file_);
  }
}

void Simulation::write_energy_row(const Energies& e) {
  std::fprintf(energy_file_, "%lld,%.6f", step_, step_ * cfg_.dt);
  for (int k = 0; k < T_COUNT; ++k) std::fprintf(energy_file_, ",%.12e", e.term[k]);
  std::fprintf(energy_file_, ",%.12e,%.12e,%.12e\n", L_[0], L_[1], L_[2]);
}

void Simulation::write_frame() {
  int N = sys_.natoms;
  std::vector<double> x(3 * (size_t)N);
  get_positions(x.data());
  std::vector<float> xf(x.begin(), x.end());
  int64_t st = step_;
  double t = step_ * cfg_.dt;
  std::fwrite(&st, 8, 1, traj_file_);
  std::fwrite(&t, 8, 1, traj_file_);
  std::fwrite(L_, 8, 3, traj_file_);
  std::fwrite(xf.data(), 4, xf.size(), traj_file_);
  if (cfg_.nstvout > 0) {
    get_velocities(x.data());
    std::vector<float> vf(x.begin(), x.end());
    std::fwrite(vf.data(), 4, vf.size(), traj_file_);
  }
}

void Simulation::get_positions(double* out) const {
  MDL_CUDA(cudaMemcpy(out, x_.p, 3 * (size_t)sys_.natoms * sizeof(double), cudaMemcpyDeviceToHost));
}
void Simulation::get_velocities(double* out) const {
  MDL_CUDA(cudaMemcpy(out, v_.p, 3 * (size_t)sys_.natoms * sizeof(double), cudaMemcpyDeviceToHost));
}
void Simulation::set_positions(const double* in) {
  for (size_t k = 0; k < 3 * (size_t)sys_.natoms; ++k)
    if (!std::isfinite(in[k])) throw Error("set_positions: non-finite value");
  MDL_CUDA(cudaMemcpy(x_.p, in, 3 * (size_t)sys_.natoms * sizeof(double), cudaMemcpyHostToDevice));
  box_changed_ = true;
  forces_valid_ = false;
  cache_valid_ = false;
}
void Simulation::set_velocities(const double* in) {
  for (size_t k = 0; k < 3 * (size_t)sys_.natoms; ++k)
    if (!std::isfinite(in[k])) throw Error("set_velocities: non-finite value");
  MDL_CUDA(cudaMemcpy(v_.p, in, 3 * (size_t)sys_.natoms * sizeof(double), cudaMemcpyHostToDevice));
  cache_valid_ = false;
}
void Simulation::set_box(const double L[3]) {
  double old[3] = {L_[0], L_[1], L_[2]};
  for (int d = 0; d < 3; ++d) L_[d] = L[d];
  try {
    for (int d = 0; d < 3; ++d)
      if (!(std::isfinite(L[d]) && L[d] > 0)) throw Error("set_box: box lengths must be positive");
    check_box();
  } catch (...) {
    for (int d = 0; d < 3; ++d) L_[d] = old[d];
    throw;
  }
  box_changed_ = true;
  forces_valid_ = false;
  cache_valid_ = false;
}

void Simulation::save_checkpoint(const std::string& path) const {
  int N = sys_.natoms;
  std::vector<double> x(3 * (size_t)N), v(3 * (size_t)N);
  get_positions(x.data());
  get_velocities(v.data());
  std::string buf;
  auto put = [&](const void* p, size_t n) { buf.append(static_cast<const char*>(p), n); };
  put(CPT_MAGIC, 8);
  int64_t st = step_;
  put(&st, 8);
  int32_t hdr[2] = {N, 0};
  put(hdr, 8);
  put(L_, 24);
  put(&integral_, 8);
  put(cvir_, 48);
  uint64_t seed = cfg_.seed;
  put(&seed, 8);
  put(&cfg_.dt, 8);
  put(x.data(), x.size() * 8);
  put(v.data(), v.size() * 8);
  uint32_t crc = crc32(reinterpret_cast<const unsigned char*>(buf.data()), buf.size());
  put(&crc, 4);
  std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) throw Error("cannot write checkpoint '" + path + "'");
    f.write(buf.data(), (std::streamsize)buf.size());
    if (!f) throw Error("cannot write checkpoint '" + path + "'");
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) throw Error("cannot write checkpoint '" + path + "'");
}

void Simulation::load_checkpoint(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error("cannot open checkpoint '" + path + "'");
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  int N = sys_.natoms;
  size_t expect = 8 + 8 + 8 + 24 + 8 + 48 + 8 + 8 + 48 * (size_t)N + 4;
  if (buf.size() < 16 || std::memcmp(buf.data(), CPT_MAGIC, 8) != 0) throw Error("checkpoint: bad magic");
  if (buf.size() != expect) throw Error("checkpoint: size does not match the system");
  uint32_t crc;
  std::memcpy(&crc, buf.data() + buf.size() - 4, 4);
  if (crc32(reinterpret_cast<const unsigned char*>(buf.data()), buf.size() - 4) != crc) throw Error("checkpoint: checksum mismatch");
  const char* p = buf.data() + 8;
  auto get = [&](void* dst, size_t n) { std::memcpy(dst, p, n); p += n; };
  int64_t st;
  get(&st, 8);
  int32_t hdr[2];
  get(hdr, 8);
  if (hdr[0] != N) throw Error("checkpoint: atom count does not match the system");
  double L[3], integral, cvir[6], dt;
  uint64_t seed;
  get(L, 24);
  get(&integral, 8);
  get(cvir, 48);
  get(&seed, 8);
  get(&dt, 8);
  std::vector<double> x(3 * (size_t)N), v(3 * (size_t)N);
  get(x.data(), x.size() * 8);
  get(v.data(), v.size() * 8);
  double Lold[3] = {L_[0], L_[1], L_[2]};
  for (int d = 0; d < 3; ++d) L_[d] = L[d];
  try {
    check_box();
  } catch (...) {
    for (int d = 0; d < 3; ++d) L_[d] = Lold[d];
    throw;
  }
  step_ = st;
  integral_ = integral;
  std::memcpy(cvir_, cvir, sizeof(cvir_));
  MDL_CUDA(cudaMemcpy(x_.p, x.data(), x.size() * 8, cudaMemcpyHostToDevice));
  MDL_CUDA(cudaMemcpy(v_.p, v.data(), v.size() * 8, cudaMemcpyHostToDevice));
  box_changed_ = true;
  forces_valid_ = false;
  cache_valid_ = false;
  last_output_step_ = -1;
  compute_forces();
}

std::string Simulation::info_json() const {
  std::ostringstream o;
  o.precision(17);
  o << "{\"natoms\": " << sys_.natoms << ", \"dof\": " << dof_ << ", \"constraints\": " << topo_.ncons
    << ", \"constraint_clusters\": " << dt_.nclusters << ", \"exclusion_pairs\": " << dt_.nexcl_pairs
    << ", \"rlist\": " << nl_.rlist << ", \"neighbor_builds\": " << nl_.builds << ", \"neighbor_capacity\": " << nl_.cap
    << ", \"cells\": [" << nl_.nc[0] << ", " << nl_.nc[1] << ", " << nl_.nc[2] << "]";
  if (pme_)
    o << ", \"ewald_beta\": " << nb_.beta << ", \"pme_grid\": [" << pme_->n[0] << ", " << pme_->n[1] << ", " << pme_->n[2] << "]";
  else
    o << ", \"ewald_beta\": null, \"pme_grid\": null";
  if (nb_.coul == 1) o << ", \"k_rf\": " << nb_.krf << ", \"c_rf\": " << nb_.crf;
  o << "}";
  return o.str();
}

}  // namespace mdl
