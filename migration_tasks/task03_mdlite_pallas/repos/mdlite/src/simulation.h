// mdlite simulation driver.
#pragma once

#include <cstdio>
#include <memory>
#include <string>

#include "engine.cuh"

namespace mdl {

// energy terms, in this order, are reported by the C API and the energy file
enum Term : int {
  T_LJ = 0, T_LJ14, T_DISPCORR, T_COUL_SR, T_COUL14, T_COUL_RECIP, T_BOND, T_ANGLE, T_DIHEDRAL,
  T_POTENTIAL, T_KINETIC, T_TOTAL, T_CONSERVED, T_TEMPERATURE, T_PRESSURE, T_VOLUME,
  T_COUNT
};
extern const char* const TERM_NAMES[T_COUNT];

struct Energies {
  double term[T_COUNT];
  double virial[9];    // r (x) f, kJ/mol
  double pressure[9];  // bar
};

class Simulation {
 public:
  Simulation(const System& sys, const Config& cfg, const std::string& out_dir);
  ~Simulation();

  void run(long long nsteps);  // integrate; writes outputs when an output directory was given
  void finish_outputs();       // write the outputs of the current step (if due) and flush files
  const Energies& energies();  // at the current state
  void forces(double* out);    // 3N, kJ/mol/nm

  void get_positions(double* out) const;
  void get_velocities(double* out) const;
  void set_positions(const double* in);
  void set_velocities(const double* in);
  void set_box(const double L[3]);
  const double* box() const { return L_; }
  long long step() const { return step_; }
  int natoms() const { return sys_.natoms; }
  int dof() const { return dof_; }

  void save_checkpoint(const std::string& path) const;
  void load_checkpoint(const std::string& path);
  std::string info_json() const;

 private:
  void setup_device();
  void check_box() const;
  void compute_forces();
  void update_neighbors(bool force);
  void kinetic_energy(double ke6[6], double mom[3]);
  void step_vv();
  void step_baoab();
  void thermostat_csvr();
  void barostat();
  void remove_com_motion();
  void constrain_positions(const double* xref, double inv_dt, bool fix_velocities);
  void constrain_velocities(double* vir6);
  void write_outputs(bool final_call);
  void write_energy_row(const Energies& e);
  void write_frame();

  System sys_;
  Config cfg_;
  Topology topo_;
  DeviceTopology dt_;
  NbParams nb_{};
  NeighborList nl_;
  std::unique_ptr<Pme> pme_;
  DevBuf<double> x_, v_, xold_;
  DevBuf<long long> f_, acc_;
  cudaStream_t s_ = 0;

  double L_[3];
  long long step_ = 0;
  double integral_ = 0;     // energy removed by thermostat / barostat velocity scaling
  double cvir_[6] = {0, 0, 0, 0, 0, 0};  // constraint virial of the latest velocity constraint
  int dof_ = 0;
  double self_energy_ = 0;  // PME self + net-charge terms / RF self term (box dependent part recomputed)
  double sum_q2_ = 0, sum_q_ = 0;
  double avg_c6_ = 0;       // <C6> over all atom pairs for the dispersion correction
  bool forces_valid_ = false;
  bool box_changed_ = true;
  double e_recip_ = 0, vir_recip_[6] = {0, 0, 0, 0, 0, 0};
  Energies cache_{};
  bool cache_valid_ = false;

  std::string out_dir_;
  FILE* energy_file_ = nullptr;
  FILE* traj_file_ = nullptr;
  long long last_output_step_ = -1;
};

double ewald_coefficient(double rc, double rtol);
int pme_grid_size(double L, double spacing, int order);

}  // namespace mdl
