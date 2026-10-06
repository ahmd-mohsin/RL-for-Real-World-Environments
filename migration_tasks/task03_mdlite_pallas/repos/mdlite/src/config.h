// mdlite run configuration ("key = value" text, '#' and ';' start comments).
#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace mdl {

enum class Integrator { VV, BAOAB };
enum class VdwModifier { PotentialShift, ForceSwitch };
enum class Coulomb { PME, ReactionField, Cutoff };
enum class Tcoupl { None, CSVR };
enum class Pcoupl { None, Berendsen, CRescale };

struct Config {
  Integrator integrator = Integrator::VV;
  double dt = 0.002;
  long long nsteps = 0;
  uint64_t seed = 1;

  double cutoff = 1.0;  // nonbonded cut-off (LJ and Coulomb), nm
  double skin = 0.1;    // Verlet buffer, nm
  VdwModifier vdw_modifier = VdwModifier::PotentialShift;
  double rvdw_switch = 0.0;
  bool dispcorr = false;

  Coulomb coulombtype = Coulomb::PME;
  double epsilon_rf = 1.0;  // 0 means infinity
  double ewald_rtol = 1e-5;
  double fourier_spacing = 0.12;
  int pme_order = 4;

  Tcoupl tcoupl = Tcoupl::None;
  double ref_t = 300.0;
  double tau_t = 1.0;
  int nsttcouple = 10;

  Pcoupl pcoupl = Pcoupl::None;
  double ref_p = 1.0;
  double tau_p = 2.0;
  double compressibility = 4.5e-5;
  int nstpcouple = 10;

  int nstcomm = 100;
  bool constraints = true;
  double shake_tol = 1e-10;
  int shake_maxiter = 1000;

  bool gen_vel = false;
  double gen_temp = 300.0;

  int nstenergy = 100;
  int nstxout = 0;
  int nstvout = 0;
  int nstcheckpoint = 0;

  std::map<std::string, std::string> raw;  // as given, for the log

  static Config parse(const std::string& text);
  void validate() const;
};

}  // namespace mdl
