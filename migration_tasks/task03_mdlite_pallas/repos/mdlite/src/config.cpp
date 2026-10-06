#include "config.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <sstream>

#include "common.cuh"

namespace mdl {
namespace {

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace((unsigned char)s[a])) ++a;
  while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
  return s.substr(a, b - a);
}

std::string lower(std::string s) {
  for (auto& c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

double to_double(const std::string& key, const std::string& v) {
  char* end = nullptr;
  errno = 0;
  double d = std::strtod(v.c_str(), &end);
  if (end == v.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(d))
    throw Error("config: '" + key + "' expects a number, got '" + v + "'");
  return d;
}

long long to_int(const std::string& key, const std::string& v) {
  char* end = nullptr;
  errno = 0;
  long long i = std::strtoll(v.c_str(), &end, 10);
  if (end == v.c_str() || *end != '\0' || errno == ERANGE)
    throw Error("config: '" + key + "' expects an integer, got '" + v + "'");
  return i;
}

bool to_bool(const std::string& key, const std::string& v) {
  std::string l = lower(v);
  if (l == "yes" || l == "true" || l == "on" || l == "1") return true;
  if (l == "no" || l == "false" || l == "off" || l == "0") return false;
  throw Error("config: '" + key + "' expects yes/no, got '" + v + "'");
}

template <class E>
E to_enum(const std::string& key, const std::string& v, std::initializer_list<std::pair<const char*, E>> opts) {
  std::string l = lower(v);
  for (auto& o : opts)
    if (l == o.first) return o.second;
  std::string msg = "config: '" + key + "' must be one of";
  for (auto& o : opts) msg += std::string(" ") + o.first;
  throw Error(msg + ", got '" + v + "'");
}

}  // namespace

Config Config::parse(const std::string& text) {
  Config c;
  std::istringstream in(text);
  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    size_t cut = line.find_first_of("#;");
    if (cut != std::string::npos) line = line.substr(0, cut);
    line = trim(line);
    if (line.empty()) continue;
    size_t eq = line.find('=');
    if (eq == std::string::npos) throw Error("config line " + std::to_string(lineno) + ": expected 'key = value'");
    std::string key = lower(trim(line.substr(0, eq)));
    std::string val = trim(line.substr(eq + 1));
    std::replace(key.begin(), key.end(), '-', '_');
    if (key.empty() || val.empty()) throw Error("config line " + std::to_string(lineno) + ": empty key or value");
    if (c.raw.count(key)) throw Error("config: duplicate key '" + key + "'");
    c.raw[key] = val;

    if (key == "integrator") c.integrator = to_enum<Integrator>(key, val, {{"vv", Integrator::VV}, {"baoab", Integrator::BAOAB}});
    else if (key == "dt") c.dt = to_double(key, val);
    else if (key == "nsteps") c.nsteps = to_int(key, val);
    else if (key == "seed") {
      long long s = to_int(key, val);
      if (s < 0) throw Error("config: 'seed' must be >= 0");
      c.seed = (uint64_t)s;
    }
    else if (key == "cutoff") c.cutoff = to_double(key, val);
    else if (key == "skin") c.skin = to_double(key, val);
    else if (key == "vdw_modifier") c.vdw_modifier = to_enum<VdwModifier>(key, val, {{"potential-shift", VdwModifier::PotentialShift}, {"force-switch", VdwModifier::ForceSwitch}});
    else if (key == "rvdw_switch") c.rvdw_switch = to_double(key, val);
    else if (key == "dispcorr") c.dispcorr = to_enum<bool>(key, val, {{"no", false}, {"enerpres", true}});
    else if (key == "coulombtype") c.coulombtype = to_enum<Coulomb>(key, val, {{"pme", Coulomb::PME}, {"reaction-field", Coulomb::ReactionField}, {"cutoff", Coulomb::Cutoff}});
    else if (key == "epsilon_rf") c.epsilon_rf = to_double(key, val);
    else if (key == "ewald_rtol") c.ewald_rtol = to_double(key, val);
    else if (key == "fourier_spacing") c.fourier_spacing = to_double(key, val);
    else if (key == "pme_order") c.pme_order = (int)to_int(key, val);
    else if (key == "tcoupl") c.tcoupl = to_enum<Tcoupl>(key, val, {{"no", Tcoupl::None}, {"csvr", Tcoupl::CSVR}, {"v-rescale", Tcoupl::CSVR}});
    else if (key == "ref_t") c.ref_t = to_double(key, val);
    else if (key == "tau_t") c.tau_t = to_double(key, val);
    else if (key == "nsttcouple") c.nsttcouple = (int)to_int(key, val);
    else if (key == "pcoupl") c.pcoupl = to_enum<Pcoupl>(key, val, {{"no", Pcoupl::None}, {"berendsen", Pcoupl::Berendsen}, {"c-rescale", Pcoupl::CRescale}});
    else if (key == "ref_p") c.ref_p = to_double(key, val);
    else if (key == "tau_p") c.tau_p = to_double(key, val);
    else if (key == "compressibility") c.compressibility = to_double(key, val);
    else if (key == "nstpcouple") c.nstpcouple = (int)to_int(key, val);
    else if (key == "nstcomm") c.nstcomm = (int)to_int(key, val);
    else if (key == "constraints") c.constraints = to_bool(key, val);
    else if (key == "shake_tol") c.shake_tol = to_double(key, val);
    else if (key == "shake_maxiter") c.shake_maxiter = (int)to_int(key, val);
    else if (key == "gen_vel") c.gen_vel = to_bool(key, val);
    else if (key == "gen_temp") c.gen_temp = to_double(key, val);
    else if (key == "nstenergy") c.nstenergy = (int)to_int(key, val);
    else if (key == "nstxout") c.nstxout = (int)to_int(key, val);
    else if (key == "nstvout") c.nstvout = (int)to_int(key, val);
    else if (key == "nstcheckpoint") c.nstcheckpoint = (int)to_int(key, val);
    else throw Error("config: unknown key '" + key + "'");
  }
  c.validate();
  return c;
}

void Config::validate() const {
  auto req = [](bool ok, const std::string& msg) {
    if (!ok) throw Error("config: " + msg);
  };
  req(dt > 0 && dt <= 0.01, "dt must be in (0, 0.01] ps");
  req(nsteps >= 0, "nsteps must be >= 0");
  req(cutoff > 0, "cutoff must be > 0");
  req(skin >= 0, "skin must be >= 0");
  if (vdw_modifier == VdwModifier::ForceSwitch)
    req(rvdw_switch >= 0 && rvdw_switch < cutoff, "rvdw_switch must be in [0, cutoff) for force-switch");
  req(epsilon_rf == 0 || epsilon_rf >= 1, "epsilon_rf must be 0 (infinity) or >= 1");
  req(ewald_rtol > 0 && ewald_rtol < 1, "ewald_rtol must be in (0, 1)");
  req(fourier_spacing > 0, "fourier_spacing must be > 0");
  req(pme_order >= 3 && pme_order <= 8, "pme_order must be in [3, 8]");
  req(ref_t >= 0, "ref_t must be >= 0");
  req(tau_t > 0, "tau_t must be > 0");
  req(nsttcouple >= 1, "nsttcouple must be >= 1");
  req(tau_p > 0, "tau_p must be > 0");
  req(compressibility >= 0, "compressibility must be >= 0");
  req(nstpcouple >= 1, "nstpcouple must be >= 1");
  req(nstcomm >= 0, "nstcomm must be >= 0");
  req(shake_tol > 0 && shake_tol < 1e-2, "shake_tol must be in (0, 0.01)");
  req(shake_maxiter >= 1, "shake_maxiter must be >= 1");
  req(gen_temp >= 0, "gen_temp must be >= 0");
  req(nstenergy >= 0 && nstxout >= 0 && nstvout >= 0 && nstcheckpoint >= 0, "output intervals must be >= 0");
  req(nstvout == 0 || nstvout == nstxout, "nstvout must be 0 or equal to nstxout (velocities are stored in the trajectory frames)");
  if (integrator == Integrator::BAOAB) req(tcoupl == Tcoupl::None, "the baoab integrator is a Langevin thermostat; use tcoupl = no");
}

}  // namespace mdl
