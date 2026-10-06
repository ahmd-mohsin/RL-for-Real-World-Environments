// C interface of libmdlite, used by the Python package through ctypes.
// Every function returns 0 on success and -1 on failure; mdl_last_error()
// then describes the failure.
#include <cstring>
#include <exception>
#include <string>

#include "simulation.h"

using mdl::Simulation;

namespace {
thread_local std::string g_error;

template <class F>
int guard(F&& f) {
  try {
    f();
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
  } catch (...) {
    g_error = "unknown error";
  }
  return -1;
}
Simulation* S(void* h) { return static_cast<Simulation*>(h); }
}  // namespace

extern "C" {

const char* mdl_last_error() { return g_error.c_str(); }
int mdl_term_count() { return mdl::T_COUNT; }
const char* mdl_term_name(int k) { return (k >= 0 && k < mdl::T_COUNT) ? mdl::TERM_NAMES[k] : ""; }

void* mdl_create(const char* sys_path, const char* cfg_text, const char* out_dir) {
  Simulation* sim = nullptr;
  int rc = guard([&] {
    mdl::System sys = mdl::System::read(sys_path);
    mdl::Config cfg = mdl::Config::parse(cfg_text ? cfg_text : "");
    sim = new Simulation(sys, cfg, out_dir ? out_dir : "");
  });
  return rc == 0 ? sim : nullptr;
}

void mdl_destroy(void* h) { delete S(h); }
int mdl_natoms(void* h) { return S(h)->natoms(); }
long long mdl_step(void* h) { return S(h)->step(); }
int mdl_dof(void* h) { return S(h)->dof(); }
int mdl_run(void* h, long long n) { return guard([&] { S(h)->run(n); }); }
int mdl_finish(void* h) { return guard([&] { S(h)->finish_outputs(); }); }

int mdl_energies(void* h, double* terms, double* virial, double* pressure) {
  return guard([&] {
    const mdl::Energies& e = S(h)->energies();
    std::memcpy(terms, e.term, sizeof(e.term));
    std::memcpy(virial, e.virial, sizeof(e.virial));
    std::memcpy(pressure, e.pressure, sizeof(e.pressure));
  });
}

int mdl_forces(void* h, double* out) { return guard([&] { S(h)->forces(out); }); }
int mdl_get_positions(void* h, double* out) { return guard([&] { S(h)->get_positions(out); }); }
int mdl_get_velocities(void* h, double* out) { return guard([&] { S(h)->get_velocities(out); }); }
int mdl_set_positions(void* h, const double* in) { return guard([&] { S(h)->set_positions(in); }); }
int mdl_set_velocities(void* h, const double* in) { return guard([&] { S(h)->set_velocities(in); }); }
int mdl_get_box(void* h, double* out) {
  return guard([&] { std::memcpy(out, S(h)->box(), 3 * sizeof(double)); });
}
int mdl_set_box(void* h, const double* in) { return guard([&] { S(h)->set_box(in); }); }
int mdl_save_checkpoint(void* h, const char* path) { return guard([&] { S(h)->save_checkpoint(path); }); }
int mdl_load_checkpoint(void* h, const char* path) { return guard([&] { S(h)->load_checkpoint(path); }); }
int mdl_info(void* h, char* buf, int n) {
  return guard([&] {
    std::string s = S(h)->info_json();
    if ((int)s.size() + 1 > n) throw std::runtime_error("info buffer too small");
    std::memcpy(buf, s.c_str(), s.size() + 1);
  });
}

}  // extern "C"
