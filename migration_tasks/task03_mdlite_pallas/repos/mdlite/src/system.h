// mdlite molecular system (.mdsys) and derived topology.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace mdl {

struct Bond { int i, j; double r0, k; };
struct Angle { int i, j, k; double theta0, kth; };          // theta0 in degrees
struct Dihedral { int i, j, k, l; double phi0, kphi; int mult; };  // phi0 in degrees
struct Pair14 { int i, j; };
struct Constraint { int i, j; double d; };
struct NbFix { int ti, tj; double sigma, epsilon; };

struct System {
  int natoms = 0;
  int ntypes = 0;
  double box[3] = {0, 0, 0};
  double fudge_lj = 1.0, fudge_qq = 1.0;
  int nrexcl = 3;
  std::vector<double> type_sigma, type_epsilon;
  std::vector<std::string> type_name;
  std::vector<NbFix> nbfix;
  std::vector<int> type, molecule;
  std::vector<double> charge, mass;
  std::vector<double> x, v;  // 3N each; v empty if absent
  std::vector<Bond> bonds;
  std::vector<Angle> angles;
  std::vector<Dihedral> dihedrals;
  std::vector<Pair14> pairs;
  std::vector<Constraint> constraints;
  std::vector<std::array<int, 2>> extra_exclusions;

  static System read(const std::string& path);
  void validate() const;
};

// Topology derived from a System: exclusions, LJ pair tables, constraint
// clusters and molecules.
struct Topology {
  // per-atom sorted exclusion lists (CSR, self excluded implicitly, not stored)
  std::vector<int> excl_ptr, excl_idx;
  // unique excluded pairs i < j (used for the long-range exclusion corrections)
  std::vector<std::array<int, 2>> excl_pairs;
  // LJ C6 / C12 per type pair (ntypes x ntypes)
  std::vector<double> c6, c12;
  // constraint clusters: connected components of the constraint graph;
  // cluster c owns constraints [cl_ptr[c], cl_ptr[c+1]) of `cl_cons` (indices into System::constraints)
  std::vector<int> cl_ptr, cl_cons;
  // molecules: atoms grouped by System::molecule id, mol_ptr/mol_atoms in CSR
  std::vector<int> mol_ptr, mol_atoms;
  int nmol = 0;
  int ncons = 0;

  static Topology build(const System& s, bool use_constraints);
};

}  // namespace mdl
