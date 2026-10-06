#include "system.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <queue>
#include <set>

#include "common.cuh"
#include "crc32.h"

namespace mdl {
namespace {

static const char SYS_MAGIC[8] = {'M', 'D', 'S', 'Y', 'S', 0, 1, 0};

struct Reader {
  const std::vector<char>& buf;
  size_t pos = 0;
  explicit Reader(const std::vector<char>& b) : buf(b) {}
  void raw(void* dst, size_t n) {
    if (pos + n > buf.size()) throw Error("mdsys: truncated file");
    std::memcpy(dst, buf.data() + pos, n);
    pos += n;
  }
  int32_t i32() { int32_t v; raw(&v, 4); return v; }
  double f64() { double v; raw(&v, 8); return v; }
};

}  // namespace

System System::read(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error("mdsys: cannot open '" + path + "'");
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (buf.size() < 12 || std::memcmp(buf.data(), SYS_MAGIC, 8) != 0) throw Error("mdsys: bad magic in '" + path + "'");
  uint32_t stored;
  std::memcpy(&stored, buf.data() + buf.size() - 4, 4);
  if (crc32(reinterpret_cast<const unsigned char*>(buf.data()), buf.size() - 4) != stored)
    throw Error("mdsys: checksum mismatch in '" + path + "'");

  Reader r(buf);
  r.pos = 8;
  System s;
  int32_t cnt[9];
  for (auto& c : cnt) c = r.i32();
  for (auto c : cnt)
    if (c < 0) throw Error("mdsys: negative count");
  s.natoms = cnt[0];
  s.ntypes = cnt[1];
  s.nrexcl = r.i32();
  int32_t has_v = r.i32();
  for (double& L : s.box) L = r.f64();
  s.fudge_lj = r.f64();
  s.fudge_qq = r.f64();
  for (int t = 0; t < s.ntypes; ++t) {
    char name[9] = {0};
    r.raw(name, 8);
    s.type_name.emplace_back(name);
    s.type_sigma.push_back(r.f64());
    s.type_epsilon.push_back(r.f64());
  }
  for (int k = 0; k < cnt[2]; ++k) {
    NbFix n;
    n.ti = r.i32(); n.tj = r.i32();
    n.sigma = r.f64(); n.epsilon = r.f64();
    s.nbfix.push_back(n);
  }
  int N = s.natoms;
  s.type.resize(N); s.molecule.resize(N); s.charge.resize(N); s.mass.resize(N); s.x.resize(3 * (size_t)N);
  r.raw(s.type.data(), 4 * (size_t)N);
  r.raw(s.molecule.data(), 4 * (size_t)N);
  r.raw(s.charge.data(), 8 * (size_t)N);
  r.raw(s.mass.data(), 8 * (size_t)N);
  r.raw(s.x.data(), 24 * (size_t)N);
  if (has_v) {
    s.v.resize(3 * (size_t)N);
    r.raw(s.v.data(), 24 * (size_t)N);
  }
  for (int k = 0; k < cnt[3]; ++k) { Bond b; b.i = r.i32(); b.j = r.i32(); b.r0 = r.f64(); b.k = r.f64(); s.bonds.push_back(b); }
  for (int k = 0; k < cnt[4]; ++k) { Angle a; a.i = r.i32(); a.j = r.i32(); a.k = r.i32(); a.theta0 = r.f64(); a.kth = r.f64(); s.angles.push_back(a); }
  for (int k = 0; k < cnt[5]; ++k) {
    Dihedral d; d.i = r.i32(); d.j = r.i32(); d.k = r.i32(); d.l = r.i32(); d.mult = r.i32();
    d.phi0 = r.f64(); d.kphi = r.f64(); s.dihedrals.push_back(d);
  }
  for (int k = 0; k < cnt[6]; ++k) { Pair14 p; p.i = r.i32(); p.j = r.i32(); s.pairs.push_back(p); }
  for (int k = 0; k < cnt[7]; ++k) { Constraint c; c.i = r.i32(); c.j = r.i32(); c.d = r.f64(); s.constraints.push_back(c); }
  for (int k = 0; k < cnt[8]; ++k) { int a = r.i32(), b = r.i32(); s.extra_exclusions.push_back({a, b}); }
  if (r.pos != buf.size() - 4) throw Error("mdsys: trailing bytes");
  s.validate();
  return s;
}

void System::validate() const {
  auto req = [](bool ok, const std::string& m) { if (!ok) throw Error("mdsys: " + m); };
  int N = natoms;
  req(N >= 1, "system has no atoms");
  req(ntypes >= 1, "system has no atom types");
  for (double L : box) req(std::isfinite(L) && L > 0, "box lengths must be positive");
  req(nrexcl >= 0 && nrexcl <= 5, "nrexcl must be in [0, 5]");
  for (int t = 0; t < ntypes; ++t) req(type_sigma[t] >= 0 && type_epsilon[t] >= 0, "negative LJ parameter");
  for (auto& n : nbfix) req(n.ti >= 0 && n.ti < ntypes && n.tj >= 0 && n.tj < ntypes && n.sigma >= 0 && n.epsilon >= 0, "bad nbfix entry");
  for (int i = 0; i < N; ++i) {
    req(type[i] >= 0 && type[i] < ntypes, "atom type out of range");
    req(mass[i] > 0 && std::isfinite(mass[i]), "masses must be positive");
    req(std::isfinite(charge[i]), "non-finite charge");
    for (int d = 0; d < 3; ++d) req(std::isfinite(x[3 * i + d]), "non-finite position");
  }
  for (double vv : v) req(std::isfinite(vv), "non-finite velocity");
  auto atom = [&](int a) { req(a >= 0 && a < N, "atom index out of range"); };
  auto same_mol = [&](int a, int b) { req(molecule[a] == molecule[b], "bonded atoms must belong to the same molecule"); };
  for (auto& b : bonds) { atom(b.i); atom(b.j); req(b.i != b.j, "bond to self"); same_mol(b.i, b.j); }
  for (auto& a : angles) { atom(a.i); atom(a.j); atom(a.k); same_mol(a.i, a.j); same_mol(a.j, a.k); }
  for (auto& d : dihedrals) { atom(d.i); atom(d.j); atom(d.k); atom(d.l); req(d.mult >= 0 && d.mult <= 6, "dihedral multiplicity must be in [0, 6]"); same_mol(d.i, d.j); same_mol(d.j, d.k); same_mol(d.k, d.l); }
  for (auto& p : pairs) { atom(p.i); atom(p.j); req(p.i != p.j, "pair to self"); }
  for (auto& c : constraints) { atom(c.i); atom(c.j); req(c.i != c.j, "constraint to self"); req(c.d > 0, "constraint length must be > 0"); same_mol(c.i, c.j); }
  for (auto& e : extra_exclusions) { atom(e[0]); atom(e[1]); }
}

Topology Topology::build(const System& s, bool use_constraints) {
  Topology t;
  int N = s.natoms;

  // exclusions: atoms within nrexcl edges of the bond + constraint graph,
  // plus explicit extra exclusions
  std::vector<std::vector<int>> adj(N);
  for (auto& b : s.bonds) { adj[b.i].push_back(b.j); adj[b.j].push_back(b.i); }
  for (auto& c : s.constraints) { adj[c.i].push_back(c.j); adj[c.j].push_back(c.i); }
  std::vector<std::set<int>> ex(N);
  if (s.nrexcl > 0) {
    std::vector<int> dist(N, -1);
    for (int i = 0; i < N; ++i) {
      if (adj[i].empty()) continue;
      std::vector<int> touched{i};
      std::queue<int> q;
      q.push(i);
      dist[i] = 0;
      while (!q.empty()) {
        int a = q.front(); q.pop();
        if (dist[a] == s.nrexcl) continue;
        for (int b : adj[a])
          if (dist[b] < 0) { dist[b] = dist[a] + 1; touched.push_back(b); q.push(b); ex[i].insert(b); }
      }
      for (int a : touched) dist[a] = -1;
    }
  }
  for (auto& e : s.extra_exclusions)
    if (e[0] != e[1]) { ex[e[0]].insert(e[1]); ex[e[1]].insert(e[0]); }
  t.excl_ptr.assign(N + 1, 0);
  for (int i = 0; i < N; ++i) {
    t.excl_ptr[i + 1] = t.excl_ptr[i] + (int)ex[i].size();
    for (int j : ex[i]) {
      t.excl_idx.push_back(j);
      if (i < j) t.excl_pairs.push_back({i, j});
    }
  }

  // Lorentz-Berthelot combination, overridden by nbfix entries
  int T = s.ntypes;
  t.c6.assign(T * T, 0.0);
  t.c12.assign(T * T, 0.0);
  auto set_pair = [&](int a, int b, double sig, double eps) {
    double s6 = std::pow(sig, 6);
    t.c6[a * T + b] = t.c6[b * T + a] = 4.0 * eps * s6;
    t.c12[a * T + b] = t.c12[b * T + a] = 4.0 * eps * s6 * s6;
  };
  for (int a = 0; a < T; ++a)
    for (int b = a; b < T; ++b)
      set_pair(a, b, 0.5 * (s.type_sigma[a] + s.type_sigma[b]), std::sqrt(s.type_epsilon[a] * s.type_epsilon[b]));
  for (auto& n : s.nbfix) set_pair(n.ti, n.tj, n.sigma, n.epsilon);

  // constraint clusters (union-find), ordered by their first constraint
  if (use_constraints && !s.constraints.empty()) {
    std::vector<int> parent(N);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int a) { return parent[a] == a ? a : parent[a] = find(parent[a]); };
    for (auto& c : s.constraints) parent[find(c.i)] = find(c.j);
    std::map<int, int> root_to_cluster;
    std::vector<std::vector<int>> members;
    for (int k = 0; k < (int)s.constraints.size(); ++k) {
      int root = find(s.constraints[k].i);
      auto it = root_to_cluster.find(root);
      if (it == root_to_cluster.end()) {
        it = root_to_cluster.emplace(root, (int)members.size()).first;
        members.emplace_back();
      }
      members[it->second].push_back(k);
    }
    t.cl_ptr.push_back(0);
    for (auto& m : members) {
      t.cl_cons.insert(t.cl_cons.end(), m.begin(), m.end());
      t.cl_ptr.push_back((int)t.cl_cons.size());
    }
    t.ncons = (int)s.constraints.size();
  } else {
    t.cl_ptr.push_back(0);
  }

  // molecules ordered by molecule id
  std::map<int, std::vector<int>> mols;
  for (int i = 0; i < N; ++i) mols[s.molecule[i]].push_back(i);
  t.mol_ptr.push_back(0);
  for (auto& kv : mols) {
    t.mol_atoms.insert(t.mol_atoms.end(), kv.second.begin(), kv.second.end());
    t.mol_ptr.push_back((int)t.mol_atoms.size());
  }
  t.nmol = (int)mols.size();
  return t;
}

}  // namespace mdl
