"""Hidden test cases shared by the golden generator (run with the CUDA original)
and the hidden tests (run with the candidate port).

Systems are stored in data/systems/*.mdsys (built once by tools/gen_golden.py)
so that the tests do not depend on builder randomness.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
SYS = DATA / "systems"


# large systems are rebuilt deterministically (pristine mdbuild) instead of being stored
LAZY = {
    "lj_1m": lambda mb: mb.lj_fluid(1_000_000, density=21.0, seed=14, jitter=0.03),
    "water_big": lambda mb: mb.spce_water(32000, seed=15),
}
CACHE = Path(os.environ.get("MDLITE_TEST_CACHE", "/tmp/mdlite_hidden_cache"))
FORCE_SUBSET = 20000   # golden forces of large systems are stored for the first atoms only


def sysfile(name):
    if name in LAZY:
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"{name}.mdsys"
        if not path.exists():
            import mdbuild as mb
            tmp = CACHE / f"{name}.{os.getpid()}.tmp"
            LAZY[name](mb).save(tmp)
            os.replace(tmp, path)
        return str(path)
    return str(SYS / f"{name}.mdsys")


# ---------------------------------------------------------------- system builders (golden generation only)
def build_systems():
    import mdbuild as mb
    import mdlite as md
    from mdlite import System, make_types

    out = {}
    rng = np.random.default_rng(1234)

    out["lj_mix"] = mb.lj_fluid(3001, mixture=True, seed=1, jitter=0.05)          # prime-ish count, nbfix
    out["water"] = mb.spce_water(1499, seed=2)
    out["salt"] = mb.salt_water(997, 23, seed=3)
    out["alkane_q"] = mb.alkane_melt(37, 13, charges=True, seed=4)
    out["alkane_cons"] = mb.alkane_melt(29, 9, charges=True, seed=5, constrain_bonds=True)
    out["netq"] = mb.add_net_charge(mb.spce_water(611, seed=6), 3)
    out["droplet"] = mb.water_droplet(700, 6.0, seed=7)
    out["crystal"] = mb.nacl_crystal(4, jitter=0.01, seed=8)

    # far-away periodic images, boundary atoms and exact box edges
    s = mb.salt_water(400, 10, seed=9)
    shift = rng.integers(-1000, 1001, size=(s.natoms // 3 + 40, 3))
    mol = s.molecule
    x = s.positions.copy()
    for m in np.unique(mol):
        x[mol == m] += shift[m] * s.box
    s.positions = x
    out["far_images"] = s

    s = mb.lj_fluid(1000, density=15.0, seed=10, jitter=0.02)
    L = s.box
    s.positions[0] = [0.0, 0.0, 0.0]
    s.positions[1] = [L[0], L[1] * 0.5, L[2] * 0.5]
    s.positions[2] = [-0.0, 0.25 * L[1], 1e-300]
    s.positions[3] = [np.nextafter(L[0], 0), 0.75 * L[1], -np.nextafter(0.0, 1.0)]
    s.positions[4] = [L[0] * 0.5, -1e-17, L[2] - 1e-17]
    s.charge = np.where(np.arange(1000) % 2 == 0, 0.3, -0.3)
    out["edges"] = s

    # minimal systems
    t = make_types([("A", 0.3, 0.6), ("B", 0.25, 0.3)])
    out["one_atom"] = System(box=[3.0, 3.1, 3.2], types=t, type=[0], molecule=[0], charge=[0.5], mass=[12.0],
                             positions=[[0.3, 2.9, -0.1]])
    out["three_atoms"] = System(box=[2.5, 2.5, 2.5], types=t, type=[0, 1, 0], molecule=[0, 1, 2], charge=[0.4, -0.7, 0.3],
                                mass=[12.0, 1.0, 30.0], positions=[[0.1, 0.1, 0.1], [0.5, 0.2, 2.4], [1.3, 1.2, 0.05]])

    # exotic topology: linear angles, planar/collinear dihedrals, mult 0, nrexcl 5, cross-molecule exclusions
    n = 60
    xs = []
    for c in range(10):
        base = np.array([0.6 * (c % 4) + 0.3, 0.6 * (c // 4) + 0.3, 0.5])
        for k in range(6):
            xs.append(base + [0.0, 0.0, 0.15 * k] + (rng.normal(scale=0.02, size=3) if c % 2 else 0.0))
    xs = np.array(xs)
    bonds, angles, dihs, pairs = [], [], [], []
    for c in range(10):
        o = 6 * c
        for k in range(5):
            bonds.append((o + k, o + k + 1, 0.15, 3e5))
        for k in range(4):
            angles.append((o + k, o + k + 1, o + k + 2, 180.0 if c % 3 == 0 else 112.0, 400.0))
        for k in range(3):
            dihs.append((o + k, o + k + 1, o + k + 2, o + k + 3, (c + k) % 7, 37.0 * k, 3.0 + k))
        pairs.append((o, o + 3))
        pairs.append((o + 1, o + 4))
    tt = make_types([("C", 0.35, 0.4), ("D", 0.3, 0.7), ("E", 0.0, 0.0)])
    out["exotic"] = System(box=[2.6, 2.4, 2.5], types=tt, type=[k % 3 for k in range(n)], molecule=np.repeat(np.arange(10), 6),
                           charge=np.tile([0.3, -0.2, 0.1, -0.4, 0.25, -0.05], 10), mass=np.tile([12.0, 14.0, 1.0, 16.0, 13.0, 15.0], 10),
                           positions=xs, bonds=bonds, angles=angles, dihedrals=dihs, pairs=pairs,
                           exclusions=[(0, 59), (7, 33), (12, 13)], nbfix=[(0, 2, 0.31, 0.2)], fudge_lj=0.5, fudge_qq=0.8333,
                           nrexcl=5)

    # an excluded pair at zero distance (PME exclusion limit) and a zero-charge PME system
    s = mb.spce_water(200, seed=11)
    s.exclusions = np.array([(0, 1), (3, 300)], dtype=s.exclusions.dtype)
    s.positions[300] = s.positions[3]
    s.charge = s.charge.copy()
    out["zero_dist_excl"] = s
    s = mb.lj_fluid(500, seed=12)
    out["zero_charge"] = s

    # dense inhomogeneous cluster: half the atoms squeezed into one corner (neighbour capacity overflow)
    s = mb.lj_fluid(4000, density=8.0, seed=13, jitter=0.0)
    x = s.positions.copy()
    k = np.arange(2000)
    g = np.stack(np.meshgrid(*[np.arange(13)] * 3, indexing="ij"), -1).reshape(-1, 3)[:2000] * 0.27 + 0.2
    x[k] = g
    s.positions = x
    s.types = make_types([("AR", 0.25, 0.8)])
    out["dense_corner"] = s

    return out


# ---------------------------------------------------------------- force / energy cases: (name, system, config)
BASE_PME = {"cutoff": 0.9}
FORCE_CASES = [
    ("lj_shift", "lj_mix", {"cutoff": 1.0, "coulombtype": "cutoff"}),
    ("lj_fsw_disp", "lj_mix", {"cutoff": 1.1, "vdw_modifier": "force-switch", "rvdw_switch": 0.85, "coulombtype": "cutoff",
                               "dispcorr": "EnerPres", "skin": 0.0}),
    ("lj_fsw0", "lj_mix", {"cutoff": 0.95, "vdw_modifier": "force-switch", "rvdw_switch": 0.0, "coulombtype": "reaction-field"}),
    ("water_pme4", "water", {"cutoff": 0.9}),
    ("water_pme3", "water", {"cutoff": 0.9, "pme_order": 3, "fourier_spacing": 0.09}),
    ("water_pme5", "water", {"cutoff": 0.9, "pme_order": 5, "fourier_spacing": 0.105}),
    ("water_pme6_tight", "water", {"cutoff": 1.05, "pme_order": 6, "fourier_spacing": 0.08, "ewald_rtol": 1e-7}),
    ("water_pme7", "water", {"cutoff": 0.95, "pme_order": 7, "fourier_spacing": 0.14, "ewald_rtol": 3e-6}),
    ("water_pme8", "water", {"cutoff": 1.0, "pme_order": 8, "fourier_spacing": 0.16, "ewald_rtol": 1e-4}),
    ("water_rf", "water", {"cutoff": 0.9, "coulombtype": "reaction-field", "epsilon_rf": 78.5}),
    ("water_rfinf", "water", {"cutoff": 0.9, "coulombtype": "reaction-field", "epsilon_rf": 0}),
    ("water_cut", "water", {"cutoff": 0.9, "coulombtype": "cutoff", "skin": 0.25}),
    ("water_flex", "water", {"cutoff": 0.9, "constraints": "no"}),
    ("salt_pme", "salt", {"cutoff": 1.0, "fourier_spacing": 0.11, "dispcorr": "EnerPres"}),
    ("alkane_pme", "alkane_q", {"cutoff": 1.0, "fourier_spacing": 0.1}),
    ("alkane_rf", "alkane_q", {"cutoff": 1.0, "coulombtype": "reaction-field", "epsilon_rf": 15}),
    ("alkane_cons", "alkane_cons", {"cutoff": 1.0}),
    ("netq", "netq", {"cutoff": 0.9}),
    ("droplet", "droplet", {"cutoff": 1.2, "fourier_spacing": 0.15}),
    ("crystal6", "crystal", {"cutoff": 0.9, "pme_order": 6, "fourier_spacing": 0.08}),
    ("far_images", "far_images", {"cutoff": 1.0}),
    ("edges", "edges", {"cutoff": 1.0, "fourier_spacing": 0.13}),
    ("one_atom", "one_atom", {"cutoff": 1.0}),
    ("three_atoms", "three_atoms", {"cutoff": 1.0, "pme_order": 5}),
    ("three_atoms_rf", "three_atoms", {"cutoff": 1.2, "coulombtype": "reaction-field", "epsilon_rf": 2.5, "skin": 0.0}),
    ("exotic_pme", "exotic", {"cutoff": 1.0, "fourier_spacing": 0.1}),
    ("exotic_rf", "exotic", {"cutoff": 1.0, "coulombtype": "reaction-field", "epsilon_rf": 0, "vdw_modifier": "force-switch",
                             "rvdw_switch": 0.7}),
    ("zero_dist_excl", "zero_dist_excl", {"cutoff": 0.8, "skin": 0.1, "constraints": "no"}),
    ("zero_charge", "zero_charge", {"cutoff": 1.0}),
    ("dense_corner", "dense_corner", {"cutoff": 1.1, "coulombtype": "cutoff"}),
    ("lj_1m", "lj_1m", {"cutoff": 1.0, "coulombtype": "cutoff"}),
    ("water_big", "water_big", {"cutoff": 1.0, "fourier_spacing": 0.125, "pme_order": 5}),
]

# ---------------------------------------------------------------- dynamics cases (equilibrated inputs)
DYN_BASE = {"cutoff": 0.9, "dt": 0.002, "seed": 20231, "nstenergy": 5}
DYN_CASES = [
    ("nve", "salt_eq", {"nstcomm": 0}),
    ("nve_comm", "salt_eq", {"nstcomm": 7}),
    ("csvr", "salt_eq", {"tcoupl": "csvr", "tau_t": 0.1, "ref_t": 340, "nsttcouple": 3}),
    ("csvr_v_rescale_alias", "salt_eq", {"tcoupl": "v-rescale", "tau_t": 0.05, "ref_t": 250, "nsttcouple": 1}),
    ("baoab", "salt_eq", {"integrator": "baoab", "tau_t": 0.2, "ref_t": 310}),
    ("baoab_flex", "salt_eq", {"integrator": "baoab", "tau_t": 0.5, "ref_t": 290, "constraints": "no", "dt": 0.0005}),
    ("berendsen", "salt_eq", {"pcoupl": "berendsen", "tau_p": 0.3, "nstpcouple": 2, "compressibility": 4.5e-4, "ref_p": 500}),
    ("crescale", "salt_eq", {"tcoupl": "csvr", "tau_t": 0.2, "pcoupl": "c-rescale", "tau_p": 0.3, "nstpcouple": 3,
                             "compressibility": 4.5e-4, "ref_p": -200}),
    ("baoab_crescale", "salt_eq", {"integrator": "baoab", "tau_t": 0.3, "pcoupl": "c-rescale", "tau_p": 0.5, "nstpcouple": 1,
                                   "compressibility": 4.5e-4}),
    ("genvel_rf", "salt_eq", {"gen_vel": True, "gen_temp": 280, "coulombtype": "reaction-field", "epsilon_rf": 0}),
    ("alkane_rf", "alkane_eq", {"cutoff": 1.0, "coulombtype": "reaction-field", "epsilon_rf": 15, "dt": 0.001,
                                "tcoupl": "csvr", "tau_t": 0.2, "ref_t": 360}),
    ("alkane_cons_baoab", "alkane_cons_eq", {"cutoff": 1.0, "integrator": "baoab", "tau_t": 0.2, "ref_t": 330, "dt": 0.002,
                                             "pcoupl": "berendsen", "tau_p": 1.0, "nstpcouple": 4}),
    ("lj_fsw_nve", "lj_eq", {"cutoff": 1.1, "vdw_modifier": "force-switch", "rvdw_switch": 0.9, "coulombtype": "cutoff",
                             "dispcorr": "EnerPres", "nstcomm": 0, "skin": 0.05}),
    # neighbour capacity growth: half of the atoms packed into one corner
    ("dense_corner_nve", "dense_corner", {"cutoff": 1.1, "coulombtype": "cutoff", "dt": 0.002, "nstcomm": 0, "skin": 0.05}),
    # strongly inhomogeneous: a water droplet in vacuum with PME
    ("droplet_baoab", "droplet", {"cutoff": 1.2, "fourier_spacing": 0.15, "integrator": "baoab", "tau_t": 0.5, "ref_t": 300}),
    # the box shrinks by several percent: cell grid and neighbour lists change, PME grid stays fixed
    ("npt_squeeze", "salt_eq", {"pcoupl": "berendsen", "ref_p": 2000, "compressibility": 4.5e-5, "tau_p": 0.1, "nstpcouple": 1,
                                "tcoupl": "csvr", "tau_t": 0.1}),
]
DYN_CHECK_STEPS = (1, 2, 10, 50, 100)

# large systems: energies after a short run, and positions of a subset of atoms
SCALE_CASES = [
    ("lj_1m_baoab", "lj_1m", {"cutoff": 1.0, "coulombtype": "cutoff", "integrator": "baoab", "tau_t": 0.5, "ref_t": 120,
                              "dt": 0.002, "seed": 77}, 20),
    ("water_big_csvr", "water_big", {"cutoff": 1.0, "fourier_spacing": 0.125, "pme_order": 5, "tcoupl": "csvr", "tau_t": 0.1,
                                     "dt": 0.001, "seed": 78}, 20),
]
SCALE_SUBSET = 3000


def merged(base, extra):
    d = dict(base)
    d.update(extra)
    return d


def relrms(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = math.sqrt(float((b ** 2).sum(-1).mean())) if b.size else 1.0
    return math.sqrt(float(((a - b) ** 2).sum(-1).mean())) / max(den, 1e-30)
