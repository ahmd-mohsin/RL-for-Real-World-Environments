"""System builders. Every builder is deterministic for a given seed.

Units: nm, amu, e, kJ/mol. Densities are given in molecules (or atoms) per nm^3.
"""
from __future__ import annotations

import math

import numpy as np
from mdlite import System, make_types

# SPC/E water (Berendsen et al. 1987)
SPCE_OH = 0.1
SPCE_HH = 0.1633
SPCE_QO, SPCE_QH = -0.8476, 0.4238
SPCE_SIG_O, SPCE_EPS_O = 0.316557, 0.650194
M_O, M_H = 15.9994, 1.008
# Joung-Cheatham ions for SPC/E
NA = ("NA", 0.2159538, 1.4754532, 22.98977, 1.0)
CL = ("CL", 0.4830453, 0.0535619, 35.453, -1.0)
# TraPPE-UA alkanes
CH3 = ("CH3", 0.375, 0.814817, 15.035)
CH2 = ("CH2", 0.395, 0.382465, 14.027)
CC_BOND, CC_K = 0.154, 2.5e5
CCC_ANGLE, CCC_K = 114.0, 519.65
# TraPPE torsion as a cosine series: c1, c2, c3 in kJ/mol, V = sum c_n (1 + cos(n phi - phi0_n))
TRAPPE_TORSION = ((1, 0.0, 2.95188), (2, 180.0, 0.56693), (3, 0.0, 6.57940))


def _rng(seed):
    return np.random.default_rng(seed)


def _lattice(n, box, rng, jitter=0.0):
    """n points on the smallest cubic-ish lattice that fits n sites in `box`, in random site order."""
    box = np.asarray(box, float)
    per = math.ceil((n * box.prod() / box.min() ** 3) ** (1 / 3))
    counts = np.maximum(1, np.round(box / box.min() * per)).astype(int)
    while counts.prod() < n:
        counts[np.argmin(counts / box)] += 1
    g = np.stack(np.meshgrid(*[np.arange(c) for c in counts], indexing="ij"), -1).reshape(-1, 3)
    sites = (g + 0.5) * (box / counts)
    sel = rng.permutation(len(sites))[:n]
    pts = sites[np.sort(sel)]
    if jitter:
        pts = pts + rng.uniform(-jitter, jitter, pts.shape) * (box / counts)
    return pts


def _rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def _water_geometry():
    half = SPCE_HH / 2
    h = math.sqrt(SPCE_OH ** 2 - half ** 2)
    com_shift = (2 * M_H * h) / (M_O + 2 * M_H)
    return np.array([[0, 0, -com_shift], [half, 0, h - com_shift], [-half, 0, h - com_shift]])


def lj_fluid(n, density=21.0, sigma=0.34, epsilon=0.996, mass=39.948, box=None, seed=0, mixture=False, jitter=0.15):
    """Lennard-Jones atoms (argon by default). ``mixture`` makes a 80:20 binary mixture with an
    nbfix override between the two types."""
    rng = _rng(seed)
    if box is None:
        L = (n / density) ** (1 / 3)
        box = np.array([L, L, L])
    box = np.asarray(box, float)
    x = _lattice(n, box, rng, jitter)
    if mixture:
        types = make_types([("A", sigma, epsilon), ("B", 0.88 * sigma, 0.5 * epsilon)])
        t = (rng.random(n) < 0.2).astype(np.int32)
        nbfix = [(0, 1, 0.8 * sigma, 1.5 * epsilon)]
    else:
        types = make_types([("AR", sigma, epsilon)])
        t = np.zeros(n, np.int32)
        nbfix = []
    return System(box=box, types=types, type=t, molecule=np.arange(n), charge=np.zeros(n), mass=np.full(n, mass),
                  positions=x, nbfix=nbfix)


def _water_block(nmol, box, rng, centres=None):
    geo = _water_geometry()
    if centres is None:
        centres = _lattice(nmol, box, rng, 0.1)
    x = np.zeros((3 * nmol, 3))
    for m in range(nmol):
        x[3 * m:3 * m + 3] = centres[m] + geo @ _rotation(rng).T
    return x


def _water_topology(nmol, first_atom=0, first_mol=0):
    cons, mol = [], []
    for m in range(nmol):
        o = first_atom + 3 * m
        cons += [(o, o + 1, SPCE_OH), (o, o + 2, SPCE_OH), (o + 1, o + 2, SPCE_HH)]
        mol += [first_mol + m] * 3
    return cons, mol


def spce_water(nmol, density=33.37, box=None, seed=0):
    """Rigid SPC/E water (3 constraints per molecule, hydrogens without LJ)."""
    rng = _rng(seed)
    if box is None:
        L = (nmol / density) ** (1 / 3)
        box = np.array([L, L, L])
    box = np.asarray(box, float)
    x = _water_block(nmol, box, rng)
    cons, mol = _water_topology(nmol)
    types = make_types([("OW", SPCE_SIG_O, SPCE_EPS_O), ("HW", 0.0, 0.0)])
    t = np.tile(np.array([0, 1, 1], np.int32), nmol)
    q = np.tile([SPCE_QO, SPCE_QH, SPCE_QH], nmol)
    m = np.tile([M_O, M_H, M_H], nmol)
    return System(box=box, types=types, type=t, molecule=mol, charge=q, mass=m, positions=x, constraints=cons)


def salt_water(nmol, npairs, density=33.37, seed=0):
    """SPC/E water with `npairs` Na+/Cl- ion pairs replacing water molecules."""
    rng = _rng(seed)
    ntot = nmol + 2 * npairs
    L = (ntot / density) ** (1 / 3)
    box = np.array([L, L, L])
    centres = _lattice(ntot, box, rng, 0.1)
    order = rng.permutation(ntot)
    ion_sites, water_sites = centres[order[:2 * npairs]], centres[order[2 * npairs:]]
    xw = _water_block(nmol, box, rng, water_sites)
    cons, mol = _water_topology(nmol)
    types = make_types([("OW", SPCE_SIG_O, SPCE_EPS_O), ("HW", 0.0, 0.0), NA[:3], CL[:3]])
    t = list(np.tile([0, 1, 1], nmol))
    q = list(np.tile([SPCE_QO, SPCE_QH, SPCE_QH], nmol))
    m = list(np.tile([M_O, M_H, M_H], nmol))
    for k in range(2 * npairs):
        ion = NA if k % 2 == 0 else CL
        t.append(2 if k % 2 == 0 else 3)
        q.append(ion[4])
        m.append(ion[3])
        mol.append(nmol + k)
    x = np.concatenate([xw, ion_sites])
    return System(box=box, types=types, type=t, molecule=mol, charge=q, mass=m, positions=x, constraints=cons)


def alkane_melt(nchains, nbeads, density=None, box=None, seed=0, charges=False, constrain_bonds=False):
    """United-atom alkane chains in the all-trans conformation on a lattice of chains.

    Bonds (harmonic, or constraints with ``constrain_bonds``), angles, TraPPE torsions as
    three periodic dihedrals per quadruplet, 1-4 pairs (fudge 0.5 / 0.8333) and, with
    ``charges``, alternating +-0.25 e partial charges on the chain ends (net neutral per chain).
    """
    rng = _rng(seed)
    half = math.radians(CCC_ANGLE) / 2
    dz = CC_BOND * math.sin(half)
    dy = CC_BOND * math.cos(half)
    length = (nbeads - 1) * dz + 0.5
    if box is None:
        per = math.ceil(math.sqrt(nchains))
        spacing = 0.48
        box = np.array([per * spacing, per * spacing, max(length + 0.2, 2.5)])
    box = np.asarray(box, float)
    per_x = max(1, int(box[0] / 0.48))
    per_y = math.ceil(nchains / per_x)
    xs, mols, types_, masses, qs = [], [], [], [], []
    bonds, angles, dihs, pairs, cons = [], [], [], [], []
    for c in range(nchains):
        base = np.array([(c % per_x + 0.5) * box[0] / per_x, (c // per_x + 0.5) * box[1] / per_y, rng.uniform(0, box[2])])
        flip = 1 if rng.random() < 0.5 else -1
        start = len(xs)
        for b in range(nbeads):
            xs.append(base + [0.0, flip * dy * (b % 2), b * dz])
            end = b in (0, nbeads - 1)
            types_.append(0 if end else 1)
            masses.append(CH3[3] if end else CH2[3])
            qs.append(0.0)
            mols.append(c)
        if charges and nbeads >= 2:
            qs[start] = 0.25
            qs[start + nbeads - 1] = -0.25
        for b in range(nbeads - 1):
            if constrain_bonds:
                cons.append((start + b, start + b + 1, CC_BOND))
            else:
                bonds.append((start + b, start + b + 1, CC_BOND, CC_K))
        for b in range(nbeads - 2):
            angles.append((start + b, start + b + 1, start + b + 2, CCC_ANGLE, CCC_K))
        for b in range(nbeads - 3):
            for mult, phi0, k in TRAPPE_TORSION:
                dihs.append((start + b, start + b + 1, start + b + 2, start + b + 3, mult, phi0, k))
            pairs.append((start + b, start + b + 3))
    types = make_types([CH3[:3], CH2[:3]])
    return System(box=box, types=types, type=types_, molecule=mols, charge=qs, mass=masses, positions=np.array(xs),
                  bonds=bonds, angles=angles, dihedrals=dihs, pairs=pairs, constraints=cons, fudge_lj=0.5, fudge_qq=0.8333)


def water_droplet(nmol, box_length, density=33.37, seed=0):
    """A spherical SPC/E droplet in a large empty box (strongly inhomogeneous density)."""
    rng = _rng(seed)
    radius = (3 * nmol / (4 * math.pi * density)) ** (1 / 3)
    L = float(box_length)
    side = 2 * radius
    cand = _lattice(int(nmol * 6 / math.pi * 1.3) + 8, np.array([side, side, side]), rng, 0.05) - radius
    cand = cand[np.argsort(np.linalg.norm(cand, axis=1), kind="stable")][:nmol]
    centres = cand + L / 2
    x = _water_block(nmol, np.array([L, L, L]), rng, centres)
    cons, mol = _water_topology(nmol)
    types = make_types([("OW", SPCE_SIG_O, SPCE_EPS_O), ("HW", 0.0, 0.0)])
    return System(box=[L, L, L], types=types, type=np.tile([0, 1, 1], nmol), molecule=mol,
                  charge=np.tile([SPCE_QO, SPCE_QH, SPCE_QH], nmol), mass=np.tile([M_O, M_H, M_H], nmol), positions=x,
                  constraints=cons)


def nacl_crystal(cells, a=0.564, seed=0, jitter=0.0):
    """Rock-salt NaCl, ``cells`` conventional cells per side (8 ions per cell)."""
    rng = _rng(seed)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    xs, t = [], []
    for i in range(cells):
        for j in range(cells):
            for k in range(cells):
                for b in basis:
                    xs.append((np.array([i, j, k]) + b) * a)
                    t.append(0)
                    xs.append((np.array([i, j, k]) + b + [0.5, 0, 0]) * a)
                    t.append(1)
    x = np.array(xs)
    if jitter:
        x += rng.normal(scale=jitter, size=x.shape)
    t = np.array(t, np.int32)
    types = make_types([NA[:3], CL[:3]])
    q = np.where(t == 0, 1.0, -1.0)
    m = np.where(t == 0, NA[3], CL[3])
    L = cells * a
    return System(box=[L, L, L], types=types, type=t, molecule=np.arange(len(t)), charge=q, mass=m, positions=x)


def add_net_charge(system: System, n_extra, seed=0):
    """Replace `n_extra` water molecules (or atoms) by Na+ ions, leaving a net positive charge."""
    rng = _rng(seed)
    s = system.copy()
    mols = np.unique(s.molecule)
    pick = set(rng.choice(mols, size=n_extra, replace=False).tolist())
    keep = np.array([m not in pick for m in s.molecule])
    old_to_new = -np.ones(s.natoms, int)
    old_to_new[keep] = np.arange(keep.sum())
    centres = []
    for m in sorted(pick):
        idx = np.where(s.molecule == m)[0]
        centres.append(s.positions[idx].mean(0))
    types = np.concatenate([s.types, make_types([NA[:3]])])
    na = len(types) - 1
    cons = [(old_to_new[c["i"]], old_to_new[c["j"]], c["d"]) for c in s.constraints if keep[c["i"]] and keep[c["j"]]]
    n_new = int(keep.sum())
    return System(box=s.box, types=types, type=np.concatenate([s.type[keep], np.full(n_extra, na)]),
                  molecule=np.concatenate([s.molecule[keep], np.arange(n_extra) + s.molecule.max() + 1]),
                  charge=np.concatenate([s.charge[keep], np.ones(n_extra)]),
                  mass=np.concatenate([s.mass[keep], np.full(n_extra, NA[3])]),
                  positions=np.concatenate([s.positions[keep], np.array(centres).reshape(-1, 3)]), constraints=cons) \
        if n_new + n_extra > 0 else s


def write_config(path, **kw):
    with open(path, "w") as f:
        for k, v in kw.items():
            if isinstance(v, bool):
                v = "yes" if v else "no"
            f.write(f"{k} = {v}\n")
