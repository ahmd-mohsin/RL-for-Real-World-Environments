"""Analysis of mdlite trajectories and energy files.

All functions take plain numpy arrays or the dicts returned by
``mdlite.read_trajectory`` / ``mdlite.read_energy``.
"""
from __future__ import annotations

import numpy as np

AMU_PER_NM3_TO_KG_PER_M3 = 1.66053906660


def _min_image(d, box):
    return d - box * np.round(d / box)


def radial_distribution(positions, box, sel_a, sel_b, rmax=1.0, nbins=100):
    """g(r) between atom selections (index arrays) of one frame or a stack of frames."""
    pos = np.asarray(positions, float)
    if pos.ndim == 2:
        pos = pos[None]
    boxes = np.broadcast_to(np.asarray(box, float).reshape(-1, 3), (len(pos), 3))
    sel_a, sel_b = np.asarray(sel_a), np.asarray(sel_b)
    edges = np.linspace(0.0, rmax, nbins + 1)
    hist = np.zeros(nbins)
    same = np.array_equal(sel_a, sel_b)
    norm = 0.0
    for x, L in zip(pos, boxes):
        a, b = x[sel_a], x[sel_b]
        for k in range(0, len(a), 512):
            d = _min_image(a[k:k + 512, None, :] - b[None, :, :], L)
            r = np.linalg.norm(d, axis=-1)
            if same:
                r = r[r > 0]
            hist += np.histogram(r, edges)[0]
        npairs = len(a) * (len(b) - 1) if same else len(a) * len(b)
        norm += npairs / L.prod()
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, hist / (shell * norm)


def msd(trajectory, sel=None):
    """Mean squared displacement relative to the first frame (positions are unwrapped in mdlite)."""
    x = np.asarray(trajectory["positions"], float)
    if sel is not None:
        x = x[:, sel]
    d = x - x[0]
    return trajectory["times"] - trajectory["times"][0], (d ** 2).sum(-1).mean(-1)


def density(masses, box):
    """Mass density in kg/m^3."""
    return float(np.sum(masses) / np.prod(box) * AMU_PER_NM3_TO_KG_PER_M3)


def energy_drift(energy, column="conserved", natoms=1):
    """Linear drift of an energy column in kJ/mol per ns (per atom when natoms is given)."""
    t = energy["time"]
    if len(t) < 2:
        return 0.0
    slope = np.polyfit(t - t[0], energy[column], 1)[0]
    return float(slope * 1000.0 / natoms)


def temperature_stats(energy, skip=0):
    t = energy["temperature"][skip:]
    return float(t.mean()), float(t.std())


def rerun(system, config, trajectory, frames=None):
    """Recompute energies for trajectory frames (like ``gmx mdrun -rerun``).

    Positions in the trajectory are single precision; energies are evaluated at
    exactly those coordinates. Returns one energies dict per frame.
    """
    import mdlite

    frames = range(len(trajectory["steps"])) if frames is None else frames
    out = []
    with mdlite.Simulation(system, config) as sim:
        for f in frames:
            sim.box = trajectory["boxes"][f]
            sim.positions = trajectory["positions"][f].astype(np.float64)
            e = sim.energies()
            out.append({k: v for k, v in e.items() if isinstance(v, float)})
    return out


def water_geometry_errors(positions, molecule_starts, oh=0.1, hh=0.1633):
    """Max relative deviation of SPC/E O-H and H-H distances (atoms O, H, H per molecule)."""
    x = np.asarray(positions, float)
    o = np.asarray(molecule_starts)
    d1 = np.linalg.norm(x[o] - x[o + 1], axis=-1)
    d2 = np.linalg.norm(x[o] - x[o + 2], axis=-1)
    d3 = np.linalg.norm(x[o + 1] - x[o + 2], axis=-1)
    return float(max(np.abs(d1 / oh - 1).max(), np.abs(d2 / oh - 1).max(), np.abs(d3 / hh - 1).max()))
