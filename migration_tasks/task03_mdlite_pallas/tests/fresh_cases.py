"""Fresh, grade-time randomized cases (the 'fresh' gate).

    fresh_cases.py build  <seed> <dir>      build systems + configs into <dir>   (pristine mdbuild)
    fresh_cases.py run    <dir> <out.npz>   run whichever mdlite is importable, save results

The seed is drawn by the grader at grading time, the CUDA original runs as root
into a root-only directory, and the candidate runs as an unprivileged user, so
no stored golden can be looked up.
"""
import json
import sys
from pathlib import Path

import numpy as np


def build(seed, d):
    import mdbuild as mb
    rng = np.random.default_rng(seed)
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    cases = []

    def r(a, b):
        return float(rng.uniform(a, b))

    # equilibrate quickly with the original before writing? no: the fresh systems are
    # used for forces and very short trajectories only, built gently (small jitter).
    s = mb.salt_water(int(rng.integers(500, 1400)), int(rng.integers(3, 20)), seed=int(rng.integers(1 << 30)))
    s.save(d / "salt.mdsys")
    s = mb.lj_fluid(int(rng.integers(1500, 4000)), density=r(14, 19), mixture=bool(rng.integers(2)), seed=int(rng.integers(1 << 30)),
                    jitter=0.02)
    s.save(d / "lj.mdsys")
    s = mb.alkane_melt(int(rng.integers(12, 40)), int(rng.integers(6, 16)), charges=True, seed=int(rng.integers(1 << 30)),
                       constrain_bonds=bool(rng.integers(2)))
    s.save(d / "alkane.mdsys")

    order = int(rng.integers(3, 9))
    cases.append(("salt_pme", "salt", {"cutoff": round(r(0.8, 1.0), 3), "pme_order": order,
                                       "fourier_spacing": round(r(0.08, 0.16), 3), "ewald_rtol": float(10 ** r(-7, -4)),
                                       "dispcorr": "EnerPres" if rng.integers(2) else "no", "constraints": "no"}))
    cases.append(("salt_rf", "salt", {"cutoff": round(r(0.8, 1.0), 3), "coulombtype": "reaction-field",
                                      "epsilon_rf": round(r(1.5, 90), 2), "constraints": "no"}))
    rc = round(r(0.9, 1.2), 3)
    cases.append(("lj", "lj", {"cutoff": rc, "coulombtype": "cutoff", "vdw_modifier": "force-switch", "rvdw_switch": round(rc - r(0.1, 0.3), 3),
                               "dispcorr": "EnerPres"}))
    cases.append(("alkane", "alkane", {"cutoff": 1.0, "fourier_spacing": round(r(0.09, 0.13), 3), "pme_order": int(rng.integers(4, 7))}))
    dyn = [("dyn_baoab", "lj", {"cutoff": 1.0, "coulombtype": "cutoff", "integrator": "baoab", "tau_t": round(r(0.1, 1.0), 3),
                                "ref_t": round(r(80, 140), 1), "seed": int(rng.integers(1 << 62)), "dt": 0.002}),
           ("dyn_csvr", "alkane", {"cutoff": 1.0, "coulombtype": "reaction-field", "tcoupl": "csvr", "tau_t": round(r(0.05, 0.5), 3),
                                   "nsttcouple": int(rng.integers(1, 4)), "seed": int(rng.integers(1 << 62)), "dt": 0.0005,
                                   "gen_vel": True, "gen_temp": round(r(250, 400), 1)}),
           ("dyn_npt", "lj", {"cutoff": 1.0, "coulombtype": "cutoff", "pcoupl": "c-rescale", "tau_p": round(r(0.2, 1.0), 3),
                              "nstpcouple": int(rng.integers(1, 4)), "compressibility": 4.5e-4, "seed": int(rng.integers(1 << 62)),
                              "dt": 0.002, "gen_vel": True, "gen_temp": 100})]
    (d / "cases.json").write_text(json.dumps({"static": cases, "dynamics": dyn, "steps": [1, 3, 10]}))


def run(d, out):
    import mdlite as md
    d = Path(d)
    spec = json.loads((d / "cases.json").read_text())
    res = {}
    for name, sname, cfg in spec["static"]:
        with md.Simulation(str(d / f"{sname}.mdsys"), cfg) as sim:
            res[f"{name}/f"] = sim.forces()
            e = sim.energies()
            res[f"{name}/e"] = np.array([e[t] for t in md.TERMS])
            res[f"{name}/natoms"] = np.array(sim.natoms)
    for name, sname, cfg in spec["dynamics"]:
        with md.Simulation(str(d / f"{sname}.mdsys"), cfg) as sim:
            for k in spec["steps"]:
                sim.run(k - sim.step)
                res[f"{name}/x{k}"] = sim.positions
                res[f"{name}/L{k}"] = sim.box
    np.savez(out, **res)


def compare(ref, cand, terms):
    """Returns a list of failure strings (empty when the candidate matches)."""
    a, b = np.load(ref), np.load(cand)
    fails = []
    if set(a.files) != set(b.files):
        return [f"missing results: {sorted(set(a.files) ^ set(b.files))[:10]}"]
    x_tol = {1: 2e-6, 3: 6e-6, 10: 2e-5}
    for k in a.files:
        name, what = k.split("/", 1)
        ra, rb = a[k], b[k]
        if what == "f":
            den = np.sqrt((ra ** 2).sum(1).mean())
            err = np.sqrt(((ra - rb) ** 2).sum(1).mean())
            if not (err <= 4e-5 * den or err <= 5e-3) or np.abs(ra - rb).max() > max(2e-4 * np.abs(ra).max(), 0.05):
                fails.append(f"{name}: forces rms {err:.3e} (ref rms {den:.3e})")
        elif what == "e":
            n = int(a[f"{name}/natoms"])
            for t, va, vb in zip(terms, ra, rb):
                tol = 0.5 if t == "pressure" else 2e-5 * abs(va) + 2e-6 * n
                if not abs(va - vb) <= tol:
                    fails.append(f"{name}: {t} {vb!r} vs {va!r}")
        elif what.startswith("x"):
            s = int(what[1:])
            dx = np.abs(ra - rb).max()
            if not dx <= x_tol[s]:
                fails.append(f"{name}: step {s} position deviation {dx:.3e}")
        elif what.startswith("L"):
            if not np.abs(ra - rb).max() <= 1e-7:
                fails.append(f"{name}: box {what}")
    return fails


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(int(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
