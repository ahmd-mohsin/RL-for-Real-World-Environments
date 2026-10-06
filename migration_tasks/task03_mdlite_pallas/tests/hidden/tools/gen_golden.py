"""Generate the hidden-test data with the CUDA original (run once, in the task image).

    PYTHONPATH=<orig mdlite>:<mdbuild> python tools/gen_golden.py [--only systems|forces|dynamics|physics]

Writes data/systems/*.mdsys, data/golden_forces.npz, data/golden_dynamics.npz,
data/golden_physics.json and data/orig_files/ (original-written output files).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import cases as C  # noqa: E402
import mdlite as md  # noqa: E402


def equilibrate(system, cfg, nsteps):
    sim = md.Simulation(system, cfg)
    sim.run(nsteps)
    s = system.copy()
    s.positions, s.velocities, s.box = sim.positions, sim.velocities, sim.box
    sim.close()
    return s


def gen_systems():
    C.SYS.mkdir(parents=True, exist_ok=True)
    import mdbuild as mb
    for name, s in C.build_systems().items():
        s.save(C.sysfile(name))
        print("system", name, s.natoms, flush=True)
    eq = {"cutoff": 0.9, "integrator": "baoab", "tau_t": 0.1, "ref_t": 300, "seed": 3}
    equilibrate(mb.salt_water(2500, 25, seed=31), dict(eq, dt=0.001), 6000).save(C.sysfile("salt_eq"))
    equilibrate(mb.alkane_melt(60, 14, charges=True, seed=32), dict(eq, cutoff=1.0, coulombtype="reaction-field", epsilon_rf=15,
                                                                    dt=0.0005, ref_t=360), 8000).save(C.sysfile("alkane_eq"))
    equilibrate(mb.alkane_melt(50, 10, charges=True, seed=33, constrain_bonds=True),
                dict(eq, cutoff=1.0, dt=0.001, ref_t=330), 6000).save(C.sysfile("alkane_cons_eq"))
    equilibrate(mb.lj_fluid(6000, density=19.0, seed=34, jitter=0.02, mixture=True),
                {"cutoff": 1.1, "coulombtype": "cutoff", "integrator": "baoab", "tau_t": 0.1, "ref_t": 110, "dt": 0.002, "seed": 3},
                5000).save(C.sysfile("lj_eq"))
    for k in ("salt_eq", "alkane_eq", "alkane_cons_eq", "lj_eq"):
        print("system", k, md.load_system(C.sysfile(k)).natoms, flush=True)


def gen_forces():
    out = {}
    for name, sname, cfg in C.FORCE_CASES:
        t0 = time.perf_counter()
        sim = md.Simulation(C.sysfile(sname), cfg)
        f = sim.forces()
        e = sim.energies()
        out[f"{name}/f"] = f[:C.FORCE_SUBSET] if sname in C.LAZY else f
        for k, v in e.items():
            out[f"{name}/e/{k}"] = np.asarray(v)
        out[f"{name}/info"] = np.array(json.dumps(sim.info))
        sim.close()
        print("forces", name, f"{time.perf_counter() - t0:.1f}s", flush=True)
    np.savez_compressed(C.DATA / "golden_forces.npz", **out)


def gen_dynamics():
    out = {}
    files = C.DATA / "orig_files"
    if files.exists():
        shutil.rmtree(files)
    files.mkdir(parents=True)
    for name, sname, extra in C.DYN_CASES:
        cfg = C.merged(C.DYN_BASE, extra)
        d = files / name
        d.mkdir()
        cfg_out = dict(cfg, nstxout=10, nstvout=10, nstcheckpoint=50)
        sim = md.Simulation(C.sysfile(sname), cfg_out, output_dir=d)
        out[f"{name}/x0"] = sim.positions
        for k in C.DYN_CHECK_STEPS:
            sim.run(k - sim.step)
            out[f"{name}/x{k}"] = sim.positions
            out[f"{name}/v{k}"] = sim.velocities
            out[f"{name}/L{k}"] = sim.box
            e = sim.energies()
            out[f"{name}/e{k}"] = np.array([e[t] for t in md.TERMS])
        sim.finish()
        sim.close()
        # keep only what the tests compare: energy file, trajectory header + 2 frames, sizes
        meta = {"traj_size": (d / "traj.mdtrj").stat().st_size,
                "checkpoints": {p.name: p.stat().st_size for p in d.glob("state_*.mdcpt")}}
        n = md.load_system(C.sysfile(sname)).natoms
        raw = (d / "traj.mdtrj").read_bytes()
        (d / "traj.mdtrj").write_bytes(raw[:32 + 2 * (40 + 24 * n)])
        for p in d.glob("state_*.mdcpt"):
            if not (name == "crescale" and p.name == "state_50.mdcpt"):
                p.unlink()
        (d / "meta.json").write_text(json.dumps(meta))
        print("dynamics", name, flush=True)
    np.savez_compressed(C.DATA / "golden_dynamics.npz", **out)


def gen_scale():
    out = {}
    for name, sname, cfg, n in C.SCALE_CASES:
        t0 = time.perf_counter()
        sim = md.Simulation(C.sysfile(sname), cfg)
        sim.run(n)
        e = sim.energies()
        out[f"{name}/e"] = np.array([e[t] for t in md.TERMS])
        out[f"{name}/x"] = sim.positions[:C.SCALE_SUBSET]
        out[f"{name}/L"] = sim.box
        sim.close()
        print("scale", name, f"{time.perf_counter() - t0:.1f}s", flush=True)
    np.savez_compressed(C.DATA / "golden_scale.npz", **out)


def gen_physics():
    """Long-run references from the original (statistics the port must reproduce)."""
    res = {}
    # NVE drift, 40 ps of salt water at 2 fs
    t0 = time.perf_counter()
    cfg = {"cutoff": 0.9, "dt": 0.002, "nstcomm": 0, "nstenergy": 250, "seed": 5}
    d = Path("/tmp/phys_nve")
    d.mkdir(exist_ok=True)
    sim = md.Simulation(C.sysfile("salt_eq"), cfg, output_dir=d)
    sim.run(20000)
    sim.finish()
    sim.close()
    e = md.read_energy(d / "energy.csv")
    n = md.load_system(C.sysfile("salt_eq")).natoms
    res["nve_drift"] = float(np.polyfit(e["time"], e["total"], 1)[0] * 1000.0 / n)
    res["nve_fluct"] = float(np.std(e["total"] - np.polyval(np.polyfit(e["time"], e["total"], 1), e["time"])))
    res["nve_T"] = float(e["temperature"].mean())
    print("physics nve", res, f"{time.perf_counter() - t0:.0f}s", flush=True)
    # NPT density with c-rescale + csvr, 60 ps
    cfg = {"cutoff": 0.9, "dt": 0.002, "tcoupl": "csvr", "tau_t": 0.5, "ref_t": 300, "pcoupl": "c-rescale", "tau_p": 1.0,
           "nstenergy": 100, "seed": 6}
    d = Path("/tmp/phys_npt")
    d.mkdir(exist_ok=True)
    sim = md.Simulation(C.sysfile("salt_eq"), cfg, output_dir=d)
    sim.run(30000)
    sim.finish()
    sim.close()
    e = md.read_energy(d / "energy.csv")
    v = e["volume"][100:]
    res["npt_volume_mean"] = float(v.mean())
    res["npt_volume_std"] = float(v.std())
    res["npt_T"] = float(e["temperature"][100:].mean())
    print("physics npt", res, flush=True)
    (C.DATA / "golden_physics.json").write_text(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["systems", "forces", "dynamics", "scale", "physics"])
    a = ap.parse_args()
    C.DATA.mkdir(parents=True, exist_ok=True)
    for name, fn in (("systems", gen_systems), ("forces", gen_forces), ("dynamics", gen_dynamics), ("scale", gen_scale),
                     ("physics", gen_physics)):
        if a.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
