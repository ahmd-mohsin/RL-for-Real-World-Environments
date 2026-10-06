"""Long-run physics against statistics of the CUDA original, and large systems."""
import json

import numpy as np
import pytest

import cases as C
import mdlite as md

PHYS = json.loads((C.DATA / "golden_physics.json").read_text())
SCALE = np.load(C.DATA / "golden_scale.npz")


def test_nve_energy_conservation(tmp_path):
    """40 ps of salt water at a 2 fs time step with SHAKE/RATTLE and PME."""
    cfg = {"cutoff": 0.9, "dt": 0.002, "nstcomm": 0, "nstenergy": 250, "seed": 5}
    with md.Simulation(C.sysfile("salt_eq"), cfg, output_dir=tmp_path) as sim:
        sim.run(20000)
        sim.finish()
        x = sim.positions
        s = sim.system
    e = md.read_energy(tmp_path / "energy.csv")
    n = s.natoms
    fit = np.polyfit(e["time"], e["total"], 1)
    drift = fit[0] * 1000.0 / n
    fluct = np.std(e["total"] - np.polyval(fit, e["time"]))
    assert abs(drift) <= max(5 * abs(PHYS["nve_drift"]), 0.003), f"energy drift {drift:.4g} kJ/mol/ns/atom"
    assert fluct <= 2.0 * PHYS["nve_fluct"], f"energy fluctuation {fluct:.3g} vs {PHYS['nve_fluct']:.3g}"
    assert abs(e["temperature"].mean() - PHYS["nve_T"]) <= 3.0
    # constraints hold to the SHAKE tolerance after 20000 steps
    c = s.constraints
    d = np.linalg.norm(x[c["i"]] - x[c["j"]], axis=1)
    assert np.abs(d / c["d"] - 1).max() <= 2e-10


def test_npt_density_and_temperature(tmp_path):
    cfg = {"cutoff": 0.9, "dt": 0.002, "tcoupl": "csvr", "tau_t": 0.5, "ref_t": 300, "pcoupl": "c-rescale", "tau_p": 1.0,
           "nstenergy": 100, "seed": 6}
    with md.Simulation(C.sysfile("salt_eq"), cfg, output_dir=tmp_path) as sim:
        sim.run(30000)
        sim.finish()
    e = md.read_energy(tmp_path / "energy.csv")
    v = e["volume"][100:]
    assert abs(v.mean() - PHYS["npt_volume_mean"]) <= 0.35, f"volume {v.mean():.3f} vs {PHYS['npt_volume_mean']:.3f}"
    assert 0.6 * PHYS["npt_volume_std"] <= v.std() <= 1.6 * PHYS["npt_volume_std"], "volume fluctuations (c-rescale noise)"
    assert abs(e["temperature"][100:].mean() - PHYS["npt_T"]) <= 2.0
    assert abs(e["temperature"][100:].mean() - 300.0) <= 2.0


@pytest.mark.parametrize("name,sname,cfg,n", C.SCALE_CASES, ids=[c[0] for c in C.SCALE_CASES])
def test_large_systems(name, sname, cfg, n):
    with md.Simulation(C.sysfile(sname), cfg) as sim:
        sim.run(n)
        e = sim.energies()
        x = sim.positions
        L = sim.box
    np.testing.assert_allclose(L, SCALE[f"{name}/L"], rtol=1e-12)
    assert np.abs(x[:C.SCALE_SUBSET] - SCALE[f"{name}/x"]).max() <= 2e-4
    ge = SCALE[f"{name}/e"]
    for t, ref in zip(md.TERMS, ge):
        if t == "pressure":
            assert abs(e[t] - ref) <= 1.0, t
        else:
            assert abs(e[t] - ref) <= 2e-5 * abs(ref) + 1e-5 * sim.natoms, (t, e[t], ref)
