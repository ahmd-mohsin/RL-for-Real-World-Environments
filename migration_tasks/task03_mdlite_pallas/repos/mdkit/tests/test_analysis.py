import numpy as np

import mdbuild as mb
import mdkit
import mdlite as md


def test_rdf_ideal_gas():
    rng = np.random.default_rng(0)
    L = np.array([5.0, 5.0, 5.0])
    x = rng.uniform(0, 5, (3000, 3))
    r, g = mdkit.radial_distribution(x, L, np.arange(3000), np.arange(3000), rmax=2.0, nbins=20)
    assert abs(g[5:].mean() - 1.0) < 0.03


def test_short_run_analysis(tmp_path):
    w = mb.spce_water(400, seed=5)
    cfg = {"cutoff": 0.9, "integrator": "baoab", "tau_t": 0.1, "ref_t": 300, "dt": 0.001, "nstenergy": 50, "nstxout": 100, "seed": 3}
    with md.Simulation(w, cfg, output_dir=tmp_path) as sim:
        sim.run(500)
        sim.finish()
    traj = md.read_trajectory(tmp_path / "traj.mdtrj")
    en = md.read_energy(tmp_path / "energy.csv")
    assert list(traj["steps"]) == [0, 100, 200, 300, 400, 500]
    assert list(en["step"]) == list(range(0, 501, 50))
    assert mdkit.water_geometry_errors(traj["positions"][-1], np.arange(0, 1200, 3)) < 1e-5
    t, m = mdkit.msd(traj)
    assert m[0] == 0 and m[-1] > 0
    assert 900 < mdkit.density(w.mass, [en["box_x"][-1], en["box_y"][-1], en["box_z"][-1]]) < 1100


def test_rerun_matches_energy_file(tmp_path):
    w = mb.lj_fluid(1000, seed=2)
    cfg = {"cutoff": 1.0, "coulombtype": "cutoff", "dt": 0.002, "nstenergy": 20, "nstxout": 20, "gen_vel": True, "gen_temp": 120}
    with md.Simulation(w, cfg, output_dir=tmp_path) as sim:
        sim.run(100)
        sim.finish()
    traj = md.read_trajectory(tmp_path / "traj.mdtrj")
    en = md.read_energy(tmp_path / "energy.csv")
    re = mdkit.rerun(w, cfg, traj)
    pot = np.array([e["potential"] for e in re])
    # single-precision trajectory coordinates: agreement to ~1e-5 relative
    np.testing.assert_allclose(pot, en["potential"], rtol=2e-5)


def test_nve_drift_small(tmp_path):
    w = mb.lj_fluid(2000, density=18.0, seed=4, jitter=0.02)
    cfg = {"cutoff": 1.0, "coulombtype": "cutoff", "dt": 0.002, "nstenergy": 50, "gen_vel": True, "gen_temp": 100, "nstcomm": 0}
    with md.Simulation(w, cfg, output_dir=tmp_path) as sim:
        sim.run(1000)
        sim.finish()
    en = md.read_energy(tmp_path / "energy.csv")
    assert abs(mdkit.energy_drift(en, "total", w.natoms)) < 0.05
