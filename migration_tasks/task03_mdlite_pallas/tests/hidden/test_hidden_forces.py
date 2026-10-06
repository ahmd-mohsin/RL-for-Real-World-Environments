"""Forces, energy terms, virial and pressure against goldens recorded with the CUDA original."""
import json
import math

import numpy as np
import pytest

import cases as C
import mdlite as md

G = np.load(C.DATA / "golden_forces.npz")
ENERGY_TERMS = ("lj", "lj_14", "disp_corr", "coulomb_sr", "coulomb_14", "coulomb_recip", "bond", "angle", "dihedral",
                "potential", "kinetic", "total", "temperature", "volume")

# tolerances (mixed precision is allowed; the reference port is ~10x inside them)
F_RELRMS = 4e-5          # RMS force error relative to the RMS force ...
F_ABSRMS = 5e-3          # ... or absolute (kJ/mol/nm), for systems with tiny or strongly cancelling net forces
F_MAXABS_REL = 2e-4      # max |dF| relative to max |F| ...
F_MAXABS = 0.05          # ... or absolute
E_REL = 2e-5             # per energy term, relative ...
E_ABS_PER_ATOM = 2e-6    # ... plus an absolute allowance (kJ/mol per atom)
P_ABS = 0.5              # bar, scalar pressure and pressure tensor
VIR_REL = 2e-5


def run_case(sname, cfg):
    with md.Simulation(C.sysfile(sname), cfg) as sim:
        return sim.forces(), sim.energies(), sim.info, sim.natoms


@pytest.mark.parametrize("name,sname,cfg", C.FORCE_CASES, ids=[c[0] for c in C.FORCE_CASES])
def test_forces_energies(name, sname, cfg):
    f, e, info, n = run_case(sname, cfg)
    fg = G[f"{name}/f"]
    assert f.shape[1] == 3 and f.dtype == np.float64 and f.shape[0] == n
    fall = f
    f = f[:fg.shape[0]]
    assert np.all(np.isfinite(f))
    absrms = math.sqrt(float(((f - fg) ** 2).sum(1).mean()))
    assert C.relrms(f, fg) <= F_RELRMS or absrms <= F_ABSRMS, f"force RMS error {C.relrms(f, fg):.3e} (abs {absrms:.3e})"
    maxerr = np.abs(f - fg).max()
    assert maxerr <= max(F_MAXABS_REL * np.abs(fg).max(), F_MAXABS), f"max force error {maxerr:.3e}"
    for t in ENERGY_TERMS:
        ref = float(G[f"{name}/e/{t}"])
        tol = E_REL * abs(ref) + E_ABS_PER_ATOM * n
        assert abs(e[t] - ref) <= tol, f"{t}: {e[t]!r} vs {ref!r} (tol {tol:.2e})"
    pref = float(G[f"{name}/e/pressure"])
    assert abs(e["pressure"] - pref) <= P_ABS + 2e-5 * abs(pref), f"pressure {e['pressure']} vs {pref}"
    vg = G[f"{name}/e/virial"]
    scale = np.abs(vg).max() + E_ABS_PER_ATOM * n
    assert np.abs(e["virial"] - vg).max() <= VIR_REL * scale + 1e-3, "virial"
    assert np.allclose(e["virial"], e["virial"].T, rtol=0, atol=1e-9 * scale + 1e-12), "virial must be symmetric"
    pg = G[f"{name}/e/pressure_tensor"]
    assert np.abs(e["pressure_tensor"] - pg).max() <= P_ABS + 2e-5 * np.abs(pg).max()
    # force balance without PME (smooth PME does not conserve momentum exactly): the
    # total force vanishes up to rounding
    if cfg.get("coulombtype", "pme") != "pme":
        assert np.abs(fall.sum(0)).max() <= 1e-5 * max(1.0, np.abs(fg).max()) * math.sqrt(n) + 1e-3
    # derived configuration values must match exactly (or to the last bits for floats)
    gi = json.loads(str(G[f"{name}/info"]))
    for k in ("natoms", "dof", "constraints", "constraint_clusters", "exclusion_pairs", "pme_grid"):
        assert info[k] == gi[k], f"info[{k}] = {info[k]} vs {gi[k]}"
    for k in ("rlist", "ewald_beta", "k_rf", "c_rf"):
        if gi.get(k) is None:
            assert info.get(k) is None
        else:
            assert math.isclose(info[k], gi[k], rel_tol=1e-12, abs_tol=1e-15), f"info[{k}]"


def test_energy_is_consistent_with_forces():
    """Central finite differences of the potential reproduce the forces (no stale/partial terms)."""
    rng = np.random.default_rng(7)
    for sname, cfg in (("exotic", {"cutoff": 1.0, "fourier_spacing": 0.1}), ("alkane_cons", {"cutoff": 1.0}),
                       ("three_atoms", {"cutoff": 1.0, "pme_order": 5})):
        with md.Simulation(C.sysfile(sname), cfg) as sim:
            x0 = sim.positions
            f = sim.forces()
            for i in rng.choice(sim.natoms, size=min(5, sim.natoms), replace=False):
                for d in range(3):
                    h = 2e-4
                    xp = x0.copy(); xp[i, d] += h
                    sim.positions = xp
                    ep = sim.energies()["potential"]
                    xm = x0.copy(); xm[i, d] -= h
                    sim.positions = xm
                    em = sim.energies()["potential"]
                    fd = -(ep - em) / (2 * h)
                    assert abs(fd - f[i, d]) <= 2e-3 * max(1.0, abs(f[i, d])) + 0.5, (sname, i, d, fd, f[i, d])
            sim.positions = x0
            np.testing.assert_array_equal(sim.forces(), f)


def test_translation_and_image_invariance():
    """Moving atoms by whole box vectors (each molecule independently, thousands of boxes away) leaves
    forces and energies unchanged; a continuous translation does too without PME."""
    s = md.load_system(C.sysfile("salt"))
    rng = np.random.default_rng(3)
    mol = s.molecule
    shifts = rng.integers(-3000, 3000, size=(mol.max() + 1, 3)) * s.box
    with md.Simulation(s, {"cutoff": 1.0}) as sim:
        f0, e0 = sim.forces(), sim.energies()["potential"]
        sim.positions = s.positions + shifts[mol]
        f1, e1 = sim.forces(), sim.energies()["potential"]
        assert C.relrms(f1, f0) <= 2e-5 and abs(e1 - e0) <= 2e-6 * abs(e0) + E_ABS_PER_ATOM * s.natoms
    with md.Simulation(s, {"cutoff": 1.0, "coulombtype": "reaction-field", "epsilon_rf": 70}) as sim:
        f0, e0 = sim.forces(), sim.energies()["potential"]
        sim.positions = s.positions + np.array([0.37, -1.91, 2.23])
        f2, e2 = sim.forces(), sim.energies()["potential"]
        assert C.relrms(f2, f0) <= 2e-5 and abs(e2 - e0) <= 2e-6 * abs(e0) + E_ABS_PER_ATOM * s.natoms


def test_set_box_and_positions_roundtrip():
    s = md.load_system(C.sysfile("lj_mix"))
    with md.Simulation(s, {"cutoff": 1.0, "coulombtype": "cutoff"}) as sim:
        f0 = sim.forces()
        sim.box = s.box * 1.01
        f1 = sim.forces()
        assert C.relrms(f1, f0) > 1e-3
        sim.box = s.box
        np.testing.assert_array_equal(sim.forces(), f0)
        with pytest.raises(md.MdliteError):
            sim.box = [2.0 * 1.05, s.box[1], s.box[2]]
        np.testing.assert_array_equal(sim.box, s.box)
        np.testing.assert_array_equal(sim.forces(), f0)
