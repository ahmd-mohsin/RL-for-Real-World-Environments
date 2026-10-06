"""Basic tests of the mdlite engine (the release suite is larger)."""
import numpy as np
import pytest

import mdlite as md
from mdlite import System, make_types


def two_atoms(r=0.4, q=(1.0, -1.0), box=5.0):
    return System(box=[box] * 3, types=make_types([("A", 0.3, 0.5)]), type=[0, 0], molecule=[0, 1], charge=list(q),
                  mass=[10.0, 10.0], positions=[[1, 1, 1], [1 + r, 1, 1]])


def test_lj_pair_potential_shift():
    s = two_atoms(q=(0.0, 0.0))
    with md.Simulation(s, {"cutoff": 1.0, "coulombtype": "cutoff"}) as sim:
        e = sim.energies()
        c6, c12 = 4 * 0.5 * 0.3 ** 6, 4 * 0.5 * 0.3 ** 12
        ref = c12 / 0.4 ** 12 - c6 / 0.4 ** 6 - (c12 - c6)
        assert abs(e["lj"] - ref) <= 1e-6 * abs(ref)
        f = sim.forces()
        fref = (12 * c12 / 0.4 ** 13 - 6 * c6 / 0.4 ** 7)
        assert abs(f[1, 0] - fref) <= 1e-6 * abs(fref) and abs(f[0, 0] + fref) <= 1e-6 * abs(fref)


def test_reaction_field_pair():
    s = two_atoms(r=0.5)
    with md.Simulation(s, {"cutoff": 1.0, "coulombtype": "reaction-field", "epsilon_rf": 0}) as sim:
        e = sim.energies()
        krf, crf = 0.5, 1.5
        pair = md.ONE_4PI_EPS0 * -1.0 * (1 / 0.5 + krf * 0.25 - crf)
        self_term = -0.5 * md.ONE_4PI_EPS0 * crf * 2.0
        assert abs(e["coulomb_sr"] - (pair + self_term)) <= 1e-6 * abs(pair + self_term)


def test_config_errors():
    s = two_atoms()
    with pytest.raises(md.MdliteError):
        md.Simulation(s, {"cutoff": 3.0})  # box too small
    with pytest.raises(md.MdliteError):
        md.Simulation(s, {"bogus": 1})
    with pytest.raises(md.MdliteError):
        md.Simulation(s, {"pme_order": 9})


def test_checkpoint_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    s = System(box=[3.0] * 3, types=make_types([("A", 0.3, 0.8)]), type=np.zeros(n), molecule=np.arange(n),
               charge=np.zeros(n), mass=np.full(n, 20.0), positions=np.stack(np.meshgrid(*[np.arange(7)] * 3), -1).reshape(-1, 3)[:n] * 0.42 + 0.1)
    cfg = {"cutoff": 1.0, "coulombtype": "cutoff", "gen_vel": True, "gen_temp": 150, "seed": 4}
    with md.Simulation(s, cfg) as a:
        a.run(40)
        a.save_checkpoint(tmp_path / "c.mdcpt")
        a.run(40)
        xa = a.positions
    with md.Simulation(s, cfg) as b:
        b.load_checkpoint(tmp_path / "c.mdcpt")
        assert b.step == 40
        b.run(40)
        np.testing.assert_array_equal(b.positions, xa)
    cp = md.read_checkpoint(tmp_path / "c.mdcpt")
    assert cp["step"] == 40 and cp["natoms"] == n
