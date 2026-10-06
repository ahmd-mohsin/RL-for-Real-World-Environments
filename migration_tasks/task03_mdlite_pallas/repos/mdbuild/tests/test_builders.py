import numpy as np
import pytest

import mdbuild as mb
import mdlite as md


def test_water_geometry_and_topology():
    s = mb.spce_water(200, seed=3)
    assert s.natoms == 600
    assert len(s.constraints) == 600
    x = s.positions
    oh = np.linalg.norm(x[0::3] - x[1::3], axis=1)
    hh = np.linalg.norm(x[1::3] - x[2::3], axis=1)
    np.testing.assert_allclose(oh, mb.builders.SPCE_OH, rtol=1e-12)
    np.testing.assert_allclose(hh, mb.builders.SPCE_HH, rtol=1e-12)
    assert abs(s.charge.sum()) < 1e-9


def test_roundtrip(tmp_path):
    s = mb.alkane_melt(6, 8, charges=True, seed=1)
    p = tmp_path / "a.mdsys"
    s.save(p)
    t = md.load_system(p)
    assert t.natoms == s.natoms
    np.testing.assert_array_equal(t.positions, s.positions)
    np.testing.assert_array_equal(t.dihedrals, s.dihedrals)
    assert t.fudge_qq == s.fudge_qq


def test_deterministic():
    a = mb.salt_water(100, 5, seed=9)
    b = mb.salt_water(100, 5, seed=9)
    np.testing.assert_array_equal(a.positions, b.positions)
    assert abs(a.charge.sum()) < 1e-9


def test_alkane_counts():
    s = mb.alkane_melt(4, 10, seed=0)
    assert len(s.bonds) == 4 * 9 and len(s.angles) == 4 * 8 and len(s.dihedrals) == 4 * 7 * 3 and len(s.pairs) == 4 * 7


def test_crystal_and_charge():
    c = mb.nacl_crystal(2)
    assert c.natoms == 64 and abs(c.charge.sum()) < 1e-12
    q = mb.add_net_charge(mb.spce_water(50, seed=1), 2)
    assert abs(q.charge.sum() - 2.0) < 1e-9
    assert len(q.constraints) == 3 * 48


@pytest.mark.parametrize("builder", ["lj", "water", "droplet"])
def test_engine_accepts(builder):
    s = {"lj": lambda: mb.lj_fluid(500, mixture=True), "water": lambda: mb.spce_water(300),
         "droplet": lambda: mb.water_droplet(100, 4.0)}[builder]()
    with md.Simulation(s, {"cutoff": 0.9}) as sim:
        e = sim.energies()
        assert np.isfinite(e["potential"])
