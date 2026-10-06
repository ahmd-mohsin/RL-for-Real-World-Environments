"""Integrators, thermostats and barostats against trajectories recorded with the CUDA original,
plus determinism, restart and output-file contracts."""
import filecmp
import json
import math
import os
import struct

import numpy as np
import pytest

import cases as C
import mdlite as md

G = np.load(C.DATA / "golden_dynamics.npz")
ORIG = C.DATA / "orig_files"
# position tolerance per checked step: the trajectories are chaotic, so the allowed
# deviation grows with time; a wrong random stream fails at step 1 already
X_TOL = {0: 1e-6, 1: 2e-6, 2: 4e-6, 10: 2e-5, 50: 5e-4, 100: 1e-2}
E_REL = {1: 2e-6, 2: 2e-6, 10: 1e-5, 50: 1e-4, 100: 2e-3}


def _dyn(name):
    for n, s, extra in C.DYN_CASES:
        if n == name:
            return s, C.merged(C.DYN_BASE, extra)
    raise KeyError(name)


@pytest.mark.parametrize("name", [c[0] for c in C.DYN_CASES])
def test_trajectory_matches_original(name, tmp_path):
    sname, cfg = _dyn(name)
    with md.Simulation(C.sysfile(sname), dict(cfg, nstxout=10, nstvout=10, nstcheckpoint=50), output_dir=tmp_path) as sim:
        for k in C.DYN_CHECK_STEPS:
            sim.run(k - sim.step)
            assert sim.step == k
            dx = np.abs(sim.positions - G[f"{name}/x{k}"]).max()
            assert dx <= X_TOL[k], f"step {k}: max position deviation {dx:.3e} nm"
            dv = np.abs(sim.velocities - G[f"{name}/v{k}"]).max()
            assert dv <= 100 * X_TOL[k], f"step {k}: max velocity deviation {dv:.3e}"
            dl = np.abs(sim.box - G[f"{name}/L{k}"]).max()
            assert dl <= 0.1 * X_TOL[k], f"step {k}: box deviation {dl:.3e}"
            e = sim.energies()
            ge = G[f"{name}/e{k}"]
            for t, ref in zip(md.TERMS, ge):
                if t == "pressure":
                    assert abs(e[t] - ref) <= 0.5 + 2e3 * E_REL[k], f"step {k} pressure {e[t]} vs {ref}"
                else:
                    n = sim.natoms
                    assert abs(e[t] - ref) <= E_REL[k] * abs(ref) + 2e-6 * n * (1 + k), f"step {k} {t}: {e[t]} vs {ref}"
        sim.finish()
    # output files: identical layout, same rows and frames as the original's
    ref_dir = ORIG / name
    ea, eb = md.read_energy(tmp_path / "energy.csv"), md.read_energy(ref_dir / "energy.csv")
    assert list(ea) == list(eb)
    np.testing.assert_array_equal(ea["step"], eb["step"])
    np.testing.assert_array_equal(ea["time"], eb["time"])
    head_a = (tmp_path / "energy.csv").read_text().splitlines()[0]
    assert head_a == (ref_dir / "energy.csv").read_text().splitlines()[0]
    meta = json.loads((ref_dir / "meta.json").read_text())
    ta, tb = md.read_trajectory(tmp_path / "traj.mdtrj"), md.read_trajectory(ref_dir / "traj.mdtrj")
    assert (tmp_path / "traj.mdtrj").stat().st_size == meta["traj_size"]
    assert (tmp_path / "traj.mdtrj").read_bytes()[:32] == (ref_dir / "traj.mdtrj").read_bytes()[:32]
    np.testing.assert_array_equal(ta["steps"][:2], tb["steps"])
    assert list(ta["steps"]) == list(range(0, 101, 10))
    assert np.abs(ta["positions"][:2] - tb["positions"]).max() <= 2e-5
    assert np.abs(ta["velocities"][:2] - tb["velocities"]).max() <= 2e-3
    cps = {p.name: p.stat().st_size for p in tmp_path.glob("state_*.mdcpt")}
    assert cps == meta["checkpoints"]
    for p in cps:
        c = md.read_checkpoint(tmp_path / p)
        assert c["step"] == int(p[6:-6]) and c["seed"] == cfg["seed"] and c["dt"] == cfg["dt"]
        assert np.abs(c["positions"] - G[f"{name}/x{c['step']}"]).max() <= X_TOL[c["step"]]


@pytest.mark.parametrize("name", ["csvr", "baoab_crescale", "alkane_cons_baoab", "berendsen"])
def test_bitwise_determinism_and_chunking(name, tmp_path):
    """Same input -> bitwise identical results, however the run is split into calls and
    whether or not energies, frames and checkpoints are written."""
    sname, cfg = _dyn(name)
    outs = []
    for k, (split, out) in enumerate((((60,), True), ((60,), True), ((1, 13, 7, 39), False), ((25, 35), True))):
        d = tmp_path / f"r{k}"
        d.mkdir()
        c = dict(cfg) if out else dict(cfg, nstenergy=0)
        with md.Simulation(C.sysfile(sname), c, output_dir=d if out else None) as sim:
            for n in split:
                sim.run(n)
            if k == 3:
                sim.energies()          # computing energies must not perturb the state
                sim.forces()
                sim.run(0)
            outs.append((sim.positions, sim.velocities, sim.box, sim.energies()))
    for k in range(1, 4):
        np.testing.assert_array_equal(outs[k][0], outs[0][0])
        np.testing.assert_array_equal(outs[k][1], outs[0][1])
        np.testing.assert_array_equal(outs[k][2], outs[0][2])
        for t in md.TERMS:
            assert outs[k][3][t] == outs[0][3][t], t
    assert filecmp.cmp(tmp_path / "r0" / "energy.csv", tmp_path / "r1" / "energy.csv", shallow=False)
    assert filecmp.cmp(tmp_path / "r0" / "energy.csv", tmp_path / "r3" / "energy.csv", shallow=False)


@pytest.mark.parametrize("name", ["nve_comm", "csvr", "baoab_crescale", "alkane_cons_baoab", "crescale"])
def test_restart_is_bitwise_continuous(name, tmp_path):
    sname, cfg = _dyn(name)
    cfg = dict(cfg, nstenergy=3)
    a = tmp_path / "a"; b1 = tmp_path / "b1"; b2 = tmp_path / "b2"
    for d in (a, b1, b2):
        d.mkdir()
    with md.Simulation(C.sysfile(sname), cfg, output_dir=a) as sim:
        sim.run(90)
        sim.save_checkpoint(a / "end.mdcpt")
    with md.Simulation(C.sysfile(sname), cfg, output_dir=b1) as sim:
        sim.run(33)                       # 33 is not a multiple of nstenergy, nsttcouple or nstpcouple
        sim.save_checkpoint(b1 / "mid.mdcpt")
    with md.Simulation(C.sysfile(sname), cfg, output_dir=b2) as sim:
        sim.load_checkpoint(b1 / "mid.mdcpt")
        assert sim.step == 33
        sim.run(57)
        sim.save_checkpoint(b2 / "end.mdcpt")
    assert (a / "end.mdcpt").read_bytes() == (b2 / "end.mdcpt").read_bytes()
    ra = (a / "energy.csv").read_text().splitlines()
    rb1 = (b1 / "energy.csv").read_text().splitlines()
    rb2 = (b2 / "energy.csv").read_text().splitlines()
    step = lambda r: int(r.split(",")[0])
    assert rb1[0] == ra[0] == rb2[0]
    assert rb1[1:] == [r for r in ra[1:] if step(r) <= 33]
    assert rb2[1:] == [r for r in ra[1:] if step(r) >= 33]


def test_checkpoints_cross_load_with_original(tmp_path):
    """A checkpoint written by the original engine restarts the port on the original's trajectory."""
    name = "crescale"
    sname, cfg = _dyn(name)
    cp = ORIG / name / "state_50.mdcpt"
    with md.Simulation(C.sysfile(sname), cfg) as sim:
        sim.load_checkpoint(cp)
        assert sim.step == 50
        np.testing.assert_array_equal(sim.positions, md.read_checkpoint(cp)["positions"])
        sim.run(50)
        assert np.abs(sim.positions - G[f"{name}/x100"]).max() <= 2e-3
        sim.save_checkpoint(tmp_path / "p.mdcpt")
    raw = (tmp_path / "p.mdcpt").read_bytes()
    ref = (ORIG / name / "state_50.mdcpt").read_bytes()
    assert len(raw) == len(ref) and raw[:8] == ref[:8]
    c = md.read_checkpoint(tmp_path / "p.mdcpt")
    assert c["step"] == 100 and c["seed"] == cfg["seed"]


def test_rng_streams_are_exact():
    """Generated velocities follow the documented Philox stream to the last bits."""
    sname, cfg = _dyn("genvel_rf")
    with md.Simulation(C.sysfile(sname), dict(cfg, constraints="no", nstcomm=0)) as sim:
        v = sim.velocities
        e = sim.energies()
    s = md.load_system(C.sysfile(sname))
    # independent implementation of the stream (gen_vel, stream 2, step 0) and the exact rescaling
    M0, M1, W0, W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
    seed = cfg["seed"]

    def philox(c0):
        x, y, z, w = c0, 0, 0, 2
        k0, k1 = seed & 0xFFFFFFFF, seed >> 32
        for _ in range(10):
            p0, p1 = M0 * x, M1 * z
            x, y, z, w = ((p1 >> 32) ^ y ^ k0) & 0xFFFFFFFF, p1 & 0xFFFFFFFF, ((p0 >> 32) ^ w ^ k1) & 0xFFFFFFFF, p0 & 0xFFFFFFFF
            k0, k1 = (k0 + W0) & 0xFFFFFFFF, (k1 + W1) & 0xFFFFFFFF
        return x, y, z, w

    def u53(a, b):
        return ((a >> 5) * 67108864.0 + (b >> 6) + 0.5) * (1.0 / 9007199254740992.0)

    def g2(r):
        rad = math.sqrt(-2.0 * math.log(u53(r[0], r[1])))
        th = 2.0 * math.pi * u53(r[2], r[3])
        return rad * math.cos(th), rad * math.sin(th)

    kT = md.BOLTZ * cfg["gen_temp"]
    for i in (0, 1, 17, s.natoms - 1):
        a0, a1 = g2(philox(2 * i))
        a2, _ = g2(philox(2 * i + 1))
        raw = np.array([a0, a1, a2]) * math.sqrt(kT / s.mass[i])
        ratio = v[i] / raw
        assert np.allclose(ratio, ratio[0], rtol=1e-12), i
    assert abs(e["temperature"] - cfg["gen_temp"]) <= 1e-9 * cfg["gen_temp"]


def test_trajectory_frame_format(tmp_path):
    sname, cfg = _dyn("nve")
    with md.Simulation(C.sysfile(sname), dict(cfg, nstxout=7, nstvout=7, nstenergy=0), output_dir=tmp_path) as sim:
        sim.run(15)
        x15 = sim.positions
    raw = (tmp_path / "traj.mdtrj").read_bytes()
    n = sim.natoms
    assert raw[:8] == b"MDTRJ\x00\x01\x00"
    assert struct.unpack_from("<iidii", raw, 8) == (n, 1, cfg["dt"], 7, 0)
    fsize = 40 + 24 * n
    assert len(raw) == 32 + 3 * fsize
    step, t = struct.unpack_from("<qd", raw, 32 + 2 * fsize)
    assert step == 14 and t == 14 * cfg["dt"]
    assert not (tmp_path / "energy.csv").read_text().splitlines()[1:]
    tr = md.read_trajectory(tmp_path / "traj.mdtrj")
    assert np.abs(tr["positions"][-1] - x15).max() < 1e-1


def test_outputs_flushed_on_finish(tmp_path):
    sname, cfg = _dyn("nve")
    sim = md.Simulation(C.sysfile(sname), dict(cfg, nstenergy=4), output_dir=tmp_path)
    sim.run(8)
    sim.finish()
    rows = (tmp_path / "energy.csv").read_text().splitlines()
    assert [r.split(",")[0] for r in rows[1:]] == ["0", "4", "8"]
    sim.run(0)
    sim.finish()
    assert len((tmp_path / "energy.csv").read_text().splitlines()) == 4
    sim.close()
