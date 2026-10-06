"""API, configuration, input validation, CLI and implementation-contract tests."""
import json
import struct
import subprocess
import sys
import zlib

import numpy as np
import pytest

import cases as C
import mdlite as md
from mdlite import System, make_types

TERMS = ("lj", "lj_14", "disp_corr", "coulomb_sr", "coulomb_14", "coulomb_recip", "bond", "angle", "dihedral", "potential",
         "kinetic", "total", "conserved", "temperature", "pressure", "volume")


def small():
    return md.load_system(C.sysfile("three_atoms"))


def test_terms_and_error_type():
    assert tuple(md.TERMS) == TERMS
    assert issubclass(md.MdliteError, RuntimeError)
    with md.Simulation(small(), {"cutoff": 1.0}) as sim:
        e = sim.energies()
        assert set(TERMS) <= set(e) and e["virial"].shape == (3, 3) and e["pressure_tensor"].shape == (3, 3)
        assert sim.forces().shape == (3, 3) and sim.positions.dtype == np.float64
        assert sim.step == 0 and sim.dof == 9 - 3


BAD_CONFIGS = [
    "cutoff = 2.0",                      # box too small
    "bogus = 1",
    "dt = 0",
    "dt = 0.0101",
    "dt = nan",
    "dt = 1e999",
    "dt = 0.002x",
    "nsteps = -1",
    "nsteps = 1.5",
    "seed = -3",
    "pme_order = 2",
    "pme_order = 9",
    "coulombtype = ewald",
    "vdw_modifier = force-switch\nrvdw_switch = 1.0",
    "epsilon_rf = 0.5",
    "ewald_rtol = 1",
    "tcoupl = csvr\nintegrator = baoab",
    "nstvout = 5\nnstxout = 10",
    "nstenergy = -5",
    "shake_tol = 0.01",
    "dt = 0.002\ndt = 0.001",
    "dt 0.002",
    "  = 3",
    "gen_vel = maybe",
    "nsttcouple = 0",
    "compressibility = -1e-5",
]


@pytest.mark.parametrize("text", BAD_CONFIGS)
def test_invalid_config_raises(text):
    with pytest.raises(md.MdliteError):
        md.Simulation(small(), text if "cutoff" in text else "cutoff = 1.0\n" + text)


def test_config_syntax_accepted():
    text = ("# comment\nCUTOFF = 1.0 ; trailing comment\n Coulombtype=Reaction-Field\nepsilon-rf = 0x1.4p+1\n"
            "GEN_VEL = On\ngen_temp=1e2\n\n\tdispcorr = enerpres\r\nnstcomm = 0\n")
    with md.Simulation(small(), text) as sim:
        info = sim.info
        assert info["pme_grid"] is None and abs(info["k_rf"] - (2.5 - 1) / ((2 * 2.5 + 1) * 1.0)) < 1e-15
        assert abs(sim.energies()["temperature"] - 100.0) < 1e-9


def _write_raw(path, data):
    path.write_bytes(data)
    return str(path)


def test_invalid_systems_raise(tmp_path):
    base = small()
    bad = []
    s = base.copy(); s.mass = np.array([12.0, 0.0, 30.0]); bad.append(s)
    s = base.copy(); s.type = np.array([0, 5, 0], np.int32); bad.append(s)
    s = base.copy(); s.positions = s.positions.copy(); s.positions[1, 2] = np.nan; bad.append(s)
    s = base.copy(); s.box = np.array([2.5, -2.5, 2.5]); bad.append(s)
    s = base.copy(); s.bonds = np.array([(0, 1, 0.1, 100.0)], dtype=System.dtypes["bonds"]); bad.append(s)     # different molecules
    s = base.copy(); s.constraints = np.array([(1, 1, 0.1)], dtype=System.dtypes["constraints"]); bad.append(s)
    s = base.copy(); s.pairs = np.array([(0, 3)], dtype=System.dtypes["pairs"]); bad.append(s)
    s = base.copy(); s.nrexcl = 6; bad.append(s)
    s = base.copy(); s.molecule = np.array([0, 0, 0], np.int32)
    s.dihedrals = np.array([(0, 1, 2, 0, 7, 0.0, 1.0)], dtype=System.dtypes["dihedrals"]); bad.append(s)
    for k, sysm in enumerate(bad):
        with pytest.raises(md.MdliteError):
            md.Simulation(sysm, {"cutoff": 1.0})
        p = tmp_path / f"b{k}.mdsys"
        sysm.save(p)
        with pytest.raises(md.MdliteError):
            md.Simulation(str(p), {"cutoff": 1.0})
    good = base.to_bytes()
    corrupt = bytearray(good); corrupt[40] ^= 1
    for data in (good[:-9] + struct.pack("<I", zlib.crc32(good[:-9]) & 0xFFFFFFFF), bytes(corrupt), b"MDSYS\x00\x02\x00" + good[8:]):
        with pytest.raises(md.MdliteError):
            md.Simulation(_write_raw(tmp_path / "x.mdsys", data), {"cutoff": 1.0})
    with pytest.raises(md.MdliteError):
        md.Simulation(str(tmp_path / "missing.mdsys"), {"cutoff": 1.0})


def test_checkpoint_errors(tmp_path):
    with md.Simulation(small(), {"cutoff": 1.0}) as sim:
        sim.save_checkpoint(tmp_path / "ok.mdcpt")
        raw = (tmp_path / "ok.mdcpt").read_bytes()
        for k, data in enumerate((raw[:-1], b"MDCPT\x00\x01\x00" + raw[8:], raw[:30] + bytes([raw[30] ^ 4]) + raw[31:])):
            p = tmp_path / f"bad{k}.mdcpt"
            p.write_bytes(data)
            with pytest.raises(md.MdliteError):
                sim.load_checkpoint(p)
        with pytest.raises(md.MdliteError):
            sim.load_checkpoint(tmp_path / "nope.mdcpt")
        with md.Simulation(md.load_system(C.sysfile("one_atom")), {"cutoff": 1.0}) as other:
            with pytest.raises(md.MdliteError):
                other.load_checkpoint(tmp_path / "ok.mdcpt")
        assert sim.step == 0
        with pytest.raises(md.MdliteError):
            sim.positions = np.zeros((2, 3))
        with pytest.raises(md.MdliteError):
            sim.velocities = np.full((3, 3), np.inf)
        with pytest.raises(md.MdliteError):
            sim.run(-1)


def test_shake_failure_is_reported():
    s = md.load_system(C.sysfile("water"))
    with md.Simulation(s, {"cutoff": 0.9, "dt": 0.002}) as sim:
        v = sim.velocities
        v[:30] = 400.0 * np.random.default_rng(0).normal(size=(30, 3))   # absurd velocities rotate bonds too far
        sim.velocities = v
        with pytest.raises(md.MdliteError, match="SHAKE|RATTLE"):
            sim.run(5)
    with pytest.raises(md.MdliteError):
        md.Simulation(s, {"cutoff": 0.9, "shake_maxiter": 1, "shake_tol": 1e-12, "gen_vel": True})


def test_box_shrinking_below_cutoff_is_an_error():
    s = md.load_system(C.sysfile("salt_eq"))
    with md.Simulation(s, {"cutoff": 0.9, "pcoupl": "berendsen", "compressibility": 0.5, "tau_p": 0.01, "ref_p": 1e5,
                           "nstpcouple": 1, "dt": 0.001}) as sim:
        with pytest.raises(md.MdliteError, match="box"):
            sim.run(500)


def test_cli_matches_api(tmp_path):
    sysp = C.sysfile("alkane_cons_eq")
    cfgp = tmp_path / "run.cfg"
    cfgp.write_text("cutoff = 1.0\nintegrator = baoab\ntau_t = 0.3\nnsteps = 40\nnstenergy = 10\nnstxout = 20\nseed = 9\n")
    out = tmp_path / "cli"
    r = subprocess.run([sys.executable, "-m", "mdlite.cli", "run", "-s", sysp, "-c", str(cfgp), "-o", str(out)],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr
    info = json.loads((out / "log.json").read_text())
    assert info["final_step"] == 40
    with pytest.raises(md.MdliteError):
        md.Simulation(sysp, cfgp, output_dir=tmp_path / "does" / "not" / "exist")
    (tmp_path / "api").mkdir()
    with md.Simulation(sysp, cfgp, output_dir=tmp_path / "api") as sim:
        sim.run(40)
        sim.finish()
        x = sim.positions
    np.testing.assert_array_equal(md.read_checkpoint(out / "final.mdcpt")["positions"], x)
    assert (out / "energy.csv").read_bytes() == (tmp_path / "api" / "energy.csv").read_bytes()
    r = subprocess.run([sys.executable, "-m", "mdlite.cli", "run", "-s", sysp, "-c", str(cfgp), "-o", str(tmp_path / "c2"),
                        "--checkpoint", str(out / "final.mdcpt"), "--nsteps", "5"], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0 and md.read_checkpoint(tmp_path / "c2" / "final.mdcpt")["step"] == 45
    bad = tmp_path / "bad.cfg"
    bad.write_text("cutoff = 9\n")
    r = subprocess.run([sys.executable, "-m", "mdlite.cli", "energy", "-s", sysp, "-c", str(bad)], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 1 and "error" in r.stderr


PROBE = r"""
import json, sys
import jax
from jax.experimental import pallas as pl
names = []
orig = pl.pallas_call
def wrapped(*a, **k):
    f = orig(*a, **k)
    kern = a[0] if a else k.get("kernel")
    tag = k.get("name") or getattr(kern, "__name__", None) or getattr(getattr(kern, "func", None), "__name__", "?")
    def call(*args, **kw):
        names.append(tag)
        return f(*args, **kw)
    return call
pl.pallas_call = wrapped
import jax.experimental.pallas
jax.experimental.pallas.pallas_call = wrapped
import mdlite as md
sys.path.insert(0, %(hidden)r)
import cases as C
calls = {}
for key, sname, cfg in (("pme", "salt_eq", {"cutoff": 0.9, "nstenergy": 0}), ("rf", "alkane_cons_eq", {"cutoff": 1.0, "coulombtype": "reaction-field"})):
    names.clear()
    with md.Simulation(C.sysfile(sname), cfg) as sim:
        sim.run(5)
        sim.energies()
    calls[key] = list(names)
print(json.dumps(calls))
"""


def test_pallas_kernels_are_used():
    import os
    hidden = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, "-c", PROBE % {"hidden": hidden}], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    calls = json.loads(r.stdout.strip().splitlines()[-1])
    # the hot paths (pair forces, neighbour search, PME spread / interpolation, constraints)
    # must trace Pallas kernels; a pure-XLA implementation traces none
    assert len(calls["pme"]) >= 4, calls
    assert len(calls["rf"]) >= 2, calls
