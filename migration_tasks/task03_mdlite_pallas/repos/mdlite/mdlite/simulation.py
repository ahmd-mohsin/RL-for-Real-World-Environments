"""Python interface to the mdlite engine."""
from __future__ import annotations

import ctypes
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from ._lib import lib, ptr
from .config import config_text
from .system import System, load_system

TERMS = tuple(lib.mdl_term_name(k).decode() for k in range(lib.mdl_term_count()))


class MdliteError(RuntimeError):
    """Raised for invalid input and for failures inside the engine."""


def _check(rc):
    if rc != 0:
        raise MdliteError(lib.mdl_last_error().decode())


class Simulation:
    """A molecular dynamics simulation on the GPU.

    ``system``      a :class:`System` or a path to a ``.mdsys`` file
    ``config``      dict, ``key = value`` text, or a path to a config file
    ``output_dir``  if given, ``energy.csv``, ``traj.mdtrj`` and periodic checkpoints are written there
    """

    def __init__(self, system, config=None, output_dir=None):
        text = config_text(config)
        if isinstance(system, System):
            fd, tmp = tempfile.mkstemp(suffix=".mdsys")
            os.close(fd)
            try:
                system.save(tmp)
                h = lib.mdl_create(tmp.encode(), text.encode(), str(output_dir or "").encode())
            finally:
                os.unlink(tmp)
            self.system = system
        else:
            h = lib.mdl_create(str(system).encode(), text.encode(), str(output_dir or "").encode())
            self.system = load_system(system) if h else None
        if not h:
            raise MdliteError(lib.mdl_last_error().decode())
        self._h = h
        self.natoms = lib.mdl_natoms(h)
        self.output_dir = Path(output_dir) if output_dir else None

    # --- lifecycle
    def close(self):
        if getattr(self, "_h", None):
            lib.mdl_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- state
    @property
    def step(self) -> int:
        return int(lib.mdl_step(self._h))

    @property
    def dof(self) -> int:
        return int(lib.mdl_dof(self._h))

    @property
    def positions(self) -> np.ndarray:
        a = np.empty((self.natoms, 3))
        _check(lib.mdl_get_positions(self._h, ptr(a)))
        return a

    @positions.setter
    def positions(self, x):
        a = np.ascontiguousarray(x, dtype=np.float64)
        if a.shape != (self.natoms, 3):
            raise MdliteError(f"positions must have shape ({self.natoms}, 3)")
        _check(lib.mdl_set_positions(self._h, ptr(a)))

    @property
    def velocities(self) -> np.ndarray:
        a = np.empty((self.natoms, 3))
        _check(lib.mdl_get_velocities(self._h, ptr(a)))
        return a

    @velocities.setter
    def velocities(self, v):
        a = np.ascontiguousarray(v, dtype=np.float64)
        if a.shape != (self.natoms, 3):
            raise MdliteError(f"velocities must have shape ({self.natoms}, 3)")
        _check(lib.mdl_set_velocities(self._h, ptr(a)))

    @property
    def box(self) -> np.ndarray:
        a = np.empty(3)
        _check(lib.mdl_get_box(self._h, ptr(a)))
        return a

    @box.setter
    def box(self, L):
        a = np.ascontiguousarray(L, dtype=np.float64).reshape(3)
        _check(lib.mdl_set_box(self._h, ptr(a)))

    @property
    def info(self) -> dict:
        buf = ctypes.create_string_buffer(4096)
        _check(lib.mdl_info(self._h, buf, len(buf)))
        return json.loads(buf.value.decode())

    # --- computation
    def run(self, nsteps: int) -> None:
        if int(nsteps) < 0:
            raise MdliteError("nsteps must be >= 0")
        _check(lib.mdl_run(self._h, int(nsteps)))

    def finish(self) -> None:
        """Write the outputs of the current step (if due) and flush the output files."""
        _check(lib.mdl_finish(self._h))

    def forces(self) -> np.ndarray:
        a = np.empty((self.natoms, 3))
        _check(lib.mdl_forces(self._h, ptr(a)))
        return a

    def energies(self) -> dict:
        """Energy terms (kJ/mol), temperature (K), pressure (bar), volume (nm^3),
        plus ``virial`` (sum of r_ij (x) f_ij, kJ/mol) and ``pressure_tensor`` (bar) as 3x3 arrays."""
        t = np.empty(len(TERMS))
        vir = np.empty(9)
        pres = np.empty(9)
        _check(lib.mdl_energies(self._h, ptr(t), ptr(vir), ptr(pres)))
        out = {name: float(val) for name, val in zip(TERMS, t)}
        out["virial"] = vir.reshape(3, 3)
        out["pressure_tensor"] = pres.reshape(3, 3)
        return out

    # --- checkpoints
    def save_checkpoint(self, path) -> None:
        _check(lib.mdl_save_checkpoint(self._h, str(path).encode()))

    def load_checkpoint(self, path) -> None:
        _check(lib.mdl_load_checkpoint(self._h, str(path).encode()))
