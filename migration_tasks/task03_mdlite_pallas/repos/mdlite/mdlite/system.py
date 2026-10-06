"""Molecular systems and the binary ``.mdsys`` format.

Layout (little endian), followed by a CRC-32 of all preceding bytes::

    8s      magic b"MDSYS\\0\\1\\0"
    9 x i32 natoms, ntypes, nnbfix, nbonds, nangles, ndihedrals, npairs, nconstraints, nexclusions
    i32     nrexcl
    i32     has_velocities
    3 x f64 box
    f64     fudge_lj
    f64     fudge_qq
    ntypes  x (8s name, f64 sigma, f64 epsilon)
    nnbfix  x (i32 ti, i32 tj, f64 sigma, f64 epsilon)
    i32[N] type, i32[N] molecule, f64[N] charge, f64[N] mass, f64[N,3] positions, [f64[N,3] velocities]
    nbonds      x (i32 i, i32 j, f64 r0, f64 k)
    nangles     x (i32 i, i32 j, i32 k, f64 theta0_deg, f64 k)
    ndihedrals  x (i32 i, i32 j, i32 k, i32 l, i32 mult, f64 phi0_deg, f64 k)
    npairs      x (i32 i, i32 j)
    nconstraints x (i32 i, i32 j, f64 length)
    nexclusions x (i32 i, i32 j)
    u32     crc32
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"MDSYS\x00\x01\x00"

_BOND = np.dtype([("i", "<i4"), ("j", "<i4"), ("r0", "<f8"), ("k", "<f8")])
_ANGLE = np.dtype([("i", "<i4"), ("j", "<i4"), ("k", "<i4"), ("theta0", "<f8"), ("kth", "<f8")])
_DIH = np.dtype([("i", "<i4"), ("j", "<i4"), ("k", "<i4"), ("l", "<i4"), ("mult", "<i4"), ("phi0", "<f8"), ("kphi", "<f8")])
_PAIR = np.dtype([("i", "<i4"), ("j", "<i4")])
_CONS = np.dtype([("i", "<i4"), ("j", "<i4"), ("d", "<f8")])
_NBFIX = np.dtype([("ti", "<i4"), ("tj", "<i4"), ("sigma", "<f8"), ("epsilon", "<f8")])
_TYPE = np.dtype([("name", "S8"), ("sigma", "<f8"), ("epsilon", "<f8")])


def _empty(dt):
    return np.zeros(0, dtype=dt)


@dataclass
class System:
    """A molecular system: atoms, force-field parameters and topology.

    ``types`` / ``nbfix`` / ``bonds`` / ``angles`` / ``dihedrals`` / ``pairs`` /
    ``constraints`` / ``exclusions`` are numpy structured arrays with the fields
    of the file format (see the module docstring).
    """

    box: np.ndarray
    types: np.ndarray
    type: np.ndarray
    molecule: np.ndarray
    charge: np.ndarray
    mass: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray | None = None
    nbfix: np.ndarray = field(default_factory=lambda: _empty(_NBFIX))
    bonds: np.ndarray = field(default_factory=lambda: _empty(_BOND))
    angles: np.ndarray = field(default_factory=lambda: _empty(_ANGLE))
    dihedrals: np.ndarray = field(default_factory=lambda: _empty(_DIH))
    pairs: np.ndarray = field(default_factory=lambda: _empty(_PAIR))
    constraints: np.ndarray = field(default_factory=lambda: _empty(_CONS))
    exclusions: np.ndarray = field(default_factory=lambda: _empty(_PAIR))
    fudge_lj: float = 1.0
    fudge_qq: float = 1.0
    nrexcl: int = 3

    dtypes = {"types": _TYPE, "nbfix": _NBFIX, "bonds": _BOND, "angles": _ANGLE, "dihedrals": _DIH,
              "pairs": _PAIR, "constraints": _CONS, "exclusions": _PAIR}

    def __post_init__(self):
        self.box = np.asarray(self.box, dtype=np.float64).reshape(3)
        self.type = np.asarray(self.type, dtype=np.int32).reshape(-1)
        n = self.type.size
        self.molecule = np.asarray(self.molecule, dtype=np.int32).reshape(n)
        self.charge = np.asarray(self.charge, dtype=np.float64).reshape(n)
        self.mass = np.asarray(self.mass, dtype=np.float64).reshape(n)
        self.positions = np.asarray(self.positions, dtype=np.float64).reshape(n, 3)
        if self.velocities is not None:
            self.velocities = np.asarray(self.velocities, dtype=np.float64).reshape(n, 3)
        for name, dt in self.dtypes.items():
            arr = getattr(self, name)
            if not (isinstance(arr, np.ndarray) and arr.dtype == dt):
                arr = np.array([tuple(r) for r in arr], dtype=dt) if len(arr) else _empty(dt)
            setattr(self, name, arr)

    @property
    def natoms(self) -> int:
        return int(self.type.size)

    def copy(self) -> "System":
        kw = {k: (getattr(self, k).copy() if isinstance(getattr(self, k), np.ndarray) else getattr(self, k))
              for k in self.__dataclass_fields__}
        return System(**kw)

    # --- I/O
    def to_bytes(self) -> bytes:
        n = self.natoms
        out = [MAGIC, struct.pack("<9i", n, len(self.types), len(self.nbfix), len(self.bonds), len(self.angles),
                                  len(self.dihedrals), len(self.pairs), len(self.constraints), len(self.exclusions)),
               struct.pack("<ii", int(self.nrexcl), 0 if self.velocities is None else 1),
               self.box.astype("<f8").tobytes(), struct.pack("<dd", float(self.fudge_lj), float(self.fudge_qq)),
               self.types.tobytes(), self.nbfix.tobytes(),
               self.type.astype("<i4").tobytes(), self.molecule.astype("<i4").tobytes(),
               self.charge.astype("<f8").tobytes(), self.mass.astype("<f8").tobytes(),
               self.positions.astype("<f8").tobytes()]
        if self.velocities is not None:
            out.append(self.velocities.astype("<f8").tobytes())
        for name in ("bonds", "angles", "dihedrals", "pairs", "constraints", "exclusions"):
            out.append(getattr(self, name).tobytes())
        body = b"".join(out)
        return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)

    def save(self, path) -> None:
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "System":
        if len(data) < 12 or data[:8] != MAGIC:
            raise ValueError("mdsys: bad magic")
        if zlib.crc32(data[:-4]) & 0xFFFFFFFF != struct.unpack("<I", data[-4:])[0]:
            raise ValueError("mdsys: checksum mismatch")
        pos = 8

        def take(fmt):
            nonlocal pos
            vals = struct.unpack_from(fmt, data, pos)
            pos += struct.calcsize(fmt)
            return vals

        def arr(dt, count, shape=None):
            nonlocal pos
            dt = np.dtype(dt)
            a = np.frombuffer(data, dtype=dt, count=count, offset=pos).copy()
            pos += dt.itemsize * count
            return a.reshape(shape) if shape else a

        n, nt, nnb, nb, na, nd, npr, nc, ne = take("<9i")
        nrexcl, has_v = take("<ii")
        box = np.array(take("<3d"))
        fudge_lj, fudge_qq = take("<dd")
        types = arr(_TYPE, nt)
        nbfix = arr(_NBFIX, nnb)
        type_ = arr("<i4", n)
        mol = arr("<i4", n)
        q = arr("<f8", n)
        m = arr("<f8", n)
        x = arr("<f8", 3 * n, (n, 3))
        v = arr("<f8", 3 * n, (n, 3)) if has_v else None
        bonds = arr(_BOND, nb)
        angles = arr(_ANGLE, na)
        dih = arr(_DIH, nd)
        pairs = arr(_PAIR, npr)
        cons = arr(_CONS, nc)
        excl = arr(_PAIR, ne)
        if pos != len(data) - 4:
            raise ValueError("mdsys: trailing bytes")
        return cls(box=box, types=types, type=type_, molecule=mol, charge=q, mass=m, positions=x, velocities=v, nbfix=nbfix,
                   bonds=bonds, angles=angles, dihedrals=dih, pairs=pairs, constraints=cons, exclusions=excl,
                   fudge_lj=fudge_lj, fudge_qq=fudge_qq, nrexcl=nrexcl)


def load_system(path) -> System:
    return System.from_bytes(Path(path).read_bytes())


def save_system(system: System, path) -> None:
    system.save(path)


def make_types(entries):
    """``[(name, sigma, epsilon), ...]`` -> structured types array."""
    return np.array([(n.encode() if isinstance(n, str) else n, s, e) for n, s, e in entries], dtype=_TYPE)
