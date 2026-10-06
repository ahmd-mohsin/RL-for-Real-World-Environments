"""Readers and writers for mdlite output files.

``energy.csv``   header ``step,time,<terms...>,box_x,box_y,box_z``; one row per
                 energy output step (``%lld,%.6f`` then ``%.12e`` per column).
``traj.mdtrj``   32-byte header: 8s magic b"MDTRJ\\0\\1\\0", i32 natoms, i32 flags
                 (bit 0: velocities), f64 dt, i32 nstxout, i32 reserved; then per frame
                 i64 step, f64 time, 3 x f64 box, f32[N,3] positions, [f32[N,3] velocities].
``*.mdcpt``      8s magic b"MDCPT\\0\\2\\0", i64 step, i32 natoms, i32 reserved,
                 3 x f64 box, f64 thermostat integral, 6 x f64 constraint virial
                 (xx, yy, zz, xy, xz, yz), u64 seed, f64 dt, f64[N,3] positions,
                 f64[N,3] velocities, u32 crc32 of everything before it.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

TRJ_MAGIC = b"MDTRJ\x00\x01\x00"
CPT_MAGIC = b"MDCPT\x00\x02\x00"


def read_energy(path) -> dict:
    lines = Path(path).read_text().strip().splitlines()
    names = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:]]
    out = {}
    for k, name in enumerate(names):
        col = [r[k] for r in rows]
        out[name] = np.array(col, dtype=np.int64) if name == "step" else np.array(col, dtype=np.float64)
    return out


def read_trajectory(path) -> dict:
    data = Path(path).read_bytes()
    if data[:8] != TRJ_MAGIC:
        raise ValueError("mdtrj: bad magic")
    natoms, flags = struct.unpack_from("<ii", data, 8)
    (dt,) = struct.unpack_from("<d", data, 16)
    nstxout, _ = struct.unpack_from("<ii", data, 24)
    has_v = bool(flags & 1)
    fsize = 8 + 8 + 24 + 12 * natoms * (2 if has_v else 1)
    body = data[32:]
    if len(body) % fsize:
        raise ValueError("mdtrj: truncated frame")
    nf = len(body) // fsize
    steps = np.zeros(nf, np.int64)
    times = np.zeros(nf)
    boxes = np.zeros((nf, 3))
    pos = np.zeros((nf, natoms, 3), np.float32)
    vel = np.zeros((nf, natoms, 3), np.float32) if has_v else None
    for f in range(nf):
        o = f * fsize
        steps[f], times[f] = struct.unpack_from("<qd", body, o)
        boxes[f] = struct.unpack_from("<3d", body, o + 16)
        pos[f] = np.frombuffer(body, "<f4", 3 * natoms, o + 40).reshape(natoms, 3)
        if has_v:
            vel[f] = np.frombuffer(body, "<f4", 3 * natoms, o + 40 + 12 * natoms).reshape(natoms, 3)
    return {"natoms": natoms, "dt": dt, "nstxout": nstxout, "steps": steps, "times": times, "boxes": boxes,
            "positions": pos, "velocities": vel}


def read_checkpoint(path) -> dict:
    data = Path(path).read_bytes()
    if data[:8] != CPT_MAGIC:
        raise ValueError("mdcpt: bad magic")
    if zlib.crc32(data[:-4]) & 0xFFFFFFFF != struct.unpack("<I", data[-4:])[0]:
        raise ValueError("mdcpt: checksum mismatch")
    step, natoms, _ = struct.unpack_from("<qii", data, 8)
    o = 24
    box = np.array(struct.unpack_from("<3d", data, o)); o += 24
    (integral,) = struct.unpack_from("<d", data, o); o += 8
    cvir = np.array(struct.unpack_from("<6d", data, o)); o += 48
    seed, dt = struct.unpack_from("<Qd", data, o); o += 16
    x = np.frombuffer(data, "<f8", 3 * natoms, o).reshape(natoms, 3).copy(); o += 24 * natoms
    v = np.frombuffer(data, "<f8", 3 * natoms, o).reshape(natoms, 3).copy(); o += 24 * natoms
    if o != len(data) - 4:
        raise ValueError("mdcpt: size mismatch")
    return {"step": step, "natoms": natoms, "box": box, "integral": integral, "constraint_virial": cvir, "seed": seed,
            "dt": dt, "positions": x, "velocities": v}


def write_checkpoint(path, *, step, box, positions, velocities, integral=0.0, constraint_virial=(0,) * 6, seed=1, dt=0.002):
    x = np.ascontiguousarray(positions, dtype="<f8")
    v = np.ascontiguousarray(velocities, dtype="<f8")
    n = x.shape[0]
    body = b"".join([CPT_MAGIC, struct.pack("<qii", int(step), n, 0), np.asarray(box, "<f8").tobytes(),
                     struct.pack("<d", float(integral)), np.asarray(constraint_virial, "<f8").tobytes(),
                     struct.pack("<Qd", int(seed), float(dt)), x.tobytes(), v.tobytes()])
    Path(path).write_bytes(body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF))
