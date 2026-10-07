"""Workloads and comparisons for task04_pcl_gpu_cuda12.

References are produced at grading time by the original (CUDA 11.8, compute_86 PTX JIT) build in
/opt/legacy_rt; the candidate's results are compared against them here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

# kinfu_replay workloads: name -> arguments
KINFU = {
    "kinfu_orbit_clean": ["--mode", "kinfu", "--traj", "orbit", "--frames", "120", "--noise", "-1"],
    "kinfu_orbit_n1": ["--mode", "kinfu", "--traj", "orbit", "--frames", "120", "--noise", "1"],
    "kinfu_orbit_n2_fast": ["--mode", "kinfu", "--traj", "orbit", "--frames", "100", "--noise", "2"],
    "kinfu_static_n3": ["--mode", "kinfu", "--traj", "static", "--frames", "40", "--noise", "3"],
    "kinfu_orbit_icptrunc": ["--mode", "kinfu", "--traj", "orbit", "--frames", "120", "--noise", "4", "--icp-trunc", "2.2"],
    "kinfuls_walk_clean": ["--mode", "kinfuls", "--frames", "240", "--noise", "-1"],
    "kinfuls_walk_n4": ["--mode", "kinfuls", "--frames", "240", "--noise", "4"],
    "kinfuls_room_orbit": ["--mode", "kinfuls", "--scene", "room", "--traj", "orbit", "--frames", "120", "--noise", "1"],
    "kinfu_orbit_n5_slow": ["--mode", "kinfu", "--traj", "orbit", "--frames", "150", "--noise", "5"],
    "kinfu_orbit_n8": ["--mode", "kinfu", "--traj", "orbit", "--frames", "110", "--noise", "8"],
    "kinfu_orbit_n34_fast": ["--mode", "kinfu", "--traj", "orbit", "--frames", "105", "--noise", "34"],
    "kinfu_orbit_n13_icp18": ["--mode", "kinfu", "--traj", "orbit", "--frames", "120", "--noise", "13", "--icp-trunc", "1.8"],
    "kinfu_static_clean": ["--mode", "kinfu", "--traj", "static", "--frames", "30", "--noise", "-1"],
    "kinfu_static_n21": ["--mode", "kinfu", "--traj", "static", "--frames", "60", "--noise", "21"],
    "kinfuls_walk_n2_slow": ["--mode", "kinfuls", "--frames", "300", "--noise", "2"],
    "kinfuls_walk_n6_fast": ["--mode", "kinfuls", "--frames", "220", "--noise", "6"],
    "kinfuls_walk_n11_shift05": ["--mode", "kinfuls", "--frames", "240", "--noise", "11", "--shift", "0.5"],
    "kinfuls_walk_n3_shift12": ["--mode", "kinfuls", "--frames", "240", "--noise", "3", "--shift", "1.2"],
    "kinfuls_room_orbit_clean": ["--mode", "kinfuls", "--scene", "room", "--traj", "orbit", "--frames", "120", "--noise", "-1"],
    "kinfuls_room_static_n5": ["--mode", "kinfuls", "--scene", "room", "--traj", "static", "--frames", "40", "--noise", "5"],
}

PEOPLE_VARIANTS = ["base", "mirror", "farther", "noise", "holes", "jitter-11", "jitter-2024", "jitter-90001", "jitter-7",
                   "jitter-13", "jitter-101", "jitter-4242", "jitter-65537", "jitter-777777", "jitter-999983", "jitter-31415"]
NCV_SEEDS = [0, 5, 31337, 2, 3, 17, 255, 1024, 65521, 99991, 1000003, 2147483629]

# concurrent runs: (name, global args, spec); every case must equal the candidate's own solo run bitwise
CONCURRENT = [
# (--extract-every: every tracker also extracts meshes and clouds while the others run; the stress cases
# run several trackers in lockstep over a short prefix and extract on every frame)
    ("conc_kinfu", ["--frames", "120", "--extract-every", "10"], "kinfu:orbit:1,kinfu:orbit:-1,kinfu:orbit:3,kinfu:orbit:6"),
    ("conc_kinfuls", ["--frames", "240", "--extract-every", "20"], "kinfuls:walk:-1,kinfuls:walk:4,kinfuls:walk:9"),
    ("conc_mixed", ["--frames", "120", "--scene", "room", "--extract-every", "10"],
     "kinfu:orbit:1,kinfuls:orbit:1,kinfu:orbit:7,kinfuls:orbit:-1,kinfu:static:2"),
    ("stress_kinfu", ["--frames", "120", "--stop", "24", "--extract-every", "1"],
     "kinfu:orbit:1,kinfu:orbit:2,kinfu:orbit:3,kinfu:orbit:4,kinfu:orbit:5,kinfu:orbit:6"),
    ("stress_kinfuls", ["--frames", "240", "--stop", "24", "--extract-every", "1"],
     "kinfuls:walk:1,kinfuls:walk:2,kinfuls:walk:3,kinfuls:walk:4,kinfuls:walk:5"),
]
PEOPLE_THREADS = 6

# compute-sanitizer workloads (candidate only)
SANITIZE = [
    ("kinfu", "kinfu_replay", ["--mode", "kinfu", "--frames", "120", "--stop", "6"]),
    ("kinfuls", "kinfu_replay", ["--mode", "kinfuls", "--frames", "240", "--stop", "10", "--shift", "0.1"]),
    ("people", "people_eval", ["--variants", "base,holes"]),
    ("ncv", "ncv_check", []),
]

TIMING_KEYS = {"mc_ms", "cloud_ms", "frame_ms_median", "frame_ms_total", "ms_median"}


def fresh_workloads(seed):
    """Workloads drawn from the secret grading seed: name -> (tool, args)."""
    import random
    rng = random.Random(seed)
    w = {}
    for i in range(3):
        w[f"kinfu{i}"] = ("kinfu_replay", ["--mode", "kinfu", "--traj", "orbit", "--frames", str(rng.randint(108, 140)),
                                            "--noise", str(rng.randint(100, 10**6))])
    for i in range(2):
        w[f"kinfuls{i}"] = ("kinfu_replay", ["--mode", "kinfuls", "--frames", str(rng.randint(225, 290)),
                                              "--noise", str(rng.randint(100, 10**6))])
    w["people"] = ("people_eval", [f"jitter-{rng.randint(1, 10**9)}" for _ in range(6)])
    for i in range(4):
        w[f"ncv{i}"] = ("ncv_check", ["--seed", str(rng.randint(1, 2**31 - 2))])
    return w


def load(d):
    return json.loads((Path(d) / "result.json").read_text())


def strip_timing(x):
    if isinstance(x, dict):
        return {k: strip_timing(v) for k, v in x.items() if k not in TIMING_KEYS}
    if isinstance(x, list):
        return [strip_timing(v) for v in x]
    return x


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def rel_poses(poses):
    m = [np.array(p, dtype=np.float64).reshape(4, 4) for p in poses]
    i0 = np.linalg.inv(m[0])
    return [i0 @ x for x in m]


def read_slices(d):
    raw = (Path(d) / "tsdf_slices.bin").read_bytes()
    res = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    return res, np.frombuffer(raw[4:], dtype=np.float32)


def compare_kinfu(cand, ref):
    """Candidate vs original on one kinfu_replay workload; returns a list of failures."""
    c, r = load(cand), load(ref)
    f = []

    def need(ok, what, cv, rv):
        if not ok:
            f.append({"check": what, "candidate": cv, "original": rv})

    ls = r["mode"] == "kinfuls"
    need(c.get("mode") == r["mode"] and c.get("frames") == r["frames"] and len(c.get("poses", [])) == len(r["poses"]),
         "run shape", [c.get("mode"), c.get("frames"), len(c.get("poses", []))], [r["mode"], r["frames"], len(r["poses"])])
    if f:
        return f
    # tracking vs ground truth: as good as the original
    need(c["trans_err_mean"] <= r["trans_err_mean"] * 1.15 + 3e-4, "trans_err_mean", c["trans_err_mean"], r["trans_err_mean"])
    need(c["trans_err_max"] <= r["trans_err_max"] * 1.25 + 6e-4, "trans_err_max", c["trans_err_max"], r["trans_err_max"])
    need(c["rot_err_max_deg"] <= r["rot_err_max_deg"] * 1.25 + 0.05, "rot_err_max_deg", c["rot_err_max_deg"], r["rot_err_max_deg"])
    # per-frame poses vs the original's
    pc, pr = rel_poses(c["poses"]), rel_poses(r["poses"])
    dt = max(float(np.linalg.norm(a[:3, 3] - b[:3, 3])) for a, b in zip(pc, pr))
    dr = max(float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))) for a, b in zip(pc, pr))
    need(dt <= POSE_TOL_M, "pose vs original (m)", dt, POSE_TOL_M)
    need(dr <= POSE_TOL_DEG, "pose vs original (deg)", dr, POSE_TOL_DEG)
    # TSDF statistics and slices
    tc, tr = c["tsdf"], r["tsdf"]
    need(tc["resolution"] == tr["resolution"], "tsdf resolution", tc["resolution"], tr["resolution"])
    for k, tol in (("observed", 0.003), ("near_surface", 0.006), ("weight_sum", 0.006), ("tsdf_abs_mean", 0.006)):
        need(rel(tc[k], tr[k]) <= tol, f"tsdf {k}", tc[k], tr[k])
    (rc_, sc), (rr_, sr) = read_slices(cand), read_slices(ref)
    if rc_ != rr_ or sc.shape != sr.shape:
        f.append({"check": "tsdf slices shape", "candidate": [rc_, sc.size], "original": [rr_, sr.size]})
    else:
        oc, orr = sc != 2.0, sr != 2.0
        agree = float(np.mean(oc == orr))
        both = oc & orr
        d = np.abs(sc[both] - sr[both])
        need(agree >= 0.995, "tsdf slices observed-mask agreement", agree, 0.995)
        need(float(d.mean()) <= 0.004, "tsdf slices mean |diff|", float(d.mean()), 0.004)
        need(float(np.mean(d > 0.1)) <= 0.004, "tsdf slices fraction |diff| > 0.1", float(np.mean(d > 0.1)), 0.004)
    if "mesh" in r:
        mc, mr = c.get("mesh"), r["mesh"]
        if mc is None:
            f.append({"check": "mesh missing", "candidate": None, "original": mr})
        else:
            need(rel(mc["triangles"], mr["triangles"]) <= 0.004, "mesh triangles", mc["triangles"], mr["triangles"])
            need(rel(mc["area"], mr["area"]) <= 0.004, "mesh area", mc["area"], mr["area"])
            need(mc["surface_err_mean"] <= mr["surface_err_mean"] * 1.10 + 5e-5, "mesh surface_err_mean", mc["surface_err_mean"],
                 mr["surface_err_mean"])
            # extremal: a single stray triangle moves it by about a centimetre
            bb = max(abs(a - b) for a, b in zip(mc["bbox"], mr["bbox"]))
            need(bb <= 0.05, "mesh bbox", mc["bbox"], mr["bbox"])
    if ls:
        need(c.get("shifts_origin") == r["shifts_origin"], "volume shifts", c.get("shifts_origin"), r["shifts_origin"])
    else:
        cc, cr = c["cloud"], r["cloud"]
        need(rel(cc["points"], cr["points"]) <= 0.004, "cloud points", cc["points"], cr["points"])
        cd = max(abs(a - b) for a, b in zip(cc["centroid"], cr["centroid"]))
        need(cd <= 0.005, "cloud centroid", cc["centroid"], cr["centroid"])
        need(cc["surface_err_mean"] <= cr["surface_err_mean"] * 1.10 + 5e-5, "cloud surface_err_mean", cc["surface_err_mean"],
             cr["surface_err_mean"])
        need(c["raycast_depth_err_mm"] <= r["raycast_depth_err_mm"] * 1.15 + 0.3, "raycast_depth_err_mm", c["raycast_depth_err_mm"],
             r["raycast_depth_err_mm"])
        need(abs(c["raycast_valid_frac"] - r["raycast_valid_frac"]) <= 0.004, "raycast_valid_frac", c["raycast_valid_frac"],
             r["raycast_valid_frac"])
    return f


POSE_TOL_M = 0.002
POSE_TOL_DEG = 0.15


def compare_bitwise_run(a, b):
    """Two kinfu_replay outputs (same workload) must be bitwise identical apart from timing."""
    f = []
    if strip_timing(load(a)) != strip_timing(load(b)):
        f.append("result.json differs")
    if (Path(a) / "tsdf_slices.bin").read_bytes() != (Path(b) / "tsdf_slices.bin").read_bytes():
        f.append("tsdf_slices.bin differs")
    return f


def compare_people(cand, ref, variants):
    f = []
    c, r = load(cand), load(ref)
    for v in variants:
        if v not in c:
            f.append({"variant": v, "error": "missing"})
            continue
        if c[v]["return"] != r[v]["return"]:
            f.append({"variant": v, "return": [c[v]["return"], r[v]["return"]]})
        lc, lr = Path(cand) / f"{v}.labels", Path(ref) / f"{v}.labels"
        a = np.frombuffer(lc.read_bytes(), np.uint8) if lc.exists() else np.zeros(0, np.uint8)
        b = np.frombuffer(lr.read_bytes(), np.uint8)
        if a.shape != b.shape:
            f.append({"variant": v, "labels": f"size {a.size} vs {b.size}"})
        elif not np.array_equal(a, b):
            f.append({"variant": v, "labels": f"{int(np.sum(a != b))} of {b.size} pixels differ"})
    return f


def compare_ncv(cand, ref):
    f = []
    c, r = load(cand), load(ref)
    for name, rv in r.items():
        cv = c.get(name)
        if cv is None or cv["status"] != rv["status"]:
            f.append({"case": name, "status": [cv and cv["status"], rv["status"]]})
            continue
        bc, br = Path(cand) / f"{name}.bin", Path(ref) / f"{name}.bin"
        if br.exists() and (not bc.exists() or bc.read_bytes() != br.read_bytes()):
            f.append({"case": name, "error": "output buffer differs" if bc.exists() else "output buffer missing"})
    return f


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
