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
    # colour volumes, normals, rendered views, ray-cast maps, last-frame buffers, host-side extraction
    "kinfu_orbit_color_api": ["--mode", "kinfu", "--traj", "orbit", "--frames", "120", "--noise", "1", "--color", "--api"],
    "kinfu_orbit_n8_api": ["--mode", "kinfu", "--traj", "orbit", "--frames", "110", "--noise", "8", "--api"],
    "kinfu_static_color_api": ["--mode", "kinfu", "--traj", "static", "--frames", "40", "--noise", "17", "--color", "--api"],
    "kinfuls_walk_color_api": ["--mode", "kinfuls", "--frames", "240", "--noise", "2", "--color", "--api"],
    "kinfuls_room_color_api": ["--mode", "kinfuls", "--scene", "room", "--traj", "orbit", "--frames", "120", "--noise", "3", "--color", "--api"],
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
    ("kinfu", "kinfu_replay", ["--mode", "kinfu", "--frames", "120", "--stop", "6", "--color", "--api"]),
    ("kinfuls", "kinfu_replay", ["--mode", "kinfuls", "--frames", "240", "--stop", "10", "--shift", "0.1", "--color", "--api"]),
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
                                            "--noise", str(rng.randint(100, 10**6))] + (["--color", "--api"] if i == 0 else []))
    for i in range(2):
        w[f"kinfuls{i}"] = ("kinfu_replay", ["--mode", "kinfuls", "--frames", str(rng.randint(225, 290)),
                                              "--noise", str(rng.randint(100, 10**6))] + (["--color", "--api"] if i == 0 else []))
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
    # KinFu LS: volume shifts are threshold decisions. On about 5% of trajectories, float-level differences between two
    # correct builds make a shift fire a frame earlier or a voxel off, and the runs then bifurcate (about 1 cm apart; 40
    # random draws measured, plus one seen in a model run). A diverged KinFu LS run is held to looser, still tight
    # bounds; shift origins further apart than the induced bifurcations always fail. KinFu itself never diverged beyond 0.03 mm.
    pc, pr = rel_poses(c["poses"]), rel_poses(r["poses"])
    dt = max(float(np.linalg.norm(a[:3, 3] - b[:3, 3])) for a, b in zip(pc, pr))
    dr = max(float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))) for a, b in zip(pc, pr))
    T = STRICT
    if ls:
        so_c, so_r = c.get("shifts_origin") or [], r["shifts_origin"]
        if so_c != so_r or dt > STRICT["pose_m"] or dr > STRICT["pose_deg"]:
            T = BIFURCATED
        if len(so_c) != 3 or any(abs(x - y) > T["shift_vox"] for x, y in zip(so_c, so_r)):
            f.append({"check": "volume shifts", "candidate": so_c, "original": so_r})
            return f
    # tracking vs ground truth: as good as the original
    need(c["trans_err_mean"] <= r["trans_err_mean"] * T["te_mean"][0] + T["te_mean"][1], "trans_err_mean", c["trans_err_mean"], r["trans_err_mean"])
    need(c["trans_err_max"] <= r["trans_err_max"] * T["te_max"][0] + T["te_max"][1], "trans_err_max", c["trans_err_max"], r["trans_err_max"])
    need(c["rot_err_max_deg"] <= r["rot_err_max_deg"] * T["rot"][0] + T["rot"][1], "rot_err_max_deg", c["rot_err_max_deg"], r["rot_err_max_deg"])
    # per-frame poses vs the original's
    need(dt <= T["pose_m"], "pose vs original (m)", dt, T["pose_m"])
    need(dr <= T["pose_deg"], "pose vs original (deg)", dr, T["pose_deg"])
    # TSDF statistics and slices
    tc, tr = c["tsdf"], r["tsdf"]
    need(tc["resolution"] == tr["resolution"], "tsdf resolution", tc["resolution"], tr["resolution"])
    for k in ("observed", "near_surface", "weight_sum", "tsdf_abs_mean"):
        need(rel(tc[k], tr[k]) <= T["tsdf_" + k], f"tsdf {k}", tc[k], tr[k])
    (rc_, sc), (rr_, sr) = read_slices(cand), read_slices(ref)
    if rc_ != rr_ or sc.shape != sr.shape:
        f.append({"check": "tsdf slices shape", "candidate": [rc_, sc.size], "original": [rr_, sr.size]})
    else:
        oc, orr = sc != 2.0, sr != 2.0
        agree = float(np.mean(oc == orr))
        both = oc & orr
        d = np.abs(sc[both] - sr[both])
        need(agree >= T["mask_agree"], "tsdf slices observed-mask agreement", agree, T["mask_agree"])
        need(float(d.mean()) <= T["slice_mean"], "tsdf slices mean |diff|", float(d.mean()), T["slice_mean"])
        need(float(np.mean(d > 0.1)) <= T["slice_gt01"], "tsdf slices fraction |diff| > 0.1", float(np.mean(d > 0.1)), T["slice_gt01"])
    if "mesh" in r:
        mc, mr = c.get("mesh"), r["mesh"]
        if mc is None:
            f.append({"check": "mesh missing", "candidate": None, "original": mr})
        else:
            need(rel(mc["triangles"], mr["triangles"]) <= T["mesh"], "mesh triangles", mc["triangles"], mr["triangles"])
            need(rel(mc["area"], mr["area"]) <= T["mesh"], "mesh area", mc["area"], mr["area"])
            need(mc["surface_err_mean"] <= mr["surface_err_mean"] * T["serr"] + 5e-5, "mesh surface_err_mean", mc["surface_err_mean"],
                 mr["surface_err_mean"])
            # extremal: a single stray triangle moves it by about a centimetre
            bb = max(abs(a - b) for a, b in zip(mc["bbox"], mr["bbox"]))
            need(bb <= T["bbox"], "mesh bbox", mc["bbox"], mr["bbox"])
    if not ls:
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
    if "api" in r:
        f += compare_api(cand, ref, c.get("api"), r["api"], T)
    return f


def read_view(path):
    raw = Path(path).read_bytes()
    rows, cols = (int(x) for x in np.frombuffer(raw[:8], np.int32))
    return np.frombuffer(raw[8:], np.uint8).reshape(rows, cols, 3).astype(np.int16)


def compare_api(cand, ref, ca, ra, T):
    """--api: normals, colours, rendered views, ray-cast maps, last frame, host-side extraction."""
    f = []

    def need(ok, what, cv, rv):
        if not ok:
            f.append({"check": "api " + what, "candidate": cv, "original": rv})

    if not isinstance(ca, dict):
        return [{"check": "api missing", "candidate": ca, "original": "present"}]
    for k in ("cloud_normals", "raycast_maps", "last_frame"):
        cv, rv = ca.get(k), ra[k]
        if not isinstance(cv, dict):
            f.append({"check": f"api {k} missing"})
            continue
        need(rel(cv["valid"], rv["valid"]) <= T["api_count"], f"{k} valid", cv["valid"], rv["valid"])
        dn = max(abs(a - b) for a, b in zip(cv["abs_normal_mean"], rv["abs_normal_mean"]))
        need(dn <= T["api_absn"], f"{k} normal distribution", cv["abs_normal_mean"], rv["abs_normal_mean"])
        if "sdf_mean" in rv:
            need(cv.get("sdf_mean", 1e9) <= rv["sdf_mean"] * T["api_sdf"][0] + T["api_sdf"][1], f"{k} sdf_mean", cv.get("sdf_mean"), rv["sdf_mean"])
            need(cv.get("angle_deg_mean", 1e9) <= rv["angle_deg_mean"] * T["api_angle"][0] + T["api_angle"][1], f"{k} normal angle",
                 cv.get("angle_deg_mean"), rv["angle_deg_mean"])
    if "cloud_colors" in ra:
        cv, rv = ca.get("cloud_colors"), ra["cloud_colors"]
        if not isinstance(cv, dict):
            f.append({"check": "api cloud_colors missing"})
        else:
            need(rel(cv["points"], rv["points"]) <= T["api_count"], "cloud_colors points", cv["points"], rv["points"])
            dm = max(abs(a - b) for a, b in zip(cv["mean"], rv["mean"]))
            need(dm <= T["api_cmean"], "cloud_colors mean", cv["mean"], rv["mean"])
            if "abs_err_mean" in rv:
                need(cv.get("abs_err_mean", 1e9) <= rv["abs_err_mean"] * T["api_color"][0] + T["api_color"][1], "cloud colour error",
                     cv.get("abs_err_mean"), rv["abs_err_mean"])
    for v in ("view", "scene_view"):
        try:
            a, b = read_view(Path(cand) / f"{v}.bin"), read_view(Path(ref) / f"{v}.bin")
        except Exception as ex:
            f.append({"check": f"api {v}", "error": repr(ex)})
            continue
        if a.shape != b.shape:
            f.append({"check": f"api {v} shape", "candidate": list(a.shape), "original": list(b.shape)})
            continue
        frac = float((np.abs(a - b).max(axis=2) > 12).mean())
        need(frac <= T["api_view"], f"{v} pixels differing by > 12", frac, T["api_view"])
    if "host_cloud_points" in ra:
        need(rel(ca.get("host_cloud_points", 0), ra["host_cloud_points"]) <= T["api_count"], "host cloud points",
             ca.get("host_cloud_points"), ra["host_cloud_points"])
        cv, rv = ca.get("tsdf_download") or {}, ra["tsdf_download"]
        need(rel(cv.get("weighted", 0), rv["weighted"]) <= T["tsdf_observed"], "tsdf download weighted", cv.get("weighted"), rv["weighted"])
        need(rel(cv.get("abs_sum", 0), rv["abs_sum"]) <= T["tsdf_weight_sum"], "tsdf download abs sum", cv.get("abs_sum"), rv["abs_sum"])
    return f


# tolerances (scale, offset) or limits, calibrated on random fresh draws of the original vs the reference migration:
# - KinFu, 40 draws: pose differences <= 0.03 mm.
# - KinFu LS, 40 draws: <= 0.33 mm, except natural bifurcations (5-11 mm, 4 seen).
# - BIFURCATED covers, with margin, 40 induced bifurcations (shift distance moved by 1 cm): pose <= 2.0 cm, tracking
#   error <= 1.4x (max 1.7x), TSDF counts <= 10%, triangles <= 5.5%, surface error <= 1.3x.
STRICT = {"te_mean": (1.15, 3e-4), "te_max": (1.25, 6e-4), "rot": (1.25, 0.05), "pose_m": 0.002, "pose_deg": 0.15,
          "tsdf_observed": 0.003, "tsdf_near_surface": 0.006, "tsdf_weight_sum": 0.006, "tsdf_tsdf_abs_mean": 0.006,
          "mask_agree": 0.995, "slice_mean": 0.004, "slice_gt01": 0.004, "mesh": 0.004, "serr": 1.10, "shift_vox": 0, "bbox": 0.05,
          "api_count": 0.004, "api_absn": 0.003, "api_sdf": (1.1, 5e-5), "api_angle": (1.05, 0.2), "api_color": (1.1, 0.3),
          "api_cmean": 0.5, "api_view": 0.01}
BIFURCATED = {"te_mean": (1.6, 1e-3), "te_max": (2.0, 3e-3), "rot": (1.5, 0.25), "pose_m": 0.03, "pose_deg": 1.0,
              "tsdf_observed": 0.06, "tsdf_near_surface": 0.12, "tsdf_weight_sum": 0.14, "tsdf_tsdf_abs_mean": 0.01,
              "mask_agree": 0.96, "slice_mean": 0.015, "slice_gt01": 0.04, "mesh": 0.07, "serr": 1.4, "shift_vox": 40, "bbox": 0.25,  # 40 voxels of 5.9 mm
              "api_count": 0.07, "api_absn": 0.02, "api_sdf": (1.4, 5e-4), "api_angle": (1.3, 1.0), "api_color": (1.4, 2.0),
              "api_cmean": 3.0, "api_view": 0.08}


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
