#!/usr/bin/env python3
"""Kernel family, the GPU half of the gates (swarm-pipeline/06 §6), run on the GPU host from a bundle that
``lh_kernel.py stage-gpu`` wrote. Standard library only (copied into the bundle as ``stage_gpu.py``).

    python3 stage_gpu.py --arch 120 [--mode docker|apptainer] [--flake 60] [--only WID ...]

Per window: build the original with CUDA 11.8 as PTX for driver JIT (root-only ``legacy/``), then grade with the kit plan
in the verifier image, as root, the candidate unprivileged: the oracle (S5: reward 1 with every milestone), the floors
(S8: 0), each fault-class mutant (S8: its milestone fails), and ``--flake`` oracle grades half at concurrency 1 and half
at 4 (S10: Clopper-Pearson upper bound <= 5%). Results: ``results/<wid>.json``.

Shared-machine rules (lab): containers are named ``crucible-k-gpu-*`` and only those are removed; nothing is pruned; each
grade waits for an idle GPU first. Docker needs ``--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all`` on the lab.
Apptainer (Marlowe): the same OCI images, ``--nv``; the grader needs root inside the container to drop privileges, so
``--fakeroot`` is required, else grades are reported ``trusted: false``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "kit"))
from crucible.factory.lanes import kernel_plan as kp

BASE_TAG, LEGACY_TAG = "crucible-k-base:12.8.1", "crucible-k-legacy:11.8.0"


def sh(cmd: list[str], *, timeout: float = 7200, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=input, check=False, errors="replace")


def idle(max_util: int = 10, max_load: float = 8.0, wait_s: float = 3600) -> dict:
    t0 = time.time()
    while True:
        u = sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], timeout=30).stdout.split(",")
        load = os.getloadavg()[0]
        util = int(u[0]) if u and u[0].strip().isdigit() else 0
        if util <= max_util and load < max_load:
            return {"gpu_util": util, "load1": round(load, 2), "waited_s": round(time.time() - t0)}
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"GPU never idle: util {util}% load {load}")
        time.sleep(60)


def ensure_images() -> None:
    for tag, df in ((BASE_TAG, HERE / "Dockerfile.base"), (LEGACY_TAG, HERE / "Dockerfile.legacy")):
        if sh(["docker", "image", "inspect", tag], timeout=60).returncode != 0:
            p = sh(["docker", "build", "-t", tag, "-"], input=df.read_text(), timeout=7200)
            if p.returncode != 0:
                raise RuntimeError(f"build {tag}: {p.stderr[-800:]}")


def build_launch_lib() -> None:
    lib = HERE / "hooks" / "libcrucible_launch.so"
    if lib.exists():
        return
    cmd = " ".join(kp.LAUNCH_BUILD)
    p = sh(["docker", "run", "--rm", "--name", f"crucible-k-gpu-cupti-{uuid.uuid4().hex[:8]}", "-v", f"{HERE / 'hooks'}:/h", "-w", "/h", BASE_TAG,
            "bash", "-c", cmd])
    if p.returncode != 0:
        raise RuntimeError(f"CUPTI injection build failed: {p.stderr[-600:]}")


def build_originals(wdir: Path, spec: dict, legacy: dict) -> dict:
    out = wdir / "legacy"
    out.mkdir(exist_ok=True)
    lines = ["set +e"]
    for pid, prog in spec["programs"].items():
        la = (legacy.get(pid) or {}).get("legacy_arch") or "60"
        p = {**prog, "defines": [*(prog.get("defines") or []), *((legacy.get(pid) or {}).get("legacy_defines") or [])]}
        cmd = kp.nvcc_command(p, root=Path("/t"), out=Path(f"/o/{pid}"), arch="0", observe=Path("/h/crucible_observe.cpp"), legacy=True,
                              legacy_arch=la, nvcc="nvcc")
        lines.append(f"( cd /t && {' '.join(map(_q, cmd))} ) > /o/{pid}.log 2>&1; echo $? > /o/{pid}.rc")
    (wdir / "legacy_build.sh").write_text("\n".join(lines) + "\n")
    p = sh(["docker", "run", "--rm", "--name", f"crucible-k-gpu-legacy-{uuid.uuid4().hex[:8]}", "--network", "none", "-v", f"{wdir / 'base'}:/t:ro",
            "-v", f"{HERE / 'hooks'}:/h:ro", "-v", f"{out}:/o", "-v", f"{wdir / 'legacy_build.sh'}:/b.sh:ro", LEGACY_TAG, "bash", "/b.sh"])
    res = {pid: (out / f"{pid}.rc").read_text().strip() == "0" if (out / f"{pid}.rc").exists() else False for pid in spec["programs"]}
    if p.returncode != 0:
        res["_container"] = p.stderr[-400:]
    return res


def _q(s: str) -> str:
    import shlex

    return shlex.quote(str(s))


def grade(wdir: Path, patch: Path, arch: str, mode: str, tag: str) -> dict:
    out = wdir / "grades" / tag
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    args = ["python3", "-I", "/crucible_verify/crucible/verify/runtime/run.py", "--plan", "crucible.factory.lanes.kernel_plan", "--out", "/out",
            "--base", "/k/base", "--patch", "/k/patch", "--param", "spec=/k/kspec.json", "--param", f"arch={arch}", "--param", "base_tree=/k/base",
            "--param", "legacy_dir=/k/legacy", "--param", "hooks_dir=/k/hooks", "--require-root"]
    mounts = [(HERE / "kit", "/crucible_verify"), (wdir / "base", "/k/base"), (wdir / "legacy", "/k/legacy"), (HERE / "hooks", "/k/hooks"),
              (wdir / "kspec.json", "/k/kspec.json"), (patch, "/k/patch"), (out, "/out")]
    if mode == "apptainer":
        cmd = ["apptainer", "exec", "--nv", "--fakeroot", "--containall", "--net", "--network", "none"]
        for src, dst in mounts:
            cmd += ["-B", f"{src}:{dst}" + ("" if dst == "/out" else ":ro")]
        cmd += [f"docker-daemon://{BASE_TAG}", *args]
    else:
        cmd = ["docker", "run", "--rm", "--name", f"crucible-k-gpu-{tag}-{uuid.uuid4().hex[:8]}", "--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all",
               "-e", f"CRUCIBLE_CUDA_ARCH={arch}", "--network", "none", "--init", "--shm-size", "2g"]
        for src, dst in mounts:
            cmd += ["-v", f"{src}:{dst}" + ("" if dst == "/out" else ":ro")]
        cmd += [BASE_TAG, *args]
    p = sh(cmd, timeout=7200)
    rj = out / "crucible_reward.json"
    doc = json.loads(rj.read_text()) if rj.exists() else {"infra": {"status": "masked", "failure_kind": "no_reward", "reason": p.stderr[-400:]}}
    return {"reward": doc.get("reward"), "gates": doc.get("gates"), "infra": doc.get("infra")}


def milestones_of(doc: dict) -> dict:
    return {g: int(v) for g, v in (doc.get("gates") or {}).items()}


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound (bisection on the binomial tail)."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        tail = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if tail > alpha else (lo, mid)
    return hi


def window(wid: str, a: argparse.Namespace) -> dict:
    wdir = HERE / "windows" / wid
    if not (wdir / "base").exists():
        with tarfile.open(wdir / "base.tar.gz") as t:
            t.extractall(wdir, filter="data")
    spec = json.loads((wdir / "kspec.json").read_text())
    res = {"id": wid, "arch": a.arch, "gpu": sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).stdout.strip(), "at": time.strftime("%FT%TZ", time.gmtime())}
    res["originals"] = build_originals(wdir, spec, json.loads((wdir / "legacy.json").read_text()))
    os.chmod(wdir / "legacy", 0o700)
    res["idle"] = idle()
    res["oracle"] = grade(wdir, wdir / "oracle.patch", a.arch, a.mode, "oracle")
    res["floors"] = {p.stem: grade(wdir, p, a.arch, a.mode, p.stem) for p in sorted(wdir.glob("floor_*.patch"))}
    res["mutants"] = {p.stem: grade(wdir, p, a.arch, a.mode, p.stem) for p in sorted(wdir.glob("mutant_*.patch"))}
    if a.flake:
        half = a.flake // 2
        runs = [grade(wdir, wdir / "oracle.patch", a.arch, a.mode, f"flake1-{i}") for i in range(half)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            runs += list(ex.map(lambda i: grade(wdir, wdir / "oracle.patch", a.arch, a.mode, f"flake4-{i}"), range(a.flake - half)))
        bad = sum(1 for r in runs if r["reward"] != 1.0)
        res["flake"] = {"n": len(runs), "false_negatives": bad, "masked": sum(1 for r in runs if (r["infra"] or {}).get("status") == "masked"),
                        "cp_upper": round(cp_upper(bad, len(runs)), 4)}
    res["S5_oracle"] = res["oracle"]["reward"] == 1.0
    res["S8_floors"] = all(f["reward"] == 0.0 for f in res["floors"].values())
    res["S8_mutants_killed"] = {k: any(v == 0 for v in milestones_of(m).values()) for k, m in res["mutants"].items()}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{wid}.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--mode", default="docker", choices=("docker", "apptainer"))
    ap.add_argument("--flake", type=int, default=60)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    ensure_images()
    build_launch_lib()
    for w in json.loads((HERE / "index.json").read_text()):
        if a.only and w["id"] not in a.only:
            continue
        if (HERE / "results" / f"{w['id']}.json").exists():
            continue
        r = window(w["id"], a)
        print(w["id"], "oracle", r["oracle"]["reward"], "floors", r["S8_floors"], "mutants", r["S8_mutants_killed"], "flake", r.get("flake"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
