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
Apptainer: the same images as SIFs (``--build-sif`` builds them from the bundle's Dockerfiles on the host, into
``CRUCIBLE_SIF_DIR``, default ``sif/``), ``--nv``. The grader needs root inside the container to drop privileges, so
``CRUCIBLE_APPTAINER_ROOT=fakeroot`` (default; Marlowe) keeps ``--fakeroot``; ``CRUCIBLE_APPTAINER_ROOT=user`` (DeltaAI: no
subuid range) runs it as the invoking user without ``--require-root`` and grades are reported ``trusted: false``.
``CRUCIBLE_APPTAINER_TMP``: the host directory each container's /tmp lives in. ``CRUCIBLE_IDLE_LOAD``: the host load below
which a timed step may start (default 8).
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
MODE = "docker"  # set by ``main`` (``--mode``)
SIF_DIR = Path(os.environ.get("CRUCIBLE_SIF_DIR") or HERE / "sif")
APPTAINER_ROOT = os.environ.get("CRUCIBLE_APPTAINER_ROOT", "fakeroot")
APPTAINER_TMP = Path(os.environ.get("CRUCIBLE_APPTAINER_TMP") or "/tmp/crucible-apptainer")
IDLE_LOAD = float(os.environ.get("CRUCIBLE_IDLE_LOAD") or 8.0)


def sh(cmd: list[str], *, timeout: float = 7200, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=input, check=False, errors="replace")


def idle(max_util: int = 10, max_load: float | None = None, wait_s: float = 3600) -> dict:
    max_load = IDLE_LOAD if max_load is None else max_load
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


def images_of(wdir: Path) -> tuple[str, str]:
    """(World/verifier image, original's image) of a window: ``images.json`` in its directory (the kernel debugging lane's
    CUDA 12 -> 13 migrations: CUDA 13 and 12.8), else the CUDA 12.8 / 11.8 pair."""
    f = wdir / "images.json"
    if f.exists():
        d = json.loads(f.read_text())
        return d.get("base", BASE_TAG), d.get("legacy", LEGACY_TAG)
    return BASE_TAG, LEGACY_TAG


def apptainer(tag: str, mounts: list[tuple[Path, str, bool]], args: list[str], *, workdir: str | None = None, net_none: bool = True,
              timeout: float = 7200) -> subprocess.CompletedProcess:
    """One gate container on an Apptainer host: image ``tag``'s SIF, this host's root mode, a fresh host /tmp."""
    env = [(k, os.environ[k]) for k in ("CRUCIBLE_CUDA_ARCH", "CUDA_VISIBLE_DEVICES") if os.environ.get(k)]
    tmp = APPTAINER_TMP / f"k-{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True)
    try:
        return sh(kp.apptainer_argv(SIF_DIR / kp.sif_name(tag), mounts, args, env=tuple(env), root=APPTAINER_ROOT, tmp=tmp, net_none=net_none, workdir=workdir),
                  timeout=timeout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ensure_sifs(needed: set[str], *, build: bool) -> None:
    """Apptainer: each needed image's SIF, built from its bundle Dockerfile when missing and ``build`` (on the GPU host, in a job)."""
    dfs = {BASE_TAG: HERE / "Dockerfile.base", LEGACY_TAG: HERE / "Dockerfile.legacy"}
    if (HERE / "images.json").exists():
        dfs = {tag: HERE / df for tag, df in json.loads((HERE / "images.json").read_text()).items()}
    for tag in sorted(needed):
        sif = SIF_DIR / kp.sif_name(tag)
        if not sif.exists() and build:
            SIF_DIR.mkdir(parents=True, exist_ok=True)
            (SIF_DIR / f"{sif.stem}.def").write_text(kp.apptainer_def(dfs[tag].read_text()))
            p = sh(["apptainer", "build", "--fakeroot", str(sif) + ".part", str(SIF_DIR / f"{sif.stem}.def")], timeout=7200)
            if p.returncode != 0:
                raise RuntimeError(f"build {sif}: {p.stderr[-800:]}")
            os.replace(str(sif) + ".part", sif)
        if not sif.exists():
            raise RuntimeError(f"no {sif}: build it on this host with --build-sif (inside a job, not on a login node)")


def ensure_images(needed: set[str] | None = None) -> None:
    """The images the selected windows name (``needed``; every image when None), built from the bundle's Dockerfiles:
    on a shared disk only what this run uses is built."""
    pairs = [(BASE_TAG, HERE / "Dockerfile.base"), (LEGACY_TAG, HERE / "Dockerfile.legacy")]
    if (HERE / "images.json").exists():  # every image a window names, with its Dockerfile in the bundle
        pairs = [(tag, HERE / df) for tag, df in json.loads((HERE / "images.json").read_text()).items()]
    for tag, df in pairs:
        if needed is not None and tag not in needed:
            continue
        if sh(["docker", "image", "inspect", tag], timeout=60).returncode != 0:
            p = sh(["docker", "build", "-t", tag, "-"], input=df.read_text(), timeout=7200)
            if p.returncode != 0:
                raise RuntimeError(f"build {tag}: {p.stderr[-800:]}")


def hooks_of(tag: str) -> Path:
    """The hooks directory for a World image: the CUPTI injection library links that image's CUPTI (12 vs 13)."""
    if tag == BASE_TAG:
        return HERE / "hooks"
    d = HERE / f"hooks-{tag.split(':', 1)[-1]}"
    if not d.exists():
        shutil.copytree(HERE / "hooks", d, ignore=shutil.ignore_patterns("*.so"))
    return d


def build_launch_lib(tag: str = BASE_TAG) -> None:
    hooks = hooks_of(tag)
    lib = hooks / "libcrucible_launch.so"
    if lib.exists():
        return
    cmd = " ".join(kp.LAUNCH_BUILD)
    if MODE == "apptainer":
        p = apptainer(tag, [(hooks, "/h", True)], ["bash", "-c", cmd], workdir="/h", net_none=False)
        if p.returncode != 0 or not (hooks / "libcrucible_launch.so").exists():
            raise RuntimeError(f"CUPTI injection build failed: {p.stderr[-600:]}")
        return
    p = sh(["docker", "run", "--rm", "--name", f"crucible-k-gpu-cupti-{uuid.uuid4().hex[:8]}", "-v", f"{hooks}:/h", "-w", "/h", tag,
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
    if MODE == "apptainer":
        p = apptainer(images_of(wdir)[1], [(wdir / "base", "/t", False), (HERE / "hooks", "/h", False), (out, "/o", True), (wdir / "legacy_build.sh", "/b.sh", False)],
                      ["bash", "/b.sh"])
    else:
        p = sh(["docker", "run", "--rm", "--name", f"crucible-k-gpu-legacy-{uuid.uuid4().hex[:8]}", "--network", "none", "-v", f"{wdir / 'base'}:/t:ro",
                "-v", f"{HERE / 'hooks'}:/h:ro", "-v", f"{out}:/o", "-v", f"{wdir / 'legacy_build.sh'}:/b.sh:ro", images_of(wdir)[1], "bash", "/b.sh"])
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
            "--param", "legacy_dir=/k/legacy", "--param", "hooks_dir=/k/hooks", *(() if mode == "apptainer" and APPTAINER_ROOT == "user" else ("--require-root",))]
    build_launch_lib(images_of(wdir)[0])
    mounts = [(HERE / "kit", "/crucible_verify"), (wdir / "base", "/k/base"), (wdir / "legacy", "/k/legacy"), (hooks_of(images_of(wdir)[0]), "/k/hooks"),
              (wdir / "kspec.json", "/k/kspec.json"), (patch, "/k/patch"), (out, "/out")]
    if mode == "apptainer":
        p = apptainer(images_of(wdir)[0], [(src, dst, dst == "/out") for src, dst in mounts], args)
    else:
        cmd = ["docker", "run", "--rm", "--name", f"crucible-k-gpu-{tag}-{uuid.uuid4().hex[:8]}", "--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all",
               "-e", f"CRUCIBLE_CUDA_ARCH={arch}", "--network", "none", "--init", "--shm-size", "2g"]
        for src, dst in mounts:
            cmd += ["-v", f"{src}:{dst}" + ("" if dst == "/out" else ":ro")]
        cmd += [images_of(wdir)[0], *args]
        p = sh(cmd, timeout=7200)
    rj = out / "crucible_reward.json"
    doc = json.loads(rj.read_text()) if rj.exists() else {"infra": {"status": "masked", "failure_kind": "no_reward", "reason": p.stderr[-400:]}}
    if not rj.exists():  # a grade that left nothing: the container's own error, in the run log
        print(f"[grade {wdir.name} {tag}] no reward (rc {p.returncode}): {p.stderr[-400:]}", file=sys.stderr, flush=True)
    return {"reward": doc.get("reward"), "gates": gates_of(out, doc), "infra": doc.get("infra")}


def gates_of(out: Path, doc: dict) -> dict[str, str]:
    """Per-gate verdicts of one grade: the report's ``gates`` (``pass``/``fail``/``skip``), else the reward document's
    ``check/<gate>`` components (1.0 -> pass). The reward document's own ``gates`` field is empty for this plan: before
    2026-10-23 S8 read it, so no fault mutant ever counted as killed (every window failed S8, whatever its verifier did)."""
    rep = out / "report.json"
    if rep.exists():
        g = json.loads(rep.read_text()).get("gates") or {}
        if g:
            return {k: str(v) for k, v in g.items()}
    comps = doc.get("components") or {}
    return {k.removeprefix("check/"): ("pass" if float(v) == 1.0 else "fail") for k, v in comps.items() if k.startswith("check/")}


def milestones_of(doc: dict) -> dict:
    return {g: int(v == "pass") for g, v in (doc.get("gates") or {}).items()}


def mutant_targets(name: str, spec: dict) -> list[str]:
    """The milestones a fault mutant is aimed at: those of the program its name ends with (``mutant_fault-<class>-<program>``)."""
    prog = name.rsplit("-", 1)[-1]
    return [m["id"] for m in spec["milestones"] if prog in (m.get("programs") or [m.get("program")])]


def mutant_killed(name: str, grade_res: dict, spec: dict) -> bool:
    """Killed: a milestone the mutant is aimed at fails (any milestone, when its program is not named)."""
    ms = milestones_of(grade_res)
    targets = [g for g in mutant_targets(name, spec) if g in ms] or list(ms)
    return any(ms[g] == 0 for g in targets)


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


def flake_verdict(runs: list[dict]) -> dict:
    """S10 over the oracle's repeated grades. A masked grade (infra: on 2026-10-23 the *original* program's launch timed
    out on a loaded GPU) is no evidence either way and leaves the denominator, as in the kernel debug lane; before
    2026-10-23 it counted as a false negative here."""
    masked = sum(1 for r in runs if (r["infra"] or {}).get("status") == "masked")
    bad = sum(1 for r in runs if r["reward"] != 1.0 and (r["infra"] or {}).get("status") != "masked")
    n = len(runs) - masked
    return {"n": len(runs), "measured": n, "false_negatives": bad, "masked": masked, "cp_upper": round(cp_upper(bad, n), 4)}


def window(wid: str, a: argparse.Namespace) -> dict:
    wdir = HERE / "windows" / wid
    if not (wdir / "base").exists():
        with tarfile.open(wdir / "base.tar.gz") as t:
            t.extractall(wdir, filter="data")
    spec = json.loads((wdir / "kspec.json").read_text())
    res = {"id": wid, "arch": a.arch, "gpu": sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).stdout.strip(), "at": time.strftime("%FT%TZ", time.gmtime()),
           "runtime": {"mode": a.mode, **({"root": APPTAINER_ROOT, "trusted": APPTAINER_ROOT == "fakeroot"} if a.mode == "apptainer" else {"trusted": True})}}
    res["originals"] = build_originals(wdir, spec, json.loads((wdir / "legacy.json").read_text()))
    os.chmod(wdir / "legacy", 0o700)
    # only a window with a timed milestone needs a quiet GPU: correctness windows can share the host with other gate runs
    res["idle"] = idle() if any(m.get("perf") for m in spec["milestones"]) else {"skipped": "no timed milestone"}
    res["oracle"] = grade(wdir, wdir / "oracle.patch", a.arch, a.mode, "oracle")
    res["floors"] = {p.stem: grade(wdir, p, a.arch, a.mode, p.stem) for p in sorted(wdir.glob("floor_*.patch"))}
    res["mutants"] = {p.stem: grade(wdir, p, a.arch, a.mode, p.stem) for p in sorted(wdir.glob("mutant_*.patch"))}
    if a.flake:
        half = a.flake // 2
        runs = [grade(wdir, wdir / "oracle.patch", a.arch, a.mode, f"flake1-{i}") for i in range(half)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            runs += list(ex.map(lambda i: grade(wdir, wdir / "oracle.patch", a.arch, a.mode, f"flake4-{i}"), range(a.flake - half)))
        res["flake"] = flake_verdict(runs)
    res["S5_oracle"] = res["oracle"]["reward"] == 1.0
    res["S8_floors"] = all(f["reward"] == 0.0 for f in res["floors"].values())
    # mutants shown equivalent on this program's own inputs (``equivalent_mutants.json``: the reason, citing the code) are
    # listed apart, never counted as caught; at least one counted mutant must remain
    eq = json.loads((wdir / "equivalent_mutants.json").read_text()) if (wdir / "equivalent_mutants.json").exists() else {}
    res["S8_mutants_excluded"] = {k: eq[k] for k in res["mutants"] if k in eq and milestones_of(res["mutants"][k]) and not mutant_killed(k, res["mutants"][k], spec)}
    res["S8_mutants_killed"] = {k: mutant_killed(k, m, spec) for k, m in res["mutants"].items() if k not in res["S8_mutants_excluded"]}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{wid}.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--mode", default="docker", choices=("docker", "apptainer"))
    ap.add_argument("--flake", type=int, default=60)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--build-sif", action="store_true", help="apptainer: build missing SIFs from the bundle's Dockerfiles")
    a = ap.parse_args()
    global MODE
    MODE = a.mode
    os.environ["CRUCIBLE_CUDA_ARCH"] = a.arch
    chosen = [w for w in json.loads((HERE / "index.json").read_text()) if not a.only or w["id"] in a.only]
    if a.mode == "apptainer":
        ensure_sifs({t for w in chosen for t in images_of(HERE / "windows" / w["id"])}, build=a.build_sif)
    else:
        ensure_images({t for w in chosen for t in images_of(HERE / "windows" / w["id"])})
    for w in chosen:
        build_launch_lib(images_of(HERE / "windows" / w["id"])[0])
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
