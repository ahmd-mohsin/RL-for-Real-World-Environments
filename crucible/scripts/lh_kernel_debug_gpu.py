#!/usr/bin/env python3
"""Kernel debugging lane, the GPU half of the gates (swarm-pipeline/10 §6), run on a CUDA host from the bundle that
``lh_kernel_debug.py stage-gpu`` wrote (copied in as ``run_gpu.py``). Standard library + the bundled kit only.

    python3 run_gpu.py --arch 120 --mode docker [--only ID ...] [--flake 60] [--recalibrate]

Per task (``tasks/<id>``):

1. **Reference** (root-only ``ref/``, 0700): B built with the grader's flags for ``sm_<arch>`` and B's op test patched by
   ``kernel_harness`` (``ref/harness.cpp``).
2. **Calibration** (S1 executed, case ownership, S6 executed edges, sanitizer and perf calibration; ``calibration.json``):
   - the oracle (A' -> B) graded in ``mode=pool``: the pool cases that pass on B on every seed (cases B itself fails
     or does not support leave the pool);
   - each revert mutant (B with only fault k reverted) in ``mode=pool``: milestone k's **cases** are the pool cases it
     breaks (capped at 32, the fix's own added cases first). A fault whose revert breaks no case and is not caught by its
     sanitizer tool is not real on this GPU: its milestone is dropped and the task with it if fewer than two remain;
   - **executed edges**: m_j requires m_k when a case of m_j also fails under fault k's revert (the faults interact);
   - sanitizer milestones: the oracle must be clean and the revert dirty on the milestone's first cases, else the
     sanitizer check is dropped from that milestone (recorded);
   - perf milestones: B vs the revert at R and 3R repetitions; the noise floor is the spread of B against itself over
     the rounds; the bar is ``1 + max(3 * noise, 0.05)`` and is kept only when the revert is at least ``bar + 0.10``
     slower (headroom), per GPU model.
3. **Gates** with the calibrated spec: the oracle scores 1 (S5), every floor 0 (S8), each mutant fails its milestone
   (S8), and ``--flake`` oracle grades half at concurrency 1 and half at 4 (S10: Clopper-Pearson upper bound <= 5%).

Results: ``results/<id>.json`` and ``tasks/<id>/kdspec.calibrated.json``. Shared-machine rules: containers are
``crucible-lh-k-gpu-*`` and only those are removed; nothing is pruned; timed steps wait for an idle GPU. Docker needs
``--runtime=nvidia`` (the lab 5090); Apptainer runs the same image as a SIF (``--build-sif`` builds it from ``Dockerfile``
on the host, into ``CRUCIBLE_SIF_DIR``, default ``sif/``). ``CRUCIBLE_APPTAINER_ROOT=fakeroot`` (default; a host with subuid
ranges, as Marlowe) keeps the grader root inside, which it needs to drop privileges for candidate code;
``CRUCIBLE_APPTAINER_ROOT=user`` (DeltaAI: no subuid range, one uid) runs the grader as the invoking user without
``--require-root``, so every grade reports ``trusted: false`` (fit for the gates, which grade the lane's own patches).
``CRUCIBLE_APPTAINER_TMP`` is the host directory each container's /tmp lives in (node-local disk; a build does not fit the
session tmpfs). ``CRUCIBLE_IDLE_LOAD`` is the host load below which a timed step may start (default 8; a many-core node
running one stream per GPU sets its own).
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
from shlex import quote as shlex_quote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "kit"))
from crucible.factory.lanes import kernel_harness as kh
from crucible.factory.lanes import kernel_plan as kp
from crucible.factory.lanes import kernel_profiles as kprof

IMAGE = "crucible-lh-k:base-12.8.1"
PLAN = "crucible.factory.lanes.kernel_debug_plan"
CASES_PER_MILESTONE = 32
PERF_HEADROOM = 0.10
SANITIZERS = ("racecheck", "synccheck", "memcheck")
MARGIN = 0.5  # an owned case B passes above this share of its tolerance on a stability grade is borderline and leaves
PROFILE_SAMPLE = 192  # pool cases per milestone whose kernel launches are profiled to choose its sanitizer cases
CCACHE = HERE / "ccache"  # gate runs only (the grader's own patches): a task grade never mounts a build cache
USE_CCACHE = True
SIF_DIR = Path(os.environ.get("CRUCIBLE_SIF_DIR") or HERE / "sif")
APPTAINER_ROOT = os.environ.get("CRUCIBLE_APPTAINER_ROOT", "fakeroot")
APPTAINER_TMP = Path(os.environ.get("CRUCIBLE_APPTAINER_TMP") or "/tmp/crucible-apptainer")
IDLE_LOAD = float(os.environ.get("CRUCIBLE_IDLE_LOAD") or 8.0)


def ccache_dir() -> Path:
    CCACHE.mkdir(exist_ok=True)
    os.chmod(CCACHE, 0o777)  # the grader builds as an unprivileged uid
    return CCACHE


def sh(cmd: list[str], *, timeout: float = 14400, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=input, check=False, errors="replace")


def rmtree(p: Path) -> None:
    """Remove a directory the containers wrote into: Docker runs the grader as root, so its outputs are root-owned on the
    host and a plain ``shutil.rmtree`` would silently leave gigabytes behind (a shared disk); the leftovers go through a
    container of this lane's image."""
    p = Path(p)
    shutil.rmtree(p, ignore_errors=True)
    if p.exists() and shutil.which("docker"):
        sh(["docker", "run", "--rm", "--name", f"crucible-lh-k-gpu-rm-{uuid.uuid4().hex[:8]}", "-v", f"{p.parent}:/x", IMAGE, "rm", "-rf", f"/x/{p.name}"], timeout=1800)
    if p.exists():
        raise RuntimeError(f"could not remove {p}")


def gpu_name() -> str:
    return sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=30).stdout.strip().splitlines()[0]


def idle(max_util: int = 10, max_load: float | None = None, wait_s: float = 3600) -> dict:
    max_load = IDLE_LOAD if max_load is None else max_load
    t0 = time.time()
    while True:
        u = sh(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], timeout=30).stdout.split()
        load = os.getloadavg()[0]
        util = int(u[0]) if u and u[0].strip().isdigit() else 0
        if util <= max_util and load < max_load:
            return {"gpu_util": util, "load1": round(load, 2), "waited_s": round(time.time() - t0)}
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"GPU never idle: util {util}% load {load}")
        time.sleep(60)


def ensure_image(mode: str, *, build: bool = False) -> None:
    if mode == "docker" and sh(["docker", "image", "inspect", IMAGE], timeout=60).returncode != 0:
        p = sh(["docker", "build", "-t", IMAGE, "-"], input=(HERE / "Dockerfile").read_text())
        if p.returncode != 0:
            raise RuntimeError(f"build {IMAGE}: {p.stderr[-800:]}")
    if mode == "apptainer":
        sif = SIF_DIR / kp.sif_name(IMAGE)
        if not sif.exists() and build:
            SIF_DIR.mkdir(parents=True, exist_ok=True)
            (SIF_DIR / f"{sif.stem}.def").write_text(kp.apptainer_def((HERE / "Dockerfile").read_text()))
            p = sh(["apptainer", "build", "--fakeroot", str(sif) + ".part", str(SIF_DIR / f"{sif.stem}.def")], timeout=7200)
            if p.returncode != 0:
                raise RuntimeError(f"build {sif}: {p.stderr[-800:]}")
            os.replace(str(sif) + ".part", sif)
        if not sif.exists():
            raise RuntimeError(f"no {sif}: build it on this host with --build-sif (inside a job, not on a login node)")


def container(mode: str, mounts: list[tuple[Path, str, bool]], args: list[str], *, arch: str, tag: str, timeout: float = 14400) -> subprocess.CompletedProcess:
    if mode == "apptainer":
        env = [("CRUCIBLE_CUDA_ARCH", arch)] + ([("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])] if os.environ.get("CUDA_VISIBLE_DEVICES") else [])
        tmp = APPTAINER_TMP / f"{tag}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        try:
            return sh(kp.apptainer_argv(SIF_DIR / kp.sif_name(IMAGE), mounts, args, env=tuple(env), root=APPTAINER_ROOT, tmp=tmp), timeout=timeout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        cmd = ["docker", "run", "--rm", "--name", f"crucible-lh-k-gpu-{tag}-{uuid.uuid4().hex[:8]}", "--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all",
               "-e", f"CRUCIBLE_CUDA_ARCH={arch}", "--network", "none", "--init", "--shm-size", "4g"]
        for src, dst, rw in mounts:
            cmd += ["-v", f"{src}:{dst}" + ("" if rw else ":ro")]
        cmd += [IMAGE, *args]
    return sh(cmd, timeout=timeout)


def build_launch_lib(mode: str, arch: str) -> None:
    lib = HERE / "hooks" / "libcrucible_launch.so"
    if lib.exists():
        return
    p = container(mode, [(HERE / "hooks", "/h", True)], ["bash", "-c", "cd /h && " + " ".join(kp.LAUNCH_BUILD)], arch=arch, tag="cupti")
    if p.returncode != 0 or not lib.exists():
        raise RuntimeError(f"CUPTI injection build failed: {p.stderr[-600:]}")


def unpack(d: Path) -> None:
    for side in ("a", "b"):
        if not (d / side).exists():
            with tarfile.open(d / f"{side}.tar.gz") as t:
                t.extractall(d, filter="data")


def build_reference(d: Path, mode: str, arch: str, spec: dict) -> dict:
    """B with the patched harness, built as root into ``ref/`` (0700): what the reference runs in every grade."""
    ref = d / "ref"
    H = kprof.of_spec(spec)
    extra = list(spec.get("harness_extra") or [])
    stamp = "\n".join([kh.cuda_arch_flag([arch]), harness_digest(H), *(f"{x.get('mode') or 'test'} {x['code']}" for x in extra),
                       *(spec.get("cmake_extra") or [])])  # flags and added cases  # the reference's identity: the GPU arch and the grader's added cases
    if (ref / H.exe).exists() and (ref / "arch").exists() and (ref / "arch").read_text() == stamp:
        return {"ok": True, "cached": True}
    if ref.exists():
        rmtree(ref)
    src = d / "ref_src"
    if src.exists():
        rmtree(src)
    shutil.copytree(d / "b", src, symlinks=True)
    if H is kprof.GGML:
        text = (src / kh.HARNESS).read_text(errors="replace")
        patched = kh.add_cases(kh.patch_harness(text, need_perf=any("perf" in m["checks"] for m in spec["milestones"]), header=kh.ggml_header(src)), extra)
    else:  # a status profile: the grader owns B's test sources it runs (a JSON map, the grader's ref/harness.cpp)
        patched = json.dumps({t: (src / t).read_text(errors="replace") for t in H.targets}, sort_keys=True)
    H.prepare(src, patched)
    ref.mkdir()
    (ref / "harness.cpp").write_text(patched)
    jobs = str(max(2, min(32, os.cpu_count() or 4)))
    flags = [*H.cmake_cuda([arch]), "-DBUILD_SHARED_LIBS=OFF", *(spec.get("cmake_extra") or []), *(kh.cmake_ccache() if USE_CCACHE else ())]  # static, as the grader builds
    script = ("export CCACHE_DIR=/ccache CCACHE_BASEDIR=/ CCACHE_NOHASHDIR=1 CCACHE_UMASK=000 CCACHE_MAXSIZE=12G; "
              f"cmake -S /src -B /ref/build -G Ninja {' '.join(flags)} > /ref/configure.log 2>&1 && "
              f"cmake --build /ref/build --target {H.target} -j {jobs} > /ref/build.log 2>&1; rc=$?; "
              "mkdir -p /ref/bin && cp -a /ref/build/bin/. /ref/bin/ ; exit $rc")
    p = container(mode, [(src, "/src", False), (ref, "/ref", True), (ccache_dir(), "/ccache", True)], ["bash", "-c", script], arch=arch, tag="ref")
    ok = p.returncode == 0 and (ref / H.exe).exists()
    if ok:
        (ref / "arch").write_text(stamp)
        rmtree(ref / "build")
        rmtree(src)
    os.chmod(ref, 0o700)
    return {"ok": ok, "rc": p.returncode, "log": "" if ok else ((ref / "build.log").read_text(errors="replace")[-1500:] if (ref / "build.log").exists() else p.stderr[-800:])}


def harness_digest(H: kprof.KernelHarness | None = None) -> str:
    """The grader-owned harness hooks (``kernel_harness``): part of the reference's identity, so a hook change (2026-10-23:
    per-case input streams, the exact attention reference) rebuilds every reference instead of reusing one built before."""
    import hashlib

    h = hashlib.sha256(Path(kh.__file__).read_bytes())
    if H is not None and H is not kprof.GGML:  # a ggml reference keeps its identity; another profile's code is part of its own
        h.update(Path(kprof.__file__).read_bytes() + json.dumps(H.describe(), sort_keys=True).encode())
    return "harness " + h.hexdigest()[:16]


def kit_digest() -> bytes:
    """The grader's own code (the bundled kit): part of every pool grade's reuse key, so a grade cached before a grader
    fix is never reused after it (found 2026-10-22: sweeps graded by an older kit were read back after the fix)."""
    import hashlib

    h = hashlib.sha256()
    for f in sorted((HERE / "kit").rglob("*.py")):
        h.update(str(f.relative_to(HERE)).encode() + b"\0" + f.read_bytes())
    return h.hexdigest().encode()


def grade(d: Path, patch: Path, spec_path: Path, mode: str, arch: str, tag: str, *, pool: bool = False, only: str = "", params: tuple[str, ...] = ()) -> dict:
    out = d / "grades" / tag
    import hashlib

    key = hashlib.sha256(b"\0".join([patch.read_bytes(), spec_path.read_bytes(), tag.encode(), str(pool).encode(), only.encode(), *[p.encode() for p in params],
                                      kit_digest()])).hexdigest()
    if pool and (out / "inputs.sha256").exists() and (out / "inputs.sha256").read_text() == key and (out / "report.json").exists():
        report = json.loads((out / "report.json").read_text())  # a pool grade of the same patch, spec and grader: reused (calibration restarts)
        doc = json.loads((out / "crucible_reward.json").read_text())
        return {"reward": doc.get("reward"), "infra": doc.get("infra"), "gates": report.get("gates") or {}, "checks": report.get("checks") or []}
    if out.exists():
        rmtree(out)
    out.mkdir(parents=True)
    args = ["python3", "-I", "/crucible_verify/crucible/verify/runtime/run.py", "--plan", PLAN, "--out", "/out", "--base", "/k/base", "--patch", "/k/patch",
            "--param", "spec=/k/kdspec.json", "--param", f"arch={arch}", "--param", "base_tree=/k/base", "--param", "ref=/k/ref", "--param", "hooks_dir=/k/hooks",
            *(() if mode == "apptainer" and APPTAINER_ROOT == "user" else ("--require-root",))] + (["--param", "mode=pool"] if pool else []) + (["--param", f"only={only}"] if only else []) + [x for p in params for x in ("--param", p)]
    mounts = [(HERE / "kit", "/crucible_verify", False), (d / "a", "/k/base", False), (d / "ref", "/k/ref", False), (HERE / "hooks", "/k/hooks", False),
              (spec_path, "/k/kdspec.json", False), (patch, "/k/patch", False), (out, "/out", True)]
    if USE_CCACHE:
        args += ["--param", "ccache=/k/ccache"]
        mounts.append((ccache_dir(), "/k/ccache", True))
    p = container(mode, mounts, args, arch=arch, tag=tag[:24])
    rj, rep = out / "crucible_reward.json", out / "report.json"
    if pool and rep.exists():
        (out / "inputs.sha256").write_text(key)
    doc = json.loads(rj.read_text()) if rj.exists() else {"infra": {"status": "masked", "failure_kind": "no_reward", "reason": p.stderr[-400:]}}
    if not rj.exists():  # a grade that left nothing: the container's own error, in the run log (a masked calibration says only "{}")
        print(f"[grade {d.name} {tag}] no reward (rc {p.returncode}): {p.stderr[-400:]}", file=sys.stderr, flush=True)
    report = json.loads(rep.read_text()) if rep.exists() else {}
    return {"reward": doc.get("reward"), "infra": doc.get("infra"), "gates": report.get("gates") or {}, "checks": report.get("checks") or []}


def check_detail(g: dict, cid: str) -> dict:
    return next((c.get("detail") or {} for c in g.get("checks") or [] if c.get("id") == cid), {})


def check_status(g: dict, cid: str) -> str | None:
    return next((c.get("status") for c in g.get("checks") or [] if c.get("id") == cid), None)


def per_case(g: dict, mid: str) -> dict[str, bool]:
    return dict(check_detail(g, f"{mid}:golden").get("per_case") or {})


def per_case_fixed(g: dict, mid: str) -> dict[str, bool]:
    """Per case on the spec's fixed seeds only (pool grades since 2026-10-23; an older grade has none: all seeds)."""
    det = check_detail(g, f"{mid}:golden")
    return dict(det.get("per_case_fixed") or det.get("per_case") or {})


def sanitizer_discriminates(row: dict) -> bool:
    """A sanitizer calibration row (``calibration.sanitizer[<m>:<tool>]``) that can grade its milestone."""
    return bool(row.get("clean_on_b") and row.get("dirty_on_revert"))


def fa_reference_verdict(ev: dict, *, margin: float = MARGIN, det: float = 1e-3) -> str:
    """One FLASH_ATTN_EXT case of B from the probe (``KDGrader.fa_probe``):
    - ``unmodelled``: the exact computation does not model the case (the CPU column stayed);
    - ``nondeterministic``: B's GPU output differs between two processes on the same inputs (no golden of it is fair);
    - ``agree``: B's GPU matches the CPU kernel (nothing to change);
    - ``cpu_wrong``: B's GPU matches the exact computation within ``margin`` of the tolerance and the CPU kernel does
      not: two independent computations agree against the CPU, so the CPU column is the wrong reference here;
    - ``gpu_disagrees``: B's GPU matches neither (the case stays as it is; B fails it and it leaves the pool)."""
    tol = float(ev.get("tol") or 0.0)
    g_c, g_x, c_x, g_g = (ev.get(k) for k in ("gpu_vs_cpu", "gpu_vs_exact", "cpu_vs_exact", "gpu_vs_gpu"))
    if not ev.get("applied"):
        return "unmodelled"
    if g_g is None or g_g > det * tol:
        return "nondeterministic"
    if g_c is not None and g_c <= tol:
        return "agree"
    if g_x is not None and g_x <= margin * tol and (c_x is None or c_x > tol):
        return "cpu_wrong"
    return "gpu_disagrees"


def fa_exact_choice(evidence: dict[str, dict], *, min_agree: int = 8) -> dict:
    """The cases whose reference becomes the exact computation, and the check that the computation itself is right: on
    cases where B's GPU and the CPU kernel agree, the exact result must agree with both (at least ``min_agree`` such
    cases); otherwise nothing is switched (``validated`` false)."""
    verdicts = {c: fa_reference_verdict(ev) for c, ev in evidence.items()}
    counts: dict[str, int] = {}
    for v in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    agree = [c for c, v in verdicts.items() if v == "agree" and (evidence[c].get("cpu_vs_exact") or 0.0) <= float(evidence[c]["tol"])
             and (evidence[c].get("gpu_vs_exact") or 0.0) <= float(evidence[c]["tol"])]
    validated = len(agree) >= min_agree
    exact = sorted(c for c, v in verdicts.items() if v == "cpu_wrong") if validated else []
    return {"exact": exact, "verdicts": counts, "validated": validated, "exact_agrees_where_cpu_and_gpu_agree": len(agree),
            "agree_checked": counts.get("agree", 0)}


def borderline(g: dict, mid: str, margin: float = MARGIN) -> set[str]:
    """Cases B passed with an error above ``margin`` of their tolerance (or with no measured error): a fresh seed fails
    them (2026-10-22: a q4_1 matmul of one 32-value block passed three stability seeds and failed the S10 run at 2x)."""
    r = check_detail(g, f"{mid}:golden").get("per_case_ratio") or {}
    return {c for c, x in r.items() if x is None or x > margin}


# ---- calibration (pure functions are unit-tested in tests/unit/test_m17_kd_gpu_math.py) ---------------------------------------
def owned_cases(b_pass: dict[str, bool], revert_pass: dict[str, bool], added: list[str], cap: int = CASES_PER_MILESTONE) -> list[str]:
    """The pool cases B passes and the single revert breaks; the fix's own added cases first."""
    broken = [c for c, ok in revert_pass.items() if b_pass.get(c) and not ok]
    first = [c for c in added if c in broken]
    return (first + sorted(c for c in broken if c not in first))[:cap]


def executed_edges(owned: dict[str, list[str]], breaks: dict[str, set[str]]) -> dict[str, list[str]]:
    """m_j -> [m_k]: a case owned by m_j also fails under fault k's revert (k != j)."""
    return {j: sorted(k for k, broken in breaks.items() if k != j and set(cs) & broken) for j, cs in owned.items()}


def perf_bar(b_ratios: list[float], revert_ratio: float, *, floor: float = 0.05, headroom: float = PERF_HEADROOM) -> dict:
    """The bar for a perf milestone from B-vs-B ratios (the noise floor) and the revert-vs-B ratio (the regression)."""
    noise = (max(b_ratios) - min(b_ratios)) / min(b_ratios) if b_ratios and min(b_ratios) > 0 else math.inf
    bar = 1.0 + max(3.0 * noise, floor)
    ok = math.isfinite(noise) and revert_ratio >= bar + headroom
    return {"ok": ok, "bar": round(bar, 3), "noise": round(noise, 4) if math.isfinite(noise) else None, "regression": round(revert_ratio, 3)}


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    if n == 0 or k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        tail = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if tail > alpha else (lo, mid)
    return hi


def start_not_real(spec: dict, built: bool, a_fails: dict[str, list[str]], a_san: dict[str, str | None]) -> bool:
    """S1 on the start: every correctness milestone's pool cases pass and every sanitizer check is clean with all faults
    in, so no fault shows on this GPU. Perf milestones are left out: their faults keep results correct, so the start passes
    every case by design (before 2026-10-22 a perf task was dropped here as "not real")."""
    bug = [m["id"] for m in spec["milestones"] if m["kind"] != "perf"]
    if not built or not bug:
        return False
    return not any(a_fails.get(m) for m in bug) and all(v == "pass" for k, v in a_san.items() if k.split(":", 1)[0] in bug)


LAUNCH_PROFILE = r"""
import json, os, subprocess
cases, out = json.load(open("/w/cases.json")), {}
for i, c in enumerate(cases):
    open("/w/c.txt", "w").write(c + "\n")
    log = "/w/l%d.tsv" % i
    env = {**os.environ, "CRUCIBLE_SEED": "7", "CRUCIBLE_CASES": "/w/c.txt", "GGML_CUDA_DISABLE_GRAPHS": "1", "LD_LIBRARY_PATH": "/k/ref/bin",
           "CUDA_INJECTION64_PATH": "/k/hooks/libcrucible_launch.so", "CRUCIBLE_LAUNCH_LOG": log}
    try:
        subprocess.run(["/k/ref/bin/test-backend-ops", "test", "-b", "CUDA0"], env=env, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        pass
    out[c] = [ln.split("\t", 1)[1] for ln in open(log).read().splitlines() if "\t" in ln] if os.path.exists(log) else []
json.dump(out, open("/w/out.json", "w"))
"""


def profile_launches(d: Path, cases: list[str], mode: str, arch: str) -> dict[str, list[str]]:
    """case -> the kernels its process launches on the reference build (B), each case alone under the CUPTI hook; cached
    in ``launch_profile.json``."""
    prof = d / "launch_profile.json"
    cached = json.loads(prof.read_text()) if prof.exists() else {}
    todo = [c for c in cases if c not in cached]
    if todo:
        work = d / "lprof"
        if work.exists():
            rmtree(work)
        work.mkdir()
        (work / "cases.json").write_text(json.dumps(todo))
        container(mode, [(d / "ref", "/k/ref", False), (HERE / "hooks", "/k/hooks", False), (work, "/w", True)], ["python3", "-I", "-c", LAUNCH_PROFILE],
                  arch=arch, tag="lprof", timeout=7200)
        if (work / "out.json").exists():
            cached.update(json.loads((work / "out.json").read_text()))
            prof.write_text(json.dumps(cached, indent=1))
        rmtree(work)
    return {c: cached.get(c, []) for c in cases}


def launches_kernel(launched: str, kernel: str) -> bool:
    """A CUPTI kernel name (mangled: the identifier with its length in front) is ``kernel``, not a longer name that starts
    with it (``cpy_scalar`` is not ``cpy_scalar_transpose``)."""
    if launched == kernel or launched.startswith((kernel + "(", kernel + "<")):
        return True
    tok = f"{len(kernel)}{kernel}"
    i = launched.find(tok)
    while i >= 0:
        if i == 0 or not launched[i - 1].isdigit():
            return True
        i = launched.find(tok, i + 1)
    return False


def sanitizer_cases(prof: dict[str, list[str]], kernels: list[str], k: int = 4) -> list[str]:
    """Up to ``k`` profiled cases that launch the fault's kernels: one per kernel, the rarest kernels first (a kernel
    every case launches says little about this fault), the shortest case name first within a kernel."""
    hits = {c: {kn for kn in kernels if any(launches_kernel(x, kn) for x in names)} for c, names in prof.items()}
    freq = {kn: sum(1 for h in hits.values() if kn in h) for kn in kernels}
    picked: list[str] = []
    for kn in sorted((kn for kn in kernels if freq[kn]), key=lambda kn: (freq[kn], kn)):
        cands = sorted((c for c, h in hits.items() if kn in h and c not in picked), key=lambda c: (len(c), c))
        if cands:
            picked.append(cands[0])
        if len(picked) >= k:
            break
    rest = sorted((c for c, h in hits.items() if h and c not in picked), key=lambda c: (-len(hits[c]), len(c), c))
    return (picked + rest)[:k]


def target_sanitizers(d: Path, spec: dict, mode: str, arch: str) -> dict:
    """Each sanitizer milestone's ``sanitizer_cases``: profiled pool cases that launch its fault's kernels
    (``fault_kernels.json``, written at staging). Without it the tools ran on the pool's first cases, which often never
    launch the racy kernel (2026-10-22: races in the transposed copy and in flash attention's vector kernel)."""
    fk_p = d / "fault_kernels.json"
    fk = json.loads(fk_p.read_text()) if fk_p.exists() else {}
    from crucible.factory.lanes import kernel_debug_plan as kdp  # the bundled kit (on sys.path)

    ms = [m for m in spec["milestones"] if fk.get(m["id"]) and any(t in m["checks"] for t in SANITIZERS)]
    if not ms:
        return {}
    sample = list(dict.fromkeys(c for m in ms for c in kdp.pool_sample(list(m.get("case_pool") or []), list(m.get("cases_added") or []), cap=PROFILE_SAMPLE)))
    prof = profile_launches(d, sample, mode, arch)
    out = {}
    for m in ms:
        sc = sanitizer_cases(prof, fk[m["id"]])
        if sc:
            m["sanitizer_cases"] = sc
        out[m["id"]] = {"kernels": fk[m["id"]][:12], "profiled": len(sample), "cases": sc}
    return out


def adopt_extra_cases(spec: dict, listed: list[str], listed_perf: list[str] | None = None) -> dict[str, list[str]]:
    """milestone -> the grader's added cases (``harness_extra``) by name. An entry that names its case (``name``) must
    appear in the reference's listing: in test mode it joins its milestone's pool and leads its samples (like a fix's own
    added cases), in perf mode it leads a perf milestone's candidates. Unnamed test-mode entries are found as the listing
    minus every pool, for the milestones the lines were written for whose ops include the case's op. Which of them a fault
    breaks, or slows, is still the calibration's verdict; a named case the reference does not list is reported."""
    extra = spec.get("harness_extra") or []
    ms = {m["id"]: m for m in spec["milestones"]}
    known = {c for m in spec["milestones"] for c in m.get("case_pool") or []}
    test_set, perf_set = set(listed), set(listed_perf or [])
    out: dict[str, list[str]] = {}

    def add(m: dict, c: str, perf: bool) -> None:
        if perf:
            m["perf_cases"] = list(dict.fromkeys([c, *(m.get("perf_cases") or [])]))
            out.setdefault(f"{m['id']}:perf", []).append(c)
        else:
            m["case_pool"] = list(dict.fromkeys([c, *(m.get("case_pool") or [])]))
            m["cases_added"] = list(dict.fromkeys([c, *(m.get("cases_added") or [])]))
            out.setdefault(m["id"], []).append(c)

    for x in extra:
        m, name = ms.get(x["milestone"]), x.get("name")
        if m is None or not name:
            continue
        if name not in test_set and name not in perf_set:
            out.setdefault("missing", []).append(name)
        if name in test_set and (x.get("mode") or "test") in ("test", "both"):
            add(m, name, False)
        if name in perf_set and "perf" in m["checks"] and (x.get("mode") or "test") in ("perf", "both"):
            add(m, name, True)
    unnamed = {x["milestone"] for x in extra if not x.get("name") and (x.get("mode") or "test") in ("test", "both")}
    new = [c for c in listed if c not in known]
    for mid in sorted(unnamed):
        m = ms.get(mid)
        if m is None:
            continue
        for c in new:
            if c.split("(", 1)[0] in set(m["ops"]):
                add(m, c, False)
    return out


def fill_pools(d: Path, spec: dict, mode: str, arch: str) -> dict:
    """Case pools the CPU gates could not list (a repository whose CPU backend needs AVX2): the reference build lists
    every case of its test and perf modes (``CRUCIBLE_LIST``, nothing computed), filtered by each milestone's ops."""
    need = [m for m in spec["milestones"] if not m.get("case_pool") or ("perf" in m["checks"] and not m.get("perf_cases"))]
    if not need:
        return {"filled": 0}
    H = kprof.of_spec(spec)
    listed = {}
    for mmode in ("test", "perf") if H.perf else ("test",):
        argv, env = H.list_argv(f"/k/ref/{H.exe}", mmode, "CUDA0")
        p = container(mode, [(d / "ref", "/k/ref", False)], ["bash", "-c", " ".join([*(f"{k}={v}" for k, v in env.items()), *map(shlex_quote, map(str, argv))])],
                      arch=arch, tag=f"list-{mmode}")
        listed[mmode] = H.listed(p.stdout)
    listed.setdefault("perf", [])
    for m in need:
        ops = set(m.get("ops") or ())
        if not m.get("case_pool"):  # ggml: the cases of the milestone's ops; a status profile: every case its test executable lists
            m["case_pool"] = [c for c in listed["test"] if c.split("(", 1)[0] in ops] if H is kprof.GGML else list(listed["test"])
        if "perf" in m["checks"] and not m.get("perf_cases"):
            m["perf_cases"] = [c for c in listed["perf"] if c.split("(", 1)[0] in ops][:8]
    return {"filled": len(need), "test_cases": len(listed["test"]), "perf_cases": len(listed["perf"])}


def seeding_verdict(s1: list, s1b: list, s2: list, status: dict[str, str], cases: list[str]) -> dict:
    """The CPU half of S5 on dumps: the same seed twice gives identical records, another seed different inputs, and the
    CPU reference passes every case (``lh_kernel_debug.seeded_check``'s rule)."""
    key = lambda recs: [(x["case"], x["out"], x["ref"]) for x in recs]
    same = bool(s1) and key(s1) == key(s1b)
    differ = bool(s2) and any(x["ref"] != y["ref"] for x, y in zip(s1, s2, strict=False))
    cpu_ok = len(status) == len(cases) and all(v == "OK" for v in status.values())
    return {"ok": same and differ and cpu_ok, "same_seed_identical": same, "new_seed_differs": differ, "cpu_reference_ok": cpu_ok, "records": len(s1)}


def cpu_computable(bld: Path, cases: list[str], mode: str, arch: str, res: dict, *, want: int = 6, tries: int = 8) -> list[str]:
    """Up to ``want`` of ``cases`` (shortest first) the CPU backend computes: a case it claims and then aborts on (ik's
    CPU flash attention with f32 K/V) is skipped and recorded, never picked for the seeding check."""
    picked, crashed, i = [], [], 0
    for _ in range(tries):
        chunk = [c for c in cases[i:i + 24] if c not in crashed]
        if not chunk or len(picked) >= want:
            break
        (bld / "probe.txt").write_text(kh.cases_file(chunk))
        q = container(mode, [(bld, "/b", True)], ["bash", "-c", f"CRUCIBLE_ALLOW_CPU=1 CRUCIBLE_SEED=11 CRUCIBLE_CASES=/b/probe.txt /b/bin/{kh.TARGET} test -b CPU 2>&1"],
                      arch=arch, tag="cpuh-probe")
        st = kh.case_status(q.stdout)
        picked += [c for c in chunk if st.get(c) == "OK" and c not in picked]
        bad = kh.crashed_case(q.stdout)
        if q.returncode != 0 and bad:
            crashed.append(bad)
            continue  # the same window again, without the case it died on
        i += 24
    res["cpu_crashed_cases"] = crashed
    return picked[:want]


def host_cpu_gates(d: Path, spec: dict, mode: str, arch: str) -> dict:
    """The CPU gates the build machine could not run (a CPU backend that needs x86 AVX2): B's op test (patched, perf mode
    for a perf task) built on the CPU backend lists each milestone's case pool; the seeding executes on up to 6 pool
    cases; A''s op test lists the start's cases, so the fixes' own added cases are B's minus A''s. Writes the pools into
    ``kdspec.json`` and ``milestones.json``; cached in ``cpu_host.json``."""
    done = d / "cpu_host.json"
    if done.exists():
        return json.loads(done.read_text())
    res: dict = {"at": time.strftime("%FT%TZ", time.gmtime())}
    perf = any("perf" in m["checks"] for m in spec["milestones"])
    jobs = str(max(2, min(32, os.cpu_count() or 4)))
    listed: dict[str, dict[str, list[str]]] = {}
    for side in ("b", "a"):
        src, bld = d / f"cpuh_src_{side}", d / f"cpuh_{side}"
        for p in (src, bld):
            if p.exists():
                rmtree(p)
        shutil.copytree(d / side, src, symlinks=True)
        text = (src / kh.HARNESS).read_text(errors="replace")
        kh.prepare_tree(src, kh.patch_harness(text, need_perf=perf and side == "b", header=kh.ggml_header(src)))
        bld.mkdir()
        flags = [*kh.CMAKE_CPU, *(spec.get("cmake_extra") or []), *(kh.cmake_ccache() if USE_CCACHE else ())]
        script = ("export CCACHE_DIR=/ccache CCACHE_BASEDIR=/ CCACHE_NOHASHDIR=1 CCACHE_UMASK=000 CCACHE_MAXSIZE=12G; "
                  f"cmake -S /src -B /b -G Ninja {' '.join(flags)} > /b/configure.log 2>&1 && cmake --build /b --target {kh.TARGET} -j {jobs} > /b/build.log 2>&1; "
                  "rc=$?; tail -c 1500 /b/build.log; exit $rc")
        p = container(mode, [(src, "/src", False), (bld, "/b", True), (ccache_dir(), "/ccache", True)], ["bash", "-c", script], arch=arch, tag=f"cpuh-{side}")
        res[f"build_{side}"] = {"ok": p.returncode == 0, "rc": p.returncode, "log": "" if p.returncode == 0 else p.stdout[-1500:]}
        if p.returncode != 0:
            break
        listed[side] = {}
        for mmode in ("test", *(("perf",) if perf and side == "b" else ())):
            q = container(mode, [(bld, "/b", False)], ["bash", "-c", f"CRUCIBLE_ALLOW_CPU=1 CRUCIBLE_LIST=1 /b/bin/{kh.TARGET} {mmode} -b CPU"], arch=arch,
                          tag=f"cpuh-list-{side}")
            listed[side][mmode] = kh.listed_cases(q.stdout)
        if side == "b":
            for m in spec["milestones"]:
                ops = set(m["ops"])
                m["case_pool"] = sorted(c for c in listed["b"]["test"] if c.split("(", 1)[0] in ops)
                if "perf" in m["checks"] and not m.get("perf_cases"):
                    m["perf_cases"] = [c for c in listed["b"].get("perf", []) if c.split("(", 1)[0] in ops][:8]
            pool = cpu_computable(bld, sorted({c for m in spec["milestones"] for c in m["case_pool"]}, key=lambda c: (len(c), c)), mode, arch, res)
            res["seeded_cases"] = pool
            if pool:
                (bld / "cases.txt").write_text(kh.cases_file(pool))
                runs = " ; ".join(f"CRUCIBLE_ALLOW_CPU=1 CRUCIBLE_SEED={s} CRUCIBLE_CASES=/b/cases.txt CRUCIBLE_DUMP=/b/{n}.dump /b/bin/{kh.TARGET} test -b CPU "
                                  f"> /b/{n}.log 2>&1" for n, s in (("s1", 11), ("s1b", 11), ("s2", 12)))
                container(mode, [(bld, "/b", True)], ["bash", "-c", runs], arch=arch, tag="cpuh-seed")
                dumps = {n: kh.read_dump((bld / f"{n}.dump").read_bytes()) if (bld / f"{n}.dump").exists() else [] for n in ("s1", "s1b", "s2")}
                status = kh.case_status((bld / "s1.log").read_text(errors="replace")) if (bld / "s1.log").exists() else {}
                res["seeded"] = seeding_verdict(dumps["s1"], dumps["s1b"], dumps["s2"], status, pool)
                if not res["seeded"]["ok"]:
                    res["seeded"]["s1_log"] = (bld / "s1.log").read_text(errors="replace")[-1500:] if (bld / "s1.log").exists() else "no log"
            else:
                res["seeded"] = {"ok": False, "why": "no case to run"}
    ms = json.loads((d / "milestones.json").read_text())
    if "a" in listed:
        a_cases = set(listed["a"]["test"])
        for m, mm in zip(spec["milestones"], ms, strict=True):
            mm["cases_added"] = m["cases_added"] = sorted(set(m["case_pool"]) - a_cases)
        (d / "milestones.json").write_text(json.dumps(ms, indent=1))
    res["pools"] = {m["id"]: {"pool": len(m.get("case_pool") or []), "added": len(m.get("cases_added") or []), "perf_pool": len(m.get("perf_cases") or [])}
                    for m in spec["milestones"]}
    res["ok"] = bool(res.get("build_b", {}).get("ok") and res.get("build_a", {}).get("ok") and (res.get("seeded") or {}).get("ok")
                     and all(v["pool"] > 0 and (v["perf_pool"] > 0 or "perf" not in m["checks"]) for m, v in zip(spec["milestones"], res["pools"].values(), strict=True)))
    if res["ok"]:
        (d / "kdspec.json").write_text(json.dumps(spec, indent=1))
    for side in ("a", "b"):
        rmtree(d / f"cpuh_src_{side}")
        rmtree(d / f"cpuh_{side}")
    done.write_text(json.dumps(res, indent=1))
    return res


def calibrate(d: Path, spec: dict, mode: str, arch: str) -> tuple[dict, dict]:
    ms = json.loads((d / "milestones.json").read_text())
    idx = json.loads((HERE / "index.json").read_text())
    entry = next(x for x in idx if x["id"] == spec["task_id"])
    gpu = gpu_name()
    cal: dict = {"gpu": gpu, "arch": arch, "at": time.strftime("%FT%TZ", time.gmtime())}
    spec = json.loads(json.dumps(spec))
    cal["pools"] = fill_pools(d, spec, mode, arch)
    if spec.get("harness_extra"):
        listed = {}
        for mm in ("test", "perf"):
            p = container(mode, [(d / "ref", "/k/ref", False)], ["bash", "-c", f"CRUCIBLE_LIST=1 /k/ref/bin/{kh.TARGET} {mm} -b CUDA0"], arch=arch, tag="list-extra")
            listed[mm] = kh.listed_cases(p.stdout)
        cal["extra_cases"] = adopt_extra_cases(spec, listed["test"], listed["perf"])
        for x in ms:
            x["cases_added"] = list(dict.fromkeys([*cal["extra_cases"].get(x["id"], []), *(x.get("cases_added") or [])]))
    cal["sanitizer_targets"] = target_sanitizers(d, spec, mode, arch)
    # S1 sweep: the start (every fault in) over every case of every pool on the fixed seeds; the calibration then grades the
    # cases the start fails plus the usual sample (so a fault that breaks a handful of the 10k flash-attention cases is seen)
    full = d / "kdspec.full.json"
    full.write_text(json.dumps(spec, indent=1))
    sweep = grade(d, d / "floor_no_op.patch", full, mode, arch, "cal-start-sweep", pool=True, params=("pool_cap=0", "pool_seeds=fixed"))
    cal["sweep"] = {m["id"]: {"cases": len(per_case(sweep, m["id"])), "start_fails": sum(1 for ok in per_case(sweep, m["id"]).values() if not ok)}
                    for m in spec["milestones"]}
    if check_status(sweep, "build:cuda") != "pass":
        cal.update(gradable=False, error="the start did not build on this host: " + json.dumps(check_detail(sweep, "build:cuda"))[-600:])
        return spec, cal
    from crucible.factory.lanes import kernel_debug_plan as kdp  # the bundled kit (on sys.path): the grader's own sampling
    for m in spec["milestones"]:
        failing = [c for c, ok in per_case(sweep, m["id"]).items() if not ok]
        sample = kdp.pool_sample(list(m.get("case_pool") or []), list(m.get("cases_added") or []))
        m["case_pool"] = list(dict.fromkeys([*failing, *sample]))
        m["cases_added"] = list(dict.fromkeys([*failing, *(m.get("cases_added") or [])]))  # first in every later sample
    pooled = d / "kdspec.pooled.json"
    pooled.write_text(json.dumps(spec, indent=1))
    b = grade(d, d / "oracle.patch", pooled, mode, arch, "cal-oracle", pool=True)
    cal["oracle_pool_infra"] = b["infra"]
    if check_status(b, "build:cuda") != "pass" or (b["infra"] or {}).get("status") == "masked":
        cal.update(gradable=False, error="the oracle did not build or was masked on this host: " + json.dumps(check_detail(b, "build:cuda"))[-600:])
        return spec, cal
    b_pass = {m["id"]: per_case(b, m["id"]) for m in spec["milestones"]}
    # B's own CPU reference wrong (2026-10-23: ik_llama.cpp's IQK CPU flash attention disagrees with ggml's semantics on
    # the 2024 op test's inputs, so B failed its own cases): the probe compares B's GPU, the CPU kernel and an exact
    # computation; where the first and last agree against the CPU, those cases switch to the exact reference
    fa_fail = sorted({c for mid, pc in b_pass.items() for c, ok in pc.items() if not ok and c.startswith("FLASH_ATTN_EXT(")})
    if fa_fail:
        fa_ok = sorted({c for pc in b_pass.values() for c, ok in pc.items() if ok and c.startswith("FLASH_ATTN_EXT(")})[:64]
        probe_spec = json.loads(json.dumps(spec))
        for m in probe_spec["milestones"]:
            m["case_pool"], m["cases_added"], m["checks"] = [*fa_fail, *fa_ok], [], ["golden"]
        probe_spec["milestones"] = probe_spec["milestones"][:1]
        pp = d / "kdspec.faprobe.json"
        pp.write_text(json.dumps(probe_spec, indent=1))
        pr = grade(d, d / "oracle.patch", pp, mode, arch, "cal-fa-probe", pool=True, params=("pool_cap=0", "fa_probe=1"))
        ev = check_detail(pr, f"{probe_spec['milestones'][0]['id']}:golden").get("fa_evidence") or {}
        choice = fa_exact_choice(ev)
        cal["fa_reference"] = {**choice, "probed": len(ev), "b_failed_fa_cases": len(fa_fail),
                               "examples": {c: ev[c] for c in list(choice["exact"])[:3]}, "infra": pr["infra"]}
        if choice["exact"]:
            spec["fa_exact_cases"] = choice["exact"]
            pooled.write_text(json.dumps(spec, indent=1))
            b = grade(d, d / "oracle.patch", pooled, mode, arch, "cal-oracle-x", pool=True)
            if check_status(b, "build:cuda") != "pass" or (b["infra"] or {}).get("status") == "masked":
                cal.update(gradable=False, error="the oracle was masked on its regrade against the exact reference")
                return spec, cal
            b_pass = {m["id"]: per_case(b, m["id"]) for m in spec["milestones"]}
    # S1 executed, first on the start: A' (every fault in) must fail some pool case or a sanitizer check; if it passes
    # everything on this GPU no fault shows here and no single revert is graded (recorded as the drop reason)
    a_grade = grade(d, d / "floor_no_op.patch", pooled, mode, arch, "cal-start", pool=True)
    a_fails = {m["id"]: sorted(c for c, ok in per_case(a_grade, m["id"]).items() if b_pass[m["id"]].get(c) and not ok) for m in spec["milestones"]}
    a_san = {f"{m['id']}:{t}": check_status(a_grade, f"{m['id']}:{t}") for m in spec["milestones"] for t in m["checks"] if t in ("racecheck", "synccheck", "memcheck")}
    cal["start_fails"] = {k: len(v) for k, v in a_fails.items()}
    cal["start_sanitizer"] = a_san
    # a perf start is correct by construction (its faults only cost time): the drop applies to correctness milestones
    if start_not_real(spec, check_status(a_grade, "build:cuda") == "pass", a_fails, a_san):
        cal.update(gradable=False, kept_milestones=[], error="not real on this GPU: the start (every fault in) passes every pool case and every sanitizer check")
        return spec, cal
    owned, breaks, revert_grades = {}, {}, {}
    for mu_id, mid in entry["mutants"].items():
        if not mu_id.startswith("revert-"):
            continue
        r = grade(d, d / f"mutant_{mu_id}.patch", pooled, mode, arch, f"cal-{mu_id}", pool=True)
        revert_grades[mid] = r
        broke = set()
        for m in spec["milestones"]:
            pc = per_case_fixed(r, m["id"])
            broke |= {c for c, ok in pc.items() if b_pass[m["id"]].get(c) and not ok}
        breaks[mid] = broke
        added = next((x.get("cases_added") or [] for x in ms if x["id"] == mid), [])
        # owned on a fixed-seed break: it repeats in every later grade (2026-10-23: cases owned on a break seen only on
        # one fresh seed let the same revert pass its milestone at S8)
        owned[mid] = owned_cases(b_pass[mid], per_case_fixed(r, mid), added)
    # milestones whose revert does not apply at B: owned by what the fix alone repairs (A' fails, A' + the fix passes, B
    # passes); no revert means no executed edge into that milestone's fault (recorded)
    if entry.get("probers"):
        for pid, mid in entry["probers"].items():
            if mid in owned:
                continue
            al = grade(d, d / f"prober_{pid}.patch", pooled, mode, arch, f"cal-{pid}", pool=True)
            a_pc, al_pc = per_case_fixed(a_grade, mid), per_case(al, mid)
            added = next((x.get("cases_added") or [] for x in ms if x["id"] == mid), [])
            repaired = {c: not (b_pass[mid].get(c) and al_pc.get(c) and not a_pc.get(c, True)) for c in b_pass[mid]}
            owned[mid] = owned_cases(b_pass[mid], repaired, added)
            cal.setdefault("owned_by_fix_alone", []).append(mid)
    cal["owned"] = {k: len(v) for k, v in owned.items()}
    edges = executed_edges(owned, breaks)
    cal["executed_edges"] = edges
    new = json.loads(json.dumps(spec))
    keep = []
    for m in new["milestones"]:
        m["cases"] = owned.get(m["id"]) or []
        if not m["cases"] and m["kind"] != "perf":
            m["cases"] = [c for c in (b_pass.get(m["id"]) or {}) if b_pass[m["id"]][c]][: CASES_PER_MILESTONE]
            m["calibration_note"] = "no case breaks under the revert: kept only if a sanitizer tool catches it"
        m["requires"] = edges.get(m["id"], [])
        keep.append(m)
    new["milestones"] = keep
    cal_path = d / "kdspec.calibrating.json"
    cal_path.write_text(json.dumps(new, indent=1))
    # stability: the oracle graded three more times on the calibrated cases, each with its own fresh seed; a case B fails on
    # any of them is input-sensitive (a borderline tolerance) and leaves the milestone
    stab = [grade(d, d / "oracle.patch", cal_path, mode, arch, f"cal-oracle-stable{i}") for i in range(3)]
    cal["unstable_cases"] = {}
    for m in new["milestones"]:
        bad = {c for g in stab for c, ok in per_case(g, m["id"]).items() if not ok} | {c for g in stab for c in borderline(g, m["id"])}
        if bad:
            m["cases"] = [c for c in m["cases"] if c not in bad]
            cal["unstable_cases"][m["id"]] = sorted(bad)[:8]
    cal_path.write_text(json.dumps(new, indent=1))
    # sanitizer: clean on B, dirty on the revert; perf: noise floor and bar
    san, perf = {}, {}
    ob = grade(d, d / "oracle.patch", cal_path, mode, arch, "cal-oracle-full")
    for m in new["milestones"]:
        tools = [c for c in m["checks"] if c in ("racecheck", "synccheck", "memcheck")]
        rv = None
        if f"revert-{m['id']}" in entry["mutants"]:
            rv = grade(d, d / f"mutant_revert-{m['id']}.patch", cal_path, mode, arch, f"cal-revert-{m['id']}-full")
        for t in tools:
            clean = check_status(ob, f"{m['id']}:{t}") == "pass"
            dirty = rv is not None and check_status(rv, f"{m['id']}:{t}") == "fail"
            san[f"{m['id']}:{t}"] = {"clean_on_b": clean, "dirty_on_revert": dirty}
            if not (clean and dirty):
                m["checks"] = [c for c in m["checks"] if c != t]
        # a sanitizer tool keeps a milestone only when it discriminates (clean on B and dirty on the revert): before
        # 2026-10-23 a tool dirty on both kept it (held-out 44b81230 m1, 5b8c07ff m3), with nothing left to grade it
        if not owned.get(m["id"]) and not any(sanitizer_discriminates(san.get(f"{m['id']}:{t}") or {}) for t in tools) and m["kind"] != "perf":
            m["dropped"] = "not real on this GPU: its revert breaks no case and no sanitizer tool catches it"
        elif not m["cases"] and m["kind"] != "perf":
            m["dropped"] = "no stable case: B itself fails every owned case on some fresh seed"
        if "perf" in m["checks"]:
            perf[m["id"]] = calibrate_perf(d, new, m, entry, mode, arch, gpu)
            if perf[m["id"]]["cases"]:
                m["perf_cases"] = perf[m["id"]]["cases"]
                new.setdefault("perf_bars", {}).setdefault(gpu, {})[f"{m['id']}_ms"] = perf[m["id"]]["bar"]
            else:
                m["checks"] = [c for c in m["checks"] if c != "perf"]
                m["dropped"] = "no perf case shows the regression above the noise floor on this GPU"
    new["milestones"] = [m for m in new["milestones"] if not m.get("dropped")]
    for m in new["milestones"]:
        m["checks"] = [c for c in m["checks"]] or ["golden"]
        m["tests"] = [f"{m['id']}:{c}" for c in m["checks"]]
    cal["sanitizer"], cal["perf"] = san, perf
    cal["kept_milestones"] = [m["id"] for m in new["milestones"]]
    cal["gradable"] = len(new["milestones"]) >= 2
    return new, cal


def added_lines(patch: str) -> list[str]:
    """A patch's added lines with all whitespace removed (a formatting-only difference vanishes)."""
    return sorted("".join(ln[1:].split()) for ln in patch.splitlines() if ln.startswith("+") and not ln.startswith("+++"))


def calibrate_partials(d: Path, spec: dict, entry: dict, mode: str, arch: str) -> tuple[dict, dict]:
    """The half-fix mutants (B with one hunk of a fix reverted), graded over the pool after the milestones are calibrated:
    a half fix that breaks pool cases adds up to 8 of them to its milestone's cases (the grader must see it); one that
    differs from the oracle only in whitespace, or breaks no pool case, is equivalent on the repository's op tests and
    one that does not build tests nothing: both are recorded and leave S8 (they are not faults the tests can show)."""
    oracle = (d / "oracle.patch").read_text(errors="replace")
    rep = d / "grades" / "cal-oracle" / "report.json"
    b = {"checks": json.loads(rep.read_text()).get("checks") or []} if rep.exists() else {"checks": []}
    new, info = json.loads(json.dumps(spec)), {}
    ms = {m["id"]: m for m in new["milestones"]}
    for mu_id, mid in entry["mutants"].items():
        if not mu_id.startswith("partial-") or mid not in ms:
            continue
        patch = (d / f"mutant_{mu_id}.patch").read_text(errors="replace")
        if added_lines(patch) == added_lines(oracle):
            info[mu_id] = {"status": "equivalent", "why": "differs from the oracle only in whitespace"}
            continue
        r = grade(d, d / f"mutant_{mu_id}.patch", d / "kdspec.pooled.json", mode, arch, f"cal-{mu_id}", pool=True)
        if check_status(r, "build:cuda") != "pass":
            info[mu_id] = {"status": "uninformative", "why": "does not build for this GPU"}
            continue
        broke = [c for c, ok in per_case(r, mid).items() if per_case(b, mid).get(c) and not ok]
        san = [t for t in ms[mid]["checks"] if t in ("racecheck", "synccheck", "memcheck") and check_status(r, f"{mid}:{t}") == "fail"]
        if broke or san:
            ms[mid]["cases"] = list(dict.fromkeys([*broke[:8], *(ms[mid].get("cases") or [])]))
            info[mu_id] = {"status": "detected", "cases_added": len(broke[:8]), "sanitizer": san}
        else:
            info[mu_id] = {"status": "equivalent", "why": "breaks no pool case and no sanitizer check on this GPU"}
    return new, info


def perf_ratio(g: dict, cid: str, key: str) -> float | None:
    d = check_detail(g, cid)
    att = d if d.get("metrics") else (d.get("attempts") or [{}])[-1]  # a passing check reports its last attempt at the top
    v = ((att.get("metrics") or {}).get(key) or {}).get("ratio")
    return float(v) if v is not None else None


def calibrate_perf(d: Path, spec: dict, m: dict, entry: dict, mode: str, arch: str, gpu: str, *, keep: int = 3) -> dict:
    """Per perf candidate case (at most 8): B against itself twice (the noise floor) and the revert against B (the
    regression), with open bars; every case is timed on its own inside one grade (``perf_each``: three grades per
    milestone instead of three per case); cases where the revert is slower than ``perf_bar`` allows are kept (at most
    ``keep``); the milestone's bar is the largest of theirs."""
    from crucible.factory.lanes import kernel_debug_plan as kdp  # the bundled kit (on sys.path)

    key, cid = f"{m['id']}_ms", f"{m['id']}:perf"
    cases = list(m.get("perf_cases") or [])[:8]
    if not cases or f"revert-{m['id']}" not in entry["mutants"]:
        return {"cases": [], "bar": None, "rows": [{"error": "no perf case" if not cases else "no revert mutant to time against B"}]}
    s = json.loads(json.dumps(spec))
    next(x for x in s["milestones"] if x["id"] == m["id"])["perf_cases"] = cases
    s["perf_bars"] = {gpu: {k: 1e9 for k in kdp.perf_groups(key, cases, True)}}
    sp = d / "kdspec.perfcal.json"
    sp.write_text(json.dumps(s))
    each = ("perf_each=1",)
    idle()
    gs = [grade(d, d / "oracle.patch", sp, mode, arch, f"cal-perf-b{i}", only=cid, params=each) for i in (1, 2)]
    gs.append(grade(d, d / f"mutant_revert-{m['id']}.patch", sp, mode, arch, "cal-perf-rv", only=cid, params=each))
    rows, kept = [], []
    for i, case in enumerate(cases):
        b1, b2, rv = (perf_ratio(g, cid, f"{key}#{i}") for g in gs)
        if None in (b1, b2, rv):
            rows.append({"case": case, "error": "unmeasured", "infra": [(g.get("infra") or {}).get("reason") for g in gs]})
            continue
        v = perf_bar([1.0, b1, b2], rv)
        rows.append({"case": case, "b_vs_b": [round(b1, 3), round(b2, 3)], **v})
        if v["ok"]:
            kept.append((case, v["bar"]))
    kept = kept[:keep]
    return {"cases": [c for c, _ in kept], "bar": max((b for _, b in kept), default=None), "rows": rows}


def flake_verdict(runs: list[dict]) -> dict:
    """S10 over the oracle's repeated grades: a masked grade (infra: a busy node, an uncalibrated bar) is no evidence
    either way and leaves the denominator; the Clopper-Pearson upper bound is over the grades that measured (before
    2026-10-22 masked grades counted as passes)."""
    masked = sum(1 for r in runs if (r["infra"] or {}).get("status") == "masked")
    bad = sum(1 for r in runs if r["reward"] != 1.0 and (r["infra"] or {}).get("status") != "masked")
    n = len(runs) - masked
    return {"n": len(runs), "measured": n, "false_negatives": bad, "masked": masked, "cp_upper": round(cp_upper(bad, n), 4)}


def load_spec(d: Path) -> dict:
    spec = json.loads((d / "kdspec.json").read_text())
    if (d / "harness_extra.json").exists():  # grader-owned cases for faults the repository's cases never reach here
        spec["harness_extra"] = json.loads((d / "harness_extra.json").read_text())
    return spec


def task(tid: str, a: argparse.Namespace) -> dict:
    d = HERE / "tasks" / tid
    unpack(d)
    spec = load_spec(d)
    res: dict = {"id": tid, "arch": a.arch, "gpu": gpu_name(), "at": time.strftime("%FT%TZ", time.gmtime()),
                 "runtime": {"mode": a.mode, **({"root": APPTAINER_ROOT, "trusted": APPTAINER_ROOT == "fakeroot"} if a.mode == "apptainer" else {"trusted": True})}}
    entry = next(x for x in json.loads((HERE / "index.json").read_text()) if x["id"] == tid)
    if entry.get("cpu_pending_host"):
        res["cpu_host"] = host_cpu_gates(d, spec, a.mode, a.arch)
        if not res["cpu_host"]["ok"]:
            return res
        spec = load_spec(d)
    res["reference"] = build_reference(d, a.mode, a.arch, spec)
    if not res["reference"]["ok"]:
        return res
    calp = d / "kdspec.calibrated.json"
    if a.recalibrate or not calp.exists():
        # only timed steps need a quiet GPU: a correctness task's calibration can share the host with another one
        res["idle"] = idle() if any(m["kind"] == "perf" for m in spec["milestones"]) else {"skipped": "no timed step"}
        new, cal = calibrate(d, spec, a.mode, a.arch)
        calp.write_text(json.dumps(new, indent=1))
        (d / "calibration.json").write_text(json.dumps(cal, indent=1))
        res["calibration"] = cal
    else:
        res["calibration"] = json.loads((d / "calibration.json").read_text())
    if not res["calibration"].get("gradable"):
        return res
    if "partials" not in res["calibration"]:
        new, info = calibrate_partials(d, json.loads(calp.read_text()), entry, a.mode, a.arch)
        calp.write_text(json.dumps(new, indent=1))
        res["calibration"]["partials"] = info
        (d / "calibration.json").write_text(json.dumps(res["calibration"], indent=1))
    res["oracle"] = grade(d, d / "oracle.patch", calp, a.mode, a.arch, "oracle")
    res["floors"] = {p.stem: grade(d, p, calp, a.mode, a.arch, p.stem) for p in sorted(d.glob("floor_*.patch"))}
    res["mutants"] = {p.stem: grade(d, p, calp, a.mode, a.arch, p.stem) for p in sorted(d.glob("mutant_*.patch"))}
    idx = {x["id"]: x for x in json.loads((HERE / "index.json").read_text())}[tid]
    res["S5_oracle"] = res["oracle"]["reward"] == 1.0
    res["S8_floors"] = all(f["reward"] == 0.0 for f in res["floors"].values())
    # a mutant counts when it builds and its milestone's gate is not passed; half fixes the calibration found equivalent
    # on the op tests (or unbuildable) are listed apart, never counted as caught
    skip = {f"mutant_{k}": v for k, v in (res["calibration"].get("partials") or {}).items() if v["status"] != "detected"}
    res["S8_mutants_excluded"] = skip
    res["S8_mutants_fail_their_milestone"] = {k: check_status(v, "build:cuda") == "pass" and (v["gates"] or {}).get(idx["mutants"].get(k.removeprefix("mutant_"), "")) != "pass"
                                              for k, v in res["mutants"].items() if k not in skip}
    if a.flake and res["S5_oracle"] and res["S8_floors"] and all(res["S8_mutants_fail_their_milestone"].values()):
        # S10 only for a task that passed S5 and S8 (sixty grades of a task already dropped measure nothing)
        # a perf task's grades time kernels: four at once would load the GPU its own load guard waits on, so they run one
        # at a time (correctness tasks: half at concurrency 1, half at 4)
        perf_task = any("perf" in m["checks"] for m in json.loads(calp.read_text())["milestones"])
        half = a.flake if perf_task else a.flake // 2
        runs = [grade(d, d / "oracle.patch", calp, a.mode, a.arch, f"flake1-{i}") for i in range(half)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            runs += list(ex.map(lambda i: grade(d, d / "oracle.patch", calp, a.mode, a.arch, f"flake4-{i}"), range(a.flake - half)))
        res["flake"] = flake_verdict(runs)
    res["S10"] = (res.get("flake") or {}).get("cp_upper", 1.0) <= 0.05
    for g in ("oracle",):
        for c in res[g].get("checks") or []:
            c.pop("detail", None)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--mode", default="docker", choices=("docker", "apptainer"))
    ap.add_argument("--flake", type=int, default=60)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--no-ccache", action="store_true", help="build every grade from scratch (the gate runs cache by default)")
    ap.add_argument("--build-sif", action="store_true", help="apptainer: build the SIF from Dockerfile when it is missing")
    a = ap.parse_args()
    global USE_CCACHE
    USE_CCACHE = not a.no_ccache
    ensure_image(a.mode, build=a.build_sif)
    build_launch_lib(a.mode, a.arch)
    (HERE / "results").mkdir(exist_ok=True)
    for t in json.loads((HERE / "index.json").read_text()):
        if a.only and t["id"] not in a.only:
            continue
        out = HERE / "results" / f"{t['id']}.json"
        if out.exists() and not a.recalibrate:
            continue
        try:
            r = task(t["id"], a)
        except Exception as e:  # noqa: BLE001 - one task's failure is recorded, the run goes on
            r = {"id": t["id"], "error": f"{type(e).__name__}: {e}"[:800]}
        out.write_text(json.dumps(r, indent=1, sort_keys=True, default=str))
        print(t["id"], "oracle", (r.get("oracle") or {}).get("reward"), "floors", r.get("S8_floors"), "S10", r.get("S10"), r.get("error", ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
