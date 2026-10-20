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
``--runtime=nvidia`` (the lab 5090); Apptainer (Marlowe) runs the same image with ``--nv --fakeroot`` (the grader drops
privileges for candidate code, so it must be root inside).
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
from crucible.factory.lanes import kernel_harness as kh
from crucible.factory.lanes import kernel_plan as kp

IMAGE = "crucible-lh-k:base-12.8.1"
PLAN = "crucible.factory.lanes.kernel_debug_plan"
CASES_PER_MILESTONE = 32
PERF_HEADROOM = 0.10
CCACHE = HERE / "ccache"  # gate runs only (the grader's own patches): a task grade never mounts a build cache
USE_CCACHE = True


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
    if p.exists():
        sh(["docker", "run", "--rm", "--name", f"crucible-lh-k-gpu-rm-{uuid.uuid4().hex[:8]}", "-v", f"{p.parent}:/x", IMAGE, "rm", "-rf", f"/x/{p.name}"], timeout=1800)
    if p.exists():
        raise RuntimeError(f"could not remove {p}")


def gpu_name() -> str:
    return sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=30).stdout.strip().splitlines()[0]


def idle(max_util: int = 10, max_load: float = 8.0, wait_s: float = 3600) -> dict:
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


def ensure_image(mode: str) -> None:
    if mode == "docker" and sh(["docker", "image", "inspect", IMAGE], timeout=60).returncode != 0:
        p = sh(["docker", "build", "-t", IMAGE, "-"], input=(HERE / "Dockerfile").read_text())
        if p.returncode != 0:
            raise RuntimeError(f"build {IMAGE}: {p.stderr[-800:]}")


def container(mode: str, mounts: list[tuple[Path, str, bool]], args: list[str], *, arch: str, tag: str, timeout: float = 14400) -> subprocess.CompletedProcess:
    if mode == "apptainer":
        cmd = ["apptainer", "exec", "--nv", "--fakeroot", "--containall", "--net", "--network", "none", "--env", f"CRUCIBLE_CUDA_ARCH={arch}"]
        for src, dst, rw in mounts:
            cmd += ["-B", f"{src}:{dst}" + ("" if rw else ":ro")]
        cmd += [f"docker-daemon://{IMAGE}", *args]
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
    if (ref / "bin" / kh.TARGET).exists() and (ref / "arch").exists() and (ref / "arch").read_text() == arch:
        return {"ok": True, "cached": True}
    if ref.exists():
        rmtree(ref)
    src = d / "ref_src"
    if src.exists():
        rmtree(src)
    shutil.copytree(d / "b", src, symlinks=True)
    text = (src / kh.HARNESS).read_text(errors="replace")
    patched = kh.patch_harness(text, need_perf=any("perf" in m["checks"] for m in spec["milestones"]), header=kh.ggml_header(src))
    kh.prepare_tree(src, patched)
    ref.mkdir()
    (ref / "harness.cpp").write_text(patched)
    jobs = str(max(2, min(32, os.cpu_count() or 4)))
    flags = [*kh.cmake_cuda([arch]), "-DBUILD_SHARED_LIBS=OFF", *(spec.get("cmake_extra") or []), *(kh.cmake_ccache() if USE_CCACHE else ())]  # static, as the grader builds
    script = ("export CCACHE_DIR=/ccache CCACHE_BASEDIR=/ CCACHE_NOHASHDIR=1 CCACHE_UMASK=000 CCACHE_MAXSIZE=15G; "
              f"cmake -S /src -B /ref/build -G Ninja {' '.join(flags)} > /ref/configure.log 2>&1 && "
              f"cmake --build /ref/build --target {kh.TARGET} -j {jobs} > /ref/build.log 2>&1; rc=$?; "
              "mkdir -p /ref/bin && cp -a /ref/build/bin/. /ref/bin/ ; exit $rc")
    p = container(mode, [(src, "/src", False), (ref, "/ref", True), (ccache_dir(), "/ccache", True)], ["bash", "-c", script], arch=arch, tag="ref")
    ok = p.returncode == 0 and (ref / "bin" / kh.TARGET).exists()
    if ok:
        (ref / "arch").write_text(arch)
        rmtree(ref / "build")
        rmtree(src)
    os.chmod(ref, 0o700)
    return {"ok": ok, "rc": p.returncode, "log": "" if ok else ((ref / "build.log").read_text(errors="replace")[-1500:] if (ref / "build.log").exists() else p.stderr[-800:])}


def grade(d: Path, patch: Path, spec_path: Path, mode: str, arch: str, tag: str, *, pool: bool = False, only: str = "") -> dict:
    out = d / "grades" / tag
    import hashlib

    key = hashlib.sha256(b"\0".join([patch.read_bytes(), spec_path.read_bytes(), tag.encode(), str(pool).encode(), only.encode()])).hexdigest()
    if pool and (out / "inputs.sha256").exists() and (out / "inputs.sha256").read_text() == key and (out / "report.json").exists():
        report = json.loads((out / "report.json").read_text())  # a pool grade of the same patch, spec and grader: reused (calibration restarts)
        doc = json.loads((out / "crucible_reward.json").read_text())
        return {"reward": doc.get("reward"), "infra": doc.get("infra"), "gates": report.get("gates") or {}, "checks": report.get("checks") or []}
    if out.exists():
        rmtree(out)
    out.mkdir(parents=True)
    args = ["python3", "-I", "/crucible_verify/crucible/verify/runtime/run.py", "--plan", PLAN, "--out", "/out", "--base", "/k/base", "--patch", "/k/patch",
            "--param", "spec=/k/kdspec.json", "--param", f"arch={arch}", "--param", "base_tree=/k/base", "--param", "ref=/k/ref", "--param", "hooks_dir=/k/hooks",
            "--require-root"] + (["--param", "mode=pool"] if pool else []) + (["--param", f"only={only}"] if only else [])
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
    report = json.loads(rep.read_text()) if rep.exists() else {}
    return {"reward": doc.get("reward"), "infra": doc.get("infra"), "gates": report.get("gates") or {}, "checks": report.get("checks") or []}


def check_detail(g: dict, cid: str) -> dict:
    return next((c.get("detail") or {} for c in g.get("checks") or [] if c.get("id") == cid), {})


def check_status(g: dict, cid: str) -> str | None:
    return next((c.get("status") for c in g.get("checks") or [] if c.get("id") == cid), None)


def per_case(g: dict, mid: str) -> dict[str, bool]:
    return dict(check_detail(g, f"{mid}:golden").get("per_case") or {})


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


def fill_pools(d: Path, spec: dict, mode: str, arch: str) -> dict:
    """Case pools the CPU gates could not list (a repository whose CPU backend needs AVX2): the reference build lists
    every case of its test and perf modes (``CRUCIBLE_LIST``, nothing computed), filtered by each milestone's ops."""
    need = [m for m in spec["milestones"] if not m.get("case_pool") or ("perf" in m["checks"] and not m.get("perf_cases"))]
    if not need:
        return {"filled": 0}
    listed = {}
    for mmode in ("test", "perf"):
        p = container(mode, [(d / "ref", "/k/ref", False)], ["bash", "-c", f"CRUCIBLE_LIST=1 /k/ref/bin/{kh.TARGET} {mmode} -b CUDA0"], arch=arch, tag=f"list-{mmode}")
        listed[mmode] = kh.listed_cases(p.stdout)
    for m in need:
        ops = set(m["ops"])
        if not m.get("case_pool"):
            m["case_pool"] = [c for c in listed["test"] if c.split("(", 1)[0] in ops]
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
        script = ("export CCACHE_DIR=/ccache CCACHE_BASEDIR=/ CCACHE_NOHASHDIR=1 CCACHE_UMASK=000 CCACHE_MAXSIZE=15G; "
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
    pooled = d / "kdspec.pooled.json"
    pooled.write_text(json.dumps(spec, indent=1))
    b = grade(d, d / "oracle.patch", pooled, mode, arch, "cal-oracle", pool=True)
    cal["oracle_pool_infra"] = b["infra"]
    if check_status(b, "build:cuda") != "pass" or (b["infra"] or {}).get("status") == "masked":
        cal.update(gradable=False, error="the oracle did not build or was masked on this host: " + json.dumps(check_detail(b, "build:cuda"))[-600:])
        return spec, cal
    b_pass = {m["id"]: per_case(b, m["id"]) for m in spec["milestones"]}
    # S1 executed, first on the start: A' (every fault in) must fail some pool case or a sanitizer check; if it passes
    # everything on this GPU no fault shows here and no single revert is graded (recorded as the drop reason)
    a_grade = grade(d, d / "floor_no_op.patch", pooled, mode, arch, "cal-start", pool=True)
    a_fails = {m["id"]: sorted(c for c, ok in per_case(a_grade, m["id"]).items() if b_pass[m["id"]].get(c) and not ok) for m in spec["milestones"]}
    a_san = {f"{m['id']}:{t}": check_status(a_grade, f"{m['id']}:{t}") for m in spec["milestones"] for t in m["checks"] if t in ("racecheck", "synccheck", "memcheck")}
    cal["start_fails"] = {k: len(v) for k, v in a_fails.items()}
    cal["start_sanitizer"] = a_san
    if check_status(a_grade, "build:cuda") == "pass" and not any(a_fails.values()) and all(v == "pass" for v in a_san.values()):
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
            pc = per_case(r, m["id"])
            broke |= {c for c, ok in pc.items() if b_pass[m["id"]].get(c) and not ok}
        breaks[mid] = broke
        added = next((x.get("cases_added") or [] for x in ms if x["id"] == mid), [])
        owned[mid] = owned_cases(b_pass[mid], per_case(r, mid), added)
    # milestones whose revert does not apply at B: owned by what the fix alone repairs (A' fails, A' + the fix passes, B
    # passes); no revert means no executed edge into that milestone's fault (recorded)
    if entry.get("probers"):
        for pid, mid in entry["probers"].items():
            if mid in owned:
                continue
            al = grade(d, d / f"prober_{pid}.patch", pooled, mode, arch, f"cal-{pid}", pool=True)
            a_pc, al_pc = per_case(a_grade, mid), per_case(al, mid)
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
        bad = {c for g in stab for c, ok in per_case(g, m["id"]).items() if not ok}
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
        if not owned.get(m["id"]) and not any(san.get(f"{m['id']}:{t}", {}).get("dirty_on_revert") for t in tools) and m["kind"] != "perf":
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
    att = (check_detail(g, cid).get("attempts") or [{}])[-1]
    v = ((att.get("metrics") or {}).get(key) or {}).get("ratio")
    return float(v) if v is not None else None


def calibrate_perf(d: Path, spec: dict, m: dict, entry: dict, mode: str, arch: str, gpu: str, *, keep: int = 3) -> dict:
    """Per perf candidate case: B against itself twice (the noise floor) and the revert against B (the regression), with
    an open bar; cases where the revert is slower than ``perf_bar`` allows are kept (at most ``keep``); the milestone's
    bar is the largest of theirs."""
    key, cid = f"{m['id']}_ms", f"{m['id']}:perf"
    rows, kept = [], []
    for case in list(m.get("perf_cases") or [])[:8]:
        s = json.loads(json.dumps(spec))
        mm = next(x for x in s["milestones"] if x["id"] == m["id"])
        mm["perf_cases"] = [case]
        s["perf_bars"] = {gpu: {key: 1e9}}
        sp = d / "kdspec.perfcal.json"
        sp.write_text(json.dumps(s))
        idle()
        b1 = perf_ratio(grade(d, d / "oracle.patch", sp, mode, arch, "cal-perf-b1", only=cid), cid, key)
        b2 = perf_ratio(grade(d, d / "oracle.patch", sp, mode, arch, "cal-perf-b2", only=cid), cid, key)
        rv = None
        if f"revert-{m['id']}" in entry["mutants"]:
            rv = perf_ratio(grade(d, d / f"mutant_revert-{m['id']}.patch", sp, mode, arch, "cal-perf-rv", only=cid), cid, key)
        if None in (b1, b2, rv):
            rows.append({"case": case, "error": "unmeasured"})
            continue
        v = perf_bar([1.0, b1, b2], rv)
        rows.append({"case": case, **v})
        if v["ok"]:
            kept.append((case, v["bar"]))
    kept = kept[:keep]
    return {"cases": [c for c, _ in kept], "bar": max((b for _, b in kept), default=None), "rows": rows}


def task(tid: str, a: argparse.Namespace) -> dict:
    d = HERE / "tasks" / tid
    unpack(d)
    spec = json.loads((d / "kdspec.json").read_text())
    res: dict = {"id": tid, "arch": a.arch, "gpu": gpu_name(), "at": time.strftime("%FT%TZ", time.gmtime())}
    entry = next(x for x in json.loads((HERE / "index.json").read_text()) if x["id"] == tid)
    if entry.get("cpu_pending_host"):
        res["cpu_host"] = host_cpu_gates(d, spec, a.mode, a.arch)
        if not res["cpu_host"]["ok"]:
            return res
        spec = json.loads((d / "kdspec.json").read_text())
    res["reference"] = build_reference(d, a.mode, a.arch, spec)
    if not res["reference"]["ok"]:
        return res
    calp = d / "kdspec.calibrated.json"
    if a.recalibrate or not calp.exists():
        res["idle"] = idle()
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
    if a.flake:
        half = a.flake // 2
        runs = [grade(d, d / "oracle.patch", calp, a.mode, a.arch, f"flake1-{i}") for i in range(half)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            runs += list(ex.map(lambda i: grade(d, d / "oracle.patch", calp, a.mode, a.arch, f"flake4-{i}"), range(a.flake - half)))
        bad = sum(1 for r in runs if r["reward"] != 1.0 and (r["infra"] or {}).get("status") != "masked")
        res["flake"] = {"n": len(runs), "false_negatives": bad, "masked": sum(1 for r in runs if (r["infra"] or {}).get("status") == "masked"),
                        "cp_upper": round(cp_upper(bad, len(runs)), 4)}
    idx = {x["id"]: x for x in json.loads((HERE / "index.json").read_text())}[tid]
    res["S5_oracle"] = res["oracle"]["reward"] == 1.0
    res["S8_floors"] = all(f["reward"] == 0.0 for f in res["floors"].values())
    # a mutant counts when it builds and its milestone's gate is not passed; half fixes the calibration found equivalent
    # on the op tests (or unbuildable) are listed apart, never counted as caught
    skip = {f"mutant_{k}": v for k, v in (res["calibration"].get("partials") or {}).items() if v["status"] != "detected"}
    res["S8_mutants_excluded"] = skip
    res["S8_mutants_fail_their_milestone"] = {k: check_status(v, "build:cuda") == "pass" and (v["gates"] or {}).get(idx["mutants"].get(k.removeprefix("mutant_"), "")) != "pass"
                                              for k, v in res["mutants"].items() if k not in skip}
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
    a = ap.parse_args()
    global USE_CCACHE
    USE_CCACHE = not a.no_ccache
    ensure_image(a.mode)
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
