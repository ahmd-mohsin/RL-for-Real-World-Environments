"""Kernel debugging lane, the World and the CPU gates that need a compiler (swarm-pipeline/10 §5). Docker only, no GPU.

Everything here runs the grader's own commands in throwaway ``crucible-lh-k-*`` containers with ``--network none``, at
most two CPUs each (the machine is shared): the repository's CMake configures the CUDA build for the grading arches
(``90-real;120-real``: SASS only, no PTX), each translation unit a fault touches is compiled with the exact command
CMake generated (``compile_commands.json``), and the grader-owned harness (``kernel_harness``) is built against the
CPU backend and run with ``CRUCIBLE_ALLOW_CPU``: the case listing, the seeded inputs and the dump format are executed
without a GPU. Images are ``crucible-lh-k:*``; only containers this module names are ever removed.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.lanes import kernel_harness as kh
from crucible.factory.lanes import kernel_world as kw

IMAGE = "crucible-lh-k:base-12.8.1"  # the kernel lane's CUDA 12.8 devel image (crucible-k-base), under this lane's tag
PREFIX = "crucible-lh-k-"
ARCHES = ("90", "120")  # Marlowe H100, lab RTX 5090
CMAKE_CPU = kh.CMAKE_CPU
cuda_arch_flag = kh.cuda_arch_flag


def ensure_image() -> str:
    """The lane's image id: ``crucible-lh-k:base-12.8.1``, tagged from the kernel lane's base image (one set of layers),
    built from its Dockerfile when neither exists."""
    if (iid := kw.image_id(IMAGE)):
        return iid
    if not kw.image_id(kw.BASE_TAG):
        kw.ensure_image(kw.BASE_TAG, kw.BASE_FROM)
    p = kw.docker("tag", kw.BASE_TAG, IMAGE, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"tag {IMAGE}: {p.stderr[-300:]}")
    return kw.image_id(IMAGE) or ""


@dataclass
class Run:
    rc: int
    stdout: str
    seconds: float
    timed_out: bool = False


def docker_run(script: str, mounts: Sequence[tuple[Path | str, str]], *, tag: str, timeout: float = 7200, cpus: str = "2", memory: str = "3800m",
               env: Mapping[str, str] | None = None, image: str = IMAGE) -> Run:
    """One throwaway container (``crucible-lh-k-<tag>-<id>``, no network) running ``bash -c script``."""
    name = f"{PREFIX}{tag}-{uuid.uuid4().hex[:8]}"
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", cpus, "--memory", memory, "--label", "crucible.lane=kernel-debug"]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    for a, b in mounts:
        cmd += ["-v", f"{Path(a).resolve() if not str(a).startswith('/') else a}:{b}"]
    cmd += [image, "bash", "-c", script]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, errors="replace")
    except subprocess.TimeoutExpired:
        kw.docker("rm", "-f", name, timeout=120)
        return Run(124, "", round(time.time() - t0, 1), True)
    return Run(p.returncode, _strip_banner(p.stdout + p.stderr), round(time.time() - t0, 1))


def _strip_banner(text: str) -> str:
    """The CUDA image's start-up banner removed from a container's output."""
    banner = ("== CUDA ==", "CUDA Version", "Container image Copyright", "governed by the NVIDIA", "By pulling and using the container",
              "NGC-DL-CONTAINER-LICENSE", "deep-learning-container-license", "NVIDIA Driver was not detected", "Use the NVIDIA Container Toolkit",
              "docs.nvidia.com/datacenter/cloud-native")
    keep = [ln for ln in text.splitlines() if ln.strip() and set(ln.strip()) != {"="} and not any(s in ln for s in banner)]
    return "\n".join(keep)


def own_containers() -> list[str]:
    p = kw.docker("ps", "-a", "--filter", f"name={PREFIX}", "--format", "{{.Names}}", timeout=60)
    return [n for n in p.stdout.split() if n.startswith(PREFIX)]


def cleanup() -> int:
    """Remove the containers this lane created (by name prefix) and nothing else."""
    names = own_containers()
    if names:
        kw.docker("rm", "-f", *names, timeout=300)
    return len(names)


# ---- the CUDA build: configure, then compile single units ------------------------------------------------------------------------
def configure(tree: Path, bld: Path, *, arches: Sequence[str] = ARCHES, timeout: float = 1800) -> dict[str, Any]:
    """CMake configure of ``tree`` for the grading arches into ``bld`` (kept: later unit compiles reuse its
    ``compile_commands.json``). The tree is mounted read-only at ``/src``."""
    bld.mkdir(parents=True, exist_ok=True)
    cc = bld / "compile_commands.json"
    if cc.exists() and (bld / "crucible_arches").exists() and (bld / "crucible_arches").read_text() == cuda_arch_flag(arches):
        return {"ok": True, "cached": True}
    args = " ".join(shlex.quote(a) for a in kh.cmake_cuda(arches))
    r = docker_run(f"cmake -S /src -B /b -G Ninja {args} > /b/configure.log 2>&1; rc=$?; tail -25 /b/configure.log; exit $rc", [(tree, "/src:ro"), (bld, "/b")],
                   tag="cfg", timeout=timeout)
    ok = r.rc == 0 and cc.exists()
    if ok:
        (bld / "crucible_arches").write_text(cuda_arch_flag(arches))
    return {"ok": ok, "rc": r.rc, "seconds": r.seconds, "log": r.stdout[-2000:] if not ok else ""}


def unit_commands(bld: Path) -> dict[str, dict[str, str]]:
    """``repo-relative source -> {command, directory}`` from the configured build (paths under ``/src``)."""
    out = {}
    for e in json.loads((bld / "compile_commands.json").read_text()):
        f = str(e["file"])
        if f.startswith("/src/"):
            out[f[len("/src/"):]] = {"command": e.get("command") or " ".join(shlex.quote(a) for a in e.get("arguments") or []), "directory": e["directory"]}
    return out


def unit_command(cmd: str, i: int, *, arch: str | None = None) -> str:
    """CMake's compile command for one unit with its object sent to the container's /tmp and its dependency-file flags
    dropped (``-MD``, ``-MT x``, ``-MF x``), so a unit compile never writes into the shared build directory. ``arch``:
    keep only that architecture's ``--generate-code`` (a mutant's build check needs one arch, not both)."""
    parts = shlex.split(cmd)
    if arch is not None:
        parts = [a for a in parts if not (a.startswith("--generate-code=") and not a.startswith(f"--generate-code=arch=compute_{arch}"))]
    out: list[str] = []
    skip = False
    for j, a in enumerate(parts):
        if skip:
            skip = False
            continue
        if a in ("-MD", "-MMD"):
            continue
        if a in ("-MT", "-MF", "-MQ"):
            skip = True
            continue
        if j > 0 and parts[j - 1] == "-o":
            a = f"/tmp/crucible_unit_{i}.o"
        out.append(a)
    return " ".join(shlex.quote(a) for a in out)


@dataclass
class UnitResult:
    ok: bool
    rc: int
    seconds: float
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "rc": self.rc, "seconds": self.seconds, "errors": self.errors}


def compile_units(tree: Path, bld: Path, units: Sequence[str], *, timeout: float = 7200, arch: str | None = None, retry: bool = True) -> dict[str, UnitResult]:
    """Compile each of ``units`` (repo-relative ``.cu``/``.cpp``) from ``tree`` with the command CMake generated in ``bld``
    (both grading arches in one invocation), objects to the container's /tmp. One container, units in sequence."""
    cmds = unit_commands(bld)
    lines = ["set +e", "mkdir -p /tmp/u"]
    known = [u for u in units if u in cmds]
    for i, u in enumerate(known):
        c = cmds[u]
        q = unit_command(c["command"], i, arch=arch)
        lines.append(f"t0=$(date +%s); ( cd {shlex.quote(c['directory'])} && {q} ) > /tmp/u/{i}.log 2>&1; echo $? $(( $(date +%s) - t0 )) > /tmp/u/{i}.rc")
    lines.append(f"for i in $(seq 0 {max(len(known) - 1, 0)}); do echo \"@@UNIT $i $(cat /tmp/u/$i.rc 2>/dev/null)\"; grep -E ' error|error:' /tmp/u/$i.log | head -12; done")
    out: dict[str, UnitResult] = {u: UnitResult(False, 127, 0.0, ["no compile command for this unit in the configured build"]) for u in units if u not in cmds}
    if not known:
        return out
    r = docker_run("\n".join(lines), [(tree, "/src:ro"), (bld, "/b")], tag="cc", timeout=timeout)
    cur: int | None = None
    res: dict[int, UnitResult] = {}
    for ln in r.stdout.splitlines():
        if ln.startswith("@@UNIT "):
            bits = ln.split()
            cur = int(bits[1])
            rc = int(bits[2]) if len(bits) > 2 and bits[2].lstrip("-").isdigit() else 1
            secs = float(bits[3]) if len(bits) > 3 and bits[3].isdigit() else 0.0
            res[cur] = UnitResult(rc == 0, rc, secs, [])
        elif cur is not None and cur in res:
            res[cur].errors.append(ln.strip()[:240])
    for i, u in enumerate(known):
        out[u] = res.get(i) or UnitResult(False, 1, 0.0, ["unit did not report (container failed)" + (" (timed out)" if r.timed_out else "")])
    killed = [u for u in known if out[u].rc in (137, -9)]
    if killed and retry:  # the shared machine's memory, not the code: once more, alone
        out.update(compile_units(tree, bld, killed, timeout=timeout, arch=arch, retry=False))
    for u in known:
        if out[u].rc in (137, -9):
            out[u].errors = ["killed (out of memory): infrastructure, not a compile failure", *out[u].errors]
    return out


# ---- the harness on the CPU backend -------------------------------------------------------------------------------------------------
# the 2024 layout's own CPU kernels (iqk) are x86/ARM-specific and do not build on this machine's aarch64 Linux: the CPU
# gate builds that layout with its portable kernels; the GPU hosts' reference build keeps the repository's defaults
CPU_GATE_EXTRA = {"ggml/src/ggml-cuda.cu": ("-DGGML_IQK_MUL_MAT=OFF", "-DGGML_IQK_FLASH_ATTENTION=OFF")}


def cpu_harness_build(tree: Path, bld: Path, harness_text: str, *, timeout: float = 5400, extra: Sequence[str] = ()) -> dict[str, Any]:
    """Build ``test-backend-ops`` with the CPU backend only from ``tree`` with ``harness_text`` (the grader's patched
    harness) in place of the tree's own file. The tree is copied (the harness file is replaced in the copy)."""
    src = bld / "src"
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(tree, src, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    kh.prepare_tree(src, harness_text)
    b = bld / "b"
    b.mkdir(parents=True, exist_ok=True)
    args = " ".join(shlex.quote(a) for a in (*CMAKE_CPU, *extra))
    r = docker_run(f"set -o pipefail; (cmake -S /src -B /b -G Ninja {args} && cmake --build /b --target test-backend-ops -j 2) > /b/crucible_build.log 2>&1; rc=$?; "
                   "tail -30 /b/crucible_build.log | grep -E 'error|FAILED' ; exit $rc", [(src, "/src:ro"), (b, "/b")], tag="cpuh", timeout=timeout)
    ok = r.rc == 0 and (b / "bin" / "test-backend-ops").exists()
    return {"ok": ok, "rc": r.rc, "seconds": r.seconds, "errors": r.stdout.splitlines()[-12:] if not ok else []}


def cpu_harness_runs(bld: Path, runs: Sequence[Mapping[str, Any]], *, timeout: float = 3600) -> list[dict[str, Any]]:
    """Run the CPU harness built by ``cpu_harness_build``: each run ``{"name", "seed", "ops": "NORM,..." | None,
    "cases": [...] | None, "dump": bool, "list": bool, "mode": "test" | "perf" | "support"}`` in one container; returns
    per run its exit code, case statuses (``kernel_harness.case_status``), the listed cases (``list``: every case of the
    mode's list, none run) and the dump path (host side) when asked."""
    out_dir = bld / "runs"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    lines = ["set +e"]
    for r in runs:
        n = str(r["name"])
        env = ["CRUCIBLE_ALLOW_CPU=1", f"CRUCIBLE_SEED={int(r.get('seed') or 0)}"]
        if r.get("list"):
            env.append("CRUCIBLE_LIST=1")
        if r.get("cases") is not None:
            (out_dir / f"{n}.cases").write_text(kh.cases_file(list(r["cases"])))
            env.append(f"CRUCIBLE_CASES=/o/{n}.cases")
        if r.get("dump"):
            env.append(f"CRUCIBLE_DUMP=/o/{n}.dump")
        args = [str(r.get("mode") or "test"), "-b", "CPU"]
        if r.get("ops"):
            args += ["-o", str(r["ops"])]
        lines.append(f"{' '.join(env)} timeout {int(r.get('timeout_s') or 1200)} /b/bin/test-backend-ops {' '.join(shlex.quote(a) for a in args)} "
                     f"> /o/{n}.log 2>&1; echo $? > /o/{n}.rc")
    r = docker_run("\n".join(lines), [(bld / "b", "/b:ro"), (out_dir, "/o")], tag="cpur", timeout=timeout)
    res = []
    for spec in runs:
        n = str(spec["name"])
        log = (out_dir / f"{n}.log").read_text(errors="replace") if (out_dir / f"{n}.log").exists() else ""
        rc_txt = (out_dir / f"{n}.rc").read_text().strip() if (out_dir / f"{n}.rc").exists() else ""
        res.append({"name": n, "rc": int(rc_txt) if rc_txt.lstrip("-").isdigit() else None, "cases": kh.case_status(log), "listed": kh.listed_cases(log),
                    "dump": str(out_dir / f"{n}.dump") if spec.get("dump") and (out_dir / f"{n}.dump").exists() else None, "log_tail": log[-800:],
                    "container_rc": r.rc})
    return res


def leak_probe(needles: Sequence[str], *, image: str = IMAGE, roots: Sequence[str] = ("/usr/local/cuda/", "/usr/share/doc", "/opt", "/root", "/home", "/usr/src",
                                                                                         "/usr/include", "/usr/local/include")) -> dict[str, Any]:
    """No file of the World image (outside the mounted workspace) holds a line only the oracle adds. Hits are printed
    behind a marker, so the image's start-up banner is never mistaken for one (``kernel_world.leak_probe`` read the
    banner as hits on the CUDA images)."""
    needles = [n.strip() for n in needles if len(n.strip()) >= 24][:40]
    if not needles:
        return {"ok": True, "probed": 0, "hits": []}
    pat = "\n".join(needles)
    roots_q = " ".join(shlex.quote(r) for r in roots)
    script = (f"cat > /tmp/n.txt <<'CRUCIBLE_EOF'\n{pat}\nCRUCIBLE_EOF\nfor r in {roots_q}; do [ -e \"$r\" ] && grep -rIlF -f /tmp/n.txt \"$r\" 2>/dev/null; done "
              "| head -20 | sed 's/^/CRUCIBLE_LEAK /'; echo CRUCIBLE_PROBE_DONE")
    r = docker_run(script, [], tag="leak", timeout=1800, image=image)
    hits = [ln[len("CRUCIBLE_LEAK "):] for ln in r.stdout.splitlines() if ln.startswith("CRUCIBLE_LEAK ")]
    done = "CRUCIBLE_PROBE_DONE" in r.stdout
    return {"ok": done and not hits, "probed": len(needles), "hits": hits, "completed": done}


# ---- the GPU grading backend (a CUDA host with the staged bundle) ---------------------------------------------------------------
class GpuBackend:
    """Grades one patch of a staged, calibrated task (``lh_kernel_debug.py stage-gpu`` + ``run_gpu.py``) with
    ``kernel_debug_plan`` in the verifier image on this host's GPU: the kit as root, the candidate unprivileged,
    ``--network none``, Docker ``--runtime=nvidia`` (Apptainer is the bundle runner's job). The task's calibrated spec
    and root-only reference build come from ``bundle/tasks/<id>``; a task not calibrated yet is masked
    (``uncalibrated``), never a 0. ``kernel_debug.KernelDebugTask(rec, GpuBackend(bundle))`` is the GPU environment."""

    def __init__(self, bundle: Path, *, arch: str = "90", gpu_args: Sequence[str] = ("--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all"),
                 timeout: float = 14400):
        self.bundle, self.arch, self.gpu_args, self.timeout = Path(bundle), str(arch), tuple(gpu_args), timeout

    def command(self, d: Path, spec: Path, patch: Path, out: Path, name: str) -> list[str]:
        mounts = [(self.bundle / "kit", "/crucible_verify:ro"), (d / "a", "/k/base:ro"), (d / "ref", "/k/ref:ro"), (self.bundle / "hooks", "/k/hooks:ro"),
                  (spec, "/k/kdspec.json:ro"), (patch, "/k/patch:ro"), (out, "/out")]
        cmd = ["docker", "run", "--rm", "--name", name, *self.gpu_args, "-e", f"CRUCIBLE_CUDA_ARCH={self.arch}", "--network", "none", "--init"]
        for a, b in mounts:
            cmd += ["-v", f"{a}:{b}"]
        return cmd + [IMAGE, "python3", "-I", "/crucible_verify/crucible/verify/runtime/run.py", "--plan", "crucible.factory.lanes.kernel_debug_plan",
                      "--out", "/out", "--base", "/k/base", "--patch", "/k/patch", "--param", "spec=/k/kdspec.json", "--param", f"arch={self.arch}",
                      "--param", "base_tree=/k/base", "--param", "ref=/k/ref", "--param", "hooks_dir=/k/hooks", "--require-root"]

    def run(self, rec: Mapping[str, Any], patch: str) -> dict[str, Any]:
        import tempfile

        d = self.bundle / "tasks" / rec["id"]
        spec = d / "kdspec.calibrated.json"
        if not spec.exists() or not (d / "ref" / "bin").exists():
            return {"infra": {"status": "masked", "failure_kind": "uncalibrated", "reason": f"{rec['id']} has no calibrated spec or reference on this host"}}
        with tempfile.TemporaryDirectory(prefix="kd-grade-") as td:
            t = Path(td)
            (t / "patch").write_text(patch)
            out = t / "out"
            out.mkdir()
            p = subprocess.run(self.command(d, spec, t / "patch", out, f"{PREFIX}gpu-grade-{uuid.uuid4().hex[:8]}"), capture_output=True, text=True,
                               timeout=self.timeout, check=False, errors="replace")
            rj, rep = out / "crucible_reward.json", out / "report.json"
            doc = json.loads(rj.read_text()) if rj.exists() else {"infra": {"status": "masked", "failure_kind": "no_reward", "reason": p.stderr[-400:]}}
            if rep.exists():
                doc["gate_status"] = json.loads(rep.read_text()).get("gates") or {}
            return doc

    async def grade(self, rec: Mapping[str, Any], patch: str) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self.run, rec, patch)
