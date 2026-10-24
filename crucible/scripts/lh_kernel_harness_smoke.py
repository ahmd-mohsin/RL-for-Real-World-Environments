"""Smoke of a kernel harness profile on a CUDA host (swarm-pipeline/10 §4.1): build B and A' (B with one real fix's source
reverted, B's tests kept: the grader owns them), list the cases, run every case on both, and show the fault changes a
verdict. Every build flag, target, command line and parser comes from the profile; this script only moves them to the
host. Our own build of a known fix (no candidate code), so configure may fetch the profile's test dependencies.

    uv run python scripts/lh_kernel_harness_smoke.py --host ahmed@100.100.101.36 --work crucible-kh --repo-dir cccl \
        --profile catch2-cccl --test cub/test/catch2_test_device_reduce_by_key.cu --b 6ad0be74 \
        --revert cub/cub/agent/agent_reduce_by_key.cuh --arch 120 --out data/kernel_debug/harness_smoke/cccl-6ad0be74.json

Host rules (the shared lab machine): ``--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all``, our containers are named
``crucible-kh-*`` and removed by name, nothing is pruned, and the run refuses to start while another worker's
``crucible-`` container is up or the disk has under 15 GB free.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crucible.factory.lanes import kernel_profiles as kprof

IMAGE = "crucible-lh-k:base-13.0.2"
MIN_FREE_GB = 15


def ssh(host: str, cmd: str, *, timeout: float = 7200) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-i", str(Path.home() / ".ssh" / "id_ed25519"), host, cmd],
                          capture_output=True, text=True, timeout=timeout, check=False)


def docker(work: str, name: str, script: str, *, gpu: bool, env: dict[str, str] | None = None) -> str:
    flags = ["--rm", "--name", name, "--user", "$(id -u):$(id -g)", "-e", "HOME=/w", "-v", f"$HOME/{work}:/w"]
    if gpu:
        flags += ["--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all"]
    for k, v in (env or {}).items():
        flags += ["-e", f"{k}={v}"]
    # the image's entrypoint prints a licence banner on stdout: bash is the entrypoint, so stdout is the program's own
    return f"docker rm -f {name} >/dev/null 2>&1; docker run {' '.join(flags)} --entrypoint /bin/bash {IMAGE} -c {shlex.quote(script)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--work", default="crucible-kh")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--revert", nargs="+", required=True, help="the fix's source paths taken from the fix's parent in A'")
    ap.add_argument("--arch", default="120")
    ap.add_argument("--cmake", default="/w/cmake/bin/cmake", help="the CMake the profile needs (in the work dir)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    H = kprof.profile(a.profile, tests=[a.test])
    t0 = time.time()

    pre = ssh(a.host, "docker ps --format '{{.Names}}' | grep '^crucible-' | grep -v '^crucible-kh-' ; df -BG --output=avail / | tail -1")
    lines = [x.strip() for x in pre.stdout.splitlines() if x.strip()]
    free = int(lines[-1].rstrip("G")) if lines and lines[-1].rstrip("G").isdigit() else 0
    if lines[:-1] or free < MIN_FREE_GB:
        sys.exit(f"host busy or full: other containers {lines[:-1]}, {free} GB free")

    w = a.work
    trees = (f"cd ~/{w}/{a.repo_dir} && git worktree remove -f ../b 2>/dev/null; git worktree remove -f ../a 2>/dev/null; "
             f"git worktree add -f --detach ../b {a.b} -q && git worktree add -f --detach ../a {a.b} -q && cd ../a && "
             f"git checkout -q {a.b}^ -- {' '.join(map(shlex.quote, a.revert))} && git -C ../a diff --cached --stat {a.b} | tail -1")
    r = ssh(a.host, trees)
    if r.returncode != 0:
        sys.exit(f"trees: {r.stderr[-800:]}")
    a_stat = r.stdout.strip()

    flags = " ".join(map(shlex.quote, H.cmake_cuda([a.arch])))
    out: dict = {"profile": H.describe(), "test": a.test, "b": a.b, "revert": a.revert, "a_vs_b": a_stat, "arch": a.arch, "image": IMAGE, "sides": {}}
    for side in ("b", "a"):
        build = (f"rm -rf /w/bld-{side} && mkdir -p /w/bld-{side} /w/cpm && {a.cmake} -S /w/{side} -B /w/bld-{side} -G Ninja {flags} > /w/bld-{side}/configure.log 2>&1 && "
                 f"{a.cmake} --build /w/bld-{side} --target {H.target} -j {a.jobs} > /w/bld-{side}/build.log 2>&1; echo rc=$?; tail -3 /w/bld-{side}/build.log")
        r = ssh(a.host, docker(w, f"crucible-kh-build-{side}", build, gpu=False, env={"CPM_SOURCE_CACHE": "/w/cpm"}), timeout=10800)
        ok = "rc=0" in r.stdout
        out["sides"][side] = {"build_ok": ok, "build_tail": r.stdout[-600:]}
        if not ok:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(out, indent=1))
            sys.exit(f"build {side} failed: {r.stdout[-800:]}")

    largv, lenv = H.list_argv(f"/w/bld-b/{H.exe}")
    r = ssh(a.host, docker(w, "crucible-kh-list", " ".join(map(shlex.quote, map(str, largv))), gpu=True, env=lenv))
    cases = H.listed(r.stdout)
    out["cases"] = len(cases)
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-i", str(Path.home() / ".ssh" / "id_ed25519"), a.host, f"cat > ~/{w}/cases.txt"],
                       input=H.cases_file(cases), text=True, capture_output=True, check=False)
    if p.returncode != 0:
        sys.exit(f"cases file: {p.stderr}")
    runs = {}
    for side in ("b", "a"):
        argv = H.argv(f"/w/bld-{side}/{H.exe}", "test", "CUDA0", "/w/cases.txt")
        r = ssh(a.host, docker(w, f"crucible-kh-run-{side}", " ".join(map(shlex.quote, map(str, argv))), gpu=True, env=H.env(0, "/w/cases.txt", None)))
        runs[side] = {"status": H.status(r.stdout), "rc": r.returncode, "crashed": H.crashed(r.stdout)}
    cmp = H.compare(runs["a"], runs["b"], cases)
    out["b_pass"] = sum(1 for v in runs["b"]["status"].values() if v == "OK")
    out["a_pass"] = sum(1 for v in runs["a"]["status"].values() if v == "OK")
    out["broken_by_revert"] = sorted(c for c, v in cmp.items() if not v["pass"] and "infra" not in v)
    out["infra"] = sorted(c for c, v in cmp.items() if "infra" in v)
    out["rc"] = {s: runs[s]["rc"] for s in runs}
    out["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "sides"}, indent=1))


if __name__ == "__main__":
    main()
