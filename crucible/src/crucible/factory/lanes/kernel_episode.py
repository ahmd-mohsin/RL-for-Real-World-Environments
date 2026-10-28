"""Kernel lanes on a GPU: the agent's World and every episode grade on a CUDA host (swarm-pipeline/10 §9).

The kernel lanes' gates run on the GPU host from a bundle (``lh_kernel_debug_gpu.py``, ``lh_kernel_gpu.py``), but the
agent's World was CPU-only, so no student, frontier or swarm episode could ever run on a kernel task. This module is the
GPU episode:

- **The World** is one container on the GPU host (``GpuHost``: the lab RTX 5090 over SSH; its Docker daemon through the
  shared tunnel ``world.docker_tunnel``), started with the GPU runtime, ``--network none``, CPU, memory and PID limits.
  Nothing of it lives on this machine: the World runs on the ``container_exec`` backend, so the file tools, the shell, git
  and the state snapshots all run inside the container (``/ws``). The agent can build the repository with nvcc, run the
  repository's own op tests on the GPU and use compute-sanitizer. The start tree is the task's gated A' (the bundle's
  ``a.tar.gz``), with history cut (a fresh repository with one ``base`` commit) and a CUDA build of A' already configured
  and built in ``/ws/build`` (gitignored, never collected), so the agent's first build is incremental, not a 30-minute
  cold build. The template is made once per task, GPU arch and flags on the host (``ensure_template``), under the host's
  build lock.
- **Collection.** The harness collects the patch from a local mirror of the container's repository (``workspace``: the
  tracked and untracked, not ignored, files and ``.git``, streamed out with tar), so the collect hook, ``repo_broken`` and the
  base commit work unchanged; the checkpoint sidecar snapshots inside the container (``snapshot_patch``).
- **Grading** stays out of the agent's reach (V4): every checkpoint and the final grade run on the host as
  ``run_gpu.py --grade-one`` (kernel bugs) or ``stage_gpu.py --grade-one`` (migrations) from the episode bundle
  (``ensure_episode_bundle``): a fresh grading container, the pristine start tree plus the patch, the task's calibrated
  hidden cases, the same GPU model the task was calibrated on. Kernel-bug grades read the gate runs' build cache read-only
  (``ccache_ro``): unchanged units are reused, the candidate can neither write nor poison it.
- **GPU contention** (``GpuLock``): grades take the host's GPU lock shared, a perf task's grades take it exclusive, and a
  perf task's episode holds it exclusive while its World is open, so no timed step overlaps another episode's GPU work
  (the World is closed before the final grade). A perf task's checkpoints are deferred to the end (graded, never timed
  during the episode). Grades and template builds also take the host's build lock one at a time: concurrent nvcc builds ran
  the shared 5090 out of memory (2026-10-23).
- **Masking** (V8): a grade that cannot run (host unreachable, no calibrated spec, no reward document) is masked, never 0.
- **Host outages** (2026-10-26, ``world.hostreach``): every call to the host (the World's ``DockerExec``, ``remote``, the
  workspace pull, the liveness check) runs under the host's ``HostReach`` (``host_reach``): a hung call is cut within about
  50 s once the host stops answering, the episode then pauses with its clocks held and resumes when the host is back (a
  replayable call runs again, a policy's cut write or shell call comes back to it as interrupted), and a host still down
  after the pause budget (30 min) masks the attempt as ``transport_unreachable`` / ``host_unreachable``. ``open`` starts no
  World while the host is known to be down, and ``launch_gate`` holds a measurement's next job (fleet stop).

DeltaAI (GH200, Apptainer, user-mode, ``trusted: false``) needs the runner on the compute node itself (no Docker daemon to
reach from here); ``GpuHost.runtime == "apptainer"`` is declared for it and refused by ``open`` until that path is built
(swarm-pipeline/10 §9.4).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from crucible.factory.lanes import kernel_debug as kd
from crucible.factory.lanes import kernel_harness as kh

PREFIX = "crucible-lh-k-ep-"  # every container this module starts on a GPU host (World, template build); only these are removed
WORLD_ROOT = "/ws"
TEMPLATE_MOUNT = "/tmpl"
MIGRATION = "kernel-migration"


@dataclass(frozen=True)
class GpuHost:
    """One CUDA host episodes run on. ``ssh``: ``user@host`` (key auth, BatchMode); ``docker_host``: the daemon the World's
    container runs on; ``bundle`` / ``migration_bundle``: the gate bundles there (tasks, windows, the gate build cache,
    hooks, kit); ``work``: this module's directory there (templates, the episode bundles, the locks)."""

    name: str
    ssh: str
    arch: str
    gpu: str
    docker_host: str
    bundle: str
    migration_bundle: str
    work: str
    image: str = "crucible-lh-k:base-12.8.1"
    runtime: str = "docker"
    world_cpus: str = "8"
    world_memory: str = "16g"
    world_pids: int = 4096
    build_jobs: int = 8
    template_cpus: str = "16"
    template_memory: str = "40g"
    max_worlds: int = 2  # Worlds open at once (each may run nvcc); the shared 5090 has 93 GB
    key: str = "~/.ssh/id_ed25519"

    def ssh_argv(self, script: str) -> list[str]:
        # 2026-10-26: a connect fails within 15 s and a live session within 60 s of the host going quiet (it took 180 s)
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
                "-i", os.path.expanduser(self.key), self.ssh, "bash -c " + shlex.quote(script)]


HOSTS: dict[str, GpuHost] = {
    "rtx5090": GpuHost(name="rtx5090", ssh="ahmed@100.100.101.36", arch="120", gpu="NVIDIA GeForce RTX 5090", docker_host="ssh://ahmed@100.100.101.36",
                       bundle="/home/ahmed/crucible-lh-k/gpu_bundle", migration_bundle="/home/ahmed/crucible-lh-k/gpu_bundle/migration",
                       work="/home/ahmed/crucible-lh-k-ep"),
    # DeltaAI GH200 (sm_90, aarch64, Apptainer, user-mode grader): declared, not yet runnable from here (module doc)
    "gh200": GpuHost(name="gh200", ssh="deltaai", arch="90", gpu="NVIDIA GH200 120GB", docker_host="", bundle="/work/hdd/bjhg/mahmedmohsin/crucible/bundle",
                     migration_bundle="/work/hdd/bjhg/mahmedmohsin/crucible/bundle/migration", work="/work/hdd/bjhg/mahmedmohsin/crucible/episodes",
                     runtime="apptainer"),
}


def host_of(name: str) -> GpuHost:
    if name not in HOSTS:
        raise KeyError(f"unknown GPU host {name!r} (known: {sorted(HOSTS)})")
    return HOSTS[name]


def host_reach(host: GpuHost) -> Any:
    """The host's reachability (``world.hostreach``), shared with the World's ``DockerExec`` on the same daemon: probed with
    ``docker version`` through the shared tunnel (the World's own path), or with ``ssh ... true`` for a host without a daemon.
    Named after the host, so records, journals and the health report group its outages by ``rtx5090``."""
    from crucible.world import hostreach

    if host.docker_host:
        return hostreach.for_docker_host(host.docker_host, name=host.name)
    return hostreach.for_ssh(host.ssh, host.ssh_argv("true"), name=host.name)


async def launch_gate(host: GpuHost, *, max_s: float | None = None) -> bool:
    """Fleet stop (2026-10-26): before a measurement launches new episodes on ``host``, wait while the host is known to be
    down (``HostReach.wait_up``, at most ``max_s``, default the pause budget). False: still down, launch nothing."""
    return await host_reach(host).wait_up(max_s)


@dataclass
class Remote:
    rc: int
    out: str
    err: str
    seconds: float


async def _ssh_once(host: GpuHost, script: str, stdin: bytes | None, timeout: float) -> Remote:
    """One SSH call; cancelled (``HostReach.guard`` cut it), it kills its ssh."""
    t0 = time.monotonic()
    p = await asyncio.create_subprocess_exec(*host.ssh_argv(script), stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                                             stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(p.communicate(stdin), timeout)
    except TimeoutError:
        p.kill()
        await p.wait()
        return Remote(124, "", f"timed out after {timeout:.0f}s", time.monotonic() - t0)
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            p.kill()
        raise
    return Remote(p.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace"), time.monotonic() - t0)


async def remote(host: GpuHost, script: str, *, stdin: str | bytes | None = None, timeout: float = 7200, retries: int = 2) -> Remote:
    """``script`` on the host over SSH, under the host's ``HostReach`` (2026-10-26). An SSH failure (rc 255) the host's probe
    confirms is an outage: the call pauses the episode's clocks, waits for the host and runs the script again once it is
    back (every script this module sends is the same when repeated: a grade, a template build, the bundle), or raises
    ``HostUnreachable`` once the pause budget is spent. A call that hangs is cut once the host stops answering probes, the
    same way. An SSH failure the probe does not confirm (a dropped session) is retried with a backoff; a script's own
    failure is returned as it is."""
    from crucible.world.exec import SSH_NOT_DISPATCHED, HostUnreachable

    reach = host_reach(host)
    data = stdin.encode() if isinstance(stdin, str) else stdin

    async def once() -> Remote:
        r = await _ssh_once(host, script, data, timeout)
        if r.rc == 255 and await reach.confirm(r.err.strip()[-300:] or "ssh exited 255"):
            raise HostUnreachable(reach.name, f"ssh to {host.name}: {r.err.strip()[-300:]}", dispatched=not SSH_NOT_DISPATCHED.search(r.err))
        return r

    for attempt in range(retries + 1):
        r = await reach.call(once, what=f"ssh to {host.name}", replay=True)
        if r.rc != 255 or attempt == retries:
            return r
        await asyncio.sleep(15 * (attempt + 1))
    raise AssertionError("unreachable")


def last_json(text: str) -> dict[str, Any] | None:
    """The last line of ``text`` that is a JSON object (the runners print their image and build notes before it)."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(doc, dict):
                return doc
    return None


# ---- the GPU lock -----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GpuLock:
    """The host's GPU lock (``flock`` on ``<work>/gpu.lock``) and build lock (``<work>/build.lock``). A correctness grade
    takes the GPU lock shared (correctness grades and other episodes' correctness work may share the GPU); a perf task's
    grade, or a perf task's open World, takes it exclusive. Every grade and template build also takes the build lock, so
    one nvcc build of the grader's runs on the host at a time."""

    host: GpuHost

    def wrap(self, cmd: str, *, exclusive: bool = False, build: bool = True) -> str:
        w = self.host.work
        inner = f"bash -c {shlex.quote(cmd)}"  # flock runs a program, not a shell line
        if build:
            inner = f"flock {shlex.quote(w + '/build.lock')} {inner}"
        return f"mkdir -p {shlex.quote(w)} && flock {'-x' if exclusive else '-s'} {shlex.quote(w + '/gpu.lock')} {inner}"

    def hold_exclusive(self) -> subprocess.Popen:
        """An exclusive hold for as long as the returned process lives (a perf task's open World): ``flock -x`` around a
        ``sleep`` that ends when the SSH session does."""
        script = f"mkdir -p {shlex.quote(self.host.work)} && exec flock -x {shlex.quote(self.host.work + '/gpu.lock')} sleep 86400"
        return subprocess.Popen(self.host.ssh_argv(script), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---- the episode bundles on the host ----------------------------------------------------------------------------------------------
_BUNDLE_LOCK = asyncio.Lock()
_BUNDLE_READY: set[tuple[str, str]] = set()
# The modules each runner imports from the kit, as ``cmd_stage_gpu`` bundles them. The host's gate kit can predate one (on
# 2026-10-25 kernel_profiles.py was missing there, so every episode grade raised ImportError and all 8 episodes of a task were
# masked): the episode kit is the host kit's files plus these.
KIT_MODULES = {"k": ("crucible/factory/lanes/kernel_debug_plan.py", "crucible/factory/lanes/kernel_harness.py",
                     "crucible/factory/lanes/kernel_profiles.py", "crucible/factory/lanes/kernel_plan.py"),
               "m": ("crucible/factory/lanes/kernel_plan.py",)}


def kit_files(host_files: Sequence[str], lane: str) -> list[str]:
    """The episode kit's ``crucible/...`` files: the host gate kit's, plus the runner's own modules and their packages."""
    out = {x for x in host_files if x.startswith("crucible/")}
    for rel in KIT_MODULES[lane]:
        out.add(rel)
        parts = rel.split("/")[:-1]
        out.update("/".join(parts[:k]) + "/__init__.py" for k in range(1, len(parts) + 1))
    return sorted(out)


def _kit_tar(kit_files: Sequence[str], src_root: Path) -> bytes:
    """The episode kit: this checkout's version of every file the gate bundle's kit holds (``crucible/...`` paths)."""
    import io

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for rel in kit_files:
            p = src_root / rel
            if p.is_file():
                tf.add(p, arcname=rel, recursive=False)
    return buf.getvalue()


async def ensure_episode_bundle(host: GpuHost, *, migration: bool = False) -> str:
    """``<work>/bundle`` (kernel bugs) or ``<work>/mbundle`` (migrations) on the host: the gate bundle's tasks, windows,
    hooks, images and build cache by symlink, and this checkout's runner and kit (so an episode grade runs the grader code
    of this checkout, without touching the gate bundle the gate runs use). Idempotent per process."""
    key = (host.name, "m" if migration else "k")
    async with _BUNDLE_LOCK:
        name = "mbundle" if migration else "bundle"
        dest = f"{host.work}/{name}"
        if key in _BUNDLE_READY:
            return dest
        src = host.migration_bundle if migration else host.bundle
        links = ("windows", "hooks", "hooks-base-12.8.1", "hooks-base-13.0.2", "images.json", "index.json", "Dockerfile.base", "Dockerfile.cuda13",
                 "Dockerfile.legacy") if migration else ("tasks", "hooks", "ccache", "index.json", "Dockerfile")
        ln = " ".join(f"[ -e {shlex.quote(src + '/' + x)} ] && ln -sfn {shlex.quote(src + '/' + x)} {shlex.quote(dest + '/' + x)};" for x in links)
        r = await remote(host, f"set -e; mkdir -p {shlex.quote(dest)}; {ln} cd {shlex.quote(src)}/kit && find crucible -name '*.py' | sort", timeout=300)
        if r.rc != 0:
            raise RuntimeError(f"episode bundle on {host.name}: {r.err[-400:]}")
        here = Path(__file__).resolve().parents[3]  # .../src
        kit = _kit_tar(kit_files(r.out.split(), key[1]), here)
        runner = Path(__file__).resolve().parents[4] / "scripts" / ("lh_kernel_gpu.py" if migration else "lh_kernel_debug_gpu.py")
        script = (f"set -e; rm -rf {shlex.quote(dest)}/kit.new; mkdir -p {shlex.quote(dest)}/kit.new; tar -xf - -C {shlex.quote(dest)}/kit.new; "
                  f"rm -rf {shlex.quote(dest)}/kit; mv {shlex.quote(dest)}/kit.new {shlex.quote(dest)}/kit")
        r = await remote(host, script, stdin=kit, timeout=300)  # bounded, and paused for the host like every other call (it was unbounded)
        if r.rc != 0:
            raise RuntimeError(f"episode kit on {host.name}: {r.err[-400:]}")
        name_on_host = "stage_gpu.py" if migration else "run_gpu.py"
        r = await remote(host, f"cat > {shlex.quote(dest + '/' + name_on_host)}", stdin=runner.read_text(), timeout=120)
        if r.rc != 0:
            raise RuntimeError(f"episode runner on {host.name}: {r.err[-400:]}")
        # preflight: the runner imports its kit (an ImportError here would mask every grade of every episode)
        r = await remote(host, f"cd {shlex.quote(dest)} && python3 {name_on_host} --help > /dev/null", timeout=120)
        if r.rc != 0:
            raise RuntimeError(f"episode runner on {host.name} does not start: {r.err[-400:]}")
        _BUNDLE_READY.add(key)
        return dest


# ---- grading ------------------------------------------------------------------------------------------------------------------------
def is_migration(rec: Mapping[str, Any]) -> bool:
    return str(rec.get("family") or "") == MIGRATION or str(rec.get("id") or "").startswith("km-")


def has_perf(rec: Mapping[str, Any]) -> bool:
    return any(m.get("kind") == "perf" or m.get("perf") for m in rec.get("milestones") or [])


@dataclass
class HostGrader:
    """``kernel_debug.GpuBackend`` on a GPU host: the patch graded by the episode bundle's runner (``--grade-one``) under
    the host's locks. ``grades`` logs every grade's seconds, for the cost per episode (GPU-minutes)."""

    host: GpuHost
    timeout_s: float = 3600.0
    grades: list[dict[str, Any]] = field(default_factory=list)

    async def grade(self, rec: Mapping[str, Any], patch: str) -> dict[str, Any]:
        from crucible.world.exec import HOST_UNREACHABLE, host_unreachable_cause

        mig = is_migration(rec)
        t0 = time.time()
        try:
            bundle = await ensure_episode_bundle(self.host, migration=mig)
        except Exception as e:  # noqa: BLE001 - the host or its bundle is unreachable: masked, never 0 (V8)
            kind = HOST_UNREACHABLE if host_unreachable_cause(e) is not None else "gpu_host"
            return {"infra": {"status": "masked", "failure_kind": kind, "reason": f"{type(e).__name__}: {e}"[:400]}}
        runner = "stage_gpu.py" if mig else "run_gpu.py"
        cmd = f"cd {shlex.quote(bundle)} && python3 {runner} --arch {self.host.arch} --mode {self.host.runtime} --grade-one {shlex.quote(str(rec['id']))}"
        try:
            r = await remote(self.host, GpuLock(self.host).wrap(cmd, exclusive=has_perf(rec)), stdin=patch, timeout=self.timeout_s)
        except Exception as e:  # the host did not come back within the pause budget (hostreach): masked, the patch regraded later
            if (gone := host_unreachable_cause(e)) is None:
                raise
            self.grades.append({"task": rec["id"], "seconds": round(time.time() - t0, 1), "grade_s": None, "rc": None, "at": time.time(), "infra": HOST_UNREACHABLE})
            return {"infra": {"status": "masked", "failure_kind": HOST_UNREACHABLE, "reason": str(gone)[:400]}}
        doc = last_json(r.out)
        row = {"task": rec["id"], "seconds": round(time.time() - t0, 1), "grade_s": (doc or {}).get("grade_s"), "rc": r.rc, "at": time.time()}
        self.grades.append(row)
        if doc is None:
            return {"infra": {"status": "masked", "failure_kind": "no_grade", "reason": (r.err or r.out)[-400:]}}
        doc.setdefault("gpu", self.host.gpu)
        return doc


# ---- the World ------------------------------------------------------------------------------------------------------------------
def world_image(host: GpuHost, rec: Mapping[str, Any]) -> str:
    """The World's image: a migration's target toolkit (CUDA 13 windows build with the 13.0.2 image, as their grader's
    ``images.json`` says), else the host's (CUDA 12.8). A record's own ``world_image`` wins."""
    if rec.get("world_image"):
        return str(rec["world_image"])
    if is_migration(rec) and "cuda13" in str(rec.get("id") or ""):
        return "crucible-lh-k:base-13.0.2"
    return host.image


def template_key(host: GpuHost, rec: Mapping[str, Any]) -> str:
    flags = " ".join(kh.cmake_cuda([host.arch]))
    return hashlib.sha256("\0".join([str(rec["id"]), host.arch, world_image(host, rec), flags, str(host.build_jobs)]).encode()).hexdigest()[:12]


def template_script(host: GpuHost, rec: Mapping[str, Any]) -> tuple[str, str]:
    """(the template's directory on the host, the script that makes it): A' unpacked from the bundle, CMake configured with
    the grader's CUDA flags for this GPU and the op-test target built, as the host user (so the template is removable
    without root), under the build lock; ``.ready`` marks a finished template."""
    tid = str(rec["id"])
    t = f"{host.work}/tmpl/{tid}-{host.arch}-{template_key(host, rec)}"
    q = shlex.quote
    if is_migration(rec):
        src = f"{host.migration_bundle}/windows/{tid}/base.tar.gz"
        unpack = f"tar -xzf {q(src)} -C {q(t + '.part')} && mv {q(t + '.part/base')} {q(t + '.part/ws')}"
        build = "true"  # a migration window's programs build in seconds: the agent builds them itself
    else:
        src = f"{host.bundle}/tasks/{tid}/a.tar.gz"
        unpack = f"tar -xzf {q(src)} -C {q(t + '.part')} && mv {q(t + '.part/a')} {q(t + '.part/ws')}"
        flags = " ".join(q(x) for x in kh.cmake_cuda([host.arch]))
        build = (f"cmake -S . -B build -G Ninja {flags} > /tmp/configure.log 2>&1 && cmake --build build --target {kh.TARGET} -j {host.build_jobs} "
                 f"> /tmp/build.log 2>&1 || {{ tail -40 /tmp/configure.log /tmp/build.log; exit 1; }}")
    name = f"{PREFIX}tmpl-{uuid.uuid4().hex[:8]}"
    run = (f"docker run --rm --name {name} --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --network none --cpus {host.template_cpus} "
           f"--memory {host.template_memory} --user $(id -u):$(id -g) -e HOME=/tmp -v {q(t + '.part/ws')}:/ws -w /ws {q(world_image(host, rec))} bash -c {q(build)}")
    inner = (f"[ -f {q(t + '/.ready')} ] && exit 0; rm -rf {q(t + '.part')}; mkdir -p {q(t + '.part')} && {unpack} && {run} && "
             f"rm -rf {q(t)} && mv {q(t + '.part')} {q(t)} && touch {q(t + '/.ready')}")
    script = f"set -e; mkdir -p {q(host.work + '/tmpl')}; [ -f {q(t + '/.ready')} ] || flock {q(host.work + '/build.lock')} bash -c {q(inner)}; echo {q(t)}"
    return t, script


async def ensure_template(host: GpuHost, rec: Mapping[str, Any], *, timeout: float = 7200) -> str:
    t, script = template_script(host, rec)
    r = await remote(host, script, timeout=timeout)
    if r.rc != 0 or t not in r.out:
        raise RuntimeError(f"World template for {rec['id']} on {host.name} failed: {(r.out + r.err)[-800:]}")
    return t


def world_run_argv(host: GpuHost, name: str, template: str, image: str | None = None) -> list[str]:
    return ["docker", "run", "-d", "--name", name, "--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all", "--network", "none", "--cpus", host.world_cpus,
            "--memory", host.world_memory, "--pids-limit", str(host.world_pids), "--shm-size", "4g", "--label", "crucible.lane=kernel-episode",
            "-e", "HOME=/tmp", "-v", f"{template}/ws:{TEMPLATE_MOUNT}:ro", "-w", WORLD_ROOT, image or host.image, "sleep", "infinity"]


def docker_env(host: GpuHost) -> dict[str, str]:
    from crucible.world.docker_tunnel import rewrite_env

    return rewrite_env(["docker"], {**os.environ, "DOCKER_HOST": host.docker_host}) if host.docker_host else dict(os.environ)


def world_spec(instruction_root: str = WORLD_ROOT) -> Any:
    from crucible.core.schema.world import WorldSpec

    return WorldSpec.model_validate({
        "world": "factory/kernel-gpu", "network": "no-network",
        "apps": [{"id": "repo", "template": "common/git_repo", "state": {"adapter": "fs", "params": {"root": instruction_root}},
                  "params": {"shell": True, "init_git": True, "max_output": 30000}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*"]}]})


ENV_NOTE = ("\n\nThe environment: the repository is at {root} inside a container with an NVIDIA {gpu} (sm_{arch}) and the CUDA toolkit "
            "(nvcc, compute-sanitizer). {build}There is no network.")
BUILD_NOTE = ("A CUDA build of the repository as you received it is already configured in {root}/build with the grader's flags for this "
              "GPU: `cmake --build build --target {target} -j {jobs}` rebuilds what you change, and `./build/bin/{target} -b CUDA0 -o <OP>` runs "
              "the op tests of one op on the GPU. ")


def env_note(host: GpuHost, rec: Mapping[str, Any]) -> str:
    build = "" if is_migration(rec) else BUILD_NOTE.format(root=WORLD_ROOT, target=kh.TARGET, jobs=host.build_jobs)
    return ENV_NOTE.format(root=WORLD_ROOT, gpu=host.gpu, arch=host.arch, build=build)


class KernelEpisodeTask(kd.KernelDebugTask):
    """A kernel task whose episodes run on a GPU host (module doc): ``KernelDebugTask`` (prompt, milestones, floors, mutants,
    ``grade_patch`` through the backend) plus the World, the snapshot and the collection the episode harness needs."""

    def __init__(self, rec: dict[str, Any], host: GpuHost, *, grader: HostGrader | None = None):
        super().__init__(rec, grader or HostGrader(host), f"kernel-debug-v1+{host.name}")
        self.host = host
        self.language = "cuda"
        self.family = str(rec.get("family") or "")
        self.split = str(rec.get("split") or "train")
        self.intent = "+".join(sorted((kd.task_intent(rec) or {}).get("skills") or [])) if not is_migration(rec) else "toolkit_migration"
        self.lane = kd.LANE
        self.id = self.task_id  # the runner's key (runtime.run)
        self.world_id = f"factory/kernel-gpu/{host.name}/{rec.get('repo', '')}"

    def identity_config(self) -> dict[str, Any]:
        """What a resume compares (``runtime.identity``): the record's content, the grader and the GPU host the episodes run
        on (a re-measurement on another GPU is another run)."""
        from crucible.factory.lanes import kernel as K

        return {"id": self.task_id, "record": K.kp_hash(self.rec), "verifier_version": self.verifier_version, "host": self.host.name, "gpu": self.host.gpu,
                "arch": self.host.arch, "world_image": world_image(self.host, self.rec), "template": template_key(self.host, self.rec)}

    @property
    def milestones(self) -> list[dict[str, Any]]:
        """Both record shapes: a kernel bug's (tests, files, ops) and a migration window's (programs, files, a compile or run
        kind); the sidecar reads ``id`` and ``requires``."""
        return [{"id": m["id"], "kind": m.get("kind"), "tests": list(m.get("tests") or []), "requires": list(m.get("requires") or []),
                 "requires_evidence": dict(m.get("requires_evidence") or {}), "files": list(m.get("files") or []), "ops": list(m.get("ops") or []),
                 "programs": list(m.get("programs") or [])} for m in self.rec["milestones"]]

    @property
    def requirement_ids(self) -> list[str]:
        return [str(q["id"]) for q in self.requirements]

    @property
    def requirement_milestones(self) -> dict[str, list[str]]:
        return {str(q["id"]): [str(q["milestone"])] for q in self.requirements if q.get("milestone")}

    @property
    def instruction(self) -> str:
        return kd.instruction(self.prompt, self.host.arch) + env_note(self.host, self.rec)

    def ref(self) -> Any:
        """``KernelDebugTask.ref`` for both record shapes: a migration window's check set is its own ``kspec``."""
        if not is_migration(self.rec):
            return super().ref()
        from crucible.factory.lanes import kernel as K
        from crucible.factory.longtask import LongTaskRef

        return LongTaskRef(lane=kd.LANE, task_id=self.task_id, source_id=self.task_id, store=str(self.rec.get("store") or ""), split=self.split,
                           prompt_sha256=K.kp_hash(self.prompt), checkset_sha256=K.kp_hash(self.rec.get("kspec") or {}), verifier_version=self.verifier_version)

    @property
    def checkpoint_defer(self) -> bool:
        """A perf task's checkpoints are graded at the end, never during the episode (a timed step must not overlap the
        agent's own GPU work)."""
        return has_perf(self.rec)

    async def open(self, workdir: Path) -> Any:
        if self.host.runtime != "docker":
            raise RuntimeError(f"GPU host {self.host.name}: the {self.host.runtime} World runs on the compute node itself; not built yet (10 §9.4)")
        from crucible.core.store import Store
        from crucible.world.exec import HostUnreachable
        from crucible.world.world import World

        # fleet stop (2026-10-26): no World starts while its host is known to be down; the episode pauses for it (its clocks
        # held), and a host that does not come back masks the attempt as host_unreachable before any model call is made
        reach = host_reach(self.host)
        await reach.ready(what="opening a GPU World")
        template = await ensure_template(self.host, self.rec)
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        name = f"{PREFIX}{uuid.uuid4().hex[:12]}"
        env = await asyncio.to_thread(docker_env, self.host)
        hold = GpuLock(self.host).hold_exclusive() if has_perf(self.rec) else None
        try:
            r = await asyncio.to_thread(subprocess.run, world_run_argv(self.host, name, template, world_image(self.host, self.rec)), capture_output=True, text=True,
                                        env=env, timeout=300, check=False)
        except subprocess.TimeoutExpired as e:
            r = subprocess.CompletedProcess(e.cmd, 124, "", f"docker run timed out after {e.timeout}s")
        if r.returncode != 0:
            if hold is not None:
                hold.terminate()
            why = f"GPU World did not start on {self.host.name}: {(r.stderr or '')[-500:]}"
            if await reach.confirm(why):  # the host does not answer a probe either: an outage, not a broken World
                raise HostUnreachable(reach.name, why, dispatched=True)
            raise RuntimeError(why)

        async def remove() -> None:
            await asyncio.to_thread(subprocess.run, ["docker", "rm", "-f", name], capture_output=True, text=True, env=env, timeout=300, check=False)
            if hold is not None:
                hold.terminate()

        try:
            # the template is the host user's; the World's tools run as the container's root: the copy keeps timestamps (the
            # pre-built build stays incremental) and hands ownership to root (git refuses a repository another uid owns)
            seed = f"mkdir -p {WORLD_ROOT} && cp -a {TEMPLATE_MOUNT}/. {WORLD_ROOT}/ && chown -R 0:0 {WORLD_ROOT}"
            cp = await asyncio.to_thread(subprocess.run, ["docker", "exec", name, "bash", "-c", seed], capture_output=True, text=True, env=env, timeout=1800,
                                         check=False)
            if cp.returncode != 0:
                why = f"GPU World seed copy failed: {cp.stderr[-400:]}"
                if await reach.confirm(why):
                    raise HostUnreachable(reach.name, why, dispatched=True)
                raise RuntimeError(why)
            w = World(world_spec(), instructions={"agent": self.instruction}, store=Store(workdir / "s"), workdir=workdir / "w", backend="container_exec",
                      backend_params={"container": name, "docker_host": self.host.docker_host or None, "base": WORLD_ROOT, "gpus": True, "network_enforced": True})
            await w.start()
            await w.reset()
        except BaseException:
            await remove()
            raise
        w._crucible_container = name
        w._crucible_env = env
        w._crucible_reach = reach  # envs.world_alive pauses for the host instead of reading an outage as a dead World
        w._crucible_mirror = workdir / "mirror"
        inner = w.close

        async def close() -> None:
            try:
                await inner()
            finally:
                await remove()

        w.close = close
        return w

    def workspace(self, world: Any) -> Path:
        """A local mirror of the container's repository (tracked and untracked, not ignored, files and ``.git``), refreshed
        on every call: what the episode harness collects, checks and reads the base commit from."""
        return self._pull(world, [])

    async def aworkspace(self, world: Any) -> Path:
        """``workspace`` off the event loop (the harness prefers it, ``harness.workspace._workspace``): a pull that stalls is
        killed, and a pull the host's outage broke pauses the episode and pulls again once the host is back (a pull is the
        same when repeated); a host that does not come back raises ``HostUnreachable`` (2026-10-26: the pull ran on the
        event loop, so one stalled stream held every episode of the process)."""
        from crucible.world.exec import HostUnreachable

        reach = getattr(world, "_crucible_reach", None) or host_reach(self.host)

        async def once() -> Path:
            procs: list[subprocess.Popen] = []
            try:
                return await asyncio.to_thread(self._pull, world, procs)
            except asyncio.CancelledError:
                for p in procs:
                    with contextlib.suppress(OSError):
                        p.kill()
                raise
            except RuntimeError as e:
                if await reach.confirm(str(e)):
                    raise HostUnreachable(reach.name, f"pulling the workspace: {e}", dispatched=True) from e
                raise

        return await reach.call(once, what="pulling the workspace", replay=True)

    def _pull(self, world: Any, procs: list[subprocess.Popen]) -> Path:
        mirror = Path(world._crucible_mirror)
        if mirror.exists():
            shutil.rmtree(mirror)
        mirror.mkdir(parents=True)
        pull_tree(world._crucible_container, WORLD_ROOT, mirror, env=world._crucible_env, procs=procs)
        return mirror

    async def snapshot_patch(self, world: Any) -> str:
        """The container's repository as a patch against ``base`` (new files included), through a private index: the
        agent's own index and history are never touched (``windows.WindowTask.snapshot_patch``, run in the GPU World)."""
        from crucible.world.exec import DockerExec

        cmd = ("set -e; I=$(mktemp); cp .git/index \"$I\" 2>/dev/null || true; GIT_INDEX_FILE=\"$I\" git -c safe.directory='*' add -A >/dev/null 2>&1; "
               "GIT_INDEX_FILE=\"$I\" git -c safe.directory='*' -c core.quotepath=off diff --cached --binary base; rm -f \"$I\"")
        out = await DockerExec(world._crucible_container, docker_host=self.host.docker_host or None).run(cmd, cwd=WORLD_ROOT, timeout_s=300)
        if not out.ok:
            raise RuntimeError(f"workspace snapshot failed: {out.stderr[-300:]}")
        return out.stdout

    async def broken(self, world: Any) -> list[str]:
        from crucible.world.exec import DockerExec

        out = await DockerExec(world._crucible_container, docker_host=self.host.docker_host or None).run(f"test -d {WORLD_ROOT}/.git", timeout_s=60)
        return [] if out.ok else ["the workspace repository (.git) was deleted"]

    async def lane_grade(self, patch: Any) -> Any:
        """The kernel lane's own grade on the GPU host (``KernelDebugTask.grade_patch``: milestones from the runner's gate
        statuses), without the path gates: what the checkpoint sidecar reads."""
        return await kd.KernelDebugTask.grade_patch(self, patch)

    async def milestone_status(self, patch: str) -> dict[str, bool]:
        rep = await self.lane_grade(patch)
        if rep.infra.get("status") != "ok":
            raise RuntimeError(f"milestones unmeasured: {rep.infra}")
        return dict(rep.meta["milestones"])

    async def grade_patch(self, patch: Any, workdir: Path | None = None, *, trail: Sequence[Any] | None = None,
                          network_events: Sequence[str] | None = None) -> Any:
        """``WorkspaceHarness``'s verifier call, as the window lane makes it (``WindowTask.grade_patch``): the end state
        graded on the GPU host, then the path (V3: state first, path second, ``pathcheck.check_paths`` on the diff and the
        trail). The reward is ``phi_weight * Phi + (1 - phi_weight) * [every milestone and every measured path gate
        passes]``, 0 on a hack-severity violation (``windows.path_reward``); a masked grade is masked (V8), never 0."""
        from crucible.factory.lanes.windows import path_reward
        from crucible.factory.pathcheck import ToolCall, check_paths, find_trail
        from crucible.verify.runtime.plan import CheckResult, GradeReport

        t0 = time.monotonic()
        lane = await self.lane_grade(patch)
        raw = dict(getattr(lane, "raw", None) or {})
        host = {"host": self.host.name, "gpu": raw.get("gpu") or self.host.gpu, "grade_s": raw.get("grade_s")}
        if lane.infra.get("status") != "ok":
            infra = {**lane.infra, "status": "masked", "why": lane.infra.get("reason") or lane.infra.get("failure_kind")}
            return GradeReport(task=self.task_id, tier="full", checks=[], gates={}, reward=0.0, outcome=0.0, infra=infra, trusted=False,
                               elapsed_s=time.monotonic() - t0, meta={"why": infra["why"], "milestones": None, "phi": None, "kinds": {}, **host})
        status = dict(lane.meta["milestones"])
        gates = {m: ("pass" if ok else "fail") for m, ok in status.items()}
        full = bool(status) and all(status.values())
        phi = float(lane.meta.get("phi") or 0.0)
        meta = {"milestones": status, "phi": phi, "full_pass": full, "kinds": {m: "outcome" for m in gates}, "outcome_reward": lane.reward, **host}
        rep = GradeReport(task=self.task_id, tier="full", checks=[CheckResult(id=m, gate=m, status=s) for m, s in gates.items()], gates=gates,
                          reward=1.0 if full else 0.0, outcome=1.0 if full else 0.0, infra={"status": "ok"}, trusted=True, elapsed_s=time.monotonic() - t0,
                          meta=meta)
        if trail is None:
            calls, source = find_trail(workdir)
        else:
            calls = [t if isinstance(t, ToolCall) else ToolCall(str(t.get("function") or t.get("function_name") or ""), t.get("arguments")) for t in trail]
            source = "given"
        pr = check_paths(self.path_rules, patch=patch if isinstance(patch, str) else "", trail=calls, trail_source=source, network_events=network_events)
        pw = path_reward({"full_pass": full, "phi": phi}, pr, 0.5)
        for gid, st in pr.gates().items():
            rep.gates[gid] = st
            meta["kinds"][gid] = "gate"
            chk = next(c for c in pr.checks if c.id == gid)
            rep.checks.append(CheckResult(id=gid, gate=gid, status=st, detail={"severity": chk.severity, "evidence": chk.evidence[:10]}))
        meta.update(path=pr.to_json(), end_state_full_pass=pw["end_state_full_pass"], full_pass=pw["full_pass"], hack_suspect=pw["hack_suspect"],
                    outcome_reward=pw["outcome_reward"], reward_rule="0 if hack_suspect else 0.5*phi + 0.5*[full_pass and no path violation]")
        rep.reward = rep.outcome = pw["outcome_reward"]
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
            (Path(workdir) / "kernel_grade.json").write_text(json.dumps({"reward": rep.reward, "milestones": status, "phi": phi, "infra": rep.infra,
                                                                         "path": meta["path"], "hack_suspect": pw["hack_suspect"], **host}, indent=1))
        return rep

    def episode_checkpoint(self, cfg: Any) -> Any:
        """The episode's checkpoint config on this task: a perf task's checkpoints deferred to the end (``checkpoint_defer``)."""
        return replace(cfg, defer=True) if self.checkpoint_defer and not cfg.defer else cfg


PULL = ("cd {root} && {{ git -c safe.directory='*' ls-files -co --exclude-standard -z; find .git -print0; }} | "
        "tar --null --no-recursion -T - -cf -")


class _Watched:
    """A pipe whose reads note the time of the last byte (``pull_tree``'s stall watchdog)."""

    def __init__(self, f: Any):
        self.f, self.last = f, time.monotonic()

    def read(self, n: int = -1) -> bytes:
        b = self.f.read(n)
        if b:
            self.last = time.monotonic()
        return b


def pull_tree(container: str, root: str, dest: Path, *, env: Mapping[str, str] | None = None, timeout: float = 900, stall_s: float = 60.0,
              procs: list[subprocess.Popen] | None = None) -> None:
    """The container repository's tracked and untracked, not ignored, files and its ``.git``, streamed into ``dest``. The
    stream is killed when no byte arrives for ``stall_s`` or the pull runs past ``timeout`` (2026-10-26: a stream that a lost
    network stalled was read forever); ``procs`` gets the process, so an async caller can kill it when it is cancelled."""
    import threading

    p = subprocess.Popen(["docker", "exec", container, "bash", "-c", PULL.format(root=root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=dict(env) if env is not None else None)
    if procs is not None:
        procs.append(p)
    assert p.stdout is not None
    src = _Watched(p.stdout)
    t0, cut = time.monotonic(), []

    def watch() -> None:
        while p.poll() is None:
            now = time.monotonic()
            if now - src.last > stall_s or now - t0 > timeout:
                cut.append(f"no data for {now - src.last:.0f}s" if now - src.last > stall_s else f"past {timeout:.0f}s")
                with contextlib.suppress(OSError):
                    p.kill()
                return
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()
    try:
        with tarfile.open(fileobj=src, mode="r|") as tf:  # type: ignore[call-overload]
            tf.extractall(dest, filter="data")
    except tarfile.TarError as e:
        if not cut:
            raise
        raise RuntimeError(f"pulling the workspace from {container} stalled ({cut[0]}): {e}") from e
    finally:
        try:
            p.wait(timeout=max(1.0, timeout - (time.monotonic() - t0)))
        except subprocess.TimeoutExpired:
            p.kill()
    if cut:
        raise RuntimeError(f"pulling the workspace from {container} stalled ({cut[0]})")
    if p.returncode not in (0, None):
        err = p.stderr.read().decode(errors="replace") if p.stderr is not None else ""
        raise RuntimeError(f"pulling the workspace from {container} failed: {err[-300:]}")


def own_containers(host: GpuHost) -> list[str]:
    env = docker_env(host)
    r = subprocess.run(["docker", "ps", "-a", "--filter", f"name={PREFIX}", "--format", "{{.Names}}"], capture_output=True, text=True, env=env, timeout=120,
                       check=False)
    return [x for x in r.stdout.split() if x.startswith(PREFIX)]


def cleanup(host: GpuHost) -> list[str]:
    """Every container this module started on the host (``crucible-lh-k-ep-*``), and nothing else."""
    names = own_containers(host)
    if names:
        subprocess.run(["docker", "rm", "-f", *names], capture_output=True, text=True, env=docker_env(host), timeout=300, check=False)
    return names


def load_record(root: Path, tid: str) -> dict[str, Any]:
    """A kernel record by id under the lane's data dir ``root`` (``data/kernel_debug``): a kernel bug or perf task in
    ``store/<split>/<id>/record.json``, a migration window in ``migrate/store/windows/<id>.json``."""
    for p in (root / "store" / "train" / tid / "record.json", root / "store" / "heldout" / tid / "record.json", root / "migrate" / "store" / "windows" / f"{tid}.json"):
        if p.exists():
            return json.loads(p.read_text())
    raise KeyError(f"{tid}: no kernel record under {root}")


def load_tasks(root: Path, ids: Sequence[str], host: GpuHost) -> list[KernelEpisodeTask]:
    """The kernel records ``ids`` as episode tasks on ``host``."""
    return [KernelEpisodeTask(load_record(root, tid), host) for tid in ids]
