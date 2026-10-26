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

DeltaAI (GH200, Apptainer, user-mode, ``trusted: false``) needs the runner on the compute node itself (no Docker daemon to
reach from here): ``gh200`` (from the Mac) stays declared and refused by ``open``; ``gh200-node`` is the host as seen from a
job on the node (``local``: every host command runs in a local shell, the World's ``docker`` is the Apptainer shim of
``scripts/deltaai``, the grader runs in its Apptainer mode, the work directory is the job's node-local scratch,
``CRUCIBLE_KERNEL_WORK``; swarm-pipeline/14).
"""

from __future__ import annotations

import asyncio
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
    local: bool = False  # this process runs on the host itself (a DeltaAI job): host commands run in a local shell
    grade_mode: str = ""  # the grader runner's --mode (default: ``runtime``); a shimmed World on Apptainer grades with "apptainer"

    def ssh_argv(self, script: str) -> list[str]:
        if self.local:
            return ["bash", "-c", script]
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
                "-i", os.path.expanduser(self.key), self.ssh, "bash -c " + shlex.quote(script)]


HOSTS: dict[str, GpuHost] = {
    "rtx5090": GpuHost(name="rtx5090", ssh="ahmed@100.100.101.36", arch="120", gpu="NVIDIA GeForce RTX 5090", docker_host="ssh://ahmed@100.100.101.36",
                       bundle="/home/ahmed/crucible-lh-k/gpu_bundle", migration_bundle="/home/ahmed/crucible-lh-k/gpu_bundle/migration",
                       work="/home/ahmed/crucible-lh-k-ep"),
    # DeltaAI GH200 (sm_90, aarch64, Apptainer, user-mode grader): declared, not yet runnable from here (module doc)
    "gh200": GpuHost(name="gh200", ssh="deltaai", arch="90", gpu="NVIDIA GH200 120GB", docker_host="", bundle="/work/hdd/bjhg/mahmedmohsin/crucible/bundle",
                     migration_bundle="/work/hdd/bjhg/mahmedmohsin/crucible/bundle/migration", work="/work/hdd/bjhg/mahmedmohsin/crucible/episodes",
                     runtime="apptainer"),
    # the same GH200 seen from a job on the node: the World through the docker shim, the grader in Apptainer mode
    "gh200-node": GpuHost(name="gh200-node", ssh="local", arch="90", gpu="NVIDIA GH200 120GB", docker_host="",
                          bundle="/work/hdd/bjhg/mahmedmohsin/crucible/bundle", migration_bundle="/work/hdd/bjhg/mahmedmohsin/crucible/bundle/migration",
                          work=os.environ.get("CRUCIBLE_KERNEL_WORK") or "/tmp/crucible-kernel-ep", runtime="docker", local=True, grade_mode="apptainer",
                          world_cpus="16", world_memory="64g"),
}


def host_of(name: str) -> GpuHost:
    if name not in HOSTS:
        raise KeyError(f"unknown GPU host {name!r} (known: {sorted(HOSTS)})")
    return HOSTS[name]


@dataclass
class Remote:
    rc: int
    out: str
    err: str
    seconds: float


async def remote(host: GpuHost, script: str, *, stdin: str | None = None, timeout: float = 7200, retries: int = 2) -> Remote:
    """``script`` on the host over SSH. An SSH failure (rc 255: the host or the network dropped) is retried with a backoff;
    a script's own failure is returned as it is."""
    for attempt in range(retries + 1):
        t0 = time.monotonic()
        p = await asyncio.create_subprocess_exec(*host.ssh_argv(script), stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(p.communicate(stdin.encode() if stdin is not None else None), timeout)
        except TimeoutError:
            p.kill()
            await p.wait()
            return Remote(124, "", f"timed out after {timeout:.0f}s", time.monotonic() - t0)
        r = Remote(p.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace"), time.monotonic() - t0)
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
        # the gate kit's files, plus every kernel-lane module of this checkout: a gate bundle staged before a module existed
        # does not list it, and this checkout's runner imports it (kernel_profiles on DeltaAI's Oct 22 bundle, smoke 3461769)
        lanes = sorted(f"crucible/factory/lanes/{p.name}" for p in (here / "crucible" / "factory" / "lanes").glob("kernel_*.py"))
        kit = _kit_tar(sorted({x for x in r.out.split() if x.startswith("crucible/")} | set(lanes)), here)
        runner = Path(__file__).resolve().parents[4] / "scripts" / ("lh_kernel_gpu.py" if migration else "lh_kernel_debug_gpu.py")
        script = (f"set -e; rm -rf {shlex.quote(dest)}/kit.new; mkdir -p {shlex.quote(dest)}/kit.new; tar -xf - -C {shlex.quote(dest)}/kit.new; "
                  f"rm -rf {shlex.quote(dest)}/kit; mv {shlex.quote(dest)}/kit.new {shlex.quote(dest)}/kit")
        p = await asyncio.create_subprocess_exec(*host.ssh_argv(script), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.PIPE)
        _, err = await p.communicate(kit)
        if p.returncode != 0:
            raise RuntimeError(f"episode kit on {host.name}: {err.decode(errors='replace')[-400:]}")
        name_on_host = "stage_gpu.py" if migration else "run_gpu.py"
        r = await remote(host, f"cat > {shlex.quote(dest + '/' + name_on_host)}", stdin=runner.read_text(), timeout=120)
        if r.rc != 0:
            raise RuntimeError(f"episode runner on {host.name}: {r.err[-400:]}")
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
        mig = is_migration(rec)
        t0 = time.time()
        try:
            bundle = await ensure_episode_bundle(self.host, migration=mig)
        except Exception as e:  # noqa: BLE001 - the host or its bundle is unreachable: masked, never 0 (V8)
            return {"infra": {"status": "masked", "failure_kind": "gpu_host", "reason": f"{type(e).__name__}: {e}"[:400]}}
        runner = "stage_gpu.py" if mig else "run_gpu.py"
        mode = self.host.grade_mode or self.host.runtime
        cmd = f"cd {shlex.quote(bundle)} && python3 {runner} --arch {self.host.arch} --mode {mode} --grade-one {shlex.quote(str(rec['id']))}"
        r = await remote(self.host, GpuLock(self.host).wrap(cmd, exclusive=has_perf(rec)), stdin=patch, timeout=self.timeout_s)
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
        from crucible.world.world import World

        template = await ensure_template(self.host, self.rec)
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        name = f"{PREFIX}{uuid.uuid4().hex[:12]}"
        env = await asyncio.to_thread(docker_env, self.host)
        hold = GpuLock(self.host).hold_exclusive() if has_perf(self.rec) else None
        r = await asyncio.to_thread(subprocess.run, world_run_argv(self.host, name, template, world_image(self.host, self.rec)), capture_output=True, text=True, env=env, timeout=300,
                                    check=False)
        if r.returncode != 0:
            if hold is not None:
                hold.terminate()
            raise RuntimeError(f"GPU World did not start on {self.host.name}: {r.stderr[-500:]}")

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
                raise RuntimeError(f"GPU World seed copy failed: {cp.stderr[-400:]}")
            w = World(world_spec(), instructions={"agent": self.instruction}, store=Store(workdir / "s"), workdir=workdir / "w", backend="container_exec",
                      backend_params={"container": name, "docker_host": self.host.docker_host or None, "base": WORLD_ROOT, "gpus": True, "network_enforced": True})
            await w.start()
            await w.reset()
        except BaseException:
            await remove()
            raise
        w._crucible_container = name
        w._crucible_env = env
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
        mirror = Path(world._crucible_mirror)
        if mirror.exists():
            shutil.rmtree(mirror)
        mirror.mkdir(parents=True)
        pull_tree(world._crucible_container, WORLD_ROOT, mirror, env=world._crucible_env)
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


def pull_tree(container: str, root: str, dest: Path, *, env: Mapping[str, str] | None = None, timeout: float = 900) -> None:
    """The container repository's tracked and untracked, not ignored, files and its ``.git``, streamed into ``dest``."""
    p = subprocess.Popen(["docker", "exec", container, "bash", "-c", PULL.format(root=root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=dict(env) if env is not None else None)
    try:
        assert p.stdout is not None
        with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
            tf.extractall(dest, filter="data")
    finally:
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
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
