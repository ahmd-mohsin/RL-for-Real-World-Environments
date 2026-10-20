"""Kernel lane, the World and the compile gates (swarm-pipeline/06 §4, §6: S3, S4, S6 build edges, S8 compile floors).

Two images, both multi-arch (arm64 here for compile gates, x86_64 on GPU hosts; Apptainer runs the same OCI image with
``--nv``):

- ``crucible-k-base:12.8.1``: ``nvidia/cuda:12.8.1-devel-ubuntu24.04`` plus build tools. It is the agent's World (the repo at
  A mounted at ``/workspace/<repo>``, one ``base`` commit, ``--network none``) and the verifier (kit bundle mounted, root);
- ``crucible-k-legacy:11.8.0``: ``nvidia/cuda:11.8.0-devel-ubuntu22.04`` plus build tools: only the grader builds the
  original there (texture references were removed in CUDA 12.0), as ``compute_<legacy>`` PTX with static cudart. The agent
  never sees it.

Compile gates run the grader's own nvcc command (``kernel_plan.nvcc_command``) in throwaway ``crucible-k-*`` containers
with no network: the start fails for sm_90 and sm_120 (or carries a removed construct the static scan finds), the oracle
builds for both, the original builds in the legacy image. Only containers this module names are ever removed.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.factory.lanes import kernel_plan as kp

BASE_FROM = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
LEGACY_FROM = "nvidia/cuda:11.8.0-devel-ubuntu22.04"
BASE_TAG = "crucible-k-base:12.8.1"
LEGACY_TAG = "crucible-k-legacy:11.8.0"
ARCHES = ("90", "120")  # Marlowe H100, lab RTX 5090
TOOLS = "build-essential cmake ninja-build git python3 make ca-certificates file"
PREFIX = "crucible-k-"


def dockerfile(base: str, *, extra: Sequence[str] = ()) -> str:
    pkgs = " ".join([TOOLS, *extra])
    return (f"FROM {base}\nENV DEBIAN_FRONTEND=noninteractive\n"
            f"RUN apt-get update && apt-get install -y --no-install-recommends {pkgs} && rm -rf /var/lib/apt/lists/*\n"
            "ENV PATH=/usr/local/cuda/bin:$PATH\n"
            "RUN useradd -m -u 1000 agent 2>/dev/null || true\n")


def docker(*args: str, timeout: float = 3600, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, input=input, check=False, errors="replace")


def image_id(tag: str) -> str | None:
    p = docker("image", "inspect", "--format", "{{.Id}}", tag, timeout=60)
    return p.stdout.strip() or None if p.returncode == 0 else None


def ensure_image(tag: str, base: str, *, extra: Sequence[str] = ()) -> str:
    if (iid := image_id(tag)):
        return iid
    p = docker("build", "-t", tag, "-", input=dockerfile(base, extra=extra), timeout=5400)
    if p.returncode != 0:
        raise RuntimeError(f"build {tag}: {p.stderr[-800:]}")
    return image_id(tag) or ""


def ensure_images() -> dict[str, str]:
    return {"base": ensure_image(BASE_TAG, BASE_FROM), "legacy": ensure_image(LEGACY_TAG, LEGACY_FROM)}


def own_containers() -> list[str]:
    p = docker("ps", "-a", "--filter", f"name={PREFIX}", "--format", "{{.Names}}", timeout=60)
    return [n for n in p.stdout.split() if n.startswith(PREFIX)]


def cleanup() -> int:
    """Remove the containers this lane created (by name prefix) and nothing else (the lab is shared)."""
    names = own_containers()
    if names:
        docker("rm", "-f", *names, timeout=300)
    return len(names)


# ---- trees -------------------------------------------------------------------------------------------------------------------
def _member_filter(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
    """``data`` filter, but a link to an absolute or escaping path is dropped instead of failing the whole export."""
    try:
        return tarfile.data_filter(member, path)
    except tarfile.FilterError:
        return None


def export_tree(repo: Path, ref: str, dest: Path, *, paths: Sequence[str] = ()) -> Path:
    """``git archive`` of ``ref`` (no history), optionally only ``paths``; blobs a partial clone lacks are fetched on demand."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as td:
        tar = Path(td) / "t.tar"
        p = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "-o", str(tar), ref, *(["--", *paths] if paths else [])],
                           capture_output=True, text=True, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"git archive {ref}: {p.stderr[-300:]}")
        with tarfile.open(tar) as t:
            t.extractall(dest, filter=_member_filter)
    return dest


def subtree_paths(repo: Path, ref: str, files: Sequence[str], *, max_files: int = 4000) -> list[str]:
    """For a large repository, the parts a window needs: each migrated file's top two directory levels plus shared
    ``include``/``common``/``util*`` directories at the root. Small repositories are exported whole (``[]``)."""
    names = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", ref], capture_output=True, text=True, check=False).stdout.split("\n")
    if len(names) <= max_files:
        return []
    keep = {"/".join(f.split("/")[:2]) if f.count("/") >= 2 else f.split("/")[0] for f in files}
    tops = {n.split("/")[0] for n in names if "/" in n}
    keep |= {t for t in tops if t.lower() in ("include", "common", "util", "utils", "shared", "cuda-common")}
    return sorted(keep)


def workspace(tree: Path, ws: Path, *, tag: str = "base") -> Path:
    """The agent's repo: the start tree as one fresh commit tagged ``base`` (history cut: no later object exists)."""
    if ws.exists():
        shutil.rmtree(ws)
    shutil.copytree(tree, ws, symlinks=True)
    g = ["git", "-C", str(ws), "-c", "user.name=crucible", "-c", "user.email=crucible@localhost", "-c", "init.defaultBranch=main"]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "start"], ["tag", tag]):
        p = subprocess.run([*g, *args], capture_output=True, text=True, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"workspace git {args[0]}: {p.stderr[-300:]}")
    return ws


def write_hooks(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "crucible_observe.cpp").write_text(kp.OBSERVE_CPP)
    (dest / "crucible_launch.cpp").write_text(kp.LAUNCH_CPP)
    return dest


def tree_digest(tree: Path, files: Sequence[str]) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        p = tree / f
        h.update(f.encode() + b"\0" + (p.read_bytes() if p.is_file() else b"<absent>") + b"\0")
    return h.hexdigest()


# ---- compile gates -------------------------------------------------------------------------------------------------------------
@dataclass
class BuildResult:
    ok: bool
    rc: int
    errors: list[str]

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "rc": self.rc, "errors": self.errors}


def compile_many(tree: Path, programs: Mapping[str, Mapping[str, Any]], *, arch: str = "90", legacy: bool = False, legacy_arch: str = "60",
                 image: str | None = None, timeout: float = 3600, keep: Path | None = None) -> dict[str, BuildResult]:
    """Build each program of ``tree`` with the grader's nvcc command in one container (no network, the tree read-only).
    ``keep``: copy the binaries there (the legacy stage keeps the original's builds)."""
    image = image or (LEGACY_TAG if legacy else BASE_TAG)
    with tempfile.TemporaryDirectory(prefix="ckb-") as td:
        work = Path(td)
        hooks = write_hooks(work / "hooks")
        out = work / "out"
        out.mkdir()
        lines = ["set +e"]
        for pid, prog in programs.items():
            if prog.get("_tu"):  # a translation-unit job (compile milestone): ``-c`` only
                cmd = kp.tu_command(dict(prog), prog["_tu"], root=Path("/t"), out=Path(f"/o/{pid}.bin"), arch=arch, legacy=legacy, legacy_arch=legacy_arch,
                                    nvcc="nvcc")
            else:
                cmd = kp.nvcc_command(dict(prog), root=Path("/t"), out=Path(f"/o/{pid}.bin"), arch=arch, observe=Path("/h/crucible_observe.cpp"),
                                      legacy=legacy, legacy_arch=legacy_arch, nvcc="nvcc")
            q = " ".join(shlex.quote(c) for c in cmd)
            lines.append(f"( cd /t && {q} ) > /o/{pid}.log 2>&1; echo $? > /o/{pid}.rc")
        (work / "build.sh").write_text("\n".join(lines) + "\n")
        name = f"{PREFIX}build-{uuid.uuid4().hex[:10]}"
        p = docker("run", "--rm", "--name", name, "--network", "none", "-v", f"{Path(tree).resolve()}:/t:ro", "-v", f"{hooks}:/h:ro",
                   "-v", f"{out}:/o", "-v", f"{work / 'build.sh'}:/build.sh:ro", image, "bash", "/build.sh", timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"compile container failed: {p.stderr[-600:]}")
        res: dict[str, BuildResult] = {}
        for pid in programs:
            rc_f, log_f = out / f"{pid}.rc", out / f"{pid}.log"
            rc = int(rc_f.read_text().strip() or 1) if rc_f.exists() else 1
            log = log_f.read_text(errors="replace") if log_f.exists() else ""
            errs = [ln.strip()[:240] for ln in log.splitlines() if " error" in ln or ln.startswith("error") or ": error" in ln][:60]
            ok = rc == 0 and (out / f"{pid}.bin").exists()
            res[pid] = BuildResult(ok, rc, errs if not ok else [])
            if keep is not None and ok:
                keep.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out / f"{pid}.bin", keep / pid)
        return res


def static_scan(tree: Path, files: Sequence[str], kinds: Sequence[str]) -> list[dict[str, Any]]:
    """The grader's static rule run on a tree (the start must hold a removed construct; the oracle must hold none)."""
    return kp.static_hits(Path(tree), list(files), list(kinds))


def leak_probe(image: str, needles: Sequence[str], *, roots: Sequence[str] = ("/usr/local/cuda/", "/usr/share/doc", "/opt", "/root", "/home", "/usr/src"),
               timeout: float = 900) -> dict[str, Any]:
    """No file in the World image (outside the mounted workspace) holds a line only the oracle adds. ``needles`` are B-only
    lines long enough to be distinctive (the caller picks them)."""
    needles = [n for n in needles if len(n.strip()) >= 24][:40]
    if not needles:
        return {"ok": True, "probed": 0, "hits": []}
    pat = "\n".join(n.strip() for n in needles)
    name = f"{PREFIX}leak-{uuid.uuid4().hex[:10]}"
    roots_q = " ".join(shlex.quote(r) for r in roots)
    script = f"cat > /tmp/n.txt <<'CRUCIBLE_EOF'\n{pat}\nCRUCIBLE_EOF\nfor r in {roots_q}; do [ -e \"$r\" ] && grep -rIlF -f /tmp/n.txt \"$r\" 2>/dev/null; done | head -20; true"
    p = docker("run", "--rm", "--name", name, "--network", "none", image, "bash", "-c", script, timeout=timeout)
    hits = [ln for ln in p.stdout.splitlines() if ln.strip()]
    return {"ok": p.returncode == 0 and not hits, "probed": len(needles), "hits": hits, "rc": p.returncode}


def history_probe(ws: Path) -> dict[str, Any]:
    """The workspace holds one commit and its tag, nothing else (no later object to read)."""
    revs = subprocess.run(["git", "-C", str(ws), "rev-list", "--all"], capture_output=True, text=True, check=False).stdout.split()
    objs = subprocess.run(["git", "-C", str(ws), "count-objects", "-v"], capture_output=True, text=True, check=False).stdout
    return {"ok": len(revs) == 1, "commits": len(revs), "objects": objs.strip().replace("\n", "; ")}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")


# ---- a grading backend (``kernel.GpuBackend``) -----------------------------------------------------------------------------------
class DockerBackend:
    """Grades a patch with ``kernel_plan`` inside the verifier image, as root, candidate code unprivileged, ``--network none``.

    ``gpu=True`` adds the GPU (``--runtime=nvidia``, the lab's working form) and needs the originals built for the window
    (``legacy_root/<window id>/<program>``, root-only); ``gpu=False`` grades the compile milestones only (``cpu_only``):
    run milestones come back ``infra`` and the lane masks the reward, while ``phi_compile`` is still reported. The same
    image runs under Apptainer with ``--nv`` (``lh_kernel_gpu.py``)."""

    def __init__(self, store: Path, *, arch: str = "90", gpu: bool = False, legacy_root: Path | None = None, image: str = BASE_TAG,
                 gpu_args: Sequence[str] = ("--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all"), timeout: float = 7200):
        self.store, self.arch, self.gpu, self.legacy_root, self.image = Path(store), str(arch), gpu, legacy_root, image
        self.gpu_args, self.timeout = tuple(gpu_args), timeout
        self._kit: Path | None = None

    def kit(self) -> Path:
        if self._kit is None:
            from crucible.verify.stage import bundle

            d = Path(tempfile.mkdtemp(prefix="ck-kit-"))
            bundle(d / "kit", {"crucible.factory.lanes.kernel_plan": Path(kp.__file__)})
            write_hooks(d / "hooks")
            self._kit = d
        return self._kit

    def run(self, window: Mapping[str, Any], patch: str) -> dict[str, Any]:
        from crucible.factory.lanes import kernel as K

        kit = self.kit()
        with tempfile.TemporaryDirectory(prefix="ck-grade-") as td:
            t = Path(td)
            (t / "patch").write_text(patch)
            (t / "kspec.json").write_text(json.dumps(K.kspec(window, legacy_dir="/k/legacy")))
            out = t / "out"
            out.mkdir()
            name = f"{PREFIX}grade-{uuid.uuid4().hex[:10]}"
            cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--init", "-e", f"CRUCIBLE_CUDA_ARCH={self.arch}"]
            if self.gpu:
                cmd += list(self.gpu_args)
                cmd += ["-v", f"{Path(self.legacy_root or '') / window['id']}:/k/legacy:ro"]
            base = self.store / window["tree_a"]
            cmd += ["-v", f"{kit / 'kit'}:/crucible_verify:ro", "-v", f"{kit / 'hooks'}:/k/hooks:ro", "-v", f"{base.resolve()}:/k/base:ro",
                    "-v", f"{t / 'patch'}:/k/patch:ro", "-v", f"{t / 'kspec.json'}:/k/kspec.json:ro", "-v", f"{out}:/out", self.image,
                    "python3", "-I", "/crucible_verify/crucible/verify/runtime/run.py", "--plan", "crucible.factory.lanes.kernel_plan", "--out", "/out",
                    "--base", "/k/base", "--patch", "/k/patch", "--param", "spec=/k/kspec.json", "--param", f"arch={self.arch}",
                    "--param", "base_tree=/k/base", "--param", "legacy_dir=/k/legacy", "--param", "hooks_dir=/k/hooks", "--require-root"]
            if not self.gpu:
                cmd += ["--param", "cpu_only=1"]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, check=False, errors="replace")
            rj, rep = out / "crucible_reward.json", out / "report.json"
            doc = json.loads(rj.read_text()) if rj.exists() else {"infra": {"status": "masked", "failure_kind": "no_reward", "reason": p.stderr[-400:]}}
            if rep.exists():
                doc["gate_status"] = json.loads(rep.read_text()).get("gates") or {}
            return doc

    async def grade(self, window: Mapping[str, Any], patch: str) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self.run, window, patch)
