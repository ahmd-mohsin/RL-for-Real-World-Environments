"""The window lane, World half (study 08 §1, §3; 09): the repository at A in a Docker CPU sandbox, its leak probes, and
the policy's World.

**Image per (repo, A).** ``crucible-lh-base:py312`` carries Python 3.12, git, the test runner and the grading guard
(``executable.GUARD_PLUGIN``). A World image adds A's own dependencies, installed at build time from A's packaging
(``pip install <A>[test,tests,testing]``, PEP 735 test groups, and the names in A's test requirement files). The
project stays installed for its metadata (A's version); ``PYTHONPATH`` puts the workspace first, so every import is the
agent's code. Images are keyed by the recipe and the packaging files' content with version strings blanked, so releases
with the same dependencies share one image. Every image and container
this module makes is named ``crucible-lh-*``.

**Leak probes** (executable, in the image, ``--network none``): ``pip download <pkg>==<B>`` fails; no file named like
B's wheel or sdist exists (the dist-info of A's own install is A's); no file in the image or the seeded workspace defines a new test of the window (``def <name>(``); the seeded workspace's
``git log --all`` shows one commit, tagged ``base``.

**The policy's World.** A ``common/git_repo`` App seeded with A (fresh ``git init``, one commit tagged ``base``) on the
host; its shell, file and git tools run through ``docker exec`` in a ``--network none`` container of the World image
that bind-mounts the workspace at the same absolute path, so ``WorkspaceHarness`` collects the patch from the host tree.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any

from crucible.factory.lanes.executable import GUARD_PLUGIN
from crucible.factory.lanes.windows_git import packaging_hash

PREFIX = "crucible-lh-"
BASE_IMAGE = "crucible-lh-base:py312"
BASE_PINS = ("pytest==8.3.3", "pytest-timeout==2.3.1")
BASE_DOCKERFILE = f"""FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir {' '.join(BASE_PINS)} hypothesis==6.112.1 freezegun==1.5.1 pytest-mock==3.14.0 pytz simplejson typing_extensions
COPY m17guard.py /opt/m17/m17guard.py
RUN git config --system safe.directory '*' && git config --system user.email agent@sandbox && git config --system user.name agent
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /w
"""
PACKAGING = ("pyproject.toml", "setup.py", "setup.cfg")
IMAGE_RECIPE = "2"  # part of every World image tag: a changed recipe never reuses an image built by the old one
REQ_FILE = re.compile(r"(^|/)(requirements[-_/]?)?(test|tests|testing|dev)[^/]*\.(txt|in)$|(^|/)requirements[-_](test|tests|testing|dev)[^/]*\.(txt|in)$"
                      r"|^requirements\.txt$", re.IGNORECASE)  # a root requirements.txt: python-dotenv lists its test deps (click, sh) there
EXTRAS = ("test", "tests", "testing")


def sha256_hex(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def docker(*args: str, timeout: float = 1800, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=False, input=input)


def image_id(tag: str) -> str | None:
    r = docker("image", "inspect", "--format", "{{.Id}}", tag, timeout=60)
    return r.stdout.strip() if r.returncode == 0 else None


def ensure_base() -> str:
    d = image_id(BASE_IMAGE)
    if d:
        return d
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "Dockerfile").write_text(BASE_DOCKERFILE)
        (Path(td) / "m17guard.py").write_text(GUARD_PLUGIN)
        r = docker("build", "-q", "-t", BASE_IMAGE, td)
        if r.returncode != 0:
            raise RuntimeError(f"base image build failed: {r.stderr[-1500:]}")
    return image_id(BASE_IMAGE) or ""


# ---- packaging ---------------------------------------------------------------------------------------------------------
def project_name(tree: Path) -> str | None:
    """The distribution name from ``pyproject.toml``, ``setup.cfg`` or ``setup.py``."""
    pp = tree / "pyproject.toml"
    if pp.exists():
        with contextlib.suppress(tomllib.TOMLDecodeError, OSError):
            d = tomllib.loads(pp.read_text())
            n = (d.get("project") or {}).get("name") or ((d.get("tool") or {}).get("poetry") or {}).get("name")
            if n:
                return str(n)
    cfg = tree / "setup.cfg"
    if cfg.exists():
        m = re.search(r"(?m)^\s*name\s*=\s*([\w.-]+)\s*$", cfg.read_text(errors="replace"))
        if m:
            return m.group(1)
    sp = tree / "setup.py"
    if sp.exists():
        m = re.search(r"""\bname\s*=\s*["']([\w.-]+)["']""", sp.read_text(errors="replace"))
        if m:
            return m.group(1)
    return None


def requirement_sources(tree: Path) -> tuple[list[str], list[str]]:
    """``(requirement files, PEP 735 group requirements)`` that install A's test dependencies."""
    files = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file() and p.stat().st_size < 50_000
                   and REQ_FILE.search(p.relative_to(tree).as_posix()) and ".git/" not in p.as_posix() and "/docs/" not in "/" + p.relative_to(tree).as_posix())
    groups: list[str] = []
    pp = tree / "pyproject.toml"
    if pp.exists():
        with contextlib.suppress(tomllib.TOMLDecodeError, OSError):
            dg = tomllib.loads(pp.read_text()).get("dependency-groups") or {}
            for k in ("test", "tests", "testing", "dev"):
                groups += [x for x in dg.get(k) or [] if isinstance(x, str)]
    return files, groups


def requirement_names(text: str) -> list[str]:
    """Requirement names with their pins dropped (old pins rarely install on 3.12); options, paths and URLs skipped."""
    out = []
    for ln in text.splitlines():
        s = ln.split("#", 1)[0].strip()
        if not s or s.startswith(("-", ".", "/", "git+", "http")) or "://" in s:
            continue
        m = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?", s)
        if m and m.group(1).lower() not in ("pytest", "pytest-timeout"):
            out.append(m.group(1) + (m.group(2) or ""))
    return out


def world_dockerfile(name: str | None, req_names: list[str], groups: list[str], *, packaged: bool = True) -> str:
    """``packaged``: A has a ``pyproject.toml`` / ``setup.py`` / ``setup.cfg`` (a bare tree installs nothing of its own)."""
    extras = ",".join(EXTRAS)
    lines = ["FROM " + BASE_IMAGE, "COPY a /opt/lh/a", "WORKDIR /opt/lh/a"]
    if packaged:
        lines.append(f'RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --no-cache-dir ".[{extras}]" || SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --no-cache-dir .')
    if groups:
        lines.append("RUN pip install --no-cache-dir " + " ".join(shlex.quote(g) for g in groups) + " || true")
    for n in req_names:  # one by one: a name that no longer installs never blocks the others
        lines.append(f"RUN pip install --no-cache-dir {shlex.quote(n)} || true")
    lines.append("RUN pip install --no-cache-dir " + " ".join(BASE_PINS))
    # the project stays installed: packages read their own metadata (humanize: importlib.metadata.version at import);
    # PYTHONPATH puts the workspace (src/ and root) ahead of site-packages, so every import is the agent's code
    lines += ["WORKDIR /w", "RUN rm -rf /opt/lh /root/.cache"]
    return "\n".join(lines) + "\n"


class ImageBook:
    """World images per (repo, A's packaging), built once, shared by every window with the same dependencies."""

    def __init__(self, *, parallel: int = 3):
        self._locks: dict[str, asyncio.Lock] = {}
        self._sem = asyncio.Semaphore(parallel)
        self.built: dict[str, dict[str, Any]] = {}

    def tag_for(self, repo: str, base_dir: Path) -> tuple[str, dict[str, Any]]:
        files, groups = requirement_sources(base_dir)
        names = sorted({n for f in files for n in requirement_names((base_dir / f).read_text(errors="replace"))})
        name = project_name(base_dir)
        h = packaging_hash(base_dir, [*PACKAGING, *files])
        slug = re.sub(r"[^a-z0-9_.-]+", "-", repo.lower())
        h = sha256_hex(IMAGE_RECIPE + h)
        return f"{PREFIX}w:{slug}-{h[:12]}", {"project": name, "requirement_files": files, "requirement_names": names, "groups": groups,
                                               "packaging_hash": h}

    async def ensure(self, repo: str, base_dir: Path) -> dict[str, Any]:
        """The World image card: tag, digest, how it was built, ``ok``."""
        tag, info = self.tag_for(repo, base_dir)
        lock = self._locks.setdefault(tag, asyncio.Lock())
        async with lock:
            if tag in self.built:
                return self.built[tag]
            digest = image_id(tag)
            card: dict[str, Any] = {"image": tag, **info}
            if digest is None:
                async with self._sem:
                    t0 = time.monotonic()
                    ok, log = await asyncio.to_thread(self._build, tag, base_dir, info)
                    card.update({"build_seconds": round(time.monotonic() - t0, 1), "build_log": log[-1500:]})
                digest = image_id(tag) if ok else None
            card.update({"digest": digest or f"unresolved:{tag}", "ok": digest is not None})
            self.built[tag] = card
            return card

    @staticmethod
    def _build(tag: str, base_dir: Path, info: dict[str, Any]) -> tuple[bool, str]:
        ensure_base()
        with tempfile.TemporaryDirectory(prefix="lhimg-") as td:
            ctx = Path(td)
            shutil.copytree(base_dir, ctx / "a", symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
            packaged = any((base_dir / f).exists() for f in PACKAGING)
            (ctx / "Dockerfile").write_text(world_dockerfile(info["project"], info["requirement_names"], info["groups"], packaged=packaged))
            r = docker("build", "-q", "-t", tag, str(ctx), timeout=1800)
            return r.returncode == 0, (r.stdout + r.stderr)


# ---- leak probes -------------------------------------------------------------------------------------------------------
LEAK_SCRIPT = r"""set -u
pip download --no-deps --no-cache-dir --retries 0 --timeout 3 -d /tmp/dl "$LH_PKG==$LH_VER" >/tmp/pipdl.log 2>&1
echo "PIP_RC=$?"
if [ "$LH_VER" = "0" ]; then echo "DIST_FILES="; else
echo "DIST_FILES=$(find / -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o \( -iname "$LH_DIST-$LH_VER*.whl" -o -iname "$LH_DIST-$LH_VER*.tar.gz" -o -iname "$LH_DIST-$LH_VER*.zip" \) -print 2>/dev/null | head -5 | tr '\n' ' ')"
fi
if [ -s /probe/names ]; then
  grep -rIlF -f /probe/names --include='*.py' / --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev 2>/dev/null > /tmp/hits || true
  echo "NAME_HITS=$(python3 -c "$LH_COUNT" /probe/names /tmp/hits | head -5 | tr '\n' ' ')"
else
  echo "NAME_HITS="
fi
cd /probe/ws && echo "GIT_COMMITS=$(git log --all --format=%H | wc -l)" && echo "GIT_TAGS=$(git tag | tr '\n' ' ')"
"""


# a file leaks B's tests when it defines several of them (one shared name such as test_other is a coincidence: pyflakes'
# own tests matched a new more-itertools test name); a window with a single new test needs that one
LEAK_COUNT = r"""import sys
names = [ln.rstrip("\n") for ln in open(sys.argv[1]) if ln.strip()]
need = min(2, len(names))
for path in open(sys.argv[2]).read().split("\n"):
    if not path:
        continue
    try:
        text = open(path, errors="replace").read()
    except OSError:
        continue
    if sum(1 for n in names if n in text) >= need:
        print(path)
"""


def seed_workspace(base_dir: Path, ws: Path) -> None:
    """What ``GitRepoApp`` provisions: the tree at A, a fresh repository, one commit tagged ``base``."""
    shutil.copytree(base_dir, ws, symlinks=True)
    cmd = ("git init -q -b main . && git add -A && git -c user.email=crucible@local -c user.name=crucible commit -qm base --allow-empty "
           "&& git tag -f base")
    subprocess.run(["bash", "-c", cmd], cwd=ws, check=True, capture_output=True)


def parse_leak(out: str) -> dict[str, str]:
    return {k: v.strip() for k, v in re.findall(r"(?m)^([A-Z_]+)=(.*)$", out)}


def judge_leak(fields: dict[str, str], *, version: str | None) -> dict[str, Any]:
    """Each probe passes, fails, or is ``n/a`` (no B version to download)."""
    pip_rc = fields.get("PIP_RC")
    probes = {
        "pip_download_b_fails": "n/a" if not version else ("pass" if pip_rc not in (None, "", "0") else "fail"),
        "no_b_distribution_file": "n/a" if not version else ("pass" if not fields.get("DIST_FILES") else "fail"),
        "no_new_test_names_in_image": "pass" if not fields.get("NAME_HITS") else "fail",
        "git_log_only_base": "pass" if fields.get("GIT_COMMITS") == "1" and fields.get("GIT_TAGS", "").split() == ["base"] else "fail",
    }
    return {"probes": probes, "ok": all(v in ("pass", "n/a") for v in probes.values()), "evidence": {k: fields.get(k, "")[:300] for k in fields}}


async def leak_probes(image: str, base_dir: Path, *, package: str | None, version: str | None, new_test_names: set[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lhleak-") as td:
        probe = Path(td)
        seed_workspace(base_dir, probe / "ws")
        # a definition of a new test, not the bare word (pip's vendored urllib3 says "timeout_value" too)
        (probe / "names").write_text("".join(f"def {n}(\n" for n in sorted(new_test_names)))
        dist = re.sub(r"[-_.]+", "?", (package or "none-such-package"))
        name = f"{PREFIX}leak-{uuid.uuid4().hex[:10]}"
        args = ["run", "--rm", "--name", name, "--network", "none", "-v", f"{probe.resolve()}:/probe:ro", "-e", f"LH_PKG={package or 'none-such-package'}",
                "-e", f"LH_VER={version or '0'}", "-e", f"LH_DIST={dist}", "-e", f"LH_COUNT={LEAK_COUNT}", image, "bash", "-c", LEAK_SCRIPT]
        r = await asyncio.to_thread(docker, *args, timeout=600)
    if r.returncode not in (0, 1) and "PIP_RC" not in r.stdout:
        return {"probes": {}, "ok": False, "evidence": {"error": (r.stderr or r.stdout)[-500:]}}
    return judge_leak(parse_leak(r.stdout), version=version)


# ---- the policy's World ------------------------------------------------------------------------------------------------
def world_spec(ws: Path, base_dir: Path) -> Any:
    from crucible.core.schema.world import WorldSpec

    return WorldSpec.model_validate({
        "world": "factory/window", "network": "no-network",
        "apps": [{"id": "repo", "template": "common/git_repo", "state": {"adapter": "fs", "params": {"root": str(ws)}},
                  "params": {"shell": True, "init_git": True, "seed_dir": str(base_dir), "max_output": 30000}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*"]}]})


async def open_world(*, image: str, base_dir: Path, workdir: Path, instruction: str, memory: str = "2g", cpus: str = "2") -> Any:
    """A started, reset World whose tools run in a ``--network none`` container of ``image`` (closing the World removes
    the container). ``world._crucible_ws`` is the host workspace."""
    from crucible.core.store import Store
    from crucible.world.exec import DockerExec
    from crucible.world.world import World

    workdir = Path(workdir).resolve()
    ws = workdir / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    name = f"{PREFIX}world-{uuid.uuid4().hex[:12]}"
    r = await asyncio.to_thread(docker, "run", "-d", "--name", name, "--network", "none", "--memory", memory, "--cpus", cpus, "--pids-limit", "512",
                                "-v", f"{ws}:{ws}", "-w", str(ws), "-e", f"PYTHONPATH={ws}/src:{ws}", "-e", "HOME=/tmp", image, "sleep", "infinity", timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"world container did not start: {r.stderr[-500:]}")
    w = World(world_spec(ws, base_dir), instructions={"agent": instruction}, store=Store(workdir / "s"), workdir=workdir / "w")
    try:
        await w.start()
        w.apps["repo"].ctx.extra["policy_exec"] = DockerExec(name)
        await w.reset()
    except BaseException:
        await asyncio.to_thread(docker, "rm", "-f", name, timeout=120)
        raise
    w._crucible_ws = ws
    w._crucible_container = name
    inner = w.close

    async def close() -> None:
        try:
            await inner()
        finally:
            await asyncio.to_thread(docker, "rm", "-f", name, timeout=120)

    w.close = close
    return w


def own_containers() -> list[str]:
    r = docker("ps", "-a", "--filter", f"name={PREFIX}", "--format", "{{.Names}}", timeout=60)
    return [x for x in r.stdout.split() if x.startswith(PREFIX)]


def card_json(card: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in card.items() if k != "build_log"}, sort_keys=True)
