"""What runs a compiled-language profile in Docker (swarm-pipeline/07 §2): the World image book, the grading sandbox and
the leak probes. Every image is tagged ``crucible-lh-d:<lang>-<repo>-<hash>`` and every container is named
``crucible-lh-d-*``: a caller finds and removes exactly its own.

**World image** (``LangImageBook``, one per profile; the ``windows_world.ImageBook`` interface: ``ensure(repo,
base_dir) -> card``). ``FROM`` a language image pinned by digest; the profile's recipe copies the start tree to ``/w``,
fetches and compiles its **dependencies only** at build time (network on), deletes ``/w`` and every compiled artifact of
the project's own code, and leaves the network switched off for the toolchain (``GOPROXY=off``,
``CARGO_NET_OFFLINE=true``, ``npm_config_offline=true``). Images are keyed by the recipe, the base digest and the
packaging files (lockfiles, manifests) with versions blanked, so every candidate of one repository and release shares
one image, and an image never holds another candidate's start tree (a fix the image compiled would be in its caches).

**Grading sandbox** (``LangSandbox``, a ``windows.NamedSandbox``): ``docker run --network none`` with the tree mounted
read-only at ``/src`` and copied to ``/w`` inside the container (the compilers write caches next to the code and the
build path matches the image's), a tmpfs ``/tmp``, CPU, memory and pid caps. The profile's script writes its machine
report (``go test -json``, libtest lines per target, the vitest/jest JSON, surefire XML) to a random path under
``/tmp``; after the run every other process is killed and the report is printed between ``@@CRUCIBLE-L-BEGIN@@`` /
``@@CRUCIBLE-L-END@@`` markers (the last block counts, as with the pytest guard). The root filesystem is writable (a
container-local copy-on-write layer: the warm build caches live there); nothing persists past ``--rm``.

**Leak probes** (``lang_leak_probes``, executed in the image with ``--network none``): fetching the project from its
registry fails; no dependency store holds a copy of the project (``no_installed_project_source``: the Python lane's
installed-copy leak, per language); no file in the image defines two or more of the window's new test names; the
seeded workspace's history is one commit tagged ``base``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.lanes.executable import TestRun, killed
from crucible.factory.lanes.windows import NamedSandbox
from crucible.factory.lanes.windows_world import docker, image_id, judge_leak, parse_leak, seed_workspace

PREFIX = "crucible-lh-d-"  # every container of the debug lane
IMAGE_REPO = "crucible-lh-d"  # every image of the debug lane: crucible-lh-d:<lang>-<repo>-<hash>
BEGIN, END = "@@CRUCIBLE-L-BEGIN@@", "@@CRUCIBLE-L-END@@"
_BLOCK = re.compile(re.escape(BEGIN) + r"\n(.*?)\n" + re.escape(END), re.DOTALL)
_VERSION_LINE = re.compile(r"""(?im)^(\s*"?version"?\s*[:=]\s*)["'][^"']*["']""")


def packaging_key(tree: Path, files: Sequence[str]) -> str:
    """What a dependency install depends on: manifests with their own version line blanked (a release bump changes no
    dependency), lockfiles byte for byte (every ``version`` line in a lockfile is a dependency's: blanking them let two
    different lockfiles share one image whose caches lacked half the crates)."""
    import hashlib

    h = hashlib.sha256()
    for rel in sorted(files):
        f = Path(tree) / rel
        if f.is_file():
            text = f.read_text(errors="replace")
            lock = "lock" in rel.rsplit("/", 1)[-1].lower() or rel.endswith((".sum", "shrinkwrap.json"))
            h.update(rel.encode() + b"\0" + (text if lock else _VERSION_LINE.sub(r'\1"X"', text)).encode() + b"\0")
    return h.hexdigest()


def policy_world_dockerfile(image: str, profile: Any) -> str | None:
    """A derived image of a pinned World for the **policy** only (the grader keeps ``image``): ``FROM image`` plus the
    profile's ``policy_world_steps()`` (S15, 2026-10-20: the node World had no ``python3``, which the World's file tools
    run, and corepack's pnpm cache under the build's ``HOME`` that the policy's ``HOME=/tmp`` never finds). None when the
    profile has no step."""
    steps = list(getattr(profile, "policy_world_steps", list)())
    return "\n".join([f"FROM {image}", *steps]) + "\n" if steps else None


def policy_world(image: str, profile: Any) -> dict[str, Any]:
    """Build (once) the policy World of ``image`` (``policy_world_dockerfile``), tagged ``<image>-pw<hash>``. Returns
    ``{ok, image, from, dockerfile, log}``; ``image`` is ``image`` itself when the profile has no step."""
    import hashlib

    df = policy_world_dockerfile(image, profile)
    if df is None:
        return {"ok": True, "image": image, "from": image, "dockerfile": None, "log": ""}
    tag = f"{image}-pw{hashlib.sha256(df.encode()).hexdigest()[:8]}"
    if image_id(tag) is not None:
        return {"ok": True, "image": tag, "from": image, "dockerfile": df, "log": "cached"}
    with tempfile.TemporaryDirectory(prefix="lhd-pw-") as td:
        (Path(td) / "Dockerfile").write_text(df)
        r = docker("build", "--progress=plain", "-t", tag, td, timeout=1800)
    return {"ok": r.returncode == 0 and image_id(tag) is not None, "image": tag, "from": image, "dockerfile": df, "log": (r.stdout + r.stderr)[-2000:]}


class LangImageBook:
    """World images of one profile per (repo, packaging), built once (see the module doc)."""

    def __init__(self, profile: Any, *, parallel: int = 2):
        self.profile = profile
        self._locks: dict[str, asyncio.Lock] = {}
        self._sem = asyncio.Semaphore(parallel)
        self.built: dict[str, dict[str, Any]] = {}

    def tag_for(self, repo: str, base_dir: Path) -> tuple[str, dict[str, Any]]:
        p = self.profile
        files = p.packaging_files(Path(base_dir))
        info = p.image_info(Path(base_dir))
        key = packaging_key(Path(base_dir), files) + json.dumps({"recipe": p.recipe, "base": p.base_image, "info": info}, sort_keys=True)
        import hashlib

        h = hashlib.sha256(key.encode()).hexdigest()
        slug = re.sub(r"[^a-z0-9_.-]+", "-", repo.lower())[:40]
        return f"{IMAGE_REPO}:{p.name}-{slug}-{h[:12]}", {"lang": p.name, "recipe": p.recipe, "packaging": files, "project": info.get("project"),
                                                         "projects": info.get("projects") or [info.get("project")], "info": info}

    async def ensure(self, repo: str, base_dir: Path) -> dict[str, Any]:
        tag, info = self.tag_for(repo, base_dir)
        lock = self._locks.setdefault(tag, asyncio.Lock())
        async with lock:
            if tag in self.built:
                return self.built[tag]
            digest = await asyncio.to_thread(image_id, tag)
            card: dict[str, Any] = {"image": tag, **info}
            if digest is None:
                async with self._sem:
                    t0 = time.monotonic()
                    ok, log = await asyncio.to_thread(self._build, tag, Path(base_dir), info)
                    card.update({"build_seconds": round(time.monotonic() - t0, 1), "build_log": log[-3000:]})
                digest = await asyncio.to_thread(image_id, tag) if ok else None
            card.update({"digest": digest or f"unresolved:{tag}", "ok": digest is not None})
            self.built[tag] = card
            return card

    def _build(self, tag: str, base_dir: Path, info: Mapping[str, Any]) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory(prefix="lhd-img-") as td:
            ctx = Path(td)
            shutil.copytree(base_dir, ctx / "a", symlinks=True, ignore=shutil.ignore_patterns(".git", "node_modules", "target", "__pycache__"))
            for name, text in self.profile.build_files(base_dir, info["info"]).items():
                (ctx / name).write_text(text)
            (ctx / "Dockerfile").write_text(self.profile.dockerfile(base_dir, info["info"]))
            r = docker("build", "--progress=plain", "-t", tag, str(ctx), timeout=3600)
            return r.returncode == 0, (r.stdout + r.stderr)


class LangSandbox(NamedSandbox):
    """The grading and mining sandbox of a compiled-language profile (see the module doc)."""

    def __init__(self, image: str, *, profile: Any, name_prefix: str = f"{PREFIX}grade-", timeout_s: float = 900.0, cpus: str = "2",
                 memory: str = "2g", parallel: int = 3, tmpfs: str = "1g"):
        super().__init__(image, guard=True, timeout_s=timeout_s, per_test_s=int(timeout_s), cpus=cpus, memory=memory, parallel=parallel,
                         name_prefix=name_prefix)
        self.profile, self.tmpfs = profile, tmpfs

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "lang": self.profile.name, "read_only": False, "tree": "copied in from a read-only mount"}

    def script(self, tree: Path, targets: Sequence[str], nonce: str) -> str:
        body = self.profile.run_script(Path(tree), list(targets))
        return (f"set -u\nO=/tmp/crl-{nonce}; mkdir -p \"$O\"; : > \"$O/report\"\n"
                "rm -rf /w 2>/dev/null; mkdir -p /w && cp -a /src/. /w/ && cd /w || { echo CRUCIBLE-INFRA-COPY; exit 125; }\n"
                f"( {body}\n) > \"$O/out\" 2>&1\n"
                "kill -9 -1 2>/dev/null; sleep 0.2\n"
                "tail -c 4000 \"$O/out\"\n"
                f"printf '\\n{BEGIN}\\n'; cat \"$O/report\"; printf '\\n{END}\\n'\n")

    async def run(self, tree: Path, targets: Sequence[str]) -> TestRun:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.parallel)
        nonce = uuid.uuid4().hex[:12]
        name = f"{self.name_prefix}{nonce}"
        tree = Path(tree).resolve()
        cmd = ["docker", "run", "--rm", "--name", name, "--label", "crucible.lane=debug", "--network", "none", "--tmpfs", f"/tmp:rw,exec,size={self.tmpfs}",
               "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", "2048", "-v", f"{tree}:/src:ro", "-e", "HOME=/tmp", self.image,
               "sh", "-c", self.script(tree, targets, nonce)]
        async with self._sem:
            self.runs += 1
            t0 = time.monotonic()
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                raw, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
            except TimeoutError:
                killer = await asyncio.create_subprocess_exec("docker", "rm", "-f", name, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await killer.wait()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), 30)
                return TestRun({}, "ok", None, time.monotonic() - t0, "", why=f"timeout after {self.timeout_s}s")
            secs = time.monotonic() - t0
        text = raw.decode("utf-8", "replace")
        blocks = _BLOCK.findall(text)
        if proc.returncode in (125, 126, 127) or (not blocks and "docker:" in text[:400].lower()):
            return TestRun({}, "masked", proc.returncode, secs, text[-3000:], why=f"sandbox did not start (exit {proc.returncode})")
        tail = _BLOCK.sub("", text)
        if k := killed(tail):  # killed from outside the tests (executable.KILLED): unmeasured, never the unreached tests failing
            return TestRun({}, "masked", proc.returncode, secs, tail[-3000:], why=f"the test run was killed ({k})")
        if not blocks:
            return TestRun({}, "ok", proc.returncode, secs, tail[-4000:], why="no report (the run ended early)")
        res, notes = self.profile.parse_report(blocks[-1], tree, list(targets), tail)
        why = "; ".join(notes)[:500]
        if any(n.startswith("killed:") for n in notes):  # the profile saw a test binary killed from outside: unmeasured
            return TestRun({}, "masked", proc.returncode, secs, tail[-3000:], why=f"the test run was killed ({why})")
        return TestRun(res, "ok", proc.returncode, secs, tail[-4000:], why=why)


def lang_sandbox_factory(*, parallel: int = 3, timeout_s: float = 900.0, cpus: str = "2", memory: str = "2g", name_prefix: str = f"{PREFIX}grade-"
                         ) -> Callable[[str], LangSandbox]:
    """One sandbox per World image, the profile read from the image tag (``crucible-lh-d:<lang>-...``), one shared
    concurrency limit (``windows.sandbox_factory``'s contract)."""
    from crucible.factory import langs

    if not name_prefix.startswith(PREFIX):
        raise ValueError(f"debug-lane containers are named {PREFIX}*, got {name_prefix!r}")
    sem = asyncio.Semaphore(parallel)
    made: dict[str, LangSandbox] = {}

    def get(image: str) -> LangSandbox:
        if image not in made:
            lang = image.split(":", 1)[1].split("-", 1)[0] if image.startswith(IMAGE_REPO + ":") else ""
            prof = langs.get(lang)
            sb = LangSandbox(image, profile=prof, name_prefix=name_prefix, timeout_s=max(timeout_s, prof.timeout_s), cpus=cpus, memory=memory,
                             parallel=parallel, tmpfs=prof.tmpfs)
            sb._sem = sem
            made[image] = sb
        return made[image]

    return get


# ---- leak probes -----------------------------------------------------------------------------------------------------------
LEAK_SCRIPT = r"""set -u
FETCH_RC=1; ( eval "$LH_FETCH" ) >/tmp/fetch.log 2>&1; FETCH_RC=$?
echo "PIP_RC=$FETCH_RC"
echo "OWN_COPIES=$( (eval "$LH_OWN") 2>/dev/null | head -5 | tr '\n' ' ')"
HITS=""
if [ -s /probe/names ]; then
  NEED=$(grep -c . /probe/names); [ "$NEED" -gt 2 ] && NEED=2
  set -f
  for f in $(grep -rIlF -f /probe/names $LH_INCLUDE / --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev --exclude-dir=probe 2>/dev/null | head -200); do
    n=$(grep -oF -f /probe/names "$f" 2>/dev/null | sort -u | wc -l)
    [ "$n" -ge "$NEED" ] && HITS="$HITS $f"
  done
fi
echo "NAME_HITS=$(echo $HITS | cut -c1-600)"
set +f
cd /probe/ws && echo "GIT_COMMITS=$(git log --all --format=%H | wc -l)" && echo "GIT_TAGS=$(git tag | tr '\n' ' ')"
"""


async def lang_leak_probes(image: str, base_dir: Path, *, profile: Any, new_test_names: set[str], package: str | None = None,
                           name_prefix: str = f"{PREFIX}leak-", **_: Any) -> dict[str, Any]:
    """The leak probes (module doc) for ``profile``'s World ``image``; the ``windows_world.leak_probes`` result shape
    (``{probes, ok, evidence}``). ``package``: the project's registry name (default: the profile reads it from the
    tree)."""
    if not name_prefix.startswith("crucible-lh-"):
        raise ValueError(f"World containers are named crucible-lh-*, got prefix {name_prefix!r}")
    project = package or profile.image_info(Path(base_dir)).get("project")
    with tempfile.TemporaryDirectory(prefix="lhd-leak-") as td:
        probe = Path(td)
        seed_workspace(base_dir, probe / "ws")
        (probe / "names").write_text("".join(profile.leak_needle(n) + "\n" for n in sorted(new_test_names)))
        include = " ".join(f"--include={g}" for g in profile.leak_globs)  # word-split unquoted under ``set -f``
        name = f"{name_prefix}{uuid.uuid4().hex[:10]}"
        args = ["run", "--rm", "--name", name, "--label", "crucible.lane=debug", "--network", "none", "-v", f"{probe.resolve()}:/probe:ro",
                "-e", f"LH_FETCH={profile.fetch_probe(project)}", "-e", f"LH_OWN={profile.own_copy_probe(Path(base_dir), project)}",
                "-e", f"LH_INCLUDE={include}", image, "sh", "-c", LEAK_SCRIPT]
        r = await asyncio.to_thread(docker, *args, timeout=900)
    if "PIP_RC" not in r.stdout:
        return {"probes": {}, "ok": False, "evidence": {"error": (r.stderr or r.stdout)[-500:]}}
    fields = parse_leak(r.stdout)
    out = judge_leak(fields, version=None)
    out["probes"].pop("pip_download_b_fails", None)
    out["probes"].pop("no_b_distribution_file", None)
    out["probes"]["fetch_project_fails"] = "pass" if fields.get("PIP_RC") not in (None, "", "0") else "fail"
    out["probes"]["no_installed_project_source"] = "pass" if not fields.get("OWN_COPIES") else "fail"
    out["ok"] = all(v in ("pass", "n/a") for v in out["probes"].values())
    out["lang"] = profile.name
    return out


async def world_setup(container: str, ws: Path, profile: Any) -> None:
    """The profile's per-World step after ``windows_world.open_world`` (node: the dependency tree linked into the
    workspace and excluded from git, so the harness never collects it)."""
    cmd = profile.world_setup_cmd(str(ws))
    if not cmd:
        return
    r = await asyncio.to_thread(subprocess.run, ["docker", "exec", container, "sh", "-c", cmd], capture_output=True, text=True, check=False, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"World setup failed: {r.stderr[-300:]}")


def own_images() -> list[str]:
    r = docker("images", IMAGE_REPO, "--format", "{{.Repository}}:{{.Tag}}", timeout=60)
    return [x for x in r.stdout.split() if x.startswith(IMAGE_REPO + ":")]
