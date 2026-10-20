"""Stitched windows (swarm-pipeline/00 §2, §7.4, §15 P5): a host repository and a consumer that depends on it, in one
workspace. The host change is a real release window of the host; the consumer change is the consumer's own commit that
moved to that release. Consumer milestones need the host change: long because the work crosses repositories.

**Pairs** (``requirement_bumps``): a consumer first-parent commit that raises the lower bound of its requirement on the
host (``host>=V``; or lifts the upper bound to admit a new series) in ``pyproject.toml`` / ``setup.py`` / ``setup.cfg`` /
``requirements*.txt``. The host window is (the release before V, V] on the host's first-parent path (at most ``max_units``
qualifying units); the consumer window is that commit (C0 = its parent, C1 = it, or the first commit after it, within
five, that changes the consumer's source or tests). Pairs come from the training repositories' own requirements and from
``sources.scouts`` (consumer records, ``evidence.host``).

**The workspace** is one tree: ``host/`` and ``consumer/``. Its synthetic history (``base.init_mono``: a repository
borrowing both clones' objects) is A' = (host at A, consumer at C0), then one commit per host unit (host at the unit,
consumer at C0), then one per consumer unit (host at B, consumer at the unit). ``windows.validate_window`` runs on it
unchanged, strict imports, with:
- ``StitchImageBook``: the World image installs the host from A (its version pinned: a pip constraint keeps the
  consumer's install from pulling a newer host, and the build fails if site-packages ends with any other host version, so
  B's code is never in the image) and the consumer from C0;
- ``StitchSandbox``: pytest runs once per side (``cd host`` / ``cd consumer``, so each side's own configuration and
  conftest apply), with ``PYTHONPATH`` = both workspaces (host first), and test ids prefixed with the side.

**Cross-repository evidence** (``host_required``, executed after validation): the consumer tests the task checks run on
(host at A, consumer at C1) and on (host at B, consumer at C1). Those failing on the first and passing on the second are
consumer tests that require the host change; ``cross_edges`` are the prober's executed edges from a consumer milestone to
a host milestone. A stitched window is *cross-dependent* when either exists.

**The task** (``StitchTask``): ``simplify('host_change')`` pre-applies every host unit (``base.GeneratedTask.
simplify_units``), leaving the consumer milestones; ``'locate'`` and ``'prerequisites'`` as for a window. The World opens
with both workspaces on ``PYTHONPATH``.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import re
import shlex
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.generators.base import (
    GRADE_PREFIX,
    WORLD_PREFIX,
    AuthorStyle,
    Candidate,
    Funnel,
    GeneratedTask,
    GenLane,
    changed_paths,
    combine_trees,
    commit_meta,
    drop_crlf_prose,
    git,
    init_mono,
    out,
    show,
    write_chain,
)
from crucible.factory.lanes.executable import TestRun, is_test_path
from crucible.factory.lanes.windows import (
    COLLECT_ALL,
    NamedSandbox,
    WindowStore,
    normalize_ids,
    passed_group,
    validate_window,
)
from crucible.factory.lanes.windows_git import RepoGit, Unit, Window, _bodies, first_parent_units, version_tags
from crucible.factory.lanes.windows_world import (
    BASE_IMAGE,
    BASE_PINS,
    DROP_INSTALLED_SOURCES,
    EXTRAS,
    EXTRAS_FALLBACK,
    ImageBook,
    docker,
    ensure_base,
    project_name,
    requirement_names,
    requirement_sources,
    sha256_hex,
    world_needs,
)
from crucible.factory.sources.lease import EvalLease
from crucible.factory.sources.records import SourceRecord

NAME = "stitched_window"
SIDES = ("host", "consumer")
PACKAGING_GLOBS = ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "requirements*.txt", "requirements/*.txt", "requirements/*.in")
STYLE = AuthorStyle(
    role="author",
    brief=("You write the request a maintainer sends a developer who works on two Python projects at once: a library (the host) "
           "and a project that depends on it (the consumer). The next host release must deliver some changes, and the consumer must "
           "move to that release. You get, per change id, the commit message, the public names it touched and which project it "
           "belongs to, plus the changelog lines and both projects' descriptions. Write one requirement per change id, each one or two "
           "plain sentences that state the observable behaviour of the public API (inputs, outputs, errors, formats) the change must "
           "have, naming the project it is about, never how to implement it. Never write file paths, module file names, test names, "
           "commit ids, issue or pull request numbers, links, version numbers, or process instructions. Return {opening, requirements: [{id, text}], "
           "closing}: opening is two or three sentences of context, closing is optional, requirements has exactly one entry per change "
           "id. Names listed in a change's must_name (or in must_name_any) are public names that must exist under exactly that name: "
           "write each one, verbatim, in the requirement of its change (must_name_any: in the one it fits)."),
    heading="What the two projects need:")


# ---- requirement bumps ------------------------------------------------------------------------------------------------------------
def _dist_rx(dist: str) -> re.Pattern[str]:
    name = re.sub(r"[-_.]+", "[-_.]", re.escape(dist).replace("\\-", "-").replace("\\.", ".").replace("\\_", "_"))
    return re.compile(rf"""(?i)(?:^|[\s"',\[(])({name})(?:\[[^\]]*\])?\s*((?:(?:[<>=!~]=?|===)\s*[\w.*+!-]+\s*,?\s*)+)""")


def spec_of(text: str, dist: str) -> str | None:
    """The version specifier of ``dist`` in requirement text (pyproject, setup.py, setup.cfg, requirements), first found."""
    m = _dist_rx(dist).search(text or "")
    return re.sub(r"\s+", "", m.group(2)).rstrip(",") if m else None


def _v(s: str) -> Any:
    from packaging.version import InvalidVersion, Version

    try:
        return Version(s.rstrip(".*"))
    except InvalidVersion:
        return None


def bounds(spec: str | None) -> tuple[Any, Any]:
    """``(lower, upper)`` versions of a specifier (``>=``/``==``/``~=``/``>`` and ``<``/``<=``)."""
    lo = hi = None
    for op, ver in re.findall(r"(===|==|>=|<=|~=|!=|>|<)\s*([\w.*+!-]+)", spec or ""):
        v = _v(ver)
        if v is None:
            continue
        if op in (">=", "==", "~=", ">", "===") and (lo is None or v > lo):
            lo = v
        if op in ("<", "<=") and (hi is None or v < hi):
            hi = v
    return lo, hi


def packaging_text(repo: Path, rev: str) -> str:
    files = out(git(repo, "ls-tree", "-r", "--name-only", rev, check=False)).splitlines()
    pick = [f for f in files if re.fullmatch(r"(pyproject\.toml|setup\.py|setup\.cfg|requirements[^/]*\.(txt|in)|requirements/[^/]+\.(txt|in))", f)]
    return "\n".join(show(repo, rev, f) or "" for f in pick)


def requirement_bumps(consumer: Path, dist: str, *, since: str = "2019-01-01") -> list[dict[str, Any]]:
    """Consumer first-parent commits that change ``dist``'s requirement so that a newer host release is required or newly
    admitted: ``{sha, parent, date, subject, old, new, target}``; ``target`` is the new lower bound, or the old upper
    bound when only the upper bound was lifted (the first release of the newly admitted series)."""
    r = git(consumer, "log", "--first-parent", f"--since={since}", "--format=%H", "-G", re.sub(r"[-_.]", "[-_.]", dist), "--",
            *PACKAGING_GLOBS, check=False)
    outp = []
    for sha in out(r).split():
        meta = commit_meta(consumer, sha)
        if not meta["parent"]:
            continue
        old, new = spec_of(packaging_text(consumer, meta["parent"]), dist), spec_of(packaging_text(consumer, sha), dist)
        if not new or old == new:
            continue
        (olo, ohi), (nlo, nhi) = bounds(old), bounds(new)
        target = None
        if nlo is not None and (olo is None or nlo > olo):
            target = nlo
        elif ohi is not None and (nhi is None or nhi > ohi):
            target = ohi
        if target is not None:
            outp.append({**meta, "old": old, "new": new, "target": str(target)})
    return outp


def imports_host(consumer: Path, rev: str, roots: Sequence[str]) -> list[str]:
    """Consumer source files (not tests) at ``rev`` that import one of the host's top-level modules: a pin in a
    requirements file is a consumer only when its code uses the host (a tox or docs pin is not)."""
    if not roots:
        return []
    rx = re.compile(rf"(?m)^\s*(?:from|import)\s+({'|'.join(re.escape(r) for r in roots)})(?:[.\s,]|$)")
    hits = []
    for p in out(git(consumer, "ls-tree", "-r", "--name-only", rev, check=False)).splitlines():
        if p.endswith(".py") and not is_test_path(p) and not p.startswith(("docs/", "doc/", "examples/")):
            txt = show(consumer, rev, p) or ""
            if rx.search(txt):
                hits.append(p)
                if len(hits) >= 5:
                    break
    return hits


def host_roots(host: Path, rev: str = "HEAD") -> list[str]:
    """The host's top-level import names at ``rev``: packages at the root or under ``src/``."""
    names = out(git(host, "ls-tree", "-r", "--name-only", rev, check=False)).splitlines()
    roots = set()
    for p in names:
        parts = p.split("/")
        if parts[0] == "src":
            parts = parts[1:]
        if len(parts) == 2 and parts[1] == "__init__.py" and parts[0].isidentifier():
            roots.add(parts[0])
        elif len(parts) == 1 and parts[0].endswith(".py") and parts[0][:-3].isidentifier() and parts[0][:-3] not in ("setup", "conftest", "noxfile"):
            roots.add(parts[0][:-3])
    return sorted(roots - {"tests", "test", "docs"})


def discover_pairs(clones: Mapping[str, Path], *, since: str = "2019-01-01") -> list[tuple[str, str]]:
    """``(consumer, host)`` among local clones: the consumer's history bumps its requirement on the host and its code
    imports the host."""
    dists = {n: project_name(Path(p)) or n for n, p in clones.items()}
    outp = []
    for c, cp in clones.items():
        for h, hp in clones.items():
            if c == h:
                continue
            bumps = requirement_bumps(Path(cp), dists[h], since=since)
            if bumps and imports_host(Path(cp), bumps[0]["sha"], host_roots(Path(hp))):
                outp.append((c, h))
    return outp


def host_window(host: Path, target: str, *, max_units: int = 25) -> dict[str, Any] | None:
    """(A, B] of the host: B = the release tag whose version equals ``target`` (``8.2`` matches ``8.2.0``), A = the newest
    earlier release on B's first-parent path; ``None`` when there is none with 1..``max_units`` qualifying units."""
    tv = _v(target)
    tags = [(t, _v(t[3])) for t in version_tags(host)]
    tags = [(t, v) for t, v in tags if v is not None]
    bs = [t for t, v in tags if v == tv]
    if not bs:
        return None
    b = bs[0]
    earlier = sorted([(v, t) for t, v in tags if v < tv and t[2] <= b[2]], key=lambda x: x[0], reverse=True)
    for _v_a, a in earlier[:4]:
        units = first_parent_units(host, a[1], b[1])
        q = sum(1 for u in units if u.qualifies)
        if units and 1 <= q <= max_units:
            return {"a": a[1], "a_ref": a[0], "a_date": a[2], "b": b[1], "b_ref": b[0], "b_date": b[2], "b_version": b[3], "units": len(units),
                    "qualifying": q}
    return None


def consumer_window(consumer: Path, bump: Mapping[str, Any], *, look: int = 5) -> dict[str, Any] | None:
    """C0 = the bump's parent; C1 = the bump, or the first of the next ``look`` first-parent commits that changes source or
    tests (the adaptation often lands just after the requirement)."""
    r = git(consumer, "rev-list", "--first-parent", "--reverse", "--ancestry-path", f"{bump['sha']}..HEAD", check=False)
    after = out(r).split()[:look]
    for sha in [bump["sha"], *after]:
        files = changed_paths(consumer, commit_meta(consumer, sha)["parent"], sha)
        if any(p.endswith(".py") for p in files):
            return {"c0": bump["parent"], "c1": sha, "bump": bump["sha"]}
    return {"c0": bump["parent"], "c1": bump["sha"], "bump": bump["sha"]}


# ---- the World ------------------------------------------------------------------------------------------------------------------
def stitch_dockerfile(host_dist: str, host_version: str, needs: Mapping[str, Mapping[str, Any]], reqs: Mapping[str, Sequence[str]],
                      groups: Mapping[str, Sequence[str]], packaged: Mapping[str, bool], *, consumer_dist: str | None = None) -> str:
    """Host from A (``SETUPTOOLS_SCM_PRETEND_VERSION`` = A's version), then the consumer from C0 under a pip constraint that
    pins the host to the installed version; the build fails when site-packages ends with another host version (B's code
    must never be in the image). Phase A (``windows_world.EXTRAS_FALLBACK`` fix 3 for two projects): both sides are
    installed again without dependencies last, and both installed copies keep only their metadata (an image shared by
    two windows of one pair held the consumer of whichever built it first)."""
    apt = sorted({p for s in SIDES for p in (needs[s].get("apt") or [])})
    pip_build = sorted({p for s in SIDES for p in (needs[s].get("pip_build") or [])})
    lines = [f"FROM {BASE_IMAGE}"]
    if apt:
        lines.append("RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends " + " ".join(shlex.quote(p) for p in apt)
                     + " >/dev/null && rm -rf /var/lib/apt/lists/*")
    if pip_build:
        lines.append("RUN pip install --no-cache-dir " + " ".join(shlex.quote(p) for p in pip_build))
    lines += ["COPY a/host /opt/lh/h", "COPY a/consumer /opt/lh/c", "WORKDIR /opt/lh/h"]
    ex_h = ",".join(dict.fromkeys([*EXTRAS, *(needs["host"].get("extras") or [])]))
    pv = shlex.quote(host_version)
    if packaged["host"]:
        lines.append(f'RUN SETUPTOOLS_SCM_PRETEND_VERSION={pv} pip install --no-cache-dir ".[{ex_h}]" || SETUPTOOLS_SCM_PRETEND_VERSION={pv} pip install --no-cache-dir .')
    probe = f"import importlib.metadata as m; print({host_dist!r} + '==' + m.version({host_dist!r}))"
    lines.append(f"RUN python -c {shlex.quote(probe)} > /opt/lh/cons.txt && cat /opt/lh/cons.txt")
    lines.append("ENV PIP_CONSTRAINT=/opt/lh/cons.txt")
    lines.append("WORKDIR /opt/lh/c")
    ex_c = ",".join(dict.fromkeys([*EXTRAS, *(needs["consumer"].get("extras") or [])]))
    if packaged["consumer"]:
        lines.append(f'RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --no-cache-dir ".[{ex_c}]" || SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install '
                     f'--no-cache-dir . || SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --no-cache-dir --no-deps .')
    for s in SIDES:
        if groups.get(s):
            lines.append("RUN pip install --no-cache-dir " + " ".join(shlex.quote(g) for g in groups[s]) + " || true")
        for n in reqs.get(s) or []:
            if re.sub(r"[-_.]+", "-", n.lower()) != re.sub(r"[-_.]+", "-", host_dist.lower()):
                lines.append(f"RUN pip install --no-cache-dir {shlex.quote(n)} || true")
        for n in needs[s].get("import_deps") or []:
            lines.append(f"RUN pip install --no-cache-dir {shlex.quote(n)} || true")
    if packaged["host"]:  # each side last, without dependencies, then with its metadata only
        lines.append(f"RUN cd /opt/lh/h && SETUPTOOLS_SCM_PRETEND_VERSION={pv} pip install --no-cache-dir --no-deps --force-reinstall .")
    if packaged["consumer"]:
        lines.append("RUN cd /opt/lh/c && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --no-cache-dir --no-deps --force-reinstall .")
    for dist in (host_dist if packaged["host"] else None, consumer_dist if packaged["consumer"] else None):
        if dist and dist != "none-such":
            lines.append(f"RUN python -c {shlex.quote(DROP_INSTALLED_SOURCES)} {shlex.quote(dist)}")
    lines.append("RUN pip install --no-cache-dir " + " ".join(BASE_PINS))
    check = f"import importlib.metadata as m; v = m.version({host_dist!r}); assert v == open('/opt/lh/cons.txt').read().strip().split('==')[1], v"
    lines += [f"RUN python -c {shlex.quote(check)}", "ENV PIP_CONSTRAINT=", "WORKDIR /w", "RUN rm -rf /opt/lh /root/.cache"]
    return "\n".join(lines) + "\n"


class StitchImageBook(ImageBook):
    """World images of stitched workspaces (``host/`` + ``consumer/``), keyed by both sides' packaging and the recipe."""

    def __init__(self, *, host_versions: Mapping[str, str] | None = None, parallel: int = 2):
        super().__init__(parallel=parallel)
        self.host_versions = dict(host_versions or {})  # repo (pair id) -> A's host version

    def tag_for(self, repo: str, base_dir: Path) -> tuple[str, dict[str, Any]]:
        from crucible.factory.lanes.windows_git import packaging_hash
        from crucible.factory.lanes.windows_world import IMAGE_RECIPE, PACKAGING

        info: dict[str, Any] = {"needs": {}, "reqs": {}, "groups": {}, "packaged": {}}
        h = IMAGE_RECIPE + "stitch2" + EXTRAS_FALLBACK  # stitch2 (Phase A): both installed copies keep only their metadata
        for s in SIDES:
            t = Path(base_dir) / s
            files, groups = requirement_sources(t)
            info["reqs"][s] = sorted({n for f in files for n in requirement_names((t / f).read_text(errors="replace"))})
            info["groups"][s] = groups
            info["needs"][s] = world_needs(t)
            info["packaged"][s] = any((t / f).exists() for f in PACKAGING)
            key = {k: info["needs"][s][k] for k in ("apt", "pip_build", "extras", "import_deps")}
            h += packaging_hash(t, [*PACKAGING, *files]) + json.dumps(key, sort_keys=True)
        info["project"] = project_name(Path(base_dir) / "host")
        info["consumer_project"] = project_name(Path(base_dir) / "consumer")
        info["projects"] = [x for x in (info["project"], info["consumer_project"]) if x]  # the leak probe's installed copies
        info["host_version"] = self.host_versions.get(repo, "0.0.0")
        h = sha256_hex(h + info["host_version"])
        slug = re.sub(r"[^a-z0-9_.-]+", "-", repo.lower())
        return f"{WORLD_PREFIX}w:stitch-{slug}-{h[:12]}", info

    @staticmethod
    def _build(tag: str, base_dir: Path, info: dict[str, Any]) -> tuple[bool, str]:
        ensure_base()
        with tempfile.TemporaryDirectory(prefix="p5img-") as td:
            ctx = Path(td)
            shutil.copytree(base_dir, ctx / "a", symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
            (ctx / "Dockerfile").write_text(stitch_dockerfile(info["project"] or "none-such", info["host_version"], info["needs"], info["reqs"],
                                                              info["groups"], info["packaged"], consumer_dist=info.get("consumer_project")))
            r = docker("build", "-q", "-t", tag, str(ctx), timeout=1800)
            return r.returncode == 0, (r.stdout + r.stderr)


_SIDE: contextvars.ContextVar[str] = contextvars.ContextVar("stitch_side", default="")
PYTHONPATH = "/w/host/src:/w/host:/w/consumer/src:/w/consumer"


class StitchSandbox(NamedSandbox):
    """pytest once per side of the workspace (``-w /w/host`` / ``-w /w/consumer``: each side's own pytest configuration
    and conftest), ``PYTHONPATH`` = both sides, host first; ids come back prefixed with their side."""

    def command(self, tree: Path, name: str, nonce: str, targets: Sequence[str], out: Path | None = None) -> list[str]:
        cmd = super().command(tree, name, nonce, targets, out)
        side = _SIDE.get()
        if side:
            for i, c in enumerate(cmd):
                if c == "-w" and i + 1 < len(cmd) and cmd[i + 1] == "/w":
                    cmd[i + 1] = f"/w/{side}"
                elif c.startswith("PYTHONPATH="):
                    cmd[i] = f"PYTHONPATH={PYTHONPATH}" + (":/opt/m17" if "/opt/m17" in c else "")
        return cmd

    async def run(self, tree: Path, targets: Sequence[str]) -> TestRun:
        groups: dict[str, list[str]] = {}
        for t in targets:
            side, _, rest = t.partition("/")
            if side in SIDES and rest:
                groups.setdefault(side, []).append(rest)
        results: dict[str, str] = {}
        tails, infra, why, secs, code = [], "ok", "", 0.0, 0
        for side, ts in groups.items():
            token = _SIDE.set(side)
            try:
                tr = await super().run(tree, ts)
            finally:
                _SIDE.reset(token)
            secs += tr.seconds
            tails.append(f"[{side}] {tr.tail[-1500:]}")
            if tr.infra != "ok":
                infra, why = tr.infra, f"{side}: {tr.why}"
            elif tr.why and not why:
                why = f"{side}: {tr.why}"
            code = code or (tr.exit_code or 0)
            results.update({f"{side}/{k}": v for k, v in tr.results.items()})
        return TestRun(results, infra, code, secs, "\n".join(tails)[-4000:], why=why)


def stitch_sandboxes(*, parallel: int = 3, timeout_s: float = 900.0, memory: str = "1536m", name_prefix: str = GRADE_PREFIX) -> Callable[[str], Any]:
    sem = asyncio.Semaphore(parallel)
    made: dict[str, Any] = {}

    def get(image: str) -> Any:
        if image not in made:
            sb = StitchSandbox(image, guard=True, timeout_s=timeout_s, per_test_s=60, cpus="2", memory=memory, parallel=parallel, extra_args=COLLECT_ALL,
                               name_prefix=name_prefix)
            sb._sem = sem
            made[image] = sb
        return made[image]

    return get


# ---- the task ------------------------------------------------------------------------------------------------------------------
class StitchTask(GeneratedTask):
    generator = NAME

    def units_of(self, side: str) -> set[str]:
        return {s for s, v in ((self.window.get(NAME) or {}).get("unit_side") or {}).items() if v == side}

    def simplify(self, kind: str, *, milestones: Sequence[str] | None = None, root: Path | None = None) -> Any:
        """``host_change``: every host unit pre-applied (the consumer milestones remain); ``locate`` / ``prerequisites``:
        as for a mined window."""
        if kind == "host_change":
            return self.simplify_units(self.units_of("host"), kind=kind, root=root)
        if kind == "locate":
            t = self._simplify_locate(milestones=milestones)
        elif kind == "prerequisites":
            t = super().simplify("prerequisites", milestones=milestones, root=root)
        else:
            raise ValueError(f"a stitched window simplifies by host_change | locate | prerequisites, not {kind!r}")
        t.__class__ = type(self)
        return t

    async def open(self, workdir: Path) -> Any:
        from crucible.factory.lanes.windows_world import open_world

        return await open_world(image=self.world_image or self.window["world"]["image"], base_dir=self.base_dir, workdir=Path(workdir),
                                instruction=self.instruction, name_prefix=self.lane.world_prefix, workspace_mode=self.workspace_mode,
                                pythonpath="{ws}/host/src:{ws}/host:{ws}/consumer/src:{ws}/consumer")


# ---- the generator --------------------------------------------------------------------------------------------------------------
class Stitch:
    """Stitched windows over (consumer, host) clone pairs (``Generator``)."""

    name = NAME

    def __init__(self, root: Path, clones: Path, *, funnel: Funnel, lease: EvalLease, extra_clones: Path | None = None, images: Any = None,
                 sandboxes: Any = None, log: Callable[[dict], None] | None = None, max_units: int = 25, per_pair: int = 2, since: str = "2019-01-01"):
        self.root = Path(root)
        self.clones_dirs = [Path(clones), *([Path(extra_clones)] if extra_clones else [])]
        self.funnel, self.lease, self.log = funnel, lease, log
        self.images = StitchImageBook(parallel=2)  # the shared ImageBook builds single-repo Worlds: a stitched World is its own
        self.sandboxes = stitch_sandboxes()  # per-side pytest; the shared sandboxes run one tree root
        self.s3_sandboxes = stitch_sandboxes(parallel=4, name_prefix=f"{WORLD_PREFIX}s3-")
        self.max_units, self.per_pair, self.since = max_units, per_pair, since
        self.store = WindowStore(self.root / "store" / "train", split="train")
        self.monos: dict[str, Path] = {}
        self._lane: GenLane | None = None
        self.records: list[SourceRecord] = []
        meta = self.root / "pairs.json"
        self.pair_meta: dict[str, Any] = json.loads(meta.read_text()) if meta.exists() else {}
        for pid, m in self.pair_meta.items():
            self.monos[pid] = Path(m["mono"])
            self.images.host_versions.update(m.get("host_versions") or {})

    def clone_of(self, name: str) -> Path:
        for d in self.clones_dirs:
            if (d / name / ".git").exists():
                return d / name
        raise FileNotFoundError(f"no clone of {name} under {self.clones_dirs}")

    def lane(self) -> GenLane:
        if self._lane is None:
            self._lane = GenLane(self.store, generator=NAME, task_cls=StitchTask, repo_dirs={k: str(v) for k, v in self.monos.items()},
                                 sandboxes=self.sandboxes)
        return self._lane

    def task(self, record_id: str, *, authored: Mapping[str, Any] | None = None) -> StitchTask:
        return self.lane().task(record_id, authored=authored)  # type: ignore[return-value]

    def _save_meta(self) -> None:
        (self.root / "pairs.json").parent.mkdir(parents=True, exist_ok=True)
        (self.root / "pairs.json").write_text(json.dumps(self.pair_meta, indent=1, sort_keys=True))

    async def candidates_for(self, consumer: str, host: str, *, sources: Sequence[Mapping[str, Any]] = ()) -> list[Candidate]:
        pid = f"{consumer}--{host}".replace("/", "_")
        for name in (consumer, host):
            self.lease.guard_repo(name)
        try:
            cdir, hdir = self.clone_of(consumer), self.clone_of(host)
        except FileNotFoundError as e:
            self.funnel.note(NAME, pid, "S1_source", False, f"no clone: {e}")
            return []
        dist = project_name(hdir) or host
        bumps = await asyncio.to_thread(requirement_bumps, cdir, dist, since=self.since)
        roots = await asyncio.to_thread(host_roots, hdir)
        bumps = [b for b in bumps if await asyncio.to_thread(imports_host, cdir, b["sha"], roots)]
        self.funnel.note(NAME, pid, "S1_source", bool(bumps), "" if bumps else f"no requirement bump on {dist} in {consumer}", bumps=len(bumps))
        outp: list[Candidate] = []
        for bump in sorted(bumps, key=lambda x: x["date"], reverse=True):
            if len(outp) >= self.per_pair:
                break
            hw = await asyncio.to_thread(host_window, hdir, bump["target"], max_units=self.max_units)
            cid = f"st-{pid}-{bump['sha'][:8]}"
            if hw is None:
                self.funnel.note(NAME, cid, "S1_executed", False, f"no host release window ending at {bump['target']} (1..{self.max_units} qualifying units)")
                continue
            cw = await asyncio.to_thread(consumer_window, cdir, bump)
            rec = SourceRecord("consumer", consumer, sha=bump["sha"], title=bump["subject"], created=bump["date"],
                               evidence={"host": host, "dist": dist, "old": bump["old"], "new": bump["new"], "target": bump["target"], "executed": False})
            self.lease.take(rec)
            self.records.append(rec)
            if not rec.lease["ok"]:
                self.funnel.note(NAME, cid, "S1_executed", False, rec.lease["reason"])
                continue
            self.funnel.note(NAME, cid, "S1_executed", True, "", host_window=hw, consumer_window=cw)
            outp.append(Candidate(cid, NAME, pid, sources=[rec.to_json(), *[dict(s) for s in sources]],
                                  params={"consumer": consumer, "host": host, "dist": dist, "host_window": hw, "consumer_window": cw, "bump": bump}))
        return outp

    async def candidates(self, pairs: Sequence[tuple[str, str]] = ()) -> list[Candidate]:
        outs = await asyncio.gather(*[self.candidates_for(c, h) for c, h in pairs])
        return [c for cs in outs for c in cs]

    def synthesize(self, cand: Candidate) -> tuple[Window, dict[str, Any]]:
        p = cand.params
        hdir, cdir = self.clone_of(p["host"]), self.clone_of(p["consumer"])
        pid = cand.repo
        mono = init_mono(self.root / "mono" / pid, [hdir, cdir])
        self.monos[pid] = mono
        hw, cw = p["host_window"], p["consumer_window"]
        hunits = first_parent_units(hdir, hw["a"], hw["b"])
        _bodies(hdir, hunits)
        cunits = first_parent_units(cdir, cw["c0"], cw["c1"])
        _bodies(cdir, cunits)
        trees = [combine_trees(mono, {"host": hw["a"], "consumer": cw["c0"]})]
        msgs = [f"stitch start: {p['host']} {hw['a_ref']} + {p['consumer']} at {cw['c0'][:10]}"]
        dates = [hw["a_date"]]
        side = []
        for u in hunits:
            trees.append(combine_trees(mono, {"host": u.sha, "consumer": cw["c0"]}))
            msgs.append(f"[{p['host']}] {u.subject}\n\n{u.body}")
            dates.append(u.date)
            side.append("host")
        for v in cunits:
            trees.append(combine_trees(mono, {"host": hw["b"], "consumer": v.sha}))
            msgs.append(f"[{p['consumer']}] {v.subject}\n\n{v.body}")
            dates.append(v.date)
            side.append("consumer")
        shas = write_chain(mono, trees, messages=msgs, dates=dates, parent=None, ref=f"refs/crucible/stitch/{cand.id}")
        units = first_parent_units(mono, shas[0], shas[-1])
        _bodies(mono, units)
        win = Window(id=cand.id, repo=pid, repo_dir=str(mono), kind="stitched", a=shas[0], b=shas[-1], a_ref=f"{hw['a_ref']}+{cw['c0'][:8]}",
                     b_ref=f"{hw['b_ref']}+{cw['c1'][:8]}", a_date=hw["a_date"], b_date=max(hw["b_date"], cunits[-1].date if cunits else hw["b_date"]),
                     b_version=hw["b_version"], units=units)
        info = {"unit_side": {u.sha: s for u, s in zip(units, side, strict=True)}, "host": p["host"], "consumer": p["consumer"], "dist": p["dist"],
                "host_window": hw, "consumer_window": cw, "pair": pid,
                "orig_units": {u.sha: o.sha for u, o in zip(units, [*hunits, *cunits], strict=True)}}
        a_version = re.sub(r"^[^\d]*", "", hw["a_ref"]) or "0.0.0"
        self.images.host_versions[pid] = a_version
        self.pair_meta[pid] = {"mono": str(mono), "host_versions": {pid: a_version}}
        self._save_meta()
        return win, info

    async def host_required(self, rec: Mapping[str, Any], mono: Path) -> dict[str, Any]:
        """The consumer tests the task checks, on (host A, consumer C1) and (host B, consumer C1): those failing on the first
        and passing on the second require the host change (executed)."""
        info = rec[NAME]
        tests = sorted({t for m in rec.get("milestones") or [] for t in m["tests"] if t.startswith("consumer/")})
        if not tests:
            return {"tests": [], "required": [], "why": "no consumer test among the milestones"}
        files = sorted({t.split("::", 1)[0] for t in tests})
        hw, cw = info["host_window"], info["consumer_window"]
        sb = self.sandboxes(rec["world"]["image"])
        res = {}
        for label, host_rev in (("host_a", hw["a"]), ("host_b", hw["b"])):
            tree = combine_trees(mono, {"host": host_rev, "consumer": cw["c1"]})
            with tempfile.TemporaryDirectory(prefix="p5st-") as td:
                dest = Path(td) / "t"
                ok, bad = await asyncio.to_thread(RepoGit(mono).build, tree, [], dest)
                if not ok:
                    return {"tests": tests, "required": [], "why": f"tree does not build: {bad}"}
                tr = await sb.run(dest, files)
            if tr.infra != "ok":
                return {"tests": tests, "required": [], "why": f"masked: {tr.why}"}
            res[label] = normalize_ids(tr.results, files)
        req = [t for t in tests if res["host_b"].get(t) == "passed" and res["host_a"].get(t) != "passed"]
        return {"tests": tests, "required": req, "host_a_pass": sum(1 for t in tests if res["host_a"].get(t) == "passed"),
                "host_b_pass": sum(1 for t in tests if res["host_b"].get(t) == "passed"), "why": ""}

    async def split_observations(self, rec: dict[str, Any], mono: Path) -> dict[str, Any]:
        """Consumer tests a host unit makes pass (prefix ownership put them in that host milestone) become their own
        milestone ``<id>c`` (unit None, ``observes`` the host unit), with an executed edge to the host milestones they
        need: they fail at A', and ddmin over the host units up to the owner finds the minimal set that makes them pass
        (each unit of it that owns a host milestone is a ``requires``, evidence ``executed``). ``00`` §2: milestones = host
        F2P groups + consumer tests that require the host change."""
        from crucible.factory.lanes.windows import ddmin, grader_tests

        side = rec[NAME]["unit_side"]
        units = [Unit(**u) for u in rec["units"]]
        idx = {u.sha: i for i, u in enumerate(units)}
        repo = RepoGit(mono)
        sb = self.sandboxes(rec["world"]["image"])
        tests_g = grader_tests(rec)[0]
        out_ms, made = [], []
        by_unit = {m["unit"]: m for m in rec["milestones"]}
        for m in rec["milestones"]:
            out_ms.append(m)
            if side.get(m["unit"]) != "host":
                continue
            cons = [t for t in m["tests"] if t.startswith("consumer/")]
            host = [t for t in m["tests"] if not t.startswith("consumer/")]
            if not cons or not host:
                continue
            m["tests"] = host
            files = sorted({t.split("::", 1)[0] for t in cons})
            cache: dict[tuple[int, ...], bool] = {}

            async def ok(subset: list[int], cons: list[str] = cons, files: list[str] = files, cache: dict = cache) -> bool:
                key = tuple(sorted(subset))
                if key not in cache:
                    with tempfile.TemporaryDirectory(prefix="p5sp-") as td:
                        t = Path(td) / "t"
                        built, _ = await asyncio.to_thread(repo.build, rec["a"], [units[k] for k in key], t, source_only=True)
                        if not built:
                            cache[key] = False
                        else:
                            for pth, txt in tests_g.items():
                                (t / pth).parent.mkdir(parents=True, exist_ok=True)
                                (t / pth).write_text(txt)
                            tr = await sb.run(t, files)
                            cache[key] = tr.infra == "ok" and passed_group(normalize_ids(tr.results, files), cons)
                return cache[key]

            earlier = [k for k in range(idx[m["unit"]] + 1) if side.get(units[k].sha) == "host"]
            cm: dict[str, Any] = {"id": f"{m['id']}c", "unit": None, "index": m["index"], "subject": f"{rec[NAME]['consumer']} observes: {m['subject']}",
                                  "tests": sorted(cons), "observes": m["unit"], "requires": [], "requires_units": [], "edges": {}, "requires_evidence": {}}
            if not await ok([]) and await ok(earlier):
                need = await ddmin(earlier, ok)
                req = [by_unit[units[k].sha]["id"] for k in need if units[k].sha in by_unit]
                cm.update(requires=req, requires_units=[units[k].sha for k in need], edges={units[k].sha: "executed" for k in need},
                          requires_evidence=dict.fromkeys(req, "executed"), probe="minimized")
            else:
                cm["probe"] = "alone" if cache.get(()) else "prefix_fails"
            out_ms.append(cm)
            made.append(cm["id"])
        rec["milestones"] = out_ms
        rec["dependency_edges"] = sum(len(x.get("requires") or []) for x in out_ms)
        rec["independent"] = rec["dependency_edges"] == 0
        return {"split": made, "edges": {x["id"]: x["requires"] for x in out_ms if x["id"] in made}}

    async def consumer_mutants(self, rec: dict[str, Any]) -> list[dict[str, Any]]:
        """Per consumer observation milestone: B with the host unit it needs reverted (the host milestone's revert mutant)
        must fail it."""
        from crucible.factory.lanes.windows import grader_tests, run_tests

        by_ms = {m["milestone"]: m for m in rec.get("mutants") or [] if m.get("solution")}
        by_id = {m["id"]: m for m in rec["milestones"]}
        sb = self.sandboxes(rec["world"]["image"])
        base = self.store.dir(rec["id"]) / "base"
        outp = []
        for m in rec["milestones"]:
            if m.get("unit") is not None or not m.get("requires"):
                continue
            src = next((by_ms[r] for r in m["requires"] if r in by_ms), None)
            if src is None:
                outp.append({"id": f"revert_{m['id']}", "milestone": m["id"], "status": "no_host_mutant"})
                continue
            files = sorted({t.split("::", 1)[0] for t in m["tests"]})
            tr = await run_tests(sb, base, src["solution"]["files"], grader_tests(rec)[0], files, rec.get("tests_deleted") or ())
            failed = tr.infra == "ok" and not passed_group(tr.results, m["tests"])
            outp.append({"id": f"revert_{m['id']}", "milestone": m["id"], "status": "valid" if failed else ("masked" if tr.infra != "ok" else "uncaught"),
                         "solution": src["solution"], "expected": {m["id"]: False, by_id[src["milestone"]]["id"]: False}, "via": src["id"]})
        return outp

    async def build(self, cand: Candidate) -> dict[str, Any]:
        from crucible.factory.lanes.windows import mined_of, probe_dependencies, revert_mutants

        for name in (cand.params["consumer"], cand.params["host"]):
            self.lease.guard_repo(name)
        win, info = await asyncio.to_thread(self.synthesize, cand)
        self.lane().repo_dirs[cand.repo] = str(self.monos[cand.repo])
        old = self.store.window(cand.id) if (self.store.dir(cand.id) / "window.json").exists() else None
        if old and old.get("milestones") and old.get("a") == win.a:
            rec = old  # validated before: only the stitched post-processing runs again
        else:
            rec = await validate_window(win, self.store, self.images, self.sandboxes, reruns=3, min_milestones=2, mutants=True, import_tolerant=False,
                                        leak_prefix=f"{WORLD_PREFIX}leak-", log=self.log)
        rec[NAME] = {**(rec.get(NAME) or {}), **info, "sources": [s["id"] for s in cand.sources]}
        mono = self.monos[cand.repo]
        base = self.store.dir(cand.id) / "base"
        if rec.get("oracle_files"):
            rec[NAME]["crlf_prose_dropped"] = drop_crlf_prose(rec, base)
        side = info["unit_side"]
        ms = rec.get("milestones") or []
        if ms and rec.get("stage") in ("leak_ok", "f2p_valid", "dependent", "mutants_valid") and not rec[NAME].get("split"):
            host_only = {**rec, "milestones": [m for m in ms if m.get("unit") is not None]}
            if not all(m.get("probe") for m in host_only["milestones"]):
                rec["prober"] = await probe_dependencies(host_only, RepoGit(mono), self.sandboxes(rec["world"]["image"]))
            rec[NAME]["split"] = await self.split_observations(rec, mono)
            ms = rec["milestones"]
            if len(ms) >= 2:
                rec["stage"], rec["reason"] = "f2p_valid", ""
                await asyncio.to_thread(self.store.save, mined_of(rec, base))
                if rec["dependency_edges"] == 0:
                    rec["reason"] = "independent: no milestone needs an earlier one (long, not long-horizon)"
                else:
                    rec["stage"] = "dependent"
                    host_rec = {**rec, "milestones": [m for m in ms if m.get("unit") is not None]}
                    mut = await revert_mutants(host_rec, RepoGit(mono), base, self.sandboxes(rec["world"]["image"]))
                    rec["mutants"] = mut["mutants"]
                    rec["mutants"] += await self.consumer_mutants(rec)
                    counts: dict[str, int] = {}
                    for x in rec["mutants"]:
                        counts[x["status"]] = counts.get(x["status"], 0) + 1
                    rec["mutant_counts"] = counts
                    if counts.get("valid", 0) >= 1:
                        rec["stage"] = "mutants_valid"
                    else:
                        rec["reason"] = f"mutants: {counts}"
        ms_by = {x["id"]: x for x in ms}
        cross_edges = [{"consumer_milestone": m["id"], "host_milestones": [r for r in m.get("requires") or [] if side.get(ms_by[r].get("unit") or "") == "host"]}
                       for m in ms if (m.get("unit") is None or side.get(m["unit"]) == "consumer")
                       and any(side.get(u) == "host" for u in m.get("requires_units") or [])]
        rec[NAME]["cross_edges"] = cross_edges
        rec[NAME]["consumer_tests_owned_by_host"] = sorted(t for m in ms if m.get("unit") is None or side.get(m["unit"]) == "host"
                                                           for t in m["tests"] if t.startswith("consumer/"))
        if rec.get("stage") in ("f2p_valid", "dependent", "mutants_valid"):
            t0 = time.monotonic()
            rec[NAME]["host_required"] = await self.host_required(rec, mono)
            rec[NAME]["host_required"]["seconds"] = round(time.monotonic() - t0, 1)
        hr = (rec[NAME].get("host_required") or {}).get("required") or []
        rec[NAME]["cross"] = bool(cross_edges or hr)
        self.store.save_window(rec)
        stage = rec.get("stage")
        self.funnel.note(NAME, cand.id, "S3_world", stage != "windows", "" if stage != "windows" else rec.get("reason", ""))
        if stage != "windows":
            ok4 = stage in ("f2p_valid", "dependent", "mutants_valid")
            self.funnel.note(NAME, cand.id, "S4_oracle", ok4, "" if ok4 else rec.get("reason", ""), milestones=len(ms))
            if ok4 and rec.get("stage") == "f2p_valid":  # mined, but no executed edge: S6 drops it here (no gate run needed)
                self.funnel.note(NAME, cand.id, "S6", False, rec.get("reason", ""), executed_edges=0)
            if ok4:
                self.funnel.note(NAME, cand.id, "S6_cross", rec[NAME]["cross"], "" if rec[NAME]["cross"] else
                                 "no consumer test requires the host change", cross_edges=len(cross_edges), host_required=len(hr))
        return rec


def pair_id(consumer: str, host: str) -> str:
    return hashlib.sha256(f"{consumer}|{host}".encode()).hexdigest()[:8]


def consumer_milestone_tests(rec: Mapping[str, Any]) -> dict[str, list[str]]:
    """Milestone id -> its consumer-side tests (what a stitched task checks in the consumer)."""
    return {m["id"]: [t for t in m["tests"] if t.startswith("consumer/")] for m in rec.get("milestones") or []}
