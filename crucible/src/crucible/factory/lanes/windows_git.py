"""The window lane, git half (study 08 §0, §5; 09): release windows and commit runs from a local clone, trees built from
A plus any subset of the window's units, and the owner of every test.

**Unit.** A window is the first-parent path from A to B. Each step on it is one *unit*: a non-merge commit, or a merged
pull request taken whole (its diff against the first parent). Units on the first-parent path compose exactly (unit k's
diff applies to the tree after unit k-1), so the oracle, the dependency prober and the mutants never fight the merge
structure. A unit *qualifies* when it changes both source ``.py`` files and test ``.py`` files.

**Windows.** Release windows run between consecutive version tags (sorted by creator date) when A is on B's
first-parent path; commit runs are consecutive stretches of qualifying units on the default branch, kept only where they
do not overlap a release window. Both keep 5-25 qualifying units by default.

**Trees.** ``RepoGit.build(base, units, dest)`` checks out ``base`` into ``dest`` through a private index file (the
clone's object store is shared, never its work tree or index, so builds run in parallel) and applies each unit's
agent-writable diff with ``git apply --3way``. A unit that does not apply is reported, not forced.

**Owners.** The owner of a test is the last unit in the window that changed the test function's AST (decorators and
parametrize cases included). A test whose function no unit changed fails at A only through its module (a new import, a
new fixture) and guards existing behaviour: it is *collateral*, a regression check, never a milestone.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import os
import re
import shutil
import subprocess
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.lanes.executable import NOT_SOURCE, is_protected, is_test_path, prepare_tree

VERSION_TAG = re.compile(r"^(?:[A-Za-z][\w.]*?[-_])?v?(\d+(?:\.\d+){1,3})$")  # 1.2, v1.2.3, click-8.1.0; no pre-releases


def _git(repo: Path, *args: str, check: bool = True, env: Mapping[str, str] | None = None, input: bytes | None = None,
         text: bool = True) -> Any:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text and input is None, check=False,
                       env={**os.environ, **env} if env else None, input=input)
    if check and r.returncode != 0:
        err = r.stderr if isinstance(r.stderr, str) else r.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"git {' '.join(args[:4])}: {err.strip()[:300]}")
    return r


def is_source(path: str) -> bool:
    return path.endswith(".py") and not is_test_path(path) and not NOT_SOURCE.search(path) and not is_protected(path)


def is_test_file(path: str) -> bool:
    return path.endswith(".py") and is_test_path(path) and not path.endswith("conftest.py")


def writable(path: str) -> bool:
    """What the agent may change (and what a unit's oracle diff carries): every path that is not protected."""
    return not is_protected(path)


@dataclass
class Unit:
    sha: str
    parent: str  # the first parent
    merge: bool
    date: str
    subject: str
    files: list[str]  # every path the unit changes
    body: str = ""

    @property
    def src_files(self) -> list[str]:
        return [p for p in self.files if is_source(p)]

    @property
    def test_files(self) -> list[str]:
        return [p for p in self.files if is_test_file(p)]

    @property
    def writable_files(self) -> list[str]:
        return [p for p in self.files if writable(p)]

    @property
    def qualifies(self) -> bool:
        return bool(self.src_files) and bool(self.test_files)


@dataclass
class Window:
    id: str
    repo: str
    repo_dir: str
    kind: str  # release | run
    a: str
    b: str
    a_ref: str
    b_ref: str
    a_date: str
    b_date: str
    b_version: str | None
    units: list[Unit] = field(default_factory=list)

    @property
    def qualifying(self) -> list[Unit]:
        return [u for u in self.units if u.qualifies]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Window:
        units = [Unit(**u) for u in d.get("units") or []]
        return cls(**{**{k: d[k] for k in cls.__dataclass_fields__ if k in d and k != "units"}, "units": units})


def window_id(repo: str, a: str, b: str) -> str:
    return f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', repo)}-w{a[:7]}-{b[:7]}"


def first_parent_units(repo_dir: Path, a: str, b: str) -> list[Unit]:
    """The units from A (exclusive) to B (inclusive), oldest first; [] when A is not on B's first-parent path."""
    log = _git(Path(repo_dir), "log", "--first-parent", "--diff-merges=first-parent", "--reverse", "--numstat",
               "--format=@@%H%x09%P%x09%aI%x09%s", f"{a}..{b}").stdout
    units: list[Unit] = []
    for line in log.splitlines():
        if line.startswith("@@"):
            sha, parents, date, subject = (line[2:].split("\t", 3) + ["", "", ""])[:4]
            ps = parents.split()
            units.append(Unit(sha=sha, parent=ps[0] if ps else "", merge=len(ps) > 1, date=date, subject=subject, files=[]))
        elif line.strip() and units:
            parts = line.split("\t")
            if len(parts) == 3:
                path = parts[2]
                if " => " in path:  # a rename: numstat prints "dir/{old => new}.py"
                    m = re.match(r"(.*)\{(.*) => (.*)\}(.*)", path)
                    path = (m.group(1) + m.group(3) + m.group(4)).replace("//", "/") if m else path.split(" => ")[-1]
                units[-1].files.append(path)
    if not units or units[0].parent != _git(Path(repo_dir), "rev-parse", f"{a}^{{commit}}").stdout.strip():
        return []
    return units


def version_tags(repo_dir: Path) -> list[tuple[str, str, str, str]]:
    """``(tag, commit, date, version)`` for release tags, oldest first by creator date."""
    out = _git(Path(repo_dir), "for-each-ref", "--sort=creatordate", "--format=%(refname:short)%09%(objectname)%09%(*objectname)%09%(creatordate:iso-strict)",
               "refs/tags").stdout
    tags = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        tag, obj, peeled, date = parts
        m = VERSION_TAG.match(tag)
        if m:
            tags.append((tag, peeled or obj, date, m.group(1)))
    return tags


def _bodies(repo_dir: Path, units: Sequence[Unit]) -> None:
    for u in units:
        if not u.body:
            u.body = _git(Path(repo_dir), "log", "-1", "--format=%b", u.sha).stdout.strip()[:4000]


def release_windows(repo_dir: Path, repo: str, *, since: str = "2019-01-01", min_units: int = 5, max_units: int = 25) -> list[Window]:
    tags = version_tags(repo_dir)
    out = []
    for (ta, ca, da, _va), (tb, cb, db, vb) in itertools.pairwise(tags):
        if db[:10] < since or ca == cb:
            continue
        units = first_parent_units(repo_dir, ca, cb)
        n = sum(1 for u in units if u.qualifies)
        if units and min_units <= n <= max_units:
            out.append(Window(window_id(repo, ca, cb), repo, str(repo_dir), "release", ca, cb, ta, tb, da, db, vb, units))
    return out


def run_windows(repo_dir: Path, repo: str, *, size: int = 12, since: str = "2019-01-01", head: str = "HEAD",
                taken: Iterable[str] = ()) -> list[Window]:
    """Commit runs: consecutive stretches of ``size`` qualifying units on the default branch (no unit in ``taken``)."""
    root = _git(Path(repo_dir), "rev-list", "--max-parents=0", head).stdout.split()[-1]
    units = first_parent_units(repo_dir, root, head)
    taken = set(taken)
    q = [i for i, u in enumerate(units) if u.qualifies and u.date[:10] >= since]
    out = []
    for s in range(0, len(q) - size + 1, size):
        lo, hi = q[s], q[s + size - 1]
        span = units[lo: hi + 1]
        if any(u.sha in taken for u in span):
            continue
        a, b = span[0].parent, span[-1].sha
        out.append(Window(window_id(repo, a, b), repo, str(repo_dir), "run", a, b, a[:10], b[:10], units[lo - 1].date if lo else "", span[-1].date, None,
                          [Unit(**asdict(u)) for u in span]))
    return out


def mine_windows(repo_dir: Path, repo: str | None = None, *, cap: int = 8, since: str = "2019-01-01", min_units: int = 5, max_units: int = 25,
                 run_size: int = 12) -> list[Window]:
    """Release windows newest first, then commit runs that overlap none of them, at most ``cap`` in all."""
    repo_dir = Path(repo_dir)
    name = repo or repo_dir.name
    rel = sorted(release_windows(repo_dir, name, since=since, min_units=min_units, max_units=max_units), key=lambda w: w.b_date, reverse=True)
    taken = {u.sha for w in rel for u in w.units}
    runs = sorted(run_windows(repo_dir, name, size=max(min_units, min(run_size, max_units)), since=since, taken=taken), key=lambda w: w.b_date, reverse=True)
    out = (rel + runs)[:cap]
    for w in out:
        _bodies(repo_dir, w.units)
    return out


# ---- trees -------------------------------------------------------------------------------------------------------------
class RepoGit:
    """Trees from one clone: ``base`` plus any ordered subset of units, built through a private index."""

    def __init__(self, repo_dir: Path):
        self.dir = Path(repo_dir)
        self.gitdir = _git(self.dir, "rev-parse", "--absolute-git-dir").stdout.strip()
        self._patches: dict[tuple[str, bool], bytes] = {}

    def patch(self, u: Unit, *, source_only: bool = False) -> bytes:
        """The unit's agent-writable diff (protected paths and tests excluded), full index lines for ``--3way``;
        ``source_only``: its source ``.py`` files only (the prober and the mutants: a changelog line every unit appends
        to must not read as a dependency)."""
        key = (u.sha, source_only)
        if key not in self._patches:
            files = u.src_files if source_only else u.writable_files
            self._patches[key] = _git(self.dir, "diff", "--full-index", "--binary", u.parent, u.sha, "--", *files, text=False).stdout if files else b""
        return self._patches[key]

    def _env(self, dest: Path) -> dict[str, str]:
        return {"GIT_DIR": self.gitdir, "GIT_WORK_TREE": str(dest), "GIT_INDEX_FILE": str(dest) + ".idx"}

    def build(self, base: str, units: Sequence[Unit], dest: Path, *, reverse: Sequence[Unit] = (), source_only: bool = False) -> tuple[bool, str | None]:
        """``dest`` = ``base`` with ``units`` applied in order, then ``reverse`` units reverted. Returns (ok, the sha
        that did not apply)."""
        dest = Path(dest)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        env = self._env(dest)
        try:
            _git(dest, "read-tree", base, env=env)
            _git(dest, "checkout-index", "-a", "-f", "-u", env=env)
            for u, rev in [(u, False) for u in units] + [(u, True) for u in reverse]:
                p = self.patch(u, source_only=source_only)
                if not p.strip():
                    continue
                args = ["apply", "--3way", "--index", "--whitespace=nowarn", *(["-R"] if rev else []), "-"]
                if _git(dest, *args, check=False, env=env, input=p).returncode != 0:
                    return False, u.sha
            prepare_tree(dest)
            return True, None
        finally:
            Path(str(dest) + ".idx").unlink(missing_ok=True)

    def show(self, rev: str, path: str, *, follow: int = 3) -> str | None:
        """The text of ``path`` at ``rev``; a symlink is followed inside the tree (its target's text)."""
        r = subprocess.run(["git", "-C", str(self.dir), "show", f"{rev}:{path}"], capture_output=True, check=False)
        if r.returncode != 0:
            return None
        if follow:
            mode = _git(self.dir, "ls-tree", rev, "--", path, check=False).stdout.split(" ", 1)[0]
            if mode == "120000":
                target = os.path.normpath(os.path.join(os.path.dirname(path), r.stdout.decode("utf-8", "replace").strip()))
                return None if target.startswith("..") else self.show(rev, target, follow=follow - 1)
        try:
            return r.stdout.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def changed(self, a: str, b: str) -> list[tuple[str, str]]:
        """``(status, path)`` for every path that differs between two revisions (renames as delete + add)."""
        out = _git(self.dir, "diff", "--name-status", "--no-renames", a, b).stdout
        return [(ln.split("\t", 1)[0][:1], ln.split("\t", 1)[1]) for ln in out.splitlines() if "\t" in ln]


# ---- owners of tests ---------------------------------------------------------------------------------------------------
def _defs(text: str | None) -> dict[str, str]:
    """``Class::name`` / ``name`` -> the AST dump of every test function (positions ignored)."""
    if not text:
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    out: dict[str, str] = {}

    def walk(nodes: Iterable[ast.AST], prefix: str) -> None:
        for n in nodes:
            if isinstance(n, ast.ClassDef):
                walk(n.body, f"{prefix}{n.name}::")
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[prefix + n.name] = ast.dump(n, include_attributes=False)

    walk(tree.body, "")
    return out


def key_of_test(test_id: str) -> tuple[str, str]:
    """``file::Class::name[param]`` -> (file, ``Class::name``)."""
    parts = test_id.split("::")
    return parts[0], "::".join(re.sub(r"\[.*$", "", p) for p in parts[1:])


def owners_of_tests(repo: RepoGit, units: Sequence[Unit], test_ids: Iterable[str]) -> dict[str, str | None]:
    """The last unit that changed each test function (None: collateral)."""
    cache: dict[tuple[str, str], dict[str, str]] = {}

    def defs(rev: str, path: str) -> dict[str, str]:
        if (rev, path) not in cache:
            cache[(rev, path)] = _defs(repo.show(rev, path))
        return cache[(rev, path)]

    out: dict[str, str | None] = {}
    for t in test_ids:
        path, qual = key_of_test(t)
        owner = None
        for u in units:
            if path in u.files and defs(u.parent, path).get(qual) != defs(u.sha, path).get(qual):
                owner = u.sha
        out[t] = owner
    return out


def names_of_tests(test_ids: Iterable[str]) -> set[str]:
    return {key_of_test(t)[1].rsplit("::", 1)[-1] for t in test_ids}


def packaging_hash(tree: Path, files: Sequence[str]) -> str:
    """What a dependency install depends on: the packaging files with their version strings blanked."""
    h = hashlib.sha256()
    for rel in sorted(files):
        f = Path(tree) / rel
        if f.is_file():
            text = re.sub(r"""(?im)^(\s*_?_?version_?_?\s*[:=]\s*)["'][^"']*["']""", r'\1"X"', f.read_text(errors="replace"))
            h.update(rel.encode() + b"\0" + text.encode() + b"\0")
    return h.hexdigest()
