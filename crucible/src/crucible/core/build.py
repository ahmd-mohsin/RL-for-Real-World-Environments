"""Which code is running (R-11): the version alone says ``0.1.0`` across hundreds of commits.

``build_info()`` identifies the running Crucible by what it is, not by what it is called:
- ``source_sha256`` hashes every file of the installed ``crucible`` package (no git needed: wheels, copies, containers);
- ``git_commit`` is trusted only from a source that is provably Crucible's (PORT with notice of NeMo-Gym
  ``nemo_gym/orchestration/jobs.py:installed_gym_commit``, Apache-2.0): the commit pip recorded for a git install
  (PEP 610 ``direct_url.json``), else the HEAD of a checkout that tracks this package's ``__init__.py``. A virtualenv
  that merely sits inside some other repository reports ``None``, never that repository's commit;
- ``git_dirty`` covers the package directory only, so an edit elsewhere in a monorepo does not mark the code modified;
- ``lock_sha256`` hashes the ``uv.lock`` beside the package's own ``pyproject.toml`` (a source checkout).
Computed once per process.
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from crucible import __version__
from crucible.core.schema.base import CrucibleModel

_PACKAGE_DIR = Path(__file__).resolve().parents[1]  # .../crucible (the package)
_GIT_TIMEOUT_S = 5


class BuildInfo(CrucibleModel):
    crucible_version: str
    source_sha256: str
    git_commit: str | None = None  # None: not a git install and not a checkout that tracks this package
    git_dirty: bool | None = None  # uncommitted changes inside the package directory; None when git_commit is unknown
    lock_sha256: str | None = None  # uv.lock of a source checkout


@functools.cache
def build_info() -> BuildInfo:
    commit = _installed_commit()
    dirty = None
    if commit is None:
        commit, dirty = _checkout_state(_PACKAGE_DIR)
    return BuildInfo(crucible_version=__version__, source_sha256=source_digest(_PACKAGE_DIR), git_commit=commit,
                     git_dirty=dirty, lock_sha256=_lock_digest(_PACKAGE_DIR))


def source_digest(package_dir: Path) -> str:
    """SHA-256 over the sorted ``relpath\\0sha256(file)`` lines of every file in ``package_dir`` (bytecode caches skipped)."""
    outer = hashlib.sha256()
    files = sorted(p for p in package_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for p in files:
        outer.update(f"{p.relative_to(package_dir).as_posix()}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
    return outer.hexdigest()


def _installed_commit(dist: str = "crucible") -> str | None:
    """The commit PEP 610 recorded for a ``pip install git+...`` of ``dist``, else None."""
    try:
        raw = distribution(dist).read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    commit = ((json.loads(raw).get("vcs_info") or {}).get("commit_id")) if raw else None
    return str(commit) if commit else None


def _checkout_state(package_dir: Path) -> tuple[str | None, bool | None]:
    """(HEAD, uncommitted changes under ``package_dir``) of the checkout that tracks ``package_dir/__init__.py``."""
    tracked = _git(package_dir, "ls-files", "--error-unmatch", "__init__.py")
    head = _git(package_dir, "rev-parse", "HEAD") if tracked is not None else None
    if head is None:
        return None, None
    status = _git(package_dir, "status", "--porcelain", "--untracked-files=no", "--", ".")
    return head, (bool(status) if status is not None else None)


def _lock_digest(package_dir: Path) -> str | None:
    root = package_dir.parents[1]  # src/crucible -> the project
    pyproject, lock = root / "pyproject.toml", root / "uv.lock"
    if not (pyproject.is_file() and lock.is_file() and 'name = "crucible"' in pyproject.read_text()):
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def _git(cwd: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=_GIT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None
