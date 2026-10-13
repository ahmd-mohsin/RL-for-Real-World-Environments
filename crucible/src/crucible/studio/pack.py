"""``crucible pack new`` and ``crucible pack check``.

**new.** Copies ``packs/_template`` with ``__PACK_NAME__`` / ``__PACK_PKG__`` substituted in file contents and names
(OpenEnv ``openenv init``). The writer is NeMo-Gym's conflict-safe scaffold (``environment/scaffold.py:909-950``):
- it plans every file first;
- an identical existing file is a no-op;
- any differing file aborts the whole write before anything is written;
- symlinks are refused, and new files are opened with ``"x"``.
A name already taken by another pack (``packs/`` or an installed pack) is refused. The template's version is recorded in
``pack.yaml`` for a later 3-way upgrade (karotte).

**check.** Levels in the OpenEnv ``validate --level`` shape; each is cumulative and every check fails closed (M14 audit):
- **static:** the manifest parses and its contract holds (name, version, threshold inside the range); no placeholder is
  left anywhere in the pack's text or ``src/`` (``__PACK_*__``, TODO, FIXME, TBD, XXX); the licence is an SPDX expression
  and agrees with ``pyproject.toml`` (name, version, licence, the crucible requirement); ``src/`` holds only the package;
  the plugin entry point in ``pyproject.toml`` equals ``entry_point`` and ``register()`` registers this pack at this
  directory; every hook lives in the pack's package; every ingest path resolves inside an allowed root and every declared
  ``redaction_env`` exists. Template text left over, no authors, a wheel that would not carry ``pack.yaml`` and a missing
  README are warnings (failures with ``--publish``).
- **fixture** (NeMo-Gym ``verifier_fixture.py``): the ``fixtures`` hook is required and returns cases; every reward is a
  finite number inside ``reward.range`` and equals the case's expected reward; a ``fails_gate`` names a gate the grader
  reports as ``fail``; at least one case sits at the best end of the range and one at the worst; a ``seeded`` pack's
  cases are re-graded on fresh Worlds (``runs`` times) with identical reward and gates. A self-graded case (``grade``
  callable) is only for packs without live tasks and must be the pack's own code.
- **semantic** (model-free; verifiers ``vf-validate``, OpenEnv RFC 008): the ``tasks`` hook is required and non-empty;
  gold reaches the best end of the range (re-run on fresh Worlds when seeded), the untouched task sits at the worst end;
  ``tasks()`` and ``generator(seed)`` rebuild the same tasks; generated items carry an oracle that passes the same gold
  and no-op bar (what ``bench run`` demands); every adversary (``{"policy", "expect_max"}`` from ``fixtures`` or
  ``policies``) stays within its bound on every task and generated item.
- **accept**: the M9 Acceptance Contract (``crucible.accept.run``) on the fixture tasks, the anchors and a generated item
  through ``crucible.studio.accept_adapter`` (A5 leak, A3 negatives and mutation, ...).
A pack's ``requires`` (docker, gpu, network, paid) describes its live tasks: docker and gpu are detected, network and paid
are vouched for with ``--allow``. When this machine cannot satisfy it the live levels are ``skip`` rows, the report is
incomplete, and the exit code is 2 unless ``--allow-skip``.

Packs are imported in isolation: a pack's ``src/`` goes first on ``sys.path`` for the check only, another copy of the same
package already in ``sys.modules`` is set aside (and restored afterwards), and every resolved module must come from the
pack's own ``src/``. ``--all`` checks each pack in its own process. A hook that raises is a ``<level>.error`` failure, not
a traceback. Exit codes: 0 pass or warn, 1 a check failed, 2 a requested level was skipped (without ``--allow-skip``),
3 an internal error.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from crucible.core.schema.pack import NAME, PackManifest

PACKS_ROOT = Path(__file__).resolve().parents[3] / "packs"
REPO_ROOT = PACKS_ROOT.parent.parent  # the checkout that holds crucible/ (the coding pack reads ../migration_tasks)
TEMPLATE = PACKS_ROOT / "_template"
TEMPLATE_VERSION = 2
POLICY = "severity-v2"
SEVERITY = {  # check id -> "fail" | "warn"
    "static.manifest": "fail", "static.placeholders": "fail", "static.template_text": "warn", "static.license": "fail",
    "static.pyproject": "fail", "static.distribution": "warn", "static.layout": "fail", "static.readme": "warn", "static.authors": "warn",
    "static.entry_point": "fail", "static.hooks": "fail", "static.ingest": "fail",
    "fixture.hook": "fail", "fixture.cases": "fail", "fixture.kinds": "fail", "fixture.determinism": "fail",
    "semantic.tasks": "fail", "semantic.gold": "fail", "semantic.noop": "fail", "semantic.determinism": "fail",
    "semantic.generator": "fail", "semantic.adversaries": "fail",
    "accept.contract": "fail", "accept.leak": "fail",
}
PUBLISH = {"static.template_text", "static.readme", "static.authors", "static.distribution"}  # warn in dev, fail with --publish
LEVELS = ("static", "fixture", "semantic", "accept")
LIVE = ("semantic", "accept")  # levels that run the pack's live tasks (``requires`` applies)
RUNS = 3  # grades per case / gold runs on fresh Worlds for a seeded pack (NeMo-Gym's determinism case; G2 pick 1)
GEN_SEEDS = (101, 202)  # generator seeds every check draws (fixed: a check is reproducible)
EPS = 1e-6
EXIT_OK, EXIT_FAIL, EXIT_INCOMPLETE, EXIT_INTERNAL = 0, 1, 2, 3
PLACEHOLDER = re.compile(r"__PACK_[A-Z_]+__|__TEMPLATE_VERSION__|\b(?:TODO|FIXME|TBD|XXX)\b|<(?:describe|your|insert|fill in)\b[^<>\n]{0,60}>", re.IGNORECASE)
MARKERS = ("TODO", "FIXME", "TBD", "XXX")  # upper case only: "todo" in prose is not a marker
TEMPLATE_TEXT = ("(edit this description)", "Replace the task in", 'description = "A Crucible domain pack"')
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".ini", ".json"}


class ScaffoldConflict(RuntimeError):
    pass


@dataclass
class CheckResult:
    id: str
    status: str  # pass | warn | fail | skip
    detail: str = ""


@dataclass
class PackReport:
    pack: str
    path: str
    level: str
    results: list[CheckResult] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (level, reason): a level that was asked for and did not run
    allow_skip: bool = False
    publish: bool = False

    @property
    def complete(self) -> bool:
        return not self.skipped

    @property
    def ok(self) -> bool:
        """No failure, and every requested level ran (a skipped level is not a pass unless ``allow_skip``: M14 audit V6)."""
        return not any(r.status == "fail" for r in self.results) and (self.complete or self.allow_skip)

    @property
    def exit_code(self) -> int:
        if any(r.status == "fail" for r in self.results):
            return EXIT_INTERNAL if any(r.id == "internal.error" for r in self.results) else EXIT_FAIL
        return EXIT_OK if self.complete or self.allow_skip else EXIT_INCOMPLETE

    def to_json(self) -> dict[str, Any]:
        return {"pack": self.pack, "path": self.path, "level": self.level, "ok": self.ok, "complete": self.complete, "exit": self.exit_code,
                "policy": POLICY, "publish": self.publish, "skipped": [list(s) for s in self.skipped],
                "results": [asdict(r) for r in self.results], "problems": [f"{r.id}: {r.detail}" for r in self.results if r.status == "fail"]}


# ---- locate ------------------------------------------------------------------------------------------------------------
def _yaml_name(d: Path) -> str | None:
    try:
        return (yaml.safe_load((d / "pack.yaml").read_text()) or {}).get("name")
    except (yaml.YAMLError, OSError, AttributeError):
        return None


def _installed_packs() -> dict[str, Path]:
    """Installed packs by name: the registry's ``pack`` kind, whose value is the pack root (G2 pick 6)."""
    from crucible.core.registry import get_registry

    out = {}
    try:
        reg = get_registry()
        names = reg.names("pack")
    except Exception:  # noqa: BLE001 - a broken plugin is the registry's failure list, not a lookup crash
        return {}
    for n in names:
        try:
            root = Path(str(reg.get("pack", n)))
        except Exception:  # noqa: BLE001, S112 - a pack whose root does not resolve is not an installed pack
            continue
        if (root / "pack.yaml").exists():
            out[n] = root.resolve()
    return out


def pack_index(*, installed: bool = True) -> dict[str, list[Path]]:
    """Every known pack by name: the directories under ``packs/`` (``_template`` excluded), then installed packs. A name
    with two different roots is ambiguous (V19)."""
    idx: dict[str, list[Path]] = {}
    for d in sorted(PACKS_ROOT.iterdir()) if PACKS_ROOT.exists() else []:
        if d.name != "_template" and (d / "pack.yaml").exists() and (n := _yaml_name(d)):
            idx.setdefault(str(n), []).append(d.resolve())
    if installed:
        for n, root in _installed_packs().items():
            if root not in idx.get(n, []) and not idx.get(n):
                idx.setdefault(n, []).append(root)
    return idx


def find_pack(ref: str | Path) -> Path:
    """A pack directory from a path, or from a name (``toy`` -> ``packs/_toy``; else an installed pack via the registry).

    A name is looked up in the pack index first, so a directory named ``toy`` in the working directory never replaces the
    toy pack (V19); a reference that is a path (a ``Path``, or a string with a separator) is only ever a path. Two packs
    with one name are an error naming both."""
    s = str(ref)
    if isinstance(ref, Path) or "/" in s or os.sep in s or s in (".", ".."):
        p = Path(ref)
        if (p / "pack.yaml").exists():
            return p.resolve()
        raise ValueError(f"no pack at {p}: the directory has no pack.yaml")
    hits = pack_index(installed=False).get(s) or ([r] if (r := _installed_packs().get(s)) else [])
    if len(hits) > 1:
        raise ValueError(f"pack name {s!r} is ambiguous: " + ", ".join(str(h) for h in hits) + " (give a path)")
    if hits:
        return hits[0]
    p = Path(ref)
    if (p / "pack.yaml").exists():  # a bare directory name that names no known pack
        return p.resolve()
    raise ValueError(f"no pack {ref!r}: give a directory with pack.yaml, or the name of a pack under {PACKS_ROOT} or of an installed pack")


def load_manifest(pack_dir: Path) -> PackManifest:
    return PackManifest.model_validate(yaml.safe_load((Path(pack_dir) / "pack.yaml").read_text()))


# ---- import in isolation ------------------------------------------------------------------------------------------------
def _in_pkg(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def _module_path(mod: Any) -> Path | None:
    f = getattr(mod, "__file__", None)
    if f:
        return Path(f).resolve()
    paths = list(getattr(mod, "__path__", None) or [])
    return Path(paths[0]).resolve() if paths else None


def _under(p: Path | None, root: Path) -> bool:
    if p is None:
        return False
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


@contextlib.contextmanager
def isolated(pack_dir: Path, package: str) -> Iterator[None]:
    """Import ``package`` from this pack's ``src/`` for the duration (V7, V15, G2 D5): ``src/`` goes first on ``sys.path``,
    any module of the same package loaded from elsewhere is set aside, and on exit ``sys.path`` is restored, the modules
    this check loaded are dropped and the set-aside ones come back."""
    src = (Path(pack_dir) / "src").resolve()
    if not package or not (src / package).is_dir():  # an installed pack imports as it is
        yield
        return
    import importlib.util

    saved_path = list(sys.path)
    # the pack's src/ is where this process imports the package from anyway (the pack is on sys.path / installed in
    # place): its modules stay loaded, so objects other code already holds keep one identity. Otherwise the modules
    # this check loads are dropped afterwards and the ones it set aside come back.
    if package in sys.modules:
        natural = _under(_module_path(sys.modules[package]), src)
    else:
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            spec = None
        natural = spec is not None and _under(Path(spec.origin).resolve() if spec.origin else _module_path(spec), src)
    before = {k for k in sys.modules if _in_pkg(k, package)}
    stash = {}
    for k in sorted(before):
        if not _under(_module_path(sys.modules[k]), src):
            stash[k] = sys.modules.pop(k)
    sys.path.insert(0, str(src))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        if not natural or stash:
            for k in [k for k in sys.modules if _in_pkg(k, package)]:
                if k not in before or k in stash:
                    sys.modules.pop(k, None)
            sys.modules.update(stash)
        sys.path[:] = saved_path
        importlib.invalidate_caches()


def resolve(pack_dir: Path, ref: str) -> Any:
    """``module:attr`` from the pack's ``src/`` (installed packs import as they are). The module must come from this pack:
    a cached copy of the same package from another directory is dropped first, so a fork or an edited copy is never run
    with the original's code (V7)."""
    mod, attr = ref.split(":", 1)
    src = (Path(pack_dir) / "src").resolve()
    top = mod.split(".")[0]
    own = (src / top).is_dir()
    if own:
        if sys.path[:1] != [str(src)]:
            if str(src) in sys.path:
                sys.path.remove(str(src))
            sys.path.insert(0, str(src))
        cached = sys.modules.get(top)
        if cached is not None and not _under(_module_path(cached), src):
            for k in [k for k in sys.modules if _in_pkg(k, top)]:
                sys.modules.pop(k, None)
            importlib.invalidate_caches()
    obj: Any = importlib.import_module(mod)
    if own and not _under(_module_path(obj), src):
        raise ImportError(f"{mod} resolves to {_module_path(obj)}, not this pack's {src}: check the pack in a fresh process")
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


async def call_hook(pack_dir: Path, ref: str, *args: Any, **kw: Any) -> Any:
    res = resolve(pack_dir, ref)(*args, **kw)
    return await res if inspect.isawaitable(res) else res


# ---- new ---------------------------------------------------------------------------------------------------------------
def plan_scaffold(name: str, dest: Path, *, template: Path = TEMPLATE) -> dict[Path, bytes]:
    if not NAME.match(name):
        raise ValueError(f"pack name {name!r}: lowercase letters, digits and _, starting with a letter")
    subs = {"__PACK_NAME__": name, "__PACK_PKG__": f"crucible_pack_{name}", "__TEMPLATE_VERSION__": str(TEMPLATE_VERSION)}

    def sub(s: str) -> str:
        for k in sorted(subs, key=len, reverse=True):  # longest token first (OpenEnv)
            s = s.replace(k, subs[k])
        return s

    out = {}
    for f in sorted(template.rglob("*")):
        if f.is_dir() or "__pycache__" in f.parts:
            continue
        rel = sub(f.relative_to(template).as_posix())
        out[Path(dest) / rel] = sub(f.read_text()).encode()
    return out


def write_scaffold(files: dict[Path, bytes], root: Path) -> dict[str, list[str]]:
    """Plan-then-write: identical files are kept, any conflict aborts before a byte is written. Nothing inside ``root``
    may be a symlink (a scaffold must not write outside the pack)."""
    created, existing, conflicts = [], [], []
    root = Path(root)
    for path, data in sorted(files.items()):
        inside = [path, *[q for q in path.parents if q == root or root in q.parents]]
        for parent in inside:
            if parent.is_symlink():
                raise ScaffoldConflict(f"refusing to write through a symlink: {parent}")
        if path.exists():
            (existing if path.read_bytes() == data else conflicts).append(str(path))
        else:
            created.append(str(path))
    if conflicts:
        raise ScaffoldConflict("these files exist with different content (nothing was written): " + ", ".join(conflicts))
    for path, data in sorted(files.items()):
        if str(path) in created:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "xb") as f:
                f.write(data)
    return {"created": created, "existing": existing}


def new(name: str, dest: Path) -> dict[str, Any]:
    taken = [p for p in pack_index().get(name, []) if p != Path(dest).resolve()]
    if taken:  # V19: a second pack with a taken name shadows or is shadowed by the first wherever a name is resolved
        raise ScaffoldConflict(f"a pack named {name!r} already exists at {', '.join(map(str, taken))}; choose another name")
    res = write_scaffold(plan_scaffold(name, dest), Path(dest))
    return {"pack": name, "path": str(Path(dest).resolve()), **res,
            "next": [f"crucible pack check {dest}", f"crucible pack check {dest} --level semantic", f"crucible bench run {dest}"]}


# ---- check: shared helpers ---------------------------------------------------------------------------------------------
def _r(cid: str, ok: bool, detail: str = "", *, skip: bool = False) -> CheckResult:
    if skip:
        return CheckResult(cid, "skip", detail)
    return CheckResult(cid, "pass" if ok else SEVERITY.get(cid, "fail"), detail)


def _num(x: Any) -> float | None:
    """A finite real reward, else None (NaN, inf, None, a bool or a string are never a reward: V3)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


class _Range:
    def __init__(self, m: PackManifest):
        self.lo, self.hi = m.reward.range
        self.hib = m.reward.higher_is_better
        self.best, self.worst = (self.hi, self.lo) if self.hib else (self.lo, self.hi)

    def inside(self, x: float) -> bool:
        return self.lo - EPS <= x <= self.hi + EPS

    def at(self, x: float | None, target: float) -> bool:
        return x is not None and abs(x - target) <= EPS

    def within_bound(self, x: float, bound: float) -> bool:
        """``x`` is no better than ``bound`` (an adversary's ``expect_max``)."""
        return x <= bound + EPS if self.hib else x >= bound - EPS

    def __str__(self) -> str:
        return f"[{self.lo:g}, {self.hi:g}]" + ("" if self.hib else " (lower is better)")


def allowed_roots(pack_dir: Path) -> list[Path]:
    """Where a pack's ingest paths may point (V17): the pack itself, the checkout that holds Crucible, and every directory in
    ``$CRUCIBLE_DATA_ROOTS`` (``os.pathsep``-separated)."""
    roots = [Path(pack_dir).resolve(), REPO_ROOT.resolve()]
    roots += [Path(x).expanduser().resolve() for x in os.environ.get("CRUCIBLE_DATA_ROOTS", "").split(os.pathsep) if x.strip()]
    return roots


def confined(path: Path, roots: list[Path]) -> bool:
    p = Path(path).resolve()  # follows symlinks: a link inside the pack to outside it is outside
    return any(_under(p, r) for r in roots)


PATH_KEYS = {"root", "dir", "path", "file"}
PATH_SUFFIXES = ("_dir", "_root", "_path", "_file")


def ingest_value(v: Any, pack_dir: Path, *, key: str | None = None, init: bool = False) -> Any:
    """One resolver for ``check_static`` and ``label`` (V20): ``./`` / ``../`` strings anywhere in ``init`` / ``scope`` and
    plain relative strings under path-typed ``init`` keys (``root``, ``*_dir``, ...) resolve against the pack root,
    recursively through lists and dicts. Anything else is passed through."""
    if isinstance(v, str):
        pathy = init and key is not None and (key in PATH_KEYS or key.endswith(PATH_SUFFIXES))
        if v.startswith(("./", "../")) or (pathy and v and not any(c in v for c in "*?[") and "://" not in v):
            return Path(pack_dir) / v
        return v
    if isinstance(v, dict):
        return {k: ingest_value(x, pack_dir, key=k, init=init) for k, x in v.items()}
    if isinstance(v, list):
        return [ingest_value(x, pack_dir, key=key, init=init) for x in v]
    return v


def _paths_in(v: Any) -> list[Path]:
    if isinstance(v, Path):
        return [v]
    if isinstance(v, dict):
        return [p for x in v.values() for p in _paths_in(x)]
    if isinstance(v, list):
        return [p for x in v for p in _paths_in(x)]
    return []


def ingest_problems(pack_dir: Path, spec: Any) -> tuple[list[str], list[Path]]:
    """What is wrong with one ingest source, and the files a file_drop source reads (each confined to the allowed roots)."""
    from crucible.core.registry import get_registry

    pack_dir = Path(pack_dir)
    roots = allowed_roots(pack_dir)
    bad: list[str] = []
    if spec.redaction_env:
        env = pack_dir / spec.redaction_env
        if not confined(env, roots):
            bad.append(f"{spec.family}: redaction_env {spec.redaction_env} resolves outside the allowed roots")
        elif not env.is_file() and not spec.redaction_env_optional:
            bad.append(f"{spec.family}: redaction_env {spec.redaction_env} does not exist (its secret values would be stored unredacted;"
                       " declare redaction_env_optional: true if the file is a developer convenience)")
    if spec.connector == "file_drop":
        base = pack_dir / (spec.root or ".")
        if not confined(base, roots):
            return [*bad, f"{spec.family}: root {spec.root} resolves outside the allowed roots"], []
        try:
            files = sorted(p for p in base.glob(spec.glob or "") if p.is_file())
        except (ValueError, NotImplementedError) as e:
            return [*bad, f"{spec.family}: glob {spec.glob!r}: {e}"], []
        outside = [str(p) for p in files if not confined(p, roots)]
        if outside:
            bad.append(f"{spec.family}: {len(outside)} matched file(s) resolve outside the allowed roots: {outside[:3]}")
        if not files:
            bad.append(f"{spec.family}: no file matches {spec.glob}")
        return bad, [p for p in files if confined(p, roots)]
    try:
        target = get_registry().get("connector", spec.connector)
    except Exception:  # noqa: BLE001 - reported
        return [*bad, f"{spec.family}: no connector named {spec.connector!r}"], []
    if isinstance(target, str):
        try:
            target = resolve(pack_dir, target)
        except Exception as e:  # noqa: BLE001
            return [*bad, f"{spec.family}: connector {spec.connector!r}: {type(e).__name__}: {e}"], []
    if (spec.discover != "discover" or spec.scope) and not callable(getattr(target, spec.discover, None)):
        bad.append(f"{spec.family}: connector {spec.connector!r} has no method {spec.discover!r}")
    for p in _paths_in(ingest_value(spec.init, pack_dir, init=True)) + _paths_in(ingest_value(spec.scope, pack_dir)):
        if not confined(p, roots):
            bad.append(f"{spec.family}: {p} resolves outside the allowed roots")
        elif not p.exists():
            bad.append(f"{spec.family}: {p} does not exist")
    return bad, []


# ---- static ------------------------------------------------------------------------------------------------------------
def _text_files(pack_dir: Path) -> list[Path]:
    out = [pack_dir / n for n in ("pack.yaml", "README.md", "pyproject.toml") if (pack_dir / n).is_file()]
    src = pack_dir / "src"
    if src.is_dir():
        out += sorted(p for p in src.rglob("*") if p.is_file() and p.suffix in TEXT_SUFFIXES and "__pycache__" not in p.parts)
    return out


def _placeholders(pack_dir: Path) -> list[str]:
    hits = []
    for p in _text_files(pack_dir):
        try:
            lines = p.read_text().splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(lines):
            for mt in PLACEHOLDER.finditer(line):
                tok = mt.group(0)
                if tok.upper() in MARKERS and tok not in MARKERS:
                    continue  # lower-case prose ("todo list")
                hits.append(f"{p.relative_to(pack_dir)}:{i + 1} {tok}")
    return hits


def _spdx(expr: str) -> str | None:
    """The canonical SPDX expression, or None. ``LicenseRef-*`` is valid SPDX (``check_static`` then wants a LICENSE file)."""
    e = (expr or "").strip()
    if not e:
        return None
    try:
        from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression
    except ImportError:  # an old packaging: a short allowlist
        known = {"apache-2.0": "Apache-2.0", "mit": "MIT", "bsd-3-clause": "BSD-3-Clause", "bsd-2-clause": "BSD-2-Clause", "mpl-2.0": "MPL-2.0",
                 "gpl-3.0-only": "GPL-3.0-only", "gpl-3.0-or-later": "GPL-3.0-or-later", "lgpl-3.0-only": "LGPL-3.0-only", "cc-by-4.0": "CC-BY-4.0",
                 "isc": "ISC", "unlicense": "Unlicense"}
        return known.get(e.lower()) or (e if re.fullmatch(r"LicenseRef-[A-Za-z0-9.\-]+", e) else None)
    try:
        return canonicalize_license_expression(e)
    except InvalidLicenseExpression:
        return None


def _norm_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def crucible_version() -> str:
    try:
        from importlib.metadata import version

        return version("crucible")
    except Exception:  # noqa: BLE001 - a source checkout without metadata: its own pyproject
        import tomllib

        return tomllib.loads((PACKS_ROOT.parent / "pyproject.toml").read_text())["project"]["version"]


def _scratch_registry() -> Any:
    from crucible.core.registry import Registry

    r = Registry()
    r._plugins_loaded = True  # a scratch registry: only what this pack's register() adds
    return r


def _pyproject_checks(pack_dir: Path, m: PackManifest, lic: str | None) -> tuple[list[CheckResult], list[str] | None]:
    """pyproject.toml agrees with pack.yaml (G2 pick 4) and the wheel is installable (G2 pick 6). Returns the rows and the
    declared ``crucible.plugins`` entry points (None without a pyproject)."""
    import tomllib

    f = pack_dir / "pyproject.toml"
    if not f.is_file():
        return [], None
    try:
        pp = tomllib.loads(f.read_text())
    except Exception as e:  # noqa: BLE001
        return [_r("static.pyproject", False, f"pyproject.toml: {type(e).__name__}: {e}"[:300])], []
    proj = pp.get("project") or {}
    disagree, dist = [], []
    if _norm_dist(str(proj.get("name", ""))) != _norm_dist(f"crucible-pack-{m.name}"):
        disagree.append(f"name {proj.get('name')!r} (expected crucible-pack-{m.name})")
    if str(proj.get("version", "")) != m.version:
        disagree.append(f"version {proj.get('version')!r} != pack.yaml {m.version!r}")
    pl = proj.get("license")
    pl = pl.get("text") if isinstance(pl, dict) else pl
    if pl is not None and (_spdx(str(pl)) or str(pl)) != (lic or m.license):
        disagree.append(f"license {pl!r} != pack.yaml {m.license!r}")
    core = None
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:  # pragma: no cover - packaging ships with pip
        Requirement = None  # type: ignore[assignment]
    for d in proj.get("dependencies") or []:
        if Requirement is None:
            break
        try:
            rq = Requirement(d)
        except InvalidRequirement:
            disagree.append(f"dependency {d!r} is not a requirement")
            continue
        if _norm_dist(rq.name) == "crucible":
            core = rq
    cv = crucible_version()
    if core is None:
        dist.append("no crucible requirement in dependencies (declare e.g. crucible>=0.1,<0.2)")
    elif not core.specifier.contains(cv, prereleases=True):
        disagree.append(f"requires {core} but this is crucible {cv}")
    wheel = (((pp.get("tool") or {}).get("hatch") or {}).get("build") or {}).get("targets", {}).get("wheel", {})
    if "pack.yaml" not in " ".join(str(k) for k in (wheel.get("force-include") or {})):
        dist.append("the wheel does not force-include pack.yaml (an installed pack cannot be found or checked)")
    ep = [str(v) for v in ((proj.get("entry-points") or {}).get("crucible.plugins") or {}).values()]
    return [_r("static.pyproject", not disagree, "; ".join(disagree) or "name, version, licence and crucible requirement agree"),
            _r("static.distribution", not dist, "; ".join(dist) or f"installable (crucible {core.specifier if core else '?'})")], ep


def check_static(pack_dir: Path) -> tuple[list[CheckResult], PackManifest | None]:
    out: list[CheckResult] = []
    pack_dir = Path(pack_dir)
    left = _placeholders(pack_dir)  # scanned whatever the manifest says (a raw template copy fails here too: V10)
    ph = _r("static.placeholders", not left, ("placeholders left: " + ", ".join(left[:10])) if left else "none")
    try:
        m = load_manifest(pack_dir)
    except Exception as e:  # noqa: BLE001 - reported as a check failure
        return [_r("static.manifest", False, f"{type(e).__name__}: {e}"[:500]), ph], None
    out += [_r("static.manifest", True, f"{m.name} {m.version}"), ph]
    tmpl = []
    for p in _text_files(pack_dir):
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        tmpl += [f"{p.relative_to(pack_dir)}: {t!r}" for t in TEMPLATE_TEXT if t in text]
    studio, pristine = pack_dir / "src" / m.package / "studio.py", TEMPLATE / "src" / "__PACK_PKG__" / "studio.py"
    if (m.kind == "domain" and studio.is_file() and pristine.is_file()
            and studio.read_text() == pristine.read_text().replace("__PACK_NAME__", m.name).replace("__PACK_PKG__", m.package)):
        tmpl.append(f"src/{m.package}/studio.py is the scaffold's greeting task, unchanged")
    out.append(_r("static.template_text", not tmpl, ("template text left: " + "; ".join(tmpl[:6])) if tmpl else "none"))
    # licence: an SPDX expression; LicenseRef-* needs the licence text (V18, G2 D10)
    lic = _spdx(m.license)
    has_text = any((pack_dir / n).exists() for n in ("LICENSE", "LICENSE.md", "LICENSE.txt"))
    lic_ok = lic is not None and (not lic.startswith("LicenseRef-") or has_text)
    out.append(_r("static.license", lic_ok, m.license if lic_ok else f"{m.license!r} is not an SPDX licence expression (e.g. Apache-2.0, MIT)"
                  + (": LicenseRef-* needs a LICENSE file" if lic and lic.startswith("LicenseRef-") else "")))
    out.append(_r("static.authors", bool(m.authors), ", ".join(m.authors) or "no authors"))
    # layout: src/ holds the package and nothing importable beside it (V15: a stray numpy.py shadowed numpy)
    src = pack_dir / "src"
    if not src.is_dir() and not (pack_dir / "pyproject.toml").exists():  # an installed pack (pack.yaml shipped in the wheel)
        import importlib.util

        spec = importlib.util.find_spec(m.package)
        out.append(_r("static.layout", spec is not None, f"installed package {m.package}" + ("" if spec else " is not importable")))
    else:
        stray = sorted(p.name for p in src.iterdir() if p.name not in (m.package, "__pycache__") and not p.name.startswith(".")
                       and not p.name.endswith((".egg-info", ".dist-info"))) if src.is_dir() else []
        lay = (src / m.package / "__init__.py").is_file() and (pack_dir / "pyproject.toml").exists() and not stray
        out.append(_r("static.layout", lay, f"src/{m.package}, pyproject.toml" + (f"; src/ also holds {stray} (it would shadow other modules)" if stray else "")))
    out.append(_r("static.readme", (pack_dir / "README.md").exists(), "README.md"))
    rows, ep_declared = _pyproject_checks(pack_dir, m, lic)
    out += rows
    # entry point: importable, equal to pyproject's crucible.plugins entry, and register() registers this pack here (V11)
    ep_bad = []
    if ep_declared is not None and m.entry_point not in ep_declared:
        ep_bad.append(f"pyproject crucible.plugins declares {ep_declared or 'nothing'}, not {m.entry_point}")
    try:
        reg = resolve(pack_dir, m.entry_point)
        if not callable(reg):
            ep_bad.append(f"{m.entry_point} is not callable")
        else:
            r = _scratch_registry()
            reg(r)
            packs = r._kinds.get("pack", {})
            if m.name not in packs:
                ep_bad.append(f"register() registers no pack {m.name!r} (it registers {sorted(packs) or 'none'})")
            elif Path(str(packs[m.name].resolve())).resolve() != pack_dir.resolve():
                ep_bad.append(f"register() points pack {m.name!r} at {packs[m.name].resolve()}, not {pack_dir.resolve()}")
    except Exception as e:  # noqa: BLE001
        ep_bad.append(f"{m.entry_point}: {type(e).__name__}: {e}"[:300])
    out.append(_r("static.entry_point", not ep_bad, "; ".join(ep_bad) or m.entry_point))
    # hooks: the pack's own code, callable (V11: hooks pointed at another pack or at builtins)
    bad = []
    for k, ref in m.hooks.model_dump().items():
        if ref is None:
            continue
        if not _in_pkg(ref.split(":", 1)[0], m.package):
            bad.append(f"{k} ({ref}) is outside the pack's package {m.package}")
            continue
        try:
            if not callable(resolve(pack_dir, ref)):
                bad.append(f"{k} ({ref}) is not callable")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{k} ({ref}): {type(e).__name__}: {e}"[:200])
    out.append(_r("static.hooks", not bad, "; ".join(bad) if bad else ", ".join(k for k, v in m.hooks.model_dump().items() if v) or "none declared"))
    missing = []
    for s in m.ingest:
        try:
            missing += ingest_problems(pack_dir, s)[0]
        except Exception as e:  # noqa: BLE001
            missing.append(f"{s.family}: {type(e).__name__}: {e}"[:200])
    out.append(_r("static.ingest", not missing, "; ".join(missing) if missing else f"{len(m.ingest)} source(s)"))
    return out, m


# ---- episodes ----------------------------------------------------------------------------------------------------------
def as_actions(raw: Any) -> list[Any]:
    """Actions from pack material: action objects pass through; ``{"kind": "tool_call" | "message", ...}`` dicts convert."""
    from crucible.core.schema.actions import Message, ToolCall

    out = []
    for i, a in enumerate(list(raw or [])):
        if not isinstance(a, dict):
            out.append(a)
        elif a.get("kind", "tool_call") == "tool_call":
            out.append(ToolCall(tool_call_id=a.get("tool_call_id") or f"a{i}", function_name=a["function_name"], arguments=a.get("arguments") or {}))
        else:
            out.append(Message(to=a["to"], text=a["text"]))
    return out


def as_dicts(acts: Any) -> list[dict]:
    return [a.model_dump(mode="json", exclude_none=True) if hasattr(a, "model_dump") else dict(a) for a in list(acts or [])]


def _fp(acts: Any) -> str:
    try:
        return hashlib.sha256(json.dumps(as_dicts(acts), sort_keys=True, default=str).encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001 - not serializable: its repr
        return hashlib.sha256(repr(acts).encode()).hexdigest()[:16]


async def run_policy(task: Any, policies: Any, wd: Path, *, max_steps: int | None = None) -> Any:
    """One episode on a fresh World of ``task``: ``policies`` is an action list (scripted) or ``{actor: Policy}``. Returns
    the task's grade report. The World is closed whatever happens."""
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode

    if isinstance(policies, dict):
        pols, steps = policies, max_steps or 1000
    else:
        acts = as_actions(policies)
        pols, steps = {"agent": ScriptedPolicy("agent", acts)}, max_steps or max(50, 2 * len(acts))
    w = await task.open(wd / "work")
    try:
        await run_episode(w, pols, limits=EpisodeLimits(max_steps=steps), reset=False)
        return await task.grade(w, wd / "grade")
    finally:
        await w.close()


def _cleanup(task: Any) -> None:
    c = getattr(task, "cleanup", None)
    if callable(c):
        with contextlib.suppress(Exception):
            c()


def _gates(rep: Any) -> dict[str, str]:
    return {str(k): str(v) for k, v in (getattr(rep, "gates", None) or {}).items()}


# ---- requirements ------------------------------------------------------------------------------------------------------
def _probe(cmd: list[str], *, want: str | None = None, timeout: float = 10.0) -> bool:
    if not shutil.which(cmd[0]):
        return False
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):  # a hung daemon is a missing one (V22)
        return False
    return r.returncode == 0 and (want is None or want in r.stdout)


def missing_requirements(m: PackManifest, allow: tuple[str, ...] = ()) -> list[str]:
    """What this machine lacks of ``requires``: docker and gpu are detected (``docker info``, ``nvidia-smi -L``, each with a
    timeout); network and paid are never assumed, the caller vouches with ``--allow``."""
    miss = []
    for r in m.requires:
        if r in allow:
            continue
        if r == "docker" and not _probe(["docker", "info"]):
            miss.append("docker")
        elif r == "gpu" and not _probe(["nvidia-smi", "-L"], want="GPU"):
            miss.append("gpu")
        elif r in ("paid", "network"):
            miss.append(r)
    return miss


# ---- fixture -----------------------------------------------------------------------------------------------------------
def _split_cases(cases: Any) -> tuple[list[Any], list[dict]]:
    cases = list(cases or [])
    return [c for c in cases if not (isinstance(c, dict) and "policy" in c)], [c for c in cases if isinstance(c, dict) and "policy" in c]


async def check_fixture(pack_dir: Path, m: PackManifest, *, runs: int = RUNS) -> list[CheckResult]:
    """A case is ``{"name", "task", "actions", "reward", "fails_gate"?}`` (an episode in a World, graded by the task's own
    grader), or ``{"name", "grade": async () -> (reward, {gate: "pass" | "fail"}), "reward", ...}`` for packs whose graders
    work on artifacts (patches, files) and that have no live tasks. ``{"name", "policy", "expect_max"}`` entries are
    adversaries for the semantic level."""
    if not m.hooks.fixtures:
        return [_r("fixture.hook", False, "no hooks.fixtures: designed negatives cannot be checked (a lenient grader would pass)")]
    rng = _Range(m)
    cases, _adv = _split_cases(await call_hook(pack_dir, m.hooks.fixtures))
    if not cases:
        return [_r("fixture.cases", False, "the fixtures hook returned no graded case")]
    runs = max(1, runs) if m.determinism == "seeded" else 1
    bad, flaky = [], []
    n_best = n_worst = 0
    with tempfile.TemporaryDirectory(prefix="pack-fixture-") as d:
        for k, c in enumerate(cases):
            if not isinstance(c, dict):
                bad.append(f"case {k}: not a dict")
                continue
            name = str(c.get("name", f"case{k}"))
            exp = _num(c.get("reward"))
            if exp is None or not rng.inside(exp):
                bad.append(f"{name}: expected reward {c.get('reward')!r} is not a finite number inside reward.range {rng}")
                continue
            if "grade" in c:
                g = c["grade"]
                if m.hooks.tasks:  # G2 D2b: a self-graded case never calls the pack's grader
                    bad.append(f"{name}: a self-graded case on a pack with live tasks; give it a task and actions so the pack's own grader decides it")
                    continue
                if not _in_pkg(str(getattr(g, "__module__", "") or ""), m.package):
                    bad.append(f"{name}: its grade callable lives in {getattr(g, '__module__', None)!r}, outside the pack")
                    continue

                async def one(i: int, g: Any = g) -> tuple[Any, dict[str, str]]:
                    reward, gates = await g()
                    return reward, {str(a): str(b) for a, b in (gates or {}).items()}
            elif "task" in c and "actions" in c:
                async def one(i: int, c: dict = c, k: int = k) -> tuple[Any, dict[str, str]]:
                    rep = await run_policy(c["task"], c["actions"], Path(d) / f"{k}-{i}")
                    return rep.reward, _gates(rep)
            else:
                bad.append(f"{name}: a case needs task + actions (or a grade callable)")
                continue
            n_best += rng.at(exp, rng.best)
            n_worst += rng.at(exp, rng.worst)
            got = [await one(i) for i in range(runs)]
            reward, gates = got[0]
            r = _num(reward)
            if r is None:
                bad.append(f"{name}: graded {reward!r}, not a finite number")
            elif not rng.inside(r):
                bad.append(f"{name}: graded {r}, outside reward.range {rng}")
            elif not abs(r - exp) <= EPS:
                bad.append(f"{name}: graded {r}, expected {exp}")
            fails = c.get("fails_gate")
            if fails and gates.get(fails) != "fail":  # V8: a gate the grader never reports is not a failed gate
                bad.append(f"{name}: gate {fails} is {gates.get(fails, 'absent')}, expected fail (reported gates: {sorted(gates)})")
            if len({(repr(_num(x[0])), tuple(sorted(x[1].items()))) for x in got}) > 1:
                flaky.append(f"{name}: {runs} grades on fresh Worlds differ: {[(x[0], x[1]) for x in got][:3]}")
    out = [_r("fixture.cases", not bad, "; ".join(bad)[:2000] or f"{len(cases)} case(s) graded exactly as expected")]
    kinds = ([f"no case expects the best reward {rng.best:g}"] if not n_best else []) + \
            ([f"no designed negative expects the worst reward {rng.worst:g}"] if not n_worst else [])
    out.append(_r("fixture.kinds", not kinds, "; ".join(kinds) or f"{n_best} case(s) at the best end, {n_worst} at the worst"))
    det = "declared nondeterministic: graded once" if m.determinism != "seeded" else f"{len(cases)} case(s) x {runs} grades on fresh Worlds"
    out.append(_r("fixture.determinism", not flaky, "; ".join(flaky)[:1500] or det))
    return out


# ---- semantic ----------------------------------------------------------------------------------------------------------
async def _adversary_specs(pack_dir: Path, m: PackManifest) -> tuple[list[tuple[str, Any, float, Any]], dict[str, Any], list[str]]:
    """(name, policy, expect_max, only_task) from ``fixtures`` ``policy`` entries and bounded ``policies`` entries, the
    pack's policies by id, and malformed entries."""
    specs: list[tuple[str, Any, Any, Any]] = []
    refs: dict[str, Any] = {}
    raw_pols = list(await call_hook(pack_dir, m.hooks.policies) or []) if m.hooks.policies else []
    for p in raw_pols:
        ref = p.get("policy") if isinstance(p, dict) else p
        if hasattr(ref, "id"):
            refs[str(ref.id)] = ref
        bound = p.get("expect_max") if isinstance(p, dict) else getattr(p, "expect_max", None)
        if bound is not None or (isinstance(p, dict) and "expect_max" in p):
            name = (p.get("name") if isinstance(p, dict) else None) or getattr(ref, "id", None) or "policy"
            specs.append((str(name), ref, bound, None))
    if m.hooks.fixtures:
        _, adv = _split_cases(await call_hook(pack_dir, m.hooks.fixtures))
        for k, c in enumerate(adv):
            pol = c["policy"]
            name = c.get("name") or (pol if isinstance(pol, str) else getattr(pol, "id", None)) or f"adversary{k}"
            specs.append((str(name), pol, c.get("expect_max"), c.get("task")))
    rng = _Range(m)
    ok, bad = [], []
    for name, pol, bound, only in specs:
        b = _num(bound)
        if b is None or not rng.inside(b):
            bad.append(f"{name}: expect_max {bound!r} is not a finite number inside reward.range {rng}")
            continue
        ok.append((name, pol, b, only))
    return ok, refs, bad


def _policies_for(spec: Any, task: Any, refs: dict[str, Any], wd: Path) -> Any:
    if isinstance(spec, str):
        if spec not in refs:
            raise LookupError(f"no policy {spec!r} in hooks.policies (known: {sorted(refs)})")
        spec = refs[spec]
    if hasattr(spec, "make"):
        return spec.make(SimpleNamespace(task=task, id="pack-check", n=0, dir=wd, sample=None, policy=spec, ledger=None, scopes=[]))
    if callable(spec):
        return as_actions(spec(task))
    raise TypeError(f"an adversary policy is a PolicyRef, a policy id or a callable task -> actions, got {type(spec).__name__}")


async def _gold_noop(task: Any, gold: Any, wd: Path, rng: _Range, runs: int, tid: str, gold_bad: list, noop_bad: list, flaky: list) -> None:
    golds = [await run_policy(task, gold, wd / f"gold{i}") for i in range(runs)]
    noop = await run_policy(task, [], wd / "noop")
    g = _num(golds[0].reward)
    if g is None or not rng.inside(g) or not rng.at(g, rng.best):
        gold_bad.append(f"{tid}: gold scored {golds[0].reward!r}, expected the best end {rng.best:g} of {rng}")
    n = _num(noop.reward)
    if n is None or not rng.inside(n) or not rng.at(n, rng.worst):  # V21: the untouched task sits at the floor, not just below the top
        noop_bad.append(f"{tid}: the untouched task scores {noop.reward!r}, expected the worst end {rng.worst:g} of {rng}")
    if len({(repr(_num(r.reward)), tuple(sorted(_gates(r).items()))) for r in golds}) > 1:
        flaky.append(f"{tid}: {runs} gold runs on fresh Worlds disagree: {[(r.reward, _gates(r)) for r in golds]}"[:400])


def _sig(task: Any, gold: Any, k: int) -> tuple[str, str, str]:
    return (str(getattr(task, "id", k)), str(getattr(task, "instruction", "")), _fp(gold))


async def check_semantic(pack_dir: Path, m: PackManifest, *, runs: int = RUNS, seeds: tuple[int, ...] = GEN_SEEDS) -> list[CheckResult]:
    from crucible.bench.run import _oracle_actions

    if not m.hooks.tasks:
        return [_r("semantic.tasks", False, "no hooks.tasks: the gold/no-op check cannot run (declare the pack's live tasks)")]
    rng = _Range(m)
    seeded = m.determinism == "seeded"
    runs = max(1, runs) if seeded else 1
    pairs = list(await call_hook(pack_dir, m.hooks.tasks) or [])
    if not pairs:
        return [_r("semantic.tasks", False, "the tasks hook returned no task")]
    out = [_r("semantic.tasks", True, f"{len(pairs)} task(s)")]
    advs, refs, adv_bad = await _adversary_specs(pack_dir, m)
    gold_bad: list[str] = []
    noop_bad: list[str] = []
    det_bad: list[str] = []
    gen_bad: list[str] = []
    n_adv = 0
    with tempfile.TemporaryDirectory(prefix="pack-check-") as d:

        async def adversaries(task: Any, tid: str, wd: Path) -> None:
            nonlocal n_adv
            for j, (name, pol, bound, only) in enumerate(advs):
                if only is not None and only is not task:
                    continue
                try:
                    rep = await run_policy(task, _policies_for(pol, task, refs, wd / f"adv{j}"), wd / f"adv{j}")
                except Exception as e:  # noqa: BLE001 - an adversary that cannot run is unmeasured: a failure
                    adv_bad.append(f"{name} on {tid}: {type(e).__name__}: {e}"[:300])
                    continue
                n_adv += 1
                r = _num(rep.reward)
                if r is None or not rng.within_bound(r, bound):
                    adv_bad.append(f"{name} on {tid}: scored {rep.reward!r}, bound {'<=' if rng.hib else '>='} {bound:g}")

        sigs = []
        try:
            for k, (task, gold) in enumerate(pairs):
                tid = str(getattr(task, "id", k))
                sigs.append(_sig(task, gold, k))
                await _gold_noop(task, gold, Path(d) / f"t{k}", rng, runs, tid, gold_bad, noop_bad, det_bad)
                await adversaries(task, tid, Path(d) / f"t{k}")
        finally:
            for task, _ in pairs:
                _cleanup(task)
        if seeded:  # the task list is a function of the pack (V9: a task drawn from SystemRandom passed)
            again = list(await call_hook(pack_dir, m.hooks.tasks) or [])
            try:
                sigs2 = [_sig(t, g, k) for k, (t, g) in enumerate(again)]
            finally:
                for t, _ in again:
                    _cleanup(t)
            if sigs2 != sigs:
                diff = [a[0] if a[0] == b[0] else f"{a[0]} != {b[0]}" for a, b in zip(sigs, sigs2, strict=False) if a != b]
                det_bad.append(f"tasks() built different tasks on a second call (id, instruction or gold): {diff[:3] or f'{len(sigs)} vs {len(sigs2)} tasks'}")
        if m.hooks.generator:
            gen = resolve(pack_dir, m.hooks.generator)
            for s in seeds:
                made: list[Any] = []
                try:
                    t1 = gen(s)
                    made.append(t1)
                    tid = str(getattr(t1, "id", f"seed {s}"))
                    acts = _oracle_actions(t1)
                    if seeded:
                        t2 = gen(s)
                        made.append(t2)
                        if _sig(t1, acts, s) != _sig(t2, _oracle_actions(t2), s):
                            det_bad.append(f"generator({s}) built two different items (id, instruction or oracle)")
                    if acts is None:  # V13: bench run rejects every fresh item without an oracle
                        gen_bad.append(f"generator({s}) -> {tid}: no oracle (oracle_actions(), oracle or bench_gold); bench run rejects such items")
                        continue
                    nb: list[str] = []
                    await _gold_noop(t1, acts, Path(d) / f"g{s}", rng, runs, tid, nb, nb, det_bad)
                    gen_bad += nb
                    await adversaries(t1, tid, Path(d) / f"g{s}")
                finally:
                    for t in made:
                        _cleanup(t)
    out.append(_r("semantic.gold", not gold_bad, "; ".join(gold_bad)[:1500] or f"{len(pairs)} task(s): gold reaches {rng.best:g}"))
    out.append(_r("semantic.noop", not noop_bad, "; ".join(noop_bad)[:1500] or f"{len(pairs)} task(s): the untouched task scores {rng.worst:g}"))
    out.append(_r("semantic.determinism", not det_bad, "; ".join(det_bad)[:1500]
                  or ("declared nondeterministic: run once" if not seeded else f"tasks rebuilt identically; gold x {runs} on fresh Worlds")))
    if m.hooks.generator:
        out.append(_r("semantic.generator", not gen_bad, "; ".join(gen_bad)[:1500] or f"seeds {list(seeds)}: oracle reaches {rng.best:g}, no-op {rng.worst:g}"))
    out.append(_r("semantic.adversaries", not adv_bad, "; ".join(adv_bad)[:1500]
                  or (f"{len(advs)} adversar{'y' if len(advs) == 1 else 'ies'}, {n_adv} run(s) within bounds" if advs else "none declared")))
    return out


# ---- accept ------------------------------------------------------------------------------------------------------------
async def check_accept(pack_dir: Path, m: PackManifest, *, allow: tuple[str, ...] = (), n_fresh: int = 1) -> list[CheckResult]:
    """The M9 Acceptance Contract on the pack's tasks (G2 pick 3): fixture tasks with their designed negatives, the
    ``tasks`` anchors and ``n_fresh`` generated items, through ``crucible.studio.accept_adapter``."""
    from crucible.accept.model import Need
    from crucible.accept.run import accept_task, verdict
    from crucible.studio.accept_adapter import pack_accept_tasks

    tasks = await pack_accept_tasks(pack_dir, m, n_fresh=n_fresh)
    if not tasks:
        return [_r("accept.contract", False, "no task to accept (no hooks.tasks, generator or fixture tasks)")]
    needs = [n for n, on in ((Need.DOCKER, "docker" in allow or "docker" in m.requires), (Need.PAID, "paid" in allow)) if on]
    rejected, leak, states = [], [], {}
    try:
        with tempfile.TemporaryDirectory(prefix="pack-accept-") as d:
            for t in tasks:
                rep = await accept_task(t, needs=needs, cache_root=Path(d), resolve=lambda s: False, reuse=False)
                v = verdict(rep)
                states[t.task_id] = v["state"]
                if v["state"] == "rejected":
                    rejected.append(f"{t.task_id}: {v['blocking']}")
                a5 = rep.results.get("A5")
                if a5 is None or str(a5.status) not in ("pass", "na"):
                    leak.append(f"{t.task_id}: A5 {a5.status if a5 else 'not run'} {a5.reasons[:2] if a5 else ''}")
    finally:
        for t in tasks:
            t.cleanup()
    return [_r("accept.leak", not leak, "; ".join(leak)[:1500] or f"{len(tasks)} task(s): nothing hidden is readable by the agent"),
            _r("accept.contract", not rejected, "; ".join(rejected)[:2000] or ", ".join(f"{k}: {v}" for k, v in states.items()))]


# ---- check -------------------------------------------------------------------------------------------------------------
async def _level(name: str, fn: Any) -> list[CheckResult]:
    """A hook that raises is a failed check with its message, never a traceback (V14)."""
    try:
        return await fn()
    except Exception as e:  # noqa: BLE001
        return [CheckResult(f"{name}.error", "fail", f"{type(e).__name__}: {e}"[:600])]


async def check(ref: str | Path, *, level: str = "static", allow: tuple[str, ...] = (), allow_skip: bool = False, publish: bool = False,
                runs: int = RUNS, seeds: tuple[int, ...] = GEN_SEEDS) -> PackReport:
    """Levels are cumulative: ``fixture`` runs static + fixture; ``semantic`` adds the live tasks; ``accept`` adds M9."""
    if level not in LEVELS:
        raise ValueError("level: " + " | ".join(LEVELS))
    pack_dir = find_pack(ref)
    want = LEVELS[: LEVELS.index(level) + 1]
    try:
        package = str((yaml.safe_load((pack_dir / "pack.yaml").read_text()) or {}).get("package") or "")
    except Exception:  # noqa: BLE001 - check_static reports it
        package = ""
    skipped: list[tuple[str, str]] = []
    with isolated(pack_dir, package if re.fullmatch(r"[A-Za-z_]\w*", package) else ""):
        results, m = check_static(pack_dir)
        if publish:
            results = [CheckResult(r.id, "fail", r.detail) if r.status == "warn" and r.id in PUBLISH else r for r in results]
        reached = m is not None and all(r.status != "fail" for r in results)
        for lv in want[1:]:
            if not reached or m is None:
                why = "static failed" if m is not None else "the manifest does not load"
                skipped.append((lv, f"not reached: {why}"))
                results.append(_r(f"{lv}.not_reached", True, why, skip=True))
                continue
            if lv in LIVE and (miss := missing_requirements(m, allow)):
                why = f"live tasks need {', '.join(miss)} (pass --allow {','.join(miss)} where they are)"
                skipped.append((lv, why))
                ids = ("semantic.gold", "semantic.noop") if lv == "semantic" else ("accept.contract",)
                results += [_r(i, True, f"skipped: {why}", skip=True) for i in ids]
                continue
            if lv == "fixture":
                results += await _level("fixture", lambda m=m: check_fixture(pack_dir, m, runs=runs))
            elif lv == "semantic":
                results += await _level("semantic", lambda m=m: check_semantic(pack_dir, m, runs=runs, seeds=seeds))
            else:
                results += await _level("accept", lambda m=m: check_accept(pack_dir, m, allow=allow))
    return PackReport(m.name if m else str(ref), str(pack_dir), level, results, skipped=skipped, allow_skip=allow_skip, publish=publish)


def check_in_subprocess(ref: str | Path, *, level: str = "static", allow: tuple[str, ...] = (), allow_skip: bool = False, publish: bool = False,
                        timeout: float = 1800.0) -> PackReport:
    """One pack checked in its own interpreter (G2 pick 5): no module cache, ``sys.path`` or registry state is shared with
    any other pack. An interpreter that crashes or prints no report is an ``internal.error`` (exit 3)."""
    cmd = [sys.executable, "-m", "crucible.cli.main", "pack", "check", str(ref), "--level", level, "--json"]
    cmd += ["--allow", ",".join(allow)] if allow else []
    cmd += ["--allow-skip"] if allow_skip else []
    cmd += ["--publish"] if publish else []
    stderr = ""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        stderr = p.stderr
        j = json.loads(p.stdout)
        rep = PackReport(j["pack"], j["path"], j["level"], [CheckResult(**r) for r in j["results"]], skipped=[tuple(s) for s in j.get("skipped", [])],
                         allow_skip=allow_skip, publish=publish)
        if p.returncode not in (EXIT_OK, EXIT_FAIL, EXIT_INCOMPLETE):
            rep.results.append(CheckResult("internal.error", "fail", f"exit {p.returncode}: {stderr.strip()[-400:]}"))
        return rep
    except (json.JSONDecodeError, KeyError, TypeError, subprocess.TimeoutExpired) as e:
        return PackReport(str(ref), str(ref), level, [CheckResult("internal.error", "fail", f"{type(e).__name__}: {stderr.strip()[-600:]}")])


def all_packs() -> list[Path]:
    return [d for d in sorted(PACKS_ROOT.iterdir()) if (d / "pack.yaml").exists() and d.name != "_template"] if PACKS_ROOT.exists() else []


def policy_refs(raw: Any) -> list[Any]:
    """``hooks.policies`` entries as ``PolicyRef`` s: a bounded entry ``{"policy", "expect_max"}`` unwraps to its policy
    (bench and build run it like any other); a callable ``task -> actions`` becomes a scripted ``PolicyRef``."""
    from crucible.actors.policy import ScriptedPolicy
    from crucible.runtime.harness.native import PolicyRef

    out = []
    for k, p in enumerate(list(raw or [])):
        name = None
        if isinstance(p, dict):
            name, p = p.get("name"), p["policy"]
        if isinstance(p, str):
            continue  # a bound on another entry of the same list (``{"policy": "scripted:noop", "expect_max": 0}``)
        if hasattr(p, "make"):
            out.append(p)
        elif callable(p):
            out.append(PolicyRef(id=str(name or getattr(p, "__name__", f"policy{k}")),
                                 make=lambda a, p=p: {"agent": ScriptedPolicy("agent", as_actions(p(a.task)))}))
        else:
            raise TypeError(f"hooks.policies entry {k}: a PolicyRef, a callable task -> actions, or {{'policy': ..., 'expect_max': ...}}")
    return out
