"""``crucible pack new`` and ``crucible pack check``.

**new.** Copies ``packs/_template`` with ``__PACK_NAME__`` / ``__PACK_PKG__`` substituted in file contents and names
(OpenEnv ``openenv init``). The writer is NeMo-Gym's conflict-safe scaffold (``environment/scaffold.py:909-950``):
- it plans every file first;
- an identical existing file is a no-op;
- any differing file aborts the whole write before anything is written;
- symlinks are refused, and new files are opened with ``"x"``.
The template's version is recorded in ``pack.yaml`` for a later 3-way upgrade (karotte).

**check.** Levels in the OpenEnv ``validate --level`` shape:
- **static:** the manifest parses and its contract holds; no ``TODO`` placeholder or ``unknown`` license is left (NeMo-Gym
  ``publication.py``); the package and README exist; the entry point and every hook import and are callable; every
  ingest glob matches a file.
- **fixture** (NeMo-Gym's scaffolded verifier fixtures): every case from ``hooks.fixtures`` grades to its expected
  reward exactly, and fails the gate it names. Designed negatives prove the grader is not lenient.
- **semantic** (model-free; verifiers ``vf-validate``, OpenEnv RFC 008): for each task from ``hooks.tasks``, the gold
  actions reach the top of ``reward.range`` and the untouched task stays below it.
A pack's ``requires`` (docker, gpu, network, paid) describes its live tasks: when this machine cannot satisfy it, the
semantic check is ``skip`` with the reason (fixtures still run); ``--allow`` names the requirements the caller vouches for.
Each check id maps to fail or warn through a versioned severity policy. The exit code is 0 for pass or warn, 1 for any
fail, and 3 for an internal error.
"""

from __future__ import annotations

import importlib
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from crucible.core.schema.pack import PackManifest

PACKS_ROOT = Path(__file__).resolve().parents[3] / "packs"
TEMPLATE = PACKS_ROOT / "_template"
TEMPLATE_VERSION = 1
SEVERITY_V1 = {  # check id -> "fail" | "warn"
    "static.manifest": "fail", "static.placeholders": "fail", "static.license": "fail", "static.layout": "fail", "static.readme": "warn",
    "static.entry_point": "fail", "static.hooks": "fail", "static.ingest": "fail", "semantic.gold": "fail", "semantic.noop": "fail",
    "semantic.tasks": "warn", "fixture.cases": "fail", "fixture.hook": "warn",
}
NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


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

    @property
    def ok(self) -> bool:
        return not any(r.status == "fail" for r in self.results)

    def to_json(self) -> dict[str, Any]:
        return {"pack": self.pack, "path": self.path, "level": self.level, "ok": self.ok, "policy": "severity-v1",
                "results": [asdict(r) for r in self.results], "problems": [f"{r.id}: {r.detail}" for r in self.results if r.status == "fail"]}


# ---- locate and load ----------------------------------------------------------------------------------------------------
def find_pack(ref: str | Path) -> Path:
    """A pack directory from a path, or from a name searched in ``packs/`` (``toy`` -> ``packs/_toy``)."""
    p = Path(ref)
    if (p / "pack.yaml").exists():
        return p.resolve()
    for d in sorted(PACKS_ROOT.iterdir()) if PACKS_ROOT.exists() else []:
        m = d / "pack.yaml"
        if m.exists() and d.name != "_template":
            try:
                if yaml.safe_load(m.read_text()).get("name") == str(ref):
                    return d.resolve()
            except yaml.YAMLError:
                continue
    raise ValueError(f"no pack {ref!r}: give a directory with pack.yaml, or the name of a pack under {PACKS_ROOT}")


def load_manifest(pack_dir: Path) -> PackManifest:
    return PackManifest.model_validate(yaml.safe_load((Path(pack_dir) / "pack.yaml").read_text()))


def resolve(pack_dir: Path, ref: str) -> Any:
    """``module:attr`` from the pack's ``src/`` (put on ``sys.path`` once; installed packs import as they are)."""
    src = str(Path(pack_dir) / "src")
    if Path(src).is_dir() and src not in sys.path:
        sys.path.insert(0, src)
    mod, attr = ref.split(":", 1)
    obj: Any = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


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
    res = write_scaffold(plan_scaffold(name, dest), Path(dest))
    return {"pack": name, "path": str(Path(dest).resolve()), **res,
            "next": [f"crucible pack check {dest}", f"crucible pack check {dest} --level semantic", f"crucible bench run {dest}"]}


# ---- check -------------------------------------------------------------------------------------------------------------
def _r(cid: str, ok: bool, detail: str = "", *, skip: bool = False) -> CheckResult:
    if skip:
        return CheckResult(cid, "skip", detail)
    return CheckResult(cid, "pass" if ok else SEVERITY_V1.get(cid, "fail"), detail)


def _local_paths(d: dict[str, Any]) -> list[str]:
    return [v for v in d.values() if isinstance(v, str) and v.startswith(("./", "../")) and not any(c in v for c in "*?[")]


def check_static(pack_dir: Path) -> tuple[list[CheckResult], PackManifest | None]:
    out: list[CheckResult] = []
    pack_dir = Path(pack_dir)
    try:
        m = load_manifest(pack_dir)
        out.append(_r("static.manifest", True, f"{m.name} {m.version}"))
    except Exception as e:  # noqa: BLE001 - reported as a check failure
        return [_r("static.manifest", False, f"{type(e).__name__}: {e}"[:500])], None
    texts = [pack_dir / "pack.yaml", pack_dir / "README.md", pack_dir / "pyproject.toml"]
    left = [f"{p.name}:{i + 1}" for p in texts if p.exists() for i, line in enumerate(p.read_text().splitlines()) if "TODO" in line]
    out.append(_r("static.placeholders", not left, ("placeholders left: " + ", ".join(left[:10])) if left else "none"))
    out.append(_r("static.license", m.license.strip().lower() not in ("", "unknown", "todo"), m.license))
    pkg = pack_dir / "src" / m.package
    out.append(_r("static.layout", pkg.is_dir() and (pack_dir / "pyproject.toml").exists(), f"src/{m.package}, pyproject.toml"))
    out.append(_r("static.readme", (pack_dir / "README.md").exists(), "README.md"))
    try:
        reg = resolve(pack_dir, m.entry_point)
        out.append(_r("static.entry_point", callable(reg), m.entry_point))
    except Exception as e:  # noqa: BLE001
        out.append(_r("static.entry_point", False, f"{m.entry_point}: {type(e).__name__}: {e}"[:300]))
    bad = []
    for k, ref in m.hooks.model_dump().items():
        if ref is None:
            continue
        try:
            if not callable(resolve(pack_dir, ref)):
                bad.append(f"{k} ({ref}) is not callable")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{k} ({ref}): {type(e).__name__}: {e}"[:200])
    out.append(_r("static.hooks", not bad, "; ".join(bad) if bad else ", ".join(k for k, v in m.hooks.model_dump().items() if v) or "none declared"))
    missing = []
    for s in m.ingest:
        if s.connector == "file_drop":
            if not list((pack_dir / (s.root or ".")).glob(s.glob or "")):
                missing.append(f"{s.family}: no file matches {s.glob}")
        else:
            from crucible.core.registry import get_registry

            try:
                get_registry().get("connector", s.connector)
            except Exception:  # noqa: BLE001 - reported
                missing.append(f"{s.family}: no connector named {s.connector!r}")
            missing += [f"{s.family}: {v} does not exist" for v in _local_paths({**s.init, **s.scope}) if not (pack_dir / v).exists()]
    out.append(_r("static.ingest", not missing, "; ".join(missing) if missing else f"{len(m.ingest)} source(s)"))
    return out, m


async def _grade(task: Any, actions: list, wd: Path) -> Any:
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode

    w = await task.open(wd / "work")
    try:
        await run_episode(w, {"agent": ScriptedPolicy("agent", list(actions))}, limits=EpisodeLimits(max_steps=max(50, 2 * len(actions))), reset=False)
        return await task.grade(w, wd / "grade")
    finally:
        await w.close()


def missing_requirements(m: PackManifest, allow: tuple[str, ...] = ()) -> list[str]:
    import shutil
    import subprocess

    miss = []
    for r in m.requires:
        if r in allow:
            continue
        if r == "docker" and not (shutil.which("docker") and subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0):
            miss.append("docker")
        elif r in ("gpu", "paid", "network"):
            miss.append(r)  # never assumed: the caller vouches with --allow
    return miss


async def check_fixture(pack_dir: Path, m: PackManifest) -> list[CheckResult]:
    """A case is ``{"name", "task", "actions", "reward", "fails_gate"?}`` (an episode in a World), or ``{"name",
    "grade": async () -> (reward, {gate: "pass" | "fail"}), "reward", ...}`` for packs whose graders work on artifacts
    (patches, files) rather than live Worlds."""
    if not m.hooks.fixtures:
        return [_r("fixture.hook", False, "no hooks.fixtures: designed negatives cannot be checked")]
    res = resolve(pack_dir, m.hooks.fixtures)()
    cases = await res if hasattr(res, "__await__") else res
    bad = []
    with tempfile.TemporaryDirectory(prefix="pack-fixture-") as d:
        for k, c in enumerate(cases):
            if "grade" in c:
                reward, gates = await c["grade"]()
            else:
                rep = await _grade(c["task"], c["actions"], Path(d) / str(k))
                reward, gates = rep.reward, rep.gates
            fails = c.get("fails_gate")
            if reward is None or abs(reward - float(c["reward"])) > 1e-6:
                bad.append(f"{c['name']}: graded {reward}, expected {c['reward']}")
            elif fails and gates.get(fails) == "pass":
                bad.append(f"{c['name']}: gate {fails} passed, expected it to fail")
    return [_r("fixture.cases", not bad, "; ".join(bad) or f"{len(cases)} case(s) graded exactly as expected")]


async def check_semantic(pack_dir: Path, m: PackManifest) -> list[CheckResult]:
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode

    if not m.hooks.tasks:
        return [_r("semantic.tasks", False, "no hooks.tasks: the gold/no-op check cannot run")]
    tasks = resolve(pack_dir, m.hooks.tasks)()
    hi = m.reward.range[1]
    gold_bad, noop_bad = [], []
    with tempfile.TemporaryDirectory(prefix="pack-check-") as d:
        for k, (task, gold) in enumerate(tasks):
            rewards = []
            for name, script in (("gold", list(gold)), ("noop", [])):
                wd = Path(d) / f"{k}-{name}"
                w = await task.open(wd / "work")
                try:
                    await run_episode(w, {"agent": ScriptedPolicy("agent", script)}, limits=EpisodeLimits(max_steps=max(50, 2 * len(script))), reset=False)
                    rewards.append((await task.grade(w, wd / "grade")).reward)
                finally:
                    await w.close()
            tid = getattr(task, "id", str(k))
            if rewards[0] is None or rewards[0] < hi - 1e-9:
                gold_bad.append(f"{tid}: gold scored {rewards[0]}")
            if rewards[1] is not None and rewards[1] >= hi - 1e-9:
                noop_bad.append(f"{tid}: the untouched task already scores {rewards[1]}")
            cleanup = getattr(task, "cleanup", None)
            if callable(cleanup):
                cleanup()
    return [_r("semantic.gold", not gold_bad, "; ".join(gold_bad) or f"{len(tasks)} task(s): gold reaches {hi}"),
            _r("semantic.noop", not noop_bad, "; ".join(noop_bad) or f"{len(tasks)} task(s): no-op stays below {hi}")]


LEVELS = ("static", "fixture", "semantic")


async def check(ref: str | Path, *, level: str = "static", allow: tuple[str, ...] = ()) -> PackReport:
    """Levels are cumulative: ``fixture`` runs static + fixture; ``semantic`` runs all three."""
    if level not in LEVELS:
        raise ValueError("level: " + " | ".join(LEVELS))
    pack_dir = find_pack(ref)
    results, m = check_static(pack_dir)
    if m is not None and level != "static" and all(r.status != "fail" for r in results):
        results += await check_fixture(pack_dir, m)
        if level == "semantic":
            miss = missing_requirements(m, allow)  # live tasks need what the pack declares; fixtures do not
            if miss:
                results.append(_r("semantic.gold", True, f"skipped: live tasks need {', '.join(miss)} (pass --allow {','.join(miss)} where they are)",
                                  skip=True))
            else:
                results += await check_semantic(pack_dir, m)
    return PackReport(m.name if m else str(ref), str(pack_dir), level, results)


def all_packs() -> list[Path]:
    return [d for d in sorted(PACKS_ROOT.iterdir()) if (d / "pack.yaml").exists() and d.name != "_template"] if PACKS_ROOT.exists() else []


