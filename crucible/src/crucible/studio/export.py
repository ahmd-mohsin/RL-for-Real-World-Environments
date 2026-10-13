"""``crucible export``: a pack's tasks or a run's episodes in the formats other runtimes and trainers read (M11 seams).

| target | what is written | needs |
|---|---|---|
| ``canonical`` | per task: the gold actions replayed into ``reward.json`` + ``steps.jsonl``. Every runtime adapter must reproduce these (the M11 consistency gate) | ``hooks.tasks``; tasks with subgoals or a plan |
| ``atif`` | per episode of an M10 run: its ATIF trajectory (Harbor's schema), validated by Harbor's validator | ``--run``; Harbor installed (or ``--no-validate``) |
| ``nemo-gym`` | a launcher for ``crucible.interop.nemo_gym.server`` over every task of the pack; the app is built here to prove the tasks load | ``hooks.tasks`` |
| ``openenv`` | a launcher module exposing ``make_environment`` over every task (``reset(task_index=)``) and the generator (``reset(item_seed=)``), its ASGI ``app`` (OpenEnv's ``create_app``) and ``openenv.yaml`` (``app: openenv_env:app``, port 8000) | ``hooks.tasks``; ``openenv`` (its own environment) |

Both pack exports are self-contained (M14 sweep G2 D11): the pack is copied to ``out/pack`` and the launchers find it
relative to themselves, so the directory can move to another machine (it still needs ``crucible`` installed there).

M15 audit (H2 V2/V3/V9/V10/V11/V12/V20):
- the bundle is an allowlist (``pack.yaml``, README, LICENSE, ``pyproject.toml``, the pack's package under ``src/`` and
  its ``data`` / ``worlds`` / ``families`` dirs) without dotfiles, ``*.env*``, key files, the raw files its ingest sources
  read or its ``redaction_env``; the copy is scanned for credentials and refused on a hit;
- export never deletes what it did not write: an out dir that is, holds or sits inside the source pack is refused, and an
  existing ``out/pack`` is replaced only when it carries the bundle marker, by an atomic swap of a finished copy;
- task ids become directory names through ``safe_dir_name`` (no ``..``, no collisions);
- an async ``hooks.tasks`` is awaited, here and in the generated launchers;
- ``canonical`` goes on past a failing task, reports it and cleans up every task;
- ``atif`` refuses a dir with no trajectories, or no validator, and reports invalid files as a failure.
"""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

TARGETS = ("canonical", "atif", "nemo-gym", "openenv")
BUNDLE_MARKER = "crucible_bundle.json"  # written into every out/pack this module made; nothing else is ever replaced
BUNDLE_FILES = ("pack.yaml", "README.md", "README.rst", "README.txt", "README", "pyproject.toml", "LICENSE", "LICENSE.md", "LICENSE.txt",
                "NOTICE", "NOTICE.md")
BUNDLE_DIRS = ("data", "worlds", "families")  # beside src/<package>
# never shipped, wherever they sit (V3): dotfiles (.env, .netrc, .git ...), env files, key material
SECRET_FILES = ("*.env", "*.env.*", "*.key", "*.pem", "*.p12", "*.pfx", "*.jks", "*.keystore", "id_rsa*", "id_ed25519*", "id_ecdsa*",
                "*.secret", "*.secrets", "credentials.json", "credentials.txt", "*.kdbx", "*.htpasswd")
CACHE = ("__pycache__", "*.pyc", "*.pyo", "*.egg-info")
_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def safe_dir_name(tid: str) -> str:
    """A task id as one directory name (V9): ``[A-Za-z0-9._-]`` only, never ``.`` / ``..`` / hidden, plus a short hash of the
    full id, so ``acme/t1`` and ``acme_t1`` stay two directories."""
    slug = _SLUG.sub("_", str(tid)).strip("._-")[:80] or "task"
    return f"{slug}-{hashlib.sha256(str(tid).encode()).hexdigest()[:8]}"


def task_dirs(ids: list[str], root: Path) -> dict[str, Path]:
    """One directory under ``root`` per task id; a repeated id is refused (V9), and every path is checked to stay inside."""
    seen: dict[str, str] = {}
    out: dict[str, Path] = {}
    base = Path(root).resolve()
    for tid in ids:
        if tid in out:
            raise ValueError(f"task id {tid!r} appears twice: export needs one directory per task")
        name = safe_dir_name(tid)
        if name in seen:
            raise ValueError(f"task ids {seen[name]!r} and {tid!r} map to one directory {name!r}")
        seen[name] = tid
        d = base / name
        if d.parent != base or name in (".", ".."):
            raise ValueError(f"task id {tid!r} leaves the export directory")
        out[tid] = d
    return out


async def awaited(x: Any) -> Any:
    """A hook's result, awaited when it is a coroutine (V10: coding's ``hooks.tasks`` is async)."""
    return await x if inspect.isawaitable(x) else x


def _pairs(raw: Any) -> list[tuple[Any, Any]]:
    return [x if isinstance(x, tuple) else (x, None) for x in raw or []]


async def export(target: str, *, out: Path, pack: str | Path | None = None, run: Path | None = None, validate: bool = True) -> dict[str, Any]:
    from crucible.studio.pack import find_pack, load_manifest, resolve

    if target not in TARGETS:
        raise ValueError("target: " + " | ".join(TARGETS))
    out = Path(out)
    if target == "atif":
        if run is None:
            raise ValueError("export atif needs --run <run dir>")
        return _atif(Path(run), out, validate=validate)
    if pack is None:
        raise ValueError(f"export {target} needs --pack")
    d = find_pack(pack)
    m = load_manifest(d)
    if not m.hooks.tasks:
        raise ValueError(f"pack {m.name} has no hooks.tasks to export")
    refuse_overlap(d, out)
    ref = m.hooks.tasks
    tasks = _pairs(await awaited(resolve(d, ref)()))  # V10: loaded (and awaited) before anything is written
    from crucible.studio.runs import provenance

    prov = provenance(d, m)  # G2 D12: every export manifest says what made it
    if target == "canonical":
        out.mkdir(parents=True, exist_ok=True)
        res = await _canonical(tasks, out)
        (out / "crucible_export.json").write_text(json.dumps({"target": target, "name": m.name, "version": m.version, "tasks": sorted(res["tasks"]),
                                                              "skipped": sorted(res["skipped"]), "errors": res["errors"], "provenance": prov},
                                                             indent=1, sort_keys=True, default=str) + "\n")
        return res
    try:
        ids = [getattr(t, "id", str(i)) for i, (t, _) in enumerate(tasks)]
        if len(set(ids)) != len(ids):
            raise ValueError(f"pack {m.name}: task ids repeat ({sorted({i for i in ids if ids.count(i) > 1})}); every exported task needs its own id")
        if target == "nemo-gym":
            from crucible.interop.nemo_gym.server import make_app

            make_app(dict(zip(ids, (t for t, _ in tasks), strict=True)))  # proves the tasks load into the server
    finally:
        _cleanup(tasks)
    out.mkdir(parents=True, exist_ok=True)
    head = ("import asyncio\nimport inspect\nimport sys\nfrom pathlib import Path\n\n"
            "PACK = Path(__file__).resolve().parent / 'pack'\n"
            "sys.path.insert(0, str(PACK / 'src'))\nfrom crucible.studio.pack import resolve  # noqa: E402\n\n\n"
            "def _run(x):\n"
            "    \"\"\"A hook's result; an async hook is awaited (also when called inside a running event loop).\"\"\"\n"
            "    if not inspect.isawaitable(x):\n        return x\n"
            "    try:\n        asyncio.get_running_loop()\n    except RuntimeError:\n        return asyncio.run(x)\n"
            "    import concurrent.futures\n\n"
            "    with concurrent.futures.ThreadPoolExecutor(1) as ex:\n        return ex.submit(asyncio.run, x).result()\n\n\n"
            f"def _tasks():\n    return [x if isinstance(x, tuple) else (x, None) for x in _run(resolve(PACK, {ref!r})())]\n\n\n")
    factory_src = ("def make_task(task_index: int = 0, item_seed: int | None = None):\n"
                   "    \"\"\"Any of the pack's tasks (``task_index``) or, with ``item_seed``, a fresh generated item.\"\"\"\n"
                   + (f"    if item_seed is not None:\n        return _run(resolve(PACK, {m.hooks.generator!r})(int(item_seed)))\n" if m.hooks.generator else
                      "    if item_seed is not None:\n        raise ValueError('this pack has no generator')\n")
                   + "    return _tasks()[int(task_index)][0]\n\n\n"
                   "def tasks():\n    return {getattr(t, 'id', str(i)): t for i, (t, _) in enumerate(_tasks())}\n")
    meta = {"name": m.name, "version": m.version, "license": m.license, "reward_range": list(m.reward.range),
            "higher_is_better": m.reward.higher_is_better, "determinism": m.determinism, "requires": list(m.requires),
            "tasks": ids, "generator": bool(m.hooks.generator), "target": target, "provenance": prov}
    if target == "nemo-gym":
        factory = out / "nemo_tasks.py"
        cmd = "PYTHONPATH=. python -m crucible.interop.nemo_gym.server --tasks nemo_tasks:tasks --port 8000"
        files = {factory: f'"""NeMo-Gym resources server factory for the {m.name} pack (crucible export)."""\n' + head + factory_src,
                 out / "serve_nemo.sh": '#!/bin/sh\ncd "$(dirname "$0")"\n' + cmd + "\n",
                 out / "crucible_export.json": json.dumps(meta, indent=1) + "\n"}
        bundle = _bundle(d, m, out)  # self-contained (G2 D11): the pack is copied, the launchers use relative paths only
        for f, text in files.items():
            f.write_text(text)
        return {"target": target, "tasks": sorted(ids), "files": [str(f) for f in files], "bundle": str(bundle), "run": cmd}
    # openenv: every task (reset(task_index=...)) and the generator (reset(item_seed=...)), not task 0 only; ``app`` is the
    # ASGI app OpenEnv's CLI and providers serve (``uvicorn openenv_env:app``), built with OpenEnv's own create_app (V11)
    import yaml

    mod = out / "openenv_env.py"
    src = (f'"""OpenEnv environment for the {m.name} pack (crucible export). Serve with: uvicorn openenv_env:app --port 8000"""\n' + head + factory_src
           + "\n\nfrom crucible.interop.openenv.export import make_environment  # noqa: E402\n"
           "from openenv.core.env_server.http_server import create_app  # noqa: E402\n"
           "from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation  # noqa: E402\n\n"
           "Environment = make_environment(make_task)\n"
           f"app = create_app(Environment, CallToolAction, CallToolObservation, env_name={m.name!r})\n")
    manifest = yaml.safe_dump({"spec_version": 1, "name": m.name, "version": m.version, "type": "space", "runtime": "fastapi",
                               "app": "openenv_env:app", "port": 8000, "description": m.description, "license": m.license,
                               "validation": {"reward_range": list(m.reward.range), "higher_is_better": m.reward.higher_is_better,
                                              "determinism": m.determinism, "requires": list(m.requires),
                                              "tasks": ids, "generator": meta["generator"]}}, sort_keys=False)
    bundle = _bundle(d, m, out)
    mod.write_text(src)
    (out / "openenv.yaml").write_text(manifest)
    (out / "crucible_export.json").write_text(json.dumps(meta, indent=1) + "\n")
    try:
        import openenv  # noqa: F401

        built = True
    except ImportError:
        built = False
    if built:  # the generated module itself is imported: its app must construct
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(f"_crucible_openenv_{m.name}", mod)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        if not callable(getattr(module, "app", None)):
            raise RuntimeError(f"{mod}: app is not an ASGI app")
    return {"target": target, "files": [str(mod), str(out / "openenv.yaml"), str(out / "crucible_export.json")], "bundle": str(bundle), "tasks": ids,
            "built_here": built, "note": None if built else "openenv is not importable here: serve the module in its environment (crucible/.envs/openenv)"}


def _cleanup(tasks: list[tuple[Any, Any]]) -> None:
    for t, _ in tasks:
        if callable(getattr(t, "cleanup", None)):
            try:
                t.cleanup()
            except Exception:  # noqa: BLE001, S110 - cleanup of one task never hides another's
                pass


def _under(p: Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def refuse_overlap(pack_dir: Path, out: Path) -> None:
    """V2: export writes ``out/pack`` and launchers beside it; an out dir that is, holds or sits inside the source pack is
    refused before anything is written or removed."""
    src, dest = Path(pack_dir).resolve(), (Path(out) / "pack").resolve()
    if _under(Path(out), src):
        raise ValueError(f"--out {out} is inside the source pack {src}: choose a directory outside it")
    if _under(src, dest) or _under(dest, src):
        raise ValueError(f"--out {out}: {dest} is (or holds) the source pack {src}; export would replace it. Choose another --out")


def _excluded(pack_dir: Path, m: Any) -> tuple[set[Path], list[Path]]:
    """The files the pack's ingest sources read (raw, unredacted) and the dirs they point at, plus ``redaction_env`` (V3)."""
    from crucible.studio.pack import _paths_in, ingest_value

    files: set[Path] = set()
    dirs: list[Path] = []
    for spec in m.ingest:
        if spec.redaction_env:
            files.add((pack_dir / spec.redaction_env).resolve())
        if spec.connector == "file_drop" and spec.glob:
            try:
                files |= {p.resolve() for p in (pack_dir / (spec.root or ".")).glob(spec.glob) if p.is_file()}
            except (ValueError, NotImplementedError):
                pass
            if spec.root:
                dirs.append((pack_dir / spec.root).resolve())
        for p in _paths_in(ingest_value(spec.init, pack_dir, init=True)) + _paths_in(ingest_value(spec.scope, pack_dir)):
            rp = Path(p).resolve()
            if rp.is_dir():
                dirs.append(rp)
            else:
                files.add(rp)
    return files, dirs


def _skip(rel: Path) -> bool:
    """Dotfiles and dot-dirs, caches and secret-shaped file names (V3)."""
    for part in rel.parts:
        if part.startswith(".") or any(fnmatch.fnmatch(part, pat) for pat in CACHE):
            return True
    return any(fnmatch.fnmatch(rel.name.lower(), pat) for pat in SECRET_FILES)


def _scan_secrets(root: Path, values: list[str] = ()) -> list[str]:
    """Credential-shaped text in the bundle (V3 / G2 D1): a hit refuses the export. Data files are scanned with every M3 /
    M10 secret pattern (provider keys, private keys, URL userinfo, query and header credentials); Python sources, which
    ship in the pack's wheel anyway and carry synthetic redaction canaries, with the provider-key formats only. Any exact
    value of a dotenv the pack names as ``redaction_env`` is a hit anywhere."""
    from crucible.core.secrets import _SECRETS, SECRET_PATTERNS

    provider = [re.compile(p) for p in SECRET_PATTERNS[:-4]]  # the last four are header / connection-string shapes
    hits = []
    for f in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        try:
            text = f.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        if any(v and v in text for v in values) or any(rx.search(text) for rx in (provider if f.suffix == ".py" else _SECRETS)):
            hits.append(str(f.relative_to(root)))
    return hits


def _env_values(pack_dir: Path, m: Any) -> list[str]:
    """The credential values of every ``redaction_env`` inside the pack (never copied; their values must not appear in a
    copied file either). A dotenv outside the pack is never read here: it is never bundled."""
    from crucible.connectors.redact import RedactionPolicy

    vals: list[str] = []
    for spec in m.ingest:
        if spec.redaction_env:
            f = (pack_dir / spec.redaction_env).resolve()
            if _under(f, pack_dir) and f.is_file():
                vals += RedactionPolicy.with_env_file(f, missing_ok=True).secret_values
    return vals


def _bundle(pack_dir: Path, m: Any, out: Path) -> Path:
    """An allowlisted copy of the pack next to the launchers (V3), built aside and swapped in (V2)."""
    pack_dir = Path(pack_dir).resolve()
    dest = out / "pack"
    if dest.is_symlink() or (dest.exists() and not (dest / BUNDLE_MARKER).is_file()):
        raise ValueError(f"{dest} exists and is not a bundle crucible export wrote (no {BUNDLE_MARKER}): it is never replaced; choose another --out")
    raw_files, raw_dirs = _excluded(pack_dir, m)
    tmp = Path(tempfile.mkdtemp(prefix=".pack-", dir=out))
    copied, left_out = [], []
    try:
        for name in BUNDLE_FILES:
            if (pack_dir / name).is_file() and not (pack_dir / name).is_symlink():
                shutil.copy2(pack_dir / name, tmp / name)
                copied.append(name)
        if not (tmp / "pack.yaml").is_file():
            raise ValueError(f"{pack_dir} has no pack.yaml to bundle")
        for top in [pack_dir / "src" / m.package] + [pack_dir / n for n in BUNDLE_DIRS]:
            if not top.is_dir() or top.is_symlink():
                continue
            for p in sorted(top.rglob("*")):
                if p.is_dir() and not p.is_symlink():
                    continue
                rel = p.relative_to(pack_dir)
                rp = p.resolve()
                if any(fnmatch.fnmatch(part, pat) for part in rel.parts for pat in CACHE):
                    continue
                if p.is_symlink() or not _under(rp, pack_dir) or _skip(rel) or rp in raw_files or any(_under(rp, x) for x in raw_dirs):
                    left_out.append(str(rel))
                    continue
                (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, tmp / rel)
                copied.append(str(rel))
        hits = _scan_secrets(tmp, _env_values(pack_dir, m))
        if hits:
            raise ValueError(f"the export bundle of {m.name} would ship credential-shaped text in {hits[:5]}; remove it from the pack (nothing was written)")
        (tmp / BUNDLE_MARKER).write_text(json.dumps({"pack": m.name, "version": m.version, "files": len(copied), "left_out": sorted(left_out)}, indent=1) + "\n")
        old = None
        if dest.exists():
            old = Path(tempfile.mkdtemp(prefix=".pack-old-", dir=out))
            os.replace(dest, old / "pack")
        os.replace(tmp, dest)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


async def _canonical(tasks: list, out: Path) -> dict[str, Any]:
    """Every task's gold replayed; a task whose replay fails is reported and the rest go on (V20); every task is cleaned up."""
    from crucible.core.schema.actions import Message
    from crucible.interop.replay import Action, replay

    tasks = _pairs(tasks)
    rows, skipped, errors = {}, {}, {}
    try:
        dirs = task_dirs([getattr(t, "id", "task") for t, _ in tasks], out)
        for t, gold in tasks:
            tid = getattr(t, "id", "task")
            if not (hasattr(t, "subgoals") or hasattr(t, "plan_for_progress")):
                skipped[tid] = "no subgoals or plan: the dense channel cannot be measured"
                continue
            d = dirs[tid]
            try:
                acts = [Action(None, {}, to=a.to, text=a.text) if isinstance(a, Message) else Action(a.function_name, dict(a.arguments)) for a in gold or []]
                r = await replay(t, acts, d, episode_key=f"{tid}|gold")
                rows[tid] = {"reward": r["reward"], "steps": r["steps"], "dir": str(d)}
            except Exception as e:  # noqa: BLE001 - reported per task
                errors[tid] = f"{type(e).__name__}: {e}"[:300]
    finally:
        _cleanup(tasks)
    res = {"target": "canonical", "tasks": rows, "skipped": skipped, "errors": errors, "ok": bool(rows) and not errors}
    if not rows:  # G2 D7: nothing replayed is a failure, not an empty success
        res["reason"] = f"no task was replayed ({len(skipped)} skipped, {len(errors)} failed, {len(tasks)} in the pack)"
    return res


def _atif_result(traj: Path, run_doc: dict[str, Any]) -> dict[str, Any]:
    """The attribution an ATIF trajectory lacks (G2 D11; Harbor keeps ``result.json`` beside a trial's trajectory): task,
    policy, harness, sample, reward, outcome, mask and the run's identity, from the attempt's ``record.json``."""
    try:
        rec = json.loads((traj.parent / "record.json").read_text())
    except (OSError, ValueError):
        rec = {}
    smp = rec.get("sample") or {}
    fail = rec.get("failure") or {}
    return {"trajectory": traj.name, "run_id": rec.get("run_id") or run_doc.get("run_id"), "run_identity": run_doc.get("identity"),
            "attempt": rec.get("attempt"), "task": rec.get("task") or smp.get("task"), "policy": rec.get("policy") or smp.get("policy"),
            "harness": smp.get("harness"), "sample_idx": smp.get("sample_idx"), "seed": rec.get("seed"), "reward": rec.get("reward"),
            "outcome": rec.get("outcome"), "masked": bool(rec.get("mask_sample")), "failure": fail.get("kind") if isinstance(fail, dict) else fail,
            "steps": rec.get("steps"), "cost_usd": rec.get("cost_usd")}


def _atif(run_dir: Path, out: Path, *, validate: bool = True) -> dict[str, Any]:
    files = sorted(run_dir.glob("samples/*/a*/trajectory.*.json"))
    if not files:  # V12: the build root instead of the run wrote 0 trajectories and exited 0
        raise ValueError(f"no samples/*/a*/trajectory.*.json under {run_dir}: give an M10 run directory (e.g. <build>/runs/<run id>)")
    try:
        from harbor.utils.trajectory_validator import TrajectoryValidator
    except ImportError:
        TrajectoryValidator = None
    if TrajectoryValidator is None and validate:
        raise ValueError("Harbor's trajectory validator is not installed here: install harbor, or pass --no-validate to copy unvalidated files")
    out.mkdir(parents=True, exist_ok=True)
    try:
        run_doc = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError):
        run_doc = {}
    done, invalid = [], {}
    for f in files:
        name = f"{f.parent.parent.name}-{f.parent.name}-{f.stem.split('.', 1)[1]}.json"
        if TrajectoryValidator is not None and validate:
            try:
                v = TrajectoryValidator()
                if not v.validate(json.loads(f.read_text()), validate_images=False):
                    invalid[name] = "; ".join(v.get_errors()[:3])[:300]
                    continue
            except Exception as e:  # noqa: BLE001 - reported per file
                invalid[name] = f"{type(e).__name__}: {e}"[:200]
                continue
        shutil.copy(f, out / name)
        (out / name.replace(".json", ".result.json")).write_text(json.dumps(_atif_result(f, run_doc), indent=1, sort_keys=True, default=str) + "\n")
        done.append(name)
    return {"target": "atif", "trajectories": len(done), "invalid": invalid, "validated": TrajectoryValidator is not None and validate, "dir": str(out),
            "ok": not invalid}
