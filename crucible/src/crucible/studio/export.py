"""``crucible export``: a pack's tasks or a run's episodes in the formats other runtimes and trainers read (M11 seams).

| target | what is written | needs |
|---|---|---|
| ``canonical`` | per task: the gold actions replayed into ``reward.json`` + ``steps.jsonl``. Every runtime adapter must reproduce these (the M11 consistency gate) | ``hooks.tasks``; tasks with subgoals or a plan |
| ``atif`` | per episode of an M10 run: its ATIF trajectory (Harbor's schema), validated by Harbor's validator when installed | ``--run`` |
| ``nemo-gym`` | a launcher for ``crucible.interop.nemo_gym.server`` over every task of the pack; the app is built here to prove the tasks load | ``hooks.tasks`` |
| ``openenv`` | a launcher module exposing ``make_environment`` over every task (``reset(task_index=)``) and the generator (``reset(item_seed=)``), plus ``openenv.yaml``; the environment class is built when OpenEnv is importable | ``hooks.tasks``; ``openenv`` (its own environment) |

Both pack exports are self-contained (M14 sweep G2 D11): the pack is copied to ``out/pack`` and the launchers find it
relative to themselves, so the directory can move to another machine (it still needs ``crucible`` installed there).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

TARGETS = ("canonical", "atif", "nemo-gym", "openenv")


async def export(target: str, *, out: Path, pack: str | Path | None = None, run: Path | None = None) -> dict[str, Any]:
    from crucible.studio.pack import find_pack, load_manifest, resolve

    if target not in TARGETS:
        raise ValueError("target: " + " | ".join(TARGETS))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if target == "atif":
        if run is None:
            raise ValueError("export atif needs --run <run dir>")
        return _atif(Path(run), out)
    if pack is None:
        raise ValueError(f"export {target} needs --pack")
    d = find_pack(pack)
    m = load_manifest(d)
    if not m.hooks.tasks:
        raise ValueError(f"pack {m.name} has no hooks.tasks to export")
    ref = m.hooks.tasks
    if target == "canonical":
        return await _canonical(resolve(d, ref)(), out)
    bundle = _bundle(d, m, out)  # self-contained (G2 D11): the pack is copied, the launchers use relative paths only
    head = (f"import sys\nfrom pathlib import Path\n\nPACK = Path(__file__).resolve().parent / {bundle.name!r}\n"
            "sys.path.insert(0, str(PACK / 'src'))\nfrom crucible.studio.pack import resolve  # noqa: E402\n\n")
    factory_src = ("def make_task(task_index: int = 0, item_seed: int | None = None):\n"
                   "    \"\"\"Any of the pack's tasks (``task_index``) or, with ``item_seed``, a fresh generated item.\"\"\"\n"
                   + (f"    if item_seed is not None:\n        return resolve(PACK, {m.hooks.generator!r})(int(item_seed))\n" if m.hooks.generator else
                      "    if item_seed is not None:\n        raise ValueError('this pack has no generator')\n")
                   + f"    return resolve(PACK, {ref!r})()[int(task_index)][0]\n\n\n"
                   f"def tasks():\n    return {{getattr(t, 'id', str(i)): t for i, (t, _) in enumerate(resolve(PACK, {ref!r})())}}\n")
    meta = {"name": m.name, "version": m.version, "license": m.license, "reward_range": list(m.reward.range),
            "higher_is_better": m.reward.higher_is_better, "determinism": m.determinism, "requires": list(m.requires),
            "tasks": None, "generator": bool(m.hooks.generator)}
    if target == "nemo-gym":
        from crucible.interop.nemo_gym.server import make_app

        tasks = {getattr(t, "id", str(i)): t for i, (t, _) in enumerate(resolve(d, ref)())}
        make_app(tasks)  # proves the tasks load into the server
        meta["tasks"] = sorted(tasks)
        factory = out / "nemo_tasks.py"
        factory.write_text(f'"""NeMo-Gym resources server factory for the {m.name} pack (crucible export)."""\n' + head + factory_src)
        cmd = "PYTHONPATH=. python -m crucible.interop.nemo_gym.server --tasks nemo_tasks:tasks --port 8000"
        (out / "serve_nemo.sh").write_text('#!/bin/sh\ncd "$(dirname "$0")"\n' + cmd + "\n")
        (out / "crucible_export.json").write_text(json.dumps(meta | {"target": target}, indent=1) + "\n")
        return {"target": target, "tasks": sorted(tasks), "files": [str(factory), str(out / "serve_nemo.sh"), str(out / "crucible_export.json")],
                "bundle": str(bundle), "run": cmd}
    # openenv: every task (reset(task_index=...)) and the generator (reset(item_seed=...)), not task 0 only
    meta["tasks"] = [getattr(t, "id", str(i)) for i, (t, _) in enumerate(resolve(d, ref)())]
    mod = out / "openenv_env.py"
    mod.write_text(f'"""OpenEnv environment for the {m.name} pack (crucible export)."""\n' + head + factory_src
                   + "\n\nfrom crucible.interop.openenv.export import make_environment  # noqa: E402\n\nEnvironment = make_environment(make_task)\n")
    import yaml

    (out / "openenv.yaml").write_text(yaml.safe_dump({"spec_version": 1, "name": m.name, "version": m.version, "type": "space", "runtime": "fastapi",
                                                      "app": "openenv_env:Environment", "description": m.description, "license": m.license,
                                                      "validation": {"reward_range": list(m.reward.range), "higher_is_better": m.reward.higher_is_better,
                                                                     "determinism": m.determinism, "requires": list(m.requires),
                                                                     "tasks": meta["tasks"], "generator": meta["generator"]}}, sort_keys=False))
    try:
        import openenv  # noqa: F401

        from crucible.interop.openenv.export import make_environment

        make_environment(lambda: resolve(d, ref)()[0][0])
        built = True
    except ImportError:
        built = False
    return {"target": target, "files": [str(mod), str(out / "openenv.yaml")], "bundle": str(bundle), "tasks": meta["tasks"], "built_here": built,
            "note": None if built else "openenv is not importable here: run the module in its environment (crucible/.envs/openenv)"}


def _bundle(pack_dir: Path, m: Any, out: Path) -> Path:
    """A copy of the pack next to the launchers (pack.yaml, README, pyproject, src/ and any data / worlds directories)."""
    dest = out / "pack"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in ("pack.yaml", "README.md", "pyproject.toml", "LICENSE"):
        if (pack_dir / name).is_file():
            shutil.copy2(pack_dir / name, dest / name)
    for name in ("src", "data", "worlds", "families"):
        if (pack_dir / name).is_dir():
            shutil.copytree(pack_dir / name, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), symlinks=True)
    return dest


async def _canonical(tasks: list, out: Path) -> dict[str, Any]:
    from crucible.core.schema.actions import Message
    from crucible.interop.replay import Action, replay

    rows, skipped = {}, {}
    for t, gold in tasks:
        tid = getattr(t, "id", "task")
        if not (hasattr(t, "subgoals") or hasattr(t, "plan_for_progress")):
            skipped[tid] = "no subgoals or plan: the dense channel cannot be measured"
            continue
        acts = [Action(None, {}, to=a.to, text=a.text) if isinstance(a, Message) else Action(a.function_name, dict(a.arguments)) for a in gold]
        d = out / tid.replace("/", "_").replace("@", "_")
        r = await replay(t, acts, d, episode_key=f"{tid}|gold")
        rows[tid] = {"reward": r["reward"], "steps": r["steps"], "dir": str(d)}
    for t, _ in tasks:
        if callable(getattr(t, "cleanup", None)):
            t.cleanup()
    return {"target": "canonical", "tasks": rows, "skipped": skipped}


def _atif(run_dir: Path, out: Path) -> dict[str, Any]:
    files = sorted(run_dir.glob("samples/*/a*/trajectory.*.json"))
    done, invalid = [], {}
    try:
        from harbor.utils.trajectory_validator import TrajectoryValidator
    except ImportError:
        TrajectoryValidator = None
    for f in files:
        name = f"{f.parent.parent.name}-{f.parent.name}-{f.stem.split('.', 1)[1]}.json"
        if TrajectoryValidator is not None:
            try:
                v = TrajectoryValidator()
                if not v.validate(json.loads(f.read_text()), validate_images=False):
                    invalid[name] = "; ".join(v.get_errors()[:3])[:300]
                    continue
            except Exception as e:  # noqa: BLE001 - reported per file
                invalid[name] = f"{type(e).__name__}: {e}"[:200]
                continue
        shutil.copy(f, out / name)
        done.append(name)
    return {"target": "atif", "trajectories": len(done), "invalid": invalid, "validated": TrajectoryValidator is not None, "dir": str(out)}
