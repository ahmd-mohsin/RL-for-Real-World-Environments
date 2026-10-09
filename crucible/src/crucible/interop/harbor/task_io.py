"""Harbor task directories <-> Crucible (DEPEND on harbor).

- ``load_harbor_task`` validates ``task.toml`` with Harbor's own ``TaskConfig`` and ``Task``
  (``harbor/models/task``), computes Harbor's publish content hash (``harbor/publisher/packager.py``)
  and reports keys Harbor would silently drop (its models use pydantic's default extra="ignore").
- ``load_crucible_task`` reads the ``crucible/`` extension (task, world, card, milestones) next to,
  or as an overlay of, a Harbor directory.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.models.task.config import TaskConfig
from harbor.models.task.task import Task
from harbor.publisher.packager import Packager
from pydantic import BaseModel

from crucible.core.schema.milestone import MilestoneSet
from crucible.core.schema.task import CrucibleTask, HarborTaskSummary, TaskCard
from crucible.core.schema.world import WorldSpec


def _ignored_keys(raw: dict[str, Any], model: type[BaseModel], prefix: str = "") -> list[str]:
    """Keys present in the TOML that the Harbor model does not declare (would be dropped)."""
    out: list[str] = []
    fields = model.model_fields
    aliases = {f.alias for f in fields.values() if f.alias}
    for key, value in raw.items():
        if key not in fields and key not in aliases:
            out.append(prefix + key)
            continue
        field = fields.get(key)
        if field is None:
            continue
        sub = _model_of(field.annotation)
        if sub is not None and isinstance(value, dict) and sub is not dict:
            if key == "metadata":
                continue
            out.extend(_ignored_keys(value, sub, f"{prefix}{key}."))
        elif sub is not None and isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    out.extend(_ignored_keys(item, sub, f"{prefix}{key}[{i}]."))
    return out


def _model_of(annotation: Any) -> type[BaseModel] | None:
    import typing

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in typing.get_args(annotation):
        m = _model_of(arg)
        if m is not None:
            return m
    return None


def _val(x: Any) -> str:
    return str(getattr(x, "value", x))


@dataclass
class LoadedHarborTask:
    path: Path
    task: Task
    config: TaskConfig
    summary: HarborTaskSummary


def load_harbor_task(task_dir: str | Path, *, require_tests: bool = True) -> LoadedHarborTask:
    task_dir = Path(task_dir).resolve()
    raw_text = (task_dir / "task.toml").read_text()
    raw = tomllib.loads(raw_text)
    import warnings

    with warnings.catch_warnings(record=True) as caught:  # Harbor's validation warnings are task-quality signals: keep them
        warnings.simplefilter("always")
        config = TaskConfig.model_validate_toml(raw_text)
        task = Task(task_dir, disable_verification=not require_tests)
    content_hash, _files = Packager.compute_content_hash(task_dir)
    env = config.environment
    ver = config.verifier
    mode = "separate" if (ver.environment_mode == "separate" or ver.environment is not None) else "shared"
    summary = HarborTaskSummary(
        name=task.name,
        schema_version=config.schema_version,
        task_hash=content_hash,
        gpus=env.gpus or 0,
        cpus=float(env.cpus) if env.cpus is not None else None,
        memory_mb=env.memory_mb,
        docker_image=env.docker_image,
        dockerfile=raw.get("environment", {}).get("dockerfile"),
        verifier_mode=mode,
        network_mode=_val(config.agent.network_mode or env.network_mode),
        baseline_network_mode=_val(env.network_mode),
        agent_timeout_sec=config.agent.timeout_sec,
        verifier_timeout_sec=ver.timeout_sec,
        collect_commands=[c.command for c in ver.collect],
        artifacts=[a if isinstance(a, str) else a.source for a in config.artifacts],
        ignored_keys=_ignored_keys(raw, TaskConfig),
        metadata={**dict(config.metadata), "harbor_warnings": sorted({str(w.message)[:300] for w in caught})},
    )
    return LoadedHarborTask(path=task_dir, task=task, config=config, summary=summary)


@dataclass
class LoadedCrucibleTask:
    overlay: Path
    harbor: LoadedHarborTask
    task: CrucibleTask
    world: WorldSpec
    card: TaskCard
    milestones: MilestoneSet | None


def load_crucible_task(path: str | Path) -> LoadedCrucibleTask:
    """``path`` is a dir containing ``crucible/task.yaml``; the Harbor dir is ``task.harbor_dir`` or ``path``."""
    path = Path(path).resolve()
    ct = CrucibleTask.load(path / "crucible" / "task.yaml")
    harbor_dir = (path / ct.harbor_dir).resolve() if ct.harbor_dir else path
    harbor = load_harbor_task(harbor_dir)
    world = WorldSpec.load(path / ct.world)
    card = TaskCard.load(path / ct.card)
    ms = MilestoneSet.load(path / ct.milestones) if ct.milestones and (path / ct.milestones).exists() else None
    if card.id != ct.id:
        raise ValueError(f"card id {card.id!r} != task id {ct.id!r}")
    return LoadedCrucibleTask(overlay=path, harbor=harbor, task=ct, world=world, card=card, milestones=ms)
