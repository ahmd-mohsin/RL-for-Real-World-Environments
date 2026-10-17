"""The run lock (M0 audit 2026-10-14 R-1): a resume refuses a run whose settings changed.

``RunSpec.identity_doc`` was an allowlist of 7 entries with tasks, policies and harnesses as display ids, so a resume at
another temperature, step limit or endpoint was accepted and pass@k pooled two experiments. The lock projects every
``RunSpec`` field (a field added later counts without anyone listing it) and every component's declared settings
(``lock_settings()``: a policy's model, endpoint, request defaults and sampling, a harness's limits), minus the
operational fields below, which change how a run executes, not what it measures. ``<run>/lock.json`` is written before
the first dispatch; on resume a different fingerprint raises ``RunLockMismatch`` naming up to 20 changed settings,
unless every one matches a glob in ``RunSpec.allow_change``, which records the change in the lock. A run directory from
before the lock is checked by its old identity once and then gets a lock. Task content enters with R-2 (digests).
"""

from __future__ import annotations

import dataclasses
import enum
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

from crucible.core.fingerprint import SCHEME, describe, diff, excluded, fingerprint, project
from crucible.core.prov import utc_now
from crucible.core.schema.run import LockChange, RunLaunch, RunLock

LOCK_FILE = "lock.json"
LAUNCH_DIR = "launches"
# Fields of RunSpec that a resume may change freely: parallelism, money and time budgets, retry pacing, the health
# stop rules' thresholds, and the allowance itself. Everything else counts.
OPERATIONAL = ("concurrency", "budget_usd", "wall_clock_s", "retry_backoff_s", "max_mask_fraction", "max_unknown_fraction",
               "min_terminal_for_health", "allow_change")
COMPONENTS = ("tasks", "policies", "harnesses")


class RunLockMismatch(ValueError):
    """A resume whose settings differ from the run's lock."""


def lock_doc(spec: Any) -> dict[str, Any]:
    """Every field of ``spec``; tasks, policies and harnesses described by their id and declared settings."""
    doc: dict[str, Any] = {}
    for f in dataclasses.fields(spec):
        value = getattr(spec, f.name)
        doc[f.name] = [component(x) for x in value] if f.name in COMPONENTS else jsonable(value)
    return doc


def component(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"id": str(getattr(obj, "id", obj))}
    hook = getattr(obj, "lock_settings", None)
    settings = hook() if callable(hook) else None
    if settings is not None:
        out["settings"] = jsonable(settings)
    if (d := task_digest(obj)) is not None:
        out["digest"] = d
    return out


def task_digest(task: Any) -> str | None:
    """The content digest a task declares (``digest()``, R-2), or None for a task that declares none."""
    hook = getattr(task, "digest", None)
    return hook() if callable(hook) else None


def definition_digest(obj: Any) -> str:
    """``sha256:<hex>`` of an in-memory task's definition: its data in canonical JSON (``jsonable``)."""
    import hashlib

    from crucible.core.canonical import dumps

    return "sha256:" + hashlib.sha256(dumps(jsonable(obj)).encode()).hexdigest()


def jsonable(v: Any) -> Any:
    """A JSON value for ``v``; an object with no data form becomes ``<TypeName>`` (never a repr with an address)."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else repr(v)
    if isinstance(v, enum.Enum):
        return jsonable(v.value)
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        items = [jsonable(x) for x in v]
        return sorted(items, key=repr) if isinstance(v, (set, frozenset)) else items
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: jsonable(getattr(v, f.name)) for f in dataclasses.fields(v)}
    dump = getattr(v, "model_dump", None)
    if callable(dump):
        return jsonable(dump(mode="json"))
    return f"<{type(v).__qualname__}>"


def build(spec: Any) -> RunLock:
    doc = lock_doc(spec)
    opaque = [f"{kind}.{i} ({c['id']})" for kind in ("policies", "harnesses") for i, c in enumerate(doc[kind]) if "settings" not in c]
    opaque += [f"tasks.{i} ({c['id']}) content" for i, c in enumerate(doc["tasks"]) if "digest" not in c]
    settings = project(doc, OPERATIONAL)
    return RunLock(run_id=spec.run_id, scheme=SCHEME, fingerprint=fingerprint(settings), excluded=list(OPERATIONAL),
                   settings=settings, opaque=opaque, created_at=utc_now())


def ensure(run_dir: Path, spec: Any) -> tuple[RunLock, str]:
    """The run's lock and what happened: ``created`` (new run), ``adopted`` (a run from before the lock), ``same``, or
    ``changed`` (every change allowed by ``spec.allow_change``, recorded). Raises RunLockMismatch otherwise."""
    from crucible.runtime.journal import atomic_write

    path = Path(run_dir) / LOCK_FILE
    new = build(spec)
    if not path.exists():
        atomic_write(path, new.model_dump_json(indent=1))
        return new, "adopted" if (Path(run_dir) / "run.json").exists() and _has_attempts(run_dir) else "created"
    old = RunLock.load(path)
    # settings the current operational list excludes do not count, even if an older lock recorded them
    before = {k: v for k, v in old.settings.items() if not excluded(tuple(k.split(".")), OPERATIONAL)}
    changes = diff(before, new.settings)
    if not changes:
        return old, "same"
    allow = tuple(getattr(spec, "allow_change", ()) or ())
    refused = {k: c for k, c in changes.items() if not excluded(tuple(k.split(".")), allow)}
    if refused:
        raise RunLockMismatch(
            f"run {spec.run_id} was started with other settings; refusing to resume it as the same run. Changed:\n{describe(refused)}\n"
            f"Start a new run_id, or resume with RunSpec(allow_change=[...]) naming these settings (globs such as "
            f"'policies.*.settings.temperature') to record the change in {path}.")
    lock = old.model_copy(update={"settings": new.settings, "fingerprint": new.fingerprint, "opaque": new.opaque, "excluded": new.excluded,
                                  "changes": [*old.changes, LockChange(at=utc_now(), changed=sorted(changes), before=old.fingerprint, after=new.fingerprint)]})
    atomic_write(path, lock.model_dump_json(indent=1))
    return lock, "changed"


def _has_attempts(run_dir: Path) -> bool:
    return (Path(run_dir) / "samples").is_dir()


def launch(run_dir: Path, spec: Any, lock: RunLock, status: str) -> RunLaunch:
    """Record this invocation in ``<run>/launches/<n>.json`` (R-1 step 2): n counts from 1, and each file is created
    exclusively, so two launches never share a number. Secrets are scrubbed from argv and the configuration."""
    from crucible.core.build import build_info
    from crucible.core.secrets import scrub_obj

    try:
        from crucible.core.config import get_config

        config = get_config().redacted()
    except Exception as e:  # noqa: BLE001 - an unreadable configuration is recorded, never a reason not to run
        config = {"error": f"{type(e).__name__}: {e}"[:300]}
    folder = Path(run_dir) / LAUNCH_DIR
    folder.mkdir(parents=True, exist_ok=True)
    n = 1 + max((int(p.stem) for p in folder.glob("*.json") if p.stem.isdigit()), default=0)
    while True:
        rec = RunLaunch(run_id=spec.run_id, n=n, started_at=utc_now(), lock=lock.fingerprint, lock_status=status,
                        argv=scrub_obj(list(sys.argv)), operational={k: jsonable(getattr(spec, k, None)) for k in OPERATIONAL},
                        config=scrub_obj(jsonable(config)), build=build_info(), python=platform.python_version())
        try:
            fd = os.open(folder / f"{n}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            n += 1
            continue
        with os.fdopen(fd, "w") as f:
            f.write(rec.model_dump_json(indent=1))
            f.flush()
            os.fsync(f.fileno())
        return rec
