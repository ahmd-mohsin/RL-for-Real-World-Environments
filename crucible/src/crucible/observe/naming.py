"""One naming scheme for every W&B run and Weave trace (swarm-pipeline/11 §2).

| what | W&B project | run name | group | job_type |
|---|---|---|---|---|
| task building (funnel, targeting, cards, audits, spend) | ``crucible-factory`` | ``factory-<lane>-<phase>-<id>`` | the lane | the phase |
| RL post-training | ``crucible-train`` | ``train-<model>-<run>`` | ``objective-<objective id>`` | ``rl`` |
| benchmark arms and comparisons | ``crucible-bench`` | ``bench-<purpose>-<model>-s<seed>-<id>`` | ``tasks-<manifest hash>`` | the purpose |

Weave traces live in the project of the run that produced them (a benchmark episode in ``crucible-bench``, a trained
episode in ``crucible-train``, an auditor's verdict beside the episode it judged), so a W&B project's Weave tab shows the
traces next to the runs. The entity is ``WANDB_ENTITY`` (unset: the key's default entity at sync time). Every run carries
tags ``key:value`` (``model:``, ``purpose:``, ``lang:``, ...), at most 64 characters each (W&B's limit).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

PROJECTS = {"factory": "crucible-factory", "train": "crucible-train", "bench": "crucible-bench"}
ENTITY_ENV = ("CRUCIBLE_WANDB_ENTITY", "WANDB_ENTITY")
TAG_MAX = 64
NAME_MAX = 128


def project(kind: str) -> str:
    """``factory`` | ``train`` | ``bench`` -> the W&B project; ``CRUCIBLE_WANDB_PROJECT_PREFIX`` renames all three (a test
    or a second team)."""
    if kind not in PROJECTS:
        raise ValueError(f"kind is one of {sorted(PROJECTS)}, got {kind!r}")
    prefix = os.environ.get("CRUCIBLE_WANDB_PROJECT_PREFIX")
    return f"{prefix}-{kind}" if prefix else PROJECTS[kind]


def entity() -> str | None:
    for k in ENTITY_ENV:
        if os.environ.get(k):
            return os.environ[k]
    return None


def slug(s: Any, n: int = 48) -> str:
    out = re.sub(r"[^a-z0-9.]+", "-", str(s).lower()).strip("-.")
    return out[:n].strip("-.") or "x"


def short_model(model: str | None) -> str:
    """``Qwen/Qwen3.6-35B-A3B:peft:262144`` -> ``qwen3.6-35b-a3b``; ``tinker:...``, ``student:...@mlx`` likewise."""
    if not model:
        return "none"
    m = str(model)
    for pre in ("tinker:", "student:", "openrouter:", "claude-code:"):
        m = m.removeprefix(pre)
    m = m.split(":", 1)[0].split("@", 1)[0]
    return slug(m.rsplit("/", 1)[-1], 40)


def digest(x: Any, n: int = 12) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:n]


def manifest_hash(task_ids: Iterable[str]) -> str:
    """The task list's identity: arms of one comparison share it (their W&B group)."""
    return digest(sorted(str(t) for t in task_ids))


def run_name(*parts: Any) -> str:
    return "-".join(slug(p, 40) for p in parts if p is not None and str(p) != "")[:NAME_MAX]


def bench_name(purpose: str, model: str, seed: int | None, ident: str) -> str:
    return run_name("bench", purpose, short_model(model), f"s{seed}" if seed is not None else None, ident[:8])


def train_name(model: str, run: str) -> str:
    return run_name("train", short_model(model), run)


def factory_name(lane: str, phase: str, ident: str | None = None) -> str:
    return run_name("factory", lane, phase, ident[:8] if ident else None)


def tags(**kv: Any) -> list[str]:
    """``tags(model="Qwen/...", purpose="control")`` -> ``["model:qwen3.6-35b-a3b", "purpose:control"]``; None skipped; a
    list value gives one tag per item."""
    out: list[str] = []
    for k, v in kv.items():
        for x in (v if isinstance(v, list | tuple | set) else [v]):
            if x is None or x == "":
                continue
            val = short_model(x) if k == "model" else str(x)
            t = f"{k}:{val}"[:TAG_MAX]
            if t not in out:
                out.append(t)
    return out


def call_url(project_name: str, call_id: str, ent: str | None = None, *, base: str | None = None) -> str:
    """The Weave page of one call. Without an entity (no key yet) the link is a ``weave:`` placeholder that
    ``scripts/observe.py links`` resolves after sync."""
    ent = ent or entity()
    base = (base or os.environ.get("WANDB_BASE_URL") or "https://wandb.ai").rstrip("/")
    base = base.replace("://api.", "://")  # WANDB_BASE_URL is the API host; the app host has no api. prefix
    if not ent:
        return f"weave:{project_name}/call/{call_id}"
    return f"{base}/{ent}/{project_name}/r/call/{call_id}"


def flat(d: Mapping[str, Any], prefix: str = "", sep: str = "/") -> dict[str, Any]:
    """A nested dict as ``a/b/c`` keys (lists left as values)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flat(v, key, sep))
        else:
            out[key] = v
    return out
