"""The acceptance key (research m9-m10/01 §15.7): acceptance binds to *everything* that decides a verdict, not to
Harbor's packager hash alone (which excludes ``crucible/``).

key = Harbor hash ⊕ overlay hash (the task's own inputs: world, checks, negatives, properties, minus outputs)
      ⊕ image digests ⊕ kit hash (every module that decides a verdict: ``accept``, ``verify``, the forge's learnability,
      ladder and hacker, ``world/leaks``; + the task's plan module) ⊕ contract, policy and
      registry versions.

Adapters provide ``key_inputs() -> {"harbor_hash"?, "overlay_hash", "images"?: {role: ref}, "plan_files"?: [paths]}``;
image refs are resolved to digests by ``resolve_image`` (a tag that cannot be resolved stays as ``unresolved:<tag>``,
which A9 fails).
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path

from crucible.accept.model import AcceptanceKey
from crucible.accept.policy import Policy

KIT_ROOT = Path(__file__).resolve().parent.parent / "verify" / "runtime"
PKG = Path(__file__).resolve().parent.parent  # src/crucible
# Every module that decides a verdict (M9 audit V5: only ``verify/runtime`` was hashed, so a changed check, calibration,
# leak audit, mutation operator or hacker reused cached verdicts, and the manual ``@register`` versions were not bumped):
DECIDING = ("accept", "verify", "forge/learnability.py", "forge/ladder.py", "forge/hacker.py", "forge/specialists.py", "world/leaks.py")


def sha256_files(paths: list[Path], root: Path | None = None) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(root) if root else p.name).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:24]


def kit_hash(plan_files: list[Path] = ()) -> str:
    files: list[Path] = []
    for rel in DECIDING:
        p = PKG / rel
        files += [p] if p.is_file() else [x for x in p.rglob("*.py") if "__pycache__" not in x.parts]
    return sha256_files(files, PKG.parent) + (":" + sha256_files([Path(p) for p in plan_files]) if plan_files else "")


def docker_digest(ref: str, docker_host: str | None = None) -> str:
    import os

    env = {**os.environ, "DOCKER_HOST": docker_host} if docker_host else None
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", ref], capture_output=True, text=True, env=env, check=False)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else f"unresolved:{ref}"


def compute(task, policy: Policy, registry_version: str, *, resolve_image: Callable[[str], str] = docker_digest) -> AcceptanceKey:
    ki = task.key_inputs() if hasattr(task, "key_inputs") else {"overlay_hash": task.task_hash}
    images = {role: (ref if ref.startswith("sha256:") else resolve_image(ref)) for role, ref in (ki.get("images") or {}).items()}
    return AcceptanceKey(task_id=task.task_id, harbor_hash=ki.get("harbor_hash") or "", overlay_hash=ki["overlay_hash"], images=images,
                         kit_hash=kit_hash([Path(p) for p in ki.get("plan_files") or []]), policy_version=policy.version(),
                         registry_version=registry_version)
