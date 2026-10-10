"""Collecting the agent's work, and telling "collection failed" from "the agent changed nothing" (Appendix B.5 row 1;
research m9-m10/02 §3.3).

task02's first frontier run produced an empty patch because a legacy ``git add ... && git diff`` chain aborted on one
path. An empty patch is two very different things: if the tree changed (``git status --porcelain`` non-empty) or the hook
reported errors, collection failed (``crucible:empty_collection``, harness blame, masked, re-collect from the preserved
workspace); if nothing changed, the agent did nothing (scored 0).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from crucible.core.schema.run import Failure
from crucible.verify.collect import collect as tolerant_collect

LEGACY = "git add -A && git diff --cached --binary {base}"  # the chain task02's task.toml used


@dataclass
class Collected:
    patch: str
    mode: str
    tree_changed: bool
    errors: list[str] = field(default_factory=list)


def _git(ws: Path, *a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ws), *a], capture_output=True, text=True, check=False)


def tree_changed(ws: Path) -> bool:
    return bool(_git(ws, "status", "--porcelain", "--untracked-files=all").stdout.strip())


def collect(ws: Path, *, base: str = "HEAD", mode: str = "tolerant") -> Collected:
    ws = Path(ws)
    changed = tree_changed(ws)
    if mode == "legacy":
        r = subprocess.run(["bash", "-c", LEGACY.format(base=base)], cwd=ws, capture_output=True, text=True, check=False)
        _git(ws, "reset", "-q")
        return Collected(r.stdout if r.returncode == 0 else "", mode, changed, [r.stderr[-400:]] if r.returncode else [])
    c = tolerant_collect(ws, base=base)
    return Collected(c["patch"], mode, changed, c["errors"])


def classify(c: Collected) -> Failure | None:
    """None when the collection is trustworthy (a patch, or an honest "nothing changed")."""
    if c.patch.strip():
        return None
    if c.tree_changed or c.errors:
        return Failure(kind="crucible:empty_collection", stage="verification", reason=f"empty patch from a changed tree ({c.mode}): {'; '.join(c.errors)[:300]}",
                       retryable=True)
    return None
