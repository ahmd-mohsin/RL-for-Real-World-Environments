"""Collecting the agent's work, and telling "collection failed" from "the agent changed nothing" (Appendix B.5 row 1;
research m9-m10/02 §3.3).

task02's first frontier run produced an empty patch because a legacy ``git add ... && git diff`` chain aborted on one
path. An empty patch is two very different things: if the tree changed (``git status --porcelain`` non-empty) or the hook
reported errors, collection failed (``crucible:empty_collection``, harness blame, masked, re-collect from the preserved
workspace); if nothing changed, the agent did nothing (scored 0).
"""

from __future__ import annotations

import re
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
    changed: list[str] = field(default_factory=list)  # the collectable paths the agent changed (``git status``, derived files dropped)


def _git(ws: Path, *a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ws), *a], capture_output=True, text=True, check=False)


def changed_paths(ws: Path) -> list[str]:
    from crucible.verify.collect import is_derived

    lines = _git(ws, "status", "--porcelain", "--untracked-files=all").stdout.splitlines()
    return [p for ln in lines if ln.strip() and not is_derived(p := ln[3:].split(" -> ")[-1].strip('"'))]


def tree_changed(ws: Path) -> bool:
    """Whether the agent changed anything the hook collects: derived files (``__pycache__``, build trees) do not count
    (M7 audit: a do-nothing agent that ran Python had an empty patch and a "changed" tree, so it was masked, not scored 0)."""
    return bool(changed_paths(ws))


def collect(ws: Path, *, base: str = "HEAD", mode: str = "tolerant") -> Collected:
    ws = Path(ws)
    paths = changed_paths(ws)
    if mode == "legacy":
        r = subprocess.run(["bash", "-c", LEGACY.format(base=base)], cwd=ws, capture_output=True, text=True, check=False)
        _git(ws, "reset", "-q")
        return Collected(r.stdout if r.returncode == 0 else "", mode, bool(paths), [r.stderr[-400:]] if r.returncode else [], paths)
    c = tolerant_collect(ws, base=base)
    return Collected(c["patch"], mode, bool(paths), c["errors"], paths)


# per-path errors git reports for a file the agent made unreadable (tolerant hook): the agent's work product
PATH_ERROR = re.compile(r"cannot add (\S+)|open\(\"([^\"]+)\"\): Permission denied|unable to index file '([^']+)'")


def classify(c: Collected) -> Failure | None:
    """None when the collection is trustworthy (a patch, or an honest "nothing changed"). In tolerant mode a per-path error
    ("cannot add <path>": a file the agent made unreadable) is the agent's own work product, not a lost collection: the
    rest is graded (M10 audit V1: a ``chmod 000`` turned a do-nothing policy's 0 into a mask). The legacy hook aborting
    on one path is the harness's failure (Appendix B.5 row 1), re-collected tolerantly."""
    if c.patch.strip():
        return None
    mine: set[str] = set()  # paths the agent made uncollectable
    errors = []  # anything else: the hook's own failure
    for e in c.errors:
        paths = {next(x for x in m if x) for m in PATH_ERROR.findall(e)} if c.mode == "tolerant" else set()
        if paths:
            mine |= paths
        else:
            errors.append(e)
    lost = [p for p in c.changed if p not in mine] if c.changed else ([] if c.mode == "tolerant" and mine else (["?"] if c.tree_changed else []))
    if lost or errors:
        return Failure(kind="crucible:empty_collection", stage="collect", reason=f"empty patch from a changed tree ({c.mode}): {'; '.join(errors)[:300]}",
                       retryable=True)
    return None


def repo_broken(ws: Path, base: str | None) -> list[str]:
    """Problems with the agent's repository that make its work uncollectable by its own doing (M10 audit V1: a corrupt
    ``.git/HEAD`` or index, or a deleted base commit, was masked as a harness failure)."""
    ws = Path(ws)
    if not (ws / ".git").exists():
        return ["the workspace repository (.git) was deleted"]
    out = []
    st = _git(ws, "status", "--porcelain")
    if st.returncode != 0:
        out.append(f"git status fails in the workspace: {st.stderr.strip()[:200]}")
    if base and _git(ws, "cat-file", "-e", f"{base}^{{commit}}").returncode != 0:
        out.append(f"the base commit {base[:12]} is gone from the workspace repository")
    return out
