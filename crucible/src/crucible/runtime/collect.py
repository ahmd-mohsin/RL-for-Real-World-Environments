"""Collecting the agent's work, and telling "collection failed" from "the agent changed nothing" (Appendix B.5 row 1;
research m9-m10/02 §3.3).

task02's first frontier run produced an empty patch because a legacy ``git add ... && git diff`` chain aborted on one
path. An empty patch is two very different things: if the tree changed (``git status --porcelain`` non-empty) or the hook
reported errors, collection failed (``crucible:empty_collection``, harness blame, masked, re-collect from the preserved
workspace); if nothing changed, the agent did nothing (scored 0).

M10 lab audit R-5 item 4: every git call here runs through ``world.exec._run_subprocess`` (off the event loop, bounded by
``verify.collect.GIT_TIMEOUT_S``, killed with its session on a timeout or a cancel), so ``collect``, ``changed_paths``,
``tree_changed`` and ``repo_broken`` are coroutines. A host stall is never the agent's outcome: a timed-out ``git status``
counts as "changed" (``['?']``), a timed-out collection is an empty patch with an error (``crucible:empty_collection``, masked,
re-collected), and ``repo_broken`` reports only a git that answered with a failure, never one that did not answer.

M10 lab audit R-5 review round 1: the repository is the agent's, so every call here runs with ``verify.collect.GIT_FLAGS`` (no
fsmonitor command, no hooks), and ``Collected.stalled`` says a git call did not answer: the first stall of a workspace is the
host's (masked, re-collected), the same preserved workspace stalling again is the policy's (``WorkspaceHarness``:
``crucible:agent_broke_env``), because a policy that plants a filter driver in its own ``.git/config`` would otherwise turn its 0
into a masked ``empty_collection`` on both attempts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from crucible.core.schema.run import Failure
from crucible.verify import collect as vcollect
from crucible.verify.collect import collect as tolerant_collect
from crucible.verify.patchio import decode
from crucible.world.exec import ExecOut, _run_subprocess, _run_subprocess_raw

_G = "git " + " ".join(vcollect.GIT_FLAGS)  # M10 lab audit R-5 review round 1: the same flags on the legacy chain
LEGACY = _G + " add -A && " + _G + " diff --cached --binary --no-textconv --no-ext-diff {base}"  # the chain task02's task.toml used


@dataclass
class Collected:
    patch: str
    mode: str
    tree_changed: bool
    errors: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)  # the collectable paths the agent changed (``git status``, derived files dropped)
    stalled: bool = False  # M10 lab audit R-5 review round 1: a git call of this collection ran past its timeout (it did not answer)


async def _git(ws: Path, *a: str) -> ExecOut:
    """One git call off the event loop, bounded by ``verify.collect.GIT_TIMEOUT_S`` (read at call time) and killed with its
    session past it (M10 lab audit R-5 item 4): ``out.timed_out`` says it did not answer. ``GIT_FLAGS``: the repository is the
    agent's (see the module doc)."""
    return await _run_subprocess(["git", "-C", str(ws), *vcollect.GIT_FLAGS, *a], cwd=None, env=None, timeout_s=vcollect.GIT_TIMEOUT_S, stdin=None)


async def _status(ws: Path) -> tuple[list[str], bool]:
    """The collectable changed paths and whether ``git status`` timed out (then ``['?']``: unknown counts as changed, so a
    stalled git is never read as "the agent changed nothing", which scores 0)."""
    out = await _git(ws, "status", "--porcelain", "--untracked-files=all")
    if out.timed_out:
        return ["?"], True
    lines = out.stdout.splitlines()
    return [p for ln in lines if ln.strip() and not vcollect.is_derived(p := ln[3:].split(" -> ")[-1].strip('"'))], False


async def changed_paths(ws: Path) -> list[str]:
    return (await _status(ws))[0]


async def tree_changed(ws: Path) -> bool:
    """Whether the agent changed anything the hook collects: derived files (``__pycache__``, build trees) do not count
    (M7 audit: a do-nothing agent that ran Python had an empty patch and a "changed" tree, so it was masked, not scored 0)."""
    return bool(await changed_paths(ws))


async def collect(ws: Path, *, base: str = "HEAD", mode: str = "tolerant") -> Collected:
    ws = Path(ws)
    paths, stalled = await _status(ws)
    status_err = [f"git status timed out after {vcollect.GIT_TIMEOUT_S:g}s"] if stalled else []
    if mode == "legacy":
        rc, raw, err, timed_out, dur = await _run_subprocess_raw(["bash", "-c", LEGACY.format(base=base)], cwd=str(ws), env=None,
                                                                 timeout_s=vcollect.GIT_TIMEOUT_S, stdin=None)
        # M7 R3: the patch is the hook's stdout bytes, decoded losslessly (CRLF, Latin-1 survive to the grader)
        out = ExecOut(rc=rc, stdout=decode(raw), stderr=err.decode(errors="replace"), duration_s=dur, timed_out=timed_out)
        if out.timed_out:  # the killed chain's lock would fail the tolerant re-collect of the preserved workspace
            try:
                (ws / ".git" / "index.lock").unlink(missing_ok=True)
            except OSError:
                pass
        await _git(ws, "reset", "-q")
        errors = [f"the legacy collect hook timed out after {vcollect.GIT_TIMEOUT_S:g}s"] if out.timed_out else ([out.stderr[-400:]] if out.rc else [])
        return Collected(out.stdout if out.ok else "", mode, bool(paths), [*status_err, *errors], paths, stalled or out.timed_out)
    c = await tolerant_collect(ws, base=base)
    return Collected(c["patch"], mode, bool(paths), [*status_err, *c["errors"]], paths, stalled or bool(c.get("stalled")))


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


async def repo_broken(ws: Path, base: str | None) -> list[str]:
    """Problems with the agent's repository that make its work uncollectable by its own doing (M10 audit V1: a corrupt
    ``.git/HEAD`` or index, or a deleted base commit, was masked as a harness failure). M10 lab audit R-5 item 4: a git call
    that timed out is *not* a problem of the repository (a host stall is not the policy's 0): only a git that answered with a
    failure is reported."""
    ws = Path(ws)
    if not (ws / ".git").exists():
        return ["the workspace repository (.git) was deleted"]
    out = []
    st = await _git(ws, "status", "--porcelain")
    if not st.timed_out and st.rc != 0:
        out.append(f"git status fails in the workspace: {st.stderr.strip()[:200]}")
    if base:
        base_ok = await _git(ws, "cat-file", "-e", f"{base}^{{commit}}")
        if not base_ok.timed_out and base_ok.rc != 0:
            out.append(f"the base commit {base[:12]} is gone from the workspace repository")
    return out
