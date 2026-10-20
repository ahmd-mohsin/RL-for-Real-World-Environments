"""The collect hook (DESIGN §8.3 layer 5, Appendix B.5): get the agent's work out of a workspace, whatever state it is in.

task04's first frontier run produced an *empty* patch: a gitignored ``build/`` made ``git add`` fail inside an ``&&``
chain. The hook here never aborts on one failing path, includes untracked files, drops derived artifacts (build
trees, caches, bytecode) that would leak or bloat the patch, and records what it dropped. ``COLLECT_SH`` is the same
logic as a shell snippet for Harbor's ``[[verifier.collect]]`` and agentenv's artifact step.

M10 lab audit R-5 item 4: ``collect`` is a coroutine and every git call of the hook goes through
``world.exec._run_subprocess`` (off the event loop, in its own session, bounded by ``GIT_TIMEOUT_S``, killed with its
session on a timeout or a cancel). A synchronous ``subprocess.run`` with no timeout froze every attempt's agent loop, every
``wait_for`` and the wall clock while one workspace's git stalled. A git call that runs past its timeout fails the whole
collection (empty patch plus an error naming the call, which ``runtime.collect.classify`` reads as
``crucible:empty_collection``: masked, re-collected from the preserved workspace): a patch built from a half-finished
``git add`` would otherwise be graded as the agent's answer (P16), and the killed git's ``index.lock`` is removed so the
re-collect and the ``git reset`` can run.

M10 lab audit R-5 review round 1: the repository is the agent's, so what its ``.git/config`` can make git run is the policy's
choice, and a stall there must not become a harness mask. Every call of the hook (and ``runtime.collect``'s) runs with
``GIT_FLAGS`` (no ``core.fsmonitor`` command, no hooks: the two ways a plain ``git status`` or ``git add`` runs a command the policy
named; the same two the grader guard turns off, ``world.guard.guard.GIT_FLAGS``), and ``git diff`` with no textconv and no
external diff driver. A filter driver (``filter.<name>.clean`` with a ``.gitattributes`` line) is still honoured by ``git add``;
it is not neutralised here because a sanitised config would also drop a task's own filters (git-lfs). A stall that remains is told
apart by the re-collect: the same preserved workspace stalling twice is the policy's (``result["stalled"]``,
``WorkspaceHarness``: ``crucible:agent_broke_env``), never a second harness mask.
"""

from __future__ import annotations

from pathlib import Path

from crucible.world.exec import ExecOut, _run_subprocess

# M10 lab audit R-5 item 4 (audit R-5 (b): verify/collect.py:54): the longest one git call of the hook may take. A module
# global read at call time (a test patches it); the legacy hook chain of runtime.collect is held to the same bound.
GIT_TIMEOUT_S = 300.0
# M10 lab audit R-5 review round 1: per call, never global (``-c``): a policy's ``core.fsmonitor`` or ``core.hooksPath`` in the
# workspace's own ``.git/config`` names no command host git runs
GIT_FLAGS = ["-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null"]

# build trees only at the repository root (M7 audit V20: `build` and `dist` matched at any depth and dropped new sources such
# as tools/build/gen.py); caches and bytecode anywhere
DERIVED = ("/build/", "/build-*/", "/cmake-build-*/", "/dist/", "*.egg-info", "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache", "*.o", "*.so",
           "*.a", ".ninja_*", "CMakeFiles/", "node_modules/", "/.venv/", ".cache/")


def is_derived(path: str) -> bool:
    """Whether a work-tree path is a derived artifact the hook drops (the same patterns, matched like git)."""
    import fnmatch

    parts = path.strip("/").split("/")
    for pat in DERIVED:
        anchored, p = pat.startswith("/"), pat.strip("/")
        if anchored:
            if fnmatch.fnmatch(parts[0], p) and (len(parts) > 1 or not pat.endswith("/")):
                return True
        elif any(fnmatch.fnmatch(x, p) for x in parts):
            return True
    return False


COLLECT_SH = r"""set +e
cd "${1:-.}" || exit 0
git config --global --add safe.directory "$(pwd)" 2>/dev/null
for p in EXCLUDES; do echo "$p" >> .git/info/exclude 2>/dev/null; done
git add -A --ignore-errors . 2>/dev/null
git ls-files -o --exclude-standard -z 2>/dev/null | xargs -0 -r git add --ignore-errors -- 2>/dev/null
git diff --cached --binary "${2:-HEAD}" > "${3:-/logs/artifacts/model.patch}" 2>/dev/null
true
""".replace("EXCLUDES", " ".join(f"'{d}'" for d in DERIVED))


async def collect(workspace: Path, *, base: str = "HEAD", extra_excludes: tuple = ()) -> dict:
    """``base``: the commit the agent started from, recorded before it ran (a sha). M7 audit: the default ``HEAD`` lost
    committed work (an agent that committed its fix yielded an empty patch), and a ``base`` tag the agent can delete
    either masked its failure or lost its work. Runs the hook on a local work tree; returns the patch and what was
    excluded or failed. M10 lab audit R-5 item 4: a coroutine (see the module doc); a git call past ``GIT_TIMEOUT_S`` fails
    the collection: ``patch`` is empty, ``errors`` names the call (``git add timed out after 300s``), ``stalled`` is True
    and nothing after it runs."""
    ws = Path(workspace)
    errors: list[str] = []
    stalled: list[str] = []

    async def git(*a: str) -> ExecOut:
        out = await _run_subprocess(["git", "-C", str(ws), *GIT_FLAGS, *a], cwd=None, env=None, timeout_s=GIT_TIMEOUT_S, stdin=None)
        if out.timed_out:
            stalled.append(f"git {a[0]} timed out after {GIT_TIMEOUT_S:g}s")
        return out

    def failed() -> dict:
        return {"patch": "", "errors": [*errors, *stalled], "excluded": [], "files": 0, "stalled": True}

    excl = ws / ".git" / "info" / "exclude"
    excl.parent.mkdir(parents=True, exist_ok=True)
    prior = excl.read_text() if excl.exists() else ""
    excl.write_text(prior + "".join(f"{d}\n" for d in (*DERIVED, *extra_excludes)))
    try:
        r = await git("add", "-A", "--ignore-errors", ".")
        if stalled:
            return failed()
        if r.rc:
            errors.append(r.stderr[-300:])
        listing = await git("ls-files", "-o", "--exclude-standard", "-z")
        if stalled:
            return failed()
        for u in filter(None, listing.stdout.split("\0")):
            added = await git("add", "--ignore-errors", "--", u)
            if stalled:
                return failed()
            if added.rc:
                errors.append(f"cannot add {u}")
        diff = await git("diff", "--cached", "--binary", "--no-textconv", "--no-ext-diff", base)
        if stalled:  # never a half-built index's patch: it would be graded as the agent's answer
            return failed()
        patch = diff.stdout
        ignored_out = await git("ls-files", "-o", "-i", "--exclude-standard", "--directory")
        if stalled:
            return failed()
        ignored = [x for x in ignored_out.stdout.splitlines() if x]
        return {"patch": patch, "errors": errors, "excluded": ignored[:50], "files": len([ln for ln in patch.splitlines() if ln.startswith("diff --git")]),
                "stalled": False}
    finally:
        excl.write_text(prior)
        if stalled:  # the killed git left its lock: the agent phase is over, so nothing else holds it
            try:
                (ws / ".git" / "index.lock").unlink(missing_ok=True)
            except OSError:
                pass
        await git("reset", "-q")
