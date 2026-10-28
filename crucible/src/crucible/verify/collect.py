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
external diff driver. Open-risks pass 2026-10-28 (harness audit R6): a filter driver the repository's own config defines
(``filter.<name>.clean`` with a ``.gitattributes`` line) ran its command on the host at ``git add``: host code execution and any
content substituted into the patch. Each one is now a pass-through (``safe_flags``); drivers from the host's own global config
(git-lfs) are the host's and stay. ``repo_hazards`` refuses a redirected repository before any host git runs in it, and the diff's
shape is fixed by flags (``DIFF_FORMAT``), not by the repository's config. A stall that remains is told
apart by the re-collect: the same preserved workspace stalling twice is the policy's (``result["stalled"]``,
``WorkspaceHarness``: ``crucible:agent_broke_env``), never a second harness mask.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from crucible.verify.patchio import decode
from crucible.world.exec import ExecOut, _run_subprocess, _run_subprocess_raw

# M10 lab audit R-5 item 4 (audit R-5 (b): verify/collect.py:54): the longest one git call of the hook may take. A module
# global read at call time (a test patches it); the legacy hook chain of runtime.collect is held to the same bound.
GIT_TIMEOUT_S = 300.0
# M10 lab audit R-5 review round 1: per call, never global (``-c``): a policy's ``core.fsmonitor`` or ``core.hooksPath`` in the
# workspace's own ``.git/config`` names no command host git runs. Open-risks pass 2026-10-28 (harness audit R6): submodule
# recursion off too (a nested repository's own config is the policy's as well)
GIT_FLAGS = ["-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "submodule.recurse=false"]
# the shape of a diff, whatever the repository's config says (``diff.noprefix``, ``diff.mnemonicPrefix``, ``color.ui=always``,
# ``diff.relative`` re-shape a patch so it no longer applies or no longer matches the other renderer's): flags, not config
DIFF_FORMAT = ["--no-ext-diff", "--no-textconv", "--no-color", "--src-prefix=a/", "--dst-prefix=b/"]
# a filter driver name ``-c filter.<name>.clean=`` can name exactly (git splits ``-c`` at the first ``=``)
SAFE_FILTER = re.compile(r"[A-Za-z0-9._+-]+")
SHA = re.compile(r"[0-9a-f]{40}([0-9a-f]{24})?")


def _repo_configs(ws: Path) -> list[Path]:
    """The repository's own config files (the policy can write both)."""
    g = Path(ws) / ".git"
    return [p for p in (g / "config", g / "config.worktree") if p.is_file()]


async def filter_drivers(ws: Path) -> list[str]:
    """The filter drivers the agent's repository defines (``filter.<name>.*`` in its own config files, includes followed).
    Read with ``git config --file``: no repository discovery, nothing in the repository runs. Open-risks pass 2026-10-28
    (harness audit R6): ``git add`` runs a driver's ``clean`` command on the host, so a planted driver was host code
    execution and could substitute any content into the collected patch."""
    names: list[str] = []
    for f in _repo_configs(ws):
        out = await _run_subprocess(["git", "config", "--file", str(f), "--includes", "--name-only", "--get-regexp", r"^filter\."], cwd=None, env=None,
                                    timeout_s=60, stdin=None)
        for k in out.stdout.splitlines():
            n = k.strip()[len("filter."):].rpartition(".")[0]
            if n and n not in names:
                names.append(n)
    return names


def filter_flags(names: list[str]) -> list[str]:
    """``-c`` overrides that make each named driver a pass-through (an empty command is no filter); a name ``-c`` cannot
    name exactly is left out (``repo_hazards`` refuses the repository first)."""
    out: list[str] = []
    for n in names:
        if SAFE_FILTER.fullmatch(n):
            out += ["-c", f"filter.{n}.clean=", "-c", f"filter.{n}.smudge=", "-c", f"filter.{n}.process=", "-c", f"filter.{n}.required=false"]
    return out


async def safe_flags(ws: Path) -> list[str]:
    """``GIT_FLAGS`` plus the overrides for the repository's own filter drivers: every host git call in the agent's repository."""
    return [*GIT_FLAGS, *filter_flags(await filter_drivers(ws))]


async def repo_hazards(ws: Path) -> list[str]:
    """Why host git must not run in the agent's repository at all (open-risks pass 2026-10-28, harness audit R6): ``.git``
    replaced by a symlink or a ``gitdir:`` file, a ``commondir`` redirect, ``core.worktree`` or ``core.bare`` in its config
    (host git would index, and reset the index of, another repository or another tree on the host), or a filter driver
    whose name no ``-c`` override can neutralise. Read without running git in the repository."""
    g = Path(ws) / ".git"
    if g.is_symlink():
        return ["the workspace's .git is a symlink (the repository was redirected)"]
    if not g.exists():
        return []  # a deleted repository is reported as such by the caller
    if not g.is_dir():
        return ["the workspace's .git is a file, not the repository (a gitdir redirect)"]
    out = []
    if (g / "commondir").exists():
        out.append("the workspace repository has a commondir redirect")
    for f in _repo_configs(ws):
        got = await _run_subprocess(["git", "config", "--file", str(f), "--includes", "--get-regexp", r"^core\.(worktree|bare)$"], cwd=None, env=None,
                                    timeout_s=60, stdin=None)
        for line in got.stdout.splitlines():
            key, _, val = line.partition(" ")
            if key == "core.worktree" and val.strip() and Path(val.strip() if Path(val.strip()).is_absolute() else g / val.strip()).resolve() != Path(ws).resolve():
                out.append(f"core.worktree points outside the workspace ({val.strip()[:120]})")
            elif key == "core.bare" and val.strip().lower() in ("true", "yes", "on", "1"):
                out.append("core.bare is set in the workspace repository")
    bad = [n for n in await filter_drivers(ws) if not SAFE_FILTER.fullmatch(n)]
    if bad:
        out.append(f"a filter driver git cannot override ({bad[0][:60]!r})")
    return out


def snapshot_command(base: str, *, require_sha: bool = True) -> str:
    """The shell (bash, run in the workspace inside the World's container) that prints the working tree as a binary patch
    against ``base`` through a private index, the way the final collect renders it: ``DERIVED`` excluded, the format fixed
    (``DIFF_FORMAT``), no system or global config, no hooks, fsmonitor or submodule recursion, the repository's own filter
    drivers made pass-throughs, ``.git`` and the work tree named explicitly (``core.worktree`` / ``core.bare`` ignored).
    Open-risks pass 2026-10-28 (harness audit R6): ``base`` is the commit recorded when the World opened, never the ``base``
    tag the policy can move. The script has no braces besides ``base`` (``train.bridges.slime.episode.SNAPSHOT_CMD`` is
    a ``str.format`` template of it)."""
    if require_sha and not SHA.fullmatch(base):
        raise ValueError(f"a snapshot diffs against the recorded base commit (a sha), got {base!r}")
    pats = " ".join(shlex.quote(d) for d in DERIVED)
    cfg = ("-c 'safe.directory=*' -c core.fsmonitor=false -c core.hooksPath=/dev/null -c core.quotepath=off -c \"core.excludesFile=$W/x\" "
           "-c color.ui=never -c diff.noprefix=false -c diff.mnemonicPrefix=false -c diff.relative=false -c submodule.recurse=false")
    return ("set -e; if [ ! -d .git ] || [ -L .git ]; then echo 'the workspace .git is not a directory' >&2; exit 3; fi; "
            "W=$(mktemp -d); trap 'rm -rf \"$W\"' EXIT; export HOME=\"$W\" XDG_CONFIG_HOME=\"$W\" GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null; "
            f"printf '%s\\n' {pats} > \"$W/x\"; : > \"$W/f\"; "
            "for c in .git/config .git/config.worktree; do if [ -f \"$c\" ]; then git config --file \"$c\" --includes --name-only "
            "--get-regexp '^filter[.]' >> \"$W/f\" 2>/dev/null || true; fi; done; set --; "
            "while IFS= read -r k; do n=$(printf '%s' \"$k\" | sed 's/^filter[.]//; s/[.][^.]*$//'); "
            "case \"$n\" in ''|*[!A-Za-z0-9._+-]*) echo \"a filter driver git cannot override: $n\" >&2; exit 3;; esac; "
            "set -- \"$@\" -c \"filter.$n.clean=\" -c \"filter.$n.smudge=\" -c \"filter.$n.process=\" -c \"filter.$n.required=false\"; done < \"$W/f\"; "
            "cp .git/index \"$W/i\" 2>/dev/null || true; set -f; "
            f"GIT_INDEX_FILE=\"$W/i\" git --git-dir=.git --work-tree=. {cfg} \"$@\" add -A --ignore-errors . >/dev/null 2>&1 || true; "
            f"GIT_INDEX_FILE=\"$W/i\" git --git-dir=.git --work-tree=. {cfg} \"$@\" diff --cached --binary {' '.join(DIFF_FORMAT)} {base}")

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

    flags = await safe_flags(ws)  # the repository's own filter drivers are pass-throughs here (harness audit R6)

    async def git(*a: str) -> ExecOut:
        out = await _run_subprocess(["git", "-C", str(ws), *flags, *a], cwd=None, env=None, timeout_s=GIT_TIMEOUT_S, stdin=None)
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
        # M7 R3: the diff's bytes (CRLF, Latin-1), decoded losslessly; ``git()`` decodes with errors="replace"
        _rc, raw, _err, timed_out, _ = await _run_subprocess_raw(["git", "-C", str(ws), *flags, "diff", "--cached", "--binary", *DIFF_FORMAT, base],
                                                                cwd=None, env=None, timeout_s=GIT_TIMEOUT_S, stdin=None)
        if timed_out:
            stalled.append(f"git diff timed out after {GIT_TIMEOUT_S:g}s")
        if stalled:  # never a half-built index's patch: it would be graded as the agent's answer
            return failed()
        patch = decode(raw)
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
