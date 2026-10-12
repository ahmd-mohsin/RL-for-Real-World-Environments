"""The collect hook (DESIGN §8.3 layer 5, Appendix B.5): get the agent's work out of a workspace, whatever state it is in.

task04's first frontier run produced an *empty* patch: a gitignored ``build/`` made ``git add`` fail inside an ``&&``
chain. The hook here never aborts on one failing path, includes untracked files, drops derived artifacts (build
trees, caches, bytecode) that would leak or bloat the patch, and records what it dropped. ``COLLECT_SH`` is the same
logic as a shell snippet for Harbor's ``[[verifier.collect]]`` and agentenv's artifact step.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

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


def collect(workspace: Path, *, base: str = "HEAD", extra_excludes: tuple = ()) -> dict:
    """``base``: the commit the agent started from, recorded before it ran (a sha). M7 audit: the default ``HEAD`` lost
    committed work (an agent that committed its fix yielded an empty patch), and a ``base`` tag the agent can delete
    either masked its failure or lost its work. Runs the hook on a local work tree; returns the patch and what was
    excluded or failed."""
    ws = Path(workspace)

    def git(*a: str, inp: bytes | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(ws), *a], capture_output=True, input=inp, check=False)

    excl = ws / ".git" / "info" / "exclude"
    excl.parent.mkdir(parents=True, exist_ok=True)
    prior = excl.read_text() if excl.exists() else ""
    excl.write_text(prior + "".join(f"{d}\n" for d in (*DERIVED, *extra_excludes)))
    try:
        errors = []
        r = git("add", "-A", "--ignore-errors", ".")
        if r.returncode:
            errors.append(r.stderr.decode(errors="replace")[-300:])
        untracked = git("ls-files", "-o", "--exclude-standard", "-z").stdout.split(b"\0")
        for u in filter(None, untracked):
            if git("add", "--ignore-errors", "--", u.decode()).returncode:
                errors.append(f"cannot add {u.decode()}")
        patch = git("diff", "--cached", "--binary", base).stdout.decode(errors="replace")
        ignored = [x for x in git("ls-files", "-o", "-i", "--exclude-standard", "--directory").stdout.decode().splitlines() if x]
        return {"patch": patch, "errors": errors, "excluded": ignored[:50], "files": len([ln for ln in patch.splitlines() if ln.startswith("diff --git")])}
    finally:
        excl.write_text(prior)
        git("reset", "-q")
