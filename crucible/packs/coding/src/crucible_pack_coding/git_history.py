"""Git facts for PR traces, from a local blobless bare clone (no API quota): commits, added lines, blame, line
tracking across later commits, and the 30 days after a merge (reverts, hotfixes).

Blame is the coding attribution rule (ARCHITECTURE M4): a review comment at ``path:line`` of commit C grades
the PR commit that last wrote that line, and a later change to it is the correction.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Commit:
    sha: str
    parents: list[str]
    author: str
    email: str
    time: str  # committer date, ISO 8601
    subject: str
    body: str = ""
    added: dict[str, list[list[int]]] = field(default_factory=dict)  # path -> [[first, last], ...] lines added in this commit


class GitRepo:
    def __init__(self, path: Path, url: str | None = None):
        self.path = Path(path)
        self.url = url

    def ensure(self) -> GitRepo:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "-q", "--bare", "--filter=blob:none", self.url, str(self.path)], check=True, timeout=3600)
            self.git("fetch", "-q", "origin", "+refs/pull/*/head:refs/pull/*/head", timeout=3600)
        return self

    def git(self, *args: str, timeout: float = 600, check: bool = True) -> str:
        r = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, check=False,  # old sources are Latin-1
                           env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(Path.home())})
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args[:3])}: {r.stderr.strip()[-400:]}")
        return r.stdout

    def has(self, rev: str) -> bool:
        return bool(self.git("cat-file", "-t", rev, check=False).strip())

    def commits(self, base: str, head: str) -> list[Commit]:
        """PR commits oldest first (merges from the base branch into the PR are skipped)."""
        shas = self.git("rev-list", "--reverse", "--no-merges", f"{base}..{head}").split()
        return [self.commit(s) for s in shas]

    def commit(self, sha: str) -> Commit:
        fmt = "%H%x00%P%x00%an%x00%ae%x00%cI%x00%s%x00%b"
        h, parents, an, ae, ci, subj, body = self.git("show", "-s", f"--format={fmt}", sha).split("\x00", 6)
        c = Commit(sha=h, parents=parents.split(), author=an, email=ae, time=ci, subject=subj, body=body.strip())
        if c.parents:
            c.added = self.added_lines(c.parents[0], c.sha)
        return c

    def added_lines(self, a: str, b: str, paths: list[str] | None = None) -> dict[str, list[list[int]]]:
        out = self.git("diff", "-U0", "--no-color", "--no-renames", a, b, "--", *(paths or []))
        res: dict[str, list[list[int]]] = {}
        path = None
        for line in out.splitlines():
            if line.startswith("+++ "):
                path = line[6:] if line.startswith("+++ b/") else None
            elif line.startswith("@@") and path:
                m = HUNK.match(line)
                if m:
                    start, n = int(m.group(3)), int(m.group(4) or 1)
                    if n:
                        res.setdefault(path, []).append([start, start + n - 1])
        return res

    def hunks(self, a: str, b: str, path: str) -> list[tuple[int, int, int, int]]:
        """(old_start, old_len, new_start, new_len) for one file between two revisions."""
        out = self.git("diff", "-U0", "--no-color", "--no-renames", a, b, "--", path)
        hs = []
        for line in out.splitlines():
            m = HUNK.match(line)
            if m:
                hs.append((int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1)))
        return hs

    def hunks_multi(self, a: str, b: str, paths: list[str] | None = None) -> dict[str, list[tuple[int, int, int, int]]]:
        """All files' hunks between two revisions in one diff: path -> [(old_start, old_len, new_start, new_len)]."""
        out = self.git("diff", "-U0", "--no-color", "--no-renames", a, b, "--", *(paths or []))
        res: dict[str, list[tuple[int, int, int, int]]] = {}
        path = None
        for line in out.splitlines():
            if line.startswith("--- "):
                path = line[6:] if line.startswith("--- a/") else None
            elif line.startswith("+++ ") and path is None:
                path = line[6:] if line.startswith("+++ b/") else None
            elif line.startswith("@@") and path:
                m = HUNK.match(line)
                if m:
                    res.setdefault(path, []).append((int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1)))
        return res

    def blame_range(self, rev_range: str, path: str, first: int, last: int) -> set[str]:
        """Commits that wrote lines [first, last] of ``path``; ``rev_range`` like ``base..merge`` bounds the history walk."""
        out = self.git("blame", "--porcelain", "-L", f"{first},{last}", rev_range, "--", path, check=False)
        return {ln.split(" ", 1)[0] for ln in out.splitlines() if re.match(r"^[0-9a-f]{40} \d+ \d+", ln)}

    def blame(self, rev: str, path: str, line: int) -> str | None:
        out = self.git("blame", "--porcelain", "-L", f"{line},{line}", rev, "--", path, check=False)
        first = out.split("\n", 1)[0].split(" ")
        return first[0] if first and len(first[0]) == 40 else None

    def track_line(self, path: str, line: int, after: list[Commit]) -> tuple[str | None, int | None]:
        """Follow ``line`` of ``path`` through ``after`` (commits in order). Returns (sha of the first commit that
        modified or deleted it, None) or (None, its final line number)."""
        cur = line
        for c in after:
            if not c.parents:
                continue
            shift = 0
            for os_, ol, _ns, nl in self.hunks(c.parents[0], c.sha, path):
                if ol and os_ <= cur <= os_ + ol - 1:
                    return c.sha, None
                if (ol and os_ + ol - 1 < cur) or (not ol and os_ < cur):
                    shift += nl - ol
            cur += shift
        return None, cur

    def track_window(self, path: str, first: int, last: int, after: list[Commit]) -> str | None:
        """Sha of the first commit in ``after`` that modifies a line in [first, last] or inserts lines right next to
        them (a reviewer's request is often satisfied one line above or below the commented line)."""
        lo, hi = first, last
        for c in after:
            if not c.parents:
                continue
            shift_lo = shift_hi = 0
            for os_, ol, _ns, nl in self.hunks(c.parents[0], c.sha, path):
                end = os_ + ol - 1
                if (ol and os_ <= hi and end >= lo) or (not ol and lo - 1 <= os_ <= hi):
                    return c.sha
                if (ol and end < lo) or (not ol and os_ < lo):
                    shift_lo += nl - ol
                if (ol and end < hi) or (not ol and os_ < hi):
                    shift_hi += nl - ol
            lo, hi = lo + shift_lo, hi + shift_hi
        return None

    def window_changed(self, a: str, b: str, path: str, first: int, last: int) -> bool:
        return any((ol and os_ <= last and os_ + ol - 1 >= first) or (not ol and first - 1 <= os_ <= last) for os_, ol, _n, _l in self.hunks(a, b, path))

    def file_lines(self, rev: str, path: str) -> list[str] | None:
        out = subprocess.run(["git", "-C", str(self.path), "show", f"{rev}:{path}"], capture_output=True, timeout=300, check=False)
        return out.stdout.decode("utf-8", "replace").splitlines() if out.returncode == 0 else None

    def log_since(self, ref: str, since: str, until: str, paths: list[str]) -> list[Commit]:
        shas = self.git("log", "--reverse", "--no-merges", "--format=%H", f"--since={since}", f"--until={until}", ref, "--", *paths).split()
        return [self.commit(s) for s in shas]
