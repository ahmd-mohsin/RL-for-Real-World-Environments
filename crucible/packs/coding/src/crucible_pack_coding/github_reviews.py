"""GitHub review threads -> grounding records (coding pack; a first slice of the M3 github connector).

Works without authentication: the repo-wide review-comments endpoint returns 100 comments per call
(path, diff_hunk, original/current commit, original line, outdated flag, body, author, time), and file
contents come from raw.githubusercontent.com, which does not count against the API rate limit.
Set ``GH_TOKEN`` to raise the API limit. Every response is cached on disk (re-runs make no requests).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from crucible.core.schema.signal import SignalTarget
from crucible.core.schema.trace import Record

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"
CODE_EXT = (".cu", ".cuh", ".cpp", ".hpp", ".h", ".cc", ".cxx", ".hxx", ".inl", ".py")


class GitHubCache:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "crucible-research"}
        if os.environ.get("GH_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['GH_TOKEN']}"
        self.http = httpx.Client(headers=headers, timeout=60, follow_redirects=True)
        self.api_calls = 0
        self.raw_calls = 0

    def _path(self, url: str) -> Path:
        return self.root / hashlib.sha256(url.encode()).hexdigest()[:24]

    def get_json(self, url: str) -> Any:
        p = self._path(url)
        if p.exists():
            return json.loads(p.read_text())
        for attempt in range(3):
            r = self.http.get(url)
            self.api_calls += 1
            if r.status_code == 403 and r.headers.get("x-ratelimit-remaining") == "0":
                reset = int(r.headers.get("x-ratelimit-reset", "0"))
                raise RuntimeError(f"GitHub API rate limit exhausted (resets at {time.ctime(reset)}); set GH_TOKEN to continue")
            if r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            p.write_text(r.text)
            return r.json()
        r.raise_for_status()

    def get_raw(self, repo: str, sha: str, path: str) -> str | None:
        url = f"{RAW}/{repo}/{sha}/{path}"
        p = self._path(url)
        if p.exists():
            t = p.read_text()
            return None if t == "\0MISSING\0" else t
        r = self.http.get(url)
        self.raw_calls += 1
        if r.status_code == 404:
            p.write_text("\0MISSING\0")
            return None
        r.raise_for_status()
        p.write_text(r.text)
        return r.text


@dataclass
class ReviewComment:
    id: int
    repo: str
    pr: int
    path: str
    body: str
    author: str
    created_at: str
    diff_hunk: str
    original_commit: str
    commit: str
    original_line: int | None
    original_start_line: int | None
    outdated: bool
    in_reply_to: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def anchor(self) -> list[str]:
        """The new-side code lines the comment is attached to (end of the diff hunk), whitespace-normalized."""
        lines = [ln for ln in self.diff_hunk.splitlines()[1:] if not ln.startswith("-")]
        n = 1 if self.original_start_line is None or self.original_line is None else max(1, self.original_line - self.original_start_line + 1)
        body = [normalize(ln[1:] if ln[:1] in "+ " else ln) for ln in lines[-max(n, 1):]]
        # extend upward with context until the anchor carries real content
        i = len(lines) - len(body)
        while sum(len(b) for b in body) < 25 and i > 0:
            i -= 1
            body.insert(0, normalize(lines[i][1:] if lines[i][:1] in "+ " else lines[i]))
        return body

    def to_record(self) -> Record:
        span = [self.original_start_line or self.original_line or 0, self.original_line or 0]
        return Record(
            id=f"{self.repo}#{self.pr}/c{self.id}",
            source=f"{self.repo}#{self.pr}",
            author=self.author,
            role="reviewer",
            time=self.created_at,
            text=self.body.strip(),
            target=SignalTarget(app="repo", object=self.path, span=span),
            tags=["review_comment"],
            meta={"original_commit": self.original_commit, "commit": self.commit, "outdated": self.outdated, "anchor": "\n".join(self.anchor)},
        )


_WS = re.compile(r"\s+")


def pr_heads(repo: str) -> dict[int, str]:
    """PR number -> final head sha via ``git ls-remote`` (one request, no API quota)."""
    import subprocess

    out = subprocess.run(["git", "ls-remote", f"https://github.com/{repo}", "refs/pull/*/head"], capture_output=True, text=True, timeout=300, check=True).stdout
    heads = {}
    for line in out.splitlines():
        sha, ref = line.split("\t")
        heads[int(ref.split("/")[2])] = sha
    return heads


def normalize(line: str) -> str:
    return _WS.sub(" ", line).strip()


def fetch_review_comments(cache: GitHubCache, repo: str, *, pages: int = 5, exts: tuple[str, ...] = CODE_EXT, min_body: int = 25) -> list[ReviewComment]:
    out: list[ReviewComment] = []
    for page in range(1, pages + 1):
        data = cache.get_json(f"{API}/repos/{repo}/pulls/comments?per_page=100&sort=created&direction=desc&page={page}")
        if not data:
            break
        for c in data:
            if c.get("in_reply_to_id") or not str(c.get("path", "")).endswith(exts):
                continue
            user = c.get("user") or {}
            login = str(user.get("login", ""))
            if user.get("type") == "Bot" or login.endswith("[bot]") or "coderabbit" in login.lower():
                continue  # automated reviewers are not expert signal
            body = (c.get("body") or "").strip()
            if len(body) < min_body or not c.get("diff_hunk") or c.get("original_line") is None:
                continue
            pr = int(str(c["pull_request_url"]).rsplit("/", 1)[1])
            out.append(ReviewComment(
                id=c["id"], repo=repo, pr=pr, path=c["path"], body=body, author=(c.get("user") or {}).get("login", "?"), created_at=c["created_at"],
                diff_hunk=c["diff_hunk"], original_commit=c["original_commit_id"], commit=c["commit_id"], original_line=c.get("original_line"),
                original_start_line=c.get("original_start_line"), outdated=c.get("position") is None and c.get("line") is None,
            ))
    return out
