"""A small GitHub REST client for scouts: unauthenticated by design (no token is read), every response cached on disk,
paced per endpoint class and backed off on limits.

- **Pacing.** ``search`` (10 requests a minute unauthenticated) waits ``search_interval`` seconds between calls; ``core``
  (60 an hour) watches ``X-RateLimit-Remaining`` and waits for ``X-RateLimit-Reset`` when it hits 0 (or raises
  ``RateLimited`` when that is further away than ``max_wait``); ``raw`` (raw.githubusercontent.com, file contents) is not
  part of the API quota and only gets a short interval.
- **Backoff.** 403/429 with ``Retry-After`` (or a reset header) wait that long, else exponential, at most ``retries``
  times; 5xx retries too. 404 is cached as a miss.
- **Cache.** ``<cache>/<sha256 of the URL>.json`` holds ``{url, status, body, headers}``; a cached URL never goes out again
  (``refresh=True`` overrides). ``offline=True`` serves only the cache (tests, reruns).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"


class RateLimited(RuntimeError):
    """The API quota is spent for longer than the caller is willing to wait."""


class GitHub:
    def __init__(self, cache: Path, *, search_interval: float = 6.5, core_interval: float = 1.0, raw_interval: float = 0.2, retries: int = 4,
                 max_wait: float = 900.0, offline: bool = False, user_agent: str = "crucible-factory-scout",
                 opener: Callable[[urllib.request.Request, float], Any] | None = None, sleep: Callable[[float], None] = time.sleep):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.interval = {"search": search_interval, "core": core_interval, "raw": raw_interval}
        self.retries, self.max_wait, self.offline, self.user_agent = retries, max_wait, offline, user_agent
        self._open = opener or (lambda req, timeout: urllib.request.urlopen(req, timeout=timeout))  # https only: fetch() refuses other hosts
        self._sleep = sleep
        self._last: dict[str, float] = {}
        self.remaining: dict[str, int | None] = {"core": None, "search": None}
        self.reset_at: dict[str, float] = {}
        self.calls = {"network": 0, "cached": 0}

    @staticmethod
    def kind_of(url: str) -> str:
        if url.startswith(RAW):
            return "raw"
        return "search" if url.startswith(f"{API}/search/") else "core"

    def _path(self, url: str) -> Path:
        return self.cache / f"{hashlib.sha256(url.encode()).hexdigest()}.json"

    def _pace(self, kind: str) -> None:
        if kind in self.reset_at and (self.remaining.get(kind) == 0):
            wait = self.reset_at[kind] - time.time() + 1
            if wait > self.max_wait:
                raise RateLimited(f"{kind} quota spent; resets in {wait:.0f}s")
            if wait > 0:
                self._sleep(wait)
        gap = self.interval[kind] - (time.monotonic() - self._last.get(kind, 0.0))
        if gap > 0:
            self._sleep(gap)
        self._last[kind] = time.monotonic()

    def fetch(self, url: str, params: Mapping[str, Any] | None = None, *, refresh: bool = False) -> tuple[int, Any]:
        """``(status, body)``: JSON for the API, text for raw files; ``(404, None)`` for a miss."""
        if params:
            url = f"{url}?{urllib.parse.urlencode(sorted((k, str(v)) for k, v in params.items()))}"
        if not url.startswith(("https://api.github.com/", "https://raw.githubusercontent.com/")):
            raise ValueError(f"not a GitHub API or raw URL: {url}")
        p = self._path(url)
        if p.exists() and not refresh:
            d = json.loads(p.read_text())
            self.calls["cached"] += 1
            return int(d["status"]), d["body"]
        if self.offline:
            raise RateLimited(f"offline and not cached: {url}")
        kind = self.kind_of(url)
        delay = 2.0
        for attempt in range(self.retries + 1):
            self._pace(kind)
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, **({"Accept": "application/vnd.github+json"} if kind != "raw" else {})})
            try:
                resp = self._open(req, 30.0)
                status, headers, raw = int(getattr(resp, "status", 200)), dict(resp.headers or {}), resp.read()
            except urllib.error.HTTPError as e:
                status, headers, raw = e.code, dict(e.headers or {}), e.read() if hasattr(e, "read") else b""
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt == self.retries:
                    raise RateLimited(f"network: {e}") from e
                self._sleep(delay)
                delay *= 2
                continue
            self.calls["network"] += 1
            if kind != "raw":
                rem = headers.get("X-RateLimit-Remaining") or headers.get("x-ratelimit-remaining")
                rst = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
                if rem is not None:
                    self.remaining[kind] = int(rem)
                if rst is not None:
                    self.reset_at[kind] = float(rst)
            if status in (403, 429) or status >= 500:
                if attempt == self.retries:
                    raise RateLimited(f"HTTP {status} on {url}")
                ra = headers.get("Retry-After") or headers.get("retry-after")
                wait = float(ra) if ra else (max(1.0, self.reset_at.get(kind, 0) - time.time() + 1) if status in (403, 429) and self.remaining.get(kind) == 0
                                             else delay)
                if wait > self.max_wait:
                    raise RateLimited(f"HTTP {status}: retry in {wait:.0f}s")
                self._sleep(wait)
                delay *= 2
                continue
            text = raw.decode("utf-8", "replace")
            body: Any = None
            if status == 200:
                body = text if kind == "raw" else json.loads(text)
            p.write_text(json.dumps({"url": url, "status": status, "body": body, "t": round(time.time(), 1)}))
            return status, body
        raise RateLimited(f"gave up on {url}")

    # -- endpoints -------------------------------------------------------------------------------------------------------------------
    def search(self, what: str, q: str, *, per_page: int = 30, sort: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"q": q, "per_page": per_page}
        if sort:
            params["sort"] = sort
        status, body = self.fetch(f"{API}/search/{what}", params)
        return list((body or {}).get("items") or []) if status == 200 else []

    def search_repos(self, q: str, **kw: Any) -> list[dict[str, Any]]:
        return self.search("repositories", q, **kw)

    def search_commits(self, q: str, **kw: Any) -> list[dict[str, Any]]:
        return self.search("commits", q, **kw)

    def search_issues(self, q: str, **kw: Any) -> list[dict[str, Any]]:
        return self.search("issues", q, **kw)

    def forks(self, full_name: str, *, per_page: int = 30) -> list[dict[str, Any]]:
        status, body = self.fetch(f"{API}/repos/{full_name}/forks", {"per_page": per_page, "sort": "stargazers"})
        return list(body or []) if status == 200 else []

    def raw(self, full_name: str, ref: str, path: str) -> str | None:
        status, body = self.fetch(f"{RAW}/{full_name}/{ref}/{path}")
        return body if status == 200 else None
