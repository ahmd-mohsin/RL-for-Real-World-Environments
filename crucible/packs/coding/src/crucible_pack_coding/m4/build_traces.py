"""Ingest PR traces for M4: every PR whose API payload is cached (run ``fetch`` first; resumable under the rate limit)."""

from __future__ import annotations

import json
import sys
import urllib.request

from crucible.connectors.redact import RedactionPolicy, redact
from crucible.connectors.trace_store import TraceStore
from crucible.core.store import Store
from crucible_pack_coding import PACK_ROOT
from crucible_pack_coding.connectors.github_pr import GitHubPRConnector

DATA = PACK_ROOT.parent.parent / "data"
NOW = "2026-10-09T12:00:00+00:00"  # outcome windows are computed against this instant (recorded for reproducibility)


def connector() -> GitHubPRConnector:
    return GitHubPRConnector(DATA / "cache" / "github", DATA / "cache" / "git", now=NOW)


def cached(c: GitHubPRConnector, ref) -> bool:
    repo, n = ref.meta["repo"], ref.meta["pr"]
    pr_url = f"https://api.github.com/repos/{repo}/pulls/{n}"
    if not c.gh._path(pr_url).exists():
        return False
    head = json.loads(c.gh._path(pr_url).read_text())["head"]["sha"]
    return all(c.gh._path(u).exists() for u in (f"{pr_url}/reviews?per_page=100", f"{pr_url}/comments?per_page=100",
                                                f"https://api.github.com/repos/{repo}/commits/{head}/check-runs?per_page=100"))


def fetch(repo: str, scope: dict) -> None:
    """Fill the API cache for as many PRs as the remaining (anonymous) quota allows."""
    c = connector()
    for ref in c.discover({"repo": repo, **scope}):
        if cached(c, ref):
            continue
        left = json.load(urllib.request.urlopen("https://api.github.com/rate_limit"))["resources"]["core"]["remaining"]
        if left < 5:
            print("rate limit: stopping (re-run after the reset)")
            return
        c.fetch(ref)


def build(repo: str, scope: dict, store: TraceStore) -> list[str]:
    c = connector()
    ids = []
    for ref in c.discover({"repo": repo, **scope}):
        if not cached(c, ref):
            continue
        t = redact(c.normalize(c.fetch(ref)), RedactionPolicy())
        store.put(t)
        ids.append(t.trace_id)
    return ids


def main() -> None:
    repo = sys.argv[1] if len(sys.argv) > 1 else "PointCloudLibrary/pcl"
    scope = {"from_review_pages": 3, "min_comments": 3}
    if "--fetch" in sys.argv:
        fetch(repo, scope)
    ts = TraceStore(Store(DATA / "m4"))
    ids = build(repo, scope, ts)
    print(f"{len(ids)} traces stored for {repo}")


if __name__ == "__main__":
    main()
