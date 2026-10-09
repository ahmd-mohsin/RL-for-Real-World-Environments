"""GitHub pull requests -> ``crucible.trace/v1`` (coding pack connector; DESIGN §5.1 ``github`` + ``vcs_history``).

One trace per PR, events in time order:
  pr_opened, commit (PR commits; ``data.added``: lines each one wrote), review (APPROVED / CHANGES_REQUESTED /
  COMMENTED), review_comment (path + line + the commit it was made on; replies linked), ci_run (check runs on the
  head), merged / closed, and post_merge_commit (the 30 days after the merge: commits on the default branch that
  revert the PR or change lines it wrote -- ``vcs_history``).

The API part (PR, reviews, comments, check runs) goes through the on-disk ``GitHubCache`` (works without a token;
``GH_TOKEN`` raises the limit); everything derivable from git comes from a local blobless clone. Facts the M4
mapper needs are computed here, once: which PR commit wrote each commented line (blame at the comment's
commit), whether and by which commit that line changed afterwards, whether a ``suggestion`` block was applied,
and which post-merge commits touched lines this PR wrote.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from crucible.connectors.base import RawRecord, SourceRef
from crucible.core.schema.signal import SignalTarget
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceSource
from crucible_pack_coding.git_history import Commit, GitRepo
from crucible_pack_coding.github_reviews import API, GitHubCache

SUGGESTION = re.compile(r"```suggestion[^\n]*\n(.*?)```", re.DOTALL)
FIX_BLOCK = re.compile(r"(?:suggest\w*(?: fix)?|fix(?:ed)? version|should be|use)\s*:?\s*\n*```[\w+-]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
REVERT = re.compile(r"^revert\b", re.IGNORECASE)
FIXLIKE = re.compile(r"\b(fix|fixes|fixed|bug|regression|revert|crash|broken|hotfix|wrong|incorrect)\b", re.IGNORECASE)
WINDOW_DAYS = 30


def uid(login: str | None) -> str:
    return "u_" + hashlib.sha256((login or "ghost").encode()).hexdigest()[:10]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


class GitHubPRConnector:
    name = "github_pr"

    def __init__(self, cache_dir: Path, git_dir: Path, *, family: str = "coding/pull_request", now: str | None = None):
        self.gh = GitHubCache(cache_dir)
        self.git_dir = git_dir
        self.family = family
        self.now = datetime.fromisoformat(now) if now else datetime.now(UTC)
        self._repos: dict[str, GitRepo] = {}

    def repo(self, name: str) -> GitRepo:
        if name not in self._repos:
            self._repos[name] = GitRepo(self.git_dir / f"{name.split('/')[1].lower()}.git", f"https://github.com/{name}.git").ensure()
        return self._repos[name]

    # ------------------------------------------------------------------ discover
    def discover(self, scope: dict[str, Any]) -> Iterator[SourceRef]:
        """scope: {"repo": "owner/name", "prs": [..]} or {"repo", "from_review_pages": 3, "min_comments": 3}."""
        repo = scope["repo"]
        prs = scope.get("prs")
        if prs is None:
            counts: dict[int, int] = {}
            for page in range(1, int(scope.get("from_review_pages", 3)) + 1):
                for cm in self.gh.get_json(f"{API}/repos/{repo}/pulls/comments?per_page=100&sort=created&direction=desc&page={page}"):
                    n = int(cm["pull_request_url"].rsplit("/", 1)[1])
                    counts[n] = counts.get(n, 0) + 1
            prs = sorted(n for n, c in counts.items() if c >= int(scope.get("min_comments", 3)))
        for n in prs:
            yield SourceRef(connector=self.name, key=f"{repo}#{n}", url=f"https://github.com/{repo}/pull/{n}", meta={"repo": repo, "pr": n})

    # ------------------------------------------------------------------ fetch (API, cached)
    def fetch(self, ref: SourceRef) -> RawRecord:
        repo, n = ref.meta["repo"], int(ref.meta["pr"])
        pr = self.gh.get_json(f"{API}/repos/{repo}/pulls/{n}")
        reviews = self.gh.get_json(f"{API}/repos/{repo}/pulls/{n}/reviews?per_page=100")
        comments = self.gh.get_json(f"{API}/repos/{repo}/pulls/{n}/comments?per_page=100")
        checks = self.gh.get_json(f"{API}/repos/{repo}/commits/{pr['head']['sha']}/check-runs?per_page=100")
        return RawRecord(ref=ref, payload={"pr": pr, "reviews": reviews, "comments": comments, "checks": checks}, privacy="public",
                         fetched_at=self.now.isoformat())

    # ------------------------------------------------------------------ normalize (git facts + events)
    def normalize(self, raw: RawRecord) -> Trace:
        p = raw.payload
        pr, repo_name = p["pr"], raw.ref.meta["repo"]
        g = self.repo(repo_name)
        head, base = pr["head"]["sha"], pr["base"]["sha"]
        if not g.has(head):
            g.git("fetch", "-q", "origin", f"+refs/pull/{pr['number']}/head:refs/pull/{pr['number']}/head", timeout=600)
        commits = g.commits(base, head)
        final = {c.sha for c in commits}
        # Rebased / force-pushed PRs: reviewers commented on commits that are no longer in the final list. Recover the
        # PR's commits as of every reviewed push (never on the base branch), so a comment grades the step it was about.
        reviewed = {cm["original_commit_id"] for cm in p["comments"]} | {r.get("commit_id") for r in p["reviews"] if r.get("commit_id")}
        trunk = f"origin/{pr['base']['ref']}" if g.has(f"origin/{pr['base']['ref']}") else pr["base"]["ref"]
        superseded: list[Commit] = []
        for s in sorted(x for x in reviewed if x and x not in final and g.has(x)):
            for sha in g.git("rev-list", "--reverse", "--no-merges", s, f"^{trunk}", f"^{base}", check=False).split()[:200]:
                if sha not in final and sha not in {c.sha for c in superseded}:
                    superseded.append(g.commit(sha))
        by_sha = {c.sha: c for c in commits + superseded}
        merge_sha = pr.get("merge_commit_sha") if pr.get("merged_at") else None
        squash = bool(merge_sha and merge_sha not in by_sha and g.has(merge_sha) and len(g.commit(merge_sha).parents) == 1)

        author = pr["user"]["login"]
        actors: dict[str, TraceActor] = {}

        def actor(login: str | None, role: str, kind: str = "human", user: dict | None = None) -> str:
            a = uid(login)
            if a not in actors:
                if (login or "").endswith("[bot]") and kind == "human":
                    kind = "system"
                if user and user.get("type") == "Bot" and kind == "human":
                    kind = "agent"  # an AI reviewer (e.g. Copilot): its signals are kept apart from human experts' in reliability
                actors[a] = TraceActor(id=a, role=role, kind=kind, handle=login)  # type: ignore[arg-type]
            return a

        actor(author, "author")
        ev: list[dict[str, Any]] = []
        ev.append({"time": pr["created_at"], "kind": "message", "actor": actor(author, "author"), "type": "pr_opened",
                   "text": f"{pr['title']}\n\n{(pr.get('body') or '')[:4000]}", "data": {"number": pr["number"], "base": base, "head": head}})
        for c in commits + superseded:
            ev.append({"time": c.time, "kind": "action", "actor": actor(author, "author"), "type": "commit", "text": f"{c.subject}\n\n{c.body}".strip(),
                       "state_ref": f"git:{c.sha}", "data": {"sha": c.sha, "git_author": c.author, "added": c.added, "superseded": c.sha not in final,
                                                             "apply_suggestion": "suggestion" in c.subject.lower(),
                                                             "writes": [f"file:{path}" for path in c.added]}})
        for r in p["reviews"]:
            if not r.get("submitted_at"):
                continue
            who = r["user"]["login"] if r.get("user") else None
            ev.append({"time": r["submitted_at"], "kind": "message", "actor": actor(who, "author" if who == author else "reviewer", user=r.get("user")), "type": "review",
                       "text": r.get("body") or "", "data": {"state": r["state"], "commit": r.get("commit_id"), "review_id": r["id"]}})
        for cm in p["comments"]:
            ev.append(self._comment_event(g, cm, commits, by_sha, author, actor, head))
        for run in (p["checks"] or {}).get("check_runs", []):
            if run.get("completed_at"):
                ev.append({"time": run["completed_at"], "kind": "check", "actor": actor((run.get("app") or {}).get("slug", "ci") + "[bot]", "ci", "system"),
                           "type": "ci_run", "text": run.get("name"), "state_ref": f"git:{run['head_sha']}",
                           "data": {"name": run.get("name"), "conclusion": run.get("conclusion"), "sha": run["head_sha"]}})
        if pr.get("merged_at"):
            ev.append({"time": pr["merged_at"], "kind": "state", "actor": actor((pr.get("merged_by") or {}).get("login"), "maintainer"), "type": "merged",
                       "state_ref": f"git:{merge_sha}" if merge_sha else None, "data": {"merge_commit": merge_sha, "squash": squash}})
            ev += self._after_merge(g, pr, commits, merge_sha, squash, actor)
        elif pr.get("closed_at"):
            ev.append({"time": pr["closed_at"], "kind": "state", "actor": None, "type": "closed_unmerged", "data": {}})

        ev.sort(key=lambda e: (e["time"] or "", {"pr_opened": 0, "commit": 1}.get(e["type"], 2)))
        events, seq_of = [], {}
        for i, e in enumerate(ev):
            key = e.pop("_id", None)
            reply = e.pop("_reply", None)
            events.append(TraceEvent(seq=i, reply_to=seq_of.get(reply) if reply else None, **e))
            if key:
                seq_of[key] = i
        # resolve commit references to event seqs (attribution facts for M4)
        seq_by_sha = {e.data["sha"]: e.seq for e in events if e.type == "commit"}
        for e in events:
            for k in ("introduced_by", "changed_by", "fixed_by"):
                if e.data.get(k):
                    e.data[k + "_seq"] = seq_by_sha.get(e.data[k])
            if e.type == "post_merge_commit":
                e.data["touches_seq"] = sorted({seq_by_sha[s] for s in e.data.get("touches_pr_commits", []) if s in seq_by_sha})
        ended = pr.get("merged_at") or pr.get("closed_at")
        return Trace(
            trace_id=f"github:{repo_name}#{pr['number']}", family=self.family,
            source=TraceSource(connector=self.name, ref=raw.ref.key, url=pr.get("html_url"), privacy="public", fetched_at=raw.fetched_at),
            actors=list(actors.values()), events=events,
            outcome={"label": "unknown", "details": {"state": pr.get("state"), "merged": bool(pr.get("merged_at")), "ended_at": ended}},  # type: ignore[arg-type]
        )

    def _comment_event(self, g: GitRepo, cm: dict, commits: list[Commit], by_sha: dict[str, Commit], author: str, actor, head: str) -> dict[str, Any]:
        who = cm["user"]["login"] if cm.get("user") else None
        path, line, side = cm["path"], cm.get("original_line"), cm.get("side") or "RIGHT"
        start = cm.get("original_start_line") or line
        oc = cm["original_commit_id"]
        data: dict[str, Any] = {"comment_id": cm["id"], "original_commit": oc, "commit": cm.get("commit_id"), "side": side,
                                "diff_hunk": cm.get("diff_hunk", "")[-1500:], "outdated": cm.get("position") is None, "in_reply_to": cm.get("in_reply_to_id")}
        sug = SUGGESTION.search(cm.get("body") or "") or FIX_BLOCK.search(cm.get("body") or "")
        if sug:
            data["suggestion"] = sug.group(1)
            data["suggestion_kind"] = "block" if sug.re is SUGGESTION else "code"
        if line and side == "RIGHT" and g.has(oc):
            intro = g.blame(oc, path, line)
            data["introduced_by"] = intro if intro in by_sha else None  # None: the line predates the PR (context, not this PR's step)
            final_idx = {c.sha: i for i, c in enumerate(commits)}
            lo, hi = min(start or line, line) - 1, line + 1  # the commented span and one line around it
            if oc in final_idx:  # precise: follow the lines through the commits pushed after the comment's commit
                later = commits[final_idx[oc] + 1:]
                exact, _ = g.track_line(path, line, later) if later else (None, None)
                near = g.track_window(path, lo, hi, later) if later else None
            else:  # the PR was rebased after the comment: compare the reviewed commit with the final head
                exact_hit = any(ol and os_ <= line <= os_ + ol - 1 for os_, ol, _n, _l in g.hunks(oc, head, path))
                near_hit = g.window_changed(oc, head, path, lo, hi)
                after = [c for c in commits if c.time >= cm["created_at"] and path in c.added]
                first = after[0].sha if after else head
                exact, near = (first if exact_hit else None), (first if near_hit else None)
            data["changed_by"] = exact or near
            data["line_changed"] = bool(exact)
            data["near_changed"] = bool(near)
            if sug:
                want = _norm(sug.group(1))
                final_text = _norm("\n".join(g.file_lines(head, path) or []))
                # applied: the suggested text is in the final file, or a deletion suggestion whose line is gone
                span_lines = [_norm(x[1:]) for x in (cm.get("diff_hunk") or "").splitlines()[1:] if x[:1] in "+ "][-(line - (start or line) + 1):]
                gone = all(s not in final_text for s in span_lines if s) and any(span_lines)
                data["suggestion_applied"] = (bool(want) and want in final_text) or (not want and gone)  # a deletion: the span's lines are gone
        return {"time": cm["created_at"], "kind": "message", "actor": actor(who, "author" if who == author else "reviewer", user=cm.get("user")), "type": "review_comment",
                "text": cm.get("body") or "", "target": SignalTarget(app="repo", object=path, span=[start, line] if line else None), "data": data,
                "_id": cm["id"], "_reply": cm.get("in_reply_to_id")}

    def _after_merge(self, g: GitRepo, pr: dict, commits: list[Commit], merge_sha: str | None, squash: bool, actor) -> list[dict[str, Any]]:
        """vcs_history: commits on the base branch within 30 days that revert the PR or change lines it wrote."""
        merged = datetime.fromisoformat(pr["merged_at"])
        until = merged + timedelta(days=WINDOW_DAYS)
        pr_shas = {c.sha for c in commits} | ({merge_sha} if squash and merge_sha else set())
        files = sorted({path for c in commits for path in c.added})
        if not files:
            return []
        out = []
        branch = f"origin/{pr['base']['ref']}" if g.has(f"origin/{pr['base']['ref']}") else pr["base"]["ref"]
        for c in g.log_since(branch, merged.isoformat(), until.isoformat(), files[:200]):
            if c.sha in pr_shas or not c.parents:
                continue
            touched: set[str] = set()
            for path in files:
                for os_, ol, _ns, _nl in g.hunks(c.parents[0], c.sha, path):
                    for ln in range(os_, os_ + ol):
                        b = g.blame(c.parents[0], path, ln)
                        if b in pr_shas:
                            touched.add(b)
            is_revert = bool(REVERT.match(c.subject)) and (f"#{pr['number']}" in c.subject + c.body or (merge_sha or "-")[:7] in c.body
                                                           or _norm(pr["title"]).lower()[:40] in c.subject.lower())
            if not touched and not is_revert:
                continue
            out.append({"time": c.time, "kind": "action", "actor": actor(f"git:{c.author}", "maintainer"), "type": "post_merge_commit",
                        "text": f"{c.subject}\n\n{c.body}".strip(), "state_ref": f"git:{c.sha}",
                        "data": {"sha": c.sha, "is_revert": is_revert, "fixlike": bool(FIXLIKE.search(c.subject + " " + c.body)),
                                 "touches_pr_commits": sorted(touched), "days_after_merge": round((datetime.fromisoformat(c.time) - merged).total_seconds() / 86400, 1)}})
        return out
