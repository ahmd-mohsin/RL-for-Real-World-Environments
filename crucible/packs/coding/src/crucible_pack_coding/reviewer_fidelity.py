"""Reviewer-simulator fidelity on real review history (ARCHITECTURE M2 exit gate, coding half).

For every human, top-level review comment on a code file in the sampled repos:
- **before** = the file at the commit the reviewer commented on (``original_commit``);
- **after**  = the file at the PR's final head (``refs/pull/N/head``).

Metrics:
- recall@before — the simulator raises the recorded comment on the code the human actually reviewed;
- location precision@before — raised comments land within ``tolerance`` lines of where the human put
  them (the gate: >= 0.8);
- silence@after-changed — when the commented code no longer exists at the PR head, the simulator stays
  silent (its concern was acted on);
- resolution rate — share of comments whose anchored code changed by the PR head (how often review
  feedback led to a code change in this history).
Run: ``python -m crucible_pack_coding.reviewer_fidelity [out.json]``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from crucible.actors.fidelity import Finding, location_metrics
from crucible_pack_coding.github_reviews import GitHubCache, fetch_review_comments, pr_heads
from crucible_pack_coding.reviewer import review_files

REPOS = ["PointCloudLibrary/pcl", "NVIDIA/cccl"]


def evaluate(repos: list[str], cache: GitHubCache, *, pages: int = 3, tolerance: int = 3) -> dict:
    per_repo = {}
    all_pred: list[Finding] = []
    all_gold: list[Finding] = []
    totals = Counter()
    examples = []
    for repo in repos:
        comments = fetch_review_comments(cache, repo, pages=pages)
        heads = pr_heads(repo)
        pred: list[Finding] = []
        gold: list[Finding] = []
        c_repo = Counter()
        for c in comments:
            before = cache.get_raw(repo, c.original_commit, c.path)
            if before is None:
                c_repo["before_missing"] += 1
                continue
            rec = c.to_record()
            gold.append(Finding(c.path, c.original_line or 0, rec.id))
            found = review_files([rec], {c.path: before})
            pred.extend(Finding(f.path, f.line, f.record.id) for f in found)
            c_repo["evaluated"] += 1
            c_repo["raised_before"] += bool(found)
            head = heads.get(c.pr)
            after = cache.get_raw(repo, head, c.path) if head else None
            if after is None:
                c_repo["after_missing_or_deleted"] += 1
                continue
            still = review_files([rec], {c.path: after})
            if still:
                c_repo["anchor_unchanged_at_head"] += 1
            else:
                c_repo["anchor_changed_at_head"] += 1
                c_repo["silent_after_changed"] += 1
            if len(examples) < 6 and found:
                examples.append({"repo": repo, "pr": c.pr, "path": c.path, "line": c.original_line, "raised_at": found[0].line,
                                 "comment": c.body[:200], "changed_at_head": not still})
        m = location_metrics(pred, gold, tolerance=tolerance)
        changed = c_repo["anchor_changed_at_head"]
        per_repo[repo] = {**dict(c_repo), **m, "comments": len(comments),
                          "resolution_rate": round(changed / max(changed + c_repo["anchor_unchanged_at_head"], 1), 4)}
        all_pred += pred
        all_gold += gold
        totals.update(c_repo)
    overall = location_metrics(all_pred, all_gold, tolerance=tolerance)
    changed = totals["anchor_changed_at_head"]
    return {
        "repos": per_repo,
        "overall": {**overall, **dict(totals), "silence_after_changed": round(totals["silent_after_changed"] / max(changed, 1), 4),
                    "resolution_rate": round(changed / max(changed + totals["anchor_unchanged_at_head"], 1), 4)},
        "tolerance_lines": tolerance,
        "api_calls": cache.api_calls,
        "raw_calls": cache.raw_calls,
        "examples": examples,
    }


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    cache = GitHubCache(Path("data/cache/github"))
    res = evaluate(REPOS, cache)
    print(json.dumps(res["overall"], indent=1))
    for r, v in res["repos"].items():
        print(r, {k: v[k] for k in ("comments", "evaluated", "precision", "recall", "resolution_rate")})
    if out:
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
