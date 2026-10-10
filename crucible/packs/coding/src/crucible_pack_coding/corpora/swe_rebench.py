"""nebius/SWE-rebench-OpenHands-Trajectories + nebius/SWE-rebench (CC-BY-4.0): 67k OpenHands runs of
Qwen3-Coder-480B on 6.3k real GitHub issues (~10 attempts each, 48% resolved), joined with each issue's gold patch,
base commit, FAIL_TO_PASS tests and creation date.

Used for the M5 replay test (re-apply each run's editor calls to the base files, compare with its recorded
``model_patch``) and the M6 S2 gate (does Φ at the midpoint predict ``resolved``, better than an LLM judge, on
issues created after the ones milestones were fitted on). Several attempts per issue let the analysis control for
task difficulty. Files are cached under ``data/cache/hf/swe-rebench`` (``crucible[analysis]`` reads parquet).

Base files are fetched from ``raw.githubusercontent.com/<repo>/<base_commit>/<path>`` (a CDN, not the GitHub API)
and cached on disk; a 404 means the file did not exist at the base commit.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from crucible.connectors.chat_messages import messages_to_trace
from crucible.core.schema.trace import Trace, TraceOutcome, TraceSource
from crucible_pack_coding import PACK_ROOT

DATA = PACK_ROOT.parent.parent / "data" / "cache" / "hf" / "swe-rebench"
RAW = PACK_ROOT.parent.parent / "data" / "cache" / "raw-github"
FAMILY = "coding/swe_issue"
# Files the OpenHands harness writes itself (issue.md is in the workspace listing before step 0; .openhands/ is its task
# tracker): not agent actions, so the replay test ignores them.
def harness_file(path: str) -> bool:
    return path == "issue.md" or path.startswith(".openhands/")


INSTANCE_FILES = ("instances-filtered.parquet", "instances-test-0.parquet", "instances-test-1.parquet")
INSTANCE_COLS = ["instance_id", "repo", "base_commit", "created_at", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "problem_statement"]


def instances(cols: list[str] | None = None) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    out: dict[str, dict[str, Any]] = {}
    for f in INSTANCE_FILES:
        for r in pq.read_table(DATA / f, columns=cols or INSTANCE_COLS).to_pylist():
            out.setdefault(r["instance_id"], r)
    return out


def index() -> list[dict[str, Any]]:
    """Every trajectory's metadata (no transcripts): trajectory_id, instance_id, repo, resolved, exit_status, row."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(DATA / "trajectories.parquet")
    rows, n = [], 0
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=["trajectory_id", "instance_id", "repo", "resolved", "exit_status"]).to_pylist():
            rows.append({**r, "row": n, "rg": rg})
            n += 1
    return rows


def load(rows: list[dict[str, Any]], columns: tuple[str, ...] = ("trajectory_id", "trajectory", "model_patch", "resolved")) -> Iterator[dict[str, Any]]:
    """Full records for the given index rows, reading only the row groups they live in."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(DATA / "trajectories.parquet")
    by_rg: dict[int, set[str]] = {}
    for r in rows:
        by_rg.setdefault(r["rg"], set()).add(r["trajectory_id"])
    for rg, ids in sorted(by_rg.items()):
        table = pf.read_row_group(rg, columns=list(columns))
        for rec in table.to_pylist():
            if rec["trajectory_id"] in ids:
                yield rec


def workspace_root(record: dict[str, Any]) -> str | None:
    for m in record["trajectory"][:3]:
        found = re.search(r"<uploaded_files>\s*(/workspace/[^\s<]+)", m.get("content") or "")
        if found:
            return found.group(1).rstrip("/") + "/"
    return None


def path_mapper(root: str | None):
    def path_of(p: str) -> str | None:
        if root and p.startswith(root):
            return p[len(root):]
        if p.startswith("/workspace/"):
            return p.split("/", 3)[3] if p.count("/") >= 3 else None
        return None  # scratch files outside the repo (/tmp) are not part of the state

    return path_of


def raw_reader(repo: str, sha: str, *, offline: bool = False):
    """path -> content at <repo>@<sha> (None = absent), cached under data/cache/raw-github."""
    cache = RAW / repo.replace("/", "__") / sha

    def read(path: str) -> str | None:
        if not path or path.endswith("/"):
            return None
        f = cache / path
        miss = cache / (path + ".__missing__")
        if f.is_dir():  # a directory path (rm -r dir): not a file at the base commit
            return None
        if f.exists():
            return f.read_text(errors="surrogateescape")
        if miss.exists():
            return None
        if offline:
            raise FileNotFoundError(f"{repo}@{sha}:{path} not cached")
        url = f"https://raw.githubusercontent.com/{repo}/{sha}/{urllib.request.quote(path)}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "crucible-m5"}), timeout=60) as r:
                body = r.read().decode("utf-8", errors="surrogateescape")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                miss.parent.mkdir(parents=True, exist_ok=True)
                miss.write_text("")
                return None
            raise
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, errors="surrogateescape")
        return body

    return read


def to_trace(record: dict[str, Any], inst: dict[str, Any] | None = None) -> Trace:
    resolved = record.get("resolved")
    label = "good" if resolved == 1 else ("bad" if resolved == 0 else "unknown")
    outcome = TraceOutcome(label=label, label_source="SWE-rebench evaluation: resolved" if label != "unknown" else None,  # type: ignore[arg-type]
                           details={"resolved": resolved, "instance_id": (inst or {}).get("instance_id"), "created_at": str((inst or {}).get("created_at"))})
    messages = []
    for m in record["trajectory"]:
        mm = {k: m.get(k) for k in ("role", "content", "tool_calls", "tool_call_id", "name") if m.get(k) is not None}
        messages.append(mm)
    return messages_to_trace(messages, trace_id=f"swe-rebench:{record['trajectory_id']}", family=FAMILY,
                             source=TraceSource(connector="swe_rebench", ref=record["trajectory_id"], privacy="public"), outcome=outcome)


def json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
