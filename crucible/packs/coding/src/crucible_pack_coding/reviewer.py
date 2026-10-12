"""Grounded reviewer simulator for coding worlds.

It replays recorded review feedback and never invents new findings. A recorded comment is raised
exactly when the code it was attached to (the comment's diff-hunk anchor) is still present in the
reviewed file; the comment text is returned verbatim, located at the matching line, with its record
id as citation. When the anchor is gone (the code was changed), the comment is not raised.

This is what hindsight and trap tasks need: the solver can ask for review, and gets the expert's actual
feedback on the expert's actual concerns, at no LLM cost and with zero fabrication by construction.

M2 audit (2026-10-12): files are read in pages (a single read stopped at 2000 lines and the World's output cap, so
findings past it were missed and the policy heard "no further comments"); an unreadable file is a masking failure,
not a clean review; ``suggestion`` blocks (the expert's replacement code) are withheld; call ids are deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from crucible.actors.base import ActorBase
from crucible.actors.fidelity import Finding
from crucible.core.schema.actions import Action, Message, ToolCall
from crucible.core.schema.trace import Record
from crucible.world.leaks import strip_suggestions
from crucible.world.world import ActorTurn
from crucible_pack_coding.github_reviews import normalize

PAGE = 400  # lines per read_file call; a page the World still had to cut is re-read at half the size
_MORE = re.compile(r"\n\.\.\. \((\d+) lines total; continue with offset=(\d+)\)\s*$")


@dataclass
class ReviewFinding:
    record: Record
    path: str
    line: int

    def as_finding(self) -> Finding:
        return Finding(self.path, self.line, self.record.id)


def locate(anchor: list[str], content: str, *, near: int | None = None) -> int | None:
    """1-based line of the anchor's last line where it occurs as consecutive (normalized) lines.

    Repeated code makes anchors ambiguous; the reviewer knows where it commented, so with ``near``
    (the recorded line) the closest occurrence wins, otherwise the first.
    """
    lines = [normalize(x) for x in content.splitlines()]
    if not anchor or not any(anchor):
        return None
    n = len(anchor)
    hits = [i + n for i in range(len(lines) - n + 1) if lines[i : i + n] == anchor]
    if not hits:
        return None
    if near:
        return min(hits, key=lambda h: (abs(h - near), h))
    return hits[0]


def review_files(records: list[Record], files: dict[str, str]) -> list[ReviewFinding]:
    out = []
    for r in records:
        if r.target is None or r.target.object not in files:
            continue
        anchor = (r.meta.get("anchor") or "").split("\n")
        near = r.target.span[1] if r.target.span else None
        line = locate(anchor, files[r.target.object], near=near)
        if line is not None:
            out.append(ReviewFinding(r, r.target.object, line))
    return sorted(out, key=lambda f: (f.path, f.line, f.record.id))


class ReviewerActor(ActorBase):
    """World actor: on any message, reads the changed files of ``repo_app`` and replies with recorded findings."""

    def __init__(self, actor_id: str, records: list[Record], *, repo_app: str, reply_latency: str | float | None = "2h"):
        super().__init__(actor_id, reply_latency=reply_latency)
        self.records = records
        self.repo_app = repo_app
        self._pending: dict[str, Any] = {}
        self._calls = 0

    def _read(self, path: str, offset: int, limit: int) -> ToolCall:
        self._calls += 1
        cid = f"rv-{self._calls}"
        self._pending["waiting"][cid] = [path, offset, limit]
        return ToolCall(tool_call_id=cid, function_name=f"{self.repo_app}.read_file", arguments={"path": path, "offset": offset, "limit": limit})

    async def respond(self, turn: ActorTurn) -> list[Action]:
        calls: list[Action] = []
        if turn.inbox:
            self._pending = {"to": turn.inbox[-1].frm, "pages": {}, "waiting": {}, "failed": []}
            for p in sorted({r.target.object for r in self.records if r.target and r.target.object}):
                calls.append(self._read(p, 1, PAGE))
            if calls:
                return calls
        for tr in turn.tool_results:
            req = self._pending.get("waiting", {}).pop(tr.tool_call_id, None)
            if req is None:
                continue
            path, offset, limit = req
            if not tr.ok:
                self._pending["failed"].append(f"{path}: {tr.error or tr.content[:200]}")
            elif tr.truncated:  # the World cut this page: read it again in halves
                if limit <= 25:
                    self._pending["failed"].append(f"{path}: lines from {offset} are too long to read")
                else:
                    calls += [self._read(path, offset, limit // 2)]
            else:
                text, more = tr.content, _MORE.search(tr.content)
                if more:
                    text = text[: more.start()]
                    calls.append(self._read(path, int(more.group(2)), limit))
                # read_file returns "<n>\t<line>"; strip the numbering
                self._pending["pages"].setdefault(path, {})[offset] = [line.split("\t", 1)[1] if "\t" in line else "" for line in text.splitlines()]
        if calls:
            return calls
        if self._pending and not self._pending.get("waiting"):
            to, failed = self._pending["to"], self._pending["failed"]
            files = {p: "\n".join(ln for o in sorted(pages) for ln in pages[o]) for p, pages in self._pending["pages"].items()}
            self._pending = {}
            if failed:  # a review of files it could not read is not "no comments": the episode is not evidence
                self.failures.append({"kind": "crucible:sim_failure", "reason": "reviewer could not read " + "; ".join(failed)[:280], "at": turn.time})
                return []
            findings = review_files(self.records, files)
            self.history.append({"findings": [f.record.id for f in findings], "at": turn.time})
            if not findings:
                return [Message(to=to, text="I have no further comments based on my notes.")]
            body = "\n\n".join(f"{f.path}:{f.line}: {strip_suggestions(f.record.text)}" for f in findings)
            return [Message(to=to, text=body, citations=[f.record.id for f in findings])]
        return []
