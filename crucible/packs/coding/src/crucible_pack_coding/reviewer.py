"""Grounded reviewer simulator for coding worlds.

It replays recorded review feedback and never invents new findings. A recorded comment is raised
exactly when the code it was attached to (the comment's diff-hunk anchor) is still present in the
reviewed file; the comment text is returned verbatim, located at the matching line, with its record
id as citation. When the anchor is gone (the code was changed), the comment is not raised.

This is what hindsight and trap tasks need: the solver can ask for review, and gets the expert's actual
feedback on the expert's actual concerns, at no LLM cost and with zero fabrication by construction.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from crucible.actors.base import ActorBase
from crucible.actors.fidelity import Finding
from crucible.core.schema.actions import Action, Message, ToolCall
from crucible.core.schema.trace import Record
from crucible.world.world import ActorTurn
from crucible_pack_coding.github_reviews import normalize


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

    async def respond(self, turn: ActorTurn) -> list[Action]:
        if turn.inbox:
            self._pending = {"to": turn.inbox[-1].frm, "files": {}, "waiting": set()}
            paths = sorted({r.target.object for r in self.records if r.target and r.target.object})
            calls = []
            for p in paths:
                cid = f"rv-{uuid.uuid4().hex[:8]}"
                self._pending["waiting"].add(cid)
                self._pending[cid] = p
                calls.append(ToolCall(tool_call_id=cid, function_name=f"{self.repo_app}.read_file", arguments={"path": p, "limit": 100000}))
            if calls:
                return calls
        for tr in turn.tool_results:
            p = self._pending.get(tr.tool_call_id)
            if p and tr.ok:
                # read_file returns "<n>\t<line>"; strip the numbering
                self._pending["files"][p] = "\n".join(line.split("\t", 1)[1] if "\t" in line else "" for line in tr.content.splitlines())
            self._pending.get("waiting", set()).discard(tr.tool_call_id)
        if self._pending and not self._pending.get("waiting"):
            findings = review_files(self.records, self._pending["files"])
            to = self._pending["to"]
            self._pending = {}
            self.history.append({"findings": [f.record.id for f in findings], "at": turn.time})
            if not findings:
                return [Message(to=to, text="I have no further comments based on my notes.")]
            body = "\n\n".join(f"{f.path}:{f.line}: {f.record.text}" for f in findings)
            return [Message(to=to, text=body, citations=[f.record.id for f in findings])]
        return []
