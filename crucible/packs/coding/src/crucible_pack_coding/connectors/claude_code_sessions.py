"""Claude Code session transcripts -> traces (coding pack ``claude_code_sessions`` connector, M3).

Our task 01-04 construction transcripts are Claude Code sessions: ``~/.claude/projects/<cwd-slug>/<id>.jsonl``
plus ``<id>/subagents/*.jsonl``. Harbor's Claude Code agent converts exactly that layout to ATIF (DEPEND); this
connector freezes a copy first (a live session keeps growing: ingesting the copy is idempotent, ingesting the
live file is a new version each time), converts it, and tags every event with the migration tasks it touched
(``data.tasks``, from ``migration_tasks/<task>/`` paths in tool calls and results) so M5 can cut the transcript
into one segment per task. Transcripts are ``internal``: ingest redacts them like any other trace.

M3 audit (2026-10-12): a frozen copy is refreshed when the live session has grown (it was frozen once and stale
forever); the frozen directory is owner-only (raw transcripts hold secrets until ingest redacts them); without
``freeze`` each session converts from a view of its own files (Harbor reads every ``*.jsonl`` in the directory, so
every trace held every session of the project).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from crucible.connectors.base import RawRecord, SourceRef
from crucible.core.schema.trace import Trace, TraceSource
from crucible.interop.harbor.atif_import import convert_claude_code_sessions, trajectory_to_trace

FAMILY = "coding/construction_session"
TASK = re.compile(r"migration_tasks/(task\d+[A-Za-z0-9_]*)")


class ClaudeCodeSessionConnector:
    name = "claude_code_sessions"

    def __init__(self, frozen_dir: Path, tasks: set[str] | None = None):
        self.frozen_dir = Path(frozen_dir)
        if tasks is None:
            from crucible_pack_coding.seeds import SEEDS

            tasks = set(SEEDS)
        self.tasks = tasks  # only whole task names count (truncated text yields fragments like "task01_pri")

    def discover(self, scope: dict[str, Any]) -> Iterator[SourceRef]:
        """scope: {"project_dir": ~/.claude/projects/<slug>, "freeze": True}. One ref per session id."""
        project = Path(scope["project_dir"]).expanduser()
        for f in sorted(project.glob("*.jsonl")):
            sid = f.stem
            dest = self.frozen_dir / sid
            frozen = dest / f.name
            if scope.get("freeze", True) and (not frozen.exists() or f.stat().st_size != frozen.stat().st_size):
                self.freeze(project, sid)
            yield SourceRef(connector=self.name, key=sid, meta={"dir": str(dest if dest.exists() else project), "session": sid})

    def freeze(self, project: Path, sid: str) -> Path:
        self.frozen_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.frozen_dir, 0o700)
        dest = self.frozen_dir / sid
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / f"{sid}.jsonl", dest / f"{sid}.jsonl")
        sub = project / sid / "subagents"
        if sub.is_dir():
            shutil.copytree(sub, dest / "subagents", dirs_exist_ok=True)
        return dest

    def fetch(self, ref: SourceRef) -> RawRecord:
        return RawRecord(ref=ref, payload={"dir": ref.meta["dir"]}, privacy="internal")

    def normalize(self, raw: RawRecord) -> Trace:
        d, sid = Path(raw.payload["dir"]), raw.ref.meta.get("session") or raw.ref.key
        if any(p.stem != sid for p in d.glob("*.jsonl")):  # a project directory: convert a view of this session only
            with tempfile.TemporaryDirectory(prefix="crucible-cc-view-") as view:
                os.symlink(d / f"{sid}.jsonl", Path(view) / f"{sid}.jsonl")
                if (d / sid / "subagents").is_dir():
                    os.symlink(d / sid / "subagents", Path(view) / "subagents")
                traj = convert_claude_code_sessions(Path(view))
        else:
            traj = convert_claude_code_sessions(d)
        if traj is None:
            raise ValueError(f"no Claude Code session in {raw.payload['dir']}")
        trace = trajectory_to_trace(traj, trace_id=f"claude_code:{raw.ref.key}", family=FAMILY,
                                    source=TraceSource(connector=self.name, ref=raw.ref.key, privacy="internal"))
        counts: Counter[str] = Counter()
        events = []
        for e in trace.events:
            tasks = sorted(set(TASK.findall((e.text or "") + " " + str(e.data.get("arguments", "")))) & self.tasks)
            if tasks:
                counts.update(tasks)
                e = e.model_copy(update={"data": {**e.data, "tasks": tasks}})
            events.append(e)
        outcome = trace.outcome.model_copy(update={"details": {**trace.outcome.details, "events_per_task": dict(sorted(counts.items()))}})
        return trace.model_copy(update={"events": events, "outcome": outcome})
