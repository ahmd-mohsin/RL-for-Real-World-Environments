"""The monotone counterexample archive (part2-paper-core §7) and its regression runner.

One append-only JSONL file per domain, ``<root>/<domain>/archive.jsonl`` (``factory/labels.py``'s format, copied since
synth sits below factory): each line is ``{seq, prev, h, rec}`` with ``h = sha256(prev \\n seq \\n canonical(rec))``, so a
changed, removed or reordered line breaks the chain and every read raises ``CorruptArchive``. A torn final line (a crash
mid-append) is ignored by reads and removed by the next append. Appends hold an advisory lock. A counterexample is added
once (its id is a content hash): re-adding is a no-op, never a second line.

``regress(archive, task, ctx, falsifiers)`` replays every counterexample archived for the task (every version) against
``task`` through its class's ``replay`` and reports the ones that **re-opened**. The CEGIS loop refuses a repair with any
re-opened counterexample: no repair may make an archived counterexample pass again.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import Field, ValidationError

from crucible.core import canonical
from crucible.core.ids import sha256_text
from crucible.core.schema.base import CrucibleModel
from crucible.synth.falsify.base import Counterexample, Falsifier, FalsifyContext
from crucible.synth.task import Task

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

GENESIS = "0" * 64
DOMAIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class CorruptArchive(ValueError):
    pass


def _h(prev: str, seq: int, rec: Mapping[str, object]) -> str:
    return sha256_text(f"{prev}\n{seq}\n{canonical.dumps(rec)}")


class Archive:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)

    def path(self, domain: str) -> Path:
        if not DOMAIN.match(domain):
            raise ValueError(f"bad domain name {domain!r}")
        return self.root / domain / "archive.jsonl"

    def _read(self, domain: str) -> tuple[list[Counterexample], str, int, int]:
        """(records, head hash, seq, length of the valid prefix)."""
        p = self.path(domain)
        if not p.exists():
            return [], GENESIS, 0, 0
        data = p.read_bytes()
        end = data.rfind(b"\n") + 1
        prev, seq, out = GENESIS, 0, []
        for n, raw in enumerate(data[:end].split(b"\n")[:-1], 1):
            try:
                line = json.loads(raw)
                rec = line["rec"]
                if line["seq"] != seq + 1 or line["prev"] != prev or line["h"] != _h(prev, seq + 1, rec):
                    raise CorruptArchive(f"{p}: line {n}: the hash chain is broken")
                out.append(Counterexample.model_validate(rec))
            except CorruptArchive:
                raise
            except (ValueError, KeyError, TypeError, ValidationError) as e:
                raise CorruptArchive(f"{p}: line {n}: not a valid record ({type(e).__name__})") from None
            prev, seq = line["h"], seq + 1
        return out, prev, seq, end

    def records(self, domain: str, task_id: str | None = None) -> list[Counterexample]:
        recs = self._read(domain)[0]
        return [r for r in recs if task_id is None or r.task_id == task_id]

    def regressions(self, domain: str) -> list[Counterexample]:
        return self.records(domain)

    def verify(self, domain: str) -> int:
        return self._read(domain)[2]

    def add(self, cx: Counterexample) -> bool:
        return bool(self.add_all([cx]))

    def add_all(self, cxs: Iterable[Counterexample]) -> list[str]:
        """Append the counterexamples not yet archived; returns the ids added."""
        by_domain: dict[str, list[Counterexample]] = {}
        for c in cxs:
            by_domain.setdefault(c.domain, []).append(c)
        added: list[str] = []
        for domain, items in by_domain.items():
            p = self.path(domain)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a+b") as fh:
                if fcntl is not None:
                    fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    recs, prev, seq, end = self._read(domain)
                    if end != os.fstat(fh.fileno()).st_size:
                        fh.truncate(end)  # drop a torn tail
                    have = {r.id for r in recs}
                    lines = []
                    for c in items:
                        if c.id in have:
                            continue
                        have.add(c.id)
                        rec = c.model_dump(mode="json")
                        seq += 1
                        h = _h(prev, seq, rec)
                        lines.append(json.dumps({"seq": seq, "prev": prev, "h": h, "rec": rec}, sort_keys=True, ensure_ascii=False) + "\n")
                        prev = h
                        added.append(c.id)
                    if lines:
                        fh.seek(0, os.SEEK_END)
                        fh.write("".join(lines).encode())
                        fh.flush()
                        os.fsync(fh.fileno())
                finally:
                    if fcntl is not None:
                        fcntl.flock(fh, fcntl.LOCK_UN)
        return added


class RegressionReport(CrucibleModel):
    task_id: str
    task_version: int
    checked: int = 0
    reopened: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)  # no falsifier of that class was given

    @property
    def ok(self) -> bool:
        return not self.reopened


async def regress(archive: Archive, task: Task, ctx: FalsifyContext, falsifiers: Iterable[Falsifier]) -> RegressionReport:
    by_cls = {f.name: f for f in falsifiers}
    rep = RegressionReport(task_id=task.id, task_version=task.version)
    for cx in archive.records(task.domain, task.id):
        f = by_cls.get(cx.cls)
        if f is None:
            rep.skipped.append(cx.id)
            continue
        rep.checked += 1
        if await f.replay(cx, task, ctx):
            rep.reopened.append(cx.id)
    return rep
