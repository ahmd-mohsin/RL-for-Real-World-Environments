"""Record store for grounded actors: BM25 retrieval over what people actually wrote.

Two rules make it safe for hindsight tasks:
- **time**: a record is visible only if its time <= the world's virtual "now" (an actor in a replayed
  2024 workflow must not know what was written in 2025 — that would leak the expert's future fixes);
- **scope**: retrieval can be restricted to sources (traces) and roles (e.g. only the reviewer's own
  past comments).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from crucible.core.schema.trace import Record, Trace, records_from_trace

_WORD = re.compile(r"[A-Za-z0-9_]+(?:[.\-][A-Za-z0-9_]+)*|[一-鿿]")
STOP = frozenset(
    "a an and are as at be been but by can could did do does for from had has have he her his i if in into is it its "
    "may me my no not of on or our she so than that the their them then there these they this to too up us was we were "
    "what when where which who will with would you your yes ok okay please thanks thank hi hello".split()
)


@lru_cache(maxsize=65536)
def _ts(text: str) -> datetime:
    from crucible.world.clock import parse_time

    return parse_time(text)


def tokens(text: str) -> list[str]:
    out: list[str] = []
    for m in _WORD.finditer(text.lower()):
        t = m.group(0)
        out.append(t)
    # CJK bigrams so Chinese records retrieve on words, not single characters
    cjk = [t for t in out if len(t) == 1 and "一" <= t <= "鿿"]
    out.extend(a + b for a, b in zip(cjk, cjk[1:], strict=False))
    return out


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOP and len(t) > 1 or ("一" <= t[:1] <= "鿿")]


class RecordStore:
    def __init__(self, records: Iterable[Record], *, k1: float = 1.5, b: float = 0.75):
        self.records = list(records)
        self.by_id = {r.id: r for r in self.records}
        if len(self.by_id) != len(self.records):
            raise ValueError("duplicate record ids")
        # What a person knows about their own note includes what it was about (e.g. the file path).
        self._tf = [Counter(content_tokens(r.text + (" " + r.target.object if r.target and r.target.object else ""))) for r in self.records]
        self._len = [sum(c.values()) for c in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 1.0
        df: Counter[str] = Counter()
        for c in self._tf:
            df.update(c.keys())
        n = max(len(self.records), 1)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.k1, self.b = k1, b

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _visible(r: Record, now: str | None) -> bool:
        # Compare instants, never strings: "…56Z" > "…56.5Z" lexically although it is earlier.
        return now is None or r.time is None or _ts(r.time) <= _ts(now)

    def get(self, rid: str, *, now: str | None = None) -> Record | None:
        r = self.by_id.get(rid)
        return r if r is not None and self._visible(r, now) else None

    def visible(self, *, now: str | None = None, sources: set[str] | None = None, roles: set[str] | None = None) -> list[Record]:
        return [r for r in self.records if self._visible(r, now) and (not sources or r.source in sources) and (not roles or r.role in roles)]

    def search(self, query: str, *, k: int = 6, now: str | None = None, sources: set[str] | None = None, roles: set[str] | None = None,
               tags: set[str] | None = None) -> list[tuple[Record, float]]:
        q = Counter(content_tokens(query))
        scored: list[tuple[Record, float]] = []
        for i, r in enumerate(self.records):
            if not self._visible(r, now) or (sources and r.source not in sources) or (roles and r.role not in roles) or (tags and not tags & set(r.tags)):
                continue
            tf, dl = self._tf[i], self._len[i]
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self._idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avg))
            if s > 0:
                scored.append((r, s))
        scored.sort(key=lambda x: (-x[1], x[0].id))
        return scored[:k]

    @classmethod
    def from_traces(cls, traces: Iterable[Trace], types: set[str] | None = None) -> RecordStore:
        return cls(r for t in traces for r in records_from_trace(t, types))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> RecordStore:
        return cls(Record.model_validate(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip())

    def to_jsonl(self, path: str | Path) -> None:
        Path(path).write_text("".join(r.model_dump_json(exclude_none=True) + "\n" for r in self.records))
