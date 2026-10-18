"""Typed source records (swarm-pipeline/00 §2): what scouts and miners hand a generator.

| type | what it points at | made by |
|---|---|---|
| ``real_change`` | a commit or release window of a repository (a real change with tests) | commit / release scouts |
| ``real_fix`` | a commit that fixed a bug and added or changed the test that catches it | fix scout; executed at B by ``rebreak`` |
| ``real_perf`` | a commit that made something faster | perf scout; its speedup executed by ``open`` |
| ``host_repo`` | a library other repositories depend on | repository search |
| ``consumer`` | a repository that depends on a host (``evidence.host``, the requirement line) | dependency snowball |
| ``request`` | an issue or feature text | issue search |

Every record carries the filter results it passed (``filters``: licence, size, CPU probe) and the eval lease decision
(``lease``) taken before any filter, so a sealed source is refused before anything else runs on it. ``observation()`` is
the record as a ``cards.ObservationRecord`` (``real_pr_tests`` for an executed commit, else ``reference_derived``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RecordType = Literal["real_change", "real_fix", "real_perf", "host_repo", "consumer", "request"]
RECORD_TYPES: tuple[str, ...] = ("real_change", "real_fix", "real_perf", "host_repo", "consumer", "request")


@dataclass
class SourceRecord:
    type: str
    repo: str  # owner/name on GitHub, or the local clone name of a known repository
    id: str = ""
    url: str = ""
    sha: str = ""
    title: str = ""
    text: str = ""
    licence: str | None = None
    size_kb: int | None = None
    stars: int | None = None
    created: str = ""  # ISO-8601: the commit, issue or repository date
    evidence: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    lease: dict[str, Any] = field(default_factory=dict)
    query: str = ""  # the expanded query that found it
    by: str = "miner:p5-scout"

    def __post_init__(self) -> None:
        if self.type not in RECORD_TYPES:
            raise ValueError(f"source record type is one of {RECORD_TYPES}, got {self.type!r}")
        if not self.id:
            self.id = f"src-{self.type}-{hashlib.sha256(f'{self.repo}|{self.sha}|{self.url}|{self.title}'.encode()).hexdigest()[:12]}"

    @property
    def passed(self) -> bool:
        """The lease granted and every filter run passed."""
        return bool(self.lease.get("ok")) and all(bool(v.get("ok")) for v in self.filters.values() if isinstance(v, Mapping))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> SourceRecord:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def observation(self) -> dict[str, Any]:
        from crucible.factory.cards import ObservationRecord

        executed = bool(self.evidence.get("executed"))
        sha = hashlib.sha256(json.dumps({"repo": self.repo, "sha": self.sha, "url": self.url, "evidence": self.evidence}, sort_keys=True, default=str).encode()).hexdigest()
        created = self.created or "1970-01-01T00:00:00+00:00"
        return ObservationRecord(id=f"obs-{self.id}", type="real_pr_tests" if executed and self.sha else "reference_derived",
                                 pointer=f"git:{self.repo}@{self.sha}" if self.sha else (self.url or f"repo:{self.repo}"), sha256=sha,
                                 created=created, by=self.by).model_dump(mode="json")
