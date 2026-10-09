"""Provenance: every artifact records what it was derived from (ARCHITECTURE §7 rule 4)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import Field

from crucible import __version__
from crucible.core.schema.base import CrucibleModel


class SourceRef(CrucibleModel):
    kind: str  # trace, task, world, run, record, file, commit, signal ...
    id: str
    hash: str | None = None


class Provenance(CrucibleModel):
    created_by: str  # "<module>[:<function>]"
    crucible_version: str = __version__
    created_at: str = Field(default_factory=lambda: utc_now())
    derived_from: list[SourceRef] = Field(default_factory=list)
    params: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    note: str | None = None


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
