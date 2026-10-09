"""Base model for every Crucible schema.

Every persisted or transmitted Crucible object is a ``CrucibleModel``:
- unknown keys are rejected (a typo in a world spec must fail loudly, not be ignored);
- it round-trips losslessly through JSON and YAML;
- its canonical JSON (sorted keys, no whitespace, UTC timestamps) has a stable content hash,
  so two equal objects always get the same id (``crucible.core.ids``).

Versioned top-level documents carry ``schema_version`` (e.g. ``crucible.world/v1``). A reader
refuses a major version it does not know instead of guessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Self

import yaml
from pydantic import BaseModel, ConfigDict


class CrucibleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True)

    def canonical_json(self) -> str:
        """Deterministic JSON: sorted keys, compact separators, defaults included, None dropped."""
        return json.dumps(
            self.model_dump(mode="json", exclude_none=True, by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def content_hash(self) -> str:
        from crucible.core.ids import sha256_text

        return sha256_text(self.canonical_json())

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json", exclude_none=True, by_alias=True),
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        return cls.model_validate(yaml.safe_load(text))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        p = Path(path)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            return cls.from_yaml(text)
        return cls.model_validate_json(text)

    def dump(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in (".yaml", ".yml"):
            p.write_text(self.to_yaml())
        else:
            p.write_text(self.model_dump_json(indent=2, exclude_none=True, by_alias=True) + "\n")
        return p


class VersionedDocument(CrucibleModel):
    """A top-level document with a ``schema_version`` such as ``crucible.world/v1``."""

    SCHEMA: ClassVar[str] = ""
    schema_version: str = ""

    def model_post_init(self, __context: Any) -> None:  # noqa: D401
        if not self.schema_version:
            object.__setattr__(self, "schema_version", self.SCHEMA)
        name, _, version = self.schema_version.partition("/")
        want_name, _, want_version = self.SCHEMA.partition("/")
        if name != want_name or version.split(".")[0] != want_version.split(".")[0]:
            raise ValueError(
                f"{type(self).__name__} reads {self.SCHEMA}; got schema_version={self.schema_version!r}"
            )
