"""Base model for every Crucible schema.

Every persisted or transmitted Crucible object is a ``CrucibleModel``:
- unknown keys are rejected (a typo in a world spec must fail loudly, not be ignored), and so is a key given twice
  in a file (``load`` and ``from_yaml`` read through ``crucible.core.serialize``);
- it round-trips losslessly through JSON and YAML;
- its canonical JSON (sorted keys, no whitespace, UTC timestamps) has a stable content hash,
  so two equal objects always get the same id (``crucible.core.ids``).

Versioned top-level documents carry ``schema_version`` (e.g. ``crucible.world/v1``). A reader
refuses a major version it does not know instead of guessing, explains a newer minor version instead of failing on
its unknown keys, and upgrades an older major version through registered upcasters (``register_upcaster``), as Harbor
migrates task configs in ``mode="before"`` validators.

Floats must be finite (``allow_inf_nan=False``); canonical JSON is ``crucible.core.canonical.dumps``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Self

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from crucible.core.serialize import load_json, load_yaml


class CrucibleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True, allow_inf_nan=False)

    def canonical_json(self) -> str:
        """Deterministic JSON: sorted keys, compact separators, defaults included, None dropped, floats finite with
        ``-0.0`` folded, strings NFC (``crucible.core.canonical``)."""
        from crucible.core.canonical import dumps

        return dumps(self.model_dump(mode="json", exclude_none=True, by_alias=True))

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
    def from_yaml(cls, text: str, *, source: str = "<yaml>") -> Self:
        return cls.model_validate(load_yaml(text, source=source))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        p = Path(path)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            return cls.from_yaml(text, source=str(p))
        load_json(text, source=str(p))  # duplicate keys and NaN refused; pydantic's JSON mode below would keep the last
        return cls.model_validate_json(text)

    def dump(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in (".yaml", ".yml"):
            p.write_text(self.to_yaml())
        else:
            p.write_text(self.model_dump_json(indent=2, exclude_none=True, by_alias=True) + "\n")
        return p


# (schema name, major version it upgrades from) -> function from that version's dict to the next major's dict
_UPCASTERS: dict[tuple[str, int], Callable[[dict[str, Any]], dict[str, Any]]] = {}


def register_upcaster(schema: str, from_major: int, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    """Register the migration of ``<schema>/v<from_major>`` documents to ``v<from_major + 1>``."""
    key = (schema, from_major)
    if key in _UPCASTERS:
        raise ValueError(f"an upcaster for {schema}/v{from_major} is already registered")
    _UPCASTERS[key] = fn


def _parse_version(text: str) -> tuple[str, int, int]:
    name, _, version = text.partition("/")
    if not version.startswith("v"):
        raise ValueError(f"schema_version {text!r} must look like '<name>/v<major>[.<minor>]'")
    major, _, minor = version[1:].partition(".")
    try:
        return name, int(major), int(minor or 0)
    except ValueError:
        raise ValueError(f"schema_version {text!r} must look like '<name>/v<major>[.<minor>]'") from None


class VersionedDocument(CrucibleModel):
    """A top-level document with a ``schema_version`` such as ``crucible.world/v1``.

    Reading: a missing version means the reader's own (documents written before versioning); another schema name or a
    newer major version is refused; a newer minor version is refused with an explicit message; an older major version
    is upgraded through the registered upcasters, one major step at a time.
    """

    SCHEMA: ClassVar[str] = ""
    schema_version: str = ""

    @model_validator(mode="before")
    @classmethod
    def _migrate(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not cls.SCHEMA:
            return data
        got = data.get("schema_version") or cls.SCHEMA
        name, major, minor = _parse_version(got)
        want_name, want_major, want_minor = _parse_version(cls.SCHEMA)
        if name != want_name:
            raise ValueError(f"{cls.__name__} reads {cls.SCHEMA}; got schema_version={got!r}")
        if major > want_major:
            raise ValueError(f"{cls.__name__} reads {cls.SCHEMA}; {got!r} is a newer major version")
        if major == want_major and minor > want_minor:
            raise ValueError(f"{cls.__name__} reads {cls.SCHEMA}; {got!r} was written by a newer minor version")
        data = dict(data)
        while major < want_major:
            fn = _UPCASTERS.get((name, major))
            if fn is None:
                raise ValueError(f"no upcaster from {name}/v{major} to v{major + 1}")
            data = fn(data)
            major += 1
            data["schema_version"] = f"{name}/v{major}"
        data.setdefault("schema_version", cls.SCHEMA)
        return data

    @field_validator("schema_version")
    @classmethod
    def _same_major(cls, v: str) -> str:
        if not cls.SCHEMA or not v:
            return v
        name, major, _ = _parse_version(v)
        want_name, want_major, _ = _parse_version(cls.SCHEMA)
        if name != want_name or major != want_major:
            raise ValueError(f"{cls.__name__} reads {cls.SCHEMA}; got schema_version={v!r}")
        return v

    def model_post_init(self, context: Any, /) -> None:
        if not self.schema_version:
            object.__setattr__(self, "schema_version", self.SCHEMA)
