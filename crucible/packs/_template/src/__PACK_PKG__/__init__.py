"""The __PACK_NAME__ Crucible pack."""

from __future__ import annotations

from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[2]


def register(registry) -> None:
    registry.register("pack", "__PACK_NAME__", "__PACK_PKG__:PACK_ROOT", source="crucible-pack-__PACK_NAME__")
