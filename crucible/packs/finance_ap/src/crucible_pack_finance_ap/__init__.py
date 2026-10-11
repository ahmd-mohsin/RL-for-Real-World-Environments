"""The finance_ap Crucible pack."""

from __future__ import annotations

from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[2]


def register(registry) -> None:
    registry.register("pack", "finance_ap", "crucible_pack_finance_ap:PACK_ROOT", source="crucible-pack-finance_ap")
