"""Crucible toy pack: a small helpdesk world (tickets in SQLite, a knowledge base, a mailbox, a simulated
customer). Exists to keep the core domain-agnostic: every module's gate runs on it, on CPU, in CI."""

from __future__ import annotations

from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[2]


def register(registry) -> None:
    registry.register("pack", "toy", "crucible_pack_toy:PACK_ROOT", source="crucible-pack-toy")
