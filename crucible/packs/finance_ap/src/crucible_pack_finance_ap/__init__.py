"""The finance_ap Crucible pack."""

from __future__ import annotations

from pathlib import Path


def _pack_root() -> Path:
    """The pack's root: the source checkout (``src/<package>/../../pack.yaml``) or, in an installed wheel, the ``_pack/``
    directory the wheel force-includes (pack.yaml, README and data next to the package)."""
    here = Path(__file__).resolve().parent
    return here.parents[1] if (here.parents[1] / "pack.yaml").is_file() else here / "_pack"


PACK_ROOT = _pack_root()


def register(registry) -> None:
    registry.register("pack", "finance_ap", "crucible_pack_finance_ap:PACK_ROOT", source="crucible-pack-finance_ap")
