"""The local store passes the store conformance kit; no reader outside core uses an unverified blob path (M0 audit
2026-10-14 R-22)."""

from __future__ import annotations

import re
from pathlib import Path

from crucible.core.schema.reward import RewardJson
from crucible.core.store import Store
from crucible.core.store.conformance import check
from crucible.core.store.protocols import BlobStore, DocIndex

SRC = Path(__file__).resolve().parents[2] / "src" / "crucible"


def _corrupt(store: Store, key: str) -> None:
    p = store.blob_path(key)
    p.chmod(0o644)
    p.write_bytes(b"tampered")


def test_the_local_store_passes_the_conformance_kit(tmp_path):
    check(Store, tmp_path, corrupt=_corrupt, doc=RewardJson(reward=0.5))
    s = Store(tmp_path / "s")
    assert isinstance(s, BlobStore) and isinstance(s, DocIndex) and s.backend == "local"


def test_no_reader_outside_core_uses_an_unverified_blob_path():
    """The audit found four readers (World state pushes, custody) reading blob files without the hash check."""
    hits = [f"{p.relative_to(SRC)}:{i}" for p in SRC.rglob("*.py") if "core/store" not in p.as_posix()
            for i, line in enumerate(p.read_text().splitlines(), 1) if re.search(r"(?<![\w_])\.blob_path\(", line)]
    assert hits == []
