"""A directory the store creates is synced into its parent (M0 audit 2026-10-14 R-25)."""

from __future__ import annotations

import crucible.core.store as store_mod
from crucible.core.store import Store


def test_a_new_blob_directory_is_synced_into_blobs(tmp_path, monkeypatch):
    synced: list[str] = []
    real = store_mod._fsync_dir
    monkeypatch.setattr(store_mod, "_fsync_dir", lambda p: (synced.append(str(p)), real(p)))
    s = Store(tmp_path / "s")
    assert str(tmp_path) in synced  # the store root's own entry
    synced.clear()
    key = s.put_bytes(b"first")
    blobs = tmp_path / "s" / "blobs"
    shard = blobs / key.split(":")[1][:2]
    assert str(blobs) in synced and str(shard) in synced  # before, only the shard was synced: its entry in blobs/ was not
    synced.clear()
    s.put_bytes(b"first")  # the shard exists: nothing new to sync into blobs/
    assert str(blobs) not in synced
