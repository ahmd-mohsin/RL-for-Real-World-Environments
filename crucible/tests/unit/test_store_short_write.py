"""A blob write is complete or fails (M0 audit 2026-10-16 R-3): ``os.write`` may write fewer bytes than asked, and the
store dropped the count, so a full disk stored half a blob under the whole one's key and every later read failed."""

from __future__ import annotations

import os
import resource
import signal

import pytest

from crucible.core.schema.base import CrucibleModel
from crucible.core.store import Store, StoreError

LIMIT = 1 << 20


class Doc(CrucibleModel):
    text: str


def test_a_short_os_write_still_stores_every_byte(tmp_path, monkeypatch):
    real = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: real(fd, bytes(data[: max(1, len(data) // 2)])))
    st = Store(tmp_path / "s")
    data = os.urandom(300_000)
    assert st.get_bytes(st.put_bytes(data)) == data
    src = tmp_path / "f.bin"
    src.write_bytes(data)
    assert st.get_bytes(st.put_file(src)) == data


@pytest.mark.skipif(not hasattr(signal, "SIGXFSZ"), reason="needs RLIMIT_FSIZE")
def test_a_write_cut_short_by_the_disk_leaves_no_blob_and_no_row(tmp_path):
    """A file-size limit makes write(2) return a partial count, then fail: the probe stand-in for a full disk."""
    st = Store(tmp_path / "s")
    src = tmp_path / "big.bin"
    src.write_bytes(b"x" * (2 * LIMIT))
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    old = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (LIMIT, hard))
    try:
        for put in (lambda: st.put_bytes(b"y" * (2 * LIMIT)), lambda: st.put_file(src),
                    lambda: st.put_doc("thing", Doc(text="z" * (2 * LIMIT)), name="n")):
            with pytest.raises((OSError, StoreError)):
                put()
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, old)
    assert not [p for p in (st.root / "blobs").rglob("*") if p.is_file()]
    assert st.list_docs("thing") == [] and not list((st.root / "staging").glob("*"))
