"""Forgetting a document and indexing one are one transaction against each other (M0 audit 2026-10-14 R-24)."""

from __future__ import annotations

import threading
import time

import crucible.core.store as store_mod
from crucible.core.schema.reward import RewardJson
from crucible.core.store import Store, StoreError

DOC = RewardJson(reward=0.25)


def _during_forget(monkeypatch, action):
    """Run ``action`` in another thread while ``forget_doc`` holds its transaction, right after the blob is set aside:
    the moment the audit's race used (a put or tag between the decision and the unlink)."""
    real = store_mod.os.replace
    started: list[threading.Thread] = []

    def replace(src, dst):
        real(src, dst)
        if str(dst).rsplit("/", 1)[-1].startswith("forgotten-") and not started:
            t = threading.Thread(target=action)
            started.append(t)
            t.start()
            time.sleep(0.2)  # the other client is now waiting for the write lock

    monkeypatch.setattr(store_mod.os, "replace", replace)
    return started


def test_a_put_during_a_forget_keeps_its_blob(tmp_path, monkeypatch):
    a, b = Store(tmp_path / "s"), Store(tmp_path / "s")  # two connections, as two processes
    key = a.put_doc("reward", DOC)
    got: list[str] = []
    started = _during_forget(monkeypatch, lambda: got.append(b.put_doc("reward", DOC)))
    assert a.forget_doc("reward", key) is True
    started[0].join()
    assert got == [key] and [r.key for r in a.list_docs("reward")] == [key]
    assert a.get_doc(key, RewardJson) == DOC  # before, the row was indexed and its blob deleted after


def test_a_tag_during_a_forget_is_refused_not_left_dangling(tmp_path, monkeypatch):
    a, b = Store(tmp_path / "s"), Store(tmp_path / "s")
    key = a.put_doc("reward", DOC)
    errors: list[Exception] = []

    def tag():
        try:
            b.tag("latest", key, expect=0)
        except StoreError as e:
            errors.append(e)

    started = _during_forget(monkeypatch, tag)
    assert a.forget_doc("reward", key) is True
    started[0].join()
    assert len(errors) == 1 and "missing blob" in str(errors[0]) and a.tag_info("latest") is None


def test_blobs_set_aside_by_a_dead_process_are_settled_on_open(tmp_path):
    s = Store(tmp_path / "s")
    kept, gone = s.put_doc("reward", DOC), s.put_bytes(b"orphan")
    staging = tmp_path / "s" / "staging"
    for key in (kept, gone):  # a crash after the rename: one forget never committed (its row remains), one did
        s.blob_path(key).rename(staging / f"forgotten-{key.split(':')[1]}-999999-1")
    s.close()
    again = Store(tmp_path / "s")
    assert again.get_doc(kept, RewardJson) == DOC and not again.has(gone) and not list(staging.glob("forgotten-*"))


def test_many_forgets_and_puts_never_leave_a_row_without_its_blob(tmp_path):
    a, b = Store(tmp_path / "s"), Store(tmp_path / "s")
    key = a.put_doc("reward", DOC)
    stop = time.monotonic() + 1.5

    def forgetter():
        while time.monotonic() < stop:
            a.forget_doc("reward", key)

    def putter():
        while time.monotonic() < stop:
            assert b.put_doc("reward", DOC) == key

    threads = [threading.Thread(target=forgetter), threading.Thread(target=putter)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = a.list_docs("reward")
    assert all(a.has(r.key) for r in rows)
    if rows:
        assert a.get_doc(key, RewardJson) == DOC


def test_a_set_aside_named_with_a_live_pid_is_settled_unless_its_lock_is_held(tmp_path):
    """M0 audit 2026-10-16 R-17: the pid in the name decided, and a restarted container's process is pid 1 again, so a
    crashed forget counted as live forever and the document's reads raised "missing blob". The lock now decides."""
    import fcntl
    import os

    for pid in (os.getpid(), 1):
        st = Store(tmp_path / f"s{pid}")
        key = st.put_doc("reward", DOC)
        blob = st.blob_path(key)
        aside = st.root / "staging" / f"forgotten-{blob.name}-{pid}-1"
        os.replace(blob, aside)  # a forget that crashed before its commit: the rows still name the key
        held = os.open(aside, os.O_RDONLY)
        fcntl.flock(held, fcntl.LOCK_EX)
        Store(st.root)  # its forget_doc still holds the lock: left alone
        assert aside.exists()
        os.close(held)
        assert Store(st.root).get_doc(key, RewardJson) == DOC and not aside.exists()
