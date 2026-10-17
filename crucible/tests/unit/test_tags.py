"""Store tags: compare-and-set on a revision, an append-only history, a name grammar (M0 audit 2026-10-14 R-16)."""

from __future__ import annotations

import random
import sqlite3
import threading
from itertools import pairwise

import pytest

from crucible.connectors.file_drop import FieldMap, rows_to_trace
from crucible.connectors.trace_store import TraceStore
from crucible.core.prov import utc_now
from crucible.core.schema.signal import Signal
from crucible.core.store import Store, StoreError, TagConflict


def test_a_tag_moves_only_from_the_revision_its_writer_read(tmp_path):
    s = Store(tmp_path / "s")
    a, b, c = (s.put_bytes(x) for x in (b"a", b"b", b"c"))
    assert s.tag("latest", a, expect=0) == 1 and s.tag("latest", b, expect=1) == 2
    with pytest.raises(TagConflict) as e:
        s.tag("latest", c, expect=1)  # a writer that read revision 1: another moved it first
    assert e.value.current.rev == 2 and e.value.current.key == b and s.resolve("latest") == b
    with pytest.raises(TagConflict):
        s.tag("latest", c, expect=0)  # create-only on an existing tag
    assert s.tag("latest", b, expect=2) == 2  # already there: no move, no history row
    assert s.tag("latest", c, expect=None, note={"why": "forced"}) == 3
    assert [(h.rev, h.key, h.previous, h.note) for h in s.tag_history("latest")] == [(1, a, None, None), (2, b, a, None), (3, c, b, {"why": "forced"})]
    assert s.tag_info("latest").note == {"why": "forced"} and s.tag_info("nope") is None


def test_racing_writers_on_two_connections_never_lose_a_move(tmp_path):
    """Two Store objects are two connections, as two processes would be: every revision is won exactly once."""
    stores = [Store(tmp_path / "s"), Store(tmp_path / "s")]
    keys = [stores[0].put_bytes(str(i).encode()) for i in range(40)]
    won: list[int] = []

    def writer(st: Store, mine: list[str]) -> None:
        for k in mine:
            while True:
                info = st.tag_info("head")
                try:
                    won.append(st.tag("head", k, expect=info.rev if info else 0))
                    break
                except TagConflict:
                    continue

    threads = [threading.Thread(target=writer, args=(stores[i % 2], keys[i::2])) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    hist = stores[0].tag_history("head")
    assert sorted(won) == list(range(1, 41)) and [h.rev for h in hist] == list(range(1, 41))
    assert all(h.previous == p.key for p, h in pairwise(hist))  # an unbroken chain of moves


@pytest.mark.parametrize("bad", ["sha256:" + "0" * 64, 7, "", "a\nb", "x" * 513])
def test_tag_names_that_could_never_be_resolved_are_refused(tmp_path, bad):
    s = Store(tmp_path / "s")
    with pytest.raises(StoreError):
        s.tag(bad, s.put_bytes(b"x"), expect=0)


def test_tags_from_an_index_before_revisions_become_revision_1(tmp_path, monkeypatch):
    import crucible.core.store as store_mod

    monkeypatch.setattr(store_mod, "INDEX_VERSION", 1)
    monkeypatch.setattr(store_mod, "_MIGRATIONS", store_mod._MIGRATIONS[:1])
    old = Store(tmp_path / "s")
    key = old.put_bytes(b"x")
    old._db.execute("INSERT INTO tags(name, key, updated_at) VALUES ('latest', ?, '2026-01-01T00:00:00.000000Z')", (key,))
    old._db.commit()
    old.close()
    monkeypatch.undo()
    s = Store(tmp_path / "s")
    assert s.tag_info("latest").rev == 1 and [(h.rev, h.key) for h in s.tag_history("latest")] == [(1, key)]
    with sqlite3.connect(tmp_path / "s" / "index.sqlite") as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == store_mod.INDEX_VERSION == 2


def _grown(n: int):
    rows = [{"case": "T1", "id": str(i), "time": f"2024-05-01T{i:02d}:00:00Z", "type": "note", "text": f"step {i}"} for i in range(n)]
    return rows_to_trace(rows, FieldMap(), case="T1", connector="file_drop")


def test_n_ingests_of_a_growing_source_keep_the_newest_head(tmp_path, monkeypatch):
    """The audit's race: a version fetched earlier but stored last used to become the head again, and purging then
    deleted the newer version. The head is now the newest fetch whatever order the puts land in."""
    monkeypatch.setenv("CRUCIBLE_REDACTION_SALT", "s")
    ts = TraceStore(Store(tmp_path / "store"))
    fetches = [(utc_now(), _grown(i)) for i in range(1, 9)]  # fetch i reads a source with i events
    random.Random(0).shuffle(fetches)
    for at, trace in fetches:
        ts.put(trace, fetched_at=at)
    assert len(ts.get("file_drop:T1").events) == 8
    ts.purge_superseded()
    assert len(ts.get("file_drop:T1").events) == 8  # the newest version survives the purge


def test_a_put_that_lost_the_head_carries_the_new_heads_signals(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_REDACTION_SALT", "s")
    ts = TraceStore(Store(tmp_path / "store"))
    ts.put(_grown(1))
    stale = ts.head("file_drop:T1")
    ts.put(_grown(1).model_copy(update={"signals": [Signal(id="s1", type="correction", polarity=-1, grades=[0], source_event=0)]}))
    reads = iter([stale])
    real = ts.head
    monkeypatch.setattr(ts, "head", lambda tid: next(reads, None) or real(tid))  # the first read predates the signal
    ts.put(_grown(2))
    got = ts.get("file_drop:T1")
    assert len(got.events) == 2 and [s.id for s in got.signals] == ["s1"]  # the stale read used to drop the signal
    assert [h.rev for h in ts.store.tag_history("trace-head:file_drop:T1")] == [1, 2, 3]
