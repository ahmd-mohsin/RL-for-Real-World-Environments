"""The store conformance kit (M0 audit 2026-10-14 R-22; PATTERN: agentenv ``tst/store/conformance.py``, which races every
backend through one suite, Apache-2.0).

``check(make, root)`` runs every check against a backend: ``make(path)`` opens the backend's store rooted at ``path``
(two calls with one path are two clients of one store, as two processes would be). A failed check raises
AssertionError naming it. ``core.store.Store`` passes it (``tests/unit/test_store_conformance.py``); a new backend must
too before it replaces it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from crucible.core.store.protocols import BlobStore, DocIndex


def _raises(fn: Callable[[], Any], what: str) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001 - any refusal is a refusal; the check is that nothing corrupt is returned
        return
    raise AssertionError(what)


def check_blobs(make: Callable[[Path], Any], root: Path, corrupt: Callable[[Any, str], None]) -> None:
    """Round trips, idempotent keys, and every verified read refusing a blob whose bytes no longer match its key
    (``corrupt(store, key)`` damages one in the backend's own way)."""
    s = make(root)
    assert isinstance(s, BlobStore) and s.backend, "the store is a BlobStore with a backend kind"
    k = s.put_bytes(b"hello")
    assert k.startswith("sha256:") and k == s.put_bytes(b"hello"), "put_bytes is content-addressed and idempotent"
    assert s.has(k) and s.get_bytes(k) == b"hello", "get_bytes returns what was put"
    src = root.parent / f"{root.name}-src.bin"
    src.write_bytes(b"x" * (3 << 20))
    big = s.put_file(src)
    with s.open_blob(big) as f:
        assert f.read() == src.read_bytes(), "open_blob streams what was put"
    out = s.export(big, root.parent / f"{root.name}-out.bin")
    assert out.read_bytes() == src.read_bytes(), "export copies what was put"
    assert s.verified_path(big).read_bytes() == src.read_bytes(), "verified_path names what was put"
    missing = "sha256:" + "0" * 64
    assert not s.has(missing)
    _raises(lambda: s.get_bytes(missing), "a missing blob is refused")
    corrupt(s, k)
    _raises(lambda: s.get_bytes(k), "get_bytes refuses a corrupt blob")
    _raises(lambda: s.open_blob(k).read(), "open_blob refuses a corrupt blob")
    _raises(lambda: s.verified_path(k), "verified_path refuses a corrupt blob")
    dest = root.parent / f"{root.name}-bad.bin"
    _raises(lambda: s.export(k, dest), "export refuses a corrupt blob")
    assert not dest.exists(), "a refused export leaves no copy"
    assert s.put_bytes(b"hello") == k and s.get_bytes(k) == b"hello", "putting the content again repairs a corrupt blob"


def check_docs(make: Callable[[Path], Any], root: Path, doc: Any) -> None:
    """Documents by kind and name, and tags by compare-and-set with a history (``doc``: any Crucible model)."""
    s = make(root)
    assert isinstance(s, DocIndex), "the store is a DocIndex"
    k = s.put_doc("probe", doc, name="n1")
    assert k == s.put_doc("probe", doc, name="n1") and s.get_doc(k, type(doc)) == doc, "put_doc is idempotent and round-trips"
    assert [r.key for r in s.list_docs("probe", name="n1")] == [k] and ("probe", 1) in list(s.kinds()), "documents are listed"
    assert s.tag("t", k, expect=0) == 1 and s.resolve("t") == k, "a tag is created at revision 1"
    _raises(lambda: s.tag("t", k, expect=0), "a stale writer cannot move a tag")
    assert [h.rev for h in s.tag_history("t")] == [1] and s.tag_info("t").rev == 1, "a tag's moves are kept"


def check_races(make: Callable[[Path], Any], root: Path, writers: int = 4, moves: int = 25) -> None:
    """Several clients putting one content and moving one tag at once: one key, every revision won exactly once."""
    stores = [make(root) for _ in range(2)]
    keys = [stores[0].put_bytes(f"v{i}".encode()) for i in range(writers * moves)]
    same: list[str] = []
    won: list[int] = []
    lock = threading.Lock()

    def writer(st: Any, mine: list[str]) -> None:
        for key in mine:
            got = st.put_bytes(b"shared content")
            while True:
                info = st.tag_info("race")
                try:
                    rev = st.tag("race", key, expect=info.rev if info else 0)
                    break
                except Exception:  # noqa: BLE001, S112 - a conflict: read the tag again
                    continue
            with lock:
                same.append(got)
                won.append(rev)

    threads = [threading.Thread(target=writer, args=(stores[i % 2], keys[i::writers])) for i in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(same)) == 1, "concurrent puts of one content give one key"
    assert sorted(won) == list(range(1, writers * moves + 1)), "every revision of a raced tag is won exactly once"
    assert [h.rev for h in stores[0].tag_history("race")] == sorted(won), "the history has every move"


def check(make: Callable[[Path], Any], root: Path, *, corrupt: Callable[[Any, str], None], doc: Any) -> None:
    root = Path(root)
    check_blobs(make, root / "blobs-check", corrupt)
    check_docs(make, root / "docs-check", doc)
    check_races(make, root / "race-check")
