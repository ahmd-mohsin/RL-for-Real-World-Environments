"""Content-addressed local store for traces, worlds, tasks, snapshots and runs.

- **Blobs** are immutable bytes addressed by ``sha256:<hex>`` under ``<root>/blobs/ab/<hex>``.
  Putting the same bytes twice returns the same key (idempotent, deduplicated).
- **Documents** are Crucible models stored as canonical JSON blobs plus an index row
  ``(kind, key, name, created_at, provenance)`` in SQLite, so they can be listed by kind and name.
- **Tags** are mutable names pointing at keys (``latest``, ``task04@v3``) — the only mutable thing. A tag moves by
  compare-and-set on its revision (``tag(name, key, expect=rev)``; ``TagConflict`` when another writer moved it first)
  and every move is appended to its history (``tag_history``): racing writers can no longer move a tag back unseen
  (M0 audit 2026-10-14 R-16; PATTERN: SkyRL ``tinker/api.py`` versioned compare-and-swap, Apache-2.0).

Writes are safe across threads and processes (PORT of agentenv ``store/object_store/local/store.py`` and
``store/document_store/sqlite_document_store.py``, Apache-2.0): a blob is written whole to a unique staged file held
under ``flock``, fsynced, checked to hold exactly the hashed bytes, and only then renamed into place (the directory
fsynced after). A write cut short (a full disk, a quota, a file-size limit) raises and leaves no blob and no index row:
``os.write`` may return a partial count, which was dropped, so a full disk stored half a blob under the whole one's key
(M0 audit 2026-10-16 R-3; PATTERN: slime ``data/checkpoint.py`` refuses a size mismatch, Apache-2.0). A
staged file whose writer died is swept when a store opens. An existing blob whose bytes no longer match its key is
rewritten by the next put of that content. The index runs in WAL mode with a busy timeout.

The interface is deliberately small so an agentenv-backed or object-store-backed implementation can
replace it (``crucible.interop.agentenv`` mirrors artifacts into agentenv's versioned stores). It is written down as two
Protocols, ``BlobStore`` and ``DocIndex`` (``core.store.protocols``), with a backend kind (``Store.backend``), and
``core.store.conformance`` checks any implementation against them (M0 audit 2026-10-14 R-22). Every read outside this
module is verified against its key (``get_bytes``, ``open_blob``, ``export``, ``verified_path``); ``blob_path`` is the
unverified location, for this module and its tests only.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import os
import sqlite3
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from crucible.core.ids import HASH_PREFIX
from crucible.core.prov import Provenance, utc_now
from crucible.core.schema.base import CrucibleModel

M = TypeVar("M", bound=CrucibleModel)

# Index migrations (R-12), append-only: entry i takes an index from ``PRAGMA user_version`` i to i + 1. Never edit a
# shipped entry; append one. Migration 1 is the schema every index had before versioning, so on those its CREATEs are
# no-ops. The index is not rebuildable from the blobs (tags and names live only here), so a version newer than
# INDEX_VERSION is refused rather than guessed at (rLLM's registry treats any unknown version as v1 and overwrites it).
_MIGRATIONS: list[tuple[str, ...]] = [
    (
        """CREATE TABLE IF NOT EXISTS docs (
             kind TEXT NOT NULL, key TEXT NOT NULL, name TEXT, created_at TEXT NOT NULL,
             provenance TEXT, size INTEGER, PRIMARY KEY (kind, key))""",
        "CREATE INDEX IF NOT EXISTS docs_name ON docs(kind, name)",
        "CREATE TABLE IF NOT EXISTS tags (name TEXT PRIMARY KEY, key TEXT NOT NULL, updated_at TEXT NOT NULL)",
        """CREATE TABLE IF NOT EXISTS doc_names (
             kind TEXT NOT NULL, key TEXT NOT NULL, name TEXT NOT NULL, created_at TEXT NOT NULL, provenance TEXT,
             PRIMARY KEY (kind, key, name))""",
        "CREATE INDEX IF NOT EXISTS doc_names_name ON doc_names(kind, name)",
    ),
    (  # 2 (R-16): a revision per tag, and every move of a tag kept (the tags present become revision 1)
        "ALTER TABLE tags ADD COLUMN rev INTEGER NOT NULL DEFAULT 1",
        """CREATE TABLE tag_history (
             name TEXT NOT NULL, rev INTEGER NOT NULL, key TEXT NOT NULL, previous TEXT, at TEXT NOT NULL, note TEXT,
             PRIMARY KEY (name, rev))""",
        "INSERT INTO tag_history(name, rev, key, previous, at, note) SELECT name, 1, key, NULL, updated_at, NULL FROM tags",
    ),
]
INDEX_VERSION = len(_MIGRATIONS)
_STAGING = "staging"
_STAGED_PREFIX = "blob-"
_STAGED_GRACE_S = 3600.0  # a staged file this old whose lock no one holds belongs to a dead writer
_BUSY_TIMEOUT_S = 5.0


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte of ``data`` or raise: ``os.write`` may write fewer bytes than asked (R-3)."""
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise StoreError(f"write made no progress with {len(view)} bytes left")
        view = view[n:]


def _synced(fd: int, size: int) -> None:
    """fsync the staged file, then check it holds exactly ``size`` bytes: only then may it take the key (R-3)."""
    os.fsync(fd)
    if (on_disk := os.fstat(fd).st_size) != size:
        raise StoreError(f"staged blob holds {on_disk} bytes, {size} were written: not stored")


def _fsync_dir(path: Path) -> None:
    with contextlib.suppress(OSError):  # not every filesystem lets a directory be opened
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _mkdir_durable(directory: Path) -> None:
    """``mkdir -p`` whose new entries survive a power loss (R-25; PORT of Karotte ``durable_write.write_durably``, MIT,
    ``LICENSES/MIT-karotte.txt``): a directory's own fsync does not cover its entry in its parent, so the parent of every
    directory created here is synced too. Without it a new ``blobs/<ab>/`` could vanish after a crash while the index
    row naming a blob inside it survived."""
    created = list(itertools.takewhile(lambda d: not d.exists(), [directory, *directory.parents]))
    directory.mkdir(parents=True, exist_ok=True)
    for parent in dict.fromkeys(d.parent for d in reversed(created)):
        _fsync_dir(parent)


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Switch to WAL; concurrent first opens race for the exclusive lock, so retry while SQLite reports busy."""
    deadline = time.monotonic() + _BUSY_TIMEOUT_S
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) and "busy" not in str(e).lower() or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def _sweep(root: Path) -> None:
    """Remove staged files of dead writers (a live writer holds an flock on its file) and temp files the pre-audit
    store left inside blob directories."""
    cutoff = time.time() - _STAGED_GRACE_S
    for staged in (root / _STAGING).glob(f"{_STAGED_PREFIX}*"):
        with contextlib.suppress(OSError):
            if staged.stat().st_mtime > cutoff:
                continue
            with staged.open("rb") as f:
                try:
                    import fcntl

                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except ImportError:  # no flock (Windows): age alone decides
                    pass
                staged.unlink()
    for leftover in (root / "blobs").glob("*/*"):
        if len(leftover.name) == 64 and all(c in "0123456789abcdef" for c in leftover.name):
            continue
        with contextlib.suppress(OSError):
            if leftover.is_file() and leftover.stat().st_mtime <= cutoff:
                leftover.unlink()



@dataclass(frozen=True)
class DocRow:
    kind: str
    key: str
    name: str | None
    created_at: str
    provenance: Provenance | None
    size: int | None


class StoreError(RuntimeError):
    pass


TAG_MAX_LEN = 512


@dataclass(frozen=True)
class TagInfo:
    name: str
    key: str
    rev: int
    updated_at: str
    note: dict | None = None  # what the writer of this revision recorded with it


@dataclass(frozen=True)
class TagEvent:
    name: str
    rev: int
    key: str
    previous: str | None  # the key the tag held before this move (None: created)
    at: str
    note: dict | None = None


class TagConflict(StoreError):
    """A tag moved since the writer read it: nothing was changed. ``current`` is the tag as it is now."""

    def __init__(self, name: str, expected: int, current: TagInfo | None):
        self.name, self.expected, self.current = name, expected, current
        now = f"revision {current.rev} ({current.key})" if current else "absent"
        super().__init__(f"tag {name!r} is {now}, not revision {expected}: another writer moved it first")


def check_tag_name(name: object) -> str:
    """A tag name is a non-empty string of at most TAG_MAX_LEN characters without control characters, and never looks
    like a key: ``resolve`` reads ``sha256:...`` as a key, so such a tag could never be resolved (R-16)."""
    if not isinstance(name, str) or not name or len(name) > TAG_MAX_LEN:
        raise StoreError(f"a tag name is a string of 1 to {TAG_MAX_LEN} characters, not {name!r}")
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise StoreError(f"tag name {name!r} has a control character")
    if name.startswith(HASH_PREFIX):
        raise StoreError(f"tag name {name!r} looks like a key ({HASH_PREFIX}...): resolve() would read it as one")
    return name


def _migrate(conn: sqlite3.Connection, index: Path) -> None:
    """Bring ``index`` to INDEX_VERSION, or raise StoreError if a newer Crucible wrote it. A current index costs one read;
    otherwise the version is re-read and the migrations applied under one write lock, so concurrent opens migrate once."""

    def version() -> int:
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        if v > INDEX_VERSION:
            raise StoreError(f"{index}: index version {v} is newer than this Crucible's ({INDEX_VERSION}); upgrade Crucible to open this store")
        return v

    if version() == INDEX_VERSION:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        start = version()
        for statements in _MIGRATIONS[start:]:
            for sql in statements:
                conn.execute(sql)
        conn.execute(f"PRAGMA user_version = {INDEX_VERSION}")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _make_private_root(root: Path) -> None:
    """Create ``root`` owner-only (0700) with a ``.gitignore`` of ``*``: stores hold traces, prompts and tool output, and
    the umask default (0755) lets every user on a shared host read them (R-6; PATTERN: agentenv
    ``store/local_state.py:ensure_state_dir``, Apache-2.0). A root that already exists is left as it is: it may be shared
    on purpose."""
    _mkdir_durable(root.parent)
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        if root.is_dir():
            return
        raise
    _fsync_dir(root.parent)  # the new root's entry (R-25)
    (root / ".gitignore").write_text("*\n")


class Store:
    backend = "local"  # the BlobStore / DocIndex implementation: a local directory and a SQLite index

    def __init__(self, root: str | Path):
        self.root = Path(root)
        _make_private_root(self.root)
        _mkdir_durable(self.root / "blobs")
        _mkdir_durable(self.root / _STAGING)
        _sweep(self.root)
        self._lock = threading.Lock()
        index = self.root / "index.sqlite"
        with contextlib.suppress(FileExistsError):  # a new index is the owner's alone; SQLite opens the empty file
            os.close(os.open(index, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        self._db = sqlite3.connect(index, check_same_thread=False, timeout=_BUSY_TIMEOUT_S)
        _enable_wal(self._db)
        self._db.execute(f"PRAGMA busy_timeout={int(_BUSY_TIMEOUT_S * 1000)}")
        try:
            _migrate(self._db, index)
        except BaseException:
            self._db.close()
            raise
        self._settle_forgotten()

    def _settle_forgotten(self) -> None:
        """Blobs a dead process set aside in ``forget_doc`` (R-24): put back if its transaction never committed (rows
        still name the key), else removed. A live process's are left to it."""
        for f in (self.root / _STAGING).glob("forgotten-*"):
            parts = f.name.split("-")
            if len(parts) < 4 or not parts[2].isdigit() or _alive(int(parts[2])):
                continue
            key = HASH_PREFIX + parts[1]
            with contextlib.suppress(OSError, StoreError, sqlite3.Error):
                used = self._db.execute("SELECT 1 FROM docs WHERE key=? UNION SELECT 1 FROM tags WHERE key=? LIMIT 1", (key, key)).fetchone()
                target = self._blob_path(key)
                if used and not target.exists():
                    _mkdir_durable(target.parent)
                    os.replace(f, target)
                    _fsync_dir(target.parent)
                else:
                    f.chmod(0o600)
                    f.unlink()

    # ---------------- blobs ----------------
    def _blob_path(self, key: str) -> Path:
        if not key.startswith(HASH_PREFIX):
            raise StoreError(f"not a blob key: {key!r}")
        h = key[len(HASH_PREFIX):]
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise StoreError(f"malformed blob key: {key!r}")
        return self.root / "blobs" / h[:2] / h

    @contextlib.contextmanager
    def _staged(self):
        """A unique empty file to write a blob into, held under flock so a sweep never takes it; removed unless
        committed."""
        fd, name = tempfile.mkstemp(dir=self.root / _STAGING, prefix=_STAGED_PREFIX)
        staged = Path(name)
        try:
            with contextlib.suppress(ImportError):
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            yield fd, staged
        finally:
            os.close(fd)
            staged.unlink(missing_ok=True)

    def _commit(self, staged: Path, key: str) -> str:
        """Put a staged, fsynced file at its key. An existing blob with the right bytes is kept; one whose bytes no
        longer match the key (a crash, bit rot) is replaced."""
        path = self._blob_path(key)
        if path.exists():
            try:
                with path.open("rb") as f:
                    ok = HASH_PREFIX + _hash_stream(f) == key
            except OSError:
                ok = False
            if ok:
                return key
            with contextlib.suppress(OSError):
                path.chmod(0o644)
        _mkdir_durable(path.parent)  # a new blobs/<ab>/ is synced into blobs/ (R-25)
        os.chmod(staged, 0o444)
        os.replace(staged, path)
        _fsync_dir(path.parent)
        return key

    def put_bytes(self, data: bytes) -> str:
        with self._staged() as (fd, staged):
            _write_all(fd, data)
            _synced(fd, len(data))
            return self._commit(staged, HASH_PREFIX + hashlib.sha256(data).hexdigest())

    def put_file(self, src: str | Path) -> str:
        """Copy ``src`` into the store. The key is the hash of the bytes actually copied, so a source that changes
        during the copy can never be stored under a hash that does not match."""
        h, size = hashlib.sha256(), 0
        with self._staged() as (fd, staged), Path(src).open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
                _write_all(fd, block)
                size += len(block)
            _synced(fd, size)
            return self._commit(staged, HASH_PREFIX + h.hexdigest())

    def has(self, key: str) -> bool:
        return self._blob_path(key).exists()

    def blob_path(self, key: str) -> Path:
        """Where the blob lies, unverified: for this module and its tests. Readers use the verified methods below."""
        p = self._blob_path(key)
        if not p.exists():
            raise StoreError(f"missing blob {key}")
        return p

    def _check(self, key: str) -> Path:
        path = self.blob_path(key)
        with path.open("rb") as f:
            if HASH_PREFIX + _hash_stream(f) != key:
                raise StoreError(f"blob {key} is corrupt")
        return path

    def verified_path(self, key: str) -> Path:
        """The blob's path after checking its bytes against the key: for a local reader that needs a path (a local exec,
        a push to a remote one). Blobs are read-only files; the check streams, so a large payload is never held whole."""
        return self._check(key)

    def open_blob(self, key: str):
        """The blob opened for reading after its bytes were checked against the key (a stream of a large payload)."""
        return self._check(key).open("rb")

    def export(self, key: str, dest: str | Path) -> Path:
        """Copy the blob to ``dest`` (made writable), hashing what is copied: a mismatch removes the copy and raises."""
        src, dest = self.blob_path(key), Path(dest)
        h = hashlib.sha256()
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")
        try:
            with src.open("rb") as f, tmp.open("wb") as out:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
                    out.write(block)
            if HASH_PREFIX + h.hexdigest() != key:
                raise StoreError(f"blob {key} is corrupt")
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        return dest

    def get_bytes(self, key: str) -> bytes:
        data = self.blob_path(key).read_bytes()
        if HASH_PREFIX + hashlib.sha256(data).hexdigest() != key:
            raise StoreError(f"blob {key} is corrupt")
        return data

    # ---------------- documents ----------------
    def put_doc(self, kind: str, doc: CrucibleModel, *, name: str | None = None, prov: Provenance | None = None) -> str:
        """Store ``doc`` as written (``stored_json``: sorted JSON, text not NFC-folded, R-20) and index it. The key is the
        hash of those bytes, which for text in NFC is the document's ``content_hash``."""
        data = doc.stored_json().encode("utf-8")
        key = self.put_bytes(data)
        now, p = utc_now(), prov.model_dump_json() if prov else None
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if not self._blob_path(key).exists():  # forgotten since put_bytes (R-24): the row must not outlive its blob
                    self.put_bytes(data)
                self._db.execute(
                    "INSERT OR IGNORE INTO docs(kind, key, name, created_at, provenance, size) VALUES (?,?,?,?,?,?)",
                    (kind, key, name, now, p, len(data)),
                )
                if name is not None:
                    self._db.execute("UPDATE docs SET name=? WHERE kind=? AND key=? AND name IS NULL", (name, kind, key))
                    # every name a document is stored under, each with its own provenance
                    self._db.execute(
                        "INSERT OR IGNORE INTO doc_names(kind, key, name, created_at, provenance) VALUES (?,?,?,?,?)",
                        (kind, key, name, now, p),
                    )
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
        return key

    def get_doc(self, key: str, cls: type[M]) -> M:
        return cls.model_validate_json(self.get_bytes(key))

    def get_json(self, key: str) -> dict:
        return json.loads(self.get_bytes(key))

    def list_docs(self, kind: str, name: str | None = None) -> list[DocRow]:
        if name is None:
            q = "SELECT kind, key, name, created_at, provenance, size FROM docs WHERE kind=? ORDER BY created_at, rowid"
            args: list[str] = [kind]
        else:  # by any name the document was stored under (doc_names), falling back to rows indexed before that table
            q = ("SELECT n.kind, n.key, n.name, n.created_at, n.provenance, d.size FROM doc_names n "
                 "JOIN docs d ON d.kind = n.kind AND d.key = n.key WHERE n.kind=? AND n.name=? "
                 "UNION SELECT kind, key, name, created_at, provenance, size FROM docs WHERE kind=? AND name=? "
                 "AND NOT EXISTS (SELECT 1 FROM doc_names m WHERE m.kind = docs.kind AND m.key = docs.key AND m.name = docs.name) "
                 "ORDER BY 4")
            args = [kind, name, kind, name]
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [
            DocRow(k, key, n, c, Provenance.model_validate_json(p) if p else None, s) for k, key, n, c, p, s in rows
        ]

    def kinds(self) -> Iterator[tuple[str, int]]:
        with self._lock:
            yield from self._db.execute("SELECT kind, COUNT(*) FROM docs GROUP BY kind ORDER BY kind").fetchall()

    # ---------------- tags ----------------
    def forget_doc(self, kind: str, key: str) -> bool:
        """Remove a document version from the index, and its blob once nothing (another kind, a tag) refers to it.
        For content that must not stay at rest (a trace stored before a redaction fix). Returns whether the blob went.

        The rows go, the check runs and the blob leaves its key in one write transaction (renamed aside), and the file
        is removed after the commit (R-24): a ``put_doc`` or ``tag`` of the same key in another process either commits
        first (the blob stays: it is used) or starts after and finds the blob gone (``put_doc`` writes it again, ``tag``
        refuses). Before, a put or tag between the commit and the unlink left a row whose blob was deleted."""
        path, aside = self._blob_path(key), None
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute("DELETE FROM docs WHERE kind=? AND key=?", (kind, key))
                self._db.execute("DELETE FROM doc_names WHERE kind=? AND key=?", (kind, key))
                used = self._db.execute("SELECT 1 FROM docs WHERE key=? UNION SELECT 1 FROM tags WHERE key=? LIMIT 1", (key, key)).fetchone()
                if not used and path.exists():
                    aside = self.root / _STAGING / f"forgotten-{path.name}-{os.getpid()}-{threading.get_ident()}"
                    os.replace(path, aside)
                self._db.commit()
            except BaseException:
                if aside is not None and aside.exists():
                    os.replace(aside, path)
                self._db.rollback()
                raise
        if used:
            return False
        if aside is not None:
            aside.chmod(0o600)
            aside.unlink(missing_ok=True)
        return True

    def tag(self, name: str, key: str, *, expect: int | None, note: dict | None = None) -> int:
        """Point ``name`` at ``key`` and return the tag's revision. ``expect`` is the revision the caller read
        (``tag_info``; 0: the tag must not exist yet; None: move it whatever its revision): if the tag is at another
        revision, ``TagConflict`` is raised and nothing changes. The check and the move are one transaction, safe across
        processes. A move is appended to the tag's history with ``note``; a tag already at ``key`` stays as it is."""
        check_tag_name(name)
        if not self.has(key):
            raise StoreError(f"cannot tag missing blob {key}")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if not self.has(key):  # forgotten since the check above (R-24): never a tag on a deleted blob
                    raise StoreError(f"cannot tag missing blob {key}")
                row = self._db.execute("SELECT key, rev FROM tags WHERE name=?", (name,)).fetchone()
                rev = row[1] if row else 0
                if expect is not None and expect != rev:
                    current = self._tag_info(name)
                    raise TagConflict(name, expect, current)
                if row and row[0] == key:
                    self._db.commit()
                    return rev
                now = utc_now()
                if row:
                    self._db.execute("UPDATE tags SET key=?, rev=?, updated_at=? WHERE name=? AND rev=?", (key, rev + 1, now, name, rev))
                else:
                    self._db.execute("INSERT INTO tags(name, key, updated_at, rev) VALUES (?,?,?,?)", (name, key, now, 1))
                self._db.execute("INSERT INTO tag_history(name, rev, key, previous, at, note) VALUES (?,?,?,?,?,?)",
                                 (name, rev + 1, key, row[0] if row else None, now, json.dumps(note, sort_keys=True) if note is not None else None))
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise
        return rev + 1

    def _tag_info(self, name: str) -> TagInfo | None:
        row = self._db.execute("SELECT t.key, t.rev, t.updated_at, h.note FROM tags t LEFT JOIN tag_history h ON h.name = t.name AND h.rev = t.rev "
                               "WHERE t.name=?", (name,)).fetchone()
        return TagInfo(name, row[0], row[1], row[2], json.loads(row[3]) if row[3] else None) if row else None

    def tag_info(self, name: str) -> TagInfo | None:
        """The tag's key, revision, time of its last move and that move's note; None for an unknown tag."""
        with self._lock:
            return self._tag_info(name)

    def tag_history(self, name: str) -> list[TagEvent]:
        """Every move of the tag, oldest first (a tag from before R-16 starts at its key then, as revision 1)."""
        with self._lock:
            rows = self._db.execute("SELECT rev, key, previous, at, note FROM tag_history WHERE name=? ORDER BY rev", (name,)).fetchall()
        return [TagEvent(name, r, k, p, a, json.loads(n) if n else None) for r, k, p, a, n in rows]

    def resolve(self, name_or_key: str) -> str:
        if name_or_key.startswith(HASH_PREFIX):
            return name_or_key
        with self._lock:
            row = self._db.execute("SELECT key FROM tags WHERE name=?", (name_or_key,)).fetchone()
        if row is None:
            raise StoreError(f"unknown tag {name_or_key!r}")
        return row[0]

    def close(self) -> None:
        with self._lock:
            self._db.close()


def _alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # it exists but belongs to someone else
        return True
    return True


def _hash_stream(f) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: f.read(1 << 20), b""):
        h.update(block)
    return h.hexdigest()


_STORE: Store | None = None


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        from crucible.core.config import get_config

        _STORE = Store(get_config().data_path() / "store")
    return _STORE


def set_store(store: Store | None) -> None:
    global _STORE
    _STORE = store
