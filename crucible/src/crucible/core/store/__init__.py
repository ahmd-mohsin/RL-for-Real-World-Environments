"""Content-addressed local store for traces, worlds, tasks, snapshots and runs.

- **Blobs** are immutable bytes addressed by ``sha256:<hex>`` under ``<root>/blobs/ab/<hex>``.
  Putting the same bytes twice returns the same key (idempotent, deduplicated).
- **Documents** are Crucible models stored as canonical JSON blobs plus an index row
  ``(kind, key, name, created_at, provenance)`` in SQLite, so they can be listed by kind and name.
- **Tags** are mutable names pointing at keys (``latest``, ``task04@v3``) — the only mutable thing.

The interface is deliberately small so an agentenv-backed or object-store-backed implementation can
replace it (``crucible.interop.agentenv`` mirrors artifacts into agentenv's versioned stores).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from crucible.core.ids import HASH_PREFIX
from crucible.core.prov import Provenance, utc_now
from crucible.core.schema.base import CrucibleModel

M = TypeVar("M", bound=CrucibleModel)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
  kind TEXT NOT NULL, key TEXT NOT NULL, name TEXT, created_at TEXT NOT NULL,
  provenance TEXT, size INTEGER, PRIMARY KEY (kind, key)
);
CREATE INDEX IF NOT EXISTS docs_name ON docs(kind, name);
CREATE TABLE IF NOT EXISTS tags (name TEXT PRIMARY KEY, key TEXT NOT NULL, updated_at TEXT NOT NULL);
"""


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


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.root / "index.sqlite", check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # ---------------- blobs ----------------
    def _blob_path(self, key: str) -> Path:
        if not key.startswith(HASH_PREFIX):
            raise StoreError(f"not a blob key: {key!r}")
        h = key[len(HASH_PREFIX):]
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise StoreError(f"malformed blob key: {key!r}")
        return self.root / "blobs" / h[:2] / h

    def put_bytes(self, data: bytes) -> str:
        key = HASH_PREFIX + hashlib.sha256(data).hexdigest()
        path = self._blob_path(key)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                tmp.write(data)
            os.replace(tmp.name, path)
            os.chmod(path, 0o444)
        return key

    def put_file(self, src: str | Path) -> str:
        src = Path(src)
        h = hashlib.sha256()
        with src.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        key = HASH_PREFIX + h.hexdigest()
        path = self._blob_path(key)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp" + str(os.getpid()))
            shutil.copyfile(src, tmp)
            os.replace(tmp, path)
            os.chmod(path, 0o444)
        return key

    def has(self, key: str) -> bool:
        return self._blob_path(key).exists()

    def blob_path(self, key: str) -> Path:
        p = self._blob_path(key)
        if not p.exists():
            raise StoreError(f"missing blob {key}")
        return p

    def get_bytes(self, key: str) -> bytes:
        data = self.blob_path(key).read_bytes()
        if HASH_PREFIX + hashlib.sha256(data).hexdigest() != key:
            raise StoreError(f"blob {key} is corrupt")
        return data

    # ---------------- documents ----------------
    def put_doc(self, kind: str, doc: CrucibleModel, *, name: str | None = None, prov: Provenance | None = None) -> str:
        data = doc.canonical_json().encode("utf-8")
        key = self.put_bytes(data)
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO docs(kind, key, name, created_at, provenance, size) VALUES (?,?,?,?,?,?)",
                (kind, key, name, utc_now(), prov.model_dump_json() if prov else None, len(data)),
            )
            if name is not None:
                self._db.execute("UPDATE docs SET name=? WHERE kind=? AND key=? AND name IS NULL", (name, kind, key))
            self._db.commit()
        return key

    def get_doc(self, key: str, cls: type[M]) -> M:
        return cls.model_validate_json(self.get_bytes(key))

    def get_json(self, key: str) -> dict:
        return json.loads(self.get_bytes(key))

    def list_docs(self, kind: str, name: str | None = None) -> list[DocRow]:
        q = "SELECT kind, key, name, created_at, provenance, size FROM docs WHERE kind=?"
        args: list[str] = [kind]
        if name is not None:
            q += " AND name=?"
            args.append(name)
        q += " ORDER BY created_at"
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [
            DocRow(k, key, n, c, Provenance.model_validate_json(p) if p else None, s) for k, key, n, c, p, s in rows
        ]

    def kinds(self) -> Iterator[tuple[str, int]]:
        with self._lock:
            yield from self._db.execute("SELECT kind, COUNT(*) FROM docs GROUP BY kind ORDER BY kind").fetchall()

    # ---------------- tags ----------------
    def tag(self, name: str, key: str) -> None:
        if not self.has(key):
            raise StoreError(f"cannot tag missing blob {key}")
        with self._lock:
            self._db.execute(
                "INSERT INTO tags(name, key, updated_at) VALUES (?,?,?) ON CONFLICT(name) DO UPDATE SET key=excluded.key, updated_at=excluded.updated_at",
                (name, key, utc_now()),
            )
            self._db.commit()

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
