"""Content hashing and ids.

Directory hashes use Harbor's packager scheme (``harbor/publisher/packager.py``): SHA-256 over the
sorted lines ``"<relpath>\\0<sha256(file)>\\n"``. A task directory's id is therefore identical to
the content hash Harbor computes when publishing it. File-tree *state* fingerprints
(``crucible.world.state``) extend each line with the file mode, because an executable bit is
state.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterable
from pathlib import Path

HASH_PREFIX = "sha256:"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def manifest_hash(entries: Iterable[tuple[str, str]]) -> str:
    """Harbor-compatible hash of (relpath, file_sha256) pairs; order-independent."""
    outer = hashlib.sha256()
    for rel, file_hash in sorted(entries):
        outer.update(f"{rel}\0{file_hash}\n".encode())
    return outer.hexdigest()


def tree_manifest_hash(entries: Iterable[tuple[str, str, str]]) -> str:
    """Hash of (relpath, mode, content_hash) triples; order-independent. Used for state fingerprints."""
    outer = hashlib.sha256()
    for rel, mode, content in sorted(entries):
        outer.update(f"{rel}\0{mode}\0{content}\n".encode())
    return outer.hexdigest()


def short(h: str, n: int = 12) -> str:
    return h.removeprefix(HASH_PREFIX)[:n]


def new_id(prefix: str) -> str:
    """Random, non-content id for runtime objects (episodes, sessions)."""
    return f"{prefix}-{secrets.token_hex(6)}"
