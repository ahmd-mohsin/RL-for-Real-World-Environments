"""Content hashing and ids.

Directory hashes use Harbor's packager scheme (``harbor/publisher/packager.py``): SHA-256 over the
sorted lines ``"<relpath>\\0<sha256(file)>\\n"``. A task directory's id is therefore identical to
the content hash Harbor computes when publishing it. File-tree *state* fingerprints
(``crucible.world.state``) extend each line with the file mode, because an executable bit is
state.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import unicodedata
from collections.abc import Iterable
from pathlib import Path

HASH_PREFIX = "sha256:"
# What a content hash covers (canonical JSON of the model, defaults included, None dropped, ``core.canonical`` forms).
# Bump on purpose when that changes; ``tests/unit/golden_hashes.json`` pins the ids of reference documents.
HASH_SCHEME = 1

# Id rules, PORT of agentenv ``store/ids.py:validate_local_id`` (Apache-2.0): letters, digits and marks of any script,
# a fixed punctuation set, no invisible characters, no detached combining marks, no empty / "." / ".." segments, and
# bounded bytes (a filesystem component holds at most 255 bytes).
MAX_ID_BYTES = 4096
MAX_SEGMENT_BYTES = 255
_ID_PUNCTUATION = frozenset("._-~/@()+,&:#=")
_ID_CATEGORIES = frozenset("LNM")
_INVISIBLE = re.compile("[\u034f\u115f\u1160\u17b4\u17b5\u180b-\u180d\u180f\u200b-\u200f\u2060-\u2064"
                        "\u3164\ufe00-\ufe0f\ufeff\uffa0\U000e0100-\U000e01ef]")


def validate_id(entity_id: str) -> str:
    """Return ``entity_id`` NFC-normalized, or raise ValueError if it is unsafe as a path or key segment."""
    if not isinstance(entity_id, str) or not entity_id:
        raise ValueError("an id must be a non-empty string")
    entity_id = unicodedata.normalize("NFC", entity_id)
    if len(entity_id.encode("utf-8", "surrogatepass")) > MAX_ID_BYTES:
        raise ValueError(f"id is longer than {MAX_ID_BYTES} bytes: {entity_id[:64]!r}")
    bad = sorted({c for c in entity_id if _INVISIBLE.match(c) or not (c in _ID_PUNCTUATION or unicodedata.category(c)[0] in _ID_CATEGORIES)})
    if bad:
        raise ValueError(f"id {entity_id!r} contains {''.join(bad)!a}")
    for seg in entity_id.split("/"):
        if seg in ("", ".", ".."):
            raise ValueError(f"id {entity_id!r} has an empty, '.' or '..' segment")
        if len(seg.encode("utf-8")) > MAX_SEGMENT_BYTES:
            raise ValueError(f"id segment longer than {MAX_SEGMENT_BYTES} bytes in {entity_id[:64]!r}")
        cats = [unicodedata.category(c)[0] for c in seg]
        if any(this == "M" and before not in _ID_CATEGORIES for before, this in zip(["Z", *cats], cats)):
            raise ValueError(f"id {entity_id!r} has a combining mark that does not follow a letter or digit")
    return entity_id


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def derive_seed(world_seed: int, name: str) -> int:
    """A per-App (or per-actor) seed from the World seed: stable, independent across names, 60-bit."""
    return int(hashlib.sha256(f"{world_seed}|{name}".encode()).hexdigest()[:15], 16)


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
