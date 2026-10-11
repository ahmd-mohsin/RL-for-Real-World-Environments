"""Contamination checks for a benchmark (DESIGN §11.3): what training data, SFT traces or a model has seen of the
bench. No checkout implements more than canaries and phrase lists; the layers here follow the decontamination literature.

1. **Canaries.** A GUID per bench version (Harbor's ``harbor-canary GUID``, ``cli/init.py:168-200``) planted in every
   task file and stripped before an agent sees the instruction; any corpus containing it is contaminated outright.
2. **Exact n-gram overlap.** 13-grams of lowercased, whitespace-normalized words for text (GPT-3's decontamination), and
   a 50-character normalized substring for code and oracle diffs (StarCoder's).
3. **MinHash LSH** for corpus-scale near duplicates: 5-word shingles, 128 hash functions, 16 bands of 8 rows (a ~0.71
   similarity threshold), candidates verified by exact Jaccard on the shingle sets (``forge.dedup`` uses 0.8).
4. **Structural fingerprints.** A file tree as sorted (path, git blob sha1) pairs.
5. **Leak hygiene.** Issue and pull-request references in an instruction (``#1234``, ``gh-1234``, ``/pull/1234``; the
   SWE-rebench-V2 filter) point at the fix.

Temporal decontamination (an item counts for a model only if it post-dates the model's cutoff) is ``splits.eligible``.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORD = re.compile(r"[a-z0-9_]+")
LEAK = re.compile(r"(?:(?<![\w/])#\d{2,}\b|\bgh-\d+\b|/pull/\d+|/issues/\d+|/commit/[0-9a-f]{7,40})", re.IGNORECASE)


def new_canary(bench: str, version: str) -> str:
    """A deterministic per-version canary GUID (the namespace is the bench id)."""
    return f"crucible-canary GUID {uuid.uuid5(uuid.NAMESPACE_URL, f'crucible-bench:{bench}@{version}')}"


def strip_canary(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if "crucible-canary GUID" not in line)


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def norm_code(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def shingles(text: str, k: int = 5) -> set[str]:
    w = words(text)
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))} if w else set()


def jaccard(a: set[Any], b: set[Any]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def _h(x: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8, salt=seed.to_bytes(8, "little")).digest(), "little")


def minhash(sh: set[str], num_perm: int = 128) -> tuple[int, ...]:
    if not sh:
        return tuple([2**64 - 1] * num_perm)
    return tuple(min(_h(s, p) for s in sh) for p in range(num_perm))


def bands(sig: Sequence[int], b: int = 16) -> list[tuple[int, int]]:
    r = len(sig) // b
    return [(i, hash(tuple(sig[i * r:(i + 1) * r]))) for i in range(b)]


def tree_fingerprint(root: Path) -> str:
    """sha256 over sorted (relative path, git blob sha1) pairs: the same files hash the same anywhere."""
    root = Path(root)
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        data = p.read_bytes()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        rows.append(f"{p.relative_to(root).as_posix()} {blob}")
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


@dataclass
class Fingerprint:
    id: str
    text: str  # instruction (canary stripped)
    code: str = ""  # oracle diff / solution, if any
    tree: str | None = None
    grams: set[tuple[str, ...]] = field(default_factory=set)
    sh: set[str] = field(default_factory=set)
    sig: tuple[int, ...] = ()


def fingerprint(task_id: str, text: str, *, code: str = "", tree: Path | None = None, num_perm: int = 128) -> Fingerprint:
    t = strip_canary(text)
    sh = shingles(t)
    return Fingerprint(task_id, t, code, tree_fingerprint(tree) if tree else None, ngrams(t), sh, minhash(sh, num_perm))


def scan(bench: Sequence[Fingerprint], corpus: Iterable[tuple[str, str]], *, canary: str | None = None, n: int = 13, code_chars: int = 50,
         j_min: float = 0.8, b: int = 16) -> list[dict[str, Any]]:
    """Hits of the bench in a corpus of (source id, text): canary, exact n-gram, code substring, near duplicate."""
    index: dict[tuple[int, int], list[int]] = {}
    for k, f in enumerate(bench):
        for key in bands(f.sig, b):
            index.setdefault(key, []).append(k)
    codes = [(f.id, norm_code(f.code)) for f in bench if len(norm_code(f.code)) >= code_chars]
    hits: list[dict[str, Any]] = []
    for src, text in corpus:
        if canary and canary in text:
            hits.append({"source": src, "task": "*", "kind": "canary"})
        tg = ngrams(text, n)
        for f in bench:
            common = f.grams & tg
            if common:
                hits.append({"source": src, "task": f.id, "kind": f"{n}-gram", "overlap": len(common)})
        nc = norm_code(text)
        for tid, c in codes:
            if any(c[i:i + code_chars] in nc for i in range(0, len(c) - code_chars + 1, max(1, code_chars // 2))):
                hits.append({"source": src, "task": tid, "kind": "code-substring"})
        sh = shingles(text)
        cands = {k for key in bands(minhash(sh, len(bench[0].sig) if bench else 128), b) for k in index.get(key, [])}
        for k in sorted(cands):
            j = jaccard(bench[k].sh, sh)
            if j >= j_min:
                hits.append({"source": src, "task": bench[k].id, "kind": "near-duplicate", "jaccard": round(j, 4)})
    return hits


def leaks(instructions: Mapping[str, str]) -> dict[str, list[str]]:
    """Instructions that reference an issue, pull request or commit (they point at the fix)."""
    return {k: LEAK.findall(v) for k, v in sorted(instructions.items()) if LEAK.search(v)}
