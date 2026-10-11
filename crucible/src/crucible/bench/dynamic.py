"""Dynamically regenerated benchmarks (DESIGN §11.4): fixed anchors plus fresh items from held-out seeds behind a secret.

- **Item seeds** are ``HMAC-SHA256(K_version, "<family>/<seed_ref>/<idx>")``: reproducible by the key holder,
  unguessable to everyone else, and collision-free (``seed + i`` collides: prime-environments ``enigmata``,
  research-environments ``uuid_ctf``; a global ``random.seed`` leaks state: SWE-smith). FineEnvs portsim keys its RNG by a
  string the same way (``dock.py:92``).
- **Commit–reveal.** A version publishes ``sha256(K)`` when it ships and reveals ``K`` when it retires; the manifest never
  holds a seed or the key.
- **Validation chain** before a fresh item is admitted, in order (each a caller-supplied async check, so packs keep their
  domain knowledge): gold passes (twice: flakiness), no-op stays below solved, the naive baseline does not solve it,
  no contamination hit, then the calibration band. A failed item is replaced by the next index (FineEnvs' top-up loop,
  ``build.py:64-77``), up to ``max_tries``.
- **Composition.** About 30% anchors (fixed forever; quarantined on drift, never rotated) and 70% fresh items, stratified
  by family round-robin.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Check = Callable[[Any], Awaitable[tuple[bool, str]]]


def item_seed(key: bytes, family: str, seed_ref: str, idx: int) -> int:
    return int.from_bytes(hmac.new(key, f"{family}/{seed_ref}/{idx}".encode(), hashlib.sha256).digest()[:8], "big")


def commitment(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()


@dataclass
class BenchItem:
    id: str
    family: str
    kind: str  # anchor | fresh
    task: Any = None
    checks: dict[str, str] = field(default_factory=dict)  # check name -> "pass" | reason


@dataclass
class Benchmark:
    name: str
    version: str
    commitment: str
    items: list[BenchItem]
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def manifest(self) -> dict[str, Any]:
        """What is published: ids, kinds and families, the key commitment and the rejection log; never a seed or the key."""
        return {"name": self.name, "version": self.version, "commitment": self.commitment,
                "items": [{"id": i.id, "family": i.family, "kind": i.kind, "checks": i.checks} for i in self.items],
                "counts": {"anchor": sum(i.kind == "anchor" for i in self.items), "fresh": sum(i.kind == "fresh" for i in self.items)},
                "rejected": self.rejected}


async def build(name: str, version: str, *, key: bytes, anchors: Sequence[tuple[str, str, Any]],
                generators: Mapping[str, Callable[[int], Any]], n_fresh: int, seed_ref: str = "heldout", checks: Sequence[tuple[str, Check]] = (),
                id_of: Callable[[Any], str] = lambda t: str(getattr(t, "id", t)), max_tries: int = 5) -> Benchmark:
    """``anchors``: (id, family, task). ``generators``: family -> (seed -> task). Fresh items are drawn family by family
    (sorted, round-robin) until ``n_fresh`` pass every check."""
    items = [BenchItem(i, f, "anchor", t, {"anchor": "fixed"}) for i, f, t in anchors]
    rejected: list[dict[str, Any]] = []
    fams = sorted(generators)
    idx = {f: 0 for f in fams}
    fresh = 0
    tries = 0
    k = 0
    while fresh < n_fresh and fams:
        fam = fams[k % len(fams)]
        k += 1
        task = generators[fam](item_seed(key, fam, seed_ref, idx[fam]))
        idx[fam] += 1
        verdicts, ok = {}, True
        for cname, check in checks:
            passed, why = await check(task)
            verdicts[cname] = "pass" if passed else why
            if not passed:
                ok = False
                break
        if ok:
            items.append(BenchItem(id_of(task), fam, "fresh", task, verdicts))
            fresh += 1
            tries = 0
        else:
            rejected.append({"family": fam, "index": idx[fam] - 1, "id": id_of(task), "checks": verdicts})
            tries += 1
            if tries >= max_tries * len(fams):
                break  # the generator cannot produce valid items: report, never loop forever
    return Benchmark(name, version, commitment(key), items, rejected)
