"""Dynamically regenerated benchmarks (DESIGN §11.4): fixed anchors plus fresh items from held-out seeds behind a secret.

- **The key K** (M13 audit V4). There is no default: ``load_key`` reads it from ``$CRUCIBLE_BENCH_KEY`` (hex or raw) or
  a key file (``$CRUCIBLE_BENCH_KEY_FILE`` / ``--key-file``), never from argv (``ps`` and shell history). ``check_key``
  refuses fewer than 32 bytes, the old public defaults (``crucible-bench-v1``, ``crucible-build-v1``) and repeated
  patterns; ``new_key`` makes one (``secrets.token_bytes(32)``).
- **Item seeds** (V11) are ``HMAC-SHA256(K, canonical JSON ["crucible-item-seed", bench, version, family, seed_ref,
  idx])``, 64 bits: reproducible by the key holder, unguessable to everyone else, collision-free (``seed + i``
  collides: prime-environments ``enigmata``, research-environments ``uuid_ctf``; a global ``random.seed`` leaks state:
  SWE-smith), bound to the bench name and version (reusing K for v2 no longer regenerates v1), and unambiguous (``/`` in
  a family no longer aliases two tuples). FineEnvs portsim keys its RNG by a string the same way (``dock.py:92``).
- **Published ids never derive from seeds** (V3). A fresh item's id is ``<bench>/<family>/f<HMAC(K, [bench, version,
  family, idx])[:16]>@<version>``; ``build`` re-keys the task to that id (``generate(seed, item_id=...)`` when the
  generator accepts it, else ``task.id = ...``) and refuses a task whose own id still spells its seed. The manifest
  and the rejection log carry only these opaque ids (packs used ``s{seed}``, and ``seed % 100_000`` made the whole
  fresh pool enumerable without K).
- **Commit–reveal** (V4). A version publishes ``sha256("crucible-bench-commit" ‖ salt ‖ K)`` with its salt when it
  ships and reveals K when it retires; the salt (per bench version) stops one precomputed dictionary from serving every
  bench, and the key policy above is what makes K unguessable. ``registry`` (a private local file of key fingerprints)
  refuses a K already used by another bench version.
- **Validation chain** before a fresh item is admitted, in order (each a caller-supplied async check, so packs keep their
  domain knowledge): gold passes (repeatedly: flakiness), no-op stays below solved, the naive baseline does not solve it,
  no contamination hit, then the calibration band. ``build`` refuses ``n_fresh > 0`` with no checks (V12: production
  builds admitted unvalidated items) unless ``allow_unvalidated``. A check that raises is a rejection
  (``error: <type>``), not an aborted build (V25). A failed item is replaced by the next index (FineEnvs' top-up loop,
  ``build.py:64-77``).
- **Composition.** About 30% anchors (fixed forever; quarantined on drift, never rotated) and 70% fresh items, stratified
  by family round-robin. A family that fails ``max_tries`` times in a row is exhausted and reported; the per-family
  yields are in the manifest; a benchmark short of ``n_fresh`` raises ``ShortBenchmark`` (V25: it returned silently);
  duplicate ids are refused.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import os
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.canonical import dumps as canonical_dumps

Check = Callable[[Any], Awaitable[tuple[bool, str]]]

MIN_KEY_BYTES = 32
KEY_ENV, KEY_FILE_ENV = "CRUCIBLE_BENCH_KEY", "CRUCIBLE_BENCH_KEY_FILE"
PUBLIC_KEYS = frozenset({b"crucible-bench-v1", b"crucible-build-v1", b"toy-bench-v1-key"})  # once shipped as defaults


class ShortBenchmark(ValueError):
    """Fewer than ``n_fresh`` items passed validation; ``.benchmark`` holds what was built (with the rejection log)."""

    def __init__(self, msg: str, benchmark: Benchmark):
        super().__init__(msg)
        self.benchmark = benchmark


def new_key() -> bytes:
    return secrets.token_bytes(MIN_KEY_BYTES)


def check_key(key: bytes) -> bytes:
    """V4: the key must be secret bytes: at least 32, not a known public default, not a short repeated pattern."""
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError("the bench key must be bytes")
    key = bytes(key)
    if len(key) < MIN_KEY_BYTES:
        raise ValueError(f"the bench key must be at least {MIN_KEY_BYTES} random bytes (got {len(key)}); make one with dynamic.new_key()")
    if any(key == k or key.startswith(k) or key.strip() == k for k in PUBLIC_KEYS):
        raise ValueError("the bench key is a published default; anyone can regenerate the held-out items")
    if len(set(key)) < 10 or any(key == key[:p] * (len(key) // p) + key[: len(key) % p] for p in range(1, 9)):
        raise ValueError("the bench key is a short repeated pattern, not random bytes")
    return key


def _decode(raw: bytes) -> bytes:
    s = raw.strip()
    try:
        txt = s.decode("ascii")
        if len(txt) >= 2 * MIN_KEY_BYTES and len(txt) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in txt):
            return bytes.fromhex(txt)
    except UnicodeDecodeError:
        pass
    return s


def load_key(*, file: str | Path | None = None, env: str = KEY_ENV, file_env: str = KEY_FILE_ENV) -> bytes:
    """The bench key from a file (``file`` or ``$CRUCIBLE_BENCH_KEY_FILE``) or ``$CRUCIBLE_BENCH_KEY``; hex text is
    decoded. Never from argv. Raises when none is set (V4: there is no default)."""
    path = file or os.environ.get(file_env)
    if path:
        return check_key(_decode(Path(path).read_bytes()))
    val = os.environ.get(env)
    if val:
        return check_key(_decode(val.encode()))
    raise ValueError(f"no bench key: set ${env} (hex) or ${file_env} (a file), e.g. from python -c 'import secrets; print(secrets.token_hex(32))'")


def _mac(key: bytes, parts: list[Any]) -> bytes:
    return hmac.new(key, canonical_dumps(parts).encode(), hashlib.sha256).digest()


def item_seed(key: bytes, family: str, seed_ref: str, idx: int, *, bench: str, version: str) -> int:
    """V11: bound to the bench name and version; fields in canonical JSON (no ``/`` aliasing); 64 bits."""
    return int.from_bytes(_mac(key, ["crucible-item-seed", bench, version, family, seed_ref, int(idx)])[:8], "big")


def item_id(key: bytes, family: str, idx: int, *, bench: str, version: str) -> str:
    """V3: an opaque published id; nothing about the seed can be read from it."""
    return f"{bench}/{family}/f{_mac(key, ['crucible-item-id', bench, version, family, int(idx)]).hex()[:16]}@{version}"


def commit_salt(bench: str, version: str) -> bytes:
    return hashlib.sha256(canonical_dumps(["crucible-commit-salt", bench, version]).encode()).digest()[:16]


def commitment(key: bytes, salt: bytes = b"") -> str:
    """``sha256("crucible-bench-commit\\0" ‖ len(salt) ‖ salt ‖ K)``; publish it with the salt, reveal K at retirement."""
    return hashlib.sha256(b"crucible-bench-commit\0" + len(salt).to_bytes(2, "big") + salt + bytes(key)).hexdigest()


def verify_commitment(key: bytes, manifest: Mapping[str, Any]) -> bool:
    return hmac.compare_digest(commitment(key, bytes.fromhex(manifest.get("commitment_salt", ""))), str(manifest["commitment"]))


def key_fingerprint(key: bytes) -> str:
    """For the private key registry only (needs K to compute; never published)."""
    return hmac.new(bytes(key), b"crucible-key-registry", hashlib.sha256).hexdigest()


def _register(registry: Path, key: bytes, bench: str, version: str) -> None:
    registry = Path(registry)
    fp = key_fingerprint(key)
    rows = [json.loads(x) for x in registry.read_text().splitlines() if x.strip()] if registry.exists() else []
    for r in rows:
        if r.get("fp") == fp and (r.get("bench"), r.get("version")) != (bench, version):
            # M14 campaign B12: the message named a new version when the clash was another benchmark. One key builds one
            # benchmark version: revealing it at retirement must not open any other
            if r.get("bench") != bench:
                raise ValueError(f"this key already built another benchmark, {r.get('bench')}@{r.get('version')}: one key builds one benchmark, "
                                 f"so {bench}@{version} needs its own key (a different --key-file) (V11)")
            raise ValueError(f"this key already built {r.get('bench')}@{r.get('version')}: a new version needs a new key (V11)")
    if not any(r.get("fp") == fp for r in rows):
        registry.parent.mkdir(parents=True, exist_ok=True)
        with registry.open("a") as f:
            f.write(json.dumps({"fp": fp, "bench": bench, "version": version}) + "\n")


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
    commitment_salt: str = ""
    yields: dict[str, dict[str, Any]] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        """What is published: opaque ids, kinds and families, the key commitment and its salt, per-family yields and the
        rejection log; never a seed, a seed-derived id or the key."""
        return {"name": self.name, "version": self.version, "commitment": self.commitment, "commitment_salt": self.commitment_salt,
                "items": [{"id": i.id, "family": i.family, "kind": i.kind, "checks": i.checks} for i in self.items],
                "counts": {"anchor": sum(i.kind == "anchor" for i in self.items), "fresh": sum(i.kind == "fresh" for i in self.items)},
                "yields": self.yields, "rejected": self.rejected}


def _accepts_item_id(fn: Callable[..., Any]) -> bool:
    try:
        ps = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "item_id" in ps or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps.values())


def _spells_seed(text: str, seed: int) -> bool:
    return any(str(x) in text for x in {seed, seed % 10**5, seed % 10**6, seed % 2**32} if x >= 100)


async def build(name: str, version: str, *, key: bytes, anchors: Sequence[tuple[str, str, Any]],
                generators: Mapping[str, Callable[..., Any]], n_fresh: int, seed_ref: str = "heldout", checks: Sequence[tuple[str, Check]] = (),
                id_of: Callable[[Any], str] = lambda t: str(getattr(t, "id", t)), max_tries: int = 5, allow_unvalidated: bool = False,
                allow_short: bool = False, registry: Path | None = None, salt: bytes | None = None) -> Benchmark:
    """``anchors``: (id, family, task). ``generators``: family -> (seed -> task), optionally taking ``item_id=``. Fresh
    items are drawn family by family (sorted, round-robin) until ``n_fresh`` pass every check."""
    key = check_key(key)
    if n_fresh > 0 and generators and not checks and not allow_unvalidated:
        raise ValueError("fresh items need a validation chain (gold solves, no-op does not, ...): pass checks= (V12)")
    if registry is not None:
        _register(registry, key, name, version)
    ids: set[str] = set()
    items: list[BenchItem] = []
    for i, f, t in anchors:
        if i in ids:
            raise ValueError(f"duplicate anchor id {i!r} (V25)")
        ids.add(i)
        items.append(BenchItem(i, f, "anchor", t, {"anchor": "fixed"}))
    rejected: list[dict[str, Any]] = []
    fams = sorted(generators)
    yields = {f: {"tried": 0, "passed": 0, "exhausted": False} for f in fams}
    idx = {f: 0 for f in fams}
    streak = {f: 0 for f in fams}
    active = list(fams)
    fresh, k = 0, 0
    while fresh < n_fresh and active:
        fam = active[k % len(active)]
        n = idx[fam]
        idx[fam] += 1
        seed = item_seed(key, fam, seed_ref, n, bench=name, version=version)
        iid = item_id(key, fam, n, bench=name, version=version)
        gen = generators[fam]
        task = gen(seed, item_id=iid) if _accepts_item_id(gen) else gen(seed)
        if inspect.isawaitable(task):  # an async generator hook (M14 campaign B6)
            task = await task
        if id_of(task) != iid:
            try:
                task.id = iid
            except (AttributeError, TypeError):
                pass
        own = id_of(task)
        if own != iid and _spells_seed(own, seed):
            raise ValueError(f"generator {fam!r} returned a task whose id spells its seed and cannot be re-keyed (V3): give the task a settable id "
                             "or accept item_id=")
        if iid in ids:
            raise ValueError(f"duplicate item id {iid!r} (V25)")
        yields[fam]["tried"] += 1
        verdicts: dict[str, str] = {}
        ok = True
        for cname, check in checks:
            try:
                passed, why = await check(task)
            except Exception as e:  # noqa: BLE001 - V25: a check that raises rejects the item, it does not abort the build
                passed, why = False, f"error: {type(e).__name__}: {str(e)[:200]}"
            verdicts[cname] = "pass" if passed else str(why)
            if not passed:
                ok = False
                break
        if ok:
            ids.add(iid)
            items.append(BenchItem(iid, fam, "fresh", task, verdicts))
            fresh += 1
            yields[fam]["passed"] += 1
            streak[fam] = 0
            k += 1
        else:
            rejected.append({"family": fam, "index": n, "id": iid, "checks": verdicts})
            streak[fam] += 1
            if streak[fam] >= max_tries:
                yields[fam]["exhausted"] = True  # the generator cannot produce valid items: reported, never looped forever
                active.remove(fam)
            else:
                k += 1
    s = commit_salt(name, version) if salt is None else bytes(salt)
    bm = Benchmark(name, version, commitment(key, s), items, rejected, s.hex(), yields)
    if fresh < n_fresh and not allow_short:
        raise ShortBenchmark(f"only {fresh} of {n_fresh} fresh items passed validation; yields {yields}", bm)
    return bm
