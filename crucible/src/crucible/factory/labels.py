"""The label and decision store (M17; 05 §3 "two labellers per item ... never edited, only superseded", §10 decision store).

One append-only JSONL file holds ``Label`` and ``Decision`` records. Rules, all enforced at append time under an
exclusive file lock (so two processes cannot race a rule):

- **No edits.** The API only appends. Each line carries ``seq``, the hash of the previous line (``prev``) and its own
  hash (``h``): a changed, removed-from-the-middle or reordered line breaks the chain and every read raises
  ``CorruptStore`` naming the line. (A cut-off tail cannot be told from a never-written record and is not detectable.)
- **Crash-safe.** A record is one ``write`` of one line ending in ``\\n``, then ``fsync``. A last line without its
  newline is a torn write that was never acknowledged: readers ignore it and the next append truncates it. A torn line
  *inside* the file is corruption, not a crash, and raises.
- **Supersede chains.** A label changes only by a new label whose ``supersedes`` names the old one. The superseder must
  be the same labeller on the same item, the target must be a label and still current (no forks: one chain per
  labeller and item). A second label by the same labeller on the same item that does not supersede is refused.
- **Item-hash binding.** A label names the sha256 of the item content it judged. ``add_label(current_item_sha256=...)``
  refuses a label on a stale hash; ``effective(..., current_sha256=...)`` returns only labels still bound to the
  current content and ``stale(...)`` the rest. A changed item needs fresh labels.
- **Pairing and agreement.** ``needs_labels`` lists items short of ``raters`` distinct fresh labellers; ``agreement``
  computes Cohen's kappa for two labellers and Fleiss' kappa for more, over fresh labels only.

Also here: ``wilson`` and ``clopper_pearson`` intervals (every rate in M17 carries its denominator and an upper bound).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

from pydantic import ValidationError

from crucible.core import canonical
from crucible.core.ids import new_id, sha256_text
from crucible.core.prov import utc_now
from crucible.factory.cards import Decision, DecisionAction, Label

try:  # POSIX advisory locking; the stores are local files
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

GENESIS = "0" * 64


class LabelStoreError(ValueError):
    """Base class: the store refused an append or a read."""


class CorruptStore(LabelStoreError):
    """The file is not a valid chain (a changed, torn-in-the-middle or reordered line)."""


class StaleItem(LabelStoreError):
    """The label is bound to an item hash that is not the item's current hash."""


class SupersedeError(LabelStoreError):
    """A supersede that is forged, crosses labellers or items, or forks a chain."""


class DuplicateLabel(LabelStoreError):
    """The labeller already has a label on this item: supersede it instead of adding another."""


def item_sha256(content: Any) -> str:
    """The hash an item is bound to: sha256 of its canonical JSON (a str is hashed as is)."""
    return sha256_text(content if isinstance(content, str) else canonical.dumps(content))


def _line_hash(prev: str, seq: int, kind: str, rec: Mapping[str, Any]) -> str:
    return sha256_text(f"{prev}\n{seq}\n{kind}\n{canonical.dumps(rec)}")


@dataclass
class _State:
    labels: dict[str, Label]
    decisions: list[Decision]
    superseded_by: dict[str, str]
    seq: int
    head: str
    good_bytes: int  # length of the valid prefix: every complete line
    torn_bytes: int  # a final line without its newline


def _parse(data: bytes) -> _State:
    labels: dict[str, Label] = {}
    decisions: list[Decision] = []
    decision_ids: set[str] = set()
    superseded_by: dict[str, str] = {}
    prev, seq = GENESIS, 0
    end = data.rfind(b"\n") + 1
    for n, raw in enumerate(data[:end].split(b"\n")[:-1], 1):
        try:
            line = json.loads(raw)
            kind, rec = line["kind"], line["rec"]
            if line["seq"] != seq + 1 or line["prev"] != prev or line["h"] != _line_hash(prev, seq + 1, kind, rec):
                raise CorruptStore(f"line {n}: the hash chain is broken (a line was changed, removed or reordered)")
            if kind == "label":
                lab = Label.model_validate(rec)
                if lab.id in labels or lab.id in decision_ids:
                    raise CorruptStore(f"line {n}: duplicate id {lab.id!r}")
                if lab.supersedes:
                    target = labels.get(lab.supersedes)
                    if target is None or target.item != lab.item or target.labeller != lab.labeller or lab.supersedes in superseded_by:
                        raise CorruptStore(f"line {n}: label {lab.id!r} has an invalid supersede of {lab.supersedes!r}")
                    superseded_by[lab.supersedes] = lab.id
                labels[lab.id] = lab
            elif kind == "decision":
                dec = Decision.model_validate(rec)
                if dec.id in labels or dec.id in decision_ids:
                    raise CorruptStore(f"line {n}: duplicate id {dec.id!r}")
                decision_ids.add(dec.id)
                decisions.append(dec)
            else:
                raise CorruptStore(f"line {n}: unknown record kind {kind!r}")
        except CorruptStore:
            raise
        except (ValueError, KeyError, TypeError, ValidationError) as e:
            raise CorruptStore(f"line {n}: not a valid record ({type(e).__name__})") from None
        prev, seq = line["h"], seq + 1
    return _State(labels, decisions, superseded_by, seq, prev, end, len(data) - end)


class LabelStore:
    """Append-only, hash-chained, crash-safe JSONL store for ``Label`` and ``Decision``."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._cache: tuple[tuple[int, int, int], _State] | None = None

    # --- reading ---------------------------------------------------------------------------------------------------

    def _state(self) -> _State:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return _parse(b"")
        key = (st.st_ino, st.st_size, st.st_mtime_ns)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        state = _parse(self.path.read_bytes())
        self._cache = (key, state)
        return state

    def verify(self) -> int:
        """Re-read the whole file and check the chain; the number of records, or ``CorruptStore``."""
        self._cache = None
        return self._state().seq

    @property
    def torn_bytes(self) -> int:
        """Bytes of an unacknowledged torn tail currently ignored (the next append removes them)."""
        return self._state().torn_bytes

    def get_label(self, label_id: str) -> Label | None:
        return self._state().labels.get(label_id)

    def labels(self, item: str | None = None, labeller: str | None = None, *, include_superseded: bool = False) -> list[Label]:
        s = self._state()
        return [lab for lab in s.labels.values()
                if (item is None or lab.item == item) and (labeller is None or lab.labeller == labeller)
                and (include_superseded or lab.id not in s.superseded_by)]

    def chain(self, label_id: str) -> list[Label]:
        """The supersede chain containing ``label_id``, oldest first."""
        s = self._state()
        if label_id not in s.labels:
            raise KeyError(label_id)
        first = s.labels[label_id]
        while first.supersedes:
            first = s.labels[first.supersedes]
        out = [first]
        while out[-1].id in s.superseded_by:
            out.append(s.labels[s.superseded_by[out[-1].id]])
        return out

    def is_superseded(self, label_id: str) -> bool:
        return label_id in self._state().superseded_by

    def effective(self, item: str, current_sha256: str | None = None) -> list[Label]:
        """The current label of each labeller on ``item``; with ``current_sha256`` only those bound to that content."""
        return [lab for lab in self.labels(item) if current_sha256 is None or lab.item_sha256 == current_sha256]

    def stale(self, item: str, current_sha256: str) -> list[Label]:
        """Current labels judged on content that is no longer the item's."""
        return [lab for lab in self.labels(item) if lab.item_sha256 != current_sha256]

    def needs_labels(self, items: Mapping[str, str], *, raters: int = 2, labeller: str | None = None) -> list[str]:
        """Items (``{id: current sha256}``, in the mapping's order) with fewer than ``raters`` distinct fresh labellers;
        with ``labeller``, only those that labeller has no fresh label on."""
        out = []
        for item, sha in items.items():
            fresh = self.effective(item, sha)
            if len({lab.labeller for lab in fresh}) >= raters:
                continue
            if labeller is not None and any(lab.labeller == labeller for lab in fresh):
                continue
            out.append(item)
        return out

    def agreement(self, items: Mapping[str, str], raters: Sequence[str] | None = None) -> Agreement:
        """Kappa over ``items`` (``{id: current sha256}``) using fresh labels. ``raters`` names the labellers (default:
        every labeller with a fresh label on these items); only items every one of them labelled count. Two raters give
        Cohen's kappa, more give Fleiss'. ``kappa`` is None when it is undefined (fewer than two raters, no shared
        items, or chance agreement is 1)."""
        fresh = {item: {lab.labeller: lab.value for lab in self.effective(item, sha)} for item, sha in items.items()}
        names = sorted(raters) if raters is not None else sorted({r for votes in fresh.values() for r in votes})
        shared = [item for item in items if names and all(r in fresh[item] for r in names)]
        if len(names) < 2 or not shared:
            return Agreement("none", None, len(shared), tuple(names), None)
        if len(names) == 2:
            a = [fresh[i][names[0]] for i in shared]
            b = [fresh[i][names[1]] for i in shared]
            po = sum(x == y for x, y in zip(a, b)) / len(a)
            return Agreement("cohen", cohen_kappa(a, b), len(shared), tuple(names), po)
        cats = sorted({fresh[i][r] for i in shared for r in names})
        table = [[sum(fresh[i][r] == c for r in names) for c in cats] for i in shared]
        n = len(names)
        po = sum((sum(c * c for c in row) - n) / (n * (n - 1)) for row in table) / len(table)
        return Agreement("fleiss", fleiss_kappa(table), len(shared), tuple(names), po)

    def decisions(self, *, subject: str | None = None, actor: str | None = None, action: DecisionAction | None = None,
                  criterion: str | None = None, recipe_version: str | None = None) -> list[Decision]:
        """Decisions in append order, filtered by any of the given fields (``criterion`` matches ``criteria``)."""
        return [d for d in self._state().decisions
                if (subject is None or d.subject == subject) and (actor is None or d.actor == actor)
                and (action is None or d.action == action) and (criterion is None or criterion in d.criteria)
                and (recipe_version is None or d.recipe_version == recipe_version)]

    def latest_decision(self, subject: str) -> Decision | None:
        found = self.decisions(subject=subject)
        return found[-1] if found else None

    # --- appending -------------------------------------------------------------------------------------------------

    def add_label(self, *, item: str, item_sha256: str, labeller: str, value: str, reason: str = "",
                  supersedes: str | None = None, current_item_sha256: str | None = None,
                  id: str | None = None, at: str | None = None) -> Label:
        """Append a label. ``current_item_sha256`` is the item's current hash as the caller knows it: a label bound to
        any other hash is refused (``StaleItem``)."""
        label = Label(id=id or new_id("lbl"), item=item, item_sha256=item_sha256, labeller=labeller, value=value,
                      reason=reason, at=at or utc_now(), supersedes=supersedes)
        if current_item_sha256 is not None and current_item_sha256 != item_sha256:
            raise StaleItem(f"label on item {item!r} is bound to {item_sha256[:12]}, but the item is now {current_item_sha256[:12]}")

        def check(s: _State) -> None:
            if label.id in s.labels or any(d.id == label.id for d in s.decisions):
                raise LabelStoreError(f"id {label.id!r} already exists")
            if supersedes is not None:
                target = s.labels.get(supersedes)
                if target is None:
                    raise SupersedeError(f"supersedes {supersedes!r}, which is not a label in this store")
                if target.item != label.item:
                    raise SupersedeError(f"supersedes a label on item {target.item!r}, not {label.item!r}")
                if target.labeller != label.labeller:
                    raise SupersedeError(f"{label.labeller!r} cannot supersede a label by {target.labeller!r}")
                if supersedes in s.superseded_by:
                    raise SupersedeError(f"label {supersedes!r} is already superseded by {s.superseded_by[supersedes]!r} (no forks)")
            elif any(lab.item == label.item and lab.labeller == label.labeller and lab.id not in s.superseded_by
                     for lab in s.labels.values()):
                raise DuplicateLabel(f"{label.labeller!r} already labelled {label.item!r}: supersede that label")

        self._append("label", label, check)
        return label

    def add_decision(self, *, subject: str, actor: str, action: DecisionAction, reason: str, criteria: Iterable[str] = (),
                     recipe_version: str, id: str | None = None, at: str | None = None) -> Decision:
        dec = Decision(id=id or new_id("dec"), subject=subject, actor=actor, action=action, reason=reason,
                       criteria=list(criteria), recipe_version=recipe_version, at=at or utc_now())

        def check(s: _State) -> None:
            if dec.id in s.labels or any(d.id == dec.id for d in s.decisions):
                raise LabelStoreError(f"id {dec.id!r} already exists")

        self._append("decision", dec, check)
        return dec

    def _append(self, kind: str, record: Label | Decision, check: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            with os.fdopen(os.dup(fd), "rb") as f:
                data = f.read()
            state = _parse(data)  # inside the lock: the rules see every record already committed
            if state.torn_bytes:
                os.ftruncate(fd, state.good_bytes)  # an unacknowledged torn write
            check(state)
            rec = record.model_dump(mode="json", exclude_none=True)
            seq = state.seq + 1
            line = canonical.dumps({"seq": seq, "prev": state.head, "kind": kind, "rec": rec,
                                    "h": _line_hash(state.head, seq, kind, rec)}) + "\n"
            buf = line.encode("utf-8")
            while buf:
                buf = buf[os.write(fd, buf):]
            os.fsync(fd)
        finally:
            os.close(fd)  # releases the lock
        if new_file:
            with contextlib.suppress(OSError):
                dfd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
        self._cache = None


@dataclass(frozen=True)
class Agreement:
    method: str  # "cohen" | "fleiss" | "none"
    kappa: float | None
    n_items: int
    raters: tuple[str, ...]
    observed: float | None  # raw agreement


# --- agreement and intervals --------------------------------------------------------------------------------------


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float | None:
    """Cohen's kappa for two raters: (po - pe) / (1 - pe). None if undefined (no items, or pe == 1)."""
    if len(a) != len(b):
        raise ValueError("the two raters must rate the same items")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return None if math.isclose(pe, 1.0) else (po - pe) / (1 - pe)


def fleiss_kappa(table: Sequence[Sequence[int]]) -> float | None:
    """Fleiss' kappa. ``table[i][j]`` is how many raters put item ``i`` in category ``j``; every row sums to the same
    ``n >= 2``. None if undefined (no items, or chance agreement is 1)."""
    if not table:
        return None
    n = sum(table[0])
    if n < 2 or any(sum(row) != n for row in table) or any(c < 0 for row in table for c in row):
        raise ValueError("every item needs the same number (>= 2) of ratings, none negative")
    N = len(table)
    pj = [sum(row[j] for row in table) / (N * n) for j in range(len(table[0]))]
    pbar = sum((sum(c * c for c in row) - n) / (n * (n - 1)) for row in table) / N
    pe = sum(p * p for p in pj)
    return None if math.isclose(pe, 1.0) else (pbar - pe) / (1 - pe)


def wilson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for ``successes`` out of ``n``; ``(0, 1)`` for ``n == 0``."""
    _check_counts(successes, n)
    if n == 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    # the interval reaches 0 at k == 0 and 1 at k == n exactly; floating point must not say 1e-17 short of it
    return (0.0 if successes == 0 else max(0.0, centre - half), 1.0 if successes == n else min(1.0, centre + half))


def clopper_pearson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval: the lower bound is the p at which P(X >= k) = alpha/2, the upper the p at
    which P(X <= k) = alpha/2. ``(0, 1)`` for ``n == 0``."""
    _check_counts(successes, n)
    if n == 0:
        return (0.0, 1.0)
    alpha = 1 - confidence
    k = successes
    lo = 0.0 if k == 0 else _beta_quantile(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_quantile(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


def _check_counts(successes: int, n: int) -> None:
    if not (0 <= successes <= n):
        raise ValueError(f"need 0 <= successes <= n, got {successes}/{n}")


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(ln_front) * _betacf(a, b, x) / a
    return 1 - math.exp(ln_front) * _betacf(b, a, 1 - x) / b


def _beta_quantile(q: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return (lo + hi) / 2

