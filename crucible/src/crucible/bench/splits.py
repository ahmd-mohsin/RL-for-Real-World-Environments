"""Benchmark splits (DESIGN §11.2–11.4): split by group, never by row; salted, stable assignment; held-out time blocks;
OOD axes disjoint by construction; a sealed manifest; and per-model temporal eligibility.

- **Group = lineage component** (M13 audit V1/V2). An item carries *every* lineage root it derives from (the curator's
  ``lineage_of`` list: transitive, and a composite carries all of its components' roots). Items and roots are joined by
  union-find, so the group is the connected component: a composite of a train root and a bench root, or a fork of a
  fork, can no longer leak through the root a caller did not pick. A single ``str`` lineage is still accepted.
- **One decision per component** (V1). Every member's rule-forced split is computed (``showcase`` > ``ood:<axis>`` >
  ``time`` > ``guard``); if any member is forced, the whole component takes the strongest label, so a sibling of an OOD
  or time-held item never trains. Otherwise ``u = HMAC-SHA256(salt, component key) / 2^256`` (key: the component's
  smallest root) and bench iff ``u < bench_frac``: stable when items are added, unpredictable without the salt (FineEnvs
  orders by ``sha1(task_id)``; prime-agent shards by ``crc32``). Row-level ``train_test_split(seed=42)`` (NeMo-Gym
  ``xlam_fc``, FineEnvs SFT scripts) leaks siblings and is the anti-pattern. ``assign`` runs ``check`` on its own
  output and raises on any violation.
- **One split authority** (V13). ``sealed`` takes the curator's lineage seals (``Curator.sealed``: root -> train|bench):
  a component holding a root sealed to bench is held, a component sealed to train trains, and a train seal on a
  component the rules hold out (or a component sealed both ways) is refused, since its material has already trained.
- **Time axis.** Contiguous held-out blocks after ``time_cutoff``, with a guard gap ``[cutoff - guard_days, cutoff)`` so
  neighbours of a held-out block do not train (FineEnvs portsim ``dock.py:31-62``). An undated public-derived item under
  a time split is ``guard`` (V10): it may post-date the cutoff, so it never trains and is never scored. Timestamps are
  RFC 3339; a naive time means UTC (V18).
- **OOD axes.** ``ood={"repo": {"pcl"}}``: components holding a held-out axis value go to ``ood:<axis>``; values are
  casefolded and stripped on both sides, and a bare ``str`` value set is refused (``set("pcl")`` is ``{p, c, l}``: V19).
- **Showcase** tasks (tasks 01–04) and their whole lineage component are never in a published split (ARCHITECTURE §6;
  V8: a derived span of task 04 in bench was a published copy of the showcase).
- **Seal** (V7). ``HMAC-SHA256(salt, canonical JSON of {every id: its split} + the assignment parameters)``
  (``core.canonical``: NFC, escaped, so ids may hold any character): relabelling showcase -> bench, guard -> train,
  bench -> time, or adding or removing any item breaks it. Written before any builder sees test traces
  (``write_splits``); ``verify_seal`` re-derives it.
- **Salt** must be at least 32 bytes (V19: an empty salt made the bench ids public); duplicate ids are refused.
- **Temporal eligibility** (V9). An item derived from a public trace counts for a model only if ``source_time >= cutoff
  + margin`` (half-open, NeMo-Gym LiveCodeBench ``prepare_utils.py:127``). It fails closed: an undated public item, any
  public item for a model whose cutoff is unknown, and an item whose ``source_kind`` was never set (``"unknown"``,
  treated as public) are all ineligible; ``eligibility_report`` lists the models with an unknown cutoff separately.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from crucible.core.canonical import dumps as canonical_dumps

SOURCE_KINDS = ("pr", "issue", "cve", "log", "synthetic", "unknown")
MIN_SALT_BYTES = 32
SEAL_SCHEME = 2


def _nfc(x: str) -> str:
    return unicodedata.normalize("NFC", str(x))


@dataclass
class SplitItem:
    """``lineage``: every root the item derives from (V2: a tuple; a plain str is one root). An item is joined to its own
    id, so another item naming this id as a root shares its component. ``source_kind`` defaults to ``"unknown"`` and is
    then treated as public (V9)."""

    id: str
    lineage: tuple[str, ...] | str = ()
    axes: dict[str, str] = field(default_factory=dict)  # repo, api_family, horizon, harness, ...
    source_time: str | int | float | None = None  # the public trace it derives from: RFC 3339 or epoch s / ms (parse_time)
    source_kind: str = "unknown"  # pr | issue | cve | log | synthetic | unknown
    showcase: bool = False

    def __post_init__(self) -> None:
        self.id = _nfc(self.id)
        lin = (self.lineage,) if isinstance(self.lineage, str) else tuple(self.lineage or ())
        self.lineage = tuple(sorted({_nfc(x) for x in lin if x}))
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(f"source_kind {self.source_kind!r} is not one of {SOURCE_KINDS}")

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(self.lineage) or (self.id,)

    @property
    def public(self) -> bool:
        return self.source_kind != "synthetic"

    @classmethod
    def from_card(cls, card: Any, *, curator: Any = None, source_time: str | None = None, source_kind: str = "unknown",
                  axes: Mapping[str, str] | None = None) -> SplitItem:
        """The adapter from M8/M9 to splits (V13): ``card.splits.showcase``, ``card.forge.seed_ref`` and, with a
        ``curator``, the curator's transitive lineage (``Curator.lineage_of``). Default axes: family and horizon."""
        seed = getattr(getattr(card, "forge", None), "seed_ref", None)
        roots = list(curator.lineage_of(card.id, [seed] if seed else [])) if curator is not None else ([seed] if seed else [])
        if curator is not None and roots == [card.id]:
            roots = []
        ax = {"family": str(card.family), "horizon": str(card.horizon)} if axes is None else dict(axes)
        return cls(card.id, tuple(roots), ax, source_time, source_kind, bool(getattr(getattr(card, "splits", None), "showcase", False)))


def swe_lineage(repo_or_instance: str, number: int | str | None = None) -> str:
    """A canonical lineage root for SWE rows (G2 pick e): ``owner/repo#N`` casefolded, from ``("owner/repo", N)`` or a
    SWE-bench / Multi-SWE instance id ``owner__repo-N``, so one PR imported by two datasets is one lineage root."""
    s = _nfc(repo_or_instance).strip()
    if number is None:
        if "__" not in s or "-" not in s:
            raise ValueError(f"{repo_or_instance!r} is not an owner__repo-N instance id; pass (repo, number)")
        head, _, n = s.rpartition("-")
        owner, _, repo = head.partition("__")
        s, number = f"{owner}/{repo}", n
    if s.count("/") != 1 or not str(number).strip().isdigit():
        raise ValueError(f"cannot form an owner/repo#N lineage from {repo_or_instance!r}, {number!r}")
    return f"{s.casefold()}#{int(str(number).strip())}"


def _u(salt: bytes, key: str) -> float:
    return int.from_bytes(hmac.new(salt, key.encode(), hashlib.sha256).digest(), "big") / 2**256


PLACEHOLDER_BEFORE = datetime(1971, 1, 1, tzinfo=UTC)


def parse_time(x: str | float | datetime | None) -> datetime | None:
    """A source or cutoff time as an aware UTC datetime, or None when undated (V18, G2-10). Accepts RFC 3339 / ISO 8601
    (``Z``, offsets, a space separator; a naive time is UTC: SWE-Lego's ``"2017-09-28 10:00:00"``), a date, and epoch
    seconds or milliseconds as a number or digit string (SWE-rebench-V2's int64 ms). A placeholder before 1971 (epoch 0)
    is undated."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, datetime):
        d = x
    elif isinstance(x, (int, float)) or (isinstance(x, str) and x.strip().lstrip("-").replace(".", "", 1).isdigit()):
        v = float(x)
        d = datetime.fromtimestamp(v / 1000.0 if abs(v) >= 1e11 else v, tz=UTC)
    else:
        s = str(x).strip()
        if not s:
            return None
        if s[-1:] in ("Z", "z"):
            s = s[:-1] + "+00:00"
        d = datetime.fromisoformat(s)
    d = d.replace(tzinfo=UTC) if d.tzinfo is None else d.astimezone(UTC)
    return None if d < PLACEHOLDER_BEFORE else d


def _t(x: str | float) -> datetime:
    d = parse_time(x)
    if d is None:
        raise ValueError(f"time {x!r} is a placeholder or empty")
    return d


def _axis_value(v: Any) -> str:
    return _nfc(str(v)).strip().casefold()


def _norm_ood(ood: Mapping[str, Iterable[str]] | None) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for k, v in (ood or {}).items():
        if isinstance(v, (str, bytes)):
            raise TypeError(f"ood[{k!r}] must be a collection of values, not the string {v!r} (set('pcl') is {{'p','c','l'}})")
        out[str(k)] = {_axis_value(x) for x in v}
    return out


def _params(bench_frac: float, ood: Mapping[str, set[str]], time_cutoff: str | None, guard_days: float) -> dict[str, Any]:
    return {"bench_frac": float(bench_frac), "ood": {k: sorted(v) for k, v in sorted(ood.items())},
            "time_cutoff": None if time_cutoff is None else _t(time_cutoff).isoformat(), "guard_days": float(guard_days)}


def _forced(it: SplitItem, ood: Mapping[str, set[str]], cut: datetime | None, guard_days: float) -> str | None:
    """The split a member's own facts force, or None (free: train or bench by HMAC)."""
    if it.showcase:
        return "showcase"
    axis = next((a for a, held in sorted(ood.items()) if a in it.axes and _axis_value(it.axes[a]) in held), None)
    if axis:
        return f"ood:{axis}"
    if cut is not None:
        t = parse_time(it.source_time)
        if t is not None:
            if t >= cut:
                return "time"
            if t >= cut - timedelta(days=guard_days):
                return "guard"  # too close to the held-out block to train on, too early to test on
        elif it.public:
            return "guard"  # V10: an undated public item may post-date the cutoff
    return None


def _strongest(forced: Iterable[str]) -> str:
    def rank(label: str) -> tuple[int, str]:
        return (0 if label == "showcase" else 1 if label.startswith("ood:") else 2 if label == "time" else 3, label)

    return min(forced, key=rank)


def _components(items: Sequence[SplitItem]) -> list[list[SplitItem]]:
    """Connected components of the item/root graph (union-find; ids and roots share one namespace)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for it in items:
        find(it.id)
        for r in it.lineage:
            ra, rb = find(it.id), find(r)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[SplitItem]] = {}
    for it in items:
        groups.setdefault(find(it.id), []).append(it)
    return [sorted(g, key=lambda x: x.id) for _, g in sorted(groups.items())]


def _component_key(group: Sequence[SplitItem]) -> str:
    return min(r for it in group for r in it.roots)


def _sealed_of(group: Sequence[SplitItem], sealed: Mapping[str, str] | None) -> set[str]:
    if not sealed:
        return set()
    names = {r for it in group for r in it.roots} | {it.id for it in group}
    return {sealed[n] for n in names if n in sealed}


def _validate(items: Sequence[SplitItem], salt: bytes, bench_frac: float) -> None:
    if not isinstance(salt, (bytes, bytearray)) or len(salt) < MIN_SALT_BYTES:
        raise ValueError(f"the split salt must be at least {MIN_SALT_BYTES} secret bytes (V19: a short or empty salt makes the bench ids public)")
    if not 0.0 <= bench_frac <= 1.0:
        raise ValueError(f"bench_frac {bench_frac} is not in [0, 1]")
    seen: set[str] = set()
    dup: set[str] = set()
    for it in items:
        (dup if it.id in seen else seen).add(it.id)
    if dup:
        raise ValueError(f"duplicate item ids would collapse into one split entry: {sorted(dup)[:5]}")


def assign(items: Sequence[SplitItem], *, salt: bytes, bench_frac: float = 0.2, ood: Mapping[str, Iterable[str]] | None = None,
           time_cutoff: str | None = None, guard_days: float = 0.0, sealed: Mapping[str, str] | None = None) -> dict[str, Any]:
    """{"splits": {id: split}, "seal": hmac, "params": ..., "counts": ...}. Splits: train | bench | ood:<axis> | time |
    showcase | guard. ``sealed``: the curator's lineage seals (root -> train | bench). Raises if the result would violate
    ``check`` (it cannot by construction; the check is the proof)."""
    _validate(items, salt, bench_frac)
    oo = _norm_ood(ood)
    cut = _t(time_cutoff) if time_cutoff else None
    out: dict[str, str] = {}
    conflicts: list[str] = []
    for group in _components(items):
        forced = [f for it in group if (f := _forced(it, oo, cut, guard_days))]
        seals = _sealed_of(group, sealed)
        ids = [it.id for it in group]
        if len(seals) > 1:
            conflicts.append(f"component {ids[:4]} has roots sealed to both {sorted(seals)}")
        if forced:
            label = _strongest(forced)
            if "train" in seals:
                conflicts.append(f"component {ids[:4]} is sealed to train by the curator but its rules hold it out as {label}")
        elif seals == {"bench"}:
            label = "bench"
        elif seals == {"train"}:
            label = "train"
        else:
            label = "bench" if _u(salt, _component_key(group)) < bench_frac else "train"
        for i in ids:
            out[i] = label
    if conflicts:
        raise ValueError("split constraints conflict (material already sealed to train cannot be held out): " + "; ".join(conflicts[:5]))
    params = _params(bench_frac, oo, time_cutoff, guard_days)
    problems = check(items, out, ood=oo, time_cutoff=time_cutoff, guard_days=guard_days, sealed=sealed)
    if problems:  # pragma: no cover - the construction above cannot produce one
        raise AssertionError(f"assign produced an invalid split: {problems[:5]}")
    counts: dict[str, int] = {}
    for s in out.values():
        counts[s] = counts.get(s, 0) + 1
    out = dict(sorted(out.items()))
    return {"splits": out, "seal": seal(out, salt, params), "params": params, "counts": dict(sorted(counts.items()))}


def seal(splits: Mapping[str, str], salt: bytes, params: Mapping[str, Any] | None = None) -> str:
    """V7: HMAC over the canonical JSON of the full labelled assignment and its parameters (every item, train and guard
    included; ids NFC and JSON-escaped, so ``"a\\nb"`` and ``["a", "b"]`` differ)."""
    doc = {"scheme": SEAL_SCHEME, "splits": {_nfc(k): str(v) for k, v in splits.items()}, "params": dict(params or {})}
    return hmac.new(bytes(salt), canonical_dumps(doc).encode(), hashlib.sha256).hexdigest()


def verify_seal(splits: Mapping[str, str], salt: bytes, expected: str, params: Mapping[str, Any] | None = None) -> bool:
    return hmac.compare_digest(seal(splits, salt, params), str(expected))


def check(items: Sequence[SplitItem], splits: Mapping[str, str], *, ood: Mapping[str, Iterable[str]] | None = None,
          time_cutoff: str | None = None, guard_days: float = 0.0, sealed: Mapping[str, str] | None = None) -> list[str]:
    """Invariant violations (V8): an unknown, missing or duplicate id; a lineage component in more than one split; a
    component whose rule-forced split (showcase, held-out OOD value, time, guard, undated public under a time split) is
    not the one it got; a train task sharing a held-out OOD value; a showcase task in a published split; a curator seal
    contradicted."""
    problems: list[str] = []
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            problems.append(f"duplicate item id {it.id}")
        seen.add(it.id)
    problems += [f"split entry {i} names no item" for i in sorted(set(splits) - seen)]
    problems += [f"item {i} has no split" for i in sorted(seen - set(splits))]
    oo = _norm_ood(ood)
    cut = _t(time_cutoff) if time_cutoff else None
    for group in _components([it for it in items if it.id in splits]):
        labels = {splits[it.id] for it in group}
        key = _component_key(group)
        if len(labels) > 1:
            kinds = {"train" if s == "train" else "guard" if s == "guard" else "held" for s in labels}
            if {"train", "held"} <= kinds:
                problems.append(f"lineage {key} straddles train and held-out")
            problems.append(f"lineage {key} spans splits {sorted(labels)}")
        forced = [f for it in group if (f := _forced(it, oo, cut, guard_days))]
        if forced:
            want = _strongest(forced)
            problems += [f"{it.id} is in {splits[it.id]} but its lineage {key} is forced to {want}" for it in group if splits[it.id] != want]
        else:
            problems += [f"{it.id} is in {splits[it.id]}, which is not a free split (train | bench)" for it in group if splits[it.id] not in ("train", "bench")]
        for it in group:
            if splits[it.id] == "train":
                problems += [f"train task {it.id} shares held-out {axis}={it.axes.get(axis)}" for axis, held in sorted(oo.items())
                             if axis in it.axes and _axis_value(it.axes[axis]) in held]
                st = parse_time(it.source_time)
                late = cut is not None and (st is None and it.public or st is not None and st >= cut - timedelta(days=guard_days))
                if late:
                    problems.append(f"train task {it.id} is in the time block or its guard gap (or undated and public)")
            if it.showcase and splits[it.id] != "showcase":
                problems.append(f"showcase task {it.id} is in split {splits[it.id]}")
        seals = _sealed_of(group, sealed)
        if len(seals) > 1:
            problems.append(f"lineage {key} is sealed to both {sorted(seals)}")
        problems += [f"{it.id} is in {splits[it.id]} but its lineage is sealed to {s}" for s in sorted(seals) for it in group
                     if (s == "train" and splits[it.id] != "train") or (s == "bench" and splits[it.id] == "train")]
    return problems


def write_splits(path: Path, result: Mapping[str, Any]) -> Path:
    """Write ``splits.json`` (splits, params, seal; never the salt) before any builder sees test traces (DESIGN §11.2)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"scheme": SEAL_SCHEME, "splits": dict(result["splits"]), "params": dict(result["params"]), "seal": result["seal"],
           "counts": dict(result.get("counts", {}))}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def verify_file(path: Path, salt: bytes) -> bool:
    doc = json.loads(Path(path).read_text())
    return verify_seal(doc["splits"], salt, doc["seal"], doc.get("params"))


def eligible(item: SplitItem, model_cutoff: str | None, *, margin_days: float = 0.0) -> bool:
    """Temporal decontamination, failing closed (V9): a synthetic undated item is always eligible; a public (or
    ``unknown``-kind) item only when it is dated, the model's cutoff is known, and it post-dates cutoff + margin."""
    st, mc = parse_time(item.source_time), parse_time(model_cutoff)
    if item.source_kind == "synthetic" and st is None:
        return True  # nothing public to have trained on
    if st is None or mc is None:
        return False
    return st >= mc + timedelta(days=margin_days)


def eligible_subsets(items: Sequence[SplitItem], cutoffs: Mapping[str, str | None], *, margin_days: float = 0.0) -> dict[str, list[str]]:
    """Per model, the ids it may be scored on (report the size; compare models on IRT ability, not raw pass rates)."""
    return {m: sorted(it.id for it in items if eligible(it, c, margin_days=margin_days)) for m, c in sorted(cutoffs.items())}


def eligibility_report(items: Sequence[SplitItem], cutoffs: Mapping[str, str | None], *, margin_days: float = 0.0) -> dict[str, Any]:
    """``eligible_subsets`` plus the models whose cutoff is unknown (scored on synthetic items only) and the undated
    public items (never scored), reported separately so a shrunken subset is visible (V9)."""
    return {"eligible": eligible_subsets(items, cutoffs, margin_days=margin_days),
            "cutoff_unknown": sorted(m for m, c in cutoffs.items() if c is None),
            "undated_public": sorted(it.id for it in items if it.public and parse_time(it.source_time) is None)}
