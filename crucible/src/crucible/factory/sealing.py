"""Sealing the held-out benchmark (study 08 §8): before H1–H3 are published, prove they are apart from training.

``seal(train, heldout)`` checks, over any two collections of windows (a window store, a list, or anything with
``windows``):

- **disjoint repos**: no repository has a window on both sides;
- **time split**: every held-out window is dated after every training window (and, given ``model_cutoff``, after the
  base model's cutoff, reported separately: "where possible"); each violation is listed with both dates;
- **contamination**: ``bench.contamination`` fingerprints the held-out prompts and oracle diffs (template boilerplate
  subtracted) and scans the training prompts and diffs for exact n-gram, code-substring, near-duplicate and containment
  hits.

A window is read through accessors (defaults: ``repo`` | ``repo_id``; ``date_b`` | ``end_date`` | ``b_date`` | ``date``;
``prompt`` | ``instruction``; ``diff`` | ``oracle_diff`` | ``patch``; ``id`` | ``window_id``), attributes or mapping keys.
Dates compare as ISO-8601 strings (or ``date``/``datetime``). ``assert_sealed`` raises ``SealError`` on any finding;
``publish_heldout`` publishes the held-out cards as their own split through ``factory.publish`` (which refuses a card in two
splits) only with a clean report.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.bench import contamination


class SealError(ValueError):
    """The held-out set is not sealed from training."""


def _get(w: Any, names: Sequence[str]) -> Any:
    for n in names:
        v = w.get(n) if isinstance(w, Mapping) else getattr(w, n, None)
        if v is not None:
            return v
    return None


def _acc(names: Sequence[str]) -> Callable[[Any], Any]:
    return lambda w: _get(w, names)


REPO = _acc(("repo", "repo_id"))
DATE = _acc(("date_b", "end_date", "b_date", "date"))
PROMPT = _acc(("prompt", "instruction"))
DIFF = _acc(("diff", "oracle_diff", "patch"))
WID = _acc(("id", "window_id"))


def _windows(store: Any) -> list[Any]:
    if hasattr(store, "windows") and not isinstance(store, Mapping):
        ws = store.windows
        return list(ws() if callable(ws) else ws)
    return list(store)


def _iso(d: Any) -> str | None:
    if d is None:
        return None
    return d.isoformat() if hasattr(d, "isoformat") else str(d)


@dataclass
class SealReport:
    shared_repos: list[str] = field(default_factory=list)
    time_violations: list[dict[str, Any]] = field(default_factory=list)
    undated: list[str] = field(default_factory=list)
    cutoff_violations: list[dict[str, Any]] = field(default_factory=list)  # reported, not fatal unless strict_cutoff
    contamination: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    strict_cutoff: bool = False

    @property
    def ok(self) -> bool:
        return not (self.shared_repos or self.time_violations or self.undated or self.contamination or (self.strict_cutoff and self.cutoff_violations))

    def problems(self) -> list[str]:
        out = []
        if self.shared_repos:
            out.append(f"repos on both sides: {self.shared_repos}")
        if self.time_violations:
            out.append(f"{len(self.time_violations)} held-out windows not after the last training window (first: {self.time_violations[0]})")
        if self.undated:
            out.append(f"undated windows: {self.undated[:5]}")
        if self.contamination:
            out.append(f"{len(self.contamination)} contamination hits (first: {self.contamination[0]})")
        if self.strict_cutoff and self.cutoff_violations:
            out.append(f"{len(self.cutoff_violations)} held-out windows not after the model cutoff")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "shared_repos": self.shared_repos, "time_violations": self.time_violations, "undated": self.undated,
                "cutoff_violations": self.cutoff_violations, "contamination": self.contamination, "counts": self.counts}


def seal(train: Any, heldout: Any, *, repo: Callable[[Any], Any] = REPO, date: Callable[[Any], Any] = DATE, prompt: Callable[[Any], Any] = PROMPT,
         diff: Callable[[Any], Any] = DIFF, wid: Callable[[Any], Any] = WID, model_cutoff: Any = None, strict_cutoff: bool = False,
         contamination_check: bool = True, template: Iterable[str] = (), **scan_kw: Any) -> SealReport:
    """``template``: boilerplate every prompt shares (the harness's STATUS.json / GIVE_UP paragraph), subtracted before the
    scan so it is not counted as overlap; ``scan_kw`` go to ``bench.contamination.scan``."""
    tr, ho = _windows(train), _windows(heldout)
    rep = SealReport(strict_cutoff=strict_cutoff, counts={"train": len(tr), "heldout": len(ho)})

    def name(w: Any, i: int, side: str) -> str:
        return str(wid(w) or f"{side}[{i}]")

    rep.shared_repos = sorted({str(repo(w)) for w in tr if repo(w) is not None} & {str(repo(w)) for w in ho if repo(w) is not None})
    t_dates = [(name(w, i, "train"), _iso(date(w))) for i, w in enumerate(tr)]
    h_dates = [(name(w, i, "heldout"), _iso(date(w))) for i, w in enumerate(ho)]
    rep.undated = [n for n, d in t_dates + h_dates if d is None]
    latest = max((d, n) for n, d in t_dates if d is not None) if any(d for _, d in t_dates) else None
    cut = _iso(model_cutoff)
    for n, d in h_dates:
        if d is None:
            continue
        if latest is not None and d <= latest[0]:
            rep.time_violations.append({"heldout": n, "date": d, "latest_train": latest[1], "train_date": latest[0]})
        if cut is not None and d <= cut:
            rep.cutoff_violations.append({"heldout": n, "date": d, "model_cutoff": cut})
    if contamination_check and ho and tr:
        bench = contamination.fingerprints([(name(w, i, "heldout"), str(prompt(w) or ""), str(diff(w) or "")) for i, w in enumerate(ho)],
                                           extra_template=list(template))
        corpus = [(f"{name(w, i, 'train')}:{part}", str(x)) for i, w in enumerate(tr) for part, x in (("prompt", prompt(w)), ("diff", diff(w))) if x]
        rep.contamination = contamination.scan(bench, corpus, **scan_kw)
    return rep


def assert_sealed(train: Any, heldout: Any, **kw: Any) -> SealReport:
    rep = seal(train, heldout, **kw)
    if not rep.ok:
        raise SealError("; ".join(rep.problems()))
    return rep


def publish_heldout(cards: Sequence[Any], train_ids: Iterable[str], heldout_ids: Iterable[str], report: SealReport, *, train_split: str = "train",
                    heldout_split: str = "heldout", **kwargs: Any) -> Any:
    """Publish one family's cards with the held-out cards as their own split (``factory.publish.publish_family``, which
    refuses a card in two splits), only on a clean ``report``."""
    if not report.ok:
        raise SealError("refusing to publish an unsealed held-out split: " + "; ".join(report.problems()))
    from crucible.factory.publish import publish_family

    return publish_family(cards, {train_split: list(train_ids), heldout_split: list(heldout_ids)}, **kwargs)
