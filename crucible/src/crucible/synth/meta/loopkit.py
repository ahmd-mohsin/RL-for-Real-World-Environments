"""What the per-task loop (``rsi.TaskLoop``) needs to run on a stored, re-certified pool (DESIGN §7.7 item 23).

- **Full re-certification after every edit** (``Recertifier``): the deterministic classes with the held-out split, ORACLE
  where a World registers policy truth, MUT-HO and the archive regressions (student-probe counterexamples included)
  in process (``recert.det_run``), then the paid classes (ALT, ATK, the LEAK probe) with the calibrated panels
  (``recert.certify_paid``). ``recheck`` is G3/T5/RS3 (an edit stands only if its version is certified or provisional),
  ``alt_solves`` is G6 (an independent ALT family passed the version), ``validity`` is the loop's P1 precondition
  (``rsi.TaskLoop(validity=)``): a rejected version is repaired with its confirmed, *shown* counterexamples (MUT-HO
  findings are never shown to a repairer), an on-hold one waits for a person. Results are cached per (id, version,
  digest) and logged; the pool's starting statuses come from the manifest (``seed``).
- **Cells** (``label_cell``): the pool's tasks carry free-text skill labels from the proposer, the grid's skills are the
  domain's ``DomainSpec`` skills. A label is mapped to the spec skill whose id and description share the most
  informative tokens with it (IDF over the domain's skills); ties and no overlap give ``<domain>:unmapped``. RESKILL's
  labels are spec skill ids, which map to themselves, so the pool's cell histogram (G7, RS5) is over one label space.
- ``DomainClassifier``: the outcome classifier per domain (each domain has its own calibrated panel).
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.archive import Archive
from crucible.synth.falsify.base import Counterexample, FalsifyContext
from crucible.synth.task import Task

STOP = {"a", "an", "the", "of", "to", "and", "or", "with", "by", "for", "in", "on", "under", "its", "it", "is", "be", "that", "this",
        "from", "as", "at", "into", "one", "decide", "decision", "task", "ap", "invoice", "invoices"}


def _tokens(text: str) -> set[str]:
    out = set()
    for w in re.split(r"[^a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text)).lower().replace("_", " ")):
        if len(w) < 3 or w in STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def label_cell(domain: str, text: str, skills: Sequence[Mapping[str, str]]) -> str:
    """The grid cell of a skill label (module docstring)."""
    ids = [str(s["id"]) for s in skills]
    if text in ids:
        return f"{domain}:{text}"
    toks = {s["id"]: _tokens(f"{s['id']} {s.get('description', '')}") for s in skills}
    df = Counter(t for ts in toks.values() for t in ts)
    n = max(1, len(skills))
    q = _tokens(text)
    score = {k: sum(math.log((n + 1) / df[t]) for t in q & ts) for k, ts in toks.items()}
    if not score:
        return f"{domain}:unmapped"
    best = max(score.values())
    top = [k for k, v in score.items() if v == best]
    return f"{domain}:{top[0]}" if best > 0 and len(top) == 1 else f"{domain}:unmapped"


class CertRow(CrucibleModel):
    task_id: str
    version: int
    digest: str
    status: str
    reasons: list[str] = Field(default_factory=list)
    alt_passes: int = 0
    alt_n: int = 0
    source: str = "recertified"  # recertified | manifest
    seconds: float = 0.0


class Recertifier:
    """Full re-certification of a task version (module docstring). ``paid(task)`` → (panel, twin_panel, alt agents,
    atk agents, extra classes) for that task; ``builders_of(task)`` its builder families (ALT and ATK exclude them)."""

    def __init__(self, *, make_ctx: Callable[[], FalsifyContext], archive: Archive, holdout: float, truths: Callable[[Task], Sequence[str]],
                 paid: Callable[[Task], Awaitable[tuple[Any, Any, Sequence[Any], Sequence[Any], Sequence[Any]]]] | None,
                 builders_of: Callable[[Task], Sequence[str]], log: Path | None = None, extra_replay: Sequence[Any] = (),
                 det_fn: Callable[[Task], Awaitable[dict[str, Any]]] | None = None):
        self.make_ctx, self.archive, self.holdout, self.truths, self.paid, self.builders_of = make_ctx, archive, holdout, truths, paid, builders_of
        self.log, self.extra_replay = log, list(extra_replay)
        self.det_fn = det_fn  # phase 1 elsewhere (a ``spawn`` worker running ``recert.det_job``): CPU off the event loop
        self.rows: dict[tuple[str, int, str], CertRow] = {}
        self.dets: dict[tuple[str, int, str], dict[str, Any]] = {}

    @staticmethod
    def key(task: Task) -> tuple[str, int, str]:
        return (task.id, task.version, task.digest())

    def seed(self, task: Task, status: str, reasons: Sequence[str] = (), alt: tuple[int, int] = (0, 0)) -> None:
        self.rows[self.key(task)] = CertRow(task_id=task.id, version=task.version, digest=task.digest(), status=status, reasons=list(reasons),
                                            alt_passes=alt[0], alt_n=alt[1], source="manifest")

    async def det(self, task: Task) -> dict[str, Any]:
        from crucible.synth.recert import det_run

        k = self.key(task)
        if k not in self.dets and self.det_fn is not None:
            self.dets[k] = await self.det_fn(task)
        if k not in self.dets:
            ctx = self.make_ctx()
            try:
                self.dets[k] = await det_run(task, ctx, self.holdout, self.truths(task), self.archive, mut_all=False, extra_replay=self.extra_replay)
            finally:
                await ctx.close()
        return self.dets[k]

    async def certify(self, task: Task) -> CertRow:
        from crucible.synth.recert import Item, OldCert, certify_paid, det_verdict, paid_rows, status_of

        k = self.key(task)
        if k in self.rows:
            return self.rows[k]
        t0 = time.monotonic()
        doc = await self.det(task)
        st, why, _ = det_verdict(task, doc)
        row = CertRow(task_id=task.id, version=task.version, digest=k[2], status="rejected", reasons=list(why))
        if st != "rejected":
            if self.paid is None:
                row.status, row.reasons = ("provisional" if why else "certified"), list(why)
            else:
                panel, twin, alt, atk, extra = await self.paid(task)
                item = Item(key=f"taskloop::{task.id}@v{task.version}", source="taskloop", kinds=["taskloop"], run_dir="", n=0, task=task,
                            old=OldCert(status=""), builders=list(self.builders_of(task)))
                ctx = self.make_ctx()
                try:
                    rep, sink = await certify_paid(item, doc, ctx=ctx, archive=self.archive, holdout=self.holdout, truths=self.truths(task), panel=panel,
                                                   twin_panel=twin, alt=alt, atk=atk, extra=extra)
                finally:
                    await ctx.close()
                row.status, row.reasons = status_of(rep)
                att = [r for r in paid_rows(rep, sink) if r["cls"] == "ALT"]
                row.alt_passes, row.alt_n = sum(1 for r in att if r["passed"]), len(att)
        row.seconds = round(time.monotonic() - t0, 1)
        self.rows[k] = row
        if self.log is not None:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open("a") as fh:
                fh.write(json.dumps(row.model_dump(mode="json")) + "\n")
        return row

    async def readjudicate(self, task: Task) -> None:
        """Forget the stored status of ``task`` (a manifest row or an earlier certification), so the next ``validity`` or
        ``recheck`` re-certifies it in full: R-panel's ``panel_decide`` once the panel is calibrated."""
        self.rows.pop(self.key(task), None)

    async def recheck(self, task: Task) -> list[str]:
        """G3 / T5 / RS3: [] when the version is certified or provisional, else its reasons."""
        row = await self.certify(task)
        return [] if row.status in ("certified", "provisional") else [f"re-certification {row.status}: {', '.join(row.reasons[:5])}"[:300]]

    async def alt_solves(self, task: Task) -> bool | None:
        """G6: an independent ALT family (non-student, non-builder) passed this version in its certification."""
        row = await self.certify(task)
        return None if row.alt_n == 0 else row.alt_passes > 0

    async def validity(self, task: Task) -> tuple[str, list[Counterexample]]:
        """P1: the version's status, and for a rejected one its confirmed counterexamples a repairer may see (the shown
        classes and ORACLE, the re-opened archive records; never MUT-HO)."""
        row = await self.certify(task)
        if row.status != "rejected":
            return row.status, []
        from crucible.synth.falsify.base import ClassResult

        doc = await self.det(task)
        cxs: list[Counterexample] = []
        for r in doc["results"]:
            c = ClassResult.model_validate(r)
            if c.cls == "MUT-HO" or not c.applicable:
                continue
            cxs += [Counterexample.of(task, f) for f in c.confirmed]
        reopened = set((doc.get("regress") or {}).get("reopened") or [])
        cxs += [c for c in self.archive.records(task.domain, task.id) if c.id in reopened]
        return "rejected", cxs


class TaskKeyedContext(FalsifyContext):
    """A ``FalsifyContext`` whose World cache is keyed by (task id, WorldRef): steps of different tasks run concurrently
    (``run_typed_round(concurrency=)``), and two tasks on one World instance would otherwise reset and run one shared
    World in turn (their calls interleaving in one start state). One task's step is sequential. At most ``max_worlds``
    Worlds stay open (least recently used closed first)."""

    def __init__(self, *args: Any, max_worlds: int = 48, **kw: Any):
        super().__init__(*args, **kw)
        self.max_worlds = max_worlds
        self._used: dict[str, float] = {}

    async def world(self, task: Task) -> Any:
        import inspect

        k = f"{task.id}|{task.world.model_dump_json()}"
        self._used[k] = time.monotonic()
        if k not in self._worlds:
            if len(self._worlds) >= self.max_worlds:
                old = min((x for x in self._worlds if x != k), key=lambda x: self._used.get(x, 0.0))
                w = self._worlds.pop(old)
                c = getattr(w, "close", None)
                if c is not None:
                    res = c()
                    if inspect.isawaitable(res):
                        await res
            if self.make_world is not None:
                w = self.make_world(task)
                self._worlds[k] = await w if inspect.isawaitable(w) else w
            else:
                self._worlds[k] = await task.world.build()
        return self._worlds[k]


class DomainClassifier:
    """``rsi.OutcomeClassifier`` per domain: each domain's ``LearnClassifier`` (its own calibrated panel)."""

    def __init__(self, by_domain: Mapping[str, Any]):
        self.by = dict(by_domain)

    async def classify(self, task: Task, rollouts: Sequence[Any]) -> list[Any]:
        return await self.by[task.domain].classify(task, rollouts)

    @property
    def records(self) -> list[Any]:
        return [r for c in self.by.values() for r in getattr(c, "records", [])]


__all__ = ["CertRow", "DomainClassifier", "Recertifier", "TaskKeyedContext", "label_cell"]
