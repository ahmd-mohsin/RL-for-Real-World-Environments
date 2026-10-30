"""Re-certification of stored tasks under the current framework (DESIGN §7.7 item 22). Measurement only.

Every certificate before item 22 was made with a weaker framework:
- **seed 0**: every model call (ALT attempts, ATK attempts, panel votes) used seed 0 unless a wire asked for seeds by
  tag, so repeated samples were copies (M4, RSI-1, yield-1/2, the frontier round);
- **ALT naming**: Nemotron Ultra and Super shared the runner name ``alt_solver-nemotron``, so Ultra ran twice and Super
  never ran (M4, RSI-1);
- **missing classes**: ORACLE (item 21) and LEAK (item 20) did not exist or did not count in admission, MUT had no read
  mutants and TAINT matched exact tokens only (item 20), TWIN predates item 18's changes;
- **held-out mutants** (Part 1 V5) were off.

This module re-runs a stored task version through the current population **without editing it**: the task, its id, its
version and its digest are what the stored run certified. The World is rebuilt from the task's ``WorldRef`` with its
``work`` directory moved under the re-certification run (``work_world``), so no old run directory is written to.

Two phases, so that money is only spent where it can change the outcome:
1. ``det_job`` (CPU, no model, a ``spawn`` worker per task): ``population(h)`` plus ORACLE where a World registers
   policy truth, ``MUTHeldOut(h)``, MUT on the whole neighbourhood (the h = 0 counterfactual, report only) and the
   archive regressions against a copy of the task's own archive records (``copy_archive``). ``det_verdict`` rejects on
   any confirmed counterexample, a re-opened one or a required class that does not apply.
2. ``certify_paid`` (model calls through injected clients): the stored deterministic results stand in for the classes
   (``offload.Offloaded`` over ``Stored``), their suspects go to the calibrated panels (MUT and MUT-HO: ``Adjudicated``;
   TWIN: ``TwinAdjudicated``), then ALT, ATK and the LEAK probe run, in ``cegis.refine`` with one round and a repairer
   that always declines. The admission rule is ``pipeline.admission_policy()``.

New statuses: ``certified``, ``provisional`` (only the MUT bound misses for want of n), ``on_hold`` (no confirmed
counterexample and no rule failure, but suspects the panels did not decide: a person's queue), ``rejected``; plus
``budget`` (cut by the spend cap: unmeasured). ``attribute`` names the cause of every change against the stored
certificate; ``ManifestRow`` is what the per-task RSI loop starts from (``load_manifest``).

Nothing here knows a model or a World: the CLI (``scripts/synth_recert.py``) binds the wire and names the policy truth
functions by import path (``truths``, resolved in the worker like a ``WorldRef`` factory).
"""

from __future__ import annotations

import asyncio
import gzip
import importlib
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.archive import Archive, regress
from crucible.synth.cegis import CegisReport, refine
from crucible.synth.certificate import AdmissionPolicy, certify
from crucible.synth.falsify import ALT, ATK, MUT, FalsifyContext, MUTHeldOut, population
from crucible.synth.falsify.base import BaseFalsifier, ClassResult, Counterexample, falsify, run_classes
from crucible.synth.falsify.mut import TRACE_OPS
from crucible.synth.falsify.oracle import ORACLE
from crucible.synth.task import Task, WorldRef

SCHEMA = "crucible.synth.recert/v1"
STATUSES = ("certified", "provisional", "on_hold", "rejected", "budget", "error")
ADMITTED = ("certified", "provisional")
CAUSES = ("new_classes", "heldout", "seed", "alt_naming", "alt_set", "other")
Status = Literal["certified", "provisional", "on_hold", "rejected", "budget", "error"]


# ---------------------------------------------------------------------------------------------------------- inputs

class Source(CrucibleModel):
    """A set of stored synthesis runs. ``dir`` is what ``perfect.load_pool`` reads (a domain directory, or a batch arm
    directory whose domain sub-runs hold ``report.json``); ``archive_root`` the ``Archive`` root those runs wrote to."""

    name: str
    kind: str  # pool | m4 | yield | frontier
    dir: str
    archive_root: str
    runs: list[str] | None = None
    statuses: list[str] = Field(default_factory=lambda: ["certified", "provisional"])
    ids: list[str] | None = None  # only these task ids (the learnability pool's list)
    seed0: bool = True  # the runs' model calls were on seed 0
    alt_names_shared: bool = False  # Ultra and Super shared one ALT runner name (Ultra ran twice, Super never)


class OldCert(CrucibleModel):
    status: str
    reasons: list[str] = Field(default_factory=list)
    classes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    path: str = ""  # the stored cegis.json (task + certificate), never written


class Item(CrucibleModel):
    key: str
    source: str
    kinds: list[str]
    run_dir: str
    n: int
    task: Task
    old: OldCert
    builders: list[str] = Field(default_factory=list)
    seed0: bool = True
    alt_names_shared: bool = False
    archive_root: str = ""

    @property
    def tdir(self) -> Path:
        return Path(self.run_dir) / "tasks" / f"t{self.n:02d}"


def _builders(report: dict[str, Any]) -> list[str]:
    fams = report.get("families") or {}
    out = {x.get("family") for r in ("proposer", "oracle_programmer", "contract_writer", "repairer") for x in fams.get(r) or []}
    return sorted(f for f in out if f)


def collect(sources: Sequence[Source]) -> list[Item]:
    """Every stored task version of the sources, once: a task stored by one run (same run directory and number) that
    several sources list (the learnability pool holds the M4 runs) is one item with every source's kind."""
    from crucible.synth.perfect import load_pool

    by: dict[tuple[str, int], Item] = {}
    for s in sources:
        root = Path(s.dir)
        for pi in load_pool(root, statuses=tuple(s.statuses), runs=s.runs):
            if s.ids is not None and pi.task.id not in set(s.ids):
                continue
            run_dir = root / pi.run
            k = (str(run_dir), pi.n)
            if k in by:
                if s.kind not in by[k].kinds:
                    by[k].kinds.append(s.kind)
                continue
            p = run_dir / "tasks" / f"t{pi.n:02d}" / "cegis.json"
            raw = json.loads(p.read_text())
            st = pi.status  # the run report's outcome: an M4-era cegis.json says "rejected" for a MUT-bound-only (provisional) task
            classes = ((raw.get("certificate") or {}).get("classes")) or {}
            rep = json.loads((run_dir / "report.json").read_text())
            by[k] = Item(key=f"{s.name}::{pi.task.id}@v{pi.task.version}", source=s.name, kinds=[s.kind], run_dir=str(run_dir), n=pi.n, task=pi.task,
                         old=OldCert(status=st, reasons=list(raw.get("reasons") or []), classes=classes, path=str(p)), builders=_builders(rep),
                         seed0=s.seed0, alt_names_shared=s.alt_names_shared, archive_root=s.archive_root)
    return list(by.values())


def slug(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:180]


def copy_archive(items: Sequence[Item], dest: Path) -> dict[str, int]:
    """Each item's archived counterexamples, copied from its source's archive into ``dest/<source slug>`` (task ids repeat
    across batch runs, so every source keeps its own archive). New findings of the re-certification go there too; the
    source archives are only read."""
    out: dict[str, int] = {}
    srcs: dict[str, list[Item]] = {}
    for it in items:
        srcs.setdefault(it.source, []).append(it)
    for name, its in srcs.items():
        src, dst = Archive(its[0].archive_root), Archive(dest / slug(name))
        n = 0
        for dom in sorted({i.task.domain for i in its}):
            if not src.path(dom).exists():
                continue
            want = {i.task.id for i in its if i.task.domain == dom}
            n += len(dst.add_all(c for c in src.records(dom) if c.task_id in want))
        out[name] = n
    return out


# ---------------------------------------------------------------------------------------------------------- worlds and truth

def work_world(work: str | Path) -> Callable[[Task], Any]:
    """A ``FalsifyContext.make_world``: the task's own World factory and parameters with ``work`` moved under ``work``
    (both World factories make a fresh sub-directory there per World). The task is not changed."""
    w = str(work)

    def make(task: Task) -> Any:
        params = dict(task.world.params)
        if "work" in params:
            params["work"] = w
        return WorldRef(factory=task.world.factory, params=params).build()

    return make


def resolve_truths(paths: Sequence[str]) -> list[Callable[[Task], Any]]:
    """Policy truth functions named ``module:callable`` (called with the task as JSON, returning ``right``, ``wrong`` or
    anything else for "not covered"), as ``ORACLE`` truths."""
    out = []
    for p in paths:
        mod, _, attr = p.partition(":")
        f = getattr(importlib.import_module(mod), attr)

        def truth(task: Task, f: Any = f) -> Any:
            v = f(task.model_dump(mode="json"))
            return v if v in ("right", "wrong") else None

        out.append(truth)
    return out


def det_population(holdout: float, truths: Sequence[str] = ()) -> list[BaseFalsifier]:
    """The deterministic classes the rounds run: ``population(h)`` and ORACLE (not applicable without a truth)."""
    return [*population(holdout), ORACLE(resolve_truths(truths))]


# ---------------------------------------------------------------------------------------------------------- phase 1

def det_job(task_json: str, work: str, max_mutants: int, holdout: float, truths: Sequence[str], archive: str) -> str:
    """Worker side of phase 1 (module docstring): JSON with ``results`` (every class, MUT-HO included), ``mut_all`` (MUT
    on the whole neighbourhood, h = 0), ``regress`` and ``cpu_s``."""

    async def go() -> str:
        task = Task.model_validate_json(task_json)
        ctx = FalsifyContext(work_world(work), max_mutants=max_mutants)
        t0 = time.monotonic()
        try:
            fs = det_population(holdout, truths)
            res = await run_classes(task, [*fs, MUTHeldOut(holdout=holdout)] if holdout > 0 else fs, ctx)
            nondet = any(r.cls == "REP" and r.c > 0 for r in res)
            mut_all = None
            if holdout > 0 and not nondet:
                mut_all = (await falsify(task, MUT(), ctx)).model_dump(mode="json")
            reg = await regress(Archive(archive), task, ctx, [*fs, ALT(), ATK()])
        finally:
            await ctx.close()
        return json.dumps({"results": [r.model_dump(mode="json") for r in res], "mut_all": mut_all, "regress": reg.model_dump(mode="json"),
                           "cpu_s": round(time.monotonic() - t0, 2)})

    return asyncio.run(go())


def save_det(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(doc, fh)
    tmp.replace(path)


def load_det(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def det_results(doc: dict[str, Any]) -> dict[str, ClassResult]:
    return {r["cls"]: ClassResult.model_validate(r) for r in doc["results"]}


def det_policy() -> AdmissionPolicy:
    """Phase 1's rule: the admission policy with the rules only paid classes can meet left out (an ALT pass, decided
    suspects): what remains rejects for good."""
    from crucible.synth.pipeline import admission_policy

    return admission_policy().model_copy(update={"require_alt_pass": False, "allow_open_suspects": True})


def det_verdict(task: Task, doc: dict[str, Any]) -> tuple[str, list[str], int]:
    """(``rejected`` | ``paid``, reasons, open suspects) of phase 1. ``MUT:upper`` alone is not final: a provisional task
    still needs its paid classes."""
    res = list(det_results(doc).values())
    cert = certify(task, res, det_policy(), reopened=doc["regress"].get("reopened") or [])
    susp = sum(s.suspects for s in cert.classes.values())
    return ("rejected" if cert.status == "rejected" else "paid"), list(cert.reasons) + list(cert.provisional_for), susp


# ---------------------------------------------------------------------------------------------------------- phase 2

class Stored:
    """``offload.Offloaded``'s pool protocol over phase-1 results: every class reads its stored result."""

    def __init__(self, results: dict[str, ClassResult]):
        self.r = results

    async def results(self, task: Task) -> dict[str, ClassResult]:
        return self.r


class Recorder(BaseFalsifier):
    """Delegates every call to ``inner`` and keeps the class's final findings (``refine`` keeps only the certificate's
    counts, and attribution needs which agent passed)."""

    def __init__(self, inner: BaseFalsifier, sink: list[Any]):
        self.inner, self.sink = inner, sink
        self.name = inner.name  # type: ignore[misc]

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        f = getattr(self.inner, "applies", None)
        return await f(task, ctx) if f is not None else None

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Any]:
        return await self.inner.propose(task, ctx)

    async def execute(self, probe: Any, task: Task, ctx: FalsifyContext) -> Any:
        return await self.inner.execute(probe, task, ctx)

    def label(self, probe: Any, outcome: Any) -> Any:
        return self.inner.label(probe, outcome)

    def finalize(self, findings: list[Any]) -> list[Any]:
        fin = getattr(self.inner, "finalize", None)
        out = fin(findings) if fin is not None else findings
        self.sink.append(ClassResult(cls=self.name, findings=list(out)))
        return out

    async def replay(self, cx: Any, task: Task, ctx: FalsifyContext) -> bool:
        return await self.inner.replay(cx, task, ctx)


async def _decline(task: Task, cxs: list[Counterexample]) -> Task | None:
    return None  # measurement only: a stored task is never edited


async def certify_paid(item: Item, doc: dict[str, Any], *, ctx: FalsifyContext, archive: Archive, holdout: float, truths: Sequence[str] = (),
                       panel: Any = None, twin_panel: Any = None, alt: Sequence[Any] = (), atk: Sequence[Any] = (),
                       extra: Sequence[BaseFalsifier] = (), concurrent: bool = True) -> tuple[CegisReport, list[ClassResult]]:
    """Phase 2 on one item (module docstring). ``alt``/``atk`` are ``AgentSpec``s (unique names; the item's builder
    families are excluded); ``extra`` the LEAK probe; ``panel`` judges MUT, MUT-HO, ALT and ATK suspects, ``twin_panel``
    TWIN's. Returns the report and the paid classes' findings (``paid_rows``)."""
    from crucible.synth.falsify.agents import Adjudicated
    from crucible.synth.falsify.twinjudge import TwinAdjudicated
    from crucible.synth.offload import ConcurrentAgents, Offloaded
    from crucible.synth.pipeline import admission_policy

    st = Stored(det_results(doc))
    cheap_f: list[BaseFalsifier] = []
    for f in det_population(holdout, truths):
        o: BaseFalsifier = Offloaded(f, st)  # type: ignore[arg-type]
        if f.name == "MUT" and panel is not None:
            o = Adjudicated(o, panel)
        elif f.name == "TWIN" and twin_panel is not None:
            o = TwinAdjudicated(o, twin_panel)
        cheap_f.append(o)
    heldout: BaseFalsifier | None = None
    if holdout > 0:
        heldout = Offloaded(MUTHeldOut(holdout=holdout), st)  # type: ignore[arg-type]
        heldout = Adjudicated(heldout, panel) if panel is not None else heldout
    ex = set(item.builders)
    paid: list[BaseFalsifier] = []
    for cls, agents in ((ALT, alt), (ATK, atk)):
        if not agents:
            continue
        f = cls(list(agents), exclude_families=ex)
        f = ConcurrentAgents(f) if concurrent else f
        paid.append(Adjudicated(f, panel) if panel is not None else f)
    paid += list(extra)
    sink: list[ClassResult] = []
    paid = [Recorder(f, sink) for f in paid]
    rep = await refine(item.task, repair=_decline, ctx=ctx, archive=archive, falsifiers=cheap_f, paid=paid, policy=admission_policy(), max_rounds=1,
                       heldout=heldout)
    return rep, sink


def status_of(rep: CegisReport | None, det_status: str = "paid", det_reasons: Sequence[str] = ()) -> tuple[Status, list[str]]:
    """The new status and its reasons (module docstring)."""
    if rep is None:
        return ("rejected", list(det_reasons)) if det_status == "rejected" else ("error", ["phase 2 did not run"])
    if rep.status == "admitted":
        return "certified", list(rep.reasons)
    if rep.status == "provisional":
        return "provisional", list(rep.reasons)
    hard = [r for r in rep.reasons if ":suspects=" not in r and not r.startswith("MUT:upper")]
    if not hard and any(":suspects=" in r for r in rep.reasons):
        return "on_hold", list(rep.reasons)
    return "rejected", list(rep.reasons)


def status_h0(status: str, reasons: Sequence[str], mut_all: dict[str, Any] | None) -> str:
    """The status the same evidence gives without held-out mutants (h = 0, report only): MUT's bound and counts from the
    whole neighbourhood (``det_job``'s ``mut_all``), MUT-HO dropped. A rejection on other grounds stands; MUT survivors
    of the whole neighbourhood are the shown and held-out ones together, so a confirmed one rejects at h = 0 too."""
    if status in ("budget", "error") or mut_all is None:
        return status
    from crucible.synth.certificate import class_stat

    m = class_stat(ClassResult.model_validate(mut_all))
    rest = [r for r in reasons if not r.startswith(("MUT:", "MUT-HO:"))]
    if m.c:
        rest.append(f"MUT:confirmed={m.c}")
    hard = [r for r in rest if ":suspects=" not in r]
    if hard:
        return "rejected"
    if any(":suspects=" in r for r in rest) or any(r.startswith(("MUT:suspects", "MUT-HO:suspects")) for r in reasons):
        return "on_hold"
    return "provisional" if m.upper > 0.05 else "certified"


# ---------------------------------------------------------------------------------------------------------- attribution

AGENT_CAUSE = {"ultra": "seed", "super": "alt_naming", "gptoss": "alt_set", "inkling": "alt_set"}


def _agent_kind(name: str) -> str:
    n = name.lower()
    for k in ("ultra", "super", "gptoss", "inkling"):
        if k in n:
            return k
    return "other"


def _findings(rep: CegisReport | None, doc: dict[str, Any] | None, cls: str) -> list[Any]:
    """Confirmed findings of one class: phase 2's certificate counts are summaries, so the findings come from phase 1
    (deterministic classes) or are re-read from the report's archive ids by the caller; here phase 1 only."""
    if doc is None:
        return []
    r = det_results(doc).get(cls)
    return list(r.confirmed) if r is not None else []


def classify_reason(reason: str, item: Item, doc: dict[str, Any] | None, paid_rows: Sequence[dict[str, Any]] = ()) -> tuple[str, str]:
    """(cause, detail) of one new reason that the stored certificate did not have (module docstring of ``attribute``)."""
    cls = reason.split(":", 1)[0]
    old = item.old.classes.get(cls) or {}
    if cls in ("ORACLE", "LEAK"):
        return "new_classes", f"{cls} (items 20/21: not in admission before)"
    if cls == "MUT-HO":
        return "heldout", "a held-out mutant survives"
    if cls == "MUT" and reason.startswith("MUT:upper"):
        allm = (doc or {}).get("mut_all")
        if allm is not None:
            r = ClassResult.model_validate(allm)
            from crucible.synth.certificate import class_stat

            if r.c == 0 and class_stat(r).upper <= 0.05:
                return "heldout", f"MUT n falls below 59 with the held-out share removed (n = {r.n} on the whole neighbourhood)"
        return "other", f"MUT n below 59 on the whole neighbourhood too (old n {old.get('n')}: the M4 duplicate-mutant count, RSI-1)"
    if cls == "MUT":
        ops = Counter(f.op for f in _findings(None, doc, "MUT"))
        if ops and all(op in TRACE_OPS for op in ops):
            return "new_classes", f"MUT read mutants (item 20): {dict(ops)}"
        return "other", f"MUT {dict(ops)} (old c {old.get('c')})"
    if cls == "TAINT":
        fs = _findings(None, doc, "TAINT")
        layers = Counter(str(f.outcome.data.get("layer")) for f in fs if f.op == "derive")
        if layers and all(k not in ("exact", "None") for k in layers):
            return "new_classes", f"TAINT layered leak matcher (item 20): {dict(layers)}"
        return "other", f"TAINT {dict(Counter(f.op for f in fs))} (old c {old.get('c')})"
    if cls == "TWIN":
        return "new_classes", "TWIN (item 18 changes: sibling rules, inline judge)"
    if cls in ("ALT", "ATK"):
        rows = [r for r in paid_rows if r.get("cls") == cls]
        kinds = Counter(_agent_kind(str(r.get("agent"))) for r in rows if r.get("decisive"))
        if cls == "ATK":
            return "seed", "ATK re-sampled with real seeds (the attackers are the stored runs' models)"
        if reason.startswith("ALT:suspects"):
            return "seed", "ALT failures the panel did not decide (re-sampled attempts and votes)"
        if reason.startswith("ALT:no_pass") or not kinds:
            return "alt_set", f"no pass from the ALT set ({sorted({str(r.get('agent')) for r in rows})}); the stored pass came from another set or one sample"
        k = kinds.most_common(1)[0][0]
        c = AGENT_CAUSE.get(k, "other")
        if c == "alt_naming" and not item.alt_names_shared:
            c = "alt_set"  # Super ran in the source (yield): its v3 harness, not the naming bug, is new
        return c, f"{cls} by {dict(kinds)}"
    if cls == "archive":
        return "other", "an archived counterexample re-opens (agent traces now replay as programs)"
    return "other", reason


def attribute(item: Item, new: str, reasons: Sequence[str], doc: dict[str, Any] | None, paid_rows: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """How the new status relates to the stored one (``outcome``) and why (``causes``; the first is ``primary``).
    ``outcome``: ``holds`` (same status), ``changes`` (certified <-> provisional), ``held`` (admitted -> on hold),
    ``falls`` (admitted -> rejected), ``rises`` (not admitted -> admitted), ``stays`` (not admitted -> not admitted),
    ``unmeasured`` (budget, error). A fall or change is explained by the reasons the stored certificate did not have;
    a rise by the agents that now pass and the old reasons that no longer hold."""
    old = item.old.status
    if new in ("budget", "error"):
        outcome = "unmeasured"
    elif new == old:
        outcome = "holds"
    elif old in ADMITTED and new in ADMITTED:
        outcome = "changes"
    elif old in ADMITTED and new == "on_hold":
        outcome = "held"
    elif old in ADMITTED:
        outcome = "falls"
    elif new in ADMITTED:
        outcome = "rises"
    else:
        outcome = "stays"
    causes: list[tuple[str, str]] = []
    soft: set[tuple[str, str]] = set()
    old_heads = {r.split("=", 1)[0] for r in item.old.reasons}
    if outcome in ("changes", "held", "falls", "stays"):
        for r in reasons:
            if r.split("=", 1)[0] in old_heads:
                continue  # the stored certificate already had this reason
            if r.startswith("archive:reopened") and doc is not None and not (doc.get("regress") or {}).get("reopened"):
                continue  # refine's last regression re-reads the counterexample this run just archived: not a cause of its own
            c = classify_reason(r, item, doc, paid_rows)
            causes.append(c)
            if r.startswith("MUT:upper") and outcome in ("falls", "held"):
                soft.add(c)  # the MUT bound alone gives a provisional task: the hard reasons explain a fall or a hold
    if outcome in ("rises", "changes", "stays"):
        new_heads = {r.split("=", 1)[0] for r in reasons}
        for r in item.old.reasons:
            h = r.split("=", 1)[0]
            if h in new_heads:
                continue
            cls = h.split(":", 1)[0]
            if cls == "ALT":
                passed = Counter(_agent_kind(str(x.get("agent"))) for x in paid_rows if x.get("cls") == "ALT" and x.get("passed"))
                k = passed.most_common(1)[0][0] if passed else "other"
                c = AGENT_CAUSE.get(k, "other")
                if c == "alt_naming" and not item.alt_names_shared:
                    c = "alt_set"
                causes.append((c, f"old {r} cleared: ALT passes by {dict(passed)}"))
            elif cls == "MUT" and h.startswith("MUT:upper"):
                causes.append(("other", f"old {r} cleared (a larger distinct-effect neighbourhood: item 20 read mutants)"))
            elif cls in ("TWIN", "TAINT", "MUT", "RED", "NULL", "PERT", "REP"):
                causes.append(("new_classes" if cls in ("TWIN", "TAINT", "MUT") else "other", f"old {r} cleared by today's {cls}"))
            else:
                causes.append(("seed" if cls in ("ATK", "LEAK") or r.endswith("suspects") else "other", f"old {r} cleared"))
    order = {c: i for i, c in enumerate(CAUSES)}
    causes.sort(key=lambda x: (x in soft, order.get(x[0], 99)))
    return {"old": old, "new": new, "outcome": outcome, "primary": causes[0][0] if causes else None,
            "causes": [{"cause": c, "detail": d} for c, d in causes]}


def paid_rows(rep: CegisReport | None, results: Sequence[ClassResult] = ()) -> list[dict[str, Any]]:
    """One row per ALT/ATK attempt of phase 2: agent, passed, label, defect; ``decisive`` for the ones that set the
    status (a confirmed finding, or ALT's passes)."""
    out = []
    for r in results:
        if r.cls not in ("ALT", "ATK"):
            continue
        for f in r.findings:
            if not f.outcome.valid:
                continue
            agent = f.probe.data.get("agent")
            dec = f.label == "confirmed" or (r.cls == "ALT" and bool(f.outcome.passed))
            out.append({"cls": r.cls, "agent": agent, "passed": bool(f.outcome.passed), "label": f.label, "defect": f.defect, "decisive": dec,
                        "calls": f.outcome.data.get("calls"), "panel": f.outcome.data.get("panel")})
    return out


# ---------------------------------------------------------------------------------------------------------- outputs

class ManifestRow(CrucibleModel):
    """One task of the re-certified pool (what the per-task RSI loop starts from)."""

    key: str
    task_id: str
    version: int
    digest: str
    domain: str
    source: str
    kinds: list[str]
    task_path: str  # the stored cegis.json holding this exact version (``load_task``)
    old_status: str
    old_reasons: list[str] = Field(default_factory=list)
    old_certificate: str = ""
    status: str
    reasons: list[str] = Field(default_factory=list)
    certificate: str = ""  # the new certificate, beside the old one
    heldout: float = 0.0
    status_h0: str = ""  # the same evidence without held-out mutants (``status_h0``; report only)
    outcome: str = ""
    causes: list[dict[str, str]] = Field(default_factory=list)
    alt: dict[str, Any] = Field(default_factory=dict)  # passes / n of phase 2, per agent

    def load_task(self) -> Task:
        raw = json.loads(Path(self.task_path).read_text())
        t = Task.model_validate(raw["task"])
        if t.digest() != self.digest:
            raise ValueError(f"{self.task_path}: the stored task changed (digest {t.digest()} != {self.digest})")
        return t


def load_manifest(path: str | Path, *, statuses: Sequence[str] | None = None) -> list[ManifestRow]:
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} manifest")
    rows = [ManifestRow.model_validate(r) for r in doc["rows"]]
    return [r for r in rows if statuses is None or r.status in statuses]


def write_new(path: Path, doc: dict[str, Any]) -> Path:
    """Write a new file and never overwrite one (old and new certificates are kept): ``FileExistsError`` otherwise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as fh:
        fh.write(json.dumps(doc, indent=1, default=str) + "\n")
    return path


def beside(item: Item, tag: str) -> Path:
    """The new certificate's place: beside the stored ``cegis.json``, versioned by the re-certification's tag."""
    return item.tdir / f"certificate.{tag}.json"


def summarize(rows: Sequence[ManifestRow]) -> dict[str, Any]:
    out: dict[str, Any] = {"tasks": len(rows), "status": dict(Counter(r.status for r in rows)),
                           "old_status": dict(Counter(r.old_status for r in rows)), "outcome": dict(Counter(r.outcome for r in rows))}
    old_adm = [r for r in rows if r.old_status in ADMITTED]
    out["old_admitted"] = {"n": len(old_adm), "outcome": dict(Counter(r.outcome for r in old_adm)),
                           "primary": dict(Counter((r.causes[0]["cause"] if r.causes else "none") for r in old_adm if r.outcome not in ("holds",))),
                           "any_cause": dict(Counter(c for r in old_adm if r.outcome != "holds" for c in {x["cause"] for x in r.causes})),
                           "primary_by_outcome": {o: dict(Counter((r.causes[0]["cause"] if r.causes else "none") for r in old_adm if r.outcome == o))
                                                  for o in sorted({r.outcome for r in old_adm})}}
    rose = [r for r in rows if r.outcome == "rises"]
    out["rises_primary"] = dict(Counter((r.causes[0]["cause"] if r.causes else "none") for r in rose))
    out["by_source"] = {s: {"n": sum(1 for r in rows if r.source == s), "status": dict(Counter(r.status for r in rows if r.source == s)),
                            "old": dict(Counter(r.old_status for r in rows if r.source == s))} for s in sorted({r.source for r in rows})}
    out["transitions"] = dict(Counter(f"{r.old_status}->{r.status}" for r in rows))
    out["status_h0"] = dict(Counter(r.status_h0 for r in rows))
    out["transitions_h0"] = dict(Counter(f"{r.old_status}->{r.status_h0}" for r in rows))
    return out


__all__ = ["ADMITTED", "CAUSES", "SCHEMA", "STATUSES", "Item", "ManifestRow", "OldCert", "Recorder", "Source", "Stored", "attribute", "beside", "certify_paid",
           "classify_reason", "collect", "copy_archive", "det_job", "det_policy", "det_population", "det_results", "det_verdict", "load_det",
           "load_manifest", "paid_rows", "resolve_truths", "save_det", "slug", "status_h0", "status_of", "summarize", "work_world", "write_new"]
