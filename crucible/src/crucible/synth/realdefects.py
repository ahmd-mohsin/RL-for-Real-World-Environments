"""The real-defect set (part2-paper-core §8 "a third, real defect set", §6 calibration, E8): historically observed
defects of known class, with provenance, and, where the defective artifact still exists, an executable version on which
falsifier detection is measured.

**Sources.**
1. **The debug lane's measurement-time failures** (``debug_lane_defects.json``, tracked beside the M5 run reports): one
   record per defect a person found while measuring or fixing a training task (the fixing notes, ``fixing/README.md``,
   ``uzair-fixed-tasks.md``, ``new-tasks/README.md``). Every training record carries the doc, the line and an exact
   quote, checked against the doc when the set is built (``verify_quotes``). Held-out and benchmark tasks appear as
   counts only: their ids, quotes and fixes are dropped on load (``heldout#k``).
2. **The M4 swarm runs' confirmed counterexamples** (``<synth data>/<domain>/archive.jsonl``, hash-chained): each is a
   natural defect of a swarm-written task, with its archive id and record hash.
3. **The finance_ap oracle finding** (§5.1 rule 8): the pack's own oracle decides invoice ids it never lists (D5).

**Classes.** ``MECHANISM_CLASS`` maps a debug-lane mechanism to D1–D11; a record is ``P1`` (a validity defect, counted
per class), ``P2`` (learnability: too easy, too hard, budget-bound: not a D class), ``P4`` (a duplicate of another task),
``pipeline`` (a false refusal by the factory's own gate: not a task defect) or ``unclassified`` (the cause is not stated).

**Independence.** Only defects found *without* the synth falsifiers (people at measurement time, the lane's gates) can
calibrate them: a defect TAINT found has detection rate 1 under TAINT by construction. ``independent`` marks the debug
lane's records; the archive and the finance_ap finding are falsifier-found and are reported apart (§8.1's caveat, one
level up).

**Executable versions** (``executable.status``): ``executable`` when the defective task version is still on disk and a
falsifier on its substrate targets the mechanism (``DETECTOR``); ``lost`` when the version was rebuilt away (only the
fixed version exists); ``no_detector`` when no class targets it on that substrate (a recall gap); ``heldout`` (never
executed); ``not_built`` (a screened window, no task). ``measure`` runs the detectors: the debug lane's through an
injected ``detect_debug`` (the factory wire builds the lane task), the archive's by re-running the counterexample's
class on the stored task version, the finance_ap finding on the pack's World.
"""

from __future__ import annotations

import importlib
import inspect
import json
import tempfile
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.archive import Archive
from crucible.synth.falsify.base import Counterexample, FalsifyContext, falsify
from crucible.synth.task import Task, WorldRef

Quality = Literal["P1", "P2", "P4", "pipeline", "unclassified"]

MECHANISM_CLASS: dict[str, tuple[Quality, str | None]] = {
    "unstated_rule": ("P1", "D5"),  # a hidden test checks a behaviour no paragraph of the report states
    "undescribed_fault": ("P1", "D5"),  # a fault left reverted that the report never asks to fix
    "fake_link_coarse_test": ("P1", "D6"),  # one test grades two milestones: one fix is rejected until the other lands
    "fake_link_load": ("P1", "D11"),  # the claimed link is a load-dependent test
    "stale_visible_test": ("P1", "D6"),  # a visible test rejects the valid fix (the agent's own verifier is over-strict)
    "changelog_hint": ("P1", "D7"),  # the start tree announces the fix
    "leak_other": ("P1", "D7"),  # the answer is readable elsewhere (visible test data)
    "grading_bug": ("P1", "D6"),  # a milestone's check rejects a correct fix (it holds another milestone's cases)
    "env_bug": ("P1", "D10"),  # the World cannot run the tests: no tool sequence reaches success
    "path_rule_false_positive": ("P1", "D6"),  # a path gate rejects a valid solution
    "too_easy": ("P2", None),
    "too_hard": ("P2", None),
    "budget_bound": ("P2", None),
}
# records whose mechanism is "other" or whose class the mechanism does not decide (the debug records' ids)
OVERRIDES: dict[str, tuple[Quality, str | None, str]] = {
    "black-ec71b918-duplicate": ("P4", None, "duplicate"),
    "mvdan-sh-04056a74-duplicate": ("P4", None, "duplicate"),
    "carbon-a783fb3a-fake-link": ("unclassified", None, "fake_link_cause_unstated"),
    "echo-61eca05b-fake-link": ("unclassified", None, "fake_link_cause_unstated"),
    "heldout-fake-link-10": ("unclassified", None, "fake_link_cause_unstated"),
    "heldout-fake-link-11": ("unclassified", None, "fake_link_cause_unstated"),
    "golang-net-61a78cba-testlegacy": ("P1", "D11", "grading_bug_load"),  # the milestone ran through a suite that flaked under load
    "golang-net-61a78cba-trailer-case": ("unclassified", None, "cause_unstated"),
    "mvdan-sh-02a801cd-d1-overhidden": ("P1", "D5", "over_hidden_feedback"),  # the agent cannot see its own regressions
    "golang-net-os-specific-packages": ("pipeline", None, "false_refusal"),
    "isort-f66fa65b-leak-probe": ("pipeline", None, "false_refusal"),
    "screen-markdown-internal-timing": ("P1", "D6", "test_calls_internal"),  # a hidden test calls an internal function
}
DETECTOR: dict[str, str] = {
    "changelog_hint": "TAINT:world_leak",
    "fake_link_coarse_test": "MUT:SharedCheck",
    "grading_bug": "MUT:SharedCheck",
    "stale_visible_test": "TAINT:visible_contradicts",
    "undescribed_fault": "TAINT:ledger",
    "unstated_rule": "TAINT:report_literal",
}


class Executable(CrucibleModel):
    status: Literal["executable", "lost", "no_detector", "heldout", "not_built", "unknown"] = "unknown"
    artifact: str | None = None  # where the defective version is (a store path, an archive task version)
    detector: str | None = None  # "<class>:<op>"
    why: str = ""


class Detection(CrucibleModel):
    detected: bool | None = None
    exhibits: bool | None = None  # the artifact still shows the defect (an archived counterexample still reopens)
    by: list[str] = Field(default_factory=list)  # "<class>:<op>:<site>" of the confirmed findings
    detail: dict[str, Any] = Field(default_factory=dict)


class RealDefect(CrucibleModel):
    id: str
    source: Literal["debug_lane", "swarm_archive", "finance_ap"]
    quality: Quality
    d_class: str | None = None
    mechanism: str
    split: Literal["train", "heldout"] = "train"
    task_version: str | None = None
    fixed_by_version: str | None = None
    detail: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)
    found_by: str = ""
    independent: bool = True
    executable: Executable = Field(default_factory=Executable)
    detection: Detection | None = None


class RealSet(CrucibleModel):
    schema_version: str = "crucible.synth.realdefects/v1"
    built_at: str = ""
    defects: list[RealDefect] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------------------------------------- loaders

class QuoteError(ValueError):
    pass


def verify_quote(docs_root: Path, doc: str, line: int, quote: str, *, slack: int = 2) -> bool:
    """The quote is an exact substring of the cited line (or of a line at most ``slack`` away)."""
    p = Path(docs_root) / doc
    if not p.is_file():
        return False
    lines = p.read_text().splitlines()
    lo, hi = max(0, line - 1 - slack), min(len(lines), line + slack)
    return any(quote in ln for ln in lines[lo:hi])


def load_debug_records(path: str | Path, docs_root: str | Path, *, verify: bool = True) -> list[RealDefect]:
    """The debug lane's records (module doc). A training quote that is not at its line raises ``QuoteError``."""
    rows = json.loads(Path(path).read_text())
    out: list[RealDefect] = []
    k = 0
    for r in rows:
        mech = str(r["mechanism"])
        quality, d, mech2 = OVERRIDES.get(r["id"], (*MECHANISM_CLASS.get(mech, ("unclassified", None)), mech))
        split = "heldout" if r.get("split") == "heldout" else "train"
        if split == "heldout":
            k += 1
            out.append(RealDefect(id=f"heldout#{k}", source="debug_lane", quality=quality, d_class=d, mechanism=mech2, split="heldout",
                                  provenance={"doc": r.get("doc"), "line": r.get("line")}, found_by=str(r.get("found_by") or ""),
                                  executable=Executable(status="heldout", why="held-out: counts only")))
            continue
        if verify and r.get("quote") and not verify_quote(Path(docs_root), str(r["doc"]), int(r["line"]), str(r["quote"])):
            raise QuoteError(f"{r['id']}: the quote is not at {r['doc']}:{r['line']}")
        built = not str(r.get("task_version") or "").count("(screened")
        out.append(RealDefect(id=str(r["id"]), source="debug_lane", quality=quality, d_class=d, mechanism=mech2, split="train",
                              task_version=r.get("task_version"), fixed_by_version=r.get("fixed_by_version"), detail=str(r.get("detail") or ""),
                              provenance={"doc": r["doc"], "line": r["line"], "quote": r.get("quote", ""), "who": r.get("who"),
                                          "milestones_or_fixes": r.get("milestones_or_fixes") or []},
                              found_by=str(r.get("found_by") or "measurement"), independent=True,
                              executable=Executable(status="unknown" if built else "not_built")))
    return out


def load_archive(synth_data: str | Path, domains: Iterable[str] | None = None) -> list[RealDefect]:
    """Every confirmed counterexample of the swarm runs' archives (hash chain verified on read)."""
    root = Path(synth_data)
    arch = Archive(root)
    out: list[RealDefect] = []
    doms = sorted(domains) if domains is not None else sorted(p.parent.name for p in root.glob("*/archive.jsonl"))
    for dom in doms:
        for cx in arch.records(dom):
            if cx.label != "confirmed":
                continue
            ex = _archived_version(root, cx)
            out.append(RealDefect(id=cx.id, source="swarm_archive", quality="P1" if cx.defect else "unclassified", d_class=cx.defect,
                                  mechanism=f"{cx.cls}:{cx.op}", task_version=f"{cx.task_id}@v{cx.task_version}",
                                  provenance={"archive": str(arch.path(dom)), "id": cx.id, "site": cx.site, "reasons_seen": cx.reasons_seen},
                                  found_by=f"falsifier:{cx.cls}", independent=False, executable=ex))
    return out


def _task_file(root: Path, task_id: str) -> Path:
    dom, run, tid = task_id.split("/", 2)
    return root / dom / run / "tasks" / tid / "cegis.json"


def _archived_version(root: Path, cx: Counterexample) -> Executable:
    f = _task_file(root, cx.task_id)
    if not f.is_file():
        return Executable(status="lost", why="no stored task")
    t = json.loads(f.read_text()).get("task") or {}
    if int(t.get("version") or 0) != cx.task_version:
        return Executable(status="lost", artifact=str(f), why=f"stored version v{t.get('version')} (repaired), counterexample on v{cx.task_version}")
    return Executable(status="executable", artifact=str(f), detector=f"{cx.cls}:{cx.op}")


def finance_ap_finding() -> RealDefect:
    return RealDefect(id="finance_ap-pack-oracle-unlisted-ids", source="finance_ap", quality="P1", d_class="D5", mechanism="TAINT:literal",
                      task_version="crucible_pack_finance_ap APTask(0).oracle",
                      detail="The pack's oracle decides invoice ids it never lists: success needs ids no instruction, tool or earlier observation gives.",
                      provenance={"doc": "docs/research/env-synthesis/part2-paper-core.md", "section": "5.1 rule 8",
                                  "test": "tests/unit/test_synth_iface.py::test_finance_ap_oracle_runs_through_the_adapter_and_passes_its_inferred_contract"},
                      found_by="falsifier:TAINT", independent=False,
                      executable=Executable(status="executable", artifact="crucible_pack_finance_ap.studio:APTask(0)", detector="TAINT:literal"))


def annotate_debug(defects: Iterable[RealDefect], debug_store: str | Path | None) -> None:
    """Which debug-lane defect versions are still on disk (a training store's ``tasks/<id>/window.json``) and have a
    detector on the repository substrate."""
    store = Path(debug_store) if debug_store is not None else None
    for d in defects:
        if d.source != "debug_lane" or d.executable.status in ("heldout", "not_built"):
            continue
        det = DETECTOR.get(d.mechanism)
        if d.quality != "P1":
            d.executable = Executable(status="no_detector", why=f"{d.quality}: not a validity defect")
            continue
        here = store is not None and d.task_version is not None and (store / "tasks" / d.task_version / "window.json").is_file()
        if det is None:
            d.executable = Executable(status="no_detector", artifact=str(store / "tasks" / str(d.task_version)) if here else None,
                                      why=f"no falsifier on the repository substrate targets {d.mechanism}")
        elif not here:
            d.executable = Executable(status="lost", detector=det, why="the defective version is not in the store (rebuilt or built elsewhere)")
        else:
            d.executable = Executable(status="executable", artifact=str(store / "tasks" / str(d.task_version)), detector=det)


def build_set(*, debug_records: str | Path | None, docs_root: str | Path, synth_data: str | Path | None = None, debug_store: str | Path | None = None,
              verify: bool = True) -> RealSet:
    defects: list[RealDefect] = []
    if debug_records is not None and Path(debug_records).is_file():
        defects += load_debug_records(debug_records, docs_root, verify=verify)
        annotate_debug(defects, debug_store)
    if synth_data is not None and Path(synth_data).is_dir():
        defects += load_archive(synth_data, domains=[p.parent.name for p in Path(synth_data).glob("*/archive.jsonl") if not p.parent.name.startswith("_")
                                                      and p.parent.name != "debug"])
    defects.append(finance_ap_finding())
    return RealSet(built_at=time.strftime("%Y-%m-%dT%H:%M:%S"), defects=defects,
                   notes=["only independent defects (found without the synth falsifiers) calibrate recall; the archive and the finance_ap finding are falsifier-found",
                          "held-out and benchmark tasks appear as counts only (heldout#k)"])


# ---------------------------------------------------------------------------------------------------------- measurement

DetectDebug = Callable[[RealDefect], Awaitable[Detection] | Detection]


async def measure_archive(d: RealDefect, synth_data: Path, work: Path) -> Detection:
    """Re-run the counterexample's class on the stored task version: does it still exhibit, and is it found again?"""
    from crucible.synth.falsify import cheap

    cx = next(c for c in Archive(synth_data).records(d.task_version.split("/", 1)[0]) if c.id == d.id) if d.task_version else None  # type: ignore[union-attr]
    if cx is None:
        return Detection(detail={"why": "counterexample not found"})
    t = json.loads(_task_file(synth_data, cx.task_id).read_text())["task"]
    task = Task.model_validate(t)
    params = {**task.world.params, "work": str(work)}  # a scratch work dir of our own, never the run's
    task = task.model_copy(update={"world": WorldRef(factory=task.world.factory, params=params)})
    by = {f.name: f for f in cheap()}
    f = by.get(cx.cls)
    if f is None:
        return Detection(detail={"why": f"class {cx.cls} needs a model (adjudication or agents)"})
    ctx = FalsifyContext()
    try:
        exhibits = await f.replay(cx, task, ctx)
        res = await falsify(task, f, ctx)
    finally:
        await ctx.close()
    hits = [x for x in res.confirmed if x.defect == cx.defect]
    return Detection(detected=bool(hits), exhibits=exhibits, by=[f"{x.cls}:{x.op}:{x.site}" for x in hits[:5]],
                     detail={"same_site": any(x.site == cx.site and x.op == cx.op for x in res.confirmed), "suspects": len(res.suspects)})


async def measure_finance_ap() -> Detection:
    """The pack's oracle as it was (no list first) under TAINT on the pack's World, and the repaired one for contrast."""
    from crucible.synth.falsify import TAINT
    from crucible.synth.iface import CrucibleWorldAdapter
    from crucible.synth.infer import skeleton
    from crucible.synth.program import Lit, Program, Step, run
    from crucible.synth.task import IntentCard

    studio = importlib.import_module("crucible_pack_finance_ap.studio")
    t = studio.APTask(0)
    with tempfile.TemporaryDirectory(prefix="synth-rd-") as td:  # a throwaway World (not data): removed on exit
        w = await t.open(Path(td) / "work")
        try:
            a = CrucibleWorldAdapter(w, seed=0)

            def prog(list_first: bool) -> Program:
                steps = [Step(id=f"o{i}", tool=x["function_name"], args={k: Lit(v=v) for k, v in x["arguments"].items()}) for i, x in enumerate(t.oracle)]
                if list_first:
                    steps.insert(0, Step(id="lst", tool="erp.list_invoices", args={"status": Lit(v="pending")}))
                return Program(steps=steps)

            r = await run(a, prog(True), t.instruction, seed=0)
            fixed = Task(id="finance_ap/pack-oracle", domain="finance-ap", world=WorldRef(factory="adapter"), instruction=t.instruction, contract=skeleton(r),
                         oracle=prog(True), intent=IntentCard(skill="AP-3"))
            raw = fixed.model_copy(update={"oracle": prog(False)})
            hits = (await falsify(raw, TAINT(), FalsifyContext(make_world=lambda _task: a))).confirmed
            clean = (await falsify(fixed, TAINT(), FalsifyContext(make_world=lambda _task: a))).c
        finally:
            await w.close()
            cleanup = getattr(t, "cleanup", None)
            if cleanup is not None:
                cleanup()
    d5 = [x for x in hits if x.defect == "D5"]
    return Detection(detected=bool(d5), exhibits=bool(d5), by=[f"TAINT:{x.op}:{x.site}" for x in d5[:5]], detail={"repaired_version_confirmed": clean})


async def measure(rs: RealSet, *, detect_debug: DetectDebug | None = None, synth_data: str | Path | None = None, work: str | Path | None = None,
                  include: Callable[[RealDefect], bool] | None = None) -> RealSet:
    """Run the detector of every executable defect (module doc); the others keep ``detection = None``."""
    out = rs.model_copy(deep=True)
    for d in out.defects:
        if d.executable.status != "executable" or (include is not None and not include(d)):
            continue
        try:
            if d.source == "debug_lane" and detect_debug is not None:
                det = detect_debug(d)
                d.detection = await det if inspect.isawaitable(det) else det  # type: ignore[assignment]
            elif d.source == "swarm_archive" and synth_data is not None and work is not None:
                d.detection = await measure_archive(d, Path(synth_data), Path(work))
            elif d.source == "finance_ap":
                d.detection = await measure_finance_ap()
        except Exception as e:  # noqa: BLE001 - a detector that cannot run is recorded, never a silent miss
            d.detection = Detection(detail={"error": f"{type(e).__name__}: {e}"[:300]})
    return out


def summary(rs: RealSet) -> dict[str, Any]:
    """Counts per class, source, quality and executability; detection per class on the independent executable ones."""
    ds = rs.defects
    p1 = [d for d in ds if d.quality == "P1"]
    det: dict[str, dict[str, int]] = {}
    for d in p1:
        if d.detection is None or d.detection.detected is None:
            continue
        key = f"{d.d_class}{'' if d.independent else ' (falsifier-found)'}"
        s = det.setdefault(key, {"n": 0, "detected": 0})
        s["n"] += 1
        s["detected"] += int(bool(d.detection.detected))
    return {
        "total": len(ds),
        "by_quality": dict(sorted(Counter(d.quality for d in ds).items())),
        "p1_by_class": dict(sorted(Counter(str(d.d_class) for d in p1).items())),
        "p1_by_class_independent": dict(sorted(Counter(str(d.d_class) for d in p1 if d.independent).items())),
        "by_source": dict(sorted(Counter(d.source for d in ds).items())),
        "by_split": dict(sorted(Counter(d.split for d in ds).items())),
        "executable": dict(sorted(Counter(d.executable.status for d in ds).items())),
        "executable_by_source": {s: sum(1 for d in ds if d.source == s and d.executable.status == "executable") for s in sorted({d.source for d in ds})},
        "detection": det,
        "misses": [{"id": d.id, "class": d.d_class, "mechanism": d.mechanism, "detail": d.detection.detail} for d in p1
                   if d.detection is not None and d.detection.detected is False][:40],
        "no_detector": dict(sorted(Counter(d.mechanism for d in ds if d.executable.status == "no_detector" and d.quality == "P1").items())),
    }


def counts_table(rs: RealSet) -> str:
    """The per-class table for the paper (markdown)."""
    s = summary(rs)
    rows = ["| Class | Independent (debug lane) | Falsifier-found (archive, finance_ap) |", "|---|---|---|"]
    classes = sorted({str(d.d_class) for d in rs.defects if d.quality == "P1"}, key=lambda c: (len(c), c))
    for c in classes:
        ind = sum(1 for d in rs.defects if d.quality == "P1" and str(d.d_class) == c and d.independent)
        fal = sum(1 for d in rs.defects if d.quality == "P1" and str(d.d_class) == c and not d.independent)
        rows.append(f"| {c} | {ind} | {fal} |")
    rows.append(f"| other (P2/P4/pipeline/unclassified) | {sum(v for k, v in s['by_quality'].items() if k != 'P1')} | |")
    return "\n".join(rows)


def load(path: str | Path) -> RealSet:
    return RealSet.model_validate_json(Path(path).read_text())


def as_dicts(defects: Iterable[RealDefect]) -> list[Mapping[str, Any]]:
    return [d.model_dump(mode="json") for d in defects]
