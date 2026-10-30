"""Certified parts, counterexample attribution and per-part repair for compositions (DESIGN §7.7 item 18, compose-2).

compose-1 reached 46–51-call composites, and none was certified. Two causes:
- defects came in from parts that were never certified on their own;
- the repairer re-emitted the whole long program and broke it (3 of 3).

The rule stays as it is: a non-builder family must pass the *whole* composite from its instruction. What changes is
how a composite is assembled and repaired.

**1. Parts are certified first.** A part is certified (or provisional) on its own World state: the base World after the
parts before it (``compose.VariantSpec`` prefix). ``PartCert`` records the outcome against the part's digest, and a
``CertBook`` keeps them (``certs.json``). ``compose(..., certs=book)`` refuses a part without an admitted certificate
for that very version (``require_certified``). ``certify_links`` certifies the links of a stored chain in order, and
``compose.build_chain(certify=...)`` certifies each link before it joins.

- **Rebase.** When a repair changes a part's program, the World state of every later part changes. Each later part is
  rebased onto the new prefix (``rebase``) and certified again.
- **Chain cut.** A link that cannot be certified ends the chain there. Later links were written about its results.

**2. Per-part repair** (``PartRepairer``, the composite Runner's ``repair_factory``). A counterexample found on the
composite is attributed to the parts that own it (``attribute``):

| evidence | owner |
|---|---|
| a probe program (mutant, null agent, agent trace) | the parts whose steps or answer differ from the composite oracle's (``program_owners``) |
| a site | the step ids (``t<j>_``) or pattern ids (``t<j>.``) it names |
| the grader's reason codes | the pattern ids they name, and a rebased skeleton pattern (``x.r<k>``) by the part that last wrote its address |

Each owner part then gets the counterexample in its own frame (``to_part``):
- the program is cut to that part's steps, with step names and refs de-prefixed;
- the answer becomes that part's element;
- the reason codes become that part's codes;
- a twin's composite instruction and contract are dropped.

The repair runs part by part:
1. Only the owner parts are repaired, with the repairer role and the part's own World catalogue. The repaired version
   must run, pass its own contract, and re-open none of the part's archived counterexamples (the monotone archive at
   the part level).
2. The repaired part is certified again (the full population, paid classes included).
3. Later parts are rebased when its program changed.
4. The composite is recomposed (the writer writes the instruction again) as the next version of the same composite
   id, so ``cegis.refine`` holds the monotone archive at the composite level.

Some translations reproduce at the part level: the part's contract also gives the wrong verdict on the cut program.
Those are archived under the part. One that does not reproduce is a composition-level defect: the parts reject it, but
the merge lost the check. No part repair can fix such a defect. It is recorded as ``composition`` (the framework
fixes these), and the composite repair declines. The merged program is never repaired.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.archive import regress
from crucible.synth.canon import cjson, content_hash, diff
from crucible.synth.compose import Composed, Composition, CompositionError, compose, concat, part_programs, rename, run_parts, split_world
from crucible.synth.contract import grade
from crucible.synth.falsify.base import BaseFalsifier, Counterexample, FalsifyContext
from crucible.synth.program import Arg, Fn, Lit, Program, Ref, Step
from crucible.synth.task import Task

ADMIT = ("certified", "provisional")


# ---------------------------------------------------------------------------------------------------------- certificates

class PartCert(CrucibleModel):
    task_id: str
    version: int
    digest: str
    status: str  # certified | provisional | rejected | dropped | error | budget
    reasons: list[str] = Field(default_factory=list)
    classes: dict[str, Any] = Field(default_factory=dict)
    rounds: int = 0
    repairs: int = 0
    note: str = ""
    at: float = 0.0


class CertBook:
    """Part certificates by (task id, version), persisted as one JSON file (``certs.json``)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self.rows: dict[str, PartCert] = {}
        if self.path is not None and self.path.exists():
            for d in json.loads(self.path.read_text()):
                c = PartCert.model_validate(d)
                self.rows[f"{c.task_id}@{c.version}"] = c

    def put(self, c: PartCert) -> None:
        self.rows[f"{c.task_id}@{c.version}"] = c
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([x.model_dump(mode="json") for x in self.rows.values()], indent=1) + "\n")

    def get(self, task: Task) -> PartCert | None:
        c = self.rows.get(f"{task.id}@{task.version}")
        return c if c is not None and c.digest == task.digest() else None

    def admitted(self, task: Task, allow: Sequence[str] = ADMIT) -> bool:
        c = self.get(task)
        return c is not None and c.status in allow


def require_certified(tasks: Sequence[Task], book: CertBook, allow: Sequence[str] = ADMIT) -> list[str]:
    """The parts' statuses; ``CompositionError`` when one has no admitted certificate for its exact version (digest)."""
    out, bad = [], []
    for k, t in enumerate(tasks):
        c = book.get(t)
        out.append(c.status if c is not None else "uncertified")
        if c is None or c.status not in allow:
            bad.append(f"part {k + 1} ({t.id} v{t.version}): {c.status if c else 'no certificate for this version'}")
    if bad:
        raise CompositionError("composition uses only parts certified or provisional on their own World state: " + "; ".join(bad))
    return out


def cert_of(task: Task, rec: Any) -> PartCert:
    return PartCert(task_id=task.id, version=task.version, digest=task.digest(), status=rec.status, reasons=list(rec.reasons)[:12],
                    classes=dict(rec.classes), rounds=rec.rounds, repairs=rec.repairs_accepted, note=rec.note[:300], at=time.time())


async def certify_part(runner: Any, task: Task, book: CertBook, *, n: int) -> tuple[Task, PartCert]:
    """One part through the full population, adjudication, its own repair loop and certification (``Runner.refine_one``
    on the part's own World). Returns the version certification reached and its certificate (also put in ``book``)."""
    from crucible.synth.pipeline import TaskRecord

    rec = TaskRecord(n=n, id=task.id, seed=task.seed, skill=task.intent.skill[:120], stage="contract")
    runner.report.tasks.append(rec)
    await runner.refine_one(rec, task)
    runner.save()
    final = runner.final_tasks.get(task.id, task)
    cert = cert_of(final, rec)
    book.put(cert)
    if final.version != task.version and rec.status not in ADMIT:  # the input version is not certified either
        book.put(cert.model_copy(update={"version": task.version, "digest": task.digest()}))
    return final, cert


# ---------------------------------------------------------------------------------------------------------- rebasing

def prefix_of(earlier: Sequence[Task]) -> Program:
    return concat([rename(t.oracle, f"t{k + 1}_") for k, t in enumerate(earlier)])


def rebase(task: Task, earlier: Sequence[Task]) -> Task:
    """``task`` on the World state the (possibly repaired) ``earlier`` parts leave: a new version with a new prefix."""
    spec = split_world(task.world)
    base = split_world(earlier[0].world) if earlier else spec
    new = spec.model_copy(update={"prefix": concat([*([base.prefix] if base.prefix is not None and base.prefix.steps else []), prefix_of(earlier)])
                                  if earlier else base.prefix}, deep=True)
    if new.model_dump_json() == spec.model_dump_json():
        return task
    return task.next_version(world=new.ref())


async def still_holds(task: Task, ctx: FalsifyContext) -> str | None:
    """Does the part's program still run and pass its own contract on its World (after a rebase)?"""
    try:
        r = await ctx.oracle(task)
    except Exception as e:  # noqa: BLE001 - the prefix does not replay
        return f"{type(e).__name__}: {str(e)[:200]}"
    errs = [r.error] if r.error else [c.obs.error for c in r.calls if c.obs.error]
    if errs:
        return f"the program no longer runs: {str(errs[0])[:200]}"
    v = grade(r, task.contract)
    return None if v.passed else f"the program fails its own contract: {v.reasons[:4]}"


async def certify_links(links: Sequence[Task], runner: Any, book: CertBook, ctx: FalsifyContext, *, log: list[dict[str, Any]] | None = None,
                        n0: int = 1) -> list[Task]:
    """Certify the links of a chain in order on their own World states; a link whose earlier links were repaired is
    rebased first. Returns the admitted prefix of the chain (the final versions); the chain is cut at the first link
    that is not admitted."""
    out: list[Task] = []
    for j, link in enumerate(links):
        row: dict[str, Any] = {"link": j + 1, "id": link.id, "version_in": link.version}
        t = rebase(link, out) if j else link
        if t.version != link.version:
            why = await still_holds(t, ctx)
            row["rebased"] = True
            if why:
                row.update(status="rebase_failed", reason=why)
                if log is not None:
                    log.append(row)
                break
        c = book.get(t)
        if c is not None and c.status in ADMIT:
            final, cert = t, c
            row["reused"] = True
        else:
            final, cert = await certify_part(runner, t, book, n=n0 + j)
        row.update(status=cert.status, reasons=cert.reasons[:6], version_out=final.version, rounds=cert.rounds, repairs=cert.repairs)
        if log is not None:
            log.append(row)
        if cert.status not in ADMIT:
            break
        out.append(final)
        if runner.stop:
            break
    return out


# ---------------------------------------------------------------------------------------------------------- attribution

PART_STEP = re.compile(r"(?<![A-Za-z0-9])t(\d+)_")
PART_PAT = re.compile(r"(?<![A-Za-z0-9])(?:x\.)?t(\d+)\.")
SKEL = re.compile(r"^x\.r(\d+)$")


def _arg_refs(a: Arg) -> list[str]:
    if isinstance(a, Ref):
        return [a.path.split(".", 1)[0]]
    if isinstance(a, Fn):
        return [h for x in a.args for h in _arg_refs(x)]
    return []


def program_owners(probe: Program, oracle: Program, comp: Composition) -> set[int]:
    """Parts whose steps (or answer element) a probe program changes, drops or adds, relative to the composite oracle."""
    from crucible.synth.compose import part_of_step

    want = {s.id: cjson(s.model_dump(mode="json")) for s in oracle.steps}
    got = {s.id: cjson(s.model_dump(mode="json")) for s in probe.steps}
    out: set[int] = set()
    for sid, body in want.items():
        if got.get(sid) != body:
            out.add(part_of_step(sid, comp))
    prev: int | None = None
    for s in probe.steps:
        if s.id in want:
            prev = part_of_step(s.id, comp)
        else:  # a step the probe adds: its own prefix, else the part of a step it refers to, else the one before it
            m = PART_STEP.match(s.id)
            refs = [h for v in s.args.values() for h in _arg_refs(v) if h in want]
            out.add(int(m.group(1)) - 1 if m else (part_of_step(refs[0], comp) if refs else (prev if prev is not None else 0)))
    if cjson(probe.answer.model_dump(mode="json") if probe.answer is not None else None) != cjson(oracle.answer.model_dump(mode="json") if oracle.answer is not None else None):
        a, b = probe.answer, oracle.answer
        if len(comp.answer_parts) > 1 and isinstance(b, Fn) and b.name == "list":
            for k, i in enumerate(comp.answer_parts):
                el = a.args[k] if isinstance(a, Fn) and a.name == "list" and k < len(a.args) else None
                if el is None or cjson(el.model_dump(mode="json")) != cjson(b.args[k].model_dump(mode="json")):
                    out.add(i)
        elif comp.answer_parts:
            out.add(comp.answer_parts[0])
    return {j for j in out if 0 <= j < max(1, len(comp.segments))}


def _ids_parts(texts: Sequence[str], n: int) -> set[int]:
    out = set()
    for t in texts:
        for m in PART_STEP.finditer(t):
            out.add(int(m.group(1)) - 1)
        for m in PART_PAT.finditer(t):
            out.add(int(m.group(1)) - 1)
    return {j for j in out if 0 <= j < n}


class Attribution(CrucibleModel):
    cx: str
    cls: str
    op: str
    parts: list[int] = Field(default_factory=list)
    by: str = ""  # program | site | reasons | address | none


async def attribute(cx: Counterexample, task: Task, comp: Composition, *, writers: Mapping[str, int] | None = None) -> Attribution:
    """The parts that own a composite counterexample (module docstring). ``writers``: address -> the last part that
    wrote it (for rebased skeleton patterns ``x.r<k>`` and forbidden effects)."""
    n = max(1, len(comp.segments))
    p = cx.probe
    if p.program is not None and cx.cls == "MUT":
        own = program_owners(p.program, task.oracle, comp)
        if own:
            return Attribution(cx=cx.id, cls=cx.cls, op=cx.op, parts=sorted(own), by="program")
    own = _ids_parts([cx.site, *[str(x) for x in (p.data or {}).values() if isinstance(x, str)]], n)
    if own:
        return Attribution(cx=cx.id, cls=cx.cls, op=cx.op, parts=sorted(own), by="site")
    codes = [r.split(":", 1)[1] for r in cx.reasons_seen if ":" in r and not r.startswith("FORBIDDEN_EFFECT:")]
    own = _ids_parts([c + "." if not c.endswith(".") else c for c in codes], n)
    skel = [int(m.group(1)) for c in codes if (m := SKEL.match(c))]
    if skel and writers:
        own |= {writers[a] for a in writers if any(a.endswith(f"#r{k}") for k in skel)}
    if writers:
        for r in cx.reasons_seen:
            if r.startswith("FORBIDDEN_EFFECT:"):
                a = r.split(":", 1)[1]
                if a in writers:
                    own.add(writers[a])
    if own:
        return Attribution(cx=cx.id, cls=cx.cls, op=cx.op, parts=sorted(own), by="reasons")
    if p.program is not None:
        own = program_owners(p.program, task.oracle, comp)
        if own:
            return Attribution(cx=cx.id, cls=cx.cls, op=cx.op, parts=sorted(own), by="program")
    return Attribution(cx=cx.id, cls=cx.cls, op=cx.op, parts=[], by="none")


async def last_writers(task: Task, comp: Composition, ctx: FalsifyContext) -> dict[str, int]:
    """Address (and ``#r<k>`` for the composite skeleton's k-th change) -> the last part that wrote it."""
    from crucible.synth.compose import merge_normalizers

    w = await ctx.world(task)
    parts = part_programs(task, comp)
    segs = await run_parts(w, parts, seed=task.seed, given=task.intent.given)
    n = task.contract.normalizers or merge_normalizers([])
    out: dict[str, int] = {}
    for j, r in enumerate(segs):
        if r.start is not None and r.end is not None:
            for ch in diff(r.start, r.end, n):
                out[str(ch.addr)] = j
    full = await ctx.oracle(task)
    if full.start is not None and full.end is not None:
        for k, ch in enumerate(diff(full.start, full.end, n)):
            if str(ch.addr) in out:
                out[f"{ch.addr}#r{k}"] = out[str(ch.addr)]
    return out


# ---------------------------------------------------------------------------------------------------------- translation

def _strip(text: str, j: int) -> str:
    return text.replace(f"x.t{j + 1}.", "").replace(f"t{j + 1}.", "").replace(f"t{j + 1}_", "")


def _strip_arg(a: Arg, j: int) -> Arg:
    if isinstance(a, Ref):
        return Ref(path=_strip(a.path, j))
    if isinstance(a, Fn):
        return Fn(name=a.name, args=[_strip_arg(x, j) for x in a.args])
    return Lit(v=a.v)


def part_slice(probe: Program, oracle: Program, comp: Composition, j: int) -> Program:
    """Part j's share of a composite probe program, in the part's own step names (module docstring)."""
    from crucible.synth.compose import part_of_step

    want = {s.id for s in oracle.steps}
    steps: list[Step] = []
    cur = 0
    for s in probe.steps:
        if s.id in want:
            cur = part_of_step(s.id, comp)
        else:
            m = PART_STEP.match(s.id)
            if m:
                cur = int(m.group(1)) - 1
        if cur == j:
            steps.append(Step(id=_strip(s.id, j), tool=s.tool, args={k: _strip_arg(v, j) for k, v in s.args.items()}))
    ans: Arg | None = None
    a = probe.answer
    if j in comp.answer_parts:
        if len(comp.answer_parts) > 1:
            k = comp.answer_parts.index(j)
            ans = a.args[k] if isinstance(a, Fn) and a.name == "list" and k < len(a.args) else a
        else:
            ans = a
    return Program(steps=steps, answer=_strip_arg(ans, j) if ans is not None else None)


def to_part(cx: Counterexample, task: Task, comp: Composition, j: int, part: Task) -> Counterexample:
    """A composite counterexample in part j's frame (module docstring)."""
    p = cx.probe
    upd: dict[str, Any] = {"instruction": None, "contract": None, "site": _strip(p.site, j),
                           "data": {**{k: v for k, v in (p.data or {}).items() if k != "finds"}, "from_composite": cx.id, "composite": task.id, "part": j + 1}}
    if p.program is not None:
        upd["program"] = part_slice(p.program, task.oracle, comp, j)
    if p.perturb:
        upd["perturb"] = []  # composite start-state edits do not apply to the part's World as they are
    q = p.model_copy(update=upd, deep=True)
    if p.perturb or p.instruction is not None or p.contract is not None:
        q = q.model_copy(update={"expect": "property"})
    reasons = []
    for r in cx.reasons_seen:
        code, _, rest = r.partition(":")
        if code == "FORBIDDEN_EFFECT" or not rest:
            reasons.append(r)
        elif re.match(rf"^(?:x\.)?t{j + 1}\.", rest):
            reasons.append(f"{code}:{_strip(rest, j)}")
    site = _strip(cx.site, j)
    cid = content_hash({"task": part.id, "cls": cx.cls, "op": cx.op, "site": site, "probe": q.digest(), "defect": cx.defect, "from": cx.id}, 24)
    return Counterexample(id=f"cx-{cid}", domain=part.domain, task_id=part.id, task_version=part.version, cls=cx.cls, op=cx.op, site=site,
                          defect=cx.defect, label=cx.label, probe=q, reasons_seen=reasons)


async def reproduces(pcx: Counterexample, part: Task, ctx: FalsifyContext) -> bool:
    """Does the part's own contract give the wrong verdict on the translated probe (and is the probe no equivalent of the
    part's oracle)? Then it is a part-level counterexample and is archived under the part."""
    p = pcx.probe
    if p.program is None or p.expect not in ("fail", "pass"):
        return False
    try:
        out = await BaseFalsifier.execute(_Prog(), p, part, ctx)
    except Exception:  # noqa: BLE001 - a slice that cannot run on the part's World reproduces nothing
        return False
    if not out.valid or out.passed is None:
        return False
    if p.expect == "fail":
        return bool(out.passed) and not bool(out.equivalent)
    return not out.passed


class _Prog(BaseFalsifier):
    name = "MUT"


# ---------------------------------------------------------------------------------------------------------- per-part repair

class PartRepairLog(CrucibleModel):
    composite: str
    from_version: int
    at: float
    attributions: list[Attribution] = Field(default_factory=list)
    composition_level: list[str] = Field(default_factory=list)  # counterexample ids no part reproduces (the merge lost the check)
    archived_parts: list[str] = Field(default_factory=list)
    parts: list[dict[str, Any]] = Field(default_factory=list)  # per owner part: repair result, recertification
    rebased: list[dict[str, Any]] = Field(default_factory=list)
    result: str = ""  # candidate | declined: ...


class PartRepairer:
    """The composite's repairer (module docstring): attribute, translate, repair the owner parts, recertify, rebase,
    recompose. ``parts`` are the composite's certified parts; ``runner`` is the parts' Runner (its default repairer,
    its archive, its panel); ``writer`` writes the recomposed instruction; ``book`` holds the part certificates."""

    def __init__(self, composed: Composed, parts: Sequence[Task], runner: Any, book: CertBook, *, writer: Any = None, log_path: Path | None = None,
                 n0: int = 100):
        self.parts = list(parts)
        self.comp = composed.comp
        self.runner, self.book, self.writer, self.log_path = runner, book, writer, log_path
        self.versions: dict[int, tuple[list[Task], Composed]] = {composed.task.version: (list(parts), composed)}
        self.n = n0
        self.logs: list[PartRepairLog] = []

    def state(self, version: int) -> tuple[list[Task], Composed] | None:
        return self.versions.get(version)

    def _log(self, row: PartRepairLog) -> None:
        self.logs.append(row)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as fh:
                fh.write(row.model_dump_json() + "\n")

    async def __call__(self, task: Task, cxs: list[Counterexample], ctx: FalsifyContext) -> Task | None:
        st = self.versions.get(task.version)
        if st is None:
            return None
        parts, cur = st
        comp = cur.comp
        row = PartRepairLog(composite=task.id, from_version=task.version, at=time.time())
        writers = await last_writers(task, comp, ctx)
        by_part: dict[int, list[Counterexample]] = {}
        for cx in cxs:
            at = await attribute(cx, task, comp, writers=writers)
            row.attributions.append(at)
            reproduced_any = False
            for j in at.parts:
                pcx = to_part(cx, task, comp, j, parts[j])
                if await reproduces(pcx, parts[j], ctx):
                    reproduced_any = True
                    row.archived_parts += self.runner.archive.add_all([pcx])
                by_part.setdefault(j, []).append(pcx)
            if not at.parts or (cx.probe.program is not None and cx.probe.expect in ("fail", "pass") and cx.cls == "MUT" and not reproduced_any):
                row.composition_level.append(cx.id)
        if not by_part:
            row.result = "declined: no part owns these counterexamples"
            self._log(row)
            return None
        new_parts = list(parts)
        first_changed: int | None = None
        for j in sorted(by_part):
            info: dict[str, Any] = {"part": j + 1, "id": parts[j].id, "counterexamples": [c.id for c in by_part[j]]}
            cand = await self.repair_one(parts[j], by_part[j], info)
            row.parts.append(info)
            if cand is None:
                continue
            new_parts[j] = cand
            if cand.oracle.digest() != parts[j].oracle.digest():
                first_changed = j if first_changed is None else min(first_changed, j)
        if all(a.digest() == b.digest() for a, b in zip(new_parts, parts, strict=True)):
            row.result = "declined: no owner part was repaired and recertified"
            self._log(row)
            return None
        if first_changed is not None:
            for k in range(first_changed + 1, len(new_parts)):
                t = rebase(new_parts[k], new_parts[:k])
                if t.digest() == new_parts[k].digest():
                    continue
                pctx = FalsifyContext(max_mutants=self.runner.cfg.max_mutants)
                try:
                    why = await still_holds(t, pctx)
                finally:
                    await pctx.close()
                if why:
                    row.rebased.append({"part": k + 1, "status": "rebase_failed", "reason": why})
                    row.result = f"declined: part {k + 1} does not hold after the repaired part {first_changed + 1}"
                    self._log(row)
                    return None
                self.n += 1
                final, cert = await certify_part(self.runner, t, self.book, n=self.n)
                row.rebased.append({"part": k + 1, "status": cert.status, "version": final.version, "reasons": cert.reasons[:4]})
                if cert.status not in ADMIT:
                    row.result = f"declined: rebased part {k + 1} is {cert.status}"
                    self._log(row)
                    return None
                new_parts[k] = final
        try:
            c2 = await compose(new_parts, ctx=ctx, writer=self.writer, task_id=task.id, tag=f"compose:{task.id}#v{task.version + 1}", certs=self.book)
        except CompositionError as e:
            row.result = f"declined: recomposition failed: {str(e)[:300]}"
            self._log(row)
            return None
        nt = c2.task.model_copy(update={"version": task.version + 1}, deep=True)
        c2 = Composed(task=nt, comp=c2.comp, run=c2.run)
        self.versions[nt.version] = (new_parts, c2)
        row.result = "candidate"
        self._log(row)
        return nt

    async def repair_one(self, part: Task, pcxs: list[Counterexample], info: dict[str, Any]) -> Task | None:
        """The repairer role on one part (the parts' Runner's own repairer: the part's World catalogue), then the part's
        monotone archive and its recertification."""
        from crucible.synth.pipeline import TaskRecord

        self.n += 1
        rec = TaskRecord(n=self.n, id=part.id, seed=part.seed, skill=part.intent.skill[:120], stage="contract")
        pctx = FalsifyContext(max_mutants=self.runner.cfg.max_mutants)
        try:
            cand = await self.runner.default_repairer(rec, pctx)(part, pcxs)
            if cand is None:
                info["repair"] = "declined or failed"
                return None
            every = [*self.runner.cheap_factory()]
            reg = await regress(self.runner.archive, cand, pctx, every)
            open_now = {c.id for c in pcxs}
            broken = sorted(set(reg.reopened) - open_now)
            if broken:
                info["repair"], info["reopened"] = "refused (re-opens archived part counterexamples)", broken
                return None
        finally:
            await pctx.close()
        final, cert = await certify_part(self.runner, cand, self.book, n=self.n)
        info.update(repair="accepted", version=final.version, status=cert.status, reasons=cert.reasons[:6])
        return final if cert.status in ADMIT else None


def part_repair_factory(repairers: Mapping[str, PartRepairer]) -> Any:
    """``Runner.repair_factory`` for a composite Runner: the composite's ``PartRepairer`` (keyed by composite id)."""

    def make(runner: Any, rec: Any, ctx: FalsifyContext) -> Any:
        pr = repairers.get(rec.id)
        if pr is None:
            async def none(task: Task, cxs: list[Counterexample]) -> Task | None:
                return None
            return none

        async def repair(task: Task, cxs: list[Counterexample]) -> Task | None:
            return await pr(task, cxs, ctx)

        return repair

    return make


__all__ = ["ADMIT", "Attribution", "CertBook", "PartCert", "PartRepairLog", "PartRepairer", "attribute", "certify_links", "certify_part",
           "last_writers", "part_repair_factory", "part_slice", "prefix_of", "program_owners", "rebase", "reproduces", "require_certified",
           "still_holds", "to_part"]
