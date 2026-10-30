"""The batch runner: many candidates through the synthesis pipeline at once, resumable, capped, measured (E6).

One **batch** runs one or more **arms** (a prompt variant and its yield settings) side by side over a **cell grid**
(domain × World instance × cell kind). Each arm has its own run directory, its own model clients and its own hard
spend cap, so two arms are compared at equal spend on fresh proposals.

**Pipelined stages.** Per arm, three worker pools pass candidates through queues, so candidate i+1 is proposed while
candidate i is falsified:

    propose (P workers, one proposer call = one batch for one cell) → dedup → pre-filters
      → build (B workers: oracle executed, screen, contract writer)  → calibration of the panel (once per domain)
      → refine (R workers: ``cegis.refine`` with the deterministic classes in worker processes (``offload.FalsifyPool``),
        ALT/ATK agents concurrently (``offload.ConcurrentAgents``), adjudication, repair, certification)

Every model call of every role goes through ``Resilient``: one semaphore bounds the calls in flight over all arms and
roles, transient errors are retried with exponential backoff and jitter, and the call's latency is booked per stage. The
stage code is ``pipeline.Runner``'s own (``build``, ``refine_one``, ``calibrate_panel``), with its hooks; nothing about
falsification or admission is re-implemented here. ``serial=True`` runs the same code one candidate and one call at a
time, in-process, with no concurrent agents: the serial baseline of the speed comparison.

**Resumable.** Each candidate is a JSON state file ``<arm>/cands/<id>.json`` written atomically after every transition;
each proposer reply is ``<arm>/props/<cell>-b<k>.json``, written before its candidates. A restart reloads both and
continues: finished candidates are kept, built ones go straight to refine, unbuilt ones are built. A refine that was
cut off restarts under a fresh attempt id (``<id>~2``), so the counterexamples the cut-off attempt archived stay on
record and never act as regressions of the new attempt (the outcome does not depend on where the crash fell).

**Spend.** The arm's wire holds a hard cap for the run and, optionally, per stage (``factory.synth_wire``: every call
holds its worst case first). ``stop_proposing_usd`` is the soft limit: no new proposals once an arm has spent it, so
candidates already in flight finish inside the hard cap. A ``BudgetExceeded`` stops the arm: its open candidates are
recorded as ``budget``.

**Scheduling.** A cell stays open while it has fewer admitted tasks than its quota and fewer proposals than its maximum;
among open cells the proposer takes the one with the largest deficit after counting in-flight candidates at the running
yield (``(admitted + 1) / (finished + 1/prior)``), ties to the cell with the fewest proposals.

**Metrics** (``arm_metrics``): the funnel (proposed → unique → pre-filter pass → oracle executed → contract → falsified
→ repaired → admitted), cost per stage (``usage.jsonl``) and per candidate (its tags; a proposer call is split over the
batch it produced), admitted per dollar and cost per admitted with bootstrap intervals over candidates, yield with a
Wilson interval, and speed: wall clock, admitted per hour, wall per admitted, busy time per stage, model latency per
stage, and the serial-equivalent time (the sum of every candidate's stage times) over the wall clock.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import Field

from crucible.actors.llm import BudgetExceeded, LLMOutputError
from crucible.core.schema.base import CrucibleModel
from crucible.synth.meta.recall import wilson
from crucible.synth.offload import ConcurrentAgents, FalsifyPool
from crucible.synth.pipeline import CELLS, DomainEnv, RunConfig, Runner, RunRoles, TaskRecord, effects_view
from crucible.synth.program import Run
from crucible.synth.roles import ProposalOut, RoleClient, RoleFailed, RoleLeak, ask, compact, tool_catalogue, variant
from crucible.synth.task import Task
from crucible.synth.yieldkit import Deduper, agreement, clarity_screen, pre_checks, reconcile

# ---------------------------------------------------------------------------------------------------------- config


@dataclass(frozen=True)
class Arm:
    """One arm: the prompt variant and the yield settings it changes (everything else is the batch's)."""

    name: str
    variant: str = "v1"
    oracle_sees_derive: bool = True
    alt_finish_nudge: bool = False
    pre: bool = False  # pre_checks before the oracle (unknown entity, leaked derive value)
    agree: str = "off"  # off | reconcile | drop: the independent oracle against the proposer's derive values
    clarity: bool = False  # the instruction-clarity self-check (one CLARITY_CHECKER call per candidate)
    lemmas: tuple[str, ...] = ()
    writer_lemmas: tuple[str, ...] = ()

    @classmethod
    def baseline(cls, name: str = "a0") -> Arm:
        return cls(name=name)

    @classmethod
    def improved(cls, name: str = "a1", **kw: Any) -> Arm:
        return cls(name=name, variant="v2", oracle_sees_derive=False, alt_finish_nudge=True, pre=True, agree="reconcile", **kw)


@dataclass(frozen=True)
class CellSpec:
    domain: str
    instance: int
    kind: str
    index: int
    quota: int = 1
    max_proposals: int = 8

    @property
    def key(self) -> str:
        return f"{self.domain}.s{self.instance}.k{self.index}"


@dataclass
class BatchDomain:
    """A domain of the grid: its ``DomainEnv`` (``seeds`` are its World instances) and its cell kinds. A ``DomainSpec``
    with a cell grid (the World-family work) is adapted by ``batch_domains``."""

    env: DomainEnv
    cells: Sequence[str] = CELLS
    quota: int = 1
    max_proposals: int = 8
    truth: Callable[[Mapping[str, Any]], str | None] | None = None  # executed ground truth of a task's oracle, when the domain has it

    def grid(self) -> list[CellSpec]:
        return [CellSpec(self.env.name, int(s), k, i, self.quota, self.max_proposals) for s in self.env.seeds for i, k in enumerate(self.cells)]


def batch_domains(specs: Sequence[Any], **kw: Any) -> list[BatchDomain]:
    """``BatchDomain``s from ``DomainEnv``s, ``BatchDomain``s, or objects with ``env`` and ``cells`` (a cell may be text or
    have ``describe()`` / ``text``): the interface a DomainSpec with a cell grid needs to meet."""
    out = []
    for s in specs:
        if isinstance(s, BatchDomain):
            out.append(s)
        elif isinstance(s, DomainEnv):
            out.append(BatchDomain(s, **kw))
        else:
            cells = [c if isinstance(c, str) else (c.describe() if hasattr(c, "describe") else str(getattr(c, "text", c))) for c in s.cells]
            out.append(BatchDomain(s.env, cells=cells, **kw))
    return out


@dataclass
class BatchConfig:
    run: str
    run_cfg: RunConfig = field(default_factory=RunConfig)  # rounds, agent budgets, λ, calibration: shared by every arm
    batch: int = 4  # proposals per proposer call
    propose_workers: int = 2
    build_workers: int = 6
    refine_workers: int = 6
    model_calls: int = 24  # model calls in flight, over every arm and role
    falsify_procs: int = 4  # worker processes for the deterministic classes (0: in-process)
    retries: int = 3
    backoff_s: float = 2.0
    stop_proposing_usd: float | None = None  # per arm (soft); the hard cap is the arm wire's
    max_candidates: int | None = None  # per arm
    max_backlog: int | None = None  # candidates waiting for build or refine before the proposer pauses (default: build + refine workers; yield-2 at 2 × queued 25 built tasks)
    prior_yield: float = 0.2
    near_dup: float = 0.8
    serial: bool = False

    def effective(self) -> BatchConfig:
        if not self.serial:
            return self
        return BatchConfig(**{**self.__dict__, "propose_workers": 1, "build_workers": 1, "refine_workers": 1, "model_calls": 1, "falsify_procs": 0})


# ---------------------------------------------------------------------------------------------------------- candidates

class Cand(CrucibleModel):
    id: str
    n: int
    arm: str
    domain: str
    seed: int
    cell: str
    cell_key: str
    batch_key: str
    stage: str = "proposed"  # proposed | built | refining | done
    status: str = "pending"  # pending | duplicate | filtered | dropped | certified | provisional | rejected | error | budget
    reasons: list[str] = Field(default_factory=list)
    proposal: dict[str, Any] = Field(default_factory=dict)
    dup_of: str | None = None
    checks: dict[str, Any] = Field(default_factory=dict)
    task: dict[str, Any] | None = None
    record: dict[str, Any] | None = None
    attempt: int = 1
    times: dict[str, list[float]] = Field(default_factory=dict)  # stage -> [start, end] (epoch seconds)
    truth: str | None = None

    @property
    def task_id(self) -> str:
        return self.id + (f"~{self.attempt}" if self.attempt > 1 else "")


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _jwrite(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(p)


# ---------------------------------------------------------------------------------------------------------- the client wrapper

def budget_cause(e: BaseException) -> BudgetExceeded | None:
    """The budget refusal behind an exception (its cause chain, or a ``failure_kind`` of credits exhausted), if any."""
    seen: set[int] = set()
    x: BaseException | None = e
    while x is not None and id(x) not in seen:
        seen.add(id(x))
        if isinstance(x, BudgetExceeded):
            return x
        if getattr(x, "failure_kind", None) == BudgetExceeded.failure_kind:
            return BudgetExceeded(str(x))
        x = x.__cause__ or x.__context__
    return None


class Resilient:
    """A ``RoleClient`` behind the batch's call gate, with retries and per-stage latency (module docstring)."""

    def __init__(self, inner: RoleClient, gate: asyncio.Semaphore, *, retries: int = 3, backoff_s: float = 2.0,
                 timing: Callable[[str, float], None] | None = None, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.inner, self.gate, self.retries, self.backoff_s, self.timing, self.sleep = inner, gate, retries, backoff_s, timing, sleep
        self.family, self.model = inner.family, inner.model
        self.retried = 0

    def __getattr__(self, k: str) -> Any:  # calls, tags, usage of the wrapped client
        return getattr(self.inner, k)

    async def complete_json(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        tag = str(kw.get("tag") or "")
        for attempt in range(self.retries + 1):
            try:
                async with self.gate:
                    t0 = time.monotonic()
                    try:
                        return await self.inner.complete_json(messages, **kw)
                    finally:
                        if self.timing is not None:
                            self.timing(tag, time.monotonic() - t0)
            except (BudgetExceeded, LLMOutputError, RoleLeak, ValueError, asyncio.CancelledError):
                raise
            except Exception as e:  # a transient failure of the provider: back off and try again
                cap = budget_cause(e)
                if cap is not None:  # a wrapped budget refusal is a stop, never retried (yield-2)
                    raise cap from e
                if attempt >= self.retries:
                    raise
                self.retried += 1
                await self.sleep(self.backoff_s * (2**attempt) * (0.5 + random.random()))
        raise RuntimeError("unreachable")


def wrap_roles(roles: RunRoles, wrap: Callable[[RoleClient], RoleClient]) -> RunRoles:
    return RunRoles(proposer=wrap(roles.proposer), oracle_programmer=wrap(roles.oracle_programmer), contract_writer=wrap(roles.contract_writer),
                    repairer=wrap(roles.repairer), alt=[(n, wrap(c)) for n, c in roles.alt], atk=[(n, wrap(c)) for n, c in roles.atk],
                    panel=[(n, wrap(c)) for n, c in roles.panel], abstractor=wrap(roles.abstractor) if roles.abstractor is not None else None)


# ---------------------------------------------------------------------------------------------------------- one arm

@dataclass
class ArmSetup:
    """What the CLI hands an arm: its roles (on its own wire and cap), the clarity checker, its cost and usage."""

    arm: Arm
    roles: RunRoles
    cost: Callable[[], dict[str, Any]]
    usage_path: Path | None = None
    clarity: RoleClient | None = None
    families: Mapping[str, Any] | None = None


class ArmRun:
    def __init__(self, setup: ArmSetup, domains: Sequence[BatchDomain], cfg: BatchConfig, root: Path, gate: asyncio.Semaphore,
                 pool: FalsifyPool | None):
        self.s, self.arm, self.cfg, self.pool = setup, setup.arm, cfg, pool
        self.dir = Path(root) / _safe(self.arm.name)
        self.domains = {d.env.name: d for d in domains}
        self.cells = [c for d in domains for c in d.grid()]
        self.timing: dict[str, list[float]] = defaultdict(list)
        wrap = (lambda c: Resilient(c, gate, retries=cfg.retries, backoff_s=cfg.backoff_s, timing=self._time))
        self.roles = wrap_roles(setup.roles, wrap)
        self.clarity = wrap(setup.clarity) if setup.clarity is not None else None
        self.cands: dict[str, Cand] = {}
        self.dedup = Deduper(cfg.near_dup)
        self.runners: dict[str, Runner] = {}
        self.world_text: dict[tuple[str, int], str] = {}
        self.calibrated: dict[str, asyncio.Event] = {}
        self.anchors: dict[str, list[tuple[TaskRecord, Task, Run]]] = defaultdict(list)
        self.cal_lock = asyncio.Lock()
        self.build_q: asyncio.Queue[Cand | None] = asyncio.Queue()
        self.refine_q: asyncio.Queue[Cand | None] = asyncio.Queue()
        self.stop: str | None = None
        self.proposing_done = asyncio.Event()
        self.started = time.time()
        self.batches: Counter[str] = Counter()
        self.reserved: Counter[str] = Counter()  # proposals requested by proposer calls in flight, per cell
        self.empty: Counter[str] = Counter()  # proposals requested by calls that returned none, per cell
        self.transcripts = self.dir / "transcripts"
        self.building = 0  # builds in progress

    # -- bookkeeping --------------------------------------------------------------------------------------------------
    def _time(self, tag: str, s: float) -> None:
        self.timing[(tag or "?").split(":", 1)[0]].append(round(s, 3))

    def save(self, c: Cand) -> None:
        _jwrite(self.dir / "cands" / f"{_safe(c.id)}.json", c.model_dump(mode="json"))

    def mark(self, c: Cand, stage: str, start: bool) -> None:
        t = c.times.setdefault(stage, [0.0, 0.0])
        t[0 if start else 1] = round(time.time(), 3)

    def spent(self) -> float:
        try:
            return float(self.s.cost().get("total_usd") or 0.0)
        except Exception:  # noqa: BLE001 - a cost callback that fails reports nothing spent
            return 0.0

    def counts(self) -> tuple[Counter[str], Counter[str], Counter[str]]:
        """(admitted, proposed, in flight) per cell; a proposer call in flight counts as its requested proposals, and a
        call that returned nothing counts against the cell's maximum (a cell the proposer fails on is not retried forever)."""
        adm: Counter[str] = Counter()
        prop: Counter[str] = self.reserved + self.empty
        infl: Counter[str] = Counter(self.reserved)
        for c in self.cands.values():
            prop[c.cell_key] += 1
            if c.status == "certified":
                adm[c.cell_key] += 1
            elif c.status == "pending":
                infl[c.cell_key] += 1
        return adm, prop, infl

    # -- setup and resume ---------------------------------------------------------------------------------------------
    async def setup(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        rc = self.cfg.run_cfg
        for name, d in self.domains.items():
            cfg = RunConfig(**{**rc.__dict__, "variant": self.arm.variant, "oracle_sees_derive": self.arm.oracle_sees_derive,
                               "alt_finish_nudge": self.arm.alt_finish_nudge, "lemmas": tuple(self.arm.lemmas) or rc.lemmas,
                               "writer_lemmas": tuple(self.arm.writer_lemmas) or rc.writer_lemmas})
            r = Runner(d.env, self.roles, cfg, self.dir / _safe(name), self.dir / "archive", run_id=f"{self.cfg.run}-{self.arm.name}",
                       cost=self.s.cost, families=self.s.families)
            await r.survey()
            r.proposals = {}
            if self.pool is not None:
                r.cheap_factory = self.pool.falsifiers
            if not self.cfg.serial:
                r.wrap_paid = lambda paid: [ConcurrentAgents(p) for p in paid]
            r.screen = self._screen
            r.agent_log = self._agent_log
            self.runners[name] = r
            self.calibrated[name] = asyncio.Event()
            for s in d.env.seeds:
                self.world_text[(name, int(s))] = await self._world_text(d.env, int(s))
            cal = r.dir / "calibration.json"
            if cal.exists():  # resumed: the panel's λ was calibrated before the restart
                r.panel = r.make_panel()
                if r.panel is not None:
                    r.panel.lam = float(json.loads(cal.read_text()).get("lam", rc.lam))
                r.report.calibration = json.loads(cal.read_text())
                self.calibrated[name].set()
            elif not rc.calibrate or len(self.roles.panel) < 2:
                r.panel = r.make_panel()
                self.calibrated[name].set()
        self.resume()

    async def _world_text(self, env: DomainEnv, seed: int) -> str:
        w = await env.world(seed).build()
        try:
            await w.reset(seed)
            c = await w.canon()
            return json.dumps(c.apps if c is not None else {}, default=str)
        finally:
            fn = getattr(w, "close", None)
            if fn is not None:
                res = fn()
                if asyncio.iscoroutine(res):
                    await res

    def resume(self) -> None:
        for p in sorted((self.dir / "cands").glob("*.json")) if (self.dir / "cands").exists() else []:
            c = Cand.model_validate_json(p.read_text())
            self.cands[c.id] = c
        for c in sorted(self.cands.values(), key=lambda x: x.n):
            if c.status not in ("duplicate",) and c.proposal.get("instruction"):
                self.dedup.add(c.id, f"{c.domain}/{c.seed}", c.proposal["instruction"], c.proposal.get("derive") or [])
            self.batches[c.cell_key] = max(self.batches[c.cell_key], int(c.batch_key.rsplit("-b", 1)[-1]) + 1)
        for p in sorted((self.dir / "props").glob("*.json")) if (self.dir / "props").exists() else []:  # replies whose candidates were never written
            doc = json.loads(p.read_text())
            if not any(c.batch_key == doc["batch_key"] for c in self.cands.values()):
                self._admit_batch(doc)
            ck, _, k = doc["batch_key"].rpartition("-b")
            self.batches[ck] = max(self.batches[ck], int(k) + 1)
        for c in sorted(self.cands.values(), key=lambda x: x.n):
            if c.status != "pending":
                continue
            if c.stage == "proposed":
                self.build_q.put_nowait(c)
            elif c.stage in ("built", "refining"):
                if c.stage == "refining":  # cut off mid-refine: a fresh attempt id (module docstring)
                    c.attempt += 1
                    c.stage = "built"
                    self.save(c)
                self.refine_q.put_nowait(c)

    # -- propose ------------------------------------------------------------------------------------------------------
    def pick(self) -> CellSpec | None:
        adm, prop, infl = self.counts()
        fin = sum(1 for c in self.cands.values() if c.status != "pending" and c.status != "duplicate")
        y = (sum(adm.values()) + 1.0) / (fin + 1.0 / max(self.cfg.prior_yield, 1e-6))
        best, key = None, None
        for c in self.cells:
            if adm[c.key] >= c.quota or prop[c.key] >= c.max_proposals:
                continue
            deficit = c.quota - adm[c.key] - infl[c.key] * y
            if deficit <= 0:
                continue
            k = (deficit, -prop[c.key])
            if key is None or k > key:
                best, key = c, k
        return best

    def proposing_open(self) -> bool:
        if self.stop:
            return False
        if self.cfg.stop_proposing_usd is not None and self.spent() >= self.cfg.stop_proposing_usd:
            return False
        return not (self.cfg.max_candidates is not None and len(self.cands) + sum(self.reserved.values()) >= self.cfg.max_candidates)

    def backlog(self) -> int:
        return sum(1 for c in self.cands.values() if c.status == "pending" and c.stage in ("proposed", "built")) + sum(self.reserved.values())

    async def propose_worker(self) -> None:
        while self.proposing_open():
            limit = self.cfg.max_backlog if self.cfg.max_backlog is not None else self.cfg.build_workers + self.cfg.refine_workers
            if self.backlog() >= limit:  # backpressure (yield-1: unthrottled proposing queued 30+ candidates per arm ahead of refine)
                await asyncio.sleep(0.5)
                continue
            cell = self.pick()
            if cell is None:
                if not any(c.status == "pending" for c in self.cands.values()):
                    return  # every cell is satisfied or exhausted
                await asyncio.sleep(0.2)  # wait for in-flight candidates to finish before proposing more
                continue
            k = self.batches[cell.key]
            self.batches[cell.key] += 1
            r = self.runners[cell.domain]
            want = min(self.cfg.batch, max(1, cell.max_proposals - self.counts()[1][cell.key]))
            if self.cfg.max_candidates is not None:
                want = max(1, min(want, self.cfg.max_candidates - len(self.cands) - sum(self.reserved.values())))
            self.reserved[cell.key] += want
            scope = f"{cell.domain}/{cell.instance}"
            existing = [c.proposal["instruction"] for c in self.cands.values() if f"{c.domain}/{c.seed}" == scope and c.proposal.get("instruction")]
            t0 = time.time()
            try:
                out: ProposalOut = await ask(self.roles.proposer, variant("proposer", self.arm.variant), tag=f"propose:{self.arm.name}/{cell.key}-b{k}",
                                             retries=1, domain=r.env.description, tools=tool_catalogue(r.tools[cell.instance]),
                                             world=compact(r.slices[cell.instance], 24000), cell=cell.kind, existing=existing[-30:] or None, count=want,
                                             lemmas=list(self.arm.lemmas) or None)
                tasks = [t.model_dump(mode="json") for t in out.tasks[:want]]
                err = None
            except BudgetExceeded as e:
                self.stop = f"budget: {e}"
                self.reserved[cell.key] -= want
                return
            except RoleFailed as e:
                tasks, err = [], str(e)[:500]
            doc = {"batch_key": f"{cell.key}-b{k}", "cell": cell.__dict__, "tasks": tasks, "error": err, "want": want,
                   "t": [round(t0, 3), round(time.time(), 3)]}
            _jwrite(self.dir / "props" / f"{_safe(doc['batch_key'])}.json", doc)
            self.reserved[cell.key] -= want
            self._admit_batch(doc)

    def _admit_batch(self, doc: Mapping[str, Any]) -> None:
        cell = CellSpec(**doc["cell"])
        t0, t1 = doc.get("t") or [time.time(), time.time()]
        if not doc["tasks"]:  # a failed proposer call still counts against the cell's maximum
            self.empty[cell.key] += int(doc.get("want") or 1)
        for j, t in enumerate(doc["tasks"]):
            cid = f"{self.arm.name}/{cell.key}-b{doc['batch_key'].rsplit('-b', 1)[-1]}-{j}"
            if cid in self.cands:
                continue
            c = Cand(id=cid, n=len(self.cands) + 1, arm=self.arm.name, domain=cell.domain, seed=cell.instance, cell=cell.kind, cell_key=cell.key,
                     batch_key=doc["batch_key"], proposal=dict(t), times={"propose": [t0, t1]})
            scope = f"{c.domain}/{c.seed}"
            kind, of = self.dedup.check(scope, t["instruction"], t.get("derive") or [])
            if kind is not None:
                c.status, c.stage, c.dup_of, c.reasons = "duplicate", "done", of, [f"dedup:{kind}"]
            else:
                self.dedup.add(cid, scope, t["instruction"], t.get("derive") or [])
                if self.arm.pre:
                    bad = pre_checks(t["instruction"], t.get("derive") or [], self.world_text.get((c.domain, c.seed), ""))
                    c.checks["pre"] = bad
                    if bad:
                        c.status, c.stage, c.reasons = "filtered", "done", bad[:4]
            self.cands[cid] = c
            self.save(c)
            if c.status == "pending":
                self.build_q.put_nowait(c)

    # -- build --------------------------------------------------------------------------------------------------------
    async def _screen(self, rec: TaskRecord, task: Task, r: Run) -> str | Task | None:
        c = self.cands[rec.id.split("~", 1)[0]]
        eff = effects_view(r, task.contract.normalizers)
        if self.arm.agree != "off":
            why = agreement(c.proposal.get("derive") or [], eff, r.answer)
            c.checks["agree"] = why
            if why and self.arm.agree == "drop":
                return why
            if why and self.arm.agree == "reconcile":
                new = reconcile(c.proposal.get("derive") or [], eff, r.answer, task.instruction)
                c.checks["derive_reconciled"] = {"from": c.proposal.get("derive"), "to": new}
                task = task.model_copy(update={"intent": task.intent.model_copy(update={"derive": new})}, deep=True)
        if self.arm.clarity and self.clarity is not None:
            runner = self.runners[c.domain]
            why, rec_ = await clarity_screen(self.clarity, instruction=task.instruction, tools=runner.tools[c.seed], world=runner.slices[c.seed],
                                             effects=eff, answer=r.answer, tag=f"clarity:{rec.id}")
            c.checks["clarity"] = {"drop": why, **rec_}
            if why:
                return why
        return task if c.checks.get("derive_reconciled") else None

    def _agent_log(self, rec: TaskRecord, name: str, row: dict[str, Any]) -> None:
        p = self.transcripts / f"{_safe(rec.id)}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps({"agent": name, **row}, default=str) + "\n")

    def _rec(self, c: Cand) -> TaskRecord:
        return TaskRecord(n=c.n, id=c.task_id, seed=c.seed, skill=str(c.proposal.get("skill", ""))[:120])

    async def build_worker(self) -> None:
        while True:
            c = await self.build_q.get()
            if c is None:
                return
            if self.stop:
                c.status, c.stage, c.reasons = "budget", "done", [self.stop[:300]]
                self.save(c)
                continue
            r = self.runners[c.domain]
            r.proposals[c.n] = {**c.proposal, "n": c.n, "seed": c.seed, "cell": c.cell}
            rec = self._rec(c)
            self.mark(c, "build", True)
            self.building += 1
            try:
                await r.build(rec)
            finally:
                self.building -= 1
            self.mark(c, "build", False)
            if r.stop and not self.stop:
                self.stop = r.stop
            got = next((x for x in r.ready if x[0] is rec), None)
            c.record = rec.model_dump(mode="json")
            if got is not None:
                r.ready.remove(got)
                c.task, c.stage = got[1].model_dump(mode="json"), "built"
                d = self.domains[c.domain]
                if d.truth is not None:
                    with contextlib.suppress(Exception):
                        c.truth = d.truth(c.task)
                self.save(c)
                if not self.calibrated[c.domain].is_set():
                    self.anchors[c.domain].append(got)
                await self.maybe_calibrate(c.domain)
                self.refine_q.put_nowait(c)
            else:
                c.status, c.reasons, c.stage = rec.status if rec.status != "pending" else "error", list(rec.reasons), "done"
                if not c.reasons and c.status == "error":
                    c.reasons = ["build: ended without a task or a status"]
                self.save(c)
                await self.maybe_calibrate(c.domain)

    async def maybe_calibrate(self, domain: str, force: bool = False) -> None:
        ev = self.calibrated[domain]
        if ev.is_set():
            return
        k = self.cfg.run_cfg.calib_tasks
        pending = any(c.domain == domain and c.status == "pending" and c.stage == "proposed" for c in self.cands.values())
        if not (force or len(self.anchors[domain]) >= k or (self.proposing_done.is_set() and not pending)):
            return
        async with self.cal_lock:
            if ev.is_set():
                return
            r = self.runners[domain]
            try:
                await r.calibrate_panel(ready=list(self.anchors[domain][:k]))
            finally:
                self.anchors[domain] = []
                if r.stop and not self.stop:
                    self.stop = r.stop
                ev.set()

    # -- refine -------------------------------------------------------------------------------------------------------
    async def refine_worker(self) -> None:
        while True:
            c = await self.refine_q.get()
            if c is None:
                return
            if self.stop:
                c.status, c.stage, c.reasons = "budget", "done", [self.stop[:300]]
                self.save(c)
                continue
            while not self.calibrated[c.domain].is_set():
                # nothing is being built and nothing waits to be: the anchors there are are all there will be for now (a
                # proposer paused by backpressure once waited for calibration that waited for more builds: speed-ser)
                idle = self.build_q.empty() and self.building == 0
                await self.maybe_calibrate(c.domain, force=idle and bool(self.anchors[c.domain]) or self.proposing_done.is_set() and idle)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.calibrated[c.domain].wait(), timeout=0.5)
            r = self.runners[c.domain]
            rec = self._rec(c)
            rec.stage = "contract"
            task = Task.model_validate(c.task).model_copy(update={"id": c.task_id})
            c.stage = "refining"
            self.mark(c, "refine", True)
            self.save(c)
            await r.refine_one(rec, task)
            self.mark(c, "refine", False)
            if r.stop and not self.stop:
                self.stop = r.stop
            if self.pool is not None:
                self.pool.forget(c.task_id)
            c.record = rec.model_dump(mode="json")
            c.status, c.reasons, c.stage = (rec.status if rec.status != "pending" else "error"), list(rec.reasons), "done"
            cj = r.dir / "tasks" / f"t{c.n:02d}" / "cegis.json"
            if cj.exists() and self.domains[c.domain].truth is not None:
                with contextlib.suppress(Exception):
                    c.truth = self.domains[c.domain].truth(json.loads(cj.read_text()).get("task") or {})
            self.save(c)

    # -- the arm ------------------------------------------------------------------------------------------------------
    async def run(self) -> dict[str, Any]:
        await self.setup()
        cfg = self.cfg
        builders = [asyncio.create_task(self.build_worker()) for _ in range(cfg.build_workers)]
        refiners = [asyncio.create_task(self.refine_worker()) for _ in range(cfg.refine_workers)]
        proposers = [asyncio.create_task(self.propose_worker()) for _ in range(cfg.propose_workers)]
        await asyncio.gather(*proposers)
        self.proposing_done.set()
        await self._drain(self.build_q, cfg.build_workers)
        await asyncio.gather(*builders)
        for d in self.calibrated:
            await self.maybe_calibrate(d, force=True)
        await self._drain(self.refine_q, cfg.refine_workers)
        await asyncio.gather(*refiners)
        for c in self.cands.values():
            if c.status == "pending":
                c.status, c.stage, c.reasons = ("budget", "done", [self.stop[:300]]) if self.stop else ("error", "done", ["not finished"])
                self.save(c)
        for name, r in self.runners.items():  # the pipeline's own report per domain (failures.mine_run, meta.rsi.load_tasks read it)
            recs = []
            for c in sorted(self.cands.values(), key=lambda x: x.n):
                if c.domain != name:
                    continue
                rec = TaskRecord.model_validate(c.record) if c.record else self._rec(c)
                rec.status, rec.reasons = c.status, list(c.reasons)
                recs.append(rec)
            r.report.tasks = recs
            r.report.finished = time.time()
            r.report.stopped = self.stop
            r.save()
        m = self.metrics()
        _jwrite(self.dir / "metrics.json", m)
        return m

    async def _drain(self, q: asyncio.Queue[Cand | None], n: int) -> None:
        while not q.empty():
            await asyncio.sleep(0.2)
        for _ in range(n):
            q.put_nowait(None)

    def metrics(self) -> dict[str, Any]:
        rows = _usage(self.s.usage_path) if self.s.usage_path is not None else []
        m = arm_metrics(list(self.cands.values()), rows, wall_s=time.time() - self.started, spent=self.spent())
        m["arm"] = self.arm.__dict__
        m["stopped"] = self.stop
        m["model_latency_s"] = {k: {"calls": len(v), "sum": round(sum(v), 1), "mean": round(sum(v) / len(v), 2)} for k, v in sorted(self.timing.items())}
        m["retried_calls"] = sum(getattr(c, "retried", 0) for c in self._clients())
        # shared by every arm of the batch: cumulative over all arms when this arm finished (worker wall time per job)
        m["falsify_pool_shared"] = {"workers": self.pool.workers, "jobs": self.pool.jobs, "busy_s": round(self.pool.busy_s, 1)} if self.pool is not None else None
        return m

    def _clients(self) -> list[Any]:
        r = self.roles
        return [r.proposer, r.oracle_programmer, r.contract_writer, r.repairer, *(c for _, c in r.alt), *(c for _, c in r.atk), *(c for _, c in r.panel),
                *([r.abstractor] if r.abstractor is not None else []), *([self.clarity] if self.clarity is not None else [])]


# ---------------------------------------------------------------------------------------------------------- the batch

async def run_batch(setups: Sequence[ArmSetup], domains: Sequence[BatchDomain], cfg: BatchConfig, root: str | Path) -> dict[str, Any]:
    """Every arm of a batch at once over the same grid (module docstring); returns each arm's metrics."""
    cfg = cfg.effective()
    gate = asyncio.Semaphore(cfg.model_calls)
    pool = FalsifyPool(cfg.falsify_procs, max_mutants=cfg.run_cfg.max_mutants) if cfg.falsify_procs > 0 else None
    root = Path(root)
    t0 = time.time()
    try:
        arms = [ArmRun(s, domains, cfg, root, gate, pool) for s in setups]
        res = await asyncio.gather(*(a.run() for a in arms))
    finally:
        if pool is not None:
            pool.close()
    out = {"run": cfg.run, "wall_s": round(time.time() - t0, 1), "serial": cfg.serial, "arms": {a.arm.name: m for a, m in zip(arms, res, strict=True)}}
    _jwrite(root / "batch.json", out)
    return out


# ---------------------------------------------------------------------------------------------------------- metrics

FUNNEL = ("proposed", "unique", "prefilter_pass", "oracle_executed", "contract", "falsified", "repaired", "admitted")


def _usage(p: Path | None) -> list[dict[str, Any]]:
    if p is None or not Path(p).exists():
        return []
    out = []
    for line in Path(p).read_text().splitlines():
        if line.strip():
            with contextlib.suppress(ValueError):
                out.append(json.loads(line))
    return out


def cand_costs(cands: Sequence[Cand], usage: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    """(USD per candidate, USD per stage). A call is a candidate's when its tag names the candidate's id (``<stage>:<id>``,
    attempts ``<id>~k`` included); a proposer call (``propose:<arm>/<cell>-b<k>``) is split equally over that batch's
    candidates; calls of no candidate (the panel's calibration) are spread over the built candidates of the arm."""
    per: dict[str, float] = defaultdict(float)
    stage: dict[str, float] = defaultdict(float)
    by_batch: dict[str, list[str]] = defaultdict(list)
    ids = {c.id for c in cands}
    for c in cands:
        by_batch[f"{c.arm}/{c.batch_key}"].append(c.id)
    shared = 0.0
    for r in usage:
        usd = float(r.get("usd") or 0.0)
        tag = str(r.get("tag") or "")
        st, _, rest = tag.partition(":")
        stage[st or "?"] += usd
        cid = rest.split("~", 1)[0]
        if st == "propose" and rest in by_batch:
            for x in by_batch[rest]:
                per[x] += usd / len(by_batch[rest])
        elif cid in ids:
            per[cid] += usd
        else:
            shared += usd
    built = [c.id for c in cands if c.task is not None] or [c.id for c in cands]
    for x in built:
        per[x] += shared / max(1, len(built))
    return dict(per), {k: round(v, 5) for k, v in sorted(stage.items())}


def bootstrap_ratio(num: Sequence[float], den: Sequence[float], *, b: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """sum(num) / sum(den) with a percentile bootstrap 95% interval over units (candidates)."""
    n = len(num)
    if n == 0 or sum(den) <= 0:
        return 0.0, 0.0, 0.0
    point = sum(num) / sum(den)
    rng = random.Random(seed)
    vals = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(den[i] for i in idx)
        vals.append(sum(num[i] for i in idx) / d if d > 0 else 0.0)
    vals.sort()
    return point, vals[int(0.025 * b)], vals[min(b - 1, math.ceil(0.975 * b) - 1)]


def _group(xs: Sequence[Cand], key: Callable[[Cand], str]) -> dict[str, list[Cand]]:
    out: dict[str, list[Cand]] = defaultdict(list)
    for x in xs:
        out[key(x)].append(x)
    return out


def funnel(cands: Sequence[Cand]) -> dict[str, int]:
    f: Counter[str] = Counter()
    for c in cands:
        f["proposed"] += 1
        if c.status == "duplicate":
            continue
        f["unique"] += 1
        if c.checks.get("pre"):
            continue
        f["prefilter_pass"] += 1
        rec = c.record or {}
        if rec.get("stage") in ("oracle", "contract", "refined") or c.task is not None:
            f["oracle_executed"] += 1
        if c.task is not None:
            f["contract"] += 1
        if rec.get("stage") == "refined":
            f["falsified"] += 1
            if rec.get("repairs_accepted"):
                f["repaired"] += 1
        if c.status == "certified":
            f["admitted"] += 1
    return {k: f.get(k, 0) for k in FUNNEL}


def arm_metrics(cands: Sequence[Cand], usage: Sequence[Mapping[str, Any]], *, wall_s: float, spent: float | None = None) -> dict[str, Any]:
    per, by_stage = cand_costs(cands, usage)
    total = round(sum(by_stage.values()), 5) if spent is None else round(spent, 5)
    admitted = [c for c in cands if c.status == "certified"]
    k, n = len(admitted), len([c for c in cands if c.status != "duplicate"])
    num = [1.0 if c.status == "certified" else 0.0 for c in cands]
    den = [per.get(c.id, 0.0) for c in cands]
    apd = bootstrap_ratio(num, den)
    cpa = bootstrap_ratio(den, num) if k else (math.inf, math.inf, math.inf)
    st: Counter[str] = Counter(c.status for c in cands)
    reasons: Counter[str] = Counter()
    for c in cands:
        if c.status not in ("certified", "pending"):
            for x in c.reasons[:4]:
                reasons[x.split("=")[0].rsplit(":", 1)[0] if x.startswith("prefilter:") else x.split("=")[0]] += 1
    busy: dict[str, float] = defaultdict(float)
    for c in cands:
        for s, (a, b_) in c.times.items():
            if a and b_ and b_ >= a:
                busy[s] += b_ - a
    by_batch = Counter(c.batch_key for c in cands)
    prop_share = sum(((c.times.get("propose") or [0, 0])[1] - (c.times.get("propose") or [0, 0])[0]) / by_batch[c.batch_key] for c in cands)
    serial_eq = prop_share + busy.get("build", 0.0) + busy.get("refine", 0.0)
    truth = Counter((c.truth or "unknown") for c in cands if c.task is not None)
    truth_adm = Counter((c.truth or "unknown") for c in admitted)
    sizes = [len(((c.task or {}).get("oracle") or {}).get("steps") or []) for c in admitted]
    writes = [int((c.record or {}).get("oracle_writes") or 0) for c in admitted]
    lo, hi = wilson(k, n) if n else (0.0, 0.0)
    first = sum(1 for c in cands if (c.record or {}).get("stage") == "refined" and (c.record or {}).get("first_round_confirmed"))
    refined = sum(1 for c in cands if (c.record or {}).get("stage") == "refined")
    return {"candidates": len(cands), "funnel": funnel(cands), "status": dict(st), "reasons": dict(reasons.most_common()),
            "spend_usd": total, "cost_by_stage": by_stage, "admitted": k, "provisional": st.get("provisional", 0),
            "admitted_per_usd": {"point": round(apd[0], 3), "lo": round(apd[1], 3), "hi": round(apd[2], 3)},
            "usd_per_admitted": {"point": round(cpa[0], 3), "lo": round(cpa[1], 3), "hi": round(cpa[2], 3)} if k else None,
            "yield": {"k": k, "n": n, "p": round(k / n, 3) if n else 0.0, "lo": round(lo, 3), "hi": round(hi, 3)},
            "first_round_confirmed": {"k": first, "n": refined},
            "speed": {"wall_s": round(wall_s, 1), "admitted_per_hour": round(3600 * k / wall_s, 2) if wall_s > 0 else 0.0,
                      "wall_s_per_admitted": round(wall_s / k, 1) if k else None, "wall_s_per_candidate": round(wall_s / max(1, len(cands)), 1),
                      "busy_s": {s: round(v, 1) for s, v in sorted(busy.items())}, "serial_equivalent_s": round(serial_eq, 1),
                      "speedup_vs_serial_equivalent": round(serial_eq / wall_s, 2) if wall_s > 0 else None},
            "oracle_truth_built": dict(truth), "oracle_truth_admitted": dict(truth_adm),
            "by_domain": {d: {"candidates": len(xs), "admitted": sum(1 for c in xs if c.status == "certified"),
                              "spend_usd": round(sum(per.get(c.id, 0.0) for c in xs), 4), "funnel": funnel(xs)}
                          for d, xs in sorted(_group(cands, lambda c: c.domain).items())},
            "admitted_size": {"oracle_steps": sizes, "oracle_writes": writes}, "admitted_cells": dict(Counter(c.cell_key for c in admitted)),
            "cells_proposed": dict(Counter(c.cell_key for c in cands))}


def load_cands(arm_dir: str | Path) -> list[Cand]:
    d = Path(arm_dir) / "cands"
    return [Cand.model_validate_json(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def report_md(batch: Mapping[str, Any], title: str = "") -> str:
    arms = batch["arms"]
    names = list(arms)
    L = [f"# {title or 'batch ' + str(batch.get('run'))}", "", f"Wall clock {batch.get('wall_s')} s; serial mode: {batch.get('serial')}.", "",
         "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

    def row(label: str, f: Callable[[Mapping[str, Any]], Any]) -> None:
        L.append(f"| {label} | " + " | ".join(str(f(arms[a])) for a in names) + " |")

    for s in FUNNEL:
        row(s, lambda m, s=s: m["funnel"].get(s, 0))
    row("provisional", lambda m: m.get("provisional", 0))
    row("spend (USD)", lambda m: f"{m['spend_usd']:.3f}")
    row("admitted per USD (95% bootstrap)", lambda m: f"{m['admitted_per_usd']['point']:.2f} ({m['admitted_per_usd']['lo']:.2f}–{m['admitted_per_usd']['hi']:.2f})")
    row("USD per admitted", lambda m: "—" if not m.get("usd_per_admitted") else
        f"{m['usd_per_admitted']['point']:.2f} ({m['usd_per_admitted']['lo']:.2f}–{m['usd_per_admitted']['hi']:.2f})")
    row("yield (Wilson)", lambda m: f"{m['yield']['k']}/{m['yield']['n']} ({m['yield']['lo']:.2f}–{m['yield']['hi']:.2f})")
    row("first-round confirmed", lambda m: f"{m['first_round_confirmed']['k']}/{m['first_round_confirmed']['n']}")
    row("wall s", lambda m: m["speed"]["wall_s"])
    row("admitted per hour", lambda m: m["speed"]["admitted_per_hour"])
    row("wall s per admitted", lambda m: m["speed"]["wall_s_per_admitted"])
    row("serial-equivalent s / wall s", lambda m: f"{m['speed']['serial_equivalent_s']} / {m['speed']['wall_s']} = {m['speed']['speedup_vs_serial_equivalent']}")
    row("oracle truth, built", lambda m: m.get("oracle_truth_built"))
    row("oracle truth, admitted", lambda m: m.get("oracle_truth_admitted"))
    doms = sorted({d for a in names for d in arms[a].get("by_domain", {})})
    def dom_cell(m: Mapping[str, Any], d: str) -> str:
        x = m.get("by_domain", {}).get(d)
        return f"{x['admitted']} / {x['candidates']} / {x['spend_usd']:.2f}" if x else "—"

    for d in doms:
        row(f"{d}: admitted / candidates / USD", lambda m, d=d: dom_cell(m, d))
    L += ["", "Cost by stage (USD):", ""]
    for a in names:
        L.append(f"- {a}: " + ", ".join(f"{k} {v:.3f}" for k, v in arms[a]["cost_by_stage"].items()))
    L += ["", "Reasons (failed candidates):", ""]
    for a in names:
        L.append(f"- {a}: " + ", ".join(f"{k} {v}" for k, v in list(arms[a]["reasons"].items())[:14]))
    return "\n".join(L) + "\n"


__all__ = ["FUNNEL", "Arm", "ArmRun", "ArmSetup", "BatchConfig", "BatchDomain", "Cand", "CellSpec", "Resilient", "arm_metrics", "batch_domains",
           "bootstrap_ratio", "cand_costs", "funnel", "load_cands", "report_md", "run_batch", "wrap_roles"]

