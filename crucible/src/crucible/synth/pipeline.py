"""One end-to-end synthesis run over a domain, with model-driven roles (part2-paper-core §7, M4).

    run_domain(env, roles):
      A  propose       the proposer writes ``n_tasks`` tasks in batches (one batch per start state / cell)
         oracle        the oracle programmer writes π*; it is EXECUTED; a dead program gets its errors back
                       (run-the-reference repair, ≤ ``oracle_retries``), then it is dropped
         contract      skeleton(run(π*)) is relaxed by the contract writer through typed edits (``roles.apply_edits``);
                       an edit that makes the oracle fail its own contract is removed
      B  calibrate     the panel's λ from executed-truth anchors (oracle traces: valid; no-op and minus-last-write
                       traces: invalid)
      C  refine        ``cegis.refine``: the deterministic falsifiers, the archive regressions, ALT and ATK, adjudication,
                       the repairer (≤ ``max_rounds``), certification
      D  abstract      the abstractor turns this run's counterexample clusters into candidate lemmas (not admitted:
                       admission is the birth gate of M3)

Outcomes: **certified** (admitted), **provisional** (every admission rule holds except the MUT bound, i.e. c = 0 with
n < 59: §6), **rejected** (with the certificate's reasons) and **dropped** (no executable oracle or contract). Every
role call carries a tag ``<stage>:<task>``, so the wire attributes spend per stage; the report holds the funnel, the
per-class counts, repair rounds, reasons and cost. Artifacts go under ``run_dir`` (durable): ``proposals.json``,
``tasks/<n>/{oracle,contract,cegis}.json``, ``calibration.json``, ``panel.jsonl``, ``human_queue.jsonl``,
``lemmas.json``, ``report.json`` and ``report.md``.

Nothing here knows a model: the roles are ``RoleClient``s (``factory/synth_wire.py`` binds them to Tinker).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import Field

from crucible.actors.llm import BudgetExceeded
from crucible.core.schema.base import CrucibleModel
from crucible.synth.archive import Archive
from crucible.synth.canon import Normalizers, diff
from crucible.synth.cegis import CegisReport, refine
from crucible.synth.certificate import AdmissionPolicy
from crucible.synth.contract import Contract, grade
from crucible.synth.falsify import ALT, ATK, DEFECTS, DETERMINISTIC, Counterexample, FalsifyContext, MUTHeldOut, population
from crucible.synth.falsify.agents import Adjudicated, AgentBudget, Anchor, Attempt, Panel, PanelMember, calibrate, llm_alt, llm_atk, trace_view
from crucible.synth.iface import SynthWorld, ToolInfo
from crucible.synth.program import Program, Run
from crucible.synth.roles import (
    ABSTRACTOR,
    CONTRACT_DOC,
    DEFECTS_DOC,
    PROGRAM_DOC,
    REPAIRER,
    ContractOut,
    RepairOut,
    RoleClient,
    RoleFailed,
    apply_edits,
    ask,
    compact,
    contract_doc,
    parse_contract,
    parse_program,
    program_doc,
    tool_catalogue,
    variant,
)
from crucible.synth.task import IntentCard, Task, WorldRef

STAGES = ("propose", "oracle", "contract", "calibrate", "repair", "ALT", "ATK", "adjudicate", "abstract")
CELLS = ("a decision on one entity that needs evidence from two or more records",
         "several related writes across entities, chosen by a stated rule",
         "a figure or list the user asks for, computed from the records, plus a write that depends on it",
         "a policy or rule application where a distractor entity looks like the target")


# ---------------------------------------------------------------------------------------------------------- inputs

@dataclass
class DomainEnv:
    """A domain the run synthesizes on. ``world(seed)`` names the World of a start state; ``slice`` returns what the
    proposer and the oracle programmer are shown of it (default: the canonical start state, ``max_rows`` per table)."""

    name: str
    description: str
    world: Callable[[int], WorldRef]
    seeds: Sequence[int] = (0,)
    normalizers: Normalizers | None = None
    slice: Callable[[SynthWorld], Awaitable[dict[str, Any]]] | None = None
    max_rows: int = 40
    cells: Sequence[str] | None = None  # the proposer's cells (default ``CELLS``); compose.py's follow-up links pass their own
    built_by: frozenset[str] = frozenset()  # families that built the World (a worldgen App): never ALT/ATK on its tasks


@dataclass
class RunRoles:
    proposer: RoleClient
    oracle_programmer: RoleClient
    contract_writer: RoleClient
    repairer: RoleClient
    alt: Sequence[tuple[str, RoleClient]] = ()
    atk: Sequence[tuple[str, RoleClient]] = ()
    panel: Sequence[tuple[str, RoleClient]] = ()
    abstractor: RoleClient | None = None

    def builder_families(self) -> set[str]:
        return {c.family for c in (self.proposer, self.oracle_programmer, self.contract_writer, self.repairer)}


@dataclass
class RunConfig:
    n_tasks: int = 20
    batch: int = 5
    max_rounds: int = 3
    oracle_retries: int = 3
    concurrency: int = 6
    alt_budget: AgentBudget = field(default_factory=AgentBudget)
    atk_budget: AgentBudget = field(default_factory=lambda: AgentBudget(max_calls=14, max_turns=18))
    lam: float = 0.7
    calibrate: bool = True
    calib_tasks: int = 4
    show_reasons: bool = True
    max_mutants: int = 240
    heldout_mutants: float = 0.0  # Part 1 V5: the share of each MUT neighbourhood the repairer never sees (MUT-HO at certification); 0 = off
    lemmas: tuple[str, ...] = ()  # the S component's design rules the proposer is shown (§9.5 synthesizer loop)
    writer_lemmas: tuple[str, ...] = ()  # ... and those the contract writer is shown (lemmas whose role is the writer)
    paid: bool = True  # ALT and ATK; off for a first-round-only measurement (the lemma test)
    repair: bool = True  # off: the repairer is never called (every repair is "declined"): first-round counterexamples only
    # yield work (synth.batch): the prompt variant of the builder roles and the ALT solver (``roles.variant``), whether the
    # oracle programmer sees the proposer's derive values (off: an independent derivation), and the ALT finish nudge
    variant: str = "v1"
    oracle_sees_derive: bool = True
    alt_finish_nudge: bool = False
    # solver harness v3 (compose-2): a progress check before finishing and a windowed history (None: the whole history)
    alt_progress: bool = False
    alt_window: int | None = None


def admission_policy() -> AdmissionPolicy:
    """§6, with ALT among the classes that must have no confirmed counterexample (a confirmed D6: a valid alternative
    solution the contract rejects)."""
    # compose-2: a confirmed ORACLE (a wrong reference decision) or LEAK (an L2 give-away the panel confirmed) finding
    # rejects too; outside this list a confirmed finding of those classes did not count (a class that did not run is skipped)
    return AdmissionPolicy(deterministic=[*DETERMINISTIC, "ALT", "ORACLE", "LEAK"])


# ---------------------------------------------------------------------------------------------------------- records

class TaskRecord(CrucibleModel):
    n: int
    id: str
    seed: int
    skill: str = ""
    stage: str = "proposed"  # proposed | oracle | contract | refined
    status: str = "pending"  # certified | provisional | rejected | dropped | budget | error
    reasons: list[str] = Field(default_factory=list)
    oracle_attempts: int = 0
    oracle_writes: int = 0
    edits_applied: int = 0
    edits_rejected: int = 0
    edits_removed: int = 0
    rounds: int = 0
    repairs_accepted: int = 0
    repairs_refused: int = 0
    repairs_declined: int = 0
    first_round_confirmed: dict[str, int] = Field(default_factory=dict)
    all_confirmed: dict[str, int] = Field(default_factory=dict)
    defects: dict[str, int] = Field(default_factory=dict)  # defect -> confirmed count over every round
    classes: dict[str, dict[str, Any]] = Field(default_factory=dict)  # the final certificate's per-class stats
    elapsed_s: float = 0.0
    note: str = ""


class RunReport(CrucibleModel):
    domain: str
    run_id: str
    started: float
    finished: float | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    families: dict[str, Any] = Field(default_factory=dict)
    tasks: list[TaskRecord] = Field(default_factory=list)
    funnel: dict[str, Any] = Field(default_factory=dict)
    cost: dict[str, Any] = Field(default_factory=dict)
    calibration: dict[str, Any] | None = None
    adjudication: dict[str, Any] = Field(default_factory=dict)  # per class: PPI estimate of the share judged valid (meta.ppi)
    lemmas: list[dict[str, Any]] = Field(default_factory=list)
    stopped: str | None = None


def _jwrite(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")
    tmp.replace(p)


# ---------------------------------------------------------------------------------------------------------- helpers

async def default_slice(world: SynthWorld, max_rows: int = 40) -> dict[str, Any]:
    c = await world.canon()
    if c is None:
        return {}
    out: dict[str, Any] = {}
    for app, colls in c.apps.items():
        for coll, rows in colls.items():
            items = list(rows.items())
            out[f"{app}.{coll}"] = {k: r for k, r in items[:max_rows]}
            if len(items) > max_rows:
                out[f"{app}.{coll}"]["..."] = f"{len(items) - max_rows} more rows"
    return out


def effects_view(r: Run, n: Normalizers | None) -> list[dict[str, Any]]:
    if r.start is None or r.end is None:
        return []
    return [{"change": str(ch.addr), "kind": ch.kind, "old": ch.old, "new": ch.new} for ch in diff(r.start, r.end, n)]


def run_problems(r: Run) -> list[str]:
    """What went wrong in an executed program, with the observations the steps returned (so a ref into a result the
    author guessed the shape of can be corrected): feedback for the oracle programmer and the repairer."""
    out = []
    if r.error:
        out.append(f"the program stopped: {r.error}")
    # a read that errors changes nothing and can be the evidence itself (the purchase order an invoice bills does not
    # exist: yield-1 dropped 3 correct po_mismatch oracles as dead); a later step that refers into it stops the program
    out += [f"step {c.step} ({c.tool}) returned an error: {c.obs.error[:300]}" for c in r.calls if c.obs.error and not c.read_only]
    if out:
        out += [f"observation of {c.step} ({c.tool}): {compact(c.obs.value, 700)}" for c in r.calls[:10] if c.obs.error is None]
    return out


def remap_patterns(old: Contract, new: Contract) -> dict[str, str]:
    """Required-pattern ids of ``old`` -> the matching ids of ``new`` (same App, collection, kind and field; the same key,
    or ``*`` once an ``id_canon`` re-keys inserts): edits written against one skeleton applied to the re-inferred one."""
    used: set[str] = set()
    out: dict[str, str] = {}
    for p in old.required:
        for q in new.required:
            if q.id in used or (q.app, q.coll, q.kind, q.field) != (p.app, p.coll, p.kind, p.field) or q.key not in (p.key, "*"):
                continue
            out[p.id] = q.id
            used.add(q.id)
            break
    return out


# What a counterexample of each class says, in words (the probe is the evidence; this is never the falsifier's code).
MEANING: dict[tuple[str, str], str] = {
    ("TAINT", "literal"): "The reference program uses this literal value, but it appears nowhere the agent can see (not in the instruction, not in "
                          "an earlier tool observation, not in the tool catalogue), and the contract depends on it. Read it through a tool first and "
                          "refer to the observation, or state it in the instruction (unless it is the decision the agent must make).",
    ("TAINT", "derive"): "A value the agent must work out appears in the instruction: the decision is given away. Remove it from the instruction.",
    ("TAINT", "r_const"): "The contract checks a constant that neither the instruction, the tools nor anything the reference program saw contains.",
    ("TAINT", "q_const"): "The answer check expects a constant that nothing the agent can see contains.",
    ("RED", "pattern"): "This required pattern already holds in the start state, so it checks nothing (vacuous): require a change, or drop it.",
    ("RED", "start_state"): "The contract passes on the untouched start state: doing nothing passes.",
    ("RED", "exec"): "An exec check already holds on the start state (vacuous).",
    ("TWIN", "twin"): "The instruction's literal was replaced by another entity's value in the instruction, program and contract: the contract "
                      "cannot tell the two tasks apart (it does not pin the entity the instruction names), or the twin program failed it.",
    ("MUT", "SwapEntity"): "The same actions on another entity also pass: the contract does not pin which entity must change.",
    ("MUT", "DropStep"): "A program without one of the reference's writes still passes: a required effect is not checked.",
    ("MUT", "Truncate"): "A program that stops early still passes: a required effect is not checked.",
    ("MUT", "ExtraWrite"): "A program that also changes an unrelated entity still passes: unrequested changes are not rejected.",
    ("MUT", "WrongAnswer"): "A wrong final answer still passes the answer check.",
    ("MUT", "Identity"): "The reference program itself fails the contract.",
    ("NULL", ""): "A degenerate agent (doing nothing, reading everything, writing everything, or a shortcut tool) passes.",
    ("REP", ""): "Running the reference program again gives a different end state or verdict: the contract reads run noise (a clock or a "
                 "generated id) without a normalizer.",
    ("PERT", ""): "Adding look-alike rows or shuffling rows flipped the verdict of the reference program or of a null agent.",
    ("ALT", ""): "An independent solver's attempt was judged a valid solution, but the contract rejected it: the contract is over-strict.",
    ("ATK", ""): "An attacker passed the contract with an attempt judged not to be a valid solution: a shortcut exists.",
}


def cx_meaning(cls: str, op: str) -> str:
    return MEANING.get((cls, op)) or MEANING.get((cls, "")) or ("A surviving program variant passes although it differs from the reference: a "
                                                               "value check may be too loose." if cls == "MUT" else "")


def cx_view(cx: Counterexample) -> dict[str, Any]:
    """A counterexample as the repairer reads it: the class, operator, defect and the probe, never falsifier code."""
    p = cx.probe
    d: dict[str, Any] = {"id": cx.id[3:13], "class": cx.cls, "operator": cx.op, "defect": f"{cx.defect} {DEFECTS.get(cx.defect or '', '')}".strip(),
                         "meaning": cx_meaning(cx.cls, cx.op),
                         "should": {"fail": "the contract should FAIL this run (it passed)", "pass": "the contract should PASS this run (it failed)",
                                    "property": "a property of the task (no run)"}[p.expect],
                         "reason_codes_seen": cx.reasons_seen}
    if p.program is not None:
        d["probe_program"] = program_doc(p.program)
    if p.instruction is not None:
        d["probe_instruction"] = p.instruction
    if p.perturb:
        d["start_state_edits"] = [x.model_dump(mode="json", exclude_defaults=True) for x in p.perturb]
    extra = {k: v for k, v in p.data.items() if k in ("value", "loc", "kind", "literal", "twin", "sibling", "entity", "pattern", "field", "op")}
    if extra:
        d["detail"] = extra
    return d


# ---------------------------------------------------------------------------------------------------------- the run

class Runner:
    def __init__(self, env: DomainEnv, roles: RunRoles, cfg: RunConfig, run_dir: Path, archive_root: Path, *, run_id: str,
                 cost: Callable[[], dict[str, Any]] | None = None, families: Mapping[str, Any] | None = None):
        self.env, self.roles, self.cfg, self.dir = env, roles, cfg, Path(run_dir)
        self.archive = Archive(archive_root)
        self.cost = cost or (dict)
        self.report = RunReport(domain=env.name, run_id=run_id, started=time.time(), families=dict(families or {}),
                                config={k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in cfg.__dict__.items()})
        self.stop: str | None = None
        self.tools: dict[int, list[ToolInfo]] = {}
        self.slices: dict[int, dict[str, Any]] = {}
        self.panel: Panel | None = None
        self.ready: list[tuple[TaskRecord, Task, Run]] = []
        self.panel_rows: list[dict[str, Any]] = []
        # hooks for synth.batch (defaults: the M4 behaviour): the deterministic classes per task, a wrapper of the paid
        # classes (concurrent agents), a screen between the executed oracle and the contract writer (a reason drops the
        # task as "filtered"), and a sink for every ALT/ATK agent turn
        self.cheap_factory: Callable[[], list[Any]] = lambda: population(cfg.heldout_mutants)
        self.wrap_paid: Callable[[list[Any]], list[Any]] | None = None
        self.screen: Callable[[TaskRecord, Task, Run], Awaitable[str | Task | None]] | None = None  # a Task: use it (intent re-derived)
        self.agent_log: Callable[[TaskRecord, str, dict[str, Any]], None] | None = None
        # compose-2 hooks: a repairer factory (composition repairs its parts: ``compose.PartRepairer``), the TWIN panel
        # (TWIN suspects decided inline like MUT's, strictly calibrated: ``falsify.twinjudge``), extra paid classes (the
        # LEAK probe), and the final version each refine reached (``final_tasks[rec.id]``)
        self.repair_factory: Callable[[Runner, TaskRecord, FalsifyContext], Callable[[Task, list[Counterexample]], Awaitable[Task | None]]] | None = None
        self.twin_panel: Panel | None = None
        self.extra_paid: Callable[[], list[Any]] | None = None
        self.final_tasks: dict[str, Task] = {}
        self.reports: dict[str, CegisReport] = {}

    # -- A: propose, oracle, contract ------------------------------------------------------------------------------
    async def survey(self) -> None:
        for s in self.env.seeds:
            w = await self.env.world(s).build()
            try:
                await w.reset(s)
                self.tools[s] = await w.tools()
                self.slices[s] = await (self.env.slice(w) if self.env.slice is not None else default_slice(w, self.env.max_rows))
            finally:
                c = getattr(w, "close", None)
                if c is not None:
                    res = c()
                    if asyncio.iscoroutine(res):
                        await res

    async def propose(self) -> list[TaskRecord]:
        recs: list[TaskRecord] = []
        proposals: list[dict[str, Any]] = []
        b = 0
        while len(recs) < self.cfg.n_tasks and not self.stop:
            seed = self.env.seeds[b % len(self.env.seeds)]
            cells = self.env.cells or CELLS
            cell = cells[b % len(cells)]
            want = min(self.cfg.batch, self.cfg.n_tasks - len(recs))
            try:
                out = await ask(self.roles.proposer, variant("proposer", self.cfg.variant), tag=f"propose:batch{b}", retries=1, domain=self.env.description,
                                tools=tool_catalogue(self.tools[seed]), world=compact(self.slices[seed], 24000), cell=cell,
                                existing=[p["instruction"] for p in proposals if "instruction" in p] or None, count=want,
                                lemmas=list(self.cfg.lemmas) or None)
            except BudgetExceeded as e:
                self.stop = f"budget: {e}"
                break
            except RoleFailed as e:
                proposals.append({"batch": b, "error": str(e)[:500]})
                b += 1
                if b > 2 * (self.cfg.n_tasks // max(1, self.cfg.batch) + 1):
                    break
                continue
            seen = {p.get("instruction") for p in proposals}
            for t in out.tasks[:want]:
                if t.instruction in seen:
                    continue
                n = len(recs) + 1
                rec = TaskRecord(n=n, id=f"{self.env.name}/{self.report.run_id}/t{n:02d}", seed=seed, skill=t.skill[:120])
                recs.append(rec)
                proposals.append({"n": n, "batch": b, "seed": seed, "cell": cell, **t.model_dump(mode="json")})
            b += 1
        _jwrite(self.dir / "proposals.json", proposals)
        self.proposals = {p["n"]: p for p in proposals if "n" in p}
        return recs

    def _task(self, rec: TaskRecord, p: Mapping[str, Any], program: Program | None = None, contract: Contract | None = None) -> Task:
        return Task(id=rec.id, domain=self.env.name, world=self.env.world(rec.seed), seed=rec.seed, instruction=p["instruction"],
                    contract=contract or Contract(normalizers=self.env.normalizers or Normalizers()), oracle=program or Program(),
                    intent=IntentCard(skill=p.get("skill", ""), given=list(p.get("given") or []), derive=list(p.get("derive") or []),
                                      notes=str(p.get("rationale", ""))[:500]))

    async def build(self, rec: TaskRecord) -> None:
        """Oracle (executed, with feedback) and contract (relaxed skeleton) for one proposal."""
        p = self.proposals[rec.n]
        tdir = self.dir / "tasks" / f"t{rec.n:02d}"
        task = self._task(rec, p)
        ctx = FalsifyContext(max_mutants=self.cfg.max_mutants)
        attempts: list[dict[str, Any]] = []
        names = {t.name for t in self.tools[rec.seed]}
        try:
            async def executed(o: Any) -> tuple[Program, Run]:
                rec.oracle_attempts += 1
                prog = parse_program({"steps": o.steps, "answer": o.answer}, names)
                r = await ctx.run(task, prog, taint=True)
                probs = run_problems(r)
                eff = effects_view(r, task.contract.normalizers)
                attempts.append({"program": program_doc(prog), "problems": probs, "effects": len(eff), "answer": r.answer})
                if probs:
                    raise ValueError("the program was executed and failed:\n" + "\n".join(probs[:14]))
                if not eff and r.answer is None:
                    raise ValueError("the program was executed: it changes nothing and reports no answer, so nothing can be checked")
                return prog, r

            try:
                prog, r = await ask(self.roles.oracle_programmer, variant("oracle_programmer", self.cfg.variant), tag=f"oracle:{rec.id}", check=executed,
                                    retries=self.cfg.oracle_retries, task=p["instruction"],
                                    intent={"skill": p.get("skill"), "given": p.get("given"), **({"derive": p.get("derive")} if self.cfg.oracle_sees_derive else {})},
                                    tools=tool_catalogue(self.tools[rec.seed]), world=compact(self.slices[rec.seed], 24000), program_format=PROGRAM_DOC)
            except RoleFailed as e:
                rec.status, rec.reasons, rec.note = "dropped", ["oracle:dead"], str(e)[:600]
                return
            finally:
                _jwrite(tdir / "oracle.json", attempts)
            rec.stage = "oracle"
            rec.oracle_writes = sum(1 for c in r.calls if not c.read_only)
            task = task.model_copy(update={"oracle": prog}, deep=True)
            if self.screen is not None:
                why = await self.screen(rec, task, r)
                if isinstance(why, Task):  # the screen re-derived the intent card from the executed oracle
                    task = why
                elif why:
                    rec.status, rec.reasons = "filtered", [why]
                    return
            from crucible.synth.infer import skeleton

            sk = skeleton(r, task.contract.normalizers)
            try:
                co: ContractOut = await ask(self.roles.contract_writer, variant("contract_writer", self.cfg.variant), tag=f"contract:{rec.id}", retries=1, task=p["instruction"],
                                            intent=task.intent.model_dump(mode="json"), tools=tool_catalogue(self.tools[rec.seed]),
                                            skeleton=contract_doc(sk), effects=effects_view(r, sk.normalizers), answer=r.answer,
                                            contract_format=CONTRACT_DOC, lemmas=list(self.cfg.writer_lemmas) or None)
                edits = list(co.edits)
            except RoleFailed as e:
                edits, co = [], ContractOut(notes=f"contract writer failed: {str(e)[:300]}")
            # normalizers change how the skeleton is inferred (an id_canon re-keys inserted rows), so they apply first and
            # the skeleton is re-inferred under them; the other edits are mapped onto it by address
            base, rest, n_bad = sk, list(edits), []
            norms = [e for e in edits if e.op == "normalize" and e.normalizers]
            if norms:
                n2 = sk.normalizers
                for e in norms:
                    try:
                        n2 = n2.merged(Normalizers.model_validate(e.normalizers))
                    except ValueError as ex:
                        n_bad.append(str(ex)[:200])
                base = skeleton(r, n2)
                ids = remap_patterns(sk, base)
                rest = [e.model_copy(update={"target": ids.get(e.target, e.target)}) if e.target else e for e in edits if e.op != "normalize"]
            res = apply_edits(base, rest, p["instruction"])
            applied = list(res.applied)
            c = res.contract
            removed: list[int] = []
            while not grade(r, c).passed and applied:  # a relaxation never makes the oracle fail: remove the newest edit
                removed.append(applied.pop())
                c = apply_edits(base, [rest[i] for i in applied], p["instruction"]).contract
            if not grade(r, c).passed:
                c = sk
            rec.edits_applied, rec.edits_rejected, rec.edits_removed = len(applied) + len(norms), len(res.rejected) + len(n_bad), len(removed)
            _jwrite(tdir / "contract.json", {"skeleton": contract_doc(sk), "renormalized_skeleton": contract_doc(base) if norms else None,
                                             "edits": [e.model_dump(mode="json", exclude_none=True) for e in edits],
                                             "applied_after_normalizers": applied, "rejected": res.rejected + [(-1, x) for x in n_bad],
                                             "removed_oracle_fails": removed, "contract": contract_doc(c), "notes": co.notes})
            task = task.model_copy(update={"contract": c}, deep=True)
            rec.stage = "contract"
            self.ready.append((rec, task, r))
        except BudgetExceeded as e:
            rec.status, rec.reasons = "budget", [str(e)[:300]]
            self.stop = f"budget: {e}"
        except Exception as e:  # noqa: BLE001 - one task's failure is recorded, the run goes on
            rec.status, rec.reasons = "error", [f"{type(e).__name__}: {str(e)[:500]}"]
        finally:
            await ctx.close()

    # -- B: calibrate ------------------------------------------------------------------------------------------------
    def make_panel(self) -> Panel | None:
        if len(self.roles.panel) < 2:
            return None
        return Panel(members=[PanelMember(n, c) for n, c in self.roles.panel], lam=self.cfg.lam, queue=self.dir / "human_queue.jsonl",
                     show_reasons=self.cfg.show_reasons)

    async def calibrate_panel(self, ready: Sequence[tuple[TaskRecord, Task, Run]] | None = None) -> None:
        """``ready``: the built tasks to take anchors from (default ``self.ready``; synth.batch passes its own, so that
        tasks other builds append meanwhile are never touched)."""
        self.panel = self.make_panel()
        src = list(self.ready if ready is None else ready)
        if self.panel is None or not self.cfg.calibrate or not src:
            return
        anchors: list[Anchor] = []
        for rec, task, r in src[: self.cfg.calib_tasks]:
            anchors.append(Anchor(task=task, attempt=Attempt(trace=trace_view(r.calls), answer=r.answer, reasons=[]), valid=True, kind="oracle"))
            anchors.append(Anchor(task=task, attempt=Attempt(trace=[], answer=None, reasons=[]), valid=False, kind="noop"))
            writes = [i for i, c in enumerate(r.calls) if not c.read_only]
            if writes:
                cut = [c for i, c in enumerate(r.calls) if i != writes[-1]]
                anchors.append(Anchor(task=task, attempt=Attempt(trace=trace_view(cut), answer=r.answer, reasons=[]), valid=False, kind="minus_last_write"))
        try:
            cal = await calibrate(self.panel, anchors, floor=self.cfg.lam)
        except BudgetExceeded as e:
            self.stop = f"budget: {e}"
            return
        self.report.calibration = cal.model_dump(mode="json")
        _jwrite(self.dir / "calibration.json", self.report.calibration)

    async def calibrate_twin(self, tasks: Sequence[Task], *, min_decided: int = 6, max_anchors: int = 18) -> None:
        """The TWIN panel (compose-2): the calibrated panel's members, λ from item 20's strict ``calibrate`` on anchors with
        executed truth: twin anchors (``falsify.twinjudge``) plus the panel's own (an oracle trace is valid, the no-op and
        the oracle minus its last write are not). Uncalibrated: λ above 1, TWIN suspects stay open. (First pilot: twin
        anchors alone were 0 on four links of two domains, so every TWIN suspect stayed open.)"""
        from crucible.synth.falsify.twinjudge import calibrate_twin_panel, twin_anchors

        base = self.panel or self.make_panel()
        if base is None:
            return
        tp = Panel(members=base.members, lam=base.lam, queue=self.dir / "human_queue.jsonl", show_reasons=base.show_reasons, tag="calibrate:twin")
        ctx = FalsifyContext(max_mutants=self.cfg.max_mutants)
        try:
            anchors = (await twin_anchors(list(tasks), ctx))[: max_anchors // 2]
            for task in list(tasks)[: self.cfg.calib_tasks]:
                if len(anchors) >= max_anchors:
                    break
                r = await ctx.oracle(task)
                anchors.append(Anchor(task=task, attempt=Attempt(trace=trace_view(r.calls), answer=r.answer, reasons=[]), valid=True, kind="oracle"))
                anchors.append(Anchor(task=task, attempt=Attempt(trace=[], answer=None, reasons=[]), valid=False, kind="noop"))
                writes = [i for i, c in enumerate(r.calls) if not c.read_only]
                if writes:
                    cut = [c for i, c in enumerate(r.calls) if i != writes[-1]]
                    anchors.append(Anchor(task=task, attempt=Attempt(trace=trace_view(cut), answer=r.answer, reasons=[]), valid=False, kind="minus_last_write"))
            cal = await calibrate_twin_panel(tp, anchors, min_decided=min_decided)
        except BudgetExceeded as e:
            self.stop = f"budget: {e}"
            return
        finally:
            await ctx.close()
        self.twin_panel = tp
        doc = cal.model_dump(mode="json") | {"anchors": len(anchors), "kinds": dict(Counter(a.kind for a in anchors))}
        _jwrite(self.dir / "twin_calibration.json", doc)

    # -- C: refine ---------------------------------------------------------------------------------------------------
    async def task_tools(self, task: Task, ctx: FalsifyContext, seed: int) -> list[ToolInfo]:
        """The tool catalogue of the task's own World (compose-1: a knob variant adds ``actors.ask`` and the ``page``
        parameter, which the base World surveyed at the start lacks); the surveyed one when the World cannot be built."""
        try:
            return list(await (await ctx.world(task)).tools())
        except Exception:  # noqa: BLE001 - fall back to the survey
            return list(self.tools.get(seed) or [])

    def repairer(self, rec: TaskRecord, ctx: FalsifyContext) -> Callable[[Task, list[Counterexample]], Awaitable[Task | None]]:
        if self.repair_factory is not None:  # composition: repair the parts, never the merged program (parts.PartRepairer)
            return self.repair_factory(self, rec, ctx)
        return self.default_repairer(rec, ctx)

    def default_repairer(self, rec: TaskRecord, ctx: FalsifyContext) -> Callable[[Task, list[Counterexample]], Awaitable[Task | None]]:
        """The repairer role on one task: it sees the counterexamples and the task's own World catalogue."""
        log_path = self.dir / "tasks" / f"t{rec.n:02d}" / "repairs.jsonl"
        calls = [0]  # a refused repair leaves the version and its counterexamples as they were: the next round's prompt
        # is the same, so each call gets its own tag (``repair:<id>#v<version>~<k>``) and, seeded by tag, a new sample (item 22)

        def log(row: dict[str, Any]) -> None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")

        async def repair(task: Task, cxs: list[Counterexample]) -> Task | None:
            if not self.cfg.repair:
                return None
            tools = await self.task_tools(task, ctx, rec.seed)
            names = {t.name for t in tools}

            async def valid(o: RepairOut) -> Task | None:
                row: dict[str, Any] = {"from_version": task.version, "counterexamples": [c.id for c in cxs], "give_up": o.give_up,
                                       "changed": [k for k in ("instruction", "oracle", "contract", "given", "derive") if getattr(o, k) is not None],
                                       "notes": o.notes[:500]}
                try:
                    if o.give_up:
                        return None
                    prog = parse_program(o.oracle, names) if o.oracle else task.oracle
                    con = parse_contract(o.contract) if o.contract else task.contract
                    intent = task.intent.model_copy(update={k: v for k, v in (("given", o.given), ("derive", o.derive)) if v is not None})
                    cand = task.next_version(instruction=(o.instruction or task.instruction).strip(), oracle=prog, contract=con, intent=intent)
                    r = await ctx.run(cand, prog, taint=True)
                    probs = run_problems(r)
                    if probs:
                        raise ValueError("the repaired reference program was executed and failed:\n" + "\n".join(probs[:14]))
                    v = grade(r, con)
                    if not v.passed:
                        raise ValueError(f"the repaired reference program fails the repaired contract: {v.reasons[:8]}")
                    row["result"] = "candidate"
                    return cand
                except ValueError as e:
                    row["result"] = f"invalid: {str(e)[:600]}"
                    raise
                finally:
                    log(row)

            try:
                calls[0] += 1
                out = await ask(self.roles.repairer, REPAIRER, tag=f"repair:{task.id}#v{task.version}~{calls[0]}", check=valid, retries=2, task=task.instruction,
                                intent=task.intent.model_dump(mode="json"), tools=tool_catalogue(tools), oracle=program_doc(task.oracle),
                                contract=contract_doc(task.contract), counterexamples=[cx_view(c) for c in cxs[:12]], defects=DEFECTS_DOC,
                                program_format=PROGRAM_DOC, contract_format=CONTRACT_DOC)
            except RoleFailed as e:
                log({"from_version": task.version, "result": f"role failed: {str(e)[:600]}"})
                return None
            return out if isinstance(out, Task) else None

        return repair

    async def refine_one(self, rec: TaskRecord, task: Task) -> None:
        t0 = time.time()
        ctx = FalsifyContext(max_mutants=self.cfg.max_mutants)
        tdir = self.dir / "tasks" / f"t{rec.n:02d}"
        builders = self.roles.builder_families() | set(self.env.built_by)

        def sink(name: str) -> Callable[[dict[str, Any]], None] | None:
            if self.agent_log is None:
                return None
            hook = self.agent_log
            return lambda row: hook(rec, name, row)

        alts = [llm_alt(n, c, role=variant("alt_solver", self.cfg.variant), budget=self.cfg.alt_budget, tag=f"ALT:{rec.id}#{n}",
                        finish_nudge=self.cfg.alt_finish_nudge, progress_check=self.cfg.alt_progress, window=self.cfg.alt_window, log=sink(n))
                for n, c in self.roles.alt]
        atks = [llm_atk(n, c, budget=self.cfg.atk_budget, tag=f"ATK:{rec.id}#{n}", log=sink(n)) for n, c in self.roles.atk]
        paid = [x for x in (ALT(alts, exclude_families=builders) if alts else None, ATK(atks, exclude_families=builders) if atks else None) if x]
        if not self.cfg.paid:
            paid = []
        if self.wrap_paid is not None:
            paid = self.wrap_paid(paid)
        panel = self.panel
        falsifiers = self.cheap_factory()
        heldout: Any = MUTHeldOut(holdout=self.cfg.heldout_mutants) if self.cfg.heldout_mutants > 0 else None
        if panel is not None:  # suspects (MUT survivors, ALT failures, ATK passes) are decided as they are found
            panel = Panel(members=panel.members, lam=panel.lam, queue=panel.queue, show_reasons=panel.show_reasons, tag=f"adjudicate:{rec.id}")
            falsifiers = [Adjudicated(f, panel) if f.name == "MUT" else f for f in falsifiers]
            paid = [Adjudicated(f, panel) for f in paid]
            heldout = Adjudicated(heldout, panel) if heldout is not None else None
        tp: Panel | None = None
        if self.twin_panel is not None:  # TWIN suspects: the twin task's own judgement, strictly calibrated λ
            from crucible.synth.falsify.twinjudge import TwinAdjudicated

            tp = Panel(members=self.twin_panel.members, lam=self.twin_panel.lam, queue=self.twin_panel.queue, show_reasons=self.twin_panel.show_reasons,
                       tag=f"adjudicate:twin:{rec.id}")
            falsifiers = [TwinAdjudicated(f, tp) if f.name == "TWIN" else f for f in falsifiers]
        if self.extra_paid is not None and self.cfg.paid:
            paid = [*paid, *self.extra_paid()]
        try:
            rep: CegisReport = await refine(task, repair=self.repairer(rec, ctx), ctx=ctx, archive=self.archive, falsifiers=falsifiers, paid=paid,
                                            adjudicate=None, policy=admission_policy(), max_rounds=self.cfg.max_rounds, heldout=heldout)
            self.record(rec, rep)
            if rep.task is not None:
                self.final_tasks[rec.id] = rep.task
            self.reports[rec.id] = rep
            _jwrite(tdir / "cegis.json", rep.model_dump(mode="json"))
            for pl in (panel, tp):
                if pl is not None and pl.log:
                    self.panel_rows += pl.log
                    with (self.dir / "panel.jsonl").open("a") as fh:
                        for row in pl.log:
                            fh.write(json.dumps(row, default=str) + "\n")
        except BudgetExceeded as e:
            rec.status, rec.reasons = "budget", [str(e)[:300]]
            self.stop = f"budget: {e}"
        except Exception as e:  # noqa: BLE001
            rec.status, rec.reasons = "error", [f"{type(e).__name__}: {str(e)[:500]}"]
        finally:
            rec.elapsed_s = round(time.time() - t0, 1)
            await ctx.close()

    def record(self, rec: TaskRecord, rep: CegisReport) -> None:
        rec.stage = "refined"
        rec.rounds = len(rep.rounds)
        rec.repairs_accepted = sum(1 for r in rep.rounds if r.repair == "accepted")
        rec.repairs_refused = sum(1 for r in rep.rounds if r.repair == "refused")
        rec.repairs_declined = sum(1 for r in rep.rounds if r.repair == "declined")
        rec.first_round_confirmed = dict(rep.rounds[0].confirmed) if rep.rounds else {}
        allc: Counter[str] = Counter()
        for r in rep.rounds:
            allc.update(r.confirmed)
        rec.all_confirmed = dict(allc)
        defects: Counter[str] = Counter()
        for cx in self.archive.records(self.env.name, rec.id):
            defects[cx.defect or "?"] += 1
        rec.defects = dict(defects)
        cert = rep.certificate
        if cert is not None:
            rec.classes = {k: {"n": s.n, "c": s.c, "U": round(s.upper, 4), "suspects": s.suspects, "passes": s.passes} if s.applicable
                           else {"applicable": False, "reason": s.reason} for k, s in cert.classes.items()}
        if rep.status == "admitted":
            rec.status, rec.reasons = "certified", []
        elif rep.reasons and all(x.startswith("MUT:upper=") for x in rep.reasons):
            rec.status, rec.reasons = "provisional", list(rep.reasons)
        else:
            rec.status, rec.reasons = "rejected", list(rep.reasons)

    # -- D: abstract -------------------------------------------------------------------------------------------------
    async def abstract(self, recs: Sequence[TaskRecord]) -> None:
        if self.roles.abstractor is None:
            return
        ids = {r.id for r in recs}
        cxs = [c for c in self.archive.records(self.env.name) if c.task_id in ids]
        if not cxs:
            return
        clusters: dict[tuple[str, str, str], list[Counterexample]] = {}
        for c in cxs:
            head = sorted({x.split(":", 1)[0] for x in c.reasons_seen})
            clusters.setdefault((c.cls, c.defect or "?", c.op + ("/" + ",".join(head) if head else "")), []).append(c)
        view = [{"cluster": f"k{i}", "class": k[0], "defect": f"{k[1]} {DEFECTS.get(k[1], '')}", "operator": k[2], "size": len(v),
                 "tasks": len({c.task_id for c in v})} for i, (k, v) in enumerate(sorted(clusters.items(), key=lambda kv: -len(kv[1])))][:30]
        try:
            out = await ask(self.roles.abstractor, ABSTRACTOR, tag="abstract:run", retries=1, clusters=view, defects=DEFECTS_DOC)
        except (RoleFailed, BudgetExceeded) as e:
            _jwrite(self.dir / "lemmas.json", {"clusters": view, "error": str(e)[:300]})
            return
        self.report.lemmas = [x.model_dump(mode="json") for x in out.lemmas]
        _jwrite(self.dir / "lemmas.json", {"status": "candidate (not admitted: the M3 birth gate decides)", "clusters": view, "lemmas": self.report.lemmas})

    # -- the whole run -----------------------------------------------------------------------------------------------
    async def run(self) -> RunReport:
        await self.survey()
        recs = await self.propose()
        self.report.tasks = recs
        sem = asyncio.Semaphore(self.cfg.concurrency)

        async def guarded(coro_fn: Callable[[], Awaitable[None]], rec: TaskRecord) -> None:
            async with sem:
                if self.stop:
                    if rec.status == "pending":
                        rec.status, rec.reasons = "budget", [self.stop[:300]]
                    return
                await coro_fn()
                self.save()

        await asyncio.gather(*(guarded(lambda r=r: self.build(r), r) for r in recs))
        self.ready.sort(key=lambda x: x[0].n)
        if not self.stop:
            await self.calibrate_panel()
        await asyncio.gather(*(guarded(lambda r=r, t=t: self.refine_one(r, t), r) for r, t, _ in self.ready))
        if not self.stop:
            await self.abstract(recs)
        for r in recs:
            if r.status == "pending":
                r.status, r.reasons = ("budget", [self.stop[:300]]) if self.stop else ("error", ["not finished"])
        if self.panel_rows:
            from crucible.synth.meta.ppi import adjudicated_rates

            self.report.adjudication = adjudicated_rates(self.panel_rows, (self.report.calibration or {}).get("votes") or [])
        self.report.stopped = self.stop
        self.report.finished = time.time()
        self.save()
        return self.report

    def save(self) -> None:
        self.report.funnel = funnel(self.report.tasks)
        self.report.cost = self.cost()
        _jwrite(self.dir / "report.json", self.report.model_dump(mode="json"))
        (self.dir / "report.md").write_text(report_md(self.report))


# ---------------------------------------------------------------------------------------------------------- reporting

def funnel(recs: Sequence[TaskRecord]) -> dict[str, Any]:
    st = Counter(r.status for r in recs)
    reached = Counter()
    order = ["proposed", "oracle", "contract", "refined"]
    for r in recs:
        for s in order[: order.index(r.stage) + 1]:
            reached[s] += 1
    first: Counter[str] = Counter()
    for r in recs:
        for k, v in r.first_round_confirmed.items():
            if v:
                first[k] += 1
    anyc: Counter[str] = Counter()
    for r in recs:
        for k, v in r.all_confirmed.items():
            if v:
                anyc[k] += 1
    reasons: Counter[str] = Counter()
    for r in recs:
        if r.status in ("rejected", "dropped", "provisional", "error", "budget"):
            for x in r.reasons[:6]:
                reasons[x.split("=")[0] if "=" in x else x[:80]] += 1
    defects: Counter[str] = Counter()
    for r in recs:
        defects.update(r.defects)
    alt_pass = sum(1 for r in recs if (r.classes.get("ALT") or {}).get("passes"))
    return {"proposed": len(recs), "oracle_executed": reached["oracle"], "contract": reached["contract"], "refined": reached["refined"],
            "tasks_with_confirmed_round1": dict(first), "tasks_with_confirmed_any_round": dict(anyc),
            "repaired_tasks": sum(1 for r in recs if r.repairs_accepted), "repair_rounds": sum(r.repairs_accepted for r in recs),
            "repairs_refused": sum(r.repairs_refused for r in recs), "repairs_declined": sum(r.repairs_declined for r in recs),
            "alt_pass_tasks": alt_pass, "status": dict(st), "reasons": dict(reasons.most_common()), "confirmed_defects": dict(defects.most_common()),
            "oracle_attempts": sum(r.oracle_attempts for r in recs)}


def report_md(rep: RunReport) -> str:
    f = rep.funnel
    lines = [f"# synth run {rep.domain} / {rep.run_id}", "", "| stage | tasks |", "|---|---|"]
    for k in ("proposed", "oracle_executed", "contract", "refined"):
        lines.append(f"| {k} | {f.get(k, 0)} |")
    for k, v in sorted((f.get("status") or {}).items()):
        lines.append(f"| final: {k} | {v} |")
    lines += ["", "Tasks with a confirmed counterexample, by falsifier class (round 1 / any round):", ""]
    allk = sorted(set(f.get("tasks_with_confirmed_round1", {})) | set(f.get("tasks_with_confirmed_any_round", {})))
    for k in allk:
        lines.append(f"- {k}: {f['tasks_with_confirmed_round1'].get(k, 0)} / {f['tasks_with_confirmed_any_round'].get(k, 0)}")
    rep_line = (f"Repairs accepted: {f.get('repair_rounds', 0)} on {f.get('repaired_tasks', 0)} tasks; refused {f.get('repairs_refused', 0)}; "
                f"declined {f.get('repairs_declined', 0)}. ALT passed on {f.get('alt_pass_tasks', 0)} tasks.")
    lines += ["", rep_line, "", "Reasons:", ""]
    lines += [f"- {k}: {v}" for k, v in (f.get("reasons") or {}).items()]
    lines += ["", "Confirmed defects (archived counterexamples):", ""] + [f"- {k}: {v}" for k, v in (f.get("confirmed_defects") or {}).items()]
    if rep.adjudication:
        lines += ["", ("Adjudicated probes judged valid (PPI mean with executed-truth anchors as labels; the anchors are oracle and degenerate "
                       "traces, not suspects):"), "", "| class | judged | valid / invalid / abstain | naive | PPI | 95% interval |", "|---|---|---|---|---|---|"]
        for k, a in rep.adjudication.items():
            lines.append(f"| {k} | {a['n_unlabeled']} | {a['decided_valid']} / {a['decided_invalid']} / {a['abstained']} | {a['naive']:.3f} | "
                         f"{a['estimate']:.3f} | {a['lo']:.3f}–{a['hi']:.3f} |")
    c = rep.cost or {}
    if c:
        lines += ["", "Cost (USD):", ""] + [f"- {k}: {v}" for k, v in (c.get("by_stage") or {}).items()] + [f"- total: {c.get('total_usd')}"]
    lines += ["", "| task | seed | status | rounds | repairs | confirmed (any round) | reasons |", "|---|---|---|---|---|---|---|"]
    for r in rep.tasks:
        lines.append(f"| t{r.n:02d} | {r.seed} | {r.status} | {r.rounds} | {r.repairs_accepted} | {r.all_confirmed} | {'; '.join(r.reasons[:3])[:160]} |")
    return "\n".join(lines) + "\n"
