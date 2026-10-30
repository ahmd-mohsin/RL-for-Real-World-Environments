"""The frontier loop K (part2-paper-core §9.5, §10): student failures become targeted tasks, under a cap and R4.

**Executed failure attribution** (``attribute``), by snapshot replay. The student's attempt is replayed call by call on
a fresh World, with a snapshot S_k after its k-th call. From each S_k the oracle π* is run to the end (``reset=False``)
and the joined episode (s₀, the student's first k calls, π*'s calls) is graded by the trusted grader:
- ``recoverable(k)``: π* can still finish the task from S_k. ``recoverable(0)`` must hold (else the replay does not
  reproduce the World: ``unreplayable``);
- the **first wrong step** is the call after which the task is no longer recoverable. Its changes classify it:
  ``forbidden`` when it changed an entity π* never touches (a side effect, a distractor acted on), else
  ``irreversible`` (a wrong action on the target);
- when every prefix is recoverable the failure is in what the student did *not* do: ``wrong_write`` (it wrote a
  required field with another value: the first such call), ``omission`` (a required change never made: the step is the
  end), ``answer`` (only the answer check fails), ``no_action`` (no write at all);
- ``protocol`` (the reply broke the agent protocol) and ``budget`` (the call budget ran out) are **routed**, never
  targeted: they are the harness's and the limits' problems, not a skill (the debug lane's ``failures.ROUTE_OWNER``).
Each attribution names the **mode** (the skill the step shows missing: decision, frame_discipline, derivation,
completion, reporting, acting, evidence_gathering when the student never read a collection π* read before the wrong
step), its collections, the task's cell and the grader's reason codes. The debug lane's Φ-checkpoint attribution
(``factory.traceaudit.first_wrong_step``) is the repository analogue; ``factory.synth_frontier_wire`` maps its findings
onto this record, so both lanes feed one frontier.

**Targeting** (``requests_from``). Attributions are clustered by (cell, mode), at least ``min_traces`` rollouts per
cluster; each cluster becomes a ``TargetRequest``: the cell, the knob changes of ``MODE_KNOBS`` (§10's typed knobs:
dependency depth, horizon, distractors, observation hiding, actor-held information, derive-vs-given) in the direction
the source tasks' band decisions give (too hard: ease along the failing axis; in band: the same level, more instances;
too easy: harden), the source tasks and the evidence ids. A knob change is kept later only if the behaviour signature
moves and p̂ moves in the declared direction (principle 15): ``knob_kept``.

**The loop's rules** (``FrontierPolicy``, ``admit``, ``compare``, ``frontier_round``):
- targeted variants are at most ``cap_frac`` = 20% of the pool; originals are never retired (``admit`` only adds);
- **R4**: after admission KL(q′ ‖ q) over the pool's cell histogram ≤ κ and H(q′) ≥ H_min (``meta.update``'s ``kl`` and
  ``entropy``, unchanged); a candidate that breaks either is refused with the rule;
- every targeted batch is compared with an **equal-budget untargeted batch** from the same proposer
  (``compare``: spend within ``tol``, perfect tasks per dollar for each arm with a bootstrap interval, the in-band rate,
  and the share of targeted tasks on which the student fails in the targeted mode).
Student reward never enters selection: a task enters on ``perfect`` (P1–P4), and the band decision only sets the
direction of a knob.
"""

from __future__ import annotations

import inspect
import random
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import content_hash, diff
from crucible.synth.contract import PEq, _globs
from crucible.synth.falsify.base import FalsifyContext, delta_of
from crucible.synth.learn import DifficultyRecord, Rollout
from crucible.synth.meta.update import entropy, kl
from crucible.synth.novelty import observed_collections
from crucible.synth.program import CallRecord, Program, Run, _loose_eq, run
from crucible.synth.task import Task

Kind = Literal["passed", "irreversible", "forbidden", "wrong_write", "omission", "answer", "no_action", "protocol", "budget", "unreplayable"]
ROUTED = {"protocol": "harness", "budget": "limits", "unreplayable": "world"}
MODE_OF = {"irreversible": "decision", "forbidden": "frame_discipline", "wrong_write": "derivation", "omission": "completion",
           "answer": "reporting", "no_action": "acting"}
KNOBS = ("dependency_depth", "horizon", "distractors", "observation_hiding", "actor_info", "derive_vs_given")
# mode -> (knob changes that ease it, knob changes that harden it); derive_vs_given -1 = one derived literal given
MODE_KNOBS: dict[str, tuple[dict[str, int], dict[str, int]]] = {
    "decision": ({"derive_vs_given": -1}, {"distractors": 1}),
    "derivation": ({"derive_vs_given": -1, "dependency_depth": -1}, {"dependency_depth": 1}),
    "frame_discipline": ({"distractors": -1}, {"distractors": 1}),
    "completion": ({"horizon": -1}, {"horizon": 1}),
    "reporting": ({"derive_vs_given": -1}, {"dependency_depth": 1}),
    "acting": ({"horizon": -1}, {"horizon": 1}),
    "evidence_gathering": ({"observation_hiding": -1}, {"observation_hiding": 1}),
}


class Attribution(CrucibleModel):
    id: str = ""
    task_id: str
    version: int = 1
    domain: str = ""
    rollout: int = 0
    student: str = ""
    checkpoint: str = "base"
    kind: Kind
    mode: str | None = None
    routed: str | None = None  # the owner of a routed failure (harness | limits | world)
    step: int | None = None  # 0-based index of the student's first wrong call (len(calls) = after the end)
    call: dict[str, Any] | None = None
    collections: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    cell: str = ""
    evidence_skipped: list[str] = Field(default_factory=list)
    recoverable: list[bool] = Field(default_factory=list)
    detail: str = ""
    lane: str = "synth"  # synth | debug


def _calls_of(prog: Program) -> list[tuple[str, dict[str, Any]]]:
    return [(s.tool, {k: getattr(v, "v", None) for k, v in s.args.items()}) for s in prog.steps]


async def _joined(world: Any, task: Task, s0: Any, prefix: list[CallRecord], ro: dict[str, bool]) -> Run:
    r = await run(world, task.oracle, task.instruction, seed=task.seed, given=task.intent.given, reset=False, taint=False)
    return Run(instruction=task.instruction, seed=task.seed, calls=[*prefix, *r.calls], answer=r.answer, answer_text=r.answer_text, start=s0, end=r.end,
               read_only=ro, error=r.error)


async def attribute(task: Task, rollout: Rollout, ctx: FalsifyContext, *, cell: str = "", student: str = "", checkpoint: str = "base",
                    budget_calls: int | None = None, max_steps: int = 40) -> Attribution:
    """Snapshot-replay attribution of one failed rollout (module docstring)."""
    base = {"task_id": task.id, "version": task.version, "domain": task.domain, "rollout": rollout.i, "student": student, "checkpoint": checkpoint,
            "cell": cell, "reasons": list(rollout.reasons)}

    def done(a: Attribution) -> Attribution:
        a.id = "at-" + content_hash({"t": a.task_id, "v": a.version, "r": a.rollout, "s": a.student, "c": a.checkpoint, "k": a.kind}, 16)
        if a.mode is None and a.kind in MODE_OF:
            a.mode = MODE_OF[a.kind]
        a.routed = ROUTED.get(a.kind)
        return a

    if rollout.passed:
        return done(Attribution(kind="passed", **base))
    prog = Program.model_validate(rollout.program) if rollout.program else Program(steps=[])
    if rollout.stop == "protocol" and not prog.steps:
        return done(Attribution(kind="protocol", detail="the reply broke the agent protocol before any call", **base))
    calls = _calls_of(prog)[:max_steps]
    world = await ctx.world(task)
    oracle = await ctx.oracle(task)
    await world.reset(task.seed)
    tools = await world.tools()
    ro = {t.name: t.read_only for t in tools}
    s0 = await world.canon()
    snaps = [await world.snapshot()]
    states = [s0]
    recs: list[CallRecord] = []
    for i, (tool, args) in enumerate(calls):
        try:
            obs = await world.call(tool, dict(args))
        except Exception as e:  # noqa: BLE001 - the World refused the replayed call: stop the replay there
            from crucible.synth.iface import Obs

            obs = Obs(error=f"replay: {type(e).__name__}")
        recs.append(CallRecord(step=f"a{i}", tool=tool, args=dict(args), obs=obs, read_only=ro.get(tool, False)))
        snaps.append(await world.snapshot())
        states.append(await world.canon())
    own = Run(instruction=task.instruction, seed=task.seed, calls=recs, answer=rollout.answer, start=s0, end=states[-1], read_only=ro)
    own_v = ctx.grade(own, task.contract)
    rec_ok: list[bool] = []
    for k in range(len(snaps)):
        await world.restore(snaps[k])
        j = await _joined(world, task, s0, recs[:k], ro)
        rec_ok.append(ctx.grade(j, task.contract).passed)
        if not rec_ok[-1]:
            break
    base["recoverable"] = rec_ok
    if own_v.passed:
        return done(Attribution(kind="unreplayable", detail="the replayed attempt passes: the rollout's verdict did not reproduce", **base))
    if not rec_ok or not rec_ok[0]:
        return done(Attribution(kind="unreplayable", detail="π* does not pass from the replayed start state", **base))
    otouch = {f"{a}.{c}.{k}" for a, c, k in {ch.addr.entity for ch in delta_of(oracle, task.contract)}}
    oracle_read = set()
    for c in oracle.calls:
        if c.read_only and c.obs.value is not None:
            oracle_read |= observed_collections(c.obs.value, oracle.start)
    reasons = list(own_v.reasons)
    base["reasons"] = reasons

    def skipped(upto: int) -> list[str]:
        seen: set[str] = set()
        for c in recs[:upto]:
            if c.read_only and c.obs.value is not None:
                seen |= observed_collections(c.obs.value, s0)
        return sorted(oracle_read - seen)

    if not rec_ok[-1]:
        k = len(rec_ok) - 1  # S_k is the first unrecoverable state: the k-th call (index k - 1) made it so
        step = k - 1
        ch = diff(states[step], states[k], task.contract.normalizers)
        ents = sorted({f"{c.addr.app}.{c.addr.coll}.{c.addr.key}" for c in ch})
        # an inserted row has a fresh key: it is "touched" when π* inserts into the same collection
        oins = {f"{c.addr.app}.{c.addr.coll}" for c in delta_of(oracle, task.contract) if c.kind == "insert"}
        ins = {f"{c.addr.app}.{c.addr.coll}.{c.addr.key}" for c in ch if c.kind == "insert"}
        outside = [e for e in ents if e not in otouch and not (e in ins and ".".join(e.split(".")[:2]) in oins)]
        kind: Kind = "forbidden" if outside else "irreversible"
        sk = skipped(step)
        mode = "evidence_gathering" if kind == "irreversible" and sk else None
        return done(Attribution(kind=kind, mode=mode, step=step, call={"tool": recs[step].tool, "args": recs[step].args}, evidence_skipped=sk,
                                collections=sorted({".".join(e.split(".")[:2]) for e in ents}),
                                detail=f"after call {step} π* can no longer finish the task" + (f"; it changed {outside} which π* never touches" if outside else ""),
                                **base))
    codes = {r.split(":", 1)[0] for r in reasons}
    missing = [r.split(":", 1)[1] for r in reasons if r.startswith("MISSING_REQUIRED:")]
    writes = [i for i, c in enumerate(recs) if not c.read_only]
    if budget_calls is not None and len(calls) >= budget_calls and rollout.answer is None:
        return done(Attribution(kind="budget", step=len(recs), patterns=missing, detail=f"the call budget ({budget_calls}) ran out", **base))
    if not writes and (missing or "FORBIDDEN_EFFECT" in codes or "MISSING_REQUIRED" in codes):
        return done(Attribution(kind="no_action", step=len(recs), patterns=missing, evidence_skipped=skipped(len(recs)), **base))
    pats = {p.id: p for p in task.contract.required}
    for i in writes:
        ch = diff(states[i], states[i + 1], task.contract.normalizers)
        for pid in missing:
            p = pats.get(pid)
            if p is None:
                continue
            for c in ch:
                same_kind = (c.kind == "insert") == (p.kind == "insert")
                if not (same_kind and c.addr.app == p.app and c.addr.coll == p.coll and p.field in ("*", c.addr.field)):
                    continue
                if p.kind == "insert":
                    # an inserted row meant for this pattern agrees with some of its eq predicates and not all
                    row = states[i + 1].collection(c.addr.app, c.addr.coll).get(c.addr.key) or {}
                    eqs = [(f, q.v) for f, q in p.where.items() if isinstance(q, PEq)]
                    hold = sum(1 for f, v in eqs if f in row and _loose_eq(row[f], v))
                    if not (0 < hold < len(eqs)):
                        continue
                elif not _globs(p, c.addr.app, c.addr.coll, c.addr.key):
                    continue
                sk = skipped(i)
                return done(Attribution(kind="wrong_write", mode="evidence_gathering" if sk else None, step=i,
                                        call={"tool": recs[i].tool, "args": recs[i].args}, patterns=[pid], evidence_skipped=sk,
                                        collections=[f"{p.app}.{p.coll}"], detail=f"call {i} wrote {c.addr} but not as {pid} requires", **base))
    if missing or "MISSING_REQUIRED" in codes:
        cols = sorted({f"{pats[m].app}.{pats[m].coll}" for m in missing if m in pats})
        return done(Attribution(kind="omission", step=len(recs), patterns=missing, collections=cols, evidence_skipped=skipped(len(recs)),
                                detail="every prefix is recoverable; the attempt stopped before a required change", **base))
    if codes <= {"ANSWER", "TRACE"} and "ANSWER" in codes:
        return done(Attribution(kind="answer", step=len(recs), detail="the changes pass; the answer does not", **base))
    if rollout.stop == "protocol":
        return done(Attribution(kind="protocol", step=len(recs), detail="the attempt ended on a protocol error", **base))
    return done(Attribution(kind="omission", step=len(recs), detail=f"recoverable failure, codes {sorted(codes)}", **base))


# ---------------------------------------------------------------------------------------------------------- targeting

class TargetRequest(CrucibleModel):
    id: str
    cell: str
    mode: str
    direction: Literal["ease", "same", "harden"]
    knobs: dict[str, int] = Field(default_factory=dict)
    source_tasks: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    student: str = ""
    traces: int = 0
    reason: str = ""


def _direction(decisions: Sequence[str | None]) -> Literal["ease", "same", "harden"]:
    c = Counter(d for d in decisions if d)
    if not c:
        return "same"
    top = c.most_common(1)[0][0]
    return "ease" if top == "too_hard" else "harden" if top == "too_easy" else "same"


def requests_from(attributions: Sequence[Attribution], records: Sequence[DifficultyRecord] = (), *, min_traces: int = 2) -> list[TargetRequest]:
    """(cell, mode) clusters of non-routed attributions → target requests (module docstring)."""
    dec = {(r.key.task_id, r.key.student): r.decision for r in records}
    groups: dict[tuple[str, str, str], list[Attribution]] = defaultdict(list)
    for a in attributions:
        if a.kind == "passed" or a.routed or not a.mode:
            continue
        groups[(a.student, a.cell, a.mode)].append(a)
    out = []
    for (student, cell, mode), xs in sorted(groups.items()):
        if len(xs) < min_traces:
            continue
        tasks = sorted({a.task_id for a in xs})
        d = _direction([dec.get((t, student)) for t in tasks])
        ease, hard = MODE_KNOBS.get(mode, ({}, {}))
        knobs = ease if d == "ease" else hard if d == "harden" else {}
        rid = "tr-" + content_hash({"s": student, "c": cell, "m": mode, "d": d, "t": tasks}, 12)
        out.append(TargetRequest(id=rid, cell=cell, mode=mode, direction=d, knobs=dict(knobs), source_tasks=tasks, evidence=sorted(a.id for a in xs),
                                 student=student, traces=len(xs),
                                 reason=f"{len(xs)} failed rollouts on {len(tasks)} task(s) whose first wrong step shows '{mode}'"))
    return out


def knob_kept(direction: str, sig_moved: bool, p_before: float | None, p_after: float | None) -> tuple[bool, str]:
    """Principle 15: a knob change is kept only if the behaviour signature moves and p̂ moves the declared way."""
    if not sig_moved:
        return False, "the behaviour signature did not move"
    if p_before is None or p_after is None:
        return False, "p̂ unmeasured"
    if direction == "ease" and p_after <= p_before:
        return False, f"declared easier, p̂ {p_before:.2f} -> {p_after:.2f}"
    if direction == "harden" and p_after >= p_before:
        return False, f"declared harder, p̂ {p_before:.2f} -> {p_after:.2f}"
    return True, "kept"


# ---------------------------------------------------------------------------------------------------------- admission under the cap and R4

class PoolEntry(CrucibleModel):
    task_id: str
    cell: str
    targeted: bool = False
    request: str | None = None


class FrontierPolicy(CrucibleModel):
    cap_frac: float = 0.2
    kappa: float = 0.1
    h_min: float | None = None  # None: 0.9 × the pool's entropy before admission (the floor never moves down within a round)
    min_traces: int = 2
    smoothing: float = 1e-3
    budget_tol: float = 0.35


def hist(entries: Sequence[PoolEntry]) -> dict[str, float]:
    return {k: float(v) for k, v in Counter(e.cell for e in entries).items()}


def admit(pool: Sequence[PoolEntry], candidates: Sequence[PoolEntry], policy: FrontierPolicy | None = None
          ) -> tuple[list[PoolEntry], list[tuple[PoolEntry, str]], dict[str, Any]]:
    """Add candidates in order (only perfect tasks should reach here). Never removes a pool entry. Refused: the
    targeted cap, R4's KL bound, R4's entropy floor."""
    pol = policy or FrontierPolicy()
    cur = list(pool)
    q0 = hist(cur)
    h_min = pol.h_min if pol.h_min is not None else 0.9 * entropy(q0)
    admitted, refused = [], []
    for c in candidates:
        nxt = [*cur, c]
        n_t = sum(1 for e in nxt if e.targeted)
        if c.targeted and n_t > pol.cap_frac * len(nxt) + 1e-9:
            refused.append((c, f"cap: {n_t} targeted of {len(nxt)} > {pol.cap_frac:.0%}"))
            continue
        q1 = hist(nxt)
        k = kl(q1, q0, pol.smoothing)
        if k > pol.kappa:
            refused.append((c, f"R4: KL(q'||q) = {k:.4f} > kappa {pol.kappa}"))
            continue
        h = entropy(q1)
        if h < h_min - 1e-12:
            refused.append((c, f"R4: H(q') = {h:.4f} < H_min {h_min:.4f}"))
            continue
        cur = nxt
        admitted.append(c)
    assert all(e in cur for e in pool), "an original was retired"
    q1 = hist(cur)
    return admitted, refused, {"kl": round(kl(q1, q0, pol.smoothing), 5), "entropy_before": round(entropy(q0), 5), "entropy_after": round(entropy(q1), 5),
                               "h_min": round(h_min, 5), "targeted": sum(1 for e in cur if e.targeted), "pool": len(cur),
                               "targeted_share": round(sum(1 for e in cur if e.targeted) / len(cur), 4) if cur else 0.0}


# ---------------------------------------------------------------------------------------------------------- equal-budget comparison

class TaskOutcome(CrucibleModel):
    task_id: str
    cell: str = ""
    perfect: bool = False
    decision: str | None = None
    cost_usd: float = 0.0
    request: str | None = None
    failed_modes: list[str] = Field(default_factory=list)  # the student's attributed modes on this task


class BatchResult(CrucibleModel):
    arm: Literal["targeted", "control"]
    budget_usd: float
    spent_usd: float
    tasks: list[TaskOutcome] = Field(default_factory=list)


def _boot(xs: Sequence[float], cost: float, b: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not xs or cost <= 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    vals = sorted(sum(rng.choice(xs) for _ in xs) / cost for _ in range(b))
    return vals[int(0.025 * b)], vals[int(0.975 * b) - 1]


def compare(targeted: BatchResult, control: BatchResult, requests: Sequence[TargetRequest] = (), *, tol: float = 0.35) -> dict[str, Any]:
    """The equal-budget comparison of a targeted batch and its untargeted control (module docstring)."""
    hi = max(targeted.spent_usd, control.spent_usd)
    equal = hi == 0 or abs(targeted.spent_usd - control.spent_usd) / hi <= tol
    modes = {r.id: r.mode for r in requests}

    def arm(b: BatchResult) -> dict[str, Any]:
        k = sum(1 for t in b.tasks if t.perfect)
        lo, up = _boot([1.0 if t.perfect else 0.0 for t in b.tasks], b.spent_usd)
        inb = sum(1 for t in b.tasks if t.decision == "in_band")
        return {"tasks": len(b.tasks), "perfect": k, "spent_usd": round(b.spent_usd, 5), "J": round(k / b.spent_usd, 4) if b.spent_usd else None,
                "J_ci": [round(lo, 4), round(up, 4)], "in_band": inb, "in_band_rate": round(inb / len(b.tasks), 4) if b.tasks else None}

    t_arm, c_arm = arm(targeted), arm(control)
    hits = [t for t in targeted.tasks if t.request and modes.get(t.request) in t.failed_modes]
    verdict = "unequal budget: no comparison" if not equal else (
        "targeted better" if t_arm["J_ci"][0] > (c_arm["J"] or 0) else "control better" if c_arm["J_ci"][0] > (t_arm["J"] or 0) else "no difference")
    return {"equal_budget": equal, "tol": tol, "targeted": t_arm, "control": c_arm,
            "targeted_mode_hit_rate": round(len(hits) / len(targeted.tasks), 4) if targeted.tasks else None, "verdict": verdict}


Propose = Callable[[Sequence[TargetRequest] | None, float], Awaitable[list[Task]] | list[Task]]
Evaluate = Callable[[Sequence[Task], str], Awaitable[BatchResult] | BatchResult]


async def _call(f: Callable[..., Any], *a: Any) -> Any:
    x = f(*a)
    return await x if inspect.isawaitable(x) else x


async def frontier_round(pool: Sequence[PoolEntry], attributions: Sequence[Attribution], records: Sequence[DifficultyRecord], *, propose: Propose,
                         evaluate: Evaluate, budget_usd: float, policy: FrontierPolicy | None = None,
                         cell_of: Callable[[Task], str] | None = None) -> dict[str, Any]:
    """One K round: requests from attributions; a targeted batch and an untargeted control batch, each at
    ``budget_usd`` from the same proposer; the comparison; admission of the perfect tasks of both arms (targeted ones
    under the cap and R4; control ones are ordinary proposals, under R4)."""
    pol = policy or FrontierPolicy()
    reqs = requests_from(attributions, records, min_traces=pol.min_traces)
    if not reqs:
        return {"requests": [], "skipped": "no attribution cluster reaches min_traces", "pool": len(pool)}
    t_tasks = await _call(propose, reqs, budget_usd)
    c_tasks = await _call(propose, None, budget_usd)
    t_res: BatchResult = await _call(evaluate, t_tasks, "targeted")
    c_res: BatchResult = await _call(evaluate, c_tasks, "control")
    cmp = compare(t_res, c_res, reqs, tol=pol.budget_tol)
    cof = cell_of or (lambda t: t.intent.skill)
    by_id = {t.id: t for t in [*t_tasks, *c_tasks]}
    cands = [PoolEntry(task_id=o.task_id, cell=o.cell or cof(by_id[o.task_id]) if o.task_id in by_id else o.cell, targeted=True, request=o.request)
             for o in t_res.tasks if o.perfect]
    cands += [PoolEntry(task_id=o.task_id, cell=o.cell, targeted=False) for o in c_res.tasks if o.perfect]
    admitted, refused, r4 = admit(pool, cands, pol) if cmp["equal_budget"] else ([], [(c, "unequal budget") for c in cands], {})
    return {"requests": [r.model_dump(mode="json") for r in reqs], "comparison": cmp, "admitted": [a.model_dump(mode="json") for a in admitted],
            "refused": [{"entry": c.model_dump(mode="json"), "why": w} for c, w in refused], "r4": r4}


def attribution_summary(attrs: Sequence[Attribution]) -> dict[str, Any]:
    failed = [a for a in attrs if a.kind != "passed"]
    return {"rollouts": len(attrs), "failed": len(failed), "by_kind": dict(Counter(a.kind for a in failed)),
            "by_mode": dict(Counter(a.mode for a in failed if a.mode and not a.routed)), "routed": dict(Counter(a.routed for a in failed if a.routed)),
            "evidence_skipped": sum(1 for a in failed if a.evidence_skipped)}


def export_attributions(attrs: Sequence[Attribution]) -> list[dict[str, Any]]:
    """The frontier attributions as the RSI loop reads them (DESIGN §7.7 item 19): first wrong step, cell, mode and
    the knob suggestion of each direction; no reward."""
    out = []
    for a in attrs:
        if a.kind == "passed":
            continue
        ease, hard = MODE_KNOBS.get(a.mode or "", ({}, {}))
        out.append({"id": a.id, "task_id": a.task_id, "version": a.version, "domain": a.domain, "student": a.student, "checkpoint": a.checkpoint,
                    "lane": a.lane, "kind": a.kind, "mode": a.mode, "routed": a.routed, "first_wrong_step": a.step, "call_tool": (a.call or {}).get("tool"),
                    "cell": a.cell, "collections": a.collections, "patterns": a.patterns, "evidence_skipped": a.evidence_skipped,
                    "knobs": {"ease": ease, "harden": hard} if not a.routed else {}})
    return out

