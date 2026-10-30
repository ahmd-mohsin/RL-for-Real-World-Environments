"""P2 learnability: per-student difficulty with predictor-first sequential band measurement (part2-paper-core §0, §10).

A task is learnable for student π when π's success rate is in the keep band, p̂_π(τ) ∈ [0.1, 0.6]. Measuring p costs
student rollouts, so the measurement is ordered and gated by a cheap predictor, then run as a two-look sequential test.

**Student probe** (``StudentProbe``). ``rollouts(task, n, start=i)`` returns n graded attempts (``Rollout``): the student
acts on a fresh World through the agent protocol and the **trusted grader** (``contract.grade``) decides each outcome.
An infrastructure failure (the client or World failed, no attempt was made) is **masked** (``passed=None``): outside n,
never a failure (forge.learnability's rule). A reply that breaks the agent protocol is the student's own failure.
``AgentStudentProbe`` is the generic implementation over any ``AgentRunner``; the Tinker binding for the real student
lives in ``factory.synth_student_wire`` (synth never imports factory). Every rollout keeps its trace and literal
program, so frontier attribution (``synth.frontier``) can replay it.

**Predictor** (``features``, ``Predictor``, ``fit``, ``loo``; the pattern of ``factory.difficulty``, ported because synth
may not import factory): a ridge logistic regression on static task features (oracle steps, writes, reads, ref depth,
required patterns, derive and given counts, observation-sourced literals, World size) plus **non-student proxies**: the
certificate's ALT pass share (Nemotron and gpt-oss solvers that ran during certification, already paid), proxy-model
rollouts, or a proxy model's estimate (``ProxyEstimate``). ``loo`` reports leave-one-task-out error against the
constant baseline and the band-classification accuracy, so the predictor shows whether it beats guessing.

**Band test** (``BandSpec``, ``band_test``, ``operating_characteristic``). A Bayesian two-look test: prior Beta(a, b)
(Jeffreys, or the predictor's prediction with ``prior_strength`` pseudo-draws when the predictor is trusted), looks at
k = 4 and k = 8. At each look the posterior masses of the three classes are computed (too_hard: p < lo, in_band,
too_easy: p > hi); the test stops when one class has posterior mass ≥ 1 − α. After the last look it reports the
argmax with ``confident = False``. The **stated error rate** is twofold: the posterior error 1 − max mass of each
decision, and the frequentist operating characteristic of the rule (exact binomial enumeration over a p grid, with an
indifference zone of ``indiff`` around each band edge), reported per class as the worst-case error outside the zone.

**Predictor-first** (``measure``). The predictor speaks first. When it is *trusted* (its leave-one-out band accuracy met
``trust_accuracy`` on at least ``trust_min`` tasks) and its prior alone puts mass ≥ 1 − ``skip_alpha`` on an
out-of-band class, the rollouts are skipped and the record says ``source = "predictor"``. Otherwise the student is
measured, and the predictor's call is recorded beside the measured decision (that is how it is calibrated).

**Difficulty records** (``DifficultyRecord``, ``DifficultyStore``): keyed by (task id, version, digest, student,
checkpoint, simulator, time mode), appended to a JSONL store under ``crucible/data/synth/difficulty/``, with every
rollout's outcome, reason codes, stop, cost, the looks taken, the decision with its posterior, the predictor's call and
the cost of the band decision. A record for the same key is resumed (its rollouts reused), never re-bought.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.credit.operators.branch_value import wilson
from crucible.synth.canon import content_hash
from crucible.synth.falsify.agents import AgentRunner, AgentSpec, run_agent, trace_program, trace_view
from crucible.synth.falsify.base import FalsifyContext
from crucible.synth.program import Fn, Program, Ref, Run, walk_args
from crucible.synth.roles import ALT_SOLVER, Role, _Out
from crucible.synth.task import Task

Decision = Literal["too_hard", "in_band", "too_easy"]
CLASSES: tuple[Decision, ...] = ("too_hard", "in_band", "too_easy")

# the student sees exactly what an ALT solver sees, through the same JSON tool protocol (``falsify.agents.llm_agent``);
# its temperature is the rollout temperature RL will sample at
STUDENT = ALT_SOLVER.model_copy(update={"name": "student", "stage": "P2", "differs_from": (), "independent": False, "min_families": 1,
                                       "temperature": 1.0})


class ProxyOut(_Out):
    p: float
    why: str = ""


DIFFICULTY_PROXY = Role(
    name="difficulty_proxy", stage="P2", output=ProxyOut, temperature=0.0, max_tokens=1024,
    sees=("instruction", "tools", "reference_steps", "student"),
    never=("contract", "gate_code", "verdicts"),
    independent=True,
    system=("You estimate how hard a task is for a given AI agent. The agent works through the tools listed, one JSON tool call per turn, "
            "with a budget of about 20 calls, and it does not see the reference solution. You see the task, the tools, the number and kind "
            "of steps of a reference solution, and which agent will attempt it. Reply with one JSON object {\"p\": <the probability, 0 to 1, "
            "that ONE attempt by that agent fully succeeds>, \"why\": <one sentence>}. Be calibrated: most multi-step tasks with policy "
            "rules are failed by mid-size models more often than they are passed."))


# ---------------------------------------------------------------------------------------------------------- rollouts

class Rollout(CrucibleModel):
    i: int
    passed: bool | None  # None: masked (no attempt was made: an infrastructure failure)
    reasons: list[str] = Field(default_factory=list)
    stop: str = "done"  # done | protocol | error | masked
    calls: int = 0
    answer: Any = None
    error: str | None = None
    cost_usd: float = 0.0
    trace: list[dict[str, Any]] = Field(default_factory=list)
    program: dict[str, Any] | None = None  # the attempt as a literal program (replayable)


@runtime_checkable
class StudentProbe(Protocol):
    """k graded rollouts of one student on one task (module docstring)."""

    student: str
    checkpoint: str
    simulator: str
    time_mode: str

    async def rollouts(self, task: Task, n: int, *, start: int = 0) -> list[Rollout]: ...


PROTOCOL_ERRORS = ("LLMOutputError", "ValidationError", "AgentProtocolError")


def rollout_of(i: int, run: Run, task: Task, ctx: FalsifyContext, *, cost_usd: float = 0.0) -> Rollout:
    """Grade one agent run with the trusted grader. ``agent: <Exc>`` errors are masked unless the exception is the
    student breaking the protocol (malformed replies after the retries), which is its own failure."""
    err = run.error
    if err is not None and err.startswith("agent:"):
        exc = err.split(":", 1)[1].strip()
        if not any(exc.startswith(p) for p in PROTOCOL_ERRORS):
            return Rollout(i=i, passed=None, stop="masked", error=err, calls=len(run.calls), cost_usd=cost_usd)
        v = ctx.grade(run, task.contract)
        return Rollout(i=i, passed=False, reasons=["PROTOCOL", *v.reasons], stop="protocol", error=err, calls=len(run.calls), answer=run.answer,
                       cost_usd=cost_usd, trace=trace_view(run.calls), program=trace_program(run.calls, run.answer).model_dump(mode="json"))
    v = ctx.grade(run, task.contract)
    return Rollout(i=i, passed=v.passed, reasons=v.reasons, stop="error" if err else "done", error=err, calls=len(run.calls), answer=run.answer,
                   cost_usd=cost_usd, trace=trace_view(run.calls), program=trace_program(run.calls, run.answer).model_dump(mode="json"))


class ContextPool:
    """One ``FalsifyContext`` per concurrent rollout: a context caches one World per ``WorldRef``, so two rollouts in
    flight must never share one (their calls would interleave in one start state)."""

    def __init__(self, make: Callable[[], FalsifyContext], size: int = 4):
        self.make, self.size = make, max(1, size)
        self._free: asyncio.Queue[FalsifyContext] | None = None
        self._all: list[FalsifyContext] = []

    async def _q(self) -> asyncio.Queue[FalsifyContext]:
        if self._free is None:
            self._free = asyncio.Queue()
            for _ in range(self.size):
                c = self.make()
                self._all.append(c)
                self._free.put_nowait(c)
        return self._free

    async def run(self, f: Callable[[FalsifyContext], Awaitable[Any]]) -> Any:
        q = await self._q()
        ctx = await q.get()
        try:
            return await f(ctx)
        finally:
            q.put_nowait(ctx)

    async def close(self) -> None:
        for c in self._all:
            await c.close()
        self._all, self._free = [], None


CostOf = Callable[[str], float]  # the measured USD booked under one rollout's tag


@dataclass
class AgentStudentProbe:
    """``StudentProbe`` over an agent runner factory. ``runner(tag)`` returns the runner for one rollout (the tag lets
    the client book that rollout's cost); ``cost_of(tag)`` reads it back (0 for scripted students)."""

    runner: Callable[[str], AgentRunner]
    contexts: ContextPool
    student: str
    checkpoint: str = "base"
    simulator: str = "json-agent/v1"
    time_mode: str = "frozen"
    family: str = "student"
    cost_of: CostOf | None = None
    concurrency: int = 4

    async def rollouts(self, task: Task, n: int, *, start: int = 0) -> list[Rollout]:
        sem = asyncio.Semaphore(max(1, self.concurrency))

        async def one(i: int) -> Rollout:
            tag = f"student:{task.id}@{task.version}:{i}"
            async with sem:
                async def go(ctx: FalsifyContext) -> Rollout:
                    r = await run_agent(AgentSpec(name=f"{self.student}#{i}", family=self.family, runner=self.runner(tag)), task, ctx)
                    return rollout_of(i, r, task, ctx, cost_usd=self.cost_of(tag) if self.cost_of else 0.0)

                return await self.contexts.run(go)

        return list(await asyncio.gather(*(one(start + j) for j in range(n))))


# ---------------------------------------------------------------------------------------------------------- features

def ref_depth(p: Program) -> tuple[int, int]:
    """(longest chain of steps linked by refs, number of ref edges) of a program."""
    deps: dict[str, set[str]] = {}
    for s in p.steps:
        d: set[str] = set()
        for name, a in s.args.items():
            for _, x in walk_args(a, f"{s.id}.{name}"):
                if isinstance(x, Ref):
                    d.add(x.path.split(".", 1)[0])
        deps[s.id] = d
    depth: dict[str, int] = {}
    for s in p.steps:
        depth[s.id] = 1 + max((depth.get(x, 0) for x in deps[s.id]), default=0)
    return max(depth.values(), default=0), sum(len(v) for v in deps.values())


def _n_fn(p: Program) -> int:
    n = 0
    for s in p.steps:
        for name, a in s.args.items():
            n += sum(1 for _, x in walk_args(a, f"{s.id}.{name}") if isinstance(x, Fn))
    return n


FEATURES = ("bias", "steps", "writes", "reads", "depth", "required", "derive", "obs_literals", "answer", "world_log", "ins_words_log",
            "alt_share", "proxy")


def features(task: Task, oracle: Run | None = None, *, alt: tuple[int, int] | None = None, proxy: float | None = None) -> dict[str, float]:
    """Static features of a task (its oracle run when given) and the non-student proxies (``alt`` = (passes, n) of
    the certificate's ALT solvers; ``proxy`` = a proxy model's pass share or estimate). A missing proxy is 0.5
    (uninformative), with an indicator so the fit can tell the two apart."""
    d, _edges = ref_depth(task.oracle)
    writes = reads = 0
    obs_lits = 0
    world = 0
    if oracle is not None:
        writes = sum(1 for c in oracle.calls if not c.read_only)
        reads = sum(1 for c in oracle.calls if c.read_only)
        obs_lits = sum(1 for e in oracle.taint if e.has("observation") and not e.has("instruction") and not e.has("given") and not e.has("structural"))
        world = sum(1 for _ in oracle.start.entities()) if oracle.start is not None else 0
    else:
        writes = len(task.oracle.steps)
    a_share = (alt[0] / alt[1]) if alt and alt[1] else 0.5
    return {"bias": 1.0, "steps": float(len(task.oracle.steps)), "writes": float(writes), "reads": float(reads), "depth": float(d),
            "required": float(len(task.contract.required)), "derive": float(len(task.intent.derive)), "obs_literals": float(obs_lits),
            "answer": 1.0 if task.oracle.answer is not None else 0.0, "world_log": math.log1p(world),
            "ins_words_log": math.log1p(len(task.instruction.split())), "alt_share": a_share, "has_alt": 1.0 if alt and alt[1] else 0.0,
            "proxy": 0.5 if proxy is None else float(proxy), "has_proxy": 0.0 if proxy is None else 1.0, "fns": float(_n_fn(task.oracle))}


async def proxy_estimate(client: Any, task: Task, tools: Sequence[Any], oracle: Run | None, *, student: str, tag: str = "") -> float | None:
    """A non-student proxy model's one-call estimate of the student's pass probability (``DIFFICULTY_PROXY``); None
    when the reply is unusable. ``client`` is a ``roles.RoleClient`` of a family other than the student's."""
    from crucible.synth.roles import tool_catalogue

    steps = [f"{'read' if c.read_only else 'write'} {c.tool}" for c in oracle.calls] if oracle is not None else [s.tool for s in task.oracle.steps]
    msgs = DIFFICULTY_PROXY.render(instruction=task.instruction, tools=tool_catalogue(list(tools)), reference_steps=steps, student=student)
    try:
        doc = await client.complete_json(msgs, schema=ProxyOut, max_tokens=DIFFICULTY_PROXY.max_tokens, temperature=0.0, tag=tag or "proxy")
        p = float(ProxyOut.model_validate(doc).p)
    except Exception as e:
        from crucible.actors.llm import BudgetExceeded

        if isinstance(e, BudgetExceeded):
            raise
        return None
    return min(1.0, max(0.0, p))


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


@dataclass
class Predictor:
    """p before any student rollout (``factory.difficulty.Predictor``'s form). ``trusted`` is set from a ``loo``
    report: only a trusted predictor may skip rollouts or shape the band test's prior."""

    coef: dict[str, float]
    mean: dict[str, float]
    scale: dict[str, float]
    n_tasks: int
    prior_strength: float = 2.0
    trusted: bool = False
    calibration: dict[str, Any] = field(default_factory=dict)

    def p(self, feats: Mapping[str, float]) -> float:
        z = sum(w * (1.0 if k == "bias" else (feats.get(k, 0.0) - self.mean.get(k, 0.0)) / self.scale.get(k, 1.0)) for k, w in self.coef.items())
        return _sig(z)

    def prior(self, feats: Mapping[str, float]) -> tuple[float, float]:
        p0 = min(0.98, max(0.02, self.p(feats)))
        return p0 * self.prior_strength, (1 - p0) * self.prior_strength

    def to_json(self) -> dict[str, Any]:
        return {"coef": self.coef, "mean": self.mean, "scale": self.scale, "n_tasks": self.n_tasks, "prior_strength": self.prior_strength,
                "trusted": self.trusted, "calibration": self.calibration}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Predictor:
        return cls(dict(d["coef"]), dict(d["mean"]), dict(d["scale"]), int(d["n_tasks"]), float(d.get("prior_strength", 2.0)), bool(d.get("trusted")),
                   dict(d.get("calibration") or {}))


class ConstantPredictor(Predictor):
    def __init__(self, p: float = 0.3):
        super().__init__({"bias": math.log(p / (1 - p))}, {}, {}, 0)


def fit(rows: Sequence[tuple[Mapping[str, float], int, int]], *, keys: Sequence[str] = FEATURES, l2: float = 2.0, iters: int = 3000,
        lr: float = 0.1) -> Predictor:
    """Ridge logistic regression on (features, passes, n) with a binomial likelihood (standardised features, the bias
    unpenalised), by gradient descent: stdlib only."""
    ks = [k for k in keys]
    cols = {k: [float(r[0].get(k, 0.0)) for r in rows] for k in ks}
    mean = {k: (statistics.fmean(v) if k != "bias" and v else 0.0) for k, v in cols.items()}
    scale = {k: ((statistics.pstdev(v) if len(v) > 1 else 0.0) or 1.0) if k != "bias" else 1.0 for k, v in cols.items()}
    X = [[1.0 if k == "bias" else (float(r[0].get(k, 0.0)) - mean[k]) / scale[k] for k in ks] for r in rows]
    w = [0.0] * len(ks)
    N = sum(r[2] for r in rows) or 1
    for _ in range(iters):
        g = [0.0] * len(ks)
        for x, (_, k, n) in zip(X, rows, strict=True):
            mu = _sig(sum(a * b for a, b in zip(w, x, strict=True)))
            for j, xj in enumerate(x):
                g[j] += (n * mu - k) * xj
        for j in range(len(ks)):
            pen = 0.0 if ks[j] == "bias" else l2 * w[j]
            w[j] -= lr * (g[j] + pen) / N
    return Predictor(dict(zip(ks, w, strict=True)), mean, scale, n_tasks=len(rows))


def band_of(p: float, lo: float = 0.1, hi: float = 0.6) -> Decision:
    return "too_hard" if p < lo else "too_easy" if p > hi else "in_band"


def loo(rows: Sequence[tuple[str, Mapping[str, float], int, int]], *, lo: float = 0.1, hi: float = 0.6, **kw: Any) -> dict[str, Any]:
    """Leave-one-task-out (every row of a task held out together): per-task predictions, MAE against the constant
    baseline (the training rows' pooled rate), Brier score, and the band-class accuracy of the prediction against the
    measured band (p̂ = passes / n)."""
    tasks = sorted({r[0] for r in rows})
    preds: list[dict[str, Any]] = []
    for t in tasks:
        train = [(f, k, n) for tid, f, k, n in rows if tid != t]
        test = [(f, k, n) for tid, f, k, n in rows if tid == t]
        if not train or not test:
            continue
        m = fit(train, **kw)
        pooled = sum(k for _, k, _ in train) / max(1, sum(n for _, _, n in train))
        for f, k, n in test:
            ph, pt = m.p(f), (k / n if n else 0.0)
            preds.append({"task": t, "p_true": round(pt, 4), "p_pred": round(ph, 4), "baseline": round(pooled, 4), "n": n,
                          "band_true": band_of(pt, lo, hi), "band_pred": band_of(ph, lo, hi), "band_base": band_of(pooled, lo, hi)})
    if not preds:
        return {"tasks": len(tasks), "rows": 0, "mae": None, "mae_baseline": None, "band_accuracy": None, "band_accuracy_baseline": None,
                "predictions": []}
    acc = sum(p["band_true"] == p["band_pred"] for p in preds)
    acc0 = sum(p["band_true"] == p["band_base"] for p in preds)
    return {"tasks": len(tasks), "rows": len(preds), "mae": round(statistics.fmean(abs(p["p_pred"] - p["p_true"]) for p in preds), 4),
            "mae_baseline": round(statistics.fmean(abs(p["baseline"] - p["p_true"]) for p in preds), 4),
            "brier": round(statistics.fmean((p["p_pred"] - p["p_true"]) ** 2 for p in preds), 4),
            "band_accuracy": round(acc / len(preds), 4), "band_accuracy_ci": [round(x, 4) for x in wilson(acc, len(preds))],
            "band_accuracy_baseline": round(acc0 / len(preds), 4), "predictions": preds}


def calibrated(rows: Sequence[tuple[str, Mapping[str, float], int, int]], *, trust_min: int = 12, trust_accuracy: float = 0.8, **kw: Any
               ) -> Predictor:
    """The predictor fit on every row, with its leave-one-out report attached; ``trusted`` only when at least
    ``trust_min`` tasks were held out and the Wilson lower bound of its band accuracy reaches ``trust_accuracy`` and it
    beats the constant baseline."""
    rep = loo(rows, **{k: v for k, v in kw.items() if k in ("lo", "hi", "keys", "l2")})
    m = fit([(f, k, n) for _, f, k, n in rows], **{k: v for k, v in kw.items() if k in ("keys", "l2")})
    acc, lo_ci = rep.get("band_accuracy"), (rep.get("band_accuracy_ci") or [0.0])[0]
    m.trusted = bool(rep["tasks"] >= trust_min and acc is not None and lo_ci >= trust_accuracy and acc > (rep.get("band_accuracy_baseline") or 0.0))
    m.calibration = {k: v for k, v in rep.items() if k != "predictions"} | {"trust_min": trust_min, "trust_accuracy": trust_accuracy}
    return m


# ---------------------------------------------------------------------------------------------------------- beta and the band test

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """I_x(a, b), the Beta(a, b) CDF at x."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


@dataclass(frozen=True)
class BandSpec:
    lo: float = 0.1
    hi: float = 0.6
    looks: tuple[int, ...] = (4, 8)
    alpha: float = 0.1  # stop when one class has posterior mass >= 1 - alpha
    prior: tuple[float, float] = (0.5, 0.5)  # Jeffreys, when no trusted predictor shapes it
    indiff: float = 0.05  # the operating characteristic's indifference zone around each band edge


def masses(a: float, b: float, spec: BandSpec) -> dict[str, float]:
    lo, hi = beta_cdf(spec.lo, a, b), beta_cdf(spec.hi, a, b)
    return {"too_hard": round(lo, 6), "in_band": round(max(0.0, hi - lo), 6), "too_easy": round(max(0.0, 1.0 - hi), 6)}


class BandResult(CrucibleModel):
    decision: Decision
    confident: bool
    posterior: dict[str, float]
    error: float  # 1 - the posterior mass of the decision
    passes: int
    n: int
    looks: list[int]
    prior: list[float]
    p_hat: float
    ci: list[float]


def decide(passes: int, n: int, spec: BandSpec, prior: tuple[float, float] | None = None, *, final: bool) -> BandResult | None:
    """The test's decision after ``n`` measured outcomes, or None (look again) when no class reaches 1 − α and this is
    not the final look."""
    a0, b0 = prior or spec.prior
    m = masses(a0 + passes, b0 + n - passes, spec)
    best = max(CLASSES, key=lambda c: m[c])
    conf = m[best] >= 1.0 - spec.alpha
    if not conf and not final:
        return None
    lo, hi = wilson(passes, n) if n else (0.0, 1.0)
    return BandResult(decision=best, confident=conf, posterior=m, error=round(1.0 - m[best], 6), passes=passes, n=n, looks=[], prior=[a0, b0],
                      p_hat=round(passes / n, 4) if n else 0.0, ci=[round(lo, 4), round(hi, 4)])


def _binom(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def operating_characteristic(spec: BandSpec, prior: tuple[float, float] | None = None, grid: Sequence[float] | None = None) -> dict[str, Any]:
    """The rule's exact decision probabilities for true p on a grid, the expected rollouts, and per true class the
    worst-case error outside the indifference zone (the stated frequentist error rate)."""
    ps = list(grid or [0.0, 0.02, 0.05, 0.08, 0.15, 0.2, 0.3, 0.4, 0.5, 0.55, 0.65, 0.7, 0.8, 0.9, 1.0])
    rows = []
    for p in ps:
        probs = {c: 0.0 for c in CLASSES}
        exp_n = 0.0

        def walk(look: int, passes: int, n: int, w: float, p: float = p, probs: dict[str, float] = probs) -> None:
            nonlocal exp_n
            step = spec.looks[look] - n
            for x in range(step + 1):
                ww = w * _binom(x, step, p)
                if ww == 0.0:
                    continue
                k2, n2 = passes + x, n + step
                final = look == len(spec.looks) - 1
                r = decide(k2, n2, spec, prior, final=final)
                if r is None:
                    walk(look + 1, k2, n2, ww)
                else:
                    probs[r.decision] += ww
                    exp_n += ww * n2

        walk(0, 0, 0, 1.0)
        rows.append({"p": p, **{c: round(v, 4) for c, v in probs.items()}, "expected_n": round(exp_n, 3)})
    worst: dict[str, float] = {}
    for c in CLASSES:
        errs = []
        for r in rows:
            p = r["p"]
            true = band_of(p, spec.lo, spec.hi)
            if true != c or abs(p - spec.lo) < spec.indiff or abs(p - spec.hi) < spec.indiff:
                continue
            errs.append(1.0 - r[c])
        worst[c] = round(max(errs), 4) if errs else None  # type: ignore[assignment]
    return {"spec": {"lo": spec.lo, "hi": spec.hi, "looks": list(spec.looks), "alpha": spec.alpha, "indiff": spec.indiff},
            "prior": list(prior or spec.prior), "grid": rows, "worst_error": worst}


# ---------------------------------------------------------------------------------------------------------- records

class DifficultyKey(CrucibleModel):
    task_id: str
    task_version: int
    task_digest: str
    student: str
    checkpoint: str = "base"
    simulator: str = "json-agent/v1"
    time_mode: str = "frozen"

    def id(self) -> str:
        return content_hash(self.model_dump(mode="json"), 24)


class DifficultyRecord(CrucibleModel):
    key: DifficultyKey
    domain: str = ""
    source: Literal["measured", "predictor", "none"] = "measured"
    rollouts: list[Rollout] = Field(default_factory=list)
    passes: int = 0
    n: int = 0
    masked: int = 0
    decision: Decision | None = None
    confident: bool = False
    posterior: dict[str, float] = Field(default_factory=dict)
    error: float | None = None
    looks: list[int] = Field(default_factory=list)
    p_hat: float | None = None
    ci: list[float] = Field(default_factory=list)
    prior: list[float] = Field(default_factory=list)
    predictor: dict[str, Any] = Field(default_factory=dict)  # p, prior masses, its own call, whether it was trusted
    features: dict[str, float] = Field(default_factory=dict)
    cost_usd: float = 0.0
    at: float = 0.0
    note: str = ""

    @property
    def learnable(self) -> bool:
        return self.decision == "in_band"


class DifficultyStore:
    """Append-only JSONL of records; the latest row per key wins (a resumed record supersedes its earlier row)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, rec: DifficultyRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(rec.model_dump_json() + "\n")

    def all(self) -> list[DifficultyRecord]:
        if not self.path.exists():
            return []
        latest: dict[str, DifficultyRecord] = {}
        for line in self.path.read_text().splitlines():
            if line.strip():
                r = DifficultyRecord.model_validate_json(line)
                latest[r.key.id()] = r
        return list(latest.values())

    def get(self, key: DifficultyKey) -> DifficultyRecord | None:
        k = key.id()
        return next((r for r in self.all() if r.key.id() == k), None)

    def for_task(self, task: Task, student: str | None = None) -> list[DifficultyRecord]:
        return [r for r in self.all() if r.key.task_id == task.id and r.key.task_version == task.version and (student is None or r.key.student == student)]


def key_of(task: Task, probe: StudentProbe) -> DifficultyKey:
    return DifficultyKey(task_id=task.id, task_version=task.version, task_digest=task.digest(), student=probe.student, checkpoint=probe.checkpoint,
                         simulator=probe.simulator, time_mode=probe.time_mode)


async def measure(task: Task, probe: StudentProbe, *, spec: BandSpec | None = None, predictor: Predictor | None = None,
                  feats: Mapping[str, float] | None = None, store: DifficultyStore | None = None, skip_alpha: float = 0.1,
                  max_masked: int = 4, note: str = "") -> DifficultyRecord:
    """Predictor-first sequential band measurement of one task (module docstring). Masked draws are re-drawn, at most
    ``max_masked`` in all. With a store, an existing record for the same key is resumed."""
    spec = spec or BandSpec()
    key = key_of(task, probe)
    prev = store.get(key) if store is not None else None
    if prev is not None and prev.source == "measured" and prev.decision is not None and prev.looks and prev.n >= spec.looks[-1]:
        return prev
    feats = dict(feats or {})
    pred: dict[str, Any] = {}
    prior: tuple[float, float] | None = None
    if predictor is not None:
        p0 = predictor.p(feats)
        a, b = predictor.prior(feats)
        pm = masses(a, b, spec)
        call = max(CLASSES, key=lambda c: pm[c])
        pred = {"p": round(p0, 4), "prior_masses": pm, "call": call, "band_call": band_of(p0, spec.lo, spec.hi), "trusted": predictor.trusted}
        if predictor.trusted:
            prior = (a, b)
            out_mass = max(pm["too_hard"], pm["too_easy"])
            if out_mass >= 1.0 - skip_alpha:
                rec = DifficultyRecord(key=key, domain=task.domain, source="predictor", decision=call, confident=True, posterior=pm,
                                       error=round(1 - out_mass, 6), prior=[a, b], predictor=pred, features=feats, at=time.time(), note=note)
                if store is not None:
                    store.append(rec)
                return rec
    rolls: list[Rollout] = list(prev.rollouts) if prev is not None else []
    looks: list[int] = []
    res: BandResult | None = None
    masked_total = sum(1 for r in rolls if r.passed is None)
    for li, target in enumerate(spec.looks):
        while sum(1 for r in rolls if r.passed is not None) < target and masked_total <= max_masked:
            need = target - sum(1 for r in rolls if r.passed is not None)
            got = await probe.rollouts(task, need, start=len(rolls))
            if not got:
                break
            rolls += got
            masked_total = sum(1 for r in rolls if r.passed is None)
        measured = [r for r in rolls if r.passed is not None]
        k, n = sum(1 for r in measured if r.passed), len(measured)
        looks.append(n)
        res = decide(k, n, spec, prior, final=(li == len(spec.looks) - 1) or masked_total > max_masked)
        if res is not None:
            break
    measured = [r for r in rolls if r.passed is not None]
    k, n = sum(1 for r in measured if r.passed), len(measured)
    if res is None or n == 0:
        rec = DifficultyRecord(key=key, domain=task.domain, source="none", rollouts=rolls, passes=k, n=n, masked=len(rolls) - n, looks=looks,
                               predictor=pred, features=feats, cost_usd=round(sum(r.cost_usd for r in rolls), 6), at=time.time(),
                               note=(note + "; " if note else "") + "no measured rollout")
    else:
        rec = DifficultyRecord(key=key, domain=task.domain, source="measured", rollouts=rolls, passes=k, n=n, masked=len(rolls) - n,
                               decision=res.decision, confident=res.confident, posterior=res.posterior, error=res.error, looks=looks, p_hat=res.p_hat,
                               ci=res.ci, prior=res.prior, predictor=pred, features=feats, cost_usd=round(sum(r.cost_usd for r in rolls), 6),
                               at=time.time(), note=note)
    if store is not None:
        store.append(rec)
    return rec


def training_rows(records: Iterable[DifficultyRecord]) -> list[tuple[str, dict[str, float], int, int]]:
    """(task, features, passes, n) of every measured record (predictor-sourced records are never training rows)."""
    return [(r.key.task_id, dict(r.features), r.passes, r.n) for r in records if r.source == "measured" and r.n > 0]


def predictor_accuracy(records: Iterable[DifficultyRecord]) -> dict[str, Any]:
    """How the predictor's recorded call (made before the rollouts) compares with the measured decision: accuracy of
    its band call, of its prior-mass call, and the confusion table. This is the online calibration of the pilot."""
    rows = [r for r in records if r.source == "measured" and r.decision is not None and r.predictor]
    conf: dict[str, dict[str, int]] = {c: {d: 0 for d in CLASSES} for c in CLASSES}
    hit = 0
    for r in rows:
        call = r.predictor.get("band_call")
        if call in conf:
            conf[call][r.decision] += 1  # type: ignore[index]
            hit += call == r.decision
    n = len(rows)
    return {"n": n, "band_call_accuracy": round(hit / n, 4) if n else None, "ci": [round(x, 4) for x in wilson(hit, n)] if n else None,
            "confusion (predicted -> measured)": conf}


def cost_report(records: Iterable[DifficultyRecord]) -> dict[str, Any]:
    rs = [r for r in records if r.source == "measured"]
    by: dict[str, list[float]] = {c: [] for c in CLASSES}
    for r in rs:
        if r.decision is not None:
            by[r.decision].append(r.cost_usd)
    n_roll = sum(len(r.rollouts) for r in rs)
    tot = sum(r.cost_usd for r in rs)
    return {"decisions": len(rs), "rollouts": n_roll, "usd": round(tot, 5), "usd_per_decision": round(tot / len(rs), 5) if rs else None,
            "usd_per_rollout": round(tot / n_roll, 5) if n_roll else None,
            "by_decision": {c: {"n": len(v), "usd": round(sum(v), 5)} for c, v in by.items()},
            "stopped_at_first_look": sum(1 for r in rs if len(r.looks) == 1)}


def records_json(records: Iterable[DifficultyRecord]) -> list[dict[str, Any]]:
    return [json.loads(r.model_dump_json(exclude={"rollouts"})) for r in records]


# ---------------------------------------------------------------------------------------------------------- exports for the RSI loop

class BandVerdict(CrucibleModel):
    """One band decision as the RSI loop reads it (DESIGN §7.7 item 19). No pass count, no p̂: the loop sees only the
    derived verdict, so student reward can never become its optimisation target."""

    task_id: str
    task_version: int
    task_digest: str
    domain: str
    student: str
    checkpoint: str
    simulator: str
    time_mode: str
    decision: Decision | None
    confident: bool
    error: float | None
    source: str
    rollouts: int
    cost_usd: float


class BandExample(CrucibleModel):
    """A too-easy or too-hard task with the task features that explain it (inputs for design lemmas): the features
    whose standardised value times the predictor coefficient pushes p̂ in the decision's direction most."""

    task_id: str
    task_version: int
    domain: str
    student: str
    decision: Decision
    features: dict[str, float]
    explaining: list[str]


def band_verdicts(records: Iterable[DifficultyRecord]) -> list[BandVerdict]:
    return [BandVerdict(task_id=r.key.task_id, task_version=r.key.task_version, task_digest=r.key.task_digest, domain=r.domain, student=r.key.student,
                        checkpoint=r.key.checkpoint, simulator=r.key.simulator, time_mode=r.key.time_mode, decision=r.decision, confident=r.confident,
                        error=r.error, source=r.source, rollouts=len(r.rollouts), cost_usd=r.cost_usd) for r in records]


def band_examples(records: Iterable[DifficultyRecord], predictor: Predictor | None = None, *, top: int = 3) -> list[BandExample]:
    out = []
    for r in records:
        if r.decision not in ("too_easy", "too_hard") or not r.features:
            continue
        expl: list[str] = []
        if predictor is not None:
            sign = 1.0 if r.decision == "too_easy" else -1.0
            contrib = {k: w * (r.features.get(k, 0.0) - predictor.mean.get(k, 0.0)) / predictor.scale.get(k, 1.0)
                       for k, w in predictor.coef.items() if k != "bias"}
            expl = [k for k, v in sorted(contrib.items(), key=lambda kv: -sign * kv[1]) if sign * v > 0][:top]
        out.append(BandExample(task_id=r.key.task_id, task_version=r.key.task_version, domain=r.domain, student=r.key.student, decision=r.decision,
                               features={k: round(v, 4) for k, v in r.features.items()}, explaining=expl))
    return out
