"""The check SPI (research m9-m10/01 §15.2; PATTERN: OpenEnv's Grader protocol with ``depends_on`` /
``requires_capabilities`` / ``applies_to``).

A check is a ``CheckDef``: id, version (bump on any logic change: it invalidates cached results), dependencies, needs,
cost class, ``applies(task) -> None | na_reason`` and ``run(task, ctx) -> Out``. Adapters expose capabilities as
optional methods; a check that needs a capability the adapter lacks **fails** (absence never implies N/A: NeMo-Gym
TE-6), unless the missing capability is itself a task fact with a closed N/A reason.

``AcceptContext.grade`` memoizes grades by (solution id, run, variant), so one grade serves several checks (A1's
oracle grades count toward A4's stable vectors and A12's timing), and persists them keyed by the task's *grading
environment* (``grade_env()``: world, checks, kit and images, but not its negatives or properties) and the solution's
content: editing one negative changes the acceptance key, yet only that negative is graded again. Infrastructure-masked
grades are retried and never stored; a grade still masked after the retries raises ``InfraUnavailable`` and the check
reports ``error``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.accept.cache import Cache
from crucible.accept.model import AcceptanceKey, Need, Status
from crucible.accept.policy import Policy
from crucible.verify.runtime.plan import GradeReport


class InfraUnavailable(RuntimeError):
    pass


class MissingCapability(RuntimeError):
    pass


def need(t: Any, name: str):
    fn = getattr(t, name, None)
    if fn is None:
        raise MissingCapability(f"the task adapter does not provide {name}() (absence never implies not_applicable)")
    return fn


@dataclass
class Out:
    status: Status
    measured: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    na_reason: str | None = None
    detectors: list[str] = field(default_factory=list)

    @classmethod
    def verdict(cls, reasons: list[str], measured: dict, **kw: Any) -> Out:
        return cls(Status.FAIL if reasons else Status.PASS, measured, reasons, **kw)


@dataclass
class AcceptContext:
    policy: Policy
    cache: Cache
    key: AcceptanceKey
    key_digest: str
    needs: frozenset[Need] = frozenset()
    resolve: Callable[[str], bool] = lambda s: False
    budget_usd: float = 0.0
    signoff_root: Path | None = None
    workdir: Path | None = None
    results: dict = field(default_factory=dict)  # check id -> CheckResult (this run, for dependents)
    grades: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    grade_store: Path | None = None  # persistent grades keyed by (grading environment, solution content, run, variant)
    new_grades: int = 0

    def _stored(self, solution: Any, run: int, variant: str) -> Path | None:
        if self.grade_store is None:
            return None
        h = hashlib.sha256(json.dumps([solution, run, variant], sort_keys=True, default=repr).encode()).hexdigest()[:32]
        return self.grade_store / f"{h}.json"

    async def grade(self, t: Any, sol_id: str, solution: Any, *, run: int = 0, variant: str = "", call: Callable[..., Awaitable] | None = None):
        k = (sol_id, run, variant)
        if k in self.grades:
            return self.grades[k]
        sp = self._stored(solution, run, variant)
        if sp is not None and sp.exists():
            d = json.loads(sp.read_text())
            rep = GradeReport(task=d["task"], tier=d["tier"], checks=[], gates=d["gates"], reward=d["reward"], outcome=d["outcome"], infra=d["infra"],
                              trusted=d["trusted"], elapsed_s=d["elapsed_s"], meta=d.get("meta") or {})
            self.grades[k] = (rep, d["fingerprint"])
            return self.grades[k]
        call = call or (lambda r: t.grade(solution, run=r))
        last = None
        for i in range(1 + self.policy.infra_retries):
            t0 = time.monotonic()
            rep, fp = await call(run + 1000 * i)
            last = rep
            self.log.append({"solution": sol_id, "run": run + 1000 * i, "variant": variant, "reward": rep.reward, "infra": rep.infra.get("status"),
                             "seconds": round(time.monotonic() - t0, 3)})
            if rep.infra.get("status", "ok") == "ok":
                self.grades[k] = (rep, fp)
                self.new_grades += 1
                if sp is not None:
                    sp.parent.mkdir(parents=True, exist_ok=True)
                    sp.write_text(json.dumps({"task": rep.task, "tier": rep.tier, "gates": rep.gates, "reward": rep.reward, "outcome": rep.outcome,
                                              "infra": rep.infra, "trusted": rep.trusted, "elapsed_s": rep.elapsed_s, "fingerprint": fp,
                                              "meta": rep.meta, "failing": rep.failing_gates()}, default=str))
                return rep, fp
        raise InfraUnavailable(f"grade of {sol_id} stayed infrastructure-masked after {1 + self.policy.infra_retries} tries: {last.infra if last else None}")


@dataclass
class CheckDef:
    id: str
    version: int
    run: Callable[[Any, AcceptContext], Awaitable[Out]]
    depends_on: tuple[str, ...] = ()
    needs: frozenset[Need] = frozenset()
    cost_class: str = "cpu"
    applies: Callable[[Any], str | None] = lambda t: None
    detectors: tuple[str, ...] = ()
    doc: str = ""


REGISTRY: dict[str, CheckDef] = {}


def register(id: str, version: int, **kw: Any):
    def deco(fn: Callable[[Any, AcceptContext], Awaitable[Out]]):
        REGISTRY[id] = CheckDef(id=id, version=version, run=fn, doc=(fn.__doc__ or "").strip().split("\n")[0], **kw)
        return fn

    return deco
