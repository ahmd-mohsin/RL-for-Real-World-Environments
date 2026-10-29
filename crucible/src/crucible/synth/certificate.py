"""Validity certificates and admission (part2-paper-core §6, claim C2).

For a task and falsifier class k: ``n_k`` valid probes, ``c_k`` confirmed counterexamples and ``U_k``, the exact
one-sided Clopper–Pearson 95% upper bound on the counterexample rate *under class k's probe distribution* (at c = 0 it
is ``1 - 0.05**(1/n)``, about 3/n; ``verify.calibrate.clopper_pearson_upper``, the bound ``factory/flake.py`` uses).
Suspects (awaiting adjudication) and equivalents are recorded beside it.

A certificate is honest about its reach: it records which classes were **applicable** and why the others were not (a
black-box World gets no frame guarantee, a World without ``perturb`` no PERT trials). Admission (``AdmissionPolicy``,
thresholds configurable, defaults from §6):
- every deterministic class that ran has ``c_k <= max_confirmed`` (0);
- ``U_k <= u_max[k]`` where a bound is set (``U_MUT <= 0.05``, which needs n >= 59 mutants);
- every class in ``require_applicable`` ran;
- ALT has at least one pass from a non-builder family (``require_alt_pass``); ATK has no confirmed hack;
- no suspects are open (``allow_open_suspects``), since a suspect is an unadjudicated possible counterexample.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.falsify.base import DETERMINISTIC, ClassResult
from crucible.synth.task import Task
from crucible.verify.calibrate import clopper_pearson_upper

ALPHA = 0.05


class ClassStat(CrucibleModel):
    cls: str
    applicable: bool = True
    reason: str | None = None
    n: int = 0
    c: int = 0
    upper: float = 1.0
    suspects: int = 0
    equivalents: int = 0
    invalid: int = 0
    passes: int = 0  # ALT: attempts that passed
    defects: dict[str, int] = Field(default_factory=dict)  # defect code -> confirmed count


class AdmissionPolicy(CrucibleModel):
    deterministic: list[str] = Field(default_factory=lambda: list(DETERMINISTIC))
    max_confirmed: int = 0
    u_max: dict[str, float] = Field(default_factory=lambda: {"MUT": 0.05})
    require_applicable: list[str] = Field(default_factory=lambda: ["MUT", "NULL", "RED", "TAINT", "REP"])
    require_alt_pass: bool = True
    atk_max_confirmed: int = 0
    allow_open_suspects: bool = False


class Certificate(CrucibleModel):
    schema_version: str = "crucible.synth.certificate/v1"
    task_id: str
    task_version: int
    task_digest: str
    classes: dict[str, ClassStat] = Field(default_factory=dict)
    admitted: bool = False
    reasons: list[str] = Field(default_factory=list)
    policy: AdmissionPolicy = Field(default_factory=AdmissionPolicy)
    regressions_reopened: list[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        """No confirmed counterexample in any class (the M2 "clean certificate")."""
        return all(s.c == 0 for s in self.classes.values())


def class_stat(r: ClassResult) -> ClassStat:
    if not r.applicable:
        return ClassStat(cls=r.cls, applicable=False, reason=r.reason)
    defects: dict[str, int] = {}
    for f in r.confirmed:
        if f.defect:
            defects[f.defect] = defects.get(f.defect, 0) + 1
    valid = [f for f in r.findings if f.outcome.valid]
    return ClassStat(cls=r.cls, n=r.n, c=r.c, upper=clopper_pearson_upper(r.c, r.n, ALPHA), suspects=len(r.suspects),
                     equivalents=sum(1 for f in valid if f.label == "equivalent"), invalid=len(r.findings) - len(valid),
                     passes=sum(1 for f in valid if f.outcome.passed), defects=defects)


def certify(task: Task, results: Iterable[ClassResult], policy: AdmissionPolicy | None = None, *, reopened: Iterable[str] = ()) -> Certificate:
    pol = policy or AdmissionPolicy()
    stats = {r.cls: class_stat(r) for r in results}
    why: list[str] = []
    for k in pol.require_applicable:
        s = stats.get(k)
        if s is None or not s.applicable:
            why.append(f"{k}:not_applicable")
    for k in pol.deterministic:
        s = stats.get(k)
        if s is not None and s.applicable and s.c > pol.max_confirmed:
            why.append(f"{k}:confirmed={s.c}")
    for k, u in pol.u_max.items():
        s = stats.get(k)
        if s is not None and s.applicable and s.upper > u:
            why.append(f"{k}:upper={s.upper:.4f}>{u}")
    alt = stats.get("ALT")
    if pol.require_alt_pass and (alt is None or not alt.applicable or alt.passes == 0):
        why.append("ALT:no_pass")
    atk = stats.get("ATK")
    if atk is not None and atk.applicable and atk.c > pol.atk_max_confirmed:
        why.append(f"ATK:confirmed={atk.c}")
    if not pol.allow_open_suspects:
        why += [f"{k}:suspects={s.suspects}" for k, s in sorted(stats.items()) if s.suspects]
    rx = sorted(reopened)
    if rx:
        why.append(f"archive:reopened={len(rx)}")
    return Certificate(task_id=task.id, task_version=task.version, task_digest=task.digest(), classes=stats, admitted=not why, reasons=why, policy=pol,
                       regressions_reopened=rx)


def certificate_json(c: Certificate) -> str:
    return c.model_dump_json(indent=1)


def summary(c: Certificate) -> dict[str, Any]:
    return {k: {"n": s.n, "c": s.c, "U": round(s.upper, 4)} if s.applicable else {"applicable": False, "reason": s.reason} for k, s in sorted(c.classes.items())}
