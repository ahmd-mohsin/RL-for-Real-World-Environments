"""Validity certificates and admission (part2-paper-core §6, claim C2).

For a task and falsifier class k: ``n_k`` valid probes, ``c_k`` confirmed counterexamples and ``U_k``, the exact
one-sided Clopper–Pearson 95% upper bound on the counterexample rate *under class k's probe distribution* (at c = 0 it
is ``1 - 0.05**(1/n)``, about 3/n; ``verify.calibrate.clopper_pearson_upper``, the bound ``factory/flake.py`` uses).
Suspects (awaiting adjudication) and equivalents are recorded beside it.

A certificate is honest about its reach: it records which classes were **applicable** and why the others were not (a
black-box World gets no frame guarantee, a World without ``perturb`` no PERT trials). Admission (``AdmissionPolicy``,
thresholds configurable, defaults from the policy ledger's R-admission-fixed, §6; DESIGN §7.7 item 24):
- every deterministic class that ran has ``c_k <= max_confirmed`` (0), and so do the held-out mutants (``MUT-HO``, Part 1
  V5: mutants no repair saw; a survivor means the contract was fitted to the shown ones);
- ``U_k <= u_max[k]`` where a bound is set (``U_MUT <= 0.05``, which needs n >= 59 mutants);
- every class in ``require_applicable`` ran;
- ALT has at least one pass from a non-builder family (``require_alt_pass``); ATK has no confirmed hack;
- no suspects are open (``allow_open_suspects``), since a suspect is an unadjudicated possible counterexample.

**Certified vs provisional (§6, M3).** A bound in ``u_max`` that is missed *only* because n is small (c = 0, so U is
the rule of three on too few distinct probes) does not reject the task: it is admitted as ``provisional`` with its U
reported, and is never counted as ``certified``. Every other reason rejects. ``status`` is ``certified``,
``provisional`` or ``rejected``; ``admitted`` is true for the first two; ``reasons`` lists the rejection reasons and
``provisional_for`` the bounds that were missed for want of n. ``provisional_ok = False`` turns the soft miss back into a
rejection.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth import policy as _policy
from crucible.synth.falsify.base import ClassResult
from crucible.synth.task import Task
from crucible.verify.calibrate import clopper_pearson_upper

ALPHA = 0.05
HELDOUT = "MUT-HO"  # synth.falsify.mut.MUTHeldOut: never in a policy's lists (no U bound: a check, not a sample-size claim)


def kill_rate(r: ClassResult) -> float | None:
    """The share of a mutant class's distinct non-equivalent mutants that the contract rejected (Identity, duplicates,
    invalid runs and equivalents excluded); None with none."""
    trials = [f for f in r.findings if f.outcome.valid and f.op != "Identity" and f.label != "equivalent"]
    return sum(1 for f in trials if f.label == "clean") / len(trials) if trials else None


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


def _adm(name: str) -> Any:
    return _policy.admission_params()[name]


class AdmissionPolicy(CrucibleModel):
    """Defaults are the policy's R-admission-fixed (DESIGN §7.7 item 24): the classes with the zero-confirmed rule
    (ORACLE and LEAK included), U_MUT <= 0.05, the required classes, the ALT-pass rule. A looser setting is allowed for
    measurement and tests, and every certificate made under it records the deviation (``Certificate.deviations``)."""

    deterministic: list[str] = Field(default_factory=lambda: list(_adm("zero_confirmed_classes")))
    max_confirmed: int = Field(default_factory=lambda: int(_adm("max_confirmed")))
    u_max: dict[str, float] = Field(default_factory=lambda: {"MUT": float(_adm("u_max_mut"))})
    require_applicable: list[str] = Field(default_factory=lambda: list(_adm("require_applicable")))
    require_alt_pass: bool = Field(default_factory=lambda: bool(_adm("require_alt_pass")))
    atk_max_confirmed: int = Field(default_factory=lambda: int(_adm("atk_max_confirmed")))
    allow_open_suspects: bool = Field(default_factory=lambda: bool(_adm("allow_open_suspects")))
    provisional_ok: bool = True
    scope: str = "tool_world"  # "repository": R-patch-scope's recorded exemption applies


class Certificate(CrucibleModel):
    schema_version: str = "crucible.synth.certificate/v1"
    task_id: str
    task_version: int
    task_digest: str
    classes: dict[str, ClassStat] = Field(default_factory=dict)
    admitted: bool = False
    status: Literal["certified", "provisional", "rejected"] = "rejected"
    reasons: list[str] = Field(default_factory=list)
    provisional_for: list[str] = Field(default_factory=list)
    policy: AdmissionPolicy = Field(default_factory=AdmissionPolicy)
    regressions_reopened: list[str] = Field(default_factory=list)
    policy_version: str | None = None  # the policy ledger's version this certificate was decided under (None: before item 24)
    deviations: list[str] = Field(default_factory=list)  # where ``policy`` is looser than the policy's admission rule for its scope

    @property
    def certified(self) -> bool:
        return self.status == "certified"

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
    return decide(task.id, task.version, task.digest(), stats, pol, reopened=reopened)


def decide(task_id: str, version: int, digest: str, stats: dict[str, ClassStat], pol: AdmissionPolicy, *, reopened: Iterable[str] = ()) -> Certificate:
    """Admission over per-class stats (``certify``'s rule; ``with_ladder`` re-decides after the ALT solver ladder)."""
    why: list[str] = []
    for k in pol.require_applicable:
        s = stats.get(k)
        if s is None or not s.applicable:
            why.append(f"{k}:not_applicable")
    for k in [*pol.deterministic, HELDOUT]:
        s = stats.get(k)
        if s is not None and s.applicable and s.c > pol.max_confirmed:
            why.append(f"{k}:confirmed={s.c}")
    soft: list[str] = []
    for k, u in pol.u_max.items():
        s = stats.get(k)
        if s is not None and s.applicable and s.upper > u:
            (soft if s.c == 0 and pol.provisional_ok else why).append(f"{k}:upper={s.upper:.4f}>{u}")
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
    status = "rejected" if why else ("provisional" if soft else "certified")
    return Certificate(task_id=task_id, task_version=version, task_digest=digest, classes=stats, admitted=not why, status=status,
                       reasons=why, provisional_for=soft, policy=pol, regressions_reopened=rx, policy_version=_policy.tag(),
                       deviations=_policy.admission_deviations(pol, pol.scope))


def with_ladder(cert: Certificate, rows: Iterable[Any], verdicts: dict[int, bool | None] | None = None) -> Certificate:
    """R-ALT-whole: re-decide a certificate after the ALT solver ladder (``solvers.escalate_alt``) ran on the WHOLE task.

    ``rows`` are the ladder's attempts (``solvers.AttemptRow``: ``passed``, ``error``), ``verdicts`` the adjudication
    of each failed attempt by index (True: a valid attempt the contract rejects, a confirmed D6; False: invalid; None or
    missing: undecided, a suspect). Passes add to ALT's passes; nothing else in the certificate changes, and the
    admission rule is the certificate's own."""
    verdicts = verdicts or {}
    rows = list(rows)
    s = cert.classes.get("ALT") or ClassStat(cls="ALT")
    valid = [(i, r) for i, r in enumerate(rows) if getattr(r, "error", None) is None]
    passes = sum(1 for _, r in valid if r.passed)
    d6 = sum(1 for i, r in valid if not r.passed and verdicts.get(i) is True)
    sus = sum(1 for i, r in valid if not r.passed and verdicts.get(i) is None)
    n, c = s.n + len(valid), s.c + d6
    defects = dict(s.defects)
    if d6:
        defects["D6"] = defects.get("D6", 0) + d6
    alt = s.model_copy(update={"applicable": True, "reason": None, "n": n, "c": c, "upper": clopper_pearson_upper(c, n, ALPHA) if n else 1.0,
                               "passes": s.passes + passes, "suspects": s.suspects + sus, "defects": defects})
    return decide(cert.task_id, cert.task_version, cert.task_digest, {**cert.classes, "ALT": alt}, cert.policy, reopened=cert.regressions_reopened)


def certificate_json(c: Certificate) -> str:
    return c.model_dump_json(indent=1)


def summary(c: Certificate) -> dict[str, Any]:
    return {k: {"n": s.n, "c": s.c, "U": round(s.upper, 4)} if s.applicable else {"applicable": False, "reason": s.reason} for k, s in sorted(c.classes.items())}
