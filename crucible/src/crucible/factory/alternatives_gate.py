"""S9b, the ten-competent-people gate made live (swarm-pipeline/00 §5.8, §8.3; ``EVALS`` L4; the HUD task note: "imagine
giving the task to 10 competent people").

A check is fair only if every functionally correct solution passes it. The oracle is one solution; S9b collects others
written **from the bug report alone** by models of families other than the task's builders, and asks both verifiers
about each of them:

- **the sources** (``harvest``): every graded solver end state stored for the task (the frontier fairness runs, the
  student's runs, and the alternative sessions S9b launches: the same World and harness as a training episode, so an
  alternative is exactly what a competent agent would hand in). A state counts once (``end_fingerprint``); masked,
  hacked and no-progress states are left out. Each keeps its policy and family.
- **verifier 1** is the task's own grader (``verifier2.WindowRunner.hidden``: the hidden tests, end state only).
- **verifier 2** is the blind second verifier's suite (``verifier2.write_shadow``: written from the request and the
  repository at A, never the oracle or a hidden test; qualified sound on the oracle by ``qualify``). Its verdict on a
  state counts only when the suite's family is not the state's (a model never confirms its own family's work).
- **the behaviour check** is the bug report's own reproductions where the task has them (S7r, ``debug_repro``), run on
  the state.

**Classes** (``classify``):

| class | when | what happens |
|---|---|---|
| ``accepted`` | verifier 1 full pass, and no independent check fails it | a valid alternative: S8's ten-competent-people test and the trace audit's ``AlternativesPort`` use it |
| ``false_fail`` | verifier 1 fails it, while verifier 2 passes every requirement and every reproduction passes (at least one of them measured, independent) | a ``false_fail`` candidate with its artifact (the patch, the failing hidden tests, verifier 2's tests): the grader may reject a correct fix. It blocks the task until the trace audit resolves it (``traceaudit.accept_false_fail``) |
| ``lucky_pass`` | verifier 1 full pass, while an independent check fails a requirement | a ``verifier_gap`` candidate for the audit |
| ``wrong`` | verifier 1 fails it and so does an independent check | an honest failure |
| ``unknown`` | nothing independent was measured | listed |

**The gate** (``gate``): ``fail`` on any ``false_fail`` candidate; ``pass`` when at least ``min_alternatives`` accepted
alternatives exist, at least one from a family other than the prompt writer's (``alternatives.coverage``); else
``unmeasured`` (too few alternatives: never a pass). Agreement between the verifiers is published per task: Cohen's κ
over every (state, requirement) pair both measured (``kappa``), with the disagreements listed for the tie-break.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.models import family_of


@dataclass
class Candidate:
    id: str
    source: str  # solver (a stored episode) | session (an S9b alternative session)
    policy: str
    family: str
    patch: str
    arm: str = ""
    phi: float | None = None
    stored_reward: float | None = None


@dataclass
class Classified:
    id: str
    cls: str  # accepted | false_fail | lucky_pass | wrong | unknown
    family: str
    v1_full: bool | None
    v1_milestones: dict[str, bool] = field(default_factory=dict)
    v2: dict[str, bool | None] = field(default_factory=dict)  # requirement -> verifier 2 (None: unmeasured or not independent)
    v2_independent: bool = True
    repro: dict[str, bool | None] = field(default_factory=dict)
    failing_hidden: list[str] = field(default_factory=list)
    why: str = ""
    policy: str = ""
    source: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _load(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def family_of_policy(policy: str) -> str:
    """``solver:tinker:deepseek-ai/DeepSeek-V4.1-Flash:peft:262144@tinker`` -> ``deepseek``: the family of the model id in a
    run record's policy string ("" when it names none)."""
    model = next((p for p in policy.split("@", 1)[0].split(":") if "/" in p), "")
    return family_of(model) if model else ""


def harvest(task_id: str, roots: Iterable[Path], *, min_phi: float = 0.5, exclude_families: Iterable[str] = ()) -> list[Candidate]:
    """Every stored, graded end state of ``task_id`` under ``roots`` (``<root>/arms/<arm>/<task>/rollouts/**/a<n>``) with
    progress (Phi >= ``min_phi`` or a full pass), not masked, not hacked, from a family not in ``exclude_families``."""
    out: list[Candidate] = []
    skip = {f for f in exclude_families if f}
    for root in roots:
        for rep_p in sorted(Path(root).glob(f"arms/*/{task_id}/rollouts/**/verifier/report.json")):
            a = rep_p.parent.parent
            rep = _load(rep_p) or {}
            meta = rep.get("meta") or {}
            if (rep.get("infra") or {}).get("status") != "ok" or meta.get("hack_suspect") or meta.get("apply_errors"):
                continue
            phi = meta.get("phi")
            if not (meta.get("end_state_full_pass") or meta.get("full_pass") or (phi is not None and float(phi) >= min_phi)):
                continue
            rec = _load(a / "record.json") or _load(a.parent / "record.json") or {}
            policy = str(rec.get("policy") or rec.get("model") or (_load(a.parents[3] / "done.json") or {}).get("model") or "")
            fam = family_of_policy(policy)
            if fam in skip:
                continue
            patch_p = a / "artifacts" / "model.patch"
            if not patch_p.exists():
                continue
            arm = a.parts[a.parts.index("arms") + 1] if "arms" in a.parts else ""
            out.append(Candidate(f"{arm}/{a.parent.name}/{a.name}", "session" if arm.startswith("alt-") else "solver", policy, fam,
                                 patch_p.read_text(errors="replace"), arm, None if phi is None else float(phi), rep.get("reward")))
    return out


def dedup(task: Any, cands: Sequence[Candidate]) -> list[Candidate]:
    """One candidate per end state (``task.end_fingerprint``): two episodes that wrote the same fix count once."""
    seen: set[str] = set()
    out = []
    for c in cands:
        try:
            fp = task.end_fingerprint(c.patch)
        except Exception:  # noqa: BLE001 - an end state that does not apply is still a candidate (the grader says so)
            fp = c.id
        if fp not in seen:
            seen.add(fp)
            out.append(c)
    return out


def classify_one(c: Candidate, *, v1: Mapping[str, Any] | None, v2: Mapping[str, bool | None] | None, v2_family: str,
                 repro: Mapping[str, bool | None] | None, req_milestone: Mapping[str, str], milestone_tests: Mapping[str, Sequence[str]]) -> Classified:
    """The class of one alternative from both verifiers and the reproductions (module doc's table)."""
    indep = bool(v2_family) and c.family != v2_family
    v2u = {r: (v if indep else None) for r, v in (v2 or {}).items()}
    rp = dict(repro or {})
    if v1 is None:
        return Classified(c.id, "unknown", c.family, None, {}, v2u, indep, rp, [], "verifier 1 unmeasured", c.policy, c.source)
    full = bool(v1.get("reward") == 1.0)
    ms = {k: bool(v) for k, v in (v1.get("milestones") or {}).items()}
    failing = [t for m, ok in ms.items() if not ok for t in milestone_tests.get(m, []) if (v1.get("statuses") or {}).get(t) != "passed"][:30]
    signals = [v for v in [*v2u.values(), *rp.values()] if v is not None]
    if not signals:
        cls, why = ("accepted" if full else "unknown"), "no independent check measured" + ("" if full else ": verifier 1 alone fails it")
        if full:
            why = "verifier 1 full pass; no independent check measured"
    elif full:
        bad = sorted([f"v2:{r}" for r, v in v2u.items() if v is False] + [f"repro:{r}" for r, v in rp.items() if v is False])
        cls, why = ("lucky_pass", f"verifier 1 passes, independent checks fail {bad}") if bad else ("accepted", "both verifiers pass")
    else:
        all_ok = all(signals) and all(v2u.get(r) is not False for r in req_milestone)
        covered = all(v2u.get(r) is True or rp.get(r) is True for r in req_milestone)
        if all_ok and covered:
            cls, why = "false_fail", f"verifier 1 fails {failing[:5]} while every requirement passes an independent check"
        else:
            cls, why = "wrong", "verifier 1 and an independent check both fail it"
    return Classified(c.id, cls, c.family, full, ms, v2u, indep, rp, failing, why, c.policy, c.source)


def kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    """Cohen's κ for two binary raters (None with fewer than 2 pairs or no variance in the expected agreement)."""
    n = len(pairs)
    if n < 2:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa, pb = sum(1 for a, _ in pairs if a) / n, sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1.0 else round((po - pe) / (1 - pe), 4)


def agreement(rows: Sequence[Mapping[str, Any]], req_milestone: Mapping[str, str]) -> dict[str, Any]:
    """κ over every (state, requirement) pair where verifier 1 (the milestone) and verifier 2 both have a verdict, plus the
    disagreements for the tie-break. ``rows``: ``{"id", "kind", "v1_milestones", "v2"}``."""
    pairs: list[tuple[bool, bool]] = []
    dis: list[dict[str, Any]] = []
    for r in rows:
        for rid, mid in req_milestone.items():
            a, b = (r.get("v1_milestones") or {}).get(mid), (r.get("v2") or {}).get(rid)
            if a is None or b is None:
                continue
            pairs.append((bool(a), bool(b)))
            if bool(a) != bool(b):
                dis.append({"state": r["id"], "kind": r.get("kind"), "requirement": rid, "milestone": mid, "v1": bool(a), "v2": bool(b)})
    return {"pairs": len(pairs), "agree": sum(1 for a, b in pairs if a == b), "kappa": kappa(pairs), "disagreements": dis}


def gate(classified: Sequence[Classified], *, writer_family: str, min_alternatives: int = 2) -> dict[str, Any]:
    """S9b's verdict (module doc)."""
    from crucible.factory.alternatives import Alternative, coverage

    acc = [c for c in classified if c.cls == "accepted"]
    # only an alternative an independent check confirmed tests the grader's fairness: one accepted by verifier 1 alone
    # proves nothing about verifier 1 (it is still a positive example for the audit's new checks)
    indep = [c for c in acc if any(v is True for v in c.v2.values()) or any(v is True for v in c.repro.values())]
    cov = coverage([Alternative(c.id, c.source, "", c.policy, c.family, accepted=True) for c in indep], writer_family=writer_family)
    ff = [c for c in classified if c.cls == "false_fail"]
    status = "fail" if ff else ("pass" if len(indep) >= min_alternatives and cov["other_family"] >= 1 else "unmeasured")
    why = (f"{len(ff)} correct-looking fix(es) the grader rejects: {[c.id for c in ff][:5]}" if ff else
           f"{len(indep)} independently confirmed alternative(s) ({cov['families']})" if status == "pass" else
           f"{len(indep)} independently confirmed alternative(s), need {min_alternatives} with one from a family other than {writer_family!r}")
    return {"gate": "S9b", "status": status, "ok": status == "pass", "why": why, "coverage": cov,
            "counts": {**{k: sum(1 for c in classified if c.cls == k) for k in ("accepted", "false_fail", "lucky_pass", "wrong", "unknown")},
                       "confirmed": len(indep)},
            "false_fail": [c.to_json() for c in ff], "lucky_pass": [c.to_json() for c in classified if c.cls == "lucky_pass"]}


async def run_gate(task: Any, cands: Sequence[Candidate], *, suite: Any = None, writer_family: str = "", repros: Mapping[str, Any] | None = None,
                   min_alternatives: int = 2, workdir: Path | None = None) -> dict[str, Any]:
    """S9b on one task: both verifiers on the oracle, the floors (the start), the mutants and every candidate; the
    qualification of verifier 2 (sound on the oracle, failing the start); each candidate's class; κ; the gate."""
    from crucible.factory.generators.debug import scope
    from crucible.factory.verifier2 import WindowRunner, qualify

    reqs = list(task.requirements)
    req_ms = {str(r["id"]): str(r["milestone"]) for r in reqs if r.get("milestone")}
    ms_tests = {m["id"]: list(m["tests"]) for m in task.milestones}
    with scope(str(getattr(task, "lang", None) or task.window.get("lang") or "python")):
        runner = WindowRunner(task)
        cands = dedup(task, cands)
        qual = None
        known: list[dict[str, Any]] = []
        if suite is not None and suite.files:
            oracle = await runner.shadow(suite, task.oracle_patch())
            start = await runner.shadow(suite, "")
            muts = {str(m.id): (str(m.milestone), await runner.shadow(suite, m.solution)) for m in task.mutants()}
            qual = qualify(suite, oracle=oracle, alternatives={}, start=start, mutants=muts, req_milestone=req_ms)
            for sid, sol, kind in [("oracle", task.oracle_patch(), "oracle"), ("start", "", "start"), *[(str(m.id), m.solution, "mutant") for m in task.mutants()]]:
                v1 = await runner.hidden(sol)
                st = await runner.shadow(suite, sol)
                known.append({"id": sid, "kind": kind, "v1_milestones": (v1 or {}).get("milestones") or {}, "v2": qual.verdict(st)})
        rows: list[Classified] = []
        for c in cands:
            v1 = await runner.hidden(c.patch)
            v2 = qual.verdict(await runner.shadow(suite, c.patch)) if qual is not None else None
            rp = None
            if repros:
                from crucible.factory.generators.debug_repro import run_on

                by_ms = await run_on(task, repros, c.patch, workdir=workdir)  # keyed by milestone; the classes read requirements
                rp = {rid: by_ms[mid] for rid, mid in req_ms.items() if mid in by_ms}
            rows.append(classify_one(c, v1=v1, v2=v2, v2_family=getattr(suite, "family", "") if suite is not None else "", repro=rp,
                                     req_milestone=req_ms, milestone_tests=ms_tests))
    g = gate(rows, writer_family=writer_family, min_alternatives=min_alternatives)
    agree = agreement([*known, *[{"id": r.id, "kind": "alternative", "v1_milestones": r.v1_milestones, "v2": r.v2} for r in rows]], req_ms)
    return {**g, "task": str(task.task_id), "agreement": agree, "qualification": qual.to_json() if qual is not None else None,
            "suite": {"model": getattr(suite, "model", ""), "family": getattr(suite, "family", ""), "files": sorted(getattr(suite, "files", {}) or {}),
                      "problems": list(getattr(suite, "problems", []) or [])} if suite is not None else None,
            "classified": [r.to_json() for r in rows],
            "accepted_patches": {r.id: c.patch for r, c in zip(rows, cands, strict=True) if r.cls == "accepted"}}


# ---- the trace audit's ports, live (traceaudit.AlternativesPort, SecondVerifierPort) ----------------------------------------
class LiveAlternatives:
    """``traceaudit.AlternativesPort`` from S9b's stored results (``<dir>/<task>.json``): the accepted alternatives."""

    def __init__(self, results_dir: Path):
        self.dir = Path(results_dir)

    async def alternatives(self, task_id: str) -> list[tuple[str, Any]]:
        res = _load(self.dir / f"{task_id.split('~', 1)[0]}.json") or {}
        return sorted((res.get("accepted_patches") or {}).items())


class LiveSecondVerifier:
    """``traceaudit.SecondVerifierPort``: verifier 2 (the stored, qualified suite of ``<dir>/<task>.suite.json``) on an end
    state. True when every requirement's sound tests pass, False when one fails, None when unmeasured. ``task_of``
    resolves a task id to the task."""

    def __init__(self, results_dir: Path, task_of: Any):
        self.dir, self.task_of = Path(results_dir), task_of

    async def verdict(self, task_id: str, solution: Any) -> bool | None:
        from crucible.factory.generators.debug import scope
        from crucible.factory.verifier2 import Qualification, ShadowSuite, WindowRunner

        tid = task_id.split("~", 1)[0]
        s, r = _load(self.dir / f"{tid}.suite.json"), _load(self.dir / f"{tid}.json") or {}
        if not s or not r.get("qualification"):
            return None
        suite = ShadowSuite.from_json(s)
        q = r["qualification"]
        qual = Qualification(q["tests"], q["requirements"])
        task = self.task_of(tid)
        with scope(str(getattr(task, "lang", None) or task.window.get("lang") or "python")):
            v = qual.verdict(await WindowRunner(task).shadow(suite, solution))
        got = [x for x in v.values() if x is not None]
        if not got:
            return None
        return all(got)
