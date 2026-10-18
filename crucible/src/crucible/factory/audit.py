"""Trace audit (05 §6.5, §7; R4, R7): a FailureCause for every failing solver trace, false-pass / false-fail candidates
where the auditor and the verifier disagree, and the cause mix with intervals.

- **Closed taxonomy** (``cards.FailureCause``). The judge (an injectable ``JsonClient`` on another model family) must
  answer one cause or ``abstain``; anything else is rejected and counted as an abstention.
- **Evidence required.** Every label quotes the trace verbatim (whitespace and case folded); a quote that is not in the
  trace rejects the label. A judge cannot assert a cause it cannot point at.
- **Facts are not judged.** A masked verdict is ``env_flake`` and a run that stopped on its step / token / time cap is
  ``budget``, without the judge (``AuditConfig.deterministic``).
- **Seeded controls.** Cases with a known cause (a World bug planted in a tool, a verifier that fails a correct answer)
  are mixed into the batch under opaque ids. The auditor is trusted only if it labels them right: a judge that always
  says ``skill_gap`` fails on the first seeded World bug.
- **FP / FN candidates.** Verifier pass + auditor "incorrect" is a false-pass candidate; verifier fail + auditor "correct"
  (or cause ``verifier_defect``) is a false-fail candidate. Each is a defect for the verifier writer (§6.5).
- **Cause mix** over the audited real failures: share per cause with a Wilson interval. *v1*: ≥ 70% skill gap, or the
  task is hard for the wrong reason; any World bug or spec defect sends the task back to its owner.

**P3 (swarm-pipeline/00 §8.4: the auditor is audited).**
- **Same-family judging is forbidden by config.** ``audit`` refuses a batch in which a trace's policy shares the
  auditor model's family (``cross_family_problems``; ``AuditConfig.forbid_same_family``): a deepseek auditor never labels a
  deepseek hacker's trace. ``split_by_family`` routes such traces to a third family or to deterministic facts only.
- **Human κ sample.** ``human_queue`` writes the 10% sample (``human_sample``) of a batch, real trace ids with the
  auditor's label and quote, for a human to label; ``kappa_vs_human`` reports Cohen's κ on the cause once labels exist.
- Seeded controls and the trust rule are unchanged: ``AuditReport.auditor_trusted`` is True only when every control is
  right, and ``factory.traceaudit`` drops every label-derived proposal of an untrusted batch.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from crucible.factory.cards import FailureCause
from crucible.factory.floors import Frac, JsonClient

CAUSES: tuple[str, ...] = get_args(FailureCause)
OWNER_OF = {"world_bug": "world", "spec_defect": "spec", "unstated_format": "prompt", "verifier_defect": "verifier"}  # where the task goes back

JUDGE_SYSTEM = """You audit one failing (or passing) attempt at an RL task. Decide WHY it failed, from a closed list:
- skill_gap: the task is fair and solvable as stated; the policy lacked the skill or made a mistake.
- spec_defect: the prompt is ambiguous, contradictory or impossible as written.
- unstated_format: the policy did the work but missed a format or convention the prompt never stated.
- world_bug: the environment (a tool, the data, another actor) behaved wrongly.
- verifier_defect: the attempt is correct but the verifier failed it (or grades something the prompt does not ask).
- budget: the policy ran out of steps, tokens or time while still making progress.
- env_flake: an infrastructure failure unrelated to the task (crash, timeout of the platform).
Use "abstain" when the trace does not let you decide. Also say whether the attempt actually satisfies every requirement
(solution_correct: true, false or null if you cannot tell). Quote the trace VERBATIM as evidence (one line or phrase).
Reply with JSON: {"cause": "<cause or abstain>", "solution_correct": true|false|null, "evidence": "<verbatim quote>",
"rationale": "<one sentence>"}"""


@dataclass
class Trace:
    id: str
    text: str  # the rendered trace the auditor reads: actions, tool results, messages, final answer
    verifier_pass: bool | None  # None: masked grade
    policy: str = ""
    failing_checks: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # {"stop": "max_steps"}, ...


@dataclass
class SeededCase:
    """A control with a known cause, planted in the batch."""

    trace: Trace
    cause: str
    solution_correct: bool | None = None
    # (M17 e2e) a control built from another known case is judged against that case's own request: an
    # ``unstated_format`` control *is* a request missing a stated format, which the batch's prompt cannot show
    prompt: str | None = None
    requirements: Sequence[Any] | None = None


@dataclass
class AuditLabel:
    trace_id: str
    cause: str | None  # None: abstained, rejected, or a passing trace
    solution_correct: bool | None
    evidence: str = ""
    rationale: str = ""
    status: Literal["labelled", "abstained", "rejected", "deterministic"] = "labelled"
    why: str = ""


@dataclass
class AuditConfig:
    skill_gap_min: float = 0.7  # v1 (05 §7)
    min_evidence_chars: int = 8
    budget_stops: tuple[str, ...] = ("max_steps", "max_tokens", "timeout", "budget", "wall_clock")
    deterministic: bool = True
    seeded_recall_min: float = 1.0  # every seeded control must be labelled right
    human_share: float = 0.10  # v1: the human samples 10%
    send_back_causes: tuple[str, ...] = ("world_bug", "spec_defect")
    concurrency: int = 8  # (M17 e2e) judge calls in flight: one at a time made a 16-trace batch a 10-minute stage
    forbid_same_family: bool = True  # P3 (00 §8.4): a trace whose policy shares the auditor's model family is refused


def _fold(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


class Auditor:
    """The failure auditor role. ``label`` never raises on a bad reply: it rejects it with a reason."""

    def __init__(self, client: JsonClient, *, model: str = "", config: AuditConfig | None = None, max_tokens: int = 800):
        self.client, self.model, self.config, self.max_tokens = client, model, config or AuditConfig(), max_tokens

    async def label(self, t: Trace, *, prompt: str, requirements: Sequence[Any] = ()) -> AuditLabel:
        c = self.config
        if c.deterministic and t.verifier_pass is None:
            return AuditLabel(t.id, "env_flake", None, status="deterministic", why="masked grade")
        if c.deterministic and t.verifier_pass is False and str(t.meta.get("stop", "")) in c.budget_stops:
            return AuditLabel(t.id, "budget", None, status="deterministic", why=f"stopped on {t.meta['stop']}")
        user = json.dumps({"task_prompt": prompt, "requirements": [{"id": r.id, "text": r.text} for r in requirements],
                           "verifier_verdict": "pass" if t.verifier_pass else "fail", "failing_checks": t.failing_checks, "trace": t.text})
        try:
            r = await self.client.complete_json([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}], max_tokens=self.max_tokens)
        except Exception as e:  # noqa: BLE001 - a failed call is an abstention with its reason, never a crash of the stage
            return AuditLabel(t.id, None, None, status="rejected", why=f"client error: {e}"[:300])
        return validate(t, r if isinstance(r, dict) else {}, config=c)


def validate(t: Trace, reply: dict, *, config: AuditConfig | None = None) -> AuditLabel:
    """A judge reply checked against the taxonomy and the trace."""
    c = config or AuditConfig()
    cause = reply.get("cause")
    sc = reply.get("solution_correct")
    sc = sc if isinstance(sc, bool) else None
    ev = str(reply.get("evidence") or "")
    rat = str(reply.get("rationale") or "")[:500]
    if cause == "abstain":
        return AuditLabel(t.id, None, sc, ev, rat, status="abstained")
    if t.verifier_pass and cause in (None, "", "none"):
        cause = None  # a passing trace needs no cause, only the correctness judgement
    elif cause not in CAUSES:
        return AuditLabel(t.id, None, None, ev, rat, status="rejected", why=f"cause {cause!r} is not in the closed taxonomy")
    if len(ev.strip()) < c.min_evidence_chars or _fold(ev) not in _fold(t.text):
        return AuditLabel(t.id, None, None, ev, rat, status="rejected", why="the evidence is not a verbatim quote of the trace")
    return AuditLabel(t.id, cause, sc, ev, rat)


@dataclass
class AuditReport:
    labels: dict[str, AuditLabel]  # real traces only
    seeded: list[dict]  # {"id", "expected", "got", "ok"}
    mix: dict[str, Frac]
    fp_candidates: list[str]
    fn_candidates: list[str]
    failing: int  # real failing traces
    abstained: list[str]
    config: AuditConfig = field(default_factory=AuditConfig)
    model: str = ""

    @property
    def seeded_recall(self) -> Frac:
        return Frac(sum(1 for s in self.seeded if s["ok"]), len(self.seeded))

    @property
    def auditor_trusted(self) -> bool | None:
        """None without seeded controls (untested, so not trusted for a gate)."""
        r = self.seeded_recall
        return None if not r.n else (r.rate or 0.0) >= self.config.seeded_recall_min

    @property
    def skill_gap_share(self) -> Frac:
        return self.mix.get("skill_gap", Frac(0, sum(f.k for f in self.mix.values())))

    @property
    def send_back(self) -> dict[str, list[str]]:
        """Owner -> trace ids whose cause sends the task back (World bug -> world, spec defect -> spec)."""
        out: dict[str, list[str]] = {}
        for tid, lab in self.labels.items():
            if lab.cause in self.config.send_back_causes:
                out.setdefault(OWNER_OF[lab.cause], []).append(tid)
        return out

    def reasons(self) -> list[str]:
        out = []
        if self.auditor_trusted is False:
            out.append("the auditor mislabelled seeded controls: " + "; ".join(f"{s['id']}: expected {s['expected']}, got {s['got']}"
                                                                              for s in self.seeded if not s["ok"]))
        elif self.auditor_trusted is None:
            out.append("no seeded controls: the auditor is untested")
        if self.fp_candidates:
            out.append(f"false-pass candidates: {self.fp_candidates}")
        if self.fn_candidates:
            out.append(f"false-fail candidates: {self.fn_candidates}")
        for owner, ids in sorted(self.send_back.items()):
            out.append(f"back to {owner}: {ids}")
        sg = self.skill_gap_share
        if sg.n and (sg.rate or 0.0) < self.config.skill_gap_min:
            out.append(f"skill-gap share {sg.k}/{sg.n} < {self.config.skill_gap_min}: hard for the wrong reason")
        return out

    @property
    def ok(self) -> bool:
        return not self.reasons()

    def to_dict(self) -> dict:
        return {"model": self.model, "failing": self.failing, "mix": {k: v.to_dict() for k, v in self.mix.items()},
                "fp_candidates": self.fp_candidates, "fn_candidates": self.fn_candidates, "abstained": self.abstained,
                "seeded": self.seeded, "auditor_trusted": self.auditor_trusted, "send_back": self.send_back, "ok": self.ok, "reasons": self.reasons(),
                "labels": {k: v.__dict__ for k, v in self.labels.items()}}


def _opaque(i: int, salt: str) -> str:
    return "t" + hashlib.sha256(f"{salt}:{i}".encode()).hexdigest()[:10]


async def audit(traces: Sequence[Trace], auditor: Auditor, *, prompt: str, requirements: Sequence[Any] = (), seeded: Sequence[SeededCase] = (),
                salt: str = "audit") -> AuditReport:
    """Label every trace (and the seeded controls, under opaque ids, in a fixed shuffled order) and summarise. Raises
    ``ValueError`` when a trace's policy shares the auditor's model family (``AuditConfig.forbid_same_family``)."""
    cfg = auditor.config
    if cfg.forbid_same_family:
        bad = cross_family_problems(auditor.model, [t.policy for t in traces])
        if bad:
            raise ValueError("same-family judging is forbidden: " + "; ".join(bad))
    batch: list[tuple[str, Trace, SeededCase | None]] = [(t.id, t, None) for t in traces]
    for j, s in enumerate(seeded):
        oid = _opaque(j, salt)
        batch.append((oid, Trace(oid, s.trace.text, s.trace.verifier_pass, s.trace.policy, s.trace.failing_checks, s.trace.meta), s))
    batch.sort(key=lambda x: hashlib.sha256(f"{salt}:{x[0]}".encode()).hexdigest())
    labels: dict[str, AuditLabel] = {}
    seeded_rows: list[dict] = []
    sem = asyncio.Semaphore(max(1, cfg.concurrency))

    async def judge(t: Trace, s: SeededCase | None) -> AuditLabel:
        own = s is not None and s.prompt is not None
        async with sem:
            return await auditor.label(t, prompt=s.prompt if own else prompt,  # type: ignore[union-attr]
                                       requirements=(s.requirements or []) if own else requirements)  # type: ignore[union-attr]

    judged = await asyncio.gather(*[judge(t, s) for _, t, s in batch])
    for (oid, t, s), lab in zip(batch, judged):
        if s is not None:
            ok = lab.cause == s.cause and (s.solution_correct is None or lab.solution_correct == s.solution_correct)
            seeded_rows.append({"id": s.trace.id, "expected": s.cause, "got": lab.cause, "status": lab.status, "ok": ok})
        else:
            labels[oid] = lab
    failing = [t for t in traces if t.verifier_pass is not True]
    counted = [labels[t.id] for t in failing if labels[t.id].cause is not None]
    total = len(counted)
    mix = {c: Frac(sum(1 for lab in counted if lab.cause == c), total) for c in CAUSES}
    fp = sorted(t.id for t in traces if t.verifier_pass is True and labels[t.id].solution_correct is False)
    fn = sorted(t.id for t in traces if t.verifier_pass is False and (labels[t.id].solution_correct is True or labels[t.id].cause == "verifier_defect"))
    abst = sorted(t.id for t in failing if labels[t.id].cause is None)
    return AuditReport(labels, seeded_rows, mix, fp, fn, len(failing), abst, cfg, auditor.model)


def human_sample(trace_ids: Sequence[str], *, share: float = 0.10, salt: str = "human") -> list[str]:
    """The ids a human labels (05 §4: 10%): at least one, chosen by a fixed hash order (reproducible, not cherry-picked)."""
    if not trace_ids:
        return []
    n = max(1, math.ceil(round(share * len(trace_ids), 9)))
    return sorted(sorted(trace_ids, key=lambda i: hashlib.sha256(f"{salt}:{i}".encode()).hexdigest())[:n])


# ---- P3: the auditor is audited (swarm-pipeline/00 §8.4) -------------------------------------------------------------------
def policy_model(policy: str) -> str:
    """The model id inside a policy label: ``solver:qwen/qwen3.7-flash@route`` -> ``qwen/qwen3.7-flash``."""
    p = str(policy or "").strip()
    if ":" in p and "/" in p.split(":", 1)[1] and "/" not in p.split(":", 1)[0]:
        p = p.split(":", 1)[1]
    return p.split("@", 1)[0]


def _family(model: str) -> str:
    from crucible.factory.models import family_of

    return family_of(policy_model(model)) if model else ""


def cross_family_problems(auditor_model: str, policies: Sequence[str]) -> list[str]:
    """One problem per policy whose model family equals the auditor's (empty policies and seeded ``control`` traces are not
    a model). No auditor model: nothing to check."""
    fam = _family(auditor_model)
    if not fam:
        return []
    seen = sorted({p for p in policies if p and p != "control" and _family(p) == fam})
    return [f"policy {p} and auditor {auditor_model} are both family {fam!r}" for p in seen]


def split_by_family(traces: Sequence[Trace], auditor_model: str) -> tuple[list[Trace], list[Trace]]:
    """(traces this auditor may judge, traces it may not: same family)."""
    fam = _family(auditor_model)
    ok, refused = [], []
    for t in traces:
        (refused if fam and t.policy and t.policy != "control" and _family(t.policy) == fam else ok).append(t)
    return ok, refused


def human_queue(report: AuditReport, path: Any, *, share: float | None = None, salt: str = "human") -> list[dict]:
    """The human kappa sample of a batch (``human_sample`` over the real traces), appended to ``path`` (JSONL): trace id, the
    auditor's cause, its quote and ``human_cause: null`` for the human to fill. Returns the rows."""
    from pathlib import Path

    ids = human_sample(sorted(report.labels), share=report.config.human_share if share is None else share, salt=salt)
    rows = [{"trace_id": i, "auditor_cause": report.labels[i].cause, "auditor_correct": report.labels[i].solution_correct,
             "evidence": report.labels[i].evidence, "status": report.labels[i].status, "human_cause": None, "model": report.model} for i in ids]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


def kappa_vs_human(labels: dict[str, AuditLabel], human: dict[str, str | None]) -> dict:
    """Cohen's kappa between the auditor's cause and the human's on the items both labelled (``None`` human labels are
    pending). Below 0.6 the auditor's causes do not count (00 §5.7)."""
    from crucible.factory.labels import cohen_kappa

    items = sorted(i for i, h in human.items() if h is not None and i in labels)
    a = [str(labels[i].cause) for i in items]
    b = [str(human[i]) for i in items]
    k = cohen_kappa(a, b) if items else None
    return {"n": len(items), "pending": sum(1 for h in human.values() if h is None), "kappa": k,
            "agree": sum(x == y for x, y in zip(a, b)), "counts": k is not None and k >= 0.6}
