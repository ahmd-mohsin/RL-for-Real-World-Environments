"""Binding of ``crucible.synth``'s P2 learnability and K frontier to Tinker and the debug lane (synth never imports factory).

- **The student** (``StudentWire.probe``): ``synth.learn.AgentStudentProbe`` over the target student
  (``factory.models.TARGET_STUDENT`` = Qwen3.6-35B-A3B on Tinker, thinking off as the bridge's default) driven through
  ``synth.falsify.agents.llm_agent`` with ``learn.STUDENT`` (the ALT solver's view and protocol, at the rollout
  temperature). Every rollout is tagged ``student:<task>@<v>:<i>`` and its measured cost is read back from
  ``usage.jsonl`` (``cost_of``), so each band decision carries its cost. The student is the only role here that may be
  Qwen; ``check_families`` refuses a Qwen proxy and a non-Qwen "student".
- **Non-student proxies** (``StudentWire.proxy_probe``, ``proxy_client``): the same probe over a proxy model of another
  family (default Nemotron-3.5-Lightning-30B-A3B, the student's size class), or one ``DIFFICULTY_PROXY`` call per task;
  both are predictor features, never decisions.
- **One hard cap**: one ``SpendMeter(cap_usd, path=<run>/spend.jsonl)`` serves the student and the proxies; a call that
  would pass it is refused before it is sent. ``P2_CAP_USD`` is the lead's cap for this work (2026-10-29: $15, student and
  proxies together); ``work_spent(root, "perfect")`` sums the work's receipts (every run path carries "perfect").
- **The debug lane** (``attribution_from_finding``, ``band_from_keep_rule``): the lane's Φ-checkpoint failure findings
  (``factory.failures.FailureFinding``: first wrong step, mode, routing) become ``synth.frontier.Attribution`` records
  (``lane = "debug"``; routed modes stay routed; the lane's own K-knobs travel in ``detail``), and the lane's keep rule
  (``targeting.KeepRule``: train band [0.1, 0.6]) becomes a ``synth.learn.BandSpec``, so one frontier and one band serve
  both lanes.

The Tinker key is read by the SDK from the process environment (the caller sources the repo's ``.env`` into the
subprocess); it is never passed, stored or printed here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from crucible.factory.models import TARGET_STUDENT, SameFamilyError, backend, parse_spec
from crucible.factory.synth_wire import Usage, WireClient
from crucible.synth.falsify.agents import AgentBudget, llm_agent
from crucible.synth.falsify.base import FalsifyContext
from crucible.synth.frontier import Attribution
from crucible.synth.learn import STUDENT, AgentStudentProbe, BandSpec, ContextPool

P2_CAP_USD = 15.0  # the lead's cap for the P2/P3/P4/K pilot (2026-10-29): student rollouts and proxies together
WORK_TAG = "perfect"
NEMOTRON_LIGHTNING = "tinker:nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
DEFAULT_PROXY = NEMOTRON_LIGHTNING


def check_families(student: str, proxies: tuple[str, ...]) -> None:
    s = parse_spec(student)
    probs = []
    if s.family != "qwen":
        probs.append(f"the student must be the Qwen target, got {s.model} ({s.family})")
    for p in proxies:
        f = parse_spec(p).family
        if f == s.family:
            probs.append(f"proxy {p} is the student's family ({f}): a proxy must be a non-student model")
    if probs:
        raise SameFamilyError("; ".join(probs))


class StudentWire:
    """Clients for the student and the proxies on one run's meter (module docstring)."""

    def __init__(self, run_dir: str | Path, *, cap_usd: float, student: str = TARGET_STUDENT, proxy: str = DEFAULT_PROXY, make: Any = None):
        from crucible.train.bridges.tinker.spend import SpendMeter

        if cap_usd > P2_CAP_USD:
            raise ValueError(f"cap {cap_usd} is above the work's cap {P2_CAP_USD}")
        check_families(student, (proxy,))
        self.dir = Path(run_dir)
        self.student_spec, self.proxy_spec = parse_spec(student), parse_spec(proxy)
        self.meter = SpendMeter(cap_usd, path=self.dir / "spend.jsonl")
        self.usage = Usage(self.dir / "usage.jsonl")
        self._make = make  # tests inject a fake maker(model, role) -> chat client
        self._clients: dict[str, WireClient] = {}

    def _client(self, which: str) -> WireClient:
        if which not in self._clients:
            spec = self.student_spec if which == "student" else self.proxy_spec
            inner = self._make(spec.model, which) if self._make is not None else backend(spec.provider).make(spec.model, role=which, meter=self.meter)
            self._clients[which] = WireClient(inner, role=which, model=spec.model, family=spec.family, usage=self.usage, seed_by_tag=True)
        return self._clients[which]

    def student_client(self) -> WireClient:
        return self._client("student")

    def proxy_client(self) -> WireClient:
        return self._client("proxy")

    def cost_of(self, tag: str) -> float:
        return round(sum(float(r.get("usd") or 0.0) for r in self.usage.rows if r.get("tag") == tag), 7)

    def _probe(self, which: str, make_ctx: Callable[[], FalsifyContext], *, budget: AgentBudget, temperature: float, concurrency: int,
               checkpoint: str, simulator: str, time_mode: str) -> AgentStudentProbe:
        client = self._client(which)
        role = STUDENT.model_copy(update={"temperature": temperature})
        spec = self.student_spec if which == "student" else self.proxy_spec

        def runner(tag: str) -> Any:
            return llm_agent(client, role, budget=budget, tag=tag if which == "student" else tag.replace("student:", "proxy:", 1))

        def cost(tag: str) -> float:
            return self.cost_of(tag if which == "student" else tag.replace("student:", "proxy:", 1))

        return AgentStudentProbe(runner=runner, contexts=ContextPool(make_ctx, size=concurrency), student=spec.model, checkpoint=checkpoint,
                                 simulator=simulator, time_mode=time_mode, family=spec.family, cost_of=cost, concurrency=concurrency)

    def probe(self, make_ctx: Callable[[], FalsifyContext], *, budget: AgentBudget | None = None, temperature: float = 1.0, concurrency: int = 4,
              checkpoint: str = "base", time_mode: str = "frozen") -> AgentStudentProbe:
        b = budget or AgentBudget(max_calls=20, max_turns=24, obs_chars=2000)
        return self._probe("student", make_ctx, budget=b, temperature=temperature, concurrency=concurrency, checkpoint=checkpoint,
                           simulator=f"json-agent/v2:calls={b.max_calls}:T={temperature}:seed=tag", time_mode=time_mode)

    def proxy_probe(self, make_ctx: Callable[[], FalsifyContext], *, budget: AgentBudget | None = None, temperature: float = 1.0, concurrency: int = 4
                    ) -> AgentStudentProbe:
        b = budget or AgentBudget(max_calls=20, max_turns=24, obs_chars=2000)
        return self._probe("proxy", make_ctx, budget=b, temperature=temperature, concurrency=concurrency, checkpoint="base",
                           simulator=f"json-agent/v2:calls={b.max_calls}:T={temperature}:seed=tag", time_mode="frozen")

    def cost(self) -> dict[str, Any]:
        return {"total_usd": round(self.meter.spent, 5), "cap_usd": self.meter.cap_usd, "by_role": self.usage.by("role"), "by_model": self.usage.by("model"),
                "calls": len(self.usage.rows), "refused": self.meter.refused}


# ---------------------------------------------------------------------------------------------------------- the debug lane

# the lane's failure modes (factory.failures.MODES) -> the synth frontier's (kind, mode)
DEBUG_MODES: dict[str, tuple[str, str]] = {
    "lost_progress": ("irreversible", "frame_discipline"),  # Φ fell and never recovered: an edit broke earlier work
    "stalled": ("omission", "completion"),
    "never_progressed": ("no_action", "acting"),
    "no_edit": ("no_action", "acting"),
    "claimed_done_without_progress": ("omission", "completion"),
}


def attribution_from_finding(f: Any, *, cell: str = "", checkpoint: str = "base") -> Attribution:
    """A debug-lane ``FailureFinding`` as a frontier ``Attribution`` (module docstring)."""
    from crucible.factory.difficulty import MODE_KNOBS

    fws = dict(getattr(f, "first_wrong_step", {}) or {})
    routed = f.routed
    kind, mode = DEBUG_MODES.get(f.mode, ("omission", f.mode))
    k: Any = kind
    if routed:
        k = {"harness": "protocol", "limits": "budget"}.get(routed, "unreplayable")
    a = Attribution(task_id=f.task, version=1, domain="debug", rollout=0, student=f.student, checkpoint=checkpoint, kind=k,
                    mode=None if routed else mode, step=fws.get("step"), call={"files": fws.get("files", [])} if fws.get("files") else None,
                    patterns=list(f.milestones_missed), reasons=[f"cause:{f.cause}"] if f.cause else [], cell=cell or (f.skill or ""),
                    detail=f"debug mode {f.mode}; first wrong step {fws.get('kind')}; lane knobs {list(MODE_KNOBS.get(f.mode, ()))}", lane="debug")
    from crucible.synth.canon import content_hash

    a.id = "at-" + content_hash({"lane": "debug", "f": f.id}, 16)
    a.routed = {"protocol": "harness", "budget": "limits", "unreplayable": routed or "world"}.get(a.kind) if routed else None
    return a


def band_from_keep_rule(rule: Any = None, *, looks: tuple[int, ...] = (4, 8), alpha: float = 0.1) -> BandSpec:
    from crucible.factory.targeting import KeepRule

    r = rule or KeepRule()
    return BandSpec(lo=r.train_lo, hi=r.train_hi, looks=looks, alpha=alpha)


def alt_history(cegis: Mapping[str, Any]) -> tuple[int, int] | None:
    """(passes, n) of the certification's ALT solvers from a stored ``cegis.json`` (free non-student proxy rollouts)."""
    cert = cegis.get("certificate") or {}
    cls = (cert.get("classes") or {}).get("ALT") if isinstance(cert.get("classes"), dict) else None
    if isinstance(cls, dict) and cls.get("n"):
        return int(cls.get("passes") or 0), int(cls["n"])
    return None
