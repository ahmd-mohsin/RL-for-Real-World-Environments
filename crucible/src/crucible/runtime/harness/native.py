"""The native harness (seam W): Crucible's own policies (scripted, ``LLMToolPolicy``, ``BashLoopPolicy``) in a World,
graded by the task's M7 plan over the verifier view.

A ``RunTask`` adapter (packs provide them) opens a fresh, reset World for an attempt and grades it::

    class RunTask(Protocol):
        id: str
        async def open(self, workdir: Path) -> World          # started + reset, responders attached
        async def grade(self, world: World, workdir: Path) -> GradeReport

A ``PolicyRef`` builds the policy for an attempt (``make(attempt) -> {actor: Policy}``) and reports its spend.

Phases run under ``infra.boundary``: seed (open) -> agent (run_episode) -> verification (grade) -> cleanup (close, which
can never override a result). An episode the World masked (a tool crash, an exposed answer, a provider outage) becomes a
masked failure; a limit stop follows the limit policy; a grade whose report is masked (``check_infra``,
``grader_exception``) becomes ``verifier_error`` (task blame, masked).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from crucible.actors.policy import EpisodeLimits, run_episode
from crucible.core.schema.run import Failure
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.infra import LimitPolicy, StageFailure, boundary, classify_verifier, failure

LIMIT_STOPS = {"max_steps", "timeout", "context_overflow", "too_many_errors"}


def save_episode(attempt: Attempt, ep: Any, res: HarnessResult, harness_id: str) -> None:
    """The episode's evidence in the attempt dir: the World trace, the model calls, and one joined ATIF trajectory per
    policy actor (``runtime.trajectories``)."""
    from crucible.runtime.trajectories import to_atif

    (attempt.dir / "trace.json").write_text(ep.trace.model_dump_json())
    res.artifacts["trace"] = "trace.json"
    res.calls = {a: list(cs) for a, cs in (ep.model_calls or {}).items()}
    if any(res.calls.values()):
        (attempt.dir / "calls.json").write_text(json.dumps(res.calls, default=str))
        res.artifacts["calls"] = "calls.json"
    for actor, cs in res.calls.items():
        model = next((c.get("model") for c in cs if c.get("model")), None)
        traj = to_atif(ep.trace, actor, calls=cs, instruction=getattr(attempt.task, "instruction", None), harness=harness_id,
                       model_name=model, session_id=attempt.id)
        (attempt.dir / f"trajectory.{actor}.json").write_text(traj.model_dump_json(exclude_none=True))
        res.artifacts[f"trajectory.{actor}"] = f"trajectory.{actor}.json"


@dataclass
class PolicyRef:
    id: str
    make: Callable[[Attempt], dict[str, Any]]  # attempt -> {actor_id: Policy}
    spend: Callable[[], tuple[float, float]] = lambda: (0.0, 0.0)  # (usd, uncertain usd) observed so far
    estimate_usd: float = 0.0


class NativeHarness:
    def __init__(self, name: str = "native:llm-tool", version: str = "1", *, limits: EpisodeLimits | None = None, limit_policy: LimitPolicy | None = None):
        self.id = f"{name}@{version}"
        self.limits = limits or EpisodeLimits(max_steps=30)
        self.limit_policy = limit_policy or LimitPolicy()

    async def run(self, attempt: Attempt) -> HarnessResult:
        t: dict[str, float] = {}
        world = None
        spend0 = attempt.policy.spend()
        res = HarnessResult()
        try:
            t0 = time.monotonic()
            with boundary("seed"):
                world = await attempt.task.open(attempt.dir / "work")
            t["seed"] = time.monotonic() - t0
            t0 = time.monotonic()
            with boundary("agent"):
                policies = attempt.policy.make(attempt)
                for p in policies.values():
                    if hasattr(p, "begin_episode"):
                        p.begin_episode(attempt.id)  # per-attempt cache key: k samples never replay one completion
                ep = await run_episode(world, policies, limits=self.limits)
            t["agent"] = time.monotonic() - t0
            res.steps, res.stop, res.truncated = ep.steps, ep.stop_condition, ep.truncated or ep.stop_condition in LIMIT_STOPS
            save_episode(attempt, ep, res, self.id)
            if ep.masked:
                f = next((x for x in ep.failures if x.get("kind")), {"kind": "agent_run_error", "reason": "episode masked"})
                raise StageFailure(failure(f["kind"], "agent", f.get("reason", "")))
            broken = await attempt.task.broken(world) if hasattr(attempt.task, "broken") else []
            if broken:  # the policy broke its environment: its own outcome, scored 0, never masked
                raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="agent", reason="; ".join(broken)[:400]))
            if ep.stop_condition in LIMIT_STOPS and self.limit_policy.for_stop(ep.stop_condition) == "mask":
                raise StageFailure(Failure(kind="agent_timeout", stage="agent", reason=ep.stop_condition, mask=True, blame="harness"))
            t0 = time.monotonic()
            with boundary("verification"):
                rep = await attempt.task.grade(world, attempt.dir / "grade")
            t["verification"] = time.monotonic() - t0
            (attempt.dir / "verifier").mkdir(exist_ok=True)
            (attempt.dir / "verifier" / "report.json").write_text(json.dumps(rep.to_dict(), default=str, indent=1))
            res.grade = {"reward": rep.reward, "gates": rep.gate_bits(), "infra": rep.infra}
            if rep.infra.get("status") == "masked":
                raise StageFailure(classify_verifier(rep.infra))
            res.reward, res.gates = rep.reward, rep.gate_bits()
        except StageFailure as sf:
            res.failure = sf.failure
        finally:
            t0 = time.monotonic()
            if world is not None:
                try:
                    await world.close()
                except Exception:  # noqa: BLE001, S110 -- cleanup never overrides a result
                    pass
            t["cleanup"] = time.monotonic() - t0
            usd, unc = attempt.policy.spend()
            res.cost_usd, res.cost_uncertain_usd = max(0.0, usd - spend0[0]), max(0.0, unc - spend0[1])
            res.timings = {k: round(v, 3) for k, v in t.items()}
        return res
