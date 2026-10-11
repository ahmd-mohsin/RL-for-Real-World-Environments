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
from pathlib import Path
from typing import Any

from crucible.actors.policy import EpisodeLimits, run_episode
from crucible.core.schema.run import Failure
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.infra import LimitPolicy, StageFailure, boundary, classify_verifier, failure
from crucible.world.custody import CustodyViolation
from crucible.world.verifier import hand_off, verifier_world, wants_separate

LIMIT_STOPS = {"max_steps", "timeout", "context_overflow", "too_many_errors"}


def save_episode(attempt: Attempt, ep: Any, res: HarnessResult, harness_id: str, policies: dict[str, Any] | None = None) -> None:
    """The episode's evidence in the attempt dir: the World trace, the model calls, and one joined ATIF trajectory per
    policy actor (``interop.harbor.trajectories``)."""
    from crucible.core.registry import load_impl

    to_atif = load_impl("crucible.interop.harbor.trajectories:to_atif")  # an interop exporter: runtime never imports Harbor

    (attempt.dir / "trace.json").write_text(ep.trace.model_dump_json())
    res.artifacts["trace"] = "trace.json"
    res.calls = {a: list(cs) for a, cs in (ep.model_calls or {}).items()}
    res.segments = {a: list(getattr(p, "segments", [])) for a, p in (policies or {}).items() if getattr(p, "segments", None)}
    res.trajectory_tokens = {a: int(getattr(p, "trajectory_tokens", 0)) for a, p in (policies or {}).items() if getattr(p, "trajectory_tokens", 0)}
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
    """``checkpoint_every`` > 0: a mid-episode checkpoint every N steps, and a masked attempt's retry resumes from the
    previous attempt's last checkpoint (``runtime.checkpoint``). A task with ``subgoals`` gets a dense progress stream
    (``progress.jsonl``, ``verify.progress``); the trace always streams to ``trace.jsonl``."""

    def __init__(self, name: str = "native:llm-tool", version: str = "1", *, limits: EpisodeLimits | None = None, limit_policy: LimitPolicy | None = None,
                 checkpoint_every: int = 0, progress_full_every: int = 250):
        self.id = f"{name}@{version}"
        self.limits = limits or EpisodeLimits(max_steps=30)
        self.limit_policy = limit_policy or LimitPolicy()
        self.checkpoint_every, self.progress_full_every = checkpoint_every, progress_full_every

    def _resumable(self, attempt: Attempt) -> tuple[Path, dict] | None:
        from crucible.runtime import checkpoint

        if not self.checkpoint_every or attempt.n <= 1:
            return None
        prev = attempt.dir.parent / f"a{attempt.n - 1}"
        rec = prev / "record.json"
        ck = checkpoint.load(prev)
        if ck is None or not rec.exists():
            return None
        r = json.loads(rec.read_text())
        return (prev, ck) if r.get("mask_sample") and (r.get("failure") or {}).get("stage") == "agent" else None

    async def run(self, attempt: Attempt) -> HarnessResult:
        from crucible.runtime import checkpoint

        t: dict[str, float] = {}
        world = None
        spend0 = attempt.policy.spend()
        res = HarnessResult()
        try:
            t0 = time.monotonic()
            resume = self._resumable(attempt)
            with boundary("seed"):
                # a resumed episode reopens the previous attempt's work dir: the checkpoint's snapshot blobs live in its store
                world = await attempt.task.open((resume[0] if resume else attempt.dir) / "work")
                world.stream_trace(attempt.dir / "trace.jsonl")
            t["seed"] = time.monotonic() - t0
            t0 = time.monotonic()
            with boundary("agent"):
                policies = attempt.policy.make(attempt)
                for p in policies.values():
                    if hasattr(p, "begin_episode"):
                        p.begin_episode(attempt.id)  # per-attempt cache key: k samples never replay one completion
                observers = []
                if getattr(attempt.task, "subgoals", None) is not None:
                    from crucible.verify.progress import ProgressTracker

                    observers.append(ProgressTracker(attempt.task.subgoals, out=attempt.dir / "progress.jsonl", full_every=self.progress_full_every))
                start = 0
                if resume:
                    start = await checkpoint.resume(world, policies, observers, resume[1], prev_dir=resume[0], new_dir=attempt.dir)
                    res.artifacts["resumed_from"] = f"{resume[0].name}@step{start}"
                ckpt = checkpoint.Checkpointer(attempt.dir, world, policies, observers, every=self.checkpoint_every) if self.checkpoint_every else None
                ep = await run_episode(world, policies, limits=self.limits, observers=observers, reset=resume is None, start_steps=start,
                                       after_step=ckpt)
                if observers:
                    res.grade["progress"] = ep.observed[0]
            t["agent"] = time.monotonic() - t0
            res.steps, res.stop, res.truncated = ep.steps, ep.stop_condition, ep.truncated or ep.stop_condition in LIMIT_STOPS
            from crucible.credit.hack_detector import rule_flags
            from crucible.credit.shaping import format_errors_by_call

            trace = world.to_trace()
            res.flags = rule_flags(trace)  # m13 C-4 level 1 (intercepts, tampering rules)
            res.format_errors = format_errors_by_call(trace, ep.model_calls)  # m13 C-8: charged to the issuing call's tokens
            save_episode(attempt, ep, res, self.id, policies)
            if ep.masked:
                f = next((x for x in ep.failures if x.get("kind")), {"kind": "agent_run_error", "reason": "episode masked"})
                raise StageFailure(failure(f["kind"], "agent", f.get("reason", "")))
            broken = await attempt.task.broken(world) if hasattr(attempt.task, "broken") else []
            if broken:  # the policy broke its environment: its own outcome, scored 0, never masked
                raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="agent", reason="; ".join(broken)[:400]))
            if ep.stop_condition in LIMIT_STOPS and self.limit_policy.for_stop(ep.stop_condition) == "mask":
                raise StageFailure(Failure(kind="agent_timeout", stage="agent", reason=ep.stop_condition, mask=True, blame="harness"))
            t0 = time.monotonic()
            (attempt.dir / "verifier").mkdir(exist_ok=True)
            if wants_separate(world):  # M1: grade a restored copy in a fresh World; the policy's World is closed first
                with boundary("verification"):
                    try:
                        final, handoff = await hand_off(world, artifacts_dir=attempt.dir / "artifacts")
                    except CustodyViolation as e:  # the policy handed in what the verifier must not touch: scored 0, never masked
                        (attempt.dir / "verifier" / "custody.json").write_text(json.dumps({"problems": e.problems}, indent=1))
                        raise StageFailure(Failure(kind="crucible:policy_misbehavior", stage="verification", reason=str(e)[:400], blame="policy", mask=False)) from e
                (attempt.dir / "verifier" / "handoff.json").write_text(json.dumps(handoff.to_dict(), indent=1))
                with boundary("verification"):
                    async with verifier_world(world, final, workdir=attempt.dir / "verifier" / "world") as vworld:
                        rep = await attempt.task.grade(vworld, attempt.dir / "grade")
                res.artifacts["handoff"] = "verifier/handoff.json"
            else:
                with boundary("verification"):
                    rep = await attempt.task.grade(world, attempt.dir / "grade")
            t["verification"] = time.monotonic() - t0
            (attempt.dir / "verifier" / "report.json").write_text(json.dumps(rep.to_dict(), default=str, indent=1))
            res.grade = {**res.grade, "reward": rep.reward, "gates": rep.gate_bits(), "infra": rep.infra}
            if rep.infra.get("status") == "masked":
                raise StageFailure(classify_verifier(rep.infra))
            res.reward, res.gates = rep.reward, rep.gate_bits()
            pen = self.limit_policy.penalty(ep.stop_condition) if ep.stop_condition in LIMIT_STOPS else None
            if pen is not None:  # m13 C-8 (A2-24): an incomplete trajectory gets the configured penalty, not its partial grade
                res.grade["limit_penalty"] = pen
                res.reward = pen
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
            own = [c for cs in (res.calls or {}).values() for c in cs if c.get("cost_usd") is not None]
            if own:  # the attempt's own model calls: a client shared by concurrent attempts makes the spend delta count theirs too
                res.cost_usd = round(sum(float(c["cost_usd"]) for c in own), 9)
            res.timings = {k: round(v, 3) for k, v in t.items()}
        return res
