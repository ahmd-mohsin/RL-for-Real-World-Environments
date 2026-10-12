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

import dataclasses
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


@dataclass
class TaskStep:
    """One step of a multi-step task (M1 item 4; Harbor ``StepConfig``). ``min_reward``: a float gates the step's
    ``reward``; a dict gates each key of its flat reward (``reward``, ``outcome``, ``gate/<g>``, ``check/<c>``), a
    missing key counting as below. ``setup(world)`` runs before the step opens (new records arrive, a file changes)."""

    name: str
    instruction: str | dict[str, str]
    grade: Callable[[Any, Path], Any]
    min_reward: float | dict[str, float] | None = None
    max_steps: int | None = None
    fresh_context: bool = False
    setup: Callable[[Any], Any] | None = None
    healthcheck: tuple[str, Any] | None = None  # (app id, HealthcheckSpec) checked after setup, before the step opens


def _below(flat: dict[str, float], gate: float | dict[str, float] | None) -> str | None:
    """Why a step's flat reward fails its gate, or None (Harbor ``_min_reward_failure``)."""
    if gate is None:
        return None
    for key, threshold in ({"reward": gate} if isinstance(gate, int | float) else gate).items():
        actual = flat.get(key, float("-inf"))
        if actual < threshold:
            return f"{key}={actual} below min_reward {threshold}"
    return None


def save_episode(attempt: Attempt, ep: Any, res: HarnessResult, harness_id: str, policies: dict[str, Any] | None = None) -> None:
    """The episode's evidence in the attempt dir: the World trace, the model calls, and one joined ATIF trajectory per
    policy actor (``interop.harbor.trajectories``)."""
    from crucible.core.registry import load_impl
    from crucible.core.secrets import scrub_json  # records leave the process (M10 audit V12)

    to_atif = load_impl("crucible.interop.harbor.trajectories:to_atif")  # an interop exporter: runtime never imports Harbor

    (attempt.dir / "trace.json").write_text(scrub_json(ep.trace.model_dump_json()))
    res.artifacts["trace"] = "trace.json"
    res.calls = {a: list(cs) for a, cs in (ep.model_calls or {}).items()}
    res.segments = {a: list(getattr(p, "segments", [])) for a, p in (policies or {}).items() if getattr(p, "segments", None)}
    res.trajectory_tokens = {a: int(getattr(p, "trajectory_tokens", 0)) for a, p in (policies or {}).items() if getattr(p, "trajectory_tokens", 0)}
    if any(res.calls.values()):
        (attempt.dir / "calls.json").write_text(scrub_json(json.dumps(res.calls, default=str)))
        res.artifacts["calls"] = "calls.json"
    for actor, cs in res.calls.items():
        model = next((c.get("model") for c in cs if c.get("model")), None)
        traj = to_atif(ep.trace, actor, calls=cs, instruction=getattr(attempt.task, "instruction", None), harness=harness_id,
                       model_name=model, session_id=attempt.id)
        (attempt.dir / f"trajectory.{actor}.json").write_text(scrub_json(traj.model_dump_json(exclude_none=True)))
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
        """(the previous attempt's dir, its checkpoint) when the previous attempt was masked mid-episode, or when the
        process died (no record; the journal moved the dir to ``aborted/``: M10 audit V9, the "dies at step 3,000" case
        never resumed). Multi-step tasks are not checkpointed (V14: a resume would reset the World at step 0)."""
        from crucible.runtime import checkpoint

        if not self.checkpoint_every or attempt.n <= 1 or getattr(attempt.task, "steps", None):
            return None
        prev = checkpoint.attempt_path(attempt.dir.parent, f"a{attempt.n - 1}")
        rec = prev / "record.json"
        ck = checkpoint.load(prev)
        if ck is None:
            return None
        if not rec.exists():
            return prev, ck  # a process crash: nothing was recorded, the checkpoint is the last word
        r = json.loads(rec.read_text())
        return (prev, ck) if r.get("mask_sample") and (r.get("failure") or {}).get("stage") == "agent" else None

    async def _grade(self, attempt: Attempt, world: Any, grade_fn: Callable, vdir: Path, gdir: Path, artifacts_dir: Path, res: HarnessResult,
                     *, close_source: bool = True) -> Any:
        """Grade ``world`` with ``grade_fn(world, gdir)``: in place, or (``verifier.mode: separate``) in a fresh World
        restored from a quiesced, custody-checked snapshot. Writes ``vdir/report.json`` (and ``handoff.json``)."""
        vdir.mkdir(parents=True, exist_ok=True)
        if wants_separate(world):  # M1: grade a restored copy in a fresh World; the policy's World is closed first
            with boundary("verification"):
                try:
                    final, handoff = await hand_off(world, artifacts_dir=artifacts_dir)
                except CustodyViolation as e:  # the policy handed in what the verifier must not touch: scored 0, never masked
                    (vdir / "custody.json").write_text(json.dumps({"problems": e.problems}, indent=1))
                    raise StageFailure(Failure(kind="crucible:policy_misbehavior", stage="verification", reason=str(e)[:400], blame="policy", mask=False)) from e
            (vdir / "handoff.json").write_text(json.dumps(handoff.to_dict(), indent=1))
            with boundary("verification"):
                async with verifier_world(world, final, workdir=vdir / "world", close_source=close_source) as vworld:
                    rep = await grade_fn(vworld, gdir)
            res.artifacts[f"handoff:{vdir.relative_to(attempt.dir)}"] = str((vdir / "handoff.json").relative_to(attempt.dir))
        else:
            with boundary("verification"):
                rep = await grade_fn(world, gdir)
        (vdir / "report.json").write_text(json.dumps(rep.to_dict(), default=str, indent=1))
        return rep

    async def _run_steps(self, attempt: Attempt, world: Any, policies: dict[str, Any], res: HarnessResult) -> tuple[Any, dict[str, Any]]:
        """A multi-step task in one World (M1 item 4; Harbor ``trial/multi_step.py``). Each ``TaskStep``: optional setup,
        its instruction opened in the running episode (``World.begin_task_step``), a policy segment (the same policy
        objects, so the same context, unless ``fresh_context``), its own grade, and its ``min_reward`` gate: a step
        below its gate ends the task. Reward (``task.step_reward``): ``mean`` over every declared step (a step never
        reached scores 0: stopping early cannot raise the mean), ``final`` (the last declared step; 0 if not reached),
        or ``mean_executed`` (Harbor's mean over the steps that ran)."""
        from crucible.verify.runtime.plan import flat_reward

        steps = list(attempt.task.steps)
        strategy = getattr(attempt.task, "step_reward", "mean")
        if strategy not in ("mean", "final", "mean_executed"):
            raise StageFailure(Failure(kind="verifier_error", stage="agent", reason=f"unknown step_reward {strategy!r}", blame="task"))
        names = [st.name for st in steps]
        if len(set(names)) != len(names):
            raise StageFailure(Failure(kind="verifier_error", stage="agent", reason=f"duplicate step names {names}", blame="task"))
        results: list[dict[str, Any]] = []
        used = [policies]
        total_steps = 0
        ep = None
        for i, st in enumerate(steps):
            sdir = attempt.dir / "steps" / f"{i:02d}-{st.name}"
            sdir.mkdir(parents=True, exist_ok=True)
            with boundary("agent"):
                if i == 0:
                    await world.reset()
                if st.setup is not None:
                    await st.setup(world)
                if st.healthcheck is not None:
                    await world.check_health(st.healthcheck[0], st.healthcheck[1])
                obs = await world.begin_task_step(st.name, i, st.instruction)
                if i > 0 and st.fresh_context:
                    policies = attempt.policy.make(attempt)
                    used.append(policies)
                limits = dataclasses.replace(self.limits, max_steps=st.max_steps) if st.max_steps else self.limits
                ep = await run_episode(world, policies, limits=limits, reset=False, start_obs=obs, sample_key=attempt.id)
            total_steps += ep.steps
            if ep.masked:
                f = next((x for x in ep.failures if x.get("kind")), {"kind": "agent_run_error", "reason": "episode masked"})
                raise StageFailure(failure(f["kind"], "agent", f"step {st.name}: {f.get('reason', '')}"))
            if any(f.get("kind") == "crucible:policy_misbehavior" for f in ep.failures):
                raise StageFailure(Failure(kind="crucible:policy_misbehavior", stage="agent", reason=f"step {st.name}", blame="policy", mask=False))
            if ep.stop_condition in LIMIT_STOPS and self.limit_policy.for_stop(ep.stop_condition) == "mask":
                raise StageFailure(Failure(kind="agent_timeout", stage="agent", reason=f"step {st.name}: {ep.stop_condition}", mask=True, blame="harness"))
            broken = await attempt.task.broken(world) if hasattr(attempt.task, "broken") else []
            if broken:  # the policy broke its environment: its own outcome, scored 0, never masked
                raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="agent", reason=f"step {st.name}: " + "; ".join(broken)[:400]))
            t0 = time.monotonic()
            rep = await self._grade(attempt, world, st.grade, sdir / "verifier", sdir / "grade", sdir / "artifacts", res,
                                    close_source=i == len(steps) - 1)
            if rep.infra.get("status") == "masked":
                raise StageFailure(classify_verifier(rep.infra))
            flat = flat_reward(rep.reward_json()) or {"reward": rep.reward}
            why = _below(flat, st.min_reward)
            results.append({"name": st.name, "reward": rep.reward, "outcome": rep.outcome, "gates": rep.gate_bits(), "stop": ep.stop_condition,
                            "steps": ep.steps, "passed_gate": why is None, "verification_s": round(time.monotonic() - t0, 3), **({"abort": why} if why else {})})
            if why is not None:
                break
        rewards = [r["reward"] for r in results]
        if strategy == "final":
            reward = rewards[-1] if len(results) == len(steps) else 0.0
        elif strategy == "mean_executed":
            reward = sum(rewards) / len(rewards)
        else:
            reward = sum(rewards) / len(steps)
        res.grade.update({"steps": results, "step_reward": strategy, "reward": round(reward, 6), "completed_steps": len(results),
                          "gates": {f"step/{r['name']}": int(r["passed_gate"]) for r in results}})
        assert ep is not None
        merged: dict[str, list] = {}
        for pols in used:
            for a, p in pols.items():
                merged.setdefault(a, []).extend(getattr(p, "model_calls", []))
        ep = dataclasses.replace(ep, steps=total_steps, model_calls=merged, trace=world.to_trace())
        return ep, policies

    async def run(self, attempt: Attempt) -> HarnessResult:
        from crucible.runtime import checkpoint

        t: dict[str, float] = {}
        world = None
        spend0 = attempt.policy.spend()
        res = HarnessResult()
        restored: set[str] = set()
        policies: dict[str, Any] = {}
        try:
            t0 = time.monotonic()
            resume = self._resumable(attempt)
            work_attempt = attempt.dir.name
            with boundary("seed"):
                # a resumed episode reopens the work dir that holds the World (the checkpoint names it): its snapshot blobs
                # live in that store, however many resumes ago it was first opened
                if resume:
                    work_attempt = resume[1].get("work_attempt") or resume[0].name
                    world = await attempt.task.open(checkpoint.attempt_path(attempt.dir.parent, work_attempt) / "work")
                else:
                    world = await attempt.task.open(attempt.dir / "work")
                world.stream_trace(attempt.dir / "trace.jsonl")
            t["seed"] = time.monotonic() - t0
            t0 = time.monotonic()
            with boundary("agent"):
                policies = attempt.policy.make(attempt)
                observers = []
                if getattr(attempt.task, "subgoals", None) is not None:
                    from crucible.verify.progress import ProgressTracker

                    observers.append(ProgressTracker(attempt.task.subgoals, out=attempt.dir / "progress.jsonl", full_every=self.progress_full_every))
                start = 0
                if resume:
                    start = await checkpoint.resume(world, policies, observers, resume[1], prev_dir=resume[0], new_dir=attempt.dir)
                    res.artifacts["resumed_from"] = f"{resume[0].name}@step{start}"
                    # calls restored from the checkpoint were paid by an earlier attempt (M10 audit V8: re-charged)
                    restored |= {c.get("call_id") for p in policies.values() for c in getattr(p, "model_calls", []) if c.get("call_id")}
                if getattr(attempt.task, "steps", None):  # M1 item 4: a multi-step task (Harbor multi-step trials)
                    ep, policies = await self._run_steps(attempt, world, policies, res)
                else:
                    ckpt = checkpoint.Checkpointer(attempt.dir, world, policies, observers, every=self.checkpoint_every,
                                                   work_attempt=work_attempt) if self.checkpoint_every else None
                    ep = await run_episode(world, policies, limits=self.limits, observers=observers, reset=resume is None, start_steps=start,
                                           after_step=ckpt, sample_key=attempt.id)  # per-attempt cache key: k samples never replay one completion
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
            if getattr(world, "turns", None) is not None and world.turns.rewards:  # per-seat outcomes of a turn-based World (M1 item 7)
                res.grade["seat_rewards"] = dict(world.turns.rewards)
            if ep.masked:
                f = next((x for x in ep.failures if x.get("kind")), {"kind": "agent_run_error", "reason": "episode masked"})
                raise StageFailure(failure(f["kind"], "agent", f.get("reason", "")))
            pe = next((x for x in ep.failures if x.get("kind") == "crucible:policy_error"), None)
            if pe is not None:  # the policy's own code or output raised: its outcome, scored 0 (M10 audit V6)
                raise StageFailure(Failure(kind="crucible:policy_error", stage="agent", reason=pe.get("reason", "")))
            broken = await attempt.task.broken(world) if hasattr(attempt.task, "broken") and res.grade.get("steps") is None else []
            if broken:  # the policy broke its environment: its own outcome, scored 0, never masked
                raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="agent", reason="; ".join(broken)[:400]))
            if ep.stop_condition in LIMIT_STOPS and self.limit_policy.for_stop(ep.stop_condition) == "mask":
                raise StageFailure(Failure(kind="agent_timeout", stage="agent", reason=ep.stop_condition, mask=True, blame="harness"))
            t0 = time.monotonic()
            if res.grade.get("steps") is not None:  # graded step by step inside _run_steps
                t["verification"] = sum(r.get("verification_s", 0.0) for r in res.grade["steps"])
                res.reward, res.gates = res.grade["reward"], res.grade.get("gates", {})
            else:
                rep = await self._grade(attempt, world, attempt.task.grade, attempt.dir / "verifier", attempt.dir / "grade", attempt.dir / "artifacts", res)
                t["verification"] = time.monotonic() - t0
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
            calls = res.calls or {a: list(getattr(p, "model_calls", [])) for a, p in policies.items()}  # a masked attempt paid too (V8)
            own = [c for cs in calls.values() for c in cs if c.get("cost_usd") is not None and c.get("call_id") not in restored]
            if own:  # the attempt's own model calls: a client shared by concurrent attempts makes the spend delta count theirs too
                res.cost_usd = round(sum(float(c["cost_usd"]) for c in own), 9)
            elif restored:
                res.cost_usd = 0.0
            res.timings = {k: round(v, 3) for k, v in t.items()}
        return res
