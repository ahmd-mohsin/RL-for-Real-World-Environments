"""The native harness (seam W): Crucible's own policies (scripted, ``LLMToolPolicy``, ``BashLoopPolicy``) in a World,
graded by the task's M7 plan over the verifier view.

A ``RunTask`` adapter (packs provide them) opens a fresh, reset World for an attempt and grades it::

    class RunTask(Protocol):
        id: str
        async def open(self, workdir: Path) -> World          # started + reset, responders attached
        async def grade(self, world: World, workdir: Path) -> GradeReport
        # optional
        async def infra_cause(self, world: World) -> Failure | None      # host-side: is the sandbox gone? (M13 audit B2)
        async def grade_control(self, workdir: Path) -> GradeReport | None  # M10 lab audit R-3: known-good work graded by the
                                                                             # same grader (oracle, reference answer); None = none
        def identity_config(self) -> dict                                    # M10 lab audit R-4: content digests and configuration
                                                                             # (required unless the task is plain data or stateless)

A ``PolicyRef`` builds the policy for an attempt (``make(attempt) -> {actor: Policy}``) and reports its spend; its
``config`` (M10 lab audit R-4) is the policy's configuration as JSON, which the run's identity needs because ``make`` is a
closure.

Phases run under ``infra.boundary``: seed (open) -> agent (run_episode) -> verification (grade) -> cleanup (close, which
can never override a result). An episode the World masked (a tool crash, an exposed answer, a provider outage) becomes a
masked failure; a limit stop follows the limit policy; a grade whose report is masked (``check_infra``,
``grader_exception``) becomes ``verifier_error`` (task blame, masked).

M10 lab audit R-3: a failed grade is graded again on the same work, never by running the agent again (before R-3 the
Runner's retry of a verification mask re-ran the whole episode here, hidden by deterministic oracles). In place the World
is gone after the attempt, so the grade is retried inside it (``runtime.grading.grade_in_place``, ``grade_retries`` /
``scoring_budget_s``) and the attempt then ends terminal-masked. A separate verifier persists its quiesced hand-off
(``verifier/handoff_ref.json``), and a retry of the attempt (or ``Runner.regrade``) re-grades it in a fresh verifier World
(``regrade``) with the agent attempt's calls, flags, stop and limit penalty; an answer-sensitive failure (a hang or a
checker outage on this answer) is graded again at once over the same snapshot. A multi-step step persists nothing and
is graded again inside the attempt like an in-place grade (a separate-verifier step in a fresh verifier World over the
same snapshot, a contaminated node or a transport failure included: review round 2), unless the task declares
``step_grades_once`` (``HarborMultiStepTask``: R-2's per-step state is used up by the step's grade), whose steps are
graded once.
A grade failure no further grade of this work will answer goes through ``grading.settle``: the policy's World is probed
(``infra_cause``) and closed, and an answer-sensitive failure is decided by the task's control (``grade_control``, own
budget ``control_timeout_s``): ``crucible:policy_error`` / ``grader_failed_on_answer`` scored 0 when the control passes,
a task-blamed ``verifier_error`` mask when it fails, a mask with flag ``no_control`` without one (``step_grades_once``
steps: always; a multi-step task's ``grade_control`` must exercise every step's grader). The harness records on the
``Attempt`` which phase it is in and whether the work is persisted (``Attempt.phase`` / ``Attempt.regradable``), so the
Runner's attempt clock firing during a grade is a verification failure, never an agent re-run. Review round 2: the
verification phase opens the grading's own window on that clock (``grading.grading_window``), so a grade that hangs ends
as ``verifier_timeout`` and is settled, never cut short; the scoring budget counts only the grade, not the verifier
World's start or close (``grading.unbudgeted``); and only an answer-sensitive failure that repeated (``grading.decided``)
is controlled.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from crucible.actors.policy import EpisodeLimits, run_episode
from crucible.core.schema.run import Failure
from crucible.runtime.grading import (
    HANDOFF_REF,
    agent_clock,
    answer_sensitive,
    decided,
    deciding_record,
    grade_in_place,
    grade_summary,
    grading_window,
    liveness,
    regradable,
    seen_before,
    settle,
    unbudgeted,
    write_reports,
)
from crucible.runtime.harness.base import Attempt, HarnessResult, settle_usage
from crucible.runtime.identity import Undeclared, harness_digest
from crucible.runtime.infra import LimitPolicy, StageFailure, boundary, episode_failure
from crucible.runtime.journal import REFUSED
from crucible.world.custody import CustodyViolation
from crucible.world.verifier import hand_off, verifier_world, wants_separate

LIMIT_STOPS = {"max_steps", "timeout", "context_overflow", "too_many_errors"}


def misbehaved(ep: Any, prefix: str = "") -> StageFailure | None:
    """The policy defeated the verification boundary (``run_episode``: its processes survived the final quiesce): its
    outcome, scored 0 and never masked, as ``crucible:policy_misbehavior`` is declared (``core.schema.run``). Harness-reward
    audit 2026-10-28: only the multi-step path enforced it; a single-step episode (native or workspace) went on to be
    graded while the survivors could still change the work being graded or collected."""
    f = next((x for x in ep.failures if x.get("kind") == "crucible:policy_misbehavior"), None)
    if f is None:
        return None
    return StageFailure(Failure(kind="crucible:policy_misbehavior", stage="agent", reason=(prefix + str(f.get("reason", "")))[:500], blame="policy",
                                mask=False))


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


def tool_cards(world: Any, actor: str) -> list[dict[str, Any]]:
    """The tools ``actor`` was offered, as OpenAI function cards (the shape chat templates and TRL take)."""
    try:
        specs = world.tools_for(actor) if world is not None else []
    except (AttributeError, KeyError, TypeError):  # a World without per-actor tools: the trajectory simply carries none
        return []
    return [{"type": "function", "function": {"name": t.qualified, "description": t.description, "parameters": t.input_schema}} for t in specs]


def instruction_for(task: Any, actor: str, world: Any = None) -> str | None:
    """The instruction text an actor was given: the task's own when it is text (or a per-actor dict of texts), else what
    the World rendered at reset. A callable instruction (OpenEnv renders it from the reset observation) is never passed
    on: ATIF wants text (M13 audit B8: every OpenEnv attempt was masked by the validation error and retried)."""
    ins = getattr(task, "instruction", None)
    if isinstance(ins, dict):
        ins = ins.get(actor)
    if isinstance(ins, str):
        return ins
    rendered = (getattr(world, "instructions", None) or {}).get(actor) if world is not None else None
    return rendered if isinstance(rendered, str) else None


def save_episode(attempt: Attempt, ep: Any, res: HarnessResult, harness_id: str, policies: dict[str, Any] | None = None, world: Any = None) -> None:
    """The episode's evidence in the attempt dir: the World trace, the model calls, and one joined ATIF trajectory per
    policy actor (``interop.harbor.trajectories``; skipped, and noted, where Harbor is not installed: an OpenEnv or
    verifiers venv runs episodes without it, M13 audit B8)."""
    from crucible.core.registry import RegistryError, load_impl
    from crucible.core.secrets import scrub_json  # records leave the process (M10 audit V12)

    try:
        to_atif = load_impl("crucible.interop.harbor.trajectories:to_atif")  # an interop exporter: runtime never imports Harbor
    except (ImportError, RegistryError):  # Harbor is not installed here
        to_atif = None

    (attempt.dir / "trace.json").write_text(scrub_json(ep.trace.model_dump_json()))
    res.artifacts["trace"] = "trace.json"
    res.calls = {a: list(cs) for a, cs in (ep.model_calls or {}).items()}
    res.segments = {a: list(getattr(p, "segments", [])) for a, p in (policies or {}).items() if getattr(p, "segments", None)}
    res.trajectory_tokens = {a: int(getattr(p, "trajectory_tokens", 0)) for a, p in (policies or {}).items() if getattr(p, "trajectory_tokens", 0)}
    if any(res.calls.values()):
        (attempt.dir / "calls.json").write_text(scrub_json(json.dumps(res.calls, default=str)))
        res.artifacts["calls"] = "calls.json"
    if to_atif is None:
        if res.calls:
            res.artifacts["trajectory_skipped"] = "Harbor (ATIF) is not installed in this environment"
        return
    for actor, cs in res.calls.items():
        model = next((c.get("model") for c in cs if c.get("model")), None)
        traj = to_atif(ep.trace, actor, calls=cs, instruction=instruction_for(attempt.task, actor, world), harness=harness_id,
                       model_name=model, session_id=attempt.id, tool_definitions=tool_cards(world, actor))
        (attempt.dir / f"trajectory.{actor}.json").write_text(scrub_json(traj.model_dump_json(exclude_none=True)))
        res.artifacts[f"trajectory.{actor}"] = f"trajectory.{actor}.json"


async def _masked_episode(task: Any, world: Any, ep: Any, *, prefix: str = "") -> Failure:
    """The failure of an episode the World masked. A task that can see the infrastructure from the host
    (``infra_cause``: Harbor in place asks Docker whether its container is gone) names the cause; else the World's first
    recorded fault (M13 audit B2: a removed container was a generic tool crash, or worse, the policy's failed command).

    M10 lab audit R-3: the episode's own failure keeps its blame, mask and retryable (``infra.episode_failure``: an
    unclassified policy exception stays unknown blame, never retried and counted by ``Health.unknown``), and the result
    carries its evidence: ``probe`` is what ``infra_cause`` said (``dead`` when it named a cause, ``alive`` when it found
    none, ``unprobed`` without the hook or when asking failed; verifiers v1/harness.py:173-193, probe before blaming the
    sandbox), ``control`` is ``none`` (no grade ran), ``status`` the cause's detail or the episode's stop."""
    cause = None
    probe = "unprobed"
    if hasattr(task, "infra_cause") and world is not None:
        try:
            cause = await task.infra_cause(world)
            probe = "dead" if cause is not None else "alive"
        except Exception:  # noqa: BLE001 -- asking the host is best effort; the World's own fault still masks
            cause = None
    if cause is not None:
        upd: dict[str, Any] = {"evidence": cause.evidence or {"probe": "dead", "control": "none", "status": cause.detail or cause.kind}}
        if prefix:
            upd["reason"] = prefix + cause.reason
        return cause.model_copy(update=upd)
    f = next((x for x in ep.failures if x.get("kind")), {"kind": "agent_run_error", "reason": "episode masked"})
    return episode_failure(f, "agent", prefix).model_copy(update={"evidence": {"probe": probe, "control": "none", "status": ep.stop_condition}})


def _flag_key(f: dict[str, Any]) -> tuple[str, str]:
    return str(f.get("rule")), json.dumps(f.get("evidence"), sort_keys=True, default=str)


def _take_grade_findings(rep: Any, res: HarnessResult, *, prefix: str = "", seen: set[tuple[str, str]] | None = None) -> None:
    """What a grade found about the policy beyond its reward: rule flags (``meta.flags``, e.g. ``grader_tampered`` with
    its evidence) join the rollout's flags, and a policy-blamed outcome in ``meta.failure`` (``crucible:grader_tampered``,
    ``crucible:agent_broke_env``) becomes the attempt's failure: scored 0, never masked, and visible to bench and the
    hack detector (M13 audit B1: the grade's verdict was only a 0 in report.json). ``seen``: the (rule, evidence) of
    flags already taken from an earlier grade of the same work (``_take_findings``), which are not taken again."""
    from crucible.core.schema.run import default_blame, is_failure_kind

    meta = getattr(rep, "meta", None) or {}
    flags = [f for f in meta.get("flags") or [] if isinstance(f, dict) and f.get("rule")]
    if seen is not None:
        flags = [f for f in flags if _flag_key(f) not in seen]
        seen.update(_flag_key(f) for f in flags)
    if flags:
        res.flags = [*res.flags, *flags]
    f = meta.get("failure") or {}
    kind = f.get("kind")
    if rep.infra.get("status") != "masked" and kind and is_failure_kind(kind) and default_blame(kind) == "policy":
        raise StageFailure(Failure(kind=kind, stage="verification", reason=prefix + str(f.get("reason", "")), blame="policy", mask=False))


class EpisodeSidecar(Protocol):
    """An opt-in episode sidecar (M17 checkpoint credit, ``factory.checkpoint.CheckpointSidecar``): ``run_episode``'s observer
    protocol (``on_start``, ``on_step``, ``on_end``; ``on_end``'s dict becomes ``res.grade["progress"]``) plus
    ``before_action``, awaited after a policy chose its action and before the World applies it (the state at a compaction
    boundary). It never feeds the agent: nothing it returns reaches an observation. Runtime defines the seam; the factory
    implements it (runtime never imports the factory)."""

    async def on_start(self, world: Any) -> None: ...
    async def on_step(self, world: Any, step: int, actor: str, action: Any, result: Any) -> Any: ...
    async def before_action(self, world: Any, actor: str, policy: Any, action: Any) -> None: ...
    async def on_end(self, world: Any) -> dict[str, Any]: ...


def episode_sidecar(make: Callable[[Attempt], Any] | None, attempt: Attempt) -> Any:
    """The attempt's sidecar, or None (off: the harness behaves exactly as without one): the harness's ``sidecar(attempt)``
    if given, else the task's ``episode_sidecar(attempt)``."""
    make = make or getattr(attempt.task, "episode_sidecar", None)
    return make(attempt) if make is not None else None


class _ActProbe:
    """A policy as ``run_episode`` sees it while a sidecar is on: every attribute is the policy's own; ``act`` shows the chosen
    action to the sidecar before the World applies it and returns it unchanged. A sidecar error never ends the episode."""

    def __init__(self, inner: Any, actor: str, sidecar: Any, world: Any):
        self.__dict__.update(_inner=inner, _actor=actor, _sidecar=sidecar, _world=world)

    async def act(self, obs: Any) -> Any:
        action = await self._inner.act(obs)
        try:
            await self._sidecar.before_action(self._world, self._actor, self._inner, action)
        except Exception:  # noqa: BLE001, S110 -- the sidecar flags its own faults; the agent's episode goes on
            pass
        return action

    def __getattr__(self, k: str) -> Any:
        return getattr(self.__dict__["_inner"], k)

    def __setattr__(self, k: str, v: Any) -> None:
        setattr(self._inner, k, v)


def probed(policies: dict[str, Any], sidecar: Any, world: Any) -> dict[str, Any]:
    """``policies`` for ``run_episode``: unchanged without a sidecar (or one without ``before_action``)."""
    if sidecar is None or not hasattr(sidecar, "before_action"):
        return policies
    return {a: _ActProbe(p, a, sidecar, world) for a, p in policies.items()}


def _take_findings(reps: list[Any], res: HarnessResult, *, prefix: str = "", seen: set[tuple[str, str]] | None = None) -> None:
    """``_take_grade_findings`` for every grade of one piece of work, in order (M10 lab audit R-3: a re-grade inside the
    attempt grades it again). A flag repeated by a later grade (or already in ``seen``: a re-grade's inherited flags) is
    taken once, since bench counts the rollout's flags; the verdict comes from the grade that carries it (the earlier
    grades were masked, and a masked report's verdict is never taken)."""
    seen = set() if seen is None else seen
    for r in reps:
        _take_grade_findings(r, res, prefix=prefix, seen=seen)


def _keep_tries(vdir: Path, reps: list[Any], res: HarnessResult) -> None:
    """M10 lab audit R-3: the grades of one piece of work that came back before a later try raised (a ``StageFailure``
    of R-2's quiesce, or an exception the boundary classified): their reports are written (``write_reports``) and their
    rule flags (``policy_process_killed``) join the rollout's, once each. No verdict is taken from them (they were masked
    or failed). Before, the tries' reports were lost with the exception, and a quiesce flag was dropped."""
    write_reports(vdir, reps)
    seen = {_flag_key(x) for x in res.flags}
    new: list[dict[str, Any]] = []
    for r in reps:
        for x in (getattr(r, "meta", None) or {}).get("flags") or []:
            if isinstance(x, dict) and x.get("rule") and _flag_key(x) not in seen:
                seen.add(_flag_key(x))
                new.append(x)
    if new:
        res.flags = [*res.flags, *new]
    if reps:
        res.grade["grades"] = len(reps)


@dataclass
class _HandedOff:
    """M10 lab audit R-3: the policy's World as ``verifier_world`` reads it (spec, store, pack root, trace, episode start
    and id), rebuilt from ``verifier/handoff_ref.json``: a re-grade starts nothing of the policy's and opens no task World.
    It has no Apps of its own, so ``verifier_world``'s shared-root check has nothing to compare."""

    spec: Any
    store: Any
    pack_root: Path | None
    log: list[Any]
    episode_start: Any
    episode_id: str
    initial: Any = None
    apps: dict[str, Any] = dataclasses.field(default_factory=dict)

    async def close(self) -> None:
        return None


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


@dataclass
class PolicyRef:
    id: str
    make: Callable[[Attempt], dict[str, Any]]  # attempt -> {actor_id: Policy}
    spend: Callable[[], tuple[float, float]] = lambda: (0.0, 0.0)  # (usd, uncertain usd) observed so far
    estimate_usd: float = 0.0
    # what the policy is beyond its id: model, endpoint, request defaults, sampling (runtime.lock, R-1); empty means
    # only the id identifies it, and the run lock lists the policy as opaque
    settings: dict[str, Any] = field(default_factory=dict)
    # M10 lab audit R-4: the policy's configuration as JSON (model, temperature, prompts or scripts, anything that decides what it
    # does): ``make`` is a closure, so nothing else says what the policy is, and the run's identity must. A PolicyRef run without
    # it is refused (``Undeclared``): left out, a changed temperature or model would resume silently into the same results
    config: dict[str, Any] | None = None
    # M10 lab audit R-5: the provider endpoint this policy calls (``base_url`` plus model, whatever names the server): policies that
    # share one share a provider breaker (``runtime.breaker``), so an outage opens it once. Defaults to the policy id. Placement,
    # not what is measured: it is not part of ``identity_config``
    endpoint: str | None = None

    def lock_settings(self) -> dict[str, Any] | None:
        return self.settings or None

    def identity_config(self) -> dict[str, Any]:
        """The policy's document in a run's identity: its id and ``config`` (``estimate_usd`` is accounting: operational)."""
        if self.config is None:
            raise Undeclared(f"policies.{self.id}: PolicyRef.make is a closure, so its model, temperature and prompts cannot be inspected: declare "
                             "PolicyRef(config={...}) (M10 lab audit R-4)")
        return {"id": self.id, "config": self.config}


class NativeHarness:
    """``checkpoint_every`` > 0: a mid-episode checkpoint every N steps, and a masked attempt's retry resumes from the
    previous attempt's last checkpoint (``runtime.checkpoint``). A task with ``subgoals`` gets a dense progress stream
    (``progress.jsonl``, ``verify.progress``); the trace always streams to ``trace.jsonl``. ``sidecar(attempt)`` (or the task's
    ``episode_sidecar``; M17 checkpoint credit) is an opt-in ``EpisodeSidecar``: it replaces the subgoal stream as the writer
    of ``progress.jsonl`` and is not attached to a resumed or multi-step attempt.

    M10 lab audit R-3, grading policy (not part of the id: it says how the same work is graded, not what produced it):
    ``grade_retries``, how many more times a failed grade is graded again inside the attempt (``grading.grade_in_place``);
    ``scoring_budget_s``, one budget for all of a grade's tries (its expiry during the first grade is ``verifier_timeout``;
    during a later grade the earlier try's failure stands, noting the budget; None: each try is bounded only by the
    grader's own timeout, as verifiers' unset scoring budget); ``control_timeout_s``, the budget of a control grade,
    raised to the task's own ``control_timeout_s`` when it declares a longer one (``grading.control_budget``:
    ``HarborDirTask``, the verifier timeout + 300 s; Harbor in place, agent + verifier timeout + margins).

    M10 lab audit R-4: the id is ``name@version#<12 hex>`` over what produces the work (``limits``, ``limit_policy``,
    ``checkpoint_every`` and a subclass's own: the collect mode, the patch contents), so a different configuration gets new
    ``SampleKey``s and its own GRPO group; the grading policy stays out of the id (R-3) and is in ``identity_config``, the
    document a resume compares."""

    def __init__(self, name: str = "native:llm-tool", version: str = "1", *, limits: EpisodeLimits | None = None, limit_policy: LimitPolicy | None = None,
                 checkpoint_every: int = 0, progress_full_every: int = 250, grade_retries: int = 1, scoring_budget_s: float | None = None,
                 control_timeout_s: float | None = 1800.0, sidecar: Callable[[Attempt], Any] | None = None):
        self.name, self.version = name, version
        self.limits = limits or EpisodeLimits(max_steps=30)
        self.limit_policy = limit_policy or LimitPolicy()
        self.checkpoint_every, self.progress_full_every = checkpoint_every, progress_full_every
        self.sidecar = sidecar
        self.grade_retries, self.scoring_budget_s, self.control_timeout_s = grade_retries, scoring_budget_s, control_timeout_s

    # how often progress is written and the grading timeouts (operational in identity.DENY too), not what the episode is
    LOCK_EXCLUDE: tuple[str, ...] = ("progress_full_every", "scoring_budget_s", "control_timeout_s")

    def lock_settings(self) -> dict[str, Any]:
        """Every public, non-callable attribute (subclasses' included, so a new setting counts without being listed)
        except ``LOCK_EXCLUDE``: limits, limit policy, checkpointing, workspace mode, RLM config (runtime.lock, R-1)."""
        from crucible.runtime.lock import jsonable

        return {k: jsonable(v) for k, v in sorted(vars(self).items())
                if not k.startswith("_") and k not in ("id", *self.LOCK_EXCLUDE) and not callable(v)}

    @property
    def id(self) -> str:
        """``name@version#<12 hex>`` (M10 lab audit R-4): the digest is over what produces the work (``_id_config``), so a
        harness with other limits is a different harness: its samples get new ``SampleKey``s and its GRPO group is its own. A
        property, not an attribute: a copy whose ``limits`` changed (bench's ``_sized``) gets its own id, and the same
        configuration gives the same id in every process."""
        return f"{self.name}@{self.version}#{harness_digest(self._id_config())}"

    @property
    def identity_key(self) -> str:
        """The key of this harness in a run's resolved document (``identity.harness_keys``): the id without its digest."""
        return f"{self.name}@{self.version}"

    def _id_config(self) -> dict[str, Any]:
        """What the id digests: the configuration that produces the work. A subclass adds its own (the collect mode, the
        patch contents). The grading policy (R-3: ``grade_retries``, ``scoring_budget_s``, ``control_timeout_s``) is not in it:
        it says how the same work is graded."""
        return {"limits": self.limits, "limit_policy": self.limit_policy, "checkpoint_every": self.checkpoint_every}

    def identity_config(self) -> dict[str, Any]:
        """The harness's document in a run's identity (M10 lab audit R-4): the id's configuration plus the grading policy and
        the progress cadence, which a resume must not change silently either. The timeouts and ``progress_full_every`` are
        operational (``identity.DENY``): recorded, never compared."""
        return {"name": self.name, "version": self.version, **self._id_config(), "grade_retries": self.grade_retries,
                "scoring_budget_s": self.scoring_budget_s, "control_timeout_s": self.control_timeout_s, "progress_full_every": self.progress_full_every}

    @staticmethod
    def _inherit_calls(res: HarnessResult, source: object) -> None:
        """A re-grade or re-collect measures the source attempt's work: its rollout is that attempt's model calls (V3: in
        train mode a call-less re-grade was masked ``capture_incomplete``)."""
        src = Path(str(source))
        src = src.parent if src.name == "preserved_ws" else src
        f = src / "calls.json"
        if f.exists():
            try:
                res.calls = json.loads(f.read_text())
                res.artifacts["calls_from"] = src.name
            except json.JSONDecodeError:
                pass

    def _regrade_source(self, attempt: Attempt) -> Path | None:
        """M10 lab audit R-3: the attempt whose persisted hand-off this attempt re-grades, or None. Only when the attempt
        that decides (``grading.deciding_record``: the newest recorded one, past an attempt the process died in or a
        grade-only retry that reached no verdict) was masked at verification (``RetryRule`` retries that only for persisted
        work); the hand-off is found by looking back to the newest attempt holding ``verifier/handoff_ref.json`` (a
        re-grade carries the ref on)."""
        if attempt.n <= 1:
            return None
        got = deciding_record(attempt.dir.parent, attempt.n)
        if got is None:
            return None
        r = got[1]
        if not (r.get("mask_sample") and (r.get("failure") or {}).get("stage") == "verification"):
            return None
        for k in range(attempt.n - 1, 0, -1):
            d = attempt.dir.parent / f"a{k}"
            if (d / "verifier" / HANDOFF_REF).exists():
                return d
        return None

    @staticmethod
    def _persist_handoff(attempt: Attempt, world: Any, final: Any, vdir: Path, res: HarnessResult) -> None:
        """M10 lab audit R-3: what ``regrade`` needs to grade this hand-off again without the policy's World:
        ``vdir/handoff_ref.json`` holds the quiesced ``WorldRef`` and the episode's start ref and id, the trace length
        (the trace itself is the attempt's ``trace.jsonl``), the spec, the store root (where the snapshot's blobs live) and
        the pack root, and names the attempt that ran the agent (``source``). Written before any verifier World starts;
        a ref that cannot be written leaves the attempt not regradable."""
        root = getattr(getattr(world, "store", None), "root", None)
        if root is None:
            return
        start = world.episode_start or world.initial
        doc = {"source": attempt.dir.name, "final": final.model_dump(mode="json"), "start": start.model_dump(mode="json") if start is not None else None,
               "episode_id": world.episode_id, "log_len": len(world.log), "spec": world.spec.model_dump(mode="json", exclude_none=True, by_alias=True),
               "store_root": str(Path(root).resolve()), "pack_root": str(world.pack_root) if world.pack_root is not None else None}
        try:
            (vdir / HANDOFF_REF).write_text(json.dumps(doc, default=str))
        except OSError:
            return
        res.regradable = attempt.regradable = True  # the Attempt's copy outlives a result the attempt clock cancels (run.py)

    @staticmethod
    def _on_again(attempt: Attempt) -> Callable[[Failure], None]:
        """M10 lab audit R-3 review round 2: ``grade_in_place``'s ``on_again`` on a persisted hand-off or patch. An
        answer-sensitive failure being graded again on this work is decided in this attempt once it repeats, so
        ``attempt.regradable`` is cleared before that grade starts: an attempt clock firing during it (or during the
        control) ends the attempt instead of scheduling a grade-only retry that grades it twice more. ``res.regradable`` is
        left as it is (a StageFailure from that try's verifier World start is still re-graded from the hand-off), and the
        caller sets ``attempt.regradable`` back from it once the grade returns."""

        def clear(f: Failure) -> None:
            if answer_sensitive(f):
                attempt.regradable = False

        return clear

    def _resumable(self, attempt: Attempt) -> tuple[Path, dict] | None:
        """(the previous attempt's dir, its checkpoint) when the previous attempt was masked mid-episode, or when the
        process died (no record; the journal moved the dir to ``aborted/``: M10 audit V9, the "dies at step 3,000" case
        never resumed). Multi-step tasks are not checkpointed (V14: a resume would reset the World at step 0)."""
        from crucible.runtime import checkpoint

        if not self.checkpoint_every or attempt.n <= 1 or getattr(attempt.task, "steps", None):
            return None
        # M10 lab audit R-4: a refused reservation (refused.json) dispatched nothing and left no checkpoint, so the resume keeps
        # looking where it did before R-4, when that attempt's dir was removed and the index reused
        k = attempt.n - 1
        while k > 1 and (attempt.dir.parent / f"a{k}" / REFUSED).exists():
            k -= 1
        prev = checkpoint.attempt_path(attempt.dir.parent, f"a{k}")
        rec = prev / "record.json"
        ck = checkpoint.load(prev)
        if ck is None:
            return None
        if not rec.exists():
            return prev, ck  # a process crash: nothing was recorded, the checkpoint is the last word
        r = json.loads(rec.read_text())
        return (prev, ck) if r.get("mask_sample") and (r.get("failure") or {}).get("stage") == "agent" else None

    async def _grade(self, attempt: Attempt, world: Any, grade_fn: Callable, vdir: Path, gdir: Path, artifacts_dir: Path, res: HarnessResult,
                     *, close_source: bool = True, persist: bool = False, retries: int | None = None) -> tuple[list[Any], Failure | None]:
        """Grade ``world`` with ``grade_fn(world, gdir)``: in place, or (``verifier.mode: separate``) in a fresh World
        restored from a quiesced, custody-checked snapshot. Writes ``vdir/report.json`` (and ``handoff.json``).

        M10 lab audit R-3: returns ``(reports, failure)`` (``runtime.grading``; ``failure`` None when the last grade
        passed through). In place the World is gone after the attempt, so a failed grade is graded again here
        (``grade_in_place``: up to ``retries`` more times, default ``grade_retries``, under ``scoring_budget_s``; with more
        than one try every report is kept and ``grades.json`` lists them) and the attempt is never regradable. Separate:
        ``persist`` (a single-step task) writes ``vdir/handoff_ref.json`` before any verifier World starts and sets
        ``res.regradable`` (``_persist_handoff``), so a failure another attempt can answer (a contaminated node, a
        transport failure) is re-graded from it by the next attempt (``regrade``); an answer-sensitive failure
        (``grading.answer_sensitive``: a hang or a checker outage on this answer) is graded again at once in a second
        verifier World over the same snapshot (``verifier_world`` tolerates the closed source) and, when it repeats, is
        decided in this attempt (no longer regradable; ``run`` settles it with the task's control).

        ``attempt.phase`` becomes ``verification`` here and ``attempt.regradable`` mirrors ``res.regradable`` (M10 lab
        audit R-3): the Runner reads them when its attempt clock cancels the harness. A later try that raises keeps the
        reports of the tries before it (``_keep_tries``).

        Review round 2: the grading's window opens on the Runner's clock (``grading.grading_window``: the scoring budget
        defaults to the clock's length); the verifier World's start and close run outside the scoring budget
        (``grading.unbudgeted``); only an answer-sensitive failure graded twice (or out of budget, ``grading.decided``)
        is decided here, one graded once stays regradable; ``attempt.regradable`` is cleared before an answer-sensitive
        failure is graded again (``on_again``), so a clock firing then ends the attempt. Without ``persist`` (a
        multi-step step) nothing is left for a fresh attempt, so a separate step's snapshot is graded again here like an
        in-place grade (``again=regradable``, a fresh verifier World per try, a failed start included)."""
        attempt.phase = "verification"
        budget = grading_window(attempt, attempt.task, budget_s=self.scoring_budget_s, control_s=self.control_timeout_s)
        vdir.mkdir(parents=True, exist_ok=True)
        tries = self.grade_retries if retries is None else retries
        reps: list[Any] = []
        if wants_separate(world):  # M1: grade a restored copy in a fresh World; the policy's World is closed first
            with boundary("verification"):
                try:
                    final, handoff = await hand_off(world, artifacts_dir=artifacts_dir)
                except CustodyViolation as e:  # the policy handed in what the verifier must not touch: scored 0, never masked
                    (vdir / "custody.json").write_text(json.dumps({"problems": e.problems}, indent=1))
                    raise StageFailure(Failure(kind="crucible:policy_misbehavior", stage="verification", reason=str(e)[:400], blame="policy", mask=False)) from e
            (vdir / "handoff.json").write_text(json.dumps(handoff.to_dict(), indent=1))
            res.artifacts[f"handoff:{vdir.relative_to(attempt.dir)}"] = str((vdir / "handoff.json").relative_to(attempt.dir))
            if persist:
                self._persist_handoff(attempt, world, final, vdir, res)

            async def separate(d: Path) -> Any:
                wd = vdir / "world" if d == gdir else d.parent / vdir.name / "world"  # each try its own verifier World
                stack = contextlib.AsyncExitStack()
                try:
                    # M10 lab audit R-3: closing the source, the verifier World's start (a serial host lock, an App's
                    # readiness), its restore and its fingerprint check are infrastructure, not this answer's grade (no
                    # grader has seen the answer yet): boundary classifies them as before R-3 (a TimeoutError ->
                    # transport_timeout, infra, retryable). grade_in_place re-raises the StageFailure unchanged, so the
                    # persisted hand-off (res.regradable) is re-graded by a fresh attempt, never retried here as
                    # verifier_timeout and never decided by a control the verifier World is not part of. Review round 2:
                    # they run outside the scoring budget (unbudgeted), so a budget expiry is only ever the grade's own (a
                    # budget that ran out in a serial lock was the answer's verifier_timeout, and a control that needs no
                    # verifier World scored the policy 0). Review round 3: the restore's length is the policy's state (its
                    # snapshot), but the same state hand_off just snapshotted on the attempt clock with no budget at all, and
                    # verifier_world cannot split the restore from the serial lock's wait, which a capped hold
                    # (unbudgeted(answer=True)) would charge to the answer; so the hold stays uncapped, bounded by the clock,
                    # which gets the time back once the World has started (grade_in_place's attempt)
                    try:
                        with boundary("verification"), unbudgeted():
                            vworld = await stack.enter_async_context(verifier_world(world, final, workdir=wd, close_source=close_source))
                    except StageFailure as sf:
                        if persist:
                            raise  # the persisted hand-off is re-graded by a fresh attempt
                        # M10 lab audit R-3 review round 2: a step persists nothing, so its snapshot is graded again in a
                        # fresh verifier World inside the attempt. Chained, so grade_in_place reads the boundary's own
                        # classification (classify_exception's StageFailure link; _raised_timeout stops at it, so never
                        # verifier_timeout), and one it does not grade again is re-raised and turned back into the same
                        # StageFailure by the outer boundary
                        raise RuntimeError(f"the verifier World did not start: {sf.failure.reason}"[:500]) from sf
                    return await grade_fn(vworld, d)
                finally:
                    with unbudgeted():
                        await stack.aclose()

            # M10 lab audit R-3 review round 2: a step (no persist) leaves no hand-off for a fresh attempt, so any regradable
            # failure (a contaminated node, a transport failure, a verifier World that did not start) is graded again here,
            # like an in-place grade; with a persisted hand-off only an answer-sensitive one is (the rest waits for regrade)
            again = answer_sensitive if persist else regradable
            try:
                with boundary("verification"):
                    _, f = await grade_in_place(separate, gdir, retries=tries, budget_s=budget, again=again, out=reps, on_again=self._on_again(attempt),
                                                attempt=attempt)
            except StageFailure:
                _keep_tries(vdir, reps, res)
                raise
            if f is not None and decided(f):
                # graded again on this snapshot already (or the scoring budget ran out during its first grade): decided here,
                # never a third grade in a fresh attempt. Review round 2: one answer-sensitive grade alone (grade_retries=0)
                # stays regradable, so a grade-only retry grades it again (seen_before) before any control decides
                res.regradable = False
            attempt.regradable = res.regradable
        else:
            try:
                with boundary("verification"):
                    _, f = await grade_in_place(lambda d: grade_fn(world, d), gdir, retries=tries, budget_s=budget, out=reps, attempt=attempt)
            except StageFailure:
                _keep_tries(vdir, reps, res)
                raise
        write_reports(vdir, reps)
        return reps, f

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
            attempt.phase = "agent"  # M10 lab audit R-3: this step's episode (an attempt clock firing here is the agent's, run.py)
            agent_clock(attempt)  # review round 2: the previous step's grading window never extends this agent phase
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
                raise StageFailure(await _masked_episode(attempt.task, world, ep, prefix=f"step {st.name}: "))
            if (mb := misbehaved(ep, prefix=f"step {st.name}: ")) is not None:
                raise mb
            if ep.stop_condition in LIMIT_STOPS and self.limit_policy.for_stop(ep.stop_condition) == "mask":
                raise StageFailure(Failure(kind="agent_timeout", stage="agent", reason=f"step {st.name}: {ep.stop_condition}", mask=True, blame="harness"))
            broken = await attempt.task.broken(world) if hasattr(attempt.task, "broken") else []
            if broken:  # the policy broke its environment: its own outcome, scored 0, never masked
                raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="agent", reason=f"step {st.name}: " + "; ".join(broken)[:400]))
            t0 = time.monotonic()
            # M10 lab audit R-3: a step persists no hand-off, and is graded again inside the attempt like an in-place grade
            # (grade_retries, scoring_budget_s; review round 2: a separate-verifier step too, in a fresh verifier World over
            # the same snapshot, a contaminated node or a transport failure included), unless the task declares its step grades unrepeatable (step_grades_once:
            # HarborMultiStepTask, whose R-2 per-step state, the setup listing and the previous grade's leftovers, is used up
            # by the step's grade, so a second grade of it would not be the same grade). A failed step grade ends the
            # attempt terminal-masked (never an agent re-run)
            once = bool(getattr(attempt.task, "step_grades_once", False))
            reps, f = await self._grade(attempt, world, st.grade, sdir / "verifier", sdir / "grade", sdir / "artifacts", res,
                                        close_source=i == len(steps) - 1, retries=0 if once else None)
            _take_findings(reps, res, prefix=f"step {st.name}: ")
            if f is not None:
                # M10 lab audit R-3 item 2: a step_grades_once task has no control (its step grade cannot be repeated on the
                # same state): an answer-sensitive step failure masks with flag no_control, never retried. Any other task's
                # control decides it, as run() does: the World probed, then closed (nothing of the policy's runs beside the
                # control; World.close is idempotent), then the control (it must exercise every step's grader)
                probe = "unprobed" if wants_separate(world) else await liveness(attempt.task, world)
                if not once:
                    with contextlib.suppress(Exception):  # cleanup never overrides a result
                        await world.close()
                raise StageFailure(await settle(attempt.task, f, workdir=sdir / "control", timeout_s=self.control_timeout_s, probe=probe, control=not once,
                                                attempt=attempt))
            rep = reps[-1]
            flat = flat_reward(rep.reward_json()) or {"reward": rep.reward}
            why = _below(flat, st.min_reward)
            results.append({"name": st.name, "reward": rep.reward, "outcome": rep.outcome, "gates": rep.gate_bits(), "gate_kinds": rep.gate_kinds(),
                            "stop": ep.stop_condition,
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
        # a step's gate is its min_reward threshold: an outcome check, unless one of the step's own hard rules failed (M6 R2)
        kinds = {f"step/{r['name']}": "outcome" if all(v for g, v in r["gates"].items() if r["gate_kinds"].get(g, "gate") == "gate") else "gate"
                 for r in results}
        res.grade.update({"steps": results, "step_reward": strategy, "reward": round(reward, 6), "completed_steps": len(results),
                          "gates": {f"step/{r['name']}": int(r["passed_gate"]) for r in results}, "gate_kinds": kinds})
        assert ep is not None
        merged: dict[str, list] = {}
        for pols in used:
            for a, p in pols.items():
                merged.setdefault(a, []).extend(getattr(p, "model_calls", []))
        ep = dataclasses.replace(ep, steps=total_steps, model_calls=merged, trace=world.to_trace())
        return ep, policies

    async def regrade(self, attempt: Attempt, prior: Any) -> HarnessResult:
        """M10 lab audit R-3: grade the hand-off an earlier attempt persisted (``verifier/handoff_ref.json``) again, for a
        grade-only retry and ``Runner.regrade``: no agent, no spend. The policy's World is rebuilt only as
        ``verifier_world`` reads it (``_HandedOff``: the persisted spec, the store holding the snapshot's blobs, the trace
        up to the hand-off, the episode's start ref and id), and a fresh verifier World restores the quiesced ``WorldRef``
        and is graded, as ``_grade`` does; an answer-sensitive failure is graded once more over the same snapshot and, when
        it repeats, decided by the task's control (``grading.settle``: scored 0, a task mask, or ``no_control``). A ref
        that cannot be read, or whose trace is missing or shorter than the hand-off (``handoff_unreadable``), or whose spec
        is not the snapshot's (``handoff_spec_mismatch``) is a
        harness fault, masked and never retried. The record measures the attempt that ran the agent (``source``): its
        calls, steps, truncation, progress, seat rewards and rule flags, and its limit stop's penalty; the ref is carried
        into this attempt, so a later re-grade finds it here too."""
        from crucible.core.schema.state import WorldRef
        from crucible.core.schema.world import WorldSpec
        from crucible.core.store import Store
        from crucible.world.world import World

        prior = Path(str(prior))
        attempt.phase = "verification"  # M10 lab audit R-3: no agent here; an attempt clock firing is this grade's (run.py)
        budget = grading_window(attempt, attempt.task, budget_s=self.scoring_budget_s, control_s=self.control_timeout_s)  # review round 2
        res = HarnessResult(stop="regraded")
        t: dict[str, float] = {}
        t0 = time.monotonic()
        vdir, gdir = attempt.dir / "verifier", attempt.dir / "grade"
        source, prev = prior, {}
        reps: list[Any] = []
        try:
            with boundary("verification"):
                vdir.mkdir(parents=True, exist_ok=True)
                try:
                    doc = json.loads((prior / "verifier" / HANDOFF_REF).read_text())
                    source = prior.parent / str(doc.get("source") or prior.name)
                    spec = WorldSpec.model_validate(doc["spec"])
                    final = WorldRef.model_validate(doc["final"])
                    start = WorldRef.model_validate(doc["start"]) if doc.get("start") is not None else None
                    if not Path(doc["store_root"]).is_dir():
                        raise FileNotFoundError(f"the hand-off's store {doc['store_root']} is gone")
                    trace, n = source / "trace.jsonl", int(doc["log_len"])
                    # M10 lab audit R-3: the trace is part of the persisted hand-off, like the store: a missing or short one is
                    # handoff_unreadable, never an empty episode (a trace-reading check would pass on events it never sees)
                    log = World.read_trace(trace, upto=n)  # a missing file raises FileNotFoundError (OSError)
                    if len(log) < n:
                        raise ValueError(f"the trace {trace.name} of {source.name} holds {len(log)} events, the hand-off {n}")
                    store = Store(doc["store_root"])
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
                    raise StageFailure(Failure(kind="agent_run_error", stage="verification", blame="harness", retryable=False, detail="handoff_unreadable",
                                               reason=f"the persisted hand-off of {prior.name} cannot be read: {type(e).__name__}: {e}"[:500])) from e
                prev = _read_json_dict(source / "record.json")
                ro = _read_json_dict(source / "rollout.json")
                res.flags = [f for f in ro.get("flags") or [] if isinstance(f, dict) and f.get("rule") != "reconcile_fatal"]  # the Runner adds that one again
                if spec.content_hash() != final.world:
                    raise StageFailure(Failure(kind="agent_run_error", stage="verification", blame="harness", retryable=False, detail="handoff_spec_mismatch",
                                               reason=f"the persisted spec of {prior.name} hashes to {spec.content_hash()[:12]}, not the snapshot's world "
                                                      f"{final.world[:12]}"))
                src = _HandedOff(spec=spec, store=store, pack_root=Path(doc["pack_root"]) if doc.get("pack_root") else None, log=log,
                                 episode_start=start, episode_id=str(doc.get("episode_id") or ""))
                (vdir / HANDOFF_REF).write_text(json.dumps({**doc, "source": source.name}, default=str))
                if (prior / "verifier" / "handoff.json").exists():
                    (vdir / "handoff.json").write_text((prior / "verifier" / "handoff.json").read_text())
                    res.artifacts["handoff:verifier"] = "verifier/handoff.json"
                res.regradable = attempt.regradable = True

                async def separate(d: Path) -> Any:
                    wd = vdir / "world" if d == gdir else d.parent / vdir.name / "world"
                    stack = contextlib.AsyncExitStack()
                    try:
                        # M10 lab audit R-3: the verifier World's own start, restore and fingerprint check are infrastructure,
                        # classified by boundary as in _grade (never this answer's verifier_timeout); review round 2: outside
                        # the scoring budget (unbudgeted), its close too
                        with boundary("verification"), unbudgeted():
                            vworld = await stack.enter_async_context(verifier_world(src, final, workdir=wd, close_source=False))  # type: ignore[arg-type]
                        return await attempt.task.grade(vworld, d)
                    finally:
                        with unbudgeted():
                            await stack.aclose()

                # review round 2: the answer-sensitive grades the attempt that decided already ran on this snapshot (seen),
                # so a repeat here is decided by the control; attempt.regradable cleared before it is graded again (on_again)
                seen = seen_before(attempt.dir.parent, attempt.n)
                try:
                    _, f = await grade_in_place(separate, gdir, retries=self.grade_retries, budget_s=budget, again=answer_sensitive,
                                                out=reps, seen=seen, on_again=self._on_again(attempt), attempt=attempt)
                except Exception:
                    _keep_tries(vdir, reps, res)  # the tries that came back before a later one raised (the boundary classifies it)
                    raise
                if f is not None and decided(f):
                    res.regradable = False  # graded twice on this snapshot now (or the budget ran out): decided here
                attempt.regradable = res.regradable
                write_reports(vdir, reps)
                t["verification"] = time.monotonic() - t0
                res.grade = {**res.grade, **grade_summary(reps, f)}
                _take_findings(reps, res, seen={_flag_key(x) for x in res.flags})
                if f is not None:
                    # M10 lab audit R-3 item 2: a failure another attempt can still answer waits for it (res.regradable);
                    # an answer-sensitive one graded twice on this snapshot is decided by the task's control (no live
                    # World of the policy's is left to probe)
                    raise StageFailure(f if res.regradable else await settle(attempt.task, f, workdir=attempt.dir / "control",
                                                                             timeout_s=self.control_timeout_s, attempt=attempt))
                res.reward, res.gates, res.gate_kinds = reps[-1].reward, reps[-1].gate_bits(), reps[-1].gate_kinds()
                stop = (prev.get("meta") or {}).get("stop")
                pen = self.limit_policy.penalty(stop) if stop in LIMIT_STOPS else None
                if pen is not None:  # the agent attempt stopped at a limit: its configured penalty, as run() applies it
                    res.grade["limit_penalty"] = pen
                    res.reward = pen
        except StageFailure as sf:
            res.failure = sf.failure
        meta = prev.get("meta") or {}
        self._inherit_calls(res, source)
        res.steps, res.truncated = int(prev.get("steps") or 0), bool(meta.get("truncated"))
        for k in ("progress", "seat_rewards"):
            if meta.get(k):
                res.grade[k] = meta[k]
        res.artifacts["regraded_from"] = source.name
        res.timings = {k: round(v, 3) for k, v in t.items()}
        return res

    async def run(self, attempt: Attempt) -> HarnessResult:
        from crucible.runtime import checkpoint

        src = self._regrade_source(attempt)
        if src is not None:
            # M10 lab audit R-3: a grade-only retry never samples the policy again (the old native a2 after a verification
            # mask was a full agent re-run, hidden by deterministic oracles): it grades the persisted hand-off
            return await self.regrade(attempt, src)
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
                side = None if resume or getattr(attempt.task, "steps", None) else episode_sidecar(self.sidecar, attempt)
                if side is not None:  # M17 checkpoint credit: the sidecar writes progress.jsonl (one writer per attempt)
                    observers.append(side)
                elif getattr(attempt.task, "subgoals", None) is not None:
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
                    ep = await run_episode(world, probed(policies, side, world), limits=self.limits, observers=observers, reset=resume is None, start_steps=start,
                                           after_step=ckpt, sample_key=attempt.id)  # per-attempt cache key: k samples never replay one completion
                if observers:  # M7 R7: a tracker that raised leaves the episode graded, its progress unavailable
                    from crucible.verify.progress import observed_progress

                    res.grade["progress"] = observed_progress(ep.observed[0], attempt.dir / "progress.jsonl")
            t["agent"] = time.monotonic() - t0
            res.steps, res.stop, res.truncated = ep.steps, ep.stop_condition, ep.truncated or ep.stop_condition in LIMIT_STOPS
            from crucible.credit.hack_detector import rule_flags
            from crucible.credit.shaping import format_errors_by_call

            trace = world.to_trace()
            # m13 C-4 level 1 (intercepts, tampering rules). M10 lab audit R-2 review round 2: merged, never assigned: a
            # multi-step task's step grades already put their meta.flags here (_run_steps -> _take_grade_findings:
            # policy_process_killed, grader_guard suspicions), which an assignment dropped; on the single-step path
            # res.flags is still empty here and the grade's flags join later
            res.flags = [*rule_flags(trace), *res.flags]
            res.format_errors = format_errors_by_call(trace, ep.model_calls)  # m13 C-8: charged to the issuing call's tokens
            save_episode(attempt, ep, res, self.id, policies, world=world)
            if getattr(world, "turns", None) is not None and world.turns.rewards:  # per-seat outcomes of a turn-based World (M1 item 7)
                res.grade["seat_rewards"] = dict(world.turns.rewards)
            if ep.masked:
                raise StageFailure(await _masked_episode(attempt.task, world, ep))
            pe = next((x for x in ep.failures if x.get("kind") == "crucible:policy_error"), None)
            if pe is not None:  # the policy's own code or output raised: its outcome, scored 0 (M10 audit V6)
                raise StageFailure(Failure(kind="crucible:policy_error", stage="agent", reason=pe.get("reason", "")))
            if (mb := misbehaved(ep)) is not None:  # processes survived the quiesce: scored 0, never graded (as _run_steps does)
                raise mb
            broken = await attempt.task.broken(world) if hasattr(attempt.task, "broken") and res.grade.get("steps") is None else []
            if broken:  # the policy broke its environment: its own outcome, scored 0, never masked
                raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="agent", reason="; ".join(broken)[:400]))
            if ep.stop_condition in LIMIT_STOPS and self.limit_policy.for_stop(ep.stop_condition) == "mask":
                raise StageFailure(Failure(kind="agent_timeout", stage="agent", reason=ep.stop_condition, mask=True, blame="harness"))
            t0 = time.monotonic()
            if res.grade.get("steps") is not None:  # graded step by step inside _run_steps
                t["verification"] = sum(r.get("verification_s", 0.0) for r in res.grade["steps"])
                res.reward, res.gates, res.gate_kinds = res.grade["reward"], res.grade.get("gates", {}), res.grade.get("gate_kinds", {})
            else:
                reps, f = await self._grade(attempt, world, attempt.task.grade, attempt.dir / "verifier", attempt.dir / "grade", attempt.dir / "artifacts", res,
                                            persist=True)
                t["verification"] = time.monotonic() - t0
                res.grade = {**res.grade, **grade_summary(reps, f)}
                _take_findings(reps, res)  # flags from every grade of this work; a policy verdict ends the attempt
                if f is not None:
                    # M10 lab audit R-3: a persisted hand-off (res.regradable) is re-graded by the next attempt; in place, or
                    # after an answer-sensitive failure was graded again here, the attempt ends here and RetryRule never
                    # turns it into an agent re-run. Item 2: that last word is grading.settle. The sandbox is probed while
                    # it still runs (in place; a separate verifier's source World is already closed), then the policy's
                    # World is closed (nothing of the policy's runs beside the control), then an answer-sensitive failure
                    # is decided by the task's control
                    if not res.regradable:
                        probe = "unprobed" if wants_separate(world) else await liveness(attempt.task, world)
                        closing, world = world, None
                        with contextlib.suppress(Exception):  # cleanup never overrides a result
                            await closing.close()
                        f = await settle(attempt.task, f, workdir=attempt.dir / "control", timeout_s=self.control_timeout_s, probe=probe, attempt=attempt)
                    raise StageFailure(f)
                res.reward, res.gates, res.gate_kinds = reps[-1].reward, reps[-1].gate_bits(), reps[-1].gate_kinds()
                if side is not None and hasattr(side, "reconcile") and reps:
                    # one verdict on the end state (factory.checkpoint): the World graded here is the end checkpoint's state
                    fm = (reps[-1].meta or {}).get("milestones") if (reps[-1].infra or {}).get("status") == "ok" else None
                    got = side.reconcile(None, fm)
                    if got is not None:
                        res.grade["progress"] = got
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
            settle_usage(res, calls, restored)
            own = [c for cs in calls.values() for c in cs if c.get("cost_usd") is not None and c.get("call_id") not in restored]
            if own:  # the attempt's own model calls: a client shared by concurrent attempts makes the spend delta count theirs too
                res.cost_usd = round(sum(float(c["cost_usd"]) for c in own), 9)
            elif restored:
                res.cost_usd = 0.0
            res.timings = {k: round(v, 3) for k, v in t.items()}
        return res
