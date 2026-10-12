"""Durable records of continuous-control episodes: ``crucible.control_rollout/v1`` (M12; M1 item 10).

**Why a sibling schema, not ``crucible.rollout/v1``.** A rollout/v1 record is a sequence of model calls whose prompt and
completion are token ids with one behaviour log-prob per completion token; ``valid_for_training`` and every trainer
bridge (``bridges.common.flatten``) read exactly that. A control step is a float observation, a numeric action vector
with **one joint** log-prob (a Gaussian's sum over dimensions, or a categorical's), a value estimate and simulator ticks.
Squeezing that into the token fields would make a control record look train-valid to the language-model trainer and be
flattened into nonsense sequences. So the step payload has its own model, under the **same envelope** as rollout/v1
(``rollout_id``, ``sample``, ``task``, ``harness``, ``policy``, ``group``, ``reward``, ``reward_components``,
``truncated``, ``stop``, ``failure_kind``, ``mask_sample``, ``timings``, ``trace_id``, ``source``): sinks, JSONL readers
and dashboards treat both alike (``JsonlSink`` writes ``{"kind": "control_rollout", "data": ...}``), and a record is
self-describing through its ``schema`` literal.

**Two producers, one record.**

- Neural policies (``train.control.ppo(..., recorder=ControlRecorder(...))``, off by default): the recorder wraps each
  ``WorldEnv`` of the ``VectorWorldEnv`` and writes one record per finished episode, with each step's sampled action
  (``raw_action``) and the action the World executed (clipped into the box), the joint log-prob and value of the sampled
  action, the observation digest (and the flat observation with ``values=True``), reward, terminated / truncated,
  ticks, and the PPO update (``policy_version``) that collected it.
- Language-model policies acting through ``ControlApp``'s ``step`` / ``wait`` / macro tools: ``record_from_world`` reads
  the App's journal after the episode and links every simulator step to the model call that issued it (``call_id``). The
  model's tokens stay in the episode's rollout/v1 record (they are what the LM trainer trains); this record is the
  simulator-side audit, joined to it by ``rollout_id`` / ``trace_id``.

**Replay is the audit.** The steps are the ``ControlApp`` journal (seed, reset digest, every tool + action + requested
ticks + observation digest), so ``replay`` rebuilds a fresh World from the task, reseeds the App and re-executes every
step through ``World.step``, checking the reset digest, every observation digest, reward and termination: a record
that does not reproduce is reported step by step, never trusted. ``config_digest`` (the App's params) refuses a replay
against a differently configured simulator.

Pattern: the Gymnasium ``step`` tuple (obs, reward, terminated, truncated) and CleanRL's PPO rollout buffers (action,
log-prob, value per step), persisted per episode with the ControlApp journal's digests. No code copied.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel

CONTROL_SCHEMA = "crucible.control_rollout/v1"
RECORD_KIND = "control_rollout"


class ControlStep(CrucibleModel):
    t: int  # decision index in the episode
    tool: str = "step"  # step | wait | <macro name>
    action: Any = None  # as executed by the World (JSON numbers; box actions clipped)
    raw_action: Any = None  # as sampled, when it differs from ``action``: the log-prob is of this one
    ticks_requested: int = 1
    ticks: int = 1  # simulator ticks that ran (fewer when the episode ended mid-hold)
    tick: int = 0  # the simulator's tick counter after the step
    reward: float = 0.0  # summed over the ticks
    terminated: bool = False
    truncated: bool = False
    obs_digest: str  # ``world.apps.control.obs_digest`` of the observation after the step (what replay checks)
    obs: list[float] | None = None  # the flat observation after the step (``values=True``)
    logprob: float | None = None  # joint log-prob of ``raw_action`` (or ``action``) under the behaviour policy
    value: float | None = None  # the critic's estimate of the observation the action was taken in
    policy_version: int | str | None = None
    call_id: str | None = None  # language-model policy: the model call that issued this tool call
    tool_call_id: str | None = None


class ControlRecord(CrucibleModel):
    schema_: Literal["crucible.control_rollout/v1"] = Field(CONTROL_SCHEMA, alias="schema")
    # the rollout/v1 envelope
    rollout_id: str
    sample: str
    task: str
    harness: str
    policy: str
    group: str | None = None
    reward: float | None = None  # the graded reward when a grader ran, else the episode return; None = masked
    reward_components: dict[str, float] = Field(default_factory=dict)
    truncated: bool = False
    stop: str | None = None  # terminated | truncated | run_end (the run stopped mid-episode) | <harness stop>
    failure_kind: str | None = None
    mask_sample: bool = False
    timings: dict[str, float] = Field(default_factory=dict)
    trace_id: str | None = None
    source: str | None = None
    # the episode
    env: str  # gymnasium:<id> | <module:factory> | <remote base url>
    app: str = "sim"
    actor: str = "agent"
    config_digest: str | None = None  # sha256 of the App's params (canonical JSON)
    seed: int
    reset_digest: str
    reset_obs: list[float] | None = None
    episode_return: float = 0.0
    ticks: int = 0
    terminated: bool = False
    steps: list[ControlStep] = Field(default_factory=list)

    def digests(self) -> list[str]:
        return [s.obs_digest for s in self.steps]

    def valid_for_training(self) -> tuple[bool, str]:
        """A neural-policy PPO record trains only with a behaviour log-prob and value on every step."""
        if self.mask_sample or self.reward is None:
            return False, f"masked ({self.failure_kind or 'no reward'})"
        if not self.steps:
            return False, "no steps recorded"
        bad = [s.t for s in self.steps if s.logprob is None]
        return (False, f"behaviour log-prob missing on {len(bad)}/{len(self.steps)} steps") if bad else (True, "ok")


def env_id(params: dict[str, Any]) -> str:
    if params.get("env"):
        return str(params["env"])
    if params.get("factory"):
        return str(params["factory"])
    return str((params.get("client") or {}).get("base_url", "unknown"))


def config_digest(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:24]


def _step_of(t: int, j: dict[str, Any], **extra: Any) -> ControlStep:
    return ControlStep(t=t, tool=j["tool"], action=j["action"], ticks_requested=int(j["ticks_requested"]), ticks=int(j["ticks"]),
                       reward=float(j["reward"]), terminated=bool(j["terminated"]), truncated=bool(j["truncated"]), obs_digest=j["digest"], **extra)


def record_from_world(world: Any, *, rollout_id: str, task: str, app: str = "sim", actor: str = "agent", harness: str = "native:world@1",
                      policy: str = "llm", model_calls: dict[str, list[dict[str, Any]]] | None = None, reward: float | None = None,
                      stop: str | None = None, group: str | None = None) -> ControlRecord:
    """The control record of a finished World episode (a language model acting through the App's tools): the App's journal
    is the step list; each step is linked to the model call that issued it (``model_calls``: ``EpisodeResult.model_calls``)
    through the trace's successful step / wait / macro results, which the journal records in the same order."""
    a = world.apps[app]
    j = a.journal
    act_tools = {f"{app}.{n}" for n in ("step", "wait", *(a.spec.params.get("macros") or {}))}
    ids = [e.data.get("id") for e in world.log if e.type == "tool_result" and e.actor == actor and e.data.get("function") in act_tools and e.data.get("ok")]
    owner = {act.split(":", 1)[1]: c["call_id"] for cs in (model_calls or {}).values() for c in cs for act in c.get("actions") or [] if act.startswith("tool:")}
    steps, tick = [], 0
    for t, s in enumerate(j.get("steps") or []):
        tick += int(s["ticks"])
        tc = ids[t] if t < len(ids) else None
        steps.append(_step_of(t, s, tick=tick, tool_call_id=tc, call_id=owner.get(tc) if tc else None))
    last = steps[-1] if steps else None
    ret = float(j.get("return", 0.0))
    return ControlRecord(rollout_id=rollout_id, sample=rollout_id, task=task, harness=harness, policy=policy, group=group,
                         reward=ret if reward is None else float(reward), reward_components={"return": ret}, env=env_id(a.spec.params), app=app, actor=actor,
                         config_digest=config_digest(a.spec.params), seed=int(j["seed"]), reset_digest=j["reset_digest"], episode_return=ret,
                         ticks=int(j.get("tick", 0)), terminated=bool(last and last.terminated), truncated=bool(last and last.truncated),
                         stop=stop or ("terminated" if last and last.terminated else "truncated" if last and last.truncated else None),
                         trace_id=getattr(world, "episode_id", None), steps=steps)


@dataclass
class ReplayReport:
    ok: bool
    steps: int
    digests: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)


async def replay(record: ControlRecord, task: Any, workdir: Path, *, check_config: bool = True) -> ReplayReport:
    """Re-run a record through a fresh World (``task.open(workdir)``): reseed the App, execute every recorded tool call
    through ``World.step`` and compare the reset digest, each observation digest, reward and termination."""
    from crucible.core.schema.actions import ToolCall
    from crucible.world.apps.control import obs_digest

    world = await task.open(Path(workdir))
    rep = ReplayReport(ok=True, steps=0)
    try:
        a = world.apps[record.app]
        if check_config and record.config_digest and config_digest(a.spec.params) != record.config_digest:
            rep.mismatches.append(f"config: the App's params differ from the recorded run ({config_digest(a.spec.params)} != {record.config_digest})")
            rep.ok = False
            return rep
        await a.reseed(record.seed)
        if obs_digest(a.obs) != record.reset_digest:
            rep.mismatches.append(f"reset: seed {record.seed} gave a different first observation")
        for s in record.steps:
            args = {"action": s.action} if s.tool == "step" else ({"ticks": s.ticks_requested} if s.tool == "wait" else {})
            res = await world.step(record.actor, ToolCall(tool_call_id=f"replay-{s.t}", function_name=f"{record.app}.{s.tool}", arguments=args))
            r = res.observation.tool_results[0]
            rep.steps += 1
            if not r.ok:
                rep.mismatches.append(f"step {s.t}: refused on replay ({r.error}: {r.content[:200]})")
                break
            d = obs_digest(a.obs)
            rep.digests.append(d)
            if d != s.obs_digest:
                rep.mismatches.append(f"step {s.t}: observation digest {d} != recorded {s.obs_digest}")
            if not math.isclose(float(r.data["reward"]), s.reward, rel_tol=1e-9, abs_tol=1e-12):
                rep.mismatches.append(f"step {s.t}: reward {r.data['reward']} != recorded {s.reward}")
            if bool(r.data["terminated"]) != s.terminated or bool(r.data["truncated"]) != s.truncated:
                rep.mismatches.append(f"step {s.t}: termination differs")
    finally:
        await world.close()
    rep.ok = rep.ok and not rep.mismatches
    return rep


class _RecordingEnv:
    """A ``WorldEnv`` seen through a ``ControlRecorder``: ``areset`` opens an episode, ``astep`` appends the step the App
    journaled (the executed action, ticks, digest) with the policy's annotations, an ended episode is emitted. Every other
    attribute is the wrapped env's."""

    def __init__(self, env: Any, index: int, rec: ControlRecorder):
        self.env, self.index, self.rec = env, index, rec

    def __getattr__(self, name: str) -> Any:
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    async def areset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        out = await self.env.areset(seed)
        self.rec._begin(self.index, self.env, out[0])
        return out

    async def astep(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        out = await self.env.astep(action)
        self.rec._step(self.index, self.env, out)
        return out


class ControlRecorder:
    """Collects ``ControlRecord``s from a ``VectorWorldEnv`` run (``attach``), one per finished episode; ``sink`` (a
    ``train.sinks.Tracking`` or any ``Sink``) receives each as ``record("control_rollout", rec, episode_index)``.
    ``keep``: also hold them in ``records``. ``values``: store flat observations, not only digests."""

    def __init__(self, sink: Any = None, *, keep: bool = True, values: bool = False, run_id: str = "ppo", harness: str = "ppo@1",
                 policy: str = "neural:ppo"):
        self.sink, self.keep, self.values = sink, keep, values
        self.run_id, self.harness, self.policy = run_id, harness, policy
        self.records: list[ControlRecord] = []
        self.emitted = 0
        self._open: dict[int, dict[str, Any]] = {}
        self._notes: dict[int, dict[str, Any]] = {}

    def attach(self, venv: Any) -> Any:
        venv.envs = [e if isinstance(e, _RecordingEnv) else _RecordingEnv(e, i, self) for i, e in enumerate(venv.envs)]
        return venv

    def annotate(self, raw_actions: list[Any], logprobs: list[float], values: list[float], version: int | str | None = None) -> None:
        """The policy's view of the actions about to be stepped, one per env (called right before ``venv.astep``)."""
        for i, (a, lp, v) in enumerate(zip(raw_actions, logprobs, values, strict=True)):
            self._notes[i] = {"raw": a, "logprob": float(lp), "value": float(v), "version": version}

    @staticmethod
    def _flat(obs: Any) -> list[float] | None:
        try:
            return [float(x) for x in (obs.tolist() if hasattr(obs, "tolist") else obs)]
        except (TypeError, ValueError):
            return None

    def _begin(self, i: int, env: Any, obs: Any) -> None:
        a = env.app
        j = a.journal
        self._open[i] = {"env": env, "seed": int(j["seed"]), "reset_digest": j["reset_digest"], "steps": [],
                         "reset_obs": self._flat(obs) if self.values else None, "params": dict(a.spec.params)}

    def _step(self, i: int, env: Any, out: tuple[Any, float, bool, bool, dict[str, Any]]) -> None:
        ep = self._open.get(i)
        if ep is None:
            return
        j = env.app.journal["steps"][-1]
        note = self._notes.pop(i, {})
        raw = note.get("raw")
        ep["steps"].append(_step_of(len(ep["steps"]), j, tick=int(env.app.journal.get("tick", 0)), raw_action=None if raw is None or raw == j["action"] else raw,
                                    obs=self._flat(out[0]) if self.values else None, logprob=note.get("logprob"), value=note.get("value"),
                                    policy_version=note.get("version")))
        if out[2] or out[3]:
            self._finish(i, "terminated" if out[2] else "truncated")

    def _finish(self, i: int, stop: str) -> None:
        ep = self._open.pop(i)
        env, steps = ep["env"], ep["steps"]
        ret = float(sum(s.reward for s in steps))
        rid = f"{self.run_id}-env{i}-e{self.emitted}"
        rec = ControlRecord(rollout_id=rid, sample=rid, task=getattr(env.task, "id", "control"), harness=self.harness, policy=self.policy,
                            reward=ret, reward_components={"return": ret}, truncated=stop == "truncated", stop=stop, env=env_id(ep["params"]),
                            app=env.app_id, actor=env.actor, config_digest=config_digest(ep["params"]), seed=ep["seed"], reset_digest=ep["reset_digest"],
                            reset_obs=ep["reset_obs"], episode_return=ret, ticks=sum(s.ticks for s in steps), terminated=stop == "terminated", steps=steps)
        if self.keep:
            self.records.append(rec)
        if self.sink is not None:
            self.sink.record(RECORD_KIND, rec, self.emitted)
        self.emitted += 1

    def flush(self) -> None:
        """Emit the episodes still running when the run stops (``stop="run_end"``: they replay, but are not finished)."""
        for i in sorted(self._open):
            if self._open[i]["steps"]:
                self._finish(i, "run_end")
            else:
                self._open.pop(i)
