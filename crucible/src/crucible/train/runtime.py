"""The M12 runtime: the asyncio control loop over the scheduler, the engines, the environments, the trainer and the
weight publisher (research m11-m12/01–06). The *decisions* are the scheduler's (``scheduler.py``, the same object the
discrete-event simulator drives); this file only executes them with real concurrency.

Planes and their synchronization points:

- **rollout plane**: one asyncio task per rollout (dispatch unit = one rollout). Model calls go to the engine pool
  (sticky routing); environment steps run on the event loop or, with ``env_workers=N``, on an ``EnvWorkerPool``
  (a blocked environment never stalls the engines or the sample barrier) and **never wait for weights** (only model
  calls do). Every call is stamped with the version that served it; an aborted call is re-issued whole
  (``version_switch="call"``, the default: call-level version purity) or resumed with its partial output on the new
  version (``"token"``, recorded as ``version_spans``: AReaL).
- **data plane**: the ``DataPool`` (tags for scheduling, values = ``crucible.rollout/v1`` records).
- **control plane**: admit (S×B·n untrained cap, per-source caps), select at the sample barrier S1, train, publish S2.
- **weight plane**: ``publish()`` → +1 version → engines load it.

Topologies (logical; *placement* is separate: one GPU can host ``disaggregated_stream`` by splitting its memory —
vLLM at a fixed fraction, trainer resident — research m11-m12/03 finding 1):

| topology               | S1 sample barrier                      | engines during training                 | S3 in-flight calls at publish |
|------------------------|----------------------------------------|-----------------------------------------|-------------------------------|
| colocated_sync         | the whole admitted batch is terminal   | idle (nothing admitted)                 | none                          |
| colocated_partial      | B selectable groups                    | **aborted + paused** (GPU handed over)  | aborted → re-issued / resumed |
| disaggregated_stream   | B selectable groups                    | **keep generating** (trainer in a thread)| finish on their version       |
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crucible.train.engine import EnginePool, GenRequest, GenResult
from crucible.train.envs import EnvInfraError
from crucible.train.monitors import StepMonitor
from crucible.train.pool import DataPool
from crucible.train.rollout import RolloutRecord, Segment, Turn  # noqa: F401
from crucible.train.scheduler import Scheduler, SchedulerConfig


@dataclass
class RuntimeConfig:
    topology: str = "colocated_partial"  # colocated_sync | colocated_partial | disaggregated_stream
    steps: int = 10
    publish_every: int = 1
    warmup_batches: int = 1
    version_switch: str = "call"  # call | token
    max_turn_tokens: int = 8
    temperature: float = 1.0
    top_p: float = 1.0  # checklist #37: default 1.0; < 1 records each token's nucleus for the trainer's renormalization
    stop_ids: tuple[int, ...] = ()
    poll_s: float = 0.002
    max_wall_s: float = 600.0
    env_workers: int = 0  # 0: environments run inline on the event loop; N: an EnvWorkerPool of N threads (checklist #8)
    # asynchronous reward (checklist #40): with a ``reward_fn`` the outcome is computed after generation, off the rollout;
    # past ``reward_deadline_s`` the rollout is *late* (counted, still pending: never trained as 0); past
    # ``reward_timeout_s`` it is masked (``reward_timeout``). ``aux_fn`` (turn credit / rubric) past ``aux_deadline_s``
    # falls back to the outcome-only advantage and records ``credit_fallback``.
    reward_deadline_s: float | None = None
    reward_timeout_s: float = 600.0
    aux_deadline_s: float | None = None


@dataclass
class StepLog:
    step: int
    version: int
    metrics: dict[str, float]


@dataclass
class RunReport:
    steps: list[StepLog] = field(default_factory=list)
    aborted_calls: int = 0
    resumed_calls: int = 0
    reprefill_tokens: int = 0
    publishes: int = 0
    max_untrained: int = 0
    masked_rollouts: int = 0
    late_rewards: int = 0
    reward_timeouts: int = 0
    credit_fallbacks: int = 0
    trained_records: list[RolloutRecord] = field(default_factory=list)

    def series(self, key: str) -> list[float]:
        return [s.metrics[key] for s in self.steps if key in s.metrics]


class Runtime:
    def __init__(self, cfg: RuntimeConfig, sched_cfg: SchedulerConfig, engines: EnginePool, trainer: Any, publisher: Any,
                 env_factory: Callable[[], Any] | None, sources: dict[str, list[str]], *, harness: str = "native:token@1",
                 rollout_fn: Callable | None = None, reward_fn: Callable | None = None, aux_fn: Callable | None = None):
        """``rollout_fn(rid, prompt, seed, gen, temperature=...) -> RolloutRecord | None`` replaces the token-env loop
        (e.g. ``world_env.world_rollout``: a Crucible World episode); ``gen(prompt_ids, max_tokens, temperature)`` is
        the engine call with the runtime's abort / version semantics. ``None`` = masked."""
        if cfg.topology == "colocated_sync":
            sched_cfg = SchedulerConfig(**{**sched_cfg.__dict__, "admission_staleness": 1, "staleness_strategy": "none"})
        self.cfg, self.sched_cfg = cfg, sched_cfg
        self.engines, self.trainer, self.publisher, self.env_factory, self.harness = engines, trainer, publisher, env_factory, harness
        self.rollout_fn, self.reward_fn, self.aux_fn = rollout_fn, reward_fn, aux_fn
        self._queues = {s: list(ps) for s, ps in sources.items()}
        self.pool = DataPool()
        self.sched = Scheduler(sched_cfg, self.pool, {s: (lambda s=s: self._queues[s].pop(0) if self._queues[s] else None) for s in sources})
        self.monitor = StepMonitor()
        self.report = RunReport()
        self.tasks: dict[str, asyncio.Task] = {}
        self.changed = asyncio.Event()
        self.version = 0
        self.envs = None
        if cfg.env_workers:
            from crucible.train.envpool import EnvWorkerPool

            self.envs = EnvWorkerPool(cfg.env_workers)

    async def _env(self, fn, *args):
        """An environment coroutine: on a worker loop when the pool is on, else inline."""
        return await (self.envs.run(fn, *args) if self.envs is not None else fn(*args))

    # ---- rollout plane --------------------------------------------------------------------------------------------
    async def _call(self, rid: str, call_id: str, prompt: list[int], max_tokens: int | None = None, temperature: float | None = None) -> GenResult:
        req = GenRequest(rid=rid, call_id=call_id, prompt_ids=list(prompt), max_tokens=max_tokens or self.cfg.max_turn_tokens,
                         temperature=self.cfg.temperature if temperature is None else temperature, stop_ids=self.cfg.stop_ids, top_p=self.cfg.top_p)
        while True:
            res = await self.engines.generate(req)
            if res.finish != "abort":
                return res
            self.report.aborted_calls += 1
            if self.cfg.version_switch == "token":  # keep the partial output; it continues on the next version
                self.report.resumed_calls += 1
                req.resume_completion, req.resume_logprobs, req.resume_candidates = res.completion_ids, res.logprobs, list(res.candidates or [])
                req.resume_routing = list(res.routing or [])
                req.resume_spans = list(res.version_spans or [(0, res.version)])
                self.report.reprefill_tokens += len(prompt) + len(res.completion_ids)
            else:  # discard the partial completion; the whole call is re-issued under the new version
                self.report.reprefill_tokens += len(prompt)
                req.resume_completion, req.resume_logprobs, req.resume_spans, req.resume_candidates, req.resume_routing = [], [], [], [], []

    async def _rollout_fn(self, rid: str, prompt: str, seed: int) -> None:
        tag = self.pool.tags[rid]
        k = [0]

        async def gen(ids: list[int], max_tokens: int, temperature: float) -> GenResult:
            k[0] += 1
            res = await self._call(rid, f"{rid}:c{k[0]}", ids, max_tokens, temperature)
            for _, v in (res.version_spans or [(0, res.version)]):
                self.pool.call(rid, v, 0)
            tag.length += len(res.completion_ids)
            return res

        try:
            if self.envs is not None:  # the whole episode on a worker loop; its model calls come back to this loop
                rec = await self.envs.run_episode(self.rollout_fn, rid, prompt, seed, self.envs.bridge(gen), temperature=self.cfg.temperature,
                                                  harness=self.harness)
            else:
                rec = await self.rollout_fn(rid, prompt, seed, gen, temperature=self.cfg.temperature, harness=self.harness)
        except EnvInfraError:
            rec = None
        except Exception:  # noqa: BLE001 -- a crashing environment is infrastructure: mask, never score 0
            rec = None
        await self._complete(rid, rec)

    async def _complete(self, rid: str, rec: RolloutRecord | None) -> None:
        """Deliver a finished rollout to the pool, grading it asynchronously first when a ``reward_fn`` is set."""
        if rec is not None and (self.reward_fn is not None or self.aux_fn is not None):
            self.pool.grading(rid)
            self.changed.set()
            rec = await self._grade(rec)
        if rec is None:
            self.pool.finish(rid, reward=None, status="masked", now=time.monotonic())
            self.report.masked_rollouts += 1
        else:
            rec.group = self.pool.tags[rid].group
            self.pool.finish(rid, reward=rec.reward, value=rec, now=time.monotonic())
        self.changed.set()

    async def _grade(self, rec: RolloutRecord) -> RolloutRecord | None:
        cfg = self.cfg
        t0 = time.monotonic()
        aux = asyncio.create_task(self.aux_fn(rec)) if self.aux_fn is not None else None
        if self.reward_fn is not None:
            task = asyncio.create_task(self.reward_fn(rec))
            try:
                if cfg.reward_deadline_s is not None:
                    done, _ = await asyncio.wait({task}, timeout=cfg.reward_deadline_s)
                    if not done:
                        self.report.late_rewards += 1  # late: still pending, never a 0
                reward = await asyncio.wait_for(task, timeout=max(cfg.reward_timeout_s - (time.monotonic() - t0), 0.0))
            except TimeoutError:
                self.report.reward_timeouts += 1
                rec.failure_kind, rec.mask_sample = "reward_timeout", True
                if aux is not None:
                    aux.cancel()
                return None
            except Exception:  # noqa: BLE001 -- a crashing grader is infrastructure: masked, never scored 0
                rec.failure_kind, rec.mask_sample = "grader_exception", True
                if aux is not None:
                    aux.cancel()
                return None
            if reward is None:
                return None
            rec.reward = float(reward)
        if aux is not None:
            try:
                left = None if cfg.aux_deadline_s is None else max(cfg.aux_deadline_s - (time.monotonic() - t0), 0.0)
                credit = await asyncio.wait_for(aux, timeout=left)
                for t in rec.turns:
                    t.advantage = (credit or {}).get(t.call_id, t.advantage)
            except (TimeoutError, Exception):  # noqa: BLE001 -- a late or failing auxiliary channel never blocks the outcome
                aux.cancel()
                rec.credit_fallback = True
                self.report.credit_fallbacks += 1
                for t in rec.turns:
                    t.advantage = None
        return rec

    async def _rollout(self, rid: str, prompt: str, seed: int) -> None:
        if self.rollout_fn is not None:
            return await self._rollout_fn(rid, prompt, seed)
        env = self.env_factory()
        tag = self.pool.tags[rid]
        try:
            obs = await self._env(env.reset, prompt, seed)
            context: list[int] = list(obs)
            prev_full: list[int] | None = None
            turns: list[Turn] = []
            reward = None
            k = 0
            while True:
                k += 1
                res = await self._call(rid, f"{rid}:c{k}", context)
                delta = context[len(prev_full):] if prev_full is not None and context[: len(prev_full)] == prev_full else context
                turns.append(Turn(call_id=f"{rid}:c{k}", actor="agent", prompt_is_delta=prev_full is not None, prompt_token_ids=list(delta),
                                  completion_token_ids=res.completion_ids, logprobs=res.logprobs, policy_version=res.version,
                                  version_spans=res.version_spans, finish=res.finish, candidates=res.candidates, routing=res.routing,
                                  sampling={"temperature": self.cfg.temperature, "top_p": self.cfg.top_p}))
                for _, v in (res.version_spans or [(0, res.version)]):
                    self.pool.call(rid, v, 0)
                tag.length += len(res.completion_ids)
                prev_full = context + res.completion_ids
                step = await self._env(env.step, res.completion_ids)
                context = prev_full + step.obs_ids
                if step.done:
                    reward = step.reward
                    break
            g = self.pool.groups[tag.group]
            rec = RolloutRecord(rollout_id=rid, sample=rid, task=g.meta.get("prompt", ""), harness=self.harness, policy="native", group=tag.group,
                                turns=turns, reward=reward, stop="done")
        except EnvInfraError:  # infrastructure: masked, never scored 0
            rec = None
        try:
            await self._complete(rid, rec)
        finally:
            self.changed.set()

    def _launch_pending(self) -> None:
        for rid, t in self.pool.tags.items():
            if t.status == "pending" and rid not in self.tasks:
                self.pool.start(rid)
                g = self.pool.groups[t.group]
                self.tasks[rid] = asyncio.create_task(self._rollout(rid, g.meta.get("prompt", ""), seed=len(self.tasks)))

    # ---- control plane --------------------------------------------------------------------------------------------
    def _admit(self) -> None:
        self.sched.admit(self.version)
        self._launch_pending()
        self.report.max_untrained = max(self.report.max_untrained, self.sched.untrained_count())

    async def _train(self, sel) -> dict[str, float]:
        recs, groups = [], []
        for g in sel.groups:
            for t in self.pool.members(g.gid, {"finished", "failure", "masked"}):
                r = self.pool.values.get(t.rid)
                if r is None:  # masked member (pad backfill): no value, excluded from statistics
                    continue
                recs.append(r)
                groups.append(g.gid)
        if self.cfg.topology == "disaggregated_stream":
            m = await asyncio.to_thread(self.trainer.step, recs, groups)  # engines keep generating meanwhile
        else:
            m = self.trainer.step(recs, groups)
        self.report.trained_records += recs
        return {**m, "_rewards": [r.reward for r in recs], "_lengths": [sum(len(t.completion_token_ids or []) for t in r.turns) for r in recs]}

    def _bubbles(self, mark: dict, train_s: float, publish_s: float) -> dict[str, float]:
        """Per-step bubbles (checklist #9) over the window since the previous step: engine idle (no call in flight anywhere:
        paused for training, or starved), trainer idle (not training or publishing), and the time environments waited
        for a worker."""
        now = time.monotonic()
        wall = now - mark["t"]
        busy = self.engines.busy_until_now() - mark["busy"]
        env_wait = (self.envs.wait_s if self.envs else 0.0) - mark["env_wait"]
        env_busy = (self.envs.busy_s if self.envs else 0.0) - mark["env_busy"]
        mark.update(t=now, busy=self.engines.busy_until_now(), env_wait=self.envs.wait_s if self.envs else 0.0, env_busy=self.envs.busy_s if self.envs else 0.0)
        return {"idle/rollout_s": max(wall - busy, 0.0), "idle/train_s": max(wall - train_s - publish_s, 0.0), "idle/env_wait_s": env_wait,
                "env/busy_s": env_busy, "step/wall_s": wall, "engine/utilization": busy / wall if wall > 0 else 0.0}

    async def run(self) -> RunReport:
        try:
            return await self._run()
        finally:
            if self.envs is not None:
                self.envs.close()

    async def _run(self) -> RunReport:
        t_start = time.monotonic()
        mark = {"t": t_start, "busy": self.engines.busy_until_now(), "env_wait": 0.0, "env_busy": 0.0}
        step = 0
        self._admit()
        while step < self.cfg.steps and time.monotonic() - t_start < self.cfg.max_wall_s:
            self._admit()
            if step == 0 and self.cfg.topology != "colocated_sync" and self.cfg.warmup_batches:
                ready = len(self.pool.terminal_groups())
                if ready < self.sched_cfg.B * (1 + self.cfg.warmup_batches) and any(not t.done() for t in self.tasks.values()):
                    await self._wait()
                    continue
            t_wait = time.monotonic()
            sel = self.sched.select(self.version)
            if sel.blocked:
                self._admit()
                await self._wait()
                continue
            sample_wait = time.monotonic() - t_wait
            aborted = 0
            if self.cfg.topology in ("colocated_partial", "colocated_sync"):
                aborted = await self.engines.abort_all()  # S1: hand the GPU to the trainer; Worlds keep their state
                await self.engines.pause()
            t0 = time.monotonic()
            m = await self._train(sel)
            train_s = time.monotonic() - t0
            self.sched.trained(sel)
            step += 1
            publish_s = 0.0
            if step % self.cfg.publish_every == 0:  # S2
                t1 = time.monotonic()
                self.version = await self.publisher.publish()
                publish_s = time.monotonic() - t1
                self.report.publishes += 1
            if self.cfg.topology in ("colocated_partial", "colocated_sync"):
                await self.engines.resume()
            rewards, lengths = m.pop("_rewards"), m.pop("_lengths")
            metrics = self.monitor.record(step, staleness=sel.staleness, spans=sel.spans, lengths=lengths, rewards=rewards,
                                          timing={"train_s": train_s, "publish_s": publish_s, "sample_wait_s": sample_wait},
                                          evicted={k: float(v) for k, v in self.sched.counters.items()})
            metrics.update({k: v for k, v in m.items()})
            metrics.update({"reward/mean": sum(rewards) / max(len(rewards), 1), "aborted_calls": float(aborted), "untrained": float(self.sched.untrained_count())})
            metrics.update(self._bubbles(mark, train_s, publish_s))
            self.report.steps.append(StepLog(step, self.version, metrics))
        for t in self.tasks.values():
            t.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        return self.report

    async def _wait(self) -> None:
        self.changed.clear()
        try:
            await asyncio.wait_for(self.changed.wait(), timeout=max(self.cfg.poll_s, 0.05))
        except TimeoutError:
            pass
