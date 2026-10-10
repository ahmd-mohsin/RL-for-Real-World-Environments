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
    # circuit breaker (the M10 stop rules): over the last ``breaker_window`` finished rollouts (once ``breaker_min`` have
    # finished), stop when the masked share exceeds ``max_masked_share`` or the harness-bug share exceeds
    # ``max_harness_share`` -- a broken pipeline must stop with its reason, never spin masking everything
    # state plane (checklist #1): a run checkpoint every ``checkpoint_every`` steps at the sample barrier (``train.state``)
    checkpoint_every: int = 0
    checkpoint_dir: str | None = None
    max_masked_share: float = 0.2
    max_harness_share: float = 0.05
    breaker_window: int = 200
    breaker_min: int = 20
    # asynchronous reward (checklist #40): with a ``reward_fn`` the outcome is computed after generation, off the rollout;
    # past ``reward_deadline_s`` the rollout is *late* (counted, still pending: never trained as 0); past
    # ``reward_timeout_s`` it is masked (``reward_timeout``). ``aux_fn`` (turn credit / rubric) past ``aux_deadline_s``
    # falls back to the outcome-only advantage and records ``credit_fallback``.
    # offload (checklist #7, colocated topologies): the trainer's state leaves the accelerator while rollouts run
    # (``cpu`` | ``nvme``; onloaded before each step); ``kv_on_train``: what engines do with their KV cache while the
    # trainer holds the GPU (``discard``: vLLM sleep; ``cpu_pool``: keep it in host memory, where the engine supports it)
    train_state_on_rollout: str = "none"
    kv_on_train: str = "discard"
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
    failures: dict[str, int] = field(default_factory=dict)  # "<blame>:<kind>" -> count (infra / harness), never silent
    failure_examples: list[dict] = field(default_factory=list)  # first tracebacks per kind
    stop_reason: str = "steps"  # steps | wall_clock | masked_share | harness_share
    restarted_rollouts: int = 0  # in flight at a checkpoint, restarted from scratch on resume
    checkpoints: list[str] = field(default_factory=list)

    def series(self, key: str) -> list[float]:
        return [s.metrics[key] for s in self.steps if key in s.metrics]


def _blame(e: BaseException) -> str:
    """Exceptions escaping a rollout: transport / OS / timeouts are infrastructure; anything else is a bug in our own
    harness or record path (a TypeError, a validation error) -- masked either way, but counted separately."""
    infra: tuple[type, ...] = (OSError, TimeoutError, ConnectionError, asyncio.TimeoutError)
    try:
        import httpx

        infra += (httpx.HTTPError,)
    except ImportError:
        pass
    return "infra" if isinstance(e, infra) else "harness"


class Runtime:
    def __init__(self, cfg: RuntimeConfig, sched_cfg: SchedulerConfig, engines: EnginePool, trainer: Any, publisher: Any,
                 env_factory: Callable[[], Any] | None, sources: dict[str, list[str]], *, harness: str = "native:token@1",
                 rollout_fn: Callable | None = None, reward_fn: Callable | None = None, aux_fn: Callable | None = None,
                 on_alarm: Callable | None = None):
        """``rollout_fn(rid, prompt, seed, gen, temperature=...) -> RolloutRecord | None`` replaces the token-env loop
        (e.g. ``world_env.world_rollout``: a Crucible World episode); ``gen(prompt_ids, max_tokens, temperature)`` is
        the engine call with the runtime's abort / version semantics. ``None`` = masked."""
        if cfg.topology == "colocated_sync":
            sched_cfg = SchedulerConfig(**{**sched_cfg.__dict__, "admission_staleness": 1, "staleness_strategy": "none"})
        self.cfg, self.sched_cfg = cfg, sched_cfg
        self.engines, self.trainer, self.publisher, self.env_factory, self.harness = engines, trainer, publisher, env_factory, harness
        self.rollout_fn, self.reward_fn, self.aux_fn = rollout_fn, reward_fn, aux_fn
        self.on_alarm = on_alarm  # checklist #52: a pass-rate jump hands the batch's records to a full-tier re-audit
        self._blocked_since: float | None = None
        self._seeds = 0  # rollout seed counter (part of the run state: a resumed run continues the sequence)
        self.start_step = 0
        from collections import deque

        self._recent: deque[str] = deque(maxlen=cfg.breaker_window)  # ok | masked | infra | harness, per finished rollout
        self.admission_blocked_s = 0.0
        self._queues = {s: list(ps) for s, ps in sources.items()}
        self.pool = DataPool()
        self.sched = Scheduler(sched_cfg, self.pool, {s: (lambda s=s: self._queues[s].pop(0) if self._queues[s] else None) for s in sources})
        self.monitor = StepMonitor()
        self.report = RunReport()
        self.tasks: dict[str, asyncio.Task] = {}
        self.changed = asyncio.Event()
        self.version = 0
        if cfg.train_state_on_rollout not in ("none", "cpu", "nvme"):
            raise ValueError("train_state_on_rollout: none | cpu | nvme")
        if cfg.train_state_on_rollout != "none" and cfg.topology == "disaggregated_stream":
            raise ValueError("disaggregated_stream keeps the trainer resident (it trains while engines generate): no offload")
        for e in engines.engines:
            modes = getattr(e, "kv_modes", ("discard",))
            if cfg.kv_on_train not in modes:
                raise ValueError(f"{type(e).__name__} supports kv_on_train in {tuple(modes)}, not {cfg.kv_on_train!r}")
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
        except EnvInfraError as e:
            self._failure("infra", e)
            return await self._complete(rid, None, failed=True)
        except Exception as e:  # noqa: BLE001 -- masked (never scored 0), but a bug in the harness / record path, recorded
            self._failure(_blame(e), e)
            return await self._complete(rid, None, failed=True)
        await self._complete(rid, rec)

    def _failure(self, blame: str, e: BaseException) -> None:
        import traceback

        key = f"{blame}:{type(e).__name__}"
        self.report.failures[key] = self.report.failures.get(key, 0) + 1
        self._recent.append(blame)
        if self.report.failures[key] <= 2:
            self.report.failure_examples.append({"kind": key, "message": str(e)[:500], "traceback": "".join(traceback.format_exception(e))[-3000:]})

    def _breaker(self) -> str | None:
        n = len(self._recent)
        if n < self.cfg.breaker_min:
            return None
        masked = sum(1 for b in self._recent if b != "ok")
        harness = sum(1 for b in self._recent if b == "harness")
        if harness / n > self.cfg.max_harness_share:
            return "harness_share"
        if masked / n > self.cfg.max_masked_share:
            return "masked_share"
        return None

    async def _complete(self, rid: str, rec: RolloutRecord | None, *, failed: bool = False) -> None:
        """Deliver a finished rollout to the pool, grading it asynchronously first when a ``reward_fn`` is set."""
        if rec is not None and (self.reward_fn is not None or self.aux_fn is not None):
            self.pool.grading(rid)
            self.changed.set()
            rec = await self._grade(rec)
        if rec is None:
            self.pool.finish(rid, reward=None, status="masked", now=time.monotonic())
            self.report.masked_rollouts += 1
            if not failed:  # masked without a recorded exception (grader returned None, reward timeout)
                self._recent.append("masked")
        else:
            self._recent.append("ok")
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
        except EnvInfraError as e:  # infrastructure: masked, never scored 0
            self._failure("infra", e)
            rec = None
        try:
            await self._complete(rid, rec, failed=rec is None)
        finally:
            self.changed.set()

    def _launch_pending(self) -> None:
        for rid, t in self.pool.tags.items():
            if t.status == "pending" and rid not in self.tasks:
                self.pool.start(rid)
                g = self.pool.groups[t.group]
                self.tasks[rid] = asyncio.create_task(self._rollout(rid, g.meta.get("prompt", ""), seed=self._seeds))
                self._seeds += 1

    # ---- control plane --------------------------------------------------------------------------------------------
    def _admit(self) -> None:
        self.sched.admit(self.version)
        now = time.monotonic()
        if self.sched.cap_blocked and self._blocked_since is None:
            self._blocked_since = now
        elif not self.sched.cap_blocked and self._blocked_since is not None:
            self.admission_blocked_s += now - self._blocked_since
            self._blocked_since = None
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
        lps = [lp for r in recs for t in r.turns for lp in (t.logprobs or [])]
        by_group: dict[str, list[float]] = {}
        for r, g in zip(recs, groups, strict=True):
            by_group.setdefault(g, []).append(r.reward)
        return {**m, "_rewards": [r.reward for r in recs], "_lengths": [sum(len(t.completion_token_ids or []) for t in r.turns) for r in recs],
                "_truncated": [bool(r.truncated or any(t.finish == "length" for t in r.turns[-1:])) for r in recs],
                "_group_rewards": list(by_group.values()),
                "_entropy": (-sum(lps) / len(lps)) if lps else None,  # behaviour-policy estimate: E[−log π(a)] over sampled tokens
                "_hacks": sum(1 for r in recs if (r.reward_components or {}).get("hack/confirmed", 0) > 0), "_records": recs}

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

    async def resume(self, path: str | None = None) -> int:
        """Load a run checkpoint (``path``, or the latest under ``cfg.checkpoint_dir``) into this freshly built runtime and
        re-publish the restored weights to the engines at the restored version. Returns the step the run continues after."""
        from crucible.train import state

        ck = path or (state.latest(self.cfg.checkpoint_dir) if self.cfg.checkpoint_dir else None)
        if ck is None:
            raise FileNotFoundError("no run checkpoint to resume from")
        self.start_step = state.restore(self, state.load(ck))
        await self.publisher.restore(self.version)
        return self.start_step

    async def run(self) -> RunReport:
        try:
            return await self._run()
        finally:
            if self.envs is not None:
                self.envs.close()

    async def _run(self) -> RunReport:
        t_start = time.monotonic()
        mark = {"t": t_start, "busy": self.engines.busy_until_now(), "env_wait": 0.0, "env_busy": 0.0}
        step = self.start_step
        offload = self.cfg.train_state_on_rollout
        if offload != "none":
            self.trainer.offload(offload)  # rollouts come first: the engine gets the accelerator
        self._admit()
        while step < self.cfg.steps:
            if time.monotonic() - t_start >= self.cfg.max_wall_s:
                self.report.stop_reason = "wall_clock"
                break
            tripped = self._breaker()
            if tripped:
                self.report.stop_reason = tripped
                break
            self._admit()
            if step == 0 and self.start_step == 0 and self.cfg.topology != "colocated_sync" and self.cfg.warmup_batches:
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
            if offload != "none":
                self.trainer.onload()
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
            if self.cfg.checkpoint_every and step % self.cfg.checkpoint_every == 0 and self.cfg.checkpoint_dir:
                from crucible.train import state

                self.report.checkpoints.append(str(state.save(state.capture(self, step), self.cfg.checkpoint_dir)))
            if offload != "none":
                self.trainer.offload(offload)  # after the publish (which read the weights): free the accelerator for rollouts
            if self.cfg.topology in ("colocated_partial", "colocated_sync"):
                await self.engines.resume()
            rewards, lengths = m.pop("_rewards"), m.pop("_lengths")
            batch = m.pop("_records")
            n_alarms = len(self.monitor.alarms)
            metrics = self.monitor.record(step, staleness=sel.staleness, spans=sel.spans, lengths=lengths, rewards=rewards, truncated=m.pop("_truncated"),
                                          group_rewards=m.pop("_group_rewards"), entropy=m.pop("_entropy"), hacks=m.pop("_hacks"),
                                          worst=sel.worst, evicted_staleness=sel.evicted_staleness,
                                          timing={"train_s": train_s, "publish_s": publish_s, "sample_wait_s": sample_wait},
                                          evicted={k: float(v) for k, v in self.sched.counters.items()})
            if len(self.monitor.alarms) > n_alarms and self.on_alarm is not None:  # checklist #52: re-audit the batch at full tier
                res = self.on_alarm(step, batch)
                if asyncio.iscoroutine(res):
                    await res
                metrics["alarm/reaudit_records"] = float(len(batch))
            now = time.monotonic()
            blocked = self.admission_blocked_s + (now - self._blocked_since if self._blocked_since is not None else 0.0)
            metrics.update({"weights/publish_s": publish_s, "rollout/reprefill_tokens": float(self.report.reprefill_tokens - mark.get("reprefill", 0)),
                            "admission_blocked_s": blocked - mark.get("blocked", 0.0)})
            mark["reprefill"], mark["blocked"] = self.report.reprefill_tokens, blocked
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
