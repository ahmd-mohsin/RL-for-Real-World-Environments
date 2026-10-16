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
| elastic                | B selectable groups                    | elastic group aborted + held; resident group keeps generating | elastic: re-issued / resumed; resident: finish |

Partial-rollout round boundaries (``partial_barrier``, M12-infra A): ``groups`` (B selectable), ``completion_frac`` λ (Kimi
K3: the finished share of untrained rollouts reaches λ, then whatever finished trains, up to B), ``token_budget`` Φ
(Ring-1T C3PO++: Φ completion tokens since the last barrier). ``max_cuts`` σ evicts rollouts cut more than σ times
(Ring's retention), and an engine declaring ``resume_keeps_kv`` resumes token-level calls without re-prefill
(DeepSeek-V4.1). σ counts barriers a *generating* rollout was cut by (a rollout being graded holds no engine work and is
not cut: M12 audit G1 H12a). σ alone does not bound staleness: buffered groups age without being cut, so σ = 0 still
trains staleness 1; τ (``wait`` / ``drop``) is the bound.

Every rollout ends in exactly one of: trained (a scored record, a policy failure included), masked (infrastructure:
reward None, a named ``failure_kind`` counted in ``RunReport.masked_kinds`` and seen by the circuit breaker) or evicted
(staleness, dynamic sampling, backfill, retention). Nothing is lost: any exception after a rollout starts masks it
(``harness:<type>`` unless known infrastructure), and a rollout task that ends with its tag still live is masked by its
done-callback (M12 audit H2 V1: a ``ValueError`` in an environment, a synchronous ``reward_fn``, an engine
``ConnectionError`` or a NaN reward left the tag live forever, uncounted, and stalled the run).
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crucible.core.schema.rollout import RolloutRecord, Segment, Turn  # noqa: F401
from crucible.train.engine import EnginePool, GenRequest, GenResult
from crucible.train.envs import EnvInfraError
from crucible.train.monitors import StepMonitor
from crucible.train.pool import LIVE, DataPool
from crucible.train.scheduler import Scheduler, SchedulerConfig


@dataclass
class RuntimeConfig:
    topology: str = "colocated_partial"  # colocated_sync | colocated_partial | disaggregated_stream | elastic
    # partial-rollout round boundary (M12-infra A; art. §3.2.3.1): ``groups`` = B groups selectable (default);
    # ``completion_frac`` = the finished share of untrained rollouts reaches ``barrier_frac`` λ (Kimi K3), then the
    # finished groups train (up to B); ``token_budget`` = ``barrier_tokens`` Φ completion tokens generated since the last
    # barrier (Ring-1T C3PO++), then the finished groups train. Unfinished rollouts continue on the next version.
    partial_barrier: str = "groups"
    barrier_frac: float = 0.8
    barrier_tokens: int = 0
    # retention σ (Ring-1T): a rollout cut by more than ``max_cuts`` barriers is evicted with its group (its oldest
    # tokens are σ versions behind); None = never
    max_cuts: int | None = None
    # ``elastic`` topology (LongCat DORA): the last ``elastic_engines`` engines form the elastic group, handed to the
    # trainer at each barrier (aborted + paused) and returned after the publish; the others keep generating meanwhile
    elastic_engines: int = 0
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
    keep_records: int | None = 4096  # RunReport.trained_records keeps the newest N (a long run must not hold every record)
    checkpoint_dir: str | None = None
    # a resume refuses a checkpoint trained with other settings (``train.identity``, R-1 step 4) unless every changed
    # setting matches one of these globs (``trainer.lr``, ``scheduler.*``); the change is recorded in later checkpoints
    allow_change: list[str] = field(default_factory=list)
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
    # full-tier audit (DESIGN §10.2, A12; m13 A-6): a deterministic ``audit_rate`` sample of trained rollouts is re-graded
    # by ``audit_fn`` (full tier); fast > 0 and full == 0 is a confirmed hack (or tier divergence) and feeds
    # ``confirmed_hack_rate`` and ``audit/*`` metrics
    audit_rate: float = 0.05
    # metric / record sinks (m13 C-1, ``train.sinks``): step metrics, trained records (``record_rollouts``), failure
    # examples and the final summary; a failing sink is disabled, never fatal
    sinks: list = field(default_factory=list)
    record_rollouts: bool = True
    # group grading stage (m13 C-2, A2-15): ``group_fn(gid, members) -> members`` runs once a group is terminal (all n
    # scored records at once) and before advantages; it may rewrite rewards and set ``RolloutRecord.advantage`` (GAR,
    # tournaments, hack zeroing). It starts as soon as the group is terminal (overlapping other rollouts); at the training
    # barrier it gets what is left of ``group_deadline_s``, then the raw advantages are used (``group_credit_fallback``).
    group_deadline_s: float = 30.0
    # teacher pool (m13 C-5): a finished rollout is scored by its routed teacher (and the reference) before it enters the
    # pool, off the training critical path; past ``teacher_deadline_s`` or on error: ``teacher_on_miss`` = "fallback"
    # (RL signal only, ``opd_fallback``) or "mask"; ``student_vocab_id`` is checked against every teacher at start
    teacher_deadline_s: float = 120.0
    per_source_metrics: bool = False  # reward / length / active-time metrics per source even with one source (always with several)
    teacher_on_miss: str = "fallback"
    student_vocab_id: str | None = None
    reward_deadline_s: float | None = None
    reward_timeout_s: float = 600.0
    aux_deadline_s: float | None = None
    # an engine call aborted *without* a sample barrier in between (an engine error the gateway turned into "abort", an
    # evicted engine) is re-issued at most ``max_call_reissues`` times, with a linear backoff of ``reissue_backoff_s``
    # per attempt (≤ 1 s), then the rollout is masked ``crucible:reissue_limit``; barrier aborts never count (M12 audit
    # H3 V9: a deterministic engine error was re-issued forever and the run stalled until max_wall_s)
    max_call_reissues: int = 32
    reissue_backoff_s: float = 0.05
    heartbeat_s: float = 30.0  # a "heartbeat" record (blocked reason, eviction counters) while no step completes (H2 V9)


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
    audits: int = 0
    confirmed_hacks: int = 0  # full-tier audit rejected what the fast tier paid
    checkpoints: list[str] = field(default_factory=list)
    kv_resumes: int = 0  # aborted calls resumed at a token boundary with the engine's KV kept (no re-prefill)
    retention_evictions: int = 0  # groups evicted because a member was cut more than max_cuts times
    masked_kinds: dict[str, int] = field(default_factory=dict)  # failure_kind -> masked rollouts (every masked rollout, named)
    blocked: dict[str, int] = field(default_factory=dict)  # barrier blocked reason -> count (need_more / wait_stale)
    doomed_evictions: int = 0  # live groups evicted after a publish under ``drop`` (they could no longer train within τ)
    restaged_groups: int = 0  # group stages re-run because refill_member changed the group's scored members (H2 V2)
    identity_check: dict | None = None  # the resumed checkpoint's identity check: same | changed (with the settings) | adopted

    def series(self, key: str) -> list[float]:
        return [s.metrics[key] for s in self.steps if key in s.metrics]


def _blame(e: BaseException) -> str:
    """Exceptions escaping a rollout: transport / OS / timeouts, ``EnvInfraError`` and any exception whose
    ``failure_kind`` the taxonomy blames on infrastructure (``gateway.EngineCallFailed``: ``provider_unavailable``) are
    infrastructure; anything else is a bug in our own harness or record path (a TypeError, a validation error) -- masked
    either way, but counted separately (the harness share stops the run sooner)."""
    from crucible.core.schema.run import default_blame

    kind = getattr(e, "failure_kind", None)
    if isinstance(e, EnvInfraError) or (isinstance(kind, str) and default_blame(kind) == "infra"):
        return "infra"
    infra: tuple[type, ...] = (OSError, TimeoutError, ConnectionError, asyncio.TimeoutError)
    try:
        import httpx

        infra += (httpx.HTTPError,)
    except ImportError:
        pass
    return "infra" if isinstance(e, infra) else "harness"


class CallReissueLimit(EnvInfraError):
    """An engine call aborted more than ``max_call_reissues`` times without a sample barrier: infrastructure, masked.
    ``failure_kind`` lets the World episode loop mask it with its name too."""

    failure_kind = "crucible:reissue_limit"


def _capture_gap(rec: Any) -> str | None:
    """Why a record cannot train for lack of token capture (None: complete). The ``valid_for_training`` gate minus its
    reward part (the reward may still be graded): M12 audit H3 V21 / G2 H9, the runtime never applied it."""
    turns = getattr(rec, "turns", None)
    if turns is None:  # a non-token record (``ControlRecord``): its own gate
        gate = getattr(rec, "valid_for_training", None)
        if gate is None:
            return None
        ok, why = gate()
        return None if ok or why.startswith("masked") else why
    if not turns:
        return "no model calls recorded"
    bad = [(t.call_id, t.missing()) for t in turns if t.missing()]
    if bad:
        return f"token record incomplete for {len(bad)}/{len(turns)} calls (first: {bad[0][0]}: {', '.join(bad[0][1])})"
    return None


def _per_source(x: Any, source: str | None) -> Any:
    """A per-source map (m13 C-6: ``{"code": fn_a, "math": fn_b, "*": default}``) or one value for every source."""
    if isinstance(x, dict):
        return x[source] if source in x else x.get("*")
    return x


class Runtime:
    def __init__(self, cfg: RuntimeConfig, sched_cfg: SchedulerConfig, engines: EnginePool, trainer: Any, publisher: Any,
                 env_factory: Callable[[], Any] | None, sources: dict[str, list[str]], *, harness: str = "native:token@1",
                 rollout_fn: Callable | None = None, reward_fn: Callable | None = None, aux_fn: Callable | None = None,
                 on_alarm: Callable | None = None, audit_fn: Callable | None = None, group_fn: Callable | None = None,
                 teacher_pool: Any = None):
        """``env_factory``, ``harness``, ``rollout_fn``, ``reward_fn`` and ``aux_fn`` may each be a per-source map (``{source:
        value, "*": default}``, m13 C-6): one run mixes token envs, World tasks and judge-graded sources, each record
        tagged with its source.

        ``rollout_fn(rid, prompt, seed, gen, temperature=...) -> RolloutRecord | None`` replaces the token-env loop
        (e.g. ``world_env.world_rollout``: a Crucible World episode); ``gen(prompt_ids, max_tokens, temperature)`` is
        the engine call with the runtime's abort / version semantics. ``None`` = masked."""
        if cfg.topology == "colocated_sync":
            sched_cfg = SchedulerConfig(**{**sched_cfg.__dict__, "admission_staleness": 1, "staleness_strategy": "none"})
        self.cfg, self.sched_cfg = cfg, sched_cfg
        self.engines, self.trainer, self.publisher, self.env_factory, self.harness = engines, trainer, publisher, env_factory, harness
        self.rollout_fn, self.reward_fn, self.aux_fn = rollout_fn, reward_fn, aux_fn
        self.on_alarm = on_alarm  # checklist #52: a pass-rate jump hands the batch's records to a full-tier re-audit
        self.audit_fn = audit_fn
        self.group_fn = group_fn  # m13 C-2: the group grading stage (see RuntimeConfig.group_deadline_s)
        self.teacher_pool = teacher_pool  # m13 C-5
        if teacher_pool is not None:
            if cfg.teacher_on_miss not in ("fallback", "mask"):
                raise ValueError(f"teacher_on_miss: fallback | mask (got {cfg.teacher_on_miss!r})")
            # A2-12: refused before any rollout. The gate was opt-in (skipped when no student id was given), so a teacher
            # with another vocabulary distilled by default (M12 audit H1 V8): the student's id is now required, from the
            # config or the trainer (``trainer.vocab_id``, e.g. ``teachers.vocab_signature(tokenizer)``)
            vocab = cfg.student_vocab_id or getattr(trainer, "vocab_id", None)
            if vocab is None:
                raise ValueError("a teacher_pool needs RuntimeConfig.student_vocab_id (teachers.vocab_signature of the student's tokenizer): "
                                 "teacher and student token log-probs are comparable only over one vocabulary")
            teacher_pool.check_vocab(vocab)
        t_cfg = getattr(getattr(trainer, "cfg", None), "temperature", None)
        if isinstance(t_cfg, int | float) and not isinstance(t_cfg, bool) and float(t_cfg) != float(cfg.temperature):
            # π_θ must be evaluated at the sampling temperature, else r_dis is biased on every token (G1 H5); the trainer
            # refuses such records one by one, the runtime refuses the configuration before any rollout
            raise ValueError(f"trainer temperature {t_cfg} != sampling temperature {cfg.temperature} (RuntimeConfig.temperature): "
                             "the trainer would score tokens under another distribution than the one that sampled them")
        if cfg.version_switch not in ("call", "token"):
            raise ValueError(f"version_switch: call | token (got {cfg.version_switch!r})")
        if cfg.version_switch == "token" and not engines.capabilities().get("token_resume", False):
            # an engine that cannot resume from kept tokens would drop the partial output while the runtime counts a
            # token-level resume (G3 D8: vLLM ignored ``resume_completion``): refused before any rollout
            raise ValueError("version_switch='token' needs engines declaring capabilities()['token_resume']; use version_switch='call'")
        if cfg.checkpoint_every and cfg.checkpoint_every % max(cfg.publish_every, 1):
            # a checkpoint on a step that did not publish stores version v with weights one step newer, and resume
            # serves them as v: version v would name two weight sets (M12 audit H2 V12)
            raise ValueError(f"checkpoint_every ({cfg.checkpoint_every}) must be a multiple of publish_every ({cfg.publish_every}): "
                             "a checkpoint must land on a publish step")
        self._teach_stats = {"scored": 0, "fallback": 0, "seconds": 0.0}
        self._started: dict[str, float] = {}
        self._group_stage: dict[str, tuple[float, asyncio.Task]] = {}
        self._seats: dict[str, list[RolloutRecord]] = {}  # rid -> per-seat records of a game rollout (rollout_fn returned a list)
        self._audit_tasks: set[asyncio.Task] = set()
        self._step_audit = {"runs": 0, "fast_pass_full_fail": 0}
        self._blocked_since: float | None = None
        self._seeds = 0  # rollout seed counter (part of the run state: a resumed run continues the sequence)
        self.start_step = 0
        from collections import deque

        self._recent: deque[str] = deque(maxlen=cfg.breaker_window)  # ok | masked | infra | harness, per finished rollout
        self.admission_blocked_s = 0.0
        self._queues = {s: list(ps) for s, ps in sources.items()}
        from crucible.train.identity import sources_doc

        self._source_identity = sources_doc(sources)  # the prompts as given, not what is left of them (R-1 step 4)
        self._identity_changes: list[dict] = []  # allowed setting changes over this run's resumes, carried by its checkpoints
        self.pool = DataPool()
        self.sched = Scheduler(sched_cfg, self.pool, {s: (lambda s=s: self._queues[s].pop(0) if self._queues[s] else None) for s in sources})
        self.sched.stage_view = self._stage_view  # the zero-variance filter reads the stage's output / every seat (G2 D-T2-4, H2 V20)
        self._barrier_epoch = 0  # +1 at every sample barrier that aborts calls: those aborts are expected, never capped
        self._closing = False
        self._last_beat = time.monotonic()
        self.monitor = StepMonitor()
        self.report = RunReport()
        from crucible.train.sinks import Tracking

        self.tracking = Tracking(cfg.sinks)
        self._step_now = 0
        self.tasks: dict[str, asyncio.Task] = {}
        self.changed = asyncio.Event()
        self.version = 0
        if cfg.topology not in ("colocated_sync", "colocated_partial", "disaggregated_stream", "elastic"):
            raise ValueError(f"unknown topology {cfg.topology!r}")
        if cfg.partial_barrier not in ("groups", "completion_frac", "token_budget"):
            raise ValueError("partial_barrier: groups | completion_frac | token_budget")
        if cfg.partial_barrier == "completion_frac" and not 0 < cfg.barrier_frac <= 1:
            raise ValueError("barrier_frac must be in (0, 1]")
        if cfg.partial_barrier == "token_budget" and cfg.barrier_tokens <= 0:
            raise ValueError("token_budget needs barrier_tokens > 0")
        if cfg.topology == "elastic" and not 0 < cfg.elastic_engines < len(engines.engines):
            raise ValueError(f"elastic needs 0 < elastic_engines < {len(engines.engines)} (a resident generation group must remain)")
        self._round_tokens = 0
        if cfg.train_state_on_rollout not in ("none", "cpu", "nvme"):
            raise ValueError("train_state_on_rollout: none | cpu | nvme")
        if cfg.train_state_on_rollout != "none" and cfg.topology in ("disaggregated_stream", "elastic"):
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
    async def _call(self, rid: str, call_id: str, prompt: list[int], max_tokens: int | None = None, temperature: float | None = None, *,
                    seed: int | None = None, stop_ids: Any = None, source: str | None = None) -> GenResult:
        """One engine call with the runtime's abort / version semantics. ``seed`` / ``stop_ids`` (from an episode's
        ``EngineLLM``: per-call seed, the codec's end-of-turn ids) are forwarded; without ``stop_ids`` the source's default
        stop ids apply (``RuntimeConfig.stop_ids``, else the rollout function's codec ``eos_ids``)."""
        stops = tuple(stop_ids) if stop_ids else self._stop_ids(source)
        req = GenRequest(rid=rid, call_id=call_id, prompt_ids=list(prompt), max_tokens=max_tokens or self.cfg.max_turn_tokens,
                         temperature=self.cfg.temperature if temperature is None else temperature, stop_ids=stops, top_p=self.cfg.top_p, seed=seed)
        epoch, spurious = self._barrier_epoch, 0
        while True:
            res = await self.engines.generate(req)
            self._round_tokens += max(0, len(res.completion_ids) - len(req.resume_completion or []))
            if res.finish != "abort":
                return res
            self.report.aborted_calls += 1
            if self._barrier_epoch != epoch:  # cut by a sample barrier: expected, re-issued on the next version
                epoch = self._barrier_epoch
            else:  # no barrier: an engine error turned into "abort" (H3 V9); bounded, backed off, then masked
                spurious += 1
                if spurious > self.cfg.max_call_reissues:
                    raise CallReissueLimit(f"call {call_id} aborted {spurious} times without a sample barrier (engine error or engines down)")
                if self.cfg.reissue_backoff_s > 0:
                    await asyncio.sleep(min(self.cfg.reissue_backoff_s * spurious, 1.0))
            if self.cfg.version_switch == "token":  # keep the partial output; it continues on the next version
                self.report.resumed_calls += 1
                req.resume_completion, req.resume_logprobs, req.resume_candidates = res.completion_ids, res.logprobs, list(res.candidates or [])
                req.resume_routing = list(res.routing or [])
                req.resume_spans = list(res.version_spans or [(0, res.version)])
                if self.engines.capabilities().get("resume_keeps_kv"):  # DeepSeek-V4.1: resume at the token boundary, no re-prefill
                    self.report.kv_resumes += 1
                else:
                    self.report.reprefill_tokens += len(prompt) + len(res.completion_ids)
            else:  # discard the partial completion; the whole call is re-issued under the new version
                self.report.reprefill_tokens += len(prompt)
                req.resume_completion, req.resume_logprobs, req.resume_spans, req.resume_candidates, req.resume_routing = [], [], [], [], []

    async def _episode_fn(self, rid: str, prompt: str, seed: int) -> tuple[Any, str | None]:
        """A ``rollout_fn`` episode: ``(record, None)`` or ``(None, kind)`` (the function returned None: masked)."""
        tag = self.pool.tags[rid]
        fn, harness = _per_source(self.rollout_fn, tag.source), _per_source(self.harness, tag.source)
        k = [0]

        async def gen(ids: list[int], max_tokens: int, temperature: float, *, seed: int | None = None, stop_ids: Any = None) -> GenResult:
            k[0] += 1
            res = await self._call(rid, f"{rid}:c{k[0]}", ids, max_tokens, temperature, seed=seed, stop_ids=stop_ids, source=tag.source)
            for _, v in (res.version_spans or [(0, res.version)]):
                self.pool.call(rid, v, 0)
            tag.length += len(res.completion_ids)
            return res

        if self.envs is not None:  # the whole episode on a worker loop; its model calls come back to this loop
            rec = await self.envs.run_episode(fn, rid, prompt, seed, self.envs.bridge(gen), temperature=self.cfg.temperature, harness=harness)
        else:
            rec = await fn(rid, prompt, seed, gen, temperature=self.cfg.temperature, harness=harness)
        if rec is None:
            return None, "crucible:rollout_fn_none"
        if isinstance(rec, list):  # a game: one record per trained seat (``train.seats``); the episode stays one pool rollout
            seats = [r for r in rec if r is not None]
            if not seats:
                return None, "crucible:no_seat_outcome"
            self._seats[rid] = seats  # expanded into per-seat groups when the batch is built (``_train``)
            rec = seats[0]
        return rec, None

    async def _episode_env(self, rid: str, prompt: str, seed: int) -> tuple[Any, str | None]:
        """The token-environment loop (``env_factory``)."""
        tag = self.pool.tags[rid]
        factory = _per_source(self.env_factory, tag.source)
        if factory is None:
            raise ValueError(f"source {tag.source!r} has neither a rollout_fn nor an env_factory")
        env = factory()
        obs = await self._env(env.reset, prompt, seed)
        context: list[int] = list(obs)
        prev_full: list[int] | None = None
        turns: list[Turn] = []
        k = 0
        while True:
            k += 1
            res = await self._call(rid, f"{rid}:c{k}", context, source=tag.source)
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
                break
        g = self.pool.groups[tag.group]
        reward, bad = step.reward, None
        if reward is not None and not (isinstance(reward, int | float) and math.isfinite(reward)):
            # a NaN / inf (or non-numeric) environment reward failed the record's construction and lost the rollout
            # (M12 audit H2 V1): it is the environment's fault, never the policy's, so the rollout is masked by name
            reward, bad = None, "crucible:env_nonfinite_reward"
        rec = RolloutRecord(rollout_id=rid, sample=rid, task=g.meta.get("prompt", ""), harness=_per_source(self.harness, tag.source), policy="native", group=tag.group,
                            turns=turns, reward=reward, stop="done", failure_kind=bad, mask_sample=bad is not None)
        rec.reward_components.update(step.info.get("reward_components") or {})  # e.g. the noise curriculum's level (train.noise)
        return rec, None

    async def _rollout(self, rid: str, prompt: str, seed: int) -> None:
        """One rollout, start to pool. Nothing escapes (M12 audit H2 V1): an exception in the episode (the factory, an
        environment, an engine transport error, the record's construction) masks the rollout under
        ``<blame>:<type>`` (an ``EnvInfraError``'s own ``failure_kind`` when it names one); an exception while it is
        delivered (grading, the teacher, the record path) masks it the same way. Either way the exception is recorded
        (``RunReport.failures``, tracebacks) and the breaker sees it."""
        self._started[rid] = time.monotonic()
        rec, kind, failed = None, None, False
        try:
            rec, kind = await (self._episode_fn(rid, prompt, seed) if _per_source(self.rollout_fn, self.pool.tags[rid].source) is not None
                               else self._episode_env(rid, prompt, seed))
        except Exception as e:  # noqa: BLE001 -- masked (never scored 0, never lost), and recorded with its blame
            blame = _blame(e)
            self._failure(blame, e)
            rec, kind, failed = None, getattr(e, "failure_kind", None) or f"{blame}:{type(e).__name__}", True
        try:
            await self._complete(rid, rec, failed=failed, kind=kind)
        except Exception as e:  # noqa: BLE001
            blame = _blame(e)
            self._failure(blame, e)
            self._mask_now(rid, f"{blame}:{type(e).__name__}", failed=True)
        finally:
            self.changed.set()

    def _failure(self, blame: str, e: BaseException, *, rollout: bool = True) -> None:
        """``rollout=False``: a failure outside any rollout (an audit, a monitor hook): recorded, but not in the breaker's
        window of finished rollouts (it masked nothing)."""
        import traceback

        key = f"{blame}:{type(e).__name__}"
        self.report.failures[key] = self.report.failures.get(key, 0) + 1
        if rollout:
            self._recent.append(blame)
        ex = {"kind": key, "message": (str(e) or repr(e))[:500], "traceback": "".join(traceback.format_exception(e))[-3000:]}
        if self.report.failures[key] <= 2:
            self.report.failure_examples.append(ex)
        self.tracking.record("failure", {**ex, "rollout": rollout}, self._step_now)  # every failure, not only the examples

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

    def _mask_now(self, rid: str, kind: str, *, failed: bool) -> None:
        """Pool a live rollout as masked: reward None, never trained, under a named ``failure_kind`` counted in
        ``RunReport.masked_rollouts`` / ``masked_kinds``. The breaker sees it once: through its recorded exception
        (``failed``) or here as ``masked``. Synchronous, so the task done-callback can use it as a backstop."""
        tag = self.pool.tags.get(rid)
        self._started.pop(rid, None)
        self._seats.pop(rid, None)
        if tag is None or tag.status not in LIVE:
            return  # evicted while it ran (retention, doomed, redundancy): counted as evicted, nothing to deliver
        self.pool.finish(rid, reward=None, status="masked", now=time.monotonic())
        self.report.masked_rollouts += 1
        self.report.masked_kinds[kind] = self.report.masked_kinds.get(kind, 0) + 1
        if not failed:
            self._recent.append("masked")
        self._after_finish(rid)
        self.changed.set()

    async def _complete(self, rid: str, rec: Any, *, failed: bool = False, kind: str | None = None) -> None:
        """Deliver a finished rollout to the pool, grading it asynchronously first when a ``reward_fn`` is set.

        A record goes the masked way (never pooled as ``finished``, never trained, never a 0) when:
        - it is None (``kind`` names why);
        - its token capture is incomplete: ``crucible:capture_incomplete``, the ``valid_for_training`` gate minus its
          reward part, applied before any grading (M12 audit H3 V21: the runtime never applied the gate, so a record
          without logprobs or with invalid spans reached ``trainer.step``);
        - it carries ``mask_sample`` or no reward after grading: its own ``failure_kind`` (M12 audit H2 V8 / G2 D-T2-5: a
          custom ``rollout_fn``'s masked record was pooled ``finished`` and counted ok, invisible to the breaker, to group
          backfill and to the zero-variance filter); a game with no seat outcome is ``crucible:no_seat_outcome``."""
        tag = self.pool.tags.get(rid)
        if tag is None or tag.status not in LIVE:
            self._started.pop(rid, None)
            self._seats.pop(rid, None)
            return
        if rec is None:
            return self._mask_now(rid, kind or "crucible:no_record", failed=failed)
        src = tag.source
        rec.source = src
        t0 = self._started.pop(rid, None)
        if t0 is not None:
            rec.timings["active_s"] = time.monotonic() - t0  # rollout wall time (per-source active time, A2-55)
        seats = self._seats.get(rid)
        gap = next((g for g in (_capture_gap(r) for r in (seats or [rec])) if g), None)
        if gap is not None:
            for r in seats or [rec]:
                if not r.mask_sample:
                    r.failure_kind, r.mask_sample = "crucible:capture_incomplete", True
            self.tracking.record("masked", {"rollout": rid, "kind": "crucible:capture_incomplete", "why": gap}, self._step_now)
            return self._mask_now(rid, "crucible:capture_incomplete", failed=False)
        if _per_source(self.reward_fn, src) is not None or _per_source(self.aux_fn, src) is not None:
            if seats is not None:  # a game: every seat's record is graded (the source's reward_fn was skipped: H2 V20)
                self.pool.grading(rid)
                self.changed.set()
                for s in seats:
                    s.source = src
                seats = list(await asyncio.gather(*(self._grade(s) for s in seats if not s.mask_sample)))
                self._seats[rid] = seats or self._seats.get(rid, [])
                rec = seats[0] if seats else rec
            elif not rec.mask_sample:
                self.pool.grading(rid)
                self.changed.set()
                rec = await self._grade(rec)
        if self.teacher_pool is not None and not rec.mask_sample:
            self.pool.grading(rid)
            self.changed.set()
            taught = await self._teach(rec)
            if taught is None:
                rec.failure_kind, rec.mask_sample = "teacher_miss", True
            else:
                rec = taught
        if seats is not None:
            if not any(s.reward is not None and not s.mask_sample for s in seats):
                return self._mask_now(rid, "crucible:no_seat_outcome", failed=False)
        elif rec.mask_sample or rec.reward is None:
            return self._mask_now(rid, rec.failure_kind or "crucible:no_reward", failed=False)
        tag = self.pool.tags.get(rid)
        if tag is None or tag.status not in LIVE:  # evicted while it was graded
            self._seats.pop(rid, None)
            return
        self._recent.append("ok")
        rec.group = tag.group
        self.pool.finish(rid, reward=rec.reward, value=rec, now=time.monotonic())
        self._after_finish(rid)
        self.changed.set()

    def _after_finish(self, rid: str) -> None:
        """A member turned terminal: trim the group's redundant members (``group_redundancy``) and start its group stage."""
        gid = self.pool.tags[rid].group
        for r in self.sched.trim_redundant(gid):
            self._drop_task(r)
        g = self.pool.groups.get(gid)
        if self.group_fn is not None and g is not None and g.status == "terminal":
            self._ensure_stage(gid)

    def _drop_task(self, rid: str) -> None:
        """Stop an evicted rollout: its task is cancelled (its tag is already ``evicted``: nothing is delivered)."""
        task = self.tasks.pop(rid, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        self._started.pop(rid, None)
        self._seats.pop(rid, None)

    # ---- group stage ----------------------------------------------------------------------------------------------
    def _stage_key(self, gid: str) -> tuple[str, ...]:
        return tuple(t.rid for t in self.pool.members(gid, {"finished"}) if t.rid in self.pool.values)

    def _ensure_stage(self, gid: str) -> tuple[float, asyncio.Task, tuple[str, ...]]:
        """The group stage for the group's *current* scored members. A stage launched before ``refill_member`` re-opened
        the group covers only the old members: it is cancelled and run again, so refilled members never skip GAR or hack
        zeroing (M12 audit H2 V2: 17 of 20 trained groups mixed stage and raw members)."""
        key = self._stage_key(gid)
        ent = self._group_stage.get(gid)
        if ent is not None and ent[2] == key:
            return ent
        if ent is not None:
            ent[1].cancel()
            self.report.restaged_groups += 1
        task = asyncio.create_task(self._run_group_fn(gid, key))
        task.add_done_callback(lambda t: t.cancelled() or t.exception())  # retrieved: an unused failure never warns
        ent = (time.monotonic(), task, key)
        self._group_stage[gid] = ent
        return ent

    def _drop_stage(self, gid: str) -> None:
        ent = self._group_stage.pop(gid, None)
        if ent is not None:
            ent[1].cancel()

    async def _run_group_fn(self, gid: str, key: tuple[str, ...]) -> list[RolloutRecord]:
        members = [self.pool.values[r] for r in key]
        res = self.group_fn(gid, [m.model_copy(deep=True) for m in members])
        if asyncio.iscoroutine(res):
            res = await res
        out = list(res)
        if len(out) != len(members) or [r.rollout_id for r in out] != [m.rollout_id for m in members]:
            raise ValueError(f"group_fn for {gid} must return its members in order (got {len(out)} of {len(members)})")
        return out

    def _stage_view(self, gid: str) -> list[float | None] | None:
        """What the zero-variance filter compares for a terminal group (``Scheduler.stage_view``), or None while that is
        not known yet:
        - with a group stage: its output (each member's advantage, else its rewritten reward), None while it runs; a
          failed stage falls back to the raw rewards, as training would (M12 audit G2 D-T2-4: the filter evicted all-pass
          groups a GRS / tournament stage would separate);
        - a game: the group has zero variance only when every seat's records do (seat 0's reward stood for all seats:
          M12 audit H2 V20);
        - else the raw rewards."""
        members = self.pool.members(gid, {"finished", "failure"})
        seat_lists = [self._seats[t.rid] for t in members if t.rid in self._seats]
        if seat_lists:
            by_seat: dict[str, list[float]] = {}
            for ss in seat_lists:
                for s in ss:
                    if s.reward is not None and not s.mask_sample:
                        by_seat.setdefault(str(s.actor), []).append(float(s.reward))
            return [0.0, 1.0] if any(len(set(v)) > 1 for v in by_seat.values()) else [0.0, 0.0]
        raw = [t.reward for t in members]
        if self.group_fn is None:
            return raw
        task = self._ensure_stage(gid)[1]
        if not task.done():
            return None
        if task.cancelled() or task.exception() is not None:
            return raw
        return [r.advantage if r.advantage is not None else r.reward for r in task.result()]

    async def _apply_group_stage(self, gids: list[str]) -> dict[str, float]:
        """At the barrier: install each selected group's stage result, or fall back to the raw advantages when it is late
        or failed (the records are flagged; a late stage is cancelled, never applied to a later step). Results are written
        back by member position, not by ``rollout_id`` (a World record's id is ``<rid>-a1``: every World group stage
        raised ``KeyError``, M12 audit H2 V7)."""
        done = fell = 0
        waited = 0.0
        for gid in gids:
            t0, task, key = self._ensure_stage(gid)  # never launched (restored from a checkpoint) or stale: run it now
            self._group_stage.pop(gid, None)
            left = self.cfg.group_deadline_s - (time.monotonic() - t0)
            w0 = time.monotonic()
            try:
                out = await asyncio.wait_for(asyncio.shield(task), timeout=max(left, 0.0)) if not task.done() else task.result()
            except TimeoutError:
                task.cancel()
                out = None
            except Exception as e:  # noqa: BLE001 -- a broken group operator falls back; it never kills the run
                self._failure("harness", e, rollout=False)
                out = None
            waited += time.monotonic() - w0
            if out is None:
                fell += 1
                for t in self.pool.members(gid, {"finished"}):
                    if t.rid in self.pool.values:
                        self.pool.values[t.rid].group_credit_fallback = True
                continue
            done += 1
            for rid, r in zip(key, out, strict=True):
                r.reward_components.setdefault("raw", float(self.pool.values[rid].reward or 0.0))
                self.pool.values[rid] = r
        return {"group/graded": float(done), "group/fallback": float(fell), "group/barrier_wait_s": waited}

    async def _teach(self, rec: RolloutRecord) -> RolloutRecord | None:
        t0 = time.monotonic()
        try:
            out = await asyncio.wait_for(self.teacher_pool.teach(rec, rec.source), timeout=self.cfg.teacher_deadline_s)
            self._teach_stats["scored"] += 1
        except Exception as e:  # noqa: BLE001 -- late (TimeoutError) or failed teacher: fallback or mask, never a crash
            if not isinstance(e, TimeoutError | asyncio.TimeoutError):
                self._failure(_blame(e), e, rollout=False)
            self._teach_stats["fallback"] += 1
            for t in rec.turns:
                t.teacher_logprobs = t.ref_logprobs = None
            if self.cfg.teacher_on_miss == "mask":
                return None
            rec.opd_fallback = True
            out = rec
        self._teach_stats["seconds"] += time.monotonic() - t0
        return out

    async def _grade(self, rec: RolloutRecord) -> RolloutRecord:
        t0 = time.monotonic()
        try:
            return await self._grade_inner(rec, t0)
        finally:  # m13 C-8 (A2-26): grader compute is a scaling axis (MiMo: 12.7% of cost)
            rec.timings["grade_s"] = rec.timings.get("grade_s", 0.0) + (time.monotonic() - t0)

    async def _grade_inner(self, rec: RolloutRecord, t0: float) -> RolloutRecord:
        """Grade ``rec``. A grader that cannot produce a finite reward masks the record by name (``reward_timeout``,
        ``grader_exception``, ``grader_no_reward``, ``grader_nonfinite``): never a 0, never a lost rollout. A synchronous
        ``reward_fn`` is accepted (its value is used as is; it blocks the loop while it runs). M12 audit H2 V1: a sync
        ``reward_fn`` raised in ``create_task`` outside the ``try`` and a NaN raised on assignment, both losing the rollout."""
        cfg = self.cfg
        aux_fn, reward_fn = _per_source(self.aux_fn, rec.source), _per_source(self.reward_fn, rec.source)
        aux = asyncio.ensure_future(aux_fn(rec)) if aux_fn is not None else None

        def masked(kind: str) -> RolloutRecord:
            rec.failure_kind, rec.mask_sample = kind, True
            if aux is not None:
                aux.cancel()
            return rec

        if reward_fn is not None:
            try:
                res = reward_fn(rec)
                if inspect.isawaitable(res):
                    task = asyncio.ensure_future(res)
                    if cfg.reward_deadline_s is not None:
                        done, _ = await asyncio.wait({task}, timeout=cfg.reward_deadline_s)
                        if not done:
                            self.report.late_rewards += 1  # late: still pending, never a 0
                    reward = await asyncio.wait_for(task, timeout=max(cfg.reward_timeout_s - (time.monotonic() - t0), 0.0))
                else:
                    reward = res
            except TimeoutError:
                self.report.reward_timeouts += 1
                return masked("reward_timeout")
            except Exception as e:  # noqa: BLE001 -- a crashing grader is infrastructure: masked, never scored 0
                self._failure("infra", e, rollout=False)
                return masked("grader_exception")
            if reward is None:
                return masked("grader_no_reward")
            try:
                reward = float(reward)
            except (TypeError, ValueError):
                return masked("grader_nonfinite")
            if not math.isfinite(reward):
                return masked("grader_nonfinite")
            rec.reward = reward
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

    def _stop_ids(self, source: str | None) -> tuple[int, ...]:
        """Default stop ids: ``RuntimeConfig.stop_ids``, else the source's rollout function's codec end-of-turn ids
        (``rollout_fn.codec.eos_ids`` or ``rollout_fn.eos_ids``): without them the engine decoded past EOS and the episode
        cut the completion afterwards."""
        if self.cfg.stop_ids:
            return tuple(self.cfg.stop_ids)
        fn = _per_source(self.rollout_fn, source)
        eos = getattr(getattr(fn, "codec", None), "eos_ids", None) or getattr(fn, "eos_ids", None)
        return tuple(eos) if eos else ()

    def _launch_pending(self) -> None:
        for rid in self.pool.pending():  # indexed, creation order (was a scan over every tag per admission)
            t = self.pool.tags[rid]
            if rid not in self.tasks:
                self.pool.start(rid)
                g = self.pool.groups[t.group]
                task = asyncio.create_task(self._rollout(rid, g.meta.get("prompt", ""), seed=self._seeds))
                task.add_done_callback(lambda _t, rid=rid: self._task_done(rid, _t))
                self.tasks[rid] = task
                self._seeds += 1

    def _task_done(self, rid: str, task: asyncio.Task) -> None:
        """Backstop (M12 audit H2 V1): a rollout task that ended with its tag still live is masked and recorded, so no
        rollout is ever lost, uncounted, holding its group open. Finished tasks are not kept for the run's life."""
        if self.tasks.get(rid) is task:
            del self.tasks[rid]
        if self._closing:
            return
        tag = self.pool.tags.get(rid)
        if tag is None or tag.status not in LIVE:
            return
        exc = None if task.cancelled() else task.exception()
        if exc is not None:
            self._failure(_blame(exc), exc)
        self._mask_now(rid, f"harness:{type(exc).__name__}" if exc is not None else ("harness:cancelled" if task.cancelled() else "harness:lost_rollout"),
                       failed=exc is not None)

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

    async def _audit(self, rec: RolloutRecord) -> None:
        try:
            full = self.audit_fn(rec)
            if asyncio.iscoroutine(full):
                full = await full
        except Exception as e:  # noqa: BLE001 -- an audit failure is recorded, never a verdict
            self._failure("infra", e, rollout=False)
            return
        self.report.audits += 1
        self._step_audit["runs"] += 1
        if full is not None and (rec.reward or 0.0) > 0 and float(full) == 0.0:
            self.report.confirmed_hacks += 1
            self._step_audit["fast_pass_full_fail"] += 1
            rec.reward_components["hack/confirmed"] = 1.0

    def _schedule_audits(self, recs: list[RolloutRecord]) -> None:
        if self.audit_fn is None or self.cfg.audit_rate <= 0:
            return
        from crucible.verify.tiers import audit_sample

        for r in recs:
            if audit_sample(r.rollout_id, rate=self.cfg.audit_rate):
                t = asyncio.create_task(self._audit(r))
                self._audit_tasks.add(t)
                t.add_done_callback(self._audit_tasks.discard)

    async def _train(self, sel) -> dict[str, float]:
        gm = await self._apply_group_stage([g.gid for g in sel.groups]) if self.group_fn is not None else {}
        recs, groups = [], []
        for g in sel.groups:
            for t in self.pool.members(g.gid, {"finished", "failure", "masked"}):
                r = self.pool.values.get(t.rid)
                if r is None:  # masked member (pad backfill): no value, excluded from statistics
                    continue
                seats = self._seats.pop(t.rid, None)
                if seats is not None:  # each seat's records form their own group: a zero-sum edge never becomes credit
                    for sr in seats:
                        if sr.reward is None or sr.mask_sample:
                            continue  # a seat without an outcome (or a masked grade) is masked, never scored 0
                        sr.source, sr.timings = r.source, {**r.timings, **sr.timings}
                        recs.append(sr)
                        groups.append(f"{g.gid}#seat={sr.actor}")
                    continue
                recs.append(r)
                groups.append(g.gid)
        if self.cfg.topology in ("disaggregated_stream", "elastic"):
            m = await asyncio.to_thread(self.trainer.step, recs, groups)  # (resident) engines keep generating meanwhile
        else:
            m = self.trainer.step(recs, groups)
        self.report.trained_records += recs
        if self.cfg.record_rollouts:
            self.tracking.records("rollout", recs, self._step_now + 1)
        self._schedule_audits(recs)
        if self.cfg.keep_records is not None and len(self.report.trained_records) > self.cfg.keep_records:
            del self.report.trained_records[: len(self.report.trained_records) - self.cfg.keep_records]
        lps = [lp for r in recs for t in r.turns for lp in (t.logprobs or [])]
        by_group: dict[str, list[float]] = {}
        for r, g in zip(recs, groups, strict=True):
            by_group.setdefault(g, []).append(r.reward)
        g_s = sum(float(r.timings.get("grade_s", 0.0)) for r in recs)
        if g_s:
            act = sum(float(r.timings.get("active_s", 0.0)) for r in recs)
            gm = {**gm, "grader/s": g_s, "grader/usd": sum(float(r.reward_components.get("grader/usd", 0.0)) for r in recs),
                  "grader/share": g_s / max(g_s + act, 1e-9)}
        by_src: dict[str, list[RolloutRecord]] = {}
        for r in recs:
            by_src.setdefault(r.source or "?", []).append(r)
        if len(by_src) > 1 or self.cfg.per_source_metrics:  # m13 C-6 (A2-55): MiMo saw 90× length / 66× time spreads across sources
            for src_, rs in by_src.items():
                lens = sorted(sum(len(t.completion_token_ids or []) for t in r.turns) for r in rs)
                act = sorted(float(r.timings.get("active_s", 0.0)) for r in rs)
                gm = {**gm, f"reward/mean/{src_}": sum(float(r.reward or 0.0) for r in rs) / len(rs), f"len/mean/{src_}": sum(lens) / len(lens),
                      f"len/p99/{src_}": float(lens[min(len(lens) - 1, int(0.99 * len(lens)))]), f"active_s/mean/{src_}": sum(act) / len(act),
                      f"active_s/p99/{src_}": act[min(len(act) - 1, int(0.99 * len(act)))], f"records/{src_}": float(len(rs))}
        if self.teacher_pool is not None:
            gm = {**gm, "teacher/scored": float(self._teach_stats["scored"]), "teacher/fallback": float(self._teach_stats["fallback"]),
                  "teacher/seconds": self._teach_stats["seconds"]}
            self._teach_stats = {"scored": 0, "fallback": 0, "seconds": 0.0}
        return {**m, **gm, "_rewards": [r.reward for r in recs], "_lengths": [sum(len(t.completion_token_ids or []) for t in r.turns) for r in recs],
                "_truncated": [bool(r.truncated or any(t.finish == "length" for t in r.turns[-1:])) for r in recs],
                "_group_rewards": list(by_group.values()),
                "_entropy": (-sum(lps) / len(lps)) if lps else None,  # behaviour-policy estimate: E[−log π(a)] over sampled tokens
                "_hacks": sum(1 for r in recs if (r.reward_components or {}).get("hack/confirmed", 0) > 0), "_records": recs,
                "_probes": sum(1 for r in recs if (r.reward_components or {}).get("probe/attempted", 0) > 0)}

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
        self.tracking.record("resume", self.report.identity_check, self.start_step)
        await self.publisher.restore(self.version)
        return self.start_step

    async def run(self) -> RunReport:
        status = "error"
        try:
            rep = await self._run()
            status = rep.stop_reason or "steps"
            return rep
        finally:
            await self._shutdown(drain_audits=status != "error")
            r = self.report
            self.tracking.summary({"status": status, "steps": len(r.steps), "version": self.version, "publishes": r.publishes,
                                   "masked_rollouts": r.masked_rollouts, "failures": dict(r.failures), "restarted_rollouts": r.restarted_rollouts,
                                   "checkpoints": list(r.checkpoints), "audits": r.audits, "confirmed_hacks": r.confirmed_hacks,
                                   "sinks_failed": dict(self.tracking.failed)})
            self.tracking.finish(status)

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
            if not self._round_boundary():
                await self._wait()
                continue
            sel = self.sched.select(self.version, min_groups=None if self.cfg.partial_barrier == "groups" else 1)
            if sel.blocked:
                self.report.blocked[sel.blocked] = self.report.blocked.get(sel.blocked, 0) + 1
                self._release_evicted()
                self._heartbeat(sel.blocked)
                self._admit()
                await self._wait()
                continue
            sample_wait = time.monotonic() - t_wait
            aborted = 0
            if self.cfg.topology in ("colocated_partial", "colocated_sync"):
                # S1: hand the GPU to the trainer; Worlds keep their state. Pause (close admission) *before* the abort: an
                # aborted call re-issued in the gap would otherwise start again on the old version and, on an engine whose
                # pause does not abort, run during training (M12 audit G3 D10, pick 7: slime / AReaL pause first)
                self._barrier_epoch += 1
                await self.engines.pause()
                aborted = await self.engines.abort_all()
            elif self.cfg.topology == "elastic":  # only the elastic group trains; the resident group keeps generating
                self._barrier_epoch += 1
                n = len(self.engines.engines)
                aborted = await self.engines.hold(list(range(n - self.cfg.elastic_engines, n)))
            self._round_tokens = 0
            if self.cfg.topology in ("colocated_partial", "elastic"):
                self._cut_live()
            t0 = time.monotonic()
            if offload != "none":
                self.trainer.onload()
            m = await self._train(sel)
            train_s = time.monotonic() - t0
            self.sched.trained(sel)
            for g in list(sel.groups) + [self.pool.groups[x] for x in sel.evicted if x in self.pool.groups]:
                self._forget(g.gid)  # trained / evicted work leaves the pool: memory stays bounded over a long run
            self._release_evicted()  # evicted at blocked barriers or with a straggler still running
            step += 1
            publish_s = 0.0
            if step % self.cfg.publish_every == 0:  # S2
                t1 = time.monotonic()
                self._barrier_epoch += 1  # an engine that aborts calls on a weight load: expected, not an engine error
                self.version = await self.publisher.publish()
                publish_s = time.monotonic() - t1
                self.report.publishes += 1
                self._evict_doomed()
            if self.cfg.checkpoint_every and step % self.cfg.checkpoint_every == 0 and self.cfg.checkpoint_dir:
                from crucible.train import state

                self.report.checkpoints.append(str(state.save(state.capture(self, step), self.cfg.checkpoint_dir, trainer=self.trainer)))
            if offload != "none":
                self.trainer.offload(offload)  # after the publish (which read the weights): free the accelerator for rollouts
            if self.cfg.topology in ("colocated_partial", "colocated_sync"):
                await self.engines.resume()
            elif self.cfg.topology == "elastic":
                await self.engines.release()
            rewards, lengths = m.pop("_rewards"), m.pop("_lengths")
            batch = m.pop("_records")
            probes = m.pop("_probes")
            audit_hacks = self._step_audit["fast_pass_full_fail"]
            n_alarms = len(self.monitor.alarms)
            # the detector's confirmations over the batch, the audit's over the audits run (M12 audit H2 V13)
            metrics = self.monitor.record(step, staleness=sel.staleness, spans=sel.spans, lengths=lengths, rewards=rewards, truncated=m.pop("_truncated"),
                                          group_rewards=m.pop("_group_rewards"), entropy=m.pop("_entropy"), hacks=m.pop("_hacks"),
                                          audit_runs=self._step_audit["runs"], audit_confirmed=audit_hacks,
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
            metrics.update({"audit/runs": float(self._step_audit["runs"]), "audit/fast_pass_full_fail": float(audit_hacks),
                            "probe/attempt_share": probes / max(len(rewards), 1)})
            self._step_audit = {"runs": 0, "fast_pass_full_fail": 0}
            metrics.update({k: v for k, v in m.items()})
            metrics.update({"reward/mean": sum(rewards) / max(len(rewards), 1), "aborted_calls": float(aborted), "untrained": float(self.sched.untrained_count()),
                            "batch/groups": float(len(sel.groups)), "rollout/retention_evictions": float(self.report.retention_evictions)})
            metrics.update(self._bubbles(mark, train_s, publish_s))
            metrics.update({f"masked/{k}": float(v) for k, v in self.report.masked_kinds.items()})
            self.report.steps.append(StepLog(step, self.version, metrics))
            self._step_now = step
            self._last_beat = time.monotonic()
            self.tracking.log(metrics, step)
        return self.report

    async def _shutdown(self, *, drain_audits: bool) -> None:
        """End of run (also on an error): stages of groups never trained are cancelled, audits already started finish
        (their verdicts belong to this run), rollout tasks are cancelled and awaited, then the environment workers cancel
        and await what still runs on their loops before they stop (M12 audit H2 V18: ``close`` alone stopped the loops
        under the episodes, so ``World.close`` and temp-dir cleanup never ran for 4 of 12 episodes)."""
        self._closing = True
        for ent in self._group_stage.values():
            ent[1].cancel()
        if self._audit_tasks:
            if not drain_audits:
                for t in self._audit_tasks:
                    t.cancel()
            await asyncio.gather(*list(self._audit_tasks), return_exceptions=True)
        tasks = list(self.tasks.values())
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.envs is not None:
            await self.envs.aclose()

    def _round_boundary(self) -> bool:
        """Whether the partial-rollout round may end now (``partial_barrier``); ``groups`` defers to the scheduler."""
        c = self.cfg
        if c.partial_barrier == "token_budget":
            return self._round_tokens >= c.barrier_tokens or not any(not t.done() for t in self.tasks.values())
        if c.partial_barrier == "completion_frac":
            untrained = [t for t in self.pool.untrained() if t.status != "evicted"]
            done = sum(1 for t in untrained if t.status in ("finished", "failure", "masked"))
            return not untrained or done / len(untrained) >= c.barrier_frac
        return True

    def _cut_live(self) -> None:
        """Every rollout still running at a barrier is cut once more (Ring-1T's retention counter ρ); with ``max_cuts``
        σ, a rollout cut more than σ times is evicted with its group and its task cancelled (freed slots are refilled)."""
        over: set[str] = set()
        for t in self.pool.live():
            if t.status == "running":  # a rollout being graded holds no engine work: never cut (M12 audit G1 H12a)
                t.cuts += 1
                if self.cfg.max_cuts is not None and t.cuts > self.cfg.max_cuts:
                    over.add(t.group)
        for gid in over:
            for rid in self.pool.evict_group(gid):
                self._drop_task(rid)
            self._drop_stage(gid)
            self.sched.counters["evicted_retention"] = self.sched.counters.get("evicted_retention", 0) + 1
            self.report.retention_evictions += 1

    def _forget(self, gid: str) -> None:
        """Release a trained or evicted group and everything the runtime keeps for it (seats, group stage)."""
        g = self.pool.groups.get(gid)
        if g is not None:
            for rid_ in g.members:
                self._seats.pop(rid_, None)
        self._drop_stage(gid)  # evicted while its stage ran: the result would have nothing to apply to
        self.pool.release(gid)

    def _release_evicted(self) -> None:
        """Evicted groups leave the pool with their seats and group stages, at every barrier, blocked ones included
        (M12 audit H2 V16: entries of groups evicted at blocked barriers leaked, 43 after 30 steps)."""
        for gid in [x for x, gr in self.pool.groups.items() if gr.status == "evicted"]:
            self._forget(gid)

    def _evict_doomed(self) -> None:
        """``drop`` after a publish: live groups that can no longer train within τ are evicted now and their rollouts
        cancelled, so their compute and slots go to admissible work (M12 audit G4 pick 9; prime-rl cancels doomed
        in-flight groups before the weight swap). Counted in ``RunReport.doomed_evictions`` and ``evicted_doomed``."""
        before = self.sched.counters.get("evicted_doomed", 0)
        for rid in self.sched.evict_doomed(self.version):
            self._drop_task(rid)
        self.report.doomed_evictions += self.sched.counters.get("evicted_doomed", 0) - before
        self._release_evicted()

    def _heartbeat(self, blocked: str) -> None:
        """While no step completes, a ``heartbeat`` record every ``heartbeat_s`` with the blocked reason and the eviction
        counters: a run spinning on dynamic sampling or a stale wait is visible (M12 audit H2 V9: 650 prompts consumed,
        0 metric lines)."""
        now = time.monotonic()
        if now - self._last_beat < self.cfg.heartbeat_s:
            return
        self._last_beat = now
        self.tracking.record("heartbeat", {"blocked": blocked, "blocked_counts": dict(self.report.blocked), "untrained": self.sched.untrained_count(),
                                           "live": self.pool.live_count(), "counters": dict(self.sched.counters), "masked": dict(self.report.masked_kinds)},
                             self._step_now)

    async def _wait(self) -> None:
        self.changed.clear()
        try:
            await asyncio.wait_for(self.changed.wait(), timeout=max(self.cfg.poll_s, 0.05))
        except TimeoutError:
            pass
