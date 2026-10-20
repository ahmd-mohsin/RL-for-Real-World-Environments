"""The Tinker RL loop: Crucible environments sample, Tinker trains (swarm-pipeline/09 §5; mirrors ``bridges/slime``).

One **step**:

1. **Groups.** Each group is ``group_size`` attempts of one task from ``make_env`` (the native harness, the checkpoint
   sidecar, ``InfraWatch``, honesty, the tool limit, the two timeouts), run by the runtime ``Runner`` in ``train`` mode (an
   incomplete token record masks the attempt) with the policy ``LLMToolPolicy`` on the ``TinkerSampler``. Attempts are
   not retried inside a group (``RetryRule(max_attempts=1)``): a failed member drops the group and a fresh group is run.
2. **Filters**, in order (the slime bridge's ``filters.group_filter``, the same rules):
   - ``objective_changed``: the task's verifier version is not the one the run pinned (00 §11.3 item 6: rewards of two
     verifier versions are never mixed in one curve): the group is dropped and the run **stops**; continue from a
     checkpoint taken before the change as a new run;
   - ``infra_in_group``: any member masked (infrastructure, a dead World, the cap, an unmeasured grade): the whole
     group is dropped (never a partial group, never a 0) and refilled (``drop_infra_groups``'s rule);
   - ``stale``: the oldest weight version that produced a token of the group is more than ``max_staleness`` optimizer
     steps behind the trainer: **dropped** (slime only logs staleness) and refilled. Checked when the group lands and
     again when the batch is built;
   - ``zero_spread`` (``drop_zero_spread``): every member has the same reward: no signal (slime's nonzero-std filter).
   Every drop is a row in ``drops.jsonl``; a step that cannot fill its batch within ``max_groups_per_step`` launches
   stops the run (``too_many_drops``) instead of looping.
3. **Datums** (``datums.build_datums``): outcome advantage over each group's live rollouts, the honesty term composed in,
   M6's per-turn credit, one datum per TITO sequence; ``loss_scale``.
4. **Update.** ``forward_backward(loss_fn)`` (``importance_sampling`` | ``ppo`` | ``cispo``, ``loss_fn_config`` e.g.
   PPO's clip thresholds) -> ``optim_step(learning_rate)`` -> ``save_weights_for_sampler`` -> the sampler samples the new
   weights (version = step). ``save_state`` (weights + optimizer) every ``save_every`` steps and at a stop.

**Off-policy bound.** ``async_groups`` > 0 keeps that many extra groups sampling while a step trains (their tokens come
from the previous weights); ``max_staleness`` bounds how old a trained group may be. ``async_groups=0,
max_staleness=0`` is synchronous on-policy RL.

**Resume (requeue-safe).** The run directory holds ``config.json`` (the fixed configuration: a resume with a different
one is refused, except ``steps`` and ``cap_usd``), ``state.json`` (written atomically after each ``save_state``: the step,
the Tinker state path, the sampler path and version of that step, the objective), ``versions.json`` (version -> sampler
path), ``spend.jsonl`` (the meter re-opens on it: one cap across restarts), ``steps.jsonl``, ``drops.jsonl``,
``rewards.jsonl`` (objective id, verifier version per row) and ``rollouts/`` (the Runner's run dirs). A restart creates
the training client from the saved state *with its optimizer* and samples from that step's weights; groups sampled but
not trained before a crash are sampled again (their spend stays booked).

**The cap** (``spend.SpendMeter``): every sample and train call holds its worst case first; past ``cap_usd`` the call is
refused, the episode in flight is masked (``crucible:credits_exhausted``), and the run stops with ``stop="cap"`` after
saving state. No optimizer step runs on a batch whose train call the cap refused.

**The model** is any Tinker id with a price row (``models.resolve``); the 64K or 256K variant is the id itself
(``models.with_context`` picks one for an episode cap) and the meter books that variant's train price. **The task band**:
a run is refused unless every task has a targeting record for this base model (``targeting[task_id]`` with
``model`` = the base and ``band.in_band``; ``require_band=False`` only for smoke runs), and unless every task is a
training task (held-out tasks never train: V9).

**Dashboards** (``sinks``, the ``train.sinks.Sink`` protocol through a ``Tracking``; ``observe.ObsRun`` is the W&B one,
swarm-pipeline/11 §4.2): after each optimizer step ``step_metrics`` logs reward mean/std/min/max, Φ final, gate pass
rate, pass rate, advantage stats (sequence and token), loss and every ``forward_backward`` metric, sample-vs-train
logprob gap (``kl/sample_vs_train``) and importance ratio stats, staleness, drops by reason, tokens and USD booked since
the previous step, cumulative spend vs the cap, and the sampler version; each drop is a ``record("drop", ...)``; the run
summary goes to ``summary``. The trainer never finishes the sinks it is given: the caller does, after its own logging.

Layering: ``crucible.train`` never imports ``crucible.factory``; environments are anything with ``task``,
``run_spec(run_id=, policies=, k=, **kw)`` and ``policy_kwargs()`` (``factory.envs.NativeEnv``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from crucible.core.schema.rollout import RolloutRecord
from crucible.train.bridges.tinker.datums import build_datums, scale
from crucible.train.bridges.tinker.models import TinkerModel, resolve
from crucible.train.bridges.tinker.sampler import TinkerSampler
from crucible.train.bridges.tinker.spend import SpendMeter, TinkerCapReached
from crucible.train.sinks import Sink, Tracking

LOSSES = ("importance_sampling", "ppo", "cispo")
RESUMABLE_CHANGES = ("steps", "cap_usd")


class TrainerEnv(Protocol):
    task: Any

    def run_spec(self, *, run_id: str, policies: Sequence[Any], k: int = 1, **kw: Any) -> Any: ...

    def policy_kwargs(self) -> dict[str, Any]: ...


class RunRefused(ValueError):
    """The run may not start (band, split, objective, a changed configuration on resume)."""


@dataclass
class TrainConfig:
    model: str  # any Tinker id with a price row (models.resolve): the run is model-agnostic
    lora_rank: int = 32
    group_size: int = 4
    groups_per_step: int = 1
    steps: int = 2
    loss_fn: str = "importance_sampling"
    loss_fn_config: dict[str, float] | None = None
    learning_rate: float = 4e-5  # tinker-cookbook rl_loop default for LoRA RL
    loss_scale: str = "sum"
    turn_eta: float = 0.1  # M6 turn-credit weight (credit.compose default); 0 trains on the outcome only
    honesty: bool = True
    temperature: float = 1.0
    max_tokens: int = 1024
    enable_thinking: bool = False
    max_staleness: int = 0
    async_groups: int = 0
    drop_zero_spread: bool = True
    max_groups_per_step: int | None = None  # default 3 x groups_per_step
    save_every: int = 1
    ttl_s: int | None = 14 * 86400  # save_state checkpoints and the final sampler weights
    # intermediate sampler weights: needed only while groups sample from them (a Qwen3.6-35B-A3B rank-32 adapter is 2.25 GB,
    # billed per GB-month), so they expire sooner; the last step keeps ``ttl_s``
    sampler_ttl_s: int | None = 3 * 86400
    cap_usd: float = 1.0
    seed: int = 0
    require_band: bool = True
    system: str | None = None

    def __post_init__(self) -> None:
        if self.loss_fn not in LOSSES:
            raise ValueError(f"loss_fn is one of {LOSSES}, got {self.loss_fn!r}")
        if self.loss_scale not in ("sum", "token_mean"):
            raise ValueError("loss_scale is sum or token_mean")
        if self.group_size < 2:
            raise ValueError("a GRPO group needs at least 2 attempts")
        if self.max_staleness < 0 or self.async_groups < 0 or self.groups_per_step < 1 or self.steps < 0:
            raise ValueError("max_staleness, async_groups >= 0; groups_per_step >= 1; steps >= 0")
        if self.async_groups and self.max_staleness < 1:
            raise ValueError("async_groups > 0 samples from older weights: set max_staleness >= 1 (or async_groups=0)")
        resolve(self.model)  # an unknown model has no price: refused here

    @property
    def max_launches(self) -> int:
        return self.max_groups_per_step or 3 * self.groups_per_step

    def identity(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k not in RESUMABLE_CHANGES}


@dataclass
class Group:
    gi: int
    env: Any
    task_id: str
    verifier_version: str
    run_id: str
    records: list[RolloutRecord]
    launched_version: int
    seconds: float
    error: str | None = None

    @property
    def masked(self) -> bool:
        return self.error is not None or any(r.mask_sample or r.reward is None for r in self.records)

    def min_version(self) -> int | None:
        vs = [int(t.policy_version) for r in self.records for t in r.turns
              if t.trainable and t.completion_token_ids and isinstance(t.policy_version, int)]
        return min(vs) if vs else None

    def rewards(self) -> list[float]:
        return [float(r.reward) for r in self.records if r.reward is not None and not r.mask_sample]


def check_band(task_ids: Sequence[str], model: str, targeting: Mapping[str, Mapping[str, Any]] | None) -> list[str]:
    """Why the task list may not train ``model`` (empty: it may): every task needs a targeting record for the model's base
    with its band measured and in band."""
    base = model.split(":", 1)[0]
    out = []
    for t in task_ids:
        rec = (targeting or {}).get(t)
        if rec is None:
            out.append(f"{t}: no targeting record")
            continue
        if str(rec.get("model", "")).split(":", 1)[0] != base:
            out.append(f"{t}: targeted for {rec.get('model')!r}, not {base}")
        elif not (rec.get("band") or {}).get("in_band"):
            out.append(f"{t}: out of band for {base} ({(rec.get('band') or {}).get('rewards')})")
    return out


def _atomic(path: Path, doc: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    tmp.replace(path)


def _append(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def _stats(xs: Sequence[float], prefix: str) -> dict[str, float]:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:
        return {}
    return {f"{prefix}/mean": statistics.fmean(xs), f"{prefix}/std": statistics.pstdev(xs) if len(xs) > 1 else 0.0, f"{prefix}/min": min(xs),
            f"{prefix}/max": max(xs)}


def step_metrics(batch: Sequence[Group], datums: Sequence[Any], fb: Any, kl: Sequence[float], *, version: int, sampler_version: int, drops: Mapping[str, int],
                 usd_step: float, usd_cum: float, cap_usd: float, tokens: Mapping[str, int], train_s: float, step_s: float | None = None) -> dict[str, float]:
    """The dashboard row of one optimizer step (module doc): flat ``a/b`` keys, numbers only."""
    recs = [r for g in batch for r in g.records if not r.mask_sample and r.reward is not None]
    out: dict[str, float] = {}
    out.update(_stats([float(r.reward) for r in recs], "reward"))  # type: ignore[arg-type]
    phis = [float(r.progress["phi_final"]) for r in recs if isinstance((r.progress or {}).get("phi_final"), int | float)]
    out.update({f"phi/final_{k.split('/')[1]}": v for k, v in _stats(phis, "phi").items()})
    gate_ok = []
    for r in recs:
        gs = [bool(v) for k, v in (r.gates or {}).items() if (r.gate_kinds or {}).get(k, "gate") == "gate"]
        if gs:
            gate_ok.append(all(gs))
    if gate_ok:
        out["gates/pass_rate"] = sum(gate_ok) / len(gate_ok)
    if recs:
        out["pass_rate"] = sum(float(r.reward) >= 1.0 - 1e-9 for r in recs) / len(recs)  # type: ignore[arg-type]
    out.update(_stats([d.meta.get("seq_adv") for d in datums if d.meta.get("seq_adv") is not None], "advantage/sequence"))
    tok_adv = [a for d in datums for a, m in zip(d.advantages, d.mask, strict=False) if m]
    out.update(_stats(tok_adv, "advantage/token"))
    if tok_adv:
        out["advantage/token_abs_mean"] = statistics.fmean(abs(a) for a in tok_adv)
    for k, v in (getattr(fb, "metrics", None) or {}).items():
        if isinstance(v, int | float) and math.isfinite(float(v)):
            out[f"train/{k}"] = float(v)
    if "train/loss:sum" in out:
        out["train/loss"] = out["train/loss:sum"]
    if kl:
        out["kl/sample_vs_train"] = statistics.fmean(kl)
        out["kl/sample_vs_train_abs_max"] = max(abs(x) for x in kl)
        ratios = [math.exp(max(-50.0, min(50.0, -x))) for x in kl]  # π_train / π_sample per token
        out.update(_stats(ratios, "ratio"))
    stale = [version - mv for g in batch if (mv := g.min_version()) is not None]
    if stale:
        out["staleness/max"], out["staleness/mean"] = float(max(stale)), statistics.fmean(stale)
    for k, v in drops.items():
        out[f"drops/{k}"] = float(v)
    out["drops/total"] = float(sum(drops.values()))
    for k, v in tokens.items():
        out[f"tokens/{k}"] = float(v)
    out.update({"usd/step": usd_step, "usd/cum": usd_cum, "usd/cap": cap_usd, "usd/cap_frac": usd_cum / cap_usd if cap_usd else 0.0,
                "sampler/version": float(sampler_version), "episodes": float(sum(len(g.records) for g in batch)), "groups": float(len(batch)),
                "datums": float(len(datums)), "time/train_s": train_s})
    if step_s is not None:
        out["time/step_s"] = step_s
    return out


def read_group(root: Path, run_id: str, k: int) -> list[RolloutRecord]:
    """The group's ``k`` rollouts from the Runner's run directory (a sample without a rollout is masked)."""
    out = []
    samples = sorted((root / run_id / "samples").glob("*")) if (root / run_id / "samples").exists() else []
    for sd in samples:
        atts = sorted(sd.glob("a*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0)
        ro = atts[-1] / "rollout.json" if atts else None
        if ro is not None and ro.exists():
            out.append(RolloutRecord.model_validate_json(ro.read_text()))
        else:
            out.append(RolloutRecord.model_validate({"rollout_id": f"{run_id}:{sd.name}", "mask_sample": True, "failure_kind": "crucible:no_rollout"}))
    while len(out) < k:
        out.append(RolloutRecord.model_validate({"rollout_id": f"{run_id}:missing{len(out)}", "mask_sample": True, "failure_kind": "crucible:no_rollout"}))
    return out


class TinkerTrainer:
    def __init__(self, envs: Sequence[TrainerEnv], *, config: TrainConfig, backend: Any, run_dir: Path, codec: Any, objective_id: str,
                 targeting: Mapping[str, Mapping[str, Any]] | None = None, name: str | None = None, sinks: Sequence[Sink] = ()):
        self.envs, self.cfg, self.backend, self.codec = list(envs), config, backend, codec
        self.tracking = Tracking(sinks)
        self._last_tokens: dict[str, int] = {}
        self._last_usd = 0.0
        self._last_step_t: float | None = None
        self.dir = Path(run_dir)
        self.model: TinkerModel = resolve(config.model)
        self.objective_id = objective_id
        self.objective = {str(e.task.task_id): str(getattr(e.task, "verifier_version", "") or "") for e in self.envs}
        self.name = name or self.dir.name
        if not self.envs:
            raise RunRefused("no environments")
        held = [t for t in self.objective if str(getattr(self._env(t).task, "split", "train")) != "train"]
        if held:
            raise RunRefused(f"held-out tasks never train (V9): {held}")
        if config.require_band and (bad := check_band(list(self.objective), config.model, targeting)):
            raise RunRefused("the task list has no recorded band for this model: " + "; ".join(bad[:5]))
        self.state: dict[str, Any] = {}
        self.stop: str | None = None
        self.meter: SpendMeter | None = None
        self.sampler: TinkerSampler | None = None
        self.trainer: Any = None
        self.drops: dict[str, int] = {}
        self.step_rows: list[dict[str, Any]] = []
        self.groups_seen = 0

    def _env(self, task_id: str) -> Any:
        return next(e for e in self.envs if str(e.task.task_id) == task_id)

    # ---- run directory ----
    def _open(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg_doc = {"config": self.cfg.identity(), "objective_id": self.objective_id, "objective": self.objective, "model": self.model.to_json()}
        cp, sp = self.dir / "config.json", self.dir / "state.json"
        if cp.exists():
            old = json.loads(cp.read_text())
            if old.get("objective_id") != self.objective_id:
                raise RunRefused(f"objective changed ({old.get('objective_id')} -> {self.objective_id}): a changed verifier is a new run, from a "
                                 "checkpoint taken before the change")
            if old.get("config") != cfg_doc["config"]:
                diff = sorted(k for k in set(old.get("config", {})) | set(cfg_doc["config"]) if old.get("config", {}).get(k) != cfg_doc["config"].get(k))
                raise RunRefused(f"resume with a different configuration ({diff}); only {RESUMABLE_CHANGES} may change")
        else:
            _atomic(cp, cfg_doc)
        self.state = json.loads(sp.read_text()) if sp.exists() else {"step": 0, "version": 0, "state_path": None, "sampler_path": None,
                                                                       "objective_id": self.objective_id, "next_group": 0}
        self.versions = json.loads((self.dir / "versions.json").read_text()) if (self.dir / "versions.json").exists() else {"0": None}

    def _save_state(self) -> None:
        _atomic(self.dir / "state.json", self.state)
        _atomic(self.dir / "versions.json", self.versions)

    # ---- groups ----
    async def _group(self, env: Any, gi: int) -> Group:
        from crucible.runtime.infra import RetryRule
        from crucible.runtime.run import Runner

        assert self.sampler is not None
        tid = str(env.task.task_id)
        run_id = f"g{gi:06d}-{hashlib.sha1(tid.encode()).hexdigest()[:8]}"
        v0 = self.sampler.version
        pol = self.sampler.policy_ref(pid=f"tinker-student-v{v0}", temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens, system=self.cfg.system,
                                      **env.policy_kwargs())
        t0 = time.monotonic()
        err = None
        try:
            spec = env.run_spec(run_id=run_id, policies=[pol], k=self.cfg.group_size, concurrency=self.cfg.group_size, rollout_mode="train",
                                run_seed=self.cfg.seed + gi, retry=RetryRule(max_attempts=1), breaker=None,  # a group is all or nothing: the
                                # trainer drops and refills it; the Runner's breakers would defer and re-run a masked member past max_attempts
                                sampling={"temperature": self.cfg.temperature, "max_tokens": self.cfg.max_tokens})
            await Runner(self.dir / "rollouts").run(spec)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- a group that could not run is infrastructure: dropped and refilled
            err = f"{type(e).__name__}: {str(e)[:300]}"
        recs = read_group(self.dir / "rollouts", run_id, self.cfg.group_size)
        return Group(gi, env, tid, str(getattr(env.task, "verifier_version", "") or ""), run_id, recs, v0, time.monotonic() - t0, err)

    def _verdict(self, g: Group) -> str | None:
        if g.verifier_version != self.objective.get(g.task_id):
            return "objective_changed"
        if g.masked:
            return "infra_in_group"
        if self._stale(g):
            return "stale"
        rs = g.rewards()
        if self.cfg.drop_zero_spread and (len(rs) < 2 or statistics.pstdev(rs) <= 1e-6):
            return "zero_spread"
        return None

    def _stale(self, g: Group) -> bool:
        mv = g.min_version()
        return mv is not None and self.state["version"] - mv > self.cfg.max_staleness

    def _drop(self, g: Group, reason: str) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1
        why = g.error or next((f"{r.rollout_id}: {r.failure_kind}" for r in g.records if r.mask_sample), None)
        row = {"step": self.state["step"], "version": self.state["version"], "group": g.run_id, "task": g.task_id, "reason": reason, "detail": why,
               "rewards": g.rewards(), "min_version": g.min_version()}
        _append(self.dir / "drops.jsonl", row)
        self.tracking.record("drop", row, int(self.state["step"]))

    # ---- the update ----
    async def _step(self, batch: list[Group]) -> None:
        assert self.meter is not None and self.sampler is not None
        cfg = self.cfg
        recs = [r for g in batch for r in g.records]
        keys = [g.run_id for g in batch for _ in g.records]
        datums = scale(build_datums(recs, keys, turn_eta=cfg.turn_eta, honesty=cfg.honesty), cfg.loss_scale)
        if not datums:
            raise RuntimeError("a kept batch produced no datum (every rollout token-incomplete?)")
        tokens = sum(len(d.input_ids) for d in datums)
        step = self.state["step"] + 1
        t0 = time.monotonic()
        with self.meter.hold(self.model.prices.usd(train=tokens), f"forward_backward {tokens} tokens"):
            fb = await self.trainer.forward_backward(datums, loss_fn=cfg.loss_fn, config=cfg.loss_fn_config)
            self.meter.book("train", self.model, train=tokens, step=step, datums=len(datums))
        opt = await self.trainer.optim_step(learning_rate=cfg.learning_rate)
        path = await self.trainer.save_for_sampler(f"{self.name}-s{step:06d}", ttl_s=cfg.ttl_s if step >= cfg.steps else cfg.sampler_ttl_s)
        self.meter.book("checkpoint", None, op="save_weights_for_sampler", path=path, step=step)
        train_s = time.monotonic() - t0
        await self.sampler.refresh(path, step)
        self.versions[str(step)] = path
        self.state.update(step=step, version=step, sampler_path=path)
        if step % cfg.save_every == 0 or step >= cfg.steps:
            await self._checkpoint()
        else:
            _atomic(self.dir / "versions.json", self.versions)
        n_tr = sum(d.n_train for d in datums)
        kl = [lp_s - lp_t for d, lt in zip(datums, fb.target_logprobs, strict=False) for lp_s, lp_t, m in zip(d.logprobs, lt, d.mask, strict=False) if m]
        for g in batch:
            for r in g.records:
                _append(self.dir / "rewards.jsonl", {"objective_id": self.objective_id, "step": step, "group": g.run_id, "episode": r.rollout_id,
                                                      "task": g.task_id, "verifier_version": g.verifier_version, "reward": r.reward})
        row = {"step": step, "groups": [g.run_id for g in batch], "tasks": [g.task_id for g in batch], "episodes": len(recs), "datums": len(datums),
               "train_tokens": tokens, "trained_tokens": n_tr, "reward_mean": statistics.fmean([x for g in batch for x in g.rewards()]),
               "rewards": [g.rewards() for g in batch], "loss": fb.metrics.get("loss:sum"), "fb_metrics": fb.metrics, "optim": opt,
               "kl_sample_vs_train": statistics.fmean(kl) if kl else None, "sampler_path": path, "train_s": round(train_s, 3),
               "usd_cum": round(self.meter.spent, 6), "drops": dict(self.drops), "min_versions": [g.min_version() for g in batch]}
        self.step_rows.append(row)
        _append(self.dir / "steps.jsonl", row)
        t = self.meter.totals
        tok = {"prefill_uncached": t.prefill, "prefill_cached": t.prefill_cached, "sampled": t.sample, "train": t.train}
        now = time.monotonic()
        self.tracking.log(step_metrics(batch, datums, fb, kl, version=step, sampler_version=self.sampler.version, drops=self.drops,
                                       usd_step=self.meter.spent - self._last_usd, usd_cum=self.meter.spent, cap_usd=self.cfg.cap_usd,
                                       tokens={**{k: v - self._last_tokens.get(k, 0) for k, v in tok.items()}, "trained": n_tr},
                                       train_s=train_s, step_s=(now - self._last_step_t) if self._last_step_t is not None else None), step)
        self._last_tokens, self._last_usd, self._last_step_t = tok, self.meter.spent, now

    async def _checkpoint(self) -> None:
        path = await self.trainer.save_state(f"{self.name}-state-s{self.state['step']:06d}", ttl_s=self.cfg.ttl_s)
        assert self.meter is not None
        self.meter.book("checkpoint", None, op="save_state", path=path, step=self.state["step"])
        self.state["state_path"] = path
        self.state["resume"] = {"step": self.state["step"], "state_path": path, "sampler_path": self.state["sampler_path"]}
        self._save_state()

    # ---- the loop ----
    async def run(self) -> dict[str, Any]:
        self._open()
        cfg = self.cfg
        self.meter = SpendMeter(cfg.cap_usd, path=self.dir / "spend.jsonl")
        t = self.meter.totals  # a resumed run's per-step deltas start from what earlier processes booked
        self._last_usd, self._last_tokens = self.meter.spent, {"prefill_uncached": t.prefill, "prefill_cached": t.prefill_cached, "sampled": t.sample,
                                                               "train": t.train}
        res = self.state.get("resume")
        if res:  # requeue-safe: back to the last saved state (weights + optimizer) and that step's sampler weights
            self.state.update(step=res["step"], version=res["step"], sampler_path=res["sampler_path"], state_path=res["state_path"])
        t_start = time.monotonic()
        self.trainer = await self.backend.trainer(self.model.name, rank=cfg.lora_rank, resume=self.state.get("state_path"), seed=cfg.seed)
        self.sampler = TinkerSampler(self.backend, self.model, meter=self.meter, codec=self.codec, path=self.state.get("sampler_path"),
                                     version=self.state["version"])
        await self.sampler.start()
        pending: dict[asyncio.Task, int] = {}
        batch: list[Group] = []
        launched = 0
        gi = int(self.state.get("next_group", 0))
        try:
            while self.state["step"] < cfg.steps and self.stop is None:
                need = cfg.groups_per_step - len(batch)
                extra = cfg.async_groups if self.state["step"] + 1 < cfg.steps else 0
                while len(pending) < need + extra and launched < cfg.max_launches + extra and not self.meter.exhausted:
                    env = self.envs[gi % len(self.envs)]
                    pending[asyncio.create_task(self._group(env, gi))] = gi
                    gi += 1
                    launched += 1
                    self.state["next_group"] = gi
                if not pending:
                    self.stop = "cap" if self.meter.exhausted else "too_many_drops"
                    break
                done, _ = await asyncio.wait(list(pending), return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    pending.pop(t)
                    g = t.result()
                    self.groups_seen += 1
                    why = self._verdict(g)
                    if why is None:
                        batch.append(g)
                    else:
                        self._drop(g, why)
                        if why == "objective_changed":
                            self.stop = "objective_changed"
                for g in [g for g in batch if self._stale(g)]:
                    batch.remove(g)
                    self._drop(g, "stale")
                if self.meter.exhausted:
                    self.stop = "cap"
                    break
                if len(batch) >= cfg.groups_per_step and self.stop is None:
                    try:
                        await self._step(batch[: cfg.groups_per_step])
                    except TinkerCapReached:
                        self.stop = "cap"
                        break
                    batch = batch[cfg.groups_per_step:]
                    launched = len(pending)
        finally:
            if pending:  # in flight at a stop: finish (the cap refuses their calls at once), never trained
                outs = await asyncio.gather(*pending, return_exceptions=True)
                for o in outs:
                    if isinstance(o, Group):
                        self._drop(o, f"unused_at_stop:{self.stop or 'done'}")
            if self.trainer is not None and self.state.get("resume", {}).get("step") != self.state["step"] and self.state["step"] > 0:
                try:
                    await self._checkpoint()
                except TinkerCapReached:
                    pass
            self._save_state()
        return self.summary(time.monotonic() - t_start)

    def summary(self, wall_s: float) -> dict[str, Any]:
        assert self.meter is not None and self.sampler is not None
        log = self.sampler.log
        lat = [r["latency_s"] for r in log]
        samp = sum(r["completion"] for r in log)
        eps = sum(len(r["groups"]) for r in self.step_rows) * self.cfg.group_size
        out = {"run": self.name, "model": self.model.name, "steps_done": self.state["step"], "steps": self.cfg.steps, "stop": self.stop or "done",
               "objective_id": self.objective_id, "sampler_path": self.state.get("sampler_path"), "state_path": self.state.get("state_path"),
               "versions": self.versions, "drops": dict(self.drops), "groups_seen": self.groups_seen, "spend": self.meter.summary(),
               "calls": len(log), "sampled_tokens": samp, "sample_tokens_per_s_per_call": (samp / sum(lat)) if lat and sum(lat) else None,
               "turn_latency_s": {"mean": statistics.fmean(lat), "p50": statistics.median(lat), "max": max(lat)} if lat else None,
               "trained_episodes": eps, "usd_per_episode": (self.meter.spent / max(1, self.groups_seen * self.cfg.group_size)),
               "wall_s": round(wall_s, 1), "steps_log": self.step_rows, "codec": self.codec.settings() if hasattr(self.codec, "settings") else None}
        _atomic(self.dir / "summary.json", out)
        self.tracking.summary({k: v for k, v in out.items() if k not in ("steps_log", "versions")})
        return out
