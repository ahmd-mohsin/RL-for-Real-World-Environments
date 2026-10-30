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
   - ``zero_spread`` (``drop_zero_spread``): every member's composed sequence advantage (outcome + honesty,
     ``datums.sequence_advantages``) is the same: no signal (slime's nonzero-std filter, prime-rl's ``no_signal``).
     Harness-reward audit R8 (2026-10-28): the rule read the rewards only, so a group without outcome spread whose honesty
     terms differ (an honest "not done" beside a false "done") was dropped and honesty never trained there.
   Every drop is a row in ``drops.jsonl``; a step that cannot fill its batch within ``max_groups_per_step`` launches
   stops the run (``too_many_drops``) instead of looping.
   **Masked share** (``max_masked_frac``; NeMo-Gym's ``max_mask_fraction``): over the last ``masked_window`` landed
   samples, more than ``max_masked_frac`` masked stops the run (``masked_frac``, state saved, the mask kinds in
   ``summary.json``), so an infrastructure failure cannot starve training while the run keeps filling batches from the
   survivors. ``None`` turns it off.
   **Quarantine** (``train.quarantine``, the backend-neutral scheduler rule): a task whose groups are flat (one reward
   for every member and no Φ spread) for ``quarantine_q`` consecutive steps (default 2) is not sampled for the next
   ``quarantine_r`` steps (default 3), then retried once; tasks are otherwise sampled round-robin. Every event is a row
   in ``quarantine.jsonl`` and a ``record("quarantine", ...)``; ``quarantine/active`` is on every step's dashboard row.
   ``quarantine_q=0`` turns it off.
   **Difficulty sampling** (``sampler="difficulty"``, ``train.curriculum``, swarm-pipeline/12): instead of round-robin, the
   next task is drawn by online difficulty: p̂ per task from this run's own groups (decayed per step by
   ``difficulty_decay``), tasks whose interval is outside [``band_lo``, ``band_hi``] skipped unless their Φ spreads or an
   ``explore`` pick re-measures them, the rest weighted by ``max(4p(1-p), 4 Var(Φ))``; the quarantine above still applies,
   and checkpoint-start variants (``task.curriculum = (parent, level)``) follow the reverse curriculum. The tracker,
   curriculum levels and RNG are saved in ``state.json``; events are rows in ``difficulty.jsonl``.
3. **Datums** (``datums.build_datums``): outcome advantage over each group's live rollouts, the honesty term composed in,
   M6's per-turn credit, one datum per TITO sequence; ``loss_scale``.
4. **Update.** ``forward_backward(loss_fn)`` (``importance_sampling`` | ``ppo`` | ``cispo``, ``loss_fn_config`` e.g.
   PPO's clip thresholds) -> ``optim_step(learning_rate)`` -> ``save_weights_for_sampler`` -> the sampler samples the new
   weights (version = step). ``save_state`` (weights + optimizer) every ``save_every`` steps and at a stop.

**Off-policy bound.** ``async_groups`` > 0 keeps that many extra groups sampling while a step trains (their tokens come
from the previous weights); ``max_staleness`` bounds how old a trained group may be. ``async_groups=0,
max_staleness=0`` is synchronous on-policy RL. Staleness is the version a step trains on (the sampler's version before the
step) minus the oldest version that produced a token of the group: 0 in a synchronous run (``staleness/*`` logged 1).
Async-RL audit R3 (2026-10-28): before each weight swap, a group still sampling whose oldest token
(``Runner.inflight_min_version``) is already too old to train under the new version is cancelled (``cancelled_stale``), as
prime-rl's ``on_version_pending`` does, instead of holding its Worlds for up to an episode to be dropped on landing. R4: at
any stop, an error included, the groups still in flight are cancelled (``cancelled_at_stop:<why>``), never awaited for up
to ``rollout_s``. R2: each step logs, per staleness bucket b, the share of trained tokens (``staleness/tokens_frac/s<b>``)
and, where the backend returns the trainer's logprobs (Tinker), the mismatch KL ``kl/mismatch/s<b>`` = E[r - 1 - log r]
and the share MIS would mask (``is/masked_frac/s<b>``, r outside ``MIS_BAND``).

**Resume (requeue-safe).** The run directory holds ``config.json`` (the fixed configuration: a resume with a different
one is refused, except ``steps`` and ``cap_usd``), ``state.json`` (written atomically after each ``save_state``: the step,
the Tinker state path, the sampler path and version of that step, the objective), ``versions.json`` (version -> sampler
path), ``spend.jsonl`` (the meter re-opens on it: one cap across restarts), ``steps.jsonl``, ``drops.jsonl``,
``rewards.jsonl`` (objective id, verifier version per row) and ``rollouts/`` (the Runner's run dirs). A restart creates
the training client from the saved state *with its optimizer* and samples from that step's weights; groups sampled but
not trained before a crash are sampled again (their spend stays booked). Async-RL audit R7 (2026-10-28): the rows of
``steps.jsonl`` and ``rewards.jsonl`` past the resumed step (steps trained after the last ``save_state``, trained again
now) move to ``steps.rolled_back.jsonl`` / ``rewards.rolled_back.jsonl`` at the restart, so each step keeps one row.

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
the previous step, cumulative spend vs the cap, and the sampler version, then the leading indicators
(``behaviour_metrics``: entropy estimate, lengths and turns, stop-reason shares and truncation, format errors, flagged
and give-up shares, within-group reward std), the flat-group share and the ``monitors.EarlyWarning`` alarms
(``alarm/<kind>``; a fired one is also a ``record("alarm", ...)`` and a log warning, never a stop) and the shaping
shares (``shaping_metrics``); each drop is a ``record("drop", ...)``; the run summary goes to ``summary``. The trainer
never finishes the sinks it is given: the caller does, after its own logging.

**Framework health** (Part 1 A7, 2026-10-30; ``health``, the factory's ``framework_pause``): after each step the check
runs over the run directory in a thread while the next groups sample; before the next step trains, an open CRITICAL
framework defect (one that corrupts reward or advantage: ``factory.framework_health``) stops the run with
``stop="framework_health"`` and a ``record("stop", ...)``. The stop saves state as every stop does; a resumed run checks
before its first step, so it trains on only once the defect is fixed and recorded (``scripts/train_monitor.py fix``). A
check that fails is a log warning, never a stop.

Layering: ``crucible.train`` never imports ``crucible.factory``; environments are anything with ``task``,
``run_spec(run_id=, policies=, k=, **kw)`` and ``policy_kwargs()`` (``factory.envs.NativeEnv``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import statistics
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from crucible.core.schema.rollout import RolloutRecord
from crucible.train.bridges.tinker.datums import FORMAT_MODES, build_datums, scale, sequence_advantages
from crucible.train.bridges.tinker.models import TinkerModel, resolve
from crucible.train.bridges.tinker.sampler import TinkerSampler
from crucible.train.bridges.tinker.spend import SpendMeter, TinkerCapReached
from crucible.train.credit import outcome_advantages
from crucible.train.curriculum import DifficultySampler, SamplerConfig, draws_of, requires_of
from crucible.train.monitors import EarlyWarning
from crucible.train.quarantine import QuarantineConfig, TaskQuarantine
from crucible.train.sinks import Sink, Tracking

log = logging.getLogger(__name__)

LOSSES = ("importance_sampling", "ppo", "cispo")
# the quarantine only changes which task is sampled when, never the objective: a resume may change it (and a run made
# before it existed resumes with it on)
RESUMABLE_CHANGES = ("steps", "cap_usd", "quarantine_q", "quarantine_r", "sampler", "band_lo", "band_hi", "explore", "difficulty_decay",
                     "max_masked_frac", "masked_window")
# fields added after runs existed, left out of the identity at their default so those runs still resume (a run that sets one
# trains another objective: it is in the identity then)
IDENTITY_DEFAULTS = {"tool_format_penalty": 0.0, "format_penalty_mode": "reduction", "format_kappa": 2.0}
# the band of the prime-rl loss's MIS (``primerl.loss.MIS_LOW`` / ``MIS_HIGH``, the paper's 0.5 and 5): ``is/masked_frac/s<b>``
MIS_BAND = (0.5, 5.0)


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
    # harness-reward audit R9 (2026-10-28): "token_mean" (a global token mean: every trained token weighs the same, divided by
    # the batch's trained-token count, as prime-rl, AReaL, SkyRL, verl, NeMo-RL and TRL normalise; async-rl-audit §3.3), so
    # the step's scale does not swing with how long the batch's 20-300-call episodes were. "sum" (the default before)
    # scaled the gradient by the token count and left only the clip (max_norm 1) to bound it, which then gave every step
    # the same length whatever its signal. Not per episode: a sequence mean weighs a long failing episode's tokens less
    loss_scale: str = "token_mean"
    turn_eta: float = 0.1  # M6 turn-credit weight (credit.compose default); 0 trains on the outcome only
    honesty: bool = True
    temperature: float = 1.0
    max_tokens: int = 1024
    enable_thinking: bool = False
    max_staleness: int = 0
    async_groups: int = 0
    drop_zero_spread: bool = True  # drop a group whose composed advantages (outcome + honesty) are all equal (R8)
    quarantine_q: int = 2  # flat groups on this many consecutive steps quarantine a task (train.quarantine); 0: off
    quarantine_r: int = 3  # steps a quarantined task is not sampled, then retried once
    sampler: str = "round_robin"  # "round_robin" (the quarantine's order) | "difficulty" (train.curriculum, swarm-pipeline/12)
    band_lo: float | None = 0.1  # the difficulty sampler's band on p̂ (forge.learnability's train band)
    band_hi: float | None = 0.7
    explore: float = 0.1  # share of difficulty picks that ignore the band (re-measure out-of-band tasks)
    difficulty_decay: float = 0.9  # per-step decay of old p̂ evidence
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
    # harness-reward audit R5 (2026-10-28): the torch trainer's format penalty, on every backend (``datums.build_datums``);
    # 0 (the default there too) is off: a malformed call already costs its tool call and is answered with the error
    tool_format_penalty: float = 0.0
    format_penalty_mode: str = "reduction"  # "reduction" | "signed" (``credit.signed``: each sign's batch mass kept)
    format_kappa: float = 2.0
    # NeMo-Gym's abort (rollout_collection.py, max_mask_fraction): over the last ``masked_window`` landed samples, more than
    # this share masked stops the run (``masked_frac``); None: off
    max_masked_frac: float | None = 0.5
    masked_window: int = 64

    def __post_init__(self) -> None:
        if self.loss_fn not in LOSSES:
            raise ValueError(f"loss_fn is one of {LOSSES}, got {self.loss_fn!r}")
        if self.sampler not in ("round_robin", "difficulty"):
            raise ValueError("sampler is round_robin or difficulty")
        if self.loss_scale not in ("sum", "token_mean"):
            raise ValueError("loss_scale is sum or token_mean")
        if self.group_size < 2:
            raise ValueError("a GRPO group needs at least 2 attempts")
        if self.max_staleness < 0 or self.async_groups < 0 or self.groups_per_step < 1 or self.steps < 0:
            raise ValueError("max_staleness, async_groups >= 0; groups_per_step >= 1; steps >= 0")
        if self.async_groups and self.max_staleness < 1:
            raise ValueError("async_groups > 0 samples from older weights: set max_staleness >= 1 (or async_groups=0)")
        if self.format_penalty_mode not in FORMAT_MODES or self.tool_format_penalty < 0 or self.format_kappa < 1.0:
            raise ValueError(f"format_penalty_mode is one of {FORMAT_MODES}; tool_format_penalty >= 0; format_kappa >= 1")
        if (self.max_masked_frac is not None and not 0.0 < self.max_masked_frac <= 1.0) or self.masked_window < 1:
            raise ValueError("max_masked_frac is in (0, 1] or None; masked_window >= 1")
        resolve(self.model)  # an unknown model has no price: refused here

    @property
    def max_launches(self) -> int:
        return self.max_groups_per_step or 3 * self.groups_per_step

    def identity(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()
                if k not in RESUMABLE_CHANGES and not (k in IDENTITY_DEFAULTS and v == IDENTITY_DEFAULTS[k])}


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
    picked_step: int | None = None  # the optimizer step its task was picked at (the quarantine dates observations by it: R6)
    probe: bool | None = None  # it is its task's probation retry (``TaskQuarantine.last_probe`` after its pick)

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
    out.update(behaviour_metrics(batch, datums))
    out.update(shaping_metrics(batch, datums))
    return out


def behaviour_metrics(batch: Sequence[Group], datums: Sequence[Any]) -> dict[str, float]:
    """Part 1 §3.2 #2 (2026-10-30; env-synthesis/notes/05-longhorizon-rl.md §6): the leading indicators of a step's batch,
    read from what its episodes already record. Every episode of the kept groups counts, masked ones too (how an episode
    ended is the policy's behaviour, a timeout included); the entropy estimate reads the trained tokens.

    - ``entropy/sampled``: -mean log p of the sampled tokens (the behaviour logprobs). Its expectation is the policy's
      entropy at the sampling temperature: a fast fall predicts the ceiling, a rebound came before collapse (2505.22617,
      INTELLECT-2).
    - ``length/call_tokens/*`` (sampled tokens per call), ``length/turns/*`` (the policy's calls per episode),
      ``length/context_tokens/*`` (``trajectory_tokens``): a length surge led collapse by about 190 steps (2602.01826).
    - ``stop/<reason>``: the share of episodes per stop reason (``agent_done`` and the limit stops always, 0 when absent,
      so a chart has no gaps); ``stop/truncated``: the share cut by a limit (``RolloutRecord.truncated``; ScaleRL).
    - ``format_errors/per_episode``, ``format_errors/episode_share``: malformed tool calls (``Turn.format_errors``).
    - ``hack/flagged_share``: episodes with a level-1 rule flag or a grade flag (``RolloutRecord.flags``).
    - ``give_up/share``: episodes ended by ``GIVE_UP`` (the checkpoint sidecar's ``progress["give_up"]``).
    - ``reward/group_std_mean``: the mean within-group reward std; its drop led collapse (RAGEN 2504.20073). Groups with no
      spread at all never reach a step: ``drops/zero_spread`` counts them."""
    from crucible.runtime.harness.native import LIMIT_STOPS

    out: dict[str, float] = {}
    sampled = [-float(lp) for d in datums for lp, m in zip(d.logprobs, d.mask, strict=False) if m]
    if sampled:
        out["entropy/sampled"] = statistics.fmean(sampled)
    eps = [r for g in batch for r in g.records]
    if not eps:
        return out
    out.update(_stats(call_lengths(batch), "length/call_tokens"))
    out.update(_stats([sum(1 for t in r.turns if t.trainable and t.completion_token_ids) for r in eps], "length/turns"))
    out.update(_stats([r.trajectory_tokens for r in eps if r.trajectory_tokens is not None], "length/context_tokens"))
    stops = Counter(r.stop or "none" for r in eps)
    for s in sorted({"agent_done", *LIMIT_STOPS, *stops}):
        out[f"stop/{s}"] = stops[s] / len(eps)
    out["stop/truncated"] = sum(r.truncated for r in eps) / len(eps)
    errs = [sum(t.format_errors for t in r.turns) for r in eps]
    out["format_errors/per_episode"] = statistics.fmean(errs)
    out["format_errors/episode_share"] = sum(e > 0 for e in errs) / len(eps)
    out["hack/flagged_share"] = sum(bool(r.flags) for r in eps) / len(eps)
    out["give_up/share"] = sum(bool((r.progress or {}).get("give_up")) for r in eps) / len(eps)
    stds = [statistics.pstdev(rs) for g in batch if len(rs := g.rewards()) > 1]
    if stds:
        out["reward/group_std_mean"] = statistics.fmean(stds)
    return out


def shaping_metrics(batch: Sequence[Group], datums: Sequence[Any]) -> dict[str, float]:
    """Part 1 A7's monitor (2026-10-30; §2(a): "watch the share of steps whose signal comes only from shaping"): how much
    of the step's advantage the shaping terms (honesty, Φ turn credit, the format penalty) put there rather than the
    outcome. Per trained token, its advantage A (before ``scale``) against its rollout's outcome advantage o (R − the
    group's mean R, ``credit.outcome_advantages``):

    - ``credit/shaping_share``: Σ|A − o| / (Σ|o| + Σ|A − o|), the shaping part of the advantage mass;
    - ``credit/shaping_only_share``: Σ|A| on tokens whose rollout has o = 0 (a flat group that honesty keeps, an episode
      at its group's mean) over Σ|A|: gradient the outcome never asked for.

    When most of the gradient is shaping, a dense-vs-sparse comparison measures the shaping, not the outcome."""
    recs = [r for g in batch for r in g.records]
    o_of = outcome_advantages(recs, [g.run_id for g in batch for _ in g.records])
    a_mass = o_mass = s_mass = only = 0.0
    for d in datums:
        o, k = o_of.get(d.meta.get("rollout"), 0.0), d.meta.get("scale") or 1.0
        for a, m in zip(d.advantages, d.mask, strict=False):
            if m:
                a /= k
                a_mass, o_mass, s_mass = a_mass + abs(a), o_mass + abs(o), s_mass + abs(a - o)
                only += abs(a) if abs(o) <= 1e-9 else 0.0
    out: dict[str, float] = {}
    if o_mass + s_mass > 0:
        out["credit/shaping_share"] = s_mass / (o_mass + s_mass)
    if a_mass > 0:
        out["credit/shaping_only_share"] = only / a_mass
    return out


def call_lengths(batch: Sequence[Group]) -> list[int]:
    """Sampled tokens per call of the policy's own calls, over every episode of the batch."""
    return [len(t.completion_token_ids) for g in batch for r in g.records for t in r.turns if t.trainable and t.completion_token_ids]


def staleness_metrics(datums: Sequence[Any], target_logprobs: Sequence[Sequence[float]] | None, trained_version: int) -> dict[str, float]:
    """Async-RL audit R2 (2026-10-28): the step's trust region per staleness bucket. A datum's bucket b is the version the
    step trains on minus the oldest version that sampled a token of its sequence (``meta["min_version"]``). Always: the
    share of trained tokens per bucket (``staleness/tokens_frac/s<b>``). With the trainer's logprobs of every datum
    (Tinker's ``target_logprobs``): per bucket, the mismatch KL ``kl/mismatch/s<b>`` = E[r - 1 - log r] over trained tokens
    (``primerl.loss``'s ``kl/mismatch``) and the share of trained tokens whose ratio r = pi_train / pi_sample is outside the
    MIS band (``is/masked_frac/s<b>``: what the prime-rl loss's MIS drops). prime-rl's trainer returns no per-token logprobs,
    so there only the token shares are logged; in a step with one bucket (every synchronous step) the trainer's own
    ``is/masked_frac`` and ``kl/mismatch`` are that bucket's."""
    tokens: Counter[int] = Counter()
    kl: dict[int, list[float]] = {}
    out_band: dict[int, list[float]] = {}
    have = bool(target_logprobs) and len(target_logprobs or ()) == len(datums)
    lo, hi = MIS_BAND
    for i, d in enumerate(datums):
        mv = (getattr(d, "meta", None) or {}).get("min_version")
        if not isinstance(mv, int) or isinstance(mv, bool):
            continue
        b = int(trained_version) - mv
        tokens[b] += int(sum(1 for m in d.mask if m))
        if have:
            for lp_s, lp_t, m in zip(d.logprobs, target_logprobs[i], d.mask, strict=False):  # type: ignore[index]
                if m:
                    lr = max(-20.0, min(20.0, float(lp_t) - float(lp_s)))
                    r = math.exp(lr)
                    kl.setdefault(b, []).append(r - 1.0 - lr)
                    out_band.setdefault(b, []).append(0.0 if lo <= r <= hi else 1.0)
    total = sum(tokens.values())
    out: dict[str, float] = {f"staleness/tokens_frac/s{b}": n / total for b, n in sorted(tokens.items())} if total else {}
    for b in sorted(kl):
        out[f"kl/mismatch/s{b}"] = statistics.fmean(kl[b])
        out[f"is/masked_frac/s{b}"] = statistics.fmean(out_band[b])
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
                 targeting: Mapping[str, Mapping[str, Any]] | None = None, name: str | None = None, sinks: Sequence[Sink] = (),
                 model: TinkerModel | None = None, health: Callable[[Path], Sequence[str]] | None = None):
        self.envs, self.cfg, self.backend, self.codec = list(envs), config, backend, codec
        self.tracking = Tracking(sinks)
        # Part 1 A7: run directory -> the open CRITICAL framework defects (empty: train on); None: no check (module doc)
        self.health = health
        self._health_task: asyncio.Task | None = None
        self.health_stop: dict[str, Any] | None = None
        self._last_tokens: dict[str, int] = {}
        self._last_usd = 0.0
        self._last_step_t: float | None = None
        self._last_drops: dict[str, int] = {}
        self.warnings = EarlyWarning()  # Part 1 §3.2 #2: alarms on the leading indicators (warnings only)
        self.dir = Path(run_dir)
        # ``model``: a row other than the Tinker price table's, e.g. the DeltaAI bridge's zero-priced row with its own
        # context (``bridges.primerl.backend.local_model``): the meter then books tokens at 0 USD (GPU hours are the cost)
        self.model: TinkerModel = model if model is not None else resolve(config.model)
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
        self.quarantine = TaskQuarantine([str(e.task.task_id) for e in self.envs], QuarantineConfig(config.quarantine_q, config.quarantine_r),
                                         splits={str(e.task.task_id): str(getattr(e.task, "split", "train") or "train") for e in self.envs})
        self._q_logged = 0
        self.difficulty: DifficultySampler | None = None
        if config.sampler == "difficulty":
            self.difficulty = DifficultySampler(
                [str(e.task.task_id) for e in self.envs],
                SamplerConfig(band_lo=config.band_lo, band_hi=config.band_hi, decay=config.difficulty_decay, explore=config.explore),
                quarantine=self.quarantine, splits={str(e.task.task_id): str(getattr(e.task, "split", "train") or "train") for e in self.envs},
                levels={str(e.task.task_id): tuple(c) for e in self.envs if (c := getattr(e.task, "curriculum", None))},
                requires={str(e.task.task_id): requires_of(getattr(e.task, "milestones", None) or ()) for e in self.envs}, seed=config.seed)
        self._d_logged = 0
        self._pending: dict[asyncio.Task, dict[str, Any]] = {}  # groups in flight -> {gi, run_id, task, picked_step, probe}
        self._runners: dict[int, Any] = {}  # gi -> the Runner of its group while it runs (``inflight_min_version``: R3)
        self._cancelled: set[asyncio.Task] = set()  # cancelled groups still winding down (their Worlds closing)
        self._recent: deque[str | None] = deque(maxlen=config.masked_window)  # per landed sample: its mask kind, None when live
        self.masked_stop: dict[str, Any] | None = None

    cancel_grace_s = 300.0  # how long a cancelled group may take to close its Worlds before the loop goes on without it

    def _q_flush(self) -> None:
        """Quarantine events not logged yet: ``quarantine.jsonl`` and the sinks' ``quarantine`` records."""
        for e in self.quarantine.events[self._q_logged:]:
            _append(self.dir / "quarantine.jsonl", e)
            self.tracking.record("quarantine", e, int(e.get("step") or 0))
        self._q_logged = len(self.quarantine.events)
        if self.difficulty is not None:
            for e in self.difficulty.events[self._d_logged:]:
                _append(self.dir / "difficulty.jsonl", e)
                self.tracking.record("difficulty", e, int(e.get("step") or 0))
            self._d_logged = len(self.difficulty.events)

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
        self.quarantine.load(self.state.get("quarantine"))
        if self.difficulty is not None:
            self.difficulty.load(self.state.get("difficulty"))
        self.versions =json.loads((self.dir / "versions.json").read_text()) if (self.dir / "versions.json").exists() else {"0": None}

    def _save_state(self) -> None:
        self.state["quarantine"] = self.quarantine.to_json()
        if self.difficulty is not None:
            self.state["difficulty"] = self.difficulty.to_json()
        _atomic(self.dir / "state.json", self.state)
        _atomic(self.dir / "versions.json", self.versions)

    # ---- groups ----
    @staticmethod
    def _run_id(env: Any, gi: int) -> str:
        return f"g{gi:06d}-{hashlib.sha1(str(env.task.task_id).encode()).hexdigest()[:8]}"

    async def _group(self, env: Any, gi: int, *, picked_step: int | None = None, probe: bool | None = None) -> Group:
        from crucible.runtime.infra import RetryRule
        from crucible.runtime.run import Runner

        assert self.sampler is not None
        tid = str(env.task.task_id)
        run_id = self._run_id(env, gi)
        v0 = self.sampler.version
        pol = self.sampler.policy_ref(pid=f"tinker-student-v{v0}", temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens, system=self.cfg.system,
                                      **env.policy_kwargs())
        t0 = time.monotonic()
        err = None
        runner = Runner(self.dir / "rollouts")
        self._runners[gi] = runner  # async-RL audit R3: its in-flight minimum version is read before each weight swap
        try:
            spec = env.run_spec(run_id=run_id, policies=[pol], k=self.cfg.group_size, concurrency=self.cfg.group_size, rollout_mode="train",
                                run_seed=self.cfg.seed + gi, retry=RetryRule(max_attempts=1), breaker=None,  # a group is all or nothing: the
                                # trainer drops and refills it; the Runner's breakers would defer and re-run a masked member past max_attempts
                                sampling={"temperature": self.cfg.temperature, "max_tokens": self.cfg.max_tokens})
            await runner.run(spec)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- a group that could not run is infrastructure: dropped and refilled
            err = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            self._runners.pop(gi, None)
        recs = read_group(self.dir / "rollouts", run_id, self.cfg.group_size)
        return Group(gi, env, tid, str(getattr(env.task, "verifier_version", "") or ""), run_id, recs, v0, time.monotonic() - t0, err,
                     picked_step=picked_step, probe=probe)

    def _verdict(self, g: Group) -> str | None:
        if g.verifier_version != self.objective.get(g.task_id):
            return "objective_changed"
        if g.masked:
            return "infra_in_group"
        if self._stale(g):
            return "stale"
        rs = g.rewards()
        if self.cfg.drop_zero_spread and (len(rs) < 2 or not self._has_signal(g)):
            return "zero_spread"
        return None

    def _has_signal(self, g: Group) -> bool:
        """Harness-reward audit R8 (2026-10-28): the group trains something when its members' composed sequence advantages
        (outcome + honesty, exactly what ``build_datums`` puts on their tokens) differ. Outcome spread always gives that; a
        group without it still does when its honesty terms differ (an honest "not done" beside a false "done": the term
        alone separates them, ``factory.honesty.compose_advantage``). Before, the rule read the rewards only, so honesty
        never trained in such a group."""
        live = [r for r in g.records if not r.mask_sample and r.reward is not None]
        adv = sequence_advantages(live, [g.run_id] * len(live), honesty=self.cfg.honesty)
        return len(adv) >= 2 and statistics.pstdev(list(adv.values())) > 1e-6

    def _stale(self, g: Group) -> bool:
        mv = g.min_version()
        return mv is not None and self.state["version"] - mv > self.cfg.max_staleness

    def _drop(self, g: Group, reason: str) -> None:
        why = g.error or next((f"{r.rollout_id}: {r.failure_kind}" for r in g.records if r.mask_sample), None)
        self._drop_row({"group": g.run_id, "task": g.task_id, "rewards": g.rewards(), "min_version": g.min_version()}, reason, why)

    def _drop_row(self, info: Mapping[str, Any], reason: str, detail: str | None) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1
        row = {"step": self.state["step"], "version": self.state["version"], "group": info.get("group") or info.get("run_id"), "task": info.get("task"),
               "reason": reason, "detail": detail, "rewards": list(info.get("rewards") or []), "min_version": info.get("min_version")}
        _append(self.dir / "drops.jsonl", row)
        self.tracking.record("drop", row, int(self.state["step"]))

    # ---- groups in flight: cancel what can never train (async-RL audit R3, R4) ----
    async def _cancel_doomed(self, new_version: int) -> None:
        """Before the weights move to ``new_version``: a group still sampling whose oldest sampled token is already more than
        ``max_staleness`` versions behind ``new_version`` can never train (every later call samples a newer version, so its
        minimum never rises), so it is cancelled now (``cancelled_stale``), as prime-rl's ``on_version_pending`` and the torch
        loop's ``Scheduler.evict_doomed`` do. A 300-call episode holds its Worlds for about 28 minutes; it used to run to the
        end and be dropped as ``stale`` on landing. A group with no versioned call yet is left to run."""
        doomed = []
        for t, info in list(self._pending.items()):
            r = self._runners.get(int(info["gi"]))
            mv = r.inflight_min_version() if r is not None and not t.done() else None
            if mv is not None and new_version - mv > self.cfg.max_staleness:
                doomed.append((t, info, mv))
        for t, info, mv in doomed:
            self._pending.pop(t, None)
            t.cancel()
            self._cancelled.add(t)
            self.quarantine.lost(str(info["task"]), probe=info.get("probe"))  # never an observation; a cancelled retry re-opens
            self._drop_row({**info, "min_version": mv}, "cancelled_stale",
                           f"in flight at the swap to v{new_version}: its oldest token is v{mv} (max_staleness {self.cfg.max_staleness})")
        if doomed:
            await self._settle_cancelled(self.cancel_grace_s)

    async def _settle_cancelled(self, timeout_s: float | None) -> None:
        """Wait (at most ``timeout_s``) for cancelled groups to close their Worlds (the Runner's teardown runs on cancel)."""
        if not self._cancelled:
            return
        done, _ = await asyncio.wait(list(self._cancelled), timeout=timeout_s)
        for t in done:
            self._cancelled.discard(t)
            if not t.cancelled():
                t.exception()  # retrieved: a group that raised while cancelling is not an unhandled-task warning

    async def _stop_pending(self, why: str) -> None:
        """Async-RL audit R4: at a stop (an error included) every group still in flight is cancelled, never awaited: one could
        hold its Worlds and the GPUs for up to ``rollout_s`` (30 minutes) after the loop had stopped. A group that had
        already finished is dropped as ``unused_at_stop:<why>``; the others as ``cancelled_at_stop:<why>``."""
        items = list(self._pending.items())
        self._pending.clear()
        if not items:
            return
        for t, _ in items:
            if not t.done():
                t.cancel()
        done, _ = await asyncio.wait([t for t, _ in items], timeout=self.cancel_grace_s)
        for t, info in items:
            g = t.result() if t in done and not t.cancelled() and t.exception() is None else None
            if isinstance(g, Group):
                self._drop(g, f"unused_at_stop:{why}")
            else:
                self.quarantine.lost(str(info["task"]), probe=info.get("probe"))
                self._drop_row(info, f"cancelled_at_stop:{why}", None if t in done else f"still closing after {self.cancel_grace_s:g}s")
                if t not in done:
                    self._cancelled.add(t)

    # ---- the masked share (NeMo-Gym's abort) ----
    def _count_masked(self, g: Group) -> None:
        if g.error is not None:
            self._recent.extend(["group_error"] * self.cfg.group_size)
        else:
            self._recent.extend([str(r.failure_kind or "masked") if r.mask_sample or r.reward is None else None for r in g.records])

    def _masked_share_exceeded(self) -> bool:
        """Over the last ``masked_window`` landed samples (once that many have landed), more than ``max_masked_frac`` masked:
        the run stops (``masked_frac``) with the mask kinds, instead of training on the survivors of a broken pipeline."""
        c = self.cfg
        if c.max_masked_frac is None or len(self._recent) < c.masked_window:
            return False
        kinds = [k for k in self._recent if k is not None]
        if len(kinds) / len(self._recent) <= c.max_masked_frac:
            return False
        self.masked_stop = {"masked": len(kinds), "window": len(self._recent), "max_masked_frac": c.max_masked_frac,
                            "kinds": dict(Counter(kinds).most_common()), "step": int(self.state["step"])}
        self.tracking.record("stop", {"stop": "masked_frac", **self.masked_stop}, int(self.state["step"]))
        return True

    # ---- framework health (Part 1 A7) ----
    def _start_health(self) -> None:
        if self.health is not None:
            self._health_task = asyncio.create_task(self._health_check(int(self.state["step"])))

    async def _health_check(self, step: int) -> list[str]:
        """``health`` in a thread: it reads every stored episode of the run (about 50 s for 358 long ones), so the next
        groups sample meanwhile. Its own failure is a warning: a false stop wastes a queue slot."""
        assert self.health is not None
        try:
            return list(await asyncio.to_thread(self.health, self.dir))
        except Exception as e:  # noqa: BLE001 -- the check failing is no defect of the run
            log.warning("after step %d: framework_health not checked (%s: %s)", step, type(e).__name__, e)
            return []

    async def _health_paused(self) -> bool:
        """Before a step trains: the check started after the previous step (or at a resume). An open CRITICAL defect stops
        the run here, before its reward or advantage reaches another step."""
        if self._health_task is None:
            return False
        t, self._health_task = self._health_task, None
        defects = await t
        if not defects:
            return False
        step = int(self.state["step"])
        self.health_stop = {"step": step, "defects": defects[:10]}
        self.tracking.record("stop", {"stop": "framework_health", **self.health_stop}, step)
        log.warning("after step %d: open critical framework defects, the run stops: %s", step, "; ".join(defects[:3]))
        return True

    # ---- resume: one row per step (async-RL audit R7) ----
    def _rollback_logs(self, keep_step: int) -> None:
        """The rows of ``steps.jsonl`` and ``rewards.jsonl`` past ``keep_step`` (steps trained after the last ``save_state``,
        which a resume trains again) move to ``<name>.rolled_back.jsonl``; nothing is deleted. A torn last line (the crash)
        moves too."""
        for name in ("steps", "rewards"):
            f = self.dir / f"{name}.jsonl"
            if not f.exists():
                continue
            keep, gone = [], []
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    step = int(json.loads(line).get("step") or 0)
                except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                    gone.append(line)
                    continue
                (keep if step <= keep_step else gone).append(line)
            if gone:
                with (self.dir / f"{name}.rolled_back.jsonl").open("a") as fh:
                    fh.write("".join(x + "\n" for x in gone))
                tmp = f.with_suffix(".jsonl.tmp")
                tmp.write_text("".join(x + "\n" for x in keep))
                tmp.replace(f)

    # ---- the update ----
    async def _step(self, batch: list[Group]) -> None:
        assert self.meter is not None and self.sampler is not None
        cfg = self.cfg
        recs = [r for g in batch for r in g.records]
        keys = [g.run_id for g in batch for _ in g.records]
        datums = scale(build_datums(recs, keys, turn_eta=cfg.turn_eta, honesty=cfg.honesty, format_penalty=cfg.tool_format_penalty,
                                    format_mode=cfg.format_penalty_mode, format_kappa=cfg.format_kappa), cfg.loss_scale)
        if not datums:
            raise RuntimeError("a kept batch produced no datum (every rollout token-incomplete?)")
        tokens = sum(len(d.input_ids) for d in datums)
        step = self.state["step"] + 1
        v_train = int(self.state["version"])  # the version this step trains on: staleness is measured against it
        await self._cancel_doomed(step)  # async-RL audit R3: the weights move to v{step}; what can no longer train stops now
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
        t = self.meter.totals
        tok = {"prefill_uncached": t.prefill, "prefill_cached": t.prefill_cached, "sampled": t.sample, "train": t.train}
        now = time.monotonic()
        m = step_metrics(batch, datums, fb, kl, version=v_train, sampler_version=self.sampler.version, drops=self.drops,
                         usd_step=self.meter.spent - self._last_usd, usd_cum=self.meter.spent, cap_usd=self.cfg.cap_usd,
                         tokens={**{k: v - self._last_tokens.get(k, 0) for k, v in tok.items()}, "trained": n_tr},
                         train_s=train_s, step_s=(now - self._last_step_t) if self._last_step_t is not None else None)
        m.update(staleness_metrics(datums, getattr(fb, "target_logprobs", None), v_train))
        m["quarantine/active"] = float(len(self.quarantine.active(step)))
        m["quarantine/events"] = float(len(self.quarantine.events))
        if self.difficulty is not None:
            m.update(self.difficulty.metrics())
        # every group finished since the last step whose rewards are all equal: the ones dropped as zero_spread never reach
        # a batch, so they are counted from the drops
        zs = self.drops.get("zero_spread", 0) - self._last_drops.get("zero_spread", 0)
        m["reward/flat_group_share"] = (zs + sum(len(set(g.rewards())) <= 1 for g in batch)) / (zs + len(batch))
        warn, fired = self.warnings.update(step, m, call_lengths(batch))
        m.update(warn)
        for a in fired:  # warnings only: nothing here stops the run
            log.warning("step %d: early-warning alarm %s: %s", step, a["kind"], a)
            self.tracking.record("alarm", a, step)
        # the dashboard row in the run's own record too (staleness buckets, drops, SC / routing / MIS numbers): a run read
        # back from its directory (an offline W&B run cannot be) shows what the dashboard showed
        row["metrics"] = m
        self.step_rows.append(row)
        _append(self.dir / "steps.jsonl", row)
        self.tracking.log(m, step)
        self._last_tokens, self._last_usd, self._last_step_t, self._last_drops = tok, self.meter.spent, now, dict(self.drops)

    async def _checkpoint(self) -> None:
        path = await self.trainer.save_state(f"{self.name}-state-s{self.state['step']:06d}", ttl_s=self.cfg.ttl_s)
        assert self.meter is not None
        self.meter.book("checkpoint", None, op="save_state", path=path, step=self.state["step"])
        self.state["state_path"] = path
        self.state["resume"] = {"step": self.state["step"], "state_path": path, "sampler_path": self.state["sampler_path"]}
        self._save_state()

    # ---- the loop ----
    async def run(self) -> dict[str, Any]:
        cfg = self.cfg
        # a backend that scores the batch at its own temperature (``primerl.PrimeRLBackend.temperature``: every trained
        # token's ``temperatures`` entry) must score at the one the groups sample with, or every ratio is off
        bt = getattr(self.backend, "temperature", None)
        if isinstance(bt, int | float) and abs(float(bt) - float(cfg.temperature)) > 1e-9:
            raise RunRefused(f"the backend trains at temperature {bt}, the groups sample at {cfg.temperature}")
        self._open()
        self.meter = SpendMeter(cfg.cap_usd, path=self.dir / "spend.jsonl")
        t = self.meter.totals  # a resumed run's per-step deltas start from what earlier processes booked
        self._last_usd, self._last_tokens = self.meter.spent, {"prefill_uncached": t.prefill, "prefill_cached": t.prefill_cached, "sampled": t.sample,
                                                               "train": t.train}
        res = self.state.get("resume")
        self._rollback_logs(int(res["step"]) if res else 0)  # async-RL audit R7: the steps a resume trains again keep one row
        if res:  # requeue-safe: back to the last saved state (weights + optimizer) and that step's sampler weights
            self.state.update(step=res["step"], version=res["step"], sampler_path=res["sampler_path"], state_path=res["state_path"])
        t_start = time.monotonic()
        self.trainer = await self.backend.trainer(self.model.name, rank=cfg.lora_rank, resume=self.state.get("state_path"), seed=cfg.seed)
        self.sampler = TinkerSampler(self.backend, self.model, meter=self.meter, codec=self.codec, path=self.state.get("sampler_path"),
                                     version=self.state["version"])
        await self.sampler.start()
        if self.state["step"] > 0:
            self._start_health()  # a resume trains nothing more while a defect that stopped it is still open
        pending = self._pending
        batch: list[Group] = []
        launched = 0
        gi = int(self.state.get("next_group", 0))
        cursor = int(self.state.get("cursor", gi))  # round-robin position over the tasks (= the group index before the quarantine)
        failed: BaseException | None = None
        try:
            while self.state["step"] < cfg.steps and self.stop is None:
                need = cfg.groups_per_step - len(batch)
                extra = cfg.async_groups if self.state["step"] + 1 < cfg.steps else 0
                while len(pending) < need + extra and launched < cfg.max_launches + extra and not self.meter.exhausted:
                    picked = int(self.state["step"])
                    i = (self.difficulty or self.quarantine).pick(cursor, picked)
                    probe = self.quarantine.last_probe  # async-RL audit R6: this group is its task's probation retry
                    self._q_flush()
                    env = self.envs[i]
                    cursor = i + 1
                    t = asyncio.create_task(self._group(env, gi))
                    pending[t] = {"gi": gi, "run_id": self._run_id(env, gi), "task": str(env.task.task_id), "picked_step": picked, "probe": probe}
                    gi += 1
                    launched += 1
                    self.state["next_group"], self.state["cursor"] = gi, cursor
                if not pending:
                    self.stop = "cap" if self.meter.exhausted else "too_many_drops"
                    break
                done, _ = await asyncio.wait(list(pending), return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    info = pending.pop(t)
                    g = t.result()
                    g.picked_step = info["picked_step"] if g.picked_step is None else g.picked_step
                    g.probe = info["probe"] if g.probe is None else g.probe
                    self.groups_seen += 1
                    self._count_masked(g)
                    why = self._verdict(g)
                    if g.masked:
                        self.quarantine.lost(g.task_id, probe=g.probe)  # infrastructure: never an observation of the task
                    elif why != "objective_changed":
                        phis = [float(r.progress["phi_final"]) if isinstance((r.progress or {}).get("phi_final"), int | float)
                                else None for r in g.records if not r.mask_sample and r.reward is not None]
                        if self.difficulty is not None:  # the sampler forwards the group to the quarantine and keeps p̂
                            self.difficulty.observe(g.task_id, draws_of(g.records), int(self.state["step"]), rewards=g.rewards(), phis=phis,
                                                    picked=g.picked_step, probe=g.probe)
                        else:
                            self.quarantine.observe(g.task_id, g.rewards(), phis, int(self.state["step"]), picked=g.picked_step, probe=g.probe)
                    self._q_flush()
                    if why is None:
                        batch.append(g)
                    else:
                        self._drop(g, why)
                        if why == "objective_changed":
                            self.stop = "objective_changed"
                if self.stop is None and self._masked_share_exceeded():
                    self.stop = "masked_frac"
                    break
                for g in [g for g in batch if self._stale(g)]:
                    batch.remove(g)
                    self._drop(g, "stale")
                if self.meter.exhausted:
                    self.stop = "cap"
                    break
                if len(batch) >= cfg.groups_per_step and self.stop is None:
                    if await self._health_paused():
                        self.stop = "framework_health"
                        break
                    try:
                        await self._step(batch[: cfg.groups_per_step])
                    except TinkerCapReached:
                        self.stop = "cap"
                        break
                    batch = batch[cfg.groups_per_step:]
                    launched = len(pending)
                    if self.state["step"] < cfg.steps:
                        self._start_health()
        except BaseException as e:
            failed = e
            raise
        finally:
            if self._health_task is not None:
                self._health_task.cancel()  # a check no step waits for any more (its thread finishes on its own)
            # async-RL audit R4: in flight at a stop (an error included): cancelled, never trained, never awaited to the end
            await self._stop_pending(self.stop or (f"error:{type(failed).__name__}" if failed is not None else "done"))
            await self._settle_cancelled(self.cancel_grace_s)
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
               "masked_stop": self.masked_stop, "health_stop": self.health_stop,
               "trained_episodes": eps, "usd_per_episode": (self.meter.spent / max(1, self.groups_seen * self.cfg.group_size)),
               "wall_s": round(wall_s, 1), "steps_log": self.step_rows, "codec": self.codec.settings() if hasattr(self.codec, "settings") else None,
               "quarantine": {"events": len(self.quarantine.events), "quarantined": sorted({e["task"] for e in self.quarantine.events
                                                                                            if e["event"] in ("quarantined", "requarantined")}),
                              "active": self.quarantine.active(int(self.state["step"])),
                              "q": self.cfg.quarantine_q, "r": self.cfg.quarantine_r},
               "difficulty": None if self.difficulty is None else {"sampler": "difficulty", "events": len(self.difficulty.events),
                                                                 "levels": dict(self.difficulty.current), "rates": self.difficulty.tracker.rows()}}
        _atomic(self.dir / "summary.json", out)
        self.tracking.summary({k: v for k, v in out.items() if k not in ("steps_log", "versions")})
        return out
