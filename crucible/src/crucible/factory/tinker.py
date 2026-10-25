"""Tasks on Tinker: RL post-training and the benchmark pass (swarm-pipeline/09 §5, §7).

The factory joins the two sides the layering keeps apart: ``factory.envs.make_env`` (the task side) and
``train.bridges.tinker`` (sampler, trainer, meter), plus ``bench.lh_metrics`` for the report.

- ``train(tasks, config=...)``: ``make_env`` per task, ``objective_of(tasks)`` pinned, the targeting records (one per
  task, for this model: ``trainer.check_band``), ``TinkerTrainer.run``, and a receipt (``factory.receipts``) with the
  spend, the versions, the drops and a ``not_verified`` list.
- ``benchmark(tasks, model=..., checkpoint=..., k=...)``: ``k`` attempts per task on the base model (``checkpoint``
  None) or on sampler weights (``tinker://.../sampler_weights/...``), the study 08 §8 metrics of ``bench.lh_metrics``
  (``summarize``: pass@1, pass^k, final Φ, milestones lost, calibration, give-up, hacks, cost per solved task) plus
  unbiased pass@k for k in 1, 2, 4, ... k, pass^k for the same ks, **task-clustered** bootstrap CIs (tasks resampled,
  attempts kept together: attempts of one task are not independent) and USD per episode and per passing episode, from
  the meter's per-episode attribution. Sealed (held-out) tasks may be benchmarked; they never train.
- **The benchmark protocol** (the user's, 2026-10-20: the student under 30% before post-training, 50-60% after):
  ``benchmark(..., purpose="selection", seed=s0)`` picks tasks the base fails; a separate ``purpose="baseline"`` run with
  a new seed is the honest pre-score (selected tasks regress upward on re-test); ``compare(...)`` runs the ``control``
  (the base model re-run) and the ``trained`` adapter at the same settings and seed, reports pass@1 over k = 8 per task
  with task-clustered CIs and the paired trained − control difference with its CI and MDE (``lh_metrics.paired``), and
  ``check_protocol`` refuses (lists) a fresh baseline that reused the selection seed, a base arm with a checkpoint, and
  arms whose harness, tool cap, context, thinking setting, sampling parameters or task list differ (``arm_settings``,
  recorded in every report).
- **Power** (``bench.power``; rl-env-use's A/A gate, MIT): every arm's report carries ``power``, the A/A noise of its own
  attempts (pass@1 and mean reward) and the MDE of a paired comparison of its N tasks × k attempts; ``compare`` stamps
  ``power`` (pre-registered from the fresh baseline arm's attempts when its directory is given, else from the control
  arm, marked post hoc) with N, k, MDE and ``powered`` / ``underpowered`` against ``target_effect`` (default +0.20, the
  protocol's goal), and the ``headline`` starts with UNDERPOWERED when it is. It never blocks a run.
- ``download_adapter(path, out)``: the LoRA adapter of a sampler checkpoint (Tinker's signed archive URL), extracted
  with every member checked to stay inside ``out``.
- **Observability** (``observe=``, an ``observe.ObserveConfig``; None: off, or on when ``CRUCIBLE_OBSERVE=1``;
  swarm-pipeline/11): ``train`` logs every step to a ``crucible-train`` W&B run (the trainer's sink) and its episodes as
  Weave traces; ``benchmark`` logs the arm (``observe.bench.log_benchmark``) and ``compare`` the comparison
  (``log_compare``: the headline pre/post chart, the paired difference, the protocol check). Offline by default.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.envs import Timeouts, make_env, objective_of

NOT_VERIFIED_TRAIN = [
    "checkpoint storage cost (billed per GB-month by Tinker; counted, not priced)",
    "Tinker's own billing: the meter prices tokens from the pinned table; reconcile with the console or the billing usage endpoint",
    "the sampled tokens equal what Tinker trained on (both are our ids; the server's logprob recomputation is reported as kl_sample_vs_train)",
]


def _codec(model: str, enable_thinking: bool, codec: Any = None) -> Any:
    if codec is not None:
        return codec
    from crucible.train.bridges.tinker.codec import TinkerChatCodec, load_tokenizer
    from crucible.train.bridges.tinker.models import resolve

    m = resolve(model)
    return TinkerChatCodec(load_tokenizer(m), m, enable_thinking=enable_thinking)


def _backend(backend: Any) -> Any:
    if backend is not None:
        return backend
    from crucible.train.bridges.tinker.backend import RealTinker

    return RealTinker()


def task_meta(tasks: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """task id -> language, family, skill (the benchmark breakdowns): a task's own ``language`` / ``family`` / ``intent``
    when it has them, else the lane's (a kernel-lane task is CUDA, a window task Python; the family is the repository)."""
    out = {}
    for t in tasks:
        w = getattr(t, "window", None) or {}
        lane = type(t).__module__.rsplit(".", 1)[-1]
        out[str(t.task_id)] = {"language": getattr(t, "language", None) or ("cuda" if "kernel" in lane else "python"),
                               "family": getattr(t, "family", None) or (w.get("repo") if isinstance(w, Mapping) else None) or lane,
                               "skill": getattr(t, "intent", None) or "unassigned", "split": str(getattr(t, "split", "") or "") or None}
    return out


def _observe_cfg(observe: Any) -> Any:
    """``observe``: an ``ObserveConfig``, True, False or None (None: on only when ``CRUCIBLE_OBSERVE=1``)."""
    import os

    if observe is None:
        observe = os.environ.get("CRUCIBLE_OBSERVE") == "1"
    if observe is False:
        return None
    from crucible.observe import ObserveConfig

    return observe if isinstance(observe, ObserveConfig) else ObserveConfig.from_env()


async def train(tasks: Sequence[Any], *, config: Any, run_dir: Path, targeting: Mapping[str, Mapping[str, Any]] | None = None, backend: Any = None,
                codec: Any = None, phi_checkpoint: Any = None, tool_limit: int | None = None, timeouts: Timeouts | None = None,
                mode: str = "workspace", observe: Any = None, model: Any = None) -> dict[str, Any]:
    """``backend``: the Tinker service (default) or any ``train.bridges.tinker.backend.Backend``, e.g. the DeltaAI
    ``train.bridges.primerl.PrimeRLBackend``; ``model``: that backend's model row (``primerl.local_model``)."""
    from dataclasses import asdict

    from crucible.factory.receipts import make_receipt, write_receipt
    from crucible.train.bridges.tinker.trainer import TinkerTrainer

    envs = [make_env(t, checkpoint=phi_checkpoint, tool_limit=tool_limit, timeouts=timeouts, mode=mode) for t in tasks]
    ocfg = _observe_cfg(observe)
    obs = None
    if ocfg is not None:
        from crucible.observe.train_views import train_run

        t = timeouts or Timeouts()
        obs = train_run(asdict(config), run_dir=Path(run_dir), objective_id=objective_of(tasks), tasks=[str(x.task_id) for x in tasks],
                        harness=getattr(envs[0].harness, "id", None) if envs else None, task_meta=task_meta(tasks), cfg=ocfg,
                        extra={"tool_limit": tool_limit, "agent_s": t.agent_s, "rollout_s": t.rollout_s, "mode": mode,
                               "phi_checkpoint": asdict(phi_checkpoint) if phi_checkpoint is not None and hasattr(phi_checkpoint, "__dataclass_fields__") else None})
    tr = TinkerTrainer(envs, config=config, backend=_backend(backend), run_dir=Path(run_dir), codec=_codec(config.model, config.enable_thinking, codec),
                       objective_id=objective_of(tasks), targeting=targeting, sinks=[obs] if obs is not None else [], model=model)
    try:
        out = await tr.run()
    except BaseException:
        if obs is not None:
            obs.finish("error")
        raise
    rec = make_receipt(scope="tinker-train", runs={out["run"]: {k: out[k] for k in ("model", "steps_done", "stop", "objective_id", "sampler_path", "state_path",
                                                                                      "drops", "groups_seen", "versions")}},
                       not_verified=NOT_VERIFIED_TRAIN, seeds={"seed": config.seed}, spend=out["spend"])
    write_receipt(Path(run_dir) / "receipt.json", rec)
    if obs is not None:
        from crucible.observe.train_views import finish_train

        out["observe"] = finish_train(obs, out, Path(run_dir))
    return out


# ---- the benchmark pass ----------------------------------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k (Chen et al. 2021): 1 − C(n−c, k)/C(n, k); None when n < k."""
    if n < k or k < 1:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float | None:
    if n < k or k < 1:
        return None
    return math.comb(c, k) / math.comb(n, k)


def ks_upto(k: int) -> list[int]:
    out, x = [], 1
    while x <= k:
        out.append(x)
        x *= 2
    return out + ([k] if k not in out else [])


def clustered_ci(per_task: Mapping[str, Sequence[Any]], stat: Any, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict[str, Any]:
    """Bootstrap CI of ``stat(list of per-task episode lists)`` resampling **tasks** (attempts of a task stay together)."""
    units = [list(v) for _, v in sorted(per_task.items())]
    point = stat(units) if units else None
    if not units or point is None:
        return {"point": point, "lo": None, "hi": None, "units": len(units)}
    rng = random.Random(seed)
    boots = sorted(v for _ in range(n_boot) if (v := stat([rng.choice(units) for _ in units])) is not None)
    if not boots:
        return {"point": point, "lo": None, "hi": None, "units": len(units)}
    lo, hi = boots[int(alpha / 2 * len(boots))], boots[min(len(boots) - 1, int((1 - alpha / 2) * len(boots)))]
    return {"point": point, "lo": lo, "hi": hi, "units": len(units), "n_boot": n_boot, "level": 1 - alpha}


def _hack(ro: Any) -> bool:
    comps = ro.reward_components or {}
    return bool(comps.get("hack/confirmed") or comps.get("hack_suspect") or (ro.progress or {}).get("hack_suspect"))


def metrics(episodes: Sequence[Any], *, k: int, usd_total: float) -> dict[str, Any]:
    """``lh_metrics.summarize`` + pass@k / pass^k for 1, 2, 4, .. k, task-clustered CIs, cost per (passing) episode."""
    from crucible.bench.lh_metrics import summarize

    base = summarize(episodes, k=k)
    live = [e for e in episodes if not e.masked]
    per_task: dict[str, list[Any]] = {}
    for e in live:
        per_task.setdefault(e.task, []).append(e)

    def pak(units: list[list[Any]], kk: int, hat: bool = False) -> float | None:
        vals = [v for es in units if (v := (pass_hat_k if hat else pass_at_k)(len(es), sum(x.passed for x in es), kk)) is not None]
        return statistics.fmean(vals) if vals else None

    ks = ks_upto(k)
    passes = sum(e.passed for e in live)
    base.update({
        "pass@k": {str(kk): pak(list(per_task.values()), kk) for kk in ks},
        "pass^k": {str(kk): pak(list(per_task.values()), kk, hat=True) for kk in ks},
        "ci": {"pass@1": clustered_ci(per_task, lambda u: pak(u, 1)), f"pass@{k}": clustered_ci(per_task, lambda u: pak(u, k)),
               f"pass^{k}": clustered_ci(per_task, lambda u: pak(u, k, hat=True)),
               "mean_final_phi": clustered_ci(per_task, lambda u: statistics.fmean(f for es in u for e in es if (f := e.final_phi) is not None)
                                              if any(e.final_phi is not None for es in u for e in es) else None),
               "units": len(per_task), "unit": "task"},
        "usd_total": round(usd_total, 6), "usd_per_episode": usd_total / len(episodes) if episodes else None,
        "usd_per_passing_episode": usd_total / passes if passes else None,
    })
    return base


def policy_id(model: str, checkpoint: str | None = None) -> str:
    """A benchmark arm's policy id: ``solver:tinker:<model>@tinker`` (base) or ``...@tinker-adapter`` (sampler weights).
    ``audit.policy_model`` / ``factory.student.student_model_of`` read the model back, so the family rule and the
    dev-numerics check see it."""
    return f"solver:tinker:{model}@tinker{'-adapter' if checkpoint else ''}"


PURPOSES = ("selection", "baseline", "control", "trained", "smoke", "fairness")  # fairness: S15's frontier solvers (a strong solver passes)


def arm_settings(*, model: str, k: int, temperature: float, max_tokens: int, enable_thinking: bool, tool_limit: int | None, timeouts: Timeouts | None,
                 system: str | None, harness: str, context: int, codec: Mapping[str, Any] | None) -> dict[str, Any]:
    """What must be identical across the arms of one comparison (the harness, tool caps, context, thinking, sampling):
    everything but the checkpoint, the seed and the purpose."""
    t = timeouts or Timeouts()
    return {"model": model.split(":", 1)[0], "context": context, "k": k, "temperature": temperature, "max_tokens": max_tokens,
            "enable_thinking": enable_thinking, "tool_limit": tool_limit, "agent_s": t.agent_s, "rollout_s": t.rollout_s, "system": system or "default",
            "harness": harness, "codec": {k_: v for k_, v in (codec or {}).items() if k_ != "model"}}


async def benchmark(tasks: Sequence[Any], *, model: str, k: int = 8, run_dir: Path, cap_usd: float, checkpoint: str | None = None, backend: Any = None,
                    codec: Any = None, temperature: float = 1.0, max_tokens: int = 1024, enable_thinking: bool = False, phi_checkpoint: Any = None,
                    tool_limit: int | None = None, timeouts: Timeouts | None = None, concurrency: int = 4, system: str | None = None,
                    seed: int = 0, purpose: str = "baseline", observe: Any = None, meta: Mapping[str, Mapping[str, Any]] | None = None,
                    meter: Any = None, target_effect: float = 0.20) -> dict[str, Any]:
    """``k`` attempts per task of ``model`` (``checkpoint``: sampler weights, None = the base model). ``seed`` is the run's
    seed (attempt seeds derive from it; a fresh baseline uses a seed the selection run did not); ``purpose`` labels the
    arm in the benchmark protocol (``check_protocol``). The report records the arm's settings (``arm_settings``).
    ``meter``: a ``SpendMeter`` shared with other arms (one hard cap over a whole phase, e.g. S15 targeting's student and
    frontier arms); None = this arm's own meter at ``cap_usd`` (``run_dir/spend.jsonl``). ``usd_total`` is then this arm's
    own spend (its sampler's bookings: arms running concurrently on one meter never count each other's calls);
    per-episode USD comes from the meter's episode attribution either way."""
    from crucible.bench.lh_metrics import LHEpisode
    from crucible.core.schema.rollout import RolloutRecord
    from crucible.factory.traceaudit import regrade_source
    from crucible.runtime.infra import RetryRule
    from crucible.runtime.run import Runner
    from crucible.train.bridges.tinker.sampler import TinkerSampler
    from crucible.train.bridges.tinker.spend import SpendMeter

    if purpose not in PURPOSES:
        raise ValueError(f"purpose is one of {PURPOSES}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shared = meter is not None
    meter = meter if shared else SpendMeter(cap_usd, path=run_dir / "spend.jsonl")
    sampler = await TinkerSampler(_backend(backend), model, meter=meter, codec=_codec(model, enable_thinking, codec), path=checkpoint).start()
    llm = sampler.llm()
    episodes, rows = [], []
    harness = ""
    for t in tasks:
        env = make_env(t, checkpoint=phi_checkpoint, tool_limit=tool_limit, timeouts=timeouts)
        harness = getattr(env.harness, "id", type(env.harness).__name__)
        # the factory's policy-id form (``solver:<provider>:<model>@<backend>``): the trace audit reads the model and its family
        # from it (``audit.policy_model``), so a same-family auditor is refused; ``bench-<model>`` read as family "bench"
        pol = sampler.policy_ref(pid=policy_id(sampler.model.name, checkpoint), temperature=temperature, max_tokens=max_tokens,
                                 system=system, llm=llm, **env.policy_kwargs())
        rid = f"bench-{purpose}-s{seed}-{t.task_id}"
        spec = env.run_spec(run_id=rid, policies=[pol], k=k, concurrency=concurrency, retry=RetryRule(max_attempts=2), breaker=None, run_seed=seed,
                            sampling={"temperature": temperature, "max_tokens": max_tokens})
        await Runner(run_dir / "rollouts").run(spec)
        for adir in sorted((run_dir / "rollouts" / rid / "samples").glob("*/a*")):
            src = regrade_source(adir)  # a re-graded attempt (``regrade``): its own grade, the agent attempt's rollout
            if not (src / "rollout.json").exists():
                continue
            if adir.name[1:].isdigit() and (adir.parent / f"a{int(adir.name[1:]) + 1}").exists():
                continue  # a retried or re-graded attempt: the later one counts
            ro = RolloutRecord.model_validate_json((src / "rollout.json").read_text())
            if src != adir:
                rec = json.loads((adir / "record.json").read_text())
                ro.reward, ro.mask_sample = rec.get("reward"), bool(rec.get("mask_sample"))
            spend = sampler.per_episode.get(ro.rollout_id) or sampler.per_episode.get(adir.parent.name)
            masked = ro.mask_sample or ro.reward is None
            ep = LHEpisode.from_progress(str(t.task_id), ro.progress or {}, passed=not masked and float(ro.reward) >= 1.0 - 1e-9, masked=masked,
                                         hack=_hack(ro), tokens=sum(len(x.completion_token_ids or []) for x in ro.turns),
                                         cost_usd=spend.usd if spend else 0.0)
            episodes.append(ep)
            rec_meta = (json.loads((adir / "record.json").read_text()).get("meta") or {}) if (adir / "record.json").exists() else {}
            rep_meta = (json.loads((adir / "verifier" / "report.json").read_text()).get("meta") or {}) if (adir / "verifier" / "report.json").exists() else {}
            rows.append({"task": str(t.task_id), "rollout": ro.rollout_id, "attempt": f"{adir.parent.name}/{adir.name}", "reward": ro.reward, "masked": masked,
                         "passed": ep.passed, "final_phi": ep.final_phi, "milestones": rep_meta.get("milestones"), "gates": rec_meta.get("gates"),
                         "regraded_from": src.name if src != adir else None, "usd": round(ep.cost_usd, 6), "failure": ro.failure_kind,
                         "turns": len(ro.turns), "stop": ro.stop})
    settings = arm_settings(model=sampler.model.name, k=k, temperature=temperature, max_tokens=max_tokens, enable_thinking=enable_thinking,
                            tool_limit=tool_limit, timeouts=timeouts, system=system, harness=harness, context=sampler.model.context,
                            codec=(sampler.settings().get("codec") if isinstance(sampler.settings().get("codec"), dict) else None))
    out = {"model": sampler.model.name, "checkpoint": checkpoint, "purpose": purpose, "seed": seed, "k": k, "temperature": temperature,
           "enable_thinking": enable_thinking, "tasks": [str(t.task_id) for t in tasks], "settings": settings,
           "metrics": metrics(episodes, k=k, usd_total=sampler.usd if shared else meter.spent), "spend": meter.summary(), "codec": sampler.settings(),
           "power": arm_power(rows, n=len(tasks), k=k, target=target_effect, source=f"{purpose} arm (A/A of its own attempts)")}
    (run_dir / "episodes.jsonl").write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows))
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    if (ocfg := _observe_cfg(observe)) is not None:
        from crucible.observe.bench import log_benchmark

        out["observe"] = log_benchmark(out, run_dir, task_meta=meta or task_meta(tasks), cfg=ocfg)
    out["episodes"] = episodes  # LHEpisode objects for ``compare`` (not in metrics.json; episodes.jsonl has the rows)
    return out


async def regrade(tasks: Sequence[Any], *, model: str, k: int, run_dir: Path, select: Any, checkpoint: str | None = None, codec: Any = None,
                  temperature: float = 1.0, max_tokens: int = 1024, enable_thinking: bool = False, phi_checkpoint: Any = None, tool_limit: int | None = None,
                  timeouts: Timeouts | None = None, concurrency: int = 4, system: str | None = None, seed: int = 0, purpose: str = "baseline",
                  max_regrades: int = 2) -> dict[str, Any]:
    """Re-grade a ``benchmark`` arm's stored episodes after a verifier or path-check fix (``Runner.regrade``: the stored
    patch and the agent's own trail, no agent and no model call). The run spec is rebuilt exactly as ``benchmark`` built
    it (same policy id and config, harness, k, seed, sampling), so the Runner accepts it as the same run; ``select(record)``
    picks the samples (``RunRecord``: e.g. a failed ``path:*`` gate). The newest terminal row of a sample wins; the agent's
    attempt keeps its trail (``traceaudit.load_bundle`` and ``observe.traces`` read a re-graded attempt's trail from its
    source). Returns ``{task: RunSummary as a dict}``."""
    from crucible.runtime.infra import RetryRule
    from crucible.runtime.run import Runner
    from crucible.train.bridges.tinker.sampler import TinkerSampler
    from crucible.train.bridges.tinker.spend import SpendMeter

    sampler = TinkerSampler(None, model, meter=SpendMeter(0.0), codec=_codec(model, enable_thinking, codec), path=checkpoint)  # never started
    llm = sampler.llm()
    out: dict[str, Any] = {}
    for t in tasks:
        env = make_env(t, checkpoint=phi_checkpoint, tool_limit=tool_limit, timeouts=timeouts)
        pol = sampler.policy_ref(pid=policy_id(sampler.model.name, checkpoint), temperature=temperature, max_tokens=max_tokens, system=system, llm=llm,
                                 **env.policy_kwargs())
        rid = f"bench-{purpose}-s{seed}-{t.task_id}"
        spec = env.run_spec(run_id=rid, policies=[pol], k=k, concurrency=concurrency, retry=RetryRule(max_attempts=2), breaker=None, run_seed=seed,
                            sampling={"temperature": temperature, "max_tokens": max_tokens})
        s = await Runner(Path(run_dir) / "rollouts").regrade(spec, select=select, max_regrades=max_regrades)
        out[str(t.task_id)] = {"regraded": s.attempts, "terminal": s.terminal, "pass_at_1": s.pass_at_1, "mean_reward": s.mean_reward}
    return out


def arm_power(rows: Sequence[Mapping[str, Any]], *, n: int, k: int, target: float = 0.20, source: str = "") -> dict[str, Any]:
    """``bench.power.power_report`` of one arm's ``episodes.jsonl`` rows for a comparison of ``n`` tasks × ``k`` attempts:
    per metric (pass@1, mean reward) the A/A noise and the gate."""
    from crucible.bench.power import power_report

    return power_report(rows, n=n, k=k, target=target, source=source)


def read_episode_rows(arm_dir: Path | str) -> list[dict[str, Any]]:
    p = Path(arm_dir) / "episodes.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def comparison_power(*, n: int, k: int, target: float = 0.20, baseline_rows: Sequence[Mapping[str, Any]] = (),
                     control_rows: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """The comparison's power stamp (pass@1 is the headline metric; reward beside it): pre-registered from the fresh
    baseline arm's attempts (the arms not yet run), else from the control arm's (``post_hoc``: the same noise model, read
    after the fact). ``verdict`` / ``headline`` are pass@1's."""
    pre = bool(baseline_rows)
    rows, src = (baseline_rows, "baseline arm (pre-registered)") if pre else (control_rows, "control arm (post hoc)")
    rep = arm_power(rows, n=n, k=k, target=target, source=src)
    g = rep["pass"]["gate"]
    return {"verdict": g["verdict"], "headline": g["headline"], "n_tasks": n, "k": k, "mde": g.get("mde"), "target": target,
            "preregistered": pre, "source": src, "metrics": rep}


def check_protocol(arms: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The benchmark protocol (2026-10-20): ``selection`` (tasks chosen because the base failed them) and a separate fresh
    ``baseline`` with new seeds (the honest pre-score: selected tasks regress upward on re-test); a ``control`` (the base
    re-run) beside the ``trained`` adapter at the same settings; every arm on the same task list with identical settings
    (``arm_settings``). Returns the problems (empty: the comparison may be reported)."""
    out = []
    for need in ("baseline", "control", "trained"):
        if need not in arms:
            out.append(f"no {need} arm")
    if "selection" in arms and "baseline" in arms and arms["selection"].get("seed") == arms["baseline"].get("seed"):
        out.append("the fresh baseline reuses the selection run's seed: it is not a fresh re-test")
    for name in ("baseline", "control"):
        if name in arms and arms[name].get("checkpoint") is not None:
            out.append(f"the {name} arm is not the base model ({arms[name].get('checkpoint')})")
    if "trained" in arms and not arms["trained"].get("checkpoint"):
        out.append("the trained arm has no checkpoint")
    ref = next(iter(arms.values()), {})
    for name, a in arms.items():
        if a.get("settings") != ref.get("settings"):
            diff = sorted(k for k in set(a.get("settings") or {}) | set(ref.get("settings") or {}) if (a.get("settings") or {}).get(k) != (ref.get("settings") or {}).get(k))
            out.append(f"arm {name} differs in settings: {diff}")
        if sorted(a.get("tasks") or []) != sorted(ref.get("tasks") or []):
            out.append(f"arm {name} ran a different task list")
    return out


async def compare(tasks: Sequence[Any], *, model: str, checkpoint: str, run_dir: Path, cap_usd: float, k: int = 8, seed: int = 1000,
                  baseline: Mapping[str, Any] | None = None, selection: Mapping[str, Any] | None = None, n_boot: int = 4000, observe: Any = None,
                  arm_dirs: Mapping[str, Path | str] | None = None, target_effect: float = 0.20, **kw: Any) -> dict[str, Any]:
    """The control (base model) and the trained adapter at the same settings and seed on the same tasks, the paired
    trained − base difference of per-task pass@1 (k attempts each) with its task-clustered bootstrap CI, P(trained >
    control) and MDE (``bench.lh_metrics.paired``), ``check_protocol`` over every arm given (pass the fresh ``baseline``
    and the ``selection`` reports when they exist), and the power stamp (``comparison_power``: pre-registered from the
    baseline arm's ``episodes.jsonl`` in ``arm_dirs["baseline"]`` before the arms run). The cap is split evenly between
    the two arms run here."""
    from crucible.bench.lh_metrics import paired

    run_dir = Path(run_dir)
    ocfg = _observe_cfg(observe)
    kw["observe"] = ocfg if ocfg is not None else False
    kw["target_effect"] = target_effect
    base_rows = read_episode_rows(arm_dirs["baseline"]) if arm_dirs and arm_dirs.get("baseline") else []
    if base_rows:  # pre-registration: the stamp exists before any paid arm runs
        pre = comparison_power(n=len(tasks), k=k, target=target_effect, baseline_rows=base_rows)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "power.json").write_text(json.dumps(pre, indent=1, sort_keys=True, default=str))
    control = await benchmark(tasks, model=model, checkpoint=None, k=k, run_dir=run_dir / "control", cap_usd=cap_usd / 2, seed=seed, purpose="control", **kw)
    trained = await benchmark(tasks, model=model, checkpoint=checkpoint, k=k, run_dir=run_dir / "trained", cap_usd=cap_usd / 2, seed=seed, purpose="trained", **kw)
    arms = {"control": control, "trained": trained, **({"baseline": baseline} if baseline else {}), **({"selection": selection} if selection else {})}
    problems = check_protocol(arms)
    diff = paired(control["episodes"], trained["episodes"], metric="pass", n_boot=n_boot, seed=seed)
    power = comparison_power(n=len(tasks), k=k, target=target_effect, baseline_rows=base_rows, control_rows=read_episode_rows(run_dir / "control"))
    strip = lambda a: {k_: v for k_, v in a.items() if k_ != "episodes"}
    md = diff.get("mean_diff")
    result = (f"trained - control pass@1 = {md:+.3f}, CI [{diff['ci'][0]:+.3f}, {diff['ci'][1]:+.3f}], P(trained > control) {diff['p_better']:.2f}"
              if md is not None else "trained - control: no shared task")
    headline = (power["headline"] + ": " + result) if power["verdict"] != "powered" else result
    out = {"model": model, "checkpoint": checkpoint, "k": k, "seed": seed, "protocol_problems": problems, "settings": control["settings"],
           "power": power, "headline": headline,
           "pass@1": {"control": control["metrics"]["pass@1"], "trained": trained["metrics"]["pass@1"],
                      **({"baseline": baseline["metrics"]["pass@1"]} if baseline else {}), **({"selection": selection["metrics"]["pass@1"]} if selection else {})},
           "ci": {"control": control["metrics"]["ci"]["pass@1"], "trained": trained["metrics"]["ci"]["pass@1"]},
           "paired_trained_minus_control": diff, "arms": {n: strip(a) for n, a in arms.items()}}
    (run_dir / "compare.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    if ocfg is not None:
        from crucible.observe.bench import log_compare

        out["observe"] = log_compare(out, run_dir, task_meta=kw.get("meta") or task_meta(tasks), cfg=ocfg, arm_dirs=arm_dirs)
    return out


async def download_adapter(path: str, out_dir: Path, *, backend: Any = None) -> dict[str, Any]:
    """Fetch and extract the LoRA adapter of a sampler checkpoint; returns the files and their sizes."""
    import httpx

    url = await _backend(backend).archive_url(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tar = out_dir / "adapter.tar"
    async with httpx.AsyncClient(timeout=600, follow_redirects=True) as c, c.stream("GET", url) as r:
        r.raise_for_status()
        with tar.open("wb") as f:
            async for chunk in r.aiter_bytes():
                f.write(chunk)
    root = out_dir.resolve()
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            dest = (root / m.name).resolve()
            if not (dest == root or root in dest.parents) or m.issym() or m.islnk():
                raise ValueError(f"unsafe archive member {m.name!r}")
        tf.extractall(root, filter="data")
    files = {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file() and p != tar}
    return {"path": path, "files": files, "archive_bytes": tar.stat().st_size}
