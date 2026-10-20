"""Per-trainer field mappings over ``common.TrainSeq`` (research m11-m12/01, /03, /04), each checked against the pinned
upstream code in ``tests/unit/test_m12_audit_bridges.py``:

- verl ``AgentLoopOutput`` (verl 5eb4826 ``verl/experimental/agent_loop/agent_loop.py:88-115``): ``verl_agent_loop_output``
  here; the per-token credit travels as ``rm_scores`` (``verl_bridge``, ``verl_agent_loop``, ``verl_plugin``);
- SkyRL ``GeneratorOutput`` (SkyRL 3d0ae22 ``skyrl/train/generators/base.py:40-69``): ``skyrl_bridge``;
- TRL ``rollout_func`` return (TRL 72cf37d ``trl/trainer/grpo_trainer.py``): ``trl_rollout`` here, driven by ``trl_bridge``;
- rLLM ``Step``/``Trajectory`` (rLLM 4e828c8 ``rllm/types.py``): ``rllm_bridge``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from crucible.train.bridges.common import TrainSeq

VERL_ESTIMATOR = "crucible_delta"  # verl_bridge.ESTIMATOR (that module imports torch; the routing below must not)


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """``cfg[key]`` for a dict, an OmegaConf ``DictConfig`` (``.get``) or an attribute config; ``default`` when absent."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:  # noqa: BLE001, S110 - a structured config without the key: fall back to the attribute
            pass
    return getattr(cfg, key, default)


def verl_mask_route(trainer_config: Any) -> str:
    """How a masked or untrainable session reaches verl (M11 lab audit R-1), from verl's whole trainer config:

    - ``"v1_placeholder"``: v1 (``trainer.use_v1``, default true, ``ppo_trainer.yaml:232``) with ``crucible_delta``: one row
      with no trainable token (``response_mask=[0]``), which ``crucible_delta`` leaves out of the group mean;
    - ``"v1_drop"``: v1 with any other estimator: the session returns ``[]``, which v1 skips (``agent_loop_tq.py:153-157``)
      while the uid group stays ``finished``. A stock estimator (``core_algos.py:267-331``, GRPO) would count a placeholder's
      0.0 in the group mean;
    - ``"legacy_placeholder"``: the legacy trainer (``use_v1: false``) with ``crucible_delta``. It has no skip channel (one
      output per session) and is deprecated (``ray_trainer.py:285``), so any other estimator is refused here.

    A config without ``algorithm`` (or without ``trainer``) still routes: an unknown estimator is a stock one. Refused too:
    ``v1_drop`` with ``algorithm.filter_groups.enable`` (verl's group filter raises for a finished uid with no row,
    ``replay_buffer.py:279-306``, which is what a uid whose every session is masked becomes). The route follows
    ``trainer.use_v1``, but some entrypoints (``verl/experimental/one_step_off_policy``) run the legacy worker under
    ``use_v1: true``; ``CrucibleAgentLoop.run`` refuses a legacy call on a ``v1_drop`` route itself."""
    est = cfg_get(cfg_get(trainer_config, "algorithm"), "adv_estimator")
    est = str(est) if est is not None else None
    use_v1 = cfg_get(cfg_get(trainer_config, "trainer"), "use_v1", True)
    if use_v1 is not None and not use_v1:
        if est != VERL_ESTIMATOR:
            raise ValueError(f"crucible_agent on verl's legacy trainer (trainer.use_v1: false) needs algorithm.adv_estimator={VERL_ESTIMATOR}, "
                             f"not {est!r}: the legacy trainer has no way to skip a masked session, so its placeholder row (reward 0.0) "
                             "would join the stock estimator's group mean as a real 0. Use trainer.use_v1=true or crucible_delta.")
        return "legacy_placeholder"
    if est == VERL_ESTIMATOR:
        return "v1_placeholder"
    if cfg_get(cfg_get(cfg_get(trainer_config, "algorithm"), "filter_groups"), "enable", False):
        raise ValueError(f"crucible_agent with algorithm.adv_estimator={est!r} and algorithm.filter_groups: a masked session sends no row, "
                         "and a uid whose every session is masked has none, which verl's group filter refuses ('Finished groups are missing "
                         f"DAPO metric'). Use algorithm.adv_estimator={VERL_ESTIMATOR} (it keeps a placeholder row) or disable filter_groups.")
    return "v1_drop"


def stamp_verl_steps(extra_fields: Sequence[dict[str, Any]], global_steps: Any) -> None:
    """Fill the ``min/max_global_steps`` tags verl v1 converts with ``np.array(..., dtype=int)`` (``trainer_base.py:1901-1902``;
    a ``None`` raises TypeError after the actor update; M11 lab audit R-1). ``extra_fields``: one session's outputs. A row
    with no tag (a placeholder, or a sequence of frozen actors only, whose ``versions_seen`` is empty) takes the min and
    max of its siblings' tags, else the trainer's current step (``global_steps``, from the run kwargs): it carries no
    sampled token, so it is neither stale nor spans versions (XiaomiMiMo-verl ``recipes/general/agent_loop.py:685-713``
    stamps the same). With no sibling tag and no current step (the legacy worker, which tolerates None) the row is left."""
    lo = [int(e["min_global_steps"]) for e in extra_fields if e.get("min_global_steps") is not None]
    hi = [int(e["max_global_steps"]) for e in extra_fields if e.get("max_global_steps") is not None]
    step = int(global_steps) if global_steps is not None else None
    for e in extra_fields:
        if e.get("min_global_steps") is None:
            e["min_global_steps"] = min(lo) if lo else step
        if e.get("max_global_steps") is None:
            e["max_global_steps"] = max(hi) if hi else step


def verl_agent_loop_output(s: TrainSeq, *, response_length: int | None = None) -> dict[str, Any]:
    """The fields of verl's ``AgentLoopOutput`` (verl 5eb4826 ``verl/experimental/agent_loop/agent_loop.py:88-115``):
    ``metrics`` is required (``:107``, ``AgentLoopMetrics`` ``:79-85``; M12 audit V19), the response is cut at
    ``rollout.response_length`` like verl's own loops (``single_turn_agent_loop.py:91-94``), ``min/max_global_steps``
    are the versions that sampled the trained tokens (read for staleness at ``v1/agent_loop_tq.py:224-227``), and
    ``reward_extra_info`` is present (``agent_loop_tq.py:175-178`` copies it across a session's outputs).
    A sequence with no sampled token of a trained actor leaves the step tags None here; ``stamp_verl_steps`` fills them.
    The per-token credit does not travel here: ``verl_agent_loop.CrucibleAgentLoopOutput`` writes it as ``rm_scores``."""
    n = len(s.response_ids) if response_length is None else min(response_length, len(s.response_ids))
    vs = s.versions_seen
    return {"prompt_ids": list(s.prompt_ids), "response_ids": s.response_ids[:n], "response_mask": s.response_mask[:n],
            "response_logprobs": [float(x) for x in s.logprobs[:n]], "reward_score": float(s.reward or 0.0), "num_turns": s.num_turns,
            "metrics": {},
            "extra_fields": {"min_global_steps": min(vs) if vs else None, "max_global_steps": max(vs) if vs else None,
                             "reward_extra_info": {}, "turn_scores": [], "tool_rewards": [],
                             "crucible_rollout": s.rollout_id, "crucible_segment": s.segment, "crucible_actor": s.actor}}


def skyrl_generator_output(records: Sequence[Any], **kw: Any) -> dict[str, Any]:
    """SkyRL ``GeneratorOutput`` for one input batch: ``records[i]`` (a ``RolloutRecord`` or None) is input row ``i``.
    Every key, one row per input (or step-wise rows), masked rollouts aligned and out of the baseline:
    ``skyrl_bridge.generator_output`` (M12 audit V25)."""
    from crucible.train.bridges.skyrl_bridge import generator_output

    return generator_output(records, **kw)


def trl_rollout(seqs: Sequence[TrainSeq | None], *, rewards: Sequence[float | None] | None = None, masked: Sequence[str | None] | None = None,
                pad_id: int = 0) -> dict[str, Any]:
    """TRL ``rollout_func`` return, one row per received prompt (TRL 72cf37d ``grpo_trainer.py:2144-2152`` required keys
    ``prompt_ids``/``completion_ids``/``logprobs``, flat per-token logprobs used as ``sampling_per_token_logps``
    ``:2392-2406``; ``env_mask`` → ``tool_mask`` ``:2199-2201``; every other key is merged into ``inputs[i]`` for the reward
    functions ``:2670-2677``). A ``None`` row (masked episode) is a placeholder that keeps the alignment: one ``pad_id``
    token (TRL reads ``ids[-1]``, ``:2224``), ``env_mask=[0]`` (no trainable token), logprob 0.0 (never read), and
    ``crucible_reward=None`` (unscorable: NaN in ``rewards_per_func``, out of the group baseline, advantage 0, ``:2685-2740``)."""
    rows = list(seqs)
    rw = list(rewards) if rewards is not None else [s.reward if s is not None else None for s in rows]
    why = list(masked) if masked is not None else [None if s is not None else "masked" for s in rows]
    if not (len(rw) == len(why) == len(rows)):
        raise ValueError(f"{len(rows)} rows, {len(rw)} rewards, {len(why)} mask reasons: one per received prompt")
    out: dict[str, list[Any]] = {"prompt_ids": [], "completion_ids": [], "logprobs": [], "env_mask": [], "crucible_reward": [],
                                 "crucible_versions": [], "crucible_masked": []}
    for s, r, w in zip(rows, rw, why, strict=True):
        if s is None:
            out["prompt_ids"].append([pad_id])
            out["completion_ids"].append([pad_id])
            out["logprobs"].append([0.0])
            out["env_mask"].append([0])
            out["crucible_reward"].append(None)
            out["crucible_versions"].append([None])
            out["crucible_masked"].append(w or "masked")
            continue
        out["prompt_ids"].append(s.prompt_ids)
        out["completion_ids"].append(s.response_ids)
        out["logprobs"].append(s.logprobs)
        out["env_mask"].append(s.response_mask)
        out["crucible_reward"].append(r)
        out["crucible_versions"].append(s.versions)
        out["crucible_masked"].append(None)
    return out


def rllm_steps(s: TrainSeq, *, backend: str = "verl") -> list[Any]:
    """rLLM ``Step`` objects (``ModelOutput``-backed, needs rLLM installed), one per call: ``rllm_bridge.steps``
    (M12 audit V26). ``backend="verl"``: no advantage (rLLM computes a scalar from ``Trajectory.reward``);
    ``"tinker"``: the per-token advantage. Whole rollouts and groups: ``rllm_bridge.trajectory_groups``."""
    from crucible.train.bridges.rllm_bridge import steps

    return steps(s, backend=backend)
