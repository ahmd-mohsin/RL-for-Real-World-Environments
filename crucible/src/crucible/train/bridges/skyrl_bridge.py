"""The SkyRL bridge: Crucible rollouts → SkyRL ``GeneratorOutput`` (M12 audit H4 V25, V28; M12-sweep-G4 H4, pick 4).

Upstream: SkyRL 3d0ae22 (``RL-Environments/06-frameworks-and-env-code/infra-novasky-ai-skyrl``). A Crucible generator
(``GeneratorInterface.generate``, ``skyrl/train/generators/base.py:75-87``) runs one World episode per input row and returns
``generator_output(records, trajectory_ids=input_batch["trajectory_ids"], step_wise=cfg.generator.step_wise_trajectories)``.

What SkyRL reads, and what this module writes for it:

- **Every ``GeneratorOutput`` key** (``base.py:40-69``). ``rollout_metrics`` is indexed unconditionally by
  ``RayPPOTrainer.generate`` (``skyrl/train/trainer.py:1068-1070``) before validation: Crucible's counts go there
  (``crucible/rollouts``, ``/masked``, ``/truncated``, ``/segments_dropped``). Side channels Crucible does not produce
  (routed experts, sample support, pixel values, generation times) are ``None``, which the validator skips
  (``trainer_utils.py:706-723, 778-819``).
- **Row count** (``validate_generator_output``, ``trainer_utils.py:680-776``). Without ``step_wise`` the trainer needs
  exactly one row per input (``:698-701``), and the mini-batch boundaries need ``n_samples_per_prompt`` rows per uid
  (``skyrl/train/dataset/preprocess.py:523-529``). So a rollout that flattens into several sequences (compaction
  segments, a second trained actor) contributes its *last* trained sequence; the dropped ones are counted in
  ``crucible/segments_dropped`` (train them with ``step_wise``). With ``step_wise`` every trained sequence is a row:
  ``trajectory_ids`` repeat the input's id, rows of one trajectory are contiguous and ``is_last_step`` closes each
  (``_validate_step_wise_fields``, ``trainer_utils.py:822-895``). SkyRL computes the advantage on the last step and
  broadcasts it with ``cumsum(is_last_step)`` (``trainer.py:1211-1247``), so only the last step's reward is read; earlier
  steps carry 0.0.
- **Rewards**: ``List[float]`` by default, which SkyRL turns into a per-token vector with the reward on the last response
  token (``trainer.py:1150-1155``); ``token_rewards=True`` writes that vector here, with the outcome on the last *sampled*
  token. Every built-in outcome estimator sums the row (``ppo_utils.py:1367``), so both give the same advantage.
- **Masked rollouts are never trained as 0.** A masked, failed or token-incomplete episode has no trainable token
  (``loss_masks`` all 0, as SkyRL's own overlong filter does, ``generators/utils.py:396-420``). Its group baseline must
  not see a 0 either: SkyRL's estimators average every row of a uid, live or not (``ppo_utils.py:1367-1390``), and only
  the zero-variance filter knows about dead rows (``trainer_utils.py:631-676``). So:
  * ``step_wise``: the masked rollout has **no row at all** (the trainer takes the uids from ``trajectory_ids``,
    ``trainer.py:379-382``), so the baseline is exactly the live rollouts'. A group whose every member is masked keeps
    one loss-masked placeholder row per member (every uid must appear once, ``preprocess.py:508-511``);
  * one row per input: a loss-masked placeholder whose reward is the **mean of its group's live rewards** (0.0 when none
    is live). The group mean is then exactly the live mean: exact for Dr.GRPO (``grpo_norm_by_std=False``) and MaxRL,
    which divide by nothing or by the mean; GRPO's std and RLOO's leave-one-out mean still count the placeholder (use
    ``step_wise`` when that matters).
- **Stop reasons** (V28): ``"length"`` when the rollout was truncated (its last sequence) or the sequence's last call
  finished on length, ``"abort"`` for an aborted call or a masked placeholder, else ``"stop"`` (vLLM's values, which
  SkyRL's overlong filtering and eval dumps read: ``generators/utils.py:396-420``, ``trainer_utils.py:367``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from crucible.core.schema.rollout import RolloutRecord, full_prompts
from crucible.train.bridges.common import TrainSeq, flatten


@dataclass
class TrajectoryID:
    """Shape of SkyRL's ``TrajectoryID`` (``generators/base.py:16-22``), used when the caller has none to pass on."""

    instance_id: str
    repetition_id: int

    def to_string(self) -> str:
        return f"{self.instance_id}_{self.repetition_id}"


def _stop_reason(s: TrainSeq) -> str:
    if (s.last and s.truncated) or s.finish == "length":
        return "length"
    if s.finish == "abort":
        return "abort"
    return "stop"


def _token_rewards(s: TrainSeq, reward: float) -> list[float]:
    r = [0.0] * len(s.response_ids)
    last = max((i for i, m in enumerate(s.response_mask) if m), default=len(r) - 1)
    if r:
        r[last] = float(reward)
    return r


def _placeholder(ro: RolloutRecord | None, prompt: Sequence[int] | None, pad_token_id: int) -> TrainSeq:
    """A row with no trainable token: the input's prompt (or the rollout's first prompt) and one pad token, mask 0."""
    p = list(prompt) if prompt else []
    if not p and ro is not None and ro.turns:
        p = list(full_prompts(ro).get(ro.turns[0].call_id) or [])
    return TrainSeq(ro.rollout_id if ro is not None else "", "", 0, p or [pad_token_id], response_ids=[pad_token_id], response_mask=[0],
                    logprobs=[0.0], advantages=[0.0], versions=[None], finish="abort")


def generator_output(records: Sequence[RolloutRecord | None], *, trajectory_ids: Sequence[Any] | None = None, step_wise: bool = False,
                     token_rewards: bool = False, frozen_actors: Sequence[str] = (), prompt_token_ids: Sequence[Sequence[int]] | None = None,
                     pad_token_id: int = 0) -> dict[str, Any]:
    """One ``GeneratorOutput`` for one ``GeneratorInput`` batch: ``records[i]`` is input row ``i``'s episode (None: the
    episode failed). ``trajectory_ids``: the input's ``TrajectoryID``s (``prepare_generator_input``,
    ``generators/utils.py:591-600``), else ``(record.group or record.task, i)``. ``prompt_token_ids``: the input rows'
    prompts, for placeholder rows of episodes that recorded none."""
    n = len(records)
    if trajectory_ids is None:
        trajectory_ids = [TrajectoryID(str((ro.group or ro.task) if ro is not None else f"row{i}"), i) for i, ro in enumerate(records)]
    if len(trajectory_ids) != n:
        raise ValueError(f"{len(trajectory_ids)} trajectory ids for {n} rollouts")
    seqs = [[s for s in flatten(ro, frozen_actors=frozen_actors) if any(s.response_mask)] if ro is not None else [] for ro in records]
    live = [bool(x) for x in seqs]
    uid = [str(getattr(t, "instance_id", t)) for t in trajectory_ids]
    by_uid: dict[str, list[float]] = defaultdict(list)
    for i in range(n):
        if live[i]:
            by_uid[uid[i]].append(float(records[i].reward))  # type: ignore[union-attr]
    group_live = {u: (sum(v) / len(v)) for u, v in by_uid.items()}

    rows: list[tuple[TrainSeq, float, Any, bool]] = []  # (sequence, reward, trajectory id, is_last_step)
    dropped = 0
    for i in range(n):
        prompt = prompt_token_ids[i] if prompt_token_ids is not None else None
        if not live[i]:
            if step_wise and uid[i] in group_live:
                continue  # its live siblings carry the uid: no row, no baseline entry
            rows.append((_placeholder(records[i], prompt, pad_token_id), group_live.get(uid[i], 0.0), trajectory_ids[i], True))
            continue
        R = float(records[i].reward)  # type: ignore[union-attr]
        if step_wise:
            for k, s in enumerate(seqs[i]):
                last = k == len(seqs[i]) - 1
                rows.append((s, R if last else 0.0, trajectory_ids[i], last))
        else:
            dropped += len(seqs[i]) - 1
            rows.append((seqs[i][-1], R, trajectory_ids[i], True))
    masked = sum(1 for x in live if not x)
    return {
        "prompt_token_ids": [list(s.prompt_ids) for s, *_ in rows],
        "response_ids": [list(s.response_ids) for s, *_ in rows],
        "rewards": [_token_rewards(s, r) if token_rewards else float(r) for s, r, *_ in rows],
        "loss_masks": [list(s.response_mask) for s, *_ in rows],
        "stop_reasons": [_stop_reason(s) for s, *_ in rows],
        "rollout_metrics": {"crucible/rollouts": n, "crucible/masked": masked,
                            "crucible/truncated": sum(1 for ro, ok in zip(records, live, strict=True) if ok and ro.truncated),  # type: ignore[union-attr]
                            "crucible/segments_dropped": dropped},
        "rollout_logprobs": [[float(x) for x in s.logprobs] for s, *_ in rows],
        "trajectory_ids": [t for _, _, t, _ in rows],
        "trajectory_generation_times": None,
        "trajectory_time_splits": None,
        "rollout_expert_indices": None,
        "rollout_sample_support": None,
        "is_last_step": [bool(e) for *_, e in rows] if step_wise else None,
        "env_metrics": None,
        "pixel_values": None,
        "image_grid_thw": None,
    }
