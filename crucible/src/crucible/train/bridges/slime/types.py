"""slime's ``Sample`` and ``DynamicFilterOutput``: the real classes when slime is importable, else mirrors.

The mirrors copy, field for field, the parts of THUDM slime 859938d (Apache-2.0) the bridge reads or writes:
- ``Sample``: ``slime/utils/types.py:107-169`` (group/index/rollout ids, prompt, tokens, response, reward, loss mask,
  weight versions, rollout log probs, ``remove_sample``, ``Status`` :148-156, metadata, session id) and
  ``get_reward_value`` (:270-271). Multimodal, top-k / top-p replay, routed experts and the trace fields are left out:
  the coding bridge never sets them.
- ``DynamicFilterOutput``: ``slime/rollout/filter_hub/base_types.py:5-14``.

``sample_types()`` returns ``(Sample, DynamicFilterOutput)`` from slime when it is installed (training), else the mirrors
(tests, the equivalence run). Code that builds samples always goes through it, so the same code runs under slime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class MirrorSample:
    group_index: int | None = None
    index: int | None = None
    rollout_id: int | None = None
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: list[float] | None = None
    remove_sample: bool = False

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)
    generate_function_path: str | None = None
    custom_rm_path: str | None = None
    train_metadata: dict | None = None
    session_id: str | None = None
    non_generation_time: float = 0.0

    def get_reward_value(self, args: Any) -> float:
        key = getattr(args, "reward_key", None)
        return self.reward if not key else self.reward[key]  # type: ignore[index,return-value]


@dataclass
class MirrorDynamicFilterOutput:
    keep: bool
    reason: str | None = None
    keep_when_insufficient: bool = False
    raw_reward_stats: tuple[float, int] | None = None


def sample_types() -> tuple[type, type]:
    """(Sample, DynamicFilterOutput): slime's when importable, else the mirrors."""
    try:
        from slime.rollout.filter_hub.base_types import DynamicFilterOutput
        from slime.utils.types import Sample
    except ImportError:
        return MirrorSample, MirrorDynamicFilterOutput
    return Sample, DynamicFilterOutput


def flat(samples: Any) -> list[Any]:
    """A group (``list[Sample | list[Sample]]``) or a batch as one list of segments."""
    out: list[Any] = []
    for s in samples:
        if isinstance(s, list):
            out += flat(s)
        else:
            out.append(s)
    return out


def trajectories(samples: Any) -> list[Any]:
    """One terminal segment per trajectory (slime counts a trajectory's reward once: ``filter_hub/base_types.py:33-36``)."""
    return [s[-1] if isinstance(s, list) else s for s in samples]


def rollout_key(s: Any, position: int) -> Any:
    """slime's trajectory key (``batch_builder.py:221-223``): ``rollout_id``, else ``index``, else the position."""
    rid = s.rollout_id if getattr(s, "rollout_id", None) is not None else getattr(s, "index", None)
    return ("rollout", rid) if rid is not None else ("position", position)


def is_masked(s: Any) -> bool:
    """A removed sample (infrastructure, never trained, never a 0: V8)."""
    return bool(getattr(s, "remove_sample", False)) or bool((getattr(s, "metadata", None) or {}).get("crucible_masked"))
