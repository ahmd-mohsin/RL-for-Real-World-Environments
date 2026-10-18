"""Group filters for slime (swarm-pipeline/00 §11 item 5, §11.3 item 5; 03 §4.2 item 5).

- ``group_filter`` (``--dynamic-sampling-filter-path``; called per GRPO group as ``fn(args, samples, **kw)`` and wrapped by
  ``call_dynamic_filter``, ``slime/rollout/filter_hub/base_types.py:30-46``), in this order:
  1. **infrastructure anywhere in the group drops the whole group** (reason ``crucible_infra_in_group``). A partial group
     is never trained, and the masked member is never a 0. slime then **refills**: the sync rollout over-samples until
     ``rollout_batch_size`` groups pass (``sglang_rollout.py:530-589``), the fully-async one keeps pulling
     (``fully_async_rollout.py:305-330``);
  2. the **staleness cap** (slime only logs staleness, ``slime/utils/staleness.py:10-29``): a group whose oldest weight
     version is more than ``tau`` behind the current one is dropped (``crucible_stale``). The current version is
     ``kw["current_weight_version"]``, ``args.crucible_current_weight_version`` or ``CRUCIBLE_SLIME_WEIGHT_VERSION``; unknown
     means the cap is not applied (the batch-level ``staleness_sample_filter`` still is);
  3. **nonzero reward spread** (slime's ``check_reward_nonzero_std``, ``dynamic_sampling_filters.py:9-17``): no spread,
     no signal.
- ``staleness_sample_filter`` (``--rollout-sample-filter-path``, ``fn(args, groups)`` mutating the batch in place,
  ``slime/rollout/base_types.py:66-73``): drops stale groups relative to the newest version in the batch (or the
  configured current version).

``tau``: ``args.crucible_max_staleness`` / ``CRUCIBLE_SLIME_MAX_STALENESS`` (weight updates), default 2.
"""

from __future__ import annotations

import os
import statistics
from typing import Any

from crucible.train.bridges.slime.types import flat, is_masked, sample_types, trajectories


def _int(x: Any) -> int | None:
    s = str(x)
    return int(s) if s.removeprefix("-").isdigit() else None


def tau_of(args: Any) -> int:
    v = getattr(args, "crucible_max_staleness", None)
    v = v if v is not None else os.environ.get("CRUCIBLE_SLIME_MAX_STALENESS")
    return int(v) if v not in (None, "") else 2


def current_version(args: Any, kw: dict[str, Any] | None = None) -> int | None:
    for v in ((kw or {}).get("current_weight_version"), getattr(args, "crucible_current_weight_version", None), os.environ.get("CRUCIBLE_SLIME_WEIGHT_VERSION")):
        if v not in (None, "") and (i := _int(v)) is not None:
            return i
    return None


def staleness(sample: Any, current: int) -> int | None:
    """slime's ``sample_staleness`` (``staleness.py:10-17``): current − the oldest version that produced a token."""
    vs = [_int(v) for v in sample.weight_versions or []]
    if not vs or any(v is None for v in vs) or max(vs) > current:  # type: ignore[type-var]
        return None
    return current - min(vs)  # type: ignore[type-var]


def group_max_staleness(samples: Any, current: int) -> int | None:
    ages = [a for s in flat(samples) if not is_masked(s) and (a := staleness(s, current)) is not None]
    return max(ages) if ages else None


def group_filter(args: Any, samples: Any, **kw: Any) -> Any:
    _, Out = sample_types()
    segs = flat(samples)
    if any(is_masked(s) for s in segs):
        return Out(keep=False, reason="crucible_infra_in_group")
    cur = current_version(args, kw)
    if cur is not None:
        age = group_max_staleness(segs, cur)
        if age is not None and age > tau_of(args):
            return Out(keep=False, reason="crucible_stale")
    rewards = [float(t.get_reward_value(args)) for t in trajectories(samples)]
    if len(rewards) < 2 or statistics.pstdev(rewards) <= 1e-6:
        return Out(keep=False, reason=f"zero_std_{round(rewards[0], 1) if rewards else 'none'}")
    return Out(keep=True)


def staleness_sample_filter(args: Any, groups: list[Any]) -> None:
    """Remove stale groups from the batch in place (see the module doc)."""
    cur = current_version(args)
    if cur is None:
        vs = [v for g in groups for s in flat(g) for x in s.weight_versions or [] if (v := _int(x)) is not None]
        if not vs:
            return
        cur = max(vs)
    tau = tau_of(args)
    keep = [g for g in groups if (a := group_max_staleness(g, cur)) is None or a <= tau]
    groups[:] = keep
