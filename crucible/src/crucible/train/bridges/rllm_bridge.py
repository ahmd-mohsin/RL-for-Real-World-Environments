"""The rLLM bridge: Crucible rollouts → rLLM ``Step`` / ``Trajectory`` / ``TrajectoryGroup`` (M12 audit H4 V26;
M12-sweep-G4 H5, pick 5).

Upstream: rLLM 4e828c8 (``RL-Environments/06-frameworks-and-env-code/infra-rllm-org-rllm``); rLLM is imported lazily, so
these functions need it installed. One rollout becomes one ``Trajectory`` per trained actor (``name`` = the actor: rLLM's
role, ``types.py:241-255``); a group of rollouts of one task becomes one ``TrajectoryGroup`` per role (``group_id``
``"<group>:<role>"``, which ``task_id``/``group_role`` split at the last colon, ``types.py:384-415``).

Per call, one ``Step`` built by ``Step.from_model_output`` (``types.py:225-238``) from a ``ModelOutput``
(``engine/rollout/rollout_engine.py:15-31``): ``prompt_ids`` = the call's **full** prompt, ``completion_ids`` and
``logprobs`` = the sampled tokens and their behaviour logprobs, ``finish_reason`` = ``Turn.finish``, ``weight_version`` =
the oldest policy version that produced the call (rLLM's async staleness reads it). Both backends consume it:

- **verl backend** reads only ``step.model_output`` and drops a step without one (``trainer/verl/transform.py:293-299``),
  then re-merges prefix-extending steps into one row per trajectory (``:302-397``: a compaction segment opens a new row
  with the same uid). Its advantage is one **scalar** per trajectory, ``float(step.advantage)``
  (``transform.py:592-597``), computed by rLLM from ``Trajectory.reward`` over the group (RL mode,
  ``algorithms/advantage.py:199-233``; it asserts every reward is set, ``:209``). So ``backend="verl"`` leaves
  ``Step.advantage`` None and sets ``Trajectory.reward``: a per-token list there raises ``TypeError``, and rLLM's
  precomputed path turns even a scalar into a list (``advantage.py:153-154``), so Crucible's dense turn credit cannot
  reach the verl backend: its advantage is rLLM's own GRPO/RLOO over the outcome.
- **tinker backend** takes per-token advantages, one per completion token (``trainer/tinker/transform.py:100-108``) under
  ``use_precomputed_advantage`` (``advantage.py:139-170``): ``backend="tinker"`` writes Crucible's per-token advantage
  (sequence advantage + turn credit, ``common.flatten``) on each step. Keep ``filter_uniform_groups`` off there: rLLM's
  buffer applies ``abs()`` to each step's advantage (``trainer/buffer.py:236-238``), which fails on a list.

``Step.reward`` carries the outcome on the rollout's final call only and ``done`` marks it; earlier calls carry 0.0 (a
multi-segment rollout used to repeat the outcome on every segment's last call). Masked, failed and token-incomplete
rollouts produce **no trajectory**: rLLM's GRPO then baselines over the live rollouts only, never over a 0.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from crucible.core.schema.rollout import RolloutRecord
from crucible.train.bridges.common import TrainSeq, flatten

BACKENDS = ("verl", "tinker")


def _rllm():
    from rllm.engine.rollout.rollout_engine import ModelOutput
    from rllm.types import Step, Trajectory, TrajectoryGroup

    return ModelOutput, Step, Trajectory, TrajectoryGroup


def _oldest(vs: Sequence[int | None]) -> int | None:
    ints = [int(v) for v in vs if v is not None]
    return min(ints) if ints else None


def steps(s: TrainSeq, *, backend: str = "verl", finishes: dict[str, str | None] | None = None) -> list[Any]:
    """One rLLM ``Step`` per call of a flattened sequence (``common.TrainSeq``)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    ModelOutput, Step, _, _ = _rllm()
    out = []
    last_call = s.call_spans[-1][0] if s.call_spans else None
    for cid, a, b in s.call_spans:
        prompt = list(s.prompt_ids) + list(s.response_ids[:a])
        mo = ModelOutput(prompt_ids=prompt, completion_ids=list(s.response_ids[a:b]), logprobs=[float(x) for x in s.logprobs[a:b]],
                         prompt_length=len(prompt), completion_length=b - a, finish_reason=(finishes or {}).get(cid),
                         weight_version=_oldest(s.versions[a:b]))
        st = Step.from_model_output(mo)
        st.id = cid
        final = s.last and cid == last_call
        st.reward = float(s.reward or 0.0) if final else 0.0
        st.done = final
        st.advantage = [float(x) for x in s.advantages[a:b]] if backend == "tinker" else None
        out.append(st)
    return out


def trajectories(ro: RolloutRecord, *, backend: str = "verl", seq_adv: float = 0.0, turn_weight: float = 0.0,
                 frozen_actors: Sequence[str] = ()) -> list[Any]:
    """One ``Trajectory`` per trained actor of the rollout (none when it is masked or token-incomplete)."""
    _, _, Trajectory, _ = _rllm()
    finishes = {t.call_id: t.finish for t in ro.turns}
    by_actor: dict[str, list[Any]] = defaultdict(list)
    seqs = [s for s in flatten(ro, seq_adv, turn_weight=turn_weight, frozen_actors=frozen_actors) if any(s.response_mask)]
    if not seqs:
        return []
    for actor in dict.fromkeys(s.actor for s in seqs):
        mine = [s for s in seqs if s.actor == actor]
        for k, s in enumerate(mine):
            s.last = k == len(mine) - 1  # the actor's final call closes its trajectory (reward, done)
            by_actor[actor] += steps(s, backend=backend, finishes=finishes)
    return [Trajectory(uid=f"{ro.rollout_id}:{actor}", name=actor, task=ro.task, steps=sts, reward=float(ro.reward),  # type: ignore[arg-type]
                       metadata={"crucible_rollout": ro.rollout_id, "truncated": ro.truncated, "stop": ro.stop})
            for actor, sts in by_actor.items()]


def trajectory_groups(records: Sequence[RolloutRecord | None], *, backend: str = "verl", group_of: Callable[[RolloutRecord], str] | None = None,
                      seq_adv: Callable[[RolloutRecord], float] | None = None, turn_weight: float = 0.0,
                      frozen_actors: Sequence[str] = ()) -> list[Any]:
    """``TrajectoryGroup`` per (group, role) over live rollouts; masked ones (and None) are left out of every group."""
    _, _, _, TrajectoryGroup = _rllm()
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for ro in records:
        if ro is None:
            continue
        g = group_of(ro) if group_of else (ro.group or ro.task)
        for tr in trajectories(ro, backend=backend, seq_adv=seq_adv(ro) if seq_adv else 0.0, turn_weight=turn_weight, frozen_actors=frozen_actors):
            groups[(g, tr.name)].append(tr)
    out = []
    for (g, role), trs in groups.items():
        vs = [st.weight_version for tr in trs for st in tr.steps if st.weight_version is not None]
        out.append(TrajectoryGroup(trajectories=trs, group_id=f"{g}:{role}", weight_version=min(vs) if vs else 0))
    return out
