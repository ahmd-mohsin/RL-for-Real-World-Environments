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
- **tinker backend** takes per-token advantages, one per completion token (``trainer/tinker/transform.py:100-108``), and
  trains them **as final**: on rLLM's default (sync) path ``transform_trajectory_groups_to_datums`` skips its own
  estimator whenever any step carries an advantage (``transform.py:151-156``), whatever ``use_precomputed_advantage``
  says; only the async buffer path (``trainer/buffer.py:228-231``) reads that flag, and with it off replaces the lists by
  rLLM's outcome GRPO with a warning (``advantage.py:203-207``). A list of the wrong length fails an assertion and the
  whole trajectory is dropped (``transform.py:105,109,192``). So ``backend="tinker"`` writes the finished advantage on
  each step: the group's outcome advantage (R − mean over its live rollouts, ``train.credit.outcome_advantages``,
  unless ``seq_adv`` is given) plus, with ``turn_weight > 0``, M6's turn credit through the same group rule the torch
  trainer applies (``train.credit.credited_records``; M6 R1: raw ΔΦ reached rLLM, +0.1 on a gate-failed call where the
  rule gives −0.5). On the async path set ``use_precomputed_advantage=true``. Keep ``filter_uniform_groups`` off: rLLM's
  buffer applies ``abs()`` to each step's advantage (``trainer/buffer.py:236-238``), which fails on a list.
- ``turn_weight > 0`` with ``backend="verl"`` is refused: that backend reads one float per trajectory
  (``trainer/verl/transform.py:598``), so turn credit cannot travel and would be dropped silently.

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
    """One ``Trajectory`` per trained actor of the rollout (none when it is masked or token-incomplete). Turn credit is a
    group rule (centred, bounded, gate-aware: ``train.credit.turn_credit_map``), so one rollout cannot carry it:
    ``turn_weight > 0`` is refused here; ``trajectory_groups`` applies it."""
    if turn_weight:
        raise ValueError("turn credit is computed over a GRPO group: use trajectory_groups(records, turn_weight=...), not trajectories()")
    return _trajectories(ro, backend=backend, seq_adv=seq_adv, turn_weight=0.0, frozen_actors=frozen_actors)


def _trajectories(ro: RolloutRecord, *, backend: str, seq_adv: float, turn_weight: float, frozen_actors: Sequence[str]) -> list[Any]:
    """``trajectories`` for a record whose ``Turn.advantage`` is already the rule's credit / ``turn_weight``."""
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
    """``TrajectoryGroup`` per (group, role) over live rollouts; masked ones (and None) are left out of every group.

    ``backend="tinker"``: each step's advantage is final (module docstring): ``seq_adv(record)`` if given, else the
    group's outcome advantage, plus M6's turn credit when ``turn_weight > 0``. ``backend="verl"`` carries no per-token
    advantage, so ``turn_weight > 0`` is refused."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if turn_weight and backend == "verl":
        raise ValueError("turn_weight > 0 with backend='verl': rLLM's verl backend reads one advantage per trajectory "
                         "(rllm/trainer/verl/transform.py:598), so turn credit would be dropped. Use backend='tinker' or turn_weight=0.")
    from crucible.train.credit import credited_records, outcome_advantages, require_turn_credit

    live = [ro for ro in records if ro is not None]
    keys = [group_of(ro) if group_of else (ro.group or ro.task) for ro in live]
    outcome = outcome_advantages(live, keys) if backend == "tinker" else {}
    if turn_weight:
        require_turn_credit(live, frozen_actors, where="rllm_bridge.trajectory_groups")  # never silently outcome-only (M6 R3)
        live = credited_records(live, keys, turn_weight=turn_weight, frozen_actors=frozen_actors)
    _, _, _, TrajectoryGroup = _rllm()
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for ro, g in zip(live, keys, strict=True):
        sa = seq_adv(ro) if seq_adv else outcome.get(ro.rollout_id, 0.0)
        for tr in _trajectories(ro, backend=backend, seq_adv=sa, turn_weight=turn_weight, frozen_actors=frozen_actors):
            groups[(g, tr.name)].append(tr)
    out = []
    for (g, role), trs in groups.items():
        vs = [st.weight_version for tr in trs for st in tr.steps if st.weight_version is not None]
        out.append(TrajectoryGroup(trajectories=trs, group_id=f"{g}:{role}", weight_version=min(vs) if vs else 0))
    return out
