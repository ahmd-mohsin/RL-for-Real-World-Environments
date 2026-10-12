"""Per-seat training records for turn-based multi-agent Worlds (M12; games of M1 item 7: several policy seats, such as two
LLM players, under ``WorldSpec.turns``).

A game episode produces one trace; training needs **one ``crucible.rollout/v1`` record per trained seat**:

- the seat's own model calls only (``Turn.actor`` == seat, ``RolloutRecord.actor`` set). An opponent's tokens are never in
  another seat's record, so they can never be trained as that seat's output;
- the seat's own outcome as ``reward``: the game App's per-seat rewards (win / loss / draw, or a per-seat score) from
  ``TurnManager.rewards``; a seat without an outcome is masked (``reward=None``), never scored 0. Every seat's outcome is
  kept in ``reward_components["seat/<id>"]`` for audit;
- its own group: ``<group>#seat=<id>``. In a zero-sum game the mean over *both* seats is ~0 whatever the policy does, and
  centering both seats against it turns a structural first-mover edge into permanent credit for one seat; the group
  statistic must compare a seat with itself across the episodes of a task (SPIRAL's role-conditioned baseline, as
  prime-rl ``orchestrator/algo/rae.py`` keys its EMA per (env, agent));
- opponents' moves reach the seat as **observations**: ``MoveLog`` (a ``run_episode`` observer) records every executed
  game move and ``SeatView`` (a policy wrapper) delivers the moves the other seats made since the seat last acted as
  inbox messages (``[from <seat>] played <tool> <args>``) on its next observation. ``LLMToolPolicy`` renders them as a user
  message, so with exact TITO they land in the seat's next prompt suffix (mask 0, never trained). A move buffered in a
  simultaneous round is released only when the round resolves, so no seat learns another's move first. Only the action
  is revealed, never the mover's tool result (it may hold private state: a dealt card).

Frozen opponents keep ``TrainerConfig.frozen_actors`` semantics: a frozen seat plays (through the engine, or through a
fixed ``opponents`` policy such as a scripted player or a pinned checkpoint) and emits no record. The seat-aware rollout
is ``train.world_env.world_rollout(..., seats=SeatConfig(...))`` (a list of records); ``single_seat`` adapts it to the M12
runtime's one-record-per-rollout contract when exactly one seat trains.

Patterns: verifiers v1 ``environments/kuhn_poker`` (each seat's interaction is one real rollout, agent-stamped, zero-sum
payoff per seat, the betting so far told to the acting seat), prime-rl ``iter_trainable_traces`` (untrainable agents
emit nothing) and RAE (per-seat baselines). No code copied.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.actors.seat_view import MoveLog, SeatView, describe_move  # noqa: F401 -- re-exported: seat views live with the actors
from crucible.core.schema.rollout import RolloutRecord, Segment

SEAT_SEP = "#seat="


def seat_group(group: str | None, seat: str) -> str | None:
    """The GRPO group of a seat's records: the episode's group (task + seed family) and the seat."""
    return None if group is None else f"{group}{SEAT_SEP}{seat}"


@dataclass
class SeatConfig:
    """Which seats of a game train and how the others are played.

    ``trained``: the seats that emit records (default: every seat not frozen and not in ``opponents``). ``frozen``: seats
    that play through the engine but carry no loss (``TrainerConfig.frozen_actors``). ``opponents``: ``{seat: make(seat)
    -> Policy}`` for seats played by a fixed policy (never trained). ``make_policy(seat, llm, temperature)`` replaces the
    default ``LLMToolPolicy`` for engine-driven seats. ``reward(world, report, seat) -> float | None`` replaces the game
    App's per-seat rewards. ``reveal``: opponents' moves on the next observation; ``show_legal``: the seat's legal
    actions too (``Observation.legal``, which ``LLMToolPolicy`` does not render by itself)."""

    trained: tuple[str, ...] | None = None
    frozen: tuple[str, ...] = ()
    opponents: dict[str, Callable[[str], Any]] = field(default_factory=dict)
    make_policy: Callable[[str, Any, float], Any] | None = None
    reward: Callable[[Any, Any, str], float | None] | None = None
    reveal: bool = True
    show_legal: bool = False

    def trained_seats(self, seats: Sequence[str]) -> list[str]:
        unknown = sorted(set(self.trained or ()) - set(seats))
        if unknown:
            raise ValueError(f"trained seats {unknown} are not seats of this game ({list(seats)})")
        fixed = set(self.frozen) | set(self.opponents)
        clash = sorted(set(self.trained or ()) & fixed)
        if clash:
            raise ValueError(f"seats {clash} are both trained and frozen / fixed opponents")
        return [s for s in seats if (s in self.trained if self.trained is not None else s not in fixed)]


def game_rewards(world: Any) -> dict[str, float | None]:
    """Per-seat outcomes from the game App (``TurnManager.rewards``): a seat the App scored gets its score; at a terminal
    state an unscored seat drew (0.0); before one, an unscored seat has no outcome (None: masked)."""
    turns = getattr(world, "turns", None)
    if turns is None:
        return {}
    return {s: (float(turns.rewards[s]) if s in turns.rewards else (0.0 if turns.terminal else None)) for s in turns.seats}


def split_seats(ro: RolloutRecord, *, rewards: Mapping[str, float | None], trained: Sequence[str], group: str | None = None,
                segments: Mapping[str, Sequence[Segment]] | None = None) -> list[RolloutRecord]:
    """One record per trained seat out of a combined game record (every seat's calls): that seat's turns only, its own
    reward (None = masked), ``actor`` set, rollout id ``<rid>#seat=<id>``, group ``<group>#seat=<id>``. ``segments``:
    each seat's compaction segments (default: the segments whose summary call is one of the seat's calls)."""
    out = []
    comps = {f"seat/{s}": float(r) for s, r in rewards.items() if r is not None}
    for s in trained:
        turns = [t.model_copy(deep=True) for t in ro.turns if t.actor == s]
        ids = {t.call_id for t in turns}
        segs = list(segments[s]) if segments is not None and s in segments else [g for g in ro.segments if g.summary_call in ids]
        r = rewards.get(s)
        masked = ro.mask_sample or r is None
        out.append(ro.model_copy(deep=True, update={
            "rollout_id": f"{ro.rollout_id}{SEAT_SEP}{s}", "sample": f"{ro.sample}{SEAT_SEP}{s}", "turns": turns, "segments": segs, "actor": s,
            "group": seat_group(group if group is not None else ro.group, s), "reward": None if masked else float(r),
            "reward_components": {**ro.reward_components, **comps}, "mask_sample": ro.mask_sample,
            "failure_kind": ro.failure_kind if ro.mask_sample or r is not None else (ro.failure_kind or "crucible:no_seat_outcome")}))
    return out


def single_seat(rollout_fn: Callable) -> Callable:
    """The M12 runtime takes one record per rollout: a seat-aware rollout with exactly one trained seat (a learner against
    frozen or fixed opponents) yields that seat's record."""

    async def rollout(*args: Any, **kw: Any) -> RolloutRecord | None:
        out = await rollout_fn(*args, **kw)
        if out is None:
            return None
        if len(out) != 1:
            raise ValueError(f"single_seat needs exactly one trained seat (got {len(out)} records)")
        return out[0]

    return rollout
