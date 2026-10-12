"""M12: per-seat training records for turn-based games (``train/seats.py``, ``world_rollout(..., seats=...)``).

Two LLM seats play Nim (rules in the App, a seeded first mover) and rock-paper-scissors (simultaneous rounds) through
the byte codec and a ``ScriptedEngine`` whose script plays from what is in the prompt only: it knows the pile from its own
last tool result minus the opponents' moves it was told about, so a missing observation would make it play wrong. One
record per trained seat: that seat's turns only, its own outcome (zero-sum in Nim), its own group, opponents' moves in
its prompts as observations (mask 0) and never as trained tokens, exact TITO per seat.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from crucible.actors.policy import ScriptedPolicy
from crucible.core.schema.actions import ToolCall, openai_name
from crucible.core.schema.rollout import RolloutRecord, Turn, full_prompts
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.train.bridges.common import flatten
from crucible.train.chat import ByteChatCodec
from crucible.train.engine import GenRequest, ScriptedEngine
from crucible.train.objectives import group_advantages
from crucible.train.seats import SeatConfig, single_seat, split_seats
from crucible.train.world_env import world_rollout
from crucible.verify.runtime.plan import GradeReport
from crucible.world.world import World
from tests.unit.test_turns import NIM, RPS  # registers the test/nim and test/rps App templates

CODEC = ByteChatCodec()


class GameTask:
    def __init__(self, spec: dict, id: str):
        self.spec, self.id, self.world_id = spec, id, spec["world"]
        self.first: str | None = None

    async def open(self, workdir):
        seats = [a["id"] for a in self.spec["actors"]]
        w = World(WorldSpec.model_validate(self.spec), instructions={s: f"You are {s}. Win the game." for s in seats},
                  store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w")
        await w.start()
        await w.reset()
        self.first = w.turns.to_move[0] if w.turns.mode != "simultaneous" else None
        return w

    async def grade(self, world, workdir):
        return GradeReport(self.id, "full", [], {}, 0.0, 1.0, {"status": "ok"}, True, 0.0)


def _text(ids) -> str:
    return bytes(t for t in ids if t < 256).decode(errors="replace")


def _call(name: str, **args) -> list[int]:
    return list(f"<tool_call>{json.dumps({'name': openai_name(name), 'arguments': args})}</tool_call>".encode()) + [CODEC.EOS]


def nim_script(rid, k, prompt):
    """Optimal Nim from the prompt alone: own last ``pile=N`` minus every opponent move reported after it."""
    text = _text(prompt)
    me = re.search(r"You are (\w+)\.", text).group(1)
    last = [m for m in re.finditer(r"pile=(\d+)", text)]
    pile, since = (int(last[-1].group(1)), last[-1].end()) if last else (int(re.search(r"PILE (\d+)", text).group(1)), 0)
    for m in re.finditer(r"\[from (\w+)\] played nim\.take \{\"n\": (\d+)\}", text[since:]):
        if m.group(1) != me:
            pile -= int(m.group(2))
    n = max(1, min(3, pile, pile % 4 or 1))
    return _call("nim.take", n=n)


def _gen(script):
    eng = ScriptedEngine(script, stop_ids=[CODEC.EOS])
    calls = [0]

    async def gen(prompt, n, temp):
        calls[0] += 1
        return await eng.generate(GenRequest(rid="game", call_id=f"c{calls[0]}", prompt_ids=prompt, max_tokens=n, temperature=temp, stop_ids=[CODEC.EOS]))

    return gen


def _nim(pile: int = 9) -> dict:
    spec = copy.deepcopy(NIM)
    spec["apps"][0]["params"]["pile"] = pile
    return spec


def _nim_task(tmp_path, pile=9):
    task = GameTask(_nim(pile), "t/nim@1")
    orig = task.open

    async def open_(workdir):  # the starting pile is public: told to both seats in their instruction
        w = await orig(workdir)
        w.instructions = {s: f"{v} PILE {pile}" for s, v in w.instructions.items()}
        return w

    task.open = open_
    return task


async def test_two_llm_seats_get_one_record_each(tmp_path):
    task = _nim_task(tmp_path)
    fn = world_rollout(lambda p: task, CODEC, max_steps=20, max_tokens=256, workroot=tmp_path, seats=SeatConfig())
    recs = await fn("r1", "nim-9", 0, _gen(nim_script), temperature=1.0)
    first = task.first
    second = "bob" if first == "alice" else "alice"
    assert [r.actor for r in recs] == ["alice", "bob"]
    by = {r.actor: r for r in recs}
    # optimal play from 9: the first mover takes 1, 3, 3 and wins; zero-sum outcome, each seat its own
    assert by[first].reward == 1.0 and by[second].reward == -1.0 and sum(r.reward for r in recs) == 0.0
    assert by[first].reward_components["seat/" + second] == -1.0
    assert {r.group for r in recs} == {"nim-9#seat=alice", "nim-9#seat=bob"}
    assert len({r.rollout_id for r in recs}) == 2 and len(by[first].turns) == 3 and len(by[second].turns) == 2
    for r in recs:
        ok, why = r.valid_for_training()
        assert ok, why
        assert all(t.actor == r.actor for t in r.turns)  # that seat's turns only
        prompts = full_prompts(r)
        assert all(t.prompt_is_delta for t in r.turns[1:]) and not any(t.prefix_break for t in r.turns)  # exact TITO per seat
        for a, b in zip(r.turns, r.turns[1:], strict=False):
            assert prompts[b.call_id][: len(prompts[a.call_id]) + len(a.completion_token_ids)] == prompts[a.call_id] + a.completion_token_ids
        other = "bob" if r.actor == "alice" else "alice"
        # the opponent's moves are in this seat's prompts (observations) ...
        seen = "".join(_text(prompts[t.call_id]) for t in r.turns)
        assert f"[from {other}] played nim.take" in seen
        # ... and every trained token (mask 1) is this seat's own completion, nothing else
        (seq,) = flatten(r, 1.0)
        trained = [tok for tok, m in zip(seq.response_ids, seq.response_mask, strict=True) if m]
        assert trained == [tok for t in r.turns for tok in t.completion_token_ids]
        assert seq.prompt_ids == prompts[r.turns[0].call_id]
        obs = _text([tok for tok, m in zip(seq.response_ids, seq.response_mask, strict=True) if not m])
        assert f"[from {other}] played nim.take" in obs or r.actor == second  # the second mover saw the first move in its first prompt
    assert f"[from {first}] played nim.take" in _text(full_prompts(by[second])[by[second].turns[0].call_id])


async def test_frozen_and_fixed_opponents_emit_no_record(tmp_path):
    task = _nim_task(tmp_path)
    await task.open(tmp_path / "probe")  # the seeded first mover
    first = task.first
    second = "bob" if first == "alice" else "alice"
    fn = world_rollout(lambda p: task, CODEC, max_steps=20, max_tokens=256, workroot=tmp_path, seats=SeatConfig(frozen=(second,)))
    (rec,) = await fn("r1", "nim-9", 0, _gen(nim_script), temperature=1.0)  # the frozen seat played through the engine, untrained
    assert rec.actor == first and all(t.actor == first for t in rec.turns) and rec.reward == 1.0

    i = iter(range(100))

    def take_one(obs):  # a fixed opponent: always takes one stone
        return ToolCall(tool_call_id=f"o{next(i)}", function_name="nim.take", arguments={"n": 1})

    fixed = SeatConfig(opponents={second: lambda s: ScriptedPolicy(s, take_one)})
    rec = await single_seat(world_rollout(lambda p: task, CODEC, max_steps=20, max_tokens=256, workroot=tmp_path, seats=fixed))(
        "r2", "nim-9", 0, _gen(nim_script), temperature=1.0)
    assert isinstance(rec, RolloutRecord) and rec.actor == first and rec.reward == 1.0 and rec.group == "nim-9#seat=" + first
    assert "[from " + second + '] played nim.take {"n": 1}' in "".join(_text(p) for p in full_prompts(rec).values())
    with pytest.raises(ValueError, match="both trained and frozen"):
        SeatConfig(trained=(first,), frozen=(first,)).trained_seats(["alice", "bob"])
    with pytest.raises(ValueError, match="exactly one"):
        await single_seat(world_rollout(lambda p: task, CODEC, max_steps=20, workroot=tmp_path, seats=SeatConfig()))("r3", "nim-9", 0, _gen(nim_script))


async def test_simultaneous_rounds_reveal_moves_only_after_resolution(tmp_path):
    task = GameTask(copy.deepcopy(RPS), "t/rps@1")

    def script(rid, k, prompt):
        return _call("rps.throw", hand="rock" if "You are p1" in _text(prompt) else "paper")

    recs = await world_rollout(lambda p: task, CODEC, max_steps=20, max_tokens=256, workroot=tmp_path, seats=SeatConfig())("r1", "rps", 0, _gen(script))
    by = {r.actor: r for r in recs}
    assert by["p1"].reward == 0.0 and by["p2"].reward == 2.0  # per-seat score (not zero-sum)
    p1, p2 = full_prompts(by["p1"]), full_prompts(by["p2"])
    assert "[from p1]" not in _text(p2[by["p2"].turns[0].call_id])  # p2 moved before p1's buffered throw was revealed
    assert '[from p2] played rps.throw {"hand": "paper"}' in _text(p1[by["p1"].turns[1].call_id])  # revealed after the round
    assert "you rock, they paper" in _text(p1[by["p1"].turns[1].call_id])  # and the App's round result reached p1 too


def test_seat_groups_keep_a_structural_edge_out_of_the_advantage():
    """Alice wins every game from the first seat: a mixed group credits her for the seat (+1 each), seat groups do not."""
    def game(rid: str) -> RolloutRecord:
        turns = [Turn(call_id=f"{s}:c1", actor=s, prompt_token_ids=[1, 2], completion_token_ids=[3], logprobs=[-0.1], policy_version=0) for s in ("alice", "bob")]
        return RolloutRecord(rollout_id=rid, sample=rid, task="nim", harness="h", policy="p", turns=turns, reward=0.0)

    recs = [r for g in ("g1", "g2") for r in split_seats(game(g), rewards={"alice": 1.0, "bob": -1.0}, trained=["alice", "bob"], group="nim")]
    rewards = [r.reward for r in recs]
    assert group_advantages(rewards, ["nim"] * 4) == [1.0, -1.0, 1.0, -1.0]
    assert group_advantages(rewards, [r.group for r in recs]) == [0.0] * 4
    masked = split_seats(game("g3"), rewards={"alice": None, "bob": 0.0}, trained=["alice", "bob"])
    assert masked[0].reward is None and masked[0].failure_kind == "crucible:no_seat_outcome" and not masked[0].valid_for_training()[0]
    assert masked[1].reward == 0.0 and masked[1].group is None


async def test_runtime_trains_every_seat_in_its_own_group(tmp_path):
    """The runtime takes a game rollout's list of seat records: one pool rollout per episode, expanded into per-seat groups
    when the batch is built (``Runtime._train``), so both seats train and a seat's group holds only that seat."""
    from crucible.train.engine import EnginePool
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.weights import InprocPublisher

    seen: list[tuple[list[RolloutRecord], list[str]]] = []

    class Trainer:
        updates = 0

        def step(self, recs, groups):
            seen.append((list(recs), list(groups)))
            self.updates += 1
            return {"train/loss": 0.0}

        def state_dict(self):
            return {}

    tr = Trainer()
    pool = EnginePool([ScriptedEngine(nim_script, stop_ids=[CODEC.EOS])])
    task = _nim_task(tmp_path)
    fn = world_rollout(lambda p: task, CODEC, max_steps=20, max_tokens=256, workroot=tmp_path, seats=SeatConfig())
    rt = Runtime(RuntimeConfig(topology="disaggregated_stream", steps=2, max_turn_tokens=256, warmup_batches=0), SchedulerConfig(n=2, B=2, max_staleness=2),
                 pool, tr, InprocPublisher(tr, pool), None, {"nim": [f"nim-{i}" for i in range(50)]}, rollout_fn=fn, harness="native:world@1")
    rep = await rt.run()
    assert len(rep.steps) == 2 and len(seen) == 2
    for recs, groups in seen:
        assert len(recs) == 8  # 2 groups x 2 episodes x 2 seats
        by_group: dict[str, list[RolloutRecord]] = {}
        for r, g in zip(recs, groups, strict=True):
            by_group.setdefault(g, []).append(r)
            assert g.endswith(f"#seat={r.actor}")
        assert all(len({r.actor for r in rs}) == 1 for rs in by_group.values())  # a seat's group holds that seat only
        assert sum(r.reward for r in recs) == 0.0  # Nim is zero-sum across the seats of each episode
    assert rt._seats == {}  # consumed or released: nothing leaks over a run
