"""M1 item 7: native turn order (``WorldSpec.turns``, ``world/turns.py``; OpenSpiel / OpenEnv game semantics).

Three games:

- sequential seats over a plain App;
- Nim with the rules in the App (``app`` mode): legal actions, a seeded die (chance) for the first mover, terminal
  rewards, snapshot and restore mid-game;
- rock-paper-scissors (``simultaneous``): no seat sees the other's move first, and the App resolves the round.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import ClassVar

import pytest

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
from crucible.core.registry import get_registry
from crucible.core.schema.actions import Done, Message, ToolCall, ToolResult, ToolSpec
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.apps.base import App
from crucible.world.world import World


class _DocApp(App):
    PATH_PARAMS = ("path",)

    def _load(self) -> dict:
        p = Path(self.state_params["path"])
        return json.loads(p.read_text()) if p.exists() else {}

    def _save(self, d: dict) -> None:
        p = Path(self.state_params["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, sort_keys=True))


class Nim(_DocApp):
    template = "test/nim"

    async def provision(self) -> None:
        first = random.Random(self.ctx.seed).choice(["alice", "bob"])  # chance node: who starts, from the App's seed
        self._save({"pile": int(self.spec.params.get("pile", 5)), "to_move": first, "winner": None})

    async def list_tools(self) -> list[ToolSpec]:
        return [ToolSpec(app=self.id, name="take", description="Take 1-3 stones.", input_schema={"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]})]

    async def call(self, name, args, *, actor, call_id):
        d = self._load()
        d["pile"] -= int(args["n"])
        if d["pile"] <= 0:
            d["winner"], d["to_move"] = actor, None  # took the last stone
        else:
            d["to_move"] = "bob" if actor == "alice" else "alice"
        self._save(d)
        return ToolResult(tool_call_id=call_id, function_name=f"{self.id}.{name}", ok=True, content=f"pile={max(d['pile'], 0)}")

    async def turn_state(self) -> dict:
        d = self._load()
        if d["winner"]:
            loser = "bob" if d["winner"] == "alice" else "alice"
            return {"to_move": [], "terminal": True, "rewards": {d["winner"]: 1.0, loser: -1.0}}
        return {"to_move": [d["to_move"]], "legal": {d["to_move"]: [{"function_name": "nim.take", "arguments": {"n": k}} for k in range(1, min(3, d["pile"]) + 1)]}}


class Rps(_DocApp):
    template = "test/rps"
    BEATS: ClassVar[dict[str, str]] = {"rock": "scissors", "scissors": "paper", "paper": "rock"}

    async def provision(self) -> None:
        self._save({"score": {"p1": 0, "p2": 0}, "rounds": []})

    async def list_tools(self) -> list[ToolSpec]:
        return [ToolSpec(app=self.id, name="throw", description="rock, paper or scissors", input_schema={"type": "object", "properties": {"hand": {"type": "string"}}, "required": ["hand"]})]

    async def resolve_round(self, actions: dict) -> dict:
        d = self._load()
        h = {s: a["arguments"]["hand"] for s, a in actions.items()}
        win = "p1" if self.BEATS[h["p1"]] == h["p2"] else "p2" if self.BEATS[h["p2"]] == h["p1"] else None
        if win:
            d["score"][win] += 1
        d["rounds"].append(h)
        self._save(d)
        return {s: ToolResult(tool_call_id=a["tool_call_id"], function_name="rps.throw", ok=True, content=f"you {h[s]}, they {h['p2' if s == 'p1' else 'p1']}")
                for s, a in actions.items()}

    async def turn_state(self) -> dict:
        d = self._load()
        return {"to_move": ["p1", "p2"], "rewards": {k: float(v) for k, v in d["score"].items()}, "terminal": len(d["rounds"]) >= 2}


get_registry().register("app_template", "test/nim", Nim, replace=True)
get_registry().register("app_template", "test/rps", Rps, replace=True)


def _world(tmp: Path, spec: dict) -> World:
    return World(WorldSpec.model_validate(spec), store=Store(tmp / "s"), workdir=tmp / "w")


def _tc(i, name, **args) -> ToolCall:
    return ToolCall(tool_call_id=f"c{i}", function_name=name, arguments=args)


async def test_sequential_seats(tmp_path):
    w = _world(tmp_path, {"world": "t/seq", "turns": {"mode": "sequential", "seats": ["a", "b", "c"]},
                          "apps": [{"id": "api", "template": "common/json_api", "state": {"adapter": "json_doc", "params": {"path": "api.json"}},
                                    "params": {"collections": ["log"]}}],
                          "actors": [{"id": x, "kind": "policy", "tools": ["api.*"]} for x in ("a", "b", "c")]})
    await w.start()
    try:
        obs = await w.reset()
        assert obs["a"].to_move == ["a"] and obs["b"].to_move == ["a"]
        r = await w.step("b", _tc(1, "api.list_resources", collection="log"))
        assert r.observation.tool_results[0].error == "not_your_turn"  # refused, not executed
        assert (await w.step("a", _tc(2, "api.list_resources", collection="log"))).observation.to_move == ["b"]
        await w.step("b", Message(to="c", text="your go"))  # a message is a move too
        assert w.turns.to_move == ["c"]
        await w.step("c", Done(reason="out"))  # Done is always accepted: c leaves the rotation
        assert w.turns.to_move == ["a"]
        await w.step("a", _tc(3, "api.list_resources", collection="log"))
        assert w.turns.to_move == ["b"]  # c is skipped from now on
    finally:
        await w.close()


NIM = {"world": "t/nim", "seed": 7, "turns": {"mode": "app", "app": "nim"},
       "apps": [{"id": "nim", "template": "test/nim", "state": {"adapter": "json_doc", "params": {"path": "nim.json"}}, "params": {"pile": 5}}],
       "actors": [{"id": "alice", "kind": "policy", "tools": ["nim.*"]}, {"id": "bob", "kind": "policy", "tools": ["nim.*"]}]}


async def test_app_rules_legal_actions_chance_and_rewards(tmp_path):
    w = _world(tmp_path, NIM)
    await w.start()
    try:
        obs = await w.reset()
        first = w.turns.to_move[0]
        other = "bob" if first == "alice" else "alice"
        assert obs[first].legal == [{"function_name": "nim.take", "arguments": {"n": k}} for k in (1, 2, 3)] and obs[other].legal is None
        assert (await w.step(first, _tc(1, "nim.take", n=4))).observation.tool_results[0].error == "illegal_action"
        assert (await w.step(other, _tc(2, "nim.take", n=1))).observation.tool_results[0].error == "not_your_turn"
        await w.step(first, _tc(3, "nim.take", n=2))  # pile 3
        mid = await w.snapshot()
        await w.step(other, _tc(4, "nim.take", n=3))  # pile 0: other wins
        assert w.turns.terminal and w.turns.rewards == {other: 1.0, first: -1.0} and w.done == {"alice", "bob"}
        await w.restore(mid)
        assert w.turns.to_move == [other] and not w.turns.terminal and w.turns.legal[other][-1]["arguments"] == {"n": 3}
    finally:
        await w.close()
    w2 = _world(tmp_path / "again", NIM)  # the chance node is the App's seed: the same spec deals the same first mover
    await w2.start()
    try:
        await w2.reset()
        assert w2.turns.to_move == [first]
    finally:
        await w2.close()


async def test_full_game_through_run_episode(tmp_path):
    w = _world(tmp_path, NIM)
    await w.start()
    try:
        await w.reset()
        first = w.turns.to_move[0]
        other = "bob" if first == "alice" else "alice"
        pols = {first: ScriptedPolicy(first, [_tc(1, "nim.take", n=1), _tc(3, "nim.take", n=1)]),  # 5 -> 4, then 1 -> 0 (wins)
                other: ScriptedPolicy(other, [_tc(2, "nim.take", n=3)])}  # 4 -> 1
        ep = await run_episode(w, pols, reset=False, limits=EpisodeLimits(max_steps=10))
        assert ep.done and w.turns.rewards == {first: 1.0, other: -1.0} and ep.steps == 3
    finally:
        await w.close()


RPS = {"world": "t/rps", "turns": {"mode": "simultaneous", "app": "rps"},
       "apps": [{"id": "rps", "template": "test/rps", "state": {"adapter": "json_doc", "params": {"path": "rps.json"}}}],
       "actors": [{"id": "p1", "kind": "policy", "tools": ["rps.*"]}, {"id": "p2", "kind": "policy", "tools": ["rps.*"]}]}


async def test_simultaneous_moves(tmp_path):
    w = _world(tmp_path, RPS)
    await w.start()
    try:
        await w.reset()
        r1 = await w.step("p1", _tc(1, "rps.throw", hand="rock"))
        assert r1.observation.tool_results == [] and r1.info.get("waiting")  # nothing resolves until both have moved
        assert (await w.step("p1", _tc(2, "rps.throw", hand="paper"))).observation.tool_results[0].error == "not_your_turn"
        r2 = await w.step("p2", _tc(3, "rps.throw", hand="scissors"))
        assert r2.observation.tool_results[0].content == "you scissors, they rock"
        assert w.turns.rewards == {"p1": 1.0, "p2": 0.0}
        deferred = w.turns.take_deferred("p1")  # p1's result waits for its next observation
        assert [d.content for d in deferred] == ["you rock, they scissors"]
    finally:
        await w.close()


async def test_simultaneous_episode_and_validation(tmp_path):
    w = _world(tmp_path, RPS)
    await w.start()
    try:
        await w.reset()
        pols = {"p1": ScriptedPolicy("p1", [_tc(1, "rps.throw", hand="rock"), _tc(2, "rps.throw", hand="rock")]),
                "p2": ScriptedPolicy("p2", [_tc(3, "rps.throw", hand="paper"), _tc(4, "rps.throw", hand="rock")])}
        ep = await run_episode(w, pols, reset=False, limits=EpisodeLimits(max_steps=10))
        assert w.turns.terminal and w.turns.rewards == {"p1": 0.0, "p2": 1.0} and ep.done
        p1_seen = [r.content for o in ep.observations["p1"] for r in o.tool_results]
        assert "you rock, they paper" in p1_seen  # the deferred result reached p1 on its next turn
    finally:
        await w.close()
    with pytest.raises(ValueError, match="seat"):
        WorldSpec.model_validate({**RPS, "turns": {"mode": "simultaneous", "seats": ["p1", "ghost"]}})
    with pytest.raises(ValueError, match="app"):
        WorldSpec.model_validate({**NIM, "turns": {"mode": "app"}})


async def test_seat_rewards_reach_the_run_record(tmp_path):
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.run import Runner, RunSpec
    from crucible.verify.runtime.plan import GradeReport

    class NimTask:
        id, world_id = "t/nim/5@1", "t/nim"

        def identity_config(self) -> dict:  # M10 lab audit R-4: it holds state once opened (``first``), so it declares its (empty) configuration
            return {}

        async def open(self, workdir):
            w = World(WorldSpec.model_validate(NIM), store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w")
            await w.start()
            await w.reset()
            self.first = w.turns.to_move[0]
            return w

        async def grade(self, world, workdir):
            r = world.verifier_view().seat_rewards()
            return GradeReport(self.id, "full", [], {}, max(0.0, r.get("alice", 0.0)), 1.0, {"status": "ok"}, True, 0.0)

    task = NimTask()

    def make(attempt):
        other = "bob" if task.first == "alice" else "alice"
        return {task.first: ScriptedPolicy(task.first, [_tc(1, "nim.take", n=1), _tc(3, "nim.take", n=1)]),
                other: ScriptedPolicy(other, [_tc(2, "nim.take", n=3)])}

    await Runner(tmp_path / "runs").run(RunSpec(run_id="nim", tasks=[task], policies=[PolicyRef("scripted:nim", make, config={})],
                                                harnesses=[NativeHarness(limits=EpisodeLimits(max_steps=10))], k=1))
    rec = json.loads(next((tmp_path / "runs" / "nim" / "samples").glob("*/a1/record.json")).read_text())
    other = "bob" if task.first == "alice" else "alice"
    assert rec["meta"]["seat_rewards"] == {task.first: 1.0, other: -1.0} and rec["failure"] is None
