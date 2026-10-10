"""World capabilities the diverse environment catalogue needs (research m11-m12/05 §5.2): per-tool call budgets,
a simulation-owned clock, per-App seeds, image tool outputs, and the in-process step overhead budget."""

from __future__ import annotations

import time

import pytest

from crucible.core.registry import get_registry
from crucible.core.schema.actions import ImageRef, Observation, ToolCall, ToolResult, ToolSpec
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.apps.base import App
from crucible.world.world import World

PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000")  # a PNG header is enough for the store


class Sim(App):
    """A tiny simulation App: ``tick(hours)`` advances the World's sim clock; ``shot`` returns an image; ``probe``
    reports this App's seed."""

    template = "test/sim"
    PATH_PARAMS = ("path",)

    async def list_tools(self):
        obj = {"type": "object", "properties": {}}
        return [ToolSpec(app=self.id, name="tick", description="advance", input_schema={"type": "object", "properties": {"hours": {"type": "number"}}}),
                ToolSpec(app=self.id, name="shot", description="render", input_schema=obj),
                ToolSpec(app=self.id, name="probe", description="seed", input_schema=obj)]

    async def tool_tick(self, hours: float = 1.0):
        self.ctx.clock.advance_sim_seconds(self._t + hours * 3600)
        self._t += hours * 3600
        return f"sim t={self._t / 3600:.0f}h"

    async def tool_shot(self):
        return ToolResult(tool_call_id="", function_name=f"{self.id}.shot", ok=True, content="frame 0", images=[self.image(PNG)])

    async def tool_probe(self):
        return str(self.ctx.seed)

    async def start(self):
        self._t = 0.0

    async def call(self, name, args, *, actor, call_id):
        r = await super().call(name, args, actor=actor, call_id=call_id)
        return r.model_copy(update={"tool_call_id": call_id})


get_registry().register("app_template", "test/sim", Sim, replace=True)


def _spec(tmp_path, **kw) -> WorldSpec:
    apps = [{"id": a, "template": "test/sim", "state": {"adapter": "json_doc", "params": {"path": str(tmp_path / f"{a}.json")}}} for a in kw.pop("apps", ["sim"])]
    return WorldSpec.model_validate({"world": "test/sim", "apps": apps, "actors": [{"id": "agent", "kind": "policy", "tools": ["*"]}], **kw})


async def _world(tmp_path, **kw) -> World:
    w = World(_spec(tmp_path, **kw), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    await w.reset()
    return w


def tc(name: str, i: int = 0, **args) -> ToolCall:
    return ToolCall(tool_call_id=f"c{i}", function_name=name, arguments=args)


async def test_tool_quota_is_a_policy_fault_and_survives_snapshot_restore(tmp_path):
    w = await _world(tmp_path, limits={"tool_quota": {"sim.probe": 2}})
    try:
        assert (await w.step("agent", tc("sim.probe", 0))).observation.tool_results[0].ok
        ref = await w.snapshot()
        assert ref.tool_counts == {"sim.probe": 1}
        assert (await w.step("agent", tc("sim.probe", 1))).observation.tool_results[0].ok
        r = (await w.step("agent", tc("sim.probe", 2))).observation.tool_results[0]
        assert not r.ok and r.error == "quota_exhausted" and r.fault == "policy" and "budget 2" in r.content
        assert (await w.step("agent", tc("sim.tick", 3, hours=1))).observation.tool_results[0].ok  # other tools unaffected
        await w.restore(ref)
        assert w.tool_counts == {"sim.probe": 1} and (await w.step("agent", tc("sim.probe", 4))).observation.tool_results[0].ok
        await w.reset()
        assert w.tool_counts == {}
    finally:
        await w.close()


async def test_no_quota_keeps_spec_hash_and_world_fingerprint_unchanged(tmp_path):
    from crucible_pack_toy import helpdesk_spec

    assert helpdesk_spec().content_hash() == "a58d2f478a60db1c31a960847377377b778b9d3b608d90cf614bdb16e8087804"  # hash before tool_quota existed
    w = await _world(tmp_path)
    try:
        ref = await w.snapshot()
        assert ref.tool_counts == {}
        spent = ref.model_copy(update={"tool_counts": {"sim.probe": 1}})
        assert w._fingerprint(spent) != w._fingerprint(ref)  # a used budget is part of the state
    finally:
        await w.close()


async def test_sim_clock_moves_only_with_the_simulation_and_fires_events(tmp_path):
    clock = {"mode": "sim", "t0": "2026-01-05T00:00:00Z"}
    ev = [{"id": "storm", "when": {"after": "2h"}, "actor": "agent", "action": {"type": "nl", "text": "storm warning", "to": "agent"}}]
    w = await _world(tmp_path, clock=clock, events=ev)
    try:
        r = await w.step("agent", tc("sim.probe", 0))
        assert r.observation.time == "2026-01-05T00:00:00Z" and not r.observation.inbox  # a step alone does not move sim time
        r = await w.step("agent", tc("sim.tick", 1, hours=1))
        assert r.observation.time == "2026-01-05T01:00:00Z" and not r.observation.inbox
        ref = await w.snapshot()
        r = await w.step("agent", tc("sim.tick", 2, hours=2))
        assert r.observation.time == "2026-01-05T03:00:00Z" and [m.text for m in r.observation.inbox] == ["storm warning"]
        await w.restore(ref)
        assert w.clock.now_iso() == "2026-01-05T01:00:00Z"
        with pytest.raises(ValueError):
            w.clock.advance_sim_seconds(0)  # the simulation never moves time backwards
    finally:
        await w.close()


async def test_per_app_seeds_are_derived_stable_and_distinct(tmp_path):
    from crucible.core.ids import derive_seed

    w = await _world(tmp_path, apps=["a", "b"], seed=7)
    try:
        sa = (await w.step("agent", tc("a.probe", 0))).observation.tool_results[0].content
        sb = (await w.step("agent", tc("b.probe", 1))).observation.tool_results[0].content
        assert int(sa) == derive_seed(7, "a") and int(sb) == derive_seed(7, "b") and sa != sb
        assert derive_seed(8, "a") != derive_seed(7, "a") and 0 <= derive_seed(7, "a") < 2**63
    finally:
        await w.close()


async def test_image_outputs_store_refs_in_the_trace_and_reach_vision_policies_only(tmp_path):
    from crucible.actors.policy import LLMToolPolicy

    w = await _world(tmp_path)
    try:
        r = (await w.step("agent", tc("sim.shot", 0))).observation.tool_results[0]
        assert r.images and r.images[0].data_b64 and w.store.get_bytes(r.images[0].ref) == PNG
        ev = next(e for e in w.log if e.type == "tool_result")
        assert ev.data["images"] == [{"mime": "image/png", "ref": r.images[0].ref}] and "data_b64" not in str(ev.data)
        assert "data_b64" not in r.model_dump_json()  # never serialized
    finally:
        await w.close()

    class _LLM:
        model = "m"

    obs = Observation(actor="agent", step=1, tool_results=[ToolResult(tool_call_id="c0", function_name="sim.shot", ok=True, content="frame 0",
                                                                      images=[ImageRef(ref="k", data_b64="AAAA")])])
    text = LLMToolPolicy("agent", _LLM())
    text._ingest(obs)
    assert text.messages[-1]["role"] == "tool" and "omitted" in text.messages[-1]["content"]
    vis = LLMToolPolicy("agent", _LLM(), vision=True)
    vis._ingest(obs)
    tool, user = vis.messages[-2:]
    assert tool["role"] == "tool" and user["role"] == "user"  # images follow the tool batch as a user turn
    assert user["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    from crucible.actors.policy import IMAGE_TOKENS, approx_tokens

    assert approx_tokens([user]) < IMAGE_TOKENS + 50  # an image counts a fixed budget, not its base64 length


async def test_in_process_step_overhead_budget(tmp_path):
    """Games, sims and math need ms steps: World.step on an in-process App stays under 5 ms on average."""
    w = await _world(tmp_path)
    try:
        for i in range(20):
            await w.step("agent", tc("sim.probe", i))
        n = 200
        t0 = time.perf_counter()
        for i in range(n):
            await w.step("agent", tc("sim.probe", 100 + i))
        per = (time.perf_counter() - t0) / n
        assert per < 0.005, f"{per * 1e3:.2f} ms per step"
    finally:
        await w.close()
