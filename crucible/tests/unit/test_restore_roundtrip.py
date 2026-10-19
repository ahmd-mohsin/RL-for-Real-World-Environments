"""``runtime.restore_roundtrip`` (M1 audit 2026-10-14 R-16): an App whose state a snapshot misses passes every check from
reset but diverges once a mid-episode snapshot is restored into a fresh World, the way a resume does."""

from __future__ import annotations

import json
from pathlib import Path

from crucible.core.registry import get_registry
from crucible.core.schema.actions import ToolCall, ToolResult, ToolSpec
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.apps.base import App
from crucible.world.audit import audit_world
from crucible.world.world import World


class CounterApp(App):
    """Hands out ids from an in-memory counter and records them in its document (the audit's probe App)."""

    template = "test/r16-counter"

    @property
    def path(self) -> Path:
        return Path(self.ctx.resolve(self.state_params["path"]))

    async def provision(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"items": []}))
        self.seq = 0

    async def after_restore(self):
        self.seq = 0  # the bug: memory reset instead of re-derived from the restored document

    async def list_tools(self):
        return [ToolSpec(app=self.id, name="create", description="create an item", input_schema={"type": "object", "properties": {}})]

    async def call(self, name, args, *, actor, call_id):
        doc = json.loads(self.path.read_text())
        self.seq += 1
        doc["items"].append(self.seq)
        self.path.write_text(json.dumps(doc))
        return ToolResult(tool_call_id=call_id, function_name=f"{self.id}.create", ok=True, content=json.dumps({"id": self.seq}))


class FixedCounterApp(CounterApp):
    template = "test/r16-counter-fixed"

    async def after_restore(self):
        self.seq = len(json.loads(self.path.read_text())["items"])  # re-derived from the restored state


for cls in (CounterApp, FixedCounterApp):
    get_registry().register("app_template", cls.template, cls, replace=True)

GOLD = [ToolCall(tool_call_id=f"g{i}", function_name="c.create", arguments={}) for i in range(3)]


async def audit(tmp_path: Path, template: str, gold=GOLD):
    spec = WorldSpec.model_validate({"world": "test/r16", "apps": [{"id": "c", "template": template, "state": {"adapter": "json_doc", "params": {"path": "c/state.json"}}}],
                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["c.*"]}]})
    w = World(spec, instructions={"agent": "create items"}, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        rep = await audit_world(w, gold=gold)
    finally:
        await w.close()
    return {c.id: c for c in rep.checks}


async def test_state_a_snapshot_misses_fails_the_round_trip(tmp_path: Path):
    checks = await audit(tmp_path, CounterApp.template)
    assert checks["episode_determinism"].status == "pass" and checks["runtime.reset_stable"].status == "pass"  # invisible from reset
    rt = checks["runtime.restore_roundtrip"]
    assert rt.status == "fail" and rt.detail["apps_differing"] == ["c"] and rt.detail["restored_at_step"] == 1
    assert rt.detail["first_result_differing"] == {"step": 1, "original": '{"id": 2}', "restored": '{"id": 1}'}


async def test_state_re_derived_on_restore_passes(tmp_path: Path):
    assert (await audit(tmp_path, FixedCounterApp.template))["runtime.restore_roundtrip"].status == "pass"


async def test_a_one_step_gold_cannot_be_split(tmp_path: Path):
    rt = (await audit(tmp_path, FixedCounterApp.template, gold=GOLD[:1]))["runtime.restore_roundtrip"]
    assert rt.status == "skip" and "fewer than 2 steps" in rt.detail["reason"]
