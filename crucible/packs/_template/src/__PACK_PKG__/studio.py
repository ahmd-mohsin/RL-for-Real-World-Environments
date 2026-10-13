"""Studio hooks of the __PACK_NAME__ pack (declared in pack.yaml)."""

from __future__ import annotations

import random
from pathlib import Path

from crucible.actors.policy import ScriptedPolicy
from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.runtime.harness.native import PolicyRef
from crucible.verify.runtime.plan import Check, Context, Gate, Plan, Verdict
from crucible.verify.runtime.trusted import Trusted
from crucible.world.world import World

WORDS = ["hello", "bonjour", "hola", "ciao", "hallo", "namaste", "salaam", "ola"]


class GreetingTask:
    """Write greeting.txt containing the seeded greeting. The answer is in the instruction; the grader reads the file."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.word = random.Random(f"__PACK_NAME__/greeting/{seed}").choice(WORDS)
        self.id = f"__PACK_NAME__/greeting/s{seed}@1"
        self.world_id = "__PACK_NAME__/workspace"
        self.instruction = f"Create greeting.txt in the workspace containing exactly the word {self.word!r} followed by a newline."

    async def open(self, workdir: Path) -> World:
        root = (Path(workdir) / "ws").resolve()
        spec = WorldSpec.model_validate({"world": self.world_id, "apps": [{"id": "ws", "template": "common/workspace",
                                         "state": {"adapter": "fs", "params": {"root": str(root)}}, "params": {"shell": False}}],
                                         "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}]})
        w = World(spec, instructions={"agent": self.instruction}, store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w")
        w._pack_ws = root
        await w.start()
        await w.reset()
        return w

    async def grade(self, world: World, workdir: Path):
        path = world._pack_ws / "greeting.txt"

        def content(_ctx) -> Verdict:
            got = path.read_text() if path.exists() else None
            return Verdict(got == self.word + "\n", {"got": got})

        return await Plan(self.id, [Gate("outcome")], [Check("greeting", "outcome", content)]).arun(Context(Trusted(Path(workdir))))

    def gold(self) -> list:
        return [ToolCall(tool_call_id="g1", function_name="ws.write_file", arguments={"path": "greeting.txt", "content": self.word + "\n"})]


def tasks() -> list:
    return [(t, t.gold()) for t in (GreetingTask(0), GreetingTask(1))]


def generate(seed: int, item_id: str | None = None) -> GreetingTask:
    """A fresh benchmark item. ``crucible.bench.dynamic.build`` passes the published opaque ``item_id``: a generated id
    must never spell its seed (M13 audit V3), so without one it is a one-way hash of the seed."""
    import hashlib

    t = GreetingTask(seed)
    t.id = item_id or f"__PACK_NAME__/greeting/g{hashlib.sha256(f'__PACK_NAME__/greeting|{seed}'.encode()).hexdigest()[:16]}@1"
    return t


def policies() -> list[PolicyRef]:
    return [PolicyRef(id="scripted:gold", make=lambda a: {"agent": ScriptedPolicy("agent", a.task.gold())}),
            PolicyRef(id="scripted:noop", make=lambda a: {"agent": ScriptedPolicy("agent", [])})]


def fixtures() -> list[dict]:
    """Designed cases with their exact expected grade: the grader must reject near misses, not only the empty episode."""
    t = GreetingTask(0)
    write = lambda path, content: [ToolCall(tool_call_id="f1", function_name="ws.write_file", arguments={"path": path, "content": content})]  # noqa: E731
    return [{"name": "gold", "task": t, "actions": t.gold(), "reward": 1.0},
            {"name": "noop", "task": t, "actions": [], "reward": 0.0, "fails_gate": "outcome"},
            {"name": "wrong_word", "task": t, "actions": write("greeting.txt", "goodbye\n"), "reward": 0.0, "fails_gate": "outcome"},
            {"name": "wrong_file", "task": t, "actions": write("greetings.txt", t.word + "\n"), "reward": 0.0, "fails_gate": "outcome"},
            {"name": "no_newline", "task": t, "actions": write("greeting.txt", t.word), "reward": 0.0, "fails_gate": "outcome"}]
