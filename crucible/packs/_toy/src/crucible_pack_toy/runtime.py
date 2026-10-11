"""The helpdesk task as an M10 ``RunTask``, and the toy policies the runtime gate uses.

``HelpdeskTask`` opens a fresh, reset helpdesk World with its responders (record-grounded customers, a rule-based
supervisor) and grades it with the M7 toy plan (refund-policy invariants + per-ticket predicates). Policies:
the scripted oracle (expects 1), a scripted no-op (expects 0), and an LLM tool policy on any model alias.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from crucible.actors.policy import LLMToolPolicy, ScriptedPolicy
from crucible.core.schema.actions import ToolCall
from crucible.core.store import Store
from crucible.runtime.harness.native import PolicyRef
from crucible.verify.runtime.plan import Context, GradeReport
from crucible.verify.runtime.trusted import Trusted
from crucible.world.world import World
from crucible_pack_toy import INSTRUCTION, PACK_ROOT, helpdesk_spec, responders
from crucible_pack_toy.verify import GOLD, plan


class HelpdeskTask:
    id = "toy/helpdesk/all_tickets@1"
    world_id = "toy/helpdesk"

    def __init__(self, instruction: str = INSTRUCTION, world_factory=None, backend: str = "local"):
        self.instruction = instruction
        self.world_factory = world_factory
        self.backend = backend  # "agentenv": the same World behind a real agentenv gateway

    async def open(self, workdir: Path) -> World:
        workdir = Path(workdir)
        if self.world_factory:
            w = self.world_factory(workdir)
        else:
            w = World(helpdesk_spec(), instructions={"agent": self.instruction}, store=Store(workdir / "s"), workdir=workdir / "w", pack_root=PACK_ROOT,
                      backend=self.backend)
            for r in responders():
                w.attach(r)
        await w.start()
        await w.reset()
        return w

    async def grade(self, world: World, workdir: Path) -> GradeReport:
        return await plan().arun(Context(Trusted(Path(workdir)), params={"view": world.verifier_view()}))

    def plan_for_progress(self):
        """The M7 plan whose checks are the dense-progress subgoals (``interop.replay.PlanSubgoals``)."""
        return plan()


def scripted(name: str, actions: list[ToolCall]) -> PolicyRef:
    return PolicyRef(id=f"scripted:{name}", make=lambda attempt: {"agent": ScriptedPolicy("agent", list(actions))})


ORACLE = scripted("oracle", GOLD)
NOOP = scripted("noop", [])


def llm_policy(llm, *, temperature: float = 0.7) -> PolicyRef:
    return PolicyRef(id=f"llm:{llm.model}", make=lambda attempt: {"agent": LLMToolPolicy("agent", llm, temperature=temperature)},
                     spend=lambda: (llm.spend.usd, getattr(llm.spend, "uncertain_usd", 0.0)), estimate_usd=0.002)


class FibTask:
    """A shell workspace task (the coding seam on CPU): create fib.py that prints the first 10 Fibonacci numbers. Graded
    from the collected patch, applied to a pristine repository by the grader (never in the agent's workspace)."""

    id = "toy/workspace/fib@1"
    world_id = "toy/workspace"
    instruction = "Create fib.py that prints the first 10 Fibonacci numbers (0 1 1 2 3 5 8 13 21 34), one per line. Run it to check, then finish."

    def spec(self, workdir: Path):
        from crucible.core.schema.world import WorldSpec

        return WorldSpec.model_validate({"world": "toy/workspace", "apps": [{"id": "ws", "template": "common/workspace",
                                         "state": {"adapter": "fs", "params": {"root": str(Path(workdir).resolve() / "ws")}},
                                         "params": {"shell": True, "allow_unsafe_local_shell": True, "init_git": True}}],
                                         "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}]})

    async def open(self, workdir: Path) -> World:
        workdir = Path(workdir)
        w = World(self.spec(workdir), instructions={"agent": self.instruction}, store=Store(workdir / "s"), workdir=workdir / "w")
        w._crucible_ws = Path(workdir).resolve() / "ws"
        await w.start()
        await w.reset()
        return w

    def workspace(self, world: World) -> Path:
        return world._crucible_ws

    async def broken(self, world: World) -> list[str]:
        return [] if (self.workspace(world) / ".git").exists() else ["the workspace repository (.git) was deleted"]

    async def grade_patch(self, patch: str, gdir: Path) -> GradeReport:
        import subprocess

        from crucible.verify.runtime.plan import Check, Gate, Plan, Verdict

        gdir = Path(gdir)
        repo = gdir / "repo"
        repo.mkdir(parents=True, exist_ok=True)

        def output(_ctx) -> Verdict:
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            if not patch.strip():
                return Verdict(False, {"error": "empty patch"})
            (gdir / "p.patch").write_text(patch)
            ap = subprocess.run(["git", "apply", str(gdir / "p.patch")], cwd=repo, capture_output=True, text=True, check=False)
            if ap.returncode:
                return Verdict(False, {"error": "patch does not apply", "log": ap.stderr[-300:]})
            r = subprocess.run(["python3", "fib.py"], cwd=repo, capture_output=True, text=True, timeout=30, check=False)
            return Verdict(r.stdout.split() == ["0", "1", "1", "2", "3", "5", "8", "13", "21", "34"], {"stdout": r.stdout[-200:]})

        return await Plan(self.id, [Gate("output")], [Check("output", "output", output)]).arun(Context(Trusted(gdir / "grade")))


def fib_actions(*, extra: list[ToolCall] = ()) -> list[ToolCall]:
    code = "a, b = 0, 1\nfor _ in range(10):\n    print(a)\n    a, b = b, a + b\n"
    return [ToolCall(tool_call_id="f1", function_name="ws.write_file", arguments={"path": "fib.py", "content": code}), *extra]


def crashing_tool_world(pack_root: Path, *, tool: str = "get_ticket", on_attempts: tuple[str, ...] = ("a1",)):
    """A toy helpdesk world factory whose ``tickets.<tool>`` raises ZeroDivisionError when the workdir path contains one of
    ``on_attempts`` (so the retry, in a fresh attempt dir, gets a healthy world)."""
    from crucible.core.store import Store
    from crucible.world.world import World
    from crucible_pack_toy import INSTRUCTION, helpdesk_spec, responders

    def factory(workdir: Path):
        root = pack_root
        if any(f"/{a}/" in str(workdir) for a in on_attempts):
            root = Path(tempfile.mkdtemp(prefix="crash-pack-"))
            shutil.copytree(pack_root / "data", root / "data")
            shutil.copytree(pack_root / "worlds", root / "worlds")
            tp = root / "data" / "helpdesk" / "tools" / "tickets.py"
            tp.write_text(tp.read_text() + f"\n\nimport functools as _ft\n_orig_{tool} = {tool}\n\n\n@_ft.wraps(_orig_{tool})\n"
                                           f"def {tool}(*a, **k):\n    return 1 / 0  # injected environment bug\n")
        w = World(helpdesk_spec(), instructions={"agent": INSTRUCTION}, store=Store(workdir / "s"), workdir=workdir / "w", pack_root=root)
        for r in responders():
            w.attach(r)
        return w

    return factory


def nemo_tasks() -> dict:
    """Env-server task factory (``python -m crucible.interop.nemo_gym.server --tasks crucible_pack_toy.runtime:nemo_tasks``)."""
    return {"toy/helpdesk": HelpdeskTask()}


class FibWorldTask:
    """The fib task as a World graded from its own workspace state (m13 C7: external agents through the A2A harness). With
    ``backend="agentenv"`` the World is served over MCP by a real agentenv gateway; the agent writes ``fib.py`` through the
    workspace tools and the grader runs it from the workspace (never in the agent's process)."""

    id = "toy/workspace/fib_world@1"
    world_id = "toy/workspace"
    instruction = FibTask.instruction

    def __init__(self, backend: str = "agentenv"):
        self.backend = backend

    async def open(self, workdir: Path) -> World:
        from crucible.core.schema.world import WorldSpec

        workdir = Path(workdir)
        root = (workdir / "ws").resolve()
        spec = WorldSpec.model_validate({"world": "toy/workspace", "backend": self.backend, "apps": [{"id": "ws", "template": "common/workspace",
                                         "state": {"adapter": "fs", "params": {"root": str(root)}},
                                         "params": {"shell": True, "allow_unsafe_local_shell": True, "init_git": True}}],
                                         "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}]})
        w = World(spec, instructions={"agent": self.instruction}, store=Store(workdir / "s"), workdir=workdir / "w")
        w._crucible_ws = root
        await w.start()
        await w.reset()
        return w

    async def grade(self, world: World, workdir: Path) -> GradeReport:
        import subprocess

        from crucible.verify.runtime.plan import Check, Gate, Plan, Verdict

        root = world._crucible_ws

        def output(_ctx) -> Verdict:
            if not (root / "fib.py").exists():
                return Verdict(False, {"error": "fib.py missing"})
            r = subprocess.run(["python3", "fib.py"], cwd=root, capture_output=True, text=True, timeout=30, check=False)
            return Verdict(r.stdout.split() == ["0", "1", "1", "2", "3", "5", "8", "13", "21", "34"], {"stdout": r.stdout[-200:]})

        return await Plan(self.id, [Gate("output")], [Check("output", "output", output)]).arun(Context(Trusted(Path(workdir) / "grade")))


WORDS = ["amber", "basil", "cedar", "delta", "ember", "fjord", "glade", "harbor", "indigo", "juniper", "kestrel", "lumen", "meadow", "nectar",
         "onyx", "pepper", "quartz", "raven", "sierra", "tundra", "umber", "violet", "willow", "xenon", "yarrow", "zephyr"]


class WordFileTask:
    """A seeded workspace task small enough for a sub-1B policy to learn by RL (the M12 learning-evidence gate): write
    ``<name>.txt`` containing exactly ``<word>`` and a newline through the workspace's ``write_file`` tool. Graded from
    the file; binary reward. Native backend, no shell, no Docker."""

    world_id = "toy/wordfile"

    def __init__(self, seed: int = 0):
        import random

        r = random.Random(f"toy/wordfile/{seed}")
        self.seed, self.word, self.name = seed, r.choice(WORDS), r.choice(WORDS) + str(r.randrange(10, 99))
        self.id = f"toy/wordfile/s{seed}@1"
        self.instruction = (f"Use the workspace tools to create the file {self.name}.txt containing exactly the word {self.word} followed by a "
                            "newline. Then reply that you are done.")

    async def open(self, workdir: Path) -> World:
        from crucible.core.schema.world import WorldSpec

        root = (Path(workdir) / "ws").resolve()
        spec = WorldSpec.model_validate({"world": self.world_id, "apps": [{"id": "ws", "template": "common/workspace",
                                         "state": {"adapter": "fs", "params": {"root": str(root)}}, "params": {"shell": False}}],
                                         "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}]})
        w = World(spec, instructions={"agent": self.instruction}, store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w")
        w._crucible_ws = root
        await w.start()
        await w.reset()
        return w

    async def grade(self, world: World, workdir: Path) -> GradeReport:
        from crucible.verify.runtime.plan import Check, Gate, Plan, Verdict

        path = world._crucible_ws / f"{self.name}.txt"

        def content(_ctx) -> Verdict:
            got = path.read_text() if path.exists() else None
            return Verdict(got is not None and got.strip() == self.word, {"got": (got or "")[:100]})

        return await Plan(self.id, [Gate("outcome")], [Check("file", "outcome", content)]).arun(Context(Trusted(Path(workdir))))
