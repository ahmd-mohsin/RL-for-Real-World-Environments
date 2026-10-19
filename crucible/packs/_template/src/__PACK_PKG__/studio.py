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

    def identity_config(self) -> dict:
        """What a run's identity needs of this task (M10 lab audit R-4): its content and its parameters, as JSON. A resume refuses
        a changed task by dotpath (``tasks.<id>.pack``: the pack's files, so an edited grader is caught; ``tasks.<id>.seed``).
        Declare every constructor parameter you store and the digest of every file or tree the grade depends on; a task with state
        and no ``identity_config()`` is refused when the run starts. Never declare the seed of a held-out (benchmark) item itself:
        the document is written to ``run.json`` under the bench's shared out directory, and the seed plus the pack's public
        generator rebuilds the item and its answer. Declare ``seed_digest(seed)``: a changed seed is still a changed digest."""
        from __PACK_PKG__ import PACK_ROOT
        from crucible.runtime.identity import seed_digest, tree_digest

        return {"pack": tree_digest(PACK_ROOT), "seed": seed_digest(self.seed)}

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

    def oracle_actions(self) -> list:
        """The gold of a generated item: ``bench run`` and ``pack check`` replay it before the item counts."""
        return self.gold()

    # ---- the M9 Acceptance Contract (``crucible pack check --level accept``). Each capability below is what one check
    # reads; keep them true when you replace the task (a missing capability fails its check, it is never skipped).

    horizon = "H1"  # episode length class: H1 one sitting, H3+ long horizon (more designed negatives needed)
    attack_surface = "world"  # the policy acts through World tools

    def _gold_dicts(self) -> list[dict]:
        return [a.model_dump(mode="json") for a in self.gold()]

    @property
    def properties(self) -> list:
        """A7: what the instruction states, quoted verbatim; every property is graded by a check below."""
        from crucible.forge.schema import StatedProperty

        return [StatedProperty(id="greeting", text=self.instruction, kind="outcome")]

    @property
    def checks(self) -> list:
        """A7/A8: what ``grade`` checks, each with the property it grades and its provenance. The task's whole spec is in
        its instruction, so the check cites it (``instruction:``); a domain check cites its policy (``kb:<doc>#<section>``)."""
        from crucible.forge.schema import CheckSpec

        return [CheckSpec(id="greeting/content", gate="outcome", gate_kind="outcome", kind="predicate", app="ws", states=["greeting"],
                          provenance="instruction:greeting.txt holds exactly the word and a newline")]

    def _write(self, path: str, content: str, k: str = "n1") -> list[dict]:
        return [{"kind": "tool_call", "tool_call_id": k, "function_name": "ws.write_file", "arguments": {"path": path, "content": content}}]

    @property
    def negatives(self) -> list:
        """A2/A3: a naive floor, cheats that each fail the gate, and a valid alternative that must pass."""
        from crucible.forge.schema import NegativeSpec

        other = next(w for w in WORDS if w != self.word)
        prov = "instruction:greeting"
        return [NegativeSpec(id="naive_empty_file", task_id=self.id, kind="floor", actions=self._write("greeting.txt", ""), expected_gate="outcome",
                             provenance=prov, description="creates the file and writes nothing"),
                NegativeSpec(id="wrong_word", task_id=self.id, kind="cheat", actions=self._write("greeting.txt", other + "\n"), expected_gate="outcome",
                             provenance=prov, description="another greeting"),
                NegativeSpec(id="every_greeting", task_id=self.id, kind="cheat", actions=self._write("greeting.txt", "\n".join(WORDS) + "\n"),
                             expected_gate="outcome", provenance=prov, description="every greeting, hoping one matches"),
                NegativeSpec(id="wrong_file", task_id=self.id, kind="cheat", actions=self._write("greetings.txt", self.word + "\n"), expected_gate="outcome",
                             provenance=prov, description="the right word in another file"),
                NegativeSpec(id="draft_then_fix", task_id=self.id, kind="valid_alternative", expected="pass", provenance=prov,
                             actions=self._write("greeting.txt", "draft\n", "v1") + self._write("greeting.txt", self.word + "\n", "v2"),
                             description="a draft first, then the right content")]

    def hidden_strings(self) -> list[str]:
        """A5: the answer key the agent must never read in its environment (here: the gold call serialized)."""
        import json

        return [x for a in self._gold_dicts() for x in (json.dumps(a), json.dumps(a, sort_keys=True))]

    def facts(self) -> dict:
        return {"images": False, "fresh_gates": False, "perf_bars": False, "reference_build": False, "sibling_overlap": False}

    def declared_env(self) -> dict:
        return {"kind": "world", "world": self.world_id, "backend": "local", "network": "none (no network tools)"}

    async def _run(self, actions: list, *, read: str | None = None):
        """``(GradeReport, end-state fingerprint)`` of ``actions`` on a fresh World, or the content of file ``read`` after them."""
        import hashlib
        import json
        import shutil
        import tempfile

        from crucible.actors.policy import EpisodeLimits, run_episode

        d = Path(tempfile.mkdtemp(prefix="__PACK_NAME__-accept-"))
        try:
            w = await self.open(d / "work")
            try:
                acts = [ToolCall(tool_call_id=a.get("tool_call_id") or f"a{i}", function_name=a["function_name"], arguments=a.get("arguments") or {})
                        if isinstance(a, dict) else a for i, a in enumerate(actions) if not isinstance(a, dict) or a.get("function_name")]
                await run_episode(w, {"agent": ScriptedPolicy("agent", acts)}, limits=EpisodeLimits(max_steps=max(20, 2 * len(acts))), reset=False)
                if read is not None:
                    f = w._pack_ws / read
                    return f.read_text() if f.is_file() else None
                fps = await w.verifier_view().fingerprints()
                return await self.grade(w, d / "grade"), hashlib.sha256(json.dumps(fps, sort_keys=True, default=str).encode()).hexdigest()[:24]
            finally:
                await w.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def isolation_audit(self) -> dict[str, str]:
        """A6: the World containment audit (reset stability, tool declarations, permissions, leaks of the hidden strings)."""
        import shutil
        import tempfile

        from crucible.world.audit import audit_world

        d = Path(tempfile.mkdtemp(prefix="__PACK_NAME__-audit-"))
        try:
            w = await self.open(d / "work")
            try:
                rep = await audit_world(w, forbidden=self.hidden_strings(), actor="agent")
                return {c.id: c.status for c in rep.checks if c.severity in ("critical", "major")}
            finally:
                await w.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def tamper_attempts(self) -> list:
        """A6: out-of-contract writes (a shell the workspace does not grant, a system restore, the verifier's view). Each must
        be refused, score 0 and leave the workspace untouched."""
        from crucible.forge.schema import Attack

        tries = {"undeclared_shell": {"function_name": "ws.run", "arguments": {"cmd": f"printf '{self.word}\\n' > greeting.txt"}},
                 "system_restore": {"function_name": "__world.restore", "arguments": {"ref": "gold"}},
                 "verifier_view": {"function_name": "verifier.write_file", "arguments": {"path": "greeting.txt", "content": self.word + "\n"}}}
        out = []
        for k, a in tries.items():
            sol = [{**a, "kind": "tool_call", "tool_call_id": f"tamper-{k}"}]

            async def changed(sol=sol) -> bool:
                return (await self._run(sol))[1] != (await self._run([]))[1]

            out.append(Attack(k, "out_of_contract_write", sol, None, changed))
        return out

    def attacks(self) -> list:
        """A13: cheats against this grading, each with a control proving it took effect (the file really holds the cheat)."""
        from crucible.forge.schema import Attack

        def wrote(sol: list, expect: str):
            async def took() -> bool:
                return (await self._run(sol, read="greeting.txt")) == expect
            return took

        every = "\n".join(WORDS) + "\n"
        padded = f"  {self.word}  \n"
        return [Attack("every_greeting", "enumerate_answers", self._write("greeting.txt", every), "outcome", wrote(self._write("greeting.txt", every), every)),
                Attack("padded_word", "near_miss", self._write("greeting.txt", padded), "outcome", wrote(self._write("greeting.txt", padded), padded))]

    def solver_trajectories(self) -> list[list[dict]]:
        return [self._gold_dicts()]

    async def property_audit(self, actions: list) -> bool:
        """A16: an independent reading of the stated property (the file's bytes), not the grader."""
        return (await self._run(list(actions), read="greeting.txt")) == self.word + "\n"


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
    # ``config`` is the policy's run identity (M10 lab audit R-4): ``make`` is a closure, so say what the policy is (a model and its
    # temperature, a script's name); a PolicyRef without it is refused when the run starts
    return [PolicyRef(id="scripted:gold", make=lambda a: {"agent": ScriptedPolicy("agent", a.task.gold())}, config={"script": "task gold"}),
            PolicyRef(id="scripted:noop", make=lambda a: {"agent": ScriptedPolicy("agent", [])}, config={"script": "noop"})]


def fixtures() -> list[dict]:
    """Designed cases with their exact expected grade: the grader must reject near misses, not only the empty episode."""
    t = GreetingTask(0)
    write = lambda path, content: [ToolCall(tool_call_id="f1", function_name="ws.write_file", arguments={"path": path, "content": content})]
    return [{"name": "gold", "task": t, "actions": t.gold(), "reward": 1.0},
            {"name": "noop", "task": t, "actions": [], "reward": 0.0, "fails_gate": "outcome"},
            {"name": "wrong_word", "task": t, "actions": write("greeting.txt", "goodbye\n"), "reward": 0.0, "fails_gate": "outcome"},
            {"name": "wrong_file", "task": t, "actions": write("greetings.txt", t.word + "\n"), "reward": 0.0, "fails_gate": "outcome"},
            {"name": "no_newline", "task": t, "actions": write("greeting.txt", t.word), "reward": 0.0, "fails_gate": "outcome"},
            # an adversary: a shortcut policy that never reads the task. ``pack check --level semantic`` runs it on every
            # task and generated item, and fails the pack when it scores above expect_max on any of them.
            {"name": "adversary:every_greeting", "policy": lambda task: write("greeting.txt", "\n".join(WORDS) + "\n"), "expect_max": 0.0}]
