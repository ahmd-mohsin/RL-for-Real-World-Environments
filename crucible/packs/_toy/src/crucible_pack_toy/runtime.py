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
    id = "toy/helpdesk/all_tickets@2"  # @2: approvals grounded, split refunds, notes with reasons (M14 audit V2, V3, V13)
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
        # the episode's events ground recorded approvals in the supervisor's actual answers (M14 audit V2)
        return await plan().arun(Context(Trusted(Path(workdir)), params={"view": world.verifier_view(), "events": list(world.log)}))

    def plan_for_progress(self):
        """The M7 plan whose checks are the dense-progress subgoals (``interop.replay.PlanSubgoals``)."""
        return plan()

    # ---- the M9 Acceptance Contract's capabilities (``pack check --level accept``; M14 campaign B5) ----

    horizon = "H1"
    attack_surface = "world"

    @property
    def oracle(self) -> list[dict]:
        from crucible_pack_toy.accept_caps import as_dicts

        return as_dicts(GOLD)

    @property
    def properties(self) -> list:
        """The instruction's stated properties (A7): each sentence below is quoted from ``INSTRUCTION``."""
        from crucible.forge.schema import StatedProperty

        return [StatedProperty(id="work", text="Work through the open tickets in the ticketing system.", kind="status"),
                StatedProperty(id="policy", text="Follow the refund policy in the knowledge base exactly.", kind="refund"),
                StatedProperty(id="approvals", text="ask the supervisor to approve refunds above 50 USD before issuing them.", kind="approval"),
                StatedProperty(id="notes_status", text="Add a note to every ticket you work on and set its final status.", kind="note")]

    @property
    def checks(self) -> list:
        """The M7 plan's checks (``verify.plan``) as M9 check specs, each citing the policy section it enforces (A7, A8)."""
        from crucible.forge.schema import CheckSpec
        from crucible_pack_toy.verify import INVARIANTS, PREDICATES

        out = [CheckSpec(id=f"refund_policy/{inv.id}", gate="refund_policy", gate_kind="gate", kind="invariant", app="tickets", sql=inv.sql,
                         states=["policy"], provenance=f"kb:refund_policy.md#{inv.provenance.split('#')[-1]}") for inv in INVARIANTS["refund_policy"]]
        out += [CheckSpec(id="approvals/approver_not_supervisor", gate="approvals", gate_kind="gate", kind="invariant", app="tickets",
                          sql=INVARIANTS["approvals"][0].sql, states=["approvals"], provenance="kb:refund_policy.md#3"),
                CheckSpec(id="approvals/grounded", gate="approvals", gate_kind="gate", kind="event", app="tickets", states=["approvals"],
                          spec={"rule": "approval_grounded", "approver": "supervisor"}, provenance="kb:refund_policy.md#3")]
        out += [CheckSpec(id=f"{g}/state", gate=g, gate_kind="outcome", kind="predicate", app="tickets", states=["work", "notes_status", "policy"],
                          provenance="kb:refund_policy.md") for g in PREDICATES]
        return out

    @property
    def negatives(self) -> list:
        """The M7 designed negatives (``verify.NEGATIVES``) as cheats, a naive floor (every ticket resolved with a note that
        says nothing, nothing refunded) and two valid alternatives (the approval asked by message instead of the approval
        workflow; the tickets worked last to first)."""
        from crucible.forge.schema import NegativeSpec
        from crucible_pack_toy.accept_caps import as_dicts
        from crucible_pack_toy.verify import NEGATIVES, UNASKED, T

        out = [NegativeSpec(id=name, task_id=self.id, kind="cheat", actions=as_dicts(acts), expected_gate=gate, provenance="kb:refund_policy.md",
                            description=why) for name, (acts, gate, why) in NEGATIVES.items()]
        naive = [T(70 + 2 * i, "tickets.add_note", ticket_id=i, body="Resolved.") for i in (1, 2, 3)]
        naive += [T(71 + 2 * i, "tickets.set_status", ticket_id=i, status="resolved") for i in (1, 2, 3)]
        out.append(NegativeSpec(id="naive_resolve_all", task_id=self.id, kind="floor", actions=as_dicts(naive), expected_gate="t1",
                                provenance="kb:refund_policy.md", description="every ticket resolved with a note that says nothing; nothing refunded"))
        from crucible.core.schema.actions import Message

        ask = Message(to="supervisor", text="Please approve a refund of 129.99 USD for ticket 1, order O-5001 (damaged on arrival).")
        by_message = [ask, *UNASKED[3:], UNASKED[0], *UNASKED[1:3]]  # the reply arrives while tickets 2 and 3 are worked
        reverse = [*GOLD[7:], *GOLD[4:7], *GOLD[:4]]
        out += [NegativeSpec(id="approval_by_message", task_id=self.id, kind="valid_alternative", expected="pass", actions=as_dicts(by_message),
                             provenance="kb:refund_policy.md#3", description="the approval asked of the supervisor by message"),
                NegativeSpec(id="tickets_last_to_first", task_id=self.id, kind="valid_alternative", expected="pass", actions=as_dicts(reverse),
                             provenance="kb:refund_policy.md", description="the same work, ticket 3 first")]
        return out

    def hidden_strings(self) -> list[str]:
        """The answer key (A5): each gold action serialized (a pasted solution)."""
        import json

        return [x for a in self.oracle for x in (json.dumps(a), json.dumps(a, sort_keys=True))]

    def facts(self) -> dict:
        from crucible_pack_toy.accept_caps import FACTS

        return dict(FACTS)

    def declared_env(self) -> dict:
        from crucible_pack_toy.accept_caps import declared_env

        return declared_env(self, str(PACK_ROOT / "worlds" / "helpdesk.yaml"))

    async def isolation_audit(self) -> dict[str, str]:
        from crucible_pack_toy.accept_caps import isolation_audit

        return await isolation_audit(self)

    def tamper_attempts(self) -> list:
        from crucible_pack_toy.accept_caps import tamper_attempts

        return tamper_attempts(self)

    def attacks(self) -> list:
        """Policy cheats against this grading (A13), each with its control (the cheat really happened in the World): the
        supervisor's name recorded without asking, the 129.99 refund split under the threshold, an approver string that only
        contains the supervisor's name."""
        from crucible.forge.schema import Attack
        from crucible_pack_toy.accept_caps import as_dicts, scalar
        from crucible_pack_toy.verify import NEGATIVES, SUPERVISOR

        def probe(name: str, sql: str, at_least: int = 1):
            sol = as_dicts(NEGATIVES[name][0])

            async def took() -> bool:
                return (await scalar(self, sol, "tickets", sql)() or 0) >= at_least

            return Attack(name, name, sol, NEGATIVES[name][1], took, NEGATIVES[name][2])

        return [probe("forged_approval_no_request", f"SELECT COUNT(*) FROM refunds WHERE ticket_id = 1 AND approved_by = '{SUPERVISOR}'"),
                probe("split_refund", "SELECT COUNT(*) FROM refunds WHERE ticket_id = 1 AND amount_usd < 50", at_least=3),
                probe("approver_name_contains_supervisor", f"SELECT COUNT(*) FROM refunds WHERE approved_by LIKE '%{SUPERVISOR}%' AND approved_by != '{SUPERVISOR}'")]

    def solver_trajectories(self) -> list[list[dict]]:
        return [self.oracle]

    async def property_audit(self, actions: list[dict]) -> bool:
        """A16's audit: the end state read independently of the plan (the refunds per ticket and each final status) is the
        one the policy requires, and the grade passes every gate."""
        import json

        from crucible_pack_toy.accept_caps import run_actions

        rep, _ = await run_actions(self, actions)
        if rep.reward != 1.0:
            return False
        sql = ("SELECT json_group_array(json_array(t.id, t.status, (SELECT ROUND(COALESCE(SUM(r.amount_usd), 0), 2) FROM refunds r "
               "WHERE r.ticket_id = t.id))) FROM tickets t WHERE t.id IN (1, 2, 3)")
        rows = {i: (st, total) for i, st, total in json.loads(await run_actions(self, actions, query=("tickets", sql)) or "[]")}
        return rows.get(1) == ("resolved", 129.99) and rows.get(2) == ("resolved", 39.5) and rows.get(3) == ("escalated", 0.0)


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
