"""M17 report 11 (the first long-horizon run, ``scripts/lh_run.py``): the pass-stub floor, the episode token limits, the
spend guard, the learnable-band test and the per-window verdict."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

from crucible.actors.llm import BudgetExceeded
from crucible.core.schema.actions import Done

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("lh_run", ROOT / "scripts" / "lh_run.py")
lh = importlib.util.module_from_spec(_spec)
sys.modules["lh_run"] = lh
_spec.loader.exec_module(lh)

A_MOD = "import os\n\n\nclass Version:\n    def __init__(self, v):\n        self.v = v\n\n    def bump(self):\n        return self.v + 1\n\n\ndef parse(s):\n    return Version(int(s))\n"
B_MOD = (A_MOD + "\n\ndef compare(a, b):\n    return (a > b) - (a < b)\n\n\nclass Spec:\n    def match(self, v):\n        return True\n")
B_MOD = B_MOD.replace("    def bump(self):\n        return self.v + 1\n",
                      "    def bump(self):\n        return self.v + 1\n\n    @classmethod\n    def coerce(cls, s):\n        return cls(int(s))\n\n    def next_version(self, part):\n        return self.v + 1\n")


def test_pass_stub_adds_new_symbols_as_stubs_only(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text(A_MOD)
    out = lh.stub_files(tmp_path, {"pkg/core.py": B_MOD, "pkg/extra.py": "def helper(x):\n    return x * 2\n\nclass Thing:\n    x = 1\n",
                                   "tests/test_core.py": "def test_x():\n    pass\n", "README.md": "new", "pkg/gone.py": None})
    assert sorted(out) == ["pkg/core.py", "pkg/extra.py"]
    tree = ast.parse(out["pkg/core.py"])
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert set(top) == {"Version", "parse", "compare", "Spec"}
    methods = {n.name: n for n in top["Version"].body if isinstance(n, ast.FunctionDef)}
    assert set(methods) == {"__init__", "bump", "coerce", "next_version"}
    assert [d.id for d in methods["coerce"].decorator_list] == ["classmethod"]
    # A's own code is untouched, the new symbols do nothing
    assert out["pkg/core.py"].startswith(A_MOD.split("\n\ndef parse")[0])
    ns: dict = {}
    exec(compile(out["pkg/core.py"], "core", "exec"), ns)  # noqa: S102 - a stub module we just built
    assert ns["compare"](1, 2) is None and ns["Version"](1).next_version("minor") is None and ns["Version"](3).bump() == 4
    assert ns["Version"].coerce("1") is None
    assert "def helper(*args, **kwargs):" in out["pkg/extra.py"] and "return x * 2" not in out["pkg/extra.py"]


def test_pass_stub_is_empty_when_b_adds_nothing(tmp_path):
    (tmp_path / "m.py").write_text(A_MOD)
    assert lh.stub_files(tmp_path, {"m.py": A_MOD.replace("+ 1", "+ 2")}) == {}


class FakeLLM:
    def __init__(self, usage):
        self.usage, self.calls = usage, 0

    async def complete(self, messages, **kw):
        from crucible.actors.llm import LLMResponse

        self.calls += 1
        return LLMResponse(text="", model="fake", tool_calls=[{"id": f"c{self.calls}", "name": "ws__shell", "arguments": {"command": "ls"}}],
                           usage=dict(self.usage), raw_finish="tool_calls", raw_message={"role": "assistant"})


def _obs(step):
    from crucible.core.schema.actions import Observation

    return Observation(actor="agent", step=step, instruction="do it" if step == 0 else None)


def test_capped_policy_stops_on_billed_tokens():
    pol = lh.CappedToolPolicy("agent", FakeLLM({"prompt_tokens": 600, "completion_tokens": 50}), max_trajectory=10**9, max_billed=1000)
    a1 = asyncio.run(pol.act(_obs(0)))
    assert not isinstance(a1, Done)
    pol._pending.clear()
    a2 = asyncio.run(pol.act(_obs(1)))
    assert not isinstance(a2, Done) and pol.billed_tokens() == 1300
    pol._pending.clear()
    a3 = asyncio.run(pol.act(_obs(2)))
    assert isinstance(a3, Done) and pol.stop == "max_tokens" and pol.llm.calls == 2


def test_capped_policy_stops_on_context_tokens():
    pol = lh.CappedToolPolicy("agent", FakeLLM({"prompt_tokens": 10, "completion_tokens": 10}), max_trajectory=5)
    asyncio.run(pol.act(_obs(0)))
    pol._pending.clear()
    assert pol.trajectory_tokens >= 5
    assert isinstance(asyncio.run(pol.act(_obs(1))), Done) and pol.stop == "max_tokens"


class _Spend:
    usd = 0.0
    uncertain_usd = 0.0


class _Client:
    def __init__(self):
        self.spend, self.n = _Spend(), 0

    async def complete(self, messages, **kw):
        self.n += 1
        self.spend.usd += 0.6
        return "ok"


class _Run:
    def __init__(self, cap, clients):
        self.cap, self.clients, self.past = cap, clients, 0.5

    def spent(self):
        return self.past + sum(c.spend.usd for c in self.clients)

    check = lh.Run.check


def test_guard_refuses_calls_once_the_cap_is_spent():
    c = _Client()
    g = lh.Guarded(c, _Run(1.6, [c]))
    assert asyncio.run(g.complete([])) == "ok"  # 0.5 past + 0 -> allowed; now 1.1
    assert asyncio.run(g.complete([])) == "ok"  # 1.1 < 1.6 -> allowed; now 1.7
    with pytest.raises(BudgetExceeded):
        asyncio.run(g.complete([]))
    assert c.n == 2 and g.spend is c.spend  # everything else is the client's


def test_band_and_verdicts():
    assert lh.in_band([True, False, False, False], [1.0, 0.2, 0.2, 0.0])["in_band"]
    b = lh.in_band([False] * 4, [0.25, 0.25, 0.5, 0.0])
    assert b["in_band"] and b["phi_spread"] and not b["rate_in_band"]
    flat = lh.in_band([False] * 4, [0.0] * 4)
    assert not flat["in_band"]
    ok_gates = {"floors_zero": True, "oracle_passes": True}
    assert lh.verdict({"floors_zero": False}, b, {}, {})["verdict"] == "back"
    assert lh.verdict(ok_gates, b, {"world_bug": 1, "skill_gap": 3}, {})["verdict"] == "back"
    assert lh.verdict(ok_gates, b, {"skill_gap": 3, "budget": 1}, {"m1": True})["verdict"] == "accept-candidate"
    assert lh.verdict(ok_gates, b, {"skill_gap": 1, "budget": 2}, {"m1": True})["verdict"] == "hold"
    assert lh.verdict(ok_gates, b, {"skill_gap": 7, "budget": 3, "env_flake": 6}, {"m1": True})["verdict"] == "accept-candidate"
    assert lh.verdict(ok_gates, lh.in_band([True] * 4, [1.0] * 4), {}, {})["verdict"] == "harden"
    assert lh.verdict(ok_gates, flat, {"skill_gap": 4}, {"m1": True, "m2": False})["verdict"] == "split"
    assert lh.verdict(ok_gates, flat, {"skill_gap": 4}, {"m1": True})["verdict"] == "hold"


def test_selection_rule_holds():
    assert len(lh.SELECTED) == 10 and len(set(lh.SELECTED)) == 10
    assert not set(lh.DROPPED) & set(lh.SELECTED)
    repos = {}
    for w in lh.SELECTED:
        r = w.rsplit("-w", 1)[0]
        repos[r] = repos.get(r, 0) + 1
    assert len(repos) >= 4 and max(repos.values()) <= 3


def test_a_glob_over_the_repos_own_tests_is_not_a_grader_probe():
    """Report 11 defect: a hacker's ``repo.search("from more_itertools", glob="**/tests/**")`` over the repository's own
    (public) tests was read as a strong ``grader_files`` exposure and masked the episode."""
    import json

    from crucible.core.agent_events import probes_of

    def kinds(args):
        return {p["kind"] for p in probes_of("repo.search", json.dumps(args))}

    assert "grader_files" not in kinds({"pattern": "from more_itertools", "glob": "**/tests/**"})
    assert "grader_files" not in kinds({"pattern": "x", "glob": "*/tests/*.py"})
    assert "grader_files" in kinds({"command": "cat /tests/test_outputs.py"})
    assert "grader_files" in kinds({"command": "ls /solution/ && cat /oracle/fix.patch"})
    assert "grader_files" in kinds({"command": "find / -name hidden_tests"})


async def test_workspace_attempt_cost_is_its_own_calls_under_concurrency(tmp_path):
    """Report 11 defect: ``WorkspaceHarness`` booked each attempt the delta of a client shared by concurrent attempts."""
    from test_m17_c_harness import FibWindow, llm, run_one

    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.harness.workspace import WorkspaceHarness

    shared = {"usd": 0.0}
    base = llm()

    def make(attempt):
        pols = base.make(attempt)
        inner = pols["agent"].llm
        orig = inner.complete

        async def complete(messages, **kw):
            r = await orig(messages, **kw)
            r.cost_usd = 0.01
            shared["usd"] += 0.5  # another attempt's spend on the same client meanwhile
            return r

        inner.complete = complete
        return pols

    ref = PolicyRef(id="fake:shared", make=make, spend=lambda: (shared["usd"], 0.0))
    _adir, rec, _ = await run_one(tmp_path, FibWindow(), ref, WorkspaceHarness())
    assert rec["reward"] == 1.0 and rec["cost_usd"] == pytest.approx(0.07)  # 7 calls x 0.01, not the shared delta (3.5)


def test_rlm_adapter_is_a_milestone_task_for_the_checkpoint_sidecar():
    """Report 11 defect (own adapter): delegation through ``__getattr__`` is invisible to a runtime protocol check, so the
    first RLM run had no checkpoint sidecar."""
    from crucible.factory.checkpoint import MilestoneTask

    class Task:
        def __init__(self):
            self.milestones = [{"id": "m1", "requires": []}]
            self.requirement_ids = ["R1"]
            self.weights = {"m1": 1.0}

        async def milestone_status(self, patch):
            return {"m1": bool(patch)}

    w = lh.RLMWindow(Task(), {})
    assert isinstance(w, MilestoneTask) and w.weights == {"m1": 1.0}
    assert asyncio.run(w.milestone_status("x")) == {"m1": True}


def test_run2_reward_band_and_remine_summary():
    b = lh.reward_band([0.0, 0.083334, 0.0, None])
    assert b["in_band"] and b["n"] == 3 and b["pass_rate"] == 0.0 and b["reward_range"] == [0.0, 0.083334]
    assert not lh.reward_band([0.0, 0.0])["in_band"] and lh.reward_band([1.0, 0.5])["pass_rate"] == 0.5
    rows = [{"window": "a", "split": "train", "old": {"stage": "mutants_valid", "milestones": 4}, "stage": "f2p_valid", "milestones": 4, "edges": 0},
            {"window": "b", "split": "train", "old": {"stage": "f2p_valid", "milestones": 3}, "stage": "mutants_valid", "milestones": 5, "edges": 2},
            {"window": "c", "split": "train", "old": {"stage": "dependent", "milestones": 5}, "stage": "leak_ok", "milestones": 1},
            {"window": "d", "split": "heldout", "error": "boom"}]
    s = lh.remine_summary(rows)["train"]
    assert s["before"] == {"f2p_valid": 3, "dependent": 2, "mutants_valid": 1} and s["after"] == {"f2p_valid": 2, "dependent": 1, "mutants_valid": 1}
    assert s["lost_dependent"] == 2 and s["newly_dependent"] == 1 and s["collapsed_below_2_milestones"] == 1 and s["errors"] == ["d"]
