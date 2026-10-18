"""P2 valid alternatives (``factory.alternatives``): the changed definitions, splicing a rewrite back (names, parameters
and decorators kept), a scripted rewriter's candidate accepted only when the real tests and the second verifier both
pass it, solver end states as alternatives, and the S8 coverage rule (>= 2, >= 1 from another family)."""

from __future__ import annotations

import asyncio

import pytest
from test_m17_p2_seeded import shadow_suite, toy

from crucible.factory.alternatives import (
    Alternative,
    accept,
    changed_defs,
    coverage,
    from_solver_states,
    pick_defs,
    rewrite_oracle,
    salvage,
    same_code,
    splice,
)
from crucible.factory.verifier2 import WindowRunner, qualify

A = "class K:\n    def keep(self):\n        return 1\n\n    def grow(self, x):\n        return x\n\n\ndef same():\n    return 0\n"
B = ("class K:\n    def keep(self):\n        return 1\n\n    @property\n    def size(self):\n        n = 0\n        for _ in range(3):\n            n += 1\n        return n\n\n"
     "    def grow(self, x, y=2):\n        total = x + y\n        return total\n\n\ndef same():\n    return 0\n\n\ndef new(a, *rest):\n    out = [a]\n    out.extend(rest)\n    return out\n")


class Scripted:
    def __init__(self, reply):
        self.reply = reply
        self.seen = []

    async def complete_json(self, messages, **kw):
        self.seen.append(messages)
        return self.reply


def test_changed_defs_and_splice() -> None:
    ds = changed_defs("m.py", A, B)
    assert [d.qualname for d in ds] == ["K.size", "K.grow", "new"] and ds[0].source.startswith("@property") and ds[0].indent == 4
    new = {"m.py::K.size": "@property\ndef size(self):\n    return len([0, 0, 0])\n",
           "m.py::K.grow": "def grow(self, x, y=2):\n    return sum((x, y))\n",
           "m.py::new": "def new(b):\n    return [b]\n"}  # renamed parameters: refused
    text, problems = splice(B, ds, new)
    assert "return len([0, 0, 0])" in text and "return sum((x, y))" in text and "out.extend(rest)" in text
    assert problems == ["m.py::new: the rewrite renames it or changes its parameters or decorators"]
    assert not same_code(text, B) and same_code(B, B + "\n# a comment\n")
    _, p2 = splice(B, ds[:1], {"m.py::K.size": "def size(self):\n    return 3\n"})  # dropped the decorator
    assert p2 and "decorators" in p2[0]
    assert pick_defs({"m.py": B}, __import__("pathlib").Path("/nonexistent"), min_lines=4)[0].qualname in ("K.size", "new")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return toy(tmp_path_factory.mktemp("p2alt"))


def test_rewrite_accepted_only_when_both_verifiers_pass(built) -> None:
    task, _, _ = built
    runner = WindowRunner(task)
    suite = shadow_suite()
    q = qualify(suite, oracle=asyncio.run(runner.shadow(suite, task.oracle_patch())), alternatives={}, start=None, mutants={},
                req_milestone={"R1": "m01", "R2": "m02", "R3": "m03"})
    good = Scripted({"defs": [{"qualname": "pkg/gamma.py::gamma", "source": "def gamma():\n    value = beta()\n    return value + 1\n"},
                              {"qualname": "pkg/beta.py::beta", "source": "def beta():\n    return 1 + 1\n"}]})
    alt = asyncio.run(rewrite_oracle(good, task, style="imperative", model="z-ai/glm-4.7-flash", family="glm", min_lines=1))
    assert alt.patch and alt.evidence["files_changed"] == ["pkg/beta.py", "pkg/gamma.py"] and "pkg/gamma.py" in good.seen[0][1]["content"]
    v1 = asyncio.run(runner.hidden(alt.patch))
    asyncio.run(accept(alt, v1=v1, v2=q.verdict(asyncio.run(runner.shadow(suite, alt.patch)))))
    assert alt.accepted and alt.evidence["v1_reward"] == 1.0
    broken = Scripted({"defs": [{"qualname": "pkg/gamma.py::gamma", "source": "def gamma():\n    return beta() + 2\n"}]})
    bad = asyncio.run(rewrite_oracle(broken, task, style="functional", model="qwen/qwen3.7-flash", family="qwen", min_lines=1))
    asyncio.run(accept(bad, v1=asyncio.run(runner.hidden(bad.patch)), v2=q.verdict(asyncio.run(runner.shadow(suite, bad.patch)))))
    assert not bad.accepted and any("real tests fail" in r for r in bad.reasons) and any("second verifier fails" in r for r in bad.reasons)
    noop = asyncio.run(rewrite_oracle(Scripted({"defs": []}), task, style="functional", family="qwen", min_lines=1))
    assert noop.patch == "" and "is the oracle" in noop.reasons[0]
    assert coverage([alt, bad], writer_family="qwen") == {"accepted": 1, "other_family": 1, "families": ["glm"], "ok": False, "ids": [alt.id]}
    twin = Alternative.from_json({**alt.to_json(), "id": "x", "family": "qwen"})
    assert coverage([alt, twin], writer_family="qwen")["ok"] and not coverage([twin, Alternative.from_json({**twin.to_json(), "id": "y"})],
                                                                             writer_family="qwen")["ok"]
    # salvage: a two-definition rewrite with one wrong definition keeps the right one (grader runs only, no model call)
    mixed = Scripted({"defs": [{"qualname": "pkg/gamma.py::gamma", "source": "def gamma():\n    return beta() + 2\n"},
                               {"qualname": "pkg/beta.py::beta", "source": "def beta():\n    return 4 // 2\n"}]})
    m = asyncio.run(rewrite_oracle(mixed, task, style="functional", family="deepseek", min_lines=1))
    assert sorted(m.evidence["rewritten"]) == ["pkg/beta.py::beta", "pkg/gamma.py::gamma"]

    async def passes(patch):
        return (await runner.hidden(patch))["reward"] == 1.0

    sub = asyncio.run(salvage(task, m, passes, chunks=2, min_lines=1))
    assert sub is not None and sub.id.endswith("-subset") and list(sub.evidence["rewritten"]) == ["pkg/beta.py::beta"]
    assert asyncio.run(passes(sub.patch)) and "4 // 2" in sub.patch
    child = Alternative.from_json({**sub.to_json(), "accepted": True})
    assert coverage([Alternative.from_json({**m.to_json(), "accepted": True}), child], writer_family="qwen")["accepted"] == 1  # one family root


def test_solver_end_states_as_alternatives() -> None:
    rows = [{"id": "e1", "patch": "p", "policy": "solver:qwen/qwen3.7-flash", "v1": {"reward": 1.0}, "v2": {"R1": True, "R2": None}},
            {"id": "e2", "patch": "p", "policy": "solver:openai/gpt-oss-20b", "v1": {"reward": 1.0}, "v2": {"R1": False}},
            {"id": "e3", "patch": "p", "policy": "solver:openai/gpt-oss-20b", "v1": {"reward": 0.0}, "v2": {"R1": True}},
            {"id": "e4", "patch": "p", "policy": "solver:x/y", "v1": {"reward": 1.0}, "v2": {"R1": None}}]
    alts = from_solver_states(rows)
    assert [a.accepted for a in alts] == [True, False, False, False] and alts[0].family == "qwen"
