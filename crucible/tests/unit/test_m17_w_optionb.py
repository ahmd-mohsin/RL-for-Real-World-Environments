"""Option B leftovers of the executable lane (07 §3-§4; 09 §5).

(a) The spec author was cut off: 6 discovery tasks died at ``finish=length`` (``runs/discovery_summary.json`` "dropped").
The cap is now ``SPEC_MAX_TOKENS`` and a truncated reply gets one compact retry that names tests by number.
(b) The trial's adapter bug: a client without ``**kwargs`` raised TypeError when a state retry added ``sample_key`` (or
the solver runner ``json_mode``). Every lane call to a client now goes through ``call_supported``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from crucible.actors.llm import LLMOutputError
from crucible.factory.lanes.executable import SPEC_MAX_TOKENS, ExecLane, LaneStore, LocalSandbox, MinedTask, call_supported, truncated
from crucible.factory.machine import StateContext

TESTS = {"tests/test_x.py": "def test_one():\n    assert 1\n\n\ndef test_two():\n    assert 2\n"}
F2P = ["tests/test_x.py::test_one", "tests/test_x.py::test_two"]


def lane(tmp: Path) -> ExecLane:
    store = LaneStore(tmp / "lane")
    t = MinedTask(id="toy-0000000001", repo="toy", sha="1" * 40, parent="0" * 40, date="2024-01-01T00:00:00+00:00", subject="fix x", body="",
                  src_files=["x.py"], test_files=sorted(TESTS), tests=dict(TESTS), base_src={"x.py": "X = 1\n"}, fix_src={"x.py": "X = 2\n"},
                  oracle={"edits": []}, f2p=list(F2P), p2p=[], excluded={}, observation={"id": "obs-toy"}, validation={})
    store.save(t)
    return ExecLane(store, LocalSandbox())


def ctx(client, *, attempt: int) -> StateContext:
    cell = {"surface_spec": {"task": "toy-0000000001"}}
    return StateContext("run/1", "spec", 1, attempt, {"intent": {"id": "i"}, "cell": cell}, [], {"cell": cell}, "planner", None, client)


class Strict:
    """A scripted client with the minimal signature: no ``sample_key``, no ``json_mode``, no ``**kwargs``."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def complete_json(self, messages, max_tokens=4000, temperature=0.0):
        self.calls.append({"max_tokens": max_tokens, "system": messages[0]["content"], "user": json.loads(messages[1]["content"])})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_call_supported_drops_only_what_the_callee_cannot_take() -> None:
    def strict(a, *, max_tokens=1):
        return (a, max_tokens)

    def loose(a, **kw):
        return (a, kw)

    assert call_supported(strict, 1, max_tokens=5, sample_key="k", json_mode=True) == (1, 5)
    assert call_supported(loose, 1, max_tokens=5, sample_key="k") == (1, {"max_tokens": 5, "sample_key": "k"})


def test_a_state_retry_reaches_a_strict_client(tmp_path: Path) -> None:
    """The trial's failure: attempt 2 adds ``sample_key``; a client without ``**kwargs`` raised TypeError."""
    client = Strict([{"requirements": [{"text": "x() returns 2.", "kind": "behaviour", "tests": F2P}]}])
    out = asyncio.run(lane(tmp_path).handlers()["spec"](ctx(client, attempt=2)))
    assert [r["text"] for r in out.outputs["spec"]["requirements"]] == ["x() returns 2."]
    assert client.calls[0]["max_tokens"] == SPEC_MAX_TOKENS >= 16000


def test_a_truncated_spec_gets_a_compact_retry_with_numbered_tests(tmp_path: Path) -> None:
    cut = LLMOutputError("output is not valid JSON (cut off at max_tokens: shorten the output): {\"requirements\": [")
    client = Strict([cut, {"requirements": [{"text": "x() returns 2.", "kind": "behaviour", "tests": ["T1"]},
                                            {"text": "y() returns 3.", "kind": "behaviour", "tests": ["T2"]}]}])
    out = asyncio.run(lane(tmp_path).handlers()["spec"](ctx(client, attempt=1)))
    assert out.outputs["spec"]["covers"] == {"r1": [F2P[0]], "r2": [F2P[1]]}
    second = client.calls[1]
    assert "compact JSON" in second["system"] and sorted(second["user"]["new_tests"]) == ["T1", "T2"]


def test_an_empty_completion_at_the_cap_also_counts_as_truncated(tmp_path: Path) -> None:
    assert truncated(LLMOutputError("empty completion (finish=length)")) and not truncated(LLMOutputError("realism.ok must be a boolean"))
    client = Strict([LLMOutputError("empty completion (finish=length)"), {"requirements": [{"text": "x and y.", "tests": ["T1", "T2"]}]}])
    out = asyncio.run(lane(tmp_path).handlers()["spec"](ctx(client, attempt=1)))
    assert out.outputs["spec"]["covers"] == {"r1": F2P}


def test_other_spec_errors_are_not_retried_here(tmp_path: Path) -> None:
    client = Strict([LLMOutputError("output is not valid JSON: nope")])
    with pytest.raises(LLMOutputError):
        asyncio.run(lane(tmp_path).handlers()["spec"](ctx(client, attempt=1)))
    assert len(client.calls) == 1
