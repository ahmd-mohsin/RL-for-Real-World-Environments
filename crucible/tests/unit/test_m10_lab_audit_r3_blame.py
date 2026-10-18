"""M10 lab audit R-3, slice A: blame pass-through, the ``__cause__`` walk and the evidence on every masked failure.

An unclassified policy exception (``actors/policy.py:586-587``, blame unknown) was rebuilt as ``failure(kind)`` by
``_masked_episode`` and the workspace harness, so it fell back to ``agent_run_error``'s harness blame, was retried and was
never counted by ``Health.unknown`` (Acceptance 3). ``boundary()`` looked at the raised exception only (NeMo-RL walks
``__cause__`` up to 8 links). Every masked ``Failure`` now carries ``evidence = {probe, control, status}``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from crucible.actors.policy import ScriptedPolicy
from crucible.core.schema.run import Failure, RunRecord
from crucible.runtime.harness.native import NativeHarness, PolicyRef, _masked_episode
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.infra import Health, StageFailure, boundary, episode_failure
from crucible.runtime.run import Runner, RunSpec


class Boom(ScriptedPolicy):
    """A policy whose own code raises something no rule classifies: run_episode records it with unknown blame."""

    async def act(self, obs):
        raise RuntimeError("policy bug")


def _boom() -> PolicyRef:
    return PolicyRef(id="boom", make=lambda a: {"agent": Boom("agent", [])})


async def _run(tmp_path, name, task, harness):
    s = await Runner(tmp_path).run(RunSpec(run_id=name, tasks=[task], policies=[_boom()], harnesses=[harness], k=1, retry_backoff_s=0.0))
    rows = [json.loads(x) for x in (tmp_path / name / "results.jsonl").read_text().splitlines() if x.strip()]
    return s, rows


# ---------------------------------------------------------------- Acceptance 3: unknown blame survives the rebuild


async def test_runtime_error_from_policy_act_is_unknown_counted_and_not_retried(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask

    s, rows = await _run(tmp_path, "boom-native", HelpdeskTask(), NativeHarness())
    assert s.attempts == 1 and len(rows) == 1, (s, rows)  # unknown blame is never retried (RetryRule's blames exclude it)
    r = rows[0]
    f = r["failure"]
    assert f["kind"] == "agent_run_error" and f["blame"] == "unknown" and r["mask_sample"] and r["reward"] is None, r
    assert "policy bug" in f["reason"]
    ev = f["evidence"]
    assert ev["control"] == "none" and ev["probe"] == "unprobed" and ev["status"] == "error", ev  # HelpdeskTask has no infra_cause
    h = Health(planned=1)
    h.add(RunRecord.model_validate(r))
    assert h.unknown == 1 and h.masked == 1


async def test_runtime_error_in_the_workspace_harness_keeps_unknown_blame(tmp_path):
    from crucible_pack_toy.runtime import FibTask

    s, rows = await _run(tmp_path, "boom-ws", FibTask(), WorkspaceHarness())
    assert s.attempts == 1 and len(rows) == 1, (s, rows)
    r = rows[0]
    f = r["failure"]
    assert f["kind"] == "agent_run_error" and f["blame"] == "unknown" and r["mask_sample"] and r["reward"] is None, r
    # the workspace harness does not probe: the Runner records the evidence as unprobed and uncontrolled
    assert f["evidence"] == {"probe": "unprobed", "control": "none", "status": "agent_run_error"}, f
    h = Health(planned=1)
    h.add(RunRecord.model_validate(r))
    assert h.unknown == 1


# ---------------------------------------------------------------- boundary(): the __cause__ walk


def _kind(stage: str, ex: BaseException) -> Failure:
    with pytest.raises(StageFailure) as ei:
        with boundary(stage):
            raise ex
    return ei.value.failure


def _from(ex: BaseException, cause: BaseException) -> BaseException:
    ex.__cause__ = cause
    return ex


def test_boundary_walks_the_cause_chain():
    class VerifierTimeoutError(Exception):  # Harbor's name, matched by NORMALIZE on the class name
        pass

    f = _kind("agent", _from(RuntimeError("wrap"), TimeoutError("t")))
    assert f.kind == "transport_timeout" and "cause of RuntimeError: wrap" in f.reason, f
    f = _kind("agent", _from(ValueError("v"), ConnectionResetError("reset by peer")))
    assert f.kind == "transport_unreachable" and "ConnectionResetError" in f.reason, f
    f = _kind("verification", _from(RuntimeError("wrap"), VerifierTimeoutError("test.sh hung")))
    assert (f.kind, f.detail, f.stage, f.who) == ("verifier_error", "VerifierTimeoutError", "verification", "task"), f
    inner = Failure(kind="crucible:node_contention", stage="verification", reason="perf rounds contaminated")
    f = _kind("verification", _from(RuntimeError("wrap"), StageFailure(inner)))
    assert f == inner  # a typed failure wrapped by a caller keeps its own classification
    # the raised exception itself still wins over its causes, as before
    f = _kind("agent", _from(TimeoutError("socket"), OSError("x")))
    assert f.kind == "transport_timeout" and "cause of" not in f.reason
    f = _kind("agent", RuntimeError("nothing to classify"))
    assert f.kind == "agent_run_error" and f.who == "unknown"


def _chain(n: int, tail: BaseException) -> BaseException:
    """``n`` links in all: ``RuntimeError('0')`` raised from ``n - 2`` more RuntimeErrors, the last raised from ``tail``."""
    head = cur = RuntimeError("0")
    for i in range(1, n - 1):
        nxt = RuntimeError(str(i))
        cur.__cause__ = nxt
        cur = nxt
    cur.__cause__ = tail
    return head


def test_boundary_cause_walk_is_bounded_and_cycle_safe():
    f = _kind("agent", _chain(8, OSError("disk")))  # the 8th link is the last one looked at
    assert f.kind == "transport_unreachable", f
    f = _kind("agent", _chain(9, OSError("disk")))  # the 9th is past _MAX_CAUSE_DEPTH
    assert f.kind == "agent_run_error" and f.who == "unknown", f
    e = RuntimeError("x")
    e.__cause__ = e
    f = _kind("agent", e)  # terminates
    assert f.kind == "agent_run_error" and f.who == "unknown", f
    a, b = RuntimeError("a"), RuntimeError("b")
    a.__cause__, b.__cause__ = b, a
    f = _kind("seed", a)
    assert f.kind == "agent_run_error" and f.who == "unknown", f


def test_cause_walk_never_blames_the_policy_for_a_wrapped_encoding_error():
    enc = UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed")
    f = _kind("agent", enc)  # the policy's own unencodable output (M14 campaign B1): scored 0, unchanged
    assert f.kind == "crucible:policy_misbehavior" and not f.masks
    enc = UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed")
    f = _kind("agent", _from(RuntimeError("backend could not write its log"), enc))
    assert f.kind == "agent_run_error" and f.who == "unknown" and f.masks, f  # an infrastructure wrapper: never the policy's 0


# ---------------------------------------------------------------- episode_failure / _masked_episode


def test_masked_episode_keeps_world_failure_fields():
    f = episode_failure({"kind": "agent_run_error", "reason": "r", "blame": "unknown"})
    assert f.who == "unknown" and f.stage == "agent" and f.masks
    f = episode_failure({"kind": "crucible:tool_crashed", "reason": "r", "blame": None, "mask": None, "retryable": True})
    assert f.who == "task" and f.retryable and f.mask is None  # None keys fall back to the kind's defaults
    f = episode_failure({"kind": "SomeHarborError"})
    assert f.who == "unknown" and f.detail == "unmapped:SomeHarborError"  # the unmapped rule still applies
    f = episode_failure({"kind": "crucible:env_container_lost", "reason": "gone", "retryable": False, "detail": "removed", "delivery": "not_sent"})
    assert (f.who, f.retryable, f.detail, f.delivery) == ("infra", False, "removed", "not_sent")
    f = episode_failure({"kind": "crucible:answer_exposed", "reason": "hit", "mask": True}, "agent", "step s1: ")
    assert f.masks and f.reason == "step s1: hit"
    f = episode_failure({"kind": "agent_run_error", "reason": "r"})
    assert f.who == "harness"  # no blame in the dict: the kind's default, as before (RLMPolicy's agent_run_error resumes)


async def test_masked_episode_records_the_probe():
    ep = SimpleNamespace(failures=[{"kind": "crucible:tool_crashed", "reason": "boom"}], stop_condition="crucible:tool_crashed")
    world = object()

    class Alive:
        async def infra_cause(self, world):
            return None

    class Dead:
        async def infra_cause(self, world):
            return Failure(kind="crucible:env_container_lost", stage="agent", reason="container gone", detail="removed")

    class Broken:
        async def infra_cause(self, world):
            raise RuntimeError("docker unreachable")

    f = await _masked_episode(Alive(), world, ep)
    assert f.kind == "crucible:tool_crashed" and f.evidence == {"probe": "alive", "control": "none", "status": "crucible:tool_crashed"}
    f = await _masked_episode(Dead(), world, ep, prefix="step s: ")
    assert f.kind == "crucible:env_container_lost" and f.reason == "step s: container gone"
    assert f.evidence == {"probe": "dead", "control": "none", "status": "removed"}
    f = await _masked_episode(Broken(), world, ep)
    assert f.kind == "crucible:tool_crashed" and f.evidence["probe"] == "unprobed"
    f = await _masked_episode(object(), world, SimpleNamespace(failures=[], stop_condition="error"))
    assert f.kind == "agent_run_error" and f.evidence == {"probe": "unprobed", "control": "none", "status": "error"}


def test_runner_fills_evidence_on_masked_records_only():
    def rec() -> RunRecord:
        return RunRecord(run_id="r", world="w", policy="p", started_at="2026-10-17T00:00:00+00:00", outcome="completed")

    r = rec()
    r.mark_failure(Failure(kind="crucible:node_contention", stage="verification", reason="contaminated", detail="check_infra"))
    Runner._evidence(r)
    assert r.mask_sample and r.reward is None and r.failure.evidence == {"probe": "unprobed", "control": "none", "status": "check_infra"}
    r = rec()
    ev = {"probe": "dead", "control": "none", "status": "removed"}
    r.mark_failure(Failure(kind="crucible:env_container_lost", stage="agent", evidence=ev))
    Runner._evidence(r)
    assert r.failure.evidence == ev  # what the harness knew is kept
    r = rec()
    r.mark_failure(Failure(kind="crucible:policy_error", stage="agent", reason="bad args"))
    Runner._evidence(r)
    assert r.failure.evidence is None and r.reward == 0.0 and not r.mask_sample  # a scored failure needs no infrastructure evidence
    r = rec()
    Runner._evidence(r)
    assert r.failure is None
