"""m13 E (audit 03 G12): AWM-1K tasks graded by AWM's own deterministic code verifiers (in AWM's sandboxed runner)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
from crucible.core.schema.actions import Done, ToolCall
from crucible.interop.openenv.awm import DEFAULT_CACHE

pytestmark = pytest.mark.skipif(not (DEFAULT_CACHE / "gen_verifier.pure_code.jsonl").exists(), reason="needs the AWM-1K cache with code verifiers")

GOAL = ("Final-year mechanical engineering student passionate about sustainable product design and renewable energy systems, seeking 2026 graduate "
        "roles in R&D or product development.")


async def _run(actions):
    from crucible.interop.openenv.awm_task import AwmTask

    t = AwmTask("social_network_8", 0)
    d = Path(tempfile.mkdtemp())
    w = await t.open(d / "w")
    try:
        await run_episode(w, {"agent": ScriptedPolicy("agent", list(actions))}, limits=EpisodeLimits(max_steps=10), reset=False)
        return await t.grade(w, d / "g")
    finally:
        await w.close()


async def test_awm_verifier_scores_the_episode_from_db_state():
    done = [Done(reason="Updated.")]
    ok = await _run([ToolCall(tool_call_id="1", function_name="api.update_profile_summary", arguments={"profile_summary": GOAL}), *done])
    assert ok.reward == 1.0 and ok.infra.get("status") == "ok"
    wrong = await _run([ToolCall(tool_call_id="1", function_name="api.update_profile_summary", arguments={"profile_summary": "Student."}), *done])
    assert wrong.reward == 0.0 and wrong.infra.get("status") == "ok"
    noop = await _run(done)
    assert noop.reward == 0.0  # nothing changed: AWM's verifier says "others"


def test_runner_failure_masks_instead_of_scoring(tmp_path):
    from crucible.interop.openenv.awm_task import run_verifier

    a, b = tmp_path / "a.db", tmp_path / "b.db"
    for p in (a, b):
        import sqlite3

        sqlite3.connect(p).close()
    out = run_verifier("def verify_task_completion(i, f, a=None):\n    import subprocess\n    return {'result': 'complete'}\n", a, b, None)
    assert out["execution_status"] == "error"  # not on AWM's import allowlist: an error, which the task masks (never a score)
    out = run_verifier("def verify_task_completion(i, f, a=None):\n    return {'result': 'complete'}\n", a, b, None)
    assert out["result"] == "complete" and out["execution_status"] == "success"
