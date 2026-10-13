"""In-place Harbor tasks (m13 audit 03, G1–G4): the agent acts in the task container and Harbor's own Verifier grades
that container. Oracle → native reward, no-op → floor (they differed only through the in-place path), a policy acting
through the World's terminal tools solves the task, multi-key rewards need a rule, a missing reward masks, and a
verifier phase network the environment cannot apply masks instead of scoring 0. Needs a local Docker daemon."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

F = Path(__file__).resolve().parents[1] / "fixtures" / "harbor_inplace"
pytestmark = pytest.mark.docker


def _docker() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0


@pytest.fixture(autouse=True)
def _need_docker():
    if not _docker():
        pytest.skip("no Docker daemon")


async def _grade(task, tmp_path, *, oracle=False, actions=None):
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode

    w = await task.open(tmp_path / "w")
    try:
        if oracle:
            await task.oracle(w, tmp_path / "o")
        if actions:
            await run_episode(w, {"agent": ScriptedPolicy("agent", actions)}, limits=EpisodeLimits(max_steps=10), reset=False)
        return await task.grade(w, tmp_path / "g")
    finally:
        await w.close()


async def test_oracle_noop_and_policy(tmp_path):
    from crucible.core.schema.actions import ToolCall
    from crucible.interop.harbor.inplace import HarborInPlaceTask

    t = HarborInPlaceTask(F / "answer")
    rep = await _grade(t, tmp_path / "a", oracle=True)
    assert rep.reward == 1.0 and rep.infra["status"] == "ok"
    rep = await _grade(t, tmp_path / "b")
    assert rep.reward == 0.0 and rep.infra["status"] == "ok"  # no-op: the floor, not the oracle's grade
    acts = [ToolCall(tool_call_id="a1", function_name="term.write_file", arguments={"path": "answer.txt", "content": "42\n"}),
            ToolCall(tool_call_id="a2", function_name="term.exec", arguments={"command": "cat answer.txt"})]
    rep = await _grade(t, tmp_path / "c", actions=acts)
    assert rep.reward == 1.0  # relative path resolved in the task workdir (/app)


async def test_multi_key_missing_reward_and_network_phase(tmp_path):
    from crucible.interop.harbor.inplace import HarborInPlaceTask

    rep = await _grade(HarborInPlaceTask(F / "multikey"), tmp_path / "a", oracle=True)
    assert rep.infra["status"] == "masked" and rep.infra["failure_kind"] == "ambiguous_reward"
    rep = await _grade(HarborInPlaceTask(F / "multikey", reward_key="correctness"), tmp_path / "b", oracle=True)
    assert rep.reward == 1.0 and rep.meta["components"] == {"harbor/correctness": 1.0, "harbor/style": 0.5}
    rep = await _grade(HarborInPlaceTask(F / "noreward"), tmp_path / "c", oracle=True)
    # in the agent's own container a missing reward is the agent's outcome (M11 audit H1 V4: a policy could break the
    # grader to mask its 0); a grader that never writes one fails acceptance A1 (the oracle scores 0)
    assert rep.infra["status"] == "ok" and rep.reward == 0.0 and rep.meta["failure"]["kind"] == "crucible:RewardFileNotFoundError"
    rep = await _grade(HarborInPlaceTask(F / "netphase"), tmp_path / "d", oracle=True)
    # the verifier's network phase differs from the agent's: Harbor's egress control applies it (A-3); a daemon that cannot
    # would mask (check_infra), never score a silent 0 offline
    assert (rep.reward == 1.0 and rep.infra["status"] == "ok") or rep.infra["failure_kind"] == "check_infra"


def _online() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen("https://pypi.org/simple/", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


async def test_allowlist_world_is_enforced_on_docker(tmp_path):
    """A-3: allowlist worlds run on Docker through Harbor's egress-control sidecar: the allowed host is reachable, any
    other host is not (fails closed where the daemon's kernel cannot enforce it)."""
    from crucible.interop.harbor.inplace import HarborInPlaceTask

    if not _online():
        pytest.skip("no internet on this host")
    rep = await _grade(HarborInPlaceTask(F / "allowlist"), tmp_path, oracle=True)
    assert rep.reward == 1.0, rep.infra  # "open closed": pypi.org reachable, example.com blocked


def test_scalar_reward_rules():
    from crucible.interop.harbor.inplace import scalar_reward

    assert scalar_reward({"reward": 0.5, "x": 1}) == (0.5, None)
    assert scalar_reward({"acc": 0.25}) == (0.25, None)
    assert scalar_reward({"a": 1, "b": 0}, "b") == (0.0, None)
    assert scalar_reward({"a": 1, "b": 0})[0] is None and scalar_reward(None)[1] == "no_reward"
