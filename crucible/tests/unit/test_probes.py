"""Answer-key probes (Repo2RLEnv "How not to build coding environments", SWE-bench #465): every attempt to reach
the published fix is recorded; an attempt that actually reached an answer source masks the episode; attempts the
environment blocked change nothing (probes are never scored: P12 / P20)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from crucible.actors.policy import ScriptedPolicy, run_episode
from crucible.connectors.agent_events import annotate_probes, exposed, probes_of
from crucible.core.schema.actions import Done, ToolCall
from crucible.core.schema.trace import TraceEvent
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World


@pytest.mark.parametrize(("tool", "args", "kinds"), [
    ("bash", "git log --oneline --all -5", {"git_history"}),
    ("bash", "cd /w && git -C pcl log origin/main -3", {"git_history"}),
    ("bash", "git show origin/main:tests/test_reader.py", {"git_history"}),
    ("bash", "git reflog", {"git_history"}),
    ("bash", "git branch -a", {"git_history"}),
    ("bash", "pip download pypdf==6.12.0 --no-deps", {"package_registry"}),
    ("bash", "pip install -e .", {"package_install"}),
    ("bash", "curl -sL https://github.com/py-pdf/pypdf/pull/3791.diff", {"code_host"}),
    ("bash", "cat /tests/test_hidden_ops.py", {"grader_files"}),
    ("bash", "python /solution/solve.sh", {"grader_files"}),
    ("bash", "sudo cat /root/answer", {"privilege"}),
    ("bash", "cat /proc/1/cmdline", {"process_snoop"}),
    ("WebFetch", '{"url": "https://github.com/py-pdf/pypdf/pull/3791.diff"}', {"web"}),
    ("ws.write_file", '{"path": "tests/test_reader.py", "content": "pass"}', {"test_edit"}),
    # ordinary work is not a probe
    ("bash", "git status && git diff && git log --oneline -3", set()),
    ("bash", "pytest -q tests/test_ops.py && cat tests/test_ops.py", set()),
    ("bash", "cd /workspace/tests && ls", set()),
    ("ws.write_file", '{"path": "src/kernel.cu", "content": "..."}', set()),
])
def test_probe_classification(tool, args, kinds):
    assert {p["kind"] for p in probes_of(tool, args)} == kinds


def test_exposure_needs_evidence():
    scrubbed = "7b7be3d (HEAD -> master, tag: base) people_eval (original)\n7058d69 (HEAD -> master, tag: base) kinfu_replay (original)\n"
    two_repos = "git -C people_eval log --all --oneline; git -C kinfu_replay log --all --oneline"
    assert not exposed("git_history", scrubbed, two_repos)  # one base commit per repo: the scrub held
    assert exposed("git_history", "a1b2c3d fix: clamp /W widths\n61fb890 (tag: base) original\n", "git log --all --oneline")
    assert exposed("git_history", "* master\n  remotes/origin/main\n", "git branch -a")
    assert exposed("package_registry", "Saved ./pypdf-6.12.0-py3-none-any.whl\n", "pip download pypdf==6.12.0")
    assert not exposed("package_registry", "ERROR: No matching distribution found", "pip download pypdf")


def test_annotate_links_results_and_summarizes():
    ev = [TraceEvent(seq=0, kind="action", actor="agent", type="tool_call", data={"tool": "bash", "arguments": '{"command": "git log --all --oneline"}'}),
          TraceEvent(seq=1, kind="observation", actor="system", type="tool_result", reply_to=0, text="9f9f9f9 fix it\n61fb890 base\n"),
          TraceEvent(seq=2, kind="action", actor="agent", type="tool_call", data={"tool": "bash", "arguments": '{"command": "curl https://pypi.org/simple/pypdf/"}'}),
          TraceEvent(seq=3, kind="observation", actor="system", type="tool_result", reply_to=2, text="curl: (6) Could not resolve host: pypi.org")]
    s = annotate_probes(ev)
    assert s["attempted"] == {"git_history": 1, "code_host": 1} and s["exposed"] == {"git_history": 1}
    assert s["first_strong_exposure"]["seq"] == 0 and ev[2].data["probe_ran"] is False and ev[2].data["probe_exposed"] == []


def _world(tmp: Path, history: int) -> World:
    ws = tmp / "w" / "apps" / "repo"  # the local backend roots Apps under <workdir>/apps
    ws.mkdir(parents=True)
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", "-b", "main", str(ws)], check=True)
    for i in range(history):
        (ws / "f.txt").write_text(f"v{i}\n")
        subprocess.run([*git, "-C", str(ws), "add", "-A"], check=True)
        subprocess.run([*git, "-C", str(ws), "commit", "-qm", f"c{i}"], check=True)
    spec = WorldSpec.model_validate({"world": "test/probe", "apps": [{"id": "ws", "template": "common/git_repo", "state": {"adapter": "fs_git", "params": {"root": "repo"}}, "params": {"shell": True, "allow_unsafe_local_shell": True}}],  # scripted test policy
                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}]})
    return World(spec, store=Store(tmp / "s"), workdir=tmp / "w")


@pytest.mark.parametrize(("history", "masked"), [(1, False), (3, True)])
async def test_world_masks_an_exposed_answer(tmp_path, history, masked):
    w = _world(tmp_path, history)
    await w.start()
    try:
        pol = ScriptedPolicy("agent", [ToolCall(tool_call_id="c1", function_name="ws.shell", arguments={"command": "git log --all --oneline"}), Done()])
        res = await run_episode(w, {"agent": pol}, max_steps=5)
    finally:
        await w.close()
    probes = res.trace.outcome.details["probes"]
    assert probes["attempted"] == {"git_history": 1} and probes["ran"] == {"git_history": 1}
    assert res.masked is masked and (res.stop_condition == "crucible:answer_exposed") is masked
    assert not any(e.data.get("probes") for e in w.log)  # the live log is not modified; the exported trace is


def test_egress_allowlist_matching():
    from crucible.world.guard.guard import ANSWER_SOURCES, allowed

    assert allowed("api.anthropic.com", ["api.anthropic.com"]) and allowed("x.openrouter.ai", ["*.openrouter.ai"])
    assert allowed("openrouter.ai", ["*.openrouter.ai"]) and not allowed("evilopenrouter.ai", ["*.openrouter.ai"])
    assert not any(allowed(h, ["api.anthropic.com", "*.openrouter.ai"]) for h in ANSWER_SOURCES)  # the model lane never opens a code host


async def test_peer_policies_share_a_workspace_and_keep_their_credit(tmp_path):
    """Swarm-style world (PI "On the nature of the swarm"): two policy agents work as peers in one shared workspace and
    talk directly (no orchestrator); a later signal about a file grades the step of the agent that wrote it."""
    from crucible.core.schema.actions import Message
    from crucible.core.schema.signal import Signal, SignalTarget
    from crucible.signals.attribute import attribute

    spec = WorldSpec.model_validate({"world": "test/peers", "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "ws"}}}],
                                     "actors": [{"id": "frontend", "kind": "policy", "tools": ["ws.*"]}, {"id": "backend", "kind": "policy", "tools": ["ws.*"]}]})
    w = World(spec, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        fe = ScriptedPolicy("frontend", [ToolCall(tool_call_id="f1", function_name="ws.write_file", arguments={"path": "ui.js", "content": "render()"}),
                                         Message(to="backend", text="ui.js calls GET /api/items"), Done()])
        be = ScriptedPolicy("backend", [ToolCall(tool_call_id="b1", function_name="ws.write_file", arguments={"path": "api.py", "content": "def items(): ..."}), Done()])
        res = await run_episode(w, {"frontend": fe, "backend": be}, max_steps=10)
    finally:
        await w.close()
    t = res.trace
    assert res.stop_condition == "all_done" and not res.masked
    writes = {e.data["writes"][0]: e.actor for e in t.events if e.data.get("writes")}
    assert writes == {"file:ui.js": "frontend", "file:api.py": "backend"}
    review = Signal(id="s1", type="correction", polarity=-1, source_event=len(t.events) - 1, target=SignalTarget(object="file:api.py"))
    graded = attribute(t, [review])[0].grades
    assert [t.events[g].actor for g in graded] == ["backend"]
