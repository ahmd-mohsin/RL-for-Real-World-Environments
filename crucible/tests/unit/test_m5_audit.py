"""M5 audit (docs/audit/2026-10-12/M05-workflow.md): workflow segmentation, replay, mining, stages and tracking. Ids:
W = the M5 defect study, S = the framework sweeps."""

from __future__ import annotations

import asyncio

import pytest

from crucible.core.schema.milestone import Expectation, Milestone, MilestoneCheck
from crucible.core.schema.state import Query
from crucible.core.schema.trace import Trace, TraceEvent, TraceSource


def shell(cmd: str, base: dict[str, str] | None = None):
    from crucible.workflow.filestate import FileState
    from crucible.workflow.shellops import apply_shell

    base = base if base is not None else {"src/a.py": "x = 1\n", "src/b.py": "y\n"}
    fs = FileState(base=base.get)
    eff = apply_shell(fs, cmd, cwd="/repo", path_of=lambda p: p[len("/repo/"):] if p.startswith("/repo/") else None)
    return fs, eff


@pytest.mark.parametrize("cmd", ["sed -Ei 's/x/z/' src/a.py", "sed --in-place=.bak 's/x/z/' src/a.py", "perl -pi -e 's/x/z/' src/a.py",
                                 "echo hi | tee src/new.py", "python3.11 -c \"open('src/a.py','w').write('q')\"", "dd if=/dev/zero of=src/a.py",
                                 "truncate -s 0 src/a.py", "printf '%s\\n' hello > src/p.txt", "cat > src/x.py <<EOF\nv = $HOME\nEOF"])
def test_W6_writes_are_applied_or_reported(cmd):
    _, eff = shell(cmd)
    assert eff.unreplayable and not eff.applied  # was: nothing applied and nothing reported


def test_W6_redirects_subshells_and_comments():
    fs, _eff = shell("mv src/a.py src/c.py 2>/dev/null")
    assert fs.files["src/c.py"] == "x = 1\n" and fs.files["src/a.py"] is None
    fs, _ = shell("(cd src && rm a.py); rm b.py")
    assert fs.files["src/a.py"] is None and fs.files.get("b.py") is None and fs.read("src/b.py") == "y\n"  # the subshell's cd did not leak
    fs, _ = shell("echo a#b > src/h.txt")
    assert fs.files["src/h.txt"] == "a#b\n"


def test_W7_no_guessed_effects():
    fs, eff = shell("git restore --staged src/a.py")
    assert "src/a.py" not in fs.touched  # was reset to base
    fs, _ = shell("mkdir -p pkg && mv src/a.py pkg")
    assert fs.files.get("pkg/a.py") == "x = 1\n" and "pkg" not in fs.files  # was a *file* named pkg
    fs, eff = shell("false && rm src/a.py")
    assert fs.read("src/a.py") == "x = 1\n" and eff.unreplayable  # deleted
    fs, _ = shell("cat > src/y.py <<'EOF'\nv = $HOME\nEOF")
    assert fs.files["src/y.py"] == "v = $HOME\n"  # a quoted tag is verbatim


def test_W4_W14_W23_segmentation():
    from crucible.workflow.steps import segment

    pr = Trace(trace_id="t", family="coding/pull_request", source=TraceSource(connector="x", ref="t"), events=[
        TraceEvent(seq=0, kind="action", type="commit", time="2024-01-01T10:30:00+02:00", data={"sha": "c1"}),
        TraceEvent(seq=1, kind="message", type="review", time="2024-01-01T09:00:00Z"),
        TraceEvent(seq=2, kind="action", type="commit", time="2024-01-01T09:30:00Z", data={"sha": "c2"})])
    assert [s.data.get("sha") or s.kind for s in segment(pr)] == ["c1", "review", "c2"]  # sorted by the time string
    ag = Trace(trace_id="t", source=TraceSource(connector="x", ref="t"), events=[
        TraceEvent(seq=0, kind="action", type="tool_call", actor="main", data={"tool": "bash", "arguments": {"command": "cat src/a.py"}}),
        TraceEvent(seq=1, kind="action", type="tool_call", actor="main", data={"tool": "bash", "arguments": {"command": "echo 'x = 2' | tee src/a.py"},
                                                                          "writes": ["file:src/a.py"]}),
        TraceEvent(seq=2, kind="message", type="agent_message", actor="subagent", text="done"),
        TraceEvent(seq=3, kind="action", type="tool_call", actor="main", data={"tool": "Edit", "arguments": {"file_path": "b.py"}})])
    st = segment(ag)
    assert [s.kind for s in st if s.kind != "message"] == ["investigate", "edit", "edit"]  # the tee write was merged into an investigation
    assert any(s.actor == "subagent" and s.seqs == [2] for s in st)  # folded into main's step


def test_W8_W20_replay_skips_refused_edits_and_excuses_only_run_output():
    from crucible.workflow.replay import EditReplayer

    ev = [TraceEvent(seq=0, kind="action", type="tool_call", data={"tool": "Write", "arguments": {"file_path": "/a.py", "content": "x = 99"}}),
          TraceEvent(seq=1, kind="observation", type="tool_result", reply_to=0, data={"ok": False, "is_error": True})]
    tr = Trace(trace_id="t", source=TraceSource(connector="x", ref="t"), events=ev)
    rr = EditReplayer({"a.py": "x = 1\n"}.get, "", path_of=lambda p: p.lstrip("/")).replay(tr)
    assert rr.ok and rr.meta["skipped_failed"] == [0]  # quarantined: the refused Write was applied
    ev = [TraceEvent(seq=0, kind="action", type="tool_call", data={"tool": "bash", "arguments": {"command": "python tools/codegen.py"}})]
    tr = Trace(trace_id="t", source=TraceSource(connector="x", ref="t"), events=ev)
    patch = "--- /dev/null\n+++ b/gen_out.py\n@@ -0,0 +1 @@\n+x = 1\n"
    rr = EditReplayer({}.get, patch, path_of=lambda p: p.lstrip("/")).replay(tr)
    assert not rr.meta["ok_agent_state"]  # a source file no call wrote was excused as a byproduct


def test_W9_renames_binaries_and_unprefixed_diffs():
    from crucible.workflow.filestate import FileState

    fs = FileState(base={"a.py": "x\n"}.get)
    fs.apply_patch("diff --git a/a.py b/c.py\nsimilarity index 100%\nrename from a.py\nrename to c.py\n")
    assert fs.files == {"a.py": None, "c.py": "x\n"}  # ignored
    fs = FileState(base={"a.py": "x\n"}.get)
    fs.apply_patch("diff --git a/i.png b/i.png\nBinary files /dev/null and b/i.png differ\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n", skip_binary=True)
    assert fs.files["a.py"] == "y\n" and fs.binary == ["i.png"]  # one image aborted every file
    fs = FileState(base={"src/m.py": "a\nb\n"}.get)
    fs.apply_patch("--- src/m.py\n+++ src/m.py\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n")
    assert fs.files["src/m.py"] == "a\nc\n"  # EditError: no a/ b/ prefix


def test_W5_W10_W11_mining():
    from crucible.workflow.filestate import FileState
    from crucible.workflow.milestones.evaluate import FileStateView, evaluate
    from crucible.workflow.milestones.mine import from_diff

    base = {"k.cu": "__global__ void k(float* dst)\n{\n    *dst = tex1Dfetch(t, 0);\n}\n"}
    ids = [c.milestone.id for c in from_diff("--- a/k.cu\n+++ b/k.cu\n@@ -1,4 +1,4 @@\n __global__ void k(float* dst)\n {\n-    *dst = tex1Dfetch(t, 0);\n+    *dst = __ldg(src);\n }\n", base.get)]
    assert any("tex1Dfetch" in i for i in ids) and any("__ldg" in i for i in ids)  # the deref line was skipped as a comment
    log = "--- a/CHANGELOG.rst\n+++ b/CHANGELOG.rst\n@@ -1 +1,2 @@\n x\n+Fix tz. Thanks to Marcin Sulikowski.\n"
    assert from_diff(log, {"CHANGELOG.rst": "x\n"}.get) == []  # a contributor's name became a milestone
    src = "int helper(int x);\n\nint helper(int x)\n{\n    return x + 1;\n}\n"
    patch = "--- a/a.c\n+++ b/a.c\n@@ -3,4 +3,4 @@\n int helper(int x)\n {\n-    return x + 1;\n+    return x + 2;\n }\n"
    ms = [c.milestone for c in from_diff(patch, {"a.c": src}.get) if "located" in c.milestone.id]
    end = FileState(base={"a.c": src}.get)
    end.apply_patch(patch)
    assert asyncio.run(evaluate(ms, FileStateView(end)))[ms[0].id] == 1.0  # bound to the forward declaration: 0 at the expert's own end


def test_W1_mined_milestones_measure_the_same_on_a_live_world(tmp_path):
    import os
    import subprocess

    from crucible.core.store import Store
    from crucible.workflow.filestate import FileState
    from crucible.workflow.milestones.evaluate import FileStateView, evaluate
    from crucible.workflow.milestones.mine import from_diff
    from crucible.world.exec import LocalExec
    from crucible.world.state.fs import FsGitAdapter

    base = {"k.cu": "void k(float* dst)\n{\n    dst[0] = tex1Dfetch(t, 0);\n}\n", "other.cu": "void o() { tex1Dfetch(t, 1); }\n"}
    root = tmp_path / "r"
    root.mkdir()
    for k, v in base.items():
        (root / k).write_text(v)
    env = {**os.environ, "GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@x", "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": "a@x"}
    for c in (["init", "-q"], ["add", "."], ["commit", "-qm", "b"], ["tag", "base"]):
        subprocess.run(["git", *c], cwd=root, env=env, check=True)
    patch = "--- a/k.cu\n+++ b/k.cu\n@@ -1,4 +1,4 @@\n void k(float* dst)\n {\n-    dst[0] = tex1Dfetch(t, 0);\n+    dst[0] = __ldg(src);\n }\n"
    ms = [c.milestone for c in from_diff(patch, base.get)]
    fs = FileState(base=base.get)
    fs.apply_patch(patch)
    (root / "k.cu").write_text(fs.files["k.cu"])
    ad = FsGitAdapter("repo", LocalExec(), {"root": str(root)}, Store(tmp_path / "s"))

    class Live:
        async def query(self, app, q):
            return await ad.query(q)

    offline, live = asyncio.run(evaluate(ms, FileStateView(fs))), asyncio.run(evaluate(ms, Live()))
    assert offline == live and set(live.values()) == {1.0}  # live: None (TypeError on `paths`, no block_changed)


def test_W12_signal_milestones_check_the_flagged_block():
    from crucible.core.schema.signal import Signal, SignalTarget
    from crucible.workflow.milestones.mine import from_signals

    s = Signal(id="s1", type="correction", polarity=-1, target=SignalTarget(object="file:a.c", span=[5, 5]))
    [c] = from_signals([s], {"a.c": "int f(int x)\n{\n    int y = 0;\n\n    return x;\n}\n"}.get)
    assert c.milestone.check.query.op == "block_changed"  # exists(path): true at the base


def test_W13_world_app_writes_store_the_written_value():
    from crucible.core.agent_events import app_writes
    from crucible.workflow.logstate import states

    args = {"ticket_id": "T7", "status": "resolved"}
    tr = Trace(trace_id="t", source=TraceSource(connector="x", ref="T7"), events=[
        TraceEvent(seq=0, kind="action", type="tool_call", data={"function": "tickets.update_ticket", "arguments": args, "writes": app_writes("tickets.update_ticket", args)})])
    assert states(tr)[-1][1].objects == {"ticket:T7.status": "resolved"}  # 'tool_call'


def _m(mid, weight=1.0, requires=()):
    return Milestone(id=mid, weight=weight, requires=list(requires), check=MilestoneCheck(app="log", query=Query(op="exists", args={"object": mid}), expect=Expectation(kind="truthy")))


def test_W15_graphs_are_validated():
    from crucible.workflow.graph import WorkflowGraph
    from crucible.workflow.stages import StageModel

    with pytest.raises(ValueError, match="weight"):
        WorkflowGraph.build("f", [_m("p", weight=-1)], StageModel(stages=[]))
    with pytest.raises(ValueError, match="unknown"):
        WorkflowGraph.build("f", [_m("r", requires=["dropped"])], StageModel(stages=[]))


def test_W16_W17_W18_stages_and_matching():
    from crucible.workflow.milestones.evaluate import PhiSeries
    from crucible.workflow.milestones.match import match
    from crucible.workflow.stages import induce, name_stages

    x, y = _m("x", requires=["y"]), _m("y")
    ach = [{"x": 0.1, "y": 0.9}] * 5
    a = induce([x, y], ach)
    b = induce([y, x], ach)
    assert [s.milestones for s in a.stages] == [s.milestones for s in b.stages] and a.conflicts == [("y", "x")]  # order-dependent
    mb, ma = _m("b", requires=["a"]), _m("a")
    mt = match([mb, ma], PhiSeries(milestones=["a", "b"], steps=[0], values=[{"a": 1.0, "b": 1.0}]))
    assert mt.out_of_order == {} and mt.achieved == {"a": 0, "b": 0}
    ms = [_m(f"diff/added/a.py:t{i}") for i in range(4)]
    model = name_stages(induce(ms, [{f"diff/added/a.py:t{i}": t for i, t in enumerate((0.1, 0.2, 0.7, 0.8))}] * 4))
    names = [s.name for s in model.stages]
    assert len(names) == len(set(names))


def test_W19_tracker_starts_from_the_base_state():
    from crucible.workflow.graph import WorkflowGraph
    from crucible.workflow.logstate import LogStateView
    from crucible.workflow.stages import StageModel
    from crucible.workflow.tracker import WorkflowTracker

    g = WorkflowGraph.build("f", [_m("a"), _m("b")], StageModel(stages=[]))

    async def go():
        tr = WorkflowTracker(g)
        await tr.start(LogStateView(objects={"a": 1}, case="c"))
        return await tr.observe(None, LogStateView(objects={"a": 1}, case="c"))

    st = asyncio.run(go())
    assert st.delta_phi == 0.0 and st.gained == []  # 'a' was credited to the first step


def test_S_from_ci_added_tests_skips_flips_and_quotes_ids():
    from crucible.workflow.milestones.mine import from_ci

    runs = [(0, {"t_old": "fail", "t_flip": "fail"}), (1, {"t_old": "pass", "t_flip": "pass", "t_new": "pass", "t'$(id)": "skip"}),
            (2, {"t_old": "pass", "t_flip": "fail", "t'$(id)": "pass"}), (3, {"t_flip": "pass", "t_new": "pass", "t_old": "pass", "t'$(id)": "pass"})]
    got = {c.evidence["test"]: c for c in from_ci(runs)}
    assert set(got) == {"t_old", "t_new", "t'$(id)"}  # t_new (added by the change) and the skip -> pass were missed; the flip counted
    assert "'t'\"'\"'$(id)'" in got["t'$(id)"].milestone.check.query.args["cmd"]  # repr() left $(id) to the shell


def test_S_replay_fails_closed_on_stale_files_and_scopes_submitted_patches():
    from crucible.workflow.milestones.evaluate import FileStateView, evaluate
    from crucible.workflow.milestones.mine import from_diff
    from crucible.workflow.replay import EditReplayer

    patch = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-y = old_helper()\n+y = new_helper()\n"
    base = {"a.py": "y = old_helper()\n"}
    ev = [TraceEvent(seq=0, kind="action", type="tool_call", data={"tool": "bash", "arguments": {"command": "sed -i 's/old/new/' a.py"}}),
          TraceEvent(seq=1, kind="action", type="tool_call", data={"tool": "Write", "arguments": {"file_path": "/a.py", "content": "y = new_helper()\n"}})]
    tr = Trace(trace_id="t", source=TraceSource(connector="x", ref="t"), events=ev)
    rp = EditReplayer(base.get, patch, path_of=lambda p: p.lstrip("/"))
    states, _ = rp.states(tr)
    ms = [c.milestone for c in from_diff(patch, base.get)]
    first = asyncio.run(evaluate(ms, FileStateView(states[0][1])))
    assert set(first.values()) == {None}  # 0.0: measured on the base content the unapplied sed left behind
    assert set(asyncio.run(evaluate(ms, FileStateView(states[-1][1]))).values()) == {1.0}
    ev2 = [TraceEvent(seq=0, kind="action", type="tool_call", data={"tool": "Write", "arguments": {"file_path": "/scratch.py", "content": "x"}}),
           *[e.model_copy(update={"seq": e.seq + 1}) for e in ev[1:]]]
    tr2 = Trace(trace_id="t", source=TraceSource(connector="x", ref="t"), events=ev2)
    assert not EditReplayer(base.get, patch, path_of=lambda p: p.lstrip("/")).replay(tr2).ok
    sub = EditReplayer(base.get, patch, path_of=lambda p: p.lstrip("/"), scope="submitted").replay(tr2)
    assert sub.ok and sub.meta["unsubmitted"] == ["scratch.py"]  # a mini-swe-agent scratch file quarantined a correct trace


def test_S_shell_writes_parse_quotes_and_steps_carry_keys():
    from crucible.core.agent_events import shell_writes
    from crucible.workflow.steps import segment

    assert shell_writes('python -c "a = 1; print(x > 0)"') == []  # wrote files named '=' and '0)'
    assert shell_writes("pytest 2>err.log && cp a b 2>/dev/null") == ["err.log", "b"]
    ag = Trace(trace_id="t", source=TraceSource(connector="x", ref="t"), events=[
        TraceEvent(seq=0, kind="action", type="tool_call", state_ref="world:abc", data={"tool": "Edit", "arguments": {}, "call_id": "c1"}),
        TraceEvent(seq=1, kind="action", type="tool_call", data={"tool": "bash", "arguments": {"command": "ls"}, "call_id": "c2"}),
        TraceEvent(seq=2, kind="action", type="tool_call", data={"tool": "bash", "arguments": {"command": "cat a"}, "call_id": "c3"})])
    st = segment(ag)
    assert st[0].state_ref == "world:abc" and st[1].data["call_ids"] == ["c2", "c3"]  # None; one call lost its credit row
