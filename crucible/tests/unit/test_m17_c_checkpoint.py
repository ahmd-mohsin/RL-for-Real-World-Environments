"""M17 study 08 §6: the checkpoint-Φ sidecar, driven directly (no World): snapshot triggers (commit, every K tool calls,
compaction boundary, end), the cap and its thinning, Φ with prerequisites, negative ΔΦ on a loss, unmeasured checkpoints,
the progress rows (close / spread) and the validity-gate weights."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from crucible.core.schema.actions import Done, ToolCall
from crucible.factory import checkpoint as ck


@dataclass
class M:
    id: str
    requires: list[str] = field(default_factory=list)
    weight: float = 1.0


class FakeTask:
    """A MilestoneTask whose "patch" is the JSON list of passing milestones (the snapshot is injected)."""

    id = "fake/window@1"

    def __init__(self, fail_on=(), delay: float = 0.0):
        self.milestones = [M("a"), M("b", ["a"]), M("c")]
        self.requirement_ids = ["b", "c"]
        self.fail_on, self.delay, self.calls = set(fail_on), delay, []

    async def milestone_status(self, patch: str) -> dict[str, bool]:
        self.calls.append(patch)
        if self.delay:
            await asyncio.sleep(self.delay)
        if patch in self.fail_on:
            raise RuntimeError("grading container lost")
        passing = set(json.loads(patch)) if patch else set()
        ids = [m["id"] if isinstance(m, dict) else m.id for m in self.milestones]
        return {i: i in passing for i in ids}


class FakePolicy:
    def __init__(self):
        self.segments: list[dict] = []


def shell(cmd: str, i: int) -> ToolCall:
    return ToolCall(tool_call_id=f"t{i}", function_name="ws.shell", arguments={"command": cmd})


def edit(i: int) -> ToolCall:
    return ToolCall(tool_call_id=f"t{i}", function_name="ws.write_file", arguments={"path": "x.py", "content": str(i)})


async def drive(task, script, tmp_path, **cfg):
    """``script``: (action, passing milestones after it, compaction before it)."""
    state = {"patch": ""}

    async def snap(_w):
        return state["patch"]

    side = ck.CheckpointSidecar(task, tmp_path, ck.CheckpointConfig(snapshot=snap, **cfg))
    await side.on_start(None)
    pol = FakePolicy()
    for i, (act, passing, comp) in enumerate(script, 1):
        if comp:
            pol.segments.append({"index": len(pol.segments) + 1, "strategy": "summary"})
        await side.before_action(None, "agent", pol, act)
        state["patch"] = json.dumps(sorted(passing)) if passing else ""
        assert await side.on_step(None, i, "agent", act, None) is None  # nothing measured travels back toward the agent
    return side, await side.on_end(None)


def rows(tmp_path):
    return [json.loads(x) for x in (tmp_path / "progress.jsonl").read_text().splitlines()]


def cps(tmp_path):
    return [json.loads(x) for x in (tmp_path / "checkpoints.jsonl").read_text().splitlines()]


def test_is_commit():
    for c in ("git commit -m x", "git add -A && git commit -qm 'w'", "cd r; git -c user.name=a -c user.email=b commit -am x",
              "git -C repo commit --amend --no-edit", "(git commit -m a)", "GIT_AUTHOR_NAME=a git commit -m x", "ls\ngit commit -m x"):
        assert ck.is_commit(shell(c, 0)), c
    for c in ("git commit-tree abc", "echo git commit", "git log --grep commit", "mygit commit"):
        assert not ck.is_commit(shell(c, 0)), c
    assert ck.is_commit(ToolCall(tool_call_id="x", function_name="ws.write_file", arguments={"content": "git commit"})) is False


def test_phi_prerequisites_and_c1_equivalence():
    from crucible.workflow.potential import potential as c1

    ms = [M("a"), M("b", ["a"]), M("c")]
    w = ck.uniform_weights(ms)
    assert ck.potential({"b": True}, ms, w) == 0.0  # b without its prerequisite a adds nothing (stubbing, M6 R5)
    assert ck.potential({"a": True, "b": True}, ms, w) == pytest.approx(2 / 3)
    for st in ({"a": True}, {"b": True, "c": True}, {"a": True, "b": True, "c": True}, {}):
        assert ck.potential(st, ms, w) == pytest.approx(c1(ms, {k: 1.0 if st.get(k) else 0.0 for k in "abc"})[0])
    with pytest.raises(ValueError):
        ck.normalized({"zz": 1.0}, ms)
    assert ck.normalized({"a": 0.0}, ms) == {"a": 0.0, "b": 0.0, "c": 0.0}  # no milestone earned weight: Φ ≡ 0


def test_weights_from_validity_gate():
    from crucible.credit.validity import validate

    ms = [M("a"), M("b", ["a"]), M("c")]
    good = validate("a", [0.1, 0.2, 0.8, 0.9, 0.7, 0.3], [0, 0, 1, 1, 1, 0])
    bad = validate("c", [0.5] * 6, [0, 0, 1, 1, 1, 0])
    w = ck.weights_from_validity([good, bad.to_dict(), {"item": "not-a-milestone", "kept": True, "weight": 1.0}], ms)
    assert good.kept and not bad.kept and w["a"] == pytest.approx(1.0) and w["c"] == 0.0 and w["b"] == 0.0


async def test_triggers_commit_periodic_compaction_end(tmp_path):
    t = FakeTask()
    script = [(edit(1), {"a"}, False), (shell("git add -A && git commit -qm a", 2), {"a"}, False), (edit(3), {"a", "b"}, False),
              (edit(4), {"a", "b"}, False), (edit(5), {"a", "b", "c"}, True), (edit(6), {"a", "b", "c"}, False), (Done(reason="done"), {"a", "b", "c"}, False)]
    side, s = await drive(t, script, tmp_path, every=3)
    got = [(c["turn"], c["triggers"]) for c in cps(tmp_path)]
    # commit after step 2, periodic at tool calls 3 and 6, the compaction boundary before step 5 is the state after step 4
    assert got == [(2, ["commit"]), (3, ["periodic"]), (4, ["compaction"]), (6, ["periodic"]), (7, ["end"])], got
    assert s["compaction_turns"] == [4] and s["kind"] == "checkpoint" and s["phi_final"] == pytest.approx(1.0) and s["phi0"] == 0.0
    assert side.phi_series == [(2, pytest.approx(1 / 3), ["a"], []), (3, pytest.approx(2 / 3), ["b"], []), (4, pytest.approx(2 / 3), [], []),
                               (6, pytest.approx(1.0), ["c"], []), (7, pytest.approx(1.0), [], [])]
    # identical patches share a grading: steps 3/4 and 6/7 are one patch each; the start state is graded once, outside the cap
    assert sorted(t.calls) == sorted(["", '["a"]', '["a", "b"]', '["a", "b", "c"]']) and s["graded"] == 3
    r = rows(tmp_path)
    assert [x["step"] for x in r] == list(range(8)) and sum(x["delta_phi"] for x in r) == pytest.approx(s["phi_final"] - s["phi0"])
    assert [round(x["delta_phi"], 4) for x in r[1:]] == [0, round(1 / 3, 4), round(1 / 3, 4), 0, 0, round(1 / 3, 4), 0]  # on the closing step
    assert r[2]["checkpoint"] == ["commit"] and r[2]["actor"] == "agent" and r[2]["tool_call_id"] == "t2"
    assert (tmp_path / "checkpoints" / f"{cps(tmp_path)[0]['patch_sha256']}.patch").read_text() == '["a"]'


async def test_lost_milestone_is_negative_delta_and_spread(tmp_path):
    t = FakeTask()
    script = [(edit(1), {"a", "b"}, False), (edit(2), {"a", "b"}, False), (edit(3), {"a"}, False), (edit(4), {"a"}, False), (edit(5), {"a", "b"}, False),
              (edit(6), {"a", "b"}, False)]
    side, s = await drive(t, script, tmp_path / "close", every=2)
    r = rows(tmp_path / "close")
    assert r[4]["delta_phi"] == pytest.approx(-1 / 3) and r[4]["lost"] == ["b"] and s["milestones_lost"] == 1
    assert [p[3] for p in side.phi_series] == [[], ["b"], []]
    _, s2 = await drive(FakeTask(), script, tmp_path / "spread", every=2, assign="spread")
    r2 = rows(tmp_path / "spread")
    assert r2[3]["delta_phi"] == pytest.approx(-1 / 6) and r2[4]["delta_phi"] == pytest.approx(-1 / 6)  # the span (2, 4] shares it
    assert sum(x["delta_phi"] for x in r2) == pytest.approx(s2["phi_final"] - s2["phi0"]) and r2[4]["lost"] == ["b"]


async def test_prerequisites_block_stubbed_progress(tmp_path):
    script = [(edit(1), {"b"}, False), (edit(2), {"b", "c"}, False), (edit(3), {"a", "b", "c"}, False)]
    side, _s = await drive(FakeTask(), script, tmp_path, every=1)
    assert [round(p[1], 4) for p in side.phi_series] == [0.0, round(1 / 3, 4), 1.0]  # the end merges with the step-3 checkpoint
    assert side.phi_series[2][2] == ["a", "b"]  # b counts only once a holds


async def test_unmeasured_checkpoint_carries_phi_never_zero(tmp_path):
    t = FakeTask(fail_on={'["a", "b"]'})
    script = [(edit(1), {"a"}, False), (edit(2), {"a", "b"}, False), (edit(3), {"a", "b", "c"}, False)]
    _side, s = await drive(t, script, tmp_path, every=1)
    c = cps(tmp_path)
    assert c[1]["state"] == "unmeasured" and c[1]["measured"] is False and "grading container lost" in c[1]["reason"] and c[1]["milestones"] is None
    assert c[1]["phi"] == pytest.approx(1 / 3)  # carried from the last measured checkpoint, not 0
    r = rows(tmp_path)
    assert r[2]["delta_phi"] == 0.0 and r[2]["measured"] is False and r[3]["delta_phi"] == pytest.approx(2 / 3)  # the next measured point gets it
    assert s["unmeasured"] == 1 and s["phi_final"] == pytest.approx(1.0)

    async def broken(_w):
        raise OSError("docker exec failed")

    side2 = ck.CheckpointSidecar(FakeTask(), tmp_path / "b", ck.CheckpointConfig(snapshot=broken))
    await side2.on_start(None)
    await side2.on_step(None, 1, "agent", edit(1), None)
    s2 = await side2.on_end(None)
    assert s2["unmeasured"] == 1 and s2["phi_final"] == 0.0 and not s2["final_measured"] and "docker exec failed" in cps(tmp_path / "b")[0]["reason"]


async def test_bad_verdicts_are_unmeasured(tmp_path):
    class Partial(FakeTask):
        async def milestone_status(self, patch):
            return {"a": True} if patch else {m.id: False for m in self.milestones}  # no verdict for b and c

    _, s = await drive(Partial(), [(edit(1), {"a"}, False)], tmp_path, every=1)
    assert s["unmeasured"] == 1 and "no verdict" in cps(tmp_path)[0]["reason"]


async def test_cap_thins_but_keeps_end_and_commits(tmp_path):
    t = FakeTask()
    script = []
    for i in range(1, 13):
        passing = {"a"} if i < 6 else {"a", "b"}
        act = shell(f"echo {i} > f && git commit -qam s{i}", i) if i in (3, 9) else edit(i)
        script.append((act, passing | ({"c"} if i % 2 else set()) | {f"x{i}"}, False))  # x<i>: every step changes the patch
    _side, s = await drive(t, script, tmp_path, every=1, cap=5, defer=True)
    kept = [c for c in cps(tmp_path) if c["state"] != "skipped"]
    assert len({c["patch_sha256"] for c in kept}) <= 5 and len([x for x in t.calls if x]) <= 5 and s["skipped_cap"] > 0
    turns = {c["turn"]: c["triggers"] for c in kept}
    assert 12 in turns and "end" in turns[12] and 3 in turns and 9 in turns, turns
    r = rows(tmp_path)
    assert sum(x["delta_phi"] for x in r) == pytest.approx(s["phi_final"] - s["phi0"])


async def test_grading_runs_during_the_episode_and_end_waits(tmp_path):
    t = FakeTask(delay=0.01)
    state = {"patch": ""}

    async def snap(_w):
        return state["patch"]

    side = ck.CheckpointSidecar(t, tmp_path, ck.CheckpointConfig(snapshot=snap, every=1, concurrency=1))
    await side.on_start(None)
    state["patch"] = '["a"]'
    await side.on_step(None, 1, "agent", edit(1), None)
    assert not (tmp_path / "progress.jsonl").exists() and not (tmp_path / "checkpoints.jsonl").exists()  # nothing on disk mid-episode
    await asyncio.sleep(0.05)
    assert '["a"]' in t.calls  # graded in the background while the "agent" went on
    s = await side.on_end(None)
    assert s["phi_final"] == pytest.approx(1 / 3) and (tmp_path / "progress.jsonl").exists()


def test_needs_a_snapshot_source(tmp_path):
    with pytest.raises(ValueError, match="snapshot"):
        ck.CheckpointSidecar(FakeTask(), tmp_path)
    assert isinstance(FakeTask(), ck.MilestoneTask)
    make = ck.checkpoint_sidecar()

    class A:
        task = object()
        dir = tmp_path

    assert make(A()) is None  # a task that is not a MilestoneTask runs without a sidecar


async def test_mapping_milestones_task_weights_and_requirement_map(tmp_path):
    """The window lane's milestones are mappings and carry their own weights; requirements map onto milestones."""

    class DictTask(FakeTask):
        def __init__(self):
            super().__init__()
            self.milestones = [{"id": "a", "requires": []}, {"id": "b", "requires": ["a"]}, {"id": "c"}]
            self.weights = {"a": 1.0, "b": 2.0, "c": 1.0}
            self.requirement_ids = ["R1", "R2"]
            self.requirement_milestones = {"R1": ["a", "b"], "R2": ["c"]}

    _side, s = await drive(DictTask(), [(edit(1), {"a", "b"}, False)], tmp_path, every=1)
    assert s["phi_final"] == pytest.approx(0.75)
    h = s["honesty"]
    assert h["scored"] == {"R1": [0.5, 1], "R2": [0.5, 0]} and h["brier"] == pytest.approx(0.25)  # no STATUS.json: p = 0.5
