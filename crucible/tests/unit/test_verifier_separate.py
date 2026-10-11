"""M1 item 1: the separate verifier World and submission custody (``world/verifier.py``, ``world/custody.py``).

The policy hands in ``out/`` and must not touch ``tests/``. Each attack below is one the in-place grader cannot see or
cannot survive: a symlink to a host file, a FIFO that blocks a reader, an oversize file, a tampered or added hidden
test, and a background process that rewrites the answer after the episode ends. With ``verifier.mode: separate`` the
first four are the policy's outcome (``crucible:policy_misbehavior``, scored 0, not masked) and the last one cannot
reach the grader, which reads a restored copy in a fresh World.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path

import pytest

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy
from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.run import Runner, RunSpec
from crucible.verify.runtime.plan import GradeReport
from crucible.world.custody import CustodyViolation, Limits, copy_untrusted
from crucible.world.verifier import verifier_world
from crucible.world.world import World, WorldError

HIDDEN = "assert answer == 42\n"


def _spec(seed: Path, *, mode: str = "separate", root: str = "ws", **verifier) -> WorldSpec:
    v = {"mode": mode, "artifacts": ["ws:out/**"], "protected": ["ws:tests/**"], **verifier}
    return WorldSpec.model_validate({
        "world": "t/custody",
        "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": root}},
                  "params": {"shell": True, "allow_unsafe_local_shell": True, "seed_dir": str(seed)}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}],
        "verifier": v,
    })


class CustodyTask:
    world_id = "t/custody"
    instruction = "Write 42 to out/answer.txt."

    def __init__(self, seed: Path, *, grade_delay: float = 0.0, **spec_kw):
        self.seed, self.grade_delay, self.spec_kw = seed, grade_delay, spec_kw
        self.id = f"t/custody/{spec_kw.get('mode', 'separate')}@1"
        self.opened: list[World] = []
        self.graded: list[World] = []

    async def open(self, workdir: Path) -> World:
        w = World(_spec(self.seed, **self.spec_kw), instructions={"agent": self.instruction}, store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w")
        await w.start()
        await w.reset()
        self.opened.append(w)
        return w

    async def grade(self, world: World, workdir: Path) -> GradeReport:
        self.graded.append(world)
        await asyncio.sleep(self.grade_delay)  # a slow grader: whatever the policy left running gets time to act
        p = Path(world.apps["ws"].state_params["root"]) / "out" / "answer.txt"
        ok = p.is_file() and not p.is_symlink() and p.read_text().strip() == "42"
        return GradeReport(task=self.id, tier="full", checks=[], gates={}, reward=1.0 if ok else 0.0, outcome=1.0 if ok else 0.0,
                           infra={"status": "ok"}, trusted=True, elapsed_s=0.0, meta={"root": str(p.parent.parent)})


def _call(i: int, name: str, **args) -> ToolCall:
    return ToolCall(tool_call_id=f"c{i}", function_name=f"ws.{name}", arguments=args)


GOOD = [_call(1, "write_file", path="out/answer.txt", content="42")]
LATE_WRITER = ("python3 -c \"import os,sys,time; d=os.getcwd(); os.fork() and sys.exit(); os.setsid(); time.sleep(0.6); "
               "open(d+'/out/answer.txt','w').write('7')\" >/dev/null 2>&1 & sleep 0.3")
ATTACKS = {
    "symlink": [_call(1, "shell", command="mkdir -p out && ln -s /etc/hosts out/answer.txt")],
    "fifo": [*GOOD, _call(2, "shell", command="mkfifo out/pipe")],
    "tamper": [*GOOD, _call(2, "write_file", path="tests/test_hidden.py", content="pass\n")],
    "added_test": [*GOOD, _call(2, "write_file", path="tests/conftest.py", content="import sys\n")],
    "deleted_test": [*GOOD, _call(2, "shell", command="rm tests/test_hidden.py")],
    "too_deep": [_call(1, "write_file", path="out/" + "/".join("d" * 40) + "/x", content="1"), *GOOD],
}


@pytest.fixture
def seed(tmp_path):
    d = tmp_path / "seed"
    (d / "tests").mkdir(parents=True)
    (d / "tests" / "test_hidden.py").write_text(HIDDEN)
    return d


def _ref(actions):
    return PolicyRef(id="scripted:p", make=lambda a: {"agent": ScriptedPolicy("agent", list(actions))})


async def _record(tmp_path, task, actions, rid="r"):
    await Runner(tmp_path / "runs").run(RunSpec(run_id=rid, tasks=[task], policies=[_ref(actions)],
                                                harnesses=[NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=20))], k=1))
    adir = next((tmp_path / "runs" / rid / "samples").glob("*/a1"))
    return json.loads((adir / "record.json").read_text()), adir


async def test_clean_handoff_grades_in_a_fresh_world(tmp_path, seed):
    task = CustodyTask(seed)
    rec, adir = await _record(tmp_path, task, GOOD)
    assert rec["reward"] == 1.0 and rec["failure"] is None
    (agent,), (graded,) = task.opened, task.graded
    assert graded is not agent and agent.closed and graded.closed  # the policy's World was closed before grading began
    assert graded.apps["ws"].state_params["root"] != agent.apps["ws"].state_params["root"]
    assert graded.log and graded.log[-1].type == agent.log[-1].type  # the trace crossed over (read-only copy)
    h = json.loads((adir / "verifier" / "handoff.json").read_text())
    assert h["checks"] == ["artifacts", "protected"] and h["artifacts"]["entries"][0]["files"] == 1
    got = adir / "artifacts" / "ws" / "out" / "answer.txt"
    assert got.read_text() == "42" and stat.S_IMODE(got.stat().st_mode) == 0o600


@pytest.mark.parametrize("attack", sorted(ATTACKS))
async def test_custody_violation_is_policy_misbehavior(tmp_path, seed, attack):
    task = CustodyTask(seed)
    rec, adir = await _record(tmp_path, task, ATTACKS[attack], rid=attack)
    f = rec["failure"]
    assert f["kind"] == "crucible:policy_misbehavior" and f["blame"] == "policy" and not rec["mask_sample"], rec
    assert rec["reward"] in (0.0, None) and not task.graded  # the grader never saw it
    assert json.loads((adir / "verifier" / "custody.json").read_text())["problems"]


async def test_oversize_artifact(tmp_path, seed):
    task = CustodyTask(seed, max_file_bytes=100)
    rec, _ = await _record(tmp_path, task, [_call(1, "write_file", path="out/answer.txt", content="4" * 200)])
    assert rec["failure"]["kind"] == "crucible:policy_misbehavior" and "too large" in rec["failure"]["reason"]


async def test_symlink_allowed_inside_tree_only(tmp_path, seed):
    inside = [*GOOD, _call(2, "shell", command="cd out && ln -s answer.txt alias")]
    rec, _ = await _record(tmp_path, CustodyTask(seed, allow_symlinks=True), inside, rid="in")
    assert rec["reward"] == 1.0
    outside = [*GOOD, _call(2, "shell", command="cd out && ln -s ../../../../etc/hosts alias")]
    rec, _ = await _record(tmp_path, CustodyTask(seed, allow_symlinks=True), outside, rid="out")
    assert rec["failure"]["kind"] == "crucible:policy_misbehavior" and "leaves the tree" in rec["failure"]["reason"]


async def test_late_writer_cannot_reach_the_separate_grader(tmp_path, seed):
    """A process the policy left behind rewrites the answer after the episode. In place it wins (local backend has no
    quiesce); in a separate verifier World it writes into a closed box the grader never reads."""
    acts = [*GOOD, _call(2, "shell", command=LATE_WRITER)]
    rec, _ = await _record(tmp_path, CustodyTask(seed, mode="in_world", grade_delay=1.0), acts, rid="inplace")
    assert rec["reward"] == 0.0  # the attack works against the in-place grader
    await asyncio.sleep(0.8)
    rec, _ = await _record(tmp_path, CustodyTask(seed, grade_delay=1.0), acts, rid="separate")
    assert rec["reward"] == 1.0


async def test_shared_absolute_root_is_refused(tmp_path, seed):
    shared = (tmp_path / "shared").resolve()
    spec = _spec(seed, root=str(shared))
    w = World(spec, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    await w.reset()
    final = await w.snapshot()
    with pytest.raises(WorldError, match="per-world state roots"):
        async with verifier_world(w, final, workdir=tmp_path / "v"):
            pass


def test_spec_validation(seed):
    with pytest.raises(ValueError, match="<app>:<glob>"):
        _spec(seed, artifacts=["out/**"])
    with pytest.raises(ValueError, match="stays inside"):
        _spec(seed, protected=["ws:../etc"])
    with pytest.raises(ValueError, match="unknown app"):
        _spec(seed, artifacts=["db:out"])
    plain = WorldSpec.model_validate({"world": "t/x", "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "ws"}}}],
                                      "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}]})
    assert "verifier" not in plain.canonical_json()  # specs without a verifier hash exactly as before


# ----------------------------------------------------------------------------------- Karotte copy (host paths)
def test_copy_untrusted_rules(tmp_path):
    src = (tmp_path / "sub").resolve()
    (src / "a" / "b").mkdir(parents=True)
    (src / "a" / "b" / "f.txt").write_text("hi")
    dest = tmp_path / "out"
    assert copy_untrusted(src, dest) == dest
    assert (dest / "a" / "b" / "f.txt").read_text() == "hi" and stat.S_IMODE((dest / "a").stat().st_mode) == 0o700
    assert copy_untrusted(src / "missing", tmp_path / "none") is None  # nothing handed in is the grader's call

    os.symlink("/etc/hosts", src / "a" / "link")
    with pytest.raises(CustodyViolation, match="symlink"):
        copy_untrusted(src, tmp_path / "o2")
    assert not (tmp_path / "o2").exists()  # a failed copy leaves nothing behind
    os.unlink(src / "a" / "link")

    os.mkfifo(src / "pipe")
    with pytest.raises(CustodyViolation, match="not a regular file"):
        copy_untrusted(src, tmp_path / "o3")
    os.unlink(src / "pipe")

    with pytest.raises(CustodyViolation, match="too large"):
        copy_untrusted(src, tmp_path / "o4", limits=Limits(max_file_bytes=1))
    with pytest.raises(CustodyViolation, match="nested too deep"):
        copy_untrusted(src, tmp_path / "o5", limits=Limits(max_depth=1))
    with pytest.raises(CustodyViolation, match="more than"):
        copy_untrusted(src, tmp_path / "o6", limits=Limits(max_entries=1))

    os.symlink(src, tmp_path / "via")  # a symlinked path component is refused too (openat walk, O_NOFOLLOW)
    with pytest.raises(CustodyViolation, match="symlink"):
        copy_untrusted(tmp_path.resolve() / "via" / "a", tmp_path / "o7")
    with pytest.raises(FileExistsError):
        copy_untrusted(src, dest)
