"""M17 study 08 §6 end to end: the checkpoint sidecar in the WorkspaceHarness and the RLM harness, through the M10 runner.

A shell workspace task with three milestones (one needs another) is driven by a scripted LLM agent that commits, breaks
its work, fixes it and writes STATUS.json. The rollout's ``Turn.advantage`` is ΔΦ on the call that closes each span
(negative where the milestone was lost) and flows unchanged through ``turn_credit_map`` -> ``token_credit``; hidden-test
results never reach an observation; the harness without a sidecar is unchanged. An RLM episode over the long-horizon
backlog compacts several times: its boundaries are checkpoints and its root turns carry the credit."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from crucible_pack_toy.runtime import FibTask

from crucible.actors.policy import LLMToolPolicy, ScriptedPolicy
from crucible.core.schema.actions import Done, ToolCall
from crucible.core.schema.rollout import RolloutRecord
from crucible.factory.checkpoint import CheckpointConfig, CheckpointSidecar, checkpoint_sidecar
from crucible.runtime.harness.native import PolicyRef
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.inject import FakeLLM, tool_reply
from crucible.runtime.run import Runner, RunSpec

FIB = "a, b = 0, 1\nfor _ in range(10):\n    print(a)\n    a, b = b, a + b\n"
IDS = ("ms_file_q7", "ms_output_q7", "ms_readme_q7")  # unusual ids: any leak into an observation is findable


@dataclass
class M:
    id: str
    requires: list[str] = field(default_factory=list)


def _status(patch: str) -> dict[str, bool]:
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        if patch.strip():
            (repo / "p.patch").write_text(patch)
            subprocess.run(["git", "apply", "p.patch"], cwd=repo, check=True, capture_output=True)
        fib = repo / "fib.py"
        out = subprocess.run(["python3", "fib.py"], cwd=repo, capture_output=True, text=True, timeout=30, check=False).stdout if fib.exists() else ""
        return {IDS[0]: fib.exists(), IDS[1]: out.split() == [str(x) for x in (0, 1, 1, 2, 3, 5, 8, 13, 21, 34)], IDS[2]: (repo / "README.md").exists()}


class FibWindow(FibTask):
    """FibTask as a MilestoneTask: the hidden per-milestone tests run on a pristine repo + the snapshot patch."""

    id = "toy/workspace/fib-window@1"

    def __init__(self, delay: float = 0.0):
        self.milestones = [M(IDS[0]), M(IDS[1], [IDS[0]]), M(IDS[2])]
        self.requirement_ids = [IDS[1], IDS[2]]
        self.delay, self.graded_at = delay, []

    async def milestone_status(self, patch: str) -> dict[str, bool]:
        if self.delay:
            await asyncio.sleep(self.delay)
        st = await asyncio.to_thread(_status, patch)
        self.graded_at.append(time.monotonic())
        return st


def script(final: str = "All done."):
    commit = "git add -A && git -c user.email=a@b -c user.name=a commit -qm fib"
    status = json.dumps({IDS[1]: 0.9, IDS[2]: 1.0, "bogus": 1.0})
    return [tool_reply("ws__write_file", {"path": "fib.py", "content": FIB}, 1), tool_reply("ws__shell", {"command": commit}, 2),
            tool_reply("ws__write_file", {"path": "fib.py", "content": "print(1)\n"}, 3), tool_reply("ws__write_file", {"path": "README.md", "content": "# fib\n"}, 4),
            tool_reply("ws__write_file", {"path": "fib.py", "content": FIB}, 5), tool_reply("ws__write_file", {"path": "STATUS.json", "content": status}, 6), final]


def llm(final: str = "All done.") -> PolicyRef:
    return PolicyRef(id="fake:window", make=lambda a: {"agent": LLMToolPolicy("agent", FakeLLM(script(final)))})


async def run_one(tmp_path, task, policy, harness, run_id="w", **kw):
    await Runner(tmp_path).run(RunSpec(run_id=run_id, tasks=[task], policies=[policy], harnesses=[harness], k=1, **kw))
    adir = min((tmp_path / run_id / "samples").glob("*/a*"))
    rec = json.loads((adir / "record.json").read_text())
    ro = RolloutRecord.model_validate_json((adir / "rollout.json").read_text()) if (adir / "rollout.json").exists() else None
    return adir, rec, ro


async def test_workspace_checkpoint_credit_end_to_end(tmp_path):
    from crucible.train.credit import token_credit, turn_credit_map

    task = FibWindow()
    adir, rec, ro = await run_one(tmp_path, task, llm(), WorkspaceHarness(sidecar=checkpoint_sidecar(CheckpointConfig(every=3))))
    assert rec["reward"] == 1.0 and not rec["mask_sample"]
    prog = rec["meta"]["progress"]
    assert prog["kind"] == "checkpoint" and prog["phi0"] == 0.0 and prog["phi_final"] == pytest.approx(1.0) and prog["milestones_lost"] == 1
    cps = [json.loads(x) for x in (adir / "checkpoints.jsonl").read_text().splitlines()]
    assert [(c["turn"], c["triggers"]) for c in cps] == [(2, ["commit"]), (3, ["periodic"]), (6, ["periodic"]), (7, ["end"])]
    assert cps[1]["lost"] == [IDS[1]] and cps[1]["milestones"] == {IDS[0]: True, IDS[1]: False, IDS[2]: False}
    assert all(len(c["patch_sha256"]) == 64 for c in cps) and (adir / "checkpoints" / f"{cps[0]['patch_sha256']}.patch").exists()
    # honesty: STATUS.json read from the final tree, scored only over the verified requirements
    hon = prog["honesty"]
    assert hon["brier"] == pytest.approx(0.005) and hon["ignored"] == ["bogus"] and hon["term"] == pytest.approx(-0.0005) and prog["give_up"] is None
    # Turn.advantage: ΔΦ on the call that closes each span (the commit +2/3, the break −1/3, the fix + README +2/3)
    adv = {t.call_id: t.advantage for t in ro.turns}
    assert adv["agent:c2"] == pytest.approx(2 / 3) and adv["agent:c3"] == pytest.approx(-1 / 3) and adv["agent:c6"] == pytest.approx(2 / 3)
    assert all(adv[f"agent:c{i}"] == 0.0 for i in (1, 4, 5, 7)) and sum(adv.values()) == pytest.approx(1.0)
    assert ro.progress["phi_final"] == pytest.approx(1.0)
    # the existing credit path applies unchanged: centred within the episode, bounded, none for a hacked rollout
    tok = [t.model_copy(update={"prompt_token_ids": [1], "completion_token_ids": [5, 6], "logprobs": [-0.1, -0.1], "policy_version": 0}) for t in ro.turns]
    good = ro.model_copy(update={"turns": tok})
    fail = good.model_copy(update={"rollout_id": "other", "reward": 0.0})
    hacked = good.model_copy(update={"rollout_id": "hacked", "reward": 0.0, "reward_components": {"hack/confirmed": 1.0}})
    cm = turn_credit_map([good, fail, hacked], ["g", "g", "g"], eta=1.0)
    g = {k[1]: v for k, v in cm.items() if k[0] == good.rollout_id}
    assert g["agent:c2"] > g["agent:c3"] and abs(sum(g.values())) < 1e-6  # centred: zero net mass
    assert all(v <= 0 for k, v in cm.items() if k[0] == "hacked")
    tc = {c.call_id: c for c in token_credit(good, 0.5, turn_weight=1.0)}
    assert tc["agent:c3"].advantages == [pytest.approx(0.5 + good.turns[2].advantage)] * 2 and tc["agent:c3"].loss_mask == [1, 1]


async def test_no_hidden_result_reaches_the_agent(tmp_path):
    task = FibWindow(delay=0.02)
    seen: list[str] = []
    files_mid: list[bool] = []
    graded_mid: list[bool] = []
    acts = [ToolCall(tool_call_id="w1", function_name="ws.write_file", arguments={"path": "fib.py", "content": FIB}),
            ToolCall(tool_call_id="w2", function_name="ws.shell", arguments={"command": "git add -A && git -c user.email=a@b -c user.name=a commit -qm x"}),
            *[ToolCall(tool_call_id=f"r{i}", function_name="ws.shell", arguments={"command": "ls"}) for i in range(4)],
            ToolCall(tool_call_id="w3", function_name="ws.write_file", arguments={"path": "README.md", "content": "x"})]

    def make(attempt):
        it = iter(acts)

        def step(obs):
            seen.append(obs.model_dump_json())
            files_mid.append(any((attempt.dir / f).exists() for f in ("checkpoints.jsonl", "progress.jsonl", "checkpoints")))
            if len(seen) == len(acts):
                graded_mid.append(bool(task.graded_at))  # gradings finished while the agent was still acting
            return next(it, Done(reason="done"))

        class Thinking(ScriptedPolicy):  # a model call takes time: the loop runs the background gradings meanwhile
            async def act(self, obs):
                await asyncio.sleep(0.05)
                return await super().act(obs)

        return {"agent": Thinking("agent", step)}

    adir, rec, _ = await run_one(tmp_path, task, PolicyRef(id="scripted:rec", make=make), WorkspaceHarness(sidecar=checkpoint_sidecar(CheckpointConfig(every=1))))
    assert rec["reward"] == 1.0 and rec["meta"]["progress"]["graded"] == 2  # two distinct patches (fib.py, then + README)
    assert graded_mid == [True] and len(files_mid) == len(acts) + 1 and not any(files_mid)  # graded during the episode; nothing on disk until it ended
    blob = "\n".join(seen)
    for needle in (*IDS, "phi", "checkpoint", "milestone"):
        assert needle not in blob, needle
    trace = (adir / "trace.json").read_text()
    assert "read-tree" not in trace and "GIT_INDEX_FILE" not in trace  # snapshots are not World actions


async def test_off_by_default_and_task_opt_in(tmp_path):
    adir, rec, ro = await run_one(tmp_path, FibWindow(), llm(), WorkspaceHarness(), run_id="off")
    assert rec["reward"] == 1.0 and "progress" not in rec["meta"] and not (adir / "progress.jsonl").exists() and not (adir / "checkpoints.jsonl").exists()
    assert all(t.advantage is None for t in ro.turns)

    class OptIn(FibWindow):
        def episode_sidecar(self, attempt):
            return CheckpointSidecar(self, attempt.dir, CheckpointConfig(every=0, on_commit=False))

    _, rec2, _ = await run_one(tmp_path, OptIn(), llm("GIVE_UP: the build tool is missing"), WorkspaceHarness(), run_id="opt")
    p = rec2["meta"]["progress"]
    assert p["checkpoints"] == 1 and p["give_up"] == "the build tool is missing" and rec2["reward"] == 1.0  # scored as it stands


async def test_rlm_compaction_boundaries_are_checkpoints(tmp_path):
    from crucible_pack_toy.longhorizon.task import BacklogTask
    from test_rlm_harness import CODEC, _config, scripted_engine

    from crucible.runtime.harness.rlm import RLMHarness
    from crucible.train.credit import turn_credit_map
    from crucible.train.world_env import EngineLLM

    base = BacklogTask(seed=0, n=12)
    ids = [g.id for g in base.subgoals.subgoals]

    class BacklogWindow:
        id = base.id + "#window"
        world_id = base.world_id
        instruction = base.instruction
        milestones = tuple(M(i) for i in ids)
        requirement_ids = ()

        async def open(self, workdir):
            return await base.open(workdir)

        async def grade(self, world, workdir):
            return await base.grade(world, workdir)

        async def milestone_status(self, patch):
            return {k: bool(v) for k, v in json.loads(patch).items()}

    async def snap(world):  # the World's verified state stands in for a patch
        return json.dumps(await base.subgoals.evaluate(world, ids), sort_keys=True)

    gen, seen = scripted_engine(base)
    h = RLMHarness(_config(), sidecar=checkpoint_sidecar(CheckpointConfig(every=4, snapshot=snap)))
    try:
        ref = PolicyRef(id="engine:scripted", make=lambda a: {"agent": EngineLLM(CODEC, gen)})
        adir, _rec, ro = await run_one(tmp_path, BacklogWindow(), ref, h, run_id="rlm", rollout_mode="train")
    finally:
        base.cleanup()
    assert ro is not None and not ro.mask_sample and ro.valid_for_training()[0], (ro.failure_kind, ro.stop)
    p = ro.progress
    assert len(p["compaction_turns"]) == len(ro.segments) >= 2 and seen["compact"] >= 2
    cps = [json.loads(x) for x in (adir / "checkpoints.jsonl").read_text().splitlines()]
    assert sum("compaction" in c["triggers"] for c in cps) >= 1 and cps[-1]["triggers"][-1] == "end"
    assert p["phi_final"] == pytest.approx(ro.reward)  # Φ over the tickets is the task's share reward
    root = [t for t in ro.turns if not t.node() and not t.agent and t.kind == "turn"]
    total = sum(t.advantage or 0.0 for t in root)
    assert total == pytest.approx(p["phi_final"] - p["phi0"] - p.get("unattributed_delta", 0.0), abs=1e-6) and total > 0
    assert all(t.advantage is None or t.node() for t in ro.turns if t.kind == "sub")  # sub-calls take their parent's credit in turn_credit_map
    other = ro.model_copy(update={"rollout_id": "b", "reward": 0.0})
    cm = turn_credit_map([ro, other], ["g", "g"], eta=1.0)
    assert any(v != 0 for (rid, _), v in cm.items() if rid == ro.rollout_id)
