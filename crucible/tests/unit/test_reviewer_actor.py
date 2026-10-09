"""Coding pack: the grounded reviewer as a live actor in a git world — it raises the recorded comment
while the defect is in the code and goes quiet once the code is fixed (no LLM)."""

from __future__ import annotations

from pathlib import Path

from crucible_pack_coding.reviewer import ReviewerActor

from crucible.core.schema.actions import Message, ToolCall
from crucible.core.schema.signal import SignalTarget
from crucible.core.schema.trace import Record
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.exec import LocalExec
from crucible.world.world import World

KERNEL = """__global__ void gather(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = __ldg(&in[i + 1]);
}
"""


def test_record_anchor_shape():
    rec = _record()
    assert rec.meta["anchor"].endswith("out[i] = __ldg(&in[i + 1]);")


def _record() -> Record:
    return Record(
        id="pcl#6012/c1", source="pcl#6012", author="maintainer", role="reviewer", time="2026-01-10T00:00:00Z",
        text="tex1Dfetch returned 0 out of range; this __ldg reads past the end when i == n - 1. Guard the index.",
        target=SignalTarget(app="repo", object="gpu/gather.cu", span=[3, 3]),
        meta={"anchor": "int i = blockIdx.x * blockDim.x + threadIdx.x;\nout[i] = __ldg(&in[i + 1]);"},
    )


async def test_reviewer_actor_replays_recorded_feedback(tmp_path: Path):
    seed = tmp_path / "seed" / "gpu"
    seed.mkdir(parents=True)
    (seed / "gather.cu").write_text(KERNEL)
    spec = WorldSpec.model_validate({
        "world": "test/reviewer",
        "apps": [{"id": "repo", "template": "coding/git_repo", "state": {"adapter": "fs_git", "params": {"root": "repo"}},
                  "params": {"seed_dir": str(tmp_path / "seed"), "init_git": True}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*"]},
                   {"id": "reviewer", "kind": "simulated", "tools": ["repo.read_file", "repo.git_diff"]}],
        "clock": {"mode": "step", "t0": "2026-03-01T09:00:00Z"},
    })
    w = World(spec, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    w.attach(ReviewerActor("reviewer", [_record()], repo_app="repo"))
    await w.start()
    try:
        await w.reset()
        r = await w.step("agent", Message(to="reviewer", text="Please review my port."))
        reply = r.observation.inbox[0]
        assert reply.text.startswith("gpu/gather.cu:3:") and "Guard the index" in reply.text and reply.citations == []
        assert any(e.data.get("citations") == ["pcl#6012/c1"] for e in w.log if e.actor == "reviewer")
        assert w.clock.now_iso() >= "2026-03-01T11:00:00Z"  # reviews take virtual time (2h latency)
        await w.step("agent", ToolCall(tool_call_id="1", function_name="repo.edit_file", arguments={
            "path": "gpu/gather.cu", "old_string": "    out[i] = __ldg(&in[i + 1]);", "new_string": "    if (i + 1 < n) out[i] = __ldg(&in[i + 1]);"}))
        r = await w.step("agent", Message(to="reviewer", text="Fixed, please re-review."))
        assert r.observation.inbox[0].text == "I have no further comments based on my notes." and not r.observation.inbox[0].citations
        log = (await LocalExec().run("git log --oneline", cwd=str(w.workdir / "apps" / "repo"))).stdout
        assert len(log.strip().splitlines()) == 1  # the reviewer only read; it never touched history
    finally:
        await w.close()
