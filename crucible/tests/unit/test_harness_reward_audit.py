"""Harness and reward audit (2026-10-28, docs/research/harness-reward-audit.md): each test fails on the code before its fix.

1. A blank first summary attempt was resampled as a link of the segment's chain: the resample's prompt never extends the
   first attempt's prompt + completion, so the record saw a prefix break and train mode masked the whole rollout. The
   resample is a tree node now, and only the accepted summary is trained.
2. The Tinker sampler's client named no context window: replies were never fitted to the room left, so ``gen`` refused
   prompt + ``max_tokens`` over the context and the episode ended ``context_overflow`` with room left.
3. ``WorkspaceHarness`` ignored its limit policy (mask / penalty), though the policy is part of its id.
4. A single-step episode whose processes survived the final quiesce (``crucible:policy_misbehavior``) was graded as usual.
5. ``make_env(mode="rlm")`` built a harness that grades the World in place for a patch-graded task.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from crucible_pack_toy.runtime import FibTask, HelpdeskTask

from crucible.actors.policy import ContextConfig, EpisodeLimits, LLMToolPolicy
from crucible.core.schema.actions import Observation, ToolResult, ToolSpec
from crucible.core.schema.run import RunRecord
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.infra import LimitPolicy
from crucible.runtime.inject import FakeLLM, tool_reply
from crucible.runtime.record import build
from crucible.runtime.run import Runner, RunSpec
from crucible.train.chat import ByteChatCodec
from crucible.train.engine import GenResult
from crucible.train.world_env import EngineLLM

CALL = '<tool_call>{"name": "ws__read", "arguments": {"path": "a"}}</tool_call>'


def _engine(script: list[list[int]]) -> tuple[EngineLLM, ByteChatCodec]:
    codec = ByteChatCodec()

    async def gen(prompt, n, temp, **kw):
        ids = script.pop(0)
        return GenResult(ids, [-0.1] * len(ids), 1, "stop", None, len(prompt))

    return EngineLLM(codec, gen), codec


async def _compacting_episode(first_summary_blank: bool) -> LLMToolPolicy:
    codec = ByteChatCodec()

    def reply(text: str) -> list[int]:
        return [*text.encode(), codec.EOS]

    summaries = [[codec.EOS], reply("SUMMARY: read a")] if first_summary_blank else [reply("SUMMARY: read a")]
    llm, _ = _engine([reply(CALL), *summaries, reply(CALL)])
    pol = LLMToolPolicy("agent", llm, context=ContextConfig(budget_tokens=400, trigger=0.5, keep_recent_turns=1), max_tokens=50)
    pol.begin_episode("ep")
    obs = Observation(actor="agent", step=0, instruction="THE TASK", tools=[ToolSpec(app="ws", name="read", description="r", input_schema={"type": "object"})])
    for i in range(2):
        a = await pol.act(obs)
        obs = Observation(actor="agent", step=i + 1, tool_results=[ToolResult(tool_call_id=a.tool_call_id, function_name="ws.read", ok=True, content="x" * 600)])
    return pol


def _record(pol: LLMToolPolicy):
    rec = RunRecord(run_id="r", task="t", world="w", policy="p", started_at="2026-10-28T00:00:00+00:00", steps=2, outcome="completed",
                    reward=1.0, attempt="r-a1", meta={})
    return build(rec, calls={"agent": pol.model_calls}, segments={"agent": pol.segments}, mode="train")


async def test_a_resampled_summary_keeps_the_rollout_trainable():
    from crucible.train.bridges.common import flatten

    pol = await _compacting_episode(first_summary_blank=True)
    kinds = [(c["kind"], c.get("edge"), c.get("accepted")) for c in pol.model_calls]
    assert kinds == [("turn", None, None), ("compaction", None, False), ("compaction", "compaction_attempt", True), ("turn", None, None)]
    ro = _record(pol)
    assert not ro.mask_sample, ro.stop  # before the fix: capture_incomplete (prefix_break on the resample)
    seqs = flatten(ro, 0.5)
    by_call = {cid: (s, a, b) for s in seqs for cid, a, b in s.call_spans}
    blank_seq, a, b = by_call["agent:c2"]
    assert blank_seq.response_mask[a:b] == [0] * (b - a)  # the discarded blank summary is context, never trained
    acc_seq, a, b = by_call["agent:c3"]
    assert acc_seq.node and acc_seq.response_mask[a:b] == [1] * (b - a) and acc_seq.advantages[a] == pytest.approx(0.5)
    assert acc_seq.prompt_ids == pol.model_calls[2]["tokens"]["prompt_token_ids"]  # exactly the prompt the resample was sent
    assert seqs[-1].segment == 1 and not seqs[-1].node  # the outcome closes on the post-compaction segment


async def test_an_accepted_first_summary_is_a_link_of_its_chain():
    pol = await _compacting_episode(first_summary_blank=False)
    assert [(c["kind"], c.get("edge"), c.get("accepted")) for c in pol.model_calls] == [("turn", None, None), ("compaction", None, True), ("turn", None, None)]
    ro = _record(pol)
    assert not ro.mask_sample and ro.turns[1].prompt_is_delta  # cheap: it extends the segment's chain


async def test_the_tinker_client_fits_replies_to_the_models_context():
    from crucible.train.bridges.tinker.fake import FakeTinker
    from crucible.train.bridges.tinker.sampler import TinkerSampler
    from crucible.train.bridges.tinker.spend import SpendMeter

    fake = FakeTinker()
    s = TinkerSampler(fake, "Qwen/Qwen3.6-35B-A3B", meter=SpendMeter(10.0), codec=ByteChatCodec())
    llm = s.llm()
    assert await llm.context_window() == s.model.context
    pol = LLMToolPolicy("agent", llm, max_tokens=8000)
    pol.messages.append({"role": "user", "content": "x" * 4 * (s.model.context - 4000)})  # ~4k tokens of room left
    fitted = await pol._reply_tokens()
    assert fitted < 8000 and fitted + pol.context_tokens() <= s.model.context  # before the fix: 8000, refused by gen


def _fib_policy(n_calls: int) -> PolicyRef:
    acts = [tool_reply("ws__shell", {"command": "ls"}, i + 1) for i in range(n_calls)] + ["done"]
    return PolicyRef(id=f"fake:ls{n_calls}", make=lambda a: {"agent": LLMToolPolicy("agent", FakeLLM(list(acts)))}, config={"ls": n_calls})


async def _run(tmp_path, task, policy, harness, run_id):
    await Runner(tmp_path).run(RunSpec(run_id=run_id, tasks=[task], policies=[policy], harnesses=[harness], k=1))
    adir = min((tmp_path / run_id / "samples").glob("*/a*"))
    return json.loads((adir / "record.json").read_text())


@pytest.mark.parametrize("rule,expect", [("penalty:-0.25", -0.25), ("mask", None)])
async def test_the_workspace_harness_applies_its_limit_policy(tmp_path, rule, expect):
    h = WorkspaceHarness(limits=EpisodeLimits(max_steps=2), limit_policy=LimitPolicy(on_steps=rule))
    rec = await _run(tmp_path, FibTask(), _fib_policy(5), h, run_id=f"lim-{rule.split(':')[0]}")
    assert rec["meta"]["stop"] == "max_steps"
    if expect is None:
        assert rec["mask_sample"] and rec["failure"]["kind"] == "agent_timeout"
    else:
        assert not rec["mask_sample"] and rec["reward"] == pytest.approx(expect)  # before the fix: the partial grade (0.0)


async def _survivors(self):
    from crucible.world.world import WorldError

    raise WorldError("policy processes survived quiesce: {'ok': False, 'survivors': [4242]}")


async def test_a_workspace_episode_whose_processes_survive_quiesce_scores_zero(tmp_path, monkeypatch):
    from crucible.world.world import World

    monkeypatch.setattr(World, "quiesce", _survivors)
    fib = "a, b = 0, 1\nfor _ in range(10):\n    print(a)\n    a, b = b, a + b\n"
    acts = [tool_reply("ws__write_file", {"path": "fib.py", "content": fib}, 1), "done"]
    pol = PolicyRef(id="fake:fib", make=lambda a: {"agent": LLMToolPolicy("agent", FakeLLM(list(acts)))}, config={"fib": 1})
    rec = await _run(tmp_path, FibTask(), pol, WorkspaceHarness(), run_id="survive-ws")
    assert rec["failure"]["kind"] == "crucible:policy_misbehavior" and not rec["mask_sample"]
    assert rec["reward"] == 0.0  # before the fix: graded as usual (1.0: the fix was correct)


async def test_a_native_episode_whose_processes_survive_quiesce_scores_zero(tmp_path, monkeypatch):
    from crucible.world.world import World

    monkeypatch.setattr(World, "quiesce", _survivors)
    acts = [tool_reply("tickets__list_tickets", {"status": "open"}, 1), "All reviewed."]
    pol = PolicyRef(id="fake:hd", make=lambda a: {"agent": LLMToolPolicy("agent", FakeLLM(list(acts)))}, config={"hd": 1})
    rec = await _run(tmp_path, HelpdeskTask(), pol, NativeHarness(), run_id="survive-native")
    assert rec["failure"]["kind"] == "crucible:policy_misbehavior" and not rec["mask_sample"] and rec["reward"] == 0.0


def test_rlm_mode_refuses_a_patch_graded_task():
    from crucible.factory.envs import make_env

    class PatchTask:
        task_id = "t"
        prompt = "p"

        async def grade_patch(self, patch, gdir):  # pragma: no cover - never reached
            raise AssertionError

    with pytest.raises(ValueError, match="grade_patch"):
        make_env(PatchTask(), mode="rlm")


def _git(ws, *args, env=None):
    import os
    import subprocess

    return subprocess.run(["git", *args], cwd=ws, check=True, capture_output=True, env={**os.environ, **(env or {})}).stdout


async def test_reconcile_finds_the_end_checkpoint_when_snapshot_and_collect_render_the_change_differently(tmp_path):
    """The windows lane's end snapshot (the agent's index copy, ``core.quotepath=off``, no derived excludes) and the harness's
    collect (derived files dropped, paths quoted) are two byte renderings of one change: reconcile matched bytes only, so the
    final grade's vector never reached the end checkpoint (the credit_grade_mismatch the fix of 2026-10-25 was for)."""
    import subprocess

    from crucible.factory import checkpoint as ck
    from crucible.verify.collect import collect

    ws = tmp_path / "ws"
    ws.mkdir()
    _git(ws, "init", "-q", "-b", "main")
    (ws / "lib.py").write_text("def f():\n    return 1\n")
    _git(ws, "add", "-A")
    _git(ws, "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "base")
    _git(ws, "tag", "base")
    base = _git(ws, "rev-parse", "base").decode().strip()
    (ws / "lib.py").write_text("def f():\n    return 2\n")
    (ws / "café.py").write_text("X = 1\n")  # a non-ASCII path
    (ws / "__pycache__").mkdir()
    (ws / "__pycache__" / "lib.cpython-312.pyc").write_bytes(b"\x00\x01compiled")  # derived, not ignored by the repo
    snap_cmd = ("set -e; I=$(mktemp); cp .git/index \"$I\"; GIT_INDEX_FILE=\"$I\" git add -A >/dev/null 2>&1; "
                "GIT_INDEX_FILE=\"$I\" git -c core.quotepath=off diff --cached --binary base; rm -f \"$I\"")
    end_patch = (await asyncio.to_thread(subprocess.run, ["bash", "-c", snap_cmd], cwd=ws, check=True, capture_output=True, text=True)).stdout
    collected = (await collect(ws, base=base))["patch"]
    assert end_patch != collected and "__pycache__" in end_patch and "__pycache__" not in collected  # two renderings
    assert ck.same_change(end_patch, collected) and not ck.same_change(end_patch, collected.replace("return 2", "return 3"))
    assert not ck.same_change('["a"]', '["b"]') and ck.same_change("", "")  # text that is not a patch is never "the same change"

    class Loaded:  # the sidecar's grading under load lost milestone "b"; the final grade passes it
        id = "fake/loaded@1"

        def __init__(self):
            self.milestones, self.requirement_ids = [{"id": "a"}, {"id": "b"}], []

        async def milestone_status(self, patch):
            return {"a": bool(patch), "b": False}

    async def snap(_w):
        return end_patch

    side = ck.CheckpointSidecar(Loaded(), tmp_path / "side", ck.CheckpointConfig(snapshot=snap, every=0, grade_start=False))
    await side.on_start(None)
    await side.on_step(None, 1, "agent", None, None)
    await side.on_end(None)
    got = side.reconcile(collected, {"a": True, "b": True})
    assert got is not None and got["phi_final"] == pytest.approx(1.0)  # before the fix: None, Φ_end stayed 0.5
    assert side.reconcile(collected.replace("return 2", "return 3"), {"a": True, "b": False}) is None  # another change: never matched


# ---- the M16 RLM (sub-audit findings C2, M1, M3; repro scripts adapted) ------------------------------------------------------


class _Interleaved(ByteChatCodec):
    def thinking_behaviour(self) -> str:
        return "interleaved"


@pytest.mark.parametrize("codec_cls", [_Interleaved, ByteChatCodec])
async def test_rlm_continuations_end_the_episode_under_an_interleaved_template(tmp_path, codec_cls):
    """A CONTINUE nudge is a user row: an interleaved template re-renders there and opens a rerender segment, which reset the
    per-segment nudge count, so a tool-less model ran to budget:turns (a truncation) instead of the scored agent_done."""
    from test_m16_f_agent import cfg, episode

    codec = codec_cls()

    async def gen(prompt, max_tokens, temperature, **kw):
        ids = [*b"I think the task is complete.", codec.EOS]  # never a tool call
        return GenResult(completion_ids=ids, logprobs=[-0.1] * len(ids), version=0, finish="stop")

    _ep, _rep, p = await episode(tmp_path, EngineLLM(codec, gen, model="policy"), config=cfg(max_turns=40, max_continuations=2, window_tokens=64_000))
    assert p.stop == "agent_done" and p.counters["turns"] == 3 and p.counters["continuations"] == 2  # before: budget:turns after 40


async def test_rlm_a_malformed_python_call_is_answered_not_run(tmp_path):
    """Arguments that did not parse, or carry no ``code`` string, ran as an empty cell that printed "(no output)"."""
    from test_m16_f_agent import ScriptLLM, cell, cfg, episode

    from crucible.actors.llm import LLMResponse

    def decide(messages, tools, role):
        if role == "summary":
            return LLMResponse(text="S")
        n = sum(1 for m in messages if m.get("role") == "assistant")
        if n == 0:
            return LLMResponse(text="", tool_calls=[{"id": "a", "name": "python", "arguments": {"__unparseable__": '{"code": "print(1)'}}])
        if n == 1:
            return LLMResponse(text="", tool_calls=[{"id": "b", "name": "python", "arguments": {"source": "print(2)"}}])
        return cell("rlm.finish('x')", 9)

    _ep, _rep, p = await episode(tmp_path, ScriptLLM(decide), config=cfg())
    tools = {m["tool_call_id"]: m["content"] for m in p.messages if m["role"] == "tool"}
    assert "did not parse" in tools["a"] and "did not parse" in tools["b"]  # before: "(no output)"
    if p.kernel is not None:  # no cell ran for the two malformed calls
        assert [e.code for e in p.kernel.journal if not e.bootstrap] == ["rlm.finish('x')"]
    assert p.stop == "agent_done"


async def test_rlm_a_summary_the_run_never_continued_from_is_not_trained(tmp_path):
    """The rebuilt context left no room (compaction_failed): the accepted attempt stayed accepted and trainable, and the
    failed segment named it as its summary call."""
    from crucible_pack_toy.runtime import INSTRUCTION
    from test_m16_f_agent import ScriptLLM, cell, cfg, episode

    from crucible.actors.llm import LLMResponse

    def decide(messages, tools, role):
        if role == "summary":
            return LLMResponse(text="SUMMARY: " + "the plan is to list tickets. " * 40)
        return cell("x = 1\nprint('y' * 4000)", 1)

    task = HelpdeskTask(instruction=INSTRUCTION + "\n" + ("Policy detail line that matters. " * 600))
    _ep, _rep, p = await episode(tmp_path, ScriptLLM(decide), task=task, config=cfg())
    assert p.stop == "compaction_failed"
    summaries = [c for c in p.model_calls if c["kind"] == "compaction"]
    assert summaries and all(c["accepted"] is False and c["trainable"] is False for c in summaries)
    assert [s.get("summary_call") for s in p.segments if s["boundary_reason"] == "compaction_failed"] == [None]


async def test_agentenv_data_restore_takes_every_part_kind(monkeypatch):
    """agentenv's ``Part`` is text | file | data; restore read every non-data part as a file part (a ValidationError on text)."""
    pytest.importorskip("agentenv_protocol")
    from types import SimpleNamespace

    from agentenv_protocol import client
    from agentenv_protocol.types import DataPart, TextPart

    from crucible.interop.agentenv.mcp_env import AgentEnvMcpApp

    sent: list = []

    async def reset_data(url):
        sent.append(("reset", url))

    async def add_data(url, parts):
        sent.append(("add", [type(p) for p in parts]))

    monkeypatch.setattr(client, "reset_data", reset_data)
    monkeypatch.setattr(client, "add_data", add_data)
    await AgentEnvMcpApp.load(SimpleNamespace(base_url="http://x"), [{"kind": "text", "text": "note"}, {"kind": "data", "data": {"a": 1}}])
    assert sent == [("reset", "http://x"), ("add", [TextPart, DataPart])]


async def test_a_masked_agentenv_grade_row_carries_no_score():
    """V8 through agentenv: a masked grade's verification row is skipped with no score (it carried 0.0, which agentenv's
    aggregate of a tolerant run scored); the step still raises ``CrucibleMaskedGrade``."""
    pytest.importorskip("agent_env")
    import httpx
    from agent_env.task_step.context import TaskStepContext

    from crucible.interop.agentenv.plugin.steps import WORLDS, CrucibleMaskedGrade, CrucibleVerifyStep, CrucibleWorldStep, instance_for
    from tests.unit.test_m11_audit import GOLD

    ctx = await CrucibleWorldStep("w", env_id="hd", task="tests.unit.test_m11_audit:FlakyTickets").execute(TaskStepContext())
    inst = instance_for(ctx, "hd")
    try:
        base = ctx.deployed_envs[-1].mcp_url.removesuffix("/mcp")
        async with httpx.AsyncClient(timeout=60) as c:
            for a in GOLD:  # its World's App crashes once: the grade is masked
                await c.post(f"{base}/step", json={"action": "call_tool", "tool_name": a.function_name.replace(".", "__", 1), "arguments": a.arguments})
        with pytest.raises(CrucibleMaskedGrade):
            await CrucibleVerifyStep("v", env_id="hd").execute(ctx)
        v = ctx.metadata["verifications"]["crucible"]
        assert v["score"] is None and v["results"] == [{"id": "reward", "result": False, "weight": 1.0, "skipped": True}]
    finally:
        if inst in WORLDS:
            await WORLDS.pop(inst)["world"].close()
