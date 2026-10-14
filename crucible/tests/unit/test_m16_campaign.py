"""M16 campaign (docs/research/long-horizon-rlm/07-campaign.md): delegating scripted RLM models and the tree measurements.

- The delegating model keeps no hidden memory: which agent a call serves (root, worker, lead, prober...) and what it does
  next come from the call's messages; its own summary text never reads as a progress marker.
- ``LockedTitoLLM`` pairs each reply with its own call when agents call concurrently.
- Episodes through the M10 Runner (host kernel, byte TITO): delegation with world="shared" children and analysis-only
  world="none" children at depth 1 and 2 score 1.0, train-valid, every World step attributed to the agent that made it,
  forwarded tokens = entered tokens, Φ telescopes on the root chain; the adversarial probe is refused at every limit
  (spawn budget, depth, World grant, arguments) and finishing with children running ends the episode cleanly.
- The open defect found by the campaign (D1): the record's positional progress credit is shifted by a child's ``done``
  action (``xfail(strict=True)``: it flips when ``runtime.record.credit_from_progress`` maps rows by tool-call id).

API-free and CPU-only; cells run in the host kernel (trusted scripted code, as the other RLM tests do).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
from crucible_pack_toy.longhorizon.ladder import LadderConfig

from crucible.runtime.scripted_rlm import CallView
from crucible.runtime.scripted_rlm_tree import CHILD_MARK, DelegatingRLM, tree_beliefs
from crucible.train.chat import ByteChatCodec
from crucible.train.tito_scripted import LockedTitoLLM

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("lh_campaign", ROOT / "scripts" / "lh_campaign.py")
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)

TOOLS = [{"type": "function", "function": {"name": "python"}}]
ROOT_SYS = {"role": "system", "content": "You are an agent working in a simulated workplace through a persistent Python REPL."}
CHILD_SYS = {"role": "system", "content": ROOT_SYS["content"] + f"\n- {CHILD_MARK}: `TASK` is the brief your parent agent gave you."}


def tool(text: str) -> dict:
    return {"role": "tool", "tool_call_id": "c", "content": text}


def worker_brief(mode: str = "shared") -> dict:
    return {"role": "user", "content": f"[brief] role=worker mode={mode} workers=0\nHandle a batch.\nBATCH [101, 102]"}


# ---- the model: no hidden memory ------------------------------------------------------------------------------------------
def test_role_comes_from_the_messages():
    m = DelegatingRLM()
    root = [ROOT_SYS, {"role": "user", "content": "You are on the support shift..."}]
    assert m.brief_role(CallView.of(root, TOOLS)) == ("root", "")
    assert m.brief_role(CallView.of([CHILD_SYS, worker_brief("none")], TOOLS)) == ("worker", "none")
    lead = {"role": "user", "content": "[brief] role=lead mode=shared workers=2\nBATCH [1, 2]"}
    assert m.brief_role(CallView.of([CHILD_SYS, lead], TOOLS)) == ("lead", "shared")
    # the opening moves of each agent follow from what is (not) visible yet
    assert m.reply_for(root, TOOLS).kind == "read_policy"
    assert m.reply_for([CHILD_SYS, worker_brief()], TOOLS).kind == "worker_define"
    assert m.reply_for([CHILD_SYS, lead], TOOLS).kind == "lead_spawn"
    assert [x["agent_role"] for x in m.log] == ["root", "worker", "lead"]


def test_worker_follows_the_latest_marker_and_finishes_at_zero():
    m = DelegatingRLM()
    base = [CHILD_SYS, worker_brief(), tool("DEFINED-WORKER step, Q, DONE, BATCH n=2")]
    r = m.reply_for([*base, tool("QUEUED 101 dup refund O-1 20.00\nBATCH-LEFT 1")], TOOLS)
    assert r.kind == "worker_step" and "batch left: 1" in r.code
    r = m.reply_for([*base, tool("BATCH-LEFT 1"), tool("HANDLED 102 info -> resolved\nBATCH-LEFT 0")], TOOLS)
    assert r.kind == "worker_finish"


def test_own_summary_never_reads_as_a_progress_marker():
    """Campaign finding (scripted-model bug, fixed): the summary's next-step line said 'until QUEUE-LEFT 0', which the
    model then read as 'the queue is empty' and skipped 37 refunds after a compaction."""
    m = DelegatingRLM()
    t = tree_beliefs(CallView.of([ROOT_SYS, {"role": "user", "content": "x"}, tool("CHILDREN-LEFT 0 QUEUE-LEFT 37")], TOOLS))
    summary = m.root_summary(m.beliefs(CallView.of([ROOT_SYS, {"role": "user", "content": "x"}], TOOLS)), t)
    after = tree_beliefs(CallView.of([ROOT_SYS, {"role": "user", "content": "x"}, {"role": "user", "content": summary}], TOOLS))
    assert after.queue_left == 37 and after.children_left == 0
    ws = m.child_summary(tree_beliefs(CallView.of([CHILD_SYS, worker_brief(), tool("DEFINED-WORKER\nBATCH-LEFT 11")], TOOLS)), "worker")
    assert tree_beliefs(CallView.of([CHILD_SYS, worker_brief(), {"role": "user", "content": ws}], TOOLS)).batch_left == 11


def test_locked_tito_pairs_each_reply_with_its_call():
    model = DelegatingRLM()
    llm = LockedTitoLLM(model, ByteChatCodec())
    root = [ROOT_SYS, {"role": "user", "content": "You are on the support shift..."}]
    child = [CHILD_SYS, worker_brief()]

    async def go():
        return await asyncio.gather(*(llm.complete(msgs, tools=TOOLS, max_tokens=4000, temperature=0.0, sample_key=k)
                                      for msgs, k in ((root, "e"), (child, "e/w0"), (root, "e2"), (child, "e2/w0"))))

    out = asyncio.run(go())
    codes = [json.dumps(r.tool_calls) for r in out]
    assert "kb.read_doc" in codes[0] and "kb.read_doc" in codes[2]
    assert "DEFINED-WORKER" in codes[1] and "DEFINED-WORKER" in codes[3]


# ---- episodes ---------------------------------------------------------------------------------------------------------------
def _run(mode: str, depth: int, tickets: int = 12, **rlm) -> dict:
    model = DelegatingRLM(mode=mode, depth=depth, children=4, poll_wait_s=5.0, overspawn=10)
    cfg = LadderConfig(rung=100, tickets=tickets, kernel="local", tito="byte", restore_probe=False, checkpoint_every=0,
                       rlm={"tree_max_depth": depth, "max_tokens": 2000, **rlm})
    return asyncio.run(lc.run_tree(cfg, model, ByteChatCodec()))


@pytest.mark.parametrize(("mode", "depth"), [("shared", 1), ("none", 2)])
def test_delegation_episode_is_correct_attributed_and_train_valid(mode, depth):
    r = _run(mode, depth)
    assert r["reward"] == 1.0 and r["stop"] == "agent_done" and r["valid_for_training"] == [True, "ok"]
    assert r["tree"]["spawns"] == (4 if depth == 1 else 6)
    kids = r["children"]
    assert kids and all(v["stop"] == "agent_done" for v in kids.values())
    assert r["attribution"]["attributed_ok"]
    if mode == "shared":  # children stepped the World themselves, each step under its own model call
        assert sum(v for k, v in r["attribution"]["by_agent"].items() if k != "root") > 0
    else:
        assert set(r["attribution"]["by_agent"]) == {"root"}
    assert r["tokens"]["ratio"] == 1.0 and r["tokens"]["sequences"] >= 1 + len(kids)  # one per agent segment (byte tokens compact)
    assert r["credit"]["telescopes"]


def test_adversarial_probe_is_refused_at_every_limit(tmp_path):
    from crucible_pack_toy.longhorizon.ladder import LadderConfig as LC

    model = DelegatingRLM(mode="adversarial", depth=2, overspawn=10)
    cfg = LC(rung=100, tickets=12, kernel="local", tito="byte", restore_probe=False, checkpoint_every=0,
             rlm={"tree_max_depth": 2, "tree_max_spawns": 8, "max_tokens": 2000})
    r = asyncio.run(lc.run_tree(cfg, model, ByteChatCodec(), keep=tmp_path))
    p = lc.adversarial_findings(max((tmp_path / "camp" / "samples").glob("*/a*")))
    assert r["stop"] == "agent_done" and r["valid_for_training"] == [True, "ok"] and r["reward"] == 0.0
    probe = p["root_lines"]["ADV-PROBE-RESULT"]
    assert "world_call\": \"refused" in probe and "PermissionError" in probe  # world='none' child: no World
    assert "spawn_shared_grandchild\": \"refused" in probe  # it cannot grant World access to a grandchild
    assert "spawn_depth_plus_1" in probe and "child agents are off at this depth" in probe  # depth 3 refused
    assert p["escalator_world_calls"] == 0
    args = json.loads(p["root_lines"]["ADV-ARGS"].split(" ", 1)[1])
    assert all(v.startswith("refused") for v in args.values()), args
    assert "budget:spawns" in p["root_lines"]["ADV-SPAWNED"]
    assert r["tree"]["spawns"] == 8 and r["tree"]["refused"] >= 2
    # finishing with children running: the episode ends, every child is stopped
    assert "running" in p["root_lines"]["ADV-LIST"]


@pytest.mark.xfail(strict=True, reason="D1 (07-campaign.md): runtime.record.credit_from_progress zips actions and progress rows by "
                                       "position; a child's rlm.finish adds a 'done' action with no World step, so every later root "
                                       "call is credited with a neighbour's ΔΦ (rows carry tool_call_id; map by it)")
def test_record_credit_matches_exact_attribution_with_children():
    r = _run("none", 1)
    assert r["credit"]["telescopes"]
    assert r["credit"]["exact_vs_record_mismatched_calls"] == 0, r["credit"]["mismatch_examples"]


class _AskingChild:
    """A root that spawns one world='shared' child; the child asks the supervisor for an approval and reads its INBOX."""

    CHILD = ("await world.send('supervisor', 'Please approve a refund of 47.66 USD for ticket 100, order O-50001.')\n"
             "await world.call('kb.list_docs')\nawait world.call('kb.list_docs')\n"
             "print('CHILD-INBOX', len(INBOX))\nrlm.finish('CHILD-INBOX %d' % len(INBOX))")

    @staticmethod
    def model():
        from crucible.runtime.scripted_rlm import Reply, ScriptedRLM, message_text

        child = _AskingChild.CHILD

        class M(ScriptedRLM):
            def decide(self, v):
                if v.role != "root":
                    return Reply("text", "ok")
                if CHILD_MARK in v.system:
                    return Reply("child", "", child)
                txt = "\n".join(message_text(m) for m in v.messages if m.get("role") in ("tool", "user"))
                if "SPAWNED" not in txt:
                    return Reply("spawn", "", "await rlm.agent.spawn('[brief] role=x mode=shared\\nAsk.', name='c', world='shared')\nprint('SPAWNED')")
                if "RESULT" not in txt:
                    return Reply("wait", "", "r = await rlm.agent.result('c', wait_s=60)\nprint('RESULT', r['answer'])")
                return Reply("finish", "", "rlm.finish('x')")

        return M()


@pytest.mark.xfail(strict=True, reason="D2 (07-campaign.md): a world='shared' child's World messages are answered into the ROOT's INBOX "
                                       "(AgentLoop._take_inbox returns [] for children), so a child cannot complete an approval "
                                       "round-trip and never sees a policy announcement")
def test_a_shared_child_receives_the_reply_to_its_own_message(tmp_path):
    cfg = LadderConfig(rung=100, tickets=12, kernel="local", tito="byte", restore_probe=False, checkpoint_every=0, rlm={"max_tokens": 2000})
    asyncio.run(lc.run_tree(cfg, _AskingChild.model(), ByteChatCodec(), keep=tmp_path))
    adir = max((tmp_path / "camp" / "samples").glob("*/a*"))
    out = [json.loads(x) for x in (adir / "rlm" / "agent" / "root" / "c" / "events.jsonl").read_text().splitlines()]
    shown = [e["text"] for e in out if e["kind"] == "tool" and "CHILD-INBOX" in e["text"]]
    assert shown and "CHILD-INBOX 0" not in shown[0]


class _CallsFinishByName:
    """A model that calls the kernel function ``rlm.finish`` as if it were a tool (Qwen2.5-0.5B did, campaign 2), then
    finishes properly; it keeps the correction it was shown."""

    def __init__(self):
        self.seen: list[str] = []
        self.n = 0

    async def complete(self, messages, *, tools=None, **kw):
        from crucible.actors.llm import LLMResponse

        if not tools:
            return LLMResponse(text="summary", usage={"prompt_tokens": 10, "completion_tokens": 2}, model="m", raw_finish="stop")
        self.n += 1
        if self.n == 1:
            return LLMResponse(text="", tool_calls=[{"id": "t1", "name": "rlm.finish", "arguments": {"summary": "done"}}],
                               usage={"prompt_tokens": 100, "completion_tokens": 5}, model="m", raw_finish="stop")
        self.seen += [str(m.get("content")) for m in messages if m.get("role") == "tool"]
        return LLMResponse(text="", tool_calls=[{"id": f"t{self.n}", "name": "python", "arguments": {"code": "rlm.finish('done')"}}],
                           usage={"prompt_tokens": 120, "completion_tokens": 5}, model="m", raw_finish="stop")


@pytest.mark.xfail(strict=True, reason="D4 (07-campaign.md): AgentLoop._correction treats every dotted name as a World tool, so a by-name "
                                       "call of the kernel function rlm.finish is 'corrected' to await world.call('rlm.finish', ...), "
                                       "an unknown World tool")
def test_by_name_correction_of_a_kernel_function_points_at_the_kernel():
    from crucible_pack_toy.longhorizon.task import BacklogTask

    from crucible.actors.policy import EpisodeLimits
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.harness.rlm import RLMConfig, RLMHarness
    from crucible.runtime.run import Runner, RunSpec

    llm = _CallsFinishByName()
    task = BacklogTask(0, 12)
    import tempfile

    try:
        asyncio.run(Runner(Path(tempfile.mkdtemp())).run(RunSpec(run_id="d4", tasks=[task], policies=[PolicyRef(id="m", make=lambda a: {"agent": llm})],
                                                                 harnesses=[RLMHarness(RLMConfig(kernel="local", allow_host=True, temperature=0.0),
                                                                                       limits=EpisodeLimits(max_steps=50))], k=1)))
    finally:
        task.cleanup()
    shown = " ".join(llm.seen)
    assert "rlm.finish" in shown
    assert "world.call('rlm.finish'" not in shown, shown
