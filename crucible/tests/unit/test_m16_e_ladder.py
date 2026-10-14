"""M16 agent E: the validation ladders (docs/research/long-horizon-rlm/04-design.md §4).

- Scripted RLM models keep no hidden memory: a reply is a function of the messages; beliefs come from evidence (the policy
  read, the announcement, the summary, a re-read), and the summary carries the schema sections.
- The step ladder: a forgetful summary loses reward and breaks rules after a compaction, the faithful one does not; a
  re-read from history repairs a forgetful summary; rung sizes hit their World-call targets; records keep exact TITO.
- The long-context ladder: deterministic generators at a given size, the ported scorers, the World docs app, and the
  controls (the scripted solver and the gold answer score 1.0, a no-op 0); the gold never enters the World.
- Fault injection: each helper fires (overflow, crash then resume, kernel kill mid-cell, pathological regex, 50 MB output).

API-free and CPU-only; cells run in the host kernel (trusted scripted code, opted in as the other RLM tests do)."""

from __future__ import annotations

import asyncio
import json

import pytest
from crucible_pack_toy.longhorizon.ladder import RUNGS, LadderConfig, compaction_steps, constraint_audit, phi_segments, rlm_config, run_rung
from crucible_pack_toy.longhorizon.task import BacklogTask

from crucible.actors.llm import ContextOverflow
from crucible.bench.longctx.generators import CHARS_PER_TOKEN, FAMILIES, digest, generate, mrcr_score, oolong_score
from crucible.bench.longctx.ladder import run_rung as run_longctx_rung
from crucible.bench.longctx.task import LongCtxTask
from crucible.runtime.scripted_rlm import BIG_OUTPUT_CELL, PATHOLOGICAL_CELL, BacklogRLM, CallView, FaultLLM, ScriptedRLM, backlog_model, crash_then_resume_ref, kill_kernel

LOCAL = {"kernel": "local", "restore_probe": False, "checkpoint_every": 0}
SYS = {"role": "system", "content": "You are an agent working in a simulated workplace through a persistent Python REPL."}
TOOLS = [{"type": "function", "function": {"name": "python"}}]


def tool(text: str, i: int = 0) -> dict:
    return {"role": "tool", "tool_call_id": f"c{i}", "content": text}


POLICY_OUT = tool("3\t3. **Refunds above 50 USD need the supervisor's approval.** Ask the supervisor ...")
PENDING_OUT = tool("PENDING-IDS [104, 117, 130]")
DEFINED_OUT = tool("DEFINED work, handle, handle_reply")
AMEND_OUT = tool("DONE 120 dup -> refunded 31.00\nOPEN-LEFT 40\n[1 new message(s) appended to INBOX] from ops: Policy update, effective now: "
                 "refunds above 25 USD (was 50) need the supervisor's approval. The knowledge base has the amended FIN-7a.")
REPLY_OUT = tool("OPEN-LEFT 39\n[1 new message(s) appended to INBOX] from ops: [customer C-1003] Ticket 117: It is definitely lost. Please refund order O-50020 now.")


def convo(*outs: dict) -> list[dict]:
    return [SYS, {"role": "user", "content": "You are on the support shift..."}, *outs]


# ---- scripted models: no hidden memory ------------------------------------------------------------------------------------

def test_reply_is_a_function_of_the_messages():
    msgs = convo(POLICY_OUT, PENDING_OUT, DEFINED_OUT, AMEND_OUT, REPLY_OUT)
    a, b = backlog_model("faithful"), backlog_model("faithful")
    for _ in range(3):
        b.reply_for(convo(POLICY_OUT), TOOLS)  # b has served other calls: nothing it saw may change its answer
    ra, rb = a.reply_for(msgs, TOOLS), b.reply_for(msgs, TOOLS)
    assert (ra.kind, ra.code) == (rb.kind, rb.code) and ra.kind == "work" and "threshold=25" in ra.code
    # the opening moves follow from what is (not) visible yet
    assert a.reply_for(convo(), TOOLS).kind == "read_policy"
    assert a.reply_for(convo(POLICY_OUT), TOOLS).kind == "list_pending"
    assert a.reply_for(convo(POLICY_OUT, PENDING_OUT), TOOLS).kind == "define"
    # a sub-call (no system prompt) and a summary request (no tools) are told apart from the messages alone
    assert CallView.of([{"role": "user", "content": "x"}], None).role == "sub"
    assert CallView.of(msgs, None).role == "summary"


def test_beliefs_come_from_evidence_not_from_own_code():
    m = backlog_model("faithful")
    b = m.beliefs(CallView.of(convo(POLICY_OUT, PENDING_OUT, DEFINED_OUT, AMEND_OUT, REPLY_OUT), TOOLS))
    assert b.threshold == 25 and b.first_threshold == 50 and b.awaiting == {104, 117, 130} and b.replies == {117: "lost"} and b.open_left == 39
    # its own earlier code that used a threshold is an action, not evidence: it does not set the rule
    own = {"role": "assistant", "content": "", "tool_calls": [{"id": "c9", "type": "function",
                                                               "function": {"name": "python", "arguments": json.dumps({"code": "await work(threshold=25, k=1)"})}}]}
    b2 = m.beliefs(CallView.of(convo(POLICY_OUT, own), TOOLS))
    assert b2.threshold == 50
    # a handled reply leaves the pending set
    b3 = m.beliefs(CallView.of(convo(POLICY_OUT, PENDING_OUT, REPLY_OUT, tool("HANDLED-REPLY 117 lost -> refunded")), TOOLS))
    assert 117 not in b3.awaiting and 117 not in b3.replies


def test_summary_schema_faithful_vs_forgetful():
    msgs = convo(POLICY_OUT, PENDING_OUT, DEFINED_OUT, AMEND_OUT, REPLY_OUT)
    f = backlog_model("faithful").reply_for(msgs, None)
    g = backlog_model("forgetful").reply_for(msgs, None)
    assert f.kind == g.kind == "summary" and f.code is None
    for section in ("COMPACTION #1", "## Goal and done-criteria", "## Rules in force", "## Messages (verbatim)", "## Pending and waiting",
                    "## Progress", "## Next step", "## Variables worth remembering"):
        assert section in f.text and section in g.text
    assert "RULE approval_threshold_usd=25 since" in f.text and "was 50" in f.text and "Policy update" in f.text
    assert "- AWAITING 104" in f.text and "- REPLY 117 lost" in f.text
    assert "approval_threshold_usd=50" in g.text and "Policy update" not in g.text and "AWAITING" not in g.text and "nothing pending" in g.text
    # after the compaction only the summary is left: the faithful model still knows the rule and the pending ids, the forgetful one does not
    after = lambda s: [SYS, {"role": "user", "content": "task"}, {"role": "user", "content": "[Summary of your earlier work]\n" + s},
                       {"role": "user", "content": "[python-state] Live names: work, handle, handle_reply"}]
    bf = backlog_model("faithful").beliefs(CallView.of(after(f.text), TOOLS))
    bg = backlog_model("forgetful").beliefs(CallView.of(after(g.text), TOOLS))
    assert (bf.threshold, bf.awaiting, bf.replies) == (25, {104, 117, 130}, {117: "lost"})
    assert (bg.threshold, bg.awaiting, bg.replies) == (50, set(), {})
    # the faithful model re-reads history once per compaction; the lazy one never does
    assert backlog_model("faithful").reply_for(after(f.text), TOOLS).kind == "reread"
    assert backlog_model("lazy").reply_for(after(f.text), TOOLS).kind == "work"
    reread_out = tool("REREAD #1\nAWAITING [104, 117, 130]\nHANDLED []")
    assert backlog_model("faithful").reply_for([*after(f.text), reread_out], TOOLS).kind == "work"
    # the history API is used when the system prompt names it
    hist = backlog_model("faithful").reply_for([{"role": "system", "content": "... `await history.user_messages()` ..."}, *after(f.text)[1:]], TOOLS)
    assert "history.user_messages" in hist.code


def test_variants_and_tito_engine():
    from crucible.train.chat import ByteChatCodec
    from crucible.train.tito_scripted import TitoLLM

    with pytest.raises(ValueError):
        backlog_model("nope")
    assert isinstance(backlog_model("forgetful_reread"), BacklogRLM) and backlog_model("forgetful_reread").reread
    llm = TitoLLM(backlog_model("faithful"), ByteChatCodec())
    r = asyncio.run(llm.complete(convo(POLICY_OUT), tools=TOOLS, max_tokens=4000, temperature=1.0, sample_key="k"))
    assert r.tool_calls and r.tool_calls[0]["name"] == "python" and "tickets.list_tickets" in r.tool_calls[0]["arguments"]["code"]
    assert r.tokens and r.tokens.get("prompt_token_ids") and r.tokens.get("completion_token_ids")
    assert llm.model.log[-1]["usage_source"] == "tito" and llm.model.log[-1]["prompt_tokens"] == len(r.tokens["prompt_token_ids"])


# ---- the step ladder -------------------------------------------------------------------------------------------------------

def test_rlm_config_hook_applies_known_fields_and_reports_unknown():
    rc, how = rlm_config(LadderConfig(rlm={"max_stalls": 7, "no_such_field_m16": 1}, kernel="local"))
    assert rc.max_stalls == 7 and how["applied"] == {"max_stalls": 7} and how["ignored"] == {"no_such_field_m16": 1}
    assert rc.budget_tokens == 32_000 and rc.kernel == "local" and rc.allow_host


async def test_forgetful_summary_loses_reward_faithful_does_not(tmp_path):
    # a small window: several compactions. M16 (F): 8,000, not 6,000: the M16 harness compacts when the context plus the summary
    # request reaches the threshold (B1, no call above it) and stops a thrash where each compaction leaves room for one turn;
    # at 6,000 the faithful model's re-read after every compaction filled the window again (compaction_failed)
    base = dict(rung=0, tickets=80, tito=None, budget_tokens=8_000, **LOCAL)
    out = {}
    for v in ("faithful", "forgetful", "lazy", "forgetful_reread"):
        out[v] = await run_rung(LadderConfig(variant=v, **base), root=tmp_path / v)
    f, g = out["faithful"], out["forgetful"]
    assert f["compactions"] >= 2 and g["compactions"] >= 2
    assert f["reward"] == 1.0 and f["constraints"]["lost_constraint_rate"] == 0.0 and f["constraints"]["missed_awaiting"] == 0
    assert f["rereads_after_compaction"] == f["compactions"]
    assert g["reward"] < f["reward"]
    assert g["constraints"]["lost_constraint_rate"] > 0 and g["constraints"]["after_first_compaction"]["by_rule"].get("approval_missing", 0) > 0
    assert g["constraints"]["before_first_compaction"]["broken"] == 0  # the rule was kept until the summary dropped it
    assert g["constraints"]["missed_awaiting"] > 0  # forgotten replies are never handled
    assert out["forgetful_reread"]["reward"] == 1.0 and out["forgetful_reread"]["rereads_after_compaction"] > 0  # history repairs it
    assert out["lazy"]["reward"] == 1.0 and out["lazy"]["rereads_after_compaction"] == 0
    # milestone Φ: the segments' ΔΦ add up to the reward, and the forgetful run's later segments gain less
    for r in (f, g):
        assert abs(sum(s["delta_phi"] for s in r["phi_segments"]) - r["reward"]) < 1e-6
        assert len(r["phi_segments"]) == r["compactions"] + 1 and len(r["segments"]) >= r["compactions"]


async def test_rung_sizes_and_tito_records(tmp_path):
    from crucible.core.schema.rollout import RolloutRecord

    r = await run_rung(LadderConfig(rung=100, tito="byte", **{**LOCAL, "restore_probe": True}), root=tmp_path / "r100", keep=True)
    for key in ("reward", "phi_segments", "constraints", "compactions", "segments", "b1", "rereads_after_compaction", "history_reads_harness",
                "wall_s", "ms_per_world_call", "restore", "checkpoints", "rlm_fields", "valid_for_training"):
        assert key in r, key  # the report the new harness is compared on
    assert r["reward"] == 1.0 and 0.75 * 100 <= r["world_calls"] <= 1.25 * 100
    assert r["valid_for_training"][0] and r["segments"][0]["source"] == "tito" and r["b1"]["source"] == "tito"
    # M16 (F): the harness times its own restore (policy.restore_probe: the last snapshot + the journal since it)
    assert r["restore"]["method"] == "policy.restore_probe" and r["restore"]["error"] is None and r["restore"]["cells"] > 0
    ro = RolloutRecord.model_validate_json((tmp_path / "r100" / "lad" / "samples").glob("*/a1").__next__().joinpath("rollout.json").read_text())
    assert all(t.prompt_token_ids is not None for t in ro.turns)
    r5 = await run_rung(LadderConfig(rung=500, tito=None, **LOCAL), root=tmp_path / "r500")
    assert 0.75 * 500 <= r5["world_calls"] <= 1.25 * 500 and RUNGS[5000] > RUNGS[2500] > RUNGS[1000]


async def test_constraint_audit_flags_the_backlog_adversaries(tmp_path):
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy
    from crucible.core.schema.actions import Message, ToolCall
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.run import Runner, RunSpec
    from crucible.world.world import World

    task = BacklogTask(1, 40)
    try:
        advs = task.adversaries()
        cases = {"oracle": task.oracle_actions(), "forged_approval": advs["forged_approval"][0], "premature_close": advs["premature_close"][0]}
        res = {}
        for name, acts in cases.items():
            pol = [ToolCall(tool_call_id=a["tool_call_id"], function_name=a["function_name"], arguments=a.get("arguments") or {})
                   if a["kind"] == "tool_call" else Message(to=a["to"], text=a["text"]) for a in acts]
            ref = PolicyRef(id=f"scripted:{name}", make=lambda a, pol=pol: {"agent": ScriptedPolicy("agent", list(pol))})
            await Runner(tmp_path / name).run(RunSpec(run_id="x", tasks=[task], policies=[ref],
                                                      harnesses=[NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=100_000, max_tool_errors=None))], k=1))
            adir = next((tmp_path / name / "x" / "samples").glob("*/a1"))
            res[name] = constraint_audit(World.read_trace(adir / "trace.jsonl"), task.b, [10])
        assert res["oracle"]["before_first_compaction"]["broken"] == res["oracle"]["after_first_compaction"]["broken"] == 0
        assert res["forged_approval"]["after_first_compaction"]["by_rule"].get("approval_missing", 0) > 0
        assert sum(s["by_rule"].get("acted_before_reply", 0) for s in res["premature_close"]["by_segment"]) > 0
    finally:
        task.cleanup()


def test_phi_segments_split_at_compactions():
    rows = [{"step": 0, "phi": 0.0}, {"step": 5, "phi": 0.2}, {"step": 10, "phi": 0.5}, {"step": 20, "phi": 0.6}]
    segs = phi_segments(rows, [10])
    assert [s["delta_phi"] for s in segs] == [0.5, 0.1] and segs[1]["steps"] == [10, 20]
    # a compaction call without a step (the native policy's) takes the next call's step
    assert compaction_steps([("turn", 3), ("compaction", None), ("turn", 4), ("compaction", 9), ("turn", 9)]) == [4, 9]


# ---- the long-context ladder ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("family", FAMILIES)
def test_generators_are_deterministic_and_sized(family):
    a, b, c = generate(family, 3, 8_000), generate(family, 3, 8_000), generate(family, 4, 8_000)
    assert digest(a) == digest(b) and a.question == b.question and a.gold == b.gold
    assert digest(a) != digest(c)
    for tokens in (8_000, 32_000):
        it = generate(family, 1, tokens)
        assert 0.9 * tokens * CHARS_PER_TOKEN <= len(it.context) <= 1.3 * tokens * CHARS_PER_TOKEN
        assert it.score(it.gold_answer())["reward"] == 1.0 and it.score("")["reward"] == 0.0 and it.score(None)["reward"] == 0.0


def test_ported_scorers():
    assert oolong_score("[12]", "ANSWER_TYPE.NUMERIC", "12") == 1.0
    assert oolong_score("[12]", "ANSWER_TYPE.NUMERIC", "10") == pytest.approx(0.75 ** 2)
    assert oolong_score("[12]", "ANSWER_TYPE.NUMERIC", "Answer: 12") == 1.0  # attempt_answer_parse takes the text after the colon
    assert oolong_score("['more common than']", "", "more common") == 1.0 and oolong_score("['location']", "", "person") == 0.0
    assert mrcr_score("abcdefghijklHere is a poem", "abcdefghijklHere is a poem") == 1.0
    assert mrcr_score("abcdefghijklHere is a poem", "Here is a poem") == 0.0  # no hash prefix, no credit
    d = generate("docqa", 0, 8_000)
    right = json.loads(d.gold_answer())
    one_wrong = {**right, "1": "0"}
    s = d.score(json.dumps(one_wrong))
    assert 0 < s["reward"] < 1 and s["full_pass"] == 0.0 and s["answer_rate"] == 1.0
    assert d.score("not json")["status"] == ["no_answer"] * len(d.gold)


def test_longctx_task_states_everything_and_hides_the_gold():
    for fam in FAMILIES:
        t = LongCtxTask(fam, 2, 8_000)
        try:
            ins = t.instruction
            assert t.item.question.split("\n")[0][:40] in ins and "answer.submit(text)" in ins and t.item.answer_format in ins
            assert "docs.read(offset, n)" in ins and "docs.search(q)" in ins
            assert t.context_path.read_text() == t.item.context
            bundle = "".join(p.read_text(errors="replace") for p in t.root.rglob("*") if p.is_file())
            assert t.item.gold_answer() not in bundle or fam == "oolong"  # an oolong answer is a bare count or label
            if fam == "mrcr":
                assert t.item.gold[:12] not in bundle  # the hash prefix exists only in the question
        finally:
            t.cleanup()
    assert "CONTEXT.read" in LongCtxTask("oolong", 0, 8_000, delivery="context").instruction


@pytest.mark.parametrize("family", FAMILIES)
async def test_longctx_controls(family):
    r = await run_longctx_rung(family, 8_000, seed=1, kernel="local")
    assert r["solver"]["reward"] == 1.0 and r["gold"]["reward"] == 1.0 and r["noop"]["reward"] == 0.0 and r["controls_ok"]
    assert r["solver"]["steps"] >= 2  # read through the docs app, then submitted: graded from World state


# ---- fault injection ----------------------------------------------------------------------------------------------------------

class _Echo(ScriptedRLM):
    def decide(self, view):
        from crucible.runtime.scripted_rlm import Reply

        return Reply("text", "ok")


def test_fault_llm_overflow_crash_and_silent():
    msgs = convo(POLICY_OUT)
    f = FaultLLM(_Echo(), overflow_at=(1,), crash_at=3)
    asyncio.run(f.complete(msgs, tools=TOOLS))
    with pytest.raises(ContextOverflow):
        asyncio.run(f.complete(msgs, tools=TOOLS))
    asyncio.run(f.complete(msgs, tools=TOOLS))
    with pytest.raises(ConnectionError):
        asyncio.run(f.complete(msgs, tools=TOOLS))
    assert [x["fault"] for x in f.fired] == ["overflow", "crash"]
    s = FaultLLM(_Echo(), overflow_when=lambda v: v.role == "summary", overflow_times=2, silent=True, window=1000)
    for _ in range(3):
        r = asyncio.run(s.complete(msgs, tools=None))
    assert len(s.fired) == 2 and r.text == "ok"  # every summary attempt overflowed until the limit, then the model answered
    r0 = asyncio.run(FaultLLM(_Echo(), overflow_at=(0,), silent=True, window=1000).complete(msgs, tools=TOOLS))
    assert r0.raw_finish == "length" and r0.usage["prompt_tokens"] == 1000 and not r0.text


async def test_kernel_faults_fire():
    from crucible.actors.rlm.kernel import Kernel, LocalLauncher

    k = Kernel(LocalLauncher(allow_host=True), cell_timeout_s=3.0)
    try:
        await k.start()

        async def host(data):
            await kill_kernel(k)  # the kernel dies while the cell waits for this reply
            return {"ok": True}

        r = await k.execute("r = await host('world.call', tool='x', args={})\nprint('after')", host)
        assert not r.ok and "after" not in (r.stdout or "")
        await k.close()
        await k.start()
        r = await k.execute(PATHOLOGICAL_CELL, host)  # catastrophic backtracking: stopped by the cell deadline
        assert r.timed_out or r.killed
        await k.close()
        await k.start()
        r = await k.execute(BIG_OUTPUT_CELL, host)  # 50 MB printed: capped by the kernel, the host survives
        assert len(r.stdout or "") < 5_000_000
    finally:
        await k.close()


def test_pathological_ledger_search_is_bounded(tmp_path):
    ledger = pytest.importorskip("crucible.actors.rlm.ledger")
    from crucible.runtime.scripted_rlm import pathological_ledger_search

    out = pathological_ledger_search(ledger.Ledger(tmp_path))
    assert out["bounded"], out


async def test_crash_then_resume_and_kill_mid_cell(tmp_path):
    from crucible.train.chat import ByteChatCodec
    from crucible.train.tito_scripted import TitoLLM

    model = backlog_model("faithful")
    ref, fired = crash_then_resume_ref(lambda attempt: TitoLLM(model, ByteChatCodec()), crash_at_call=10)
    from crucible_pack_toy.longhorizon.ladder import _Capture, make_harness

    from crucible.runtime.run import Runner, RunSpec

    cfg = LadderConfig(rung=100, **{**LOCAL, "checkpoint_every": 20})
    h, _ = make_harness(cfg, _Capture())
    task = BacklogTask(0, RUNGS[100])
    try:
        s = await Runner(tmp_path / "c").run(RunSpec(run_id="c", tasks=[task], policies=[ref], harnesses=[h], k=1, rollout_mode="train"))
    finally:
        task.cleanup()
    assert fired and fired[0].fired[0]["fault"] == "crash"
    recs = [json.loads((d / "record.json").read_text()) for d in sorted((tmp_path / "c" / "c" / "samples").glob("*/a*"))]
    assert recs[0]["failure"]["kind"] == "agent_run_error" and len(recs) >= 2  # the lost worker is masked and retried
    assert any((r.get("meta") or {}).get("artifacts", {}).get("resumed_from") for r in recs[1:])  # resumed from the checkpoint
    assert s.mean_reward == 1.0
    k = await run_rung(LadderConfig(rung=100, faults={"kill_at_world_call": 30}, tito=None, **LOCAL), root=tmp_path / "k")
    assert k["faults"]["kills"][0]["fired"] and k["faults"]["kills"][0]["killed"] and k["reward"] == 1.0
