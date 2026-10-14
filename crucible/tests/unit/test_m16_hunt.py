"""M16 hunt: red-team of the very-long-horizon RLM harness (``actors/rlm/agent.py``, ``policy.py``, ``kernel.py``,
``repl.py``, ``runtime/harness/rlm.py``, ``runtime/record.py``). Each test failed on the tree before its fix (``Hn``)."""

from __future__ import annotations

import itertools
import json

from crucible.actors.llm import LLMResponse
from tests.unit.test_m16_f_agent import ScriptLLM, cell, cfg, episode, last_tool, text_of


def tools_out(p) -> list[str]:
    return [text_of(m) for m in p.messages if m.get("role") == "tool"]


# ---- H1: a history reply the ledger cannot fit is the cell's error, never a masked episode -------------------------------
async def test_h1_history_get_of_a_wide_unicode_world_call_is_not_masked(tmp_path):
    def decide(messages, tools, role):
        t = last_tool(messages)
        if "GOT" in t:
            return cell("rlm.finish('ok')", 3)
        if "CALLED" in t:
            return cell("try:\n    e = await history.search('tickets.nope', kinds=['world_call'])\n    g = await history.get(e[0]['i'])\n"
                        "    print('GOT', len(str(g)))\nexcept Exception as ex:\n    print('GOT error', type(ex).__name__)", 2)
        return cell("r = await world.call('tickets.nope', a='\\u4e00' * 60000)\nprint('CALLED', r['ok'])", 1)

    ep, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg())
    assert not ep.masked and p.failure_kind is None, (p.failure_kind, p.error)
    assert ep.stop_condition == "agent_done"


# ---- H2: least privilege is inherited: a child without the World cannot spawn a grandchild that has it ------------------------
async def test_h2_a_no_world_child_cannot_grant_the_world_to_a_grandchild(tmp_path):
    def decide(messages, tools, role):
        t = last_tool(messages)
        system = next(text_of(m) for m in messages if m.get("role") == "system")
        task = next(text_of(m) for m in messages if m.get("role") == "user")
        if role == "child" and task.startswith("GRAND"):
            if "LISTED" in t:
                return cell(f"rlm.finish({t.strip()[:200]!r})", 2)
            return cell("try:\n    r = await world.call('tickets.list_tickets', status='open')\n    print('LISTED ok', r['ok'])\n"
                        "except Exception as e:\n    print('LISTED refused')", 1)
        if role == "child":
            assert "no World access" in system
            if "GC" in t:
                return cell(f"rlm.finish({t.strip()[:300]!r})", 3)
            if "SPAWN" in t:
                return cell("g = await rlm.agent.result('g', wait_s=60)\nprint('GC', g['status'], g['answer'])", 2)
            return cell("try:\n    await rlm.agent.spawn('GRAND list the open tickets', name='g', world='shared')\n    print('SPAWN ok')\n"
                        "except Exception as e:\n    print('SPAWN refused GC', type(e).__name__)", 1)
        if "RES" in t:
            return cell("rlm.finish('done')", 3)
        if "SPAWNED" in t:
            return cell("a = await rlm.agent.result('c', wait_s=120)\nprint('RES', a['answer'])", 2)
        return cell("await rlm.agent.spawn('compute', name='c')\nprint('SPAWNED')", 1)

    ep, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg(tree_max_depth=2))
    res = [o for o in tools_out(p) if o.startswith("RES")]
    assert ep.steps == 1, "a grandchild of a no-World child stepped the World"  # only the final Done
    assert res and "LISTED ok" not in res[0], res


# ---- H3: a kernel the policy made unrestorable is its own (scored) problem, never a masked episode -------------------------------
BOMB = "class Bomb:\n    def __reduce__(self):\n        return (int, ('not a number',))\nbomb = Bomb()\nprint('PLANTED')"


async def test_h3_an_unloadable_snapshot_does_not_mask_the_episode(tmp_path):
    def decide(messages, tools, role):
        outs = tools_out_msgs(messages)
        k = len(outs)
        if k == 0:
            return cell("x = 1\nprint('ONE')", 0)
        if k == 1:
            return cell("import time\ntime.sleep(30)", 1)  # times out: restored from the cadence snapshot (journal offset > 0)
        if k == 2:
            return cell(BOMB, 2)
        if k == 3:
            return cell("import time\ntime.sleep(30)", 3)  # times out again: the snapshot now holds the bomb
        return cell("print('AFTER', 'x' in globals())\nrlm.finish('after the restore')", 4)

    ep, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg(snapshot_every=1, cell_timeout_s=1.0))
    assert not ep.masked and p.failure_kind is None, (p.failure_kind, p.error)
    assert ep.stop_condition == "agent_done"
    out = tools_out(p)
    assert any("AFTER" in o for o in out) and any("could not be restored" in o for o in out), out


def tools_out_msgs(messages) -> list[str]:
    return [text_of(m) for m in messages if m.get("role") == "tool"]


# ---- H10: a resume hides what the crashed attempt wrote past its checkpoint (history.* never shows an undone future) -----------
def _mark_model(attempt: int):
    """Each cell steps the World once and prints ``MARK-a<attempt>-<k>`` (k = tool results in its visible context); from k = 10
    it searches the ledger for every mark of attempt 1 and finishes. No hidden memory: k comes from the messages."""
    from crucible.runtime.scripted_rlm import Reply, ScriptedRLM

    class M(ScriptedRLM):
        name = "mark-rlm"

        def decide(self, view):
            if view.role == "summary":
                return Reply("summary", text="SUMMARY: marks")
            k = sum(1 for m in view.messages if m.get("role") == "tool")
            if k >= 10 and not any("FOUND" in text_of(m) for m in view.messages if m.get("role") == "tool"):
                return Reply("work", code="h = await history.search('MARK-a1-', kinds=['tool'], limit=100)\n"
                                          "print('FOUND', sorted({int(x['snippet'].split('MARK-a1-')[1].split()[0]) for x in h}))")
            if k >= 10:
                return Reply("work", code="rlm.finish('done')")
            return Reply("work", code=f"r = await world.call('tickets.list_tickets', status='open')\nprint('MARK-a{attempt}-{k} ', r['ok'])")

    return M()


async def test_h10_resume_hides_the_abandoned_ledger_events_and_stays_train_valid(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask

    from crucible.actors.policy import EpisodeLimits
    from crucible.actors.rlm.agent import RLMConfig
    from crucible.runtime.harness.rlm import RLMHarness
    from crucible.runtime.run import Runner, RunSpec
    from crucible.runtime.scripted_rlm import TitoLLM, crash_then_resume_ref
    from crucible.train.chat import ByteChatCodec

    ref, fired = crash_then_resume_ref(lambda attempt: TitoLLM(_mark_model(attempt.n), ByteChatCodec()), crash_at_call=9)
    h = RLMHarness(RLMConfig(window_tokens=200_000, max_tokens=4_000, kernel="local", allow_host=True, temperature=0.0),
                   limits=EpisodeLimits(max_steps=1_000, max_tool_errors=None), checkpoint_every=4)
    s = await Runner(tmp_path).run(RunSpec(run_id="c", tasks=[HelpdeskTask()], policies=[ref], harnesses=[h], k=1, rollout_mode="train"))
    assert fired and fired[0].fired[0]["fault"] == "crash"
    adirs = sorted((tmp_path / "c" / "samples").glob("*/a*"))
    assert len(adirs) == 2
    last = json.loads((adirs[-1] / "record.json").read_text())
    assert (last.get("meta") or {}).get("artifacts", {}).get("resumed_from")
    from crucible.core.schema.rollout import RolloutRecord

    ro = RolloutRecord.model_validate_json((adirs[-1] / "rollout.json").read_text())
    assert not ro.mask_sample and ro.valid_for_training() == (True, "ok"), (ro.stop, ro.failure_kind)
    # attempt 1 printed MARK-a1-0..7; the checkpoint covers only some of them; the resumed context shows those
    ck = json.loads((adirs[0] / "checkpoint.json").read_text())["policies"]["agent"]
    kept = sorted(int(text_of(m).split("MARK-a1-")[1].split()[0]) for m in ck["messages"] if m.get("role") == "tool" and "MARK-a1-" in text_of(m))
    printed = sorted({int(e["text"].split("MARK-a1-")[1].split()[0]) for p in (tmp_path / "c" / "samples").rglob("events.jsonl")
                      for e in map(json.loads, p.read_text().splitlines()) if e.get("kind") == "tool" and "MARK-a1-" in str(e.get("text"))})
    assert kept and set(printed) > set(kept), (printed, kept)  # attempt 1 wrote marks past its checkpoint
    events = [json.loads(x) for p in (tmp_path / "c" / "samples").rglob("events.jsonl") for x in p.read_text().splitlines()]
    out = [e.get("text", "") for e in events if e.get("kind") == "tool" and str(e.get("text", "")).startswith("FOUND")]
    assert out
    seen = json.loads(out[-1].split("FOUND", 1)[1].strip().splitlines()[0])
    # the checkpoint was taken inside cell len(kept) (waiting on its World call): the resume finishes that cell, so its mark is
    # this attempt's; attempt 1's later marks (max(printed)) belong to the undone future
    assert max(printed) > len(kept) and seen == [*kept, len(kept)], f"history showed abandoned events: {seen} (the checkpoint kept {kept})"
    assert s.mean_reward is not None


# ---- H4: CONTEXT search runs off the event loop, and its journal record replays to the same reply ------------------------------
def _ctx_policy(tmp_path, text: str):
    from crucible.actors.rlm.policy import RLMPolicy
    from tests.unit.test_m16_f_agent import local

    f = tmp_path / "doc.txt"
    f.write_text(text)
    return RLMPolicy("agent", ScriptLLM(lambda *a: None), launcher=local(), config=cfg(), context_path=f)


async def test_h4_a_pathological_context_regex_does_not_block_the_host_loop(tmp_path):
    import asyncio
    import time

    p = _ctx_policy(tmp_path, "a" * 40 + "!")
    ticks: list[float] = []
    stop = asyncio.Event()

    async def ticker():
        while not stop.is_set():
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)
        ticks.append(time.monotonic())

    t = asyncio.ensure_future(ticker())
    await asyncio.sleep(0.12)
    r = await p._host({"type": "context.search", "pattern": "(a+)+$", "regex": True, "limit": 5, "context_chars": 10})
    stop.set()
    await t
    gaps = [b - a for a, b in itertools.pairwise(ticks)]
    assert r.reply["timed_out"] and max(gaps) < 0.5, f"the host loop stalled {max(gaps):.2f} s on one CONTEXT.search"
    await p.aclose()


async def test_h4_a_context_search_replays_to_the_reply_the_cell_saw(tmp_path, monkeypatch):
    import subprocess

    import crucible.actors.rlm.agent as agent_mod

    p = _ctx_policy(tmp_path, "needle " * 50)
    real = subprocess.run
    calls = {"n": 0}

    def flaky(*a, **kw):  # the live call hits the 2 s bound (a loaded host); the replay does not
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(a[0], 2.0)
        return real(*a, **kw)

    monkeypatch.setattr(agent_mod.subprocess, "run", flaky)
    req = {"type": "context.search", "pattern": "need+le", "regex": True, "limit": 5, "context_chars": 10}
    live = await p._host(req)
    again = p._resolve_record(req, live.record)
    assert again == live.reply, (again, live.reply)
    req2 = {"type": "context.search", "pattern": "needle", "regex": False, "limit": 3, "context_chars": 5}
    live2 = await p._host(req2)
    assert p._resolve_record(req2, live2.record) == live2.reply
    await p.aclose()


# ---- H5 (campaign): real rollouts with compaction, children, sub-calls and a resume through runtime.record.build ------------------
def _tree_model(n_cells: int):
    """A root that counts cells in the kernel (each cell steps the World and prints 3,000 characters), spawns a child at cell 2,
    makes a sub-call batch at cell 3 and collects the child at cell 5; the child runs three cells and finishes."""
    from tests.unit.test_m16_f_agent import counter_model

    def extra(k: int) -> str:
        code = "await world.call('tickets.list_tickets', status='open')"
        if k == 2:
            code += "\nawait rlm.agent.spawn('count to three', name='c1')"
        if k == 3:
            code += "\nsubs = await rlm.batch(['say a', 'say b'])"
        if k == 5:
            code += "\nres = await rlm.agent.result('c1', wait_s=60)\nprint('CHILD', res['status'])"
        return code

    root = counter_model(n_cells, extra=extra)

    def decide(messages, tools, role):
        if role == "child":
            import re as _re

            m = _re.findall(r"\bM (\d+)\b", last_tool(messages))
            k = int(m[-1]) if m else 0
            if k >= 3:
                return cell("rlm.finish('child: three')", k)
            return cell("m = globals().get('m', 0) + 1\nprint('y' * 500)\nprint('M', m)", k)
        return root(messages, tools, role)

    return decide


def _tito(decide):
    from crucible.runtime.scripted_rlm import Reply, ScriptedRLM, TitoLLM
    from crucible.train.chat import ByteChatCodec
    from tests.unit.test_m16_f_agent import role_of

    class M(ScriptedRLM):
        name = "tree-rlm"

        def decide(self, view):
            r = decide(view.messages, [1] if view.role == "root" else None, role_of(view.messages, [1] if view.role == "root" else None))
            code = r.tool_calls[0]["arguments"]["code"] if r.tool_calls else None
            return Reply("work" if code else "text", text=r.text or "", code=code)

    return TitoLLM(M(), ByteChatCodec())


def _check_train_record(ro, *, sub_frozen: bool = False):
    from crucible.core.schema.rollout import full_prompts
    from crucible.train.bridges.common import flatten

    assert not ro.mask_sample and ro.valid_for_training() == (True, "ok"), (ro.stop, ro.failure_kind)
    ids = [t.call_id for t in ro.turns]
    assert len(ids) == len(set(ids)), "a model call appears twice in the record"
    seqs = flatten(ro, 1.0)
    spans = [cid for s in seqs for cid, _, _ in s.call_spans]
    assert sorted(spans) == sorted(ids), "a call is forwarded twice (or never)"
    # forwarded tokens = entered tokens: every chain call adds its prompt delta + completion once; a node its whole prompt
    prompts = full_prompts(ro)
    entered = sum(len(t.prompt_token_ids or []) + len(t.completion_token_ids or []) for t in ro.turns if t.completion_token_ids is not None)
    forwarded = sum(len(s.prompt_ids) + len(s.response_ids) for s in seqs)
    assert forwarded == entered, (forwarded, entered)
    trained = sum(sum(s.response_mask) for s in seqs)
    assert trained == sum(len(t.completion_token_ids or []) for t in ro.turns if t.trainable)
    comp = [t for t in ro.turns if t.kind == "compaction"]
    assert comp and all(t.trainable == (t.edge == "compaction_attempt") for t in comp)
    kids = [t for t in ro.turns if t.agent]
    assert kids and kids[0].depends_on, "the child's first call names the root turn that spawned it"
    subs = [t for t in ro.turns if t.kind == "sub"]
    assert subs and all(t.trainable is (not sub_frozen) for t in subs)
    assert all(prompts[t.call_id] for t in ro.turns if t.trainable)
    return seqs


async def test_h5_tree_rollout_with_compaction_children_subs_and_resume_is_train_valid(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask

    from crucible.actors.policy import EpisodeLimits
    from crucible.actors.rlm.agent import RLMConfig
    from crucible.core.schema.rollout import RolloutRecord
    from crucible.runtime.harness.rlm import RLMHarness
    from crucible.runtime.run import Runner, RunSpec
    from crucible.runtime.scripted_rlm import crash_then_resume_ref

    ref, fired = crash_then_resume_ref(lambda attempt: _tito(_tree_model(14)), crash_at_call=17)
    h = RLMHarness(RLMConfig(window_tokens=24_000, max_tokens=4_000, kernel="local", allow_host=True, temperature=0.0),
                   limits=EpisodeLimits(max_steps=1_000, max_tool_errors=None), checkpoint_every=3)
    await Runner(tmp_path).run(RunSpec(run_id="c", tasks=[HelpdeskTask()], policies=[ref], harnesses=[h], k=1, rollout_mode="train"))
    assert fired and fired[0].fired
    adirs = sorted((tmp_path / "c" / "samples").glob("*/a*"))
    assert len(adirs) == 2 and json.loads((adirs[-1] / "record.json").read_text()).get("meta", {}).get("artifacts", {}).get("resumed_from")
    ro = RolloutRecord.model_validate_json((adirs[-1] / "rollout.json").read_text())
    assert any(g.boundary_reason == "resume" for g in ro.segments) and any(g.boundary_reason == "compaction" for g in ro.segments)
    _check_train_record(ro)


async def test_h5_other_model_sub_calls_are_frozen_in_a_real_tree_rollout(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask

    from crucible.actors.policy import EpisodeLimits
    from crucible.actors.rlm.agent import RLMConfig
    from crucible.core.schema.rollout import RolloutRecord
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.harness.rlm import RLMHarness
    from crucible.runtime.run import Runner, RunSpec

    def make(attempt):
        llm = _tito(_tree_model(10))
        llm.sub_llm = _tito(lambda m, t, r: LLMResponse(text="other model"))
        return {"agent": llm}

    h = RLMHarness(RLMConfig(window_tokens=24_000, max_tokens=4_000, kernel="local", allow_host=True, temperature=0.0),
                   limits=EpisodeLimits(max_steps=1_000, max_tool_errors=None))
    await Runner(tmp_path).run(RunSpec(run_id="c", tasks=[HelpdeskTask()], policies=[PolicyRef(id="p", make=make)], harnesses=[h], k=1,
                                       rollout_mode="train"))
    adir = max((tmp_path / "c" / "samples").glob("*/a*"))
    _check_train_record(RolloutRecord.model_validate_json((adir / "rollout.json").read_text()), sub_frozen=True)


# ---- H6: the host's paths never reach the model (a restore's fallback reasons are readable through history) ------------------
async def test_h6_restore_fallback_reasons_do_not_expose_host_paths(tmp_path):
    import shutil

    def decide(messages, tools, role):
        outs = tools_out_msgs(messages)
        k = len(outs)
        if k == 0:
            return cell("x = 1\nprint('ONE')", 0)
        if k == 1:
            shutil.rmtree(tmp_path / "ledger" / "root" / ".snapshots", ignore_errors=True)  # the blob is gone (host side)
            return cell("import time\ntime.sleep(30)", 1)
        if k == 2:
            return cell("h = await history.search('deadline', kinds=['restore'])\nprint('SEEN', [await history.get(e['i']) for e in h])", 2)
        return cell("rlm.finish('ok')", 3)

    _, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg(snapshot_every=1, cell_timeout_s=1.0))
    seen = next(o for o in tools_out(p) if o.startswith("SEEN"))
    assert "fallback" in seen and str(tmp_path) not in seen and "/private/" not in seen, seen[:2000]


# ---- H9: the ledger has a disk cap; past it the policy's own outputs are cut (scored), World results stay whole -------------------
async def test_h9_the_ledger_disk_cap_bounds_what_cells_write(tmp_path):
    def decide(messages, tools, role):
        k = len(tools_out_msgs(messages))
        if k < 6:
            return cell(f"import random\nprint(''.join(random.choice('abcdef') for _ in range(400_000)))\nprint('K{k}')", k)
        if k == 6:
            return cell("r = await world.call('tickets.list_tickets', status='open')\nprint('W', len(str(r['content'])))", k)
        return cell("rlm.finish('ok')", k)

    ep, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg(ledger_max_bytes=1_000_000, window_tokens=2_000_000))
    assert not ep.masked and ep.stop_condition == "agent_done"
    on_disk = sum(f.stat().st_size for f in (tmp_path / "ledger").rglob("*") if f.is_file() and ".snapshots" not in f.parts)
    assert on_disk < 1_500_000, on_disk  # 6 x 400 kB of output would be 2.4 MB
    st = p.tree.ledger_budget.stats()
    assert st["cut_fields"] >= 3
    ev = [json.loads(x) for x in (tmp_path / "ledger" / "root" / "events.jsonl").read_text().splitlines()]
    wc = [e for e in ev if e["kind"] == "world_call"]
    assert wc and "not kept" not in json.dumps(wc)  # World results are never cut (a replay reads them back)


# ---- H7: a checkpoint taken during a compaction (a shared child's World step) never trains the summary the resume discards ---------
async def test_h7_a_checkpoint_mid_compaction_does_not_train_the_discarded_summary(tmp_path):
    from crucible.actors.rlm.policy import RLMPolicy
    from tests.unit.test_m16_f_agent import counter_model, local

    holder: dict = {}
    llm = ScriptLLM(counter_model(12))
    orig = RLMPolicy._compact

    async def spy(self, reason):
        # the runner checkpoints whenever any agent emits a World action: here, while the root's summary attempts are recorded
        # but before the context is rebuilt (the state a shared child's World step during a compaction would capture)
        real = self.tree.calls.append

        def append(call):
            out = real(call)
            if call.get("kind") == "compaction" and "state" not in holder:
                holder["state"] = json.loads(json.dumps(self.state_dict(), default=str))
            return out

        self.tree.calls.append = append
        try:
            return await orig(self, reason)
        finally:
            del self.tree.calls.append

    RLMPolicy._compact = spy
    try:
        await episode(tmp_path, llm, config=cfg())
    finally:
        RLMPolicy._compact = orig
    st = holder["state"]
    p2 = RLMPolicy("agent", llm, launcher=local(), config=cfg(), ledger_dir=tmp_path / "ledger")
    p2.load_state_dict(st)
    summaries = {s.get("summary_call") for s in p2.tree.segments}
    trained = [c for c in p2.model_calls if c["kind"] == "compaction" and c.get("trainable") is not False and c.get("accepted")]
    assert all(c["call_id"] in summaries for c in trained), "an accepted summary the resume never continued from is still trainable"
    await p2.aclose()


# ---- H8: a resumed cell that takes another path on replay is restored in the episode, never a second masked attempt ---------------
def _flag_model(attempt: int, flag):
    from crucible.runtime.scripted_rlm import Reply, ScriptedRLM

    class M(ScriptedRLM):
        name = "flag-rlm"

        def decide(self, view):
            if view.role == "summary":
                return Reply("summary", text="SUMMARY")
            k = sum(1 for m in view.messages if m.get("role") == "tool")
            if k >= 10:
                return Reply("work", code="rlm.finish('done')")
            pre = ""
            if k == 7:  # outside state the replay cannot reproduce: the second run of this cell asks the host for something else first
                pre = (f"import os\nif os.path.exists({str(flag)!r}):\n    await rlm.budget()\nopen({str(flag)!r}, 'w').close()\n"
                       "await history.stats()\n")
            return Reply("work", code=pre + f"r = await world.call('tickets.list_tickets', status='open')\nprint('MARK-a{attempt}-{k} ', r['ok'])")

    return M()


async def test_h8_a_diverging_resumed_cell_is_restored_not_masked(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask

    from crucible.actors.policy import EpisodeLimits
    from crucible.actors.rlm.agent import RLMConfig
    from crucible.core.schema.rollout import RolloutRecord
    from crucible.runtime.harness.rlm import RLMHarness
    from crucible.runtime.run import Runner, RunSpec
    from crucible.runtime.scripted_rlm import TitoLLM, crash_then_resume_ref
    from crucible.train.chat import ByteChatCodec

    flag = tmp_path / "flag"
    ref, fired = crash_then_resume_ref(lambda attempt: TitoLLM(_flag_model(attempt.n, flag), ByteChatCodec()), crash_at_call=9)
    h = RLMHarness(RLMConfig(window_tokens=200_000, max_tokens=4_000, kernel="local", allow_host=True, temperature=0.0),
                   limits=EpisodeLimits(max_steps=1_000, max_tool_errors=None), checkpoint_every=4)
    await Runner(tmp_path / "run").run(RunSpec(run_id="c", tasks=[HelpdeskTask()], policies=[ref], harnesses=[h], k=1, rollout_mode="train"))
    adirs = sorted((tmp_path / "run" / "c" / "samples").glob("*/a*"))
    assert fired and fired[0].fired and len(adirs) >= 2
    last = json.loads((adirs[1] / "record.json").read_text())
    assert (last.get("meta") or {}).get("artifacts", {}).get("resumed_from")
    ro = RolloutRecord.model_validate_json((adirs[1] / "rollout.json").read_text())
    assert not ro.mask_sample, (ro.stop, ro.failure_kind, last.get("failure"))
    assert ro.valid_for_training() == (True, "ok")


# ---- H11: a task that outlives its cell cannot act (or be credited) during a later cell ----------------------------------------
async def test_h11_a_stray_task_of_an_earlier_cell_is_refused_and_not_credited(tmp_path):
    def decide(messages, tools, role):
        k = len(tools_out_msgs(messages))
        if k == 0:
            return cell("import asyncio\nasync def bg():\n    await asyncio.sleep(0.4)\n    try:\n        await world.call('tickets.list_tickets', status='open')\n"
                        "        globals()['BG'] = 'served'\n    except Exception as e:\n        globals()['BG'] = repr(e)\n_t = asyncio.create_task(bg())\nprint('STARTED')", 0)
        if k == 1:
            return cell("import asyncio\nawait asyncio.sleep(1.0)\nprint('TWO', globals().get('BG'))", 1)
        return cell("rlm.finish('ok')", 2)

    ep, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg())
    assert ep.steps == 1, "the stray task's World call was stepped"  # the final Done only
    two = next(o for o in tools_out(p) if o.startswith("TWO"))
    assert "earlier python call" in two and "served" not in two, two
    calls = [c for c in p.model_calls if c["kind"] == "turn"]
    assert not any(a.startswith("tool:") for c in calls for a in c["actions"]), "the stray call was credited to a model call"
    assert p.counters["stray_requests"] == 1 and "refused" in two


# ---- H12: rlm.compact() spam: refused under two turns; a compaction that freed nothing takes no snapshot --------------------------
async def test_h12_compact_spam_is_rate_limited_and_takes_no_snapshot(tmp_path):
    def decide(messages, tools, role):
        if role == "summary":
            return LLMResponse(text="SUMMARY: spamming compact")
        k = len(tools_out_msgs(messages))
        if k >= 8:
            return cell("rlm.finish('ok')", k)
        return cell("r = await rlm.compact()\nprint('C', r)", k)

    _, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg(snapshot_every=200))
    c = p.counters
    assert c["compact_requests_refused"] >= 3 and any("refused: too soon" in o for o in tools_out(p)), c
    assert c["compactions"] >= 2 and c["compactions"] <= c["turns"] // 2 + 1, c
    assert c["snapshots"] == 0 and c["snapshots_skipped"] == c["compactions"], c  # nothing was freed: no snapshot per compaction


async def test_h12_a_compaction_that_frees_context_still_snapshots(tmp_path):
    from tests.unit.test_m16_f_agent import counter_model

    _, _, p = await episode(tmp_path, ScriptLLM(counter_model(12)), config=cfg())
    assert p.counters["compactions"] >= 1 and p.counters["snapshots"] == p.counters["compactions"], p.counters
