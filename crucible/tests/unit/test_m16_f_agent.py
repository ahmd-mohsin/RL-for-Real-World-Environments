"""M16 agent F: the integrated very-long-horizon RLM loop (``actors/rlm/agent.py``, ``policy.py``; design §3.6, §5).

Scripted models here are functions of the messages they are sent (no hidden memory), served by ``ScriptLLM``, whose usage
is 1.42 x the chars/4 estimate (the B1 ratio measured on RLM cells), so the usage-anchored meter is what keeps calls in
bounds. Cells run in the host kernel (trusted scripted code, opted in as the other RLM tests do).

- B1: no call (root, compaction, child) is sent above the compaction threshold; the summary request is bounded too.
- B2: a raised or silent overflow is discarded, compacted and retried without charging a turn; with compaction off it is
  the scored stop ``context_overflow`` (never masked); a failed compaction is the clean stop ``compaction_failed``
  (a truncation, every attempt recorded and untrained); a provider fault in a summary call masks with its kind.
- The ledger keeps every full output (the context names ``OUT(i)``), ``history`` reads are counted and journaled as small
  markers, World results are stored once and journaled as ledger indices (B5).
- Kernel faults: a kernel killed mid-cell is restored in the episode (the World actions it took are reported, not masked);
  a kernel lost between cells is restored and the cell run; a timed-out cell restores from a snapshot + suffix.
- B10: a World tool called by name gets the equivalent ``world.call`` line and no continuation is spent.
- The tree: children with and without a World grant, attribution of their calls and World steps, tree budgets, ``agent.*``
  only from cells.
- ``CONTEXT``, ``rlm.give_up``, ``rlm.compact``, ``rlm.budget``, the footer and the wrap-up notice; ``RLMConfig``."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from typing import Any

import pytest
from crucible_pack_toy.runtime import HelpdeskTask

from crucible.actors.llm import ContextOverflow, LLMError, LLMResponse
from crucible.actors.policy import EpisodeLimits, approx_tokens, run_episode
from crucible.actors.rlm.agent import RLMConfig, truncating_stop
from crucible.actors.rlm.kernel import LocalLauncher
from crucible.actors.rlm.policy import RLMPolicy

RATIO = 1.42  # real tokens per chars/4 token on RLM cells (03-crucible-stress.md B1)


def local() -> LocalLauncher:
    return LocalLauncher(allow_host=True)


def text_of(m: dict[str, Any]) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c, default=str)


def role_of(messages: list[dict[str, Any]], tools: Any) -> str:
    system = next((text_of(m) for m in messages if m.get("role") == "system"), "")
    if not system:
        return "sub"
    if not tools:
        return "summary"
    return "child" if "You are a child agent" in system else "root"


def cell(code: str, i: int = 0, text: str = "") -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[{"id": f"py{i}", "name": "python", "arguments": {"code": code}}])


class ScriptLLM:
    """``decide(messages, tools, role) -> LLMResponse``; usage = RATIO x chars/4 (prompt) of what it is sent. ``window``: a
    served model's window (a longer prompt raises ``ContextOverflow``); ``fail(role, n)`` may return an exception to raise."""

    model = "script"

    def __init__(self, decide: Callable[..., LLMResponse], *, window: int | None = None, fail: Callable[[str, int], BaseException | None] | None = None):
        self.decide, self.window, self.fail = decide, window, fail
        self.calls: list[tuple[str, int]] = []

    async def complete(self, messages, *, tools=None, max_tokens=2000, **kw):
        role = role_of(messages, tools)
        prompt = int(approx_tokens(messages) * RATIO)
        n = sum(1 for r, _ in self.calls if r == role)
        self.calls.append((role, prompt))
        exc = self.fail(role, n) if self.fail else None
        if exc is not None:
            raise exc
        if self.window and prompt > self.window:
            raise ContextOverflow(f"prompt of {prompt} tokens exceeds the {self.window}-token context window")
        r = self.decide(messages, tools, role)
        r.usage = {"prompt_tokens": prompt, "completion_tokens": max(1, int(approx_tokens([{"role": "assistant", "content": r.text,
                                                                                            "tool_calls": r.tool_calls}]) * RATIO))}
        r.model = self.model
        return r


def last_tool(messages: list[dict[str, Any]]) -> str:
    return next((text_of(m) for m in reversed(messages) if m.get("role") == "tool"), "")


def counter_model(n_cells: int, *, out_chars: int = 3000, finish: str = "rlm.finish('done')", extra: Callable[[int], str] | None = None,
                  summary: str = "SUMMARY: counting cells; n is in the kernel.") -> Callable[..., LLMResponse]:
    """A root that runs ``n_cells`` printing cells (its count lives in the kernel and is read back from the last output)."""

    def decide(messages, tools, role):
        if role == "summary":
            return LLMResponse(text=summary)
        if role == "sub":
            return LLMResponse(text="sub answer")
        m = re.findall(r"\bN (\d+)\b", last_tool(messages))
        k = int(m[-1]) if m else 0
        if k >= n_cells:
            return cell(finish, k)
        pre = extra(k + 1) if extra else ""
        return cell(f"{pre}\nn = globals().get('n', 0) + 1\nprint('x' * {out_chars})\nprint('N', n)", k)

    return decide


async def episode(tmp_path, llm, *, task=None, limits=None, **kw):
    task = task or HelpdeskTask()
    w = await task.open(tmp_path / "work")
    kw.setdefault("launcher", local())
    p = RLMPolicy("agent", llm, ledger_dir=tmp_path / "ledger", **kw)
    try:
        ep = await run_episode(w, {"agent": p}, limits=limits or EpisodeLimits(max_steps=500))
        rep = await task.grade(w, tmp_path / "grade")
        return ep, rep, p
    finally:
        await w.close()
        await p.aclose()


def cfg(**kw) -> RLMConfig:
    base = {"temperature": 0.0, "kernel": "local", "allow_host": True, "window_tokens": 12_000, "max_tokens": 2_000, "cell_timeout_s": 20.0}
    return RLMConfig(**{**base, **kw})


# ---- B1: no call above the threshold, root, compaction and child ------------------------------------------------------------
async def test_b1_no_call_is_sent_above_the_threshold(tmp_path):
    def child(messages, tools, role):
        m = re.findall(r"\bM (\d+)\b", last_tool(messages))
        k = int(m[-1]) if m else 0
        if k >= 12:
            return cell("rlm.finish('child answer: twelve')", k)
        return cell("m = globals().get('m', 0) + 1\nprint('y' * 3000)\nprint('M', m)", k)

    root = counter_model(30, extra=lambda k: ("await rlm.agent.spawn('count to twelve', name='c1')" if k == 3 else
                                              "r = await rlm.agent.result('c1', wait_s=60)\nprint('CHILD', r['status'])" if k == 6 else ""))

    def decide(messages, tools, role):
        return child(messages, tools, role) if role == "child" else root(messages, tools, role)

    llm = ScriptLLM(decide)
    ep, _, p = await episode(tmp_path, llm, config=cfg())
    th = p.context_spec.threshold()
    assert ep.stop_condition == "agent_done" and not ep.masked, (ep.stop_condition, ep.failures)
    roles = {r for r, _ in llm.calls}
    assert {"root", "child", "summary"} <= roles, roles
    over = [(r, n) for r, n in llm.calls if n > th]
    assert not over, (th, over)  # B1: 1.42x the estimate, and still no call above the threshold
    assert p.counters["compactions"] >= 2 and p.tree.agents["root/c1"].counters["compactions"] >= 1
    assert p.context_health()["largest_prompt_over_threshold"] <= 1.0
    # child calls are their own chain: agent path, depth, the spawning root turn as the first call's parent
    kids = [c for c in p.model_calls if c.get("agent") == "c1"]
    assert kids and kids[0]["edge"] == "subagent_call" and kids[0]["depth"] == 1
    spawner = next(c for c in p.model_calls if c["call_id"] == kids[0]["parent"])
    assert spawner.get("agent") == "" and spawner["kind"] == "turn"
    usage = p.tree.supervisor.usage()
    assert usage["by_depth"][1]["turns"] == sum(1 for c in kids if c["kind"] == "turn") and usage["spawns"] == 1


# ---- B2: overflow ----------------------------------------------------------------------------------------------------------
async def test_b2_raised_overflow_compacts_and_retries_without_a_turn(tmp_path):
    over = {"n": 0}

    def fail(role, n):
        if role == "root" and n == 6 and not over["n"]:
            over["n"] += 1
            return ContextOverflow("prompt exceeds the context window (injected)")
        return None

    llm = ScriptLLM(counter_model(10), fail=fail)
    ep, _, p = await episode(tmp_path, llm, config=cfg(window_tokens=128_000))
    assert ep.stop_condition == "agent_done" and not ep.masked and p.failure_kind is None
    assert p.counters["overflow_retries"] == 1 and p.counters["compactions"] == 1
    assert p.tree.supervisor.usage()["turns"] == p.counters["turns"]  # the overflowed attempt and its retry are one turn
    assert p.segments[-1]["reason"] == "overflow" and p.segments[-1]["boundary_reason"] == "compaction"


async def test_b2_silent_overflow_is_discarded_and_not_recorded(tmp_path):
    seen = {"n": 0}

    def decide(messages, tools, role):
        if role == "root" and seen["n"] == 0 and len([m for m in messages if m.get("role") == "tool"]) == 4:
            seen["n"] += 1
            return LLMResponse(text="", raw_finish="length")
        return counter_model(8)(messages, tools, role)

    class Silent(ScriptLLM):
        async def complete(self, messages, **kw):
            r = await super().complete(messages, **kw)
            if r.raw_finish == "length" and not r.text:
                r.usage = {"prompt_tokens": 12_000, "completion_tokens": 0}  # the provider's silent refusal: the window, no output
            return r

    ep, _, p = await episode(tmp_path, Silent(decide), config=cfg())
    assert ep.stop_condition == "agent_done" and p.counters["overflow_retries"] == 1
    assert not [c for c in p.model_calls if c.get("finish") == "length"]  # the discarded reply is not a trajectory call


async def test_b2_overflow_with_compaction_off_is_a_scored_stop(tmp_path):
    llm = ScriptLLM(counter_model(30), window=2_500)
    ep, _, p = await episode(tmp_path, llm, config=cfg(budget_tokens=0, window_tokens=0))
    assert p.stop == "context_overflow" and ep.stop_condition == "context_overflow"
    assert not ep.masked and p.failure_kind is None and p.error is None  # scored, never masked (B2)
    assert ep.truncated and truncating_stop(p.stop)


async def test_compaction_failure_is_a_clean_truncation_with_every_attempt_recorded(tmp_path):
    def decide(messages, tools, role):
        if role == "summary":
            return LLMResponse(text="")  # empty summaries: rejected and resampled until the attempts run out
        return counter_model(40)(messages, tools, role)

    ep, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg(max_compaction_attempts=3))
    assert p.stop == "compaction_failed" and ep.stop_condition == "compaction_failed" and not ep.masked and p.failure_kind is None
    tries = [c for c in p.model_calls if c["kind"] == "compaction"]
    assert len(tries) == 3 and all(c["edge"] == "compaction_attempt" and c["accepted"] is False and c["trainable"] is False for c in tries)
    seg = p.segments[-1]
    assert seg["failed"] and seg["boundary_reason"] == "compaction_failed" and seg["attempts"] == 3
    assert p.counters["compaction_failures"] == 1


async def test_a_provider_fault_in_a_summary_call_masks(tmp_path):
    def fail(role, n):
        return LLMError("provider 503") if role == "summary" else None

    ep, _, p = await episode(tmp_path, ScriptLLM(counter_model(40), fail=fail), config=cfg())
    assert ep.masked and p.failure_kind == "provider_unavailable" and p.stop == "policy_error"


async def test_accepted_compaction_is_trained_rejected_ones_are_not(tmp_path):
    n = {"k": 0}

    def decide(messages, tools, role):
        if role == "summary":
            n["k"] += 1
            return LLMResponse(text="" if n["k"] == 1 else "SUMMARY: second try")  # the first attempt is rejected
        return counter_model(12)(messages, tools, role)

    _, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg())
    comp = [c for c in p.model_calls if c["kind"] == "compaction"]
    assert all(c["edge"] == "compaction_attempt" for c in comp) and not comp[0]["accepted"] and comp[1]["accepted"] and comp[1]["trainable"]
    seg = next(s for s in p.segments if s["boundary_reason"] == "compaction")
    assert seg["summary_call"] == comp[1]["call_id"] and seg["attempts"] == 2 and seg["tokens_source"] in ("usage", "estimate")
    after = [c for c in p.model_calls if c["kind"] == "turn" and c["segment"] == seg["index"]]
    assert after and after[0]["edge"] == "compaction"  # D's contract: the first turn after an accepted summary


# ---- the ledger ----------------------------------------------------------------------------------------------------------
async def test_full_output_goes_to_the_ledger_and_history_pages_it_back(tmp_path):
    big = "print(''.join(f'{i:06d}|' for i in range(4000)))"  # 28,000 chars: cut in the context

    def decide(messages, tools, role):
        t = last_tool(messages)
        if "PAGE-END" in t:
            return cell("rlm.finish('read')", 9)
        cut = re.search(r"full output: await OUT\((\d+), cursor=(\d+)\)", t)
        if cut:
            return cell(f"pages, c = [], 0\nwhile c is not None:\n    p = await OUT({cut.group(1)}, cursor=c, max_chars=6000)\n"
                        "    pages.append(p['text'])\n    c = p['next']\ntxt = ''.join(pages)\nprint(len(txt), txt[-14:], 'PAGE-END')", 2)
        if "000000|" not in t:
            return cell(big, 1)
        return cell("rlm.finish('?')", 3)

    task = HelpdeskTask()
    _, _, p = await episode(tmp_path, ScriptLLM(decide), task=task, config=cfg(window_tokens=64_000))
    outs = [text_of(m) for m in p.messages if m.get("role") == "tool"]
    assert "characters truncated here; full output: await OUT(" in outs[0] and len(outs[0]) <= 8192 + 50
    assert outs[1].startswith("28000 003998|003999| PAGE-END")  # the whole output, back from the ledger in pages
    assert p.counters["history_reads"] >= 5
    # the journal holds < 1 KiB markers for history replies, not the replies (B5)
    for e in p.kernel.journal:
        for rep in e.replies:
            if str((rep.get("request") or {}).get("type", "")).startswith("history."):
                assert "record" in rep and "reply" not in rep and len(json.dumps(rep)) < 1024


async def test_world_results_are_stored_once_and_journaled_as_ledger_indices(tmp_path):
    def decide(messages, tools, role):
        if "OPEN" in last_tool(messages):
            return cell("rlm.finish('listed')", 2)
        return cell("r = await world.call('tickets.list_tickets', status='open')\nprint('OPEN', len(r['content']), len(OBS))", 1)

    ep, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg())
    assert ep.steps == 2  # one World call + the final Done
    from crucible.actors.rlm.ledger import Ledger

    led = Ledger.open(tmp_path / "ledger", name="root")
    ev = [led.get(i) for i in range(len(led)) if led.get(i)["kind"] == "world_call"]
    assert len(ev) == 1 and ev[0]["tool"] == "tickets.list_tickets" and ev[0]["content"]
    rep = next(r for e in p.kernel.journal for r in e.replies if r["request"]["type"] == "world.call")
    assert rep["record"] == {"world": ev[0]["i"]} and "reply" not in rep  # the journal names the ledger event (B5)


# ---- kernel faults -------------------------------------------------------------------------------------------------------
async def test_kernel_killed_mid_cell_is_restored_and_the_episode_continues(tmp_path):
    from crucible.runtime.scripted_rlm import kill_mid_cell

    def decide(messages, tools, role):
        t = last_tool(messages)
        if "AFTER" in t:
            return cell("rlm.finish('ok')", 4)
        if "World action(s) it took still happened" in t:
            return cell("print('AFTER', keep)", 3)
        if "KEEP" in t:
            return cell("x = await world.call('tickets.list_tickets', status='open')\ny = await world.call('tickets.get_ticket', ticket_id=1)\nprint('NO')", 2)
        return cell("keep = 41\nprint('KEEP')", 1)

    task = HelpdeskTask()
    w = await task.open(tmp_path / "work")
    p = RLMPolicy("agent", ScriptLLM(decide), launcher=local(), config=cfg(), ledger_dir=tmp_path / "ledger")
    rec = kill_mid_cell(p, at_world_call=2)
    try:
        ep = await run_episode(w, {"agent": p}, limits=EpisodeLimits(max_steps=50))
    finally:
        await w.close()
        await p.aclose()
    assert rec["fired"] and rec["killed"]
    assert ep.stop_condition == "agent_done" and not ep.masked, (ep.stop_condition, ep.failures)  # E's fault: no longer masked
    outs = [text_of(m) for m in p.messages if m.get("role") == "tool"]
    assert "the kernel process died while this cell ran" in outs[1] and "2 World action(s) it took still happened" in outs[1]
    assert outs[2].startswith("AFTER 41")  # restored to before the cell, then the episode went on
    assert p.counters["killed_mid_cell"] == 1 and p.counters["restores"] == 1 and ep.steps == 3


async def test_kernel_lost_between_cells_is_restored_and_the_cell_runs(tmp_path):
    from crucible.runtime.scripted_rlm import kill_kernel

    p_ref: dict[str, Any] = {}

    def decide(messages, tools, role):
        t = last_tool(messages)
        if "SECOND" in t:
            return cell("rlm.finish('ok')", 3)
        if "FIRST" in t:
            return cell("print('SECOND', keep + 1)", 2)
        return cell("keep = 41\nprint('FIRST')", 1)

    class Killing(ScriptLLM):
        async def complete(self, messages, **kw):
            if "FIRST" in last_tool(messages) and not p_ref.get("killed"):
                p_ref["killed"] = await kill_kernel(p_ref["p"].kernel)
                await asyncio.sleep(0.2)
            return await super().complete(messages, **kw)

    task = HelpdeskTask()
    w = await task.open(tmp_path / "work")
    p = p_ref["p"] = RLMPolicy("agent", Killing(decide), launcher=local(), config=cfg(), ledger_dir=tmp_path / "ledger")
    try:
        ep = await run_episode(w, {"agent": p}, limits=EpisodeLimits(max_steps=50))
    finally:
        await w.close()
        await p.aclose()
    assert p_ref["killed"] and ep.stop_condition == "agent_done"
    out = [text_of(m) for m in p.messages if m.get("role") == "tool"][1]
    assert out.startswith("SECOND 42") and "the kernel was lost before this cell ran (its code did not run)" in out
    assert p.counters["lost_between_cells"] == 1


async def test_a_timed_out_cell_restores_from_the_snapshot_and_the_suffix(tmp_path):
    def decide(messages, tools, role):
        t = last_tool(messages)
        if "VAL" in t:
            return cell("rlm.finish('ok')", 9)
        if "deadline" in t:
            return cell("print('VAL', a + b + c)", 8)
        k = len([m for m in messages if m.get("role") == "tool"])
        if k < 4:
            return cell(f"{'abcd'[k]} = {k + 1}", k)
        return cell("c = 100\nimport time\nwhile True:\n    time.sleep(0.05)", 7)

    ep, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg(snapshot_every=2, cell_timeout_s=1.0))
    out = [text_of(m) for m in p.messages if m.get("role") == "tool"]
    assert ep.stop_condition == "agent_done" and out[-2].startswith("VAL 6")  # c's partial effect rolled back
    assert p.counters["snapshots"] >= 2 and p.health["restores_by_mode"] == {"snapshot": 1}, p.health["restores_by_mode"]
    assert "the kernel state was restored to before this cell" in out[-3]


# ---- B10: a World tool called by name --------------------------------------------------------------------------------------
async def test_b10_a_world_tool_called_by_name_gets_the_code_line_and_no_continuation(tmp_path):
    def decide(messages, tools, role):
        t = last_tool(messages)
        if "OK-LISTED" in t:
            return cell("rlm.finish('ok')", 3)
        m = re.search(r"r = await world\.call\('tickets\.list_tickets', \*\*(\{.*\})\)", t)
        if m:  # the model takes the correction's line
            return cell(f"r = await world.call('tickets.list_tickets', **{m.group(1)})\nprint('OK-LISTED', r['ok'])", 2)
        return LLMResponse(text="", tool_calls=[{"id": "t1", "name": "tickets.list_tickets", "arguments": {"status": "open"}}])

    ep, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg())
    assert ep.stop_condition == "agent_done" and p.counters["misrouted_calls"] == 1 and p.counters["continuations"] == 0
    first = next(text_of(m) for m in p.messages if m.get("role") == "tool")
    assert "is a World tool" in first and "r = await world.call('tickets.list_tickets', **{\"status\": \"open\"})" in first
    assert "World tools are called from code, never as tools" in p.system and "A first cell:" in p.system


async def test_b10_repeated_misrouting_ends_as_a_scored_stall(tmp_path):
    def decide(messages, tools, role):
        return LLMResponse(text="", tool_calls=[{"id": "t1", "name": "tickets.list_tickets", "arguments": {"status": "open"}}])

    ep, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg(stall_stop_after=2))
    assert p.stop == "stalled" and ep.stop_condition == "stalled" and not ep.masked and p.counters["continuations"] == 0


# ---- the tree ------------------------------------------------------------------------------------------------------------
async def test_children_world_grants_attribution_and_agent_requests_only_from_cells(tmp_path):
    def decide(messages, tools, role):
        t = last_tool(messages)
        if role == "child":
            system = next(text_of(m) for m in messages if m.get("role") == "system")
            task = next(text_of(m) for m in messages if m.get("role") == "user")
            if "DONE-" in t:
                return cell(f"rlm.finish({t.strip()!r})", 2)
            if "no World access" in system:
                return cell("try:\n    await world.call('tickets.list_tickets', status='open')\n    print('DONE-ALLOWED')\nexcept Exception as e:\n"
                            "    print('DONE-REFUSED', 'no World access' in str(e))", 1)
            assert "list the open tickets" in task
            return cell("r = await world.call('tickets.list_tickets', status='open')\nprint('DONE-LISTED', r['ok'])", 1)
        if "RESULTS" in t:
            return cell("rlm.finish('children done')", 3)
        if "SPAWNED" in t:
            return cell("a = await rlm.agent.result('nw', wait_s=60)\nb = await rlm.agent.result('sw', wait_s=60)\nprint('RESULTS', a['answer'], b['answer'])", 2)
        return cell("await rlm.agent.spawn('compute something', name='nw')\nawait rlm.agent.spawn('list the open tickets', name='sw', world='shared')\n"
                    "print('SPAWNED')", 1)

    ep, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg())
    out = [text_of(m) for m in p.messages if m.get("role") == "tool"]
    assert ep.stop_condition == "agent_done" and "DONE-REFUSED True" in out[1] and "DONE-LISTED True" in out[1]
    assert ep.steps == 2  # the shared child's World call was a World step (+ the final Done)
    sw = [c for c in p.model_calls if c.get("agent") == "sw" and c["kind"] == "turn"]
    assert any(a.startswith("tool:") for a in sw[0]["actions"])  # attributed to the child's model call
    assert p.tree.supervisor.usage()["spawns"] == 2 and p.counters["agent_requests"] == 4
    assert (tmp_path / "ledger" / "root" / "sw" / "events.jsonl").exists()  # the child's own ledger
    with pytest.raises(PermissionError):  # agent.* outside a cell
        await p._host({"type": "agent.list"})


async def test_sub_calls_charge_the_tree_budget(tmp_path):
    def decide(messages, tools, role):
        if role == "sub":
            return LLMResponse(text="s")
        t = last_tool(messages)
        if "SUBS" in t:
            return cell("rlm.finish('ok')", 2)
        return cell("try:\n    await rlm.batch(['q'] * 5)\n    print('SUBS ok')\nexcept Exception as e:\n    print('SUBS refused', e)", 1)

    _, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg(tree_max_sub_calls=3))
    assert "SUBS refused" in next(text_of(m) for m in p.messages if m.get("role") == "tool")
    _, _, p2 = await episode(tmp_path / "b", ScriptLLM(decide), config=cfg())
    assert p2.tree.supervisor.usage()["sub_calls"] == 5 and p2.tree.supervisor.usage()["sub_tokens"] > 0
    subs = [c for c in p2.model_calls if c["kind"] == "sub"]
    assert len(subs) == 5 and all(c["edge"] == "sub" and c["parent"] and c["agent"] == "" for c in subs)


# ---- CONTEXT, give_up, compact, budget, footer, wrap-up -------------------------------------------------------------------------
async def test_context_handle_reads_and_searches_a_host_file(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("alpha " * 5000 + "NEEDLE-42 " + "omega " * 5000)

    def decide(messages, tools, role):
        t = last_tool(messages)
        if "FOUND" in t:
            return cell("rlm.finish('found')", 2)
        return cell("i = CONTEXT.info()\nh = await CONTEXT.search('NEEDLE')\nx = await CONTEXT.read(h['items'][0]['offset'], 9)\n"
                    "print('FOUND', i['chars'], x)", 1)

    _, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg(), context_path=doc)
    assert "CONTEXT" in p.system and "[context] CONTEXT holds 60,010 characters" in text_of(p.messages[1])
    assert "FOUND 60010 NEEDLE-42" in next(text_of(m) for m in p.messages if m.get("role") == "tool")
    reps = [r for e in p.kernel.journal for r in e.replies if r["request"]["type"].startswith("context.")]
    assert reps and all("record" in r and "reply" not in r for r in reps) and p.counters["context_reads"] == 2


async def test_give_up_is_a_recorded_scored_stop(tmp_path):
    ep, _, p = await episode(tmp_path, ScriptLLM(lambda m, t, r: cell("rlm.give_up('the order system is down', evidence='503 x3')")), config=cfg())
    assert p.stop == "agent_gave_up" and ep.stop_condition == "agent_gave_up" and not ep.masked and p.failure_kind is None
    assert "the order system is down" in p.stop_reason


async def test_rlm_compact_budget_footer_and_wrap_up(tmp_path):
    def decide(messages, tools, role):
        if role == "summary":
            return LLMResponse(text="SUMMARY: asked for a checkpoint")
        t = last_tool(messages)
        if "BUDGET" in t:
            return cell("rlm.finish('ok')", 3)
        if "ASKED" in t:
            return cell("b = await rlm.budget()\nprint('BUDGET', sorted(b), b['context']['threshold'])", 2)
        return cell("print('ASKED', await rlm.compact(instructions='keep the ids'))", 1)

    _, _, p = await episode(tmp_path, ScriptLLM(decide), config=cfg(max_turns=3, footer_from=0.0))
    assert p.counters["compactions"] == 1 and p.segments[0]["reason"] == "request"
    outs = [text_of(m) for m in p.messages if m.get("role") == "tool"]
    assert "BUDGET ['context', 'new_tokens', 'sub_calls', 'tree_turns', 'turns']" in outs[-2]
    assert all("<context_usage>" in o for o in outs)  # footer_from=0: every observation
    assert any("[budget: turns]" in text_of(m) for m in p.messages if m.get("role") == "user")  # 0.9 of max_turns=3


# ---- RLMConfig ----------------------------------------------------------------------------------------------------------
def test_config_aliases_digest_and_notes():
    a, b = RLMConfig(), RLMConfig(snapshot_every=200)
    assert a.digest() == b.digest()  # an M16 field at its default never changes the id
    assert RLMConfig(window_tokens=64_000).digest() != a.digest()
    c = RLMConfig(budget_tokens=32_000, trigger=0.7, keep_recent_turns=5, max_stalls=9, max_sub_calls=12, max_tokens=50_000)
    assert c.context_spec().window_tokens == 32_000 and c.stall_kwargs()["stop_after"] == 9 and c.tree_limits().max_sub_calls == 12
    assert c.max_out() == 8_000 and len(c.notes()) == 3  # trigger and keep_recent_turns ignored, max_tokens clamped: recorded
    assert RLMConfig(budget_tokens=0).mode() == "none" and RLMConfig().agent_max_turns() == 20_000  # B6: thousands, not 500
    json.dumps(a.__dict__)
    with pytest.raises(ValueError):
        RLMConfig(context_mode="sometimes")


async def test_a_context_loop_without_world_progress_is_a_scored_stall(tmp_path):
    """M16 validation: a model that re-read its past after every compaction and paged it back filled the window again,
    2,814 compactions at the 5,000 rung, each round's cells different (no cell-level stall signature). A World-acting root
    whose compactions keep coming with no World action between them is stopped as ``stalled``."""
    first = "await world.call('tickets.list_tickets', status='open')"
    ep, _, p = await episode(tmp_path, ScriptLLM(counter_model(400, extra=lambda k: first if k == 1 else "")), config=cfg())
    assert p.stop == "stalled" and ep.stop_condition == "stalled" and not ep.masked and "no World action between them" in p.stop_reason
    assert p.counters["compactions"] == 7 and ep.steps == 2
