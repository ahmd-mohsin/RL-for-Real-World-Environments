"""m13 C8: the full trigger language in Crucible's step-mode scheduler, with agentenv's semantics: ``where`` predicates,
``${args.*}`` / ``${result.*}`` / ``${binding}`` templating, state triggers (check pipelines with bindings), seeded
stochastic ``every`` that reproduces after snapshot/restore, ``after_event`` + ``offset`` anchoring, ``until``, per-action
``verify`` and the provoking-call barrier. Every spec translates to a trigger agentenv's own engine accepts."""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.core.triggers import TemplateError, apply_extract, eval_predicate, template
from crucible.world.world import World


def test_helpers_match_agentenv():
    assert eval_predicate({"regex": "^Q[0-9]"}, "Q3 report") and not eval_predicate({"equals": 3}, "3") and eval_predicate({"exists": False}, [])
    data = {"messages": [{"id": 1, "subject": "hi"}, {"id": 7, "subject": "report"}]}
    assert apply_extract({"path": "messages", "match": {"subject": {"regex": "rep"}}, "take": "id"}, data, "") == 7
    assert apply_extract("messages[1].subject", data, "") == "report"
    ctx = {"args": {"to": ["a@x"], "subject": "Q3"}, "result": {"id": 5}}
    assert template({"to": "${args.to}", "subject": "Re: ${args.subject} #${result.id}", "cc": "${args.cc?}"}, {}, ctx) == {"to": ["a@x"], "subject": "Re: Q3 #5"}
    assert template("${mid}", {"mid": 9}) == "9" and template("${unknown}", {}) == "${unknown}"
    with pytest.raises(TemplateError):
        template({"x": "${args.missing}"}, {}, ctx)


def _world(events, tmp_path: Path, *, clock_mode="step") -> World:
    spec = WorldSpec.model_validate({
        "world": "test/triggers", "network": "no-network",
        "clock": {"mode": clock_mode, "t0": "2026-03-02T09:00:00Z", "step_dt_s": 60, "rate": 3600},
        "apps": [{"id": "mail", "template": "common/mailbox", "state": {"adapter": "sql", "params": {"path": "mail.db"}}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["mail.*"]}, {"id": "boss", "kind": "system"}, {"id": "ops", "kind": "system"}],
        "events": events,
    })
    return World(spec, store=Store(tmp_path / "s"), workdir=tmp_path / "w")


async def send(w, to, subject, body="x"):
    r = await w.step("agent", ToolCall(tool_call_id=f"c{time.monotonic_ns()}", function_name="mail.send_email", arguments={"to": to, "subject": subject, "body": body}))
    return r.observation.tool_results[0]


async def inbox(w, who="agent"):
    import json

    return json.loads((await w._call_tool(who, ToolCall(tool_call_id="q", function_name="mail.list_inbox", arguments={}), system=True)).content)["messages"]


ACK = {"id": "ack", "actor": "boss", "when": {"on_tool": "mail.send_email", "repeat": True, "where": {"args.subject": {"regex": "^Q[0-9] report"}}},
       "action": {"type": "tool", "tool": "mail.send_email", "args": {"to": ["agent@world.local"], "subject": "Re: ${args.subject} (id ${result.id})", "body": "got it"},
                  "verify": {"steps": [{"tool": "mail.search_email", "args": {"query": "Re: ${args.subject}"}, "extract": "messages[0].subject", "predicate": {"exists": True}}]}},
       "barrier": "provoking_call"}
STATE = {"id": "noticed", "actor": "boss",
         "when": {"state": {"steps": [{"tool": "mail.list_inbox", "extract": {"path": "messages", "match": {"subject": {"regex": "budget"}}, "take": "id"}, "bind": "mid",
                                       "predicate": {"exists": True}}]}},
         "action": {"type": "tool", "tool": "mail.send_email", "args": {"to": ["agent@world.local"], "subject": "Budget mail #${mid} noticed", "body": "ok"}}}
FOLLOW = {"id": "follow_up", "actor": "boss", "when": {"after_event": "noticed", "offset": "10m"},
          "action": {"type": "nl", "to": "agent", "text": "Any update on the budget?"}}


async def test_where_templating_verify_and_barrier(tmp_path):
    w = _world([ACK], tmp_path)
    await w.start()
    try:
        await w.reset()
        assert (await send(w, ["boss@world.local"], "lunch?")).ok
        assert [m["subject"] for m in await inbox(w)] == []  # where did not match
        r = await send(w, ["boss@world.local"], "Q3 report")
        # barrier / step semantics: the provoking call's step returns after the action ran, so the agent sees it now
        assert [m["subject"] for m in await inbox(w)] == [f"Re: Q3 report (id {__import__('json').loads(r.content)['id']})"]
        verify = [e for e in w.log if e.type == "event_verify"]
        assert verify and verify[-1].data == {"event": "ack", "ok": True}
        assert [f["event"] for f in w.firing_log()] == ["ack"]
    finally:
        await w.close()


async def test_state_trigger_bindings_and_after_event_anchor(tmp_path):
    w = _world([STATE, FOLLOW], tmp_path)
    await w.start()
    try:
        await w.reset()
        await send(w, ["ops@world.local"], "hello")
        assert w.firing_log() == []
        await send(w, ["boss@world.local"], "budget draft")
        log = w.firing_log()
        assert [f["event"] for f in log] == ["noticed"]
        assert any(m["subject"].startswith("Budget mail #") and m["subject"].endswith(" noticed") for m in await inbox(w))
        from crucible.world.clock import parse_time

        t_noticed = parse_time(log[0]["at"].replace("+00:00", "Z"))  # the anchor's own firing instant
        for _ in range(12):  # 12 steps of 60 s: the anchored follow-up is due 10 min after the anchor fired
            await send(w, ["ops@world.local"], "tick")
        log = w.firing_log()
        assert [f["event"] for f in log] == ["noticed", "follow_up"]
        assert parse_time(log[1]["at"].replace("+00:00", "Z")) == t_noticed + timedelta(minutes=10)
        await send(w, ["boss@world.local"], "budget v2")  # a state trigger fires once
        assert [f["event"] for f in w.firing_log()].count("noticed") == 1
    finally:
        await w.close()


ARRIVALS = {"id": "arrival", "actor": "ops", "when": {"every": {"dist": "exp", "mean": "5m"}, "seed": 7, "until": "PT2H"},
            "action": {"type": "tool", "tool": "mail.send_email", "args": {"to": ["agent@world.local"], "subject": "ticket", "body": "new"}}}


async def test_seeded_arrivals_reproduce_after_restore_and_stop_at_until(tmp_path):
    async def run(restore_at: int | None):
        w = _world([ARRIVALS], tmp_path / f"r{restore_at}")
        await w.start()
        try:
            await w.reset()
            ref = None
            for i in range(150):  # 150 min > until (2 h)
                if restore_at is not None and i == restore_at:
                    ref = await w.snapshot()
                await send(w, ["ops@world.local"], "tick")
            if ref is not None:  # rewind to the snapshot and replay: the same arrivals come again
                await w.restore(ref)
                before = len(w.firing_log())
                for _ in range(150 - restore_at):
                    await send(w, ["ops@world.local"], "tick")
                return [f["at"] for f in w.firing_log()], before
            return [f["at"] for f in w.firing_log()], None
        finally:
            await w.close()

    base, _ = await run(None)
    assert 10 < len(base) < 50  # ~24 arrivals at a 5-minute mean over 2 h
    from crucible.world.clock import parse_time

    assert all(parse_time(a.replace("+00:00", "Z")) <= parse_time("2026-03-02T11:00:00Z") for a in base)  # until
    gaps = sorted({(parse_time(b.replace("+00:00", "Z")) - parse_time(a.replace("+00:00", "Z"))).total_seconds() for a, b in __import__('itertools').pairwise(base)})
    assert len(gaps) > 3  # exponential, not a fixed interval
    replayed, _ = await run(40)
    assert replayed[-len(base) + sum(1 for a in base if parse_time(a.replace("+00:00", "Z")) <= parse_time("2026-03-02T09:40:00Z")):] == \
        [a for a in base if parse_time(a.replace("+00:00", "Z")) > parse_time("2026-03-02T09:40:00Z")]


def test_every_spec_translates_to_a_trigger_agentenv_accepts(tmp_path):
    pytest.importorskip("agent_env")
    from agent_env.env.gateway.triggers import TriggerEngine

    from crucible.interop.agentenv.world_backend import AgentEnvBackend

    events = [ACK, STATE, FOLLOW, ARRIVALS, {"id": "nudge", "actor": "ops", "when": {"after": "30m"}, "action": {"type": "nl", "to": "agent", "text": "status?"}}]
    spec = WorldSpec.model_validate({**_world(events, tmp_path, clock_mode="wall").spec.model_dump(), "backend": "agentenv"})
    b = AgentEnvBackend(spec, {}, workdir=tmp_path / "b")

    class FakeWorld:
        pass

    fw = FakeWorld()
    fw.spec = spec
    eng = TriggerEngine.__new__(TriggerEngine)  # validation only: agentenv's own normalizer is the judge
    executor = {"a2a_url": "http://relay", "timeout_seconds": 60.0, "role": "executor"}
    for ev in spec.events:
        t = b._trigger(fw, ev)
        assert t is not None, ev.id
        out = TriggerEngine._normalize_trigger(eng, __import__("copy").deepcopy(t), executor)
        assert out["id"] == ev.id
    ack = b._trigger(fw, spec.events[0])
    assert ack["barrier"] == {"at": "provoking_call"} and ack["when"]["where"] and "verify" in ack["actions"][0]
    arr = b._trigger(fw, spec.events[3])
    assert arr["when"]["every"] == {"dist": "exp", "mean": "PT300S"} and arr["when"]["seed"] == 7 and arr["when"]["until"] == "2026-03-02T11:00:00Z"
    assert b._trigger(fw, spec.events[2])["when"] == {"type": "time", "after": "noticed", "offset": "PT600S"}


def test_schema_refusals():
    from crucible.core.schema.world import EventWhen

    with pytest.raises(ValueError, match="requires a `seed`"):
        EventWhen(every={"dist": "exp", "mean": "5m"})
    with pytest.raises(ValueError, match="go together"):
        EventWhen(after_event="x")
    with pytest.raises(ValueError, match="only valid on on_tool"):
        EventWhen(after="5m", where={"args.x": {"exists": True}})
    with pytest.raises(ValueError, match="carries a predicate"):
        EventWhen(state={"steps": [{"tool": "mail.list_inbox"}]})


async def test_live_agentenv_gateway_matches_step_mode(tmp_path):
    """The same where + templated + verified + barrier event, delegated to a real agentenv gateway (wall clock), has the
    same effect the step-mode scheduler has: the agent's next call sees the templated acknowledgement."""
    pytest.importorskip("agent_env")
    import json

    results = {}
    for mode, backend in (("local", "local"), ("gateway", "agentenv")):
        spec = WorldSpec.model_validate({**_world([ACK], tmp_path / mode, clock_mode="wall").spec.model_dump(), "backend": backend})
        w = World(spec, store=Store(tmp_path / mode / "s2"), workdir=tmp_path / mode / "w2")
        await w.start()
        try:
            await w.reset()
            await send(w, ["boss@world.local"], "lunch?")
            r = await send(w, ["boss@world.local"], "Q3 report")
            rid = json.loads(r.content)["id"]
            got = await w.step("agent", ToolCall(tool_call_id="i", function_name="mail.list_inbox", arguments={}))
            results[mode] = [m["subject"].replace(f"id {rid}", "id N") for m in json.loads(got.observation.tool_results[0].content)["messages"]]
            if backend == "agentenv":
                st = (await w.backend.http.get("/triggers/state")).json()
                kinds = [e["kind"] for e in st["events"] if e.get("trigger_id") == "ack"]
                assert "fired" in kinds and "verify_ok" in kinds, kinds
        finally:
            await w.close()
    assert results["local"] == results["gateway"] == ["Re: Q3 report (id N)"], results
