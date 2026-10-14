"""M16 (C): ``actors/rlm/tree.py`` — tree-wide budgets, per-depth reserved concurrency, persistent children.

Every child here is a fake ``ChildRunner`` (sleeps, loops, fails, ignores cancellation, spawns grandchildren); the
real ``AgentLoop`` is F's. Each test ends with no pending asyncio task (``_no_leaks``)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from types import SimpleNamespace
from typing import Any

import pytest

from crucible.actors.rlm.tree import (
    AGENT_API,
    COMPLETED_TAIL_BYTES,
    LOST_AT_RESUME,
    RESULT_ANSWER_CHARS,
    BudgetExhausted,
    ChildRegistry,
    ChildRunner,
    TreeLimits,
    TreeSupervisor,
)

POLL = 0.01
GRACE = 0.3


# ---- fakes ---------------------------------------------------------------------------------------------------------------
class Fake:
    """A child whose behaviour is the first word of its brief (see ``SCRIPTS``)."""

    def __init__(self, *, name: str, brief: str, depth: int, world: str, persistent: bool, max_turns: int, path: str,
                 registry: ChildRegistry):
        self.name, self.brief, self.depth, self.world, self.persistent = name, brief, depth, world, persistent
        self.max_turns, self.path, self.registry = max_turns, path, registry
        self.turns = self.tokens = self.runs = self.swallowed = 0
        self.last = ""
        self.inbox: list[tuple[str, str]] = []
        self.cancel_requested = False
        self.closed = False
        self.finished = False
        self.sup: TreeSupervisor = registry.supervisor

    async def run(self) -> dict:
        self.runs += 1
        try:
            return await SCRIPTS[self.brief.split()[0]](self)
        finally:
            self.finished = True

    def deliver(self, text: str, kind: str) -> None:
        self.inbox.append((kind, text))

    def cancel(self) -> None:
        self.cancel_requested = True

    def progress(self) -> dict:
        return {"turns": self.turns, "tokens": self.tokens, "last": self.last}

    async def aclose(self) -> None:
        self.closed = True

    async def turn(self, tokens: int = 100, dt: float = 0.005) -> None:
        self.sup.charge_turn(self.depth)
        self.turns += 1
        await asyncio.sleep(dt)
        self.sup.charge_tokens(self.depth, tokens, 1000)
        self.tokens += tokens
        self.last = f"turn {self.turns} of {self.name}"


async def _sleep(f: Fake) -> dict:
    await f.turn()
    return {"status": "done", "answer": f"ok:{f.name}", "error": None}


async def _wait_child(reg: ChildRegistry, name: str) -> dict:
    while True:
        r = await reg.result(name, wait_s=0.2)
        if r["status"] != "running":
            return r


async def _nest(f: Fake) -> dict:
    """Spawn one grandchild and wait for it while holding this child's slot (the deadlock shape)."""
    await f.turn()
    h = await f.registry.spawn("sleep grandchild")
    r = await _wait_child(f.registry, h["name"])
    return {"status": "done", "answer": f"{f.name}<-{r['answer']}", "error": None}


async def _fanout(f: Fake) -> dict:
    """Spawn two children (nesting further while depth allows) and wait for both."""
    await f.turn()
    kind = "fanout" if f.depth + 1 < f.sup.limits.max_depth else "sleep"
    names = [(await f.registry.spawn(f"{kind} x"))["name"] for _ in range(2)]
    rs = [await _wait_child(f.registry, n) for n in names]
    assert all(r["status"] == "done" for r in rs), rs
    return {"status": "done", "answer": "+".join(r["answer"] for r in rs), "error": None}


async def _forever(f: Fake) -> dict:  # ignores the cooperative cancel(); only a task cancel stops it
    while True:
        await f.turn(dt=0.005)


async def _stubborn(f: Fake) -> dict:  # swallows the first task cancellation too
    while True:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            f.swallowed += 1
            if f.swallowed >= 2:
                raise


async def _fail(f: Fake) -> dict:
    await f.turn()
    raise RuntimeError("boom")


async def _spend(f: Fake) -> dict:  # spends the tree's turn budget from inside the child
    while True:
        await f.turn(dt=0.001)


async def _big(f: Fake) -> dict:
    return {"status": "done", "answer": "HEAD" + "x" * 30_000 + "é" * 10 + "END", "error": None}


async def _turns(f: Fake) -> dict:
    for _ in range(12):
        await f.turn(tokens=100, dt=0.02)
    return {"status": "done", "answer": "counted", "error": None}


async def _followup(f: Fake) -> dict:
    if f.runs == 1:
        return {"status": "done", "answer": "first", "error": None}
    return {"status": "done", "answer": f"follow-up: {f.inbox[-1][1]}", "error": None}


async def _parent_of_forever(f: Fake) -> dict:
    await f.registry.spawn("forever grandchild", name="gc")
    while True:
        await asyncio.sleep(0.01)


async def _bad_result(f: Fake) -> dict:
    return {"status": "great", "answer": 1}  # type: ignore[return-value]


SCRIPTS = {"sleep": _sleep, "nest": _nest, "fanout": _fanout, "forever": _forever, "stubborn": _stubborn, "fail": _fail,
           "spend": _spend, "big": _big, "turns": _turns, "followup": _followup, "parent_of_forever": _parent_of_forever,
           "bad_result": _bad_result}


class Factory:
    def __init__(self) -> None:
        self.made: list[Fake] = []
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kw: Any) -> Fake:
        self.calls.append({k: v for k, v in kw.items() if k != "registry"})
        f = Fake(**kw)
        self.made.append(f)
        return f

    def get(self, name: str) -> Fake:
        return next(f for f in self.made if f.name == name)


def _tree(**limits: Any) -> tuple[TreeSupervisor, Factory, ChildRegistry]:
    sup = TreeSupervisor(TreeLimits(**limits))
    fac = Factory()
    return sup, fac, ChildRegistry(sup, fac, depth=0, poll_s=POLL, cancel_grace_s=GRACE)


async def _no_leaks() -> None:
    for _ in range(3):
        await asyncio.sleep(0)
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert not pending, pending


# ---- limits and the supervisor -------------------------------------------------------------------------------------------
def test_limits_validate_and_reserve() -> None:
    lim = TreeLimits(max_depth=2, max_concurrent=2, reserve_per_depth=1)
    assert (lim.cap(1), lim.cap(2)) == (1, 2)
    with pytest.raises(ValueError, match="max_depth"):
        TreeLimits(max_depth=4)
    with pytest.raises(ValueError, match="deadlock"):
        TreeLimits(max_depth=2, reserve_per_depth=0)
    with pytest.raises(ValueError, match="depth-1 children get a slot"):
        TreeLimits(max_depth=3, max_concurrent=2, reserve_per_depth=1)
    with pytest.raises(TypeError):
        TreeLimits(max_spawns=True)
    with pytest.raises(ValueError):
        TreeLimits(child_wall_s=math.inf)
    assert TreeLimits.from_mapping({"max_depth": 2, "max_turns": 9, "seq_len": 1}).max_depth == 2
    d = TreeLimits()
    assert (d.max_total_new_tokens, d.max_total_turns, d.max_sub_calls, d.max_events, d.max_event_bytes) == (10_000_000, 20_000, 4_096, 10_000, 65_536)


def test_budgets_are_tree_wide_and_typed() -> None:
    sup = TreeSupervisor(TreeLimits(max_total_turns=3, max_total_new_tokens=500, max_sub_calls=4, max_sub_tokens=100, max_spawns=1, max_depth=2))
    sup.charge_turn(0)
    sup.charge_turn(1)
    sup.charge_turn(2)  # three agents at three depths share one turn budget
    with pytest.raises(BudgetExhausted) as e:
        sup.charge_turn(0)
    assert e.value.kind == "budget:turns" and str(e.value).startswith("budget:turns")

    sup = TreeSupervisor(TreeLimits(max_total_new_tokens=500, max_sub_calls=4, max_sub_tokens=100, max_spawns=1, max_depth=1))
    sup.charge_tokens(0, 300, 5000)
    sup.charge_tokens(1, 150, summary=True)  # a summary call: tokens, never a turn
    assert sup.usage()["turns"] == 0 and sup.usage()["summary_calls"] == 1
    with pytest.raises(BudgetExhausted) as e:
        sup.charge_tokens(1, 100)  # recorded first (honest totals), then raised
    assert e.value.kind == "budget:new_tokens" and sup.usage()["new_tokens"] == 550
    with pytest.raises(BudgetExhausted) as e:
        sup.charge_turn(0)
    assert e.value.kind == "budget:new_tokens"
    with pytest.raises(BudgetExhausted) as e:
        sup.check_spawn(0)
    assert e.value.kind == "budget:new_tokens"

    sup.charge_sub(3)
    with pytest.raises(BudgetExhausted) as e:
        sup.charge_sub(2)  # all or nothing
    assert e.value.kind == "budget:sub_calls" and sup.usage()["sub_calls"] == 3
    with pytest.raises(BudgetExhausted) as e:
        sup.charge_sub_tokens(150)
    assert e.value.kind == "budget:sub_tokens"
    with pytest.raises(BudgetExhausted) as e:
        sup.charge_sub(1)
    assert e.value.kind == "budget:sub_tokens"

    sup = TreeSupervisor(TreeLimits(max_spawns=1, max_depth=1))
    with pytest.raises(BudgetExhausted) as e:
        sup.check_spawn(1)
    assert e.value.kind == "budget:depth"
    sup.charge_spawn(0)
    with pytest.raises(BudgetExhausted) as e:
        sup.charge_spawn(0)
    assert e.value.kind == "budget:spawns"
    with pytest.raises(ValueError):
        sup.charge_tokens(0, -1)
    with pytest.raises(ValueError):
        sup.charge_turn(5)


def test_usage_sources_and_state_roundtrip() -> None:
    sup = TreeSupervisor(TreeLimits(max_depth=2))
    sup.charge_turn(0)
    sup.charge_tokens(0, 100, 900)
    assert sup.usage()["new_tokens_source"] == "usage"
    sup.charge_tokens(2, 50, estimate=True)
    u = sup.usage()
    assert u["new_tokens_source"] == "usage+estimate" and u["by_depth"][2]["new_tokens"] == 50 and u["left"]["turns"] == 19_999
    assert 0 < u["fraction"] < 1
    sup.charge_sub(2)
    sup.charge_sub_tokens(70)
    st = json.loads(json.dumps(sup.state_dict()))
    sup2 = TreeSupervisor(TreeLimits(max_depth=2))
    sup2.load_state_dict(st)
    assert {k: v for k, v in sup2.usage().items() if k != "fraction"} == {k: v for k, v in sup.usage().items() if k != "fraction"}
    with pytest.raises(ValueError):
        sup2.load_state_dict({"version": 1, "counters": {"turns": 1}})
    with pytest.raises(ValueError):
        sup2.load_state_dict({**st, "counters": {**st["counters"], "turns": -1}})


# ---- per-depth reserved concurrency --------------------------------------------------------------------------------------
class NaiveSupervisor(TreeSupervisor):
    """One shared pool of ``max_concurrent`` slots for every depth: what nano-rlm's reservation exists to avoid."""

    def __init__(self, limits: TreeLimits):
        super().__init__(limits)
        self._sem: asyncio.Semaphore | None = None

    @contextlib.asynccontextmanager
    async def slot(self, depth: int):  # type: ignore[override]
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.limits.max_concurrent)
        async with self._sem:
            yield


async def _all_settle(reg: ChildRegistry, names: list[str]) -> list[dict]:
    return [await _wait_child(reg, n) for n in names]


async def test_nested_children_do_not_deadlock_with_reserved_capacity() -> None:
    sup, _fac, reg = _tree(max_concurrent=2, reserve_per_depth=1, max_depth=2)
    names = [(await reg.spawn("nest a"))["name"] for _ in range(3)]
    rs = await asyncio.wait_for(_all_settle(reg, names), timeout=10)
    assert [r["status"] for r in rs] == ["done"] * 3
    assert rs[0]["answer"] == "child-1<-ok:child-1"
    assert sup.usage()["peak_running"] <= 2 and sup.usage()["running"] == 0
    await reg.aclose()
    await _no_leaks()


async def test_naive_shared_semaphore_deadlocks_on_the_same_tree() -> None:
    sup = NaiveSupervisor(TreeLimits(max_concurrent=2, reserve_per_depth=1, max_depth=2))
    reg = ChildRegistry(sup, Factory(), depth=0, poll_s=POLL, cancel_grace_s=GRACE)
    names = [(await reg.spawn("nest a"))["name"] for _ in range(3)]
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(_all_settle(reg, names), timeout=1.0)  # both slots held by depth-1 parents waiting on queued children
    await reg.aclose()  # and the registry still cleans the stuck tree up
    assert all(r["status"] == "cancelled" for r in reg.list())
    await _no_leaks()


@pytest.mark.parametrize(("max_depth", "max_concurrent", "reserve"), [(2, 2, 1), (2, 3, 1), (3, 3, 1), (3, 5, 2), (3, 4, 1)])
async def test_fanout_trees_complete_within_the_concurrency_cap(max_depth: int, max_concurrent: int, reserve: int) -> None:
    sup, _fac, reg = _tree(max_depth=max_depth, max_concurrent=max_concurrent, reserve_per_depth=reserve, max_spawns=64)
    names = [(await reg.spawn("fanout root"))["name"] for _ in range(3)]
    rs = await asyncio.wait_for(_all_settle(reg, names), timeout=20)
    assert all(r["status"] == "done" for r in rs)
    u = sup.usage()
    assert u["peak_running"] <= max_concurrent and u["spawns"] == 3 * (2 ** max_depth - 1)
    await reg.aclose()
    await _no_leaks()


# ---- children: statuses, hard limits, cancellation -----------------------------------------------------------------------
async def test_child_budget_stop_is_status_budget_not_a_crash() -> None:
    sup, _fac, reg = _tree(max_total_turns=5)
    await reg.spawn("spend all", name="spender")
    r = await _wait_child(reg, "spender")
    assert r["status"] == "budget" and r["error"].startswith("budget:turns")
    assert r["answer"].startswith("turn ")  # salvaged from the child's last output
    evs = reg.events()
    assert [(e["type"], e["status"]) for e in evs] == [("child.failed", "budget")]
    with pytest.raises(BudgetExhausted) as e:
        sup.charge_turn(0)  # the root sees the same tree budget
    assert e.value.kind == "budget:turns"
    with pytest.raises(BudgetExhausted) as e:
        await reg.spawn("sleep")
    assert e.value.kind == "budget:turns"
    await reg.aclose()
    await _no_leaks()


async def test_failed_and_invalid_children_are_reported_and_the_parent_goes_on() -> None:
    _sup, _fac, reg = _tree(max_depth=1)
    await reg.spawn("fail now", name="f")
    await reg.spawn("bad_result", name="b")
    r = await _wait_child(reg, "f")
    assert r["status"] == "failed" and r["error"] == "RuntimeError: boom"
    r = await _wait_child(reg, "b")
    assert r["status"] == "failed" and "invalid result" in r["error"]
    assert {e["type"] for e in reg.events()} == {"child.failed"}
    await reg.aclose()
    await _no_leaks()


async def test_wall_budget_is_enforced_hard() -> None:
    _sup, fac, reg = _tree(child_wall_s=0.2)
    await reg.spawn("forever", name="loop")  # ignores cancel()
    await reg.spawn("stubborn", name="mule")  # swallows a task cancellation too
    t0 = asyncio.get_running_loop().time()
    r1, r2 = await _wait_child(reg, "loop"), await _wait_child(reg, "mule")
    assert asyncio.get_running_loop().time() - t0 < 0.2 + 3 * GRACE + 1.0
    for r in (r1, r2):
        assert r["status"] == "budget" and r["error"].startswith("budget:child_wall_s")
    assert fac.get("loop").cancel_requested and fac.get("loop").finished and fac.get("mule").swallowed == 2
    assert fac.get("loop").closed  # its kernel was closed
    await reg.aclose()
    await _no_leaks()


async def test_turn_cap_is_enforced_hard_and_passed_to_the_factory() -> None:
    _sup, fac, reg = _tree(child_max_turns=50)
    h = await reg.spawn("forever", name="loop", max_turns=4)
    assert h["max_turns"] == 4 and fac.calls[0]["max_turns"] == 4
    r = await _wait_child(reg, "loop")
    assert r["status"] == "budget" and r["error"].startswith("budget:child_turns") and r["turns"] > 4
    with pytest.raises(ValueError, match="max_turns"):
        await reg.spawn("sleep", max_turns=51)  # may lower the cap, never raise it
    await reg.spawn("sleep", name="d")
    assert fac.calls[-1]["max_turns"] == 50
    await reg.aclose()
    await _no_leaks()


async def test_cancel_cascades_to_grandchildren() -> None:
    _sup, fac, reg = _tree(max_depth=2, max_concurrent=3)
    await reg.spawn("parent_of_forever", name="p")
    for _ in range(200):
        if any(f.name == "gc" and f.turns > 2 for f in fac.made):
            break
        await asyncio.sleep(0.01)
    gc = fac.get("gc")
    child_reg = fac.get("p").registry
    out = await reg.cancel("p")
    assert out["status"] == "cancelled"
    assert gc.finished and child_reg.list()[0]["status"] == "cancelled" and gc.closed and fac.get("p").closed
    turns = gc.turns
    await asyncio.sleep(0.05)
    assert gc.turns == turns  # really stopped
    await reg.aclose()
    await _no_leaks()


async def test_aclose_cancels_everything_queued_and_running() -> None:
    sup, fac, reg = _tree(max_depth=2, max_concurrent=3, reserve_per_depth=1)
    for _ in range(4):
        await reg.spawn("parent_of_forever")
    await reg.spawn("stubborn", name="mule")
    await asyncio.sleep(0.1)
    assert any(r["queued"] for r in reg.list())
    await reg.aclose()
    assert all(r["status"] == "cancelled" for r in reg.list())
    assert all(f.finished or f.runs == 0 for f in fac.made) and all(f.closed for f in fac.made)
    assert sup.usage()["running"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        await reg.spawn("sleep")
    await reg.aclose()  # idempotent
    await _no_leaks()


# ---- events, results, watch ----------------------------------------------------------------------------------------------
async def test_inbox_is_bounded_drop_oldest_with_a_counted_notice() -> None:
    _sup, _fac, reg = _tree(max_events=3, max_spawns=8, max_concurrent=8)
    names = [(await reg.spawn("sleep"))["name"] for _ in range(5)]
    await _all_settle(reg, names)
    evs = reg.events()
    assert evs[0]["type"] == "events.dropped" and evs[0]["count"] == 2
    assert [e["type"] for e in evs[1:]] == ["child.completed"] * 3
    assert [e["seq"] for e in evs[1:]] == sorted(e["seq"] for e in evs[1:]) and evs[1]["seq"] == 3  # the oldest two went
    assert reg.events() == []  # drained
    await reg.aclose()
    await _no_leaks()


async def test_completed_event_tail_and_result_cap_with_ledger_pointer() -> None:
    _sup, _fac, reg = _tree()
    await reg.spawn("big", name="big")
    r = await _wait_child(reg, "big")
    assert r["status"] == "done" and r["truncated"] and r["answer_chars"] == 30_017
    assert len(r["answer"]) <= RESULT_ANSWER_CHARS and r["answer"].startswith("HEAD") and r["answer"].endswith("END")
    assert "history.child('big')" in r["answer"] and "30,017" in r["answer"]
    (ev,) = reg.events()
    assert ev["type"] == "child.completed" and len(ev["answer_tail"].encode()) <= COMPLETED_TAIL_BYTES and ev["answer_tail"].endswith("éEND")
    assert ev["answer_chars"] == 30_017
    await reg.aclose()
    await _no_leaks()


async def test_event_payloads_fit_max_event_bytes() -> None:
    _sup, _fac, reg = _tree(max_event_bytes=1024)
    await reg.spawn("big", name="big")
    await _wait_child(reg, "big")
    (ev,) = reg.events()
    assert len(json.dumps(ev, ensure_ascii=False).encode()) <= 1024 and ev["truncated"] and ev["answer_tail"].endswith("END")
    with pytest.raises(ValueError, match="1024 bytes"):
        await reg.send("big", "y" * 2000)
    await reg.aclose()
    await _no_leaks()


async def test_watch_every_turns_and_every_tokens_yield_progress_events() -> None:
    _sup, _fac, reg = _tree()
    await reg.spawn("turns", name="t")
    reg.watch("t", every_turns=3)
    await _wait_child(reg, "t")
    prog = [e for e in reg.events() if e["type"] == "child.progress"]
    assert len(prog) >= 2 and all(e["name"] == "t" and e["turns"] >= 3 for e in prog)
    assert [e["turns"] // 3 for e in prog] == sorted({e["turns"] // 3 for e in prog})  # one per crossed multiple

    await reg.spawn("turns", name="k")
    reg.watch("k", every_tokens=500)
    await _wait_child(reg, "k")
    prog = [e for e in reg.events() if e["type"] == "child.progress"]
    assert len(prog) >= 1 and all(e["tokens"] >= 500 for e in prog) and prog[0]["last"].startswith("turn ")
    with pytest.raises(ValueError):
        reg.watch("k", every_turns=0)
    await reg.aclose()
    await _no_leaks()


async def test_persistent_follow_up_send_and_steer() -> None:
    _sup, fac, reg = _tree()
    await reg.spawn("followup", name="p")
    r = await _wait_child(reg, "p")
    assert (r["status"], r["answer"], r["idle"]) == ("done", "first", True)
    out = await reg.send("p", "what about B?")
    assert out["status"] == "running" and out["round"] == 2
    r = await _wait_child(reg, "p")
    assert r["answer"] == "follow-up: what about B?" and fac.get("p").runs == 2

    await reg.spawn("followup", name="once", persistent=False)
    await _wait_child(reg, "once")
    assert fac.get("once").closed  # a non-persistent child's kernel is closed when it settles
    with pytest.raises(RuntimeError, match="not persistent"):
        await reg.send("once", "again?")

    await reg.spawn("turns", name="busy")
    await reg.steer("busy", "focus on X")
    await reg.send("busy", "and note Y")
    assert fac.get("busy").inbox == [("steer", "focus on X"), ("message", "and note Y")]
    await _wait_child(reg, "busy")
    with pytest.raises(RuntimeError, match="not running"):
        await reg.steer("busy", "late")
    await reg.aclose()
    await _no_leaks()


# ---- requests from the kernel --------------------------------------------------------------------------------------------
async def test_handle_validates_requests_strictly() -> None:
    _sup, fac, reg = _tree(max_depth=1)
    h = reg.handle
    assert (await h({"type": "agent.spawn", "brief": "sleep", "name": None, "persistent": True, "world": "none", "max_turns": None}))["name"] == "child-1"
    assert (await h({"type": "agent.spawn", "brief": "sleep"}))["name"] == "child-2"
    assert (await h({"type": "agent.spawn", "brief": "sleep", "name": "worker_A-1", "world": "shared"}))["world"] == "shared"
    bad = [
        ({"type": "agent.run", "brief": "x"}, "unknown agent op"),
        ({"type": "world.call", "tool": "x"}, "not an agent request"),
        ({"type": "agent.spawn", "brief": "sleep", "role": "x"}, "unexpected arguments"),
        ({"type": "agent.spawn"}, "needs"),
        ({"type": "agent.spawn", "brief": "sleep", "name": "bad name!"}, "must match"),
        ({"type": "agent.spawn", "brief": "sleep", "name": "x" * 65}, "must match"),
        ({"type": "agent.spawn", "brief": "sleep", "name": "worker_A-1"}, "already exists"),
        ({"type": "agent.spawn", "brief": "sleep", "world": "all"}, "world must be"),
        ({"type": "agent.spawn", "brief": "sleep", "max_turns": True}, "max_turns"),
        ({"type": "agent.spawn", "brief": "   "}, "brief"),
        ({"type": "agent.spawn", "brief": 3}, "brief"),
        ({"type": "agent.result", "name": "child-1", "wait_s": float("nan")}, "wait_s"),
        ({"type": "agent.result", "name": "nobody"}, "no child named"),
        ({"type": "agent.send", "name": "child-1", "text": ""}, "text"),
        ({"type": "agent.watch", "name": "child-1", "every_turns": 2.5}, "every_turns"),
        ({"type": "agent.events", "x": 1}, "unexpected"),
    ]
    for req, msg in bad:
        with pytest.raises((ValueError, TypeError), match=msg):
            await h(req)
    with pytest.raises(TypeError):
        await h(["agent.list"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        await h({"type": "agent.spawn", "brief": "sleep", "persistent": "yes"})
    assert len(fac.made) == 3  # no refused request created a child
    rows = await h({"type": "agent.list"})
    assert [(r["name"], r["world"]) for r in rows] == [("child-1", "none"), ("child-2", "none"), ("worker_A-1", "shared")]
    assert [c["world"] for c in fac.calls] == ["none", "none", "shared"]  # the grant reaches the factory
    assert {c["path"] for c in fac.calls} == {"root/child-1", "root/child-2", "root/worker_A-1"} and {c["depth"] for c in fac.calls} == {1}
    await reg.aclose()
    await _no_leaks()


async def test_spawn_refusals_are_typed_and_uncharged_on_factory_failure() -> None:
    sup, fac, reg = _tree(max_spawns=2, max_depth=1)

    def broken(**kw: Any) -> Any:
        raise OSError("no kernel")

    reg2 = ChildRegistry(sup, broken, depth=0, poll_s=POLL, cancel_grace_s=GRACE)
    with pytest.raises(OSError):
        await reg2.spawn("sleep", name="x")
    assert sup.usage()["spawns"] == 0 and reg2.list() == []
    await reg.spawn("sleep")
    await reg.spawn("sleep")
    with pytest.raises(BudgetExhausted) as e:
        await reg.handle({"type": "agent.spawn", "brief": "sleep"})
    assert e.value.kind == "budget:spawns"
    child = ChildRegistry(sup, fac, depth=1, poll_s=POLL, cancel_grace_s=GRACE)  # a depth-1 agent at max_depth 1
    sup2 = TreeSupervisor(TreeLimits(max_depth=1))
    with pytest.raises(BudgetExhausted) as e:
        await ChildRegistry(sup2, fac, depth=1).spawn("sleep")
    assert e.value.kind == "budget:depth"
    await child.aclose()
    await reg.aclose()
    await reg2.aclose()
    await _no_leaks()


async def test_agent_api_kernel_side_round_trip() -> None:
    _sup, fac, reg = _tree()

    async def host(type: str, **payload: Any) -> Any:  # the kernel's host(): JSON both ways
        req = json.loads(json.dumps({"type": type, **payload}))
        return json.loads(json.dumps(await reg.handle(req)))

    ns: dict[str, Any] = {"host": host, "rlm": SimpleNamespace()}
    exec(AGENT_API, ns)  # noqa: S102 - the kernel-side source, run as the kernel would
    agent = ns["rlm"].agent
    h = await agent.spawn("followup", name="w")
    assert h["status"] == "running" and "spawn" in repr(agent)
    await agent.watch(h, every_turns=5)
    r = await agent.result(h, wait_s=5)
    assert (r["status"], r["answer"]) == ("done", "first")
    await agent.send("w", "more")
    r = await agent.result("w", wait_s=5)
    assert r["answer"] == "follow-up: more"
    assert [e["type"] for e in await agent.events()] == ["child.completed", "child.completed"]
    assert [x["name"] for x in await agent.list()] == ["w"]
    await agent.spawn("forever", name="z")
    await agent.steer("z", "stop soon")
    assert (await agent.cancel("z"))["status"] == "cancelled"
    assert fac.get("z").inbox == [("steer", "stop soon")]
    await reg.aclose()
    await _no_leaks()


# ---- resume --------------------------------------------------------------------------------------------------------------
async def test_resume_reports_running_children_as_lost() -> None:
    sup, _fac, reg = _tree()
    await reg.spawn("big", name="done1")
    await _wait_child(reg, "done1")
    await reg.spawn("forever", name="busy")
    await asyncio.sleep(0.05)
    sup_state = json.loads(json.dumps(sup.state_dict()))
    reg_state = json.loads(json.dumps(reg.state_dict()))
    await reg.aclose()

    sup2 = TreeSupervisor(TreeLimits())
    sup2.load_state_dict(sup_state)
    assert sup2.usage()["spawns"] == 2 and sup2.usage()["running"] == 0
    reg2 = ChildRegistry(sup2, Factory(), depth=0, poll_s=POLL, cancel_grace_s=GRACE)
    reg2.load_state_dict(reg_state)
    rows = {r["name"]: r for r in reg2.list()}
    assert rows["busy"]["status"] == "failed" and rows["busy"]["error"] == LOST_AT_RESUME and rows["busy"]["restored"]
    lost = [e for e in reg2.events() if e["type"] == "child.failed"]
    assert lost[-1]["name"] == "busy" and "not resumed mid-flight" in lost[-1]["note"]
    r = await reg2.result("done1")
    assert r["status"] == "done" and r["truncated"] and len(r["answer"]) <= RESULT_ANSWER_CHARS and r["answer_chars"] == 30_017
    with pytest.raises(RuntimeError, match=LOST_AT_RESUME):
        await reg2.send("done1", "more?")
    assert (await reg2.cancel("busy"))["status"] == "failed"
    with pytest.raises(ValueError, match="already exists"):
        await reg2.spawn("sleep", name="busy")
    assert (await reg2.spawn("sleep"))["name"] == "child-1"
    with pytest.raises(RuntimeError, match="fresh"):
        reg2.load_state_dict(reg_state)
    await reg2.aclose()
    await _no_leaks()


def test_runner_protocol_is_checkable() -> None:
    reg = ChildRegistry(TreeSupervisor(), Factory(), depth=0)
    f = Fake(name="a", brief="sleep", depth=1, world="none", persistent=True, max_turns=1, path="root/a", registry=reg)
    assert isinstance(f, ChildRunner) and not isinstance(object(), ChildRunner)
