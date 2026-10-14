"""M16 owner B: ``actors/rlm/stall.py`` (design §3.2; R11, H9; stress B6)."""

from __future__ import annotations

import json

import pytest

from crucible.actors.rlm.stall import StallDetector, StallVerdict

POLL = 'r = await world.call("tickets.get", id=5)\nprint(r)'


def cell(d: StallDetector, code: str, calls: list[tuple] = (), ok: bool = True, out: str | None = None) -> StallVerdict | None:
    """One cell: its World calls ``(tool, args, digest[, kind])`` in order, then the cell end and the check."""
    for c in calls:
        d.on_world(c[0], c[1], c[2], **({"kind": c[3]} if len(c) > 3 else {}))
    d.on_cell(code, ok, output_digest=out)
    return d.check()


def honest_work(d: StallDetector, i: int) -> StallVerdict | None:
    return cell(d, f"r = await world.call('orders.get', id={i})", [("orders.get", {"id": i}, f"order-{i}")], out=f"o{i}")


# ---- waiting is not looping -------------------------------------------------------------------------------------------------
def test_honest_poll_after_a_send_does_not_stop():
    d = StallDetector()
    assert cell(d, "await world.send('alice', 'Can you confirm the address?')", [("send", {"to": "alice", "text": "confirm?"}, "sent", "send")]) is None
    for _ in range(40):  # the customer has not replied: identical polls, identical results
        assert cell(d, POLL, [("tickets.get", {"id": 5}, "status=pending")], out="pending") is None
    assert d.waiting
    assert cell(d, POLL, [("tickets.get", {"id": 5}, "status=replied")], out="replied") is None
    assert not d.waiting
    for i in range(30):
        assert honest_work(d, i) is None


def test_poll_with_a_changing_result_is_waiting_not_looping():
    d = StallDetector()
    for t in range(300):  # no send; the result carries a rising clock
        assert cell(d, POLL, [("tickets.get", {"id": 5}, f"status=pending at={t}")], out=f"pending {t}") is None
    assert d.verdicts == 0


def test_b6_three_polls_150_cells_apart_never_end_an_honest_episode():
    """Stress B6: three identical get_ticket results in one cell, every 150 cells, ended a 1,834-step episode."""
    d = StallDetector()
    verdicts = []
    for i in range(2_000):
        if i % 150 == 0:
            v = cell(d, "for _ in range(3): print(await world.call('tickets.get', id=5))", [("tickets.get", {"id": 5}, "pending")] * 3)
        else:
            v = honest_work(d, i)
        if v is not None:
            verdicts.append(v)
    assert all(v.action != "stop" for v in verdicts)
    assert verdicts == []


def test_a_long_honest_wait_gets_notices_never_a_stop():
    d = StallDetector()
    cell(d, "await world.send('bob', 'ping')", [("send", {"to": "bob"}, "sent", "send")])
    verdicts = [v for v in (cell(d, POLL, [("tickets.get", {"id": 5}, "pending")], out="pending") for _ in range(600)) if v is not None]
    assert verdicts and all(v.signature == "static_results" for v in verdicts)
    assert all(v.action == "notice" for v in verdicts)  # 100 cells apart: the streak decays between them


def test_note_progress_ends_a_wait():
    d = StallDetector()
    cell(d, "await world.send('a', 'x')", [("send", {"to": "a"}, "sent", "send")])
    assert d.waiting
    d.note_progress("inbox")
    assert not d.waiting
    verdicts = [cell(d, POLL, [("tickets.get", {"id": 5}, "pending")], out="pending") for _ in range(3)]
    assert verdicts[-1] is not None and verdicts[-1].signature == "repeat_cells"


# ---- looping is caught ------------------------------------------------------------------------------------------------------
def test_alternating_ab_loop_is_caught_and_escalates_to_stop():
    d = StallDetector()
    a = ("A = await world.call('kb.search', q='refund')", [("kb.search", {"q": "refund"}, "no results")])
    b = ("B = await world.call('kb.search', q='return')", [("kb.search", {"q": "return"}, "no results")])
    actions = []
    for i in range(60):
        code, calls = a if i % 2 == 0 else b
        v = cell(d, code, calls, out="no results")
        if v is not None:
            actions.append((i + 1, v.signature, v.action))
            if v.action == "stop":
                break
    assert actions[0] == (6, "cell_cycle", "notice")  # A/B repeated 3 times
    assert [x[2] for x in actions] == ["notice", "notice", "reflect", "reflect", "reflect", "stop"]
    assert actions[-1][0] == 36
    assert d.by_signature["cell_cycle"] == 6


def test_alternating_world_calls_with_varying_code_are_caught_as_a_call_cycle():
    d = StallDetector()
    v = None
    for i in range(10):
        key = "a" if i % 2 == 0 else "b"
        v = cell(d, f"# attempt {i}\nx = await world.call('search', q='{key}')", [("search", {"q": key}, "[]")], out=f"attempt {i}: []")
    assert v is not None and v.signature == "call_cycle" and "cycle of 2" in v.evidence


def test_same_cell_three_times_is_caught():
    d = StallDetector()
    vs = [cell(d, POLL, [("tickets.get", {"id": 5}, "pending")], out="pending") for _ in range(3)]
    assert vs[:2] == [None, None] and vs[2].signature == "repeat_cells" and vs[2].action == "notice"
    assert "[stall-detected:" in vs[2].message()


def test_same_call_six_of_ten_is_caught():
    d = StallDetector()
    seq = ["A", "A", "x1", "A", "A", "x2", "A", "A", "x3", "A"]
    got = []
    for i, k in enumerate(seq):
        calls = [("orders.get", {"id": 1}, "same")] if k == "A" else [("kb.search", {"q": k}, f"r-{k}")]
        v = cell(d, f"step {i}", calls, out=f"s{i}")
        if v is not None:
            got.append((i, v))
    assert len(got) == 1
    i, v = got[0]
    assert i == 7 and v.signature == "same_call" and "orders.get" in v.evidence and "6 times" in v.evidence  # the 6th hit


def test_consecutive_cell_errors_are_caught_and_a_success_resets_them():
    d = StallDetector()
    assert [cell(d, f"bad{i}", ok=False, out=f"e{i}") for i in range(3)] == [None] * 3
    assert cell(d, "good", out="ok") is None
    vs = [cell(d, f"bad again {i}", ok=False, out=f"f{i}") for i in range(4)]
    assert vs[-1] is not None and vs[-1].signature == "cell_errors" and "4 cells in a row failed" in vs[-1].evidence


def test_repeated_identical_sends_are_a_loop_not_a_wait():
    """Continual Harness: the same payload sent 842 times. A send starts a wait, but resending it is looping."""
    d = StallDetector()
    v = None
    for _ in range(3):
        v = cell(d, "await world.send('alice', 'please confirm')", [("send", {"to": "alice", "text": "please confirm"}, "sent", "send")], out="sent")
    assert v is not None and v.signature == "repeat_cells"


# ---- the window -----------------------------------------------------------------------------------------------------------
def test_streak_decays_between_distant_stalls():
    d = StallDetector(decay_cells=50)
    streaks = []
    for _ in range(4):
        vs = [cell(d, POLL, [("tickets.get", {"id": 5}, "pending")], out="pending") for _ in range(3)]
        streaks.append(vs[-1].streak)
        for j in range(60):
            d.on_cell(f"x{j} = f({j})", True, output_digest=f"o{j}")  # kernel-only work, no World calls, no verdicts
            assert d.check() is None
    assert streaks == [1, 1, 1, 1]


def test_new_world_information_resets_the_streak():
    d = StallDetector()
    for _ in range(6):
        cell(d, POLL, [("tickets.get", {"id": 5}, "pending")], out="pending")
    assert d.streak == 2
    honest_work(d, 999)  # a result never seen before
    assert d.streak == 0
    vs = [cell(d, POLL, [("tickets.get", {"id": 5}, "pending")], out="pending") for _ in range(3)]
    assert vs[-1].streak == 1


def test_state_dict_round_trip_continues_identically():
    d = StallDetector()
    cell(d, "await world.send('a', 'x')", [("send", {"to": "a"}, "sent", "send")])
    for i in range(7):
        cell(d, "A" if i % 2 else "B", [("t", {"k": i % 2}, "r")], out="r")
    d.on_world("t", {"k": 0}, "r")  # a call inside a running cell
    d2 = StallDetector()
    d2.load_state_dict(json.loads(json.dumps(d.state_dict())))
    assert d2.state_dict() == d.state_dict()
    for i in range(40):
        code, calls = ("A" if i % 2 else "B"), [("t", {"k": i % 2}, "r")]
        assert cell(d, code, calls, out="r") == cell(d2, code, calls, out="r")


@pytest.mark.parametrize("kw", [{"escalate_after": 0}, {"escalate_after": 7, "stop_after": 6}, {"repeat_cells": 1}, {"same_call_window": 3, "same_call_hits": 6}])
def test_bad_thresholds_rejected(kw):
    with pytest.raises(ValueError):
        StallDetector(**kw)


def test_verdict_messages():
    assert StallVerdict("same_call", "ev", "stop", 6).message() == "stalled: ev"
    r = StallVerdict("same_call", "ev", "reflect", 3).message()
    assert "reflect in plain text" in r and "3 stalls in a row" in r
