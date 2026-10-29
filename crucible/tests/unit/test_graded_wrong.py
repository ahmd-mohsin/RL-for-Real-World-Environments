"""M4 audit (2026-10-27) R-1: a policy turn graded wrong by a negative step signal that the trace does not contradict is never
a positive target. ``crucible label`` marked only turns a corrective signal graded (``pivots``: correction,
suggestion_accepted, revert), so on the toy pack a reply the customer rejected and one rejected then escalated were SFT rows
and KTO-desirable completions (2 of 10 each) (``docs/research/improvements/2026-10-27/M04-signals.md`` R-1)."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource
from crucible.signals.pivots import graded_wrong, pivots
from crucible.studio import datasets as ds
from crucible.studio.label import label

REJECTED = ("The carrier shows it at your building's parcel room.", "Please regenerate your API key and retry.")


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    out = tmp_path_factory.mktemp("graded_wrong") / "lab"
    s = asyncio.run(label(out=out, pack="toy"))
    return out, s


def _rows(p) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _says(row: dict, text: str) -> bool:
    return text in json.dumps(row.get("completion"))


def test_rejected_and_escalated_replies_are_never_positive_targets(toy):
    out, s = toy
    sft, kto = _rows(out / "sft.jsonl"), _rows(out / "kto.jsonl")
    for text in REJECTED:
        assert not [r for r in sft if _says(r, text)]  # was an SFT row
        assert [r["label"] for r in kto if _says(r, text)] == [False]  # was KTO-desirable
    assert s["datasets"]["sft"] == 8 and s["datasets"]["kto_balance"] == {"desirable": 8, "undesirable": 4}  # was 10 and 10/2
    assert s["skipped"]["sft_graded_wrong"] == 2
    assert s["pivots"] == {"pairs": 0, "without_policy_fix": 2}  # pairing is still pivots' own


def _trace(*signals: Signal) -> Trace:
    ev = [
        TraceEvent(seq=1, kind="message", actor="c", type="ticket_opened", text="The build is red."),
        TraceEvent(seq=2, kind="action", actor="a", type="commit", text="fix the flaky test"),
        TraceEvent(seq=3, kind="check", actor="ci", type="ci_run", text="tests: fail"),
        TraceEvent(seq=4, kind="check", actor="ci", type="ci_run", text="tests: pass"),
        TraceEvent(seq=5, kind="message", actor="a", type="agent_reply", text="Fixed."),
    ]
    return Trace(
        trace_id="t:1",
        family="toy/x",
        source=TraceSource(connector="file_drop", ref="T1"),
        actors=[TraceActor(id="c", role="customer", kind="human"), TraceActor(id="a", role="agent", kind="human"), TraceActor(id="ci", role="system", kind="system")],
        events=ev,
        outcome=TraceOutcome(label="good"),
        signals=list(signals),
    )


def test_only_uncontradicted_negative_step_signals_mark_a_turn():
    fail = Signal(id="f", type="check_result", polarity=-1, grades=[2], source_event=3, actor="ci", meta={"check": "tests"})
    passed = Signal(id="p", type="check_result", polarity=1, grades=[2], source_event=4, actor="ci", meta={"check": "tests"})
    assert graded_wrong(_trace(fail)) == {2: ["f"]}
    assert graded_wrong(_trace(fail, passed)) == {}  # the same check passed on that step later: noise
    every = Signal(id="r", type="revert", polarity=-1, grades=[2], source_event=5, actor="c", meta={"attribution": "all_commits"})
    trace_level = Signal(id="o", type="outcome", polarity=-1, grades=[2], source_event=5, actor="c")
    assert graded_wrong(_trace(every, trace_level)) == {}  # no decision point known; a trace-level verdict
    fix = Signal(id="c", type="correction", polarity=-1, grades=[2], source_event=5, actor="a")
    assert graded_wrong(_trace(fix)) == pivots(_trace(fix)) == {2: ["c"]}  # a corrective signal still marks its turn (V8)
    x = ds.from_trace(_trace(fail), reward=1.0)
    assert [m["content"] for i, m in enumerate(x.messages) if i in x.wrong] == ["fix the flaky test\nFixed."]  # one turn (checks are no messages)
    assert not ds.from_trace(_trace(fail, passed), reward=1.0).wrong
