"""M5 R1 (docs/research/improvements/2026-10-14/M05-workflow.md §7): a pack's Φ reaches a dataset only through a held-out
validity verdict, one per channel (``credit.channels``), applied by ``crucible label``.

Before the gate, ``label --pack toy`` wrote every turn weight with ``credit: true`` and paid failed tickets partial
credit, although Φ in the middle of a ticket predicts its outcome at chance (held-out AUC 0.58) and the end-state Φ's
95 % lower bound is 0.50. The wrong answer of failed ticket T1011 was the highest-weighted KTO turn.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from pathlib import Path

import pytest

from crucible.core.schema.reward import InfraStatus
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource
from crucible.credit import channels
from crucible.credit.compose import Coefficients, compose, without_progress
from crucible.studio.label import label

FAM = "x/fam"


def _trace(i: int, label_: str) -> Trace:
    ev = [TraceEvent(seq=s, kind="action", actor="a", type="act", text=f"step {s}") for s in range(4)]
    return Trace(trace_id=f"t{i}", family=FAM, source=TraceSource(connector="file_drop", ref=f"T{i}"),
                 actors=[TraceActor(id="a", role="agent", kind="human")], events=ev, outcome=TraceOutcome(label=label_))


def _row(phi_by_seq: dict[int, float], *, outcome: bool, split: str = "test", gates_ok: bool = True) -> dict:
    end = max(phi_by_seq.items())[1] if phi_by_seq else 0.0
    rj = compose(gates={"g": gates_ok}, outcome=outcome, phi=end)
    return {"reward": rj.reward, "reward_json": rj.model_dump(mode="json"), "phi_by_seq": phi_by_seq, "split": split,
            "outcome": "good" if outcome else "bad"}


def _family(goods: list[dict[int, float]], bads: list[dict[int, float]], split: str = "test"):
    traces, credit = [], {}
    for i, phi in enumerate(goods):
        t = _trace(i, "good")
        traces.append(t)
        credit[t.trace_id] = _row(phi, outcome=True, split=split)
    for j, phi in enumerate(bads, start=len(goods)):
        t = _trace(j, "bad")
        traces.append(t)
        credit[t.trace_id] = _row(phi, outcome=False, split=split)
    return traces, credit


# Φ climbs early in good traces and never moves in bad ones (the midpoint event of 4 is seq 2)
GOOD = {1: 0.5, 2: 1.0, 3: 1.0}
BAD = {1: 0.0, 2: 0.0, 3: 0.0}


def test_U1_a_separating_family_releases_both_channels():
    traces, credit = _family([GOOD] * 20, [BAD] * 20)
    v = channels.decide(FAM, traces, credit)
    assert v.step.released and v.terminal.released, v.to_dict()
    assert v.step.validity.auc == 1.0 and v.holdout == {"rule": "credit hook split != train", "traces": 40, "good": 20, "bad": 20, "unknown": 0}


def test_U2_a_holdout_without_failures_releases_nothing():
    traces, credit = _family([GOOD] * 10, [])
    v = channels.decide(FAM, traces, credit)
    assert not v.step.released and not v.terminal.released
    assert v.step.reason == v.terminal.reason == "holdout lacks one class"


def test_U3_the_point_rule_alone_is_not_enough():
    """AUC 0.667 on 3 good / 3 bad passes ``min_auc`` but its 95 % lower bound does not exclude chance (the toy's
    end-state Φ)."""
    traces, credit = _family([{3: 1.0}] * 3, [{3: 1.0}, {3: 1.0}, {3: 0.0}])
    v = channels.decide(FAM, traces, credit)
    assert v.terminal.validity.auc == pytest.approx(2 / 3) and v.terminal.validity.kept
    assert not v.terminal.released and v.terminal.reason.startswith("95% lower bound") and "is not above 0.5" in v.terminal.reason


def test_U4_phi_only_at_the_last_event_is_no_step_credit():
    """The coding hook records Φ of the final patch at the last event: the midpoint statistic is 0 for every trace, so
    the step channel is unreleased while the terminal one is judged on the end value."""
    traces, credit = _family([{3: 1.0}] * 20, [{3: 0.0}] * 20)
    v = channels.decide(FAM, traces, credit)
    assert channels.step_statistic([0, 1, 2, 3], {3: 1.0}) == 0.0
    assert not v.step.released and v.step.reason == "holdout AUC 0.500 < 0.6"
    assert v.terminal.released


def test_U5_no_held_out_traces_releases_nothing():
    traces, credit = _family([GOOD] * 20, [BAD] * 20, split="train")
    v = channels.decide(FAM, traces, credit)
    assert not v.step.released and not v.terminal.released and v.step.reason == channels.NO_HOLDOUT


def test_U5b_a_hook_without_phi_is_left_as_it_is():
    traces, credit = _family([GOOD] * 4, [BAD] * 4)
    for r in credit.values():
        r.pop("phi_by_seq")
        r["reward_json"]["progress"] = 0.0
    v = channels.decide(FAM, traces, credit)
    assert v.step.reason == channels.NO_PHI
    for r in credit.values():
        row, phi = channels.apply(r, v, hook="h")
        assert phi is None and row["reward"] == r["reward"] and "reward_with_progress" not in row


def test_U6_apply_withholds_exactly_the_unreleased_channels():
    traces, credit = _family([GOOD] * 20, [BAD] * 20)
    released = channels.decide(FAM, traces, credit)
    failed = _row({1: 0.5, 2: 0.6667, 3: 0.6667}, outcome=False)  # partial credit κ·Φ = 0.133
    ok, masked = _row(GOOD, outcome=True), dict(_row(GOOD, outcome=False), reward=None)
    masked["reward_json"] = compose(gates={}, outcome=False, phi=0.5, masked=InfraStatus(status="masked", failure_kind="infra")).model_dump(mode="json")
    row, phi = channels.apply(failed, released, hook="h")  # both released: nothing changes
    assert row["reward"] == failed["reward"] and phi == failed["phi_by_seq"] and row["channels"] == {"step": "released", "terminal": "released"}
    held = channels.FamilyVerdict(FAM, channels.ChannelVerdict("step", "s", reason="r1"), channels.ChannelVerdict("terminal", "t", reason="r2"))
    row, phi = channels.apply(failed, held, hook="h")
    assert phi is None and row["reward"] == 0.0 and row["reward_with_progress"] == pytest.approx(0.13334)
    assert row["phi_by_seq"] == failed["phi_by_seq"] and row["channels"] == {"step": "unreleased: r1", "terminal": "unreleased: r2"}
    row, _ = channels.apply(ok, held, hook="h")
    assert row["reward"] == 1.0 and "reward_with_progress" not in row  # a success keeps its reward
    row, _ = channels.apply(masked, held, hook="h")
    assert row["reward"] is None  # masked stays masked
    gated = _row({3: 0.9}, outcome=False, gates_ok=False)
    assert channels.apply(gated, held, hook="h")[0]["reward"] == 0.0


def test_U7_dense_credit_without_reward_json_is_refused():
    held = channels.FamilyVerdict(FAM, channels.ChannelVerdict("step", "s", reason="r"), channels.ChannelVerdict("terminal", "t", reason="r"))
    row = {"reward": 0.13, "phi_by_seq": {3: 0.66}, "split": "train"}
    with pytest.raises(ValueError, match="credit hook pkg:credit returned dense credit without reward_json"):
        channels.apply(row, held, hook="pkg:credit")


def test_U8_without_progress_is_compose_with_phi_zero():
    for gates, outcome, phi, rubric, eff, kappa in itertools.product(({}, {"a": True}, {"a": True, "b": False}), (True, False), (0.0, 0.4, 1.0),
                                                                    (0.0, 0.7), (0.0, 1.5), (0.0, 0.2, 0.3)):
        coef = Coefficients(kappa=kappa)
        rj = compose(gates=gates, outcome=outcome, phi=phi, rubric=rubric, efficiency=eff, coef=coef)
        want = compose(gates=gates, outcome=outcome, phi=0.0, rubric=rubric, efficiency=eff, coef=coef).reward
        assert without_progress(rj) == want == without_progress(rj.model_dump(mode="json"))


def test_U10_verdicts_are_deterministic():
    traces, credit = _family([GOOD] * 6 + [BAD] * 2, [BAD] * 5 + [GOOD] * 2)
    assert channels.decide(FAM, traces, credit).to_dict() == channels.decide(FAM, traces, credit).to_dict()


# ---- the toy pack end to end -----------------------------------------------------------------------------------------------
def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


@pytest.fixture(scope="module")
def toy_labelled(tmp_path_factory):
    out = tmp_path_factory.mktemp("m5r1") / "lab"
    return out, asyncio.run(label(out=out, pack="toy"))


def test_U9_toy_label_withholds_both_channels_and_keeps_every_row(toy_labelled, tmp_path, monkeypatch):
    out, s = toy_labelled
    (fam, v), = s["credit_channels"].items()
    assert fam == "toy/support_ticket"
    assert v["holdout"]["traces"] == 12 and v["holdout"]["bad"] >= 3
    assert not v["step"]["released"] and v["step"]["reason"].startswith("holdout AUC") and v["step"]["auc"] < 0.6
    assert not v["terminal"]["released"] and "lower bound" in v["terminal"]["reason"]
    assert set(v["milestones"]) == {"log/assignee", "log/resolution", "log/status=resolved"}
    rows = {f: _rows(out / f"{f}.jsonl") for f in ("sft", "kto", "dpo")}
    assert rows["sft"] and rows["kto"]
    assert all(r["credit"] is False and r["weight"] == 0.0 for f in ("sft", "kto") for r in rows[f])
    rewards = {r["trace_id"].rsplit(":", 1)[1]: r for r in _rows(out / "rewards.jsonl")}
    for t in ("T1011", "T1019"):  # wrong answers, reopened by the customer: no partial credit
        assert rewards[t]["reward"] == 0.0 and rewards[t]["reward_with_progress"] == pytest.approx(0.13333, abs=1e-4)
    assert rewards["T1001"]["reward"] == 1.0 and rewards["T1001"]["phi_by_seq"]  # the measurement stays in rewards.jsonl
    assert all(r["channels"]["step"].startswith("unreleased") for r in rewards.values())

    # the same data with both channels released: identical rows, only the numbers differ
    def release(family, traces, credit):
        return channels.FamilyVerdict(family, channels.ChannelVerdict("step", "s", released=True), channels.ChannelVerdict("terminal", "t", released=True))

    monkeypatch.setattr(channels, "decide", release)
    asyncio.run(label(out=tmp_path / "rel", pack="toy"))
    for f in ("sft", "kto", "dpo"):
        assert {r["row_id"] for r in rows[f]} == {r["row_id"] for r in _rows(tmp_path / "rel" / f"{f}.jsonl")}
    assert any(r["credit"] for r in _rows(tmp_path / "rel" / "sft.jsonl"))
