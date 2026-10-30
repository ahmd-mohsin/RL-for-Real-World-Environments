"""factory.synth_student_wire: the student and proxy binding (scripted chat clients, no Tinker call), the family
rules, the debug-lane bridges."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from crucible.factory.failures import FailureFinding
from crucible.factory.models import SameFamilyError
from crucible.factory.synth_student_wire import (
    P2_CAP_USD,
    StudentWire,
    alt_history,
    attribution_from_finding,
    band_from_keep_rule,
    check_families,
)
from crucible.synth.falsify import FalsifyContext
from tests.unit import synth_fixtures as fx


class _Resp:
    def __init__(self, text: str):
        self.text, self.cost_usd, self.raw_finish = text, 0.001, "stop"


class FakeChat:
    """An LLMClient-shaped chat client replaying π*'s calls of t1 one per turn, then finishing."""

    def __init__(self, calls: list[tuple[str, dict[str, Any]]]):
        self.calls, self.n, self.seeds = calls, 0, []

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> _Resp:
        self.seeds.append(kw.get("seed"))
        i = self.n
        self.n += 1
        if i < len(self.calls):
            tool, args = self.calls[i]
            return _Resp(json.dumps({"call": {"tool": tool, "args": args}}))
        return _Resp(json.dumps({"done": True, "final": None}))


def test_family_rules() -> None:
    check_families("tinker:Qwen/Qwen3.6-35B-A3B", ("tinker:nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",))
    with pytest.raises(SameFamilyError):
        check_families("tinker:Qwen/Qwen3.6-35B-A3B", ("tinker:Qwen/Qwen3.5-397B-A17B",))
    with pytest.raises(SameFamilyError):
        check_families("tinker:openai/gpt-oss-120b", ())


def test_student_probe_over_a_scripted_chat_client(tmp_path: Path) -> None:
    t = fx.t1_hold()
    calls = [("ap.find_invoice", {"number": "INV-1003"}), ("ap.get_purchase_order", {"po_id": "PO-3"}),
             ("ap.hold_invoice", {"invoice_id": 103, "reason": "price_variance"})]
    with pytest.raises(ValueError):
        StudentWire(tmp_path, cap_usd=P2_CAP_USD + 1)
    wire = StudentWire(tmp_path, cap_usd=1.0, make=lambda model, role: FakeChat(calls))
    probe = wire.probe(FalsifyContext, concurrency=1)
    assert probe.student == "Qwen/Qwen3.6-35B-A3B" and probe.family == "qwen" and "T=1.0" in probe.simulator
    rs = asyncio.run(probe.rollouts(t, 1))
    assert rs[0].passed is True and rs[0].calls == 3 and rs[0].cost_usd == pytest.approx(0.004)  # 3 calls + the finish
    assert wire.cost()["by_role"] == {"student": 0.004}
    px = wire.proxy_probe(FalsifyContext, concurrency=1)
    assert px.family == "nemotron"


def test_rollouts_are_independent_samples(tmp_path: Path) -> None:
    """The P2 pilot's bug: every rollout was sampled with seed 0, so k rollouts were one trajectory copied k times."""
    chat = FakeChat([])
    wire = StudentWire(tmp_path, cap_usd=1.0, make=lambda model, role: chat)
    probe = wire.probe(FalsifyContext, concurrency=1)
    asyncio.run(probe.rollouts(fx.t1_hold(), 3))
    assert len(set(chat.seeds)) == 3 and all(isinstance(s, int) and 0 <= s < 2**31 for s in chat.seeds)
    assert "seed=tag" in probe.simulator


def test_debug_lane_bridges() -> None:
    f = FailureFinding(id="ff1", trace="tr", task="db-x", window="w", policy="p", family="qwen", student="Qwen/Qwen3.6-35B-A3B", mode="lost_progress",
                       cause=None, cause_source="facts", quote="q", milestones_missed=["m02"], first_wrong_step={"kind": "lost", "step": 7, "files": ["a.py"]})
    a = attribution_from_finding(f, cell="debug")
    assert a.lane == "debug" and a.kind == "irreversible" and a.mode == "frame_discipline" and a.step == 7 and a.routed is None and a.id.startswith("at-")
    r = attribution_from_finding(FailureFinding(id="ff2", trace="tr", task="db-x", window="w", policy="p", family="qwen", student="s", mode="budget",
                                                cause=None, cause_source="facts", quote="q"))
    assert r.routed == "limits" and r.mode is None and r.kind == "budget"
    b = band_from_keep_rule()
    assert (b.lo, b.hi, b.looks) == (0.1, 0.6, (4, 8))
    assert alt_history({"certificate": {"classes": {"ALT": {"n": 3, "passes": 2}}}}) == (2, 3)
    assert alt_history({}) is None
