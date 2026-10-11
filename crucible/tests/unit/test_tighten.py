"""G-tighten: a tightened bar must keep A1's 15% oracle headroom and must fail the solution that solved the task before
(the task04 v1 -> v3 pattern; the ratios here are illustrative, not measured)."""

from __future__ import annotations

from crucible.forge.tighten import add_gate, tighten_perf, tightest_bar


def test_tightest_bar_keeps_exactly_the_headroom():
    b = tightest_bar(0.53)
    assert b == 0.63 and 1 - 0.53 / b >= 0.15 and 1 - 0.53 / (b - 0.01) < 0.15


def test_tighten_perf_fair_and_separating():
    d = tighten_perf({"kinfu_frame_ms": 0.9}, {"kinfu_frame_ms": 0.53}, {"kinfu_frame_ms": 0.71})
    assert d.ok and d.bars == {"kinfu_frame_ms": 0.63} and d.separates == {"kinfu_frame_ms": True} and d.headroom["ok"]
    unfair = tighten_perf({"kinfu_frame_ms": 0.9}, {"kinfu_frame_ms": 0.53}, {"kinfu_frame_ms": 0.71}, new={"kinfu_frame_ms": 0.58})
    assert not unfair.ok and "A1 requires 0.15" in unfair.reasons[0]
    useless = tighten_perf({"kinfu_frame_ms": 0.9}, {"kinfu_frame_ms": 0.53}, {"kinfu_frame_ms": 0.6})
    assert not useless.ok and "not separating" in useless.reasons[-1]
    looser = tighten_perf({"kinfu_frame_ms": 0.6}, {"kinfu_frame_ms": 0.4}, {"kinfu_frame_ms": 0.7}, new={"kinfu_frame_ms": 0.65})
    assert not looser.ok and "not stricter" in looser.reasons[0]


def test_add_gate():
    assert add_gate("concurrency", [True] * 3, [True, False]).ok
    assert "not fair" in add_gate("concurrency", [True, False, True], [False]).reasons[0]
    assert "not separating" in add_gate("sanitizer", [True] * 3, [True, True]).reasons[0]
    assert "need >= 3" in add_gate("sanitizer", [True], [False]).reasons[0]
