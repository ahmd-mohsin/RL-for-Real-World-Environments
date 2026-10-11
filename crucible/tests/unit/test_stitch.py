"""m13 G-10: stitch. Synthetic spans stitch back to the seed (nested spans merge, residual helpers go into the oracle);
conflicts, inconsistent starts and uncovered files are errors; the real task04 spans reproduce task04 v3 file for file
with the seed's 11 gates (skipped when the local task data is not present)."""

from __future__ import annotations

import pytest

from crucible.forge.stitch import Span, stitch

SEED = {"a/x": "dx", "a/y": "dy", "b/z": "dz", "a/n/k": "dk", "shared/h": "dh"}


def _spans():
    def span(i, roots, gates):
        own = {p: d for p, d in SEED.items() if p.startswith(roots)}
        return Span(i, {p: d for p, d in SEED.items() if p not in own}, own, gates)

    return [span("A", ("a/",), ["static", "a"]), span("N", ("a/n/",), ["static", "n"]), span("B", ("b/",), ["static", "b", "perf"])]


def test_stitch_reproduces_the_seed():
    st = stitch(_spans(), gate_order=["static", "a", "n", "b", "perf"])
    assert st.files() == SEED and st.start == {} and st.residual == ["shared/h"]
    assert st.nested == [("N", "A")] and st.owners["a/n/k"] == ["A", "N"] and st.gates == ["static", "a", "n", "b", "perf"]
    easy = stitch(_spans(), residual="start")
    assert easy.start == {"shared/h": "dh"} and "shared/h" not in easy.oracle and easy.files() == SEED


def test_stitch_errors():
    sp = _spans()
    sp[1].oracle = {**sp[1].oracle, "a/n/k": "other"}
    with pytest.raises(ValueError, match="differently: cut from different end states"):
        stitch(sp)
    sp = _spans()
    sp[2].start = {**sp[2].start, "a/x": "stale"}
    with pytest.raises(ValueError, match="pre-applies a/x differently"):
        stitch(sp)
    sp = _spans()
    sp[0].start = {**sp[0].start, "extra/only_here": "d"}
    with pytest.raises(ValueError, match="only some starts"):
        stitch(sp)
    sp = _spans()
    sp[0].start = {**sp[0].start, "shared/h": "dh2"}
    with pytest.raises(ValueError, match="disagree on the residual"):
        stitch(sp)


def test_task04_spans_stitch_back_to_v3():
    from crucible_pack_coding.forge import stitch_task04

    if not all((stitch_task04.SPANS / s / "start.patch").exists() for s in stitch_task04.SCOPES):
        pytest.skip("task04 span tasks not derived locally (python -m crucible_pack_coding.forge.derive)")
    try:
        rep = stitch_task04.run(write=False)
    except FileNotFoundError as ex:
        pytest.skip(f"task04 seed data not present: {ex}")
    assert rep["reproduces_v3"], {k: rep[k] for k in ("missing_vs_v3", "extra_vs_v3", "differing_vs_v3", "gates")}
    assert rep["nested"] == [["ncv", "people"]] or rep["nested"] == [("ncv", "people")]
