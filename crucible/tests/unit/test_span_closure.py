"""m13 structural findings from the forge GPU run, as CPU detectors: a span must be closed under the grader's consumer
dependencies (the KinFu / KinFu-LS spans left a sibling the pristine kinfu_replay consumer links unported, so their
oracles could never pass ``consumers``), and a recorded alternative is evidence about a span only if it depends on the
same outside state as the span's start (Astra's people/NCV ports used its own TextureBinder helpers and could not build)."""

from __future__ import annotations

from crucible_pack_coding.forge.derive import alternative_admissible, closure_violations

NEEDS = {"kinfu_replay": {"gpu/containers", "gpu/kinfu", "gpu/kinfu_large_scale"}, "people_eval": {"gpu/containers", "gpu/people"}}


def test_closure_flags_unported_siblings_a_consumer_needs():
    want = "kinfu_replay needs gpu/kinfu_large_scale, which the kinfu span neither ports in its start (an unported sibling) nor includes"
    assert closure_violations("kinfu", NEEDS) == [want]
    assert closure_violations("kinfu_large_scale", NEEDS)[0].startswith("kinfu_replay needs gpu/kinfu,")
    assert closure_violations("kinfu_family", NEEDS) == [] and closure_violations("people", NEEDS) == []


def test_alternative_admissibility():
    def inside(p):
        return p.startswith("gpu/people/")

    oracle = {"gpu/people/a.cu": "oracle-a", "gpu/containers/tb.hpp": "oracle-helper"}
    same_outside = "diff --git a/gpu/people/a.cu b/gpu/people/a.cu\nalt-a\ndiff --git a/gpu/containers/tb.hpp b/gpu/containers/tb.hpp\noracle-helper\n"
    ok, diff = alternative_admissible(same_outside, {**oracle, "gpu/containers/tb.hpp": "diff --git a/gpu/containers/tb.hpp b/gpu/containers/tb.hpp\noracle-helper\n"},
                                      inside)
    assert ok and diff == []
    own_helpers = "diff --git a/gpu/people/a.cu b/gpu/people/a.cu\nalt-a\ndiff --git a/gpu/containers/tb.hpp b/gpu/containers/tb.hpp\nastra-helper\n"
    ok, diff = alternative_admissible(own_helpers, oracle, inside)
    assert not ok and diff == ["gpu/containers/tb.hpp"]
    inside_only = "diff --git a/gpu/people/a.cu b/gpu/people/a.cu\nalt-a\n"
    assert alternative_admissible(inside_only, oracle, inside) == (True, [])
