"""m13 perturb: a cosmetic distractor leaves the unperturbed oracle passing; a semantic policy amendment re-derives the
oracle (A1 holds), states its property, and makes the unperturbed oracle fail at ``policy`` (A18); a mislabelled
perturbation is caught."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from crucible.accept.policy import Policy
from crucible.accept.run import accept_task
from crucible.forge.perturb import Perturbation


@pytest.fixture(scope="module")
def t1():
    import asyncio

    from crucible_pack_toy.forge.world_synth import forge

    col: list = []
    asyncio.run(forge(collect=col))
    return next(t for t, _ in col if t.task_id.endswith("t1_duplicate"))


async def _accept(task, only):
    return await accept_task(task, policy=Policy(required=list(only)), cache_root=Path(tempfile.mkdtemp()), only=only, reuse=False)


async def test_cosmetic_distractor(t1, tmp_path):
    from crucible_pack_toy.forge.perturbed import distractor

    t = distractor(t1, tmp_path / "d")
    assert await t.start_fingerprint() != await t1.start_fingerprint()  # the World really changed
    r = await _accept(t, ["A1", "A18"])
    assert {k: str(v.status) for k, v in r.results.items()} == {"A1": "pass", "A18": "pass"}, {k: v.reasons for k, v in r.results.items()}


async def test_semantic_policy_amendment(t1, tmp_path):
    from crucible_pack_toy.forge.perturbed import policy_amendment

    t = policy_amendment(t1, tmp_path / "p")
    assert "whatever the amount" in t.instruction and t.properties[-1].id == "perturb_duplicate_needs_approval"
    assert "FIN-7a" in (t.pack_root / "data" / "kb" / "refund_policy.md").read_text()
    r = await _accept(t, ["A1", "A18"])
    assert {k: str(v.status) for k, v in r.results.items()} == {"A1": "pass", "A18": "pass"}, {k: v.reasons for k, v in r.results.items()}
    assert "policy" in r.results["A18"].measured["failing"]
    # mislabelled: the same World change called cosmetic is caught (the old oracle's verdict changed)
    t.perturbation = Perturbation("duplicate_needs_approval", "policy_amendment", semantic=False)
    r2 = await _accept(t, ["A18"])
    assert str(r2.results["A18"].status) == "fail" and "changed the verdict" in r2.results["A18"].reasons[0]


async def test_semantic_label_on_a_cosmetic_change_is_caught(t1, tmp_path):
    from crucible_pack_toy.forge.perturbed import distractor

    t = distractor(t1, tmp_path / "d2")
    t.perturbation = Perturbation("extra_open_ticket", "distractor", semantic=True, sentence="Leave ticket 97 alone.", gate="scope")
    r = await _accept(t, ["A18"])
    assert str(r.results["A18"].status) == "fail" and "changed nothing the grader sees" in r.results["A18"].reasons[0]
    assert (await _accept(t1, ["A18"])).results["A18"].na_reason == "not_perturbed"
