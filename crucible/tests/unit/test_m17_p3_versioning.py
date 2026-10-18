"""P3 (swarm-pipeline/00 §8.3, §11.3 item 6): verifier and task versions with lineage; objectives never mix."""

from __future__ import annotations

import json

import pytest

from crucible.factory.versioning import (
    ObjectiveMixError,
    OfflineGrade,
    TrainingObjective,
    VersionError,
    VersionStore,
    curve,
    merge_rewards,
    objective_id,
    regrade_offline,
)


def test_root_derive_lineage_and_hash_chain(tmp_path):
    st = VersionStore(tmp_path / "v.jsonl")
    r = st.root("t1", "verifier", {"checks": {"c_m1": ["a"]}})
    assert st.root("t1", "verifier", {"checks": {"c_m1": ["a"]}}).id == r.id  # idempotent
    with pytest.raises(VersionError):
        st.root("t1", "verifier", {"checks": {}})  # a second, different root
    with pytest.raises(VersionError):
        st.derive(r.id, content={"checks": {"c_m1": ["a", "b"]}}, change={"type": "add_check"}, proposal_id="", evidence={"x": 1})
    with pytest.raises(VersionError):
        st.derive(r.id, content={"checks": {"c_m1": ["a", "b"]}}, change={"type": "add_check"}, proposal_id="p1", evidence={})
    with pytest.raises(VersionError):
        st.derive(r.id, content={"checks": {"c_m1": ["a"]}}, change={}, proposal_id="p1", evidence={"x": 1})  # unchanged
    v1 = st.derive(r.id, content={"checks": {"c_m1": ["a", "b"]}}, change={"type": "add_check"}, proposal_id="p1", evidence={"oracle": True})
    with pytest.raises(VersionError):  # no forks: the parent must be the head
        st.derive(r.id, content={"checks": {"c_m1": ["c"]}}, change={}, proposal_id="p2", evidence={"x": 1})
    v2 = st.derive(v1.id, content={"checks": {"c_m1": ["a", "b", "c"]}}, change={}, proposal_id="p2", evidence={"x": 2})
    assert [v.id for v in st.lineage(v2.id)] == [r.id, v1.id, v2.id]
    assert v2.parent == v1.id and v2.proposal_id == "p2" and v2.evidence_hash and st.head("t1", "verifier").id == v2.id
    assert st.verify() == []
    again = VersionStore(tmp_path / "v.jsonl")  # reloaded from disk
    assert again.head("t1", "verifier").id == v2.id and again.verify() == []


def test_an_edited_line_breaks_the_store(tmp_path):
    p = tmp_path / "v.jsonl"
    st = VersionStore(p)
    r = st.root("t", "task", {"prompt": "a"})
    st.derive(r.id, content={"prompt": "b"}, change={}, proposal_id="p", evidence={"lint": []})
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    rows[1]["content"]["prompt"] = "c"  # tampered without rehashing
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    probs = VersionStore(p).verify()
    assert any("content hash" in x for x in probs) and any("edited" in x for x in probs)
    p.write_text(json.dumps(rows[1]) + "\n")  # the root dropped
    assert any("chain is broken" in x or "parent" in x for x in VersionStore(p).verify())


def test_objective_pins_versions_and_changes_are_new_objectives():
    o = TrainingObjective("run1", {"t1": "va", "t2": "vb"})
    assert o.objective_id == objective_id({"t2": "vb", "t1": "va"})
    o.record(episode="e1", task="t1", verifier_version="va", reward=0.5, step=1)
    with pytest.raises(ObjectiveMixError):
        o.record(episode="e2", task="t1", verifier_version="va2", reward=1.0, step=2)  # graded by another version
    with pytest.raises(ObjectiveMixError):
        o.record(episode="e3", task="t9", verifier_version="vz", reward=1.0, step=2)
    with pytest.raises(ObjectiveMixError):
        o.after_change({"t1": "va2"}, change_step=10, resume_from_step=10)  # the checkpoint must predate the change
    with pytest.raises(ObjectiveMixError):
        o.after_change({"t1": "va"}, change_step=10)  # nothing changed
    resumed = o.after_change({"t1": "va2"}, change_step=10, resume_from_step=8)
    assert resumed.run_id == "run1@8" and resumed.parent == o.objective_id and resumed.objective_id != o.objective_id
    with pytest.raises(ObjectiveMixError):
        resumed.record(episode="e4", task="t1", verifier_version="va2", reward=1.0, step=8)  # not after the checkpoint
    resumed.record(episode="e4", task="t1", verifier_version="va2", reward=1.0, step=9)
    fresh = o.after_change({"t1": "va2"}, change_step=10)
    assert fresh.run_id.startswith("run1+") and fresh.resumed_from_step is None
    assert curve(o.rows) == [(1, 0.5)]
    with pytest.raises(ObjectiveMixError):
        merge_rewards(o.rows, resumed.rows)  # two verifier versions never share a curve


async def test_offline_regrade_is_reporting_only():
    async def grade(sol):
        return 1.0 if sol == "good" else 0.0

    rows = await regrade_offline([{"episode": "e1", "task": "t1", "solution": "good", "reward": 0.0, "verifier_version": "va"}], grade, version="va2")
    assert rows == [OfflineGrade("e1", "t1", "va2", 1.0, 0.0, "va", True)]
    o = TrainingObjective("run1", {"t1": "va2"})
    with pytest.raises(ObjectiveMixError):
        o.accept(rows[0])
    with pytest.raises(ObjectiveMixError):
        curve(rows)  # type: ignore[arg-type]
