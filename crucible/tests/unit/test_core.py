"""M0 core: schemas round-trip, validation catches mistakes, ids are stable, store/registry/config behave."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from crucible.core import ids
from crucible.core.config import Config, resolve_secret
from crucible.core.prov import Provenance, SourceRef
from crucible.core.registry import Registry, RegistryError, get_registry
from crucible.core.schema import ALL_DOCUMENTS
from crucible.core.schema.milestone import Milestone, MilestoneCheck, MilestoneSet
from crucible.core.schema.reward import RewardJson
from crucible.core.schema.run import Failure, RunRecord
from crucible.core.schema.state import Query
from crucible.core.schema.trace import Trace, records_from_trace
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store, StoreError

WORLD = {
    "world": "toy/helpdesk",
    "apps": [
        {"id": "tickets", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "tickets.db"}}},
        {"id": "kb", "template": "common/doc_library", "state": {"adapter": "fs", "params": {"root": "kb"}}},
    ],
    "actors": [
        {"id": "agent", "kind": "policy", "tools": ["tickets.*", "kb.read*"]},
        {"id": "customer", "kind": "simulated", "grounded_on": ["traces/tickets"]},
    ],
    "clock": {"mode": "step", "t0": "2026-03-02T09:00:00Z", "step_dt_s": 60},
    "events": [
        {"id": "nudge", "when": {"after": "2h"}, "actor": "customer", "action": {"type": "nl", "text": "any update?", "to": "agent"}}
    ],
}


def test_world_roundtrip_and_hash_stability():
    w = WorldSpec.model_validate(WORLD)
    assert w.schema_version == "crucible.world/v1"
    w2 = WorldSpec.from_yaml(w.to_yaml())
    assert w2 == w and w2.content_hash() == w.content_hash()
    w3 = WorldSpec.model_validate_json(w.model_dump_json())
    assert w3.content_hash() == w.content_hash()
    assert w.actor("agent").allows("tickets.update_ticket") and not w.actor("agent").allows("kb.write_doc")


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda d: d["apps"].append(dict(d["apps"][0])), "duplicate app"),
        (lambda d: d["actors"].__setitem__(0, {**d["actors"][0], "kind": "simulated"}), "policy actor"),
        (lambda d: d["actors"][0]["tools"].append("nope.*"), "unknown app"),
        (lambda d: d.__setitem__("clock", None), "no clock"),
        (lambda d: d["events"][0].__setitem__("actor", "ghost"), "unknown actor"),
        (lambda d: d.__setitem__("typo_field", 1), "Extra inputs"),
        (lambda d: d.__setitem__("schema_version", "crucible.world/v2"), "reads crucible.world/v1"),
    ],
)
def test_world_validation_rejects_mistakes(mutate, msg):
    d = json.loads(json.dumps(WORLD))
    mutate(d)
    with pytest.raises(ValidationError, match=msg):
        WorldSpec.model_validate(d)


def test_all_documents_export_json_schema():
    for name, cls in ALL_DOCUMENTS.items():
        schema = cls.model_json_schema()
        assert schema.get("type") == "object", name


def test_reward_gate_forces_zero():
    RewardJson(reward=0.0, gates={"build": 1, "perf": 0})
    with pytest.raises(ValidationError, match="failed gate"):
        RewardJson(reward=0.7, gates={"build": 1, "perf": 0})
    r = RewardJson(reward=1.05, gates={"build": 1}, outcome=1.0, progress=1.0, rubric={"r1": 1.0})
    assert r.reward_components()["gate/build"] == 1.0


def test_milestone_dag_rejects_cycle():
    chk = MilestoneCheck(app="repo", query=Query(op="exists", args={"path": "x"}))
    MilestoneSet(family="f", milestones=[Milestone(id="a", check=chk), Milestone(id="b", check=chk, requires=["a"])])
    with pytest.raises(ValidationError, match="cycle"):
        MilestoneSet(family="f", milestones=[Milestone(id="a", check=chk, requires=["b"]), Milestone(id="b", check=chk, requires=["a"])])


def test_trace_validation_and_records():
    t = Trace.model_validate(
        {
            "trace_id": "pr-1",
            "source": {"connector": "github", "ref": "o/r#1", "privacy": "public"},
            "actors": [{"id": "a", "role": "author", "kind": "human"}, {"id": "r", "role": "reviewer", "kind": "human"}],
            "events": [
                {"seq": 0, "kind": "action", "actor": "a", "type": "commit", "text": "port kernel"},
                {"seq": 1, "kind": "message", "actor": "r", "type": "review_comment", "text": "OOB read here", "reply_to": 0,
                 "target": {"object": "k.cu", "span": [80, 95]}, "time": "2026-01-02T00:00:00Z"},
            ],
            "signals": [{"id": "s1", "type": "rejection", "polarity": -1, "grades": [0], "source_event": 1}],
        }
    )
    recs = records_from_trace(t, types={"review_comment"})
    assert [r.id for r in recs] == ["pr-1#1"] and recs[0].role == "reviewer" and recs[0].reply_to == "pr-1#0"
    with pytest.raises(ValidationError, match="unknown events"):
        Trace.model_validate({**t.model_dump(), "signals": [{"id": "s", "type": "approval", "polarity": 1, "grades": [9]}]})


def test_run_record_masks_infra_failures():
    r = RunRecord(run_id="r1", world="w", policy="p", started_at="2026-01-01T00:00:00Z")
    r.mark_failure(Failure(kind="crucible:usage_blank", stage="agent"))
    assert r.mask_sample and r.outcome == "masked"
    with pytest.raises(ValueError, match="unknown failure kind"):
        r.mark_failure(Failure(kind="made_up", stage="agent"))


def test_harbor_compatible_manifest_hash():
    # Same scheme as harbor/publisher/packager.py: sha256 over sorted "rel\0hash\n" lines.
    import hashlib

    entries = [("b.txt", "22"), ("a.txt", "11")]
    outer = hashlib.sha256()
    for rel, h in sorted(entries):
        outer.update(f"{rel}\0{h}\n".encode())
    assert ids.manifest_hash(entries) == outer.hexdigest()
    assert ids.tree_manifest_hash([("a", "644", "x")]) != ids.tree_manifest_hash([("a", "755", "x")])


def test_store_blobs_docs_tags(tmp_path):
    s = Store(tmp_path)
    k = s.put_bytes(b"hello")
    assert k == s.put_bytes(b"hello") and s.get_bytes(k) == b"hello"
    w = WorldSpec.model_validate(WORLD)
    prov = Provenance(created_by="test", derived_from=[SourceRef(kind="file", id="x")])
    dk = s.put_doc("world", w, name="toy/helpdesk", prov=prov)
    assert dk == s.put_doc("world", w)  # idempotent
    assert s.get_doc(dk, WorldSpec) == w
    rows = s.list_docs("world", name="toy/helpdesk")
    assert len(rows) == 1 and rows[0].provenance.created_by == "test"
    s.tag("latest", dk)
    assert s.resolve("latest") == dk
    with pytest.raises(StoreError):
        s.resolve("nope")
    with pytest.raises(StoreError):
        s.get_bytes("sha256:" + "0" * 64)
    # corruption is detected
    p = s.blob_path(k)
    p.chmod(0o644)
    p.write_bytes(b"tampered")
    with pytest.raises(StoreError, match="corrupt"):
        s.get_bytes(k)


def test_registry_lazy_and_duplicates():
    r = Registry()
    r.register("thing", "a", "json:dumps")
    assert r.get("thing", "a") is json.dumps
    with pytest.raises(RegistryError, match="already registered"):
        r.register("thing", "a", "json:loads")
    with pytest.raises(RegistryError, match="no thing named"):
        r.get("thing", "b")
    g = get_registry()
    assert "local" in g.names("world_backend") and "harbor" in g.names("world_backend")


def test_config_secrets(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("FOO_KEY='abc'\n# comment\nexport BAR=1\n")
    monkeypatch.delenv("FOO_KEY", raising=False)
    assert resolve_secret("env:FOO_KEY", search_from=tmp_path) == "abc"
    assert resolve_secret("plain") == "plain"
    with pytest.raises(KeyError):
        resolve_secret("env:DEFINITELY_MISSING_KEY_123", search_from=tmp_path)
    c = Config()
    prov, model = c.model("sim-default")
    assert prov.base_url.startswith("https://") and "/" in model
    assert "abc" not in json.dumps(c.redacted())
