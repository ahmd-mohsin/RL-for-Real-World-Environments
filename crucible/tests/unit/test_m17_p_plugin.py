"""M17 P: the label routes of the Crucible explorer plugin (/api/v1/crucible/labels), with FastAPI's TestClient."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("agent_env")
pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from crucible.factory.explorer import LabelsExplorerPlugin, read_items
from crucible.factory.labels import LabelStore, item_sha256

P = "/api/v1/crucible/labels"
ITEMS = [{"id": "i1", "family": "fam1", "payload": {"trace": "t1"}, "choices": ["pass", "fail"]},
         {"id": "i2", "family": "fam1", "payload": {"trace": "t2"}, "blind": True},
         {"id": "i3", "family": "fam2", "payload": {"trace": "t3"}}]
H = {r["id"]: item_sha256(r["payload"]) for r in ITEMS}
ALICE, BOB = {"X-Labeller": "human:alice"}, {"X-Labeller": "human:bob"}


def make(tmp_path, *, header="X-Labeller", secret=None, items=ITEMS, raters=2):
    (tmp_path / "factory").mkdir(exist_ok=True)
    (tmp_path / "factory" / "label_items.jsonl").write_text("\n".join(json.dumps(r) for r in items) + "\n")
    plugin = LabelsExplorerPlugin(root=tmp_path, identity_header=header, identity_secret=secret, raters=raters)
    app = FastAPI()
    app.include_router(plugin.router)
    return plugin, TestClient(app)


def post(c, item="i1", value="pass", sha=None, headers=ALICE, **kw):
    return c.post(f"{P}/label", json={"item": item, "item_sha256": sha or H[item], "value": value, **kw}, headers=headers)


def test_plugin_registers_like_the_crucible_plugin(tmp_path):
    plugin, _ = make(tmp_path)
    assert plugin.type == "crucible_labels" and plugin.router.prefix == P
    assert plugin.router is plugin.router


def test_queue_lists_items_needing_labels(tmp_path):
    _, c = make(tmp_path)
    q = c.get(f"{P}/queue").json()
    assert q["total"] == 3 and [r["item"] for r in q["items"]] == ["i1", "i2", "i3"]
    assert q["items"][0]["item_sha256"] == H["i1"] and q["items"][0]["choices"] == ["pass", "fail"] and q["labeller"] is None
    assert [r["item"] for r in c.get(f"{P}/queue", params={"family": "fam2"}).json()["items"]] == ["i3"]
    assert len(c.get(f"{P}/queue", params={"limit": 1, "offset": 1}).json()["items"]) == 1
    assert post(c).status_code == 200
    # alice has labelled i1: gone from her queue, still in bob's (one rater of two); an anonymous queue still lists it
    assert [r["item"] for r in c.get(f"{P}/queue", headers=ALICE).json()["items"]] == ["i2", "i3"]
    assert [r["item"] for r in c.get(f"{P}/queue", headers=BOB).json()["items"]] == ["i1", "i2", "i3"]
    assert post(c, headers=BOB).status_code == 200  # the pair is complete
    assert [r["item"] for r in c.get(f"{P}/queue", headers=BOB).json()["items"]] == ["i2", "i3"]
    assert c.get(f"{P}/queue", params={"limit": 0}).status_code == 422


def test_post_label_roundtrip_and_store_contents(tmp_path):
    plugin, c = make(tmp_path)
    r = post(c, reason="looks right")
    assert r.status_code == 200
    lab = r.json()
    assert lab["labeller"] == "human:alice" and lab["item"] == "i1" and lab["item_sha256"] == H["i1"] and lab["reason"] == "looks right"
    assert LabelStore(tmp_path / "factory" / "labels.jsonl").labels("i1")[0].id == lab["id"]
    assert plugin.store.verify() == 1


def test_labeller_comes_only_from_the_header_never_the_body(tmp_path):
    _, c = make(tmp_path)
    r = c.post(f"{P}/label", json={"item": "i1", "item_sha256": H["i1"], "value": "pass", "labeller": "human:mallory"}, headers=ALICE)
    assert r.status_code == 422  # extra fields are refused
    assert post(c).json()["labeller"] == "human:alice"


def test_labelling_is_refused_when_no_identity_header_is_configured(tmp_path):
    _, c = make(tmp_path, header=None)
    r = post(c, headers={"X-Labeller": "human:alice"})
    assert r.status_code == 403 and "not configured" in r.json()["detail"]
    assert c.get(f"{P}/queue").status_code == 200  # reads stay open


def test_missing_or_malformed_identity(tmp_path):
    _, c = make(tmp_path)
    assert post(c, headers={}).status_code == 401
    assert post(c, headers={"X-Labeller": "alice"}).status_code == 400
    assert post(c, headers={"X-Labeller": "human:"}).status_code == 400
    assert post(c, headers={"X-Labeller": "root:alice"}).status_code == 400


def test_identity_secret_stops_a_forged_header(tmp_path):
    _, c = make(tmp_path, secret="s3cret")
    assert post(c, headers=ALICE).status_code == 403
    assert post(c, headers={**ALICE, "X-Crucible-Identity-Secret": "wrong"}).status_code == 403
    ok = post(c, headers={**ALICE, "X-Crucible-Identity-Secret": "s3cret"})
    assert ok.status_code == 200 and ok.json()["labeller"] == "human:alice"


def test_label_on_stale_item_hash_is_409_with_the_current_hash(tmp_path):
    plugin, c = make(tmp_path)
    r = post(c, sha=item_sha256({"trace": "OLD"}))
    assert r.status_code == 409 and r.json()["detail"]["current_item_sha256"] == H["i1"]
    assert plugin.store.labels() == []
    # the item changes on disk after alice labelled: her label is now stale and the queue offers it again
    assert post(c).status_code == 200
    changed = [{**ITEMS[0], "payload": {"trace": "t1-v2"}}, *ITEMS[1:]]
    _, c2 = make(tmp_path, items=changed)
    new = item_sha256({"trace": "t1-v2"})
    assert "i1" in [r["item"] for r in c2.get(f"{P}/queue", headers=ALICE).json()["items"]]
    item = c2.get(f"{P}/item", params={"id": "i1"}).json()
    assert item["item_sha256"] == new and item["labels"][0]["fresh"] is False
    first = item["labels"][0]["id"]
    assert post(c2, sha=H["i1"]).status_code == 409  # still the old hash
    assert post(c2, sha=new).status_code == 409  # a second independent label by alice is refused: she must supersede
    ok = c2.post(f"{P}/label", json={"item": "i1", "item_sha256": new, "value": "fail", "supersedes": first}, headers=ALICE)
    assert ok.status_code == 200
    assert c2.get(f"{P}/item", params={"id": "i1"}).json()["labels"][0]["fresh"] is True


def test_supersede_rules_over_http(tmp_path):
    _, c = make(tmp_path)
    mine = post(c).json()["id"]
    theirs = post(c, headers=BOB).json()["id"]
    sup = lambda target, who: c.post(f"{P}/label", json={"item": "i1", "item_sha256": H["i1"], "value": "fail", "supersedes": target}, headers=who)
    assert sup(theirs, ALICE).status_code == 409  # cannot supersede another labeller's label
    assert sup("lbl-nope", ALICE).status_code == 409
    assert sup(mine, ALICE).status_code == 200
    assert sup(mine, ALICE).status_code == 409  # already superseded: no fork
    assert post(c, item="i3", headers=ALICE).status_code == 200
    r = c.post(f"{P}/label", json={"item": "i3", "item_sha256": H["i3"], "value": "fail", "supersedes": mine}, headers=ALICE)
    assert r.status_code == 409  # a label on another item


def test_choices_unknown_item_and_bad_bodies(tmp_path):
    _, c = make(tmp_path)
    assert post(c, value="maybe").status_code == 422
    assert post(c, item="i2", value="anything").status_code == 200  # no choices declared
    r = c.post(f"{P}/label", json={"item": "nope", "item_sha256": "0" * 64, "value": "pass"}, headers=ALICE)
    assert r.status_code == 404
    assert c.post(f"{P}/label", json={"item": "i1"}, headers=ALICE).status_code == 422
    assert c.post(f"{P}/label", json={"item": "i1", "item_sha256": "short", "value": "pass"}, headers=ALICE).status_code in (409, 422)
    assert c.post(f"{P}/label", json={"item": "i1", "item_sha256": H["i1"], "value": "pass", "reason": "x" * 10_001}, headers=ALICE).status_code == 422


def test_blind_items_hide_other_labellers_labels(tmp_path):
    _, c = make(tmp_path)
    post(c, item="i2", value="pass", headers=ALICE)
    post(c, item="i1", value="pass", headers=ALICE)
    blind = c.get(f"{P}/item", params={"id": "i2"}, headers=BOB).json()
    assert blind["blind"] and blind["labels"] == [] and blind["hidden_labels"] == 1
    assert [x["labeller"] for x in c.get(f"{P}/item", params={"id": "i2"}, headers=ALICE).json()["labels"]] == ["human:alice"]
    assert [x["labeller"] for x in c.get(f"{P}/item", params={"id": "i1"}, headers=BOB).json()["labels"]] == ["human:alice"]  # not blind
    assert c.get(f"{P}/item", params={"id": "nope"}).status_code == 404
    assert c.get(f"{P}/item", params={"id": "i1"}).json()["payload"] == {"trace": "t1"}


def test_decisions_listing_and_filters(tmp_path):
    plugin, c = make(tmp_path)
    plugin.store.add_decision(subject="t1", actor="human:alice", action="drop", reason="amb", criteria=["ambiguity"], recipe_version="r1")
    plugin.store.add_decision(subject="t2", actor="model:judge", action="keep", reason="ok", criteria=["format"], recipe_version="r2")
    assert c.get(f"{P}/decisions").json()["total"] == 2
    assert [d["subject"] for d in c.get(f"{P}/decisions", params={"criterion": "ambiguity"}).json()["decisions"]] == ["t1"]
    assert [d["subject"] for d in c.get(f"{P}/decisions", params={"recipe_version": "r2"}).json()["decisions"]] == ["t2"]
    assert c.get(f"{P}/decisions", params={"criterion": "ambiguity", "recipe_version": "r2"}).json()["decisions"] == []
    assert c.get(f"{P}/decisions", params={"actor": "model:judge", "action": "keep"}).json()["total"] == 1
    assert [d["subject"] for d in c.get(f"{P}/decisions", params={"limit": 1}).json()["decisions"]] == ["t2"]  # the latest


def test_kappa_per_family(tmp_path):
    _, c = make(tmp_path, items=[{"id": f"a{n}", "family": "famA", "payload": n} for n in range(10)] + [{"id": "b0", "family": "famB", "payload": 0}])
    votes = [("pass", "pass")] * 4 + [("pass", "fail")] * 1 + [("fail", "pass")] * 1 + [("fail", "fail")] * 4
    for n, (a, b) in enumerate(votes):
        for who, v in ((ALICE, a), (BOB, b)):
            r = c.post(f"{P}/label", json={"item": f"a{n}", "item_sha256": item_sha256(n), "value": v}, headers=who)
            assert r.status_code == 200
    k = c.get(f"{P}/kappa").json()["families"]
    # po = 0.8, pe = 0.5 -> kappa 0.6
    assert k["famA"]["method"] == "cohen" and k["famA"]["kappa"] == pytest.approx(0.6) and k["famA"]["n_items"] == 10
    assert k["famA"]["observed_agreement"] == pytest.approx(0.8) and k["famA"]["raters"] == ["human:alice", "human:bob"]
    assert k["famB"]["kappa"] is None and k["famB"]["n_items"] == 0 and k["famB"]["items_in_family"] == 1
    assert list(c.get(f"{P}/kappa", params={"family": "famB"}).json()["families"]) == ["famB"]
    assert c.get(f"{P}/kappa", params={"family": "famA", "raters": "human:alice,model:judge"}).json()["families"]["famA"]["kappa"] is None


def test_corrupt_store_is_a_500_without_paths(tmp_path):
    _, c = make(tmp_path)
    post(c)
    p = tmp_path / "factory" / "labels.jsonl"
    p.write_text(p.read_text().replace("pass", "fail"))
    for r in (c.get(f"{P}/queue"), c.get(f"{P}/decisions"), c.get(f"{P}/kappa"), c.get(f"{P}/item", params={"id": "i1"}), post(c, headers=BOB)):
        assert r.status_code == 500 and str(tmp_path) not in r.text


def test_item_file_loader_skips_bad_lines_and_hashes_the_payload(tmp_path):
    f = tmp_path / "items.jsonl"
    f.write_text("\n".join([json.dumps({"id": "ok", "family": "f", "payload": {"a": 1}, "sha256": "forged"}), "not json",
                            json.dumps({"id": "no family", "family": "f", "payload": 1}), json.dumps({"id": "ok", "family": "f", "payload": 2}),
                            json.dumps({"family": "f"})]))
    items = read_items(f)
    assert [i.id for i in items] == ["ok"] and items[0].sha256 == item_sha256({"a": 1})  # the first wins; the hash is computed
    assert read_items(tmp_path / "absent.jsonl") == []


def test_from_config_reads_settings(tmp_path, monkeypatch):
    import crucible.factory.explorer as mod

    monkeypatch.setattr(mod, "_settings", lambda: {"label_identity_header": "X-Who", "label_identity_secret": "k", "label_raters": 3})
    p = LabelsExplorerPlugin.from_config()
    assert (p.identity_header, p.identity_secret, p.raters) == ("X-Who", "k", 3)
    monkeypatch.setattr(mod, "_settings", dict)
    assert LabelsExplorerPlugin.from_config().identity_header is None  # untrusted unless configured
