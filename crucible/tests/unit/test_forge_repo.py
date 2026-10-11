"""m13 G-7: the zoned forge repo. A Verifier-writer commit quoting an oracle line fails as a breach (and writes nothing);
citations into unreadable zones are rejected; views export only readable zones; GitSwarm commit format; reopen keeps the
index."""

from __future__ import annotations

import json
import os

import pytest

from crucible.forge.repo import ForgeRepo, ZoneBreach

ORACLE = json.dumps([{"kind": "tool_call", "function_name": "tickets.issue_refund", "arguments": {"ticket_id": 7, "amount_usd": 42.5}},
                     {"kind": "tool_call", "function_name": "tickets.set_status", "arguments": {"ticket_id": 7, "status": "resolved"}}], indent=1)
SPEC = "Refund the duplicate charge on ticket 7 and resolve the ticket.\nThe customer must be notified of the outcome."


@pytest.fixture
def repo(tmp_path):
    r = ForgeRepo(tmp_path / "toy", family="toy")
    spec = r.commit("proposer", {"public/spec.md": SPEC}, type_="spec", summary="t7 duplicate refund")
    orc = r.commit("builder", {"oracle/t7.json": ORACLE, "oracle/notes.md": SPEC.splitlines()[0] + "\n"}, type_="oracle", summary="t7 oracle",
                   informed_by=[spec.sha], parent=spec.sha)
    return r, spec, orc


def test_quoting_an_oracle_line_is_a_breach(repo):
    r, spec, _orc = repo
    n = len(r.commits())
    leak = 'SELECT 1 WHERE 1\n  "function_name": "tickets.issue_refund",\n'
    with pytest.raises(ZoneBreach) as ex:
        r.commit("verifier_writer", {"checks/t7.sql": leak}, type_="checks", summary="t7 checks", informed_by=[spec.sha], parent=spec.sha)
    assert any("quotes a line of oracle/" in x for x in ex.value.reasons)
    assert len(r.commits()) == n and r.breaches[-1]["role"] == "verifier_writer"  # nothing was written
    with pytest.raises(ZoneBreach, match="canary"):
        r.commit("verifier_writer", {"checks/x.txt": f"see {r._index['canaries']['oracle']}"}, type_="checks", summary="x")
    # a line that the oracle shares with the public spec is not evidence; clean checks commit fine
    ok = r.commit("verifier_writer", {"checks/t7.sql": "SELECT COUNT(*) FROM refunds WHERE ticket_id = 7 AND amount_usd = 42.5\n" + SPEC.splitlines()[0]},
                  type_="checks", summary="t7 checks", informed_by=[spec.sha], parent=spec.sha)
    assert r.parse_message(ok.sha) == {"type": "checks", "informed_by": [spec.sha], "summary": "t7 checks"}
    assert ok.parent == spec.sha and ok.zones == ["checks"]


def test_citations_and_writes_into_unreadable_zones_are_rejected(repo):
    r, _spec, orc = repo
    with pytest.raises(ZoneBreach, match="cites a commit in \\['oracle'\\]"):
        r.commit("verifier_writer", {"checks/a.sql": "SELECT 1"}, type_="checks", summary="a", informed_by=[orc.sha])
    with pytest.raises(ZoneBreach, match="unknown commit"):
        r.commit("verifier_writer", {"checks/a.sql": "SELECT 1"}, type_="checks", summary="a", informed_by=["0" * 40])
    with pytest.raises(ZoneBreach, match="cannot read zone oracle/"):
        r.commit("verifier_writer", {"oracle/a.json": "[]"}, type_="oracle", summary="a")
    with pytest.raises(ZoneBreach, match="not in a zone"):
        r.commit("builder", {"elsewhere/a": "x"}, type_="x", summary="a")
    adv = r.commit("adversary", {"checks/neg.json": "[]"}, type_="negatives", summary="negatives", informed_by=[orc.sha], parent=orc.sha)
    assert adv.informed_by == [orc.sha]  # the adversary reads oracle/ and may cite it


def test_views_are_zoned_and_read_only(repo, tmp_path):
    r, _spec, orc = repo
    assert set(r.view("verifier_writer", orc.sha)) == {"public/CANARY", "public/spec.md", "checks/CANARY", "solver/CANARY"}
    assert "oracle/t7.json" in r.view("builder", orc.sha) and "oracle/t7.json" not in r.view("hacker_blackbox", orc.sha)
    d = r.materialize("hacker_blackbox", orc.sha, tmp_path / "mnt")
    files = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())
    assert files == ["public/CANARY", "public/spec.md"] and not os.access(d / "public/spec.md", os.W_OK)


def test_reopen_keeps_index_and_scans_the_same(repo):
    r, _spec, _orc = repo
    r2 = ForgeRepo(r.path, family="toy")
    assert r2.root == r.root and {c.sha for c in r2.commits()} == {c.sha for c in r.commits()}
    with pytest.raises(ZoneBreach, match="quotes a line of oracle/"):
        r2.commit("verifier_writer", {"checks/t.sql": '"function_name": "tickets.set_status",'}, type_="checks", summary="t")
