"""M3 audit (2026-10-12), batch A: redaction, file_drop, sql_audit_log and ingest. Each test reproduced a defect before
its fix (``docs/audit/2026-10-12/M03-connectors.md`` §2)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from crucible.connectors.base import IngestError, SourceRef, ingest
from crucible.connectors.file_drop import FieldMap, FileDropConnector, rows_to_trace
from crucible.connectors.redact import RedactionPolicy, Redactor, pseudonym, redact
from crucible.connectors.sql_audit_log import SqlAuditLogConnector
from crucible.connectors.trace_store import TraceStore
from crucible.core.schema.signal import SignalTarget
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceSource
from crucible.core.store import Store


@pytest.fixture
def R(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_REDACTION_SALT", "test-tenant-salt")
    return Redactor(RedactionPolicy())


@pytest.mark.parametrize("text,keep", [
    ("Port task-04-gpu-migration-harness to sm_120", "task-04-gpu-migration-harness"),  # A1: became "ta<secret>"
    ("Check disk-usage-monitoring-scripts before merge", "disk-usage-monitoring-scripts"),
    ("use threading.local() for per-thread state", "threading.local()"),  # A2: became a host pseudonym
    ("load icon@2x.png and logo@3x.svg", "icon@2x.png"),  # A3: became an e-mail pseudonym
])
def test_A1_A3_redaction_keeps_ordinary_code_and_words(R, text, keep):
    assert keep in R.text(text)


@pytest.mark.parametrize("text,secret", [
    ("git clone https://alice:S3cr3tPassw0rd@git.example.com/repo.git", "S3cr3tPassw0rd"),  # A4: parsed as an e-mail
    ("DATABASE_URL=postgres://admin:hunter2pass@db.example.com:5432/prod", "hunter2pass"),
    ("password: hunter2", "hunter2"),  # A5: 7 characters passed
    ("AccountName=acme;AccountKey=q9Zx8Yw7Vu6Ts5Rr4Qp3On2Ml1Kk0Jj9Ii8Hh7Gg6Ff5Ee4Dd3Cc2Bb1Aa0==;", "q9Zx8Yw7Vu6Ts5Rr4Qp3On2"),  # A6
    ('"private_key_id": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"', "a1b2c3d4e5f6a7b8c9d0"),
    ("card 4111 1111 1111 1111 exp 12/27", "4111 1111 1111 1111"),  # A7: card numbers (Luhn) and phone numbers
    ("call me at +1 (415) 555-0132", "555-0132"),
])
def test_A4_A7_redaction_removes_more_credential_and_personal_formats(R, text, secret):
    out = R.text(text)
    assert secret not in out, out


def test_A4_url_userinfo_keeps_the_host(R):
    out = R.text("git clone https://alice:S3cr3tPassw0rd@git.example.com/repo.git")
    assert "git.example.com/repo.git" in out and "alice" not in out


def _trace() -> Trace:
    return Trace(trace_id="tickets:alice@acme.com", source=TraceSource(connector="file_drop", ref="alice@acme.com", url="https://bob:pw123456789@crm.acme.internal/t/1"),
                 actors=[TraceActor(id="alice@acme.com", role="customer", kind="human", handle="alice@acme.com")],
                 events=[TraceEvent(seq=0, kind="action", actor="alice@acme.com", type="update", text="ok", target=SignalTarget(object="customer:alice@acme.com"),
                                    state_ref="crm:https://u:pw123456789@x.acme.internal", data={"alice@acme.com": 1, "sk-proj-" + "Z9" * 15: "v"})])


def test_A8_every_field_is_redacted(R):
    d = json.dumps(redact(_trace()).model_dump(mode="json"))  # trace_id, source.url, target, state_ref and dict keys kept them
    for needle in ("alice@acme.com", "pw123456789", "Z9Z9Z9Z9", "acme.internal"):
        assert needle not in d, needle


def test_A9_redaction_is_idempotent(R):
    once = redact(_trace())
    assert redact(once).content_hash() == once.content_hash()  # the handle was pseudonymized again


def test_A10_pseudonyms_use_a_tenant_secret_not_a_public_default(tmp_path, monkeypatch):
    import crucible.connectors.redact as rd

    monkeypatch.delenv("CRUCIBLE_REDACTION_SALT", raising=False)
    monkeypatch.setattr(rd, "salt_path", lambda: tmp_path / "redaction.salt")
    red = Redactor(RedactionPolicy()).text("from admin@acme.com").split()[-1]
    assert red != pseudonym("email", "admin@acme.com", "crucible")  # anyone could reverse a guessed address
    assert Redactor(RedactionPolicy()).text("from admin@acme.com").split()[-1] == red  # stable within the tenant
    assert "salt" not in RedactionPolicy(salt="x").model_dump()  # a stored policy cannot carry the key
    assert (tmp_path / "redaction.salt").stat().st_mode & 0o077 == 0


def test_A11_A14_file_drop_reads_real_exports(tmp_path):
    (tmp_path / "bom.csv").write_bytes("﻿case,time,actor,text\n1,2024-05-01T10:00:00Z,alice,hi\n".encode())
    assert [r.key for r in FileDropConnector().discover({"glob": str(tmp_path / "bom.csv")})] == ["1"]  # A11: KeyError 'case'
    rows = [{"case": "1", "time": "2024-05-01T10:00:00+02:00", "actor": "a", "type": "second", "text": "08:00Z"},
            {"case": "1", "time": "2024-05-01T09:00:00Z", "actor": "a", "type": "first", "text": "09:00Z"}]
    assert [e.text for e in rows_to_trace(rows, FieldMap(), case="1", connector="file_drop").events] == ["08:00Z", "09:00Z"]  # A12: sorted as strings
    t = rows_to_trace([{"case": "1", "time": "2024-05-01T09:00:00Z", "actor": "a", "text": 42}], FieldMap(), case="1", connector="file_drop")
    assert t.events[0].text == "42"  # A13: ValidationError
    (tmp_path / "a.jsonl").write_text('{"case":"1","time":"2024-05-01T10:00:00Z","actor":"a","text":"x"}\n{"time":"2024-05-01T11:00:00Z","text":"no case"}\n')
    c = FileDropConnector()
    refs = list(c.discover({"glob": str(tmp_path / "a.jsonl")}))
    assert [r.key for r in refs] == ["1"] and c.skipped == 1  # A14: KeyError 'case' on one row lost the whole export


def _db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute("create table audit(case_id, time, actor, op, table_name, row_id)")
    con.executemany("insert into audit values (?,?,?,?,?,?)", [("INV-1", "2024-01-01T00:00:00Z", "al", "update", "invoice", "1"),
                                                              (None, "2024-01-02T00:00:00Z", "bo", "delete", "invoice", "2"),
                                                              (None, "2024-01-03T00:00:00Z", "cy", "delete", "po", "9")])
    con.commit()
    con.close()


def test_A15_A17_sql_audit_log(tmp_path):
    q = "select * from audit"
    for name in ("plain.db", "acct#2024.db", "q?.db"):  # A16: '#' and '?' opened another database
        _db(tmp_path / name)
        c = SqlAuditLogConnector(FieldMap(case="case_id", kind="kind", type="op"))
        assert [r.key for r in c.discover({"dsn": str(tmp_path / name), "query": q})] == ["INV-1"]  # A15: NULL cases became one "None" trace
        assert c.skipped == 2
    c = SqlAuditLogConnector(FieldMap(case="case_id", kind="kind", type="op"))
    with pytest.raises(LookupError, match="INV-404"):  # A17: a bare KeyError
        c.fetch(SourceRef(connector="sql_audit_log", key="INV-404", meta={"dsn": str(tmp_path / "plain.db"), "query": q}))


def test_A18_one_bad_source_does_not_abort_an_ingest(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_REDACTION_SALT", "s")
    (tmp_path / "t.jsonl").write_text('{"case":"1","time":"2024-05-01T10:00:00Z","actor":"a","text":"x"}\n')
    good = FileDropConnector()
    refs = [*good.discover({"glob": str(tmp_path / "t.jsonl")}), SourceRef(connector="file_drop", key="2", meta={"files": [str(tmp_path / "gone.jsonl")]})]
    store = TraceStore(Store(tmp_path / "store"))
    with pytest.raises(IngestError) as ei:
        ingest(good, refs, store)
    assert len(ei.value.keys) == 1 and [r.key for r, _ in ei.value.failed] == ["2"]  # the good trace is stored
    errors: list = []
    assert len(ingest(good, refs, store, errors=errors)) == 1 and len(errors) == 1


# ---------------------------------------------------------------- redaction, second pass (from the redaction study)
def test_B1_values_under_secret_shaped_keys_are_removed(R):
    t = Trace(trace_id="t", source=TraceSource(connector="c", ref="r"), events=[TraceEvent(seq=0, kind="action", type="x", data={
        "password": "hunter2hunter2", "headers": {"Authorization": "Basic YWRtaW46aHVudGVyMg==", "Cookie": "sid=abcdef0123456789"},
        "env": {"DB_PASS": "Tr0ub4dor&3"}, "max_tokens": 1200, "token_count": "4096", "author": "dana", "sort_key": "inv-7"})])
    d = redact(t).events[0].data
    assert d["password"] == d["headers"]["Authorization"] == d["headers"]["Cookie"] == d["env"]["DB_PASS"] == "<secret>"
    assert d["max_tokens"] == 1200 and d["token_count"] == "4096" and d["sort_key"] == "inv-7"  # not over-redacted
    assert d["author"] != "<secret>" and d["author"].startswith("<user:")  # a person, pseudonymized, not a secret


@pytest.mark.parametrize("text,secret", [
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA" + "Ab3" * 100, "MIIEowIBAAKCAQEA"),  # B2: a key cut before its END line
    ("Authorization: Basic YWxpY2U6czNjcjN0UGFzcw==", "YWxpY2U6czNjcjN0UGFzcw"),  # B4
    ("Cookie: sessionid=8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c", "8f7a6b5c4d3e2f1a"),
    ("SECRET_KEY=Zq8!kLp2Wm4Xr6Ty", "Zq8!kLp2Wm4Xr6Ty"),  # B5: env-style names
    ("DB_PASS=Zq8kLp2Wm4", "Zq8kLp2Wm4"),
    ('PASSWORD=\\"Zq8kLp2Wm4\\"', "Zq8kLp2Wm4"),
    ("password%3DZq8kLp2Wm4", "Zq8kLp2Wm4"),
    ('password: "correct horse battery staple"', "battery staple"),
    ("ya29.a0AfH6SMBx7Yq8Zr2" + "x" * 40, "a0AfH6SMBx7Yq8Zr2"),  # B6: more formats
    ("hvs.CAESIJ" + "q" * 40, "CAESIJqqqq"),
    ("https://hooks.slack.com/" + "services/T" + "0" * 8 + "/B" + "0" * 8 + "/" + "X" * 24, "X" * 16),  # built at runtime: no literal webhook in the repo
    ("IBAN DE89 3704 0044 0532 0130 00", "3704 0044 0532"),  # B9: personal data in AP invoices
    ("SSN 123-45-6789", "123-45-6789"),
    ("Signed-off-by: Jane Q Public <jane@acme.com>", "Jane Q Public"),
])
def test_B2_B9_more_formats(R, text, secret):
    out = R.text(text)
    assert secret not in out, out


@pytest.mark.parametrize("text", ["token = tokenizer.encode(text)", "password = getpass.getpass()", "pwd = os.getcwd()",
                                  "y = x@self.weights", "git@github.com:org/repo.git", "+1 2024-05-01 release"])
def test_B10_code_is_not_redacted(R, text):
    assert R.text(text) == text


def test_B7_env_files_cover_dsns_and_strip_comments(tmp_path):
    (tmp_path / ".env").write_text("DATABASE_URL=postgres://u:LongPassw0rd@db/x\nOPENAI_API_KEY=sk-abc123def456 # rotated\nAPP_CREDENTIALS='c0ffee-c0ffee'\n")
    vals = RedactionPolicy.with_env_file(tmp_path / ".env").secret_values
    assert "postgres://u:LongPassw0rd@db/x" in vals and "sk-abc123def456" in vals and "c0ffee-c0ffee" in vals


def test_B8_redaction_is_linear_time(R):
    import time

    t0 = time.perf_counter()
    R.text("A" * 200_000 + " " + "a.b-c_" * 30_000)
    assert time.perf_counter() - t0 < 2.0  # 18.6 s: one 1 MB tool output stalled ingest for minutes


def test_B9_known_people_are_pseudonymized_in_text_and_identity_fields(R):
    t = Trace(trace_id="t", source=TraceSource(connector="c", ref="r"), actors=[TraceActor(id="Jane Doe", role="author", kind="human", handle="jdoe")],
              events=[TraceEvent(seq=0, kind="action", actor="Jane Doe", type="commit", text="thanks @jdoe and jane doe for the fix",
                                 data={"git_author": "Jane Doe", "committer": "Jane Doe"})])
    out = redact(t)
    d = json.dumps(out.model_dump(mode="json")).lower()
    assert "jane doe" not in d and "@jdoe" not in d  # the name and the @mention survived the actor pseudonym
    assert out.actors[0].handle == redact(t.model_copy(update={"actors": [TraceActor(id="JANE DOE", role="author", kind="human", handle="JDOE")]})).actors[0].handle  # B14


def test_B13_agent_handles_stay_readable(R):
    t = Trace(trace_id="t", source=TraceSource(connector="c", ref="r"), actors=[TraceActor(id="agent", role="agent", kind="agent", handle="claude-opus-5-5")])
    assert redact(t).actors[0].handle == "claude-opus-5-5"


def test_B3_B11_stored_traces_are_private_stamped_and_purgeable(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_REDACTION_SALT", "s")
    from crucible.connectors.redact import REDACTION_VERSION

    ts = TraceStore(Store(tmp_path / "store"))
    assert (tmp_path / "store").stat().st_mode & 0o077 == 0  # 0755 with 0444 blobs: any local user read the traces
    old = Trace(trace_id="t1", source=TraceSource(connector="c", ref="r"), events=[TraceEvent(seq=0, kind="action", type="x", text="leaked v1")])
    k1 = ts.put(old, redaction="old-patterns")
    k2 = ts.put(old.model_copy(update={"events": [TraceEvent(seq=0, kind="action", type="x", text="clean v2")]}))
    assert ts.stale() == [] and ts.redaction_of(k2) == REDACTION_VERSION
    assert ts.purge_superseded() == [k1]  # the leaked first version stayed addressable forever
    from crucible.core.store import StoreError

    with pytest.raises(StoreError, match="missing blob"):
        ts.store.get_bytes(k1)
    assert ts.get("t1").events[0].text == "clean v2"
