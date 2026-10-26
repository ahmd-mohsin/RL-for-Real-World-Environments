"""M3 audit (2026-10-20) R-4: a lone surrogate (half of a UTF-16 pair: a cut emoji in tool output) passed ``normalize`` and
``redact`` and made ``Store.put_doc`` raise ``UnicodeEncodeError`` outside ``ingest``'s per-source ``try``, so that source
and every later one were lost (0 of 3 stored). The capture proxy writes such characters on purpose, as ``\\udXXX`` escapes.
``redact()`` now escapes them as text (``core.text.safe_json``) before the store sees the trace
(``docs/research/improvements/2026-10-20/M03-connectors.md`` R-4)."""

from __future__ import annotations

import json
from pathlib import Path

from crucible.connectors import redact as redact_mod
from crucible.connectors.base import ingest
from crucible.connectors.capture_proxy import CaptureLogConnector
from crucible.connectors.trace_store import TraceStore
from crucible.core.store import Store


def _session(d: Path, name: str, text: str, arg: str) -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": arg})}}
    row = {
        "api": "openai",
        "digest": name,
        "request": {"model": "m", "messages": [{"role": "user", "content": text}]},
        "status": 200,
        "response": {"model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [call]}}]},
    }
    (d / f"{name}.jsonl").write_text(json.dumps(row) + "\n")  # json.dumps writes a lone surrogate as the escape \ud83d


def _sessions(d: Path) -> list:
    d.mkdir()
    _session(d, "a-bad", "tool output cut inside an emoji: \ud83d", "echo \udcff")
    _session(d, "b-ok", "fix the bug", "ls")
    _session(d, "c-ok", "run the tests", "pytest")
    return list(CaptureLogConnector().discover({"dir": str(d)}))


def test_a_lone_surrogate_no_longer_aborts_the_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_REDACTION_SALT", "s")
    store = TraceStore(Store(tmp_path / "store"))
    errors: list = []
    keys = ingest(CaptureLogConnector(), _sessions(tmp_path / "cap"), store, errors=errors)
    assert len(keys) == 3 and errors == []  # was UnicodeEncodeError out of ingest: 0 of 3 stored, no source named
    ev = store.get("capture:a-bad").events
    assert [e.text for e in ev if e.type == "user_message"] == ["tool output cut inside an emoji: \\ud83d"]  # kept as escape text
    assert "echo \\udcff" in next(e.data["arguments"] for e in ev if e.type == "tool_call")  # in a tool argument too


def test_a_trace_without_surrogates_keeps_its_key(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_REDACTION_SALT", "s")
    refs = _sessions(tmp_path / "cap")[1:]
    now = TraceStore(Store(tmp_path / "now"))
    keys = ingest(CaptureLogConnector(), refs, now)
    monkeypatch.setattr(redact_mod, "safe_json", lambda x: x)  # redact() as it was
    before = TraceStore(Store(tmp_path / "before"))
    assert ingest(CaptureLogConnector(), refs, before) == keys  # the same content-addressed versions
