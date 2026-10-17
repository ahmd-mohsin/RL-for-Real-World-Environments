"""Documents are stored as written; NFC is for identity only (M0 audit 2026-10-14 R-20)."""

from __future__ import annotations

import hashlib
import unicodedata

import pytest

from crucible.connectors.file_drop import FieldMap, rows_to_trace
from crucible.connectors.trace_store import TraceStore
from crucible.core.schema.reward import RewardJson
from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace
from crucible.core.store import Store

NFD = unicodedata.normalize("NFD", "café crème")


def _doc(text: str) -> Trace:
    return Trace.model_validate({"trace_id": "t1", "source": {"connector": "c", "ref": "r"}, "events": [{"seq": 0, "kind": "action", "type": "x", "text": text}]})


def test_nfd_text_round_trips_and_keeps_its_identity(tmp_path):
    s = Store(tmp_path / "s")
    nfd, nfc = _doc(NFD), _doc(unicodedata.normalize("NFC", NFD))
    got = s.get_doc(s.put_doc("trace", nfd), Trace)
    assert got.events[0].text == NFD and got == nfd  # came back NFC before
    assert nfd.content_hash() == nfc.content_hash()  # one identity for both spellings
    key = s.put_doc("trace", nfc)  # text already NFC: the key is still the canonical hash, as before
    assert key == "sha256:" + hashlib.sha256(nfc.canonical_json().encode()).hexdigest() == "sha256:" + nfc.content_hash()


def test_keys_that_are_one_key_under_nfc_are_refused(tmp_path):
    doc = RewardJson.model_validate({"reward": 1.0, "components": {NFD: 1.0, unicodedata.normalize("NFC", NFD): 0.0}})
    with pytest.raises(ValueError, match="one key under NFC"):
        doc.content_hash()  # the canonical form kept one of the two silently
    with pytest.raises(ValueError, match="one key under NFC"):
        Store(tmp_path / "s").put_doc("reward", doc)


def test_a_signal_on_an_nfd_event_survives_a_re_ingest(tmp_path, monkeypatch):
    """The audit's probe: the stored head was NFC-folded while the re-ingested source was NFD, so the carried signal's
    event (named by its raw text) was not found and the signal was dropped."""
    monkeypatch.setenv("CRUCIBLE_REDACTION_SALT", "s")
    f = FieldMap()
    row = {"case": "T1", "time": "2024-05-01T01:00:00Z", "type": "note", "text": NFD}
    v1 = rows_to_trace([row], f, case="T1", connector="file_drop")
    ts = TraceStore(Store(tmp_path / "store"))
    ts.put(v1)
    ts.put(v1.model_copy(update={"signals": [Signal(id="s1", type="correction", polarity=-1, grades=[0], source_event=0)]}))
    v2 = rows_to_trace([{"case": "T1", "time": "2024-05-01T00:00:00Z", "type": "note", "text": "opened"}, row], f, case="T1", connector="file_drop")
    ts.put(v2)
    got = ts.get("file_drop:T1")
    assert [s.grades for s in got.signals] == [[1]] and "signals_dropped" not in got.outcome.details
