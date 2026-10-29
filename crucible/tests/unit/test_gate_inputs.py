"""M4 audit (2026-10-27) R-4: an M4 gate report names the trace content and the code its numbers came from, beside the
labels' hash. Reports named only the labels file, so one edited toy reply moved type accuracy 1.0 -> 0.988 under the same
hash, and the coding gate's next rebuild scores whatever changed upstream
(``docs/research/improvements/2026-10-27/M04-signals.md`` R-4)."""

from __future__ import annotations

import pytest

from crucible.connectors.redact import REDACTION_VERSION, pseudonym
from crucible.signals.gate import inputs


def test_toy_reports_name_their_traces_and_code(monkeypatch):
    from crucible_pack_toy.m4 import gate

    a, b = gate.run(), gate.run()
    assert a["inputs"] == b["inputs"] and len(a["inputs"]["traces"]) == 12  # the 12 labelled test tickets
    assert a["code"]["git_sha"] and a["code"]["source_sha256"] and a["code"]["pack_sha256"]
    real = gate.traces

    def edited():
        out = real()
        t = out["toy/support_ticket:T1013"]
        ev = [e.model_copy(update={"text": (e.text or "") + " (edited)"}) if i == 1 else e for i, e in enumerate(t.events)]
        out[t.trace_id] = t.model_copy(update={"events": ev})
        return out

    monkeypatch.setattr(gate, "traces", edited)
    c = gate.run()
    moved = {k for k in a["inputs"]["traces"] if a["inputs"]["traces"][k] != c["inputs"]["traces"][k]}
    assert moved == {"toy/support_ticket:T1013"} and c["inputs"]["traces_sha256"] != a["inputs"]["traces_sha256"]
    assert c["labels_sha256"] == a["labels_sha256"]  # the labels' hash alone could not tell the two runs apart


def test_the_digest_ignores_carried_signals():
    from crucible_pack_toy.m4.gate import traces

    from crucible.core.schema.signal import Signal

    raw = list(traces().values())[:3]
    carried = [t.model_copy(update={"signals": [Signal(id="x", type="approval", polarity=1, grades=[0], source_event=1)]}) for t in raw]
    assert inputs(raw) == inputs(carried)  # a store carries old signals into a version: they are output, not input


def test_coding_reports_carry_the_redaction_salt_id(monkeypatch):
    pytest.importorskip("crucible_pack_coding")
    from crucible_pack_coding.m4 import gate

    class NoStore:
        def __init__(self, *a, **kw):
            pass

        def ids(self):
            return []

    monkeypatch.setattr(gate, "Store", NoStore)
    monkeypatch.setattr(gate, "TraceStore", NoStore)
    monkeypatch.setenv("CRUCIBLE_REDACTION_SALT", "tenant-a")
    rep = gate.run(rebuild=False)
    assert rep["redaction"] == {"version": REDACTION_VERSION, "salt_id": pseudonym("salt", "id", "tenant-a")}
    assert rep["inputs"]["traces"] == {} and rep["code"]["pack_sha256"]
