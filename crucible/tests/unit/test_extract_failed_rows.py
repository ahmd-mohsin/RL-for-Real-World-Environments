"""M4 audit (2026-10-27) R-2: a trace whose ``extract()`` raised has no M4 verdict and yields no SFT, KTO or DPO row.
``crucible label`` reported it (A15) but still made rows from it, with no signals, so nothing was graded wrong: a corrected
or rejected step became a positive target (``docs/research/improvements/2026-10-27/M04-signals.md`` R-2)."""

from __future__ import annotations

import asyncio
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from crucible.signals import extract as extract_mod
from crucible.studio.label import label

FAILS = "toy/support_ticket:T1005"


def _rows(out) -> dict[str, list[dict]]:
    return {k: [json.loads(x) for x in (out / f"{k}.jsonl").read_text().splitlines() if x.strip()] for k in ("sft", "dpo", "kto")}


def test_a_trace_whose_extraction_raised_gives_no_rows(tmp_path, monkeypatch):
    asyncio.run(label(out=tmp_path / "base", pack="toy"))
    base = _rows(tmp_path / "base")
    assert any(r["episode"] == FAILS for r in base["sft"] + base["kto"])
    real = extract_mod.extract

    def flaky(trace, *a, **kw):
        if trace.trace_id == FAILS:
            raise RuntimeError("mapper bug")
        return real(trace, *a, **kw)

    monkeypatch.setattr(extract_mod, "extract", flaky)
    s = asyncio.run(label(out=tmp_path / "failed", pack="toy"))
    got = _rows(tmp_path / "failed")
    assert [f["trace_id"] for f in s["extract_failed"]] == [FAILS] and s["skipped"]["extract_failed"] == 1
    assert not [r for k in got for r in got[k] if r["episode"] == FAILS]  # was 2 SFT and 2 KTO rows, unextracted
    assert got == {k: [r for r in base[k] if r["episode"] != FAILS] for k in base}  # the other traces' rows are unchanged
    traces = [json.loads(x)["trace_id"] for x in (tmp_path / "failed" / "traces.jsonl").read_text().splitlines()]
    assert FAILS in traces  # kept as evidence
