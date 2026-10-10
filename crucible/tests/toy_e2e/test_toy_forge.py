"""M8 toy gate, end to end on CPU with 0 paid calls: world_synth forges 5 tasks and G-hindsight turns recorded ticket
cases into 6; every task passes the M9 subset (A1 A2 A3 A4-lite A5 A7 A8)."""

from __future__ import annotations

import pytest
from crucible_pack_toy import PACK_ROOT

from crucible.connectors.trace_store import TraceStore
from crucible.core.store import Store

M3 = PACK_ROOT.parent.parent / "data" / "m3" / "store"
pytestmark = pytest.mark.skipif(not any(i.startswith("toy/") for i in TraceStore(Store(M3)).ids()) if M3.exists() else True,
                                reason="needs the M3 toy ticket traces (run the M3 gate first)")


async def test_world_synth_five_tasks_accepted(tmp_path):
    from crucible_pack_toy.forge.world_synth import forge

    rep = await forge(seed=7, out=tmp_path / "f")
    assert rep["state"] == "constructed" and rep["audit"]["sound"], rep.get("reason")
    assert rep["summary"]["accepted"] == 5, rep["summary"]
    traps = [t for t, a in rep["accept"].items() if any(c["id"] == "A8" and c["detail"]["trap_signals"] for c in a["checks"])]
    assert len(traps) == 2  # the approval trap and the decoy-policy trap cite the M4 signal


async def test_hindsight_tasks_from_recorded_cases(tmp_path):
    from crucible_pack_toy.forge.hindsight import forge_hindsight

    rep = await forge_hindsight(out=tmp_path / "h")
    assert rep["pass"], rep["summary"]
    assert {k for k, v in rep["cases"].items() if v["trap"]} == {"T1003", "T1013", "T1015", "T1021"}
