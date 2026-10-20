"""M10 exit gate R3 (M10 lab audit R-1): mini-swe-agent 2.4.6, installed by Harbor, runs task02 live through ``RunSpec``
and the separate-verifier patch seam, its model traffic through the capture proxy; ``crucible_pack_coding.runtime.r3``
writes ``docs/modules/reports/m10_r3.json``.

Needs the lab RTX 5090 over ``DOCKER_HOST=ssh://`` (with nftables egress control for the agent phase), a paid upstream
(``OPENROUTER_API_KEY``), the forwarded model priced in crucible.toml ``[prices]``, and ``CRUCIBLE_R3_PROXY_HOST`` (this
machine as the lab container reaches it). Skipped otherwise; about $3-15 and several hours per run.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.docker, pytest.mark.gpu, pytest.mark.llm, pytest.mark.slow]


async def test_m10_gate_r3_live_mini_swe_agent_on_task02():
    if not os.environ.get("CRUCIBLE_R3_PROXY_HOST"):
        pytest.skip("CRUCIBLE_R3_PROXY_HOST is not set (the lab container must reach this machine's capture proxy)")
    pytest.importorskip("harbor")
    pytest.importorskip("crucible_pack_coding")
    from crucible_pack_coding.runtime import r3

    problems = r3.preflight()
    if problems:
        pytest.skip("; ".join(problems))
    out = await r3.main()
    assert r3.REPORT.exists()
    assert out["pass"], {k: out.get(k) for k in ("unmet_acceptance", "pass_relaxed", "profile_equal", "record", "agent", "reconcile", "infra", "gates",
                                                 "recorded_gates", "ops_failed", "recorded_ops_failed")}
