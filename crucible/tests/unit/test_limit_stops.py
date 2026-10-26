"""A reply cut at our --max-tokens limit is a limit stop, never a skill failure (2026-10-25: 3 of the 4 student failures
of km-cuda12-cudahandbook-b62f9ad-4 ended on max_output_tokens at 4096 and would have been labelled skill gaps)."""

from __future__ import annotations


def test_max_output_tokens_is_a_limit_stop_in_the_audit_and_the_difficulty_tracker():
    from crucible.factory import traceaudit
    from crucible.forge import difficulty

    assert "max_output_tokens" in traceaudit.BUDGET_STOPS
    assert "max_output_tokens" in difficulty.BUDGET_STOPS
    from crucible.actors import policy

    assert "max_output_tokens" in policy.POLICY_STOP_ORDER  # the label the policy really emits
