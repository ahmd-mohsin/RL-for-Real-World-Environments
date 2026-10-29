"""M4 audit (2026-10-27) R-3: a PR check run that timed out or never started says nothing about the code, so it grades its
commit with polarity 0 (``meta.inconclusive``), not -1. The same pack already masks GitHub's runner time limit as infra on
the CI-log path (P16); on the PR path it was a failure, and a later hotfix "confirmed" it in the gate's reliability row
(``docs/research/improvements/2026-10-27/M04-signals.md`` R-3)."""

from __future__ import annotations

from crucible.signals.reliability import verdict
from tests.unit.test_m4_audit_packs import AUTHOR, CI, coding, ev, pr_trace


def _run(seq: int, name: str, conclusion: str):
    return ev(seq, "ci_run", kind="check", actor="ci", name=name, sha="a", conclusion=conclusion, app="gha")


def test_a_timeout_or_startup_failure_is_inconclusive():
    t = pr_trace(
        [
            ev(0, "commit", sha="a"),
            _run(1, "build", "timed_out"),
            _run(2, "lint", "STARTUP_FAILURE"),
            _run(3, "test", "failure"),
            _run(4, "docs", "success"),
            ev(5, "post_merge_commit", fixlike=True, touches_seq=[0], days_after_merge=3.0),
        ],
        [AUTHOR, CI],
    )
    sigs = coding(t)
    checks = [s for s in sigs if s.type == "check_result"]
    assert [s.polarity for s in checks] == [0, 0, -1, 1]  # was [-1, -1, -1, 1]
    assert [s.meta.get("inconclusive") for s in checks] == ["timed_out", "startup_failure", None, None] and checks[0].meta["reason"]
    t = t.model_copy(update={"signals": sigs})
    assert verdict(t, checks[0]) is None and verdict(t, checks[2]) is True  # the hotfix confirmed the timeout's -1


def test_a_timeout_then_a_pass_is_a_pass_not_flaky():
    [timeout, ok] = coding(pr_trace([ev(0, "commit", sha="a"), _run(1, "build", "timed_out"), _run(2, "build", "success")], [AUTHOR, CI]))
    assert (timeout.polarity, ok.polarity, ok.meta["flaky"]) == (0, 1, False)  # was flaky: both 0


# ---- M4 audit R-5: one CI-check identity in ``verdict``; a later pass never contradicts a pass ------------------------------
def _graded(events):
    t = pr_trace(events, [AUTHOR, CI], label="good")
    return t.model_copy(update={"signals": coding(t)})


def test_a_pass_then_a_pass_confirms_both():
    t = _graded([ev(0, "commit", sha="a"), _run(1, "build", "success"), _run(2, "build", "success")])
    assert [verdict(t, s) for s in t.signals] == [True, True]  # was [False, True]: the second pass "contradicted" the first


def test_checks_of_two_apps_are_two_checks():
    azure = ev(1, "ci_run", kind="check", actor="ci", name="build", sha="a", conclusion="failure", app="azure")
    t = _graded([ev(0, "commit", sha="a"), azure, _run(2, "build", "success")])
    assert [s.meta["app"] for s in t.signals] == ["azure", "gha"]
    assert [verdict(t, s) for s in t.signals] == [None, True]  # was [False, True]: Actions' pass "contradicted" Azure's failure
    t = _graded([ev(0, "commit", sha="a"), _run(1, "build", "failure"), _run(2, "build", "success")])
    assert [s.polarity for s in t.signals] == [0, 0]  # one app, both outcomes: flaky, as before
