"""Unit-suite isolation from the host's Docker (M10 lab audit R-6).

A Runner start sweeps orphan containers (``lifecycle.sweep``, keyed on ``docker_available``) and the first container a process
starts installs an atexit reaper (``lifecycle._install_reaper``). A unit test must never reach a real daemon through either: on
a machine with a docker CLI (a developer's Docker Desktop, a lab box) every Runner-based test would issue ``docker ps`` and could
remove a real orphan. Both are off for every test here; a test that exercises them (``test_m10_lab_audit_r6_*``) patches them back
explicitly (``test_m10_lab_audit_r6_sweep`` sets ``docker_available`` back to True and installs a ``FakeDocker``). ``raising=False``
is kept so the fixture also works against a tree without the sweep. Tests marked ``docker`` talk to a real daemon on purpose and keep
the real reaper, sweep and re-check delay. ``RECHECK_S`` (the second look after a cancelled ``docker run``'s removal) is 10 ms here, so
a cancelled grade in an older test does not wait 2 s of wall time.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_host_docker_hygiene(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("docker"):
        return  # real-daemon tests keep the real reaper and sweep
    from crucible.world import lifecycle

    monkeypatch.setattr(lifecycle, "docker_available", lambda: False, raising=False)
    monkeypatch.setattr(lifecycle, "_install_reaper", lambda: None)
    # a removal that follows a cancelled or failed ``docker run`` looks again ``RECHECK_S`` (2 s) later: the R-5 HarborDirTask cancel tests
    # would each sleep that long in real time (the R-6 tests that care set their own)
    monkeypatch.setattr(lifecycle, "RECHECK_S", 0.01)
