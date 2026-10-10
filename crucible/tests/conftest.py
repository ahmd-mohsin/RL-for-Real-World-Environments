"""Suite-wide gating: tests that need an external capability skip, never fail, where it is absent (the lab campaign runs
the suite inside a vLLM container without a Docker daemon)."""

from __future__ import annotations

import functools
import shutil
import subprocess

import pytest


@functools.cache
def _docker_ok() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(["docker", "info"], capture_output=True, check=False, timeout=30).returncode == 0


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("docker") and not _docker_ok():
        pytest.skip("needs a reachable Docker daemon")
