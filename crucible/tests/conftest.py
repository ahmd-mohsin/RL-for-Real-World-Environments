"""Suite-wide gating: tests that need an external capability skip, never fail, where it is absent (the lab campaign runs
the suite inside a vLLM container without a Docker daemon)."""

from __future__ import annotations

import functools
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# Optional frameworks: a test module that imports one at top level is not collected where it is not installed (CI runs
# without the agentenv checkout). Detected from the source, so a new test needs no list update.
_OPTIONAL = ("agent_env", "agentenv_protocol", "verifiers", "openenv", "nemo_gym", "torch", "transformers")
_TOP_IMPORT = re.compile(r"^(?:from|import)\s+(" + "|".join(_OPTIONAL) + r")\b", re.MULTILINE)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if collection_path.suffix != ".py" or not collection_path.name.startswith("test_"):
        return None
    try:
        text = collection_path.read_text()
    except OSError:
        return None
    missing = {m for m in _TOP_IMPORT.findall(text) if importlib.util.find_spec(m) is None}
    return True if missing else None


@functools.cache
def _docker_ok() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(["docker", "info"], capture_output=True, check=False, timeout=30).returncode == 0


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("docker") and not _docker_ok():
        pytest.skip("needs a reachable Docker daemon")


@pytest.fixture(scope="session", autouse=True)
def _toy_traces_ingested() -> None:
    """Several forge tests mine signals from the toy traces in ``data/m3/store``. Build that store from the committed toy
    logs when it is absent, so the suite never depends on a manual ``crucible ingest toy`` (CI starts from a clean
    checkout). Idempotent: the ingest is content-addressed."""
    try:
        from crucible_pack_toy.forge.world_synth import M3

        from crucible.studio.label import ingest
        from crucible.studio.pack import find_pack, load_manifest
    except ImportError:  # the toy pack is not installed: the tests that need it are not collected either
        return
    if any(M3.glob("blobs/*/*")):
        return
    pack = find_pack("toy")
    ingest(pack, load_manifest(pack), M3)
