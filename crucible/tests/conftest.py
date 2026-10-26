"""Suite-wide gating: tests that need an external capability skip, never fail, where it is absent (the lab campaign runs
the suite inside a vLLM container without a Docker daemon)."""

from __future__ import annotations

import functools
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

# Optional frameworks: a test module that imports one at top level is not collected where it is not installed (CI runs
# without the agentenv checkout). Detected from the module's own top-level import statements (parsed, so a code string
# that a test runs in a framework's Python does not count: M11 audit, a regex over the text silently dropped
# test_interop_gate.py and test_m11_audit.py from every full run).
_OPTIONAL = ("agent_env", "agentenv_protocol", "verifiers", "openenv", "nemo_gym", "torch", "transformers")


def _top_imports(text: str) -> set[str]:
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            out.add(node.module.split(".")[0])
    return out


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if collection_path.suffix != ".py" or not collection_path.name.startswith("test_"):
        return None
    try:
        text = collection_path.read_text()
    except OSError:
        return None
    missing = {m for m in _top_imports(text) if m in _OPTIONAL and importlib.util.find_spec(m) is None}
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


@pytest.fixture(autouse=True)
def _host_reach_isolated():
    """``world.hostreach`` keeps one ``HostReach`` per remote host for the process: every test starts with none, so a host a
    test left down never pauses another. A fake daemon that also fails its probe would pause an episode for the real budget
    (30 min): hosts made without a rule get a short one here."""
    from dataclasses import replace

    from crucible.world import hostreach

    hostreach.reset()
    saved = hostreach.DEFAULT_RULE
    hostreach.DEFAULT_RULE = replace(saved, pause_s=1.0, backoff_s=0.05, backoff_cap_s=0.2, probe_timeout_s=2.0)
    try:
        yield
    finally:
        hostreach.DEFAULT_RULE = saved
        hostreach.reset()
