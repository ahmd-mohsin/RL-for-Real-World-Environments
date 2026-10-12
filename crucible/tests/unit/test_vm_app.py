"""``common/vm`` (QEMU/KVM, pre-booted pools, RAM snapshots): runs ``tests/live/vm_check.py`` against a Docker daemon with
``/dev/kvm``. Set ``CRUCIBLE_VM_DOCKER_HOST`` (e.g. ``ssh://user@lab``) to run it; skipped otherwise."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HOST = os.environ.get("CRUCIBLE_VM_DOCKER_HOST")


def test_vm_template_registered_and_tools():
    from crucible.core.registry import get_registry
    from crucible.world.apps.vm import IMAGE, VMApp

    assert get_registry().get("app_template", "common/vm") is VMApp and IMAGE.startswith("crucible-vm:")


@pytest.mark.skipif(not HOST, reason="needs a Docker daemon with /dev/kvm (CRUCIBLE_VM_DOCKER_HOST)")
def test_vm_pool_reset_isolation_and_ram_snapshots(tmp_path):
    import json

    script = Path(__file__).resolve().parents[1] / "live" / "vm_check.py"
    p = subprocess.run([sys.executable, str(script), "--pool", "2", "--resets", "6", "--out", str(tmp_path)], env={**os.environ, "DOCKER_HOST": HOST},
                       capture_output=True, text=True, timeout=1800, check=False)
    res = json.loads((tmp_path / "vm_check.json").read_text())
    assert res.get("pass"), (res, p.stderr[-1500:])
