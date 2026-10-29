"""The CPU smoke scripts (``scripts/smoke/``): each one compiles and parses its arguments, and their quick parts run here
(the JSON-document service with snapshot and restore; score centering training a toy policy). The full runs are
documented in DESIGN §14 (smoke scripts) and take minutes, or Docker."""

from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SMOKE = Path(__file__).resolve().parents[2] / "scripts" / "smoke"
NAMES = ("debug_episode", "world_sims", "rl_loop", "audit_credit")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"smoke_{name}", SMOKE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", NAMES)
def test_each_smoke_script_shows_its_help(name: str) -> None:
    r = subprocess.run([sys.executable, str(SMOKE / f"{name}.py"), "--help"], capture_output=True, text=True, check=False, timeout=120)
    assert r.returncode == 0 and "usage" in r.stdout.lower(), r.stderr[-1500:]


def test_the_json_api_world_restores_its_snapshot(tmp_path: Path) -> None:
    res = asyncio.run(_load("world_sims").json_api(tmp_path))
    assert res["ok"], res


def test_score_centering_trains_the_toy_policy_against_a_mismatched_sampler() -> None:
    pytest.importorskip("torch")
    res = _load("rl_loop").score_centering(steps=120)
    assert res["ok"], res
