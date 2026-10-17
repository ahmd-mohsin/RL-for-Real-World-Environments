"""Registry kinds are declared; an undeclared kind is refused (M0 audit 2026-10-14 R-28)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from crucible.core.registry import KINDS, Registry, RegistryError, get_registry

ROOT = Path(__file__).resolve().parents[2]


def test_a_misspelled_kind_is_refused_by_name():
    r = Registry()
    with pytest.raises(RegistryError, match="unknown registry kind 'segmentor'.*did you mean segmenter"):
        r.register("segmentor", "coding", "x:Y")  # it registered, and workflow.steps never read it
    with pytest.raises(RegistryError, match="unknown registry kind"):
        r.names("replayers")


def test_a_plugin_claim_of_an_undeclared_kind_is_refused_and_reported(monkeypatch):
    class EP:
        name, value = "p", "p:register"
        dist = type("D", (), {"name": "pack-p", "version": "1.0", "requires": []})()

        def load(self):
            return lambda reg: (reg.register("segmentor", "s", "p:S"), reg.register("segmenter", "s", "p:S"))

    monkeypatch.setattr("crucible.core.registry.entry_points", lambda group: [EP()])
    r = Registry()
    r.load_plugins()
    assert "unknown registry kind" in r.failures["p"] and r.has("segmenter", "s")


def test_every_kind_the_code_uses_is_declared_and_every_pack_loads():
    used = set()
    for base in (ROOT / "src", ROOT / "packs"):
        for p in base.rglob("*.py"):
            text = p.read_text(errors="replace")
            used |= set(re.findall(r"\b(?:reg|registry|get_registry\(\))\.(?:register|get|has|names)\(\s*\"([a-z_]+)\"", text))
    assert used and used <= set(KINDS), sorted(used - set(KINDS))
    reg = get_registry()
    reg.describe()
    assert not {k: v for k, v in reg.failures.items() if "unknown registry kind" in v}
