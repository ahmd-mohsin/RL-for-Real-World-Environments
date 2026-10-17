"""Failure-kind names under fullmatch; an unknown kind read warns once (M0 audit 2026-10-14 R-27)."""

from __future__ import annotations

import logging

import pytest

from crucible.core.schema import run as run_mod
from crucible.core.schema.run import Failure, register_failure_kind


@pytest.mark.parametrize("bad", ["pack:kind\n", "9probe:_x", "pack:_x", "Pack:kind", "pack", "crucible:mine", "pack:kind:extra", 3])
def test_malformed_pack_kinds_are_refused(bad):
    with pytest.raises(ValueError):
        register_failure_kind(bad, blame="infra")


def test_a_well_formed_pack_kind_registers():
    register_failure_kind("probe_pack:env_flake", blame="infra")
    assert Failure(kind="probe_pack:env_flake", stage="agent").who == "infra"


def test_an_unknown_kind_read_warns_once(caplog):
    run_mod._WARNED.discard("probe_pack:misspeled")
    with caplog.at_level(logging.WARNING, logger=run_mod.__name__):
        f = Failure(kind="probe_pack:misspeled", stage="agent")
        assert f.who == "unknown" and f.masks
        assert f.who == "unknown"
    assert [r.getMessage() for r in caplog.records].count(caplog.records[0].getMessage()) == 1 and "misspeled" in caplog.records[0].getMessage()
