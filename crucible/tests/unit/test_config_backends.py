"""The unread ``[backends]`` table is gone: a file setting it fails (M0 audit 2026-10-14 R-31)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crucible.core.config import Config


def test_a_backends_table_is_refused_not_ignored():
    """``[backends.harbor] docker_host`` was accepted and read by nothing (gpu_hosts holds docker hosts)."""
    with pytest.raises(ValidationError, match="backends"):
        Config.model_validate({"backends": {"harbor": {"docker_host": "ssh://x"}}})
    assert Config.model_validate({"gpu_hosts": {"lab": {"ssh": "me@lab", "docker_host": "ssh://me@lab"}}}).gpu_hosts["lab"].docker_host
