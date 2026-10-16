"""``.env`` files read with python-dotenv's grammar (M0 audit 2026-10-14 R-29)."""

from __future__ import annotations

import io

import pytest

from crucible.core.config import _dotenv, resolve_secret

# The audit's fixture (probe p_v3_config e), with what the old parser returned for each key
FIXTURE = """# keys
A_KEY=sk-or-v1-abc123  # personal key
B_KEY="quoted value" # note
C_KEY='single'
D_KEY="has \\"escaped\\" quote"
export E_KEY=plain
F_KEY=
G_KEY=#not-a-comment
H_KEY="multi
line"
"""
WANT = {"A_KEY": "sk-or-v1-abc123", "B_KEY": "quoted value", "C_KEY": "single", "D_KEY": 'has "escaped" quote', "E_KEY": "plain",
        "F_KEY": "", "G_KEY": "#not-a-comment", "H_KEY": "multi\nline"}


def test_dot_env_parses_like_python_dotenv(tmp_path):
    (tmp_path / ".env").write_text(FIXTURE)
    assert _dotenv(tmp_path) == WANT  # A_KEY kept "  # personal key" and B_KEY a quote and the note
    dotenv = pytest.importorskip("dotenv")
    assert dict(dotenv.dotenv_values(stream=io.StringIO(FIXTURE), interpolate=False)) == WANT  # the same as the library


def test_a_secret_from_dot_env_has_no_trailing_comment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(FIXTURE)
    monkeypatch.delenv("A_KEY", raising=False)
    assert resolve_secret("env:A_KEY", search_from=tmp_path) == "sk-or-v1-abc123"
