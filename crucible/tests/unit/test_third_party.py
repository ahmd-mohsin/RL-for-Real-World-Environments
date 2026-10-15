"""THIRD_PARTY.md and LICENSES/ agree (M0 audit 2026-10-14 R-9): a licence or notice file nobody names, or a named one that
is missing, is how agentenv's NOTICE went unshipped."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_every_licence_file_is_named_and_every_named_one_exists():
    named = set(re.findall(r"LICENSES/[A-Za-z0-9._-]+", (ROOT / "THIRD_PARTY.md").read_text()))
    shipped = {f"LICENSES/{p.name}" for p in (ROOT / "LICENSES").iterdir() if p.is_file()}
    assert not named - shipped, f"named in THIRD_PARTY.md but missing: {sorted(named - shipped)}"
    assert not shipped - named, f"in LICENSES/ but never named in THIRD_PARTY.md: {sorted(shipped - named)}"


def test_the_agentenv_notice_is_reproduced():
    text = (ROOT / "LICENSES" / "NOTICE-agentenv.txt").read_text()
    assert "Copyright 2026 Scale AI" in text and "software developed at Scale AI" in text
