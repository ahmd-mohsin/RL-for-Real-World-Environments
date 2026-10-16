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


def test_every_ported_mit_or_bsd_row_names_its_notice():
    """MIT, BSD and ISC require the copyright notice with every copy (M0 audit 2026-10-16 R-11): the verifiers and
    mini-swe-agent ports shipped none, and the test above only checked files that were already named."""
    section, missing = "", []
    for n, line in enumerate((ROOT / "THIRD_PARTY.md").read_text().splitlines(), 1):
        if line.startswith("## "):
            section = line.lower()
        if "ported" not in section or not line.startswith("| ") or line.startswith("|---") or "pattern only" in line:
            continue
        licence = [c.strip() for c in line.strip().strip("|").split("|")][3]
        if not re.search(r"\b(MIT|BSD|ISC)\b", licence):
            continue
        files = re.findall(r"LICENSES/[A-Za-z0-9._-]+", licence)
        if not files or not all("Copyright" in (ROOT / f).read_text() for f in files if (ROOT / f).is_file()):
            missing.append(f"THIRD_PARTY.md:{n}")
    assert not missing, f"ported MIT/BSD/ISC rows without a notice file holding the copyright line: {missing}"
