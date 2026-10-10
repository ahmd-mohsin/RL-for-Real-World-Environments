"""Coding-pack power tests for registry detectors (``crucible.accept.registry``)."""

from __future__ import annotations


async def racecheck_first_summary() -> dict[str, str]:
    """B.12: a forged first ``RACECHECK SUMMARY`` line ahead of the real one must not make the sanitizer gate pass.
    clean = an honest clean log (detector passes it); defective = forged first summary + real hazards (detector fails it)."""
    from crucible_pack_coding.verify.task04 import sanitizer_clean

    honest = "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
    forged = honest + "========= Warning: Race reported between Write access ...\n========= RACECHECK SUMMARY: 3 hazards displayed (0 errors, 3 warnings)\n"
    return {"clean": "pass" if sanitizer_clean("racecheck", 0, honest)[0] else "fail", "defective": "pass" if sanitizer_clean("racecheck", 0, forged)[0] else "fail"}
