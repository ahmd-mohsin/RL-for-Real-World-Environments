"""Patches as bytes (M7 R3). A collected patch is git's output, which may hold any bytes: CRLF line endings, a Latin-1
source file, a binary diff. It travels as ``str`` (the harness and grader interfaces), decoded with ``surrogateescape``,
so every byte that is not UTF-8 maps to a lone surrogate and back to the same byte; a UTF-8 patch is unchanged.

Every read and write of a patch file goes through ``read_patch`` / ``write_patch``: ``Path.read_text`` turns CRLF into
LF (universal newlines) and ``errors="replace"`` turns other bytes into U+FFFD, so a fix to a CRLF or Latin-1 file scored
0 when it was re-graded from its stored copy, and an oracle of such a file failed its own control. A patch string that
still holds surrogates raises in ``Path.write_text``: a site that was missed is a crash in tests, never a silent 0."""

from __future__ import annotations

from pathlib import Path


def decode(b: bytes) -> str:
    return b.decode("utf-8", "surrogateescape")


def encode(s: str) -> bytes:
    return s.encode("utf-8", "surrogateescape")


def read_patch(path: str | Path) -> str:
    return decode(Path(path).read_bytes())


def write_patch(path: str | Path, s: str) -> None:
    Path(path).write_bytes(encode(s))
