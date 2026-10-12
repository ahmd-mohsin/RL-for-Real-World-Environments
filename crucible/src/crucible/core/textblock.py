"""Code blocks by anchor: the one definition of "the block the expert changed", shared by offline evaluation (M5
``FileStateView``) and the live fs adapter, so a ``block_changed`` milestone reads the same on both (M5 audit: the live
adapter had no such op, and every mined coding milestone was unmeasurable on a World)."""

from __future__ import annotations

import re


def block(text: str, anchor: str) -> str | None:
    """The block starting at the first line matching ``anchor`` (a regex). Brace languages: from the anchor to the
    brace that closes the body (the opening brace may sit on the anchor line or the next line); otherwise
    (Python/YAML) the anchor plus every following blank or deeper-indented line."""
    lines = text.split("\n")
    rx = re.compile(anchor)
    for i, ln in enumerate(lines):
        if not rx.search(ln):
            continue
        nxt = next((j for j in range(i + 1, min(i + 4, len(lines))) if lines[j].strip()), None)
        if "{" in ln or (nxt is not None and lines[nxt].lstrip().startswith("{")):
            depth, started = 0, False
            for j in range(i, len(lines)):
                code = re.sub(r"//.*$|\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'", "", lines[j])
                depth += code.count("{") - code.count("}")
                started = started or "{" in code
                if started and depth <= 0:
                    return "\n".join(lines[i:j + 1]).rstrip()
            return "\n".join(lines[i:]).rstrip()
        ind = len(ln) - len(ln.lstrip())
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > ind):
            j += 1
        return "\n".join(lines[i:j]).rstrip()
    return None
