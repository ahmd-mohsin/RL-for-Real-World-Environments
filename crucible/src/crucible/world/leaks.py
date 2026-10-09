"""Leak scrubbing for anything a policy reads: instructions, simulated colleagues' replies, CI output.

Ported from Hugging Face Repo2RLEnv (Apache-2.0; ``pipelines/cve_patches.py`` ``_strip_fix_leaks`` and
``pipelines/pr_runtime.py`` ``_LEAK_PATTERNS``), whose agents earned perfect scores by following a PR
link, diffing ``origin/main`` and downloading the fixed wheel. The lesson both they and Preference Model's
Karotte draw: the environment enforces, the prompt never asks. Scrubbing is one layer; network and
git-history isolation (``world/guard/``) are the others.

``scrub_text(text, mode)``:
- ``"strict"`` (instructions): also drops whole sections and lines that point at fixes
  ("References", "Patches", "fixed in 1.2.3", "upgrade to …", "see/apply …");
- ``"inline"`` (default; replies, CI output): removes pointer tokens but keeps the prose.
"""

from __future__ import annotations

import re

# --- Repo2RLEnv cve_patches.py (verbatim patterns) ---
_LEAK_SECTION_RE = re.compile(
    r"(?ims)^[ \t]*#{1,6}[ \t]*"
    r"(workarounds?|remediation|mitigations?|references?|fix(?:es|ed)?|patch(?:es|ed)?|"
    r"solutions?|credits?|acknowledge?ments?|resources?|links?|see also|further reading)\b"
    r".*?(?=^[ \t]*#{1,6}[ \t]|\Z)"
)
_LEAK_LINE_RES = [
    re.compile(r"(?im)^.*\b(?:apply|see|refer to|backport|cherry[- ]pick)\b.*$"),
    re.compile(r"(?im)^.*\b(?:fixed|resolved|patched|addressed)\b.*\bin\b.*\bv?\d+\.\d+.*$"),
    re.compile(r"(?im)^.*\bupgrade\b.*\b(?:to|version)\b.*$"),
]
_LEAK_TOKEN_RES = [
    re.compile(r"https?://\S*github\.com/\S+", re.I),
    re.compile(r"https?://\S+", re.I),
    re.compile(r"\b(?:CVE-\d{4}-\d{4,7}|GHSA-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4})\b", re.I),
    re.compile(r"\b(?:PR|pull request|pull|commit|issue)\s*#?\d+\b", re.I),
    re.compile(r"(?<![\w/])#\d{2,}\b"),
]
# --- Repo2RLEnv pr_runtime.py (verbatim patterns) ---
_PR_LEAK_PATTERNS = (
    re.compile(r"\b(?:cherry[-\s]?pick(?:ed|ing|s)?|back[-\s]?port(?:ed|ing|s)?|commit)\b[^\n]*?\b[0-9a-f]{7,40}\b", re.IGNORECASE),
    re.compile(r"\b[0-9a-f]{40}\b"),
    re.compile(r"\[[^\]]*\]\(https?://github\.com/[^\s)]*?/(?:pull|issues|commit)/[^\s)]*\)"),
    re.compile(r"https?://(?:\w+\.)?github\.com/[^\s)]*?/(?:pull|issues|commit)/\S+"),
    re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|see|refs?|ref|follow[-\s]?up(?:\s+to)?)\b[:\s]*(?:[\w.-]+/[\w.-]+)?#\d+", re.IGNORECASE),
    re.compile(r"\[#\d+\]\([^)]*\)"),
    re.compile(r"\s*\(#\d+\)"),
    re.compile(r"\b[a-zA-Z][\w.-]*#\d+\b"),
)
# --- Crucible additions ---
_SHORT_SHA_CONTEXT = re.compile(r"\b(?:sha|hash|rev(?:ision)?|in|at|from)\s+[0-9a-f]{7,12}\b", re.I)
_CODE_HOSTS = re.compile(r"\b(?:github\.com|gitlab\.com|bitbucket\.org|pypi\.org|files\.pythonhosted\.org|raw\.githubusercontent\.com)\S*", re.I)
_PKG_PIN = re.compile(r"\b(?:pip|uv pip|conda|npm|cargo)\s+(?:install|download|add)\s+\S+==?\s*v?\d+(?:\.\d+)+\S*", re.I)
# version pointers to a fix ("fixed in 1.14.1", "upgrade to v2.3 or later", "released in 6.12.0")
_VERSION_POINTER = re.compile(r"\b(?:fixed|resolved|patched|addressed|released|landed|shipped)\s+in\s+(?:version\s+)?v?\d+(?:\.\d+)+\w*"
                              r"|\bupgrad(?:e|ing)\s+to\s+(?:version\s+)?v?\d+(?:\.\d+)+\w*(?:\s+or\s+(?:later|newer|above))?", re.I)
_DEBRIS = [(re.compile(r"\(\s*\)|\[\s*\]"), ""), (re.compile(r"\b(?:See|Refer to)\s+for details\.?", re.I), ""),
           (re.compile(r"(?m)^[\s.,;:/]+(?=\S)"), ""), (re.compile(r"(?m)^[\s.,;:/]+$"), ""), (re.compile(r";\s*(?=[.;]|$)"), ""),
           (re.compile(r"(?m)\s+/\.?$"), "")]


def scrub_text(text: str, mode: str = "inline") -> str:
    if not text:
        return text
    out = text
    if mode == "strict":
        out = _LEAK_SECTION_RE.sub("", out)
        for rx in _LEAK_LINE_RES:
            out = rx.sub("", out)
    for rx in (*_PR_LEAK_PATTERNS, *_LEAK_TOKEN_RES, _SHORT_SHA_CONTEXT, _CODE_HOSTS, _PKG_PIN, _VERSION_POINTER):
        out = rx.sub("", out)
    for _ in range(2):  # removing tokens leaves debris; normalize spacing, then clean, twice
        out = re.sub(r"[ \t]{2,}", " ", out)
        out = re.sub(r" +([.,;:!?])", r"\1", out)
        for rx, rep in _DEBRIS:
            out = rx.sub(rep, out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def leak_tokens(text: str) -> list[str]:
    """What ``scrub_text`` would remove (for audits: an instruction with leak tokens fails the leak gate)."""
    found: list[str] = []
    for rx in (*_PR_LEAK_PATTERNS, *_LEAK_TOKEN_RES, _CODE_HOSTS, _PKG_PIN, _VERSION_POINTER):
        found += [m.group(0).strip() for m in rx.finditer(text)]
    return sorted(set(found))
