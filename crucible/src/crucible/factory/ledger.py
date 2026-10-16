"""The requirement ledger check (R1, 05 §5-§6): two ways, deterministic, no model.

``problems(ledger, observations, prompt=...)`` returns one ``"<code>: <detail>"`` string per defect:

| code | meaning |
|---|---|
| ``unchecked_requirement`` | no check covers the requirement (an unenforced ask) |
| ``unstated_check`` | a check names no requirement (it grades something the prompt never asked: the unfair-check defect) |
| ``unknown_requirement`` | a check names a requirement id the ledger does not have |
| ``unknown_observation`` | a check names an observation id that was not supplied |
| ``ungrounded_gate`` | a gating check has no first-class observation (only reference-derived or simulator state, or none) |
| ``span_out_of_prompt`` | a requirement's ``prompt_span`` runs outside the prompt (needs ``prompt``) |
| ``span_mismatch`` | the prompt text under the span shares too little wording with the requirement text (needs ``prompt``) |

The other direction, a sentence in the prompt that asks for something no requirement records, is ``unmapped_candidates``:
it finds candidate imperative sentences with fixed heuristics (modal verbs, constraint words, imperative mood, polite
requests) and reports those that overlap no requirement. It flags; a model or a human confirms, so it is not part of
``problems``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crucible.factory.cards import ObservationRecord, Requirement, RequirementLedger

CODES = ("unchecked_requirement", "unstated_check", "unknown_requirement", "unknown_observation", "ungrounded_gate",
         "span_out_of_prompt", "span_mismatch")
SPAN_SHARED_MIN = 0.25  # wording a sentence must share with a requirement whose span covers it
SPAN_OVERLAP_MIN = 0.5  # share of the requirement's content words that must appear in the text under its span

_split = str.split
_STOP = frozenset(_split("""a an and are as at be been by do does for from has have if in into is it its of on or that the their
them then there these they this to was were will with you your we our i he she him her his not no"""))
_TOKEN = re.compile(r"[a-z0-9_]+")


def tokens(text: str) -> set[str]:
    """Content words: lower-case alphanumerics without stop words (all words if nothing else is left)."""
    words = _TOKEN.findall(text.lower())
    content = {w for w in words if w not in _STOP and (len(w) > 1 or w.isdigit())}
    return content or set(words)


def overlap(requirement_text: str, prompt_text: str) -> float:
    """Share of the requirement's content words found in ``prompt_text``; 1.0 when the text is contained verbatim
    (case and whitespace folded)."""
    fold = lambda s: " ".join(s.lower().split())
    if fold(requirement_text) in fold(prompt_text):
        return 1.0
    want = tokens(requirement_text)
    return len(want & tokens(prompt_text)) / len(want) if want else 0.0


def problems(ledger: RequirementLedger, observations: dict[str, ObservationRecord], *, prompt: str | None = None) -> list[str]:
    out: list[str] = []
    req_ids = {r.id for r in ledger.requirements}
    covered: set[str] = set()
    for c in ledger.checks:
        if not c.requirement_ids:
            out.append(f"unstated_check: check {c.id!r} names no requirement")
        for rid in c.requirement_ids:
            if rid in req_ids:
                covered.add(rid)
            else:
                out.append(f"unknown_requirement: check {c.id!r} names requirement {rid!r}, which the ledger does not have")
        known = [observations[o] for o in c.observation_ids if o in observations]
        for oid in c.observation_ids:
            if oid not in observations:
                out.append(f"unknown_observation: check {c.id!r} names observation {oid!r}, which was not supplied")
        if c.gating and not any(o.first_class for o in known):
            out.append(f"ungrounded_gate: gating check {c.id!r} has no first-class observation "
                       f"({'none named' if not c.observation_ids else 'only second-class or unknown ones'})")
    for r in ledger.requirements:
        if r.id not in covered:
            out.append(f"unchecked_requirement: requirement {r.id!r} has no check")
        if prompt is not None:
            out.extend(_span_problems(r, prompt))
    return out


def _span_problems(r: Requirement, prompt: str) -> list[str]:
    start, end = r.prompt_span
    if end > len(prompt):
        return [f"span_out_of_prompt: requirement {r.id!r} span {r.prompt_span} runs past the prompt ({len(prompt)} characters)"]
    share = overlap(r.text, prompt[start:end])
    if share < SPAN_OVERLAP_MIN:
        return [(f"span_mismatch: requirement {r.id!r}: the prompt text under {r.prompt_span} shares {share:.0%} of the "
                 f"requirement's words (needs {SPAN_OVERLAP_MIN:.0%})")]
    return []


# --- candidate requirements in a prompt -------------------------------------------------------------------------------

_MODAL = re.compile(
    r"\b(must|shall|should|need to|needs to|has to|have to|required to|is required|are required|be sure to|make sure|"
    r"ensure|cannot|can't|may not|do not|don't|never|always|only|at least|at most|no more than|no fewer than|"
    r"exactly|without|at no point|under no circumstances)\b")
_REQUEST = re.compile(r"^(please|kindly|can you|could you|would you|i need you to|i want you to|you need to)\b")
_LEAD = re.compile(r"^(?:[-*+•]\s+|\(?\d+[.)]\s+|\(?[a-z][.)]\s+)")  # bullet or list number
_LINKERS = frozenset({"then", "also", "next", "finally", "first", "second", "third", "lastly", "afterwards", "and", "but",
                      "now", "otherwise", "additionally"})
IMPERATIVE_VERBS = frozenset(_split("""add align apply approve archive ask assign attach avoid book build calculate cancel change
check choose classify close collect compare compute confirm connect convert copy correct count create decline define
delete deploy describe detect determine disable do download draft edit email enable enter escalate evaluate execute
export extract fetch fill filter find fix flag follow forward generate get give identify implement import include
insert install invoice issue keep label link list load locate log look make mark match merge migrate move name
notify open order output pay place post prepare print process produce provide publish pull push put read reassign
record refactor refund reject remove rename reopen reorder repair replace reply report request reroute reset resolve
respond restore retrieve return review revise rewrite run save schedule search select send set show sort split start
stop store submit summarize summarise tag take test total track transfer translate trigger update upload use validate
verify view write"""))
_ABBREV = frozenset({"e.g", "i.e", "etc", "vs", "mr", "mrs", "dr", "approx", "no", "fig", "inc", "ltd", "st"})


@dataclass(frozen=True)
class Candidate:
    """A sentence that looks like it asks for something: ``text == prompt[start:end]``."""

    start: int
    end: int
    text: str
    cues: tuple[str, ...]


def sentences(prompt: str) -> list[tuple[int, int]]:
    """Sentence spans ``(start, end)`` into ``prompt`` (stripped of surrounding whitespace). A newline ends a sentence
    (list items and short lines stand alone); inside a line ``.``, ``!`` and ``?`` end one unless the word before is a
    known abbreviation, a single letter or a number with a decimal point."""
    spans: list[tuple[int, int]] = []
    line_start = 0
    for line in prompt.split("\n"):
        i, seg_start, n = 0, 0, len(line)
        while i < n:
            ch = line[i]
            if ch in ".!?":
                j = i
                while j + 1 < n and line[j + 1] in ".!?\"')]":
                    j += 1
                nxt = line[j + 1] if j + 1 < n else ""
                word = re.search(r"([A-Za-z.]+)$", line[seg_start:i])
                abbrev = bool(word) and ch == "." and (word.group(1).rstrip(".").lower() in _ABBREV or len(word.group(1).rstrip(".")) == 1)
                if (nxt == "" or nxt.isspace()) and not abbrev:
                    _push(spans, line, line_start, seg_start, j + 1)
                    seg_start = j + 1
                i = j
            i += 1
        _push(spans, line, line_start, seg_start, n)
        line_start += len(line) + 1
    return spans


def _push(spans: list[tuple[int, int]], line: str, base: int, a: int, b: int) -> None:
    chunk = line[a:b]
    stripped = chunk.strip()
    if not stripped:
        return
    lead = len(chunk) - len(chunk.lstrip())
    spans.append((base + a + lead, base + a + lead + len(stripped)))


def cues(sentence: str) -> tuple[str, ...]:
    """Why a sentence looks like a requirement: ``modal:<word>``, ``request``, ``imperative:<verb>``; empty if none."""
    s = sentence.strip()
    low = s.lower()
    found: list[str] = [f"modal:{m.group(1)}" for m in _MODAL.finditer(low)]
    body = _LEAD.sub("", low).lstrip()
    if _REQUEST.match(body):
        found.append("request")
    words = re.findall(r"[a-z']+", body)
    while words and words[0] in _LINKERS:
        words = words[1:]
    if words and words[0] in IMPERATIVE_VERBS and not s.endswith("?"):
        found.append(f"imperative:{words[0]}")
    return tuple(dict.fromkeys(found))


def candidates(prompt: str) -> list[Candidate]:
    """Every sentence of ``prompt`` that carries at least one cue, in order."""
    out = []
    for a, b in sentences(prompt):
        c = cues(prompt[a:b])
        if c:
            out.append(Candidate(a, b, prompt[a:b], c))
    return out


def _shared(requirement_text: str, sentence: str) -> float:
    """Shared content words over the smaller of the two word sets."""
    a, b = tokens(requirement_text), tokens(sentence)
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def unmapped_candidates(prompt: str, requirements: list[Requirement]) -> list[Candidate]:
    """Candidates that no requirement accounts for. A requirement accounts for a sentence when its span overlaps the
    sentence *and* they share wording (``SPAN_SHARED_MIN``: so one wide span cannot cover a whole prompt), or when they
    share ``SPAN_OVERLAP_MIN`` of their wording wherever the span is. A flag for review, not a verdict."""
    out = []
    for cand in candidates(prompt):
        for r in requirements:
            a, b = r.prompt_span
            shared = _shared(r.text, cand.text)
            if (a < cand.end and cand.start < b and shared >= SPAN_SHARED_MIN) or shared >= SPAN_OVERLAP_MIN:
                break
        else:
            out.append(cand)
    return out
