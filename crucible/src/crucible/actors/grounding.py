"""Citation-based grounding checks for simulated actors (the anti-fabrication gate of M2).

A grounded reply is a list of sentences; every *factual* sentence must cite at least one record
(``[R:<record id>]``) that (1) exists, (2) is visible at the world's current time, and (3) supports it:
- every number / amount / date / identifier-like token in the sentence appears verbatim in a cited record, **as a whole
  token** ("10" is not supported by "100", and a date is one token, so "2024-01-05" is not supported by "2024-05-01");
- at least ``min_overlap`` of the sentence's content words appear in the cited records;
- its **polarity** matches the record sentence it draws on: a claim negated where that sentence is not (or the reverse)
  is unsupported ("was not approved" against "was approved");
- it cites at most ``max_citations`` records, each of which contributes a fact or a content word: pooling many records
  lets almost any sentence pass the overlap test.
Non-factual sentences (questions, greetings, acknowledgements, "no record") need no citation. A greeting prefix does not
make a sentence non-factual: "Sure, the refund was 900 USD." is checked on what follows "Sure,".
(M2 audit 2026-10-12: each rule above closes a fabrication that passed the gate, reproduced in ``test_m2_audit``.)

The check is deterministic and cheap; an optional LLM entailment judge (``fidelity.py``) audits samples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from crucible.actors.records import RecordStore, content_tokens

CITE = re.compile(r"\[R:([^\]\s]+)\]")
_SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
# numbers (incl. 1,234.50 / 12% / 3.5x), dates, and identifier-like tokens (INV-1042, PR#6012, sha prefixes)
# order matters: a date or an identifier is one fact, never re-read as its digit runs
_FACT_TOKEN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b[A-Z]{2,}[-_#]?\d+\b|\b[0-9a-f]{7,40}\b|\b\d[\d,]*(?:\.\d+)?%?")
_PHATIC_WORDS = r"(?:hi|hello|hey|thanks|thank you|ok(?:ay)?|sure|got it|understood|noted|sounds good|great|perfect|no problem|you're welcome|bye|goodbye)"
_PHATIC = re.compile(rf"^(?:{_PHATIC_WORDS}[\s,!.]*)+$", re.IGNORECASE)  # nothing but greetings and acknowledgements
_PHATIC_LEAD = re.compile(rf"^(?:{_PHATIC_WORDS}\b[\s,!.]*)+", re.IGNORECASE)
_NEG = re.compile(r"\b(?:not|no|never|none|neither|nor|cannot|without)\b|n't\b", re.IGNORECASE)
NO_RECORD = "I have no record of that."


@dataclass
class SentenceCheck:
    text: str
    citations: list[str]
    factual: bool
    supported: bool
    reason: str = ""


@dataclass
class GroundingReport:
    sentences: list[SentenceCheck] = field(default_factory=list)

    @property
    def factual(self) -> list[SentenceCheck]:
        return [s for s in self.sentences if s.factual]

    @property
    def unsupported(self) -> list[SentenceCheck]:
        return [s for s in self.sentences if s.factual and not s.supported]

    @property
    def ok(self) -> bool:
        return not self.unsupported

    @property
    def citations(self) -> list[str]:
        return sorted({c for s in self.sentences for c in s.citations})

    def clean_text(self, *, drop_unsupported: bool = True) -> str:
        keep = [s for s in self.sentences if not (drop_unsupported and s.factual and not s.supported)]
        return " ".join(CITE.sub("", s.text).strip() for s in keep).strip()


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT.split(text.strip()) if p and p.strip()]
    # glue a trailing citation-only fragment back onto the previous sentence
    out: list[str] = []
    for p in parts:
        if out and CITE.fullmatch(p.replace(" ", "")) is not None:
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def is_factual(sentence: str) -> bool:
    bare = CITE.sub("", sentence).strip()
    if not bare or bare.endswith("?") or _PHATIC.match(bare) or bare.rstrip(".") == NO_RECORD.rstrip("."):
        return False
    if _LIST_MARKER.fullmatch(bare):  # "2." from a numbered list split into sentences: no claim
        return False
    rest = _PHATIC_LEAD.sub("", bare).strip()  # "Sure, the refund was 900 USD." is a claim about the refund
    return len(content_tokens(rest)) >= 2 or bool(_FACT_TOKEN.search(rest))


_LIST_MARKER = re.compile(r"[(\[]?(?:\d{1,2}|[a-z]|[ivx]{1,4})[.)\]]?\s*[.:]?", re.IGNORECASE)


def _facts(text: str) -> set[str]:
    return set(_FACT_TOKEN.findall(text))


def _negated(text: str) -> bool:
    return bool(_NEG.search(text))


def _closest_sentence(claim_tokens: set[str], records: list) -> str:
    best, score = "", -1.0
    for r in records:
        for rs in split_sentences(r.text) or [r.text]:
            t = set(content_tokens(rs))
            sc = len(claim_tokens & t) / max(len(claim_tokens), 1)
            if sc > score:
                best, score = rs, sc
    return best


def _words(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def _polarity_conflict(bare: str, claim_tokens: set[str], records: list) -> str | None:
    """The record sentence whose negation the claim flips, or None. A claim copied verbatim from a cited record cannot
    flip it; otherwise any record sentence within 0.15 of the best overlap that shares the claim's polarity clears it
    (code blocks and lists split records unevenly: the M2 gate replay found two verbatim quotes judged flipped)."""
    w = _words(bare)
    if w and any(w in _words(r.text) for r in records):
        return None
    scored = []
    for r in records:
        for rs in split_sentences(r.text) or [r.text]:
            t = set(content_tokens(rs))
            scored.append((len(claim_tokens & t) / max(len(claim_tokens), 1), rs))
    if not scored:
        return None
    top = max(s for s, _ in scored)
    near = [rs for s, rs in scored if s >= top - 0.15]
    neg = _negated(bare)
    return None if any(_negated(rs) == neg for rs in near) else near[0]


def check(text: str, store: RecordStore, *, now: str | None = None, allowed_sources: set[str] | None = None, min_overlap: float = 0.5,
          max_citations: int = 2, allowed_ids: set[str] | None = None) -> GroundingReport:
    rep = GroundingReport()
    for sent in split_sentences(text):
        cites = CITE.findall(sent)
        factual = is_factual(sent)
        if not factual:
            rep.sentences.append(SentenceCheck(sent, cites, False, True))
            continue
        if not cites:
            rep.sentences.append(SentenceCheck(sent, cites, True, False, "no citation"))
            continue
        recs = []
        bad = []
        for c in cites:
            r = store.get(c, now=now)
            if r is None or (allowed_sources and r.source not in allowed_sources) or (allowed_ids is not None and r.id not in allowed_ids):
                bad.append(c)
            else:
                recs.append(r)
        if bad:
            rep.sentences.append(SentenceCheck(sent, cites, True, False, f"invalid or future citation(s): {bad}"))
            continue
        if len(set(cites)) > max_citations:
            rep.sentences.append(SentenceCheck(sent, cites, True, False, f"{len(set(cites))} citations (at most {max_citations}): pooled records support anything"))
            continue
        bare = _PHATIC_LEAD.sub("", CITE.sub("", sent)).strip()
        corpus = " ".join(r.text for r in recs)
        missing_facts = sorted(_facts(bare) - _facts(corpus))
        if missing_facts:
            rep.sentences.append(SentenceCheck(sent, cites, True, False, f"facts not in cited records: {missing_facts}"))
            continue
        st = set(content_tokens(bare))
        idle = [r.id for r in recs if not (st & set(content_tokens(r.text)) or _facts(bare) & _facts(r.text))]
        if idle:
            rep.sentences.append(SentenceCheck(sent, cites, True, False, f"citation(s) that support nothing in the sentence: {idle}"))
            continue
        rt = set(content_tokens(corpus))
        overlap = len(st & rt) / max(len(st), 1)
        if overlap < min_overlap:
            rep.sentences.append(SentenceCheck(sent, cites, True, False, f"low overlap {overlap:.2f}"))
            continue
        src = _polarity_conflict(bare, st, recs)
        if src is not None:
            rep.sentences.append(SentenceCheck(sent, cites, True, False, f"polarity differs from the cited record: {src[:80]!r}"))
            continue
        rep.sentences.append(SentenceCheck(sent, cites, True, True, ""))
    return rep
