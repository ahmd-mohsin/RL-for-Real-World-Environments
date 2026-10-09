"""Citation-based grounding checks for simulated actors (the anti-fabrication gate of M2).

A grounded reply is a list of sentences; every *factual* sentence must cite at least one record
(``[R:<record id>]``) that (1) exists, (2) is visible at the world's current time, and (3) supports it:
- every number / amount / date / identifier-like token in the sentence appears verbatim in a cited record;
- at least ``min_overlap`` of the sentence's content words appear in the cited records.
Non-factual sentences (questions, greetings, acknowledgements, "no record") need no citation.

The check is deterministic and cheap; an optional LLM entailment judge (``fidelity.py``) audits samples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from crucible.actors.records import RecordStore, content_tokens

CITE = re.compile(r"\[R:([^\]\s]+)\]")
_SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
# numbers (incl. 1,234.50 / 12% / 3.5x), dates, and identifier-like tokens (INV-1042, PR#6012, sha prefixes)
_FACT_TOKEN = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?|\b[A-Z]{2,}[-_#]?\d+\b|\b[0-9a-f]{7,40}\b|\b\d{4}-\d{2}-\d{2}\b")
_PHATIC = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|ok(?:ay)?|sure|got it|understood|noted|sounds good|great|perfect|no problem|you're welcome|bye|goodbye)\b[\s\w,!.]*$",
    re.IGNORECASE,
)
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
    return len(content_tokens(bare)) >= 2 or bool(_FACT_TOKEN.search(bare))


def check(text: str, store: RecordStore, *, now: str | None = None, allowed_sources: set[str] | None = None, min_overlap: float = 0.5) -> GroundingReport:
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
            if r is None or (allowed_sources and r.source not in allowed_sources):
                bad.append(c)
            else:
                recs.append(r)
        if bad:
            rep.sentences.append(SentenceCheck(sent, cites, True, False, f"invalid or future citation(s): {bad}"))
            continue
        bare = CITE.sub("", sent)
        corpus = " ".join(r.text for r in recs)
        missing_facts = [f for f in _FACT_TOKEN.findall(bare) if f not in corpus]
        if missing_facts:
            rep.sentences.append(SentenceCheck(sent, cites, True, False, f"facts not in cited records: {missing_facts}"))
            continue
        st = set(content_tokens(bare))
        rt = set(content_tokens(corpus))
        overlap = len(st & rt) / max(len(st), 1)
        ok = overlap >= min_overlap
        rep.sentences.append(SentenceCheck(sent, cites, True, ok, "" if ok else f"low overlap {overlap:.2f}"))
    return rep
