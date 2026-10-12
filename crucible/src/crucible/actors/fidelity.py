"""Fidelity measurement for simulated actors (ARCHITECTURE M2 exit gate).

- ``grounding_audit``: re-checks every delivered reply of grounded actors against the records (deterministic),
  optionally asking an LLM judge (different family) whether each factual sentence is *entailed* by its
  cited records. Reports the fabrication rate = unsupported factual sentences / factual sentences.
- ``consistency_summary``: per-reply consistency verdicts collected by scenario actors' judges.
- ``location_metrics``: precision / recall of predicted (file, line) findings against recorded ones,
  with a line tolerance — the reviewer-simulator fidelity measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crucible.actors.grounding import CITE, NO_RECORD, check, split_sentences
from crucible.actors.llm import LLMClient, LLMError
from crucible.actors.records import RecordStore

ENTAIL = """The records below were written by the speaker of the statement ("I" in the statement is the author of the records).
Records:
{records}

Statement: {statement}

Does the statement say only what the records say? Count it as NOT entailed if it adds a fact, number, reason or decision,
turns a question into a claim (or the reverse), or states with certainty what a record states tentatively.
Ignore harmless changes of tense or wording. Return only JSON {{"entailed": true|false, "reason": "<one short sentence>"}}."""


@dataclass
class GroundingAudit:
    replies: int = 0
    no_record: int = 0
    factual_sentences: int = 0
    unsupported_sentences: int = 0
    judged: int = 0
    judge_rejected: int = 0
    judge_failed: int = 0
    verbatim_sentences: int = 0  # factual sentences copied verbatim from a cited record (supported by construction)
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fabrication_rate(self) -> float:
        denom = self.factual_sentences
        return (self.unsupported_sentences + self.judge_rejected) / denom if denom else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**{k: v for k, v in self.__dict__.items() if k != "examples"}, "fabrication_rate": round(self.fabrication_rate, 4), "examples": self.examples[:20]}


async def grounding_audit(replies: list[tuple[str, list[str], str | None]], store: RecordStore, *, judge: LLMClient | None = None,
                          judge_every: int = 1) -> GroundingAudit:
    """``replies``: (delivered text, citations, virtual time). Text has citation markers stripped, so
    each sentence is re-attached to the reply's citations before checking."""
    a = GroundingAudit()
    for i, (text, cites, now) in enumerate(replies):
        a.replies += 1
        if text.strip() == NO_RECORD:
            a.no_record += 1
            continue
        # citations go before the sentence's closing punctuation, so they stay with that sentence. Each sentence gets the
        # reply's citations that share a word or fact with it (all of them if none does): a two-record quote reply
        # tagged every sentence with both records, and the record a sentence did not come from looked like padding
        marked = " ".join(_cite(s, " ".join(f"[R:{c}]" for c in _supporting(s, cites, store))) for s in split_sentences(text))
        rep = check(marked, store, now=now)
        a.factual_sentences += len(rep.factual)
        a.unsupported_sentences += len(rep.unsupported)
        for s in rep.unsupported:
            a.examples.append({"sentence": CITE.sub("", s.text).strip(), "reason": s.reason})
        if judge is not None and i % judge_every == 0:
            recs = "\n".join(f"- {store.get(c).text}" for c in cites if store.get(c) is not None)
            corpus = _norm(" ".join(store.get(c).text for c in cites if store.get(c) is not None))
            for s in rep.factual:
                if not s.supported:
                    continue
                if _norm(CITE.sub("", s.text)) in corpus:
                    a.verbatim_sentences += 1  # a verbatim quote cannot add or change meaning: no judge needed
                    continue
                prompt = [{"role": "user", "content": ENTAIL.format(records=recs, statement=CITE.sub("", s.text).strip())}]
                try:
                    v = await judge.complete_json(prompt, max_tokens=500)
                except LLMError as e:
                    v = _salvage(str(e))
                    if v is None:
                        try:
                            v = await judge.complete_json(prompt, max_tokens=900, temperature=0.0, seed=1)
                        except LLMError as e2:
                            v = _salvage(str(e2))
                    if v is None:
                        a.judge_failed += 1
                        a.examples.append({"sentence": CITE.sub("", s.text).strip(), "reason": "judge failed (no verdict)"})
                        continue
                a.judged += 1
                if not v.get("entailed", False):
                    a.judge_rejected += 1
                    a.examples.append({"sentence": CITE.sub("", s.text).strip(), "reason": "judge: " + str(v.get("reason", ""))[:200]})
    return a


def _norm(text: str) -> str:
    import re

    t = " ".join(text.split())
    t = re.sub(r"\s+([.,;:!?。！？])", r"\1", t)  # "rebase ." -> "rebase." (left behind by removed citation tags)
    return t.rstrip(".!?。！？ ").strip()


def _supporting(sentence: str, cites: list[str], store: RecordStore) -> list[str]:
    from crucible.actors.grounding import _facts
    from crucible.actors.records import content_tokens

    st, facts = set(content_tokens(sentence)), _facts(sentence)
    if not st and not facts:  # a bare fence or bullet: no claim to cite (the re-split would glue its tags onto the next sentence)
        return []
    keep = [c for c in cites if (r := store.get(c)) is not None and (st & set(content_tokens(r.text)) or facts & _facts(r.text))]
    return keep or cites


def _cite(sentence: str, tags: str) -> str:
    if not tags:
        return sentence
    stripped = sentence.rstrip()
    end = ""
    while stripped and stripped[-1] in ".!?。！？":
        end = stripped[-1] + end
        stripped = stripped[:-1]
    return f"{stripped} {tags}{end}"


def _salvage(err: str) -> dict[str, Any] | None:
    """Recover the verdict from a truncated JSON reply ('{"entailed": false, "reason": "...')."""
    import re

    m = re.search(r'"entailed"\s*:\s*(true|false)', err)
    return {"entailed": m.group(1) == "true", "reason": "(truncated)"} if m else None


def consistency_summary(audit: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(audit)
    bad = sum(1 for x in audit if not x.get("consistent", True))
    return {"judged_replies": n, "inconsistent": bad, "inconsistency_rate": round(bad / n, 4) if n else 0.0}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    key: str = ""  # optional identity (e.g. recorded comment id)


def location_metrics(pred: list[Finding], gold: list[Finding], *, tolerance: int = 3) -> dict[str, float]:
    matched_gold: set[int] = set()
    tp = 0
    for p in pred:
        for gi, g in enumerate(gold):
            if gi in matched_gold or g.path != p.path or abs(g.line - p.line) > tolerance:
                continue
            matched_gold.add(gi)
            tp += 1
            break
    precision = tp / len(pred) if pred else 1.0
    recall = tp / len(gold) if gold else 1.0
    return {"tp": tp, "pred": len(pred), "gold": len(gold), "precision": round(precision, 4), "recall": round(recall, 4)}
