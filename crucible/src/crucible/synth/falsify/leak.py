"""Paraphrase-aware leak detection (D7; DESIGN §7.7 item 20, the humanised-leak blind spot of RSI-1/2).

TAINT's ``derive`` probe asks whether the instruction gives away a value the agent must work out. Its M2 matcher needs
the exact token, so "because of a price variance" for ``price_variance`` was missed (locked set: 14 of 18 leaks found).
This module adds **structural** layers (executed text evidence, no model) and an **L2** model layer.

**Structural layers** (``LeakMatcher``; each hit names the first layer that found it, so recall and false alarms are
measured per layer):

| layer | finds | example |
|---|---|---|
| ``exact`` | M2's ``text_spans``: the value as a token sequence, numbers numerically | "reason code price_variance" |
| ``sep`` | the value's words in order, separators and case folded (``_ - . /``, camelCase) | "price variance", "Price-Variance" |
| ``number`` | a numeric value written in words | "one hundred and eight" for 108 |
| ``lemma`` | the value's content words by stem, any order, within a window of n + 3 tokens | "the vendor is blocked" for ``vendor_blocked`` |
| ``label`` | a short label the agent can read for the value (tool catalogue, π*'s observations): ``code (Label)``, ``code: Label``, ``Label (code)`` | "Critical" for ``P1`` when the policy reads "P1 (Critical)" |
| ``synonym`` | the lemma rule with words replaced through a ``SynonymMap`` (data learned from confirmed leaks) | "price mismatch" when the map holds variance ~ mismatch |

Identifiers (a value with a digit, e.g. ``T-1001``) get ``exact`` and ``sep`` only; values whose words are all shorter
than 3 letters get ``exact`` only. **Options guard:** a paraphrase hit (every layer but ``exact``, whose M2 behaviour is
kept) in a sentence that also names a *sibling* value (another value of the same enum or field) is an enumeration of the
possible outcomes ("approve it, put it on hold or reject it"), not a give-away: it is recorded with ``options = True``
and not counted.

**Synonym maps are data** (``SynonymMap``): versioned entries ``stem -> variants`` with their sources and their evidence
kind. ``learn_synonyms`` aligns the words of a confirmed leak's value with the words of its quoted span (stems that
match are dropped; one unmatched value word maps to the unmatched quote words), so a leak the L2 layer found once is
caught structurally the next time. An entry learned from a panel decision is ``kind = "judge"``; TAINT labels a hit that
used one a *suspect*, and ``synonym_update`` builds the map change as an ``update.Update`` with judge evidence, so R8's
round cap applies to it. Entries from executed witnesses (a planted leak's recorded span) are ``executed``.

**L2** (``LeakProbe``, class ``LEAK``): one model of a non-builder family (``roles.LEAK_PROBE``) reads the instruction
and the decisions to work out, and quotes any give-away; a quote that is not in the instruction is fed back as an error.
Each quote is a **suspect** D7. ``LeakPanel`` (two families, ``roles.LEAK_PANEL``; calibrated on executed truth by
``calibrate_leak_panel``: planted leaks' recorded spans are leaks, certified anchors' values with a non-leaking quote are
not) decides them: agreement at confidence ≥ λ confirms (``leak``) or clears; otherwise the suspect goes to the human
queue. ``max_calls`` caps the probe's calls per run; a probe past the cap is invalid (not a trial).
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import cjson, content_hash, is_number, norm_text
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, Label, Outcome, Probe
from crucible.synth.program import leaves, text_spans
from crucible.synth.task import Task

LAYERS = ("exact", "sep", "number", "lemma", "label", "synonym")
STOP = frozenset(["a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "by", "at", "as", "is", "are", "be", "it", "its", "this", "that", "from", "into", "per", "via", "no", "not"])
_WORD = re.compile(r"[A-Za-z][A-Za-z']*|\d+(?:[.,]\d+)*")
_SENT = re.compile(r"[.!?;\n]+")

# ---------------------------------------------------------------------------------------------------------- words


@dataclass(frozen=True)
class Tok:
    w: str  # lower case
    a: int
    b: int


def tokens(text: str) -> list[Tok]:
    return [Tok(m.group(0).lower().strip("'"), m.start(), m.end()) for m in _WORD.finditer(text)]


def value_words(v: Any) -> list[str]:
    """The words of a value: ``price_variance`` / ``PriceVariance`` / ``price-variance`` -> ["price", "variance"]."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(v))
    return [w.lower() for w in re.split(r"[^A-Za-z0-9]+", s) if w]


def is_identifier(v: Any) -> bool:
    return isinstance(v, str) and any(c.isdigit() for c in v)


_SUFFIXES = ("izations", "ization", "isation", "ational", "fulness", "iveness", "ications", "ication", "ations", "ation", "ments", "ment",
             "nesses", "ness", "ings", "ing", "ied", "ies", "ed", "es", "s", "ly", "ances", "ance", "ences", "ence", "ity", "ive", "al", "er")


def stem(w: str) -> str:
    """A light suffix stripper (no dictionary): "verification" and "verify" -> "verif"; "blocked" -> "block"."""
    w = w.lower()
    if len(w) <= 3 or w.isdigit():
        return w
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)] + ("y" if suf in ("ied", "ies") else "")
            break
    if w.endswith("y") and len(w) > 4:
        w = w[:-1]
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    if len(w) > 4 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]
    return w


def stems_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 4 or a[0] != b[0]:
        return False
    short, long_ = sorted((a, b), key=len)
    return long_.startswith(short) and len(long_) - len(short) <= 3


_UNITS = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"])}
_TENS = {w: 10 * (i + 2) for i, w in enumerate(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"])}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}


def number_mentions(text: str) -> list[tuple[float, int, int]]:
    """Numbers written in words ("one hundred and eight", "twenty-five", "a dozen"), with their spans."""
    flat = text.replace("-", " ")
    toks = tokens(flat)
    out: list[tuple[float, int, int]] = []

    def joined(j: int) -> bool:  # no punctuation between token j-1 and token j
        return not flat[toks[j - 1].b:toks[j].a].strip()

    i = 0
    while i < len(toks):
        w = toks[i].w
        if w == "dozen" and i > 0 and toks[i - 1].w in ("a", "one"):
            out.append((12.0, toks[i - 1].a, toks[i].b))
            i += 1
            continue
        if w not in _UNITS and w not in _TENS:
            i += 1
            continue
        total, cur, j, last = 0, 0, i, i
        while j < len(toks):
            if j > i and not joined(j):
                break
            x = toks[j].w
            if x in _UNITS:
                cur += _UNITS[x]
            elif x in _TENS:
                cur += _TENS[x]
            elif x in _SCALES and (cur or total):
                cur = max(cur, 1) * _SCALES[x]
                if _SCALES[x] >= 1000:
                    total, cur = total + cur, 0
            elif x == "and" and j + 1 < len(toks) and (toks[j + 1].w in _UNITS or toks[j + 1].w in _TENS):
                j += 1
                continue
            else:
                break
            last = j
            j += 1
        out.append((float(total + cur), toks[i].a, toks[last].b))
        i = last + 1
    return out


def sentence_of(text: str, a: int) -> tuple[int, int]:
    starts = [0] + [m.end() for m in _SENT.finditer(text) if m.end() <= a]
    s = starts[-1]
    m = _SENT.search(text, a)
    return s, (m.start() if m else len(text))


# ---------------------------------------------------------------------------------------------------------- synonym maps

class SynEntry(CrucibleModel):
    word: str  # a stem
    variants: list[str] = Field(default_factory=list)  # stems or space-joined stem phrases
    sources: list[str] = Field(default_factory=list)
    kind: Literal["executed", "judge"] = "executed"


class SynonymMap(CrucibleModel):
    schema_version: str = "crucible.synth.leak-synonyms/v1"
    version: int = 1
    entries: list[SynEntry] = Field(default_factory=list)

    def variants(self, s: str) -> list[tuple[list[str], str]]:
        """(stem phrase, kind) for every variant of stem ``s`` (both directions of an entry)."""
        out: list[tuple[list[str], str]] = []
        for e in self.entries:
            if stems_match(e.word, s):
                out += [(v.split(" "), e.kind) for v in e.variants]
            elif any(" " not in v and stems_match(v, s) for v in e.variants):
                out.append(([e.word], e.kind))
        return out

    def digest(self) -> str:
        return content_hash(self.model_dump(mode="json"), 16)

    def merged(self, new: Iterable[SynEntry]) -> SynonymMap:
        by = {e.word: e.model_copy(deep=True) for e in self.entries}
        for e in new:
            cur = by.get(e.word)
            if cur is None:
                by[e.word] = e.model_copy(deep=True)
                continue
            cur.variants = sorted(set(cur.variants) | set(e.variants))
            cur.sources = sorted(set(cur.sources) | set(e.sources))
            cur.kind = "judge" if "judge" in (cur.kind, e.kind) else "executed"
        return SynonymMap(version=self.version + 1, entries=sorted(by.values(), key=lambda e: e.word))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=1))
        return p

    @classmethod
    def load(cls, path: str | Path) -> SynonymMap:
        p = Path(path)
        return cls.model_validate_json(p.read_text()) if p.is_file() else cls()


def _content(words: Iterable[str]) -> list[str]:
    return [w for w in words if w not in STOP]


def align(value: Any, quote: str) -> list[tuple[str, str]]:
    """(value stem, quote phrase) pairs from one confirmed leak: value words whose stem appears in the quote are dropped;
    if exactly one value word is left, it maps to the remaining content words of the quote (at most 3, as one phrase)."""
    vs = [stem(w) for w in _content(value_words(value)) if len(w) >= 3]
    qs = [stem(t.w) for t in tokens(quote) if t.w not in STOP and len(t.w) >= 3]
    left_v = [v for v in vs if not any(stems_match(v, q) for q in qs)]
    left_q = [q for q in qs if not any(stems_match(v, q) for v in vs)]
    if len(left_v) == 1 and 1 <= len(left_q) <= 3:
        return [(left_v[0], " ".join(left_q))]
    if len(left_v) == len(left_q) == 2 and len(vs) == 2:  # two words, both reworded, kept in order
        return [(left_v[0], left_q[0]), (left_v[1], left_q[1])]
    return []


def learn_synonyms(confirmed: Iterable[dict[str, Any]], *, base: SynonymMap | None = None) -> SynonymMap:
    """A new map version from confirmed leaks: each item is ``{"value", "quote", "source", "kind"}`` (an archived L2 leak
    the panel confirmed: kind judge; a planted leak's recorded span: kind executed)."""
    new: list[SynEntry] = []
    for c in confirmed:
        for v, q in align(c["value"], str(c["quote"])):
            if v == q:
                continue
            new.append(SynEntry(word=v, variants=[q], sources=[str(c.get("source") or "")], kind=c.get("kind", "executed")))
    return (base or SynonymMap(version=0)).merged(new)


def leaks_from_archive(cxs: Iterable[Any]) -> list[dict[str, Any]]:
    """Confirmed leaks with their quotes from archived counterexamples (``archive.Archive.records``): the L2 class's
    panel-confirmed finds (kind judge). TAINT's own D7 records are exact matches and teach no synonym."""
    out = []
    for cx in cxs:
        if getattr(cx, "cls", "") != "LEAK" or getattr(cx, "label", "") != "confirmed":
            continue
        for f in cx.probe.data.get("finds") or []:
            if f.get("panel") == "leak":
                out.append({"value": f["value"], "quote": f["quote"], "source": cx.id, "kind": "judge"})
    return out


def synonym_update(old: SynonymMap, new: SynonymMap, *, round: int, author_family: str = "", new_executed: int = 0) -> Any:
    """The map change as an ``update.Update`` (component ``leak_synonyms``) with judge evidence when any added entry came
    from a panel decision, so R8's judge-only round cap applies (``UpdateEngine.decide``)."""
    from crucible.synth.meta.update import Evidence, Update

    olds = {(e.word, v) for e in old.entries for v in e.variants}
    added = [(e.word, v, e.kind) for e in new.entries for v in e.variants if (e.word, v) not in olds]
    kind = "judge" if any(k == "judge" for *_, k in added) else "executed"
    return Update(id=f"leak_synonyms-r{round}-{new.digest()}", component="leak_synonyms", data=new.model_dump(mode="json"),
                  edits=[f"{w} ~ {v}" for w, v, _ in added], items=[f"{w}~{v}" for w, v, _ in added], author_family=author_family,
                  evidence=Evidence(kind=kind, new_executed=new_executed))


# ---------------------------------------------------------------------------------------------------------- labels

_LABEL_AFTER = r"\s*(?:\(|:|=|-|–|—)\s*([A-Za-z][A-Za-z '\-]{1,40}?)\s*(?:\)|[,.;=\n]|$)"
_LABEL_BEFORE = r"([A-Za-z][A-Za-z '\-]{1,40}?)\s*\(\s*{v}\s*\)"


def extract_labels(value: Any, texts: Iterable[str], *, max_words: int = 4) -> list[str]:
    """Short labels paired with a value in texts the agent can read: ``v (Label)``, ``v: Label``, ``v = Label``,
    ``v - Label``, ``Label (v)``. A label of more than ``max_words`` words, or one that is the value itself, is dropped."""
    v = re.escape(str(value))
    out: list[str] = []
    for t in texts:
        if not t or str(value) not in t:
            continue
        for m in re.finditer(rf"(?<![\w-]){v}(?![\w-])" + _LABEL_AFTER, t):
            out.append(m.group(1))
        for m in re.finditer(_LABEL_BEFORE.replace("{v}", v), t):
            out.append(m.group(1).split(",")[-1])
    keep: list[str] = []
    vw = value_words(value)
    for lab in out:
        lab = lab.strip(" -'")
        ws = _content(value_words(lab))
        if not ws or len(ws) > max_words or ws == vw or not any(len(w) >= 3 for w in ws):
            continue
        if lab.lower() not in [k.lower() for k in keep]:
            keep.append(lab)
    return keep


# ---------------------------------------------------------------------------------------------------------- the matcher

class LeakHit(CrucibleModel):
    layer: str
    span: tuple[int, int]
    quoted: str
    options: bool = False  # the sentence also names a sibling value: an enumeration, not a give-away
    via: list[str] = Field(default_factory=list)  # synonym entries or the label used
    judge: bool = False  # a synonym entry with judge evidence was used


class LeakMatcher:
    """The structural layers (module docstring) over one text."""

    def __init__(self, layers: Sequence[str] = LAYERS, synonyms: SynonymMap | None = None, *, window_slack: int = 3):
        bad = [x for x in layers if x not in LAYERS]
        if bad:
            raise ValueError(f"unknown leak layers {bad}")
        self.layers = tuple(x for x in LAYERS if x in layers)
        self.synonyms = synonyms or SynonymMap()
        self.slack = window_slack

    def find(self, value: Any, text: str, *, labels: Sequence[str] = (), siblings: Sequence[Any] = ()) -> list[LeakHit]:
        """Every hit of ``value`` in ``text`` by the first layer (in ``LAYERS`` order) that finds one; options-guarded."""
        for layer in self.layers:
            hits = self._layer(layer, value, text, labels)
            if not hits:
                continue
            if layer != "exact":
                for h in hits:
                    h.options = self._options(h, text, value, siblings)
            return hits
        return []

    def counted(self, hits: Sequence[LeakHit]) -> list[LeakHit]:
        return [h for h in hits if not h.options]

    # ---- layers ----
    def _layer(self, layer: str, value: Any, text: str, labels: Sequence[str]) -> list[LeakHit]:
        if layer == "exact":
            return [LeakHit(layer="exact", span=s, quoted=text[s[0]:s[1]]) for s in text_spans(value, text)]
        num = _numeric(value)
        if layer == "number":
            if num is None:
                return []
            return [LeakHit(layer="number", span=(a, b), quoted=text[a:b]) for x, a, b in number_mentions(text) if math.isclose(x, num)]
        if num is not None and not isinstance(value, str):
            return []
        vw = value_words(value)
        if not vw:
            return []
        toks = tokens(text)
        if layer == "sep":
            if len(vw) < 2 and not is_identifier(value):
                return []
            return [self._hit("sep", text, toks, i, i + len(vw) - 1) for i in range(len(toks) - len(vw) + 1)
                    if [t.w for t in toks[i:i + len(vw)]] == vw]
        if layer == "label":
            return self._labels(text, toks, labels)
        if is_identifier(value) or not any(len(w) >= 3 for w in vw):
            return []
        content = [stem(w) for w in _content(vw)]
        if layer == "lemma":
            return self._window("lemma", text, toks, [[[c]] for c in content])
        if layer == "synonym":
            alts = []
            for c in content:
                vs = self.synonyms.variants(c)
                alts.append([[c]] + [v for v, _ in vs])
            if all(len(a) == 1 for a in alts):
                return []
            hits = self._window("synonym", text, toks, alts, need_alt=True)
            for h in hits:
                h.judge = any(k == "judge" for c in content for v, k in self.synonyms.variants(c) if " ".join(v) in h.via)
            return hits
        return []

    def _labels(self, text: str, toks: list[Tok], labels: Sequence[str]) -> list[LeakHit]:
        out: list[LeakHit] = []
        for lab in labels:
            lw = value_words(lab)
            for i in range(len(toks) - len(lw) + 1):
                if [t.w for t in toks[i:i + len(lw)]] == lw:
                    h = self._hit("label", text, toks, i, i + len(lw) - 1)
                    h.via = [lab]
                    out.append(h)
            if not out:
                lc = [[[stem(w)]] for w in _content(lw) if len(w) >= 3]
                if lc:
                    for h in self._window("label", text, toks, lc):
                        h.via = [lab]
                        out.append(h)
        return out

    def _hit(self, layer: str, text: str, toks: list[Tok], i: int, j: int) -> LeakHit:
        return LeakHit(layer=layer, span=(toks[i].a, toks[j].b), quoted=text[toks[i].a:toks[j].b])

    def _window(self, layer: str, text: str, toks: list[Tok], alts: list[list[list[str]]], *, need_alt: bool = False) -> list[LeakHit]:
        """Every value word matched (by one of its stem phrases) inside a window of len + slack tokens, any order."""
        n = len(alts)
        if n == 0:
            return []
        stems = [stem(t.w) for t in toks]
        width = n + self.slack if n > 1 else 1
        out: list[LeakHit] = []
        seen: set[tuple[int, int]] = set()
        for i in range(len(toks)):
            used: set[int] = set()
            via: list[str] = []
            alt_used = False
            ok = True
            for k, options in enumerate(alts):
                got = False
                for oi, phrase in enumerate(options):
                    for j in range(i, min(len(toks), i + max(width, len(phrase)))):
                        if any(x in used for x in range(j, j + len(phrase))) or j + len(phrase) > len(toks):
                            continue
                        if all(stems_match(phrase[q], stems[j + q]) for q in range(len(phrase))):
                            used.update(range(j, j + len(phrase)))
                            if oi > 0:
                                alt_used = True
                                via.append(" ".join(phrase))
                            got = True
                            break
                    if got:
                        break
                if not got:
                    ok = False
                    break
            if not ok or (need_alt and not alt_used) or i not in used:
                continue
            span = (toks[min(used)].a, toks[max(used)].b)
            if span in seen:
                continue
            seen.add(span)
            out.append(LeakHit(layer=layer, span=span, quoted=text[span[0]:span[1]], via=via))
        return out

    def _options(self, h: LeakHit, text: str, value: Any, siblings: Sequence[Any]) -> bool:
        a, b = sentence_of(text, h.span[0])
        sent = text[a:b]
        inner = LeakMatcher(("exact", "sep", "lemma"))
        for s in siblings:
            if s is None or cjson(s) == cjson(value):
                continue
            if inner._layer("exact", s, sent, ()) or inner._layer("sep", s, sent, ()) or inner._layer("lemma", s, sent, ()):
                return True
        return False


def _numeric(v: Any) -> float | None:
    if is_number(v):
        return float(v)
    if isinstance(v, str) and re.fullmatch(r"\s*[-+]?\d+(\.\d+)?\s*", v):
        return float(v)
    return None


def readable_texts(tools: Sequence[Any], observations: Iterable[Any]) -> list[str]:
    """Texts an agent can read about the values: tool descriptions and enums, and string leaves of π*'s observations."""
    out = []
    for t in tools:
        if getattr(t, "description", ""):
            out.append(t.description)
        for p in ((getattr(t, "input_schema", None) or {}).get("properties") or {}).values():
            if isinstance(p, dict) and p.get("description"):
                out.append(str(p["description"]))
    for obs in observations:
        for _, x in leaves(obs):
            if isinstance(x, str) and len(x) > 3:
                out.append(x)
    return out


def sibling_values(value: Any, tools: Sequence[Any], start: Any, delta: Sequence[Any] = ()) -> list[Any]:
    """Other values of the same kind: the enums of the tool catalogue that hold ``value``, and the start-state values of
    every field that ``value`` is written to (Δ*) or held in at s₀."""
    out: list[Any] = []
    for t in tools:
        for p in ((getattr(t, "input_schema", None) or {}).get("properties") or {}).values():
            enum = (p or {}).get("enum") if isinstance(p, dict) else None
            if enum and any(cjson(e) == cjson(value) for e in enum):
                out += list(enum)
    fields: set[tuple[str, str, str]] = set()
    for ch in delta:
        if getattr(ch, "kind", "") == "update" and cjson(getattr(ch, "new", None)) == cjson(value):
            fields.add((ch.addr.app, ch.addr.coll, ch.addr.field or ""))
    if start is not None:
        for app, coll, _key, row in start.entities():
            for f, x in row.items():
                if cjson(x) == cjson(value):
                    fields.add((app, coll, f))
        for app, coll, f in fields:
            for row in start.collection(app, coll).values():
                x = row.get(f)
                if isinstance(x, str | int) and not isinstance(x, bool):
                    out.append(x)
    seen: list[Any] = []
    for x in out:
        if cjson(x) != cjson(value) and x not in seen:
            seen.append(x)
    return seen[:40]


# ---------------------------------------------------------------------------------------------------------- L2

@dataclass
class LeakPanel:
    """Two or more families decide a leak suspect (``roles.LEAK_PANEL``): agreement at confidence ≥ λ, else abstain and
    queue it for a person. ``members`` are ``agents.PanelMember``s."""

    members: list[Any]
    lam: float = 0.7
    queue: Path | None = None
    tag: str = "adjudicate:leak"
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        from crucible.synth.roles import LEAK_PANEL

        if len({m.family for m in self.members}) < LEAK_PANEL.min_families:
            raise ValueError(f"the leak panel needs >= {LEAK_PANEL.min_families} families")

    async def judge(self, instruction: str, decision: Any, quote: str, *, tag: str | None = None) -> dict[str, Any]:
        from crucible.actors.llm import BudgetExceeded
        from crucible.synth.roles import LEAK_PANEL, LeakVerdictOut

        msgs = LEAK_PANEL.render(instruction=instruction, decision=decision, quote=quote)

        async def one(m: Any) -> dict[str, Any]:
            try:
                doc = await m.client.complete_json(msgs, schema=LeakVerdictOut, max_tokens=LEAK_PANEL.max_tokens, temperature=0.0, tag=tag or self.tag)
                o = LeakVerdictOut.model_validate(doc)
                return {"member": m.name, "family": m.family, "leak": o.leak, "confidence": o.confidence, "why": o.why[:300]}
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001 - a member that cannot answer abstains
                return {"member": m.name, "family": m.family, "leak": None, "confidence": 0.0, "error": f"{type(e).__name__}: {str(e)[:200]}"}

        votes = list(await asyncio.gather(*(one(m) for m in self.members)))
        return {"votes": votes, "decision": self.decision(votes, self.lam)}

    @staticmethod
    def decision(votes: Sequence[dict[str, Any]], lam: float) -> str:
        vals = {v.get("leak") for v in votes}
        if None in vals or len(vals) != 1 or min(float(v.get("confidence") or 0) for v in votes) < lam:
            return "abstain"
        return "leak" if vals.pop() else "clean"

    async def decide(self, task: Task, value: Any, quote: str) -> str:
        v = await self.judge(task.instruction, value, quote)
        row = {"task": task.id, "version": task.version, "value": value, "quote": quote, "decision": v["decision"], "votes": v["votes"], "lam": self.lam}
        self.log.append(row)
        if v["decision"] == "abstain" and self.queue is not None:
            self.queue.parent.mkdir(parents=True, exist_ok=True)
            with self.queue.open("a") as fh:
                fh.write(json.dumps({**row, "cls": "LEAK", "instruction": task.instruction}, default=str) + "\n")
        return v["decision"]


class LeakAnchor(CrucibleModel):
    instruction: str
    value: Any
    quote: str
    truth: bool
    kind: str = ""


async def calibrate_leak_panel(panel: LeakPanel, anchors: Sequence[LeakAnchor], *, target: float = 0.9,
                               grid: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95), floor: float = 0.7, concurrency: int = 8,
                               min_decided: int = 10) -> dict[str, Any]:
    """λ from executed-truth anchors (the panel's ``calibrate`` rule): the smallest grid value whose decided anchors reach
    ``target`` accuracy, never below ``floor``; none reaches it -> the largest, flagged uncalibrated."""
    from crucible.verify.calibrate import clopper_pearson_upper

    sem = asyncio.Semaphore(concurrency)

    async def one(a: LeakAnchor) -> dict[str, Any]:
        async with sem:
            return await panel.judge(a.instruction, a.value, a.quote, tag="calibrate:leak")

    verdicts = await asyncio.gather(*(one(a) for a in anchors))
    rows, chosen = [], None
    for lam in grid:
        dec = cor = 0
        truths: set[bool] = set()
        for a, v in zip(anchors, verdicts, strict=True):
            d = LeakPanel.decision(v["votes"], lam)
            if d == "abstain":
                continue
            dec += 1
            truths.add(a.truth)
            cor += int((d == "leak") == a.truth)
        acc = cor / dec if dec else 0.0
        rows.append({"lam": lam, "decided": dec, "correct": cor, "accuracy": round(acc, 4), "coverage": round(dec / max(1, len(anchors)), 4),
                     "error_upper95": round(clopper_pearson_upper(dec - cor, dec, 0.05), 4) if dec else 1.0})
        if chosen is None and dec >= min_decided and acc >= target and len(truths) == 2:  # both truths decided, enough of them
            chosen = lam
    panel.lam = max(floor, chosen if chosen is not None else 1.01)  # uncalibrated: decide nothing (agents.calibrate, item 20)
    return {"lam": panel.lam, "calibrated": chosen is not None, "rows": rows,
            "votes": [{"kind": a.kind, "truth": a.truth, "value": a.value, "quote": a.quote, "votes": v["votes"]} for a, v in zip(anchors, verdicts, strict=True)]}


class LeakProbe(BaseFalsifier):
    """L2 leak evidence (module docstring). ``client`` is a ``roles.RoleClient`` of a non-builder family; ``panel`` an
    optional ``LeakPanel``; ``skip_structural`` leaves out values the structural layers already confirm (no money spent
    on what TAINT finds); ``max_calls`` caps the probe's model calls per run (R8's round cap on judge-only evidence)."""

    name: ClassVar[str] = "LEAK"

    def __init__(self, client: Any, *, panel: LeakPanel | None = None, max_calls: int = 200, min_confidence: float = 0.5,
                 skip_structural: bool = True, matcher: LeakMatcher | None = None):
        self.client, self.panel, self.max_calls, self.min_confidence = client, panel, max_calls, min_confidence
        self.skip_structural = skip_structural
        self.matcher = matcher or LeakMatcher()
        self.calls = 0
        self._cache: dict[str, list[dict[str, Any]]] = {}

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        return None if task.intent.derive else "no derive values: nothing to give away"

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        vals = list(task.intent.derive)
        if self.skip_structural:
            vals = [v for v in vals if not self.matcher.counted(self.matcher.find(v, task.instruction))]
        if not vals:
            return []
        return [Probe(cls="LEAK", op="model", site=f"model:{content_hash([cjson(v) for v in vals], 12)}", expect="property", data={"values": vals})]

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        from crucible.synth.roles import LEAK_PROBE, ask

        vals = probe.data["values"]
        key = content_hash({"ins": task.instruction, "vals": [cjson(v) for v in vals]}, 24)
        if key not in self._cache:
            if self.calls >= self.max_calls:
                return Outcome(valid=False, error="leak probe call cap reached")
            self.calls += 1
            ins_n = norm_text(task.instruction)
            want = {cjson(v) for v in vals}

            def check(out: Any) -> Any:
                for f in out.leaks:
                    if cjson(f.value) not in want and str(f.value) not in {str(v) for v in vals}:
                        raise ValueError(f"{f.value!r} is not one of the listed decisions")
                    if norm_text(f.quote) not in ins_n:
                        raise ValueError(f"the quote {f.quote!r} is not in the instruction: copy its exact words")
                return None

            try:
                out = await ask(self.client, LEAK_PROBE, tag=f"LEAK:{task.id}", check=check, instruction=task.instruction, decisions=vals)
            except Exception as e:
                from crucible.actors.llm import BudgetExceeded

                if isinstance(e, BudgetExceeded):
                    raise
                return Outcome(valid=False, error=f"{type(e).__name__}: {str(e)[:200]}")
            self._cache[key] = [f.model_dump(mode="json") for f in out.leaks if f.confidence >= self.min_confidence]
        finds = self._cache[key]
        decided = []
        for f in finds:
            d = await self.panel.decide(task, f["value"], f["quote"]) if self.panel is not None else "open"
            decided.append({**f, "panel": d})
        return Outcome(data={"finds": decided, "hit": bool(decided), "evidence": "L2"})

    def finalize(self, findings: list[Any]) -> list[Any]:
        """The finds ride on the probe, so an archived counterexample keeps its quotes (``leaks_from_archive``)."""
        out = []
        for f in findings:
            finds = f.outcome.data.get("finds")
            out.append(f.model_copy(update={"probe": f.probe.model_copy(update={"data": {**f.probe.data, "finds": finds}})}) if finds else f)
        return out

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        finds = outcome.data.get("finds") or []
        if any(f.get("panel") == "leak" for f in finds):
            return "confirmed", "D7"
        if any(f.get("panel") in ("open", "abstain") for f in finds):
            return "suspect", "D7"
        return "clean", None


__all__ = ["LAYERS", "LeakAnchor", "LeakHit", "LeakMatcher", "LeakPanel", "LeakProbe", "SynEntry", "SynonymMap", "align", "calibrate_leak_panel",
           "extract_labels", "is_identifier", "leaks_from_archive", "learn_synonyms", "number_mentions", "readable_texts", "sibling_values", "stem", "stems_match",
           "synonym_update", "tokens", "value_words"]
