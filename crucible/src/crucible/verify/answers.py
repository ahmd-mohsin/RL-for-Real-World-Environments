"""Answer parsers and checkers for single-turn tasks (M1 item 2; P1 and P16 of the M1 taxonomy).

A single-turn task is graded in two steps: a **parser** pulls the answer out of the completion, then a **checker**
compares it with the reference. Both are named, data-described (``ParserSpec`` / ``CheckerSpec``) so a dataset row or a
task file selects them without code.

Ported (see THIRD_PARTY.md):

- NeMo-Gym ``resources_servers/mcqa/app.py`` (Apache-2.0): last-``\\boxed{}`` extraction with brace counting, LaTeX
  ``\\text{}`` unwrapping, strict and lenient multiple-choice letter parsing, ``Answer:`` payloads, option-text matching
  that must be unambiguous, ambiguous choice lists ("A or B") refused, non-Latin letter forms folded;
- NeMo-Gym ``resources_servers/equivalence_rule/app.py`` (Apache-2.0): ``exact``, ``seq_match``, ``weighted_seq_match``;
- NeMo-Gym ``resources_servers/math_with_judge/app.py`` (Apache-2.0): math-verify in a killable child process with a
  deadline (a wedged SymPy call cannot hang the grader), outer ``$``/``\\(`` delimiters stripped from the reference;
- verifiers v1 ``utils/score.py`` (MIT): an unclosed ``<think>`` is no answer; the answer is read after the last
  ``</think>``; judge verdict parsing (``FINAL VERDICT: X``, boxed, last standalone choice).

Verdict semantics: a missing or unparsable answer scores 0 (the policy's failure). A checker that cannot decide (a
judge outage, math-verify not installed when ``math`` is requested) raises ``CheckerUnavailable``, which the
single-turn grader reports as ``infra``: the sample is masked, never scored 0.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import itertools
import json
import math
import multiprocessing as mp
import os
import re
import sys
import unicodedata
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

BOXED = "\\boxed{"


class CheckerUnavailable(RuntimeError):
    """The checker could not reach a verdict for reasons that are not the policy's (masked, not scored)."""


# ----------------------------------------------------------------------------------------------- reasoning traces
def after_think(text: str, *, require_closed: bool = True) -> str | None:
    """The text after the last ``</think>``. An opened, never-closed ``<think>`` is no answer (verifiers v1)."""
    if text is None:
        return None
    if "<think>" in text and "</think>" not in text:
        return None if require_closed else text.split("<think>", 1)[0]
    return text.rsplit("</think>", 1)[-1]


# ----------------------------------------------------------------------------------------------- extraction
def last_boxed(text: str) -> str | None:
    """Everything inside the last ``\\boxed{...}`` (braces counted), or None (NeMo-Gym ``_extract_boxed_inner``)."""
    start = text.rfind(BOXED)
    if start == -1:
        return None
    depth, buf = 1, []
    for ch in text[start + len(BOXED):]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf)
        buf.append(ch)
    return None


_TEXT_WRAP = re.compile(r"\\text\{\s*(.*?)\s*\}", re.DOTALL)


def strip_latex_text(s: str) -> str:
    while (m := _TEXT_WRAP.fullmatch(s)) is not None:
        s = m.group(1)
    return s


_ANSWER_PAYLOAD = re.compile(r"(?im)[*_]{0,2}Answer[*_]{0,2}\s*:[*_\s]{0,2}(.+)")


def answer_payloads(text: str, prefix: str | None = None) -> list[str]:
    """Every ``Answer: ...`` payload (markdown-tolerant) and every line after a row-given ``prefix``, in text order."""
    text = unicodedata.normalize("NFKC", text)
    found = [(m.start(), m.group(1)) for m in _ANSWER_PAYLOAD.finditer(text)]
    if prefix:
        pat = re.compile(re.escape(unicodedata.normalize("NFKC", prefix)) + r"\s*", re.IGNORECASE)
        for m in pat.finditer(text):
            end = text.find("\n", m.end())
            found.append((m.start(), text[m.end(): end if end >= 0 else len(text)]))
    return [p for _, p in sorted(found, key=lambda x: x[0])]


def xml_tag(text: str, tag: str) -> str | None:
    """The content of the last complete ``<tag>...</tag>``."""
    hits = re.findall(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
    return hits[-1].strip() if hits else None


def code_block(text: str, lang: str | None = None) -> str | None:
    """The last fenced code block (optionally only ``lang`` blocks)."""
    hits = re.findall(r"```([^\n`]*)\n(.*?)```", text, re.DOTALL)
    if lang:
        hits = [h for h in hits if h[0].strip().lower() == lang.lower()]
    return hits[-1][1] if hits else None


def json_value(text: str) -> Any:
    """The last JSON object or array that parses (a fenced ``json`` block first). Raises ValueError when none does."""
    block = code_block(text, "json")
    if block is not None:
        with contextlib.suppress(ValueError):
            return json.loads(block)
    dec = json.JSONDecoder()
    last: Any = _NONE
    last_end = -1
    i = 0
    while i < len(text):  # the outermost value that ends last (M7 audit V19: the innermost won, '{"a": {"b": 1}}' gave {"b": 1})
        if text[i] in "{[":
            try:
                val, end = dec.raw_decode(text, i)
            except (ValueError, RecursionError):
                i += 1
                continue
            if end > last_end:
                last, last_end = val, end
            i = end
            continue
        i += 1
    if last is _NONE:
        raise ValueError("no JSON value in the answer")
    return last


_NONE = object()


# ----------------------------------------------------------------------------------------------- multiple choice
_CHOICE = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")
_BOXED_LETTER_ONLY = re.compile(r"^[^A-Za-z]*([A-Z])[^A-Za-z]*$")
_BOXED_LETTER_LABEL = re.compile(r"^\s*([A-Z])\s*[:).\-]")
# M7 R2: a leading letter followed by anything but a letter or digit is the answer (NeMo-Gym ``^([a-zA-Z])(?![a-zA-Z0-9])``):
# "B)", "B.", "B, because A is wrong", "B (A is a distractor)", "B I am sure"; except an article "A" / "I" followed by a
# lowercase word ("A good choice is C", "I think B", M7 audit V19), read by the rules after it
_LEADING_LETTER = re.compile(r"^([a-zA-Z])(?![a-zA-Z0-9])")
_ARTICLE = re.compile(r"^[AI]\s+[a-z]")
_LAST_LETTER = re.compile(r"\b[A-Z]\b(?!.*\b[A-Z]\b)", re.DOTALL)
_LIST_SEP = re.compile(r"(?:\s*[/|&,;]\s*|\s+(?:or|and)\s+|\s+)", re.IGNORECASE)
_FOLD = str.maketrans({"أ": " A", "ب": " B", "ج": " C", "د": " D", "অ": " A", "ব": " B", "ড": " C", "ঢ": " D"})


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_FOLD).strip()  # NFKC folds fullwidth Ａ-Ｄ


def _norm_space(s: str) -> str:
    return " ".join(s.lower().split())


def _ambiguous(text: str, letters: set[str]) -> bool:
    hits = [(m.group(1).upper(), m.span(1)) for m in _CHOICE.finditer(text) if m.group(1).upper() in letters]
    return any(a != b and _LIST_SEP.fullmatch(text[sa[1]: sb[0]]) is not None for (a, sa), (b, sb) in itertools.pairwise(hits))


def choice_letter(payload: str, letters: set[str]) -> str | None:
    """One valid choice letter from an answer payload ("B", "B) because", "the answer is B"); None if ambiguous."""
    cand = _fold(payload).strip().strip("*_` \t\n\r.,;:")
    if _ambiguous(cand, letters):
        return None
    if len(cand) == 1:
        letter = cand
    elif (m := _LEADING_LETTER.match(cand)) and not _ARTICLE.match(cand):
        letter = m.group(1)
    else:
        m2 = _LAST_LETTER.search(cand)
        letter = m2.group(0) if m2 else None
    return letter.upper() if letter and letter.upper() in letters else None


def mcq_letter(text: str, options: dict[str, str], mode: str = "strict_boxed", prefix: str | None = None) -> str | None:
    """The chosen option letter. Modes (NeMo-Gym mcqa): ``strict_boxed`` (a letter in the last box), ``lenient_boxed``
    (also the text of exactly one option inside the box), ``answer_colon`` (``Answer: X`` or an option's text),
    ``answer_colon_md`` (the last markdown-tolerant ``Answer:`` / ``prefix`` payload, then the box)."""
    letters = {k.upper() for k, v in options.items() if v is not None and len(k) == 1 and k.isalpha()}

    def strict() -> str | None:
        inner = last_boxed(text)
        if inner is None:
            return None
        inner = strip_latex_text(inner.strip())
        m = _BOXED_LETTER_ONLY.match(inner) or _BOXED_LETTER_LABEL.match(inner)
        return m.group(1).upper() if m and m.group(1).upper() in letters else None

    def option_text(region: str | None, *, contains: bool) -> str | None:
        if region is None:
            return None
        cands = {_norm_space(region), _norm_space(strip_latex_text(region.strip()))}
        hit = {k.upper() for k, v in options.items() if v and k.upper() in letters
               for c in cands if (_norm_space(v) in c if contains else _norm_space(v) == c)}
        return next(iter(hit)) if len(hit) == 1 else None

    if mode == "strict_boxed":
        return strict()
    if mode == "lenient_boxed":
        return strict() or option_text(last_boxed(text), contains=True)
    if mode == "answer_colon":
        m = re.search(r"(?i)answer\s*:\s*(.+)", text)
        if not m:
            return None
        cand = strip_latex_text(m.group(1)).strip()
        if len(cand) == 1 and cand.upper() in letters:
            return cand.upper()
        return option_text(cand, contains=False)
    if mode == "answer_colon_md":
        for payload in reversed(answer_payloads(text, prefix)):
            if (got := choice_letter(payload, letters)) is not None:
                return got
        return strict() if prefix else None
    raise ValueError(f"unknown mcq mode {mode!r}")


# ----------------------------------------------------------------------------------------------- parser spec
@dataclass(frozen=True)
class ParserSpec:
    """How to read the answer. ``kind``: ``full`` (the whole text), ``boxed``, ``answer_colon``, ``xml`` (``tag``),
    ``code`` (``lang``), ``json``, ``mcq`` (``mode``, options from the task), ``regex`` (``pattern``, last match,
    group 1 if any; a list of patterns is tried in turn and the first that captures a non-empty answer wins, NeMo-Gym
    ``mcqa``; with options, a captured option text is its letter), ``last_line``. ``think``: ``strip`` (read after the last ``</think>``; an unclosed one is no
    answer), ``keep``. ``fallback_full``: a missing marker yields the whole text (verifiers' non-strict boxed)."""

    kind: str = "full"
    think: str = "strip"
    tag: str = "answer"
    lang: str | None = None
    pattern: str | tuple[str, ...] | None = None  # a list is kept as a tuple (the spec is frozen and hashable)
    mode: str = "strict_boxed"
    prefix: str | None = None
    fallback_full: bool = False

    KINDS = ("full", "boxed", "answer_colon", "xml", "code", "json", "mcq", "regex", "last_line")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError(f"unknown parser kind {self.kind!r} (known: {self.KINDS})")
        if self.think not in ("strip", "keep"):
            raise ValueError("think must be 'strip' or 'keep'")
        if isinstance(self.pattern, list):
            object.__setattr__(self, "pattern", tuple(self.pattern))
        if self.kind == "regex":
            if not self.pattern:
                raise ValueError("regex parser needs a pattern")
            for p in self.patterns():
                re.compile(p)

    def patterns(self) -> list[str]:
        return [self.pattern] if isinstance(self.pattern, str) else list(self.pattern or [])

    @classmethod
    def of(cls, v: ParserSpec | dict | str | None) -> ParserSpec:
        if isinstance(v, ParserSpec):
            return v
        if v is None:
            return cls()
        return cls(kind=v) if isinstance(v, str) else cls(**v)

    def parse(self, text: str | None, *, options: dict[str, str] | None = None) -> Any:
        """The extracted answer, or None when there is none (format failure)."""
        if text is None:
            return None
        body = after_think(text) if self.think == "strip" else text
        if body is None:
            return None
        got: Any
        if self.kind == "full":
            got = body.strip()
        elif self.kind == "boxed":
            got = last_boxed(body)
        elif self.kind == "answer_colon":
            ps = answer_payloads(body, self.prefix)
            got = ps[-1].strip() if ps else None
        elif self.kind == "xml":
            got = xml_tag(body, self.tag)
        elif self.kind == "code":
            got = code_block(body, self.lang)
        elif self.kind == "json":
            try:
                return json_value(body)
            except (ValueError, RecursionError):  # deeply nested input is no answer, not a grader crash (M7 audit V19)
                return None
        elif self.kind == "mcq":
            return mcq_letter(body, options or {}, self.mode, self.prefix)
        elif self.kind == "regex":
            got = None
            for pat in self.patterns():
                hits = list(re.finditer(pat, body, re.IGNORECASE | re.DOTALL))
                got = (hits[-1].group(1) if hits[-1].groups() else hits[-1].group(0)) if hits else None
                if got is not None and got.strip():
                    break
            if got is not None and options and got.strip().upper() not in {k.upper() for k in options}:
                hit = [k.upper() for k, v in options.items() if v and _norm_space(v) == _norm_space(got)]
                got = hit[0] if len(hit) == 1 else got
        else:  # last_line
            lines = [ln for ln in body.strip().splitlines() if ln.strip()]
            got = lines[-1].strip() if lines else None
        if got is None and self.fallback_full:
            got = body.strip()
        if isinstance(got, str):
            got = got.strip()
            return got or None
        return got


# ----------------------------------------------------------------------------------------------- checkers
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).strip().lower())


def _strip_math_delims(s: str) -> str:
    s = s.strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1].strip()
    return s


_NUM = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_INT = re.compile(r"^[+-]?\d+$")
# M7 R2: a comma is a thousands separator only in a grouped string ("1,234", "-12,345.5"); "1,2" or "2,16" is a list or a
# decimal comma, not a number (it read as 12 / 216: rLLM ``math_utils/utils.py`` keeps only grouped thousands)
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")


def _number_text(s: Any) -> str | None:
    """The text of a number as ``to_number`` reads it (delimiters, grouped thousands, ``\\%``, a trailing period), or None
    when a comma makes it not a number."""
    t = _strip_math_delims(str(s)).replace("\\%", "%").strip().rstrip(".")
    if "," in t:
        core = t.rstrip("%").strip()
        if not _THOUSANDS.match(core):
            return None
        t = t.replace(",", "")
    return t


def to_number(s: Any) -> float | None:
    """A finite float, or None (not a number; an integer too large for a float, which scored as infinity and matched any
    other: M7 R2)."""
    if isinstance(s, bool):
        return None
    try:
        if isinstance(s, int | float):
            v = float(s)
            return v if math.isfinite(v) else None
        t = _number_text(s)
        if t is None:
            return None
        pct = t.endswith("%")
        t = t.rstrip("%").strip()
        if m := re.fullmatch(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}", t):
            a, b = to_number(m.group(1)), to_number(m.group(2))
            return a / b if a is not None and b not in (None, 0) else None
        if m := re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", t):
            v = int(m.group(1)) / int(m.group(2)) if int(m.group(2)) else None
        elif not _NUM.match(t):
            return None
        else:
            v = float(t)
            v = v / 100 if pct else v
    except (OverflowError, ValueError):  # an integer too large for a float, or for int() itself (Python's digit limit)
        return None
    return v if v is not None and math.isfinite(v) else None


def to_integer(s: Any) -> int | None:
    """An integer-valued answer as an exact int (no fraction, no exponent, no percent), else None."""
    if isinstance(s, bool):
        return None
    if isinstance(s, int):
        return s
    t = _number_text(s)
    if t is None or not _INT.match(t):
        return None
    try:
        return int(t)
    except ValueError:  # beyond Python's int-string digit limit
        return None


def numbers_equal(a: Any, b: Any, *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """M7 R2: two integers compare exactly ("1000000001" is not "1000000000", which a relative tolerance accepted; the
    references all score it 0); a tolerance applies only when either side has a fraction or an exponent."""
    x, y = to_integer(a), to_integer(b)
    if x is not None and y is not None:
        return x == y
    p, q = to_number(a), to_number(b)
    return p is not None and q is not None and math.isclose(p, q, rel_tol=rel_tol, abs_tol=abs_tol)


def seq_match(answer: str, gold: str) -> float:
    return float(SequenceMatcher(None, normalize_text(answer), normalize_text(gold)).ratio())


def weighted_seq_match(answer: str, gold: str) -> float:
    """85% full-string similarity + 15% similarity on the reference's first 10 characters (NeMo-Gym)."""
    a, g = normalize_text(answer), normalize_text(gold)
    sim = SequenceMatcher(None, a, g).ratio()
    p = g[:10]
    pre = SequenceMatcher(None, a[: len(p)], p).ratio()
    return max(0.0, min(1.0, 0.85 * sim + 0.15 * pre))


def math_verify_available() -> bool:
    try:
        import math_verify  # noqa: F401
    except ImportError:
        return False
    return True


_STARTED = "started"
START_TIMEOUT_S = 120.0  # an interpreter that cannot start a child in this long is the node's problem (CheckerUnavailable)
_SLOTS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()


def _slots() -> asyncio.Semaphore:
    """One check per CPU at a time, per event loop (NeMo-Gym ``math_with_judge``'s semaphore shape)."""
    loop = asyncio.get_running_loop()
    sem = _SLOTS.get(loop)
    if sem is None:
        sem = _SLOTS[loop] = asyncio.Semaphore(max(1, os.cpu_count() or 1))
    return sem


def _math_child(gold: str, pred: str, conn: Any) -> None:
    try:
        from math_verify import parse, verify
    except ImportError as e:
        conn.send(("unavailable", f"{type(e).__name__}: {e}"))
        conn.close()
        return
    conn.send(_STARTED)  # the deadline starts now: the queue and the interpreter's start-up are not the answer's time
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            g = parse(f"\\boxed{{{_strip_math_delims(gold)}}}", parsing_timeout=None)
            p = parse(f"\\boxed{{{pred}}}", parsing_timeout=None)
            conn.send(float(verify(g, p, timeout_seconds=None)) if g and p else 0.0)
    except BaseException:  # noqa: BLE001 -- malformed math scores 0 (math-verify's TimeoutException is a BaseException)
        conn.send(0.0)
    finally:
        conn.close()


async def math_equal(pred: str, gold: str, *, timeout_s: float = 10.0) -> float:
    """Symbolic equivalence by math-verify in a child process with a deadline (NeMo-Gym ``math_with_judge``). Equal
    numbers short-circuit without SymPy (integers exactly, ``numbers_equal``). A child that overruns is killed and the
    answer scores 0 (a pathological expression is the policy's).

    M7 R2: math-verify missing raises ``CheckerUnavailable`` before any comparison (the numeric short-circuit let a task's
    control pass while its answers masked; a ``math`` criterion is refused at load without it). At most one check per
    CPU runs at a time, and the deadline starts when the child reports it has started (on macOS, 256 concurrent spawns
    charged the queue and the start-up to the answer: 30 of 256 correct answers scored). A child that dies without a
    verdict is the checker's fault (``CheckerUnavailable``: graded again, then decided by the control), never a 0."""
    if not math_verify_available():
        raise CheckerUnavailable("math-verify is not installed (pip install 'crucible[answers]')")
    if numbers_equal(pred, gold):
        return 1.0
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")  # fork in a threaded macOS process can crash
    async with _slots():
        recv, send = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_math_child, args=(gold, pred, send), daemon=True)
        proc.start()
        send.close()
        loop = asyncio.get_running_loop()
        began, deadline = loop.time(), None
        try:
            while True:
                got: Any = _NONE
                try:
                    if recv.poll():
                        got = recv.recv()
                except (EOFError, OSError):
                    raise CheckerUnavailable(f"the math-verify child exited without a verdict (exit code {proc.exitcode})") from None
                if got == _STARTED:
                    deadline = loop.time() + timeout_s
                    continue
                if isinstance(got, tuple):
                    raise CheckerUnavailable(f"math-verify cannot be imported in the child: {got[1]}"[:300])
                if isinstance(got, float):
                    return got
                if not proc.is_alive() and not recv.poll():
                    raise CheckerUnavailable(f"the math-verify child exited without a verdict (exit code {proc.exitcode})")
                now = loop.time()
                if deadline is None and now - began > START_TIMEOUT_S:
                    raise CheckerUnavailable(f"the math-verify child did not start within {START_TIMEOUT_S:g}s")
                if deadline is not None and now >= deadline:
                    proc.kill()
                    proc.join(1.0)
                    return 0.0
                await asyncio.sleep(0.02)
        finally:
            if proc.is_alive():
                proc.kill()
                proc.join(1.0)
            recv.close()


Judge = Callable[[dict[str, Any]], Awaitable[float | bool | None]]


@dataclass(frozen=True)
class CheckerSpec:
    """How to compare. ``kind``: ``exact`` (normalized text), ``exact_raw``, ``numeric`` (``rel_tol``, ``abs_tol``),
    ``math`` (symbolic, math-verify), ``mcq`` (letter equality), ``contains`` (normalized reference inside the
    answer), ``regex`` (the reference is a pattern the whole answer must match), ``seq_match``,
    ``weighted_seq_match``, ``json_subset`` (every reference key/value present), ``any_of`` (the reference is a list:
    exact match with one), ``judge`` (a named judge from the task's judges; errors mask). ``threshold``: a graded
    checker's score at or above it counts as correct for the outcome (the reward keeps the graded score)."""

    kind: str = "exact"
    rel_tol: float = 1e-6
    abs_tol: float = 1e-9
    timeout_s: float = 10.0
    judge: str | None = None
    threshold: float = 1.0
    max_chars: int = 8192  # M7 R2: a longer answer scores 0 before a judge is called (NeMo-Gym's cap): its error was a mask

    KINDS = ("exact", "exact_raw", "numeric", "math", "mcq", "contains", "regex", "seq_match", "weighted_seq_match", "json_subset", "any_of", "judge")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError(f"unknown checker kind {self.kind!r} (known: {self.KINDS})")
        if self.kind == "judge" and not self.judge:
            raise ValueError("judge checker needs a judge name")
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")

    @classmethod
    def of(cls, v: CheckerSpec | dict | str | None) -> CheckerSpec:
        if isinstance(v, CheckerSpec):
            return v
        if v is None:
            return cls()
        return cls(kind=v) if isinstance(v, str) else cls(**v)

    async def score(self, answer: Any, gold: Any, *, row: dict[str, Any] | None = None, judges: dict[str, Judge] | None = None) -> float:
        """A score in [0, 1]. ``answer`` is the parsed answer (never None here)."""
        k = self.kind
        if k == "exact":
            return float(normalize_text(str(answer)) == normalize_text(str(gold)))
        if k == "exact_raw":
            return float(str(answer) == str(gold))
        if k == "numeric":
            return float(numbers_equal(answer, gold, rel_tol=self.rel_tol, abs_tol=self.abs_tol))
        if k == "math":
            return await math_equal(str(answer), str(gold), timeout_s=self.timeout_s)
        if k == "mcq":
            return float(str(answer).strip().upper() == str(gold).strip().upper())
        if k == "contains":
            return float(normalize_text(str(gold)) in normalize_text(str(answer)))
        if k == "regex":
            return float(re.fullmatch(str(gold), str(answer).strip(), re.DOTALL) is not None)
        if k == "seq_match":
            return seq_match(str(answer), str(gold))
        if k == "weighted_seq_match":
            return weighted_seq_match(str(answer), str(gold))
        if k == "json_subset":
            return float(_subset(gold, answer))
        if k == "any_of":
            golds = gold if isinstance(gold, list) else [gold]
            return float(any(normalize_text(str(answer)) == normalize_text(str(g)) for g in golds))
        judge = (judges or {}).get(self.judge or "")
        if judge is None:
            raise CheckerUnavailable(f"judge {self.judge!r} is not configured")
        if len(str(answer)) > self.max_chars:
            return 0.0
        try:
            v = await judge({**(row or {}), "answer": answer, "gold": gold})
        except Exception as e:
            raise CheckerUnavailable(f"judge {self.judge!r} failed: {type(e).__name__}: {e}") from e
        if v is None:
            raise CheckerUnavailable(f"judge {self.judge!r} gave no verdict")
        try:
            x = float(v)
        except (TypeError, ValueError) as e:
            raise CheckerUnavailable(f"judge {self.judge!r} gave a non-numeric verdict {v!r}") from e
        if not math.isfinite(x):  # NaN scored 1.0 through max(0, min(1, nan)) (M7 audit V19)
            raise CheckerUnavailable(f"judge {self.judge!r} gave a non-finite verdict {v!r}")
        return max(0.0, min(1.0, x))


def _subset(gold: Any, got: Any) -> bool:
    if isinstance(gold, dict):
        return isinstance(got, dict) and all(k in got and _subset(v, got[k]) for k, v in gold.items())
    if isinstance(gold, list):
        return isinstance(got, list) and len(gold) == len(got) and all(_subset(a, b) for a, b in zip(gold, got, strict=True))
    if isinstance(gold, float) or isinstance(got, float):
        return numbers_equal(gold, got)
    return gold == got


def parse_judge_choice(content: str | None, choices: tuple[str, ...] = ("A", "B", "C")) -> str | None:
    """A judge model's verdict letter (verifiers v1): ``FINAL VERDICT: X`` lines win, then a boxed letter, then a
    ``VERDICT:``/``ANSWER:`` label, then the last standalone choice. An unclosed ``<think>`` is no verdict."""
    if not content or ("<think>" in content and "</think>" not in content):
        return None
    text = content.rsplit("</think>", 1)[-1].strip()
    up = text.upper()
    by_up = {c.upper(): c for c in choices}
    allowed = "|".join(re.escape(c) for c in by_up)
    choice_re = rf"(?<!\w)({allowed})(?!\w)"
    hs = r"[^\S\n]"
    finals = re.findall(rf"^{hs}*FINAL{hs}+(?:JUDGMENT|ANSWER|VERDICT){hs}*(?:IS{hs}*)?[:\-]?{hs}*{choice_re}", up, re.MULTILINE)
    if finals:
        return by_up.get(finals[-1])
    boxed = last_boxed(text)
    text = (boxed or "").strip() or text
    up = text.upper()
    verdict = re.search(r"(?:^|\n)\s*(?:FINAL\s+JUDGMENT|FINAL\s+ANSWER|FINAL\s+VERDICT|JUDGMENT|VERDICT|ANSWER)\s*(?:IS\s*)?[:\-]?\s*", up)
    if verdict:
        m = re.search(choice_re, up[verdict.end():])
        return by_up.get(m.group(1)) if m else None
    hits = re.findall(choice_re, up)
    return by_up.get(hits[-1]) if hits else None


# ----------------------------------------------------------------------------------------------- rubric
@dataclass(frozen=True)
class Criterion:
    """One weighted rubric line: its own parser (default: the task's), checker and reference (default: the task's)."""

    id: str
    checker: CheckerSpec
    weight: float = 1.0
    parser: ParserSpec | None = None
    gold: Any = None

    @classmethod
    def of(cls, v: Criterion | dict) -> Criterion:
        if isinstance(v, Criterion):
            return v
        d = dict(v)
        return cls(id=d["id"], checker=CheckerSpec.of(d.get("checker")), weight=float(d.get("weight", 1.0)),
                   parser=ParserSpec.of(d["parser"]) if d.get("parser") is not None else None, gold=d.get("gold"))


@dataclass
class AnswerScore:
    score: float  # weighted rubric score in [0, 1]
    correct: bool  # every criterion at or above its threshold
    parsed: Any
    formatted: bool  # the main parser found an answer
    per_criterion: dict[str, dict[str, Any]] = field(default_factory=dict)


def validate_criteria(criteria: list[Criterion]) -> None:
    ids = [c.id for c in criteria]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate criterion ids: {sorted({x for x in ids if ids.count(x) > 1})}")
    if any(not (math.isfinite(c.weight) and c.weight >= 0) for c in criteria) or (criteria and not any(c.weight > 0 for c in criteria)):
        raise ValueError(f"criterion weights must be finite, >= 0 and not all 0: {[(c.id, c.weight) for c in criteria]}")


async def score_answer(text: str | None, gold: Any, *, parser: ParserSpec, criteria: list[Criterion], options: dict[str, str] | None = None,
                       row: dict[str, Any] | None = None, judges: dict[str, Judge] | None = None) -> AnswerScore:
    """Parse once per distinct parser, score each criterion, weight. ``CheckerUnavailable`` propagates (mask).
    M6 audit (verifiers' rubric validation): weights must be finite and >= 0 with at least one > 0, and criterion ids
    unique (a negative weight returned -1.0; all-zero weights scored 0 with outcome 1; a duplicate id crashed every grade)."""
    validate_criteria(criteria)
    parsed = parser.parse(text, options=options)
    total_w = sum(c.weight for c in criteria) or 1.0
    per: dict[str, dict[str, Any]] = {}
    acc, correct = 0.0, True
    for c in criteria:
        p = parsed if c.parser is None else c.parser.parse(text, options=options)
        g = gold if c.gold is None else c.gold
        s = 0.0 if p is None else await c.checker.score(p, g, row=row, judges=judges)
        per[c.id] = {"score": round(s, 6), "weight": c.weight, "parsed": p if isinstance(p, str | int | float | bool) or p is None else json.dumps(p)[:500]}
        acc += c.weight * s
        correct = correct and s >= c.checker.threshold
    return AnswerScore(score=round(acc / total_w, 6), correct=correct, parsed=parsed, formatted=parsed is not None, per_criterion=per)
