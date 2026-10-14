"""Long-context task families, generated from a seed (M16 design §4).

Portions of this file are ported from research-environments (PrimeIntellect-ai/research-environments @ d4fb80a,
``environments/long_context``), licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
Ported: the Oolong synth answer parser and scorer (``oolong_synth/oolong_synth/taskset.py:40-78``), the MRCR v2 metric
(``mrcr_v2/mrcr_v2/taskset.py:91-104``) and the long-document Q&A statuses, exact-match path and document pass rate
(``longpdfs/longpdfs/longpdfs.py:73-80, 162-176, 244-277, 296-301``). Changes: the upstream tasks load fixed datasets
(``oolongbench/oolong-synth``, the ``mrcr_v2`` CSVs, the private ``PrimeIntellect/longpdfs``); here each family is
*generated* deterministically from a seed at any size, offline, and the long-document judge (an LLM) is replaced by a
normalised exact match so the grade is deterministic.

Three families, each at rungs of 8k to 4M tokens (``SIZES``; 4 characters per token, the size the generator aims at):

- ``oolong`` -- aggregation over a labelled record stream (Oolong synth with labels): counts, the most common label, a
  comparison of two labels, a count for one user. Numeric answers get partial credit 0.75^|error| (upstream rule).
- ``mrcr`` -- multi-round coreference retrieval (MRCR v2): a long chat transcript holds several identical requests ("write
  a poem about tapirs") with different replies; the answer is the n-th one, prefixed with a 12-character hash, scored by
  ``difflib`` similarity gated on the prefix (upstream metric).
- ``docqa`` -- long-document Q&A (longpdfs): several generated reports with many similar facts; each question names its
  document; the answer is a JSON object {question number: answer}; the reward is the share of documents whose questions
  are all right (upstream ``document_pass_rate``).

Everything a task checks is stated in its question (no hidden requirements, study 02 P1).
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import random
import re
import string
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

CHARS_PER_TOKEN = 4
SIZES = [8_000, 32_000, 128_000, 512_000, 1_000_000, 2_000_000, 4_000_000]
FAMILIES = ("oolong", "mrcr", "docqa")

_WORD_TEXT = ("account able across action after again against agent air all almost along already also always among amount and animal another "
              "answer any area around art ask away back bank base basis bear beat become bed before begin behind best better between big bird "
              "black blue board boat body book both bottom box bring brother build business buy call camp care carry case catch cause center "
              "certain chair chance change check child choose city class clear close cloud coast cold color come common company cool copy corn "
              "cost could count country course cover cross crowd current cut dark day deal deep design detail differ direct dog door double "
              "draw dream drive drop dry during each early earth east easy edge effect egg energy enough enter even event every exact example "
              "face fact fall family farm fast father favor feel field figure fill final find fine fire first fish five floor flow follow food "
              "force forest form forward free fresh friend front fruit full game garden gather general gentle glass gold good grass great green "
              "ground group grow guess guide half hand happy hard head heart heat heavy help high hill history hold home hope horse hour house "
              "huge idea inch include iron island join just keep kind king knowledge lake land large late lead learn leave level light line list "
              "little live long look lost low machine main major make map mark market master matter meet metal middle mind minute modern moment "
              "month morning mountain move music name nation natural near need never night noise north note number object ocean offer office "
              "open order other page paper part pass path pattern people period picture piece place plain plan plant play point port power "
              "present press price print problem process produce product program proper pull purpose quick quiet rain range reach ready reason "
              "record region remember report rest result river road rock room round rule safe sail salt same sand save scale school science "
              "season seat second section seed sense serve settle shape share ship shore short side sign silver simple single size skill sky "
              "small smile snow soft soil solve sound south space speak special speed spring square stand star start state station stay steel "
              "step still stone store story straight stream street strong study subject summer supply surface system table tall teach team "
              "test thick thin thought time together tool top total touch toward town track trade train travel tree true turn type unit until "
              "valley value village visit voice wait walk warm watch water wave weather week weight west wheel white whole wide wind window "
              "winter wish wonder wood word work world write year young")
WORDS = _WORD_TEXT.split()


def words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def sentence(rng: random.Random, lo: int = 8, hi: int = 18) -> str:
    s = words(rng, rng.randint(lo, hi))
    return s[0].upper() + s[1:] + "."


@dataclass
class LongCtxItem:
    family: str
    seed: int
    tokens: int  # the target size (CHARS_PER_TOKEN characters per token)
    context: str
    question: str  # what is asked, with the answer format and how it is scored
    gold: Any
    answer_format: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"longctx/{self.family}/s{self.seed}-t{self.tokens}"

    def score(self, answer: str | None) -> dict[str, Any]:
        return SCORERS[self.family](self, answer)

    def gold_answer(self) -> str:
        """The answer text that scores 1.0 (the oracle's submission)."""
        if self.family == "oolong":
            return str(parse_gold(self.gold))
        if self.family == "mrcr":
            return str(self.gold)
        return json.dumps({str(i + 1): q["answer"] for i, q in enumerate(self.gold)})


# ---- oolong: aggregation over a labelled record stream ---------------------------------------------------------------------

OOLONG_LABELS = ("location", "person", "number", "date", "entity", "abbreviation")


# Ported from research-environments oolong_synth/oolong_synth/taskset.py:40-48 (Apache-2.0)
def attempt_answer_parse(answer: str) -> str:
    """Extract the candidate answer from the agent's output."""
    if ":" not in answer:
        return answer
    candidate = answer.split(":")[-1].strip().replace("*", "").replace("[", "").replace("]", "")
    for phrase in ("more common", "less common", "same frequency"):
        if phrase in candidate:
            return phrase
    return candidate


# Ported from research-environments oolong_synth/oolong_synth/taskset.py:51-55 (Apache-2.0); dates are not generated here
def parse_gold(answer_raw: str):
    """Parse the serialized gold answer (a list literal)."""
    return ast.literal_eval(answer_raw)[0]


# Ported from research-environments oolong_synth/oolong_synth/taskset.py:58-78 (Apache-2.0); the DATE branch is dropped
def oolong_score(answer_raw: str, answer_type: str, output: str) -> float:
    gold = parse_gold(answer_raw)
    trimmed_output = attempt_answer_parse(output)
    if str(trimmed_output) == str(gold):
        return 1.0
    elif str(trimmed_output) in ["more common", "less common", "same frequency"]:
        if str(trimmed_output) in str(gold):
            return 1.0
    elif answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            return float(0.75 ** abs(int(gold) - int(trimmed_output)))
        except (TypeError, ValueError):  # upstream catches everything: an unparseable number scores 0
            return 0.0
    return 0.0


def gen_oolong(seed: int, tokens: int) -> LongCtxItem:
    rng = random.Random(f"oolong:{seed}:{tokens}")
    chars = tokens * CHARS_PER_TOKEN
    weights = [rng.uniform(0.5, 2.0) for _ in OOLONG_LABELS]
    users = [rng.randint(10_000, 99_999) for _ in range(max(8, tokens // 4_000))]
    lines, size = [], 0
    counts: dict[str, int] = dict.fromkeys(OOLONG_LABELS, 0)
    per_user: dict[tuple[int, str], int] = {}
    while size < chars:
        label = rng.choices(OOLONG_LABELS, weights)[0]
        user = rng.choice(users)
        day = f"{rng.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])} {rng.randint(1, 28):02d}, {rng.randint(2022, 2024)}"
        line = f"Date: {day} || User: {user} || Instance: {sentence(rng, 6, 14)} || Label: {label}"
        lines.append(line)
        size += len(line) + 1
        counts[label] += 1
        per_user[(user, label)] = per_user.get((user, label), 0) + 1
    kind = rng.choice(["count", "most_common", "compare", "user_count"])
    top2 = sorted(counts.values())[-2:]
    if kind == "most_common" and top2[0] == top2[1]:
        kind = "count"  # a tie has no single answer
    if kind == "count":
        label = rng.choice(OOLONG_LABELS)
        q, gold, at = f"How many data points have the label '{label}'?", f"[{counts[label]}]", "ANSWER_TYPE.NUMERIC"
    elif kind == "most_common":
        top = max(counts, key=lambda k: counts[k])
        q, gold, at = f"Which label is the most common? Answer with one of: {', '.join(OOLONG_LABELS)}.", f"[{top!r}]", ""
    elif kind == "compare":
        a, b = rng.sample(OOLONG_LABELS, 2)
        rel = "more common than" if counts[a] > counts[b] else "less common than" if counts[a] < counts[b] else "same frequency as"
        q = f"Is the label '{a}' more common, less common, or the same frequency as the label '{b}'? Answer with 'more common', 'less common' or 'same frequency'."
        gold, at = f"[{rel!r}]", ""
    else:
        user = rng.choice(users)
        label = rng.choice(OOLONG_LABELS)
        q, gold, at = f"How many data points of user {user} have the label '{label}'?", f"[{per_user.get((user, label), 0)}]", "ANSWER_TYPE.NUMERIC"
    fmt = ("Answer with the single number, label or phrase only. A numeric answer scores 0.75 to the power of its distance from the true "
           "count; any other answer must match exactly.")
    head = ("Each line of the context is one data point: its date, its user, an instance text and its label "
            "(one of " + ", ".join(OOLONG_LABELS) + ").")
    return LongCtxItem("oolong", seed, tokens, "\n".join(lines) + "\n", f"{head} {q}", gold, fmt, {"kind": kind, "answer_type": at, "records": len(lines)})


def _oolong(item: LongCtxItem, answer: str | None) -> dict[str, Any]:
    if not answer or not answer.strip():
        return {"reward": 0.0, "status": "no_answer"}
    r = oolong_score(item.gold, item.meta["answer_type"], answer.strip())
    return {"reward": r, "status": "correct" if r == 1.0 else "wrong_content"}


# ---- mrcr: multi-round coreference retrieval -----------------------------------------------------------------------------

MRCR_FORMATS = ("poem", "riddle", "short story", "limerick", "haiku", "product description", "letter", "toast")
MRCR_TOPICS = ("tapirs", "lighthouses", "glaciers", "copper kettles", "night trains", "paper boats", "salt marshes", "violins", "comets",
               "beekeeping", "sandstorms", "old maps", "harbor cranes", "orchards", "typewriters", "owls")
ORD = {1: "1st", 2: "2nd", 3: "3rd"}


# Ported from research-environments mrcr_v2/mrcr_v2/taskset.py:91-104 (Apache-2.0): the official MRCR v2 metric
def mrcr_score(target: str, prediction: str | None) -> float:
    target = target.strip()
    if not prediction or len(target) < 12:
        return 0.0
    prediction = prediction.strip()
    start = prediction.rfind(target[:12])
    if start == -1:
        return 0.0
    return difflib.SequenceMatcher(a=target[12:].strip(), b=prediction[start + 12:].strip()).ratio()


def gen_mrcr(seed: int, tokens: int, needles: int = 8) -> LongCtxItem:
    rng = random.Random(f"mrcr:{seed}:{tokens}")
    chars = tokens * CHARS_PER_TOKEN
    fmt, topic = rng.choice(MRCR_FORMATS), rng.choice(MRCR_TOPICS)
    ask = f"write a {fmt} about {topic}"
    # the needles sit at random character offsets of the transcript; every other turn is a distractor (another format or
    # topic, often the same format or the same topic)
    at = sorted(rng.uniform(0, chars) for _ in range(needles))
    turns: list[str] = []
    needle_replies: list[str] = []
    size = 0
    while size < chars or len(needle_replies) < needles:
        if len(needle_replies) < needles and at[len(needle_replies)] <= size:
            f2, t2 = fmt, topic
        else:
            f2, t2 = rng.choice(MRCR_FORMATS), rng.choice(MRCR_TOPICS)
            if (f2, t2) == (fmt, topic):
                t2 = next(t for t in MRCR_TOPICS if t != topic)
        body = " ".join(sentence(rng) for _ in range(rng.randint(2, 6)))
        reply = f"Here is a {f2} about {t2}: {body}"
        turns.append(f"User: {('Please ' if rng.random() < 0.3 else '')}{'write' if rng.random() < 0.8 else 'Write'} a {f2} about {t2}\n\nAssistant: {reply}\n")
        if (f2, t2) == (fmt, topic):
            needle_replies.append(reply)
        size += len(turns[-1])
    pick = rng.randint(1, len(needle_replies))
    hash12 = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(12))
    gold = hash12 + needle_replies[pick - 1]
    q = (f"The context is a conversation transcript. Prepend {hash12} to the {ORD.get(pick, f'{pick}th')} (1-indexed) reply of the "
         f"assistant to a request to {ask}. Do not include any other text in your answer.")
    fmt_s = (f"Your answer must begin with {hash12} followed immediately by the requested reply; it scores the text similarity "
             "(difflib ratio) of what follows the prefix to that reply, and 0 without the prefix.")
    return LongCtxItem("mrcr", seed, tokens, "".join(turns), q, gold, fmt_s, {"needles": len(needle_replies), "pick": pick, "ask": ask,
                                                                              "turns": len(turns)})


def _mrcr(item: LongCtxItem, answer: str | None) -> dict[str, Any]:
    if not answer or not answer.strip():
        return {"reward": 0.0, "status": "no_answer"}
    r = mrcr_score(str(item.gold), answer)
    return {"reward": r, "status": "correct" if r == 1.0 else "wrong_content", "exact_match": float(answer.strip() == str(item.gold).strip())}


# ---- docqa: long-document Q&A -----------------------------------------------------------------------------------------------

class Status(StrEnum):
    """Ported from research-environments longpdfs/longpdfs/longpdfs.py:73-80 (Apache-2.0)."""

    NO_ANSWER = "no_answer"
    WRONG_CONTENT = "wrong_content"
    CORRECT = "correct"


CITIES = ("Lund", "Tarbes", "Kestrel Bay", "Oulu", "Marbella", "Gdynia", "Arles", "Bergen", "Linz", "Porto", "Ghent", "Tromso", "Valence",
          "Ystad", "Brno", "Cork")
MATERIALS = ("copper", "zinc", "granite", "timber", "wool", "barley", "cement", "glass", "nickel", "cotton")
DOC_KINDS = ("annual report", "audit memo", "plant survey", "shipping ledger", "board minutes")


def _norm_answer(s: Any) -> str:
    return re.sub(r"[\s,]+", "", str(s).strip().lower()).rstrip(".")


def gen_docqa(seed: int, tokens: int, docs: int | None = None, questions_per_doc: int = 2) -> LongCtxItem:
    rng = random.Random(f"docqa:{seed}:{tokens}")
    chars = tokens * CHARS_PER_TOKEN
    n_docs = docs or max(2, min(10, tokens // 4_000))
    per_doc = chars // n_docs
    parts, gold = [], []
    for d in range(n_docs):
        name = f"{rng.choice(DOC_KINDS).replace(' ', '_')}_{d + 1:02d}.txt"
        facts: dict[tuple[str, str, int], int] = {}
        body, size, sec = [], 0, 0
        while size < per_doc:
            sec += 1
            city, mat, year = rng.choice(CITIES), rng.choice(MATERIALS), rng.randint(2015, 2025)
            val = rng.randint(100, 99_999)
            key = (city, mat, year)
            if key in facts:
                val = facts[key]  # a fact restated is restated with the same value (the documents never contradict themselves)
            facts[key] = val
            para = (f"Section {sec}. {sentence(rng)} The {city} facility processed {val:,} tonnes of {mat} in {year}. "
                    + " ".join(sentence(rng) for _ in range(rng.randint(2, 6))))
            body.append(para)
            size += len(para) + 2
        text = f"=== Document: {name} ===\n" + "\n\n".join(body) + "\n"
        parts.append(text)
        for key in rng.sample(sorted(facts), min(questions_per_doc, len(facts))):
            city, mat, year = key
            gold.append({"doc": name, "question": f"How many tonnes of {mat} did the {city} facility process in {year}?",
                         "answer": f"{facts[key]}"})
    qs = "\n".join(f"{i + 1}. [Document: {g['doc']}] {g['question']}" for i, g in enumerate(gold))
    q = ("The context holds several documents, each starting with a line '=== Document: <name> ==='. Answer every question about "
         f"the named document:\n{qs}")
    fmt = ("Answer with a JSON object mapping each question number (from 1) to its answer, like {\"1\": \"4312\", \"2\": \"87\"}. A "
           "numeric answer is compared without spaces or thousands separators. The score is the share of documents whose questions are "
           "all answered correctly.")
    return LongCtxItem("docqa", seed, tokens, "".join(parts), q, gold, fmt, {"docs": n_docs, "questions": len(gold)})


# Ported from research-environments longpdfs/longpdfs/longpdfs.py:162-176 (Apache-2.0)
def _document_passes(entries: list[dict], statuses: list[str]) -> list[bool]:
    passes: list[bool] = []
    offset = 0
    for entry in entries:
        qa_count = len(entry["qa_pairs"])
        doc_statuses = statuses[offset: offset + qa_count]
        passes.append(bool(doc_statuses) and all(s == Status.CORRECT for s in doc_statuses))
        offset += qa_count
    return passes


def _docqa(item: LongCtxItem, answer: str | None) -> dict[str, Any]:
    try:
        model_answers = json.loads(answer or "")
    except ValueError:
        model_answers = None
    statuses: list[str] = []
    for i, g in enumerate(item.gold):
        resp = model_answers.get(str(i + 1)) if isinstance(model_answers, dict) else None
        # longpdfs.py:296-301: a missing or blank answer is no_answer; the exact match (normalised here) is correct
        if resp is None or not str(resp).strip():
            statuses.append(Status.NO_ANSWER)
        else:
            statuses.append(Status.CORRECT if _norm_answer(resp) == _norm_answer(g["answer"]) else Status.WRONG_CONTENT)
    entries: list[dict] = []
    for g in item.gold:
        if not entries or entries[-1]["filename"] != g["doc"]:
            entries.append({"filename": g["doc"], "qa_pairs": []})
        entries[-1]["qa_pairs"].append(g)
    passes = _document_passes(entries, statuses)
    judged = [s for s in statuses if s in (Status.WRONG_CONTENT, Status.CORRECT)]
    return {"reward": sum(passes) / len(passes) if passes else 0.0,  # longpdfs.py:244-250 document_pass_rate
            "full_pass": float(bool(statuses) and all(s == Status.CORRECT for s in statuses)),  # :252-257
            "answer_rate": sum(s != Status.NO_ANSWER for s in statuses) / len(statuses) if statuses else 0.0,  # :258-266
            "question_accuracy": sum(s == Status.CORRECT for s in judged) / len(judged) if judged else 0.0,  # :268-277
            "status": [str(s) for s in statuses]}


GENERATORS = {"oolong": gen_oolong, "mrcr": gen_mrcr, "docqa": gen_docqa}
SCORERS = {"oolong": _oolong, "mrcr": _mrcr, "docqa": _docqa}


def generate(family: str, seed: int, tokens: int) -> LongCtxItem:
    if family not in GENERATORS:
        raise ValueError(f"family: {' | '.join(FAMILIES)} (got {family!r})")
    return GENERATORS[family](seed, tokens)


def digest(item: LongCtxItem) -> str:
    return hashlib.sha256(item.context.encode()).hexdigest()[:16]
