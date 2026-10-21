"""M1 item 2: single-turn tasks (``runtime/single_turn.py``) with answer parsers and checkers (``verify/answers.py``).

Parser and checker cases follow the frameworks they port (NeMo-Gym mcqa / equivalence_rule / math_with_judge, verifiers
v1 score utilities). End to end, single-turn rows in NeMo-Gym, verifiers and plain shapes run through the native harness
like any World task: gold scores 1, wrong 0, an unparsable reply fails the format gate, a long reasoning reply keeps its
answer past the trace cut, and a judge outage masks instead of scoring 0.
"""

from __future__ import annotations

import json

import pytest

from crucible.actors.llm import LLMResponse
from crucible.actors.policy import EpisodeLimits
from crucible.runtime.harness.native import NativeHarness
from crucible.runtime.inject import FakeSpend
from crucible.runtime.run import Runner, RunSpec
from crucible.runtime.single_turn import SingleTurnTask, answer_policy, load_tasks, single_turn_policy, task_from_row
from crucible.verify.answers import (
    CheckerSpec,
    CheckerUnavailable,
    ParserSpec,
    after_think,
    last_boxed,
    math_verify_available,
    mcq_letter,
    parse_judge_choice,
    to_number,
    weighted_seq_match,
)

OPTS = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"}


# ----------------------------------------------------------------------------------------------- parsers
def test_boxed_and_think():
    assert last_boxed(r"first \boxed{1} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{unclosed") is None
    assert after_think("<think>still thinking") is None  # an unclosed think is no answer
    assert after_think("<think>x</think> y </think> z") == " z"
    p = ParserSpec(kind="boxed")
    assert p.parse(r"<think>\boxed{9}</think> so \boxed{42}") == "42"
    assert p.parse(r"<think>\boxed{9}") is None
    assert ParserSpec(kind="boxed", think="keep").parse(r"<think>\boxed{9}") == "9"
    assert ParserSpec(kind="boxed", fallback_full=True).parse("42") == "42"


def test_other_parsers():
    assert ParserSpec(kind="answer_colon").parse("work\n**Answer:** 17\nmore\nAnswer: 18") == "18"
    assert ParserSpec(kind="xml", tag="answer").parse("<answer>a</answer> <answer> b </answer>") == "b"
    assert ParserSpec(kind="code", lang="python").parse("```js\nx\n```\n```python\nprint(1)\n```") == "print(1)"
    assert ParserSpec(kind="json").parse('text {"a": 1} and ```json\n{"b": [1, 2]}\n```') == {"b": [1, 2]}
    assert ParserSpec(kind="json").parse("no json here") is None
    assert ParserSpec(kind="regex", pattern=r"final=(\d+)").parse("final=1 ... final=2") == "2"
    assert ParserSpec(kind="last_line").parse("a\n\n  b  \n") == "b"
    with pytest.raises(ValueError, match="unknown parser"):
        ParserSpec(kind="nope")
    with pytest.raises(ValueError, match="pattern"):
        ParserSpec(kind="regex")


def test_mcq_modes():
    assert mcq_letter(r"I think \boxed{B}", OPTS) == "B"
    assert mcq_letter(r"\boxed{\text{C}}", OPTS) == "C"
    assert mcq_letter(r"\boxed{D: Madrid}", OPTS) == "D"
    assert mcq_letter(r"\boxed{E}", OPTS) is None  # not an option
    assert mcq_letter(r"\boxed{Paris}", OPTS) is None and mcq_letter(r"\boxed{Paris}", OPTS, "lenient_boxed") == "A"
    assert mcq_letter(r"\boxed{Paris or London}", OPTS, "lenient_boxed") is None  # two options matched: ambiguous
    assert mcq_letter("Answer: berlin", OPTS, "answer_colon") == "C"
    assert mcq_letter("**Answer:** B because", OPTS, "answer_colon_md") == "B"
    assert mcq_letter("Answer: A or B", OPTS, "answer_colon_md") is None  # a choice list is no choice
    assert mcq_letter("Answer: Ｃ", OPTS, "answer_colon_md") == "C"  # fullwidth folded (NFKC)
    assert mcq_letter("الإجابة:\nAnswer: ب", OPTS, "answer_colon_md") == "B"  # Arabic letter form folded


def test_judge_choice():
    assert parse_judge_choice("reasoning...\nFINAL VERDICT: B") == "B"
    assert parse_judge_choice(r"so \boxed{C}") == "C"
    assert parse_judge_choice("<think>A") is None
    assert parse_judge_choice("Verdict: a") == "A"


# ----------------------------------------------------------------------------------------------- checkers
async def test_checkers():
    assert to_number("1,234.5") == 1234.5 and to_number(r"\frac{1}{4}") == 0.25 and to_number("50%") == 0.5 and to_number("3/4") == 0.75
    assert to_number("$12$") == 12 and to_number("abc") is None
    assert await CheckerSpec("numeric").score("0.333", "1/3") == 0.0  # default rel_tol 1e-6
    assert await CheckerSpec("numeric", rel_tol=1e-2).score("0.333", "1/3") == 1.0
    assert await CheckerSpec("numeric").score("0.3333333", "1/3") == 1.0
    assert await CheckerSpec("exact").score("  Paris ", "paris") == 1.0
    assert await CheckerSpec("exact_raw").score("Paris", "paris") == 0.0
    assert await CheckerSpec("contains").score("The capital is Paris.", "paris") == 1.0
    assert await CheckerSpec("regex").score("2024-01-05", r"\d{4}-\d{2}-\d{2}") == 1.0
    assert await CheckerSpec("any_of").score("NYC", ["New York", "nyc"]) == 1.0
    assert await CheckerSpec("json_subset").score({"a": 1, "b": {"c": 2.0, "x": 0}}, {"b": {"c": 2}}) == 1.0
    assert await CheckerSpec("json_subset").score({"a": 1}, {"a": 2}) == 0.0
    assert 0.9 < weighted_seq_match("the cat sat", "the cat sat.") <= 1.0
    assert await CheckerSpec("mcq").score("b", "B") == 1.0
    if math_verify_available():
        assert await CheckerSpec("math").score("0.5", r"\frac{1}{2}") == 1.0  # equal numbers need no SymPy
    else:  # M7 R2 (D3): no answer passes while others mask (the control would pass); a math task is refused at load
        with pytest.raises(CheckerUnavailable):
            await CheckerSpec("math").score("0.5", r"\frac{1}{2}")
    if math_verify_available():
        assert await CheckerSpec("math").score(r"\sqrt{2}/2", r"\frac{1}{\sqrt{2}}") == 1.0
        assert await CheckerSpec("math").score("x+1", "x+2") == 0.0
    else:
        with pytest.raises(CheckerUnavailable):
            await CheckerSpec("math").score(r"\sqrt{2}/2", r"\frac{1}{\sqrt{2}}")
    with pytest.raises(ValueError, match="threshold"):
        CheckerSpec("seq_match", threshold=0)


# ----------------------------------------------------------------------------------------------- tasks
NEMO_ROWS = [
    {"responses_create_params": {"input": [{"role": "system", "content": "Answer with a boxed letter."},
                                           {"role": "user", "content": "Capital of France?"}]},
     "options": [{"A": "Paris"}, {"B": "London"}, {"C": "Berlin"}, {"D": "Madrid"}], "expected_answer": "A", "uuid": "q1"},
    {"responses_create_params": {"input": "What is 6*7?"}, "expected_answer": "42", "uuid": "q2", "parser": "boxed", "criteria": [{"id": "c", "checker": "numeric"}]},
]
NEMO_MODE = "strict_single_letter_boxed"  # the benchmark config's mode for these rows ("Answer with a boxed letter."; M7 R2: required)
VF_ROWS = [{"question": "Name the largest planet.", "answer": "Jupiter", "info": {"source": "toy"}},
           {"prompt": [{"role": "user", "content": "Spell cat backwards"}], "answer": "tac"}]


def test_row_shapes():
    t1, t2 = load_tasks(NEMO_ROWS, dataset="nemo", grading_mode=NEMO_MODE)
    assert t1.parser.kind == "mcq" and t1.options == OPTS and t1.system == "Answer with a boxed letter." and t1.instruction == "Capital of France?"
    assert t1.criteria[0].checker.kind == "mcq" and t1.id == "nemo/q1@1"
    assert t2.parser.kind == "boxed" and t2.criteria[0].checker.kind == "numeric"
    a, b = load_tasks(VF_ROWS, dataset="vf")
    assert a.instruction == "Name the largest planet." and a.meta == {"source": "toy"} and b.instruction == "Spell cat backwards"
    with pytest.raises(ValueError, match="needs a prompt"):
        task_from_row({"answer": "x"}, dataset="d", index=0)
    with pytest.raises(ValueError, match="end with a user message"):
        task_from_row({"prompt": [{"role": "assistant", "content": "hi"}], "answer": "x"}, dataset="d", index=0)
    with pytest.raises(ValueError, match="judge"):
        SingleTurnTask("t/x@1", "q", "a", criteria=[{"id": "j", "checker": {"kind": "judge", "judge": "gpt"}}])
    with pytest.raises(ValueError, match="duplicate"):
        load_tasks([NEMO_ROWS[0], NEMO_ROWS[0]], dataset="nemo", grading_mode=NEMO_MODE)


async def _run(tmp_path, tasks, pol, rid="st"):
    await Runner(tmp_path / "runs").run(RunSpec(run_id=rid, tasks=tasks, policies=[pol], harnesses=[NativeHarness("native:single", limits=EpisodeLimits(max_steps=1))], k=1))
    recs = {}
    for f in (tmp_path / "runs" / rid / "samples").glob("*/a1/record.json"):
        r = json.loads(f.read_text())
        recs[r["task"]] = r
    return recs


GOLD = {"nemo/q1@1": r"Paris is the capital. \boxed{A}", "nemo/q2@1": r"6*7 = \boxed{42}"}


async def test_end_to_end_gold_wrong_unformatted(tmp_path):
    tasks = load_tasks(NEMO_ROWS, dataset="nemo", grading_mode=NEMO_MODE)
    recs = await _run(tmp_path, tasks, answer_policy(lambda t: GOLD[t.id]), "gold")
    assert {k: r["reward"] for k, r in recs.items()} == {"nemo/q1@1": 1.0, "nemo/q2@1": 1.0}
    recs = await _run(tmp_path, tasks, answer_policy(lambda t: r"\boxed{B}" if "q1" in t.id else r"\boxed{41}"), "wrong")
    assert all(r["reward"] == 0.0 and r["failure"] is None for r in recs.values())
    recs = await _run(tmp_path, tasks, answer_policy(lambda t: "I am not sure."), "unformatted")
    for r in recs.values():
        assert r["reward"] == 0.0 and r["meta"]["gates"]["format"] == 0


async def test_long_reply_keeps_its_answer(tmp_path):
    t = load_tasks(NEMO_ROWS[1:], dataset="nemo")
    reasoning = "<think>" + "step. " * 2000 + "</think>" + r" The product is \boxed{42}."
    recs = await _run(tmp_path, t, answer_policy(lambda _t: reasoning), "long")
    assert next(iter(recs.values()))["reward"] == 1.0  # the answer sits past the 2000-character trace cut


async def test_weighted_rubric_and_threshold():
    t = SingleTurnTask("t/rubric@1", "Summarize.", "the cat sat on the mat", parser="full",
                       criteria=[{"id": "close", "checker": {"kind": "seq_match", "threshold": 0.8}, "weight": 3},
                                 {"id": "mentions_mat", "checker": "contains", "gold": "mat", "weight": 1}])
    rep = await t.grade_text("the cat sat on a mat")
    assert 0.8 < rep.reward < 1.0 and rep.outcome == 1.0 and rep.gates == {"format": "pass", "answer": "pass"}
    rep = await t.grade_text("a dog")
    assert rep.reward < 0.5 and rep.outcome == 0.0 and rep.gates["answer"] == "fail"


async def test_judge_outage_masks(tmp_path):
    async def flaky(row):
        raise ConnectionError("judge endpoint down")

    async def none(row):
        return None

    async def fair(row):
        return 1.0 if "jupiter" in row["answer"].lower() else 0.0

    for judge, masked in ((flaky, True), (none, True), (fair, False)):
        t = SingleTurnTask("t/judge@1", "Largest planet?", "Jupiter", criteria=[{"id": "j", "checker": {"kind": "judge", "judge": "j1"}}], judges={"j1": judge})
        rep = await t.grade_text("It is Jupiter.")
        assert (rep.infra["status"] == "masked") is masked
        if not masked:
            assert rep.reward == 1.0
    t = SingleTurnTask("t/judge@1", "Largest planet?", "Jupiter", criteria=[{"id": "j", "checker": {"kind": "judge", "judge": "j1"}}], judges={"j1": flaky})
    recs = await _run(tmp_path, [t], answer_policy(lambda _t: "Jupiter"), "judge")
    r = next(iter(recs.values()))
    assert r["mask_sample"] and r["failure"]["kind"] in ("check_infra", "verifier_error"), r["failure"]


class EchoLLM:
    """Records what the policy sends and answers with a fixed reply."""

    model = "echo"

    def __init__(self, reply: str):
        self.reply, self.seen, self.spend = reply, [], FakeSpend()

    async def complete(self, messages, **kw):
        self.seen.append({"messages": [dict(m) for m in messages], "tools": kw.get("tools")})
        return LLMResponse(text=self.reply, usage={"prompt_tokens": 10, "completion_tokens": 5}, model=self.model)


async def test_llm_policy_sees_system_fewshot_and_no_tools(tmp_path):
    row = {"prompt": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": r"\boxed{4}"},
                      {"role": "user", "content": "3+3?"}], "answer": "6", "parser": "boxed", "criteria": [{"id": "c", "checker": "numeric"}], "id": "fs"}
    llm = EchoLLM(r"\boxed{6}")
    recs = await _run(tmp_path, load_tasks([row], dataset="fewshot"), single_turn_policy(llm), "fs")
    assert next(iter(recs.values()))["reward"] == 1.0
    (call,) = llm.seen  # exactly one completion
    assert [m["role"] for m in call["messages"]] == ["system", "user", "assistant", "user"] and call["messages"][-1]["content"] == "3+3?"
    assert call["messages"][0]["content"] == "Be terse." and not call["tools"]


async def test_blank_reply_is_the_answer_no_nudge(tmp_path):
    llm = EchoLLM("")
    recs = await _run(tmp_path, load_tasks(VF_ROWS[:1], dataset="vf"), single_turn_policy(llm), "blank")
    assert len(llm.seen) == 1 and next(iter(recs.values()))["reward"] == 0.0
