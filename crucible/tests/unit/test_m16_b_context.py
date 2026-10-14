"""M16 owner B: ``actors/rlm/context.py`` (design §3.2; stress B1, B2, B8, B10)."""

from __future__ import annotations

import json
import math
import random

import pytest

from crucible.actors.llm import CONTEXT_OVERFLOW_MARKERS, BudgetExceeded, ContextOverflow, LLMError, LLMResponse
from crucible.actors.policy import approx_tokens
from crucible.actors.rlm.context import (
    SUMMARY_PREFIX,
    SUMMARY_PROMPT,
    UPDATE_PROMPT,
    Compactor,
    ContextMeter,
    ContextSpec,
    boundary_marker,
    footer,
    hollow_middle,
    is_overflow,
    is_silent_overflow,
    python_state_row,
    rebuild,
    split_tail,
    summary_request,
    summary_view,
    wrap_up_notice,
)


# ---- helpers ----------------------------------------------------------------------------------------------------------
def asst(i: int, code: str = "x = 1", n_calls: int = 1) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}_{k}", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}} for k in range(n_calls)]}


def tool(i: int, out: str, k: int = 0) -> dict:
    return {"role": "tool", "tool_call_id": f"c{i}_{k}", "content": out}


def convo(turns: int, out_chars: int = 400, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "TASK: refund order 17 and email the customer."}]
    for i in range(turns):
        msgs.append(asst(i, code=f"r = await world.call('orders.get', id={rng.randint(1, 10**6)})\nprint(r)"))
        msgs.append(tool(i, "".join(rng.choice("0123456789abcdef{}\":, ") for _ in range(out_chars))))
    return msgs


def pairing_ok(msgs: list[dict]) -> bool:
    """Every tool row follows its calling assistant row; every tool call of an assistant row is answered."""
    called: set = set()
    answered = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"}
    for m in msgs:
        if m.get("role") == "assistant":
            ids = {tc["id"] for tc in m.get("tool_calls") or []}
            if not ids <= answered:
                return False
            called |= ids
        elif m.get("role") == "tool" and m.get("tool_call_id") not in called:
            return False
    return True


def random_messages(rng: random.Random) -> list[dict]:
    """A random but well-formed history: tool results follow their call, sometimes with a notice in between."""
    msgs: list[dict] = [{"role": "system", "content": "s"}, {"role": "user", "content": "task " * rng.randint(1, 40)}]
    for i in range(rng.randint(0, 30)):
        r = rng.random()
        if r < 0.15:
            msgs.append({"role": "user", "content": "[inbox] " + "m" * rng.randint(1, 600)})
            continue
        if r < 0.3:
            msgs.append({"role": "assistant", "content": "thinking " * rng.randint(1, 80)})
            continue
        n = rng.randint(1, 3)
        msgs.append(asst(i, code="c" * rng.randint(1, 500), n_calls=n))
        for k in range(n):
            if rng.random() < 0.2:
                msgs.append({"role": "user", "content": "[notice] between results"})
            msgs.append(tool(i, "o" * rng.randint(0, 3000), k))
    return msgs


class FakeLLM:
    """Scripted ``complete``: each item is an ``LLMResponse`` or an exception to raise. Records every request."""

    def __init__(self, script: list):
        self.script = list(script)
        self.requests: list[dict] = []

    async def complete(self, messages, *, tools=None, max_tokens=800, temperature=0.0, sample_key=None, seed=0, **kw):
        self.requests.append({"messages": messages, "tools": tools, "max_tokens": max_tokens, "temperature": temperature, "sample_key": sample_key, "seed": seed})
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item


def ok_reply(text: str = "1. Goal: refund order 17.\n9. Variables: orders (list).", prompt_tokens: int = 1000) -> LLMResponse:
    return LLMResponse(text=text, usage={"prompt_tokens": prompt_tokens, "completion_tokens": 50}, raw_finish="stop")


class Http400(Exception):
    def __init__(self, msg: str, status: int = 400, body: object = None):
        super().__init__(msg)
        self.status_code = status
        self.body = body


# ---- ContextSpec.threshold ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("w", "max_out", "seq_len", "expected"), [
    (128_000, 2_000, None, 109_616),          # W − max_out − reserve
    (1_000_000, 2_000, None, 950_000),        # 0.95·W binds on a large window
    (32_768, 2_000, None, 16_384),            # the W//2 floor (W − max_out − reserve = 14,384)
    (131_072, 8_192, 131_072, 106_496),       # seq_len − max_out does not bind
    (131_072, 8_192, 65_536, 57_344),         # seq_len − max_out binds (no silent trainer tail cut)
    (4_096, 3_000, None, 1_096),              # the floor never lets a reply overrun the window
    (200_000, 32_000, None, 151_616),
])
def test_threshold_table(w, max_out, seq_len, expected):
    assert ContextSpec(window_tokens=w, max_out=max_out, seq_len=seq_len).threshold() == expected


def test_threshold_invariants_and_room_for_the_summary_call():
    rng = random.Random(7)
    for _ in range(500):
        w = rng.randint(2_000, 2_000_000)
        max_out = rng.randint(1, w // 2)
        seq = rng.choice([None, rng.randint(max_out + 1, 2 * w)])
        s = ContextSpec(window_tokens=w, max_out=max_out, seq_len=seq)
        t = s.threshold()
        assert 1 <= t <= w - max_out
        if seq is not None:
            assert t <= seq - max_out
        # the summary call fits: its input budget plus its output never exceed the window (nor seq_len)
        assert s.summary_input_tokens() + s.summary_tokens() <= w
        if seq is not None:
            assert s.summary_input_tokens() + s.summary_tokens() <= max(seq, s.summary_tokens() + 1)
    big = ContextSpec(window_tokens=128_000)
    assert big.summary_tokens() == 2_000 and big.summary_input_tokens() + big.summary_tokens() + big.reserve == 128_000
    assert ContextSpec(window_tokens=1_000_000).summary_tokens() == 15_625
    assert ContextSpec(window_tokens=4_000_000).summary_tokens() == 16_000


@pytest.mark.parametrize("kw", [{"max_out": 0}, {"max_out": 200_000}, {"seq_len": 1_000}, {"mode": "compact"}, {"max_compaction_attempts": 0}])
def test_spec_rejects_nonsense(kw):
    with pytest.raises(ValueError):
        ContextSpec(**kw)


# ---- ContextMeter ---------------------------------------------------------------------------------------------------------
def test_meter_anchor_suffix_reset_and_source():
    m = ContextMeter()
    msgs = convo(3)
    assert m.tokens(msgs) == approx_tokens(msgs) and m.source == "estimate"
    m.anchor({"prompt_tokens": 5_000, "completion_tokens": 100}, len(msgs))
    assert m.tokens(msgs) == 5_100 and m.source == "usage"
    extra = [tool(9, "z" * 400)]
    assert m.tokens(msgs + extra) == 5_100 + math.ceil(approx_tokens(extra) * m.ratio)
    # a shorter list than the anchor covers (the context was replaced without reset): an estimate, never the stale anchor
    assert m.tokens(msgs[:2]) < 5_000 and m.source == "estimate"
    m.reset()
    small = msgs[:2]
    assert m.tokens(small) < 1_000 and m.source == "estimate"  # pre-compaction usage never re-triggers
    m.anchor({"blank": 1}, 2)  # tolerant mode's blank usage leaves the meter alone
    assert m.source == "estimate" and m.tokens(small) < 1_000


def test_meter_tokenizer_counts_the_suffix_exactly():
    m = ContextMeter(tokenizer=lambda s: len(s.split()), per_message=3)
    msgs = [{"role": "user", "content": "a b c"}]
    m.anchor({"prompt_tokens": 10, "completion_tokens": 2}, 1)
    assert m.tokens([*msgs, {"role": "tool", "content": "one two three four"}]) == 12 + 4 + 3


def test_meter_state_dict_round_trip():
    m = ContextMeter()
    msgs = convo(5)
    m.anchor({"prompt_tokens": 3_000, "completion_tokens": 30}, len(msgs))
    m.tokens(msgs)
    m2 = ContextMeter()
    m2.load_state_dict(json.loads(json.dumps(m.state_dict())))
    more = [*msgs, tool(99, "q" * 999)]
    assert m2.tokens(more) == m.tokens(more) and m2.source == m.source == "usage" and m2.ratio == m.ratio


def test_meter_bounds_real_prompts_at_1_42x_where_chars4_overflows():
    """Stress B1: RLM cells tokenize at 1.42 real tokens per chars/4 token. With the meter, no prompt sent ever exceeds
    the threshold; with the old chars/4 trigger the same run sends prompts past the window."""
    def real(msgs: list[dict]) -> int:
        return sum(round(1.42 * approx_tokens([x])) for x in msgs)

    spec = ContextSpec(window_tokens=32_768, max_out=2_000, tail_tokens=3_000)
    thr = spec.threshold()

    def run(use_meter: bool, thr: int = thr) -> tuple[int, int]:
        rng = random.Random(11)
        meter = ContextMeter()
        msgs = convo(0)
        worst, compactions = 0, 0
        for turn in range(600):
            size = meter.tokens(msgs) if use_meter else approx_tokens(msgs)
            if size >= thr:
                _head, tail = split_tail(msgs[2:], int(spec.tail_tokens / max(1.0, meter.ratio)))
                msgs = rebuild(msgs[0], msgs[1], "summary " * 300, "[python-state] x: int", boundary_marker(compactions + 1, size, 0, compactions + 1), tail)
                meter.reset()
                compactions += 1
                assert (meter.tokens(msgs) if use_meter else approx_tokens(msgs)) < thr  # a rebuilt context never re-triggers at once
                continue
            sent = real(msgs)
            worst = max(worst, sent)
            reply = asst(turn, code=f"rows = [r for r in OBS if r['id'] in {[rng.randint(0, 10**9) for _ in range(rng.randint(5, 40))]}]")
            msgs.append(reply)
            meter.anchor({"prompt_tokens": sent, "completion_tokens": real([reply])}, len(msgs))
            msgs.append(tool(turn, json.dumps([{"id": rng.randint(0, 10**9), "amt": rng.random()} for _ in range(rng.randint(5, 120))])))
        return worst, compactions

    worst, n = run(True)
    assert n >= 5
    assert worst <= thr + 8, (worst, thr)  # rounding slack only
    naive_worst, _ = run(False)
    assert naive_worst > 1.35 * thr  # the defect the meter closes
    old_worst, _ = run(False, int(0.8 * spec.window_tokens))  # m14's trigger: 0.8 x budget on chars/4
    assert old_worst > spec.window_tokens


# ---- overflow detection ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("marker", CONTEXT_OVERFLOW_MARKERS)
@pytest.mark.parametrize("status", [400, 413])
def test_is_overflow_markers(marker, status):
    assert len(CONTEXT_OVERFLOW_MARKERS) == 9
    assert is_overflow(Http400(f"Error: {marker.upper()} (details)", status))
    assert is_overflow(status=status, text=f"... {marker} ...")
    assert not is_overflow(Http400(f"Error: {marker}", 500))
    assert not is_overflow(Http400("bad request: unknown parameter", status))


def test_is_overflow_crucible_classes_and_chains():
    assert is_overflow(ContextOverflow("m: finish=length with 0 completion tokens"))
    try:
        try:
            raise Http400("x", 400, body={"error": {"code": "context_length_exceeded"}})
        except Http400 as inner:
            raise LLMError("wrapped") from inner
    except LLMError as e:
        assert is_overflow(e)
    assert not is_overflow(LLMError("provider down"))
    assert not is_overflow(BudgetExceeded("credits"))
    assert not is_overflow(Http400("rate limit reached: maximum context length is irrelevant here", 400))
    assert not is_overflow(None)

    class Resp:
        status_code = 413
        text = "Request Entity Too Large"

    class HttpxLike(Exception):
        response = Resp()

    assert is_overflow(HttpxLike("413"))


def test_is_silent_overflow_pa_rules():
    w = 32_768
    assert is_silent_overflow({"prompt_tokens": w + 1, "completion_tokens": 30}, "stop", w, "ok", [])
    assert is_silent_overflow({"prompt_tokens": int(0.995 * w), "completion_tokens": 0}, "length", w, "", [])
    assert not is_silent_overflow({"prompt_tokens": int(0.98 * w), "completion_tokens": 0}, "length", w, "", [])
    assert not is_silent_overflow({"prompt_tokens": int(0.995 * w), "completion_tokens": 12}, "length", w, "partial", [])
    assert not is_silent_overflow({"prompt_tokens": 1_000, "completion_tokens": 0}, "stop", w, "", [])
    # no completion count: "no output" = no text and no tool calls
    assert is_silent_overflow({"prompt_tokens": w}, "length", w, "", [])
    assert not is_silent_overflow({"prompt_tokens": w}, "length", w, "", [{"name": "python"}])
    assert not is_silent_overflow({"blank": 1}, "length", w, "", [])
    # Anthropic-style cache fields count as input
    assert is_silent_overflow({"prompt_tokens": 100, "cache_read_input_tokens": w, "completion_tokens": 5}, "stop", w, "x", [])


# ---- summary_view ---------------------------------------------------------------------------------------------------------
def test_summary_view_caps_tool_outputs_and_keeps_the_source_intact():
    spec = ContextSpec()
    big = "H" * 3000 + "M" * 20_000 + "T" * 600
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}, asst(0), tool(0, big), asst(1), tool(1, "short")]
    snapshot = json.dumps(msgs)
    v = summary_view(msgs, spec)
    assert json.dumps(msgs) == snapshot  # not mutated
    out = v[3]["content"]
    assert out.startswith("H" * 2000) and out.endswith("T" * 500) and "omitted here; the ledger holds the full text" in out
    assert len(out) < 2000 + 500 + 120
    assert v[5]["content"] == "short"


def test_summary_view_prunes_old_images_and_reasoning():
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for i in range(5):
        msgs += [{**asst(i), "reasoning_content": f"why {i}", "content": f"<think>hmm {i}</think>ok {i}"},
                 {"role": "tool", "tool_call_id": f"c{i}_0", "content": [{"type": "text", "text": f"shot {i}"}, img]}]
    v = summary_view(msgs, ContextSpec())
    images = [p for m in v if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url"]
    assert len(images) == 3
    placeholders = [p for m in v if isinstance(m.get("content"), list) for p in m["content"] if "image omitted" in str(p.get("text", ""))]
    assert len(placeholders) == 2
    first_tool = next(m for m in v if m.get("role") == "tool")
    assert any("image omitted" in str(p.get("text", "")) for p in first_tool["content"])  # the oldest go first
    assistants = [m for m in v if m.get("role") == "assistant"]
    assert all("reasoning_content" not in m and "<think>" not in m["content"] for m in assistants[:-1])
    assert assistants[-1]["reasoning_content"] == "why 4"


def test_summary_view_clips_long_code_as_valid_json_and_drops_unpaired_rows():
    long_code = "x = [" + ",".join(str(i) for i in range(20_000)) + "]"
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}, asst(0, code=long_code), tool(0, "ok"),
            {"role": "tool", "tool_call_id": "ghost", "content": "orphan"}, asst(1)]  # c1 never answered
    v = summary_view(msgs, ContextSpec())
    args = json.loads(v[2]["tool_calls"][0]["function"]["arguments"])
    assert len(args["code"]) < 3_000 and "omitted here" in args["code"]
    assert all(m.get("tool_call_id") != "ghost" for m in v)
    assert "tool_calls" not in v[-1]
    assert pairing_ok(v)


# ---- hollow_middle --------------------------------------------------------------------------------------------------------
def test_hollow_middle_keeps_start_end_and_marks_the_gap():
    msgs = convo(60, out_chars=2_000)
    msgs.insert(2, {"role": "user", "content": SUMMARY_PREFIX + " earlier summary"})
    total = approx_tokens(msgs)
    target = total // 2
    out = hollow_middle(msgs, target)
    assert out[0] == msgs[0] and out[1] == msgs[1] and out[2] == msgs[2]  # system, task, summary row (before the first turn)
    assert out[-2:] == msgs[-2:]  # the last exchange
    markers = [m for m in out if str(m.get("content")).startswith("[hollowed:")]
    assert len(markers) == 1 and "the ledger holds every one" in markers[0]["content"]
    removed = len(msgs) - (len(out) - 1)
    assert f"[hollowed: {removed} earlier messages" in markers[0]["content"]
    assert approx_tokens(out) <= target
    assert total - approx_tokens(out) >= 1.25 * (total - target) * 0.95  # the 1.25 over-removal (whole exchanges: >=)
    assert pairing_ok(out)
    # again: the old marker's count carries over into one marker
    out2 = hollow_middle(out, approx_tokens(out) // 2)
    markers2 = [m for m in out2 if str(m.get("content")).startswith("[hollowed:")]
    assert len(markers2) == 1
    assert f"[hollowed: {len(msgs) - (len(out2) - 1)} earlier messages" in markers2[0]["content"]


def test_hollow_middle_no_op_cases():
    msgs = convo(4)
    assert hollow_middle(msgs, approx_tokens(msgs) + 10) is msgs
    tiny = convo(1, out_chars=50_000)
    assert hollow_middle(tiny, 10) == tiny  # only the task and the last exchange: nothing removable


def test_hollow_middle_pairing_property():
    rng = random.Random(1234)
    for _ in range(400):
        msgs = random_messages(rng)
        assert pairing_ok(msgs)
        out = hollow_middle(msgs, max(1, int(approx_tokens(msgs) * rng.uniform(0.05, 1.1))), over=rng.choice([1.0, 1.25, 2.0]))
        assert pairing_ok(out), out
        assert out[0] == msgs[0] and out[1] == msgs[1]
        if out is not msgs and len(out) != len(msgs):
            assert any(str(m.get("content", "")).startswith("[hollowed:") for m in out)


# ---- split_tail -----------------------------------------------------------------------------------------------------------
def test_split_tail_pairing_property():
    rng = random.Random(4321)
    nonempty = 0
    for _ in range(600):
        msgs = random_messages(rng)
        budget = rng.choice([0, 50, 300, 1_000, 5_000, 100_000])
        head, tail = split_tail(msgs, budget)
        assert head + tail == msgs
        if not tail:
            continue
        nonempty += 1
        assert tail[0]["role"] == "assistant"
        assert approx_tokens(tail) <= budget
        assert pairing_ok(tail) and pairing_ok(head)
        # maximal: no earlier assistant cut both fits and keeps pairs
        cut = len(head)
        for i in range(cut):
            if msgs[i]["role"] == "assistant" and approx_tokens(msgs[i:]) <= budget:
                assert not (pairing_ok(msgs[:i]) and pairing_ok(msgs[i:]))
    assert nonempty > 100


def test_split_tail_never_splits_an_exchange_with_a_notice_inside():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}, asst(0, n_calls=2), tool(0, "a", 0),
            {"role": "user", "content": "[inbox] hi"}, tool(0, "b", 1), asst(1), tool(1, "c")]
    head, tail = split_tail(msgs, 10_000)
    assert tail[0] is msgs[2]
    head, tail = split_tail(msgs, approx_tokens(msgs[6:]))
    assert tail == msgs[6:] and pairing_ok(head)
    assert split_tail(msgs, 0) == (msgs, [])


# ---- prompts, state row, boundary, rebuild -------------------------------------------------------------------------------
def test_summary_prompts_schema_and_framing():
    for p in (SUMMARY_PROMPT, UPDATE_PROMPT):
        low = p.lower()
        for needle in ("goal and done-criteria", "constraints and rules in force", "verbatim", "entities and ids", "pending and waiting",
                       "errors and fixes", "current hypothesis", "next step", "variables worth remembering", "kernel keeps running",
                       "history.search", "out(i)", "no tool call", "do not invent"):
            assert needle in low, needle
        assert "another" not in low and "language model" not in low and "llm" not in low
    assert "<previous-summary>" in UPDATE_PROMPT and "{previous_summary}" in UPDATE_PROMPT
    req = summary_request("old {braces} summary", instructions="keep ticket ids", max_tokens=2_000, note="retry note")
    assert "<previous-summary>\nold {braces} summary\n</previous-summary>" in req
    assert "keep ticket ids" in req and "retry note" in req and "1200 words" in req
    assert summary_request(None).startswith(SUMMARY_PROMPT)


def test_python_state_row_is_capped_and_names_lost_variables():
    entries = [{"name": f"ticket_{i}", "type": "dict", "len": i} for i in range(5_000)]
    entries[:0] = [{"name": "_private", "type": "int"}, {"name": "world", "type": "_World"}, "bare_name",
                   {"name": "orders", "type": "list", "len": 312}, {"name": "doc", "type": "str", "chars": 10_240}]
    row = python_state_row(entries, lost=["df", "model"])
    assert row.startswith("[python-state]") and len(row) <= 2_000
    assert "orders: list (312 items)" in row and "doc: str (10,240 chars)" in row and "bare_name" in row
    assert "_private" not in row and "world:" not in row
    assert "Lost in a kernel restart" in row and "df, model" in row
    shown = row.count("ticket_")
    assert shown <= 60 - 3 and "more" in row
    assert python_state_row([{"name": f"v{i}", "type": "int"} for i in range(200)], max_names=60).count(": int") <= 60
    assert "(none)" in python_state_row([])


def test_boundary_marker_names_history_and_the_gap_rule():
    b = boundary_marker(3, 109_000, 21_000, 4)
    assert "[compaction 3]" in b and "109,000" in b and "21,000" in b and "window 4" in b
    assert "state the gap before relying on missing details" in b
    assert "history.messages(window=3)" in b and "OUT(i)" in b and "history.search" in b


def test_rebuild_keeps_task_verbatim_and_fabricates_no_turn():
    system = {"role": "system", "content": "sys"}
    task = {"role": "user", "content": [{"type": "text", "text": "TASK verbatim: refund 17 (amount $12.50)"}]}
    tail = [tool(0, "orphan from before"), asst(5), tool(5, "kept")]
    out = rebuild(system, task, "the summary", "[python-state] x: int", boundary_marker(1, 10, 5, 1), tail)
    assert out[0] == system and out[1] == task and out[1] is not task
    assert out[2]["role"] == "user" and out[2]["content"].startswith(SUMMARY_PREFIX)
    assert "authoritative" in out[2]["content"] and "may lag" in out[2]["content"] and "the summary" in out[2]["content"]
    assert out[3]["content"].startswith("[python-state]") and out[4]["content"].startswith("[compaction 1]")
    assert [m["role"] for m in out].count("assistant") == 1 and out[5] is tail[1]  # only the tail's own assistant row
    assert all(m.get("content") != "orphan from before" for m in out)
    assert pairing_ok(out)
    no_state = rebuild(system, task, "s", "", "", [])
    assert [m["role"] for m in no_state] == ["system", "user", "user"]


# ---- Compactor --------------------------------------------------------------------------------------------------------------
def calls_log():
    log: list[tuple] = []
    return log, (lambda r, attempt, accepted, n: log.append((r, attempt, accepted, n)))


async def test_compactor_accepts_first_good_summary():
    llm = FakeLLM([ok_reply()])
    log, cb = calls_log()
    out = await Compactor().run(llm, convo(10), ContextSpec(), previous_summary=None, on_call=cb, sample_key="ep1", temperature=0.0)
    assert out.ok and out.attempts == 1 and out.failure is None and "Goal" in out.summary and out.input_tokens == 1000
    assert log == [(llm.script[0], 1, True, 1000)]
    req = llm.requests[0]
    assert req["tools"] is None and req["sample_key"] == "ep1" and req["seed"] == 0 and req["max_tokens"] == 2_000
    assert req["messages"][-1]["content"].startswith(SUMMARY_PROMPT)


async def test_compactor_overflow_hollows_and_retries():
    msgs = convo(80, out_chars=1_500)
    good = ok_reply()
    llm = FakeLLM([ContextOverflow("m: prompt is too long"), Http400("maximum context length is 32768 tokens", 400), good])
    log, cb = calls_log()
    out = await Compactor().run(llm, msgs, ContextSpec(window_tokens=128_000), on_call=cb)
    assert out.ok and out.attempts == 3
    sizes = [approx_tokens(r["messages"]) for r in llm.requests]
    assert sizes[0] > sizes[1] > sizes[2]
    assert any(str(m.get("content")).startswith("[hollowed:") for m in llm.requests[1]["messages"])
    assert [(a, acc) for _, a, acc, _ in log] == [(1, None), (2, None), (3, True)]
    assert isinstance(log[0][0], ContextOverflow) and log[2][0] is good
    for r in llm.requests:
        assert pairing_ok(r["messages"][:-1])


async def test_compactor_resamples_empty_tool_calling_reasoning_only_and_truncated_replies():
    script = [LLMResponse(text="", usage={"prompt_tokens": 900, "completion_tokens": 0}, raw_finish="stop"),
              LLMResponse(text="", tool_calls=[{"id": "t", "name": "python", "arguments": {}}], usage={"prompt_tokens": 900, "completion_tokens": 9}, raw_finish="tool_calls"),
              LLMResponse(text="<think>I should summarize</think>", usage={"prompt_tokens": 900, "completion_tokens": 9}, raw_finish="stop"),
              LLMResponse(text="1. Goal ... (cut", usage={"prompt_tokens": 900, "completion_tokens": 2_000}, raw_finish="length"),
              LLMResponse(text="<think>ok</think>1. Goal: done.", usage={"prompt_tokens": 950, "completion_tokens": 40}, raw_finish="stop")]
    llm = FakeLLM(script)
    log, cb = calls_log()
    out = await Compactor().run(llm, convo(6), ContextSpec(), on_call=cb, temperature=0.7)
    assert out.ok and out.attempts == 5 and out.summary == "1. Goal: done."
    assert [acc for _, _, acc, _ in log] == [False, False, False, False, True]
    prompts = [r["messages"][-1]["content"] for r in llm.requests]
    assert "no summary text" in prompts[1] and "called a tool" in prompts[2] and "no summary text" in prompts[3] and "cut off" in prompts[4]
    seeds = [r["seed"] for r in llm.requests]
    assert seeds[0] is None and len(set(seeds[1:])) == 4  # a replay cache never returns the rejected reply
    assert all(r["temperature"] == 0.7 for r in llm.requests)  # the sampling temperature never changes (trainable)


@pytest.mark.parametrize("fault", [LLMError("provider down"), BudgetExceeded("credits"), RuntimeError("bug")])
async def test_compactor_propagates_provider_faults_after_reporting_them(fault):
    llm = FakeLLM([fault])
    log, cb = calls_log()
    with pytest.raises(type(fault)):
        await Compactor().run(llm, convo(5), ContextSpec(), on_call=cb)
    assert len(log) == 1 and log[0][0] is fault and log[0][1:3] == (1, None) and log[0][3] > 0


async def test_compactor_all_attempts_fail_is_compaction_failed():
    llm = FakeLLM([LLMResponse(text="  ", usage={"prompt_tokens": 500, "completion_tokens": 1}, raw_finish="stop")])
    log, cb = calls_log()
    out = await Compactor().run(llm, convo(5), ContextSpec(max_compaction_attempts=3), on_call=cb)
    assert not out.ok and out.failure == "compaction_failed" and out.attempts == 3 and out.summary == "" and out.detail == "empty"
    assert [a for _, a, _, _ in log] == [1, 2, 3]


async def test_compactor_overflow_that_cannot_shrink_stops_early():
    msgs = convo(1, out_chars=500)
    llm = FakeLLM([ContextOverflow("too long")])
    _log, cb = calls_log()
    out = await Compactor().run(llm, msgs, ContextSpec(), on_call=cb)
    assert not out.ok and out.failure == "compaction_failed" and out.attempts == 1 and out.detail == "overflow"


async def test_compactor_silent_overflow_is_hollowed():
    spec = ContextSpec(window_tokens=128_000)
    llm = FakeLLM([LLMResponse(text="garbage", usage={"prompt_tokens": 130_000, "completion_tokens": 5}, raw_finish="stop"), ok_reply()])
    log, cb = calls_log()
    out = await Compactor().run(llm, convo(80, out_chars=1_500), spec, on_call=cb)
    assert out.ok and out.attempts == 2 and log[0][2] is False and log[0][3] == 130_000
    assert approx_tokens(llm.requests[1]["messages"]) < approx_tokens(llm.requests[0]["messages"])


async def test_compactor_request_is_bounded_whatever_the_context_holds():
    """B1: the compaction call used to be the segment's largest prompt. A context far past the window is capped by the view
    and hollowed to summary_input_tokens() before the first request."""
    spec = ContextSpec(window_tokens=32_768, max_out=2_000)
    msgs = convo(400, out_chars=8_000)
    assert approx_tokens(msgs) > 20 * spec.window_tokens
    meter = ContextMeter()
    meter.ratio = 1.42
    llm = FakeLLM([ok_reply()])
    await Compactor(meter=meter).run(llm, msgs, spec)
    sent = llm.requests[0]["messages"]
    assert approx_tokens(sent) * 1.42 <= spec.summary_input_tokens()
    assert sent[1] == msgs[1]  # the task is in the summarizer's view
    assert pairing_ok(sent[:-1])


async def test_compactor_update_mode_and_instructions():
    msgs = convo(4)
    msgs.insert(2, {"role": "user", "content": f"{SUMMARY_PREFIX}\nold summary text"})
    llm = FakeLLM([ok_reply()])
    await Compactor().run(llm, msgs, ContextSpec(), previous_summary="old summary text", instructions="keep the refund ids")
    sent = llm.requests[0]["messages"]
    assert not any(str(m.get("content")).startswith(SUMMARY_PREFIX) for m in sent)
    assert "<previous-summary>\nold summary text\n</previous-summary>" in sent[-1]["content"]
    assert "keep the refund ids" in sent[-1]["content"]


# ---- footer and wrap-up -------------------------------------------------------------------------------------------------------
def test_footer_is_compact_and_factual():
    f = footer(used=98_312, threshold=109_616, window=128_000, step=812, turns=1_204,
               budgets={"tokens": (796_000, 2_000_000), "sub_calls": (31, 256), "wall_s": (3_100, 14_400)}, elapsed_s=2.43, source="usage")
    assert f.startswith("<context_usage>") and f.endswith("</context_usage>")
    for needle in ("step 812", "turn 1,204", "98,312 of 109,616", "90%", "window 128,000", "source: usage", "tokens 796,000 of 2,000,000",
                   "sub_calls 31 of 256", "wall_s 3,100 of 14,400", "last cell 2.4 s"):
        assert needle in f, needle
    assert len(f) < 400 and "rlm.compact" not in f
    assert "rlm.compact()" in footer(used=1, threshold=2, window=3, step=0, turns=0, budgets={}, elapsed_s=None, source="estimate", compact_hint=True)


def test_wrap_up_notice():
    n = wrap_up_notice("turns", 120)
    assert "120 turns left" in n and "rlm.finish" in n and "do not compact" in n.lower()
    assert "seconds" in wrap_up_notice("wall_s", 600)
