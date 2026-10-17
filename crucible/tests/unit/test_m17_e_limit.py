"""M17 report 11 F5: the tool-call limit is told truthfully (one note when ``limit_warning`` calls remain) and, at the
limit, one final turn offers only write_file, for STATUS.json; then the episode ends as ``max_steps``. Off by default."""

from __future__ import annotations

import asyncio

from crucible.actors.llm import LLMResponse
from crucible.actors.policy import LLMToolPolicy
from crucible.core.schema.actions import Done, Observation, ToolCall, ToolSpec

TOOLS = [ToolSpec(app="repo", name="shell", description="run", input_schema={"type": "object", "properties": {"command": {"type": "string"}}}),
         ToolSpec(app="repo", name="write_file", description="write", input_schema={"type": "object", "properties": {"path": {"type": "string"},
                                                                                                                    "content": {"type": "string"}}})]


class Fake:
    """Two shell calls per reply until the limit turn; on the final turn it writes STATUS.json (or not)."""

    def __init__(self, final_writes: bool = True):
        self.final_writes, self.seen_tools, self.seen_msgs, self.n = final_writes, [], [], 0

    async def complete(self, messages, tools=None, **kw):
        self.n += 1
        self.seen_tools.append([t["function"]["name"] for t in tools or []])
        self.seen_msgs.append([m.get("content") for m in messages if m["role"] == "user"])
        if len(tools or []) == 1:
            if not self.final_writes:
                return LLMResponse(text="done, no status", raw_finish="stop", usage={"prompt_tokens": 1, "completion_tokens": 1})
            return LLMResponse(text="", tool_calls=[{"id": "fin", "name": "repo__write_file", "arguments": {"path": "STATUS.json", "content": "{}"}}],
                               raw_finish="tool_calls", usage={"prompt_tokens": 1, "completion_tokens": 1})
        return LLMResponse(text="", tool_calls=[{"id": f"a{self.n}", "name": "repo__shell", "arguments": {"command": "ls"}},
                                                {"id": f"b{self.n}", "name": "repo__shell", "arguments": {"command": "pwd"}}],
                           raw_finish="tool_calls", usage={"prompt_tokens": 1, "completion_tokens": 1})


def drive(pol: LLMToolPolicy, steps: int = 20) -> list:
    from crucible.core.schema.actions import ToolResult

    acts, prev = [], None
    for i in range(steps):
        res = [ToolResult(tool_call_id=prev.tool_call_id, function_name=prev.function_name, ok=True, content="ok")] if isinstance(prev, ToolCall) else []
        obs = Observation(actor="agent", step=i, instruction="task" if i == 0 else None, tools=TOOLS if i == 0 else [], tool_results=res)
        a = asyncio.run(pol.act(obs))
        acts.append(a)
        prev = a
        if isinstance(a, Done):
            break
    return acts


def test_the_limit_is_told_and_one_final_status_turn_follows():
    llm = Fake()
    pol = LLMToolPolicy("agent", llm, tool_limit=5, limit_warning=2)
    acts = drive(pol)
    calls = [a for a in acts if isinstance(a, ToolCall)]
    assert [a.function_name for a in calls] == ["repo.shell"] * 5 + ["repo.write_file"] and calls[-1].arguments["path"] == "STATUS.json"
    assert isinstance(acts[-1], Done) and pol.stop == "max_steps" and pol.tool_calls_issued == 5
    notes = [m for m in llm.seen_msgs[-1] if m and m.startswith("[harness]")]
    assert notes[0].startswith("[harness] 1 tool calls remain") or notes[0].startswith("[harness] 2 tool calls remain")
    assert len([n for n in notes if "remain" in n]) == 1 and "only tool is write_file" in notes[-1]
    assert llm.seen_tools[-1] == ["repo__write_file"]  # the final turn offers only the write tool
    # the reply cut by the limit (its 6th call) got a tool reply and was never run
    assert any(m.get("content") == "not run: the tool-call limit was reached" for m in pol.messages if m["role"] == "tool")
    assert [e["kind"] for e in pol.limit_events] == ["warning", "final_turn", "final_turn_status_written"]


def test_a_final_turn_without_status_ends_the_episode():
    pol = LLMToolPolicy("agent", Fake(final_writes=False), tool_limit=3, limit_warning=1)
    acts = drive(pol)
    assert sum(isinstance(a, ToolCall) for a in acts) == 3 and isinstance(acts[-1], Done) and pol.stop == "max_steps"
    assert pol.limit_events[-1]["kind"] == "final_turn_unused"


def test_off_by_default():
    llm = Fake()
    pol = LLMToolPolicy("agent", llm)
    acts = drive(pol, steps=12)
    assert all(isinstance(a, ToolCall) for a in acts) and not any("[harness]" in str(m.get("content")) for m in pol.messages)
