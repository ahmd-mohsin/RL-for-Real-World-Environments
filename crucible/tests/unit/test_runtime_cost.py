"""An attempt is charged for its own model calls. Concurrent attempts sharing one LLM client used to be charged the
client's cumulative spend delta over their window, so overlapping attempts counted each other's calls (the m13 live
benchmark showed $2.23 recorded for $1.32 spent)."""

from __future__ import annotations

import asyncio
import json

from crucible_pack_toy.runtime import HelpdeskTask

from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.inject import FakeLLM
from crucible.runtime.run import Runner, RunSpec


class SlowFake(FakeLLM):
    async def complete(self, messages, **kw):
        await asyncio.sleep(0.2)  # every attempt's call is in flight at once
        return await super().complete(messages, **kw)


async def test_concurrent_attempts_are_charged_their_own_calls(tmp_path):
    from crucible.actors.policy import LLMToolPolicy

    llm = SlowFake([], cost_per_call=0.01)
    ref = PolicyRef(id="llm:shared", make=lambda a: {"agent": LLMToolPolicy("agent", llm)}, spend=lambda: (llm.spend.usd, 0.0))
    await Runner(tmp_path).run(RunSpec(run_id="r", tasks=[HelpdeskTask()], policies=[ref], harnesses=[NativeHarness()], k=4, concurrency=4))
    rows = [json.loads(x) for x in (tmp_path / "r" / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and llm.spend.usd > 0.03
    assert all(abs(r["cost_usd"] - 0.01 * llm.spend.calls / 4) < 1e-9 for r in rows), [r["cost_usd"] for r in rows]
    assert abs(sum(r["cost_usd"] for r in rows) - llm.spend.usd) < 1e-9  # the attempts' costs add up to what was spent
