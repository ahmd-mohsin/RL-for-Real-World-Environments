"""M17 owner S: the fixes the end-to-end run forced in P/Q/R files, each marked "(M17 e2e)" in the source.

- audit: a seeded control built from another known case is judged against its own request (an ``unstated_format``
  control *is* a request missing a stated format); judge calls run concurrently and keep the batch order.
- root: a run stopped by the budget is resumed by the next lease (not replaced by a new run that repays every settled
  state); a restarted root books what earlier processes spent, from the machine journals.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from crucible.factory.audit import AuditConfig, Auditor, SeededCase, Trace, audit
from crucible.factory.root import LeaseBook, RootCurator
from crucible.forge.budget import Ledger
from tests.unit.m17_r_fakes import ScriptedClient, user_payload
from tests.unit.test_m17_q_driver import FakeLLM, FakeStages
from tests.unit.test_m17_q_root import maker, matrix


def _judge(seen: list[str]):
    def reply(messages):
        p = user_payload(messages)
        seen.append(p["task_prompt"])
        line = p["trace"].splitlines()[0]
        cause = "unstated_format" if "format" in p["task_prompt"] else "skill_gap"
        return {"cause": cause, "solution_correct": None, "evidence": line, "rationale": "x"}

    return ScriptedClient(reply)


async def test_a_seeded_control_is_judged_against_its_own_request() -> None:
    seen: list[str] = []
    real = [Trace(f"r{i}", f"attempt {i}: wrong answer 7", False, policy="p") for i in range(3)]
    ctl = SeededCase(Trace("ctl", "attempt: right content, other format", False, policy="control"), "unstated_format",
                     prompt="the control's own request (no format stated)")
    rep = await audit(real, Auditor(_judge(seen)), prompt="the batch's request", seeded=[ctl])
    assert rep.seeded == [{"id": "ctl", "expected": "unstated_format", "got": "unstated_format", "status": "labelled", "ok": True}]
    assert seen.count("the batch's request") == 3 and seen.count("the control's own request (no format stated)") == 1
    assert {lab.cause for lab in rep.labels.values()} == {"skill_gap"}  # real traces still see the batch prompt


async def test_judge_calls_run_concurrently_in_batch_order() -> None:
    inflight, peak = [0], [0]

    class Slow:
        async def complete_json(self, messages, **kw):
            inflight[0] += 1
            peak[0] = max(peak[0], inflight[0])
            await asyncio.sleep(0.01)
            inflight[0] -= 1
            t = user_payload(messages)["trace"]
            return {"cause": "skill_gap", "solution_correct": False, "evidence": t, "rationale": "x"}

    traces = [Trace(f"t{i}", f"trace number {i:02d} failed", False) for i in range(12)]
    rep = await audit(traces, Auditor(Slow(), config=AuditConfig(concurrency=4)), prompt="p")
    assert 1 < peak[0] <= 4
    assert all(rep.labels[f"t{i}"].evidence == f"trace number {i:02d} failed" for i in range(12))  # each label on its own trace


def test_a_stopped_run_is_resumed_by_the_next_lease(tmp_path: Path) -> None:
    lb = LeaseBook(tmp_path)
    a = lb.acquire("c", "root")
    lb.release(a, stopped=True)
    b = lb.acquire("c", "root")
    assert b.run_id == a.run_id  # the same run, resumed from its journal
    lb.release(b)
    c = lb.acquire("c", "root")
    assert c.run_id != a.run_id  # a run that finished is not resumed again


async def test_a_restarted_root_books_past_spend_and_resumes_its_stopped_runs(tmp_path: Path) -> None:
    llm, stages = FakeLLM(usd=0.02), FakeStages()
    ledger = Ledger(cap_usd=0.15)
    root = RootCurator(tmp_path, matrix(4), make_run=maker(llm, stages, ledger=ledger), ledger=ledger, concurrency=2, max_surface_share=1.0)
    rep = await root.run()
    assert rep.stop_reason.startswith("budget") and rep.stopped
    spent = ledger.committed
    journal_usd = sum(json.loads(x).get("usd", 0.0) for j in (tmp_path / "runs").rglob("journal.jsonl") for x in j.read_text().splitlines()
                      if json.loads(x).get("event") == "spend")
    assert abs(journal_usd - spent) < 1e-9
    # a new process: a fresh ledger with a larger cap; the root books the journals' spend before launching anything
    ledger2 = Ledger(cap_usd=5.0)
    root2 = RootCurator(tmp_path, matrix(4), make_run=maker(llm, stages, ledger=ledger2), ledger=ledger2, concurrency=2, max_surface_share=1.0)
    assert abs(root2.past_usd - spent) < 1e-9 and abs(ledger2.committed - spent) < 1e-9
    rep2 = await root2.run()
    resumed = set(rep.stopped) & {lc["run_id"] for lc in rep2.launches}
    assert resumed == set(rep.stopped)  # every stopped run continued under its own id
    assert set(rep2.merged) >= resumed


async def test_a_state_retry_asks_for_a_new_sample() -> None:
    """With the LLM record cache keyed by the messages, a retry without a new sample key replays the malformed reply."""
    from crucible.factory.driver import _ask
    from crucible.factory.machine import StateContext

    seen: list[dict] = []

    class C:
        async def complete_json(self, messages, **kw):
            seen.append(kw)
            return {"ok": True}

    for attempt in (1, 2):
        await _ask(StateContext("r", "realism", 1, attempt, {}, [], {}, "realism", None, C()), "brief", {"x": 1})
    assert "sample_key" not in seen[0] and seen[1]["sample_key"] == "realism#1.2"


async def test_models_with_a_broken_json_mode_get_prompt_only_json() -> None:
    from crucible.factory.models import ModelBook, PromptJsonClient

    book = ModelBook()
    seen: list[dict] = []

    class Raw:
        spend = type("S", (), {"usd": 0.25})()

        async def complete_json(self, messages, **kw):
            seen.append(kw)
            return {"ok": True}

    book._clients["role:realism"] = Raw()  # type: ignore[assignment]
    book._clients["role:author"] = Raw()  # type: ignore[assignment]
    c = book.client("realism")  # deepseek-v4-flash by default
    assert isinstance(c, PromptJsonClient) and c.spend.usd == 0.25
    await c.complete_json([], max_tokens=5)
    await book.client("author").complete_json([], max_tokens=5)  # qwen keeps the provider's JSON mode
    assert seen[0]["json_mode"] is False and "json_mode" not in seen[1]
    assert book.observed() == 0.5
