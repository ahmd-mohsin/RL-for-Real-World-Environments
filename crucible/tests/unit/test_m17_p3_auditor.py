"""P3 (swarm-pipeline/00 §8.2 step 3, §8.4): the RLM auditor reads the bundle through code; the auditor is audited
(seeded controls trust or drop a batch, same-family judging refused, a 10% human kappa queue, harness edits only through
``HarnessState.evaluate``)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from crucible.actors.llm import LLMResponse
from crucible.actors.rlm.agent import RLMConfig
from crucible.actors.rlm.kernel import LocalLauncher
from crucible.factory.audit import (
    AuditConfig,
    AuditLabel,
    Auditor,
    SeededCase,
    Trace,
    audit,
    cross_family_problems,
    kappa_vs_human,
    policy_model,
    split_by_family,
)
from crucible.factory.traceaudit import FACTS_MARK, AuditorHarness, EpisodeBundle, RLMAuditor, ToolStep, run_loop
from crucible.factory.versioning import VersionStore

TRACE = ("[episode] r/s/a1 task=w1~x\n[request]\nAdd format_duration.\n[tool 1] repo.write_file {\"path\": \"d.py\"}\n"
         "[result 1] (ok) wrote d.py\n[grader tail]\nE   ImportError: cannot import name 'format_duration' from 'durations'\n"
         + FACTS_MARK.lstrip("\n") + "reply_before_action: claimed done\n[first wrong step] flat step=0")


def cell(code: str, i: int) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[{"id": f"py{i}", "name": "python", "arguments": {"code": code}}], usage={"prompt_tokens": 50, "completion_tokens": 10})


class Script:
    """A scripted auditor model: searches CONTEXT, then finishes with ``reply``."""

    model = "script"

    def __init__(self, reply: dict[str, Any]):
        self.reply, self.n = reply, 0

    async def complete(self, messages: list[dict[str, Any]], *, tools: Any = None, max_tokens: int = 2000, **kw: Any) -> LLMResponse:
        self.n += 1
        if self.n == 1:
            return cell("h = await CONTEXT.search('ImportError')\nprint(h['items'][0]['snippet'])", 1)
        return cell(f"import json\nrlm.finish(json.dumps({self.reply!r}))", 2)


def auditor(tmp_path, reply: dict[str, Any], **kw: Any) -> RLMAuditor:
    cfg = RLMConfig(temperature=0.0, kernel="local", allow_host=True, tree_max_depth=0, max_turns=6, window_tokens=12_000, max_tokens=2000)
    return RLMAuditor(Script(reply), model="deepseek/deepseek-v4-flash", workdir=tmp_path / "aud", launcher=lambda: LocalLauncher(allow_host=True),
                      rlm_config=cfg, **kw)


async def test_rlm_auditor_reads_context_through_code_and_quotes_the_attempt(tmp_path):
    a = auditor(tmp_path, {"cause": "spec_defect", "solution_correct": False, "evidence": "cannot import name 'format_duration'", "rationale": "unnamed",
                           "defect": {"kind": "spec_defect", "pointer": "tests/test_format.py::test_format"}})
    lab = await a.label(Trace("r/s/a1", TRACE, False, policy="solver:qwen/qwen3.7-flash"), prompt="")
    assert lab.status == "labelled" and lab.cause == "spec_defect" and "defect=spec_defect" in lab.why
    s = a.sessions[0]
    assert s["finished"] and s["context_reads"] >= 1 and s["stop"] == "agent_done"
    own = auditor(tmp_path, {"cause": "skill_gap", "solution_correct": False, "evidence": "reply_before_action: claimed done", "rationale": "x"})
    lab2 = await own.label(Trace("r/s/a2", TRACE, False), prompt="")
    assert lab2.status == "rejected" and "derived lines" in lab2.why  # the factory's own facts are not the attempt
    bad = auditor(tmp_path, {"cause": "laziness", "evidence": "wrote d.py", "rationale": "x"})
    assert (await bad.label(Trace("r/s/a3", TRACE, False), prompt="")).status == "rejected"
    masked = auditor(tmp_path, {})
    lab3 = await masked.label(Trace("r/s/a4", TRACE, None), prompt="")
    assert lab3.cause == "env_flake" and lab3.status == "deterministic" and masked.sessions == []


def test_same_family_judging_is_refused_by_config():
    assert policy_model("solver:qwen/qwen3.7-flash@route") == "qwen/qwen3.7-flash"
    assert cross_family_problems("deepseek/deepseek-v4-flash", ["hacker:deepseek/deepseek-v4-flash", "solver:qwen/qwen3.7-flash", "control", ""]) == [
        "policy hacker:deepseek/deepseek-v4-flash and auditor deepseek/deepseek-v4-flash are both family 'deepseek'"]
    ok, no = split_by_family([Trace("a", "x", False, policy="hacker:deepseek/deepseek-v4-flash"), Trace("b", "x", False, policy="solver:meta-llama/llama-3.1-8b-instruct")],
                             "deepseek/deepseek-v4-flash")
    assert [t.id for t in ok] == ["b"] and [t.id for t in no] == ["a"]


class Judge:
    """A plain auditor client: labels a trace by the ``[cause:x]`` tag in it (or always ``always``)."""

    def __init__(self, always: str | None = None, correct: bool | None = False):
        self.always, self.correct = always, correct

    async def complete_json(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        tr = json.loads(messages[-1]["content"])["trace"]
        cause = self.always or tr.split("[cause:", 1)[1].split("]", 1)[0]
        return {"cause": cause, "solution_correct": self.correct, "evidence": tr.split("\n", 1)[0], "rationale": "r"}


async def test_audit_refuses_a_same_family_batch():
    with pytest.raises(ValueError, match="same-family"):
        await audit([Trace("a", "x [cause:skill_gap]", False, policy="solver:deepseek/deepseek-v4-flash")], Auditor(Judge(), model="deepseek/deepseek-v4-flash"),
                    prompt="p")


def bundle(i: int, family: str = "llama") -> EpisodeBundle:
    return EpisodeBundle(id=f"r/s{i}/a1", dir="", run="r", task="w1~x", window="w1", policy=f"solver:meta-llama/llama-{i}", model="meta-llama/llama", family=family,
                         prompt="[cause:verifier_defect] the request", steps=[ToolStep(1, "c1", "repo.read_file", {"path": "m.py"}, "x = 1")], phi0=0.0, reward=0.0,
                         failing_tests=["t::x"], patch="diff --git a/m.py b/m.py\n")


SEEDS = [SeededCase(Trace("ctl-world", "the tool crashed [cause:world_bug]", False, policy="control"), "world_bug"),
         SeededCase(Trace("ctl-skill", "it gave up [cause:skill_gap]", False, policy="control"), "skill_gap")]


async def test_untrusted_batches_drop_label_derived_proposals(tmp_path):
    bs = [bundle(i) for i in range(4)]
    trusted = await run_loop(bs, out=tmp_path / "t", store=VersionStore(tmp_path / "t.jsonl"), auditor=Auditor(Judge(correct=True), model="deepseek/x"),
                             seeded=SEEDS)
    assert trusted["auditor"]["batches"][0]["trusted"] and trusted["proposals"]["by_kind_status"]["false_fail"] == {"needs_input": 4}  # no verifier port: proposed, never applied
    liar = await run_loop(bs, out=tmp_path / "u", store=VersionStore(tmp_path / "u.jsonl"),
                          auditor=Auditor(Judge(always="verifier_defect", correct=True), model="deepseek/x"), seeded=SEEDS)
    assert liar["auditor"]["batches"][0]["trusted"] is False and "false_fail" not in liar["proposals"]["by_kind_status"]
    assert liar["auditor"]["trusted_labels"] == 0 and liar["causes"]["labelled_trusted"] == {}
    rows = [json.loads(x) for x in (tmp_path / "t" / "human_queue.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["human_cause"] is None  # 10% of 4, at least one


def test_kappa_vs_human():
    labels = {f"t{i}": AuditLabel(f"t{i}", c, False) for i, c in enumerate(["skill_gap", "skill_gap", "world_bug", "spec_defect"])}
    k = kappa_vs_human(labels, {"t0": "skill_gap", "t1": "skill_gap", "t2": "world_bug", "t3": "skill_gap", "t9": None})
    assert k["n"] == 4 and k["agree"] == 3 and k["pending"] == 1 and 0.0 < k["kappa"] < 1.0


async def test_auditor_harness_edits_only_through_measured_evaluate(tmp_path):
    h = AuditorHarness(tmp_path / "h.json")
    assert not [m for m in ("propose", "apply_unmeasured", "rollback_to", "add", "update") if hasattr(h, m)]
    edit = [{"action": "create", "kind": "note", "id": "n1", "title": "quote tool results", "content": "quote a [result] line", "family": "auditor"}]

    async def no_help(entries: dict[str, Any], seed: int) -> dict[str, float]:
        return {"seeded_accuracy": 0.8, "label_rate": 0.9}

    ev = await h.refine(edit, no_help, [1, 2, 3])
    assert not ev["accepted"] and h.entries == {} and h.digest("quote") == ""

    async def helps(entries: dict[str, Any], seed: int) -> dict[str, float]:
        return {"seeded_accuracy": 1.0 if "n1" in entries else 0.6, "label_rate": 0.9}

    ev2 = await h.refine(edit, helps, [1, 2, 3])
    assert ev2["accepted"] and "n1" in h.entries and "quote a [result] line" in h.digest("quote")
    assert AuditorHarness(tmp_path / "h.json").entries.keys() == {"n1"}  # persisted, versioned by HarnessState


def test_audit_config_default_forbids_same_family():
    assert AuditConfig().forbid_same_family
