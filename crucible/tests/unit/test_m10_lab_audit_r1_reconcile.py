"""M10 lab audit R-1 (2026-10-15), slice 3: the capture-vs-ATIF reconcile (an adapted port of OpenEnv
``src/openenv/harbor/atif.py``) replaces ``trajectories.cross_check``, and the Runner records the report on every record and
masks a FATAL one in train mode only.

Pure dicts and a fake harness: no Harbor, no agentenv, no Docker, no network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from crucible.connectors.capture_proxy import LIMIT_STOP_TEXT
from crucible.core.schema.run import Failure
from crucible.interop.harbor.reconcile import COVERAGE_FLOOR, ReconcileReport, agent_steps, reconcile
from crucible.runtime.harness.base import HarnessResult
from crucible.runtime.harness.native import PolicyRef
from crucible.runtime.infra import RetryRule
from crucible.runtime.run import Runner, RunSpec


def step(tokens, msg="x", **kw):
    return {"source": "agent", "message": msg, "metrics": {"completion_tokens": tokens}, **kw}


def call(i, tokens):
    return {"call_id": f"agent:cap{i}", "usage": {"completion_tokens": tokens}}


def atif_of(*tokens, **kw):
    return {"schema_version": "ATIF-v1.7", "steps": [{"source": "user", "message": "do it"}, *(step(t) for t in tokens)], **kw}


def calls_of(*tokens):
    return [call(i, t) for i, t in enumerate(tokens, 1)]


def codes(rep: ReconcileReport) -> list[str]:
    return [f["code"] for f in rep.findings]


# ---------------------------------------------------------------- the comparison


def test_exact_match_is_ok():
    rep = reconcile(atif_of(5, 7, 9), calls_of(5, 7, 9))
    assert (rep.level, rep.code) == ("info", "turns_match") and rep.aux_call_ids == []
    d = rep.to_dict()
    assert d["level"] == "info" and d["code"] == "turns_match" and d["findings"][0]["code"] == "turns_match" and d["aux_call_ids"] == []
    assert json.loads(json.dumps(d)) == d  # it travels in record meta


def test_mini_swe_agent_five_of_six_is_aux_not_fatal():
    """OpenEnv atif.py:401: mini-swe-agent makes an auxiliary call it does not log as an agent step. cross_check
    required equal counts and would have failed every gate R3 attempt."""
    rep = reconcile(atif_of(58, 132, 266, 370, 33), calls_of(58, 132, 266, 370, 105, 33))
    assert (rep.level, rep.code) == ("warn", "atif_aux_calls")
    assert rep.aux_call_ids == ["agent:cap5"]


def test_low_coverage_refuses():
    ours = [3, 11, 4, 5, 9, 2, 6, 8, 1, 12]
    rep = reconcile(atif_of(11, 8), calls_of(*ours))  # 2 of 10 embed as a subsequence, under the 50% floor
    assert 2 < COVERAGE_FLOOR * 10
    assert (rep.level, rep.code) == ("fatal", "atif_coverage_too_low") and rep.aux_call_ids == []


def test_missing_and_mismatched_calls_are_fatal():
    rep = reconcile(atif_of(5, 7, 9), calls_of(5, 7))  # ATIF logged a call the capture never saw
    assert (rep.level, rep.code) == ("fatal", "atif_calls_missing")
    rep = reconcile(atif_of(5, 7, 9), calls_of(5, 8, 9))
    assert (rep.level, rep.code) == ("fatal", "turn_mismatch")
    rep = reconcile(atif_of(5, 7), calls_of(5, 6, 9))  # longer capture that does not embed ATIF: not aux, a mismatch
    assert (rep.level, rep.code) == ("fatal", "turn_mismatch")
    rep = reconcile(atif_of(5, 7), [])  # agent steps but no call reached the proxy (a model reached around it)
    assert (rep.level, rep.code) == ("fatal", "atif_calls_missing")


def test_limit_stops_and_synthetic_steps_are_excluded():
    stop = step(0, msg=LIMIT_STOP_TEXT)
    stop["metrics"]["prompt_tokens"] = 0
    atif = {"steps": [step(5), step(7), stop]}
    rep = reconcile(atif, calls_of(5, 7), limit_stops=1)
    assert rep.level == "info" and rep.code == "turns_match" and "proxy_limit_stops" in codes(rep)
    assert reconcile(atif, calls_of(5, 7)).level == "fatal"  # no proxy count: the stop step stays and is a missing call
    two = {"steps": [step(5), step(7), stop, dict(stop)]}
    assert reconcile(two, calls_of(5, 7), limit_stops=1).level == "fatal"  # never more than the proxy answered
    sampled = {"steps": [step(5), step(7), step(12, msg=LIMIT_STOP_TEXT)]}
    assert reconcile(sampled, calls_of(5, 7), limit_stops=1).level == "fatal"  # a sampled copy of the text is a real step
    synth = step(0, msg="API Error: 529 overloaded", model_name="<synthetic>")
    synth["metrics"]["prompt_tokens"] = 0
    rep = reconcile({"steps": [step(5), synth, step(7)]}, calls_of(5, 7))
    assert (rep.level, rep.code) == ("warn", "atif_synthetic_api_error") and "turns_match" in codes(rep)


def test_dispatch_steps_and_failed_calls_are_left_out():
    """ATIF v1.7: llm_call_count 0 on an agent step is a deterministic dispatch, not a model call (Crucible's own to_atif
    writes one per extra action of a call). A captured call that failed upstream gave the agent no reply to log."""
    second = {"source": "agent", "message": "", "llm_call_count": 0}
    atif = {"steps": [step(5, llm_call_count=1), second, step(7)]}
    assert [s.get("llm_call_count") for s in agent_steps(atif)] == [1, None]
    rep = reconcile(atif, calls_of(5, 7))
    assert (rep.level, rep.code) == ("info", "turns_match") and "atif_dispatch_steps" in codes(rep)
    failed = {"call_id": "agent:capX", "usage": {}, "status": 503, "error": "HTTP 503"}
    rep = reconcile(atif_of(5, 7), [call(1, 5), failed, call(3, 7)])
    assert (rep.level, rep.code) == ("info", "turns_match") and "failed_calls" in codes(rep)


def test_no_token_counts_and_counts_only_paths():
    rep = reconcile(atif_of(0, 0, 0), calls_of(5, 7, 9))
    assert (rep.level, rep.code) == ("warn", "atif_no_token_counts")
    bare = lambda n: [{"call_id": f"agent:cap{i}", "usage": {}} for i in range(1, n + 1)]  # noqa: E731
    rep = reconcile(atif_of(5, 7, 9), bare(2))
    assert (rep.level, rep.code) == ("fatal", "atif_calls_missing")
    rep = reconcile(atif_of(5, 7), bare(3))
    assert (rep.level, rep.code) == ("info", "atif_extra_calls")
    rep = reconcile(atif_of(5, 7), bare(2))
    assert (rep.level, rep.code) == ("info", "counts_match")
    anthropic = [{"call_id": "agent:cap1", "usage": {"input_tokens": 10, "output_tokens": 5}}]
    assert reconcile(atif_of(5), anthropic).code == "turns_match"  # Anthropic / Responses usage names


def test_prompt_and_retry_warnings():
    first = step(5)
    first["metrics"]["prompt_tokens"] = 100
    rep = reconcile({"steps": [first, step(7, llm_call_count=2)]}, [{"call_id": "c1", "usage": {"prompt_tokens": 90, "completion_tokens": 5}}, call(2, 7)])
    assert rep.level == "warn" and {"prompt_len_mismatch", "atif_retries"} <= set(codes(rep)) and "turns_match" in codes(rep)


def test_subagents_are_fatal():
    rep = reconcile({"steps": [], "subagent_trajectories": [{}]}, [])
    assert (rep.level, rep.code) == ("fatal", "atif_subagents")
    rep = reconcile(atif_of(58, 132, 266, 370, 33, subagent_trajectories=[{}]), calls_of(58, 132, 266, 370, 105, 33))
    assert (rep.level, rep.code) == ("fatal", "atif_subagents")  # the aux path no longer returns before the subagent check
    rep = reconcile(None, calls_of(5))
    assert (rep.level, rep.code) == ("info", "no_atif")
    assert ReconcileReport().level == "ok" and ReconcileReport().code is None
    with pytest.raises(ValueError):
        ReconcileReport().add("error", "x", "y")


# ---------------------------------------------------------------- the Runner's consequence

TOK = {"completion_token_ids": [7, 8], "logprobs": [-0.1, -0.2], "policy_version": 3}


def token_calls(n: int) -> list[dict]:
    """Train-complete calls (as in test_runtime_records.toks): each prompt extends the previous prompt + completion."""
    out, prompt = [], [1, 2, 3]
    for i in range(n):
        out.append({"call_id": f"agent:cap{i + 1}", "model": "m", "usage": {"prompt_tokens": len(prompt), "completion_tokens": 2}, "kind": "turn",
                    "tokens": {**TOK, "prompt_token_ids": list(prompt)}})
        prompt = prompt + TOK["completion_token_ids"] + [100 + i]
    return out


class FakeHarness:
    id = "fake@1"

    def __init__(self, level: str):
        self.level = level

    async def run(self, attempt):
        rc = {"level": self.level, "code": "turn_mismatch", "findings": [{"level": self.level, "code": "turn_mismatch", "detail": "d"}], "aux_call_ids": []}
        return HarnessResult(reward=1.0, gates={}, calls={"agent": token_calls(2)}, grade={"reconcile": rc, "agent": {"harbor_version": "0.24.0"}})


async def _run(tmp_path, name: str, level: str, mode: str):
    spec = RunSpec(run_id=name, tasks=[SimpleNamespace(id="t", world_id="w")], policies=[PolicyRef(id="p", make=lambda a: {})],
                   harnesses=[FakeHarness(level)], k=1, rollout_mode=mode, retry=RetryRule(max_attempts=3))
    s = await Runner(tmp_path).run(spec)
    row = json.loads((tmp_path / name / "results.jsonl").read_text().splitlines()[-1])
    ro = json.loads(next((tmp_path / name / "samples").glob("*/a*/rollout.json")).read_text())
    return s, row, ro


async def test_runner_masks_fatal_reconcile_in_train_only(tmp_path):
    s, row, ro = await _run(tmp_path, "tw", "warn", "train")  # the control: this fixture is train-valid
    assert not row["mask_sample"] and row["reward"] == 1.0 and not ro["mask_sample"] and ro["reward"] == 1.0
    assert row["meta"]["reconcile"]["level"] == "warn" and row["meta"]["agent"]["harbor_version"] == "0.24.0"
    assert not any(f.get("rule") == "reconcile_fatal" for f in ro.get("flags") or [])

    s, row, ro = await _run(tmp_path, "tf", "fatal", "train")
    assert row["mask_sample"] and row["reward"] is None and row["failure"]["kind"] == "crucible:capture_incomplete"
    assert row["failure"]["stage"] == "collect" and row["failure"]["blame"] == "unknown" and row["failure"]["retryable"] is False
    assert "turn_mismatch" in row["failure"]["reason"] and row["meta"]["reconcile"]["level"] == "fatal"
    assert ro["mask_sample"] and ro.get("reward") is None and ro["failure_kind"] == "crucible:capture_incomplete"  # rollout.json drops None
    assert any(f["rule"] == "reconcile_fatal" and f["code"] == "turn_mismatch" for f in ro["flags"])
    assert s.attempts == 1 and s.scored == 0  # never retried until it reconciles (survivorship)

    s, row, ro = await _run(tmp_path, "ef", "fatal", "eval")
    assert not row["mask_sample"] and row["reward"] == 1.0 and row.get("failure") is None and row["meta"]["reconcile"]["level"] == "fatal"
    assert not ro["mask_sample"] and ro["reward"] == 1.0 and any(f["rule"] == "reconcile_fatal" for f in ro["flags"])  # recorded, flagged
    assert s.scored == 1


class ContendedHarness:
    """The patch seam's shape: the agent attempt stores its patch and is masked at verification (node contention) with a
    FATAL reconcile; ``regrade`` scores the stored patch with the report re-attached (as HarborAgentHarness.regrade)."""

    id = "contended@1"
    RC = {"level": "fatal", "code": "atif_calls_missing", "findings": [{"level": "fatal", "code": "atif_calls_missing", "detail": "d"}], "aux_call_ids": []}

    async def run(self, attempt):
        (attempt.dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (attempt.dir / "artifacts" / "model.patch").write_text("diff --git a/x b/x\n")
        return HarnessResult(calls={"agent": token_calls(2)}, grade={"reconcile": dict(self.RC)},
                             failure=Failure(kind="crucible:node_contention", stage="verification", reason="node never quiet"))

    async def regrade(self, attempt, prior):
        return HarnessResult(stop="regraded", reward=1.0, gates={}, calls={"agent": token_calls(2)}, grade={"reconcile": dict(self.RC)})


async def test_runner_regrade_keeps_the_fatal_reconcile_mask_in_train(tmp_path):
    """M10 lab audit R-1: Runner.regrade never reaches _rollout, so a sample masked at verification (its FATAL reconcile
    only flagged) and then re-graded was scored in train mode."""
    for name, mode in (("rt", "train"), ("re", "eval")):
        spec = RunSpec(run_id=name, tasks=[SimpleNamespace(id="t", world_id="w")], policies=[PolicyRef(id="p", make=lambda a: {})],
                       harnesses=[ContendedHarness()], k=1, rollout_mode=mode, retry=RetryRule(max_attempts=1))
        s = await Runner(tmp_path).run(spec)
        assert s.masked_terminal == 1 and s.scored == 0
        s = await Runner(tmp_path).regrade(spec)
        row = json.loads((tmp_path / name / "results.jsonl").read_text().splitlines()[-1])
        assert s.attempts == 1 and row["attempt"].endswith("-a2") and row["meta"]["stop"] == "regraded" and row["meta"]["reconcile"]["level"] == "fatal"
        if mode == "train":
            f = row["failure"]
            assert row["mask_sample"] and row["reward"] is None and f["kind"] == "crucible:capture_incomplete" and f["detail"] == "reconcile_fatal"
            assert f["stage"] == "collect" and s.scored == 0
        else:
            assert not row["mask_sample"] and row["reward"] == 1.0 and s.scored == 1
