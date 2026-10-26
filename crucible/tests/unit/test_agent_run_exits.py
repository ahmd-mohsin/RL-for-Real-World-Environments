"""M3 audit (2026-10-20) R-3: a recorded frontier run is judged on its reward only if it ended on the agent's own stop.
mini-swe-agent records an exception's class name as ``exit_status`` and a killed run keeps an empty one; with a reward < 1
such a run was labelled ``bad`` and ``AgentRunSignalMapper`` emitted -1s (M8 turns them into traps), though the provider or
the harness ended it (``docs/research/improvements/2026-10-20/M03-connectors.md`` R-3). The mapper also scored runs graded
under an older task version, which ``normalize`` already labels ``unknown``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TASKS = Path(__file__).resolve().parents[3] / "migration_tasks"
TASK = "task04_pcl_gpu_cuda12"
RUNS = TASKS / TASK / "runs"

pytestmark = pytest.mark.skipif(not RUNS.is_dir(), reason="needs ../migration_tasks")


def _run(root: Path, name: str, source: str, *, reward: float, gates: dict | None = None, exit_status: str | None = None):
    """A run directory under ``root`` from a real recorded trajectory, with a grade report (and its exit status replaced)."""
    from crucible_pack_coding.connectors.agent_runs import AgentRunConnector

    d = root / TASK / "runs" / name
    d.mkdir(parents=True)
    traj = json.loads((RUNS / source / "trajectory.json").read_text())
    if exit_status is not None:
        traj["info"]["exit_status"] = exit_status
    (d / "trajectory.json").write_text(json.dumps(traj))
    (d / "grade_report.json").write_text(json.dumps({"reward": reward, **({"gates": gates} if gates else {})}))
    (d / "model.patch").write_text("diff --git a/x.cpp b/x.cpp\n--- a/x.cpp\n+++ b/x.cpp\n@@ -1 +1 @@\n-a\n+b\n")
    c = AgentRunConnector(root)
    return c.normalize(c.fetch(next(r for r in c.discover() if r.meta["run"] == name)))


@pytest.mark.parametrize(
    "source, status, gates",
    [
        ("gpt-6-astra-v2-run1", "OpenRouterAPIError", None),  # HTTP 402 from the provider
        ("gpt-6-astra-v2-run1", "OpenRouterAPIError", {"tests": {"pass": False}}),
        ("gpt-6-astra-v3-aborted", "RuntimeError", None),
        ("gpt-6-astra-v3-aborted", "RuntimeError", {"tests": {"pass": False}}),
        ("gpt-6-astra-v3-run2", "", None),  # killed: the last save before the kill has no exit status
    ],
)
def test_a_run_its_provider_or_harness_ended_is_unknown_and_unscored(tmp_path, source, status, gates):
    from crucible_pack_coding.signals import AgentRunSignalMapper

    t = _run(tmp_path, "r3", source, reward=0.0, gates=gates, exit_status=None if status else "")
    assert t.outcome.label == "unknown" and t.outcome.details["exit_status"] == status  # was "bad"
    assert "before it was done" in t.outcome.label_source
    assert AgentRunSignalMapper().map(t) == []  # was -1 signals: M8 traps from a provider's 402


@pytest.mark.parametrize("status", ["Submitted", "RepeatedFormatError"])
def test_the_agents_own_stops_stay_scored(tmp_path, status):
    from crucible_pack_coding.signals import AgentRunSignalMapper

    t = _run(tmp_path, "r3", "gpt-6-astra-v3-run2", reward=0.0, gates={"tests": {"pass": False}}, exit_status=status)
    assert t.outcome.label == "bad" and t.outcome.details["exit_status"] == status
    assert [s.polarity for s in AgentRunSignalMapper().map(t)] == [-1]


def test_a_trace_stored_without_an_exit_status_keeps_its_signals(tmp_path):
    from crucible_pack_coding.signals import AgentRunSignalMapper

    t = _run(tmp_path, "r3", "gpt-6-astra-v2-run1", reward=0.0)
    old = {k: v for k, v in t.outcome.details.items() if k != "exit_status"}  # a store written before the exit was recorded
    t = t.model_copy(update={"outcome": t.outcome.model_copy(update={"details": old})})
    assert [s.polarity for s in AgentRunSignalMapper().map(t)] == [-1]


def test_the_real_runs_keep_their_labels_and_stale_runs_are_unscored():
    from crucible_pack_coding.connectors.agent_runs import AgentRunConnector
    from crucible_pack_coding.signals import AgentRunSignalMapper

    c = AgentRunConnector(TASKS)
    traces = {r.key: c.normalize(c.fetch(r)) for r in c.discover() if r.meta["task"] == TASK}
    bad = traces[f"{TASK}/gpt-6-astra-v3-run2"]
    assert bad.outcome.label == "bad" and bad.outcome.details["exit_status"] == "Submitted"
    assert AgentRunSignalMapper().map(bad) and all(s.polarity == -1 for s in AgentRunSignalMapper().map(bad))
    stale = [t for t in traces.values() if t.outcome.details.get("stale_version")]
    assert stale and all(t.outcome.label == "unknown" for t in stale)
    assert [AgentRunSignalMapper().map(t) for t in stale] == [[]] * len(stale)  # was +1 "every hidden gate passed" under v1
