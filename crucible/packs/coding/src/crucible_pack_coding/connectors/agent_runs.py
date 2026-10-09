"""Recorded frontier runs on the migration tasks -> traces (coding pack ``mini_swe_agent_runs`` connector, M3).

A run directory (``migration_tasks/<task>/runs/<run>/``) holds ``trajectory.json`` (mini-swe-agent), the submitted
``model.patch`` and, for tasks 01-03, ``grade_report.json`` from the hidden verifier. The trajectory goes through
Harbor's own mini-swe-agent -> ATIF converter (DEPEND) and ``atif_import``; then the run's facts are appended:
``final_patch`` (files and lines the submission changed), one ``check`` event per graded gate (pass / fail), and
the outcome from the verifier's reward -- or, for runs without a report, from the calibration record the pack
keeps for every seed task (``seeds.SEEDS[...]["frontier"]``). Runs with no recorded result stay ``unknown``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from crucible.connectors.base import RawRecord, SourceRef
from crucible.core.schema.trace import Trace, TraceEvent, TraceOutcome, TraceSource
from crucible.interop.harbor.atif_import import convert_mini_swe_agent, trajectory_to_trace

FAMILY = "coding/agent_run"


def patch_stats(text: str) -> dict[str, Any]:
    files = re.findall(r"^diff --git a/(\S+) b/", text, re.MULTILINE)
    added = sum(1 for ln in text.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in text.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    return {"files": len(files), "paths": files[:200], "lines_added": added, "lines_removed": removed, "bytes": len(text)}


class AgentRunConnector:
    name = "agent_runs"

    def __init__(self, root: Path):
        self.root = Path(root)

    def discover(self, scope: dict[str, Any] | None = None) -> Iterator[SourceRef]:
        for traj in sorted(self.root.glob("*/runs/*/trajectory.json")):
            run_dir = traj.parent
            task, run = run_dir.parent.parent.name, run_dir.name
            yield SourceRef(connector=self.name, key=f"{task}/{run}", meta={"dir": str(run_dir), "task": task, "run": run})

    def fetch(self, ref: SourceRef) -> RawRecord:
        d = Path(ref.meta["dir"])
        payload: dict[str, Any] = {"trajectory": json.loads((d / "trajectory.json").read_text())}
        for name, key in (("grade_report.json", "grade_report"), ("RESULT.md", "result_md"), ("model.patch", "patch")):
            p = d / name
            if p.exists():
                payload[key] = json.loads(p.read_text()) if name.endswith(".json") else p.read_text(errors="replace")
        payload["recorded"] = _recorded(ref.meta["task"], ref.meta["run"])
        return RawRecord(ref=ref, payload=payload, privacy="internal")

    def normalize(self, raw: RawRecord) -> Trace:
        p, task, run = raw.payload, raw.ref.meta["task"], raw.ref.meta["run"]
        traj = convert_mini_swe_agent(p["trajectory"], session_id=f"{task}/{run}")
        trace = trajectory_to_trace(traj, trace_id=f"run:{task}/{run}", family=FAMILY,
                                    source=TraceSource(connector=self.name, ref=raw.ref.key, privacy="internal"))
        events = list(trace.events)
        last_time = next((e.time for e in reversed(events) if e.time), None)

        def add(**kw: Any) -> None:
            events.append(TraceEvent(seq=len(events), time=last_time, **kw))

        if p.get("patch") is not None:
            st = patch_stats(p["patch"])
            add(kind="state", actor="agent", type="final_patch", text=f"{st['files']} files, +{st['lines_added']} -{st['lines_removed']}",
                data={**st, "writes": [f"file:{x}" for x in st["paths"]]})
        report = p.get("grade_report") or {}
        for gate, res in (report.get("gates") or {}).items():
            ok = bool(res.get("pass")) if isinstance(res, dict) else bool(res)
            add(kind="check", actor="system", type="grade_gate", text=f"{gate}: {'pass' if ok else 'fail'}",
                data={"gate": gate, "pass": ok, **({"summary": res.get("summary")} if isinstance(res, dict) and res.get("summary") else {})})
        details = {**trace.outcome.details, "task": task, "run": run, "info": {k: v for k, v in (p["trajectory"].get("info") or {}).items() if k != "config"}}
        reward = report.get("reward")
        rec = p.get("recorded")
        if reward is None and rec:
            reward = rec["score"]
            details["failing_gates"] = rec.get("failing_gates", [])
        if reward is None and p.get("result_md"):
            m = re.search(r"\*\*Reward:\s*([0-9.]+)", p["result_md"])
            reward = float(m.group(1)) if m else None
        if reward is None:
            outcome = TraceOutcome(label="unknown", label_source="no recorded verifier result for this run", details=details)
        else:
            outcome = TraceOutcome(label="good" if float(reward) >= 1.0 else "bad", label_source=f"hidden verifier reward {reward}",
                                   details={**details, "reward": float(reward)})
        return Trace(trace_id=trace.trace_id, family=FAMILY, source=trace.source, actors=trace.actors, events=events, outcome=outcome)


def _recorded(task: str, run: str) -> dict[str, Any] | None:
    from crucible_pack_coding.seeds import SEEDS

    for f in SEEDS.get(task, {}).get("frontier", []):
        if f.run_ref.rstrip("/").endswith(f"runs/{run}"):
            return {"score": f.score, "model": f.model, "failing_gates": list(f.failing_gates or []), "task_version": f.task_version}
    return None
