#!/usr/bin/env python
"""S15 targeting of the kernel lanes on a GPU host (swarm-pipeline/10 §9): the debug lane's targeting phase
(``lh_debug_target.py``: the same stages, meter, keep rule, trace audit and presets) over kernel tasks whose episodes run
on the GPU host (``factory.lanes.kernel_episode``: the World and every grade on its GPU).

    set -a; . ../.env; set +a; .venv/bin/python <worktree>/crucible/scripts/lh_kernel_target.py student frontier \
        --host rtx5090 --tasks kd-bug-llamacpp-ca21f3a1 --k 8 --frontier-k 2 --cap 150
    ... audit mine report [--preset swarm]
    ... cleanup                     # this phase's containers on the GPU host (crucible-lh-k-ep-*) and nothing else

Tasks: ``--tasks`` (default: every kernel task the lane's funnel lists as passing every GPU gate on the host's GPU,
``docs/research/swarm-pipeline/runs/kernel_debug_funnel.json``). Kernel records come from ``--kernel`` (the main checkout's
``data/kernel_debug``). Every row of ``report.json`` carries the host, the GPU and the GPU-minutes of its grades.

What differs from the debug lane: no ``worldfix``, ``disclose``, ``regrade`` or ``fairness`` stage (S9b's alternatives and
second verifier have no CUDA writer yet: 10 §9.3); the honest-trace checks protect the grader's op test, not a test tree;
a failure's skill is the kernel intent's (``kernel_debug.task_intent``), on intent when the judges label it a skill gap.

Host outages (``world.hostreach``, 2026-10-26): an episode pauses while the GPU host is unreachable and resumes when it is
back; one still down after the pause budget is masked ``host_unreachable``. No job starts while the host is down
(``kernel_episode.launch_gate``: skipped with that reason, picked up by the next run of the stage), and the host's events
go to ``<out>/journal/hosts.jsonl``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lh_debug_target as T

PREFIX = "crucible-lh-k-ep-"
PILOT_TASK = "kd-bug-llamacpp-ca21f3a1"
FUNNEL = Path(__file__).resolve().parents[1] / "docs" / "research" / "swarm-pipeline" / "runs" / "kernel_debug_funnel.json"


def gpu_passing(funnel: Path, gpu: str) -> list[str]:
    """Kernel tasks the funnel lists as passing every GPU gate on ``gpu`` (``gpu_by_hardware``), else on the task's own
    calibration GPU (``gpu_pass``, recorded with ``gpu``)."""
    doc = json.loads(funnel.read_text()) if funnel.exists() else {}
    out = []
    for t in doc.get("tasks") or []:
        hw = (t.get("gpu_by_hardware") or {}).get(gpu) or {}
        if hw.get("pass") or (t.get("gpu_pass") and str(t.get("gpu") or "") in ("", gpu)):
            out.append(t["id"])
    return sorted(out)


class KernelTasks:
    """``lh_debug_target.Tasks`` for kernel tasks: ``rows`` (the fields ``task_record`` reads) and ``task`` (an episode task
    on the host)."""

    def __init__(self, root: Path, host: Any, ids: list[str]):
        from crucible.factory.lanes import kernel_episode as ke

        self.root, self.host, self.ke = root, host, ke
        self._tasks: dict[str, Any] = {}
        self.rows: dict[str, dict[str, Any]] = {}
        for tid in ids:
            rec = ke.load_record(root, tid)
            t = self.task_of(rec)
            self.rows[tid] = {"id": tid, "split": t.split, "lang": "cuda", "repo": rec["repo"], "faults": len(rec["milestones"]),
                              "level": rec.get("family"), "skills": t.intent.split("+") if t.intent else []}

    def task_of(self, rec: dict[str, Any]) -> Any:
        tid = str(rec["id"])
        if tid not in self._tasks:
            self._tasks[tid] = self.ke.KernelEpisodeTask(rec, self.host)
        return self._tasks[tid]

    def task(self, tid: str) -> Any:
        return self._tasks[tid]

    def gen(self, split: str) -> Any:  # the debug lane's generator: no kernel equivalent is needed by the stages used here
        raise NotImplementedError("kernel targeting has no debug-lane generator")


class KernelPhase(T.Phase):
    def __init__(self, a: argparse.Namespace):
        from crucible.factory.lanes import kernel_episode as ke

        self.host = ke.host_of(a.host)
        ids = [t for t in a.tasks.split(",") if t] or gpu_passing(Path(a.funnel), self.host.gpu)
        self._ids = ids
        orig = T.Tasks
        T.Tasks = lambda *_a, **_k: KernelTasks(Path(a.kernel), self.host, ids)  # the base Phase builds its tasks here
        try:
            super().__init__(a)
        finally:
            T.Tasks = orig
        self.slots = T.Slots(min(a.slots, self.host.max_worlds))
        from crucible.factory.swarm_journal import SwarmJournal

        # the GPU host's outages (down, each paused operation, up, held launches): <out>/journal/hosts.jsonl, read by
        # swarm_journal.tables (``scripts/swarm.py diagnose``); each episode's own outages are in its record (meta.host_outages)
        self.journal = SwarmJournal(self.out / "journal")

    def settings(self) -> dict[str, Any]:
        s = super().settings()
        h = self.host
        s.update({"world": f"GPU World on {h.name} ({h.gpu}, sm_{h.arch}): --runtime=nvidia, --network none, {h.world_cpus} CPU / {h.world_memory}, "
                           f"pre-built A' in /ws/build", "containers": f"{PREFIX}* on {h.name}", "gpu_host": h.name, "gpu": h.gpu, "arch": h.arch})
        return s

    async def job(self, model: str, tid: str, *, k: int, seed: int, purpose: str, concurrency: int, tag: str) -> dict[str, Any]:
        from crucible.factory.lanes import kernel_episode as ke

        done = T.jload(self.run_dir(tag, tid) / "done.json")
        if not (done and done.get("k") == k) and not await ke.launch_gate(self.host):
            # fleet stop (2026-10-26): the GPU host has been unreachable for the whole pause budget: no job starts (no World, no
            # model call, nothing reserved on the meter); a later run of the stage picks it up
            why = f"host_unreachable: {self.host.name} did not answer within the pause budget; not started"
            self.journal.host({"event": "launch_held", "host": self.host.name, "why": why, "task": tid, "arm": tag})
            print(f"[{tag}] {tid}: {why}", flush=True)
            return {"task": tid, "arm": tag, "skipped": why}
        t0 = time.time()
        row = await super().job(model, tid, k=k, seed=seed, purpose=purpose, concurrency=concurrency, tag=tag)
        task = self.tasks.task(tid)
        grades = [g for g in task.backend.grades if g["at"] >= t0 and g["task"] == tid]
        if grades and "metrics" in row:
            secs = sum(float(g.get("grade_s") or g["seconds"]) for g in grades)
            row["gpu"] = {"host": self.host.name, "gpu": self.host.gpu, "grades": len(grades), "grade_minutes": round(secs / 60, 1),
                          "grade_minutes_per_episode": round(secs / 60 / max(1, k), 1), "world_minutes": round(row.get("wall_s", 0) / 60 * min(k, concurrency), 1)}
            T.jsave(self.run_dir(tag, tid) / "done.json", row)
        return row


def kernel_honest_of(task: Any, b: Any) -> list[Any]:
    """The honest-trace checks of a kernel episode: the grader's own op test is the protected path, nothing is hidden on
    disk (the grader's cases live in its reference build, never in the World)."""
    import re

    from crucible.factory.lanes import kernel_harness as kh
    from crucible.factory.traceaudit import TEST_RUN, ToolCall, commands_of, honest_trace_facts

    fs = honest_trace_facts(b, requirement_milestones={r["id"]: r.get("milestone") for r in task.requirements}, protected=lambda p: p == kh.HARNESS,
                            hidden_paths=[])
    first_edit = next((st.i for st in b.steps if re.search(r"write|edit|create|replace|patch", st.function, re.IGNORECASE)), None)
    gpu_run = re.compile(r"test-backend-ops|compute-sanitizer|\./build/bin/|nvidia-smi|\bmake\b.*run|\./[\w./-]+")
    ran = [st.i for st in b.steps for _, c in commands_of([ToolCall(st.function, st.arguments)]) if TEST_RUN.search(c) or gpu_run.search(c)]
    b.meta["ran_before_edit"] = bool(ran and (first_edit is None or min(ran) < first_edit))
    b.meta["test_runs"] = len(ran)
    return fs


def kernel_failure_skill(finding: dict[str, Any], intent: dict[str, Any], *, ran_before_edit: bool | None = None, honest: Any = None) -> dict[str, Any]:
    """A kernel failure's skill: the task intent's first skill, on intent when the judges labelled it a skill gap (the
    debug lane's mode-to-skill map has no kernel rows yet: coarser, recorded as such)."""
    skills = list((intent or {}).get("skills") or [])
    on = finding.get("cause") == "skill_gap"
    return {"skill": skills[0] if skills else None, "on_intent": on, "source": "kernel intent (coarse: cause label only)", "ran_before_edit": ran_before_edit}


def kernel_cleanup(host: Any) -> None:
    from crucible.factory.lanes import kernel_episode as ke

    print(f"removed {ke.cleanup(host)}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    import sys as _s

    argv = list(_s.argv[1:] if argv is None else argv)
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)  # never read --k as --kernel
    extra.add_argument("--host", default="rtx5090")
    extra.add_argument("--kernel", default="", help="the kernel lanes' data dir (default <data>/kernel_debug)")
    extra.add_argument("--funnel", default=str(FUNNEL))
    extra.add_argument("--env-file", default="", help="KEY=VALUE file loaded into the environment first (values never printed; swarm.load_env_file)")
    extra.add_argument("--no-preflight", action="store_true", help="skip grading each task's oracle and empty patch before paid episodes")
    k, rest = extra.parse_known_args(argv)
    if k.env_file:
        from swarm import load_env_file

        load_env_file(k.env_file)
    if "--pilot-task" not in rest:
        rest += ["--pilot-task", PILOT_TASK]
    if "--out" not in rest:
        rest += ["--out", str(T.data_dir() / "targeting" / f"kernel-{k.host}")]
    a = T.parse_args(rest)
    a.host, a.kernel, a.funnel = k.host, k.kernel or str(T.data_dir() / "kernel_debug"), k.funnel
    a.no_preflight = k.no_preflight
    return a


async def preflight(ph: Any, tids: list[str]) -> list[str]:
    """Before any paid episode, every task's endpoints are graded the way an episode is (``lane_grade`` on the GPU host):
    the oracle a full pass, the empty patch no milestone. A task whose grader is broken or masked is left out of the paid
    stages, so a grader regression costs two GPU grades, not 8 masked episodes (2026-10-25: an ImportError, then a
    TypeError in the episode grader masked all 8 episodes of 1b8b8fb1, 10.6 USD)."""
    from crucible.factory.lanes import kernel_debug as kd

    ok = []
    for tid in tids:
        task = ph.tasks.task(tid)
        bad = kd.report_gate(task.rec)
        if bad:  # S7k: the report against the graded milestones (V5, disclosure): re-author or revise before measuring
            print(f"[preflight] {tid}: REFUSED report gate {bad}", flush=True)
            continue
        oracle = str(task.rec.get("oracle_patch") or "")
        got = {}
        for name, patch in (("oracle", oracle), ("empty", "")):
            rep = await task.lane_grade(patch)
            got[name] = (rep.infra.get("status"), dict((rep.meta or {}).get("milestones") or {}))
        (os_, om), (es, em) = got["oracle"], got["empty"]
        good = bool(oracle) and os_ == "ok" and es == "ok" and bool(om) and all(om.values()) and not any(em.values())
        print(f"[preflight] {tid}: {'ok' if good else 'REFUSED'} oracle {os_} {om} empty {es} {em}", flush=True)
        if good:
            ok.append(tid)
    return ok


async def amain(a: argparse.Namespace) -> None:
    unsupported = [s for s in a.stages if s in ("worldfix", "disclose", "regrade", "fairness", "diag")]
    if unsupported:
        raise SystemExit(f"not a kernel targeting stage: {unsupported} (10 §9.3)")
    if any(s in a.stages for s in ("pilot", "student", "frontier", "arms")) and not T.os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment (set -a; . ../.env; set +a)")
    import crucible.factory.generators.debug as gd
    from crucible.factory.lanes import kernel_debug as kd

    T.PREFIX = PREFIX
    T.honest_of = kernel_honest_of  # the audit stage's honest-trace checks
    gd.failure_skill, gd.task_intent = kernel_failure_skill, kd.task_intent  # the mine stage's skill of a kernel failure (this process only)
    from crucible.world import hostreach

    ph = KernelPhase(a)
    unlisten = hostreach.listen(ph.journal.host)
    tids = sorted(ph.tasks.rows)
    print(f"[kernel] host {ph.host.name} ({ph.host.gpu}); tasks {tids}", flush=True)
    if any(s in a.stages for s in ("pilot", "student", "frontier", "arms")) and not a.no_preflight:
        tids = await preflight(ph, tids)
        if not tids:
            raise SystemExit("[preflight] no task's endpoints grade as expected: nothing to measure")
    try:
        for st in a.stages:
            if st == "pilot":
                await T.stage_pilot(ph)
            elif st in ("student", "frontier", "arms"):
                await T.stage_arms(ph, ("student", "frontier") if st == "arms" else (st,), tids)
            elif st == "audit":
                await T.stage_audit(ph, tids)
            elif st == "mine":
                T.stage_mine(ph, tids)
            elif st == "report":
                r = T.stage_report(ph, tids)
                print(json.dumps({"verdicts": r["verdicts"], "spend_total": r["spend_total"], "spend_by_model": r["spend_by_model"]}, indent=1), flush=True)
            elif st == "summary":
                T.stage_summary(ph, docs=Path(a.docs) if a.docs else None)
            elif st == "cleanup":
                kernel_cleanup(ph.host)
            else:
                raise SystemExit(f"unknown stage {st}")
    finally:
        unlisten()
        print(f"spent {ph.meter.spent:.4f} of {ph.meter.cap_usd}", flush=True)


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
