"""Studio hooks of the coding pack (declared in ``pack.yaml``). Live grading needs the seeds' GPU verifier images, so
the hooks here are the CPU parts: dense credit for recorded agent runs, and fixtures over the M6 dense reward.

Per-step Φ is not derivable for the recorded frontier runs: their agents wrote files with python and ``sed``, which the
bash edit replayer cannot follow (0–12 replayable states out of 46–200 calls per run, measured 2026-10-11). Credit is the
final patch's Φ, placed at the run's last event. Crucible's own World runs record state at every step and do not have
this limit.
"""

from __future__ import annotations

from typing import Any

SEEDS = ("task01_prims_omp", "task02_pcops_triton", "task03_mdlite_pallas", "task04_pcl_gpu_cuda12")


async def _mine(task: str):
    from crucible.workflow.filestate import FileState
    from crucible.workflow.milestones.evaluate import FileStateView, evaluate
    from crucible_pack_coding.m56.task_properties import TASKS, TEST, base_reader, from_diff

    base = base_reader(task)
    oracle_patch = (TASKS / task / "solution" / "solution.patch").read_text(errors="replace")
    cands = [c.milestone for c in from_diff(oracle_patch, base, include=lambda p: not TEST.search(p), max_per_file=6)]
    oracle = FileState(base=base)
    oracle.apply_patch(oracle_patch)
    at_base = await evaluate(cands, FileStateView(FileState(base=base)))
    at_oracle = await evaluate(cands, FileStateView(oracle))
    return [m for m in cands if at_base[m.id] == 0.0 and at_oracle[m.id] == 1.0]


async def _phi(task: str, patch: str | None) -> float:
    from crucible.credit.operators.potential import potential
    from crucible.workflow.filestate import FileState
    from crucible.workflow.milestones.evaluate import FileStateView, evaluate
    from crucible_pack_coding.m56.task_properties import base_reader

    ms = await _mine_cached(task)
    fs = FileState(base=base_reader(task))
    if patch:
        fs.apply_patch(patch)
    return potential(ms, await evaluate(ms, FileStateView(fs)))[0]


_MS: dict[str, Any] = {}


async def _mine_cached(task: str):
    if task not in _MS:
        _MS[task] = await _mine(task)
    return _MS[task]


async def credit(traces: list) -> dict[str, dict[str, Any]]:
    """For each recorded agent run: Φ of its final patch over the seed's oracle-derived milestones, the recorded outcome,
    and the composed dense reward. Other families (pull requests) carry no task-relative Φ here."""
    from crucible.credit.compose import compose
    from crucible_pack_coding.m56.task_properties import TASKS, recorded_outcome

    out = {}
    for t in traces:
        if t.family != "coding/agent_run" or not t.trace_id.startswith("run:"):
            continue
        task, run = t.trace_id[4:].split("/", 1)
        rd = TASKS / task / "runs" / run
        patch = (rd / "model.patch").read_text(errors="replace") if (rd / "model.patch").exists() else None
        y = recorded_outcome(task, rd)
        try:
            phi = await _phi(task, patch)
        except Exception as e:  # noqa: BLE001 - a patch that does not apply to the base is reported, not guessed
            out[t.trace_id] = {"reward": None, "error": f"{type(e).__name__}: {str(e)[:160]}"}
            continue
        last = max((e.seq for e in t.events), default=0)
        reward = compose(gates={}, outcome=bool(y), phi=phi).reward if y is not None else None
        out[t.trace_id] = {"reward": reward, "phi_by_seq": {last: round(phi, 6)}, "phi_final": round(phi, 6),
                           "outcome": None if y is None else ("good" if y >= 1 else "bad")}
    return out


async def fixtures() -> list[dict]:
    """CPU fixtures over the M6 dense reward of every seed: the oracle patch scores 1, the untouched base 0 (R1, R2)."""
    from crucible.credit.compose import compose
    from crucible_pack_coding.m56.task_properties import TASKS

    cases = []
    for task in SEEDS:
        oracle = (TASKS / task / "solution" / "solution.patch").read_text(errors="replace")

        async def g_oracle(task=task, oracle=oracle):
            phi = await _phi(task, oracle)
            return compose(gates={}, outcome=True, phi=phi).reward, {"outcome": "pass" if phi >= 1.0 else "fail"}

        async def g_base(task=task):
            phi = await _phi(task, None)
            return compose(gates={}, outcome=False, phi=phi).reward, {"outcome": "fail" if phi == 0.0 else "pass"}

        cases.append({"name": f"{task}:oracle", "grade": g_oracle, "reward": 1.0})
        cases.append({"name": f"{task}:base", "grade": g_base, "reward": 0.0, "fails_gate": "outcome"})
    return cases
