"""M13 exit gate, coding part, from RECORDED frontier runs: a step-level report on tasks 01–04 (each task version is its
own item) and the task04 span tasks, built from the runs already graded. No new spend.

Sources, all recorded:
- ``seeds.SEEDS`` frontier results (model, version, score, steps, cost, failing gates);
- ``migration_tasks/<task>/runs/<run>/`` for runs SEEDS does not list. A run with no grade (an API error or an aborted
  attempt) is **masked**: the infrastructure, not the model, ended it;
- grade reports (tasks 01–03, ``grade_report.json``) and the task04 grade history (``data/m7/history/task04/glogs/*``)
  for gate profiles;
- the M8 GPU span grades (``m8_coding_gpu.json``): ``trap_sol_v3`` is GPT-5.6 Sol's v3 code on the people and NCV spans.

Φ per run is the end-state Φ of its final patch over the seed's oracle-derived milestones (M6 C1). Per-step progress is
not reconstructable for these runs (their agents edited files through python and ``sed``), so the report marks progress
as **not tracked** rather than drawing a curve it does not have.

The gate also asks for **an open model**. No recorded run used one, so that part needs a live run on the GPU Worlds.

    python -m crucible_pack_coding.bench_recorded
"""

from __future__ import annotations

import asyncio
import json

from crucible.bench.irt import fit
from crucible.bench.metrics import Episode, summarize
from crucible.bench.report import write
from crucible.bench.run import irt_counts
from crucible_pack_coding import PACK_ROOT
from crucible_pack_coding.seeds import SEEDS

CRUCIBLE = PACK_ROOT.parent.parent
TASKS = CRUCIBLE.parent / "migration_tasks"
HISTORY = CRUCIBLE / "data" / "m7" / "history" / "task04" / "glogs"
OUT = CRUCIBLE / "docs" / "modules" / "reports" / "m13_coding_recorded"
HARNESS = "mini-swe-agent@2.4.6"
T04_HISTORY = {"gpt-6-astra-run1": "astra2", "gpt-5.6-sol-run1": "sol2", "gpt-6-astra-v3-run2": "astra_v3b", "gpt-5.6-sol-v3-run1": "sol_v3",
               "gpt-6-astra-v2-run1": "astra_v2_partial"}
MODEL = {"fable-5.1": "claude-fable-5.1", "gpt-5.6-sol": "gpt-5.6-sol", "gpt-6-astra": "gpt-6-astra"}


def _gates_of(task: str, run: str) -> dict[str, int]:
    rd = TASKS / task / "runs" / run
    g = rd / "grade_report.json"
    if g.exists():
        return {k: int(bool(v.get("pass"))) for k, v in json.loads(g.read_text()).get("gates", {}).items()}
    h = HISTORY / T04_HISTORY.get(run, "") / "verifier" / "report.json"
    if task.startswith("task04") and h.exists():
        return {k: int(bool(v.get("pass"))) for k, v in json.loads(h.read_text()).get("gates", {}).items()}
    return {}


def _stats(task: str, run: str) -> tuple[int, float]:
    t = TASKS / task / "runs" / run / "trajectory.json"
    if not t.exists():
        return 0, 0.0
    ms = (json.loads(t.read_text()).get("info") or {}).get("model_stats") or {}
    return int(ms.get("api_calls") or 0), float(ms.get("instance_cost") or 0.0)


async def episodes() -> list[Episode]:
    from crucible_pack_coding.studio import _phi

    eps: list[Episode] = []
    listed: set[tuple[str, str]] = set()
    for task, seed in SEEDS.items():
        for fr in seed["frontier"]:
            run = (fr.run_ref or "").split("/")[-1]
            listed.add((task, run))
            calls, cost = _stats(task, run)
            patch = TASKS / task / "runs" / run / "model.patch"
            try:
                phi = await _phi(task, patch.read_text(errors="replace") if patch.exists() else None)
            except Exception:  # noqa: BLE001 - a patch that does not apply leaves Φ unknown
                phi = None
            eps.append(Episode(task=f"{task}@{fr.task_version}", policy=MODEL.get(fr.model, fr.model), harness=HARNESS, sample_idx=0, attempt=f"{task}/{run}",
                               reward=fr.score, gates=_gates_of(task, run), steps=fr.steps or calls, cost_usd=fr.cost_usd or cost, tool_calls=calls,
                               phi_final=None if phi is None else round(phi, 6), progress_tracked=False))
    for rd in sorted(TASKS.glob("*/runs/*")):  # recorded attempts SEEDS does not list: no grade = masked (infra ended them)
        task, run = rd.parent.parent.name, rd.name
        if (task, run) in listed or not (rd / "trajectory.json").exists():
            continue
        calls, cost = _stats(task, run)
        model = next((v for k, v in MODEL.items() if run.startswith(k)), run)
        version = next((v for v in ("v3", "v2") if f"-{v}-" in run), "v1")
        eps.append(Episode(task=f"{task}@{version}", policy=model, harness=HARNESS, sample_idx=1, attempt=f"{task}/{run}", reward=None, steps=calls,
                           cost_usd=cost, tool_calls=calls, progress_tracked=False, tags=["masked:no_grade"]))
    spans = json.loads((CRUCIBLE / "docs" / "modules" / "reports" / "m8_coding_gpu.json").read_text())["grades"]
    for name, g in sorted(spans.items()):
        span, sol = name.split("/", 1)
        if sol == "trap_sol_v3" and g.get("reward") is not None:
            eps.append(Episode(task=f"task04_span/{span}@v3", policy="gpt-5.6-sol", harness=HARNESS, sample_idx=0, attempt=f"span/{name}", reward=float(g["reward"]),
                               gates={k: int(v) for k, v in (g.get("gates") or {}).items()}, progress_tracked=False))
    return eps


def main() -> dict:
    eps = asyncio.run(episodes())
    summary = summarize(eps, ks=(1,), horizon_of=lambda t: "span" if t.startswith("task04_span") else "H3/H4")
    counts = irt_counts(eps)
    extras = {"sources": "recorded frontier runs (SEEDS, run dirs, grade reports, task04 history, M8 span grades); no new spend",
              "open_model": "not run: no recorded open-model attempt; needs a live run on the GPU Worlds",
              "per_episode_phi": {e.attempt: e.phi_final for e in eps},
              "irt": fit(counts, model="1pl").to_json() if len({r for r, _ in counts}) >= 2 else None}
    paths = write(OUT, "Coding tasks 01-04 + spans, recorded frontier runs", summary, eps, extras=extras)
    by = {m["policy"]: m["overall"]["deterministic"] for m in summary["models"]}
    out = {"episodes": len(eps), "masked": sum(e.masked for e in eps), "models": {p: {"scored": d["coverage"]["measured"], "mean_reward": d.get("mean_reward"),
                                                                                   "phi_final": d.get("phi_final")} for p, d in by.items()}, "paths": paths}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
