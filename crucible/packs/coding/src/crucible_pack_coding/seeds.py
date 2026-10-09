"""Generate the seed overlays for tasks 01-04 (run: ``python -m crucible_pack_coding.seeds``).

Everything here is read from the task directories themselves (``task.toml``, ``repos/``, the
``[[verifier.collect]]`` command, ``runs/*/RESULT.md``) plus the calibration record in each
MIGRATION_LOG, so regenerating is deterministic. The overlay never copies the task: ``harbor_dir``
points at it, and ``load_crucible_task`` validates it with Harbor's own models.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from crucible.core.schema.task import CrucibleTask, FrontierResult, Grounding, Splits, TaskCard, ValueProof
from crucible.core.schema.world import WorldSpec
from crucible.interop.harbor.task_io import load_harbor_task
from crucible_pack_coding import SEEDS_ROOT

MIGRATION_TASKS = SEEDS_ROOT.parents[5] / "migration_tasks"  # <repo>/migration_tasks

# Facts not derivable from files: images, GPU archs, horizons, and calibration (from MIGRATION_LOG / RESULT.md).
SEEDS = {
    "task01_prims_omp": {
        "image": "ahmedmohsin/task01-prims-omp:v1", "arch": "sm_89", "horizon": "H3", "version": "v1", "title": "tsr-prims: CUDA -> OpenMP target offload",
        "frontier": [FrontierResult(model="fable-5.1", harness="mini-swe-agent 2.4.6", task_version="v1", score=1.0, passed=True, steps=200, cost_usd=50.46,
                                    run_ref="runs/fable-5.1-run1", note="solved on first attempt -> superseded by task01v2")],
        "bench": False,
    },
    "task01v2_tessera_omp": {
        "image": "ahmedmohsin/task01v2-tessera-omp:v1", "arch": "sm_89", "horizon": "H4", "version": "v2", "title": "Tessera renderer: CUDA -> OpenMP target offload",
        "frontier": [FrontierResult(model="gpt-5.6-sol", harness="mini-swe-agent 2.4.6", task_version="v2", score=0.0, passed=False, steps=95, cost_usd=3.03,
                                    run_ref="runs/gpt-5.6-sol-run1")],
        "bench": True,
    },
    "task02_pcops_triton": {
        "image": "task02-pcops-triton:v1", "arch": "sm_120", "horizon": "H4", "version": "v1", "title": "pcops CUDA extension -> Triton",
        "frontier": [FrontierResult(model="gpt-5.6-sol", harness="mini-swe-agent 2.4.6", task_version="v1", score=0.0, passed=False, steps=88, cost_usd=2.29,
                                    run_ref="runs/gpt-5.6-sol-run1")],
        "bench": True,
    },
    "task03_mdlite_pallas": {
        "image": "task03-mdlite-pallas:v1", "arch": "sm_120", "horizon": "H4", "version": "v1", "title": "mdlite MD engine: CUDA C++ -> JAX + Pallas",
        "frontier": [FrontierResult(model="gpt-5.6-sol", harness="mini-swe-agent 2.4.6", task_version="v1", score=0.0, passed=False, steps=81, cost_usd=2.96,
                                    run_ref="runs/gpt-5.6-sol-run1")],
        "bench": True,
    },
    "task04_pcl_gpu_cuda12": {
        "image": "task04-pcl-gpu-cuda12:v1", "arch": "sm_120", "horizon": "H4", "version": "v3", "title": "PCL GPU modules: legacy CUDA -> CUDA 12.8 (speed + concurrency bars)",
        "frontier": [
            FrontierResult(model="gpt-6-astra", harness="mini-swe-agent 2.4.6", task_version="v1", score=1.0, passed=True, steps=71, cost_usd=11.08, run_ref="runs/gpt-6-astra-run1"),
            FrontierResult(model="gpt-5.6-sol", harness="mini-swe-agent 2.4.6", task_version="v1", score=1.0, passed=True, steps=128, cost_usd=3.63, run_ref="runs/gpt-5.6-sol-run1"),
            FrontierResult(model="gpt-6-astra", harness="mini-swe-agent 2.4.6", task_version="v3", score=0.0, passed=False, steps=134, cost_usd=39.75,
                           failing_gates=["perf/kinfu_frame", "perf/kinfuls_frame", "perf/concurrent_trackers"], run_ref="runs/gpt-6-astra-v3-run2"),
            FrontierResult(model="gpt-5.6-sol", harness="mini-swe-agent 2.4.6", task_version="v3", score=0.0, passed=False, steps=99, cost_usd=3.08,
                           failing_gates=["correctness/ncv", "fresh"], run_ref="runs/gpt-5.6-sol-v3-run1"),
        ],
        "bench": True,
    },
}

DERIVED = ["**/build*/**", "build*/**", "install*/**", "**/__pycache__/**", "*.egg-info/**", "**/*.egg-info/**"]


def _collect_repo(collect: str) -> str | None:
    m = re.search(r"cd\s+(/\S+)\s*&&", collect)
    return m.group(1) if m else None


def build_seed(task_id: str) -> Path:
    info = SEEDS[task_id]
    task_dir = MIGRATION_TASKS / task_id
    h = load_harbor_task(task_dir, require_tests=True)
    main_repo = _collect_repo(h.summary.collect_commands[0]) if h.summary.collect_commands else None
    if (task_dir / "repos").is_dir():
        repos = sorted(p.name for p in (task_dir / "repos").iterdir() if p.is_dir())
        if main_repo and Path(main_repo).name not in repos:
            repos = [Path(main_repo).name, *repos]
        roots = {r: f"/workspace/{r}" for r in repos}
    else:
        roots = {"app": main_repo or "/app"}
    main_id = Path(main_repo).name if main_repo else next(iter(roots))
    apps = []
    for rid, root in roots.items():
        apps.append({
            "id": rid,
            "template": "coding/git_repo",
            "state": {"adapter": "fs_git", "params": {"root": root, "exclude": DERIVED}},
            "params": {"shell": rid == main_id},
            "description": "the repository to migrate" if rid == main_id else "consumer repository (must keep working unmodified)",
        })
    apps.append({"id": "gpu", "template": "coding/gpu_toolchain", "state": {"adapter": "process", "params": {"cwd": roots[main_id]}},
                 "resources": {"gpus": 1, "gpu_arch": info["arch"], "cpus": h.summary.cpus, "memory_mb": h.summary.memory_mb}})
    overlay = SEEDS_ROOT / task_id
    rel_task = os.path.relpath(task_dir, overlay)
    world = WorldSpec.model_validate({
        "world": f"coding/legacy_gpu_migration/{task_id}",
        "description": info["title"],
        "backend": "harbor",
        "backend_params": {"task_dir": rel_task, "docker_image": info["image"], "gpu_host": "lab5090"},
        "network": "no-network",
        "apps": apps,
        "actors": [{"id": "agent", "kind": "policy", "tools": [f"{r}.*" for r in roots]}],
    })
    tid = f"coding/legacy_gpu_migration/{task_id}@{info['version']}"
    task = CrucibleTask(id=tid, family="legacy_gpu_migration", horizon=info["horizon"], harbor_dir=rel_task, milestones=None)
    card = TaskCard(
        id=tid, family="legacy_gpu_migration", horizon=info["horizon"], title=info["title"],
        grounding=Grounding(traces=[f"{task_id}/MIGRATION_LOG.md"], oracle_source="builder_oracle"),
        value_proof=ValueProof(frontier=info["frontier"]),
        splits=Splits(showcase=True, bench=False, train=False),
        history=[{"note": "imported as an MVP showcase seed (DESIGN §12.1); never a published OOD split"}],
    )
    (overlay / "crucible").mkdir(parents=True, exist_ok=True)
    world.dump(overlay / "crucible" / "world.yaml")
    task.dump(overlay / "crucible" / "task.yaml")
    card.dump(overlay / "crucible" / "card.yaml")
    return overlay


def main() -> None:
    for t in SEEDS:
        print(build_seed(t))


if __name__ == "__main__":
    main()
