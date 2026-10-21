"""RL post-training in W&B (swarm-pipeline/11 §4.2).

``train_run(config, ...)`` opens the ``crucible-train`` run that ``TinkerTrainer`` logs into as one of its sinks (per-step
metrics: ``trainer.step_metrics``; drops as records). ``finish_train(run, summary, run_dir)`` adds what exists only at
the end: the trained episodes as Weave traces and a W&B table with their links, the artifacts (config, receipt,
``steps.jsonl``, ``drops.jsonl``, ``spend.jsonl``, ``versions.json``; the adapters' ``tinker://`` paths in the artifact
metadata, since W&B cannot fetch them), and the summary, then finishes the run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.observe.naming import manifest_hash, project, run_name, tags, train_name
from crucible.observe.run import ObserveConfig, ObsRun
from crucible.observe.traces import EPISODE_COLUMNS, TraceStore, episode_rows


def train_run(config: Mapping[str, Any], *, run_dir: Path | str, objective_id: str, tasks: Sequence[str], harness: str | None = None,
              task_meta: Mapping[str, Mapping[str, Any]] | None = None, cfg: ObserveConfig | None = None, extra: Mapping[str, Any] | None = None) -> ObsRun:
    """``config``: ``TrainConfig`` as a dict (model, LoRA rank, loss, group size, caps, thinking, seed, ...)."""
    run_dir = Path(run_dir)
    model = str(config.get("model") or "")
    mh = manifest_hash(tasks)
    langs = sorted({str((task_meta or {}).get(t, {}).get("language")) for t in tasks if (task_meta or {}).get(t, {}).get("language")})
    full = {"kind": "train", **dict(config), "objective_id": objective_id, "tasks_manifest_hash": mh, "n_tasks": len(tasks), "tasks": list(tasks),
            "harness": harness, "run_dir": str(run_dir), **dict(extra or {})}
    return ObsRun(project("train"), train_name(model, run_dir.name), group=f"objective-{str(objective_id)[:16]}", job_type="rl", cfg=cfg, config=full,
                  tags=tags(model=model, loss=config.get("loss_fn"), rank=config.get("lora_rank"), group=config.get("group_size"),
                            thinking=config.get("enable_thinking"), lang=langs))


def finish_train(run: ObsRun, summary: Mapping[str, Any], run_dir: Path | str, *, traces: bool = True, status: str | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = run.cfg
    n = 0
    if traces and (run_dir / "rollouts").exists():
        store = TraceStore(cfg.root, run.project, scrubber=run.s)
        rows = store.export_all(run_dir / "rollouts", arm="train", run=run.name, model=summary.get("model"), max_chars=cfg.max_chars)
        n = len(rows)
        if rows:
            run.table("train/episodes", EPISODE_COLUMNS, episode_rows([store.index()[r["episode"]] for r in rows]))
            run.artifact(run_name("episodes", run.name), "episode-traces", paths=store.full_paths([r["episode"] for r in rows]),
                         metadata={"project": run.project, "episodes": n})
    steps = [json.loads(x) for x in (run_dir / "steps.jsonl").read_text().splitlines() if x.strip()] if (run_dir / "steps.jsonl").exists() else []
    if steps:
        run.table("train/steps", ["step", "reward_mean", "loss", "kl_sample_vs_train", "train_tokens", "trained_tokens", "usd_cum", "sampler_path", "tasks"],
                  [[s.get("step"), s.get("reward_mean"), s.get("loss"), s.get("kl_sample_vs_train"), s.get("train_tokens"), s.get("trained_tokens"),
                    s.get("usd_cum"), s.get("sampler_path"), s.get("tasks")] for s in steps])
    versions = summary.get("versions") or (json.loads((run_dir / "versions.json").read_text()) if (run_dir / "versions.json").exists() else {})
    run.artifact(run_name("train-run", run.name), "train-run",
                 paths={p.name: p for p in (run_dir / f for f in ("config.json", "receipt.json", "steps.jsonl", "drops.jsonl", "spend.jsonl", "versions.json",
                                                                    "state.json", "summary.json", "rewards.jsonl", "quarantine.jsonl")) if p.exists()},
                 metadata={"sampler_path": summary.get("sampler_path"), "state_path": summary.get("state_path"), "versions": versions,
                           "objective_id": summary.get("objective_id"), "model": summary.get("model")})
    if summary.get("sampler_path"):
        run.artifact(run_name("adapter", run.name), "adapter-ref", files={"adapter.json": {"sampler_path": summary.get("sampler_path"),
                                                                                           "state_path": summary.get("state_path"), "model": summary.get("model"),
                                                                                           "steps_done": summary.get("steps_done")}},
                     metadata={"tinker_path": summary.get("sampler_path")}, aliases=["latest"])
    run.summary({k: v for k, v in summary.items() if k not in ("steps_log", "versions")})
    run.finish(status or ("ok" if summary.get("stop") in ("done", None) else str(summary.get("stop"))))
    return {"run": str(run.dir), "sync": [str(p) for p in run.sync_dirs], "traces": n}
