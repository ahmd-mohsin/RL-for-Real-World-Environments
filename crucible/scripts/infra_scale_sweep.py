"""M12-infra C: scale sweeps (``docs/research/m12-infra/01-scale-sweeps.md``).

1. Control plane at scale: the discrete-event simulator drives the real ``Scheduler`` / ``DataPool`` with up to 2048
   engines (131k sequence slots) and 32k environment workers, per topology. Reports the scheduling outcome (engine
   utilization, trainer idle, staleness, wasted decode) and the control plane's own cost (wall seconds per simulated
   step: the scheduler's work per decision at that scale).
2. The staleness window at 1024 engines / 32k envs: τ ∈ {1, 2, 4, 8} against utilization and step time.
3. Placement: the planner on 8 to 1024 nodes for an 8B dense, a 70B dense and a 1T-total / 32B-active MoE model.

    uv run python scripts/infra_scale_sweep.py [--quick] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import time

from crucible.train.placement import TOPOLOGIES, ClusterSpec, ModelSpec, PlacementError, Workload, plan
from crucible.train.scheduler import SchedulerConfig
from crucible.train.sim import SimConfig, default_shapes, simulate

SCALES = [  # (engines, slots per engine, env workers, groups per step B, n): slots = envs / 2 (a live rollout spends part of
    # its life in environment steps), and the admission cap S·B·n with S = 2 admits ≈ the env pool, so slots bind
    (128, 16, 4_096, 85, 16),
    (512, 16, 16_384, 341, 16),
    (1_024, 16, 32_768, 682, 16),
    (2_048, 16, 32_768, 682, 16),
]
SIM_TOPOLOGIES = ("colocated_sync", "colocated_partial", "disaggregated_stream", "elastic")
MODELS = {"8B dense": (ModelSpec(8, 32, 8, 128), Workload(4096, 2048, 8192)),
          "70B dense": (ModelSpec(70, 80, 8, 128), Workload(4096, 2048, 8192)),
          "1T MoE (32B active)": (ModelSpec(1000, 61, 8, 128, active_b=32), Workload(8192, 4096, 16384, turns=8, env_s_per_turn=5))}
NODES = (8, 32, 128, 512, 1024)


def _mean_tokens(shape, k: int = 2000) -> float:
    tot = 0
    for i in range(k):
        sh = shape(f"p{i}", i % 16)
        tot += sum(p + c for p, c, _ in sh.turns)
    return tot / k


def control_plane(scales, steps: int) -> list[dict]:
    rows = []
    shape = default_shapes(0, tail_frac=0.2, tail_mult=8.0)
    mean_tok = _mean_tokens(shape)
    for engines, per, envs, B, n in scales:
        slots = engines * per
        src = {s: [f"{s}{i}" for i in range(B * (steps + 4) * 2)] for s in ("math", "code")}
        for topo in SIM_TOPOLOGIES:
            # a fixed 120 s of training per step at every scale (30 s fixed + 90 s over the batch's tokens): the trainer
            # scales with the batch, as a planner-sized trainer would
            per_ktok = 90.0 / (B * n * mean_tok / 1000.0)
            cfg = SimConfig(topology=topo, slots=slots, env_workers=envs, steps=steps, train_s_base=30.0, train_s_per_ktok=per_ktok,
                            publish_s=5.0, elastic_frac=0.25, elastic_train_speedup=2.0)
            t0 = time.perf_counter()
            r = simulate(SchedulerConfig(n=n, B=B, max_staleness=2), cfg, src, shape)
            wall = time.perf_counter() - t0
            rows.append({"engines": engines, "slots": slots, "envs": envs, "B": B, "n": n, "topology": topo, "steps": r.steps,
                         "sim_s_per_step": r.time / max(r.steps, 1), "engine_util": r.engine_utilization(slots), "trainer_idle_s": r.trainer_idle_s,
                         "max_staleness": r.max_staleness(), "wasted_decode_tok": r.wasted_decode_tokens, "reprefill_tok": r.reprefill_tokens,
                         "max_inflight": r.max_inflight_seen, "wall_s_per_step": wall / max(r.steps, 1)})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def staleness_window(steps: int, engines: int = 1_024, per: int = 16, envs: int = 32_768, B: int = 682, n: int = 16, *,
                     taus=(1, 2, 4, 8), train_scale: float = 1.0) -> list[dict]:
    """At DORA scale: the staleness bound τ (and the admission window S = τ) against utilization and step time. With
    a long tail and whole groups as the training unit, the in-flight cap S·B·n (not the engines) sets throughput."""
    rows = []
    shape = default_shapes(0, tail_frac=0.2, tail_mult=8.0)
    per_ktok = train_scale * 90.0 / (B * n * _mean_tokens(shape) / 1000.0)
    src = {s: [f"{s}{i}" for i in range(B * (steps + 12) * 2)] for s in ("math", "code")}
    for tau in taus:
        for topo in ("colocated_partial", "disaggregated_stream", "elastic"):
            cfg = SimConfig(topology=topo, slots=engines * per, env_workers=envs, steps=steps, train_s_base=30.0 * train_scale, train_s_per_ktok=per_ktok,
                            publish_s=5.0, elastic_frac=0.25, elastic_train_speedup=2.0)
            t0 = time.perf_counter()
            r = simulate(SchedulerConfig(n=n, B=B, max_staleness=tau), cfg, src, shape)
            rows.append({"tau": tau, "train_scale": train_scale, "topology": topo, "steps": r.steps, "sim_s_per_step": r.time / max(r.steps, 1),
                         "engine_util": r.engine_utilization(engines * per), "trainer_idle_s_per_step": r.trainer_idle_s / max(r.steps, 1),
                         "max_staleness": r.max_staleness(), "max_inflight": r.max_inflight_seen, "reprefill_tok": r.reprefill_tokens,
                         "wall_s_per_step": (time.perf_counter() - t0) / max(r.steps, 1)})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def placement(nodes=NODES) -> list[dict]:
    rows = []
    for name, (m, w) in MODELS.items():
        for N in nodes:
            c = ClusterSpec(N)
            try:
                p = plan(c, m, w)
            except PlacementError as e:
                rows.append({"model": name, "nodes": N, "error": str(e)})
                continue
            rows.append({"model": name, "nodes": N, "gpus": c.gpus, "best": p.topology, "step_s": p.step_s, "engine_gpus": p.engine_gpus,
                         "engines": p.engines, "train_gpus": p.train_gpus, "elastic_gpus": p.elastic_gpus, "mesh": list(p.mesh),
                         "env_workers": p.env_workers, "alternatives": {t: p.alternatives.get(t) for t in TOPOLOGIES}, "times": p.times})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", default="infra_scale_sweep.json")
    a = ap.parse_args()
    scales = SCALES[:2] if a.quick else SCALES
    res = {"control_plane": control_plane(scales, a.steps), "staleness_window": [] if a.quick else staleness_window(a.steps),
           "train_bound": [] if a.quick else staleness_window(a.steps, taus=(4,), train_scale=4.0),
           "placement": placement(NODES[:3] if a.quick else NODES)}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
