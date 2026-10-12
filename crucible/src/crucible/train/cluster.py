"""Launching a placement across nodes (M12-infra C): from ``placement.Placement`` to processes on GPUs. (``train.launch``
is the single-job supervisor of the HTTP process model: engine servers, workers, env servers with restarts; this module
places SPMD trainer ranks and engine groups on a multi-node cluster.)

``process_specs`` turns the GPU layout into one ``ProcSpec`` per process with the environment torch.distributed
expects (``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` / ``NODE_RANK`` / ``MASTER_ADDR`` / ``MASTER_PORT``) and the
devices it owns (``CUDA_VISIBLE_DEVICES``):

- trainer GPUs: one process per GPU, consecutive ranks (rank 0 is the driver: it runs the ``Runtime`` and wraps the
  trainer in ``spmd.SPMDDriver``; the others ``spmd.follow``);
- engines: one process per engine and node, owning the engine's GPUs on that node (a multi-node engine is one process
  per node with ``ENGINE_NODE_RANK``); the engine server (vLLM / SGLang / ``TorchEngine``) is the entry's business;
- colocated (``both``): one process per GPU holding the trainer rank and that GPU's engine shard (the hybrid
  engine: one process, two roles, the GPU handed over at each barrier);
- elastic GPUs: trainer ranks (after the dedicated ones) whose processes also host an engine (``ENGINE``).

Two launchers run the specs with the same ``entry`` (``"module:function"``, called as ``fn(spec)``):

- ``LocalLauncher``: subprocesses on this machine (tests, a single node);
- ``RayLauncher``: one ``STRICT_PACK`` placement group per node, one bundle per GPU, each process a Ray actor pinned
  to its bundle (pattern from OpenRLHF's ray launcher, Apache-2.0, with per-node groups instead of one group sorted by
  node id: the node ↔ group mapping is then fixed by construction). Ray is optional (``pip install ray``).
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.train.placement import Placement


@dataclass
class ProcSpec:
    role: str  # trainer | engine | both | elastic
    node: int
    gpus: list[int]  # node-local GPU indices this process owns
    rank: int | None = None  # trainer rank
    engine: int | None = None  # engine index
    env: dict[str, str] = field(default_factory=dict)


def process_specs(p: Placement, *, master_addr: str = "127.0.0.1", master_port: int = 29500) -> list[ProcSpec]:
    c = p.cluster
    trainer_world = c.gpus if p.topology.startswith("colocated") else p.train_gpus + p.elastic_gpus
    specs: list[ProcSpec] = []
    engine_parts: dict[tuple[int, int], list[int]] = {}  # (engine, node) → GPUs
    for n, row in enumerate(p.layout):
        for g, role in enumerate(row):
            kind, _, rest = role.partition(":")
            parts = rest.split("/")
            if kind == "train":
                specs.append(ProcSpec("trainer", n, [g], rank=int(parts[0])))
            elif kind == "both":
                specs.append(ProcSpec("both", n, [g], rank=int(parts[1]), engine=int(parts[0])))
            elif kind == "elastic":
                specs.append(ProcSpec("elastic", n, [g], rank=int(parts[2]), engine=int(parts[0])))
            elif kind == "engine":
                engine_parts.setdefault((int(parts[0]), n), []).append(g)
            else:
                raise ValueError(f"unassigned GPU {n}/{g} in the layout")
    by_engine: dict[int, list[int]] = {}
    for (e, n), gs in sorted(engine_parts.items()):
        by_engine.setdefault(e, []).append(n)
        specs.append(ProcSpec("engine", n, gs, engine=e))
    for s in specs:
        s.env = {"NODE_RANK": str(s.node), "CUDA_VISIBLE_DEVICES": ",".join(map(str, s.gpus)), "CRUCIBLE_ROLE": s.role,
                 "MASTER_ADDR": master_addr, "MASTER_PORT": str(master_port)}
        if s.rank is not None:
            s.env.update(RANK=str(s.rank), WORLD_SIZE=str(trainer_world), LOCAL_RANK=str(s.gpus[0]))
        if s.engine is not None:
            s.env.update(ENGINE=str(s.engine), ENGINE_GPUS=str(p.engine_gpus))
            if s.role == "engine":
                s.env.update(ENGINE_NODE_RANK=str(by_engine[s.engine].index(s.node)), ENGINE_NODES=str(len(by_engine[s.engine])))
    return sorted(specs, key=lambda s: (s.rank is None, s.rank if s.rank is not None else 0, s.engine or 0, s.node))


def _resolve(entry: str):
    mod, _, fn = entry.partition(":")
    return getattr(importlib.import_module(mod), fn)


def run_entry() -> None:
    """Subprocess side of ``LocalLauncher``: the spec arrives as JSON in ``CRUCIBLE_SPEC``."""
    spec = ProcSpec(**json.loads(os.environ["CRUCIBLE_SPEC"]))
    out = _resolve(sys.argv[1])(spec)
    if out is not None:
        print("CRUCIBLE_RESULT " + json.dumps(out), flush=True)


def first_traceback(err: str, limit: int = 2500) -> str:
    """The first Python traceback in a stderr (the cause), not its tail: a dead rank's peers fill the tail with NCCL
    watchdog dumps that hide it (found live: an 8B run's driver error was lost under them)."""
    i = err.find("Traceback (most recent call last)")
    if i < 0:
        return err[-limit:]
    j = err.find("\n\n", i)
    return err[i: (j if j > 0 else i + limit)][:limit]


class LocalLauncher:
    def __init__(self, *, extra_env: dict[str, str] | None = None, timeout_s: float = 300.0, log_dir: str | Path | None = None):
        """``log_dir``: each process's full stdout / stderr is kept there (``<role>-r<rank>-e<engine>.log``)."""
        self.extra_env, self.timeout_s = extra_env or {}, timeout_s
        self.log_dir = Path(log_dir) if log_dir else None

    def run(self, specs: list[ProcSpec], entry: str) -> list[Any]:
        procs = []
        for s in specs:
            env = {**os.environ, **self.extra_env, **s.env, "CRUCIBLE_SPEC": json.dumps(asdict(s))}
            procs.append(subprocess.Popen([sys.executable, "-c", "from crucible.train.cluster import run_entry; run_entry()", entry], env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        results, errs = [], []
        for s, p in zip(specs, procs, strict=True):
            try:
                out, err = p.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                p.kill()
                out, err = p.communicate()
                err = "TIMEOUT\n" + err
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                (self.log_dir / f"{s.role}-r{s.rank}-e{s.engine}.log").write_text(out + "\n--- stderr ---\n" + err)
            if p.returncode:
                errs.append(f"{s.role} rank={s.rank} engine={s.engine}: {first_traceback(err)}")
            line = [ln for ln in out.splitlines() if ln.startswith("CRUCIBLE_RESULT ")]
            results.append(json.loads(line[-1].split(" ", 1)[1]) if line else None)
        if errs:
            raise RuntimeError("launched processes failed:\n" + "\n".join(errs))
        return results


def ray_bundles(p: Placement, *, cpus_per_gpu: float = 1.0, gpu_resource: str = "GPU") -> list[list[dict[str, float]]]:
    """Per node, one bundle per GPU (``gpu_resource="CPU"`` for CPU-only clusters and tests)."""
    if gpu_resource == "CPU":
        return [[{"CPU": cpus_per_gpu} for _ in range(p.cluster.gpus_per_node)] for _ in range(p.cluster.nodes)]
    return [[{gpu_resource: 1.0, "CPU": cpus_per_gpu} for _ in range(p.cluster.gpus_per_node)] for _ in range(p.cluster.nodes)]


class RayLauncher:
    def __init__(self, *, cpus_per_gpu: float = 1.0, gpu_resource: str = "GPU", timeout_s: float = 600.0):
        import ray  # optional dependency

        self.ray, self.cpus_per_gpu, self.gpu_resource, self.timeout_s = ray, cpus_per_gpu, gpu_resource, timeout_s

    def run(self, p: Placement, specs: list[ProcSpec], entry: str) -> list[Any]:
        ray = self.ray
        from ray.util.placement_group import placement_group, remove_placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        pgs = [placement_group(b, strategy="STRICT_PACK") for b in ray_bundles(p, cpus_per_gpu=self.cpus_per_gpu, gpu_resource=self.gpu_resource)]
        ray.get([pg.ready() for pg in pgs], timeout=self.timeout_s)

        @ray.remote
        class Role:
            def run(self, spec_d: dict, entry: str) -> Any:
                spec = ProcSpec(**spec_d)
                os.environ.update(spec.env)
                if spec.env.get("CRUCIBLE_CPU_ONLY"):
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                return _resolve(entry)(spec)

        try:
            actors = []
            for s in specs:
                gpu = 0.0 if self.gpu_resource == "CPU" else 1.0 / max(1, len(s.gpus))  # a multi-GPU engine process spreads over its bundles
                actors.append(Role.options(num_cpus=self.cpus_per_gpu * 0.5 if self.gpu_resource == "CPU" else 0.01, num_gpus=gpu * len(s.gpus) if gpu else 0,
                                           scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pgs[s.node], placement_group_bundle_index=s.gpus[0])).remote())
            return ray.get([a.run.remote(asdict(s), entry) for a, s in zip(actors, specs, strict=True)], timeout=self.timeout_s)
        finally:
            for pg in pgs:
                remove_placement_group(pg)


def smoke(spec: ProcSpec) -> dict[str, Any]:
    """A placement smoke test, before spending a run on it: every trainer rank joins the process group (NCCL on GPUs,
    gloo otherwise) and all-reduces its rank; every process reports what it was given."""
    import socket

    out: dict[str, Any] = {"role": spec.role, "node": spec.node, "gpus": spec.gpus, "rank": spec.rank, "engine": spec.engine,
                           "host": socket.gethostname(), "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if spec.rank is not None:
        import torch
        import torch.distributed as dist

        backend = "nccl" if torch.cuda.is_available() and not os.environ.get("CRUCIBLE_CPU_ONLY") else "gloo"
        dist.init_process_group(backend, init_method="env://")
        t = torch.tensor([float(spec.rank)], device="cuda" if backend == "nccl" else "cpu")
        dist.all_reduce(t)
        out.update(world=dist.get_world_size(), rank_sum=float(t.item()))
        dist.destroy_process_group()
    return out


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
