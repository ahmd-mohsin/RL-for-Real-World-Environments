"""Launching a placement across nodes (M12-infra C): from ``placement.Placement`` to processes on GPUs. (``train.launch``
is the single-job supervisor of the HTTP process model: engine servers, workers, env servers with restarts; this module
places SPMD trainer ranks and engine groups on a multi-node cluster.)

``process_specs`` turns the GPU layout into one ``ProcSpec`` per process with the environment torch.distributed
expects (``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` / ``LOCAL_WORLD_SIZE`` / ``NODE_RANK`` / ``MASTER_ADDR`` /
``MASTER_PORT``) and the devices it owns (``CUDA_VISIBLE_DEVICES``; the physical indices also in ``CRUCIBLE_GPUS``):

- trainer GPUs: one process per GPU, consecutive ranks (rank 0 is the driver: it runs the ``Runtime`` and wraps the
  trainer in ``spmd.SPMDDriver``; the others ``spmd.follow``);
- engines: one process per engine and node, owning the engine's GPUs on that node (a multi-node engine is one process
  per node with ``ENGINE_NODE_RANK``); the engine server (vLLM / SGLang / ``TorchEngine``) is the entry's business;
- colocated (``both``) and elastic GPUs: one trainer process per GPU. With one GPU per engine the same process hosts
  the engine (``ENGINE_HOST=1``: the hybrid engine, the GPU handed over at each barrier); with ``k > 1`` GPUs per engine a
  separate engine process (role ``engine``, ``ENGINE_COLOCATED=1``) sees all ``k`` GPUs, because a TP-k server cannot
  start from processes that each see one GPU (M12 audit H4 V10).

``LOCAL_RANK`` indexes the visible devices (M12 audit H4 V8 / G5 D6): a process that sees one GPU has ``LOCAL_RANK=0``,
so torchrun-convention code (``torch.cuda.set_device(LOCAL_RANK)``, ``init_device_mesh``, accelerate, vLLM's
``external_launcher``) works on every node; ``init_rank()`` does that and initializes the process group with an
explicit timeout.

Launchers run the specs with one ``entry`` (``"module:function"``, called as ``fn(spec)``):

- ``LocalLauncher(node=k)``: subprocesses on this machine for node ``k``'s specs; specs of several nodes are refused
  unless ``single_host=True`` (tests that emulate nodes on one host). The first process that fails stops the others
  (``srun --kill-on-bad-exit`` semantics), so a dead rank never leaves its peers holding GPUs until the timeout;
- ``run_node``: the multi-node path under Slurm (or any launcher that runs one task per node): node rank and master
  address come from the environment (``SLURM_NODEID`` / ``SLURM_JOB_NODELIST``, or ``NODE_RANK`` / ``MASTER_ADDR``),
  and this node's specs run under ``LocalLauncher(node=...)`` (``scripts/repacss_train.sbatch`` in the repository);
- ``slurm_env()``: the torchrun-style variables for one task per GPU (``srun --ntasks-per-node=G``);
- ``RayLauncher``: one ``STRICT_PACK`` placement group per node, one bundle per GPU, each process a Ray actor pinned
  to its bundle (pattern from OpenRLHF's ray launcher, Apache-2.0, with per-node groups instead of one group sorted by
  node id: the node ↔ group mapping is then fixed by construction); ``MASTER_ADDR`` is the IP of the node that holds
  trainer rank 0 (H4 V9: it was copied from the specs, ``127.0.0.1``). Ray is optional (``pip install ray``).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.train.placement import Placement

# the launch environment every rank gets unless the caller set it: NCCL errors abort the communicator instead of hanging
# (torch's watchdog then ends the process), and the first NCCL failure prints which rank and why
NCCL_DEFAULTS = {"TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", "TORCH_NCCL_DUMP_ON_TIMEOUT": "1"}
DEFAULT_PG_TIMEOUT_S = 1800.0


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
    engine_parts: dict[tuple[int, int], list[int]] = {}  # (engine, node) → GPUs of an engine-only process
    shared_parts: dict[tuple[int, int], list[int]] = {}  # (engine, node) → GPUs of a colocated / elastic engine of k > 1 GPUs
    multi = p.engine_gpus > 1
    for n, row in enumerate(p.layout):
        for g, role in enumerate(row):
            kind, _, rest = role.partition(":")
            parts = rest.split("/")
            if kind == "train":
                specs.append(ProcSpec("trainer", n, [g], rank=int(parts[0])))
            elif kind in ("both", "elastic"):
                e = int(parts[0])
                specs.append(ProcSpec(kind, n, [g], rank=int(parts[1] if kind == "both" else parts[2]), engine=e))
                if multi:
                    shared_parts.setdefault((e, n), []).append(g)
            elif kind == "engine":
                engine_parts.setdefault((int(parts[0]), n), []).append(g)
            else:
                raise ValueError(f"unassigned GPU {n}/{g} in the layout")
    by_engine: dict[int, list[int]] = {}
    for (e, n), gs in sorted({**engine_parts, **shared_parts}.items()):
        by_engine.setdefault(e, []).append(n)
        specs.append(ProcSpec("engine", n, gs, engine=e))
    local_world = {n: sum(1 for s in specs if s.rank is not None and s.node == n) for n in range(c.nodes)}
    for s in specs:
        s.env = {"NODE_RANK": str(s.node), "NNODES": str(c.nodes), "CUDA_VISIBLE_DEVICES": ",".join(map(str, s.gpus)),
                 "CRUCIBLE_GPUS": ",".join(map(str, s.gpus)), "CRUCIBLE_ROLE": s.role, "MASTER_ADDR": master_addr, "MASTER_PORT": str(master_port)}
        if s.rank is not None:
            # the device index *within* CUDA_VISIBLE_DEVICES: this process sees one GPU, so it is 0 (H4 V8)
            s.env.update(RANK=str(s.rank), WORLD_SIZE=str(trainer_world), LOCAL_RANK="0", LOCAL_WORLD_SIZE=str(local_world[s.node]))
        if s.engine is not None:
            s.env.update(ENGINE=str(s.engine), ENGINE_GPUS=str(p.engine_gpus))
            if s.role == "engine":
                s.env.update(ENGINE_HOST="1", ENGINE_NODE_RANK=str(by_engine[s.engine].index(s.node)), ENGINE_NODES=str(len(by_engine[s.engine])))
                if (s.engine, s.node) in shared_parts:
                    s.env["ENGINE_COLOCATED"] = "1"  # shares its GPUs with the trainer ranks of the same engine
            else:
                s.env["ENGINE_HOST"] = "0" if multi else "1"
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
    def __init__(self, *, extra_env: dict[str, str] | None = None, timeout_s: float = 300.0, log_dir: str | Path | None = None,
                 node: int | None = None, single_host: bool = False, kill_on_failure: bool = True, grace_s: float = 5.0):
        """``node``: run only that node's specs (the multi-node path: one launcher per node). ``single_host``: run every
        node's specs here (CPU tests that emulate nodes). ``log_dir``: each process's full stdout / stderr is kept there
        (``<role>-r<rank>-e<engine>.log``). ``kill_on_failure``: the first process that exits non-zero ends the others
        (after ``grace_s``, so its peers can print their own errors)."""
        self.extra_env, self.timeout_s = extra_env or {}, timeout_s
        self.log_dir = Path(log_dir) if log_dir else None
        self.node, self.single_host, self.kill_on_failure, self.grace_s = node, single_host, kill_on_failure, grace_s

    def select(self, specs: list[ProcSpec]) -> list[ProcSpec]:
        if self.node is not None:
            mine = [s for s in specs if s.node == self.node]
            if not mine:
                raise ValueError(f"no process of this placement is on node {self.node}")
            return mine
        nodes = {s.node for s in specs}
        if len(nodes) > 1 and not self.single_host:
            raise ValueError(f"the specs span nodes {sorted(nodes)}: run one LocalLauncher(node=k) per node (cluster.run_node under Slurm), "
                             "or single_host=True to emulate them on this machine (H4 V9: every node's GPUs were claimed here)")
        return specs

    def run(self, specs: list[ProcSpec], entry: str) -> list[Any]:
        specs = self.select(specs)
        base = {**NCCL_DEFAULTS, **os.environ, **self.extra_env}
        procs = []
        for s in specs:
            env = {**base, **s.env, "CRUCIBLE_SPEC": json.dumps(asdict(s))}
            procs.append(subprocess.Popen([sys.executable, "-c", "from crucible.train.cluster import run_entry; run_entry()", entry], env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        outs = _supervise(procs, self.timeout_s, self.kill_on_failure, self.grace_s)
        results, errs = [], []
        for s, p, (out, err, why) in zip(specs, procs, outs, strict=True):
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                (self.log_dir / f"{s.role}-r{s.rank}-e{s.engine}.log").write_text(out + "\n--- stderr ---\n" + err)
            if p.returncode or why:
                errs.append(f"{s.role} rank={s.rank} engine={s.engine} node={s.node}: {why or 'exit ' + str(p.returncode)}\n{first_traceback(err)}")
            line = [ln for ln in out.splitlines() if ln.startswith("CRUCIBLE_RESULT ")]
            results.append(json.loads(line[-1].split(" ", 1)[1]) if line else None)
        if errs:
            raise RuntimeError("launched processes failed:\n" + "\n".join(errs))
        return results


def _supervise(procs: list[subprocess.Popen], timeout_s: float, kill_on_failure: bool, grace_s: float) -> list[tuple[str, str, str]]:
    """Wait for every process with one deadline; read their pipes in threads (no pipe fills up); on the first non-zero
    exit, give the others ``grace_s`` and kill the rest. Returns (stdout, stderr, why-killed) per process."""
    import threading

    bufs = [["", ""] for _ in procs]

    def pump(i: int) -> None:
        o, e = procs[i].communicate()
        bufs[i][0], bufs[i][1] = o or "", e or ""

    ths = [threading.Thread(target=pump, args=(i,), daemon=True) for i in range(len(procs))]
    for t in ths:
        t.start()
    why = [""] * len(procs)
    deadline = time.monotonic() + timeout_s
    failed_at: float | None = None
    while any(p.poll() is None for p in procs):
        now = time.monotonic()
        if failed_at is None and kill_on_failure and any(p.returncode not in (None, 0) for p in procs):
            failed_at = now
        if now > deadline or (failed_at is not None and now - failed_at > grace_s):
            for i, p in enumerate(procs):
                if p.poll() is None:
                    why[i] = "TIMEOUT" if now > deadline else "killed: another process of the job failed"
                    p.kill()
            break
        time.sleep(0.05)
    for t in ths:
        t.join()
    return [(o, ("TIMEOUT\n" if w == "TIMEOUT" else "") + e, w) for (o, e), w in zip(bufs, why, strict=True)]


# ---- multi-node launch: Slurm / torchrun-style environment ---------------------------------------------------------
def expand_nodelist(nodelist: str) -> list[str]:
    """Slurm's compressed host list (``SLURM_JOB_NODELIST``) → host names, in order: ``rpg-93-[1-2,7]`` →
    ``rpg-93-1, rpg-93-2, rpg-93-7``; zero padding kept (``gpu[001-002]``); several groups separated by commas.
    (``scontrol show hostnames`` does the same where it exists.)"""
    out: list[str] = []
    for m in re.finditer(r"([^,\[]+)(?:\[([^\]]+)\])?(?:,|$)", nodelist):
        prefix, ranges = m.group(1), m.group(2)
        if not prefix and not ranges:
            continue
        if ranges is None:
            out.append(prefix)
            continue
        for part in ranges.split(","):
            lo, _, hi = part.partition("-")
            if not hi:
                out.append(prefix + lo)
                continue
            width = len(lo) if lo.startswith("0") else 0
            out.extend(prefix + str(i).zfill(width) for i in range(int(lo), int(hi) + 1))
    return out


def slurm_env(environ: dict[str, str] | None = None, *, master_port: int = 29500) -> dict[str, str]:
    """For ``srun --ntasks-per-node=G`` (one task per GPU): RANK = SLURM_PROCID, LOCAL_RANK = SLURM_LOCALID (the device
    index when every task sees the node's GPUs; 0 when Slurm bound a single GPU to the task), WORLD_SIZE = SLURM_NTASKS,
    NODE_RANK = SLURM_NODEID, MASTER_ADDR = the first host of the job (already-set MASTER_ADDR / MASTER_PORT win)."""
    e = dict(os.environ if environ is None else environ)
    hosts = expand_nodelist(e.get("SLURM_JOB_NODELIST") or e.get("SLURM_NODELIST") or "")
    master = e.get("MASTER_ADDR") or (hosts[0] if hosts else "127.0.0.1")
    nnodes = int(e.get("SLURM_JOB_NUM_NODES") or e.get("SLURM_NNODES") or max(1, len(hosts)))
    ntasks = int(e.get("SLURM_NTASKS") or e.get("WORLD_SIZE") or 1)
    local = e.get("SLURM_LOCALID", "0")
    visible = [v for v in e.get("CUDA_VISIBLE_DEVICES", "").split(",") if v]
    if len(visible) == 1:  # Slurm bound one GPU to this task (--gpus-per-task=1 / --gpu-bind): it is device 0 here
        local = "0"
    return {"RANK": e.get("SLURM_PROCID", "0"), "LOCAL_RANK": local, "WORLD_SIZE": str(ntasks),
            "LOCAL_WORLD_SIZE": str(ntasks // nnodes if nnodes else ntasks), "NODE_RANK": e.get("SLURM_NODEID", "0"), "NNODES": str(nnodes),
            "MASTER_ADDR": master, "MASTER_PORT": e.get("MASTER_PORT", str(master_port))}


def node_identity(environ: dict[str, str] | None = None) -> tuple[int, str]:
    """(node rank, master address) of this host: Slurm first (one task per node), then NODE_RANK / MASTER_ADDR."""
    e = dict(os.environ if environ is None else environ)
    if "SLURM_NODEID" in e:
        hosts = expand_nodelist(e.get("SLURM_JOB_NODELIST") or e.get("SLURM_NODELIST") or "")
        return int(e["SLURM_NODEID"]), e.get("MASTER_ADDR") or (hosts[0] if hosts else "127.0.0.1")
    return int(e.get("NODE_RANK", "0")), e.get("MASTER_ADDR", "127.0.0.1")


def run_node(p: Placement, entry: str, *, master_port: int = 29500, node: int | None = None, master_addr: str | None = None,
             environ: dict[str, str] | None = None, **launcher_kw: Any) -> list[Any]:
    """Run this node's processes of a multi-node placement (``srun --ntasks-per-node=1`` runs it once per node). The
    plan must be computed identically on every node (``placement.plan`` is deterministic)."""
    n, addr = node_identity(environ)
    node = n if node is None else node
    if master_addr is None:
        if p.cluster.nodes > 1 and addr in ("127.0.0.1", "localhost"):
            raise ValueError("a multi-node placement needs MASTER_ADDR (the first host of the job): run under Slurm or set it")
        master_addr = addr
    specs = process_specs(p, master_addr=master_addr, master_port=master_port)
    return LocalLauncher(node=node, **launcher_kw).run(specs, entry)


def init_rank(backend: str | None = None, *, timeout_s: float | None = None) -> tuple[int, int, Any]:
    """In a launched trainer process: pick the device from ``LOCAL_RANK`` (an index into the visible devices), set it
    *before* the process group exists (NCCL binds the communicator to it), and initialize the default group from the
    environment with an explicit timeout (``CRUCIBLE_PG_TIMEOUT_S``, default 1800 s; torch's default would be 10 min on
    NCCL, 30 on gloo). Returns (rank, world size, device)."""
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    for k, v in NCCL_DEFAULTS.items():
        os.environ.setdefault(k, v)
    if timeout_s is None:
        timeout_s = float(os.environ.get("CRUCIBLE_PG_TIMEOUT_S", DEFAULT_PG_TIMEOUT_S))
    cuda = torch.cuda.is_available() and not os.environ.get("CRUCIBLE_CPU_ONLY")
    backend = backend or ("nccl" if cuda else "gloo")
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "nccl":
        n = torch.cuda.device_count()
        if local >= n:
            raise RuntimeError(f"LOCAL_RANK={local} but this process sees {n} GPU(s) (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
        dev = torch.device("cuda", local)
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")
    if not dist.is_initialized():
        kw: dict[str, Any] = {"device_id": dev} if backend == "nccl" else {}
        dist.init_process_group(backend, init_method="env://", timeout=timedelta(seconds=timeout_s), **kw)
    return dist.get_rank(), dist.get_world_size(), dev


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
                for k, v in NCCL_DEFAULTS.items():
                    os.environ.setdefault(k, v)
                os.environ.update(spec.env)
                if spec.env.get("CRUCIBLE_CPU_ONLY"):
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                return _resolve(entry)(spec)

        @ray.remote(num_cpus=0)
        def node_ip() -> str:
            return ray.util.get_node_ip_address()

        try:
            r0 = next((s for s in specs if s.rank == 0), None)
            if r0 is not None:  # H4 V9: the rendezvous address is the node that holds trainer rank 0, learned after placement
                addr = ray.get(node_ip.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pgs[r0.node], placement_group_bundle_index=r0.gpus[0])).remote(), timeout=self.timeout_s)
                for s in specs:
                    s.env["MASTER_ADDR"] = addr
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
    gloo otherwise; ``init_rank``) and all-reduces its rank; every process reports what it was given."""
    import socket

    out: dict[str, Any] = {"role": spec.role, "node": spec.node, "gpus": spec.gpus, "rank": spec.rank, "engine": spec.engine,
                           "host": socket.gethostname(), "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "local_rank": os.environ.get("LOCAL_RANK")}
    if spec.rank is not None:
        import torch
        import torch.distributed as dist

        _, _, dev = init_rank()
        t = torch.tensor([float(spec.rank)], device=dev)
        dist.all_reduce(t)
        out.update(world=dist.get_world_size(), rank_sum=float(t.item()), device=str(dev))
        dist.destroy_process_group()
    return out


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
