"""The DeltaAI deployment of a Crucible RL run on prime-rl (swarm-pipeline/14): configs, the Slurm job, the GPU-hour plan.

One Slurm job on ``ghx4`` (4 GH200 per node) runs three things that share ``out`` (``/work/hdd/bjhg/...``):

- the **prime-rl trainer** (``trainer @ trainer.toml``, torchrun over ``train_gpus``): the student with LoRA (experts
  included), FSDP2, expert parallel ``ep``, context parallel ``cp`` (ring) for the long sequences, activation checkpointing
  with offload to Grace memory, Crucible's loss (``loss.crucible_loss``), the filesystem rollout transport and weight
  broadcast, a checkpoint every step (``ckpt.interval = 1``: every step is resumable);
- the **vLLM server** (``inference @ inference.toml``, ``infer_gpus``): the base model with LoRA hot-loading and prefix
  caching;
- **Crucible's loop** (``scripts/delta_rl.py train``): the Tinker loop over ``backend.PrimeRLBackend``, the Worlds in
  Apptainer on the node's CPUs.

**Account rules** (DeltaAI memory, 2026-10-22): the job is charged to ``bjhg-dtai-gh`` (CIS262200) only; the login
environment exports ``SBATCH_ACCOUNT=bfgp-dtai-gh``, which overrides ``#SBATCH --account``, so the submit command unsets it
and passes ``--account`` on the command line (``submit_command``). Charging follows the largest node share a job takes,
so memory requests stay proportional to GPUs (``mem_gb``).

**The plan** (``plan``) is an estimate from the stored 300-call student runs (median sequence 160K tokens, p90 195K,
max 217K; median 73K sampled tokens per episode) and GH200 peak numbers; it is stated as such in every output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.train.bridges.primerl.loss import KINDS

ACCOUNT = "bjhg-dtai-gh"  # CIS262200; never bfgp-dtai-gh (another project)
WORK = "/work/hdd/bjhg/mahmedmohsin"
PRIME_RL_VERSION = "v0.10.0"  # PrimeIntellect-ai/prime-rl, Apache-2.0 (2026-10-13)
LOSS_IMPORT = "crucible.train.bridges.primerl.loss.crucible_loss"


@dataclass
class DeltaConfig:
    base_model: str = "Qwen/Qwen3.6-35B-A3B"  # the HF repo vLLM and the trainer load
    account: str = ACCOUNT
    partition: str = "ghx4"
    nodes: int = 1
    gpus_per_node: int = 4
    train_gpus: int = 2
    infer_gpus: int = 2
    ep: int = 2  # expert parallel over the train GPUs (256 experts)
    cp: int = 2  # context parallel (ring): seq_len / cp tokens per GPU
    infer_tp: int = 2  # the 70 GB of bf16 weights split two ways leave room for ~30 concurrent 160K-token episodes (plan)
    seq_len: int = 262_144
    lora_rank: int = 32
    lora_alpha: float = 64.0
    kl_coeff: float = 0.0  # the stable_reinforce-style sampler anchor (loss doc)
    ratio_low: float = 0.8
    ratio_high: float = 1.2
    max_importance_ratio: float = 1e4
    gpu_memory_utilization: float = 0.90
    time: str = "12:00:00"
    mem_gb_per_gpu: int = 110  # a quarter node per GPU: charging follows the largest share
    cpus_per_gpu: int = 18
    workdir: str = f"{WORK}/crucible-train"
    port: int = 8000
    extra_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.account != ACCOUNT:
            raise ValueError(f"DeltaAI runs are charged to {ACCOUNT} (CIS262200) only, not {self.account!r}")
        if self.train_gpus + self.infer_gpus > self.nodes * self.gpus_per_node:
            raise ValueError("train_gpus + infer_gpus exceed the job's GPUs")
        if self.train_gpus % self.ep or self.train_gpus % self.cp:
            raise ValueError("ep and cp must divide train_gpus")
        if self.seq_len % self.cp:
            raise ValueError("cp must divide seq_len")
        if self.infer_gpus % self.infer_tp:
            raise ValueError("infer_tp must divide infer_gpus")

    @property
    def gpus(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def data_ranks(self) -> int:
        """The trainer's data-parallel world: the ranks the packer fills (FSDP over train_gpus; cp splits sequences, ep
        experts, neither adds a data dimension)."""
        return self.train_gpus // self.cp


def trainer_toml(train: Any, d: DeltaConfig, out: Path, *, resume_step: int | None = None) -> dict[str, Any]:
    """The prime-rl ``TrainerConfig`` (``prime_rl/configs/trainer.py``, v0.10.0) of a Crucible run. ``train`` is the
    Tinker bridge's ``TrainConfig`` (the same flags as ``tinker_rl.py train``)."""
    if train.loss_fn not in KINDS:
        raise ValueError(f"loss_fn is one of {KINDS}")
    doc: dict[str, Any] = {
        "output_dir": str(out),
        "max_steps": int(train.steps),
        "model": {"name": d.base_model, "seq_len": d.seq_len, "ep": d.ep, "cp": d.cp, "attn": "flash_attention_3", "offload": "optimizer",
                  "ac": {"freq": 1}, "ac_offloading": {"max_inflight_activations": 5},
                  "lora": {"rank": int(train.lora_rank), "alpha": float(d.lora_alpha), "dropout": 0.0}},
        "loss": {"type": "custom", "import_path": LOSS_IMPORT,
                 "kwargs": {"kind": train.loss_fn, "kl_coeff": d.kl_coeff, "ratio_low": d.ratio_low, "ratio_high": d.ratio_high,
                            "max_importance_ratio": d.max_importance_ratio, **{k: float(v) for k, v in (train.loss_fn_config or {}).items()
                                                                                if k in ("ratio_low", "ratio_high", "kl_coeff")}}},
        "optim": {"type": "adamw", "lr": float(train.learning_rate), "max_norm": 1.0},
        "ckpt": {"interval": 1, "keep_last": 3},
        "weight_broadcast": {"type": "filesystem"},
        "rollout_transport": {"type": "filesystem"},
        "monitors": {"file": {}},
    }
    if resume_step is not None:
        doc["resume"] = {"step": int(resume_step)}
    return doc


def inference_toml(train: Any, d: DeltaConfig) -> dict[str, Any]:
    """The prime-rl ``InferenceConfig`` (``prime_rl/configs/inference.py``, v0.10.0): vLLM with LoRA hot-loading."""
    return {"server": {"port": d.port},
            "vllm": {"model": d.base_model, "max_model_len": d.seq_len, "tensor_parallel_size": d.infer_tp,
                     "data_parallel_size": d.infer_gpus // d.infer_tp, "gpu_memory_utilization": d.gpu_memory_utilization,
                     "enable_prefix_caching": True, "enable_lora": True, "max_lora_rank": int(train.lora_rank)},
            "weight_broadcast": {"type": "filesystem"}}


def submit_command(script: Path, d: DeltaConfig) -> list[str]:
    """``sbatch`` with the account on the command line and the login environment's override removed (module doc)."""
    return ["env", "-u", "SBATCH_ACCOUNT", "-u", "SLURM_ACCOUNT", "sbatch", f"--account={d.account}", str(script)]


def sbatch_script(d: DeltaConfig, *, out: Path, run_name: str, crucible_cmd: str, setup: str = "") -> str:
    """The job (module doc): trainer on the first ``train_gpus`` GPUs, vLLM on the rest, Crucible's loop on the CPUs."""
    tr = ",".join(str(i) for i in range(d.train_gpus))
    inf = ",".join(str(i) for i in range(d.train_gpus, d.train_gpus + d.infer_gpus))
    env = "".join(f"export {k}={v}\n" for k, v in sorted(d.extra_env.items()))
    return f"""#!/bin/bash
#SBATCH --job-name=crucible-rl-{run_name}
#SBATCH --account={d.account}
#SBATCH --partition={d.partition}
#SBATCH --nodes={d.nodes}
#SBATCH --gpus-per-node={d.gpus_per_node}
#SBATCH --cpus-per-gpu={d.cpus_per_gpu}
#SBATCH --mem={d.mem_gb_per_gpu * d.gpus_per_node}G
#SBATCH --time={d.time}
#SBATCH --output={out}/slurm-%j.out
set -euo pipefail
unset SBATCH_ACCOUNT SLURM_ACCOUNT
{env}{setup}
mkdir -p {out}
cd {d.workdir}
CUDA_VISIBLE_DEVICES={inf} uv run inference @ {out}/inference.toml > {out}/inference.log 2>&1 &
INF=$!
CUDA_VISIBLE_DEVICES={tr} uv run trainer @ {out}/trainer.toml > {out}/trainer.log 2>&1 &
TRN=$!
until curl -sf localhost:{d.port}/health > /dev/null; do sleep 10; kill -0 $INF || exit 3; done
{crucible_cmd} > {out}/crucible.log 2>&1
RC=$?
kill $INF $TRN 2>/dev/null || true
exit $RC
"""


@dataclass
class Plan:
    steps: int
    episodes_per_step: int
    concurrent_episodes: int
    sampling_waves: int
    gpu_hours_per_step: float
    gpu_hours_total: float
    wall_hours_per_step: float
    balance: float | None
    fits: bool | None
    notes: list[str]


# Qwen3.6-35B-A3B (config.json, 2026-10-25): 10 full-attention layers with 2 KV heads of 256 (the 30 gated delta-net layers
# keep a constant ~31 MB state per sequence), so the bf16 KV cache is 10 * 2 * 2 * 256 * 2 = 20,480 bytes per token
KV_BYTES_PER_TOKEN = 20_480
WEIGHTS_GB = 70.0  # 35B parameters in bf16
HBM_GB = 96.0  # GH200 120GB nodes show 97,871 MiB (smoke job 3439628)


def plan(*, steps: int, tasks_per_step: int, group_size: int, d: DeltaConfig, balance: float | None = None, episode_wall_min: float = 28.0,
         sequence_tokens: int = 160_000, mfu: float = 0.30, peak_tflops: float = 989.0, active_params: float = 3.0e9, attn_layers: int = 10,
         heads: int = 16, head_dim: int = 256, reserve: float = 60.0) -> Plan:
    """GPU-hours for a run (module doc). Sampling: as many episodes run at once as the inference GPUs' KV cache holds at
    the median sequence (HBM after the weights, sharded ``infer_tp`` ways), each wave about one measured 300-call episode
    (median ~28 min on Tinker, Worlds included); training: LoRA forward + backward FLOPs (dense over the ~3B active
    parameters, plus the full-attention layers' quadratic term) at the stated MFU. The step's wall time is the slower of
    the two; the job holds every GPU of its nodes for it, which is what Slurm charges. ``reserve``: the benchmark's hours."""
    eps = tasks_per_step * group_size
    tokens = eps * sequence_tokens
    flops = 4.0 * active_params * tokens + 3.0 * attn_layers * 4.0 * sequence_tokens**2 * heads * head_dim * eps
    train_h = flops / (mfu * peak_tflops * 1e12 * d.train_gpus) / 3600.0
    kv_free = d.infer_gpus * max(0.0, HBM_GB * d.gpu_memory_utilization - WEIGHTS_GB / d.infer_tp)
    concurrent = max(1, int(kv_free * 1e9 // (sequence_tokens * KV_BYTES_PER_TOKEN)))
    waves = -(-eps // concurrent)
    sample_h = waves * episode_wall_min / 60.0
    wall = max(train_h, sample_h) + 0.1  # + broadcast, adapter load, packing
    gph = wall * d.gpus
    total = gph * steps
    fits = None if balance is None else total <= balance - reserve
    return Plan(steps=steps, episodes_per_step=eps, concurrent_episodes=concurrent, sampling_waves=waves, gpu_hours_per_step=round(gph, 2),
                gpu_hours_total=round(total, 1), wall_hours_per_step=round(wall, 2), balance=balance, fits=fits,
                notes=["estimate: sequence and episode sizes from the stored 300-call student runs; FLOPs at the stated MFU of GH200 bf16 peak",
                       (f"train {train_h:.2f} h, sampling {sample_h:.2f} h ({waves} wave(s) of {concurrent} episodes) per step on "
                        f"{d.train_gpus}+{d.infer_gpus} GPUs; Slurm charges all {d.gpus}"),
                       f"{reserve:.0f} GPU-hours reserved for the before/after benchmark"])


def render(train: Any, d: DeltaConfig, out: Path, *, run_name: str, crucible_cmd: str, resume_step: int | None = None, setup: str = "") -> dict[str, Path]:
    """Write ``trainer.toml``, ``inference.toml``, ``job.sbatch`` and ``deploy.json`` into ``out`` (a local staging copy of
    the run directory; the paths inside point at the cluster's ``out``)."""
    import tomli_w

    out.mkdir(parents=True, exist_ok=True)
    paths = {"trainer": out / "trainer.toml", "inference": out / "inference.toml", "job": out / "job.sbatch", "deploy": out / "deploy.json"}
    cluster_out = Path(d.workdir) / "runs" / run_name
    paths["trainer"].write_text(tomli_w.dumps(trainer_toml(train, d, cluster_out, resume_step=resume_step)))
    paths["inference"].write_text(tomli_w.dumps(inference_toml(train, d)))
    paths["job"].write_text(sbatch_script(d, out=cluster_out, run_name=run_name, crucible_cmd=crucible_cmd, setup=setup))
    import json

    paths["deploy"].write_text(json.dumps({"delta": asdict(d), "train": asdict(train) if hasattr(train, "__dataclass_fields__") else dict(train),
                                           "prime_rl": PRIME_RL_VERSION, "cluster_out": str(cluster_out), "resume_step": resume_step}, indent=1, default=str))
    return paths


def refuse_tasks(tasks: Sequence[Any], targeting: Mapping[str, Mapping[str, Any]] | None, model: str, *, require_band: bool = True) -> list[str]:
    """Why these tasks may not train on DeltaAI (empty: they may): held-out tasks never train (V9); every task needs a
    recorded band for the student (the Tinker trainer's own rule, ``check_band``)."""
    from crucible.train.bridges.tinker.trainer import check_band

    out = [f"{t.task_id}: held out (V9)" for t in tasks if str(getattr(t, "split", "train") or "train") != "train"]
    if require_band:
        out += check_band([str(t.task_id) for t in tasks], model, targeting)
    return out
