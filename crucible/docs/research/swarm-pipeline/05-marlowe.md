# Crucible on Marlowe (Stanford, 31 nodes × 8 H100 80 GB)

Brought up 2026-10-19. Code: `world/backends/apptainer.py`, `scripts/marlowe/`, `tests/unit/test_marlowe_apptainer.py`.
The user's own notes: `inference-time-uncertainty/MARLOWE.md` (operations) and `MARLOWE_STRUCTURE.md` (accounts).

## 1. The only route, and the rules

- `-A marlowe-m000159 -p preempt --requeue`: 4 h per slot, uncharged, GraceTime 900 s (SIGTERM, then a kill 15 min later).
  m000215 is blocked and its scratch is 90% full: never use it. `marlowe-users` has MaxJobs=0.
- Priority is multifactor: age dominates (weight 10M, max 7 days), then QoS and GPU count (an 8-GPU job is predicted
  earlier than a 1-GPU job). **A requeue restarts the age clock**, so a job never requeues itself without need.
- Slurm is not on PATH: `source scripts/marlowe/env.sh` sets `SLURM_CONF` and the path, plus every directory below.
- Access: `ssh marlowe` through the user's ControlMaster (`ssh -O check marlowe`; a human opens it daily).

## 2. Where things live

`/users/muahmed` was **over its quota** on 2026-10-19 (every write fails with EDQUOT: about 18 GB used, of which
`~/.cache/pip` is 7.3 GB and `~/containers` 6.4 GB). Persistent state therefore lives on the m000159 scratch
(`CRU_ROOT`, 5 TB group quota, **512k-file group limit**). Point `CRU_ROOT` at `/users/muahmed` once space is freed.
`env.sh` also sends every library cache that defaults to `$HOME` (triton, inductor, CUDA JIT, vLLM, flashinfer) to
node-local `/tmp`: one write under the full home would fail with EDQUOT and take a job down.

| Path (under `/scratch/m000159/muahmed`) | Holds |
|---|---|
| `crucible/` | the code (rsync of the repo's `crucible/`, no `.git`, `.env`, `.venv`, `data/`) |
| `crucible-env/` | the venv: Python 3.12.15, torch 2.13.0+cu130, vLLM 0.32.0, transformers 5.19.0, NCCL 2.29.7; freeze in `crucible-env.freeze.txt` |
| `sif/` | `world-py312.sif` (72 MB, sha256 7d72426d1b81…), `cuda128.sif` (4.8 GB, f50f5cad0350…) |
| `hf/` (`HF_HOME`) | Qwen2.5-Coder-7B-Instruct @ c03e6d35, Qwen2.5-7B-Instruct @ a09a3545 (15 GB each, 4 safetensors shards, 339 tensors) |
| `crucible-runs/<job name>-<job id>/` | one run: `markers/`, `logs/`, `driver.log`, receipts, `summary.json`; Slurm output in `crucible-runs/slurm/` |
| `cache/apptainer-config` | `APPTAINER_CONFIGDIR`: instance state (the default `~/.apptainer` is on the full home) |
| `.setup/` | setup markers; `.setup/ready` gates every job |

**CUDA.** The nodes run driver 580.173.02 (CUDA 13.0). vLLM 0.32.0 pins torch 2.13.0 for CUDA 13.0, and that pair is
the one that imports: the user's earlier env (vLLM 0.24 forced onto torch 2.11+cu128) died with `libcudart.so.13` missing.
CUDA 12.x containers (`cuda128.sif`, migration task04's base) run on the same driver through `--nv`.

## 3. Setup (login node, no GPU)

```bash
rsync -a --exclude .venv --exclude .env --exclude .git --exclude data crucible/ marlowe:/scratch/m000159/muahmed/crucible/
ssh marlowe 'bash /scratch/m000159/muahmed/crucible/scripts/marlowe/setup_env.sh > /scratch/m000159/muahmed/crucible-setup.log 2>&1'
```

Idempotent steps with markers: uv Python → venv → vLLM → Crucible `[train,llm,answers]` + the toy pack (editable,
`--no-sources`) → `build_sifs.sh` (x86_64 SIFs built on the cluster from `defs/*.def`, unprivileged; never ship Mac
images, they are arm64) → model downloads (public, no token) → an import check → the uv cache removed (it alone held 64k
files) → `.setup/ready`. No `.env` and no secret is ever copied to the cluster.

## 4. The Apptainer sandbox: what isolation is achieved

Tested on the login node (apptainer 1.5.4, non-setuid, user namespaces allowed):

| Property | Result |
|---|---|
| Network | `--net --network none` works unprivileged: loopback only, a public connect fails (ENETUNREACH); `egress_probe` and `isolation_report` check it from inside |
| Filesystem | `--containall`: private /tmp and /dev; no host home, /users, code, venv, HF cache or /cm/shared; image read-only, writes go to a tmpfs overlay; only the workspace is bound, at its host path |
| Environment | `--cleanenv` on start and on every exec (an exec without it leaks the host environment in: measured) |
| Processes | own PID namespace (4 processes visible); `instance stop` kills what the policy left behind (checked with a detached `setsid sleep`) |
| Identity | the invoking user: no root, no setuid, no capabilities; no `policy_user` demotion is possible |
| Limits | `--memory/--cpus/--pids-limit` **refused on the login node** (no delegated cgroup); `cgroups="auto"` starts without them and the description says so; refused on compute node n02 too (job 523884), so inside a job the Slurm cgroup bounds the World |
| GPU | `--nv` binds the driver and the job's GPUs (`gpu_container` stage) |

No fallback was needed: every flag above works unprivileged. enroot is also installed (not tested; pyxis not checked).

Smoke on the login node (CPU): the window lane's World (`common/git_repo`, seeded with one commit tagged `base`) with its
tools in the instance; a scripted policy fixed the toy repository; graded by hidden tests in a fresh container: reward
1.0, isolation ok, instance gone after close (`stage.sh selftest`, receipt in `crucible-runs/selftest-login3/`).

## 5. Jobs: the stage driver

`scripts/marlowe/stage.sh <plan>` runs inside `sbatch/*.sbatch` (thin wrappers: the plan is read from `stage.sh` at every
start, so it can be fixed while the job waits).
- Each stage writes `markers/<stage>.done` only on exit 0 and never runs again; a requeued job keeps its id and resumes
  in the same run directory on any node. A failure is retried once (`CRU_STAGE_TRIES`), then marked and skipped;
  `--after` names dependencies.
- SIGTERM (preemption) is forwarded to the stage's process group, which checkpoints; the driver waits up to 840 s,
  records `interruptions.jsonl` and exits for Slurm's requeue. `--signal=B:USR1@600` (time limit near) does the same and
  then requeues the job itself.
- Without `.setup/ready` a job requeues instead of burning its slot.

Plans: `smoke1` (1 GPU: `probe`, `gpu_container`, `rollout`), `smoke8` (8 GPUs: `probe`, `nccl`, `units`, `train_a`,
`train_b`), `selftest` (login node). `sbatch/train.sbatch` is the template for real training plans.

**smoke1.** `gpu_container`: a `--nv`, no-network instance of `cuda128.sif`: nvidia-smi, nvcc `-arch=sm_90` of a
vector-add kernel, run in the instance and again in a fresh container. `rollout`: vLLM serves Qwen2.5-Coder-7B
(127.0.0.1, hermes tool parser), `LLMToolPolicy` works the toy task through the Apptainer World, graded, receipt
`rollout-vllm.receipt.json`.

**smoke8.** `nccl` and `units` reuse `scripts/slurm_check.py` (the tests/unit FSDP2, HSDP and SPMD scenarios on NCCL at
W=8). `train_a`/`train_b` (`train_smoke.py`): Qwen2.5-7B on an HSDP 2×4 mesh, meta-device build with the fp32 state
scattered from rank 0, bf16 mixed precision, 16 scripted 3-turn rollouts in 4 groups with per-call `Turn.advantage`
(`turn_weight` 0.5: the token credit path), behaviour log-probs from the initial weights. Phase a: steps 1–4, a DCP
checkpoint after step 2, then SIGTERM to every rank (as a preemption sends it) and a checkpoint inside the grace period.
Phase b, in new processes: restore step 2 (shard checksums must match bitwise), redo steps 3–4 (loss and grad norm
within 2%), compare with the TERM checkpoint, `verdict.json`. Checkpoints (~90 GB each) are removed after a pass.
The tiny-model CPU run of both phases passes on the login node and on a laptop.

**Deviation.** The "forced requeue" is a forced stop and restart inside the allocation: a real `scontrol requeue` would
reset the job's age priority and cost another multi-day wait. Phase b runs exactly the requeue path (new processes,
state only from shared storage); a real preemption goes through the same handler and markers.

## 6. Jobs (2026-10-19)

| Job | Plan | Resources | Status |
|---|---|---|---|
| 523884 | smoke1 | 1 GPU, 14 CPU, 160 GB, 2 h | backfilled at once, ran 3 min on n02. `probe` ok (H100, driver 580.173.02, torch cu130, NCCL 2.29.7). `gpu_container` **passed**: nvidia-smi in the instance, nvcc 12.8 `sm_90` kernel correct in the instance and in a fresh container, no network. `rollout` failed: vLLM JIT-compiled flashinfer's sampler against the node's `/usr/local/cuda`, which lacks `curand.h`. Fixed (`VLLM_USE_FLASHINFER_SAMPLER=0`, `CUDA_HOME` = the venv's CUDA 13 toolkit, checked with nvcc on the login node) |
| 524074 | smoke1 | the same | queued again for the rollout (`--test-only`: 2026-10-23 05:17) |
| 523883 | smoke8 | 1 node, 8 GPU, exclusive, 3 h | queued (`--test-only` at submit: 2026-10-21 22:47; `squeue --start`: 2026-10-26 00:06) |

Check: `squeue -u $USER --start`; results: `crucible-runs/cru-smoke{1,8}-<id>/summary.json` and the receipts beside it.
