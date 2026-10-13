#!/bin/bash
# Crucible multi-GPU stages on REPACSS (TTU): 2 nodes x 4 NVIDIA H100 NVL 94 GB = 8 GPUs, so every 8-GPU stage is multi-node.
# The staged plan of docs/audit/2026-10-12/M12-sweep-G5.md section 5. Nothing here runs by itself: GPU time needs the
# user's approval. Submit one stage at a time from the login node (scripts/repacss.sh), from the repository root:
#
#   STAGE=stage0 sbatch --time=00:20:00 scripts/repacss_train.sbatch    # allocation, topology, NCCL bandwidth   ~0.3 h x 8 GPUs =  2 GPU-h
#   STAGE=stage1 sbatch --time=00:30:00 scripts/repacss_train.sbatch    # unit scenarios on NCCL, W=4 and W=8    ~0.4 h x 8      =  3 GPU-h
#   STAGE=stage2 sbatch --time=00:30:00 scripts/repacss_train.sbatch    # 8B bf16 weight broadcast + disk publish ~0.4 h x 8     =  3 GPU-h
#   STAGE=stage3 sbatch --time=01:00:00 scripts/repacss_train.sbatch    # FSDP2 1.7B parity, 8B peak memory      ~0.75 h x 8     =  6 GPU-h
#   STAGE=stage4 sbatch --nodes=1 --ntasks-per-node=1 --time=01:00:00 scripts/repacss_train.sbatch
#                                                                        # vLLM sleep/wake/load, TP1 and TP2      ~0.75 h x 4     =  3 GPU-h
#   STAGE=stage5 sbatch --time=01:00:00 scripts/repacss_train.sbatch    # disaggregated GRPO, Qwen3-1.7B, 20 steps ~0.75 h x 8    =  6 GPU-h
#   STAGE=stage6 sbatch --time=05:00:00 scripts/repacss_train.sbatch    # full GRPO, Qwen3-8B backlog, 60 steps  3-4 h x 8        = 24-32 GPU-h
#                                                                        # total about 50-55 GPU-h
#
# Pass criteria are printed by each stage and written as JSON under $OUT (stage0.json ... gate_dist.json).
#
# Knobs (environment at submit time):
#   REPO            repository root on shared storage (default: the submit directory)
#   PY              python with crucible installed (default: $REPO/crucible/.venv/bin/python)
#   RUN_ROOT        where runs go; MUST be storage every node sees (home / project / Lustre), never /tmp
#   CKPT_DIR        published weights for the engines (default $OUT/ckpt; shared, since engines on node 1 read it)
#   NCCL_CUMEM_ENABLE           trainer NCCL (default 0: Modal needed it with NCCL 2.30.7; stage0 measures 0 and 1)
#   ENGINE_NCCL_CUMEM_ENABLE    vLLM servers (default 1: OpenRLHF reports that 0 breaks ncclCommInitRank on NCCL >= 2.27
#                               with vLLM >= 0.16, trainer/ray/ppo_actor.py:502-509)
#   NCCL_SOCKET_IFNAME / NCCL_IB_HCA   the inter-node interface, once stage0 shows which one carries the traffic
#   CRUCIBLE_PG_TIMEOUT_S       process-group timeout (default 1800 s, explicit on every init_process_group)
#   CRUCIBLE_SPMD_STALE_S       a driver / follower silent this long is lost and the job ends (default 300 s)
#   MODEL_SMALL / MODEL_LARGE   Qwen/Qwen3-1.7B and Qwen/Qwen3-8B (pre-download into $HF_HOME on the login node)
#
#SBATCH --job-name=crucible-m12
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:nvidia_h100_nvl:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=crucible-m12-%j.out

set -euo pipefail
STAGE=${STAGE:?set STAGE=stage0 ... stage6}
REPO=${REPO:-$SLURM_SUBMIT_DIR}
CRU=$REPO/crucible
PY=${PY:-$CRU/.venv/bin/python}
RUN_ROOT=${RUN_ROOT:-$REPO/runs}
OUT=$RUN_ROOT/$SLURM_JOB_ID-$STAGE
CKPT_DIR=${CKPT_DIR:-$OUT/ckpt}
MODEL_SMALL=${MODEL_SMALL:-Qwen/Qwen3-1.7B}
MODEL_LARGE=${MODEL_LARGE:-Qwen/Qwen3-8B}
mkdir -p "$OUT" "$CKPT_DIR"
case "$RUN_ROOT" in /tmp/*|/dev/shm/*) echo "RUN_ROOT must be shared storage, not $RUN_ROOT" >&2; exit 2 ;; esac

# rendezvous: the first host of the allocation; one port per job
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export NNODES=$SLURM_JOB_NUM_NODES

# NCCL / torch.distributed baseline
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export CRUCIBLE_ENGINE_NCCL_CUMEM_ENABLE=${ENGINE_NCCL_CUMEM_ENABLE:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
if [[ -n ${NCCL_SOCKET_IFNAME:-} ]]; then export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}; fi
export CRUCIBLE_PG_TIMEOUT_S=${CRUCIBLE_PG_TIMEOUT_S:-1800}
export CRUCIBLE_SPMD_STALE_S=${CRUCIBLE_SPMD_STALE_S:-300}
export CRUCIBLE_SPMD_JOIN_S=${CRUCIBLE_SPMD_JOIN_S:-3600}
export PYTHONPATH=$CRU/src:$CRU/packs/_toy/src${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # compute nodes read the cache the login node filled
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

GPN=${SLURM_GPUS_ON_NODE:-4}
# one task per GPU (torchrun style: RANK = SLURM_PROCID, LOCAL_RANK = SLURM_LOCALID, from crucible.train.cluster.slurm_env)
per_gpu() { local nodes=$1; shift; srun --kill-on-bad-exit=1 --nodes="$nodes" --ntasks-per-node="$GPN" --gres=gpu:nvidia_h100_nvl:"$GPN" "$@"; }
# one task per node: cluster.run_node starts that node's processes of the placement
per_node() { srun --kill-on-bad-exit=1 --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --gres=gpu:nvidia_h100_nvl:"$GPN" "$@"; }
CHECK=("$PY" "$CRU/scripts/slurm_check.py")

echo "stage=$STAGE nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR:$MASTER_PORT out=$OUT cumem=$NCCL_CUMEM_ENABLE"
case "$STAGE" in
  stage0)  # allocation smoke: rank sum 28 at W=8, topology (NV pairs vs PCIe), busbw pair / node / inter-node, both CUMEM settings
    for cm in 0 1; do
      NCCL_CUMEM_ENABLE=$cm per_gpu "$SLURM_JOB_NUM_NODES" "${CHECK[@]}" nccl --out "$OUT/cumem$cm"
    done ;;
  stage1)  # the gloo unit scenarios on NCCL: W=4 inside node 0, W=8 across both nodes; the D1 regression (7 rows on 8 ranks)
    for sc in fsdp hf_layer few_items; do per_gpu 1 "${CHECK[@]}" unit "$sc" --out "$OUT/w4"; done
    per_gpu 1 "${CHECK[@]}" unit spmd_run --out "$OUT/w4"
    for sc in fsdp hsdp hf_layer few_items spmd_run; do per_gpu 2 "${CHECK[@]}" unit "$sc" --out "$OUT/w8"; done
    "$PY" "$CRU/scripts/slurm_check.py" report --out "$OUT/w4"
    "$PY" "$CRU/scripts/slurm_check.py" report --out "$OUT/w8" ;;
  stage2)  # Qwen3-8B bf16 (16.4 GB) from rank 0 to 3 and to 7 receivers: bitwise, >= 7 GB/s in node, publish < 10 s; disk write NVMe vs shared
    per_gpu 1 "${CHECK[@]}" weightsync --model "$MODEL_LARGE" --out "$OUT/w4" --local-dir "${TMPDIR:-/tmp}/crucible-$SLURM_JOB_ID"
    per_gpu 2 "${CHECK[@]}" weightsync --model "$MODEL_LARGE" --out "$OUT/w8" --local-dir "${TMPDIR:-/tmp}/crucible-$SLURM_JOB_ID" ;;
  stage3)  # FSDP2 parity on a real model (meta-device load from rank 0): (1,8) vs (2,4) vs one GPU; then 8B peak memory at (2,4)
    W=$((SLURM_JOB_NUM_NODES * GPN)); HSDP=${SLURM_JOB_NUM_NODES}x$GPN   # (1,8) across nodes vs (2,4): shards inside each node
    per_gpu 2 "${CHECK[@]}" fsdp --model "$MODEL_SMALL" --meshes "1x$W,$HSDP" --tag parity --out "$OUT"
    for tb in 8192 16384; do
      per_gpu 2 "${CHECK[@]}" fsdp --model "$MODEL_LARGE" --meshes "$HSDP" --no-ref --items 16 --prompt 4096 --completion 4096 \
        --token-budget "$tb" --max-peak-gb 85 --tag "8b-tb$tb" --out "$OUT"
    done ;;
  stage4)  # one node: vLLM sleep -> load -> wake x50 at TP1, then TP2 on a bridged pair; memory after wake within 1 GB of the first
    NCCL_CUMEM_ENABLE=$CRUCIBLE_ENGINE_NCCL_CUMEM_ENABLE srun --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --gres=gpu:nvidia_h100_nvl:"$GPN" \
      "${CHECK[@]}" vllm --model "$MODEL_SMALL" --tp 1 --out "$OUT"
    NCCL_CUMEM_ENABLE=$CRUCIBLE_ENGINE_NCCL_CUMEM_ENABLE srun --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --gres=gpu:nvidia_h100_nvl:"$GPN" \
      "${CHECK[@]}" vllm --model "$MODEL_SMALL" --tp 2 --out "$OUT" ;;
  stage5)  # disaggregated GRPO: node 0 = 4 FSDP ranks (mesh inside the node), node 1 = engines; word-file task, 20 steps
    per_node "$PY" -m crucible_pack_toy.m12.gate_dist --nodes 2 --gpus "$GPN" --cluster repacss --train-gpus "$GPN" \
      --model "$MODEL_SMALL" --params-b 1.7 --steps 20 --eval 24 --checkpoint-every 10 --master-port "$MASTER_PORT" \
      --ckpt-dir "$CKPT_DIR" --out "$OUT" --sinks "jsonl:$OUT/metrics" ;;
  stage6)  # full GRPO on Qwen3-8B, backlog task, 60 steps, checkpoint every 10, 4 h wall clock
    per_node "$PY" -m crucible_pack_toy.m12.gate_dist --nodes 2 --gpus "$GPN" --cluster repacss --train-gpus "$GPN" \
      --model "$MODEL_LARGE" --params-b 8.2 --task backlog --tickets 3 --steps 60 --eval 24 --checkpoint-every 10 \
      --max-wall-s 14400 --lr 1e-6 --master-port "$MASTER_PORT" --ckpt-dir "$CKPT_DIR" --out "$OUT" --sinks "jsonl:$OUT/metrics" ;;
  *) echo "unknown STAGE=$STAGE (stage0 ... stage6)" >&2; exit 2 ;;
esac
echo "done: $OUT"
