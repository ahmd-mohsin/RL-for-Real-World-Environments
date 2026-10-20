#!/bin/bash
# 8-GPU smoke on one node (scripts/marlowe/stage.sh plan smoke8): NCCL, the FSDP2/HSDP unit scenarios on NCCL, then
# Qwen2.5-7B training steps through Crucible's trainer (token credit path) with a sharded checkpoint, a forced stop (SIGTERM,
# as a preemption sends) and a restore in a new process group. Submit from the login node:
#   source scripts/marlowe/env.sh && sbatch scripts/marlowe/sbatch/smoke8.sbatch
#SBATCH --job-name=cru-smoke8
#SBATCH --account=marlowe-m000159
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/scratch/m000159/muahmed/crucible-runs/slurm/%x-%j.out
exec bash /scratch/m000159/muahmed/crucible/scripts/marlowe/stage.sh smoke8
