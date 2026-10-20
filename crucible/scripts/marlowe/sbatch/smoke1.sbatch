#!/bin/bash
# 1-GPU smoke (scripts/marlowe/stage.sh plan smoke1): GPU in an Apptainer container, vLLM serving Qwen2.5-Coder-7B, one
# Crucible rollout through the Apptainer World, graded, with a receipt. Submit from the login node:
#   source scripts/marlowe/env.sh && sbatch scripts/marlowe/sbatch/smoke1.sbatch
# The script is copied at submission; the stages it runs are read from stage.sh when the job starts (and on every requeue).
#SBATCH --job-name=cru-smoke1
#SBATCH --account=marlowe-m000159
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/scratch/m000159/muahmed/crucible-runs/slurm/%x-%j.out
exec bash /scratch/m000159/muahmed/crucible/scripts/marlowe/stage.sh smoke1
