#!/bin/bash
# Template for a training plan (one node, 8 GPUs, 4 h, resumable). Copy, set --job-name and the plan, then submit:
#   source scripts/marlowe/env.sh && sbatch --job-name=cru-<name> scripts/marlowe/sbatch/train.sbatch <plan>
# Add the plan to stage.sh first: stages there are guarded by markers in $CRU_RUNS/<job name>-<job id>/markers, and a
# training stage must checkpoint on SIGTERM (preemption, 900 s grace) and resume from its latest checkpoint.
#SBATCH --account=marlowe-m000159
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/scratch/m000159/muahmed/crucible-runs/slurm/%x-%j.out
exec bash /scratch/m000159/muahmed/crucible/scripts/marlowe/stage.sh "${1:?plan}"
