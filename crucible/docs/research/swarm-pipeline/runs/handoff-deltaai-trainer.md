# Handoff: the DeltaAI trainer (2026-10-26)

This is for the worker who continues the DeltaAI trainer for Qwen/Qwen3.6-35B-A3B. The design and its evidence are in
`docs/research/swarm-pipeline/14-deltaai-trainer.md`; read §2 (design), §3 (plan) and §5.1 (attempts) first.

## State

- **Code**: done and tested on the Mac.
  - The prime-rl v0.10.0 bridge, the deployment, routing replay, the GPU kernel check, and the build/run split.
  - Tests pass (`tests/unit/test_primerl_bridge.py`, `tests/unit/test_primerl_routing.py`: 19). The 5 import contracts hold.
  - The work sits on branch `worktree-agent-ac56c6865b3f80e19`, which is not merged.
- **Environment on DeltaAI: built.**
  - `/work/hdd/bjhg/mahmedmohsin/crucible-train/prime-rl` at v0.10.0, its venv from the lockfile, marker
    `.crucible-installed-v0.10.0`.
  - torch 2.13.0+cu130, vLLM 0.31.0. The compute nodes run driver 595.71.05.
  - Crucible's source is rsynced to `/work/hdd/bjhg/mahmedmohsin/crucible-train/crucible/{src,scripts,pyproject.toml}`.
- **No RL step has run on a GPU yet.**
  - Attempts 1, 3 and 4 failed in the install. All three causes are fixed (doc 14 §5.1). Attempt 2 was cancelled.
  - Attempt 5 ran as job **3467688** (`ghx4-interactive`) and failed after 20:52 (0.70 GPU-hours at 2 GPUs): no
    `Python.h` for Triton, no `vllm-router`, and `attn = "sdpa"` invalid (doc 14 §5.1). All three are fixed in the
    code (2026-10-27), but **not yet on the cluster**: the SSH control socket closed again before the rsync, the
    rebuild (`build_env.sh`, which now moves the venv to a uv-managed CPython 3.12) and attempt 6.
  - Next: `ssh -fN deltaai` (the user), rsync `crucible/{src,scripts,pyproject.toml}`, re-render the smoke
    (`render-smoke` below) and rsync it to `runs/smoke1`, run `build_env.sh` on the login node until `BUILD_OK`, then
    submit from `runs/smoke1`.
- **GPU-hours used**: about 2 on bjhg-dtai-gh for attempt 1 (the interactive partition billed a whole node) and 0.70 for
  attempt 5. The last balance seen was 974.

## Next steps

1. **Reconnect.** The user runs `ssh -fN deltaai` in their own Terminal, because DeltaAI needs password + Duo. Never
   enter a password yourself.
2. **Find out what happened to attempt 5.** On the login node, if Slurm hits DNS errors, set
   `SLURM_CONF=/var/spool/slurmd/conf-cache/slurm.conf`:
   `sacct -X -j 3464731 -o JobID,Partition,State,Elapsed,AllocTRES%60`.
   - **If it ran,** read `runs/smoke1/`:
     - `slurm-3464731.out`: versions, the kernel check, both phases, and the peak GPU memory per GPU at the end;
     - `smoke1.json`, `smoke2.json`: the trainer step indices, `probe_changed` (LoRA sync), the checkpoint ref;
     - `kernel_check.json`: the 35B's Gated DeltaNet kernels and vLLM support;
     - `trainer*.log`, `inference.log`.
   - **If it is still pending,** you can also submit a copy to ghx4-interactive (`sbatch --account=bjhg-dtai-gh
     job.sbatch` in `runs/smoke1`, its default partition). That partition allows one job per user, and the inference
     worker uses it too. Cancel the copy that does not start, and only our `crucible-rl-smoke` on bjhg-dtai-gh.
3. **Report the smoke:** pass or fail, GPU-hours per step (sacct AllocTRES × Elapsed / 3 steps), and peak memory. Then
   fill in doc 14 §5.1.
4. **Run the real-model 2-step run** (about 8 GPU-hours): `delta_rl.py render --steps 2 ...` (doc 14 §4), on 2 trainer
   + 2 vLLM GPUs with `cp = 2` at `seq_len 262144`. Check:
   - **memory headroom:** does `cp = 2` fit, or does the trainer need the whole node?
   - **routing replay:** `routing/replayed_share` and `routing/coverage_mean` should be 1.0;
   - **`kl/mismatch`** compared with one `--no-router-replay` step.
5. **Only then** the 40-step run (about 165 GPU-hours, doc 14 §3).

## Rules

These come from the user and the coordinator.

- **Accounts:** DeltaAI account `bjhg-dtai-gh` only. Unset `SBATCH_ACCOUNT` / `SLURM_ACCOUNT` and pass `--account` on the
  command line; verify with `squeue -o %a`. Never touch the bfgp jobs.
- **Resources:** keep the memory request proportional to the GPUs. No compute on login nodes: the environment build there
  only downloads, unpacks and imports.
- **Spend:** no Tinker spend.
- **Checkpoints:** report only on a job's final state, with no interim "still queued" messages.

## Files

| what | where |
|---|---|
| bridge | `crucible/src/crucible/train/bridges/primerl/` (`backend.py`, `wire.py`, `loss.py`, `routing.py`, `deploy.py`, `smoke.py`, `kernel_check.py`) |
| CLI | `crucible/scripts/delta_rl.py` (`plan`, `render`, `render-smoke`, `train`, `smoke`); `crucible/scripts/tinker_rl.py` (shared flag builders) |
| tests | `crucible/tests/unit/test_primerl_bridge.py`, `crucible/tests/unit/test_primerl_routing.py` |
| docs | `crucible/docs/research/swarm-pipeline/14-deltaai-trainer.md`; `crucible/DESIGN.md` (the DeltaAI trainer bridge paragraph); `crucible/THIRD_PARTY.md` (prime-rl row) |
| cluster | `/work/hdd/bjhg/mahmedmohsin/crucible-train/{prime-rl,crucible,runs/smoke1,login_build.log,uv-cache,hf_cache}` |
| smoke staging (Mac, scratch) | regenerate with `delta_rl.py render-smoke --stage <dir> --partition ghx4-interactive --time 01:55:00 --gpus 2 --mem-gb 120` |

## Commits (branch `worktree-agent-ac56c6865b3f80e19`, oldest first)

| commit | what |
|---|---|
| 8f85ed5d | the bridge, loss, deployment, plan, shared flags |
| b0c34f99 | install fixes after attempt 1 |
| 7a38fbdb | merge of main (b4d7f1d1's limit-stop rule) |
| 8f494aab | 8192 output tokens per call; a truncated reply is a limit stop |
| 460d2fe7 | stale install marker guard |
| a5013772 | frozen lockfile install, full-history submodules |
| 5cc80386 | login-node build (`build_env.sh`); driver and linear-attention findings |
| b14bfa06 | GPU kernel check |
| dfb15631 | attempt 5 noted in doc 14 |
| b0316bc7 | peak GPU memory in the smoke |
| c427c397 | MoE routing replay |
