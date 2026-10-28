# Handoff: the DeltaAI inference path (2026-10-26)

> 2026-10-28: the GPU kernel lane (kernel debug and migration) was removed by the lead's decision; history in git before
> c26f069a. The kernel rows and steps below are obsolete.

For the worker who takes over the baseline benchmark harness on DeltaAI. Design and measured numbers are in
`docs/research/swarm-pipeline/14-deltaai.md`. This note is the state, the commands and what is left.

## State

| Piece | State | Evidence |
|---|---|---|
| vLLM serving of `Qwen/Qwen3.6-35B-A3B` (NCSA's v0.28.0 arm64 SIF, TP=2, 262144 context, runtime LoRA) | **works** | ready in 210-334 s on every smoke; KV cache 3.87M tokens (14.8 full-context sequences) |
| `vllm` sampler backend (`train/bridges/vllm`), final backend, backend + engine + `max_tokens` (8192) in each arm's settings | **works** | unit tests; the debug episode below |
| Apptainer `docker` shim with the lockdown (`scripts/deltaai/docker`, `docker_shim.py`) | **works** | lockdown probe clean in the go-toml World on every smoke since 3458633 |
| Debug-lane episode + grade on DeltaAI (`lh_debug_target.py student --backend vllm`) | **works end to end** | smoke 3461769 and 3462338: go-toml, 20 tool calls, 2 compactions, `infra: ok`, reward 0.0 (expected at a 20-call cap) |
| Kernel episode on a node (`lh_kernel_target.py --host gh200-node`) | **agent runs; grade fixed in code, not yet re-run** | 3462338: 20 calls, 2 compactions, then every grade failed in `kernel_profiles.of_spec` (unhashable dict). Fixed in `d0883e51`, with a test |
| Throughput | measured | doc 14 §5 |

**GPU-hours.** The smoke cap of 20 charged GPU-hours is used (20.07 by `sacct`, jobs 3455852-3462338). No run 9 was
submitted. Each smoke billed 4,000 per hour (a full node): the `--mem=200g` request, not the 2 GPUs, set the share.
Request `--mem=110g` or less per 2 GPUs, so a job bills 2 GPU-hours per hour. The account showed 954 of 1,000 left on
2026-10-26, which includes other workers' jobs.

**Tinker:** $0 spent.

## Where things are

- **Code (this branch):**
  - `crucible/src/crucible/train/bridges/vllm/` (`client.py`, `serve.py`, `sampler.py`);
  - `crucible/src/crucible/world/lockdown.py`;
  - `crucible/scripts/deltaai/` (`docker`, `docker_shim.py`, `probe.sbatch`, `smoke.sbatch`, `throughput.py`);
  - `--backend vllm` in `scripts/lh_debug_target.py`, `scripts/tinker_rl.py` and (through it) `scripts/lh_kernel_target.py`;
  - the `gh200-node` host in `factory/lanes/kernel_episode.py`;
  - `require_offline` in `factory/tinker.py`;
  - the knobs in `core/env.py`;
  - tests in `tests/unit/test_deltaai_vllm.py`.
- **On DeltaAI** (`W=/work/hdd/bjhg/mahmedmohsin/crucible-infer`):
  - `$W/code/crucible`: an rsync of this worktree's `crucible/`; re-sync before a run;
  - `$W/venv`: Python 3.12, crucible editable, transformers, CPU torch;
  - `$W/hf`: the weights, HF cache, revision `995ad96`;
  - `$W/vcache`: vLLM, Triton and FlashInfer caches;
  - `$W/sif`: World SIFs plus `.json` side files; the kernel base SIF is a symlink to `../crucible/sif`;
  - `$W/images`: `docker save` archives from the Mac;
  - `$W/data/debug`: training data only, no held-out stores, plus the 14 training clones in `src/`;
  - `$W/data/kernel_debug/store/train/kd-bug-llamacpp-ca21f3a1`;
  - `$W/logs` and `$W/out/smoke-<job>` (`episode.log`, `kepisode.log`, `shim.log`, `throughput.jsonl`, `engine.json`, `run/`, `krun/`).
- **The kernel GH200 gate bundle** (another worker's): `/work/hdd/bjhg/mahmedmohsin/crucible/bundle`.

## How to run a smoke (and the pattern for benchmark jobs)

```bash
# from the Mac, after code changes
rsync -a --delete --exclude data --exclude .venv --exclude wandb --exclude .coverage --exclude __pycache__ \
  <worktree>/crucible/ deltaai:/work/hdd/bjhg/mahmedmohsin/crucible-infer/code/crucible/
# the shim and job files live at $W/bin and $W (copy scripts/deltaai/docker and docker_shim.py into $W/bin)
ssh deltaai 'cd /work/hdd/bjhg/mahmedmohsin/crucible-infer && unset SBATCH_ACCOUNT SLURM_ACCOUNT \
  && export SLURM_CONF=/var/spool/slurmd/conf-cache/slurm.conf \
  && sbatch --account=bjhg-dtai-gh --partition=ghx4 --mem=110g smoke.sbatch'
```

Rules:
- **Account:** `bjhg-dtai-gh` only. Verify it with `squeue -o %a`.
- **Partition:** `ghx4-interactive` allows **one** job per user, and the trainer worker uses it. Use `ghx4`, which waits on fair-share, often 24 h or more.
- **Login nodes:** no compute there.
- **HF token:** it enters a job only through a mode-600 env file.

What every job must do (it's all in `smoke.sbatch`):
1. Put `$W/bin` (the shim) and `$W/venv/bin` first on `PATH`.
2. Set `CRUCIBLE_SIF_DIR`, `CRUCIBLE_SHIM_{STATE,SCRATCH,CONFIGDIR,ALLOW}` (allow = the job's node-local `$J`), `TMPDIR=$J` and `CRUCIBLE_SHIM_LOG`.
3. **Copy the lane data to node-local disk without ACLs.** `/work`'s ACLs ride into the graders' temp trees, and `cp -a` into the fuse overlay fails with exit 125. Then repoint the clones' git alternates.
4. Run the lockdown probe in the task's World image and refuse to continue unless `lockdown.verdict` is empty.
5. Start the server with `ServeSpec(..., cache_dir=$W/vcache)` and wait for `wait_ready`.
6. Run the arms with `--backend vllm --vllm-engine $O/engine.json`; `max_tokens` defaults to 8192.

## Next steps (the benchmark harness)

1. **Re-run the kernel smoke on `ghx4`** (needs new GPU-hours approved; about 2-4) to confirm `d0883e51`: the kernel episode should then grade with `infra: ok`.
2. **Images for the benchmark tasks:** `docker save` each sealed task's World image (and its `-pw` policy image if any) on the Mac, copy it to `$W/images`, and build SIFs with the `.json` side file (`docker image inspect --format '{{json .Config}}'`). Held-out data stays off git and off the Mac's public paths; on DeltaAI keep it in a separate, mode-700 directory.
3. **A `bench.sbatch`** from `smoke.sbatch`: one server, then `tinker_rl.py eval --backend vllm --purpose baseline --seed <fresh from lh_debug.py seal> --k 8 --rows <sealed rows>` (or `lh_debug_target.py student --backend vllm --k 8` per task). Use `--student-concurrency` up to about 14 per server (the KV cache holds 14.8 full contexts). One node with TP=2 + TP=2 can run two servers.
4. **The trained arm:** `tinker_rl.py compare --backend vllm --checkpoint <PEFT adapter dir, rank ≤ 64>` on the same engine build. The trainer must export PEFT format.
5. **Not done:**
   - `crucible-lh-base:py312` has no SIF. The window lane inspects it, though go-toml never needed it; export it if a Python task does.
   - Per-container cgroup limits are impossible in fakeroot mode.
   - The NCSA ticket (doc 14 §3) is drafted but not sent.
6. **Safety caveat for RL at scale:** model code runs as the user's uid (no subuid range). The lockdown holds (offline, no host files, no secrets, private PID namespace), but there is no second uid. See doc 14 §3, "What stays unprotected".

## Commits (this branch)

`41795e68` the inference path · `dc4e8358` the on-node kernel host · `38eb7e0a` fakeroot command off · `8c203254`
8192 tokens, HOME/caches, key check · `3cb00897` `-w` dirs · `3493034e` doc 14 numbers · `033810ee` cp owner, exec
pwd, alternates · `b7e0128b` cp rename, shim log · `b2f51a72` node-local data without ACLs, cp clean env · `6f7c3f2d`
kernel kit modules · `d0883e51` kernel spec harness descriptor.
