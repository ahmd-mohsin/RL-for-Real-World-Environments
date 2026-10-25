# 14. DeltaAI: the benchmark and RL episodes on our own GH200s

**The decision (the user, 2026-10-25).** The before/after benchmark of `Qwen/Qwen3.6-35B-A3B` and the RL training run on
NCSA DeltaAI (allocation CIS262200, Slurm account `bjhg-dtai-gh`, 1,000 GPU-hours), not on Tinker. Tinker stays the
task-creation backend (the swarm's builders and checkers). This doc covers everything but the trainer: serving the
student, the episode Worlds, the sandbox for model-written code, and the benchmark harness.

**The hardware.** GH200 nodes: aarch64 Grace CPU (72 cores) and four H100-class GPUs per node (sm_90, 96 GB usable each),
3.5 TB node-local NVMe at `/tmp`, Slingshot 11. No Docker; Apptainer 1.4.2 (non-setuid, user namespaces, fuse-overlayfs).
One uid per user and no subuid range.

## 1. What runs where

| Piece | On DeltaAI | Same as on Tinker / the Mac |
|---|---|---|
| Student sampler | our vLLM server on the node (`train/bridges/vllm`), token ids in and out | the Tinker bridge's engine side: `TinkerSampler`, `TinkerChatCodec`, per-call version and log |
| Episode Worlds and graders | Apptainer instances through the `docker` shim (`scripts/deltaai/docker`) | the lanes' own Docker code, unchanged; the SIFs are the Mac's images (`docker save`, then `apptainer build docker-archive://`) |
| Benchmark arms | `tinker_rl.py eval` / `compare` and `lh_debug_target.py student` with `--backend vllm` | the same `factory.tinker.benchmark` and protocol check |
| Trainer | the other worker's (not this doc) | — |

## 2. Serving the student

**The engine: NCSA LLMHub's vLLM container** `/sw/llmhub/vllm-v0.28.0-slingshot-deltaai.sif`, sha256
`4395a0c67b8fc3752d9571c898456820e9bdfc7feb21d075332be1f71497146c`: upstream `vllm/vllm-openai:v0.28.0` (arm64, Ubuntu
24.04, CUDA 13) with the site's Slingshot libfabric and aws-ofi-nccl. Chosen over a pinned aarch64 wheel because the site
already solved the DeltaAI-specific constraints in it (its recipe: libfabric needs glibc ≥ 2.38, so v0.28.0 is the
first usable arm64 base; NCCL over Slingshot). Measured in the probe job (3455864): `vllm --version` 0.28.0; it registers
`Qwen3_5MoeForConditionalGeneration`, the student's architecture; every flag we use exists (`--enable-lora`,
`--max-lora-rank`, `--max-loras`, `--generation-config`, `--enable-prefix-caching`, `--enable-prompt-tokens-details`,
`--return-tokens-as-token-ids`; `--disable-log-requests` is gone in 0.28: requests are not logged by default).

**The weights.** `Qwen/Qwen3.6-35B-A3B` at HF revision `995ad96eacd98c81ed38be0c5b274b04031597b0`: 71.9 GB of bf16
safetensors (40 files), `model_type` `qwen3_5_moe`, `max_position_embeddings` 262144. Downloaded inside a job (the login
nodes cannot reach Hugging Face) into `$W/hf`, which the server and the tokenizer then read with `HF_HUB_OFFLINE=1`.

**The command** (`train/bridges/vllm/serve.py`, `ServeSpec`): `vllm serve Qwen/Qwen3.6-35B-A3B --served-model-name
Qwen/Qwen3.6-35B-A3B --max-model-len 262144 --tensor-parallel-size <tp> --seed 0 --generation-config vllm
--enable-prefix-caching --enable-prompt-tokens-details --enable-lora --max-lora-rank 64 --max-loras 2`, on 127.0.0.1,
under `apptainer exec --nv --cleanenv --no-home` with the HF cache bound. `ServeSpec.record()` (engine, version, SIF and
its sha256, tp, context, seed, LoRA settings) goes into every arm's `settings.engine`, so `check_protocol` refuses a
comparison whose arms ran on different engine builds.

**Equal arms.** The trained arm is the base plus a PEFT LoRA adapter loaded at runtime (`POST /v1/load_lora_adapter`,
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) into the *same* server: one engine build, one tokenizer, one renderer, the same
request sampling (the request's temperature; `--generation-config vllm` keeps the repo's generation config out). The
adapter directory must be PEFT format (`adapter_config.json`, `adapter_model.safetensors`) with rank ≤ 64: the trainer's
export has to match.

**Thinking off.** The renderer (`TinkerChatCodec`, `enable_thinking=False`, recorded) builds the prompt token ids; the
server never applies a chat template.

**Counting.** `vllm` is a final backend (`factory.student.FINAL_BACKENDS`), like `tinker`; MLX on the Mac stays dev only.
Policy ids are `solver:vllm:<model>@vllm` (`…@vllm-adapter` for the trained arm). Note: the S15 targeting numbers were
measured on Tinker. The benchmark's honest pre-score is the fresh `baseline` arm on vLLM, never the Tinker selection.

## 3. The sandbox for model-written code

DeltaAI gives one uid and no subuid range, so code in a World runs as the invoking user on the host. The `docker` shim
(`scripts/deltaai/docker_shim.py`, standard library only, run under the site's Python 3.11) applies the same lockdown to
every container, whatever the caller asked:

| Measure | How |
|---|---|
| no network | `--net --network none`; a run asking for any other network is refused (exit 125) |
| no host files | `--containall --no-home`; binds only where the caller asked and only under `CRUCIBLE_SHIM_ALLOW` (the job's node-local scratch); a bind of the home, `/work`, `/projects`, `/u` or `/sw` is refused |
| no host environment | `--cleanenv` on start and on every exec; the caller's `-e` values travel in an env file |
| private processes, IPC, /dev | `--containall` (PID namespace: 7 processes visible, no host daemon) |
| a writable layer per container | `--overlay <node-local dir>` (fuse-overlayfs; the site's writable tmpfs is capped at 64 MiB) plus a node-local `/tmp` |
| root inside, the user outside | `--fakeroot` (a root-mapped user namespace; `uid_map 0 98407 1`) |
| GPUs only on request | `--nv` only for `--gpus` / `--runtime=nvidia` |

**Probe results** (job 3455864, gh152, `world/lockdown.py` `PROBE` inside the container): TCP to 1.1.1.1:443 and
8.8.8.8:53 blocked, DNS blocked, interfaces `lo` only; the home, `~/.ssh`, `~/.hf_token`, `/u`, `/projects`, `/work`,
`/sw` and the munge socket (present on the host) absent; 0 sensitive environment variables; 0 Slingshot or InfiniBand
devices; 7 processes, no host daemon; no Lustre/NFS mount; a 1 GiB write into `/opt` and 128 MiB into `/tmp` succeed.
Refusals measured: `--network host`, a bind of the home and a bind of the work tree all exit 125. `lockdown.verdict` is
empty for this report, and the episode jobs call `lockdown.require` on the task's own World image before any model
episode (no lockdown, no episodes).

**What stays unprotected** (written down, not hidden):
- cgroup limits per container are not available (rootless cgroups are refused in fakeroot mode); the Slurm job's cgroup
  bounds memory and CPUs for all Worlds of a job together, so one World can starve its siblings;
- with `--gpus`, `/dev` shows the node's four GPU device files; Slurm's device cgroup limits access to the job's own;
- the uid is the user's: a kernel or Apptainer escape would act as `mahmedmohsin` with access to the allocation's
  storage. There is no second line of defence such as a separate unprivileged uid;
- what a World can reach through its own binds (the task workspace, node-local) it can modify; nothing secret is bound;
- the vLLM server keeps the host network (it listens on 127.0.0.1 only); Worlds have no network, so they cannot reach it.

**A ticket for NCSA (draft, not sent):**

> Subject: Recommended sandbox for running untrusted, model-generated code on DeltaAI (allocation CIS262200)
>
> We run RL episodes in which a language model writes and runs code inside Apptainer containers on ghx4 nodes (account
> bjhg-dtai-gh). We use `--containall --cleanenv --no-home --net --network none --fakeroot --overlay` and bind only
> node-local scratch. Because our user has no subuid/subgid range, code inside runs as our own uid on the host. Could
> you (1) grant a subuid/subgid range to mahmedmohsin (so containers can map untrusted code to a separate unprivileged
> uid), or (2) recommend NCSA's preferred sandbox for untrusted code on DeltaAI (for example a site gVisor or Kata
> setup)? We would also like to know whether rootless cgroup delegation can be enabled for per-container memory and CPU
> limits. Thank you.

## 4. The benchmark harness

- **One job, one node:** `scripts/deltaai/smoke.sbatch` is the template: build the environment once (uv, Python 3.12,
  crucible editable), import the World SIFs, download the weights once, run the lockdown probe in the task's World,
  start the server, then run the arms.
- **The arms:** `tinker_rl.py eval --backend vllm --purpose baseline --seed <fresh>` for the pre-score;
  `tinker_rl.py compare --backend vllm --checkpoint <adapter dir>` for control and trained at identical settings and
  seed. `lh_debug_target.py student --backend vllm` for a debug-lane phase. They write the same `rollouts/`,
  `episodes.jsonl`, `metrics.json`, Weave-shaped traces and spend files as on Tinker, so the monitor, audit and keep
  rule read them unchanged.
- **Held-out tasks are offline:** `factory.tinker.require_offline` refuses a held-out task whose World declares a
  network (`WindowTask.world_network = "none"`), and the shim refuses any network at the container level.

## 5. Smoke and throughput

See §5.1 (filled from job logs).

## 6. Rules for jobs (from the earlier DeltaAI work)

- account `bjhg-dtai-gh` only: `unset SBATCH_ACCOUNT SLURM_ACCOUNT` (the login environment sets the other project's
  account, which overrides `#SBATCH` lines) and pass `--account` on the command line; check `squeue -o %a`;
- `SLURM_CONF=/var/spool/slurmd/conf-cache/slurm.conf` when a login node's Slurm DNS lookup fails;
- keep memory requests small: billing is the largest node share;
- the `test` partition is closed to this account; `ghx4-interactive` (2 h) starts short jobs within minutes, `ghx4`
  waits on fair-share;
- no compute on login nodes; the HF token stays in `~/.hf_token` and enters a job only through a mode-600 env file.
