# 14 The DeltaAI trainer: Crucible's RL loop on prime-rl

**Scope.** RL post-training of the student, Qwen/Qwen3.6-35B-A3B, on NCSA DeltaAI GH200 nodes instead of Tinker. The
user's decision of 2026-10-25: train on DeltaAI GPUs, keep Tinker for task creation. Serving, the arm64 Worlds, the
sandbox and the benchmark harness on DeltaAI are a separate worker's (this doc covers the trainer). Code:
`crucible/src/crucible/train/bridges/primerl/`, `scripts/delta_rl.py`, `tests/unit/test_primerl_bridge.py`,
`test_primerl_routing.py`, `test_primerl_score_centering.py`.

## 1. The framework choice

The rule is to build on a framework (DEPEND / PORT / PATTERN), never to reinvent. These were evaluated on the local clones
under `RL-Environments/06-frameworks-and-env-code` and on each project's current release:

| | slime (THUDM) | verl | SkyRL | NeMo-RL | **prime-rl** |
|---|---|---|---|---|---|
| the student's architecture (`qwen3_5_moe`: 30 gated delta-net + 10 full-attention layers, 256 experts, top 8) | Qwen3-MoE via Megatron; nothing for qwen3_5 | Megatron backend for Qwen3-MoE | via its backends | Megatron | **native model code** (`trainer/models/qwen3_5`, `AutoConfig.register("qwen3_5_moe")`) |
| LoRA on the routed experts | 6 files mention LoRA and experts | FSDP LoRA; Megatron LoRA partial | yes | yes | **grouped-expert LoRA** (`MultiLoRAGroupedExperts`), exported in vLLM's per-expert adapter format |
| 100K+ sequences | context parallel | Ulysses / CP | yes | CP | **ring / Ulysses CP**; its SWE example trains the 30B-A3B at 131K with `cp = 2` |
| multi-turn agentic rollouts, external environments | its agent harness | agent loop | yes | yes | verifiers v1 environments, sandboxed SWE example |
| weight sync to the inference engine | SGLang, full weights | NCCL / LoRA | LoRA-only sync | yes | **filesystem / NCCL / NIXL**, LoRA hot-load over `/load_lora_adapter` |
| aarch64 / GH200 | GB10 Dockerfile only | arm64 Dockerfiles | not found | arm64 in README | **aarch64 wheel markers** (flash-attn, torchao, router) and a multi-arch CUDA 13 image |
| Slurm | no launcher | no launcher | 11 files | 53 files | **native**: the `rl` entrypoint submits sbatch from a `[slurm]` table |
| custom loss / credit | hooks | estimator plugin | yes | yes | **import-path custom loss** with per-token advantages, trainer and sampler logprobs |

**Chosen: prime-rl v0.10.0** (PrimeIntellect-ai/prime-rl, Apache-2.0, released 2026-10-13; read at the local clone
f7e8ab6 and checked at the tag for every interface used). The runner-up is NeMo-RL (strongest Slurm support), but it would
need Megatron conversion and a LoRA path for an architecture it does not model natively. prime-rl is already a dependency
pattern here: verifiers v1 (M11) and two THIRD_PARTY rows.

## 2. The design: Crucible keeps the loop, prime-rl does the GPU work

prime-rl's own orchestrator runs verifiers harnesses. Crucible's training semantics live in its own runtime: the native
harness, the checkpoint sidecar (Φ), path gates, honesty, compaction, the infra/task split, and M6 credit. Re-hosting them
inside a verifiers harness would change what is trained. So the bridge keeps the Tinker loop
(`bridges/tinker/trainer.TinkerTrainer`) and replaces only its two remote calls, behind the same `Backend` protocol
(`backend.PrimeRLBackend`):

| call | Tinker | DeltaAI |
|---|---|---|
| sample | Tinker sampling client | vLLM `/v1/completions` on the prime-rl inference server: token ids in, `return_tokens_as_token_ids` + `logprobs=0` out, the adapter of that weight version as `model` |
| forward_backward + optim_step | Tinker training client | the batch as prime-rl `TrainingSample`s, packed by prime-rl's `prepare_batch` into `batches/step_<s>/rank_<i>.bin`; the prime-rl trainer trains and broadcasts the adapter to `broadcasts/step_<s>`; the bridge answers the handshake (`.sender_ready` -> `.receiver_ready`, waits `.finished`) and hot-loads it as `crucible-s<step>` |
| save_for_sampler | sampler weights | the broadcast adapter directory |
| save_state | Tinker state | prime-rl's own checkpoint of that step (`checkpoints/step_<s>`, `ckpt.interval = 1`) |

Everything above that is unchanged and shared, so a run means the same thing on both backends. A test proves it: the same
scripted groups run through Tinker's fake and through the bridge's fake trainer, and the bridge sends exactly
`wire.samples(the datums Tinker received)`, token for token.

- **What is ported.** The loop's filters (verifier version, infra groups dropped whole and refilled, staleness, zero
  spread), quarantine, the difficulty sampler with its reverse curriculum, the Dr.GRPO outcome advantage with honesty in
  the advantage (never the reward), M6 per-turn credit on completion tokens (`turn_eta`, centred, bounded, sign-preserving,
  none on gate-failed or hacked episodes), masked infra failures (never 0), checkpoint and resume, and W&B (`crucible-train`;
  the same `RolloutRecord`s and `steps.jsonl` the train monitor reads). New: `local_model` books tokens at 0 USD, since GPU
  hours are the cost.
- **The loss** (`loss.crucible_loss`, prime-rl `trainer.loss.type = "custom"`): Tinker's `importance_sampling` (−r·A, with
  the ratio capped and the gradient straight-through), `ppo` and `cispo`, plus a sampler–learner anchor
  β·Σ(log q − log p_θ) on the sampled tokens. That is the k1 estimate of KL(q‖p_θ) that Tinker's `stable_reinforce`
  adds; its gradient −β∇log p_θ keeps the learner near the policy that produced the data, without a reference model.
  `--kl-coeff` sets β (default 0). On top of that, the off-policy correction `--pg-correction` defaults to **score
  centering with masked importance sampling** (`sc+mis`, §7).
- **The loss scale stays exact.** prime-rl divides the rl loss by the step's trainable-token count; the bridge sends the
  advantages multiplied by that count (`wire.undo_token_mean`), so `loss_scale = sum` or `token_mean` trains exactly what
  it does on Tinker.
- **Off-policy.** Each weight version is served under its own adapter name, so groups still sampling from an older version
  keep it, and `--max-staleness` / `--async-groups` are the trainer's bound, as on Tinker.
- **MoE routing replay** (`primerl/routing.py`; on by default, `--no-router-replay` turns it off). The student routes
  every token to 8 of 256 experts on all 40 layers. A trainer that recomputes the router can pick another expert set for
  a token whose scores are close, so its logprob comes from a different sub-network than the one that sampled it. That
  is the MoE train/inference mismatch. Both halves exist upstream; the bridge joins them:
  - **Sampling** goes through `/inference/v1/generate` on prime-rl's vLLM server (`enable_return_routed_experts`). It
    returns the experts of every token the call forwarded, as `{data, shape [rows, layers, top_k], dtype, start}` (prime-rl
    `inference/vllm/serving_tokens.py:88-99`). The prefix cache is salted per episode (`cache_salt`), so every position
    an episode feeds was forwarded by one of its own calls.
  - **Out of band, since Crucible keeps its own records.** The routes sit in a `RoutingStore` keyed by the sha of the
    call's prompt + completion ids. At train time `stitch` rebuilds each sequence's `[T, layers, top_k]`: the first call
    that forwarded a position wins, and the never-forwarded last token gets a zero row.
  - **The trainer** gathers them in place of its own top-k (`enable_router_replay`; prime-rl
    `trainer/models/layers/moe.py:226-228`), sharded with the sequence under context parallel.
  - **All or none per step**, and incomplete coverage is refused, never zero-filled, as the torch trainer's R3 replay
    does (M12 #36). prime-rl packs one routing stream per micro batch.
  - **Metrics:** `routing/coverage_mean`, `routing/coverage_min`, `routing/replayed_share`. With replay off, the
    mismatch the loss logs (`kl/mismatch` = E[r − 1 − log r] per sampled token) is the signal routing drift feeds.
    `topk_overlap` compares two routings of the same tokens when both are known. prime-rl exports no trainer-side
    routes, so it is a library function, not a logged metric.
  - **Cost:** the routes are uint8, 40 × 8 bytes per token, about 51 MB for a 160K-token episode. They are held in the
    loop's memory until the step trains, then dropped.
  - **Verification so far:** unit tests with a faked server. The smoke model is dense (Qwen3-0.6B), so replay is off
    there. The 2-step run of the real 35B is the first live check: replayed share 1.0, coverage 1.0, and `kl/mismatch`
    compared with a `--no-router-replay` step.
- **Step numbering** (prime-rl v0.10.0, `trainer/ckpt.Progress.step = 1`). A fresh trainer's first step is 1
  (`backend.FIRST_STEP`). Before it, the trainer broadcasts its startup policy to `broadcasts/step_0`, and `trainer()`
  acknowledges it. Step s reads `batches/step_<s>` and writes `broadcasts/step_<s>` and `checkpoints/step_<s>`, so
  Crucible's version v (after v optimizer steps) is prime-rl's step v. A resumed run relaunches the trainer with
  `resume.step = s`, acknowledges the startup broadcast `step_<s>`, and continues at batch step s + 1. A checkpoint counts
  once torch DCP has written its `trainer/.metadata`.
- **The base model's served name** is `<model>-base`: with LoRA on, prime-rl serves it so, because vLLM rejects an adapter
  named like a served base model (`InferenceConfig.to_namespace`).

## 3. Memory and parallelism on GH200

Measured on the stored 300-call student episodes (40 runs, `data/targeting/debug-300`): median 294 calls, 73K sampled
tokens, a **median training sequence of 160K tokens (p90 195K, max 217K)**. So `seq_len = 262144`.

- **Trainer** (`train_gpus = 2` of a 4-GPU node): LoRA rank 32 on attention, MLP and the grouped experts. The frozen base
  is FSDP2-sharded (70 GB bf16, 35 GB per GPU) and the optimizer offloaded (`offload = "optimizer"`). Experts use `ep = 2`
  (128 per GPU) and context parallel `cp = 2` (ring), so 131K tokens per GPU. Activations are checkpointed every layer and
  offloaded to Grace memory (`ac_offloading`), and the router is frozen (prime-rl's RL default). The 30 delta-net layers are
  linear in length; the 10 full-attention layers are the quadratic term.
- **Inference** (`infer_gpus = 2`, TP 2): the bf16 KV cache is 20,480 bytes per token (10 layers × 2 KV heads × 256 × K,V
  × 2 bytes), so a 160K episode holds 3.3 GB. After the weights there is room for **about 31 concurrent episodes**, and
  prefix caching makes each call's prefill incremental.
- **Why one node.** Two-node jobs wait far longer in this account's queue (fair-share 0.08 against a 1,000-share account),
  and charging is by node share anyway.

**GPU-hours** (`delta_rl.py plan`, an estimate from those sizes, 30% MFU of GH200 bf16 peak, and ~28 min per 300-call
episode wave, Worlds included):

| per step | episodes | waves | wall | GPU-hours |
|---|---|---|---|---|
| 4 tasks × 8 rollouts | 32 | 2 | ~1.0 h | **~4.1** |
| 8 tasks × 8 rollouts | 64 | 3 | ~1.5 h | ~6.0 |

Sampling dominates: training is ~0.2 h per 32-episode step. With 976 GPU-hours left on 2026-10-25 and 60 reserved for the
before/after benchmark, **the recommended first run is 40 steps of 4 × 8, about 165 GPU-hours**. Extend in 20-step blocks
while the curves move: the budget allows about 200 steps of that size. A 2-step run of the real student comes first, about
8 GPU-hours.

## 4. Running it

```
uv run python scripts/delta_rl.py plan --steps 40 --groups-per-step 4 --group 8 --balance 976
uv run python scripts/delta_rl.py render --rows R.jsonl --targeting T.json --run-name rl-001 --stage stage/rl-001 \
    --steps 40 --groups-per-step 4 --group 8 --sampler difficulty --eta 0.1   # SC+MIS, k = 32 (§7); --pg-correction none --loss cispo for CISPO
rsync -a crucible/src crucible/scripts crucible/pyproject.toml deltaai:/work/hdd/bjhg/mahmedmohsin/crucible-train/crucible/
rsync -a stage/rl-001/ deltaai:/work/hdd/bjhg/mahmedmohsin/crucible-train/runs/rl-001/
ssh deltaai 'bash /work/hdd/bjhg/mahmedmohsin/crucible-train/runs/rl-001/build_env.sh'   # once, on the login node: BUILD_OK
ssh deltaai 'cd /work/hdd/bjhg/mahmedmohsin/crucible-train/runs/rl-001 && env -u SBATCH_ACCOUNT -u SLURM_ACCOUNT sbatch --account=bjhg-dtai-gh job.sbatch'
```

- `build_env.sh` builds prime-rl's environment once on the login node (§5.1). The job only checks its marker, puts
  Crucible's source on `PYTHONPATH` (the trainer imports the loss from it), starts vLLM and the trainer, and runs the
  loop.
- Account rules (DeltaAI memory): `bjhg-dtai-gh` only. `DeltaConfig` refuses any other, and the submit unsets
  `SBATCH_ACCOUNT`, which the login environment sets to another project. Memory is requested per GPU, because charging
  follows the largest node share.
- The `test` partition is closed to this account (`AllowAccounts=bbka-dtai-gh`). Smokes use `ghx4-interactive` (2 h max,
  partition priority factor 10,000 against ghx4's 100).
- Same flags as `tinker_rl.py train` (a test pins it), plus the deployment's, except two defaults: no USD cap, and
  `--max-tokens 8192` per model call (4096 cut the replies of ~90 stored episodes and ended GPU migration episodes). A
  reply cut there is a limit stop (`max_output_tokens`, a budget stop for the audit and the difficulty tracker, never a
  skill failure; a test pins it on the bridge's path). Held-out tasks and tasks without a recorded
  band are refused.

## 5. The smoke

`delta_rl.py render-smoke` runs on `Qwen/Qwen3-0.6B` with 1 trainer GPU and 1 vLLM GPU. A synthetic word-reversal task
(graded reward, so groups have spread) runs through the real bridge for two steps. The trainer is then killed and
relaunched from its step-1 checkpoint, and one more step runs. Checked: the trainer's step indices, a greedy probe that
changes under the new adapter (LoRA sync), and the checkpoint. Results: §5.1.

### 5.1 Results

- **Attempt 1** (job 3455884, `ghx4-interactive`, 2026-10-25, 31 min, about 2 GPU-hours: the interactive partition billed
  the whole node, `billing=4000`): it failed in the install. prime-rl's submodules (`deps/verifiers`, `deps/renderers`)
  are declared with SSH URLs, and the cluster has no GitHub key, so `renderers` was empty and `uv sync` refused it. The job
  then waited 30 min for a vLLM that never started. Fixed:
  - the submodules are fetched over HTTPS (`git -c url."https://github.com/".insteadOf=git@github.com:`);
  - the install is limited to `--extra gpu --extra flash-attn` (the prebuilt aarch64 FA2 wheel, nothing compiled);
  - an import check gates the install marker, and the job stops at once if vLLM or the trainer exits.
- **Attempt 2** (job 3457642, `ghx4`): cancelled before it started, to move to `ghx4-interactive` (its estimated start
  was 2026-10-27).
- **Attempts 3 and 4** (jobs 3460686, 3462156; `ghx4-interactive`, 2 GPUs, 120 GB): both failed in the install, in 14 s
  and 5 min. Attempt 1 had left a marker without its submodules, and the shallow submodules gave verifiers a version too
  low for prime-rl-configs, so uv's resolution failed.
- **The environment is now built on the login node, once** (`build_env.sh`, written by `render` / `render-smoke`). It only
  downloads, unpacks and imports, under `nice`, and no compiler runs. Steps:
  - the submodules over HTTPS with their history;
  - `uv sync --frozen --extra gpu --extra flash-attn` from prime-rl's lockfile;
  - `mooncake-transfer-engine` not installed (its aarch64 wheel needs glibc 2.39; DeltaAI has 2.38; it is for KV-cache
    offload, unused here);
  - the mamba and causal-conv1d CUDA builds skipped.

  `BUILD_OK` on 2026-10-26 in 13 min: torch 2.13.0+cu130, vLLM 0.31.0, verifiers (lock-pinned), and Crucible's bridge
  importable. The GPU job only checks the build marker; it never installs.
- **Attempt 5** (job 3467688, `ghx4-interactive`, 2 GPUs, 120 GB, 2026-10-27; job 3464731 on `ghx4` was cancelled
  before it started): it failed after 20:52 (`AllocTRES gres/gpu=2`, `billing=4000`), 0.70 GPU-hours at 2 GPUs. The
  versions came up (aarch64, GH200 120GB, driver 595.71.05, torch 2.13.0+cu130 with CUDA 13.0, vLLM 0.31.0). No RL step
  ran. Three blockers:
  - **No Python headers.** The venv was built on the system python (`/usr/lib64/python3.12`), which has no `Python.h`.
    Triton compiles a small C launcher (`cuda_utils.c`) on the GPU node, so every Triton kernel failed, and
    flash-linear-attention reported "Triton is not supported on current platform, roll back to CPU". The kernel check
    hung on the CPU path and hit its 600 s limit with no JSON. The student's Gated DeltaNet kernels and vLLM's own Triton
    kernels would all have failed the same way. Fixed: the venv is rebuilt on a uv-managed CPython 3.12
    (`uv python install 3.12`, `UV_PYTHON_PREFERENCE=only-managed`, installed under the work directory), whose headers
    ship with it. `build_env.sh` fails unless `sysconfig.get_paths()['include']/Python.h` exists, before and after
    `uv sync`. A venv on another interpreter is removed and rebuilt from the uv cache. The job checks it again at start.
  - **No `vllm-router`.** prime-rl's `inference` entrypoint starts a router in front of the engine by default
    (`InferenceConfig.router = VllmRouterConfig()`). The router binary sits in the `disagg` extra, with deep-ep, deep-gemm
    and nixl, and the build does not install it, so vLLM exited with `FileNotFoundError: 'vllm-router'`. Fixed without new
    packages: the inference config sets `router = "None"` (pydantic-config's null for TOML). prime-rl documents this mode
    and uses it for its own per-rank engines: the engine then serves `server.port` itself. The router only adds
    session-affine routing across data-parallel engines, and the run has one engine (TP 2), so `inference_toml` refuses
    more than one.
  - **`attn = "sdpa"` is not a v0.10.0 option** (`flash_attention_2|3|4|auto`), so the trainer's config failed
    validation. `auto` picks FA3 on Hopper (SM90), and the build installs only the prebuilt FA2 wheel. Fixed: both the
    smoke and the real run use `flash_attention_2` (`deploy.ATTN`). The real run's config had asked for FA3, which would
    have failed in the same way.

  The smoke job now also starts vLLM before the kernel check, so the check overlaps vLLM's load.
- **Rebuild** (2026-10-27, login node, about 22 min, mostly Lustre I/O on the first imports): `BUILD_OK` on CPython
  3.12.15 (uv-managed, `Python.h` present), torch 2.13.0+cu130, vLLM 0.31.0, Triton 3.7.1, and Crucible's bridge
  importable.
- **Attempt 6** (job 3468656, `ghx4-interactive`, 2 GPUs, 120 GB, account `bjhg-dtai-gh` checked with `squeue -o %a`):
  it failed after 21:05 (`ExitCode 6`, the job's "trainer exited"), 0.70 GPU-hours. The headers fix held: CPython 3.12.15
  with `Python.h`, Triton 3.7.1. Two findings:
  - **The trainer imports `prime_kernels`** (`trainer/distributed/collectives.py`, unconditionally), and that package sits
    in prime-rl's `kernels` extra. Fixed: the build adds `--extra kernels` (a prebuilt aarch64 cu130 wheel, nothing
    compiled). The build now imports every module the job imports (`JOB_IMPORTS`: the trainer's `train`, the inference
    entrypoint and vLLM server, the fla ops, the smoke), so a missing package fails the build on the login node, not a GPU
    job.
  - **Imports from /work/hdd on a cold GPU node are very slow.** The versions line (`import torch, vllm, triton`) took
    8 min. vLLM and the trainer were still importing 13 min after they started, and the kernel check hit its 600 s limit
    before it printed anything (rc 124). The venv held almost no bytecode (97 `.pyc` in vLLM), and its files are
    hard links into uv's cache on the HDD pool. Fixed:
    - the prime-rl checkout, its venv, uv's cache and the managed CPython move to `/work/nvme/bjhg/mahmedmohsin/crucible-train`
      (`DeltaConfig.envdir`). Runs, Crucible's source and the HF cache stay on `/work/hdd`;
    - the build's imports write the bytecode on the login node;
    - the job starts vLLM, the trainer and the kernel check together, waits up to 40 min for vLLM, and stops with the
      logs if phase 1 fails.
  - Also fixed: the real run's `job.sbatch` ran `cd {workdir}` before `uv run`, outside prime-rl's project, so `uv` would
    not have found prime-rl's venv. It now stays in `{envdir}/prime-rl`.
- **Attempt 7** (job 3471445, 2026-10-27): it failed after 2:17, 0.08 GPU-hours. The NVMe environment fixed the
  imports: vLLM and the trainer were through them in about 1.5 min (13+ min before), and the kernel check was compiling
  flash-linear-attention's Triton kernels on the GPU when the job ended. The trainer then stopped at
  `init_process_group`: `RANK expected, but not set`. prime-rl's `trainer` console script does not set up distributed;
  its `rl` launcher runs `torchrun --role=trainer ... -m prime_rl.trainer.rl.train @ config` (`utils/process.torchrun_cmd`)
  with its default trainer environment. Fixed: `deploy.trainer_command` builds that command, with a rendezvous port and
  id per launch, for the real job and both smoke launches. The job sets prime-rl's common environment (`COMMON_ENV`, a
  per-user Triton cache on node-local /tmp), and torchrun's workers are stopped before the trainer is relaunched.
- **Attempt 8** (job 3473175, 2026-10-27): it hit the 1:55 time limit, 3.83 GPU-hours. vLLM was healthy after 180 s.
  **The kernel check passed on the GH200**: `chunk_gated_delta_rule` (output `[1, 64, 2, 32]`) and fla's `causal_conv1d`
  ran forward and backward with gradients, and vLLM supports `Qwen3_5MoeForConditionalGeneration` (fla 0.5.2, Triton
  3.7.1, torch 2.13.0+cu130, CPython 3.12.15). No RL step ran. The trainer wrote `broadcasts/step_0/.sender_ready` and
  waited 3600 s for `.receiver_ready`. The bridge waited for `broadcasts/step_-1`, which prime-rl never writes: in v0.10.0,
  steps count from 1 (`Progress.step = 1`), not 0, so the startup broadcast is `step_0`. Then nothing failed until the
  time limit: prime-rl's batch sender waits for the trainer's request with no bound, and the smoke's own bounds were
  hours. Fixed:
  - `backend.FIRST_STEP = 1` (§2). The smoke resumes from checkpoint step 2, and a fresh run loads no startup adapter
    (the zero-initialised LoRA is the base model);
  - every wait is bounded: `FileIO.send_timeout_s` around prime-rl's sender, and the smoke's step and startup bounds are
    15 and 20 min (`SMOKE_TIMEOUTS`);
  - a checkpoint counts only after DCP has written `trainer/.metadata`, so stopping the trainer after phase 1 cannot
    leave a half-written checkpoint;
  - two problems that had not surfaced yet, found by reading prime-rl: the base model is sampled as `<model>-base`, the
    name prime-rl serves under LoRA, and the metrics reader skips undecodable chunks.
- **Attempt 9** (job 3474394, 2026-10-27, 1:00 limit): it failed in 20:21 (exit 9, phase 1), 0.68 GPU-hours. The bounded
  waits did their job. vLLM was healthy after 280 s, the startup handshake at `broadcasts/step_0` completed, 16 samples
  came back, and the batch went to `batches/step_1`. The trainer then failed to decode it: `msgspec.ValidationError:
  Expected int, got str - at $[0][0][0]`, the first token of `input_ids`. The smoke's prompt encoder had called
  `list(tok.apply_chat_template(..., tokenize=True))`, and in transformers 5.6.2 that call returns a `BatchEncoding`,
  so the "prompt" was its keys, `['input_ids', 'attention_mask']`. vLLM read a list of strings as two text prompts and
  answered 200. Crucible's own chat codec already passes `return_dict=False` and was not affected. Fixed: the smoke renders
  the template as text and encodes it (`delta_rl.smoke_encode`). The bridge now refuses non-integer token ids when it
  samples and on the wire (`wire.to_sample`), so this class of bug fails where it starts.
- **Attempt 10** (job 3474513, 2026-10-28): it failed in 18:32 (phase 1), 0.62 GPU-hours. The batch now decoded, and the
  trainer reached Crucible's loss, which failed: `clamp(max=torch.tensor(...))` put the importance-ratio bound on the CPU
  beside CUDA logprobs. The loss tests ran on the CPU only, so they could not see it. Fixed: the bound is a Python float
  (`math.log`). A test runs all three objectives forward and backward on an accelerator (MPS on the Mac, CUDA where
  present), and it fails on the old code.
- **Attempt 11** (job 3474615, 2026-10-28): **phase 1 passed**, and phase 2 failed. 19:59 in total, 0.67 GPU-hours. vLLM was
  healthy after 190 s. Two real RL steps ran through the bridge:
  - trainer step 1: 16 datums, 445 trained tokens, reward mean 0.17, 24 s; trainer step 2: 16 datums, 46 tokens, reward
    mean 0.29, 5.4 s;
  - each adapter was broadcast, hot-loaded (`crucible-s000001`, `crucible-s000002`) and served. The greedy probe's
    logprob went from -8.68 (base) to -0.97 to -0.03, so the LoRA sync works;
  - checkpoints were written at steps 1 and 2. The relaunched trainer loaded step 2 ("Resuming from step 2",
    `total_tokens=901`).

  Phase 2 then raced on stale markers. The relaunched trainer re-offers its resume step in `broadcasts/step_2` after
  deleting the directory. The bridge got there about 30 s first, found phase 1's answered markers (`.sender_ready`,
  `.receiver_ready`, `.finished`) and moved on, and the trainer waited for a receiver that had gone. Fixed:
  `FileIO.wait_broadcast` answers only an unanswered offer (`.sender_ready` without `.receiver_ready`). A test replays
  the race and fails on the old code. The kernel check passed again. Peak GPU memory: 5,050 MiB on the trainer GPU and
  88,671 MiB on the vLLM GPU (`gpu_memory_utilization = 0.9`).
- **Attempt 12** (job 3474848, 2026-10-28, 0:40 limit): **the smoke passed** (COMPLETED, rc 0) in 6:25, 0.21 GPU-hours.
  vLLM was healthy after 231 s.

  | phase | trainer step | wall (trainer log) | loss | grad norm | entropy | mismatch KL | reward mean | probe logprob |
  |---|---|---|---|---|---|---|---|---|
  | 1 | 1 | 3 min 55 s (first step: compile, cold start) | 1.532 | 107.1 | 0.64 | 0.0019 | 0.168 | -8.68 -> -1.03 |
  | 1 | 2 | 11.9 s | 0.264 | 72.4 | 1.32 | 0.0085 | 0.298 | -1.03 -> -0.020 |
  | 2 (resumed from checkpoint 2) | 3 | 14.5 s | 0.000 | 0.0 | 0.07 | 0.0004 | 0.185 | -0.020 -> -0.013 |

  - **The plumbing**: every step went batch -> prime-rl trainer -> adapter broadcast -> vLLM hot-load
    (`crucible-s00000{1,2,3}`), and the next samples were served under the new adapter. Checkpoints were written at
    steps 1 to 3. The relaunched trainer logged "Resuming from step 2" and trained step 3. The bridge acknowledged its
    fresh re-offer, not the stale markers.
  - **Step 3 had no gradient**: by then the policy was nearly deterministic (entropy 0.07), so every group's samples
    scored the same and every advantage was 0. Its probe still moved. With weight decay 0 and a zero gradient, only Adam's
    moments restored from checkpoint 2 can move the weights, and a fresh optimizer would not. So this is indirect evidence
    that the optimizer state resumed. It does not show a nonzero gradient after a resume.
  - **Kernel check** (the student's kernels, on the GH200): fla 0.5.2 `chunk_gated_delta_rule` and `causal_conv1d`
    forward and backward with gradients, and vLLM 0.31.0 supports `Qwen3_5MoeForConditionalGeneration`.
  - **Peak GPU memory** (`gpu_mem.csv`, 10 s samples): trainer GPU 5,052 MiB (prime-rl reports a 3.3 GiB peak), vLLM GPU
    88,673 MiB (its reservation at `gpu_memory_utilization = 0.9`).
  - **GPU-hours**: 0.21 for the passing job, or 0.071 per step with startup included. A warm step is about 12 to 15 s on
    2 GPUs, about 0.008 GPU-hours. All smoke attempts on bjhg-dtai-gh: 8.7 GPU-hours by AllocTRES (gres/gpu=2) ×
    Elapsed, or about 9.5 if attempt 1 is counted as the whole node it was billed as.
- **Driver**: the compute nodes run NVIDIA **595.71.05** (`nvidia-smi` in jobs 3455884 and 3462156, GH200 120GB). That
  is above 580, the minimum for CUDA 13.0, so the cu130 torch build runs as is.
- **The student's linear attention**: its config.json is `Qwen3_5MoeForConditionalGeneration`, 30 Gated DeltaNet layers
  plus 10 full-attention layers. prime-rl's Gated DeltaNet layer (`trainer/models/qwen3_5/gated_delta_net.py`) takes
  its kernels from flash-linear-attention (`fla.modules.conv.causal_conv1d`, `fla.ops.gated_delta_rule`; Triton,
  installed from the lockfile). It does not use the `causal-conv1d` or `mamba-ssm` packages, whose CUDA builds are
  skipped: those serve only `nemotron_h`.

## 6. Open items

- **Trainer metrics arrive late at the bridge.** prime-rl logs a step's metrics after its checkpoint, and the bridge reads
  them right after the broadcast, so in the passing smoke steps 1 and 2 carried no trainer loss or grad norm (step 3 did).
  The Tinker loop's `steps.jsonl` and W&B would miss them the same way. A fix reads them after `save_state`, or attaches
  step s's to step s + 1.
- **A gradient after a resume is not yet shown live**: step 3 of the smoke had zero-spread groups (§5.1). The 2-step real
  run, or a smoke with higher temperature, shows it.

- **MoE routing replay is built but not yet run live** (§2): it is first checked on the 2-step real-model run.
- **Score centering (SC+MIS) is built but not yet run live** (§7.6): the 2-step run checks the head on the generate
  endpoint, full coverage, alignment and the server and trainer cost.
- **Model-written code is graded as the user on DeltaAI** (Apptainer user mode, `trusted: false`): acceptable for our own
  patches, not for RL episodes at scale, until NCSA grants a subuid range or a different sandbox is used. This blocks the
  real run's grading, not the trainer.
- **The real 35B on 2 trainer GPUs is untested** (memory headroom at 262K with `cp = 2`). The 2-step real-model run decides
  between `cp = 2` and a whole node for the trainer.

## 7. Score centering with masked importance sampling (SC+MIS)

The student samples in bf16 vLLM with its own MoE routing (replayed, §2) and trains in prime-rl's FSDP model, and the
loop is asynchronous (`--max-staleness`, `--async-groups`). So the sampler q never equals the learner p_θ, even on the
step it was saved. "Score Centering Stabilizes Off-policy Reinforcement Learning" (Marek and Ryabinin, arXiv 2609.20807,
2026; reference code `martin-marek/score-centering`, MIT, read at 6484296) splits the expected policy gradient at a
prefix into E_q[R·s_y] = E_q[R]·s̄ + Cov_q(R, s_y), with s̄ = Σ_v q_v ∇log p_v. The first term is **drift**: it carries no
information about which rollouts succeeded, it vanishes only on-policy, and it accumulates over steps. Score centering
subtracts the score's expectation under the effective sampling mass, which removes the drift and keeps the signal:

  per-token loss = −A·(sg(w_y)·log p_y − Σ_{v∈H} sg(q_v·w(p_v/q_v) − α·p_v)·log p_v)

H is the sampler's top-k head at that position. The tail is modeled as ρ·p_v with ρ = (1 − Σ_H q)/(1 − Σ_H p), so its
centering term is −α·Σ_H p_v∇log p_v with α = ρ·w(1/ρ), and only the k head logprobs carry gradient. The token weight w
is 1 (SC), TIS min(r, 2) or MIS r·𝟙[0.5 ≤ r ≤ 5]. As the reference code does: float32, no renormalization of the head,
and the sampled token's own logprob kept when it is outside the head. With exact IS weights (w = r) the centering term
vanishes; on-policy, SC is plain PG.

### 7.1 The sampler returns the head (`topk.py`, `topk_server.py`)

- **What vLLM 0.31.0 returns.** With `sampling_params.logprobs = k`, vLLM's sampler gathers k + 1 entries per generated
  token: the sampled token first, then the top k (`Sampler.gather_logprobs`; the sampled token is repeated when it is in
  the head). The values come from the temperature-processed distribution, since prime-rl sets `logprobs_mode =
  "processed_logprobs"`, so they match the trainer's temperature-scaled logprobs. The API server keeps them as
  `FlatLogprobs`: flat Python lists of ids, logprobs, ranks and decoded tokens.
  - Upstream's `/inference/v1/generate` answers with `logprobs.content`, one JSON object per token, holding a
    `top_logprobs` list of `{"token": "token_id:<n>", "logprob": x}` keyed by token id. A sampled token outside the head
    pushes the k-th head entry out of that list (`_create_tokens_logprobs` keeps the first k of the k + 1 dict items).
  - **prime-rl v0.10.0's handler** (`inference/vllm/serving_tokens.py`, `_PackedOutputs`) replaces that with
    `completion_logprobs`: only the sampled entry, as a base64 float32 array. It drops the rest. So on the endpoint the
    bridge uses, **the head is not returned at all**, whatever `logprobs` asks for.
  - vLLM's `max_logprobs` defaults to 20, below k = 32.
  - vLLM PR 62446 (open, not merged) returns top-k as flat base64 `.npy` arrays on this endpoint and reports 4–5× less
    server CPU at k = 8. The bridge does not depend on it.
- **The patch.** `topk_server.install` makes `_PackedOutputs` also pack entries 1..k of each position as `topk_ids`
  (int32) and `topk_logprobs` (float32) `[n, k]`, in prime-rl's own `{data, shape, dtype}` base64 form. The response
  models gain the two fields. It loads as a vLLM `--middleware` import path: vLLM imports it in every API-server process
  while it builds the app. `inference_toml` sets `vllm.middleware` and `vllm.max_logprobs = k` (prime-rl passes extra
  `vllm` keys through to vLLM's namespace). Nothing in prime-rl or vLLM is edited on disk.
- **The bridge** (`backend._Sampler`) samples over the generate endpoint whenever routing replay or score centering is
  on, with `logprobs = k`. It keeps each call's heads in a `TopKStore` keyed by the sha of prompt + completion, as
  `RoutingStore` keeps routes. At train time `topk.stitch` gives every trained position the head of the call that
  sampled it.
- **Expected overhead.**
  - **GPU:** one `torch.topk` over the 248K vocabulary per sampled token. The log-softmax is already computed for
    `logprobs = 0`. The device-to-host copy grows from 1 to 33 entries per token.
  - **API-server CPU and memory:** the output processor converts and appends k + 1 entries per token instead of 1, and
    holds them as Python objects until the call ends. That is about 2.8 KB per token at k = 32, ~23 MB for an
    8192-token reply, and ~0.7 GB across 30 concurrent calls. The packing is one numpy pass per call, and no per-token
    JSON objects are built. The response carries ~350 B per token of base64 instead of ~1.6 KB of `top_logprobs` objects.
  - This is where vLLM's own top-k path costs CPU (the PR's 4–5× at k = 8 is mostly the JSON objects, which this path
    skips). It must be measured: `delta_rl.py topk-bench` times n concurrent calls at k = 0, 8 and 32 (wall time,
    tokens/s, bytes per token). The smoke job runs it on the 0.6B model after its RL steps.

### 7.2 The trainer gets the learner's head logprobs (`sc_head.py`)

- **prime-rl's custom-loss interface does not have them.** `LossInputs` (`trainer/rl/loss.py:21-37`) has the sampled
  token's learner and sampler logprobs, the advantages, the mask and the weights. Score centering needs log p_v for the k
  head ids, and Σ_H p_v, at every trained position.
- **The narrowest hook is the fused LM head.** `FusedOutputLinear` (`trainer/models/layers/lm_head.py`) computes the
  logprobs in `_SequenceChunkedLogProbEntropyFn`, over token chunks × vocab chunks of 8192 with an online logsumexp, so
  [T, V] (262K × 248K) is never materialized. That loop already gathers logits at per-position ids for prime-rl's
  top-p/top-k sampling-mask replay. `HeadLogProbFn` is the same loop without the replay renormalization, plus k gathers
  per vocab chunk. It returns the full-vocabulary logprob and entropy, unchanged, and `head_lp` [N, k] with its
  gradient. The backward pass recomputes per chunk, as prime-rl's does: g_tok·e_y + Σ_v g_v·e_v − (g_tok + Σ_v g_v)·softmax.
- **The data rides prime-rl's sampling-mask stream.** `TrainingSample.sampling_mask` is the one per-token int stream that
  prime-rl already packs, shifts to the labels, shards under context parallel and passes to the LM head. A head row is
  `[SC_MAGIC, ids_1..k, bits(logprob_1)..bits(logprob_k)]` (float32 bit-cast to int32), only at trained positions; other
  positions have count 0. The marker tells the hook that the row is a head, never a sampling mask (Crucible never samples
  with top-p or top-k).
- **What is patched**, by `install`, when prime-rl's trainer imports the loss (`setup_rl_loss_fn`):
  1. `FusedOutputLinear.forward` runs `HeadLogProbFn` on a payload and stashes the micro batch's head. Without a payload
     it runs the stock path.
  2. `VanillaOutputLinear.forward` refuses a payload, because prime-rl's train loop would renormalize the sampled
     logprobs over it. The trainer config pins `model.fused_lm_head_token_chunk_size = 8192` (prime-rl's default).
  3. The train module's `gather_for_cp` records the context-parallel group. Its `compute_loss` gathers the head over
     that group with prime-rl's own differentiable all-gather (as for the logprobs, so the gradient scale is the same),
     shifts it to the sampled-token positions, splits it per sequence, and keys each piece by the `trainer_logprobs`
     tensor that `compute_loss` hands that sequence's `LossInputs`. `crucible_loss` looks it up.

  The patch fails at startup when these names are not where v0.10.0 has them, never silently.
- **Cost:** `head_lp` is 4·k bytes per token, 33.5 MB at 262K tokens and k = 32, plus its gradient. The extra compute is
  k gathers and one scatter per token per vocab chunk, small against the head's three matmuls.

### 7.3 The default configuration, and how it composes

- **Default: `--pg-correction sc+mis`, k = 32, `--loss importance_sampling`, no PPO clip, `--kl-coeff 0`.**
  - **SC:** the paper's evidence is that SC alone matches or beats every IS method under quantization mismatch, with the
    gap growing as the mismatch grows. That is the bf16-sampler and MoE-routing case here.
  - **+MIS:** under staleness, SC composed with an IS correction performs best, ahead of vanilla SC. This loop is async
    with a staleness bound. MIS (r kept in [0.5, 5], else the token weight is 0) drops the rare token whose ratio is
    wild, such as a routing flip that replay missed, instead of truncating it. Its tail factor α is 1 or 0. TRL's async
    GRPO trainer picks the same default (below).
  - **k = 32:** in the paper's top-k sweep (App. A.4), k = 32 and k = 128 match full centering in every setting
    tested. 128 is the paper's "conservative default"; 32 costs a quarter of the server work.
- **No PPO / CISPO clip with SC.** The paper composes SC with TIS or MIS only. It never adds a PPO clip, whose bounds are
  meant for policy movement, not mismatch. TRL replaces its clipped objective when score centering is on. So
  `pg_correction != none` requires `--loss importance_sampling` (a test pins the refusal), and `ratio_low` /
  `ratio_high` are not used. `ppo` and `cispo` stay available with `--pg-correction none`, and `tis` / `mis` alone
  stay available too.
- **The sampler anchor is orthogonal.** In expectation, its gradient is β·∇KL(q‖p_θ), a pull toward q with a stated
  strength that does not depend on the reward (a test pins this). SC removes the reward-dependent pull E_q[R]·s̄; the
  anchor is a separate trust region. The paper uses none, so β stays 0 by default. Raise it only if `sc/kl_head`
  grows.
- **The advantages are unchanged:** Dr.GRPO, honesty in the advantage, M6 turn credit, × the token count for
  `loss_scale`.
- **Values:** the SC / TIS / MIS surrogates take the IS surrogate's value −A·w, so the logged loss stays comparable to
  `importance_sampling`; only the gradients differ.

### 7.4 All or none per step, and diagnostics

- **A step centers every sequence or none** (`topk.all_or_none`), mirroring routing replay. If any sequence lacks a head
  at a trained position (a call whose heads were not returned, or a re-rendered prompt whose key does not match), the
  step sends no heads. The loss then falls back to its importance part for the whole step: `sc` and `sc+mis` to MIS,
  `sc+tis` to TIS. The bridge logs a warning, and the trainer refuses a sequence with partial heads.
- **Logged by the bridge:** `sc/coverage_mean`, `sc/coverage_min`, `sc/centered_share`, `sc/fallback` and
  `sc/fallback_steps_share`.
- **Logged by the trainer,** as `primerl/<name>/{mean,min,max,...}`:
  - `sc/head_mass` and `sc/head_mass_min` (Σ_H q);
  - `sc/rho`, `sc/rho_min` and `sc/rho_max`;
  - `sc/score_mean_l1`, the drift proxy: Σ_H |q_v − p_v| + |q_tail − p_tail|, the L1 norm of s̄ in logit space
    (∇_z log p_v = e_v − p, so s̄_z = q − p), a lower bound of 2·TV(q, p);
  - `sc/kl_head` (KL(q‖p) from the head) and `sc/tail_dropped` (the tails MIS removed);
  - `is/masked_frac` (MIS's dropped tokens);
  - `sc/align_err` (the largest gap between a sampled token's logprob and its own head entry: 0 when the heads are
    aligned);
  - `sc/fallback`.

### 7.5 Memory per 160K-token episode (median 73K sampled tokens; k = 32)

| where | what | size |
|---|---|---|
| the loop (Grace memory) | `TopKStore`: int32 id + float32 logprob per head entry, sampled tokens only | 73K × 32 × 8 B = **18.7 MB** (routes: 51 MB) |
| the batch file | `sampling_mask` rows of 1 + 2k int32 per trained token, plus a count per position | 73K × 260 B + 160K × 4 B = **19.6 MB** |
| the trainer, per micro batch (262K tokens, `cp = 2`) | the padded [T, 65] int32 stream, ~68 MB on host and on GPU; per rank: ids (int64) 33.5 MB, `head_lp` + gradient 33.5 MB, the gathered head ~100 MB | **~0.3 GB** of 96 GB |
| the vLLM API server | `FlatLogprobs` of an in-flight call | ~2.8 KB per generated token |

A 4 × 8 step holds about 0.6 GB of heads in the loop until it trains. Groups that never train (zero spread, infra
drops) are evicted with the oldest calls past `max_calls`, as routes are.

### 7.6 Checks

- **TRL.** Confirmed: score centering composed with MIS is the default in TRL's async GRPO trainer
  (`trl/experimental/async_grpo`, main at 9824938, 2026-10-27). `AsyncGRPOConfig.score_centering = True`;
  `vllm_importance_sampling_clip_min / _max` = 0.5 / 5 drop tokens outside the band (MIS); `num_sampler_logprobs = 8`.
  - The loss (`compute_loss`, "Appendix A.2") is −A·(w·log p − Σ_H sg(p_v·𝟙[0.5 q_v ≤ p_v ≤ 5 q_v] − 𝟙[0.5 ≤ 1/ρ ≤ 5]·p_v)·log p_v),
    with w = r masked. That is this formula with w = MIS: q·w(p/q) = p in the band, and α = ρ·w(1/ρ) is the band
    indicator on 1/ρ.
  - With SC on, the PPO clip is not applied.
  - It gathers the head logprobs in a fused LM head (`score_ids`), the same design as `sc_head`.
  - The differences: TRL's default k is 8 (here 32), and TRL's loss value includes the log-probs (here the IS surrogate's
    value; the gradients are equal).
- **Tests** (`tests/unit/test_primerl_score_centering.py`, CPU, small vocabularies):
  - with k = V, the gradient equals full score centering;
  - with a top-k head, the gradient equals the reference implementation's (its README PyTorch function, as the oracle);
  - exact IS weights make the centering term vanish;
  - on-policy SC is plain PG;
  - the expected gradient under a mismatched q is 0 with SC (k = V and a one-token tail) while plain PG drifts, and the
    anchor's expected gradient is β(p − q);
  - the reference repository's drift test, ported;
  - the sampled token outside the head, and that only the tail's mass matters;
  - the MIS band edges at 0.5 and 5;
  - the fallback and all-or-none rule;
  - the server packing, the store, the stitching and the payload round trip;
  - `HeadLogProbFn` against a full softmax (values and gradients, multi-chunk);
  - prime-rl's micro-batch path emulated end to end (payload → shift → fused head → split → custom loss), equal to the
    full-softmax loss;
  - `install` and `topk_server.install` against stand-ins with prime-rl's v0.10.0 names.
- **Not yet run live.** The 2-step 35B run must show:
  - the generate endpoint returning `topk_ids` (the middleware loaded; `max_logprobs = 32` accepted);
  - `sc/coverage_min` = 1 and `sc/centered_share` = 1 on both steps (no fallback);
  - `sc/align_err` ≈ 0 (the heads aligned with the sampled tokens after prime-rl's shift and CP shard);
  - `sc/head_mass` and `sc/rho` (a head that holds most of the sampler's mass, ρ near 1; `sc/head_mass_min` shows the
    flattest positions);
  - the trainer's peak memory and step time against a `--pg-correction mis` step;
  - `topk-bench` on the 35B server: tokens/s at k = 32 against k = 0.
