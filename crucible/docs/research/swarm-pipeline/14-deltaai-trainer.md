# 14 The DeltaAI trainer: Crucible's RL loop on prime-rl

**Scope.** RL post-training of the student, Qwen/Qwen3.6-35B-A3B, on NCSA DeltaAI GH200 nodes instead of Tinker. The
user's decision of 2026-10-25: train on DeltaAI GPUs, keep Tinker for task creation. Serving, the arm64 Worlds, the
sandbox and the benchmark harness on DeltaAI are a separate worker's (this doc covers the trainer). Code:
`crucible/src/crucible/train/bridges/primerl/`, `scripts/delta_rl.py`, `tests/unit/test_primerl_bridge.py`.

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
  `--kl-coeff` sets β (default 0).
- **The loss scale stays exact.** prime-rl divides the rl loss by the step's trainable-token count; the bridge sends the
  advantages multiplied by that count (`wire.undo_token_mean`), so `loss_scale = sum` or `token_mean` trains exactly what
  it does on Tinker.
- **Off-policy.** Each weight version is served under its own adapter name, so groups still sampling from an older version
  keep it, and `--max-staleness` / `--async-groups` are the trainer's bound, as on Tinker. MoE routing replay (prime-rl's
  `enable_router_replay` with vLLM's routed-expert capture, against train/inference routing mismatch) is not wired yet: it
  needs the completions path to return routed experts.
- **Step numbering.** Crucible's version v (after v optimizer steps) is prime-rl's policy `v{v-1}`: prime-rl's step index
  starts at 0, and its startup broadcast is policy `v-1`, acknowledged by `trainer()`. A resumed run relaunches the trainer
  with `resume.step = s` and continues at batch step s + 1.

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
    --steps 40 --groups-per-step 4 --group 8 --sampler difficulty --eta 0.1 --loss cispo --kl-coeff 0.01
rsync -a crucible/src crucible/scripts crucible/pyproject.toml deltaai:/work/hdd/bjhg/mahmedmohsin/crucible-train/crucible/
rsync -a stage/rl-001/ deltaai:/work/hdd/bjhg/mahmedmohsin/crucible-train/runs/rl-001/
ssh deltaai 'cd /work/hdd/bjhg/mahmedmohsin/crucible-train/runs/rl-001 && env -u SBATCH_ACCOUNT -u SLURM_ACCOUNT sbatch --account=bjhg-dtai-gh job.sbatch'
```

- The job installs prime-rl at the pin once (`uv sync --all-extras` in `crucible-train/prime-rl`), puts Crucible's source on
  `PYTHONPATH` (the trainer imports the loss from it), starts vLLM and the trainer, and runs the loop.
- Account rules (DeltaAI memory): `bjhg-dtai-gh` only. `DeltaConfig` refuses any other, and the submit unsets
  `SBATCH_ACCOUNT`, which the login environment sets to another project. Memory is requested per GPU, because charging
  follows the largest node share.
- The `test` partition is closed to this account (`AllowAccounts=bbka-dtai-gh`). Smokes use `ghx4-interactive` (2 h max,
  partition priority factor 10,000 against ghx4's 100).
- Same flags as `tinker_rl.py train` (a test pins it), plus the deployment's. Held-out tasks and tasks without a recorded
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
- **Attempt 2** (job 3457642, `ghx4`, 2 GPUs, 200 GB, at most 2 h 45 min): queued. `ghx4-interactive` allows one job per
  user, and another worker holds it.

## 6. Open items

- **MoE routing replay**: return routed experts through the sampling path and enable `enable_router_replay`.
- **Model-written code is graded as the user on DeltaAI** (Apptainer user mode, `trusted: false`): acceptable for our own
  patches, not for RL episodes at scale, until NCSA grants a subuid range or a different sandbox is used. This blocks the
  real run's grading, not the trainer.
- **The real 35B on 2 trainer GPUs is untested** (memory headroom at 262K with `cp = 2`). The 2-step real-model run decides
  between `cp = 2` and a whole node for the trainer.
