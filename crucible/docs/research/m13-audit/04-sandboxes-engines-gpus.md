# M13 audit 04: sandboxes, inference engines and GPU paths

**Question.** What sandbox backends, inference engines and GPU paths must an RL-environment framework support? What do the locally cloned frameworks offer? What does Crucible support today?

**Method.** I read the code; nothing below is inferred from READMEs alone.
- Upstream paths are relative to `RL-Environments/06-frameworks-and-env-code/`. Harbor environments are under `harbor-framework-harbor/src/harbor/environments/` (written `HB/`), and agentenv providers under `agentenv-framework/src/agent_env/providers/` (written `AE/`).
- Crucible paths are relative to `crucible/src/crucible/`.
- Machine facts come from read-only probes run on 2026-10-10.

## 0. Test beds (probed, not assumed)

| Test bed | What it has | What it lacks |
|---|---|---|
| **This Mac.** M4 Pro, macOS 15.5, arm64, 24 GB RAM | <ul><li>Docker Desktop 29.8: linuxkit kernel 7.0.12, aarch64, cgroup v2, 14 CPUs, 7.7 GiB VM.</li><li>torch 2.14 with MPS available.</li><li>`harbor` and `agent_env` importable in the uv env.</li></ul> | <ul><li>Docker runtimes are only `runc`.</li><li>No `/dev/kvm`, even inside a `--privileged` container.</li><li>No podman, firecracker, qemu, runsc, Apple `container`, vllm, sglang, mlx or llama.cpp.</li><li>No Modal, E2B or Daytona credentials (`~/.modal.toml` absent; none in `.env`).</li></ul> |
| **Lab RTX 5090** (`lab5090` in `crucible.toml`). Linux 6.17, sm_120, 32 GB, shared | <ul><li>Docker 29.8 with runtimes `runc` and `nvidia`.</li><li>`/dev/kvm` exists. The ACL grants only user `hpc`, but `ahmed` is in the `docker` group, so KVM is reachable through `docker run --device /dev/kvm`.</li><li>Image `vllm/vllm-openai:v0.31.0` (22.8 GB) is present.</li><li>222 GB free disk.</li></ul> | <ul><li>No runsc, kata-runtime, firecracker, qemu or podman on the host.</li><li>Registering a Docker runtime needs sudo plus a daemon restart on a shared machine.</li><li>`--gpus` fails; Crucible already uses `--runtime=nvidia`.</li><li>One GPU only, so NCCL between two ranks is impossible.</li></ul> |
| **Runpod RTX 4090** (from memory notes; not probed today) | CUDA 12.8, 96 cores, native processes | No Docker, no user namespaces, no CAP_SYS_ADMIN. Only host-process sandboxes are possible. |

Consequences for testing:
- Container work and CPU/MPS training run on the Mac.
- Everything CUDA, every KVM-based isolation (Firecracker or QEMU inside a container with `--device /dev/kvm`), and vLLM run on the lab machine.
- Multi-GPU work (TP>1, NCCL) needs a rented multi-GPU box.
- Cloud sandboxes need credentials the user does not have yet.

---

## 1. Sandboxes

### 1.1 What exists upstream, by isolation kind

| Kind | Upstream implementations (file) | Features actually in code | Crucible status |
|---|---|---|---|
| **Function call / host process** | <ul><li>skyrl-gym `tools/python.py:13` (`subprocess.run`, timeout 10)</li><li>`envs/lcb/livecodebench.py:503` (`reliability_guard`: "NOT a security sandbox")</li><li>ROLL `utils/local_code/execute_utils.py` (RLIMIT_AS, `os.system=None`)</li><li>verifiers `v1/runtimes/subprocess.py:83`</li><li>NeMo-Gym `local/provider.py:43` ("No isolation")</li><li>rLLM `backends/local.py`</li><li>OpenEnv `UVProvider`</li></ul> | <ul><li>Timeout and rlimits only.</li><li>Startup in milliseconds.</li></ul> | **Done.** `world/backends/local.py` (`LocalBackend`: isolation none; refuses GPU apps and policy shells unless opted in). `GpuHost.kind="ssh_exec"` is declared in `runtime/gpu.py` but has **no implementation**, so Runpod-style hosts without Docker cannot run GPU worlds. |
| **Container (docker/podman)** | <ul><li>Harbor `docker/docker.py:311-336`: compose; no-network overlay; **allowlist through an egress-control sidecar** (nftables, gost); cpu/mem limits; **no GPU**.</li><li>Harbor `podman.py:128`: Docker subclass.</li><li>NeMo-Gym `docker/provider.py:203` (`--gpus N`, `--network`, `--memory`).</li><li>verifiers `runtimes/docker` (`--gpus`; Podman via CDI; iptables egress plus proxy).</li><li>OpenEnv `LocalDockerProvider` (no limits).</li><li>agentenv `local_sandbox.py` (no network policy).</li><li>rLLM docker (no limits).</li></ul> | <ul><li>Startup in seconds; images cached.</li><li>No-network and allowlist as above.</li><li>cpu/mem/pids limits.</li><li>GPU only in NeMo-Gym and verifiers.</li><li>No snapshot, pause or fork in any of them.</li></ul> | **Done.** Three paths:<ul><li>`world/sandbox.py` `DockerSandbox` (`--init`, `--network none`, pids, memory, `--runtime=nvidia` or `--gpus`, `DOCKER_HOST` to lab).</li><li>`interop/harbor/gpu_docker.py` `GpuDockerEnvironment` (Harbor Docker plus a compose overlay: nvidia runtime or device reservations, `network_mode: none`, init, shm, pids, cap_add, mem, security_opt).</li><li>`interop/agentenv/plugin/gpu_docker.py` (agentenv local provider plus GPU flags).</li></ul>Hardening comes from `world/guard/` (demoted uid, quiesce, egress probe, git trust). **Gap:** the allowlist is not enforced on Docker (§1.3). Podman is reachable only through Harbor's generic branch and is untested. |
| **gVisor (runsc)** | <ul><li>No framework configures runsc.</li><li>gVisor appears only as Modal's default runtime: Harbor `modal.py:592` ("gVisor Modal sandboxes lack netlink…"); agentenv `modal_sandbox.py:330` (`enable_docker_in_gvisor`).</li></ul> | Syscall interception; network handled by the provider. | **Indirect only.** A Modal world through agentenv or Harbor runs under gVisor, but this is untested (no credentials). No local runsc backend. |
| **microVM (Firecracker / Kata / CH)** | <ul><li>Harbor `kata.py:113`: Docker plus a `runtime:` overlay. Needs Linux, `/dev/kvm` and a registered kata runtime. **Network policies rejected**; builds run under runc; **no GPU**.</li><li>Firecracker appears in no code at all. E2B (`HB/e2b.py`, `AE/e2b`) is Firecracker-based as a service, though the code never says so.</li><li>Cloud microVMs: Harbor `blaxel.py`, `tensorlake.py` (TCG fallback), `islo.py`.</li><li>Harbor `smol.py` SmolVM: **copy-on-write `fork_batch`** from a checkpointed golden machine (`:216`, `auto_checkpoint=True`, batches of 32), with one-GPU `cuda`/`vulkan` passthrough when `target=="local"`.</li></ul> | <ul><li>Kernel isolation.</li><li>Smol is the only upstream fork primitive.</li></ul> | **Missing.** `BackendCapabilities.isolation` allows `"vm"`, but no backend sets it. `environment_type="kata"` would go through Harbor's generic branch, but Kata refuses the network policy Crucible always passes for no-network worlds. |
| **Full VM (QEMU / EC2 / desktop VMs)** | <ul><li>Harbor `ec2.py:466` (remote compose over SSH), `apple_container.py:80` (arm64 Macs; mounted only; no GPU), `cua_cloud.py` / `use_computer.py` (desktop-VM warm pools).</li><li>Modal `vm_runtime` (Harbor `modal.py:746`: GPU rejected under VM; agentenv `modal_vm_sandbox.py`: `gpu=None`).</li><li>agentenv `sail_vm`.</li><li>NeMo-Gym OSWorld QEMU/KVM (`resources_servers/osworld/config.py:16`); its `save_state` raises "Live VM snapshots are not available".</li></ul> | <ul><li>Real kernel.</li><li>Nested Docker.</li><li>Slow startup.</li></ul> | **Missing.** None of these has been exercised. |
| **Remote clouds** | <ul><li>Harbor factory registers 31 types (`factory.py:31-181`). GPU-capable: Modal (`modal.py:1124` `"{type}:{n}"`), GKE (`gke.py:782` `nvidia.com/gpu` limits and toleration), Daytona (`daytona/environment.py:1170`), Beam, Prime, OpenSandbox.</li><li>Dynamic network policy (`_apply_network_policy`): docker, daytona, modal, e2b, prime, novita, runta, beam, islo, tensorlake, vercel, hyperbrowser.</li><li>NeMo-Gym: Daytona (`snapshot_id`), E2B (template), **OpenSandbox `pause()`/`resume()`** (`:1097/:1151`, a server-side checkpoint), OpenShell, ECS Fargate (GPU refused).</li><li>verifiers: Modal (`vm`, `outbound_cidr_allowlist`), E2B, Prime.</li><li>**prime_sandboxes SDK 0.4.3**: `checkpoint(sandbox_id)` and `CreateSandboxRequest.checkpoint_id`. This is real snapshot/restore/fork, but no caller in these trees uses it.</li><li>rLLM Modal `snapshot_filesystem()` (`modal_backend.py:342`), Daytona snapshots, `WarmQueue`.</li><li>OpenEnv: Daytona, Modal, ACA (auto-suspend Memory/Disk, `deny_all_egress`), Novita, HF; `KubernetesProvider` is a placeholder.</li><li>ROLL: ROCK SDK (external; timeouts only).</li><li>AReaL: Daytona only.</li></ul> | <ul><li>Warm pools: SkyPilot `pool`, ACK OpenKruise SandboxClaim, Cua, Smol golden.</li><li>Most "snapshots" are image templates that only cut startup time.</li></ul> | **Code path exists, untested.**<ul><li>`interop/harbor/world_backend.py:175-193`: any Harbor `environment_type` with `NetworkPolicy` mapped from the world, `gpus`/`gpu_types`, guard applied afterwards.</li><li>`interop/agentenv/sandbox.py`: Modal with `gpu=`, E2B, Sail, any registered provider; empty allowlist for no-network.</li><li>Tests: `tests/gpu/test_gpu_sandboxes.py` has a Modal test gated on `CRUCIBLE_MODAL=1`, never run.</li></ul> |

### 1.2 Feature matrix that matters for RL (upstream best vs Crucible)

| Feature | Why RL needs it | Best upstream | Crucible |
|---|---|---|---|
| Logical snapshot/restore of world state | GRPO groups from one start state; tree search; resets | Not a sandbox feature anywhere upstream | **Done, and stronger than upstream.** `World.snapshot()/restore()` (`world/world.py:574,605`) over State Adapters (fs, fs_git, sql, postgres, json_doc, …) plus clock, events, ACL and inboxes. `check_image` refuses cross-image restores. |
| Process / memory snapshot | Long-running services and REPL state | prime_sandboxes `checkpoint`; Smol CoW; OpenSandbox pause | **Missing.** The `process` adapter is fingerprint-only (`world/state/process.py`). No CRIU, `docker checkpoint` or VM snapshot. |
| Fork N copies cheaply | Group rollouts from a mid-episode state | Smol `fork_batch`; prime `checkpoint_id` | **Logical only.** Restore one WorldRef into N Worlds. Cost scales with state size; each World needs its own container start. |
| Pause / resume sandbox | Idle environments while the trainer holds the GPU; long-horizon tasks | OpenSandbox; ACA auto-suspend | **Missing.** `docker pause` would cost a few lines. |
| No-network | Default for coding worlds | Everyone | **Done.** All three container paths, plus the guard's egress probe that refuses to start. |
| Egress allowlist | Model-API-only worlds | Harbor Docker sidecar plus 11 cloud providers; verifiers proxy | **Clouds only** (Harbor generic branch, agentenv). See §1.3. |
| GPU passthrough | Kernel and CUDA tasks | NeMo-Gym Docker, Harbor Modal/GKE/Daytona/Beam/Prime, verifiers | **Done** for local and remote Docker (lab-proven in M7/M8) and coded for Modal. `runtime/gpu.py::preflight` refuses a wrong `compute_cap` or a non-init PID 1. |
| Resource limits | Fork bombs, OOM | Harbor `resource_policies.py` (limit/request modes) | **Partial.** pids, memory, cpus. No disk quota (verifiers also calls disk "advisory"). |
| Warm pools | Startup latency at scale | SkyPilot, ACK, Cua, rLLM `WarmQueue` | **Missing.** `envpool.py` pools environment *workers* (threads), not sandboxes. |

### 1.3 Crucible-specific findings (evidence)

1. **Allowlist worlds cannot run on local or remote Docker.** For Docker, `HarborBackend.provision` builds `GpuDockerEnvironment` with `network_none` only (`world_backend.py:196-200`). It never passes the world's `NetworkPolicy`, so Harbor's own egress-control sidecar (`HB/docker/docker.py:116-270`) is never enabled. `DockerSandbox.argv` has the same gap: it only knows `--network none`. Because `guard/isolate.py:120-125` raises `IsolationError` when the policy user can reach the internet in an allowlist world, the failure is closed (safe), but the feature is unusable. **Fix (S):** forward `network_policy=` to the Docker subclass and let Harbor's sidecar enforce it. This can be tested on the Mac.
2. **The capability model is too coarse to choose a backend.** `BackendCapabilities` (`world/backends/base.py:24-31`) has `isolation ∈ {none, container, vm}`, `gpus` and `network_enforced`. It has no gVisor or microVM tier, and no snapshot kind (logical, process or VM), fork, pause, allowlist or warm-start latency. Harbor's `EnvironmentCapabilities` already carries most of these and can be mapped.
3. **Harbor's generic branch and Kata are incompatible.** `world_backend.py:181-185` always builds a `NetworkPolicy`. For no-network worlds that is `NO_NETWORK`, and `kata.py` rejects every non-public policy. A no-network world on Kata fails before start, even though `--network none` would be enforceable.
4. **Hosts without Docker are unsupported.** The `GpuHost.kind` enum lists `ssh_exec`, `harbor_cloud` and `agentenv_cloud`, but nothing in `src/` reads them. The Runpod pod, with no Docker and no userns, can therefore only run graders by hand.

### 1.4 Sandbox gaps

| # | Gap | Priority | Effort | Acceptance test (where it runs) |
|---|---|---|---|---|
| S1 | Docker allowlist via Harbor's egress sidecar (and `DockerSandbox` via a proxy) | **P0** | S | **Mac.** An allowlist world `allowed_hosts=["pypi.org"]` on the Harbor Docker backend: the guard egress probe reports `isolated=true`, `lanes_ok=true`; `curl example.com` as the policy uid fails; `curl https://pypi.org` succeeds. |
| S2 | Capability model v2 (isolation tier, snapshot kinds, fork, pause, allowlist, gpu types) mapped from Harbor `EnvironmentCapabilities` / agentenv `supports_network_policy` | **P0** | S | **Mac unit test.** For every registered backend, `validate()` refuses a spec needing an absent capability (allowlist on Kata, GPU on local, fork on docker) before provisioning. |
| S3 | Cloud acceptance: Modal (gVisor and GPU) and one non-Modal provider (Daytona or E2B) through both the Harbor and agentenv paths | **P0** | S code / needs credentials | **Cloud.** `CRUCIBLE_MODAL=1 pytest tests/gpu/test_gpu_sandboxes.py::test_agentenv_gpu_sandbox_on_modal` passes (L4, `nvidia-smi` in-sandbox, snapshot/restore round trip, no-network egress probe isolated). The same world with `environment_type: daytona` passes. |
| S4 | `ssh_exec` host-process backend (rlimits, cgroup-less) for Docker-less GPU hosts | P1 | M | **Runpod.** The M8 GPU world runs with `backend: ssh_exec`. The preflight sees sm_89. The verifier score equals a manual run. The capability reports `isolation="none"` and refuses an untrusted policy shell unless opted in. |
| S5 | gVisor tier: `docker run --runtime=runsc` when registered, else runsc inside a privileged container | P1 | M | **Lab (runsc in a privileged container, no daemon change).** The toy coding world runs under runsc. `dmesg` inside shows the gVisor banner. The guard egress probe passes. Snapshot/restore round trip works. Also on the Mac (arm64 runsc, systrap platform). |
| S6 | Pause/resume (`docker pause` / Modal or OpenSandbox native) used while the trainer holds the GPU | P2 | S | **Mac.** A world paused for 30 s keeps `World.snapshot()` fingerprints identical, shows 0% CPU in `docker stats`, and resumes without a clock jump (virtual clock unchanged). |
| S7 | Process-level snapshot/fork: prime_sandboxes `checkpoint_id`, Harbor Smol `fork_batch`; local CRIU (`docker checkpoint`) | P2 | L | **Lab** (CRIU needs an experimental daemon, so prefer the Firecracker snapshot in S8) or **Prime cloud.** A world with a running `process` App (a REPL holding variable `x`) is forked to 4 copies; each sees `x`; diverging writes do not leak across copies. |
| S8 | microVM tier: Firecracker in a container with `--device /dev/kvm` (no sudo), and Kata through Harbor with a no-network fix (S1.3 #3) | P2 | L | **Lab.** A Firecracker microVM boots the world image's rootfs in under 2 s. Exec and file push/pull work through `Exec`. VM snapshot/restore restores a running process. `isolation="microvm"` is reported. |
| S9 | Full VM (QEMU/KVM, OSWorld-style desktop worlds) | P2 | L | **Lab.** A QEMU VM from a qcow2 with `--device /dev/kvm`; the `browser` App drives it; restore by overlay-discard takes under 10 s. |
| S10 | Sandbox warm pool (pre-started containers per image) | P1 | M | **Mac.** 64 rollouts on the toy pack: p50 `provision()` time with the pool is under 10% of the cold time; pooled containers are reset by `after_restore` hygiene (`/tmp`, home, IPC emptied). |
| S11 | Podman / Apple container / Apptainer via the Harbor generic branch | P2 | S each | **Mac** (Apple container needs macOS 26; podman needs install) / **HPC.** The toy world passes `test_harbor_backend` with `environment_type` swapped. |

---

## 2. Inference engines

### 2.1 What the trainers use (evidence)

| Trainer | vLLM | SGLang | Other |
|---|---|---|---|
| verl / MiMo-verl | <ul><li>In-process `AsyncLLM` in a Ray actor that also serves HTTP (`vllm_async_server.py:41,497`).</li><li>Custom `worker_extension_cls`; CUDA IPC buckets over ZMQ.</li><li>`sleep`/`wake_up(tags)`.</li><li>`pause_generation`; per-request `abort_requests_async`.</li><li>`logprobs_mode=processed_logprobs`; top-k.</li><li>`enable_return_routed_experts` (`:441-454,745-747`).</li></ul> | <ul><li>In-process tokenizer_manager plus HTTP.</li><li>`update_weights_from_tensor`; `release/resume_memory_occupation`.</li><li>`pause_generation(mode="abort")`.</li><li>`return_routed_experts`.</li><li>MiMo recipes all use SGLang.</li></ul> | <ul><li>TRT-LLM `AsyncLLM` (`trtllm_async_server.py:183`).</li><li>HF `generate` (`hf_rollout.py:112`).</li><li>checkpoint-engine backends (incl. Kimi).</li></ul> |
| slime | none | <ul><li>Server subprocess.</li><li>`update_weights_from_tensor/distributed/disk`.</li><li>`pause_generation(abort\|in_place)`; `abort_request{abort_all}` until idle.</li><li>`return_logprob`, `return_routed_experts` plus `routed_experts_start_len` for continuations (`sglang_rollout.py:190-238`).</li></ul> | Megatron training |
| AReaL | <ul><li>HTTP server with `/areal_*` routes.</li><li>`collective_rpc("areal_update_weight_xccl")`.</li><li>`/sleep`.</li><li>`pause_generation` before every update.</li></ul> | <ul><li>HTTP.</li><li>`update_weights_from_disk/distributed`.</li><li>Two-stage pause.</li><li>`return_routed_experts` (base64 int32).</li></ul> | AWEX colocated transfer. **Per-token `output_versions`.** |
| NeMo-RL | <ul><li>`LLM` or `AsyncLLM`.</li><li>IPC/ZMQ, collective, nccl_reshard, NIXL refit.</li><li>`pause_generation(mode="keep")`.</li><li>configurable `logprobs_mode`.</li><li>routed experts with R3 stats.</li></ul> | <ul><li>HTTP.</li><li>`pause_generation_mode: abort\|retract\|in_place`.</li></ul> | <ul><li>TRT-LLM.</li><li>**Megatron `DynamicInferenceEngine`**.</li><li>Dynamo.</li></ul> |
| prime-rl | <ul><li>Separate server: `/pause?mode=keep`, `/update_weights`, `/init_broadcaster`, a tokens API.</li><li>`processed_logprobs`; **`sampling_mask`**; routed experts.</li><li>In-flight update without abort (`orchestrator/clients.py:408`).</li></ul> | none | Dynamo |
| OpenRLHF | <ul><li>`AsyncLLMEngine` plus `WorkerWrap`.</li><li>NCCL or CUDA IPC.</li><li>`sleep`/`wake_up` tag by tag.</li><li>`pause(keep)`.</li></ul> | none | DeepSpeed |
| TRL | <ul><li>Server mode: vLLM-native NCCL weight transfer (`/init_weight_transfer_engine`, `/update_weights`); `--logprobs-mode processed_logprobs --max-logprobs -1` (`vllm_serve.py:223,270-275`).</li><li>Colocate: `LLM`, `sleep(level=2)`.</li></ul> | none | HF `generate` (`grpo_trainer.py:1761`) |
| ROLL / SkyRL | <ul><li>ROLL: `CustomAsyncLLM` with routed experts.</li><li>SkyRL: vLLM-native transfer; `PauseMode` ABORT/KEEP/WAIT; candidate logprobs as `sample_support`; routed experts.</li></ul> | <ul><li>ROLL: Engine or HTTP with routed experts.</li><li>SkyRL: none.</li></ul> | ROLL `hf_infer`; SkyRL JAX |

Cross-cutting facts:
- No trainer uses llama.cpp or MLX (zero grep hits).
- R3 is implemented in verl/MiMo, slime, AReaL (SGLang), NeMo-RL, prime-rl, ROLL and SkyRL.
- Pause semantics split into two camps:
  - **keep** (requests frozen, then continued on new weights): prime-rl, OpenRLHF, NeMo-RL vLLM, TRL async, SkyRL, slime `in_place`.
  - **abort + client resubmit**: AReaL, verl, slime `abort`, rLLM, ROLL.
- Moonshot checkpoint-engine is consumed only by verl's `kimi_checkpoint_engine.py`. It does H2D, then broadcast into a CUDA-IPC buffer, then `VllmColocateWorkerExtension.update_weights_from_ipc` via `/collective_rpc`.

### 2.2 Crucible today

| Piece | Status (evidence) |
|---|---|
| `Engine` protocol (`train/engine.py:67-75`) | `generate`, `abort_all`, `pause`, `resume`, `load_weights(state, version)`, `capacity`, `version`. `GenRequest` carries `resume_*` (token-level resume), `top_p`, `resume_candidates`, `resume_routing`. `GenResult` carries `completion_ids`, `logprobs`, `version`, `finish`, `version_spans`, `candidates`, `routing`, `reprefill_tokens`. |
| `EnginePool` | Sticky crc32 routing with least-loaded fallback, plus busy-time accounting. |
| `TorchEngine` / `ScriptedEngine` | **CPU only and TinyPolicy only.** `forward` calls `serve(torch.tensor([seq]))` without a device, and `HFPolicy` (`train/hf.py`) has no `forward()`. Today **no engine can serve a real HF model on CPU or MPS**. Numerics `fp32 \| bf16 \| bf16_full`. Candidates and routing are reported. |
| `VLLMServerEngine` (`train/vllm_http.py`) | Uses `/v1/completions` with token ids, `return_token_ids`, `logprobs=0`. Calls `/pause?mode=abort`, `/sleep?level=1` (colocated), `/wake_up?tags=weights`, then `/collective_rpc → CrucibleWorkerExtension.crucible_load_weights` (safetensors from disk), `/reset_prefix_cache`, `/wake_up?tags=kv_cache`, `/abort_requests`. Gaps:<ul><li>Refuses `top_p<1` (`NotImplementedError`).</li><li>Returns no `routing`.</li><li>Has **zero tests** (not even against a fake HTTP server).</li><li>Has **never run on a GPU**: the M12 GPU gate's first run (Qwen3-0.6B, lab) produced 0 steps because of a chat-codec bug; it has been re-staged (`docs/modules/M12-train.md:6,134`).</li></ul> |
| Weight plane | <ul><li>`InprocPublisher`, `CheckpointPublisher` (`train/weights.py`).</li><li>`BroadcastWeightSync` (`train/weightsync.py`): checkpoint-engine-style bucket plan, double buffer, atomic commit, subgroup updates. **Tested over gloo only** (`tests/unit/_weightsync_worker.py:41`), and **not wired to vLLM**; its docstring defers CUDA IPC.</li></ul> |
| Actor / judge LLM client | `actors/llm.py`: any OpenAI-compatible endpoint (vLLM/SGLang), recording a token-in-token-out record. This is for actors, not the training engine. |
| Bridges | TRL `rollout_func` over TRL's colocated vLLM (`bridges/trl_bridge.py`); verl estimator (`bridges/verl_bridge.py`); adapters for SkyRL and rLLM. These delegate engines to the host trainer. |

**Colocated memory-order bug (P0, evidence).** In `runtime.py` (around line 425), the step sequence is: `abort_all` → `pause` (vLLM sleeps) → train → `publish()`, then `trainer.offload(...)`, then `engines.resume()`. Inside `publish`, `VLLMServerEngine.load_weights` wakes **both** the weights and the **KV cache** (`vllm_http.py:99-105`). That happens before the trainer offloads. The peak is therefore trainer weights + optimizer + grads **plus** a fully woken engine, which defeats sleep mode on one GPU. Two fixes are needed:
- Wake `kv_cache` in `resume()` (after the offload), not in `load_weights`.
- When `sleep_level=2`, reload from disk after the offload.

TRL does this split: `wake_up(tags=["weights"])` for the sync (`vllm_generation.py:491`) and `["kv_cache"]` later (`:705`).

### 2.3 What the `Engine` protocol needs for each engine

| Need | vLLM | SGLang | HF generate (CPU/MPS/CUDA) | TRT-LLM | llama.cpp / MLX (Mac) | Protocol change |
|---|---|---|---|---|---|---|
| Token ids in, ids + behaviour logprobs out | `return_token_ids`, `logprobs=0`, `--logprobs-mode processed_logprobs` | `/generate` with `input_ids`, `return_logprob` | direct | `AsyncLLM` | llama-server `/completion` with `n_probs`; mlx-lm logits | none |
| Version stamping | ours | ours (or `meta_info.weight_version`) | ours | ours | ours | none |
| Abort one request | `/abort_requests` (verl uses per-request) | `abort_request(rid)` | cooperative flag | `abort` | cancel | **add `abort(call_id)`** (today only `abort_all`) |
| Pause mode | `/pause?mode=abort\|keep` | `pause_generation(abort\|retract\|in_place)` | flag | `pause_generation` | n/a | **add `pause(mode)`**; keep mode enables prime-rl-style in-flight updates and `version_spans` without resubmission |
| Sleep / wake split | `/sleep?level`, `/wake_up?tags` | `release/resume_memory_occupation(tags)` | `.to("cpu")` | `release_memory_occupation` | n/a | **separate `sleep(level)` / `wake(tags)`** from `pause` (see bug above) |
| Weight update | `/collective_rpc` worker ext (disk today); IPC; vLLM-native NCCL transfer | `update_weights_from_tensor/distributed/disk` | `load_state_dict` | `update_weights_from_ipc_handles` | reload GGUF / `mlx` weights (slow) | `load_weights(handle, version)` with a handle kind: `state \| path \| ipc \| nccl` |
| Top-p candidates | top-k logprobs with `--max-logprobs -1` are approximate; exact = prime-rl `sampling_mask` (V2 runner) | `return_top_logprob`/top-p fields (slime `rollout_top_p_*`) | exact | top-k | `n_probs` | add `GenResult.topk` plus a `candidates_exact: bool` capability |
| Routed experts (R3) | `enable_return_routed_experts` → `outputs[0].routed_experts` | `return_routed_experts` + `routed_experts_start_len` | hook | no | no | `routing` exists; add a `routing_start` offset for continuations (slime pattern) |
| Introspection | server args | `/get_server_info` | – | – | – | **add `capabilities()`**: logprobs_mode, candidates, routing, pause modes, sleep levels, max_logprobs; refuse misconfigured servers at start |

### 2.4 Engine gaps

| # | Gap | Priority | Effort | Acceptance test |
|---|---|---|---|---|
| E1 | `VLLMServerEngine` unit tests against a fake vLLM HTTP server (abort, pause, sleep/wake order, version stamping, finish mapping) | **P0** | S | **Mac.** pytest with an `httpx.MockTransport` that records the endpoint sequence. Assert that a colocated publish issues `wake_up?tags=weights` → `collective_rpc` → `reset_prefix_cache`, with no `kv_cache` wake until `resume()`. |
| E2 | Colocated wake-order fix (§2.2) plus an engine `sleep`/`wake` split | **P0** | S | **Lab.** Qwen3-0.6B colocated GRPO, 20 steps. `torch.cuda.max_memory_allocated` (trainer) + vLLM `nvidia-smi` used at publish ≤ max(train peak, engine peak) + 10%. Logged per step. |
| E3 | M12 GPU gate passes with vLLM 0.31 (already staged) | **P0** | S | **Lab.** 0 masked-by-harness rollouts, ≥ 20 steps, reward slope > 0 on the toy pack. `rollout_probs_diff` p99 < 0.05 with `processed_logprobs`. Staleness recorded. |
| E4 | `Engine.capabilities()` with a startup probe: send a fixed prompt, recompute logprobs with `HFPolicy`, and refuse if the gap shows the wrong `logprobs_mode` | **P0** | S | **Lab.** A server started *without* `--logprobs-mode processed_logprobs` at T=0.7 is refused at engine start; one started with it is accepted. |
| E5 | `HFGenerateEngine` (KV-cached `generate`, device-aware, numerics knob, candidates exact, MoE routing hook) | **P0** | M | **Mac (MPS).** Qwen3-0.6B colocated `colocated_partial` for 10 steps on MPS with `HFPolicy`. Behaviour logprobs equal trainer recompute within 1e-3 (fp32). Abort and token-resume work. This also closes "no real model on Mac". |
| E6 | `SGLangServerEngine` (`/generate` with ids, `return_logprob`, `pause_generation(mode)`, `release/resume_memory_occupation`, `update_weights_from_disk` then `_from_tensor`, `return_routed_experts`) | **P0** | M | **Lab** (`lmsysorg/sglang` image, `--runtime=nvidia`). The same gate as E3 passes on SGLang. A publish with `pause_generation(in_place)` keeps in-flight requests and records `version_spans`. |
| E7 | `pause(mode="keep")` end to end (prime-rl in-flight update): calls continue on new weights, recorded as spans, no resubmission | P1 | M | **Lab** (vLLM `/pause?mode=keep`). `reprefill_tokens == 0` across 10 publishes; every crossing call has `version_spans` with 2 entries; trainer refuses or corrects per checklist #34. |
| E8 | vLLM routed-experts capture for R3 | P1 | S | **Lab.** A small MoE (Qwen1.5-MoE-A2.7B fits in 32 GB at bf16? If not, OLMoE-1B-7B). `GenResult.routing` covers 100% of completion tokens. `routing_replay="r3"` reduces the trainer/engine logprob gap on the same batch. |
| E9 | vLLM top-p candidates (top-k with `--max-logprobs -1`, or the V2 `sampling_mask`) | P1 | M | **Lab.** `top_p=0.9`: every token has `candidates`; the renormalized trainer logprob matches the engine within 1e-2. |
| E10 | TRT-LLM engine | P2 | L | **Lab** (TRT-LLM supports sm_120 in recent releases; verify). Gate E3 on TRT-LLM. |
| E11 | llama.cpp / MLX engine for Mac evaluation (weight update by file reload, so eval/actors only) | P2 | M | **Mac.** An actor or eval run against `mlx_lm.server` / `llama-server` records TITO with logprobs. No training path. |

---

## 3. GPU paths

| Path | Upstream (evidence) | Crucible status | Testable where |
|---|---|---|---|
| **Single GPU colocated, sleep/wake** | <ul><li>verl HYBRID/COLOCATED (`replica.py:49-62`).</li><li>TRL colocate `sleep(level=2)`.</li><li>OpenRLHF `colocate_all` plus vLLM/DeepSpeed sleep.</li><li>NeMo-RL `colocated.enabled`.</li><li>SkyRL `colocate_all=True` (default).</li><li>slime `--colocate` forces offload.</li><li>AReaL torch_memory_saver.</li></ul> | <ul><li>**Coded, unvalidated:** `colocated_partial`/`colocated_sync` topologies; vLLM sleep level 1; `TorchTrainer.offload(cpu\|nvme)` (moments preserved; ≤5% resident measured on MPS).</li><li>Wake-order bug (§2.2).</li></ul> | Lab |
| **Memory split on one GPU** | <ul><li>`gpu_memory_utilization`: verl 0.5, TRL 0.3, NeMo-RL 0.6, SkyRL 0.8, AReaL 0.9.</li><li>prime-rl uses disjoint GPU lists (no colocate).</li></ul> | `VLLMServerEngine(placement="split")` never sleeps; `disaggregated_stream` topology. Unvalidated on GPU. | Lab |
| **Disaggregated pools** | verl STANDALONE; prime-rl `partition_gpus`; AReaL `+` allocation; ROLL `device_mapping` | Topology `disaggregated_stream` (trainer in a thread or another process). There is no placement or launcher layer (Ray or `CUDA_VISIBLE_DEVICES` partitioning). | Multi-GPU rental |
| **Engine TP/DP/EP, replicas, routing** | <ul><li>Every trainer exposes TP/DP/PP/EP.</li><li>Routers: verl `GlobalRequestLoadBalancer` (sticky + least-inflight), NeMo-RL least-outstanding, slime sglang_router, prime-rl vllm-router.</li></ul> | <ul><li>`EnginePool` sticky plus least-loaded over N engines (tested with CPU engines).</li><li>TP is a server flag outside Crucible.</li><li>No launcher.</li><li>`EnginePool.load_weights` is sequential, so N replicas take N× publish time.</li></ul> | <ul><li>DP: lab (2 vLLM servers at 0.3 each).</li><li>TP>1: rental.</li></ul> |
| **NCCL weight sync** | <ul><li>OpenRLHF/prime-rl `PyNcclCommunicator`+`StatelessProcessGroup`.</li><li>verl nccl checkpoint engine (cupy, 2× bucket).</li><li>TRL/SkyRL vLLM-native NCCL transfer.</li><li>NeMo-RL nccl_reshard.</li></ul> | `BroadcastWeightSync` is correct over gloo with real processes. NCCL is untested and the engine receiver is not wired. | Needs ≥ 2 GPUs: NCCL refuses two ranks on one device |
| **CUDA IPC (colocated zero-copy)** | verl buckets over ZMQ; OpenRLHF `reduce_tensor`; checkpoint-engine `VllmColocateWorkerExtension`; ROLL, SkyRL | Missing (disk safetensors only) | Lab: one GPU, trainer process + vLLM container with `--ipc=host` |
| **Trainer sharding** | <ul><li>FSDP2: verl, AReaL, prime-rl, NeMo-RL, ROLL, SkyRL, TRL.</li><li>Megatron: verl, slime (only), AReaL, NeMo-RL, ROLL, SkyRL.</li><li>DeepSpeed: OpenRLHF, TRL.</li></ul> | `DistributedTorchTrainer`: DP gradient all-reduce, no sharding. FSDP and Megatron only through the verl bridge (estimator plus `AgentLoopOutput`; never run end to end). | FSDP2 with 1 rank: lab; real sharding: rental |
| **CUDA inside the sandbox** | <ul><li>NeMo-Gym Docker `--gpus`, Apptainer `--nv`, Enroot.</li><li>Harbor Modal/GKE/Daytona/Beam/Prime/Smol; Harbor Docker has **no** GPU.</li><li>agentenv Modal only.</li><li>OpenEnv rejects GPU on docker-local.</li><li>Nobody upstream uses `runtime=nvidia`.</li></ul> | **Done and lab-proven:** `DockerSandbox`, `GpuDockerEnvironment`, agentenv `GpuDockerSandboxProvider` (`--runtime=nvidia` for the lab quirk), `preflight` arch check, Modal coded. | Lab; Modal (credentials) |
| **Apple MPS** | Only TRL handles MPS (device_map, `empty_cache`); no MLX anywhere | Trainer offload measured on MPS; float32 loss terms on MPS (`trainer.py:297-299`). **No MPS engine** for real models (E5). | Mac |

### 3.1 GPU gaps

| # | Gap | Priority | Effort | Acceptance test |
|---|---|---|---|---|
| G1 | Validate colocated and split placements on the lab GPU (E2/E3 cover colocated) | **P0** | S | **Lab.** The same 20-step run in `disaggregated_stream` with `placement="split"` (vLLM 0.45, trainer resident): engines never pause; `engine_idle_s` < 10% of wall; no OOM. |
| G2 | CUDA IPC publish to a colocated vLLM (port checkpoint-engine's `VllmColocateWorkerExtension` pattern into `vllm_ext.py`) | P1 | M | **Lab.** Publish latency for Qwen3-0.6B drops by ≥ 3× versus `CheckpointPublisher`. Post-publish greedy outputs are byte-identical to a disk reload. |
| G3 | `BroadcastWeightSync` over NCCL into vLLM workers (`EngineReceiver` → worker extension), with parallel `EnginePool.load_weights` | P1 | M | **2-GPU rental** (Runpod 2×4090 has no Docker, so run vLLM natively). Trainer on GPU0, vLLM on GPU1. A plan-crc mismatch is refused on both. An injected failing bucket leaves the old version serving (atomic commit). |
| G4 | Placement/launcher layer: partition `CUDA_VISIBLE_DEVICES`; start N engine servers (TP, DP) from a run spec; health checks | P1 | M | **Lab** (DP=2 at 0.3 memory each) plus **rental** (TP=2). The run spec starts the servers, `EnginePool` routes stickily, and kill -9 of one engine is recovered by a subgroup update (`ranks=[...]`). |
| G5 | FSDP2 trainer backend (`fully_shard`, CPU offload policy) behind `TrainerBackend`, or a proven verl end-to-end bridge run | P1 | L | **Lab** (1 rank: loss and gradients equal `TorchTrainer` within 1e-5) plus **rental 2 GPUs** (sharded = unsharded). Alternatively, a verl GRPO run with Crucible worlds on the lab GPU, 20 steps. |
| G6 | Megatron (MoE, EP) via the slime or verl bridge | P2 | L | **Rental** ≥ 2 GPUs. An OLMoE run with R3 through slime's `RoutingReplay`. |
| G7 | MPS colocated real-model loop (needs E5) | P1 | S after E5 | **Mac.** Qwen3-0.6B, 10 steps, `colocated_partial` on MPS, offload `cpu`. Memory stays within the 24 GB unified limit. |
| G8 | GPU sandbox on a cloud class (H100/L4) through Harbor `modal` and agentenv Modal | P1 | S (credentials) | **Modal.** `preflight(gpu_arch="8.9"/"9.0")` passes; a CUDA kernel task from the M8 pack verifies. |

---

## 4. Priorities in one place

| Priority | Items | Rationale |
|---|---|---|
| **P0** | S1 allowlist on Docker, S2 capability model, S3 cloud acceptance (needs credentials), E1 vLLM engine tests, E2 wake-order fix, E3 GPU gate, E4 capability probe, E5 HF engine (Mac real model), E6 SGLang engine, G1 placement validation | Without these the one real engine is unproven, allowlist worlds cannot run on the main backend, the Mac cannot train a real model, and the engine most R3/MoE stacks use (SGLang) is absent |
| **P1** | S4 ssh_exec, S5 gVisor, S10 warm pool, E7 keep-mode pause, E8 vLLM R3, E9 top-p, G2 CUDA IPC, G3 NCCL, G4 launcher, G5 FSDP2, G7 MPS loop, G8 cloud GPU | Scale and fidelity: each is used by ≥ 3 upstream trainers or unlocks a test bed we own (Runpod, the lab's KVM) |
| **P2** | S6 pause, S7 process snapshot/fork, S8 microVM, S9 full VM, S11 other runtimes, E10 TRT-LLM, E11 llama.cpp/MLX, G6 Megatron | Upstream support is thin (Smol, prime checkpoint and OpenSandbox are the only fork/pause primitives; no trainer uses MLX or llama.cpp) |

**Test-bed reality.**
- Mac: S1, S2, S6, S10, S11, E1, E5, E11 and G7.
- Lab RTX 5090: E2–E4, E6–E10, G1, G2, the 1-rank half of G5, and S5/S8/S9. Isolation work there must run inside containers given `/dev/kvm` or `--privileged`, because adding a daemon runtime needs sudo on a shared host.
- Runpod: S4.
- Multi-GPU rental: G3, the TP half of G4, multi-rank G5, and G6.
- Modal or other credentials: S3 and G8.
