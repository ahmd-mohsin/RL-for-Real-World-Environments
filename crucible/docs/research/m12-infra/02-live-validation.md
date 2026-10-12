# M12 infra: live validation on real backends

Raw results: `02-live-validation.json` (Modal) and `live/` (lab). This document grows as each backend's checks land.

## 1. Multi-GPU NCCL on Modal (4 × L4, driver 610.57, torch 2.14.1+cu130, NCCL 2.30.7)

`scripts/modal_infra_check.py` re-runs the gloo multi-process scenarios (`tests/unit/_weightsync_worker.py`,
`_fsdp_worker.py`) with one process per GPU over NCCL and applies the same assertions.

| scenario | ranks | result | soak | notes |
|---|---|---|---|---|
| weight-sync broadcast (many buckets, every dtype, two versions) | 3 | pass | 10 rounds, 3.8–4.1 s each | bitwise on every receiver |
| sharded owners, colocated receivers | 3 | pass | 1 | |
| failed reload → all-rank abort, nobody commits | 3 | pass | 1 | every rank names rank 2, bucket 2 |
| subgroup update + elastic plan | 3 | pass | 1 | |
| `DistributedPublisher` → `EngineReceiver` | 2 | pass | 1 | the engine serves version 2 bitwise |
| slow reload, double buffer | 3 | pass | 1 | |
| DP replicas = full-batch update (`dis`, `ppo`) | 2 | pass | 10 rounds | replicas bitwise identical |
| DP with the batch broadcast from rank 0 | 2 | pass | 1 | |
| FSDP2 step and layer modes, 2 updates, skewed batch | 2 | pass | 5 rounds | max diff vs single process 1.2e-6 to 3.9e-6 |
| HSDP 2 × 2 | 4 | pass | 1 | 1.8e-6 / 3.3e-6 |
| HF Llama (3 blocks), layer mode | 2 | pass | 1 | |
| `Runtime` through `SPMDDriver` over 4 HSDP ranks, DCP checkpoint, resume | 4 | pass | 1 | reward 0.05 → 0.24, staleness ≤ 1, resume exact |

Defects found here and fixed (DESIGN.md B.23–B.26):
- NCCL control tensors were on the CPU (DP scalar reduce, FSDP count, weight-sync header and error code).
- GRU weight flattening under FSDP2 on CUDA read stale weights.
- `TorchEngine` built its inputs on the CPU.
- Modal needs `NCCL_CUMEM_ENABLE=0`.

## Cost tally (Modal, estimated from GPU-minutes; budget $30)

| run | GPUs | approx. minutes | approx. $ |
|---|---|---|---|
| first contact + cuMem diagnosis | L4:2 | 8 | 0.25 |
| full NCCL suite + 10-round soaks | L4:4 | 32 | 1.70 |
| FSDP debug, probes (4 short runs) | L4:2 | 16 | 0.45 |
| FSDP reruns (2) | L4:4 | 25 | 1.35 |
| **total so far** | | | **≈ 3.75** |
