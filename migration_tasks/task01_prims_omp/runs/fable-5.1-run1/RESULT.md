# Fable 5.1 single-attempt run: task01_prims_omp

**Reward: 1.0. Every gate passed on the first attempt.**

- **Harness:** mini-swe-agent 2.4.6, bash only, yolo mode, 250-step / $60 cap, reasoning effort high (`common/mswea/migration.yaml`).
- **Sandbox:** fresh Runpod RTX 4090 pod from `ahmedmohsin/task01-prims-omp:v1`. The hidden tests were uploaded only after submission.
- **Usage:** 200 steps, $50.46 API, about 3 h wall time. Fable submitted on its own.
- **Audit:** no network commands; no access to the grader or hidden tests.

Hidden suite: 15/15 test cases (133 assertions) on the GPU and on the host-offload device.

| case | CUDA ms | Fable ms | ratio | bar | my oracle |
|---|---|---|---|---|---|
| exclusive_scan_u32 | 1.179 | 0.999 | 0.85 | 1.30 | 0.97 |
| inclusive_scan_u32 | 1.180 | 1.037 | 0.88 | 1.30 | 1.00 |
| reduce_sum_f32 | 0.295 | 0.366 | 1.24 | 1.60 | 1.36 |
| reduce_max_f32 | 0.301 | 0.380 | 1.26 | 1.60 | 1.33 |
| histogram256 | 0.441 | 0.388 | 0.88 | 1.30 | 0.99 |
| radix_sort_u64_44bit | 26.36 | 29.20 | 1.11 | 1.50 | 1.22 |
| radix_sort_u32_32bit | 15.29 | 15.59 | 1.02 | 1.50 | 1.25 |
| compact_indices | 2.032 | 0.528 | 0.26 | 1.45 | 1.16 |

What it did:

- **Baseline first.** It built the CUDA original and measured the targets, then probed OpenMP team-memory, SPMD and globalization behaviour, launch latency, and the host device with `OMP_NUM_THREADS=3`.
- **Portable lane loops.** Grid-strided team kernels that adapt to the thread count actually granted, so GPU and host share one source.
- **Fused final steps.** It used my oracle's last-team-done trick (ticket + flush) everywhere: reduce and histogram take 1 launch, scan and compact 2, sort 2 per digit.
- **Bugs it found and fixed:**
  - `omp_get_initial_device` resolving to libomp's stub, depending on the consumer's link order;
  - a team scan that assumed ≥16 threads (my pitfall 7);
  - a register-hungry scan that halved occupancy;
  - weak device functions with OpenMP constructs mismerged across translation units.

**Conclusion:** the task as specified is not hard enough to defeat a frontier model in one attempt. Fable beat my reference port on every case. The design of the hidden checks was sound: Fable had to anticipate the portability trap itself, and it did.
