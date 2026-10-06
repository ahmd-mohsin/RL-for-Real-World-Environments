# Task 02: pcops CUDA extension → Triton (reference port log, RTX 5090)

**Workspace:** three repos, a migration target plus two pristine consumers.
- **`pcops`** is a compiled PyTorch CUDA extension with 15 operators:
  - from `pytorch_cluster` (MIT, vendored at `e9a855c`): fps, knn, radius, grid;
  - from `pytorch_scatter` (MIT, vendored at `f514c10`): segment_csr, gather_csr, scatter;
  - written for this task: PointNet++ ball query, group/gather points, three_nn, three_interpolate.
- **`pcnet`** is a PointNet++ segmentation model, an edge-convolution branch and a pipeline. It uses every operator and compiles with `fullgraph=True`.
- **`pcdata`** is a synthetic data generator (duplicates, integer lattices, outliers).

The work was developed without a GPU (Triton interpreter mode on CPU against a CPU reference), then validated and tuned on the lab RTX 5090.

## Bugs found in the original while building the task (fixed in `repos/`)

| # | Bug | How it showed | Fix |
|---|---|---|---|
| 1 | `fps_cuda` read the output size with `cudaMemcpy` on the legacy stream (upstream `pytorch_cluster`) | Illegal memory access when called on a non-default stream: the copy raced the size computation | `out_ptr[-1].item()`, which also removes a leaked `malloc` |
| 2 | `gather_csr` had no autograd formula | `pcnet` training crashed in backward | backward = `segment_csr(sum)` |
| 3 | The extension module had no `PyInit__C` | `import pcops._C` failed | empty module definition |
| 4 | Integer `segment_csr` mean truncates (C++ division); `scatter` mean floors | Reference mismatch | reference follows the extension |

## Traps for a port (the hidden suite checks each)

- **Exact tie-breaking.** kNN and three_nn keep the earliest index on equal distances (strict `<` insertion). Radius and ball query return the first hits in index order; ball query pads with the first hit and leaves 0 when there is none. `torch.topk`, `argsort` and atomic compaction all break ties differently. The checks use integer lattices, where squared distances are exact.
- **FPS race and properties.** The CUDA FPS reduction is racy on ties, so FPS is checked by property: each sample maximises the minimum distance, capped at 5e4. Other cases: any dimensionality (the original accepts D ≠ 3), `random_start`, and examples of 0, 1 or 150k points.
- **Interpreter versus compiler.** Code that passes `TRITON_INTERPRET=1` can fail to compile. Loop-carried values must keep one dtype (int64 `start` versus int32 `argmax`).
- **Registration contract.** Every operator is run through `torch.library.opcheck`: schema, fake tensors, autograd registration, AOT dispatch with dynamic shapes. The backward operators stay registered, and data-dependent sizes use `new_dynamic_size`.
- **Determinism.** Every operator except `scatter` is bitwise reproducible, including a 3M-row fp32 `segment_csr`. An atomics-based `segment_csr` fails this.
- **Streams and threads.** Results must be identical on a side stream (queued behind a long matmul) and from 6 concurrent threads. Global scratch buffers fail this.
- **Precision.** Half and bf16 inputs accumulate in fp32 (the extension uses half atomics, so contended half sums are only compared in fp32). Double inputs need fp64 accumulation for gradcheck.
- **Consumers stay pristine.** `pcnet` must reproduce the CUDA extension's logits and its 12-step training-loss trajectories, eagerly and compiled.

## Reference port design (`solution/solution.patch`, 29 files)

- **fps:** each example's coordinates and running distances stay in registers for the whole run (examples up to 8192 points, D ≤ 3). A chunked global-memory kernel handles larger examples. One example per program.
- **knn:** packed int64 keys (order-preserving float bits, then the index) and a bitonic top-k merge per 64-point tile: `min(A[i], B[k-1-i])`, then re-sort. Cosine distance uses a signed-float order transform.
- **three_nn:** three min-extractions per tile plus a 3-element insertion network, instead of a full sort.
- **radius / ball_query:** prefix-sum compaction per tile (ordered, no atomics), tuned to BQ=8/BX=256 and BQ=64/BX=128 on the 5090.
- **segment_csr / gather_csr:** one program per segment for the reduction. `gather_csr` uses a binary search per output row in 256×32 tiles, and writes every row so no memset is needed.
- **scatter:** atomics into an fp32 (or native) accumulator, plus an arg pass (smallest index on ties).
- **Registration:** `torch.library.custom_op` with the original schemas. The original `_registration.py` (fake implementations and autograd) is reused unchanged.

## Performance (RTX 5090, grader run, ms)

| case | CUDA extension | Triton reference | ratio | bar |
|---|---|---|---|---|
| fps 16×16k r0.25 | 112.2 | 30.0 | 0.27 | 0.50 |
| fps 4×65k r0.0625 | 432.7 | 110.5 | 0.26 | 0.50 |
| knn k16 / k64 | 22.1 / 132.5 | 19.3 / 6.4 | 0.87 / 0.05 | 1.00 / 0.25 |
| radius / ball_query | 4.79 / 1.92 | 3.88 / 1.55 | 0.81 / 0.81 | 1.10 |
| three_nn | 6.68 | 3.08 | 0.46 | 0.75 |
| group fwd / bwd | 1.15 / 3.20 | 1.04 / 3.07 | 0.91 / 0.96 | 1.15 / 1.20 |
| three_interpolate fwd / bwd | 0.43 / 1.53 | 0.48 / 1.65 | 1.12 / 1.08 | 1.35 / 1.30 |
| segment sum / max, gather_csr | 0.33 / 0.35 / 0.71 | 0.34 / 0.37 / 0.74 | ≤ 1.08 | 1.30 |
| scatter sum / mean / max / contended | 1.9 / 3.5 / 3.6 / 1.7 | ≈ same | ≤ 1.01 | 1.25 |
| pcnet train step / inference | 14.5 / 11.6 | 8.0 / 4.9 | 0.55 / 0.42 | 0.80 / 0.65 |

## Grader validation (`tests/grade.py`)

| candidate | static | install | ops | hard | app | perf | reward |
|---|---|---|---|---|---|---|---|
| original CUDA extension | FAIL | FAIL | – | – | – | – | 0 |
| knn via `torch.cdist` + `topk` | pass | pass | FAIL (tie order) | FAIL (no Triton launch) | pass | – | 0 |
| straightforward Triton FPS (256-point chunks like the CUDA block) | pass | pass | pass | pass | pass | FAIL (fps 1.15×, pcnet 0.98–1.0×) | 0 |
| **reference port** (in the task image via `tests/test.sh`, 3 runs) | pass | pass | 138/138 | 37/37 | 8 + pcnet/pcdata | 20/20 | **1** |

The hidden suite has 183 test checks plus 20 performance cases. The CUDA extension passes every correctness check except the Triton-launch check (by design).
