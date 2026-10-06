# Replace the pcops CUDA extension with Triton kernels

`/workspace` contains three repositories (each its own git repo):

| repo | what it is |
|---|---|
| `pcops` | Point-cloud and sparse-reduction operators for PyTorch, built as a compiled CUDA extension (`csrc/`, `setup.py`). It contains farthest point sampling, kNN, radius graphs and grid clustering (from `pytorch_cluster`); segment and scatter reductions (from `pytorch_scatter`); and PointNet++ ball query, grouping, three-NN and interpolation. Operators are registered as `torch.ops.pcops.*`; autograd and `torch.compile` support live in `pcops/_registration.py`. |
| `pcnet` | A PointNet++-style part-segmentation model and inference pipeline built on `pcops`. |
| `pcdata` | Synthetic point-cloud generator used by both. |

The compiled extension is a constant source of pain: it pins a CUDA toolkit, takes minutes to build, and has to be rebuilt for every GPU architecture and PyTorch version. **Port `pcops` to Triton.** The package must become pure Python and Triton, with the same behaviour and at least the same performance.

**The machine:** an NVIDIA RTX 5090 (Blackwell), CUDA 12.8, PyTorch with Triton in `/opt/venv`. The original extension is built and installed from `/workspace/pcops`, so you can compare against it at any time.

## Requirements

1. **No native code.**
   - `pcops` must contain no C/C++/CUDA sources and no extension build (`CUDAExtension`, `cpp_extension`, `load_inline`).
   - It may not load shared libraries or use other kernel libraries (`ctypes`, `cffi`, `cupy`, `numba`, `pycuda`, `torch_scatter`, `torch_cluster`).
   - Every operator must run its GPU work in Triton kernels (`@triton.jit`).
   - `pip install ./pcops` must work on a machine without a CUDA compiler.
2. **Same operators.**
   - Every `torch.ops.pcops.*` operator keeps its name and schema (see `csrc/pcops.cpp`), including the `*_backward` operators.
   - The public Python API in `pcops/__init__.py` stays the same.
   - Registrations must pass `torch.library.opcheck`, so they work with autograd, fake tensors, dynamic shapes and `torch.compile(fullgraph=True)`, as they do today.
   - Invalid arguments raise `RuntimeError` where the extension does.
3. **Same results.**
   - Integer outputs must match the extension exactly: sampled indices, neighbour lists and their order, cluster ids, `arg` outputs. That includes how the extension breaks distance ties.
   - Floating-point outputs must match up to rounding, for every dtype the extension supports, for contiguous and non-contiguous inputs, and for empty and degenerate inputs.
   - Every operator except `scatter` must be bitwise deterministic.
   - Operators must be correct on non-default CUDA streams and when called from several threads at once.
4. **The consumers keep working unchanged.**
   - `pcnet` and `pcdata` are graded from their original sources. Your changes to them are discarded.
   - `pcnet` must produce the same logits and the same training trajectory as with the CUDA extension, eagerly and under `torch.compile`.
5. **Performance.** The grader times each case below against the original extension on the same GPU (median of 15 calls after warm-up, synchronised). Your time may not exceed the given multiple of the extension's time. Several limits are below 1, where the extension's algorithm leaves a lot on the table.

| case | max time / CUDA extension |
|---|---|
| `fps` (16 × 16k points, ratio 0.25; 4 × 65k points, ratio 0.0625) | 0.50 |
| `knn` k = 16 / k = 64 | 0.60 / 0.25 |
| `three_nn` | 0.75 |
| `radius`, `ball_query` | 1.10 |
| `group_points` forward / backward | 1.15 / 1.20 |
| `three_interpolate` forward / backward | 1.35 / 1.30 |
| `segment_csr` (sum, max), `gather_csr` | 1.30 |
| `scatter` (sum, mean, max, highly contended sum) | 1.25 |
| `pcnet` training step / inference (8 × 4096 points) | 0.65 / 0.50 |

## Grading

`pcops` from `/workspace/pcops` (your diff against the `base` tag) is installed without a CUDA compiler. Hidden tests then exercise every operator: semantics, dtypes, memory layouts, edge cases, gradients, registration checks, determinism, streams and threads, dynamic-shape compilation, and large inputs. They also run the original `pcnet` and `pcdata` suites, the golden `pcnet` results recorded with the CUDA extension, and the benchmark above. All of it must pass.
