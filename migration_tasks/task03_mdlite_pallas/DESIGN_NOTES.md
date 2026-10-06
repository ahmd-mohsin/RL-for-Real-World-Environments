# Task 03 design notes (not shown to the agent)

## Research: where frontier models fail on similar work (collected 2026-10-05)

| source | finding | how task 03 uses it |
|---|---|---|
| FESOM2 Fortran→C→Kokkos LLM port (arXiv 2606.11356) | latent wrong constants that only blow up after long runs at production dt; defaults used instead of namelist values; loop-bound coverage bugs break conservation | long NVE drift runs at production dt; every config key is varied by the hidden suite; conservation checks |
| JAX `.at` docs and Sharp Bits | OOB gathers clamp silently and OOB scatters drop silently; duplicate-index scatter order is undefined; x64 is off by default | dense clusters that overflow a static neighbour capacity; atoms at or outside box edges and 1000 boxes away; fp64 unwrapped positions |
| XLA determinism doc | GPU scatter uses atomics and is nondeterministic | bitwise run-to-run and checkpoint-restart reproducibility |
| Pallas BlockSpec docs, Correctness Illusion (arXiv 2606.20128) | power-of-two block rules, unspecified padding, shapes that pass but fail on fuzzed sizes | prime and odd atom counts, N=1 and N=3, grid sizes with factors 3·5·7 |
| JAXBench (Pallas on TPU) | 99.7% of best-of-N samples failed on Pallas API misuse | Pallas use is required for the hot kernels (trace probe) |
| METR / Sakana / KernelBench-Verified | reward hacking: caching outputs, reading grader state, shape-specific bypasses | fresh random systems at test time; references are recomputed in a separate process from the original |
| FreshBrew | agents delete or weaken tests | consumers and tests are graded from pristine copies |

## Probed Pallas (Triton backend) facts on the RTX 5090, JAX 0.11.2

- `jax[cuda12]` fails with "Unable to load cuSPARSE" when `LD_LIBRARY_PATH` points at the system CUDA, so the image must clear it.
- `erfc` and `round` have no Pallas Triton lowering. A port needs its own erfc, accurate enough for the force tolerances.
- `plgpu.atomic_add` works for f32, f64 and i64. With int32 it silently returned zeros in our probe.
- Gather loads (`x_ref[idx]`, `x_ref[idx, 0]`), masks, dynamic `fori_loop` bounds and uint64 mul-hi all work.
