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
- **Integer atomics are unreliable, and silently so.**
  - int32 `atomic_add` produces zeros.
  - int64 and uint64 `atomic_add` with a *negative* operand overwrites the cell instead of adding (seen as a lost charge-grid contribution).
  - Float atomics are correct but order dependent, so a PME spread built on atomics is either wrong (ints) or nondeterministic (floats).
  - The reference port emits per-atom stencil contributions from Pallas and sums them with an exact int64 XLA scatter-add.
- `jnp.any` (`reduce_or`) has no lowering; use `max` of an int32 cast.
- A scalar boolean `jnp.where` / `&` chain inside a Pallas kernel was ignored. The duplicate-offset filter for boxes with fewer than 3 cells did nothing, so neighbours were counted 3 times. The fix is to pass a precomputed table.
- With x64 enabled, `jnp.sum` of int32 promotes to int64 and breaks the `fori_loop` carry dtype. Pallas `mod` asserts on mixed integer widths.
- 27 statically unrolled offset blocks, each with a dynamic inner loop, take more than 10 minutes to compile in Triton; roll the loop.
- Gather loads (`x_ref[idx]`), scattered stores with masks, in-kernel `jnp.cumsum` (stream compaction), dynamic `fori_loop` / `while_loop` bounds and uint64 mul-hi all work.

## Hardness levers in the task

- **fp64 everywhere in the original:** consumer Blackwell runs fp64 at 1/64 rate, so the performance bars force a real mixed-precision design. That means float32 pair math on hi/lo coordinates (minimum image applied with a split box length) and a hand-written `erfc`. Accuracy tests then catch sloppy float32: atoms thousands of boxes away, 1M atoms, energy sums.
- **Fixed-point accumulation:** the original accumulates in 64-bit fixed point, so it is bitwise reproducible and restarts are bitwise continuous. A port must find another order-independent accumulation. Float scatter-add is nondeterministic, and integer atomics are broken in Pallas (above).
- **Exact RNG semantics:** the port must reproduce, bit for bit:
  - Philox4x32-10 with the documented key/counter layout;
  - 53-bit uniforms and Box–Muller;
  - three deviates per atom from two blocks;
  - Marsaglia–Tsang gamma deviates and Bussi's sum of noises, including the draw order and the "no uniform when t <= 0" rule.

  Trajectories are compared from step 1.
- **Physics semantics hidden in the code:**
  - force switch;
  - dispersion-correction virial (2E per diagonal component);
  - PME exclusion correction at any distance, against reaction field only inside the cut-off;
  - RF self term;
  - net-charge term and its virial;
  - odd-order B-spline moduli fix;
  - grid size rule (even, 2·3·5·7-smooth, at least 2p);
  - Ewald coefficient bisection;
  - `constraints = no` turning constraints into k = 4e5 bonds;
  - DOF counting;
  - BAOAB heat bookkeeping;
  - C-rescale noise and kT/V term;
  - molecule-COM scaling;
  - PME grid fixed from the initial box;
  - step-0 constraint virial;
  - checkpoint layout including the constraint virial.
- **Static shapes versus dynamic data:**
  - neighbour capacity overflow in dense regions (a JAX scatter silently drops entries);
  - cell counts that change under NPT;
  - error reporting from inside jitted loops (SHAKE failure, box too small).
