# Task 03: mdlite (CUDA C++ MD engine) → JAX + Pallas (reference port log, RTX 5090)

**Workspace:** three repos.
- **`mdlite`** is the migration target. It is a GPU molecular dynamics engine:
  - about 2,900 lines of CUDA C++ in 17 files, plus a Python package that wraps it through `ctypes`;
  - smooth PME and reaction-field electrostatics; LJ with potential-shift or force-switch; dispersion correction;
  - harmonic bonds and angles, periodic dihedrals, 1-4 pairs;
  - SHAKE/RATTLE; velocity Verlet and BAOAB; CSVR; Berendsen and C-rescale;
  - Philox4x32-10 RNG streams; 64-bit fixed-point accumulation (bitwise deterministic, restart-exact);
  - binary system, trajectory and checkpoint formats, and a CLI.
- **`mdbuild`** (system builders) and **`mdkit`** (analysis and trajectory re-evaluation) are pristine consumers.

## Bugs found in the original while building the task (fixed in `repos/`)

| # | Bug | How it showed | Fix |
|---|---|---|---|
| 1 | BAOAB heat bookkeeping measured the pre-O kinetic energy before projecting velocities onto the constraint manifold | the "conserved" energy drifted by about 900 kJ/mol in 300 steps for rigid water (LJ was fine) | RATTLE the velocities after the first A-step SHAKE, before measuring KE₀ |
| 2 | Nonbonded and exclusion kernels shared virial and energy accumulator slots, but only the pair sums are halved | exclusion corrections would have been halved | separate doubly-counted (`A_*2`) and single-counted slots |
| 3 | The visible LJ / RF unit tests demanded 1e-8 absolute agreement | impossible for any mixed-precision port | relative 1e-6 |

**Validation of the original:**
- every force term matches central finite differences of its energy (worst relative error 4e-5);
- PME converges to a direct Ewald sum (3e-8 at order 8);
- NVE drift is 0.0066 kJ/mol/ns/atom over 40 ps of salt water;
- runs are bitwise deterministic, with bitwise restarts in every integrator and coupling mode.

## Toolchain traps found while porting (JAX 0.11.2, Pallas Triton backend, sm_120)

See `DESIGN_NOTES.md`. In short:
- integer `atomic_add` is silently wrong (int32 produces zeros; int64 with negative operands overwrites);
- float atomics are order dependent;
- `erfc`, `round` and `any` don't lower;
- scalar boolean `where` chains were ignored inside a kernel;
- x64 promotion breaks loop carries;
- unrolled 27-cell loops take more than 10 minutes to compile;
- `jax[cuda12]` needs an empty `LD_LIBRARY_PATH`.

## Reference port design (`solution/solution.patch`)

- **Nonbonded (Pallas):**
  - 32 atoms per program, neighbour chunks of 32, dynamic trip count;
  - float32 pair math on float32 hi/lo coordinates wrapped in float64, so displacements keep float64 accuracy;
  - minimum image with a split box length;
  - hand-written float32 erfc (Chebyshev fit, about 1e-7 relative);
  - every pair contribution converted to int64 fixed point before summing, so results are bitwise reproducible across list rebuilds and restarts;
  - separate force-only and energy+virial variants.
- **Neighbour search (Pallas):**
  - cell sort in XLA;
  - a rolled 27-offset loop with deduplication for boxes with fewer than 3 cells;
  - in-register prefix-sum compaction;
  - capacities grow on overflow, and the whole chunk is replayed from its start state, never truncated.
- **PME:**
  - a Pallas kernel emits each atom's fixed-point B-spline stencil without atomics, and an exact int64 XLA scatter-add builds the grid;
  - float32 FFT; influence function, energy and virial in float64;
  - Pallas gather for the forces;
  - the grid is fixed from the initial box, as in the engine.
- **Constraints (Pallas, float64):**
  - one lane per cluster, Gauss-Seidel in the engine's constraint order;
  - a register-resident variant for clusters of up to 4 atoms (water).
- **Integrators:**
  - jitted multi-step `fori_loop` chunks between output events, with neighbour rebuilds, barostat, thermostat and COM removal inside the loop;
  - failure flags (SHAKE/RATTLE, box too small) are raised on the host as `MdliteError` with the step number.
- **RNG:** Philox4x32-10, Box–Muller, Marsaglia–Tsang gamma and Bussi's sum of noises, bit-identical, including inside jit.
- **Determinism:** no float scatter-adds anywhere. Molecule COMs use a fixed-order gather, and every accumulation is fixed point or a deterministic reduction.

## Results

**Grader validation** (task image, `tests/test.sh`, hardened grader):

| candidate | reward | caught by |
|---|---|---|
| **reference port** | **1.0** | perf 0.267 / 0.328 / 0.38 / 0.276 of CUDA (bars 0.45 / 0.55 / 0.60 / 0.55) |
| original CUDA engine | 0 | static |
| `jax.random` instead of Philox | 0 | 7 dynamics trajectories, 1M-atom BAOAB |
| float32 PME grid (float scatter-add) | 0 | 11 determinism/restart tests, FD consistency, NVE drift |
| single float32 coordinates (no hi/lo) | 0 | far-images forces, image invariance, NVE drift |
| neighbour lists truncated instead of grown | 0 | droplet and dense-corner forces and trajectories |
| A&S 7.1.26 erfc (1.5e-7 absolute) | 0 | PME forces (order 8, salt, far images), C-rescale trajectory, NVE drift |
| loose-SHAKE criterion \|r−d\|/d | (test only) | loose_shake trajectory at step 2 |
| reference port plus a tampering `setup.py` | 1.0 | every write to the originals, venv and grader was blocked (unprivileged install) |

**Hidden suite:** about 117 tests (api 34, forces 35, dynamics 40, physics 4, plus pristine consumer suites), a fresh gate with a secret seed, and 4 perf bars.

**GPT 5.6 sol single attempt:** reward 0. See `runs/gpt-5.6-sol-run1/RESULT.md`. It passed forces and the fresh gate, but failed:
- bitwise determinism, call-splitting invariance and restarts;
- NVE drift;
- three API contracts (hex floats, error text, missing output directory);
- performance, at 1.7–2.9x slower than CUDA against bars of 0.45–0.6.
