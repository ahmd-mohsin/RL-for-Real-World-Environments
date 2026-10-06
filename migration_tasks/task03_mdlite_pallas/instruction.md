# Port the mdlite molecular dynamics engine from CUDA C++ to JAX + Pallas

`/workspace` contains three repositories, each its own git repo:

| repo | what it is |
|---|---|
| `mdlite` | A GPU molecular dynamics engine. The engine is CUDA C++ in `src/`, built into `libmdlite.so` by CMake (`setup.py` drives the build), with a Python package `mdlite/` that wraps it through `ctypes`. Its physics: smooth PME and reaction-field electrostatics, Lennard-Jones with potential-shift or force-switch, dispersion correction, bonds, angles, proper dihedrals, 1-4 pairs, SHAKE/RATTLE constraints, velocity Verlet and BAOAB Langevin integrators, the CSVR thermostat, and Berendsen and C-rescale barostats. It also handles energy, trajectory and checkpoint files and provides a CLI (`mdlite run`). |
| `mdbuild` | Builds systems for mdlite (water, salt water, alkane melts, droplets, crystals, LJ fluids). |
| `mdkit` | Analysis tools built on mdlite: RDF, MSD, drift, and re-evaluating energies along trajectories. |

The team wants to drop the CUDA toolchain. **Port `mdlite` to JAX with Pallas kernels.**
- The package must become pure Python (JAX, Pallas, NumPy).
- It must keep the same Python API, CLI, file formats and physics.
- It must run substantially faster than the CUDA engine on the target GPU.

**The machine:** an NVIDIA RTX 5090 (Blackwell, sm_120), CUDA 12.8, and JAX 0.11 with the CUDA 12 plugin in `/opt/venv`. Pallas uses the Triton backend. The original engine is built and installed from `/workspace/mdlite`, so you can compare against it at any time. Keep `LD_LIBRARY_PATH` empty for JAX.

## Requirements

1. **No native code.**
   - `mdlite` must contain no C/C++/CUDA sources and no native build (CMake, `nvcc`, extensions).
   - It must not load shared libraries (`ctypes`, `cffi`).
   - It must not use other GPU kernel libraries (`cupy`, `numba`, `pycuda`, `torch`, raw `triton`, …).
   - It must not use host callbacks (`pure_callback`, `io_callback`, FFI custom calls).
   - `pip install ./mdlite` must work on a machine without a CUDA compiler.
2. **Pallas for the hot paths.** These must run as Pallas kernels (`pallas_call`):
   - the nonbonded pair interactions;
   - the neighbour search;
   - the PME charge spreading and force interpolation;
   - the constraint solvers.

   FFTs and small bookkeeping may use XLA.
3. **Same interface.**
   - Everything exported by `mdlite/__init__.py` keeps its signature and behaviour: `Simulation` (all methods and properties), `TERMS`, `MdliteError`, the system and file-format helpers, and `mdlite.cli`.
   - The `info` values must match: `natoms`, `dof`, `constraints`, `constraint_clusters`, `exclusion_pairs`, `rlist`, `ewald_beta`, `pme_grid`, `k_rf` and `c_rf`.
   - Invalid configurations, systems, checkpoints, arguments, and runtime failures (SHAKE/RATTLE non-convergence, a box smaller than twice the cut-off plus skin) raise `MdliteError` wherever the engine reports an error.
   - The run configuration is parsed exactly as the engine parses it.
4. **Same results.** Mixed precision is allowed, within these tolerances:
   - **Forces:** RMS error at most 4e-5 relative to the RMS force (or 5e-3 kJ/mol/nm absolute).
   - **Energy terms:** each within 2e-5 relative, plus 2e-6 kJ/mol per atom.
   - **Pressure:** within 0.5 bar.
   - **Derived values:** the Ewald coefficient, PME grid sizes, reaction-field constants and degrees of freedom match exactly.
   - **Trajectories:** follow the engine's to rounding in the first steps, for every integrator, thermostat and barostat. That includes every random number: Philox4x32-10 streams, Box–Muller, Marsaglia–Tsang gamma deviates.
   - **Configurations covered:** these hold for any system the engine accepts. That includes atoms thousands of boxes away from the origin, strongly inhomogeneous systems, a single atom, and 10⁶ atoms.
5. **Same guarantees.**
   - Results are bitwise deterministic.
   - They don't depend on how a run is split into `run()` calls, or on which outputs are written.
   - A restart from a checkpoint continues bitwise identically, including the energy file rows.
   - Output files are byte-compatible with the engine's formats, and checkpoints load in both directions.
6. **The consumers keep working unchanged.**
   - `mdbuild`, `mdkit` and mdlite's own tests are graded from their original sources; your changes to them are discarded.
   - Long runs must reproduce the engine's physics: NVE energy conservation, NPT density, and constraint satisfaction.
7. **Performance.**
   - The grader times each case below against the original engine on the same GPU.
   - Timing is ms per MD step, best of three blocks, after 200 warm-up steps, with the outputs given.
   - Your time may not exceed the stated fraction of the engine's.

| case | max time / CUDA engine |
|---|---|
| 33k-atom SPC/E water, PME, velocity Verlet + CSVR, SHAKE/RATTLE | 0.45 |
| 256k-atom Lennard-Jones fluid, BAOAB Langevin | 0.55 |
| 15k-atom salt water, NPT (C-rescale + CSVR), PME | 0.60 |
| 6.4k-atom flexible alkane melt with partial charges, reaction field | 0.55 |

## Grading

The grader takes your diff of `/workspace/mdlite` against its `base` tag and installs it without a CUDA compiler. Hidden tests then check, against results recorded with the CUDA engine:

- the API and error contracts;
- forces and energies for many systems and settings;
- trajectories and output files of every integrator, thermostat and barostat;
- determinism and restarts;
- long-run physics and large systems;
- that Pallas kernels run.

They also run the original `mdbuild`, `mdkit` and `mdlite` test suites and the benchmark above. All of it must pass.
