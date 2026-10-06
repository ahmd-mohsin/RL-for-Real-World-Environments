# GPT 5.6 sol single-attempt run: task03_mdlite_pallas

**Reward: 0.**

- **Harness:** mini-swe-agent 2.4.6, bash only, 350-step / $80 cap (`run_agent.sh`).
- **Sandbox:** a fresh `--network none` container from `task03-mdlite-pallas:v1` on the lab RTX 5090. The hidden tests exist only in the grading container.
- **Usage:** GPT submitted on its own after **81 API calls, $2.96**. The patch deletes `src/` and CMake and adds about 890 lines of Python in `_kernels.py`, `_physics.py`, `_rng.py`, `_topology.py`, `_config.py` and `simulation.py`.
- **Grading:** the hardened grader (unprivileged candidate, secret-seed fresh gate).

**What GPT got right (strong attempt):**
- Pallas kernels for neighbour search, nonbonded pairs, PME spreading and interpolation, and SHAKE/RATTLE.
- int64 fixed-point accumulation, and it noticed that "Triton's int64 atomics are unreliable on sm_120", using an exact scatter instead.
- A hand-rolled Philox.
- It registered libdevice `erfc` into JAX's private Pallas-Triton lowering table.

| gate | result |
|---|---|
| static, install, app | pass |
| forces (35 golden cases) | pass |
| fresh (secret-seed randomized cases vs the CUDA original) | pass |
| **api** | **3 failed:** `epsilon_rf = 0x1.4p+1` rejected (the engine's `strtod` accepts hex floats); SHAKE failure reported as "constraint solver did not converge" (the contract is the SHAKE/RATTLE error); a missing output directory does not raise `MdliteError` |
| **dynamics** | **7 failed:** results depend on how a run is split into `run()` calls (positions differ by about 1e-10, csvr / baoab_crescale / alkane_cons_baoab / berendsen); restarts not bitwise continuous (baoab_crescale, alkane_cons_baoab, crescale) |
| **physics** | **NVE drift 0.0047 kJ/mol/ns/atom** (limit 0.003; the original drifts 0.00027) |
| **perf** (measured separately; the grader skips it after failures) | 1.7–2.9x *slower* than the CUDA engine, against bars of 0.45–0.60 |

| case | CUDA ms | GPT ms | ratio | bar | reference port |
|---|---|---|---|---|---|
| water PME + CSVR (33k) | 7.69 | 12.7 | **1.66** | 0.45 | 0.27 |
| LJ 256k BAOAB | 4.28 | 8.77 | **2.05** | 0.55 | 0.33 |
| salt NPT C-rescale (15k) | 7.14 | 18.0 | **2.52** | 0.60 | 0.38 |
| alkane RF (6.4k) | 0.72 | 2.07 | **2.88** | 0.55 | 0.24 |

GPT's two benchmark runs also ended in different water energies (−15.3386 vs −15.3377 kJ/mol per atom), which shows the port is nondeterministic.
