"""Hidden benchmark: ms per MD step for realistic systems, with whichever mdlite is importable.

Each case runs warm-up steps first (compilation, neighbour lists), then three timed
blocks; the best block is reported. One JSON line per case: {"case", "ms_per_step"}.
"""
import json
import os
import sys
import tempfile
import time

import mdlite as md

HERE = os.path.dirname(os.path.abspath(__file__))
SYS = os.path.join(HERE, "data", "bench")
CASES = {
    # 33k-atom rigid water, PME, velocity Verlet + CSVR, energies every 100 steps
    "water_pme_csvr": ("water.mdsys", {"cutoff": 0.9, "dt": 0.002, "tcoupl": "csvr", "tau_t": 0.5, "nstenergy": 100, "nstxout": 500}, 400),
    # 256k-atom LJ fluid, Langevin BAOAB
    "lj_256k_baoab": ("lj.mdsys", {"cutoff": 1.0, "coulombtype": "cutoff", "integrator": "baoab", "tau_t": 1.0, "ref_t": 120,
                                   "nstenergy": 100}, 200),
    # 15k-atom salt water, NPT with C-rescale every 10 steps
    "salt_npt_crescale": ("salt.mdsys", {"cutoff": 1.0, "dt": 0.002, "tcoupl": "csvr", "tau_t": 0.5, "pcoupl": "c-rescale",
                                         "tau_p": 2.0, "nstenergy": 100}, 400),
    # 6.4k-atom flexible alkane melt with partial charges, reaction field
    "alkane_rf_flex": ("alkane.mdsys", {"cutoff": 1.0, "coulombtype": "reaction-field", "epsilon_rf": 15, "dt": 0.001,
                                        "nstenergy": 100}, 400),
}


def main():
    sel = sys.argv[1:] or list(CASES)
    for name in sel:
        fname, cfg, n = CASES[name]
        with tempfile.TemporaryDirectory() as out:
            sim = md.Simulation(os.path.join(SYS, fname), cfg, output_dir=out)
            sim.run(200)
            best = float("inf")
            for _ in range(3):
                t0 = time.perf_counter()
                sim.run(n)
                sim.finish()
                best = min(best, (time.perf_counter() - t0) / n)
            sim.close()
        print(json.dumps({"case": name, "ms_per_step": round(best * 1e3, 4)}), flush=True)


if __name__ == "__main__":
    main()
