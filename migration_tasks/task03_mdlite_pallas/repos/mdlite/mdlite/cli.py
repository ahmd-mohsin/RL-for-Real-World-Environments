"""``mdlite`` command line.

    mdlite run  -s system.mdsys -c run.cfg -o outdir [--checkpoint state.mdcpt] [--nsteps N]
    mdlite info -s system.mdsys -c run.cfg
    mdlite energy -s system.mdsys -c run.cfg      # energies of the initial state as JSON
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import config_text, read_config


def run_simulation(system, config, output_dir, checkpoint=None, nsteps=None):
    from .simulation import Simulation

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = config_text(config)
    sim = Simulation(system, text, output_dir=out)
    try:
        if checkpoint:
            sim.load_checkpoint(checkpoint)
        if nsteps is None:
            cfg = {}
            for line in text.splitlines():
                for c in "#;":
                    line = line.split(c, 1)[0]
                if "=" in line:
                    k, _, v = line.partition("=")
                    cfg[k.strip().lower().replace("-", "_")] = v.strip()
            nsteps = int(cfg.get("nsteps", "0"))
        t0 = time.perf_counter()
        sim.run(nsteps)
        elapsed = time.perf_counter() - t0
        sim.save_checkpoint(out / "final.mdcpt")
        info = sim.info
        info.update({"steps": int(nsteps), "final_step": sim.step, "seconds": elapsed,
                     "steps_per_second": nsteps / elapsed if elapsed > 0 else None})
        (out / "log.json").write_text(json.dumps(info, indent=2) + "\n")
        return info
    finally:
        sim.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mdlite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("-s", "--system", required=True)
    r.add_argument("-c", "--config", required=True)
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--checkpoint")
    r.add_argument("--nsteps", type=int)
    for name in ("info", "energy"):
        p = sub.add_parser(name)
        p.add_argument("-s", "--system", required=True)
        p.add_argument("-c", "--config", required=True)
    a = ap.parse_args(argv)
    from .simulation import MdliteError, Simulation
    try:
        if a.cmd == "run":
            info = run_simulation(a.system, Path(a.config), a.output, checkpoint=a.checkpoint, nsteps=a.nsteps)
            print(json.dumps(info))
        else:
            with Simulation(a.system, Path(a.config)) as sim:
                if a.cmd == "info":
                    print(json.dumps(sim.info))
                else:
                    e = sim.energies()
                    print(json.dumps({k: v for k, v in e.items() if isinstance(v, float)}))
    except MdliteError as e:
        print(f"mdlite: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
