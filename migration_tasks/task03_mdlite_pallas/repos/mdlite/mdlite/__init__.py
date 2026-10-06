"""mdlite: a small GPU molecular dynamics engine.

PME / reaction-field electrostatics, Lennard-Jones with potential-shift or
force-switch, bonds, angles, proper dihedrals, 1-4 pairs, SHAKE/RATTLE
constraints, velocity Verlet and BAOAB Langevin integrators, CSVR thermostat,
Berendsen and C-rescale barostats. Results are deterministic and restarts from
checkpoints are bitwise continuous.
"""
from .config import config_text, read_config
from .formats import read_checkpoint, read_energy, read_trajectory, write_checkpoint
from .simulation import TERMS, MdliteError, Simulation
from .system import System, load_system, make_types, save_system
from .units import BOLTZ, ONE_4PI_EPS0, PRESFAC

__all__ = ["Simulation", "MdliteError", "TERMS", "System", "load_system", "save_system", "make_types", "config_text",
           "read_config", "read_energy", "read_trajectory", "read_checkpoint", "write_checkpoint", "BOLTZ",
           "ONE_4PI_EPS0", "PRESFAC", "run"]
__version__ = "1.3.0"


def run(system, config, output_dir, checkpoint=None, nsteps=None):
    """Run a simulation to completion like the ``mdlite run`` command; returns the info dict."""
    from .cli import run_simulation
    return run_simulation(system, config, output_dir, checkpoint=checkpoint, nsteps=nsteps)
