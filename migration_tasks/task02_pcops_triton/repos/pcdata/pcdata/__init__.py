"""pcdata - deterministic synthetic point clouds for pcops / pcnet tests and benchmarks."""
from .synth import Batch, make_batch, make_scene, lattice_scene

__all__ = ["Batch", "make_batch", "make_scene", "lattice_scene"]
