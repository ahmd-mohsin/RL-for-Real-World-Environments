"""pcops - GPU operators for point clouds and sparse reductions.

Sampling / neighbourhoods: fps, knn, radius, ball_query, three_nn, grid_cluster
Feature movement:          group_points, gather_points, three_interpolate
Sparse reductions:         segment_csr, gather_csr, scatter

All operators run on CUDA tensors and are registered with the PyTorch
dispatcher as torch.ops.pcops.*, so they work under autograd and
torch.compile (see pcops/_registration.py).
"""
from . import _C  # noqa: F401  (registers torch.ops.pcops.* kernels)
from . import _registration  # noqa: F401  (fake impls + autograd)
from .ops import (
    ball_query,
    fps,
    gather_csr,
    gather_points,
    grid_cluster,
    group_points,
    knn,
    radius,
    scatter,
    segment_csr,
    three_interpolate,
    three_nn,
)

__version__ = "0.4.0"
__all__ = [
    "ball_query", "fps", "gather_csr", "gather_points", "grid_cluster", "group_points", "knn", "radius",
    "scatter", "segment_csr", "three_interpolate", "three_nn",
]
