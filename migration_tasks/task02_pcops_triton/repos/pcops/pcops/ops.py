"""Public API of pcops. Thin wrappers over torch.ops.pcops.*."""
from typing import Optional

import torch


def _ptr(n: int, ptr: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    if ptr is None:
        return torch.tensor([0, n], dtype=torch.long, device=like.device)
    return ptr


def fps(src: torch.Tensor, ptr: Optional[torch.Tensor] = None, ratio: float = 0.5, random_start: bool = False) -> torch.Tensor:
    """Farthest point sampling per example. `ptr` is the CSR offset vector of
    the examples in `src` ([N, D]); returns global indices of the samples,
    ceil(ratio * n_b) per example, starting from each example's first point
    (or a random one)."""
    return torch.ops.pcops.fps(src, _ptr(src.size(0), ptr, src), float(ratio), bool(random_start))


def knn(x: torch.Tensor, y: torch.Tensor, k: int, ptr_x: Optional[torch.Tensor] = None, ptr_y: Optional[torch.Tensor] = None,
        cosine: bool = False) -> torch.Tensor:
    """For every row of `y`, the `k` nearest rows of `x` within the same
    example. Returns a [2, E] edge index (row = index into y, col = index into x),
    nearest first."""
    return torch.ops.pcops.knn(x, y, ptr_x, ptr_y, int(k), bool(cosine))


def radius(x: torch.Tensor, y: torch.Tensor, r: float, ptr_x: Optional[torch.Tensor] = None, ptr_y: Optional[torch.Tensor] = None,
           max_num_neighbors: int = 32, ignore_same_index: bool = False) -> torch.Tensor:
    """For every row of `y`, up to `max_num_neighbors` rows of `x` within
    distance `r` (same example), in increasing index order. [2, E] edge index."""
    return torch.ops.pcops.radius(x, y, ptr_x, ptr_y, float(r), int(max_num_neighbors), bool(ignore_same_index))


def grid_cluster(pos: torch.Tensor, size: torch.Tensor, start: Optional[torch.Tensor] = None, end: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Voxel id of every point for a regular grid of cell `size`."""
    return torch.ops.pcops.grid(pos, size, start, end)


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    """Reduce contiguous segments of `src` along dim `indptr.dim() - 1`.
    reduce in {sum, mean, min, max}; empty segments give 0."""
    return torch.ops.pcops.segment_csr(src, indptr, reduce)[0]


def gather_csr(src: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
    """Inverse of segment_csr: repeat every segment value over its segment."""
    return torch.ops.pcops.gather_csr(src, indptr)


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None, reduce: str = "sum") -> torch.Tensor:
    """Reduce all values of `src` with the same `index` along `dim`.
    `index` is broadcast to the shape of `src`. reduce in {sum, mean, min, max};
    empty outputs give 0."""
    dim = dim % src.dim()
    if index.dim() == 1:
        view = [1] * src.dim()
        view[dim] = -1
        index = index.view(view)
    index = index.expand_as(src).contiguous()
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    return torch.ops.pcops.scatter(src, index, dim, int(dim_size), reduce)[0]


def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """For every centre in `new_xyz` [B, M, 3], the indices of the first
    `nsample` points of `xyz` [B, N, 3] closer than `radius`, padded with the
    first one found; int32 [B, M, nsample]."""
    return torch.ops.pcops.ball_query(new_xyz, xyz, float(radius), int(nsample))


def group_points(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features [B, C, N], idx [B, M, S] -> [B, C, M, S] (differentiable in features)."""
    return torch.ops.pcops.group_points(features, idx)


def gather_points(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features [B, C, N], idx [B, M] -> [B, C, M] (differentiable in features)."""
    return torch.ops.pcops.gather_points(features, idx)


def three_nn(unknown: torch.Tensor, known: torch.Tensor):
    """Squared distances and indices of the three nearest `known` points of
    every `unknown` point: ([B, n, 3], int32 [B, n, 3])."""
    return torch.ops.pcops.three_nn(unknown, known)


def three_interpolate(features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """features [B, C, m], idx/weight [B, n, 3] -> [B, C, n] (differentiable in features)."""
    return torch.ops.pcops.three_interpolate(features, idx, weight)
