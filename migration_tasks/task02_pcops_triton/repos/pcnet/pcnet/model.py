"""PointNet++-style part segmentation with an edge-convolution branch.

Every geometric step runs through pcops:
  set abstraction    fps -> gather_points -> ball_query -> group_points
  feature propagation three_nn -> three_interpolate
  edge branch        knn -> per-edge MLP -> scatter(max)
  global context     segment_csr(mean) per example -> gather_csr back to points

The model takes equal-size examples ([B, N, 3] / [B, F, N]) and is written to
compile with torch.compile(fullgraph=True): the data-dependent output sizes of
fps and knn are pinned with torch._check.
"""
import math
from typing import List, Sequence

import torch
from torch import nn

import pcops


def _mlp2d(channels: Sequence[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for a, b in zip(channels[:-1], channels[1:]):
        layers += [nn.Conv2d(a, b, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU()]
    return nn.Sequential(*layers)


def _mlp1d(channels: Sequence[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for a, b in zip(channels[:-1], channels[1:]):
        layers += [nn.Conv1d(a, b, 1, bias=False), nn.BatchNorm1d(b), nn.ReLU()]
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    def __init__(self, ratio: float, radius: float, nsample: int, in_ch: int, channels: Sequence[int]):
        super().__init__()
        self.ratio, self.radius, self.nsample = ratio, radius, nsample
        self.mlp = _mlp2d([in_ch + 3, *channels])

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor):
        B, N, _ = xyz.shape
        M = int(math.ceil(N * self.ratio))
        ptr = torch.arange(B + 1, device=xyz.device) * N
        idx = pcops.fps(xyz.reshape(B * N, 3), ptr, self.ratio)
        torch._check(idx.numel() == B * M)
        idx = (idx.view(B, M) - ptr[:-1, None]).to(torch.int32)
        xyz_t = xyz.transpose(1, 2).contiguous()                       # [B, 3, N]
        new_xyz = pcops.gather_points(xyz_t, idx).transpose(1, 2).contiguous()  # [B, M, 3]
        nbr = pcops.ball_query(self.radius, self.nsample, xyz, new_xyz)  # [B, M, S]
        rel = pcops.group_points(xyz_t, nbr) - new_xyz.transpose(1, 2).unsqueeze(-1)
        grouped = torch.cat([rel / self.radius, pcops.group_points(feats, nbr)], 1)
        return new_xyz, self.mlp(grouped).amax(-1)                      # [B, C, M]


class FeaturePropagation(nn.Module):
    def __init__(self, in_ch: int, channels: Sequence[int]):
        super().__init__()
        self.mlp = _mlp1d([in_ch, *channels])

    def forward(self, xyz_dense, xyz_sparse, feats_dense, feats_sparse):
        d2, idx = pcops.three_nn(xyz_dense, xyz_sparse)
        w = 1.0 / (d2 + 1e-8)
        w = w / w.sum(-1, keepdim=True)
        up = pcops.three_interpolate(feats_sparse, idx, w)
        return self.mlp(torch.cat([up, feats_dense], 1) if feats_dense is not None else up)


class EdgeBranch(nn.Module):
    """EdgeConv on the k-NN graph of every example, max-aggregated."""

    def __init__(self, k: int, in_ch: int, out_ch: int):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(nn.Linear(2 * in_ch, out_ch), nn.ReLU(), nn.Linear(out_ch, out_ch))

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor):
        B, N, _ = xyz.shape
        ptr = torch.arange(B + 1, device=xyz.device) * N
        flat = xyz.reshape(B * N, 3)
        edges = pcops.knn(flat, flat, self.k, ptr, ptr)
        torch._check(edges.size(1) == B * N * self.k)
        row, col = edges[0], edges[1]
        x = feats.transpose(1, 2).reshape(B * N, -1)
        msg = self.mlp(torch.cat([x[row], x[col] - x[row]], 1))
        out = pcops.scatter(msg, row, dim=0, dim_size=B * N, reduce="max")
        return out.view(B, N, -1).transpose(1, 2)                       # [B, C, N]


class PartSegNet(nn.Module):
    def __init__(self, in_feat: int = 3, num_classes: int = 4, k: int = 8):
        super().__init__()
        self.sa1 = SetAbstraction(0.25, 0.25, 16, in_feat, [32, 32, 64])
        self.sa2 = SetAbstraction(0.25, 0.5, 16, 64, [64, 64, 128])
        self.fp2 = FeaturePropagation(128 + 64, [128, 64])
        self.fp1 = FeaturePropagation(64 + in_feat, [64, 64])
        self.edge = EdgeBranch(k, in_feat, 32)
        self.head = nn.Sequential(nn.Conv1d(64 + 32 + 64, 64, 1), nn.ReLU(), nn.Conv1d(64, num_classes, 1))

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """xyz [B, N, 3], feats [B, F, N] -> logits [B, num_classes, N]."""
        B, N, _ = xyz.shape
        xyz1, f1 = self.sa1(xyz, feats)
        xyz2, f2 = self.sa2(xyz1, f1)
        u1 = self.fp2(xyz1, xyz2, f1, f2)
        u0 = self.fp1(xyz, xyz1, feats, u1)
        e = self.edge(xyz, feats)
        # global context: per-example mean of u0, broadcast back to the points
        flat = u0.transpose(1, 2).reshape(B * N, -1)
        ptr = torch.arange(B + 1, device=xyz.device) * N
        g = pcops.segment_csr(flat, ptr, "mean")                     # [B, C]
        g = pcops.gather_csr(g, ptr)                                   # [B * N, C]
        torch._check(g.size(0) == B * N)
        g = g.view(B, N, -1).transpose(1, 2)
        return self.head(torch.cat([u0, e, g], 1))
