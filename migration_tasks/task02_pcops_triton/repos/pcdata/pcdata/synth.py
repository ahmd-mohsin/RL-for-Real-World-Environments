"""Synthetic labelled point clouds.

Scenes mix four part classes (shell, plane, box, cluster) plus optional
adversarial content: exact duplicate points, integer-lattice regions whose
pairwise distances tie exactly, and isolated outliers that have no
neighbours within typical query radii.
"""
from dataclasses import dataclass
from typing import List

import torch

NUM_CLASSES = 4


@dataclass
class Batch:
    pos: torch.Tensor     # [N, 3] float32, examples concatenated
    feat: torch.Tensor    # [N, F] float32 per-point input features
    label: torch.Tensor   # [N] int64 part labels in [0, NUM_CLASSES)
    ptr: torch.Tensor     # [B + 1] int64 CSR offsets of the examples

    def to(self, device):
        return Batch(self.pos.to(device), self.feat.to(device), self.label.to(device), self.ptr.to(device))

    @property
    def num_examples(self) -> int:
        return self.ptr.numel() - 1

    def dense(self, n: int):
        """[B, n, 3] / [B, F, n] views; requires every example to have n points."""
        B = self.num_examples
        assert self.pos.size(0) == B * n, "dense() needs equal-size examples"
        return self.pos.view(B, n, 3), self.feat.view(B, n, -1).transpose(1, 2).contiguous()


def _shell(g, n, c, r):
    v = torch.randn(n, 3, generator=g)
    return c + r * v / v.norm(dim=1, keepdim=True)


def _plane(g, n, c, s):
    p = torch.rand(n, 3, generator=g) - 0.5
    p[:, 1] = 0.0
    return c + s * p


def _box(g, n, c, s):
    p = torch.rand(n, 3, generator=g) - 0.5
    face = torch.randint(0, 3, (n,), generator=g)
    sign = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    p[torch.arange(n), face] = 0.5 * sign
    return c + s * p


def _cluster(g, n, c, s):
    return c + s * torch.randn(n, 3, generator=g)


def make_scene(n: int, seed: int, duplicates: float = 0.0, outliers: float = 0.0, feat_dim: int = 3):
    """One example with exactly n points."""
    g = torch.Generator().manual_seed(seed)
    if n == 0:
        return torch.zeros(0, 3), torch.zeros(0, feat_dim), torch.zeros(0, dtype=torch.long)
    parts = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    pos = torch.empty(n, 3)
    centres = torch.rand(NUM_CLASSES, 3, generator=g) * 2 - 1
    makers = (_shell, _plane, _box, _cluster)
    scales = (0.35, 0.8, 0.5, 0.12)
    for k in range(NUM_CLASSES):
        m = parts == k
        cnt = int(m.sum())
        if cnt:
            pos[m] = makers[k](g, cnt, centres[k], scales[k])
    if duplicates > 0 and n > 1:
        d = max(1, int(n * duplicates))
        src = torch.randint(0, n, (d,), generator=g)
        dst = torch.randint(0, n, (d,), generator=g)
        pos[dst] = pos[src]
        parts[dst] = parts[src]
    if outliers > 0:
        o = max(1, int(n * outliers))
        dst = torch.randint(0, n, (o,), generator=g)
        pos[dst] = (torch.rand(o, 3, generator=g) * 2 - 1) * 40.0
    feat = torch.cat([pos, torch.randn(n, max(feat_dim - 3, 0), generator=g)], 1)[:, :feat_dim]
    return pos.float(), feat.float(), parts.long()


def lattice_scene(n: int, side: int, seed: int, feat_dim: int = 3):
    """Integer lattice coordinates: every squared distance is an exact small
    integer, so distance ties are exact and tie-breaking is observable."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.randint(0, side, (n, 3), generator=g).float()
    label = (pos.sum(1).long() % NUM_CLASSES)
    feat = torch.cat([pos, torch.randn(n, max(feat_dim - 3, 0), generator=g)], 1)[:, :feat_dim]
    return pos, feat.float(), label


def make_batch(counts: List[int], seed: int = 0, duplicates: float = 0.0, outliers: float = 0.0, feat_dim: int = 3,
               lattice_side: int = 0) -> Batch:
    pos, feat, lab = [], [], []
    for i, n in enumerate(counts):
        if lattice_side > 0:
            p, f, l = lattice_scene(n, lattice_side, seed * 1000 + i, feat_dim)
        else:
            p, f, l = make_scene(n, seed * 1000 + i, duplicates, outliers, feat_dim)
        pos.append(p)
        feat.append(f)
        lab.append(l)
    ptr = torch.zeros(len(counts) + 1, dtype=torch.long)
    ptr[1:] = torch.tensor(counts, dtype=torch.long).cumsum(0)
    return Batch(torch.cat(pos), torch.cat(feat), torch.cat(lab), ptr)
