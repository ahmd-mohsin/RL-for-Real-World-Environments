"""Inference pipeline: voxel downsampling, fixed-size resampling, prediction."""
import torch

import pcops
from pcdata import Batch


def voxel_downsample(batch: Batch, voxel: float) -> Batch:
    """Average the points of every occupied voxel, per example (eager only:
    the number of voxels is data dependent)."""
    pos, feat, label, ptr = batch.pos, batch.feat, batch.label, batch.ptr
    B = ptr.numel() - 1
    example = torch.repeat_interleave(torch.arange(B, device=pos.device), ptr[1:] - ptr[:-1])
    size = torch.full((4,), voxel, device=pos.device, dtype=pos.dtype)
    size[0] = 1.0
    key_space = torch.cat([example[:, None].to(pos.dtype), pos], 1)
    start = torch.cat([torch.zeros(1, device=pos.device, dtype=pos.dtype), pos.min(0).values])
    end = torch.cat([torch.full((1,), float(B), device=pos.device, dtype=pos.dtype), pos.max(0).values])
    cluster = pcops.grid_cluster(key_space, size, start, end)
    uniq, inv = torch.unique(cluster, sorted=True, return_inverse=True)
    n = uniq.numel()
    new_pos = pcops.scatter(pos, inv, dim=0, dim_size=n, reduce="mean")
    new_feat = pcops.scatter(feat, inv, dim=0, dim_size=n, reduce="mean")
    new_label = pcops.scatter(label, inv, dim=0, dim_size=n, reduce="max")
    new_example = pcops.scatter(example, inv, dim=0, dim_size=n, reduce="max")
    counts = torch.bincount(new_example, minlength=B)
    new_ptr = torch.zeros(B + 1, dtype=torch.long, device=pos.device)
    new_ptr[1:] = counts.cumsum(0)
    return Batch(new_pos, new_feat, new_label, new_ptr)


def resample(batch: Batch, n: int) -> Batch:
    """Exactly n points per example: farthest point sampling when an example
    is larger, repetition when it is smaller."""
    out_pos, out_feat, out_lab = [], [], []
    for b in range(batch.num_examples):
        s, e = int(batch.ptr[b]), int(batch.ptr[b + 1])
        cnt = e - s
        if cnt == 0:
            raise ValueError(f"example {b} is empty")
        if cnt >= n:
            ptr = torch.tensor([0, cnt], device=batch.pos.device)
            idx = pcops.fps(batch.pos[s:e], ptr, n / cnt)[:n] + s
        else:
            idx = torch.arange(n, device=batch.pos.device) % cnt + s
        out_pos.append(batch.pos[idx])
        out_feat.append(batch.feat[idx])
        out_lab.append(batch.label[idx])
    ptr = torch.arange(batch.num_examples + 1, device=batch.pos.device) * n
    return Batch(torch.cat(out_pos), torch.cat(out_feat), torch.cat(out_lab), ptr)


@torch.no_grad()
def predict(model, batch: Batch, n: int = 1024, voxel: float = 0.02) -> torch.Tensor:
    """Per-point labels of the resampled, voxelised batch, [B, n]."""
    model.eval()
    b = resample(voxel_downsample(batch, voxel), n)
    xyz, feats = b.dense(n)
    return model(xyz, feats).argmax(1)
