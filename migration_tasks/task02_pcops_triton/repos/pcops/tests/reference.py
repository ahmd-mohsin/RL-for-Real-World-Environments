"""Straightforward CPU implementations of the pcops semantics.

Distances are computed in float64 here. Tests compare exactly on inputs
whose distances are exactly representable (integer coordinates), where the
GPU's tie-breaking is fully determined, and with tolerances / properties on
random inputs.
"""
import math

import torch


def ptr_from_counts(counts, device=None):
    p = torch.zeros(len(counts) + 1, dtype=torch.long)
    p[1:] = torch.tensor(counts, dtype=torch.long).cumsum(0)
    return p.to(device) if device is not None else p


# ---------------------------------------------------------------- sampling

def fps_counts(ptr, ratio):
    deg = (ptr[1:] - ptr[:-1]).tolist()
    return [int(math.ceil(d * ratio)) for d in deg]


def check_fps(src, ptr, ratio, out, rel_tol=1e-5):
    """Farthest point sampling property: sample m of every example maximises
    the distance (capped at sqrt(5e4)) to the samples 0..m-1 of that example.
    Ties may be broken arbitrarily. Returns a list of problems."""
    problems = []
    src = src.double().view(src.size(0), -1).cpu()
    out = out.cpu()
    counts = fps_counts(ptr.cpu(), ratio)
    if out.numel() != sum(counts):
        return [f"expected {sum(counts)} samples, got {out.numel()}"]
    o = 0
    for b, cnt in enumerate(counts):
        s, e = int(ptr[b]), int(ptr[b + 1])
        if cnt == 0:
            continue
        sel = out[o : o + cnt].tolist()
        o += cnt
        if sel[0] != s:
            problems.append(f"example {b}: first sample {sel[0]} != {s}")
        pts = src[s:e]
        dist = torch.full((e - s,), 5e4, dtype=torch.float64)
        for m in range(1, cnt):
            prev = sel[m - 1] - s
            dist = torch.minimum(dist, ((pts - pts[prev]) ** 2).sum(1))
            j = sel[m] - s
            if not (0 <= j < e - s):
                problems.append(f"example {b}: sample {m} out of range ({sel[m]})")
                break
            best = float(dist.max())
            if dist[j] < best * (1 - rel_tol) - 1e-12:
                problems.append(f"example {b}: sample {m} = {sel[m]} has distance {float(dist[j])} < max {best}")
                break
    return problems


def knn(x, y, k, ptr_x=None, ptr_y=None):
    """[2, E] edges, row = y index, col = x index; k nearest by squared
    distance, ties broken by the smaller x index; nearest first."""
    x, y = x.double().cpu(), y.double().cpu()
    ptr_x = ptr_x.cpu() if ptr_x is not None else torch.tensor([0, x.size(0)])
    ptr_y = ptr_y.cpu() if ptr_y is not None else torch.tensor([0, y.size(0)])
    rows, cols = [], []
    for b in range(ptr_y.numel() - 1):
        xs, xe = int(ptr_x[b]), int(ptr_x[b + 1])
        for i in range(int(ptr_y[b]), int(ptr_y[b + 1])):
            d = ((x[xs:xe] - y[i]) ** 2).sum(1)
            order = sorted(range(xe - xs), key=lambda j: (float(d[j]), j))[:k]
            for j in order:
                rows.append(i)
                cols.append(xs + j)
    return torch.tensor([rows, cols], dtype=torch.long).view(2, -1)


def radius(x, y, r, ptr_x=None, ptr_y=None, max_num_neighbors=32, ignore_same_index=False):
    """[2, E]: for every y, the first max_num_neighbors x (index order) with
    squared distance < r^2."""
    x, y = x.double().cpu(), y.double().cpu()
    ptr_x = ptr_x.cpu() if ptr_x is not None else torch.tensor([0, x.size(0)])
    ptr_y = ptr_y.cpu() if ptr_y is not None else torch.tensor([0, y.size(0)])
    rows, cols = [], []
    for b in range(ptr_y.numel() - 1):
        xs, xe = int(ptr_x[b]), int(ptr_x[b + 1])
        for i in range(int(ptr_y[b]), int(ptr_y[b + 1])):
            d = ((x[xs:xe] - y[i]) ** 2).sum(1)
            c = 0
            for j in range(xe - xs):
                if c >= max_num_neighbors:
                    break
                if d[j] < r * r and not (ignore_same_index and xs + j == i):
                    rows.append(i)
                    cols.append(xs + j)
                    c += 1
    return torch.tensor([rows, cols], dtype=torch.long).view(2, -1)


def grid(pos, size, start=None, end=None):
    pos = pos.double().cpu().view(pos.size(0), -1)
    size = size.double().cpu()
    start = pos.min(0).values if start is None else start.double().cpu()
    end = pos.max(0).values if end is None else end.double().cpu()
    c = torch.zeros(pos.size(0), dtype=torch.long)
    k = 1
    for d in range(pos.size(1)):
        c += torch.trunc((pos[:, d] - start[d]) / size[d]).long() * k
        k *= int((end[d] - start[d]) / size[d]) + 1
    return c


def ball_query(radius_, nsample, xyz, new_xyz):
    xyz, new_xyz = xyz.double().cpu(), new_xyz.double().cpu()
    B, M = new_xyz.shape[:2]
    out = torch.zeros(B, M, nsample, dtype=torch.int32)
    r2 = float(radius_) ** 2
    for b in range(B):
        d = ((xyz[b][None, :, :] - new_xyz[b][:, None, :]) ** 2).sum(-1)  # [M, N]
        for i in range(M):
            hits = torch.nonzero(d[i] < r2).flatten()[:nsample].tolist()
            if hits:
                out[b, i, :] = hits[0]
                out[b, i, : len(hits)] = torch.tensor(hits, dtype=torch.int32)
    return out


def three_nn(unknown, known):
    u, k = unknown.double().cpu(), known.double().cpu()
    B, n = u.shape[:2]
    m = k.size(1)
    dist = torch.full((B, n, 3), float("inf"), dtype=torch.float64)
    idx = torch.zeros(B, n, 3, dtype=torch.int32)
    for b in range(B):
        d = ((u[b][:, None, :] - k[b][None, :, :]) ** 2).sum(-1)  # [n, m]
        for i in range(n):
            order = sorted(range(m), key=lambda j: (float(d[i, j]), j))[:3]
            for s, j in enumerate(order):
                dist[b, i, s] = d[i, j]
                idx[b, i, s] = j
    return dist, idx


# ---------------------------------------------------------------- features

def group_points(features, idx):
    f = features.cpu()
    i = idx.long().cpu()
    B, C, _ = f.shape
    _, M, S = i.shape
    return torch.gather(f, 2, i.view(B, 1, M * S).expand(B, C, M * S)).view(B, C, M, S)


def gather_points(features, idx):
    f = features.cpu()
    i = idx.long().cpu()
    B, C, _ = f.shape
    return torch.gather(f, 2, i.view(B, 1, -1).expand(B, C, i.size(1)))


def three_interpolate(features, idx, weight):
    f = features.double().cpu()
    i = idx.long().cpu()
    w = weight.double().cpu()
    B, C, _ = f.shape
    n = i.size(1)
    g = torch.gather(f, 2, i.view(B, 1, n * 3).expand(B, C, n * 3)).view(B, C, n, 3)
    return (g * w[:, None, :, :]).sum(-1)


# ---------------------------------------------------------------- reductions

def segment_csr(src, indptr, reduce):
    src = src.cpu()
    indptr = indptr.cpu()
    dim = indptr.dim() - 1
    lead = indptr.shape[:-1]
    out_shape = list(src.shape)
    out_shape[dim] = indptr.size(-1) - 1
    out = torch.zeros(out_shape, dtype=src.dtype)
    arg = torch.full(out_shape, src.size(dim), dtype=torch.long) if reduce in ("min", "max") else None
    for li in torch.cartesian_prod(*[torch.arange(s) for s in lead]).view(-1, len(lead)) if len(lead) else [()]:
        li = tuple(int(v) for v in li) if len(lead) else ()
        p = indptr[li] if len(lead) else indptr
        for s in range(p.numel() - 1):
            a, e = int(p[s]), int(p[s + 1])
            if e <= a:
                continue
            seg = src[li + (slice(a, e),)]
            if reduce == "sum":
                v = seg.sum(0)
            elif reduce == "mean":
                # integer means truncate towards zero (C++ division in the CUDA kernel)
                v = seg.sum(0) / (e - a) if src.is_floating_point() else torch.div(seg.sum(0), e - a, rounding_mode="trunc")
            elif reduce == "max":
                v, j = seg.max(0)
                arg[li + (s,)] = j + a
            else:
                v, j = seg.min(0)
                arg[li + (s,)] = j + a
            out[li + (s,)] = v
    return out, arg


def scatter(src, index, dim, dim_size, reduce):
    src = src.cpu()
    dim = dim % src.dim()
    if index.dim() == 1:
        view = [1] * src.dim()
        view[dim] = -1
        index = index.view(view)
    index = index.expand_as(src).cpu()
    shape = list(src.shape)
    shape[dim] = dim_size
    if reduce in ("sum", "mean"):
        out = torch.zeros(shape, dtype=src.dtype).scatter_add_(dim, index, src)
        if reduce == "mean":
            cnt = torch.zeros(shape, dtype=src.dtype).scatter_add_(dim, index, torch.ones_like(src)).clamp_(min=1)
            out = out / cnt if src.is_floating_point() else torch.div(out, cnt, rounding_mode="floor")
        return out
    red = "amax" if reduce == "max" else "amin"
    out = torch.zeros(shape, dtype=src.dtype).scatter_reduce_(dim, index, src, red, include_self=False)
    return out
