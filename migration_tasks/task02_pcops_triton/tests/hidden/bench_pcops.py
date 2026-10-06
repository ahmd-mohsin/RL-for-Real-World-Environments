"""Hidden benchmark: pcops operators and a pcnet training step at realistic sizes.

Runs against whichever pcops is importable (the original CUDA extension or a
port). Each case is the median of R timed calls after warm-up, bracketed by
torch.cuda.synchronize(). One JSON line per case: {"case", "median_ms"}.

    python bench_pcops.py [--reps R] [--quick]
"""
import argparse
import json
import math
import time

import torch

import pcops


def median_ms(fn, reps):
    fn()
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def emit(name, ms):
    print(json.dumps({"case": name, "median_ms": round(ms, 4)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    q = 4 if a.quick else 1
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)

    # --- sampling / neighbourhoods on a batch of 16 clouds x 16384 points
    B, N = 16, 16384 // q
    pos = torch.rand(B * N, 3, device=dev, generator=g)
    ptr = torch.arange(B + 1, device=dev) * N
    emit("fps_16x16k_r0.25", median_ms(lambda: pcops.fps(pos, ptr, 0.25), a.reps))
    ptr4 = torch.arange(5, device=dev) * (B * N // 4)   # the same points as 4 clouds of 65536
    emit("fps_4x65k_r0.0625", median_ms(lambda: pcops.fps(pos, ptr4, 0.0625), a.reps))
    xs = pos
    ys = pos[:: 4].contiguous()
    ptr_y = torch.arange(B + 1, device=dev) * (N // 4)
    emit("knn_k16", median_ms(lambda: pcops.knn(xs, ys, 16, ptr, ptr_y), a.reps))
    emit("knn_k64", median_ms(lambda: pcops.knn(xs, ys[: ys.size(0) // 4], 64, ptr, ptr_y // 4), a.reps))
    emit("radius_r0.05_m64", median_ms(lambda: pcops.radius(xs, ys, 0.05, ptr, ptr_y, 64), a.reps))

    # --- PointNet++ set abstraction / feature propagation, [B, N, 3]
    xyz = pos.view(B, N, 3)
    new_xyz = xyz[:, :: 4].contiguous()
    emit("ball_query_r0.1_s32", median_ms(lambda: pcops.ball_query(0.1, 32, xyz, new_xyz), a.reps))
    idx = pcops.ball_query(0.1, 32, xyz, new_xyz)
    feats = torch.randn(B, 64, N, device=dev, generator=g)
    emit("group_points_c64_s32", median_ms(lambda: pcops.group_points(feats, idx), a.reps))
    fg = feats.clone().requires_grad_()
    out = pcops.group_points(fg, idx)
    go = torch.randn_like(out)
    emit("group_points_backward", median_ms(lambda: torch.autograd.grad(pcops.group_points(fg, idx), fg, go), a.reps))
    emit("three_nn", median_ms(lambda: pcops.three_nn(xyz, new_xyz), a.reps))
    d, i = pcops.three_nn(xyz, new_xyz)
    w = (1.0 / (d + 1e-8))
    w = w / w.sum(-1, keepdim=True)
    sfeat = torch.randn(B, 128, new_xyz.size(1), device=dev, generator=g)
    emit("three_interpolate_c128", median_ms(lambda: pcops.three_interpolate(sfeat, i, w), a.reps))
    sf = sfeat.clone().requires_grad_()
    gout = torch.randn(B, 128, N, device=dev, generator=g)
    emit("three_interpolate_backward", median_ms(lambda: torch.autograd.grad(pcops.three_interpolate(sf, i, w), sf, gout), a.reps))

    # --- reductions: 4M rows x 32 features
    E, C = 4_000_000 // q, 32
    src = torch.randn(E, C, device=dev, generator=g)
    seg = torch.randint(1, 64, (E // 32,), device=dev, generator=g)
    indptr = torch.cat([torch.zeros(1, device=dev, dtype=torch.long), seg.cumsum(0)])
    indptr = indptr[indptr <= E]
    indptr = torch.cat([indptr, torch.full((1,), E, device=dev, dtype=torch.long)]) if int(indptr[-1]) != E else indptr
    for r in ("sum", "max"):
        emit(f"segment_csr_{r}", median_ms(lambda r=r: pcops.segment_csr(src, indptr, r), a.reps))
    emit("gather_csr", median_ms(lambda: pcops.gather_csr(pcops.segment_csr(src, indptr, "sum"), indptr), a.reps))
    index = torch.randint(0, E // 16, (E,), device=dev, generator=g)
    hot = torch.randint(0, 64, (E,), device=dev, generator=g)
    for r in ("sum", "mean", "max"):
        emit(f"scatter_{r}", median_ms(lambda r=r: pcops.scatter(src, index, 0, E // 16, r), a.reps))
    emit("scatter_sum_contended", median_ms(lambda: pcops.scatter(src, hot, 0, 64, "sum"), a.reps))
    gpos = torch.rand(E, 3, device=dev, generator=g) * 10
    size = torch.full((3,), 0.05, device=dev)
    emit("grid_cluster", median_ms(lambda: pcops.grid_cluster(gpos, size), a.reps))

    # --- end to end: pcnet training step and inference
    try:
        from pcdata import make_batch
        from pcnet import PartSegNet
    except ImportError:
        return
    torch.manual_seed(0)
    model = PartSegNet(in_feat=3, num_classes=4).to(dev)
    n = 4096 // q
    b = make_batch([n] * 8, seed=0, duplicates=0.02).to(dev)
    xyz8, f8 = b.dense(n)
    lab = b.label.view(8, n)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(xyz8, f8), lab)
        loss.backward()
        opt.step()

    emit("pcnet_train_step", median_ms(step, max(5, a.reps // 2)))
    model.eval()
    with torch.no_grad():
        emit("pcnet_inference", median_ms(lambda: model(xyz8, f8), a.reps))


if __name__ == "__main__":
    main()
