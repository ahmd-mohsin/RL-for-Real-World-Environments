"""Hidden hard tests: operator registration contract, determinism, streams,
threads, dynamic-shape compilation, error behaviour, skewed / large inputs,
and proof that every operator runs Triton kernels."""
import os
import threading

import pytest
import torch

import pcops
import reference as ref

DEV = os.environ.get("PCOPS_TEST_DEVICE", "cuda")
GPU = DEV == "cuda"


def rnd(*shape, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g, dtype=torch.float64).to(dtype).to(DEV)


def lattice(n, side, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, side, (n, 3), generator=g).float().to(DEV)


def all_op_calls():
    """name -> zero-argument callable exercising one public operator."""
    pos, ptr = rnd(3000, 3, seed=1), torch.tensor([0, 1000, 1000, 3000], device=DEV)
    xyz = rnd(2, 2048, 3, seed=2)
    new = xyz[:, ::8].contiguous()
    feats = rnd(2, 16, 2048, seed=3)
    gi = pcops.ball_query(0.2, 16, xyz, new)
    d, ti = pcops.three_nn(xyz, new)
    w = 1.0 / (d + 1e-8)
    w = w / w.sum(-1, keepdim=True)
    sfeat = rnd(2, 16, new.size(1), seed=4)
    src = rnd(5000, 8, seed=5)
    indptr = torch.tensor([0, 7, 7, 1000, 4999, 5000], device=DEV)
    index = torch.randint(0, 300, (5000,), device=DEV)
    return {
        "fps": lambda: pcops.fps(pos, ptr, 0.2),
        "knn": lambda: pcops.knn(pos, pos[::5].contiguous(), 9, ptr, ptr // 5),
        "radius": lambda: pcops.radius(pos, pos[::5].contiguous(), 0.1, ptr, ptr // 5, 16),
        "grid_cluster": lambda: pcops.grid_cluster(pos, torch.full((3,), 0.07, device=DEV)),
        "ball_query": lambda: pcops.ball_query(0.2, 16, xyz, new),
        "three_nn": lambda: pcops.three_nn(xyz, new),
        "group_points": lambda: pcops.group_points(feats, gi),
        "gather_points": lambda: pcops.gather_points(feats, gi[:, :, 0].contiguous()),
        "three_interpolate": lambda: pcops.three_interpolate(sfeat, ti, w),
        "segment_csr_sum": lambda: pcops.segment_csr(src, indptr, "sum"),
        "segment_csr_max": lambda: pcops.segment_csr(src, indptr, "max"),
        "gather_csr": lambda: pcops.gather_csr(src[:5], indptr),
        "scatter_sum": lambda: pcops.scatter(src, index, 0, 300, "sum"),
        "scatter_max": lambda: pcops.scatter(src, index, 0, 300, "max"),
    }


def _same(a, b):
    if isinstance(a, (tuple, list)):
        return all(_same(x, y) for x, y in zip(a, b))
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


# ---------------------------------------------------------------- registration contract

def opcheck_cases():
    g = torch.Generator(device=DEV).manual_seed(0) if GPU else None
    pos = torch.rand(300, 3, device=DEV, generator=g)
    ptr = torch.tensor([0, 120, 300], device=DEV)
    xyz = torch.rand(2, 64, 3, device=DEV, generator=g)
    new = xyz[:, :16].contiguous()
    f = torch.randn(2, 4, 64, device=DEV, dtype=torch.double, generator=g, requires_grad=True)
    gi = torch.randint(0, 64, (2, 16, 8), device=DEV, dtype=torch.int32, generator=g)
    pi = torch.randint(0, 64, (2, 16), device=DEV, dtype=torch.int32, generator=g)
    ti = torch.randint(0, 64, (2, 20, 3), device=DEV, dtype=torch.int32, generator=g)
    w = torch.rand(2, 20, 3, device=DEV, dtype=torch.double, generator=g)
    src = torch.randn(20, 3, device=DEV, dtype=torch.double, generator=g, requires_grad=True)
    ip = torch.tensor([0, 5, 5, 20], device=DEV)
    idx = torch.randint(0, 6, (4, 20), device=DEV, generator=g)
    ops = torch.ops.pcops
    return [
        ("fps", ops.fps.default, (pos, ptr, 0.5, False)),
        ("knn", ops.knn.default, (pos, pos, ptr, ptr, 4, False)),
        ("radius", ops.radius.default, (pos, pos, ptr, ptr, 0.2, 8, False)),
        ("grid", ops.grid.default, (pos, torch.full((3,), 0.1, device=DEV), None, None)),
        ("ball_query", ops.ball_query.default, (new, xyz, 0.3, 8)),
        ("three_nn", ops.three_nn.default, (xyz, new)),
        ("group_points", ops.group_points.default, (f, gi)),
        ("gather_points", ops.gather_points.default, (f, pi)),
        ("three_interpolate", ops.three_interpolate.default, (f, ti, w)),
        ("segment_csr_sum", ops.segment_csr.default, (src, ip, "sum")),
        ("segment_csr_mean", ops.segment_csr.default, (src, ip, "mean")),
        ("segment_csr_max", ops.segment_csr.default, (src, ip, "max")),
        ("gather_csr", ops.gather_csr.default, (torch.randn(3, 3, device=DEV, dtype=torch.double, requires_grad=True), ip)),
        ("scatter_sum", ops.scatter.default, (torch.randn(4, 20, device=DEV, dtype=torch.double, requires_grad=True), idx, 1, 7, "sum")),
        ("scatter_mean", ops.scatter.default, (torch.randn(4, 20, device=DEV, dtype=torch.double, requires_grad=True), idx, 1, 7, "mean")),
        ("scatter_min", ops.scatter.default, (torch.randn(4, 20, device=DEV, dtype=torch.double, requires_grad=True), idx, 1, 7, "min")),
    ]


@pytest.mark.parametrize("case", opcheck_cases(), ids=lambda c: c[0])
def test_opcheck(case):
    """Schema, fake-tensor, autograd-registration and dynamic-shape checks of
    torch.library for every operator (as registered by the original)."""
    _, op, args = case
    result = torch.library.opcheck(op, args)
    assert all(v == "SUCCESS" for v in result.values()), result


def test_backward_ops_are_registered():
    for name in ("group_points_backward", "gather_points_backward", "three_interpolate_backward"):
        assert hasattr(torch.ops.pcops, name), name


# ---------------------------------------------------------------- determinism

def test_bitwise_determinism():
    calls = all_op_calls()
    # scatter uses atomics and is exempt, every other operator is deterministic
    for name, fn in calls.items():
        if name.startswith("scatter"):
            continue
        first = fn()
        for _ in range(3):
            assert _same(first, fn()), f"{name} is not deterministic"


def test_large_segment_sum_is_deterministic_in_float32():
    src = rnd(3_000_000, 4, seed=6) - 0.5
    indptr = torch.tensor([0, 1, 2_999_999, 3_000_000], device=DEV)
    a = pcops.segment_csr(src, indptr, "sum")
    for _ in range(3):
        assert torch.equal(a, pcops.segment_csr(src, indptr, "sum"))
    assert torch.allclose(a.double(), ref.segment_csr(src.cpu().double(), indptr.cpu(), "sum")[0].to(DEV), rtol=1e-4, atol=1e-2)


# ---------------------------------------------------------------- streams / threads

@pytest.mark.skipif(not GPU, reason="CUDA streams")
def test_side_stream_correctness():
    calls = all_op_calls()
    expected = {k: fn() for k, fn in calls.items() if not k.startswith("scatter")}
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    big = torch.randn(4096, 4096, device=DEV)
    with torch.cuda.stream(s):
        _ = big @ big  # keep the side stream busy first
        got = {k: calls[k]() for k in expected}
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    for k in expected:
        assert _same(expected[k], got[k]), k


@pytest.mark.skipif(not GPU, reason="concurrency")
def test_concurrent_threads():
    calls = all_op_calls()
    expected = {k: fn() for k, fn in calls.items() if not k.startswith("scatter")}
    torch.cuda.synchronize()
    errors = []

    def worker():
        try:
            with torch.cuda.stream(torch.cuda.Stream()):
                for _ in range(3):
                    for k in expected:
                        if not _same(expected[k], calls[k]()):
                            errors.append(k)
            torch.cuda.synchronize()
        except Exception as e:  # pragma: no cover
            errors.append(repr(e))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:5]


# ---------------------------------------------------------------- compilation

@pytest.mark.skipif(not GPU, reason="inductor")
def test_compile_dynamic_shapes_no_graph_breaks():
    def f(xyz, feats, src, indptr, index):
        new = xyz[:, ::4].contiguous()
        idx = pcops.ball_query(0.25, 8, xyz, new)
        g = pcops.group_points(feats, idx).amax(-1)
        d, i = pcops.three_nn(xyz, new)
        w = 1.0 / (d + 1e-8)
        up = pcops.three_interpolate(g, i, w / w.sum(-1, keepdim=True))
        seg = pcops.segment_csr(src, indptr, "mean")
        sc = pcops.scatter(src, index, 0, 64, "max")
        return up.sum(1), seg, sc

    cf = torch.compile(f, fullgraph=True, dynamic=True)
    for n, e in ((256, 300), (512, 1000), (1024, 77)):
        xyz, feats = rnd(2, n, 3, seed=n), rnd(2, 5, n, seed=n + 1)
        src = rnd(e, 3, seed=e)
        indptr = torch.linspace(0, e, 9, device=DEV).long()
        index = torch.randint(0, 64, (e,), device=DEV)
        for a, b in zip(f(xyz, feats, src, indptr, index), cf(xyz, feats, src, indptr, index)):
            assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.skipif(not GPU, reason="inductor")
def test_compiled_training_step_matches_eager():
    xyz, feats = rnd(2, 512, 3, seed=8), rnd(2, 6, 512, seed=9)

    def loss(feats):
        idx = pcops.ball_query(0.3, 8, xyz, xyz[:, ::4].contiguous())
        g = pcops.group_points(feats, idx)
        d, i = pcops.three_nn(xyz, xyz[:, ::4].contiguous())
        up = pcops.three_interpolate(g.amax(-1), i, (1 / (d + 1e-8)) / (1 / (d + 1e-8)).sum(-1, keepdim=True))
        return up.square().mean() + pcops.gather_points(feats, idx[:, :, 0].contiguous()).mean()

    fe = feats.clone().requires_grad_()
    fc = feats.clone().requires_grad_()
    loss(fe).backward()
    torch.compile(loss, fullgraph=True)(fc).backward()
    assert torch.allclose(fe.grad, fc.grad, atol=1e-5)


# ---------------------------------------------------------------- errors match the original

@pytest.mark.parametrize("bad", [
    lambda: pcops.knn(rnd(10, 3), rnd(4, 3), 101),
    lambda: pcops.segment_csr(rnd(10, 3), torch.tensor([0, 10], device=DEV), "mul"),
    lambda: pcops.scatter(rnd(10), torch.zeros(10, dtype=torch.long, device=DEV), 0, 3, "median"),
    lambda: pcops.group_points(rnd(2, 3, 10), torch.zeros(2, 4, 2, dtype=torch.long, device=DEV)),
    lambda: pcops.ball_query(0.1, 4, rnd(2, 10, 3), rnd(3, 4, 3)),
    lambda: pcops.ball_query(0.1, 4, rnd(2, 10, 2), rnd(2, 4, 2)),
    lambda: pcops.three_interpolate(rnd(2, 3, 10), torch.zeros(2, 5, 3, dtype=torch.int32, device=DEV), rnd(2, 5, 3, dtype=torch.float64)),
    lambda: pcops.fps(rnd(10, 3), None, 0.0),
], ids=["knn_k_over_100", "segment_bad_reduce", "scatter_bad_reduce", "group_int64_idx", "ball_batch_mismatch", "ball_not_3d",
        "interp_weight_dtype", "fps_ratio_zero"])
def test_invalid_arguments_raise(bad):
    with pytest.raises(RuntimeError):
        bad()


# ---------------------------------------------------------------- skewed / large inputs

def test_skewed_segments_and_empty_segments():
    E = 2_000_000
    src = rnd(E, 3, seed=10) - 0.5
    indptr = torch.tensor([0, 0, 0, 5, 1_999_990, E, E, E], device=DEV)
    for reduce in ("sum", "mean", "min", "max"):
        got = pcops.segment_csr(src, indptr, reduce)
        want = ref.segment_csr(src.cpu().double(), indptr.cpu(), reduce)[0]
        assert torch.allclose(got.cpu().double(), want, rtol=1e-4, atol=1e-2), reduce
    back = pcops.gather_csr(pcops.segment_csr(src, indptr, "max"), indptr)
    assert back.shape == src.shape


def test_scatter_contended_sum_and_half_max():
    src = (rnd(400_000, 2, seed=11) * 4 - 2)
    index = torch.randint(0, 3, (400_000,), device=DEV)
    got = pcops.scatter(src, index, 0, 3, "sum")
    want = ref.scatter(src.cpu().double(), index.cpu(), 0, 3, "sum")
    assert torch.allclose(got.cpu().double(), want, rtol=1e-3, atol=0.5)
    # half-precision sums depend on the accumulation order (the original uses
    # half atomics); max is exact in every precision
    for dtype in (torch.float16, torch.bfloat16):
        mx = pcops.scatter(src.to(dtype), index, 0, 5, "max").cpu()
        assert mx.dtype == dtype
        assert torch.equal(mx[:3], ref.scatter(src.to(dtype).cpu(), index.cpu(), 0, 3, "max").to(dtype))
        assert (mx[3:] == 0).all()


def test_ball_query_and_radius_very_large_neighbourhoods():
    xyz, new = lattice(2 * 3000, 12, 12).view(2, 3000, 3), lattice(2 * 40, 12, 13).view(2, 40, 3)
    got = pcops.ball_query(4.0, 700, xyz, new).cpu()
    assert torch.equal(got, ref.ball_query(4.0, 700, xyz.cpu(), new.cpu()))
    x, y = lattice(3000, 12, 14), lattice(30, 12, 15)
    assert torch.equal(pcops.radius(x, y, 4.0, max_num_neighbors=1500).cpu(), ref.radius(x.cpu(), y.cpu(), 4.0, max_num_neighbors=1500))


def test_knn_k100_large():
    x, y = lattice(20000, 30, 16), lattice(300, 30, 17)
    assert torch.equal(pcops.knn(x, y, 100).cpu(), ref.knn(x.cpu(), y.cpu(), 100))


def test_fps_huge_example_and_many_tiny_examples():
    pos = rnd(150_000, 3, seed=18)
    ptr = torch.tensor([0, 150_000], device=DEV)
    assert ref.check_fps(pos.cpu(), ptr.cpu(), 0.002, pcops.fps(pos, ptr, 0.002)) == []
    counts = [1, 2, 3] * 300
    p2 = rnd(sum(counts), 3, seed=19)
    ptr2 = ref.ptr_from_counts(counts).to(DEV)
    assert ref.check_fps(p2.cpu(), ptr2.cpu(), 0.5, pcops.fps(p2, ptr2, 0.5)) == []


# ---------------------------------------------------------------- the implementation is Triton

@pytest.mark.skipif(not GPU, reason="Triton launch hooks need a GPU")
def test_every_operator_launches_triton_kernels():
    import triton

    calls = all_op_calls()
    counts = {}
    hook_owner = triton.knobs.runtime
    old = hook_owner.launch_enter_hook
    current = [None]

    def hook(*_args, **_kwargs):
        counts[current[0]] = counts.get(current[0], 0) + 1

    hook_owner.launch_enter_hook = hook
    try:
        for name, fn in calls.items():
            current[0] = name
            fn()
            torch.cuda.synchronize()
    finally:
        hook_owner.launch_enter_hook = old
    missing = [n for n in calls if counts.get(n, 0) == 0]
    assert not missing, f"operators that launched no Triton kernel: {missing}"
