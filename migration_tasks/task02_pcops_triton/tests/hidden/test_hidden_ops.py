"""Hidden operator tests: semantics, dtypes, layouts, edge cases, gradients."""
import os

import pytest
import torch

import pcops
import reference as ref

DEV = os.environ.get("PCOPS_TEST_DEVICE", "cuda")
FLOATS = [torch.float32, torch.float64, torch.float16, torch.bfloat16]


def lattice(n, side, seed, dim=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, side, (n, dim), generator=g).float()


def dev(t):
    return t.to(DEV)


def noncontig(t):
    """A non-contiguous view with the same values (transpose round trip)."""
    if t.dim() < 2:
        return torch.stack([t, t], 1)[:, 0]
    big = torch.empty(t.shape[::-1], dtype=t.dtype, device=t.device).transpose(0, t.dim() - 1) if t.dim() == 2 else None
    if big is not None:
        big.copy_(t)
        assert not big.is_contiguous()
        return big
    out = torch.empty(list(t.shape[:-1]) + [t.shape[-1] * 2], dtype=t.dtype, device=t.device)[..., ::2]
    out.copy_(t)
    assert not out.is_contiguous()
    return out


# ---------------------------------------------------------------- fps

@pytest.mark.parametrize("counts", [[1], [2], [5, 0, 3], [700], [129, 1, 255, 1024]])
@pytest.mark.parametrize("ratio", [0.01, 0.25, 0.5, 1.0])
def test_fps_property(counts, ratio):
    g = torch.Generator().manual_seed(sum(counts))
    pos = torch.rand(sum(counts), 3, generator=g)
    ptr = ref.ptr_from_counts(counts)
    out = pcops.fps(dev(pos), dev(ptr), ratio)
    assert ref.check_fps(pos, ptr, ratio, out) == []


def test_fps_duplicates_lattice_and_far_points():
    pos = torch.cat([lattice(400, 3, 1), torch.zeros(50, 3), torch.full((5, 3), 400.0)])
    ptr = ref.ptr_from_counts([200, 255])
    out = pcops.fps(dev(pos), dev(ptr), 0.4)
    assert ref.check_fps(pos, ptr, 0.4, out) == []


@pytest.mark.parametrize("dim", [1, 2, 5, 16])
def test_fps_any_dimension(dim):
    g = torch.Generator().manual_seed(dim)
    pos = torch.rand(600, dim, generator=g)
    ptr = ref.ptr_from_counts([350, 250])
    assert ref.check_fps(pos, ptr, 0.1, pcops.fps(dev(pos), dev(ptr), 0.1)) == []


def test_fps_large_example():
    g = torch.Generator().manual_seed(9)
    pos = torch.rand(20000, 3, generator=g)
    ptr = ref.ptr_from_counts([20000])
    out = pcops.fps(dev(pos), dev(ptr), 0.004)
    assert ref.check_fps(pos, ptr, 0.004, out) == []


def test_fps_random_start():
    g = torch.Generator().manual_seed(3)
    pos = torch.rand(300, 3, generator=g)
    ptr = ref.ptr_from_counts([100, 200])
    starts = set()
    for _ in range(6):
        out = pcops.fps(dev(pos), dev(ptr), 0.1, random_start=True).cpu()
        assert out.numel() == sum(ref.fps_counts(ptr, 0.1))
        assert 0 <= int(out[0]) < 100 and 100 <= int(out[10]) < 300
        starts.add(int(out[0]))
        # property from the chosen start
        for b, (s, e) in enumerate([(0, 100), (100, 300)]):
            sel = out[0:10] if b == 0 else out[10:30]
            d = torch.full((e - s,), 5e4, dtype=torch.float64)
            p = pos[s:e].double()
            for m in range(1, sel.numel()):
                d = torch.minimum(d, ((p - p[int(sel[m - 1]) - s]) ** 2).sum(1))
                assert d[int(sel[m]) - s] >= d.max() * (1 - 1e-6)
    assert len(starts) > 1


def test_fps_rejects_bad_ratio():
    pos = dev(torch.rand(10, 3))
    with pytest.raises(Exception):
        pcops.fps(pos, None, 0.0)
    with pytest.raises(Exception):
        pcops.fps(pos, None, 1.5)


# ---------------------------------------------------------------- knn / radius

@pytest.mark.parametrize("k", [1, 2, 3, 8, 17, 64, 100])
def test_knn_exact_with_ties(k):
    x, y = lattice(500, 5, 1), lattice(120, 5, 2)
    px, py = ref.ptr_from_counts([300, 0, 200]), ref.ptr_from_counts([40, 30, 50])
    assert torch.equal(pcops.knn(dev(x), dev(y), k, dev(px), dev(py)).cpu(), ref.knn(x, y, k, px, py))


def test_knn_fewer_points_than_k_and_empty_queries():
    x, y = lattice(5, 4, 3), lattice(9, 4, 4)
    px, py = ref.ptr_from_counts([2, 3, 0]), ref.ptr_from_counts([3, 0, 6])
    assert torch.equal(pcops.knn(dev(x), dev(y), 4, dev(px), dev(py)).cpu(), ref.knn(x, y, 4, px, py))


@pytest.mark.parametrize("dim", [1, 2, 7])
def test_knn_dimensions(dim):
    x, y = lattice(200, 4, 5, dim), lattice(30, 4, 6, dim)
    assert torch.equal(pcops.knn(dev(x), dev(y), 6).cpu(), ref.knn(x, y, 6))


def test_knn_noncontiguous_and_half():
    x, y = lattice(150, 6, 7), lattice(40, 6, 8)
    got = pcops.knn(noncontig(dev(x)), noncontig(dev(y)), 5).cpu()
    assert torch.equal(got, ref.knn(x, y, 5))
    got16 = pcops.knn(dev(x).half(), dev(y).half(), 5).cpu()
    assert torch.equal(got16, ref.knn(x, y, 5))


def test_knn_cosine():
    g = torch.Generator().manual_seed(10)
    x, y = torch.randn(300, 8, generator=g), torch.randn(25, 8, generator=g)
    e = pcops.knn(dev(x), dev(y), 4, cosine=True).cpu()
    cos = 1 - (y.double() @ x.double().T) / (y.double().norm(dim=1)[:, None] * x.double().norm(dim=1)[None, :])
    want = cos.argsort(dim=1, stable=True)[:, :4]
    assert torch.equal(e[1].view(25, 4), want)


def test_knn_large_random_spotcheck():
    g = torch.Generator().manual_seed(11)
    x, y = torch.rand(60000, 3, generator=g), torch.rand(2000, 3, generator=g)
    e = pcops.knn(dev(x), dev(y), 16).cpu()
    assert e.shape == (2, 2000 * 16)
    rows = torch.randint(0, 2000, (40,), generator=g)
    for r in rows.tolist():
        d = ((x.double() - y[r].double()) ** 2).sum(1)
        got = e[1].view(2000, 16)[r]
        assert torch.allclose(d[got], d.sort().values[:16], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("mnb", [1, 2, 7, 32, 200])
def test_radius_exact(mnb):
    x, y = lattice(400, 7, 12), lattice(80, 7, 13)
    px, py = ref.ptr_from_counts([250, 150]), ref.ptr_from_counts([50, 30])
    got = pcops.radius(dev(x), dev(y), 2.0, dev(px), dev(py), mnb).cpu()
    assert torch.equal(got, ref.radius(x, y, 2.0, px, py, mnb))


def test_radius_ignore_same_index_and_noncontig():
    x = lattice(200, 5, 14)
    got = pcops.radius(noncontig(dev(x)), noncontig(dev(x)), 1.5, max_num_neighbors=9, ignore_same_index=True).cpu()
    assert torch.equal(got, ref.radius(x, x, 1.5, max_num_neighbors=9, ignore_same_index=True))


def test_radius_no_neighbours():
    x, y = lattice(30, 3, 15), torch.full((4, 3), 99.0)
    assert pcops.radius(dev(x), dev(y), 0.5).numel() == 0


# ---------------------------------------------------------------- grid

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grid(dtype):
    g = torch.Generator().manual_seed(16)
    pos = (torch.rand(5000, 3, generator=g) * 7 - 2).to(dtype)
    size = torch.tensor([0.31, 0.5, 1.7], dtype=dtype)
    assert torch.equal(pcops.grid_cluster(dev(pos), dev(size)).cpu(), ref.grid(pos, size))
    start, end = torch.tensor([-3.0, -3.0, -3.0], dtype=dtype), torch.tensor([6.0, 6.0, 6.0], dtype=dtype)
    assert torch.equal(pcops.grid_cluster(dev(pos), dev(size), dev(start), dev(end)).cpu(), ref.grid(pos, size, start, end))


# ---------------------------------------------------------------- ball query / three_nn

@pytest.mark.parametrize("nsample", [1, 3, 16, 64, 300])
def test_ball_query_exact(nsample):
    xyz, new = lattice(2 * 400, 8, 17).view(2, 400, 3), lattice(2 * 70, 8, 18).view(2, 70, 3)
    got = pcops.ball_query(2.2, nsample, dev(xyz), dev(new)).cpu()
    assert torch.equal(got, ref.ball_query(2.2, nsample, xyz, new))


def test_ball_query_padding_empty_and_noncontig():
    xyz = lattice(3 * 100, 6, 19).view(3, 100, 3)
    new = torch.cat([lattice(3 * 10, 6, 20).view(3, 10, 3), torch.full((3, 2, 3), 50.0)], 1)
    got = pcops.ball_query(1.1, 12, noncontig(dev(xyz)), noncontig(dev(new))).cpu()
    assert torch.equal(got, ref.ball_query(1.1, 12, xyz, new))
    assert got.dtype == torch.int32


@pytest.mark.parametrize("dtype", FLOATS)
def test_ball_query_dtypes(dtype):
    xyz, new = lattice(2 * 120, 6, 21).view(2, 120, 3), lattice(2 * 20, 6, 22).view(2, 20, 3)
    got = pcops.ball_query(1.6, 10, dev(xyz).to(dtype), dev(new).to(dtype)).cpu()
    assert torch.equal(got, ref.ball_query(1.6, 10, xyz, new))


def test_ball_query_zero_sizes():
    assert pcops.ball_query(1.0, 4, dev(torch.zeros(2, 0, 3)), dev(torch.zeros(2, 5, 3))).shape == (2, 5, 4)
    assert pcops.ball_query(1.0, 4, dev(torch.zeros(2, 7, 3)), dev(torch.zeros(2, 0, 3))).shape == (2, 0, 4)


@pytest.mark.parametrize("dtype", FLOATS)
def test_three_nn_exact(dtype):
    known, unknown = lattice(3 * 90, 7, 23).view(3, 90, 3), lattice(3 * 41, 7, 24).view(3, 41, 3)
    d, i = pcops.three_nn(dev(unknown).to(dtype), dev(known).to(dtype))
    rd, ri = ref.three_nn(unknown, known)
    assert torch.equal(i.cpu(), ri)
    assert d.dtype == dtype and torch.equal(d.cpu().double(), rd)


def test_three_nn_fewer_than_three_known():
    known, unknown = lattice(2 * 2, 4, 25).view(2, 2, 3), lattice(2 * 6, 4, 26).view(2, 6, 3)
    d, i = pcops.three_nn(dev(unknown), dev(known))
    assert torch.isinf(d[..., 2].cpu()).all() and (i[..., 2].cpu() == 0).all()
    rd, ri = ref.three_nn(unknown, known)
    assert torch.equal(i[..., :2].cpu(), ri[..., :2])


# ---------------------------------------------------------------- feature movement

@pytest.mark.parametrize("dtype", FLOATS)
def test_group_and_gather_points_dtypes(dtype):
    feats = torch.randn(2, 9, 70).to(dtype)
    gi = torch.randint(0, 70, (2, 13, 7), dtype=torch.int32)
    pi = torch.randint(0, 70, (2, 21), dtype=torch.int32)
    assert torch.equal(pcops.group_points(noncontig(dev(feats)), dev(gi)).cpu(), ref.group_points(feats, gi))
    assert torch.equal(pcops.gather_points(dev(feats), noncontig(dev(pi))).cpu(), ref.gather_points(feats, pi))


@pytest.mark.parametrize("dtype", FLOATS)
def test_group_points_backward_dtypes(dtype):
    feats = dev(torch.randn(2, 4, 30).to(dtype)).requires_grad_()
    gi = dev(torch.randint(0, 30, (2, 11, 9), dtype=torch.int32))
    out = pcops.group_points(feats, gi)
    go = torch.randn_like(out)
    out.backward(go)
    want = torch.zeros(2, 4, 30, dtype=torch.float64).scatter_add_(
        2, gi.long().cpu().view(2, 1, -1).expand(2, 4, 99), go.cpu().double().view(2, 4, -1))
    tol = 1e-12 if dtype == torch.float64 else (1e-5 if dtype == torch.float32 else 3e-2)
    assert feats.grad.dtype == dtype
    assert torch.allclose(feats.grad.cpu().double(), want, atol=tol * 10, rtol=tol)


@pytest.mark.parametrize("dtype", FLOATS)
def test_three_interpolate_dtypes(dtype):
    feats = torch.randn(2, 6, 50)
    idx = torch.randint(0, 50, (2, 33, 3), dtype=torch.int32)
    w = torch.rand(2, 33, 3)
    got = pcops.three_interpolate(dev(feats).to(dtype), dev(idx), dev(w).to(dtype)).cpu()
    want = ref.three_interpolate(feats.to(dtype), idx, w.to(dtype))
    tol = 1e-12 if dtype == torch.float64 else (1e-5 if dtype == torch.float32 else 2e-2)
    assert got.dtype == dtype and torch.allclose(got.double(), want, atol=tol, rtol=tol)


def test_gradchecks():
    f = dev(torch.randn(2, 3, 25, dtype=torch.double)).requires_grad_()
    gi = dev(torch.randint(0, 25, (2, 4, 6), dtype=torch.int32))
    pi = dev(torch.randint(0, 25, (2, 8), dtype=torch.int32))
    ti = dev(torch.randint(0, 25, (2, 7, 3), dtype=torch.int32))
    w = dev(torch.rand(2, 7, 3, dtype=torch.double))
    assert torch.autograd.gradcheck(lambda t: pcops.group_points(t, gi), (f,))
    assert torch.autograd.gradcheck(lambda t: pcops.gather_points(t, pi), (f,))
    assert torch.autograd.gradcheck(lambda t: pcops.three_interpolate(t, ti, w), (f,))


# ---------------------------------------------------------------- reductions

@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64])
def test_segment_csr_dtypes(reduce, dtype):
    g = torch.Generator().manual_seed(27)
    src = (torch.randn(40, 5, generator=g) * 10).to(dtype)
    indptr = torch.tensor([0, 0, 3, 3, 19, 40])
    got = pcops.segment_csr(dev(src), dev(indptr), reduce).cpu()
    want = ref.segment_csr(src.double() if dtype.is_floating_point else src, indptr, reduce)[0]
    if dtype.is_floating_point:
        tol = 1e-6 if dtype in (torch.float32, torch.float64) else 3e-2
        assert got.dtype == dtype and torch.allclose(got.double(), want.double(), rtol=tol, atol=tol * 10)
    else:
        assert torch.equal(got, want)


def test_segment_csr_batched_indptr_and_noncontig():
    g = torch.Generator().manual_seed(28)
    src = torch.randn(3, 17, 4, generator=g, dtype=torch.double)
    indptr = torch.tensor([[0, 5, 17], [0, 0, 17], [0, 17, 17]])
    for reduce in ("sum", "mean", "max"):
        got = pcops.segment_csr(noncontig(dev(src)), dev(indptr), reduce).cpu()
        assert torch.allclose(got, ref.segment_csr(src, indptr, reduce)[0])


@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_segment_and_gather_csr_grads(reduce):
    g = torch.Generator().manual_seed(29)
    src = dev(torch.randn(23, 3, generator=g, dtype=torch.double)).requires_grad_()
    indptr = dev(torch.tensor([0, 4, 4, 11, 23]))
    assert torch.autograd.gradcheck(lambda t: pcops.segment_csr(t, indptr, reduce), (src,))
    seg = dev(torch.randn(4, 3, dtype=torch.double)).requires_grad_()
    assert torch.autograd.gradcheck(lambda t: pcops.gather_csr(t, indptr), (seg,))


@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32, torch.int64])
def test_scatter_dtypes(reduce, dtype):
    g = torch.Generator().manual_seed(30)
    src = (torch.randn(5, 60, generator=g) * 20).to(dtype)
    index = torch.randint(0, 12, (60,), generator=g)
    got = pcops.scatter(dev(src), dev(index), dim=1, dim_size=14, reduce=reduce).cpu()
    want = ref.scatter(src.double() if dtype.is_floating_point else src, index, 1, 14, reduce)
    if dtype.is_floating_point:
        tol = 1e-6 if dtype in (torch.float32, torch.float64) else 3e-2
        assert got.dtype == dtype and torch.allclose(got.double(), want.double(), rtol=tol, atol=tol * 20)
    else:
        assert torch.equal(got, want)


@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_scatter_dims_and_grads(reduce):
    g = torch.Generator().manual_seed(31)
    src = torch.randn(4, 7, 9, generator=g, dtype=torch.double)
    for dim in (0, 1, 2):
        index = torch.randint(0, 5, src.shape, generator=g)
        got = pcops.scatter(dev(src), dev(index), dim=dim, dim_size=6, reduce=reduce).cpu()
        assert torch.allclose(got, ref.scatter(src, index, dim, 6, reduce))
    s = dev(src).requires_grad_()
    idx = dev(torch.randint(0, 5, (9,), generator=g))
    assert torch.autograd.gradcheck(lambda t: pcops.scatter(t, idx, dim=2, dim_size=6, reduce=reduce), (s,))


def test_scatter_large_index_space():
    g = torch.Generator().manual_seed(32)
    src = torch.randn(200000, 4, generator=g)
    index = torch.randint(0, 50000, (200000,), generator=g)
    got = pcops.scatter(dev(src), dev(index), dim=0, dim_size=50000, reduce="mean").cpu()
    assert torch.allclose(got, ref.scatter(src, index, 0, 50000, "mean"), atol=1e-5)
