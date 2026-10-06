import pytest
import torch

import pcops
import reference as ref

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEV = "cuda"


def grid_points(n, side, seed):
    """Integer coordinates: every squared distance is exact in float32."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, side, (n, 3), generator=g).float()


@cuda
def test_fps_property():
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(700, 3, generator=g)
    ptr = ref.ptr_from_counts([300, 1, 399])
    out = pcops.fps(pos.to(DEV), ptr.to(DEV), ratio=0.25)
    assert ref.check_fps(pos, ptr, 0.25, out) == []


@cuda
def test_knn_matches_reference_with_ties():
    x = grid_points(200, 6, 1)
    y = grid_points(50, 6, 2)
    ptr_x = ref.ptr_from_counts([120, 80])
    ptr_y = ref.ptr_from_counts([20, 30])
    got = pcops.knn(x.to(DEV), y.to(DEV), 7, ptr_x.to(DEV), ptr_y.to(DEV)).cpu()
    assert torch.equal(got, ref.knn(x, y, 7, ptr_x, ptr_y))


@cuda
def test_radius_matches_reference():
    x = grid_points(300, 8, 3)
    y = grid_points(40, 8, 4)
    got = pcops.radius(x.to(DEV), y.to(DEV), 2.5, max_num_neighbors=16).cpu()
    assert torch.equal(got, ref.radius(x, y, 2.5, max_num_neighbors=16))


@cuda
def test_grid_cluster():
    g = torch.Generator().manual_seed(5)
    pos = torch.rand(1000, 3, generator=g) * 4
    size = torch.tensor([0.5, 0.75, 1.0])
    assert torch.equal(pcops.grid_cluster(pos.to(DEV), size.to(DEV)).cpu(), ref.grid(pos, size))


@cuda
def test_ball_query_first_hits_and_padding():
    xyz = grid_points(512, 7, 6).view(2, 256, 3)
    new = grid_points(64, 7, 7).view(2, 32, 3)
    got = pcops.ball_query(1.6, 9, xyz.to(DEV), new.to(DEV)).cpu()
    assert torch.equal(got, ref.ball_query(1.6, 9, xyz, new))


@cuda
def test_three_nn_and_interpolate():
    known = grid_points(2 * 90, 9, 8).view(2, 90, 3)
    unknown = grid_points(2 * 40, 9, 9).view(2, 40, 3)
    d, i = pcops.three_nn(unknown.to(DEV), known.to(DEV))
    rd, ri = ref.three_nn(unknown, known)
    assert torch.equal(i.cpu(), ri)
    assert torch.allclose(d.cpu().double(), rd)
    w = 1.0 / (d + 1e-8)
    w = w / w.sum(-1, keepdim=True)
    feats = torch.randn(2, 5, 90, device=DEV)
    out = pcops.three_interpolate(feats, i, w)
    assert torch.allclose(out.cpu().double(), ref.three_interpolate(feats, i, w), atol=1e-5)


@cuda
def test_group_and_gather_points_with_grad():
    feats = torch.randn(2, 4, 50, device=DEV, dtype=torch.double, requires_grad=True)
    idx = torch.randint(0, 50, (2, 6, 5), device=DEV, dtype=torch.int32)
    assert torch.equal(pcops.group_points(feats, idx).detach().cpu(), ref.group_points(feats.detach(), idx))
    torch.autograd.gradcheck(lambda f: pcops.group_points(f, idx), (feats,))
    idx2 = torch.randint(0, 50, (2, 7), device=DEV, dtype=torch.int32)
    torch.autograd.gradcheck(lambda f: pcops.gather_points(f, idx2), (feats,))


@cuda
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_segment_csr(reduce):
    g = torch.Generator().manual_seed(10)
    src = torch.randn(20, 3, generator=g, dtype=torch.double)
    indptr = torch.tensor([0, 2, 2, 7, 20])
    got = pcops.segment_csr(src.to(DEV), indptr.to(DEV), reduce).cpu()
    assert torch.allclose(got, ref.segment_csr(src, indptr, reduce)[0])
    s = src.to(DEV).requires_grad_()
    torch.autograd.gradcheck(lambda t: pcops.segment_csr(t, indptr.to(DEV), reduce), (s,))


@cuda
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_scatter(reduce):
    g = torch.Generator().manual_seed(11)
    src = torch.randn(4, 30, generator=g, dtype=torch.double)
    index = torch.randint(0, 9, (30,), generator=g)
    got = pcops.scatter(src.to(DEV), index.to(DEV), dim=1, dim_size=10, reduce=reduce).cpu()
    assert torch.allclose(got, ref.scatter(src, index, 1, 10, reduce))


@cuda
def test_torch_compile_fullgraph():
    xyz = torch.rand(2, 128, 3, device=DEV)
    feats = torch.randn(2, 8, 128, device=DEV)

    def f(xyz, feats):
        idx = pcops.ball_query(0.3, 8, xyz, xyz[:, :32].contiguous())
        g = pcops.group_points(feats, idx)
        d, i = pcops.three_nn(xyz, xyz[:, :32].contiguous())
        w = (1.0 / (d + 1e-8))
        w = w / w.sum(-1, keepdim=True)
        return g.amax(-1), pcops.three_interpolate(g.amax(-1), i, w)

    eager = f(xyz, feats)
    compiled = torch.compile(f, fullgraph=True)(xyz, feats)
    for a, b in zip(eager, compiled):
        assert torch.allclose(a, b)
