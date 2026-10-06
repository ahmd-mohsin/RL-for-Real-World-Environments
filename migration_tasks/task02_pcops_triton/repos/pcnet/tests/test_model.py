import pytest
import torch

from pcdata import make_batch
from pcnet import PartSegNet, predict, voxel_downsample

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEV = "cuda"


def model(seed=0):
    torch.manual_seed(seed)
    return PartSegNet(in_feat=3, num_classes=4).to(DEV).eval()


def dense_batch(B=2, N=512, seed=0):
    b = make_batch([N] * B, seed=seed, duplicates=0.05).to(DEV)
    return b.dense(N)


@cuda
def test_forward_shapes_and_backward():
    m = model().train()
    xyz, feats = dense_batch()
    feats.requires_grad_()
    out = m(xyz, feats)
    assert out.shape == (2, 4, 512)
    out.square().mean().backward()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


@cuda
def test_eval_is_deterministic():
    m = model()
    xyz, feats = dense_batch(seed=1)
    with torch.no_grad():
        assert torch.equal(m(xyz, feats), m(xyz, feats))


@cuda
def test_compiled_model_matches_eager():
    m = model()
    xyz, feats = dense_batch(seed=2)
    with torch.no_grad():
        eager = m(xyz, feats)
        compiled = torch.compile(m, fullgraph=True)(xyz, feats)
    assert torch.allclose(eager, compiled, atol=1e-4, rtol=1e-4)


@cuda
def test_pipeline():
    b = make_batch([3000, 1500, 40], seed=4, duplicates=0.1, outliers=0.01).to(DEV)
    v = voxel_downsample(b, 0.05)
    assert v.ptr[-1] <= b.ptr[-1]
    assert (v.ptr[1:] >= v.ptr[:-1]).all()
    labels = predict(model(), b, n=256)
    assert labels.shape == (3, 256)
    assert labels.min() >= 0 and labels.max() < 4
