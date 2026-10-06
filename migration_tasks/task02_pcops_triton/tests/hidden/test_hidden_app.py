"""Hidden application tests: pcnet must behave like it did on the CUDA pcops."""
import os

import pytest
import torch

from app_cases import CASES, build, train_losses

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "data", "app_golden.pt")
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
def test_inference_matches_cuda_original(case, golden):
    model, xyz, feats, _ = build(case)
    model.eval()
    with torch.no_grad():
        logits = model(xyz, feats).cpu()
    want = golden[case[0]]["logits"]
    assert logits.shape == want.shape
    assert torch.allclose(logits, want, atol=2e-3, rtol=2e-3), (logits - want).abs().max()
    assert (logits.argmax(1) == want.argmax(1)).float().mean() > 0.999


@pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
def test_training_trajectory_matches_cuda_original(case, golden):
    model, xyz, feats, labels = build(case)
    losses = train_losses(model, xyz, feats, labels)
    want = golden[case[0]]["losses"]
    for step, (a, b) in enumerate(zip(losses, want)):
        assert abs(a - b) <= 2e-3 + 2e-3 * abs(b), f"step {step}: {a} vs {b}"


def test_compiled_model_inference_and_training():
    model, xyz, feats, labels = build(CASES[0])
    model.eval()
    with torch.no_grad():
        eager = model(xyz, feats)
        compiled = torch.compile(model, fullgraph=True)(xyz, feats)
    assert torch.allclose(eager, compiled, atol=1e-3, rtol=1e-3)
    model.train()
    cm = torch.compile(model, fullgraph=True)
    loss = torch.nn.functional.cross_entropy(cm(xyz, feats), labels)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


def test_pipeline_end_to_end():
    from pcdata import make_batch
    from pcnet import predict

    model, *_ = build(CASES[0])
    batch = make_batch([5000, 3333, 777, 64], seed=7, duplicates=0.1, outliers=0.01).to("cuda")
    a = predict(model, batch, n=512, voxel=0.03)
    b = predict(model, batch, n=512, voxel=0.03)
    assert a.shape == (4, 512) and torch.equal(a, b)
