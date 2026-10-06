import torch

from pcdata import make_batch, make_scene, lattice_scene


def test_deterministic_and_sized():
    a = make_batch([10, 0, 7], seed=3, duplicates=0.2, outliers=0.1, feat_dim=6)
    b = make_batch([10, 0, 7], seed=3, duplicates=0.2, outliers=0.1, feat_dim=6)
    assert torch.equal(a.pos, b.pos) and torch.equal(a.feat, b.feat) and torch.equal(a.label, b.label)
    assert a.ptr.tolist() == [0, 10, 10, 17]
    assert a.feat.shape == (17, 6)


def test_duplicates_and_lattice_ties():
    p, _, _ = make_scene(500, 1, duplicates=0.3)
    assert p.unique(dim=0).size(0) < 500
    q, _, _ = lattice_scene(200, 4, 2)
    assert torch.equal(q, q.round())
