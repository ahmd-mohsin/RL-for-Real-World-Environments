"""pcnet scenarios shared by the golden generator (CUDA original) and the test."""
import torch

from pcdata import make_batch
from pcnet import PartSegNet

CASES = [
    # name, counts per example (equal), seed, duplicates, outliers, lattice side
    ("ssg_8x2048", 8, 2048, 101, 0.02, 0.0, 0),
    ("ssg_4x4096_dups", 4, 4096, 102, 0.15, 0.002, 0),
    ("ssg_6x1024_lattice", 6, 1024, 103, 0.0, 0.0, 9),
]


def build(case, device="cuda"):
    name, B, N, seed, dups, outl, side = case
    torch.manual_seed(seed)
    model = PartSegNet(in_feat=3, num_classes=4).to(device)
    batch = make_batch([N] * B, seed=seed, duplicates=dups, outliers=outl, lattice_side=side).to(device)
    xyz, feats = batch.dense(N)
    return model, xyz, feats, batch.label.view(B, N)


def train_losses(model, xyz, feats, labels, steps=12):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(xyz, feats), labels)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses
