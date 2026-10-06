"""Run with the ORIGINAL CUDA pcops installed: writes data/app_golden.pt."""
import sys

import torch

sys.path.insert(0, ".")
from app_cases import CASES, build, train_losses

torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
out = {}
for case in CASES:
    model, xyz, feats, labels = build(case)
    model.eval()
    with torch.no_grad():
        logits = model(xyz, feats).cpu()
    losses = train_losses(model, xyz, feats, labels)
    out[case[0]] = {"logits": logits, "losses": losses}
    print(case[0], tuple(logits.shape), [round(l, 4) for l in losses[:4]], "...", round(losses[-1], 4))
torch.save(out, "data/app_golden.pt")
