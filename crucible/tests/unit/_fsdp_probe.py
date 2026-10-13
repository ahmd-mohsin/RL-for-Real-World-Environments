"""FSDP2 internals probe (``scripts/modal_infra_check.py --only fsdp_probe``): after an optimizer step and an explicit
reshard / unshard, does the gathered copy equal the shards?"""
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

out = Path(sys.argv[2])
backend = os.environ.get("CRUCIBLE_DIST_BACKEND", "gloo")
dist.init_process_group(backend, init_method=f"file://{out / 'store'}", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]),
                        timeout=timedelta(seconds=120))
rank = dist.get_rank()
local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count()) if backend == "nccl" else 0))  # node-local (G5 D6)
dev = torch.device(f"cuda:{local}") if backend == "nccl" else torch.device("cpu")
if dev.type == "cuda":
    torch.cuda.set_device(dev)
torch.manual_seed(0)
m = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4)).to(dev)
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh(dev.type, (dist.get_world_size(),))
for u in m:
    fully_shard(u, reshard_after_forward=False, mesh=mesh)
fully_shard(m, reshard_after_forward=False, mesh=mesh)
mods = [x for x in m.modules() if hasattr(x, "unshard")]
for x in mods:
    x.set_requires_gradient_sync(False, recurse=False)
    x.set_reshard_after_backward(False, recurse=False)
opt = torch.optim.AdamW(m.parameters(), lr=0.1)
x_in = torch.randn(4, 8, device=dev)
res = {"rank": rank, "backend": backend}


def state():
    g = m[0]._get_fsdp_state()._fsdp_param_group
    return {"is_sharded": g.is_sharded, "is_unsharded": getattr(g, "is_unsharded", None)}


def gathered_vs_shards():
    fp = m[0]._get_fsdp_state()._fsdp_param_group.fsdp_params[0]
    full = fp.sharded_param.full_tensor().detach()
    return {"diff_gathered_vs_shards": float((m[0].weight.detach() - full).abs().max()), "moved_by_step": float((full - w0).abs().max()),
            "gathered_vs_pre_step": float((m[0].weight.detach() - w0).abs().max())}


for x in mods:
    x.unshard()
res["after_unshard"] = state()
w0 = m[0].weight.detach().clone()
m(x_in).sum().backward()
for x in mods:
    x.set_requires_gradient_sync(True, recurse=False)
(0.0 * m(x_in).sum()).backward()
for x in mods:
    x.set_requires_gradient_sync(False, recurse=False)
for x in mods:
    x.reshard()
res["after_reshard"] = state()
opt.step()
if dev.type == "cuda":
    torch.cuda.synchronize()
for x in mods:
    x.unshard()
res["after_unshard2"] = state()
res["gathered"] = gathered_vs_shards()
# the same sequence on a GRU unit (cuDNN on CUDA keeps its own _flat_weights references)
g = torch.nn.GRU(8, 8, batch_first=True).to(dev)
fully_shard(g, reshard_after_forward=False, mesh=mesh)
g.set_requires_gradient_sync(False)
g.set_reshard_after_backward(False)
og = torch.optim.AdamW(g.parameters(), lr=0.1)
seq = torch.randn(2, 5, 8, device=dev)
g.unshard()
g(seq)[0].sum().backward()
g.set_requires_gradient_sync(True)
(0.0 * g(seq)[0].sum()).backward()
g.set_requires_gradient_sync(False)
g.reshard()
og.step()
g.unshard()
fw = g._get_fsdp_state()._fsdp_param_group.fsdp_params
attr = g.weight_ih_l0
res["gru"] = {"flat_is_attr": g._flat_weights[0] is attr, "flat_ptr_eq": g._flat_weights[0].data_ptr() == attr.data_ptr(),
              "attr_vs_shards": float((attr.detach() - fw[0].sharded_param.full_tensor().detach()).abs().max()),
              "flat_vs_shards": float((g._flat_weights[0].detach() - fw[0].sharded_param.full_tensor().detach()).abs().max())}
(out / f"r{rank}.json").write_text(json.dumps(res, default=str))
dist.destroy_process_group()
