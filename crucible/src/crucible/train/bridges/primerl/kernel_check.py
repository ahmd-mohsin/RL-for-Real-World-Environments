"""On a GPU node, before training the student: its Gated DeltaNet kernels run, and vLLM serves its architecture.

Qwen3.6-35B-A3B (``Qwen3_5MoeForConditionalGeneration``) has 30 Gated DeltaNet layers beside 10 full-attention ones.
prime-rl's layer (``trainer/models/qwen3_5/gated_delta_net.py``) takes its kernels from flash-linear-attention (Triton),
which needs a GPU, so the login node can only import it. This check runs both kernels forward and backward on tiny
tensors (seconds) and asks vLLM's registry for the architecture. ``python -m crucible.train.bridges.primerl.kernel_check``
prints one JSON line and exits non-zero on a failure.
"""

from __future__ import annotations

import json
import sys

ARCH = "Qwen3_5MoeForConditionalGeneration"


def main() -> int:
    out: dict[str, object] = {"arch": ARCH}
    ok = True
    try:
        import fla
        import torch
        import triton
        from fla.modules.conv import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        out["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
                           "fla": getattr(fla, "__version__", "?"), "gpu": torch.cuda.get_device_name(0)}
        dev = "cuda"
        b, t, h, k, v = 1, 64, 2, 32, 32
        q = torch.randn(b, t, h, k, device=dev, dtype=torch.bfloat16, requires_grad=True)
        kk = torch.randn(b, t, h, k, device=dev, dtype=torch.bfloat16, requires_grad=True)
        vv = torch.randn(b, t, h, v, device=dev, dtype=torch.bfloat16, requires_grad=True)
        g = torch.nn.functional.logsigmoid(torch.randn(b, t, h, device=dev, dtype=torch.float32))
        beta = torch.rand(b, t, h, device=dev, dtype=torch.float32).sigmoid()
        o, _ = chunk_gated_delta_rule(q, torch.nn.functional.normalize(kk, dim=-1), vv, g, beta)
        o.float().sum().backward()
        x = torch.randn(b, t, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(64, 4, device=dev, dtype=torch.bfloat16, requires_grad=True)
        y = causal_conv1d(x, w)
        (y[0] if isinstance(y, tuple) else y).float().sum().backward()
        out["fla"] = {"gated_delta_rule": list(o.shape), "causal_conv1d": "ok", "grads": bool(q.grad is not None and x.grad is not None)}
    except Exception as e:  # noqa: BLE001 -- reported, the job decides
        ok = False
        out["fla_error"] = f"{type(e).__name__}: {e}"[:400]
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        out["vllm_supports"] = ARCH in set(ModelRegistry.get_supported_archs())
        ok = ok and bool(out["vllm_supports"])
    except Exception as e:  # noqa: BLE001
        ok = False
        out["vllm_error"] = f"{type(e).__name__}: {e}"[:400]
    out["ok"] = ok
    print(json.dumps(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
