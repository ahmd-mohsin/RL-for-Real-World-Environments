"""vLLM worker extension for Crucible weight publishing (mounted into the vLLM container and passed as
``--worker-extension-cls crucible_vllm_ext.CrucibleWorkerExtension``; called through vLLM's dev ``/collective_rpc``).

Standalone on purpose (torch + safetensors only, no Crucible import): it runs inside vLLM's worker processes.
``crucible_load_weights(path)`` loads ``<path>/model.safetensors`` (HF parameter names, as ``HFPolicy.state_dict``
writes them) through the model's own ``load_weights`` (vLLM handles fused / sharded layouts), then synchronizes.
"""

from __future__ import annotations


class CrucibleWorkerExtension:
    def crucible_load_weights(self, path: str) -> str:
        import os

        import torch
        from safetensors.torch import load_file

        sd = load_file(os.path.join(path, "model.safetensors"), device="cpu")
        model = self.model_runner.model
        loaded = model.load_weights(weights=iter(sd.items()))
        torch.cuda.synchronize()
        return f"loaded {len(loaded) if loaded is not None else len(sd)} tensors from {path}"

    def crucible_ping(self) -> str:
        return "ok"
