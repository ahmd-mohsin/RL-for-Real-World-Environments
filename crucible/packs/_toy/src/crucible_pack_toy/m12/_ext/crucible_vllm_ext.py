"""vLLM worker extension for Crucible weight publishing (mounted into the vLLM container and passed as
``--worker-extension-cls crucible_vllm_ext.CrucibleWorkerExtension``; called through vLLM's dev ``/collective_rpc``).

Standalone on purpose (torch + safetensors only, no Crucible import): it runs inside vLLM's worker processes.
``crucible_load_weights(path, version)`` loads ``<path>/model.safetensors`` (HF parameter names, as ``HFPolicy.state_dict``
writes them) through the model's own ``load_weights`` (vLLM handles fused / sharded layouts), then synchronizes, and
keeps ``version``. ``crucible_version()`` reads it back (M12 lab audit C-4: ``VLLMServerEngine.health``): None when
nothing was loaded since the worker started (the base weights), -1 after a load that raised part-way (torn weights).
Two identical copies: ``src/crucible/train/vllm_ext.py`` and ``packs/_toy/src/crucible_pack_toy/m12/_ext/crucible_vllm_ext.py``.
"""

from __future__ import annotations


class CrucibleWorkerExtension:
    def crucible_load_weights(self, path: str, version: int | None = None) -> str:
        import os

        import torch
        from safetensors.torch import load_file

        sd = load_file(os.path.join(path, "model.safetensors"), device="cpu")
        model = self.model_runner.model
        self._crucible_version = -1  # torn until the load completes
        loaded = model.load_weights(weights=iter(sd.items()))
        torch.cuda.synchronize()
        self._crucible_version = version
        return f"loaded {len(loaded) if loaded is not None else len(sd)} tensors from {path}"

    def crucible_version(self) -> dict:
        # a dict: vLLM's /collective_rpc returns dicts as they are and str() of anything else
        return {"version": getattr(self, "_crucible_version", None)}

    def crucible_ping(self) -> str:
        return "ok"
