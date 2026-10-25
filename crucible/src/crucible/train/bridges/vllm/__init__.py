"""The vLLM bridge: the student served by our own vLLM on a GPU host (DeltaAI GH200; doc 14-deltaai).

The Tinker bridge's engine side (``TinkerSampler``: exact token-in/token-out, our ``TinkerChatCodec`` renderer, per-call
version and log, per-episode attribution) is reused unchanged. Only the service under it changes:

| module | content |
|---|---|
| ``client`` | ``VLLMModel`` (a model row priced at 0 with provider ``vllm``), ``vllm_model``, ``VLLMBackend`` (``sampler(model, path)``: the OpenAI-compatible ``/v1/completions`` endpoint fed token ids, LoRA adapters loaded at runtime) |
| ``serve`` | ``ServeSpec``: the ``vllm serve`` command line, inside NCSA's DeltaAI vLLM container through Apptainer, and the readiness wait |
| ``sampler`` | ``make``: the ``factory.models`` sampler backend ``vllm`` (registered ``final=True``) |

Both arms of a benchmark comparison (base and trained adapter) run on one server with the same sampling settings; the
backend is recorded with every run (``settings()["provider"] == "vllm"``, policy ids ``solver:vllm:<model>@vllm``).
"""

from crucible.train.bridges.vllm.client import VLLMBackend, VLLMModel, adapter_name, vllm_model

__all__ = ["VLLMBackend", "VLLMModel", "adapter_name", "vllm_model"]
