"""A token-in-token-out vLLM engine on Modal for live RL rollouts (m14 RLM harness smoke).

    modal secret create crucible-vllm VLLM_API_KEY=<random>   # once; put the same key in .env as CRUCIBLE_VLLM_KEY
    modal deploy scripts/modal_vllm.py                       # prints the https URL; scales to zero after 5 idle minutes
    CRUCIBLE_VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 CRUCIBLE_VLLM_GPU=L4 modal deploy scripts/modal_vllm.py
    modal app stop crucible-vllm                             # stop billing at once

Crucible talks to it with ``LLMProvider(base_url=<url>/v1, api_key="env:CRUCIBLE_VLLM_KEY", tito=True, policy_version=0)``.
Prompts are rendered with the model's chat template locally and sent as token ids to ``/v1/completions``
(``return_token_ids``, logprobs), so every call's token record is what the engine sampled. ``--logprobs-mode
processed_logprobs`` makes the logprobs the post-temperature values a trainer recomputes. The bearer key keeps the public
URL from being an open LLM.
"""

import os
import subprocess

import modal

MODEL = os.environ.get("CRUCIBLE_VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
GPU = os.environ.get("CRUCIBLE_VLLM_GPU", "L4")
MAX_LEN = int(os.environ.get("CRUCIBLE_VLLM_MAX_LEN", "32768"))

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12").entrypoint([])
         .uv_pip_install("vllm==0.11.0", "transformers>=4.56,<5", "huggingface_hub[hf_transfer]")  # vLLM 0.11 breaks on transformers 5
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))
hf_cache = modal.Volume.from_name("crucible-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("crucible-vllm-cache", create_if_missing=True)
app = modal.App("crucible-vllm")


@app.function(image=image, gpu=GPU, scaledown_window=300, timeout=3600, secrets=[modal.Secret.from_name("crucible-vllm")],
              volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache})
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=900)
def serve():
    cmd = ["vllm", "serve", MODEL, "--host", "0.0.0.0", "--port", "8000", "--max-model-len", str(MAX_LEN), "--served-model-name", MODEL,
           "--logprobs-mode", "processed_logprobs", "--api-key", os.environ["VLLM_API_KEY"], "--gpu-memory-utilization", "0.90"]
    subprocess.Popen(cmd)
