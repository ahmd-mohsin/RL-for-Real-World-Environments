"""The Crucible plugin for agentenv (research m11-m12/05 §4), registered by entry points (``agent_env.*`` groups):
the ``crucible_world`` and ``crucible_verify`` task steps and the ``crucible_ref`` artifact type, plus an opt-in that adds
GPU flags to agentenv's builtin local Docker provider (``gpu_docker.local_gpu``). Only ``crucible.interop`` imports agentenv."""
