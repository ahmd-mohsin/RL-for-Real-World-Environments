"""Lab full-stack campaign: every built module (M1–M12) on the lab RTX 5090 with a small, locally served LLM, on the
same environments the CPU gates use (toy helpdesk, workspace, long-horizon backlog, forge variants).

- ``lab.py``   stage / status / collect from the laptop (polite runner: waits for a quiet GPU, one container, cleans up);
- ``run.py``   inside the container: starts vLLM with tool calling, points every model alias at it (``crucible.lab.toml``),
               runs each module's gate as a timed subprocess, then the unit suite on Linux + CUDA, then the M12 GRPO gate;
               writes ``out/labstack.json`` (one entry per phase: rc, seconds, parsed result, output tail).
- ``phases.py`` the phases that had no CLI gate: M10 runner samples with the LLM policy, M11 replay identity of an
               LLM-produced trajectory.
"""
