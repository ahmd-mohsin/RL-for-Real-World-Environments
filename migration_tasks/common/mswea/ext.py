"""mini-swe-agent extensions for the GPU migration tasks.

SSHEnvironment     runs every action inside the task container (a Runpod pod
                   started from the task image) over SSH. The agent loop and
                   the API key stay on the machine running mini-swe-agent.
LongOpenRouterModel  OpenRouterModel with a configurable request timeout
                   (upstream hard-codes 60 s, too short for long reasoning turns).
"""
import json
import re
import shlex
import subprocess
from typing import Any

import requests
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.models.openrouter_model import (
    OpenRouterAPIError,
    OpenRouterAuthenticationError,
    OpenRouterModel,
    OpenRouterModelConfig,
    OpenRouterRateLimitError,
)
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

# The task is no-network but a pod cannot be cut off from the internet, so
# commands that reach out are refused here and the trajectory is audited.
NETWORK = re.compile(
    r"\b(curl|wget|pip3?\s+(install|download)|apt(-get)?\s+(install|update|download)|git\s+(clone|fetch|pull|ls-remote)|ssh|scp|nc|telnet)\b"
    r"|https?://|github\.com|huggingface"
)


class SSHEnvironmentConfig(LocalEnvironmentConfig):
    ssh: str
    """ssh command prefix, e.g. 'ssh -p 22 -i ~/.ssh/id_ed25519 root@HOST'"""
    cwd: str = "/app"
    timeout: int = 1800
    network_guard: bool = True


class SSHEnvironment(LocalEnvironment):
    def __init__(self, **kwargs):
        super().__init__(config_class=SSHEnvironmentConfig, **kwargs)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        if self.config.network_guard and NETWORK.search(command):
            output = {"output": "error: this environment has no network access", "returncode": 1, "exception_info": ""}
            self._check_finished(output)
            return output
        timeout = timeout or self.config.timeout
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.config.env.items())
        remote = f"{'export ' + exports + '; ' if exports else ''}cd {shlex.quote(cwd or self.config.cwd)} && timeout {timeout} bash -s 2>&1"
        ssh = shlex.split(self.config.ssh) + ["-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", remote]
        try:
            p = subprocess.run(ssh, input=command, capture_output=True, text=True, errors="replace", timeout=timeout + 120)
            rc = p.returncode
            output = {"output": p.stdout + p.stderr, "returncode": rc,
                      "exception_info": f"command timed out after {timeout}s" if rc == 124 else ""}
        except subprocess.TimeoutExpired as e:
            out = e.output.decode(errors="replace") if isinstance(e.output, bytes) else (e.output or "")
            output = {"output": out, "returncode": -1, "exception_info": f"command timed out after {timeout}s"}
        self._check_finished(output)
        return output


class LongOpenRouterModelConfig(OpenRouterModelConfig):
    request_timeout: int = 1200


class LongOpenRouterModel(OpenRouterModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = LongOpenRouterModelConfig(**kwargs)

    def _query(self, messages, **kwargs):
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"model": self.config.model_name, "messages": messages, "tools": [BASH_TOOL],
                   "usage": {"include": True}, **(self.config.model_kwargs | kwargs)}
        try:
            response = requests.post(self._api_url, headers=headers, data=json.dumps(payload),
                                     timeout=self.config.request_timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise OpenRouterAuthenticationError("Authentication failed") from e
            if response.status_code == 429:
                raise OpenRouterRateLimitError("Rate limit exceeded") from e
            raise OpenRouterAPIError(f"HTTP {response.status_code}: {response.text}") from e
        except requests.exceptions.RequestException as e:
            raise OpenRouterAPIError(f"Request failed: {e}") from e
