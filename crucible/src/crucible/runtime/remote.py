"""Control-channel commands the harness runs on hosts and sandboxes, safe against killing themselves (Appendix B.5 row 8).

A cleanup step that runs ``pkill`` with the full-command-line flag through ``bash -c`` or ``ssh host '...'`` matches its *own* command line and
kills the channel: the step "fails" with a signal and no output, and the harness's view of the sandbox is gone. Here:

- ``control(cmd)`` runs a command with a sentinel; if the process dies by a signal and the sentinel never printed while
  the host is alive, the failure is ``crucible:harness_self_kill`` (harness blame, masked), not a policy outcome;
- ``kill_by_pidfile`` is the supported way to stop a runner process;
- ``lint_self_matching(paths)`` finds full-command-line ``pkill`` / ``pgrep`` patterns in code (a CI lint; patterns must use the ``[x]yz`` form or
  pidfiles).
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
from pathlib import Path

from crucible.core.schema.run import Failure

SENTINEL = "__CRUCIBLE_CONTROL_DONE__"


def control(cmd: str, *, timeout: float = 60.0, ssh: list[str] | None = None) -> tuple[subprocess.CompletedProcess, Failure | None]:
    script = f"{cmd}\necho {SENTINEL}"
    argv = [*(ssh or []), script] if ssh else ["bash", "-c", script]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if SENTINEL in r.stdout:
        return r, None
    if ssh and r.returncode == 255:  # ssh's own connection failure, not a kill (M10 audit V15)
        return r, Failure(kind="transport_unreachable", stage="cleanup", reason=f"ssh failed (rc 255): {r.stderr.strip()[-200:]}")
    killed = r.returncode < 0 or r.returncode in (128 + signal.SIGTERM, 128 + signal.SIGKILL, 255)
    if killed:
        return r, Failure(kind="crucible:harness_self_kill", stage="cleanup", reason=f"control channel died (rc {r.returncode}) before finishing: {cmd[:120]}")
    return r, None


def kill_by_pidfile(pidfile: Path, sig: int = signal.SIGTERM) -> bool:
    try:
        pid = int(Path(pidfile).read_text().strip())
        os.kill(pid, sig)
        return True
    except (OSError, ValueError):
        return False


# pkill/pgrep matching the full command line (any short-flag cluster with ``f``: -f, -af, -fl, -9f; or --full) on a
# pattern that is not bracket-guarded ("[f]oo" never matches its own command line). M10 audit V15: ``pgrep -af`` (the
# B.10 incident's common form) and ``--full`` were missed, and the quoted safe form was flagged.
SELF_MATCH = re.compile(r"\b(?:pkill|pgrep)\b(?:\s+(?:-\w*f\w*|--full)\b)(?:\s+-\S+)*\s+(?!['\"]?\[)['\"]?[^\s'\"\[]"
                        r"|\b(?:pkill|pgrep)\b(?:\s+-\S+)+\s+(?:-\w*f\w*|--full)\b(?:\s+-\S+)*\s+(?!['\"]?\[)['\"]?[^\s'\"\[]")


def lint_self_matching(paths: list[Path]) -> list[str]:
    hits = []
    for p in paths:
        for i, line in enumerate(Path(p).read_text(errors="replace").splitlines(), 1):
            if SELF_MATCH.search(line) and "lint_self_matching" not in line and "SELF_MATCH" not in line:
                hits.append(f"{p}:{i}: {line.strip()[:120]}")
    return hits
