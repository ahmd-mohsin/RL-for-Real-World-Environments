"""Content-addressed acceptance cache (research m9-m10/01 §15.3): ``<root>/<key_digest>/<check>@<version>.json`` with
evidence files beside it. A cached result is reused only if its evidence files still hash to what it recorded; that
is how A4's draws accumulate across sessions on a shared GPU and why a re-run after an unrelated change costs nothing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from crucible.accept.model import CheckResult, Evidence


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Cache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def dir(self, key_digest: str) -> Path:
        d = self.root / key_digest
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get(self, key_digest: str, check: str, version: int) -> CheckResult | None:
        p = self.root / key_digest / f"{check}@{version}.json"
        if not p.exists():
            return None
        try:
            r = CheckResult.model_validate_json(p.read_text())
        except ValueError:  # a torn or stale-schema entry is a miss, never an abort of the whole run (M9 audit V12)
            p.unlink(missing_ok=True)
            return None
        for e in r.evidence:
            f = self.root / key_digest / e.path
            if not f.exists() or _sha(f) != e.sha256:
                return None  # tampered or lost evidence: re-run
        return r

    def evidence(self, key_digest: str, check: str, name: str, data: Any, kind: str = "measured") -> Evidence:
        rel = f"evidence/{check}/{name}"
        f = self.dir(key_digest) / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(data if isinstance(data, str) else json.dumps(data, indent=1, sort_keys=True, default=str))
        return Evidence(path=rel, sha256=_sha(f), kind=kind)

    def put(self, key_digest: str, r: CheckResult) -> None:
        p = self.dir(key_digest) / f"{r.id}@{r.version}.json"
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        tmp.write_text(r.model_dump_json(indent=1))
        os.replace(tmp, p)
