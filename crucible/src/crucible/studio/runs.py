"""Run directories of the studio commands (M15 audit G2 D6 / D12, H2 V17): one resolved config per command, compared
before any step writes, and a provenance stamp.

``record(out, cmd, cfg)`` is the single entry point every command calls *before* it writes anything else:

- ``cfg`` is scrubbed of credential-shaped values (``crucible.core.secrets.scrub_obj``) and stamped with ``provenance``
  (crucible version, git sha and dirty flag of the checkout, Python, the pack's name, version and content hash);
- when ``out/configs/resolved/<cmd>.json`` already exists, its ``identity`` keys (the flags that change the outputs) are
  compared with the new ones; a change raises ``ConfigChanged`` naming the keys ("use a new --out"), and nothing is
  written (the verifiers / Repo2RLEnv resume receipt; Harbor's job-config check);
- otherwise the file is (re)written and its path returned.

``label`` should call ``record(out, "label", cfg, identity=(...))`` the same way, before ingest.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from crucible.core.build import build_info


class ConfigChanged(ValueError):
    """A re-run into an existing ``--out`` with different identity flags."""


def pack_hash(pack_dir: Path) -> str:
    """sha256 over the pack's files (paths and bytes; caches, dotfiles and VCS dirs left out)."""
    h = hashlib.sha256()
    root = Path(pack_dir).resolve()
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or any(part.startswith(".") or part == "__pycache__" or part.endswith(".egg-info") for part in rel.parts):
            continue
        h.update(rel.as_posix().encode() + b"\0")
        try:
            h.update(hashlib.sha256(p.read_bytes()).digest())
        except OSError:
            h.update(b"<unreadable>")
    return h.hexdigest()


def provenance(pack_dir: Path | None = None, manifest: Any = None) -> dict[str, Any]:
    from crucible.studio.pack import crucible_version

    try:
        ver = crucible_version()
    except Exception:  # noqa: BLE001 - recorded as unknown, never a failure
        ver = None
    bi = build_info()  # R-11: git trusted only for Crucible's own checkout; dirty = the package's files only
    try:
        from importlib.metadata import version

        agentenv = version("agent-env")
    except Exception:  # noqa: BLE001
        agentenv = None
    out: dict[str, Any] = {"crucible": ver, "git_sha": bi.git_commit, "git_dirty": bi.git_dirty, "source_sha256": bi.source_sha256,
                           "lock_sha256": bi.lock_sha256, "python": platform.python_version(), "agentenv": agentenv}
    if pack_dir is not None:
        out["pack"] = {"name": getattr(manifest, "name", None), "version": getattr(manifest, "version", None), "dir": str(Path(pack_dir).resolve()),
                       "content_sha256": pack_hash(pack_dir)}
    return out


def config_path(out: Path, cmd: str) -> Path:
    return Path(out) / "configs" / "resolved" / f"{cmd}.json"


def check(out: Path, cmd: str, identity: dict[str, Any]) -> None:
    """Refuse a re-run whose identity flags differ from the recorded ones (nothing is written)."""
    p = config_path(out, cmd)
    if not p.is_file():
        return
    try:
        old = json.loads(p.read_text()).get("identity")
    except (OSError, ValueError):
        raise ConfigChanged(f"{p} is unreadable: use a new --out") from None
    if old is None:
        return
    new = json.loads(json.dumps(identity, sort_keys=True, default=str))
    changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    if changed:
        raise ConfigChanged(f"{Path(out)} holds a {cmd} run with different {', '.join(changed)} "
                            f"({', '.join(f'{k}: {old.get(k)!r} -> {new.get(k)!r}' for k in changed)}): use a new --out")


def record(out: Path, cmd: str, cfg: dict[str, Any], *, identity: dict[str, Any] | None = None, pack_dir: Path | None = None,
           manifest: Any = None) -> Path:
    """Compare, then write ``out/configs/resolved/<cmd>.json`` (scrubbed, stamped). ``identity`` defaults to ``cfg``."""
    from crucible.core.secrets import scrub_obj

    ident = scrub_obj(json.loads(json.dumps(identity if identity is not None else cfg, sort_keys=True, default=str)))
    check(out, cmd, ident)  # type: ignore[arg-type]
    doc = {"command": cmd, **scrub_obj(json.loads(json.dumps(cfg, default=str))), "identity": ident,  # type: ignore[dict-item]
           "provenance": provenance(pack_dir, manifest)}
    p = config_path(out, cmd)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return p
