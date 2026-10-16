"""Path dependencies match uv.lock. uv records only the version a path source had when the lock was written, and
``uv sync --frozen`` installs whatever the folder holds, so the Harbor and agentenv clones had moved on (0.24.0 and
0.9.1324) while the lock and CI pinned 0.23.0 and 0.9.1301: local runs tested other code than CI, with nothing to say so.
A path source whose folder or installed metadata disagrees with the lock fails here: check the clone out at the commit
THIRD_PARTY.md pins (or re-lock on purpose)."""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_every_path_dependency_is_the_version_uv_lock_records():
    wrong = []
    for pkg in tomllib.loads((ROOT / "uv.lock").read_text())["package"]:
        source, locked = pkg.get("source", {}), pkg.get("version")
        folder = source.get("editable") or source.get("directory")
        if folder is None or locked is None:  # not a path source, or a dynamic version (verifiers)
            continue
        project = ROOT / folder / "pyproject.toml"
        on_disk = tomllib.loads(project.read_text()).get("project", {}).get("version") if project.is_file() else None
        try:
            installed = metadata.version(pkg["name"])
        except metadata.PackageNotFoundError:  # an extra this environment did not install (agentenv in CI)
            installed = None
        for what, version in (("its folder holds", on_disk), ("installed", installed)):
            if version is not None and version != locked:
                wrong.append(f"{pkg['name']}: {what} {version}, uv.lock records {locked} ({folder})")
    assert not wrong, "path dependencies differ from uv.lock:\n  " + "\n  ".join(wrong)
