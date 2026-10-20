"""The ``BuildInfo`` record (R-11): which code produced a run. ``crucible.core.build.build_info()`` computes it.

The model lives in the schema package, beside ``RunRecord`` that embeds it: defined in ``crucible.core.build`` it closed
an import cycle (``core.build`` → ``core.schema`` package → ``schema.run`` → ``core.build``), so any process whose first
Crucible import was ``core.build`` (``crucible label`` through ``studio.runs``) failed with ``ImportError``.
"""

from __future__ import annotations

from crucible.core.schema.base import CrucibleModel


class BuildInfo(CrucibleModel):
    crucible_version: str
    source_sha256: str
    git_commit: str | None = None  # None: not a git install and not a checkout that tracks this package
    git_dirty: bool | None = None  # uncommitted changes inside the package directory; None when git_commit is unknown
    lock_sha256: str | None = None  # uv.lock of a source checkout
