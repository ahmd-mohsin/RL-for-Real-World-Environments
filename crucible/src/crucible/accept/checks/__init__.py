"""Acceptance checks A1-A17 and S1 (DESIGN §9; A14/A15/A17 for composite tasks, m13 G-3). Importing this package registers them
in ``base.REGISTRY``."""

from crucible.accept.checks import composition, core, extended  # noqa: F401
from crucible.accept.checks.base import REGISTRY, AcceptContext, CheckDef, Out

__all__ = ["REGISTRY", "AcceptContext", "CheckDef", "Out"]
