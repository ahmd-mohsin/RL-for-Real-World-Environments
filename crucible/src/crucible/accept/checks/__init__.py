"""Acceptance checks A1-A13 and S1 (DESIGN §9). Importing this package registers them in ``base.REGISTRY``."""

from crucible.accept.checks import core, extended  # noqa: F401
from crucible.accept.checks.base import REGISTRY, AcceptContext, CheckDef, Out

__all__ = ["REGISTRY", "AcceptContext", "CheckDef", "Out"]
