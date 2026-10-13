"""The verifiers plugin id ``crucible-vf``: ``vf-eval crucible-vf --taskset.path <export dir>``. verifiers' loader imports
the top-level module the id names and takes the ``Taskset`` subclass exported in ``__all__`` (dotted ids do not load).
The implementation is ``crucible.interop.verifiers.vf_taskset``; this module only re-exports it."""

from crucible.interop.verifiers.vf_taskset import CrucibleTaskset, HarborEnv

__all__ = ["CrucibleTaskset", "HarborEnv"]
