"""M15 ``studio``: the two products behind one CLI (ARCHITECTURE §4 M15). Study: ``docs/research/bench-studio/01``.

- ``pack``: ``crucible pack new`` (a conflict-safe scaffold from ``packs/_template``) and ``crucible pack check``
  (static / semantic levels, a severity policy, exit codes).
- ``label``: ``crucible label``, workflows to traces with dense rewards plus SFT, DPO and KTO data (M3 → M4 → M5/M6 →
  ``credit.datasets``).
- ``build``: ``crucible build``, workflows to RL tasks and a benchmark (M3 → M8 forge → M9 accept → M11 export → M13
  bench).

Every command writes a run directory: ``configs/resolved/<cmd>.json``, its outputs and ``summary.json`` (verifiers'
convention).
"""
