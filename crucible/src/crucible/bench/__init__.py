"""M13 ``bench``: step-level and OOD benchmarks (ARCHITECTURE §4 M13, DESIGN §11).

- ``metrics``: every step metric of DESIGN §11.1 computed from a run directory (the M10 journal): outcome pass@k, gate
  profile, progress AUC, stage reach, first irrecoverable step, regressions, recovery, cost, communication, probe profile.
  Masked samples are never samples.
- ``splits``: lineage-sealed train / bench / OOD-axis assignment; showcase tasks never enter a published split.
- ``irt``: 2PL item response theory; fresh items are placed on the anchors' scale.
- ``contamination``: task fingerprints, a corpus scan, and temporal decontamination.
- ``dynamic``: a benchmark of fixed anchors plus fresh items regenerated from held-out seeds behind a secret.
- ``report``: a deterministic JSON summary and an HTML report with per-step Φ heatmaps.
- ``run``: run a benchmark through the M10 runner and report it.

``crucible.bench`` imports no pack (ARCHITECTURE §2): tasks, generators and stage maps are passed in.
"""
