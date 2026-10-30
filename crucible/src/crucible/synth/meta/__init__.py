"""crucible.synth.meta: the yardstick and the regularised self-improvement loop (part2-paper-core §8–§9, M3).

- ``plant``: contract and World mutation (the planter): eleven operators, one defect class each, every plant carrying an
  executed witness; natural-likeness features and an RBF-kernel MMD.
- ``recall``: the falsifier population on planted sets: FR_k,d and FA_k with Wilson intervals, the error-seeding
  residual estimate and the Chapman capture–recapture check; dev and locked sets (the locked set is report-only).
- ``update``: the regularised update rule R1–R10 over the versioned system state Θ = (S, F, P, K, A), with its ledger.
- ``library``: the typed operator DSL compiled into probes, the birth gate, retirement and the planter-loop acceptance.
- ``lemmas``: the lemma bank and its equal-budget acceptance test.

Nothing here calls a model: everything model-driven (proposers, evaluators that need roles, the leakage critic of
another family) is an injected ``Protocol``; the defaults are deterministic.
"""
