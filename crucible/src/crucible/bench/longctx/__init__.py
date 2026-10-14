"""The long-context ladder (M16 design §4): oolong-style aggregation, MRCR-style multi-round retrieval and long-document
Q&A, generated from a seed at 8k to 4M tokens (``generators``; scorers ported from research-environments, Apache-2.0),
delivered as a host file a harness can mount as ``CONTEXT`` and through a World ``docs`` app, the answer submitted with
``answer.submit`` and graded from World state (``task``). Controls: a scripted RLM solver per family (1.0) and a no-op
(0) (``oracle``); ``ladder.run_longctx_ladder`` runs them per rung."""

from crucible.bench.longctx.generators import FAMILIES, SIZES, LongCtxItem, generate

__all__ = ["FAMILIES", "SIZES", "LongCtxItem", "generate"]
