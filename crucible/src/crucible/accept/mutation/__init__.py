"""Generic defect-class mutants (research m9-m10/01 §15.6). A mutant is "the oracle minus one property": an operator
applied inside the oracle's footprint. Packs register operators per solution kind; ``actions`` covers World tasks whose
oracle is a tool-call sequence, ``patch`` covers unified diffs (Python, C/C++/CUDA added lines)."""

from crucible.accept.mutation.base import Mutant

__all__ = ["Mutant"]
