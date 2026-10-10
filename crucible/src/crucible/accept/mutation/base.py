from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Mutant:
    id: str
    family: str  # the defect class (A3 reports kills per family)
    solution: Any
    description: str = ""
