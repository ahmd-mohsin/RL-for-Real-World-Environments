"""Canonical JSON: the one serialization every content hash in Crucible is computed over.

- Floats must be finite (NaN and ±inf have no JSON form, and a NaN reward would poison every aggregate); ``-0.0`` is
  folded to ``0.0`` so two equal values share an id.
- Strings (keys and values) are NFC-normalized, so the composed and decomposed spellings of one name share an id
  (Harbor normalizes step names the same way, ``harbor/models/task/step_name.py``).
- Keys are sorted, separators are compact, and non-ASCII text is kept as UTF-8.

``crucible.interop.canonical`` (the cross-runtime identity gate) uses the same normalizer.
"""

from __future__ import annotations

import json
import math
import unicodedata
from typing import Any


def normalize(o: Any) -> Any:
    """The canonical form of a JSON-shaped value. Raises ValueError on a non-finite float."""
    if isinstance(o, bool) or o is None or isinstance(o, int):
        return o
    if isinstance(o, float):
        if not math.isfinite(o):
            raise ValueError("non-finite float in a canonical document")
        return 0.0 if o == 0 else o  # -0.0 -> 0.0
    if isinstance(o, str):
        return unicodedata.normalize("NFC", o)
    if isinstance(o, dict):
        return {unicodedata.normalize("NFC", str(k)): normalize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [normalize(v) for v in o]
    return o


def dumps(obj: Any) -> str:
    """Canonical JSON text of a JSON-shaped value."""
    return json.dumps(normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
