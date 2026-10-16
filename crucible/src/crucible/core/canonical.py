"""Canonical JSON: the one serialization every content hash in Crucible is computed over.

- Floats must be finite (NaN and ±inf have no JSON form, and a NaN reward would poison every aggregate); ``-0.0`` is
  folded to ``0.0`` so two equal values share an id.
- Strings (keys and values) are NFC-normalized, so the composed and decomposed spellings of one name share an id
  (Harbor normalizes step names the same way, ``harbor/models/task/step_name.py``).
- Keys are sorted, separators are compact, and non-ASCII text is kept as UTF-8.

``crucible.interop.canonical`` (the cross-runtime identity gate) uses the same normalizer.

NFC is for identity only (M0 audit 2026-10-14 R-20; PATTERN: NeMo-Gym ``token_id_capture/fingerprint.py`` folds only to
compare, Apache-2.0). What the store keeps is ``dumps_stored``: the same sorted, compact JSON with the text exactly as
given, so a decomposed (NFD) string reads back as it was written and an event named by its raw text keeps its name.
For text already in NFC both forms are the same bytes. Two keys of one object that are one key under NFC are refused in
both forms: the canonical form used to keep one of them silently.
"""

from __future__ import annotations

import json
import math
import unicodedata
from typing import Any


def _form(o: Any, fold: bool) -> Any:
    if isinstance(o, bool) or o is None or isinstance(o, int):
        return o
    if isinstance(o, float):
        if not math.isfinite(o):
            raise ValueError("non-finite float in a canonical document")
        return 0.0 if o == 0 else o  # -0.0 -> 0.0
    if isinstance(o, str):
        return unicodedata.normalize("NFC", o) if fold else o
    if isinstance(o, dict):
        seen: dict[str, str] = {}
        out: dict[str, Any] = {}
        for k, v in o.items():
            k = str(k)
            nfc = unicodedata.normalize("NFC", k)
            if nfc in seen:
                raise ValueError(f"keys {seen[nfc]!r} and {k!r} are one key under NFC: rename one (the canonical form would keep only one)")
            seen[nfc] = k
            out[nfc if fold else k] = _form(v, fold)
        return out
    if isinstance(o, (list, tuple)):
        return [_form(v, fold) for v in o]
    return o


def normalize(o: Any) -> Any:
    """The canonical form of a JSON-shaped value. Raises ValueError on a non-finite float or on two keys of one object
    that are one key under NFC."""
    return _form(o, True)


def dumps(obj: Any) -> str:
    """Canonical JSON text of a JSON-shaped value: what identities hash."""
    return json.dumps(normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


# The serialization each content-id scheme hashed (R-21). An entry is never changed: when canonical JSON must change, the
# old ``dumps`` is kept here under its scheme, self-contained (a copy of ``_form`` with it), a new one is added, and
# ``core.ids.HASH_SCHEME`` names it. ``tests/unit/test_canonical_vectors.py`` pins scheme 1's text and ids per rule.
SCHEMES: dict[int, Any] = {}


def dumps_stored(obj: Any) -> str:
    """The JSON a store keeps: sorted and compact like ``dumps``, floats as ``dumps`` checks them, strings as given."""
    return json.dumps(_form(obj, False), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


SCHEMES[1] = dumps
