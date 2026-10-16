"""Golden vectors for canonical JSON, scheme 1 (M0 audit 2026-10-16 R-10; PATTERN: NeMo-Gym ships golden digest vectors,
``token_id_capture/staging/conformance``, Apache-2.0).

Every content id is a hash of this text, and ``SCHEMES[1]`` is the live ``dumps``: switching to ``ensure_ascii``, another
float spelling or dropping a fold passed every other test while every stored id moved. One vector per rule pins the
exact text and id. If one fails, the canonical form changed: keep the old function, self-contained, as ``SCHEMES[1]``
and add the new one as scheme 2 (``core.canonical``), instead of editing these vectors."""

from __future__ import annotations

import math

import pytest

from crucible.core import ids
from crucible.core.canonical import SCHEMES, dumps_stored

VECTORS = [  # (rule, value, scheme-1 text, content id)
    ("keys sorted", {"b": 1, "a": {"d": 2, "c": 3}}, '{"a":{"c":3,"d":2},"b":1}',
     "78d48859c3252943aab7306f76c80f3f07783582e05ab8f944ce0696f2dbfc67"),
    ("compact separators", [1, {"a": [2, 3]}], '[1,{"a":[2,3]}]',
     "f26ebfea4bdb40c725d7fb268dbbcacbe5878d110b6f51db3448bba3fa1a72ad"),
    ("non-ASCII kept as UTF-8", {"name": "Zürich ✓"}, '{"name":"Zürich ✓"}',
     "a5fa840abdd243c0ab3e603fa8dc19b01e297edb0b7c4973407584c9b704a67d"),
    ("NFC folded", "é", '"é"', "f2886017e9c7abacf804b54d64787dce2b611c9544ba21f3affdd126a6e50086"),
    ("negative zero folded", [-0.0, 0.0], "[0.0,0.0]", "f29310de4c3ebf47f2b596ae85f92daedae7a426dd2e145ab2a1ebec019199e6"),
    ("float spelling", [0.1, 1.5, 3.0, 1e-07, 1e21, 123456789.125], "[0.1,1.5,3.0,1e-07,1e+21,123456789.125]",
     "b174847a050f838e8ae71b258d2b4b0d55a006f521f5d80f637e55852cb9cdf7"),
    ("big integer", 2**64, "18446744073709551616", "8b292fc2d32f1fd491784610c61b90dc4632a6e4add19261eca4c3979e4ea6d2"),
    ("literals", [True, False, None], "[true,false,null]", "4d0f18de2133118249c26acc481838d4f6bb6bc1de882d99abbd19ae9397e8df"),
    ("escapes", 'a"b\\c\nd\te\x00f\x7fg h', '"a\\"b\\\\c\\nd\\te\\u0000f\x7fg h"',
     "df81e3f2b021e63fe372dcd8e4299525fdd0832aa40a68a6c99e75ef9c10a062"),
    ("non-string key", {1: "a"}, '{"1":"a"}', "13f79ea251e19f42b4e3fbcbf52766de09dcc12e05b56e5728a0c46e25594215"),
    ("tuple as list", (1, 2), "[1,2]", "49a64717d5d4cb19952e6eac2946415cf6879adacf9908e7d872332d32c6e684"),
]


@pytest.mark.parametrize("rule, value, text, content_id", VECTORS, ids=[v[0] for v in VECTORS])
def test_scheme_1_text_and_id(rule, value, text, content_id):
    assert SCHEMES[1](value) == text
    assert ids.format_content_id(ids.sha256_text(text), 1) == content_id


@pytest.mark.parametrize("value", [math.nan, [math.inf], {"x": -math.inf}, {"é": 1, "é": 2}],
                         ids=["nan", "inf", "-inf", "keys one under NFC"])
def test_scheme_1_refuses(value):
    with pytest.raises(ValueError):
        SCHEMES[1](value)


def test_the_stored_form_keeps_text_as_given():
    assert dumps_stored({"n": "é"}) == '{"n":"é"}'  # identity folds to NFC, storage does not (R-20)
