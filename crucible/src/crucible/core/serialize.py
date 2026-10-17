"""Strict loaders for documents read from files (R-7; PORT with notice of OpenEnv
``src/openenv/discovery/serialization.py:_unique_object``, BSD-3-Clause, Copyright (c) Meta Platforms, Inc. and affiliates).

``yaml.safe_load`` and ``json.loads`` keep the last of two equal keys, so a seed world.yaml with ``network: none`` and,
further down, ``network: public`` loads public without a word. Here a key given twice in one mapping is an error: YAML
raises ``yaml.constructor.ConstructorError`` (a ``yaml.YAMLError``, so handlers of malformed YAML treat it as one) naming
both lines; JSON raises ``ValueError``. JSON also refuses ``NaN`` and ``Infinity``, which are not JSON but which
``json.loads`` accepts. A YAML merge (``<<: *base``) is not a duplicate: keys written beside it override it, as YAML
intends. ``yaml.safe_load`` and ``yaml.load`` are banned outside this module (ruff TID251, pyproject.toml).
"""

from __future__ import annotations

import json
from typing import Any

import yaml
from yaml.constructor import ConstructorError

_MERGE = "tag:yaml.org,2002:merge"


class _StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: dict[Any, yaml.Node] = {}
        for key_node, _ in node.value:
            if key_node.tag == _MERGE:
                continue
            key = self.construct_object(key_node, deep=True)
            try:
                first = seen.get(key)
            except TypeError:  # an unhashable key: the base constructor reports it
                continue
            if first is not None:
                raise ConstructorError(f"while constructing a mapping, key {key!r} first given", first.start_mark,
                                       f"found duplicate key {key!r}", key_node.start_mark)
            seen[key] = key_node
        return super().construct_mapping(node, deep=deep)


def load_yaml(text: str, *, source: str = "<yaml>") -> Any:
    """``yaml.safe_load`` that refuses a key given twice in one mapping; ``source`` names the document in errors."""
    loader = _StrictLoader(text)
    loader.name = source
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def load_json(text: str | bytes, *, source: str = "<json>") -> Any:
    """``json.loads`` that refuses a key given twice in one object, and ``NaN`` / ``Infinity``."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"{source}: duplicate key {key!r}")
            out[key] = value
        return out

    def not_json(name: str) -> None:
        raise ValueError(f"{source}: {name} is not a JSON number")

    return json.loads(text, object_pairs_hook=unique, parse_constant=not_json)
