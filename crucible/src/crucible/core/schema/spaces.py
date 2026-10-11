"""Typed action and observation spaces (M1 item 10; the Gymnasium ``spaces`` vocabulary and dm_env ``specs``, as OpenEnv's
dm_control / Unity / FinRL environments export them).

A ``Space`` is data (JSON-serializable, part of a World's description), not a library object:

- ``box``: real vectors of ``shape`` within ``low``/``high`` (scalars broadcast); continuous control, prices;
- ``discrete``: one integer in ``[start, start + n)``;
- ``multi_discrete``: one integer per entry of ``nvec``;
- ``multi_binary``: ``n`` zeros and ones;
- ``dict`` / ``tuple``: compositions (dm_control observation dicts).

Each space checks a value (``contains`` / ``problem``), samples one reproducibly, flattens to a float vector for a
neural policy (``flat_dim`` / ``flatten``), and describes itself as a JSON schema for an LLM's tool call
(``json_schema``). ``from_gymnasium`` / ``from_spec_dict`` read the two foreign forms.
"""

from __future__ import annotations

import math
import random
from typing import Any, Literal

from pydantic import Field, model_validator

from crucible.core.schema.base import CrucibleModel


def _prod(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _flat(v: Any) -> list[Any]:
    if isinstance(v, list | tuple):
        return [x for e in v for x in _flat(e)]
    return [v]


def _reshape(flat: list[Any], shape: list[int]) -> Any:
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat[: shape[0]])
    step = _prod(shape[1:])
    return [_reshape(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


class Space(CrucibleModel):
    kind: Literal["box", "discrete", "multi_discrete", "multi_binary", "dict", "tuple"]
    shape: list[int] = Field(default_factory=list)  # box
    low: float | list[float | None] | None = None  # box: a scalar broadcasts; None (whole or per entry) = unbounded
    high: float | list[float | None] | None = None
    dtype: Literal["float32", "float64", "int64"] = "float32"
    n: int | None = None  # discrete: number of values; multi_binary: length
    start: int = 0  # discrete
    nvec: list[int] = Field(default_factory=list)  # multi_discrete
    spaces: dict[str, Space] | list[Space] | None = None  # dict / tuple
    description: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Space:
        k = self.kind
        if k == "box":
            if not self.shape or any(d <= 0 for d in self.shape):
                raise ValueError("box needs a shape of positive dimensions")
            for name in ("low", "high"):
                b = getattr(self, name)
                if isinstance(b, list) and len(b) != _prod(self.shape):
                    raise ValueError(f"box {name} has {len(b)} values for shape {self.shape}")
            lo, hi = self.bounds()
            if any(a > b for a, b in zip(lo, hi, strict=True)):
                raise ValueError("box low exceeds high")
        elif k == "discrete":
            if not self.n or self.n < 1:
                raise ValueError("discrete needs n >= 1")
        elif k == "multi_discrete":
            if not self.nvec or any(v < 1 for v in self.nvec):
                raise ValueError("multi_discrete needs nvec of positive sizes")
        elif k == "multi_binary":
            if not self.n or self.n < 1:
                raise ValueError("multi_binary needs n >= 1")
        elif k == "dict":
            if not isinstance(self.spaces, dict) or not self.spaces:
                raise ValueError("dict needs named spaces")
        elif k == "tuple" and (not isinstance(self.spaces, list) or not self.spaces):
            raise ValueError("tuple needs a list of spaces")
        return self

    # ---- bounds and checks -------------------------------------------------------------------------------------
    def bounds(self) -> tuple[list[float], list[float]]:
        size = _prod(self.shape)

        def expand(b: float | list[float | None] | None, inf: float) -> list[float]:
            if b is None:
                return [inf] * size
            return [inf if x is None else float(x) for x in b] if isinstance(b, list) else [float(b)] * size

        return expand(self.low, -math.inf), expand(self.high, math.inf)

    def problem(self, value: Any) -> str | None:
        """Why ``value`` is not in the space, or None."""
        k = self.kind
        try:
            if k == "box":
                flat = _flat(value)
                if len(flat) != _prod(self.shape):
                    return f"expected {_prod(self.shape)} numbers (shape {self.shape}), got {len(flat)}"
                lo, hi = self.bounds()
                for i, (x, a, b) in enumerate(zip(flat, lo, hi, strict=True)):
                    if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(float(x)):
                        return f"entry {i} is not a finite number"
                    if not a <= float(x) <= b:
                        return f"entry {i} = {x} is outside [{a}, {b}]"
                return None
            if k == "discrete":
                if isinstance(value, bool) or not isinstance(value, int):
                    return "expected an integer"
                return None if self.start <= value < self.start + int(self.n or 0) else f"{value} is outside [{self.start}, {self.start + int(self.n or 0)})"
            if k == "multi_discrete":
                if not isinstance(value, list) or len(value) != len(self.nvec):
                    return f"expected {len(self.nvec)} integers"
                bad = [i for i, (v, n) in enumerate(zip(value, self.nvec, strict=True)) if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < n]
                return f"entries {bad} are out of range" if bad else None
            if k == "multi_binary":
                if not isinstance(value, list) or len(value) != self.n or any(v not in (0, 1) or isinstance(v, float) for v in value):
                    return f"expected {self.n} zeros or ones"
                return None
            if k == "dict":
                assert isinstance(self.spaces, dict)
                if not isinstance(value, dict) or set(value) != set(self.spaces):
                    return f"expected keys {sorted(self.spaces)}"
                for name, sp in self.spaces.items():
                    if (p := sp.problem(value[name])) is not None:
                        return f"{name}: {p}"
                return None
            assert isinstance(self.spaces, list)
            if not isinstance(value, list | tuple) or len(value) != len(self.spaces):
                return f"expected {len(self.spaces)} parts"
            for i, sp in enumerate(self.spaces):
                if (p := sp.problem(value[i])) is not None:
                    return f"[{i}]: {p}"
            return None
        except (TypeError, ValueError) as e:
            return f"malformed value: {e}"

    def contains(self, value: Any) -> bool:
        return self.problem(value) is None

    def clip(self, value: Any) -> Any:
        """A box value clipped into the bounds (a neural policy's raw output); other kinds unchanged."""
        if self.kind != "box":
            return value
        lo, hi = self.bounds()
        flat = [min(max(float(x), a), b) for x, a, b in zip(_flat(value), lo, hi, strict=True)]
        return _reshape(flat, self.shape)

    # ---- sampling and flattening --------------------------------------------------------------------------------
    def sample(self, rng: random.Random) -> Any:
        k = self.kind
        if k == "box":
            lo, hi = self.bounds()
            flat = [rng.uniform(a if math.isfinite(a) else -1.0, b if math.isfinite(b) else 1.0) for a, b in zip(lo, hi, strict=True)]
            return _reshape(flat, self.shape)
        if k == "discrete":
            return self.start + rng.randrange(int(self.n or 1))
        if k == "multi_discrete":
            return [rng.randrange(n) for n in self.nvec]
        if k == "multi_binary":
            return [rng.randrange(2) for _ in range(int(self.n or 0))]
        if k == "dict":
            assert isinstance(self.spaces, dict)
            return {name: sp.sample(rng) for name, sp in self.spaces.items()}
        assert isinstance(self.spaces, list)
        return [sp.sample(rng) for sp in self.spaces]

    def flat_dim(self) -> int:
        k = self.kind
        if k == "box":
            return _prod(self.shape)
        if k == "discrete":
            return int(self.n or 0)  # one-hot
        if k == "multi_discrete":
            return sum(self.nvec)
        if k == "multi_binary":
            return int(self.n or 0)
        parts = self.spaces.values() if isinstance(self.spaces, dict) else (self.spaces or [])
        return sum(sp.flat_dim() for sp in parts)

    def flatten(self, value: Any) -> list[float]:
        """A float vector for a neural policy: boxes as numbers, discrete values one-hot, compositions concatenated
        (dict parts in key order)."""
        k = self.kind
        if k == "box":
            return [float(x) for x in _flat(value)]
        if k == "discrete":
            v = [0.0] * int(self.n or 0)
            v[int(value) - self.start] = 1.0
            return v
        if k == "multi_discrete":
            out: list[float] = []
            for x, n in zip(value, self.nvec, strict=True):
                oh = [0.0] * n
                oh[int(x)] = 1.0
                out += oh
            return out
        if k == "multi_binary":
            return [float(x) for x in value]
        if k == "dict":
            assert isinstance(self.spaces, dict)
            return [x for name in sorted(self.spaces) for x in self.spaces[name].flatten(value[name])]
        assert isinstance(self.spaces, list)
        return [x for sp, v in zip(self.spaces, value, strict=True) for x in sp.flatten(v)]

    # ---- for language models -------------------------------------------------------------------------------------
    def json_schema(self) -> dict[str, Any]:
        k = self.kind
        if k == "box":
            lo, hi = self.bounds()
            item: dict[str, Any] = {"type": "number"}
            if len(set(lo)) == 1 and math.isfinite(lo[0]):
                item["minimum"] = lo[0]
            if len(set(hi)) == 1 and math.isfinite(hi[0]):
                item["maximum"] = hi[0]
            size = _prod(self.shape)
            desc = self.description or f"{size} numbers" + (f" (shape {self.shape})" if len(self.shape) > 1 else "") + \
                ("" if "minimum" in item and "maximum" in item else f"; bounds low={lo} high={hi}")
            return {"type": "array", "items": item, "minItems": size, "maxItems": size, "description": desc}
        if k == "discrete":
            return {"type": "integer", "minimum": self.start, "maximum": self.start + int(self.n or 1) - 1, **({"description": self.description} if self.description else {})}
        if k == "multi_discrete":
            return {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": len(self.nvec), "maxItems": len(self.nvec),
                    "description": self.description or f"one integer per slot, slot i in [0, nvec[i]) with nvec={self.nvec}"}
        if k == "multi_binary":
            return {"type": "array", "items": {"type": "integer", "enum": [0, 1]}, "minItems": self.n, "maxItems": self.n}
        if k == "dict":
            assert isinstance(self.spaces, dict)
            return {"type": "object", "properties": {n: sp.json_schema() for n, sp in self.spaces.items()}, "required": sorted(self.spaces),
                    "additionalProperties": False}
        assert isinstance(self.spaces, list)
        return {"type": "array", "prefixItems": [sp.json_schema() for sp in self.spaces], "minItems": len(self.spaces), "maxItems": len(self.spaces)}


Space.model_rebuild()


def box(shape: list[int] | int, low: float | list[float] | None = None, high: float | list[float] | None = None, **kw: Any) -> Space:
    return Space(kind="box", shape=[shape] if isinstance(shape, int) else list(shape), low=low, high=high, **kw)


def discrete(n: int, start: int = 0, **kw: Any) -> Space:
    return Space(kind="discrete", n=n, start=start, **kw)


def from_gymnasium(sp: Any) -> Space:
    """A ``gymnasium.spaces`` object -> ``Space`` (duck-typed: no Gymnasium import needed)."""
    name = type(sp).__name__
    if name == "Box":
        def bound(arr: Any) -> float | list[float | None] | None:
            vals = [None if math.isinf(float(x)) else float(x) for x in arr.flatten().tolist()]
            if all(v is None for v in vals):
                return None
            return vals[0] if len(set(vals)) == 1 else vals

        return Space(kind="box", shape=list(sp.shape), low=bound(sp.low), high=bound(sp.high), dtype="float64" if "64" in str(sp.dtype) else "float32")
    if name == "Discrete":
        return Space(kind="discrete", n=int(sp.n), start=int(sp.start))
    if name == "MultiDiscrete":
        return Space(kind="multi_discrete", nvec=[int(x) for x in sp.nvec.flatten()])
    if name == "MultiBinary":
        return Space(kind="multi_binary", n=int(sp.n if isinstance(sp.n, int) else _prod(list(sp.n))))
    if name == "Dict":
        return Space(kind="dict", spaces={k: from_gymnasium(v) for k, v in sp.spaces.items()})
    if name == "Tuple":
        return Space(kind="tuple", spaces=[from_gymnasium(v) for v in sp.spaces])
    raise ValueError(f"unsupported Gymnasium space {name}")


def from_spec_dict(d: dict[str, Any]) -> Space:
    """A dm_env-style spec dict as OpenEnv's dm_control exports it: ``{shape, dtype, minimum, maximum}`` (a bounded array),
    or ``{name: {shape, dtype}}`` (an observation dict)."""
    if "shape" in d:
        shape = list(d["shape"]) or [1]
        lo, hi = d.get("minimum"), d.get("maximum")
        flat = lambda b: None if b is None else ([float(x) for x in _flat(b)] if isinstance(b, list) else float(b))
        lo, hi = flat(lo), flat(hi)
        if isinstance(lo, list) and len(set(lo)) == 1:
            lo = lo[0]
        if isinstance(hi, list) and len(set(hi)) == 1:
            hi = hi[0]
        return Space(kind="box", shape=shape, low=lo, high=hi, dtype="float64" if "64" in str(d.get("dtype", "")) else "float32")
    return Space(kind="dict", spaces={k: from_spec_dict(v) for k, v in d.items()})
