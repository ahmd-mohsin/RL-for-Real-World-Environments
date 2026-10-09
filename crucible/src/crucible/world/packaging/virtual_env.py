"""Deterministic time and randomness for Python tool modules (MiMo style), without editing them.

1,174 of the 1,796 MiMo tool modules call ``datetime.now()`` (timestamps, ids like ``"CHG-" + now.strftime(...)``)
and many use ``uuid4``/``random``: replaying the same call twice from the same state gives different databases,
and the timestamps contradict the World's virtual clock. ``patch_module`` rebinds those names *in the loaded
modules' globals* (never globally) to a ``VirtualEnv``:

- time: the World's virtual clock, plus 1 ms per reading within the same virtual instant (ids stay unique,
  readings stay monotonic), so the same call sequence from the same WorldRef yields the same values;
- randomness: ``random`` / ``uuid4`` / ``uuid1`` / ``secrets.token_hex`` draw from a stream derived from
  (World seed, virtual instant, reading index).

Covers ``import datetime``, ``from datetime import datetime, date``, ``import time``, ``from time import time``,
``import uuid``, ``from uuid import uuid4``, ``import random``, ``from random import ...``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import random as _random
import time as _time
import types
import uuid as _uuid
from collections.abc import Callable
from typing import Any


class VirtualEnv:
    def __init__(self, now: Callable[[], _dt.datetime], seed: int = 0):
        self._now, self.seed = now, seed
        self._key: _dt.datetime | None = None
        self._n = 0

    def reset(self) -> None:
        """Called when the World restores a WorldRef: replays start counting readings from zero again."""
        self._key, self._n = None, 0

    def now(self) -> _dt.datetime:
        base = self._now()
        if base != self._key:
            self._key, self._n = base, 0
        else:
            self._n += 1
        return base + _dt.timedelta(milliseconds=self._n)

    def rng(self) -> _random.Random:
        t = self.now()
        h = hashlib.sha256(f"{self.seed}|{t.isoformat()}".encode()).digest()
        return _random.Random(int.from_bytes(h[:8], "big"))


def _datetime_class(env: VirtualEnv) -> type:
    class datetime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            t = env.now()
            return t.astimezone(tz) if tz is not None else t.astimezone(_dt.UTC).replace(tzinfo=None)

        @classmethod
        def utcnow(cls):
            return env.now().astimezone(_dt.UTC).replace(tzinfo=None)

        @classmethod
        def today(cls):
            return cls.now()

    return datetime


def _date_class(env: VirtualEnv) -> type:
    class date(_dt.date):
        @classmethod
        def today(cls):
            return env.now().date()

    return date


def _shim(module: types.ModuleType, name: str, **overrides: Any) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    m.__dict__.update(overrides)
    return m


def shims(env: VirtualEnv) -> dict[str, Any]:
    vdt, vdate = _datetime_class(env), _date_class(env)

    def vtime() -> float:
        return env.now().timestamp()

    def vtime_ns() -> int:
        return int(env.now().timestamp() * 1e9)

    def vlocaltime(secs: float | None = None):
        return _time.localtime(vtime() if secs is None else secs)

    def vgmtime(secs: float | None = None):
        return _time.gmtime(vtime() if secs is None else secs)

    def vstrftime(fmt: str, t=None):
        return _time.strftime(fmt, vgmtime() if t is None else t)

    def uuid4() -> _uuid.UUID:
        return _uuid.UUID(int=env.rng().getrandbits(128), version=4)

    def uuid1(node=None, clock_seq=None) -> _uuid.UUID:
        return _uuid.UUID(int=env.rng().getrandbits(128), version=1)

    class _Rand(_random.Random):  # every draw re-derives its stream from the virtual instant
        def random(self) -> float:
            return env.rng().random()

        def getrandbits(self, k: int) -> int:
            return env.rng().getrandbits(k)

    rand = _Rand()
    dt_mod = _shim(_dt, "datetime", datetime=vdt, date=vdate)
    time_mod = _shim(_time, "time", time=vtime, time_ns=vtime_ns, localtime=vlocaltime, gmtime=vgmtime, strftime=vstrftime)
    uuid_mod = _shim(_uuid, "uuid", uuid4=uuid4, uuid1=uuid1)
    rand_funcs = {n: getattr(rand, n) for n in ("random", "randint", "randrange", "choice", "choices", "sample", "shuffle", "uniform", "getrandbits", "gauss")}
    rand_mod = _shim(_random, "random", **rand_funcs)
    return {"datetime_module": dt_mod, "datetime": vdt, "date": vdate, "time_module": time_mod, "uuid_module": uuid_mod,
            "random_module": rand_mod, "time_funcs": {"time": vtime, "time_ns": vtime_ns, "localtime": vlocaltime, "gmtime": vgmtime, "strftime": vstrftime},
            "uuid_funcs": {"uuid4": uuid4, "uuid1": uuid1}, "random_funcs": rand_funcs}


def patch_module(module: types.ModuleType, env: VirtualEnv, s: dict[str, Any] | None = None) -> list[str]:
    """Rebind the module's references to time/random sources; returns the names that were rebound."""
    s = s or shims(env)
    real = {id(_dt): s["datetime_module"], id(_dt.datetime): s["datetime"], id(_dt.date): s["date"], id(_time): s["time_module"],
            id(_uuid): s["uuid_module"], id(_random): s["random_module"]}
    for name, fn in s["time_funcs"].items():
        real[id(getattr(_time, name))] = fn
    for name, fn in s["uuid_funcs"].items():
        real[id(getattr(_uuid, name))] = fn
    for name, fn in s["random_funcs"].items():
        real[id(getattr(_random, name))] = fn
    patched = []
    for k, v in list(vars(module).items()):
        if not k.startswith("__") and id(v) in real:
            setattr(module, k, real[id(v)])
            patched.append(k)
    return patched
