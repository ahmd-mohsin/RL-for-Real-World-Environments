"""Load pinned upstream framework source files **unchanged** for contract tests (M12 audit, bridges).

verl, TRL, SkyRL and rLLM are not installed in the Crucible venv; their pinned checkouts live under
``RL-Environments/06-frameworks-and-env-code`` (verl 5eb4826, TRL 72cf37d, SkyRL 3d0ae22, rLLM 4e828c8). ``upstream(...)``
executes the named upstream modules from those files as real modules (``sys.modules`` names intact, so their own
``from verl.x import y`` lines resolve), while every *other* module they import that is missing (or listed in
``stub``) becomes an inert stub: attribute access returns a stub class, calling one with a single callable returns it
(decorators), subscripting returns it (typing). Only modules this loader created are removed on exit; the code under test
is always the upstream text itself.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

FRAMEWORKS = Path(os.environ.get("CRUCIBLE_FRAMEWORKS_DIR") or Path(__file__).resolve().parents[3] / "RL-Environments" / "06-frameworks-and-env-code")
ROOTS = {"verl": "infra-volcengine-verl", "trl": "infra-huggingface-trl", "skyrl": "infra-novasky-ai-skyrl", "rllm": "infra-rllm-org-rllm",
         "verifiers": "primeintellect-ai-verifiers"}


class _StubMeta(type):
    def __getattr__(cls, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _stub_class(f"{cls.__name__}.{name}")

    def __call__(cls, *a, **k):
        if cls.__dict__.get("_upstream_stub"):
            if len(a) == 1 and not k and (callable(a[0]) or isinstance(a[0], type)):
                return a[0]  # used as a decorator
            return _stub_class(f"{cls.__name__}()")
        return super().__call__(*a, **k)

    def __getitem__(cls, item):
        return cls

    def __or__(cls, other):
        return cls

    def __ror__(cls, other):
        return cls

    def __iter__(cls):
        return iter(())

    def __len__(cls):
        return 0


def _stub_class(name: str) -> type:
    return _StubMeta(name.rsplit(".", 1)[-1] or "Stub", (), {"_upstream_stub": True, "__qualname__": name})


class _StubModule(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        v = _stub_class(f"{self.__name__}.{name}")
        setattr(self, name, v)
        return v


class _Loader(importlib.abc.Loader):
    def __init__(self, kind: str, path: Path | None = None):
        self.kind, self.path = kind, path

    def create_module(self, spec):
        if self.kind == "stub":
            m = _StubModule(spec.name)
            m.__path__ = []  # a package: submodules are stubs too
            return m
        return types.ModuleType(spec.name)

    def exec_module(self, module) -> None:
        if self.kind == "package":
            module.__path__ = [str(self.path)]
        elif self.kind == "file":
            module.__file__ = str(self.path)
            code = compile(self.path.read_text(), str(self.path), "exec")
            exec(code, module.__dict__)  # noqa: S102 — runs the pinned upstream source file unchanged (the point of these tests)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, framework: str, real: list[str], stub: tuple[str, ...]):
        self.root = FRAMEWORKS / ROOTS[framework]
        self.top = framework
        self.real = {m: self._file(m) for m in real}
        self.stub = stub
        self.created: list[str] = []

    def _file(self, mod: str) -> Path:
        p = self.root.joinpath(*mod.split("."))
        return p / "__init__.py" if p.is_dir() else p.with_suffix(".py")

    def _from_upstream(self) -> bool:
        """Only an upstream file's own imports are stubbed (a library's ``try: import _winapi`` must still fail)."""
        f = sys._getframe(2)
        while f is not None and f.f_code.co_filename.startswith("<frozen importlib"):
            f = f.f_back
        return f is not None and f.f_code.co_filename.startswith(str(FRAMEWORKS))

    def _findable(self, fullname: str) -> bool:
        for f in sys.meta_path:
            if f is not self and hasattr(f, "find_spec"):
                with contextlib.suppress(ImportError, ValueError):
                    if f.find_spec(fullname, None) is not None:
                        return True
        return False

    def find_spec(self, fullname, path=None, target=None):
        spec = None
        if fullname in self.real:
            f = self.real[fullname]
            spec = importlib.machinery.ModuleSpec(fullname, _Loader("file", f), origin=str(f), is_package=f.name == "__init__.py")
            if f.name == "__init__.py":
                spec.submodule_search_locations = [str(f.parent)]
        elif fullname == self.top or fullname.startswith(self.top + "."):
            d = self.root.joinpath(*fullname.split("."))
            if d.is_dir() and any(r.startswith(fullname + ".") for r in self.real):
                spec = importlib.machinery.ModuleSpec(fullname, _Loader("package", d), is_package=True)
                spec.submodule_search_locations = [str(d)]
            else:
                spec = importlib.machinery.ModuleSpec(fullname, _Loader("stub"), is_package=True)
        elif self._from_upstream() and (fullname.split(".")[0] in self.stub or (path is None and not self._findable(fullname))) or "." in fullname and isinstance(sys.modules.get(fullname.rsplit(".", 1)[0]), _StubModule):
            spec = importlib.machinery.ModuleSpec(fullname, _Loader("stub"), is_package=True)
        if spec is not None:
            self.created.append(fullname)
        return spec


@contextlib.contextmanager
def upstream(framework: str, real: list[str], *, stub: tuple[str, ...] = ()) -> Iterator[types.SimpleNamespace]:
    """Import ``real`` (dotted module names under the framework's pinned checkout) from the upstream files; yields
    ``ns.<last name>`` per module. Skips the test (with the reason) when the checkout is absent."""
    root = FRAMEWORKS / ROOTS[framework]
    if not root.is_dir():
        pytest.skip(f"upstream checkout {root} absent (set CRUCIBLE_FRAMEWORKS_DIR)")
    fnd = _Finder(framework, real, stub)
    clash = [k for k in list(sys.modules) if k == framework or k.startswith(framework + ".")]
    saved = {k: sys.modules.pop(k) for k in clash}
    sys.meta_path.insert(0, fnd)
    try:
        ns = types.SimpleNamespace()
        for m in real:
            mod = importlib.import_module(m)
            setattr(ns, m.rsplit(".", 1)[-1], mod)
        yield ns
    finally:
        sys.meta_path.remove(fnd)
        for k in fnd.created:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
