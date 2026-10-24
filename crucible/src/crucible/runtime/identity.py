"""The run identity, v2 (M10 lab audit R-4): a resolved document of everything that decides what a run measures, compared leaf
by leaf on a resume, so a changed configuration is refused by name instead of resuming silently into mixed results.

Before R-4 a run's identity was a hash of ids: ``native:llm-tool@1`` says nothing of ``max_steps``, ``RunSpec.sampling`` was
not hashed and a grader edited between two runs resumed as the same run (a partial run mixed the two configurations; a
finished one dispatched nothing and the report stated the new configuration for the old results).

- **Resolved document.** ``describe(obj, where)`` turns every task, policy and harness of a ``RunSpec`` into JSON: what the
  object declares through ``identity_config()`` (its configuration and the *content* digests of what it grades with: a
  Harbor task tree, a pack tree, an image reference, a patch file), or, for plain data, what it holds (a scalar, a
  mapping, a dataclass's fields, a pydantic model, a ``SimpleNamespace``, a stateless object as its type). Anything else
  (a closure, a callable, an object with state and no declaration) raises ``Undeclared`` naming its dotpath: fail closed, so
  a new knob nobody declared is refused instead of left out. Runtime never imports the layers above it, so an adapter
  declares its own document through this duck-typed method (a ``PolicyRef`` through ``config=``).
- **Comparison.** ``compared(doc)`` drops an operational denylist (``DENY``: concurrency, budgets, wall clocks, timeouts,
  backoff, health thresholds, container TTLs and the sweep, a few placement keys (docker and GPU host, sandbox id), paths); every other leaf counts, an
  endpoint (a URL, a host) included where it names the grader, the model, the agent or the hardware that is measured.
  ``diff`` lists the dotpaths that differ (``harnesses.native:llm-tool@1.limits.max_steps: 30 -> 100``). The denylist is
  matched below the id of a task, policy or harness (``tasks.<id>.*``): an id never decides whether its configuration is
  compared. Content-digest maps (``files``, ``patches``) are never filtered: a task file named ``token`` or a directory
  named ``root`` is still content.
- **Stored form.** ``normalize(doc)`` is what ``run.json`` keeps and what is compared: a value under a credential-named key
  (``api_key``, ``HF_TOKEN``, ``password``) is replaced by ``<secret>`` (a rotation is not drift, the credential is never
  written), and any other credential-shaped text by ``<secret#<12 hex of its sha256>>``, so a changed value is still seen
  without storing it (a low-entropy value's digest can be guessed offline: the run directory is as private as the prompts
  in it).
- **Epochs.** A mismatch refuses (``IdentityDrift``), unless every changed path is accepted (``RunSpec.accept_drift``), which
  opens a new identity epoch in ``run.json``; records written in an epoch after the first carry it (``meta.epoch``).
  Code versions (``code_versions``: crucible version, git sha and dirty flag, Harbor and verifiers versions) are recorded and
  open an epoch when they differ but never refuse: a hotfix can still resume (Harbor's choice).

Patterns: NeMo-RL ``rollout_checkpoint.py`` (fail-closed fingerprint over configuration leaves, an operational denylist, a
dotpath diff; Apache-2.0), Harbor ``lock.json`` (task-content digests compared; its own version recorded and not compared),
Repo2RLEnv ``run.json`` (the run's configuration, resumed only when identical). Imports stdlib, pydantic, ``crucible.core``
and ``crucible.runtime.keys`` only.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path, PurePath
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from crucible.core.secrets import scrub_secrets
from crucible.runtime.keys import identity

_MAX_STR = 1024  # a longer text (a prompt, a script) is a digest
_MAX_SEQ = 256  # a longer sequence is its length and a digest
_MAX_DEPTH = 40  # a cycle, or a structure nobody meant to describe
_MISSING = object()


class Undeclared(ValueError):
    """An object in a run's tasks, policies or harnesses that cannot be inspected and declares no ``identity_config()``
    (M10 lab audit R-4). A ValueError, so the CLI reports it as a user error; raised before the run directory is touched."""


class IdentityDrift(ValueError):
    """A run directory exists with a different configuration (M10 lab audit R-4): the dotpaths that differ are listed and
    the resume is refused. ``accepted``: the ``--accept-drift`` patterns the caller gave, which covered none of them.
    ``hint``: what to do instead of the generic advice, for a refusal the listed paths cannot be accepted by (a run written
    before identity v2 has nothing to compare; only ``*`` accepts it)."""

    def __init__(self, run_id: str, changes: list[str], *, accepted: Sequence[str] = (), hint: str | None = None):
        self.run_id, self.changes, self.accepted = run_id, list(changes), tuple(accepted)
        more = f" (+{len(changes) - 20} more)" if len(changes) > 20 else ""
        seen = f" (--accept-drift {', '.join(self.accepted)} does not cover these)" if self.accepted and hint is None else ""
        super().__init__(f"run {run_id} exists with a different configuration: {'; '.join(changes[:20])}{more}{seen}: resume refused (M10 lab audit R-4). "
                         + (hint or "Use a new run id (bench run: a different --out), or accept these paths with RunSpec.accept_drift / bench run --accept-drift "
                            "<dotpath>, which opens a new identity epoch (accepting a harness configuration change gives every sample a new key, "
                            "so they all run again in the new epoch)"))


# ------------------------------------------------------------------------------------------------- describing objects


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _where_type(obj: Any) -> str:
    return code_ref(obj) if callable(obj) else f"{type(obj).__module__}.{type(obj).__qualname__}"


def describe(obj: Any, where: str) -> Any:
    """The JSON document for ``obj`` (``where``: its dotpath in the run's document, named by ``Undeclared``). The first rule
    that matches applies:

    1. a class-level ``identity_config()``: its mapping is described in turn (looked up on the type, so a wrapper that
       delegates attributes with ``__getattr__`` never answers for the object it wraps, and a ``Mock`` has none);
    2. None, bool, int, str (a text over 1,024 characters becomes ``sha256:<16 hex>``), a finite float (else its repr);
    3. bytes (a ``sha256:`` digest), a path (posix text), an Enum (its value);
    4. a Mapping (string keys), a list or tuple, a set (sorted); a sequence over 256 items is its length and a digest;
    5. a dataclass instance (every field), a pydantic model (``model_dump``), a ``SimpleNamespace`` (its attributes);
    6. a non-callable object whose ``__dict__`` is empty: stateless, its behaviour is its code, which ``code_versions``
       records; described as its type;
    7. anything else (callables included): ``Undeclared``."""
    return _describe(obj, where, 0)


def _describe(obj: Any, where: str, depth: int) -> Any:
    if depth > _MAX_DEPTH:
        raise Undeclared(f"{where}: nested more than {_MAX_DEPTH} levels, or cyclic: it cannot be inspected for the run identity (M10 lab audit R-4)")
    hook = getattr(type(obj), "identity_config", None)
    if callable(hook):
        doc = obj.identity_config()
        if not isinstance(doc, Mapping):
            raise Undeclared(f"{where}: identity_config() must return a mapping, not {type(doc).__name__} (M10 lab audit R-4)")
        return _describe(dict(doc), where, depth + 1)
    if isinstance(obj, Enum):
        return _describe(obj.value, where, depth + 1)
    if obj is None or isinstance(obj, bool | int):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= _MAX_STR else "sha256:" + hashlib.sha256(obj.encode()).hexdigest()[:16]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else repr(obj)
    if isinstance(obj, bytes | bytearray):
        return "sha256:" + hashlib.sha256(bytes(obj)).hexdigest()[:16]
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, Mapping):
        return {str(k): _describe(v, f"{where}.{k}", depth + 1) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        items = [_describe(v, f"{where}.{i}", depth + 1) for i, v in enumerate(obj)]
        if isinstance(obj, set | frozenset):
            items.sort(key=lambda x: json.dumps(x, sort_keys=True, default=str))
        return items if len(items) <= _MAX_SEQ else {"len": len(items), "sha256": _sha(items)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _describe(getattr(obj, f.name), f"{where}.{f.name}", depth + 1) for f in dataclasses.fields(obj)}
    if isinstance(obj, BaseModel):
        return _describe(obj.model_dump(mode="json", exclude_none=True), where, depth + 1)
    if isinstance(obj, SimpleNamespace):
        return _describe(vars(obj), where, depth + 1)
    if not callable(obj) and hasattr(obj, "__dict__") and not vars(obj):
        return {"type": f"{type(obj).__module__}.{type(obj).__qualname__}"}
    raise Undeclared(f"{where}: a {_where_type(obj)} cannot be inspected for the run identity (M10 lab audit R-4): declare identity_config() returning its "
                     "configuration and content digests (a PolicyRef: config=...); left out, a change to it would resume silently")


def config_digest(obj: Any) -> str:
    """The 12 hex characters that follow ``#`` in a harness id: a digest of ``describe(obj)`` (the same configuration gives
    the same id in every process)."""
    return _sha(describe(obj, "config"))[:12]


def harness_digest(obj: Any) -> str:
    """The 12 hex characters that follow ``#`` in a harness id (M10 lab audit R-4 review): a digest of ``describe(obj)`` as
    the run's identity compares it (``normalize``, then ``compared``, under ``harnesses.<key>``), so the id and the
    comparison see the same leaves. A leaf the comparison drops (a path, a credential, a timeout) is not in the id either:
    otherwise changing it re-keyed every sample (they all ran again, paid) with no refusal and no epoch."""
    return _sha(compared(normalize({"harnesses": {"_": describe(obj, "config")}}))["harnesses"]["_"])[:12]


# ------------------------------------------------------------------------------------------------- content helpers


def code_ref(fn: Any) -> str | None:
    """``module.qualname`` of a function or class: how an adapter names a callable it holds (the callable's behaviour is its
    code, which ``code_versions`` records). None for None. It names, it does not inspect: for a factory that may be a
    ``functools.partial``, a lambda or a closure use ``factory_ref``."""
    if fn is None:
        return None
    return f"{getattr(fn, '__module__', None)}.{getattr(fn, '__qualname__', None) or type(fn).__qualname__}"


def factory_ref(fn: Any, where: str) -> Any:
    """``code_ref`` for a factory whose bound or captured configuration is part of what it produces (M10 lab audit R-4
    review): a class, a module-level function or a method is named; a ``functools.partial`` is its function's name with its
    bound arguments described (so ``partial(Agent, model="a")`` and ``partial(Agent, model="b")`` differ, and an argument that
    cannot be inspected is ``Undeclared``); a lambda or a closure, whose configuration lives in what it captured, raises
    ``Undeclared`` (``code_ref`` named it ``<locals>.<lambda>``, the same for every one made by the same function); any other
    callable object is described (its ``identity_config()``, or stateless as its type). ``where``: the dotpath ``Undeclared`` names."""
    if fn is None:
        return None
    if isinstance(fn, functools.partial):
        return {"ref": factory_ref(fn.func, where), "args": describe(list(fn.args), f"{where}.args"), "keywords": describe(dict(fn.keywords or {}), f"{where}.keywords")}
    if inspect.isfunction(fn):
        if fn.__closure__ or fn.__name__ == "<lambda>":
            raise Undeclared(f"{where}: a lambda or a closure ({code_ref(fn)}) holds configuration that cannot be inspected for the run identity (M10 lab audit R-4): "
                             "pass a class or a module-level function (functools.partial with described arguments, or the agent's settings in agent_config)")
        return code_ref(fn)
    if isinstance(fn, type) or inspect.isroutine(fn):
        return code_ref(fn)
    return describe(fn, where)


def file_digest(path: str | Path | None) -> str | None:
    """sha256 of the file's bytes, None when it is missing or unreadable."""
    if path is None:
        return None
    h = hashlib.sha256()
    try:
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def seed_digest(seed: Any) -> str:
    """A one-way stand-in for a seed in an identity document (M10 lab audit R-4 review). ``run.json`` is written under the
    bench's out directory, which is shared; the seed of a held-out item is an HMAC of the secret bench key (``bench.dynamic``)
    and the pack's generator is public, so a seed there would let anyone rebuild the item and its answer. A changed seed is
    still a changed digest."""
    return "sha256:" + hashlib.sha256(f"crucible-identity-seed|{seed}".encode()).hexdigest()[:16]


def _skipped(rel: Path) -> bool:
    """``studio.runs.pack_hash``'s rule, for a pack tree: dotfiles and dotdirs (VCS), bytecode caches and egg-info are not content."""
    return any(part.startswith(".") or part == "__pycache__" or part.endswith(".egg-info") for part in rel.parts)


_VCS_DIRS = (".git", ".hg", ".svn")
_JUNK_SUFFIXES = (".pyc", ".swp", ".swo", "~", ".egg-info")  # Harbor's DEFAULT_IGNORES (bytecode, editor swap and backup files), egg-info


def _task_skipped(rel: Path) -> bool:
    """What a task tree leaves out (M10 lab audit R-4 review): version control, bytecode caches, editor and OS junk, as Harbor's
    ``Packager.collect_files`` does, and nothing else. A dotfile is content here (``environment/.dockerignore`` decides what
    is built, ``tests/.expected_output.json`` what is graded), unlike in a pack tree (``_skipped``)."""
    return any(part in _VCS_DIRS or part in ("__pycache__", ".DS_Store") or part.endswith(_JUNK_SUFFIXES) for part in rel.parts)


_TREES: dict[tuple[Path, bool], tuple[tuple, str]] = {}  # (resolved root, task mode) -> (stat signature, digest)


def tree_digest(path: str | Path | None, *, task: bool = False) -> str | None:
    """sha256 over the sorted (relative posix path, sha256 of bytes) of the files under ``path``, None when it is not a
    directory. ``task``: a task's tree (``_task_skipped``: only VCS dirs and junk are left out, dotfiles count), else a pack's
    (``_skipped``: every dotfile is left out). Memoized per resolved path and mode on the files' stat signature ((path, size,
    mtime_ns) of each), so 1,000 tasks of one pack read its bytes once per process; an edit that keeps both size and mtime
    within the filesystem's resolution is missed until the process restarts."""
    if path is None:
        return None
    root = Path(path)
    if not root.is_dir():
        return None
    root = root.resolve()
    skipped = _task_skipped if task else _skipped
    files: list[tuple[Path, os.stat_result]] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if skipped(rel):
            continue
        try:
            if p.is_file():
                files.append((rel, p.stat()))
        except OSError:
            continue
    sig = tuple((rel.as_posix(), st.st_size, st.st_mtime_ns) for rel, st in files)
    memo = _TREES.get((root, task))
    if memo is not None and memo[0] == sig:
        return memo[1]
    h = hashlib.sha256()
    for rel, _ in files:
        h.update(rel.as_posix().encode() + b"\0")
        h.update((file_digest(root / rel) or "<unreadable>").encode())
    digest = h.hexdigest()
    _TREES[(root, task)] = (sig, digest)
    return digest


def tree_digests(path: str | Path | None) -> dict[str, str | None]:
    """One digest per top-level entry of a task directory (a file's, or a directory's tree; ``tree_digest`` in task mode:
    hidden files are content, version control and junk are not), so a changed-content refusal names the part that changed
    (``files.tests``, ``files.environment``, ``files.task.toml``) instead of one opaque hash. Harbor's task hash covers the
    same files (it also honours the task's ``.gitignore``; this digests a little more, never less). Empty for a path that is
    not a directory."""
    if path is None or not Path(path).is_dir():
        return {}
    out: dict[str, str | None] = {}
    for p in sorted(Path(path).iterdir()):
        if _task_skipped(Path(p.name)):
            continue
        out[p.name] = tree_digest(p, task=True) if p.is_dir() else file_digest(p)
    return out


def harness_keys(harnesses: Iterable[Any]) -> list[str]:
    """The key of each harness in the resolved document: ``identity_key`` when it has one (``name@version``, which an RLM
    harness keeps free of its own config digest), else its id without the ``#digest`` suffix, so a configuration change is
    a leaf under the same key (named by dotpath), not one harness removed and another added. When two keys collide the full
    ids are used for all."""
    hs = list(harnesses)
    keys = [str(getattr(h, "identity_key", None) or str(h.id).split("#", 1)[0]) for h in hs]
    return keys if len(set(keys)) == len(keys) else [str(h.id) for h in hs]


# ------------------------------------------------------------------------------------------------- comparison

# The operational denylist: patterns over dotpaths (``fnmatch``: ``*`` spans dots) of leaves that say how or where a run is
# executed, not what it measures. A change to any of them resumes without a word. Everything else is compared.
# ``EpisodeLimits.timeout_s`` and an agent's own ``timeout_s`` are limits the policy lives under (a hung episode is scored or
# masked by them), so they are NOT here; the grading budgets and the attempt clock are operational. An endpoint (``url``,
# ``host``, ``base_url``) is NOT here either: where it names the grader (a NeMo-Gym server), the model server, the agent (an
# A2A agent) or the hardware a perf grade is measured on, it is what the run measures. An adapter leaves out the ones that
# are only this process's (a loopback port it started itself). Credentials are not here: ``normalize`` replaces them.
# The patterns are matched below the id of a task, policy or harness (``_child``): ``tasks.<id>`` is ``tasks._``.
DENY: tuple[str, ...] = (
    # run-level operational fields (RunSpec)
    "concurrency", "budget_usd", "wall_clock_s", "attempt_timeout_s", "retry_backoff_s",
    # M10 lab audit R-5: the backoff's caps (RetryRule): how long a retry waits, not what is measured
    "retry.backoff_cap_s", "retry.retry_after_cap_s",
    # M10 lab audit R-5: the breakers' thresholds and cool-downs (``RunSpec.breaker``, whole subtree): when a run pauses for an outage,
    # not what it measures
    "breaker",
    # M10 lab audit R-6: container TTLs and the sweep (RunSpec.hygiene, whole subtree): how long an orphan lives, not what is measured
    "hygiene",
    # health thresholds: they decide when to stop, not what a sample is
    "max_mask_fraction", "max_unknown_fraction", "min_terminal_for_health",
    # timeouts and budgets of grading, tools and the harness's own housekeeping; a price is accounting
    "*.scoring_budget_s", "*.control_timeout_s", "*.tool_timeout_s", "*.timeout", "*.estimate_usd", "*.prices", "*.self_hosted", "*.progress_full_every",
    # placement: which docker daemon, which sandbox
    "*.docker_host", "*.gpu_host", "*.keep_containers", "*.sandbox_id",
    # paths and directories (the content under them is digested by the adapter that owns it)
    "*.dir", "*_dir", "*.path", "*_path", "*.root", "*_root", "*.workdir",
)
_DIGEST_MAPS = (".files.", ".patches.")  # content digests an adapter declares: never dropped, whatever their file names are
_ID_MAPS = ("tasks", "policies", "harnesses")  # the resolved document's maps keyed by an object's id (a harness: its key)
# a key that names a credential, in any case (``api_key``, ``OPENAI_API_KEY``, ``HF_TOKEN``, ``password``): NeMo-RL's rule that a
# credential is never part of an identity. ``eos_token`` and ``max_tokens`` are not credentials
_CRED_KEY = re.compile(r"(?:^|[_.-])(?:api[_-]?key|password|passwd|secret|secret[_-]?key|private[_-]?key|access[_-]?key|credentials?)$"
                       r"|(?:^|[_.-])(?:access|auth|api|bearer|hf|id|refresh|session)[_-]?token$|^token$", re.IGNORECASE)
_REDACTED = "<secret>"


def _child(path: str, key: Any) -> str:
    """The dotpath of ``path``'s child ``key`` as the denylist and the digest-map rule see it: the id level under ``tasks``,
    ``policies`` and ``harnesses`` is the constant ``_``, so what an object is called (``cleanup_build_dir``, ``llm:reset_password``,
    a name with a dot) never decides whether its configuration is compared (M10 lab audit R-4 review)."""
    if path in _ID_MAPS:
        return f"{path}._"
    return f"{path}.{key}" if path else str(key)


def _keyed(value: str) -> str:
    """What stands for credential-shaped text in the stored document: a digest of it, so a changed value is a changed
    document. The ``#`` keeps it from matching ``core.secrets``' assignment rule again."""
    return f"<secret#{hashlib.sha256(value.encode()).hexdigest()[:12]}>"


def normalize(doc: Any) -> dict[str, Any]:
    """The document as it is stored and compared: JSON round-tripped (tuples are lists, odd values their text), the value of
    every credential-named key (``_CRED_KEY``) replaced by ``<secret>`` (a rotated key is not drift, and the credential is
    never written to ``run.json``; a key under a content-digest map, a file named ``token``, is content and kept), and every
    other credential-shaped text scrubbed to a keyed digest (``core.secrets``: ``password=<secret#ab12...>``), not to a
    constant, so an edited prompt or answer that happens to match the patterns is still a changed document (M10 lab audit R-4
    review)."""

    def walk(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                p = _child(path, k)
                key = scrub_secrets(k, _keyed) if isinstance(k, str) else k
                # an id under ``tasks``, ``policies`` or ``harnesses`` names an object, never a credential: ``llm:reset_password`` keeps its configuration
                if (isinstance(k, str) and path not in _ID_MAPS and v not in (None, "") and _CRED_KEY.search(k)
                        and not any(m in p for m in _DIGEST_MAPS)):
                    out[key] = _REDACTED
                else:
                    out[key] = walk(v, p)
            return out
        if isinstance(node, list):
            return [walk(v, f"{path}.{i}") for i, v in enumerate(node)]
        return scrub_secrets(node, _keyed) if isinstance(node, str) else node

    return walk(json.loads(json.dumps(doc, default=str)), "")


def _denied(path: str, deny: Sequence[str]) -> bool:
    return not any(m in path for m in _DIGEST_MAPS) and any(fnmatchcase(path, pat) for pat in deny)


def compared(doc: Any, deny: Sequence[str] = DENY) -> Any:
    """``doc`` without the leaves (or subtrees) whose dotpath (below the id of a task, policy or harness: ``_child``) matches
    ``deny``."""

    def walk(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                p = _child(path, k)
                if not _denied(p, deny):
                    out[k] = walk(v, p)
            return out
        if isinstance(node, list):
            return [walk(v, f"{path}.{i}") for i, v in enumerate(node)]
        return node

    return walk(doc, "")


def _show(v: Any) -> str:
    return "<missing>" if v is _MISSING else f"{v!r:.160}"


def diff(old: Any, new: Any, path: str = "") -> list[tuple[str, str]]:
    """The leaves that differ between two documents as (dotpath, ``'<path>: <old> -> <new>'``), sorted by path: mappings and
    lists are walked (a list element's segment is its index), a side that lacks the key reads ``<missing>`` (NeMo-RL's
    dotpath diff)."""
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[tuple[str, str]] = []
        for k in sorted(set(old) | set(new), key=str):
            p = f"{path}.{k}" if path else str(k)
            a, b = old.get(k, _MISSING), new.get(k, _MISSING)
            out += [(p, f"{p}: {_show(a)} -> {_show(b)}")] if a is _MISSING or b is _MISSING else diff(a, b, p)
        return out
    if isinstance(old, list) and isinstance(new, list):
        out = []
        for i in range(max(len(old), len(new))):
            p = f"{path}.{i}" if path else str(i)
            a, b = (old[i] if i < len(old) else _MISSING), (new[i] if i < len(new) else _MISSING)
            out += [(p, f"{p}: {_show(a)} -> {_show(b)}")] if a is _MISSING or b is _MISSING else diff(a, b, p)
        return out
    return [] if old == new else [(path, f"{path}: {_show(old)} -> {_show(new)}")]


def changes(old: Any, new: Any) -> list[str]:
    """``diff`` as the lines ``IdentityDrift`` lists."""
    return [text for _, text in diff(old, new)]


def accepts(path: str, patterns: Iterable[str]) -> bool:
    """Whether ``--accept-drift`` covers ``path``: an exact path, a prefix at a dot boundary (a whole subtree), or an
    ``fnmatch`` pattern (``policies.*.config.temperature``; a ``[`` in an id needs escaping as ``[[]``)."""
    return any(path == p or path.startswith(p + ".") or fnmatchcase(path, p) for p in patterns)


def fingerprint(doc: Any) -> str:
    """The identity string of a resolved document: a digest of its compared leaves."""
    return identity(compared(normalize(doc)))


# ------------------------------------------------------------------------------------------------- code versions


def _git(cwd: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=False, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def dist_source(name: str) -> dict[str, Any]:
    """An installed distribution's ``version`` and the ``commit`` it was installed from (M11 lab audit R-3): PEP 610's
    ``direct_url.json`` names it for a VCS install (``vcs_info.commit_id``); for an install from a local directory (editable
    or not) that is the root of its own git checkout, it is that checkout's ``HEAD``, and ``dirty`` says whether tracked
    files differ from it. A shallow editable clone of a hatch-vcs project reports version 0.0.0 (verifiers
    ``pyproject.toml:169``), so the commit is what identifies it. None for what cannot be told; nothing is imported."""
    out: dict[str, Any] = {"version": None, "commit": None, "editable": None, "dirty": None}
    try:
        d = importlib.metadata.distribution(name)
    except Exception:  # noqa: BLE001 -- not installed (or unreadable metadata): not told
        return out
    out["version"] = d.version
    try:
        info = json.loads(d.read_text("direct_url.json") or "null") or {}
    except (OSError, ValueError):
        info = {}
    if not isinstance(info, dict):
        return out
    out["commit"] = (info.get("vcs_info") or {}).get("commit_id")
    if isinstance(info.get("dir_info"), dict):
        out["editable"] = bool(info["dir_info"].get("editable", False))
    url = str(info.get("url") or "")
    if out["commit"] is None and url.startswith("file://"):
        from urllib.parse import urlparse
        from urllib.request import url2pathname

        local = Path(url2pathname(urlparse(url).path))  # url2pathname unquotes (once)
        top = _git(local, "rev-parse", "--show-toplevel") if local.is_dir() else None
        # the checkout's own repository only: a directory inside another one (crucible/.envs/src) would report that one's HEAD
        if top is not None and Path(top).resolve() == local.resolve():
            out["commit"] = _git(local, "rev-parse", "HEAD")
            status = _git(local, "status", "--porcelain", "--untracked-files=no")
            out["dirty"] = None if status is None else bool(status)
    return out


@functools.cache
def code_versions() -> dict[str, Any]:
    """What code the process runs, recorded in ``run.json`` and compared only to open an identity epoch (never to refuse):
    the crucible version, the git sha and dirty flag of the checkout it was loaded from, the installed Harbor and verifiers
    versions (distribution metadata: no framework is imported), and the verifiers commit (``dist_source``: an editable clone
    reports 0.0.0; M11 lab audit R-3). Cached for the life of the process, which is what the loaded modules are; a value that
    cannot be told is None. Callers take it through the module attribute, so a test can patch it."""

    def dist(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except Exception:  # noqa: BLE001 -- not installed (or unreadable metadata): not told
            return None

    here = Path(__file__).resolve().parent
    sha = _git(here, "rev-parse", "HEAD")
    status = _git(here, "status", "--porcelain", "--untracked-files=no") if sha else None
    return {"crucible": dist("crucible"), "git_sha": sha or None, "git_dirty": None if status is None else bool(status),
            "harbor": dist("harbor"), "verifiers": dist("verifiers"), "verifiers_commit": dist_source("verifiers")["commit"]}
