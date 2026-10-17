"""``pack.yaml``: a domain pack's manifest (ARCHITECTURE §4 M14; M15 ``crucible pack new / check``).

The shape follows NeMo-Gym's ``EnvironmentManifest`` (a pydantic model with a cross-field contract and an exported JSON
Schema) and OpenEnv's ``openenv.yaml`` (reward range, determinism, capabilities). Hooks are ``module:attr`` strings
resolved against the pack's ``src/`` and must live in the pack's own package (``pack check`` refuses foreign hooks). Every
studio command reads them, so the core imports no pack.

Every field is honoured somewhere (M14 audit V12): ``reward.range`` bounds every graded reward in ``pack check``;
``higher_is_better`` orients gold (the best end), no-op (the worst end) and adversary bounds; ``label.threshold`` (inside
the range) is what ``label`` and ``bench run`` count as solved; ``determinism: seeded`` makes ``pack check`` re-grade
on fresh Worlds and rebuild tasks; ``requires`` is detected (docker, gpu) or vouched for (``--allow``) and a level that
cannot run is a loud skip.

An adversary (``{"policy", "expect_max"}``) is a policy the grader must keep low: a ``PolicyRef``, the id of one of the
pack's ``hooks.policies``, or a callable ``task -> actions``. ``pack check --level semantic`` runs it on every task and on
generated items; its reward must not be better than ``expect_max`` (``<=`` when higher is better, ``>=`` otherwise).

| hook | signature | used by |
|---|---|---|
| ``credit`` | ``async (traces) -> {trace_id: {"reward": float, "phi_by_seq": {seq: Φ}, "reward_json", "split", "group"?, "phi_end_by_milestone"?}}`` | ``label`` (dense credit, released per channel on the hook's held-out traces: ``split`` ≠ ``train``; ``reward_json`` is required with partial credit; ``credit.channels``) |
| ``tasks`` | ``() -> [(RunTask, gold actions)]`` | ``pack check`` (gold passes, no-op fails), ``bench`` anchors |
| ``generator`` | ``(seed: int) -> RunTask`` | ``bench`` fresh items |
| ``stages`` | ``(task_id) -> [(stage, [milestone ids])]`` | ``bench`` stage reach |
| ``forge`` | ``async (out_dir, store_dir) -> [(task, resolve)]`` | ``build`` |
| ``policies`` | ``() -> [PolicyRef | {"policy": PolicyRef, "expect_max": float}]`` | ``bench`` / ``build`` without a model; bounded ones are adversaries |
| ``fixtures`` | ``() -> [{"name", "task", "actions", "reward", "fails_gate"?} | {"name", "policy", "expect_max"}]`` | ``pack check --level fixture`` (exact expected grades); ``policy`` entries are adversaries for ``--level semantic`` |
| ``model`` | ``async (traces) -> {"milestones": [...], "stages": [...], ...}`` | ``crucible model`` (the M5 workflow model) |
"""

from __future__ import annotations

import math
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal

from pydantic import Field, model_validator

from crucible.core.schema.base import CrucibleModel

NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")  # a pack name (``pack new`` and the manifest agree, M14 audit V10)
PACKAGE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VERSION = re.compile(r"^v?\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?(\+[A-Za-z0-9.]+)?$")  # PEP 440 public versions (+ local)


def _absolute(p: str) -> bool:
    return PurePosixPath(p).is_absolute() or PureWindowsPath(p).is_absolute() or p.startswith("~")


class IngestSpec(CrucibleModel):
    """One M3 source. ``connector`` is any registered connector (``crucible registry``). ``file_drop`` takes ``glob`` (+
    ``extra`` fields); every other connector is constructed with ``init`` and discovered with ``scope`` through its
    ``discover`` method (e.g. ``discover_git``). String values in ``init`` / ``scope`` / ``root`` that start with ``./`` or
    ``../`` are resolved against the pack root, so a pack can point at data outside itself without absolute paths.
    ``redaction_env``: a dotenv file whose credential values are redacted as well (``RedactionPolicy.with_env_file``)."""

    connector: str = "file_drop"
    family: str
    glob: str | None = None  # file_drop: relative to ``root`` (default: the pack root)
    root: str | None = None
    extra: list[str] = Field(default_factory=list)
    init: dict[str, Any] = Field(default_factory=dict)
    scope: dict[str, Any] = Field(default_factory=dict)
    discover: str = "discover"
    redaction_env: str | None = None
    # a declared ``redaction_env`` must exist (M14 audit V5: a misspelled one stored the secret it was meant to remove);
    # a pack whose dotenv is a developer convenience says so here, and ``label`` records that no values were loaded
    redaction_env_optional: bool = False
    label: bool = True  # labelled by ``crucible label`` (sources like policy documents are context, not episodes)

    @model_validator(mode="after")
    def _file_drop_needs_glob(self) -> IngestSpec:
        if self.connector == "file_drop" and not self.glob:
            raise ValueError("a file_drop ingest source needs a glob")
        # paths are pack-relative and confined (M14 audit V17): an absolute root or glob reads arbitrary host files
        for k in ("glob", "root", "redaction_env"):
            v = getattr(self, k)
            if v is not None and _absolute(v):
                raise ValueError(f"ingest {self.family}: {k} {v!r} is absolute; give a path relative to the pack")
        # fields the connector kind ignores are refused, not silently dropped (M14 audit V20)
        if self.connector == "file_drop":
            unused = [k for k, bad in (("init", bool(self.init)), ("scope", bool(self.scope)), ("discover", self.discover != "discover")) if bad]
        else:
            unused = [k for k, bad in (("glob", self.glob is not None), ("root", self.root is not None), ("extra", bool(self.extra))) if bad]
        if unused:
            raise ValueError(f"ingest {self.family}: connector {self.connector!r} does not use {', '.join(unused)}")
        return self


class RewardSpec(CrucibleModel):
    range: tuple[float, float] = (0.0, 1.0)
    higher_is_better: bool = True


class LabelSpec(CrucibleModel):
    policy_roles: list[str] = Field(default_factory=lambda: ["agent"])  # whose actions are the policy's (assistant turns)
    threshold: float = 1.0  # reward at or above which an episode counts as solved


class PackHooks(CrucibleModel):
    credit: str | None = None
    tasks: str | None = None
    generator: str | None = None
    stages: str | None = None
    forge: str | None = None
    policies: str | None = None
    fixtures: str | None = None
    model: str | None = None


class PackManifest(CrucibleModel):
    schema_version: Literal[1] = 1
    name: str
    version: str
    description: str
    license: str
    authors: list[str] = Field(default_factory=list)
    kind: Literal["domain", "coding", "toy"] = "domain"
    package: str  # the import name under src/
    entry_point: str  # module:register (the crucible.plugins entry point)
    template_version: int | None = None  # the _template version the pack was scaffolded from (a later 3-way upgrade)
    reward: RewardSpec = Field(default_factory=RewardSpec)
    determinism: Literal["seeded", "nondeterministic"] = "seeded"
    requires: list[Literal["docker", "gpu", "network", "paid"]] = Field(default_factory=list)  # what running its tasks needs (semantic checks skip without it)
    ingest: list[IngestSpec] = Field(default_factory=list)
    label: LabelSpec = Field(default_factory=LabelSpec)
    hooks: PackHooks = Field(default_factory=PackHooks)

    @model_validator(mode="before")
    @classmethod
    def _newer_version(cls, data: Any) -> Any:
        """The versioned-document rule for pack.yaml's integer version: a newer one is refused by name (R-18)."""
        v = data.get("schema_version") if isinstance(data, dict) else None
        if isinstance(v, int) and not isinstance(v, bool) and v > 1:
            raise ValueError(f"pack.yaml schema_version {v} was written for a newer Crucible (this one reads 1): upgrade Crucible")
        return data

    @model_validator(mode="after")
    def _contract(self) -> PackManifest:
        lo, hi = self.reward.range
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"reward.range must be finite, got {self.reward.range}")
        if not lo < hi:
            raise ValueError(f"reward.range must be increasing, got {self.reward.range}")
        # identity (M14 audit V10, G2 D12): the manifest is validated, not only `pack new`'s argument
        if not NAME.match(self.name):
            raise ValueError(f"name {self.name!r}: lowercase letters, digits and _, starting with a letter (2-41 characters)")
        if not VERSION.match(self.version):
            raise ValueError(f"version {self.version!r} is not a PEP 440 version (e.g. 0.1.0)")
        if not PACKAGE.match(self.package):
            raise ValueError(f"package {self.package!r} is not an importable top-level name")
        if not (lo <= self.label.threshold <= hi):  # M14 audit V12: a threshold outside the range marks nothing solved
            raise ValueError(f"label.threshold {self.label.threshold} is outside reward.range {list(self.reward.range)}")
        if len(set(self.requires)) != len(self.requires):
            raise ValueError(f"requires lists a requirement twice: {self.requires}")
        if self.hooks.credit and not self.ingest:
            raise ValueError("hooks.credit labels ingested traces: declare at least one ingest source")
        if ":" not in self.entry_point:
            raise ValueError("entry_point must be module:attr")
        ep_mod = self.entry_point.split(":", 1)[0]
        if ep_mod != self.package and not ep_mod.startswith(self.package + "."):
            raise ValueError(f"entry_point {self.entry_point!r} is outside the pack's package {self.package!r}")
        for k, v in self.hooks.model_dump().items():
            if v is not None and ":" not in v:
                raise ValueError(f"hooks.{k} must be module:attr, got {v!r}")
        return self
