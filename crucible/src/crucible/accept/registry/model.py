"""Registry entries: ``defects/*.yaml``, one file per defect.

- ``incident``: when, where and how it was found, with provenance (log lines, DESIGN Appendix B rows).
- ``detector``: the check (and sub-measurement) that catches it. ``per_task`` detectors run inside acceptance on every task
  they apply to; ``per_kit`` detectors are red-team / unit tests that run once per kit version (``test`` names them).
- ``reinjection``: the detector's power test. ``fixture`` is ``module:function`` returning ``(clean, defective)``
  callables or task adapters; the detector must pass on the clean one and fail on the defective one. A detector that
  cannot be shown to fire is not admitted (``admitted()``).
- ``status``: open | fixed | design_limit.

``Registry.version()`` hashes every entry and is part of every acceptance key: adding a defect makes every acceptance
non-current until the backfill has run the new detector.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from crucible.core.schema.base import CrucibleModel

DEFECTS = Path(__file__).parent / "defects"


class Incident(CrucibleModel):
    date: str
    task: str | None = None
    found_by: str
    summary: str
    provenance: list[str] = Field(default_factory=list)


class Detector(CrucibleModel):
    check: str  # "A4" or "A4.reference_self_consistency"
    scope: Literal["per_task", "per_kit"] = "per_task"
    applies_to: dict[str, Any] = Field(default_factory=dict)  # selector over task facts, e.g. {has: [perf_bars]}
    test: str | None = None  # per_kit: the pytest node id that exercises it
    params: dict[str, Any] = Field(default_factory=dict)


class Reinjection(CrucibleModel):
    fixture: str  # module:function -> (clean, defective)
    expect: dict[str, str] = Field(default_factory=lambda: {"clean": "pass", "defective": "fail"})
    needs: list[str] = Field(default_factory=list)


class Fix(CrucibleModel):
    kind: Literal["task", "kit", "runtime", "world", "forge", "none"] = "task"
    ref: str = ""


class Defect(CrucibleModel):
    id: str
    incident: Incident
    detector: Detector
    fix: Fix = Field(default_factory=Fix)
    reinjection: Reinjection | None = None
    status: Literal["open", "fixed", "design_limit"] = "fixed"

    def admitted(self) -> bool:
        """Admitted only with a power test (reinjection) or, for per_kit detectors, a named test."""
        return self.reinjection is not None or (self.detector.scope == "per_kit" and bool(self.detector.test))

    def applies(self, facts: dict[str, Any]) -> bool:
        sel = self.detector.applies_to
        has = sel.get("has") or []
        return all(facts.get(h) for h in has) and all(facts.get(k) == v for k, v in (sel.get("equals") or {}).items())


class Registry(CrucibleModel):
    defects: list[Defect]

    def version(self) -> str:
        h = hashlib.sha256(json.dumps([d.model_dump(mode="json") for d in sorted(self.defects, key=lambda d: d.id)], sort_keys=True).encode())
        return f"reg:{len(self.defects)}:{h.hexdigest()[:12]}"

    def by_check(self, check: str) -> list[Defect]:
        return [d for d in self.defects if d.detector.check.split(".")[0] == check]

    def get(self, id: str) -> Defect:
        return next(d for d in self.defects if d.id == id)


def load_registry(root: Path = DEFECTS, extra: list[Defect] = ()) -> Registry:
    ds = [Defect.model_validate(yaml.safe_load(p.read_text())) for p in sorted(Path(root).glob("*.yaml"))]
    return Registry(defects=[*ds, *extra])
