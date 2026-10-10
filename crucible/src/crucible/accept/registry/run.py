"""Registry operations (research m9-m10/01 §15.4).

- ``power_tests(registry)``: run every admitted per-task detector's re-injection; a detector must report the expected
  status on the clean fixture and on the defective one. Fixtures needing unavailable resources are reported, not run.
- ``backfill(defect, reports, tasks)``: evaluate a new detector on every task whose current report is accepted; tasks it
  fails move to ``quarantined`` (the pool refuses them), and every other cached result is reused.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from crucible.accept.registry.model import Defect, Registry


async def power_test(d: Defect, available: Sequence[str] = ()) -> dict[str, Any]:
    if d.reinjection is None:
        return {"id": d.id, "status": "per_kit" if d.detector.scope == "per_kit" else "no_power_test", "test": d.detector.test}
    missing = [n for n in d.reinjection.needs if n not in available]
    if missing:
        return {"id": d.id, "status": "needs", "needs": missing}
    mod, fn = d.reinjection.fixture.split(":")
    got = await getattr(importlib.import_module(mod), fn)()
    ok = all(got.get(k) == v for k, v in d.reinjection.expect.items())
    return {"id": d.id, "status": "proven" if ok else "not_proven", "observed": got, "expected": d.reinjection.expect}


async def power_tests(reg: Registry, available: Sequence[str] = ()) -> list[dict[str, Any]]:
    return [await power_test(d, available) for d in reg.defects]


async def backfill(d: Defect, tasks: Sequence[Any], *, cache_root: Path, resolve=lambda s: False, **kw: Any) -> dict[str, Any]:
    """Run the defect's check on every task (cached results for other checks are untouched); a task whose check fails is
    quarantined. The new registry version keys new results, so nothing stale is reused for the new detector."""
    from crucible.accept.run import accept_task

    check = d.detector.check.split(".")[0]
    out = {}
    for t in tasks:
        facts = t.facts() if hasattr(t, "facts") else {}
        if d.detector.applies_to and not d.applies(facts):
            out[t.task_id] = "not_applicable"
            continue
        rep = await accept_task(t, only=[check], cache_root=cache_root, resolve=resolve, **kw)
        st = str(rep.results[check].status)
        out[t.task_id] = "quarantined" if st == "fail" else "kept"
    return out
