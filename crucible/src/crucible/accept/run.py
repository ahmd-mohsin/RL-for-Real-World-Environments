"""The contract runner (research m9-m10/01 §15): key -> plan -> cached-or-run each check -> evidence-bound report.

    report = await accept_task(task, policy=DEFAULT, needs={Need.DOCKER}, cache_root=..., resolve=...)

Per check: N/A if ``applies`` derives a reason; ``pending`` if a need is unavailable or a dependency did not pass;
reuse a cached result for the same key and check version (evidence re-hashed); otherwise run it. ``measured`` is always
written as an evidence file, so a pass or fail is never without evidence. A crash or unavailable infrastructure is
``error``; a missing adapter capability is ``fail``.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from crucible.accept import key as keymod
from crucible.accept.cache import Cache
from crucible.accept.checks import REGISTRY, AcceptContext
from crucible.accept.checks.base import InfraUnavailable, MissingCapability, Out
from crucible.accept.model import AcceptanceReport, CheckResult, Need, Status
from crucible.accept.plan import order
from crucible.accept.policy import DEFAULT, Policy


def registry_version() -> str:
    from crucible.accept.registry.model import load_registry

    return load_registry().version()


async def accept_task(task, *, policy: Policy = DEFAULT, needs: Sequence[Need] = (), cache_root: Path, resolve: Callable[[str], bool] = lambda s: False,
                      only: Sequence[str] | None = None, budget_usd: float = 0.0, signoff_root: Path | None = None, reuse: bool = True,
                      resolve_image: Callable[[str], str] = keymod.docker_digest, reg_version: str | None = None) -> AcceptanceReport:
    k = keymod.compute(task, policy, reg_version or registry_version(), resolve_image=resolve_image)
    kd = k.digest()
    cache = Cache(cache_root)
    genv = task.grade_env() if hasattr(task, "grade_env") else kd
    genv = hashlib.sha256(f"{genv}|{k.kit_hash}|{sorted(k.images.items())}".encode()).hexdigest()[:24]
    ctx = AcceptContext(policy=policy, cache=cache, key=k, key_digest=kd, needs=frozenset(needs), resolve=resolve, budget_usd=budget_usd,
                        signoff_root=signoff_root, workdir=cache.dir(kd), grade_store=Path(cache_root) / "grades" / genv if reuse else None)
    selected = list(only or policy.required)
    report = AcceptanceReport(task_id=task.task_id, key=k, key_digest=kd, required=list(policy.required))
    for c in order(REGISTRY, selected):
        t0 = time.monotonic()
        na = c.applies(task)
        cached = cache.get(kd, c.id, c.version) if reuse else None
        if na is not None:
            out = Out(Status.NA, {}, [], na_reason=na)
        elif cached is not None and cached.status not in (Status.PENDING, Status.ERROR):
            report.results[c.id] = ctx.results[c.id] = cached
            continue
        elif missing := sorted(n for n in c.needs if n not in ctx.needs):
            out = Out(Status.PENDING, {"missing_needs": missing}, [f"needs {missing} (unavailable in this run)"])
        elif blocked := [d for d in c.depends_on if d in ctx.results and ctx.results[d].status not in (Status.PASS, Status.NA)]:
            out = Out(Status.PENDING, {"blocked_by": {d: str(ctx.results[d].status) for d in blocked}}, [f"blocked: {blocked} did not pass"])
        else:
            try:
                out = await c.run(task, ctx)
            except MissingCapability as e:
                out = Out(Status.FAIL, {}, [str(e)])
            except InfraUnavailable as e:
                out = Out(Status.ERROR, {}, [str(e)])
            except Exception as e:  # noqa: BLE001 -- a crashing check is an error result, never a pass
                out = Out(Status.ERROR, {"exception": type(e).__name__}, [f"{type(e).__name__}: {e}"[:500]])
        ev = [cache.evidence(kd, c.id, "measured.json", {"status": str(out.status), "measured": out.measured, "reasons": out.reasons})]
        r = CheckResult(id=c.id, version=c.version, status=out.status, measured=out.measured, reasons=out.reasons, evidence=ev,
                        seconds=round(time.monotonic() - t0, 3), detectors=sorted({*c.detectors, *out.detectors}), key_digest=kd, na_reason=out.na_reason)
        cache.put(kd, r)
        report.results[c.id] = ctx.results[c.id] = r
    if ctx.log:
        cache.evidence(kd, "grades", f"log-{int(time.time())}.json", ctx.log, kind="grades")
    report.stats = {"new_grades": ctx.new_grades, "grade_env": genv}  # grades actually run (not reused from the grade store)
    return report
