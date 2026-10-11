"""Run a benchmark through the M10 runner and report it (DEPEND: the runner's journal is the result store).

``run_bench`` runs tasks × policies × harnesses with k samples (masked attempts retried by the runner), loads the
episodes, computes the summary, optionally fits IRT (respondent = policy@harness, item = task), and writes
``summary.json`` / ``episodes.json`` / ``report.html`` next to the run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.bench import irt, report
from crucible.bench.metrics import Episode, load_run, summarize


def irt_counts(eps: Sequence[Episode], *, threshold: float = 1.0) -> dict[tuple[str, str], tuple[float, int]]:
    out: dict[tuple[str, str], list[int]] = {}
    for e in eps:
        if not e.masked:
            c = out.setdefault((f"{e.policy}@{e.harness}", e.task), [0, 0])
            c[0] += int(e.passed(threshold))
            c[1] += 1
    return {k: (float(v[0]), v[1]) for k, v in out.items()}


async def run_bench(name: str, *, root: Path, tasks: Sequence[Any], policies: Sequence[Any], harnesses: Sequence[Any], k: int = 1,
                    ks: Sequence[int] = (1,), threshold: float = 1.0, stages_of: Callable[[str], Any] | None = None,
                    horizon_of: Callable[[str], str] | None = None, anchors: Mapping[str, irt.Item] | None = None,
                    extras: dict[str, Any] | None = None, run_id: str | None = None) -> dict[str, Any]:
    from crucible.runtime.run import Runner, RunSpec

    run_id = run_id or f"bench-{name}"
    await Runner(root).run(RunSpec(run_id=run_id, tasks=list(tasks), policies=list(policies), harnesses=list(harnesses), k=k))
    return report_run(name, Path(root) / run_id, ks=ks, threshold=threshold, stages_of=stages_of, horizon_of=horizon_of, anchors=anchors, extras=extras)


def report_run(name: str, run_dir: Path, *, ks: Sequence[int] = (1,), threshold: float = 1.0, stages_of: Callable[[str], Any] | None = None,
               horizon_of: Callable[[str], str] | None = None, anchors: Mapping[str, irt.Item] | None = None, extras: dict[str, Any] | None = None,
               out_dir: Path | None = None) -> dict[str, Any]:
    eps = load_run(run_dir)
    summary = summarize(eps, ks=ks, threshold=threshold, stages_of=stages_of, horizon_of=horizon_of)
    counts = irt_counts(eps, threshold=threshold)
    ex = dict(extras or {})
    if counts and len({r for r, _ in counts}) >= 2:
        f = irt.fit(counts, anchors=anchors)
        ex["irt"] = f.to_json() | {"item_qa_flagged": irt.item_qa(f)}
        if anchors:
            ex["irt"]["anchor_drift"] = irt.drift(anchors, counts)
    paths = report.write(out_dir or (Path(run_dir) / "bench"), name, summary, eps, extras=ex)
    return {"summary": summary, "episodes": eps, "extras": ex, "paths": paths}
