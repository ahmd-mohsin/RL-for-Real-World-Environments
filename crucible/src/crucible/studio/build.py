"""``crucible build``: workflows → RL tasks and a benchmark (ARCHITECTURE §4 M15: M3 → M8 → M9 → M11 → M13).

Steps, each recorded in ``summary.json``:
1. **Ingest** the pack's workflow sources into ``out/store`` (M3; idempotent).
2. **Forge** tasks from them with the pack's ``forge`` hook (M8; the toy pack uses scripted roles and makes no paid
   calls). Tasks are written as Harbor-style directories with the Crucible overlay.
3. **Accept** each task with the M9 contract's model-free checks: A1 oracle, A2 floors, A3 negatives, A4-lite, A5 leak,
   A7 coverage and A8 grounding. The full contract (``accept_task``, A1–A13) needs Docker and paid calls; run it
   with ``crucible accept``.
4. **Export** (M11): each accepted task's oracle is replayed into canonical ``reward.json`` + ``steps.jsonl``, the outputs
   every runtime adapter (OpenEnv, NeMo-Gym, verifiers, Harbor through ATIF replay) must reproduce. A World task is not
   a Harbor-native directory (it has no Dockerfile or ``tests/``); Harbor reaches it through ATIF replay.
5. **Bench** (M13): the accepted tasks are the anchors, plus ``n_fresh`` items from the pack's generator under an HMAC
   key, validated (gold solves, no-op does not). They run with the pack's scripted policies and are reported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


async def build(pack: str | Path, *, out: Path, n_fresh: int = 2, k: int = 1, key: bytes = b"crucible-build-v1", dry_run: bool = False) -> dict[str, Any]:
    from crucible.accept.contract import accept
    from crucible.actors.policy import EpisodeLimits
    from crucible.bench import dynamic
    from crucible.bench.run import run_bench
    from crucible.runtime.harness.native import NativeHarness
    from crucible.studio.label import _resolved, ingest
    from crucible.studio.pack import find_pack, load_manifest, resolve

    pack_dir = find_pack(pack)
    m = load_manifest(pack_dir)
    missing = [h for h in ("forge", "policies") if getattr(m.hooks, h) is None]
    if missing:
        raise ValueError(f"pack {m.name} has no {', '.join('hooks.' + h for h in missing)}: build needs them")
    if dry_run:
        return {"command": "build", "dry_run": True, "pack": m.name, "ingest": [s.glob for s in m.ingest], "hooks": m.hooks.model_dump()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _resolved(out, "build", {"pack": m.name, "pack_dir": str(pack_dir), "n_fresh": n_fresh, "k": k, "key_commitment": dynamic.commitment(key)})
    summary: dict[str, Any] = {"command": "build", "pack": m.name}
    traces = ingest(pack_dir, m, out / "store")
    summary["ingest"] = {"traces": len(traces)}
    forged = await resolve(pack_dir, m.hooks.forge)(out / "forge", out / "store")
    summary["forge"] = {"tasks": len(forged), "dir": str(out / "forge" / "tasks")}
    reports = {}
    accepted = []
    for t, res in forged:
        rep = await accept(t, resolve=res)
        reports[t.task_id] = {c.id: c.status for c in rep.checks}
        if rep.ok:
            accepted.append(t)
    (out / "accept.json").write_text(json.dumps(reports, indent=1, sort_keys=True) + "\n")
    summary["accept"] = {"accepted": len(accepted), "total": len(forged), "failing": {k: [c for c, s in v.items() if s == "fail"] for k, v in reports.items()
                                                                                       if any(s == "fail" for s in v.values())}}
    from crucible.interop.replay import Action, replay

    exported, export_errors = {}, {}
    for t in accepted:  # M11: the canonical outputs every runtime adapter (OpenEnv, NeMo-Gym, verifiers, Harbor via ATIF) must reproduce
        acts = [Action(a["function_name"], dict(a.get("arguments") or {})) if a.get("kind", "tool_call") == "tool_call"
                else Action(None, {}, to=a.get("to"), text=a.get("text")) for a in t.oracle]
        d = out / "export" / t.task_id.replace("/", "_").replace("@", "_")
        try:
            r = await replay(t.runtime_task(), acts, d, episode_key=f"{t.task_id}|oracle", plan=t.plan)
            exported[t.task_id] = {"reward": r["reward"], "steps": r["steps"], "dir": str(d)}
        except Exception as e:  # noqa: BLE001 - reported per task
            export_errors[t.task_id] = f"{type(e).__name__}: {e}"[:300]
    summary["export"] = {"canonical": len(exported), "oracle_rewards": {k: v["reward"] for k, v in sorted(exported.items())}, "errors": export_errors,
                         "dir": str(out / "export")}
    anchors = []
    for t in accepted:
        rt = t.runtime_task()
        rt.oracle = t.oracle  # the scripted policies replay it
        anchors.append((t.task_id, "forged", rt))
    made: list[Any] = []
    if m.hooks.generator and n_fresh:
        gen = resolve(pack_dir, m.hooks.generator)

        def generator(seed: int) -> Any:
            made.append(gen(seed))
            return made[-1]

        bm = await dynamic.build(f"{m.name}-build", "v1", key=key, anchors=anchors, generators={"generated": generator}, n_fresh=n_fresh)
    else:
        bm = await dynamic.build(f"{m.name}-build", "v1", key=key, anchors=anchors, generators={}, n_fresh=0)
    stages = resolve(pack_dir, m.hooks.stages) if m.hooks.stages else None
    try:
        res = await run_bench(f"{m.name}-build", root=out / "runs", tasks=[i.task for i in bm.items], policies=resolve(pack_dir, m.hooks.policies)(),
                              harnesses=[NativeHarness(limits=EpisodeLimits(max_steps=1000))], k=k, stages_of=stages, extras={"manifest": bm.manifest()})
    finally:
        for t in made:
            if callable(getattr(t, "cleanup", None)):
                t.cleanup()
    summary["bench"] = {"items": bm.manifest()["counts"], "report": res["paths"]["report"], "digest": json.loads(Path(res["paths"]["summary"]).read_text())["digest"],
                        "models": {mm["policy"]: mm["overall"]["deterministic"].get("mean_reward") for mm in res["summary"]["models"]}}
    summary["next"] = [f"open {res['paths']['report']}", f"crucible label --pack {m.name}"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    return summary
