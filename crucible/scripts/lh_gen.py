"""P5a (swarm-pipeline/00 §2, §15 P5): the generators and the scouts, end to end, at $0 (model calls only in ``author``).

    python crucible/scripts/lh_gen.py --root <p5> --clones <lhrepos> scout   [--hosts click,marshmallow] [--max-pairs 6]
    python crucible/scripts/lh_gen.py --root <p5> --clones <lhrepos> rebreak [--repos a,b] [--groups 3]
    python crucible/scripts/lh_gen.py --root <p5> --clones <lhrepos> stitch  [--pairs typer:click,...]
    python crucible/scripts/lh_gen.py --root <p5> --clones <lhrepos> open    [--repos a,b]
    python crucible/scripts/lh_gen.py --root <p5> --clones <lhrepos> gates   --gen rebreak|stitched_window|open [--only id,...]
    python crucible/scripts/lh_gen.py --root <p5> --clones <lhrepos> author  --gen rebreak --n 4 [--usd 1]
    python crucible/scripts/lh_gen.py --root <p5> report [--out <json>]
    python crucible/scripts/lh_gen.py --root <p5> cleanup     (removes this phase's containers, never images)

Every generator writes its records under ``<root>/<generator>/store/train`` (a training ``WindowStore``: held-out
repositories are refused there and by the eval lease before anything is cloned), its synthetic history in its own derived
clones (``<root>/<generator>/clones``; the source clones are never written), and one funnel row per (candidate, stage) in
``<root>/funnel.jsonl``. ``gates`` runs S6, S5, S8, S3 and S10 (``generators.base.deterministic_gates``) and stores each
task's result as ``gates.json`` next to its record. ``author`` writes prompts for at most ``--n`` accepted tasks with the
author role (one USD ledger, ``<root>/spend.jsonl``, capped by ``--usd``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from crucible.factory.generators.base import GRADE_PREFIX, WORLD_PREFIX, Funnel, GateConfig, deterministic_gates
from crucible.factory.lanes.windows import HELD_OUT_REPOS, POOL_REPOS, SMALL_REPOS, ImageBook, sandbox_factory
from crucible.factory.lanes.windows_world import docker
from crucible.factory.sources.lease import EvalLease

TRAIN = [*SMALL_REPOS, *POOL_REPOS]


def jsave(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    tmp.replace(path)


def jload(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(ev: dict) -> None:
        with path.open("a") as fh:
            fh.write(json.dumps({"t": round(time.time(), 1), **ev}, default=str) + "\n")
        print(json.dumps(ev, default=str)[:300], flush=True)

    return log


def lease_of(a: argparse.Namespace) -> EvalLease:
    return EvalLease.default(Path(a.sealed) if a.sealed else None)


def shared(a: argparse.Namespace) -> dict[str, Any]:
    root = Path(a.root)
    return {"funnel": Funnel(root / "funnel.jsonl"), "lease": lease_of(a), "images": ImageBook(parallel=a.builds),
            "sandboxes": sandbox_factory(parallel=a.parallel, memory="1536m", name_prefix=GRADE_PREFIX), "log": logger(root / "gen_log.jsonl")}


def generator(a: argparse.Namespace, name: str, **kw: Any) -> Any:
    root, clones = Path(a.root), Path(a.clones)
    sh = shared(a)
    if name == "rebreak":
        from crucible.factory.generators.rebreak import Rebreak

        repos = [r for r in (a.repos.split(",") if a.repos else TRAIN) if r not in HELD_OUT_REPOS]
        return Rebreak(root / "rebreak", {r: clones / r for r in repos}, max_groups_per_repo=a.groups, full_suite=getattr(a, "full_suite", False), **sh, **kw)
    if name == "stitched_window":
        from crucible.factory.generators.stitch import Stitch

        return Stitch(root / "stitch", clones, extra_clones=root / "repos", per_pair=getattr(a, "per_pair", 2), max_units=getattr(a, "max_units", 25), **sh, **kw)
    if name == "open":
        from crucible.factory.generators.open import OpenGen

        repos = [r for r in (a.repos.split(",") if a.repos else TRAIN) if r not in HELD_OUT_REPOS]
        return OpenGen(root / "open", {r: clones / r for r in repos}, **sh, **kw)
    raise SystemExit(f"unknown generator {name}")


async def build_all(gen: Any, cands: list[Any], parallel: int) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(parallel)

    async def one(c: Any) -> dict[str, Any]:
        async with sem:
            try:
                rec = await gen.build(c)
            except Exception as e:  # noqa: BLE001 - one candidate's failure is recorded, never the run's
                gen.funnel.note(gen.name, c.id, "S3_world", False, f"error: {type(e).__name__}: {e}"[:300])
                return {"id": c.id, "error": f"{type(e).__name__}: {e}"[:300]}
            print(json.dumps({"built": c.id, "stage": rec.get("stage"), "reason": rec.get("reason"), "edges": rec.get("dependency_edges")})[:300], flush=True)
            return rec

    return list(await asyncio.gather(*[one(c) for c in cands]))


# the one generic intent of P5a (a candidate card: no human has approved it; generators never read the eval store)
INTENT = {"id": "intent-p5-generic-change", "behaviour": "carry out a dependent multi-step change in a real Python library and the code that uses it",
          "failure_mode": "loses track of an earlier step, of interacting faults, or of the downstream code that depends on the change",
          "difficulty_axes": ["dependency", "cross-repository", "regression", "performance"],
          "coverage": {"behaviour": "dependent library change", "surface": "python-library-change"}, "status": "candidate"}


async def cmd_scout(a: argparse.Namespace) -> None:
    """Consumers of the training hosts (and the other record types, to show the scouts), filtered, then up to
    ``--max-pairs`` new (consumer, host) pairs that pass the CPU probe and have a usable requirement bump."""
    from crucible.factory.cards import IntentCard
    from crucible.factory.generators.stitch import consumer_window, host_roots, host_window, imports_host, requirement_bumps
    from crucible.factory.lanes.windows import REPO_URLS
    from crucible.factory.lanes.windows_world import project_name
    from crucible.factory.sources.github import GitHub
    from crucible.factory.sources.scouts import Scout, clone, cpu_probe, expand

    root, clones = Path(a.root), Path(a.clones)
    card = IntentCard.model_validate(INTENT)
    lease = lease_of(a)
    gh = GitHub(root / "scout" / "cache")
    scout = Scout(gh, lease, exclude=[REPO_URLS[n] for n in REPO_URLS])
    hosts = [h for h in (a.hosts.split(",") if a.hosts else TRAIN) if h not in HELD_OUT_REPOS and (clones / h / ".git").exists()]
    dists = {h: project_name(clones / h) or h for h in hosts}
    records: list[Any] = []
    queries: dict[str, list[str]] = {"host_repo": expand(card, "host_repo", limit=3), "request": expand(card, "request", limit=2),
                                     "real_perf": expand(card, "real_perf", limit=1), "real_fix": expand(card, "real_fix", limit=1)}
    records += scout.repos(queries["host_repo"], "host_repo", per_query=20)
    records += scout.issues(queries["request"])
    records += scout.commits(queries["real_perf"], "real_perf") + scout.commits(queries["real_fix"], "real_fix")
    consumers = []
    for h in hosts:
        qs = expand(card, "consumer", hosts=[h if h not in ("dateutil",) else "dateutil"], limit=a.queries_per_host)
        queries[f"consumer:{h}"] = qs
        consumers += scout.consumers_of(h, dists[h], qs, max_scan=a.scan)
    # libraries first: a repository named after its host (a plugin or extension of it), then the smallest (an application
    # with a large suite rarely builds and passes in a CPU World: strawberry's 57 failures in the first run)
    ok = sorted((c for c in consumers if c.passed and c.filters.get("depends_on_host", {}).get("ok")),
                key=lambda c: (c.evidence["host"].lower().replace("-", "") not in c.repo.split("/")[-1].lower().replace("-", ""), c.size_kb or 0))
    if ok:
        records += scout.forks(ok[0], limit=5)  # snowball: forks of the best consumer (the API lists them; dependents come from requirement scans)
    records += consumers
    pairs: list[dict[str, Any]] = []
    images = ImageBook(parallel=a.builds)
    sandboxes = sandbox_factory(parallel=a.parallel, memory="1536m", name_prefix=GRADE_PREFIX)
    seen: set[str] = set()
    for c in ok:
        if len(pairs) >= a.max_pairs:
            break
        if c.repo.lower() in seen:
            continue
        seen.add(c.repo.lower())
        host = c.evidence["host"]
        try:
            d = await asyncio.to_thread(clone, c.repo, root / "repos")
        except Exception as e:  # noqa: BLE001 - recorded on the record
            c.filters["bump"] = {"ok": False, "why": f"clone: {e}"[:200]}
            continue
        bumps = await asyncio.to_thread(requirement_bumps, d, dists[host])
        roots = host_roots(clones / host)
        usable = []
        for b in bumps:
            if not await asyncio.to_thread(imports_host, d, b["sha"], roots):
                continue
            hw = await asyncio.to_thread(host_window, clones / host, b["target"])
            if hw:
                usable.append({"bump": b["sha"], "target": b["target"], "host_window": hw, "consumer_window": consumer_window(d, b)})
        c.filters["bump"] = {"ok": bool(usable), "bumps": len(bumps), "usable": len(usable)}
        if not usable:
            continue
        probe = await cpu_probe(c, root / "repos", images=images, sandboxes=sandboxes)
        pairs.append({"consumer": c.repo, "consumer_local": d.name, "host": host, "host_local": host, "usable": usable[:3], "probe_ok": probe,
                      "probe": c.filters.get("cpu_probe")})
        print(json.dumps({"pair": c.repo, "host": host, "usable": len(usable), "probe": c.filters.get("cpu_probe")})[:300], flush=True)
    out = root / "scout"
    jsave(out / "records.json", [r.to_json() for r in records])
    jsave(out / "pairs.json", pairs)
    by_type: dict[str, dict[str, int]] = {}
    for r in records:
        t = by_type.setdefault(r.type, {"found": 0, "lease_ok": 0, "passed_filters": 0})
        t["found"] += 1
        t["lease_ok"] += int(bool(r.lease.get("ok")))
        t["passed_filters"] += int(r.passed)
    jsave(out / "scout_report.json", {"intent": INTENT, "queries": queries, "by_type": by_type, "calls": gh.calls, "log": scout.log,
                                      "consumers": {"found": len(consumers), "depend_on_host": sum(1 for c in consumers if c.filters.get("depends_on_host", {}).get("ok")),
                                                    "passed": len(ok)}, "pairs": len(pairs), "pairs_probe_ok": sum(1 for p in pairs if p["probe_ok"])})
    print(json.dumps({"by_type": by_type, "pairs": [(p["consumer"], p["host"], p["probe_ok"]) for p in pairs], "calls": gh.calls}, indent=1))


async def cmd_rebreak(a: argparse.Namespace) -> None:
    gen = generator(a, "rebreak")
    cands = await gen.candidates()
    jsave(Path(a.root) / "rebreak" / "candidates.json", [c.to_json() for c in cands])
    jsave(Path(a.root) / "rebreak" / "real_fix_records.json", [r.to_json() for r in gen.records])
    print(f"{len(cands)} rebreak candidates", flush=True)
    await build_all(gen, cands, a.windows)


async def cmd_stitch(a: argparse.Namespace) -> None:
    from crucible.factory.generators.stitch import discover_pairs

    root, clones = Path(a.root), Path(a.clones)
    gen = generator(a, "stitched_window")
    if a.pairs:
        pairs = [tuple(x.split(":", 1)) for x in a.pairs.split(",") if ":" in x]
    else:
        pairs = await asyncio.to_thread(discover_pairs, {n: clones / n for n in TRAIN if (clones / n / ".git").exists()})
        pairs += [(p["consumer_local"], p["host_local"]) for p in jload(root / "scout" / "pairs.json") or [] if p.get("probe_ok")]
    print(f"pairs: {pairs}", flush=True)
    cands = await gen.candidates(pairs)
    jsave(root / "stitch" / "candidates.json", [c.to_json() for c in cands])
    print(f"{len(cands)} stitch candidates", flush=True)
    await build_all(gen, cands, a.windows)


async def cmd_open(a: argparse.Namespace) -> None:
    gen = generator(a, "open")
    cands = await gen.candidates()
    if a.only:
        cands = [c for c in cands if c.id in a.only.split(",")]
    jsave(Path(a.root) / "open" / "candidates.json", [c.to_json() for c in cands])
    jsave(Path(a.root) / "open" / "real_perf_records.json", [r.to_json() for r in gen.records])
    print(f"{len(cands)} open candidates", flush=True)
    await build_all(gen, cands, a.windows)


async def cmd_gates(a: argparse.Namespace) -> None:
    gen = generator(a, a.gen)
    cfg = GateConfig(s3_reruns=a.s3_reruns, flake_reruns=a.flake_reruns)
    ready = "accepted_source" if a.gen == "open" else "mutants_valid"
    ids = [w for w in gen.store.window_ids() if gen.store.window(w).get("stage") == ready]
    if a.only:
        ids = [w for w in ids if w in a.only.split(",")]
    sem = asyncio.Semaphore(a.windows)

    async def one(wid: str) -> None:
        p = gen.store.dir(wid) / "gates.json"
        if p.exists() and not a.redo:
            return
        async with sem:
            task = gen.task(wid)
            if a.gen == "open":
                from crucible.factory.generators.open import open_gates

                res = await open_gates(task, flake_reruns=a.flake_reruns, s3_reruns=a.s3_reruns)
            else:
                res = await deterministic_gates(task, cfg=cfg, s3_sandboxes=getattr(gen, "s3_sandboxes", None))
        jsave(p, res)
        for g, r in res["gates"].items():
            if r["ok"] is not None:
                gen.funnel.note(gen.name, wid, g, bool(r["ok"]), "; ".join(r.get("findings") or [])[:300])
        print(json.dumps({"gates": wid, "ok": res["ok"], "passed": res["passed"]}), flush=True)

    await asyncio.gather(*[one(w) for w in ids])


OPEN_OPENING = ("Our users call {names} from {repo} constantly, and it has become a bottleneck. Please make that code faster on the "
                "kinds of inputs it gets today, with exactly the same results.")
OPEN_REQS = (("R1", "m_correct", "Every result stays exactly the same as today, including which errors are raised, and every existing test keeps passing."),
             ("R2", "m_faster", "The calls above run measurably faster than they do today on fresh inputs of the same kinds."),
             ("R3", "m_bar", "The faster the better: there is no upper limit on how much speedup counts."))


def open_prompt(rec: dict[str, Any]) -> dict[str, Any]:
    """An open task's request from a fixed template (no model call): what is slow (the traced entries), the three
    requirements every grade checks, the epilogue."""
    from crucible.factory.generators.base import compose_prompt
    from crucible.factory.lanes.windows import lint_prompt, locate

    names = ", ".join(f"`{e['function']}`" for e in rec["open"]["spec"]["entries"])
    opening = OPEN_OPENING.format(names=names, repo=rec["repo"])
    reqs = [{"id": i, "milestone": m, "text": t, "kind": "behaviour"} for i, m, t in OPEN_REQS]
    prompt = compose_prompt(opening, reqs, heading="What we need:")
    return {"prompt": prompt, "requirements": reqs, "spans": {r["id"]: list(loc) for r in reqs if (loc := locate(prompt, r["text"]))},
            "findings": lint_prompt("\n".join([opening, *[r["text"] for r in reqs]])), "authored_by": "template"}


async def cmd_author(a: argparse.Namespace) -> None:
    """Prompts for at most ``--n`` gate-passing tasks of one generator: the author role (one USD ledger, capped), up to
    three tries with the lane's findings sent back; the ledger and lint decide, and the authored task must conform."""
    from crucible.factory.generators.base import author_generated
    from crucible.factory.lanes.windows_git import RepoGit
    from crucible.factory.longtask import check_longtask
    from crucible.factory.models import ModelBook, booked
    from crucible.forge.budget import Ledger

    root = Path(a.root)
    gen = generator(a, a.gen)
    ids = [p.parent.name for p in sorted((gen.store.root / "tasks").glob("*/gates.json")) if (jload(p) or {}).get("ok")][: a.n]
    spend_path = root / "spend.jsonl"
    past = sum(float(json.loads(x).get("usd") or 0.0) for x in spend_path.read_text().splitlines() if x.strip()) if spend_path.exists() else 0.0
    rows = []
    book = ledger = None
    for wid in ids:
        rec = gen.store.window(wid)
        done = jload(gen.store.dir(wid) / "prompt.json")
        if done and not done.get("findings") and not a.redo:
            rows.append(done)
            continue
        if a.gen == "open":
            res = open_prompt(rec)
        else:
            if past >= a.usd:
                print(f"cap {a.usd} USD reached ({past:.4f} spent)")
                break
            if book is None:
                book = ModelBook.from_config(budget_usd_per_client=a.usd - past)
                ledger = Ledger(cap_usd=a.usd)
                if past:
                    ledger.settle("past:processes", past)
            client = book.client("author")
            repo = RepoGit(Path(gen.lane().repo_dirs[rec["repo"]]))
            base = gen.store.dir(wid) / "base"
            if a.gen == "rebreak":
                from crucible.factory.generators.rebreak import STYLE

                extra: dict[str, Any] = {}
            else:
                from crucible.factory.generators.stitch import STYLE

                st = rec["stitched_window"]
                side = st["unit_side"]
                extra = {"projects": {s: {"name": st[s].split("__")[-1], "readme_head": next((p.read_text(errors="replace")[:1200] for p in sorted((base / s).glob("README*"))), "")}
                                      for s in ("host", "consumer")},
                         "change_project": {m["id"]: st[side[m["unit"]]].split("__")[-1] if m.get("unit")
                                            else f"{st['host']} (observed through {st['consumer'].split('__')[-1]}'s tests)"
                                            for m in rec["milestones"]}}
            findings: list[str] = []
            res = None
            for k in range(1, 4):
                scope = None
                try:
                    async with booked(ledger, f"author/{wid}/{k}", estimate_usd=0.01) as scope:
                        res = await author_generated(client, rec, repo, base, style=STYLE, findings=findings, extra=extra,
                                                     sample_key=f"p5-author-{wid}-{k}" if k > 1 else None)
                except Exception as e:  # noqa: BLE001 - recorded, retried
                    findings = [f"error: {e}"[:300]]
                    continue
                finally:
                    usd = scope.usd if scope is not None else 0.0
                    past += usd
                    with spend_path.open("a") as fh:
                        fh.write(json.dumps({"t": round(time.time(), 1), "stage": "author", "role": "author", "who": f"{wid}#{k}", "usd": round(usd, 8)}) + "\n")
                findings = res["findings"] if res else findings
                if res and not findings:
                    break
        if res is None:
            rows.append({"task": wid, "findings": findings})
            continue
        row = {"task": wid, "findings": res["findings"], "prompt": res["prompt"], "requirements": res["requirements"], "spans": res["spans"],
               "authored_by": res.get("authored_by", "model:" + (book.spec("author").ref if book else "")), "names_stated_by_lane": res.get("names_stated_by_lane")}
        if not res["findings"]:
            task = gen.task(wid, authored=res)
            row["conformance"] = check_longtask(task, long_horizon=a.gen != "open", need_prompt=True)
            try:
                row["card"] = task.card(intent=INTENT["id"]).model_dump(mode="json")["lineage"]
            except Exception as e:  # noqa: BLE001
                row["card_error"] = str(e)[:300]
        jsave(gen.store.dir(wid) / "prompt.json", row)
        rows.append(row)
        print(json.dumps({"task": wid, "findings": row["findings"], "conformance": row.get("conformance")})[:400], flush=True)
    jsave(root / f"author_{a.gen}.json", {"rows": rows, "spend_usd_total": round(past, 6)})
    print(f"spend so far {past:.5f} USD")


def cmd_report(a: argparse.Namespace) -> None:
    root = Path(a.root)
    rep: dict[str, Any] = {"funnel": Funnel(root / "funnel.jsonl").summary(), "accepted": {}}
    for gen, sub in (("rebreak", "rebreak"), ("stitched_window", "stitch"), ("open", "open")):
        tdir = root / sub / "store" / "train" / "tasks"
        acc = []
        for g in sorted(tdir.glob("*/gates.json")) if tdir.exists() else []:
            r = jload(g)
            if r and r.get("ok"):
                acc.append(g.parent.name)
        rep["accepted"][gen] = acc
    spend = root / "spend.jsonl"
    rep["spend_usd"] = round(sum(json.loads(ln).get("usd", 0.0) for ln in spend.read_text().splitlines() if ln.strip()), 5) if spend.exists() else 0.0
    if a.out:
        jsave(Path(a.out), rep)
    print(json.dumps(rep, indent=1, default=str))


def cmd_cleanup(a: argparse.Namespace) -> None:
    names = []
    for prefix in (WORLD_PREFIX, GRADE_PREFIX):
        r = docker("ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}", timeout=60)
        names += [x for x in r.stdout.split() if x.startswith(prefix)]
    if names:
        docker("rm", "-f", *names, timeout=300)
    print(f"removed {len(names)} containers ({WORLD_PREFIX}*, {GRADE_PREFIX}*)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--clones", default=None)
    ap.add_argument("--sealed", default=None, help="a sealed benchmark manifest (sealed.json) the eval lease also refuses")
    ap.add_argument("--repos", default="")
    ap.add_argument("--parallel", type=int, default=5, help="grading containers at once")
    ap.add_argument("--builds", type=int, default=2, help="World image builds at once")
    ap.add_argument("--windows", type=int, default=3, help="candidates built (or gated) at once")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rebreak")
    p.add_argument("--groups", type=int, default=3)
    p.add_argument("--full-suite", action="store_true", help="the interaction screen runs B's whole suite per revert")
    p = sub.add_parser("stitch")
    p.add_argument("--pairs", default="")
    p.add_argument("--per-pair", type=int, default=2)
    p.add_argument("--max-units", type=int, default=25, help="qualifying host units at most in the host window")
    p = sub.add_parser("open")
    p.add_argument("--only", default="")
    p = sub.add_parser("scout")
    p.add_argument("--hosts", default="")
    p.add_argument("--max-pairs", type=int, default=6)
    p.add_argument("--queries-per-host", type=int, default=2)
    p.add_argument("--scan", type=int, default=25, help="consumers per host whose packaging is read")
    p = sub.add_parser("gates")
    p.add_argument("--gen", required=True)
    p.add_argument("--only", default="")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--s3-reruns", type=int, default=60)
    p.add_argument("--flake-reruns", type=int, default=30)
    p.add_argument("--groups", type=int, default=3)
    p = sub.add_parser("author")
    p.add_argument("--gen", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--usd", type=float, default=1.0)
    p.add_argument("--redo", action="store_true")
    p.add_argument("--groups", type=int, default=3)
    p = sub.add_parser("report")
    p.add_argument("--out", default="")
    sub.add_parser("cleanup")
    a = ap.parse_args()
    if a.cmd in ("report",):
        return cmd_report(a)
    if a.cmd == "cleanup":
        return cmd_cleanup(a)
    fn = getattr(sys.modules[__name__], f"cmd_{a.cmd}")
    asyncio.run(fn(a))


if __name__ == "__main__":
    main()
