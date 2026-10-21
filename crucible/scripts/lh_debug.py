"""The multi-language debugging lane's funnel on CPU (swarm-pipeline/07): scout, mine, build, gate, author, report.

    python scripts/lh_debug.py scout                      # licence, language, fix units since 2025-09-01 per clone (no Docker)
    python scripts/lh_debug.py mine  --repos a,b          # executed fix check at HEAD, interacting groups, A'..B', validate_window
    python scripts/lh_debug.py subset --repo a --fixes s1,s2,s3 [--b B] [--id ID --parent P]   # one task of exactly these fixes
    python scripts/lh_debug.py gates [--ids ...]          # S6, S5, S8, S3, S10 on every built window that reached mutants_valid
    python scripts/lh_debug.py author                     # the builder's bug reports (prompts/<id>.json) through lint + ledger
    python scripts/lh_debug.py report                     # funnel per language, candidates, image disk

Everything lives under ``data/debug`` (clones in ``src/``, generator state in ``gen/``, funnel in ``funnel.jsonl``). No
model is called: the prompts are written by the builder (``generators.debug.BUILDER``) into ``prompts/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# the durable store: CRUCIBLE_DEBUG_ROOT (the main checkout's crucible/data/debug when run from a worktree), else this tree's
ROOT = Path(os.environ.get("CRUCIBLE_DEBUG_ROOT") or Path(__file__).resolve().parents[1] / "data" / "debug")
LICENCE_RX = (("MIT", r"\bMIT License\b|Permission is hereby granted, free of charge"), ("Apache-2.0", r"Apache License,?\s+Version 2\.0"),
              ("BSD", r"Redistribution and use in source and binary forms"), ("ISC", r"\bISC License\b|Permission to use, copy, modify, and/or distribute"),
              ("Unlicense", r"This is free and unencumbered software"))


def local_name(full: str) -> str:
    from crucible.factory.lanes.windows import DEBUG_REPO_URLS

    by_url = {v.lower(): k for k, v in DEBUG_REPO_URLS.items()}
    return by_url.get(full.lower(), full.split("/", 1)[1].lower())


def licence_of(d: Path) -> str:
    for f in sorted(d.iterdir()):
        if re.match(r"(?i)^(licen[cs]e|copying)", f.name) and f.is_file():
            t = f.read_text(errors="replace")
            hits = [n for n, rx in LICENCE_RX if re.search(rx, t)]
            if hits:
                return "+".join(hits)
            return "other"
    return "none"


def clones() -> dict[str, tuple[str, Path]]:
    out = {}
    for d in sorted((ROOT / "src").iterdir()):
        if (d / ".git").exists():
            full = d.name.replace("__", "/", 1)
            out[local_name(full)] = (full, d)
    return out


def cmd_scout(args: argparse.Namespace) -> None:
    from crucible.factory import langs
    from crucible.factory.generators.debug import SINCE, scope
    from crucible.factory.generators.rebreak import fix_units

    out = ROOT / "scout.json"
    rows = json.loads(out.read_text()) if out.exists() else {}  # a --repos scout adds to the earlier rows
    spec = Path(args.repos[1:]).read_text().replace("\n", ",") if args.repos.startswith("@") else args.repos
    only = {r.strip() for r in spec.split(",") if r.strip()}
    for name, (full, d) in clones().items():
        if only and name not in only and full not in only:
            continue
        lang = langs.detect(d)
        row = {"full": full, "lang": lang, "licence": licence_of(d)}
        if lang is None:
            rows[name] = {**row, "why": "no language"}
            continue
        p = langs.get(lang)
        if lang == "node" and (why := p.supported(d)):
            rows[name] = {**row, "why": why}
            continue
        with scope(lang):
            t0 = time.monotonic()
            fx = fix_units(d, "HEAD", since=args.since or SINCE)
        row.update(fixes=len(fx), seconds=round(time.monotonic() - t0, 1), sample=[u.subject[:70] for u in fx[-3:]],
                   size_mb=round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and ".git" not in f.parts) / 1e6, 1))
        rows[name] = row
        print(f"{lang:6} {name:22} {row['licence']:12} fixes={row['fixes']:3} {row['size_mb']}MB", flush=True)
    out.write_text(json.dumps(rows, indent=1))


def generator(repos: list[str], split: str, *, max_pool: int = 40) -> object:
    """The lane over ``repos``. A training run takes the eval lease (held-out sources refused); the held-out run is the
    benchmark builder's (its own store, no training role reads it: V9)."""
    from crucible.factory import langs
    from crucible.factory.generators.base import Funnel
    from crucible.factory.generators.debug import Debug
    from crucible.factory.sources.lease import EvalLease

    cl = clones()
    pick = {n: cl[n][1] for n in repos}
    return Debug(ROOT / "gen", pick, langs_of={n: langs.detect(p) for n, p in pick.items()}, funnel=Funnel(ROOT / "funnel.jsonl"),
                 lease=EvalLease.default() if split == "train" else EvalLease(), split=split, max_pool=max_pool, log=lambda e: print(json.dumps(e), flush=True))


def split_of(name: str) -> str:
    from crucible.factory.lanes.windows import split_of as so

    return so(name)


async def mine_one(gen: object, name: str, since: str | None, sem: asyncio.Semaphore) -> None:
    async with sem:
        try:
            cands = await gen.candidates_for(name, since=since)
        except Exception as e:  # noqa: BLE001 - one repository's failure is recorded, never the run's
            gen.note(f"{gen.langs_of[name]}/{name}", "S1_executed", False, f"error: {type(e).__name__}: {e}"[:300], lang=gen.langs_of[name])
            print(f"[{name}] error {e}", flush=True)
            return
        print(f"[{name}] {len(cands)} candidate group(s)", flush=True)
        for c in cands:
            try:
                rec = await gen.build(c)
                print(f"[{name}] {c.id}: {rec.get('stage')} {rec.get('reason', '')[:160]}", flush=True)
            except Exception as e:  # noqa: BLE001
                gen.note(c.id, "S3_world", False, f"error: {type(e).__name__}: {e}"[:300], lang=c.params["lang"])
                print(f"[{name}] {c.id}: error {e}", flush=True)
    removed = await asyncio.to_thread(drop_images, name)
    print(f"[{name}] done; removed {len(removed)} image(s) no stored World uses", flush=True)


def drop_images(repo: str) -> list[str]:
    """This lane's images of ``repo`` (``crucible-lh-d:<lang>-<repo>-<hash>``) that no stored window uses as its World and
    no container runs: the disk budget (B points and fixes each build one). Never touches another image."""
    keep = set()
    for f in ROOT.glob("gen/store/*/tasks/*/window.json"):
        w = json.loads(f.read_text())
        if w.get("stage") in ("mutants_valid", "dependent", "f2p_valid") and (w.get("world") or {}).get("image"):
            keep.add(w["world"]["image"])
    running = set(subprocess.run(["docker", "ps", "--format", "{{.Image}}"], capture_output=True, text=True, check=False).stdout.split())
    out = []
    for img in subprocess.run(["docker", "images", "crucible-lh-d", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True,
                              check=False).stdout.split():
        tag = img.split(":", 1)[1] if img.startswith("crucible-lh-d:") else ""
        mine = "-" in tag and tag.split("-", 1)[1].rsplit("-", 1)[0] == repo and img not in keep and img not in running
        if mine and subprocess.run(["docker", "rmi", img], capture_output=True, check=False).returncode == 0:
            out.append(img)
    return out


async def mine(args: argparse.Namespace) -> None:
    repos = [r for r in args.repos.split(",") if r]
    for split in ("train", "heldout"):
        mine_repos = [r for r in repos if split_of(r) == split]
        if mine_repos:
            gen = generator(mine_repos, split, max_pool=args.max_pool)
            sem = asyncio.Semaphore(args.parallel)
            await asyncio.gather(*[mine_one(gen, n, args.since or None, sem) for n in mine_repos])


async def subset(args: argparse.Namespace) -> None:
    """The task of exactly ``--fixes`` at ``--b`` (``Debug.subset``), built as ``mine`` builds a group: a mined task
    rebuilt from its fixes, or eased by leaving one of them applied. The repository's images are kept for the next build."""
    gen = generator([args.repo], split_of(args.repo))
    try:
        cand = await gen.subset(args.repo, [s for s in args.fixes.split(",") if s], b=args.b, since=args.since or None, cid=args.id, parent=args.parent)
    except ValueError as e:
        sys.exit(f"[{args.repo}] {e}")
    needs = {s[:10]: [x[:10] for x in v] for s, v in cand.params["needs"].items() if v}
    print(f"[{args.repo}] {cand.id}: fixes {[s[:10] for s in cand.params['fixes']]}, executed edges {needs}", flush=True)
    rec = await gen.build(cand)
    print(f"[{args.repo}] {cand.id}: {rec.get('stage')} {rec.get('reason', '')[:160]}", flush=True)


async def gates(args: argparse.Namespace) -> None:
    """Gate every window at ``mutants_valid`` not gated yet; with ``--loop`` keep scanning (every 2 min) until the file
    ``STOP_GATES`` exists."""
    while True:
        await gates_once(args)
        if not args.loop or (ROOT / "STOP_GATES").exists():
            return
        await asyncio.sleep(120)


async def gates_once(args: argparse.Namespace) -> None:
    from crucible.factory.generators.base import GateConfig
    from crucible.factory.lanes.windows import WindowStore

    for split in ("train", "heldout"):
        sd = ROOT / "gen" / "store" / split
        if not (sd / "split.json").exists():
            continue
        store = WindowStore(sd, split=split)
        ids = [w for w in store.window_ids() if store.window(w).get("stage") == "mutants_valid" and (not args.ids or w in args.ids.split(","))]
        repos = sorted({store.window(w)["repo"] for w in ids})
        if not ids:
            continue
        gen = generator(repos, split)
        cfg = GateConfig(s3_reruns=args.s3_reruns, flake_reruns=args.flake_reruns)
        for w in ids:
            out = ROOT / "gates" / f"{w}.json"
            lock = ROOT / "gates" / f"{w}.running"
            if (out.exists() and not args.redo) or lock.exists():
                continue
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock.write_text(str(time.time()))
            t0 = time.monotonic()
            try:
                res = await gen.gates(w, cfg=cfg)
            except Exception as e:  # noqa: BLE001
                res = {"task": w, "ok": False, "error": f"{type(e).__name__}: {e}"[:500], "gates": {}}
            res["seconds"] = round(time.monotonic() - t0, 1)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1, default=str))
            lock.unlink(missing_ok=True)
            print(f"{w}: ok={res.get('ok')} passed={res.get('passed')} {res.get('error', '')} ({res['seconds']}s)", flush=True)


def cmd_author(args: argparse.Namespace) -> None:
    from crucible.factory.generators.base import Funnel
    from crucible.factory.generators.debug import author_debug
    from crucible.factory.lanes.windows import WindowStore
    from crucible.factory.lanes.windows_git import RepoGit

    cl = clones()
    funnel = Funnel(ROOT / "funnel.jsonl")
    for split in ("train", "heldout"):
        sd = ROOT / "gen" / "store" / split
        if not (sd / "split.json").exists():
            continue
        store = WindowStore(sd, split=split)
        for w in store.window_ids():
            pf = ROOT / "prompts" / f"{w}.json"
            if not pf.exists():
                continue
            rec = store.window(w)
            repo = RepoGit(ROOT / "gen" / "clones" / rec["repo"]) if (ROOT / "gen" / "clones" / rec["repo"]).exists() else RepoGit(cl[rec["repo"]][1])
            res = author_debug(rec, json.loads(pf.read_text()), repo)
            (ROOT / "authored").mkdir(exist_ok=True)
            (ROOT / "authored" / f"{w}.json").write_text(json.dumps(res, indent=1))
            funnel.note("debug", w, "S7", not res["findings"], "; ".join(res["findings"])[:300], lang=rec.get("lang"))
            print(f"{w}: findings={len(res['findings'])} {res['findings'][:3]}")
    refused = json.loads((ROOT / "prompt_refusals.json").read_text()) if (ROOT / "prompt_refusals.json").exists() else {}
    for w, why in refused.items():  # the builder found no symptom-only report the hidden checks allow
        funnel.note("debug", w, "S7", False, f"refused by the builder: {why['reason']}", lang=w.split("-")[1])


def cmd_intent(args: argparse.Namespace) -> None:
    """Every stored window with milestones gets its intent card (``generators.debug.task_intent``); the family-level
    skill cards go to ``intent_cards.json``; ``candidates.json`` lists every window with its language, skills, level,
    split, cutoff and stage (the matched train / benchmark split is built from it)."""
    from crucible.factory.generators.debug import NAME, faulty_names, intent_cards, task_intent, visible_test_changes
    from crucible.factory.lanes.windows import WindowStore
    from crucible.factory.lanes.windows_git import RepoGit

    rows = []
    for split in ("train", "heldout"):
        sd = ROOT / "gen" / "store" / split
        if not (sd / "split.json").exists():
            continue
        store = WindowStore(sd, split=split)
        for w in store.window_ids():
            rec = store.window(w)
            if not rec.get("milestones"):
                continue
            clone = ROOT / "gen" / "clones" / rec["repo"]
            repo = RepoGit(clone) if clone.exists() else None
            hidden = rec.setdefault(NAME, {}).setdefault("hidden", {})
            if repo is not None and hidden.get("mode") == "strip" and "visible_test_changes" not in hidden:
                hidden["visible_test_changes"] = visible_test_changes(rec, repo)  # windows built before the field existed
            rec[NAME]["intent"] = task_intent(rec, faulty_names(rec, repo) if repo is not None else ())
            store.save_window(rec)
            g = ROOT / "gates" / f"{w}.json"
            gate = json.loads(g.read_text()) if g.exists() else {}
            it = rec[NAME]["intent"]
            import difflib

            base = sd / "tasks" / w / "base"
            size = 0
            for path, txt in (rec.get("oracle_files") or {}).items():  # the oracle's size: changed lines over A'
                if not any(path.endswith(e) for e in (".go", ".rs", ".py", ".ts", ".js", ".mjs", ".java", ".tsx", ".cts", ".mts")):
                    continue
                old_txt = (base / path).read_text(errors="replace") if (base / path).is_file() else ""
                size += sum(1 for ln in difflib.unified_diff(old_txt.splitlines(), (txt or "").splitlines(), lineterm="", n=0)
                            if ln[:1] in "+-" and not ln.startswith(("+++", "---")))
            rows.append({"id": w, "lang": it["lang"], "repo": rec["repo"], "split": split, "stage": rec.get("stage"), "skills": it["skills"],
                         "level": it["level"], "faults": it["faults"], "modules": len(it["modules"]), "edges": int(rec.get("dependency_edges") or 0), "oracle_lines": size,
                         "milestone_tests": sum(len(m["tests"]) for m in rec.get("milestones") or []),
                         "cutoff": rec[NAME].get("cutoff"), "contamination": rec[NAME].get("contamination"), "gates_ok": gate.get("ok"),
                         "gates_passed": gate.get("passed"), "prompt": (ROOT / "authored" / f"{w}.json").exists(),
                         "fairness_flags": it.get("fairness_flags") or []})
    fixes_of = {}
    for split in ("train", "heldout"):
        sd = ROOT / "gen" / "store" / split
        for f in sd.glob("tasks/*/window.json"):
            w = json.loads(f.read_text())
            fixes_of[w["id"]] = {x["sha"] for x in (w.get(NAME) or {}).get("fixes") or []}
    for r in rows:  # candidates that share a fix are variants of one another (the herding guard counts them once)
        r["shares_fixes_with"] = sorted(o for o, fx in fixes_of.items() if o != r["id"] and fx & fixes_of.get(r["id"], set()))
    (ROOT / "intent_cards.json").write_text(json.dumps([c.model_dump(mode="json") for c in intent_cards()], indent=1))
    (ROOT / "candidates.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps(r))


def cmd_brief(args: argparse.Namespace) -> None:
    """What the prompt builder reads for one window (the builder writes ``prompts/<id>.json`` from it): per milestone the
    fix's message, its source diff (to understand the bug, never to be quoted), and the hidden tests' diff (what a user
    would observe)."""
    from crucible.factory.generators.debug import faulty_names, hidden_names, scope
    from crucible.factory.lanes.windows import WindowStore
    from crucible.factory.lanes.windows_git import RepoGit, Unit

    for split in ("train", "heldout"):
        sd = ROOT / "gen" / "store" / split
        if not (sd / "split.json").exists() or not (sd / "tasks" / args.id).exists():
            continue
        rec = WindowStore(sd, split=split).window(args.id)
        repo = RepoGit(ROOT / "gen" / "clones" / rec["repo"])
        units = {u["sha"]: Unit(**u) for u in rec["units"]}
        print(f"# {args.id} ({rec['lang']}, {rec['repo']}, {split}) faulty={sorted(faulty_names(rec, repo))} hidden={sorted(hidden_names(rec))[:12]}")
        with scope(rec["lang"]):
            for m in rec["milestones"]:
                u = units[m["unit"]]
                print(f"\n## {m['id']} requires={m.get('requires')} tests={len(m['tests'])} e.g. {m['tests'][:4]}\n{u.subject}\n{u.body[:600]}")
                src = subprocess.run(["git", "-C", str(repo.dir), "diff", u.parent, u.sha, "--", *u.src_files], capture_output=True, text=True, check=False).stdout
                rest = [p for p in u.files if p not in u.src_files]
                tst = subprocess.run(["git", "-C", str(repo.dir), "diff", u.parent, u.sha, "--", *rest], capture_output=True, text=True,
                                     check=False).stdout if rest else "(none)"
                print("--- source diff ---\n" + src[: args.chars] + "\n--- test diff ---\n" + tst[: args.chars])


def funnel_table() -> tuple[str, dict[str, Any]]:
    """The yield ledger per language (``00`` §9): fix units mined -> real fixes executed -> interacting groups -> windows
    validated (World, leak probes, fail-to-pass) -> dependent with valid mutants -> every CPU gate -> prompt (S7), with
    the top drop reasons per language. The last row of each (candidate, stage) counts."""
    import collections

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for ln in (ROOT / "funnel.jsonl").read_text().splitlines():
        r = json.loads(ln)
        rows[(r["candidate"], r["stage"])] = r
    per: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {"fixes": set(), "real": set(), "groups": set(), "windows": set(), "valid": set(),
                                                                      "gates": set(), "s7": set(), "why": collections.Counter()})
    for (c, st), r in rows.items():
        lang = (r.get("evidence") or {}).get("lang") or (c.split("/", 1)[0] if "/" in c else c.split("-")[1])
        p = per[lang]
        fix = ":" in c and "/" in c
        if st == "S1_source" and fix:
            p["fixes"].add(c)
        elif st == "S1_executed" and fix:
            (p["real"].add(c) if r["ok"] else p["why"].update([r["reason"].split(" (")[0][:60]]))
        elif c.startswith("db-") and st == "S3_world":
            p["groups"].add(c)
            if not r["ok"]:
                p["why"].update([r["reason"].split(" (")[0][:60]])
    for f in ROOT.glob("gen/store/*/tasks/*/window.json"):
        w = json.loads(f.read_text())
        lang = w.get("lang") or w["id"].split("-")[1]
        if w.get("stage") != "windows":
            per[lang]["windows"].add(w["id"])
        else:
            per[lang]["groups"].add(w["id"])
        if w.get("stage") == "mutants_valid":
            per[lang]["valid"].add(w["id"])
        elif w.get("stage") != "windows" and w.get("reason"):
            per[lang]["why"].update([w["reason"].split(":")[0][:60]])
        g = ROOT / "gates" / f"{w['id']}.json"
        if g.exists():
            gate = json.loads(g.read_text())
            if gate.get("ok"):
                per[lang]["gates"].add(w["id"])
            else:
                failed = [k for k, v in (gate.get("gates") or {}).items() if v.get("ok") is False]
                per[lang]["why"].update([f"gate {','.join(failed) or 'error'}"])
        a = ROOT / "authored" / f"{w['id']}.json"
        if a.exists() and not json.loads(a.read_text())["findings"]:
            per[lang]["s7"].add(w["id"])
    lines = ["| language | fix units | real fixes | interacting groups | validated windows | dependent + valid mutants | all CPU gates | prompt (S7) | top drop reasons |",
             "|---|---|---|---|---|---|---|---|---|"]
    out: dict[str, Any] = {}
    for lang in sorted(per):
        p = per[lang]
        n = {k: len(p[k]) for k in ("fixes", "real", "groups", "windows", "valid", "gates", "s7")}
        out[lang] = {**n, "why": dict(p["why"].most_common(5))}
        why = "; ".join(f"{k} ({v})" for k, v in p["why"].most_common(4))
        lines.append(f"| {lang} | {n['fixes']} | {n['real']} | {n['groups']} | {n['windows']} | {n['valid']} | {n['gates']} | {n['s7']} | {why} |")
    return "\n".join(lines), out


def cmd_report(args: argparse.Namespace) -> None:
    table, data = funnel_table()
    print(table)
    (ROOT / "funnel_table.md").write_text(table + "\n")
    (ROOT / "funnel_table.json").write_text(json.dumps(data, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scout")
    s.add_argument("--since", default="")
    s.add_argument("--repos", default="", help="scout only these (local names or owner/repo, or @file with one per line); the rows merge into scout.json")
    m = sub.add_parser("mine")
    m.add_argument("--repos", required=True)
    m.add_argument("--parallel", type=int, default=2)
    m.add_argument("--max-pool", type=int, default=40)
    m.add_argument("--since", default="", help="the fix cutoff (default generators.debug.SINCE; FALLBACK_SINCE is the marked fallback)")
    su = sub.add_parser("subset")
    su.add_argument("--repo", required=True)
    su.add_argument("--fixes", required=True, help="the fix shas (prefixes) reverted together, comma-separated")
    su.add_argument("--b", default="HEAD", help="B, the oracle commit (default the clone's HEAD)")
    su.add_argument("--since", default="", help="the fix cutoff (default generators.debug.SINCE)")
    su.add_argument("--id", default="", help="the task id (default db-<lang>-<repo>-<hash of the fixes>, as mine names it)")
    su.add_argument("--parent", default="", help="the task this one rebuilds or eases, recorded in its window")
    g = sub.add_parser("gates")
    g.add_argument("--ids", default="")
    g.add_argument("--s3-reruns", type=int, default=60)
    g.add_argument("--flake-reruns", type=int, default=30)
    g.add_argument("--redo", action="store_true")
    g.add_argument("--loop", action="store_true")
    sub.add_parser("author")
    sub.add_parser("intent")
    br = sub.add_parser("brief")
    br.add_argument("id")
    br.add_argument("--chars", type=int, default=3000)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "scout":
        cmd_scout(args)
    elif args.cmd == "mine":
        asyncio.run(mine(args))
    elif args.cmd == "subset":
        asyncio.run(subset(args))
    elif args.cmd == "gates":
        asyncio.run(gates(args))
    elif args.cmd == "author":
        cmd_author(args)
    elif args.cmd == "intent":
        cmd_intent(args)
    elif args.cmd == "brief":
        cmd_brief(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
