"""The window lane's mining run (study 08 §9 steps 2-3; report 09).

    python scripts/lh_windows.py --work <dir> mine   [--repos a,b] [--clones <dir>] [--cap 8] [--hours 2]
    python scripts/lh_windows.py --work <dir> report [--out docs/research/task-factory/runs/window_funnel.json]
    python scripts/lh_windows.py --work <dir> author --n 4 [--usd 2]     (prompt authoring smoke test, paid)
    python scripts/lh_windows.py --work <dir> cleanup                    (removes this lane's containers, never images)

``mine`` clones what is missing (``REPO_URLS``), mines windows per repository, builds the World image at A, runs the
leak probes, the fail-to-pass / pass-to-pass reruns, the dependency prober and the commit-revert mutants, and writes one
record per window under ``<work>/store/{train,heldout}/tasks/<id>/``. Held-out repositories (``HELD_OUT_REPOS``, fixed
before mining) only ever go to the held-out store. ``report`` writes the funnel per stage per repository.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from crucible.factory.lanes.windows import (
    HELD_OUT_REPOS,
    POOL_REPOS,
    REPO_URLS,
    SMALL_REPOS,
    ImageBook,
    WindowStore,
    funnel,
    mine,
    open_store,
    review_realism,
    sandbox_factory,
    split_of,
)
from crucible.factory.lanes.windows_git import RepoGit
from crucible.factory.lanes.windows_world import PREFIX, docker, own_containers


def clones(root: Path, names: list[str]) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for n in names:
        d = root / n
        if not d.exists():
            subprocess.run(["git", "clone", "-q", f"https://github.com/{REPO_URLS[n]}.git", str(d)], check=True)
        out[n] = d
    return out


def logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(ev: dict) -> None:
        with path.open("a") as fh:
            fh.write(json.dumps({"t": round(time.time(), 1), **ev}, default=str) + "\n")
        print(json.dumps(ev, default=str)[:300], flush=True)

    return log


async def cmd_mine(a: argparse.Namespace) -> None:
    work = Path(a.work)
    names = a.repos.split(",") if a.repos else [*SMALL_REPOS, *HELD_OUT_REPOS, *POOL_REPOS]
    repos = clones(Path(a.clones) if a.clones else work / "repos", names)
    deadline = time.time() + a.hours * 3600 if a.hours else None
    images = ImageBook(parallel=a.builds)
    sandboxes = sandbox_factory(parallel=a.parallel)
    log = logger(work / "mine_log.jsonl")
    sem = asyncio.Semaphore(a.windows)  # every repository's windows share one limit, each into its own split's store

    async def one(name: str, path: Path) -> None:
        store = open_store(work / "store", name)
        t0 = time.time()
        n = await mine({name: path}, store, images=images, sandboxes=sandboxes, cap=a.cap, since=a.since, reruns=a.reruns,
                       deadline=deadline, log=log, window_sem=sem, redo=("deadline", "error", *[x for x in a.redo.split(",") if x]))
        log({"event": "repo_done", "repo": name, "split": split_of(name), "windows": n[name], "seconds": round(time.time() - t0, 1)})

    await asyncio.gather(*[one(n, p) for n, p in repos.items()])


def cmd_report(a: argparse.Namespace) -> None:
    work = Path(a.work)
    out = {}
    for split in ("train", "heldout"):
        root = work / "store" / split
        if (root / "split.json").exists():
            out[split] = funnel(WindowStore(root, split=split))
    out["held_out_repos"] = HELD_OUT_REPOS
    text = json.dumps(out, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)


async def cmd_author(a: argparse.Namespace) -> None:
    """Prompt authoring on a handful of accepted training windows: author (qwen3.7-flash) then realism (deepseek)."""
    from crucible.factory.models import ModelBook

    work = Path(a.work)
    store = WindowStore(work / "store" / "train", split="train")
    book = ModelBook.from_config(budget_usd_per_client=a.usd / 2)
    author, realism = book.client("author"), book.client("realism")
    ids = [w for w in store.window_ids() if store.window(w).get("stage") == "mutants_valid"][: a.n]
    rows = []
    for wid in ids:
        from crucible.factory.lanes.windows import author_window

        rec = store.window(wid)
        repo = RepoGit((Path(a.clones) if a.clones else work / "repos") / rec["repo"])
        findings: list[str] = []
        res = None
        for k in range(1, 3):
            try:
                res = await author_window(author, rec, repo, store.dir(wid) / "base", findings=findings, sample_key=f"lh-author-{wid}-{k}" if k > 1 else None)
            except Exception as e:  # noqa: BLE001
                findings = [f"error: {e}"[:300]]
                continue
            findings = res["findings"]
            if not findings:
                break
        row = {"window": wid, "findings": findings, "prompt": res["prompt"] if res else None, "requirements": res["requirements"] if res else None}
        if res and not findings:
            row["realism"] = await review_realism(realism, res["prompt"])
        rows.append(row)
        (store.dir(wid) / "prompt.json").write_text(json.dumps(row, indent=1))
        print(json.dumps({k: row[k] for k in ("window", "findings")}), flush=True)
    spend = {r: book.client(r).spend.usd for r in ("author", "realism")}
    (work / "author_smoke.json").write_text(json.dumps({"rows": rows, "spend_usd": spend}, indent=1))
    print("spend", spend)


def cmd_cleanup(a: argparse.Namespace) -> None:
    names = own_containers()
    if names:
        docker("rm", "-f", *names, timeout=300)
    print(f"removed {len(names)} {PREFIX}* containers")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--clones", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--repos", default="")
    m.add_argument("--cap", type=int, default=8)
    m.add_argument("--since", default="2019-01-01")
    m.add_argument("--reruns", type=int, default=3)
    m.add_argument("--hours", type=float, default=2.0)
    m.add_argument("--parallel", type=int, default=8, help="concurrent test containers")
    m.add_argument("--windows", type=int, default=4, help="concurrent windows")
    m.add_argument("--builds", type=int, default=3, help="concurrent image builds")
    m.add_argument("--redo", default="", help="comma-separated reason prefixes to validate again (after a lane fix)")
    r = sub.add_parser("report")
    r.add_argument("--out", default=None)
    au = sub.add_parser("author")
    au.add_argument("--n", type=int, default=4)
    au.add_argument("--usd", type=float, default=2.0)
    sub.add_parser("cleanup")
    a = ap.parse_args()
    if a.cmd == "mine":
        asyncio.run(cmd_mine(a))
    elif a.cmd == "report":
        cmd_report(a)
    elif a.cmd == "author":
        asyncio.run(cmd_author(a))
    elif a.cmd == "cleanup":
        cmd_cleanup(a)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
