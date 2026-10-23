"""Real links between one repository's fixes: the debug lane's own fix check (``Debug._executed``) over every fix unit since
the cutoff at B = HEAD minus ``days``, edges from ``interaction_edges`` (parent tests, tests that only stopped building and
tests both fixes wrote are no link). Lists the usable fixes and every linked pair, marking the given fixes (a task's).
Writes ``$CRUCIBLE_DEBUG_ROOT/search/<repo>-links-<days>d.json``. Uses Docker (the repo's World image), no model call.

    cd crucible && set -a && . ../.env && set +a
    .venv/bin/python scripts/lh_debug_links.py <repo> <lang> <since> <days> [task fix prefixes,...]
    .venv/bin/python scripts/lh_debug_links.py pulldown-cmark rust 2025-09-01 0 671cf7c,7e01b77,0424329,def3889
"""

import asyncio
import datetime as dt
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts")
import lh_debug as L

from crucible.factory.generators.base import git, out
from crucible.factory.generators.debug import scope
from crucible.factory.generators.rebreak import fix_units
from crucible.factory.lanes.windows import subtests_of

if len(sys.argv) < 5:
    sys.exit(__doc__)
repo, lang, since, days = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ahmed = [p for p in (sys.argv[5] if len(sys.argv) > 5 else "").split(",") if p]


async def main() -> None:
    gen = L.generator([repo], L.split_of(repo), max_pool=500)
    d = gen.clones[repo]
    with scope(lang):
        head = out(git(d, "rev-parse", "HEAD"))
        if days:
            cut = (dt.datetime.fromisoformat(out(git(d, "log", "-1", "--format=%cI", head))) - dt.timedelta(days=days)).isoformat()
            head = out(git(d, "rev-list", "--first-parent", "-1", f"--before={cut}", head))
        fixes = fix_units(d, head, since=since)
        print(f"[{repo}] {len(fixes)} fixes since {since} at {head[:10]} (HEAD-{days}d)", flush=True)
        kept, need, _ = await gen._executed(repo, lang, head, f"HEAD-{days}d" if days else "HEAD", fixes)
    work = Path(os.environ["CRUCIBLE_DEBUG_ROOT"]) / "gen" / "work" / repo / head[:10]
    evid = json.loads(max(work.glob("evidence-*.json"), key=lambda p: p.stat().st_mtime).read_text())[0]
    subj = {u.sha: u.subject for u in fixes}
    pairs = []
    for a in sorted(need):
        for b in sorted(need[a]):
            if a < b:
                ca = set(evid[a]["caught"]) - set(evid[a].get("caught_unbuilt") or [])
                cb = set(evid[b]["caught"]) - set(evid[b].get("caught_unbuilt") or [])
                wa, wb = set(evid[a].get("caught_written") or []), set(evid[b].get("caught_written") or [])
                shared = sorted(t for t in ca & cb if not subtests_of(t, ca | cb) and not (t in wa and t in wb))
                pairs.append({"a": a, "b": b, "a_subject": subj[a], "b_subject": subj[b], "shared": shared[:6], "n_shared": len(shared),
                              "a_wrote": [t for t in shared if t in wa][:3], "b_wrote": [t for t in shared if t in wb][:3],
                              "ahmed": [s[:7] for s in (a, b) if any(s.startswith(p) for p in ahmed)]})
    res = {"repo": repo, "b": head, "days": days, "fixes": len(fixes), "kept": [u.sha for u in kept],
           "why_not_kept": dict(Counter(ev.get("why") or "ok" for ev in evid.values())), "pairs": pairs}
    dest = Path(os.environ["CRUCIBLE_DEBUG_ROOT"]) / "search" / f"{repo}-links-{days}d.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(res, indent=1))
    print(f"[{repo}] kept {len(kept)} of {len(fixes)}; not kept: {res['why_not_kept']}; {len(pairs)} real linked pairs", flush=True)
    for p in pairs:
        tag = f"  <- task {p['ahmed']}" if p["ahmed"] else ""
        print(f"  {p['a'][:7]} {p['a_subject'][:55]!r} x {p['b'][:7]} {p['b_subject'][:55]!r}: shared {p['n_shared']}{tag}", flush=True)
    print(f"[{repo}] written {dest}", flush=True)


asyncio.run(main())
