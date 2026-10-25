"""Rank fresh held-out sources for the swarm (``factory.heldout_supply``): disjoint owners, permissive licences, the
lanes' languages, post-cutoff fixes with tests. Discovery goes through ``gh api`` (authenticated); everything else is
local git. The output names the next held-out repositories, so it goes under the sealed dir (never git).

    uv run python scripts/heldout_supply.py [--per-lang 12] [--out <sealed>/heldout_supply.json]
    uv run python scripts/heldout_supply.py draw --slugs owner/repo,...   # hold the chosen ones out (heldout_draw.json)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from crucible.factory import heldout_supply as hs
from crucible.factory.lanes.windows import REPO_URLS, SEALED_DIR

QUERIES = {"go": "language:Go", "rust": "language:Rust", "node": "language:TypeScript", "python": "language:Python", "java": "language:Java"}


def search(q: str, n: int) -> list[dict]:
    r = subprocess.run(["gh", "api", "-X", "GET", "search/repositories", "-f", f"q={q}", "-f", "sort=updated", "-f", f"per_page={n}"],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout).get("items", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="rank", choices=("rank", "draw"))
    ap.add_argument("--per-lang", type=int, default=12, help="candidates measured per language")
    ap.add_argument("--stars", default="400..15000")
    ap.add_argument("--out", type=Path, default=SEALED_DIR / "heldout_supply.json")
    ap.add_argument("--slugs", default="")
    a = ap.parse_args()
    if a.cmd == "draw":
        rows = json.loads(a.out.read_text())["ranked"]
        chosen = [hs.Candidate(**{k: v for k, v in r.items() if k in hs.Candidate.__dataclass_fields__}) for r in rows if r["slug"] in a.slugs.split(",")]
        print(json.dumps(hs.draw(SEALED_DIR / "heldout_draw.json", chosen), indent=1))
        return
    owners = hs.registered_owners(REPO_URLS)
    found: dict[str, list[hs.Candidate]] = {}
    for lang, q in QUERIES.items():
        for lic in ("mit", "apache-2.0", "bsd-3-clause"):
            items = search(f"{q} license:{lic} stars:{a.stars} pushed:>2026-10-01 archived:false fork:false", 50)
            found.setdefault(lang, []).extend(c for c in hs.from_search(items) if c.lang == lang)
    allc = {c.slug: c for cs in found.values() for c in cs}
    print(f"{len(allc)} found; checking our history", flush=True)
    hits = hs.seen_many(HERE.parent, [x for c in allc.values() for x in (c.slug, c.owner)])
    cands: list[hs.Candidate] = []
    used_owners: set[str] = set()
    for lang, cs in found.items():
        picked = 0
        for c in sorted({c.slug: c for c in cs}.values(), key=lambda c: -c.stars):
            if c.owner in used_owners:
                continue  # one repository per owner: an owner's repositories are held out together
            c.refused = hs.refuse(c, owners=owners, seen_fn=lambda x: x.lower() in hits)
            if not c.refused:
                if picked >= a.per_lang:
                    continue
                hs.measure(c, SEALED_DIR / "supply_clones")
                if not c.refused and c.manifest and c.fixes_with_tests:
                    picked += 1
                used_owners.add(c.owner)
            cands.append(c)
            print(f"{lang:6} {c.slug:45} {c.refused or ''} {c.fixes_with_tests if c.fixes_with_tests is not None else ''}", flush=True)
    hs.write(a.out, cands, meta={"at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "cutoff": hs.CUTOFF,
                                 "rule": "owners disjoint from every registered source and from our git history; permissive licence; "
                                         "fix commits after the cutoff changing a test file and a source file"})
    r = hs.ranked(cands)
    print(f"{len(r)} ranked, {sum(1 for c in cands if c.refused)} refused -> {a.out}")


if __name__ == "__main__":
    main()
