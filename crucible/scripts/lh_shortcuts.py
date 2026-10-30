"""Closed-loop shortcut discovery (Part 1 V11, ``factory.shortcuts``): label, propose, accept.

    python scripts/lh_shortcuts.py label --roots DIR [DIR ...] --out data/observe/shortcuts/labelled-<date>.json [--labels OVERRIDES.json]
    python scripts/lh_shortcuts.py propose --roots INTERVAL_DIR [...] --labelled FILE --out RUN_DIR --cap USD [--env-file ../.env]
        [--role auditor] [--preset tinker] [--batch 6] [--max-episodes 60]
    python scripts/lh_shortcuts.py accept --run RUN_DIR [--dry-run] [--ids l1a2b3c4d ...] [--repath-roots DIR ... --store STORE]

``label``: freeze the labelled set from stored episodes (no model call; never overwrites a file). ``--labels``: a JSON
object, attempt directory -> honest | hack | attack | gate, a person's overrides of the stored verdicts.

``propose`` (paid, capped): the auditor role reads the interval's passing and flagged episodes and proposes patterns; each
is validated at once against the frozen set (free). Writes ``RUN_DIR/run.json`` and ``RUN_DIR/validation.json``.

``accept``: validates the run's proposals again (the labelled set and the interval's episodes are re-read, so an edited
``validation.json`` changes nothing), appends the ones that hold to ``src/crucible/factory/learned_path_rules.json`` (review
the diff, then commit), and with ``--store`` re-paths the stored episodes under the new rules (``RUN_DIR/repath.json``).
``--dry-run``: validate again and print, write nothing. Read each rule that holds before accepting it: the frozen set proves
a rule fires on no honest episode, not that what it catches is a shortcut (``--ids`` takes the ones that are).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from train_monitor import load_env_file


def cmd_label(a: argparse.Namespace) -> None:
    from crucible.factory.shortcuts import episodes, freeze

    labels = json.loads(Path(a.labels).read_text()) if a.labels else None
    eps = episodes(a.roots, labels=labels)
    sha = freeze(eps, Path(a.out), roots=a.roots)
    counts = {lab: sum(1 for e in eps if e.label == lab) for lab in ("honest", "hack", "attack", "gate")}
    print(json.dumps({"out": a.out, "episodes": len(eps), "counts": counts, "sha256": sha}, indent=1))


def _validate_all(rules: list[dict[str, Any]], roots: list[str], labelled_path: str) -> tuple[list[dict[str, Any]], str]:
    from crucible.factory.pathcheck import LearnedRule, load_learned
    from crucible.factory.shortcuts import episodes, load_labelled, validate

    labelled, sha = load_labelled(Path(labelled_path))
    interval = {e.id: e for e in episodes(roots)}
    current = load_learned()
    out = []
    for d in rules:
        r = LearnedRule.from_json(d) if d.get("regex") else None
        if r is None:
            continue
        out.append(validate(r, [interval[i] for i in r.episodes if i in interval], labelled, labelled_sha=sha, current=current))
    return out, sha


def cmd_propose(a: argparse.Namespace) -> None:
    from crucible.factory.shortcuts import episodes, load_labelled, propose

    load_labelled(Path(a.labelled))  # refuse a changed set before spending anything
    if a.env_file:
        load_env_file(a.env_file)
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment (--env-file ../.env)")
    from crucible.factory.models import ModelBook, PromptJsonClient
    from crucible.train.bridges.tinker.sampler import ensure_cap, shared_meter

    ensure_cap(a.cap)
    book = ModelBook(preset=a.preset)
    spec = book.spec(a.role)
    start = shared_meter().totals.usd
    res = asyncio.run(propose(episodes(a.roots), PromptJsonClient(book.client(a.role)), model=spec.model, cap_usd=a.cap,
                              spent=lambda: shared_meter().totals.usd - start, batch=a.batch, max_episodes=a.max_episodes))
    res["usd"] = round(shared_meter().totals.usd - start, 4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({**res, "roots": a.roots, "labelled": a.labelled}, indent=1))
    val, sha = _validate_all(res["rules"], a.roots, a.labelled)
    (out / "validation.json").write_text(json.dumps(val, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rules"} | {"proposed": len(res["rules"]), "hold": sum(1 for v in val if v["ok"]),
                      "labelled_set": sha}, indent=1))
    for v in val:
        print(f"  {'HOLDS' if v['ok'] else 'no   '} {v['rule']['id']} {v['rule']['on']}:{v['rule']['regex']!r} ({v['why']})")


def cmd_accept(a: argparse.Namespace) -> None:
    from crucible.factory.pathcheck import LEARNED_PATH_RULES, load_learned, rules_version
    from crucible.factory.shortcuts import accept

    run = json.loads((Path(a.run) / "run.json").read_text())
    val, _ = _validate_all(run["rules"], run["roots"], run["labelled"])
    if a.dry_run:
        for v in val:
            print(f"  {'HOLDS' if v['ok'] else 'no   '} {v['rule']['id']} {v['rule']['on']}:{v['rule']['regex']!r} ({v['why']})")
        return
    new = accept(val, ids=a.ids or ())
    learned = load_learned()
    print(json.dumps({"accepted": [r.id for r in new], "rules_file": str(LEARNED_PATH_RULES), "rules": len(learned),
                      "rules_version": rules_version(learned)}, indent=1))
    if a.store and new:
        from crucible.factory.repath import scan

        res = scan([Path(r) for r in a.repath_roots or run["roots"]], Path(a.store), learned=learned)
        (Path(a.run) / "repath.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "flips"}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("label")
    s.add_argument("--roots", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--labels", help="JSON: attempt directory -> honest | hack | attack | gate")
    s.set_defaults(fn=cmd_label)
    s = sub.add_parser("propose")
    s.add_argument("--roots", nargs="+", required=True, help="the interval's episodes")
    s.add_argument("--labelled", required=True, help="the frozen labelled set (label --out)")
    s.add_argument("--out", required=True)
    s.add_argument("--cap", type=float, required=True, help="USD cap on the auditor's calls")
    s.add_argument("--env-file")
    s.add_argument("--role", default="auditor")
    s.add_argument("--preset", default="tinker")
    s.add_argument("--batch", type=int, default=6)
    s.add_argument("--max-episodes", type=int, default=60)
    s.set_defaults(fn=cmd_propose)
    s = sub.add_parser("accept")
    s.add_argument("--run", required=True, help="a propose --out directory")
    s.add_argument("--ids", nargs="*", help="accept only these rule ids (default: every rule that holds)")
    s.add_argument("--dry-run", action="store_true", help="validate again and print; write nothing")
    s.add_argument("--store", help="the debug lane's window store, to re-path stored episodes under the new rules")
    s.add_argument("--repath-roots", nargs="*", help="default: the run's interval roots")
    s.set_defaults(fn=cmd_accept)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
