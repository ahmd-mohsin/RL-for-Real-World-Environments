#!/usr/bin/env python3
"""The Tinker swarm: its role map, its smoke, its credit and its self-diagnosis (swarm-pipeline/13).

    python scripts/swarm.py roles [--preset swarm]          # the role -> model map, families, the family rule's verdict
    python scripts/swarm.py smoke --out DIR [--cap 5]       # one JSON call per role model; one prompt-writer session per builder family
                                                            # (DeepSeek, GLM, Kimi) on the RLM harness, offline sandbox, forced compaction
    python scripts/swarm.py diagnose --root JOURNAL [--md]  # per-role/model yield and cost, blame, bottlenecks, proposals (never applied)
    python scripts/swarm.py credit --root JOURNAL           # per-turn builder credit (Φ deltas booked on the closing call)

Keys come from the environment only (``set -a; . ../.env; set +a``); the Tinker meter's hard cap is
``CRUCIBLE_TINKER_CAP_USD`` (``smoke`` sets it to ``--cap``). Nothing here edits a gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]


def cmd_roles(a: argparse.Namespace) -> None:
    from crucible.factory.models import ModelBook

    book = ModelBook(preset=a.preset)
    print(json.dumps({**book.describe(), "builder_families": book.builder_families(), "family_problems": book.family_problems()}, indent=1))


def cmd_diagnose(a: argparse.Namespace) -> None:
    from crucible.factory.swarm_journal import diagnose, render

    d = diagnose(a.root)
    print(render(d) if a.md else json.dumps(d, indent=1, default=str))


def cmd_credit(a: argparse.Namespace) -> None:
    from crucible.factory.swarm_journal import builder_credit

    for row in builder_credit(a.root):
        print(json.dumps(row, default=str))


# ---- the smoke ----------------------------------------------------------------------------------------------------------------------
CALC = '''"""A tiny calculator library."""


def mean(xs):
    """The arithmetic mean of a non-empty list."""
    return sum(xs) / (len(xs) + 1)


def clamp(x, lo, hi):
    """x limited to [lo, hi]."""
    return max(lo, min(x, lo))
'''
SMOKE_BRIEF = {
    "task": "smoke-calc", "lane": "smoke", "language": "python", "repository": "local/calc", "milestone_ids": ["m1", "m2"],
    "milestones": [
        {"id": "m1", "requires": [], "fix_message": "fix mean: divide by len(xs)",
         "source_diff": "-    return sum(xs) / (len(xs) + 1)\n+    return sum(xs) / len(xs)", "test_diff": "+    assert mean([2, 4]) == 3"},
        {"id": "m2", "requires": [], "fix_message": "fix clamp: upper bound", "source_diff": "-    return max(lo, min(x, lo))\n+    return max(lo, min(x, hi))",
         "test_diff": "+    assert clamp(5, 0, 10) == 5"}],
    "forbidden_strings": ["sum(xs) / (len(xs) + 1)", "min(x, lo)"], "calibration": "",
    "note": "Before writing, print README.md in full, 3000 characters per cell (one cell per part, about eight cells): it is the users' own guide and the report must use their words. Then read calc.py."}


def smoke_lint(text: str) -> list[str]:
    out = [f"lint hidden: {s!r}" for s in SMOKE_BRIEF["forbidden_strings"] if s in text]
    out += [f"lint leak: {w!r}" for w in ("len(xs) + 1", "min(x, lo)") if w in text]
    return out


def smoke_check(draft: dict[str, Any]) -> list[str]:
    reqs = draft.get("requirements") or {}
    found = [f"unmapped_milestone: {m}" for m in SMOKE_BRIEF["milestone_ids"] if not str(reqs.get(m) or "").strip()]
    if not str(draft.get("opening") or "").strip():
        found.append("author.opening is empty")
    return found + smoke_lint(json.dumps(draft))


async def smoke_json(book: Any, out: Path) -> list[dict[str, Any]]:
    from crucible.actors.llm import SpendScope

    rows, seen = [], {}
    for role, spec in sorted(book.roles.items()):
        if spec.ref in seen:  # one call per distinct model; the other roles on it share its client class and its result
            rows.append({**seen[spec.ref], "role": role, "shared_with": seen[spec.ref]["role"]})
            continue
        t0, scope = time.monotonic(), SpendScope()
        row: dict[str, Any] = {"role": role, "model": spec.ref, "family": spec.family}
        try:
            with scope:
                d = await book.client(role).complete_json([{"role": "user", "content": f'You are the {role} of a task factory. Reply with {{"ok": true, "role": "{role}"}}.'}],
                                                          schema=["ok"], max_tokens=1500)
            row.update(ok=bool(d.get("ok")), reply=d)
        except Exception as e:  # noqa: BLE001 -- the smoke records every failure
            row.update(ok=False, error=f"{type(e).__name__}: {e}"[:400])
        c = book.client(role)
        log = getattr(getattr(c, "sampler", None), "log", None) or []
        row.update(usd=round(scope.usd, 6), seconds=round(time.monotonic() - t0, 2), prompt_tokens=sum(x.get("prompt", 0) for x in log),
                   completion_tokens=sum(x.get("completion", 0) for x in log))
        seen[spec.ref] = row
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("role", "model", "ok", "usd", "prompt_tokens", "completion_tokens", "error")}), flush=True)
    (out / "json_roles.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    return rows


async def smoke_sessions(book: Any, out: Path, *, window: int, max_turns: int, budget: float, families: Any = ("deepseek", "glm", "kimi")) -> list[dict[str, Any]]:
    import dataclasses

    from crucible.actors.rlm.agent import RLMConfig
    from crucible.factory.models import DEEPSEEK, GLM, KIMI, ModelBook
    from crucible.factory.swarm_journal import SwarmJournal
    from crucible.factory.swarm_prompts import PromptJob, write_prompt

    tree = out / "calc-tree"
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "calc.py").write_text(CALC)
    # a long user guide the writer reads for the user's voice: its tool output pushes the context past the window's
    # threshold mid-session, so the compaction path runs with a real summary (the hand-off alone is ~3K tokens)
    (tree / "README.md").write_text("# calc\n\nA tiny calculator library used by the reporting team.\n\n" + "\n".join(
        f"## Recipe {i}\n\nTo compute the average of the weekly totals for team {i}, call `mean(totals)`; to keep a score inside the allowed "
        f"band, call `clamp(score, 0, {10 + i})`. Both are used in the nightly report and in the dashboard export." for i in range(110)))
    journal = SwarmJournal(out / "journal")
    cfg = RLMConfig(temperature=0.0, tree_max_depth=0, max_turns=max_turns, window_tokens=window, max_tokens=2000, wall_s=1200.0, cell_timeout_s=120.0)
    rows = []
    for fam, ref in [(f, r) for f, r in (("deepseek", DEEPSEEK), ("glm", GLM), ("kimi", KIMI)) if f in families]:
        b = ModelBook({"prompt_author": ref}, preset="swarm")  # the family rule still holds: the reviewer is Nemotron-Ultra
        job = PromptJob(f"smoke-calc-{fam}", ["m1", "m2"], {**SMOKE_BRIEF, "task": f"smoke-calc-{fam}"}, tree, smoke_check, smoke_lint, out / "prompts" / f"{fam}.json")
        t0 = time.monotonic()
        try:
            res = await write_prompt(job, book=b, root=out / "swarm", journal=journal, budget_usd=budget, max_turns=max_turns, max_writes=1,
                                     rlm_config=dataclasses.replace(cfg))
        except Exception as e:  # noqa: BLE001 -- the smoke records every failure
            res = {"task": job.task_id, "status": "error", "error": f"{type(e).__name__}: {e}"[:600], "usd": 0.0}
        res["seconds"] = round(time.monotonic() - t0, 1)
        res["family"] = fam
        rows.append(res)
        print(json.dumps({k: res.get(k) for k in ("task", "status", "reason", "usd", "sessions", "phi_final", "error", "seconds")}, default=str), flush=True)
    (out / "sessions.json").write_text(json.dumps(rows, indent=1, default=str))
    return rows


def load_env_file(path: str) -> list[str]:
    """``KEY=VALUE`` lines of ``path`` into the environment (the same as ``set -a; . path; set +a`` for plain assignments),
    keeping variables already set; returns the names it set, never a value."""
    names = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, _, v = ln.removeprefix("export ").partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.isidentifier() and k not in os.environ:
            os.environ[k] = v
            names.append(k)
    return names


def cmd_smoke(a: argparse.Namespace) -> None:
    if a.env_file:
        load_env_file(a.env_file)
    os.environ["CRUCIBLE_TINKER_CAP_USD"] = str(a.cap)
    os.environ.setdefault("CRUCIBLE_TINKER_SPEND", str(Path(a.out) / "tinker_spend.jsonl"))
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not in the environment (set -a; . ../.env; set +a)")
    from crucible.factory.models import ModelBook
    from crucible.factory.sessions import ensure_image
    from crucible.train.bridges.tinker.sampler import shared_meter

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    book = ModelBook(preset="swarm")
    res: dict[str, Any] = {"roles": book.describe()["roles"]}
    if "json" in a.parts:
        res["json"] = asyncio.run(smoke_json(book, out))
    if "sessions" in a.parts:
        ensure_image()
        res["sessions"] = asyncio.run(smoke_sessions(book, out, window=a.window, max_turns=a.max_turns, budget=a.budget, families=a.families))
    t = shared_meter().totals
    res["tinker_usd"] = round(t.usd, 6)
    res["tinker_by_model_usd"] = {k: round(v, 6) for k, v in t.by_model_usd.items()}
    res["tinker_tokens"] = {"prefill": t.prefill, "prefill_cached": t.prefill_cached, "sample": t.sample, "calls": t.calls}
    (out / "smoke.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"tinker_usd": res["tinker_usd"], "out": str(out)}))


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(HERE / "src"))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("roles")
    s.add_argument("--preset", default="swarm")
    s = sub.add_parser("diagnose")
    s.add_argument("--root", required=True)
    s.add_argument("--md", action="store_true")
    s = sub.add_parser("credit")
    s.add_argument("--root", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--out", required=True)
    s.add_argument("--cap", type=float, default=5.0)
    s.add_argument("--parts", nargs="*", default=["json", "sessions"])
    s.add_argument("--window", type=int, default=12_000, help="the sessions' compaction window (small: compaction is forced)")
    s.add_argument("--max-turns", type=int, default=24)
    s.add_argument("--families", nargs="*", default=["deepseek", "glm", "kimi"], help="the builder families whose sessions run")
    s.add_argument("--budget", type=float, default=0.8, help="USD per session machine")
    s.add_argument("--env-file", default="", help="KEY=VALUE file loaded into the environment first (values never printed)")
    a = ap.parse_args(argv)
    {"roles": cmd_roles, "diagnose": cmd_diagnose, "credit": cmd_credit, "smoke": cmd_smoke}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
