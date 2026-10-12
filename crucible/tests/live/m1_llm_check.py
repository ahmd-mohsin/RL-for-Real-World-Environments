"""Live M1 checks with real models over OpenRouter (``python tests/live/m1_llm_check.py --budget 0.5``).

- **desktop**: a vision model (``judge-default``) drives the ``common/desktop`` Tk app from screenshots alone (no
  coordinates given): type a word, click Save, click Approve. The grader reads the files the app wrote. Needs Docker.
- **seats**: two different LLMs play Nim as two policy seats under the World's native turn order (``turns: app``):
  legal actions offered, illegal and out-of-turn moves refused as policy errors, the game reaching a terminal state with
  zero-sum rewards.
- **single_turn**: ``SingleTurnTask`` with real completions: boxed math (math-verify), multiple choice, and a JSON answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "research" / "m12-infra" / "live"


async def desktop(budget: float) -> dict:
    from crucible.actors.llm import LLMClient
    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
    from crucible.core.store import Store
    from crucible.world.world import World
    from tests.unit.test_desktop import APP, _spec

    pack = Path(tempfile.mkdtemp(prefix="desk-live-"))
    (pack / "app.py").write_text(APP)
    w = World(_spec(), store=Store(pack / "s"), workdir=pack / "w", pack_root=pack,
              instructions={"agent": "A small window is open on the screen. Type the word cobalt into its text field, click its Save button, then click "
                                     "its Approve button. Use the computer tool; look at each screenshot before you act. Say done when finished."})
    llm = LLMClient("judge-default", budget_usd=budget, cache_mode="off")
    await w.start()
    t0 = time.time()
    try:
        res = await run_episode(w, {"agent": LLMToolPolicy("agent", llm, vision=True, temperature=0.0)}, limits=EpisodeLimits(max_steps=16))
        app = w.apps["desk"]
        saved = (await app.inspect("cat /root/saved.txt 2>/dev/null")).stdout.strip()
        approved = (await app.inspect("cat /root/approved.txt 2>/dev/null")).stdout.strip()
        acts = [e.data.get("arguments", {}).get("action") for e in res.trace.events if e.type == "tool_call"]
        return {"model": llm.model, "steps": res.steps, "stop": res.stop_condition, "masked": res.masked, "actions": acts, "saved": saved, "approved": approved,
                "spend_usd": round(llm.spend.usd, 4), "s": round(time.time() - t0, 1), "pass": saved == "cobalt" and approved == "yes"}
    finally:
        await w.close()


async def seats(budget: float) -> dict:
    from crucible.actors.llm import LLMClient
    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
    from crucible.core.schema.world import WorldSpec
    from crucible.core.store import Store
    from crucible.world.world import World
    from tests.unit.test_turns import NIM  # registers the test/nim App template

    games = []
    for g in range(5):
        spec = json.loads(json.dumps(NIM))
        spec["seed"] = 100 + g
        spec["apps"][0]["params"]["pile"] = 9
        d = Path(tempfile.mkdtemp(prefix="nim-live-"))
        inst = ("You play Nim against another player. The pile starts at 9 stones; players alternate taking 1, 2 or 3 stones with the nim.take tool; "
                "whoever takes the last stone wins. Act only when it is your turn.")
        w = World(WorldSpec.model_validate(spec), instructions={"alice": inst, "bob": inst}, store=Store(d / "s"), workdir=d / "w")
        a, b = LLMClient("policy-default", budget_usd=budget / 6, cache_mode="off"), LLMClient("judge-default", budget_usd=budget / 6, cache_mode="off")
        await w.start()
        try:
            res = await run_episode(w, {"alice": LLMToolPolicy("alice", a, temperature=0.3), "bob": LLMToolPolicy("bob", b, temperature=0.3)},
                                    limits=EpisodeLimits(max_steps=40))
            errs = [e.data.get("error") for e in res.trace.events if e.type == "tool_result" and not (e.data or {}).get("ok", True)]
            moves = [(e.actor, (e.data or {}).get("arguments")) for e in res.trace.events if e.type == "tool_call"]
            games.append({"terminal": w.turns.terminal, "rewards": w.turns.rewards, "moves": moves, "refused": errs, "steps": res.steps, "masked": res.masked,
                          "spend_usd": round(a.spend.usd + b.spend.usd, 4)})
        finally:
            await w.close()
    ok = all(gm["terminal"] and sum(gm["rewards"].values()) == 0 and not gm["masked"] for gm in games)
    return {"models": ["policy-default (alice)", "judge-default (bob)"], "games": games, "pass": ok}


async def single_turn(budget: float) -> dict:
    from crucible.actors.llm import LLMClient
    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
    from crucible.runtime.single_turn import SingleTurnTask

    llm = LLMClient("policy-default", budget_usd=budget, cache_mode="off")
    tasks = [
        SingleTurnTask("live/math1@1", "What is the sum of the roots of x^2 - 7x + 10 = 0? Put the final answer in \\boxed{}.", "7", parser="boxed",
                       criteria=[{"id": "m", "checker": {"kind": "math"}}]),
        SingleTurnTask("live/math2@1", "Simplify (x^2 - 1)/(x - 1) for x != 1. Put the final answer in \\boxed{}.", "x+1", parser="boxed",
                       criteria=[{"id": "m", "checker": {"kind": "math"}}]),
        SingleTurnTask("live/mcq@1", "Which planet is largest?", "C", options={"A": "Mars", "B": "Venus", "C": "Jupiter", "D": "Mercury"},
                       parser={"kind": "mcq", "mode": "answer_colon"}, criteria=[{"id": "m", "checker": {"kind": "mcq"}}]),
        SingleTurnTask("live/json@1", 'Return a JSON object with keys "city" and "country" for the capital of Japan. Reply with the JSON only.',
                       {"city": "Tokyo", "country": "Japan"}, parser="json", criteria=[{"id": "j", "checker": {"kind": "json_subset"}}]),
    ]
    rows = []
    for t in tasks:
        d = Path(tempfile.mkdtemp(prefix="st-live-"))
        w = await t.open(d)
        try:
            await run_episode(w, {"agent": LLMToolPolicy("agent", llm, temperature=0.0)}, limits=EpisodeLimits(max_steps=2))
            g = await t.grade(w, d)
            rows.append({"task": t.id, "reward": g.reward, "gates": g.gates, "answer": (SingleTurnTask.completion(w) or "")[-120:]})
        finally:
            await w.close()
    return {"model": llm.model, "rows": rows, "spend_usd": round(llm.spend.usd, 4), "pass": sum(r["reward"] == 1.0 for r in rows) >= 3}


async def main(a) -> dict:
    out = {}
    for name, fn in (("single_turn", single_turn), ("seats", seats), ("desktop", desktop)):
        if a.only and name not in a.only.split(","):
            continue
        t0 = time.time()
        try:
            out[name] = await fn(a.budget / 3)
        except Exception as e:  # noqa: BLE001
            import traceback

            out[name] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
        out[name]["wall_s"] = round(time.time() - t0, 1)
        print(name, json.dumps({k: v for k, v in out[name].items() if k not in ("traceback",)}, default=str)[:1500], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.6)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    res = asyncio.run(main(a))
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "m1_llm.json"
    prev = json.loads(dest.read_text()) if dest.exists() else {}
    dest.write_text(json.dumps({**prev, **res}, indent=1, default=str))  # each check keeps its latest run
