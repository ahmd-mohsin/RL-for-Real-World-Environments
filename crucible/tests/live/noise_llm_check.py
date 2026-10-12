"""Live LLM checks over OpenRouter for M12-infra B and D.

    python tests/live/noise_llm_check.py noise --seeds 10 --budget 1.5
    python tests/live/noise_llm_check.py judges --budget 0.5

- **noise**: an LLM agent (``policy-default``) on toy enterprise Worlds (the helpdesk backlog, 3 tickets; the workspace
  word-file task) at World noise intensity 0, 0.5 and 1.0 (``world.noise.WorldNoise``: instruction typos and
  distractors, transient tool errors, damaged tool output). Records the reward, steps, tool errors, the noise events
  in the trace, and whether any episode was masked (noise must never mask). The clean / noisy gap is what
  ``NoiseCurriculum`` steers on.
- **judges**: the group operators with real LLM judges (``judge-default``) on a group of answers of known quality:
  ``bt_pairwise`` against a reference, a round-robin ``tournament``, and ``grs`` rubric scores, then ``gar`` on top.
  The judges must rank the known-good answer above the known-bad one, and a failed test must stay at reward 0 under
  GRS whatever the rubric says.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "m12-infra" / "live"


async def noise(a) -> dict:
    from crucible_pack_toy.longhorizon.task import BacklogTask
    from crucible_pack_toy.runtime import WordFileTask

    from crucible.actors.llm import LLMClient
    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
    from crucible.world.noise import NoiseConfig, WorldNoise

    llm = LLMClient("policy-default", budget_usd=a.budget, cache_mode="off")
    rows: list[dict] = []
    sem = asyncio.Semaphore(6)

    async def one(kind: str, seed: int, level: float) -> None:
        async with sem:
            task = BacklogTask(seed=seed, n=3) if kind == "backlog" else WordFileTask(seed=seed)
            d = Path(tempfile.mkdtemp(prefix="noise-live-"))
            w = await task.open(d)
            protect = [task.word] if kind == "wordfile" else []  # the exact value the task asks for (names, ids and numbers are protected by default)
            w.noise = WorldNoise(NoiseConfig(intensity=level, protect=protect), seed=f"{kind}/{seed}") if level > 0 else None
            t0 = time.time()
            try:
                res = await run_episode(w, {"agent": LLMToolPolicy("agent", llm, temperature=0.3)}, limits=EpisodeLimits(max_steps=30 if kind == "backlog" else 12))
                g = await task.grade(w, d)
                ev = res.trace.events
                rows.append({"task": kind, "seed": seed, "noise": level, "reward": g.reward, "steps": res.steps, "masked": res.masked, "stop": res.stop_condition,
                             "tool_errors": sum(1 for e in ev if e.type == "tool_result" and not (e.data or {}).get("ok", True)),
                             "noise_events": sum(1 for e in ev if e.type == "noise"),
                             "noise_ops": sorted({op for e in ev if e.type == "noise" for op in (e.data or {}).get("ops", [])}), "s": round(time.time() - t0, 1)})
            except Exception as e:  # noqa: BLE001
                rows.append({"task": kind, "seed": seed, "noise": level, "error": f"{type(e).__name__}: {e}"[:300]})
            finally:
                await w.close()

    await asyncio.gather(*(one(k, s, lv) for k in ("backlog", "wordfile") for s in range(a.seeds) for lv in (0.0, 0.5, 1.0)))
    summary = {}
    for k in ("backlog", "wordfile"):
        for lv in (0.0, 0.5, 1.0):
            rs = [r for r in rows if r["task"] == k and r["noise"] == lv and "reward" in r]
            summary[f"{k}@{lv}"] = {"n": len(rs), "mean_reward": round(sum(r["reward"] for r in rs) / len(rs), 3) if rs else None,
                                    "masked": sum(r["masked"] for r in rs), "mean_noise_events": round(sum(r["noise_events"] for r in rs) / max(1, len(rs)), 1),
                                    "errors": sum(1 for r in rows if r["task"] == k and r["noise"] == lv and "error" in r)}
    return {"model": llm.model, "spend_usd": round(llm.spend.usd, 4), "summary": summary, "rows": rows,
            "pass": all(v["masked"] == 0 for v in summary.values()) and all(v["errors"] == 0 for v in summary.values())
            and summary["backlog@0.0"]["mean_noise_events"] == 0 and summary["backlog@1.0"]["mean_noise_events"] > 0}


QUESTION = "A shop sells pens at 3 for $2. How much do 12 pens cost? Explain briefly."
ANSWERS = {  # known quality: good (right, clear), ok (right, sloppy), wrong, wrong-and-confident
    "good": "12 pens is 4 groups of 3 pens. Each group costs $2, so 4 × $2 = $8.",
    "ok": "pens 3 = 2 dollars so 12 is like 8 dollars i think",
    "wrong": "Each pen costs $2/3, so 12 pens cost 12 × 3/2 = $18.",
    "bad": "$24, because 12 × 2 = 24.",
}
REFERENCE = "12 pens = 4 sets of 3; 4 × $2 = $8."


async def judges(a) -> dict:
    from crucible.actors.llm import LLMClient
    from crucible.core.schema.rollout import RolloutRecord
    from crucible.train.group_ops import bt_pairwise, compose, gar, grs, tournament, zero_hacks

    judge = LLMClient("judge-default", budget_usd=a.budget, cache_mode="off")

    async def ask(prompt: str) -> str:
        r = await judge.complete([{"role": "user", "content": prompt}], max_tokens=400, temperature=0.0)
        return r.text or ""

    def num(text: str, default: float = 0.5) -> float:
        m = re.findall(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", text)
        return float(m[-1]) if m else default

    def recs():
        out = []
        for k in ANSWERS:
            r = RolloutRecord(rollout_id=k, sample=k, task="pens", harness="live", policy="p", turns=[], reward=1.0 if k in ("good", "ok") else 0.0)
            out.append(r)
        return out

    async def bt_judge(rec, ref):
        t = await ask(f"Question: {QUESTION}\nReference answer: {ref}\nCandidate answer: {ANSWERS[rec.rollout_id]}\n\nHow likely is it that the candidate answer is at least as good "
                      "as the reference (correct and clearly explained)? Reply with a single probability between 0 and 1 on the last line.")
        return num(t)

    async def compare(x, y):
        t = await ask(f"Question: {QUESTION}\nAnswer A: {ANSWERS[x.rollout_id]}\nAnswer B: {ANSWERS[y.rollout_id]}\n\nWhich answer is better (correct first, then clear)? "
                      "Reply with exactly one of: A, B, TIE on the last line.")
        last = t.strip().splitlines()[-1].upper() if t.strip() else "TIE"
        return 1.0 if "A" in last and "B" not in last else (0.0 if "B" in last and "A" not in last else 0.5)

    async def rubric(rec):
        t = await ask(f"Question: {QUESTION}\nAnswer: {ANSWERS[rec.rollout_id]}\n\nScore the solution quality (reasoning shown, correct) and the behaviour (clear, no "
                      "hedging) each in [0, 1]. Reply on the last line as: solution=<x> behaviour=<y>")
        s = re.search(r"solution\s*=\s*([01](?:\.\d+)?)", t)
        b = re.search(r"behaviour\s*=\s*([01](?:\.\d+)?)", t)
        return (float(s.group(1)) if s else 0.5, float(b.group(1)) if b else 0.5)

    bt = await bt_pairwise(bt_judge, REFERENCE)("g", recs())
    tr = await tournament(compare)("g", recs())
    chain = await compose(zero_hacks(lambda r: r.rollout_id == "bad"), grs(rubric), gar(lambda r: max(0.05, r.reward_components.get("grs/solution", 1.0))))("g", recs())
    by = lambda rs: {r.rollout_id: round(float(r.reward or 0.0), 3) for r in rs}
    adv = {r.rollout_id: round(float(r.advantage), 3) if r.advantage is not None else None for r in chain}
    out = {"model": judge.model, "spend_usd": round(judge.spend.usd, 4), "bt_pairwise": by(bt), "tournament": by(tr), "grs_gar_reward": by(chain), "grs_gar_advantage": adv}
    out["pass"] = bool(out["bt_pairwise"]["good"] > out["bt_pairwise"]["bad"] and out["tournament"]["good"] > out["tournament"]["wrong"]
                       and out["grs_gar_reward"]["wrong"] == 0.0 and out["grs_gar_reward"]["bad"] == 0.0 and adv["good"] >= adv["ok"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["noise", "judges"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=float, default=1.5)
    a = ap.parse_args()
    t0 = time.time()
    res = asyncio.run(noise(a) if a.which == "noise" else judges(a))
    res["wall_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"llm_{a.which}.json").write_text(json.dumps(res, indent=1, default=str))
    show = {k: v for k, v in res.items() if k != "rows"}
    print(json.dumps(show, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
