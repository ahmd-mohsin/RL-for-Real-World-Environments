"""m13 G-8: the swarm stage runner. A Proposer swarm at C=5, B=30 yields >= 10 distinct composition plans with <= 10%
announce collisions; abstentions do not consume B; collisions are cancelled before editing (no duplicate plans even
from workers that ignore the dashboard); breaches and invalid contributions are charged and rejected; readout ties
break on the smallest sha."""

from __future__ import annotations

import asyncio
import json
import random

from crucible.forge.repo import ForgeRepo
from crucible.forge.swarm import Abstain, Contribute, Nominate, SwarmStage, winner

TASKS = [f"t{i}" for i in range(8)]


def _repo(tmp_path):
    r = ForgeRepo(tmp_path / "toy", family="toy")
    seed = r.commit("proposer", {f"public/tasks/{t}.md": f"base task {t}: resolve the ticket backlog item {t}" for t in TASKS}, type_="seeds", summary="seeds")
    return r, seed.sha


def _validate(files):
    out = []
    for p, v in files.items():
        if p.startswith("public/plans/"):
            d = json.loads(v)
            if d.get("operator") not in ("chain", "fan_in") or len(d.get("components", [])) < 2:
                out.append(f"{p}: a plan needs an operator and >= 2 components")
    return out


def proposer(seed_sha, *, polite=True, abstain_p=0.25):
    def make(wid):
        rnd = random.Random(wid)

        async def run(ep):
            view = ep.view(seed_sha)
            tasks = sorted(p.split("/")[-1][:-3] for p in view if p.startswith("public/tasks/"))
            done = set()
            for sha in ep.heads():
                done |= {p.split("/")[-1][:-5] for p in ep.stage.repo.view("proposer", sha) if p.startswith("public/plans/")}
            busy = {g.split(" | ")[0].split()[-1] for g in ep.dashboard["in_flight"]} if polite else set()
            free = [f"chain_{a}_{b}" for a in tasks for b in tasks if a != b and f"chain_{a}_{b}" not in done | busy]
            if not free or rnd.random() < abstain_p:
                return Abstain("nothing I can add")
            pick = rnd.choice(free)
            await asyncio.sleep(rnd.random() * 0.002)  # the read-to-announce race window
            ep.announce(f"compose {pick}", "a chain plan", "A14 holds", "a component is padding", key=pick)
            await asyncio.sleep(rnd.random() * 0.004)  # the work
            a, b = pick.split("_")[1:]
            return Contribute({f"public/plans/{pick}.json": json.dumps({"operator": "chain", "components": [a, b]})}, "plan", pick,
                              informed_by=[seed_sha], parent=seed_sha)

        return run

    return make


async def test_proposer_swarm_yields_distinct_plans_with_few_collisions(tmp_path):
    r, seed = _repo(tmp_path)
    st = SwarmStage(r, "proposer", proposer(seed), C=5, B=30, validate=_validate)
    res = await st.run()
    plans = [r.get(s).summary for s in res.commits]
    assert len(set(plans)) == len(plans) >= 10
    assert res.collision_rate <= 0.10, (res.collisions, res.announcements)
    assert res.charged == 30 == len(res.commits) + len(res.rejected)  # B charged episodes, contributions only
    assert res.abstentions > 0 and res.episodes == 30 + res.abstentions + res.collisions  # abstentions were free
    assert res.dashboard["budget"] == {"used": 30, "left": 0} and "nominations (not correctness evidence)" in res.dashboard


async def test_impolite_workers_collide_but_never_duplicate(tmp_path):
    r, seed = _repo(tmp_path)
    tasks_small = TASKS[:3]

    def make(wid):
        async def run(ep):
            pick = f"chain_{tasks_small[0]}_{tasks_small[1]}"  # every worker wants the same plan and ignores the dashboard
            await asyncio.sleep(0)
            ep.announce(f"compose {pick}", key=pick)
            await asyncio.sleep(0.002)
            return Contribute({f"public/plans/{pick}.json": json.dumps({"operator": "chain", "components": tasks_small[:2]})}, "plan", pick, parent=seed)

        return run

    res = await SwarmStage(r, "proposer", make, C=5, B=3, max_episodes=20).run()
    assert len(res.commits) == 1 and res.collisions == 19 and res.charged == 1  # collisions are cancelled, not charged


async def test_breach_invalid_and_nominations(tmp_path):
    r, seed = _repo(tmp_path)
    outs = iter([Contribute({"oracle/x.json": "[]"}, "oracle", "x"),  # the proposer cannot read oracle/: breach, charged
                 Contribute({"public/plans/bad.json": json.dumps({"operator": "zip", "components": []})}, "plan", "bad"),  # invalid
                 Abstain("pass")])

    def make(wid):
        async def run(ep):
            return next(outs)

        return run

    res = await SwarmStage(r, "proposer", make, C=1, B=2, validate=_validate, require_announce=False).run()  # validation, not protocol
    kinds = sorted(x["kind"] for x in res.rejected)
    assert kinds == ["breach", "invalid"] and res.charged == 2 and res.commits == [] and res.abstentions == 0
    a = r.commit("proposer", {"public/plans/a.json": "{}"}, type_="plan", summary="a", parent=seed)
    b = r.commit("proposer", {"public/plans/b.json": "{}"}, type_="plan", summary="b", parent=seed)
    votes = iter([Nominate(a.sha), Nominate(b.sha), Nominate("f" * 40)])

    def judge(wid):
        async def run(ep):
            return next(votes)

        return run

    res = await SwarmStage(r, "proposer", judge, C=1, B=3).run()
    assert res.nominations == {a.sha: 1, b.sha: 1} and len(res.rejected) == 1
    assert winner(r, res.nominations) == min(a.sha, b.sha) and winner(r, {}) == r.root


async def test_abstain_only_stage_stops_at_max_episodes(tmp_path):
    r, _ = _repo(tmp_path)

    def make(wid):
        async def run(ep):
            return Abstain()

        return run

    res = await SwarmStage(r, "proposer", make, C=5, B=10).run()
    assert res.charged == 0 and res.episodes == 40 and res.abstentions == 40
