import asyncio, sys, tempfile, time, json
from pathlib import Path
from crucible.factory import langs
from crucible.factory.langs.base import use
from crucible.factory.langs.runtime import lang_sandbox_factory
from crucible.factory.lanes.windows_git import RepoGit
from crucible.factory.lanes.windows import suite_targets

ROOT = Path("/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/data/debug/src")


async def main(name, lang):
    d = ROOT / name
    p = langs.get(lang)
    with use(p):
        with tempfile.TemporaryDirectory(prefix="smoke-") as td:
            t = Path(td) / "t"
            ok, bad = RepoGit(d).build("HEAD", [], t)
            print("checkout", ok, bad)
            book = p.images()
            t0 = time.time()
            card = await book.ensure(name, t)
            print("image", card["ok"], card["image"], round(time.time() - t0), "s")
            if not card["ok"]:
                print(card.get("build_log", "")[-3000:])
                return
            sb = lang_sandbox_factory(parallel=1)(card["image"])
            tg = suite_targets(t, {})
            print("targets", len(tg), tg[:5])
            t0 = time.time()
            tr = await sb.run(t, tg)
            c = {}
            for v in tr.results.values():
                c[v] = c.get(v, 0) + 1
            print("run", tr.infra, c, round(time.time() - t0), "s", tr.why[:300])
            print(list(tr.results.items())[:5])
            if not tr.results:
                print(tr.tail[-3000:])
            leak = await p.leak_probes(card["image"], t, new_test_names={"TestNoSuchThingAtAll"})
            print("leak", json.dumps(leak)[:800])


asyncio.run(main(sys.argv[1], sys.argv[2]))
