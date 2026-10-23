p = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/scripts/lh_debug.py'
s = open(p).read()
a = s.index("async def mine(args: argparse.Namespace) -> None:")
b = s.index("async def gates(args: argparse.Namespace) -> None:")
new = '''async def mine_one(gen: object, name: str, since: str | None, sem: asyncio.Semaphore) -> None:
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


async def mine(args: argparse.Namespace) -> None:
    repos = [r for r in args.repos.split(",") if r]
    for split in ("train", "heldout"):
        mine_repos = [r for r in repos if split_of(r) == split]
        if mine_repos:
            gen = generator(mine_repos, split)
            sem = asyncio.Semaphore(args.parallel)
            await asyncio.gather(*[mine_one(gen, n, args.since or None, sem) for n in mine_repos])


'''
s = s[:a] + new + s[b:]
s = s.replace("capture_output=True, text=True).stdout\n                tst", "capture_output=True, text=True, check=False).stdout\n                tst")
s = s.replace("                                     capture_output=True, text=True).stdout", "                                     capture_output=True, text=True, check=False).stdout")
open(p, 'w').write(s)
print("ok")
