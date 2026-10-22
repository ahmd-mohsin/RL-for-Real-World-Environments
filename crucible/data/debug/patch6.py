p = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src/crucible/factory/generators/debug.py'
s = open(p).read()


def rep(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new)


rep('''    def synthesize(self, cand: Candidate) -> tuple[Window | None, dict[str, Any]]:''',
    '''    def synthesize(self, cand: Candidate, *, mode: str = "revert") -> tuple[Window | None, dict[str, Any]]:''')
rep('''            mode = "revert"
            while len(fixes) >= 2:''', '''            while len(fixes) >= 2:''')
rep('''            win, info = await asyncio.to_thread(self.synthesize, cand)
            if win is None:
                self.note(cand.id, "S3_world", False, "fewer than two fixes revert together", lang=lang, **info)
                return {"id": cand.id, "stage": "source", "reason": "fewer than two fixes revert together", **info}
            why = await self.precheck(win, lang)''', '''            why = ""
            for mode in ("revert", "strip"):  # a reverted test hunk can take a helper later tests use: then strip only
                win, info = await asyncio.to_thread(self.synthesize, cand, mode=mode)
                if win is None:
                    self.note(cand.id, "S3_world", False, "fewer than two fixes revert together", lang=lang, **info)
                    return {"id": cand.id, "stage": "source", "reason": "fewer than two fixes revert together", **info}
                why = await self.precheck(win, lang)
                if not why.startswith("the agent's start tree does not build") or info.get("hide_mode") == "strip":
                    break''')
open(p, 'w').write(s)
print("ok")
