p = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src/crucible/factory/generators/debug.py'
s = open(p).read()


def rep(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new)


rep('''    def synthesize(self, cand: Candidate, *, mode: str = "revert") -> tuple[Window | None, dict[str, Any]]:''',
    '''    def synthesize(self, cand: Candidate, *, mode: str = "revert", extra_hide: Mapping[str, set[str]] | None = None
                   ) -> tuple[Window | None, dict[str, Any]]:''')
rep('''            forward = []
            for k, t in enumerate(src_forward):
                pending = fixes[k:]
                files: dict[str, str | None] = {}''', '''            extra = {p: set(ks) for p, ks in (extra_hide or {}).items() if repo.show(b, p) is not None}
            forward = []
            for k, t in enumerate(src_forward):
                pending = fixes[k:]
                files: dict[str, str | None] = {}
                for p, ks in extra.items() if pending else ():  # tests that fail visibly in the agent's tree: hidden until B'
                    cur = repo.show(t, p)
                    if cur is not None:
                        files[p] = prof.strip_tests(cur, ks, p)''')
rep('''            info["hidden"] = {"mode": mode, "prose": sorted(prose_b),''', '''            info["hidden"] = {"mode": mode, "prose": sorted(prose_b), "visible_stripped": {p: sorted(ks) for p, ks in extra.items()},''')
old = s[s.index('''            why = ""
            for mode in ("revert", "strip"):'''):s.index('''            rec["lang"] = lang\n''')]
new = '''            rec: dict[str, Any] = {}
            extra: dict[str, set[str]] = {}
            for _round in range(2):  # round 2: the milestone tests that failed in the agent's own tree are stripped too
                why = ""
                for mode in ("revert", "strip"):  # a reverted test hunk can take a helper later tests use: then strip only
                    win, info = await asyncio.to_thread(self.synthesize, cand, mode=mode, extra_hide=extra)
                    if win is None:
                        self.note(cand.id, "S3_world", False, "fewer than two fixes revert together", lang=lang, **info)
                        return {"id": cand.id, "stage": "source", "reason": "fewer than two fixes revert together", **info}
                    why = await self.precheck(win, lang)
                    if not why.startswith("the agent's start tree does not build") or info.get("hide_mode") == "strip":
                        break
                if why:
                    self.note(cand.id, "S3_world", False, why, lang=lang)
                    rec = {**{k: v for k, v in win.to_json().items() if k != "repo_dir"}, "split": self.store.split, "stage": "windows", "reason": why,
                           "lane": "window", "lang": lang, NAME: info}
                    self.store.save_window(rec)
                    return rec
                import shutil

                shutil.rmtree(self.store.dir(cand.id), ignore_errors=True)  # a second round validates a new start tree
                rec = await validate_window(win, self.store, self.images_for(lang), self.sandboxes, reruns=3, min_milestones=2, mutants=True,
                                            import_tolerant=lang == "python", leak_prefix=f"{PREFIX}leak-", log=self.log)
                view = self.views.pop(win.id, {})
                vis = sorted(t for m in rec.get("milestones") or [] for t in m["tests"] if view.get(t) in ("failed", "error"))
                info["visible_checks"] = vis[:20]
                if not vis or rec.get("stage") == "windows":
                    break
                with scope(lang) as prof:
                    for t in vis:
                        f, k = prof.key_of_test(t)
                        extra.setdefault(f, set()).add(k)
                if _round == 1:
                    rec["stage"], rec["reason"] = "windows", f"hidden checks visible: {len(vis)} milestone test(s) fail in the agent's own tree"
'''
s = s.replace(old, new)
rep('''            view = self.views.pop(win.id, {})
            vis = sorted(t for m in rec.get("milestones") or [] for t in m["tests"] if view.get(t) in ("failed", "error"))
            rec[NAME]["visible_checks"] = vis[:20]
            if vis and rec.get("stage") != "windows":  # a hidden check the agent can run in its own tree is not hidden
                rec["stage"], rec["reason"] = "windows", f"hidden checks visible: {len(vis)} milestone test(s) fail in the agent's own tree"
            self.store.save_window(rec)''', '''            self.store.save_window(rec)''')
open(p, 'w').write(s)
print("ok")
