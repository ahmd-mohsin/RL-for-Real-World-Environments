p = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src/crucible/factory/generators/debug.py'
s = open(p).read()


def rep(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new)


old = s[s.index("            def reverted(f: Mapping[str, Any]) -> list[str]:  # source + non-code test data"):s.index("            tb = tree_of(repo_dir, b)\n")]
new = '''            def reverted(f: Mapping[str, Any], tests: bool) -> list[str]:  # source + non-code test data (+ test code)
                return [p for p in f["files"] if prof.is_source(p) or (is_test_path(p) and not PROSE.search(p) and (tests or not prof.is_test_file(p)))]

            # mode "revert": the fixes' test hunks reverted too (a table row, a changed expectation, a new test: all gone from
            # A'); mode "strip" when that conflicts: only the changed test definitions removed (``profile.strip_tests``)
            mode = "revert"
            while len(fixes) >= 2:
                steps = [(full_patch(repo_dir, f["parent"], f["sha"], reverted(f, mode == "revert")), True) for f in reversed(fixes)]
                trees, bad = apply_chain(repo_dir, b, steps)
                if bad is None:
                    break
                if mode == "revert":
                    mode = "strip"
                    continue
                drop = list(reversed(fixes))[bad]
                info["dropped"].append({"sha": drop["sha"], "why": "its revert conflicts with the fixes after it"})
                fixes = [f for f in fixes if f["sha"] != drop["sha"]]
            if len(fixes) < 2:
                return None, info
            info["hide_mode"] = mode
'''
s = s.replace(old, new)
rep('''                for p, txt in tests_b.items():
                    ks = set().union(*[keys.get(f["sha"], {}).get(p, set()) for f in pending]) if pending else set()''',
    '''                for p, txt in tests_b.items() if mode == "strip" else ():
                    ks = set().union(*[keys.get(f["sha"], {}).get(p, set()) for f in pending]) if pending else set()''')
rep('''            info["hidden"] = {"stripped": {p: sorted(set().union(*[keys.get(f["sha"], {}).get(p, set()) for f in fixes])) for p in tests_b},
                              "prose": sorted(prose_b)}''', '''            info["hidden"] = {"mode": mode, "prose": sorted(prose_b),
                              **({"stripped": {p: sorted(set().union(*[keys.get(f["sha"], {}).get(p, set()) for f in fixes])) for p in tests_b}}
                                 if mode == "strip" else {"reverted": sorted({p for f in fixes for p in f["files"] if is_test_path(p)})})}''')
rep('''        """Why the window cannot be graded fairly ("" when it can): B's test files must build against A', and the agent's
        own view of A' must build (a compiled language's ``build failed`` notes)."""
        if lang not in COMPILED:
            return ""
        repo''', '''        """Why the window cannot be graded fairly ("" when it can): the agent's own view of A' must build, and B's test files
        must build against A' (a compiled language's ``build failed`` notes). The view's results are kept
        (``self.views``): after validation no milestone test may fail in the agent's own tree (``visible_checks``)."""
        repo''')
rep('''            view = await sb.run(a, suite_targets(a, {}))
            if "build failed" in view.why:''', '''            view = await sb.run(a, suite_targets(a, {}))
            self.views[win.id] = dict(view.results) if view.infra == "ok" else {}
            if "build failed" in view.why:''')
rep('''            hid = await sb.run(a, files) if files else None''', '''            hid = await sb.run(a, files) if files and lang in COMPILED else None''')
rep('''            rec["rebreak"] = rec[NAME]  # GeneratedTask.oracle_steps reads ``window[generator]``; rebreak tooling reads this
            self.store.save_window(rec)''', '''            rec["rebreak"] = rec[NAME]  # GeneratedTask.oracle_steps reads ``window[generator]``; rebreak tooling reads this
            view = self.views.pop(win.id, {})
            vis = sorted(t for m in rec.get("milestones") or [] for t in m["tests"] if view.get(t) in ("failed", "error"))
            rec[NAME]["visible_checks"] = vis[:20]
            if vis and rec.get("stage") != "windows":  # a hidden check the agent can run in its own tree is not hidden
                rec["stage"], rec["reason"] = "windows", f"hidden checks visible: {len(vis)} milestone test(s) fail in the agent's own tree"
            self.store.save_window(rec)''')
rep('''        self.since, self.max_pool = since, max_pool''', '''        self.since, self.max_pool = since, max_pool
        self.views: dict[str, dict[str, str]] = {}''')
open(p, 'w').write(s)
print("ok")
