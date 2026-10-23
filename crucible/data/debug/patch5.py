base = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src/crucible/factory/'


def rep(p, old, new):
    s = open(base + p).read()
    assert old in s, old[:80]
    open(base + p, 'w').write(s.replace(old, new))


rep('langs/node.py', '''            m = re.search(r"(?:--config|-c)[= ]+([^\\s&;|]+)", cmd)
            if m and (Path(tree) / m.group(1)).is_file():
                return m.group(1).removeprefix("./")''', '''            first = re.split(r"&&|;|\\|\\|", cmd)[0] if key == "test" else cmd
            m = re.search(r"(?:--config|-c)[= ]+([^\\s&;|]+)", first)
            if m and (Path(tree) / m.group(1)).is_file():
                return m.group(1).removeprefix("./")
            if key == "test" and re.search(rf"\\b{runner}\\b", first):
                return None  # the main test script runs the runner with its default configuration''')
rep('generators/debug.py', '''    bad_states = ("failed",) if lang in COMPILED else ("failed", "error")
''', '''    bad_states = ("failed",) if lang in COMPILED else ("failed", "error")
    if union and sum(1 for v in b_res.values() if v == "passed") == 0:  # a World mismatch, not a property of the fixes
        why = f"the suite does not run at B ({(at_b.why if at_b else '')[:160]})"
        return {u.sha: {"ok": False, "why": why} for u in units}, card
''')
print("ok")
