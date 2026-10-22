p = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src/crucible/factory/generators/debug.py'
s = open(p).read()
old = '''                for p, txt in tests_b.items() if mode == "strip" else ():
                    ks = set().union(*[keys.get(f["sha"], {}).get(p, set()) for f in pending]) if pending else set()
                    if ks:
                        files[p] = prof.strip_tests(txt, ks, p)'''
new = '''                for p, txt in tests_b.items() if mode == "strip" else ():
                    ks = set().union(*[keys.get(f["sha"], {}).get(p, set()) for f in pending]) if pending else set()
                    if ks:
                        files[p] = prof.strip_tests(txt, ks | (extra.get(p, set()) if pending else set()), p)'''
assert old in s
s = s.replace(old, new)
old2 = '''                for p, ks in extra.items() if pending else ():  # tests that fail visibly in the agent's tree: hidden until B'
                    cur = repo.show(t, p)'''
new2 = '''                for p, ks in extra.items() if pending else ():  # tests that fail visibly in the agent's tree: hidden until B'
                    if mode == "strip" and p in tests_b:
                        continue  # stripped together with the fixes' own tests below
                    cur = repo.show(t, p)'''
assert old2 in s
s = s.replace(old2, new2)
open(p, 'w').write(s)
print("ok")
