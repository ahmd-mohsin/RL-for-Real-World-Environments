import re
base = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/'
p = base + 'src/crucible/factory/generators/debug.py'
s = open(p).read()
old = '''            info["hidden"] = {"mode": mode, "prose": sorted(prose_b), "visible_stripped": {p: sorted(ks) for p, ks in extra.items()},'''
new = '''            # strip mode leaves the fixes' other test-file changes in A' (a table row a driver test reads): recorded as a risk
            vis_data = sorted({p for f in fixes for p in f["files"] if prof.is_test_file(p) and p not in tests_b and p not in extra}) if mode == "strip" else []
            info["hidden"] = {"mode": mode, "prose": sorted(prose_b), "visible_stripped": {p: sorted(ks) for p, ks in extra.items()},
                              "visible_test_changes": vis_data,'''
assert old in s
s = s.replace(old, new)
open(p, 'w').write(s)
p = base + 'scripts/lh_debug.py'
s = open(p).read()
old = '''                tst = subprocess.run(["git", "-C", str(repo.dir), "diff", u.parent, u.sha, "--", *[p for p in u.files if p not in u.src_files]],
                                     capture_output=True, text=True, check=False).stdout'''
new = '''                rest = [p for p in u.files if p not in u.src_files]
                tst = subprocess.run(["git", "-C", str(repo.dir), "diff", u.parent, u.sha, "--", *rest], capture_output=True, text=True,
                                     check=False).stdout if rest else "(none)"'''
assert old in s
s = s.replace(old, new)
open(p, 'w').write(s)
print("ok")
