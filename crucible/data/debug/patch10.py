p = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src/crucible/factory/generators/debug.py'
s = open(p).read()
old = '''    keys = fix_test_names(repo, u)
    files = set(u.test_files)
    proposed = {k.split("::", 1)[0] for k in keys}
    out = []
    for t in caught:
        f, q = key_of_test(t)
        if f not in files:
            continue
        if f"{f}::{q}" in keys or f not in proposed:
            out.append(t)
    return out'''
new = '''    keys = fix_test_names(repo, u)
    files = set(u.test_files)
    proposed = {k.split("::", 1)[0] for k in keys}
    # data-driven suites: a case or golden file the fix added or changed (black's ``tests/data/cases/<case>.py``) owns the
    # caught test whose id names it (``test_simple_format[<case>]``)
    data = {Path(p).stem for p in u.files if is_test_path(p) and p not in proposed and len(Path(p).stem) >= 4}
    out = []
    for t in caught:
        f, q = key_of_test(t)
        if f in files and (f"{f}::{q}" in keys or f not in proposed):
            out.append(t)
        elif data and any(re.search(rf"(?<![A-Za-z0-9]){re.escape(d)}(?![A-Za-z0-9])", t.split("::", 1)[-1]) for d in data):
            out.append(t)
    return out'''
assert old in s
s = s.replace(old, new)
s = s.replace('''    """The caught tests the fix owns (module doc): in its test files, proposed by the definition diff, or any test of a
    file for which the diff proposes none."""''', '''    """The caught tests the fix owns (module doc): in its test files, proposed by the definition diff, or any test of a
    file for which the diff proposes none; or a test whose id names a case or golden file the fix added or changed."""''')
open(p, 'w').write(s)
print("ok")
