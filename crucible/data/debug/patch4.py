p = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src/crucible/factory/langs/node.py'
s = open(p).read()


def rep(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new)


rep('''def manager_of(tree: Path) -> str:''', '''def runner_config(tree: Path) -> str | None:
    """The configuration file the repository's own test script passes its runner (``jest --config scripts/jest.config.js``),
    or None (the runner finds its default)."""
    runner = runner_of(tree)
    scripts = package_json(tree).get("scripts") or {}
    for key in sorted(scripts, key=lambda k: (k != "test", k)):
        cmd = str(scripts[key])
        if runner and re.search(rf"\\b{runner}\\b", cmd):
            m = re.search(r"(?:--config|-c)[= ]+([^\\s&;|]+)", cmd)
            if m and (Path(tree) / m.group(1)).is_file():
                return m.group(1).removeprefix("./")
    return None


def manager_of(tree: Path) -> str:''')
rep('''        files = " ".join(_q(t) for t in targets)''', '''        files = " ".join(_q(t) for t in targets)
        cfg = runner_config(tree)
        conf = f"--config {_q(cfg)} " if cfg else ""''')
rep('''--coverage=false --watchman=false --passWithNoTests '
                    f'--runTestsByPath {files}''', '''--coverage=false --watchman=false --passWithNoTests {conf}'
                    f'--runTestsByPath {files}''')
rep('''--coverage.enabled=false --passWithNoTests '
                f'{files}''', '''--coverage.enabled=false --passWithNoTests {conf}'
                f'{files}''')
open(p, 'w').write(s)
print("ok")
