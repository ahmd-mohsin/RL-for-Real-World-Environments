p = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src/crucible/factory/generators/debug.py'
s = open(p).read()
old = '''        if inspect.iscoroutinefunction(attr):
            async def run_async(*a: Any, **k: Any) -> Any:
                with scope(lang):
                    return await attr(*a, **k)
            return run_async

        def run(*a: Any, **k: Any) -> Any:
            with scope(lang):
                return attr(*a, **k)
        return run'''
new = '''        if inspect.iscoroutinefunction(attr):
            @functools.wraps(attr)  # the signature stays visible (``longtask.check_longtask`` reads it)
            async def run_async(*a: Any, **k: Any) -> Any:
                with scope(lang):
                    return await attr(*a, **k)
            return run_async

        @functools.wraps(attr)
        def run(*a: Any, **k: Any) -> Any:
            with scope(lang):
                return attr(*a, **k)
        return run'''
assert old in s
s = s.replace(old, new)
s = s.replace("import contextlib\nimport hashlib\n", "import contextlib\nimport functools\nimport hashlib\n", 1)
open(p, 'w').write(s)
print("ok")
