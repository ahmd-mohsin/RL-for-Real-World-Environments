base = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src/crucible/factory/langs/'
p = base + 'rust.py'
s = open(p).read()
old = '''            ("RUN if cargo test --workspace --all-features --no-run -j 4; then echo --all-features > /opt/crl_features; "
             "else cargo test --workspace --no-run -j 4 && : > /opt/crl_features; fi"),'''
new = '''            # all features when they compile together, else the defaults, else each package that compiles on its own (a
            # workspace member binding Python or wasm cannot block the rest); a run uses what compiled
            ("RUN if cargo test --workspace --all-features --no-run -j 4; then echo --all-features > /opt/crl_features; "
             "elif cargo test --workspace --no-run -j 4; then : > /opt/crl_features; "
             "else : > /opt/crl_features; for p in $(python3 /opt/crl_clean.py --names); do cargo test -p $p --no-run -j 4 || true; done; fi"),'''
assert old in s
s = s.replace(old, new)
old2 = '''meta = json.loads(subprocess.run(["cargo", "metadata", "--no-deps", "--format-version", "1", "--offline"], capture_output=True, text=True, check=True).stdout)'''
new2 = '''meta = json.loads(subprocess.run(["cargo", "metadata", "--no-deps", "--format-version", "1", "--offline"], capture_output=True, text=True, check=True).stdout)
if sys.argv[1:] == ["--names"]:
    print(" ".join(p["name"] for p in meta["packages"]))
    sys.exit(0)'''
assert old2 in s
s = s.replace(old2, new2)
open(p, 'w').write(s)
p = base + 'node.py'
s = open(p).read()
old = '''"pnpm": "corepack enable && pnpm install --frozen-lockfile",'''
new = '''"pnpm": ("(corepack enable && pnpm install --frozen-lockfile) || (corepack disable; npm i -g pnpm@10 && pnpm install --frozen-lockfile) "
                         "|| (npm i -g pnpm@9 && pnpm install --frozen-lockfile)"),  # a packageManager corepack cannot fetch'''
assert old in s
s = s.replace(old, new)
open(p, 'w').write(s)
print("ok")
