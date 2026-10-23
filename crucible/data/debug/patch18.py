p = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src/crucible/factory/langs/node.py'
s = open(p).read()
old = '''            "RUN mkdir -p /opt/deps && mv /w/node_modules /opt/deps/node_modules && rm -rf /w && mkdir -p /w && chmod -R a+rwX /opt/deps",'''
new = '''            "RUN mkdir -p /opt/deps && mv /w/node_modules /opt/deps/node_modules && rm -rf /w && mkdir -p /w && chmod -R a+rwX /opt/deps",
            # a dependency that depends on the project itself ships a release of it (the Python lane's installed-copy leak):
            # every such copy is deleted, the dependents keep working only if the tests never load them
            *([f"RUN find /opt/deps/node_modules -path '*/node_modules/{info['project']}' -prune -exec rm -rf {{}} + ; "
               f"find /opt/deps/node_modules -name package.json -path '*/{info['project'].split('/')[-1]}/package.json' -prune 2>/dev/null | head -0"]
              if info.get("project") and re.fullmatch(r"(@[\\w.-]+/)?[\\w.-]+", str(info["project"])) else []),'''
assert old in s
s = s.replace(old, new)
s = s.replace('recipe = "node-2"', 'recipe = "node-3"')
s = s.replace("- **Leaks.** ``npm view <name>`` fails; no package in the dependency tree is the project itself (a tool that depends\n  on the project would ship a release of it);",
              "- **Leaks.** ``npm view <name>`` fails; no package in the dependency tree is the project itself (a tool that depends\n  on the project would ship a release of it: the World build deletes every such copy, the probe checks it);")
open(p, 'w').write(s)
print("ok")
