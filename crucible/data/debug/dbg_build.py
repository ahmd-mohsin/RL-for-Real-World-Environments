"""Debug: check out A' of a debug candidate and run a command in its World image."""
import subprocess, sys, tempfile
from pathlib import Path
from crucible.factory.lanes.windows_git import RepoGit

cid, repo, image, cmd = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
clone = Path("data/debug/gen/clones") / repo
shas = subprocess.run(["git", "-C", str(clone), "rev-list", "--reverse", f"refs/crucible/debug/{cid}"], capture_output=True, text=True).stdout.split()
b = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
chain = shas[shas.index(next(s for s in shas if subprocess.run(["git", "-C", str(clone), "log", "-1", "--format=%s", s], capture_output=True, text=True).stdout.startswith("debug start"))):]
print("A'", chain[0][:10], "units", len(chain) - 1)
with tempfile.TemporaryDirectory(prefix="dbg-") as td:
    t = Path(td) / "t"
    RepoGit(clone).build(chain[0], [], t)
    print(subprocess.run(["docker", "run", "--rm", "--name", "crucible-lh-d-dbg2", "--network", "none", "-v", f"{t}:/src:ro", image, "sh", "-c",
                          f"cp -a /src/. /w/ && cd /w && {cmd}"], capture_output=True, text=True).stdout[-4000:])
