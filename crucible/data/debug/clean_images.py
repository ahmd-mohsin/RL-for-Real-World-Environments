"""Remove this lane's own images (crucible-lh-d:*) for the given repositories, except images a stored window uses as its
World, and except images a running container uses. Never touches any other image."""
import json, subprocess, sys
from pathlib import Path

d = Path(__file__).parent
repos = set(sys.argv[1:])
keep = set()
for f in d.glob("gen/store/*/tasks/*/window.json"):
    w = json.loads(f.read_text())
    if w.get("stage") in ("mutants_valid", "dependent", "f2p_valid") and (w.get("world") or {}).get("image"):
        keep.add(w["world"]["image"])
running = set(subprocess.run(["docker", "ps", "--format", "{{.Image}}"], capture_output=True, text=True).stdout.split())
imgs = subprocess.run(["docker", "images", "crucible-lh-d", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True).stdout.split()
for img in imgs:
    if not img.startswith("crucible-lh-d:"):
        continue
    tag = img.split(":", 1)[1]
    lang, rest = tag.split("-", 1)
    slug = rest.rsplit("-", 1)[0]
    if slug in repos and img not in keep and img not in running:
        r = subprocess.run(["docker", "rmi", img], capture_output=True, text=True)
        print("removed" if r.returncode == 0 else "kept(err)", img)
    elif slug in repos:
        print("kept", img)
