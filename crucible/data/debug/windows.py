"""Every stored debug window: id, stage, reason, milestones, edges, gate verdict, prompt lint."""
import json
from pathlib import Path

d = Path(__file__).parent
for f in sorted(d.glob("gen/store/*/tasks/*/window.json")):
    r = json.loads(f.read_text())
    g = d / "gates" / f"{r['id']}.json"
    gate = json.loads(g.read_text()) if g.exists() else {}
    a = d / "authored" / f"{r['id']}.json"
    lint = len(json.loads(a.read_text())["findings"]) if a.exists() else None
    print(f"{f.parts[-4]:7} {r['id']:34} {r['stage']:14} ms={len(r.get('milestones') or [])} edges={r.get('dependency_edges')} "
          f"gates={gate.get('ok')}{'' if gate.get('ok') else ' ' + str(gate.get('passed'))} lint={lint} {r.get('reason', '')[:70]}")
