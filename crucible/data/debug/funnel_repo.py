"""Per repository: fixes, real fixes, interacting fixes, groups, top drop reasons (last row per candidate and stage)."""
import collections, json
from pathlib import Path

d = Path(__file__).parent
rows = {}
for ln in (d / "funnel.jsonl").read_text().splitlines():
    r = json.loads(ln)
    rows[(r["candidate"], r["stage"])] = r
per = collections.defaultdict(lambda: {"fixes": 0, "real": 0, "why": collections.Counter(), "groups": None, "inter": None})
for (c, st), r in rows.items():
    if "/" not in c:
        continue
    lang_repo = c.split(":")[0]
    p = per[lang_repo]
    if st == "S1_source" and ":" in c:
        p["fixes"] += 1
    elif st == "S1_executed" and ":" in c:
        if r["ok"]:
            p["real"] += 1
        else:
            p["why"][r["reason"][:45]] += 1
    elif st == "S1_groups":
        p["groups"] = (r.get("evidence") or {}).get("groups")
        p["inter"] = (r.get("evidence") or {}).get("interacting_fixes")
for k, v in sorted(per.items()):
    print(f"{k:28} fixes={v['fixes']:3} real={v['real']:3} interacting={v['inter']} groups={v['groups']}  {dict(v['why'].most_common(3))}")
