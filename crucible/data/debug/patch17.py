p = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/scripts/lh_debug.py'
s = open(p).read()
a = s.index("def cmd_report(args: argparse.Namespace) -> None:")
b = s.index("def main() -> None:")
new = '''def funnel_table() -> tuple[str, dict[str, Any]]:
    """The yield ledger per language (``00`` §9): fix units mined -> real fixes executed -> interacting groups -> windows
    validated (World, leak probes, fail-to-pass) -> dependent with valid mutants -> every CPU gate -> prompt (S7), with
    the top drop reasons per language. The last row of each (candidate, stage) counts."""
    import collections

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for ln in (ROOT / "funnel.jsonl").read_text().splitlines():
        r = json.loads(ln)
        rows[(r["candidate"], r["stage"])] = r
    per: dict[str, dict[str, Any]] = collections.defaultdict(lambda: {"fixes": set(), "real": set(), "groups": set(), "windows": set(), "valid": set(),
                                                                      "gates": set(), "s7": set(), "why": collections.Counter()})
    for (c, st), r in rows.items():
        lang = (r.get("evidence") or {}).get("lang") or (c.split("/", 1)[0] if "/" in c else c.split("-")[1])
        p = per[lang]
        fix = ":" in c and "/" in c
        if st == "S1_source" and fix:
            p["fixes"].add(c)
        elif st == "S1_executed" and fix:
            (p["real"].add(c) if r["ok"] else p["why"].update([r["reason"].split(" (")[0][:60]]))
        elif c.startswith("db-") and st == "S3_world":
            p["groups"].add(c)
            if not r["ok"]:
                p["why"].update([r["reason"].split(" (")[0][:60]])
    for f in ROOT.glob("gen/store/*/tasks/*/window.json"):
        w = json.loads(f.read_text())
        lang = w.get("lang") or w["id"].split("-")[1]
        if w.get("stage") != "windows":
            per[lang]["windows"].add(w["id"])
        else:
            per[lang]["groups"].add(w["id"])
        if w.get("stage") == "mutants_valid":
            per[lang]["valid"].add(w["id"])
        elif w.get("stage") != "windows" and w.get("reason"):
            per[lang]["why"].update([w["reason"].split(":")[0][:60]])
        g = ROOT / "gates" / f"{w['id']}.json"
        if g.exists():
            gate = json.loads(g.read_text())
            if gate.get("ok"):
                per[lang]["gates"].add(w["id"])
            else:
                failed = [k for k, v in (gate.get("gates") or {}).items() if v.get("ok") is False]
                per[lang]["why"].update([f"gate {','.join(failed) or 'error'}"])
        a = ROOT / "authored" / f"{w['id']}.json"
        if a.exists() and not json.loads(a.read_text())["findings"]:
            per[lang]["s7"].add(w["id"])
    lines = ["| language | fix units | real fixes | interacting groups | validated windows | dependent + valid mutants | all CPU gates | prompt (S7) | top drop reasons |",
             "|---|---|---|---|---|---|---|---|---|"]
    out: dict[str, Any] = {}
    for lang in sorted(per):
        p = per[lang]
        n = {k: len(p[k]) for k in ("fixes", "real", "groups", "windows", "valid", "gates", "s7")}
        out[lang] = {**n, "why": dict(p["why"].most_common(5))}
        why = "; ".join(f"{k} ({v})" for k, v in p["why"].most_common(4))
        lines.append(f"| {lang} | {n['fixes']} | {n['real']} | {n['groups']} | {n['windows']} | {n['valid']} | {n['gates']} | {n['s7']} | {why} |")
    return "\\n".join(lines), out


def cmd_report(args: argparse.Namespace) -> None:
    table, data = funnel_table()
    print(table)
    (ROOT / "funnel_table.md").write_text(table + "\\n")
    (ROOT / "funnel_table.json").write_text(json.dumps(data, indent=1))


'''
s = s[:a] + new + s[b:]
if "from typing import Any" not in s:
    s = s.replace("from pathlib import Path\n", "from pathlib import Path\nfrom typing import Any\n", 1)
open(p, 'w').write(s)
print("ok")
