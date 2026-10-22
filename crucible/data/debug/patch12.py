p = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src/crucible/factory/generators/debug.py'
s = open(p).read()


def rep(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new)


rep('''# ---- the generator ------------------------------------------------------------------------------------------------------------''', '''# ---- intent (00 §7: the task fails for the intended reason) -------------------------------------------------------------------
SKILLS: dict[str, dict[str, str]] = {
    "multi_fault_localization": {
        "behaviour": "finds several faults from user-visible symptoms alone, across files and modules, without a failing test to start from",
        "failure_mode": "fixes the first fault found and stops, or edits code near the symptom instead of the faulty module",
        "evidence": "milestone vector at each checkpoint (progress.jsonl): some milestones never pass; files the agent edited vs the faults' "
                    "source files (task lineage, never shown to the agent)"},
    "interacting_faults": {
        "behaviour": "keeps debugging after a correct fix whose symptom persists because a second fault masks it (an executed requires edge)",
        "failure_mode": "reverts or abandons a correct fix because its tests still fail, or declares success after the first fix",
        "evidence": "a prerequisite milestone passes while its dependent never does; the dependent's fault file untouched; a revert of a "
                    "change that had made a milestone pass in the checkpoint series; STATUS.json claims vs the hidden tests"},
    "reproduce_from_report": {
        "behaviour": "turns a symptom report into a reproduction and checks each fix against it (the checks are hidden, none fails visibly)",
        "failure_mode": "edits without ever running a reproduction, or claims a fix it never exercised",
        "evidence": "commands in the trail before the first edit (a test, a script or a REPL run touching the reported API); STATUS.json "
                    "probabilities vs verified milestones (Brier)"},
    "regression_safety": {
        "behaviour": "fixes the faults without breaking behaviour the existing suite covers",
        "failure_mode": "a fix that passes its milestone but fails pass-to-pass tests",
        "evidence": "the regression check's failing tests in the GradeReport; whether the agent ran the existing suite before finishing"},
}


def task_intent(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The task's intent card (``00`` §7): skills (``SKILLS``), language, why it is long-horizon, its milestones, the
    expected failure modes and the trace evidence that attributes a failure to the skill. Derived from the executed
    window (faults, files, edges), never from the prompt; stored with the record and on the TaskCard lineage."""
    lang = str(rec.get("lang") or "python")
    ms = list(rec.get("milestones") or [])
    units = {u["sha"]: u for u in rec.get("units") or []}
    with scope(lang) as prof:
        files = {m["id"]: sorted(p for p in units.get(m["unit"], {}).get("files") or [] if prof.is_source(p)) for m in ms}
    all_files = sorted({f for v in files.values() for f in v})
    modules = sorted({f.rsplit("/", 1)[0] if "/" in f else "." for f in all_files})
    edges = [(m["id"], r) for m in ms for r in m.get("requires") or []]
    skills = ["reproduce_from_report"]
    if len(ms) >= 2:
        skills.insert(0, "multi_fault_localization")
    if edges:
        skills.insert(1, "interacting_faults")
    if set(rec.get("p2p") or []) | set(rec.get("collateral") or []):
        skills.append("regression_safety")
    depth = 0
    req = {m["id"]: list(m.get("requires") or []) for m in ms}

    def d(x: str, seen: frozenset[str] = frozenset()) -> int:
        return 1 + max((d(y, seen | {x}) for y in req.get(x, []) if y not in seen), default=0)

    depth = max((d(m["id"]) for m in ms), default=0)
    level = "hard" if len(ms) >= 3 and len(all_files) >= 2 else "easy" if len(ms) <= 2 else "medium"
    return {"skills": skills, "lang": lang, "level": level, "faults": len(ms), "source_files": all_files, "modules": modules,
            "why_long": (f"{len(edges)} executed requires edge(s) ({', '.join(f'{a} needs {b}' for a, b in edges)}): a dependent fault's checks "
                         f"fail until its prerequisite is fixed; prerequisite chain depth {depth}"),
            "milestones": [{"id": m["id"], "requires": list(m.get("requires") or []), "tests": len(m["tests"]), "files": files[m["id"]]} for m in ms],
            "expected_failures": {k: SKILLS[k]["failure_mode"] for k in skills},
            "trace_evidence": {k: SKILLS[k]["evidence"] for k in skills},
            "attribution_rule": "a failure counts toward a skill gap only when its trace evidence is present; S15 needs >= 70% of failures so attributed"}


def intent_cards() -> list[Any]:
    """The family-level ``cards.IntentCard`` per skill (``candidate``: no model failure observed yet; S13/S14 add the
    evidence and a human approves)."""
    from crucible.factory.cards import IntentCard

    return [IntentCard(id=f"intent-debug-{k.replace('_', '-')}", behaviour=v["behaviour"], failure_mode=v["failure_mode"],
                       difficulty_axes=["faults", "modules", "edge depth", "language"], coverage={"behaviour": k, "surface": "debugging, any language"})
            for k, v in SKILLS.items()]


# ---- the generator ------------------------------------------------------------------------------------------------------------''')
rep('''            rec["rebreak"] = rec[NAME]  # GeneratedTask.oracle_steps reads ``window[generator]``; rebreak tooling reads this''',
    '''            if rec.get("milestones"):
                rec[NAME]["intent"] = task_intent(rec)
            rec["rebreak"] = rec[NAME]  # GeneratedTask.oracle_steps reads ``window[generator]``; rebreak tooling reads this''')
rep('''                                    "lineage": {**c.lineage, "lang": self.lang, "prompt_builder": prov.get("builder")}})''',
    '''                                    "lineage": {**c.lineage, "lang": self.lang, "prompt_builder": prov.get("builder"),
                                                "intent": (self.window.get(NAME) or {}).get("intent") or task_intent(self.window)}})''')
open(p, 'w').write(s)
print("ok")
