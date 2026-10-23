p = '/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src/crucible/factory/generators/debug.py'
s = open(p).read()
a = s.index("    async def candidates_for(self, name: str, *, since: str | None = None) -> list[Candidate]:")
b = s.index("            return out_\n", a) + len("            return out_\n")
new = '''    async def candidates_for(self, name: str, *, since: str | None = None) -> list[Candidate]:
        """The pool = every fix since ``since`` (newest ``max_pool``) on the default branch. B is the head, then (for the
        fixes not yet grouped) the first-parent commit ``B_STEPS`` days earlier: a fix whose revert no longer applies at
        the head often still applies closer to its own date. At each B the pool is executed (``fix_check``) and grouped as
        rebreak groups it (an executed interaction in every group); a fix joins at most one group."""
        lang = self.langs_of[name]
        with scope(lang):
            repo_dir = self.clones[name]
            from crucible.factory.generators.base import git, out

            head = out(git(repo_dir, "rev-parse", "HEAD"))
            head_date = out(git(repo_dir, "log", "-1", "--format=%cI", head))
            since = since or self.since
            fixes = (await asyncio.to_thread(fix_units, repo_dir, head, since=since))[-self.max_pool:]
            for u in fixes:
                self.note(f"{lang}/{name}:{u.sha[:10]}", "S1_source", True, "", subject=u.subject[:120], lang=lang)
            if len(fixes) < 2:
                self.note(f"{lang}/{name}", "S1_source", False, f"{len(fixes)} fix unit(s) since {since}", lang=lang)
                return []
            import datetime as dt

            points = [(head, "HEAD", head_date)]
            for days in B_STEPS:
                cut = (dt.datetime.fromisoformat(head_date) - dt.timedelta(days=days)).isoformat()
                c = out(git(repo_dir, "rev-list", "--first-parent", "-1", f"--before={cut}", head, check=False))
                if c and cut[:10] > since and c not in {x[0] for x in points}:
                    points.append((c, f"HEAD-{days}d", out(git(repo_dir, "log", "-1", "--format=%cI", c))))
            out_: list[Candidate] = []
            used: set[str] = set()
            stats = {"kept": 0, "interacting_fixes": 0}
            for b, b_ref, b_date in points:
                pool = [u for u in fixes if u.sha not in used and git(repo_dir, "merge-base", "--is-ancestor", u.sha, b, check=False).returncode == 0]
                if len(pool) < 2 or len(out_) >= self.max_groups:
                    continue
                got = await self._groups_at(name, lang, b, b_ref, b_date, pool, since, stats)
                for c in got:
                    if len(out_) < self.max_groups:
                        out_.append(c)
                        used |= set(c.params["fixes"])
            self.note(f"{lang}/{name}", "S1_groups", bool(out_), "" if out_ else "no interacting group", lang=lang, fixes=len(fixes),
                      groups=len(out_), points=[x[1] for x in points], **stats)
            return out_

    async def _groups_at(self, name: str, lang: str, b: str, b_ref: str, b_date: str, fixes: list[Unit], since: str,
                         stats: dict[str, int]) -> list[Candidate]:
        repo = RepoGit(self.clones[name])
        work = self.root / "work" / name / b[:10]
        work.mkdir(parents=True, exist_ok=True)
        key = "|".join(u.sha for u in fixes) + ("|full" if self.full_suite else "")
        cache = work / f"evidence-{hashlib.sha256(key.encode()).hexdigest()[:10]}.json"
        if cache.exists():
            evid, card = json.loads(cache.read_text())
        else:
            evid, card = await fix_check(repo, b, fixes, self.sandboxes, self.images_for(lang), name, work, lang=lang, full_suite=self.full_suite)
            cache.write_text(json.dumps([evid, {k: v for k, v in card.items() if k != "build_log"}], default=str))
        if not card.get("ok"):
            self.note(f"{lang}/{name}", "S1_executed", False, "the World image of B does not build", lang=lang, log=str(card.get("build_log", ""))[-600:])
        kept = []
        for u in fixes:
            ev = evid[u.sha]
            rec = SourceRecord("real_fix", name, sha=u.sha, title=u.subject, text=u.body[:1500], created=u.date,
                               evidence={"executed": True, "b": b, "b_ref": b_ref, "lang": lang, **ev})
            self.lease.take(rec)
            self.records.append(rec)
            ok = bool(ev["ok"] and rec.lease["ok"])
            if ok or b_ref == "HEAD":  # an earlier B only adds what the head could not execute
                self.note(f"{lang}/{name}:{u.sha[:10]}", "S1_executed", ok, ev.get("why") or rec.lease.get("reason", ""), lang=lang, b_ref=b_ref)
            if ok:
                kept.append(u)
        funcs = {u.sha: lang_changed_functions(repo, u, profile_of(lang)) for u in kept}
        need = interaction_edges({u.sha: evid[u.sha] for u in kept})
        stats["kept"] += len(kept)
        stats["interacting_fixes"] += len(need)
        recs = {r.sha: r for r in self.records if r.repo == name and r.evidence.get("b") == b}
        out_: list[Candidate] = []
        tags = version_tags(self.clones[name])
        for g in group_fixes(kept, funcs, need=need, max_size=self.max_size):
            cid = f"db-{lang}-{name}-{hashlib.sha256('|'.join(u.sha for u in g).encode()).hexdigest()[:8]}"
            inter = any(need.get(u.sha, set()) & {x.sha for x in g} for u in g)
            if not inter:
                self.note(cid, "S1_executed", False, "no executed interaction at B (no test caught by two reverts)", lang=lang, fixes=[u.sha[:10] for u in g])
                continue
            out_.append(Candidate(cid, NAME, name, sources=[recs[u.sha].to_json() for u in g],
                                  params={"lang": lang, "b": b, "b_ref": b_ref, "b_date": b_date, "b_version": None, "fixes": [u.sha for u in g],
                                          "image_b": card.get("image"), "tags": len(tags), "cutoff": since,
                                          "fix_dates": {u.sha: u.date for u in g},
                                          "contamination": "post-release" if min(u.date[:10] for u in g) >= SINCE else "fallback",
                                          "needs": {u.sha: sorted(need.get(u.sha, set()) & {x.sha for x in g}) for u in g}}))
        return out_
'''
s = s[:a] + new + s[b:]
s = s.replace('FULL_SUITE_MAX = 400', 'B_STEPS = (45, 90)  # days before the head: the earlier B points of ``candidates_for``\nFULL_SUITE_MAX = 400', 1)
open(p, 'w').write(s)
print("ok")
