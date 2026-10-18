"""P5 scouts (swarm-pipeline/00 §2): typed records, the eval lease before anything runs, the cached and paced GitHub
client, query expansion from the intent card (no model) and the filters before any model call."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from crucible.factory.cards import IntentCard, ObservationRecord
from crucible.factory.sources.github import API, RAW, GitHub, RateLimited
from crucible.factory.sources.lease import EvalLease, SealedSourceError
from crucible.factory.sources.records import SourceRecord
from crucible.factory.sources.scouts import Scout, card_terms, depends_on, expand

CARD = {"id": "intent-x", "behaviour": "carry out a dependent multi-step change in a real Python library and the code that uses it",
        "failure_mode": "loses track of interacting faults", "difficulty_axes": ["dependency", "cross-repository"],
        "coverage": {"behaviour": "dependent library change", "surface": "python-library-change"}}


# ---- records and the lease ---------------------------------------------------------------------------------------------------------
def test_records_are_typed_and_become_observations() -> None:
    with pytest.raises(ValueError):
        SourceRecord("guess", "a/b")
    r = SourceRecord("real_fix", "a/b", sha="0" * 40, created="2024-01-02T03:04:05+00:00", evidence={"executed": True})
    assert r.id == SourceRecord("real_fix", "a/b", sha="0" * 40).id and r.id.startswith("src-real_fix-")
    obs = ObservationRecord.model_validate(r.observation())
    assert obs.type == "real_pr_tests" and obs.pointer == f"git:a/b@{'0' * 40}"
    assert SourceRecord.from_json(json.loads(json.dumps(r.to_json()))) == r
    assert ObservationRecord.model_validate(SourceRecord("request", "a/b", url="https://x").observation()).type == "reference_derived"
    r.lease = {"ok": True}
    r.filters = {"licence": {"ok": True}, "size": {"ok": False}}
    assert not r.passed
    r.filters["size"]["ok"] = True
    assert r.passed


def test_eval_lease_refuses_sealed_repos_orgs_and_time() -> None:
    lease = EvalLease.default()
    assert not lease.check({"repo": "sqlparse"})["ok"] and not lease.check({"repo": "andialbrecht/sqlparse"})["ok"]
    assert lease.check({"repo": "alecthomas/kong"})["rule"] == "sealed_org"  # owns a held-out repository and no training one
    assert lease.check({"repo": "pallets/click"})["ok"] and not lease.check({"repo": "pallets/jinja"})["ok"]  # pallets owns both: per repo
    assert not lease.check({"repo": "lepture__mistune"})["ok"]  # a scouted clone name
    assert not lease.check({"repo": "me/consumer", "evidence": {"host": "tomlkit"}})["ok"]  # a pair with a sealed host
    assert lease.check({"repo": "me/consumer", "evidence": {"host": "click"}})["ok"]
    with pytest.raises(SealedSourceError):
        lease.guard_repo("voluptuous")
    timed = EvalLease(set(), set(), after="2025-06-01")
    assert timed.check({"repo": "a/b", "created": "2025-07-01T00:00:00Z"})["rule"] == "sealed_time"
    assert timed.check({"repo": "a/b", "created": "2025-05-01T00:00:00Z"})["ok"]
    rec = SourceRecord("consumer", "jinja")
    assert not lease.take(rec) and rec.lease["rule"] == "sealed_repo"


def test_lease_reads_a_sealed_manifest(tmp_path: Path) -> None:
    (tmp_path / "sealed.json").write_text(json.dumps({"repos": ["Some/Thing"], "orgs": ["sealedorg"], "after": "2026-01-01"}))
    lease = EvalLease.default(tmp_path / "sealed.json")
    assert not lease.check({"repo": "some/thing"})["ok"] and not lease.check({"repo": "sealedorg/x"})["ok"] and lease.after == "2026-01-01"


# ---- the GitHub client -------------------------------------------------------------------------------------------------------------
class FakeResp:
    def __init__(self, status: int, body: Any, headers: dict[str, str] | None = None):
        self.status, self.headers = status, headers or {}
        self._b = (body if isinstance(body, str) else json.dumps(body)).encode()

    def read(self) -> bytes:
        return self._b


class FakeNet:
    def __init__(self, script: list[Any]):
        self.script, self.urls = list(script), []

    def __call__(self, req: Any, timeout: float) -> FakeResp:
        self.urls.append(req.full_url)
        x = self.script.pop(0)
        if isinstance(x, urllib.error.HTTPError):
            raise x
        return x


def _http(code: int, headers: dict[str, str]) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.github.com/x", code, "limit", headers, io.BytesIO(b"{}"))  # type: ignore[arg-type]


def test_github_client_caches_paces_and_backs_off(tmp_path: Path) -> None:
    sleeps: list[float] = []
    net = FakeNet([_http(403, {"Retry-After": "7"}), FakeResp(200, {"items": [{"full_name": "a/b"}]}, {"X-RateLimit-Remaining": "9"}),
                   FakeResp(200, {"items": []})])
    gh = GitHub(tmp_path / "c", opener=net, sleep=sleeps.append, search_interval=6.5)
    assert gh.search_repos("q") == [{"full_name": "a/b"}]
    assert 7.0 in sleeps and gh.remaining["search"] == 9  # Retry-After honoured, the quota read
    assert gh.search_repos("q") == [{"full_name": "a/b"}] and gh.calls["cached"] == 1 and len(net.urls) == 2  # cached: no request
    gh.search_repos("other")
    assert any(abs(s - 6.5) < 0.5 for s in sleeps)  # two searches in a row are paced
    off = GitHub(tmp_path / "c", offline=True)
    assert off.search_repos("q") == [{"full_name": "a/b"}]
    with pytest.raises(RateLimited):
        off.search_repos("never seen")
    with pytest.raises(ValueError):
        gh.fetch("https://example.com/x")


def test_github_client_raises_when_the_quota_resets_too_late(tmp_path: Path) -> None:
    import time

    net = FakeNet([FakeResp(200, [], {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(time.time() + 3600)})])
    gh = GitHub(tmp_path / "c", opener=net, sleep=lambda s: None, max_wait=60)
    assert gh.forks("a/b") == []
    with pytest.raises(RateLimited):
        gh.forks("c/d")  # 0 left for an hour: never a silent hour-long wait


def test_github_raw_files_and_misses_are_cached(tmp_path: Path) -> None:
    net = FakeNet([FakeResp(200, "dependencies = ['click>=8']\n"), urllib.error.HTTPError(f"{RAW}/a/b/main/setup.py", 404, "nf", {}, io.BytesIO(b""))])  # type: ignore[arg-type]
    gh = GitHub(tmp_path / "c", opener=net, sleep=lambda s: None)
    assert gh.raw("a/b", "main", "pyproject.toml") == "dependencies = ['click>=8']\n"
    assert gh.raw("a/b", "main", "setup.py") is None and gh.raw("a/b", "main", "setup.py") is None and len(net.urls) == 2
    assert gh.kind_of(f"{API}/search/commits") == "search" and gh.kind_of(f"{API}/repos/a/b") == "core" and gh.kind_of(f"{RAW}/a/b/x") == "raw"


# ---- query expansion and filters --------------------------------------------------------------------------------------------------
def test_queries_come_from_the_card_through_templates_not_a_model() -> None:
    card = IntentCard.model_validate({**CARD, "status": "candidate"})
    terms = card_terms(card)
    assert terms == card_terms(CARD) and "dependent" in terms and "the" not in terms and "carry" not in terms
    assert any(" " in t for t in terms)  # bigrams of the behaviour
    q = expand(card, "host_repo", limit=4)
    assert q[0].endswith("library language:python") and q[1].endswith("in:name,description language:python") and len(q) == 4
    assert expand(card, "consumer", hosts=["click", "attrs"], limit=3) == ["click in:name,description language:python", "click plugin language:python",
                                                                            "click extension language:python"]
    assert expand(card, "request")[0].endswith("is:issue is:closed language:python")


def test_depends_on_reads_requirement_lines() -> None:
    assert depends_on('dependencies = [\n  "click>=8.0",\n]', "click")
    assert depends_on('install_requires=["Click >= 7.0, <9"]', "click")
    assert depends_on("python-dateutil>=2.7\n", "python-dateutil") and depends_on("python_dateutil\n", "python-dateutil")
    assert depends_on("click-default-group>=1\n", "click") is None and depends_on("# click is great\n", "click") is None


class FakeGH:
    def __init__(self, repos: list[dict[str, Any]], raw: dict[tuple[str, str], str]):
        self._repos, self._raw, self.searches = repos, raw, []

    def search_repos(self, q: str, **kw: Any) -> list[dict[str, Any]]:
        self.searches.append(q)
        return self._repos

    def raw(self, name: str, ref: str, path: str) -> str | None:
        return self._raw.get((name, path))

    def forks(self, name: str, **kw: Any) -> list[dict[str, Any]]:
        return [{"full_name": "f/" + name.split("/")[1], "license": {"spdx_id": "MIT"}, "size": 10, "stargazers_count": 50, "fork": True}]

    def search_issues(self, q: str, **kw: Any) -> list[dict[str, Any]]:
        return [{"repository_url": "https://api.github.com/repos/o/r", "html_url": "https://github.com/o/r/issues/1", "title": "t", "state": "closed"}]

    def search_commits(self, q: str, **kw: Any) -> list[dict[str, Any]]:
        return [{"repository": {"full_name": "andialbrecht/sqlparse"}, "sha": "a" * 40, "commit": {"message": "faster", "committer": {"date": "2024-01-01"}}}]


def _repo(name: str, lic: str | None = "MIT", size: int = 100, stars: int = 99, **kw: Any) -> dict[str, Any]:
    return {"full_name": name, "license": {"spdx_id": lic} if lic else None, "size": size, "stargazers_count": stars, "default_branch": "main", **kw}


def test_scout_filters_before_anything_costly_and_snowballs_to_consumers() -> None:
    gh = FakeGH([_repo("ok/cons"), _repo("gpl/cons", lic="GPL-3.0"), _repo("big/cons", size=10**6), _repo("lepture/x"), _repo("fk/cons", fork=True),
                 _repo("ok/nodep"), _repo("pallets/click")], {("ok/cons", "pyproject.toml"): 'dependencies = ["marshmallow>=3.13"]',
                                                               ("ok/nodep", "setup.py"): "install_requires=['attrs']"})
    sc = Scout(gh, EvalLease.default(), exclude=["pallets/click"])  # type: ignore[arg-type]
    recs = {r.repo: r for r in sc.consumers_of("marshmallow", "marshmallow", ["q1"])}
    assert set(recs) == {"ok/cons", "gpl/cons", "big/cons", "lepture/x", "fk/cons", "ok/nodep"}  # the excluded known repository is not "new"
    assert recs["lepture/x"].lease["rule"] == "sealed_org" and not recs["lepture/x"].filters  # the lease first: no filter ran on it
    assert not recs["gpl/cons"].filters["licence"]["ok"] and not recs["big/cons"].filters["size"]["ok"] and not recs["fk/cons"].filters["size"]["ok"]
    assert recs["ok/cons"].passed and recs["ok/cons"].filters["depends_on_host"]["ok"] and recs["ok/cons"].evidence["host"] == "marshmallow"
    assert not recs["ok/nodep"].filters["depends_on_host"]["ok"]
    assert "depends_on_host" not in recs["gpl/cons"].filters  # no raw read for a record that failed a cheaper filter
    forks = sc.forks(recs["ok/cons"])
    assert forks[0].evidence["via"] == "fork_of:ok/cons" and forks[0].type == "consumer"
    assert sc.issues(["x"])[0].type == "request" and sc.issues(["x"])[0].repo == "o/r"
    c = sc.commits(["faster x"], "real_perf")[0]
    assert c.type == "real_perf" and not c.lease["ok"]  # a sealed repository's commit is refused at the lease
