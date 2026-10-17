"""Window lane (study 08 §3; 09): the release request, its ledger (each requirement <-> its milestone's tests, both ways),
the prompt lint (no process instruction a check does not enforce, no paths, test names or commit ids) and the truthful
epilogue. The author is a scripted client with a strict signature (no ``**kwargs``)."""

from __future__ import annotations

import asyncio
import json

import pytest
from test_m17_w_miner import built_record

from crucible.factory.lanes.executable import REGRESSION_REQ
from crucible.factory.lanes.windows import EPILOGUE, author_window, compose_window_prompt, lint_prompt, public_names, window_ledger


class StrictAuthor:
    """Answers with ``reply(payload)``; takes only the keywords a minimal client has (a ``sample_key`` would raise)."""

    def __init__(self, reply):
        self.reply, self.payloads = reply, []

    async def complete_json(self, messages, max_tokens=4000, temperature=0.0):
        payload = json.loads(messages[-1]["content"])
        self.payloads.append(payload)
        return self.reply(payload)


def good(payload):
    texts = {"m01": "alpha() returns 1.", "m02": "beta() returns 2.", "m03": "gamma() returns one more than beta() does."}
    return {"opening": "For the next release of our toy library we need three small additions.",
            "requirements": [{"id": c["id"], "text": texts[c["id"]]} for c in payload["changes"]], "closing": "Thanks!"}


@pytest.fixture(scope="module")
def rec_repo(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wprompt")
    rec, repo, base = built_record(tmp)
    return rec, repo, base


def test_a_good_draft_maps_every_requirement_to_one_milestone(rec_repo) -> None:
    rec, repo, base = rec_repo
    client = StrictAuthor(good)
    out = asyncio.run(author_window(client, rec, repo, base, sample_key="retry-2"))
    assert out["findings"] == []
    assert [(r["id"], r["milestone"]) for r in out["requirements"]] == [("R1", "m01"), ("R2", "m02"), ("R3", "m03"), (REGRESSION_REQ, None)]
    p = out["prompt"]
    assert "[R1] alpha() returns 1." in p and p.endswith(EPILOGUE)
    for r in out["requirements"]:  # every requirement is located verbatim in the prompt
        a, b = out["spans"][r["id"]]
        assert p[a:b] == r["text"]
    checks = {c["id"]: c for c in out["checkset"]["checks"]}
    assert checks["c_m03"]["requirement_ids"] == ["R3"] and checks["c_m03"]["spec"]["tests"] == ["tests/test_gamma.py::test_gamma"]
    assert checks["c_regress"]["requirement_ids"] == [REGRESSION_REQ]
    # what the author saw: messages and public names, never test bodies or file paths
    seen = json.dumps(client.payloads[0])
    assert "assert gamma() == 3" not in seen and "pkg/gamma.py" not in seen and "tests/" not in seen
    assert client.payloads[0]["changes"][2]["public_names"] == ["gamma"]


def test_unmapped_either_way_goes_back_to_the_author(rec_repo) -> None:
    rec, repo, base = rec_repo

    def missing_one(payload):
        out = good(payload)
        out["requirements"] = out["requirements"][:2] + [{"id": "m99", "text": "Something extra."}]
        return out

    out = asyncio.run(author_window(StrictAuthor(missing_one), rec, repo, base))
    assert any(f.startswith("unmapped_milestone: change m03") for f in out["findings"])
    assert any(f.startswith("unknown_requirement_id: m99") for f in out["findings"])
    _, problems = window_ledger(rec, out["requirements"], out["spans"], out["prompt"])
    assert any("m03" in p for p in problems)


def test_a_draft_with_a_process_instruction_or_a_path_is_refused(rec_repo) -> None:
    rec, repo, base = rec_repo

    def leaky(payload):
        out = good(payload)
        out["requirements"][0]["text"] = "alpha() in pkg/core.py returns 1; do not use the git history and implement it by hand."
        return out

    out = asyncio.run(author_window(StrictAuthor(leaky), rec, repo, base))
    kinds = {f.split(":", 1)[0] for f in out["findings"]}
    assert {"lint process", "lint file_path"} <= kinds


@pytest.mark.parametrize("text,kind", [
    ("Please don't use any third-party library.", "process"),
    ("Do it manually.", "process"),
    ("Without looking at the commit history, add X.", "process"),
    ("Fix tests/test_core.py too.", "file_path"),
    ("Make test_alpha pass.", "test_name"),
    ("As in 3f92aa5b, restore it.", "commit_id"),
    ("See https://example.com and #123.", "link"),
])
def test_lint_catches(text: str, kind: str) -> None:
    assert any(f.startswith(f"lint {kind}") for f in lint_prompt(text))


def test_lint_passes_plain_requirements_and_enforced_process_lines() -> None:
    assert lint_prompt("parse() accepts a trailing newline and returns the same Version as without it.") == []
    assert lint_prompt("Don't use the network.", enforced=[r"network"]) == []


def test_the_epilogue_states_how_the_episode_works() -> None:
    p = compose_window_prompt("Hi.", [{"id": "R1", "text": "x() returns 1."}])
    for must in ("sandboxed", "no network", "Hidden tests", "existing tests must keep passing", "STATUS.json", "GIVE_UP: <reason>", "R1"):
        assert must in p
    assert lint_prompt(p.split("How this works:")[0]) == []


def test_public_names_come_from_the_diff_never_private_or_tests() -> None:
    patch = "@@ -1,3 +1,9 @@ class Version:\n+def parse_loose(s):\n+def _helper():\n+    def test_inner():\n+class Range:\n"
    assert public_names(patch) == ["Range", "Version", "parse_loose"]


def test_the_realism_reviewer_judges_the_maintainers_text_only() -> None:
    from crucible.factory.lanes.windows import review_realism

    client = StrictAuthor(lambda payload: {"ok": True, "findings": []})
    p = compose_window_prompt("We need a release.", [{"id": "R1", "text": "x() returns 1."}])
    out = asyncio.run(review_realism(client, p))
    assert out == {"ok": True, "findings": []}
    seen = client.payloads[0]["request"]
    assert "[R1] x() returns 1." in seen and "STATUS.json" not in seen and "GIVE_UP" not in seen
