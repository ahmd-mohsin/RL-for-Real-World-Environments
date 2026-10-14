"""M16 (A): the episode ledger (``actors/rlm/ledger.py``, design §3.1; R1 / H1 / G3 / G4 / B5 / B10). Append-only events with
stable indices, blobs content-addressed, windows with reasons, paged reads, a bounded search (a catastrophic regex returns
within 2 s), strict kernel requests with capped path-free replies, sanitized child names, crash recovery of a torn line,
O(1) lookups at 200k events, and ``HISTORY_API`` driven from cell code in a real kernel."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time

import pytest

from crucible.actors.rlm import ledger as M
from crucible.actors.rlm.kernel import Kernel, LocalLauncher
from crucible.actors.rlm.ledger import HISTORY_API, Ledger, LedgerError

HOST = LocalLauncher(allow_host=True)  # trusted test code only


def _paths_absent(reply, *paths) -> None:
    s = json.dumps(reply, default=str)
    for p in paths:
        assert str(p) not in s


# ---- events, blobs, windows ----------------------------------------------------------------------------------------------
def test_append_get_read_and_blob_dedup(tmp_path):
    L = Ledger(tmp_path)
    assert L.window == -1 and len(L) == 0
    i0 = L.append("system", text="you are an agent")
    assert i0 == 1 and L.window == 0  # the first append opened window 0 ("start") at index 0
    assert L.windows() == [{"w": 0, "reason": "start", "start_index": 0, "meta": {}, "end_index": 1}]
    big = "".join(f"line {k} ok\n" for k in range(20_000))  # > 64 KiB: a blob
    i1 = L.append("tool", step=3, text=big, shown_chars=8192, code="print(x)")
    i2 = L.append("world_call", step=4, tool="crm.get", args={"id": 7}, text=big)  # the same text again: one blob
    st = L.stats()
    assert st["blobs"] == 1 and st["dedup_hits"] == 1 and st["blob_bytes"] == len(big.encode())
    assert len(list((tmp_path / "root" / "blobs").glob("*.txt"))) == 1
    e = L.get(i1)
    assert {k: e[k] for k in ("i", "kind", "window", "step", "agent")} == {"i": i1, "kind": "tool", "window": 0, "step": 3, "agent": ""}
    assert e["text"] == big[:M.GET_CHARS] and e["more"] == {"text": len(big)} and e["shown_chars"] == 8192 and e["code"] == "print(x)"
    assert L.get(-1)["i"] == i2 and L.get(i2)["args"] == {"id": 7}
    assert L.get(i2 - 1)["t"] <= L.get(i2)["t"]  # monotonic clock
    out, cur = "", 0  # paging through the blob reproduces it exactly
    while cur is not None:
        r = L.read(i1, cursor=cur, max_chars=50_000)
        assert r["total"] == len(big)
        out += r["text"]
        cur = r["next"]
    assert out == big
    assert L.read(i0) == {"text": "you are an agent", "next": None, "total": 16}
    assert L.read(i0, cursor=11, max_chars=3) == {"text": "age", "next": 14, "total": 16}
    assert L.read(i2, field="args")["text"] == '{"id":7}'  # a non-text field reads as JSON
    # step defaults to the last one; text kept verbatim incl. non-ASCII; a non-ASCII blob pages by chars
    uni = "é漢🙂" * 30_000
    i3 = L.append("tool", text=uni)
    assert L.get(i3)["step"] == 4
    r = L.read(i3, cursor=89_999, max_chars=4)
    assert r["text"] == uni[89_999:90_003] and r["total"] == len(uni) and r["next"] is None
    # ref(): store once, cite from several events
    ref = L.ref("a world result " * 10)
    a, b = L.append("world_call", result=ref), L.append("tool", text=ref)
    assert L.get(a)["result"] == L.get(b)["text"] == "a world result " * 10
    assert L.stats()["blobs"] == 3
    # a huge non-text field goes to a blob and comes back as its value
    rows = [{"k": k, "v": "x" * 50} for k in range(1_500)]
    i4 = L.append("world_call", rows=rows)
    e4 = L.get(i4)
    assert e4["more"]["rows"] > M.GET_CHARS and e4["rows"] == json.dumps(rows, separators=(",", ":"))[: M.GET_CHARS]
    assert json.loads(L.read(i4, field="rows", max_chars=M.READ_MAX)["text"]) == rows
    L.close()


def test_append_validation(tmp_path):
    L = Ledger(tmp_path)
    for bad in ("", "Tool", "window", "abandon", "x" * 33, "a-b", 3):
        with pytest.raises(ValueError):
            L.append(bad, text="x")
    for kw in ({"i": 1}, {"kind": "x"}, {"window": 2}, {"t": 0}, {"agent": "a"}, {"more": 1}, {"bad-name": 1}):
        with pytest.raises(ValueError):
            L.append("tool", **kw)
    with pytest.raises(ValueError):
        L.append("tool", value=float("nan"))
    with pytest.raises(ValueError):
        L.append("tool", value=object())
    with pytest.raises(ValueError):
        L.append("tool", text={"$blob": "0" * 64})  # a reference to no blob of this ledger
    with pytest.raises(ValueError):
        L.append("tool", step=-1)
    with pytest.raises(ValueError):
        L.append("tool", step=True)
    with pytest.raises(ValueError):
        L.open_window("because")
    assert len(L) == 0  # nothing was half-written
    L.append("tool", text="a\udcff b")  # a lone surrogate (undecodable kernel output) is replaced, not a crash
    assert L.get(-1)["text"].startswith("a")
    with pytest.raises(FileExistsError):  # a second episode never mixes into this one
        Ledger(tmp_path)
    L.close()
    with pytest.raises(LedgerError):
        L.append("tool", text="x")


def test_windows_messages_user_messages_since(tmp_path):
    L = Ledger(tmp_path)
    L.append("system", step=0, text="sys")
    L.append("user", step=0, text="the task: refund order 17")
    for k in range(10):
        L.append("assistant", step=k, text=f"turn {k}")
        L.append("tool", step=k, text=f"out {k} " + "z" * 500)
    L.append("inbox", step=9, text="supervisor: also refund order 18", sender="boss")
    w = L.open_window("compaction", summary_chars=900, step=9)
    assert w == 1 and L.window == 1
    L.append("summary", text="did 0..9")
    L.append("tool", step=11, text="out 11")
    ws = L.windows()
    assert [(x["w"], x["reason"], x["start_index"], x["end_index"]) for x in ws] == [(0, "start", 0, 23), (1, "compaction", 24, 26)]
    assert ws[1]["meta"] == {"summary_chars": 900}
    cur = L.messages(window=-1)
    assert [m["kind"] for m in cur] == ["window", "summary", "tool"]
    tools = L.messages(window=0, kinds=["tool"], preview_chars=10)
    assert len(tools) == 10 and tools[0]["text"] == "out 0 zzzz" and tools[0]["chars"] == {"text": 506}
    last2 = L.messages(window=0, start=-2)
    assert [m["i"] for m in last2] == [22, 23]
    page = L.messages(start=0, limit=5)
    assert len(page) == 5 and page.next == 5
    assert L.messages(start=page.next, limit=100).next is None
    um = L.user_messages()
    assert [m["text"] for m in um] == ["the task: refund order 17", "supervisor: also refund order 18"] and um[1]["sender"] == "boss"
    s = L.since(9)
    assert s[0]["step"] == 9 and all(m["step"] >= 9 for m in s) and s[-1]["text"] == "out 11"
    # a resume may rewind the World step: since() still answers (linear path)
    L.open_window("resume", step=5)
    L.append("tool", step=5, text="after resume")
    assert [m["text"] for m in L.since(11) if "text" in m] == ["out 11"]
    assert L.since(5)[-1]["text"] == "after resume"
    with pytest.raises(ValueError):
        L.messages(window=7)
    L.close()


# ---- search -------------------------------------------------------------------------------------------------------------
def test_search_substring_filters_paging_and_blobs(tmp_path):
    L = Ledger(tmp_path)
    for k in range(50):
        L.append("tool" if k % 2 else "assistant", step=k, text=f"event {k} " + ("refund approved" if k % 10 == 3 else "nothing"))
    hits = L.search("refund")
    assert [h["i"] for h in hits] == [44, 34, 24, 14, 4]  # newest first (index = k + 1: window 0's event is index 0)
    h = hits[0]
    assert h["kind"] == "tool" and h["field"] == "text" and h["step"] == 43 and "refund approved" in h["snippet"]
    assert L.read(h["i"], cursor=h["pos"], max_chars=6)["text"] == "refund"
    assert hits.next is None and not hits.timed_out
    p1 = L.search("refund", limit=2)
    assert [x["i"] for x in p1] == [44, 34] and p1.reason == "limit"
    p2 = L.search("refund", limit=2, before=p1.next)
    assert [x["i"] for x in p2] == [24, 14]
    assert [x["i"] for x in L.search("event 4 ", kinds=["assistant"])] == [5] and L.search("event 4 ", kinds=["tool"]) == []
    assert len(L.search("event 40 ")) == 1
    w = L.open_window("compaction")
    L.append("tool", text="refund in the new window")
    assert [x["i"] for x in L.search("refund", window=w)] == [52] and len(L.search("refund", window=0)) == 5
    assert L.search('"quoted" refund\n') == []  # escapes in the pattern: no prefilter, still correct
    L.append("tool", text='say "quoted" refund\nnow')
    assert len(L.search('"quoted" refund\n')) == 1
    # a match deep inside a blob, also across a CHUNK boundary
    big = "q" * (M.CHUNK - 3) + "NEEDLE" + "q" * 100_000
    ib = L.append("tool", text=big)
    hb = L.search("NEEDLE", context_chars=5)
    assert [x["i"] for x in hb] == [ib] and hb[0]["pos"] == M.CHUNK - 3 and hb[0]["snippet"] == "qqqqqNEEDLEqqqqq"
    r = L.search("NEEDLE", regex=True, context_chars=5)
    assert r[0]["i"] == ib and r[0]["pos"] == M.CHUNK - 3
    rx = L.search(r"event 4[0-9] refund", regex=True)
    assert [x["i"] for x in rx] == [44] and rx.scanned_bytes > 0
    with pytest.raises(ValueError):
        L.search("(", regex=True)
    with pytest.raises(ValueError):
        L.search("")
    with pytest.raises(ValueError):
        L.search("x" * 1001)
    with pytest.raises(ValueError):
        L.search("x", limit=0)
    L.close()


def test_search_catastrophic_regex_is_bounded(tmp_path):
    L = Ledger(tmp_path)
    old = L.append("tool", text="aaaa the old answer is 42")
    evil = L.append("tool", text="a" * 40 + "b")  # (a+)+$ backtracks ~2^40 steps here
    t0 = time.monotonic()
    r = L.search(r"(a+)+$", regex=True)
    dt = time.monotonic() - t0
    assert dt <= M.SEARCH_WALL_S, dt
    assert r.timed_out and r.reason == "timeout" and r.incomplete == evil and r.next == evil  # skipped by next
    t0 = time.monotonic()
    r2 = L.search(r"answer is (\d+)", regex=True, before=r.next)  # the rest of the history is still searchable
    assert time.monotonic() - t0 < M.SEARCH_WALL_S and [x["i"] for x in r2] == [old] and not r2.timed_out
    # the host stays responsive while the worker is hung: the loop keeps running during ahandle
    async def go():
        ticks = 0
        async def tick():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1
        tk = asyncio.ensure_future(tick())
        rep = await L.ahandle({"type": "history.search", "pattern": r"(a+)+$", "regex": True})
        tk.cancel()
        return rep, ticks
    rep, ticks = asyncio.run(go())
    assert rep["timed_out"] and rep["incomplete"] == evil and ticks >= 10
    L.close()


def test_search_scan_cap_64_mib(tmp_path):
    L = Ledger(tmp_path)
    L.append("tool", text="the needle is here")
    for k in range(3):  # 3 x 25 MiB of distinct blobs: more than one call may scan
        L.append("tool", text=chr(ord("a") + k) * (25 * 1024 * 1024))
    t0 = time.monotonic()
    r = L.search("needle")
    assert time.monotonic() - t0 <= M.SEARCH_WALL_S
    assert r == [] and r.reason == "scan_cap" and r.scanned_bytes <= M.SEARCH_BYTES and r.next == 3  # cut in the 3rd blob: next call rescans it
    r2 = L.search("needle", before=r.next)
    assert [x["i"] for x in r2] == [1] and r2.next is None and r2.scanned_bytes <= M.SEARCH_BYTES
    L.close()


# ---- kernel requests ----------------------------------------------------------------------------------------------------
def test_handle_validates_strictly_and_replies_capped_without_paths(tmp_path):
    L = Ledger(tmp_path)
    L.append("user", text="task")
    nul = L.append("tool", text="\x00" * 1_000_000)  # escapes to 6 wire bytes per char
    uni = L.append("tool", text="漢" * 300_000)
    many = [L.append("tool", text="\x01" * 30_000 + "Z" + "\x01" * 30_000) for _ in range(12)]
    bad = [
        "not a dict", {"type": "history.nope"}, {"type": "world.call"}, {"type": 3}, {},
        {"type": "history.get"}, {"type": "history.get", "i": "1"}, {"type": "history.get", "i": True}, {"type": "history.get", "i": 1.0},
        {"type": "history.get", "i": 10_000}, {"type": "history.get", "i": 1, "path": "/etc/passwd"},
        {"type": "history.read", "i": 1, "max_chars": 0}, {"type": "history.read", "i": 1, "max_chars": 10**9},
        {"type": "history.read", "i": 1, "cursor": -1}, {"type": "history.read", "i": 1, "field": "nope"},
        {"type": "history.read", "i": 1, "field": ["text"]}, {"type": "history.read", "i": 1, "field": "kind"},
        {"type": "history.search", "pattern": ""}, {"type": "history.search", "pattern": 5}, {"type": "history.search"},
        {"type": "history.search", "pattern": "x", "limit": 10**6}, {"type": "history.search", "pattern": "x", "regex": "yes"},
        {"type": "history.search", "pattern": "x", "kinds": "tool"}, {"type": "history.search", "pattern": "x", "kinds": ["../x"]},
        {"type": "history.search", "pattern": "x", "context_chars": 10**6}, {"type": "history.search", "pattern": "[", "regex": True},
        {"type": "history.messages", "preview_chars": -1}, {"type": "history.messages", "window": 99},
        {"type": "history.since"}, {"type": "history.since", "step": -1}, {"type": "history.windows", "start": "0"},
        {"type": "history.get", "i": 0, "child": "../x"}, {"type": "history.get", "i": 0, "child": "/abs"},
        {"type": "history.get", "i": 0, "child": "nope"}, {"type": "history.get", "i": 0, "child": 5},
    ]
    for req in bad:
        with pytest.raises(ValueError):
            L.handle(req)
    replies = [L.handle({"type": "history.get", "i": nul}), L.handle({"type": "history.get", "i": uni}),
               L.handle({"type": "history.read", "i": nul, "max_chars": M.READ_MAX}),
               L.handle({"type": "history.read", "i": uni, "max_chars": M.READ_MAX}),
               L.handle({"type": "history.messages", "window": None, "preview_chars": M.MAX_PREVIEW}),
               L.handle({"type": "history.user_messages"}), L.handle({"type": "history.windows"}),
               L.handle({"type": "history.since", "step": 0}), L.handle({"type": "history.stats"}),
               L.handle({"type": "history.search", "pattern": "Z", "limit": 100, "context_chars": M.MAX_CONTEXT})]
    for rep in replies:
        assert len(json.dumps(rep)) <= M.REPLY_CAP  # the kernel's frame encoding (ASCII-escaped)
        assert json.loads(json.dumps(rep, allow_nan=False)) == rep  # plain JSON
        _paths_absent(rep, tmp_path, tmp_path.resolve())
    g = replies[0]
    assert g["more"]["text"] == 1_000_000 and 0 < len(g["text"]) < M.GET_CHARS
    r = replies[2]
    assert r["next"] == len(r["text"]) and r["total"] == 1_000_000  # cut to fit, the cursor continues exactly
    r2 = L.handle({"type": "history.read", "i": nul, "cursor": r["next"]})
    assert r2["text"] and set(r2["text"]) == {"\x00"}
    msgs = replies[4]
    k = len(msgs["items"])
    assert msgs["truncated"] and msgs["next"] == k and [m["i"] for m in msgs["items"]] == list(range(k))
    more = L.handle({"type": "history.messages", "window": None, "preview_chars": M.MAX_PREVIEW, "start": msgs["next"]})
    assert more["items"][0]["i"] == k
    full = [L.handle({"type": "history.get", "i": j}) for j in many]
    assert all(len(json.dumps(x)) <= M.REPLY_CAP for x in full)
    s = replies[-1]
    assert s["truncated"] and s["next"] == s["items"][-1]["i"] and len(s["items"]) < 12  # 12 hits of 24 KiB-escaped snippets
    st = replies[8]
    assert st["events"] == len(L) and "dir" not in st
    # a ledger fault is LedgerError, without a path
    for blob in (tmp_path / "root" / "blobs").glob("*.txt"):
        blob.unlink()
    with pytest.raises(LedgerError) as ei:
        L.handle({"type": "history.get", "i": nul})
    assert str(tmp_path) not in str(ei.value)
    L.close()


def test_child_ledgers_and_names(tmp_path):
    L = Ledger(tmp_path)
    L.append("child_spawn", child="scan", brief="read the logs")
    c = L.child("scan")
    assert c.dir == tmp_path / "root" / "scan" and c.agent == "scan"
    c.append("tool", text="child saw refund 9")
    g = c.child("sub2")
    g.append("tool", text="grandchild saw refund 10")
    assert g.agent == "scan/sub2" and g.get(-1)["agent"] == "scan/sub2"
    assert L.child("scan") is c
    for bad in ("../x", "..", ".", "/etc", "/abs/path", "a/b", "a\x00b", "", "blobs", "events.jsonl", "index.bin", ".hidden", "-x",
                "a" * 65, "a..b", 7, None):
        with pytest.raises(ValueError):
            L.child(bad)
        with pytest.raises(ValueError):
            Ledger(tmp_path, name=bad)
    assert sorted(p.name for p in (tmp_path / "root").iterdir()) == ["blobs", "events.jsonl", "index.bin", "scan"]
    rep = L.handle({"type": "history.search", "pattern": "refund", "child": "scan/sub2"})
    assert [x["snippet"] for x in rep["items"]] == ["grandchild saw refund 10"] and rep["items"][0]["agent"] == "scan/sub2"
    assert L.handle({"type": "history.get", "i": -1, "child": "scan"})["text"] == "child saw refund 9"
    assert L.stats()["children"] == ["scan"]
    for bad in ("scan/../..", "scan//sub2", "scan/nope", "../root"):
        with pytest.raises(ValueError):
            L.handle({"type": "history.get", "i": 0, "child": bad})
    L.close()
    L2 = Ledger.open(tmp_path)  # children reopen with their agent path
    assert L2.child("scan").child("sub2").agent == "scan/sub2" and len(L2.child("scan")) == 2
    L2.close()
    with Ledger.open(tmp_path / "root", name="scan") as c2:
        assert c2.agent == "scan"


# ---- crash safety -------------------------------------------------------------------------------------------------------
def test_open_recovers_a_torn_line_and_continues(tmp_path):
    L = Ledger(tmp_path)
    for k in range(5):
        L.append("tool", step=k, text=f"out {k}")
    L.open_window("compaction")
    L.append("summary", text="s")
    L.flush()
    n = len(L)
    L.close()
    ev = tmp_path / "root" / "events.jsonl"
    with open(ev, "ab") as f:  # the writer was killed mid-event
        f.write(b'{"i":%d,"kind":"tool","window":1,"step":5,"t":9.0,"agent":"","text":"half an ev' % n)
    L2 = Ledger.open(tmp_path)
    assert len(L2) == n and L2.torn_bytes > 0 and L2.window == 1 and L2.scanned_events == 0
    i = L2.append("tool", text="after the crash")
    assert i == n and L2.get(i)["text"] == "after the crash" and L2.get(n - 1)["kind"] == "summary"
    assert L2.get(i)["t"] >= L2.get(n - 1)["t"]  # the clock continues
    for line in ev.read_bytes().splitlines():
        json.loads(line)  # every line whole again
    # the sidecar is behind the log (buffered records lost): the tail is scanned
    L2.close()
    idx = tmp_path / "root" / "index.bin"
    idx.write_bytes(idx.read_bytes()[: 3 * M._REC.size + 5])  # also a torn sidecar record
    L3 = Ledger.open(tmp_path)
    assert len(L3) == n + 1 and L3.scanned_events == n + 1 - 3 and L3.get(-1)["text"] == "after the crash"
    assert [w["reason"] for w in L3.windows()] == ["start", "compaction"]
    L3.close()
    # a sidecar that disagrees with the log is rebuilt; a missing one too
    idx.write_bytes(M._REC.pack(0, 0, 0, b"bogus") * 2)
    L4 = Ledger.open(tmp_path)
    assert len(L4) == n + 1 and L4.scanned_events == n + 1 and [L4.get(j)["i"] for j in range(n + 1)] == list(range(n + 1))
    L4.close()
    idx.unlink()
    L5 = Ledger.open(tmp_path)
    assert len(L5) == n + 1 and L5.search("crash")[0]["i"] == n
    L5.close()
    # a torn blob write (crash before the rename) is cleaned; a complete-but-garbled last line counts as torn
    (tmp_path / "root" / "blobs" / f"{'a' * 64}.123.tmp").write_bytes(b"partial")
    with open(ev, "ab") as f:
        f.write(b'{"i":99, garbage}\n')
    L6 = Ledger.open(tmp_path)
    assert len(L6) == n + 1 and not list((tmp_path / "root" / "blobs").glob("*.tmp"))
    assert L6.append("tool", text="x") == n + 1
    L6.close()
    # corruption in the middle is not silently dropped
    lines = ev.read_bytes().splitlines(keepends=True)
    lines[2] = b'{"i":2,"kind":"tool" BROKEN}\n'
    ev.write_bytes(b"".join(lines))
    idx.unlink()
    with pytest.raises(LedgerError):
        Ledger.open(tmp_path)


async def test_ahandle_search_while_appending(tmp_path):
    L = Ledger(tmp_path)
    for k in range(2_000):
        L.append("tool", step=k, text=f"row {k} " + "y" * 2_000)
    n0 = len(L)
    task = asyncio.ensure_future(L.ahandle({"type": "history.search", "pattern": "row 7 ", "limit": 5}))
    added = 0
    while not task.done():
        L.append("tool", text="appended during the search row 7 ")
        added += 1
        await asyncio.sleep(0)
    rep = await task
    assert all(x["i"] < n0 or "appended" in x["snippet"] for x in rep["items"])
    assert len(L) == n0 + added and L.get(-1)["i"] == n0 + added - 1
    L.close()


# ---- scale --------------------------------------------------------------------------------------------------------------
def test_scale_200k_events(tmp_path):
    N = 200_000
    L = Ledger(tmp_path)
    t0 = time.perf_counter()
    for k in range(N):
        if k % 1_000 == 999:
            L.open_window("compaction")
        L.append("tool" if k % 3 else "assistant", step=k // 2, text=f"cell {k} ticket T-{k:06d} " + "x" * 260, shown_chars=290)
    t_append = time.perf_counter() - t0
    n = len(L)
    rng = random.Random(0)
    picks = [rng.randrange(n) for _ in range(20_000)]
    t0 = time.perf_counter()
    for j in picks:
        L.get(j)
    t_get = (time.perf_counter() - t0) / len(picks)
    assert L.get(n - 1)["text"].startswith("cell 199999")
    t0 = time.perf_counter()
    s_new = L.search("T-199990")
    t_search = time.perf_counter() - t0
    size = L.stats()["bytes"]
    L.close()
    t0 = time.perf_counter()
    L2 = Ledger.open(tmp_path)
    t_open = time.perf_counter() - t0
    assert len(L2) == n and L2.window == 200 and L2.scanned_events == 0
    L2.close()
    os.remove(tmp_path / "root" / "index.bin")
    t0 = time.perf_counter()
    L3 = Ledger.open(tmp_path)
    t_rebuild = time.perf_counter() - t0
    assert len(L3) == n and L3.scanned_events == n and L3.get(123_456)["i"] == 123_456
    t0 = time.perf_counter()
    s_old = L3.search("T-000005")  # beyond one call's 64 MiB: stops at the cap, continues with next
    t_cap = time.perf_counter() - t0
    L3.close()
    print(f"\n[ledger 200k] {n} events, {size / 2**20:.1f} MiB: append {N / t_append:,.0f} ev/s ({t_append:.2f} s); get {t_get * 1e6:.1f} us; "
          f"reopen {t_open * 1e3:.0f} ms (sidecar), {t_rebuild * 1e3:.0f} ms (rebuilt); search newest {t_search * 1e3:.1f} ms; "
          f"search to the 64 MiB cap {t_cap * 1e3:.0f} ms")
    assert len(s_new) == 1 and "T-199990" in s_new[0]["snippet"] and s_new.reason == "scan_cap"  # then on to the cap
    assert s_old.reason == "scan_cap" and s_old.next is not None and t_cap <= M.SEARCH_WALL_S
    assert t_get < 1e-3 and t_open < 1.0 and t_rebuild < 5.0  # loose bounds; the measured numbers are printed


# ---- the kernel half ----------------------------------------------------------------------------------------------------
async def test_history_api_in_a_real_kernel(tmp_path):
    L = Ledger(tmp_path)
    L.append("system", text="system prompt")
    L.append("user", text="TASK: refund order 17 by Friday")
    big = "".join(f"row {k}: status ok\n" for k in range(30_000))  # ~600k chars, a blob
    i_big = L.append("tool", step=2, code="print(rows)", text=big, shown_chars=8192)
    L.open_window("compaction", step=3)
    L.append("summary", step=3, text="refunds pending: 17")
    L.append("inbox", step=4, text="supervisor: also refund 18")
    c = L.child("scan")
    c.append("tool", text="child found refund 99")
    seen: list[str] = []

    async def handler(data):
        seen.append(data["type"])
        if data["type"].startswith("history."):
            return await L.ahandle(data)
        raise ValueError(f"unknown host request {data['type']!r}")

    k = Kernel(HOST, cell_timeout_s=30)
    try:
        r = await k.execute(HISTORY_API, handler)
        assert r.ok, r.error
        r = await k.execute(f"e = await history.get({i_big})\n(e['kind'], e['step'], e['more']['text'], len(e['text']))", handler)
        assert r.ok, r.error
        assert r.result == repr(("tool", 2, len(big), M.GET_CHARS))
        code = (f"parts, cur = [], 0\nwhile cur is not None:\n    c = await OUT({i_big}, cursor=cur, max_chars=100000)\n"
                "    parts.append(c['text'])\n    cur = c['next']\n(len(''.join(parts)), len(parts))")
        r = await k.execute(code, handler)
        assert r.ok, r.error
        assert r.result == repr((len(big), -(-len(big) // 100_000)))
        r = await k.execute(f"print(await OUT({i_big}, max_chars=20))", handler)
        assert r.stdout.startswith("row 0: status ok\nrow") and f"OUT({i_big}, cursor=20)" in r.stdout
        r = await k.execute("h = await history.search('refund')\n[(x['i'], x['kind']) for x in h], h.next, h.timed_out", handler)
        assert r.result == repr(([(6, "inbox"), (5, "summary"), (2, "user")], None, False))
        r = await k.execute("h = await history.search(r'row 2999\\d:', regex=True, limit=3)\n[x['pos'] for x in h], h.reason", handler)
        assert r.ok, r.error
        assert r.result == repr(([big.index("row 29990:")], None))  # one hit per event: the first match in it
        r = await k.execute("[(w['w'], w['reason'], w['start_index'], w['end_index']) for w in await history.windows()]", handler)
        assert r.result == repr([(0, "start", 0, 3), (1, "compaction", 4, 6)])
        r = await k.execute("[m['kind'] for m in await history.messages()], [m['i'] for m in await history.messages(window=None, start=-2)]", handler)
        assert r.result == repr((["window", "summary", "inbox"], [5, 6]))
        r = await k.execute("[m['text'] for m in await history.user_messages()]", handler)
        assert r.result == repr(["TASK: refund order 17 by Friday", "supervisor: also refund 18"])
        r = await k.execute("[m['i'] for m in await history.since(3)]", handler)
        assert r.result == repr([4, 5, 6])
        r = await k.execute("ch = history.child('scan')\n[(x['snippet'], x['agent']) for x in await ch.search('refund')], (await ch.get(-1))['text']", handler)
        assert r.result == repr(([("child found refund 99", "scan")], "child found refund 99"))
        r = await k.execute("await history.get(10**6)", handler)  # a bad request is the cell's exception
        assert r.error and "no event" in r.error
        r = await k.execute("await history.child('..').get(0)", handler)
        assert r.error and "invalid ledger name" in r.error
        r = await k.execute("await host('history.get', i=0, child='/etc')", handler)
        assert r.error and "invalid ledger name" in r.error
        r = await k.execute("await host('history.rm', i=0)", handler)
        assert r.error and "unknown history request" in r.error
        r = await k.execute("str(await history.stats())", handler)
        assert str(tmp_path) not in (r.result or "") and "'events': 7" in r.result
        r = await k.execute("h = await history.search(r'(a+)+$', regex=True)\nh.timed_out", handler)
        assert r.result == "False"  # nothing pathological in this ledger
        assert all(t.startswith("history.") for t in seen)
    finally:
        await k.close()
        L.close()


# ---- deterministic replay (B5: the kernel journal keeps a marker, not the reply) ----------------------------------------
async def test_journal_markers_replay_every_op_exactly(tmp_path, monkeypatch):
    L = Ledger(tmp_path)
    L.append("system", text="system prompt")
    L.append("user", step=0, text="TASK: refund order 17")
    for k in range(300):
        L.append("tool", step=k, code=f"c{k}", text=f"cell {k} refund status " + ("ok" if k % 7 else "pending") + " " + "w" * 300)
        if k == 150:
            L.open_window("compaction", step=k)
            L.append("summary", step=k, text="refunds pending: 17")
    L.append("tool", step=299, text="big " + "r" * 200_000 + " refund at the end")  # a blob
    L.append("inbox", step=299, text="supervisor: also refund 18")
    evil = L.append("tool", step=299, text="a" * 40 + "b")
    c = L.child("scan")
    for k in range(20):
        c.append("tool", text=f"child row {k} refund")

    requests = [
        {"type": "history.get", "i": -1}, {"type": "history.get", "i": 5}, {"type": "history.get", "i": -2},
        {"type": "history.read", "i": evil - 2, "cursor": 100_000, "max_chars": 50_000},
        {"type": "history.read", "i": -1},
        {"type": "history.windows"}, {"type": "history.messages"}, {"type": "history.messages", "window": 0, "start": -3},
        {"type": "history.messages", "window": None, "kinds": ["inbox", "user"], "preview_chars": 50},
        {"type": "history.user_messages"}, {"type": "history.since", "step": 298}, {"type": "history.stats"},
        {"type": "history.search", "pattern": "refund"}, {"type": "history.search", "pattern": "pending", "limit": 3},
        {"type": "history.search", "pattern": "refund", "window": -1, "kinds": ["tool"], "before": 250},
        {"type": "history.search", "pattern": r"cell 1\d\d refund status pending", "regex": True},
        {"type": "history.search", "pattern": "refund", "child": "scan", "limit": 5},
        {"type": "history.get", "i": -1, "child": "scan"}, {"type": "history.messages", "child": "scan", "start": -2},
    ]
    recorded = [await L.ahandle_recorded(r) for r in requests]
    # the bound-dependent stops: a regex that hangs (timeout), a byte cap and a substring time-out
    rec_evil = await L.ahandle_recorded({"type": "history.search", "pattern": r"(a+)+$", "regex": True})
    assert rec_evil[0]["timed_out"] and rec_evil[0]["incomplete"] == evil
    monkeypatch.setattr(M, "SEARCH_BYTES", 40_000)
    rec_cap = [await L.ahandle_recorded({"type": "history.search", "pattern": "refund", "regex": rx}) for rx in (False, True)]
    assert all(r[0]["reason"] == "scan_cap" for r in rec_cap)
    monkeypatch.setattr(M, "SEARCH_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(M, "SEARCH_WALL_S", M._MARGIN_S + 0.0005)
    rec_slow = await L.ahandle_recorded({"type": "history.search", "pattern": "never there"})
    assert rec_slow[0]["timed_out"] and rec_slow[0]["next"] is not None
    monkeypatch.setattr(M, "SEARCH_WALL_S", 2.0)
    recorded += [rec_evil, *rec_cap, rec_slow]

    for (reply, marker), req in zip(recorded, [*requests, None, None, None, None], strict=True):
        assert len(json.dumps(marker)) < 1024, marker
        if req is not None:
            assert reply == L.handle(req)  # the recorded view equals a live call made at the same moment
    # the episode goes on: 1,000 more events, a new window, new user messages and blobs, a World step rewind, child events
    for k in range(1_000):
        L.append("tool", step=300 + k, text=f"later {k} refund " + ("x" * 70_000 if k % 250 == 0 else ""))
    L.append("user", text="a new refund instruction")
    L.open_window("resume", step=10)
    L.append("tool", step=10, text="refund after resume")
    for k in range(30):
        c.append("tool", text=f"child later {k} refund")
    for reply, marker in recorded:
        assert L.replay_reply(json.loads(json.dumps(marker))) == reply, marker  # through JSON, as a journal stores it
    assert L.handle({"type": "history.windows"}) != recorded[5][0]  # the live view did move on
    # a kernel cannot set the view itself; a marker beyond the ledger is a fault
    with pytest.raises(ValueError):
        L.handle({"type": "history.get", "i": 0, "upto": 3})
    with pytest.raises(LedgerError):
        L.replay_reply({"history": {"type": "history.get", "i": 0}, "upto": 10**9})
    with pytest.raises(LedgerError):
        L.replay_reply({"history": {"type": "history.search", "pattern": "x"}, "upto": 5})
    # the same views reopened after a restart
    L.close()
    L2 = Ledger.open(tmp_path)
    for reply, marker in recorded:
        assert L2.replay_reply(marker) == reply
    L2.close()


# ---- a resume abandons what the crashed attempt wrote past the checkpoint -----------------------------------------------
async def test_resume_abandons_the_crashed_attempts_events(tmp_path):
    L = Ledger(tmp_path)
    c = L.child("scan")
    L.append("user", step=0, text="TASK: refund order 17")
    L.append("compaction_attempt", step=0, text="a 18-character kind")  # kinds up to 32 chars
    for k in range(2, 97):  # with the 2 events the crashed attempt adds at 80: 100 events
        L.append("tool", step=k, text=f"row {k} refund ok")
        if k < 7:
            c.append("tool", text=f"child row {k} refund")
        if len(L) == 50:
            at50 = [await L.ahandle_recorded(r) for r in (
                {"type": "history.search", "pattern": "refund", "limit": 100}, {"type": "history.windows"},
                {"type": "history.messages", "window": None, "start": -5}, {"type": "history.get", "i": -1},
                {"type": "history.stats"}, {"type": "history.since", "step": 40}, {"type": "history.user_messages"},
                {"type": "history.search", "pattern": "child", "child": "scan"})]
        if len(L) == 60:
            saved = L.lengths()  # the checkpoint
            assert saved == {"": 60, "scan": 6}  # the child: its window event + 5 rows
        if len(L) == 80:
            L.open_window("compaction", step=k)  # a window the crashed attempt opened
            L.append("user", step=k, text="a supervisor message from the undone future")
            for j in range(5):
                c.append("tool", text=f"child future {j} refund")
            L.child("late").append("tool", text="a child spawned after the checkpoint")
    assert len(L) == 100
    at90 = [await L.ahandle_recorded({"type": "history.search", "pattern": "refund", "limit": 100}),
            await L.ahandle_recorded({"type": "history.windows"})]
    L.close()  # the crash

    L = Ledger.open(tmp_path)  # the resume
    with pytest.raises(ValueError):
        L.abandon(10, 10**6, "resume")
    with pytest.raises(ValueError):
        L.abandon(60, 100, "")
    assert L.rewind_to(saved) == {"": 40, "scan": 5, "late": 2}
    w = L.open_window("resume", step=60)
    i_new = L.append("tool", step=60, text="refund after the resume")
    assert i_new >= 101 and w == 2

    def kernel(req):  # what the kernel sees
        return L.handle(req)

    hits = kernel({"type": "history.search", "pattern": "refund", "limit": 100})["items"]
    assert hits[0]["i"] == i_new and hits[-1]["i"] == 1 and not [h for h in hits if 60 <= h["i"] < 100] and len(hits) == 1 + 57 + 1
    assert not kernel({"type": "history.search", "pattern": r"row 9\d", "regex": True})["items"]
    for i in (60, 75, 99, -4):
        with pytest.raises(ValueError, match="abandoned by a resume"):
            kernel({"type": "history.get", "i": i})
        with pytest.raises(ValueError, match="abandoned by a resume"):
            kernel({"type": "history.read", "i": i})
    ab = kernel({"type": "history.get", "i": 100})
    assert ab["kind"] == "abandon" and (ab["start"], ab["end"], ab["reason"]) == (60, 100, "resume")
    ws = kernel({"type": "history.windows"})["items"]
    assert [(x["w"], x["reason"], x["start_index"], x["end_index"]) for x in ws] == [(0, "start", 0, 59), (2, "resume", 101, i_new)]
    with pytest.raises(ValueError, match="abandoned"):
        kernel({"type": "history.messages", "window": 1})
    assert [m["i"] for m in kernel({"type": "history.messages", "window": -1})["items"]] == [101, i_new]  # window=-1: the resume window
    allm = [m["i"] for m in kernel({"type": "history.messages", "window": None, "start": 0, "end": 500})["items"]]
    assert allm == [*range(60), 100, 101, i_new]
    assert [m["text"] for m in kernel({"type": "history.user_messages"})["items"]] == ["TASK: refund order 17"]
    assert [m["i"] for m in kernel({"type": "history.since", "step": 59})["items"]] == [101, i_new]  # the abandon event carries the step before its range
    assert kernel({"type": "history.stats"}) == {"events": len(L) - 40, "next_index": len(L), "windows": 2, "agent": ""}
    st = L.stats()
    assert st["abandoned"] == 40 and st["abandons"] == [{"i": 100, "start": 60, "end": 100, "reason": "resume"}]
    # child ledgers: the child's future and a child born after the checkpoint are gone from every view
    assert [h["snippet"] for h in kernel({"type": "history.search", "pattern": "refund", "child": "scan"})["items"]] == \
        [f"child row {k} refund" for k in range(6, 1, -1)]
    with pytest.raises(ValueError, match="abandoned"):
        kernel({"type": "history.get", "i": 0, "child": "late"})
    assert kernel({"type": "history.stats", "child": "late"})["events"] == 1  # only its abandon event
    # journal markers recorded before the abandon replay exactly (the checkpointed attempt's and the crashed one's)
    for reply, marker in [*at50, *at90]:
        assert L.replay_reply(marker) == reply
    # persisted: a second restart keeps the abandon, and the markers still replay
    L.close()
    L = Ledger.open(tmp_path)
    assert L.stats()["abandoned"] == 40 and L.child("scan").stats()["abandoned"] == 5
    assert L.handle({"type": "history.messages", "window": None, "start": 0, "end": 500})["items"][-1]["i"] == i_new
    with pytest.raises(ValueError, match="abandoned"):
        L.handle({"type": "history.get", "i": 70})
    for reply, marker in [*at50, *at90]:
        assert L.replay_reply(marker) == reply
    assert L.append("tool", text="still appending") == i_new + 1
    L.close()


def test_structured_fields_shrink_to_capped_json_under_the_reply_cap(tmp_path):
    """H1: a ``world.call`` whose args hold 60k CJK chars (360 KiB on the wire, ASCII-escaped) read back by ``history.get``
    was a ``LedgerError`` (the reply could not be cut); now the dict becomes capped JSON text with ``more``."""
    L = Ledger(tmp_path)
    args = {"body": "漢" * 60_000, "to": "ops"}
    i = L.append("world_call", step=1, tool="mail.send", args=args, result={"ok": True, "rows": [{"k": k} for k in range(9_000)]})
    rep = L.handle({"type": "history.get", "i": i})
    assert len(json.dumps(rep)) <= M.REPLY_CAP and json.loads(json.dumps(rep)) == rep
    full = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    assert isinstance(rep["args"], str) and full.startswith(rep["args"]) and rep["more"]["args"] == len(full)
    assert rep["tool"] == "mail.send" and rep["kind"] == "world_call"
    assert "result" not in rep["more"] or isinstance(rep["result"], str)  # cut only if needed, then as JSON text
    # the rest pages through read(field=), capped the same way
    r = L.handle({"type": "history.read", "i": i, "field": "args", "cursor": len(rep["args"]), "max_chars": M.READ_MAX})
    assert len(json.dumps(r)) <= M.REPLY_CAP and r["total"] == len(full) and r["text"] == full[len(rep["args"]): len(rep["args"]) + len(r["text"])]
    text, cur = "", 0
    while cur is not None:
        r = L.handle({"type": "history.read", "i": i, "field": "args", "cursor": cur, "max_chars": M.READ_MAX})
        text, cur = text + r["text"], r["next"]
    assert json.loads(text) == args
    # every other kernel view of the event stays under the cap and raises nothing
    for req in ({"type": "history.messages", "window": None, "preview_chars": M.MAX_PREVIEW}, {"type": "history.since", "step": 0},
                {"type": "history.search", "pattern": "漢漢", "context_chars": M.MAX_CONTEXT}, {"type": "history.user_messages"}):
        out = L.handle(req)
        assert len(json.dumps(out)) <= M.REPLY_CAP
    L.close()
