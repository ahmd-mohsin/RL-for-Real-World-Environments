"""SQLite state canonicalization. STANDALONE: stdlib only; shipped verbatim into containers.

Ops (``main({"op": ..., ...})``):
  dump     {path, include, exclude, rows}  -> {tables: {name: {schema, count, rows_hash, rows?}}, fingerprint}
  backup   {path, out, include, exclude}   -> dump of the copy (consistent online backup)
  restore  {path, src, include, exclude}   -> dump after copying src INTO the live db (same inode,
                                              so open connections in App servers see the new state)
  query    {path, sql, params, max_rows}   -> {columns, rows, truncated}; read-only connection

Canonical form: every schema object (table, index, view, trigger; excluding sqlite_ internals)
contributes its whitespace-normalized SQL; every table contributes the sorted list of its rows, each
row encoded as JSON with typed cells (ints as ints, floats via float.hex, blobs as hex, text as
text). Row order and storage layout never affect the fingerprint; values and schema always do.
"""

import hashlib
import json
import os
import re
import sqlite3


def _cell(v):
    if v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return {"f": v.hex()}
    if isinstance(v, (bytes, memoryview)):
        return {"b": bytes(v).hex()}
    return {"x": repr(v)}


def _norm_sql(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def _match(name, pats):
    import fnmatch

    return any(fnmatch.fnmatchcase(name, p) for p in pats)


def _dump(con, include, exclude, rows):
    objs = con.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_autoindex_%' ORDER BY type, name"
    ).fetchall()
    tables = {}
    schema_lines = []
    for typ, name, tbl, sql in objs:
        if name.startswith("sqlite_") and name != "sqlite_sequence":
            continue
        if include and not _match(tbl, include):
            continue
        if exclude and _match(tbl, exclude):
            continue
        schema_lines.append(f"{typ}\0{name}\0{_norm_sql(sql)}")
        if typ != "table":
            continue
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")').fetchall()]
        enc = sorted(json.dumps([_cell(v) for v in r], separators=(",", ":"), ensure_ascii=False) for r in con.execute(f'SELECT * FROM "{name}"'))
        h = hashlib.sha256()
        for line in enc:
            h.update(line.encode("utf-8"))
            h.update(b"\n")
        t = {"schema": _norm_sql(sql), "columns": cols, "count": len(enc), "rows_hash": h.hexdigest()}
        if rows:
            t["rows"] = enc
        tables[name] = t
    outer = hashlib.sha256()
    for line in sorted(schema_lines):
        outer.update(line.encode("utf-8"))
        outer.update(b"\n")
    for name in sorted(tables):
        outer.update(f"{name}\0{tables[name]['rows_hash']}\n".encode())
    return {"tables": tables, "fingerprint": outer.hexdigest()}


def _ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def dump(a):
    if not os.path.exists(a["path"]):
        return {"tables": {}, "fingerprint": _EMPTY, "missing": True}
    con = _ro(a["path"])
    try:
        return _dump(con, a.get("include") or [], a.get("exclude") or [], bool(a.get("rows")))
    finally:
        con.close()


_EMPTY = hashlib.sha256(b"").hexdigest()


def backup(a):
    if not os.path.exists(a["path"]):
        return {"tables": {}, "fingerprint": _EMPTY, "missing": True}
    src = sqlite3.connect(a["path"])
    dst = sqlite3.connect(a["out"])
    try:
        src.backup(dst)
        return _dump(dst, a.get("include") or [], a.get("exclude") or [], False)
    finally:
        dst.close()
        src.close()


def restore(a):
    if a.get("src") is None:  # the snapshot had no database: make it absent again
        for suffix in ("", "-journal", "-wal", "-shm"):
            if os.path.exists(a["path"] + suffix):
                os.unlink(a["path"] + suffix)
        return {"tables": {}, "fingerprint": _EMPTY, "missing": True}
    src = sqlite3.connect(f"file:{a['src']}?mode=ro", uri=True)
    os.makedirs(os.path.dirname(a["path"]) or ".", exist_ok=True)
    dst = sqlite3.connect(a["path"])
    try:
        src.backup(dst)
        dst.commit()
        return _dump(dst, a.get("include") or [], a.get("exclude") or [], False)
    finally:
        dst.close()
        src.close()


def _guard(con, read_only):
    """A policy's SQL may touch only this database (M11 audit H3 V15: ``VACUUM INTO`` and ``ATTACH`` wrote files outside
    the World, even from the read-only tool, which creates the target before it fails): no ATTACH/DETACH (VACUUM INTO
    attaches), no PRAGMA that sets a value; read-only allows only SELECT, reads, functions and reading pragmas."""
    allowed_ro = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_PRAGMA, getattr(sqlite3, "SQLITE_RECURSIVE", 33)}

    def auth(action, arg1, arg2, db, trigger):
        if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA and arg2 is not None:
            return sqlite3.SQLITE_DENY
        if read_only and action not in allowed_ro:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    con.set_authorizer(auth)


def _pin(con, now):
    """SQLite's ``now`` is the World's time, not the host's (H3 V14): the date/time functions get ``now`` for 'now'."""
    if not now:
        return
    aux = sqlite3.connect(":memory:", check_same_thread=False)

    def wrap(name, at):
        def f(*args):
            v = list(args)
            if len(v) <= at:
                v.append(now)
            elif isinstance(v[at], str) and v[at].strip().lower() == "now":
                v[at] = now
            return aux.execute(f"SELECT {name}({','.join('?' * len(v))})", v).fetchone()[0]

        return f

    for name in ("datetime", "date", "time", "julianday", "unixepoch"):
        con.create_function(name, -1, wrap(name, 0), deterministic=True)
    con.create_function("strftime", -1, wrap("strftime", 1), deterministic=True)


def execute(a):
    con = sqlite3.connect(a["path"])
    try:
        _pin(con, a.get("now"))
        _guard(con, False)
        cur = con.execute(a["sql"], a.get("params") or [])
        con.commit()
        return {"rowcount": cur.rowcount, "lastrowid": cur.lastrowid}
    finally:
        con.close()


def query(a):
    con = _ro(a["path"])
    try:
        con.execute("PRAGMA query_only=ON")
        if a.get("guard"):
            _pin(con, a.get("now"))
            _guard(con, True)
        cur = con.execute(a["sql"], a.get("params") or [])
        cols = [d[0] for d in (cur.description or [])]
        max_rows = int(a.get("max_rows", 1000))
        rows = cur.fetchmany(max_rows + 1)
        return {"columns": cols, "rows": [[_cell(v) for v in r] for r in rows[:max_rows]], "truncated": len(rows) > max_rows}
    finally:
        con.close()


OPS = {"dump": dump, "backup": backup, "restore": restore, "query": query, "execute": execute}


def main(a):
    return OPS[a["op"]](a)
