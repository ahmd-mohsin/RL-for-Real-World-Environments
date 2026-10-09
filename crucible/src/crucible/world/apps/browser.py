"""Browser App: a real headless browser driven through Microsoft's Playwright MCP server.

What a human clerk uses to work web systems (vendor portals, SaaS admin pages). Tools are Playwright
MCP's own (``browser_navigate``, ``browser_snapshot``, ``browser_click``, ``browser_type``,
``browser_fill_form``, ``browser_select_option``, ``browser_press_key``, ``browser_navigate_back``,
``browser_wait_for`` ...), narrowed by ``allow_tools``. ``browser_evaluate`` (arbitrary JS) is withheld from
the policy and used only by the state adapter.

Navigation is restricted to the World's own sites with ``--allowed-origins`` and ``--blocked-origins``.
Playwright documents that this is NOT a security boundary, so a world that must not reach the internet
also needs backend network enforcement (the Repo2RLEnv lesson: the environment enforces, the prompt never asks).

State (``browser`` adapter): current URL, ``document.cookie`` and ``localStorage``. Durable business state
belongs to the website's database App; the browser carries only navigation and session.
Params: ``start_url`` (may use ``{app:<id>}``), ``allowed_origins`` (default: every website App in the
World), ``allow_tools``, ``package`` (pinned ``@playwright/mcp`` version), ``browser``.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from crucible.core.ids import sha256_text
from crucible.core.schema.state import StateDiff, StateRef
from crucible.world.apps.mcp_server import McpServerApp
from crucible.world.state.base import SnapshotMismatch, StateAdapter

PINNED = "@playwright/mcp@0.0.83"
POLICY_TOOLS = ["browser_navigate", "browser_navigate_back", "browser_snapshot", "browser_click", "browser_type", "browser_fill_form",
                "browser_select_option", "browser_press_key", "browser_hover", "browser_wait_for", "browser_tabs", "browser_handle_dialog"]
# Error pages (blocked origins) forbid cookie/storage access: read defensively.
_READ = ("() => { const g = (f, d) => { try { return f(); } catch (e) { return d; } }; "
         "return {url: location.href, cookie: g(() => document.cookie, ''), "
         "storage: g(() => JSON.stringify(Object.fromEntries(Object.entries(localStorage))), '{}')}; }")


class BrowserApp(McpServerApp):
    template = "common/browser"

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self.state = BrowserStateAdapter(spec.id, ctx.exec, dict(spec.state.params), ctx.store, app=self)

    def _origins(self) -> list[str]:
        if self.spec.params.get("allowed_origins"):
            return [self._resolve_refs(o) for o in self.spec.params["allowed_origins"]]
        peers = self.ctx.extra.get("peers", {})
        return [p.url for p in peers.values() if getattr(p, "url", None)]

    def server_command(self) -> list[str]:
        p = self.spec.params
        out_dir = self.ctx.host_workdir / f"{self.id}-playwright"  # page snapshots and traces stay with the world, not in the cwd
        cmd = ["npx", "-y", p.get("package", PINNED), "--headless", "--isolated", "--output-dir", str(out_dir)]
        if p.get("browser"):
            cmd += ["--browser", p["browser"]]
        origins = self._origins()
        if origins:
            cmd += ["--allowed-origins", ";".join(origins)]
        return cmd

    async def start(self) -> None:
        self.spec.params.setdefault("allow_tools", POLICY_TOOLS)
        await super().start()
        start = self.spec.params.get("start_url")
        if start:
            r = await self.call_unlisted("browser_navigate", {"url": self._resolve_refs(start)})
            if not r.ok:
                raise RuntimeError(f"{self.id}: cannot open {start}: {r.content[:300]}")

    async def evaluate(self, fn: str) -> Any:
        import asyncio

        for attempt in range(5):  # a navigation in flight destroys the JS context: wait and retry
            r = await self.call_unlisted("browser_evaluate", {"function": fn})
            if r.ok or "Execution context was destroyed" not in r.content:
                break
            await asyncio.sleep(0.3 * (attempt + 1))
        if not r.ok:
            raise RuntimeError(f"browser_evaluate failed: {r.content[:300]}")
        text = r.content
        # Playwright MCP replies in sections ("### Result", "### Ran Playwright code", ...): parse the result only
        if "### Result" in text:
            text = text.split("### Result", 1)[1].split("\n### ", 1)[0]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return text.strip()


class BrowserStateAdapter(StateAdapter):
    kind = "browser"
    OPS: ClassVar[dict[str, str]] = {"page": "current {url, cookie, storage}"}

    def __init__(self, app_id, exec_, params, store, *, app: BrowserApp | None = None):
        super().__init__(app_id, exec_, params, store)
        self.browser = app

    async def _read(self) -> dict[str, Any]:
        if self.browser.session is None:
            return {"url": "about:blank", "cookie": "", "storage": "{}"}
        v = await self.browser.evaluate(_READ)
        return v if isinstance(v, dict) else {"url": str(v), "cookie": "", "storage": "{}"}

    async def fingerprint(self) -> str:
        return sha256_text(json.dumps(await self._read(), sort_keys=True))

    async def snapshot(self) -> StateRef:
        page = await self._read()
        text = json.dumps(page, sort_keys=True)
        key = self.store.put_bytes(text.encode())
        return StateRef(app=self.app, adapter=self.kind, fingerprint=sha256_text(text), blob=key, meta={"url": page["url"]})

    async def restore(self, ref: StateRef) -> None:
        page = json.loads(self.store.get_bytes(ref.blob or ""))
        if self.browser.session is None:
            return
        if page["url"] and page["url"] != "about:blank":
            await self.browser.call_unlisted("browser_navigate", {"url": page["url"]})
        storage = json.loads(page.get("storage") or "{}")
        cookies = [c.strip() for c in (page.get("cookie") or "").split(";") if c.strip()]
        js = ("() => { localStorage.clear(); const s = " + json.dumps(storage) + "; for (const k in s) localStorage.setItem(k, s[k]); "
              + "".join(f"document.cookie = {json.dumps(c + '; path=/')}; " for c in cookies) + "return 'ok'; }")
        await self.browser.evaluate(js)
        if page["url"] and page["url"] != "about:blank":
            await self.browser.call_unlisted("browser_navigate", {"url": page["url"]})
        now = await self._read()
        if sha256_text(json.dumps(now, sort_keys=True)) != ref.fingerprint:
            raise SnapshotMismatch(f"{self.app}: browser state differs after restore ({now.get('url')} vs {page.get('url')})")

    async def diff(self, a: StateRef, b: StateRef) -> StateDiff:
        return StateDiff(app=self.app, adapter=self.kind, before=a.fingerprint, after=b.fingerprint,
                         detail={"url_before": a.meta.get("url"), "url_after": b.meta.get("url")})

    async def q_page(self) -> Any:
        return await self._read()
