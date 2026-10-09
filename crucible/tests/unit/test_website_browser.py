"""Website + browser Apps: a clerk works a vendor portal through a real headless browser (Playwright MCP);
the portal's database is the state verifiers read; browser navigation state snapshots and restores."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from crucible_pack_toy import PACK_ROOT

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World

pytestmark = [pytest.mark.slow]
if shutil.which("npx") is None:
    pytest.skip("needs node/npx for @playwright/mcp", allow_module_level=True)


def spec() -> WorldSpec:
    return WorldSpec.model_validate({
        "world": "test/vendor-portal",
        "apps": [
            {"id": "portal", "template": "common/website", "state": {"adapter": "sql", "params": {"path": "portal/portal.db"}},
             "params": {"app": "data/portal/site.py:create_app"}},
            {"id": "web", "template": "common/browser", "state": {"adapter": "browser"}, "params": {"start_url": "{app:portal}/"}},
        ],
        "actors": [{"id": "clerk", "kind": "policy", "tools": ["web.*"]}],
    })


def _ref(snapshot: str, label: str) -> str:
    """Find the element ref for a labelled control in a Playwright accessibility snapshot."""
    for line in snapshot.splitlines():
        if label in line and "[ref=" in line:
            return re.search(r"\[ref=([^\]]+)\]", line).group(1)
    raise AssertionError(f"{label!r} not in snapshot:\n{snapshot[:3000]}")


async def test_clerk_disputes_duplicate_invoice_in_browser(tmp_path: Path):
    w = World(spec(), store=Store(tmp_path / "s"), workdir=tmp_path / "w", pack_root=PACK_ROOT)
    await w.start()
    try:
        obs = (await w.reset())["clerk"]
        names = {t.name for t in obs.tools}
        assert "browser_click" in names and "browser_evaluate" not in names  # arbitrary JS is withheld from the policy
        portal = w.apps["portal"].url
        r = await w.step("clerk", ToolCall(tool_call_id="1", function_name="web.browser_navigate", arguments={"url": f"{portal}/invoices/INV-1003"}))
        assert r.observation.tool_results[0].ok, r.observation.tool_results[0].content
        snap = (await w.step("clerk", ToolCall(tool_call_id="2", function_name="web.browser_snapshot", arguments={}))).observation.tool_results[0].content
        assert "Invoice INV-1003" in snap
        before = await w.snapshot()
        for i, (fn, args) in enumerate([
            ("web.browser_select_option", {"element": "Action", "target": _ref(snap, "combobox"), "values": ["dispute"]}),
            ("web.browser_type", {"element": "Note", "target": _ref(snap, "textbox"), "text": "Duplicate of INV-1001 (same vendor, same amount)."}),
        ]):
            r = await w.step("clerk", ToolCall(tool_call_id=f"f{i}", function_name=fn, arguments=args))
            assert r.observation.tool_results[0].ok, r.observation.tool_results[0].content
        r = await w.step("clerk", ToolCall(tool_call_id="5", function_name="web.browser_click", arguments={"element": "Submit", "target": _ref(snap, 'button "Submit"')}))
        assert r.observation.tool_results[0].ok, r.observation.tool_results[0].content
        v = await w.verifier_view().query("portal", {"op": "sql", "args": {"sql": "SELECT id, status, note FROM invoices ORDER BY id"}})
        assert v.value["rows"][2] == ["INV-1003", "disputed", "Duplicate of INV-1001 (same vendor, same amount)."]
        page = await w.verifier_view().query("web", {"op": "page"})
        assert page.value["url"].endswith("/invoices/INV-1003?saved=1")
        # world restore: the portal DB rewinds and the browser goes back to the pre-submit page
        await w.restore(before)
        v = await w.verifier_view().query("portal", {"op": "scalar", "args": {"sql": "SELECT status FROM invoices WHERE id = 'INV-1003'"}})
        assert v.value == "pending"
        assert (await w.verifier_view().query("web", {"op": "page"})).value["url"].endswith("/invoices/INV-1003")
        # navigation is limited to the world's own sites (best effort in-browser; backends enforce networks)
        r = await w.step("clerk", ToolCall(tool_call_id="6", function_name="web.browser_navigate", arguments={"url": "https://github.com/"}))
        assert "github" not in (await w.verifier_view().query("web", {"op": "page"})).value["url"]
    finally:
        await w.close()
