"""M1 item 11: the desktop App (``world/apps/desktop.py``): a policy uses a Linux desktop with mouse and keyboard.

A Tk window (no window manager, so it sits at 0,0) has a text field, a Save button and an Approve button. The policy
sees screenshots (scaled to ``max_width``), types into the field and clicks the buttons in screenshot coordinates; the
grader reads the files the app wrote through ``inspect``. Checked: coordinate scaling, every action returning a
screenshot, invalid actions as policy errors, the reprovision reset tier (a fresh desktop each episode, mid-episode
restore refused) and the replay tier (a new desktop rebuilt by re-issuing actions with screenshot digests checked).
Needs a local Docker daemon (the image builds once).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.state.base import SnapshotMismatch
from crucible.world.world import World

pytestmark = pytest.mark.docker

APP = '''
import tkinter as tk
root = tk.Tk()
root.geometry("400x300+0+0")
entry = tk.Entry(root)
entry.place(x=20, y=20, width=200, height=30)
def save():
    open("/root/saved.txt", "w").write(entry.get())
def approve():
    open("/root/approved.txt", "w").write("yes")
tk.Button(root, text="Save", command=save).place(x=240, y=20, width=80, height=30)
tk.Button(root, text="Approve", command=approve).place(x=20, y=100, width=120, height=50)
root.mainloop()
'''


def _docker_ok() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(["docker", "info"], capture_output=True, timeout=30, check=False).returncode == 0


@pytest.fixture
def pack(tmp_path):
    if not _docker_ok():
        pytest.skip("no local Docker daemon")
    (tmp_path / "app.py").write_text(APP)
    return tmp_path


def _spec(reset: str = "reprovision", **extra) -> WorldSpec:
    return WorldSpec.model_validate({"world": "t/desktop", "apps": [{"id": "desk", "template": "common/desktop", "state": {"adapter": "json_doc", "params": {"path": "desk.json"}},
                                                                      "params": {"screen": "1024x768", "max_width": 512, "files": {"/root/app.py": "app.py"},
                                                                                 "setup": ["python3 /root/app.py"], "settle_s": 1.5, "reset": reset}, **extra}],
                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["desk.*"]}]})


async def _world(pack, tmp, reset="reprovision", **extra):
    w = World(_spec(reset, **extra), store=Store(tmp / "s"), workdir=tmp / "w", pack_root=pack)
    await w.start()
    await w.reset()
    return w


async def _act(w, i, **a):
    return (await w.step("agent", ToolCall(tool_call_id=f"c{i}", function_name="desk.computer", arguments=a))).observation.tool_results[0]


# Screen coordinates halve in the 512-wide screenshot: the Approve button's centre (80, 125) is (40, 62) for the model.
async def test_mouse_keyboard_and_screenshots(pack, tmp_path):
    w = await _world(pack, tmp_path)
    try:
        r = await _act(w, 1, action="screenshot")
        assert r.ok and r.images and "512x384" in r.content
        assert r.images[0].data_b64 and r.images[0].ref  # pixels on the live observation, a store ref in the trace
        await _act(w, 2, action="click", x=60, y=17)  # into the text field (120, 35)
        await _act(w, 3, action="type", text="INV-1042")
        r = await _act(w, 4, action="click", x=140, y=17)  # Save (280, 35)
        assert r.ok and r.data["screenshot_digest"]
        r = await _act(w, 5, action="click", x=40, y=62)  # Approve
        app = w.apps["desk"]
        assert (await app.inspect("cat /root/saved.txt")).stdout == "INV-1042"
        assert (await app.inspect("cat /root/approved.txt")).stdout == "yes"
        bad = await _act(w, 6, action="click", x=600, y=10)
        assert bad.error == "invalid_action" and bad.fault == "policy" and "off the screen" in bad.content
        bad = await _act(w, 7, action="fly")
        assert bad.error == "invalid_arguments"  # not in the action enum: the World's schema check
        assert len(app.journal["steps"]) == 4  # screenshots and refused actions are not journaled
    finally:
        await w.close()


async def test_reset_tiers(pack, tmp_path):
    w = await _world(pack, tmp_path)
    try:
        await _act(w, 1, action="click", x=40, y=62)
        app = w.apps["desk"]
        assert (await app.inspect("test -f /root/approved.txt")).ok
        with pytest.raises(SnapshotMismatch, match="episode start only"):
            await w.restore(await w.snapshot())
        await w.reset()  # a fresh desktop: the file is gone
        assert not (await w.apps["desk"].inspect("test -f /root/approved.txt")).ok
    finally:
        await w.close()
    w = await _world(pack, tmp_path / "replay", reset="replay")
    try:
        await _act(w, 1, action="click", x=60, y=17)
        await _act(w, 2, action="type", text="hello")
        mid = await w.snapshot()
        await _act(w, 3, action="click", x=140, y=17)
        await w.restore(mid)  # a new container; the click and the typing are replayed, each screen checked
        await _act(w, 4, action="click", x=140, y=17)
        assert (await w.apps["desk"].inspect("cat /root/saved.txt")).stdout == "hello"
    finally:
        await w.close()


async def test_health_check_runs_inside_the_desktop(pack, tmp_path):
    """A health check on a desktop App runs in its container (the Tk window must exist), after start and every reset."""
    w = await _world(pack, tmp_path, healthcheck={"command": "DISPLAY=:99 xdotool search --name tk", "start_period_s": 10, "start_interval_s": 0.3})
    try:
        assert w.health["desk"]["healthy"]
    finally:
        await w.close()


async def test_screenshots_reach_a_vision_policy(pack, tmp_path):
    """The language-model path: LLMToolPolicy(vision=True) gets each action's screenshot as an image part."""
    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
    from crucible.runtime.inject import ReplayLLM

    class Recording(ReplayLLM):
        def __init__(self, actions):
            super().__init__(actions)
            self.seen = []

        async def complete(self, messages, **kw):
            self.seen.append([dict(m) for m in messages])
            return await super().complete(messages, **kw)

    llm = Recording([{"function_name": "desk.computer", "arguments": {"action": "screenshot"}},
                     {"function_name": "desk.computer", "arguments": {"action": "click", "x": 40, "y": 62}}])
    w = await _world(pack, tmp_path)
    try:
        ep = await run_episode(w, {"agent": LLMToolPolicy("agent", llm, vision=True)}, reset=False, limits=EpisodeLimits(max_steps=5))
        assert ep.stop_condition == "agent_done"
        images = [p for m in llm.seen[-1] if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url"]
        assert len(images) == 2 and images[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert (await w.apps["desk"].inspect("cat /root/approved.txt")).stdout == "yes"
    finally:
        await w.close()
