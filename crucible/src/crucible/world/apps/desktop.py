"""Desktop App: a Linux desktop the policy uses with mouse and keyboard (M1 item 11; the action vocabulary of Harbor's
``computer-1`` (``agents/computer_1/runtime.py``, Apache-2.0), Anthropic's ``computer`` tool as FineEnvs' desktop
environment mirrors it, and OSWorld; xdotool on an X virtual display as Harbor and OpenEnv's helium environment run it).

The App runs its own container (``--network none``): Xvfb on ``:99`` at ``screen`` size, xdotool for input, ImageMagick
for screenshots, and ``setup`` commands that start the task's applications. ``files`` copies host files in first.

**One tool, ``computer``**, with ``action`` in click, double_click, right_click, middle_click, drag, scroll, move,
type, key, hold_key, wait, screenshot. Coordinates are in the screenshot the model sees: screenshots wider than
``max_width`` are scaled down and coordinates are scaled back to the screen (Harbor's coordinate scaling).
``modifier`` holds shift/ctrl/alt/super during a click or scroll. Every action returns a fresh screenshot as an image
(``ToolResult.images``) and a one-line description; the first observation's screenshot is the ``screenshot`` action.

**State and resets.** The journal (``json_doc`` adapter at ``path``) records every action with the digest of the
screenshot after it. A desktop's processes and memory cannot be snapshotted generically, so two reset tiers:

- ``reset: reprovision`` (default): every fresh episode starts a new container from the image and reruns ``setup``;
  a mid-episode restore is refused (branching and resume are off for this App);
- ``reset: replay``: a restore also starts a new container, then re-issues the journal's actions and checks every
  screenshot digest, so a desktop that does not reproduce its screens fails loudly instead of diverging.

Graders read the desktop through ``inspect(cmd)`` (a root shell inside the container) and ``screenshot()``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import posixpath
import shlex
import uuid
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.world.apps.base import App, ToolError
from crucible.world.exec import DockerExec, _run_subprocess
from crucible.world.state.base import SnapshotMismatch

DOCKERFILE = """FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends xvfb xdotool imagemagick x11-utils python3 python3-tk fonts-dejavu-core \\
    && rm -rf /var/lib/apt/lists/*
"""
IMAGE = "crucible-desktop:" + hashlib.sha256(DOCKERFILE.encode()).hexdigest()[:12]
DISPLAY = ":99"
ACTIONS = ("click", "double_click", "right_click", "middle_click", "drag", "scroll", "move", "type", "key", "hold_key", "wait", "screenshot")
MODIFIERS = {"shift": "shift", "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "super": "super"}
_XY = {"type": "integer", "minimum": 0}


async def ensure_image(docker_host: str | None = None, image: str = IMAGE) -> str:
    env = {"DOCKER_HOST": docker_host} if docker_host else None
    if (await _run_subprocess(["docker", "image", "inspect", image], cwd=None, env=env, timeout_s=60, stdin=None)).ok:
        return image
    out = await _run_subprocess(["docker", "build", "-t", image, "-"], cwd=None, env=env, timeout_s=1800, stdin=DOCKERFILE.encode())
    if not out.ok:
        raise RuntimeError(f"building {image} failed: {(out.stderr or out.stdout)[-800:]}")
    return image


class DesktopApp(App):
    template = "common/desktop"
    PATH_PARAMS = ("path",)

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        p = spec.params
        self.docker_host = p.get("docker_host")
        self.width, self.height = (int(x) for x in str(p.get("screen", "1024x768")).lower().split("x")[:2])
        self.max_width = int(p.get("max_width", 1280))
        self.scale = min(1.0, self.max_width / self.width)
        self.container: str | None = None
        self.dexec: DockerExec | None = None
        self.journal: dict[str, Any] = {}

    @property
    def health_exec(self) -> DockerExec | None:  # health checks run inside the desktop, not on the host
        return self.dexec

    def _env(self) -> dict[str, str] | None:
        return {"DOCKER_HOST": self.docker_host} if self.docker_host else None

    # ---- the container ---------------------------------------------------------------------------------------------
    async def _boot(self) -> None:
        await self._shutdown()
        image = self.spec.params.get("image") or await ensure_image(self.docker_host)
        name = f"crucible-desktop-{uuid.uuid4().hex[:10]}"
        out = await _run_subprocess(["docker", "run", "-d", "--rm", "--init", "--network", "none", "--name", name, "--entrypoint", "", image, "sleep", "infinity"],
                                    cwd=None, env=self._env(), timeout_s=300, stdin=None)
        if not out.ok:
            raise RuntimeError(f"desktop container failed to start: {(out.stderr or out.stdout)[-500:]}")
        self.container, self.dexec = name, DockerExec(name, docker_host=self.docker_host)
        await self.dexec.run(f"nohup Xvfb {DISPLAY} -screen 0 {self.width}x{self.height}x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &", timeout_s=30)
        for _ in range(100):
            if (await self.dexec.run(f"xdpyinfo -display {DISPLAY} >/dev/null 2>&1", timeout_s=10)).ok:
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("the virtual display did not come up")
        for dest, src in (self.spec.params.get("files") or {}).items():
            await self.dexec.run(f"mkdir -p {shlex.quote(posixpath.dirname(dest))}")
            await self.dexec.push_file(self.ctx.host_seed(src), dest)
        for cmd in self.spec.params.get("setup") or []:
            await self.dexec.run(f"cd /root && DISPLAY={DISPLAY} nohup sh -c {shlex.quote(cmd)} >>/tmp/setup.log 2>&1 &", timeout_s=60)
        await asyncio.sleep(float(self.spec.params.get("settle_s", 1.0)))

    async def _shutdown(self) -> None:
        if self.container:
            await _run_subprocess(["docker", "rm", "-f", self.container], cwd=None, env=self._env(), timeout_s=120, stdin=None)
            self.container, self.dexec = None, None

    async def start(self) -> None:
        await self._boot()
        self.journal = {"reset": self.spec.params.get("reset", "reprovision"), "start_digest": self._digest(await self.screenshot()), "steps": []}
        await self._write()

    async def stop(self) -> None:
        await self._shutdown()

    async def _write(self) -> None:
        path = self.state_params["path"]
        await self.ctx.exec.run(f"mkdir -p {shlex.quote(posixpath.dirname(path))}")
        await self.ctx.exec.write_text(path, json.dumps(self.journal, sort_keys=True, separators=(",", ":")))

    async def after_restore(self) -> None:
        doc = json.loads(await self.ctx.exec.read_text(self.state_params["path"]))
        tier = self.spec.params.get("reset", "reprovision")
        if doc["steps"] and tier != "replay":
            raise SnapshotMismatch(f"{self.id}: a desktop restores the episode start only (reset: reprovision); use reset: replay to rebuild mid-episode states")
        await self._boot()
        if not await self._shows(doc["start_digest"]):
            raise SnapshotMismatch(f"{self.id}: the fresh desktop does not show the episode's first screen")
        for i, s in enumerate(doc["steps"]):
            await self._do(s["args"])
            if not await self._shows(s["digest"]):
                raise SnapshotMismatch(f"{self.id}: replayed action {i} ({s['args'].get('action')}) gave a different screen")
        self.journal = doc

    async def _shows(self, digest: str, *, tries: int = 8, every_s: float = 0.15) -> bool:
        """Whether the screen shows ``digest`` within about a second: a blinking caret has two screens, both legitimate."""
        for _ in range(tries):
            if self._digest(await self.screenshot()) == digest:
                return True
            await asyncio.sleep(every_s)
        return False

    # ---- reading the desktop -----------------------------------------------------------------------------------------
    async def screenshot(self) -> bytes:
        assert self.dexec is not None
        resize = f" -resize {int(self.width * self.scale)}x{int(self.height * self.scale)}!" if self.scale < 1 else ""
        out = await self.dexec.run(f"import -display {DISPLAY} -window root{resize} png:- | base64 -w0", timeout_s=60)
        if not out.ok or not out.stdout.strip():
            raise RuntimeError(f"screenshot failed: {out.stderr[-300:]}")
        return base64.b64decode(out.stdout.strip())

    @staticmethod
    def _digest(png: bytes) -> str:
        return hashlib.sha256(png).hexdigest()[:24]

    async def inspect(self, cmd: str, timeout_s: float = 60) -> Any:
        """A root shell inside the desktop, for graders (never a policy tool)."""
        assert self.dexec is not None
        return await self.dexec.run(cmd, timeout_s=timeout_s)

    # ---- acting ------------------------------------------------------------------------------------------------------
    def _native(self, v: Any, name: str) -> int:
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ToolError(f"{name} must be a non-negative integer")
        n = round(v / self.scale)
        limit = self.width if name in ("x", "end_x") else self.height
        if n >= limit:
            raise ToolError(f"{name}={v} is off the screen ({int(limit * self.scale)} pixels wide/high in the screenshot)")
        return n

    async def _do(self, a: dict[str, Any]) -> str:
        assert self.dexec is not None
        act = a.get("action")
        if act not in ACTIONS:
            raise ToolError(f"unknown action {act!r}; one of {', '.join(ACTIONS)}")
        mods = []
        if a.get("modifier"):
            if a["modifier"] not in MODIFIERS:
                raise ToolError(f"modifier must be one of {sorted(MODIFIERS)}")
            mods = [MODIFIERS[a["modifier"]]]
        down = "".join(f" keydown {m}" for m in mods)
        up = "".join(f" keyup {m}" for m in mods)
        if act in ("click", "double_click", "right_click", "middle_click", "move", "scroll", "drag"):
            x, y = self._native(a.get("x"), "x"), self._native(a.get("y"), "y")
        if act in ("click", "double_click", "right_click", "middle_click"):
            button = {"click": 1, "double_click": 1, "right_click": 3, "middle_click": 2}[act]
            rep = " --repeat 2 --delay 80" if act == "double_click" else ""
            cmd, what = f"xdotool mousemove {x} {y}{down} click{rep} {button}{up}", f"{act} at ({a['x']}, {a['y']})"
        elif act == "move":
            cmd, what = f"xdotool mousemove {x} {y}", f"moved to ({a['x']}, {a['y']})"
        elif act == "drag":
            ex, ey = self._native(a.get("end_x"), "end_x"), self._native(a.get("end_y"), "end_y")
            cmd, what = f"xdotool mousemove {x} {y} mousedown 1 mousemove {ex} {ey} mouseup 1", f"dragged ({a['x']}, {a['y']}) -> ({a['end_x']}, {a['end_y']})"
        elif act == "scroll":
            dy = int(a.get("scroll_y") or 0)
            dx = int(a.get("scroll_x") or 0)
            if not dx and not dy:
                raise ToolError("scroll needs scroll_y or scroll_x")
            clicks = ([f"click --repeat {abs(dy)} {5 if dy > 0 else 4}"] if dy else []) + ([f"click --repeat {abs(dx)} {7 if dx > 0 else 6}"] if dx else [])
            cmd, what = f"xdotool mousemove {x} {y}{down} {' '.join(clicks)}{up}", f"scrolled ({dx}, {dy}) at ({a['x']}, {a['y']})"
        elif act == "type":
            text = a.get("text")
            if not isinstance(text, str) or not text:
                raise ToolError("type needs text")
            cmd, what = f"xdotool type --delay 12 -- {shlex.quote(text)}", f"typed {len(text)} characters"
        elif act == "key":
            keys = a.get("keys") or ([a["text"]] if a.get("text") else [])
            if not keys or not all(isinstance(k, str) and k.replace("+", "").replace("_", "").isalnum() for k in keys):
                raise ToolError("key needs keys like ['ctrl+s'] or ['Return']")
            cmd, what = "xdotool key " + " ".join(shlex.quote(k) for k in keys), f"pressed {' '.join(keys)}"
        elif act == "hold_key":
            key, dur = a.get("keys", [None])[0] if a.get("keys") else a.get("text"), float(a.get("duration") or 0.5)
            if not key or not 0 < dur <= 10:
                raise ToolError("hold_key needs one key and a duration in (0, 10] s")
            cmd, what = f"xdotool keydown {shlex.quote(key)} sleep {dur} keyup {shlex.quote(key)}", f"held {key} for {dur} s"
        elif act == "wait":
            dur = float(a.get("duration") or 1.0)
            if not 0 < dur <= 30:
                raise ToolError("wait needs a duration in (0, 30] s")
            await asyncio.sleep(dur)
            return f"waited {dur} s"
        else:  # screenshot
            return "screenshot"
        out = await self.dexec.run(f"DISPLAY={DISPLAY} {cmd}", timeout_s=60)
        if not out.ok:
            raise ToolError(f"{act} failed: {(out.stderr or out.stdout)[-300:]}")
        await asyncio.sleep(float(self.spec.params.get("wait_after_action_s", 0.3)))
        return what

    async def list_tools(self) -> list[ToolSpec]:
        w, h = int(self.width * self.scale), int(self.height * self.scale)
        props = {"action": {"type": "string", "enum": list(ACTIONS)}, "x": _XY, "y": _XY, "end_x": _XY, "end_y": _XY,
                 "text": {"type": "string"}, "keys": {"type": "array", "items": {"type": "string"}},
                 "scroll_x": {"type": "integer"}, "scroll_y": {"type": "integer"},
                 "modifier": {"type": "string", "enum": sorted(MODIFIERS)}, "duration": {"type": "number"}}
        return [ToolSpec(app=self.id, name="computer",
                         description=f"Use the desktop with mouse and keyboard. Coordinates are pixels in the {w}x{h} screenshot. Every action returns a new screenshot.",
                         input_schema={"type": "object", "properties": props, "required": ["action"], "additionalProperties": False})]

    async def call(self, name: str, args: dict[str, Any], *, actor: str, call_id: str) -> ToolResult:
        q = f"{self.id}.{name}"
        if name != "computer":
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content=f"Unknown tool: {name}", error="unknown_tool")
        try:
            what = await self._do(args)
        except ToolError as e:
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content=str(e), error="invalid_action", fault="policy")
        png = await self.screenshot()
        if args.get("action") != "screenshot":
            self.journal["steps"].append({"args": dict(args), "digest": self._digest(png)})
            await self._write()
        w, h = int(self.width * self.scale), int(self.height * self.scale)
        return ToolResult(tool_call_id=call_id, function_name=q, ok=True, content=f"{what}; screenshot {w}x{h} attached", images=[self.image(png)],
                          data={"action": args.get("action"), "screenshot_digest": self._digest(png)})
