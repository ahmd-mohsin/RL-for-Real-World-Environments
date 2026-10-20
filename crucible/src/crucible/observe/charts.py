"""Self-contained HTML charts for what W&B's native charts cannot draw: bars with confidence intervals (swarm-pipeline/11 §4).

W&B's built-in chart presets have no error bars, so the headline pre/post chart (pass@1 per arm with its task-clustered
CI) and the per-task chart are inline SVG in an HTML panel (``ObsRun.html``): the same file opens offline in a browser.
Pure Python, no plotting dependency. Colors: the categorical slots of the dataviz reference palette in fixed order
(an arm keeps its color in every chart), light and dark steps, values as direct labels, a ``<title>`` tooltip on every
mark; the numbers are also logged as a W&B table (the table view).
"""

from __future__ import annotations

import html as _html
from collections.abc import Mapping, Sequence

# arm -> categorical slot (fixed: color follows the arm, never its rank)
ARM_SLOT = {"selection": 1, "baseline": 2, "control": 3, "trained": 4}
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

_STYLE = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--grid:#e4e3df;%(light)s
 background:var(--surface-1);color:var(--text-primary);font:13px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;padding:12px;margin:0}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;--text-primary:#f0efec;
 --text-secondary:#c3c2b7;--grid:#3a3a37;%(dark)s}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#3a3a37;%(dark)s}
.viz-root h3{font-size:14px;margin:0 0 4px}.viz-root p{margin:0 0 8px;color:var(--text-secondary)}
.viz-root svg text{fill:var(--text-secondary);font-size:11px}.viz-root svg text.v{fill:var(--text-primary)}
.viz-root .lg{display:flex;gap:14px;flex-wrap:wrap;margin:4px 0 8px;color:var(--text-secondary)}
.viz-root .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
"""


def _vars() -> dict[str, str]:
    return {"light": "".join(f"--series-{i + 1}:{c};" for i, c in enumerate(LIGHT)), "dark": "".join(f"--series-{i + 1}:{c};" for i, c in enumerate(DARK))}


def slot_of(name: str, order: Sequence[str]) -> int:
    if name in ARM_SLOT:
        return ARM_SLOT[name]
    return (list(order).index(name) % len(LIGHT)) + 1 if name in order else 8


def _page(title: str, subtitle: str, body: str) -> str:
    e = _html.escape
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{e(title)}</title><style>{_STYLE % _vars()}</style></head>"
            f"<body class='viz-root'><h3>{e(title)}</h3><p>{e(subtitle)}</p>{body}</body></html>")


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.0f}%"


def grouped_bars_ci(groups: Sequence[str], series: Sequence[str], values: Mapping[tuple[str, str], tuple[float | None, float | None, float | None]], *,
                    title: str, subtitle: str = "", ymax: float = 1.0, percent: bool = True, width: int = 760) -> str:
    """Vertical bars per (group, series) with a CI whisker: ``values[(group, series)] = (point, lo, hi)``. One group with
    several series is the headline arm chart; many groups (tasks) with the arms as series is the per-task chart."""
    e = _html.escape
    pad_l, pad_r, pad_t, pad_b, h = 40, 12, 18, 54, 260
    n_s = max(1, len(series))
    gw = (width - pad_l - pad_r) / max(1, len(groups))
    bw = max(4.0, min(36.0, (gw - 10) / n_s - 2))
    y = lambda v: pad_t + h * (1 - max(0.0, min(ymax, v)) / ymax)
    out = [f"<svg viewBox='0 0 {width} {pad_t + h + pad_b}' width='100%' role='img' aria-label='{e(title)}'>"]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        yy = y(t * ymax)
        out.append(f"<line x1='{pad_l}' x2='{width - pad_r}' y1='{yy:.1f}' y2='{yy:.1f}' stroke='var(--grid)' stroke-width='1'/>"
                   f"<text x='{pad_l - 6}' y='{yy + 4:.1f}' text-anchor='end'>{_pct(t * ymax) if percent else f'{t * ymax:.2g}'}</text>")
    for gi, g in enumerate(groups):
        x0 = pad_l + gi * gw + (gw - n_s * (bw + 2)) / 2
        for si, s in enumerate(series):
            pt = values.get((g, s))
            if not pt or pt[0] is None:
                continue
            v, lo, hi = pt
            x = x0 + si * (bw + 2)
            top, base = y(v), y(0)
            hgt = max(0.0, base - top)
            r = min(4.0, bw / 2, hgt)
            col = f"var(--series-{slot_of(s, series)})"
            tip = f"{g} | {s}: {_pct(v) if percent else f'{v:.3g}'}" + (f" (CI {_pct(lo)} to {_pct(hi)})" if lo is not None and hi is not None else "")
            # a bar with 4px rounded data end, square at the baseline
            path = (f"M{x:.1f},{base:.1f} L{x:.1f},{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} L{x + bw - r:.1f},{top:.1f} "
                    f"Q{x + bw:.1f},{top:.1f} {x + bw:.1f},{top + r:.1f} L{x + bw:.1f},{base:.1f} Z")
            out.append(f"<g><title>{e(tip)}</title><path d='{path}' fill='{col}'/>")
            if lo is not None and hi is not None:
                cx = x + bw / 2
                out.append(f"<line x1='{cx:.1f}' x2='{cx:.1f}' y1='{y(hi):.1f}' y2='{y(lo):.1f}' stroke='var(--text-primary)' stroke-width='1.5'/>"
                           f"<line x1='{cx - 4:.1f}' x2='{cx + 4:.1f}' y1='{y(hi):.1f}' y2='{y(hi):.1f}' stroke='var(--text-primary)' stroke-width='1.5'/>"
                           f"<line x1='{cx - 4:.1f}' x2='{cx + 4:.1f}' y1='{y(lo):.1f}' y2='{y(lo):.1f}' stroke='var(--text-primary)' stroke-width='1.5'/>")
            if len(groups) <= 4:  # direct labels only where they fit
                out.append(f"<text class='v' x='{x + bw / 2:.1f}' y='{min(top, y(hi) if hi is not None else top) - 4:.1f}' text-anchor='middle'>"
                           f"{_pct(v) if percent else f'{v:.3g}'}</text>")
            out.append("</g>")
        lab = g if len(g) <= 22 else g[:20] + ".."
        out.append(f"<text x='{pad_l + gi * gw + gw / 2:.1f}' y='{pad_t + h + 16}' text-anchor='middle'><title>{e(g)}</title>{e(lab)}</text>")
    out.append("</svg>")
    legend = "" if len(series) < 2 else "<div class='lg'>" + "".join(
        f"<span><span class='sw' style='background:var(--series-{slot_of(s, series)})'></span>{e(s)}</span>" for s in series) + "</div>"
    return _page(title, subtitle, legend + "".join(out))


def lines(xs: Sequence[float], ys: Mapping[str, Sequence[float | None]], *, title: str, subtitle: str = "", xname: str = "k", width: int = 560) -> str:
    """Lines over ``xs`` (pass@k and pass^k per arm), y in [0, 1]; 2px lines, 8px markers, a tooltip per point."""
    e = _html.escape
    pad_l, pad_r, pad_t, pad_b, h = 40, 12, 14, 36, 220
    names = list(ys)
    if not xs:
        return _page(title, subtitle, "<p>no data</p>")
    x_lo, x_hi = min(xs), max(xs)
    fx = lambda x: pad_l + (width - pad_l - pad_r) * ((x - x_lo) / (x_hi - x_lo) if x_hi > x_lo else 0.5)
    fy = lambda v: pad_t + h * (1 - max(0.0, min(1.0, v)))
    out = [f"<svg viewBox='0 0 {width} {pad_t + h + pad_b}' width='100%' role='img' aria-label='{e(title)}'>"]
    for t in (0, 0.5, 1.0):
        out.append(f"<line x1='{pad_l}' x2='{width - pad_r}' y1='{fy(t):.1f}' y2='{fy(t):.1f}' stroke='var(--grid)'/>"
                   f"<text x='{pad_l - 6}' y='{fy(t) + 4:.1f}' text-anchor='end'>{_pct(t)}</text>")
    for x in xs:
        out.append(f"<text x='{fx(x):.1f}' y='{pad_t + h + 16}' text-anchor='middle'>{e(xname)}={x:g}</text>")
    for n in names:
        col = f"var(--series-{slot_of(n, names)})"
        pts = [(x, v) for x, v in zip(xs, ys[n], strict=False) if v is not None]
        if len(pts) > 1:
            out.append(f"<polyline fill='none' stroke='{col}' stroke-width='2' points='{' '.join(f'{fx(x):.1f},{fy(v):.1f}' for x, v in pts)}'/>")
        for x, v in pts:
            out.append(f"<g><title>{e(n)} {e(xname)}={x:g}: {_pct(v)}</title><circle cx='{fx(x):.1f}' cy='{fy(v):.1f}' r='4' fill='{col}' "
                       f"stroke='var(--surface-1)' stroke-width='2'/></g>")
    out.append("</svg>")
    legend = "" if len(names) < 2 else "<div class='lg'>" + "".join(
        f"<span><span class='sw' style='background:var(--series-{slot_of(n, names)})'></span>{e(n)}</span>" for n in names) + "</div>"
    return _page(title, subtitle, legend + "".join(out))
