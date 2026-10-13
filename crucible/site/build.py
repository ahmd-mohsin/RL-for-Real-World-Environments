"""Wrap page.html (the page body, as published) into a standalone index.html for local preview or static hosting.

    python site/build.py            # writes site/index.html
    python -m http.server -d site   # then open http://localhost:8000

The page's leading ``<title>``, ``<meta>`` and ``<link>`` lines belong in ``<head>`` (crawlers and link previews read the
description only there), so ``render`` lifts them out of the body.
"""

from pathlib import Path

HERE = Path(__file__).parent
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
"""
HEAD_TAGS = ("<title", "<meta", "<link")


def render(page: str | None = None) -> str:
    """index.html for ``page`` (default: page.html): its head lines in ``<head>``, the rest in ``<body>``."""
    page = (HERE / "page.html").read_text() if page is None else page
    lines = page.splitlines(keepends=True)
    head = []
    while lines and lines[0].lstrip().startswith(HEAD_TAGS):
        head.append(lines.pop(0))
    return HEAD + "".join(head) + "</head>\n<body>\n" + "".join(lines) + "\n</body>\n</html>\n"


def main() -> None:
    (HERE / "index.html").write_text(render())
    print(f"wrote {HERE / 'index.html'}")


if __name__ == "__main__":
    main()
