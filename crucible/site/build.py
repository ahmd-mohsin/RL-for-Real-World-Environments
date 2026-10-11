"""Wrap page.html (the page body, as published) into a standalone index.html for local preview or static hosting.

    python site/build.py            # writes site/index.html
    python -m http.server -d site   # then open http://localhost:8000
"""

from pathlib import Path

HERE = Path(__file__).parent
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
"""


def main() -> None:
    page = (HERE / "page.html").read_text()
    (HERE / "index.html").write_text(HEAD + page + "\n</body>\n</html>\n")
    print(f"wrote {HERE / 'index.html'}")


if __name__ == "__main__":
    main()
