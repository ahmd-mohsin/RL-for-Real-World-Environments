#!/bin/bash
# Publish the site to https://ahmd-mohsin.github.io/crucible/ (the public/crucible/ folder of the personal site repo).
#   crucible/site/publish.sh "what changed"
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${CRUCIBLE_SITE_REPO:-$HOME/.cache/crucible-site/ahmd-mohsin.github.io}"
MSG="${1:-Update the Crucible website}"

python3 "$HERE/build.py" >/dev/null
if [ -d "$REPO/.git" ]; then git -C "$REPO" pull -q --ff-only origin main
else mkdir -p "$(dirname "$REPO")"; git clone -q https://github.com/ahmd-mohsin/ahmd-mohsin.github.io.git "$REPO"; fi

DEST="$REPO/public/crucible"
rm -rf "$DEST"; mkdir -p "$DEST/assets"
cp "$HERE/index.html" "$DEST/index.html"
cp "$HERE/workflow.html" "$DEST/workflow.html"
cp "$HERE"/assets/*.js "$DEST/assets/"

cd "$REPO"
git add public/crucible
if git diff --cached --quiet; then echo "no changes to publish"; exit 0; fi
git commit -q -m "$MSG"
git push -q origin main
echo "pushed; live in about two minutes at https://ahmd-mohsin.github.io/crucible/"
