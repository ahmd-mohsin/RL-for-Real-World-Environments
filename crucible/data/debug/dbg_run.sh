#!/bin/sh
# debug: run a command inside a debug-lane image on a tree copy: dbg_run.sh <tree> <image> <command...>
tree="$1"; image="$2"; shift 2
exec docker run --rm --name crucible-lh-d-dbg-$$ --network none -v "$tree":/src:ro "$image" sh -c "cp -a /src/. /w/ && cd /w && $*"
