# Part 1: Image. A frozen filesystem holding the repo at the task's starting state with all dependencies installed.
#
# Layers, cheapest-to-change last so rebuilds reuse the cache:
#   base OS + Python  ->  system tools  ->  repo at base commit + deps  ->  task hygiene
#
# Build:  docker build -t rlsandbox/python-slugify:modern-hard-cut sandbox/image
# Training clusters are x86: add --platform linux/amd64 when building images to ship.

FROM python:3.12-slim

# Tools the agent's bash tool relies on. Keep this list explicit: it is part of the environment.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates ripgrep procps \
 && rm -rf /var/lib/apt/lists/*

ARG REPO_URL=https://github.com/un33k/python-slugify.git
ARG BASE_COMMIT=c442cd4cb61763c85b078d6ea83b5959c3ff364a

WORKDIR /testbed

# Repo at the commit just before the upstream fix, plus runtime and test dependencies.
RUN git clone --quiet "$REPO_URL" . \
 && git checkout --quiet "$BASE_COMMIT" \
 && pip install --no-cache-dir -e . "pytest>=8,<10"

# Prove the environment works before freezing it: the existing suite must pass at the base commit.
RUN python -m pytest -q tests

# Hygiene: nothing the agent can read may reveal the answer.
#  - Truncate git history to a single "base" commit. With the real history, `git log --all`
#    would show the upstream fix (8f9a550) and every later commit.
#  - Drop packaging residue that the editable install or the test run left behind.
RUN rm -rf .git .pytest_cache build *.egg-info \
 && find . -name __pycache__ -type d -prune -exec rm -rf {} + \
 && git init --quiet \
 && git config --global user.email "sandbox@local" \
 && git config --global user.name "sandbox" \
 && git config --global --add safe.directory /testbed \
 && git add -A \
 && git commit --quiet -m "base" \
 && git tag base

ENV PYTHONDONTWRITEBYTECODE=1
CMD ["sleep", "infinity"]
