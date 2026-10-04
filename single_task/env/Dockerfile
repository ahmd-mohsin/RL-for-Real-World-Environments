# Image for the single-task sandbox: the Tessera repository at /testbed,
# built and tested once so a broken environment can never ship.
#
# Build from the single_task/ directory:
#   docker build -t rlsandbox/tessera:base -f env/Dockerfile .
FROM ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake ninja-build git ca-certificates ripgrep python3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /testbed
COPY repo/ /testbed/

# The environment must work before it is frozen: configure, build, run the suite.
RUN cmake -S . -B build -G Ninja \
 && cmake --build build \
 && ctest --test-dir build --output-on-failure

# Hygiene: keep the configured build directory (agents get a warm build), but
# start version control from a single "base" commit so the history reveals
# nothing.
RUN git init --quiet \
 && git config --global user.email "sandbox@local" \
 && git config --global user.name "sandbox" \
 && git config --global --add safe.directory /testbed \
 && git add -A \
 && git commit --quiet -m "base" \
 && git tag base

ENV TSR_EMU_WORKERS=4
CMD ["sleep", "infinity"]
