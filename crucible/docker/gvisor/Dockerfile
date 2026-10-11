# gVisor toolbox for the runsc-in-container sandbox tier (m13 S5): runsc and its sidecars at /opt/gvisor, pinned and
# checksum-verified. Mounted read-only into a World's privileged outer container with --volumes-from.
FROM debian:bookworm-slim
ARG GVISOR=20261005.0
RUN apt-get update -qq && apt-get install -y -qq curl ca-certificates bzip2 >/dev/null && \
    ARCH=$(uname -m) && U=https://storage.googleapis.com/gvisor/releases/release/${GVISOR}/${ARCH} && \
    curl -fsSL -o /tmp/g.tar.bz2 $U/gvisor.tar.bz2 && curl -fsSL -o /tmp/g.sha512 $U/gvisor.tar.bz2.sha512 && \
    (cd /tmp && echo "$(cut -d' ' -f1 g.sha512)  g.tar.bz2" | sha512sum -c -) && \
    mkdir -p /opt/gvisor && tar -xjf /tmp/g.tar.bz2 -C /opt/gvisor runsc gvisor-bin && \
    rm -f /tmp/g.tar.bz2 /tmp/g.sha512 && rm -rf /var/lib/apt/lists/* && /opt/gvisor/runsc --version
VOLUME /opt/gvisor
