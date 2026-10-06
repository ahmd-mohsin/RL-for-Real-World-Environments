# Sandbox image for task03_mdlite_pallas. Build from the task directory:
#   docker build -f environment/Dockerfile.base -t task03-base:v1 environment
#   docker build -f environment/Dockerfile -t <registry>/task03-mdlite-pallas:v1 .
# Run with a GPU: docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ...  (or --gpus all)
# The image holds only the original repositories; the hidden tests are mounted at /tests by the verifier.
ARG BASE=task03-base:v1
FROM ${BASE}

# pristine copies for the verifier
COPY repos/ /opt/originals/
# agent workspace: one git repo per project, tagged `base`
COPY repos/ /workspace/
RUN set -e; for r in mdlite mdbuild mdkit; do \
      cd /workspace/$r && git init -q && git add -A \
      && git -c user.name=mdlite -c user.email=mdlite@localhost commit -qm "$r (original)" && git tag base; \
    done \
 && git config --global user.name agent && git config --global user.email agent@localhost \
 && for r in mdlite mdbuild mdkit; do git config --global --add safe.directory /workspace/$r; done \
 && cd /workspace/mdlite && MAX_JOBS=16 pip install --no-build-isolation -q -e . \
 && pip install --no-build-isolation -q -e /workspace/mdbuild -e /workspace/mdkit \
 && chmod -R a+rwX /workspace
COPY environment/sandbox-start.sh /usr/local/bin/sandbox-start.sh
RUN chmod +x /usr/local/bin/sandbox-start.sh
WORKDIR /workspace
CMD ["/usr/local/bin/sandbox-start.sh"]
