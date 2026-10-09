# Origin: Xiaomi MiMo v2.6 RL release — general-agent sidecar runtime

- Source: `RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss/general/envs/<any env>/` (identical in every env)
  - `mcp_http.py`  md5 69484c5fbcb935c1d809887233af748b
  - `mcp_bridge.py` md5 98d327a918248a0297f8283061917e14
- Upstream: HF `XiaomiMiMo/MiMo-V2.6-RL-OSS` (release README: `license: apache-2.0`); driven upstream by
  `xiaomimimo-verl/recipes/general` (DatasetEnvActor) inside a Kubernetes sidecar pod.
- License: Apache-2.0 (see `../../../../LICENSES/Apache-2.0.txt`).

## Changes (every changed line is tagged `# [crucible]`)
`mcp_http.py`
1. `serve(..., host="127.0.0.1")` and a `--host` CLI flag. Upstream bound `0.0.0.0` because it only ran in a
   pod's sidecar container; we also run it as a plain local process, where binding all interfaces is unsafe.

`mcp_bridge.py`: unchanged.

Not ported: `sidecar_entrypoint.py`. Its readiness probe parses `/proc/net/tcp` (Linux-only), its `/dump`
and `/verify` endpoints are advertised but not implemented, and `--start-and-detach` loses the health
thread. `crucible.world.packaging.sidecar` replaces it (socket readiness, supervision, state via
State Adapters, verification via M7).
