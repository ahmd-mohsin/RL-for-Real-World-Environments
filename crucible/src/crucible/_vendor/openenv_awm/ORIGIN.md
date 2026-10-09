# Origin: OpenEnv — AgentWorldModel-1K environment (meta-pytorch/OpenEnv)

- Source: `RL-Environments/06-frameworks-and-env-code/meta-pytorch-openenv/envs/agent_world_model_env/server/`
  at commit 49aa302 (2026-10-02). License: BSD-3-Clause (`LICENSES/BSD-3-Clause-openenv.txt`).
- Files:
  - `data_loader.py` — copied; changed lines tagged `# [crucible]`: downloads only the JSONL files an
    accessor needs (upstream downloads all 7, incl. ~290 MB of verifier code, on first access).
  - `db_manager.py` — copied unchanged (SQLite from AWM DDL + sample INSERTs).
  - `patch.py` — verbatim copy of `_patch_env_code` from `scenario_manager.py` (points the generated
    FastAPI app's SQLAlchemy engine at the session DB, mounts `FastApiMCP`, fixes host/port).
- Data: HF dataset `Snowflake/AgentWorldModel-1K` (generated environments; see its card for terms).
