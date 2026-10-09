"""Verbatim copy of ``_patch_env_code`` from OpenEnv ``envs/agent_world_model_env/server/scenario_manager.py``."""

import textwrap


def _patch_env_code(full_code: str, db_path: str, host: str, port: int) -> str:
    """
    Patch the generated FastAPI code:
    1. Replace create_engine() call to point to the session-specific DB.
    2. Inject FastApiMCP mount before uvicorn.run().
    3. Fix uvicorn.run() to use the assigned host/port.
    """
    new_lines = [
        "import warnings",
        'warnings.filterwarnings("ignore", category=DeprecationWarning)',
    ]

    sql_path = f"sqlite:///{db_path}"

    for line in full_code.split("\n"):
        if "create_engine(" in line:
            left = line.split("create_engine(")[0]
            line = f"{left}create_engine('{sql_path}', connect_args={{'check_same_thread': False}})"

        if "uvicorn.run(app" in line:
            mcp_inject = textwrap.dedent("""\
                from fastapi_mcp import FastApiMCP
                mcp = FastApiMCP(app)
                mcp.mount_http()
            """)
            for inject_line in mcp_inject.strip().split("\n"):
                new_lines.append(f"    {inject_line}")

            line = f"    uvicorn.run(app, host='{host}', port={port})"

        new_lines.append(line)

    return "\n".join(new_lines)
