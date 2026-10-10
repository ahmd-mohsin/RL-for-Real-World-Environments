"""Serve one NeMo-Gym resources-server class (runs in the NeMo-Gym Python, ``crucible/.envs/nemo``; imports only nemo_gym).
Pattern: verifiers ``v1/tasksets/nemo_gym/server.py``.

    NEMO_GYM_RESOURCE_SERVER=resources_servers.workplace_assistant.app:WorkbenchResourcesServer \\
        python serve_resource.py PORT_FILE
"""

import os
import socket
import sys
from importlib import import_module
from pathlib import Path


def main() -> None:
    import uvicorn
    from nemo_gym.config_types import BaseServerConfig
    from nemo_gym.server_utils import ServerClient
    from omegaconf import OmegaConf

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    module_name, class_name = os.environ["NEMO_GYM_RESOURCE_SERVER"].split(":", 1)
    cls = getattr(import_module(module_name), class_name)
    config_cls = cls.model_fields["config"].annotation
    name = module_name.rsplit(".", 2)[-2] if "." in module_name else module_name
    server = cls(config=config_cls(name=name, host="127.0.0.1", port=port, entrypoint="app.py"),
                 server_client=ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=11000), global_config_dict=OmegaConf.create({})))
    app = server.setup_webserver()
    server.setup_liveness(app)
    server.setup_exception_middleware(app)
    Path(sys.argv[1]).write_text(str(port), encoding="ascii")
    uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")).run(sockets=[sock])


if __name__ == "__main__":
    main()
