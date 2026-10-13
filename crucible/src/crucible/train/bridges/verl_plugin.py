"""The ``verl.plugins`` entry point (``pyproject.toml``: ``crucible = "crucible.train.bridges.verl_plugin"``).

verl imports every entry point of that group on ``import verl`` (verl 5eb4826 ``verl/__init__.py:55-75``, policy
``VERL_USE_EXTERNAL_PLUGINS``, default ``auto``), so this module runs in the driver and in every Ray actor that imports
verl: the ``TaskRunnerV1`` that computes advantages (``verl/trainer/main_ppo.py:102-166``) and the agent loop workers.
Importing it registers, in that process:

- the advantage estimator ``crucible_delta`` (``verl/trainer/ppo/core_algos.py:116-133``; select it with
  ``algorithm.adv_estimator=crucible_delta``);
- the agent loop ``crucible_agent`` (``verl/experimental/agent_loop/agent_loop.py:484-495``; select it per sample with the
  dataset's ``agent_name`` or with ``actor_rollout_ref.rollout.agent.default_agent_loop=crucible_agent``).

verl swallows a plugin's import error at debug level (``verl/__init__.py:72-75``); ``register()`` raises instead, so a
caller can check. The agent loop can also be named without the plugin through ``agent_loop_config_path`` (``AGENT_LOOP_YAML``,
read at ``agent_loop.py:551-557``): hydra imports the ``_target_`` itself.
"""

from __future__ import annotations

AGENT_LOOP_YAML = """- name: crucible_agent
  _target_: crucible.train.bridges.verl_agent_loop.CrucibleAgentLoop
  # rollout: my_pkg.my_mod:make_rollout   # factory(agent_loop) -> world_rollout(...) callable (or CRUCIBLE_VERL_ROLLOUT)
  # turn_weight: 0.1                      # M6 turn credit weight (0: outcome only)
"""


def register() -> None:
    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY, register_adv_est

    from crucible.train.bridges.verl_bridge import AGENT_LOOP, ESTIMATOR, crucible_delta

    if ADV_ESTIMATOR_REGISTRY.get(ESTIMATOR) is not crucible_delta:
        ADV_ESTIMATOR_REGISTRY.pop(ESTIMATOR, None)  # a reloaded bridge module replaces its own older function
        register_adv_est(ESTIMATOR)(crucible_delta)
    from verl.experimental.agent_loop.agent_loop import register as register_agent_loop

    from crucible.train.bridges.verl_agent_loop import CrucibleAgentLoop

    register_agent_loop(AGENT_LOOP)(CrucibleAgentLoop)  # {"_target_": its fqdn}: hydra instantiates it per session


register()
