"""Ground-truth lanes of the task factory (05 §3). A lane turns a machine draft into a gradeable accept-contract task
(``driver.LANE_TASK``) and supplies the lane-specific state handlers and stage adapters.

- ``executable``: real fix + test commits mined from git repositories, graded by the commit's own tests in a Docker CPU
  sandbox (ObservationRecord type ``real_pr_tests``).
"""
