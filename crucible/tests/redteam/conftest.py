"""The red-team suite's task fixtures are data, not tests: ``grader_task/hidden/test_mylib.py`` is a hidden test of the
seeded grader task (it imports the task's ``mylib``, which exists only inside that task's tree). Collected as a test it
broke ``pytest tests/redteam -m docker`` (and a bare ``pytest``) at collection (smoke 2026-10-29)."""

collect_ignore_glob = ["grader_task/*", "grader_task/*/*", "runtime_task/*"]
