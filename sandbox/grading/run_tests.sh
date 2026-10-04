#!/usr/bin/env bash
# Hidden verifier script. Copied into the container only at grading time, never during the rollout.
# Exit 0 = resolved. Two checks, as in SWE-bench:
#   FAIL_TO_PASS: the new regression tests for this issue now pass
#   PASS_TO_PASS: the existing suite still passes (no regressions)
set -uo pipefail
cd /testbed

echo "=== FAIL_TO_PASS ==="
python -m pytest -q -p no:cacheprovider \
  "tests/test_release.py::ReleaseRegressionTests::test_modern_hard_cut_does_not_end_in_post_replacement_delimiters" \
  "tests/test_release.py::ReleaseRegressionTests::test_modern_word_boundary_hard_cut_keeps_delimiters_with_the_next_word"
f2p=$?

echo "=== PASS_TO_PASS ==="
python -m pytest -q -p no:cacheprovider tests
p2p=$?

echo "=== RESULT fail_to_pass_rc=$f2p pass_to_pass_rc=$p2p ==="
exit $(( f2p | p2p ))
