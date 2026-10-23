#!/bin/sh
# run the crucible venv python against the worktree sources
cd /Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible || exit 1
export PYTHONPATH=/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/.claude/worktrees/agent-a0405eb80c561ac86/crucible/src
exec /Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/crucible/.venv/bin/python "$@"
