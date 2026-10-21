"""A plan whose single check sleeps (verifier-timeout injection)."""

import time

from crucible.verify.runtime.plan import Check, Gate, Plan


def plan(ctx):
    return Plan("redteam/sleepy", [Gate("g", "gate")], [Check("g", "g", lambda c: time.sleep(float(c.params.get("sleep", 120))) or True)])
